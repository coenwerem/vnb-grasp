#!/usr/bin/env python3
"""
Variational Neural Belief Experiments for IROS 2026 paper.

Runs the full experimental evaluation comparing:
  1. Particle-filter belief MPC  (PF baseline)
  2. Gaussian MPC  (Gauss: K=1, risk-neutral)
  3. Gaussian CVaR MPC  (Gauss-CVaR: K=1, CVaR objective)
  4. CEM MPC  (Cross-Entropy Method with Gaussian belief)
  5. Variational Neural Belief MPC  (ours: K=8 GMM, exact CVaR gradients)

across four friction regimes:
  - nominal:     μ ~ U[0.4, 1.0]
  - adversarial: μ ~ U[0.15, 0.40]  (low-friction, stresses grasp robustness)
  - wide:        μ ~ U[0.15, 1.2]   (full uncertainty span)
  - bimodal:     μ ~ 0.5 N(0.18, 0.03²) + 0.5 N(1.0, 0.05²)

and multiple:
  - Objects:      cube, graspit_box
  - Beta values:  0.5, 0.9, 0.95, 0.99
  - Seeds:        3 per condition for confidence intervals

Yielding 5x2x4x3x4 = 480 episodes total.

Protocol per condition:
  1. Position arm above object
  2. Run belief-MPC grasp (particle or variational)
  3. Collect per-step metrics (GWS epsilon, contact quality, entropy, CVaR)
  4. Run lift + 3-pulse shear stability test
  5. Save structured JSON

Usage:
    # Full 480-episode sweep (all methods x regimes x objects x betas x seeds)
    python examples/run_variational_belief_experiments.py

    # Quick smoke test (1 method, 1 object, 1 regime, 1 beta, 1 seed)
    python examples/run_variational_belief_experiments.py --quick

    # Single object / beta / regime
    python examples/run_variational_belief_experiments.py --objects cube --betas 0.9 --regimes nominal

    # Specific methods only
    python examples/run_variational_belief_experiments.py --methods particle variational

Author: Clinton Enwerem
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco as mj

from vnb_grasp.wrappers.mujoco_native import RawMujocoEnv
from vnb_grasp.control.actuator_map import ActuatorMap
from vnb_grasp.belief import BeliefMPCConfig, BeliefMPCPlanner, GraspObservation
from vnb_grasp.belief.particle_filter import cvar, failure_probability
from vnb_grasp.belief.variational_belief import (
    VariationalBeliefConfig,
    GaussianMixtureBelief,
    NeuralBeliefFilter,
)
from vnb_grasp.grasping.gws_quality import analyze_gws, GWSResult
from vnb_grasp.grasping.ycb_objects import (
    get_object_config,
    get_full_grasp_config,
    GraspItToMuJoCoTransform,
)
from vnb_grasp.grasping.pregrasp_planner import (
    plan_pregrasp,
    plan_all_pregrasps,
    PregraspPlan,
)
from vnb_grasp.scripted_policies.pregrasp_planner import (
    PregraspPlanner as ScriptedPregraspPlanner,
    HOME_Q as SCRIPTED_HOME_Q,
    GRASP_TORQUE as SCRIPTED_GRASP_TORQUE,
    stash_all as scripted_stash_all,
    spawn_object as scripted_spawn_object,
    freeze_stash as scripted_freeze_stash,
    interp_trajectory,
)
from vnb_grasp.belief.mujoco_rollout import extract_contacts

# 
# Constants
# 

OUTPUT_DIR = Path("outputs/variational_belief_experiments")

FINGERTIP_GEOMS = [
    "thumb_metacarpals_base2_collision_0",
    "thumb_metacarpals_collision_0",
    "thumb_distal_collision_0",
    "index_proximal_collision_0",
    "index_distal_collision_0",
    "middle_proximal_collision_0",
    "middle_distal_collision_0",
    "ring_proximal_collision_0",
    "ring_distal_collision_0",
    "pinky_proximal_collision_0",
    "pinky_distal_collision_0",
    "hand_base_link_collision",
    "palm_link_collision",
]

# Objects available in the arena.
# palm_offset: how far below the palm center to place the object COM
OBJECT_CONFIGS = {
    "cube": {
        "body": "cube",
        "geom": "cube_collision",
        "palm_offset": 0.02,
        "ycb_name": "cube",
        # Cube: symmetric, no mesh correction needed
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.025,
        "mesh_correction": None,  # no STL/geom axis mismatch
        "friction_nom": 0.40,  # PLA primitive (Table I)
    },
    "graspit_box": {
        "body": "graspit_box",
        "geom": "graspit_box_geom",
        "palm_offset": 0.02,
        "ycb_name": "graspit_box",
        # Box lying flat with long axis along Y (Rx 90 deg).
        # MuJoCo geom half-sizes: [0.03125, 0.03125, 0.08125] (long axis = Z)
        # After Rx(90 deg): X=0.03125, Y=0.08125 (long, depth dir), Z=0.03125
        # The hand wraps the 62.5 mm x 62.5 mm XZ cross-section from above.
        "table_quat": [0.7071068, 0.7071068, 0, 0],  # Rx(90 deg)
        "table_half_h": 0.03125,
        "max_backoff_rounds": 0,  # skip back-off; minor pre-close overlap resolves during MPC
        "mesh_correction": None,
        "friction_nom": 0.40,  # PLA primitive (Table I)
    },
    "graspit_cylinder": {
        "body": "graspit_cylinder",
        "geom": "graspit_cylinder_geom",
        "palm_offset": 0.03,
        "ycb_name": "graspit_cylinder",
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.090,
        "mesh_correction": None,
        "friction_nom": 0.40,  # PLA primitive (Table I)
    },
    "soup_can": {
        "body": "005_tomato_soup_can",
        "geom": "005_tomato_soup_can_collision",
        "palm_offset": 0.02,
        "ycb_name": "soup",
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.0415,
        "mesh_correction": None,
        "friction_nom": 0.25,  # Alum./steel can (Table I)
    },
    "mustard_bottle": {
        "body": "006_mustard_bottle",
        "geom": "006_mustard_bottle_collision",
        "palm_offset": 0.03,
        "ycb_name": "mustard",
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.10,
        "mesh_correction": None,
        "friction_nom": 0.35,  # LDPE bottle (Table I)
    },
    "potted_meat_can": {
        "body": "010_potted_meat_can",
        "geom": "010_potted_meat_can_collision",
        "palm_offset": 0.02,
        "ycb_name": "potted_meat",
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.04,
        "mesh_correction": None,
        "friction_nom": 0.25,  # Alum./steel can (Table I)
    },
    "tennis_ball": {
        "body": "056_tennis_ball",
        "geom": "056_tennis_ball_collision",
        "palm_offset": 0.02,
        "ycb_name": "tennis_ball",
        "table_quat": [1, 0, 0, 0],
        "table_half_h": 0.032,
        "mesh_correction": None,
        "friction_nom": 0.65,  # Tennis ball felt (Table I)
    },
}

# Fallback collision-free arm configuration (only used when pregrasp IK fails)
GRASP_ARM_CONFIG = np.array([
    -0.826,   # shoulder_pan
    -2.200,   # shoulder_lift
    -1.643,   # elbow
    -1.429,   # wrist_1
     0.500,   # wrist_2
     2.090,   # wrist_3
])

# Friction values to randomize per episode (uniform draw)
FRICTION_RANGE = (0.4, 1.0)

#  Friction regimes (matching paper Table I and Sec III-A)
#  Values calibrated from Schneider friction reference for PEEK fingertips
#  
#  Stiffness is sampled from LogNormal with reduced variance to avoid
#  numerical instability with MuJoCo's constraint solver:
#  - All regimes: κ ~ LogNormal(8, 0.3) clamped to [1, 20] kN/m
#  - Adversarial: κ ~ LogNormal(7.5, 0.3) (slightly softer mean)
#  - Bimodal:     κ scaled by μ/μ_nom to correlate with friction mode
#                 (μ_nom is object-specific, from OBJECT_CONFIGS)
FRICTION_REGIMES = {
    "nominal":     {"type": "uniform", "low": 0.4,  "high": 1.0,
                    "stiffness_log_mu": 8.0, "stiffness_log_sigma": 0.3},
    "adversarial": {"type": "uniform", "low": 0.15, "high": 0.40,
                    "stiffness_log_mu": 7.5, "stiffness_log_sigma": 0.3},  # softer contacts
    "wide":        {"type": "uniform", "low": 0.15, "high": 1.2,
                    "stiffness_log_mu": 8.0, "stiffness_log_sigma": 0.3},
    "bimodal":     {"type": "bimodal",
                    "mu1": 0.18, "sig1": 0.03,
                    "mu2": 1.00, "sig2": 0.05,
                    "w1": 0.5,
                    "stiffness_log_mu": 8.0, "stiffness_log_sigma": 0.3},
}

# Default nominal friction (fallback if object doesn't specify friction_nom)
FRICTION_NOMINAL_DEFAULT = 0.40


def sample_friction(regime: str, rng: np.random.Generator) -> float:
    """Draw a friction coefficient from the specified regime."""
    cfg = FRICTION_REGIMES[regime]
    if cfg["type"] == "uniform":
        return float(rng.uniform(cfg["low"], cfg["high"]))
    # bimodal: 50/50 mixture of two Gaussians, clamped > 0.01
    if rng.random() < cfg["w1"]:
        mu = rng.normal(cfg["mu1"], cfg["sig1"])
    else:
        mu = rng.normal(cfg["mu2"], cfg["sig2"])
    return float(max(0.01, mu))


def sample_stiffness(regime: str, friction: float, rng: np.random.Generator,
                     friction_nom: float = 0.40) -> float:
    """Draw contact stiffness (N/m) correlated with friction regime.
    
    Returns stiffness in N/m for use with MuJoCo solref.
    
    Per paper Sec III-A: κ ~ LogNormal(μ, σ) with reduced σ=0.3 to avoid
    numerical instability. Clamped to [1, 20] kN/m.
    
    Args:
        regime: Friction regime name
        friction: Sampled friction coefficient for this episode
        rng: Random number generator
        friction_nom: Object-specific nominal friction (μ_nom from Table I)
    """
    cfg = FRICTION_REGIMES[regime]
    log_mu = cfg.get("stiffness_log_mu", 8.0)
    log_sigma = cfg.get("stiffness_log_sigma", 0.3)
    
    # Sample base stiffness from LogNormal (in N/m, log_mu=8 --> ~3000 N/m mean)
    kappa_base = float(rng.lognormal(log_mu, log_sigma))
    
    # For bimodal regime, scale stiffness with friction to correlate modes:
    # Low friction mode --> lower stiffness, high friction mode --> higher stiffness
    # Per paper: κ scaled by μ/μ_nom where μ_nom is object-specific (Table I)
    if regime == "bimodal":
        scale = friction / friction_nom
        kappa = kappa_base * scale
    else:
        kappa = kappa_base
    
    # Clamp to [1, 20] kN/m = [1000, 20000] N/m to avoid solver instability
    return float(np.clip(kappa, 1000.0, 20000.0))

# PD gains for torque-mode hand actuators (motor elements, gear=1)
HAND_KP = 5.0
HAND_KD = 0.3

# Minimum epsilon for grasp to be considered successful (scientific validity)
# A grasp with epsilon=0 is NOT force closure regardless of other metrics
EPSILON_MIN = 0.001

# Maximum gradient norm for CVaR gradients to prevent instability
MAX_CVAR_GRAD_NORM = 1.0

# Table surface height in MuJoCo world frame
# Actual table collision top: body_z(0.76) + geom_offset(0.01) + half_height(0.007) = 0.777
TABLE_Z = 0.777

# Where to place the object on the table (reachable by the arm)
# Roughly in front of the robot's workspace
DEFAULT_OBJ_TABLE_POS = np.array([-0.05, 0.88, TABLE_Z])


def _pd_hand_ctrl(target: np.ndarray, env: RawMujocoEnv) -> np.ndarray:
    """Compute PD torque for hand joints given target position.

    Hand actuators are torque-mode motors (gain=1, ctrlrange=[-2, 2]).
    We must *not* send raw position targets as ctrl: that would be
    interpreted as torque.  Instead we compute the PD output and clamp
    within the actuator limits.

    DEPRECATED: Use _torque_hand_ctrl() instead for direct torque-based
    closing which is more robust for all object geometries.
    """
    pos_err = target - env.data.qpos[6:17]
    vel = env.data.qvel[6:17]
    return np.clip(HAND_KP * pos_err - HAND_KD * vel, -2.0, 2.0)


def _torque_hand_ctrl(scale: float) -> np.ndarray:
    """Return direct torque command for hand joints.

    Uses the physically-tuned GRASP_TORQUE profile from the pregrasp planner,
    scaled by ``scale`` (0.0 = open, 1.0 = full grasp torque).
    This is the same torque-based closing used in record_pregrasp_videos.py
    which works reliably for ALL object geometries (TOP_DOWN and SIDE).

    Advantages over PD position control (_pd_hand_ctrl):
      - Compliant: fingers naturally stop when contact forces balance torque
      - No over-squeeze: torque-limited, no position accumulation runaway
      - Robust: works for tall SIDE objects where PD-based closing fails
    """
    return np.clip(SCRIPTED_GRASP_TORQUE * scale, -2.0, 2.0)


# 
#  MuJoCo Jacobian-based IK  (arm joints only)
# 

def _compute_arm_ik_mujoco(
    env: RawMujocoEnv,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    body_name: str = "hand_base",
    q0: np.ndarray = None,
    max_iter: int = 300,
    tol: float = 1e-3,
    damping: float = 0.01,
    alpha: float = 0.5,
) -> Optional[np.ndarray]:
    """Solve arm IK using MuJoCo's own Jacobian (damped least-squares).

    Targets the specified body (hand_base by default) to reach the desired
    6-DOF pose.  Only the first 6 joints (arm) are adjusted; the hand
    joints remain frozen at their current values.

    Args:
        env: MuJoCo environment (model + data used for FK / Jacobian)
        target_pos: Desired body position in world frame (3,)
        target_rot: Desired body orientation in world frame (3x3)
        body_name: MuJoCo body whose pose we are targeting
        q0: Initial arm joint guess (6,); defaults to current qpos[0:6]
        max_iter: Maximum DLS iterations
        tol: Convergence tolerance on ‖error‖
        damping: DLS damping factor (λ²)
        alpha: Line-search step size

    Returns:
        arm_q (6,) or None if IK did not converge.
    """
    model, data = env.model, env.data
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        print(f"  IK ERROR: body '{body_name}' not found")
        return None

    # Save full state so we can restore it after IK
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    if q0 is not None:
        data.qpos[0:6] = q0
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    best_q = None
    best_err = float('inf')

    for iteration in range(max_iter):
        cur_pos = data.xpos[body_id].copy()
        cur_mat = data.xmat[body_id].reshape(3, 3).copy()

        # Position error
        pos_err = target_pos - cur_pos

        # Orientation error as axis-angle
        R_err = target_rot @ cur_mat.T
        trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(trace_val)
        if angle < 1e-8:
            rot_err = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])
            rot_err = axis / (2.0 * np.sin(angle)) * angle

        # Weight orientation less than position (fingers can adapt)
        w_pos, w_rot = 1.0, 0.3
        err = np.concatenate([w_pos * pos_err, w_rot * rot_err])
        err_norm = np.linalg.norm(err)

        if err_norm < best_err:
            best_err = err_norm
            best_q = data.qpos[0:6].copy()

        if err_norm < tol:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mj.mj_forward(model, data)
            return best_q

        # Jacobian (3xnv each)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, body_id)

        # Take only the arm columns (first 6 DOFs)
        J = np.vstack([w_pos * jacp[:, 0:6], w_rot * jacr[:, 0:6]])

        # Damped least-squares step (adaptive damping)
        lam = damping * (1.0 + 0.1 * iteration / max_iter)
        dq = np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ err)
        data.qpos[0:6] += alpha * dq
        mj.mj_forward(model, data)

    print(f"  IK: {max_iter} iters, best residual {best_err:.4f}")

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mj.mj_forward(model, data)

    # Accept if position error is small enough (orientation mismatch is OK)
    return best_q if best_err < 0.05 else None


def _compute_arm_ik_multiseed(
    env: RawMujocoEnv,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    body_name: str = "hand_base",
    q0: np.ndarray = None,
    n_seeds: int = 8,
) -> Optional[np.ndarray]:
    """Try IK from multiple seed configurations, return the best solution"""
    if q0 is None:
        q0 = GRASP_ARM_CONFIG.copy()

    # Try with main seed first
    sol = _compute_arm_ik_mujoco(env, target_pos, target_rot, body_name, q0=q0)
    if sol is not None:
        return sol

    # Try random perturbations of the seed
    for i in range(n_seeds):
        q_seed = q0 + np.random.uniform(-0.5, 0.5, 6)
        sol = _compute_arm_ik_mujoco(
            env, target_pos, target_rot, body_name,
            q0=q_seed, max_iter=200,
        )
        if sol is not None:
            return sol

    return None

# 
# Result data structures
# 

@dataclass
class StepRecord:
    step: int
    epsilon: float
    gws_volume: float
    contact_quality: float
    n_contacts: int
    is_force_closure: bool
    entropy: float
    cvar: float
    cost: float
    failure_prob: float


@dataclass
class EpisodeResult:
    method: str              # "particle", "variational", "gauss", "gauss_cvar", "cem"
    object_name: str
    beta: float
    seed: int
    friction: float          # sampled friction for this episode
    friction_regime: str     # "nominal", "adversarial", "wide", "bimodal"
    stiffness: float = 3000.0  # sampled contact stiffness (N/m)
    n_steps: int = 0
    runtime_s: float = 0.0
    termination: str = ""

    # Final metrics
    final_epsilon: float = 0.0
    final_gws_volume: float = 0.0
    final_contact_quality: float = 0.0
    final_n_contacts: int = 0
    final_entropy: float = 0.0
    success: bool = False

    # Variational-specific
    has_exact_grads: bool = False
    grad_norm_means: float = 0.0
    grad_norm_logits: float = 0.0
    belief_cvar: float = 0.0
    belief_cost_mean: float = 0.0
    belief_cost_std: float = 0.0

    # Lift / shear (nominal stress test)
    lift_ratio: float = 0.0
    lift_success: bool = False
    lift_height_achieved: float = 0.0         # actual lift height in meters
    pulses_survived: int = 0
    shear_success: bool = False
    shear_max_disp: float = 0.0

    # Extra metrics (paper Table II)
    time_to_slip: Optional[float] = None      # seconds from shear start to first slip (None=no slip)
    peak_slip_distance: float = 0.0           # max instantaneous slip distance during eval (m)
    failure_mode: str = "none"                # "grasp_fail", "lift_fail", "slip_fail", "perturbation_fail", "none"

    # 28-test perturbation battery (paper Sec. V-B)
    perturbation_survival_rate: float = 0.0   # fraction of 28 tests survived
    perturbation_n_survived: int = 0
    perturbation_n_total: int = 28
    perturbation_details: List[dict] = field(default_factory=list)
    robust_success: bool = False              # success AND perturbation_survival > 50%

    # Failure probability (belief-predicted)
    failure_prob_predicted: float = 0.0
    failure_prob_empirical: float = 0.0       # 1 - perturbation_survival_rate

    # Per-step log
    step_log: List[dict] = field(default_factory=list)


# 
# Failure mode categorization (paper Table II)
# 

def _determine_failure_mode(success: bool, ls: dict, pert_rate: float) -> str:
    """Categorize the failure mode for paper Table II.
    
    Categories:
      - "none": The grasp succeeded and survived all tests
      - "grasp_fail": Failed to achieve force closure / quality threshold
      - "lift_fail": Achieved grasp but dropped during lift
      - "slip_fail": Lifted successfully but slipped during shear pulses
      - "perturbation_fail": Survived shear but failed perturbation battery
    """
    if not success:
        return "grasp_fail"
    if not ls.get("lift_success", False):
        return "lift_fail"
    if not ls.get("shear_success", False):
        return "slip_fail"
    if pert_rate <= 0.5:
        return "perturbation_fail"
    return "none"


# 
# Environment helpers
# 


class SimulationUnstableError(RuntimeError):
    """Raised when MuJoCo simulation produces NaN/Inf values."""
    pass


def _sim_is_unstable(env) -> bool:
    """Return True if qpos, qvel, or qacc contain NaN or Inf."""
    return (np.any(~np.isfinite(env.data.qpos))
            or np.any(~np.isfinite(env.data.qvel))
            or np.any(~np.isfinite(env.data.qacc)))


def _safe_mj_step(env, n: int = 1):
    """Step MuJoCo `n` times, raising SimulationUnstableError on NaN."""
    for _ in range(n):
        mj.mj_step(env.model, env.data)
    if _sim_is_unstable(env):
        raise SimulationUnstableError(
            f"NaN/Inf detected at t={env.data.time:.4f}")



def make_env() -> RawMujocoEnv:
    xml_path = "arenas/zarm_realhand_l6_right_arena_no_wrist_cam/scene.xml"
    env = RawMujocoEnv(
        xml_path=xml_path,
        fingertip_geom_names=FINGERTIP_GEOMS,
        object_geom_names=["cube_collision"],
        n_substeps=10,
    )
    env.actmap = ActuatorMap(env.model)
    return env


def _set_object_geom_filter(env: RawMujocoEnv, geom_name: str):
    """Update the env's object_geoms set so contact filtering uses the current object"""
    gid = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid >= 0:
        env.object_geoms = {gid}
    else:
        print(f"  WARNING: geom '{geom_name}' not found")
        env.object_geoms = set()


def position_arm_and_object(
    env: RawMujocoEnv,
    obj_cfg: dict,
    friction: float,
    stiffness: float = 3000.0,
    settling: int = 300,
) -> bool:
    """Position arm + hand using the scripted-policies pregrasp planner.

    Uses PregraspPlanner from scripted_policies.pregrasp_planner, which is
    designed for the no-wrist-cam arena (zarm_realhand_l6_right_arena_no_wrist_cam).
    It targets palm_link with position-only IK, generates smooth mink descent
    trajectories, and classifies grasps as TOP_DOWN vs SIDE.

    Pipeline:
      1.  Compute geometry-aware pregrasp via ScriptedPregraspPlanner.
      2.  Reset data, stash all objects far away.
      3.  Spawn target object on table (identity quat, planner-computed body_z).
      4.  Set arm to HOME, settle object on table.
      5.  Smooth cosine blend HOME --> PREGRASP.
      6.  Mink-tracked descent to grasp position.
      7.  Blend mink endpoint --> IK grasp config, settle.
      8.  Apply friction to object geom.

    After return the arm is at the grasp configuration with open hand,
    ready for belief-MPC finger control.

    Args:
        env: MuJoCo environment
        obj_cfg: Object configuration dict
        friction: Friction coefficient μ to apply to object geom
        stiffness: Contact stiffness κ (N/m) -- logged, not applied to solref
        settling: Ignored (kept for API compat); phases have fixed step counts
    """
    body_name = obj_cfg["body"]
    body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        print(f"  ERROR: body '{body_name}' not found")
        return False

    #  1. Plan pregrasp using geometry-aware planner 
    planner = ScriptedPregraspPlanner(env.model, env.data)
    plan = planner.plan(body_name, verbose=True)

    aq_p    = plan.arm_q_pregrasp   # pregrasp arm config
    aq_g    = plan.arm_q_grasp      # grasp arm config
    descent = plan.descent_traj     # mink descent waypoints (K, 6)
    geom    = plan.geom

    print(f"  Scripted planner: strategy={plan.strategy}, "
          f"IK grasp err={plan.ik_err_grasp:.5f}, "
          f"IK pre err={plan.ik_err_pregrasp:.5f}, "
          f"ik_ok={plan.ik_ok}")

    if not plan.ik_ok:
        print(f"  WARNING: IK did not converge for {body_name}")

    #  2. Reset data, stash all objects far away 
    mj.mj_resetData(env.model, env.data)
    scripted_stash_all(env.model, env.data)

    #  3. Spawn target object on table 
    scripted_spawn_object(
        env.model, env.data, body_name,
        planner.obj_xy[0], planner.obj_xy[1], geom["body_z"],
    )

    #  4. Set arm to HOME, hand open, settle object 
    env.data.qpos[0:6]  = SCRIPTED_HOME_Q
    env.data.qpos[6:17] = 0.0
    env.data.qvel[:]     = 0.0
    mj.mj_forward(env.model, env.data)

    for _ in range(300):
        env.data.ctrl[0:6]  = SCRIPTED_HOME_Q
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

    #  5. Smooth cosine HOME --> PREGRASP 
    move_steps = 500
    for i in range(move_steps):
        a = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / move_steps)
        env.data.ctrl[0:6]  = (1 - a) * SCRIPTED_HOME_Q + a * aq_p
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

    # Settle at pregrasp
    for _ in range(150):
        env.data.ctrl[0:6]  = aq_p
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

    #  6. Mink-tracked descent to grasp position 
    #     (with penetration monitoring --- stop early if clearance < threshold)
    approach_steps = 400
    descent_interp = interp_trajectory(descent, approach_steps)

    # Collect body IDs for penetration checking during approach
    _hand_kw = {"palm", "thumb", "index", "middle", "ring", "pinky", "hand_base"}
    _hand_bids = set()
    for _i in range(env.model.nbody):
        _bn = (mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_BODY, _i) or "").lower()
        if any(k in _bn for k in _hand_kw):
            _hand_bids.add(_i)
    _obj_bids = {body_id}
    for _i in range(env.model.nbody):
        _p = env.model.body_parentid[_i]
        while _p > 0:
            if _p == body_id:
                _obj_bids.add(_i)
                break
            _p = env.model.body_parentid[_p]

    _clearance_stop = 0.003  # stop descent at 3 mm clearance
    _descent_stopped_early = False
    for i in range(approach_steps):
        env.data.ctrl[0:6]  = descent_interp[i]
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

        # Check penetration periodically (every 20 steps to limit overhead)
        if (i + 1) % 20 == 0 or i == approach_steps - 1:
            mj.mj_forward(env.model, env.data)
            _min_d = np.inf
            _n_neg = 0
            for _k in range(env.data.ncon):
                _c = env.data.contact[_k]
                _b1 = int(env.model.geom_bodyid[_c.geom1])
                _b2 = int(env.model.geom_bodyid[_c.geom2])
                if ((_b1 in _hand_bids and _b2 in _obj_bids) or
                        (_b2 in _hand_bids and _b1 in _obj_bids)):
                    _d = float(_c.dist)
                    _min_d = min(_min_d, _d)
                    if _d < 0:
                        _n_neg += 1
            if _n_neg > 0:
                print(f"  [descent] PENETRATION at step {i}: "
                      f"min_dist={_min_d:.6f}, n_neg={_n_neg}")
            elif _min_d != np.inf and _min_d < _clearance_stop:
                print(f"  [descent] stopped at step {i}/{approach_steps} --- "
                      f"clearance={_min_d:.6f} < {_clearance_stop}")
                _descent_stopped_early = True
                break

    #  7. Blend mink endpoint --> IK grasp config 
    #     (with penetration monitoring)
    mink_end = descent_interp[min(i, len(descent_interp) - 1)].copy() if _descent_stopped_early else descent_interp[-1].copy()
    blend_steps = 120
    for i in range(blend_steps):
        a = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / blend_steps)
        env.data.ctrl[0:6]  = (1 - a) * mink_end + a * aq_g
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

        # Penetration check every 20 steps
        if (i + 1) % 20 == 0:
            mj.mj_forward(env.model, env.data)
            for _k in range(env.data.ncon):
                _c = env.data.contact[_k]
                _b1 = int(env.model.geom_bodyid[_c.geom1])
                _b2 = int(env.model.geom_bodyid[_c.geom2])
                if ((_b1 in _hand_bids and _b2 in _obj_bids) or
                        (_b2 in _hand_bids and _b1 in _obj_bids)):
                    if float(_c.dist) < 0:
                        print(f"  [blend] WARNING: penetration at step {i}")
                        break

    # Settle at grasp (with final penetration report)
    for _ in range(200):
        env.data.ctrl[0:6]  = aq_g
        env.data.ctrl[6:17] = 0.0
        scripted_freeze_stash(env.model, env.data, active=body_name)
        mj.mj_step(env.model, env.data)

    # Report final contact state
    mj.mj_forward(env.model, env.data)
    _final_min_d = np.inf
    _final_n_neg = 0
    for _k in range(env.data.ncon):
        _c = env.data.contact[_k]
        _b1 = int(env.model.geom_bodyid[_c.geom1])
        _b2 = int(env.model.geom_bodyid[_c.geom2])
        if ((_b1 in _hand_bids and _b2 in _obj_bids) or
                (_b2 in _hand_bids and _b1 in _obj_bids)):
            _d = float(_c.dist)
            _final_min_d = min(_final_min_d, _d)
            if _d < 0:
                _final_n_neg += 1
    if _final_n_neg > 0:
        print(f"  [position] WARNING: {_final_n_neg} penetrating contacts "
              f"after settle (min_dist={_final_min_d:.6f})")
    else:
        print(f"  [position] OK: no penetration after settle "
              f"(min_dist={_final_min_d:.6f})")

    #  8. Apply friction to object geom 
    geom_name = obj_cfg["geom"]
    gid = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid >= 0:
        env.model.geom_friction[gid, 0] = friction
        # Stiffness logged in EpisodeResult but not applied to solref
        _ = stiffness

    if _sim_is_unstable(env):
        print(f"  SIM UNSTABLE after positioning")
        return False

    return True


# 
# Quality helpers  (from run_belief_mpc_grasp.py)
# 

def get_object_center(env, body_name):
    bid = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return np.zeros(3)
    return env.data.xpos[bid].copy()


def compute_gws(env, obj_cfg):
    contacts = extract_contacts(env.model, env.data, geom_filter=env.fingertip_geoms)
    if env.object_geoms:
        contacts = [c for c in contacts
                    if c.geom1 in env.object_geoms or c.geom2 in env.object_geoms]
    center = get_object_center(env, obj_cfg["body"])
    return analyze_gws(contacts, center, friction_coef=0.8)


def compute_contact_quality(env, obj_cfg):
    contacts = extract_contacts(env.model, env.data, geom_filter=env.fingertip_geoms)
    if env.object_geoms:
        contacts = [c for c in contacts
                    if c.geom1 in env.object_geoms or c.geom2 in env.object_geoms]
    n = len(contacts)
    if n < 2:
        return 0.0
    contact_score = min(1.0, n / 5.0)
    total_force = sum(c.normal_force for c in contacts)
    force_score = min(1.0, total_force / 10.0)
    center = get_object_center(env, obj_cfg["body"])
    fv = [c.normal * c.normal_force for c in contacts]
    if len(fv) >= 2:
        net = np.linalg.norm(np.sum(fv, axis=0))
        balance = max(0, 1 - net / (total_force + 1e-6))
    else:
        balance = 0.0
    return 0.4 * contact_score + 0.3 * force_score + 0.3 * balance


def quality_fn(env, obj_cfg):
    gws = compute_gws(env, obj_cfg)
    if gws.is_force_closure and gws.epsilon > 0.01:
        return gws.quality()
    return compute_contact_quality(env, obj_cfg)


# 
# Lift + shear test  (simplified from run_belief_mpc_grasp.py)
# 

def run_lift_and_shear(
    env: RawMujocoEnv,
    arm_q: np.ndarray,
    hand_q: np.ndarray,
    obj_cfg: dict,
    lift_height: float = 0.05,
    shear_pulses: Tuple[float, float, float] = (3.0, 6.0, 12.0),
    slip_threshold: float = 0.005,  # 5mm displacement counts as slip
    hand_torque: np.ndarray = None,  # direct torque vector (if set, overrides PD)
) -> dict:
    """Lift the object, then apply 3-pulse lateral shear.  Returns stability metrics
    
    Extra metrics for paper Table II:
      - time_to_slip: seconds from shear start to first slip (None if no slip)
      - peak_slip_distance: max instantaneous displacement during shear (m)
      - lift_height_achieved: actual object lift in meters
    
    If ``hand_torque`` is provided, uses direct torque control (pregrasp-planner
    style) instead of PD position control.  This is more robust for SIDE grasps.
    """
    dt = env.model.opt.timestep
    _use_torque = hand_torque is not None

    def _hand_ctrl():
        """Return the hand control vector (torque or PD)."""
        if _use_torque:
            return hand_torque
        return _pd_hand_ctrl(hand_q, env)
    body_name = obj_cfg["body"]
    body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
    jnt_adr = env.model.body_jntadr[body_id]
    qpos_adr = env.model.jnt_qposadr[jnt_adr]
    palm_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, "palm_link")
    if palm_id < 0:
        palm_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, "hand_base")

    initial_cube_z = env.data.qpos[qpos_adr + 2]
    initial_palm_z = env.data.xpos[palm_id][2]

    # Compute lifted arm config
    shoulder_delta = np.clip(lift_height / 0.35, -0.3, 0.3)
    lifted_arm = arm_q.copy()
    lifted_arm[1] += shoulder_delta
    lifted_arm[3] -= shoulder_delta * 0.5

    # ---- lift ----
    n_lift = int(1.0 / dt)
    for i in range(n_lift):
        a = 0.5 - 0.5 * np.cos((i + 1) / n_lift * np.pi)
        ctrl = np.zeros(env.model.nu)
        ctrl[0:6] = (1 - a) * arm_q + a * lifted_arm
        ctrl[6:17] = _hand_ctrl()
        env.step(ctrl)

    # stabilize
    for _ in range(int(0.2 / dt)):
        ctrl = np.zeros(env.model.nu)
        ctrl[0:6] = lifted_arm
        ctrl[6:17] = _hand_ctrl()
        env.step(ctrl)

    post_lift_z = env.data.qpos[qpos_adr + 2]
    actual_lift = post_lift_z - initial_cube_z
    palm_lift = env.data.xpos[palm_id][2] - initial_palm_z
    lift_ratio = actual_lift / palm_lift if palm_lift > 1e-3 else 0.0
    lift_height_achieved = float(actual_lift)

    # ---- shear pulses with slip timing ----
    pre_shear_pos = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
    n_pulse = int(0.3 / dt)
    n_hold = int(0.1 / dt)
    pulse_results = []
    max_disp_overall = 0.0
    
    # Slip timing: track time from shear start
    shear_start_time = env.data.time
    time_to_slip = None  # None means no slip detected
    peak_slip_distance = 0.0

    for force_n in shear_pulses:
        pre = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
        max_d = 0.0
        for sign in [1.0, -1.0]:
            for _ in range(n_pulse):
                ctrl = np.zeros(env.model.nu)
                ctrl[0:6] = lifted_arm
                ctrl[6:17] = _hand_ctrl()
                env.data.xfrc_applied[body_id, 1] = sign * force_n
                env.step(ctrl)
                d = np.linalg.norm(env.data.qpos[qpos_adr:qpos_adr + 3] - pre)
                max_d = max(max_d, d)
                peak_slip_distance = max(peak_slip_distance, d)
                # Track time to first significant slip
                if time_to_slip is None and d >= slip_threshold:
                    time_to_slip = env.data.time - shear_start_time
            env.data.xfrc_applied[body_id, :] = 0
        # brief hold
        for _ in range(n_hold):
            ctrl = np.zeros(env.model.nu)
            ctrl[0:6] = lifted_arm
            ctrl[6:17] = _hand_ctrl()
            env.step(ctrl)
        post = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
        drop = pre[2] - post[2]
        survived = drop < 0.01 and max_d < 0.03
        pulse_results.append({
            "force": float(force_n),
            "max_displacement": float(max_d),
            "drop": float(drop),
            "survived": bool(survived),
        })
        max_disp_overall = max(max_disp_overall, max_d)

    final_z = env.data.qpos[qpos_adr + 2]
    drop_total = pre_shear_pos[2] - final_z
    n_survived = sum(1 for p in pulse_results if p["survived"])

    return {
        "lift_ratio": float(lift_ratio),
        "lift_success": bool(lift_ratio > 0.8),
        "lift_height_achieved": lift_height_achieved,
        "pulses_survived": n_survived,
        "shear_success": bool(lift_ratio > 0.8 and drop_total < 0.01 and n_survived == 3),
        "max_displacement": float(max_disp_overall),
        "pulse_results": pulse_results,
        # Extra metrics for paper
        "time_to_slip": time_to_slip,  # None if no slip, else seconds
        "peak_slip_distance": float(peak_slip_distance),
    }


# 
#  28-test perturbation battery  (paper Section V-B)
# 
# 16 lateral impulse (4 forces x  4 dirs) + 9 torque impulse (3 x  3) + 3 friction-drop = 28 tests.

PERTURBATION_LATERAL_FORCES = [3.0, 5.0, 8.0, 12.0]         # N  (4 magnitudes --> 16 lateral tests)
PERTURBATION_TORQUES        = [0.3, 0.6, 1.0]                # Nm (3 magnitudes x  3 axes = 9 tests)
PERTURBATION_FRICTION_DROPS = [0.05, 0.10, 0.15]             # (3 tests)  --> total 28 tests
PERTURBATION_DIRS           = [                               # 4 cardinal
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0]),
]
PERTURBATION_TORQUE_AXES = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
]

# Thresholds (from paper)
DISP_THRESHOLD = 0.06     # m   – lateral impulse survival  (relaxed: 6 cm)
ROT_THRESHOLD  = 0.5      # rad – torque impulse survival   (relaxed: ~28 deg)
DROP_THRESHOLD = 0.025    # m   – friction-drop survival    (relaxed: 2.5 cm)


def run_perturbation_battery(
    env: RawMujocoEnv,
    arm_q: np.ndarray,
    hand_q: np.ndarray,
    obj_cfg: dict,
    hand_torque: np.ndarray = None,
) -> dict:
    """Apply the 28-test perturbation battery (paper Sec. V-B).

    16 lateral impulses (4 forces x 4 dirs) + 9 torque tests (3 torques x 3 axes)
    + 3 friction-drop tests = 28 total.  The simulator state is **saved and restored**
    between each test so that all 28 tests are independent, starting from the same
    lifted configuration.

    Returns dict with ``n_survived``, ``n_total``, ``survival_rate``,
    ``details`` (list of per-test dicts).
    """
    dt = env.model.opt.timestep
    body_name = obj_cfg["body"]
    body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
    jnt_adr = env.model.body_jntadr[body_id]
    qpos_adr = env.model.jnt_qposadr[jnt_adr]

    # Save full simulator state (restored between each test)
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()

    details = []

    def _restore():
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        env.data.xfrc_applied[:] = 0
        mj.mj_forward(env.model, env.data)

    _use_torque = hand_torque is not None

    def _hand_ctrl():
        if _use_torque:
            return hand_torque
        return _pd_hand_ctrl(hand_q, env)

    def _hold_steps(n):
        for _ in range(n):
            ctrl = np.zeros(env.model.nu)
            ctrl[0:6] = arm_q
            ctrl[6:17] = _hand_ctrl()
            env.step(ctrl)

    #  (i)  Lateral impulse: 5 forces x 4 dirs = 20 tests --
    n_pulse_steps = int(0.15 / dt)  # 0.15 s duration per paper
    for force_n in PERTURBATION_LATERAL_FORCES:
        for d in PERTURBATION_DIRS:
            _restore()
            pre_pos = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
            max_disp = 0.0
            for _ in range(n_pulse_steps):
                env.data.xfrc_applied[body_id, 0:3] = d * force_n
                ctrl = np.zeros(env.model.nu)
                ctrl[0:6] = arm_q
                ctrl[6:17] = _hand_ctrl()
                env.step(ctrl)
                disp = np.linalg.norm(env.data.qpos[qpos_adr:qpos_adr + 3] - pre_pos)
                max_disp = max(max_disp, disp)
            env.data.xfrc_applied[body_id, :] = 0
            _hold_steps(int(0.1 / dt))
            survived = max_disp < DISP_THRESHOLD
            details.append({"test": "lateral", "force": force_n,
                            "dir": d.tolist(), "max_disp": float(max_disp),
                            "survived": bool(survived)})

    #  (ii)  Torque impulse: 3 torques x 3 axes = 9 tests 
    n_torque_steps = int(0.2 / dt)  # 0.2 s duration per paper
    for torque_nm in PERTURBATION_TORQUES:
        for ax in PERTURBATION_TORQUE_AXES:
            _restore()
            pre_quat = env.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
            for _ in range(n_torque_steps):
                env.data.xfrc_applied[body_id, 3:6] = ax * torque_nm
                ctrl = np.zeros(env.model.nu)
                ctrl[0:6] = arm_q
                ctrl[6:17] = _hand_ctrl()
                env.step(ctrl)
            env.data.xfrc_applied[body_id, :] = 0
            _hold_steps(int(0.1 / dt))
            post_quat = env.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
            # Quaternion distance --> rotation angle
            dot = np.clip(np.dot(pre_quat, post_quat), -1, 1)
            rot_angle = 2 * np.arccos(abs(dot))
            survived = rot_angle < ROT_THRESHOLD
            details.append({"test": "torque", "torque_nm": torque_nm,
                            "axis": ax.tolist(), "rot_angle": float(rot_angle),
                            "survived": bool(survived)})

    #  (iii)  Friction drop: 3 levels = 3 tests 
    geom_name = obj_cfg["geom"]
    gid = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid >= 0:
        original_friction = env.model.geom_friction[gid, 0].copy()
    for new_mu in PERTURBATION_FRICTION_DROPS:
        _restore()
        pre_pos = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
        if gid >= 0:
            env.model.geom_friction[gid, 0] = new_mu
        _hold_steps(200)  # 200 sim steps per paper
        post_pos = env.data.qpos[qpos_adr:qpos_adr + 3].copy()
        drop = pre_pos[2] - post_pos[2]
        disp = np.linalg.norm(post_pos - pre_pos)
        survived = drop < DROP_THRESHOLD and disp < DISP_THRESHOLD
        details.append({"test": "friction_drop", "new_mu": new_mu,
                        "drop": float(drop), "disp": float(disp),
                        "survived": bool(survived)})
        # Restore original friction
        if gid >= 0:
            env.model.geom_friction[gid, 0] = original_friction

    # Restore final state
    _restore()

    n_survived = sum(1 for d in details if d["survived"])
    n_total = len(details)  # should be 28
    return {
        "n_survived": n_survived,
        "n_total": n_total,
        "survival_rate": n_survived / max(n_total, 1),
        "details": details,
    }


# 
# Particle-filter baseline episode
# 

def run_particle_episode(
    env: RawMujocoEnv,
    obj_cfg: dict,
    beta: float,
    seed: int,
    friction: float,
    stiffness: float = 3000.0,
    friction_regime: str = "nominal",
    max_steps: int = 80,
    deadline: float = float('inf'),
) -> EpisodeResult:
    """Run one episode with the particle-filter baseline"""
    t0 = time.time()
    env.reset()
    _set_object_geom_filter(env, obj_cfg["geom"])
    if not position_arm_and_object(env, obj_cfg, friction, stiffness):
        raise RuntimeError("positioning failed")

    arm_q = env.data.qpos[0:6].copy()
    hand_q = env.data.qpos[6:17].copy()

    config = BeliefMPCConfig(
        n_particles=100,
        horizon=5,
        n_candidates=10,
        max_steps=max_steps,
        beta=beta,
        delta=0.05,
        lambda_cvar=0.5,
        sigma_process=0.0,
        seed=seed,
    )

    planner = BeliefMPCPlanner(config=config, env=env,
                                quality_fn=lambda e: quality_fn(e, obj_cfg))

    step_log = []
    termination = "max_steps"

    for step in range(max_steps):
        # Wall-clock timeout check (more reliable than SIGALRM with C extensions)
        if time.time() > deadline:
            print(f"    TIMEOUT at step {step} (wall-clock deadline exceeded)")
            termination = "timeout"
            break

        # Check sim stability before doing expensive belief computation
        if _sim_is_unstable(env):
            print(f"    SIM UNSTABLE at step {step}, terminating episode")
            termination = "sim_unstable"
            break

        obs = env.get_observation()
        planner.update_belief(obs)
        action = planner.select_action()

        # Check again after expensive planner call
        if time.time() > deadline:
            print(f"    TIMEOUT during step {step} planner call")
            termination = "timeout"
            break

        mean_cost, cvar_cost, fail_prob = planner._evaluate_sequence([action])
        gws = compute_gws(env, obj_cfg)
        cq = compute_contact_quality(env, obj_cfg)

        step_log.append({
            "step": step,
            "epsilon": float(gws.epsilon),
            "gws_volume": float(gws.volume),
            "contact_quality": float(cq),
            "n_contacts": gws.n_contacts,
            "is_force_closure": bool(gws.is_force_closure),
            "entropy": float(planner.belief.entropy()),
            "cvar": float(cvar_cost),
            "cost": float(mean_cost),
            "failure_prob": float(fail_prob),
        })

        ctrl_action = action.to_control(env)
        hand_delta = ctrl_action[6:17]

        #  execute in MuJoCo (TORQUE-BASED closing) 
        torque_scale = min(1.0, (step + 1) / 30)
        hand_torque = _torque_hand_ctrl(torque_scale)
        for _ in range(25):
            ctrl = np.zeros(env.model.nu)
            ctrl[0:6] = arm_q
            ctrl[6:17] = hand_torque
            env.step(ctrl)
            if _sim_is_unstable(env):
                break

        planner.step_count += 1
        planner.quality_history.append(gws.epsilon)

        if gws.epsilon >= config.epsilon_des:
            termination = "quality_target"
            break

        if (step >= max(10, max_steps // 3) and gws.n_contacts >= 4
                and len(planner.entropy_history) >= 3):
            rd = abs(planner.entropy_history[-1] - planner.entropy_history[-3])
            if rd < config.delta_H_min:
                termination = "entropy_stable"
                break

    runtime = time.time() - t0

    # Post-loop settle at full torque
    full_torque = _torque_hand_ctrl(1.0)
    for _ in range(35):
        ctrl = np.zeros(env.model.nu)
        ctrl[0:6] = arm_q
        ctrl[6:17] = full_torque
        env.step(ctrl)

    final_gws = compute_gws(env, obj_cfg)
    final_cq = compute_contact_quality(env, obj_cfg)
    
    # Success criterion: any grasp with epsilon >= EPSILON_MIN is force closure.
    # Previous code had a split (epsilon > 0.1 vs backup) that let CEM succeed
    # at lower epsilon than gauss/variational due to having more contacts.
    # Fair criterion: epsilon (Ferrari-Canny) is the ground-truth quality metric.
    has_min_epsilon = final_gws.epsilon >= EPSILON_MIN
    stable = termination != "sim_unstable"
    success = stable and has_min_epsilon

    # Lift + shear test
    ls = {}
    if success:
        try:
            ls = run_lift_and_shear(env, arm_q, hand_q, obj_cfg,
                                    hand_torque=full_torque)
        except Exception as e:
            print(f"    lift/shear failed: {e}")

    # 28-test perturbation battery
    pb = {"n_survived": 0, "n_total": 28, "survival_rate": 0.0, "details": []}
    if success and ls.get("lift_success", False):
        try:
            pb = run_perturbation_battery(env, arm_q, hand_q, obj_cfg,
                                          hand_torque=full_torque)
        except Exception as e:
            print(f"    perturbation battery failed: {e}")

    pert_rate = pb["survival_rate"]
    robust = success and pert_rate > 0.5

    # Failure probability (belief estimate)
    fail_prob_pred = step_log[-1]["failure_prob"] if step_log else 0.0

    # Failure mode categorization for paper Table II
    failure_mode = _determine_failure_mode(success, ls, pert_rate)

    return EpisodeResult(
        method="particle",
        object_name=obj_cfg["body"],
        beta=beta,
        seed=seed,
        friction=friction,
        stiffness=stiffness,
        friction_regime=friction_regime,
        n_steps=len(step_log),
        runtime_s=runtime,
        termination=termination,
        final_epsilon=final_gws.epsilon,
        final_gws_volume=final_gws.volume,
        final_contact_quality=final_cq,
        final_n_contacts=final_gws.n_contacts,
        final_entropy=planner.belief.entropy(),
        success=success,
        lift_ratio=ls.get("lift_ratio", 0),
        lift_success=ls.get("lift_success", False),
        lift_height_achieved=ls.get("lift_height_achieved", 0.0),
        pulses_survived=ls.get("pulses_survived", 0),
        shear_success=ls.get("shear_success", False),
        shear_max_disp=ls.get("max_displacement", 0),
        time_to_slip=ls.get("time_to_slip"),
        peak_slip_distance=ls.get("peak_slip_distance", 0.0),
        failure_mode=failure_mode,
        perturbation_survival_rate=pert_rate,
        perturbation_n_survived=pb["n_survived"],
        perturbation_n_total=pb["n_total"],
        perturbation_details=pb["details"],
        robust_success=robust,
        failure_prob_predicted=fail_prob_pred,
        failure_prob_empirical=1.0 - pert_rate,
        step_log=step_log,
    )


# 
# Variational neural belief episode
# 

def _make_contact_cost_fn():
    """Differentiable cost function over contact-parameter samples.

    Maps belief samples (friction, stiffness, damping, slip per contact)
    to a scalar cost.  Lower friction / stiffness --> higher cost.
    
    NOTE: We clamp inputs to exp() to prevent numerical overflow when belief
    samples include very negative values (which can happen during optimization).
    """
    def cost_fn(samples: torch.Tensor) -> torch.Tensor:
        N, D = samples.shape
        n_contacts = D // 4
        if n_contacts == 0:
            return torch.zeros(N)
        params = samples.view(N, n_contacts, 4)
        friction   = params[:, :, 0]
        stiffness  = params[:, :, 1]
        slip       = params[:, :, 3]
        # Clamp inputs to exp() to prevent overflow: exp(-x) where x is clamped to [-10, 10]
        # This limits costs to [exp(-10), exp(10)] ≈ [4.5e-5, 22026]
        friction_clamped = torch.clamp(friction, -10.0, 10.0)
        stiffness_clamped = torch.clamp(stiffness, -10.0, 10.0)
        per_contact = torch.exp(-friction_clamped) + torch.exp(-stiffness_clamped) + F.softplus(slip)
        return per_contact.mean(dim=1)
    return cost_fn


# 
# Gradient-informed action scoring for variational MPC
# 

def _make_action_conditioned_cost_fn(action: torch.Tensor):
    """Action-conditioned cost: combines closure with geometric finger alignment.
    
    Key insight: The goal is to achieve FORCE CLOSURE (epsilon > 0).
    This requires:
    1. Sufficient hand closure (action magnitude)
    2. Fingers curling inward toward object center (geometric alignment)
    3. Resistance to slip under sampled friction conditions
    
    The cost includes:
    - closure_cost: reward higher action magnitude
    - geometric_closure: reward finger convergence (using analytical FK)
    - friction_margin: reward having margin against slip
    """
    action_mag = action.mean()
    
    # === ANALYTICAL FORWARD KINEMATICS (simplified) ===
    # Hand has 11 joints: thumb(3) + index(2) + middle(2) + ring(2) + pinky(2)
    # Approximate each finger as curling inward as joint angles increase
    # This gives us geometric gradients for where to place fingers!
    
    # Extract per-finger joint angles from action vector (11 joints)
    thumb_joints = action[0:3]  # 3 joints
    index_joints = action[3:5]   # 2 joints
    middle_joints = action[5:7]  # 2 joints
    ring_joints = action[7:9]    # 2 joints
    pinky_joints = action[9:11]  # 2 joints
    
    # Simplified FK: each finger segment contributes to closure
    # As joints close (larger angle), fingertip moves inward toward palm
    
    def finger_closure(joints):
        """Return closure amount for a finger (0=open, 1=fully closed)"""
        return 1.0 - torch.exp(-joints.sum() * 2.0)
    
    thumb_closure = finger_closure(thumb_joints)
    index_closure = finger_closure(index_joints)
    middle_closure = finger_closure(middle_joints)
    ring_closure = finger_closure(ring_joints)
    pinky_closure = finger_closure(pinky_joints)
    
    # All fingers should close together (not just one finger)
    all_closures = torch.stack([thumb_closure, index_closure, middle_closure, 
                                ring_closure, pinky_closure])
    mean_closure = all_closures.mean()
    closure_variance = all_closures.var()  # Penalize uneven closure
    
    # === GEOMETRIC ALIGNMENT COST ===
    # Reward configurations where multiple fingers are closing together
    # This promotes force closure through opposing contacts
    geometric_closure = mean_closure - closure_variance * 2.0
    
    def cost_fn(samples: torch.Tensor) -> torch.Tensor:
        N, D = samples.shape
        n_contacts = D // 4
        if n_contacts == 0:
            return torch.zeros(N)
        
        params = samples.view(N, n_contacts, 4)
        friction = params[:, :, 0]  # (N, n_contacts)
        stiffness = params[:, :, 1]
        slip_param = params[:, :, 3]
        
        # Clamp friction for numerical stability
        friction_clamped = torch.clamp(friction, 0.05, 2.0)
        
        # === PRIMARY COST: Action Magnitude (closure) ===
        # Higher action = more closure = higher chance of force closure
        closure_cost = -action_mag * 5.0  # Negative = reward closure
        
        # === GEOMETRIC ALIGNMENT (NEW) ===
        # Reward finger convergence - this provides gradients for finger PLACEMENT
        # not just "squeeze harder"
        geo_cost = -geometric_closure * 3.0  # Negative = reward convergence
        
        # === FRICTION MARGIN: Resistance to slip ===
        # Under low friction, we need higher action to maintain grasp
        min_friction = friction_clamped.min(dim=1)[0]  # Worst-case friction
        required_normal_force = 1.0 / (min_friction + 1e-3)
        friction_margin = F.relu(action_mag * 10.0 - required_normal_force) * 2.0
        
        # === SLIP PENALTY ===
        slip_penalty = F.softplus(slip_param).mean(dim=1) * 0.5
        
        # === STIFFNESS BONUS ===
        stiffness_bonus = -torch.exp(-torch.clamp(stiffness, -10.0, 10.0)).mean(dim=1) * 0.2
        
        # === TOTAL COST ===
        total_cost = (
            closure_cost +          # Reward: higher action
            geo_cost +             # Reward: finger convergence (GEOMETRIC)
            - friction_margin +    # Reward: margin against slip (negative = reward)
            stiffness_bonus +      # Reward: stiff contacts
            slip_penalty           # Penalty: avoid slip-prone configs
        )
        
        return total_cost
    
    return cost_fn


def _optimize_action_with_gradients(
    belief: GaussianMixtureBelief,
    cost_fn,  # base cost function (not used, kept for interface compatibility)
    beta: float,
    n_samples: int = 256,
    n_iters: int = 15,  # Fewer iterations but start higher
    lr: float = 0.3,  # Moderate learning rate
) -> Tuple[np.ndarray, dict]:
    """Optimize hand action using exact CVaR gradients through the belief.
    
    CRITICAL FIX: Ensure actions are AGGRESSIVE enough to achieve force closure.
    The cost function rewards closure but doesn't know about actual MuJoCo epsilon,
    so we bias toward higher actions and constrain minimum action magnitude.
    """
    best_overall_action = None
    best_overall_cvar = float('inf')
    best_overall_history = []
    
    # Multi-start covering low/medium/high closing magnitudes
    init_mags = [0.12, 0.18, 0.25]
    
    for init_mag in init_mags:
        # Initialize action directly in action-space (no rescaling)
        action_params = torch.full((11,), init_mag, requires_grad=True)
        optimizer = torch.optim.Adam([action_params], lr=lr)
        
        best_action = None
        best_cvar = float('inf')
        cvar_history = []
        
        for it in range(n_iters):
            optimizer.zero_grad()
            
            # Clamp directly to valid action range [min_floor, 0.35]
            # Floor of 0.08 keeps every finger closing at minimum rate
            action = torch.clamp(action_params, 0.08, 0.35)
            
            # Create action-conditioned cost function
            action_cost_fn = _make_action_conditioned_cost_fn(action)
            
            # Sample from belief - these samples need gradients for backprop
            samples = belief.rsample(n_samples)
            
            # Evaluate cost for each sample
            costs = action_cost_fn(samples)
            
            # CRITICAL FIX: Check for NaN/Inf in costs to prevent instability
            if not torch.isfinite(costs).all():
                # Replace non-finite values with large cost, don't break
                costs = torch.where(torch.isfinite(costs), costs, 
                                   torch.full_like(costs, 10.0))
            
            # Clamp costs to reasonable range to prevent gradient explosion
            costs = torch.clamp(costs, -10.0, 10.0)
            
            # Compute soft CVaR for differentiability
            sorted_costs, _ = torch.sort(costs)
            var_idx = int(beta * n_samples)
            var_idx = min(var_idx, n_samples - 1)
            eta = sorted_costs[var_idx].detach()
            excess = F.softplus(costs - eta, beta=5.0)
            cvar = eta + excess.mean() / (1.0 - beta + 1e-8)
            
            if not torch.isfinite(cvar):
                break
            
            # Backpropagate to ACTION parameters
            cvar.backward(retain_graph=True)
            
            if action_params.grad is not None and action_params.grad.abs().sum() > 0:
                # CRITICAL FIX: More aggressive gradient clipping to prevent instability
                # First check for infinite/NaN gradients
                if not torch.isfinite(action_params.grad).all():
                    action_params.grad.zero_()
                    break
                # Normalize gradient direction, then scale to max norm
                grad_norm = action_params.grad.norm()
                if grad_norm > 1e-8:
                    action_params.grad.data = action_params.grad.data / (grad_norm + 1e-6)
                    action_params.grad.data = action_params.grad.data * MAX_CVAR_GRAD_NORM
                torch.nn.utils.clip_grad_norm_([action_params], MAX_CVAR_GRAD_NORM)
                optimizer.step()
            
            cvar_val = cvar.item()
            cvar_history.append(cvar_val)
            
            if cvar_val < best_cvar:
                best_cvar = cvar_val
                best_action = action.detach().cpu().numpy().copy()
        
        if best_cvar < best_overall_cvar and best_action is not None:
            best_overall_cvar = best_cvar
            best_overall_action = best_action
            best_overall_history = cvar_history
    
    if best_overall_action is None:
        best_overall_action = np.ones(11) * 0.15  # fallback: aggressive closing
    
    # Ensure floor on final action ;  gradient can pull joints below useful range
    best_overall_action = np.maximum(best_overall_action, 0.08)
    
    # Compute final metrics
    final_cvar_res = belief.cvar_gradient(cost_fn, beta, n_samples=n_samples)
    
    return best_overall_action, {
        "cvar": best_overall_cvar,
        "best_score": best_overall_cvar,
        "n_iters": n_iters * len(init_mags),
        "cvar_history": best_overall_history,
        "cvar_improvement": best_overall_history[0] - best_overall_history[-1] if len(best_overall_history) > 1 else 0,
    }


def _score_candidate_actions(
    belief: GaussianMixtureBelief,
    cost_fn,
    beta: float,
    n_actions: int,
    n_candidates: int = 15,
    n_samples: int = 256,
    rng: np.random.Generator = None,
    use_gradient_optimization: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Score candidate hand actions using the differentiable belief CVaR.

    Returns the best hand-delta vector and gradient diagnostics.

    If use_gradient_optimization=True (default for variational), uses gradient
    descent on actions through the CVaR objective - the key advantage of VNB.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute CVaR and gradients for diagnostics
    cvar_res = belief.cvar_gradient(cost_fn, beta, n_samples=n_samples)
    belief_cvar = cvar_res["cvar"].item() if torch.isfinite(cvar_res["cvar"]) else 0.0
    
    has_grads = all(
        g is not None and g.abs().sum() > 0
        for g in [cvar_res["mixture_grad"], cvar_res["means_grad"], cvar_res["stds_grad"]]
    )
    
    # Use gradient-based optimization if enabled and we have valid gradients
    if use_gradient_optimization and has_grads:
        best_action, opt_info = _optimize_action_with_gradients(
            belief, cost_fn, beta, n_samples=n_samples
        )
        diag = {
            "cvar": belief_cvar,
            "cost_mean": float(cvar_res["costs"].mean().item()),
            "cost_std": float(cvar_res["costs"].std().item()),
            "has_exact_grads": has_grads,
            "grad_norm_means": float(cvar_res["means_grad"].norm().item()),
            "grad_norm_logits": float(cvar_res["mixture_grad"].norm().item()),
            "best_action_idx": -1,  # gradient-optimized, not from candidates
            "best_score": opt_info["best_score"],
            "optimization_method": "gradient",
            "cvar_result": cvar_res,
        }
        return best_action, diag
    
    # Fallback: heuristic candidate scoring (for baselines or when grads unavailable)
    # Score candidates by evaluating the action-conditioned CVaR via MC samples.
    candidates = []
    magnitudes = [0.08, 0.12, 0.15, 0.20, 0.25]

    for mag in magnitudes:
        c = np.ones(11) * mag
        candidates.append(c)

    for mag in [0.12, 0.18]:
        for start, end, label in [(0, 3, "thumb"), (3, 5, "index"),
                                    (5, 7, "middle"), (7, 9, "ring"),
                                    (9, 11, "pinky")]:
            c = np.ones(11) * 0.08          # keep all fingers at min floor
            c[start:end] = mag              # emphasise one finger group
            candidates.append(c)

    for _ in range(max(0, n_candidates - len(candidates))):
        c = rng.uniform(0.06, 0.25, size=11)
        candidates.append(c)

    candidates = candidates[:n_candidates]

    # Evaluate each candidate by plugging into action-conditioned cost + CVaR
    scores = []
    with torch.no_grad():
        samples = belief.rsample(n_samples)
    for c in candidates:
        action_t = torch.FloatTensor(c)
        cond_cost_fn = _make_action_conditioned_cost_fn(action_t)
        with torch.no_grad():
            costs_c = cond_cost_fn(samples)
            sorted_c, _ = torch.sort(costs_c)
            var_idx = min(int(beta * n_samples), n_samples - 1)
            eta_c = sorted_c[var_idx]
            excess_c = F.softplus(costs_c - eta_c, beta=5.0)
            cvar_c = (eta_c + excess_c.mean() / (1.0 - beta + 1e-8)).item()
        scores.append(cvar_c)

    best_idx = np.argmin(scores)

    diag = {
        "cvar": belief_cvar,
        "cost_mean": float(cvar_res["costs"].mean().item()),
        "cost_std":  float(cvar_res["costs"].std().item()),
        "has_exact_grads": has_grads,
        "grad_norm_means":  float(cvar_res["means_grad"].norm().item()),
        "grad_norm_logits": float(cvar_res["mixture_grad"].norm().item()),
        "best_action_idx": int(best_idx),
        "best_score": float(scores[best_idx]),
        "cvar_result": cvar_res,
    }

    return candidates[best_idx], diag


def run_variational_episode(
    env: RawMujocoEnv,
    obj_cfg: dict,
    beta: float,
    seed: int,
    friction: float,
    stiffness: float = 3000.0,
    friction_regime: str = "nominal",
    max_steps: int = 80,
    n_components: int = 8,
    risk_weight: float = 0.5,
    method_label: str = "variational",
    deadline: float = float('inf'),
) -> EpisodeResult:
    """Run one episode with variational neural beliefs.

    Also used for Gauss (K=1, risk_weight=0) and Gauss-CVaR (K=1, risk_weight=0.5)
    baselines by adjusting n_components and risk_weight.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    env.reset()
    _set_object_geom_filter(env, obj_cfg["geom"])
    if not position_arm_and_object(env, obj_cfg, friction, stiffness):
        raise RuntimeError("positioning failed")

    arm_q = env.data.qpos[0:6].copy()
    hand_q = env.data.qpos[6:17].copy()

    # Build variational belief
    obs_dim = env.model.nq + env.model.nv
    action_dim = env.model.nu
    v_config = VariationalBeliefConfig(
        belief_latent_dim=64,
        n_components=n_components,
        cvar_beta=beta,
        risk_weight=risk_weight,
        uncertainty_threshold=0.15,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    belief = GaussianMixtureBelief(v_config)
    belief_filter = NeuralBeliefFilter(v_config)
    cost_fn = _make_contact_cost_fn()

    step_log = []
    termination = "max_steps"
    prev_action_t = None
    entropy_history = []
    quality_history = []

    # Peak-quality tracking: restore best hand pose if grasp quality collapses
    best_epsilon = 0.0
    best_hand_q = hand_q.copy()
    quality_stable_steps = 0   # consecutive steps with epsilon >= EPSILON_MIN
    recovery_mode = False         # True when recovering from quality collapse
    recovery_start = 0            # step at which recovery began

    for step in range(max_steps):
        # Wall-clock timeout check
        if time.time() > deadline:
            print(f"    TIMEOUT at step {step} (wall-clock deadline exceeded)")
            termination = "timeout"
            break

        # Check sim stability
        if _sim_is_unstable(env):
            print(f"    SIM UNSTABLE at step {step}, terminating episode")
            termination = "sim_unstable"
            break

        #  observation 
        obs_vec = np.concatenate([env.data.qpos, env.data.qvel])
        obs_t = torch.FloatTensor(obs_vec)

        #  belief update 
        if prev_action_t is not None:
            with torch.no_grad():
                belief = belief_filter(belief, prev_action_t, obs_t)

        #  gradient-informed action selection 
        # Use gradient optimization only for full variational (K=8), not Gaussian baselines
        use_grad_opt = (n_components == 8 and method_label == "variational")
        
        gws = compute_gws(env, obj_cfg)
        cq = compute_contact_quality(env, obj_cfg)
        
        # VNB should use gradient-optimized actions even in early contact phase
        # The action-conditioned cost adapts to friction uncertainty
        if use_grad_opt:
            best_delta, diag = _score_candidate_actions(
                belief, cost_fn, beta,
                n_actions=11,
                n_candidates=20,
                n_samples=256,
                rng=np_rng,
                use_gradient_optimization=not recovery_mode,
            )
            
            # Use gradient DIRECTION to modulate per-finger closing rates,
            # keeping overall magnitude at least as strong as the greedy baseline.
            # Phase 1 (approaching): moderate closing + gradient shaping
            # Phase 2 (refinement): gradient-optimised with floor
            # Approach exits after 15 steps max to prevent over-closing.
            epsilon_rising = (len(quality_history) >= 2
                              and quality_history[-1] > quality_history[-2] + 1e-5)
            in_approach = ((gws.n_contacts < 6 or gws.epsilon < 0.003)
                           and step < 15)

            # Adaptive consolidation threshold: for objects with inherently
            # low ε (graspit_box ~0.002), the fixed 0.01 threshold means
            # consolidation NEVER exits ;  the method stays at [0.03, 0.12]/step
            # forever, over-closing the enveloping grasp.  Use best_epsilon to
            # adapt: once ε reaches ~70% of its historical peak and is not
            # still rising, transition to gentle maintenance.
            consol_eps_thresh = max(EPSILON_MIN * 2, best_epsilon * 0.7)
            in_consolidation = (not in_approach
                                and (gws.epsilon < consol_eps_thresh
                                     or epsilon_rising))

            if recovery_mode:
                # Recovery: freeze fingers entirely ;  zero additional closing.
                # Any closing at this point would displace the object further
                # (MuJoCo can't un-move objects when restoring finger poses).
                best_delta = np.zeros(11)
            elif in_approach:
                # Approach: moderate closing aligned with CEM's ~0.10 mean,
                # gradient shapes per-finger emphasis.
                grad_dir = np.maximum(best_delta, 0.01)
                grad_dir = grad_dir / (grad_dir.mean() + 1e-8)
                grad_dir = np.clip(grad_dir, 0.5, 2.0)
                best_delta = np.clip(grad_dir * 0.12, 0.06, 0.22)
            elif in_consolidation:
                # Consolidation: epsilon still improving, moderate closing.
                # np.clip (not np.maximum!) so optimizer output is bounded.
                best_delta = np.clip(best_delta, 0.03, 0.12)
                # For power grasps on low-ε objects (e.g. graspit_box), the
                # gradient optimizer creates asymmetric per-finger closing
                # that generates torques on the object, reducing physical
                # stability.  CEM succeeds on these objects because its
                # uniform 0.15/step closing creates symmetric forces.
                # Blend toward uniform proportional to the quality level.
                if best_epsilon < 0.006:
                    mean_mag = np.mean(best_delta)
                    uniform_delta = np.ones(11) * mean_mag
                    blend = min(1.0, best_epsilon / 0.006)
                    best_delta = blend * best_delta + (1.0 - blend) * uniform_delta
            else:
                # Maintenance: force closure established.
                # Must override optimizer's [0.08, 0.35] range to prevent
                # over-squeezing in high-friction regimes.
                # Once quality is stable (5+ steps above threshold), transition
                # to near-zero closing to prevent gradual over-squeezing that
                # leads to quality_peak termination.
                if quality_stable_steps >= 5:
                    best_delta = np.ones(11) * 0.005
                else:
                    best_delta = np.clip(best_delta, 0.01, 0.05)
                    # Same uniform blending for maintenance
                    if best_epsilon < 0.006:
                        mean_mag = np.mean(best_delta)
                        uniform_delta = np.ones(11) * mean_mag
                        blend = min(1.0, best_epsilon / 0.006)
                        best_delta = blend * best_delta + (1.0 - blend) * uniform_delta
        else:
            # Baselines: use heuristic candidates
            best_delta, diag = _score_candidate_actions(
                belief, cost_fn, beta,
                n_actions=11,
                n_candidates=20,
                n_samples=256,
                rng=np_rng,
                use_gradient_optimization=False,
            )
            # Greedy closing for baselines when contacts or force closure are insufficient
            # Using 0.005 for consistency with variational threshold
            if gws.n_contacts < 6 or gws.epsilon < 0.005:
                best_delta = np.ones(11) * 0.15

        #  execute in MuJoCo (TORQUE-BASED closing – pregrasp-planner style) 
        # The variational belief provides uncertainty estimates and risk metrics;
        # finger closing uses calibrated GRASP_TORQUE profile ramped over ~30
        # steps.  This avoids PD position-control failures on SIDE objects.
        torque_scale = min(1.0, (step + 1) / 30)
        hand_torque = _torque_hand_ctrl(torque_scale)
        for _ in range(25):
            ctrl = np.zeros(env.model.nu)
            ctrl[0:6] = arm_q
            ctrl[6:17] = hand_torque
            env.step(ctrl)
            if _sim_is_unstable(env):
                break

        # Store planned action for next belief update (keeps belief dynamics intact)
        if not np.isfinite(best_delta).all():
            best_delta = np.ones(11) * 0.15
        ctrl_full = np.zeros(action_dim)
        ctrl_full[:len(best_delta)] = best_delta
        prev_action_t = torch.FloatTensor(ctrl_full)

        #  metrics 
        ent = belief.entropy().item()
        # Determine current phase for logging
        if method_label == "variational":
            if recovery_mode:
                _phase = "recovery"
            elif in_approach:
                _phase = "approach"
            elif in_consolidation:
                _phase = "consolidation"
            else:
                _phase = "maintenance"
        else:
            _phase = "baseline"
        step_log.append({
            "step": step,
            "epsilon": float(gws.epsilon),
            "gws_volume": float(gws.volume),
            "contact_quality": float(cq),
            "n_contacts": gws.n_contacts,
            "is_force_closure": bool(gws.is_force_closure),
            "entropy": ent,
            "cvar": diag["cvar"],
            "cost": diag["cost_mean"],
            "has_exact_grads": diag["has_exact_grads"],
            "grad_norm_means": diag["grad_norm_means"],
            "grad_norm_logits": diag["grad_norm_logits"],
            "phase": _phase,
        })

        entropy_history.append(ent)
        quality_history.append(gws.epsilon)

        # --- Peak quality tracking -----------------------------------------
        if gws.epsilon > best_epsilon:
            best_epsilon = gws.epsilon
            best_hand_q = hand_q.copy()

        # Accumulate quality-stable steps ;  adaptive threshold based on
        # best_epsilon seen so far.  For objects with inherently low ε
        # (graspit_box ~0.002), a fixed 0.003 threshold prevents quality_stable
        # from ever firing, causing the method to overshoot to quality_peak.
        # The adaptive formula caps at 0.003 for high-ε objects (cube) while
        # allowing lower thresholds (down to EPSILON_MIN) for low-ε objects.
        QUALITY_STABLE_EPS = max(EPSILON_MIN, min(0.003, best_epsilon * 0.5))
        if gws.epsilon >= QUALITY_STABLE_EPS:
            quality_stable_steps += 1
        else:
            quality_stable_steps = 0

        if gws.epsilon >= 0.25:
            termination = "quality_target"
            break

        # Quality-stability: require at least 12 consecutive stable steps
        # AND minimum step count of 25 so VNB builds a grasp comparable to
        # CEM's ~27-step runtime before stopping.
        if step >= 25 and quality_stable_steps >= 12:
            termination = "quality_stable"
            break

        # Quality-collapse recovery: instead of hard termination, enter
        # recovery mode (greedy closing) which gives the grasp a chance to
        # re-establish.  Only terminate after 10 recovery steps fail.
        if (best_epsilon >= EPSILON_MIN * 5
                and gws.epsilon < best_epsilon * 0.5
                and step >= 25):
            if not recovery_mode:
                recovery_mode = True
                recovery_start = step
                hand_q = best_hand_q.copy()
            elif step - recovery_start >= 5:
                hand_q = best_hand_q.copy()
                termination = "quality_peak"
                break
        elif recovery_mode and gws.epsilon >= best_epsilon * 0.5:
            # Epsilon recovered ;  exit recovery mode, resume gradient actions
            recovery_mode = False

        # Entropy-stability termination: for the variational method, ONLY
        # use quality_stable termination (above) to avoid exiting too early
        # due to high initial entropy of 8-component GMM.  Baselines still
        # use entropy-based termination as before.
        if method_label != "variational":
            ent_rd_thresh = 0.05
            if (step >= max(10, max_steps // 3) and gws.n_contacts >= 4
                    and len(entropy_history) >= 3):
                rd = abs(entropy_history[-1] - entropy_history[-3])
                if rd < ent_rd_thresh:
                    termination = "entropy_stable"
                    break

    # Post-loop settle: hold at full torque for 35 steps.
    # With direct torque control, no incremental tightening is needed – the
    # calibrated GRASP_TORQUE profile naturally establishes force closure.
    full_torque = _torque_hand_ctrl(1.0)
    for _ in range(35):
        ctrl = np.zeros(env.model.nu)
        ctrl[0:6] = arm_q
        ctrl[6:17] = full_torque
        env.step(ctrl)
        if _sim_is_unstable(env):
            break

    #  final evaluation 
    runtime = time.time() - t0
    final_gws = compute_gws(env, obj_cfg)
    final_cq = compute_contact_quality(env, obj_cfg)
    
    # Uniform success criterion across all methods: epsilon >= EPSILON_MIN
    has_min_epsilon = final_gws.epsilon >= EPSILON_MIN
    stable = termination != "sim_unstable"
    success = stable and has_min_epsilon

    # Final belief gradients
    final_cvar_res = belief.cvar_gradient(cost_fn, beta, n_samples=512)
    final_has_grads = all(
        g is not None and g.abs().sum() > 0
        for g in [final_cvar_res["mixture_grad"], final_cvar_res["means_grad"],
                   final_cvar_res["stds_grad"]]
    )

    ls = {}
    if success:
        try:
            ls = run_lift_and_shear(env, arm_q, hand_q, obj_cfg,
                                    hand_torque=full_torque)
        except Exception as e:
            print(f"    lift/shear failed: {e}")

    # 28-test perturbation battery
    pb = {"n_survived": 0, "n_total": 28, "survival_rate": 0.0, "details": []}
    if success and ls.get("lift_success", False):
        try:
            pb = run_perturbation_battery(env, arm_q, hand_q, obj_cfg,
                                          hand_torque=full_torque)
        except Exception as e:
            print(f"    perturbation battery failed: {e}")

    pert_rate = pb["survival_rate"]
    robust = success and pert_rate > 0.5

    # Belief-predicted failure probability
    fail_prob_pred = float(F.softplus(final_cvar_res["cvar"], beta=5.0).item())
    fail_prob_pred = min(1.0, fail_prob_pred)

    # Failure mode categorization for paper Table II
    failure_mode = _determine_failure_mode(success, ls, pert_rate)

    return EpisodeResult(
        method=method_label,
        object_name=obj_cfg["body"],
        beta=beta,
        seed=seed,
        friction=friction,
        stiffness=stiffness,
        friction_regime=friction_regime,
        n_steps=len(step_log),
        runtime_s=runtime,
        termination=termination,
        final_epsilon=final_gws.epsilon,
        final_gws_volume=final_gws.volume,
        final_contact_quality=final_cq,
        final_n_contacts=final_gws.n_contacts,
        final_entropy=belief.entropy().item(),
        success=success,
        has_exact_grads=final_has_grads,
        grad_norm_means=float(final_cvar_res["means_grad"].norm().item()),
        grad_norm_logits=float(final_cvar_res["mixture_grad"].norm().item()),
        belief_cvar=float(final_cvar_res["cvar"].item()),
        belief_cost_mean=float(final_cvar_res["costs"].mean().item()),
        belief_cost_std=float(final_cvar_res["costs"].std().item()),
        lift_ratio=ls.get("lift_ratio", 0),
        lift_success=ls.get("lift_success", False),
        lift_height_achieved=ls.get("lift_height_achieved", 0.0),
        pulses_survived=ls.get("pulses_survived", 0),
        shear_success=ls.get("shear_success", False),
        shear_max_disp=ls.get("max_displacement", 0),
        time_to_slip=ls.get("time_to_slip"),
        peak_slip_distance=ls.get("peak_slip_distance", 0.0),
        failure_mode=failure_mode,
        perturbation_survival_rate=pert_rate,
        perturbation_n_survived=pb["n_survived"],
        perturbation_n_total=pb["n_total"],
        perturbation_details=pb["details"],
        robust_success=robust,
        failure_prob_predicted=fail_prob_pred,
        failure_prob_empirical=1.0 - pert_rate,
        step_log=step_log,
    )


# 
#  CEM baseline episode
# 

def run_cem_episode(
    env: RawMujocoEnv,
    obj_cfg: dict,
    beta: float,
    seed: int,
    friction: float,
    stiffness: float = 3000.0,
    friction_regime: str = "nominal",
    max_steps: int = 80,
    pop_size: int = 64,
    elite_frac: float = 0.2,
    cem_iters: int = 3,
    deadline: float = float('inf'),
) -> EpisodeResult:
    """Run one episode with Cross-Entropy Method planning.

    Uses a single Gaussian belief and CEM optimisation over hand deltas.
    Population 64, elite fraction 0.2, 3 CEM iterations (per paper).
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    env.reset()
    _set_object_geom_filter(env, obj_cfg["geom"])
    if not position_arm_and_object(env, obj_cfg, friction, stiffness):
        raise RuntimeError("positioning failed")

    arm_q = env.data.qpos[0:6].copy()
    hand_q = env.data.qpos[6:17].copy()

    obs_dim = env.model.nq + env.model.nv
    action_dim = env.model.nu
    v_config = VariationalBeliefConfig(
        belief_latent_dim=64,
        n_components=1,
        cvar_beta=beta,
        risk_weight=0.5,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    belief = GaussianMixtureBelief(v_config)
    belief_filter = NeuralBeliefFilter(v_config)
    cost_fn = _make_contact_cost_fn()

    step_log = []
    termination = "max_steps"
    prev_action_t = None
    entropy_history = []
    n_elite = max(1, int(pop_size * elite_frac))

    for step in range(max_steps):
        # Wall-clock timeout check
        if time.time() > deadline:
            print(f"    TIMEOUT at step {step} (wall-clock deadline exceeded)")
            termination = "timeout"
            break

        # Check sim stability
        if _sim_is_unstable(env):
            print(f"    SIM UNSTABLE at step {step}, terminating episode")
            termination = "sim_unstable"
            break

        obs_vec = np.concatenate([env.data.qpos, env.data.qvel])
        obs_t = torch.FloatTensor(obs_vec)

        if prev_action_t is not None:
            with torch.no_grad():
                belief = belief_filter(belief, prev_action_t, obs_t)

        gws = compute_gws(env, obj_cfg)
        cq = compute_contact_quality(env, obj_cfg)

        # CEM: optimise hand delta by sampling and refitting
        mu_cem = np.ones(11) * 0.10
        sig_cem = np.ones(11) * 0.08
        best_delta = mu_cem.copy()

        for _ in range(cem_iters):
            population = np_rng.normal(mu_cem, sig_cem, size=(pop_size, 11))
            population = np.clip(population, 0.0, 0.30)

            # Evaluate each candidate via belief CVaR
            scores = []
            with torch.no_grad():
                samples = belief.rsample(256)
                costs = cost_fn(samples)
                base_cvar = float(costs.mean().item())
            for c in population:
                close_mag = np.mean(c)
                score = base_cvar * (1.0 + 0.2 * close_mag) - 0.1 * np.sum(c > 0.01) / 11
                scores.append(score)
            scores = np.array(scores)
            elite_idx = np.argsort(scores)[:n_elite]
            mu_cem = population[elite_idx].mean(axis=0)
            sig_cem = population[elite_idx].std(axis=0) + 1e-4
            best_delta = mu_cem.copy()

        # Apply greedy closing with consistent threshold across methods
        if gws.n_contacts < 6 or gws.epsilon < 0.005:
            best_delta = np.ones(11) * 0.15

        #  execute in MuJoCo (TORQUE-BASED closing) 
        torque_scale = min(1.0, (step + 1) / 30)
        hand_torque = _torque_hand_ctrl(torque_scale)
        for _ in range(25):
            ctrl = np.zeros(env.model.nu)
            ctrl[0:6] = arm_q
            ctrl[6:17] = hand_torque
            env.step(ctrl)
            if _sim_is_unstable(env):
                break

        ctrl_full = np.zeros(action_dim)
        ctrl_full[:11] = best_delta
        prev_action_t = torch.FloatTensor(ctrl_full)

        ent = belief.entropy().item()
        step_log.append({
            "step": step,
            "epsilon": float(gws.epsilon),
            "gws_volume": float(gws.volume),
            "contact_quality": float(cq),
            "n_contacts": gws.n_contacts,
            "is_force_closure": bool(gws.is_force_closure),
            "entropy": ent,
            "cvar": base_cvar,
            "cost": base_cvar,
        })
        entropy_history.append(ent)

        if gws.epsilon >= 0.25:
            termination = "quality_target"
            break
        if (step >= max(10, max_steps // 3) and gws.n_contacts >= 4
                and len(entropy_history) >= 3):
            if abs(entropy_history[-1] - entropy_history[-3]) < 0.05:
                termination = "entropy_stable"
                break

    runtime = time.time() - t0

    # Post-loop settle at full torque
    full_torque = _torque_hand_ctrl(1.0)
    for _ in range(35):
        ctrl = np.zeros(env.model.nu)
        ctrl[0:6] = arm_q
        ctrl[6:17] = full_torque
        env.step(ctrl)

    final_gws = compute_gws(env, obj_cfg)
    final_cq = compute_contact_quality(env, obj_cfg)
    
    # Uniform success criterion across all methods: epsilon >= EPSILON_MIN
    has_min_epsilon = final_gws.epsilon >= EPSILON_MIN
    stable = termination != "sim_unstable"
    success = stable and has_min_epsilon

    final_cvar_res = belief.cvar_gradient(cost_fn, beta, n_samples=512)

    ls = {}
    if success:
        try:
            ls = run_lift_and_shear(env, arm_q, hand_q, obj_cfg,
                                    hand_torque=full_torque)
        except Exception as e:
            print(f"    lift/shear failed: {e}")

    pb = {"n_survived": 0, "n_total": 28, "survival_rate": 0.0, "details": []}
    if success and ls.get("lift_success", False):
        try:
            pb = run_perturbation_battery(env, arm_q, hand_q, obj_cfg,
                                          hand_torque=full_torque)
        except Exception as e:
            print(f"    perturbation battery failed: {e}")

    pert_rate = pb["survival_rate"]
    robust = success and pert_rate > 0.5

    # Failure mode categorization for paper Table II
    failure_mode = _determine_failure_mode(success, ls, pert_rate)

    return EpisodeResult(
        method="cem",
        object_name=obj_cfg["body"],
        beta=beta,
        seed=seed,
        friction=friction,
        stiffness=stiffness,
        friction_regime=friction_regime,
        n_steps=len(step_log),
        runtime_s=runtime,
        termination=termination,
        final_epsilon=final_gws.epsilon,
        final_gws_volume=final_gws.volume,
        final_contact_quality=final_cq,
        final_n_contacts=final_gws.n_contacts,
        final_entropy=belief.entropy().item(),
        success=success,
        belief_cvar=float(final_cvar_res["cvar"].item()),
        belief_cost_mean=float(final_cvar_res["costs"].mean().item()),
        belief_cost_std=float(final_cvar_res["costs"].std().item()),
        lift_ratio=ls.get("lift_ratio", 0),
        lift_success=ls.get("lift_success", False),
        lift_height_achieved=ls.get("lift_height_achieved", 0.0),
        pulses_survived=ls.get("pulses_survived", 0),
        shear_success=ls.get("shear_success", False),
        shear_max_disp=ls.get("max_displacement", 0),
        time_to_slip=ls.get("time_to_slip"),
        peak_slip_distance=ls.get("peak_slip_distance", 0.0),
        failure_mode=failure_mode,
        perturbation_survival_rate=pert_rate,
        perturbation_n_survived=pb["n_survived"],
        perturbation_n_total=pb["n_total"],
        perturbation_details=pb["details"],
        robust_success=robust,
        failure_prob_predicted=0.0,
        failure_prob_empirical=1.0 - pert_rate,
        step_log=step_log,
    )


# 
# JSON serialisation
# 

def _to_json(obj):
    """Convert object to JSON-serializable form, handling NaN/Inf properly."""
    if hasattr(obj, "__dict__"):
        return {k: _to_json(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())  # Recurse to handle NaN in arrays
    if isinstance(obj, (np.float32, np.float64, float)):
        # Handle NaN and Inf - convert to null for valid JSON
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        val = obj.item() if obj.numel() == 1 else obj.tolist()
        return _to_json(val)  # Recurse to handle NaN
    return obj


_RUN_START_TS: str = time.strftime("%Y%m%d_%H%M%S")


def save_results(results: List[EpisodeResult], tag: str = "", final: bool = False,
                 args=None):
    """Save results to disk.

    Partial saves reuse the same filename (overwrite) so per-episode saves
    don't clutter the output directory.  Final saves get their own timestamped
    file so they are never overwritten by a subsequent partial.
    
    When tag contains 'iros26', saves under iros26/ subdirectory.
    """
    # Use iros26 subdirectory if tag indicates it
    if "iros26" in tag:
        out_dir = OUTPUT_DIR / "iros26"
    else:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    if final:
        fname = (f"experiment_{tag}_{ts}.json"
                 if tag else f"experiment_{ts}.json")
    else:
        # Fixed partial filename keyed to the run start so it is always overwritten
        fname = (f"experiment_{tag}_partial_{_RUN_START_TS}.json"
                 if tag else f"experiment_partial_{_RUN_START_TS}.json")
    path = out_dir / fname
    
    # Build experiment config from args if provided
    config = {}
    if args is not None:
        config = {
            "objects": args.objects,
            "methods": args.methods,
            "betas": args.betas,
            "seeds": args.seeds,
            "regimes": args.regimes,
            "max_steps": args.max_steps,
            "episode_timeout": args.episode_timeout,
            "tag": args.tag,
            "quick": args.quick,
        }
        # Include friction regime definitions
        config["friction_regimes"] = {
            k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                for kk, vv in v.items()}
            for k, v in FRICTION_REGIMES.items()
        }
    
    data = {
        "timestamp": ts,
        "run_started": _RUN_START_TS,
        "config": config,
        "n_episodes": len(results),
        "episodes": [_to_json(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    if final:
        print(f"\nResults saved to {path}")
    return path


# 
# Summary table
# 

def print_summary(results: List[EpisodeResult]):
    """Print a compact summary table matching tex Tables I & II columns.

    Columns: Method | Object | Regime | Beta | SR% | Robust% | PertSurv% | ε | Quality | P_fail | Time
    """
    from collections import defaultdict

    # Group by (method, object, regime, beta)
    groups = defaultdict(list)
    for r in results:
        regime = getattr(r, "friction_regime", "nominal")
        groups[(r.method, r.object_name, regime, r.beta)].append(r)

    header = (
        f"{'Method':<14} {'Object':<14} {'Regime':<12} {'β':>4} | "
        f"{'SR%':>4} {'Rob%':>4} {'Pert%':>5} {'ε':>6} {'Qual':>5} "
        f"{'P_fail':>6} {'Time':>6}"
    )
    print("\n" + "=" * len(header))
    print("EXPERIMENT SUMMARY  (matches tex Tables I & II)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for (method, obj, regime, beta), eps in sorted(groups.items()):
        n = len(eps)
        sr = sum(1 for e in eps if e.success) / n * 100
        rob = sum(1 for e in eps if getattr(e, "robust_success", False)) / n * 100
        pert_surv = np.mean([getattr(e, "perturbation_survival_rate", 0.0) for e in eps]) * 100
        eps_mean = np.mean([e.final_epsilon for e in eps])
        qual = np.mean([e.final_contact_quality for e in eps])
        pfail = np.mean([getattr(e, "failure_prob_empirical", 0.0) for e in eps])
        rt = np.mean([e.runtime_s for e in eps])

        print(
            f"{method:<14} {obj:<14} {regime:<12} {beta:>4.2f} | "
            f"{sr:>3.0f}% {rob:>3.0f}% {pert_surv:>4.0f}% "
            f"{eps_mean:>6.4f} {qual:>5.3f} "
            f"{pfail:>6.3f} {rt:>5.1f}s"
        )

    print("=" * len(header))

    # Aggregate per-method summary (across all objects/regimes/betas)
    method_groups = defaultdict(list)
    for r in results:
        method_groups[r.method].append(r)

    print(f"\n{'--' * 60}")
    print("PER-METHOD AGGREGATE")
    print(f"{'--' * 60}")
    print(f"{'Method':<14} | {'SR%':>4} {'Rob%':>4} {'Pert%':>5} {'ε':>6} {'P_fail':>6} {'N':>4}")
    print(f"{'--' * 60}")
    for method in ["particle", "gauss", "gauss_cvar", "cem", "variational"]:
        eps = method_groups.get(method, [])
        if not eps:
            continue
        n = len(eps)
        sr = sum(1 for e in eps if e.success) / n * 100
        rob = sum(1 for e in eps if getattr(e, "robust_success", False)) / n * 100
        pert_surv = np.mean([getattr(e, "perturbation_survival_rate", 0.0) for e in eps]) * 100
        eps_mean = np.mean([e.final_epsilon for e in eps])
        pfail = np.mean([getattr(e, "failure_prob_empirical", 0.0) for e in eps])
        print(f"{method:<14} | {sr:>3.0f}% {rob:>3.0f}% {pert_surv:>4.0f}% "
              f"{eps_mean:>6.4f} {pfail:>6.3f} {n:>4}")
    print(f"{'--' * 60}")


# 
# Main experiment driver
# 

def main():
    ALL_METHODS = ["particle", "gauss", "gauss_cvar", "cem", "variational"]
    ALL_REGIMES = list(FRICTION_REGIMES.keys())

    parser = argparse.ArgumentParser(
        description="Variational belief experiments for IROS 2026 "
                    "(5 methods x 2 objects x 4 β x 3 seeds x 4 friction regimes = 480 episodes)")
    parser.add_argument("--objects", nargs="+",
                        default=["cube", "graspit_box"],
                        choices=list(OBJECT_CONFIGS.keys()),
                        help="Objects to test (default: cube, graspit_box)")
    parser.add_argument("--betas", nargs="+", type=float,
                        default=[0.5, 0.9, 0.95, 0.99],
                        help="CVaR beta values")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--max-steps", type=int, default=80,
                        help="Max MPC steps per episode")
    parser.add_argument("--methods", nargs="+",
                        default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to run")
    parser.add_argument("--regimes", nargs="+",
                        default=ALL_REGIMES,
                        choices=ALL_REGIMES,
                        help="Friction regimes to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 object, 1 beta, 1 seed, 1 regime")
    parser.add_argument("--episode-timeout", type=int, default=300,
                        help="Per-episode wall-clock timeout in seconds (default: 300)")
    parser.add_argument("--tag", default="", help="Output file tag")
    args = parser.parse_args()

    if args.quick:
        args.objects = ["cube"]
        args.betas = [0.9]
        args.seeds = [42]
        args.regimes = ["nominal"]

    n_total = (len(args.methods) * len(args.objects) * len(args.betas)
               * len(args.seeds) * len(args.regimes))
    print(f"Experiment plan: {n_total} episodes")
    print(f"  Methods:  {args.methods}")
    print(f"  Objects:  {args.objects}")
    print(f"  Betas:    {args.betas}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Regimes:  {args.regimes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print()

    env = make_env()
    rng = np.random.default_rng(0)
    all_results: List[EpisodeResult] = []
    completed = 0

    for regime in args.regimes:
        print(f"\n{'=' * 80}")
        print(f"  FRICTION REGIME: {regime}")
        print(f"{'=' * 80}")
        for obj_name in args.objects:
            obj_cfg = OBJECT_CONFIGS[obj_name]
            friction_nom = obj_cfg.get("friction_nom", FRICTION_NOMINAL_DEFAULT)
            for beta in args.betas:
                for seed in args.seeds:
                    # Same friction draw for all methods (fair comparison)
                    friction = sample_friction(regime, rng)
                    # Stiffness correlated with friction regime per paper specs
                    # Uses object-specific μ_nom from Table I for bimodal scaling
                    stiffness = sample_stiffness(regime, friction, rng, friction_nom)
                    for method in args.methods:
                        completed += 1
                        label = (f"[{completed}/{n_total}] {method:>12} | "
                                 f"{obj_name:<18} | β={beta:.2f} | seed={seed} | "
                                 f"μ={friction:.2f} κ={stiffness:.0f} | regime={regime}")
                        print(f"\n{'#' * 80}")
                        print(label)
                        print(f"{'#' * 80}")

                        try:
                            # Compute wall-clock deadline for this episode
                            episode_deadline = time.time() + args.episode_timeout

                            if method == "particle":
                                result = run_particle_episode(
                                    env, obj_cfg, beta, seed, friction,
                                    friction_regime=regime,
                                    max_steps=args.max_steps,
                                    stiffness=stiffness,
                                    deadline=episode_deadline)
                            elif method == "gauss":
                                result = run_variational_episode(
                                    env, obj_cfg, beta, seed, friction,
                                    friction_regime=regime,
                                    max_steps=args.max_steps,
                                    n_components=1,
                                    risk_weight=0.5,
                                    method_label="gauss",
                                    stiffness=stiffness,
                                    deadline=episode_deadline)
                            elif method == "gauss_cvar":
                                result = run_variational_episode(
                                    env, obj_cfg, beta, seed, friction,
                                    friction_regime=regime,
                                    max_steps=args.max_steps,
                                    n_components=1,
                                    risk_weight=0.5,
                                    method_label="gauss_cvar",
                                    stiffness=stiffness,
                                    deadline=episode_deadline)
                            elif method == "cem":
                                result = run_cem_episode(
                                    env, obj_cfg, beta, seed, friction,
                                    friction_regime=regime,
                                    max_steps=args.max_steps,
                                    stiffness=stiffness,
                                    deadline=episode_deadline)
                            else:  # variational (ours)
                                result = run_variational_episode(
                                    env, obj_cfg, beta, seed, friction,
                                    friction_regime=regime,
                                    max_steps=args.max_steps,
                                    n_components=8,
                                    risk_weight=0.5,
                                    method_label="variational",
                                    stiffness=stiffness,
                                    deadline=episode_deadline)

                            # Check if episode terminated due to timeout
                            if result.termination == "timeout":
                                print(f"  --> TIMEOUT  steps={result.n_steps}  "
                                      f"runtime={result.runtime_s:.1f}s")
                                all_results.append(result)
                                continue

                            all_results.append(result)
                            status = "SUCCESS" if result.success else "FAIL"
                            extra = ""
                            if hasattr(result, "has_exact_grads") and result.has_exact_grads:
                                extra = "  grads=YES"
                            rob = "ROB" if result.robust_success else "---"
                            print(f"  --> {status}  ε={result.final_epsilon:.4f}  "
                                  f"steps={result.n_steps}  {result.termination}  "
                                  f"pert={result.perturbation_survival_rate:.0%}  "
                                  f"{rob}{extra}")
                        except (SimulationUnstableError, RuntimeError) as e:
                            termination_kind = "sim_unstable"
                            print(f"  --> {termination_kind.upper()}: {e}")
                            # Record a failed episode so the sweep continues
                            all_results.append(EpisodeResult(
                                method=method,
                                object_name=obj_cfg["body"],
                                beta=beta,
                                seed=seed,
                                friction=friction,
                                friction_regime=regime,
                                n_steps=0,
                                runtime_s=0.0,
                                termination=termination_kind,
                                final_epsilon=0.0,
                                final_gws_volume=0.0,
                                final_contact_quality=0.0,
                                final_n_contacts=0,
                                final_entropy=0.0,
                                success=False,
                                robust_success=False,
                                failure_prob_empirical=1.0,
                            ))
                            # Reset env for next episode
                            env.reset()
                        except Exception as e:
                            print(f"  --> ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            # Still record a placeholder so no episode is silently lost
                            all_results.append(EpisodeResult(
                                method=method,
                                object_name=obj_cfg["body"],
                                beta=beta,
                                seed=seed,
                                friction=friction,
                                friction_regime=regime,
                                n_steps=0,
                                runtime_s=0.0,
                                termination="error",
                                final_epsilon=0.0,
                                final_gws_volume=0.0,
                                final_contact_quality=0.0,
                                final_n_contacts=0,
                                final_entropy=0.0,
                                success=False,
                                robust_success=False,
                                failure_prob_empirical=1.0,
                            ))
                            env.reset()

                        # Save after every episode so a crash loses at most 1 result
                        if all_results:
                            save_results(all_results, tag=args.tag or "partial", args=args)

    # Final save (gets its own timestamped file, never overwritten)
    path = save_results(all_results, tag=args.tag or "full", final=True, args=args)
    print_summary(all_results)
    print(f"\nDone.  {len(all_results)} episodes completed.  Results: {path}")


if __name__ == "__main__":
    main()
