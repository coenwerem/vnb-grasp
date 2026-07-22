"""Automatic per-object pregrasp planner.

Computes object-pose and geometry-tailored arm-hand pregrasp configurations
using GraspIt-informed hand preshapes and MuJoCo Jacobian-based IK for the
6-DOF arm.  Designed to replace hard-coded GRASP_ARM_CONFIG with a per-object
computed config suitable as the starting point for belief-MPC.

Pipeline for each object:
  1. Look up GraspIt grasp (best by epsilon metric) for hand DOF preshape.
  2. Determine grasp strategy (TOP_DOWN / SIDE_APPROACH / ENVELOPING) from
     the YCBObjectConfig registry.
  3. Compute the desired palm world pose via
     ``compute_strategy_specific_palm_pose()``.
  4. Convert palm pose to hand_base pose (subtract 0.07 m along palm_Z).
  5. Solve 6-DOF arm IK (damped least-squares targeting hand_base) from
     multiple random seeds.
  6. Return the full pregrasp: arm_q (6,), hand_q (11,), palm_pose, flags.

Author: Clinton Enwerem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

try:
    import mujoco as mj
except ImportError:
    mj = None  # type: ignore

from .ycb_objects import (
    YCB_OBJECTS,
    YCBObjectConfig,
    GraspStrategy,
    get_object_config,
    get_full_grasp_config,
    compute_strategy_specific_palm_pose,
    quat_to_rotmat,
)

# Offset from hand_base to palm_link along the kinematic chain (local Z).
HAND_BASE_TO_PALM_OFFSET = 0.07  # m

# Default table surface height
TABLE_Z = 0.777

# Map MuJoCo body names to YCB config short names.
BODY_TO_YCB_KEY: Dict[str, str] = {
    "cube": "cube",
    "graspit_box": "graspit_box",
    "graspit_cylinder": "graspit_cylinder",
    "005_tomato_soup_can": "soup",
    "006_mustard_bottle": "mustard",
    "010_potted_meat_can": "potted_meat",
    "056_tennis_ball": "tennis_ball",
}


@dataclass
class PregraspPlan:
    """Result of the pregrasp planner for one object."""

    object_name: str
    ycb_key: str
    grasp_strategy: str

    # Arm joint config (6 DOFs) from IK
    arm_q: np.ndarray               # (6,)
    # Hand DOF pregrasp (11 DOFs) from GraspIt
    hand_q: np.ndarray              # (11,)

    # Desired palm pose in world frame
    palm_world_pos: np.ndarray      # (3,)
    palm_world_rot: np.ndarray      # (3, 3)

    # Desired object placement (world frame)
    object_world_pos: np.ndarray    # (3,)
    object_world_quat: np.ndarray   # (4,) wxyz

    # Quality info
    ik_converged: bool = True
    ik_residual: float = 0.0
    graspit_epsilon: float = 0.0
    graspit_n_contacts: int = 0


# ---------------------------------------------------------------------------
#  MuJoCo Jacobian IK (arm-only, targeting hand_base)
# ---------------------------------------------------------------------------

def _compute_arm_ik(
    model,
    data,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    body_name: str = "hand_base",
    q0: np.ndarray = None,
    max_iter: int = 400,
    tol: float = 1e-3,
    damping: float = 0.01,
    alpha: float = 0.5,
) -> Tuple[Optional[np.ndarray], float]:
    """Damped least-squares IK for the 6-DOF arm targeting *body_name*.

    Returns (arm_q, residual).  arm_q is None when IK does not converge.
    """
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None, float("inf")

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    if q0 is not None:
        data.qpos[0:6] = q0
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    best_q = None
    best_err = float("inf")

    for iteration in range(max_iter):
        cur_pos = data.xpos[body_id].copy()
        cur_mat = data.xmat[body_id].reshape(3, 3).copy()

        pos_err = target_pos - cur_pos

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
            return best_q, best_err

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, body_id)

        J = np.vstack([w_pos * jacp[:, 0:6], w_rot * jacr[:, 0:6]])
        lam = damping * (1.0 + 0.1 * iteration / max_iter)
        dq = np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ err)
        data.qpos[0:6] += alpha * dq
        mj.mj_forward(model, data)

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mj.mj_forward(model, data)
    return (best_q if best_err < 0.05 else None), best_err


def _compute_arm_ik_multiseed(
    model,
    data,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    body_name: str = "hand_base",
    q0: np.ndarray = None,
    n_seeds: int = 12,
) -> Tuple[Optional[np.ndarray], float]:
    """Try IK from multiple seeds, return the best converged solution."""
    if q0 is None:
        q0 = np.zeros(6)

    best_q, best_res = _compute_arm_ik(
        model, data, target_pos, target_rot, body_name, q0=q0,
    )
    if best_q is not None and best_res < 1e-3:
        return best_q, best_res

    rng = np.random.default_rng(42)
    for _ in range(n_seeds):
        q_seed = q0 + rng.uniform(-0.6, 0.6, 6)
        sol, res = _compute_arm_ik(
            model, data, target_pos, target_rot, body_name,
            q0=q_seed, max_iter=300,
        )
        if sol is not None:
            if best_q is None or res < best_res:
                best_q, best_res = sol, res
            if res < 1e-3:
                return best_q, best_res

    return best_q, best_res


# ---------------------------------------------------------------------------
#  Core planner
# ---------------------------------------------------------------------------

def plan_pregrasp(
    model,
    data,
    body_name: str,
    object_world_pos: np.ndarray = None,
    object_world_quat: np.ndarray = None,
    grasp_index: int = 0,
    arm_seed: np.ndarray = None,
) -> PregraspPlan:
    """Compute a pregrasp arm-hand configuration for the given object.

    Args:
        model: MuJoCo model
        data:  MuJoCo data  (used for Jacobian IK; FK happens inside)
        body_name: MuJoCo body name of the object (e.g. "cube")
        object_world_pos: desired object position on table; if None use default
        object_world_quat: desired object orientation (wxyz); if None identity
        grasp_index: which GraspIt grasp to use (0 = best by epsilon)
        arm_seed: initial arm joint guess (6,)

    Returns:
        PregraspPlan with computed arm_q, hand_q, and placement info.
    """
    ycb_key = BODY_TO_YCB_KEY.get(body_name, body_name)
    try:
        ycb_cfg = get_object_config(ycb_key)
    except ValueError:
        # Unknown object -> default top-down with open hand
        ycb_cfg = None

    # Defaults
    if object_world_pos is None:
        object_world_pos = np.array([-0.05, 0.88, TABLE_Z])
    if object_world_quat is None:
        object_world_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- Hand DOFs from GraspIt ----
    hand_q = np.zeros(11)
    graspit_eps = 0.0
    graspit_nctc = 0
    if ycb_cfg is not None:
        try:
            gc = get_full_grasp_config(ycb_key, grasp_index=grasp_index, metric="epsilon")
            if gc is not None:
                hand_q = gc["hand_q"] * ycb_cfg.pregrasp_fraction
                graspit_eps = gc.get("epsilon_quality", 0.0)
                graspit_nctc = gc.get("n_contacts", 0)
        except Exception:
            pass  # fall back to open hand

    # ---- Palm approach pose ----
    strategy_name = "TOP_DOWN"
    if ycb_cfg is not None:
        strategy_name = ycb_cfg.grasp_strategy.value
        palm_pos, palm_rot = compute_strategy_specific_palm_pose(
            object_world_pos, object_world_quat, ycb_cfg,
        )
    else:
        palm_pos = object_world_pos.copy()
        palm_pos[2] += 0.08
        palm_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)

    # ---- Convert palm pose --> hand_base pose (IK target) ----
    # palm_link sits 0.07 m along local Z from hand_base.
    # hand_base_pos = palm_pos - 0.07 * palm_z_world
    palm_z_world = palm_rot[:, 2]
    hand_base_pos = palm_pos - HAND_BASE_TO_PALM_OFFSET * palm_z_world
    hand_base_rot = palm_rot  # same orientation (no rotation between them)

    # ---- Solve arm IK ----
    arm_q_sol, ik_res = _compute_arm_ik_multiseed(
        model, data, hand_base_pos, hand_base_rot,
        body_name="hand_base",
        q0=arm_seed,
        n_seeds=16,
    )

    ik_ok = arm_q_sol is not None
    if not ik_ok:
        # Return a fallback default arm config
        arm_q_sol = np.array([-0.826, -2.200, -1.643, -1.429, 0.5, 2.09])

    return PregraspPlan(
        object_name=body_name,
        ycb_key=ycb_key,
        grasp_strategy=strategy_name,
        arm_q=arm_q_sol,
        hand_q=hand_q,
        palm_world_pos=palm_pos,
        palm_world_rot=palm_rot,
        object_world_pos=object_world_pos,
        object_world_quat=object_world_quat,
        ik_converged=ik_ok,
        ik_residual=float(ik_res),
        graspit_epsilon=float(graspit_eps),
        graspit_n_contacts=int(graspit_nctc),
    )


def plan_all_pregrasps(
    model,
    data,
    object_table_pos: np.ndarray = None,
    objects: List[str] = None,
) -> Dict[str, PregraspPlan]:
    """Plan pregrasps for all (or specified) objects.

    Args:
        model, data: MuJoCo model & data
        object_table_pos: XY + table_z position (3,); default [-0.05, 0.88, 0.777]
        objects: list of MuJoCo body names to plan for; if None plans for all

    Returns:
        dict mapping body_name --> PregraspPlan
    """
    if objects is None:
        objects = list(BODY_TO_YCB_KEY.keys())
    if object_table_pos is None:
        object_table_pos = np.array([-0.05, 0.88, TABLE_Z])

    plans: Dict[str, PregraspPlan] = {}
    for body_name in objects:
        ycb_key = BODY_TO_YCB_KEY.get(body_name, body_name)
        try:
            ycb_cfg = get_object_config(ycb_key)
            table_half_h = ycb_cfg.half_height
        except ValueError:
            table_half_h = 0.025

        obj_pos = object_table_pos.copy()
        obj_pos[2] = TABLE_Z + table_half_h + 0.001

        plan = plan_pregrasp(model, data, body_name, object_world_pos=obj_pos)
        plans[body_name] = plan

        status = "OK" if plan.ik_converged else "IK-FAIL"
        print(
            f"  Pregrasp [{body_name:>22s}]  strategy={plan.grasp_strategy:<14s}  "
            f"IK={status}  residual={plan.ik_residual:.4f}  "
            f"ε_grasp={plan.graspit_epsilon:.3f}"
        )

    return plans
