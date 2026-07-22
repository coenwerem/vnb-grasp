#!/usr/bin/env python3
"""Collision-based closure grasp solver with thumb opposition.

Instead of IK-to-target-points, this solver:
1. Positions the hand at diverse approach poses around the object
2. Sets thumb opposition (cmc_yaw) FIRST; this is required for the
   thumb to face the other fingers and produce force closure
3. Incrementally curls finger flexion joints until MuJoCo detects contact
4. Evaluates GWS quality on actual MuJoCo contact points
5. Renders the best grasps

Key insight: the thumb_cmc_yaw joint (qa=14, axis=[0,0,-1]) rotates the
thumb across the palm.  Without setting it to ≥50% of its range, the
thumb points away from the fingers and never contacts the object on the
opposing side; killing force closure.

Author: Clinton Enwerem
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linprog

# Ensure project root on path  (file is at vnb_grasp/grasping/closure_grasp_solver.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mujoco


# ═══
#  LP-based Ferrari-Canny GWS analysis (correct multi-contact formulation)
# ═══

@dataclass
class ContactInfo:
    """Single MuJoCo contact."""
    geom1: int
    geom2: int
    pos: NDArray       # (3,) world position
    normal: NDArray    # (3,) contact normal (into the object)
    dist: float        # penetration depth (negative = penetrating)
    finger: str = ""   # which finger ("thumb", "index", …)


@dataclass
class GWSResult:
    """Result of wrench-space analysis."""
    epsilon: float            # Ferrari-Canny ε (LP-based, bounded forces)
    min_singular: float       # σ_min of the 6x3K grasp matrix
    is_force_closure: bool    # ε > 0
    n_contacts: int           # number of contact points used
    gravity_margin: float = 0.0     # max force-scale that can resist gravity
    directional_margins: Optional[NDArray] = None  # per-direction margins
    rank: int = 0             # numerical rank of grasp matrix

    def quality(self) -> float:
        if not self.is_force_closure:
            return 0.0
        return self.epsilon


def _build_wrench_generators(
    contacts: List[ContactInfo],
    obj_center: NDArray,
    mu: float = 0.5,
    n_edges: int = 16,
) -> Tuple[NDArray, int]:
    """Build the wrench-generator matrix for polyhedral friction cones.

    Returns
    -------
    W : (6, n_contacts * n_edges); each column is a *unit* primitive wrench
        w_{ik} = [f_{ik};  r_i x f_{ik}] where f_{ik} is an edge of the
        linearised Coulomb cone at contact i.
    n_contacts : number of contacts that contributed generators.

    These are *directions*; the LP decides the non-negative magnitudes \alpha _{ik}
    such that  W \alpha  = λ u  with per-contact force bounded.
    """
    angles = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    cols: List[NDArray] = []
    nc = 0
    for c in contacts:
        n = c.normal / (np.linalg.norm(c.normal) + 1e-15)
        # Build tangent frame
        if abs(n[2]) < 0.9:
            t1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
        else:
            t1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
        t1_n = np.linalg.norm(t1)
        if t1_n < 1e-12:
            continue
        t1 /= t1_n
        t2 = np.cross(n, t1)
        r = c.pos - obj_center
        for a in angles:
            fd = n + mu * (np.cos(a) * t1 + np.sin(a) * t2)
            fd /= np.linalg.norm(fd)
            cols.append(np.concatenate([fd, np.cross(r, fd)]))
        nc += 1
    if not cols:
        return np.zeros((6, 1)), 0
    return np.column_stack(cols), nc


def _lp_directional_margin(
    W: NDArray,
    u: NDArray,
    n_contacts: int,
    n_edges: int,
    f_max: float = 1.0,
) -> float:
    """Solve the LP:  max λ  s.t.  W \alpha  = λ u,  \alpha  ≥ 0,
       Σ_k \alpha _{ik} ≤ f_max  for each contact i.

    Returns λ* (0 if infeasible / origin outside in direction u).
    """
    # Decision variables:  x = [\alpha _1 … \alpha _{n_contacts*n_edges},  λ]
    n_vars = n_contacts * n_edges + 1
    # Objective: maximise λ  -->  minimise -λ
    c_obj = np.zeros(n_vars)
    c_obj[-1] = -1.0  # coefficient of λ

    # Equality constraint:  W \alpha   -  λ u  =  0
    A_eq = np.zeros((6, n_vars))
    A_eq[:, :n_contacts * n_edges] = W[:, :n_contacts * n_edges]
    A_eq[:, -1] = -u
    b_eq = np.zeros(6)

    # Inequality:  Σ_k \alpha _{ik} ≤ f_max   (one row per contact)
    A_ub = np.zeros((n_contacts, n_vars))
    for i in range(n_contacts):
        A_ub[i, i * n_edges:(i + 1) * n_edges] = 1.0
    b_ub = np.full(n_contacts, f_max)

    # Bounds:  \alpha  ≥ 0,  λ ≥ 0
    bounds = [(0, None)] * (n_contacts * n_edges) + [(0, None)]

    res = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
        options={"presolve": True, "time_limit": 0.05},
    )
    if res.success and res.x is not None:
        return float(res.x[-1])
    return 0.0


def _sample_unit_wrenches(n: int = 500, rng=None) -> NDArray:
    """Sample n unit vectors uniformly on S^5."""
    if rng is None:
        rng = np.random.default_rng(0)
    raw = rng.standard_normal((n, 6))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-15)


# Canonical task directions (gravity, pushes, torques); 18 total
_TASK_DIRS: Optional[NDArray] = None


def _get_task_directions() -> NDArray:
    """Return (18, 6) matrix of canonical task wrenches (unit)."""
    global _TASK_DIRS
    if _TASK_DIRS is not None:
        return _TASK_DIRS
    dirs = []
    # ±forces along x, y, z
    for ax in range(3):
        for sign in (+1, -1):
            w = np.zeros(6)
            w[ax] = sign
            dirs.append(w)
    # ±torques about x, y, z
    for ax in range(3):
        for sign in (+1, -1):
            w = np.zeros(6)
            w[3 + ax] = sign
            dirs.append(w)
    _TASK_DIRS = np.array(dirs)
    return _TASK_DIRS


def _analyze_gws_fast(
    contacts: List[ContactInfo],
    obj_center: NDArray,
    mu: float = 0.5,
    f_max: float = 1.0,
    n_edges: int = 16,
    object_mass_kg: float = 0.05,
) -> GWSResult:
    """Fast pre-screen: SVD rank check + gravity + 18 task-direction LPs.

    Used in the inner curl loop.  ~19 LPs instead of 218.
    Returns conservative (lower-bound) ε.
    """
    if len(contacts) < 2:
        return GWSResult(0, 0, False, len(contacts))

    W, nc = _build_wrench_generators(contacts, obj_center, mu, n_edges)
    if nc < 2 or W.shape[1] < 7:
        return GWSResult(0, 0, False, len(contacts))

    _, s, _ = np.linalg.svd(W, full_matrices=False)
    msv = float(s[-1]) if len(s) else 0.0
    rank = int(np.sum(s > 1e-6 * s[0]))
    if rank < 6:
        return GWSResult(0, msv, False, len(contacts), rank=rank)

    # Gravity margin
    w_grav = np.zeros(6); w_grav[2] = object_mass_kg * 9.81
    grav_margin = _lp_directional_margin(
        W, w_grav / np.linalg.norm(w_grav), nc, n_edges, f_max,
    )

    # Task directions only (18 LPs); no random directions
    task_dirs = _get_task_directions()
    margins = np.zeros(len(task_dirs))
    for j, u in enumerate(task_dirs):
        margins[j] = _lp_directional_margin(W, u, nc, n_edges, f_max)

    eps = float(margins.min())
    is_fc = eps > 1e-8
    return GWSResult(
        epsilon=eps, min_singular=msv, is_force_closure=is_fc,
        n_contacts=len(contacts), gravity_margin=grav_margin,
        directional_margins=margins, rank=rank,
    )


def _analyze_gws_full(
    contacts: List[ContactInfo],
    obj_center: NDArray,
    mu: float = 0.5,
    f_max: float = 1.0,
    n_edges: int = 16,
    n_random_dirs: int = 500,
    object_mass_kg: float = 0.05,
) -> GWSResult:
    """Full LP-based Ferrari-Canny ε with bounded per-contact forces.

    Used only for re-scoring the top candidates after the search.
    Evaluates 18 task + n_random_dirs random directions.
    """
    if len(contacts) < 2:
        return GWSResult(0, 0, False, len(contacts))

    W, nc = _build_wrench_generators(contacts, obj_center, mu, n_edges)
    if nc < 2 or W.shape[1] < 7:
        return GWSResult(0, 0, False, len(contacts))

    _, s, _ = np.linalg.svd(W, full_matrices=False)
    msv = float(s[-1]) if len(s) else 0.0
    rank = int(np.sum(s > 1e-6 * s[0]))
    if rank < 6:
        return GWSResult(0, msv, False, len(contacts), rank=rank)

    # Gravity margin
    w_grav = np.zeros(6); w_grav[2] = object_mass_kg * 9.81
    grav_margin = _lp_directional_margin(
        W, w_grav / np.linalg.norm(w_grav), nc, n_edges, f_max,
    )

    # Task + random directions
    task_dirs = _get_task_directions()                       # (18, 6)
    rng = np.random.default_rng(7)
    rand_dirs = _sample_unit_wrenches(n_random_dirs, rng)    # (500, 6)
    all_dirs = np.vstack([task_dirs, rand_dirs])

    margins = np.zeros(len(all_dirs))
    for j, u in enumerate(all_dirs):
        margins[j] = _lp_directional_margin(W, u, nc, n_edges, f_max)

    eps = float(margins.min())
    is_fc = eps > 1e-8
    return GWSResult(
        epsilon=eps, min_singular=msv, is_force_closure=is_fc,
        n_contacts=len(contacts), gravity_margin=grav_margin,
        directional_margins=margins, rank=rank,
    )


#  Model introspection 

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Populated by init_model_info()
_FINGER_FLEX_QA: Dict[str, List[int]] = {}   # flexion joints only
_THUMB_YAW_QA: Optional[int] = None          # separate: opposition joint
_JOINT_RANGE: Dict[int, Tuple[float, float]] = {}
_CUBE_GIDS: Set[int] = set()
_DISTAL_GIDS: Dict[str, Set[int]] = {}
_FLEX_AI: List[int] = []              # actuator indices for flexion joints
_YAW_AI: Optional[int] = None        # actuator index for thumb yaw
_AI_TO_QA: Dict[int, int] = {}       # actuator index --> qpos address
_QA_TO_VA: Dict[int, int] = {}       # qpos address --> dof velocity address


def init_model_info(model):
    global _THUMB_YAW_QA, _YAW_AI
    _FINGER_FLEX_QA.clear(); _JOINT_RANGE.clear()
    _CUBE_GIDS.clear(); _DISTAL_GIDS.clear()
    _FLEX_AI.clear(); _AI_TO_QA.clear(); _QA_TO_VA.clear()
    _THUMB_YAW_QA = None
    _YAW_AI = None

    for ji in range(model.njnt):
        if model.jnt_type[ji] != 3:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji) or ""
        qa = int(model.jnt_qposadr[ji])
        va = int(model.jnt_dofadr[ji])
        _JOINT_RANGE[qa] = (float(model.jnt_range[ji, 0]),
                            float(model.jnt_range[ji, 1]))
        _QA_TO_VA[qa] = va

        # Thumb CMC yaw is the opposition joint; NOT a flexion joint
        if "thumb_cmc_yaw" in jname:
            _THUMB_YAW_QA = qa
            continue

        for fn in FINGER_NAMES:
            if fn in jname:
                _FINGER_FLEX_QA.setdefault(fn, []).append(qa)
                break

    for gi in range(model.ngeom):
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if "cube" in gn:
            _CUBE_GIDS.add(gi)
        for fn in FINGER_NAMES:
            if fn in gn and "distal" in gn and "collision" in gn:
                _DISTAL_GIDS.setdefault(fn, set()).add(gi)

    # Actuator mapping
    for ai in range(model.nu):
        jid = int(model.actuator_trnid[ai, 0])
        qa = int(model.jnt_qposadr[jid])
        _AI_TO_QA[ai] = qa
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if "cmc_yaw" in aname:
            _YAW_AI = ai
        else:
            _FLEX_AI.append(ai)


#  Approach-pose generation 

# Known-good seed configs (from 83K-config grid search):
_SEED_CONFIGS = [
    # (offset_xyz, euler_XYZ_deg) ; all produce 5-finger distal contact
    ((0.00, -0.08, -0.04), (135, 180, 270)),
    ((0.00, -0.08, -0.04), (315,   0,  90)),
    ((0.00, -0.08,  0.04), ( 45, 180,  90)),
    ((0.00, -0.08,  0.04), (225,   0, 270)),
    ((0.00, -0.04,  0.08), ( 45, 180, 270)),
    ((0.00, -0.04,  0.08), (225,   0,  90)),
    ((0.00,  0.04,  0.08), (135,   0, 270)),
    ((0.00,  0.04,  0.08), (315, 180,  90)),
    ((0.00,  0.08, -0.04), ( 45,   0, 270)),
    ((0.00,  0.08, -0.04), (225, 180,  90)),
    ((0.00,  0.08,  0.04), (135,   0,  90)),
    ((0.00,  0.08,  0.04), (315, 180, 270)),
    ((-0.08, -0.08, 0.00), (  0,  90,  90)),
    ((-0.08, -0.08, 0.00), ( 90,  90,   0)),
    ((-0.08, -0.08, 0.00), (270,   0,   0)),
    ((-0.08, -0.08, 0.08), ( 45, 180, 180)),
    ((-0.08, -0.08, 0.08), (225,   0,   0)),
]


def _euler_to_quat_wxyz(euler_deg):
    q = R.from_euler("XYZ", euler_deg, degrees=True).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def generate_approach_poses(cube_pos, n_poses=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    poses = []

    # Seed configs (exact)
    for off, eul in _SEED_CONFIGS:
        poses.append((cube_pos + np.array(off), _euler_to_quat_wxyz(eul)))

    # Perturbations around each seed
    for off, eul in _SEED_CONFIGS:
        for _ in range(20):
            dp = rng.normal(0, 0.006, 3)
            de = rng.normal(0, 8, 3)
            poses.append((
                cube_pos + np.array(off) + dp,
                _euler_to_quat_wxyz(np.array(eul) + de),
            ))

    # Fill remainder with random spherical approaches
    while len(poses) < n_poses:
        th = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(-np.pi / 3, np.pi / 3)
        r  = rng.uniform(0.04, 0.10)
        dp = r * np.array([np.cos(ph)*np.cos(th),
                           np.cos(ph)*np.sin(th),
                           np.sin(ph)])
        quat = _euler_to_quat_wxyz(rng.uniform(0, 360, 3))
        poses.append((cube_pos + dp, quat))

    return poses[:n_poses]


#  Incremental closure with thumb opposition 

@dataclass
class ClosureResult:
    qpos: NDArray
    contacts: List[ContactInfo]
    contact_fingers: Set[str]
    gws: GWSResult
    has_palm_collision: bool
    has_proximal_penetration: bool
    approach_idx: int = 0
    thumb_yaw: float = 0.0


def _extract_all_contacts(model, data):
    """Extract ALL cube contacts for wrench analysis.

    Returns (contacts, distal_fingers, has_deep_palm, has_deep_prox).

    Changes from previous _extract_cube_contacts:
    - Keeps ALL valid distal contacts (not just one per finger).
    - Includes palm/hand-base contacts in wrench analysis.
    - Relaxed depth threshold: 3 mm (physics settle gives realistic depths).
    """
    contacts: List[ContactInfo] = []
    distal_fingers: Set[str] = set()
    has_deep_palm = False
    has_deep_prox = False

    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in _CUBE_GIDS and g2 not in _CUBE_GIDS:
            continue

        other = g1 if g2 in _CUBE_GIDS else g2
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        gn_lower = gn.lower()

        is_palm = "palm" in gn_lower or "hand_base" in gn_lower
        is_proximal = "proximal" in gn_lower and "distal" not in gn_lower
        is_distal = "distal" in gn_lower

        # Flag deep penetrations
        if is_palm and c.dist < -0.003:
            has_deep_palm = True
        if is_proximal and c.dist < -0.003:
            has_deep_prox = True

        # Skip very deep contacts (severe physics artifacts)
        if c.dist < -0.003:
            continue

        # Accept distal AND palm contacts for wrench analysis
        if not (is_distal or is_palm):
            continue

        # Identify finger
        finger = ""
        if is_distal:
            for fn in FINGER_NAMES:
                if fn in gn_lower:
                    finger = fn
                    distal_fingers.add(fn)
                    break
        elif is_palm:
            finger = "palm"

        if not finger:
            continue

        # Contact normal: frame[:3]
        normal = c.frame[:3].copy()
        # MuJoCo: normal from geom2 --> geom1.  We want into-object.
        if g1 in _CUBE_GIDS:
            pass          # other --> cube: correct
        else:
            normal = -normal  # cube --> other: flip

        contacts.append(ContactInfo(
            geom1=g1, geom2=g2,
            pos=c.pos[:3].copy(),
            normal=normal,
            dist=float(c.dist),
            finger=finger,
        ))

    return contacts, distal_fingers, has_deep_palm, has_deep_prox


def physical_closure(
    model, data, hand_pos, hand_quat, cube_pos,
    thumb_yaw_frac=0.7, approach_idx=0,
    n_steps=250, kp=8.0,
):
    """Close fingers via physics simulation: proper contact conforming.

    Uses mj_step with a P controller to drive every finger toward full
    flexion.  MuJoCo's constraint solver prevents penetration, so
    fingers wrap around the cube naturally.  The hand base and cube are
    pinned in place each substep.

    The caller must adjust model.opt.gravity, model.opt.timestep, and
    model.dof_damping (for finger DOFs) BEFORE calling this function
    and restore them afterwards.  This avoids per-call save/restore
    overhead across 1 800 trials.
    """
    mujoco.mj_resetData(model, data)

    # Place cube (pinned throughout)
    data.qpos[0:3] = cube_pos
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0

    # Place hand base (pinned throughout)
    data.qpos[7:10] = hand_pos
    data.qpos[10:14] = hand_quat

    # Thumb opposition: set BEFORE closure
    yaw_target = 0.0
    if _THUMB_YAW_QA is not None:
        lo, hi = _JOINT_RANGE[_THUMB_YAW_QA]
        yaw_target = lo + thumb_yaw_frac * (hi - lo)
        data.qpos[_THUMB_YAW_QA] = yaw_target

    # Flexion joints: start at minimum, target = 95 % of range
    flex_targets: Dict[int, float] = {}
    for fn in FINGER_NAMES:
        for qa in _FINGER_FLEX_QA.get(fn, []):
            lo, hi = _JOINT_RANGE[qa]
            flex_targets[qa] = lo + 0.95 * (hi - lo)
            data.qpos[qa] = lo

    data.qvel[:] = 0

    # Cached pinned poses
    cube_qpos = np.array([cube_pos[0], cube_pos[1], cube_pos[2],
                          1.0, 0.0, 0.0, 0.0])
    hand_qpos = np.concatenate([hand_pos, hand_quat])

    #  Physics loop: P-controlled finger closure 
    # Evaluate at checkpoints to capture intermediate contact states
    # (e.g., thumb may contact briefly before sliding off an edge)
    check_interval = max(1, n_steps // 5)
    checkpoints = set(range(check_interval - 1, n_steps, check_interval))
    checkpoints.add(n_steps - 1)  # always check at the end
    best = None

    for step in range(n_steps):
        for ai, qa in _AI_TO_QA.items():
            if qa == _THUMB_YAW_QA:
                target = yaw_target
            elif qa in flex_targets:
                target = flex_targets[qa]
            else:
                data.ctrl[ai] = 0.0
                continue
            err = target - data.qpos[qa]
            data.ctrl[ai] = np.clip(kp * err, -0.2, 0.2)

        mujoco.mj_step(model, data)

        # Pin cube + hand base every step
        data.qvel[:6] = 0       # cube free-joint
        data.qvel[6:12] = 0     # hand free-joint
        data.qpos[:7] = cube_qpos
        data.qpos[7:14] = hand_qpos

        # Checkpoint evaluation
        if step in checkpoints:
            mujoco.mj_forward(model, data)
            contacts, distal, deep_palm, prox = _extract_all_contacts(
                model, data)

            if len(distal) < 2:
                continue

            # Coplanarity filter
            if len(contacts) >= 2:
                n0 = contacts[0].normal
                if all(abs(np.dot(n0, c.normal)) > 0.85
                       for c in contacts[1:]):
                    continue

            gws = _analyze_gws_fast(
                contacts, cube_pos, mu=0.8, f_max=1.0, n_edges=16)

            if best is None or gws.epsilon > best.gws.epsilon:
                best = ClosureResult(
                    qpos=data.qpos.copy(),
                    contacts=contacts,
                    contact_fingers=distal.copy(),
                    gws=gws,
                    has_palm_collision=deep_palm,
                    has_proximal_penetration=prox,
                    approach_idx=approach_idx,
                    thumb_yaw=thumb_yaw_frac,
                )

    return best


#  Main solver 

def solve_closure_grasps(
    scene_xml, n_approaches=600, top_k=10, seed=42, verbose=True,
):
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    init_model_info(model)

    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_bid].copy()

    if verbose:
        print(f"Cube position      : {cube_pos}")
        print(f"Thumb yaw qa       : {_THUMB_YAW_QA}")
        print(f"Finger flexion qa  : {_FINGER_FLEX_QA}")
        print(f"Actuators (flex)   : {_FLEX_AI}")
        print(f"Actuator (yaw)     : {_YAW_AI}")

    rng = np.random.default_rng(seed)
    poses = generate_approach_poses(cube_pos, n_approaches, rng)

    #  Prepare model for fast physics closure 
    saved_grav = model.opt.gravity.copy()
    saved_dt   = model.opt.timestep
    saved_damp = model.dof_damping.copy()

    model.opt.gravity[:] = 0       # no gravity during closure
    model.opt.timestep = 0.005     # larger dt for speed
    # Reduce finger damping so P controller can close in ~250 steps
    for qa, va in _QA_TO_VA.items():
        if qa != _THUMB_YAW_QA:
            model.dof_damping[va] = 0.3

    results = []
    n_valid = n_fc = 0
    t0 = time.time()

    yaw_fracs = [0.5, 0.7, 0.9]

    try:
        for i, (pos, quat) in enumerate(poses):
            for yf in yaw_fracs:
                res = physical_closure(
                    model, data, pos, quat, cube_pos,
                    thumb_yaw_frac=yf, approach_idx=i,
                    n_steps=250, kp=8.0,
                )
                if res is not None and len(res.contact_fingers) >= 2:
                    results.append(res)
                    n_valid += 1
                    if res.gws.is_force_closure:
                        n_fc += 1

            if verbose and (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(poses)}] valid={n_valid} fc={n_fc} "
                      f"({(i+1)/dt:.0f} poses/s)")
    finally:
        # Always restore model parameters
        model.opt.gravity[:] = saved_grav
        model.opt.timestep = saved_dt
        model.dof_damping[:] = saved_damp

    dt = time.time() - t0
    if verbose:
        n_thumb = sum(1 for r in results if "thumb" in r.contact_fingers)
        print(f"\nDone: {len(poses)} poses x {len(yaw_fracs)} yaw = "
              f"{len(poses)*len(yaw_fracs)} trials in {dt:.1f}s")
        print(f"  valid={n_valid}  force-closure={n_fc}  with-thumb={n_thumb}")

    # Sort by fast ε
    results.sort(key=lambda r: (
        r.gws.is_force_closure,
        r.gws.epsilon,
        r.gws.gravity_margin,
        len(r.contact_fingers),
    ), reverse=True)

    # Re-score top candidates with full LP evaluation (500 random dirs)
    candidates = results[:top_k * 3]
    if verbose:
        print(f"\nRe-scoring top {len(candidates)} candidates with full LP ...")
    t_rescore = time.time()
    for r in candidates:
        r.gws = _analyze_gws_full(
            r.contacts, cube_pos, mu=0.8, f_max=1.0,
            n_edges=16, n_random_dirs=500,
        )
    if verbose:
        print(f"  Re-scoring done in {time.time() - t_rescore:.1f}s")

    candidates.sort(key=lambda r: (
        r.gws.is_force_closure,
        r.gws.epsilon,
        r.gws.gravity_margin,
        len(r.contact_fingers),
    ), reverse=True)

    return candidates[:top_k]


#  Lightweight inline renderer; avoids heavy import chain that segfaults 

def _prepare_model_for_render(model):
    """Hide collision geoms, floor, table, markers; keep only visual meshes."""
    hide_patterns = ["floor", "table", "ground", "plane", "marker"]
    for i in range(model.ngeom):
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        bid = model.geom_bodyid[i]
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        combined = (gn + " " + bn).lower()

        # Hide floor / table / marker
        if any(p in combined for p in hide_patterns):
            model.geom_group[i] = 4
            continue

        # Hide ALL collision primitives (except cube_collision; that's the object)
        if "collision" in gn.lower() and "cube" not in gn.lower():
            model.geom_group[i] = 4

    # Uniform headlight; no harsh shadows
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]


def _make_camera(model, data, preset, lookat=None):
    """Return an MjvCamera positioned at *preset* looking at *lookat*."""
    cam = mujoco.MjvCamera()
    if lookat is None:
        # Auto-detect: centroid of hand + object bodies
        pts = []
        for i in range(1, model.nbody):
            bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if any(k in bn.lower() for k in
                   ("thumb", "index", "middle", "ring", "pinky",
                    "palm", "cube", "hand")):
                pts.append(data.xpos[i].copy())
        if pts:
            lookat = np.mean(pts, axis=0)
        else:
            lookat = np.array([0.0, 1.2, 0.85])

    cam.lookat[:] = lookat
    presets = {
        "front": (0.40, -20.0, 90.0),
        "side":  (0.40, -15.0, 0.0),
        "iso":   (0.45, -30.0, 135.0),
        "top":   (0.45, -89.0, 90.0),
    }
    dist, elev, azim = presets.get(preset, presets["iso"])
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim
    return cam


def render_grasps(scene_xml, results, output_dir, max_render=8):
    """Render top grasps using only mujoco.Renderer; no heavy imports."""
    os.makedirs(output_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    init_model_info(model)
    _prepare_model_for_render(model)

    W, H = 1280, 960
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H

    # Vis options; no labels, no yellow contacts
    opt = mujoco.MjvOption()
    opt.label = mujoco.mjtLabel.mjLABEL_NONE
    for flag in (mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
                 mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
                 mujoco.mjtVisFlag.mjVIS_CONVEXHULL,
                 mujoco.mjtVisFlag.mjVIS_JOINT,
                 mujoco.mjtVisFlag.mjVIS_ACTUATOR,
                 mujoco.mjtVisFlag.mjVIS_COM,
                 mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
                 mujoco.mjtVisFlag.mjVIS_ACTIVATION,
                 mujoco.mjtVisFlag.mjVIS_SELECT):
        opt.flags[flag] = False

    # Single renderer instance; never close / recreate
    renderer = mujoco.Renderer(model, height=H, width=W)

    for idx, res in enumerate(results[:max_render]):
        # Apply grasp qpos
        data.qpos[:model.nq] = res.qpos[:model.nq]
        mujoco.mj_forward(model, data)

        fc  = "FC" if res.gws.is_force_closure else "noFC"
        eps = f"eps{res.gws.epsilon:.4f}_grav{res.gws.gravity_margin:.3f}"
        nf  = len(res.contact_fingers)
        tag = f"grasp_{idx:02d}_{fc}_{eps}_{nf}f_yaw{res.thumb_yaw:.1f}"

        lookat = None  # auto-detect once per grasp

        views = ["front", "side", "iso", "top"]
        view_imgs = {}
        for view in views:
            cam = _make_camera(model, data, view, lookat)
            if lookat is None:
                lookat = cam.lookat.copy()

            renderer.update_scene(data, camera=cam, scene_option=opt)
            # Clean render flags
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

            rgb = renderer.render().copy()

            # White background via depth masking
            try:
                renderer.enable_depth_rendering()
                depth = renderer.render()
                renderer.disable_depth_rendering()
                is_bg = depth >= (depth.max() - 1e-6)
                rgb[is_bg] = 255
            except Exception:
                if np.all(rgb[0, 0] == 0):
                    rgb[np.all(rgb == 0, axis=2)] = 255

            view_imgs[view] = rgb

            path = os.path.join(output_dir, f"{tag}_{view}.png")
            try:
                from PIL import Image
                Image.fromarray(rgb).save(path)
            except ImportError:
                import cv2
                cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Multi-view composite (front + side + iso)
        combo_views = [view_imgs[v] for v in ("front", "side", "iso")]
        pad = 4
        total_w = 3 * W + 2 * pad
        combined = np.full((H, total_w, 3), 255, dtype=np.uint8)
        for i, img in enumerate(combo_views):
            x = i * (W + pad)
            combined[:, x:x + W] = img
        multi_path = os.path.join(output_dir, f"{tag}_multiview.png")
        try:
            from PIL import Image
            Image.fromarray(combined).save(multi_path)
        except ImportError:
            import cv2
            cv2.imwrite(multi_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        print(f"  #{idx}: {fc} ε={res.gws.epsilon:.5f} "
              f"grav={res.gws.gravity_margin:.4f} "
              f"fingers={sorted(res.contact_fingers)} "
              f"yaw={res.thumb_yaw:.1f} palm={res.has_palm_collision} "
              f"-> {tag}")


#  Entry point 

def main():
    scene_xml = os.path.join(_PROJECT_ROOT,
                             "arenas", "hand_object_testbed", "scene.xml")
    output_dir = os.path.join(_PROJECT_ROOT, "outputs", "closure_grasps")

    print("=" * 60)
    print("CLOSURE-BASED GRASP SOLVER  (with thumb opposition)")
    print("=" * 60)

    results = solve_closure_grasps(scene_xml, n_approaches=600, top_k=10,
                                   seed=42, verbose=True)
    if not results:
        print("\nNo valid grasps found!")
        return

    print("\n" + "=" * 60)
    print("TOP GRASPS")
    print("=" * 60)
    for i, r in enumerate(results):
        fc = "FORCE-CLOSURE" if r.gws.is_force_closure else "no-FC"
        print(f"  #{i}: ε={r.gws.epsilon:.5f} {fc} "
              f"grav={r.gws.gravity_margin:.4f} "
              f"σ_min={r.gws.min_singular:.4f} rank={r.gws.rank} "
              f"contacts={len(r.contact_fingers)} "
              f"fingers={sorted(r.contact_fingers)} "
              f"yaw={r.thumb_yaw:.1f} palm={r.has_palm_collision}")

    print("\nRendering top grasps ...")
    render_grasps(scene_xml, results, output_dir, max_render=8)
    print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
