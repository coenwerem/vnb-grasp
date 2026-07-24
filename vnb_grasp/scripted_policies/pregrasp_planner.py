"""Geometry-aware pregrasp planner for the ZArm and RealHand L6.

This module plans and classifies grasps (TOP_DOWN vs SIDE), solves IK,
and generates smooth approach trajectories via mink differential IK.

SIDE grasps use the same TOP_DOWN arm configuration (for reliable PD
tracking) but with:

    1. Lower grasp height - palm at the object mid-section
    2. Wrist3 rotated up to 90 deg CW - fingers re-oriented to wrap the sides
    3. Jacobian-based droop compensation - ctrl adjusted so the PD
       steady state places the palm at the *actual* target

Designed to be imported by video-recording scripts and learning pipelines.

Usage::

    from vnb_grasp.scripted_policies.pregrasp_planner import (
        PregraspPlanner, PregraspPlan,
    )

    planner = PregraspPlanner(model, data)
    plan    = planner.plan("graspit_cylinder")
    # plan.strategy, plan.arm_q_grasp, plan.descent_traj, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logging.getLogger("mink").setLevel(logging.ERROR)

import mujoco as mj

try:
    import mink
except ImportError:
    mink = None


# 
#  Constants
# 
TABLE_Z      = 0.777
OBJ_XY       = np.array([-0.05, 0.88])
HOME_Q       = np.array([0.0, -1.57, 0.0003, -1.57, 0.0, 0.0])
FINGER_REACH = 0.075

# IK seed - works well for both TOP_DOWN and SIDE (PD-stable config)
GRASP_SEED = np.array([-0.826, -2.200, -1.643, -1.429, 0.500, 2.090])

GRASP_TORQUE = np.array([
    1.0,  0.55, 0.85,
    1.20, 1.10,
    1.20, 1.10,
    1.20, 1.10,
    1.20, 1.10,
])

ALL_OBJECTS = [
    "cube", "graspit_cylinder", "graspit_box",
    "005_tomato_soup_can", "006_mustard_bottle",
    "010_potted_meat_can", "056_tennis_ball",
]

SIDE_RATIO_THRESHOLD = 1.3      # h / diameter above this --> SIDE
SIDE_WRIST3_OFFSET   = -np.pi / 2  # -90 deg CW from default wrist3 (side wrap)


# 
#  Data classes
# 
@dataclass
class PregraspPlan:
    """Everything needed to execute a pregrasp --> grasp --> lift."""
    body_name:       str
    strategy:        str
    geom:            dict

    grasp_pos:       np.ndarray
    pregrasp_pos:    np.ndarray

    arm_q_grasp:     np.ndarray
    arm_q_pregrasp:  np.ndarray
    wrist3:          float

    ik_err_grasp:    float
    ik_err_pregrasp: float
    ik_ok:           bool

    descent_traj:    np.ndarray     # (K, 6)
    ratio:           float

    seed_used:       np.ndarray = field(
        default_factory=lambda: GRASP_SEED.copy())


# 
#  Geometry
# 
def _geom_half(m, gi, axis):
    t, s = m.geom_type[gi], m.geom_size[gi]
    if   t == 6: return s[axis]        # box
    elif t == 5: return s[0] if axis < 2 else s[1]   # cylinder
    elif t == 7: return s[axis]        # mesh
    else:        return s[0]


def get_geom_info(model, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0, f"Body '{body_name}' not found"
    zmin, zmax = float("inf"), float("-inf")
    sx, sy, n = 0.0, 0.0, 0
    mhx, mhy = 0.0, 0.0
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] != bid or model.geom_contype[gi] == 0:
            continue
        gp = model.geom_pos[gi]
        hz = _geom_half(model, gi, 2)
        zmin = min(zmin, gp[2] - hz)
        zmax = max(zmax, gp[2] + hz)
        mhx = max(mhx, _geom_half(model, gi, 0))
        mhy = max(mhy, _geom_half(model, gi, 1))
        sx += gp[0]; sy += gp[1]; n += 1
    if zmin > zmax:
        zmin, zmax = -0.025, 0.025
    n = max(n, 1)
    return dict(body_z=TABLE_Z - zmin, height=zmax - zmin,
                half_x=mhx, half_y=mhy,
                radius=max(mhx, mhy),
                off_x=sx / n, off_y=sy / n)


# 
#  Strategy
# 
def classify_strategy(geom, threshold=SIDE_RATIO_THRESHOLD):
    h = geom["height"]
    d = 2 * max(geom["half_x"], geom["half_y"])
    return "SIDE" if h / max(d, 0.01) > threshold else "TOP_DOWN"


# 
#  Position planning
# 
def plan_grasp_pos(geom, strategy, obj_xy=OBJ_XY):
    h = geom["height"]
    if strategy == "SIDE":
        # SIDE grasps: palm at object mid-section for natural side wrap.
        # Place palm at ~50% height so fingers wrap the widest part.
        z = TABLE_Z + h * 0.50 + 0.015
    elif h <= FINGER_REACH:
        z = TABLE_Z + h + 0.012
    else:
        d = 2 * max(geom["half_x"], geom["half_y"])
        coeff = 0.25 if d >= 0.070 else 0.40
        z = TABLE_Z + h - FINGER_REACH * coeff
    return np.array([obj_xy[0], obj_xy[1], z])


def plan_pregrasp_pos(geom, strategy, grasp_pos, obj_xy=OBJ_XY):
    """Both strategies use a pregrasp directly above the object top
    so the arm never collides with the object on the way down."""
    z = TABLE_Z + geom["height"] + FINGER_REACH + 0.02
    return np.array([obj_xy[0], obj_xy[1], z])


# 
#  Wrist-3 selection
# 
def pick_wrist3(model, data, seed, half_x, half_y, strategy="TOP_DOWN"):
    """Choose wrist3 angle.  For SIDE strategy, applies an additional
    CW rotation offset so the fingers wrap the sides."""
    palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm_link")

    if abs(half_x - half_y) < 0.004:
        base_w3 = seed[5]
    else:
        narrow = np.array([1, 0, 0]) if half_x < half_y else np.array([0, 1, 0])
        saved = data.qpos.copy()
        best_w3, best_score = seed[5], 0.0
        for w3 in np.linspace(-3.0, 3.0, 37):
            data.qpos[0:6] = seed.copy()
            data.qpos[5] = w3
            data.qvel[:] = 0
            mj.mj_forward(model, data)
            py = data.xmat[palm_bid].reshape(3, 3)[:, 1]
            score = abs(py[0] * narrow[0] + py[1] * narrow[1])
            if score > best_score:
                best_score, best_w3 = score, w3
        data.qpos[:] = saved
        mj.mj_forward(model, data)
        base_w3 = best_w3

    if strategy == "SIDE":
        return base_w3 + SIDE_WRIST3_OFFSET
    return base_w3


# 
#  IK  (position-only, soft wrist3 constraint)
# 
def solve_ik(model, data, target_pos, *, seed=None, wrist3=None,
             max_iter=500, tol=0.003):
    pbid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm_link")
    saved = data.qpos.copy()

    s = (seed if seed is not None else GRASP_SEED).copy()
    if wrist3 is not None:
        s[5] = wrist3
    data.qpos[0:6] = s
    data.qpos[6:17] = 0.0
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    best_q, best_err = s.copy(), float("inf")
    w_w3 = 0.30

    for i in range(max_iter):
        perr = target_pos - data.xpos[pbid]
        jacp = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, None, pbid)
        J = jacp[:, 0:6]
        err = perr.copy()

        if wrist3 is not None:
            w3e = w_w3 * (wrist3 - data.qpos[5])
            Jw = np.zeros((1, 6)); Jw[0, 5] = w_w3
            J = np.vstack([J, Jw])
            err = np.append(err, w3e)

        enorm = np.linalg.norm(perr)
        if enorm < best_err:
            best_err, best_q = enorm, data.qpos[0:6].copy()
        if enorm < tol:
            break

        lam = 0.01 * (1 + 0.15 * i / max_iter)
        dq = np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ err)
        data.qpos[0:6] += 0.5 * dq
        mj.mj_forward(model, data)

    data.qpos[:] = saved; data.qvel[:] = 0; mj.mj_forward(model, data)
    return best_q, best_err


def solve_ik_multi(model, data, target_pos, *, seed=None, wrist3=None,
                   n=12):
    _seed = seed if seed is not None else GRASP_SEED
    best_q, best_e = solve_ik(model, data, target_pos,
                              seed=_seed, wrist3=wrist3)
    if best_e < 0.003:
        return best_q, best_e
    rng = np.random.default_rng(42)
    for _ in range(n):
        s = _seed + rng.uniform(-0.5, 0.5, 6)
        q, e = solve_ik(model, data, target_pos,
                        seed=s, wrist3=wrist3, max_iter=350)
        if e < best_e:
            best_q, best_e = q, e
        if e < 0.003:
            break
    return best_q, best_e


# 
#  Droop compensation
# 
def compensate_droop(model, data, arm_q, target_pos, iterations=3,
                     settle_steps=300, gain=0.5):
    """Iteratively adjust ctrl to compensate PD gravity droop.

    Simulates the arm at *arm_q*, measures the actual palm position,
    then uses the Jacobian to nudge ctrl so the steady-state palm
    ends up closer to *target_pos*.

    Returns the compensated (6,) ctrl array.
    """
    pbid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm_link")
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    ctrl = arm_q.copy()
    for it in range(iterations):
        # reset arm to a fresh start from ctrl
        data.qpos[0:6]  = ctrl
        data.qpos[6:17] = 0.0
        data.qvel[:]     = 0
        data.ctrl[0:6]  = ctrl
        data.ctrl[6:17] = 0.0
        mj.mj_forward(model, data)
        # settle under PD control
        for _ in range(settle_steps):
            data.ctrl[0:6] = ctrl
            mj.mj_step(model, data)
        # measure droop
        actual = data.xpos[pbid].copy()
        droop = actual - target_pos
        if np.linalg.norm(droop) < 0.005:
            break
        # Jacobian correction
        jacp = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, None, pbid)
        J = jacp[:, 0:6]
        dq = np.linalg.lstsq(J, -droop * gain, rcond=None)[0]
        ctrl = ctrl + dq

    # restore state
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl
    mj.mj_forward(model, data)
    return ctrl


# 
#  Mink trajectory planner
# 
def plan_mink_descent(model, start_arm_q, target_palm_pos, max_steps=300):
    if mink is None:
        raise RuntimeError("mink not installed")

    cfg = mink.Configuration(model)
    q = np.zeros(model.nq)
    q[0:6] = start_arm_q
    cfg.update(q)

    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "palm_site")
    palm_start = cfg.data.site_xpos[sid].copy()

    pos_task = mink.FrameTask(
        frame_name="palm_site", frame_type="site",
        position_cost=1.0, orientation_cost=0.0,
    )
    posture_task = mink.PostureTask(model, cost=0.005)
    posture_task.set_target_from_configuration(cfg)
    limits = [mink.ConfigurationLimit(model)]

    traj = [start_arm_q.copy()]
    dt = 0.005
    for step in range(max_steps):
        a = min(1.0, (step + 1) / max_steps)
        pos_now = (1 - a) * palm_start + a * target_palm_pos
        T = mink.SE3.from_rotation_and_translation(
            mink.SO3.identity(), pos_now)
        pos_task.set_target(T)
        vel = mink.solve_ik(
            cfg, [pos_task, posture_task],
            dt=dt, solver="quadprog", damping=1e-3, limits=limits,
        )
        cfg.integrate_inplace(vel, dt)
        traj.append(cfg.data.qpos[0:6].copy())
        err = np.linalg.norm(cfg.data.site_xpos[sid] - target_palm_pos)
        if err < 0.002 and a > 0.9:
            break

    return np.array(traj)


def interp_trajectory(traj, n_steps):
    t_orig = np.linspace(0, 1, len(traj))
    t_new  = np.linspace(0, 1, n_steps)
    out = np.zeros((n_steps, traj.shape[1]))
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(t_new, t_orig, traj[:, j])
    return out


# 
#  Object management
# 
def _body_addrs(model, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None, None, None
    j = model.body_jntadr[bid]
    return bid, model.jnt_qposadr[j], model.jnt_dofadr[j]


def stash_all(model, data, objects=ALL_OBJECTS):
    for b in objects:
        bid, qa, da = _body_addrs(model, b)
        if bid is None:
            continue
        data.qpos[qa:qa+3] = [0, 5.0, 0.85]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        data.qvel[da:da+6] = 0


def freeze_stash(model, data, active=None, objects=ALL_OBJECTS):
    for b in objects:
        if b == active:
            continue
        bid, _, da = _body_addrs(model, b)
        if bid is None:
            continue
        data.qvel[da:da+6] = 0


def spawn_object(model, data, body_name, x, y, body_z, quat=None):
    _, qa, da = _body_addrs(model, body_name)
    data.qpos[qa:qa+3] = [x, y, body_z]
    data.qpos[qa+3:qa+7] = quat if quat is not None else [1, 0, 0, 0]
    data.qvel[da:da+6] = 0


# 
#  Planner class
# 
class PregraspPlanner:
    """Plan pregrasp --> grasp sequences for the ZArm and RealHand.

    TOP_DOWN objects descend vertically with the default wrist.
    SIDE objects use the same descent path (base wrist angle) but the
    grasp IK includes a 45 deg CW wrist rotation.  The rotation is applied
    during the blend/settle phase after the arm has safely descended
    alongside the object.  Narrow-aligned objects keep their natural
    wrist alignment (no offset) since it already looks side-like.

    Parameters
    ----------
    model, data : MuJoCo model and data.
    table_z : float, overrides the default table surface height.
    obj_xy  : (2,), overrides the default object XY.
    """

    def __init__(self, model, data, *, table_z=TABLE_Z, obj_xy=None):
        self.model = model
        self.data  = data
        self.table_z = table_z
        self.obj_xy = np.asarray(obj_xy) if obj_xy is not None else OBJ_XY.copy()
        self.palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm_link")

    def plan(self, body_name, verbose=True):
        G = get_geom_info(self.model, body_name)
        strategy = classify_strategy(G)
        h  = G["height"]
        hx, hy = G["half_x"], G["half_y"]
        diameter = 2 * max(hx, hy)
        ratio = h / max(diameter, 0.01)

        grasp_pos    = plan_grasp_pos(G, strategy, self.obj_xy)
        pregrasp_pos = plan_pregrasp_pos(G, strategy, grasp_pos, self.obj_xy)

        seed = GRASP_SEED  # same seed for both strategies (good PD tracking)

        # Base wrist3: narrow-axis alignment without any SIDE offset.
        # Used for pregrasp/descent so the approach path is safe.
        w3_base = pick_wrist3(self.model, self.data, seed, hx, hy,
                              strategy="TOP_DOWN")

        # Grasp wrist3: SIDE objects always get the wrist rotation offset
        # for proper side-wrap finger orientation (visible in teaser render).
        if strategy == "SIDE":
            w3_grasp = w3_base + SIDE_WRIST3_OFFSET
        else:
            w3_grasp = w3_base

        # Pregrasp IK uses base wrist --> identical descent path to TOP_DOWN.
        aq_p, ep = solve_ik_multi(self.model, self.data, pregrasp_pos,
                                  seed=seed, wrist3=w3_base)
        # Grasp IK uses rotated wrist --> final grasp has visible rotation.
        aq_g, eg = solve_ik_multi(self.model, self.data, grasp_pos,
                                  seed=seed, wrist3=w3_grasp)

        # Mink descent starts from SAFE pregrasp (base wrist).
        # Wrist rotates during the blend phase (Phase 3.5) in the video script.
        descent = plan_mink_descent(self.model, aq_p, grasp_pos)

        plan = PregraspPlan(
            body_name=body_name, strategy=strategy, geom=G,
            grasp_pos=grasp_pos, pregrasp_pos=pregrasp_pos,
            arm_q_grasp=aq_g,
            arm_q_pregrasp=aq_p,
            wrist3=w3_grasp,
            ik_err_grasp=eg, ik_err_pregrasp=ep,
            ik_ok=eg < 0.01,
            descent_traj=descent, ratio=ratio,
            seed_used=seed.copy(),
        )

        if verbose:
            sym = "symmetric" if abs(hx - hy) < 0.004 else "narrow-aligned"
            print(f"  Geom  h={h*100:.1f}cm  d={diameter*100:.1f}cm  "
                  f"ratio={ratio:.1f}  off=({G['off_x']:.3f},{G['off_y']:.3f})")
            print(f"  Strategy={strategy}  "
                  f"Grasp={np.round(grasp_pos, 3).tolist()}  "
                  f"Pre={np.round(pregrasp_pos, 3).tolist()}")
            print(f"  Wrist3 base={w3_base:+.2f}  grasp={w3_grasp:+.2f} ({sym})"
                  f"{'  SIDE offset=' + str(round(SIDE_WRIST3_OFFSET,2)) if strategy=='SIDE' else ''}")
            print(f"  IK  grasp err={eg:.5f}  pre err={ep:.5f}  "
                  f"{'OK' if plan.ik_ok else 'WARN'}")
            print(f"  Mink descent: {len(descent)} waypoints")
        return plan
