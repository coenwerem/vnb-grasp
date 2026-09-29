"""Transfer a named FirmGrasp hand pose to the unchanged IROS release scene.

The bank supplies object-relative hand geometry, not an execution outcome.
The active cube remains free during approach, closure, and evaluation.
"""
from pathlib import Path
import hashlib
import json

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

BANK = Path('/home/drce/ResearchProjects/Datasets/realhand_o6_iron_hw_v2/cube.json')


def contact_audit(model, data, object_name='cube'):
    ob = model.body(object_name).id
    R = data.xmat[ob].reshape(3, 3)
    rows = []
    for i, c in enumerate(data.contact):
        if ob not in model.geom_bodyid[[c.geom1, c.geom2]]:
            continue
        other = c.geom2 if model.geom_bodyid[c.geom1] == ob else c.geom1
        name = model.geom(other).name
        force = np.zeros(6)
        mj.mj_contactForce(model, data, i, force)
        # Normal outward from the cube, regardless of geom ordering.
        normal = c.frame[:3] * (1 if model.geom_bodyid[c.geom1] == ob else -1)
        rows.append(dict(geom=name, position_object=(R.T @ (c.pos-data.xpos[ob])).tolist(),
                         normal_object=(R.T @ normal).tolist(), distance_m=float(c.dist),
                         normal_force_N=float(force[0])))
    tips = [r for r in rows if 'distal' in r['geom'] and r['normal_force_N'] > .1]
    thumbs = [r for r in tips if 'thumb' in r['geom']]
    fingers = [r for r in tips if 'thumb' not in r['geom']]
    # Use the line between contact patches, not vectors from the body origin.
    # Contacts on opposite sides near the end of a long cylinder can both lie
    # above its origin and still form a proper opposing pinch.
    pairs = []
    for t in thumbs:
        for f in fingers:
            separation = np.asarray(t['position_object']) - f['position_object']
            dot = float(np.dot(t['normal_object'], f['normal_object']))
            if (dot < -.8 and np.dot(separation, t['normal_object']) > 0
                    and np.dot(separation, f['normal_object']) < 0):
                pairs.append(dict(thumb=t['geom'], finger=f['geom'], normal_dot=dot,
                                  separation_m=float(np.linalg.norm(separation))))
    return dict(thumb_tip_opposition=bool(pairs), opposition_pairs=pairs,
                opposition_definition='Opposed outward normals and contact-separation projections', contacts=rows)


def arm_ik(model, data, position, rotation, seed):
    scratch = mj.MjData(model)
    scratch.qpos[:] = data.qpos
    bid = model.body('hand_base').id
    bounds = model.jnt_range[:6].copy()
    bounds[0] = [-2*np.pi, 2*np.pi]  # continuous shoulder pan

    def error(q):
        scratch.qpos[:6] = q
        mj.mj_kinematics(model, scratch)
        return np.r_[scratch.xpos[bid]-position,
                     .2*Rotation.from_matrix(rotation @ scratch.xmat[bid].reshape(3, 3).T).as_rotvec()]

    result = least_squares(error, np.clip(seed, bounds[:, 0]+1e-6, bounds[:, 1]-1e-6),
                           bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=500,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if np.linalg.norm(result.fun) > 1e-5:
        raise RuntimeError(f'FirmGrasp transfer IK failed: {result.fun}')
    return result.x


def initialize(env, mod, cfg, friction, output, offset_xy=(0., 0.), opening=.2):
    if cfg['body'] != 'cube':
        raise ValueError('This transfer is only defined for the 50 mm cube')
    bank = json.loads(BANK.read_text())
    m, d = env.model, env.data
    mj.mj_resetData(m, d)
    mod.scripted_stash_all(m, d)
    objpos = np.array([-.05+offset_xy[0], .88+offset_xy[1], .802])
    mod.scripted_spawn_object(m, d, 'cube', *objpos)
    q_bank = np.array([bank['q'][bank['dof_order'].index(m.joint(i).name)] for i in range(6, 17)])
    # Index is retracted in this bank grasp. Keep it retracted and open the
    # thumb flexion + three contact fingers; establish opposition before closing.
    q_open = q_bank.copy()
    q_open[[1, 2, 5, 6, 7, 8, 9, 10]] -= opening
    q_open = np.clip(q_open, m.jnt_range[6:17, 0], m.jnt_range[6:17, 1])
    d.qpos[6:17] = q_open
    T = np.array(bank['X_WB'])
    Z = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    # Keep the planned hand pose nominal: offset_xy is an unobserved object
    # placement error, rather than a coordinated translation of the whole task.
    position = np.array([-.05, .88, .802]) + Z @ T[:3, 3]
    rotation = Z @ T[:3, :3]
    seed = [-.7864192, -2.0627282, -2.1306253, -.7826520, .3213678, 2.6449529]
    q_goal = arm_ik(m, d, position, rotation, seed)
    q_high = arm_ik(m, d, position+[0., 0., .06], rotation, q_goal)
    d.qpos[:6] = q_high
    d.qvel[:] = 0
    ob = m.body('cube').id
    for gid in set(np.where(m.geom_bodyid == ob)[0]) | set(env.fingertip_geoms):
        m.geom_friction[gid, 0] = friction
    mj.mj_forward(m, d)
    initial_contacts = contact_audit(m, d)
    if any('table' not in c['geom'] and c['distance_m'] < 0 for c in initial_contacts['contacts']):
        raise RuntimeError('Raised preshape starts in collision with cube')
    trace = []
    def advance(q, n):
        for _ in range(n):
            d.ctrl[:6] = q
            d.ctrl[6:17] = mod._pd_hand_ctrl(q_open, env)
            mod.scripted_freeze_stash(m, d, active='cube')
            mj.mj_step(m, d)
        trace.append((float(d.time), d.qpos.copy()))
    advance(q_high, 300)
    # Cartesian descent keeps the bank's orientation fixed.
    current = q_high
    for alpha in np.linspace(0, 1, 76)[1:]:
        z = .06*(.5+.5*np.cos(np.pi*alpha))
        current = arm_ik(m, d, position+[0., 0., z], rotation, current)
        advance(current, 10)
    advance(q_goal, 200)
    mj.mj_forward(m, d)
    audit = dict(bank_path=str(BANK), bank_sha256=hashlib.sha256(BANK.read_bytes()).hexdigest(),
                 bank=bank, yaw_about_cube_deg=90, opening_rad=opening,
                 arm_target=q_goal.tolist(), hand_open_target=q_open.tolist(),
                 initial_contacts=initial_contacts, approach_end=contact_audit(m, d),
                 active_object_free=True, object_offset_xy_m=list(offset_xy),
                 offset_semantics='Object only; planned hand target remains nominal')
    (output/'opposing_initialization.json').write_text(json.dumps(audit, indent=2))
    np.savez_compressed(output/'approach_states.npz', time_s=[t for t,q in trace], qpos=[q for t,q in trace])
    return True
