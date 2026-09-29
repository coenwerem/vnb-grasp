"""Paired dataset-preshape experiments using the recovered March 1 controller.

The dataset supplies geometry only. Every transfer, protocol change and outcome
is recorded separately from the archived submission reproduction.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from opposing_grasp_setup import arm_ik, contact_audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/iros_advantage_diagnosis_20260929/recovered_mar01_snapshot'


def load_runner(hold_seconds=2.):
    os.chdir(SOURCE)
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'examples')]
    sys.modules['mujoco.mjx'] = None
    source = (SOURCE / 'examples/run_variational_belief_experiments.py').read_text()
    # The new trial uses physical seconds, accounting for the wrapper's substeps.
    start = source.index('def run_lift_and_shear(')
    end = source.index('\ndef ', start + 1)
    stress = source[start:end].replace('dt = env.model.opt.timestep',
                                      'dt = env.model.opt.timestep * env.n_substeps')
    stress = stress.replace('range(int(0.2 / dt))', f'range(int({hold_seconds!r} / dt))')
    source = source[:start] + stress + source[end:]
    spec = importlib.util.spec_from_loader('dataset_trial_runner', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(SOURCE / 'examples/run_variational_belief_experiments.py')
    sys.modules[spec.name] = mod
    exec(compile(source, str(SOURCE / 'examples/run_variational_belief_experiments.py'), 'exec'), mod.__dict__)
    return mod, source


def object_bounds(m, body):
    points = []
    for g in np.flatnonzero(m.geom_bodyid == m.body(body).id):
        if not m.geom_contype[g]:
            continue
        R = np.empty(9)
        mj.mju_quat2Mat(R, m.geom_quat[g])
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        elif m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX:
            v = np.array(np.meshgrid(*[[-s, s] for s in m.geom_size[g]])).reshape(3, -1).T
        elif m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE:
            v = np.vstack((np.eye(3), -np.eye(3))) * m.geom_size[g, 0]
        else:
            raise ValueError('Only native box/cylinder mesh transfers are supported in this trial')
        points.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
    v = np.concatenate(points)
    return v.min(axis=0), v.max(axis=0)


def prepare(env, mod, cfg, bank_path, yaw, friction, out, opening=.2):
    bank = json.loads(bank_path.read_text())
    m, d = env.model, env.data
    low, high = object_bounds(m, cfg['body'])
    extents = (high - low) * 1000
    if not np.allclose(extents, bank['grasp_extents_mm'], atol=.5, rtol=0):
        raise ValueError(f"Dataset/release dimensions differ: {bank['grasp_extents_mm']} versus {extents}")
    mj.mj_resetData(m, d)
    mod.scripted_stash_all(m, d)
    Z = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    # Bank objects are centered. The release body may have a mesh-center offset.
    center = (low + high) / 2
    obj_center = np.array([-.05, .88, mod.TABLE_Z + (high[2]-low[2])/2])
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, Z.ravel())
    mod.scripted_spawn_object(m, d, cfg['body'], *(obj_center-Z @ center), quat=quat)
    T = np.asarray(bank['X_WB'])
    pos = obj_center + Z @ T[:3, 3]
    rot = Z @ T[:3, :3]
    q_bank = np.array([bank['q'][bank['dof_order'].index(m.joint(i).name)] for i in range(6, 17)])
    active = np.array([i != 0 and bank.get('seating', {}).get(m.joint(i+6).name.split('_')[0], {}).get('reg', True) for i in range(11)])
    q_open = q_bank.copy()
    q_open[active] -= opening
    q_open = np.clip(q_open, m.jnt_range[6:17, 0], m.jnt_range[6:17, 1])
    d.qpos[6:17] = q_open
    mj.mj_forward(m, d)
    seeds = [mod.GRASP_ARM_CONFIG, [-.7864192,-2.0627282,-2.1306253,-.782652,.3213678,2.6449529],
             [-.8,-1.6,-1.9,-1.0,1.3,1.5]]
    solutions=[]
    for seed in seeds:
        try:
            q_goal = arm_ik(m,d,pos,rot,seed)
            q_high = arm_ik(m,d,pos+[0,0,.07],rot,q_goal)
            d.qpos[:6]=q_high; mj.mj_forward(m,d)
            # Reject penetration with the table or robot; intended fingertip/object
            # contact is checked separately at the lower grasp pose.
            bad=[float(c.dist) for c in d.contact if c.dist < -.001 and
                 ('table' in m.geom(c.geom1).name or 'table' in m.geom(c.geom2).name) and
                 m.geom_bodyid[c.geom1] != m.body(cfg['body']).id and m.geom_bodyid[c.geom2] != m.body(cfg['body']).id]
            if not bad:solutions.append((q_goal,q_high))
        except RuntimeError:
            pass
    if not solutions:
        raise RuntimeError('No reachable, table-clear raised preshape')
    q_goal,q_high=solutions[0]
    d.qpos[:6]=q_goal;d.qpos[6:17]=q_bank;mj.mj_forward(m,d)
    planned=contact_audit(m,d,cfg['body'])
    d.qpos[:6]=q_high;d.qpos[6:17]=q_open;d.qvel[:]=0
    for gid in set(np.flatnonzero(m.geom_bodyid==m.body(cfg['body']).id))|set(env.fingertip_geoms):
        m.geom_friction[gid,0]=friction
    mj.mj_forward(m,d)
    initial=contact_audit(m,d,cfg['body'])
    if any(c['distance_m'] < -.0005 and 'table' not in c['geom'] for c in initial['contacts']):
        raise RuntimeError('Raised preshape intersects the object')
    def advance(q,n):
        for _ in range(n):
            d.ctrl[:6]=q;d.ctrl[6:17]=mod._pd_hand_ctrl(q_open,env)
            mod.scripted_freeze_stash(m,d,active=cfg['body'])
            mj.mj_step(m,d)
    advance(q_high,300)
    q=q_high
    for alpha in np.linspace(0,1,76)[1:]:
        q=arm_ik(m,d,pos+[0,0,.07*(.5+.5*np.cos(np.pi*alpha))],rot,q)
        advance(q,10)
    advance(q_goal,200)
    mj.mj_forward(m,d)
    metadata={'bank':str(bank_path),'bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              'bank_record':bank,'release_extents_mm':extents.tolist(),'yaw_deg':yaw,
              'object_center':obj_center.tolist(),'q_bank':q_bank.tolist(),'active_flexion':active.tolist(),
              'arm_target':q_goal.tolist(),'planned_contacts':planned,'initial_contacts':initial,
              'approach_end':contact_audit(m,d,cfg['body']), 'opening_rad':opening}
    (out/'initialization.json').write_text(json.dumps(metadata,indent=2))
    return q_bank,active


def main():
    trial_source=Path(__file__).read_text()
    audit_source=(Path(__file__).parent/'opposing_grasp_setup.py').read_text()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--object',required=True)
    p.add_argument('--method',choices=['variational','cem','hold'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--yaw',type=float,default=90)
    p.add_argument('--friction',type=float,default=.7)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=80)
    p.add_argument('--beta',type=float,default=.9)
    p.add_argument('--lock-opposition',action='store_true')
    p.add_argument('--closure-limit',type=float)
    p.add_argument('--torque-limit',type=float,default=2.)
    p.add_argument('--contact-timeconst',type=float)
    p.add_argument('--freeze-inactive',action='store_true')
    p.add_argument('--always-lift',action='store_true',help='Physically evaluate acquisition failures as well as quality-gated successes')
    p.add_argument('--hold-seconds',type=float,default=2.)
    p.add_argument('--audit-every',type=int,default=5)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);np.random.seed(a.seed);torch.manual_seed(a.seed)
    mod,source=load_runner(a.hold_seconds);(a.output/'runner_snapshot.py').write_text(source)
    (a.output/'trial_snapshot.py').write_text(trial_source)
    (a.output/'setup_snapshot.py').write_text(audit_source)
    env=mod.make_env();cfg=mod.OBJECT_CONFIGS['cube' if a.object=='sphere' else a.object]
    scene=mj.MjSpec.from_file(str(SOURCE/'arenas/zarm_realhand_l6_right_arena_no_wrist_cam/scene.xml'))
    if a.object=='sphere':
        # Match the native 60 mm sphere in the dataset, retaining the release
        # cube's 50 g mass and updating inertia to that of a uniform sphere.
        g=next(g for g in scene.geoms if g.name=='cube_collision')
        g.type=mj.mjtGeom.mjGEOM_SPHERE;g.size[:]=[.03,0,0]
    for e in list(scene.excludes):
        if cfg['body'] in {e.bodyname1,e.bodyname2} and {e.bodyname1,e.bodyname2}&{'world','table'}:scene.delete(e)
    env.model=scene.compile();env.data=mj.MjData(env.model);env.actmap=mod.ActuatorMap(env.model)
    m,d=env.model,env.data;ob=m.body(cfg['body']).id;palm=m.body('palm_link').id
    if a.contact_timeconst is not None:
        for gid in set(np.flatnonzero(m.geom_bodyid==ob))|set(env.fingertip_geoms):
            m.geom_solref[gid]=[a.contact_timeconst,1.]
    if a.object=='sphere':
        m.body_inertia[ob]=.4*m.body_mass[ob]*.03**2
        mj.mj_setConst(m,d)
    phase=['initialization'];origin=[None];samples=[];contacts=[];events=[];target=[None];preset=[None]
    raw_pd=mod._pd_hand_ctrl
    def pd(q,e):
        q=np.array(q).copy()
        if preset[0] is not None:
            bank,active=preset[0]
            if a.lock_opposition:q[~active]=bank[~active]
            if a.closure_limit is not None:q=np.minimum(q,bank+a.closure_limit)
        target[0]=q.copy()
        return np.clip(raw_pd(q,e),-a.torque_limit,a.torque_limit)
    mod._pd_hand_ctrl=pd
    def record():
        if origin[0] is None:return
        t=float(d.time-origin[0]);R=d.xmat[palm].reshape(3,3)
        samples.append((t,phase[0],d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.xfrc_applied[ob].copy(),R.T@(d.xpos[ob]-d.xpos[palm])))
        if (len(samples)-1)%a.audit_every==0:contacts.append({'time_s':t,'phase':phase[0],**contact_audit(m,d,cfg['body'])})
    def position(*args,**kwargs):
        preset[0]=prepare(env,mod,cfg,a.bank,a.yaw,a.friction,a.output)
        origin[0]=float(d.time);phase[0]='grasp';record();return True
    mod.position_arm_and_object=position
    step=env.step
    def wrapped_step(ctrl):
        if a.freeze_inactive:
            mod.scripted_freeze_stash(m,d,active=cfg['body'])
        val=step(ctrl);record();return val
    env.step=wrapped_step
    lift=mod.run_lift_and_shear
    def wrapped_lift(*args,**kwargs):
        phase[0]='lift_and_shear';events.append({'phase':phase[0],'time_s':float(d.time-origin[0])})
        # Preserve a model after initialization for a renderer without saving
        # hundreds of MB for unsuccessful contact-acquisition candidates.
        mj.mj_saveModel(m,str(a.output/'model.mjb'))
        return lift(*args,**kwargs)
    mod.run_lift_and_shear=wrapped_lift
    mod.run_perturbation_battery=lambda *args,**kwargs:dict(n_survived=0,n_total=0,survival_rate=0.,details=[])
    result={}
    try:
        if a.method=='hold':
            env.reset();mod._set_object_geom_filter(env,cfg['geom']);position()
            bank,active=preset[0];q0=d.qpos[6:17].copy();aq=d.qpos[:6].copy()
            for alpha in np.linspace(0,1,50):
                ctrl=np.r_[aq,pd((1-alpha)*q0+alpha*bank,env)];env.step(ctrl)
            result=wrapped_lift(env,aq,bank,cfg)
        else:
            fn=mod.run_variational_episode if a.method=='variational' else mod.run_cem_episode
            result=mod._to_json(asdict(fn(env,cfg,beta=a.beta,seed=a.seed,friction=a.friction,max_steps=a.max_steps,deadline=time.time()+600)))
            if a.always_lift and not events:
                stress_result=wrapped_lift(env,d.ctrl[:6].copy(),target[0].copy(),cfg)
                result.update(stress_result)
                result['shear_max_disp']=stress_result['max_displacement']
                result['evaluation_after_quality_gate_failure']=True
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
    except Exception as exc:
        (a.output/'error.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)},indent=2));raise
    finally:
        provenance={'settings':vars(a),'events':events,'mujoco':mj.__version__,'source_root':str(SOURCE),
                    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                    'trial_sha256':hashlib.sha256(trial_source.encode()).hexdigest(),
                    'setup_sha256':hashlib.sha256(audit_source.encode()).hexdigest(),
                    'final_hand_target':None if target[0] is None else target[0].tolist(),
                    'final_arm_target':d.ctrl[:6].tolist(),
                    'protocol_changes':['Dataset preshape with matching object dimensions','Active object/table collisions restored',
                    'Identical hand/object sliding friction',f'Stress uses physical seconds and {a.hold_seconds:g}-second hold','Separate battery omitted; robust_success not evaluated'],
                    'phase_semantics':'Controller actions unchanged except explicitly requested shared joint constraints'}
        (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2,default=str))
        (a.output/'contact_audit.json').write_text(json.dumps(contacts))
        if samples:
            np.savez_compressed(a.output/'states.npz',time_s=[x[0] for x in samples],phase=[x[1] for x in samples],qpos=[x[2] for x in samples],qvel=[x[3] for x in samples],ctrl=[x[4] for x in samples],object_wrench=[x[5] for x in samples],object_in_palm=[x[6] for x in samples])
    print(json.dumps({k:v for k,v in result.items() if k!='step_log'},indent=2))


if __name__=='__main__':main()
