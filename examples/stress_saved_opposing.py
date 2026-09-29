"""Continue recorded controller grasps through identical Cartesian-lift stress tests.

This is an additional physical evaluation, not an untouched submission replay.
The selected controller's final joint target is held throughout each test.
"""
import argparse, hashlib, json
from pathlib import Path
import mujoco as mj
import numpy as np
from opposing_grasp_setup import arm_ik, contact_audit


def run(source, output):
    output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'provenance.json').read_text());old=np.load(source/'states.npz')
    m=mj.MjModel.from_binary_path(str(source/'model.mjb'));d=mj.MjData(m)
    t0=next(e['time_s'] for e in meta['events'] if e['phase']=='lift_and_shear')
    idx=int(np.argmin(abs(old['time_s']-t0)));d.qpos[:]=old['qpos'][idx];d.qvel[:]=old['qvel'][idx];d.ctrl[:]=old['ctrl'][idx]
    d.time=t0;d.xfrc_applied[:]=0;mj.mj_forward(m,d)
    body='cube' if meta['settings']['object']=='sphere' else meta['settings']['object'];ob=m.body(body).id;palm=m.body('palm_link').id;hand=m.body('hand_base').id
    target=np.array(meta['final_hand_target']);arm=d.ctrl[:6].copy();initial_pos=d.xpos[ob].copy();hand_pos=d.xpos[hand].copy();hand_rot=d.xmat[hand].reshape(3,3).copy()
    dt=m.opt.timestep*10;frames=[];audits=[];events=[]
    def record(phase):
        R=d.xmat[palm].reshape(3,3)
        frames.append((float(d.time),phase,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.xfrc_applied[ob].copy(),R.T@(d.xpos[ob]-d.xpos[palm])))
        if len(frames)%5==1:audits.append({'time_s':float(d.time),'phase':phase,**contact_audit(m,d,body)})
    def step(q,phase):
        d.ctrl[:6]=q;d.ctrl[6:17]=np.clip(5*(target-d.qpos[6:17])-.3*d.qvel[6:17],-2,2)
        mj.mj_step(m,d,nstep=10);record(phase)
    # Keep full measured hand orientation while lifting vertically by 80 mm.
    q=arm
    for i in range(75):
        a=.5-.5*np.cos(np.pi*(i+1)/75)
        q=arm_ik(m,d,hand_pos+[0,0,.08*a],hand_rot,q)
        step(q,'lift_and_shear')
    for _ in range(75):step(q,'lift_and_shear')
    lift_frames=list(frames);lift_audits=list(audits)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    friction=m.geom_friction.copy();basepos=d.xpos[ob].copy();basequat=d.xquat[ob].copy();base_rel=d.xmat[palm].reshape(3,3).T@(basepos-d.xpos[palm]);stress_start=float(d.time)
    tests=[('friction',mu,0) for mu in [.3,.15,.1,.05]]+[(kind,mag,axis) for kind,mags in [('force',[3,6,12]),('torque',[.1,.2,.4])] for mag in mags for axis in range(3)]
    results=[]
    for kind,mag,axis in tests:
        mj.mj_setState(m,d,state,spec);m.geom_friction[:]=friction;mj.mj_forward(m,d);frames=list(lift_frames);audits=list(lift_audits)
        if kind=='friction':
            hand_bodies={m.body(i).id for i in range(m.nbody) if any(k in m.body(i).name for k in ['thumb','index','middle','ring','pinky','palm'])}
            for g in range(m.ngeom):
                if m.geom_bodyid[g]==ob or m.geom_bodyid[g] in hand_bodies:m.geom_friction[g,0]=mag
        max_disp=0.;max_angle=0.;max_drop=0.;loss_time=None
        for i in range(200):
            d.xfrc_applied[ob]=0
            if kind!='friction' and i*dt<.3:d.xfrc_applied[ob,axis+(3 if kind=='torque' else 0)]=mag
            step(q,'stress')
            rel=d.xmat[palm].reshape(3,3).T@(d.xpos[ob]-d.xpos[palm]);disp=float(np.linalg.norm(rel-base_rel));drop=float(basepos[2]-d.xpos[ob,2]);angle=float(2*np.arccos(np.clip(abs(np.dot(basequat,d.xquat[ob])),0,1)))
            max_disp=max(max_disp,disp);max_drop=max(max_drop,drop);max_angle=max(max_angle,angle)
            if loss_time is None and (disp>.03 or drop>.025):loss_time=float(d.time-stress_start)
        survived=loss_time is None
        name=f'{kind}_{mag:g}_a{axis}';dest=output/name;dest.mkdir()
        result=dict(kind=kind,magnitude=mag,axis=axis,lift_height_m=float(basepos[2]-initial_pos[2]),max_displacement_m=max_disp,max_drop_m=max_drop,max_angle_rad=max_angle,loss_time_s=loss_time,survived=survived)
        results.append(result|{'run':name});(dest/'result.json').write_text(json.dumps(result,indent=2))
        (dest/'contact_audit.json').write_text(json.dumps(audits))
        # Include original acquisition states so both files retain their common reset.
        keys=['time_s','phase','qpos','qvel','ctrl','object_wrench','object_in_palm']
        np.savez_compressed(dest/'states.npz',**{k:np.concatenate([old[k][:idx+1],np.asarray([f[j] for f in frames])]) for j,k in enumerate(keys)})
        (dest/'model.mjb').symlink_to((source/'model.mjb').resolve())
        provenance=meta.copy();provenance['settings']=meta['settings']|{'output':str(dest),'stress_kind':kind,'stress_magnitude':mag,'stress_axis':axis,'stress_protocol':'80mm Cartesian lift, 1.5s hold, 0.3s impulse or four-second friction change'}
        provenance['events']=[{'phase':'lift_and_shear','time_s':t0},{'phase':'friction_drop' if kind=='friction' else 'stress','time_s':stress_start,'after':mag,'kind':kind}]
        provenance['continuation_source']=str(source);provenance['continuation_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();provenance['model_sha256']=hashlib.sha256((source/'model.mjb').read_bytes()).hexdigest()
        provenance['protocol_changes']=meta['protocol_changes']+['Additional Cartesian-lift stress evaluation from recorded acquisition state; integration warm start initialized to zero at continuation; no controller re-optimization during stress']
        (dest/'provenance.json').write_text(json.dumps(provenance,indent=2));print(name,result,flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.source,a.output)
