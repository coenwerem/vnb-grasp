"""Summarize live contacts and paired states; no stability inference from pictures."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco as mj


def audit(path, model):
    meta = json.loads((path/'provenance.json').read_text())
    result = json.loads((path/'result.json').read_text())
    contacts = json.loads((path/'contact_audit.json').read_text())
    state = np.load(path/'states.npz')
    events = {e['phase']: e for e in meta['events']}
    lift = events['lift_and_shear']['time_s']
    shear = events['friction_drop']['time_s']
    hold = (state['time_s'] >= lift+1.02) & (state['time_s'] < shear-1e-6)
    ids = np.flatnonzero(hold)
    qa = model.jnt_qposadr[model.body_jntadr[model.body('cube').id]]
    hold_contacts = [contacts[i] for i in ids]
    hand_contacts = [c for row in hold_contacts for c in row['contacts'] if 'table' not in c['geom']]
    relative = state['object_in_palm'][hold]
    start_id = np.argmin(abs(state['time_s']-lift))
    loss = events.get('grasp_lost', {})
    return dict(path=str(path.resolve()), method=meta['settings']['method'], seed=meta['settings']['seed'],
                placement_offset_xy_m=meta['settings']['object_offset_xy'],
                lift_mm=float((state['qpos'][ids[-1], qa+2]-state['qpos'][start_id, qa+2])*1000),
                sampled_hold_duration_s=float(state['time_s'][ids[-1]]-state['time_s'][ids[0]]),
                hold_samples=len(ids),
                opposed_hold_samples=sum(row['thumb_tip_opposition'] for row in hold_contacts),
                table_supported_hold_samples=sum(any('table' in c['geom'] and c['normal_force_N'] > .01
                                                     for c in row['contacts']) for row in hold_contacts),
                relative_hold_motion_mm=float(np.linalg.norm(relative-relative[0], axis=1).max()*1000),
                deepest_hand_penetration_hold_mm=float(max(0., -min(c['distance_m'] for c in hand_contacts))*1000),
                displacement_limit_time_after_shear_s=loss.get('time_s', shear)-shear if loss else None,
                final_object_z_m=float(state['qpos'][-1, qa+2]),
                final_opposition=contacts[-1]['thumb_tip_opposition'],
                final_table_contact=any('table' in c['geom'] and c['normal_force_N'] > .01 for c in contacts[-1]['contacts']),
                controller_termination=result['termination'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    paths=sorted(a.root.glob('hold2_drop01_s*'))
    paths=[p for p in paths if p.is_dir()]
    model=mj.MjModel.from_binary_path(str(paths[0]/'model.mjb'))
    rows=[audit(path,model) for path in paths]
    pairs=[]
    for seed in sorted(set(row['seed'] for row in rows)):
        episodes=[a.root/f'hold2_drop01_s{seed}_{method}' for method in ['variational','cem']]
        meta=[json.loads((p/'provenance.json').read_text()) for p in episodes]
        states=[np.load(p/'states.npz') for p in episodes]
        settings=[{k:v for k,v in m['settings'].items() if k not in ('method','output')} for m in meta]
        same_settings=settings[0]==settings[1]
        same_model=meta[0]['model_sha256']==meta[1]['model_sha256']
        same_initial=all(np.array_equal(states[0][k][0],states[1][k][0]) for k in ['qpos','qvel','ctrl'])
        assert same_settings and same_model and same_initial
        pairs.append(dict(seed=seed,identical_settings=same_settings,identical_model=same_model,identical_initial_state=same_initial))
    payload=dict(rows=rows,pair_checks=pairs,
        hold_pass_definition='Lift >=30 mm, no positive table support, thumb-tip vs finger-tip opposition in every sampled hold frame',
        hold_pass_count=sum(r['lift_mm']>=30 and r['table_supported_hold_samples']==0 and r['opposed_hold_samples']==r['hold_samples'] for r in rows),
        total=len(rows),
        scope='Exploratory paired checks of a FirmGrasp-seeded, bounded closure controller; untrained belief filter; not a paper benchmark or a statistical robustness claim')
    (a.root/'audit_summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
