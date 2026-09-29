"""Summarize every attempted dataset transfer, including rejected and failed runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    row={'run':path.name}
    if (path/'error.json').exists():
        return row|{'status':'error','error':json.loads((path/'error.json').read_text())}
    if not (path/'result.json').exists():return row|{'status':'incomplete'}
    result=json.loads((path/'result.json').read_text())
    meta=json.loads((path/'provenance.json').read_text())
    row.update({'status':'completed','settings':meta['settings'],'events':meta['events'],
                'metrics':{k:result.get(k) for k in ['n_steps','termination','success','lift_success',
                          'lift_height_achieved','shear_success','shear_max_disp','max_displacement']}})
    contacts=json.loads((path/'contact_audit.json').read_text())
    lift=next((e['time_s'] for e in meta['events'] if e['phase']=='lift_and_shear'),None)
    if lift is not None:
        hold=[c for c in contacts if lift+1.05 <= c['time_s'] <= lift+2.95]
        row['hold_audit']={'sampled_frames':len(hold),'opposed_frames':sum(c['thumb_tip_opposition'] for c in hold),
            'table_supported_frames':sum(any('table' in z['geom'] and z['normal_force_N']>.1 for z in c['contacts']) for c in hold),
            'proximal_contact_frames':sum(any('distal' not in z['geom'] and 'table' not in z['geom'] and z['normal_force_N']>.1 for z in c['contacts']) for c in hold),
            'max_hand_overlap_mm':max([-z['distance_m']*1000 for c in hold for z in c['contacts'] if 'table' not in z['geom']]+[0])}
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    rows=[summarize(x) for x in sorted(a.root.iterdir()) if x.is_dir()]
    (a.root/'all_trials_summary.json').write_text(json.dumps(rows,indent=2))
    for r in rows:
        m=r.get('metrics',{});h=r.get('hold_audit',{})
        print(r['run'],r['status'],m.get('n_steps'),'lift',m.get('lift_success'),'shear',m.get('shear_success'),
              'opposed',h.get('opposed_frames'),'/',h.get('sampled_frames'),'proximal',h.get('proximal_contact_frames'),'overlap',h.get('max_hand_overlap_mm'))


if __name__=='__main__':main()
