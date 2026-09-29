"""Audit the final paired opposing-box demo against its recorded physics states."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def validate(root):
 rows=[];pairs=[]
 for seed in [42,123,456,789]:
  loaded=[]
  for method in ['variational','cem']:
   p=root/f'seed{seed}_{method}';meta=json.loads((p/'provenance.json').read_text());res=json.loads((p/'result.json').read_text());states=np.load(p/'states.npz');audit=json.loads((p/'contact_audit.json').read_text());t=meta['events'][0]['time_s']
   hold=[c for c in audit if 1.-1e-6<=c['time_s']-t<=7.+1e-6]
   mask=(states['time_s']>=t-1e-6)&(states['time_s']<=t+7.+1e-6)
   assert np.all(states['object_wrench'][mask]==0),'External force during lift or hold'
   hand=[z for c in hold for z in c['contacts'] if 'table' not in z['geom']]
   proximal=[]
   for c in hold:
    hh=[z for z in c['contacts'] if 'table' not in z['geom']]
    proximal.append(sum(z['normal_force_N'] for z in hh if 'distal' not in z['geom'])/max(1e-12,sum(z['normal_force_N'] for z in hh)))
   row={'seed':seed,'method':method,'lift_success':res['lift_success'],'shear_success':res['shear_success'],'height_change_mm':res['lift_height_achieved']*1000,'acquisition_steps':res['n_steps'],'lift_start_s':t,'hold_frames':len(hold),'opposed_hold_frames':sum(c['thumb_tip_opposition'] for c in hold),'table_supported_frames':sum(any('table' in z['geom'] and z['normal_force_N']>.1 for z in c['contacts']) for c in hold),'palm_supported_frames':sum(any(any(n in z['geom'] for n in ['palm','hand_base']) and z['normal_force_N']>.1 for z in c['contacts']) for c in hold),'max_hand_overlap_mm':max([-1000*z['distance_m'] for z in hand]+[0]),'max_proximal_normal_force_fraction':max(proximal+[0]),'no_external_wrench_during_lift_hold':True}
   if method=='variational':
    assert res['lift_success'] and res['shear_success']
    assert row['opposed_hold_frames']==row['hold_frames']>=300
    assert row['table_supported_frames']==row['palm_supported_frames']==0
   else:assert not res['lift_success']
   rows.append(row);loaded.append((meta,states,p))
  lm,ls,lp=loaded[0];rm,rs,rp=loaded[1]
  for key in ['source_sha256','trial_sha256','setup_sha256','model_sha256']:assert lm[key]==rm[key],key
  for key in set(lm['settings'])|set(rm['settings']):
   if key not in ['method','output']:assert lm['settings'][key]==rm['settings'][key],key
  for key in ['qpos','qvel','ctrl','object_wrench']:assert np.array_equal(ls[key][0],rs[key][0]),key
  assert not np.array_equal(lm['final_hand_target'],rm['final_hand_target'])
  pairs.append({'seed':seed,'identical_initial_state_and_control':True,'identical_model_hash':lm['model_sha256'],'identical_settings_except_method_and_output':True,'distinct_executed_final_hand_targets':True})
 result={'scope':'Four-seed repeat on one exploratory selected opposing grasp. This is not a population success-rate benchmark or an optimizer-isolation ablation.','pairs':pairs,'runs':rows,'VNB_retained':4,'CEM_retained':0,'contact_model_limit':'Stock soft contacts permit millimeter-scale penetration and proximal thumb contact. Shared stiffer-contact checks can eliminate the exclusive retention advantage.'}
 (root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();validate(a.root)
