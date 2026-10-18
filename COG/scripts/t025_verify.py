from t025_common import *
import pandas as pd
out=provenance();out['checks']=[];out['definition_changes']={}
def ck(name,v):out['checks'].append(dict(name=name,passed=bool(v)))
for co in ['T','S']:
 d=json.loads((OUT/f'information_{co}.json').read_text());z=np.load(OUT/f'information_{co}_predictions.npz')
 for name,b in [('E_given_F','M-F1'),('F_given_E','M-F2')]:
  loss=(z['M-F3']-z['y'])**2-(z[b]-z['y'])**2
  means=pd.Series(loss).groupby(z['match']).mean();ck(co+name,abs(means.mean()-d['contrasts'][name]['estimate'])<1e-12)
 ck(co+' full parity',d['full_arm_archived_q_max_abs_diff']==0);ck(co+' convergence',all(v['converged'] for v in d['arms'].values()))
def rows(arm):
 r={}
 for co in ['T','S']:
  with np.load(OUT/f'definition_{arm}_{co}.npz') as z:
   for m,a,s,y,q,pt in zip(z['match'],z['anchor'],z['s'],z['y'],z['q'],z['pt']):
    k=(str(m),tuple(a.tolist()));assert k not in r
    r[k]=dict(co=co,s=int(s),y=int(y),q=float(q),pt=float(pt))
 return r
ref=rows('ref')
for arm in ['R1400','R1800','I2000','I4264','no_shop']:
 r=rows(arm);common=ref.keys()&r.keys();same=[k for k in common if ref[k]['co']==r[k]['co']]
 d=dict(common_TS_anchors=len(common),reference_only=len(ref.keys()-r.keys()),alternative_only=len(r.keys()-ref.keys()),scale_changes=sum(ref[k]['co']!=r[k]['co'] for k in common),label_changes=sum(ref[k]['y']!=r[k]['y'] for k in common),same_scale={})
 for co in ['T','S']:
  kk=sorted(k for k in same if ref[k]['co']==co)
  diffs=[((r[k]['q']-r[k]['y'])**2-(r[k]['pt']-r[k]['y'])**2)-((ref[k]['q']-ref[k]['y'])**2-(ref[k]['pt']-ref[k]['y'])**2) for k in kk]
  means=pd.Series(diffs).groupby([k[0] for k in kk]).mean().to_numpy();rng=np.random.default_rng(7);boot=[means[rng.integers(0,len(means),len(means))].mean() for _ in range(2000)]
  d['same_scale'][co]=dict(n=len(kk),matches=len(means),delta_of_contrasts=float(means.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
 out['definition_changes'][arm]=d
before=json.loads((OUT/'frozen_hashes_before.json').read_text());ck('15 frozen files preserved',all(sha(REPO/p)==v for p,v in before.items()))
out['frozen_files_checked']=len(before)
dump('verification.json',out)
print('checks',out['checks'],'changes',out['definition_changes'])
