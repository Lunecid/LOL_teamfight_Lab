from t025_common import *
import joblib
import fc20260915_data as D,fc20260915_common as C
from v_redesign_feature_adapters import match_holdout_mask
r=D.load_v_rows(D.Layout(False),'MAIN',[f'fold{k}' for k in range(5)],bucket_only=True,with_X=False)
g=r['match'].astype(str);sr=r['sub_role'];stop0=match_holdout_mask(g,.15,seed=7)
out=provenance();out['folds']=[]
for k in range(5):
 held=sr==f'fold{k}';fit=~held&~stop0;stop=~held&stop0
 obj=joblib.load(REPO/f'outputs/q_newv_fit85_20260920/oof_evaluators/V_oof_fold{k}_mlp_expanded.joblib')
 counts=dict(n_fit_rows=int(fit.sum()),n_stop_rows=int(stop.sum()),n_held_matches=len(np.unique(g[held])))
 out['folds'].append(dict(fold=k,counts=counts,saved_counts={x:int(obj[x]) for x in counts},counts_match=all(counts[x]==obj[x] for x in counts),held_fit_overlap=len(set(g[held])&set(g[fit])),held_stop_overlap=len(set(g[held])&set(g[stop])),held_set_sha256=hashlib.sha256('\n'.join(sorted(set(g[held]))).encode()).hexdigest()))
out['labels']={}
for co in ['T','S']:
 with np.load(lab_dir(co)/'TRAIN_oof_h90.npz',allow_pickle=False) as z:
  out['labels'][co]=dict(keys=z.files,n=len(z['match']))
  key='fold_id' if 'fold_id' in z.files else 'fold'
  if key in z.files:out['labels'][co]['fold_hash_mismatches']=sum(int(f)!=C.train_fold(str(m)) for f,m in zip(z[key],z['match']))
out['scope']='Reconstruction from current extraction membership and deterministic split, checked against saved counts; historical fit membership hashes were not stored, so historical execution is only partially auditable.'
dump('integrity.json',out)
print('DONE integrity',[(x['fold'],x['counts_match']) for x in out['folds']],out['labels'])
