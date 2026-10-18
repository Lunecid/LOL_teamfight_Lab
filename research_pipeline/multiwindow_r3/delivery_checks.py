from pathlib import Path
import json,re,importlib
import numpy as np
from validate import sha,dump
from holdout_r3.split import fit_role_names, heldout_role_names
HERE=Path(__file__).resolve().parent;RUN=HERE/'production_01'
def main():
 import dataset
 checks={}
 fit_roles=sorted(fit_role_names());heldout_roles=sorted(heldout_role_names())
 if not fit_roles or not heldout_roles:raise ValueError('Local cohort policy must define FIT and held-out roles')
 for L in (1,3,5):
  x,y,m=next(dataset.iter_shards(RUN,[fit_roles[0]],L=L));assert x['player_raw'].shape[1]==L and x['history_ms'].shape[1]==L
  assert not {'selection_supported','endpoint_ms','first_kill_ms','end_ms','winning_team','net_gold','component_id','match_id'}&set(x)
  assert np.array_equal(y['economic_supported'],np.isfinite(y['net_gold']));assert (y['economic_class'][~y['economic_supported']]==-1).all();checks[f'loader_history_L{L}']=True
 for role in fit_roles[1:3]:
  x,y,m=next(dataset.iter_shards(RUN,[role]));assert (m['split_role']==role).all();checks['loader_'+role]=True
 try:next(dataset.iter_shards(RUN,[heldout_roles[0]]))
 except ValueError:checks['holdout_request_rejected']=True
 else:raise AssertionError('holdout request accepted')
 out=RUN/'review';manifest=json.loads((out/'MANIFEST.json').read_text(encoding='utf-8'));assert all(sha(out/p)==x['sha256'] for p,x in manifest['files'].items());checks['review_hashes']=len(manifest['files'])
 links=re.findall(r'(?:src|href)="([^"]+)',(out/'index.html').read_text(encoding='utf-8'));assert all((out/p).exists() for p in links if not p.startswith('#'));checks['local_links']=len(links)
 r=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'));n=r['queries'];assert n==596785
 for h,vals in r['labels'].items():
  for category in ('economic','terminal','pareto'):assert sum(vals[category].values())==n
  for e,v in r['rare_events'][h].items():assert v['positive_windows']<=v['supported_windows'] and v['supported_windows']+v['unsupported_windows']==n and v['unique_matches_with_positive_window']<=v['positive_windows']
 checks['label_denominators']=True
 result={'pass':True,'checks':checks,'dataset_sha256':sha(HERE/'dataset.py'),'validation_sha256':sha(RUN/'VALIDATION.json'),'training_performed':False}
 dump(RUN/'LOADER_AND_DELIVERY_CHECKS.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
