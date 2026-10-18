"""Full shard checks, raw recalculation, original-query parity; no fitting."""
from pathlib import Path
import json,hashlib,os,time,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from reference import validate_match
from holdout_r3.split import roles as policy_roles, bucket as policy_bucket
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def check_shard(job):
 path,rows=job;aud={r['match_id']:r for r in rows};checks=Counter();per=defaultdict(list)
 receipt=path.parent.parent/'RAW_CHECKPOINTS_v2'/(path.stem+'.json');cached=None;digest=sha(path)
 if receipt.exists():
  candidate=json.loads(receipt.read_text(encoding='utf-8'))
  if candidate.get('passed') and candidate['shard_sha256']==digest and candidate['reference_sha256']==sha(HERE/'reference.py') and candidate['checker_sha256']==sha(Path(__file__)) and candidate['audit_subset_sha256']==hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest():cached=candidate
 with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
 assert set(z['match_id'])<=set(aud)
 for i in range(len(z['row_id'])):
  r={k:v[i] for k,v in z.items()};mid=str(r['match_id']);per[mid].append(r)
  assert (str(r['split_role']),str(r['macro_role']))==policy_roles(aud[mid]['patch'],policy_bucket(mid))
  assert str(r['macro_role'])=='FIT'
  for value,mask in [('player_raw','player_support'),('prefix_team_events','prefix_team_events_support'),('skill_level_up_count','skill_level_up_count_support'),('cum_team_gold','gold_supported'),('slice_team_gold','slice_gold_supported'),('cum_team_events','cum_event_supported'),('slice_team_events','slice_event_supported'),('focal_team_kills','focal_kills_supported')]:
   assert np.array_equal(np.isfinite(r[value]),r[mask]),(str(path),i,value);checks['finite_iff_support_arrays']+=1
  assert not r['direct_reward_supported'].any() and not r['remaining_value_supported'].any() and not r['actual_buff_holder_supported'].any()
  assert ((r['history_ms']<=r['t0_ms'])|(r['history_ms']<0)).all()
 if cached:checks.update(cached['raw_checks'])
 else:
  raw_checks=Counter()
  for mid,rr in per.items():raw_checks.update(validate_match(rr,aud[mid]))
  checks.update(raw_checks);receipt.parent.mkdir(exist_ok=True)
  dump(receipt,{'passed':True,'shard_sha256':digest,'reference_sha256':sha(HERE/'reference.py'),'checker_sha256':sha(Path(__file__)),'audit_subset_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'raw_checks':dict(raw_checks)})
 selected=('row_id','match_id','patch','component_id','t0_ms','tau_ms','first_kill_ms','last_kill_ms','selection_supported','net_gold','economic_sign','cum_team_events','cum_event_supported','gold_supported','terminal_class','terminal_supported','hazard_class','hazard_at_risk_supported','focal_kills_supported','pareto_class','pareto_supported','actual_elapsed_ms','endpoint_supported','soul_owned','soul_source_status')
 return dict(checks),{k:z[k] for k in selected},digest

def main(out,workers=3):
 assert 1<=workers<=3;summary=json.loads((out/'BUILD_SUMMARY.json').read_text(encoding='utf-8'));auditpath=ROOT/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl';assert sha(auditpath)==summary['source_audit_sha256']
 assert sha(out/'PRODUCER_EXECUTED.py')==summary['producer_sha256']
 audit={}
 for line in auditpath.open(encoding='utf-8'):
  r=json.loads(line)
  if r['disposition'].startswith('RETAIN'):audit[r['match_id']]=r
 sources=[json.loads(s) for s in (out/'SOURCE_LEDGER.jsonl').open(encoding='utf-8')];ids=[s['match_id'] for s in sources];assert len(ids)==len(set(ids));assert set(ids)<=set(audit)
 if summary['full_population']:assert set(ids)==set(audit)
 for s in sources:
  for key in ('detail_sha256','timeline_sha256'):assert s[key]==audit[s['match_id']][key]
 # The producing batches have 128 source matches; shards are named by batch number,
 # including batches with no queries. No inference from shard row counts is needed.
 jobs=[(out/'shards'/name,[audit[s['match_id']] for s in sources[int(Path(name).stem)*128:(int(Path(name).stem)+1)*128]]) for name in summary['shards']]
 checks=Counter();arrays=defaultdict(list);hashes={};start=time.time()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for i,(ck,part,h) in enumerate(pool.map(check_shard,jobs,chunksize=1)):
   checks.update(ck);hashes[summary['shards'][i]]=h
   for k,v in part.items():arrays[k].append(v)
   if i%20==0 or i==len(jobs)-1:
    progress={'shards_checked':i+1,'shards_total':len(jobs),'raw_queries_checked':checks['raw_query_recalculations'],'elapsed_s':round(time.time()-start,1)};dump(out/'VALIDATION_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
 data={k:np.concatenate(v) for k,v in arrays.items()};n=len(data['row_id']);assert n==summary['n_queries'];assert len(set(data['row_id']))==n
 querycounts=Counter(data['match_id']);assert all(querycounts[s['match_id']]==s['n_queries'] for s in sources)
 assert sum(data['selection_supported'])==summary['selection_supported_queries']
 # Full prior r2 FIT parity by match identity, component ID and exact boundaries.
 newkeys={(str(data['match_id'][i]),int(data['component_id'][i]),int(data['t0_ms'][i]),int(data['tau_ms'][i]),int(data['first_kill_ms'][i]),int(data['last_kill_ms'][i])) for i in range(n)}
 oldkeys=set();oldmids=set()
 oldout=ROOT/'multiwindow_r2/production_01';oldsummary=json.loads((oldout/'BUILD_SUMMARY.json').read_text(encoding='utf-8'))
 for name in oldsummary['shards']:
  with np.load(oldout/'shards'/name,allow_pickle=False) as z:
   use=np.flatnonzero(np.char.startswith(z['split_role'],'FIT_'))
   for i in use:
    mid=str(z['match_id'][i])
    if mid in ids_set(ids):
     oldmids.add(mid);oldkeys.add((mid,int(z['component_id'][i]),int(z['t0_ms'][i]),int(z['tau_ms'][i]),int(z['first_kill_ms'][i]),int(z['last_kill_ms'][i])))
 expectednew={k for k in newkeys if k[0] in oldmids};missing=oldkeys-expectednew;extra=expectednew-oldkeys
 assert not missing and not extra,('prior query mismatch',len(missing),len(extra))
 before=json.loads((out/'ORIGINAL_HASHES_BEFORE.json').read_text(encoding='utf-8'));changed=[p for p,h in before.items() if sha(p)!=h];assert not changed
 np.savez_compressed(out/'VALIDATED_DISTRIBUTIONS.npz',**data)
 result={'pass':True,'full_population':summary['full_population'],'matches_covered':len(ids),'matches_with_queries':len(querycounts),'queries_checked':n,'checks':dict(checks),'prior_r2_fit_queries_exactly_preserved':len(oldkeys),'prior_r2_fit_query_matches':len(oldmids),'prior_query_missing':len(missing),'prior_query_extra':len(extra),'retained_sources_without_queries':len(ids)-len(querycounts),'original_files_changed':changed,'shard_sha256':hashes,'source_ledger_sha256':sha(out/'SOURCE_LEDGER.jsonl'),'build_summary_sha256':sha(out/'BUILD_SUMMARY.json'),'distributions_sha256':sha(out/'VALIDATED_DISTRIBUTIONS.npz'),'validator_sha256':sha(Path(__file__)),'reference_sha256':sha(HERE/'reference.py'),'training_performed':False,'scope':'Every emitted query rechecked against raw source windows; no-queries sources covered by production source hashes and counts. Static skills, exact buffs, terrain, causal values remain outside support.'}
 dump(out/'VALIDATION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='shard_sha256'},ensure_ascii=False))

_IDS=None
def ids_set(ids):
 global _IDS
 if _IDS is None:_IDS=set(ids)
 return _IDS

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();main(a.out,a.workers)
