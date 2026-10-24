from pathlib import Path
from collections import Counter
import json, hashlib, os
import numpy as np
from receipts import verify_aggregation, digest
HERE=Path(__file__).resolve().parent;RUN=HERE/'full_02';OUT=HERE/'review_02'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 receipt=verify_aggregation()
 inv=json.loads((RUN/'INVENTORY.json').read_text(encoding='utf-8'));end=json.loads((RUN/'RUN_COMPLETE.json').read_text(encoding='utf-8'));summary=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'))
 assert end['completed'] and end['all_hash_eligible_scanned'] and end['auditor_error_count']==0
 assert sha((RUN/'AUDITOR_EXECUTED.py').read_bytes())==end['script_sha256']
 ids=set();sc=Counter();pre=np.zeros(18,np.int64);post=pre.copy();masks=0;checks=Counter()
 for line in (RUN/'MATCH_AUDIT.jsonl').open(encoding='utf-8'):
  r=json.loads(line);mid=r['match_id'];assert mid not in ids;ids.add(mid)
  expect=int(sha(('rift-v5-multipatch-r1:'+mid).encode())[:16],16)%100;assert expect==r['bucket']<=79;checks['split_and_unique_matches']+=1
  sc[r['disposition']]+=1
  for key in ('detail_sha256','timeline_sha256'):
   if key in r:assert len(r[key])==64 and all(c in '0123456789abcdef' for c in r[key]);checks['source_hash_formats']+=1
  m=r['metrics']
  if 'supported_player_cells' in m:
   assert all(0<=v<=m['player_cells_per_feature'] for v in m['supported_player_cells']);pre+=m['supported_player_cells']
   if r['disposition'].startswith('RETAIN'):post+=m['supported_player_cells']
  for section,limit in [('field_masks',m.get('n_frames',0)*10),('event_masks',m.get('n_events',0))]:
   for key,rr in r[section].items():
    prev=-2
    for a,b in rr:
     assert -1<=a<=b<limit and a>prev;prev=b;masks+=b-a+1;checks['mask_ranges']+=1
  for x in r.get('rule_masks',[]):assert 0<=x['raw_ordinal']<m['n_events'] and x['timestamp']>=0;checks['timed_rule_masks']+=1
  if r['disposition'].startswith('RETAIN'):assert not r['hard_reasons'] and not r['population_reasons']
 hh=np.load(RUN/'FEATURE_HISTOGRAMS.npz')
 for j in range(18):assert hh[f'before_{j}'].sum()==pre[j] and hh[f'after_{j}'].sum()==post[j];checks['histogram_denominators']+=2
 assert dict(sc)==summary['dispositions'];checks['disposition_reconciliation']=1
 raw=Path(inv['raw_root']);inventory_unchanged=True
 for kind in ('detail','timeline'):
  with os.scandir(raw/kind) as it:names=sorted(e.name[:-5] for e in it if e.is_file() and e.name.endswith('.json'))
  same=sha('\n'.join(names).encode())==inv[kind+'_name_list_sha256'];assert same;checks['source_filename_inventory_stable']+=1
 result={'passed':True,'checks':dict(checks),'audit_records':len(ids),'mask_coordinates_inclusive_count':masks,'new_training_ledger_created':False,'original_files_moved_or_deleted':False,'note':'Full audit-record structural validation and histogram reconciliation, plus source filename snapshot and old-r2 hash comparison. Not a new training-label validation.'}
 result['aggregation_receipt_sha256']=receipt
 ex=json.loads((OUT/'AMBIGUOUS_KILL_AUDIT.json').read_text(encoding='utf-8'))
 assert ex['completed'] and ex['records_scanned']==len(ids) and ex['audit_source_sha256']==digest(RUN/'MATCH_AUDIT.jsonl') and ex['run_complete_sha256']==digest(RUN/'RUN_COMPLETE.json')
 result['exception_audit_sha256']=digest(OUT/'AMBIGUOUS_KILL_AUDIT.json')
 (OUT/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
