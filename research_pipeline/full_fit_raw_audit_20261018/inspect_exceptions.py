import os
"""Re-open every ambiguous same-time death case, not a sample of those cases."""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib
from receipts import digest
HERE=Path(__file__).resolve().parent;RUN=HERE/'full_02';RAW=Path(os.environ["LOL_RAW_MATCH_ROOT"])
def flatten_ranges(rr):return [i for a,b in rr for i in range(a,b+1)]
def main():
 assert (RUN/'RUN_COMPLETE.json').exists()
 counts=Counter();groups=[];source_checked=0;soulglobal=0;auditorerror=0;rare_reasons=Counter();records_scanned=0
 for line in (RUN/'MATCH_AUDIT.jsonl').open(encoding='utf-8'):
  r=json.loads(line);records_scanned+=1
  if r['disposition']=='AUDITOR_ERROR':auditorerror+=1
  if 'soul:both_teams_have_four_dragons' in r['event_masks']:soulglobal+=1
  if 'kills:ambiguous_same_timestamp_victim' not in r['event_masks']:continue
  mid=r['match_id'];db=(RAW/'detail'/f'{r["source_files"]["detail"][0]}.json').read_bytes();tb=(RAW/'timeline'/f'{r["source_files"]["timeline"][0]}.json').read_bytes()
  assert hashlib.sha256(db).hexdigest()==r['detail_sha256'] and hashlib.sha256(tb).hexdigest()==r['timeline_sha256'];source_checked+=1
  detail=json.loads(db);tl=json.loads(tb);events=[e for f in tl['info']['frames'] for e in f['events']]
  pmap={p['participantId']:p for p in detail['info']['participants']};gg=defaultdict(list);all_deaths=Counter();unique_deaths=defaultdict(set)
  for i,e in enumerate(events):
   if e.get('type')=='CHAMPION_KILL':all_deaths[e.get('victimId')]+=1;unique_deaths[e.get('victimId')].add(e['timestamp'])
  for i in flatten_ranges(r['event_masks']['kills:ambiguous_same_timestamp_victim']):
   e=events[i];gg[(e['timestamp'],e['victimId'])].append((i,e))
  matchchamps=set()
  for (ts,v),ee in gg.items():
   champ=pmap[v].get('championName',str(pmap[v].get('championId')));matchchamps.add(champ)
   sigs={json.dumps(e,sort_keys=True,separators=(',',':')) for _,e in ee};same=len(sigs)==1
   counts['groups']+=1;counts['group:'+champ]+=1;counts['identical_payload_groups' if same else 'different_payload_groups']+=1
   g={'match_id':mid,'patch':r['patch'],'disposition':r['disposition'],'timestamp':ts,'victim_id':v,'champion':champ,'raw_ordinals':[i for i,e in ee],'killer_ids':[e.get('killerId') for i,e in ee],'identical_payloads':same,'raw_match_deaths_for_victim':all_deaths[v],'distinct_death_timestamps_for_victim':len(unique_deaths[v]),'detail_deaths_for_victim':pmap[v].get('deaths'),'bounties':[e.get('bounty') for i,e in ee]}
   groups.append(g)
  for c in matchchamps:counts['match:'+c]+=1
 report={'ambiguous_matches_reopened':source_checked,'all_reopened_sources_hash_match':True,'counts':dict(counts),'global_soul_conflict_masks':soulglobal,'auditor_errors':auditorerror,'interpretation':'No automatic deduplication or attribution repair. Temporal boundaries/count targets crossing these ordinals must be masked or explicitly resolved in next producer. Per-match postgame death totals are audit evidence only, never runtime inputs.','groups':groups}
 report['audit_source_sha256']=digest(RUN/'MATCH_AUDIT.jsonl')
 report['completed']=True;report['records_scanned']=records_scanned;report['run_complete_sha256']=digest(RUN/'RUN_COMPLETE.json')
 p=HERE/'review_02';p.mkdir(exist_ok=True);(p/'AMBIGUOUS_KILL_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='groups'},ensure_ascii=False))
if __name__=='__main__':main()
