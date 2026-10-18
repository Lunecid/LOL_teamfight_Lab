"""One read-only validator alongside three production workers; completed shards only."""
from pathlib import Path
import json,time,argparse
from validate import check_shard,dump,ROOT
def main(run):
 rows=[]
 for line in (ROOT/'holdout_r3/audit_01/MATCH_AUDIT.jsonl').open(encoding='utf-8'):
  r=json.loads(line)
  if r['disposition'].startswith('RETAIN'):rows.append(r)
 done=set();n=0;start=time.time()
 while True:
  pp=run/'PROGRESS.json'
  try:progress=json.loads(pp.read_text(encoding='utf-8'))
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
  full=(run/'BUILD_SUMMARY.json').exists();count=progress['matches_processed'];nb=(count+127)//128 if full else count//128
  for index in range(nb):
   if index in done:continue
   path=run/'shards'/f'{index:04d}.npz'
   if path.exists():
    batch=rows[index*128:(index+1)*128];ck,part,h=check_shard((path,batch));n+=ck.get('raw_query_recalculations',0)
   done.add(index)
   if len(done)%10==0 or full:
    status={'batch_checkpoints':len(done),'raw_queries_checked':n,'production_matches_seen':count,'elapsed_s':round(time.time()-start,1)};dump(run/'RAW_CHECKPOINT_PROGRESS.json',status);print(json.dumps(status),flush=True)
  if full:break
  time.sleep(2)
 dump(run/'RAW_CHECKPOINT_COMPLETE.json',{'completed':True,'batch_checkpoints':len(done),'raw_queries_checked':n,'full_validation_still_required':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run',type=Path);main(a.parse_args().run)
