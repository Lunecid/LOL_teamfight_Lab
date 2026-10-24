"""Hand over this run's serial streaming tail to three final validator workers."""
from pathlib import Path
import json,time,subprocess,sys
import psutil
H=Path(__file__).resolve().parent;O=H/'production_01'
OWN=[(52220,1792283671.8388019,'run_after_audit.py'),(52728,1792284211.9935548,'stream_validate.py')]
def own_process(pid,started,name):
 p=psutil.Process(pid)
 assert abs(p.create_time()-started)<.01 and Path(p.cwd())==H
 assert any(Path(x).name==name for x in p.cmdline())
 return p
def main():
 # Validate identity before waiting; never stop the producer or an unrelated process.
 for spec in OWN:own_process(*spec)
 while True:
  try:
   s=json.loads((O/'BUILD_SUMMARY.json').read_text(encoding='utf-8'))
   if s['full_population']:break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(2)
 assert s['audit_retained_matches']==s['counts']['matches_processed']
 stopped=[]
 for spec in OWN:
  try:p=own_process(*spec);p.terminate();p.wait(timeout=10);stopped.append({'pid':spec[0],'script':spec[2]})
  except psutil.NoSuchProcess:pass
 incomplete=[]
 for p in (O/'RAW_CHECKPOINTS_v2').glob('*.json'):
  try:json.loads(p.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
   dst=p.with_suffix('.incomplete');assert not dst.exists();p.rename(dst);incomplete.append(dst.name)
 (O/'VALIDATOR_HANDOVER.json').write_text(json.dumps({'intentional':True,'reason':'After successful full production, use three final validators instead of waiting for the serial streaming tail. Valid hash-bound receipts reused; interrupted receipts preserved and recomputed.','stopped_own_helpers':stopped,'incomplete_receipts_preserved':incomplete,'producer_stopped':False,'maximum_compute_workers':4},indent=2),encoding='utf-8')
 for script,log,args in [('validate.py','validation.log',[str(O),'--workers','3']),('compare.py','report.log',[]),('finalize.py','finalize.log',[])]:
  print('Starting',script,flush=True)
  with (H/log).open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-B',str(H/script),*args],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Full validation, comparison and packaging completed.',flush=True)
if __name__=='__main__':main()
