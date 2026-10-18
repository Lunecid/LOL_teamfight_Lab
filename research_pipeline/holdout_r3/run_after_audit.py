"""Bounded local pipeline; three production workers plus one raw validator."""
from pathlib import Path
import subprocess,sys,json,time
H=Path(__file__).resolve().parent;O=H/'production_01'
def main():
 while not (H/'audit_01/VALIDATION.json').exists():time.sleep(5)
 v=json.loads((H/'audit_01/VALIDATION.json').read_text(encoding='utf-8'));assert v['passed'] and v['full_population']
 assert not O.exists()
 with (H/'production_01.log').open('w',encoding='utf-8') as pl,(H/'stream_validation.log').open('w',encoding='utf-8') as vl:
  p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(H/'producer.py'),'--output',str(O),'--workers','3'],cwd=H,stdout=pl,stderr=subprocess.STDOUT)
  validator=subprocess.Popen([sys.executable,'-X','utf8','-B',str(H/'stream_validate.py'),str(O)],cwd=H,stdout=vl,stderr=subprocess.STDOUT)
  rc=p.wait()
  if rc:validator.terminate();raise RuntimeError('production failed; see production_01.log')
  assert validator.wait()==0,'stream validator failed; see log'
 with (H/'validation.log').open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,'-X','utf8','-B',str(H/'validate.py'),str(O),'--workers','3'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (H/'report.log').open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,'-X','utf8','-B',str(H/'compare.py')],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Audit, production, full raw validation and visual report completed.',flush=True)
if __name__=='__main__':main()
