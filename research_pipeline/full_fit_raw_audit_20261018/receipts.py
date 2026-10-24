from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify_aggregation():
 out=HERE/'review_02'; marker=out/'AGGREGATION_COMPLETE.json'
 r=json.loads(marker.read_text(encoding='utf-8'));assert r['complete']
 for name,h in r['inputs'].items():assert digest(HERE/name)==h,name
 for name,h in r['outputs'].items():assert digest(out/name)==h,name
 return digest(marker)
