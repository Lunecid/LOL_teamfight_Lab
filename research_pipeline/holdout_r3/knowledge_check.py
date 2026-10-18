from pathlib import Path
import json,hashlib
from split import patches
H=Path(__file__).resolve().parent;K=H.parent/'multiwindow_r2/knowledge_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((K/'INDEX.json').read_text(encoding='utf-8'));out=[]
for entry in index['bundles']:
 if entry['patch'] not in patches():continue
 p=K/entry['path'];assert sha(p)==entry['sha256'];b=json.loads(p.read_text(encoding='utf-8'));sources=[]
 for s in b['sources']:
  f=p.parent/s['path'];assert sha(f)==s['sha256'];v=json.loads(f.read_text(encoding='utf-8'));assert v['version']==s['version'];sources.append({'path':str(f),'sha256':sha(f),'version':v['version']})
 out.append({'patch':entry['patch'],'bundle_sha256':sha(p),'sources':sources,'semantic_claims':entry['semantic_claims'],'readiness':b['readiness'],'objectives':b['objectives']})
(H/'KNOWLEDGE_CHECK.json').write_text(json.dumps({'source_hashes_and_versions_pass':True,'bundles':out,'K3_enabled':False,'terrain_enabled':False,'exact_buff_holders_enabled':False,'note':'Hash/version checks are provenance checks, not independent semantic or numeric correctness certification.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(f"Verified {len(out)} configured bundle hashes and source hashes/versions; semantic/terrain features remain off.")
