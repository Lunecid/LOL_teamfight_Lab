"""Validate the exported package with only the Python standard library."""
from pathlib import Path
import json,hashlib,importlib.util
R=Path(__file__).parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 spec=importlib.util.spec_from_file_location('corecheck',R/'_base_validator.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 report=core.validate(load(R/'bundle.json'))
 spec=importlib.util.spec_from_file_location('semcheck',R/'integrate_semantics.py');sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
 corpus={r['skill_key']:r for r in load(R/'ability_corpus.json')};rows=load(R/'ability_semantics.json')['records']
 assert len(rows)==855 and {r['skill_key'] for r in rows}==set(corpus)
 n=0
 for row in rows:
  sem.validate_record(row,corpus)
  for c in row['claims']:
   e=c['evidence'][0];original=next(x for x in corpus[row['skill_key']]['evidence'] if x['locator'].endswith('/'+c['source_field']))
   assert e['source_id']==original['source_id'] and e['locator']==original['locator']
   assert e['normalized_quote']==c['quote'];n+=1
 manifest=R/'artifact_manifest.json'
 if manifest.exists():
  for rel,expected in load(manifest)['file_sha256'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==expected,rel
 print(json.dumps({'status':'PASS_EXPORT_INTEGRITY','skills':len(rows),'claims':n,'core':report,'manifest_checked':manifest.exists(),'semantic_entailment_independently_certified':False}))
if __name__=='__main__':main()
