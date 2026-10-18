"""Bounded adapter parity and raw target check before the full build."""
from pathlib import Path
import json,ast
import producer as p
from reference import validate_match
from split import roles,bucket
H=Path(__file__).resolve().parent
for name in ('producer.py','audit_core.py','audit_run.py','reference.py','validate.py','compare.py','dataset.py'):
 ast.parse((H/name).read_text(encoding='utf-8'))
rows=[json.loads(s) for s in p.AUDIT.read_text(encoding='utf-8').splitlines() if s.strip()]
checked=0;queries=0
for row in rows:
 if not row['disposition'].startswith('RETAIN'):continue
 row['row_index']=checked;rr,src=p.process(row)
 if not rr:continue
 validate_match(rr,row)
 assert all((x['split_role'],x['macro_role'])==roles(row['patch'],bucket(row['match_id'])) for x in rr)
 checked+=1;queries+=len(rr)
 if checked==5:break
assert checked==5
p.dump(H/'SMOKE.json',{'passed':True,'matches':checked,'queries_raw_recomputed':queries,'scope':'bounded STOP smoke; full validation required'})
print('Smoke passed:',checked,'matches,',queries,'queries.')
