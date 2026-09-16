"""Validate document integrity, local links, references and required surfaces."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BUNDLE=ROOT/'docs/tog_delta_v_20260916'
expected=['README.md','manuscript.md','reviewer_response_matrix.md',
          'collaborator_specification.md','claim_evidence_ledger.md','references.bib']
checks=[]
def check(name, ok, detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))

texts={}
for name in expected:
    p=BUNDLE/name
    check('exists/'+name,p.is_file())
    if not p.is_file():continue
    text=p.read_text(encoding='utf-8')
    texts[name]=text
    check('populated/'+name,len(text)>500)
    check('conflict_markers/'+name,not re.search(r'^(<<<<<<<|>>>>>>>|=======$)',text,re.M))
    check('unicode_integrity/'+name,'\ufffd' not in text)
    if p.suffix=='.md':
        expected_columns=None
        for line_no,line in enumerate(text.splitlines(),1):
            if line.startswith('|'):
                columns=len(re.findall(r'(?<!\\)\|',line))-1
                if expected_columns is None:expected_columns=columns
                check(f'table_columns/{name}/{line_no}',columns==expected_columns,
                      {'actual':columns,'expected':expected_columns})
            else:expected_columns=None
        for i,m in enumerate(re.finditer(r'\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)',text)):
            target=m.group(1).strip('<>')
            if re.match(r'^(https?://|mailto:|#)',target):continue
            target=unquote(target.split('#')[0])
            target=re.sub(r':\d+$','',target)
            if not target:continue
            path=Path(target)
            if not path.is_absolute():path=p.parent/path
            check(f'local_link/{name}/{i}',path.exists(),target)

bib=texts.get('references.bib','')
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bib))
used=set()
for name,text in texts.items():
    if not name.endswith('.md'):continue
    for key in set(re.findall(r'\[([A-Z][A-Za-z]+20\d\d)\]',text)):
        used.add(key)
        check(f'citation/{name}/{key}',key in keys)
check('six_audited_references',keys=={'Kim2020','Maymin2021','Hodge2021','Halfaker2015','Lundberg2017','Schubert2016'})
check('bibliography_used',keys<=used)
summary={'scope':'Document presence, encoding, links and reference-key integrity; semantic/numeric review recorded separately',
         'checks':len(checks),'passed':sum(x['passed'] for x in checks),'failed':[x for x in checks if not x['passed']]}
data={'summary':summary,'files':{name:hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() for name in texts},'checks':checks}
(OUT/'document_validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True))
raise SystemExit(bool(summary['failed']))
