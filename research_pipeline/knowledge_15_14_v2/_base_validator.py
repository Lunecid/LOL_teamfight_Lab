"""Offline validation of the JSON Schema keyword subset actually used here and provenance.
Not a general Draft 2020-12 implementation or gameplay-engine acceptance test.
"""
from pathlib import Path
import json,re,hashlib,copy,html
ROOT=Path(__file__).parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
SCHEMA=load(ROOT/'schema.json')
def fail(m):raise ValueError(m)
def check(x,s,p='$'):
    if '$ref' in s:
        z=SCHEMA
        for k in s['$ref'].split('/')[1:]:z=z[k]
        check(x,z,p)
    if 'const' in s and (x!=s['const'] or type(x)!=type(s['const'])):fail(p+' const')
    if 'enum' in s and x not in s['enum']:fail(p+' enum')
    types={'object':lambda v:isinstance(v,dict),'array':lambda v:isinstance(v,list),'string':lambda v:isinstance(v,str),'number':lambda v:isinstance(v,(int,float)) and not isinstance(v,bool),'integer':lambda v:isinstance(v,int) and not isinstance(v,bool),'boolean':lambda v:isinstance(v,bool),'null':lambda v:v is None}
    if 'type' in s:
        allowed=s['type'] if isinstance(s['type'],list) else [s['type']]
        if not any(types[t](x) for t in allowed):fail(p+' type')
    if isinstance(x,dict):
        if set(s.get('required',[]))-set(x):fail(p+' required')
        if s.get('additionalProperties') is False and set(x)-set(s.get('properties',{})):fail(p+' additionalProperties')
        for k,v in s.get('properties',{}).items():
            if k in x:check(x[k],v,p+'.'+k)
    if isinstance(x,list):
        if len(x)<s.get('minItems',0):fail(p+' minItems')
        for i,v in enumerate(x):
            if 'items' in s:check(v,s['items'],p+f'[{i}]')
    if isinstance(x,str):
        if len(x)<s.get('minLength',0):fail(p+' minLength')
        if 'pattern' in s and not re.search(s['pattern'],x):fail(p+' pattern')
    if isinstance(x,(float,int)) and not isinstance(x,bool) and 'minimum' in s and x<s['minimum']:fail(p+' minimum')
    def passes(v):
        try:check(x,v,p);return True
        except ValueError:return False
    if 'anyOf' in s and not any(passes(v) for v in s['anyOf']):fail(p+' anyOf')
    for v in s.get('allOf',[]):check(x,v,p)
    if 'not' in s and passes(s['not']):fail(p+' not')
    if 'if' in s and passes(s['if']) and 'then' in s:check(x,s['then'],p)

def validate(bundle):
    check(bundle,SCHEMA)
    loc=load(ROOT/'source_locations.json'); docs={}; srcs={s['source_id']:s for s in bundle['sources']}
    assert len(srcs)==len(bundle['sources'])
    for sid,s in srcs.items():
        path=ROOT/loc[sid]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256'],sid
        docs[sid]=load(path) if path.suffix=='.json' else html.unescape(re.sub('<[^>]+>',' ',path.read_text(encoding='utf-8')))
    evidence_n=0;fact_n=0
    def walk(x):
        nonlocal evidence_n,fact_n
        if isinstance(x,dict):
            if set(x)=={'source_id','locator'}:
                evidence_n+=1;assert x['source_id'] in srcs,x
                data=docs[x['source_id']];p=x['locator']
                if p.startswith('text:'):assert p[5:] in data,p
                else:
                    assert p.startswith('/'),p
                    for raw in p[1:].split('/'):
                        key=raw.replace('~1','/').replace('~0','~')
                        data=data[int(key)] if isinstance(data,list) else data[key]
            if 'key' in x and 'evidence' in x:
                fact_n+=1
                if x['key'].startswith('raw.') or x['key'].startswith('raw_cdragon.'):
                    assert x['evidence']
                    r=x['evidence'][0];v=docs[r['source_id']]
                    for raw in r['locator'][1:].split('/'):
                        k=raw.replace('~1','/').replace('~0','~');v=v[int(k)] if isinstance(v,list) else v[k]
                    assert x['value']==v, x['key']
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(bundle)
    assert len(bundle['champions'])==171
    assert len({c['champion_id'] for c in bundle['champions']})==171
    index=load(ROOT/'sources/ddragon/champion.json')['data']
    assert {int(c['key']) for c in index.values()}=={c['champion_id'] for c in bundle['champions']}
    for c in bundle['champions']:
        assert [s['slot'] for s in c['skills']]==['PASSIVE','Q','W','E','R']
    ruleids={r['rule_id'] for r in bundle['rules']};entityids={e['entity_id'] for e in bundle['world']};statekeys=set(bundle['ontology']['state_keys'])
    for e in bundle['world']:
        assert set(e['rule_ids'])<=ruleids
        assert set(e['related_entity_ids'])<=entityids
    for r in bundle['rules']:
        assert set(r['state_dependencies']+r['outputs'])<=statekeys
        assert {g['state_key'] for g in r['guards']}<=set(r['state_dependencies'])
        assert r['operation'] in bundle['ontology']['supported_operations']
    assert bundle['manifest']['match_patch']=='15.14'
    assert not bundle['semantic_artifacts']
    return dict(evidence_references_checked=evidence_n,facts_checked=fact_n,source_hashes_checked=len(srcs))

def main():
    b=load(ROOT/'bundle.json');result=validate(b);negative=[]
    cases={
      'verified_null':lambda x:x['champions'][0]['base_facts'][0].update(value=None),
      'unknown_with_numeric_value':lambda x:x['champions'][0]['base_facts'][0].update(status='UNKNOWN',value=1),
      'missing_source':lambda x:x['champions'][0]['base_facts'][0]['evidence'][0].update(source_id='missing'),
      'tampered_raw_value':lambda x:x['champions'][0]['base_facts'][0].update(value=-999),
      'unknown_state_dependency':lambda x:x['rules'][0]['state_dependencies'].append('unknown_state'),
      'dangling_world_rule':lambda x:x['world'][0]['rule_ids'].append('missing_rule')}
    for name,mut in cases.items():
        bad=copy.deepcopy(b);mut(bad)
        try:validate(bad)
        except (AssertionError,ValueError,KeyError):negative.append(name)
        else:raise AssertionError('Failed to reject '+name)
    d=load(ROOT/'unresolved_fields.json');q=load(ROOT/'semantic_extraction_queue.json')
    result.update(status='PASS_STATIC_PACKAGE_CHECKS',schema_validation_scope='All schema keywords used by this schema checked by local subset validator; not general Draft2020-12 certification',negative_cases_rejected=negative,counts=dict(champions=len(b['champions']),ability_entries=sum(len(c['skills']) for c in b['champions']),items=len(b['items']),runes=len(b['runes']),summoner_spells=len(b['summoner_spells']),world_entries=len(b['world']),rule_candidates=len(b['rules']),semantic_seed_entries=sum(x['status']=='CANDIDATE_SEED_REVIEW_REQUIRED' for x in q),semantic_pending_entries=sum(x['status']=='PENDING_SEMANTIC_REVIEW' for x in q),entries_with_unresolved_tokens=sum(bool(x['unresolved_tokens']) for x in d),entries_with_all_zero_effect_fields=sum(x['effect_all_zero'] for x in d)),not_tested=['gameplay engine correctness','actual match feature reconstruction','payout formula validity','semantic tag reviewer agreement','model predictive performance','causal effect identification','complete patch/hotfix overlay'])
    (ROOT/'validation_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
