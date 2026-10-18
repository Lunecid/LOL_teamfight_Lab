"""Merge source-inspected extraction shards and validate every entry/quote.
Does not call an LLM, train a model, or read any match data.
"""
from pathlib import Path
import json,hashlib,collections,copy,re,html,shutil
R=Path(__file__).parent; V1=R.parent/'knowledge_15_14_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
DIMS={'damage','cc','mobility','defense','sustain','targeting','area','summon','resource','scaling','exception','vision','attack_modifier'}
CANONICAL_ALIASES={
 'exception.magic_resist_reduction':'debuff.magic_resist_reduction',
 'defense.magic_resist_reduction':'debuff.magic_resist_reduction',
 'attack_modifier.magic_resist_reduction':'debuff.magic_resist_reduction',
 'exception.armor_reduction':'debuff.armor_reduction',
 'defense.armor_reduction':'debuff.armor_reduction',
 'attack_modifier.armor_reduction':'debuff.armor_reduction',
 'exception.attack_damage_reduction':'debuff.attack_damage_reduction',
 'exception.healing_reduction':'debuff.healing_reduction',
 'sustain.healing_reduction':'debuff.healing_reduction',
 'exception.damage_output_reduction':'debuff.damage_output_reduction',
 'exception.damage_amplification':'damage.amplification',
 'cc.knockup':'cc.airborne','cc.knock_up':'cc.airborne',
 'mobility.movement_speed':'mobility.move_speed'}
CANONICAL_ALIASES.update({
 'defense.resistance_reduction':'debuff.resistance_reduction',
 'defense.shield_break':'debuff.shield_break',
 'attack_modifier.shield_break':'debuff.shield_break',
 'attack_modifier.attack_speed_reduction':'debuff.attack_speed_reduction',
 'attack_modifier.heal_reduction':'debuff.healing_reduction',
 'damage.repeated':'damage.repeat','damage.untyped':'damage.unspecified',
 'sustain.life_steal':'sustain.lifesteal',
 'sustain.heal_amplification':'sustain.healing_amplification',
 'scaling.champion_level':'scaling.level',
 'scaling.critical_strike_chance':'scaling.critical_chance',
 'scaling.charge_time':'scaling.charge_duration',
 'scaling.champions_hit':'scaling.champion_hits',
 'area.persistent_zone':'area.persistent',
 'attack_modifier.attack_range':'attack_modifier.range'})
DIMS.add('debuff')
def source_text(doc,locator):
 v=doc
 for raw in locator.lstrip('/').split('/'):
  k=raw.replace('~1','/').replace('~0','~');v=v[int(k)] if isinstance(v,list) else v[k]
 return html.unescape(re.sub('<[^>]*>',' ',v))
def validate_record(r,corpus):
 assert r['skill_key'] in corpus,r['skill_key']
 assert r['review_scope']=='DD_DESCRIPTION_AND_TOOLTIP'
 assert set(r['unknown_dimensions'])<=DIMS
 for c in r['claims']:
  assert c['dimension'] in DIMS,c
  assert re.fullmatch('[a-z0-9_.]+',c['tag']),c['tag']
  assert c['tag'].split('.')[0]==c['dimension']
  assert c['source_field'] in ['description','tooltip']
  assert isinstance(c['quote'],str) and len(c['quote'].strip())>=3
  assert c['quote'] in corpus[r['skill_key']][c['source_field']],(r['skill_key'],c['quote'])
  assert c['assertion'] in ['SOURCE_EXPLICIT','SOURCE_CONDITIONAL','AMBIGUOUS']
  assert c['current_state']=='UNKNOWN_FROM_CURRENT_SCHEMA'
  assert isinstance(c['runtime_dependencies'],list)
  assert c['condition'] is None or isinstance(c['condition'],str)
  if c['assertion']=='SOURCE_CONDITIONAL':assert c['condition'],r['skill_key']

def main():
 corpus={r['skill_key']:r for r in read(V1/'ability_corpus.json')}
 rows=[];exceptions=[];shard_meta=[]
 for name in 'ABC':
  path=R/'shards'/f'shard_{name}.json';s=read(path)
  assert len(s['records'])==285,(name,len(s['records']))
  expected_champs=sorted({k.rsplit('.',1)[0] for k in corpus})[{'A':0,'B':57,'C':114}[name]:{'A':57,'B':114,'C':171}[name]]
  expected={k for k in corpus if k.rsplit('.',1)[0] in expected_champs}
  assert {r['skill_key'] for r in s['records']}==expected,name
  for r in s['records']:validate_record(r,corpus)
  rows.extend(s['records']);exceptions.extend(s['exceptions']);shard_meta.append(dict(shard=name,path=str(path.relative_to(R)),sha256=sha(path),producer=s['producer'],notes=s['notes']))
 assert len(rows)==855 and len({r['skill_key'] for r in rows})==855
 assert {r['skill_key'] for r in rows}==set(corpus)
 normalizations=[]
 for r in rows:
  for c in r['claims']:
   if c['tag'] in CANONICAL_ALIASES:
    old=c['tag'];c['source_extraction_tag']=old;c['tag']=CANONICAL_ALIASES[old];c['dimension']=c['tag'].split('.')[0]
    normalizations.append(dict(skill_key=r['skill_key'],source_tag=old,canonical_tag=c['tag']))
  r['unknown_dimensions']=sorted(DIMS-{c['dimension'] for c in r['claims'] if c['assertion']!='AMBIGUOUS'})
 for e in exceptions:e['source_claim_tags']=sorted({CANONICAL_ALIASES.get(t,t) for t in e['source_claim_tags']})
 write(R/'semantic_normalization.json',dict(version='canonical-tags-v2',aliases=CANONICAL_ALIASES,changes=normalizations,reason='Unify terminology; distinguish offensive debuffs from defensive protection. Original shards retained unchanged.'))
 rows.sort(key=lambda r:r['skill_key'])
 locations=read(V1/'source_locations.json');source_docs={}
 rawbundle=read(V1/'bundle.json');sources={s['source_id']:s for s in rawbundle['sources']}
 claims=[];tag_counts=collections.Counter();status_counts=collections.Counter()
 for r in rows:
  row=corpus[r['skill_key']]
  for j,c in enumerate(r['claims']):
   evidence=next(e for e in row['evidence'] if e['locator'].endswith('/'+c['source_field']))
   sid=evidence['source_id']
   if sid not in source_docs:
    path=V1/locations[sid];assert sha(path)==sources[sid]['sha256'];source_docs[sid]=read(path)
   assert source_text(source_docs[sid],evidence['locator'])==row[c['source_field']],r['skill_key']
   c['claim_id']=r['skill_key']+':'+str(j)
   c['evidence']=[dict(evidence,source_sha256=sources[sid]['sha256'],normalized_quote=c['quote'])]
   c['source_quote_check']='PASS_EXACT_SUBSTRING'
   c['independent_semantic_review']='NOT_DONE'
   claims.append(c);tag_counts[c['tag']]+=1;status_counts[c['assertion']]+=1
 for i,e in enumerate(exceptions):
  e['exception_id']='exception.'+str(i)
  assert e['skill_keys'] and set(e['skill_keys'])<=set(corpus)
  assert all(k.rsplit('.',1)[0]==e['champion'] for k in e['skill_keys']),e
  assert e['engine_policy']=='UNKNOWN_UNLESS_OBSERVED'
  available_tags={c['tag'] for r in rows if r['skill_key'] in e['skill_keys'] for c in r['claims']}
  assert set(e['source_claim_tags'])<=available_tags,(e['champion'],set(e['source_claim_tags'])-available_tags)
  e['evidence_skill_refs']=e['skill_keys']
  e['independent_semantic_review']='NOT_DONE'
 write(R/'ability_semantics.json',dict(version='ability-semantics-v2',patch='15.14',build='15.14.1',scope='All 855 DD primary catalog entries; not all server scripts/alternate forms',producer_manifest=shard_meta,records=rows))
 write(R/'exception_registry.json',dict(version='exception-registry-v2',patch='15.14',records=exceptions,policy='Source-known mechanism != observed current activation. Do not enable executable exception rules merely from tag presence.'))
 vocab=sorted(tag_counts)
 write(R/'semantic_feature_catalog.json',dict(version='semantic-feature-catalog-v2',dimensions=sorted(DIMS),vocabulary=vocab,tag_counts=dict(sorted(tag_counts.items())),encoding=dict(present_supported=[1,1],absent_or_unstated=[0,0],ambiguous=[0,0],explicitly_absent='requires explicit negative evidence; not inferred by this extractor',conditional='capability tag with condition text/dependencies, never current-active=1',independent_review_mask='separate NOT_DONE status; not silently ACCEPTED'),records=[dict(skill_key=r['skill_key'],supported_capability_tags=sorted({c['tag'] for c in r['claims'] if c['assertion']!='AMBIGUOUS'}),ambiguous_tags=sorted({c['tag'] for c in r['claims'] if c['assertion']=='AMBIGUOUS'}),runtime_current_activation=None) for r in rows],embedding_vectors_generated=False))
 write(R/'observability_contract.json',dict(version='ability-observability-v2',static_known=['champion identity','source-stated skill capabilities','conditional capabilities and exception dependencies','raw cooldown/cost/range arrays'],prefix_observable_if_supported=['LEVEL_UP-derived level','SKILL_LEVEL_UP-derived skill rank','observed death event','inventory transactions with complete undo semantics','objective/structure event ownership'],not_directly_observed=['ability cast times','ability cooldown remaining','actual hit/CC duration','current transformation/weapon/stance','copied/stolen skill identity','current clone/pet state','remote ball/soldier/trap location','marks/stacks/charges','revive proc/target buff state','continuous champion position/HP'],rules_only_if_complete=['death/respawn interval','object spawn/expiry','buff holder eligibility and loss'],policy='Absent dynamic telemetry remains UNKNOWN; no inference from future events, final items, or LLM world knowledge. Candidate rules are not activated by this package.'))
 bundle=copy.deepcopy(rawbundle);bundle['schema_version']='patch-knowledge-v2';bundle['manifest']['bundle_id']='sr-15.14-dd15.14.1-v2-full-semantics';bundle['manifest']['compiler']['version']='integrate-semantics-v2';bundle['manifest']['package_status']='PARTIALLY_SUPPORTED'
 bykey={r['skill_key']:r for r in rows}
 for ch in bundle['champions']:
  for sk in ch['skills']:
   sk['mechanic_facts']=[]
   for j,c in enumerate(bykey[sk['skill_key']]['claims']):
    sk['mechanic_facts'].append(dict(key='semantic.'+c['tag']+'.'+str(j),value=True if c['assertion']!='AMBIGUOUS' else None,unit='capability_not_activation',axis='scalar',status='UNVERIFIED' if c['assertion']!='AMBIGUOUS' else 'UNKNOWN',evidence=[{k:e[k] for k in ['source_id','locator']} for e in c['evidence']],reason='Source support='+c['assertion']+'; condition='+str(c['condition'])+'; independent review NOT_DONE; full dependencies in ability_semantics.json.'))
 bundle['ontology']['fact_keys']=sorted(set(bundle['ontology']['fact_keys'])|{f['key'] for ch in bundle['champions'] for sk in ch['skills'] for f in sk['mechanic_facts']})
 bundle['semantics']=dict(entries='ability_semantics.json',exceptions='exception_registry.json',features='semantic_feature_catalog.json',observability='observability_contract.json',extraction_coverage_entries=855,independent_review_status='NOT_DONE',activation_policy='runtime_policy.json',base_bundle_sha256=sha(V1/'bundle.json'))
 schema=read(V1/'schema.json');schema['properties']['schema_version']['const']='patch-knowledge-v2';schema['title']='PatchKnowledgeBundle v2: full catalog semantics overlay';schema['properties']['semantics']={'type':'object','additionalProperties':False,'properties':{k:{'type':'integer' if isinstance(v,int) else 'string'} for k,v in bundle['semantics'].items()},'required':list(bundle['semantics'])};schema['required'].append('semantics')
 write(R/'schema.json',schema);write(R/'bundle.json',bundle)
 shutil.copytree(V1/'sources',R/'sources',dirs_exist_ok=True)
 shutil.copyfile(V1/'ability_corpus.json',R/'ability_corpus.json')
 shutil.copyfile(V1/'validate_package.py',R/'_base_validator.py')
 write(R/'source_locations.json',locations)
 policy=read(V1/'runtime_policy.json');policy['semantic_coverage']='ALL_855_DD_ENTRIES';policy['llm_semantics_primary_enabled']=False;policy['semantic_research_arm']='full source-supported tags available; independent agreement pending for confirmatory use';policy['current_ability_activation_from_tags']=False
 write(R/'runtime_policy.json',policy)
 # Reuse original schema/provenance validator against the extended schema.
 import importlib.util
 spec=importlib.util.spec_from_file_location('v1check',V1/'validate_package.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.ROOT=R;mod.SCHEMA=schema
 core_checks=mod.validate(bundle)
 negative=[]
 test=copy.deepcopy(rows[0]);test['claims'][0]['quote']='NOT A REAL SOURCE SENTENCE 938272'
 try:validate_record(test,corpus)
 except AssertionError:negative.append('fabricated_quote_rejected')
 else:raise AssertionError('quote validator failed')
 test=copy.deepcopy(rows[0]);test['claims'][0]['current_state']='KNOWN_ACTIVE'
 try:validate_record(test,corpus)
 except AssertionError:negative.append('runtime_activation_upgrade_rejected')
 else:raise AssertionError('runtime validator failed')
 summary=dict(status='PASS_STATIC_EXTRACTION_AND_PACKAGE_CHECKS',champions=171,skills=855,claims=len(claims),tags=len(vocab),exceptions=len(exceptions),exception_champions=len({e['champion'] for e in exceptions}),source_support_counts=dict(status_counts),empty_claim_entries=[r['skill_key'] for r in rows if not r['claims']],source_files_hashed=len(sources),champion_source_files_resolved=len(source_docs),quote_checks=len(claims),core_package_checks=core_checks,independent_semantic_review='NOT_DONE',scope='Exact quotes/source integrity/schema/coverage; not independent entailment certification or gameplay correctness',negative_checks=negative,training_run=False,match_records_scanned=0,embedding_vectors_generated=False)
 write(R/'semantic_validation_report.json',summary);print(json.dumps(summary,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
