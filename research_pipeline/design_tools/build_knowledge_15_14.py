import os
"""Offline, deterministic static-knowledge build. Never reads matches or trains models."""
from pathlib import Path
import json, hashlib, re, shutil, datetime, html

BASE=Path(__file__).parent
ROOT=BASE/'knowledge_15_14_v1'
LOCAL=Path(os.environ["LOL_RESEARCH_ROOT"]) / 'worktrees' / 'engagement-state-value' / 'config' / 'game_rules' / 'objective_rules.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(s): return str(s).replace('~','~0').replace('/','~1')
def ptr(*s): return '/'+ '/'.join(esc(x) for x in s)
def clean(s): return html.unescape(re.sub('<[^>]*>',' ',s))

def main():
    sources=[]; locations={}; receipts=[]
    for filename in ['collection_manifest.json','supplement_collection_manifest.json','cdragon_directory_manifest.json','cdragon_data_manifest.json','coverage_supplement_manifest.json']:
        m=read(ROOT/filename); receipts.extend(m if isinstance(m,list) else m['receipts'])
    for r in receipts:
        if r.get('status')!=200: continue
        rel=Path(r['path']); sid=re.sub('[^A-Za-z0-9_.:-]','_',rel.as_posix())
        cls='RIOT_DDRAGON' if 'ddragon' in str(rel) else 'RIOT_PATCH_NOTE' if 'riot' in str(rel) else 'COMMUNITYDRAGON'
        sources.append(dict(source_id=sid,source_class=cls,uri=r['url'],build='15.14.1' if cls=='RIOT_DDRAGON' else '15.14' if cls=='COMMUNITYDRAGON' else None,sha256=r['sha256'],retrieved_at=r['retrieved_at'],published_at=None))
        locations[sid]=rel.as_posix()
    localdest=ROOT/'sources/local/objective_rules.json'; localdest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(LOCAL,localdest)
    ls='local_objective_registry'; locations[ls]='sources/local/objective_rules.json'
    sources.append(dict(source_id=ls,source_class='LOCAL_REGISTRY',uri=str(LOCAL),build=None,sha256=sha(localdest),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),published_at=None))
    def sid(path):
        return next(k for k,v in locations.items() if v==path)
    def ref(s,p): return dict(source_id=s,locator=p)
    def fact(k,v,s=None,p=None,unit='category',axis='scalar',status='VERIFIED',reason=None):
        return dict(key=k,value=v,unit=unit,axis=axis,status=status,evidence=[ref(s,p)] if s else [],reason=reason)
    def missing(k,reason):return fact(k,None,status='UNKNOWN',reason=reason)
    schema=read(BASE/'PATCH_KNOWLEDGE_SCHEMA_V1.json')
    schema['title']='PatchKnowledgeBundle v1.1: actual 15.14 candidate'
    schema['properties']['schema_version']['const']='patch-knowledge-v1.1'
    schema['$defs']['source']['properties']['source_class']['enum'].append('LOCAL_REGISTRY')
    write(ROOT/'schema.json',schema)
    bundle=dict(schema_version='patch-knowledge-v1.1',manifest=dict(bundle_id='sr-15.14-dd15.14.1-v1',match_patch='15.14',public_patch='25.14',game_mode='SUMMONERS_RIFT',region_scope=['KR'],ddragon_build='15.14.1',cdragon_build='15.14',knowledge_cutoff=None,knowledge_time_policy='RETROSPECTIVE_VERSION_MATCHED',engine_interface_version='rift-state-v1-proposal',ontology_version='rift-knowledge-v1.1',geometry_version=None,compiler=dict(version='build-knowledge-15.14-v1',parser_version='stdlib-json-v1',llm_model=None,prompt_hash=None),package_status='PARTIALLY_SUPPORTED'),sources=sources,ontology=dict(fact_keys=[],state_keys=[],relationship_types=['member_of','ally','opponent','owns_asset','supports_mechanic','spatial_last_observed'],supported_operations=schema['$defs']['rule']['properties']['operation']['enum']),world=[],champions=[],items=[],runes=[],summoner_spells=[],rules=[],semantic_artifacts=[])
    corpus=[]; queue=[]; diagnostics=[]
    index=read(ROOT/'sources/ddragon/champion.json')
    for name in sorted(index['data']):
        path='sources/ddragon/champion/'+name+'.json'; src=sid(path); doc=read(ROOT/path); c=doc['data'][name]
        assert doc['version']=='15.14.1'
        champ=dict(champion_id=int(c['key']),canonical_name=name,availability=fact('listed_in_build',True,src,ptr('data',name,'id'),reason='Catalog membership only; regional enablement and mid-patch release are separate.'),base_facts=[],growth_facts=[],skills=[],mechanic_facts=[])
        for k,v in c['stats'].items():
            target=champ['growth_facts'] if k.endswith('perlevel') else champ['base_facts']
            target.append(fact('raw.'+k,v,src,ptr('data',name,'stats',k),unit='ddragon_declared_'+k,reason='Source field verified; growth formula is not inferred from the field name.'))
        for k in ['tags','partype']:
            champ['base_facts'].append(fact('raw.'+k,c[k],src,ptr('data',name,k),axis='category_set' if isinstance(c[k],list) else 'scalar'))
        spell_items=[('PASSIVE',c['passive'],ptr('data',name,'passive'))]+[(slot,sp,ptr('data',name,'spells',i)) for i,(slot,sp) in enumerate(zip('QWER',c['spells']))]
        assert len(c['spells'])==4
        for slot,sp,loc in spell_items:
            key=f'{name}.{slot}'; tokens=sorted(set(re.findall(r'\{\{\s*([^{}]+?)\s*\}\}',sp.get('tooltip','')+' '+sp.get('description',''))))
            sk=dict(skill_key=key,slot=slot,form_key=None,description_refs=[ref(src,loc+'/description')],numerical_facts=[],mechanic_facts=[],unresolved_tokens=tokens)
            if 'tooltip' in sp: sk['description_refs'].append(ref(src,loc+'/tooltip'))
            for k,unit in [('cooldown','seconds'),('cost','declared_resource'),('range','game_units'),('maxrank','rank_count')]:
                if k in sp:sk['numerical_facts'].append(fact('raw.'+k,sp[k],src,loc+'/'+k,unit=unit,axis='rank' if isinstance(sp[k],list) else 'scalar',reason='Raw declared field, not actual readiness/castability; zero, sentinel, recast and form semantics require review.'))
            if slot!='PASSIVE':sk['numerical_facts'].append(missing('resolved_damage_formula','Tooltip variables/calculation dependencies are not fully resolved by this importer.'))
            champ['skills'].append(sk)
            corpus.append(dict(skill_key=key,champion_id=champ['champion_id'],slot=slot,name=sp['name'],description=clean(sp['description']),tooltip=clean(sp.get('tooltip','')),evidence=sk['description_refs'],unresolved_tokens=tokens,form_coverage='DD_PRIMARY_ENTRY_ONLY'))
            queue.append(dict(skill_key=key,status='PENDING_SEMANTIC_REVIEW',source_refs=sk['description_refs'],required_outputs=['damage_types','target_types','cc','mobility','shield_heal','range_shape','conditions','form_dependencies','engine_exceptions'],negative_policy='Missing mention is UNKNOWN, not false.'))
            if tokens or (slot!='PASSIVE' and sp.get('effectBurn') and all(v in [None,'0'] for v in sp['effectBurn'])):
                diagnostics.append(dict(skill_key=key,unresolved_tokens=tokens,effect_all_zero=all(v in [None,'0'] for v in sp.get('effectBurn',[])),note='Do not turn effect zero or unresolved variables into damage zero.'))
        if name=='Yunara':
            champ['mechanic_facts'].append(fact('announced_release_utc','2025-07-16T19:00:00Z',sid('sources/riot/patch-25-14.html'),'text:July 16, 2025 at 19:00 UTC',unit='UTC',reason='Release announcement; regional deployment/hotfix exactness must be checked separately.'))
        bundle['champions'].append(champ)
    # The following is a source-inspected LLM semantic seed, deliberately not a complete catalog.
    seeds={
      'Ahri.Q':['magic_damage_outward','true_damage_return'], 'Ahri.E':['charm','movement_interrupt'], 'Ahri.R':['dash','conditional_recast'],
      'Malphite.R':['provides_airborne','area_damage','engage_movement'],
      'Yasuo.Q':['conditional_airborne_after_stacks'], 'Yasuo.W':['projectile_block'], 'Yasuo.E':['targeted_dash','same_target_recast_restriction'], 'Yasuo.R':['requires_airborne_enemy','blink_to_airborne_enemy'],
      'Orianna.E':['ally_shield','ball_attachment'], 'Orianna.R':['area_displacement_toward_ball','requires_ball_location'],
      'Akshan.W':['conditional_ally_revive','scoundrel_dependency','conditional_camouflage'],
      'Renata.W':['conditional_death_delay_and_rescue'], 'Renata.R':['berserk'],
      'Yunara.Q':['transform_dependent_attack_enhancement'], 'Yunara.E':['movement_speed','dash_only_when_transcendent'], 'Yunara.R':['ability_transform']}
    seed_rows=[]
    for ch in bundle['champions']:
        for sk in ch['skills']:
            if sk['skill_key'] in seeds:
                r=sk['description_refs'][0]
                f=fact('candidate.mechanic_tags',seeds[sk['skill_key']],r['source_id'],r['locator'],axis='category_set',status='UNVERIFIED',reason='LLM source-inspected seed; reviewer acceptance and ontology consistency pending. Capability does not imply current availability.')
                sk['mechanic_facts'].append(f);seed_rows.append(dict(skill_key=sk['skill_key'],fact=f))
                next(q for q in queue if q['skill_key']==sk['skill_key'])['status']='CANDIDATE_SEED_REVIEW_REQUIRED'
    # Catalogs are statically available knowledge, never end-of-match inventory features.
    for category,file in [('items','item'),('summoner_spells','summoner')]:
        src=sid('sources/ddragon/'+file+'.json'); data=read(ROOT/locations[src])['data']
        for eid,row in data.items():
            if category=='items' and not row.get('maps',{}).get('11',False):continue
            if category=='summoner_spells' and 'CLASSIC' not in row.get('modes',[]):continue
            loc=ptr('data',eid); fs=[]
            vals=({'gold.total':row.get('gold',{}).get('total'),'gold.base':row.get('gold',{}).get('base'),'gold.sell':row.get('gold',{}).get('sell'),**{'stats.'+k:v for k,v in row.get('stats',{}).items()}} if category=='items' else {k:row.get(k) for k in ['cooldown','range','summonerLevel']})
            for k,v in vals.items():
                if v is not None:fs.append(fact('raw.'+k,v,src,loc+'/'+k.replace('.','/'),unit='gold' if k.startswith('gold.') else 'source_declared',axis='rank' if isinstance(v,list) else 'scalar'))
            bundle[category].append(dict(entity_id=f'{file}.{eid}',canonical_name=row['name'],availability=fact('catalog_mode_eligible',True,src,loc+('/maps/11' if category=='items' else '/modes'),reason='Catalog map/mode flag; not confirmation that every champion can purchase/use this entry.'),facts=fs,description_refs=[ref(src,loc+'/description')]))
    src=sid('sources/ddragon/runesReforged.json')
    for i,tree in enumerate(read(ROOT/locations[src])):
        for j,slot in enumerate(tree['slots']):
            for k,rune in enumerate(slot['runes']):
                loc=ptr(i,'slots',j,'runes',k)
                bundle['runes'].append(dict(entity_id='rune.'+str(rune['id']),canonical_name=rune['name'],availability=fact('listed_in_build',True,src,loc+'/id'),facts=[fact('tree_id',tree['id'],src,ptr(i,'id')),fact('slot_index',j,src,ptr(i,'slots'),status='UNVERIFIED',reason='Index into source tree slots, not a measured participant selection.')],description_refs=[ref(src,loc+'/shortDesc'),ref(src,loc+'/longDesc')]))
    # Static objective constants from a local registry remain unverified in this new package.
    registry=read(localdest); selected={}
    ver=lambda s:tuple(map(int,s.split('.')))
    owners={'dragon':'dragon','rift_element':'dragon','elder':'elder','baron':'baron','herald':'herald','voidgrub':'voidgrub','atakhan':'atakhan','inhibitor':'inhibitor','nexus_turret':'nexus_turret','turret_plates':'outer_turret'}
    entities={n:dict(entity_id=n,entity_type='STRUCTURE' if 'turret' in n or n=='inhibitor' else 'TEAM_ASSET' if n=='soul' else 'OBJECTIVE',availability=fact('available_15_14',True,ls,ptr('rules'),status='UNVERIFIED',reason='Registry-backed candidate, per-entity runtime availability is event/rule-derived.'),facts=[],rule_ids=[],related_entity_ids=[]) for n in ['dragon','soul','elder','baron','herald','voidgrub','atakhan','inhibitor','nexus_turret','outer_turret']}
    official=sid('sources/riot/patch-25-09.html')
    for key,rows in registry['rules'].items():
        matches=[(i,v) for i,v in enumerate(rows) if ver(v['patch_range'][0])<=ver('15.14')<=ver(v['patch_range'][1])];assert len(matches)==1
        i,row=matches[0]; selected[key]=row
        owner=next(v for k,v in owners.items() if key.startswith(k));owner='soul' if key=='dragon_soul_count' else owner
        f=fact(key,row['value'],ls,ptr('rules',key,i,'value'),unit='ms' if key.endswith('_ms') else 'count',status='UNVERIFIED',reason='Inherited registry status='+row['status']+'; prior event-consistency claims were not rerun. No new exact-state validation.')
        if key in ['herald_spawn_ms','voidgrub_spawn_ms']:
            f['status']='VERIFIED';f['evidence'].append(ref(official,'text:Now spawns at 15:00 minutes' if key=='herald_spawn_ms' else 'text:Spawn timer: 6:00'));f['reason']='Official 25.09 change, carried forward as a candidate 15.14 rule; intermediate patch completeness remains unchecked.'
        entities[owner]['facts'].append(f)
        if 'combat_grace_ms' in row:entities[owner]['facts'].append(fact(key+'.combat_grace_ms',row['combat_grace_ms'],ls,ptr('rules',key,i,'combat_grace_ms'),unit='ms',status='UNVERIFIED',reason='Combat state unobserved; despawn must be an interval unless an explicit removal event exists.'))
    for n,e in entities.items():
        e['facts'].extend([missing('direct_gold_verified_formula','Recipient eligibility, local/global splitting, script overrides and objective bounties are not resolved.'),missing('current_hp','No continuous objective HP telemetry in the current match schema.')])
    entities['soul']['related_entity_ids']=['dragon','elder']; entities['dragon']['related_entity_ids']=['soul','elder']
    entities['atakhan']['facts'] += [fact('form','Thornbound',official,'text:Voracious and Ruinous Atakhan are replaced with a singular form',reason='Official 25.09 form replacement; do not import old Voracious revive.'),fact('spiritual_purification_duration','rest_of_game',official,'text:Spiritual Purification buff for the rest of the game'),fact('on_takedown_effects',['area_slow','area_damage','team_spirit_petal'],official,'text:On Takedown, bless the area',axis='category_set')]
    entities['atakhan']['facts'] += [fact('site_logic_changed_in','25.12',sid('sources/riot/patch-25-12.html'),'text:spawn logic in order to make top lane vs. bot lane spawns more equitable'),missing('site_decision_formula','25.12 changes the site-selection logic but does not provide an exact formula. Do not reuse season-start weights.')]
    entities['voidgrub']['facts'] += [fact('respawns',False,official,'text:No longer respawns'),fact('hunger_threshold',3,official,'text:when you kill 3 Grubs',unit='count')]
    # Import only named Root fields. Other forms/modes stay in the archived raw files.
    raw_objs=[('baron','sru_baron'),('dragon','sru_dragon'),('elder','sru_dragon_elder'),('voidgrub','sru_horde'),('herald','sru_riftherald'),('atakhan','sru_atakhan')]+[('dragon','sru_dragon_'+s) for s in ['fire','air','earth','water','chemtech','hextech']]
    cdragon_rows=[]
    for name,fn in raw_objs:
        src=sid('sources/cdragon/'+fn+'.bin.json'); data=read(ROOT/locations[src]);roots=[k for k in data if k.lower().endswith('/characterrecords/root')];assert len(roots)==1
        rootkey=roots[0]
        for k,v in data[rootkey].items():
            if isinstance(v,(int,float,bool)) and ('gold' in k.lower() or 'exp' in k.lower() or k in ['baseHP','baseDamage','baseArmor','baseSpellBlock']):
                f=fact('raw_cdragon.'+fn+'.'+k,v,src,ptr(rootkey,k),unit='source_declared',reason='Named Root field only; inheritance/server scripts/mode overrides not resolved. Not an executable payout formula.')
                entities[name]['facts'].append(f);cdragon_rows.append(dict(entity_id=name,record=rootkey,fact=f))
    entities['dragon']['facts'].append(fact('cdragon_subtype_coverage',['fire','air','earth','water','chemtech','hextech'],sid('sources/cdragon/sru_dragon_fire.bin.json'),ptr('Characters/sru_dragon_fire/CharacterRecords/Root'),axis='category_set',status='UNVERIFIED',reason='Collection coverage metadata; all six source files archived. No inheritance across records assumed.'))
    # Rules are executable-shape candidates, runtime activation is deliberately gated in a sidecar.
    rule_specs=[('dragon','respawn','ELITE_MONSTER_KILL','schedule_respawn','dragon_respawn_ms',['event.timestamp','event.monster_type','event.killer_team','team.soul_owner']),('baron','respawn','ELITE_MONSTER_KILL','schedule_respawn','baron_respawn_ms',['event.timestamp','event.monster_type','event.killer_team']),('elder','respawn','ELITE_MONSTER_KILL','schedule_respawn','elder_respawn_ms',['event.timestamp','event.monster_type','event.killer_team']),('inhibitor','respawn','BUILDING_KILL','schedule_respawn','inhibitor_respawn_ms',['event.timestamp','event.structure_id']),('nexus_turret','respawn','BUILDING_KILL','schedule_respawn','nexus_turret_respawn_ms',['event.timestamp','event.structure_id'])]
    for n in ['dragon','baron','herald','voidgrub','atakhan']:
        key=next(k for k in selected if k.startswith(n+'_') and (k.endswith('first_spawn_ms') or k==n+'_spawn_ms'))
        rule_specs.append((n,'first_spawn','MATCH_START','spawn_after',key,['match.start','match.patch']))
    for n,action,trig,op,key,deps in rule_specs:
        f=next(f for f in entities[n]['facts'] if f['key']==key);rid=f'{n}.{action}'
        guards=[dict(state_key='event.monster_type',operator='eq',value=n),dict(state_key='event.killer_team',operator='in',value=[100,200])] if trig=='ELITE_MONSTER_KILL' else []
        if n=='dragon' and action=='respawn':guards.append(dict(state_key='team.soul_owner',operator='eq',value='NONE'))
        if trig=='BUILDING_KILL':guards.append(dict(state_key='event.structure_class',operator='eq',value=n));deps.append('event.structure_class')
        bundle['rules'].append(dict(rule_id=rid,trigger_type=trig,guards=guards,operation=op,parameters=[f],state_dependencies=deps,outputs=[n+'.next_spawn_at'],status='UNVERIFIED',evidence=f['evidence'],exception_policy='PROPAGATE_UNKNOWN',reason='Typed rule candidate; engine adapter, same-time ordering and complete patch-chain validation pending.'))
        entities[n]['rule_ids'].append(rid)
    bundle['world']=list(entities.values())
    keys=set()
    def visit(x):
        if isinstance(x,dict):
            if 'key' in x and 'evidence' in x:keys.add(x['key'])
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(bundle)
    bundle['ontology']['fact_keys']=sorted(keys)
    bundle['ontology']['state_keys']=sorted({s for r in bundle['rules'] for s in r['state_dependencies']+r['outputs']})
    write(ROOT/'bundle.json',bundle);write(ROOT/'source_locations.json',locations)
    write(ROOT/'ability_corpus.json',corpus);write(ROOT/'semantic_extraction_queue.json',queue);write(ROOT/'semantic_seed.json',dict(producer='Codex source-inspected drafting in this conversation; exact model revision unavailable',accepted_for_primary_model=False,rows=seed_rows))
    write(ROOT/'unresolved_fields.json',diagnostics);write(ROOT/'objective_raw_fields.json',cdragon_rows)
    write(ROOT/'selected_local_rules_15_14.json',selected)
    write(ROOT/'runtime_policy.json',dict(default_rule_activation=False,activation_requires=['resolve source and patch-chain validation','pass event-prefix and same-time tests','validate state dependencies','explicit integration into approved engine'],raw_field_semantics='VERIFIED means exact source extraction, not proof of complete server mechanic',unknown_policy='UNKNOWN is distinct from false, zero, MISSING and STRUCTURAL_NA',llm_semantics_primary_enabled=False,model_estimate_primary_enabled=False,direct_gold_formula_enabled=False,geometry_enabled=False,known_coverage_gaps=['complete hotfix/region and all-mechanic overlay audit; 25.10-25.13 objective-keyword review completed','all champion alternate forms and hidden mechanics','resolved server gold payout and bounty semantics','region/time-specific enablement including Yunara release','full respawn/revive exceptions','complete buff-holder telemetry'],future_patch_policy='Require a new bundle and regression checks; never blindly reuse patch constants.'))
    basic=read(BASE/'BASIC_FEATURE_CONTRACT_V1.json')
    contract=dict(version='heads-input-15.14-v1',status='DESIGN_NOT_DATASET_PRODUCER',time=dict(anchor='tau=first_kill_ms-10000',frame='F_tau=max(real frame timestamp <= tau)',eligibility='0<=tau-F_tau<=10000 ms',history='latest 3 real frames at/before tau; left pad with explicit mask',events='prefix timestamp<=tau; equal-ms ordering unresolved -> UNKNOWN',label_future='never supplied to input'),join_keys=['match_patch','champion_id','participant_id','team_id'],heads={
      'individual':dict(observed_shape=['B',3,2,5,18],observed_order=basic['player_continuous']['ordered_fields'],static=['champion base/growth raw fields','passive and QWER description tokens','rank arrays with padding/rank mask','resource type and Riot tags','candidate mechanics only in separate reviewed arm'],dynamic_prefix=['KDA','LEVEL_UP-derived level with provenance','skill-rank upgrade history if supported','inventory from purchase/sell/undo events if complete','death timestamp and conditional respawn interval','buff eligibility with support mask'],metadata=['observation timestamp','age_ms','provenance_type','support_mask','uncertainty_flag'],forbidden=['final match KDA/inventory/role','future first kill position or participants','unobserved ability cooldown readiness','interpolated XY presented as observed']),
      'team':dict(member_shape=['B',2,5,'d_player'],readout=['masked set aggregation','team gold/XP/HP summaries with common frame age','alive count and unknown count separately','objective stacks and asset ownership'],relations=['20 directed distinct ally pairs per team','25 cross-team pairs per focal team; reverse orientation separately','potential mechanic compatibility, not fixed counter score'],example='Malphite.R provides airborne -> Yasuo.R requires airborne; capability relation only, execution unknown',symmetry='Within-team order permutation invariant; swap-team gain sign/probability complements tested'),
      'rift':dict(nodes=['objective entity/site','team assets','structure slots with stable ID'],inputs=['patch availability mask','time since start','kill/removal event history','rule-derived spawn/despawn interval','team ownership and elemental stacks','structure destroyed/respawn candidate','remaining asset duration when supported'],spatial='last-observed player XY plus age/mask; fixed geometry excluded until provenance verified',unknown=['objective HP','exact Atakhan site before observed without complete determining PvP data','current ward vision','minion wave positions']),
      'fusion':dict(inputs=['player readout','two team readouts and differences','rift readout','query time/support masks'],model_ablation=['tree on same source fields with explicit aggregates','set/attention','temporal model then set','heterogeneous graph with typed edges'],knowledge_ablation=['K0 dynamic baseline','K1 raw structured numerical knowledge','K2 source-text encoder','K3 reviewed semantic mechanics','K4 relational edges'],comparability='Hold sample, temporal cutoff, feature availability and outcome definitions fixed; architecture and knowledge source ablations are separate')},embedding=dict(generated=False,proposal='Frozen shared description encoder plus numeric MLP; optional source-grounded trait encoder; no LLM call at match inference',fit_scope='Any corpus adaptation/scalers/relationship learning on training data only',caution='Text embedding is semantic similarity, not effect size or causal value'),split=dict(train='15.14',validation='15.15',test='15.16',limitation='A single training patch cannot identify response to patch-constant rule changes.'),outcomes=dict(label_change=False,knowledge_is_not_gold='Do not convert mechanics tags or embedding distance to gold',value_branch='Predict disjoint future net-gold increments using shared representation; causal claims require identification beyond predictive validation',no_double_count='Realized interval [tau,b] and future interval (b,H] must be disjoint; residual prediction excludes already counted immediate rewards'))
    write(ROOT/'head_input_contract.json',contract)
    print(json.dumps(dict(champions=len(bundle['champions']),skill_entries=len(corpus),semantic_seed_entries=len(seed_rows),items=len(bundle['items']),runes=len(bundle['runes']),summoner_spells=len(bundle['summoner_spells']),world_entries=len(bundle['world']),rule_candidates=len(bundle['rules']),sources=len(sources)),indent=2))

if __name__=='__main__':main()
