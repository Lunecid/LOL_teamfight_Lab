import os
"""Versioned multi-patch protocol; preserve all old split/artifact records."""
import copy, hashlib, json, pathlib, re
ROOT=pathlib.Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build(contract,base_registry,base_recipes):
    audit=read('MULTIPATCH_DATA_AUDIT.json')
    cached_manifest=pathlib.Path(os.environ["LOL_EXTERNAL_ROOT"]) / 'audit' / 'kr_15_18_22' / 'AUDIT_MANIFEST.json'
    prepared_manifest=pathlib.Path(os.environ["LOL_EXTERNAL_ROOT"]) / 'preparation' / 'kr_15_18_22' / 'RUN_MANIFEST.json'
    support={'scope':'Read existing reports, not reaccept old caches as v5 data','sources':[{'path':str(p),'sha256':sha(p)} for p in [cached_manifest,prepared_manifest]],'cache_audit':read(str(cached_manifest)),'preparation_status':{k:v for k,v in read(str(prepared_manifest)).items() if k in ['accepted_preparation','complete','training_ready','completed_utc','entries_binding']},'conclusion':'Legacy-converted caches are reusable source inventories, not certified v5 training inputs; verify source clock and static-ID provenance from raw pairs before G1/G3.'}
    write('MULTIPATCH_CACHE_SUPPORT_REVIEW.json',support)
    split={
      'id':'MP15_RAW_18_20_V21_T22_r1','status':'DESIGN_FIXED_PRODUCER_PENDING','region':'KR','queue_id':420,'map_id':11,'game_mode':'CLASSIC',
      'primary_source':'Lol_project/matches/kr raw detail+timeline pairs, normalized by one new versioned producer for ALL arms',
      'train_patches':['15.18','15.19','15.20'],'single_train_patch':'15.20','validation_patch':'15.21','benchmark_test_patch':'15.22',
      'train_fit_buckets':[0,79],'stop':{'patches':['15.20'],'buckets':[80,99],'shared_across_arms':True},
      'unused_reserved':{'patches':['15.18','15.19'],'buckets':[80,99],'training_allowed':False},
      'validation_buckets':{'DEV':[0,49],'CAL':[50,74],'AUDIT':[75,99]},
      'hash_salt':'rift-v5-multipatch-r1:','bucket_rule':'int(first_16_hex_of_SHA256(salt+canonical_match_key),16) modulo 100',
      'match_key':'platform uppercase plus match ID; deduplicate across folders before split; never participant IDs',
      'benchmark_status':'PREVIOUSLY_EXPOSED_BENCHMARK_NOT_PRISTINE',
      'exposure_source':'MULTIPATCH_DATA_AUDIT.json#/prior_exposure',
      'same_model_fit_test_overlap_allowed':False,'previous_model_exposure_erased_by_new_split':False,
      'external_diagnostic':['KR_16.13','NA1_16.13','KR_16.14_pilot','KR_16.15'],
      'fresh_confirmatory_test':{'status':'NOT_IDENTIFIED_IN_AUDITED_HOLDINGS','requirement':'new later unseen corpus plus exposure ledger; lock before outcome access; do not collect in this task'},
      'legacy_profile':'LEGACY_15_14_V15_T16 preserved verbatim in PIPELINE_CONTRACT_V5.json#/split; never overwrite old manifests',
      'missing_patch_policy':'no 15.17/pre-15.14 candidates in audited holdings; no inferred fills',
      'new_split_implemented':False}
    features={
      'version':'COMMON_PLUS_PATCH_CONTEXT_V1','status':'SEMANTIC_ALLOWLIST_PENDING_FIELD_BINDING',
      'COMMON':{
        'player':['ordered base18 with original support and observation age','champion and starting summoner spell IDs','observed skill rank and learned support under verified prefix reconstruction, no semantic-effect gating'],
        'team':['same declared observed team economy/statistics','exact prefix kill/death/assist and objective acquisition records','role proxy and early observed frames/support, supplied equally in all A-D arms'],
        'rift':['actual game clock','observed structure destruction and supported counts','last observed objective events and elapsed time since event; do not infer ready/spawned from patch timers','raw observed XY and shared Euclidean pair descriptors'],
        'engine':'E1 exact event facts plus held frame values; no scheduled RULE_DERIVED timers in the model-visible overlay',
        'measurement_adapter':'correct patch/build parsers, legal rank reconstruction, team/object identity and missingness apply to every arm; common does not mean knowingly mismeasuring the game',
        'not_an_invariance_guarantee':'ID, observed stats and support patterns can identify patch indirectly; do not claim invariant or causal representation'},
      'PATCH_CONTEXT':{
        'player':['reviewed skill semantics and conditions in existing K3 schema','explicit learned-skill gating, not cast readiness'],
        'team':['joint object/buff rules and supported timers','composition relations only derived from declared abilities, no LLM win/matchup prices'],
        'rift':['patch-applicable object availability/spawn/expiry schedules','validated static/dynamic terrain descriptors and supported paths','known transition/variant applicability at each own historical cutoff'],
        'engine':'E2: add only validated rule schedules; unknown current effects stay UNKNOWN',
        'runtime_patch_id_embedding':False,'unknown_mechanic':'STRUCTURAL_NA for inapplicable, UNKNOWN for applicable but unsupported; no forced zero'},
      'feature_mapping':{'COMMON_config':{'knowledge':'id_only','rank':True,'gating':False,'terrain':'off','non_skill_non_terrain_engine':'events_only'},'CONTEXT_config':'same common source information plus declared patch context; Q09 full configuration; role and history held equal'},
      'strict_field_registry':'Every value, mask, provenance status used as input must have COMMON or PATCH_CONTEXT classification and dependency closure; unresolved binding blocks G4. Numerical ability fields not yet in the model manifest require explicit manifest version before execution, not silent addition.',
      'optional_numeric_ability_extension':{'enabled':False,'reason':'per-rank cooldown/range/cost/damage scalars are not silently added to K3; require audited sources, fixed slots/units/support, new feature manifest and rerun all affected controls'},
      'static_knowledge_union':'schema fixed from training-patch sources; reserve UNK for future mechanics; heldout package values may be read at inference but not used to fit vocab/encoder/scaler',
      'historical_knowledge':'versioned patch/build sources available at prediction time; no modern retrospective LLM knowledge leakage',
      'training_variation_audit':'for each context field/rule and champion-slot, report FIT patch/build distinct values and support; constant or unseen mechanics do not establish a learned response to changes; changing collection/meta composition is not proof of invariance',
      'forbidden':['patch ID lookup embedding','collector/source folder ID feature','final role or item build','future fight participants/locations','actual label window length/support/reason','future objective acquisition as input','LLM-generated matchup scores or object prices'],
      'same_target':'observed terminal/gold v5 label unchanged; input K/engine/terrain arms cannot alter outcome source, mapper or eligibility',
      'bundle_interpretation':'C versus D estimates joint incremental information/representation of declared patch package, not terrain-only or semantic-only causal effect; Q13-Q20 isolate components'}
    design={
      'version':'v5-multipatch-r2','active_split':split,'feature_partition':features,
      'matched_design':{
        'unit':'eligible FIT matches with at least one fresh supported Q target on the COMMON/CONTEXT paired row policy',
        'N':'N=3*min(floor(n_15.20/3),n_15.18,n_15.19,n_15.20), using eligible FIT counts only',
        'single':'select N eligible 15.20 FIT matches by SHA256(mp-match-size-r1:+canonical_match_key)',
        'multi':'select N/3 per 15.18/15.19/15.20 by the same fixed ranking; no labels/signs/losses used for ranking',
        'paired_inputs':'A/B share exact matches and rows; C_M/D_M share exact matches and rows; all cells evaluated on same locked future rows',
        'same_N_not_same_queries':'report match/query/fight/family counts; match weighting prevents long games dominating; no claim randomization identifies a pure causal patch-diversity effect',
        'scale_fit':'each arm FIT only, identical definition; P families fixed from training support; no heldout rescaling',
        'maximum_updates':'matched A/B/C_M/D_M neural runs use common max_steps=100*ceil(A eligible FIT query rows with swap views/256); evaluate every ceil(A rows with swap/256) steps, patience 10 such intervals; restart shuffled epochs as needed. Tree max 3000 rounds unchanged',
        'small_support':'N=0 or unavailable primary label support -> block contrast, do not substitute another patch'},
      'full_design':'C_F and D_F use all eligible FIT of three train patches. Q/P losses average patch-specific match risks equally. They are separate from matched-size four cells.',
      'weighting':'Q: mean_p mean_m mean_supported_rows loss, independently per task. P: within each patch retain 0.5 overall +0.5 family-match risk, then average patches. Unknown target never zero; active family sets and weights frozen from FIT.',
      'earlier_cache_extension':{'id':'S_OLD_CACHE','enabled':False,'patches':['15.14','15.15','15.16'],'reason':'source/provenance shift may confound patch diversity; only after same producer semantics or explicit source-sensitivity comparison; do not retrain frozen lineage'},
      'cross_season_training_extension':{'enabled':False,'reason':'16.x reserved external diagnostics; very few supported patch environments and historical exposure; season-crossing train split needs a separate new holdout'},
      'rolling_origin_extension':{'enabled':False,'candidate':'train15.18, val15.19, benchmark15.20; main train15.18-20,val15.21,test15.22','interpretation':'diagnostic temporal stability, not repeated pristine confirmation'},
      'training_executed':False,'new_labels_generated':False,'audit':'MULTIPATCH_DATA_AUDIT.json','cache_support_review':'MULTIPATCH_CACHE_SUPPORT_REVIEW.json'}
    cards=[]
    for old in base_registry['cards']:
        card=copy.deepcopy(old);card['id']='MP.'+old['id'];card['template_id']=old['id'];card['split_profile_id']=split['id'];card['training_regime']='MULTI_FULL';card['feature_partition']='CONTEXT_WITH_TEMPLATE_ABLATION';card['hyperparameter_source']='MP.'+old['hyperparameter_source'];card['recipe_profile']='MP_FULL';card['inputs']['source_policy']='RAW_SAME_PRODUCER_ALL_ARMS';card['status']='SPECIFIED_PENDING_GATES';cards.append(card)
    aliases={}
    for backbone,parent in [('QH','Q09'),('QT','Q01')]:
        reference=copy.deepcopy(next(x for x in cards if x['template_id']==parent))
        aliases['D_F.'+backbone]=reference['id']
        for cell,regime,partition in [('A','SINGLE_MATCHED','COMMON'),('B','SINGLE_MATCHED','CONTEXT'),('C_M','MULTI_MATCHED','COMMON'),('D_M','MULTI_MATCHED','CONTEXT'),('C_F','MULTI_FULL','COMMON')]:
            card=copy.deepcopy(reference);card['id']='MP.'+cell+'.'+backbone;card['name']=cell+'_'+backbone;card['group']='patch_diversity_by_context';card['training_regime']=regime;card['feature_partition']=partition;card['hyperparameter_source']=card['id'];card['stage']='S3';card['recipe_profile']='MP_MATCHED' if regime.endswith('MATCHED') else 'MP_FULL';card['comparison_cell']=cell
            if partition=='COMMON':
                card['config'].update(features['feature_mapping']['COMMON_config']);card['required_gates']=[g for g in card['required_gates'] if g!='G2T'];card['required_parameter_locks']=[p for p in card['required_parameter_locks'] if p!='P04'];card['inputs']['forbidden_extension_dependencies']='semantic tags/effect gating/rule timers/terrain descriptors and their support shortcuts'
            cards.append(card);aliases[cell+'.'+backbone]=card['id']
    extra=[{'id':'C_MP_DATA','comparison':'MP.C_M.QH vs MP.A.QH','claim':'matched eligible match count, multi-patch versus latest single-patch under common inputs'}, {'id':'C_MP_CONTEXT','comparison':'MP.D_M.QH vs MP.C_M.QH','claim':'same multi-patch matched rows, added patch-context bundle'}, {'id':'C_MP_COMBINED','comparison':'MP.D_M.QH vs MP.B.QH','claim':'matched context-input comparison across single and multiple training patches'}]
    primary=[dict(x,comparison=' vs '.join('MP.'+i for i in x['comparison'].split(' vs '))) for x in base_registry['primary_claim_contrasts']]+extra
    searches=['MP.'+x for x in base_recipes['development']['search_groups']]+[x['id'] for x in cards if x['group']=='patch_diversity_by_context']
    registry={'version':'v5-multipatch-r2','status':'ACTIVE_DESIGN_NOT_EXECUTED','legacy_registry':'EXPERIMENT_REGISTRY_V5.json','split_profile_id':split['id'],'cards':cards,'comparison_aliases':aliases,'primary_claim_contrasts':primary,'multiplicity':{'n':8,'adjustment':'Holm','tree_replicates':'secondary; not additional uncorrected primary claims','prior_five_claim_tests_superseded_for_MP':True},'patch_context_interaction':'R(MP.B.QH)+R(MP.C_M.QH)-R(MP.D_M.QH)-R(MP.A.QH); diagnostic','P_scope':'P_TREE/P_HIER now use multiple train patches; four-cell comparative generalization claims restricted to Q unless P four cells separately preregistered','execution_stages':copy.deepcopy(base_registry['execution_stages']),'search_groups':searches,'counts':{'inherited_ablation_cards':30,'additional_patch_comparison_cards':10,'active_total_cards':40,'seeds_per_card':5,'final_seed_runs':200,'search_groups':23,'maximum_search_seed_runs':552,'nominal_final_plus_search':752,'legacy_r1_runs_automatically_added':False,'tree_multioutput_count_is_not_estimator_fit_count':True},'required_upstream':['MP_G1_RAW_PROVENANCE','MP_G4_FEATURE_PARTITION','MP_G5_SPLIT_EXPOSURE_AND_MATCHING'],'optional_extensions':copy.deepcopy(base_registry['conditional_extensions'])+[design['earlier_cache_extension']],'training_executed':False}
    for stage in registry['execution_stages']:
        if stage['id']=='S0':stage['action']='refresh source inventory/provenance, exposure ledger, same raw producer; TRAIN 15.18-20-only source/label locks'
        if stage['id']=='S2':stage['action']='lock MP split and field partition; paired row and matched-match manifests; 23 bounded search groups'
        if stage['id']=='S3':stage['action']='13 inherited structural/P search groups plus 10 patch-comparison groups; STOP 15.20, DEV 15.21; matched budget profile where declared'
        if stage['id']=='S6':stage['action']='15.21 CAL/AUDIT; freeze all eight main contrasts and benchmark exposure statement before new benchmark evaluation'
        if stage['id']=='S7':stage['action']='15.22 previously exposed benchmark, then supported 16.x external diagnostics; no claim of pristine confirmation'
    recipes=copy.deepcopy(base_recipes);recipes['version']='v5-multipatch-r2';recipes['development'].update(fit='15.18/15.19/15.20 FIT per card regime',early_stop='15.20 STOP shared across arms',select='15.21 DEV',calibrate='15.21 CAL',audit='15.21 AUDIT',test='15.22 BENCHMARK_TEST_PRIOR_EXPOSURE',external='16.x prior-exposure diagnostics',search_groups=searches,search_group_count=23,maximum_search_seed_runs=552,input_ablation_recipe='MP.Q09 selected recipe for matched input ablations; MP.Q24 from MP.Q01; patch-comparison cards each get identical independent 12-card search budget')
    recipes['weighting']['multi_patch']=design['weighting'];recipes['profiles']={'MP_FULL':'inherited max100 epochs; equal patch-match risks; all eligible FIT in regime','MP_MATCHED':design['matched_design']['maximum_updates']};recipes['matched_design']=design['matched_design'];recipes['execution']['primary_source']=split['primary_source']
    for card in cards:
        card['fit_patches']=['15.20'] if card['training_regime']=='SINGLE_MATCHED' else list(split['train_patches'])
        card['required_mp_gates']=list(registry['required_upstream'])
    write('MULTIPATCH_DESIGN_V5.json',design);write('MULTIPATCH_EXPERIMENT_REGISTRY_V5.json',registry);write('MULTIPATCH_TRAINING_RECIPES_V5.json',recipes)
    # Full-resolution card lookup and experiment stages are authoritative for MP runs.
    contract['revision']='v5-multipatch-r2';contract['active_split_profile_id']=split['id'];contract['split_profiles']={'LEGACY_15_14_V15_T16':copy.deepcopy(contract['split']),split['id']:split};contract['split_resolution']='Every MP card MUST resolve split_profile_id; top-level split is preserved legacy only, never the active default'
    contract['experiment_protocol'].update(registry='MULTIPATCH_EXPERIMENT_REGISTRY_V5.json',recipes='MULTIPATCH_TRAINING_RECIPES_V5.json',multipatch_design='MULTIPATCH_DESIGN_V5.json',required_cards=[x['id'] for x in cards],counts=registry['counts'],legacy_registry='EXPERIMENT_REGISTRY_V5.json')
    contract['evaluation']['active_multipatch_multiplicity']=registry['multiplicity'];contract['evaluation']['active_benchmark_status']=split['benchmark_status'];contract['evaluation']['primary_claim_multiplicity']['scope']='legacy r1 only; active MP has eight contrasts'
    contract['model_selection']['active_multipatch_registry']='MULTIPATCH_EXPERIMENT_REGISTRY_V5.json';contract['model_selection']['legacy_fields_scope']='unprefixed Q/P template catalogue; resolve active MP cards and recipes before fit'
    contract['architecture_controls']['template_models']=contract['architecture_controls']['base_models']
    contract['architecture_controls']['base_models']=[{'id':x['id'],'architecture':x['config']['architecture']} for x in cards]
    contract['model_selection']['active_primary_structural_candidates']=['MP.Q'+str(i).zfill(2) for i in range(1,10)]
    contract['open_contracts'][6].update(field='active_train_validation_test_and_external_patch_knowledge_support',candidate='15.18-15.22 and evaluated16.x packages; 15.14 package cannot stand in for them')
    contract['multipatch_gates']=[{'id':'MP_G1_RAW_PROVENANCE','status':'PENDING','requires':['full eligible raw-pair census','explicit original timestamp/missingness','same producer all splits','prior exclusion and duplicate resolution']},{'id':'MP_G4_FEATURE_PARTITION','status':'PENDING','requires':['all field/value/support dependencies assigned COMMON or CONTEXT','training-only schema/vocabulary/scales','all used patch packages and geometry reviewed']},{'id':'MP_G5_SPLIT_EXPOSURE_AND_MATCHING','status':'PENDING','requires':['canonical match dedup and split disjointness','exposure ledger acknowledged','matched N and parent row manifests locked','no outcome-dependent rank selection']}]
    for name in ['MULTIPATCH_DATA_AUDIT.json','MULTIPATCH_CACHE_SUPPORT_REVIEW.json','MULTIPATCH_DESIGN_V5.json','MULTIPATCH_EXPERIMENT_REGISTRY_V5.json','MULTIPATCH_TRAINING_RECIPES_V5.json','MULTIPATCH_PROTOCOL_KO.md']:
        contract['source_manifest'].append({'path':name,'sha256':sha(ROOT/name)})
    return design,registry,recipes

def sync_narrative():
    doc=ROOT/'FULL_DESIGN_SPECIFICATION_V5_KO.md';txt=doc.read_text(encoding='utf-8');protocol=(ROOT/'MULTIPATCH_PROTOCOL_KO.md').read_text(encoding='utf-8')
    marker='## 20. 다중 패치 학습·피처·비교 설계 — multipatch-r2'
    if marker in txt:txt=txt[:txt.index(marker)].rstrip()+'\n\n'
    txt+=marker+'\n\n'+protocol[protocol.index('\n\n')+2:]+'\n'
    doc.write_text(txt,encoding='utf-8')
