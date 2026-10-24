"""Static consistency and synthetic specification probes. Not a production-engine test."""
import copy,hashlib,json,pathlib,math
ROOT=pathlib.Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=load('PIPELINE_CONTRACT_V5.json'); base=load('../knowledge_15_14_v2/PIPELINE_CONTRACT_V4.json')
records=load('INTERFACE_RECORDS.json')['records']
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def invariant_errors(x):
    errors=[]
    for k in ['split','time_contract']:
        if x[k]!=base[k]:errors.append('immutable_'+k)
    if x['feature_contract']['ordered_18']!=base['feature_contract']['ordered_18']:errors.append('base18')
    if x['fight_labels']['primary']!=base['fight_labels']['primary']:errors.append('observed_label')
    if x['cohort_profiles']['Q_fresh_v5']['age_ms_max_inclusive']!=10000:errors.append('fresh_boundary')
    if x['cohort_profiles']['Q_allphase_v5']['age_ms_max_inclusive'] is not None:errors.append('allphase_cap')
    if x['asset_research']['enabled'] or x['model_roles']['U_residual']['status']!='disabled_pending_identification':errors.append('asset_activation')
    if x['terrain']['runtime_enabled'] and x['terrain']['terrain_package']['walkability_status']!='VALIDATED':errors.append('terrain_activation')
    if x['skills']['learned_implies_cast_ready']:errors.append('rank_readiness')
    if not x['canonical_order']['role_proxy']['proxy_and_support_given_to_all_models']:errors.append('role_parity')
    if not x['architecture_controls']['no_spatial_ablation_removes_role_proxy']:errors.append('spatial_role_leak')
    return errors

check('v5 invariants',not invariant_errors(c))
for s in c['source_manifest']:check('source hash '+s['path'],sha((ROOT/s['path']).resolve())==s['sha256'])
check('P outcomes copy consistent',c['model_roles']['P']['outputs']==c['outcomes'])
check('U copies consistent',c['model_roles']['U_residual']==c['fight_labels']['asset_extension'])
check('Q classes exclude unresolved',c['model_roles']['Q']['classes']==['BLUE_GAIN','RED_GAIN','TIE'])
check('all seven families',len(set(c['objective_families']))==7)
check('skill factorial isolates added ranks',c['skills']['factorial_inputs']==[['K0','NO_RANK'],['K3','NO_RANK'],['K0','RANK'],['K3','RANK']])
check('role fixed across history but may vary across query',c['canonical_order']['different_query_permutations_allowed'] and 'every historical frame' in c['canonical_order']['history_scope'])
check('asset B not identified',c['asset_research']['B']['reference_status']=='UNDEFINED_NOT_IDENTIFIED')
check('unknown geometry explicitly disabled',not c['terrain']['runtime_enabled'] and c['terrain']['terrain_package']['walkability_status']=='NOT_OBTAINED')
check('candidate gates registered',len(c['open_contracts'])==8 and all(x['lock_stage'] for x in c['open_contracts']))
check('training not executed',not c['training_executed'] and not c['new_match_labels_generated'])
check('interface names unique',len(records)==11)
for name,r in records.items():
    check('interface fields '+name,bool(r['fields']) and all(set(x)=={'type','meaning','nullable'} for x in r['fields'].values()))
check('class gold support separated',{'classification_support','gold_support'}<=set(records['LabelRecord']['fields']))
for forbidden in ['s_ms','c_ms','E_actual_ms','observed_gold_score','label','future_members']:
    check('query excludes '+forbidden,forbidden not in records['QueryRecord']['fields'])

# Negative mutations prove the contract guards reject consequential changes.
mutations=[
 ('future cutoff change',lambda x:x['time_contract'].__setitem__('query_tau','first_kill_ms')),
 ('test split change',lambda x:x['split'].__setitem__('test_patch','15.15')),
 ('asset activation',lambda x:x['asset_research'].__setitem__('enabled',True)),
 ('geometry activation without grid',lambda x:x['terrain'].__setitem__('runtime_enabled',True)),
 ('learned to ready',lambda x:x['skills'].__setitem__('learned_implies_cast_ready',True)),
 ('role extra information only tree',lambda x:x['canonical_order']['role_proxy'].__setitem__('proxy_and_support_given_to_all_models',False)),
 ('age 60s silent exclusion',lambda x:x['cohort_profiles']['Q_allphase_v5'].__setitem__('age_ms_max_inclusive',60000))]
for name,mutate in mutations:
    bad=copy.deepcopy(c);mutate(bad);check('reject '+name,bool(invariant_errors(bad)))

# Tiny reference oracles below verify declared boundary cases, not real match behavior.
def membership(tau,frames):
    eligible=[t for t in frames if t<=tau]
    if not eligible:return False,False,None
    age=tau-max(eligible);return 0<=age<=10000,age>=0,age
check('age zero included',membership(60000,[0,60000])==(True,True,0))
check('age 10000 included',membership(70000,[0,60000])==(True,True,10000))
check('age 10001 only allphase',membership(70001,[0,60000])==(False,True,10001))
check('age above 60s retained with valid clock',membership(125001,[0,60000])==(False,True,65001))
check('future frame not read',membership(70000,[0,60000,120000])==membership(70000,[0,60000]))
check('no prior frame unsupported',membership(-1,[0,60000])==(False,False,None))
def decide(gold,terminal):
    if terminal=='BLUE_END':return 'BLUE_GAIN'
    if terminal=='RED_END':return 'RED_GAIN'
    if terminal!='NO_END' or gold is None:return 'UNRESOLVED'
    return 'BLUE_GAIN' if gold>0 else 'RED_GAIN' if gold<0 else 'TIE'
check('terminal overrides negative gold',decide(-250,'BLUE_END')=='BLUE_GAIN')
check('terminal supports class without gold',decide(None,'RED_END')=='RED_GAIN')
check('unknown terminal not silently ignored',decide(250,'UNKNOWN')=='UNRESOLVED')
check('missing gold not tie',decide(None,'NO_END')=='UNRESOLVED')
check('observed exact zero tie',decide(0,'NO_END')=='TIE')
check('nowcast replacement identity',(130-100)+(170-130)==170-100)
check('asset A need not equal observed at E',(170-100)-25!=170-100)
check('asset B equality needs only terminal zeros',(170-100)+0-0==170-100)

registry=load('EXPERIMENT_REGISTRY_V5.json')
recipes=load('TRAINING_RECIPES_V5.json')
metrics=load('EVALUATION_PROTOCOL_V5.json')
operators=load('MODEL_OPERATORS_V5.json')
cards={x['id']:x for x in registry['cards']}
expected_ids={f'Q{i:02}' for i in range(1,29)}|{'P_TREE','P_HIER'}
check('30 unique resolved cards',len(registry['cards'])==len(cards)==30 and set(cards)==expected_ids)
check('parameter model namespaces distinct',not set(cards)&{x['id'] for x in c['open_contracts']})
check('final execution count',registry['counts']['final_seed_runs']==sum(len(x['seeds']) for x in cards.values())==150)
check('bounded search count',len(recipes['development']['search_groups'])==13 and 13*12*2==registry['counts']['maximum_search_seed_runs'])
check('shared seeds preserved',all(x['seeds']==recipes['development']['final_seeds'] for x in cards.values()))
check('legacy registry retained',c['experiment_protocol']['legacy_registry']=='EXPERIMENT_REGISTRY_V5.json')
check('current factorial not legacy alias',c['model_selection']['factorial_2x2']==[['Q06','Q07'],['Q08','Q09']])
check('raw primary explicit',c['evaluation']['Q_primary']==metrics['Q_RAW']['primary'] and 'raw' in metrics['Q_RAW']['primary'])
check('five claim correction separate from family',c['evaluation']['primary_claim_multiplicity']['n']==5 and c['evaluation']['multiplicity']['n']==7)
check('all conditional experiments disabled',all(not x['enabled'] for x in registry['conditional_extensions']))
check('invariance mandatory not conditional','S_INVARIANCE' not in {x['id'] for x in registry['conditional_extensions']} and len(registry['mandatory_nontraining_checks'])>=8)
check('fixed weights package swap', 'same frozen trained weights' in next(x for x in registry['conditional_extensions'] if x['id']=='S_PATCH')['design'])
check('same source matrix unmodified',sha((ROOT/registry['source_path']).resolve())==registry['source_sha256'])
known_gates={x['id'] for x in c['execution_gates']}
known_locks={x['id'] for x in c['open_contracts']}
stage_map={x['id']:x for x in registry['execution_stages']}

def experiment_errors(card):
    errors=[]; cfg=card['config']
    if cfg['gating'] and (not cfg['rank'] or cfg['knowledge']!='semantic_tags'):errors.append('gating_inputs')
    if not cfg['player_spatial'] and (cfg['role_proxy'] or cfg['ordering']!='canonical' or not cfg['remove_spatial_dependency_closure']):errors.append('spatial_leak')
    if card['loss_id']=='Q_CLASS' and (cfg['gold_prediction_output'] or cfg['gold_loss_weight']!=0):errors.append('class_gold')
    if cfg['architecture']=='lightgbm' and card['loss_id'] not in ['Q_TREE','P_TREE']:errors.append('tree_joint')
    if cfg['terrain']!='off' and ('G2T' not in card['required_gates'] or 'P04' not in card['required_parameter_locks']):errors.append('terrain_gate')
    if cfg['rank'] and 'P05' not in card['required_parameter_locks']:errors.append('rank_lock')
    if cfg['role_proxy'] and 'P03' not in card['required_parameter_locks']:errors.append('role_lock')
    if card['inputs']['future_label_metadata_visible']:errors.append('future_input')
    if card['inputs']['history_shape'][0]!=cfg['history_frames']:errors.append('history_shape')
    if card['hyperparameter_source'] not in recipes['development']['search_groups']:errors.append('unregistered_search')
    return errors

for cid,card in cards.items():
    check('resolved fields '+cid,set(card['config'])==set(cards['Q09']['config']))
    check('loss metric stage references '+cid,card['loss_id'] in recipes['losses'] and all(x in metrics for x in card['metric_ids']) and card['stage'] in stage_map)
    check('gate references '+cid,set(card['required_gates'])<=known_gates and set(card['required_parameter_locks'])<=known_locks)
    check('row policy reference '+cid,card['inputs']['row_policy'] in registry['row_policies'])
    check('card invariants '+cid,not experiment_errors(card))
    check('pending not executed '+cid,card['status']=='SPECIFIED_PENDING_GATES' and not card['training_executed'])

def differences(a,b):
    return {k for k in cards[a]['config'] if cards[a]['config'][k]!=cards[b]['config'][k]}
for cid,fields in {
 'Q10':{'readouts'},'Q11':{'readouts'},'Q12':{'readouts'},
 'Q13':{'non_skill_non_terrain_engine'},'Q14':{'non_skill_non_terrain_engine'},
 'Q18':{'gating'},'Q19':{'terrain'},'Q20':{'terrain'},'Q22':{'history_frames'},
 'Q23':{'profile'},'Q25':{'gold_loss_weight','gold_prediction_output'}
}.items():check('single factor '+cid,differences('Q09',cid)==fields)
check('semantic primary identical ranks',differences('Q18','Q17')=={'knowledge'})
check('role sort only tree',differences('Q01','Q26')=={'ordering'})
check('role sort only mlp',differences('Q02','Q27')=={'ordering'})
check('allphase spatial contrast isolated',differences('Q23','Q28')=={'player_spatial','role_proxy','remove_spatial_dependency_closure'})
check('temporal factorial isolated',differences('Q06','Q07')=={'temporal'} and differences('Q08','Q09')=={'temporal'})
check('relation factorial isolated',differences('Q06','Q08')=={'relation'} and differences('Q07','Q09')=={'relation'})
check('classification-only has no gold objective',recipes['losses']['Q_CLASS']['gold'] is None)
check('P tree not joint neural encoder',not recipes['losses']['P_TREE']['shared_encoder'])
check('train only scales',recipes['weighting']['scale_fit_scope'].startswith('FIT only'))
check('calibration separate',recipes['calibration']['primary'].startswith('raw'))
check('full CPU cap',recipes['execution']['max_total_CPU_threads']==4 and not recipes['execution']['nested_parallelism'])
check('no hidden pretrained ablation',not recipes['execution']['full_model_embedding_reused_in_ablation'] and all(x['config']['pretraining']=='none' for x in cards.values()))
check('model operators fixed time before relation','temporal encoder THEN' in operators['relations']['order'])
check('full pair information shared',not operators['input_view']['new_geography_only_for_graph'] and 'EVERY' in operators['input_view']['raw_pair_context'])
check('bootstrap confidence explicit',metrics['AGGREGATION']['interval'].startswith('95%'))
check('nonlinear metric seeds after transform','THEN average seed metrics' in metrics['AGGREGATION']['nonlinear_metrics'])
check('development lock not circular',len(recipes['development']['lock_phases'])==2 and 'AFTER_DEV' in c['open_contracts'][5]['lock_stage'])

pending=set(stage_map);completed=set()
while pending:
    ready={s for s in pending if set(stage_map[s]['depends_on'])<=completed}
    check('DAG frontier '+','.join(sorted(pending)),bool(ready))
    completed.update(ready);pending-=ready
check('parallel ablations independent',stage_map['S4']['depends_on']==stage_map['S5']['depends_on']==['S3'])
check('test after calibration audit',stage_map['S7']['depends_on']==['S6'])

negative_cards=[
 ('Q21','spatial role leak',lambda x:x['config'].__setitem__('role_proxy',True)),
 ('Q25','class-only gold head',lambda x:x['config'].__setitem__('gold_prediction_output',True)),
 ('Q01','tree joint encoder loss',lambda x:x.__setitem__('loss_id','Q_JOINT')),
 ('Q09','terrain gate bypass',lambda x:x['required_gates'].remove('G2T')),
 ('Q09','future label feature',lambda x:x['inputs'].__setitem__('future_label_metadata_visible',True)),
 ('Q22','one-frame adapter mismatch',lambda x:x['inputs'].__setitem__('history_shape',[3,2,5,18])),
 ('Q09','gating without rank',lambda x:x['config'].__setitem__('rank',False)),
 ('Q09','unbounded search parent',lambda x:x.__setitem__('hyperparameter_source','NEW_SEARCH'))]
for cid,name,mutate in negative_cards:
    bad=copy.deepcopy(cards[cid]);mutate(bad);check('reject experiment '+name,bool(experiment_errors(bad)))

# Independent small arithmetic probes for the new specifications, not ML tests.
losses={'m1':[1.,3.],'m2':[10.]}
risk=sum(sum(v)/len(v) for v in losses.values())/len(losses)
weighted=sum(v/(len(losses)*len(values)) for values in losses.values() for v in values)
check('match equal risk differs from row average',risk==weighted==6. and risk!=14/3)
check('half-weight swapped views preserve total weight',sum(2*(1/(2*len(values)))/2 for values in losses.values() for _ in values)==1.)
# Hazard rows [blue,red,survive]; conditional hazards must compose before swap averaging.
survival=1.;blue=red=0.
for b,r,s in [(0.1,0.2,0.7),(0.2,0.1,0.7)]:
    blue+=survival*b;red+=survival*r;survival*=s
check('hazard cumulative probabilities normalize',math.isclose(blue+red+survival,1.) and math.isclose(blue,0.24))
check('disjoint gold bins may decrease',sum([100,-40,20])==80)
check('mean RMSE differs from RMSE pooled across seeds',(math.sqrt(1)+math.sqrt(9))/2 != math.sqrt((1+9)/2))
protocol=(ROOT/'EXPERIMENT_TRAINING_PROTOCOL_KO.md').read_text(encoding='utf-8')
main=(ROOT/'FULL_DESIGN_SPECIFICATION_V5_KO.md').read_text(encoding='utf-8')
check('integrated section contains exact protocol',protocol[protocol.index('\n\n')+2:] in main)
for cid in cards:check('card documented '+cid,('| '+cid+' |') in protocol)

mp=load('MULTIPATCH_DESIGN_V5.json');mr=load('MULTIPATCH_EXPERIMENT_REGISTRY_V5.json');mt=load('MULTIPATCH_TRAINING_RECIPES_V5.json');ma=load('MULTIPATCH_DATA_AUDIT.json')
mc={x['id']:x for x in mr['cards']};ms=mp['active_split']
check('active MP profile explicit',c['active_split_profile_id']==ms['id'] and c['experiment_protocol']['registry']=='MULTIPATCH_EXPERIMENT_REGISTRY_V5.json')
check('old split immutable alongside new',c['split_profiles']['LEGACY_15_14_V15_T16']==base['split'] and c['split']==base['split'])
check('40 unique active MP cards',len(mc)==len(mr['cards'])==40 and set(mc)==set(c['experiment_protocol']['required_cards']))
check('new split chronological',max(tuple(map(int,p.split('.'))) for p in ms['train_patches']) < tuple(map(int,ms['validation_patch'].split('.'))) < tuple(map(int,ms['benchmark_test_patch'].split('.'))))
check('common latest STOP',ms['stop']=={'patches':['15.20'],'buckets':[80,99],'shared_across_arms':True})
check('unused older STOP not silently fit',not ms['unused_reserved']['training_allowed'])
check('prior exposure explicit',ms['benchmark_status']=='PREVIOUSLY_EXPOSED_BENCHMARK_NOT_PRISTINE' and not ms['previous_model_exposure_erased_by_new_split'])
check('new unseen corpus not invented',ms['fresh_confirmatory_test']['status']=='NOT_IDENTIFIED_IN_AUDITED_HOLDINGS')
check('new test not 15.16',mt['development']['test']=='15.22 BENCHMARK_TEST_PRIOR_EXPOSURE')
check('matched final run budget',mr['counts']['final_seed_runs']==sum(len(x['seeds']) for x in mc.values())==200)
check('bounded MP search budget',len(set(mr['search_groups']))==23 and mr['counts']['maximum_search_seed_runs']==23*12*2==552)
check('active eight contrast correction',len(mr['primary_claim_contrasts'])==mr['multiplicity']['n']==8)
check('unchanged primary target',c['fight_labels']['primary']==base['fight_labels']['primary'])
check('no opaque patch embedding',not mp['feature_partition']['PATCH_CONTEXT']['runtime_patch_id_embedding'])
check('numeric skill expansion not silently added',not mp['feature_partition']['optional_numeric_ability_extension']['enabled'])
check('single patch is latest train',ms['single_train_patch']==max(ms['train_patches'],key=lambda p:tuple(map(int,p.split('.')))))
check('index audit not production training',not mp['training_executed'] and ma['npz_payloads_opened']==0 and ma['raw_payloads_opened']==0)
check('legacy source support remains unaccepted',not load('MULTIPATCH_CACHE_SUPPORT_REVIEW.json')['preparation_status']['training_ready'])
for s in load('MULTIPATCH_CACHE_SUPPORT_REVIEW.json')['sources']:
    check('cache support source '+pathlib.Path(s['path']).name,sha(pathlib.Path(s['path']))==s['sha256'])
for source,v in ma['indexed_caches'].items():
    check('live index evidence '+source,sha(pathlib.Path(v['index_path']))==v['index_sha256'])
    check('bounded metadata probe '+source,all(g['metadata_probe_n']<=3 and g['metadata_patch_mismatch']==0 and g['metadata_errors']==0 for g in v['groups'].values()))

def mp_errors(card):
    err=[]
    if card['split_profile_id']!=ms['id']:err.append('legacy_split_fallback')
    expected=['15.20'] if card['training_regime']=='SINGLE_MATCHED' else ms['train_patches']
    if card['fit_patches']!=expected:err.append('fit_patch_leak')
    if card['feature_partition']=='COMMON':
        if any(card['config'][k]!=v for k,v in mp['feature_partition']['feature_mapping']['COMMON_config'].items()):err.append('context_leak')
    if card['recipe_profile']=='MP_MATCHED' and card['training_regime']=='MULTI_FULL':err.append('regime_budget')
    if set(card['required_mp_gates'])!=set(mr['required_upstream']):err.append('mp_gate_bypass')
    if card['hyperparameter_source'] not in mr['search_groups']:err.append('search_parent')
    return err
for cid,card in mc.items():
    check('MP card '+cid,not mp_errors(card))
    check('MP loss unchanged '+cid,card['loss_id'] in recipes['losses'])
for backbone,parent in [('QH','MP.Q09'),('QT','MP.Q01')]:
    check('D full alias reuse '+backbone,mr['comparison_aliases']['D_F.'+backbone]==parent)
    for cell in ['A','B','C_M','D_M','C_F']:
        check('four cells and full '+cell+backbone,('MP.'+cell+'.'+backbone) in mc)
    check('paired common configs '+backbone,mc['MP.A.'+backbone]['config']==mc['MP.C_M.'+backbone]['config'])
    check('paired context configs '+backbone,mc['MP.B.'+backbone]['config']==mc['MP.D_M.'+backbone]['config']==mc[parent]['config'])
for compare in mr['primary_claim_contrasts']:
    check('MP contrast references '+compare['id'],all(x in mc for x in compare['comparison'].split(' vs ')))
for name,mutate in [
 ('legacy implicit split',lambda x:x.__setitem__('split_profile_id','LEGACY_15_14_V15_T16')),
 ('test in fit',lambda x:x['fit_patches'].append('15.22')),
 ('context backdoor',lambda x:x['config'].__setitem__('terrain','supported_dynamic_descriptors')),
 ('missing MP gate',lambda x:x['required_mp_gates'].pop())]:
    bad=copy.deepcopy(mc['MP.A.QH']);mutate(bad);check('reject MP '+name,bool(mp_errors(bad)))
def match_n(n18,n19,n20):return 3*min(n20//3,n18,n19,n20)
check('matching counts balanced divisible',match_n(90,100,100)==99 and match_n(90,100,100)//3==33)
check('matching respects scarce patch',match_n(5,100,100)==15)
check('matching empty patch blocks',match_n(0,100,100)==0)
patch_match_risk=(1+9)/2
check('patch balance differs from row pooling',patch_match_risk==5 and patch_match_risk!=(100*1+9)/101)
mp_text=(ROOT/'MULTIPATCH_PROTOCOL_KO.md').read_text(encoding='utf-8')
check('integrated MP narrative exact',mp_text[mp_text.index('\n\n')+2:] in main)
check('split feature execution interfaces present',{'split_profile_id','exposure_ledger_hash','matched_membership_hash'}<=set(records['ExperimentRunRecord']['fields']))

docs=['FULL_DESIGN_SPECIFICATION_V5_KO.md','INTERFACES_AND_TASKS_KO.md','EXPERIMENT_TRAINING_PROTOCOL_KO.md','MULTIPATCH_PROTOCOL_KO.md','README_KO.md']
for name in docs:
    text=(ROOT/name).read_text(encoding='utf-8')
    check('UTF8/fences '+name,'\ufffd' not in text and text.count('```')%2==0)
for gate in ['G0','G1','G2','G2T','G3','G4','G5','G6','G7']:
    check('gate documented '+gate,gate in (ROOT/docs[0]).read_text(encoding='utf-8'))
report={'status':'PASS_STATIC_CONTRACT_AND_SYNTHETIC_PROBES','checks_passed':len(checks),'checks':checks,'scope':'source hashes, preserved definitions, legacy templates and active MP cards, split chronology, previous exposure, matched-size controls, loss/metric references, stage DAG and negative mutations','not_verified':['full-corpus schema/retention','actual skill rank replay','actual terrain geometry','production implementation','optimizer and model execution','model performance','causal identification'],'legacy_templates_checked':len(cards),'experiment_cards_checked':len(mc),'production_gates_passed':False,'raw_matches_read_by_validator':0,'separate_inventory_audit':'MULTIPATCH_DATA_AUDIT.json','models_trained':0,'new_empirical_labels_produced':0}
(ROOT/'VALIDATION_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=[]
for p in sorted(ROOT.iterdir()):
    if p.is_file() and p.name!='ARTIFACT_MANIFEST.json':manifest.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'version':'integrated-design-v5','files':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'checks_passed':len(checks),'production_gates_passed':False}))
