"""Resolve the v5 discussion into executable-spec cards. Never trains or reads matches."""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'paper_ablation_v5/EXPERIMENT_MATRIX.json'

def write(name, obj):
    (ROOT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def build():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    ref = copy.deepcopy(source['reference'])
    ref.pop('id')
    ref.update(hierarchical_aggregation=True, gold_prediction_output=True,
               remove_spatial_dependency_closure=False)
    seeds = source['training']['seeds']
    recipes = {
      'version':'v5-experiments-r1', 'status':'SPECIFIED_PENDING_IMPLEMENTATION_AND_LOCK',
      'neural':{
        'optimizer':'AdamW', 'learning_rates':[0.0003,0.001],
        'hidden_widths':[64,128,256], 'dropouts':[0.0,0.2], 'candidates_per_search_group':12,
        'weight_decay':0.0001, 'adam_betas':[0.9,0.999], 'adam_epsilon':1e-8, 'layernorm_epsilon':1e-5, 'batch_rows':256, 'max_epochs':100,
        'early_stop_patience_epochs':10, 'early_stop_min_delta':0.0,
        'gradient_clip_global_norm':1.0, 'embedding_dim':32,
        'gru_layers':1, 'message_blocks':2, 'attention_blocks':2, 'attention_heads':4,
        'mlp_hidden_layers':2, 'activation':'GELU', 'normalization':'LayerNorm',
        'dropout_location':'after each hidden activation, not output or masks',
        'initialization':'framework Linear defaults; seed fixed; GRU defaults; exact framework version in locked ModelCard',
        'restore_best_STOP_checkpoint':True, 'mixed_precision':False,
        'minibatch_sampling':'uniform shuffle without replacement over union of task-supported FIT rows',
        'loss_estimator':'N/batch_size times sum of precomputed global row-task weights times losses; no batch support renormalization',
        'P_STOP_metric':'P_PRIMARY', 'Q_STOP_metric':'Q_RAW',
        'nonfinite_policy':'fail run and preserve error, never silently change recipe'},
      'tree':{
        'implementation':'LightGBM; exact package version and objective mapping must pass adapter fixtures before lock',
        'num_leaves':[15,31,63], 'min_data_in_leaf':[50,200], 'learning_rates':[0.03,0.1],
        'candidates_per_search_group':12, 'max_rounds':3000, 'early_stop_rounds':100,
        'max_depth':-1, 'feature_fraction':1.0, 'bagging_fraction':1.0,
        'bagging_freq':0, 'lambda_l1':0.0, 'lambda_l2':0.0, 'num_threads':4,
        'missing_numeric':'native missing plus explicit support channels',
        'categorical':'declared ID columns as categorical; no ordinal numerical champion meaning',
        'Q_class_objective':'three-class cross entropy',
        'gold_objective':'normalized Huber delta=1 via validated custom objective; hessian floor=1e-6 for numeric optimization only',
        'P_structure_objective':'Poisson count likelihood',
        'P_hazard_objective':'three-class interval-conditional cross entropy',
        'head_training':'independent estimators; shared selected hyperparameter tuple per card, no shared learned encoder',
        'STOP':'each estimator own task loss; card selection uses joint P_PRIMARY or Q_RAW on DEV',
        'class_weight':'none; match weights only', 'deterministic':True,
        'zero_support_or_single_class_head':'zero support -> unsupported head; single class -> explicit constant distribution; report counts',
        'identical_seed_predictions':'report as deterministic, not five independent training discoveries'},
      'losses':{
        'Q_JOINT':{'class':'match-risk CE', 'gold':'independently support-normalized match-risk Huber((gold_hat-gold)/s_Q, delta=1)', 'coefficients':{'class':1,'gold':1}, 'shared_encoder':True},
        'Q_CLASS':{'class':'match-risk CE','gold':None,'coefficients':{'class':1,'gold':0},'shared_encoder':True},
        'Q_TREE':{'class':'match-risk CE','gold':'independent match-risk normalized Huber regression','shared_encoder':False,'not_joint_neural_loss':True},
        'P_JOINT':{'gold':'mean of supported disjoint-bin normalized Huber losses','terminal':'mean of supported at-risk interval conditional CE losses','structure':'mean of supported bin-channel Poisson NLL losses','coefficients':{'gold':1,'terminal':1,'structure':1},'risk':'each component: 0.5 overall match risk + 0.5 mean supported family match risks','shared_encoder':True},
        'P_TREE':{'targets':'same P_JOINT targets and weights; independent estimator per bin/channel/hazard','shared_encoder':False,'not_joint_neural_loss':True}},
      'weighting':{
        'match_risk':'mean over matches with supported rows, then mean over supported rows within match',
        'task_support':'separate denominator per class/gold/bin/channel and family; missing targets never become zero',
        'P_family_membership':'inherited family definition; used only for loss/evaluation, never as future-dependent model input',
        'P_absent_family':'report unsupported; use mean of FIT-supported families and publish exact fixed set; never invent missing-family score',
        'Q_gold_scale':'max(1 gold, sqrt(match-weighted FIT variance of raw observed gold score)); divide without centering',
        'P_gold_scales':'same FIT-only weighted SD, floor 1 gold, separately per disjoint bin; evaluation scales separately per family at 180s',
        'scale_fit_scope':'FIT only; store counts, exact weights, values and hash; no STOP/DEV/CAL/TEST fitting',
        'loss_coefficients_are_objective_prices':False},
      'development':{
        'seeds':seeds[:2], 'final_seeds':seeds,
        'fit':'15.14 FIT', 'early_stop':'15.14 STOP', 'select':'15.15 DEV',
        'calibrate':'15.15 CAL', 'audit':'15.15 AUDIT', 'test':'15.16 TEST',
        'external':'16.13 KR/NA', 'refit_FIT_plus_STOP':False,
        'search_groups':['Q01','Q02','Q03','Q04','Q05','Q06','Q07','Q08','Q09','Q26','Q27','P_TREE','P_HIER'],
        'search_group_count':13, 'maximum_search_seed_runs':13*12*2,
        'selection':'mean of two development-seed raw primary DEV scores; smaller parameter count then lexicographic hyperparameter tuple breaks exact ties',
        'input_ablation_recipe':'freeze selected Q09 width/LR/dropout for Q10-Q23/Q25/Q28; Q24 inherits Q01; no per-ablation retuning',
        'architecture_capacity':'same width search budget; publish actual parameters/FLOPs/latency; no claim exact parameter matching',
        'lock_phases':['DEV_CANDIDATE_LOCK: exact candidate tuple and predeclared search plan before that candidate fit','FINAL_RECIPE_LOCK: after DEV selection, before final fit and input ablations'],
        'final_fit':'retrain each selected card from scratch per seed; identical already-completed runs may be reused only with identical full hashes',
        'AUDIT_failure':'bug -> versioned repair, invalidate affected predictions; scientific weakness -> report; no adaptive tuning on AUDIT/TEST'},
      'calibration':{
        'Q':'single scalar temperature on CAL only, weighted class logloss',
        'log_temperature_bounds':[-3,3], 'optimizer':'bounded scalar golden-section, tolerance 1e-6, max 200 iterations; compare endpoints too',
        'probability_floor':1e-12, 'tree_logits':'log(max(raw_probability,1e-12))',
        'fit_per':'card and seed; never per TEST subgroup',
        'primary':'raw probabilities; calibrated values secondary',
        'insufficient_CAL_support':'raw metrics remain; calibrated metrics unavailable with reason; no TEST fallback',
        'P':'NONE in mandatory plan; report uncalibrated calibration diagnostics'},
      'execution':{'models_trained':0,'training_authorized_by_document_edit':False,'max_total_CPU_threads':4,'nested_parallelism':False,'primary_pretraining':'none','full_model_embedding_reused_in_ablation':False}
    }
    metrics = {
      'Q_RAW':{'primary':'match-weighted raw multiclass logloss','secondary':['multiclass Brier','class recall','gold MAE/RMSE on gold-supported rows','10 fixed equal-width top-confidence ECE bins','terminal/economic strata'],'rows':'same predeclared class-support intersection within contrast; separate gold intersection; same matches/seeds; coverage always reported'},
      'Q_CAL':{'primary':None,'secondary':['CAL-temperature logloss','Brier','ECE'],'rows':'same frozen evaluation row set as raw; never choose model by TEST calibration'},
      'P_PRIMARY':{'primary':'macro over FIT-supported families of (match-weighted 180s gold RMSE / FIT family 180s SD floored at 1 gold)','secondary':['unscaled gold RMSE/MAE at 60/120/180/300s','cumulative terminal logloss/Brier','structure MAE and Poisson deviance','gold calibration intercept/slope'],'rows':'joint gold support for comparison; hazard and structure independently supported; common supported family set and denominators'},
      'AGGREGATION':{'seeds':'mean of individual-model scores plus min/max and SD; no implicit ensemble','uncertainty':'paired match bootstrap 2000 draws, seed=20261016, same draws for contrasts; Q logloss: average fixed-seed per-match losses before resampling','nonlinear_metrics':'P RMSE/macro and other nonlinear metrics: recompute complete metric for each seed on each match resample, THEN average seed metrics; never sqrt of seed-averaged MSE as replacement','interval':'95% percentile CI; conditional on trained seeds, not retraining uncertainty','primary_p':'two-sided centered paired-bootstrap p=(1+count(abs(delta_star-delta_hat)>=abs(delta_hat)))/2001; approximate test, report Monte Carlo resolution','multiplicity':'Holm across five primary contrasts; seven-family analyses separate secondary family; no uncorrected 35-cell confirmatory claims','strata':['cohort','frame_age','patch','region','objective_family','terrain_support','rank_support','role_support']}
    }
    cards=[]
    additions=[{'id':'Q28','name':'ALLPHASE_NO_PLAYER_POSITION','group':'cohort_position_control','overrides':{'profile':'Q_allphase_v5','player_spatial':False,'role_proxy':False,'remove_spatial_dependency_closure':True}}]
    stages = {**{f'Q{i:02}':'S3' for i in range(1,10)}, **{f'Q{i:02}':'S4' for i in range(10,23)},'Q23':'S5','Q24':'S5','Q25':'S4','Q26':'S3','Q27':'S3','Q28':'S5'}
    for item in source['cards']+additions:
        cfg=copy.deepcopy(ref);cfg.update(copy.deepcopy(item['overrides']))
        tree=cfg['architecture']=='lightgbm'
        cid=item['id']
        parent='Q01' if cid=='Q24' else ('Q09' if cid not in recipes['development']['search_groups'] else cid)
        loss='Q_TREE' if tree else ('Q_CLASS' if cid=='Q25' else 'Q_JOINT')
        inputs={'base18_order':'PIPELINE_CONTRACT_V5.json#/feature_contract/ordered_18','history_shape':[cfg['history_frames'],2,5,18], 'cutoff':'tau=first_kill_ms-10000','feature_view':'fixed shared player/team/world descriptors and pair arrays', 'feature_dependency_closure_required':True, 'row_policy':'Q_COMMON', 'future_label_metadata_visible':False}
        gates=['G1','G2','G3','G4','G5']
        if cfg['knowledge']=='semantic_tags' or cfg['rank']:gates.append('G0')
        if cfg['terrain']!='off':gates.append('G2T')
        locks=['P01','P02','P06']
        if cfg['role_proxy']:locks.append('P03')
        if cfg['terrain']!='off':locks.append('P04')
        if cfg['rank']:locks.append('P05')
        if cfg['knowledge']=='semantic_tags' or cfg['rank']:locks.append('P07')
        cards.append({'id':cid,'name':item['name'],'requirement':'MANDATORY_FOR_FULL_V5_CLAIMS','group':item['group'], 'config':cfg,'inputs':inputs,'model_spec':'MODEL_OPERATORS_V5.json','loss_id':loss,'recipe_id':'tree' if tree else 'neural','hyperparameter_source':parent,'metric_ids':['Q_RAW','Q_CAL'],'stage':stages[cid],'required_gates':gates,'required_parameter_locks':locks,'seeds':seeds,'status':'SPECIFIED_PENDING_GATES','training_executed':False})
    for cid,alias,src in [('P_TREE','P01','Q01'),('P_HIER','P02','Q09')]:
        c=copy.deepcopy(next(x for x in cards if x['id']==src))
        c.update(id=cid,name=cid,source_discussion_alias=alias,group='P_prediction',stage='S3',hyperparameter_source=cid,loss_id='P_TREE' if cid=='P_TREE' else 'P_JOINT',metric_ids=['P_PRIMARY'])
        c['config']['profile']='P_landmarks_v5'
        c['required_parameter_locks'].remove('P02')
        c['inputs'].update(cutoff='actual preterminal landmark t',row_policy='P_COMMON',targets='four nonoverlapping gold bins, conditional hazards, six structure count channels')
        cards.append(c)
    extensions=[x for x in copy.deepcopy(source['conditional_extensions']) if x['id']!='S_INVARIANCE']
    for e in extensions:
        e.update(requirement='CONDITIONAL_ON_CLAIM',enabled=False,status='TEMPLATE_REQUIRES_NEW_LOCKED_CARDS',selection_data='DEV only',evaluation_data='fresh untouched holdout or preregistered existing holdout plan before unblinding')
        if e['id']=='S_PATCH':e['design']='same frozen trained weights evaluated with correct external package versus frozen TRAIN package; common schema/support; unsupported/removed mechanics remain STRUCTURAL_NA; no external outcomes used to fit either arm'
    extensions += [
      {'id':'S_P_MECHANISM','requirement':'CONDITIONAL_ON_CLAIM','trigger':'mechanism claims about P, not just comparative predictive accuracy','design':'repeat the claimed hierarchy/engine/knowledge/terrain contrasts on P with same P losses and P_PRIMARY','enabled':False,'status':'TEMPLATE_REQUIRES_NEW_LOCKED_CARDS'},
      {'id':'S_CAPACITY','requirement':'CONDITIONAL_ON_CLAIM','trigger':'claim gains independent of parameter count or training compute','design':'DEV-only parameter/compute matched comparison; same inputs and row sets','enabled':False,'status':'TEMPLATE_REQUIRES_NEW_LOCKED_CARDS'},
      {'id':'S_WORLD_GRAPH','requirement':'OPTIONAL_EXPLORATION','trigger':'study variable world graph or raster/CNN beyond shared descriptors','design':'new feature manifests, adapter parity or explicit additional-information claim; no automatic reuse of primary architecture contrast','enabled':False,'status':'TEMPLATE_REQUIRES_NEW_LOCKED_CARDS'},
      {'id':'S_OTHER_ARCH','requirement':'OPTIONAL_EXPLORATION','trigger':'study TCN, snapshot-GNN-before-time, GAT, raw text embeddings or relation knowledge','design':'new cards and DEV budget; inherited v4 aliases are not executable cards','enabled':False,'status':'TEMPLATE_REQUIRES_NEW_LOCKED_CARDS'}]
    stages=[
      {'id':'S0','depends_on':[],'action':'source/phase/support audit and TRAIN boundary locks; G1/G3; G0/G2/G2T may develop in parallel'},
      {'id':'S1','depends_on':['S0'],'action':'build prefix, labels and common features; split isolation; required invariance/parity/label accounting fixtures'},
      {'id':'S2','depends_on':['S1'],'action':'lock feature manifests, family weights, model operators and 13 search-group plans; get feature-scoped G0-G5 receipts'},
      {'id':'S3','depends_on':['S2'],'action':'FIT/STOP development, at most 12 configs x 2 seeds per search group; DEV choose recipes; final 5-seed core architecture and P runs'},
      {'id':'S4','depends_on':['S3'],'action':'from-scratch matched ablations using frozen Q09 recipe; no copied trained embeddings'},
      {'id':'S5','depends_on':['S3'],'action':'allphase Q23/Q24/Q28 with frozen parent recipes; common fresh comparison and allphase age/spatial comparison'},
      {'id':'S6','depends_on':['S4','S5'],'action':'CAL per-card temperature; AUDIT correctness; freeze run/row/contrast manifest before TEST; optional experiments declared before holdout unblinding'},
      {'id':'S7','depends_on':['S6'],'action':'TEST once plus supported external inference; paired reports, multiplicity and limitations; no redesign on test outcomes'}]
    registry={'version':'v5-experiments-r1','status':'SPECIFIED_NOT_EXECUTED_NOT_PREREG_FROZEN','source_path':'../paper_ablation_v5/EXPERIMENT_MATRIX.json','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cards':cards,'conditional_extensions':extensions,'primary_claim_contrasts':source['primary_claim_contrasts'],'critical_interactions':source['critical_interactions'],'mandatory_nontraining_checks':['prefix append invariance','within-query player permutation and both pair axes','team swap output equivariance','source-information parity','spatial dependency closure','unknown versus zero','label/gold support separation','nonoverlap gold accounting','split and preprocessing isolation'],'execution_stages':stages,'counts':{'Q_core':28,'P_minimum':2,'final_seed_runs':150,'maximum_search_seed_runs':312,'final_plus_search_nominal_upper_seed_runs':462,'includes_optional':False,'unit':'card x seed; multi-output tree has multiple estimators; exact duplicate runs can be reused by hashes'},'integration_changes':['P model IDs renamed P_TREE/P_HIER to avoid P01/P02 parameter lock collision','Q28 closes the previously promised allphase nonspatial control','S_INVARIANCE moved to mandatory nontraining gates','S_PATCH isolates package swap with fixed weights; no external retraining','old model aliases archived; no second conflicting factorial'], 'unmet_gates_block_full_claim':True,'partial_development':'DEV_BASE_* separate manifest; never substitute for Q09 or call the full 30-card plan complete'}
    registry['row_policies']={
      'Q_COMMON':{'base':'frozen detector/non-age eligibility plus profile age eligibility','class':'class-supported intersection of predeclared input-eligible rows for each contrast; fixed before scores','gold':'independent gold-supported intersection, not necessarily class row set','missing_inputs':'declared UNKNOWN masks do not automatically exclude a row; no feature imputation to bypass missing package gates','pairing':'within contrast match keys and query IDs identical; seed runs use identical sets','prediction_failure':'fail run or disclose abstention with original denominator; never silently shrink comparison intersection after observing errors'},
      'P_COMMON':{'base':'all eligible actual preterminal frame landmarks','targets':'independent supported sets for each bin, cumulative horizon, hazard risk set and structure channel','comparison':'same landmark IDs and FIT-fixed family set for each metric; missing terminal gold does not remove supported terminal class','unsupported_family':'explicit NA and denominator, no replacement by zero'}}
    write('EXPERIMENT_REGISTRY_V5.json',registry)
    write('TRAINING_RECIPES_V5.json',recipes)
    write('EVALUATION_PROTOCOL_V5.json',metrics)
    descriptions = {
      'Q01':('동일 raw 이력·pair flatten','LightGBM'),
      'Q02':('동일 raw 이력·pair flatten','flat MLP'),
      'Q03':('동일 정보; 계층 없음','시간 MLP + flat attention'),
      'Q04':('동일 정보; 계층 없음','개인 GRU + flat readout'),
      'Q05':('동일 시간·관계; 계층 없음','GRU + typed message + flat readout'),
      'Q06':('시간 MLP·관계 pooling','H00'), 'Q07':('GRU·관계 pooling','H10'),
      'Q08':('시간 MLP·typed message','H01'), 'Q09':('전체 기준 입력','H11 / 세 readout'),
      'Q10':('I 직접 readout 제거','H11'), 'Q11':('T 직접 readout 제거','H11'), 'Q12':('W 직접 readout 제거','H11'),
      'Q13':('비스킬·비지형 overlay E0 held','H11'), 'Q14':('동일 overlay E1 events-only','H11'),
      'Q15':('ID; rank 없음; gating 없음','H11'), 'Q16':('ID+태그; rank 없음; gating 없음','H11'),
      'Q17':('ID+rank; 의미/gating 없음','H11'), 'Q18':('ID+태그+rank concat; gating 없음','H11'),
      'Q19':('지형 제거; raw XY 유지','H11'), 'Q20':('기본 정적 지형만','H11'),
      'Q21':('선수 위치·공간 파생·role 제거','H11'), 'Q22':('최근 1프레임; 영구 prefix 유지','H11'),
      'Q23':('allphase; 나머지 Q09 동일','H11'), 'Q24':('allphase; 나머지 Q01 동일','LightGBM'),
      'Q25':('입력 동일; 골드 head/loss 제거','H11'), 'Q26':('Q01 동일 정보, role 정렬','LightGBM'),
      'Q27':('Q02 동일 정보, role 정렬','flat MLP'), 'Q28':('allphase; 공간·role 제거','H11'),
      'P_TREE':('일반 landmark; Q01 정보; P targets','LightGBM 별도 heads'),
      'P_HIER':('일반 landmark; Q09 정보; P targets','H11 / P heads')}
    lines=['<!-- BEGIN_GENERATED_CARD_TABLE -->','','| ID | 입력/경로 변경 | 모델 | 손실 | 평가 | 단계 |','|---|---|---|---|---|---|']
    for c in cards:
        d,m=descriptions[c['id']]
        lines.append(f"| {c['id']} | {d} | {m} | {c['loss_id']} | {' / '.join(c['metric_ids'])} | {c['stage']} |")
    lines.append('<!-- END_GENERATED_CARD_TABLE -->')
    protocol=ROOT/'EXPERIMENT_TRAINING_PROTOCOL_KO.md'
    txt=protocol.read_text(encoding='utf-8')
    table='\n'.join(lines)
    if '<!-- GENERATED_CARD_TABLE -->' in txt:
        txt=txt.replace('<!-- GENERATED_CARD_TABLE -->',table)
    else:
        txt=re.sub(r'<!-- BEGIN_GENERATED_CARD_TABLE -->.*?<!-- END_GENERATED_CARD_TABLE -->',lambda _:table,txt,flags=re.S)
    protocol.write_text(txt,encoding='utf-8')
    return registry,recipes,metrics

if __name__=='__main__':
    registry,_,_=build()
    print('Resolved',len(registry['cards']),'mandatory cards; no training.')
