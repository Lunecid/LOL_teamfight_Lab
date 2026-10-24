"""Build the integrated research contract. Does not read matches or train models."""
import copy, hashlib, json, pathlib
from build_experiments import build as build_experiments
from build_multipatch import build as build_multipatch, sync_narrative
ROOT=pathlib.Path(__file__).resolve().parent
PARENT=ROOT.parent
def read(rel): return json.loads((PARENT/rel).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

base=read('knowledge_15_14_v2/PIPELINE_CONTRACT_V4.json')
terrain=read('terrain_v1/PIPELINE_TERRAIN_EXTENSION_V4_1.json')
c=copy.deepcopy(base)
c.update(version='rift-integrated-v5',date_kst='2026-10-16',status='IMPLEMENTABLE_DESIGN_WITH_GATES_NOT_PREREG_FROZEN',training_executed=False,new_match_labels_generated=False)
c['inherited_source_sha256']=c.pop('source_sha256')
sources=['knowledge_15_14_v2/PIPELINE_CONTRACT_V4.json','knowledge_15_14_v2/END_TO_END_DESIGN_KO.md','knowledge_15_14_v2/semantic_validation_report.json','PIPELINE_FOUR_POINT_REVIEW_KO.md','terrain_v1/PIPELINE_TERRAIN_EXTENSION_V4_1.json','terrain_v1/TERRAIN_ENCODER_DESIGN_KO.md','terrain_v1/SOURCE_AUDIT.json','terrain_v1/VALIDATION_REPORT.json','BASIC_FEATURE_CONTRACT_V1.json','ALL_OBJECTIVE_DESIGN_V3.json']
c['source_manifest']=[{'path':'../'+rel,'sha256':sha(PARENT/rel)} for rel in sources]
c['definition_precedence']=['PIPELINE_CONTRACT_V5.json and FULL_DESIGN_SPECIFICATION_V5_KO.md must agree; conflicts block the affected stage','v5 supersedes v4 proposal on cohort extensions, skills, ordering, terrain and asset interpretation','Previous source packages and executed results remain immutable; no historical result is relabeled']
c['status_axes']={
 'decision':['FIXED_V5','CANDIDATE_REQUIRES_TRAIN_AUDIT','DISABLED_RESEARCH'],
 'evidence':['VERIFIED_ARTIFACT','PARTIALLY_SUPPORTED','NOT_VALIDATED'],
 'execution':['NOT_IMPLEMENTED','STATIC_CHECKED','PRODUCTION_VERIFIED'],
 'fixed_does_not_mean_empirically_verified':True}
c['objective_families']=['BARON','HERALD','ELEMENTAL_DRAGON','DRAGON_SOUL','ELDER','VOID_GRUB','ATAKHAN']
c['cohort']['profile_id']='Q_fresh_v5'
c['cohort']['interpretation']='retrospective fresh-frame conditional population, not representative of every fight'
c['cohort_profiles']={
 'Q_fresh_v5':{'decision':'FIXED_V5','age_ms_min':0,'age_ms_max_inclusive':10000,'requires_prior_frame':True,'same_non_age_eligibility':True},
 'Q_allphase_v5':{'decision':'FIXED_V5','age_ms_min':0,'age_ms_max_inclusive':None,'requires_prior_frame':True,'same_non_age_eligibility':True,'invalid_or_missing_clock':'exclude with explicit support reason','long_gap_policy':'retain valid timestamps; age>60000 separate stratum; no trajectory interpolation','changes_label_or_query_time':False},
 'P_landmarks_v5':{'decision':'FIXED_V5','unit':'every eligible actual preterminal frame','Q_age_filter_applied':False}
}
c['cohort_audit']={'expected_retention_uniform_regular_clock':1/6,'is_measured_retention':False,'denominator':'non-age-eligible detector components with a supported prior actual frame','report_before_and_after_label_support':True,'counts':['fights','unique_matches','per_family_support','overlapping_exclusion_reasons'],'age_bins_ms':[[0,10000],[10000,20000],[20000,30000],[30000,40000],[40000,50000],[50000,60000],[60000,None]],'bin_boundaries':'first closed; subsequent lower open, upper closed','phase_fields':['f_mod_60000','tau_mod_60000','patch_specific_wave_spawn_phase','time_since_or_until_supported_objective_spawn'],'wave_spawn_equals_wave_arrival':False,'weighting_recovers_zero_probability_phases':False,'comparisons':['fresh-trained vs allphase-trained on same fresh rows','allphase performance by age with common nonspatial control'],'preexisting_h90_eligibility_reused':False}
c['feature_contract']['age_is_model_input']=True
c['feature_contract']['additional_blocks']=['skill_rank_v1','role_proxy_v1','terrain_descriptor_v1']
c['feature_contract']['extensions_never_overwrite_base18']=True
c['feature_contract']['feature_set_key']=['base_version','engine_arm','knowledge_arm','rank_arm','role_arm','terrain_arm','spatial_arm','history_count']
c['skills']={
 'decision':'FIXED_V5','evidence':'PARTIALLY_SUPPORTED','execution':'NOT_IMPLEMENTED',
 'slots':['Q','W','E','R'],'passive_rank':'not forced into QWER rank vector',
 'input_fields':['tag_present','tag_supported','skill_rank','rank_support','learned_state','current_form','cast_readiness','effect_active','as_of_ms','known_at_ms'],
 'learned_states':['LEARNED','UNLEARNED','UNKNOWN'],
 'rank_absence_requires_complete_prefix':True,'learned_implies_cast_ready':False,'learned_implies_effect_active':False,
 'rank_reconstruction':'validated initialization plus normalized unique upgrade transitions plus explicit automatic/shared-rank rules',
 'duplicate_policy':'preserve source identities; quarantine incompatible transitions; no universal count or arbitrary time-near dedup',
 'exception_classes':['selectable_at_level_one','automatically_granted','automatic_rank_growth','shared_form_rank','copied_or_stolen_ability','awakened_variant','level_gated_effect'],
 'unknown_gate':'retain static tag and separate UNKNOWN learned/form mask; never multiply unknown by zero',
 'gating_is_new_information_given_id_and_rank':False,
 'factorial_inputs':[['K0','NO_RANK'],['K3','NO_RANK'],['K0','RANK'],['K3','RANK']],
 'gating_control':'K3+R concatenation versus K3+R explicit learned gating on identical rows',
 'acceptance':['prefix completeness audit','duplicate audit by patch','legal ranks/point budget and auto-rank exceptions','unknown does not become unlearned','static tag not converted to current activation']}
c['canonical_order']={
 'decision':'FIXED_V5','identity_key':'participant_id for joins only; never player identity embedding',
 'history_scope':'one permutation per query applied to every historical frame, current overlay, static fields and pair matrices',
 'different_query_permutations_allowed':True,'persistent_hidden_state_key':'participant_id; never sorted slot',
 'canonical_key':['champion_id','min(summoner_ids)','max(summoner_ids)','canonical serialized model-visible history and support fields for ties'],
 'exact_ties':'exchangeable identical model inputs; tie identity metadata not fed to model',
 'candidates':['CANONICAL','ROLE_PROXY'],
 'role_order':['TOP','JUNGLE','MID','BOT_CARRY','BOT_SUPPORT'],
 'role_proxy':{'decision':'CANDIDATE_REQUIRES_TRAIN_AUDIT','observation_end_ms':180000,'early_query_policy':'whole-team CANONICAL fallback; role support UNKNOWN','features':'actual prefix frames in (0,180000], start summoner IDs, lane-region occupancy, observed lane-CS progress','unique_smite_is':'candidate evidence, not verified role','assignment':'only a complete nonconflicting one-to-one supported assignment enables role order; otherwise whole-team fallback','min_lane_observations':2,'min_lane_share':2/3,'bot_tie_break':'higher observed lane-CS progress; unresolved ties cause fallback','smite_or_lane_conflict':'fallback, do not force a true-role label','proxy_and_support_given_to_all_models':True,'extra_early_frames_are_new_information':True,'thresholds_and_lane_regions_must_be_locked_before_DEV_comparison':True},
 'nonspatial_arm':'CANONICAL without XY-derived tie keys or role-proxy information; remove spatial tie-break fields',
 'selection':'paired DEV comparison, then freeze; never select on TEST'}
c['terrain']={k:copy.deepcopy(terrain[k]) for k in ['terrain_package','prefix_policy','spatial_policy','encoders','unobserved_runtime','llm_role','deterministic_role','comparison','implementation_gates']}
c['terrain'].update(decision='FIXED_V5',evidence='PARTIALLY_SUPPORTED',execution='NOT_IMPLEMENTED',runtime_enabled=False)
c['terrain']['activation_gate']='validated grid + coordinate transform + flag semantics + patch/queue applicability + supported variant'
c['terrain']['feature_arms']=['TERRAIN_OFF','STATIC_DESCRIPTORS','SUPPORTED_DYNAMIC_DESCRIPTORS','RASTER_SECONDARY']
c['terrain']['artifact_presence_does_not_imply_patch_activation']=True
c['terrain']['same_time_geometry']='historical positions use their own frame-time geometry; tau overlays are separate'
c['terrain']['shared_primary_view']='fixed declared player/object descriptors and pair arrays; graph cannot consume extra geography; variable world graph/raster is a separate information arm unless lossless parity is demonstrated'
c['architecture_controls']['new_three_level_hierarchy']='individual/team/rift readouts; fixed-width shared descriptors; world graph must have a new model-card ID and parity contract'
c['architecture_controls']['dynamic_skill_factorial']=c['skills']['factorial_inputs']
c['architecture_controls']['ordering_candidates']=c['canonical_order']['candidates']
c['architecture_controls']['K3_activation']='static extraction checks passed; production semantic review and dynamic skill gates remain pending'
c['architecture_controls']['no_spatial_ablation_removes_role_proxy']=True
c['model_roles']['Q']['profiles']=['Q_fresh_v5','Q_allphase_v5']
c['model_roles']['Q']['primary_metric']='multiclass_logloss_on_supported_labels'
c['model_roles']['Q']['P_weights_reuse']='from-scratch primary; TRAIN-only pretraining is a separately locked secondary arm'
c['model_selection']['selection']='P: DEV_180_second_family_macro_gold_RMSE; Q: DEV_multiclass_logloss within profile; never compare scores across different populations as model effects'
c['model_selection']['same_summary_control_meaning']='same canonical lossless player histories and identical declared nonplayer source information'
c['evaluation']['Q_primary_proposal']='superseded by fixed Q_primary field in v5'
c['evaluation']['Q_primary']='multiclass_logloss_on_common_supported_rows_per_profile'
c['evaluation']['representation_vs_identification']='predictive gain is not causal price validation'
c['model_roles']['V_episode']['name']='V_episode: adjusted acquisition episode association; not a future-gold value function'
c['asset_research']={
 'decision':'DISABLED_RESEARCH','enabled':False,
 'U_definition':'joint noncash asset residual incremental net-gold value relative to an explicit supported reference and policy',
 'A':{'score':'G(c)-G(s)+U(S_c,c;E)-U(S_s,s;E)','name':'realized net gold plus residual asset valuation change','equals_observed_at_E':False,'automatically_zero_mean_innovation':False},
 'B':{'score':'G(c)-G(s)+U(S_c,c;E)-U(S_tilde_s_to_c,c;E)','name':'realized net gold plus same-time reference residual-asset difference','equals_observed_at_fixed_E_if_both_U_zero':True,'reference_status':'UNDEFINED_NOT_IDENTIFIED','reference_options':['explicit accounting reference, not no-fight counterfactual','identified no-fight state distribution; causal requirements pending'],'endpoint_equality_sufficient_for_validity':False},
 'innovation_theory':{'notation':'M_t^E = E[G(E)-G(t)|H_t]','score':'G(c)-G(s)+M_c^E-M_s^E','requires':'same fixed ex-ante E or coherent shared random-target conditioning; correct conditional expectations and valid stopping conditions','U_is_M':False,'V_episode_is_M':False,'future_selected_E_c_plus_180_treated_as_known_at_s':False,'is_fight_causal_effect':False},
 'activation_requires':['explicit estimand/reference/policy','joint asset support and overlap','fold-local teacher and reference pools','time-consistent supported horizons','sensitivity and falsification review'],
 'observed_label_overwrite_allowed':False,'unsupported_rows_silently_fall_back_in_same_column':False}
u=copy.deepcopy(c['model_roles']['U_residual']);u.update(status='disabled_pending_identification',interpretation=c['asset_research']['A']['name'],automatically_abnormal_gain=False,alternative_B=c['asset_research']['B'])
c['model_roles']['U_residual']=u;c['fight_labels']['asset_extension']=copy.deepcopy(u)
c['accounting']['asset_interpretation']=c['asset_research']['A']['name']
c['accounting']['innovation_uses_distinct_M_not_V_episode']=True
c['outputs']['prefight_prediction']+=['cohort_profile','frame_age_ms','ordering_policy','role_support','terrain_hash','terrain_support','skill_support','calibration_hash']
c['outputs']['value_research']+=['estimand_id','reference_id','teacher_hash','not_observed_ground_truth']
c['execution_gates']=[
 {'id':'G0','scope':'knowledge','status':'PARTIAL','requires':['source hashes and 855-entry coverage','independent semantics and rank exceptions','patch applicability']},
 {'id':'G1','scope':'source audit','status':'PENDING','requires':['full-corpus timestamp/schema audit','fresh/allphase loss ledger','skill prefix/duplicates','actual phase distribution']},
 {'id':'G2','scope':'state engine','status':'PENDING','requires':['prefix replay','rank/learned state separation','same-ms uncertainty','10-player identity retained']},
 {'id':'G2T','scope':'terrain','status':'BLOCKED_ON_GEOMETRY','requires':['grid acquisition','coordinates/flags/overlays validated','no future variant lookup','runtime applicability']},
 {'id':'G3','scope':'labels','status':'PENDING','requires':['TRAIN endpoint and alignment locks','terminal/gold independent support','nonoverlap accounting']},
 {'id':'G4','scope':'features/models','status':'PENDING','requires':['same information manifest','role fallback and spatial ablations','skill factorial','frozen model cards']},
 {'id':'G5','scope':'splits','status':'PENDING','requires':['match split disjointness','fold-local transformations','no held-out outcomes in pretraining']},
 {'id':'G6','scope':'prediction','status':'PENDING','requires':['common-row evaluation','age/family/patch support','calibration and match bootstrap']},
 {'id':'G7','scope':'asset identification','status':'DISABLED','requires':c['asset_research']['activation_requires']}
]
c['open_contracts']=[
 {'id':'P01','field':'endpoint_tolerance_seconds','candidate':5,'owner_gate':'G3','lock_stage':'TRAIN_SUPPORT_AUDIT','blocks':'outcome production'},
 {'id':'P02','field':'fight_alignment_gap_seconds','candidate':65,'owner_gate':'G3','lock_stage':'TRAIN_SUPPORT_AUDIT','blocks':'outcome production'},
 {'id':'P03','field':'role_proxy_regions_and_thresholds','candidate':'180s, 2 samples, 2/3 share; exact lane regions not yet validated','owner_gate':'G4','lock_stage':'TRAIN_SUPPORT_AUDIT','blocks':'ROLE_PROXY arm only'},
 {'id':'P04','field':'terrain_resolution_coordinate_parser','candidate':None,'owner_gate':'G2T','lock_stage':'SOURCE_AND_TRAIN_VALIDATION','blocks':'terrain-enabled arms only'},
 {'id':'P05','field':'dynamic_rank_exception_rules','candidate':None,'owner_gate':'G2','lock_stage':'SOURCE_AND_TRAIN_VALIDATION','blocks':'supported dynamic rank gating'},
 {'id':'P06','field':'training_optimizer_losses_capacity_and_calibration','candidate':'model-card specification required; no undeclared defaults','owner_gate':'G4','lock_stage':'BEFORE_DEV_MODEL_SELECTION','blocks':'training'},
 {'id':'P07','field':'15.15_15.16_16.13_knowledge_support','candidate':None,'owner_gate':'G0','lock_stage':'BEFORE_PATCH_INFERENCE','blocks':'knowledge-enabled inference for unsupported patch'},
 {'id':'P08','field':'U_reference_and_identification','candidate':None,'owner_gate':'G7','lock_stage':'SEPARATE_RESEARCH_APPROVAL','blocks':'all asset label activation'}]
c['run_policy']={'design_creation_authorizes_full_training':False,'max_cpu_workers':4,'training_gate_required':True,'audit_can_run_before_training_gate':True,'unsupported_feature_arm_does_not_block_basic_arm':True,'fail_closed_for_unlocked_label_parameters':True,'each_run_saves':['contract_hash','knowledge_hash','terrain_hash_or_null','feature_manifest','split_hash','model_card_hash','gate_receipts','environment','seed','support_ledger','metrics_or_error']}
# Explicit experiment revision: old IDs are history, never a second executable matrix.
registry, recipes, metrics = build_experiments()
c['revision']='v5-experiments-r1'
c['architecture_controls']['legacy_model_catalog_non_executable']=c['architecture_controls'].pop('base_models')
c['architecture_controls']['base_models']=[{'id':x['id'],'architecture':x['config']['architecture']} for x in registry['cards']]
c['model_selection'].update(primary_structural_candidates=[f'Q{i:02}' for i in range(1,10)],tree_reference='Q01',same_summary_control=['Q01','Q02'],factorial_2x2=[['Q06','Q07'],['Q08','Q09']],secondary_relation_models=[],selection='TRAIN FIT/STOP; DEV mean raw primary over two seeds; CAL only calibration; see TRAINING_RECIPES_V5.json',training_entrypoint_implemented=False)
c['model_roles']['Q']['primary_metric']=metrics['Q_RAW']['primary']
c['evaluation']['Q_primary']=metrics['Q_RAW']['primary']
c['evaluation']['P_primary']=metrics['P_PRIMARY']['primary']
c['evaluation']['primary_claim_multiplicity']={'n':5,'adjustment':'Holm','contrast_ids':[x['id'] for x in registry['primary_claim_contrasts']]}
c['evaluation']['multiplicity']['scope']='separate secondary within-family analysis; does not cover five primary claims'
c['evaluation']['uncertainty'].update(confidence_level=0.95,conditional_on_trained_seed_set=True,details='EVALUATION_PROTOCOL_V5.json#/AGGREGATION')
c['open_contracts'][5]['candidate']='Recipes/operators specified; DEV candidate lock before each fit; FINAL recipe lock after DEV selection; both require exact shapes/environment/fixtures'
c['open_contracts'][5]['lock_stage']='DEV_CANDIDATE_LOCK_BEFORE_FIT_THEN_FINAL_RECIPE_LOCK_AFTER_DEV'
c['experiment_protocol']={'registry':'EXPERIMENT_REGISTRY_V5.json','recipes':'TRAINING_RECIPES_V5.json','operators':'MODEL_OPERATORS_V5.json','evaluation':'EVALUATION_PROTOCOL_V5.json','narrative':'EXPERIMENT_TRAINING_PROTOCOL_KO.md','counts':registry['counts'],'required_cards':[x['id'] for x in registry['cards']],'optional_enabled':False,'training_executed':False}
c['run_policy'].update(max_total_CPU_threads=4,nested_parallelism=False)
for rel in ['paper_ablation_v5/EXPERIMENT_MATRIX.json','paper_ablation_v5/ABLATION_TRAINING_PLAN_KO.md']:
    c['source_manifest'].append({'path':'../'+rel,'sha256':sha(PARENT/rel)})
for name in ['EXPERIMENT_REGISTRY_V5.json','TRAINING_RECIPES_V5.json','MODEL_OPERATORS_V5.json','EVALUATION_PROTOCOL_V5.json','EXPERIMENT_TRAINING_PROTOCOL_KO.md']:
    c['source_manifest'].append({'path':name,'sha256':sha(ROOT/name)})
mp_design,mp_registry,mp_recipes=build_multipatch(c,registry,recipes)
write('PIPELINE_CONTRACT_V5.json',c)
write('DECISION_REGISTER.json',{'version':'v5','revision':c['revision'],'fixed':['fresh and allphase definitions','tau and history','observed label and no double counting','skills ID/rank/semantics factorial','per-query aligned sorting','terrain layer separation and source gating','legacy patch split preserved; active MP split independently versioned','U disabled','30 inherited templates plus 10 patch-comparison cards = 40 active MP cards','explicit loss/optimizer/calibration specs; implementation lock still required','eight active MP primary contrasts; previous test exposure disclosed','S0-S7 execution order'],'pending':c['open_contracts'],'multipatch_gates':c['multipatch_gates'],'artifact_evidence':{'knowledge':read('knowledge_15_14_v2/semantic_validation_report.json'),'terrain':read('terrain_v1/VALIDATION_REPORT.json'),'inventory':'MULTIPATCH_DATA_AUDIT.json'},'production_training_ready':False})
# Keep the integrated narrative in sync with the same protocol/table.
doc=ROOT/'FULL_DESIGN_SPECIFICATION_V5_KO.md'
txt=doc.read_text(encoding='utf-8')
start=txt.index('## 11.');end=txt.index('## 12.')
protocol=(ROOT/'EXPERIMENT_TRAINING_PROTOCOL_KO.md').read_text(encoding='utf-8')
body=protocol[protocol.index('\n\n')+2:]
txt=txt[:start]+'## 11. Ablation·연산 템플릿 — experiments-r1, 주 분할은 20절 r2\n\n'+body+'\n\n'+txt[end:]
doc.write_text(txt,encoding='utf-8')
sync_narrative()
print('Wrote v5 integrated experiment contract and narrative; no match data accessed.')
