import os
"""Resolve feature declarations without reading matches or fitting anything."""
from pathlib import Path
import hashlib
import importlib.util
import json

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ["LOL_RESEARCH_ROOT"])
extract_path = ROOT/'studies/input_specification_20261002/schema_extract.py'
module_spec = importlib.util.spec_from_file_location('feature_contract_schema_extract', extract_path)
extract = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(extract)
config, _ = extract.parse_config_literals()
inventory = extract.build_inventory()
source_node = {f['actual_name']:f for f in inventory['fields'] if f['storage_block']=='node'}
policy_path = ROOT/'reports/input_population_specification_20261002/INPUT_POLICY.json'
policy = json.loads(policy_path.read_text(encoding='utf-8'))
projected_names = sum((policy['node_primary_frame_channels'][g] for g in ['E','R','D']), [])
econ = policy['node_primary_frame_channels']['E']
stats = ['hp_pct','mp_pct','cs_health','cs_healthMax','cs_attackDamage','cs_abilityPower','cs_armor','cs_magicResist','cs_attackSpeed','cs_movementSpeed']
fields = []
for n in econ + stats:
    f = source_node[n]
    fields.append({
        'order':len(fields), 'source_name':n,
        'semantic_name':'last_frame_power_fraction' if n=='mp_pct' else n,
        'group':'E' if n in econ else 'R',
        'raw_node_minute_index':config['NODE_FEATURE_NAMES'].index(n),
        'observable_node_frame_index':projected_names.index(n),
        'source_paths':[v['path'] for v in f['raw_sources']],
        'cached_transform':f['transform'],
        'source_pointers':f.get('source_pointers',[]),
        'value_policy':'retain_cache_value_no_inverse_clipping',
        'raw_presence_verified':False,
    })
for axis in ['x','y']:
    fields.append({
        'order':len(fields),'source_name':axis,'semantic_name':axis,'group':'S',
        'raw_array':'xy_raw_minute','observable_array':'xy',
        'coordinate_index':0 if axis=='x' else 1,
        'source_paths':[f'timeline.info.frames[].participantFrames[].position.{axis}'],
        'cached_transform':'raw_coordinate_array_identity',
        'value_policy':'retain_coordinate_then_optional_fold_local_affine_scaling',
        'raw_presence_verified':False,
    })
leaves=['infernal_dragon','mountain_dragon','ocean_dragon','cloud_dragon','hextech_dragon','chemtech_dragon','elder','baron','herald','grubs','atakhan']
team_fields=[{'name':'team_total_gold','kind':'numeric','source':'gold_team[frame,team] or gold_team_minute[frame,team]','status':'existing_array'}]
team_fields += [{'name':f'{k}_acquisition_count','kind':'numeric','source':'verified typed event prefix by acquirer team and subtype','status':'new_minimal_prefix_contract'} for k in leaves]
team_fields += [{'name':f'{k}_last_acquisition_age_seconds','kind':'numeric','source':'frame_time minus latest valid acquisition time for this team and subtype','status':'new_minimal_prefix_contract','has_event_mask_required':True} for k in leaves]
team_fields += [
    {'name':'soul_type','kind':'categorical','source':'validated team DRAGON_SOUL_GIVEN prefix','status':'new_minimal_prefix_contract','vocabulary':['NONE','INFERNAL','MOUNTAIN','OCEAN','CLOUD','HEXTECH','CHEMTECH','UNKNOWN']},
    {'name':'soul_acquisition_age_seconds','kind':'numeric','source':'validated team soul acquisition time','status':'new_minimal_prefix_contract','has_event_mask_required':True},
]
team_fields += [{'name':f'own_{k}_loss_event_count','kind':'numeric','source':'verified structure event prefix indexed by owner team','status':'new_minimal_prefix_contract'} for k in ['tower','inhibitor','plate']]
sources = [
    extract_path,policy_path,
    ROOT/'studies/observable_state_20261002/schema.py',
    ROOT/'studies/observable_state_20261002/kernel.py',
    ROOT/'reports/observable_state_20261002/ACCEPTANCE.json',
    ROOT/'worktrees/engagement-state-value/core/config.py',
    ROOT/'worktrees/engagement-state-value/gameplay/pipeline_cache.py',
]
contract={
    'version':'basic-feature-contract-v1', 'feature_set_id':'F_basic_v1',
    'date_kst':'2026-10-16','status':'source_mapped_design_not_full_population_producer',
    'document':'BASIC_FEATURE_CONTRACT_V1_KO.md',
    'source_schema':'observable_state_v1_20261002',
    'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'time':{'history_frames':3,'query_elapsed_minutes':True,'frame_age_seconds':True,'future_time_inputs':False,'rolling_120_seconds':'optional_not_primary'},
    'player_continuous':{'shape_without_batch':[3,2,5,18],'ordered_fields':fields},
    'player_static':{'shape_without_batch':[2,5,3],'source':'static_ids or verified meta.static_meta','ordered_fields':['champion_id','summoner_spell_1_id','summoner_spell_2_id'],'source_indices':[0,1,2],'categorical_not_ordinal':True},
    'player_history':{'shape_without_batch':[3,2,5,3],'fields':['observed_kills','observed_deaths','observed_assists'],'source':'query_numeric(frame_time).kda','source_shape':[3,10],'adapter':'transpose(1,0), then verified team/player mapping','final_meta_kda_allowed':False},
    'team_history':{'shape_without_batch':[3,2,28],'numeric_fields_per_team':27,'categorical_fields_per_team':1,'ordered_fields':team_fields,'masks_are_separate':True},
    'masks':['frame_present','finite_cache_value','static_id_supported','event_prefix_supported','age_has_event','patch_available','structural_na'],
    'unknown_raw_presence_inferred_from_zero':False,
    'routing_metadata_only':['patch_id','canonical_match_key','split','fold'],
    'structure_only':['participant_id_for_alignment','team_id_for_membership_and_physical_side'],
    'legacy_objective_summary':{'categories':9,'counts_shape':[9,3],'counts_axes':['type','total_blue_red'],'last_age_shape':[9],'last_age_is_team_specific':False,'use_as_new_objective_block_without_reconstruction':False},
    'tree_adapter':{
        'primary_model_id':'T0','aggregation':'none_player_values_preserved',
        'canonical_sort_keys':['champion_id','summoner_spell_1_id','summoner_spell_2_id','full_visible_player_history_and_masks_lexicographic_tiebreak'],
        'sort_before_learned_transforms':True,
        'move_complete_player_history_together':True,
        'participant_id_or_final_role_as_sort_key':False,
        'same_flatten_mlp_control':'B2',
    },
    'primary_graph':{'connectivity':'complete_players_and_team_membership','edge_type':'self_ally_opponent','coordinates_in_nodes':True,'explicit_distance_edge_features':False,'knn_mask':False,'asset_or_map_nodes':'optional_secondary'},
    'secondary_relations':{'model_ids':['T1','A1'],'fields':['dx','dy','distance'],'same_derived_array_available_to_comparators':True,'composite_domain_scores':False},
    'scaling':{'cached_values':'retain_existing_scale','additional_numeric':'fold_local_affine_only','new_clipping':False,'counts_and_ages':'numeric_plus_support_masks','categorical':'fold_local_vocabulary','nan_adapter':'zero_fill_with_explicit_mask'},
    'extensions':{
        'F_fullschema':{'continuous_width':48,'node_projected_names':projected_names,'xy_fields':['x','y'],'changes_static_or_history':False},
        'C14':{'source':'static_ids','width':14,'separate_from_fullschema_test':True},
    },
    'excluded':['gps_norm','alive','champion_name_id','legacy_buff_status','final_roles','final_items','final_kda','final_damage','final_win','full_match_normalized_time','future_fight_or_kill_features','estimated_objective_prices','composite_combat_or_siege_scores','team_aggregate_replacement_of_players'],
    'training_executed':False,'match_data_read':False,'new_feature_producer_implemented':False,
}
assert len(fields)==18 and len(set(f['source_name'] for f in fields))==18
assert len(projected_names)==46 and len(set(projected_names))==46
assert len(econ)==6 and len(stats)==10
assert len(team_fields)==28 and sum(f['kind']=='numeric' for f in team_fields)==27
assert len(leaves)==11
assert [f['observable_node_frame_index'] for f in fields[:16]]==[0,1,2,3,4,5,6,7,19,20,13,9,10,25,14,26]
doc=(OUT/contract['document']).read_text(encoding='utf-8')
assert all(f['source_name'] in doc for f in fields)
assert 'objective_age_seconds' in doc and '[3,10]' in doc
fp=OUT/'BASIC_FEATURE_CONTRACT_V1.json'
fp.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

# Synchronize only our design artifact, not the research source or any frozen protocol.
design_path=OUT/'ALL_OBJECTIVE_DESIGN_V3.json'
design=json.loads(design_path.read_text(encoding='utf-8'))
old_split=design['split'].copy()
old_horizons=design['outcomes']['horizons_seconds'].copy()
design['feature_contract']='BASIC_FEATURE_CONTRACT_V1.json'
design['feature_contract_revision']='v3.1_basic_features'
design['primary_feature_set']='F_basic_v1'
design['panel']['continuous_player_channels']=18
design['panel']['static_player_channels']=3
design['panel']['event_player_channels']=3
design['panel']['recent_event_seconds']=120
design['panel']['recent_event_input']='optional_rolling_window; primary_uses_frame_aligned_cumulative_prefixes'
design['panel']['learned_patch_id_input']=False
design['model_comparison']['tree_reference']='T0'
design['model_comparison']['same_summary_control']=['T0','B2']
design['model_comparison']['same_summary_control_meaning']='same_lossless_canonical_flatten_not_team_summary'
design['model_comparison']['primary_structural_candidates']=['S0','A0','R0','R1','G0','G1','ST0','ST1']
design['model_comparison']['secondary_relation_models']=['T1','A1']
design['primary_comparison']='DEV_selected_F_basic_structural_model_vs_T0_15_16_180_second_macro_gold_RMSE'
for m in design['models']:
    if m['id']=='T0':m.update(temporal='raw_ordered_player_lags',relational='canonical_player_flatten_with_shared_team_context')
    elif m['id']=='T1':m.update(temporal='same_as_T0',relational='T0_plus_shared_pairwise_array_secondary')
    elif m['id']=='B2':m.update(temporal='same_as_T0',relational='same_as_T0')
    elif m['id']=='A1':m['relational']='secondary_shared_pairwise_array'
design['graph']['edge_attributes']=['same_team','self_or_opponent_relation']
design['graph']['secondary_edge_attributes']=['relative_xy','distance','observability']
design['graph']['coordinates_in_nodes']=True
design['graph']['optional_asset_nodes']='secondary_only_team_by_seven_families_distinct_from_map_entities'
design['graph']['optional_knn_scope']='secondary_only'
assert design['split']==old_split and design['outcomes']['horizons_seconds']==old_horizons
design_path.write_text(json.dumps(design,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
main_doc=(OUT/design['canonical_spec']).read_text(encoding='utf-8')
assert '최선 대 T0' in main_doc
assert 'BASIC_FEATURE_CONTRACT_V1_KO.md' in main_doc
assert '| B2 |' in main_doc and '| T0 |' in main_doc
report={
    'status':'PASS','scope':'source declaration mapping and design consistency only',
    'checks':['18_source_fields','raw_vs_projected_indices','KDA_3_by_10_adapter','11_acquisition_leaf_types','27_numeric_plus_1_categorical_team_fields','48_channel_fullschema_extension','T0_primary_and_B2_same_flatten','unchanged_splits_and_horizons'],
    'match_data_read':False,'training_executed':False,'full_population_support_verified':False,
    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [fp,OUT/contract['document'],design_path,OUT/design['canonical_spec']]},
}
(OUT/'BASIC_FEATURE_CHECK_V1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','continuous_player_fields':18,'static_ids':3,'history_counts':3,'team_numeric':27,'team_categorical':1,'comparison':'T0 basic raw flatten','scope':report['scope']},ensure_ascii=True))
