"""Extract source metadata and build a design-only terrain extension; never infer runtime activation."""
import hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent/'knowledge_15_14_v2'/'PIPELINE_CONTRACT_V4.json'
def digest(b): return hashlib.sha256(b).hexdigest()
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def walk(x,path=''):
    yield path,x
    if isinstance(x,dict):
        for k,v in x.items(): yield from walk(v,path+'/'+k.replace('~','~0').replace('/','~1'))
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from walk(v,path+'/'+str(i))

audits=[]
for patch in ['15_14','16_13']:
    p=ROOT/'sources'/f'cdragon_{patch}_map11.bin.json'; data=json.loads(p.read_text(encoding='utf-8'))
    extract={k:v for k,v in data.items() if isinstance(v,dict) and v.get('__type') in ['NavGridConfig','NavGridTerrainConfig','MapNavGridOverlays']}
    save(f'nav_config_{patch}.json',extract)
    refs=[{'json_pointer':path,'asset_reference':v} for path,v in walk(data) if isinstance(v,str) and v.lower().endswith(('.ngrid','.ngrid_overlay','.rg_overlay'))]
    tags=[{'json_pointer':path,'name':v['name'],'flags':v.get('flags')} for path,v in walk(extract) if isinstance(v,dict) and 'name' in v and 'flags' in v]
    overlays=[v for v in extract.values() if v.get('__type')=='MapNavGridOverlays']
    audits.append({'patch':patch.replace('_','.'),'source_file':str(p.relative_to(ROOT)), 'sha256':digest(p.read_bytes()), 'top_level_entries':len(data),'nav_config_file':f'nav_config_{patch}.json','navigation_references':refs,'terrain_tags':tags,'overlay_entries':sum(len(v['overlays']) for v in overlays),'contains_atakhan_reference':any('atakhan' in x['asset_reference'].lower() for x in refs),'actual_navigation_grid_obtained':False,'runtime_activation_verified_from_asset_presence':False})
save('SOURCE_AUDIT.json',{'scope':'public client configuration only; no match scan','patches':audits,'geometry_readiness':'BLOCKED_ON_GRID_AND_COORDINATE_VALIDATION','png_masks':'mNavmeshFxMaskName references; not certified walkability grids','asset_presence_is_runtime_activation':False})

base=json.loads(BASE.read_text(encoding='utf-8'))
protected=['cohort','time_contract','fight_labels','outcomes','split','accounting']
contract={
 'version':'pipeline-v4.1-terrain-design', 'status':'DESIGN_ONLY_NOT_MODEL_IMPLEMENTATION',
 'inherits':{'path':'../knowledge_15_14_v2/PIPELINE_CONTRACT_V4.json','sha256':digest(BASE.read_bytes()),'protected_sections_sha256':{k:digest(json.dumps(base[k],sort_keys=True,ensure_ascii=False).encode()) for k in protected}},
 'extension_only':True, 'feature_set_id':'F_basic_v1_plus_terrain_v1',
 'terrain_package':{
  'key_fields':['map_id','queue_ruleset','game_patch','client_build','geometry_hash','rules_hash','coordinate_transform_version'],
  'query_fields':['query_ms','elemental_variant','baron_variant','atakhan_site','transition_phase','dynamic_structure_state'],
  'asset_fields':['source_url','retrieved_utc','source_sha256','source_pointer','source_support','independent_review','runtime_applicability'],
  'layer_types':['walkability','vision_occlusion','brush','semantic_regions','traversal_links','dynamic_map_geometry','structure_state','potential_vision_regions'],
  'missing_values':['UNKNOWN','MISSING','STRUCTURAL_NA'],
  'coordinate_transform_status':'NOT_VALIDATED', 'walkability_status':'NOT_OBTAINED',
  'collision_grid_from_minimap_rgb':False, 'collision_grid_from_fx_mask':False,
  'automatic_activation_from_file_presence':False, 'year_only_lookup':False,
  'unknown_patch_fallback':'explicit unsupported; never silently use latest',
  'nav_mode':'ordinary ground walking only; gated mobility links separate'
 },
 'prefix_policy':{
  'history_terrain_at':'each historical frame timestamp', 'current_overlay_at':'tau',
  'known_at_ms_lte_query_required':True, 'valid_from_and_valid_to_required':True,
  'future_atakhan_kill_position_for_earlier_site':False,
  'future_dragon_type_for_earlier_variant':False,
  'uncertain_variant':'mask affected layers; keep unaffected supported regions; no unweighted average map',
  'transition_duration':'source-specific and validated; unknown interval remains masked'
 },
 'spatial_policy':{
  'position_is':'last observed frame position with timestamp, age, source and mask',
  'terrain_lookup_is_current_position':False, 'interpolate_trajectories_primary':False,
  'same_time_pair_features':True, 'dead_players_removed':False,
  'current_alive_masks_spatial_activity':True,
  'path_distance_requires':['validated_grid','validated_coordinate_transform','supported_variant','supported_endpoints','explicit_traversal_mode'],
  'path_distance_is_exact_arrival_time':False,
  'unsupported_path':'value null plus reason mask; never zero',
  'reachable_set':'secondary model estimate only; cannot certify actual location or speed bound without mobility assumptions',
  'no_spatial_ablation_removes':['xy','local_terrain_lookup','terrain_relative_distance','path_distance','position_derived_edges','position_occupancy','region_assignment']
 },
 'encoders':{
  'recommended':'shared source-derived terrain descriptors -> individual/team/world readouts',
  'individual':['base18_history','static_champion_semantics','local_region','distance_to_wall_brush','walk_path_to_supported_assets','position_age','terrain_support'],
  'team':['set_of_5','same_time_pair_path_distances','shared_region_relations','objective_access_summaries','support_masks'],
  'rift':['region_graph','objective_structure_tokens','supported_dynamic_overlays','potential_vision_geometry','state_support'],
  'architecture_ablation':'tree / attention / temporal / typed graph on same canonical information',
  'secondary':'raster CNN local/global tokens; extra-input feature experiment unless all controls receive identical representation',
  'fixed_width_new_regions':'variable token count, fixed descriptor width, UNK types',
  'new_runtime_operation':'requires engine/schema version, not a prompt-only edit',
  'cosmetic_theme_primary':False
 },
 'unobserved_runtime':['current_ward_coverage','active_champion_created_wall','active_temporary_brush','individual_mobility_cooldown','precise_minion_wave_positions'],
 'llm_role':['source-grounded rule extraction','candidate region/link semantics','patch diff proposals'],
 'deterministic_role':['asset parsing','coordinate transforms','prefix state transitions','validated grid traversal','hashing','mask propagation'],
 'llm_may_invent_wall_coordinates':False,
 'comparison':{
  'feature_ladder':['F0 basic XY','F1 supported static terrain descriptors','F2 supported dynamic terrain descriptors'],
  'same_population_and_masks':True,'tree_receives_individual_canonical_flatten':True,
  'additional_raster_information_not_pure_architecture_gain':True,
  'train_only_scaling_and_learned_encoder':True,
  'external_geometry_used_for_training_representation':False,
  'patch_constant_geometry_causal_effect_identified':False,
  'cross_year_performance_drop_is_pure_terrain_effect':False,
  'value_identification_enabled_by_terrain':False
 },
 'implementation_gates':['actual grid acquisition','queue/build applicability','coordinate/flag decoding','dynamic overlay composition','prefix metamorphic tests','feature parity','frozen split evaluation'],
 'training_executed':False
}
save('PIPELINE_TERRAIN_EXTENSION_V4_1.json',contract)
checks=[]
for name in ['discovery_manifest.json']+[f'discovery_manifest_{i}.json' for i in range(2,6)]:
    for r in json.loads((ROOT/name).read_text(encoding='utf-8')):
        if r['status']=='DOWNLOADED':
            p=ROOT/'sources'/r['name']; assert digest(p.read_bytes())==r['sha256']; checks.append(r['name'])
assert len(audits)==2 and all(a['contains_atakhan_reference'] for a in audits)
assert all(a['overlay_entries']==8 for a in audits)
assert contract['inherits']['sha256']==digest(BASE.read_bytes())
save('VALIDATION_REPORT.json',{'source_hashes_verified':len(checks),'source_files':checks,'patch_json_parsed':2,'base_contract_hash_verified':True,'protected_sections':protected,'navigation_grid_validated':False,'model_test_or_training':False,'status':'METADATA_CHECKS_PASS_GEOMETRY_NOT_READY'})
print(json.dumps({'source_hashes_verified':len(checks),'patches':[{k:a[k] for k in ['patch','top_level_entries','overlay_entries','contains_atakhan_reference']} for a in audits],'geometry_ready':False}))
