"""Contract consistency checks, not model/engine execution tests."""
from pathlib import Path
import json,hashlib
R=Path(__file__).parent
def main():
 p=json.loads((R/'PIPELINE_CONTRACT_V4.json').read_text(encoding='utf-8'));checks={}
 checks['source_snapshots_unchanged']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['source_sha256'].items())
 checks['raw_player_information_preserved']=len(p['feature_contract']['ordered_18'])==18 and len({r['source_name'] for r in p['feature_contract']['ordered_18']})==18 and 'lossless' in p['feature_contract']['tree_adapter']
 checks['same_split']=p['split']['train_patch']=='15.14' and p['split']['validation_patch']=='15.15' and p['split']['test_patch']=='15.16'
 checks['query_and_score_start_not_collapsed']=p['time_contract']['query_tau_not_label_s'] and 'tau' in p['time_contract']['input_cutoff_resolution']
 checks['Q_future_horizon_not_input']=p['model_roles']['Q']['future_window_length_not_input'] and 'E-tau' in p['time_contract']['forbidden_Q_inputs']
 checks['P_disjoint_bins_cover_horizons']=p['outcomes']['disjoint_bins_seconds']==[[0,60],[60,120],[120,180],[180,300]] and p['outcomes']['horizons_seconds']==[60,120,180,300]
 checks['terminal_not_survivor_filtered']=p['outcomes']['include_terminal_matches_in_gold_regression'] and not p['outcomes']['extrapolate_gold_after_game_end']
 checks['no_price_double_add']=not p['accounting']['direct_reward_addition_to_total_gold'] and not p['accounting']['sum_separate_objective_prices']
 checks['uncertain_asset_label_disabled']=not p['accounting']['asset_extension_enabled'] and not p['accounting']['asset_to_observed_silent_fallback']
 checks['unsupported_not_game_class']=p['model_roles']['Q']['unresolved_is_output_status_not_class'] and 'UNRESOLVED' not in p['model_roles']['Q']['classes']
 checks['fitting_match_local']=p['model_roles']['V_episode']['split_unit']=='match' and p['model_roles']['V_episode']['all_encoders_and_transforms_fold_local']
 checks['no_new_training_or_labels']=not p['training_executed'] and not p['new_match_labels_generated']
 # Arithmetic fixtures use hypothetical values, not empirical research numbers.
 fixtures=[]
 for gs,gc,ge in [(1000,1300,1200),(-500,-100,800),(400,400,400)]:
  assert (gc-gs)+(ge-gc)==ge-gs
  fixtures.append({'start_gold_diff':gs,'postfight_gold_diff':gc,'endpoint_gold_diff':ge,'observed_score':ge-gs,'telescopes':True})
 checks['observed_settlement_telescopes']=all(x['telescopes'] for x in fixtures)
 assert all(checks.values()),checks
 out={'status':'PASS_DESIGN_CONSISTENCY','checks':checks,'hypothetical_accounting_fixtures':fixtures,'not_executed':['state replay','full-population eligibility/endpoint audit','label production','model training','causal identification'],'specification_status':'PROPOSAL_WITH_EXPLICIT_EXECUTION_GATES'}
 (R/'pipeline_validation_report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
