import os
"""Build/check design artifacts only. Does not read a match or train a model."""
from pathlib import Path
import hashlib
import json
import runpy

ROOT = Path(__file__).resolve().parent
spec = json.loads((ROOT / 'ALL_OBJECTIVE_PILOT_V2.json').read_text(encoding='utf-8-sig'))
spec.update(
    version='all-objective-design-v3',
    status='integrated_design_not_experiment_ready',
    canonical_spec='ALL_OBJECTIVE_DESIGN_V3_KO.md',
    supersedes_scope='all-objective-pilot-v2',
    training_executed=False,
    new_labels_generated=False,
    causal_values_estimated=False,
)
spec['panel'].update(
    tensor_axes=['batch', 'time', 'team', 'player', 'continuous_channel'],
    teams=2, players_per_team=5,
    history_frames=3, history_frame_ablations=[1, 3, 6],
    recent_event_seconds=120,
    cumulative_asset_history='prefix_to_each_frame_not_only_last_frame',
    participant_id_usage='within_match_alignment_only',
    death_is_padding=False,
)
spec['outcomes'].pop('max_nonterminal_alignment_overrun_seconds', None)
spec['outcomes'].update(
    secondary_nominal_horizons_seconds=[60, 120, 300],
    horizons_seconds=[60, 120, 180, 300],
    disjoint_bins_seconds=[[0, 60], [60, 120], [120, 180], [180, 300]],
    endpoint='nearest_actual_frame_to_nominal_endpoint_or_supported_earlier_game_end',
    tie_break='earlier_frame',
    candidate_alignment_tolerance_seconds=5,
    alignment_contract_status='pending_train_time_support_audit',
    post_terminal_increment='zero_by_absorbing_outcome_definition_only_if_terminal_gold_supported',
    gold_monotone_constraint=False,
    terminal_head='interval_conditional_three_state_hazard_to_cumulative_probabilities',
)
model_specs = [
    ('B0', 'no_change_and_past_trend', 'summary', 'team_summary'),
    ('B1', 'linear', 'summary', 'team_pair_summary'),
    ('T0', 'lightgbm', 'lag_and_delta', 'invariant_team_summary'),
    ('T1', 'lightgbm', 'lag_and_delta', 'invariant_team_and_graph_summary'),
    ('B2', 'tabular_mlp', 'same_as_T1', 'same_as_T1'),
    ('S0', 'hierarchical_deepsets', 'per_player_time_mlp', 'pool_only'),
    ('A0', 'hierarchical_set_attention', 'per_player_time_mlp', 'dense_attention'),
    ('A1', 'relational_set_attention', 'per_player_time_mlp', 'same_edge_attributes_as_graph'),
    ('R0', 'hierarchical_deepsets', 'per_player_gru', 'pool_only'),
    ('R1', 'hierarchical_deepsets', 'per_player_causal_tcn', 'pool_only'),
    ('G0', 'typed_mean_message_gnn', 'per_player_time_mlp', 'current_graph'),
    ('G1', 'typed_gat', 'per_player_time_mlp', 'same_current_graph_as_G0'),
    ('G2', 'snapshot_gnn_then_time_mlp', 'per_player_time_mlp', 'graph_at_each_frame'),
    ('ST0', 'snapshot_gnn_then_gru', 'per_player_gru', 'same_snapshot_graph_layer_as_G2'),
    ('ST1', 'gru_then_current_gnn', 'per_player_gru', 'current_graph'),
]
spec['models'] = [dict(id=i, architecture=a, temporal=t, relational=r) for i, a, t, r in model_specs]
spec['model_families'] = [m['id'] for m in spec['models']]
spec['model_comparison'] = {
    'nn_development_cards_per_family': 12,
    'default_embedding_dim': 32,
    'hidden_candidates': [64, 128, 256],
    'primary_structural_candidates': ['S0', 'A0', 'A1', 'R0', 'R1', 'G0', 'G1', 'ST0', 'ST1'],
    'selection': 'DEV_180_second_seven_family_gold_macro_before_AUDIT_and_TEST',
    'tree_reference': 'T1',
    'factorial_2x2': [['S0', 'R0'], ['G2', 'ST0']],
    'same_summary_control': ['T1', 'B2'],
    'primary_input_frames': 3,
    'count_equivalent_attention_and_graph_operators_as_distinct': False,
    'training_entrypoint_implemented': False,
}
spec['primary_comparison'] = 'DEV_selected_structural_model_vs_T1_15_16_180_second_macro_gold_RMSE'
spec['graph'] = {
    'primary_nodes_per_frame': {'player': 10, 'team_readout': 2},
    'team_nodes_send_feedback_in_primary_comparison': False,
    'primary_player_edges': 'complete_directed_typed_plus_self',
    'edge_attributes': ['same_team', 'relative_xy', 'distance', 'observability'],
    'optional_asset_nodes': 'team_by_seven_families_distinct_from_map_entities',
    'temporal_edges': 'same_entity_past_to_present_only',
    'optional_knn': [2, 4],
    'knn_boundary_ties': 'include_all_equal_distance_ties',
    'future_kill_locations_allowed': False,
    'proximity_means_observed_combat': False,
}
spec['value_estimation'] = {
    'status': 'design_only_no_effects_estimated',
    'input': 'pre_exposure_raw_covariates_plus_fold_trained_embedding',
    'risk_set': 'patch_and_prefix_supported_acquisition_opportunity',
    'exposure_window_seconds': 60,
    'exposure': ['blue_acquisition', 'red_acquisition', 'none_first_60', 'explicit_bundle_or_count'],
    'later_acquisition_in_control_allowed': True,
    'nonoverlapping_exposure_windows': 'greedy_eligible_actual_frames_at_least_60_seconds_apart',
    'outcome_origin': 'same_pre_exposure_frame_for_all_groups',
    'estimator': 'cross_fitted_multivalued_propensity_and_outcome_DR_contrast',
    'cross_fit_folds': 5,
    'split_unit': 'match',
    'all_encoders_and_transforms_fold_local': True,
    'default_claim': 'baseline_adjusted_acquisition_episode_association',
    'causal_claim_requires_additional_identification': True,
    'exclude_direct_reward_only_when_verified': True,
    'episode_effect_equals_remaining_asset_price': False,
    'individual_pseudo_outcome_is_true_effect': False,
}
spec['fight_labels'] = {
    'fight_detection': 'existing_frozen_detector',
    's': 'last_actual_frame_strictly_before_fight_start',
    'c': 'first_actual_frame_at_or_after_fight_end',
    'candidate_alignment_gap_cap_seconds': 65,
    'E': 'c_plus_180_seconds_mapped_by_common_endpoint_rule',
    'primary': {
        'version': 'observed_terminal_gold_v1',
        'score': 'G(E)-G(s)',
        'priority': 'observed_terminal_winner_before_economic_sign',
        'economic_tie_threshold': 0,
        'classes': ['BLUE_GAIN', 'RED_GAIN', 'TIE', 'UNRESOLVED'],
        'reason': ['TERMINAL', 'ECONOMIC', 'MISSING_SUPPORT'],
        'add_model_value': False,
    },
    'nowcast': {
        'score': 'G(c)-G(s)+m_P(c,E)',
        'observed_ground_truth': False,
        'replace_future_prediction_with_observation_when_available': True,
    },
    'asset_extension': {
        'version': 'asset_adjusted_mtm_v1',
        'status': 'disabled_pending_identification',
        'score': 'G(c)-G(s)+U(S_c,c;E)-U(S_s,s;E)',
        'U': 'joint_non_cash_residual_incremental_value_relative_to_defined_reference',
        'U_Q_candidate': 'm_P(S,h=E-r)-weighted_m_P(observed_supported_reference_states,h=E-r)',
        'reference_state_game_times_overwritten': False,
        'U_Q_is_automatically_asset_causal_price': False,
        'same_absolute_endpoint_and_teacher': True,
        'pre_existing_assets_subtracted': True,
        'U_at_game_end_or_E': 0,
        'requires_supported_remaining_horizon': True,
        'sums_single_objective_prices': False,
        'unsupported_row_fallback_to_primary_in_same_label_column': False,
        'equals_observed_label_at_E': False,
        'value_change_is_fight_causal_effect': False,
    },
}
spec['pending_execution_contracts'] = [
    'full_population_time_alignment_support',
    'patch_specific_direct_reward_receipt_support',
    'objective_opportunity_and_residual_state_support',
    'exact_model_cards_and_operator_equivalence_review',
    'asset_reference_and_identification_gate_for_extension_only',
]
spec['parallel']['stages'] = ['common_contract', 'match_shards', 'validate_merge', 'predictive_model_queue_and_value_risksets', 'observed_label_audit', 'gated_asset_extension', 'separate_prefight_prediction']
spec['claims'].update(
    graph_proximity_is_observed_interaction=False,
    attention_weights_are_individual_contribution=False,
    cross_fitting_removes_unmeasured_confounding=False,
    changing_currency_from_win_probability_to_gold_solves_all_deltaV_problems=False,
)
out = ROOT / 'ALL_OBJECTIVE_DESIGN_V3.json'
out.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

sources = [
    ('Zaheer et al.', 2017, 'Deep Sets', 'https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html', 'permutation invariant sets', 'conference'),
    ('Lee et al.', 2019, 'Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks', 'https://proceedings.mlr.press/v97/lee19d.html', 'within-set attention', 'conference'),
    ('Veličković et al.', 2018, 'Graph Attention Networks', 'https://petar-v.com/GAT/', 'masked neighbor attention', 'author_page_for_conference_paper'),
    ('Rossi et al.', 2020, 'Temporal Graph Networks for Deep Learning on Dynamic Graphs', 'https://arxiv.org/abs/2006.10637', 'temporal event graphs; not proof of performance here', 'preprint'),
    ('Bai et al.', 2018, 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling', 'https://arxiv.org/abs/1803.01271', 'temporal architecture comparison', 'preprint'),
    ('Shalit et al.', 2017, 'Estimating individual treatment effect: generalization bounds and algorithms', 'https://proceedings.mlr.press/v70/shalit17a.html', 'representation learning under ignorability', 'conference'),
    ('Johansson et al.', 2022, 'Generalization Bounds and Representation Learning for Estimation of Potential Outcomes and Causal Effects', 'https://jmlr.org/papers/v23/19-511.html', 'follow-up representation and weighting assumptions', 'journal'),
    ('Kennedy', 2020, 'Towards optimal doubly robust estimation of heterogeneous causal effects', 'https://arxiv.org/abs/2004.14497', 'cross-fitted DR estimation; cited manuscript version', 'preprint_version'),
    ('Barreto et al.', 2017, 'Successor Features for Transfer in Reinforcement Learning', 'https://proceedings.neurips.cc/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html', 'SF interpretation boundary', 'conference'),
]
source_path = ROOT / 'V3_METHOD_SOURCES.json'
source_path.write_text(json.dumps([
    dict(authors=a, year=y, title=t, url=u, use=p, type=k, retrieved='2026-10-16', empirical_validation_for_this_dataset=False)
    for a,y,t,u,p,k in sources
], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

doc = (ROOT / spec['canonical_spec']).read_text(encoding='utf-8')
assert len(spec['families']) == len(set(f['id'] for f in spec['families'])) == 7
assert spec['split'] == json.loads((ROOT / 'ALL_OBJECTIVE_PILOT_V2.json').read_text(encoding='utf-8-sig'))['split']
assert spec['outcomes']['horizons_seconds'] == [b[1] for b in spec['outcomes']['disjoint_bins_seconds']]
assert all(a[1] == b[0] for a,b in zip(spec['outcomes']['disjoint_bins_seconds'], spec['outcomes']['disjoint_bins_seconds'][1:]))
assert len(spec['models']) == len(set(m['id'] for m in spec['models']))
assert all('| '+m['id']+' |' in doc for m in spec['models'])
assert all(s[3] in doc for s in sources)
assert next(f for f in spec['families'] if f['id']=='atakhan')['available_external_16_13'] is False
assert not spec['training_executed'] and not spec['new_labels_generated']
assert spec['parallel']['max_cpu_workers'] == 4
assert spec['fight_labels']['asset_extension']['status'] == 'disabled_pending_identification'

# Illustrative accounting identities only; no research data or empirical metrics.
def mtm(g_start, g_end, u_start, u_end):
    return g_end-g_start+u_end-u_start
assert mtm(0,700,500,100) == 300
assert mtm(0,300,500,200)+mtm(300,700,200,100) == mtm(0,700,500,100)
assert mtm(0,700,500,0) == 200  # Different estimand from observed 700.
assert 300+400 == 700  # Observed prefix plus non-overlapping future segment.

# The skill's deterministic prose check is run once on this draft.
kernel = runpy.run_path(os.environ["LOL_LITERATURE_KERNEL"])
style = kernel['style_pass'](doc)
result = {
    'status':'PASS', 'scope':'design consistency and illustrative algebra only',
    'checks':['json_schema_values','seven_objectives','unchanged_split','four_disjoint_horizons','model_ids_match_document','source_urls_match','Atakhan16_mask','cpu_budget','disabled_asset_extension','illustrative_accounting'],
    'style_pass':style,
    'research_data_read':False, 'training_executed':False, 'causal_validity_proven':False,
    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/spec['canonical_spec'],out,source_path]},
}
(ROOT/'V3_DESIGN_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'], 'models':len(spec['models']), 'families':7, 'horizons':4, 'style':style, 'scope':result['scope']},ensure_ascii=True))
