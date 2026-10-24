"""P2 REPORT.md and before_after.md from the stored JSON artefacts (no recomputation)."""
from __future__ import annotations

import json
from pathlib import Path


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _f(x, d=4):
    return 'NA' if x is None else f'{x:.{d}f}'


def _ci(bs, metric, d=4):
    if not bs or bs.get('status') != 'ok':
        return 'NA' if not bs else f"NA ({bs.get('status')})"
    m = bs[metric]
    return f"{m['point']:+.{d}f} [{m['ci95'][0]:+.{d}f}, {m['ci95'][1]:+.{d}f}]"


def _metric_row(label, c):
    if not c:
        return f'| {label} | - | - | - | - | - | - | - | - |'
    return (f"| {label} | {c['rows']} | {c['matches']} | {c['status']} | {_f(c['auc'])} | {_f(c['brier'])} | "
            f"{_f(c['log_loss'])} | {_f(c['calibration_intercept'], 3)} | {_f(c['calibration_slope'], 3)} |")


HEAD = '| model | rows | matches | status | AUC | Brier | log loss | cal. intercept | cal. slope |\n|---|---|---|---|---|---|---|---|---|'


def write_report(out):
    out = Path(out)
    R, V, P, S = (_read(out / n) for n in ('results.json', 'validation.json', 'protocol.json', 'selection.json'))
    TM = _read(out / 'metrics' / 'test_metrics.json')
    TR = _read(out / 'trajectories' / 'trajectory_summary.json')
    EP = _read(out / 'endpoint_diagnostics' / 'endpoint_summary_DIAGNOSTIC.json')
    FA = _read(out / 'frame_update_audit' / 'frame_update_audit_summary.json')
    RA = _read(out / 'validation_replay_audits.json')
    BT = _read(out / 'models' / 'candidate_B_training_summary.json')
    AC = _read(out / 'states' / 'assembly_checks.json')
    L = ['# P2: temporal win probability (exploratory v3)', '']
    if R and R.get('smoke'):
        L += ['**SMOKE RUN on a tiny subset: metrics are not interpretable.**', '']
    L += [f"- Run status: `{R['status'] if R else 'incomplete'}`; validation: `{V['status'] if V else 'missing'}`"
          + (f"; failed: {', '.join(V['failed_checks'])}" if V and V['failed_checks'] else '')
          + (f"; incomplete: {', '.join(V['incomplete_checks'])}" if V and V['incomplete_checks'] else ''),
          f"- Stage status: {json.dumps(V['stage_status']) if V else '-'}",
          f"- Blockers: {'none' if not V or not V['blockers'] else '; '.join(b['stage'] + ': ' + b['error'] for b in V['blockers'])}",
          f"- Frozen P2 candidate chosen on SELECT: **{S['chosen_candidate'] if S else '-'}** "
          f"(A select log loss {_f(S['select_log_loss']['A'], 5) if S else '-'}, B primary {_f(S['select_log_loss']['B_primary'], 5) if S else '-'}).",
          '- Target: W = final Blue win. p_t approximates P(W=1 | observations available at t). A = f(X_t, t) (frozen P1 '
          'expanded adapter); B = f(X_{t-8min}, ..., X_t) (SimpleRNN). Both are associational, not causal.',
          '- Exploratory: the historical test partition (patches 15.14-15.16 corpus) was examined in earlier work. 15.16 and '
          '26.13 are not untouched tests. No all-patch claim; independent future-patch confirmation is pending.', '']
    L += ['## Reference-aligned implementation vs scientific validation', '',
          '- *Implemented (reference-aligned)*: Silva et al. (SBGames 2018)-style SimpleRNN comparator (9 positions, 8 units, '
          'dropout .25) adapted to StateV2 histories; probability-quality evaluation in the spirit of Kim et al. (CoG 2020) '
          'without their uncertainty-aware loss; Maymin (2021)-style snapshot probability kept as candidate A; the reused '
          '5-minute bucket query keys follow Jalovaara (2024, master thesis). Hodge et al. (IEEE ToG 2021) is cited only as '
          'live-prediction motivation.',
          '- *Validated here*: data/protocol integrity (splits, query keys, history membership, masks, future invariance, '
          'replay equality, guards, frozen hashes) and retrospective probability quality on the same exploratory corpus.',
          '- *Not validated*: causal effects, real-time API behaviour, generalisation to future patches, the engagement '
          'endpoint rule, any label or q model.', '']
    if P and AC:
        pc = P['protocol_checks']
        L += ['## Protocol reuse and histories', '', '| partition | matches | sampled rows | history states built | rows with masked positions |',
              '|---|---|---|---|---|']
        for p in ('fit', 'calibrate', 'select', 'test'):
            L.append(f"| {p} | {pc['split_sizes'][p]} | {pc['sampled_by_role'][p]['rows']} | {AC['built_states_by_partition'].get(p)} | "
                     f"{AC['rows_with_any_masked_position'].get(p)} |")
        L += [f"| test grid | {pc['split_sizes']['test']} | {AC['rows'].get('test_grid')} (grid rows) | (test table) | "
              f"{AC['rows_with_any_masked_position'].get('test_grid')} |", '',
              f"- Splits and query keys equal P1/original: {pc['p1_splits_equal_original'] and pc['p1_sampled_keys_equal_source_order'] and pc['p1_grid_keys_equal_source']}. "
              f"Query-position states bitwise equal to P1: {AC['query_states_bitwise_equal_p1']}.",
              f"- Engagement overlap with V partitions (all must be 0): {json.dumps(pc['engagement_overlap_with_V'])}.",
              f"- History membership violations: {json.dumps({k: v['violations'] for k, v in AC['membership'].items()})}; "
              f"unsupported planned positions: {AC['unsupported_planned_times']}; first-frame support values: {AC['support_start_ms_values']}.",
              '- Masking: positions before the first observed frame (0 ms) are zero-filled and masked; this affects queries '
              'before minute 8. Snapshot age is kept as audit metadata only (as P1).', '']
    if BT:
        fo = BT['preprocessing_fit_origin']
        L += ['## Candidate B (training on FIT only)', '',
              f"- Preprocessing fit origin: {fo['partition']} ({fo['unique_states']} unique states, {fo['matches']} matches); "
              f"numeric inputs {fo['n_numeric']}, champion one-hot columns {fo['n_champion_columns']} "
              f"(categories per slot {fo['champion_categories_per_slot']}). {fo['unseen_champion_behaviour']}.",
              f"- Unseen champion counts: {json.dumps(fo['unseen_counts'])}.",
              f"- Training: {BT['config']['epochs']} epochs, batch {BT['config']['batch_size']}, RMSprop lr {BT['config']['lr']} "
              f"(rho {BT['config']['rho']}, eps {BT['config']['eps']}), unweighted BCE, {BT['config']['fit_rows']} FIT query rows, "
              f"torch {BT['config']['torch_version']} CPU. Seconds per seed {BT['training_seconds']}; final-epoch train BCE (with dropout) "
              f"{ {k: round(v, 4) for k, v in BT['final_epoch_train_bce'].items()} }.",
              f"- numpy float64 inference vs torch float32 forward max |diff|: "
              f"{ {k: v['max_abs_diff_numpy_float64_vs_torch_float32'] for k, v in BT['torch_numpy_parity'].items()} }; "
              f"save/load identity {BT['save_load_identity']}; bundle `{BT['bundle_sha256']}`.", '']
    if S:
        L += ['## Selection (SELECT partition only; written before any TEST prediction)', '',
              '| candidate | calibration variant | SELECT log loss | chosen |', '|---|---|---|---|',
              f"| A (frozen P1) | {S['A']['calibration']} | {_f(S['A']['select_metrics']['log_loss'], 5)} | {'yes' if S['chosen_candidate'] == 'A' else ''} |"]
        if S.get('B'):
            for seed, v in S['B']['members'].items():
                for m, ll in v['select_log_loss'].items():
                    L.append(f"| B seed {seed} | {m} | {_f(ll, 5)} | {'per-seed choice' if m == v['chosen_calibration'] else ''} |")
            L.append(f"| B primary (mean of 3 calibrated seeds) | {S['B']['calibration']} | {_f(S['B']['select_metrics']['log_loss'], 5)} | "
                     f"{'yes' if S['chosen_candidate'] == 'B' else ''} |")
        L += ['', f"- Rule: {S['rule']}.", f"- B minus A SELECT log loss: {_f(S['select_log_loss']['B_minus_A'], 5)}.",
              '- Caveats: ' + '; '.join(S['caveats']) + '.', '']
    if TM:
        one, grid = TM['test_one_minute'], TM['test_grid_overall']
        keys = [k for k in one if k == 'A' or k == 'B' or k.startswith('B_seed')]
        L += ['## TEST results (computed after the freeze; A and B both reported)', '',
              f"### Exact P1 one query per test match", '', HEAD]
        L += [_metric_row(k, one[k]) for k in keys]
        L += ['', f"- Paired match bootstrap B primary minus A ({TM.get('primary_bootstrap_label', '-')}): AUC {_ci(one.get('paired_bootstrap_B_minus_A'), 'auc')}; "
              f"Brier {_ci(one.get('paired_bootstrap_B_minus_A'), 'brier')}; log loss {_ci(one.get('paired_bootstrap_B_minus_A'), 'log_loss')}.",
              f"- A predictions bitwise equal to P1 stored: one-minute {TM['checks'].get('A_one_minute_bitwise_equal_p1_stored')}, grid {TM['checks'].get('A_grid_bitwise_equal_p1_stored')}.", '',
              '### Full stored test grid', '', HEAD]
        L += [_metric_row(k, grid[k]) for k in keys]
        L += ['', f"- Paired match bootstrap B minus A: AUC {_ci(grid.get('paired_bootstrap_B_minus_A'), 'auc')}; Brier "
              f"{_ci(grid.get('paired_bootstrap_B_minus_A'), 'brier')}; log loss {_ci(grid.get('paired_bootstrap_B_minus_A'), 'log_loss')}.",
              f"- Agreement B vs A: {json.dumps(TM.get('agreement_B_vs_A'))}", '',
              '### Time bands (full grid; exploratory CIs)', '',
              '| band | rows | matches | AUC A/B | Brier A/B | log loss A/B | slope A/B | log loss diff B-A [95% CI] |', '|---|---|---|---|---|---|---|---|']
        for b, c in TM['test_grid_time_bands'].items():
            bB = c.get('B') or {}
            L.append(f"| {b} | {c['rows']} | {c['matches']} | {_f(c['A']['auc'])}/{_f(bB.get('auc'))} | {_f(c['A']['brier'])}/{_f(bB.get('brier'))} | "
                     f"{_f(c['A']['log_loss'])}/{_f(bB.get('log_loss'))} | {_f(c['A']['calibration_slope'], 3)}/{_f(bB.get('calibration_slope'), 3)} | "
                     f"{_ci(c.get('paired_bootstrap_B_minus_A'), 'log_loss')} |")
        L += ['', '### Every one-minute band (full grid; one row per match per minute; exploratory, no multiple-testing correction)', '',
              'CSV: `metrics/test_grid_one_minute_bands.csv`; plot: `metrics/test_grid_one_minute_bands.png`.', '',
              '| minute | matches | status | log loss A | log loss B | diff B-A [95% CI] | Brier A | Brier B | AUC A | AUC B |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for mnt, c in TM['test_grid_one_minute_bands'].items():
            bB = c.get('B') or {}
            L.append(f"| {mnt} | {c['matches']} | {c['A']['status']} | {_f(c['A']['log_loss'])} | {_f(bB.get('log_loss'))} | "
                     f"{_ci(c.get('paired_bootstrap_B_minus_A'), 'log_loss')} | {_f(c['A']['brier'])} | {_f(bB.get('brier'))} | "
                     f"{_f(c['A']['auc'])} | {_f(bB.get('auc'))} |")
        L += ['', '### Objective strata at the query (full grid; P1 definitions; descriptive, confounded with time)', '',
              '| stratum | rows | matches | status | log loss A/B | Brier A/B | slope A/B | log loss diff B-A [95% CI] |', '|---|---|---|---|---|---|---|---|']
        for n, c in TM['test_grid_objective_strata'].items():
            bB = c.get('B') or {}
            L.append(f"| {n} | {c['rows']} | {c['matches']} | {c['A']['status']} | {_f(c['A']['log_loss'])}/{_f(bB.get('log_loss'))} | "
                     f"{_f(c['A']['brier'])}/{_f(bB.get('brier'))} | {_f(c['A']['calibration_slope'], 3)}/{_f(bB.get('calibration_slope'), 3)} | "
                     f"{_ci(c.get('paired_bootstrap_B_minus_A'), 'log_loss')} |")
        L += ['', f"- NA policy: {TM['na_policy']['rule']}. Strata are not used to tune anything.",
              f"- History masking on TEST: {json.dumps(TM['history_masking'])}.", '']
    if TR:
        L += ['## Trajectories (12 hash-selected test matches)', '',
              '- Files: `trajectories/trajectories_12_matches.html` (interactive), `trajectories/trajectories_12_matches.png`, '
              '`trajectories/trajectory_points.csv`, `trajectory_events.csv`, `trajectory_frames.csv`.',
              f"- Matches: {', '.join(TR['matches'])} (rule: {TR['selection']}).",
              f"- Queries per match: {json.dumps(TR['queries'])}; raw frame interval quantiles (0/5/50/95/100%) ms: {TR['frame_interval_ms_quantiles']}.",
              f"- Resolution: {TR['resolution_statement']}",
              f"- Incremental ObservationLog replay vs batch (exact): {json.dumps({m: {k: v for k, v in r.items() if k.endswith('mismatch')} for m, r in TR['incremental_replay_equality'].items()})}.", '']
    if EP:
        c = EP['checks']
        L += ['## DIAGNOSTIC engagement endpoint transport (not labels, not q, not V training)', '',
              f"- Rows {c['rows']} (expected {c['rows_expected']}); pre/post states bitwise equal to state_value_v2_fix: "
              f"{c['pre_states_bitwise_equal_v2_fix']}/{c['post_states_bitwise_equal_v2_fix']}; snapshots equal rows.csv: "
              f"{c['pre_snapshot_equal_rows_csv']}/{c['post_snapshot_equal_rows_csv']}; post query before terminal: {c['post_query_before_terminal']}.",
              f"- Definitions: {json.dumps(EP['definitions'])}", '', '| model / endpoint | rows | matches | status | AUC | Brier | log loss | cal. intercept | cal. slope |',
              '|---|---|---|---|---|---|---|---|---|']
        for cand, cells in EP['endpoint_probability_quality_vs_final_W'].items():
            for ep, cell in cells.items():
                L.append(_metric_row(f'{cand} {ep}', cell))
        L += ['', f"- Pooled endpoint log loss/Brier/AUC B minus A (exploratory): log loss {_ci(EP.get('pooled_endpoint_bootstrap_B_minus_A_exploratory'), 'log_loss')}; "
              f"Brier {_ci(EP.get('pooled_endpoint_bootstrap_B_minus_A_exploratory'), 'brier')}.",
              f"- Delta overall: {json.dumps(EP['delta_overall'])}", f"- Delta quantiles (0,1,5,25,50,75,95,99,100%): {json.dumps(EP['delta_quantiles'])}",
              f"- Source freshness: {json.dumps(EP['source_freshness'])}", '',
              '| delta stratum | rows | matches | mean delta A | mean delta B | sign disagreement (match-weighted) | corr(dA,dB) |', '|---|---|---|---|---|---|---|']
        for group in ('disagreement_by_delta_magnitude', 'delta_by_engagement_start_band', 'delta_by_pre_snapshot_age_s', 'delta_by_post_snapshot_age_s'):
            for k, cell in EP[group].items():
                L.append(f"| {group.replace('delta_by_', '').replace('disagreement_by_', '')}: {k} | {cell['rows']} | {cell.get('matches', 0)} | "
                         f"{_f(cell.get('mean_delta_A'))} | {_f(cell.get('mean_delta_B'))} | {_f(cell.get('sign_disagreement_rate_match_weighted'), 3)} | "
                         f"{_f(cell.get('pearson_delta_A_B'), 3)} |")
        L += ['', '| objective event inside (pre, post] | rows present | mean delta A | mean delta B | sign disagreement |', '|---|---|---|---|---|']
        for k, cell in EP['delta_by_objective_event_within_interval'].items():
            pr = cell['present']
            L.append(f"| {k} | {pr['rows']} | {_f(pr.get('mean_delta_A'))} | {_f(pr.get('mean_delta_B'))} | {_f(pr.get('sign_disagreement_rate_match_weighted'), 3)} |")
        L += ['', f"- {EP['attribution_note']} Final W evaluates p quality, not the unobservable true delta.",
              '- Files: `endpoint_diagnostics/endpoint_rows_DIAGNOSTIC.csv|.npz`, `endpoint_summary_DIAGNOSTIC.json`, `endpoint_delta_A_vs_B_DIAGNOSTIC.png`.', '']
    if FA:
        L += ['## Frame and event update audit (deterministic bounded sample)', '',
              f"- {len(FA['matches'])} test matches, {FA['pairs']} boundary pairs; {FA['boundaries']}; precision {FA['query_precision']}.",
              f"- Duplicate/concurrent timestamps: {json.dumps(FA['duplicate_timestamps'])}.", '',
              '| update class | pairs | abs dp A q50/q90/q99/max | abs dp B q50/q90/q99/max | no discrete input change | >1 input group changed | B earlier-position change |',
              '|---|---|---|---|---|---|---|']
        for cls, c in FA['by_update_class'].items():
            fmt = lambda q: '/'.join(f'{v:.4f}' for v in q) if q else 'NA'
            L.append(f"| {cls} | {c['pairs']} | {fmt(c['abs_dp_A_q50_q90_q99_max'])} | {fmt(c.get('abs_dp_B_q50_q90_q99_max'))} | "
                     f"{c['pairs_with_no_discrete_input_change_at_query']} | {c['pairs_with_more_than_one_input_group_changed']} | "
                     f"{c.get('pairs_with_B_earlier_position_change', 'NA')} |")
        L += ['', f"- {FA['interpretation']}", '- File: `frame_update_audit/update_pairs.csv`.', '']
    L += ['## Inference API', '',
          '- Module `worktrees/engagement-state-value/train/temporal_history_winprob_v3.py`: `ObservationLog` (chronological frames/events '
          'per match, roster at construction), `BatchSource` (retrospective replay), `CausalHistory.from_observed_states`, the single '
          '`assemble_history`, `TemporalWinProbabilityService.predict/predict_many/predict_source` returning `Prediction` (p, query, '
          'snapshot, age, history times/mask/snapshots, candidate, version, model hash, calibration), and `endpoint_delta` (same frozen '
          'model/calibration or ModelVersionError; same match; pre < post).',
          '- Rejected: model/version/calibration mismatch, out-of-order appends, cross-match data, a query while the log holds later '
          'observations, queries at/after an observed GAME_END, wrong state/node schema, tampered bundles. Probabilities are not '
          'smoothed or forced monotone. Plot/query cadence (10 s here) is separate from the one-minute training sampling cadence.',
          f"- Runtime guard checks on a real match: {json.dumps(V['api_guard_checks']) if V else '-'}.",
          '- Equality between incremental replay and batch replay is retrospective; it is not a real-time API validation.', '']
    if RA:
        L += ['## Real-cache replay audits', '', f"- Future perturbation: {json.dumps(RA['future_perturbation'])}",
              f"- Incremental replay: {json.dumps(RA['incremental_replay'])}", f"- Pipeline vs API: {json.dumps(RA['pipeline_vs_api'])}", '']
    if V:
        L += ['## Validation', '', '| check | result |', '|---|---|'] + [f'| {k} | {v} |' for k, v in V['checks'].items()]
        L += ['', f"- Tests: `{V['tests']['command']}` -> {V['tests']['summary']}", f"- Files changed during run (must be none): {V['changed_files_during_run'] or 'none'}", '']
    L += ['## Limitations and unresolved', '',
          '- Same exploratory corpus as P1; TEST was examined historically, so any A-vs-B TEST difference is exploratory.',
          '- Bootstrap intervals hold models, calibrators and selections fixed; they omit training, seed and selection variability. '
          'Subgroup (band/stratum) intervals are exploratory and uncorrected for multiplicity.',
          '- Both candidates were trained on minute-aligned queries whose snapshot age is ~60 s; off-grid queries (trajectories, '
          'engagement endpoints) have other ages. Snapshot age is audited, not modelled.',
          '- Masked history positions exist only before minute 8; B sees fewer positions early in a match.',
          '- The endpoint transport uses the stored engagement endpoints; it does not validate them, and final W cannot score the '
          'true delta. Event co-occurrence is not attribution.',
          '- Replay equality is retrospective over stored data; live API availability, latency and ordering are not validated.',
          '- No causal claims; no reproduction of Kim et al. uncertainty loss or Silva et al. results; no all-patch generalisation.', '',
          '## References (use boundaries)', '']
    if P:
        L += [f'- {k}: {v}' for k, v in P['references'].items()]
    L += ['', '## Artifacts', '', '- `protocol.json`, `status.json`, `run.log`, `selection.json`, `models/frozen_models.json`, `results.json`, '
          '`validation.json`, `validation_replay_audits.json`, `hashes_before.json`, `hashes_after.json`, `source_snapshot/`, `pytest_output.log`',
          '- `states/` (chunks, state tables, sequences, match table, assembly checks); `models/candidate_B/` (bundle), `models/training/`',
          '- `predictions/test_one_minute_A_B.csv|.npz`, `predictions/test_grid_A_B.npz`; `metrics/`; `trajectories/`; '
          '`endpoint_diagnostics/`; `frame_update_audit/`; `before_after.md`']
    (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    write_before_after(out, R, V, S, TM, EP, TR, FA)


def write_before_after(out, R, V, S, TM, EP, TR, FA):
    ck = (V or {}).get('checks', {})

    def st(key):
        v = ck.get(key)
        return 'validated' if v is True else ('FAILED' if v is False else 'unresolved/blocked')
    one = (TM or {}).get('test_one_minute', {})
    grid = (TM or {}).get('test_grid_overall', {})

    def ll(c, k):
        return _f((c.get(k) or {}).get('log_loss')) if c else 'NA'
    L = ['# P2 before/after', '',
         'Before = P1 (`outputs/independent_v2_participant_order`, snapshot V2). After = P2 (`outputs/temporal_winprob_v3`). '
         'Status: *implemented* = code exists and ran; *validated* = a stored check passed; *unresolved* = not established.', '',
         '| aspect | before (P1) | after (P2) | status | sources |', '|---|---|---|---|---|',
         f"| probability model | snapshot f(X_t,t), expanded logistic (`models/expanded_model_v2.joblib`) | A = same frozen adapter; "
         f"B = SimpleRNN over 9 one-minute positions | implemented; A reproduces P1: {st('candidate_A_reproduces_p1_test')} | "
         "[P1 REPORT](../independent_v2_participant_order/REPORT.md), [module](../../worktrees/engagement-state-value/train/temporal_history_winprob_v3.py) |",
         f"| history assembly | none (single state) | causal `assemble_history`, masks before first frame | {st('history_positions_same_match_and_partition')} / "
         f"{st('history_masks_consistent_and_query_position_valid')} | [assembly_checks](states/assembly_checks.json) |",
         f"| future leakage | P1 future invariance on states | history-level future perturbation invariance (real cache + tests) | "
         f"{st('future_perturbation_invariance_real_cache')} | [replay audits](validation_replay_audits.json), [tests](../../tests/test_temporal_winprob_v3_history.py) |",
         f"| splits / queries | P1 exact splits and keys | identical reuse; query states bitwise equal P1 | {st('query_states_bitwise_equal_p1')} | [protocol](protocol.json) |",
         f"| selection | P1 chose calibration on SELECT | A vs B on SELECT, frozen before TEST (chosen: {(S or {}).get('chosen_candidate', '-')}) | "
         f"{st('selection_written_before_test_predictions')} | [selection](selection.json) |",
         f"| TEST one-minute log loss | A {ll(one, 'A')} | B primary {ll(one, 'B')} | exploratory (TEST previously examined) | [test_metrics](metrics/test_metrics.json) |",
         f"| TEST full-grid log loss | A {ll(grid, 'A')} | B primary {ll(grid, 'B')} | exploratory | [one-minute bands](metrics/test_grid_one_minute_bands.csv) |",
         f"| on-demand inference | adapter on StateV2 matrices | service + ObservationLog + endpoint_delta with version/hash guards | "
         f"{st('api_guards_all_raise')}; incremental=batch: {st('incremental_equals_batch_current_query')} (retrospective replay only) | [module](../../worktrees/engagement-state-value/train/temporal_history_winprob_v3.py) |",
         f"| save/load | joblib adapter + hash | hashed numpy bundle for B | {st('candidate_B_save_load_identity')} | [bundle](models/candidate_B/manifest.json) |",
         f"| trajectories | not produced | 12 hash-selected test matches, HTML/PNG/CSV | implemented | [trajectories](trajectories/trajectories_12_matches.html) |",
         f"| engagement endpoints | V2 states only (`state_value_v2_fix`) | DIAGNOSTIC p_pre/p_post/delta for A and B | {st('endpoint_rows_and_states_equal_v2_fix')} | "
         "[endpoint summary](endpoint_diagnostics/endpoint_summary_DIAGNOSTIC.json) |",
         f"| frame/event update behaviour | not audited | bounded ms-precision audit, co-occurrence only | implemented | [frame audit](frame_update_audit/frame_update_audit_summary.json) |",
         f"| legacy files | frozen | unchanged during run and equal to P1 record | {st('legacy_and_p1_files_unchanged_during_run')} / {st('frozen_legacy_hashes_equal_p1_record')} | "
         "[hashes_before](hashes_before.json), [hashes_after](hashes_after.json) |",
         '| causal interpretation | none | none | unresolved (not claimed) | - |',
         '| real-time public API | not validated | not validated | unresolved | - |',
         '| future-patch confirmation | pending | pending | unresolved | - |',
         '| replacement labels / q | not produced | not produced (diagnostics only) | unresolved (out of scope) | - |', '']
    if V and V['blockers']:
        L += ['Blockers: ' + '; '.join(f"{b['stage']}: {b['error']}" for b in V['blockers']), '']
    (out / 'before_after.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
