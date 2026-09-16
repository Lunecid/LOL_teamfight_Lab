"""P3 report generator: REPORT.md, DEFINITION_AND_EVIDENCE.md, TIMELINE_EXAMPLES.md from stored artifacts only.

Reads the JSON artifacts written by scripts/run_engagement_labels_v3_sensitivity.py; recomputes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
HS = (60, 90, 120)


def rj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(x, d=2):
    return 'NA' if x is None else f'{100 * x:.{d}f}%'


def ci(c, key='match_weighted_ci95', d=2):
    if not c or c.get(key) is None:
        return 'NA'
    lo, hi = c[key]
    return f'{100 * lo:.{d}f}–{100 * hi:.{d}f}%'


def f4(x):
    return 'NA' if x is None else f'{x:.4f}'


def mmss(ms):
    if ms is None or ms < 0:
        return '—'
    s, rem = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    return f'{m:02d}:{s:02d}.{rem:03d}'


def _cell(c):
    return str(c).replace('|', '\\|')


def table(header, rows):
    out = ['| ' + ' | '.join(_cell(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(_cell(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def plateau_limit(o, d90):
    p = o / 'post_run_spot_checks.json'
    if not p.exists():
        return 'Model-B output plateaus were not checked (post_run_spot_checks.json missing).'
    pl = rj(p)['model_B_repeated_value_plateaus (>=3 distinct matches)']
    if not pl:
        return 'No repeated model-B output plateau was found across 3 or more matches.'
    n = sum(x['queries_within_1e-9'] for x in pl)
    return (f'Model B has rare output plateaus: {n} P3 queries fall on {len(pl)} near-constant p_B values shared across matches (Section 8b lists model A on the same queries).'
            f' This produces the exact-zero B deltas. It touches far fewer rows than the {d90["label_disagreement"]["count"]:,} B90 A-vs-B disagreements,'
            ' so it cannot account for most of them. It remains a comparator issue to investigate before any B-based use.')


def report(o):
    P, V, Rz = rj(o / 'protocol.json'), rj(o / 'validation.json'), rj(o / 'results.json')
    A, SP = rj(o / 'analysis_results.json'), rj(o / 'source_provenance.json')
    LA, RA = rj(o / 'label_audits.json'), rj(o / 'independent_raw_event_audit.json')
    RB, RP = rj(o / 'reference_audit_b90_endpoints.json'), rj(o / 'reference_audit_p2_b90_values.json')
    FS, BC = rj(o / 'feature_schema.json'), rj(o / 'boundary_checks.json')
    census = SP['lineage_checks']['census']
    fm = Rz['frozen_models']
    L = []
    w = L.append
    w('# P3 — Frozen-A engagement labels: window and model sensitivity')
    w('')
    w(f'**Status:** run `{Rz["status"]}`, validation **`{V["status"]}`**'
      + (f' (failed: {", ".join(V["failed_checks"])})' if V['failed_checks'] else ' (all checks passed)') + f'. Smoke: {Rz["smoke"]}.')
    w('')
    w('**Role: EXPLORATORY CANDIDATE LABELS.** These are not validated ground truth and they do not replace any production label.'
      ' Nothing was trained, recalibrated or selected, and no q model was fitted. The working primary stays **B90 with frozen model A**.'
      ' The 60 s and 120 s windows, and model B at every window, are sensitivity analyses only.'
      ' No horizon or model was picked from these results.')
    w('')
    w('Design: Codex (docs/CLAUDE_EXECUTE_P3_LABEL_SENSITIVITY.md). Implementation and execution: Claude Opus 5.'
      f' protocol.json was written before execution (sha256 `{P["protocol_sha256"]}`, {P["written_at"]}).'
      f' Elapsed: {Rz["elapsed_seconds"]} s.')
    w('')
    w('## 1. Frozen inputs')
    w('')
    w(table(['Item', 'Identity'], [
        ['Model A (primary valuation)', f'{fm["A"]["model_version"]}, calibration {fm["A"]["calibration"]}, sha256 `{fm["A"]["sha256"]}`'],
        ['Model B (diagnostic comparator)', f'{fm["B"]["model_version"]}, {fm["B"]["calibration"]}, bundle sha256 `{fm["B"]["bundle_sha256"]}`'],
        ['History adapter', f'{fm["history_version"]} via train/temporal_history_winprob_v3.assemble_history'],
        ['State version', FS['state_version']],
        ['Raw cache (read-only)', f'{SP["raw_cache"]}; cohort files combined sha256 `{SP["raw_cache_cohort_files_sha256"]}`'],
    ]))
    w('')
    w('## 2. Cohort and historic selection (disclosure)')
    w('')
    w(f'The cohort is the **exact (match, s) set and order** of `outputs/state_value_v2_fix/rows.csv`: {Rz["cohort"]["rows"]:,} engagements in'
      f' {Rz["cohort"]["matches"]:,} matches, giving {Rz["cohort"]["label_rows"]:,} (match, s, h) label rows. No engagement was re-detected, added or dropped.')
    w('')
    w('**This is not "all engagements."** Tracing the chain back through the stored artifacts shows how the rows were selected:')
    w('')
    for i, step in enumerate(SP['historic_selection_chain'], 1):
        w(f'{i}. {step}')
    w('')
    w(table(['Census item', 'Value'], [
        ['V1 engagement split matches (= 50k-corpus predict_test)', f'{census["v1_engagement_split_matches"]:,} (equals predict_test: {census["v1_engagement_split_equals_eval_predict_test"]})'],
        ['… with >=1 parent exposure row', f'{census["engagement_matches_with_exposure_rows"]:,}'],
        ['… without any exposure row (cache files present)', f'{census["engagement_matches_without_any_exposure_row"]:,} ({census["engagement_matches_without_exposure_cache_files_present"]:,})'],
        ['Exposure rows in engagement-split matches', f'{census["exposure_rows_in_engagement_matches"]:,}'],
        ['Excluded: next eligible start <= L (overlap)', f'{census["excluded_same_match_overlap_rows (next_start <= L)"]:,}'],
        ['Other historic h=120 availability exclusions', f'{census["other_availability_exclusions (endpoint beyond last frame / negative)"]:,}'],
        ['Cohort = exposure rows minus overlap rows', census['cohort_equals_exposure_rows_minus_overlap']],
        ['Cohort rows by patch', ', '.join(f'{k}: {v:,}' for k, v in census['patch_rows_cohort'].items())],
    ]))
    w('')
    w(f'{census["corpus_note"]} The {census["engagement_matches_without_any_exposure_row"]:,} matches without an exposure row had no detector engagement row in the parent run.'
      ' Their cause was not re-derived, because re-detection is out of scope.'
      ' The historic h=120 availability filter removed only the overlap rows. Selection is still conditional on the V1 split,'
      ' on the frozen v3.3 detector and on non-overlap. Rates here therefore describe this cohort, not the full corpus.')
    w('')
    w('## 3. Boundary reconstruction and reference identity (audited before scoring)')
    w('')
    w('For each h: `endpoint_h = min(L + 1000h, next_kill − 1, next_eligible_start − 1, game_end − 1)` ms, with closed inclusion (≤ query).'
      ' next_kill is the first raw CHAMPION_KILL strictly after L. The exposure sentinel next_start ≥ game_end means that no later engagement exists.'
      ' Objective acquisitions never end a window.')
    w('')
    rows = [['B90 endpoints equal to ALL state_value_v2_fix endpoints', f'{RB["rows_compared"]:,} compared, **{RB["mismatches"]} mismatches**'],
            ['B60/B90/B120 equal to the prior V1 B-rule run (b_boundary_60_90_120)', ', '.join(f'h{h}: {RB[f"b_boundary_v1_run_h{h}_endpoint_mismatches"]} mismatches' for h in HS)],
            ['Guard: rows where the A-style rule of validate_windows.py would differ', f'{RB["guard_A_style_next_engagement_only_rule_differs_rows"]:,} (so it was not used)'],
            ['First kill s+15 s present in raw / simultaneous raw kills at it', f'missing {BC["first_kill_missing"]}, simultaneous {BC["first_kill_simultaneous"]} (flagged, never excluded)'],
            ['Last kill L present in raw', f'missing {BC["last_kill_missing"]}'],
            ['Stored next_start = recomputed from the parent exposures', f'mismatches {BC["next_start_recompute_mismatch"]}; sentinel conflicts {BC["sentinel_conflicts"]}'],
            ['Raw GAME_END = exposure end / ambiguous GAME_END', f'mismatches {BC["game_end_exposure_mismatch"]}; ambiguous {BC["game_end_ambiguous"]}; end_observed=0 rows {BC["end_observed_zero"]}'],
            ['Invalid rows (endpoint < L, ≤ q_pre, outside frames)', ', '.join(f'h{k}: {v}' for k, v in BC['invalid_rows'].items())],
            ['Non-monotone endpoints (e60 ≤ e90 ≤ e120)', BC['non_monotone_rows']],
            ['Rows with some raw kill in [s, first kill) (anywhere on the map; informational)', f'{BC["kills_in_s_to_first_kill_rows"]:,}']]
    w(table(['Check', 'Result'], rows))
    w('')
    p2 = RP['p2_b90_identity']
    w(f'**P2 B90 identity.** Declared rule: states bitwise equal, probability tolerance ≤ {P["reference_tolerances_declared_before_scoring"]["p2_b90_probabilities"]["declared_abs_tolerance"]}, bitwise preferred.'
      f' Pre and post-90 StateV2 vectors bitwise equal to state_value_v2_fix: {RP["pre_states_bitwise_equal_v2_fix"]} / {RP["post90_states_bitwise_equal_v2_fix"]}.'
      f' Every P2 probability field bitwise equal: **{RP["p2_probabilities_all_bitwise_equal"]}**'
      f' (max |diff| over fields = {max(v["max_abs_diff"] for v in p2.values() if "max_abs_diff" in v):.3g}). Integer fields equal: {RP["p2_integer_fields_equal"]}.')
    w('')
    w('## 4. Label distributions per window (same cohort)')
    w('')
    rows = []
    for h in HS:
        c = A['per_window'][str(h)]
        a, b = c['A'], c['B']
        fu = c['effective_followup_after_last_kill_s']
        rows.append([f'B{h}', f'{a["positive"]["count"]:,}', pct(a['positive']['row_weighted']), pct(a['positive']['match_weighted']),
                     ci(a['positive_rate_bootstrap']), a['exact_zero']['count'],
                     ' / '.join(f'{a["near_zero_descriptions"][str(t)]["count"]:,}' for t in (0.005, 0.01, 0.02)),
                     f4(a['delta_mean']['row_weighted']), f4(a['abs_delta_mean']['row_weighted']),
                     pct(b['positive']['row_weighted']), f'{fu["mean"]["row_weighted"]:.2f}', f'{fu["quantiles"]["0.5"]:.2f}',
                     pct(fu['reached_horizon']['row_weighted'])])
    w(table(['Window', 'Y_A=1 n', 'Y_A=1 row-wt', 'Y_A=1 match-wt', 'match-wt 95% CI', 'δ_A exactly 0', '|δ_A|≤.005/.01/.02 (desc.)',
             'mean δ_A', 'mean |δ_A|', 'Y_B=1 row-wt', 'mean follow-up s', 'median s', 'reached horizon'], rows))
    w('')
    w('Near-zero bands only describe the data; they are not label thresholds. An exact δ = 0 is labeled Y = 0 (non-improvement).'
      ' "Match-wt" gives each match equal total weight. δ is an estimated change in the final-Blue-win probability under a frozen model.'
      ' It is not a causal effect and not a monetary value.')
    w('')
    rows = []
    for h in HS:
        c = A['per_window'][str(h)]
        tr = c['terminal_reasons_inclusive']
        rows.append([f'B{h}'] + [f'{tr[r]["count"]:,} ({pct(tr[r]["row_weighted"], 1)})' for r in ('horizon', 'next_kill', 'next_engagement_start', 'game_end')]
                    + [c['rows_with_tied_reasons'], '; '.join(f'{k}: {v:,}' for k, v in c['terminal_reason_tie_sets'].items())])
    w(table(['Window', 'horizon', 'next_kill', 'next_engagement_start', 'game_end', 'rows with ties', 'exact tie sets'], rows))
    w('')
    w('Reason counts are inclusive: a tied row counts under every reason it has. All tied reasons are stored in `end_reasons`.')
    w('')
    w('### Availability and source staleness')
    w('')
    rows = []
    for h in HS:
        st, av = A['per_window'][str(h)]['staleness'], A['per_window'][str(h)]['availability']
        rows.append([f'B{h}', f'{av["valid_rows"]:,}/{av["valid_rows"] + av["invalid_rows"]:,}', f'{st["post_snapshot_age_s_quantiles"]["0.5"]:.1f}',
                     pct(st['post_snapshot_not_after_L']['row_weighted']), pct(st['post_snapshot_not_after_L']['match_weighted']),
                     pct(st['same_snapshot_frame_pre_and_post']['row_weighted']), pct(st['post_snapshot_age_gt_30s']['row_weighted']),
                     pct(st['post_snapshot_age_gt_45s']['row_weighted'])])
    w(table(['Window', 'valid', 'median post frame age s', 'no frame after L (row)', 'no frame after L (match)', 'same frame pre&post', 'post age>30 s', 'post age>45 s'], rows))
    w('')
    pa = A['pre_snapshot_age_s']
    w(f'Pre-query frame age: median {pa["quantiles"]["0.5"]:.1f} s; age>30 s {pct(pa["age_gt_30s"]["row_weighted"])}, >45 s {pct(pa["age_gt_45s"]["row_weighted"])} (row-weighted).'
      ' Frames arrive about once a minute, so post-states often carry node fields from a frame before L, while events up to the endpoint are included.'
      ' snapshot_age_s is audit-only and not a predictor.')
    w('')
    w('## 5. Paired window sign flips (model A primary, B shown)')
    w('')
    rows = []
    for k, c in A['paired_window_flips'].items():
        for mdl in ('A', 'B'):
            f = c[mdl]
            rows.append([k.replace('_vs_', ' vs '), mdl, f'{f["flip"]["count"]:,}', pct(f['flip']['row_weighted'], 3), ci(f['flip_bootstrap'], 'row_weighted_ci95', 3),
                         pct(f['flip']['match_weighted'], 3), ci(f['flip_bootstrap'], 'match_weighted_ci95', 3), pct(c['same_endpoint']['row_weighted']),
                         pct(f['flip_among_different_endpoints']['row_weighted'], 2), f['flips_among_same_endpoints']])
    w(table(['Pair', 'Model', 'flips', 'row-wt', 'row-wt 95% CI', 'match-wt', 'match-wt 95% CI', 'same endpoint', 'flip among different endpoints', 'flips among same endpoints'], rows))
    w('')
    w(f'{A["bootstrap_note"]}')
    w('')
    w('## 6. Model A vs model B label disagreement (diagnostic; not an error rate)')
    w('')
    rows = []
    for h in HS:
        d = A['model_A_vs_B_disagreement'][str(h)]
        rows.append([f'B{h}', f'{d["label_disagreement"]["count"]:,}', pct(d['label_disagreement']['row_weighted']), ci(d['bootstrap'], 'row_weighted_ci95'),
                     pct(d['label_disagreement']['match_weighted']), ci(d['bootstrap']), pct(d['three_valued_sign_disagreement']['row_weighted']),
                     f4(d['pearson_delta_A_B'])])
    w(table(['Window', 'Y_A≠Y_B', 'row-wt', '95% CI', 'match-wt', '95% CI', 'sign(δ) 3-valued disagreement', 'Pearson δ_A,δ_B'], rows))
    w('')
    w('A and B are two frozen estimators of the same unobserved quantity. Disagreement between them measures model sensitivity; neither one is truth.')
    w('')
    w('### Older V1 values (exact keys and B-rule windows only)')
    w('')
    rows = []
    for h in HS:
        d = A['v1_comparison'][str(h)]
        rows.append([f'B{h}', f'{d["rows_exact_key_and_endpoint"]:,}', d['rows_not_comparable'], pct(d['v1_positive_rate']['row_weighted']),
                     pct(d['v2A_positive_rate_same_rows']['row_weighted']), pct(d['label_disagreement_v1_vs_v2A']['row_weighted']),
                     pct(d['label_disagreement_v1_vs_v2A']['match_weighted']), ci(d.get('bootstrap'), 'row_weighted_ci95'), f4(d['pearson_delta_v1_v2A'])])
    w(table(['Window', 'comparable rows', 'not comparable', 'V1 Y=1', 'V2-A Y=1', 'V1 vs V2-A disagreement (row)', '(match)', 'row 95% CI', 'Pearson δ'], rows))
    w('')
    w('V1 means the frozen v1 expanded model on objective_history_v1 states (role-slot schema with the soul-name defect). Disagreement is between model versions, not an error rate.')
    w('')
    w('## 7. Breakdowns (descriptive; no h/model optimization)')
    w('')
    rows = []
    for k, c in A['breakdown_patch_and_time'].items():
        if not c['rows']:
            rows.append([k, 0, 0] + ['NA'] * 6)
            continue
        rows.append([k + (' (sparse)' if c['sparse'] else ''), f'{c["rows"]:,}', f'{c["matches"]:,}', pct(c['Y_A_positive']['match_weighted']),
                     ci(c.get('Y_A_positive_bootstrap')), pct(c['A_vs_B_disagreement']['match_weighted']),
                     pct(c['flip_A_60_vs_90']['match_weighted'], 2), pct(c['flip_A_90_vs_120']['match_weighted'], 2), pct(c['flip_A_60_vs_120']['match_weighted'], 2)])
    w('Cells use the B90 labels; flips are model A; all rates are match-weighted.')
    w('')
    w(table(['Cell', 'rows', 'matches', 'Y_A=1', '95% CI', 'A≠B', 'flip 60/90', 'flip 90/120', 'flip 60/120'], rows))
    w('')
    for h in HS:
        w(f'**Raw events after the last kill, in (L, e{h}]** (both teams, no spatial filter, not attributed to the engagement):')
        w('')
        rows = []
        for cat, c in A['breakdown_objectives_after_last_kill'][str(h)].items():
            p = c['present_after_last_kill']
            owned = f'{c["present_blue"]:,}/{c["present_red"]:,}' if 'present_blue' in c else '—'
            if not p['rows']:
                rows.append([cat, 0, 0, 0, owned, 'NA', 'NA', 'NA'])
                continue
            rows.append([cat + (' (sparse)' if p['sparse'] else ''), f'{c["events_after_last_kill"]:,}', f'{p["rows"]:,}', f'{p["matches"]:,}', owned,
                         pct(p['Y_A_positive']['match_weighted']), ci(p.get('Y_A_positive_bootstrap')), pct(p['A_vs_B_disagreement']['match_weighted'])])
        w(table(['Category', 'events', 'rows present', 'matches', 'rows blue/red owner', 'Y_A=1 | present', '95% CI', 'A≠B | present'], rows))
        w('')
    rows = [[k, v['rows_with_additional_events_in_longer_window']] for k, v in A['objective_events_added_by_longer_window']['90_to_120'].items()]
    rows60 = {k: v['rows_with_additional_events_in_longer_window'] for k, v in A['objective_events_added_by_longer_window']['60_to_90'].items()}
    w(table(['Category', 'rows gaining events 60→90', 'rows gaining events 90→120'], [[k, rows60[k], v] for k, v in rows]))
    w('')
    w(A['objective_note'] + ' "Sparse" means fewer than 30 matches; its rates and intervals are unreliable.'
      ' The owner columns count rows with an event credited to blue or red.')
    w('')
    w('**Events during the engagement (q_pre, L]** (reported separately from after-last-kill events):')
    w('')
    rows = []
    for cat, c in A['breakdown_objectives_during_engagement_q_pre_to_L'].items():
        p = c['present_during_engagement']
        rows.append([cat + (' (sparse)' if p.get('sparse') else ''), f'{c["events_during"]:,}', f'{p["rows"]:,}', pct(p['Y_A_positive']['match_weighted']) if p['rows'] else 'NA'])
    w(table(['Category', 'events', 'rows present', 'Y_A=1 (B90) | present'], rows))
    w('')
    w('**Simultaneous event occurrences**')
    w('')
    w(table(['Occurrence', 'rows'], [[k, f'{v:,}'] for k, v in A['simultaneous_event_occurrences'].items()]))
    w('')
    w('## 8. Verification')
    w('')
    w(table(['Check', 'Pass'], [[k, v] for k, v in V['checks'].items()]))
    w('')
    ra_fail = {k: v for k, v in RA['failure_counts'].items() if v}
    w(f'Independent raw-event audit (separate numpy re-read): {sum(RA["checked"].values()):,} checks over {len(RA["checked"])} check types;'
      f' hard failures: {RA["hard_failures"] or "none"}; non-zero failure counts: {ra_fail or "none"}.')
    w('')
    diagA, diagB = LA['batch_composition_diagnostic_A'], LA['batch_composition_diagnostic_B']
    w('Label audits: p_pre is bitwise identical for every h; δ = p_post − p_pre is exact bitwise; Y = 1(δ > 0) exactly;'
      ' history snapshots never come after their positions; mixing A and B endpoints in one δ is rejected by the adapter guard.')
    w(f'Rows whose endpoints coincide across windows have bitwise-identical StateV2 vectors (violations 60/90: {LA["duplicate_endpoint_state_bitwise_60_90_violations"]},'
      f' 90/120: {LA["duplicate_endpoint_state_bitwise_90_120_violations"]}, 60/120: {LA["duplicate_endpoint_state_bitwise_60_120_violations"]}), so they share one canonical prediction.')
    w(f'**Batch-composition rounding diagnostic** (informational). Re-scoring the post-60/120 queries in an independent batch changes model A in'
      f' {diagA["h60"]["rows_not_bitwise_equal"]}/{diagA["h120"]["rows_not_bitwise_equal"]} rows (max |diff| {diagA["h60"]["max_abs_diff"]:.2g}/{diagA["h120"]["max_abs_diff"]:.2g})'
      f' and model B in {diagB["h60"]["rows_not_bitwise_equal"]}/{diagB["h120"]["rows_not_bitwise_equal"]} rows (max |diff| {diagB["h60"]["max_abs_diff"]:.2g}/{diagB["h120"]["max_abs_diff"]:.2g}).'
      f' Label changes: A {diagA["h60"]["label_changes_under_independent_batch"]}/{diagA["h120"]["label_changes_under_independent_batch"]},'
      f' B {diagB["h60"]["label_changes_under_independent_batch"]}/{diagB["h120"]["label_changes_under_independent_batch"]}.'
      f' Pre-query batch invariance: A rows not equal {LA["p_pre_A_batch_invariant_rows_not_equal"]}, B {LA["p_pre_B_batch_invariant_rows_not_equal"]} (max {LA["p_pre_B_batch_invariant_max_abs_diff"]:.2g}).'
      ' Canonical predictions come from P2\'s exact batch composition, so P2 is reproduced bitwise.')
    w('')
    cr = LA['cross_row_endpoint_equals_later_q_pre']
    w('Cross-row coincidences, where an endpoint equals a later engagement\'s q_pre in the same match: '
      + '; '.join(f'{k}: {v["rows"]:,} rows' + (f', state violations {v["state_bitwise_violations"]}, A max|diff| {v["A_max_abs_diff"]:.2g}, B max|diff| {v["B_max_abs_diff"]:.2g}' if v['rows'] else '') for k, v in cr.items()) + '.')
    w('')
    det, pert, parts = V['determinism'], V['future_perturbation'], V['partitions']
    w(f'Determinism: {len(det["matches"])} matches re-scored in fresh processes; arrays not bitwise equal: {det["arrays_not_bitwise_equal"] or "none"}; boundary rerun equal: {det["boundary_rerun_equal"]}.'
      f' Future perturbation ({len(pert["matches"])} matches, {pert["what_is_perturbed"]}): changed arrays: {pert["arrays_changed"] or "none"}.')
    w(f'Partition overlap (matches): V1 buckets {parts["v1_buckets"]}, P1 {parts["p1"]}, P2 {parts["p2"]}, V1 eval {parts["v1_eval"]}; the cohort is a subset of the engagement split: {parts["p2_cohort_subset_of_engagement"]}.')
    w(f'Feature schema guards: {FS["feature_guards"]}.')
    w(f'Frozen inputs changed during the run: {V["changed_frozen_files"] or "none"} (hashes_before.json / hashes_after.json). Tests: {V["tests"]["summary"]}.')
    w('')
    PR = rj(o / 'post_run_spot_checks.json') if (o / 'post_run_spot_checks.json').exists() else None
    if PR:
        w('### 8b. Post-run spot checks (after validation; scripts/engagement_labels_v3_postrun_checks.py)')
        w('')
        c = PR['csv_float_round_trip']
        worst = max(v['default_parser_max_abs_diff'] for k, v in c.items() if isinstance(v, dict))
        w(f'- **CSV reader caveat.** labels_long.csv is written with exact float repr. Read with `float_precision="round_trip"`, every checked float column is bitwise equal to labels_long.npz:'
          f' {all(v["round_trip_parser_bitwise_equal_npz"] for v in c.values() if isinstance(v, dict))}. pandas\' default parser differs by up to {worst:.2g}.'
          f' Label changes under the default parser: A {c["Y_A_label_changes_if_default_parsed_delta_used"]}, B {c["Y_B_label_changes_if_default_parsed_delta_used"]}.'
          ' For bitwise work use the npz or round-trip parsing.')
        w(f'- labels_long.npz B90 values bitwise equal to P2 endpoint diagnostics (independent of runner code): {PR["npz_b90_bitwise_equal_p2"]}; features p_pre_A equal: {PR["features_p_pre_A_bitwise_equal_npz"]}.')
        nz = PR['near_zero_deltas']
        w(f'- Near-zero deltas: A min |δ| {nz["A"]["min_abs_delta"]:.3g}, exact zeros {nz["A"]["exact_zero_rows"]};'
          f' B min |δ| {nz["B"]["min_abs_delta"]:.3g}, exact zeros {nz["B"]["exact_zero_rows"]}, |δ|<1e-12 {nz["B"]["abs_delta_lt_1e-12_rows"]} rows ({nz["B"]["abs_delta_lt_1e-12_engagements"]} engagements).')
        pl = PR['model_B_repeated_value_plateaus (>=3 distinct matches)']
        for p in pl:
            w(f'- **Model-B plateau**: p_B ≈ {p["p_B_rounded_12"]:.12f} appears in {p["queries_within_1e-9"]} P3 queries ({p["queries_by_kind"]}) across {p["distinct_matches"]} matches'
              f'; model A on the same queries ranges {p["p_A_range_on_same_queries"][0]:.4f}–{p["p_A_range_on_same_queries"][1]:.4f}.'
              f' Seed component ranges: {p["seed_component_ranges"]}. The same value occurs in {p["p2_test_grid_queries_within_1e-9"]}/{p["p2_test_grid_queries_total"]:,} P2 test-grid queries'
              f' and in {p["p2_test_one_minute_queries_within_1e-9"]} P2 one-minute test queries.')
        if not pl:
            w('- No model-B probability value is repeated across 3 or more distinct matches.')
        w(f'- {PR["model_B_plateau_note"]}')
        w(f'- New sources changed after the run (documentation generator only expected): {list(PR["new_sources_changed_after_run"]) or "none"}; unchanged: {PR["new_sources_unchanged_after_run"]}.')
        w('')
    w('## 9. Artifacts')
    w('')
    w(table(['File', 'Content / role'], [
        ['protocol.json', 'definitions, declared tolerances, settings (written before execution)'],
        ['checkpoints/status.json, checkpoints/boundaries.csv, checkpoints/score_chunks/', 'stage status and resumable intermediate results'],
        ['logs/run.log, logs/pytest_output.log, full_stdout.log / full_stderr.log', 'full logs'],
        ['labels_long.csv / labels_long.npz', 'one row per (match, s, h): identifiers, patch, L, all boundary candidates, tied reasons, durations, frame ages, p_pre/p_post/δ/Y for A and B, validity flags, raw event counts, model and source hashes. The npz holds exact float64; read the CSV with float_precision="round_trip" for bitwise values'],
        ['post_run_spot_checks.json', 'post-validation spot checks: CSV round-trip, npz vs P2, model-B plateaus, source hashes after the run'],
        ['features_pre_only.npz', 'PRE-ONLY q inputs: StateV2 at s−1 without snapshot_age_s (champion IDs categorical) plus p_pre_A'],
        ['feature_schema.json', 'column roles: input / input_categorical / key / target candidate / diagnostic / audit-only / forbidden'],
        ['audit_only/*.npz', 'post states, history snapshots, final winner and snapshot ages — outcome side, NOT features'],
        ['source_provenance.json', 'lineage chain, census, input paths, model keys, source hashes'],
        ['reference_audit_b90_endpoints.json, reference_audit_p2_b90_values.json', 'reference identity audits'],
        ['boundary_checks.json, label_audits.json, independent_raw_event_audit.json', 'verification details'],
        ['analysis_results.json, results.json, validation.json', 'all analyses, headline numbers, check summary'],
        ['timeline_examples.json, TIMELINE_EXAMPLES.md, DEFINITION_AND_EVIDENCE.md', 'actual cohort examples; definitions with evidence types'],
    ]))
    w('')
    w('## 10. Readiness for P4 and remaining scientific limits')
    w('')
    w(f'**P4 gate:** {V["p4_gate"]}.')
    w('')
    w('Remaining limits (they are not resolved by passing validation):')
    w('')
    d90 = A['model_A_vs_B_disagreement']['90']
    v190 = A['v1_comparison']['90']['label_disagreement_v1_vs_v2A']
    fl = A['paired_window_flips']
    model_dep = (f'Model dependence: at B90, A and B labels disagree in {pct(d90["label_disagreement"]["row_weighted"])} of rows'
                 f' (match-weighted {pct(d90["label_disagreement"]["match_weighted"])}, 95% CI {ci(d90["bootstrap"])}).'
                 f' V1 and V2-A disagree in {pct(v190["row_weighted"])} of comparable rows. Labels depend on the frozen valuation model,'
                 ' so any q result should be reported under both A and B labels.')
    window_dep = ('Window dependence (model A flips, row-weighted): '
                  + '; '.join(f'{k.replace("_vs_", " vs ")}: {pct(v["A"]["flip"]["row_weighted"], 3)} overall, {pct(v["A"]["flip_among_different_endpoints"]["row_weighted"])} among rows whose endpoint changes'
                              for k, v in fl.items())
                  + '. Rows with identical endpoints cannot flip. B90 remains an operational choice, not an estimated optimum.')
    for item in [
        'Label validity is unestablished. δ_A is a frozen-model probability change; no human or independent semantic check shows that Y_A=1 marks a good engagement outcome. The V1 blinded review forms were never completed (per the project records).',
        'The cohort is selected. It covers the V1 engagement split of the 50k exploratory corpus, only matches that have detector rows, and excludes overlap rows. It is not all engagements, and patches 15.14–15.16 were examined before.',
        model_dep,
        window_dep,
        plateau_limit(o, d90),
        'The B rule ends a window at ANY later kill anywhere on the map and at the retro-dated start of the next detected engagement. It does not separate pursuit kills from new fights and ignores resets. Objectives after L are co-occurrences, not causal consequences.',
        'State staleness: node fields change only about once a minute, so many post-states carry frame data from before L. Event-based fields update at the endpoint.',
        'Both models were trained on minute-aligned queries whose frame age is about 60 s. Engagement queries have ages of 0–60 s, which is extrapolation in the age dimension. P2 reported age strata but did not model them.',
        'Bootstrap intervals cover match resampling only. They exclude training, calibration and selection uncertainty and the arbitrariness of the detector.',
        'Near-zero δ rows (see the |δ|≤0.005/0.01/0.02 counts) make labels sensitive to tiny numerical or model changes. They are kept as Y by the preset rule, not re-thresholded.',
        'P4 must keep q inputs PRE-ONLY (features_pre_only.npz and feature_schema.json), keep patch as a split variable, and must not select h or model on q test results.',
    ]:
        w(f'- {item}')
    w('')
    w('Claims not made: ' + '; '.join(Rz['claims_not_made']) + '.')
    w('')
    (o / 'REPORT.md').write_text('\n'.join(L), encoding='utf-8')


def definitions(o):
    P = rj(o / 'protocol.json')
    d = P['definitions']
    L = []
    w = L.append
    w('# P3 definitions and evidence register')
    w('')
    w('Scope: the exploratory candidate labels in outputs/engagement_labels_v3_sensitivity. Each definition is tagged by evidence type:'
      ' **literature method**, **game rule**, **data estimate** or **researcher (operational) choice**.'
      ' Literature entries are reused from the project documents listed at the end. They were **not re-verified against the original papers during this run**'
      ' (no web access was used), and no new citation was added.')
    w('')
    rows = [
        ['Engagement population, first kill K, s = K − 15 s, last kill L', 'unchanged frozen v3.3 detector outputs (exposures.csv); s/G/D/R/participant rules not modified',
         'researcher choice + game rule (15 s kill/assist attribution scale; see DEFINITION_EVIDENCE_REGISTER_20260914.md)', 'P3 only confirms K and L against raw CHAMPION_KILL events'],
        ['Cohort', 'exact rows and order of state_value_v2_fix/rows.csv', 'historic selection (Section 2 of REPORT.md)', 'availability/selection bias disclosed; not all engagements'],
        ['q_pre', d['q_pre_ms'], 'researcher choice (prediction cutoff before the retro-dated start)', 's is known only after detection; not a real-time stopping time'],
        ['next_kill', d['next_kill_ms'], 'researcher operational choice (re-engagement proxy)', 'does not distinguish pursuit kills from new fights'],
        ['next eligible engagement start', d['next_eligible_start_ms'], 'researcher operational choice', 'retro-dated start of the next detected engagement anywhere on the map'],
        ['game end', d['game_end_ms'], 'data (raw GAME_END)', 'state at game_end − 1 ms; final winner is never imputed as p_post'],
        ['endpoint_h', d['endpoint_h_ms'], 'OUR operational B rule (docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md); h = 60/90/120 are researcher choices', 'no paper proves this rule or these numbers; B90 is the working primary'],
        ['closed inclusion', d['inclusion'], 'implementation contract of gameplay.state_value(_v2).StateBuilder', 'terminal events excluded by the −1 ms'],
        ['tied reasons', d['tied_reasons'], 'researcher choice (no arbitrary ordering of same-timestamp events)', ''],
        ['objective acquisition', d['objective_acquisition'], 'researcher choice', 'objectives before the endpoint accumulate in the state'],
        ['event attribution', d['event_attribution'], 'descriptive only', 'co-occurrence, not causal attribution'],
        ['p_pre_A, p_post_A, δ_A', f'{d["p_pre_A"]}; {d["p_post_A_h"]}; {d["delta_A_h"]}', 'literature method (state-based win probability and valuing changes in it; Maymin 2021)', 'our model, states and windows differ from Maymin; frozen P1 expanded V2 selected in P2'],
        ['Y_A', d['Y_A_h'], 'researcher choice', 'near-zero bands are descriptions only'],
        ['model B', d['B'], 'comparator adapted from Silva et al. (SBGames 2018) architecture in P2', 'diagnostic model sensitivity only'],
        ['value semantics', d['value_semantics'], '—', 'not causal, not monetary'],
    ]
    w('| Definition | Exact rule | Evidence type / source | Limits |')
    w('|---|---|---|---|')
    for r in rows:
        w('| ' + ' | '.join(str(x).replace('|', '\\|') for x in r) + ' |')
    w('')
    w('## Literature used and what it does NOT establish')
    w('')
    for k, v in P['references'].items():
        w(f'- **{k}**: {v}.')
    w('- **Kim, Lee, Chung (CoG 2020)** and **Hodge et al. (IEEE ToG 13(4):368–379, DOI 10.1109/TG.2019.2948469)**: cited in docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md as motivation for calibrated, time-varying win probability. They are not cited for, and are not used as evidence for, the engagement windows.')
    w('- **Truong, Oudre, Vayatis (2020), DOI 10.1016/j.sigpro.2019.107299**: change-point methods were considered in docs/POST_ENGAGEMENT_BOUNDARY_METHODS_20260914.md. They were not used here, and a statistical change point would not be a strategic fight end.')
    w('')
    w('Competing-event (CIF) analysis motivates diagnostics such as "which terminal event came first." It does **not** show that postkill objectives were caused by the fight.'
      ' Maymin motivates valuing a change in win probability between states. The 60/90/120 s cap, the next-kill truncation and the next-engagement truncation are **our** event-based design, not results of that paper.')
    w('')
    w('## Evidence that could not be recovered in this run')
    w('')
    w('- The original literature texts were not re-opened. Bibliographic details are copied from the project documents below.')
    w('- The exact derivation of the detector constants (G = 13.7 s, D = 4,264 u, R = 1,600 u, look-back 15 s) lives in the original repository documents'
      ' (C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/…), which this run did not re-audit. P3 changes none of them.')
    n_no = rj(o / 'source_provenance.json')['lineage_checks']['census']['engagement_matches_without_any_exposure_row']
    w(f'- For the {n_no:,} engagement-split matches without exposure rows, the cause was not re-derived. Re-detection is out of scope.')
    w('')
    w('## Source documents')
    w('')
    for p in ('docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md', 'docs/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md',
              'docs/DEFINITION_EVIDENCE_REGISTER_20260914.md', 'docs/POST_ENGAGEMENT_BOUNDARY_METHODS_20260914.md',
              'docs/EVENT_BOUNDARY_CIF_PROTOCOL_20260914.md', 'docs/B_BOUNDARY_60_90_120_PROTOCOL_20260914.md',
              'docs/B_BOUNDARY_60_90_120_RESULTS_20260914.md', 'docs/CLAUDE_EXECUTE_P3_LABEL_SENSITIVITY.md'):
        w(f'- {p}')
    w('')
    (o / 'DEFINITION_AND_EVIDENCE.md').write_text('\n'.join(L), encoding='utf-8')


def timelines(o):
    E = rj(o / 'timeline_examples.json')
    L = []
    w = L.append
    w('# P3 timeline examples')
    w('')
    w('Every example below except the last section is an **ACTUAL cohort row**. It was selected deterministically by the smallest sha256("p3_example:<type>:<match>:<s>")'
      ' among qualifying rows, never by label value. Times are game clock mm:ss.mmm. Probabilities come from frozen models and are exploratory; they are not ground truth.'
      ' killerId/victimId are Match-V5 participantId values (1–10), not player identities.')
    w('')
    for ex in E['examples']:
        w(f'## {ex["example_id"]} (actual case; {ex["qualifying_rows"]:,} qualifying rows)')
        w('')
        w(f'Match `{ex["match"]}`, patch {ex["patch"]}, row_index {ex["row_index"]}. Selection: {ex["selection_rule"]}.')
        w('')
        w('```text')
        w(f'q_pre (s-1)         {mmss(ex["q_pre_ms"])}   p_pre A={ex["p_pre_A"]:.4f}  B={ex["p_pre_B"]:.4f}   (pre frame at {mmss(ex["pre_snapshot_ms"])})')
        w(f'first kill K=s+15s  {mmss(ex["first_kill_ms"])}   raw kills at K: {ex["first_kill_raw_count"]}')
        w(f'last kill L         {mmss(ex["L_ms"])}   raw kills at L: {ex["kills_at_L_count"]}')
        w(f'next raw kill       {mmss(ex["next_kill_ms"]) if ex["next_kill_ms"] >= 0 else "none (+inf)"}')
        w(f'next eligible start {mmss(ex["next_start_effective_ms"]) if ex["next_start_effective_ms"] >= 0 else "none (sentinel: stored " + mmss(ex["next_start_stored_ms"]) + " = game end)"}')
        w(f'game end            {mmss(ex["game_end_ms"])}')
        for h in HS:
            win = ex['windows'][str(h)]
            w(f'endpoint B{h:<3}       {mmss(win["endpoint_ms"])}   reasons={win["end_reasons"]:<32} follow-up {win["duration_after_last_kill_s"]:7.3f} s   '
              f'A: p_post={win["p_post_A"]:.4f} δ={win["delta_A"]:+.4f} Y={win["Y_A"]}   B: δ={win["delta_B"]:+.4f} Y={win["Y_B"]}   (post frame {mmss(win["post_snapshot_ms"])})')
        w('```')
        w('')
        w('Raw events in the displayed window (read-only cache; kills, elite monsters, souls, buildings, plates, game end):')
        w('')
        w('| time | type | categories | team | subtype/building | killer→victim | relation |')
        w('|---|---|---|---|---|---|---|')
        e90 = ex['windows']['90']['endpoint_ms']
        for ev in ex['raw_events_in_window']:
            t = ev['timestamp_ms']
            rel = ('≤ q_pre' if t <= ex['q_pre_ms'] else 'during (q_pre, L]' if t <= ex['L_ms'] else 'after L, ≤ e90 (in B90 state)' if t <= e90 else 'after e90 (excluded from B90)')
            kv = f'{ev.get("killerId")}→{ev.get("victimId")}' if ev['type'] == 'CHAMPION_KILL' else (str(ev.get('killerId')) if ev.get('killerId') is not None else '')
            w(f'| {mmss(t)} | {ev["type"]} | {",".join(ev["categories"])} | {ev["team"] or ""} | {ev.get("monsterSubType") or ev.get("buildingType") or ""} | {kv} | {rel} |')
        w('')
        cands = ex['windows']['90']['candidates_ms']
        w('B90 candidates: ' + ', '.join(f'{k}={mmss(v) if v >= 0 else "+inf"}' for k, v in cands.items()) + '.')
        w('')
    w('## Types without an actual case in the cohort')
    w('')
    if E['types_without_actual_case']:
        for t in E['types_without_actual_case']:
            w(f'- `{t}`: no qualifying cohort row. See the SYNTHETIC illustration below.')
    else:
        w('- none: every requested type has an actual case above.')
    w('')
    w('## SYNTHETIC illustration (invented integers; not an observed case)')
    w('')
    w('Mirrors tests/test_engagement_labels_v3_rules.py::test_all_tied_reasons_are_kept_synthetic, shown only to explain tie storage.')
    w('')
    w('```text')
    w('SYNTHETIC  L = 01:55.000, next raw kill = 02:55.001, next eligible start = 02:55.001, game end = 33:20.000')
    w('SYNTHETIC  B60 endpoint = min(01:55.000+60 s, 02:55.001-1 ms, 02:55.001-1 ms, 33:19.999) = 02:55.000')
    w('SYNTHETIC  stored reasons = horizon|next_kill|next_engagement_start (all ties kept; none chosen arbitrarily)')
    w('```')
    w('')
    (o / 'TIMELINE_EXAMPLES.md').write_text('\n'.join(L), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default=str(OUT_BASE))
    o = Path(ap.parse_args().out)
    report(o)
    definitions(o)
    timelines(o)
    print('reports written to', o)


if __name__ == '__main__':
    main()
