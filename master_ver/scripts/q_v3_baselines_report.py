"""Write REPORT.md and DEFINITION_AND_EVIDENCE.md for outputs/q_v3_baselines from stored JSON (no recomputation)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q_v3_baselines'
P3 = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
CANDS = ('constant', 'p_pre_logistic', 'p_pre_spline', 'ridge_raw', 'ridge_sigmoid', 'ridge_isotonic',
         'economic_raw', 'economic_sigmoid', 'economic_isotonic')
SPLITS = ('train', 'calibrate', 'select', 'test')


def rd(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def f(x, d=4):
    if x is None: return 'NA'
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return f'{x:,}'
    return f'{x:.{d}f}'


def sgn(x, d=4):
    return 'NA' if x is None else f'{x:+.{d}f}'


def pct(x, d=2): return 'NA' if x is None else f'{100 * x:.{d}f}%'


def ci(c, d=4): return f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c).replace('|', '\\|') for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    proto = rd(OUT / 'protocol.json'); status = rd(OUT / 'status.json'); val = rd(OUT / 'validation.json')
    res = rd(OUT / 'results.json'); boot = rd(OUT / 'bootstrap.json')['results']; sub = rd(OUT / 'subgroups_test.json')['subgroups']
    shp = rd(OUT / 'shap' / 'shap_summary.json') if (OUT / 'shap' / 'shap_summary.json').exists() else None
    post = rd(OUT / 'post_run_checks.json') if (OUT / 'post_run_checks.json').exists() else None
    dchk = rd(OUT / 'data_checks.json'); mapping = rd(OUT / 'schema_mapping_checks.json'); prov = rd(OUT / 'provenance.json')
    p3a = rd(P3 / 'analysis_results.json')['per_window']['90']
    p3v = rd(P3 / 'validation.json'); codex = rd(P3 / 'codex_label_audit.json')
    man = res['manifests']['A90']; sel = res['selections']; M = res['metrics']
    chosen = sel['A90']['chosen']
    elapsed = status['history'][-1]['elapsed_s']
    L = []
    w = L.append
    w('# P4 — Pre-engagement q baselines and explanation (exploratory)')
    w('')
    w(f"**Status:** run `{status['state']}`, validation **`{val['status']}`** ({len(val['checks'])} run checks, failed: {val['failed_checks'] or 'none'}); "
      f"independent post-run checks **`{post['status'] if post else 'not run'}`**. Smoke: {proto['smoke']}. Elapsed {elapsed} s.")
    w('')
    w('**Readiness: EXPLORATORY ONLY.** q here predicts a *model-defined candidate label* (frozen V says Blue win probability rose between '
      'q_pre and the B-rule endpoint). It is not observed engagement success, not ground truth, not a causal effect and not production-ready. '
      'Good or bad q metrics do not resolve the P3 measurement issues listed in section 11.')
    w('')
    w(f"Design: Codex (docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md, sha256 `{proto['spec_sha256']}`). Implementation and execution: Claude Opus 5. "
      f"protocol.json was written before any data verification or fitting (sha256 `{proto['protocol_sha256']}`, {proto['written_at']}).")
    w('')
    w('## 1. Frozen inputs and identity')
    w('')
    w(table(['Item', 'Identity'], [
        ['Labels (P3)', f"{man['label_source']} sha256 `{man['label_source_sha256']}`; P3 validation `{p3v['status']}`; Codex audit rows {codex['rows']:,}, Y_A mismatch {codex['Y_A_mismatch']}, future snapshots {codex['future_snapshots']}"],
        ['Pre-only inputs (P3)', f"{man['input_features']}; schema sha256 `{man['input_schema_sha256']}`"],
        ['Predictor sets', f"sha256 `{man['predictor_sets_sha256']}` (pre_only_schema.json)"],
        ['Label model A (and p_pre_A)', f"`{man['p_pre_A_input_model_sha256']}` (P1 independent_wp_v2_participant_order, frozen)"],
        ['Label model B (B90 sensitivity only)', f"`{res['manifests']['B90']['label_model_sha256']}` (P2 SimpleRNN bundle, frozen, not repaired)"],
        ['Environment', f"python {prov['python'].split()[0]}, numpy {prov['versions']['numpy']}, pandas {prov['versions']['pandas']}, sklearn {prov['versions']['sklearn']}, lightgbm {prov['versions']['lightgbm']}; BLAS 1 thread, LightGBM n_jobs 4, GPU hidden; shap library not installed/not used"],
        ['Runner', f"scripts/run_q_v3_baselines.py sha256 `{prov['runner_sha256']}` (copy in source_snapshot/)"],
    ]))
    w('')
    w('Every manifest (selection files, model bundles, predictions/manifest.json, results.json, bootstrap.json, shap_summary.json) stores the label '
      'source, horizon, label-model hash, p_pre_A model hash and the input schema hash, and says the output is a q probability of a candidate '
      'label, NOT a V win probability.')
    w('')
    w('## 2. Target, inputs and differences from the old q experiment')
    w('')
    w('- Target (primary, fixed before fitting): **A90**: Y = 1(delta_A > 0) at the B-rule 90 s endpoint; exact zero -> 0 '
      f"(A exact zeros: {res['label_exact_zero']['h90_A_exact_zero_rows']}; B90 exact zeros: {res['label_exact_zero']['h90_B_exact_zero_rows']}, all labelled 0).")
    w(f"- Inputs: P3 features_pre_only.npz only. Ridge uses {len(proto['inputs']['ridge_names'])} numeric inputs (361 pre StateV2 numerics minus 10 champion IDs, plus p_pre_A). "
      f"The economic/aggregate LightGBM uses {proto['inputs']['economic_count']} inputs. That subset is the historic `econ` rule mapped `slotK_` -> `participant_slotK_`, "
      "with snapshot_age_s removed. It includes team aggregate kill, objective, dragon/soul and structure counts, so it is **not gold-only**. The full list is in protocol.json.")
    w('- Excluded from all predictors: snapshot_age_s (audit only), champion IDs (categorical, never numeric ranks), positions, post-state, L, '
      'duration, endpoint/reason, future membership, final outcome, event counts after q_pre, target-source diagnostics and match IDs. IDs are used only for grouping.')
    w('')
    w(table(['Aspect', 'Old q (validation_suite_20260914)', 'This run (P4)'], [
        ['Label', 'V1 expanded delta > 0 at B90', 'P3 frozen P1 model A delta > 0 at B90 (A60/A120/B90 sensitivity)'],
        ['p_pre input', mapping['old_p_pre_source'], mapping['new_p_pre_source']],
        ['snapshot_age_s', f"INCLUDED (economic: {mapping['snapshot_age_in_old_economic']}, ridge: {mapping['snapshot_age_in_old_ridge']})", 'EXCLUDED (audit only)'],
        ['Economic inputs', f"{mapping['old_economic_names']}", f"{mapping['new_economic_names']} (old list mapped minus snapshot_age_s: {mapping['economic_equals_old_mapped_minus_snapshot_age']})"],
        ['Ridge inputs', f"{mapping['old_ridge_names']}", f"{mapping['new_ridge_names']} (old list mapped minus snapshot_age_s: {mapping['ridge_equals_old_mapped_minus_snapshot_age']})"],
        ['Candidate pool', 'constant, p_pre x2, full/position LightGBM (+sig/iso), economic (+sig/iso), ridge (+sig/iso)',
         'constant, p_pre x2, ridge raw/sig/iso, economic raw/sig/iso (full/position removed: old schema/target)'],
        ['Selected on development/SELECT', res['old_experiment_context']['old_chosen'], chosen],
    ]))
    w('')
    agree = res['old_experiment_context']['old_vs_new_label_agreement']
    w(f"Old V1 labels agree with the new A90 labels on {pct(agree['A90']['agreement_all'])} of rows ({pct(agree['A90']['agreement_by_split']['test'])} on 15.16), "
      f"and with B90 on {pct(agree['B90']['agreement_all'])}. The old test scores are context only, not an apples-to-apples baseline: "
      + '; '.join(f"{k} AUC {f(v['auc'])}/Brier {f(v['brier'], 5)}/LL {f(v['logloss'])}" for k, v in res['old_experiment_context']['old_test'].items()) + '.')
    if post and 'old_q_selected_shap_snapshot_age_rank' in post:
        rk = post['old_q_selected_shap_snapshot_age_rank']['by_patch']
        w('')
        w('snapshot_age_s in the old selected q (ridge_sigmoid, calibrated log-odds SHAP): '
          + '; '.join(f"patch {k}: rank {v['rank']} of {v['features']} by mean abs SHAP ({v['mean_abs_shap']:.4f})" for k, v in rk.items())
          + '. It was a mid-ranked input there. It is removed here because the P1-P4 contract keeps snapshot_age_s as audit-only metadata; this is not a performance-driven choice. '
          'The in-run field `old_selected_shap_snapshot_age_rank_by_patch` in results.json sorted by the wrong column; the post-run value above supersedes it.')
    w('')
    w('## 3. Split, weights and partitions')
    w('')
    sc = res['split_counts']
    w(table(['Split', 'Rule', 'Rows', 'Matches', 'Expected'], [
        [k, proto['split'][k], f(sc[k]['rows']), f(sc[k]['matches']), f"{proto['split']['expected'][k]['rows']:,}/{proto['split']['expected'][k]['matches']:,}"] for k in SPLITS]))
    w('')
    ow = res['v_partition_overlap']
    w("Keys, patches and order equal P3 features, validation_suite rows.csv and q_predictions.csv exactly. Each split equals the old q_predictions split and the saved "
      "q_results counts. The q partitions are mutually disjoint by match. Match overlap with every V partition is zero: "
      + '; '.join(f"{k} {v}" for k, v in ow.items()) + '.')
    rw = val['metric_reconstruction']['weights']
    w(f"Weights use the existing `weights(g)`: (1/rows per match)/mean, recomputed inside each fitted or evaluated subset, so mean weight = 1 and each match has equal total weight "
      f"(train per-match total {rw['train']['per_match_total_min']:.6f} = rows/matches {rw['train']['rows_over_matches']:.6f}). "
      "Scalers and spline knots were fitted on TRAIN rows only, unweighted, inside the pipelines as before. Logistic and LightGBM losses use TRAIN match weights. "
      "Calibrators were fitted on CALIBRATE with match weights.")
    w('')
    w('## 4. SELECT results and selection (saved before any TEST prediction)')
    w('')
    for t in ('A90', 'A60', 'A120', 'B90'):
        s = sel[t]
        rows = []
        for c in s['ranking']:
            m = M[t][c]['select']
            rows.append([('**' + c + '**') if c == s['chosen'] else c, f(m['brier'], 5), f(m['logloss'], 5), f(m['auc'])])
        w(f"**{t}** — chosen `{s['chosen']}`; selected ridge variant `{s['selected_ridge_variant']}`; selected economic variant `{s['selected_economic_variant']}`; "
          f"selection file sha256 `{s['file_sha256']}`.")
        w('')
        w(table(['Candidate (SELECT rank order)', 'Brier', 'Log loss', 'AUC'], rows))
        w('')
    a90s = [M['A90'][c]['select']['brier'] for c in sel['A90']['ranking'][:2]]
    w(f"The A90 winner beat the runner-up (`{sel['A90']['ranking'][1]}`) by {a90s[1] - a90s[0]:.6f} SELECT Brier. That gap is tiny, but the rule was predeclared and applied as written. "
      'All four selections, together with the bundle hashes, were written before any bundle was reloaded for TEST prediction (validation `selection_before_test`).')
    w('')
    w('## 5. Primary A90: all candidates, all splits')
    w('')
    rows = []
    for c in CANDS:
        for k in SPLITS:
            m = M['A90'][c][k]
            rows.append([('**' + c + '**') if c == chosen else c, k, f(m['n']), f(m['auc']), f(m['brier'], 5), f(m['logloss'], 5),
                         f(m.get('intercept'), 3), f(m.get('slope'), 3), f(m.get('citl_intercept_offset'), 3)])
    w(table(['Candidate', 'Split', 'n', 'AUC', 'Brier', 'Log loss', 'Recal. intercept', 'Recal. slope', 'CITL (offset)'], rows))
    w('')
    w('Recalibration intercept and slope come from the restored `calibration()`: a joint logistic fit of Y on logit(p). CITL is the intercept with logit(p) as an offset (slope fixed at 1). '
      'Calibrated variants are in-sample on CALIBRATE. Constant predictions have no slope.')
    w('')
    op = res['optimism']['A90']
    w(table(['Candidate', 'train−test AUC', 'train−test Brier', 'train−test LL', 'select−test AUC', 'select−test Brier', 'select−test LL'], [
        [c, sgn(op[c]['train_minus_test']['auc']), sgn(op[c]['train_minus_test']['brier'], 5), sgn(op[c]['train_minus_test']['logloss']),
         sgn(op[c]['select_minus_test']['auc']), sgn(op[c]['select_minus_test']['brier'], 5), sgn(op[c]['select_minus_test']['logloss'])] for c in CANDS]))
    w('')
    pr = res['probability_ranges']['A90']
    w(f"Optimism: positive AUC differences and negative Brier/log-loss differences mean the split looked better than TEST. The economic LightGBM fits TRAIN far better than it predicts TEST "
      f"(TRAIN AUC {f(M['A90']['economic_raw']['train']['auc'])} vs TEST {f(M['A90']['economic_raw']['test']['auc'])}). "
      f"Probability ranges over all rows: `{chosen}` min {f(pr[chosen]['min'])}, max {f(pr[chosen]['max'])}, exact 0 or 1 on {pr[chosen]['exact_0_or_1']} rows; "
      f"economic_isotonic exact 0/1 on {pr['economic_isotonic']['exact_0_or_1']} rows.")
    te = M['A90'][chosen]['test']
    w('')
    w(f"**Selected A90 q on TEST (15.16):** AUC {f(te['auc'])}, Brier {f(te['brier'], 5)} (constant {f(M['A90']['constant']['test']['brier'], 5)}), "
      f"log loss {f(te['logloss'], 5)} (constant {f(M['A90']['constant']['test']['logloss'], 5)}), recalibration slope {f(te.get('slope'), 3)}, CITL {f(te.get('citl_intercept_offset'), 3)}. "
      + (lambda x: (f"The isotonic step function outputs exact 0/1 probabilities in its extreme bins: {x['rows_exact_0_or_1']} TEST rows ({x['rows_exact_1']} at 1.0, {x['rows_exact_0']} at 0.0), "
                    f"{x['rows_contradicted']} of them with the opposite label. Log loss clips them at machine epsilon, so those {x['rows_contradicted']} rows alone add "
                    f"{x['logloss_contribution_contradicted_rows']:.4f} to the match-weighted TEST log loss. That is why log loss is worse than the constant while Brier is better. "
                    'This is a descriptive decomposition from post_run_checks.json; the reported metric is unchanged.')
         if x else '')(post.get('a90_selected_test_extreme_probabilities') if post and chosen.endswith('_isotonic') else None))
    w('')
    w('Reliability bins (TEST, match-weighted means):')
    w('')
    w(table(['Bin lo', 'n', 'Predicted', 'Observed'], [[f(b['lo'], 1), f(b['n']), f(b['predicted']), f(b['observed'])] for b in te['bins']]))
    w('')
    w('## 6. Paired match bootstrap on TEST (A90; fixed models, no refitting)')
    w('')
    b = boot['A90']
    rows = []
    for p in b['pairs']:
        for k in ('auc', 'brier', 'logloss'):
            d = p['delta_a_minus_b'][k]
            rows.append([f"{p['a']} − {p['b']}", k, sgn(d['estimate'], 5), ci(d['ci95'], 5), f(d['fraction_replicates_a_better'], 3)])
    w(table(['Pair', 'Metric', 'Estimate', '95% percentile CI', 'Fraction replicates a better'], rows))
    w('')
    w(f"{b['replicates']} replicates, seed {b['seed']}, unit {b['unit']}; {b['matches']:,} matches / {b['rows']:,} rows. Skipped identical pairs: {b.get('skipped_identical_pairs') or 'none'}. "
      'The intervals are conditional on the fitted and selected models. They exclude training, selection and label-model uncertainty. They support no causal, all-patch or '
      'untouched-test claim: 15.16 was already used in the earlier q work.')
    w('')
    w(table(['Model', 'AUC (95% CI)', 'Brier (95% CI)', 'Log loss (95% CI)'], [
        [m, f"{f(v['auc']['estimate'])} [{f(v['auc']['ci95'][0])}, {f(v['auc']['ci95'][1])}]", f"{f(v['brier']['estimate'], 5)} [{f(v['brier']['ci95'][0], 5)}, {f(v['brier']['ci95'][1], 5)}]",
         f"{f(v['logloss']['estimate'], 5)} [{f(v['logloss']['ci95'][0], 5)}, {f(v['logloss']['ci95'][1], 5)}]"] for m, v in b['model_ci'].items()]))
    w('')
    w('## 7. TEST subgroups (A90)')
    w('')
    w('Weights are renormalized within each stratum. AUC is NA for a single class. Strata with fewer than 30 matches are flagged sparse. The 0–2 minute band is empty because the cohort starts at 2 minutes. '
      'Patch is constant on TEST (15.16); the per-split patch results are in section 5. `pre_history_*` strata use objective acquisition at or before q_pre (available to q). '
      '`posthoc_*` strata use events after q_pre, either during (q_pre, L] or after the last kill (L, endpoint_90]. They are **descriptive only**, are not available at prediction time, '
      'and co-occur with the label window.')
    w('')
    rows = []
    for r in sub['A90']:
        name = r['stratum']
        if name.startswith('posthoc_') and name.endswith('_absent'): continue
        if not r['by_model']:
            rows.append([name, 0, 0, 'NA', 'NA', 'NA', 'NA', 'NA', 'empty']); continue
        m = r['by_model'][chosen]; c0 = r['by_model']['constant']; sp = r['by_model']['p_pre_spline']
        rows.append([name, f(r['rows']), f(r['matches']), f(m['auc'], 3), f(m['brier'], 4), f(c0['brier'], 4), f(sp['brier'], 4), f(m['logloss'], 3),
                     ('SPARSE' if m['sparse_lt30_matches'] else '') + ('' if m['auc_na_reason'] is None else ' ' + m['auc_na_reason'])])
    w(table(['Stratum', 'Rows', 'Matches', f'AUC {chosen}', f'Brier {chosen}', 'Brier constant', 'Brier p_pre_spline', f'LL {chosen}', 'Flag'], rows))
    w('')
    worse = [r for r in sub['A90'] if r['by_model'] and not r['by_model'][chosen]['sparse_lt30_matches']
             and r['by_model'][chosen]['brier'] > r['by_model']['constant']['brier'] and not r['stratum'].startswith('posthoc_')]
    w('`posthoc_*_absent` rows and every candidate are in subgroups_test.json. Non-sparse pre-query strata where the selected q has a worse Brier than the constant: '
      + ('; '.join(f"{r['stratum']} ({f(r['by_model'][chosen]['brier'], 4)} vs {f(r['by_model']['constant']['brier'], 4)}, slope {f(r['by_model'][chosen].get('slope'), 2)})" for r in worse) or 'none')
      + '. These are exploratory observations, not tested hypotheses.')
    w('')
    w('## 8. Sensitivity targets (identical pipeline, independent fit and selection)')
    w('')
    rows = []
    for t in ('A90', 'A60', 'A120', 'B90'):
        c = sel[t]['chosen']
        for cand in dict.fromkeys([c, 'constant', 'p_pre_spline', sel[t]['selected_economic_variant'], sel[t]['selected_ridge_variant']]):
            m = M[t][cand]['test']
            rows.append([t, ('**' + cand + '**') if cand == c else cand, f(m['auc']), f(m['brier'], 5), f(m['logloss'], 5), f(m.get('slope'), 3)])
    w(table(['Target', 'Candidate (TEST)', 'AUC', 'Brier', 'Log loss', 'Recal. slope'], rows))
    w('')
    rows = []
    for t in ('A60', 'A120', 'B90'):
        bt = boot[t]
        if 'pairs' not in bt: rows.append([t, bt.get('skipped'), '', '', '']); continue
        p = bt['pairs'][0]
        rows.append([t, f"{p['a']} − {p['b']}"] + [f"{sgn(p['delta_a_minus_b'][k]['estimate'], 5)} {ci(p['delta_a_minus_b'][k]['ci95'], 5)}" for k in ('auc', 'brier', 'logloss')])
    w(table(['Target', 'Pair (supplementary, 1000 replicates)', 'ΔAUC', 'ΔBrier', 'ΔLog loss'], rows))
    w('')
    w('No horizon was selected on TEST. A90 stays primary regardless of these numbers. B90 changes only the label source (frozen P2 model B, with its known rare output plateaus left unrepaired). '
      'Inputs, including p_pre_A from model A, are fixed, which isolates the target choice.')
    w('')
    w('## 9. Cross-target disagreement diagnostic (TEST)')
    w('')
    ct = res['cross_target']
    w(table(['Scoring', 'AUC', 'Brier', 'Log loss', 'Recal. slope'], [
        [f"A90 selected q ({ct['a90_selected']}) vs Y_A90", f(ct['a90_q_vs_Y_A90']['auc']), f(ct['a90_q_vs_Y_A90']['brier'], 5), f(ct['a90_q_vs_Y_A90']['logloss'], 5), f(ct['a90_q_vs_Y_A90'].get('slope'), 3)],
        [f"A90 selected q ({ct['a90_selected']}) vs Y_B90", f(ct['a90_q_vs_Y_B90']['auc']), f(ct['a90_q_vs_Y_B90']['brier'], 5), f(ct['a90_q_vs_Y_B90']['logloss'], 5), f(ct['a90_q_vs_Y_B90'].get('slope'), 3)],
        [f"B90 selected q ({ct['b90_selected']}) vs Y_B90", f(ct['b90_q_vs_Y_B90']['auc']), f(ct['b90_q_vs_Y_B90']['brier'], 5), f(ct['b90_q_vs_Y_B90']['logloss'], 5), f(ct['b90_q_vs_Y_B90'].get('slope'), 3)],
    ]))
    w('')
    w(f"A90 vs B90 labels disagree on {pct(ct['label_disagreement_A90_vs_B90_test']['rows'])} of TEST rows ({pct(ct['label_disagreement_A90_vs_B90_test']['match_weighted'])} match-weighted). "
      f"The Spearman correlation between the A90 and B90 selected q on TEST is {f(ct['spearman_a90_q_vs_b90_q_test'], 3)}. This is disagreement between two model-defined labels. Neither label is truth, "
      'so the A90-q-vs-Y_B90 row is not an accuracy.')
    w('')
    w('## 10. SHAP explanation of the A90 selected q')
    w('')
    if shp is None:
        w('SHAP did not complete; see shap/shap_failure.json.')
    else:
        gx = shp.get('group_exact_final_output', {})
        w(f"Model `{shp['model']}`. Explained rows: {shp['explained_rows']} TEST rows ({shp['explained_matches']} matches). Background: {shp['background_rows']} TRAIN rows "
          f"({shp['background_matches']} matches), uniform. Both use deterministic sha256 ranks ({shp['selection_rule']}).")
        w('')
        w(table(['Layer', 'Method', 'Output scale', 'Additivity / verification'], [
            ['Feature level', shp['method'], shp['scale'],
             f"max |sum phi + base − linear score| {shp.get('additivity_max_abs_residual_vs_linear_reconstruction', 'NA'):.2e}; vs actual pipeline decision {shp.get('additivity_max_abs_residual_vs_actual_pipeline_output', float('nan')):.2e}; base {f(shp['base_value'], 5)} = background mean output (diff {shp.get('base_minus_background_mean_actual_output', float('nan')):.1e}); logit clipping active on {shp.get('logit_clip_active_rows', 'NA')} rows"
             if 'additivity_max_abs_residual_vs_linear_reconstruction' in shp else json.dumps({k: shp[k] for k in shp if 'additivity' in k})],
            ['Group level (final output)', gx.get('method', 'NA'), gx.get('scale', 'NA'),
             f"{gx.get('coalitions')} coalitions x {gx.get('background_rows')} background rows; max |sum phi + v(empty) − q| {gx.get('additivity_max_abs_residual', float('nan')):.2e}; "
             + (f"implementation check on raw base score ({gx['implementation_check_on_raw_base_score']['rows']} rows) vs grouped linear SHAP max diff {gx['implementation_check_on_raw_base_score']['max_abs_diff_vs_grouped_linear_shap']:.1e}" if 'implementation_check_on_raw_base_score' in gx else '')
             + (f"; vs grouped feature-level SHAP max diff {gx['max_abs_diff_vs_grouped_feature_level_exact_shap']:.1e}" if 'max_abs_diff_vs_grouped_feature_level_exact_shap' in gx else '')],
        ]))
        w('')
        if shp['model'].endswith('_isotonic'):
            iso = gx.get('isotonic_notes', {})
            w('**Scale warning.** The feature-level values explain the *raw ridge log-odds (base score)*, not the final isotonic probability. The final q is a monotone step function of '
              f"the raw ridge probability (Spearman on explained rows {f(iso.get('spearman_final_vs_raw_explained'), 4)}; {iso.get('explained_rows_distinct_final_values')} distinct final values). "
              'The group-level table below is the exact interventional Shapley decomposition of the *final* probability over the historic feature groups, and it verifies additivity.')
            w('')
        w(table(['Group', 'Features', 'mean abs sum of feature SHAP (base score)', 'mean abs exact group Shapley (final output)'], [
            [g['group'], g['n_features'], f(g['mean_abs_sum_of_feature_shap'], 4), f(g['mean_abs_group_exact_shapley_final_output'], 4)] for g in shp['group_table']]))
        w('')
        w(table(['Top features by mean abs SHAP (base score)', 'Group', 'Unit', 'mean abs', 'mean signed'], [
            [t['feature'], t['group'], t['unit'], f(t['mean_abs_shap'], 4), sgn(t['mean_signed_shap'], 4)] for t in shp['top_features']]))
        w('')
        w(f"**Correlation caution:** {shp['correlated_top30_pairs_abs_r_ge_0_8_count']} pairs among the top-30 features have |Pearson r| >= 0.8 on TRAIN. They involve "
          f"{len(shp['correlated_top30_features_involved'])} features, mostly participant-slot gold/xp/level/CS (see shap_summary.json). The ridge penalty spreads weight across these collinear columns. "
          'Individual slot-feature attributions can be large and offsetting within a row (mean signed values near 0), so read them at group level. The historic grouping puts '
          '`unknown_objective_team_count` under health_mana_other and baron/elder death-since-acquisition flags under objectives. Groups are a reading aid, not causal blocks. '
          'Interventional SHAP breaks feature dependence (off-manifold combinations). Attributions describe this prediction function only: **no causal, intervention or player-skill claims**. '
          'Reference: Lundberg & Lee (2017), from the project reference registry.')
        w('')
        rows = []
        for cse in shp['local_cases']:
            top = '; '.join(f"{t['feature']}={t['value']:.3g} ({t['shap']:+.3f})" for t in cse.get('top_features', [])[:4])
            grp = ', '.join(f"{k} {v:+.3f}" for k, v in sorted(cse.get('group_exact_final_output', {}).items(), key=lambda kv: -abs(kv[1]))[:3])
            rows.append([cse['row_index'], cse['y'], f(cse['q_final'], 3), top, grp])
        w(table(['Row index (IDs in predictions/ids.csv)', 'Y_A90', 'q', 'Top-4 feature SHAP (base score)', 'Top-3 exact group Shapley (final q)'], rows))
        w('')
        w(f"Determinism: {shp['determinism']}. Arrays: shap/shap_values.npz; tables: shap/global_feature_mean_abs_shap.csv, shap/global_group_shap.csv.")
    w('')
    w('## 11. Measurement limits that q metrics do not solve')
    w('')
    stale = p3a['staleness']['post_snapshot_not_after_L']; nk = p3a['terminal_reasons_inclusive']['next_kill']
    w(f"- **Source staleness (P3):** at B90, {pct(stale['row_weighted'])} of rows ({pct(stale['match_weighted'])} match-weighted) have no node frame after L, so p_post_A often uses pre-L node fields plus events up to the endpoint.")
    w(f"- **Broad next-kill boundary (P3):** {pct(nk['row_weighted'])} of B90 endpoints are set by the next kill anywhere on the map, which does not separate pursuit from a new fight.")
    w(f"- **Model-dependent labels:** Y depends on frozen V. A90 vs B90 labels disagree on {pct(ct['label_disagreement_A90_vs_B90_test']['rows'])} of TEST rows. V calibration does not make delta_V a valid engagement outcome.")
    w(f"- **Selected cohort:** {sum(v['rows'] for v in sc.values()):,} historic engagements (V1 engagement split, overlap-excluded), not all engagements. 15.16 was already observed by earlier q work, so it is not an untouched test.")
    w(f"- **Modest signal:** the selected A90 q has TEST AUC {f(te['auc'], 3)}, recalibration slope {f(te.get('slope'), 2)} and log loss {f(te['logloss'], 4)} "
      f"(constant {f(M['A90']['constant']['test']['logloss'], 4)}). Its AUC gain over p_pre_spline is {sgn(te['auc'] - M['A90']['p_pre_spline']['test']['auc'], 3)} (bootstrap in section 6).")
    w('')
    w('## 12. Implemented vs validated vs unresolved')
    w('')
    w(table(['Item', 'State'], [
        ['Exact cohort/order/labels vs P3; split recovery; V/q disjointness', 'implemented + validated (data_checks.json, post_run_checks.json)'],
        ['9 candidates x 4 targets; selection before TEST; bundles', 'implemented + validated (selection/, serialization_identity.json)'],
        ['Metrics, optimism, reliability, subgroups; bootstrap', 'implemented; metrics independently reconstructed (runner and post-run script)'],
        ['Linear SHAP on base score + exact group Shapley on final q', 'implemented + additivity/determinism validated'],
        ['Transformer/calibrator fitting origins', 'validated by refits on TRAIN/CALIBRATE (validation.json transformer_origins)'],
        ['LightGBM seed-7 refit reproducibility', f"diagnostic: max abs prediction diff {val['transformer_origins'].get('lightgbm_seed7_A90_train_refit_max_abs_pred_diff')}"],
        ['Semantic validity of Y (does "delta_A > 0" mean a won fight?)', 'UNRESOLVED (needs blinded human review)'],
        ['Future-patch generalization of q', 'UNRESOLVED (needs an untouched later patch/season)'],
        ['Label staleness / boundary fixes', 'UNRESOLVED (P3 limits persist)'],
    ]))
    w('')
    w('## 13. Next independent validation (not done here; no scope extension)')
    w('')
    w('1. Future-patch validation: freeze the A90 selection and bundles, then score an untouched later patch with the same P3 label pipeline and no refitting or reselection.')
    w('2. Semantic validation of the label: blinded human review of stratified cases (label disagreement A vs B, next-kill-terminated, stale post frames, objective co-occurrence) with inter-rater agreement.')
    w('3. Staleness-aware label variant (frame-aligned endpoints) as a separately protocolled P3 revision, followed by re-running this P4 protocol unchanged.')
    w('')
    w('## 14. Checks')
    w('')
    w(table(['Run check', 'Result'], [[k, v] for k, v in val['checks'].items()]))
    w('')
    if post:
        w(table(['Independent post-run check', 'Result'], [[k, v] for k, v in post['checks'].items()]))
        w('')
    w(f"Frozen files hashed before/after: {len(rd(OUT / 'hashes_before.json'))}; changed: {val['frozen_changed'] or 'none'}; added by others: {val['frozen_added_not_by_this_run'] or 'none'}.")
    w('')
    w('## 15. References (from the existing registry docs/LITERATURE_GUIDED_NEXT_STEPS_20260914.md; not re-verified in this run)')
    w('')
    for k, v in proto['references'].items():
        w(f'- **{k}**: {v}.')
    w('')
    w('## 16. Files')
    w('')
    files = ['protocol.json', 'status.json', 'provenance.json', 'pre_only_schema.json', 'schema_mapping_checks.json', 'data_checks.json',
             'selection/A90.json', 'selection/A60.json', 'selection/A120.json', 'selection/B90.json', 'fit_logs.json', 'serialization_identity.json',
             'predictions/ids.csv', 'predictions/manifest.json', 'results.json', 'subgroups_test.json', 'bootstrap.json',
             'shap/shap_summary.json', 'shap/shap_values.npz', 'shap/global_feature_mean_abs_shap.csv', 'shap/global_group_shap.csv',
             'validation.json', 'post_run_checks.json', 'hashes_before.json', 'hashes_after.json', 'RUN_NOTES.md', 'DEFINITION_AND_EVIDENCE.md']
    w(table(['File', 'sha256'], [[p, f"`{sha(OUT / p)}`" if (OUT / p).exists() else 'missing'] for p in files]))
    w('')
    w('Model bundles: models/<target>/<candidate>.joblib (hashes in selection/*.json). Per-target predictions: predictions/<target>.npz and .csv (row_index + split, no match IDs).')
    (OUT / 'REPORT.md').write_bytes(('\n'.join(L) + '\n').encode('utf-8'))

    # ------------------------------------------------------------------ DEFINITION_AND_EVIDENCE.md
    D = []
    d = D.append
    d('# P4 definitions and evidence register')
    d('')
    d('Scope: exploratory q baselines in outputs/q_v3_baselines. Evidence types: **literature method**, **existing project convention**, **researcher choice**, **data estimate**. '
      'Literature entries come from the project reference registry (docs/LITERATURE_GUIDED_NEXT_STEPS_20260914.md). They were **not re-verified against the original papers in this run**, and no new citation was added.')
    d('')
    d(table(['Definition', 'Exact rule', 'Evidence type / source', 'Limits'], [
        ['q estimand', proto['estimand'], 'researcher choice (Codex P4 design)', 'selected cohort; model-defined outcome'],
        ['Primary target', 'A90: 1(delta_A > 0) at B-rule 90 s endpoint; exact zero -> 0', 'P3 labels (researcher operational B rule; value-change rationale Maymin 2021)', 'no paper justifies 90 s or the B rule'],
        ['Sensitivity targets', 'A60, A120, B90; same pipeline; independent selection', 'researcher choice', 'no horizon selected on TEST'],
        ['Inputs', 'P3 features_pre_only.npz: StateV2 at s−1 (without snapshot_age_s) + p_pre_A; champion IDs excluded', 'P3 schema contract', 'node fields up to ~60 s stale'],
        ['Economic/aggregate set', proto['inputs']['economic_rule'], 'existing project convention (run_validation_suite.py)', 'not gold-only'],
        ['Split', 'train 15.14; 15.15 by sha256("calibration17:"+match) first 8 hex mod 2 (0 calibrate, 1 select); test 15.16', 'existing project convention', '15.16 not untouched'],
        ['Weights', proto['weights'], 'existing project convention (weights())', 'renormalized per subset'],
        ['Candidates', '; '.join(f"{k}: {v}" for k, v in proto['candidates'].items()), 'existing project convention (hyperparameters restored, no search)', 'reduced pool vs old q'],
        ['Selection', proto['selection'], 'researcher choice (predeclared)', 'tiny SELECT differences can decide'],
        ['Metrics', proto['evaluation']['metrics'], 'literature method (Van Calster et al. 2019 calibration assessment)', 'medical sample-size rules not transferred; Brier mixes calibration and discrimination'],
        ['Uncertainty', 'paired match bootstrap, percentile 95%, fixed models', 'existing project convention', 'excludes training/selection/label uncertainty'],
        ['Subgroups', proto['evaluation']['subgroups'], 'existing P3/V-audit strata', 'posthoc strata are descriptive, not prediction-time'],
        ['SHAP', 'exact linear interventional SHAP on the selected model output scale; tree/isotonic winners get base-score labels; exact group Shapley by coalition enumeration on the final output',
         'literature method (Lundberg & Lee 2017)', 'not causal; collinearity; off-manifold interventions'],
        ['Feature groups', 'historic shap_audit group() with p_pre -> p_pre_A', 'existing project convention', 'reading aid only'],
    ]))
    d('')
    d('## Literature used and what it does NOT establish')
    d('')
    for k, v in proto['references'].items():
        d(f'- **{k}**: {v}.')
    d('')
    d('SHAP (Lundberg & Lee 2017) decomposes a model prediction additively. It does not show that a feature causes engagement success, and it does not measure player skill. '
      'Van Calster et al. (2019) motivates reporting calibration intercept, slope and curves together with discrimination. Good calibration of q against model-defined labels does not validate the labels. '
      'Maymin (2021) motivates valuing win-probability changes; it does not validate our endpoint rule, our cohort or q performance. Decroos et al. (2019) separates component probability models, design '
      'choices and case checks; it does not justify 90 s.')
    d('')
    d('## Evidence that could not be recovered in this run')
    d('')
    d('- The original papers were not re-opened (no web access); bibliographic details come from the registry above.')
    d('- No human semantic review of labels exists yet (validation_suite review forms were never completed). q quality against these labels is not evidence that the labels mean a won fight.')
    d('- No untouched future patch was available within this protocol.')
    d('')
    d('## Source documents')
    d('')
    for s_ in ['docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md', 'docs/LITERATURE_GUIDED_NEXT_STEPS_20260914.md', 'docs/VALIDATION_SUITE_AFTER_20260914.md',
               'outputs/engagement_labels_v3_sensitivity/REPORT.md', 'outputs/engagement_labels_v3_sensitivity/DEFINITION_AND_EVIDENCE.md',
               'scripts/run_validation_suite.py (restored helpers, not imported)']:
        d(f'- {s_}')
    (OUT / 'DEFINITION_AND_EVIDENCE.md').write_bytes(('\n'.join(D) + '\n').encode('utf-8'))
    print('report written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
