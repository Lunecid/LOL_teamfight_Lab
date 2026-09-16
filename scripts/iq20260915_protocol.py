"""Stage P: write protocol.json before any smoke or full fit (spec/parent hashes, schema, counts, registry, policies)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402


def main():
    Q.log_command()
    if (Q.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    for d in ('models', 'selection', 'smoke_train_only'):
        if (Q.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate every fit')
    snap = Q.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    schema = C.read_json(Q.FC / 'q_pre_only_schema.json')
    ridge = list(schema['predictor_sets']['ridge'])
    names = list(schema['input_names_all'])
    problems = []
    if len(ridge) != Q.EXPECTED_RIDGE_COUNT:
        problems.append(f'ridge count {len(ridge)}')
    if C.sha256_json(C.q_feature_sets(names)) != schema['predictor_sets_sha256'] or C.q_feature_sets(names)['ridge'] != ridge:
        problems.append('predictor sets differ from q_feature_sets(input_names_all)')
    if any(n.endswith('champion_id') or 'snapshot_age' in n for n in ridge):
        problems.append('champion_id / snapshot_age among ridge inputs')
    if not all(n in ridge for n in Q.PT_INPUTS):
        problems.append('PT inputs missing from ridge schema')
    bad_tokens = [n for n in ridge if n != C.P_PRE and any(t in n for t in C.Q_FORBIDDEN_TOKENS)]
    if bad_tokens:
        problems.append(f'forbidden tokens {bad_tokens[:5]}')
    if problems:
        raise SystemExit(f'schema contract failed: {problems}')
    cm = C.read_json(Q.CR / 'cohorts' / 'cohort_manifest.json')
    manifest_counts = {'TRAIN': {c: sum(cm['sets']['MAIN_TRAIN']['by_sub_role'][f'fold{k}']['h90'][c] for k in range(5)) for c in Q.COHORTS},
                       'Q_CAL': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_CAL']['h90'][c] for c in Q.COHORTS},
                       'Q_SELECT': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_SELECT']['h90'][c] for c in Q.COHORTS},
                       'MAIN_TEST': {c: cm['sets']['MAIN_TEST']['by_sub_role']['TEST']['h90'][c] for c in Q.COHORTS}}
    for s in Q.EXT:
        manifest_counts[f'EXT_{s}'] = {c: cm['sets'][f'EXT_{s}']['by_sub_role']['EXTERNAL']['h90'][c] for c in Q.COHORTS}
    if manifest_counts != Q.EXPECTED_COUNTS:
        raise SystemExit(f'parent cohort manifest counts differ from the specification: {manifest_counts}')
    fz_cr = C.read_json(Q.CR / 'frozen_manifest.json')
    fz_fc = C.read_json(Q.FC / 'frozen_manifest.json')
    vman = C.read_json(Q.FC / 'v_models_manifest.json')
    lman_tr = C.read_json(Q.FC / 'labels' / 'labels_trainval_manifest.json')
    oofx = lman_tr['summaries']['MAIN_TRAIN']['oof_self_outcome_exclusion']
    before = C.read_json(snap)
    parent_expected = {k: v for k, v in before['sha256_full_corpus_read_targets'].items()}
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items()
                            if k.split('/')[-1].startswith(('A_MAIN_TEST_h90', 'A_EXT_')) and '_h90_' in k or '/cohorts/' in k
                            or k.endswith(('frozen_manifest.json', 'protocol.json'))})
    registry = {f: dict(configs={cfg: Q.config_params(f, cfg) for cfg in Q.config_names(f)},
                        candidates_lexical_tie_order=Q.candidate_names(f), n_candidates=len(Q.candidate_names(f)))
                for f in Q.FAMILIES}
    if any(r['n_candidates'] != 18 for r in registry.values()):
        raise SystemExit('registry must hold 18 joint candidates per family')
    proto = dict(
        version=Q.VERSION, role=Q.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_INCREMENTAL_Q_TRAIN_20260915.md (Codex design/audit); implementation and execution Claude Opus 5',
        spec_sha256=C.sha256_file(Q.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in Q.PROVENANCE_DOCS},
        prior_test_exposure=('Exploratory follow-up. MAIN TEST 15.16 and all four external sets were already evaluated for the pooled q, '
                             'the old T/N specialists and role arms. Old TEST results are known to the designers and implementer; '
                             'this run is not an untouched confirmation. No choice here uses TEST performance.'),
        parents=dict(
            read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/label_validity_full_20260915',
                       'outputs/objective_channel_ablation_20260915', 'outputs/collaborator_critique_audit_20260915', 'source implementations',
                       'raw data', 'original manuscript'],
            integrity_snapshot_before_sha256=C.sha256_file(snap),
            full_corpus_frozen_manifest_sha256=C.sha256_file(Q.FC / 'frozen_manifest.json'),
            cohort_role_frozen_manifest_sha256=C.sha256_file(Q.CR / 'frozen_manifest.json'),
            cohort_manifest_sha256=C.sha256_file(Q.CR / 'cohorts' / 'cohort_manifest.json'),
            q_pre_only_schema_sha256=C.sha256_file(Q.FC / 'q_pre_only_schema.json'),
            v_models_manifest_sha256=C.sha256_file(Q.FC / 'v_models_manifest.json'), v_oof_sha256=vman['oof_sha256'],
            v_final_sha256=fz_fc['v_final_sha256'],
            labels_trainval_manifest_sha256=C.sha256_file(Q.FC / 'labels' / 'labels_trainval_manifest.json'),
            labels_test_external_manifest_sha256=C.sha256_file(Q.FC / 'labels' / 'labels_test_external_manifest.json'),
            parent_oof_self_outcome_exclusion=oofx,
            legacy_A_specialist_h90_chosen={c: fz_cr['A_specialists'][f'{c}_h90']['chosen'] for c in Q.COHORTS},
            legacy_pooled_h90_chosen=fz_cr['pooled_reference']['q_selections']['h90']['chosen'],
            expected_parent_file_sha256=parent_expected),
        data_contract=dict(
            corpus='MAIN 210000 matches: TRAIN 15.14 (74673), VALIDATION 15.15 (74748), TEST 15.16 (60579); external KR_16.13, KR_16.14_pilot, KR_16.15, NA1_16.13 separately',
            horizon='h90 only; h60/h120 definitions frozen, NEW refits for those horizons NOT executed in this bounded stage',
            label='Y = 1[delta_h90 > 0] from parent labels unchanged; TRAIN p_pre/Y from own-match-excluded OOF V (verified, V not refit)',
            cohorts='T: known min(cluster_blue, cluster_red) >= 4; N: known min < 4; parent cohort arrays joined row-exact',
            expected_h90_counts=Q.EXPECTED_COUNTS, count_mismatch_policy='any mismatch blocks fitting / evaluation until traced',
            roles='TRAIN = parent sub_role fold0..4; Q_CAL / Q_SELECT = parent VALIDATION roles; V_CAL / V_SELECT unused',
            final_training='ALL eligible cohort TRAIN rows; no balanced-only training, per-class or delta reweighting, no new exclusions'),
        inputs=dict(
            full_input_learners='predictor_sets["ridge"] of parent q_pre_only_schema.json in exact order (352 numeric incl. p_pre_V; no champion IDs, no snapshot_age)',
            ridge_names=ridge, ridge_names_sha256=C.sha256_json(ridge), predictor_sets_sha256=schema['predictor_sets_sha256'],
            input_names_all_sha256=C.sha256_json(names), pt_inputs=list(Q.PT_INPUTS),
            pt_time_definition='time_minutes = q_pre / 60000 (pre time) from the parent pre-only matrix',
            finiteness='assert finite on every TRAIN/Q_CAL/Q_SELECT/evaluation input row; stop and document otherwise',
            imputation='SimpleImputer(median) fit on the applicable TRAIN subset only (identity on finite data; explicit precaution)',
            scaling='StandardScaler for logistic families only (PT design and full logistic); LightGBM unscaled',
            no_additions='no draft / role / position / cohort / post-state / final winner inputs'),
        families=dict(
            common_C_grid=list(Q.C_GRID),
            logistic=dict(solver='lbfgs', tol=Q.LOGIT_TOL, max_iter=Q.LOGIT_MAX_ITER, random_state=Q.LOGIT_RANDOM_STATE,
                          retry=f'only if not converged: one retry with max_iter {Q.LOGIT_RETRY_MAX_ITER}, unchanged tol/data/settings; first attempt retained',
                          converged='n_iter < max_iter and no sklearn ConvergenceWarning',
                          still_nonconverged='documented failure: configuration and its 3 calibrations are ineligible for selection; predictions retained'),
            PT=dict(spline=Q.SPLINE, design='[Bp (6), Bt (6), Bp_i*Bt_j p-major/t-minor (36)] = 48 columns', expected_dimension=Q.PT_DIM,
                    then='StandardScaler (TRAIN) -> weighted LogisticRegression per C',
                    degeneracy='duplicate knots / rank deficiency recorded as implementation issue; grid unchanged'),
            logit=dict(pipeline='SimpleImputer(median) -> StandardScaler -> weighted LogisticRegression per C (352 inputs)'),
            lgbm=dict(grid=[dict(num_leaves=a, min_child_samples=b) for a, b in Q.LGBM_GRID], fixed=Q.LGBM_FIXED, seeds=list(Q.LGBM_SEEDS),
                      prediction='mean raw probability over seeds 7, 42, 123 (never best seed); seed probabilities stored',
                      stopping=dict(allocation=f"int(sha256('{Q.STOP_TAG}'+match_id).hexdigest()[:8],16) % {Q.STOP_MOD} == 0 -> stop10 (same match allocation for T and N)",
                                    preprocessing='median imputer fit on fit90 rows only in the stop phase',
                                    metric='explicit match-weighted Brier on stop10 (weights recomputed inside stop10); metric=None disables built-in metrics',
                                    patience=Q.LGBM_PATIENCE, ties='strict improvement required (min_delta 0) -> earlier iteration kept',
                                    refit='from scratch on ALL cohort TRAIN rows, imputer refit there, n_estimators = chosen iteration, per config and seed',
                                    never='Q_CAL / Q_SELECT / TEST never used for stopping'),
                      threads='n_jobs=4, sequential fits'),
            weights='fc20260915_common.weights: equal total weight per match within EVERY fit subset (TRAIN, fit90, stop10, Q_CAL, Q_SELECT, evaluation cell), row mean 1',
            comparison_note=('PT vs full families deliberately differ in input information; full logistic vs full LightGBM hold the 352 inputs fixed. '
                             'Six configs per family is a bounded search convention, not equal computation or global optimality.')),
        calibration_selection=dict(
            calibrators='raw identity; sigmoid and isotonic fit on cohort Q_CAL only via cr20260915_common.fit_calibrators (sigmoid: LogisticRegression(C=1e6, max_iter=1000) on logit(clip(raw, 1e-8)); no positive-slope constraint; isotonic out_of_bounds=clip)',
            tree_order='three-seed raw mean first, then calibration (never per seed)',
            q_cal_weights='weights(match ids of cohort Q_CAL rows)', q_select_weights='weights(match ids of cohort Q_SELECT rows)',
            registry=registry,
            rule='rank all 18 joint candidates of a family by cohort Q_SELECT match-weighted Brier, then log loss, then candidate name (lexical)',
            metric_implementation='fc20260915_common.evaluate (sklearn brier_score_loss / log_loss with match weights)',
            overall='same rule over all 54 joint candidates (identical to ranking the three family winners); all family winners retained',
            no_subset_selection='no selection within balanced / time / external subsets',
            freeze='frozen_manifest.json with all bundles, selections and hashes for BOTH cohorts before any TEST/external array or parent TEST prediction is opened',
            reload='all saved bundles reloaded and predictions compared exactly before the freeze'),
        evaluation=dict(
            sets=list(Q.EVAL_SETS), primary='T h90', secondary='N h90',
            cells=dict(all='all h90-valid cohort rows', B40='0.40 <= p_pre <= 0.60 (inclusive)', B45='0.45 <= p_pre <= 0.55 (inclusive)',
                       time='time_minutes bins [0,10), [10,20), [20,30), [30,inf)', B40_x_time='B40 within each time bin',
                       membership='frozen p_pre and pre time only; common to all models; no small-delta exclusion; empty/single-class cells reported'),
            comparators=dict(new=list(Q.NEW_WINNERS), overall_winner='alias of the selected overall joint candidate',
                             legacy=dict(source='outputs/cohort_role_training_20260915/eval/predictions/A_<set>_h90_<cohort>.npz joined by exact (match, s_ms)',
                                         mapping={k: (v or 'spec_<frozen specialist_chosen>') for k, v in Q.LEGACY.items()},
                                         label='legacy references with an older fixed fitting budget, not newly tuned families')),
            metrics=('rows, distinct matches, positives / rate, match-weighted Brier / log loss / AUC, calibration intercept / slope (descriptive), '
                     'fixed 10-bin reliability incl. empty bins, ECE, exact 0/1 probability counts; weights recomputed inside each cell; '
                     'AUC NA for single-class cells; <30 matches marked sparse without intervals or broad conclusions'),
            contrasts=[dict(a=a, b=b, label=l) for a, b, l in Q.CONTRASTS],
            bootstrap=dict(replicates=Q.BOOT_REPS, seed=Q.BOOT_SEED, unit='match', paired='same draw for all models in a cell',
                           where='all / B40 / B45 cells of every set and cohort with >= 30 matches and both label classes; all five planned contrasts',
                           time_cells='point estimates and reliability only; CI explicitly not computed',
                           implementation='per-match weighted loss sums; exact weighted AUC (ties 1/2); degenerate replicates counted',
                           scope='fixed-model evaluation uncertainty only; no multiple-comparison adjustment')),
        deferred=['h60/h120 refits', 'SHAP', 'revised V', 'logit-label decomposition', 'raw timing correction', 'detector retraining',
                  'MLP / residual MLP', 'full CoG lineup', 'C role models (outside this bounded comparison)'],
        not_tested='final-W causal effects; semantic correctness of Y',
        interpretation_choices=[
            'Preprocessing (spline quantile knots, median imputer, StandardScaler) is fit on unweighted cohort TRAIN rows, as in the parent pipelines '
            '(sample_weight only reaches the estimator); all model losses use the equal-per-match weights.',
            'Logistic convergence = n_iter < max_iter and no ConvergenceWarning; a configuration still not converged after the one retry is ineligible '
            'for selection (documented failure), its predictions are kept.',
            'LightGBM deterministic=True and force_col_wise=True are reproducibility / threading switches, not tuned hyperparameters; subsample=1 '
            'means no bagging; the stop set is binned with the training Dataset reference (LightGBM default).',
            'LightGBM candidates share one median imputer per fit subset across seeds (identical fit on identical rows).',
            'Selection metrics come from fc20260915_common.evaluate; ranking tuples compare floats exactly, then candidate name.',
            'A candidate also needs finite Q_CAL/Q_SELECT predictions to be eligible.',
            'Legacy old_A_specialist = spec_<specialist_chosen> in the parent A npz, asserted equal to the parent frozen A_specialists h90 choice.',
            'All 54 joint candidates receive descriptive all/B40 metrics on every evaluation set; these are never used for selection.',
            "Smoke = TRAIN-only rows of matches with sha256('iq20260915_smoke_subset:'+match)[:8] mod 8 == 0 and pseudo roles fold0-2 TRAIN / "
            'fold3 Q_CAL / fold4 Q_SELECT, output under smoke_train_only/; smoke evaluation uses pseudo Q_SELECT rows and stand-in legacy columns; '
            'smoke models are never final models.'],
        environment=Q.environment(), source_sha256=Q.own_source_hashes())
    sha = C.write_json(Q.OUT / 'protocol.json', proto)
    print('protocol sha256', sha)


if __name__ == '__main__':
    main()
