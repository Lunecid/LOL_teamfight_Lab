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
import ta20260916_common as T  # noqa: E402


def main():
    T.log_command()
    if (T.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json exists (never overwritten)')
    for d in ('models', 'selection', 'smoke_train_only'):
        if (T.OUT / d).exists():
            raise SystemExit(f'{d}/ exists: protocol must predate every fit')
    snap = T.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        raise SystemExit('integrity snapshot_before.json must precede the protocol')
    schema = C.read_json(T.FC / 'q_pre_only_schema.json')
    ridge = list(schema['predictor_sets']['ridge'])
    names = list(schema['input_names_all'])
    problems = []
    if len(ridge) != T.EXPECTED_RIDGE_COUNT:
        problems.append(f'ridge count {len(ridge)}')
    if C.sha256_json(C.q_feature_sets(names)) != schema['predictor_sets_sha256'] or C.q_feature_sets(names)['ridge'] != ridge:
        problems.append('predictor sets differ from q_feature_sets(input_names_all)')
    if any(n.endswith('champion_id') or 'snapshot_age' in n for n in ridge):
        problems.append('champion_id / snapshot_age among ridge inputs')
    bad_tokens = [n for n in ridge if n != C.P_PRE and any(t in n for t in C.Q_FORBIDDEN_TOKENS)]
    if bad_tokens:
        problems.append(f'forbidden tokens {bad_tokens[:5]}')
    if problems:
        raise SystemExit(f'schema contract failed: {problems}')
    cm = C.read_json(T.CR / 'cohorts' / 'cohort_manifest.json')
    manifest_counts = {'TRAIN': {c: sum(cm['sets']['MAIN_TRAIN']['by_sub_role'][f'fold{k}']['h90'][c] for k in range(5)) for c in T.COHORTS},
                       'Q_CAL': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_CAL']['h90'][c] for c in T.COHORTS},
                       'Q_SELECT': {c: cm['sets']['MAIN_VALIDATION']['by_sub_role']['Q_SELECT']['h90'][c] for c in T.COHORTS},
                       'MAIN_TEST': {c: cm['sets']['MAIN_TEST']['by_sub_role']['TEST']['h90'][c] for c in T.COHORTS}}
    for s in T.EXT:
        manifest_counts[f'EXT_{s}'] = {c: cm['sets'][f'EXT_{s}']['by_sub_role']['EXTERNAL']['h90'][c] for c in T.COHORTS}
    if manifest_counts != T.EXPECTED_COUNTS:
        raise SystemExit(f'parent cohort manifest counts differ from the specification: {manifest_counts}')
    iq_fz = T.iq_frozen()
    iq_proto = C.read_json(T.IQ / 'protocol.json')
    if iq_proto['inputs']['ridge_names_sha256'] != C.sha256_json(ridge):
        raise SystemExit('iq ridge schema differs from the parent schema now on disk')
    fz_fc = C.read_json(T.FC / 'frozen_manifest.json')
    vman = C.read_json(T.FC / 'v_models_manifest.json')
    lman_tr = C.read_json(T.FC / 'labels' / 'labels_trainval_manifest.json')
    before = C.read_json(snap)
    parent_expected = dict(before['sha256_full_corpus_read_targets'])
    parent_expected.update({k: v for k, v in before['sha256_incremental_q'].items()
                            if k.endswith(('frozen_manifest.json', 'protocol.json')) or '/eval/predictions/' in k or '/selection/' in k
                            or '/predictions/' in k})
    parent_expected.update({k: v for k, v in before['sha256_cohort_role'].items() if '/cohorts/' in k or k.endswith('frozen_manifest.json')})
    registry = {f: dict(configs={cfg: T.config_params(f, cfg) for cfg in T.config_names(f)},
                        candidates_lexical_tie_order=T.candidate_names(f), n_candidates=len(T.candidate_names(f)))
                for f in T.FAMILIES}
    if any(r['n_candidates'] != 18 for r in registry.values()):
        raise SystemExit('registry must hold 18 joint candidates per family')
    arch = {}
    for f in T.FAMILIES:
        for cfg in T.config_names(f):
            p = T.config_params(f, cfg)
            m = T.build_module(f, T.EXPECTED_RIDGE_COUNT, p['width'], p['dropout'])
            arch[cfg] = dict(n_params=T.param_count(m), layers=T.layer_shapes(m), repr=str(m))
    proto = dict(
        version=T.VERSION, role=T.ROLE_TAG, written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        design='docs/CLAUDE_TRACK_A_MLP_20260916.md (Claude, continuing the Codex design line); implementation and execution in the same session',
        spec_sha256=C.sha256_file(T.SPEC), provenance_docs_sha256={p.name: C.sha256_file(p) for p in T.PROVENANCE_DOCS},
        gpu_authorisation='user authorised GPU (cuda) for this stage on 2026-09-16; the old-target deep search shares the device',
        prior_test_exposure=('Exploratory follow-up. MAIN TEST 15.16 and all four external sets were already evaluated for the pooled q, '
                             'the old T/N specialists, role arms and the iq PT / logistic / LightGBM winners. Old TEST results are known; '
                             'this run is not an untouched confirmation. No choice here uses TEST performance.'),
        parents=dict(
            read_only=['outputs/full_corpus_training_20260915', 'outputs/cohort_role_training_20260915', 'outputs/incremental_q_training_20260915',
                       'outputs/label_validity_full_20260915', 'outputs/objective_channel_ablation_20260915',
                       'outputs/collaborator_critique_audit_20260915', 'source implementations', 'raw data', 'original manuscript'],
            integrity_snapshot_before_sha256=C.sha256_file(snap),
            full_corpus_frozen_manifest_sha256=C.sha256_file(T.FC / 'frozen_manifest.json'),
            cohort_role_frozen_manifest_sha256=C.sha256_file(T.CR / 'frozen_manifest.json'),
            incremental_q_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(T.IQ)),
            incremental_q_protocol_sha256=C.sha256_file(T.IQ / 'protocol.json'),
            incremental_q_family_winners={k: v['chosen'] for k, v in iq_fz['family_winners'].items()},
            incremental_q_overall_winner={c: o['chosen'] for c, o in iq_fz['overall_winner'].items()},
            cohort_manifest_sha256=C.sha256_file(T.CR / 'cohorts' / 'cohort_manifest.json'),
            q_pre_only_schema_sha256=C.sha256_file(T.FC / 'q_pre_only_schema.json'),
            v_models_manifest_sha256=C.sha256_file(T.FC / 'v_models_manifest.json'), v_oof_sha256=vman['oof_sha256'],
            v_final_sha256=fz_fc['v_final_sha256'],
            labels_trainval_manifest_sha256=C.sha256_file(T.FC / 'labels' / 'labels_trainval_manifest.json'),
            labels_test_external_manifest_sha256=C.sha256_file(T.FC / 'labels' / 'labels_test_external_manifest.json'),
            parent_oof_self_outcome_exclusion=lman_tr['summaries']['MAIN_TRAIN']['oof_self_outcome_exclusion'],
            expected_parent_file_sha256=parent_expected),
        data_contract=dict(
            corpus='MAIN 210000 matches: TRAIN 15.14 (74673), VALIDATION 15.15 (74748), TEST 15.16 (60579); external KR_16.13, KR_16.14_pilot, KR_16.15, NA1_16.13 separately',
            horizon='h90 only; h60/h120 NOT refit in this bounded stage',
            label='Y = 1[delta_h90 > 0] from parent labels unchanged; TRAIN p_pre/Y from own-match-excluded OOF V (verified, V not refit)',
            cohorts='T: known min(cluster_blue, cluster_red) >= 4; N: known min < 4; parent cohort arrays joined row-exact',
            expected_h90_counts=T.EXPECTED_COUNTS, count_mismatch_policy='any mismatch blocks fitting / evaluation until traced',
            roles='TRAIN = parent sub_role fold0..4; Q_CAL / Q_SELECT = parent VALIDATION roles; V_CAL / V_SELECT unused',
            final_training='ALL eligible cohort TRAIN rows; no balanced-only training, per-class or delta reweighting, no new exclusions'),
        inputs=dict(
            learners='predictor_sets["ridge"] of parent q_pre_only_schema.json in exact order (352 numeric incl. p_pre_V; no champion IDs, no snapshot_age)',
            ridge_names=ridge, ridge_names_sha256=C.sha256_json(ridge), predictor_sets_sha256=schema['predictor_sets_sha256'],
            input_names_all_sha256=C.sha256_json(names),
            finiteness='assert finite on every TRAIN/Q_CAL/Q_SELECT/evaluation input row; stop and document otherwise',
            preprocessing='SimpleImputer(median) then StandardScaler, both fit on the applicable TRAIN subset (fit90 in the stop phase; all cohort TRAIN in the refit)',
            no_additions='no draft / role / position / cohort / post-state / final winner inputs'),
        families=dict(
            grid=dict(width=list(T.WIDTHS), dropout=list(T.DROPOUTS), configurations=[dict(width=w, dropout=d) for w, d in T.GRID]),
            mlp='Linear(352,w)-ReLU-Dropout(d)-Linear(w,w)-ReLU-Dropout(d)-Linear(w,1) (plain matched-input MLP, CoG lineage)',
            resmlp='Linear(352,w) projection; 2 x [LayerNorm(w)-Linear(w,2w)-ReLU-Dropout(d)-Linear(2w,w)-Dropout(d)-residual]; LayerNorm(w)-ReLU-Linear(w,1)',
            architectures=arch, optimizer=T.OPT, seeds=list(T.SEEDS),
            loss='minibatch mean(w_i * BCEWithLogits_i), uniform shuffled minibatches, no per-batch renormalisation, no weighted sampler; '
                 'population loss sum(w*BCE)/sum(w) (fixture in the contract tests)',
            prediction='mean raw probability over seeds 7, 42, 123 (never best seed); seed probabilities stored',
            prediction_definition=f'saved float32 weights evaluated on CPU in float64, torch threads {T.PRED_THREADS}, chunk {T.PRED_CHUNK} rows; '
                                  'the stop curve is a GPU float32 diagnostic',
            stopping=dict(allocation=f"int(sha256('{T.STOP_TAG}'+match_id).hexdigest()[:8],16) % {Q.STOP_MOD} == 0 -> stop10 (iq allocation; same matches as the LightGBM arm, same for T and N)",
                          preprocessing='imputer + scaler fit on fit90 rows only in the stop phase',
                          metric='match-weighted Brier on stop10 after every epoch (weights recomputed inside stop10; GPU float32, eval mode)',
                          patience=T.OPT['patience'], cap=T.OPT['max_epochs'], ties='strict improvement required -> earlier epoch kept (first minimum)',
                          refit='from scratch on ALL cohort TRAIN rows, preprocessing refit there, exactly best_epoch epochs, same seed, per config and seed',
                          never='Q_CAL / Q_SELECT / TEST never used for stopping'),
            determinism='torch.manual_seed(seed), CPU generator for minibatch order, cudnn deterministic, torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8',
            device='cuda (training); cpu float64 (predictions)', threads=T.CPU_THREADS,
            weights='fc20260915_common.weights: equal total weight per match within EVERY fit subset (TRAIN, fit90, stop10, Q_CAL, Q_SELECT, evaluation cell), row mean 1',
            comparison_note=('Both families read the identical 352 inputs as the iq full logistic / LightGBM; the PT baseline reads only p_pre and time. '
                             'Six configurations per family is a bounded search convention, not equal computation or global optimality.')),
        calibration_selection=dict(
            calibrators='raw identity; sigmoid and isotonic fit on cohort Q_CAL only via cr20260915_common.fit_calibrators (sigmoid: LogisticRegression(C=1e6, max_iter=1000) on logit(clip(raw, 1e-8)); no positive-slope constraint; isotonic out_of_bounds=clip)',
            order='three-seed raw mean first, then calibration (never per seed)',
            q_cal_weights='weights(match ids of cohort Q_CAL rows)', q_select_weights='weights(match ids of cohort Q_SELECT rows)',
            registry=registry,
            rule='rank all 18 joint candidates of a family by cohort Q_SELECT match-weighted Brier, then log loss, then candidate name (lexical)',
            metric_implementation='fc20260915_common.evaluate (sklearn brier_score_loss / log_loss with match weights)',
            overall_mlp_families='same rule over the 36 joint candidates of the two MLP families (alias overall_winner_mlp_families)',
            five_family_ranking='information only: same rule over the iq PT / logistic / LightGBM winners and the two MLP winners on the identical cohort Q_SELECT rows (iq winners not refit)',
            no_subset_selection='no selection within balanced / time / external subsets',
            freeze='frozen_manifest.json with all bundles, selections and hashes for BOTH cohorts, plus the iq frozen manifest and iq evaluation prediction file hashes, before any TEST/external array or iq/parent TEST prediction is opened; sealed access gated on THIS run\'s freeze',
            reload='all saved bundles reloaded and predictions compared exactly before the freeze'),
        evaluation=dict(
            sets=list(T.EVAL_SETS), primary='T h90', secondary='N h90',
            cells=dict(all='all h90-valid cohort rows', B40='0.40 <= p_pre <= 0.60 (inclusive)', B45='0.45 <= p_pre <= 0.55 (inclusive)',
                       time='time_minutes bins [0,10), [10,20), [20,30), [30,inf)', B40_x_time='B40 within each time bin',
                       membership='frozen p_pre and pre time only; common to all models; no small-delta exclusion; empty/single-class cells reported'),
            comparators=dict(new=list(T.NEW_WINNERS), overall_mlp='alias overall_winner_mlp_families',
                             iq=dict(source='outputs/incremental_q_training_20260915/eval/predictions/<set>_h90_<cohort>.npz named__* columns joined by exact (match, s_ms)',
                                     winners=list(T.IQ_WINNERS), legacy=list(T.IQ_LEGACY),
                                     label='frozen iq winners (same protocol, older stage) and legacy references with an older fixed fitting budget')),
            metrics=('rows, distinct matches, positives / rate, match-weighted Brier / log loss / AUC, calibration intercept / slope (descriptive), '
                     'fixed 10-bin reliability incl. empty bins, ECE, exact 0/1 probability counts; weights recomputed inside each cell; '
                     'AUC NA for single-class cells; <30 matches marked sparse without intervals or broad conclusions'),
            contrasts=[dict(a=a, b=b, label=l) for a, b, l in T.CONTRASTS],
            bootstrap=dict(replicates=T.BOOT_REPS, seed=T.BOOT_SEED, unit='match', paired='same draw for all models in a cell',
                           where='all / B40 / B45 cells of every set and cohort with >= 30 matches and both label classes; all six planned contrasts',
                           time_cells='point estimates and reliability only; CI explicitly not computed',
                           implementation='per-match weighted loss sums; exact weighted AUC (ties 1/2); degenerate replicates counted',
                           scope='fixed-model evaluation uncertainty only; no multiple-comparison adjustment')),
        deferred=['h60/h120 refits', 'SHAP', 'revised V', 'logit-label decomposition', 'Track B CoG representation adapters',
                  'development-only detector re-estimation', 'C role models (outside this comparison)'],
        not_tested='final-W causal effects; semantic correctness of Y',
        interpretation_choices=[
            'Preprocessing (median imputer, StandardScaler) is fit on unweighted rows of the applicable TRAIN subset, as in the parent pipelines; '
            'the model loss uses the equal-per-match weights.',
            'The refit uses the same seed as the stop phase, so the initial weights coincide; the data (all TRAIN rows) and the fixed epoch count differ.',
            'A candidate needs finite Q_CAL/Q_SELECT predictions to be eligible; MLP fits have no convergence criterion beyond the epoch rule.',
            'Predictions are defined on CPU in float64 (device-independent); GPU float32 stop-curve values may differ in the last digits from a CPU '
            'recomputation and are diagnostics only.',
            "Smoke = TRAIN-only rows of matches with sha256('iq20260915_smoke_subset:'+match)[:8] mod 8 == 0 and pseudo roles fold0-2 TRAIN / "
            'fold3 Q_CAL / fold4 Q_SELECT, output under smoke_train_only/, identical settings; smoke evaluation uses pseudo Q_SELECT rows and '
            'stand-in comparator columns; smoke models are never final models.'],
        environment=T.environment(), source_sha256=T.own_source_hashes())
    sha = C.write_json(T.OUT / 'protocol.json', proto)
    print('protocol sha256', sha)


if __name__ == '__main__':
    main()
