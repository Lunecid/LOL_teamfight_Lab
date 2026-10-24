"""Contract tests for the 2026-09-15 incremental q training run (synthetic and TRAIN-only fixtures; no VALIDATION/TEST rows).

Covers: exact 352-input schema and PT input whitelist / 48-column design and tensor order, TRAIN-only knots, imputers
and scalers, logistic retry policy, fit90/stop10 match disjointness and T/N allocation, equal-match weights, LightGBM
match-weighted Brier stopping with earlier ties and from-scratch full refit, raw seed ensemble before calibration,
Q_CAL/Q_SELECT weight separation, deterministic selection with ties, balanced boundary inclusion and time bins, row-key
completeness, own-match OOF provenance, bundle reload identity, the sealed-set freeze gate, bootstrap/metric identities
and parent preservation.
"""
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('iq20260915_*.py'))


def _synthetic(n_matches=400, rows_per=(1, 6), p=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(rows_per[0], rows_per[1], size=n_matches)
    g = np.repeat(np.array([f'KR_{i:07d}' for i in range(n_matches)]), counts)
    n = len(g)
    X = rng.normal(size=(n, p))
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, g


# ------------------------------------------------------------------------------------------ inputs / PT design
def test_ridge_schema_is_exact_352_ordered_and_pt_inputs_whitelisted():
    schema = C.read_json(Q.FC / 'q_pre_only_schema.json')
    ridge = schema['predictor_sets']['ridge']
    assert len(ridge) == Q.EXPECTED_RIDGE_COUNT == 352
    assert C.q_feature_sets(schema['input_names_all'])['ridge'] == ridge
    assert C.sha256_json(C.q_feature_sets(schema['input_names_all'])) == schema['predictor_sets_sha256']
    assert not [n for n in ridge if n.endswith('champion_id') or 'snapshot_age' in n]
    assert set(Q.PT_INPUTS) <= set(ridge) and ridge[-1] == 'p_pre_V'
    names = schema['input_names_all']
    assert [names[i] for i in Q.input_columns(names, 'pt', ridge)] == ['p_pre_V', 'time_minutes']
    assert [names[i] for i in Q.input_columns(names, 'logit', ridge)] == ridge
    with pytest.raises(ValueError):
        Q.LinearQBase('pt', 'pt_C1', ['p_pre_V', 'time_minutes', 'blue_kills'])


def test_pt_design_has_48_columns_in_p_major_t_minor_order():
    rng = np.random.default_rng(1)
    p, t = rng.random(3000), rng.uniform(2, 60, 3000)
    d = Q.PTDesign().fit(p, t)
    B = d.transform(p, t)
    assert B.shape[1] == Q.PT_DIM == len(d.names) == 48
    assert d.names[:6] == [f'Bp{i}' for i in range(6)] and d.names[6:12] == [f'Bt{j}' for j in range(6)]
    for i in range(6):
        for j in range(6):
            k = 12 + i * 6 + j
            assert d.names[k] == f'Bp{i}*Bt{j}'
            assert np.array_equal(B[:, k], B[:, i] * B[:, 6 + j])
    assert np.linalg.matrix_rank(B) == 48


def test_pt_knots_and_scaler_come_from_train_rows_only():
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    rng = np.random.default_rng(2)
    Xtr = np.column_stack([rng.beta(2, 2, 4000), rng.uniform(2, 50, 4000)])
    Xte = np.column_stack([rng.beta(5, 1, 4000), rng.uniform(20, 70, 4000)])
    y = (rng.random(4000) < Xtr[:, 0]).astype(int)
    w = np.ones(4000)
    b = Q.LinearQBase('pt', 'pt_C1', list(Q.PT_INPUTS)).fit(Xtr, y, w)
    ref = SplineTransformer(**Q.SPLINE).fit(Xtr[:, :1])
    assert np.array_equal(b.pt.p_spline.bsplines_[0].t, ref.bsplines_[0].t)
    other = SplineTransformer(**Q.SPLINE).fit(Xte[:, :1])
    assert not np.array_equal(ref.bsplines_[0].t, other.bsplines_[0].t)
    sc = StandardScaler().fit(b.pt.transform(Xtr[:, 0], Xtr[:, 1]))
    assert np.array_equal(sc.mean_, b.scaler.mean_) and np.array_equal(sc.scale_, b.scaler.scale_)
    before = b.pt.p_spline.bsplines_[0].t.copy()
    b.raw(Xte)
    assert np.array_equal(before, b.pt.p_spline.bsplines_[0].t)
    assert b.design_record['scaler_n_samples_seen'] == 4000 and b.design_record['columns'] == 48


def test_logit_imputer_and_scaler_are_train_fitted():
    X, y, g = _synthetic()
    names = [f'x{i}' for i in range(X.shape[1])]
    tr = np.arange(len(y)) < len(y) // 2
    b = Q.LinearQBase('logit', 'logit_C1', names).fit(X[tr], y[tr], Q.weights(g[tr]))
    assert np.array_equal(b.imputer.statistics_, np.median(X[tr], axis=0))
    assert b.scaler.n_samples_seen_ == tr.sum()
    assert np.allclose(b.scaler.mean_, X[tr].mean(axis=0), rtol=0, atol=1e-12)
    assert b.eligible and b.attempts[0]['converged'] and len(b.attempts) == 1


def test_logistic_retry_only_after_nonconvergence_and_first_attempt_retained(monkeypatch):
    X, y, g = _synthetic(p=8)
    monkeypatch.setattr(Q, 'LOGIT_MAX_ITER', 2)
    model, attempts, eligible = Q.fit_logistic_policy(X, y, Q.weights(g), 1.0)
    assert len(attempts) == 2 and not attempts[0]['converged'] and attempts[1]['max_iter'] == Q.LOGIT_RETRY_MAX_ITER
    assert len(attempts[0]['retained_first_attempt_coef']) == 8 and attempts[1]['tol'] == attempts[0]['tol'] == 1e-8
    assert eligible and attempts[1]['converged']


# ------------------------------------------------------------------------------------------ weights / allocations
def test_equal_match_weights_are_recomputed_inside_each_subset():
    _, _, g = _synthetic()
    cal = np.isin(g, np.unique(g)[:150])
    sel = ~cal
    for m in (cal, sel):
        w = Q.weights(g[m])
        rec = Q.weight_record(g[m], w)
        assert abs(w.mean() - 1) < 1e-12
        assert abs(rec['per_match_total_min'] - rec['per_match_total_max']) < 1e-12
    union = Q.weights(g)
    assert not np.allclose(union[cal], Q.weights(g[cal]))


def test_stop_allocation_is_match_level_disjoint_and_hash_defined():
    _, _, g = _synthetic(n_matches=3000)
    s = Q.stop_mask(g)
    for m in np.unique(g)[:200]:
        rows = s[g == m]
        assert rows.all() or not rows.any()
        assert bool(rows[0]) == (int(__import__('hashlib').sha256(('iq20260915_stop:' + m).encode()).hexdigest()[:8], 16) % 10 == 0)
    assert not set(g[s]) & set(g[~s])
    assert 0.07 < len(set(g[s])) / len(set(g)) < 0.13


def test_train_fixture_counts_oof_provenance_and_shared_stop_allocation(tmp_path):
    F, Lb, Co, checks = Q.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test TRAIN-only fixture')
    assert all(checks.values())
    valid = Lb['valid_h90'] == 1
    for coh in Q.COHORTS:
        assert int((valid & (Co['cohort'] == Q.COHORT_CODE[coh])).sum()) == Q.EXPECTED_COUNTS['TRAIN'][coh]
    vman = C.read_json(Q.FC / 'v_models_manifest.json')
    g = F['match'].astype(str)
    prov = Q.oof_provenance(g, F['sub_role'], Lb['adapter_id'], Lb['adapter_sha256'], vman)
    assert prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']
    lman = C.read_json(Q.FC / 'labels' / 'labels_trainval_manifest.json')['summaries']['MAIN_TRAIN']['oof_self_outcome_exclusion']
    assert lman['rows_whose_adapter_fit_includes_match'] == 0
    assert all(v['reconstructed_fit_set_sha_equals_adapter'] for v in lman['adapter_membership'].values())
    T = valid & (Co['cohort'] == 1)
    N = valid & (Co['cohort'] == 0)
    sT, sN = Q.stop_mask(g[T]), Q.stop_mask(g[N])
    flagT = dict(zip(g[T].tolist(), sT.tolist()))
    flagN = dict(zip(g[N].tolist(), sN.tolist()))
    shared = set(flagT) & set(flagN)
    assert len(shared) > 1000 and all(flagT[m] == flagN[m] for m in shared)
    assert 0.08 < np.mean(list(flagT.values())) < 0.12
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_oof_provenance_detects_a_foreign_adapter():
    g = np.array(['KR_1', 'KR_1', 'KR_2', 'KR_3'])
    folds = np.array([C.train_fold(m) for m in g])
    vman = dict(oof_sha256={f'fold{k}': f'h{k}' for k in range(5)})
    aid = np.char.add('oof_fold', folds.astype(str))
    sha = np.array([f'h{k}' for k in folds])
    sr = np.char.add('fold', folds.astype(str))
    ok = Q.oof_provenance(g, sr, aid, sha, vman)
    assert ok['adapter_id_equals_own_heldout_fold'] == ok['adapter_sha_equals_manifest_oof_hash'] == 4
    aid_bad = aid.copy()
    aid_bad[2] = f'oof_fold{(folds[2] + 1) % 5}'
    assert Q.oof_provenance(g, sr, aid_bad, sha, vman)['adapter_id_equals_own_heldout_fold'] == 3


# ------------------------------------------------------------------------------------------ LightGBM stopping / refit / ensemble
def _small_lgbm(monkeypatch):
    fixed = dict(Q.LGBM_FIXED, n_estimators=300, n_jobs=1)
    return fixed


def test_lgbm_stops_on_weighted_brier_and_refits_all_rows_from_scratch(monkeypatch):
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    X, y, g = _synthetic(n_matches=1500, p=5, seed=3)
    fixed = _small_lgbm(monkeypatch)
    b = Q.LgbmQBase('lgbm_L15_M50', [f'x{i}' for i in range(5)], seeds=(7, 42), fixed=fixed).fit(X, y, g)
    rec = b.stop_record
    assert rec['fit90_stop10_match_overlap'] == 0 and rec['full']['rows'] == len(y)
    assert rec['fit90']['rows'] + rec['stop10']['rows'] == len(y)
    for j, (sd, r) in enumerate(rec['seeds'].items()):
        assert r['evaluator_metrics_seen'] == ['weighted_brier']
        assert r['best_iteration'] == r['first_argmin_iteration'] == int(np.argmin(r['stop_curve'])) + 1
        assert abs(r['stop_brier_recomputed_from_predict'] - r['best_stop_weighted_brier']) < 1e-9
        assert r['early_stopped'] == (r['rounds_evaluated'] == r['best_iteration'] + Q.LGBM_PATIENCE)
        assert r['final_n_estimators'] == r['final_num_trees'] == r['best_iteration']
        imp = SimpleImputer(strategy='median').fit(X)
        manual = LGBMClassifier(random_state=int(sd), **dict(fixed, num_leaves=15, min_child_samples=50, n_estimators=r['best_iteration']))
        manual.fit(imp.transform(X), y, sample_weight=Q.weights(g))
        assert np.array_equal(manual.predict_proba(X)[:, 1], b.seed_raw(X)[j])
    assert np.array_equal(b.raw(X), b.seed_raw(X).mean(axis=0))


def test_lgbm_early_stopping_keeps_earlier_iteration_on_ties():
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    X, y, g = _synthetic(n_matches=400, p=4, seed=4)
    rec = {}
    est = LGBMClassifier(random_state=7, **dict(Q.LGBM_FIXED, n_estimators=200, n_jobs=1, num_leaves=7, min_child_samples=20))
    est.fit(X, y, eval_set=[(X.copy(), y.copy())], eval_metric=lambda yt, yp: ('flat', 0.25, False),
            callbacks=[lgb.early_stopping(Q.LGBM_PATIENCE, verbose=False, min_delta=0.0), lgb.record_evaluation(rec)])
    assert est.best_iteration_ == 1 and len(rec['valid_0']['flat']) == 1 + Q.LGBM_PATIENCE
    assert Q.first_argmin_iteration([0.3, 0.2, 0.2, 0.25]) == 2


def test_brier_evaluator_is_match_weighted_and_rejects_foreign_weights():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.4, 0.1])
    w = Q.weights(np.array(['a', 'a', 'a', 'b']))
    name, val, higher = Q.brier_evaluator(w)(y, p, w)
    assert name == 'weighted_brier' and not higher
    assert abs(val - C.brier_direct(y, p, w)) < 1e-15
    with pytest.raises(ValueError):
        Q.brier_evaluator(w)(y, p, np.ones(4))


def test_tree_calibration_uses_raw_seed_mean_not_calibrated_seeds():
    X, y, g = _synthetic(n_matches=1200, p=5, seed=5)
    cal = np.isin(g, np.unique(g)[::2])
    b = Q.LgbmQBase('lgbm_L15_M50', [f'x{i}' for i in range(5)], seeds=(7, 42, 123), fixed=dict(Q.LGBM_FIXED, n_estimators=120, n_jobs=1)).fit(X[~cal], y[~cal], g[~cal])
    raw = b.raw(X[cal])
    cals = K.fit_calibrators(raw, y[cal], Q.weights(g[cal]))
    bundle = Q.make_bundle('lgbm', 'lgbm_L15_M50', b, cals, [f'x{i}' for i in range(5)], [f'x{i}' for i in range(5)], {})
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw, y[cal], sample_weight=Q.weights(g[cal]))
    assert np.array_equal(Q.bundle_predict(bundle, 'isotonic', X[cal]), iso.predict(raw))
    per_seed = np.mean([IsotonicRegression(out_of_bounds='clip').fit(s, y[cal], sample_weight=Q.weights(g[cal])).predict(s) for s in b.seed_raw(X[cal])], axis=0)
    assert not np.array_equal(per_seed, Q.bundle_predict(bundle, 'isotonic', X[cal]))
    assert np.array_equal(Q.bundle_predict(bundle, 'raw', X[cal]), raw)


def test_fit_stage_uses_q_cal_weights_for_calibrators_and_q_select_weights_for_selection():
    src = (ROOT / 'scripts' / 'iq20260915_fit.py').read_text(encoding='utf-8')
    assert "W = {k: Q.weights(g[m]) for k, m in M.items()}" in src
    assert "K.fit_calibrators(raws[cfg]['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])" in src
    assert "C.evaluate(y[m], P[c][k], g[m]" in src and "(met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss'])" in src
    fr = (ROOT / 'scripts' / 'iq20260915_freeze.py').read_text(encoding='utf-8')
    assert "ws = Q.weights(D['g'][ms])" in fr


# ------------------------------------------------------------------------------------------ selection / cells / joins
def test_candidate_registry_and_deterministic_selection_with_ties():
    for fam in Q.FAMILIES:
        names = Q.candidate_names(fam)
        assert len(names) == len(set(names)) == 18 and names == sorted(names)
        assert all(Q.split_candidate(n)[0] in Q.config_names(fam) and Q.split_candidate(n)[1] in Q.CALS for n in names)
    m = {'lgbm_L31_M50__raw': (0.2, 0.6), 'lgbm_L15_M50__sigmoid': (0.2, 0.6), 'lgbm_L15_M50__isotonic': (0.2, 0.6), 'x__raw': (0.1999, 0.9)}
    assert Q.select_rule(m)[0] == 'x__raw'
    m.pop('x__raw')
    chosen, ranking = Q.select_rule(m)
    assert chosen == 'lgbm_L15_M50__isotonic' and ranking == sorted(m)
    m['lgbm_L63_M100__raw'] = (0.2, 0.5999)
    assert Q.select_rule(m)[0] == 'lgbm_L63_M100__raw'


def test_balanced_cells_include_boundaries_and_time_bins_partition():
    p = np.array([0.3999999, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6000001])
    t = np.array([2.0, 9.9999, 10.0, 19.99, 20.0, 30.0, 66.0])
    cells = Q.cell_masks(p, t)
    assert cells['B40'].tolist() == [False, True, True, True, True, True, False]
    assert cells['B45'].tolist() == [False, False, True, True, True, False, False]
    tb = [k for k in cells if k.startswith('time_')]
    assert tb == ['time_0_10', 'time_10_20', 'time_20_30', 'time_30_inf']
    assert np.all(np.sum([cells[k] for k in tb], axis=0) == 1)
    assert cells['time_0_10'].tolist() == [True, True, False, False, False, False, False]
    assert np.array_equal(cells['B40_x_time_20_30'], cells['B40'] & cells['time_20_30'])


def test_key_join_requires_complete_unique_keys():
    g = np.array(['a', 'a', 'b'])
    s = np.array([1, 2, 1])
    ix, how = Q.key_join(g, s, g[::-1], s[::-1])
    assert how == 'exact key map' and np.array_equal(g[::-1][ix], g) and np.array_equal(s[::-1][ix], s)
    with pytest.raises(ValueError):
        Q.key_join(g, s, np.array(['a', 'a', 'c']), s)
    with pytest.raises(ValueError):
        Q.key_join(g, s, np.array(['a', 'a', 'a']), np.array([1, 1, 2]))
    with pytest.raises(ValueError):
        Q.key_join(g, s, g[:2], s[:2])


def test_cell_metrics_report_empty_and_single_class_cells():
    e = Q.cell_metrics(np.array([], dtype=int), np.array([]), np.array([], dtype=str))
    assert e['rows'] == 0 and e['empty'] and e['auc'] is None
    s = Q.cell_metrics(np.array([1, 1, 1]), np.array([0.2, 0.9, 1.0]), np.array(['a', 'b', 'b']))
    assert s['single_class'] and s['auc'] is None and s['sparse_lt30_matches'] and s['exact_prob_1'] == 1
    assert len(s['reliability_fixed10']) == 10 and sum(b['rows'] for b in s['reliability_fixed10']) == 3


def test_paired_bootstrap_point_estimates_equal_direct_metrics_and_draws_are_paired():
    X, y, g = _synthetic(n_matches=200, seed=6)
    rng = np.random.default_rng(0)
    preds = {'a': np.clip(1 / (1 + np.exp(-X[:, 0])), 0, 1), 'b': rng.random(len(y))}
    out = Q.paired_bootstrap(y, preds, g, [('a', 'b', 'test')], reps=50)
    w = Q.weights(g)
    assert abs(out['model_ci']['a']['brier']['estimate'] - C.brier_direct(y, preds['a'], w)) < 1e-12
    assert abs(out['model_ci']['a']['auc']['estimate'] - C.auc_direct(y, preds['a'], w)) < 1e-12
    d = out['pairs'][0]['a_minus_b']['brier']
    assert abs(d['estimate'] - (C.brier_direct(y, preds['a'], w) - C.brier_direct(y, preds['b'], w))) < 1e-12
    assert out['degenerate_single_class_replicates'] == 0 and d['finite_replicates'] == 50
    again = Q.paired_bootstrap(y, preds, g, [('a', 'b', 'test')], reps=50)
    assert again['pairs'][0]['a_minus_b']['brier']['ci95'] == d['ci95']


# ------------------------------------------------------------------------------------------ bundles / gates / parents
def test_bundle_reload_gives_identical_predictions(tmp_path):
    import joblib
    rng = np.random.default_rng(7)
    names_all = ['blue_kills', 'time_minutes', 'p_pre_V']
    X = np.column_stack([rng.integers(0, 5, 2000), rng.uniform(2, 40, 2000), rng.random(2000)])
    y = (rng.random(2000) < X[:, 2]).astype(int)
    g = np.array([f'KR_{i // 3}' for i in range(2000)])
    base = Q.LinearQBase('pt', 'pt_C0.1', list(Q.PT_INPUTS)).fit(X[:, [2, 1]], y, Q.weights(g))
    raw = base.raw(X[:, [2, 1]])
    b = Q.make_bundle('pt', 'pt_C0.1', base, K.fit_calibrators(raw, y, Q.weights(g)), names_all, ['blue_kills', 'time_minutes', 'p_pre_V'], {})
    assert b['input_columns'] == [2, 1]
    joblib.dump(b, tmp_path / 'b.joblib')
    b2 = joblib.load(tmp_path / 'b.joblib')
    for cal in Q.CALS:
        assert np.array_equal(Q.bundle_predict(b, cal, X, names_all), Q.bundle_predict(b2, cal, X, names_all))
    with pytest.raises(ValueError):
        Q.bundle_predict(b2, 'raw', X, ['a', 'b', 'c'])


def test_sealed_sets_refuse_before_this_run_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, 'OUT', tmp_path / 'run')
    for name in Q.SEALED_SETS:
        with pytest.raises(PermissionError):
            Q.load_parent_set(name, tmp_path, 'contract test gate')
    with pytest.raises(PermissionError):
        Q.log_access(tmp_path, 'legacy', 'parent A_MAIN_TEST_h90_T.npz', 'parent predictions', True)
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    Q.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    rec = json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))
    assert rec['frozen_manifest_exists'] and rec['sealed']


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'LV.', 'C.aggregate_status(')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if f in src], p.name
    assert Q.OUT == Q.ROOT / 'outputs' / 'incremental_q_training_20260915' and Q.SMOKE.parent == Q.OUT
    src = inspect.getsource(Q)
    assert 'OUT / ' in src and 'CR / \'cohorts\'' in src


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = Q.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for rel, sha in before['sha256_full_corpus_read_targets'].items():
        assert C.sha256_file(Q.ROOT / rel) == sha, rel
    for rel, sha in before['sha256_cohort_role'].items():
        if '/cohorts/' in rel or rel.endswith('frozen_manifest.json'):
            assert C.sha256_file(Q.ROOT / rel) == sha, rel
