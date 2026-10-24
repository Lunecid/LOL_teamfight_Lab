"""Contract tests for the 2026-09-16 Track A MLP run (synthetic and TRAIN-only fixtures; no VALIDATION/TEST rows).

Covers: exact 352-input schema, both architectures (shapes, parameter counts), the weighted minibatch loss fixture,
fit90/stop10 disjointness on the iq allocation, preprocessing fit on fit90 in the stop phase and refit on all TRAIN,
first-minimum tie rule and cap, deterministic training for a fixed seed, raw seed ensemble before calibration,
Q_CAL/Q_SELECT weight separation, deterministic selection with ties, balanced boundaries and time bins, key joins,
own-match OOF provenance on the TRAIN fixture, bundle reload identity (CPU float64), the sealed gate keyed on THIS run's
freeze although the iq freeze exists, scripts writing only under this root, and parent preservation.
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
import ta20260916_common as T  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('ta20260916_*.py'))
DEV = 'cpu'


def _synthetic(n_matches=400, rows_per=(1, 6), p=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(rows_per[0], rows_per[1], size=n_matches)
    g = np.repeat(np.array([f'KR_{i:07d}' for i in range(n_matches)]), counts)
    n = len(g)
    X = rng.normal(size=(n, p))
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, g


# ------------------------------------------------------------------------------------------ inputs / registry
def test_ridge_schema_is_exact_352_ordered_and_forbidden_fields_absent():
    schema = C.read_json(T.FC / 'q_pre_only_schema.json')
    ridge = schema['predictor_sets']['ridge']
    assert len(ridge) == T.EXPECTED_RIDGE_COUNT == 352
    assert C.q_feature_sets(schema['input_names_all'])['ridge'] == ridge
    assert not [n for n in ridge if n.endswith('champion_id') or 'snapshot_age' in n]
    assert not [n for n in ridge if n != C.P_PRE and any(t in n for t in C.Q_FORBIDDEN_TOKENS)]
    names = schema['input_names_all']
    assert [names[i] for i in T.input_columns(names, ridge)] == ridge


def test_candidate_registry_and_deterministic_selection_with_ties():
    for fam in T.FAMILIES:
        cfgs = T.config_names(fam)
        assert len(cfgs) == 6 and [T.config_params(fam, c) for c in cfgs] == [dict(width=w, dropout=d) for w, d in T.GRID]
        names = T.candidate_names(fam)
        assert len(names) == len(set(names)) == 18 and names == sorted(names)
        assert all(T.split_candidate(n)[0] in cfgs and T.split_candidate(n)[1] in T.CALS for n in names)
    m = {'mlp_W256_D0.1__raw': (0.2, 0.6), 'mlp_W128_D0.1__sigmoid': (0.2, 0.6), 'mlp_W128_D0.1__isotonic': (0.2, 0.6), 'x__raw': (0.1999, 0.9)}
    assert T.select_rule(m)[0] == 'x__raw'
    m.pop('x__raw')
    chosen, ranking = T.select_rule(m)
    assert chosen == 'mlp_W128_D0.1__isotonic' and ranking == sorted(m)


# ------------------------------------------------------------------------------------------ architectures / loss
def test_architectures_have_declared_shapes_and_parameter_counts():
    for w, d in T.GRID:
        m = T.build_module('mlp', 352, w, d)
        shapes = T.layer_shapes(m)
        assert shapes['0.weight'] == [w, 352] and shapes['3.weight'] == [w, w] and shapes['6.weight'] == [1, w]
        assert T.param_count(m) == (352 * w + w) + (w * w + w) + (w + 1)
        r = T.build_module('resmlp', 352, w, d)
        rs = T.layer_shapes(r)
        assert rs['proj.weight'] == [w, 352] and rs['blocks.0.lin1.weight'] == [2 * w, w] and rs['blocks.1.lin2.weight'] == [w, 2 * w]
        assert rs['norm.weight'] == [w] and rs['head.weight'] == [1, w]
        block = (2 * w) + (w * 2 * w + 2 * w) + (2 * w * w + w)  # LN + lin1 + lin2
        assert T.param_count(r) == (352 * w + w) + 2 * block + 2 * w + (w + 1)
        dropouts = [x.p for x in r.modules() if x.__class__.__name__ == 'Dropout']
        assert dropouts == [d] * 4
    with pytest.raises(ValueError):
        T.build_module('cnn', 352, 128, 0.1)


def test_weighted_minibatch_loss_equals_population_loss_on_full_batch():
    torch = T.torch_import()
    rng = np.random.default_rng(3)
    z = rng.normal(size=5000)
    y = (rng.random(5000) < 0.5).astype(np.float64)
    g = np.repeat(np.arange(1000), 5)[rng.permutation(5000)]
    w = T.weights(g)
    assert abs(w.mean() - 1) < 1e-12
    full = T.weighted_bce_mean(torch.tensor(z), torch.tensor(y), torch.tensor(w)).item()
    pop = T.population_weighted_bce(z, y, w)
    assert abs(full - pop) < 1e-10
    # a uniform partition into minibatches averages (per-row) to the same population loss
    order = rng.permutation(5000)
    parts = [order[i:i + 512] for i in range(0, 5000, 512)]
    tot = sum(T.weighted_bce_mean(torch.tensor(z[p]), torch.tensor(y[p]), torch.tensor(w[p])).item() * len(p) for p in parts) / 5000
    assert abs(tot - pop) < 1e-10
    # a renormalised batch loss is NOT the protocol loss
    p = parts[0]
    renorm = (torch.tensor(w[p]) * torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(z[p]), torch.tensor(y[p]), reduction='none')).sum() / torch.tensor(w[p]).sum()
    assert abs(renorm.item() - T.weighted_bce_mean(torch.tensor(z[p]), torch.tensor(y[p]), torch.tensor(w[p])).item()) > 1e-6 or abs(w[p].mean() - 1) < 1e-9


# ------------------------------------------------------------------------------------------ allocation / preprocessing / stopping
def test_stop_allocation_reuses_iq_allocation_and_is_match_disjoint():
    _, _, g = _synthetic(n_matches=3000)
    s = T.stop_mask(g)
    assert np.array_equal(s, Q.stop_mask(g)) and T.STOP_TAG == 'iq20260915_stop:'
    assert not set(g[s]) & set(g[~s])
    assert 0.07 < len(set(g[s])) / len(set(g)) < 0.13


def test_stop_phase_preprocessing_on_fit90_and_full_refit_for_best_epoch():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    X, y, g = _synthetic(n_matches=600, p=5, seed=4)
    b = T.MlpQBase('mlp', 'mlp_W128_D0.1', [f'x{i}' for i in range(5)], seeds=(7,), device=DEV, max_epochs=6, patience=2).fit(X, y, g)
    rec = b.stop_record
    assert rec['fit90_stop10_match_overlap'] == 0 and rec['fit90']['rows'] + rec['stop10']['rows'] == len(y) == rec['full']['rows']
    sc = StandardScaler().fit(SimpleImputer(strategy='median').fit(X).transform(X))
    assert np.array_equal(b.scaler.mean_, sc.mean_) and np.array_equal(b.imputer.statistics_, np.median(X, axis=0))
    r = rec['seeds']['7']
    assert r['best_epoch'] == r['first_argmin_epoch'] == int(np.argmin(r['stop_curve'])) + 1
    assert r['refit_epochs'] == r['best_epoch'] and 1 <= r['best_epoch'] <= 6
    assert r['early_stopped'] == (r['epochs_evaluated'] < 6)
    assert b.arch['n_params'] == T.param_count(T.build_module('mlp', 5, 128, 0.1))
    # refit from scratch on all rows for best_epoch epochs reproduces the saved weights exactly
    stop = T.stop_mask(g)
    Zall = b.scaler.transform(b.imputer.transform(X))
    m2, _, _, ran = T.run_training('mlp', 128, 0.1, Zall, y, T.weights(g), 7, DEV, r['best_epoch'])
    assert ran == r['best_epoch']
    s2 = T.state_to_numpy(m2)
    assert all(np.array_equal(s2[k], b.states[0][k]) for k in s2)
    # and differs from a stop-phase model trained on fit90 only
    imp90 = SimpleImputer(strategy='median').fit(X[~stop])
    sc90 = StandardScaler().fit(imp90.transform(X[~stop]))
    m3, _, _, _ = T.run_training('mlp', 128, 0.1, sc90.transform(imp90.transform(X[~stop])), y[~stop], T.weights(g[~stop]), 7, DEV, r['best_epoch'])
    assert not all(np.array_equal(T.state_to_numpy(m3)[k], b.states[0][k]) for k in s2)


def test_first_minimum_tie_rule_and_patience_cap(monkeypatch):
    assert T.first_argmin([0.3, 0.2, 0.2, 0.25]) == 2
    X, y, g = _synthetic(n_matches=300, p=4, seed=5)
    stop = T.stop_mask(g)
    Z = X.astype(np.float64)
    calls = []
    real = T._stop_brier_gpu

    def flat(torch, module, Xs, ys, ws):
        calls.append(1)
        return 0.25
    monkeypatch.setattr(T, '_stop_brier_gpu', flat)
    _, curve, best, ran = T.run_training('mlp', 16, 0.1, Z[~stop], y[~stop], T.weights(g[~stop]), 7, DEV, 50,
                                         stop=(Z[stop], y[stop], T.weights(g[stop])), patience=3)
    assert best == 1 and ran == 4 and curve == [0.25] * 4 and len(calls) == 4
    monkeypatch.setattr(T, '_stop_brier_gpu', real)
    _, curve, best, ran = T.run_training('mlp', 16, 0.1, Z[~stop], y[~stop], T.weights(g[~stop]), 7, DEV, 3,
                                         stop=(Z[stop], y[stop], T.weights(g[stop])), patience=100)
    assert ran == 3 and len(curve) == 3 and best == int(np.argmin(curve)) + 1


def test_training_is_deterministic_for_a_fixed_seed_and_seed_dependent():
    X, y, g = _synthetic(n_matches=300, p=5, seed=6)
    Z = X.astype(np.float64)
    w = T.weights(g)
    a, _, _, _ = T.run_training('resmlp', 32, 0.3, Z, y, w, 7, DEV, 3)
    b, _, _, _ = T.run_training('resmlp', 32, 0.3, Z, y, w, 7, DEV, 3)
    c, _, _, _ = T.run_training('resmlp', 32, 0.3, Z, y, w, 42, DEV, 3)
    sa, sb, sc = T.state_to_numpy(a), T.state_to_numpy(b), T.state_to_numpy(c)
    assert all(np.array_equal(sa[k], sb[k]) for k in sa)
    assert not all(np.array_equal(sa[k], sc[k]) for k in sa)
    p1 = T.predict_cpu64('resmlp', 32, 0.3, sa, Z)
    p2 = T.predict_cpu64('resmlp', 32, 0.3, sb, Z)
    assert np.array_equal(p1, p2) and p1.dtype == np.float64 and np.all((p1 > 0) & (p1 < 1))


def test_seed_ensemble_is_raw_mean_before_calibration():
    X, y, g = _synthetic(n_matches=500, p=5, seed=7)
    cal = np.isin(g, np.unique(g)[::2])
    b = T.MlpQBase('resmlp', 'resmlp_W128_D0.3', [f'x{i}' for i in range(5)], seeds=(7, 42, 123), device=DEV, max_epochs=3, patience=2).fit(X[~cal], y[~cal], g[~cal])
    raw = b.raw(X[cal])
    assert np.array_equal(raw, b.seed_raw(X[cal]).mean(axis=0)) and b.seed_raw(X[cal]).shape == (3, int(cal.sum()))
    cals = K.fit_calibrators(raw, y[cal], T.weights(g[cal]))
    bundle = T.make_bundle('resmlp', 'resmlp_W128_D0.3', b, cals, [f'x{i}' for i in range(5)], [f'x{i}' for i in range(5)], {})
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw, y[cal], sample_weight=T.weights(g[cal]))
    assert np.array_equal(T.bundle_predict(bundle, 'isotonic', X[cal]), iso.predict(raw))
    per_seed = np.mean([IsotonicRegression(out_of_bounds='clip').fit(s, y[cal], sample_weight=T.weights(g[cal])).predict(s) for s in b.seed_raw(X[cal])], axis=0)
    assert not np.array_equal(per_seed, T.bundle_predict(bundle, 'isotonic', X[cal]))
    assert np.array_equal(T.bundle_predict(bundle, 'raw', X[cal]), raw)


def test_bundle_reload_gives_identical_predictions(tmp_path):
    import joblib
    X, y, g = _synthetic(n_matches=400, p=5, seed=8)
    names_all = ['a', 'x0', 'x1', 'x2', 'x3', 'x4', 'p_pre_V']
    Xall = np.column_stack([np.zeros(len(y)), X, np.random.default_rng(1).random(len(y))])
    ridge = ['x0', 'x1', 'x2', 'x3', 'x4', 'p_pre_V']
    cols = T.input_columns(names_all, ridge)
    b = T.MlpQBase('mlp', 'mlp_W128_D0.1', ridge, seeds=(7, 42), device=DEV, max_epochs=2, patience=2).fit(Xall[:, cols], y, g)
    raw = b.raw(Xall[:, cols])
    bundle = T.make_bundle('mlp', 'mlp_W128_D0.1', b, K.fit_calibrators(raw, y, T.weights(g)), names_all, ridge, {})
    assert bundle['input_columns'] == cols and bundle['version'] == T.VERSION
    joblib.dump(bundle, tmp_path / 'b.joblib')
    b2 = joblib.load(tmp_path / 'b.joblib')
    for cal in T.CALS:
        assert np.array_equal(T.bundle_predict(bundle, cal, Xall, names_all), T.bundle_predict(b2, cal, Xall, names_all))
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for st in b2['base'].states for v in st.values())
    with pytest.raises(ValueError):
        T.bundle_predict(b2, 'raw', Xall, ['q'] * 7)


# ------------------------------------------------------------------------------------------ weights / cells / joins
def test_equal_match_weights_are_recomputed_inside_each_subset():
    _, _, g = _synthetic()
    cal = np.isin(g, np.unique(g)[:150])
    for m in (cal, ~cal):
        w = T.weights(g[m])
        rec = T.weight_record(g[m], w)
        assert abs(w.mean() - 1) < 1e-12 and abs(rec['per_match_total_min'] - rec['per_match_total_max']) < 1e-12
    assert not np.allclose(T.weights(g)[cal], T.weights(g[cal]))


def test_fit_stage_uses_q_cal_weights_for_calibrators_and_q_select_weights_for_selection():
    src = (ROOT / 'scripts' / 'ta20260916_fit.py').read_text(encoding='utf-8')
    assert "W = {k: T.weights(g[m]) for k, m in M.items()}" in src
    assert "K.fit_calibrators(raws[cfg]['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])" in src
    assert "C.evaluate(y[m], P[c][k], g[m]" in src and "(met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss'])" in src
    fr = (ROOT / 'scripts' / 'ta20260916_freeze.py').read_text(encoding='utf-8')
    assert "ws = T.weights(D['g'][ms])" in fr


def test_balanced_cells_include_boundaries_and_time_bins_partition():
    p = np.array([0.3999999, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6000001])
    t = np.array([2.0, 9.9999, 10.0, 19.99, 20.0, 30.0, 66.0])
    cells = T.cell_masks(p, t)
    assert cells['B40'].tolist() == [False, True, True, True, True, True, False]
    assert cells['B45'].tolist() == [False, False, True, True, True, False, False]
    tb = [k for k in cells if k.startswith('time_')]
    assert tb == ['time_0_10', 'time_10_20', 'time_20_30', 'time_30_inf'] and np.all(np.sum([cells[k] for k in tb], axis=0) == 1)


def test_key_join_requires_complete_unique_keys():
    g = np.array(['a', 'a', 'b'])
    s = np.array([1, 2, 1])
    ix, how = T.key_join(g, s, g[::-1], s[::-1])
    assert how == 'exact key map' and np.array_equal(g[::-1][ix], g)
    with pytest.raises(ValueError):
        T.key_join(g, s, np.array(['a', 'a', 'c']), s)
    with pytest.raises(ValueError):
        T.key_join(g, s, g[:2], s[:2])


def test_paired_bootstrap_point_estimates_equal_direct_metrics():
    X, y, g = _synthetic(n_matches=200, seed=9)
    preds = {'a': 1 / (1 + np.exp(-X[:, 0])), 'b': np.random.default_rng(0).random(len(y))}
    out = T.paired_bootstrap(y, preds, g, [('a', 'b', 'test')], reps=50)
    w = T.weights(g)
    assert abs(out['model_ci']['a']['brier']['estimate'] - C.brier_direct(y, preds['a'], w)) < 1e-12
    assert abs(out['pairs'][0]['a_minus_b']['brier']['estimate'] - (C.brier_direct(y, preds['a'], w) - C.brier_direct(y, preds['b'], w))) < 1e-12
    assert out['pairs'][0]['a_minus_b']['brier']['finite_replicates'] == 50


# ------------------------------------------------------------------------------------------ parents / gates
def test_train_fixture_counts_and_oof_provenance(tmp_path):
    F, Lb, Co, checks = T.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test TRAIN-only fixture')
    assert all(checks.values())
    valid = Lb['valid_h90'] == 1
    for coh in T.COHORTS:
        assert int((valid & (Co['cohort'] == T.COHORT_CODE[coh])).sum()) == T.EXPECTED_COUNTS['TRAIN'][coh]
    vman = C.read_json(T.FC / 'v_models_manifest.json')
    prov = T.oof_provenance(F['match'].astype(str), F['sub_role'], Lb['adapter_id'], Lb['adapter_sha256'], vman)
    assert prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_sealed_sets_refuse_until_this_run_freezes_even_though_iq_is_frozen(tmp_path, monkeypatch):
    assert Q.frozen_path(T.IQ).exists()  # the iq freeze exists on disk ...
    monkeypatch.setattr(T, 'OUT', tmp_path / 'run')
    for name in T.SEALED_SETS:  # ... and must not open sealed sets for this run
        with pytest.raises(PermissionError):
            T.load_parent_set(name, tmp_path, 'contract test gate')
    with pytest.raises(PermissionError):
        T.log_access(tmp_path, 'iq winners', 'iq MAIN_TEST_h90_T.npz', 'parent predictions', True)
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    T.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    rec = json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))
    assert rec['frozen_manifest_exists'] and rec['sealed']


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(',
                 'Q.Status(', 'Q.log_command(', 'Q.log_failure(', 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(', 'Q.aggregate_status(',
                 'Q.OUT /', 'Q.SMOKE')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if f in src], p.name
    assert T.OUT == T.ROOT / 'outputs' / 'track_a_mlp_20260916' and T.SMOKE.parent == T.OUT
    src = inspect.getsource(T.log_access)
    assert 'frozen_path(OUT)' in src


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = T.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for rel, sha in before['sha256_full_corpus_read_targets'].items():
        assert C.sha256_file(T.ROOT / rel) == sha, rel
    for rel, sha in before['sha256_incremental_q'].items():
        assert C.sha256_file(T.ROOT / rel) == sha, rel
