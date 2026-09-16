"""Contract tests for the 2026-09-16 balanced-state SHAP run (synthetic and TRAIN-only fixtures; no TEST rows)."""
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
import fc20260915_shap as FS  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402
import bsh20260916_common as B  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('bsh20260916_*.py'))


def test_groups_partition_the_352_ridge_inputs_with_protocol_sizes():
    schema = C.read_json(B.FC / 'q_pre_only_schema.json')
    ridge, names = schema['predictor_sets']['ridge'], schema['input_names_all']
    gidx = B.group_index(names, ridge, schema['shap_groups'])
    assert [len(ix) for ix in gidx] == [B.EXPECTED_GROUP_SIZES[g] for g in B.GROUPS] and sum(map(len, gidx)) == 352
    assert ridge[gidx[0][0]] == 'p_pre_V' and all('unknown_objective_team_count' != ridge[i] for i in gidx[4])
    rec = B.group_map_record(ridge, schema['shap_groups'])
    assert rec['mapping_sha256'] == C.sha256_json({n: schema['shap_groups'][n] for n in ridge}) and rec['order'] == list(B.GROUPS)
    with pytest.raises(ValueError):
        B.group_index(names, ridge, {**schema['shap_groups'], ridge[5]: 'objectives'})


def test_explained_models_are_the_parent_frozen_winners_and_final_q_is_the_five_family_winner():
    ta_fz = C.read_json(T.frozen_path(B.TA))
    for coh in B.COHORTS:
        path, cal, sha, chosen = B.parent_winner('final_q', coh)
        assert path.exists() and C.sha256_file(path) == sha and chosen.startswith('lgbm_')
        assert ta_fz['five_family_q_select_ranking'][coh]['chosen'] == f'iq_lgbm:{chosen}'
        path, cal, sha, chosen = B.parent_winner('plain_mlp', coh)
        assert path.exists() and C.sha256_file(path) == sha and chosen.startswith('mlp_')
        p, key = B.parent_eval_predictions('plain_mlp', coh)
        assert p.exists() and key == 'named__mlp_winner'


def test_row_selection_is_deterministic_prefix_stable_and_model_independent():
    rng = np.random.default_rng(0)
    match = np.array([f'KR_{i:07d}' for i in rng.integers(0, 5000, 3000)])
    s = rng.integers(120000, 2000000, 3000)
    rows = np.arange(3000)
    a = B.select_rows(B.TAG_EXPLAIN, rows, match, s, 256, 'T', 'all')
    b = B.select_rows(B.TAG_EXPLAIN, rows, match, s, 256, 'T', 'all')
    c = B.select_rows(B.TAG_EXPLAIN, rows, match, s, 128, 'T', 'all')
    d = B.select_rows(B.TAG_EXPLAIN, rows, match, s, 256, 'T', 'B40')
    assert np.array_equal(a, b) and np.array_equal(a[:128], c) and not np.array_equal(a, d)
    assert len(set(a.tolist())) == 256


def test_exact_group_shapley_is_additive_and_equals_linear_attribution_for_a_linear_model():
    rng = np.random.default_rng(1)
    d = 12
    beta = rng.normal(size=d)
    f = lambda Z: Z @ beta + 0.3  # noqa: E731
    groups = [[0, 1], [2, 3, 4], [5], [6, 7, 8, 9], [10, 11]]
    Bg = rng.normal(size=(40, d))
    x = rng.normal(size=d)
    phi, v0, vall = FS.group_exact_shapley(f, x, Bg, groups)
    assert abs(phi.sum() + v0 - f(x[None, :])[0]) < 1e-10 and abs(vall - f(x[None, :])[0]) < 1e-10
    lin = [float(np.sum(beta[g] * (x[g] - Bg.mean(0)[g]))) for g in groups]
    assert np.allclose(phi, lin, atol=1e-10)


def test_model_fn_matches_bundle_predict_for_lgbm_and_mlp_bundles(tmp_path):
    import joblib
    rng = np.random.default_rng(2)
    n = 900
    g = np.array([f'KR_{i // 3}' for i in range(n)])
    X = rng.normal(size=(n, 5))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))).astype(int)
    names_all = ['x0', 'x1', 'x2', 'x3', 'x4']
    w = B.weights(g)
    lg = Q.LgbmQBase('lgbm_L15_M50', names_all, seeds=(7,), fixed=dict(Q.LGBM_FIXED, n_estimators=60, n_jobs=1)).fit(X, y, g)
    cals = K.fit_calibrators(lg.raw(X), y, w)
    bl = Q.make_bundle('lgbm', 'lgbm_L15_M50', lg, cals, names_all, names_all, {})
    for cal in B.CALS if hasattr(B, 'CALS') else ('raw', 'sigmoid', 'isotonic'):
        assert np.array_equal(B.model_fn(bl, cal)(X), Q.bundle_predict(bl, cal, X, names_all))
    ml = T.MlpQBase('mlp', 'mlp_W128_D0.1', names_all, seeds=(7,), device='cpu', max_epochs=2, patience=2).fit(X, y, g)
    bm = T.make_bundle('mlp', 'mlp_W128_D0.1', ml, K.fit_calibrators(ml.raw(X), y, w), names_all, names_all, {})
    joblib.dump(bm, tmp_path / 'm.joblib')
    bm2 = joblib.load(tmp_path / 'm.joblib')
    assert np.array_equal(B.model_fn(bm2, 'sigmoid')(X), T.bundle_predict(bm, 'sigmoid', X, names_all))
    phi, v0, vall = FS.group_exact_shapley(B.model_fn(bl, 'raw'), X[0], X[1:9], [[0, 1], [2], [3, 4]])
    assert abs(phi.sum() + v0 - B.model_fn(bl, 'raw')(X[:1])[0]) < 1e-10


def test_b40_mask_is_inclusive():
    assert B.b40_mask(np.array([0.3999999, 0.4, 0.5, 0.6, 0.6000001])).tolist() == [False, True, True, True, False]


def test_train_fixture_counts_and_background_rule(tmp_path):
    F, Lb, Co, checks = B.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test TRAIN-only fixture')
    assert all(checks.values())
    for coh in B.COHORTS:
        rows = np.flatnonzero((Lb['valid_h90'] == 1) & (Co['cohort'] == B.COHORT_CODE[coh]))
        assert len(rows) == B.EXPECTED_COUNTS['TRAIN'][coh]
        bg = B.select_rows(B.TAG_BG, rows, F['match'].astype(str), F['s_ms'], B.N_BG, coh)
        assert len(bg) == B.N_BG and len(set(bg.tolist())) == B.N_BG and set(bg.tolist()) <= set(rows.tolist())
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_sealed_sets_refuse_until_this_run_freezes(tmp_path, monkeypatch):
    assert Q.frozen_path(B.IQ).exists() and T.frozen_path(B.TA).exists()
    monkeypatch.setattr(B, 'OUT', tmp_path / 'run')
    with pytest.raises(PermissionError):
        B.load_parent_set('MAIN_TEST', tmp_path, 'contract test gate')
    with pytest.raises(PermissionError):
        B.log_access(tmp_path, 'parent eval', 'iq MAIN_TEST_h90_T.npz', 'parent predictions', True)
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    B.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    assert json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))['sealed']


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(',
                 'Q.Status(', 'Q.log_command(', 'Q.log_failure(', 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(',
                 'T.Status(', 'T.log_command(', 'T.log_failure(', 'T.log_access(', 'T.load_parent_set(', 'T.load_trainval(', 'Q.OUT /', 'T.OUT /', 'HS.')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if f in src], p.name
    assert B.OUT == B.ROOT / 'outputs' / 'balanced_shap_20260916' and B.SMOKE.parent == B.OUT
    assert 'frozen_path(OUT)' in inspect.getsource(B.log_access)


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = B.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for key in ('sha256_full_corpus_read_targets', 'sha256_incremental_q', 'sha256_track_a'):
        for rel, sha in before[key].items():
            assert C.sha256_file(B.ROOT / rel) == sha, rel
