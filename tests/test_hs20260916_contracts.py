"""Contract tests for the 2026-09-16 horizon-sensitivity run (synthetic and TRAIN-only fixtures; no VALIDATION/TEST rows)."""
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
import hs20260916_common as HS  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('hs20260916_*.py'))


def _synthetic(n_matches=400, rows_per=(1, 6), p=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(rows_per[0], rows_per[1], size=n_matches)
    g = np.repeat(np.array([f'KR_{i:07d}' for i in range(n_matches)]), counts)
    n = len(g)
    X = rng.normal(size=(n, p))
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, g


def test_registry_equals_parent_frozen_h90_winners_and_horizons_fixed():
    reg, src = HS.h90_registry()
    iq_fz = C.read_json(Q.frozen_path(HS.IQ))
    ta_fz = C.read_json(T.frozen_path(HS.TA))
    for coh in HS.COHORTS:
        for f in HS.IQ_FAMILIES:
            assert reg[coh][f] == iq_fz['family_winners'][f'{f}_{coh}']['config'] == Q.split_candidate(iq_fz['family_winners'][f'{f}_{coh}']['chosen'])[0]
        for f in HS.TA_FAMILIES:
            assert reg[coh][f] == ta_fz['family_winners'][f'{f}_{coh}']['config']
        assert set(reg[coh]) == set(HS.FAMILIES)
    assert HS.HORIZONS == (60, 120) and HS.H90 == 90
    assert src['iq_frozen_manifest_sha256'] == C.sha256_file(Q.frozen_path(HS.IQ))


def test_calibration_only_candidates_and_selection_with_ties():
    names = HS.candidate_names('lgbm_L15_M100')
    assert names == sorted(names) and len(names) == 3 and all(HS.split_candidate(n)[1] in HS.CALS for n in names)
    m = {'x__raw': (0.2, 0.6), 'x__isotonic': (0.2, 0.6), 'x__sigmoid': (0.2, 0.5999)}
    assert HS.select_rule(m)[0] == 'x__sigmoid'
    m['x__sigmoid'] = (0.2, 0.6)
    assert HS.select_rule(m)[0] == 'x__isotonic'


def test_family_inputs_and_dispatch_to_reused_classes():
    schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
    names, ridge = schema['input_names_all'], schema['predictor_sets']['ridge']
    assert [names[i] for i in HS.input_columns(names, 'pt', ridge)] == ['p_pre_V', 'time_minutes']
    assert [names[i] for i in HS.input_columns(names, 'lgbm', ridge)] == ridge and len(ridge) == 352
    X, y, g = _synthetic(n_matches=500, p=5, seed=1)
    w = HS.weights(g)
    cols = [f'x{i}' for i in range(5)]
    base, elig, rec, stop = HS.fit_family('logit', 'logit_C1', cols, X, y, g, w)
    assert isinstance(base, Q.LinearQBase) and elig and stop is None and rec['params']['C'] == 1.0
    base, elig, rec, stop = HS.fit_family('lgbm', 'lgbm_L15_M50', cols, X, y, g, w)
    assert isinstance(base, Q.LgbmQBase) and stop['fit90_stop10_match_overlap'] == 0 and len(base.models) == 3
    base, elig, rec, stop = HS.fit_family('mlp', 'mlp_W128_D0.1', cols, X, y, g, w, device='cpu')
    assert isinstance(base, T.MlpQBase) and rec['arch']['n_params'] == T.param_count(T.build_module('mlp', 5, 128, 0.1))
    with pytest.raises(ValueError):
        HS.fit_family('cnn', 'x', cols, X, y, g, w)


def test_bundle_reload_identity_for_each_family(tmp_path):
    import joblib
    X, y, g = _synthetic(n_matches=400, p=5, seed=2)
    names_all = ['a', 'x0', 'x1', 'x2', 'time_minutes', 'p_pre_V']
    Xall = np.column_stack([np.zeros(len(y)), X[:, :3], np.random.default_rng(3).uniform(2, 40, len(y)), np.random.default_rng(4).random(len(y))])
    ridge = ['x0', 'x1', 'x2', 'time_minutes', 'p_pre_V']
    w = HS.weights(g)
    for fam, cfg in (('pt', 'pt_C1'), ('logit', 'logit_C0.1'), ('lgbm', 'lgbm_L15_M50'), ('resmlp', 'resmlp_W128_D0.3')):
        cols = HS.input_columns(names_all, fam, ridge)
        kw = dict(device='cpu') if fam in HS.TA_FAMILIES else {}
        base, elig, rec, stop = HS.fit_family(fam, cfg, HS.family_inputs(fam, ridge), Xall[:, cols], y, g, w, **kw)
        raw = base.raw(Xall[:, cols])
        b = HS.make_bundle(fam, cfg, 60, base, K.fit_calibrators(raw, y, w), names_all, ridge, {})
        assert b['input_columns'] == cols and b['horizon_s'] == 60 and b['version'] == HS.VERSION
        joblib.dump(b, tmp_path / f'{fam}.joblib')
        b2 = joblib.load(tmp_path / f'{fam}.joblib')
        for cal in HS.CALS:
            assert np.array_equal(HS.bundle_predict(b, cal, Xall, names_all), HS.bundle_predict(b2, cal, Xall, names_all))


def test_train_fixture_counts_per_horizon_and_oof_provenance(tmp_path):
    vman = C.read_json(HS.FC / 'v_models_manifest.json')
    for h in HS.HORIZONS:
        F, Lb, Co, checks = HS.load_parent_set('MAIN_TRAIN', tmp_path, f'contract test h{h}', h)
        assert all(checks.values())
        valid = Lb[f'valid_h{h}'] == 1
        for coh in HS.COHORTS:
            assert int((valid & (Co['cohort'] == HS.COHORT_CODE[coh])).sum()) == HS.EXPECTED_COUNTS['TRAIN'][coh]
        assert set(np.unique(Lb[f'Y_h{h}'][valid]).tolist()) <= {0, 1}
        prov = HS.oof_provenance(F['match'].astype(str), F['sub_role'], Lb['adapter_id'], Lb['adapter_sha256'], vman)
        assert prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash']
    with np.load(HS.FC / 'labels' / 'MAIN_TRAIN_labels.npz', allow_pickle=False) as z:
        y60, y120, y90, v = z['Y_h60'], z['Y_h120'], z['Y_h90'], z['valid_h90'] == 1
    assert not np.array_equal(y60[v], y90[v]) and not np.array_equal(y120[v], y90[v])  # the horizons are different labels
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_legacy_reference_files_exist_per_horizon_with_expected_keys():
    fz_cr = C.read_json(HS.CR / 'frozen_manifest.json')
    for h in HS.HORIZONS:
        for coh in HS.COHORTS:
            chosen = fz_cr['A_specialists'][f'{coh}_h{h}']['chosen']
            p = HS.legacy_path('MAIN_TEST', h, coh)
            assert p.exists()
            with np.load(p, allow_pickle=False) as z:  # member names only; no sealed array is read before this run's freeze
                assert {'match', 's_ms', 'y', 'pooled', f'spec_{chosen}', 'spec_p_pre_spline', 'spec_p_pre_logistic', 'spec_constant', 'specialist_chosen'} <= set(z.files)


def test_sealed_sets_refuse_until_this_run_freezes(tmp_path, monkeypatch):
    assert Q.frozen_path(HS.IQ).exists() and T.frozen_path(HS.TA).exists()
    monkeypatch.setattr(HS, 'OUT', tmp_path / 'run')
    for name in HS.SEALED_SETS:
        with pytest.raises(PermissionError):
            HS.load_parent_set(name, tmp_path, 'contract test gate', 60)
    with pytest.raises(PermissionError):
        HS.log_access(tmp_path, 'legacy', 'A_MAIN_TEST_h60_T.npz', 'parent predictions', True)
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    HS.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    assert json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))['sealed']


def test_fit_stage_uses_q_cal_weights_for_calibrators_and_q_select_for_selection():
    src = (ROOT / 'scripts' / 'hs20260916_fit.py').read_text(encoding='utf-8')
    assert "W = {k: HS.weights(g[m]) for k, m in M.items()}" in src
    assert "K.fit_calibrators(raw['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])" in src
    assert "(met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss'])" in src
    fr = (ROOT / 'scripts' / 'hs20260916_freeze.py').read_text(encoding='utf-8')
    assert "ws = HS.weights(D['g'][ms])" in fr


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(',
                 'Q.Status(', 'Q.log_command(', 'Q.log_failure(', 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(',
                 'T.Status(', 'T.log_command(', 'T.log_failure(', 'T.log_access(', 'T.load_parent_set(', 'T.load_trainval(', 'Q.OUT /', 'T.OUT /')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if f in src], p.name
    assert HS.OUT == HS.ROOT / 'outputs' / 'horizon_sensitivity_20260916' and HS.SMOKE.parent == HS.OUT
    assert 'frozen_path(OUT)' in inspect.getsource(HS.log_access)


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = HS.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for key in ('sha256_full_corpus_read_targets', 'sha256_incremental_q', 'sha256_track_a'):
        for rel, sha in before[key].items():
            assert C.sha256_file(HS.ROOT / rel) == sha, rel
