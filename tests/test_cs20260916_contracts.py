"""Contract tests for the 2026-09-16 composite-SHAP / Rashomon run (TRAIN-only and synthetic fixtures; no VALIDATION/TEST rows)."""
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import fc20260915_shap as FS  # noqa: E402
import cs20260916_common as CS  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('cs20260916_*.py'))


@pytest.fixture(scope='module')
def ctx():
    fz = C.read_json(CS.FC / 'frozen_manifest.json')
    ad = C.load_v_adapter(CS.V_ADAPTER, fz['v_final_sha256'])
    schema = C.read_json(CS.FC / 'q_pre_only_schema.json')
    return ad, list(schema['input_names_all']), list(schema['predictor_sets']['ridge']), schema


def test_state_map_partitions_and_reconstructs_train_rows(ctx, tmp_path):
    ad, names, ridge, schema = ctx
    smap = CS.StateMap(ad, names, ridge)
    gidx = smap.base_groups()
    assert len(smap.base_names) == 266 and len(smap.derived) == 95 and sorted(sum(gidx, [])) == list(range(266))
    assert {g: len(ix) for g, ix in zip(CS.BASE_GROUPS, gidx)} == {'time': 1, 'economy_and_experience': 60, 'combat_and_survival': 64, 'objectives': 98, 'structures': 12, 'health_mana_other': 21, 'champion_identity': 10}
    F, Lb, Co = CS.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test reconstruction')
    rows = CS.cohort_rows(F, Lb, Co, 'T')[:300]
    X = F['X_input'][rows]
    S = smap.state_from_base(smap.base_from_inputs(X))
    sidx = [j for j, n in enumerate(smap.state_names) if n != 'snapshot_age_s']
    cols = [smap.input_pos[smap.state_names[j]] for j in sidx]
    assert np.max(np.abs(S[:, sidx] - X[:, cols])) < 1e-12
    # TRAIN rows carry the own held-out-fold OOF p_pre: reconstruct with the fold adapters (the final adapter is for VALIDATION/TEST rows)
    vman = C.read_json(CS.FC / 'v_models_manifest.json')
    folds = np.asarray([C.train_fold(m) for m in F['match'][rows].astype(str).tolist()])
    p = np.full(len(rows), np.nan)
    for k in np.unique(folds):
        adk = C.load_v_adapter(CS.FC / vman['oof_paths'][f'fold{k}'], vman['oof_sha256'][f'fold{k}'])
        p[folds == k] = adk.predict_matrix(S[folds == k], smap.state_names, adk.state_version)
    assert np.max(np.abs(p - X[:, -1])) < 1e-12
    assert np.max(np.abs(ad.predict_matrix(S, smap.state_names, ad.state_version) - X[:, -1])) > 1e-6  # final V differs from OOF V on TRAIN rows
    import joblib
    path, cal, sha, cand = CS.parent_winner('final_q', 'T')
    b = joblib.load(path)
    Xq = smap.q_inputs(S, p)
    assert np.max(np.abs(Q.calibrate(b, cal, b['base'].raw(Xq)) - Q.bundle_predict(b, cal, X, names))) < 1e-9
    f = CS.composite_fn(ad, smap, {'final_q': (b, cal)})
    assert f(smap.base_from_inputs(X[:5]))['final_q'].shape == (5,)
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_multi_model_exact_shapley_equals_reference_and_is_efficient():
    rng = np.random.default_rng(0)
    d = 9
    groups = [[0, 1], [2], [3, 4, 5], [6], [7, 8]]
    Bg = rng.normal(size=(16, d))
    x = rng.normal(size=d)
    w1, w2 = rng.normal(size=d), rng.normal(size=d)

    def f_single(Z):
        return 1 / (1 + np.exp(-(Z @ w1 + 0.5 * Z[:, 0] * Z[:, 3])))

    def f_multi(Z):
        return {'a': f_single(Z), 'b': np.tanh(Z @ w2)}
    phi_ref, v0_ref, vall_ref = FS.group_exact_shapley(f_single, x, Bg, groups)
    res = CS.exact_group_shapley_multi(f_multi, ['a', 'b'], x, Bg, groups)
    assert np.allclose(res['a'][0], phi_ref, atol=1e-12) and abs(res['a'][1] - v0_ref) < 1e-12 and abs(res['a'][2] - vall_ref) < 1e-12
    for m in ('a', 'b'):
        phi, v0, vall = res[m]
        fx = f_multi(x[None, :])[m][0]
        assert abs(phi.sum() + v0 - fx) < 1e-10 and abs(vall - fx) < 1e-12


def test_candidate_inventory_and_saved_prediction_columns(ctx):
    ad, names, ridge, schema = ctx
    for coh in CS.COHORTS:
        cands = CS.candidate_bundles(coh)
        assert len(cands) == 26 and {(s, f) for s, f, _, _ in cands} == set(CS.CANDIDATE_SOURCES)
        for src, fam in CS.CANDIDATE_SOURCES:
            p = CS.candidate_predictions(src, fam, coh)
            assert p.exists()
            with np.load(p, allow_pickle=False) as z:  # member names only
                cfgs = [cfg for s, f, cfg, _ in cands if (s, f) == (src, fam)]
                assert all(f'{cfg}__{cal}' in z.files for cfg in cfgs for cal in CS.CALS) and 'role' in z.files
    assert CS.EPSILONS == (0.0005, 0.001) and CS.PRIMARY_EPS in CS.EPSILONS and len(CS.PERM_SEEDS) == 5


def test_sealed_sets_refuse_until_this_run_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, 'OUT', tmp_path / 'run')
    with pytest.raises(PermissionError):
        CS.load_parent_set('MAIN_TEST', tmp_path, 'contract test gate')
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    CS.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    assert json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))['sealed']


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(', 'Q.Status(', 'Q.log_command(', 'Q.log_failure(',
                 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(', 'T.Status(', 'T.log_access(', 'T.load_parent_set(', 'B.Status(', 'B.log_access(', 'B.load_parent_set(', 'B.log_command(')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if re.search(r'(?<![A-Za-z0-9_])' + re.escape(f), src)], p.name
    assert CS.OUT == CS.ROOT / 'outputs' / 'composite_shap_20260916' and CS.SMOKE.parent == CS.OUT
    assert 'frozen_path(OUT)' in inspect.getsource(CS.log_access)


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = CS.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for key in ('sha256_full_corpus_read_targets', 'sha256_incremental_q', 'sha256_track_a', 'sha256_balanced_shap'):
        for rel, sha in before[key].items():
            assert C.sha256_file(CS.ROOT / rel) == sha, rel
