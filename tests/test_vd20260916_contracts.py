"""Contract tests for the 2026-09-16 V-mechanism decomposition (adapter identity, column map, exact decomposition on VALIDATION rows)."""
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
import vd20260916_common as V  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('vd20260916_*.py'))


@pytest.fixture(scope='module')
def adapter():
    fz = C.read_json(V.FC / 'frozen_manifest.json')
    return C.load_v_adapter(V.V_ADAPTER, fz['v_final_sha256'])


def test_frozen_adapter_is_raw_logistic_pipeline(adapter):
    assert adapter.calibration == 'raw' and C.read_json(V.FC / 'selection_v.json')['chosen'] == 'raw'
    assert [n for n, _ in adapter.base.steps] == ['preprocess', 'model'] and adapter.base.steps[-1][1].__class__.__name__ == 'LogisticRegression'
    assert len(adapter.state_names) == 362 and len(adapter.feature_names) == 361 and 'snapshot_age_s' not in adapter.feature_names


def test_column_map_partitions_transformed_columns_and_flags_time(adapter):
    names, groups, gidx, tmask = V.column_map(adapter)
    b, b0 = V.beta(adapter)
    assert len(names) == len(b) == 1733 and sum(len(v) for v in gidx.values()) == 1733
    assert all(n.startswith('champions__') for n in np.array(names)[gidx['champion_identity']]) and len(gidx['champion_identity']) > 1000
    assert tmask.sum() == 96 + 2 - 2 or tmask.sum() >= 90  # time_minutes, time_minutes_sq and every _x_time column
    assert V.is_time_related('numeric__blue_kills_x_time') and V.is_time_related('numeric__time_minutes_sq') and not V.is_time_related('numeric__blue_kills')
    assert V.column_group('numeric__participant_slot0_totalGold_norm') == 'economy_and_experience' and V.column_group('champions__participant_slot9_champion_id_950.0') == 'champion_identity'


def test_exact_decomposition_on_validation_rows(adapter, tmp_path):
    Lb, Co = V.load_labels('MAIN_VALIDATION', tmp_path, 'contract test')
    valid = Lb['valid_h90'] == 1
    keys = {(m, int(s)): i for i, (m, s) in enumerate(zip(Lb['match'].astype(str).tolist(), Lb['s'].tolist())) if valid[i]}
    b, b0 = V.beta(adapter)
    names, groups, gidx, tmask = V.column_map(adapter)
    found = 0
    for f in V.chunk_files()[370:410]:  # chunks are ordered TRAIN, VALIDATION, TEST
        with np.load(f, allow_pickle=False) as z:
            if str(z['set_id']) != 'MAIN':
                continue
            em, es = z['e_match'].astype(str), z['e_s']
            ix = np.asarray([i for i, (m, s) in enumerate(zip(em.tolist(), es.tolist())) if (m, int(s)) in keys])
            if len(ix) == 0:
                continue
            assert [str(n) for n in z['names']] == adapter.state_names and str(z['state_version']) == adapter.state_version
            Xp, Xe = z['e_X_pre'][ix], z['e_X_post_h90'][ix]
            li = np.asarray([keys[(m, int(s))] for m, s in zip(em[ix].tolist(), es[ix].tolist())])
        assert np.all(Lb['adapter_id'][li] == 'final')
        p_pre = adapter.predict_matrix(Xp, adapter.state_names, adapter.state_version)
        p_post = adapter.predict_matrix(Xe, adapter.state_names, adapter.state_version)
        assert np.array_equal(p_pre, Lb['p_pre'][li]) and np.array_equal(p_post, Lb['p_post_h90'][li])
        dl = V.decision(adapter, Xe) - V.decision(adapter, Xp)
        cont = b[None, :] * (V.transform(adapter, Xe) - V.transform(adapter, Xp))
        assert np.max(np.abs(cont.sum(1) - dl)) < 1e-9
        assert np.max(np.abs(cont[:, gidx['champion_identity']])) == 0.0
        assert np.max(np.abs((V.logit(p_post) - V.logit(p_pre)) - dl)) < 1e-8
        assert np.array_equal((dl > 0).astype(int), Lb['Y_h90'][li].astype(int))
        found += len(ix)
        if found > 500:
            break
    assert found > 500


def test_strata_and_bands():
    assert V.delta_stratum(np.array([0.0, 0.005, 0.0051, 0.02, 0.5])).tolist() == [0, 0, 1, 2, 3]
    assert V.band_index(np.array([0.0, 9.99, 10.0, 29.9, 30.0, 60.0])).tolist() == [0, 0, 1, 2, 3, 3]
    assert V.stratum_name(0) == '0-0.5pp' and V.stratum_name(3) == '>2pp'


def test_sealed_test_refuses_until_this_run_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'OUT', tmp_path / 'run')
    with pytest.raises(PermissionError):
        V.load_labels('MAIN_TEST', tmp_path, 'contract test gate')
    with pytest.raises(PermissionError):
        V.state_access(tmp_path, 'contract test gate', True)
    assert not (tmp_path / 'access_log.jsonl').exists()


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(', 'Q.Status(', 'Q.log_command(',
                 'Q.log_failure(', 'Q.log_access(', 'Q.load_parent_set(', 'T.Status(', 'T.log_command(', 'T.log_failure(', 'T.load_parent_set(', 'Q.OUT /', 'T.OUT /')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if f in src], p.name
    assert V.OUT == V.ROOT / 'outputs' / 'v_mechanism_20260916' and 'frozen_path(OUT)' in inspect.getsource(V.log_access)


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = V.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for rel, sha in before['sha256_full_corpus_read_targets'].items():
        assert C.sha256_file(V.ROOT / rel) == sha, rel
