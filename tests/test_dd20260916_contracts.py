"""Contract tests for the 2026-09-16 development-only definition run (TRAIN-only fixtures; no VALIDATION/TEST rows)."""
import inspect
import json
import os
import subprocess
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
import dd20260916_common as DD  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('dd20260916_*.py'))
PY = sys.executable


def _run(code, timeout=1800):
    proc = subprocess.run([PY, '-B', '-c', code], cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                          env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'CUDA_VISIBLE_DEVICES': ''})
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_definitions_follow_rounding_rule_and_frozen_preset():
    dev, pooled = C.read_json(DD.DEV_SPEC), C.read_json(DD.POOLED_SPEC)
    assert DD.DEFINITIONS['dev'] == dict(TF2_KILL_CLUSTER_GAP_MS=int(round(dev['gap_s'], 1) * 1000), CLUSTER_MAX_DIAMETER=float(round(dev['diameter_u'])))
    assert DD.DEFINITIONS['frozen'] == dict(TF2_KILL_CLUSTER_GAP_MS=int(round(pooled['gap_s'], 1) * 1000), CLUSTER_MAX_DIAMETER=float(round(pooled['diameter_u'])))
    assert DD.DEFINITIONS['frozen'] == dict(TF2_KILL_CLUSTER_GAP_MS=13700, CLUSTER_MAX_DIAMETER=4264.0) and DD.DEFINITIONS['dev'] == dict(TF2_KILL_CLUSTER_GAP_MS=14000, CLUSTER_MAX_DIAMETER=4285.0)
    assert dev['scope'] == 'patch:15.14' and pooled['scope'] == 'pooled' and dev['validity_radius_u'] == pooled['validity_radius_u'] == 1600.0 and dev['lead_s'] == 15.0
    assert len(DD.CONTRASTS) == 4 and sum(c[2].startswith('PRIMARY') for c in DD.CONTRASTS) == 1 and all(a in DD.NAMED and b in DD.NAMED for a, b, _ in DD.CONTRASTS)


def test_frozen_detection_reproduces_parent_exposures_in_process():
    rows = DD.set_matches('MAIN_TRAIN')[:40]
    par, _ = DD.parent_exposures('MAIN_TRAIN')
    out = DD.detect_chunk(dict(definition='frozen', cache_dir=str(C.CACHE_MAIN), matches=[r[0] for r in rows]))
    assert DD.detector_settings('frozen')['TF2_KILL_CLUSTER_GAP_MS'] == 13700
    assert all(not r['error'] and r['loaded'] for r in out)
    assert sum(len(r['rows']) for r in out) > 50
    for rec in out:
        assert [tuple(x[k] for k in DD.EXPO_FIELDS) for x in rec['rows']] == par.get(rec['match'], []), rec['match']
        assert all(x['cluster_blue'] >= 0 and x['cluster_red'] >= 0 and x['ref_cluster_blue'] in (x['cluster_blue'], -1) for x in rec['rows'])
    with pytest.raises(RuntimeError):
        DD.detector_context('dev')  # one definition per process


def test_dev_detection_applies_constants_and_changes_some_matches():
    code = ('import sys, json; sys.path.insert(0, "scripts"); import dd20260916_common as DD\n'
            'rows = DD.set_matches("MAIN_TRAIN")[:300]; par, _ = DD.parent_exposures("MAIN_TRAIN")\n'
            'out = DD.detect_chunk(dict(definition="dev", cache_dir=str(DD.C.CACHE_MAIN), matches=[r[0] for r in rows]))\n'
            'st = DD.detector_settings("dev"); ch = sum(set(tuple(x[k] for k in DD.EXPO_FIELDS) for x in r["rows"]) != set(par.get(r["match"], [])) for r in out)\n'
            'print(json.dumps(dict(gap=st["TF2_KILL_CLUSTER_GAP_MS"], diam=st["CLUSTER_MAX_DIAMETER"], radius=st["TF2_VALIDITY_RADIUS"], errors=sum(bool(r["error"]) for r in out), changed=ch, n=len(out),\n'
            '                      unknown=sum(1 for r in out for x in r["rows"] if x["cluster_blue"] < 0 or x["cluster_red"] < 0))))')
    r = _run(code)
    assert r['gap'] == 14000 and r['diam'] == 4285.0 and r['radius'] == 1600.0 and r['errors'] == 0 and r['unknown'] == 0 and r['n'] == 300
    assert 0 < r['changed'] < 100


def test_rebuild_fixture_reproduces_parent_rows_in_subprocess(tmp_path):
    out = tmp_path / 'fixture'
    code = ('import sys, json; sys.path.insert(0, "scripts"); import dd20260916_common as DD; import dd20260916_rebuild as RB\n'
            f'st = DD.Status("test_fixture", out=r"{tmp_path}")\n'
            f'res = RB.fixture_check(12, r"{out}", 1, st, r"{tmp_path}")\n'
            'print(json.dumps(res["checks"]))')
    chk = _run(code)
    assert chk['all_equal'] and chk['rows'] == chk['rows_expected'] > 10 and chk['X_input_equal'] and chk['p_pre_equal'] and chk['Y_h90_equal'] and chk['input_names_equal']
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_fit_family_bundle_reload_and_layout_invariance(tmp_path):
    import joblib
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 6, size=300)
    g = np.repeat(np.array([f'KR_{i:07d}' for i in range(300)]), counts)
    n = len(g)
    names = ['a', 'x0', 'x1', 'x2', 'time_minutes', 'p_pre_V']
    X = np.column_stack([np.zeros(n), rng.normal(size=(n, 3)), rng.uniform(2, 40, n), rng.random(n)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * X[:, 1] - 0.8 * X[:, 2])))).astype(int)
    ridge = ['x0', 'x1', 'x2', 'time_minutes', 'p_pre_V']
    w = DD.weights(g)
    for fam, cfg in (('logit', 'logit_C0.1'), ('lgbm', 'lgbm_L15_M50')):
        cols = [names.index(r) for r in ridge]
        base, elig, rec, stop = DD.fit_family(fam, cfg, ridge, np.ascontiguousarray(X[:, cols]), y, g, w)
        raw = base.raw(np.ascontiguousarray(X[:, cols]))
        b = DD.make_bundle(fam, cfg, base, K.fit_calibrators(raw, y, w), names, ridge, {})
        assert b['input_columns'] == cols and b['arm'] == 'base'
        joblib.dump(b, tmp_path / f'{fam}.joblib')
        b2 = joblib.load(tmp_path / f'{fam}.joblib')
        for cal in DD.CALS:
            assert np.array_equal(DD.bundle_predict(b, cal, X, names), DD.bundle_predict(b2, cal, X, names))
            assert np.array_equal(DD.bundle_predict(b2, cal, np.asfortranarray(X), names), DD.calibrate(b, cal, raw))
    m = {'x__raw': (0.2, 0.6), 'x__isotonic': (0.2, 0.6), 'x__sigmoid': (0.2, 0.5999)}
    assert DD.select_rule(m)[0] == 'x__sigmoid'


def test_sealed_sets_refuse_until_this_run_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(DD, 'OUT', tmp_path / 'run')
    for name in DD.SEALED_SETS:
        with pytest.raises(PermissionError):
            DD.load_parent_set(name, tmp_path, 'contract test gate')
        with pytest.raises(PermissionError):
            DD.load_dev_set(name, tmp_path, 'contract test gate')
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    DD.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    assert json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))['sealed']
    F, Lb, Co = DD.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test unsealed')
    assert len(F['match']) == len(Lb['match']) == len(Co['match']) and 'n_min' in Co


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    import re
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(', 'Q.Status(', 'Q.log_command(', 'Q.log_failure(',
                 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(', 'Q.OUT /' + ' \'', 'X.main(', 'fc20260915_extract.main', 'import fc20260915_labels', 'import cr20260915_cohorts')
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if re.search(r'(?<![A-Za-z0-9_])' + re.escape(f), src)], p.name
    assert DD.OUT == DD.ROOT / 'outputs' / 'definition_dev_20260916'
    assert 'frozen_path(OUT)' in inspect.getsource(DD.log_access)
    src = (ROOT / 'scripts' / 'dd20260916_rebuild.py').read_text(encoding='utf-8')
    assert "os.environ['LOL_OUTPUT_ROOT'] = str(DD.OUT / 'runtime')" in src and 'X.extract_chunk(task)' in src


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = DD.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for key in ('sha256_full_corpus_read_targets', 'sha256_incremental_q', 'sha256_parent_exposures'):
        for rel, sha in before[key].items():
            assert C.sha256_file(DD.ROOT / rel) == sha, rel
