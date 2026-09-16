"""Synthetic contracts for the objective-channel ablation (no TEST/external data, no real fits)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import fc20260915_common as C  # noqa: E402
import lv20260915_analysis as A  # noqa: E402
import oc20260915_common as K  # noqa: E402


def evidence():
    return json.loads((K.OUT / 'feature_evidence.json').read_text(encoding='utf-8'))


def test_exact_counts_order_and_hashes():
    ev = evidence()
    prim, ret = ev['primary_ordered'], ev['retained_ordered']
    drop = [d['name'] for d in ev['dropped_ordered']]
    assert (len(prim), len(drop), len(ret)) == (361, 176, 185)
    assert ret == [n for n in prim if n not in set(drop)] and drop == [n for n in prim if n not in set(ret)]
    assert C.sha256_json(ret) == ev['retained_sha256'] and C.sha256_json(drop) == ev['dropped_sha256']
    vman = json.loads((K.FC / 'v_models_manifest.json').read_text(encoding='utf-8'))
    assert prim == vman['feature_names']


def test_token_rule_case_insensitive_and_complete():
    assert K.dropped_token('BLUE_BARON_EVER') == ['baron'] and K.dropped_token('x_Dragon_Soul') == ['dragon', 'soul']
    assert K.dropped_token('participant_slot3_elder_death_since_acquisition') == ['elder']
    assert K.dropped_token('red_tower_OUTER_TURRET_x_time') == []
    ev = evidence()
    for n in ev['retained_ordered']:
        assert not any(t in n.lower() for t in K.OBJECTIVE_TOKENS), n
    for d in ev['dropped_ordered']:
        assert d['tokens'] and all(t in d['name'].lower() for t in d['tokens'])


def test_required_retained_and_excluded_columns():
    ret = set(evidence()['retained_ordered'])
    must = ['time_minutes', 'time_minutes_sq', 'unknown_objective_team_count', 'blue_kills', 'red_kills_x_time', 'blue_plates',
            'red_inhibitor_kills_x_time', 'blue_tower_NEXUS_TURRET', 'red_tower_OTHER_x_time']
    must += [f'participant_slot{i}_{f}' for i in range(10) for f in ('totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'hp_pct',
                                                                     'mp_pct', 'alive', 'laneCS_norm', 'jgCS_norm', 'champion_id', 'kills',
                                                                     'deaths', 'death_since_snapshot', 'death_last_30s', 'death_age_minutes')]
    assert set(must) <= ret and len(set(must)) == 159
    team = [n for n in ret if n.startswith(('blue_', 'red_'))]
    slots = [n for n in ret if n.startswith('participant_slot')]
    assert len(slots) == 150 and len(team) == 32 and len(ret) == 2 + 1 + 150 + 32
    assert all(any(k in n for k in ('_kills', '_plates', '_inhibitor_kills', '_tower_')) for n in team)
    assert 'snapshot_age_s' not in ret
    for bad in ('blue_dragons', 'red_soul_AIR_x_time', 'blue_baron_acquired_last_60s', 'participant_slot0_baron_death_since_acquisition',
                'red_horde_age_minutes', 'blue_atakhan_ever_x_time', 'red_herald'):
        assert bad not in ret


def test_feature_split_function_matches_evidence():
    ev = evidence()
    r, d = K.feature_split(ev['primary_ordered'])
    assert r == ev['retained_ordered'] and d == [x['name'] for x in ev['dropped_ordered']]


def test_shared_positive_slope_calibration_preserves_sign():
    rng = np.random.default_rng(3)
    pre = rng.uniform(1e-6, 1 - 1e-6, 20000)
    post = np.clip(pre + rng.normal(0, .05, 20000), 1e-6, 1 - 1e-6)
    for a, b in ((0.1, 0.9), (-0.3, 1.4), (0.0, 1e-3)):
        s = C.PositiveSlopeSigmoid()
        s.a, s.b = a, b
        d_raw, d_cal = post - pre, s.predict(post) - s.predict(pre)
        ok = d_raw != 0
        mism = np.sum(np.sign(d_raw[ok]) != np.sign(d_cal[ok]))
        # only finite precision can collapse tiny differences to exact zero, never reverse direction
        assert np.sum((d_raw[ok] * d_cal[ok]) < 0) == 0
        assert mism == np.sum(d_cal[ok] == 0)


def test_gate_blocks_sealed_access_before_freeze(tmp_path):
    import lv20260915_common as LV
    P = LV.ParentReadOnly(base=tmp_path)
    P._gate('outcome_W', 'MAIN', ['fold0', 'V_CAL', 'V_SELECT'], 'test')
    for set_id, roles in (('MAIN', ['TEST']), ('MAIN', ['Q_SELECT']), ('NA1_16.13', ['EXTERNAL']), ('MAIN', None)):
        with pytest.raises(PermissionError):
            P._gate('outcome_W', set_id, roles, 'test')
    with pytest.raises(PermissionError):
        P.label_file('MAIN_TEST', 'test', keys=['match'])
    assert K.parent_reader().base == K.OUT


def test_b_adapter_guards(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    names = ['time_minutes', 'blue_baron', 'x']
    X = np.random.default_rng(0).random((60, 2))
    base = Pipeline([('model', LogisticRegression())]).fit(X, (X[:, 0] > .5).astype(int))
    with pytest.raises(ValueError):
        K.NoObjWinProbV(names, ['time_minutes', 'blue_baron'], base, C.IdentityCalibrator(), {})
    ad = K.NoObjWinProbV(names, ['time_minutes', 'x'], base, C.IdentityCalibrator(), {})
    p = tmp_path / 'b.joblib'
    joblib.dump(ad, p)
    re = K.load_b_adapter(p, C.sha256_file(p))
    Xs = np.random.default_rng(1).random((4, 3))
    assert np.array_equal(re.predict_matrix(Xs, names, C.STATE_VERSION), ad.predict_matrix(Xs, names, C.STATE_VERSION))
    with pytest.raises(ValueError):
        K.load_b_adapter(p, '0' * 64)


def test_oriented_delta_and_zero_logic():
    d = np.array([.02, -.01, 0.0, np.nan])
    assert A.label_from_delta(d).tolist() == [1, 0, 0, -1]
    blue_only = np.array([True, False, False, False])
    red_only = np.array([False, True, False, False])
    oriented = np.where(blue_only, d, np.where(red_only, -d, np.nan))
    assert oriented[0] == .02 and oriented[1] == .01 and np.isnan(oriented[2])
