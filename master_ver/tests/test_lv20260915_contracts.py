"""Synthetic contracts for the label validity study (no TEST/external data, no fitting of real models)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import fc20260915_common as C  # noqa: E402
import engagement_labels_v3_rules as R  # noqa: E402
import lv20260915_analysis as A  # noqa: E402
import lv20260915_common as K  # noqa: E402


def state_names():
    return json.loads((K.OUT / 'feature_lists.json').read_text(encoding='utf-8'))['state_names']


# ------------------------------------------------------------------ features
def test_econ_feature_rule_exact_and_no_forbidden_channels():
    names = state_names()
    f = K.econ_features(names)
    assert len(f) == 72 and f[:2] == ['time_minutes', 'time_minutes_sq']
    assert sum(n.endswith('champion_id') for n in f) == 10
    banned = ('kills', 'deaths', 'alive', 'hp_pct', 'mp_pct', 'baron', 'dragon', 'soul', 'herald', 'horde', 'atakhan',
              'tower', 'inhibitor', 'plates', 'age', 'snapshot', 'x_time', 'death')
    assert not [n for n in f if any(b in n for b in banned)]
    assert set(f) == set(json.loads((K.OUT / 'feature_lists.json').read_text(encoding='utf-8'))['B_econ']['features'])


def test_breg_features_equal_primary_expanded():
    names = state_names()
    fl = json.loads((K.OUT / 'feature_lists.json').read_text(encoding='utf-8'))
    assert K.model_features('B_reg', names) == fl['A_primary']['features'] and 'snapshot_age_s' not in fl['B_reg']['features']
    assert K.MODEL_SPECS['B_reg']['C'] == 0.1 and K.MODEL_SPECS['B_econ']['C'] == 0.01 == C.V_C


def test_future_or_label_tokens_absent_from_comparator_features():
    names = state_names()
    for m in K.MODELS:
        for n in K.model_features(m, names):
            assert not any(t in n for t in ('p_post', 'delta', 'endpoint', 'label', 'winner', 'game_end', 'next_kill',
                                            'reason', 'horizon', 'after_last', 'position', 'xy_'))


# ------------------------------------------------------------------ labels and ties
def test_zero_delta_and_invalid_label_logic():
    d = np.array([0.0, 1e-17, -1e-17, np.nan, 0.3, -0.0])
    assert A.label_from_delta(d).tolist() == [0, 1, 0, -1, 1, 0]


def test_kill_tie_is_separate_category():
    assert A.sign3([2, 0, -1, 0.0]).tolist() == [1, 0, -1, 0]
    cat = A.resource_category([0.0, 0.0, 0.1, -0.2], [60000, 60000, 60000, 60000], [60000, 120000, 120000, 120000])
    assert cat.tolist() == [2, 0, 1, -1]


def test_direction_conflict_ignores_stale_and_ties():
    y = np.array([1, 1, 0, 0, 1])
    ks = np.array([-1, 0, 1, 0, 0])
    gc = np.array([2, -1, 2, -1, 2])
    kc, gcf = A.direction_conflict(y, ks, gc)
    assert kc.tolist() == [True, False, True, False, False]
    assert gcf.tolist() == [False, True, False, False, False]


class FakeAdapter:
    def __init__(self, offset):
        self.offset = offset
        self.calls = 0

    def predict_matrix(self, X, names, version):
        self.calls += 1
        return X[:, 0] + self.offset


def test_generate_labels_uses_one_adapter_for_both_ends():
    X_pre = np.array([[0.2], [0.5], [0.4], [0.1]])
    X_post = {90: np.array([[0.3], [0.5], [0.1], [0.9]])}
    valid = {90: np.array([1, 1, 1, 0])}
    ads = {'f0': FakeAdapter(0.0), 'f1': FakeAdapter(0.25)}
    out = A.generate_labels(np.array(['f0', 'f1', 'f0', 'f1']), ads, ['x'], 'v', X_pre, np.array([1, 1, 1, 1]), X_post, valid)
    assert np.allclose(out['delta'][90][:3], [0.1, 0.0, -0.3])
    assert out['Y'][90].tolist() == [1, 0, 0, -1]
    assert np.isnan(out['delta'][90][3]) and out['adapter_used'].tolist() == ['f0', 'f1', 'f0', 'f1']


def test_generate_labels_refuses_valid_row_without_pre():
    with pytest.raises(ValueError):
        A.generate_labels(np.array(['f0']), {'f0': FakeAdapter(0)}, ['x'], 'v', np.array([[0.1]]), np.array([0]),
                          {90: np.array([[0.2]])}, {90: np.array([1])})


# ------------------------------------------------------------------ agreement / bootstrap
def test_agreement_cell_weights_and_counts():
    g = np.array(['a', 'a', 'a', 'b'])
    yA = np.array([1, 1, 0, 1])
    yB = np.array([1, 0, 0, 0])
    c = A.agreement_cell(yA, yB, np.array([.1, .1, -.1, .2]), np.array([.2, -.1, -.1, -.2]), g)
    assert c['disagreement_rows'] == 2 and c['disagreement_row'] == 0.5
    assert abs(c['disagreement_match_weighted'] - (1 / 3 + 1) / 2) < 1e-12
    assert c['A_pos_B_neg_rows'] == 2 and c['A_neg_B_pos_rows'] == 0 and not c['empty']
    assert A.agreement_cell(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))['empty']


def test_bootstrap_matches_naive_recomputation():
    rng = np.random.default_rng(1)
    g = np.repeat([f'm{i}' for i in range(40)], rng.integers(1, 6, 40))
    ind = rng.random(len(g)) < .3
    fast = A.paired_match_bootstrap(g, {'d': ind}, reps=25, seed=7)
    naive = A.naive_bootstrap_replicates(g, ind, reps=25, seed=7)
    first = fast['_draws_first5']['d']
    for r in range(5):
        assert abs(first['row'][r] - naive[r][0]) < 1e-12
        assert abs(first['match_weighted'][r] - naive[r][1]) < 1e-12


# ------------------------------------------------------------------ bins
def test_bins_boundaries():
    assert A.bin_abs_delta([0, .005, .0050001, .01, .02, .0200001, -.007]).tolist() == \
        ['[0,.005]', '[0,.005]', '(.005,.01]', '(.005,.01]', '(.01,.02]', '>.02', '(.005,.01]']
    assert A.bin_p_pre([0, .2, .5999, .6, .8, 1.0]).tolist() == ['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]', '[.8,1]']
    assert A.bin_start_minutes([119999, 120000, 600000, 1200000, 1800000]).tolist() == ['<2', '[2,10)', '[10,20)', '[20,30)', '>=30']
    assert A.bin_age_s([0, 14.9, 15, 59.99, 60]).tolist() == ['[0,15)', '[0,15)', '[15,30)', '[45,60)', '>=60']


# ------------------------------------------------------------------ event intervals
def synthetic_events():
    return [dict(type='CHAMPION_KILL', timestamp=1000, killerId=1, victimId=6),
            dict(type='CHAMPION_KILL', timestamp=2000, killerId=7, victimId=2),
            dict(type='CHAMPION_KILL', timestamp=2000, killerId=0, victimId=3),
            dict(type='ELITE_MONSTER_KILL', timestamp=3000, killerTeamId=100, monsterType='DRAGON', monsterSubType='FIRE_DRAGON'),
            dict(type='ELITE_MONSTER_KILL', timestamp=3500, killerTeamId=300, monsterType='HORDE'),
            dict(type='DRAGON_SOUL_GIVEN', timestamp=4000, teamId=0, name='Mountain'),
            dict(type='BUILDING_KILL', timestamp=5000, teamId=200, buildingType='TOWER_BUILDING', towerType='OUTER_TURRET'),
            dict(type='ELITE_MONSTER_KILL', timestamp=6000, killerTeamId=200, monsterType='DRAGON', monsterSubType='ELDER_DRAGON')]


def test_raw_recount_equals_stored_rule_counter_and_boundaries():
    ev = synthetic_events()
    tm = {p: 100 if p <= 5 else 200 for p in range(1, 11)}
    rows = A.raw_event_rows(ev)
    cat = sorted(((int(e['timestamp']),) + tuple(R.categorize(e)) for e in ev if R.categorize(e)[0]), key=lambda x: x[0])
    for lo, hi in ((999, 2000), (1000, 2000), (2000, 6000), (0, 5999), (-1, 10 ** 9)):
        mine = A.raw_interval_counts(rows, tm, lo, hi)
        ref = R.count_events(cat, lo, hi)
        for k in R.COUNT_KEYS:
            assert mine.get(k, 0) == ref.get(k, 0), (lo, hi, k)
    c = A.raw_interval_counts(rows, tm, 1000, 2000)   # lo exclusive, hi inclusive
    assert c.get('champion_kill') == 2 and c.get('credited_kill_red') == 1 and 'credited_kill_blue' not in c
    c = A.raw_interval_counts(rows, tm, 999, 1000)
    assert c.get('credited_kill_blue') == 1
    c = A.raw_interval_counts(rows, tm, 3000, 4000)
    assert c.get('horde') == 1 and 'horde_blue' not in c and 'horde_red' not in c and c.get('soul_teamid0_unassigned') == 1


def test_observed_from_states_team_sums():
    names = state_names()
    ix = {n: i for i, n in enumerate(names)}
    Xp = np.zeros((1, len(names)))
    Xq = np.zeros((1, len(names)))
    Xq[0, ix['blue_kills']] = 3
    Xq[0, ix['red_kills']] = 1
    for i in range(10):
        Xp[0, ix[f'participant_slot{i}_totalGold_norm']] = .1
        Xq[0, ix[f'participant_slot{i}_totalGold_norm']] = .2 if i < 5 else .1
    Xq[0, ix['red_dragon_FIRE']] = 1
    o = A.observed_from_states(names, Xp, Xq)
    assert o['kill_diff'][0] == 2 and abs(o['gold_diff_change_norm'][0] - .5) < 1e-12
    assert o['state_dragon_FIRE_red'][0] == 1 and o['state_dragon_FIRE_blue'][0] == 0


# ------------------------------------------------------------------ packet
def test_packet_unique_matches_priority_determinism_and_shortage():
    match = np.array(['m1', 'm1', 'm2', 'm3', 'm3', 'm4', 'm5'])
    s = np.array([10, 20, 30, 40, 50, 60, 70])
    s1 = np.array([1, 1, 1, 0, 0, 0, 0], bool)
    s2 = np.array([1, 1, 1, 1, 1, 0, 0], bool)
    s3 = np.array([0, 0, 0, 0, 0, 1, 1], bool)
    strata = [('S1', 2, s1), ('S2', 5, s2), ('S3', 1, s3)]
    c1, r1 = A.select_packet(match, s, strata, 'seedX')
    c2, _ = A.select_packet(match, s, strata, 'seedX')
    assert c1 == c2
    ms = [c['match'] for c in c1]
    assert len(ms) == len(set(ms))
    assert {c['match'] for c in c1 if c['stratum'] == 'S1'} == {'m1', 'm2'}
    assert [c['match'] for c in c1 if c['stratum'] == 'S2'] == ['m3']
    assert r1[1]['shortage'] == 4 and r1[1]['eligible_matches_remaining'] == 1 and r1[1]['eligible_matches'] == 3
    m1 = [c for c in c1 if c['match'] == 'm1'][0]
    assert abs(m1['inclusion_probability_conditional'] - 1.0 / 2) < 1e-12
    assert len([c for c in c1 if c['stratum'] == 'S3']) == 1


def test_reviewer_forbidden_tokens_cover_hidden_fields():
    for tok in ('p_pre', 'delta', 'stratum', 'winner', 'B_reg', 'B_econ', 'KR_', 'NA1_'):
        assert tok in A.REVIEWER_FORBIDDEN_TOKENS


# ------------------------------------------------------------------ access gate
def test_gate_blocks_sealed_access_before_freeze(tmp_path):
    P = K.ParentReadOnly(base=tmp_path)
    P._gate('outcome_W', 'MAIN', ['fold0', 'V_CAL', 'V_SELECT'], 'test')
    for set_id, roles in (('MAIN', ['TEST']), ('MAIN', ['Q_CAL']), ('MAIN', ['Q_SELECT']), ('KR_16.13', ['EXTERNAL']),
                          ('MAIN', None), ('MAIN', ['fold0', 'TEST'])):
        with pytest.raises(PermissionError):
            P._gate('outcome_W', set_id, roles, 'test')
    with pytest.raises(PermissionError):
        P.label_file('MAIN_TEST', 'test', keys=['match'])
    with pytest.raises(PermissionError):
        P.label_file('EXT_NA1_16.13', 'test', keys=['match'])
    (tmp_path / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    P._gate('outcome_W', 'MAIN', ['TEST'], 'test')


def test_alt_adapter_type_and_schema_guards(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    names = ['time_minutes', 'snapshot_age_s', 'x']
    X = np.random.default_rng(0).random((50, 2))
    y = (X[:, 0] > .5).astype(int)
    base = Pipeline([('model', LogisticRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        K.AltWinProbV('B_reg', names, ['time_minutes', 'snapshot_age_s'], base, C.IdentityCalibrator(), {})
    ad = K.AltWinProbV('B_econ', names, ['time_minutes', 'x'], base, C.IdentityCalibrator(), {})
    p = tmp_path / 'a.joblib'
    joblib.dump(ad, p)
    sha = C.sha256_file(p)
    re = K.load_alt_adapter(p, sha, 'B_econ')
    Xs = np.random.default_rng(1).random((5, 3))
    assert np.array_equal(re.predict_matrix(Xs, names, C.STATE_VERSION), ad.predict_matrix(Xs, names, C.STATE_VERSION))
    with pytest.raises(TypeError):
        K.load_alt_adapter(p, sha, 'B_reg')
    with pytest.raises(ValueError):
        re.predict_matrix(Xs, names[::-1], C.STATE_VERSION)
