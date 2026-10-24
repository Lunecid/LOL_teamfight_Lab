"""Contract tests for scripts/run_q_v3_baselines.py helpers (synthetic data only)."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_q_v3_baselines as q  # noqa: E402  (no side effects at import)


def test_weights_equal_total_per_match_and_mean_one():
    g = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    w = q.weights(g)
    assert math.isclose(w.mean(), 1.0)
    totals = {m: w[g == m].sum() for m in 'abc'}
    assert np.allclose(list(totals.values()), totals['a'])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_direct_metrics_match_sklearn_with_ties_and_weights(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400), 1)  # many ties
    p[:5] = 0.0; p[5:10] = 1.0
    w = rng.random(400) + 0.1
    assert abs(q.auc_direct(y, p, w) - roc_auc_score(y, p, sample_weight=w)) < 1e-12
    assert abs(q.brier_direct(y, p, w) - brier_score_loss(y, p, sample_weight=w)) < 1e-12
    assert abs(q.logloss_direct(y, p, w) - log_loss(y, p, labels=[0, 1], sample_weight=w)) < 1e-10


def test_auc_direct_single_class_is_none():
    assert q.auc_direct(np.ones(5, int), np.linspace(0, 1, 5), np.ones(5)) is None


def test_group_exact_shapley_linear_equals_grouped_linear_shap():
    rng = np.random.default_rng(3)
    d = 7; beta = rng.normal(size=d); b0 = 0.3
    f = lambda Z: b0 + Z @ beta
    Bg = rng.normal(size=(20, d)); x = rng.normal(size=d)
    groups = [[0, 1], [2], [3, 4, 5], [6]]
    phi, v0, v1 = q.group_exact_shapley(f, x, Bg, groups)
    lin = beta * (x - Bg.mean(0))
    assert np.allclose(phi, [lin[g].sum() for g in groups], atol=1e-12)
    assert abs(phi.sum() + v0 - f(x[None])[0]) < 1e-12
    assert abs(v1 - f(x[None])[0]) < 1e-12


def test_group_exact_shapley_nonlinear_additivity_and_symmetry():
    rng = np.random.default_rng(4)
    f = lambda Z: np.tanh(Z[:, 0] * Z[:, 1]) + Z[:, 2] ** 2
    Bg = rng.normal(size=(15, 3)); x = np.array([0.7, 0.7, -1.2])
    Bg[:, 1] = Bg[:, 0]  # exchangeable features 0 and 1
    phi, v0, _ = q.group_exact_shapley(f, x, Bg, [[0], [1], [2]])
    assert abs(phi.sum() + v0 - f(x[None])[0]) < 1e-12
    assert abs(phi[0] - phi[1]) < 1e-12


def test_econ_rule_is_old_rule_with_participant_prefix_and_no_snapshot_age():
    old = ['time_minutes', 'snapshot_age_s', 'time_minutes_sq', 'slot0_totalGold_norm', 'slot0_hp_pct', 'slot3_deaths',
           'slot3_death_age_minutes', 'blue_kills', 'blue_kills_x_time', 'red_baron_age_minutes', 'red_baron_ever',
           'blue_elder_acquired_last_60s', 'red_tower_BASE_TURRET', 'unknown_objective_team_count', 'p_pre']
    new = [n.replace('slot', 'participant_slot', 1) if n.startswith('slot') else n for n in old if n != 'snapshot_age_s']
    kept_old = [n for n in old if q.old_econ_rule(n) and n != 'snapshot_age_s']
    kept_new = [n for n in new if q.econ_rule(n)]
    assert kept_new == [n.replace('slot', 'participant_slot', 1) if n.startswith('slot') else n for n in kept_old]
    assert kept_new == ['time_minutes', 'participant_slot0_totalGold_norm', 'participant_slot3_deaths', 'blue_kills',
                        'red_tower_BASE_TURRET']
    assert not q.econ_rule('snapshot_age_s')


def test_split_hash_rule():
    m = np.array(['KR_1', 'KR_2', 'KR_3', 'KR_4'])
    patch = np.array(['15.14', '15.15', '15.15', '15.16'])
    M = q.split_masks(m, patch)
    for i in (1, 2):
        half = int(hashlib.sha256(('calibration17:' + m[i]).encode()).hexdigest()[:8], 16) % 2
        assert M['calibrate'][i] == (half == 0) and M['select'][i] == (half == 1)
    assert M['train'].tolist() == [True, False, False, False] and M['test'].tolist() == [False, False, False, True]


def test_citl_recovers_shift():
    rng = np.random.default_rng(5)
    lo = rng.normal(size=20000); p_true = 1 / (1 + np.exp(-lo))
    y = (rng.random(20000) < p_true).astype(int)
    p_shift = 1 / (1 + np.exp(-(lo - 0.5)))
    a = q.citl(y, p_shift, np.ones(len(y)))
    assert abs(a - 0.5) < 0.06
    assert q.citl(np.ones(3, int), np.array([.2, .3, .4]), np.ones(3)) is None


def test_grouping_and_units():
    assert q.shap_group('p_pre_A') == 'prior_win_probability'
    assert q.shap_group('participant_slot2_baron_death_since_acquisition') == 'objectives'
    assert q.shap_group('blue_plates_x_time') == 'structures'
    assert 'minutes' in q.feature_unit('blue_baron_age_minutes')
    assert 'query_minutes/30' in q.feature_unit('red_kills_x_time')
    assert q.feature_unit('participant_slot0_hp_pct').startswith('fraction')
