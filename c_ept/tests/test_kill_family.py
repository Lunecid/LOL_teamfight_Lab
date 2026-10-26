"""Tests of ``c_ept.checks.kill_family``: the C5 joint kill-family measurement tool (S4b preparation, task M16; decisions D5 and D10 P8).

What the tool does. For the FIT labels of the units (the kill counts (K_blue, K_red), the first-kill mark m, the size class s, a weight w) it fits, in every cell (size x mark kind:
champion mark 0 .. 49 or execution mark 50, 51), four kill laws by weighted maximum likelihood and reports, for each law, the NLL per unit, the predicted share of units with no kill at all
and the predicted share of ties (K_blue = K_red): independent Poisson (the plain law of every earlier run), the shifted Poisson of D5 (K_d = 1 + N_d, N_d ~ Poisson), the shifted negative
binomial (N_d overdispersed, NB2) and the joint categorical law of the two counts (the empirical table, zero outside the support). ``compare_b1`` fits the kill head of the B1 path
(``baselines_b1.M3Kills``) under the plain and the shifted law on one set of units that both laws accept.

Every number below that is not a property of the law is a hand value: a cell of four units (K_first, K_other) = (1, 0) gives the Poisson rates (1, 0), the NLL 1 per unit, the
no-kill share e^-1 = 0.3679 and the tie share e^-1 (P(first = 0) P(other = 0)); the shifted law gives the rates (0, 0), the NLL 0 and the shares 0. A negative binomial with mean 2 and
alpha = 0.5 (r = 2) gives P(N = 0) = (2/4)^2 = 0.25 and P(N = 1) = 2 (2/4)^2 (2/4) = 0.25. The synthetic worlds draw the counts from the laws themselves with fixed seeds.
"""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from c_ept.checks import kill_family
from c_ept.domain import units
from c_ept.models import baselines_b1, killlaw, nll as nll_scores

SOURCE = Path(kill_family.__file__)
LAWS = ('poisson', 'shifted', 'shifted_nb', 'joint')
SMALL = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 30, 'min_data_per_group': 5, 'cat_smooth': 1.0}


def champion_world(n, seed, lam=0.8, mu=1.2, size=0, alpha=0.0):
    """n units of one size with a champion first-kill mark: K_d = 1 + N_d (N_d Poisson(lam), or negative binomial of mean lam and dispersion alpha), the other team Poisson(mu)."""
    rng = np.random.default_rng(seed)
    mark = rng.integers(0, 50, n)
    d = mark // 25
    if alpha > 0:
        r = 1.0 / alpha
        excess = rng.negative_binomial(r, r / (r + lam), n)
    else:
        excess = rng.poisson(lam, n)
    K = np.empty((n, 2), dtype=np.int64)
    K[np.arange(n), d] = 1 + excess
    K[np.arange(n), 1 - d] = rng.poisson(mu, n)
    return K, mark, np.full(n, size), np.ones(n)


def execution_world(n, seed, mu=1.0, size=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(mu, (n, 2)).astype(np.int64), rng.integers(50, 52, n), np.full(n, size), np.ones(n)


def cell(result, key='T/champion'):
    return result['cells'][key]


def nll_of(result, key='T/champion'):
    return {law: cell(result, key)['laws'][law]['nll'] for law in LAWS}


# =====================================================================================================================================
# the three tests of the plan
# =====================================================================================================================================
def test_shifted_poisson_beats_plain_on_synthetic_shifted_data():
    """Data from the shifted law (K_d = 1 + N_d): the shifted Poisson scores better than the plain Poisson, which gives the first team a zero count with probability e^-mean."""
    result = kill_family.compare(*champion_world(8000, seed=1))
    nll = nll_of(result)
    assert nll['shifted'] < nll['poisson'] - 0.05
    laws = cell(result)['laws']
    assert laws['shifted']['no_kill_share'] == 0.0                              # the first team has a kill: no unit without one
    assert 0.04 < laws['poisson']['no_kill_share'] < 0.06                       # e^-(1.8 + 1.2) = 0.0498 for the plain law: a share of units that the data cannot have
    assert cell(result)['observed']['no_kill_share'] == 0.0
    # the law the data came from: the NLL of the shifted law is the entropy of the two Poisson laws, H(0.8) + H(1.2) nat per unit (the shift moves no mass)
    def entropy(rate):
        k = np.arange(0, 80)
        p = np.exp(-nll_scores.poisson(k, rate, 1.0))
        return float(-(p * np.log(p)).sum())
    assert nll['shifted'] == pytest.approx(entropy(0.8) + entropy(1.2), abs=0.04)


def test_negative_binomial_reduces_to_poisson_without_overdispersion():
    """N_d drawn from a Poisson: the dispersion is estimated at about 0 and the NB score equals the shifted Poisson score; the pmf of alpha = 0 is the Poisson pmf exactly, and the hand
    values of an NB2 with mean 2 and alpha = 0.5 are 0.25 and 0.25."""
    k = np.arange(0, 12)
    assert np.array_equal(kill_family.nb_logpmf(k, 3.0, 0.0), -nll_scores.poisson(k, 3.0, 1.0))
    assert np.allclose(kill_family.nb_logpmf(k, 3.0, 1e-9), -nll_scores.poisson(k, 3.0, 1.0), atol=1e-6)
    assert np.allclose(np.exp(kill_family.nb_logpmf(np.array([0, 1]), 2.0, 0.5)), [0.25, 0.25])
    assert math.isclose(float(np.exp(kill_family.nb_logpmf(np.arange(0, 400), 2.0, 0.5)).sum()), 1.0, rel_tol=1e-12)
    result = kill_family.compare(*champion_world(20000, seed=2))
    nb = cell(result)['laws']['shifted_nb']
    assert nb['params']['alpha'] < 0.065                                       # the sample dispersion of 20,000 Poisson draws has a standard error of about 0.016
    assert nb['nll'] <= cell(result)['laws']['shifted']['nll']                  # the NB nests the Poisson: never worse in sample
    # the likelihood-ratio gain of one extra parameter is chi2(1) / (2 n) per unit: 5e-4 is chi2 = 20 for n = 20,000 (a 1e-5 tail)
    assert cell(result)['laws']['shifted']['nll'] - nb['nll'] < 5e-4
    # a sample that is not overdispersed at all (every excess is 1): the Poisson law is as good as any NB, so alpha is exactly 0 and the two scores are the same number
    under = kill_family.compare(np.tile([[2, 1], [2, 0], [2, 3]], (50, 1)), np.zeros(150, dtype=np.int64), np.zeros(150, dtype=np.int64), np.ones(150))
    assert cell(under)['laws']['shifted_nb']['params']['alpha'] == 0.0
    assert cell(under)['laws']['shifted_nb']['nll'] == cell(under)['laws']['shifted']['nll']


def test_negative_binomial_finds_the_overdispersion_and_beats_the_poisson():
    result = kill_family.compare(*champion_world(20000, seed=3, lam=1.5, alpha=1.0))
    nb = cell(result)['laws']['shifted_nb']
    assert 0.8 < nb['params']['alpha'] < 1.2                                     # truth 1.0
    assert nb['nll'] < cell(result)['laws']['shifted']['nll'] - 0.05
    assert math.isclose(nb['params']['mean_excess'], 1.5, rel_tol=0.05)


def test_joint_law_gives_zero_mass_to_impossible_pairs():
    """The table has mass only on the pairs seen: a pair that the data cannot have (here: no kill for the first team, and any pair beyond the largest count) gets 0, so the predicted
    no-kill share is exactly 0; the plain law gives that pair a positive mass."""
    K, mark, size, w = champion_world(4000, seed=4)
    result = kill_family.compare(K, mark, size, w)
    joint = cell(result)['laws']['joint']
    table = joint['tables']['all']
    assert math.isclose(sum(p for _, _, p in table), 1.0, rel_tol=1e-12)
    assert all(kf >= 1 for kf, _, _ in table)                                    # the pairs of the table: the first team (axis 0) always has a kill
    assert joint['no_kill_share'] == 0.0
    assert joint['axes'] == ['first_team', 'other_team']
    seen = {(int(a), int(b)) for a, b in np.where(mark[:, None] >= 25, K[:, ::-1], K)}
    assert {(kf, ko) for kf, ko, _ in table} == seen                             # no pair outside the support gets a row
    assert math.isclose(joint['tie_share'], sum(p for kf, ko, p in table if kf == ko), rel_tol=1e-12)
    assert cell(result)['laws']['poisson']['no_kill_share'] > 0.04


# =====================================================================================================================================
# hand values
# =====================================================================================================================================
def test_hand_values_of_a_cell_of_four_units():
    K = np.array([[1, 0]] * 4)                                                   # mark 0: the blue team took the first kill and scored 1, the red team 0
    result = kill_family.compare(K, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.ones(4))
    laws = cell(result)['laws']
    e1 = math.exp(-1.0)
    assert laws['poisson']['nll'] == pytest.approx(1.0, abs=1e-12)               # rates (1, 0): the first team scores 1 - 1 log 1 + log 1! = 1, the other 0
    assert laws['poisson']['no_kill_share'] == pytest.approx(e1, abs=1e-12)      # P(first = 0) P(other = 0) = e^-1 x 1
    assert laws['poisson']['tie_share'] == pytest.approx(e1, abs=1e-12)          # sum_k P(first = k) P(other = k) = P(first = 0) x 1
    for law in ('shifted', 'shifted_nb', 'joint'):
        assert laws[law]['nll'] == pytest.approx(0.0, abs=1e-12)
        assert laws[law]['no_kill_share'] == 0.0
        assert laws[law]['tie_share'] == pytest.approx(0.0, abs=1e-12)
    assert laws['poisson']['params'] == {'rate_first_team': 1.0, 'rate_other_team': 0.0}
    assert laws['shifted']['params'] == {'rate_first_team': 0.0, 'rate_other_team': 0.0}
    assert [laws[law]['n_params'] for law in LAWS] == [2, 2, 3, 0]
    assert cell(result)['observed'] == {'no_kill_share': 0.0, 'tie_share': 0.0}


def test_the_red_team_as_the_first_team_is_the_same_cell_turned_round():
    """Mark 25 is a red first kill: the counts are read in the order (first team, other team), so the red-first units are the blue-first units with the columns swapped."""
    K = np.array([[1, 0]] * 4)
    a = kill_family.compare(K, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.ones(4))
    b = kill_family.compare(K[:, ::-1], np.full(4, 25), np.zeros(4, dtype=np.int64), np.ones(4))
    assert cell(a) == cell(b)


def test_the_execution_cell_keeps_the_plain_law_for_every_law():
    K, mark, size, w = execution_world(3000, seed=5)
    result = kill_family.compare(K, mark, size, w)
    laws = cell(result, 'T/execution')['laws']
    assert laws['poisson']['nll'] == laws['shifted']['nll'] == laws['shifted_nb']['nll']   # no first team: nothing to shift
    assert laws['shifted_nb']['params']['alpha'] is None
    assert laws['poisson']['no_kill_share'] == laws['shifted']['no_kill_share']
    assert laws['poisson']['no_kill_share'] == pytest.approx(math.exp(-2.0), abs=0.01)  # e^-(1 + 1): an execution unit may have no champion kill
    assert cell(result, 'T/execution')['n_dropped'] == 0
    assert laws['joint']['axes'] == ['blue_team', 'red_team']


# =====================================================================================================================================
# the same-unit-set rule (M1 review m9)
# =====================================================================================================================================
def test_every_law_is_scored_on_the_units_that_the_shifted_law_accepts():
    """A champion mark whose first team scored no kill is refused by the shifted law (probability 0) and accepted by the plain law: the unit is left out of the four laws alike, counted and
    explained, so that the four NLLs are those of one set of units. The result equals the result without those units, bit for bit."""
    K, mark, size, w = champion_world(3000, seed=6)
    bad_K = np.array([[0, 2], [0, 1], [3, 0], [1, 0], [0, 0]])
    bad_mark = np.array([3, 10, 40, 30, 0])                                       # blue first with 0 kills; blue first with 0; red first with 0; red first with 0; blue first with 0
    # units: (0,2) mark 3 -> refused; (0,1) mark 10 -> refused; (3,0) mark 40 -> red first, K_red = 0 -> refused; (1,0) mark 30 -> red first, K_red = 0 -> refused; (0,0) mark 0 -> refused
    clean = kill_family.compare(K, mark, size, w)
    dirty = kill_family.compare(np.vstack([K, bad_K]), np.concatenate([mark, bad_mark]), np.concatenate([size, np.zeros(5, dtype=np.int64)]), np.concatenate([w, np.ones(5)]))
    c, d = cell(clean), cell(dirty)
    assert (c['n_units'], c['n_dropped']) == (3000, 0)
    assert (d['n_units'], d['n_dropped']) == (3005, 5)
    assert d['laws'] == c['laws'] and d['observed'] == c['observed']
    assert dirty['total']['n_dropped'] == 5 and dirty['total']['n_units'] == 3005
    assert 'first-kill team' in dirty['dropped_reason'] and 'no kill' in dirty['dropped_reason']
    assert 'dropped_reason' in clean and clean['total']['n_dropped'] == 0


def test_a_unit_with_a_kill_for_the_first_team_is_never_dropped():
    K = np.array([[1, 0], [0, 1], [2, 2]])
    mark = np.array([0, 25, 50])                                                  # blue first (1 kill), red first (1 kill), an execution (no first team)
    result = kill_family.compare(K, mark, np.zeros(3, dtype=np.int64), np.ones(3))
    assert result['total']['n_dropped'] == 0
    assert kill_family.refused_units(K, mark).tolist() == [False, False, False]
    assert kill_family.refused_units(np.array([[0, 5], [5, 0], [0, 0]]), np.array([0, 25, 51])).tolist() == [True, True, False]


# =====================================================================================================================================
# weights, cells, totals
# =====================================================================================================================================
def test_a_weight_of_two_is_a_unit_counted_twice_and_a_weight_of_zero_is_no_unit():
    K, mark, size, w = champion_world(500, seed=7)
    once = kill_family.compare(K, mark, size, np.full(len(K), 2.0))
    twice = kill_family.compare(np.vstack([K, K]), np.concatenate([mark, mark]), np.concatenate([size, size]), np.ones(2 * len(K)))
    for law in LAWS:
        assert cell(once)['laws'][law]['nll'] == pytest.approx(cell(twice)['laws'][law]['nll'], abs=1e-12)
        assert cell(once)['laws'][law]['tie_share'] == pytest.approx(cell(twice)['laws'][law]['tie_share'], abs=1e-12)
    assert cell(once)['weight'] == 1000.0
    base = kill_family.compare(K, mark, size, w)
    padded = kill_family.compare(np.vstack([K, [[40, 40]]]), np.concatenate([mark, [3]]), np.concatenate([size, [0]]), np.concatenate([w, [0.0]]))
    assert cell(padded)['n_units'] == 501
    for law in LAWS:
        assert cell(padded)['laws'][law]['nll'] == pytest.approx(cell(base)['laws'][law]['nll'], abs=1e-12)
    assert cell(padded)['laws']['joint']['tables'] == cell(base)['laws']['joint']['tables']     # a pair seen only with weight 0 is not in the support


def test_the_weighted_nll_by_hand():
    """Two kinds of unit with the weights 3 and 1: (K_first, K_other) = (1, 0) and (2, 0). The weighted rate of the first team is (3 x 1 + 1 x 2) / 4 = 1.25; the NLL is the weighted mean
    of the two scores (3 l_1 + l_2) / 4 with l_k = 1.25 - k log 1.25 + log k! (the other team: rate 0, count 0, score 0). The shifted law has lambda = 0.25 and the scores 0.25 - (k - 1)
    log 0.25 + log (k - 1)!. The table has the masses 0.75 and 0.25."""
    K = np.array([[1, 0], [2, 0]])
    result = kill_family.compare(K, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.array([3.0, 1.0]))
    laws = cell(result)['laws']
    score = lambda rate, k: rate - k * math.log(rate) + math.lgamma(k + 1)
    assert laws['poisson']['params']['rate_first_team'] == pytest.approx(1.25, abs=1e-15)
    assert laws['poisson']['nll'] == pytest.approx((3 * score(1.25, 1) + score(1.25, 2)) / 4, abs=1e-12)
    assert laws['shifted']['nll'] == pytest.approx((3 * score(0.25, 0) + score(0.25, 1)) / 4, abs=1e-12)
    assert laws['joint']['nll'] == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)), abs=1e-12)
    assert result['total']['laws']['poisson']['nll'] == laws['poisson']['nll'] and result['total']['weight'] == 4.0


def test_weights_change_the_fit():
    K, mark, size, w = champion_world(2000, seed=8)
    heavy = w.copy()
    heavy[K[:, 0] + K[:, 1] >= 4] = 5.0
    a = kill_family.compare(K, mark, size, w)
    b = kill_family.compare(K, mark, size, heavy)
    assert cell(a)['laws']['poisson']['nll'] != cell(b)['laws']['poisson']['nll']


def test_cells_are_fitted_apart_and_empty_cells_are_listed():
    K0, mark0, _, w0 = champion_world(1500, seed=9, lam=0.5, size=0)
    K1, mark1, _, w1 = champion_world(1500, seed=10, lam=2.0, size=2)
    K, mark = np.vstack([K0, K1]), np.concatenate([mark0, mark1])
    size, w = np.concatenate([np.zeros(1500, dtype=np.int64), np.full(1500, 2)]), np.concatenate([w0, w1])
    both = kill_family.compare(K, mark, size, w)
    only = kill_family.compare(K1, mark1, np.full(1500, 2), w1)
    assert cell(both, 'ASYM/champion') == cell(only, 'ASYM/champion')            # a cell does not read the units of another
    assert cell(both, 'T/champion')['laws']['shifted']['params']['rate_first_team'] < 0.6 < 1.9 < cell(both, 'ASYM/champion')['laws']['shifted']['params']['rate_first_team']
    assert list(both['cells']) == [f'{s}/{k}' for s in units.SIZE_NAMES for k in ('champion', 'execution')]
    empty = cell(both, 'S/champion')
    assert empty['n_units'] == 0 and empty['weight'] == 0.0 and empty['laws'] is None and empty['observed'] is None
    assert both['sizes'] == list(units.SIZE_NAMES) and both['kinds'] == ['champion', 'execution'] and both['laws'] == list(LAWS)


def test_nothing_in_a_cell_when_every_unit_is_dropped_or_the_input_is_empty():
    result = kill_family.compare(np.array([[0, 3], [0, 1]]), np.array([0, 1]), np.zeros(2, dtype=np.int64), np.ones(2))
    assert cell(result)['n_units'] == 2 and cell(result)['n_dropped'] == 2 and cell(result)['laws'] is None
    nothing = kill_family.compare(np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    assert all(c['n_units'] == 0 and c['laws'] is None for c in nothing['cells'].values())
    assert nothing['total']['n_units'] == 0 and nothing['total']['laws'] is None
    zero_weight = kill_family.compare(np.array([[1, 1]]), np.array([0]), np.zeros(1, dtype=np.int64), np.zeros(1))
    assert cell(zero_weight)['laws'] is None and cell(zero_weight)['n_units'] == 1


def test_the_total_is_the_weighted_mean_of_the_cells_and_the_result_is_json():
    K0, mark0, _, _ = champion_world(1000, seed=11, size=0)
    K1, mark1, _, _ = execution_world(500, seed=12, size=1)
    K, mark = np.vstack([K0, K1]), np.concatenate([mark0, mark1])
    size = np.concatenate([np.zeros(1000, dtype=np.int64), np.ones(500, dtype=np.int64)])
    w = np.concatenate([np.full(1000, 1.0), np.full(500, 3.0)])
    result = kill_family.compare(K, mark, size, w)
    for law in LAWS:
        parts = [(c['weight'], c['laws'][law]['nll']) for c in result['cells'].values() if c['laws'] is not None]
        assert result['total']['laws'][law]['nll'] == pytest.approx(sum(a * b for a, b in parts) / sum(a for a, _ in parts), abs=1e-12)
    assert result['total']['weight'] == 1000.0 + 1500.0
    assert json.loads(json.dumps(result)) == result                              # lists, numbers, strings and None only


def test_the_joint_law_can_be_given_the_exact_mark():
    K, mark, size, w = champion_world(6000, seed=13)
    by_role = kill_family.compare(K, mark, size, w)
    by_mark = kill_family.compare(K, mark, size, w, joint_given='mark')
    a, b = cell(by_role)['laws']['joint'], cell(by_mark)['laws']['joint']
    assert set(a['tables']) == {'all'} and len(b['tables']) == 50
    assert b['nll'] <= a['nll'] + 1e-12                                          # more parameters, never worse in sample
    assert b['n_params'] > a['n_params']
    for law in ('poisson', 'shifted', 'shifted_nb'):
        assert cell(by_role)['laws'][law] == cell(by_mark)['laws'][law]
    assert math.isclose(sum(p for rows in b['tables'].values() for _, _, p in rows), len(b['tables']), rel_tol=1e-9)    # one table of mass 1 for each mark


# =====================================================================================================================================
# input validation
# =====================================================================================================================================
@pytest.mark.parametrize('what', ['K_shape', 'K_negative', 'K_float', 'mark_range', 'mark_float', 'size_range', 'size_len', 'w_negative', 'w_nan', 'w_len', 'w_shape', 'joint_given'])
def test_inputs_are_checked(what):
    K, mark, size, w = np.array([[1, 0], [1, 1]]), np.array([0, 25]), np.array([0, 1]), np.ones(2)
    kwargs = {}
    exc = ValueError
    if what == 'K_shape':
        K = np.array([1, 0])
    elif what == 'K_negative':
        K = np.array([[1, -1], [1, 1]])
    elif what == 'K_float':
        K, exc = K.astype(float), TypeError
    elif what == 'mark_range':
        mark = np.array([0, 52])
    elif what == 'mark_float':
        mark, exc = mark.astype(float), TypeError
    elif what == 'size_range':
        size = np.array([0, 4])
    elif what == 'size_len':
        size = np.array([0])
    elif what == 'w_negative':
        w = np.array([1.0, -1.0])
    elif what == 'w_nan':
        w = np.array([1.0, np.nan])
    elif what == 'w_len':
        w = np.ones(3)
    elif what == 'w_shape':
        w = np.ones((2, 1))
    elif what == 'joint_given':
        kwargs = {'joint_given': 'size'}
    with pytest.raises(exc):
        kill_family.compare(K, mark, size, w, **kwargs)


def test_the_inputs_are_not_changed():
    K, mark, size, w = champion_world(300, seed=14)
    copies = [a.copy() for a in (K, mark, size, w)]
    kill_family.compare(K, mark, size, w)
    assert all(np.array_equal(a, b) for a, b in zip((K, mark, size, w), copies))


# =====================================================================================================================================
# the B1-level comparison
# =====================================================================================================================================
def b1_rows(n, seed, n_refused=0, w=None):
    """Rows of the kill head: the design [first-team code d (-1, 0, 1), two noise columns], the counts, the marks. 80 percent champion marks (shifted law), 20 percent executions."""
    rng = np.random.default_rng(seed)
    n_champion = int(0.8 * n)
    K1, m1, _, _ = champion_world(n_champion, seed + 1000)
    K2, m2, _, _ = execution_world(n - n_champion, seed + 2000)
    K, mark = np.vstack([K1, K2]), np.concatenate([m1, m2])
    if n_refused:
        K = np.vstack([K, np.tile([[0, 2]], (n_refused, 1))])
        mark = np.concatenate([mark, np.zeros(n_refused, dtype=np.int64)])
    order = rng.permutation(len(K))
    K, mark = K[order], mark[order]
    X = np.column_stack([killlaw.first_team(mark), rng.normal(size=len(K)), rng.normal(size=len(K))]).astype(np.float32)
    return kill_family.B1Rows(X, K, mark, w)


def test_b1_comparison_scores_both_heads_on_the_units_that_both_laws_accept():
    fit, stop, ev = b1_rows(4000, 1, n_refused=6), b1_rows(1500, 2, n_refused=2), b1_rows(2000, 3, n_refused=3)
    result = kill_family.compare_b1(fit, stop, ev, params=SMALL)
    assert result['n_units'] == {'fit': {'total': 4006, 'dropped': 6}, 'stop': {'total': 1502, 'dropped': 2}, 'evaluate': {'total': 2003, 'dropped': 3}}
    assert 'first-kill team' in result['dropped_reason'] and 'no kill' in result['dropped_reason']
    assert result['laws'] == ['poisson', 'shifted']
    assert result['nll']['shifted'] < result['nll']['poisson']                    # the shifted head knows that the first team has a kill
    assert result['difference'] == pytest.approx(result['nll']['poisson'] - result['nll']['shifted'], abs=1e-15)
    assert result['difference'] > 0.05
    # the plain head by hand on the same kept units, through the B1 path
    kept = lambda rows: ~kill_family.refused_units(rows.K, rows.mark)
    head = baselines_b1.M3Kills('masked', SMALL)
    head.fit(fit.X[kept(fit)], fit.K[kept(fit)], np.ones(int(kept(fit).sum())), valid=(stop.X[kept(stop)], stop.K[kept(stop)], np.ones(int(kept(stop).sum()))))
    by_hand = float(head.nll(ev.X[kept(ev)], ev.K[kept(ev)], np.ones(int(kept(ev).sum()))).mean())
    assert result['nll']['poisson'] == pytest.approx(by_hand, abs=1e-12)
    assert json.loads(json.dumps(result)) == result


def test_b1_comparison_is_weighted_by_the_evaluation_weights():
    fit, stop, ev = b1_rows(3000, 4), b1_rows(1000, 5), b1_rows(1500, 6)
    plain = kill_family.compare_b1(fit, stop, ev, params=SMALL)
    doubled = kill_family.compare_b1(fit, stop, ev._replace(w=np.full(1500, 2.0)), params=SMALL)
    assert doubled['nll'] == pytest.approx(plain['nll'], abs=1e-12)               # a constant weight does not move a mean
    ramp = np.linspace(0.1, 3.0, 1500)
    weighted = kill_family.compare_b1(fit, stop, ev._replace(w=ramp), params=SMALL)
    assert weighted['nll']['poisson'] != plain['nll']['poisson']
    assert plain['n_units']['fit']['dropped'] == 0


def test_b1_comparison_refuses_what_it_cannot_score():
    fit, stop, ev = b1_rows(600, 7), b1_rows(300, 8), b1_rows(300, 9)
    with pytest.raises(ValueError):
        kill_family.compare_b1(fit, stop, ev._replace(X=ev.X[:, :2]), params=SMALL)              # another design width
    with pytest.raises(ValueError):
        kill_family.compare_b1(fit, stop, ev._replace(K=ev.K[:10]), params=SMALL)                # lengths differ
    with pytest.raises(ValueError):
        kill_family.compare_b1(fit, stop, ev._replace(w=-np.ones(300)), params=SMALL)
    all_bad = kill_family.B1Rows(ev.X[:3], np.array([[0, 1]] * 3), np.zeros(3, dtype=np.int64), None)
    with pytest.raises(ValueError, match='no unit'):
        kill_family.compare_b1(fit, stop, all_bad, params=SMALL)


# =====================================================================================================================================
# the module
# =====================================================================================================================================
def test_the_module_uses_the_kill_law_module_and_no_private_names_of_other_packages():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert 'c_ept.models' in imported or 'c_ept.models.killlaw' in imported
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('c_ept'):
            assert not any(a.name.startswith('_') for a in node.names), node.module
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ('killlaw', 'baselines_b1', 'nll_scores', 'units'):
            assert not node.attr.startswith('_'), node.attr
    text = SOURCE.read_text(encoding='utf-8')
    assert 'FIT labels only' in text                                              # the caller's duty is documented
    for forbidden in ('cohort', 'holdout_r3', 'outputs', '.npz', 'torch'):
        assert forbidden not in text.replace('cohort_policy', ''), forbidden


def test_a_synthetic_run_is_deterministic():
    args = champion_world(2000, seed=15, lam=1.2, alpha=0.5)
    assert kill_family.compare(*args) == kill_family.compare(*args)


# =====================================================================================================================================
# fix round (M16 review I1, I2, m1 - m6; author decision D29)
# =====================================================================================================================================


def shared_driver_world(n, seed, b=0.9):
    """Champion units whose two teams are both driven by one covariate x: N_d ~ Poisson(exp(-0.3 + b x)) and the other team Poisson(exp(b x)), conditionally independent given x.
    The design of the rows is [first-team code d, x]. At the label level the counts are overdispersed and dependent (a shared intensity); given x they are neither."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mark = rng.integers(0, 50, n)
    d = mark // 25
    K = np.empty((n, 2), dtype=np.int64)
    K[np.arange(n), d] = 1 + rng.poisson(np.exp(-0.3 + b * x))
    K[np.arange(n), 1 - d] = rng.poisson(np.exp(b * x))
    return kill_family.B1Rows(np.column_stack([d, x]).astype(np.float32), K, mark)


HEAD = {**SMALL, 'num_leaves': 15, 'min_data_in_leaf': 20, 'num_iterations': 120}


def test_the_independent_table_and_the_mutual_information_by_hand():
    """Four units (1,0), (1,1), (2,0), (2,1) of weight 1: the table is the product of its margins, so the mutual information is 0, G = 0, df = 1, p = 1. Four units, two of (1,0) and two of
    (2,1): the table has p = 1/2 on two pairs (NLL log 2) and the margins have 1/2 each (independent-table NLL 2 log 2), so MI = log 2, G = 2 x 4 x log 2 = 5.545, df = (2-1)(2-1) = 1."""
    z = np.zeros(4, dtype=np.int64)
    free = cell(kill_family.compare(np.array([[1, 0], [1, 1], [2, 0], [2, 1]]), z, z, np.ones(4)))['dependence']
    assert free['mutual_information'] == pytest.approx(0.0, abs=1e-12) and free['G'] == pytest.approx(0.0, abs=1e-12)
    assert free['df'] == 1 and free['p_value'] == pytest.approx(1.0) and free['levels'] == [2, 2]
    tied = cell(kill_family.compare(np.array([[1, 0], [1, 0], [2, 1], [2, 1]]), z, z, np.ones(4)))
    dep = tied['dependence']
    assert tied['laws']['joint']['nll'] == pytest.approx(math.log(2), abs=1e-12)
    assert dep['independent_table']['nll'] == pytest.approx(2 * math.log(2), abs=1e-12) and dep['independent_table']['n_params'] == 2
    assert dep['mutual_information'] == pytest.approx(math.log(2), abs=1e-12)
    assert dep['G'] == pytest.approx(2 * 4 * math.log(2), abs=1e-12)
    assert dep['p_value'] == pytest.approx(chi2.sf(2 * 4 * math.log(2), 1), abs=1e-12)


def test_the_mutual_information_isolates_dependence_where_the_joint_nll_always_wins():
    """On counts that are independent the table still beats the independent Poisson laws in sample (Gibbs), but the mutual information is near 0 and G is not significant; a shared
    intensity makes it large and significant."""
    K, mark, size, w = champion_world(8000, seed=21)
    free = cell(kill_family.compare(K, mark, size, w))
    assert free['laws']['joint']['nll'] < free['laws']['shifted']['nll']
    assert 0.0 <= free['dependence']['mutual_information'] < 0.01 and free['dependence']['p_value'] > 1e-3
    shared = shared_driver_world(8000, seed=22)
    c = cell(kill_family.compare(shared.K, shared.mark, np.zeros(8000, dtype=np.int64), np.ones(8000)))
    dep = c['dependence']
    assert dep['mutual_information'] > 0.05 and dep['p_value'] < 1e-10
    # the gap of the table to the shifted laws holds the margins' misfit as well as the dependence: the independent table has the same margins, so MI is the part that is dependence
    assert dep['mutual_information'] <= c['laws']['shifted']['nll'] - c['laws']['joint']['nll'] + 1e-12


def test_the_mutual_information_of_the_mark_tables_is_read_against_the_margins_of_each_mark():
    K, mark, size, w = champion_world(6000, seed=23)
    by_mark = cell(kill_family.compare(K, mark, size, w, joint_given='mark'))
    dep = by_mark['dependence']
    assert dep['mutual_information'] >= 0.0
    assert dep['df'] > cell(kill_family.compare(K, mark, size, w))['dependence']['df']          # one table per mark: more levels, more degrees of freedom


def test_unweighted_count_effective_count_and_aic():
    K = np.array([[1, 0], [2, 0]])
    result = kill_family.compare(K, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.array([3.0, 1.0]))
    c = cell(result)
    assert c['n_kept'] == 2 and c['n_eff'] == pytest.approx(16 / 10, abs=1e-15)               # Kish: (3 + 1)^2 / (9 + 1)
    for law, k in c['laws'].items():
        assert k['aic'] == pytest.approx(2 * c['n_eff'] * k['nll'] + 2 * k['n_params'], abs=1e-12)
    ind = c['dependence']['independent_table']
    assert ind['aic'] == pytest.approx(2 * c['n_eff'] * ind['nll'] + 2 * ind['n_params'], abs=1e-12)
    padded = kill_family.compare(np.vstack([K, [[40, 40], [0, 5]]]), np.array([0, 0, 3, 0]), np.zeros(4, dtype=np.int64), np.array([3.0, 1.0, 0.0, 1.0]))
    assert cell(padded)['n_kept'] == 2 and cell(padded)['n_units'] == 4 and cell(padded)['n_dropped'] == 1         # a zero weight and a refused unit are not kept
    empty = cell(kill_family.compare(K, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.ones(2)), 'S/champion')
    assert empty['n_kept'] == 0 and empty['n_eff'] is None and empty['dependence'] is None and empty['family_test'] is None and empty['support'] is None


def test_the_boundary_likelihood_ratio_test_of_the_negative_binomial():
    """LR = 2 n_eff (NLL_shifted - NLL_nb) with the p value of the mixture 1/2 chi2(0) + 1/2 chi2(1): half of the chi2(1) tail. An underdispersed sample has LR = 0 and p = 1; an
    overdispersed one rejects; an execution cell has no test."""
    over = cell(kill_family.compare(*champion_world(20000, seed=3, lam=1.5, alpha=1.0)))
    test = over['family_test']
    lr = 2 * over['n_eff'] * (over['laws']['shifted']['nll'] - over['laws']['shifted_nb']['nll'])
    assert test['lr'] == pytest.approx(lr, abs=1e-9) and lr > 100
    assert test['p_value'] == pytest.approx(0.5 * chi2.sf(lr, 1), abs=1e-300) and test['p_value'] < 1e-20
    assert test['alpha'] == over['laws']['shifted_nb']['params']['alpha']
    under = cell(kill_family.compare(np.tile([[2, 1], [2, 0], [2, 3]], (50, 1)), np.zeros(150, dtype=np.int64), np.zeros(150, dtype=np.int64), np.ones(150)))
    assert under['family_test']['lr'] == 0.0 and under['family_test']['p_value'] == 1.0
    assert cell(kill_family.compare(*execution_world(500, seed=24)), 'T/execution')['family_test'] is None


def test_the_share_of_the_first_team_with_no_kill_is_the_mass_the_plain_law_misplaces():
    """Hand values of the four-unit cell (rates 1 and 0): the plain law gives the first team no kill with P(K_1 = 0) = e^-1; the three laws that know the support give 0; an execution has
    no first team (None). The no-kill share of the plain law there is P(K_1 = 0) P(K_2 = 0), a different number (m1 of the review)."""
    K = np.array([[1, 0]] * 4)
    laws = cell(kill_family.compare(K, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.ones(4)))['laws']
    assert laws['poisson']['first_team_zero_share'] == pytest.approx(math.exp(-1.0), abs=1e-12)
    assert [laws[law]['first_team_zero_share'] for law in ('shifted', 'shifted_nb', 'joint')] == [0.0, 0.0, 0.0]
    execution = cell(kill_family.compare(*execution_world(300, seed=25)), 'T/execution')['laws']
    assert all(execution[law]['first_team_zero_share'] is None for law in LAWS)
    big = cell(kill_family.compare(*champion_world(5000, seed=26)))['laws']['poisson']
    assert big['first_team_zero_share'] > big['no_kill_share'] > 0                                # e^-r1 > e^-(r1 + r2)


def test_the_support_widens_until_the_tail_is_small_and_never_aborts(monkeypatch):
    """A start width too short for the fitted laws is doubled until the missing mass of every pmf is below 1e-6, and the result is that of the ample width; at the cap the run goes on
    and the cell reports the tail mass it could not reach (before: ValueError, and the other cells were lost)."""
    K, mark, size, w = champion_world(3000, seed=27, lam=5.0, mu=5.0)
    normal = kill_family.compare(K, mark, size, w)
    assert normal['cells']['T/champion']['support']['tail_mass'] < 1e-6
    monkeypatch.setattr(kill_family, 'SUPPORT_MIN', 2)
    monkeypatch.setattr(kill_family, 'SUPPORT_PER_COUNT', 0)
    monkeypatch.setattr(kill_family, 'SUPPORT_SLACK', 0)
    narrow = kill_family.compare(K, mark, size, w)
    assert narrow['cells']['T/champion']['support']['width'] >= 20 and narrow['cells']['T/champion']['support']['tail_mass'] < 1e-6
    for law in LAWS:
        for key in ('nll', 'no_kill_share', 'tie_share'):
            assert narrow['cells']['T/champion']['laws'][law][key] == pytest.approx(normal['cells']['T/champion']['laws'][law][key], abs=1e-6)
    monkeypatch.setattr(kill_family, 'SUPPORT_CAP', 8)
    capped = kill_family.compare(K, mark, size, w)                                         # no ValueError
    assert capped['cells']['T/champion']['support']['width'] == 8 and capped['cells']['T/champion']['support']['tail_mass'] > 1e-3
    assert capped['cells']['T/champion']['laws']['poisson']['nll'] == normal['cells']['T/champion']['laws']['poisson']['nll']      # the NLL does not read the support


def test_nb_logpmf_checks_its_arguments():
    for bad_alpha in (-0.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            kill_family.nb_logpmf(np.array([1]), 1.0, bad_alpha)
    with pytest.raises(ValueError):
        kill_family.nb_logpmf(np.array([-1]), 1.0, 0.5)
    with pytest.raises(ValueError):
        kill_family.nb_logpmf(np.array([1.5]), 1.0, 0.5)
    out = kill_family.nb_logpmf(np.array([0, 1]), 0.0, 0.5)                                   # mean 0: all the mass on 0
    assert out[0] == 0.0 and out[1] == -np.inf


def test_b1_comparison_refuses_weights_it_would_ignore():
    fit, stop, ev = b1_rows(600, 31), b1_rows(300, 32), b1_rows(300, 33)
    with pytest.raises(ValueError, match='fit'):
        kill_family.compare_b1(fit._replace(w=np.ones(len(fit.K))), stop, ev, params=SMALL)
    with pytest.raises(ValueError, match='stop'):
        kill_family.compare_b1(fit, stop._replace(w=np.ones(len(stop.K))), ev, params=SMALL)


def test_b1_comparison_without_an_execution_unit_and_on_the_open_layout():
    """No execution unit among the evaluation rows: the execution part is None. An open head with the layout of the shifted law (its design has the first-team category d): the
    comparison runs on the real open layout and both laws see the same design."""
    layout = baselines_b1.make_layout('M3Kills', 'open', ('f0', 'f1'), 'shifted')
    assert layout.names == ('f0', 'f1', 's', 'd')

    def rows(n, seed):
        rng = np.random.default_rng(seed)
        K, mark, size, _ = champion_world(n, seed, size=0)
        X = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.integers(0, 4, n), killlaw.first_team(mark) + 1]).astype(np.float32)   # d is stored as d + 1
        return kill_family.B1Rows(X, K, mark)
    result = kill_family.compare_b1(rows(3000, 41), rows(1000, 42), rows(1500, 43), kind='open', layout=layout, params=SMALL)
    assert result['nll']['shifted'] < result['nll']['poisson']
    assert result['conditional']['execution'] is None and result['conditional']['champion']['n_units'] == 1500
    with pytest.raises(ValueError):
        kill_family.compare_b1(rows(300, 44)._replace(X=rows(300, 44).X[:, :3]), rows(300, 45), rows(300, 46), kind='open', layout=layout, params=SMALL)


def test_the_conditional_statistics_vanish_where_the_label_level_ones_appear():
    """One covariate drives both teams (conditionally independent given x). At the label level the NB dispersion is large and the two teams' counts are dependent (a shared
    intensity); given the head's means they are neither: alpha_conditional ~ 0 and the residual correlation ~ 0 (author decision D29: these conditional statistics decide the
    D5 family and dependence, the label-level ones are descriptive)."""
    fit, stop, ev = shared_driver_world(8000, 51), shared_driver_world(3000, 52), shared_driver_world(6000, 53)
    n = len(ev.K)
    label = cell(kill_family.compare(ev.K, ev.mark, np.zeros(n, dtype=np.int64), np.ones(n)))
    assert label['laws']['shifted_nb']['params']['alpha'] > 0.2 and label['family_test']['p_value'] < 1e-10
    assert label['dependence']['mutual_information'] > 0.05 and label['dependence']['p_value'] < 1e-10
    result = kill_family.compare_b1(fit, stop, ev, params=HEAD)
    cond = result['conditional']['champion']
    assert cond['n_units'] == n and cond['alpha'] < 0.05 and cond['alpha'] < label['laws']['shifted_nb']['params']['alpha'] / 5
    assert cond['p_value'] > 1e-3 or cond['gain'] < 1e-3
    assert abs(cond['residual_correlation']) < 0.06
    assert cond['nll_nb'] <= cond['nll_poisson'] and cond['gain'] == pytest.approx(cond['nll_poisson'] - cond['nll_nb'], abs=1e-15)
    assert json.loads(json.dumps(result)) == result


def test_the_conditional_statistics_find_a_real_conditional_overdispersion_and_dependence():
    """Given x the other team's count shares a latent factor with N_d (a Gamma frailty of both): the conditional dispersion and the residual correlation are then positive."""
    def frail(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        mark = rng.integers(0, 50, n)
        d = mark // 25
        g = rng.gamma(2.0, 0.5, n)                                                        # mean 1, variance 0.5
        K = np.empty((n, 2), dtype=np.int64)
        K[np.arange(n), d] = 1 + rng.poisson(np.exp(-0.3 + 0.5 * x) * g)
        K[np.arange(n), 1 - d] = rng.poisson(np.exp(0.5 * x) * g)
        return kill_family.B1Rows(np.column_stack([d, x]).astype(np.float32), K, mark)
    result = kill_family.compare_b1(frail(8000, 61), frail(3000, 62), frail(6000, 63), params=HEAD)
    cond = result['conditional']['champion']
    assert 0.3 < cond['alpha'] < 0.9 and cond['p_value'] < 1e-10                           # Var N = lam + 0.5 lam^2 (the frailty variance), up to the head's error
    assert cond['residual_correlation'] > 0.2


def test_the_text_says_which_level_decides():
    text = SOURCE.read_text(encoding='utf-8')
    assert 'D29' in text and 'descriptive' in text and 'S4b run plan' in text
    assert 'mutual information' in text and 'held-out' in text and 'marks 0 .. 24' in text
    assert 'D29' in kill_family.compare_b1.__doc__ and 'production plain open head' in kill_family.compare_b1.__doc__


def test_the_boundary_p_value_by_hand():
    """chi2(1) has its 95 percent point at 3.841459, so the boundary mixture gives p = 0.025 there (half of 0.05), and 1 for LR = 0."""
    assert kill_family.boundary_p(3.841458820694124) == pytest.approx(0.025, abs=1e-12)
    assert kill_family.boundary_p(0.0) == 1.0


def test_the_conditional_statistics_by_hand_with_a_stub_head():
    """Four champion units, blue first (mark 0), with the head's means (lambda_i, mu_i) = (2, 1), (1, 2), (3, 1), (2, 0.5) (the scaling of a residual by its mean matters only when the means differ) and the counts (k_1, k_2) = (1, 0), (3, 1), (2, 3), (4, 0) with weights 1, 1, 2, 1.
    The excess counts k_1 - 1 are (0, 2, 1, 3), so the residuals are e_1 = (k_1 - 1 - lambda_i) / sqrt(lambda_i) and e_2 = (k_2 - mu_i) / sqrt(mu_i); the weighted correlation is computed here with numpy by its definition."""
    rates = np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [2.0, 0.5]])

    class Stub:
        def predict(self, X):
            return rates
    K = np.array([[1, 0], [3, 1], [2, 3], [4, 0]])
    w = np.array([1.0, 1.0, 2.0, 1.0])
    out = kill_family.conditional_statistics(Stub(), np.zeros((4, 1)), K, np.zeros(4, dtype=np.int64), w)['champion']
    e1 = (K[:, 0] - 1 - rates[:, 0]) / np.sqrt(rates[:, 0])
    e2 = (K[:, 1] - rates[:, 1]) / np.sqrt(rates[:, 1])
    m1, m2 = np.dot(w, e1) / w.sum(), np.dot(w, e2) / w.sum()
    expected = np.dot(w, (e1 - m1) * (e2 - m2)) / math.sqrt(np.dot(w, (e1 - m1) ** 2) * np.dot(w, (e2 - m2) ** 2))
    assert out['residual_correlation'] == pytest.approx(expected, abs=1e-12)
    assert out['n_units'] == 4 and out['n_eff'] == pytest.approx(25 / 7)
    n = K[:, 0] - 1
    assert out['nll_poisson'] == pytest.approx(float(np.dot(w, rates[:, 0] - n * np.log(rates[:, 0]) + np.array([math.lgamma(v + 1) for v in n])) / w.sum()), abs=1e-12)
    assert out['alpha'] >= 0.0 and out['nll_nb'] <= out['nll_poisson'] + 1e-15
    assert kill_family.conditional_statistics(Stub(), np.zeros((4, 1)), K, np.zeros(4, dtype=np.int64), w)['execution'] is None


def test_the_conditional_statistics_refuse_a_head_of_another_kill_law():
    """The excess N_d = k_1 - 1 is read around the head's mean: a plain-law head gives the mean of K_d there, so the statistics refuse it (ValueError), in the function and through
    compare_b1; a head that does not name a law (the stub) or names the shifted law is accepted."""
    class Plain:
        kill_law = 'poisson'

        def predict(self, X):
            return np.ones((len(X), 2))

    class Shifted(Plain):
        kill_law = 'shifted'
    K, mark, w = np.array([[1, 0], [2, 1]]), np.zeros(2, dtype=np.int64), np.ones(2)
    with pytest.raises(ValueError, match='shifted'):
        kill_family.conditional_statistics(Plain(), np.zeros((2, 1)), K, mark, w)
    assert kill_family.conditional_statistics(Shifted(), np.zeros((2, 1)), K, mark, w)['champion']['n_units'] == 2
    fit, stop, ev = b1_rows(600, 71), b1_rows(300, 72), b1_rows(300, 73)
    result = kill_family.compare_b1(fit, stop, ev, params=SMALL)
    assert result['conditional']['champion'] is not None                            # compare_b1 passes the shifted head


def test_the_execution_part_has_the_effective_count():
    class Stub:
        def predict(self, X):
            return np.full((len(X), 2), 1.5)
    K, mark, w = np.array([[1, 0], [2, 1], [0, 3]]), np.array([50, 51, 50]), np.array([2.0, 1.0, 1.0])
    out = kill_family.conditional_statistics(Stub(), np.zeros((3, 1)), K, mark, w)['execution']
    assert out['n_units'] == 3 and out['n_eff'] == pytest.approx(16 / 6, abs=1e-15)      # Kish: (2 + 1 + 1)^2 / (4 + 1 + 1)
    zero = kill_family.conditional_statistics(Stub(), np.zeros((2, 1)), K[:2], mark[:2], np.zeros(2))['execution']
    assert zero['n_eff'] is None and zero['residual_correlation'] is None
