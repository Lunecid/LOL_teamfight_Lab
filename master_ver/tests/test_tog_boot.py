"""Tests for the E3 variance functions (PRESPEC_E3_KO.md section 3; PLAN_E1_E5_KO.md Task 8; SDD Ruling 18).

The first four tests are the plan's, verbatim; they hold with the union-indicator V_U as the primary two-way variance.
The rest pin V_U against the shared-count V_2 on hand-computed cases (a linked pair counts once in V_U and c times in
V_2), the -1 rule, the influence residuals of a stratum difference, a dense brute force, the input checks, and the size
of the sparse product on a synthetic matrix shaped like the external player matrix.  Every input is synthetic: no data
file is read.
"""
import math
from statistics import NormalDist
import time
import tracemalloc
import warnings

import numpy as np
import pytest

from studies.tog_experiments_20261012.boot import match_var, cgm_twoway_var, twoway_interval
from studies.tog_experiments_20261012.boot import (
    Z_PRESPEC_95, residual_interval, residual_variances, stratum_difference, union_var)

Z95 = 1.96                       # PRESPEC_E3 section 3: "theta_hat -/+ 1.96 sqrt(V)" at level 0.95 (Ruling 38)


def test_unique_players_equal_match_variance():
    rng = np.random.default_rng(1)
    M = 3000
    players = np.arange(M * 10).reshape(M, 10)
    values = rng.normal(0, 1, M)
    assert np.isclose(cgm_twoway_var(values, players), match_var(values))

def test_repeated_players_widen():
    rng = np.random.default_rng(0)
    M, P = 2000, 200
    players = np.stack([rng.choice(P, 10, replace=False) for _ in range(M)])
    effect = rng.normal(0, 1, P)
    values = effect[players].mean(axis=1) + rng.normal(0, 0.1, M)
    out = twoway_interval(values, players)
    assert out["design_effect"] > 3 and not out["fallback"]

def test_unidentified_players_do_not_count():
    values = np.array([1.0, -1.0, 0.5, -0.5])
    assert np.isclose(cgm_twoway_var(values, np.full((4, 10), -1)), match_var(values))

def test_negative_v2_falls_back():
    values = np.array([1.0, -1.0])                     # opposite residuals
    players = np.array([list(range(10)), list(range(9)) + [10]])  # share 9 of 10 players
    assert cgm_twoway_var(values, players) < match_var(values)
    out = twoway_interval(values, players)
    assert out["fallback"] and np.isclose(out["se_twoway"], out["se_match"])


# ---------------------------------------------------------------------------------------------------------------------

def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def rows(*matches, width=10):
    """Player rows padded with -1 (unidentified) to ``width`` columns."""
    return np.array([list(m) + [-1] * (width - len(m)) for m in matches], dtype=np.int64)


def dense_terms(e, players):
    """V1, V2, VU and the pair counts from the dense M x M shared-player matrix (brute force, small M only)."""
    e = np.asarray(e, dtype=float)
    m = e.size
    a = np.zeros((m, int(players.max()) + 1), dtype=np.int64)
    for i, row in enumerate(players):
        a[i, row[row >= 0]] = 1
    c = a @ a.T
    off = ~np.eye(m, dtype=bool)
    outer = np.outer(e, e)
    s1 = float(e @ e)
    upper = np.triu(np.ones((m, m), dtype=bool), 1)
    return {"V1": s1 / m**2, "VU": (s1 + outer[off & (c > 0)].sum()) / m**2,
            "V2": (s1 + (c * outer)[off].sum()) / m**2,
            "pairs_c_ge_1": int((upper & (c >= 1)).sum()), "pairs_c_ge_2": int((upper & (c >= 2)).sum())}


def v2_player_form(e, players):
    """The plan's V2 = (1/M^2)[sum e^2 + sum_p (sum_{m in p} e_m)^2 - sum_m k_m e_m^2], by player sums."""
    e = np.asarray(e, dtype=float)
    known = players >= 0
    by_player = np.bincount(players[known], weights=np.broadcast_to(e[:, None], players.shape)[known])
    k = known.sum(axis=1)
    return (float(e @ e) + float(by_player @ by_player) - float(k @ (e * e))) / e.size**2


def test_negative_v2_case_by_hand():
    # e = (1, -1), M^2 = 4, one pair sharing c = 9 players:
    # V1 = 2/4, V2 = (2 + 2*9*(-1))/4 = -4, VU = (2 + 2*(-1))/4 = 0 < V1, so V = V1 and DE = 1.
    values = np.array([1.0, -1.0])
    players = np.array([list(range(10)), list(range(9)) + [10]])
    out = twoway_interval(values, players)
    assert close(out["V1"], 0.5) and close(out["V2"], -4.0) and close(out["VU"], 0.0)
    assert close(out["cross_U"], -0.5) and out["cross_U_sign"] == -1
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"], out["share_c_ge_2"]) == (1, 1, 1.0)
    assert out["fallback"] and close(out["V"], 0.5) and close(out["design_effect"], 1.0)
    assert close(out["se_twoway"], math.sqrt(0.5)) and close(out["point"], 0.0)
    assert close(union_var(values, players), 0.0) and close(cgm_twoway_var(values, players), -4.0)
    assert close(v2_player_form(values, players), -4.0)


@pytest.mark.parametrize("c", [1, 2, 3])
def test_vu_counts_a_linked_pair_once_v2_counts_each_shared_player(c):
    # e = (1, 1, -2) (mean 0), M^2 = 9; matches 0 and 1 share c players, match 2 shares none:
    # V1 = 6/9, VU = (6 + 2*1*1)/9 = 8/9 whatever c is, V2 = (6 + 2*c*1*1)/9.
    values = np.array([1.0, 1.0, -2.0])
    players = rows([0, 1, 2], list(range(c)) + list(range(10, 13 - c)), [20, 21, 22])
    out = twoway_interval(values, players)
    assert close(out["V1"], 6 / 9) and close(out["VU"], 8 / 9) and close(out["V2"], (6 + 2 * c) / 9)
    assert close(out["cross_U"], 2 / 9) and out["cross_U_sign"] == 1 and not out["fallback"]
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"]) == (1, int(c >= 2))
    assert out["share_c_ge_2"] == (1.0 if c >= 2 else 0.0)
    assert close(union_var(values, players), 8 / 9) and close(cgm_twoway_var(values, players), (6 + 2 * c) / 9)
    assert close(v2_player_form(values, players), (6 + 2 * c) / 9)


def test_pairs_with_one_and_with_two_shared_players():
    # e = (1, 1, -1, -1) (mean 0), M^2 = 16; c01 = 2 (players 0, 1), c02 = 1 (player 2), c23 = 1 (player 5), others 0.
    # V1 = 4/16; VU = (4 + 2*(1 - 1 + 1))/16 = 6/16; V2 = (4 + 2*(2*1 + 1*(-1) + 1*1))/16 = 8/16.
    # Player form of V2: sum_p (sum e)^2 = 4+4+0+1+1+4+1+1 = 16, sum k e^2 = 3*4 = 12, (4 + 16 - 12)/16 = 8/16.
    values = np.array([1.0, 1.0, -1.0, -1.0])
    players = rows([0, 1, 2], [0, 1, 3], [2, 4, 5], [5, 6, 7])
    out = twoway_interval(values, players)
    assert close(out["V1"], 0.25) and close(out["VU"], 0.375) and close(out["V2"], 0.5)
    assert close(v2_player_form(values, players), 0.5)
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"]) == (3, 1) and close(out["share_c_ge_2"], 1 / 3)
    assert close(out["cross_U"], 0.125) and out["cross_U_sign"] == 1 and not out["fallback"]
    assert close(out["point"], 0.0) and close(out["V"], 0.375)
    assert close(out["se_match"], 0.5) and close(out["se_twoway"], math.sqrt(0.375))
    assert close(out["design_effect"], math.sqrt(1.5))
    assert out["level"] == 0.95 and out["z"] == Z95 == 1.96
    lo, hi = out["interval"]
    assert close(lo, -Z95 * math.sqrt(0.375)) and close(hi, Z95 * math.sqrt(0.375))
    assert out["matches"] == 4


@pytest.mark.parametrize("level", [0.90, 0.975, 0.99, 0.5])
def test_z_is_the_prespec_196_at_95_and_the_normal_quantile_elsewhere(level):
    # PRESPEC_E3 section 3 writes theta_hat -/+ 1.96 sqrt(V): level 0.95 uses the literal 1.96 (not Phi^-1(0.975) =
    # 1.959964); any other level uses Phi^-1((1 + level)/2), here from the standard library (independent of scipy).
    values = np.array([1.0, 1.0, -1.0, -1.0])
    players = rows([0, 1, 2], [0, 1, 3], [2, 4, 5], [5, 6, 7])       # se_twoway = sqrt(0.375), as above
    at_95 = twoway_interval(values, players)
    assert Z_PRESPEC_95 == 1.96 and at_95["z"] == 1.96 and at_95["z"] != NormalDist().inv_cdf(0.975)
    assert at_95["interval"] == (0.0 - 1.96 * at_95["se_twoway"], 0.0 + 1.96 * at_95["se_twoway"])
    other = twoway_interval(values, players, level=level)
    z = NormalDist().inv_cdf((1 + level) / 2)
    assert other["level"] == level and math.isclose(other["z"], z, rel_tol=1e-12)
    lo, hi = other["interval"]
    assert close(lo, -z * math.sqrt(0.375)) and close(hi, z * math.sqrt(0.375))
    assert {k: v for k, v in other.items() if k not in ("level", "z", "interval")} == \
        {k: v for k, v in at_95.items() if k not in ("level", "z", "interval")}      # only z and the bounds move


@pytest.mark.parametrize("values, players, v1, vu, v2, pairs", [
    # e = (1, 1, -2), M^2 = 9; c01 = 3 (players 0, 1, 2), c02 = 1 (player 9):
    # VU = (6 + 2*(1 - 2))/9 = 4/9 < V1 = 6/9 < V2 = (6 + 2*(3*1 + 1*(-2)))/9 = 8/9, so V = V1 and fallback.
    ([1.0, 1.0, -2.0], rows([0, 1, 2, 9], [0, 1, 2], [9]), 6 / 9, 4 / 9, 8 / 9, (2, 1)),
    # e = (2, 2, -1, -3), M^2 = 16; c01 = 1 (player 0), c02 = 5 (players 10-14), c23 = 1 (player 30):
    # V2 = (18 + 2*(4 - 5*2 + 3))/16 = 12/16 < V1 = 18/16 < VU = (18 + 2*(4 - 2 + 3))/16 = 28/16, so V = VU.
    ([2.0, 2.0, -1.0, -3.0], rows([0, 10, 11, 12, 13, 14], [0, 20], [10, 11, 12, 13, 14, 30], [30, 40]),
     18 / 16, 28 / 16, 12 / 16, (3, 1)),
])
def test_primary_variance_follows_vu_not_v2(values, players, v1, vu, v2, pairs):
    out = twoway_interval(np.array(values), players)
    assert close(out["V1"], v1) and close(out["VU"], vu) and close(out["V2"], v2)
    assert close(v2_player_form(np.array(values), players), v2)
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"]) == pairs
    assert out["fallback"] == (vu < v1) and close(out["V"], max(v1, vu))
    assert close(out["se_twoway"], math.sqrt(max(v1, vu))) and close(out["design_effect"], math.sqrt(max(v1, vu) / v1))


def test_unique_players_give_three_equal_variances():
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1.0, 50)
    out = twoway_interval(values, np.arange(500).reshape(50, 10))
    assert out["V1"] == out["V2"] == out["VU"] == match_var(values)       # no linked pair: no cross term at all
    assert out["cross_U"] == 0.0 and out["cross_U_sign"] == 0
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"]) == (0, 0) and math.isnan(out["share_c_ge_2"])
    assert out["design_effect"] == 1.0 and not out["fallback"] and out["se_twoway"] == out["se_match"]


def test_unidentified_players_link_nothing():
    values = np.array([0.3, -0.1, 0.7, -0.9])
    out = twoway_interval(values, np.full((4, 10), -1))
    assert out["V1"] == out["V2"] == out["VU"] and out["pairs_c_ge_1"] == 0
    # -1 in nine columns of every match is not a shared player
    out = twoway_interval(values, rows([5], [6], [], []))
    assert out["V1"] == out["V2"] == out["VU"] and out["pairs_c_ge_1"] == 0
    # one identified shared player links matches 0 and 1 (c = 1)
    out = twoway_interval(values, rows([5], [5], [], []))
    e = values - values.mean()
    assert out["pairs_c_ge_1"] == 1 and out["pairs_c_ge_2"] == 0
    assert close(out["VU"], out["V1"] + 2 * e[0] * e[1] / 16) and close(out["V2"], out["VU"])


def test_stratum_difference_by_hand():
    # S_hi = {0, 1} with D_hi = (4, 2): theta_hi = 3.  S_lo = {1, 2, 3} with D_lo = (0, 3, 3): theta_lo = 2.  Delta = 1.
    # Match 4 is in neither stratum, so M = 5 (all matches, not |S_hi u S_lo| = 4); M/M_hi = 5/2, M/M_lo = 5/3:
    # e = (5/2*(4-3), 5/2*(2-3) - 5/3*(0-2), -5/3*(3-2), -5/3*(3-2), 0) = (5/2, 5/6, -5/3, -5/3, 0), sum 0.
    # V1 = (25/4 + 25/36 + 25/9 + 25/9)/25 = 1/2 = Var(hi) + Var(lo) - 2 Cov = (1+1)/4 + (4+1+1)/9 - 2*(-1)(-2)/(2*3).
    # (Weights M/M_s with M = |S_hi u S_lo| = 4 would give e = (2, 2/3, -4/3, -4/3, 0) and V1 = 8/25.)
    d_hi = np.array([4.0, 2.0, np.nan, np.nan, np.nan])          # NaN outside the stratum must not leak
    d_lo = np.array([np.nan, 0.0, 3.0, 3.0, np.nan])
    in_hi = np.array([True, True, False, False, False])
    in_lo = np.array([False, True, True, True, False])
    delta, e = stratum_difference(d_hi, in_hi, d_lo, in_lo)
    assert close(delta, 1.0)
    assert np.allclose(e, [5 / 2, 5 / 6, -5 / 3, -5 / 3, 0.0], rtol=1e-12, atol=0) and abs(e.sum()) < 1e-12
    assert e[4] == 0.0
    var_hi, var_lo, cov = (1 + 1) / 2**2, (4 + 1 + 1) / 3**2, (2 - 3) * (0 - 2) / (2 * 3)
    unique = residual_variances(e, np.arange(50).reshape(5, 10))
    assert close(unique["V1"], 0.5) and close(unique["V1"], var_hi + var_lo - 2 * cov)
    assert unique["VU"] == unique["V2"] == unique["V1"]
    # c01 = 2 (players 0, 1), c23 = 1 (player 6), c04 = 1 (player 2; e4 = 0, so this link adds nothing), M^2 = 25:
    # VU = 1/2 + 2*(25/12 + 25/9 + 0)/25 = 8/9;  V2 = 1/2 + 2*(2*25/12 + 1*25/9 + 1*0)/25 = 19/18.
    # Player form of V2: sum_p (sum e)^2 = 1850/36, sum k e^2 = 1350/36, sum e^2 = 450/36: (450 + 1850 - 1350)/36/25.
    players = rows([0, 1, 2], [0, 1, 3], [4, 5, 6], [6, 7, 8], [2, 9])
    out = residual_interval(delta, e, players)
    assert close(out["V1"], 0.5) and close(out["VU"], 8 / 9) and close(out["V2"], 19 / 18)
    assert close(v2_player_form(e, players), 19 / 18)
    assert (out["pairs_c_ge_1"], out["pairs_c_ge_2"]) == (3, 1) and close(out["share_c_ge_2"], 1 / 3)
    assert close(out["point"], 1.0) and close(out["design_effect"], 4 / 3) and not out["fallback"]
    assert close(out["se_twoway"], math.sqrt(8 / 9)) and close(out["se_match"], math.sqrt(0.5))
    assert out["matches"] == 5
    lo, hi = out["interval"]
    assert close(lo, 1.0 - Z95 * math.sqrt(8 / 9)) and close(hi, 1.0 + Z95 * math.sqrt(8 / 9))


def random_players(rng, m, p, missing=0.1):
    """M matches of ten distinct players out of p, each slot unidentified (-1) with probability ``missing``."""
    players = np.stack([rng.choice(p, 10, replace=False) for _ in range(m)])
    players[rng.random(players.shape) < missing] = -1
    return players


def test_residual_functions_agree_with_twoway_interval():
    rng = np.random.default_rng(3)
    players = random_players(rng, 400, 60)
    values = rng.normal(0.02, 0.3, 400)
    out = twoway_interval(values, players)
    e = values - values.mean()
    variances = residual_variances(e, players)
    for key, value in variances.items():
        assert math.isclose(out[key], value, rel_tol=1e-12, abs_tol=1e-18) if isinstance(value, float) else out[key] == value
    again = residual_interval(values.mean(), e, players)
    assert again.keys() == out.keys()
    for key, value in again.items():
        if key == "interval":
            assert np.allclose(value, out[key], rtol=1e-12, atol=0)
        elif isinstance(value, float):
            assert math.isclose(out[key], value, rel_tol=1e-12, abs_tol=1e-18)
        else:
            assert out[key] == value
    assert close(match_var(values), out["V1"]) and close(cgm_twoway_var(values, players), out["V2"])
    assert close(union_var(values, players), out["VU"])
    assert out["pairs_c_ge_2"] > 0 and out["pairs_c_ge_1"] > out["pairs_c_ge_2"]     # both kinds of linked pair occur


def test_matches_dense_brute_force():
    rng = np.random.default_rng(11)
    players = random_players(rng, 300, 40)
    e = rng.normal(0.5, 1.0, 300)                       # residuals are used as given (not re-centred)
    got = residual_variances(e, players)
    want = dense_terms(e, players)
    for key in ("V1", "V2", "VU"):
        assert math.isclose(got[key], want[key], rel_tol=1e-10), key
    assert math.isclose(got["V2"], v2_player_form(e, players), rel_tol=1e-10)
    assert (got["pairs_c_ge_1"], got["pairs_c_ge_2"]) == (want["pairs_c_ge_1"], want["pairs_c_ge_2"])
    assert close(got["share_c_ge_2"], want["pairs_c_ge_2"] / want["pairs_c_ge_1"])
    assert close(got["cross_U"], got["VU"] - got["V1"]) and got["matches"] == 300


def test_constant_values_have_zero_variance_without_warnings():
    players = rows([0, 1], [0, 1], [1, 2], [3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = twoway_interval(np.full(4, 0.25), players)
    assert out["V1"] == out["VU"] == out["V2"] == out["V"] == 0.0 and out["cross_U_sign"] == 0
    assert math.isnan(out["design_effect"]) and not out["fallback"]
    assert out["interval"] == (0.25, 0.25) and out["se_twoway"] == 0.0


PAIR = rows([0, 1], [2, 3])


@pytest.mark.parametrize("call, error", [
    (lambda: match_var([]), ValueError),
    (lambda: match_var([1.0, np.nan]), ValueError),
    (lambda: match_var([[1.0, 2.0]]), ValueError),
    (lambda: union_var([1.0, 2.0], rows([0], [1], [2])), ValueError),                 # rows != matches
    (lambda: union_var([1.0, 2.0], np.array([0, 1])), ValueError),                     # not (M, k)
    (lambda: union_var([1.0, 2.0], PAIR.astype(float)), TypeError),                    # not integers
    (lambda: union_var([1.0, 2.0], rows([0, 0], [1])), ValueError),                    # a player twice in one match
    (lambda: cgm_twoway_var([1.0, 2.0], rows([-2], [1])), ValueError),                 # below -1
    (lambda: residual_variances([1.0, np.inf], PAIR), ValueError),
    (lambda: twoway_interval([1.0, 2.0], PAIR, level=1.0), ValueError),
    (lambda: twoway_interval([1.0, 2.0], PAIR, level=0.0), ValueError),
    (lambda: twoway_interval([1.0, 2.0], PAIR, level=float("nan")), ValueError),
    (lambda: residual_interval(float("nan"), [1.0, -1.0], PAIR), ValueError),
])
def test_invalid_inputs_are_refused(call, error):
    with pytest.raises(error):
        call()


GOOD = np.array([1.0, 2.0, 3.0])
IN = np.array([True, True, False])


@pytest.mark.parametrize("args, error", [
    ((GOOD, np.zeros(3, dtype=bool), GOOD, IN), ValueError),                           # empty stratum
    ((GOOD, IN, GOOD, np.array([0, 1, 1])), TypeError),                                # mask not boolean
    ((GOOD, IN, GOOD[:2], IN[:2]), ValueError),                                        # lengths differ
    ((np.array([np.nan, 2.0, 3.0]), IN, GOOD, IN), ValueError),                        # NaN inside the stratum
])
def test_stratum_difference_refuses_bad_strata(args, error):
    with pytest.raises(error):
        stratum_difference(*args)


# ---------------------------------------------------------------------------------------------------------------------
# Size check.  The external player matrix (Task 7 report) has 38,424 matches x 10 players, 64,171 players, matches per
# player median 2, mean 5.99, max 131, 42 % of players in one match, and sum_p n_p(n_p - 1)/2 = 4,042,838 player-pair
# links.  The synthetic matrix draws matches per player from k^-1.3 e^(-k/35) on 1..131 (realized: median 2, mean
# 5.87, max 130, 39.7 % ones, 2 % unidentified slots), which gives 4,368,166 links: 4,357,952 linked match pairs, 10,201
# of them with c >= 2.  Measured 10-12 on the author's machine: twoway_interval takes 0.14-0.19 s with a traced peak of
# 286 MB (tracemalloc); the bounds below (10 s, 1 GB) leave room for a busy machine.

M_EXT, P_EXT, K_MAX = 38_424, 64_171, 131


def realistic_players(rng):
    k = np.arange(1, K_MAX + 1)
    w = k ** -1.3 * np.exp(-k / 35.0)
    per_player = rng.choice(k, size=P_EXT, p=w / w.sum())
    slots = np.repeat(np.arange(P_EXT), per_player)
    rng.shuffle(slots)
    need = M_EXT * 10
    slots = np.concatenate([slots[:need], np.full(max(need - slots.size, 0), -1)])     # short: unidentified slots
    rng.shuffle(slots)
    players = np.sort(slots.reshape(M_EXT, 10), axis=1)
    twice = np.zeros(players.shape, dtype=bool)
    twice[:, 1:] = (players[:, 1:] == players[:, :-1]) & (players[:, 1:] >= 0)
    players[twice] = -1                                  # a player at most once per match, as in the real matrix
    return players


def test_size_on_a_realistic_player_matrix():
    rng = np.random.default_rng(20261012)
    players = realistic_players(rng)
    values = rng.normal(0.0, 0.05, M_EXT)
    tracemalloc.start()
    try:
        tick = time.perf_counter()
        out = twoway_interval(values, players)
        seconds = time.perf_counter() - tick
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert seconds < 10.0, seconds
    assert peak < 1.0e9, peak
    n = np.bincount(players[players >= 0])
    links = int((n * (n - 1) // 2).sum())               # = sum over linked pairs of c
    assert 3.5e6 < links < 5.5e6
    assert out["pairs_c_ge_1"] + out["pairs_c_ge_2"] <= links <= out["pairs_c_ge_1"] + 9 * out["pairs_c_ge_2"]
    assert math.isclose(out["V2"], v2_player_form(values - values.mean(), players), rel_tol=1e-9)
    assert 0.0 < out["share_c_ge_2"] < 1.0 and out["matches"] == M_EXT
    assert np.isfinite([out["V1"], out["V2"], out["VU"], out["design_effect"]]).all()
