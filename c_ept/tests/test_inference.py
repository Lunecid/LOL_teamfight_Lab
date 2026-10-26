"""Synthetic CPU checks of ``c_ept.evaluation.inference`` (S4b-prep M10, D15 as amended by D20): the one common match-resample matrix,
the ratio bootstrap, the two-sided p, Holm over the seven rows, the non-inferiority rule and the claim / withdrawn / inconclusive verdict.
All examples are small and computed by hand; no data, no model, no device.
"""
import inspect

import numpy as np
import pytest

from c_ept.evaluation import inference

SEEDS = (20261022, 20261023, 20261024)


# ---- the common resample matrix -------------------------------------------------------------------------------------------------------
def test_resample_defaults_are_the_d15_policy():
    sig = inspect.signature(inference.resample_matrix)
    assert sig.parameters['B'].default == 10_000 and sig.parameters['seed'].default == 20261022
    assert inference.SEEDS == SEEDS


def test_resample_matrix_is_shared_and_seeded():
    n, B = 17, 600                                                         # more than one internal block of replicates
    a =inference.resample_matrix(n, B=B, seed=20261022)
    b = inference.resample_matrix(n, B=B, seed=20261022)
    assert a.shape == (B, n) and a.dtype.kind == 'i'
    assert np.array_equal(a, b)                                            # the same call is the same matrix: shared by every comparison
    assert (a.sum(axis=1) == n).all() and (a >= 0).all()                   # each row counts n draws of the n matches
    assert not np.array_equal(a, inference.resample_matrix(n, B=B, seed=20261023))
    # what the matrix is, written out: row b counts the draws of rng.integers(n, size=n), one row after the other
    rng = np.random.default_rng(20261022)
    want = np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(B)])
    assert np.array_equal(a, want)
    # the default policy is B = 10,000 and seed 20261022 (the shape only: no further work)
    assert inference.resample_matrix(5).shape == (10_000, 5)


def test_resample_matrix_refuses_nonsense():
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            inference.resample_matrix(bad, B=4)
    with pytest.raises(ValueError):
        inference.resample_matrix(3, B=0)


def test_match_aggregate_puts_the_one_over_n_m_weight_into_the_per_match_mean():
    # matches 'b' (2 rows) and 'a' (1 row): sorted match order a, b; per-match mean = sum over rows of (1/n_m) * value
    ids, mean = inference.match_aggregate(['b', 'a', 'b'], np.array([4.0, 10.0, 6.0]))
    assert list(ids) == ['a', 'b'] and np.array_equal(mean, [10.0, 5.0])


def test_match_aggregate_onto_a_common_id_list_puts_absent_matches_at_zero():
    common = np.array(['a', 'b', 'c'])
    ids, mean = inference.match_aggregate(['b', 'b', 'a'], np.array([4.0, 6.0, 10.0]), ids=common)
    assert list(ids) == ['a', 'b', 'c'] and np.array_equal(mean, [10.0, 5.0, 0.0])        # 'c' is absent: 0
    assert np.array_equal(inference.match_presence(['b', 'b', 'a'], common), [1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        inference.match_aggregate(['b', 'z'], np.array([1.0, 2.0]), ids=common)           # unknown id
    with pytest.raises(ValueError):
        inference.match_aggregate(['a'], np.array([1.0]), ids=np.array(['b', 'a']))       # the id list must be sorted and unique
    with pytest.raises(ValueError):
        inference.match_presence(['z'], common)


def test_a_head_missing_a_match_keeps_the_common_matrix():
    common = np.array([1, 2, 3, 4])
    matrix = inference.resample_matrix(4, B=50)
    # head with rows in matches 1, 2, 4 only (match 3 absent): numerator / denominator rows of two heads on the same rows
    match = [1, 1, 2, 4, 4, 4]
    nll_model = np.array([2.0, 4.0, 3.0, 1.0, 2.0, 3.0])
    nll_b0 = np.array([4.0, 4.0, 3.0, 2.0, 2.0, 2.0])
    _, a = inference.match_aggregate(match, nll_model, ids=common)
    _, d = inference.match_aggregate(match, nll_b0, ids=common)
    assert np.array_equal(a, [3.0, 3.0, 0.0, 2.0]) and np.array_equal(d, [4.0, 3.0, 0.0, 2.0])
    got = inference.bootstrap_stat(a, d, matrix)
    # direct: only the present matches contribute, with the counts of the full common matrix
    want = (matrix[:, [0, 1, 3]] @ np.array([3.0, 3.0, 2.0])) / (matrix[:, [0, 1, 3]] @ np.array([4.0, 3.0, 2.0]))
    assert np.array_equal(got, want)
    # a mean under the presence convention: the denominator is 1 for a present match and 0 for an absent one
    pres = inference.match_presence(match, common)
    mean_draws = inference.bootstrap_stat(a, pres, matrix)
    assert np.allclose(mean_draws, (matrix @ a) / (matrix @ pres))
    # the vector over the head's own three matches does not fit the common matrix: refused by length
    with pytest.raises(ValueError):
        inference.bootstrap_stat(a[[0, 1, 3]], d[[0, 1, 3]], matrix)


def test_resample_matrix_prefix_property_for_a_smaller_b():
    # the C4 interval (D13, B = 2,000) is the first 2,000 rows of the B = 10,000 matrix of the same n and seed
    assert np.array_equal(inference.resample_matrix(7, B=300), inference.resample_matrix(7, B=700)[:300])


# ---- the ratio bootstrap --------------------------------------------------------------------------------------------------------------
def test_ratio_bootstrap_recomputes_numerator_and_denominator_together():
    num = np.array([2.0, 4.0, 6.0])
    den = np.array([1.0, 2.0, 1.0])
    matrix = np.array([[1, 1, 1], [3, 0, 0], [0, 2, 1]])
    got = inference.bootstrap_stat(num, den, matrix)
    # (sum_m c_m num_m) / (sum_m c_m den_m) in each resample: 12/4, 6/3, 14/5
    assert np.array_equal(got, np.array([3.0, 2.0, 14.0 / 5.0]))
    # not a fixed denominator (that would give 12/4, 6/4, 14/4) and not a mean of per-match ratios (2 for every resample here)
    assert not np.allclose(got, matrix @ num / den.sum())
    # the point estimate is the resample with every match once
    assert inference.bootstrap_stat(num, den, np.ones((1, 3), dtype=int))[0] == 12.0 / 4.0


def test_ratio_bootstrap_with_unit_denominators_is_the_resampled_mean():
    num = np.array([1.0, 2.0, 6.0])
    matrix = inference.resample_matrix(3, B=50)
    got = inference.bootstrap_stat(num, np.ones(3), matrix)
    assert np.allclose(got, matrix @ num / 3.0)


def test_ratio_bootstrap_refuses_a_zero_denominator_and_bad_shapes():
    with pytest.raises(ValueError):
        inference.bootstrap_stat(np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([[2, 0]]))     # replicate with denominator 0
    with pytest.raises(ValueError):
        inference.bootstrap_stat(np.ones(3), np.ones(2), np.ones((1, 3), dtype=int))
    with pytest.raises(ValueError):
        inference.bootstrap_stat(np.ones(3), np.ones(3), np.ones((1, 2), dtype=int))


# ---- the paired seeds -----------------------------------------------------------------------------------------------------------------
def test_paired_seed_diffs_pair_by_equal_seed_value():
    ept = {20261022: 0.50, 20261023: 0.40, 20261024: 0.60}
    cmp_ = {20261024: 0.70, 20261022: 0.55, 20261023: 0.50}
    got = inference.paired_seed_diffs(ept, cmp_)
    assert got == {20261022: pytest.approx(-0.05), 20261023: pytest.approx(-0.10), 20261024: pytest.approx(-0.10)}   # EPT - comparator
    with pytest.raises(ValueError):
        inference.paired_seed_diffs(ept, {20261022: 0.5, 20261023: 0.5, 20261025: 0.5})              # different seed value
    with pytest.raises(ValueError):
        inference.paired_seed_diffs({20261022: 0.5, 20261023: 0.5}, {20261022: 0.5, 20261023: 0.5})  # not the three seeds


def test_paired_seed_mean_is_the_mean_of_the_three_paired_differences():
    assert inference.paired_seed_mean({20261022: -0.05, 20261023: -0.10, 20261024: -0.15}) == pytest.approx(-0.10)
    assert inference.paired_seed_mean([-0.3, 0.0, 0.3]) == 0.0
    # draws: the mean of the three per-seed draw vectors, element by element
    draws = inference.paired_seed_mean([np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 9.0])])
    assert np.array_equal(draws, np.array([3.0, 5.0]))
    with pytest.raises(ValueError):
        inference.paired_seed_mean([0.1, 0.2])
    with pytest.raises(ValueError):
        inference.paired_seed_mean({20261022: 0.1, 20261023: 0.2, 20261030: 0.3})


# ---- the two-sided p ------------------------------------------------------------------------------------------------------------------
def test_two_sided_p_formula():
    # B = 9: observed -1 (favours EPT), 1 draw on the far side of 0 (>= 0): p = 2 (1 + 1) / (9 + 1) = 0.4
    draws = np.array([-3.0, -2.0, -2.5, -1.0, -0.5, -1.5, -2.2, 0.5, -0.1])
    assert inference.two_sided_p(draws, -1.0) == pytest.approx(0.4, abs=0, rel=1e-15)
    # no far-side draw: p = 2 / (B + 1) = 0.2
    assert inference.two_sided_p(-np.abs(draws) - 0.1, -1.0) == pytest.approx(0.2, abs=0, rel=1e-15)
    # the mirror image: observed +1, far side is <= 0
    assert inference.two_sided_p(-draws, 1.0) == pytest.approx(0.4, abs=0, rel=1e-15)
    # capped at 1 and an observed 0 has p = 1
    assert inference.two_sided_p(np.array([1.0, 2.0, -1.0, 3.0]), -1.0) == 1.0              # 2 * (1 + 4) / 5 > 1
    assert inference.two_sided_p(np.array([1.0, -1.0]), 0.0) == 1.0
    # a draw exactly at 0 counts as far side (conservative)
    assert inference.two_sided_p(np.array([0.0, -1.0, -2.0, -3.0, -4.0]), -1.0) == pytest.approx(2 * 2 / 6)
    with pytest.raises(ValueError):
        inference.two_sided_p(np.array([]), -1.0)


# ---- Holm over the seven rows ---------------------------------------------------------------------------------------------------------
def test_holm_on_seven_rows():
    rows = ['structure', 'C1', 'C2a', 'C2b', 'C3-edges', 'C3-terrain', 'C3-patch']
    assert tuple(rows) == inference.HOLM_ROWS
    p = dict(zip(rows, [0.020, 0.001, 0.600, 0.008, 0.200, 0.030, 0.012]))
    got = inference.holm(p)                                                   # alpha = 0.05
    # ascending: 0.001 x7, 0.008 x6, 0.012 x5, 0.020 x4, 0.030 x3, 0.200 x2, 0.600 x1
    want = {'C1': 0.007, 'C2b': 0.048, 'C3-patch': 0.060, 'structure': 0.080, 'C3-terrain': 0.090, 'C3-edges': 0.400, 'C2a': 0.600}
    assert {k: v['adjusted_p'] for k, v in got.items()} == pytest.approx(want, abs=0, rel=1e-12)
    assert {k for k, v in got.items() if v['reject']} == {'C1', 'C2b'}      # adjusted p < 0.05
    assert all(got[k]['p'] == p[k] for k in rows)


def test_holm_enforces_monotone_adjusted_p_and_the_cap_at_one():
    got = inference.holm({'a': 0.010, 'b': 0.011, 'c': 0.5})
    # 3 * 0.010 = 0.030 ; 2 * 0.011 = 0.022 -> raised to 0.030 ; 1 * 0.5
    assert got['a']['adjusted_p'] == pytest.approx(0.030) and got['b']['adjusted_p'] == pytest.approx(0.030)
    assert got['c']['adjusted_p'] == 0.5
    assert inference.holm({'a': 0.6, 'b': 0.7})['a']['adjusted_p'] == 1.0         # 2 * 0.6 > 1
    # the adjusted level is the family alpha itself: with alpha 0.025 the first row is no longer rejected
    assert inference.holm({'a': 0.010, 'b': 0.011, 'c': 0.5}, alpha=0.025)['a']['reject'] is False
    with pytest.raises(ValueError):
        inference.holm({})
    with pytest.raises(ValueError):
        inference.holm({'a': 1.5})


# ---- non-inferiority ------------------------------------------------------------------------------------------------------------------
def test_noninferiority_uses_upper_end_against_eps():
    draws = np.arange(1, 201) / 200.0                       # 0.005 .. 1.0
    upper = float(np.quantile(draws, (1 + 0.95) / 2))     # upper end of the 95% percentile interval
    assert upper == pytest.approx(0.98, abs=0.01)
    assert inference.noninferior(draws, upper + 1e-9) is True
    assert inference.noninferior(draws, upper - 1e-9) is False
    assert inference.noninferior(draws, upper) is False     # strict: the upper end must be below eps
    # the mean (0.5025) below eps is not enough: the rule reads the upper end
    assert float(draws.mean()) < 0.6 < upper and inference.noninferior(draws, 0.6) is False
    # level moves the end: a 50% interval has its upper end at the 75th percentile
    assert inference.noninferior(draws, 0.8, level=0.5) is True
    with pytest.raises(ValueError):
        inference.noninferior(draws, 0.0)
    with pytest.raises(ValueError):
        inference.noninferior(draws, 0.1, level=1.0)


# ---- the verdict ----------------------------------------------------------------------------------------------------------------------
def test_verdict_requires_all_three_seed_signs():
    all_better = [-0.2, -0.1, -0.3]
    assert inference.verdict(all_better, adjusted_p=0.01) == 'claim'
    # mean favours EPT and the adjusted p is small, but one seed does not: no claim
    assert inference.verdict([-0.5, -0.4, 0.1], adjusted_p=0.01) == 'inconclusive'
    assert inference.verdict([-0.5, -0.4, 0.0], adjusted_p=0.01) == 'inconclusive'        # a zero is not "favours EPT"
    # all signs fine but the adjusted p is not below alpha (strict)
    assert inference.verdict(all_better, adjusted_p=0.05) == 'inconclusive'
    assert inference.verdict(all_better, adjusted_p=0.049, alpha=0.05) == 'claim'


def test_verdict_withdrawal_is_the_comparator_significantly_better_at_the_same_level():
    assert inference.verdict([0.2, 0.1, 0.3], adjusted_p=0.01) == 'withdrawn'
    assert inference.verdict([0.2, 0.1, 0.3], adjusted_p=0.06) == 'inconclusive'          # same adjusted level, not met
    # withdrawal needs the point estimate, not all three seeds
    assert inference.verdict([0.5, 0.4, -0.1], adjusted_p=0.01) == 'withdrawn'
    assert inference.verdict([0.0, 0.0, 0.0], adjusted_p=0.01) == 'inconclusive'


def test_verdict_direction_can_be_flipped_for_higher_is_better_metrics():
    assert inference.verdict([0.2, 0.1, 0.3], adjusted_p=0.01, lower_is_better=False) == 'claim'
    assert inference.verdict([-0.2, -0.1, -0.3], adjusted_p=0.01, lower_is_better=False) == 'withdrawn'


def test_verdict_refuses_malformed_input():
    for bad in ([-0.1, -0.2], [-0.1, -0.2, np.nan]):
        with pytest.raises(ValueError):
            inference.verdict(bad, adjusted_p=0.01)
    with pytest.raises(ValueError):
        inference.verdict([-0.1, -0.2, -0.3], adjusted_p=1.2)
    with pytest.raises(ValueError):
        inference.verdict([-0.1, -0.2, -0.3], adjusted_p=0.01, alpha=0.0)


def test_c3_patch_passes_only_if_both_parts_pass():
    assert inference.c3_patch_passes(noninferior_dev=True, superiority_verdict='claim') is True
    assert inference.c3_patch_passes(noninferior_dev=False, superiority_verdict='claim') is False
    assert inference.c3_patch_passes(noninferior_dev=True, superiority_verdict='inconclusive') is False
    assert inference.c3_patch_passes(noninferior_dev=True, superiority_verdict='withdrawn') is False
    with pytest.raises(ValueError):
        inference.c3_patch_passes(noninferior_dev=True, superiority_verdict='maybe')


def test_end_to_end_hand_pipeline_on_a_toy_comparison():
    """matches -> per-match mean Brier of both arms for three seeds -> shared matrix -> draws of the mean paired difference -> p -> Holm
    -> verdict, with the same matrix for the three seeds."""
    n = 30
    rng = np.random.default_rng(5)
    base = rng.uniform(0.2, 0.3, size=n)
    matrix = inference.resample_matrix(n, B=400)
    per_seed = {}
    draws_by_seed = {}
    for k, seed in enumerate(SEEDS):
        ept = base - 0.05 - 0.01 * k                                    # EPT better in every match and seed
        comp = base
        per_seed[seed] = inference.bootstrap_stat(ept, np.ones(n), matrix) - inference.bootstrap_stat(comp, np.ones(n), matrix)
        draws_by_seed[seed] = per_seed[seed]
    diffs = {s: float((base - 0.05 - 0.01 * k).mean() - base.mean()) for k, s in enumerate(SEEDS)}
    observed = inference.paired_seed_mean(diffs)
    draws = inference.paired_seed_mean(draws_by_seed)
    p = inference.two_sided_p(draws, observed)
    assert p == pytest.approx(2 / 401)
    adjusted = inference.holm({'C1': p})['C1']['adjusted_p']
    assert inference.verdict(list(diffs.values()), adjusted_p=adjusted) == 'claim'
