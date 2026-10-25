"""Tests of ``c_ept.nll``: the NLL scores that every baseline and the later EPT share (stage S3 Task 5).

Truth comes from four places: the numbers of math-control note 03 section C (the Poisson NLL with exposure, ``TOY_*`` of ``s3_toys``) and D
(a bin with probability 0.4), a second implementation written here from the formulas (``scipy.stats.poisson.logpmf``, plain loops for the
bootstrap), small hand examples, and refusals of inputs that cannot be trusted (a count in a cell that was not observed, a probability row
that does not add up to 1, a NaN that would turn an average into NaN without a word).

The two tests of the task brief come first, word for word.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from c_ept.valuation import flowprice
from c_ept.models import nll
from c_ept.valuation import value
from c_ept.tests.s3_toys import TOY_CAL_W, TOY_EXP, TOY_GOLD, TOY_GOLD_BINS, TOY_POISSON_NLL, TOY_RATE, TOY_Y

SOURCE = Path(nll.__file__)


# ---- the two tests of the task brief -----------------------------------------------------------------------------------------------------
def test_zero_exposure_bins_add_nothing():
    v = nll.poisson(np.array([0, 0]), np.array([0.3, 0.3]), np.array([10.0, 0.0]))
    assert v[1] == 0.0 and np.isfinite(v).all()


def test_poisson_matches_toy_note():                                  # 검산 노트 03
    assert nll.poisson(TOY_Y, TOY_RATE, TOY_EXP).sum() == pytest.approx(TOY_POISSON_NLL)


# ---- Poisson with exposure ---------------------------------------------------------------------------------------------------------------
def test_poisson_cells_of_note_03_one_by_one():
    v = nll.poisson(TOY_Y, TOY_RATE, TOY_EXP)
    # cell 0: 0.2 - ln 0.2 = 1.80944; cell 1: 0.2; cell 2: not observed; cell 3: 0.5 - 2 ln 0.5 + ln 2 = 2.57944
    assert v == pytest.approx([0.2 - np.log(0.2), 0.2, 0.0, 0.5 - 2 * np.log(0.5) + np.log(2)])
    assert v[0] == pytest.approx(1.80944, abs=1e-5) and v[3] == pytest.approx(2.57944, abs=1e-5)


def test_poisson_note_03_try_it_yourself():                          # the exposure of cell 0 cut to 5 s: 0.1 - ln 0.1 = 2.40259
    v = nll.poisson(TOY_Y, TOY_RATE, (5.0, 10.0, 0.0, 10.0))
    assert v[0] == pytest.approx(2.40259, abs=1e-5) and v[0] == pytest.approx(0.1 - np.log(0.1))


def test_poisson_includes_log_y_factorial():
    # the term that does not depend on the rate: y = 4 at mu = 4 is 4 - 4 ln 4 + ln 24
    assert nll.poisson([4], [1.0], [4.0])[0] == pytest.approx(4 - 4 * np.log(4) + np.log(24))
    assert nll.poisson([0], [1.0], [4.0])[0] == pytest.approx(4.0)            # y = 0: mu - 0 + ln 0! = mu


def test_poisson_equals_minus_the_scipy_log_pmf_on_random_data():
    rng = np.random.default_rng(1)
    y = rng.poisson(3.0, 500)
    rate = rng.uniform(0.001, 0.3, 500)
    exposure = rng.uniform(0.5, 60.0, 500)
    assert np.allclose(nll.poisson(y, rate, exposure), -stats.poisson.logpmf(y, rate * exposure), rtol=1e-12, atol=0)


def test_poisson_large_counts_stay_accurate():
    y = np.array([100, 1000, 20_000])
    mu = np.array([90.0, 1100.0, 20_000.0])
    assert np.allclose(nll.poisson(y, mu, np.ones(3)), -stats.poisson.logpmf(y, mu), rtol=1e-10)


def test_poisson_exposure_scales_the_mean_not_the_count():
    one = nll.poisson([2], [0.05], [10.0])[0]
    assert one == pytest.approx(0.5 - 2 * np.log(0.5) + np.log(2))
    assert nll.poisson([2], [0.5], [1.0])[0] == pytest.approx(one)             # rate x exposure is all that counts


def test_poisson_a_zero_rate_with_exposure():
    v = nll.poisson([0, 1], [0.0, 0.0], [10.0, 10.0])
    assert v[0] == 0.0 and v[1] == np.inf                                      # a count that the model gave no chance: infinite loss


def test_poisson_a_count_in_an_unobserved_cell_is_refused():
    with pytest.raises(ValueError, match='no exposure'):
        nll.poisson([0, 1], [0.1, 0.1], [10.0, 0.0])


def test_poisson_the_rate_of_an_unobserved_cell_is_not_read():
    v = nll.poisson([0, 0, 0], [np.nan, np.inf, -1.0], [0.0, 0.0, 0.0])        # nothing was observed: nothing to score
    assert v.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize('y, rate, exposure, text', [
    ([-1], [0.1], [10.0], 'whole'), ([0.5], [0.1], [10.0], 'whole'), ([np.nan], [0.1], [10.0], 'whole'), ([np.inf], [0.1], [10.0], 'whole'),
    ([1], [-0.1], [10.0], 'rate'), ([1], [np.nan], [10.0], 'rate'), ([1], [np.inf], [10.0], 'rate'),
    ([1], [0.1], [-1.0], 'exposure'), ([1], [0.1], [np.nan], 'exposure'), ([1], [0.1], [np.inf], 'exposure'),
    ([0], [0.1], [-1.0], 'negative or not finite'), ([0], [0.1], [np.nan], 'negative or not finite'),      # with no count: not a silent 0
    ([0], [0.1], [np.inf], 'negative or not finite')])
def test_poisson_refuses_values_that_cannot_be_counts_rates_or_times(y, rate, exposure, text):
    with pytest.raises(ValueError, match=text):
        nll.poisson(y, rate, exposure)


def test_poisson_broadcasts_and_keeps_the_shape():
    y = np.array([[0, 1, 0], [3, 0, 0]])
    rate = np.full((2, 3), 0.05)
    out = nll.poisson(y, rate, 10.0)                                           # one exposure for every cell
    assert out.shape == (2, 3) and out.dtype == np.float64
    assert np.allclose(out, nll.poisson(y, rate, np.full((2, 3), 10.0)))
    per_column = nll.poisson(y, rate, np.array([[10.0, 10.0, 0.0]]))           # the third column was not observed in either row
    assert per_column.shape == (2, 3) and (per_column[:, 2] == 0).all() and np.array_equal(per_column[:, :2], out[:, :2])


def test_poisson_does_not_change_its_inputs():
    y, rate, exposure = np.array([1, 0, 2]), np.array([0.1, 0.2, 0.3]), np.array([10.0, 0.0, 5.0])
    before = [a.copy() for a in (y, rate, exposure)]
    nll.poisson(y, rate, exposure)
    assert all(np.array_equal(a, b) for a, b in zip((y, rate, exposure), before))


def test_poisson_returns_float64_for_every_input_dtype():
    out = nll.poisson(np.array([3], dtype=np.int8), np.array([0.5], dtype=np.float32), np.array([10], dtype=np.int32))
    assert out.dtype == np.float64 and out[0] == pytest.approx(5 - 3 * np.log(5) + np.log(6), rel=1e-12)       # mu = 5, ln 3! = ln 6


def test_poisson_a_scalar_cell():
    assert float(nll.poisson(1, 0.02, 10.0)) == pytest.approx(0.2 - np.log(0.2))


# ---- categorical and gold ----------------------------------------------------------------------------------------------------------------
def test_categorical_is_minus_the_log_of_the_chosen_probability():
    prob = np.array([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
    assert nll.categorical([0, 1, 1], prob) == pytest.approx([-np.log(0.7), -np.log(0.6), -np.log(0.9)])


def test_categorical_note_03_section_d():                                    # p(bin) = 0.4 gives 0.91629
    prob = np.array([[0.4, 0.6]])
    assert nll.categorical([0], prob)[0] == pytest.approx(0.91629, abs=1e-5)


def test_categorical_a_missing_category_adds_nothing_and_its_row_is_not_read():
    prob = np.array([[0.5, 0.5], [np.nan, np.nan], [0.2, 0.8]])                 # the middle row was never scored (no zone, no kill)
    out = nll.categorical([0, -1, 1], prob)
    assert out.tolist() == pytest.approx([-np.log(0.5), 0.0, -np.log(0.8)]) and np.isfinite(out).all()


def test_categorical_a_zero_probability_is_an_infinite_loss():
    out = nll.categorical([0, 1], np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert out.tolist() == [np.inf, 0.0]


def test_categorical_a_certain_outcome_costs_nothing():
    assert nll.categorical([2], np.array([[0.0, 0.0, 1.0]]))[0] == 0.0


def test_categorical_accepts_float32_probabilities_that_add_up_to_one_in_float32():
    rng = np.random.default_rng(3)
    p = rng.random((50, 52)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    out = nll.categorical(rng.integers(0, 52, 50), p)
    assert out.shape == (50,) and np.isfinite(out).all() and out.dtype == np.float64


@pytest.mark.parametrize('idx, prob, text', [
    ([2], [[0.5, 0.5]], 'category'), ([-2], [[0.5, 0.5]], 'category'), ([0, 1], [[0.5, 0.5]], 'rows'),
    ([0], [[0.5, 0.4]], 'add up'), ([0], [[0.6, 0.6]], 'add up'), ([0], [[-0.1, 1.1]], 'probabilit'), ([0], [[np.nan, 1.0]], 'probabilit'),
    ([0], [0.5, 0.5], 'table')])
def test_categorical_refuses_what_is_not_a_table_of_probabilities(idx, prob, text):
    with pytest.raises(ValueError, match=text):
        nll.categorical(idx, np.array(prob))


def test_categorical_refuses_a_non_integer_category_number():
    with pytest.raises(TypeError, match='integer'):
        nll.categorical(np.array([0.0]), np.array([[0.5, 0.5]]))
    with pytest.raises(TypeError, match='integer'):
        nll.categorical(np.array([True]), np.array([[0.5, 0.5]]))


def test_categorical_does_not_change_its_inputs():
    idx, prob = np.array([0, -1]), np.array([[0.5, 0.5], [0.9, 0.1]])
    before = idx.copy(), prob.copy()
    nll.categorical(idx, prob)
    assert np.array_equal(idx, before[0]) and np.array_equal(prob, before[1])


def test_gold_has_31_bins_and_a_missing_bin_adds_nothing():
    prob = np.full((3, 31), 1 / 31)
    out = nll.gold([5, -1, 30], prob)                                          # the second team had no kill: no gold category
    assert out.tolist() == pytest.approx([np.log(31), 0.0, np.log(31)])


def test_gold_note_03_section_d_bins_and_probability():
    assert value.gold_bin(np.array(TOY_GOLD)).tolist() == list(TOY_GOLD_BINS)
    prob = np.full((len(TOY_GOLD), 31), 0.6 / 30)
    prob[np.arange(len(TOY_GOLD)), TOY_GOLD_BINS] = 0.4                        # p(bin) = 0.4 for the bin that happened
    assert nll.gold(TOY_GOLD_BINS, prob) == pytest.approx([-np.log(0.4)] * len(TOY_GOLD))
    assert -np.log(0.4) == pytest.approx(0.91629, abs=1e-5)


@pytest.mark.parametrize('width', [30, 32, 2, 52])
def test_gold_refuses_a_table_that_is_not_31_wide(width):
    with pytest.raises(ValueError, match='31'):
        nll.gold([0], np.full((1, width), 1 / width))


def test_gold_equals_categorical_on_31_bins():
    rng = np.random.default_rng(4)
    p = rng.random((40, 31))
    p /= p.sum(axis=1, keepdims=True)
    idx = rng.integers(-1, 31, 40)
    assert np.array_equal(nll.gold(idx, p), nll.categorical(idx, p))


# ---- the weighted mean -------------------------------------------------------------------------------------------------------------------
def test_weighted_mean_hand_example():
    assert nll.weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]) == pytest.approx((1 + 2 + 6) / 4)


def test_weighted_mean_with_match_weights_of_note_03_section_f():
    # match A has four rows, match B two: the weights are 1/4 and 1/2, and the mean is the mean of the two match means
    v = np.array([0.0, 50.0, 50.0, 150.0, 150.0, 250.0])
    assert nll.weighted_mean(v, TOY_CAL_W) == pytest.approx(np.mean([np.mean(v[:4]), np.mean(v[4:])]))
    assert nll.weighted_mean(v, TOY_CAL_W) == pytest.approx(131.25)


def test_weighted_mean_of_equal_weights_is_the_plain_mean():
    v = np.random.default_rng(5).normal(size=77)
    assert nll.weighted_mean(v, np.full(77, 0.3)) == pytest.approx(v.mean(), rel=1e-12)


def test_weighted_mean_ignores_a_row_with_weight_zero_and_the_scale_of_the_weights():
    assert nll.weighted_mean([1.0, 100.0, 3.0], [1.0, 0.0, 1.0]) == pytest.approx(2.0)
    assert nll.weighted_mean([1.0, 2.0], [10.0, 30.0]) == nll.weighted_mean([1.0, 2.0], [1.0, 3.0])


def test_weighted_mean_does_not_depend_on_the_order_of_rows():
    rng = np.random.default_rng(6)
    v, w = rng.normal(size=500), rng.random(500)
    perm = rng.permutation(500)
    assert nll.weighted_mean(v, w) == nll.weighted_mean(v[perm], w[perm])       # exact: the sums are exact (math.fsum)


def test_weighted_mean_returns_a_python_float():
    assert type(nll.weighted_mean([1, 2], [1, 1])) is float


@pytest.mark.parametrize('values, w', [
    ([np.nan, 1.0], [1.0, 1.0]), ([np.inf, 1.0], [1.0, 1.0]), ([1.0, 2.0], [-1.0, 2.0]), ([1.0, 2.0], [np.nan, 1.0]),
    ([1.0, 2.0], [0.0, 0.0]), ([1.0, 2.0], [1.0]), ([], []), ([[1.0, 2.0]], [[1.0, 1.0]])])
def test_weighted_mean_refuses_what_it_cannot_average(values, w):
    with pytest.raises(ValueError):
        nll.weighted_mean(values, w)


# ---- the match bootstrap -----------------------------------------------------------------------------------------------------------------
def make_rows(n_matches=60, seed=0, rows=(1, 8)):
    rng = np.random.default_rng(seed)
    per = rng.integers(rows[0], rows[1] + 1, n_matches)
    match = np.repeat(np.arange(n_matches), per)
    value_ = rng.gamma(2.0, 1.0, len(match)) + 0.3 * rng.normal(size=n_matches)[match]
    return value_, match, nll_weights(match)


def nll_weights(match):
    """1/n_m, written here so that the tests of ``nll`` do not need ``features``."""
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def oracle_bootstrap(values, match, w, B, seed):
    """Replicate b is the b-th draw of one stream: the counts of ``default_rng(seed).multinomial(M, 1/M)`` over the matches in increasing
    order, applied to the weighted sums of each match, with plain loops."""
    ids = sorted(set(match.tolist()))
    num = [sum(w[i] * values[i] for i in range(len(values)) if match[i] == m) for m in ids]
    den = [sum(w[i] for i in range(len(values)) if match[i] == m) for m in ids]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        c = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)))
        out.append(sum(c[j] * num[j] for j in range(len(ids))) / sum(c[j] * den[j] for j in range(len(ids))))
    return np.array(out)


def test_bootstrap_has_b_replicates_and_is_deterministic():
    v, m, w = make_rows()
    a = nll.match_bootstrap(v, m, w, B=50, seed=7)
    assert a.shape == (50,) and a.dtype == np.float64 and np.isfinite(a).all()
    assert np.array_equal(a, nll.match_bootstrap(v, m, w, B=50, seed=7))
    assert not np.array_equal(a, nll.match_bootstrap(v, m, w, B=50, seed=8))


def test_replicate_b_is_the_b_th_draw_of_one_stream():
    v, m, w = make_rows(n_matches=12, seed=1)
    assert np.allclose(nll.match_bootstrap(v, m, w, B=40, seed=11), oracle_bootstrap(v, m, w, 40, 11), rtol=1e-12)


def test_the_first_replicates_do_not_depend_on_b():
    v, m, w = make_rows(n_matches=30, seed=2)
    assert np.array_equal(nll.match_bootstrap(v, m, w, B=10, seed=3), nll.match_bootstrap(v, m, w, B=60, seed=3)[:10])


def test_a_replicate_is_the_same_whatever_else_is_asked_for():
    v, m, w = make_rows(n_matches=40, seed=3)
    whole = nll.match_bootstrap(v, m, w, B=100, seed=5)
    for b in (1, 7, 33):                                                        # bit for bit, not only to rounding: no BLAS in the sums
        assert np.array_equal(whole[:b], nll.match_bootstrap(v, m, w, B=b, seed=5))


def test_bootstrap_of_one_match_is_its_mean_in_every_replicate():
    out = nll.match_bootstrap([1.0, 2.0, 6.0], [4, 4, 4], [1 / 3] * 3, B=20, seed=1)
    assert np.allclose(out, 3.0)


def test_bootstrap_of_matches_with_one_value_has_no_spread():
    v, m, _ = make_rows(n_matches=25, seed=4)
    out = nll.match_bootstrap(np.full(len(v), 2.5), m, nll_weights(m), B=30, seed=1)
    assert np.allclose(out, 2.5, rtol=1e-12)


def test_bootstrap_centres_on_the_weighted_mean_with_the_textbook_spread():
    rng = np.random.default_rng(9)
    n_matches = 400
    match_means = rng.normal(2.0, 1.0, n_matches)                              # one row per match: a replicate is a mean of 400 draws
    out = nll.match_bootstrap(match_means, np.arange(n_matches), np.ones(n_matches), B=2000, seed=20261021)
    se = match_means.std(ddof=0) / np.sqrt(n_matches)
    assert abs(out.mean() - match_means.mean()) < 4 * se / np.sqrt(2000)        # the Monte Carlo error of the mean of 2,000 replicates
    assert out.std() == pytest.approx(se, rel=0.1)


def test_two_heads_on_the_same_matches_get_the_same_resamples():
    v, m, w = make_rows(n_matches=50, seed=5)
    first = nll.match_bootstrap(v, m, w, B=80, seed=2)
    second = nll.match_bootstrap(2.0 * v + 3.0, m, w, B=80, seed=2)           # a weighted mean is affine in the values
    assert np.allclose(second, 2.0 * first + 3.0, rtol=1e-12)                  # so the replicates pair up exactly


def test_bootstrap_does_not_depend_on_the_order_of_rows():
    v, m, w = make_rows(n_matches=30, seed=6)
    perm = np.random.default_rng(1).permutation(len(v))
    assert np.allclose(nll.match_bootstrap(v, m, w, B=30, seed=4), nll.match_bootstrap(v[perm], m[perm], w[perm], B=30, seed=4),
                       rtol=1e-12)


def test_bootstrap_matches_are_taken_in_increasing_order_of_their_number():
    v, m, w = make_rows(n_matches=20, seed=7)
    a = nll.match_bootstrap(v, m, w, B=30, seed=6)
    assert np.allclose(a, nll.match_bootstrap(v, 1000 + 7 * m, w, B=30, seed=6), rtol=1e-12)       # an order-preserving relabelling
    assert np.allclose(a, nll.match_bootstrap(v, -50 + m, w, B=30, seed=6), rtol=1e-12)


def test_bootstrap_with_a_match_of_weight_zero():
    v, m, w = make_rows(n_matches=40, seed=10)
    w = np.where(m == 17, 0.0, w)
    v = np.where(m == 17, 1e6, v)                                              # match 17 has no weight: its value is in no replicate
    out = nll.match_bootstrap(v, m, w, B=100, seed=1)
    assert np.isfinite(out).all() and out.max() < 1e3
    assert np.allclose(out, oracle_bootstrap(v, m, w, 100, 1), rtol=1e-12)


def test_a_replicate_that_draws_only_matches_without_weight_is_nan():
    out = nll.match_bootstrap([1.0, 2.0], [0, 1], [0.0, 1.0], B=200, seed=1)    # P(both draws on match 0) = 1/4 per replicate
    assert np.isnan(out).any() and (out[~np.isnan(out)] == 2.0).all()


def test_bootstrap_defaults_are_the_ones_of_the_flow_price_bootstrap():
    assert (nll.BOOT_B, nll.BOOT_SEED) == (flowprice.BOOT_B, flowprice.BOOT_SEED) == (2000, 20261021)
    assert nll.N_GOLD_BINS == value.N_GOLD_BINS == 31
    v, m, w = make_rows(n_matches=10, seed=8)
    assert nll.match_bootstrap(v, m, w).shape == (2000,)
    assert np.array_equal(nll.match_bootstrap(v, m, w, B=5), nll.match_bootstrap(v, m, w, B=5, seed=20261021))


def test_bootstrap_signature():
    assert list(inspect.signature(nll.match_bootstrap).parameters) == ['values', 'match', 'w', 'B', 'seed']


@pytest.mark.parametrize('kwargs', [dict(B=0), dict(B=-3), dict(B=2.5), dict(B=True), dict(seed=-1), dict(seed=1.5)])
def test_bootstrap_refuses_bad_b_or_seed(kwargs):
    v, m, w = make_rows(n_matches=5)
    with pytest.raises((ValueError, TypeError)):
        nll.match_bootstrap(v, m, w, **kwargs)


@pytest.mark.parametrize('case', ['nan value', 'inf value', 'negative weight', 'length', 'empty', 'two dimensions', 'float match',
                                  'nan weight'])
def test_bootstrap_refuses_rows_it_cannot_resample(case):
    v, m, w = (np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]), np.array([0.5, 0.5, 1.0]))
    bad = {'nan value': (np.array([1.0, np.nan, 3.0]), m, w), 'inf value': (np.array([1.0, np.inf, 3.0]), m, w),
           'negative weight': (v, m, np.array([0.5, -0.5, 1.0])), 'length': (v, m[:2], w), 'empty': (v[:0], m[:0], w[:0]),
           'two dimensions': (v[None], m[None], w[None]), 'float match': (v, m.astype(float), w),
           'nan weight': (v, m, np.array([0.5, np.nan, 1.0]))}
    with pytest.raises((ValueError, TypeError)):
        nll.match_bootstrap(*bad[case], B=5, seed=1)


# ---- structure ---------------------------------------------------------------------------------------------------------------------------
def test_the_module_imports_only_numpy_scipy_and_the_standard_library():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or '').split('.')[0])
    assert modules <= {'__future__', 'math', 'numpy', 'scipy'}, modules


def test_the_module_has_no_holdout_switch_and_reads_no_store():
    text = SOURCE.read_text(encoding='utf-8')
    assert 'allow_holdout' not in text and 'matches/key' not in text and 's1.array' not in text
