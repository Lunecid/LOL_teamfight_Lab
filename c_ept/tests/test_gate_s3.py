"""Tests of ``c_ept.gate_s3``: the S3 composition gate metrics, the overdispersion test and the verdict (stage S3 Task 10).

Truth comes from six places: the numbers of math-control note 03 sections F and G (``TOY_CAL_*`` and ``TOY_PHI*`` of ``s3_toys``, and the
occurrence example of section F, written below with its exact fractions), a second implementation written here at the level of the rows
(a replicate is recomputed from the row weights w_i c_m with a least-squares solver, not from per-match sums), the shared bootstrap
(``nll.match_bootstrap``, whose draws every S3 interval must use: ruling R26), simulated data with a known answer (calibrated counts pass,
scaled values and negative-binomial counts fail), hand-written result dicts for the verdict, and refusals of inputs that cannot be trusted.

The four tests of the task brief come first, word for word. The synthetic helpers (``calibrated_counts``, ``miscalibrated_values``,
``negbin_counts``) are local to this file (ruling R25). The simulations are large enough (about 20,000 rows, 2,000 matches) that a model
which is right meets the pre-registered bands by 4 standard errors or more, so no test here depends on the luck of a seed.
"""
import ast
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from c_ept import gate_s3, nll
from c_ept.tests.s3_toys import TOY_CAL_BIAS, TOY_CAL_E, TOY_CAL_SLOPE, TOY_CAL_V, TOY_CAL_W, TOY_CAL_WSLOPE, TOY_PHI, TOY_PHI_Y

SOURCE = Path(gate_s3.__file__)


# ---- synthetic data (local to this file) -------------------------------------------------------------------------------------------------
def match_weights(match):
    """1/n_m, written here so that these tests do not depend on ``features``."""
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def make_matches(n_matches, rng, rows=(4, 16)):
    """The match number of every row (matches of 4 to 16 rows, as unequal as real ones) and the weights 1/n_m."""
    per = rng.integers(rows[0], rows[1] + 1, n_matches)
    match = np.repeat(np.arange(n_matches), per)
    return match, match_weights(match)


def calibrated_counts(seed, n_matches=2000, low=0.1, high=1.5, a=0.0, b=1.0):
    """(expected, observed, match, w): the observed count is Poisson with mean a + b x expected. a = 0 and b = 1 is a model that is
    calibrated by construction. The expected values carry a match-level factor that the model knows, so the rows of a match are alike."""
    rng = np.random.default_rng(seed)
    match, w = make_matches(n_matches, rng)
    level = rng.lognormal(0.0, 0.3, n_matches)[match]
    expected = rng.uniform(low, high, len(match)) * level
    return expected, rng.poisson(np.maximum(a + b * expected, 0.0)), match, w


def miscalibrated_values(scale, seed, n_matches=2000, shift=0.0):
    """(expected_v, realized_v, match, w) in gold: realized = scale x expected + shift + noise (a row part and a match part).
    scale = 1 and shift = 0 is calibrated."""
    rng = np.random.default_rng(seed)
    match, w = make_matches(n_matches, rng)
    expected = rng.normal(0.0, 200.0, len(match)) + 100.0 * rng.normal(size=n_matches)[match]
    noise = rng.normal(0.0, 600.0, len(match)) + 80.0 * rng.normal(size=n_matches)[match]
    return expected, scale * expected + shift + noise, match, w


def negbin_counts(alpha, seed, n_matches=400):
    """(y, mu, match, w): negative-binomial counts with mean mu and variance mu (1 + alpha mu), so that the Pearson ratio has mean
    1 + alpha mu. alpha = 0 gives Poisson counts. The weights are those of the gate, 1/n_m."""
    rng = np.random.default_rng(seed)
    match, w = make_matches(n_matches, rng)
    mu = rng.uniform(0.2, 2.5, len(match))
    if alpha == 0:
        return rng.poisson(mu), mu, match, w
    size = 1.0 / alpha
    return rng.negative_binomial(size, size / (size + mu)), mu, match, w


# the occurrence example of note 03 section F, with its exact fractions: sum(E) = 4 = sum(O), S_xx = 58/75, S_xy = 14/15
OCC_E = (0.2, 0.8, 0.2, 0.8, 0.8, 1.2)
OCC_O = (0, 1, 0, 1, 1, 1)
OCC_SLOPE = 35 / 29                                                 # (14/15) / (58/75) = 1.2069
OCC_INTERCEPT = -4 / 29                                             # 2/3 - slope x 2/3


# ---- a second implementation, at the level of the rows -----------------------------------------------------------------------------------
def oracle_replicates(columns, match, w, stat, B, seed):
    """B replicates of a row-level statistic. Replicate b is the b-th draw of one stream, ``default_rng(seed).multinomial(M, 1/M)`` with the
    matches in increasing order; row i gets the weight w_i c_m(i), and ``stat(columns of the rows with weight > 0, those weights)`` gives
    the replicate. No per-match sums: the statistic sees the rows."""
    ids = sorted(set(match.tolist()))
    position = {m: j for j, m in enumerate(ids)}
    index = np.array([position[m] for m in match.tolist()])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)))
        weight = w * counts[index]
        keep = weight > 0
        out.append(stat([c[keep] for c in columns], weight[keep]))
    return np.array(out)


def stat_oe(cols, weight):
    expected, observed = cols
    return [np.sum(weight * observed) / np.sum(weight * expected)]


def stat_line(cols, weight):
    """slope and intercept of the weighted least-squares line y = intercept + slope x, by a solver (not by the normal equations)."""
    x, y = cols
    root = np.sqrt(weight)
    design = np.column_stack([np.ones(len(x)), x]) * root[:, None]
    beta = np.linalg.lstsq(design, y * root, rcond=None)[0]
    return [beta[1], beta[0]]


def stat_mean_difference(cols, weight):
    x, y = cols
    return [np.sum(weight * (y - x)) / np.sum(weight)]


def stat_pearson(cols, weight):
    y, mu = cols
    return [np.sum(weight * (y - mu) ** 2 / mu) / np.sum(weight)]


def percentile_ci(reps):
    return [float(v) for v in np.percentile(reps, [2.5, 97.5])]


# ---- the four tests of the task brief, word for word -------------------------------------------------------------------------------------
def test_calibrated_counts_pass():
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=1), B=300, seed=1)
    assert gate_s3.verdict({'occurrence': r})['occurrence']['pass']


def test_scaled_values_fail_value_calibration():
    r = gate_s3.value_calibration(*miscalibrated_values(scale=1.5, seed=2), B=300, seed=2)
    assert not gate_s3.verdict({'value': r})['value']['pass']


def test_overdispersion_detects_negative_binomial():
    assert gate_s3.overdispersion(*negbin_counts(alpha=0.5, seed=3), B=200, seed=3)['phi_ci'][0] > 1.1


def test_thresholds_are_the_preregistered_ones():
    assert (gate_s3.OE_BAND, gate_s3.OCC_SLOPE_BAND, gate_s3.VALUE_SLOPE_BAND, gate_s3.BIAS_BAND,
            gate_s3.MC_SHARE, gate_s3.DISPERSION_LIMIT) == ((0.95, 1.05), (0.9, 1.1), (0.8, 1.2), (-50.0, 50.0), 0.99, 1.1)


# ---- the thresholds and the signatures ---------------------------------------------------------------------------------------------------
def test_the_bands_are_tuples_of_floats_and_the_limits_are_floats():
    for band in (gate_s3.OE_BAND, gate_s3.OCC_SLOPE_BAND, gate_s3.VALUE_SLOPE_BAND, gate_s3.BIAS_BAND):
        assert type(band) is tuple and len(band) == 2 and all(type(v) is float for v in band) and band[0] < band[1]
    assert type(gate_s3.MC_SHARE) is float and type(gate_s3.DISPERSION_LIMIT) is float


def test_the_bands_are_centred_on_the_value_of_perfect_calibration():
    # observed / expected and the slopes are 1 for a calibrated model, the mean error is 0 gold: every band is the symmetric neighbourhood
    for band, centre in ((gate_s3.OE_BAND, 1.0), (gate_s3.OCC_SLOPE_BAND, 1.0), (gate_s3.VALUE_SLOPE_BAND, 1.0), (gate_s3.BIAS_BAND, 0.0)):
        assert (band[0] + band[1]) / 2 == pytest.approx(centre)


def test_signatures_and_defaults():
    sig = inspect.signature
    assert list(sig(gate_s3.occurrence_calibration).parameters) == ['expected', 'observed', 'match', 'w', 'B', 'seed']
    assert list(sig(gate_s3.value_calibration).parameters) == ['expected_v', 'realized_v', 'match', 'w', 'B', 'seed']
    assert list(sig(gate_s3.mc_agreement).parameters) == ['mc_mean', 'mc_sd', 'S', 'analytic']
    assert list(sig(gate_s3.overdispersion).parameters) == ['y', 'mu', 'match', 'w', 'B', 'seed']
    assert list(sig(gate_s3.verdict).parameters) == ['results']
    for f in (gate_s3.occurrence_calibration, gate_s3.value_calibration, gate_s3.overdispersion):
        assert sig(f).parameters['B'].default == 2000 and sig(f).parameters['seed'].default == 20261021       # D7: B = 2,000, the seed
    assert sig(gate_s3.overdispersion).parameters['w'].default is None
    assert sig(gate_s3.occurrence_calibration).parameters['w'].default is inspect.Parameter.empty            # the gate always passes 1/n_m


def test_the_defaults_are_the_ones_of_the_shared_bootstrap():
    assert (nll.BOOT_B, nll.BOOT_SEED) == (2000, 20261021)
    y, mu, m, w = negbin_counts(alpha=0.2, seed=5, n_matches=20)
    r = gate_s3.overdispersion(y, mu, m, w)
    assert (r['B'], r['seed']) == (2000, 20261021)
    assert r == gate_s3.overdispersion(y, mu, m, w, B=2000, seed=20261021)


# ---- the numbers of math-control note 03 -------------------------------------------------------------------------------------------------
def test_value_slope_and_mean_error_of_note_03_f():
    r = gate_s3.value_calibration(TOY_CAL_E, TOY_CAL_V, np.arange(6), np.ones(6), B=50, seed=1)
    assert r['slope'] == pytest.approx(TOY_CAL_SLOPE, rel=1e-12) and r['slope'] == pytest.approx(0.875, rel=1e-12)
    assert r['bias'] == pytest.approx(TOY_CAL_BIAS, rel=1e-12) and r['bias'] == pytest.approx(25 / 3, rel=1e-12)


def test_weighted_slope_of_note_03_f():
    match = np.array([0, 0, 0, 0, 1, 1])                                # match A has four rows, match B two
    assert np.array_equal(match_weights(match), TOY_CAL_W)               # the weights of the note are 1/n_m
    r = gate_s3.value_calibration(TOY_CAL_E, TOY_CAL_V, match, TOY_CAL_W, B=50, seed=1)
    assert r['slope'] == pytest.approx(TOY_CAL_WSLOPE, rel=1e-12) and r['slope'] == pytest.approx(0.8864, abs=5e-5)
    # the weighted mean error by hand: V - E = (0, 50, -50, 50, -50, 50), sum(w (V - E)) = 0.25 x 50 + 0.5 x 0 = 12.5, sum(w) = 2
    assert r['bias'] == pytest.approx(12.5 / 2, rel=1e-12)


def test_the_line_of_note_03_f_through_the_occurrence_function():
    # the same six rows as counts (non-negative whole numbers): the line is V = 125/6 + 0.875 E, and observed / expected is 650 / 600
    r = gate_s3.occurrence_calibration(TOY_CAL_E, TOY_CAL_V, np.arange(6), np.ones(6), B=50, seed=1)
    assert r['slope'] == pytest.approx(0.875, rel=1e-12) and r['intercept'] == pytest.approx(125 / 6, rel=1e-12)
    assert r['oe'] == pytest.approx(650 / 600, rel=1e-12)


def test_occurrence_example_of_note_03_f():
    r = gate_s3.occurrence_calibration(OCC_E, OCC_O, np.arange(6), np.ones(6), B=50, seed=1)
    assert r['oe'] == pytest.approx(1.0, rel=1e-12)                      # 4 / 4
    assert r['slope'] == pytest.approx(OCC_SLOPE, rel=1e-12) and r['slope'] == pytest.approx(1.207, abs=5e-4)
    assert r['intercept'] == pytest.approx(OCC_INTERCEPT, rel=1e-12)


def test_pearson_phi_of_note_03_g():
    r = gate_s3.overdispersion(TOY_PHI_Y, np.ones(6), np.arange(6), B=50, seed=1)           # equal weights by default
    assert r['phi'] == pytest.approx(TOY_PHI, rel=1e-12) and r['phi'] == pytest.approx(4 / 3, rel=1e-12)
    assert r['n_rows'] == 6 and r['n_zero_mu'] == 0
    assert r['phi'] == pytest.approx(gate_s3.overdispersion(TOY_PHI_Y, np.ones(6), np.arange(6), np.ones(6), B=50, seed=1)['phi'])


def test_pearson_phi_of_note_03_g_with_the_weights_of_the_gate():
    # three matches of two rows: the weights 1/n_m are all 1/2, so phi does not change
    r = gate_s3.overdispersion(TOY_PHI_Y, np.ones(6), np.array([0, 0, 1, 1, 2, 2]), np.full(6, 0.5), B=50, seed=1)
    assert r['phi'] == pytest.approx(4 / 3, rel=1e-12)
    # unequal matches: weights (1/4 x 4, 1/2 x 2): by hand, sum(w (y - 1)^2) = 0.25 (1 + 1 + 4 + 1) + 0.5 (0 + 1) = 2.25, sum(w) = 2
    r = gate_s3.overdispersion(TOY_PHI_Y, np.ones(6), np.array([0, 0, 0, 0, 1, 1]), TOY_CAL_W, B=50, seed=1)
    assert r['phi'] == pytest.approx(2.25 / 2, rel=1e-12)


# ---- the intervals are the percentiles of match-bootstrap replicates, checked against a second implementation ----------------------------
def test_occurrence_intervals_are_the_percentiles_of_row_level_replicates():
    e, o, m, w = calibrated_counts(seed=11, n_matches=25)
    r = gate_s3.occurrence_calibration(e, o, m, w, B=60, seed=5)
    oe = oracle_replicates([e, o], m, w, stat_oe, 60, 5)
    line = oracle_replicates([e, o], m, w, stat_line, 60, 5)
    assert r['oe_ci'] == pytest.approx(percentile_ci(oe[:, 0]), rel=1e-9)
    assert r['slope_ci'] == pytest.approx(percentile_ci(line[:, 0]), rel=1e-9)
    assert r['intercept_ci'] == pytest.approx(percentile_ci(line[:, 1]), rel=1e-7, abs=1e-9)
    # and the estimates themselves are the statistics of the whole sample
    assert r['oe'] == pytest.approx(stat_oe([e, o], w)[0], rel=1e-12)
    assert [r['slope'], r['intercept']] == pytest.approx(stat_line([e, o], w), rel=1e-9, abs=1e-12)


def test_value_intervals_are_the_percentiles_of_row_level_replicates():
    ev, rv, m, w = miscalibrated_values(scale=1.2, seed=12, n_matches=25, shift=30.0)
    r = gate_s3.value_calibration(ev, rv, m, w, B=60, seed=6)
    line = oracle_replicates([ev, rv], m, w, stat_line, 60, 6)
    bias = oracle_replicates([ev, rv], m, w, stat_mean_difference, 60, 6)
    assert r['slope_ci'] == pytest.approx(percentile_ci(line[:, 0]), rel=1e-9)
    assert r['bias_ci'] == pytest.approx(percentile_ci(bias[:, 0]), rel=1e-9)
    assert [r['slope'], r['bias']] == pytest.approx([stat_line([ev, rv], w)[0], stat_mean_difference([ev, rv], w)[0]], rel=1e-9)


def test_phi_interval_is_the_percentiles_of_row_level_replicates():
    y, mu, m, w = negbin_counts(alpha=0.4, seed=13, n_matches=25)
    r = gate_s3.overdispersion(y, mu, m, w, B=60, seed=7)
    reps = oracle_replicates([y, mu], m, w, stat_pearson, 60, 7)
    assert r['phi_ci'] == pytest.approx(percentile_ci(reps[:, 0]), rel=1e-9)
    assert r['phi'] == pytest.approx(stat_pearson([y, mu], w)[0], rel=1e-12)


def test_the_intervals_are_95_percent_percentile_intervals():
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=14, n_matches=30)
    r = gate_s3.value_calibration(ev, rv, m, w, B=200, seed=3)
    reps = oracle_replicates([ev, rv], m, w, stat_mean_difference, 200, 3)[:, 0]
    lo, hi = np.percentile(reps, 2.5), np.percentile(reps, 97.5)
    assert r['bias_ci'] == pytest.approx([lo, hi], rel=1e-9)
    assert np.sum(reps < lo) == 5 and np.sum(reps > hi) == 5                      # 2.5 per cent of 200 replicates on each side


def test_bias_interval_is_the_one_of_the_shared_match_bootstrap():
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=6, n_matches=50)
    r = gate_s3.value_calibration(ev, rv, m, w, B=100, seed=9)
    reps = nll.match_bootstrap(rv - ev, m, w, B=100, seed=9)                       # the draws of every other S3 interval (ruling R26)
    assert r['bias_ci'] == pytest.approx(percentile_ci(reps), rel=1e-9)
    assert r['bias'] == pytest.approx(nll.weighted_mean(rv - ev, w), rel=1e-12)


def test_phi_interval_is_the_one_of_the_shared_match_bootstrap():
    y, mu, m, w = negbin_counts(alpha=0.3, seed=7, n_matches=50)
    r = gate_s3.overdispersion(y, mu, m, w, B=100, seed=9)
    reps = nll.match_bootstrap((y - mu) ** 2 / mu, m, w, B=100, seed=9)
    assert r['phi_ci'] == pytest.approx(percentile_ci(reps), rel=1e-9)
    assert r['phi'] == pytest.approx(nll.weighted_mean((y - mu) ** 2 / mu, w), rel=1e-12)


def test_oe_interval_is_the_one_of_the_shared_bootstrap_when_it_is_a_weighted_mean():
    # a weighted mean of the ratios o/e with the weights w e is sum(w o) / sum(w e): the same replicates, from the shared function
    e, o, m, w = calibrated_counts(seed=15, n_matches=40, low=0.3)
    reps = nll.match_bootstrap(o / e, m, w * e, B=80, seed=4)
    r = gate_s3.occurrence_calibration(e, o, m, w, B=80, seed=4)
    assert r['oe_ci'] == pytest.approx(percentile_ci(reps), rel=1e-9)


# ---- determinism, seeds and the order of the rows ----------------------------------------------------------------------------------------
def test_results_are_deterministic_and_depend_on_the_seed():
    e, o, m, w = calibrated_counts(seed=3, n_matches=40)
    a = gate_s3.occurrence_calibration(e, o, m, w, B=80, seed=7)
    assert a == gate_s3.occurrence_calibration(e, o, m, w, B=80, seed=7)
    other = gate_s3.occurrence_calibration(e, o, m, w, B=80, seed=8)
    assert other['oe_ci'] != a['oe_ci'] and other['slope_ci'] != a['slope_ci']
    assert other['oe'] == a['oe'] and other['slope'] == a['slope']                  # the estimates are not random
    ev, rv, m, w = miscalibrated_values(scale=1.1, seed=3, n_matches=40)
    assert gate_s3.value_calibration(ev, rv, m, w, B=80, seed=7) == gate_s3.value_calibration(ev, rv, m, w, B=80, seed=7)
    y, mu, m, w = negbin_counts(alpha=0.5, seed=3, n_matches=40)
    assert gate_s3.overdispersion(y, mu, m, w, B=80, seed=7) == gate_s3.overdispersion(y, mu, m, w, B=80, seed=7)


def test_more_replicates_change_the_interval_but_not_the_estimate():
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=3, n_matches=40)
    a = gate_s3.value_calibration(ev, rv, m, w, B=50, seed=2)
    b = gate_s3.value_calibration(ev, rv, m, w, B=300, seed=2)
    assert a['slope'] == b['slope'] and a['bias'] == b['bias'] and a['slope_ci'] != b['slope_ci']


def test_results_do_not_depend_on_the_order_of_the_rows():
    e, o, m, w = calibrated_counts(seed=4, n_matches=40)
    p = np.random.default_rng(1).permutation(len(e))
    assert gate_s3.occurrence_calibration(e, o, m, w, B=80, seed=2) == gate_s3.occurrence_calibration(e[p], o[p], m[p], w[p], B=80, seed=2)
    ev, rv, m, w = miscalibrated_values(scale=1.3, seed=4, n_matches=40)
    p = np.random.default_rng(2).permutation(len(ev))
    assert gate_s3.value_calibration(ev, rv, m, w, B=80, seed=2) == gate_s3.value_calibration(ev[p], rv[p], m[p], w[p], B=80, seed=2)
    y, mu, m, w = negbin_counts(alpha=0.4, seed=4, n_matches=40)
    p = np.random.default_rng(3).permutation(len(y))
    assert gate_s3.overdispersion(y, mu, m, w, B=80, seed=2) == gate_s3.overdispersion(y[p], mu[p], m[p], w[p], B=80, seed=2)


def test_results_do_not_depend_on_the_labels_of_the_matches_only_on_their_order():
    e, o, m, w = calibrated_counts(seed=5, n_matches=30)
    a = gate_s3.occurrence_calibration(e, o, m, w, B=60, seed=2)
    assert a == gate_s3.occurrence_calibration(e, o, 1000 + 7 * m, w, B=60, seed=2)         # an order-preserving relabelling
    assert a == gate_s3.occurrence_calibration(e, o, m - 50, w, B=60, seed=2)
    assert a == gate_s3.occurrence_calibration(e, o, m.astype(np.int32), w, B=60, seed=2)


def test_a_change_of_the_weight_scale_changes_nothing():
    ev, rv, m, w = miscalibrated_values(scale=1.2, seed=5, n_matches=30)
    a = gate_s3.value_calibration(ev, rv, m, w, B=60, seed=2)
    b = gate_s3.value_calibration(ev, rv, m, 8.0 * w, B=60, seed=2)                     # a power of two: exact in floating point
    for key in ('slope', 'slope_ci', 'bias', 'bias_ci'):
        assert a[key] == b[key], key
    assert [d['mean_expected'] for d in a['deciles']] == [d['mean_expected'] for d in b['deciles']]
    assert [8.0 * d['weight'] for d in a['deciles']] == [d['weight'] for d in b['deciles']]          # only the weight itself scales


def test_inputs_may_be_lists_or_integer_arrays():
    a = gate_s3.occurrence_calibration(OCC_E, OCC_O, [0, 0, 1, 1, 2, 2], [0.5] * 6, B=40, seed=1)
    b = gate_s3.occurrence_calibration(np.array(OCC_E), np.array(OCC_O, dtype=np.int64), np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8),
                                       np.full(6, 0.5), B=40, seed=1)
    assert a == b


def test_inputs_are_not_modified_and_may_be_read_only():
    # the runner may pass views of memory-mapped arrays, which are read-only
    cases = [(gate_s3.occurrence_calibration, calibrated_counts(seed=51, n_matches=20)),
             (gate_s3.value_calibration, miscalibrated_values(scale=1.0, seed=51, n_matches=20)),
             (gate_s3.overdispersion, negbin_counts(alpha=0.3, seed=51, n_matches=20))]
    for fn, data in cases:
        before = [a.copy() for a in data]
        for a in data:
            a.flags.writeable = False
        fn(*data, B=10, seed=1)
        assert all(np.array_equal(a, b) for a, b in zip(data, before)), fn.__name__
    mean, sd, analytic = np.array([1.0, 2.0]), np.array([1.0, 0.0]), np.array([1.5, 2.0])
    before = [a.copy() for a in (mean, sd, analytic)]
    for a in (mean, sd, analytic):
        a.flags.writeable = False
    gate_s3.mc_agreement(mean, sd, 100, analytic)
    assert all(np.array_equal(a, b) for a, b in zip((mean, sd, analytic), before))


# ---- the textbook behaviour of the intervals ---------------------------------------------------------------------------------------------
def test_slope_interval_has_the_textbook_width_for_independent_rows():
    rng = np.random.default_rng(21)
    n = 400
    x = rng.normal(0.0, 1.0, n)
    y = 2.0 * x + rng.normal(0.0, 1.5, n)
    r = gate_s3.value_calibration(x, y, np.arange(n), np.ones(n), B=2000, seed=20261021)       # one row per match: the ordinary bootstrap
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    resid = y - y.mean() - slope * (x - x.mean())
    se = np.sqrt(np.sum(resid ** 2) / (n - 2)) / np.sqrt(np.sum((x - x.mean()) ** 2))           # the OLS standard error
    assert r['slope'] == pytest.approx(slope, rel=1e-12)
    assert (r['slope_ci'][1] - r['slope_ci'][0]) / 3.92 == pytest.approx(se, rel=0.15)
    assert r['slope_ci'][0] < r['slope'] < r['slope_ci'][1]


def test_oe_interval_has_the_delta_method_width_for_independent_rows():
    rng = np.random.default_rng(22)
    n = 3000
    e = rng.uniform(0.2, 1.8, n)
    o = rng.poisson(e)
    r = gate_s3.occurrence_calibration(e, o, np.arange(n), np.ones(n), B=2000, seed=20261021)
    ratio = o.sum() / e.sum()
    se = np.sqrt(np.sum((o - ratio * e) ** 2)) / e.sum()                                         # delta method for a ratio of sums
    assert r['oe'] == pytest.approx(ratio, rel=1e-12)
    assert (r['oe_ci'][1] - r['oe_ci'][0]) / 3.92 == pytest.approx(se, rel=0.15)


def test_clustered_rows_widen_the_interval_against_independent_rows():
    # the same 600 numbers as 60 matches of 10 rows (a match effect in the noise) and as 600 matches of one row: the match bootstrap
    # of the clusters is wider, which is the reason the gate resamples matches
    rng = np.random.default_rng(23)
    effect = np.repeat(rng.normal(0.0, 150.0, 60), 10)
    e = rng.normal(0.0, 200.0, 600)
    v = e + effect + rng.normal(0.0, 100.0, 600)
    clustered = gate_s3.value_calibration(e, v, np.repeat(np.arange(60), 10), np.full(600, 0.1), B=1000, seed=1)
    flat = gate_s3.value_calibration(e, v, np.arange(600), np.ones(600), B=1000, seed=1)

    def width(r):
        return r['bias_ci'][1] - r['bias_ci'][0]
    assert width(clustered) > 2.0 * width(flat)


def test_slope_and_mean_error_survive_a_huge_common_offset():
    # a slope from raw sums loses its digits when the values sit far from 0 (sum x^2 and n mean^2 almost cancel); the gate centres first
    ev, rv, m, w = miscalibrated_values(scale=1.25, seed=24, n_matches=40, shift=20.0)
    base = gate_s3.value_calibration(ev, rv, m, w, B=100, seed=2)
    big = gate_s3.value_calibration(ev + 1e9, rv + 1e9, m, w, B=100, seed=2)
    assert big['slope'] == pytest.approx(base['slope'], rel=1e-6)
    assert big['bias'] == pytest.approx(base['bias'], abs=1e-5)                 # (v + c) - (e + c) = v - e
    assert big['slope_ci'] == pytest.approx(base['slope_ci'], rel=1e-5)
    assert big['bias_ci'] == pytest.approx(base['bias_ci'], abs=1e-4)


def test_expected_counts_with_a_large_offset_keep_the_line():
    e, o, m, w = calibrated_counts(seed=25, n_matches=40)
    base = gate_s3.occurrence_calibration(e, o, m, w, B=100, seed=2)
    off = gate_s3.occurrence_calibration(e + 1e8, o + 1e8, m, w, B=100, seed=2)
    assert off['slope'] == pytest.approx(base['slope'], rel=1e-6)
    # y' = y + c and x' = x + c: the slope is the same and the intercept moves by c (1 - slope)
    assert off['intercept'] == pytest.approx(base['intercept'] + 1e8 * (1.0 - base['slope']), rel=1e-6, abs=1.0)


# ---- simulated models with a known answer ------------------------------------------------------------------------------------------------
def test_calibrated_counts_have_oe_and_slope_near_one():
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=1), B=300, seed=1)
    assert abs(r['oe'] - 1.0) < 0.03 and abs(r['slope'] - 1.0) < 0.08 and abs(r['intercept']) < 0.08          # 4 standard errors or more
    assert r['oe_ci'][0] < r['oe'] < r['oe_ci'][1] and r['slope_ci'][0] < r['slope'] < r['slope_ci'][1]
    assert r['oe_ci'][1] - r['oe_ci'][0] < 0.04 and r['slope_ci'][1] - r['slope_ci'][0] < 0.12          # intervals narrower than the bands


def test_calibrated_counts_pass_for_every_one_of_twenty_seeds():
    # the pass of the brief's test is not luck of seed 1: the simulation is large enough that the bands are met by a calibrated model
    for seed in range(1, 21):
        r = gate_s3.occurrence_calibration(*calibrated_counts(seed=seed), B=200, seed=seed)
        assert gate_s3.verdict({'occurrence': r})['occurrence']['pass'], seed


def test_calibrated_values_pass_and_the_estimates_are_near_the_truth():
    r = gate_s3.value_calibration(*miscalibrated_values(scale=1.0, seed=2), B=300, seed=2)
    assert gate_s3.verdict({'value': r})['value']['pass']
    assert abs(r['slope'] - 1.0) < 0.1 and abs(r['bias']) < 25.0


def test_calibrated_values_pass_for_every_one_of_twenty_seeds():
    for seed in range(1, 21):
        r = gate_s3.value_calibration(*miscalibrated_values(scale=1.0, seed=seed), B=200, seed=seed)
        assert gate_s3.verdict({'value': r})['value']['pass'], seed


def test_scaled_values_have_the_scale_as_their_slope():
    r = gate_s3.value_calibration(*miscalibrated_values(scale=1.5, seed=2), B=300, seed=2)
    assert r['slope_ci'][0] > 1.2 and abs(r['slope'] - 1.5) < 0.15


def test_a_shift_of_the_values_fails_the_mean_error_band_only():
    r = gate_s3.value_calibration(*miscalibrated_values(scale=1.0, seed=2, shift=120.0), B=300, seed=2)
    v = gate_s3.verdict({'value': r})['value']
    assert v['slope_in_band'] and not v['bias_in_band'] and not v['pass']
    assert r['bias'] == pytest.approx(120.0, abs=40.0) and r['bias_ci'][0] > 50.0


def test_counts_that_are_too_high_fail_the_oe_band_only():
    # the observed count has the mean 0.15 + expected, so that observed / expected is about 1.18 and the slope is still 1
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=1, a=0.15), B=300, seed=1)
    v = gate_s3.verdict({'occurrence': r})['occurrence']
    assert v['slope_in_band'] and not v['oe_in_band'] and not v['pass']
    assert r['oe_ci'][0] > 1.05


def test_a_steeper_line_with_equal_totals_fails_the_slope_band_only():
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=1, low=0.5, high=1.5, a=-0.315, b=1.3), B=300, seed=1)
    v = gate_s3.verdict({'occurrence': r})['occurrence']
    assert v['oe_in_band'] and not v['slope_in_band'] and not v['pass']
    assert r['slope_ci'][0] > 1.1


def test_poisson_counts_are_not_overdispersed_and_negative_binomial_counts_are():
    poisson = gate_s3.overdispersion(*negbin_counts(alpha=0, seed=8), B=300, seed=1)
    assert abs(poisson['phi'] - 1.0) < 0.1 and poisson['phi_ci'][0] < 1.1 and poisson['overdispersed'] is False
    nb = gate_s3.overdispersion(*negbin_counts(alpha=0.5, seed=3), B=300, seed=1)
    assert nb['overdispersed'] is True and nb['phi_ci'][0] > 1.1 and nb['phi'] > 1.4


def test_phi_estimates_one_plus_alpha_mu():
    # E[(y - mu)^2 / mu] = 1 + alpha mu for the negative binomial with variance mu (1 + alpha mu)
    for alpha in (0.25, 0.5):
        y, mu, m, w = negbin_counts(alpha=alpha, seed=9, n_matches=800)
        r = gate_s3.overdispersion(y, mu, m, w, B=300, seed=2)
        target = 1.0 + alpha * nll.weighted_mean(mu, w)
        assert abs(r['phi'] - target) < 0.2, alpha                                    # 5 standard errors
        assert 0.0 < r['phi_ci'][1] - r['phi_ci'][0] < 0.4, alpha


def test_overdispersed_means_the_lower_end_is_above_the_limit():
    y, mu, m, w = negbin_counts(alpha=0.5, seed=3)
    r = gate_s3.overdispersion(y, mu, m, w, B=200, seed=3)
    assert r['overdispersed'] is (r['phi_ci'][0] > gate_s3.DISPERSION_LIMIT)
    assert r['overdispersed'] is True


def test_overdispersed_needs_the_lower_end_strictly_above_the_limit():
    # one match, so that every replicate is the sample and the interval is [phi, phi]: phi = (4 + 7 x 1 + 2 x 0) / 10 = 1.1 exactly
    one_match = np.zeros(10, dtype=np.int64)
    r = gate_s3.overdispersion(np.array([3, 0, 0, 0, 0, 0, 0, 0, 1, 1]), np.ones(10), one_match, B=20, seed=1)
    assert r['phi'] == 1.1 and r['phi_ci'] == [1.1, 1.1] and r['overdispersed'] is False
    r = gate_s3.overdispersion(np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 1]), np.ones(10), one_match, B=20, seed=1)       # phi = 12 / 10
    assert r['phi'] == pytest.approx(1.2) and r['overdispersed'] is True


# ---- the deciles -------------------------------------------------------------------------------------------------------------------------
def test_deciles_of_a_hundred_distinct_rows():
    x = np.arange(1.0, 101.0)
    y = x + (np.arange(100) % 3)
    w = np.random.default_rng(1).uniform(0.5, 2.0, 100)
    r = gate_s3.value_calibration(x, y, np.arange(100) // 5, w, B=20, seed=1)
    table = r['deciles']
    assert len(table) == 10 and [d['count'] for d in table] == [10] * 10
    for k, d in enumerate(table):
        rows = slice(10 * k, 10 * k + 10)                                      # bin k holds the expected values 10 k + 1 .. 10 k + 10
        assert d['mean_expected'] == pytest.approx(np.sum(w[rows] * x[rows]) / np.sum(w[rows]), rel=1e-12)
        assert d['mean_realized'] == pytest.approx(np.sum(w[rows] * y[rows]) / np.sum(w[rows]), rel=1e-12)
        assert d['weight'] == pytest.approx(np.sum(w[rows]), rel=1e-12)
    assert [d['mean_expected'] for d in table] == sorted(d['mean_expected'] for d in table)


def test_occurrence_deciles_have_observed_means():
    e, o, m, w = calibrated_counts(seed=31, n_matches=60)
    r = gate_s3.occurrence_calibration(e, o, m, w, B=20, seed=1)
    table = r['deciles']
    assert len(table) == 10 and sum(d['count'] for d in table) == len(e) == r['n_rows']
    assert sum(d['weight'] for d in table) == pytest.approx(w.sum(), rel=1e-12)
    assert sum(d['weight'] * d['mean_observed'] for d in table) == pytest.approx(np.sum(w * o), rel=1e-12)    # the bins partition the rows
    assert sum(d['weight'] * d['mean_expected'] for d in table) == pytest.approx(np.sum(w * e), rel=1e-12)
    assert [d['mean_expected'] for d in table] == sorted(d['mean_expected'] for d in table)
    assert all(set(d) == {'count', 'weight', 'mean_expected', 'mean_observed'} for d in table)


def test_deciles_never_split_equal_expected_values():
    x = np.concatenate([np.zeros(60), np.linspace(1.0, 40.0, 40)])             # 60 per cent of the rows have the same expected value
    y = np.concatenate([np.zeros(60), np.ones(40)])
    r = gate_s3.value_calibration(x, y, np.arange(100) // 4, np.ones(100), B=10, seed=1)
    table = r['deciles']
    assert sum(d['count'] for d in table) == 100 and max(d['count'] for d in table) >= 60                 # all the zeros sit in one bin
    empty = [d for d in table if d['count'] == 0]
    assert empty and all(d['mean_expected'] is None and d['mean_realized'] is None and d['weight'] == 0.0 for d in empty)


def test_deciles_are_half_open_and_their_edges_are_exact():
    x = np.arange(11.0)                                # the 10th ... 90th percentiles are exactly 1, 2 ... 9: a row on an edge goes up
    table = gate_s3.value_calibration(x, x + 1.0, np.arange(11) // 3, np.ones(11), B=10, seed=1)['deciles']
    assert [d['count'] for d in table] == [1] * 9 + [2]
    assert [d['mean_expected'] for d in table] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5]


def test_decile_bins_agree_with_the_numpy_percentiles_when_no_row_is_on_an_edge():
    rng = np.random.default_rng(5)
    for n in (7, 10, 25, 100, 1000):
        x = rng.normal(size=n)
        edges = np.quantile(x, np.arange(1, 10) / 10)
        counts = np.bincount(np.searchsorted(edges, x, side='right'), minlength=10)
        table = gate_s3.value_calibration(x, 2.0 * x, np.arange(n) // 3, np.ones(n), B=5, seed=1)['deciles']
        assert [d['count'] for d in table] == counts.tolist(), n


def test_deciles_are_the_same_for_any_order_of_the_rows():
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=32, n_matches=30)
    p = np.random.default_rng(3).permutation(len(ev))
    assert gate_s3.value_calibration(ev, rv, m, w, B=10, seed=1)['deciles'] == \
        gate_s3.value_calibration(ev[p], rv[p], m[p], w[p], B=10, seed=1)['deciles']


def test_a_bin_of_rows_without_weight_has_no_mean():
    x = np.arange(1.0, 21.0)
    y = x + 1.0
    w = np.ones(20)
    w[:2] = 0.0                                                                  # the two rows of the lowest bin carry no weight
    r = gate_s3.value_calibration(x, y, np.arange(20) // 4, w, B=10, seed=1)
    assert r['deciles'][0]['count'] == 2 and r['deciles'][0]['weight'] == 0.0 and r['deciles'][0]['mean_expected'] is None


# ---- counts in the output ----------------------------------------------------------------------------------------------------------------
def test_row_and_match_counts():
    e, o, m, w = calibrated_counts(seed=33, n_matches=37)
    r = gate_s3.occurrence_calibration(e, o, m, w, B=10, seed=1)
    assert r['n_rows'] == len(e) and r['n_matches'] == 37 and (r['B'], r['seed']) == (10, 1)
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=33, n_matches=21)
    r = gate_s3.value_calibration(ev, rv, m, w, B=10, seed=2)
    assert r['n_rows'] == len(ev) and r['n_matches'] == 21 and (r['B'], r['seed']) == (10, 2)


def test_the_output_keys_are_those_of_the_brief():
    e, o, m, w = calibrated_counts(seed=34, n_matches=10)
    r = gate_s3.occurrence_calibration(e, o, m, w, B=10, seed=1)
    assert {'oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci', 'deciles', 'n_rows', 'n_matches'} <= set(r)
    ev, rv, m, w = miscalibrated_values(scale=1.0, seed=34, n_matches=10)
    r = gate_s3.value_calibration(ev, rv, m, w, B=10, seed=1)
    assert {'slope', 'slope_ci', 'bias', 'bias_ci', 'deciles', 'n_rows', 'n_matches'} <= set(r)
    y, mu, m, w = negbin_counts(alpha=0.3, seed=34, n_matches=10)
    r = gate_s3.overdispersion(y, mu, m, w, B=10, seed=1)
    assert {'phi', 'phi_ci', 'overdispersed', 'n_rows'} <= set(r)
    r = gate_s3.mc_agreement([1.0], [1.0], 10, [1.0])
    assert {'share', 'n_rows', 'worst_z'} <= set(r)


# ---- replicates that have no value -------------------------------------------------------------------------------------------------------
def test_replicates_without_a_slope_make_the_interval_nan_and_the_verdict_fail():
    # two matches whose expected values are constant inside each match: the resamples that draw one match twice have no slope
    e = np.array([1.0, 1.0, 3.0, 3.0])
    o = np.array([1, 2, 3, 5])
    r = gate_s3.occurrence_calibration(e, o, np.array([0, 0, 1, 1]), np.full(4, 0.5), B=100, seed=1)
    assert np.isfinite(r['slope']) and math.isnan(r['slope_ci'][0]) and math.isnan(r['slope_ci'][1])
    assert np.isfinite(r['oe_ci']).all()                                       # the ratio of sums exists in every replicate
    v = gate_s3.verdict({'occurrence': r})['occurrence']
    assert not v['slope_in_band'] and not v['pass']


def test_resamples_whose_expected_values_vary_only_by_rounding_have_no_slope():
    # two matches with one expected value each (2.607 and 0.801): a resample that draws one match twice has no spread in x, but its sums
    # leave a variance of 1.1e-16 > 0 from rounding (for both matches), and the quotient of two rounding errors looks like a slope
    e = np.array([2.607] * 3 + [0.801] * 3)
    o = np.array([2, 1, 1, 0, 3, 0])
    match = np.array([0, 0, 0, 1, 1, 1])
    r = gate_s3.occurrence_calibration(e, o, match, match_weights(match), B=100, seed=1)
    assert np.isfinite(r['slope']) and np.isfinite(r['intercept'])
    assert math.isnan(r['slope_ci'][0]) and math.isnan(r['slope_ci'][1]) and math.isnan(r['intercept_ci'][0])


def test_a_single_match_gives_a_degenerate_interval_at_the_estimate():
    ev, rv, _, _ = miscalibrated_values(scale=1.0, seed=35, n_matches=5)
    r = gate_s3.value_calibration(ev, rv, np.zeros(len(ev), dtype=np.int64), np.ones(len(ev)), B=30, seed=1)
    assert r['n_matches'] == 1
    assert r['slope_ci'] == pytest.approx([r['slope'], r['slope']], rel=1e-9) and r['bias_ci'] == pytest.approx([r['bias']] * 2, rel=1e-9)


# ---- refusals ----------------------------------------------------------------------------------------------------------------------------
def good_rows():
    e = np.array([0.5, 1.0, 1.5, 2.0])
    return e, np.array([0, 1, 2, 2]), np.array([0, 0, 1, 1]), np.full(4, 0.5)


BAD_OCCURRENCE = {
    'nan expected': lambda e, o, m, w: (np.array([0.5, np.nan, 1.5, 2.0]), o, m, w),
    'inf observed': lambda e, o, m, w: (e, np.array([0, np.inf, 2, 2]), m, w),
    'negative expected': lambda e, o, m, w: (np.array([0.5, -1.0, 1.5, 2.0]), o, m, w),
    'negative observed': lambda e, o, m, w: (e, np.array([0, -1, 2, 2]), m, w),
    'fractional observed': lambda e, o, m, w: (e, np.array([0, 0.5, 2, 2]), m, w),
    'observed length': lambda e, o, m, w: (e, o[:3], m, w),
    'match length': lambda e, o, m, w: (e, o, m[:3], w),
    'weight length': lambda e, o, m, w: (e, o, m, w[:3]),
    'negative weight': lambda e, o, m, w: (e, o, m, np.array([0.5, -0.5, 0.5, 0.5])),
    'nan weight': lambda e, o, m, w: (e, o, m, np.array([0.5, np.nan, 0.5, 0.5])),
    'zero weights': lambda e, o, m, w: (e, o, m, np.zeros(4)),
    'empty': lambda e, o, m, w: (e[:0], o[:0], m[:0], w[:0]),
    'two dimensions': lambda e, o, m, w: (e[None], o[None], m[None], w[None]),
    'constant expected': lambda e, o, m, w: (np.full(4, 1.0), o, m, w),
    'expected add up to zero': lambda e, o, m, w: (np.zeros(4), o, m, w),
}


@pytest.mark.parametrize('case', sorted(BAD_OCCURRENCE))
def test_occurrence_calibration_refuses_what_it_cannot_judge(case):
    with pytest.raises(ValueError):
        gate_s3.occurrence_calibration(*BAD_OCCURRENCE[case](*good_rows()), B=10, seed=1)


BAD_VALUE = {
    'nan expected': lambda e, v, m, w: (np.array([0.5, np.nan, 1.5, 2.0]), v, m, w),
    'inf realized': lambda e, v, m, w: (e, np.array([0.0, np.inf, 2.0, 2.0]), m, w),
    'realized length': lambda e, v, m, w: (e, v[:3], m, w),
    'weight length': lambda e, v, m, w: (e, v, m, w[:3]),
    'negative weight': lambda e, v, m, w: (e, v, m, np.array([0.5, -0.5, 0.5, 0.5])),
    'zero weights': lambda e, v, m, w: (e, v, m, np.zeros(4)),
    'empty': lambda e, v, m, w: (e[:0], v[:0], m[:0], w[:0]),
    'constant expected': lambda e, v, m, w: (np.full(4, 1.0), v, m, w),
}


@pytest.mark.parametrize('case', sorted(BAD_VALUE))
def test_value_calibration_refuses_what_it_cannot_judge(case):
    e, o, m, w = good_rows()
    with pytest.raises(ValueError):
        gate_s3.value_calibration(*BAD_VALUE[case](e, o.astype(float), m, w), B=10, seed=1)


def test_weights_that_add_up_to_zero_are_refused_by_the_function_that_got_them():
    e, o, m, _ = good_rows()
    with pytest.raises(ValueError, match='^occurrence_calibration: the weights add up to 0'):
        gate_s3.occurrence_calibration(e, o, m, np.zeros(4), B=10, seed=1)
    with pytest.raises(ValueError, match='^value_calibration: the weights add up to 0'):
        gate_s3.value_calibration(e, o.astype(float), m, np.zeros(4), B=10, seed=1)
    with pytest.raises(ValueError, match='^overdispersion: the weights add up to 0'):
        gate_s3.overdispersion(o, e, m, np.zeros(4), B=10, seed=1)


@pytest.mark.parametrize('fn', ['occurrence', 'value', 'overdispersion'])
def test_match_numbers_must_be_integers(fn):
    e, o, m, w = good_rows()
    with pytest.raises(TypeError):
        if fn == 'occurrence':
            gate_s3.occurrence_calibration(e, o, m.astype(float), w, B=10, seed=1)
        elif fn == 'value':
            gate_s3.value_calibration(e, o.astype(float), m.astype(float), w, B=10, seed=1)
        else:
            gate_s3.overdispersion(o, e, m.astype(float), w, B=10, seed=1)


@pytest.mark.parametrize('kwargs', [dict(B=0), dict(B=-3), dict(B=2.5), dict(B=True), dict(seed=-1), dict(seed=1.5), dict(seed=True)])
def test_bad_b_or_seed_is_refused(kwargs):
    e, o, m, w = good_rows()
    with pytest.raises((ValueError, TypeError)):
        gate_s3.occurrence_calibration(e, o, m, w, **kwargs)
    with pytest.raises((ValueError, TypeError)):
        gate_s3.value_calibration(e, o.astype(float), m, w, **kwargs)
    with pytest.raises((ValueError, TypeError)):
        gate_s3.overdispersion(o, e, m, w, **kwargs)


def test_error_messages_hold_counts_but_never_match_numbers():
    # a match number can be a key (security rule): a refusal says how many rows are wrong, not which match they belong to
    e, o, m, w = good_rows()
    secret = np.array([987654321, 987654321, 123456789, 123456789])
    bad = {'nan expected': (np.array([0.5, np.nan, 1.5, 2.0]), o, secret, w),
           'negative weight': (e, o, secret, w - np.array([0, 1.0, 0, 0])),
           'length': (e, o, secret[:3], w), 'float match': (e, o, secret.astype(float), w)}
    for case, args in bad.items():
        with pytest.raises((ValueError, TypeError)) as caught:
            gate_s3.occurrence_calibration(*args, B=10, seed=1)
        assert '987654321' not in str(caught.value) and '123456789' not in str(caught.value), case


# ---- the agreement of the simulation with the closed form --------------------------------------------------------------------------------
def test_mc_agreement_counts_the_rows_within_three_standard_errors():
    mean = np.array([10.0, 5.0, 3.0, 7.0])
    sd = np.array([2.0, 2.0, 2.0, 0.0])
    analytic = np.array([10.5, 5.7, 3.0, 7.0])
    r = gate_s3.mc_agreement(mean, sd, 100, analytic)                          # 3 standard errors = 3 x 2 / 10 = 0.6
    assert r['n_rows'] == 4 and r['share'] == 0.75 and r['n_agree'] == 3
    assert r['worst_z'] == pytest.approx(3.5, rel=1e-12)                       # row 1: |5.0 - 5.7| / 0.2
    assert r['n_sd_zero'] == 1 and r['n_sd_zero_disagree'] == 0


def test_mc_agreement_needs_a_strict_inequality():
    # S = 4, sd = 2: 3 standard errors = 3 exactly. A difference of exactly 3 is not inside; a hair less is
    on_the_edge = gate_s3.mc_agreement([4.0], [2.0], 4, [1.0])
    assert on_the_edge['share'] == 0.0 and on_the_edge['worst_z'] == 3.0
    just_inside = gate_s3.mc_agreement([np.nextafter(4.0, 0.0)], [2.0], 4, [1.0])
    assert just_inside['share'] == 1.0


def test_mc_agreement_rows_without_spread_must_be_equal():
    equal = gate_s3.mc_agreement([5.0, 6.0], [0.0, 0.0], 1000, [5.0, 6.0])
    assert equal['share'] == 1.0 and equal['n_sd_zero'] == 2 and equal['n_sd_zero_disagree'] == 0 and equal['worst_z'] is None
    off = gate_s3.mc_agreement([5.0, 6.0], [0.0, 0.0], 1000, [5.0, np.nextafter(6.0, 7.0)])
    assert off['share'] == 0.5 and off['n_sd_zero'] == 2 and off['n_sd_zero_disagree'] == 1


def test_mc_agreement_mixed_rows_with_and_without_spread():
    r = gate_s3.mc_agreement([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], 100, [1.0, 2.05, 3.5])         # row 1: 0.05 < 0.3 agree; row 2 differs, sd 0
    assert r['share'] == pytest.approx(2 / 3) and r['n_sd_zero'] == 2 and r['n_sd_zero_disagree'] == 1
    assert r['worst_z'] == pytest.approx(0.5)


def test_mc_agreement_perfect_and_total_disagreement():
    rng = np.random.default_rng(1)
    analytic = rng.normal(size=500)
    sd = rng.uniform(0.5, 2.0, 500)
    assert gate_s3.mc_agreement(analytic, sd, 1000, analytic)['share'] == 1.0
    assert gate_s3.mc_agreement(analytic + 100.0, sd, 1000, analytic)['share'] == 0.0


def test_mc_agreement_of_a_real_simulation_is_near_one():
    # S = 1000 draws of a Poisson count per row: the mean is within 3 standard errors for almost every row (3 sigma: 99.7 per cent)
    rng = np.random.default_rng(2)
    rates = rng.uniform(0.2, 3.0, 1000)
    draws = rng.poisson(rates[:, None], size=(1000, 1000))
    r = gate_s3.mc_agreement(draws.mean(axis=1), draws.std(axis=1, ddof=1), 1000, rates)
    assert r['share'] >= 0.99 and r['n_rows'] == 1000 and r['worst_z'] < 5.0


def test_mc_agreement_accepts_numpy_integers_and_lists():
    a = gate_s3.mc_agreement([1.0, 2.0], [1.0, 1.0], np.int64(100), [1.0, 2.0])
    assert a == gate_s3.mc_agreement(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 100, np.array([1.0, 2.0]))


@pytest.mark.parametrize('case', ['nan mean', 'inf mean', 'nan sd', 'negative sd', 'nan analytic', 'length', 'empty', 'two dimensions',
                                  'S zero', 'S negative', 'S float', 'S bool', 'S string'])
def test_mc_agreement_refuses_what_it_cannot_judge(case):
    base = dict(mc_mean=[1.0, 2.0], mc_sd=[1.0, 1.0], S=100, analytic=[1.0, 2.0])
    bad = {'nan mean': dict(mc_mean=[1.0, np.nan]), 'inf mean': dict(mc_mean=[1.0, np.inf]), 'nan sd': dict(mc_sd=[1.0, np.nan]),
           'negative sd': dict(mc_sd=[1.0, -1.0]), 'nan analytic': dict(analytic=[1.0, np.nan]), 'length': dict(analytic=[1.0]),
           'empty': dict(mc_mean=[], mc_sd=[], analytic=[]), 'two dimensions': dict(mc_mean=[[1.0, 2.0]]),
           'S zero': dict(S=0), 'S negative': dict(S=-5), 'S float': dict(S=100.5), 'S bool': dict(S=True), 'S string': dict(S='100')}
    with pytest.raises((ValueError, TypeError)):
        gate_s3.mc_agreement(**{**base, **bad[case]})


# ---- the over-dispersion test: the rows without an expected count (ruling R22) -------------------------------------------------------
def test_rows_with_a_zero_expected_count_are_left_out_and_counted():
    y = np.array([0, 0, 3, 0, 1, 2, 0, 0])
    mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])                       # the last two rows cover no time: mu = 0 and y = 0
    m = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    w = np.array([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.4, 0.4])
    r = gate_s3.overdispersion(y, mu, m, w, B=40, seed=3)
    kept = gate_s3.overdispersion(y[:6], mu[:6], m[:6], w[:6], B=40, seed=3)
    assert r['n_rows'] == 6 and r['n_zero_mu'] == 2
    assert r['phi'] == kept['phi'] and r['phi_ci'] == kept['phi_ci'] and r['n_matches'] == kept['n_matches'] == 3
    assert kept['n_zero_mu'] == 0


def test_a_match_with_only_zero_expected_rows_is_not_resampled():
    y = np.array([0, 2, 1, 0, 0])
    mu = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    m = np.array([0, 0, 1, 2, 2])                                                   # match 2 has no row with an expected count
    r = gate_s3.overdispersion(y, mu, m, np.ones(5), B=40, seed=3)
    assert r['n_matches'] == 2 and r['n_rows'] == 3 and r['n_zero_mu'] == 2
    assert r['phi_ci'] == gate_s3.overdispersion(y[:3], mu[:3], m[:3], np.ones(3), B=40, seed=3)['phi_ci']


def test_a_count_where_the_expected_count_is_zero_is_refused():
    with pytest.raises(ValueError, match='expected count'):
        gate_s3.overdispersion([0, 1, 2], [1.0, 0.0, 1.0], [0, 1, 2], B=10, seed=1)


BAD_PHI = {
    'nan y': lambda y, mu, m, w: (np.array([0, np.nan, 2, 1]), mu, m, w),
    'negative y': lambda y, mu, m, w: (np.array([0, -1, 2, 1]), mu, m, w),
    'fractional y': lambda y, mu, m, w: (np.array([0, 0.5, 2, 1]), mu, m, w),
    'negative mu': lambda y, mu, m, w: (y, np.array([1.0, -1.0, 1.0, 1.0]), m, w),
    'nan mu': lambda y, mu, m, w: (y, np.array([1.0, np.nan, 1.0, 1.0]), m, w),
    'inf mu': lambda y, mu, m, w: (y, np.array([1.0, np.inf, 1.0, 1.0]), m, w),
    'mu length': lambda y, mu, m, w: (y, mu[:3], m, w),
    'weight length': lambda y, mu, m, w: (y, mu, m, w[:3]),
    'negative weight': lambda y, mu, m, w: (y, mu, m, np.array([1.0, -1.0, 1.0, 1.0])),
    'zero weights': lambda y, mu, m, w: (y, mu, m, np.zeros(4)),
    'all mu zero': lambda y, mu, m, w: (np.zeros(4), np.zeros(4), m, w),
    'empty': lambda y, mu, m, w: (y[:0], mu[:0], m[:0], w[:0]),
    'two dimensions': lambda y, mu, m, w: (y[None], mu[None], m[None], w[None]),
}


@pytest.mark.parametrize('case', sorted(BAD_PHI))
def test_overdispersion_refuses_what_it_cannot_judge(case):
    base = (np.array([0, 1, 2, 1]), np.ones(4), np.array([0, 0, 1, 1]), np.ones(4))
    with pytest.raises(ValueError):
        gate_s3.overdispersion(*BAD_PHI[case](*base), B=10, seed=1)


def test_overdispersion_with_equal_weights_by_default():
    y, mu, m, _ = negbin_counts(alpha=0.3, seed=36, n_matches=30)
    assert gate_s3.overdispersion(y, mu, m, B=40, seed=1) == gate_s3.overdispersion(y, mu, m, np.ones(len(y)), B=40, seed=1)
    weighted = gate_s3.overdispersion(y, mu, m, match_weights(m), B=40, seed=1)
    assert weighted['phi'] != gate_s3.overdispersion(y, mu, m, B=40, seed=1)['phi']            # the weights of the gate are not equal


# ---- the verdict ------------------------------------------------------------------------------------------------------------------------
def occ_result(oe_ci=(0.97, 1.03), slope_ci=(0.95, 1.05), oe=1.0, slope=1.0):
    return {'oe': oe, 'oe_ci': oe_ci, 'slope': slope, 'slope_ci': slope_ci}


def value_result(slope_ci=(0.9, 1.1), bias_ci=(-20.0, 20.0), slope=1.0, bias=0.0):
    return {'slope': slope, 'slope_ci': slope_ci, 'bias': bias, 'bias_ci': bias_ci}


def everything(**override):
    full = {'occurrence': occ_result(), 'value': value_result(), 'accounting': {'max_abs_error': 0},
            'mc': {'share': 1.0, 'n_rows': 100}, 'overdispersion': {'phi': 1.0, 'phi_ci': (0.95, 1.05)}}
    full.update(override)
    return full


def test_verdict_of_a_clean_gate_passes_every_criterion():
    v = gate_s3.verdict(everything())
    assert v['all_pass'] is True and v['missing'] == []
    for name in ('occurrence', 'value', 'accounting', 'mc'):
        assert v[name]['pass'] is True, name


@pytest.mark.parametrize('ci, ok', [((0.95, 1.05), True), ((0.9500001, 1.0499999), True), ((0.9499999, 1.03), False),
                                    ((0.97, 1.0500001), False), ((0.8, 1.2), False), ((1.06, 1.10), False)])
def test_occurrence_oe_band_is_closed_and_the_whole_interval_must_be_inside(ci, ok):
    v = gate_s3.verdict({'occurrence': occ_result(oe_ci=ci)})['occurrence']
    assert v['oe_in_band'] is ok and v['slope_in_band'] is True and v['pass'] is ok


@pytest.mark.parametrize('ci, ok', [((0.9, 1.1), True), ((0.9000001, 1.0999999), True), ((0.8999999, 1.0), False),
                                    ((1.0, 1.1000001), False), ((1.15, 1.3), False)])
def test_occurrence_slope_band_is_closed_and_the_whole_interval_must_be_inside(ci, ok):
    v = gate_s3.verdict({'occurrence': occ_result(slope_ci=ci)})['occurrence']
    assert v['slope_in_band'] is ok and v['oe_in_band'] is True and v['pass'] is ok


@pytest.mark.parametrize('ci, ok', [((0.8, 1.2), True), ((0.7999999, 1.0), False), ((1.0, 1.2000001), False), ((0.5, 1.5), False)])
def test_value_slope_band(ci, ok):
    v = gate_s3.verdict({'value': value_result(slope_ci=ci)})['value']
    assert v['slope_in_band'] is ok and v['bias_in_band'] is True and v['pass'] is ok


@pytest.mark.parametrize('ci, ok', [((-50.0, 50.0), True), ((-50.0001, 0.0), False), ((0.0, 50.0001), False), ((-20.0, 20.0), True),
                                    ((60.0, 90.0), False)])
def test_value_bias_band(ci, ok):
    v = gate_s3.verdict({'value': value_result(bias_ci=ci)})['value']
    assert v['bias_in_band'] is ok and v['slope_in_band'] is True and v['pass'] is ok


def test_verdict_judges_the_intervals_and_not_the_estimates():
    # the estimate is inside the band, but the interval reaches out of it: not calibrated
    v = gate_s3.verdict({'occurrence': occ_result(oe_ci=(0.9, 1.1), oe=1.0)})['occurrence']
    assert v['oe'] == 1.0 and not v['pass']
    # the rule of the brief is "the interval lies inside the band": the point estimate is reported, not judged
    v = gate_s3.verdict({'occurrence': occ_result(oe_ci=(0.96, 1.04), oe=2.0)})['occurrence']
    assert v['pass']


def test_verdict_entries_carry_the_numbers_and_the_bands():
    v = gate_s3.verdict(everything(occurrence=occ_result(oe=1.01, slope=0.98), value=value_result(slope=1.05, bias=3.5)))
    occ = v['occurrence']
    assert occ['oe'] == 1.01 and occ['oe_ci'] == [0.97, 1.03] and occ['oe_band'] == [0.95, 1.05]
    assert occ['slope'] == 0.98 and occ['slope_ci'] == [0.95, 1.05] and occ['slope_band'] == [0.9, 1.1]
    val = v['value']
    assert val['slope'] == 1.05 and val['slope_ci'] == [0.9, 1.1] and val['slope_band'] == [0.8, 1.2]
    assert val['bias'] == 3.5 and val['bias_ci'] == [-20.0, 20.0] and val['bias_band'] == [-50.0, 50.0]
    assert v['accounting']['max_abs_error'] == 0 and v['accounting']['must_equal'] == 0.0
    assert v['mc']['share'] == 1.0 and v['mc']['min_share'] == 0.99
    assert v['overdispersion']['phi'] == 1.0 and v['overdispersion']['phi_ci'] == [0.95, 1.05] and v['overdispersion']['limit'] == 1.1


def test_verdict_bands_are_read_from_the_module_constants(monkeypatch):
    monkeypatch.setattr(gate_s3, 'OE_BAND', (0.5, 1.5))
    monkeypatch.setattr(gate_s3, 'MC_SHARE', 0.5)
    monkeypatch.setattr(gate_s3, 'DISPERSION_LIMIT', 3.0)
    v = gate_s3.verdict({'occurrence': occ_result(oe_ci=(0.6, 1.4)), 'mc': {'share': 0.6},
                         'overdispersion': {'phi': 2.0, 'phi_ci': (2.0, 2.5)}})
    assert v['occurrence']['oe_in_band'] and v['occurrence']['oe_band'] == [0.5, 1.5]
    assert v['mc']['pass'] and v['mc']['min_share'] == 0.5
    assert v['overdispersion']['overdispersed'] is False and v['overdispersion']['limit'] == 3.0


@pytest.mark.parametrize('error, ok', [(0, True), (0.0, True), (1e-300, False), (1e-12, False), (0.5, False), (float('nan'), False)])
def test_accounting_passes_only_for_an_error_of_exactly_zero(error, ok):
    v = gate_s3.verdict({'accounting': {'max_abs_error': error}})['accounting']
    assert v['pass'] is ok and (v['max_abs_error'] == error or math.isnan(error))


@pytest.mark.parametrize('share, ok', [(1.0, True), (0.99, True), (0.9900001, True), (0.9899999, False), (0.98, False), (0.0, False)])
def test_mc_passes_from_a_share_of_99_percent(share, ok):
    v = gate_s3.verdict({'mc': {'share': share}})['mc']
    assert v['pass'] is ok and v['share'] == share


def test_mc_share_of_a_real_agreement_result_is_judged_exactly_at_the_limit():
    ok = gate_s3.mc_agreement(np.zeros(100), np.ones(100), 100, np.concatenate([np.zeros(99), [10.0]]))        # 99 of 100 rows agree
    assert ok['share'] == 0.99 and gate_s3.verdict({'mc': ok})['mc']['pass'] is True
    bad = gate_s3.mc_agreement(np.zeros(100), np.ones(100), 100, np.concatenate([np.zeros(98), [10.0, 10.0]]))
    assert bad['share'] == 0.98 and gate_s3.verdict({'mc': bad})['mc']['pass'] is False


@pytest.mark.parametrize('ci, over', [((1.1, 1.3), False), ((1.1000001, 1.3), True), ((1.5, 2.0), True), ((0.9, 1.2), False),
                                      ((1.0, 1.05), False)])
def test_overdispersion_is_reported_not_judged(ci, over):
    v = gate_s3.verdict({'overdispersion': {'phi': 1.2, 'phi_ci': ci}})['overdispersion']
    assert v['overdispersed'] is over and 'pass' not in v


def test_overdispersion_does_not_enter_all_pass():
    v = gate_s3.verdict(everything(overdispersion={'phi': 3.0, 'phi_ci': (2.5, 3.5)}))
    assert v['overdispersion']['overdispersed'] is True and v['all_pass'] is True
    v = gate_s3.verdict({'overdispersion': {'phi': 3.0, 'phi_ci': (2.5, 3.5)}, 'mc': {'share': 1.0}})
    assert v['all_pass'] is True


@pytest.mark.parametrize('failing', ['occurrence', 'value', 'accounting', 'mc'])
def test_one_failed_criterion_fails_all_pass(failing):
    bad = {'occurrence': occ_result(oe_ci=(0.5, 1.5)), 'value': value_result(bias_ci=(60.0, 90.0)), 'accounting': {'max_abs_error': 1},
           'mc': {'share': 0.5}}[failing]
    v = gate_s3.verdict(everything(**{failing: bad}))
    assert v['all_pass'] is False and v[failing]['pass'] is False
    assert all(v[k]['pass'] for k in ('occurrence', 'value', 'accounting', 'mc') if k != failing)


def test_all_pass_covers_the_criteria_that_are_present_and_missing_ones_are_listed():
    v = gate_s3.verdict({'occurrence': occ_result()})
    assert v['all_pass'] is True and v['missing'] == ['value', 'accounting', 'mc'] and set(v) == {'occurrence', 'all_pass', 'missing'}
    v = gate_s3.verdict({'occurrence': occ_result(), 'mc': {'share': 0.5}})
    assert v['all_pass'] is False and v['missing'] == ['value', 'accounting']
    v = gate_s3.verdict({'accounting': {'max_abs_error': 0}, 'mc': {'share': 1.0}, 'occurrence': occ_result(), 'value': value_result()})
    assert v['missing'] == [] and v['all_pass'] is True


def test_overdispersion_alone_is_not_a_gate_criterion_so_four_are_missing():
    v = gate_s3.verdict({'overdispersion': {'phi': 1.0, 'phi_ci': (0.9, 1.1)}})
    assert v['missing'] == ['occurrence', 'value', 'accounting', 'mc'] and v['all_pass'] is False            # nothing was judged: no pass


def test_an_empty_result_is_no_pass():
    v = gate_s3.verdict({})
    assert v == {'all_pass': False, 'missing': ['occurrence', 'value', 'accounting', 'mc']}


def test_verdict_takes_intervals_as_lists_tuples_or_arrays():
    for ci in ([0.97, 1.03], (0.97, 1.03), np.array([0.97, 1.03])):
        assert gate_s3.verdict({'occurrence': occ_result(oe_ci=ci)})['occurrence']['oe_in_band'] is True


def test_verdict_ignores_the_pass_flags_it_is_given():
    # the verdict is computed from the numbers and the module constants: a flag in the input cannot make a criterion pass
    r = occ_result(oe_ci=(0.5, 1.5))
    r['pass'] = True
    assert gate_s3.verdict({'occurrence': r})['occurrence']['pass'] is False
    m = {'share': 0.5, 'pass': True}
    assert gate_s3.verdict({'mc': m})['mc']['pass'] is False
    o = {'phi': 1.0, 'phi_ci': (0.9, 1.0), 'overdispersed': True}
    assert gate_s3.verdict({'overdispersion': o})['overdispersion']['overdispersed'] is False


def test_verdict_of_the_functions_own_results_agrees_with_the_by_hand_rule():
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=2), B=200, seed=2)
    v = gate_s3.verdict({'occurrence': r})['occurrence']
    assert v['pass'] == (0.95 <= r['oe_ci'][0] and r['oe_ci'][1] <= 1.05 and 0.9 <= r['slope_ci'][0] and r['slope_ci'][1] <= 1.1)
    r = gate_s3.value_calibration(*miscalibrated_values(scale=1.1, seed=2), B=200, seed=2)
    v = gate_s3.verdict({'value': r})['value']
    assert v['pass'] == (0.8 <= r['slope_ci'][0] and r['slope_ci'][1] <= 1.2 and -50.0 <= r['bias_ci'][0] and r['bias_ci'][1] <= 50.0)


def test_verdict_with_an_nan_interval_fails():
    nan = float('nan')
    v = gate_s3.verdict({'occurrence': occ_result(slope_ci=(nan, nan)), 'value': value_result(bias_ci=(nan, 1.0))})
    assert v['occurrence']['pass'] is False and v['value']['pass'] is False and v['all_pass'] is False


@pytest.mark.parametrize('results', [
    {'occurence': {}},                                                              # a typo must not become a "missing" criterion silently
    {'occurrence': occ_result(), 'extra': 1},
    [('occurrence', occ_result())],
    None,
    {'occurrence': None},
    {'occurrence': 1.0},
    {'occurrence': {'oe': 1.0, 'slope': 1.0, 'slope_ci': (0.95, 1.05)}},            # no oe_ci
    {'occurrence': occ_result(oe_ci=(0.9, 1.0, 1.1))},                              # not an interval
    {'occurrence': occ_result(oe_ci=(1.1, 0.9))},                                   # lower end above the upper end
    {'occurrence': occ_result(oe_ci='ab')},
    {'value': {'slope': 1.0, 'slope_ci': (0.9, 1.1), 'bias': 0.0}},                 # no bias_ci
    {'accounting': {'max_abs_value_error': 0.0}},                                   # no max_abs_error
    {'accounting': 0},
    {'mc': {}},
    {'mc': {'share': 1.5}},
    {'mc': {'share': -0.1}},
    {'mc': {'share': 'high'}},
    {'overdispersion': {'phi': 1.0}},                                               # no phi_ci
])
def test_verdict_refuses_malformed_results(results):
    with pytest.raises((ValueError, TypeError)):
        gate_s3.verdict(results)


def test_verdict_does_not_change_its_input():
    results = everything()
    snapshot = json.dumps(results, sort_keys=True)
    gate_s3.verdict(results)
    assert json.dumps(results, sort_keys=True) == snapshot


# ---- the outputs are plain JSON values (the S3 report writes them) -----------------------------------------------------------------------
def assert_plain(obj, path='result'):
    """Only dict (str keys), list, str, bool, int, float, None: no numpy scalar or array and no tuple anywhere."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert type(k) is str, (path, k)
            assert_plain(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_plain(v, f'{path}[{i}]')
    else:
        assert type(obj) in (str, bool, int, float, type(None)), (path, type(obj))


def test_every_output_is_made_of_plain_json_values():
    outputs = {
        'occurrence': gate_s3.occurrence_calibration(*calibrated_counts(seed=41, n_matches=20), B=20, seed=1),
        'value': gate_s3.value_calibration(*miscalibrated_values(scale=1.0, seed=41, n_matches=20), B=20, seed=1),
        'overdispersion': gate_s3.overdispersion(*negbin_counts(alpha=0.3, seed=41, n_matches=20), B=20, seed=1),
        'mc': gate_s3.mc_agreement(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 0.0]), np.int64(100), np.array([1.0, 2.1, 3.0])),
        'mc_no_spread': gate_s3.mc_agreement(np.array([1.0]), np.array([0.0]), 100, np.array([1.0])),
    }
    for name, out in outputs.items():
        assert_plain(out, name)
        assert json.loads(json.dumps(out)) == out, name                          # nothing is lost on the way to S3_REPORT.json
    results = {k: outputs[k] for k in ('occurrence', 'value', 'overdispersion', 'mc')}
    results['accounting'] = {'max_abs_error': 0}
    v = gate_s3.verdict(results)
    assert_plain(v, 'verdict')
    assert json.loads(json.dumps(v)) == v


def test_the_interval_is_a_list_of_two_python_floats():
    r = gate_s3.occurrence_calibration(*calibrated_counts(seed=42, n_matches=20), B=20, seed=1)
    for key in ('oe_ci', 'slope_ci', 'intercept_ci'):
        assert type(r[key]) is list and len(r[key]) == 2 and all(type(v) is float for v in r[key]) and r[key][0] <= r[key][1]


# ---- structure ---------------------------------------------------------------------------------------------------------------------------
def test_the_module_imports_only_numpy_math_and_nll():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    modules, from_c_ept = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or '').split('.')[0])
            if node.module == 'c_ept':
                from_c_ept |= {a.name for a in node.names}
    assert modules <= {'__future__', 'collections', 'math', 'numpy', 'c_ept'}, modules         # collections.abc.Mapping for the results
    assert from_c_ept == {'nll'}


def test_the_module_reads_no_store_no_file_and_has_no_holdout_switch():
    text = SOURCE.read_text(encoding='utf-8')
    assert 'allow_holdout' not in text and 'matches/key' not in text and 's1.array' not in text
    assert 'open(' not in text and 'read_text' not in text and 'np.load' not in text and 'Path(' not in text


def test_the_thresholds_are_set_in_one_place_and_only_there():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    assigned = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    for name, value in (('OE_BAND', (0.95, 1.05)), ('OCC_SLOPE_BAND', (0.9, 1.1)), ('VALUE_SLOPE_BAND', (0.8, 1.2)),
                        ('BIAS_BAND', (-50.0, 50.0)), ('MC_SHARE', 0.99), ('DISPERSION_LIMIT', 1.1)):
        assert ast.literal_eval(assigned[name]) == value, name                  # written as literals at module level, not computed
