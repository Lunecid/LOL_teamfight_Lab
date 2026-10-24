"""Tests of ``c_ept.report_s3``: S3_REPORT.json and the figures of a finished S3 run folder (stage S3 Task 13).

No real run folder, raw file or index is read. The pieces of the report are tested first on small arrays and dicts whose answers were
worked out by hand or are written a second time here with plain loops and numpy (groups A-G); then the whole report is made from a toy run
that the real runner (``run_s3.main``, all five stages, two spawned workers) wrote on the random 48-match world of ``test_run_s3`` (group
H; the ``world`` fixture is that file's, imported and not copied; the s3a folder gets a full D7 prices document first, ``build_toy_run``).
The toy numbers mean nothing (the heads are fitted on a few hundred rows); what is checked is that every item holds what the files hold,
that the lines are the statistic they say they are, and the refusals.

    A  PIT, quantile hits, CRPS of 19 quantiles, window expectations     E  the gate without the smoke DEV matches, the gate check
    B  observed onsets per window and the occurrence line of a window    F  V_start statistics and the composition summary of B1
    C1 criterion (iv) with the flat rows, the D7 section                 G  the findings at the top of the report
    C2 B0 convergence (B4), the NLL table, the cost record               H  the toy run: items 1-11, the carries, figures, keys, refusals,
    D  the key scan (full keys and bare game ids)                           the command line

The toy run is built once per module (about 60 s); the tests that make a report of their own use B = 50 bootstrap replicates (``BOOT_B``
is patched) unless they compare with the gate, and the report that the items are read from (``reported``) uses the gate's B = 2,000.
Test-first record: each group was run before its functions existed (an AttributeError) and after; the groups C1 and C2 were shown red
again with their functions deleted in-process.
"""

import contextlib
import gc
import io
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import baselines_b0, report_s3, run_s3
from c_ept import config as C
from c_ept.tests.conftest import empty_process_list
from c_ept.tests.s3_toys import TOY_PRICES, random_value_record
from c_ept.tests.test_run_s3 import cli_args, sha_of, write_spec
from c_ept.tests.test_run_s3 import world

SHARED_FIXTURES = (world,)  # the world fixture of test_run_s3 is shared by import, not copied: pytest finds it by this name

# ---- group A: the small statistics (PIT, quantile hits, CRPS from 19 quantiles, window expectations) -------------------------------------
Q_STEP = np.tile(np.arange(1, 20, dtype=float), (4, 1))  # four rows whose quantile k (k = 1 .. 19) is the number k
PIT_REALIZED = np.array([0.5, 19.5, 7.5, 10.0])  # below all, above all, between q7 and q8, exactly q10
PIT_W = np.array([1.0, 1.0, 2.0, 4.0])


def test_the_pit_histogram_puts_each_row_in_its_bin_and_spreads_a_tie_over_the_bins_it_covers():
    out = report_s3.pit_histogram(Q_STEP, PIT_REALIZED, PIT_W)
    want = np.zeros(20)
    want[0], want[19], want[7] = 1 / 8, 1 / 8, 2 / 8
    want[9] = want[10] = 2 / 8  # the row equal to q10: n_lt = 9, n_le = 10, half of its weight each
    assert out['bins'] == 20 and out['n_rows'] == 4 and out['expected_mass'] == 0.05
    assert out['mass'] == pytest.approx(want, abs=1e-15)
    assert out['n_tied_rows'] == 1
    assert out['max_abs_deviation'] == pytest.approx(0.2)  # 2/8 - 0.05 is the largest distance from the uniform 0.05


def test_a_point_mass_at_the_realized_value_is_uniform_because_the_tie_covers_all_twenty_bins():
    out = report_s3.pit_histogram(np.zeros((3, 19)), np.zeros(3), np.ones(3))
    assert out['mass'] == pytest.approx([0.05] * 20) and out['n_tied_rows'] == 3
    assert out['max_abs_deviation'] == pytest.approx(0.0, abs=1e-15)


def test_the_mass_adds_up_to_one_and_does_not_depend_on_the_scale_of_the_weights():
    rng = np.random.default_rng(3)
    q = np.sort(rng.integers(0, 6, (200, 19)).astype(float), axis=1)  # many ties on purpose
    y = rng.integers(0, 6, 200).astype(float)
    w = rng.uniform(0.1, 2.0, 200)
    a, b = report_s3.pit_histogram(q, y, w), report_s3.pit_histogram(q, y, 7.5 * w)
    assert sum(a['mass']) == pytest.approx(1.0, abs=1e-12) and a['mass'] == pytest.approx(b['mass'], abs=1e-12)


def test_values_drawn_from_the_predictive_law_give_a_flat_histogram_and_a_skewed_law_does_not():
    rng = np.random.default_rng(5)
    n = 40000
    q = np.tile(np.arange(1, 20) / 20.0, (n, 1))  # the quantiles of U(0, 1)
    flat = report_s3.pit_histogram(q, rng.random(n), np.ones(n))
    skew = report_s3.pit_histogram(q, rng.random(n) ** 2, np.ones(n))  # the values pile up near 0
    assert flat['max_abs_deviation'] < 0.01
    assert skew['mass'][0] > 0.2 and skew['max_abs_deviation'] > 0.15


@pytest.mark.parametrize(
    'q19, y, w',
    [
        (np.ones((2, 18)), np.ones(2), np.ones(2)),  # not 19 quantiles
        (np.array([np.arange(19.0)[::-1]]), np.ones(1), np.ones(1)),  # a decreasing row
        (np.where(np.arange(19) == 5, np.nan, np.arange(19.0))[None, :], np.ones(1), np.ones(1)),  # NaN in q19
        (np.arange(19.0)[None, :], np.array([np.inf]), np.ones(1)),  # infinite value
        (np.arange(19.0)[None, :], np.ones(1), np.array([-1.0])),  # negative weight
        (np.arange(19.0)[None, :], np.ones(1), np.zeros(1)),  # weights add up to 0
        (np.arange(19.0)[None, :], np.ones(2), np.ones(1)),  # lengths differ
    ],
)
def test_the_pit_histogram_refuses_inputs_it_cannot_judge(q19, y, w):
    with pytest.raises(ValueError):
        report_s3.pit_histogram(q19, y, w)


def test_the_quantile_hits_count_the_rows_below_and_at_or_below_each_quantile_by_weight():
    out = report_s3.quantile_hits(Q_STEP, PIT_REALIZED, PIT_W)
    k = np.arange(1, 20)
    below = (1 + 2 * (k >= 8) + 4 * (k >= 11)) / 8  # 0.5 < k always; 7.5 < k from 8; 10.0 < k from 11
    at_or_below = (1 + 2 * (k >= 8) + 4 * (k >= 10)) / 8  # and 10.0 <= k from 10
    assert (
        out['levels'] == pytest.approx(k / 20) and out['below'] == pytest.approx(below) and out['at_or_below'] == pytest.approx(at_or_below)
    )


def test_the_crps_from_19_quantiles_of_a_point_mass_is_the_absolute_error_and_weights_average_the_rows():
    assert report_s3.crps19(np.zeros((1, 19)), np.array([10.0]), np.ones(1)) == pytest.approx(10.0)
    assert report_s3.crps19(np.zeros((1, 19)), np.array([-10.0]), np.ones(1)) == pytest.approx(10.0)
    assert report_s3.crps19(np.zeros((2, 19)), np.array([10.0, -4.0]), np.array([1.0, 3.0])) == pytest.approx(5.5)


def test_the_crps_from_19_quantiles_is_the_pinball_loss_written_out_level_by_level():
    q, y = np.array([[-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0]]), 0.9
    tau, loss = np.arange(1, 20) / 20.0, 0.0
    for t, qk in zip(tau, q[0]):
        loss += t * (y - qk) if y >= qk else (1.0 - t) * (qk - y)
    assert report_s3.crps19(q, np.array([y]), np.ones(1)) == pytest.approx(2.0 * loss / 19.0, rel=1e-12)


def test_the_window_expectations_are_the_first_column_and_the_differences_of_the_cumulative_columns():
    out = report_s3.window_expectations(np.array([[1.0, 1.5, 3.0], [0.2, 0.2, 0.7]]))
    assert out.shape == (2, 3) and out == pytest.approx(np.array([[1.0, 0.5, 1.5], [0.2, 0.0, 0.5]]))


def test_a_difference_of_rounding_size_is_set_to_zero_and_a_real_decrease_is_refused():
    out = report_s3.window_expectations(np.array([[1.0, 1.0 - 2.2e-16, 2.0]]))
    assert out[0, 1] == 0.0 and (out >= 0.0).all() and out[0, 2] == pytest.approx(1.0)
    with pytest.raises(report_s3.ReportInputError):
        report_s3.window_expectations(np.array([[1.0, 0.5, 2.0]]))


@pytest.mark.parametrize('cum', [np.ones((2, 4)), np.ones(3), np.array([[1.0, np.nan, 2.0]]), np.array([[1.0, np.inf, np.inf]])])
def test_window_expectations_refuses_a_wrong_shape_and_values_that_are_not_finite(cum):
    with pytest.raises(report_s3.ReportInputError):
        report_s3.window_expectations(cum)


# ---- group B: observed onsets per window and the occurrence line of each window ----------------------------------------------------------
class FakeS1:
    """The three S1 arrays that ``window_observed`` reads, as a store double (5 frame rows, 8 onsets)."""

    def __init__(self):
        self.arrays = {
            'onsets/row': np.array([0, 0, 1, 1, 1, 2, 4, 4]),
            'onsets/bin': np.array([0, 5, 6, 11, 12, 17, 2, 17]),
            'frame_rows/t0': np.zeros(5),
        }

    def array(self, name):
        return self.arrays[name]


def test_the_observed_onsets_of_a_row_are_counted_in_the_bins_0_5_6_11_and_12_17():
    got = report_s3.window_observed(FakeS1(), np.array([0, 1, 2, 3, 4]))
    assert got.shape == (5, 3) and np.array_equal(got, [[2, 0, 0], [0, 2, 1], [0, 0, 1], [0, 0, 0], [1, 0, 1]])


def test_the_observed_onsets_follow_the_order_of_the_frame_rows_that_are_asked_for():
    assert np.array_equal(report_s3.window_observed(FakeS1(), np.array([4, 1])), [[1, 0, 1], [0, 2, 1]])


def window_world(sizes, seed=1):
    """(expected_cum, observed, match, w): matches of the given sizes, cumulative expected onsets, Poisson counts per window, 1/n_m."""
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(len(sizes)), sizes)
    n = len(match)
    expected_cum = np.cumsum(rng.uniform(0.1, 1.0, (n, 3)), axis=1) / 3
    observed = rng.poisson(0.6, (n, 3))
    w = 1.0 / np.repeat(np.asarray(sizes, dtype=float), sizes)
    return expected_cum, observed, match, w


def direct_line(x, y, w):
    """The weighted least-squares line of y on x and the ratio of the weighted sums, from the textbook formulas (numpy only)."""
    xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
    slope = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
    return slope, ym - slope * xm, np.sum(w * y) / np.sum(w * x)


def test_each_window_line_is_the_weighted_line_of_the_observed_on_the_expected_counts_of_that_window():
    expected_cum, observed, match, w = window_world([4, 9, 12, 7, 18, 5, 11, 10, 3, 6, 8, 14, 2, 13, 5, 9])  # matches of unequal size
    out = report_s3.window_lines(expected_cum, observed, match, B=60)
    assert [(d['from_s'], d['to_s']) for d in out['windows']] == [(0, 60), (60, 120), (120, 180), (0, 180)]
    expected = np.diff(expected_cum, axis=1, prepend=0.0)
    pairs = [(expected[:, j], observed[:, j]) for j in range(3)] + [(expected_cum[:, 2], observed.sum(axis=1))]
    for line, (x, y) in zip(out['windows'], pairs):
        slope, intercept, oe = direct_line(x, y, w)
        assert line['computed'] is True and line['n_rows'] == len(x) and line['n_matches'] == 16
        assert line['slope'] == pytest.approx(slope, rel=1e-9) and line['intercept'] == pytest.approx(intercept, rel=1e-9, abs=1e-12)
        assert line['oe'] == pytest.approx(oe, rel=1e-12)
        assert line['slope_ci'][0] < line['slope'] < line['slope_ci'][1] and line['oe_ci'][0] < line['oe'] < line['oe_ci'][1]
        assert line['sum_expected'] == pytest.approx(x.sum()) and line['sum_observed'] == pytest.approx(y.sum())


def test_a_window_line_uses_the_gate_bootstrap_by_default_and_says_which_it_used():
    expected_cum, observed, match, _ = window_world([10] * 30)
    out = report_s3.window_lines(expected_cum, observed, match)
    assert (out['B'], out['seed']) == (2000, 20261021) and all((d['B'], d['seed']) == (2000, 20261021) for d in out['windows'])
    assert report_s3.BOOT_B == 2000 and report_s3.SEED == 20261021


def test_the_first_window_line_equals_the_occurrence_calibration_of_the_gate_on_the_same_rows():
    from c_ept import features, gate_s3

    expected_cum, observed, match, w = window_world([6, 8, 5, 9, 12, 7, 6, 10, 11, 4])
    line = report_s3.window_lines(expected_cum, observed, match, B=100)['windows'][0]
    gate = gate_s3.occurrence_calibration(expected_cum[:, 0], observed[:, 0], match, features.match_weights(match), B=100)
    for key in ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci', 'n_rows', 'n_matches'):
        assert line[key] == gate[key], key


@pytest.mark.parametrize('exact', [False, True])
def test_a_window_whose_expected_counts_do_not_vary_is_reported_as_not_computed_and_the_others_still_are(exact):
    expected_cum, observed, match, _ = window_world([10] * 12)
    if exact:  # multiples of 1/64: adding 0.25 and subtracting again is exact
        expected_cum[:, 0] = np.random.default_rng(4).integers(1, 40, len(match)) / 64.0
    expected_cum[:, 1] = expected_cum[:, 0] + 0.25  # the window 60-120 s has the same expected count in every row
    # (without `exact` the 0.25 is only equal up to rounding noise, which the gate's own check would take for variation)
    expected_cum[:, 2] = expected_cum[:, 1] + np.random.default_rng(2).uniform(0.1, 1.0, len(match))
    out = report_s3.window_lines(expected_cum, observed, match, B=50)['windows']
    assert out[1]['computed'] is False and 'vary' in out[1]['reason'] and out[1]['name'] == '60-120 s'
    assert [d['computed'] for d in out] == [True, False, True, True] and 'slope' not in out[1]


def test_window_lines_refuses_counts_and_matches_that_do_not_belong_to_the_rows():
    expected_cum, observed, match, _ = window_world([10] * 5)
    with pytest.raises(report_s3.ReportInputError):
        report_s3.window_lines(expected_cum, observed[:-1], match)
    with pytest.raises(report_s3.ReportInputError):
        report_s3.window_lines(expected_cum, observed, match[:-1])
    with pytest.raises(report_s3.ReportInputError):
        report_s3.window_lines(expected_cum, observed - 5, match)


# ---- group C1: sections made from the recorded dicts (criterion (iv), D7 prices) ---------------------------------------------------------
def mc_record(**kw):
    """The criterion-(iv) result of ``gate_s3.mc_agreement`` and the matching verdict entry: 1,000 rows, 985 agree, 12 are flat (MC sd 0)
    and 9 of those disagree, so 15 rows disagree and 6 of them have a spread."""
    mc = {'S': 1000, 'n_rows': 1000, 'n_agree': 985, 'n_sd_zero': 12, 'n_sd_zero_disagree': 9, 'share': 0.985, 'worst_z': 4.2}
    mc.update(kw)
    return mc, {'pass': mc['share'] >= 0.99, 'share': mc['share'], 'min_share': 0.99}


def test_the_mc_section_gives_the_flat_row_counts_beside_the_agreement_share_and_says_where_the_disagreements_come_from():
    out = report_s3.mc_section(*mc_record())
    assert (out['share'], out['min_share'], out['pass'], out['S'], out['worst_z']) == (0.985, 0.99, False, 1000, 4.2)
    assert (out['n_rows'], out['n_agree'], out['n_disagree']) == (1000, 985, 15)
    assert (out['n_sd_zero'], out['n_sd_zero_disagree'], out['n_disagree_with_spread']) == (12, 9, 6)
    assert '9 of the 15' in out['statement'] and 'flat' in out['statement'] and 'R28' in out['statement']
    assert '6' in out['statement']  # the rest come from rows that have a spread


def test_the_mc_section_says_that_every_row_agrees_when_none_disagrees_and_that_no_row_is_flat_when_none_is():
    out = report_s3.mc_section(*mc_record(n_agree=1000, n_sd_zero=0, n_sd_zero_disagree=0, share=1.0))
    assert out['n_disagree'] == 0 and 'every row agrees' in out['statement']
    out = report_s3.mc_section(*mc_record(n_agree=990, n_sd_zero=0, n_sd_zero_disagree=0, share=0.99))
    assert out['n_disagree'] == 10 and '0 of the 10' in out['statement'] and out['n_disagree_with_spread'] == 10


@pytest.mark.parametrize(
    'change',
    [
        {'n_agree': 1001},
        {'n_sd_zero_disagree': 13},
        {'n_sd_zero_disagree': 9, 'n_agree': 995},
        {'n_sd_zero': 1001},
        {'n_rows': 0, 'n_agree': 0},
    ],
)
def test_the_mc_section_refuses_counts_that_cannot_belong_together(change):
    with pytest.raises(report_s3.ReportInputError):
        report_s3.mc_section(*mc_record(**change))


def test_the_agreement_by_rate_cuts_the_rows_at_the_quantiles_of_the_rate_and_judges_each_bin_by_the_rule_of_the_gate():
    rate = np.arange(1, 11.0)
    mc_mean = np.array([5.0, 5.0] + [0.5] * 8)  # standard error 10 / sqrt(100) = 1, bound 3: the two rows of the lowest rate are far off
    out = report_s3.mc_by_rate(mc_mean, np.full(10, 10.0), 100, np.zeros(10), rate, n_bins=5)
    bins = out['bins']
    assert out['n_bins'] == 5 and [b['bin'] for b in bins] == [0, 1, 2, 3, 4]
    assert [b['n_rows'] for b in bins] == [2] * 5 and [b['n_agree'] for b in bins] == [0, 2, 2, 2, 2]
    assert [b['share'] for b in bins] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert [b['rate_min'] for b in bins] == [1, 3, 5, 7, 9] and [b['rate_max'] for b in bins] == [2, 4, 6, 8, 10]
    assert [b['worst_z'] for b in bins] == pytest.approx([5.0, 0.5, 0.5, 0.5, 0.5])
    assert report_s3.MC_RATE_BINS == 5 and report_s3.mc_by_rate(mc_mean, np.full(10, 10.0), 100, np.zeros(10), rate)['bins'] == bins


def test_the_agreement_by_rate_never_splits_equal_rates_leaves_an_empty_bin_out_and_counts_the_flat_rows_of_each_bin():
    rate = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])  # four bins were asked for, two rates exist: two bins
    mc_mean = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0])
    mc_sd = np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0])  # S = 9: the bound of a row with a spread is 3 x 3 / 3 = 3
    out = report_s3.mc_by_rate(mc_mean, mc_sd, 9, np.zeros(8), rate, n_bins=4)
    low, high = out['bins']
    assert out['n_bins'] == 2 and (low['bin'], high['bin']) == (0, 1) and (low['n_rows'], high['n_rows']) == (4, 4)
    assert (low['rate_min'], low['rate_max'], high['rate_min'], high['rate_max']) == (1.0, 1.0, 5.0, 5.0)
    assert (low['n_agree'], low['n_sd_zero'], low['n_sd_zero_disagree']) == (3, 2, 1)  # the flat row off by 0.2 disagrees
    assert (high['n_agree'], high['n_sd_zero'], high['n_sd_zero_disagree']) == (3, 3, 1)  # the flat row off by 9 disagrees
    single = report_s3.mc_by_rate(mc_mean, mc_sd, 9, np.zeros(8), np.full(8, 2.0), n_bins=5)
    assert single['n_bins'] == 1 and single['bins'][0]['n_rows'] == 8 and single['bins'][0]['n_agree'] == 6


def test_a_row_on_the_edge_of_two_bins_goes_to_the_upper_bin():
    rate = np.array([1.0, 2.0, 2.0, 3.0])  # the median, the one edge of two bins, is 2.0
    low, high = report_s3.mc_by_rate(np.zeros(4), np.ones(4), 100, np.zeros(4), rate, n_bins=2)['bins']
    assert (low['n_rows'], low['rate_min'], low['rate_max']) == (1, 1.0, 1.0)
    assert (high['n_rows'], high['rate_min'], high['rate_max']) == (3, 2.0, 3.0)


def test_the_agreement_by_rate_adds_up_to_the_agreement_of_the_gate():
    from c_ept import gate_s3

    rng = np.random.default_rng(3)
    n, S = 500, 200
    rate, mc_sd = rng.uniform(0.1, 3.0, n), rng.uniform(1.0, 5.0, n)
    mc_sd[::50] = 0.0  # ten flat rows
    analytic = rng.normal(0.0, 1.0, n)
    mc_mean = analytic + rng.normal(0.0, 1.0, n) * mc_sd / np.sqrt(S)
    mc_mean[::100] += 0.5  # five flat rows are off
    whole = gate_s3.mc_agreement(mc_mean, mc_sd, S, analytic)
    out = report_s3.mc_by_rate(mc_mean, mc_sd, S, analytic, rate, n_bins=5)
    for key in ('n_rows', 'n_agree', 'n_sd_zero', 'n_sd_zero_disagree'):
        assert sum(b[key] for b in out['bins']) == whole[key], key
    assert out['n_bins'] == 5 and [b['n_rows'] for b in out['bins']] == [100] * 5
    assert all(a['rate_max'] < b['rate_min'] for a, b in zip(out['bins'], out['bins'][1:]))
    assert out['bins'][0]['rate_min'] == rate.min() and out['bins'][-1]['rate_max'] == rate.max()


@pytest.mark.parametrize(
    'change',
    [
        {'mc_sd': np.ones(4)},
        {'rate': np.arange(4.0)},
        {'analytic': np.zeros((5, 1))},
        {'mc_mean': np.zeros((5, 1)), 'mc_sd': np.ones((5, 1)), 'analytic': np.zeros((5, 1)), 'rate': np.arange(5.0).reshape(5, 1)},
        {'mc_mean': np.zeros(0), 'mc_sd': np.zeros(0), 'analytic': np.zeros(0), 'rate': np.zeros(0)},
        {'rate': np.array([1.0, 2.0, np.nan, 4.0, 5.0])},
        {'rate': np.array([1.0, 2.0, np.inf, 4.0, 5.0])},
        {'n_bins': 0},
        {'n_bins': 2.5},
    ],
)
def test_the_agreement_by_rate_refuses_arrays_that_do_not_belong_together_and_rates_that_are_not_numbers(change):
    args = {'mc_mean': np.zeros(5), 'mc_sd': np.ones(5), 'S': 100, 'analytic': np.zeros(5), 'rate': np.arange(5.0), **change}
    with pytest.raises(report_s3.ReportInputError):
        report_s3.mc_by_rate(**args)


def group_entry(i, **kw):
    """One group of a PRICES.json as ``flowprice.price_table`` writes it (the numbers are only recognisable, not realistic)."""
    entry = {
        'fixed_immediate': False,
        'n_events': 75 + i,
        'n_excluded': 25,
        'n_event_matches': 60,
        'beta': [10.0 * i + r for r in range(7)],
        'beta_ci': [[10.0 * i + r - 1, 10.0 * i + r + 1] for r in range(7)],
        'price_flow': 100.0 + i,
        'price_flow_ci': [90.0 + i, 110.0 + i],
        'pretrend': -3.0 + i,
        'pretrend_ci': [-9.0 + i, 3.0 + i],
        'adopted': True,
        'immediate': 200.0 + i,
        'immediate_ci': [190.0 + i, 210.0 + i],
        'used': 'flow',
        'value': 100.0 + i,
        'n_obj_unwindowed': 4,
        'obj_unwindowed_share': 0.04,
    }
    entry.update(kw)
    return entry


def prices_document(**per_group):
    """The whole PRICES.json: the twelve groups (``per_group``: name -> fields to change) and the counts of the window fit."""
    doc = {
        'n_matches': 19100,
        'n_window_rows': 448209,
        'n_dropped_last': 9069,
        'n_unknown_objectives': 96,
        'kill_gold_coef': 1.12,
        'kill_gold_coef_ci': [1.11, 1.13],
        'B': 2000,
        'seed': 20261021,
        'window_controls_without_variation': [],
        'window_rank': {'live': 48, 'rank': 48},
    }
    doc.update({g: group_entry(i, **per_group.get(g, {})) for i, g in enumerate(report_s3.GROUPS)})
    return doc


def test_the_d7_section_gives_every_group_its_beta_curve_prices_adoption_and_the_share_of_excluded_events():
    doc = prices_document(
        baron={'n_events': 75, 'n_excluded': 25}, dragon={'fixed_immediate': False, 'used': 'immediate', 'adopted': False}
    )
    out = report_s3.d7_section(doc)
    assert out['relative_minutes'] == [-4, -3, -2, -1, 0, 1, 2] and list(out['groups']) == list(report_s3.GROUPS)
    baron = out['groups']['baron']
    assert baron['excluded_event_share'] == 0.25 and baron['beta'] == doc['baron']['beta'] and baron['beta_ci'] == doc['baron']['beta_ci']
    assert (baron['price_flow'], baron['pretrend'], baron['immediate'], baron['value'], baron['used']) == (104.0, 1.0, 204.0, 104.0, 'flow')
    assert baron['adopted'] is True and baron['immediate_ci'] == [194.0, 214.0] and baron['n_events'] == 75 and baron['n_excluded'] == 25
    assert out['groups']['dragon']['adopted'] is False and out['groups']['dragon']['used'] == 'immediate'
    assert 'dragon' in out['used_immediate'] and 'baron' in out['used_flow'] and len(out['used_flow']) + len(out['used_immediate']) == 12
    assert (out['n_matches'], out['n_window_rows'], out['kill_gold_coef'], out['B'], out['seed']) == (19100, 448209, 1.12, 2000, 20261021)


def test_the_d7_section_says_why_each_group_uses_the_value_it_uses():
    doc = prices_document(
        grubs={'used': 'immediate', 'adopted': False},
        tower_nexus={'used': 'immediate', 'fixed_immediate': True},
        herald={'used': 'immediate', 'adopted': True, 'price_flow': None, 'beta': None, 'beta_ci': None},
    )
    groups = report_s3.d7_section(doc)['groups']
    assert 'contains 0' in groups['dragon']['used_because'] and 'excludes 0' in groups['grubs']['used_because']
    assert 'fixed' in groups['tower_nexus']['used_because'] and 'not estimable' in groups['herald']['used_because']
    assert groups['herald']['beta'] is None and groups['herald']['price_flow'] is None  # the flow fields stay None


def test_a_group_without_any_event_has_no_excluded_share():
    out = report_s3.d7_section(prices_document(plate={'n_events': 0, 'n_excluded': 0}))
    assert out['groups']['plate']['excluded_event_share'] is None


@pytest.mark.parametrize(
    'field',
    [
        'beta',
        'beta_ci',
        'price_flow',
        'pretrend',
        'pretrend_ci',
        'adopted',
        'fixed_immediate',
        'used',
        'value',
        'immediate',
        'immediate_ci',
        'n_events',
        'n_excluded',
    ],
)
def test_the_d7_section_names_the_group_and_the_field_that_a_price_table_lacks(field):
    doc = prices_document()
    del doc['baron'][field]
    with pytest.raises(report_s3.ReportInputError, match=rf'baron.*{field}|{field}.*baron'):
        report_s3.d7_section(doc)


def test_the_d7_section_refuses_a_missing_group_a_beta_of_the_wrong_length_and_a_used_value_it_does_not_know():
    doc = prices_document()
    del doc['elder']
    with pytest.raises(report_s3.ReportInputError, match='elder'):
        report_s3.d7_section(doc)
    with pytest.raises(report_s3.ReportInputError, match=r'beta.*atakhan|atakhan.*beta'):
        report_s3.d7_section(prices_document(atakhan={'beta': [0.0] * 6}))
    with pytest.raises(report_s3.ReportInputError, match=r'used.*plate|plate.*used'):
        report_s3.d7_section(prices_document(plate={'used': 'both'}))


# ---- group C2: convergence of B0, the NLL table, the cost record -------------------------------------------------------------------------
FITS = {'poisson': {'alpha': 1e-6, 'max_iter': 1000, 'tol': 1e-8}, 'logistic': {'C': 1.0, 'max_iter': 2000}}  # b0_heads.json "fits"


def b0_summary_record():
    """Four heads of B0: a Poisson head with one converged cell, M3Kills with both cells at the limit, M3Gold (logistic: no n_events) with
    one cell at its limit of 2,000, and an open M3Obj with a cell that did not converge below the limit (a warning of scikit-learn)."""
    return {
        'masked': {
            'M2Rate': {'n_events': [9], 'converged': [True], 'n_iter': [101]},
            'M3Kills': {'n_events': [5, 6], 'converged': [False, False], 'n_iter': [1000, 1000]},
            'M3Gold': {'converged': [False, True], 'n_iter': [2000, 1923], 'teams': {}},
        },
        'open': {'M3Obj': {'n_events': [1, 2, 3], 'converged': [True, True, False], 'n_iter': [226, 608, 777]}},
    }


def test_the_convergence_section_counts_the_cells_that_did_not_converge_and_those_that_hit_the_iteration_limit():
    out = report_s3.convergence_section(b0_summary_record(), FITS)
    assert out['limits'] == {'poisson': 1000, 'logistic': 2000}
    assert (out['n_cells'], out['n_not_converged'], out['n_hit_limit']) == (8, 4, 3)
    assert out['cells_at_limit'] == ['masked.M3Kills#0', 'masked.M3Kills#1', 'masked.M3Gold#0']
    assert out['heads']['masked.M2Rate'] == {
        'family': 'poisson',
        'limit': 1000,
        'n_cells': 1,
        'n_not_converged': 0,
        'n_hit_limit': 0,
        'cells': [{'cell': 0, 'converged': True, 'n_iter': 101, 'hit_limit': False}],
    }
    gold, obj = out['heads']['masked.M3Gold'], out['heads']['open.M3Obj']
    assert gold['family'] == 'logistic' and gold['limit'] == 2000 and gold['n_hit_limit'] == 1 and gold['n_not_converged'] == 1
    assert [c['hit_limit'] for c in gold['cells']] == [True, False] and [c['n_iter'] for c in gold['cells']] == [2000, 1923]
    assert obj['n_not_converged'] == 1 and obj['n_hit_limit'] == 0 and obj['cells'][2]['converged'] is False  # below the limit: a warning


def test_the_convergence_section_reads_the_limits_from_the_fit_record_and_not_from_a_constant():
    out = report_s3.convergence_section(b0_summary_record(), {'poisson': {'max_iter': 700}, 'logistic': {'max_iter': 1950}})
    assert out['limits'] == {'poisson': 700, 'logistic': 1950}
    assert out['cells_at_limit'] == ['masked.M3Kills#0', 'masked.M3Kills#1', 'masked.M3Gold#0', 'open.M3Obj#2']  # 777 >= 700; 1923 < 1950
    assert out['n_hit_limit'] == 4


@pytest.mark.parametrize(
    'summary',
    [
        {'masked': {'M2Rate': {'n_events': [1], 'converged': [True, False], 'n_iter': [101]}}, 'open': {}},  # lists of different lengths
        {'masked': {'M2Rate': {'n_events': [1], 'n_iter': [101]}}, 'open': {}},  # no converged
        {'masked': {'M2Rate': {'n_events': [1], 'converged': True, 'n_iter': 101}}, 'open': {}},  # not lists
        {'masked': {'M2Rate': {'n_events': [1], 'converged': [], 'n_iter': []}}, 'open': {}},  # no cell
    ],
)
def test_the_convergence_section_refuses_a_head_record_it_cannot_read(summary):
    with pytest.raises(report_s3.ReportInputError, match='M2Rate'):
        report_s3.convergence_section(summary, FITS)


def test_the_convergence_section_refuses_fit_limits_that_are_not_recorded():
    with pytest.raises(report_s3.ReportInputError, match='max_iter'):
        report_s3.convergence_section(b0_summary_record(), {'poisson': {'alpha': 1e-6}, 'logistic': {'max_iter': 2000}})


NLL_CELL = {'nll': 0.5, 'nll_ci': [0.4, 0.6], 'n_rows': 10, 'n_design_rows': 12, 'n_unscored': 2}
MASKED_HEADS = ('M2Rate', 'M2Mark', 'M2Zone', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
OPEN_HEADS = ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')


def dev_nll_record():
    """DEV_NLL.json as the runner writes it (its keys come out sorted): masked heads for both families, open heads without the M2 heads."""

    def cells(names, shift):
        return {h: {**NLL_CELL, 'nll': NLL_CELL['nll'] + shift + i / 100} for i, h in enumerate(sorted(names))}

    return {
        'b0': {'masked': cells(MASKED_HEADS, 0.0), 'open': cells(OPEN_HEADS, 0.1)},
        'b1': {'masked': cells(MASKED_HEADS, 0.2), 'open': cells(OPEN_HEADS, 0.3)},
        'b2': {'nll': NLL_CELL, 'brier': 0.17, 'brier_ci': [0.16, 0.18], 'n_rows': 7064},
    }


def test_the_nll_section_is_a_table_of_heads_by_family_and_input_set_in_the_order_of_the_model():
    doc = dev_nll_record()
    out = report_s3.nll_section(doc)
    assert out['columns'] == ['b0_masked', 'b0_open', 'b1_masked', 'b1_open'] and out['heads'] == list(MASKED_HEADS)
    assert list(out['table']) == list(MASKED_HEADS)
    assert out['table']['M2Rate']['b0_open'] is None and out['table']['M2Mark']['b1_open'] is None  # the open sets have no M2 head
    assert out['table']['M3Size']['b0_open'] == doc['b0']['open']['M3Size']
    assert out['table']['M2Zone']['b1_masked'] == doc['b1']['masked']['M2Zone'] and out['table']['M4Conv']['b0_masked']['n_unscored'] == 2
    assert out['b2'] == doc['b2']


def test_the_nll_section_refuses_a_head_it_has_no_place_for_and_a_missing_family():
    doc = dev_nll_record()
    doc['b1']['open']['M9Extra'] = NLL_CELL
    with pytest.raises(report_s3.ReportInputError, match='M9Extra'):
        report_s3.nll_section(doc)
    doc = dev_nll_record()
    del doc['b1']
    with pytest.raises(report_s3.ReportInputError, match='b1'):
        report_s3.nll_section(doc)


LIFETIME_LABEL = 'lifetime peak bound (monotone within a call), not a per-stage peak'


def cost_docs():
    """(manifest, ledger, fit, compose): the parts of RUN.json, STAGES.json, FIT_FACTS.json and COMPOSE.json that the cost record reads, in
    the layout after the G-S3b fix round. The lifetime bound 60 GiB is above the cap of 40 GiB on purpose: it never falls within a call, so
    it says nothing about one stage, and the cap is judged on the sampled peak (2.5 GiB; ruling R43)."""

    def peaks(gib, up, wall):
        return {'wall_s': wall, 'peak_private_gib': gib, 'peak_private_upper_gib': up}

    stage_wall = {'levels': 5.0, 'labels': 1.0, 'fit': 90.0, 'compose': 600.0, 'gate': 1.0}
    manifest = {
        's3_result': {**peaks(2.5, 60.0, 100.0), 'within_memory_cap': True, 'calls': 1, 'stage_wall_s': stage_wall},
        'memory_cap_gib': 40.0,
        'cpu': 'c',
        'gpu': 'g',
        'ram_gib': 61.7,
        'python': 'p',
        'numpy': 'n',
    }

    def stage(wall, gib, up):
        return {
            **peaks(gib, up, wall),
            'complete': True,
            'within_memory_cap': True,
            'e6_check_calls': 4,
            'e6_check_s_total': 1.6,
            'e6_check_s_mean': 0.4,
            'started': 't0',
            'finished': 't1',
        }

    ledger = {
        'stages': {name: stage(wall, 1.8 if name == 'fit' else 0.7, 60.0) for name, wall in stage_wall.items()},
        'calls': [
            {
                **peaks(2.5, 60.0, 100.0),
                'check_calls': 4,
                'stages_run': ['fit'],
                'stages_skipped': [],
                'stopped_by': None,
                'workers': 8,
                'resumed': False,
                'memory_samples': 900,
                'started': 't0',
                'finished': 't1',
            }
        ],
    }

    def head(gib, up, wall, **extra):
        return {**peaks(gib, up, wall), **extra}

    def summary(best, rounds):
        return {
            'summary': {
                'best_iteration': best,
                'rounds_run': rounds,
                'stopped_early': [True] * len(best),
                'num_trees': [2 * r for r in rounds],
            }
        }

    fit = {
        'b0': {**peaks(0.4, 1.0, 5.0), 'setup': peaks(0.1, 0.2, 0.6), 'heads': {'masked.M2Rate': peaks(0.4, 1.0, 2.7)}},
        'b1': {
            **peaks(1.8, 60.0, 80.0),
            'heads': {
                'masked.M2Rate': head(1.8, 60.0, 30.0, loaded=False, **summary([185], [235])),
                'masked.M3Gold': head(1.1, 60.0, 31.0, loaded=False, **summary([33, 25], [83, 75])),
            },
        },
        'tier': peaks(0.1, 0.3, 2.0),
        'b2': peaks(0.6, 1.0, 2.7),
        'dev': peaks(0.9, 1.5, 22.0),
    }
    call = {
        'head_set': 'B0',
        'method': 'expect',
        'rows': 64,
        'block': 64,
        'workers': 8,
        'n_blocks': 1,
        'wall_s': 12.0,
        'seconds_per_block_wall': 12.0,
        'block_compute_s_estimate': 12.0,
        'call_check_calls': 5,
        'call_check_s_total': 3.0,
        'call_check_s_mean': 0.6,
        'call_check_s_longest': 1.0,
        'call_check_budget_s': 4.8,
    }
    compose = {
        **peaks(2.5, 60.0, 600.0),
        'workers': 8,
        'check_calls': 318,
        'check_s_total': 220.0,
        'check_s_mean': 0.7,
        'check_s_longest': 2.9,
        'chunks_total': 2,
        'chunks_loaded': 0,
        'chunks_computed': 2,
        'rows_computed': 90,
        'chunks_by_head_set': {'B0': {'chunks_total': 1, 'chunks_loaded': 0, 'chunks_computed': 1, 'rows_computed': 60}},
        'calls': [call],
        'blocks': {'b0_expect': 64},
        'chunk_rows': 8192,
        'S_b0': 1000,
        'S_b1': 200,
        'rows': 90,
        'matches': 5,
        'b1_rows': 40,
        'b1_sample': 'all the DEV matches',
    }
    return manifest, ledger, fit, compose


def all_keys(obj):
    """Every dict key anywhere in a nested structure of dicts and lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from all_keys(v)


def test_the_lifetime_peak_bound_is_named_and_labelled_and_the_memory_cap_is_judged_on_the_sampled_peak():
    section, missing = report_s3.cost_section(*cost_docs())
    assert 'peak_private_upper_gib' not in set(all_keys(section))  # the field is renamed, never shown as a per-stage peak
    assert report_s3.LIFETIME_PEAK_LABEL == LIFETIME_LABEL and section['memory']['lifetime_peak_bound_label'] == LIFETIME_LABEL
    assert LIFETIME_LABEL in section['field_labels']['lifetime_peak_bound_gib'] and 'R43' in section['field_labels']['peak_private_gib']
    assert section['stages']['fit']['lifetime_peak_bound_gib'] == 60.0 and section['stages']['gate']['lifetime_peak_bound_gib'] == 60.0
    assert (
        section['b1_heads']['masked.M2Rate']['lifetime_peak_bound_gib'] == 60.0
        and section['b0_heads']['setup']['lifetime_peak_bound_gib'] == 0.2
    )
    assert section['models']['tier']['lifetime_peak_bound_gib'] == 0.3 and section['compose']['lifetime_peak_bound_gib'] == 60.0
    assert section['runner_calls'][0]['lifetime_peak_bound_gib'] == 60.0
    memory = section['memory']
    assert (memory['judged_on'], memory['peak_private_gib'], memory['cap_gib'], memory['lifetime_peak_bound_gib']) == (
        'peak_private_gib',
        2.5,
        40.0,
        60.0,
    )
    assert memory['within_cap'] is True and missing == []  # 60 > 40, but the cap is judged on 2.5 (ruling R43)


def test_the_cost_section_holds_time_and_peak_per_stage_per_head_and_the_b1_rounds_per_head_and_cell():
    section, _ = report_s3.cost_section(*cost_docs())
    fit = section['stages']['fit']
    assert (fit['wall_s'], fit['peak_private_gib'], fit['within_memory_cap'], fit['e6_check_calls'], fit['e6_check_s_total']) == (
        90.0,
        1.8,
        True,
        4,
        1.6,
    )
    assert (
        section['run']['wall_s'] == 100.0
        and section['run']['stage_wall_s']['fit'] == 90.0
        and list(section['stages']) == list(run_s3.STAGES)
    )
    assert section['machine'] == {'cpu': 'c', 'gpu': 'g', 'ram_gib': 61.7, 'python': 'p', 'numpy': 'n'}
    assert section['b0_heads']['masked.M2Rate']['wall_s'] == 2.7 and section['b0_heads']['setup']['wall_s'] == 0.6
    gold = section['b1_heads']['masked.M3Gold']
    assert gold['best_iteration'] == [33, 25] and gold['rounds_run'] == [83, 75] and gold['stopped_early'] == [True, True]
    assert section['b1_heads']['masked.M2Rate']['best_iteration'] == [185] and gold['wall_s'] == 31.0 and gold['loaded'] is False
    assert [section['models'][k]['wall_s'] for k in ('tier', 'b0', 'b1', 'b2', 'dev_scores')] == [2.0, 5.0, 80.0, 2.7, 22.0]


def test_the_cost_section_holds_the_e6_check_counters_the_chunks_and_every_compose_call():
    section, _ = report_s3.cost_section(*cost_docs())
    c = section['compose']
    assert c['e6_checks'] == {'check_calls': 318, 'check_s_total': 220.0, 'check_s_mean': 0.7, 'check_s_longest': 2.9}
    assert c['chunks'] == {'chunks_total': 2, 'chunks_loaded': 0, 'chunks_computed': 2, 'rows_computed': 90}
    assert c['chunks_by_head_set']['B0']['rows_computed'] == 60 and c['workers'] == 8 and c['b1_sample'] == 'all the DEV matches'
    call = c['calls'][0]
    assert (call['head_set'], call['method'], call['rows'], call['wall_s'], call['call_check_calls'], call['call_check_budget_s']) == (
        'B0',
        'expect',
        64,
        12.0,
        5,
        4.8,
    )
    assert call['peak_private_gib'] is None and call['lifetime_peak_bound_gib'] is None  # a compose call records no peak of its own


def test_a_run_from_before_the_peak_bound_and_the_chunk_counters_still_reports_with_none_and_names_the_missing_fields():
    manifest, ledger, fit, compose = cost_docs()

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k != 'peak_private_upper_gib' and not k.startswith('chunks_') and k != 'rows_computed'
            }
        return [strip(v) for v in obj] if isinstance(obj, list) else obj

    section, missing = report_s3.cost_section(*[strip(d) for d in (manifest, ledger, fit, compose)])
    assert 'peak_private_upper_gib' in missing and 'chunks_total' in missing and missing == sorted(set(missing))
    assert (
        section['stages']['fit']['lifetime_peak_bound_gib'] is None
        and section['b1_heads']['masked.M3Gold']['lifetime_peak_bound_gib'] is None
    )
    assert section['memory']['lifetime_peak_bound_gib'] is None and section['memory']['within_cap'] is True
    assert section['compose']['chunks'] == {'chunks_total': None, 'chunks_loaded': None, 'chunks_computed': None, 'rows_computed': None}
    assert section['compose']['chunks_by_head_set'] is None
    assert section['stages']['fit']['peak_private_gib'] == 1.8  # what is there is kept


def test_the_counters_of_a_compose_call_of_the_first_runner_are_read_under_their_new_names():
    manifest, ledger, fit, compose = cost_docs()
    call = {k: v for k, v in compose['calls'][0].items() if not k.startswith('call_check')}
    call.update(check_calls=7, check_s_total=2.5, check_s_mean=0.35, check_s_longest=0.9, check_budget_s=2.8)  # no call_ prefix
    compose['calls'] = [call]
    got = report_s3.cost_section(manifest, ledger, fit, compose)[0]['compose']['calls'][0]
    assert (got['call_check_calls'], got['call_check_s_total'], got['call_check_s_mean']) == (7, 2.5, 0.35)
    assert (got['call_check_s_longest'], got['call_check_budget_s']) == (0.9, 2.8)
    assert 'check_calls' not in got and got['head_set'] == 'B0' and got['wall_s'] == 12.0


def test_a_sampled_peak_over_the_cap_is_a_memory_finding_and_an_unmeasured_one_is_unknown():
    manifest, ledger, fit, compose = cost_docs()
    manifest['s3_result']['peak_private_gib'] = 41.5
    assert report_s3.cost_section(manifest, ledger, fit, compose)[0]['memory']['within_cap'] is False
    manifest['s3_result']['peak_private_gib'] = None
    assert report_s3.cost_section(manifest, ledger, fit, compose)[0]['memory']['within_cap'] is None


# ---- group D: the key scan --------------------------------------------------------------------------------------------------------------
class KeyCtx:
    """What the scan needs of a context: ``keys()``, the S1 match keys as text (the real context is ``run_s3.Context``)."""

    def __init__(self, keys):
        self._keys = keys
        self.reads = 0

    def keys(self):
        self.reads += 1
        return self._keys


SCAN_KEYS = ['EUW1_6543210987', 'KR_7012345678', 'TOYKEY_2005']


def test_a_full_match_key_in_the_text_is_refused_and_only_the_number_of_hits_is_named():
    with pytest.raises(RuntimeError) as err:
        report_s3.assert_clean(KeyCtx(SCAN_KEYS), '{"a": 1, "b": "EUW1_6543210987", "c": "KR_7012345678"}')
    assert (
        '2' in str(err.value) and '6543210987' not in str(err.value) and '7012345678' not in str(err.value) and 'EUW1' not in str(err.value)
    )


@pytest.mark.parametrize('text', ['{"match": 7012345678}', '{"match": [7012345678, 3]}', '[ 6543210987 ]', '{"id":6543210987,"k":2}'])
def test_a_bare_game_id_that_stands_alone_as_a_number_is_refused(text):
    with pytest.raises(RuntimeError, match='1 game ids') as err:
        report_s3.assert_clean(KeyCtx(SCAN_KEYS), text)
    assert '7012345678' not in str(err.value) and '6543210987' not in str(err.value)


def test_the_digits_of_a_game_id_inside_a_longer_number_are_not_a_hit_and_neither_is_a_short_id():
    text = '{"x": 0.65432109871234, "y": 16543210987, "z": "run_65432109870", "n": 2005, "w": 70123456789}'
    report_s3.assert_clean(KeyCtx(SCAN_KEYS), text)  # 2005 is the id of TOYKEY_2005, too short (4 digits) to be looked for bare
    with pytest.raises(RuntimeError):
        report_s3.assert_clean(KeyCtx(SCAN_KEYS), 'TOYKEY_2005')  # but its full key is found
    assert report_s3.MIN_GAME_ID_DIGITS == 9


def test_an_id_of_nine_digits_is_looked_for_as_a_whole_number_and_one_of_eight_is_not():
    keys = ['NA1_123456789', 'EUW1_12345678']  # nine digits and eight
    with pytest.raises(report_s3.KeyInReport, match='^1 game ids'):
        report_s3.assert_clean(KeyCtx(keys), '{"a": 123456789}')
    report_s3.assert_clean(KeyCtx(keys), '{"a": 12345678}')
    assert report_s3.game_ids(keys) == {'123456789'}


def test_the_scan_goes_through_the_runner_scan_for_the_full_keys_and_reads_the_keys_through_the_context(monkeypatch):
    seen = []
    monkeypatch.setattr(run_s3.Context, 'assert_no_keys', lambda self, text: seen.append((self, text)))
    ctx = KeyCtx(SCAN_KEYS)
    report_s3.assert_clean(ctx, '{"a": 1}')
    assert seen == [(ctx, '{"a": 1}')]  # the runner's own method does the full-key scan, with this context
    assert ctx.reads == 1


def test_a_context_without_keys_and_a_text_without_digits_pass():
    report_s3.assert_clean(KeyCtx([]), '{"a": 1, "b": 20261021}')
    report_s3.assert_clean(KeyCtx(['EUW1_6543210987']), '{"a": "text"}')


# ---- group E: the gate recomputed without the smoke DEV matches, and the check against the gate ------------------------------------------
HAND = dict(
    match=np.array([10, 10, 11, 11, 11, 12, 12, 13]),
    expected_onsets=np.array([1.0, 2.0, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0]),
    observed=np.array([1, 2, 0, 1, 1, 3, 0, 1]),
    expected_value=np.array([100.0, 200.0, 50.0, 50.0, 100.0, 200.0, 100.0, 100.0]),
    realized=np.array([110.0, 190.0, 0.0, 0.0, 0.0, 230.0, 70.0, 0.0]),
)


def test_leaving_out_the_listed_matches_gives_the_counts_and_the_values_worked_out_by_hand():
    out = report_s3.sensitivity_without_smoke(**HAND, listed=[11, 13, 99], B=100)
    assert out['computed'] is True
    assert (out['n_listed_dev_matches'], out['n_listed_in_this_run']) == (3, 2)
    assert (out['n_rows_all'], out['n_matches_all'], out['n_rows_dropped'], out['n_matches_dropped']) == (8, 4, 4, 2)
    assert (out['n_rows_kept'], out['n_matches_kept'], out['B'], out['seed']) == (4, 2, 100, 20261021)
    # kept: matches 10 and 12, two rows each, every weight 1/2.  occurrence: x = 1,2,2,1  y = 1,2,3,0  ->  oe = 6/6, slope = 2
    assert out['occurrence']['oe'] == pytest.approx(1.0) and out['occurrence']['slope'] == pytest.approx(2.0)
    # value: x = 100,200,200,100  y = 110,190,230,70  ->  slope = 12000 / 10000, bias = (10 - 10 + 30 - 30) / 4
    assert out['value']['slope'] == pytest.approx(1.2) and out['value']['bias'] == pytest.approx(0.0, abs=1e-12)
    assert out['occurrence']['n_rows'] == 4 and out['occurrence']['n_matches'] == 2 and out['value']['n_matches'] == 2
    for part, keys in (('occurrence', ('oe_in_band', 'slope_in_band')), ('value', ('slope_in_band', 'bias_in_band'))):
        assert all(isinstance(out[part][k], bool) for k in keys)
    assert 'all_pass' not in out and 'gate_pass' not in out and 'pass' not in out['occurrence']  # the report never judges the gate again


def test_leaving_out_no_match_gives_the_numbers_of_the_gate_functions_on_all_the_rows():
    from c_ept import features, gate_s3

    out = report_s3.sensitivity_without_smoke(**HAND, listed=[], B=100)
    w = features.match_weights(HAND['match'])
    occ = gate_s3.occurrence_calibration(HAND['expected_onsets'], HAND['observed'], HAND['match'], w, B=100)
    val = gate_s3.value_calibration(HAND['expected_value'], HAND['realized'], HAND['match'], w, B=100)
    assert out['n_rows_dropped'] == 0 and out['n_matches_dropped'] == 0 and out['n_rows_kept'] == 8 and out['n_listed_in_this_run'] == 0
    for key in ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept'):
        assert out['occurrence'][key] == occ[key], key
    for key in ('slope', 'slope_ci', 'bias', 'bias_ci'):
        assert out['value'][key] == val[key], key
    # the weights are 1 / n_m over the rows that are kept, here all of them (4 matches of 2, 3, 2 and 1 rows)
    w_hand = np.array([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1.0])
    assert out['occurrence']['oe'] == pytest.approx(np.sum(w_hand * HAND['observed']) / np.sum(w_hand * HAND['expected_onsets']))


def test_the_kept_matches_keep_their_weights_of_one_over_their_number_of_rows():
    match = np.array([1, 1, 1, 2, 2, 3])  # matches of 3, 2 and 1 rows
    x, y = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 2.0]), np.array([0, 2, 2, 1, 3, 5])
    for gone, weights in ((3, [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]), (2, [1 / 3, 1 / 3, 1 / 3, 1.0]), (1, [0.5, 0.5, 1.0])):
        out = report_s3.sensitivity_without_smoke(match, x, y, x * 100, x * 90, listed=[gone], B=50)
        keep, w = match != gone, np.array(weights)
        assert out['n_rows_kept'] == len(w) and out['occurrence']['oe'] == pytest.approx(np.sum(w * y[keep]) / np.sum(w * x[keep]))


def test_leaving_out_every_match_is_not_computed_and_still_gives_the_counts():
    out = report_s3.sensitivity_without_smoke(**HAND, listed=[10, 11, 12, 13])
    assert out['computed'] is False and out['reason'] and out['n_rows_kept'] == 0 and out['n_rows_dropped'] == 8 and 'occurrence' not in out


def test_the_sensitivity_uses_the_gate_bootstrap_by_default():
    out = report_s3.sensitivity_without_smoke(**HAND, listed=[13])
    assert (out['B'], out['seed']) == (2000, 20261021) and out['occurrence']['B'] == 2000 and out['value']['seed'] == 20261021


def test_the_check_against_the_gate_compares_the_numbers_of_a_line_with_the_occurrence_result_of_the_gate():
    gate = {
        'oe': 1.002,
        'oe_ci': [0.98, 1.025],
        'slope': 1.159,
        'slope_ci': [1.068, 1.257],
        'intercept': -0.197,
        'intercept_ci': [-0.4, 0.01],
        'n_rows': 94491,
        'n_matches': 3938,
        'deciles': [],
    }
    line = {
        'computed': True,
        **{k: gate[k] for k in ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'intercept_ci', 'n_rows', 'n_matches')},
    }
    assert report_s3.compare_with_gate(line, gate) == {'equal': True, 'differing': []}
    line['slope_ci'] = [1.068, 1.2571]
    line['oe'] = 1.0021
    out = report_s3.compare_with_gate(line, gate)
    assert out['equal'] is False and out['differing'] == ['oe', 'slope_ci']
    assert report_s3.compare_with_gate({'computed': False, 'reason': 'x'}, gate)['equal'] is False


def test_a_line_made_with_another_bootstrap_than_the_gate_is_not_compared_with_it():
    gate = {
        'oe': 1.002,
        'oe_ci': [0.98, 1.025],
        'slope': 1.159,
        'slope_ci': [1.068, 1.257],
        'intercept': -0.197,
        'intercept_ci': [-0.4, 0.01],
        'n_rows': 94491,
        'n_matches': 3938,
        'B': 2000,
        'seed': 20261021,
    }
    line = {'computed': True, **{k: gate[k] for k in report_s3.GATE_COMPARED}, 'B': 50, 'seed': 20261021}
    out = report_s3.compare_with_gate(line, gate)
    assert out['equal'] is None and out['differing'] == [] and 'B' in out['reason']
    assert report_s3.compare_with_gate({**line, 'B': 2000}, gate)['equal'] is True


def test_the_check_against_the_gate_also_compares_the_intercept_interval_and_the_counts_of_rows_and_matches():
    gate = {
        'oe': 1.002,
        'oe_ci': [0.98, 1.025],
        'slope': 1.159,
        'slope_ci': [1.068, 1.257],
        'intercept': -0.197,
        'intercept_ci': [-0.4, 0.01],
        'n_rows': 94491,
        'n_matches': 3938,
    }
    assert set(report_s3.GATE_COMPARED) == {*gate}  # what a window line shares with the occurrence result of the gate
    line = {'computed': True, **gate}
    assert report_s3.compare_with_gate(line, gate) == {'equal': True, 'differing': []}
    for key, other in (('intercept_ci', [-0.4, 0.02]), ('n_rows', 94490), ('n_matches', 3937)):
        assert report_s3.compare_with_gate({**line, key: other}, gate) == {'equal': False, 'differing': [key]}
    two = {**line, 'n_matches': 3937, 'intercept_ci': [-0.4, 0.02]}  # two numbers differ: they are named in the order of GATE_COMPARED
    assert report_s3.compare_with_gate(two, gate) == {'equal': False, 'differing': ['intercept_ci', 'n_matches']}
    assert report_s3.compare_with_gate({**line, 'n_rows': 94491.0}, gate)['equal'] is True  # the same count as a float is the same count


# ---- group F: the value statistics and the composition summary of B1 ---------------------------------------------------------------------
@pytest.mark.parametrize(
    'mean, sd, S, w, want',
    [
        ([0.0, 100.0, 200.0, 300.0], [50.0] * 4, 100, [1.0] * 4, 0.998),  # Var = 12500, se2 = 25
        ([0.0, 100.0], [10.0, 30.0], 1, [3.0, 1.0], 0.84),  # xbar = 25, Var = 1875, Ew[se2] = (3 * 100 + 900) / 4 = 300
        ([0.0, 1.0], [100.0, 100.0], 1, [1.0, 1.0], 0.0),  # the noise is larger than the spread: clipped at 0
        ([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], 1, [1.0] * 3, None),  # no spread, no slope
    ],
)
def test_the_slope_that_a_calibrated_model_shows_is_cut_down_by_the_error_of_the_monte_carlo_mean(mean, sd, S, w, want):
    got = report_s3.expected_slope_if_calibrated(np.array(mean), np.array(sd), S, np.array(w))
    assert got is None if want is None else got == pytest.approx(want)


@pytest.mark.parametrize('S', [0, -3, 2.5, True, None])
def test_the_number_of_simulations_of_a_monte_carlo_mean_must_be_a_positive_whole_number(S):
    with pytest.raises(report_s3.ReportInputError, match='positive whole number'):
        report_s3.expected_slope_if_calibrated(np.array([0.0, 1.0]), np.array([1.0, 1.0]), S, np.ones(2))


def test_a_value_summary_is_the_weighted_calibration_with_the_weighted_means_beside_it():
    out = report_s3.value_summary(
        np.array([100.0, 200.0, 200.0, 100.0]), np.array([110.0, 190.0, 230.0, 70.0]), np.array([1, 1, 2, 2]), np.full(4, 0.5), B=60
    )
    assert out['slope'] == pytest.approx(1.2) and out['bias'] == pytest.approx(0.0, abs=1e-12)
    assert out['mean_expected'] == pytest.approx(150.0) and out['mean_realized'] == pytest.approx(150.0)
    assert (
        (out['n_rows'], out['n_matches'], out['B'], out['seed']) == (4, 2, 60, 20261021)
        and len(out['slope_ci']) == 2
        and len(out['bias_ci']) == 2
    )
    assert isinstance(out['slope_in_band'], bool) and isinstance(out['bias_in_band'], bool)
    assert 'deciles' not in out  # the summary keeps the numbers, not the ten bins


def test_values_that_equal_their_expectation_give_slope_one_and_bias_zero():
    rng = np.random.default_rng(42)
    x = rng.uniform(10.0, 200.0, 40)
    out = report_s3.value_summary(x, x.copy(), np.repeat(np.arange(8), 5), np.full(40, 0.2), B=50)
    assert out['slope'] == pytest.approx(1.0) and out['bias'] == pytest.approx(0.0, abs=1e-12)


def composition_world(n_matches=14, seed=11, keep_every=3):
    """(b0, b1, realized, observed): the arrays of b0_dev.npz for every row of ``n_matches`` DEV matches and of b1_dev.npz for every
    ``keep_every``-th match (the same rows, a subset), the realized V_start at 60 s of the B0 rows and the observed onsets of the three
    windows. The predictive laws are normal, so the quantiles are exact."""
    from statistics import NormalDist

    z = np.array([NormalDist().inv_cdf(p) for p in report_s3.LEVELS])
    rng = np.random.default_rng(seed)
    sizes = rng.integers(6, 14, n_matches)
    match = np.repeat(100 + np.arange(n_matches), sizes)
    n = len(match)
    cum = np.cumsum(rng.uniform(0.05, 0.6, (n, 3)), axis=1)
    center = 20.0 * rng.normal(size=(n, 3)) + np.array([0.0, 30.0, 60.0])
    sd = rng.uniform(200.0, 400.0, (n, 3))

    def head_arrays(rows, S, noise):
        mc_mean = center[rows] + noise * rng.normal(size=center[rows].shape)
        return {
            'frame_row': 1000 + rows,
            'match': match[rows],
            'expected_onsets': cum[rows],
            'mc_mean': mc_mean,
            'mc_sd': sd[rows],
            'q19': mc_mean[:, :, None] + sd[rows][:, :, None] * z,
            'S': np.array([S]),
            'seed': np.array([20261021]),
        }

    b0 = head_arrays(np.arange(n), 1000, 5.0)
    b0['expected'] = center
    rows1 = np.flatnonzero(np.isin(match, 100 + np.arange(0, n_matches, keep_every)))
    b1 = head_arrays(rows1, 200, 12.0)
    realized = center[:, 0] + sd[:, 0] * rng.normal(size=n)
    observed = rng.poisson(np.diff(cum, axis=1, prepend=0.0))
    return b0, b1, realized, observed


def test_the_composition_summary_of_b1_is_the_gate_statistics_on_the_sampled_rows_with_b0_on_the_same_rows():
    from c_ept import features, gate_s3

    b0, b1, realized, observed = composition_world()
    out = report_s3.b1_composition(b0, b1, realized, observed, sample='a seeded sample of 500 DEV matches', B=50)
    pos = np.searchsorted(b0['frame_row'], b1['frame_row'])  # the rows of B1 among the rows of B0
    y, w = realized[pos], features.match_weights(b1['match'])
    assert (out['rows'], out['matches'], out['dev_rows'], out['S_b1'], out['S_b0']) == (
        len(pos),
        len(np.unique(b1['match'])),
        len(realized),
        200,
        1000,
    )
    assert out['sample'] == 'a seeded sample of 500 DEV matches' and out['horizon_s'] == 60
    assert out['share_of_dev_rows'] == pytest.approx(len(pos) / len(realized))
    for block, mean, sd, q19 in (
        (out['b1'], b1['mc_mean'], b1['mc_sd'], b1['q19']),
        (out['b0_same_rows'], b0['mc_mean'][pos], b0['mc_sd'][pos], b0['q19'][pos]),
    ):
        want = report_s3.mc_value_summary(mean[:, 0], sd[:, 0], block['S'], y, b1['match'], w, B=50)  # a Monte Carlo mean: no band flags
        assert block['value_simulated'] == want and block['crps19'] == pytest.approx(report_s3.crps19(q19[:, 0, :], y, w))
        assert block['pit']['mass'] == pytest.approx(report_s3.pit_histogram(q19[:, 0, :], y, w)['mass'])
        assert block['mc_se']['expected_slope_if_calibrated'] == report_s3.expected_slope_if_calibrated(mean[:, 0], sd[:, 0], block['S'], w)
    want = report_s3.value_summary(b0['expected'][pos, 0], y, b1['match'], w, B=50)
    assert out['b0_same_rows']['value_closed_form'] == want and 'value_closed_form' not in out['b1']  # B1 has no closed form
    for block, cum in ((out['b1'], b1['expected_onsets']), (out['b0_same_rows'], b0['expected_onsets'][pos])):
        occ = gate_s3.occurrence_calibration(cum[:, 0], observed[pos, 0], b1['match'], w, B=50)
        assert (block['occurrence_60']['oe'], block['occurrence_60']['slope'], block['occurrence_60']['slope_ci']) == (
            occ['oe'],
            occ['slope'],
            occ['slope_ci'],
        )


def test_the_composition_summary_says_that_b1_has_fewer_simulations_and_where_the_attenuation_comes_from():
    b0, b1, realized, observed = composition_world()
    out = report_s3.b1_composition(b0, b1, realized, observed, sample='x', B=40)
    assert 'S = 200' in out['note'] and 'attenuat' in out['note'] and 'report only' in out['note']
    assert (
        out['b1']['mc_se']['rms_gold'] > out['b0_same_rows']['mc_se']['rms_gold']
    )  # S = 200 against S = 1,000 (the noise 12 against 5 in the toy)


def test_the_composition_summary_refuses_sampled_rows_that_are_not_rows_of_the_dev_table():
    b0, b1, realized, observed = composition_world()
    b1['frame_row'] = b1['frame_row'] + 1  # no longer the frame rows of B0
    with pytest.raises(report_s3.ReportInputError, match='B1'):
        report_s3.b1_composition(b0, b1, realized, observed, sample='x', B=40)
    b0, b1, realized, observed = composition_world()
    b1['match'] = b1['match'] + 1  # the same rows, another match
    with pytest.raises(report_s3.ReportInputError, match='B1'):
        report_s3.b1_composition(b0, b1, realized, observed, sample='x', B=40)


# ---- group G: the findings at the top of the report --------------------------------------------------------------------------------------
def quiet_doc():
    """The parts of a report that the findings read, in a state that gives no finding."""
    line = {'computed': True}
    return {
        'run': {'smoke': False},
        'cost': {'memory': {'within_cap': True}, 'b1_rounds_missing': {}},
        'b0_convergence': {'n_hit_limit': 0, 'n_cells': 100},
        'window_slopes': {
            'check_against_gate': {'equal': True, 'differing': []},
            'b0': {'windows': [line] * 4},
            'b1': {'windows': [line] * 4},
        },
        'mc_agreement': {'n_sd_zero_disagree': 0, 'n_sd_zero': 0},
        'inputs': {'missing_optional_fields': []},
        'sensitivity_without_smoke_dev': {'requested': False, 'computed': False},
        'accounting': {'value_check': {'max_abs_value_error': 1.5e-08, 'tolerance_gold': 1e-06, 'pass': True}},
    }


def test_a_report_with_nothing_to_note_has_no_findings():
    assert report_s3.findings(quiet_doc()) == []


@pytest.mark.parametrize(
    'code, change',
    [
        ('SMOKE_RUN', lambda d: d['run'].update(smoke=True)),
        ('MEMORY_OVER_CAP', lambda d: d['cost']['memory'].update(within_cap=False)),
        ('MEMORY_NOT_MEASURED', lambda d: d['cost']['memory'].update(within_cap=None)),
        ('B0_CELLS_AT_ITERATION_LIMIT', lambda d: d['b0_convergence'].update(n_hit_limit=3)),
        ('GATE_RECOMPUTATION_DIFFERS', lambda d: d['window_slopes']['check_against_gate'].update(equal=False, differing=['slope'])),
        ('MC_FLAT_ROW_DISAGREEMENT', lambda d: d['mc_agreement'].update(n_sd_zero_disagree=9, n_sd_zero=12)),
        (
            'WINDOW_NOT_COMPUTED',
            lambda d: d['window_slopes']['b1']['windows'].__setitem__(1, {'computed': False, 'name': '60-120 s', 'reason': 'r'}),
        ),
        (
            'SENSITIVITY_NOT_COMPUTED',
            lambda d: d['sensitivity_without_smoke_dev'].update(requested=True, computed=False, reason='none left'),
        ),
        ('OLD_RECORD_LAYOUT', lambda d: d['inputs'].update(missing_optional_fields=['peak_private_upper_gib'])),
        (
            'ACCOUNTING_GOLD_OVER_TOLERANCE',
            lambda d: d['accounting']['value_check'].update({'pass': False, 'max_abs_value_error': 0.5}),
        ),
        ('B1_ROUNDS_NOT_RECORDED', lambda d: d['cost'].update(b1_rounds_missing={'stopped_early': ['masked.M2Rate']})),
    ],
)
def test_each_finding_has_its_own_code_and_a_message_with_the_numbers_behind_it(code, change):
    doc = quiet_doc()
    doc['window_slopes']['b1']['windows'] = list(doc['window_slopes']['b1']['windows'])
    change(doc)
    out = report_s3.findings(doc)
    assert [f['code'] for f in out] == [code] and out[0]['message'] and isinstance(out[0]['message'], str)


def test_the_messages_of_the_findings_carry_their_numbers():
    doc = quiet_doc()
    doc['b0_convergence'].update(n_hit_limit=3, n_cells=100)
    doc['mc_agreement'].update(n_sd_zero_disagree=9, n_sd_zero=12)
    doc['inputs'].update(missing_optional_fields=['chunks_total', 'peak_private_upper_gib'])
    messages = {f['code']: f['message'] for f in report_s3.findings(doc)}
    assert '3 of 100' in messages['B0_CELLS_AT_ITERATION_LIMIT'] and 'B4' in messages['B0_CELLS_AT_ITERATION_LIMIT']
    assert '9' in messages['MC_FLAT_ROW_DISAGREEMENT'] and 'R28' in messages['MC_FLAT_ROW_DISAGREEMENT']
    assert 'chunks_total' in messages['OLD_RECORD_LAYOUT'] and 'peak_private_upper_gib' in messages['OLD_RECORD_LAYOUT']


# ---- group H: a toy run made by the real runner, and the report of it --------------------------------------------------------------------
@pytest.fixture(scope='module', autouse=True)
def _no_real_runs_root(tmp_path_factory):
    """No test of this file reads a real run folder: where the report looks for a run next to the run folder and then in the runs root,
    the runs root is an empty folder of pytest's (the real ``D:\\LOL_Data\\runs`` holds an S1 run of the same name as the toy's)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, 'RUNS_ROOT', tmp_path_factory.mktemp('no_runs_here'))
        yield


def build_toy_run(world, out: Path) -> Path:
    """The real runner (all five stages, a full run: no smoke, two spawned workers) on the random world of ``test_run_s3``, whose s3a folder
    gets a full D7 prices document first (the prices of the toy, the fields of a real PRICES.json): the decision record is written again
    with the new hash, so that the runner accepts the file. Returns the run folder (under ``out / 'runs'``)."""
    prices = world.s3a_run / 'prices' / 'PRICES.json'
    prices.write_text(json.dumps(prices_document(**{g: {'value': v} for g, v in TOY_PRICES.items()})), encoding='utf-8')
    write_spec(world.spec, sha_of(prices), sha_of(world.s3a_run / 'rules' / 'RULE_PAYOUTS.json'))
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(baselines_b0, 'ALPHA', 1.0)  # the strong ridge of the e2e fixture: a few hundred rows give no absurd rates
        with empty_process_list(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = run_s3.main(cli_args(out, world.s1, world.s2, world.s3a_run, world.spec, '--workers', '1'), records=random_value_record)
    assert rc == 0, stderr.getvalue()[-3000:]
    return sorted((out / 'runs').iterdir())[0]


@pytest.fixture(scope='module')
def toy_run(world, tmp_path_factory):
    """The run folder of ``build_toy_run`` and the inputs the report is given (the S1 run and the s3a run are not next to it)."""
    out = tmp_path_factory.mktemp('report_s3_run')
    run = build_toy_run(world, out)
    yield types.SimpleNamespace(run=run, s1=world.s1, s3a=world.s3a_run, world=world, out=out)
    gc.collect()
    shutil.rmtree(out, ignore_errors=True)


ITEM_SECTIONS = (
    'd7',
    'dev_nll',
    'brier',
    'occurrence_deciles',
    'value_deciles',
    'mc_agreement',
    'overdispersion',
    'accounting',
    'winner',
    'dropped_columns',
    'b1_composition',
)  # the items 1-11 of the plan, in order
CARRY_SECTIONS = ('sensitivity_without_smoke_dev', 'window_slopes', 'b0_convergence', 'cost', 'key_scan')  # the carries 3-7
FIGURES = ('fig1_d7_beta', 'fig2_occurrence_deciles', 'fig3_value_deciles_pit', 'fig4_window_slopes', 'fig5_brier')


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def reported(toy_run):
    """The report of the toy run, in the report folder of the run itself, with the gate's own B = 2,000 and seed."""
    path = report_s3.report(toy_run.run, s1=toy_run.s1, s3a=toy_run.s3a)
    text = path.read_text(encoding='utf-8')
    return types.SimpleNamespace(path=path, text=text, doc=json.loads(text), dir=path.parent, run=toy_run.run)


def copy_of(toy_run, tmp_path) -> Path:
    """A copy of the finished run folder (without a report folder) to damage."""
    dest = tmp_path / toy_run.run.name
    shutil.copytree(toy_run.run, dest, ignore=shutil.ignore_patterns('report'))
    return dest


def ask(toy_run, run, **kw):
    return report_s3.report(run, s1=toy_run.s1, s3a=toy_run.s3a, **kw)


def toy_keys(toy_run) -> list:
    """The match keys of the toy S1 store (sentinels: ``TOYKEY_nnnn``), read here to look for them in what the report wrote."""
    from c_ept.run_s1 import load_s1

    s1 = load_s1(toy_run.s1, verify=False)
    keys = [k.decode('ascii') for k in np.asarray(s1.array('matches/key')).tolist() if k]
    s1.maps.clear()
    return keys


def test_the_report_goes_into_the_report_folder_of_the_run_and_has_a_section_for_every_item_and_every_carry(reported, toy_run):
    assert reported.path == toy_run.run / 'report' / 'S3_REPORT.json' and reported.doc['report'] == 'S3_REPORT'
    for key in ITEM_SECTIONS + CARRY_SECTIONS + ('run', 'inputs', 'gate', 'flags', 'figures', 'tower_pricing', 'created', 'report_code'):
        assert key in reported.doc, key
    assert list(reported.doc['figures']) == list(FIGURES) and isinstance(reported.doc['flags'], list)


# -- the items 1-11 hold what the files of the run hold
def test_item_1_the_d7_section_is_the_price_table_of_the_s3a_run_with_the_share_of_excluded_events(reported, toy_run):
    prices = read_json(toy_run.s3a / 'prices' / 'PRICES.json')
    d7 = reported.doc['d7']
    assert d7['group_order'] == list(report_s3.GROUPS) and set(d7['groups']) == set(
        report_s3.GROUPS
    )  # a JSON object has no order: the list has
    assert d7['relative_minutes'] == [-4, -3, -2, -1, 0, 1, 2]
    for g in report_s3.GROUPS:
        for field in (
            'beta',
            'beta_ci',
            'price_flow',
            'pretrend',
            'pretrend_ci',
            'adopted',
            'immediate',
            'immediate_ci',
            'used',
            'value',
            'n_events',
            'n_excluded',
            'fixed_immediate',
        ):
            assert d7['groups'][g][field] == prices[g][field], (g, field)
        assert d7['groups'][g]['excluded_event_share'] == pytest.approx(
            prices[g]['n_excluded'] / (prices[g]['n_events'] + prices[g]['n_excluded'])
        )
    assert d7['kill_gold_coef'] == prices['kill_gold_coef'] and d7['n_matches'] == prices['n_matches']


def test_item_2_the_nll_table_holds_every_head_for_b0_and_b1_masked_and_open(reported, toy_run):
    doc, raw = reported.doc['dev_nll'], read_json(toy_run.run / 'fit' / 'DEV_NLL.json')
    assert doc['columns'] == ['b0_masked', 'b0_open', 'b1_masked', 'b1_open'] and doc['heads'] == list(MASKED_HEADS)
    for head in MASKED_HEADS:
        for family in ('b0', 'b1'):
            assert doc['table'][head][f'{family}_masked'] == raw[family]['masked'][head]
            assert doc['table'][head][f'{family}_open'] == (raw[family]['open'][head] if head in OPEN_HEADS else None)
    assert doc['b2'] == raw['b2'] and doc['table']['M2Rate']['b0_masked']['nll'] is not None


def test_item_3_the_brier_section_compares_m2_of_b0_and_b1_with_b2(reported, toy_run):
    brier, gate = reported.doc['brier'], read_json(toy_run.run / 'gate' / 'GATE.json')['brier']
    for key in ('m2_b0', 'm2_b1', 'b2', 'difference', 'n_rows', 'n_matches', 'positive_share', 'B', 'seed'):
        assert brier[key] == gate[key], key
    assert set(brier['difference']) == {'m2_b0_minus_b2', 'm2_b1_minus_b2', 'm2_b0_minus_m2_b1'} and 'not judged' in brier['note']


def test_item_4_the_occurrence_section_holds_the_deciles_and_the_intervals_of_the_gate(reported, toy_run):
    gate = read_json(toy_run.run / 'gate' / 'GATE.json')['results']['occurrence']
    doc = reported.doc['occurrence_deciles']
    assert len(doc['deciles']) == 10 and doc['deciles'] == gate['deciles'] and doc['horizon_s'] == 60
    for key in ('oe', 'oe_ci', 'slope', 'slope_ci', 'intercept', 'n_rows', 'n_matches', 'B', 'seed'):
        assert doc[key] == gate[key], key
    assert doc['bands'] == {'oe': [0.95, 1.05], 'slope': [0.9, 1.1]}


def brute_pit(q19, y, w):
    """The PIT histogram written out row by row (a loop, not the vectorised function)."""
    mass = [0.0] * 20
    for q, v, weight in zip(q19, y, w):
        low, high = sum(1 for x in q if x < v), sum(1 for x in q if x <= v)
        for j in range(low, high + 1):
            mass[j] += weight / (high - low + 1)
    return [m / sum(w) for m in mass]


def test_item_5_the_value_section_holds_the_deciles_and_the_pit_of_h_60_s_for_b0_and_b1(reported, toy_run):
    from c_ept import features
    from c_ept.store_core import Store

    gate = read_json(toy_run.run / 'gate' / 'GATE.json')['results']['value']
    doc = reported.doc['value_deciles']
    assert len(doc['deciles']) == 10 and doc['deciles'] == gate['deciles'] and doc['horizon_s'] == 60
    for key in ('slope', 'slope_ci', 'bias', 'bias_ci', 'n_rows', 'n_matches'):
        assert doc[key] == gate[key], key
    with np.load(toy_run.run / 'compose' / 'b0_dev.npz') as z:
        frame_row, match, q19 = z['frame_row'], z['match'], z['q19'][:, 0, :]
    labels = Store(toy_run.run / 'labels' / 'store')
    realized = np.asarray(labels.array('labels/v_start'))[frame_row, 0]
    labels.maps.clear()
    mass = brute_pit(q19, realized, features.match_weights(match))
    assert doc['pit']['b0']['mass'] == pytest.approx(mass, abs=1e-12) and sum(doc['pit']['b0']['mass']) == pytest.approx(1.0)
    assert doc['pit']['b0']['n_rows'] == len(frame_row) and doc['pit']['b0']['bins'] == 20 and len(doc['pit']['b1']['mass']) == 20
    assert doc['pit']['b1'] == reported.doc['b1_composition']['b1']['pit']
    assert doc['crps19']['b0'] > 0 and doc['crps19']['b1'] > 0 and set(doc['quantile_hits']) == {'b0', 'b1'}


def test_item_6_the_simulation_against_the_closed_form_comes_with_the_flat_rows_of_ruling_r28(reported, toy_run):
    results = read_json(toy_run.run / 'gate' / 'GATE.json')['results']
    mc = reported.doc['mc_agreement']
    for key in ('share', 'n_rows', 'n_agree', 'n_sd_zero', 'n_sd_zero_disagree', 'worst_z', 'S'):
        assert mc[key] == results['mc'][key], key
    assert mc['n_disagree'] == mc['n_rows'] - mc['n_agree'] and mc['n_disagree_with_spread'] == mc['n_disagree'] - mc['n_sd_zero_disagree']
    assert 'flat' in mc['statement'] and 'R28' in mc['statement'] and mc['min_share'] == 0.99 and isinstance(mc['pass'], bool)


def test_item_6_the_agreement_by_the_expected_onsets_of_the_row_adds_up_to_the_counts_of_the_gate(reported, toy_run):
    mc = reported.doc['mc_agreement']
    by = mc['by_expected_onsets']
    bins = by['bins']
    assert by['n_bins'] == len(bins) >= 2 and [b['bin'] for b in bins] == list(range(len(bins)))
    for key in ('n_rows', 'n_agree', 'n_sd_zero', 'n_sd_zero_disagree'):
        assert sum(b[key] for b in bins) == mc[key], key
    arrays = load_arrays(toy_run.run / 'compose' / 'b0_dev.npz')  # the columns of the first 60 s, and the S of the run
    rate, S = arrays['expected_onsets'][:, 0], int(np.ravel(arrays['S'])[0])
    assert bins[0]['rate_min'] == rate.min() and bins[-1]['rate_max'] == rate.max() and S == mc['S']
    assert bins == report_s3.mc_by_rate(arrays['mc_mean'][:, 0], arrays['mc_sd'][:, 0], S, arrays['expected'][:, 0], rate)['bins']
    assert all(a['rate_max'] < b['rate_min'] for a, b in zip(bins, bins[1:]))
    assert 'expected_onsets' in by['variable'] and '60 s' in by['variable'] and by['note']


def test_item_7_the_overdispersion_section_is_phi_with_its_interval(reported, toy_run):
    gate = read_json(toy_run.run / 'gate' / 'GATE.json')
    doc = reported.doc['overdispersion']
    for key in ('phi', 'phi_ci', 'overdispersed', 'n_rows', 'n_zero_mu', 'n_matches', 'B', 'seed'):
        assert doc[key] == gate['results']['overdispersion'][key], key
    assert doc['limit'] == gate['verdict']['overdispersion']['limit'] == 1.1


def test_item_8_the_accounting_section_holds_the_identity_for_kill_gold_and_objective_value(reported, toy_run):
    raw = read_json(toy_run.run / 'labels' / 'ACCOUNTING.json')
    doc = reported.doc['accounting']
    assert doc['max_abs_error'] == raw['max_abs_error'] == 0.0 and doc['max_abs_value_error'] == raw['max_abs_value_error']
    assert doc['n_matches'] == raw['n_matches'] and doc['n_valued_matches'] == raw['n_valued_matches'] and doc['total'] == raw['total']
    assert doc['gate_pass'] is True and doc['must_equal'] == 0.0


def test_item_9_the_winner_section_holds_the_shares_of_winners_and_draws(reported, toy_run):
    raw = read_json(toy_run.run / 'labels' / 'LABELS_SUMMARY.json')
    doc = reported.doc['winner']
    for key in ('winner_blue', 'winner_red', 'draw', 'sure', 'n_units', 'n_valued_units', 'n_rows', 'n_rows_with_value'):
        assert doc[key] == raw[key], key
    assert doc['sure_threshold_gold'] == 150.0 and doc['winner_blue'] + doc['winner_red'] + doc['draw'] == pytest.approx(1.0)


def test_item_10_the_dropped_columns_are_listed_with_the_views_they_were_judged_on(reported, toy_run):
    raw = read_json(toy_run.run / 'fit' / 'FIT_INPUTS.json')
    doc = reported.doc['dropped_columns']
    assert doc['dropped'] == raw['dropped'] and doc['n_dropped'] == len(raw['dropped']) and 't_s' in doc['dropped']
    assert doc['n_fit_views'] == raw['n_fit_views'] and doc['n_fit_matches'] == raw['n_fit_matches']
    assert doc['b0_fill_counts'] == {n: c for n, c in zip(raw['b0_names'], raw['b0_fill_counts']) if c} and doc['n_b0_cells_filled'] == sum(
        raw['b0_fill_counts']
    )


def test_item_11_the_b1_composition_is_summarised_on_its_sample_with_b0_on_the_same_rows(reported, toy_run):
    compose = read_json(toy_run.run / 'compose' / 'COMPOSE.json')
    doc = reported.doc['b1_composition']
    assert (
        doc['rows'] == compose['b1_rows'] == compose['rows']
        and doc['sample'] == compose['b1_sample']
        and doc['S_b1'] == 200
        and doc['S_b0'] == 1000
    )
    assert doc['matches'] == len(compose['b1_matches']) and doc['horizon_s'] == 60 and 'report only' in doc['note']
    assert set(doc) >= {'b1', 'b0_same_rows', 'share_of_dev_rows', 'dev_rows'} and 'value_closed_form' in doc['b0_same_rows']


# -- the carries: the slope per window, the convergence of B0, the cost, the gate without the smoke matches
def direct_window_lines(s1_path, arrays):
    """The slope and observed / expected of the three windows and the whole 180 s, from the arrays of a compose file and the S1 onsets, with
    numpy only: the onsets of each window are counted by the bin range (not by differences), the expected counts are the differences of the
    cumulative columns, the weights are 1 / n_m, and the slope is the textbook weighted least-squares formula."""
    from c_ept.run_s1 import load_s1

    s1 = load_s1(s1_path, verify=False)
    row, bin_ = np.asarray(s1.array('onsets/row')).astype(np.int64), np.asarray(s1.array('onsets/bin')).astype(np.int64)
    n_frame_rows = len(np.asarray(s1.array('frame_rows/t0')))
    s1.maps.clear()
    counts = np.stack([np.bincount(row[(bin_ >= lo) & (bin_ < lo + 6)], minlength=n_frame_rows) for lo in (0, 6, 12)], axis=1)[
        arrays['frame_row']
    ]
    cum = arrays['expected_onsets']
    expected = np.maximum(np.diff(cum, axis=1, prepend=0.0), 0.0)
    _, inverse, per_match = np.unique(arrays['match'], return_inverse=True, return_counts=True)
    w = 1.0 / per_match[inverse.reshape(-1)]
    pairs = [(expected[:, j], counts[:, j]) for j in range(3)] + [(cum[:, 2], counts.sum(axis=1))]
    return [direct_line(x, y, w) for x, y in pairs], pairs


def load_arrays(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize('head, file', [('b0', 'b0_dev.npz'), ('b1', 'b1_dev.npz')])
def test_carry_4_the_slope_per_window_is_the_numpy_slope_of_the_onsets_of_that_window(reported, toy_run, head, file):
    arrays = load_arrays(toy_run.run / 'compose' / file)
    lines, pairs = direct_window_lines(toy_run.s1, arrays)
    doc = reported.doc['window_slopes'][head]
    assert (
        [w['name'] for w in doc['windows']] == ['0-60 s', '60-120 s', '120-180 s', '0-180 s']
        and doc['B'] == 2000
        and doc['seed'] == 20261021
    )
    for window, (slope, intercept, oe), (x, y) in zip(doc['windows'], lines, pairs):
        assert window['computed'] is True and window['n_rows'] == len(x) and window['n_matches'] == len(np.unique(arrays['match']))
        assert window['slope'] == pytest.approx(slope, rel=1e-9) and window['oe'] == pytest.approx(oe, rel=1e-12)
        assert window['sum_expected'] == pytest.approx(x.sum()) and window['sum_observed'] == pytest.approx(y.sum())
        assert len(window['slope_ci']) == 2 and window['slope_ci'][0] <= window['slope_ci'][1] and len(window['oe_ci']) == 2
    assert doc['rows'] == len(arrays['frame_row']) and doc['bands'] == {'oe': [0.95, 1.05], 'slope': [0.9, 1.1]}


def test_carry_4_the_window_lines_say_that_b1_covers_only_its_sample_and_b0_the_whole_dev_set(reported):
    slopes = reported.doc['window_slopes']
    assert (
        slopes['b1']['sample'] == reported.doc['b1_composition']['sample']
        and 'B1' in slopes['b1']['note']
        and 'S = 200' in slopes['b1']['note']
    )
    assert slopes['b0']['rows'] == reported.doc['b1_composition']['dev_rows'] and 'all' in slopes['b0']['note']


def test_the_note_on_the_coverage_of_b1_says_when_it_covers_only_a_sample():
    full = report_s3.b1_coverage_note('all the DEV matches', 218, 218)
    part = report_s3.b1_coverage_note('a seeded sample of 500 DEV matches', 12000, 94500)
    assert 'all' in full and 'only' not in full
    assert 'only' in part and 'a seeded sample of 500 DEV matches' in part and '12,000' in part and '94,500' in part


def test_carry_4_the_first_window_of_b0_is_the_occurrence_result_of_the_gate(reported):
    check = reported.doc['window_slopes']['check_against_gate']
    assert check['equal'] is True and check['differing'] == [] and 'GATE.json' in check['compared']
    assert 'GATE_RECOMPUTATION_DIFFERS' not in [f['code'] for f in reported.doc['flags']]


def test_carry_5_the_convergence_of_every_b0_cell_comes_from_the_fit_record(reported, toy_run):
    facts = read_json(toy_run.run / 'fit' / 'FIT_FACTS.json')
    heads = read_json(toy_run.run / 'fit' / 'b0' / 'b0_heads.json')
    doc = reported.doc['b0_convergence']
    assert doc['limits'] == {'poisson': heads['fits']['poisson']['max_iter'], 'logistic': heads['fits']['logistic']['max_iter']}
    cells = [
        (f'{kind}.{name}', i, c, n)
        for kind in ('masked', 'open')
        for name, s in facts['b0']['summary'][kind].items()
        for i, (c, n) in enumerate(zip(s['converged'], s['n_iter']))
    ]
    assert doc['n_cells'] == len(cells) and doc['n_not_converged'] == sum(not c for _, _, c, _ in cells)
    for label, i, converged, n_iter in cells:
        assert doc['heads'][label]['cells'][i]['converged'] == converged and doc['heads'][label]['cells'][i]['n_iter'] == n_iter
    assert doc['n_hit_limit'] == len(doc['cells_at_limit']) and doc['n_hit_limit'] <= doc['n_not_converged']


def test_carry_6_the_cost_record_has_the_stages_the_heads_the_compose_calls_and_the_checks(reported, toy_run):
    ledger, facts, compose = (
        read_json(toy_run.run / 'STAGES.json'),
        read_json(toy_run.run / 'fit' / 'FIT_FACTS.json'),
        read_json(toy_run.run / 'compose' / 'COMPOSE.json'),
    )
    cost = reported.doc['cost']
    for stage in run_s3.STAGES:
        assert cost['stages'][stage]['wall_s'] == ledger['stages'][stage]['wall_s'] > 0
        assert cost['stages'][stage]['peak_private_gib'] == ledger['stages'][stage]['peak_private_gib']
        assert cost['stages'][stage]['lifetime_peak_bound_gib'] == ledger['stages'][stage]['peak_private_upper_gib']
    assert set(cost['b0_heads']) == {'setup', *facts['b0']['heads']} and len(cost['b0_heads']) == 14
    assert set(cost['b1_heads']) == set(facts['b1']['heads']) and len(cost['b1_heads']) == 13
    for label, head in facts['b1']['heads'].items():
        assert cost['b1_heads'][label]['best_iteration'] == head['summary']['best_iteration']
        assert (
            cost['b1_heads'][label]['rounds_run'] == head['summary']['rounds_run'] and cost['b1_heads'][label]['wall_s'] == head['wall_s']
        )
    c = cost['compose']
    assert c['e6_checks'] == {k: compose[k] for k in ('check_calls', 'check_s_total', 'check_s_mean', 'check_s_longest')}
    assert c['chunks'] == {k: compose[k] for k in ('chunks_total', 'chunks_loaded', 'chunks_computed', 'rows_computed')}
    assert len(c['calls']) == len(compose['calls']) >= 5 and c['calls'][0]['call_check_calls'] == compose['calls'][0]['call_check_calls']
    assert [(x['head_set'], x['method']) for x in c['calls']] == [(x['head_set'], x['method']) for x in compose['calls']]
    assert cost['memory']['peak_private_gib'] == read_json(toy_run.run / 'RUN.json')['s3_result']['peak_private_gib']


def test_the_lifetime_peak_bound_is_labelled_everywhere_in_the_report_and_never_shown_as_a_stage_peak(reported):
    keys = set(all_keys(reported.doc))
    assert 'peak_private_upper_gib' not in keys and 'lifetime_peak_bound_gib' in keys
    assert (
        reported.doc['cost']['memory']['lifetime_peak_bound_label'] == 'lifetime peak bound (monotone within a call), not a per-stage peak'
    )
    assert reported.doc['cost']['memory']['judged_on'] == 'peak_private_gib' and reported.doc['cost']['memory']['within_cap'] is True


def test_the_flat_rows_of_the_mc_criterion_are_beside_the_agreement_share_in_the_gate_summary_too(reported, toy_run):
    results = read_json(toy_run.run / 'gate' / 'GATE.json')['results']['mc']
    mc = reported.doc['gate']['verdict']['mc']
    for key in ('share', 'n_sd_zero', 'n_sd_zero_disagree', 'n_disagree_with_spread'):
        assert key in mc, key
    assert (
        mc['n_sd_zero'] == results['n_sd_zero']
        and mc['n_sd_zero_disagree'] == results['n_sd_zero_disagree']
        and mc['share'] == results['share']
    )


def test_the_gate_summary_is_the_verdict_of_gate_json_and_the_report_never_judges_the_gate_again(reported, toy_run):
    gate = read_json(toy_run.run / 'gate' / 'GATE.json')
    doc = reported.doc['gate']
    assert doc['gate_pass'] == gate['gate_pass'] and doc['constants'] == gate['constants']
    for name in ('occurrence', 'value', 'accounting'):
        assert doc['verdict'][name] == gate['verdict'][name]
    assert doc['verdict']['all_pass'] == gate['verdict']['all_pass'] and doc['verdict']['missing'] == gate['verdict']['missing']
    assert doc['rule'] == 'gate_pass = all_pass and nothing missing (R27)'


def test_the_sensitivity_without_the_smoke_matches_says_why_when_the_decision_record_names_no_smoke_run(reported):
    sens = reported.doc['sensitivity_without_smoke_dev']
    assert sens['requested'] is True and sens['computed'] is False and 'decisions.S3_run.smoke_run' in sens['reason']  # a full run asks
    assert 'SENSITIVITY_NOT_COMPUTED' in [f['code'] for f in reported.doc['flags']]  # ... so the missing line is a finding


def fake_smoke_run(root: Path, toy_run, n: int = 12, **changes) -> Path:
    """A run folder that holds just the RUN.json of a smoke run made on the S1 store of the toy run: the smoke block is what
    ``run_s3.select_smoke(S1, n)`` gives (a seeded selection of ceil(0.3 n) DEV matches among others), as the runner writes it."""
    from c_ept.run_s1 import load_s1

    store = load_s1(toy_run.s1, verify=False)
    selection = run_s3.select_smoke(store, n)
    store.maps.clear()
    manifest = read_json(toy_run.run / 'RUN.json')
    doc = {'stage': 's3', 's1': dict(manifest['s1']), 'smoke': selection}
    doc.update(changes)
    folder = root / '20261022_c_ept_s3_9'
    folder.mkdir(parents=True)
    (folder / 'RUN.json').write_text(json.dumps(doc), encoding='utf-8')
    return folder


def smoke_dev(smoke_run: Path) -> list:
    """The DEV matches of the selection in the RUN.json of a fake smoke run."""
    return read_json(smoke_run / 'RUN.json')['smoke']['matches']['DEV']


def test_carry_3_leaving_out_the_smoke_dev_matches_drops_exactly_those_rows_and_gives_the_hand_checked_o_over_e(
    toy_run, tmp_path, monkeypatch
):
    monkeypatch.setattr(report_s3, 'BOOT_B', 100)
    b0 = load_arrays(toy_run.run / 'compose' / 'b0_dev.npz')
    matches = np.unique(b0['match'])
    smoke = fake_smoke_run(tmp_path, toy_run)
    listed = smoke_dev(smoke)  # the DEV matches of the seeded selection: all of them are DEV matches of this run
    n_in = int(np.isin(listed, matches).sum())
    assert len(listed) == 4 and n_in == 4
    doc = read_json(ask(toy_run, copy_of(toy_run, tmp_path), exclude_smoke=smoke, report_dir=tmp_path / 'rep'))
    sens = doc['sensitivity_without_smoke_dev']
    keep = ~np.isin(b0['match'], listed)
    assert sens['computed'] is True and sens['requested'] is True and sens['smoke_run'] == smoke.name
    assert (sens['n_listed_dev_matches'], sens['n_listed_in_this_run']) == (4, n_in)
    assert (sens['n_rows_all'], sens['n_matches_all'], sens['n_rows_kept'], sens['n_matches_kept']) == (
        len(keep),
        8,
        int(keep.sum()),
        8 - n_in,
    )
    assert (
        sens['n_rows_dropped'] == int((~keep).sum()) and sens['n_matches_dropped'] == n_in and sens['B'] == 100 and sens['seed'] == 20261021
    )
    # by hand: the observed 60 s onsets from the S1 onsets table, the weights 1 / n_m over the kept rows
    lines, pairs = direct_window_lines(toy_run.s1, b0)
    x, y = pairs[0][0][keep], pairs[0][1][keep]
    kept_match = b0['match'][keep]
    _, inverse, per_match = np.unique(kept_match, return_inverse=True, return_counts=True)
    w = 1.0 / per_match[inverse.reshape(-1)]
    assert sens['occurrence']['oe'] == pytest.approx(np.sum(w * y) / np.sum(w * x), rel=1e-12)
    assert sens['occurrence']['slope'] == pytest.approx(direct_line(x, y, w)[0], rel=1e-9)
    assert sens['occurrence']['n_rows'] == int(keep.sum()) and sens['occurrence']['n_matches'] == 8 - n_in
    from c_ept.store_core import Store

    labels = Store(toy_run.run / 'labels' / 'store')
    realized = np.asarray(labels.array('labels/v_start'))[b0['frame_row'], 0][keep]
    labels.maps.clear()
    xv = b0['expected'][keep, 0]
    assert sens['value']['bias'] == pytest.approx(np.sum(w * (realized - xv)) / np.sum(w), rel=1e-9, abs=1e-9)
    assert sens['value']['slope'] == pytest.approx(direct_line(xv, realized, w)[0], rel=1e-9)
    gate = read_json(toy_run.run / 'gate' / 'GATE.json')['results']  # the gate itself is not touched: its verdict is the gate's
    assert (
        sens['gate_values']['occurrence']['slope'] == gate['occurrence']['slope']
        and sens['gate_values']['value']['bias'] == gate['value']['bias']
    )
    assert sens['difference']['occurrence_slope'] == pytest.approx(sens['occurrence']['slope'] - gate['occurrence']['slope'])
    assert (
        'verdict' in sens['note']
        and 'GATE.json' in sens['note']
        and 'all_pass' not in json.dumps(sens)
        and 'gate_pass' not in json.dumps(sens)
    )


def test_carry_3_a_smoke_run_on_another_s1_store_or_without_a_smoke_selection_is_refused(toy_run, tmp_path):
    other = fake_smoke_run(tmp_path / 'a', toy_run)
    doc = read_json(other / 'RUN.json')
    doc['s1']['files_sha256'] = 'b' * 64
    (other / 'RUN.json').write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(report_s3.ReportInputError, match='S1'):
        ask(toy_run, copy_of(toy_run, tmp_path / 'c1'), exclude_smoke=other)
    not_smoke = fake_smoke_run(tmp_path / 'b', toy_run, smoke=None)
    with pytest.raises(report_s3.ReportInputError, match='smoke'):
        ask(toy_run, copy_of(toy_run, tmp_path / 'c2'), exclude_smoke=not_smoke)
    with pytest.raises(report_s3.ReportInputError, match='smoke'):
        ask(toy_run, copy_of(toy_run, tmp_path / 'c3'), exclude_smoke=tmp_path / 'there_is_no_such_run')


# -- the figures, the keys, the refusals
def test_the_report_draws_the_five_figures_as_png_files_and_lists_them_with_their_numbers(reported):
    assert len(FIGURES) >= 4
    for name in FIGURES:
        path = reported.dir / f'{name}.png'
        assert path.is_file() and path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and path.stat().st_size > 20_000, name
        entry = reported.doc['figures'][name]
        assert entry['file'] == f'{name}.png' and entry['numbers'] and entry['title']
    assert sorted(p.name for p in reported.dir.iterdir()) == sorted([*(f'{n}.png' for n in FIGURES), 'S3_REPORT.json'])


def test_the_figure_numbers_are_the_numbers_of_the_sections_they_show(reported):
    doc, fig = reported.doc, reported.doc['figures']
    assert fig['fig4_window_slopes']['numbers']['b0']['slope'] == [w['slope'] for w in doc['window_slopes']['b0']['windows']]
    assert fig['fig4_window_slopes']['numbers']['band'] == [0.9, 1.1]
    assert fig['fig2_occurrence_deciles']['numbers']['mean_observed'] == [d['mean_observed'] for d in doc['occurrence_deciles']['deciles']]
    assert fig['fig3_value_deciles_pit']['numbers']['pit_b0'] == doc['value_deciles']['pit']['b0']['mass']
    assert fig['fig5_brier']['numbers']['brier'] == {k: doc['brier'][k]['brier'] for k in ('m2_b0', 'm2_b1', 'b2')}
    assert fig['fig1_d7_beta']['numbers']['groups'] == list(report_s3.GROUPS)


def test_the_figures_have_korean_labels_in_malgun_gothic_and_a_missing_glyph_stops_the_figure():
    plt = report_s3._plt()
    assert plt.rcParams['font.family'] == ['Malgun Gothic'] and plt.rcParams['axes.unicode_minus'] is False
    fig, ax = plt.subplots()
    ax.set_title('한글 \U0001d11e')  # Hangul is drawn; the musical symbol is a glyph that Malgun Gothic lacks
    with pytest.raises(RuntimeError, match='missing from font'):
        report_s3._render(plt, fig)
    fig, ax = plt.subplots()
    ax.set_title(r'한글 $t_0$, $\beta_{-1}$')  # Hangul and mathtext subscripts are fine
    assert report_s3._render(plt, fig)[:8] == b'\x89PNG\r\n\x1a\n'


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def test_the_d7_figure_draws_a_group_whose_flow_fit_is_not_estimable():
    flat = {'used': 'immediate', 'adopted': False, 'price_flow': None, 'beta': None, 'beta_ci': None, 'pretrend': None, 'pretrend_ci': None}
    d7 = report_s3.d7_section(prices_document(herald=flat))
    title, numbers, png = report_s3.fig_d7(report_s3._plt(), {'d7': d7})
    assert (
        png[:8] == PNG_SIGNATURE
        and title.startswith('그림 1')
        and numbers['beta']['herald'] is None
        and numbers['used']['herald'] == 'immediate'
    )


def test_the_deciles_figures_draw_bins_that_are_empty_and_a_brier_interval_that_could_not_be_computed():
    from c_ept import features, gate_s3

    rng = np.random.default_rng(8)
    n, match = 120, np.repeat(np.arange(12), 10)
    expected = rng.choice([1.0, 2.0, 3.0], n)  # three values only: most of the ten bins are empty
    observed = rng.poisson(expected)
    w = features.match_weights(match)
    occ = gate_s3.occurrence_calibration(expected, observed, match, w, B=40)
    val = gate_s3.value_calibration(100 * expected, 100 * observed, match, w, B=40)
    assert any(d['mean_expected'] is None for d in occ['deciles'])  # the case the figure must survive
    q = np.sort(rng.normal(size=(n, 19)), axis=1)
    pit = report_s3.pit_histogram(q, rng.normal(size=n), w)
    value = {
        **{k: val[k] for k in ('deciles', 'slope', 'slope_ci', 'bias', 'bias_ci')},
        'bands': {'slope': [0.8, 1.2], 'bias_gold': [-50.0, 50.0]},
        'pit': {'b0': pit, 'b1': pit},
        'crps19': {'b0': 1.0, 'b1': 2.0},
    }
    plt = report_s3._plt()
    assert (
        report_s3.fig_occurrence(plt, {'occurrence_deciles': {**occ, 'bands': {'oe': [0.95, 1.05], 'slope': [0.9, 1.1]}}})[2][:8]
        == PNG_SIGNATURE
    )
    assert report_s3.fig_value(plt, {'value_deciles': value})[2][:8] == PNG_SIGNATURE
    cell = {'brier': 0.2, 'ci': [0.19, 0.21]}
    diff = {k: {'value': 0.01, 'ci': [None, None]} for k in ('m2_b0_minus_b2', 'm2_b1_minus_b2', 'm2_b0_minus_m2_b1')}
    brier = {
        'm2_b0': cell,
        'm2_b1': {**cell, 'ci': [None, None]},
        'b2': cell,
        'difference': diff,
        'n_rows': 120,
        'n_matches': 12,
        'positive_share': 0.4,
    }
    assert report_s3.fig_brier(plt, {'brier': brier})[2][:8] == PNG_SIGNATURE


def test_the_window_figure_draws_a_window_that_could_not_be_computed_and_says_how_much_of_the_dev_set_b1_covers():
    expected_cum, observed, match, _ = window_world([10] * 12)
    expected_cum[:, 1] = expected_cum[:, 0] + 0.25  # the window 60-120 s is flat
    expected_cum[:, 2] = expected_cum[:, 1] + np.random.default_rng(2).uniform(0.1, 1.0, len(match))
    lines = report_s3.window_lines(expected_cum, observed, match, B=40)
    assert [w['computed'] for w in lines['windows']] == [True, False, True, True]
    sample = {'b0': {'rows': 120, 'matches': 12, **lines}, 'b1': {'rows': 40, 'matches': 4, **lines}}
    title, numbers, png = report_s3.fig_windows(report_s3._plt(), {'window_slopes': sample})
    assert (
        png[:8] == PNG_SIGNATURE
        and numbers['b0']['slope'][1] is None
        and numbers['windows'] == ['0-60 s', '60-120 s', '120-180 s', '0-180 s']
    )


def test_the_window_figure_shades_the_band_from_its_lower_to_its_upper_limit(monkeypatch):
    expected_cum, observed, match, _ = window_world([10] * 12)
    lines = report_s3.window_lines(expected_cum, observed, match, B=40)
    sample = {'b0': {'rows': 120, 'matches': 12, **lines}, 'b1': {'rows': 120, 'matches': 12, **lines}}
    drawn = []
    monkeypatch.setattr(report_s3, '_render', lambda plt, fig: drawn.append(fig) or PNG_SIGNATURE)
    report_s3.fig_windows(report_s3._plt(), {'window_slopes': sample})
    (band,) = drawn[0].axes[0].patches  # the one shaded band
    low, high = lines['bands']['slope']
    assert low < high and (band.get_y(), band.get_y() + band.get_height()) == pytest.approx((low, high))
    report_s3._plt().close(drawn[0])


def test_no_match_key_and_no_game_id_is_in_the_report_text(reported, toy_run):
    keys = toy_keys(toy_run)
    assert len(keys) == 48 and not any(k in reported.text for k in keys)
    assert reported.doc['key_scan']['checked'] is True and reported.doc['key_scan']['n_keys'] == 48
    assert reported.doc['key_scan']['files'] == ['S3_REPORT.json'] and reported.doc['key_scan']['min_game_id_digits'] == 9


def test_a_match_key_in_a_field_that_the_report_copies_is_refused_and_nothing_is_written(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    key = toy_keys(toy_run)[3]
    manifest = read_json(run / 'RUN.json')
    manifest['cpu'] = f'CPU of {key}'  # RUN.json is not hash-protected, and the report copies the machine
    (run / 'RUN.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(report_s3.KeyInReport, match='^1 match keys occur in the text that was to be written$') as err:
        ask(toy_run, run)
    assert isinstance(err.value, RuntimeError) and key not in str(err.value) and not (run / 'report').exists()


@pytest.mark.parametrize(
    'damage, match',
    [
        (lambda run: set_json(run / 'STAGES.json', lambda d: d['stages']['compose'].update(complete=False)), 'compose'),
        (lambda run: set_json(run / 'STAGES.json', lambda d: d['stages'].pop('gate')), 'gate'),
        (lambda run: set_json(run / 'RUN.json', lambda d: d.pop('s3_result')), 's3_result'),
        (lambda run: set_json(run / 'RUN.json', lambda d: d.pop('gate_result')), 'gate_result'),
        (lambda run: set_json(run / 'RUN.json', lambda d: d.update(stage='s2')), 'S3 run'),
        (
            lambda run: (run / 'gate' / 'GATE.json').write_text(
                (run / 'gate' / 'GATE.json').read_text(encoding='utf-8') + ' ', encoding='utf-8'
            ),
            'gate',
        ),
        (lambda run: (run / 'compose' / 'b1_dev.npz').write_bytes(b'x' + (run / 'compose' / 'b1_dev.npz').read_bytes()), 'compose'),
        (lambda run: (run / 'STAGES.json').unlink(), 'STAGES.json'),
        (lambda run: (run / 'RUN.json').unlink(), 'RUN.json'),
    ],
)
def test_a_run_that_is_not_finished_or_was_changed_after_it_finished_is_refused_and_nothing_is_written(toy_run, tmp_path, damage, match):
    run = copy_of(toy_run, tmp_path)
    damage(run)
    with pytest.raises(report_s3.ReportInputError, match=match):
        ask(toy_run, run)
    assert not (run / 'report').exists()


def set_json(path: Path, change) -> None:
    """Change the JSON file ``path`` in place: ``change(doc)`` edits the dict."""
    doc = read_json(path)
    change(doc)
    path.write_text(json.dumps(doc), encoding='utf-8')


def rewrite_json(path: Path, make) -> None:
    """Replace the JSON file ``path`` by ``make(doc)``."""
    path.write_text(json.dumps(make(read_json(path))), encoding='utf-8')


def rehash(run: Path) -> None:
    """Tell STAGES.json of the run copy the SHA-256 of the stage outputs as they are now (after a test changed one on purpose)."""
    set_json(run / 'STAGES.json', lambda d: [d['stages'][s].update(outputs=run_s3.stage_outputs(run, s)) for s in d['stages']])


def test_a_record_that_lacks_a_field_the_report_reads_is_refused_with_the_name_of_the_field(toy_run, tmp_path):
    run = copy_of(toy_run, tmp_path)
    set_json(run / 'gate' / 'GATE.json', lambda d: d.pop('brier'))
    rehash(run)  # an output that changed is refused earlier: the ledger is told
    with pytest.raises(report_s3.ReportInputError, match='brier'):
        ask(toy_run, run)
    assert not (run / 'report').exists()


def test_an_s3_result_that_does_not_say_the_run_is_complete_is_refused(toy_run, tmp_path):
    run = copy_of(toy_run, tmp_path)
    set_json(run / 'RUN.json', lambda d: d['s3_result'].update(complete=False))
    with pytest.raises(report_s3.ReportInputError, match='s3_result does not say'):
        ask(toy_run, run)
    assert not (run / 'report').exists()


def test_the_s1_and_s3a_inputs_are_found_checked_and_refused_when_they_are_not_the_ones_of_the_run(toy_run, tmp_path):
    run = copy_of(toy_run, tmp_path / 'a')
    with pytest.raises(report_s3.ReportInputError, match='S1.*--s1|--s1.*S1'):
        report_s3.report(run, s3a=toy_run.s3a)  # the S1 run is not next to the run folder
    with pytest.raises(report_s3.ReportInputError, match='s3a.*--s3a|--s3a.*s3a'):
        report_s3.report(run, s1=toy_run.s1)
    set_json(run / 'RUN.json', lambda d: d['s1'].update(files_sha256='c' * 64))
    with pytest.raises(report_s3.ReportInputError, match='S1 store'):
        ask(toy_run, run)
    run = copy_of(toy_run, tmp_path / 'b')
    s3a = tmp_path / 's3a'
    shutil.copytree(toy_run.s3a, s3a)
    prices = s3a / 'prices' / 'PRICES.json'
    prices.write_text(prices.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(report_s3.ReportInputError, match='PRICES.json'):
        report_s3.report(run, s1=toy_run.s1, s3a=s3a)
    rules = s3a / 'rules' / 'RULE_PAYOUTS.json'
    shutil.copy(toy_run.s3a / 'prices' / 'PRICES.json', prices)
    rules.write_text(rules.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(report_s3.ReportInputError, match='RULE_PAYOUTS.json'):
        report_s3.report(run, s1=toy_run.s1, s3a=s3a)
    assert not (run / 'report').exists()


def test_a_folder_that_is_given_with_s1_or_s3a_but_is_not_there_is_refused_by_name(toy_run, tmp_path):
    run = copy_of(toy_run, tmp_path)
    with pytest.raises(report_s3.ReportInputError, match='--s1'):
        report_s3.report(run, s1=tmp_path / 'no_such_s1_run', s3a=toy_run.s3a)
    with pytest.raises(report_s3.ReportInputError, match='--s3a'):
        report_s3.report(run, s1=toy_run.s1, s3a=tmp_path / 'no_such_s3a_run')


def test_the_smoke_selection_may_be_given_as_the_run_json_file_of_the_smoke_run(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    smoke = fake_smoke_run(tmp_path / 'sm', toy_run, n=5)  # ceil(0.3 * 5) = 2 DEV matches
    doc = read_json(ask(toy_run, copy_of(toy_run, tmp_path / 'c'), exclude_smoke=smoke / 'RUN.json', report_dir=tmp_path / 'rep'))
    sens = doc['sensitivity_without_smoke_dev']
    assert sens['smoke_run'] == smoke.name and sens['n_listed_in_this_run'] == 2 and sens['n_matches_kept'] == 6
    recorded = read_json(smoke / 'RUN.json')['smoke']
    assert doc['inputs']['exclude_smoke'] == {
        'run': smoke.name,
        'selection_sha256': recorded['selection_sha256'],
        'n': 5,
        'seed': recorded['seed'],
        'source': report_s3.SMOKE_SOURCE_ARGUMENT,
        'checked': True,
    }


def test_the_s1_and_s3a_runs_are_found_next_to_the_run_folder_when_they_are_not_given(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    root = tmp_path / 'runs'
    root.mkdir()
    run = copy_of(toy_run, root)
    shutil.copytree(toy_run.world.s1, root / toy_run.world.s1.name)
    shutil.copytree(toy_run.s3a, root / toy_run.s3a.name)
    assert report_s3.report(run).parent == run / 'report'
    assert read_json(run / 'report' / 'S3_REPORT.json')['inputs']['s1']['run'] == toy_run.world.s1.name


def test_a_report_leaves_every_file_of_the_run_as_it_was(toy_run, tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)

    def tree():
        return {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()}

    before = tree()
    ask(toy_run, run)
    after = tree()
    assert {k: v for k, v in after.items() if not k.startswith('report/')} == before
    assert sorted(k for k in after if k.startswith('report/')) == sorted(
        f'report/{n}' for n in [*(f'{n}.png' for n in FIGURES), 'S3_REPORT.json']
    )


def test_the_same_run_gives_the_same_report_twice_apart_from_its_time_stamp(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    first = read_json(ask(toy_run, run, report_dir=tmp_path / 'one'))
    second = read_json(ask(toy_run, run, report_dir=tmp_path / 'two'))
    for doc in (first, second):
        doc.pop('created')
        assert doc.pop('report_dir') in ('one', 'two')  # the name of the folder it was written to
    assert first == second and (tmp_path / 'one' / 'S3_REPORT.json').stat().st_size > 10_000


def test_a_report_of_a_smoke_run_says_so_and_a_run_from_before_the_peak_bound_still_reports_with_none(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    selection = fake_smoke_run(tmp_path / 'sm', toy_run)
    set_json(run / 'RUN.json', lambda d: d.update(smoke=read_json(selection / 'RUN.json')['smoke']))

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k != 'peak_private_upper_gib' and not k.startswith('chunks_') and k != 'rows_computed'
            }
        return [strip(v) for v in obj] if isinstance(obj, list) else obj

    for rel in ('compose/COMPOSE.json', 'fit/FIT_FACTS.json'):  # stage outputs: the ledger is told their new hashes
        rewrite_json(run / rel, strip)
    rewrite_json(run / 'STAGES.json', lambda d: {**strip(d), 'stages': {s: strip(v) for s, v in d['stages'].items()}})
    rehash(run)
    rewrite_json(run / 'RUN.json', lambda d: {k: strip(v) if k.endswith('_result') else v for k, v in d.items()})
    doc = read_json(ask(toy_run, run))
    codes = [f['code'] for f in doc['flags']]
    assert 'SMOKE_RUN' in codes and 'OLD_RECORD_LAYOUT' in codes and doc['run']['smoke'] is True
    assert doc['run']['smoke_selection']['n'] == 12 and doc['run']['smoke_selection']['n_dev_matches'] == 4
    sens = doc['sensitivity_without_smoke_dev']  # a smoke run has no other smoke run to leave out: nothing was asked, nothing is flagged
    assert sens['requested'] is False and sens['computed'] is False and 'smoke run' in sens['reason']
    assert 'SENSITIVITY_NOT_COMPUTED' not in codes
    assert (
        'peak_private_upper_gib' in doc['inputs']['missing_optional_fields'] and 'chunks_total' in doc['inputs']['missing_optional_fields']
    )
    assert doc['cost']['stages']['fit']['lifetime_peak_bound_gib'] is None and doc['cost']['compose']['chunks']['chunks_total'] is None
    assert doc['cost']['stages']['fit']['peak_private_gib'] is not None and doc['cost']['memory']['within_cap'] is True


def test_the_command_line_reports_a_run_and_refuses_what_the_function_refuses(toy_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    args = ['--run', str(run), '--s1', str(toy_run.s1), '--s3a', str(toy_run.s3a), '--report-dir', str(tmp_path / 'rep')]
    assert report_s3.main(args) == 0
    out = capsys.readouterr().out
    assert 'S3_REPORT.json' in out and (tmp_path / 'rep' / 'S3_REPORT.json').is_file() and 'seconds' in out
    set_json(run / 'STAGES.json', lambda d: d['stages']['fit'].update(complete=False))
    assert report_s3.main(args) == 2
    err = capsys.readouterr().err
    assert 'report refused' in err and 'fit' in err
    assert (
        report_s3.main(['--run', str(run)]) == 2 and 'report refused' in capsys.readouterr().err
    )  # no S1 run next to it: refused, not a traceback


def test_the_report_module_imports_nothing_that_reads_a_raw_match_or_the_cohort_split_and_no_runner_at_module_level():
    from c_ept.tests.conftest import all_imports, module_level_imports

    source = Path(report_s3.__file__)
    assert 'c_ept.run_s1' not in module_level_imports(source)
    assert not ({'c_ept.reader', 'c_ept.cohort'} & all_imports(source))
    assert 'split_module' not in source.read_text(encoding='utf-8') and 'read_match' not in source.read_text(encoding='utf-8')


def test_the_gate_constants_of_the_report_are_those_of_the_gate_and_the_cli_name_the_options():
    assert (report_s3.BOOT_B, report_s3.SEED) == (2000, 20261021) == (run_s3.BOOT_B, run_s3.SEED)
    from c_ept import nll

    assert (nll.BOOT_B, nll.BOOT_SEED) == (2000, 20261021)


# ---- group I: the post-run fix pass (task-13-fix-brief, items 1-11) ----------------------------------------------------------------------
# Test-first record. The tests of items 1-6 and 8-10 at the level of the findings, the cost record, the smoke-run checks, the figure, the
# winner section, the toy-run report and the command line were written before the change they pin and were run red against the code of
# 18bdea3 (AttributeError, KeyError or a failed assert). The pin of the current layout (item 7) is green there by design (it pins what
# is already true). The unit tests of ``resolve_smoke``, ``scrubbed_traceback`` and ``winner_labels``, the two-number case of
# ``compare_with_gate`` and the test of the compose module text were written once the code stood (the compose text was shown red on the
# text of 18bdea3 by a script); they are evidenced by one-line mutants (the R35 standard).
RULE_R48 = (
    'R48 reading of P7(iii): counts exact (max_abs_error == 0) and gold within float rounding (<= 1e-6 gold); GATE.json keeps its verdict'
)


# -- item 1: criterion (iii) at the gold level (ruling R48)
@pytest.mark.parametrize(
    'err, passed',
    [(0.0, True), (1.4901161193847656e-08, True), (1e-06, True), (1.0000001e-06, False), (0.5, False), (float('nan'), False)],
)
def test_the_gold_check_of_the_accounting_passes_up_to_the_float_rounding_tolerance_of_ruling_r48(err, passed):
    out = report_s3.accounting_value_check(err)
    assert set(out) == {'max_abs_value_error', 'tolerance_gold', 'pass', 'rule'}  # exactly the four fields of the brief
    assert out['tolerance_gold'] == 1e-6 and out['rule'] == RULE_R48 and out['pass'] is passed
    assert out['max_abs_value_error'] == err or err != err  # copied as it is (a NaN is not equal to itself)


@pytest.mark.parametrize('bad', [None, 'x', True, -1e-9, [1.0]])
def test_the_gold_check_refuses_an_error_that_is_not_a_real_number_or_is_negative(bad):
    with pytest.raises(report_s3.ReportInputError, match='max_abs_value_error'):
        report_s3.accounting_value_check(bad)


def test_the_gold_error_over_the_tolerance_is_a_finding_that_names_the_error_and_the_tolerance():
    doc = quiet_doc()
    doc['accounting']['value_check'].update({'pass': False, 'max_abs_value_error': 0.5})
    (flag,) = report_s3.findings(doc)
    assert flag['code'] == 'ACCOUNTING_GOLD_OVER_TOLERANCE'
    assert '0.5' in flag['message'] and '1e-06' in flag['message'] and 'R48' in flag['message']


def test_the_accounting_section_holds_the_gold_check_beside_the_verdict_of_the_gate(reported, toy_run):
    raw = read_json(toy_run.run / 'labels' / 'ACCOUNTING.json')
    doc = reported.doc['accounting']
    assert doc['value_check'] == {'max_abs_value_error': raw['max_abs_value_error'], 'tolerance_gold': 1e-6, 'pass': True, 'rule': RULE_R48}
    assert doc['gate_pass'] is True and doc['must_equal'] == 0.0  # the verdict of GATE.json is where it was
    assert 'ACCOUNTING_GOLD_OVER_TOLERANCE' not in [f['code'] for f in reported.doc['flags']]


def test_a_gold_error_over_the_tolerance_raises_the_flag_and_does_not_touch_the_verdict_of_the_gate(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    set_json(run / 'labels' / 'ACCOUNTING.json', lambda d: d.update(max_abs_value_error=0.25))
    rehash(run)  # an output that changed is refused earlier: the ledger is told
    doc = read_json(ask(toy_run, run, report_dir=tmp_path / 'rep'))
    check = doc['accounting']['value_check']
    assert check['pass'] is False and check['max_abs_value_error'] == 0.25 and check['tolerance_gold'] == 1e-6
    gate = read_json(run / 'gate' / 'GATE.json')
    assert doc['accounting']['gate_pass'] is gate['verdict']['accounting']['pass'] and doc['gate']['gate_pass'] is gate['gate_pass']
    assert doc['gate']['verdict']['accounting'] == gate['verdict']['accounting']  # the report never judges the gate again
    flags = {f['code']: f['message'] for f in doc['flags']}
    assert '0.25' in flags['ACCOUNTING_GOLD_OVER_TOLERANCE']


# -- item 2: the sensitivity line cannot go missing silently
class FakeRoles:
    """The one thing that ``run_s3.select_smoke`` asks of an S1 store: ``array('matches/role')``, a role code per match (40 matches: 20 FIT,
    8 STOP, 10 DEV and 2 UNUSED in a fixed shuffled order)."""

    def __init__(self):
        names = ['FIT'] * 20 + ['STOP'] * 8 + ['DEV'] * 10 + ['UNUSED'] * 2
        order = np.random.default_rng(3).permutation(len(names))
        self.codes = np.array([C.ROLES.index(names[i]) for i in order], dtype=np.int64)

    def array(self, name):
        assert name == 'matches/role'
        return self.codes


def smoke_block(n=6, seed=None):
    """``(store, smoke)``: a fake S1 store and the smoke dict ``read_smoke_selection`` would give for the selection of ``n`` on it."""
    store = FakeRoles()
    selection = run_s3.select_smoke(store, n) if seed is None else run_s3.select_smoke(store, n, seed)
    return store, {'run': '20261022_c_ept_s3_1', **selection}


def write_record(path: Path, smoke_run, key='S3_run') -> Path:
    """A decision record with ``decisions.<key>.smoke_run``."""
    path.write_text(json.dumps({'decisions': {key: {'smoke_run': smoke_run}}}), encoding='utf-8')
    return path


def test_a_selection_that_is_the_one_select_smoke_gives_on_this_store_is_accepted_whatever_the_seed_it_records():
    store, smoke = smoke_block(6)
    assert report_s3.check_smoke_selection(smoke, store) is None
    store, smoke = smoke_block(7, seed=11)
    assert smoke['seed'] == 11 and report_s3.check_smoke_selection(smoke, store) is None


SELECTION_DAMAGE = {
    'a DEV match is another one': lambda s: s['matches']['DEV'].__setitem__(0, s['matches']['DEV'][0] + 1),
    'the hash is another one': lambda s: s.update(selection_sha256='b' * 64),
    'n is another one': lambda s: s.update(n=s['n'] + 1),
    'the seed is another one': lambda s: s.update(seed=s['seed'] + 1),
    'no seed is recorded': lambda s: s.pop('seed'),
    'no n is recorded': lambda s: s.pop('n'),
    'n is a bool': lambda s: s.update(n=True),
    'a role is missing': lambda s: s['matches'].pop('STOP'),
}


@pytest.mark.parametrize('damage', list(SELECTION_DAMAGE.values()), ids=list(SELECTION_DAMAGE))
def test_a_selection_that_is_not_the_one_select_smoke_gives_for_its_n_and_seed_is_refused(damage):
    store, smoke = smoke_block(6)
    damage(smoke)
    with pytest.raises(report_s3.ReportInputError, match='selection'):
        report_s3.check_smoke_selection(smoke, store)


def test_the_smoke_run_that_the_decision_record_names_is_read_from_the_record_that_run_json_names(tmp_path):
    record = write_record(tmp_path / 'SPEC.json', '20261022_c_ept_s3_1')
    assert report_s3.decision_smoke_run({'args': {'spec': str(record)}}) == ('20261022_c_ept_s3_1', '')
    other = write_record(tmp_path / 'OTHER.json', '20261022_c_ept_s3_7', key='S3_prices')  # a name under another decision is not the one
    name, why = report_s3.decision_smoke_run({'args': {'spec': str(other)}})
    assert name is None and 'decisions.S3_run.smoke_run' in why


@pytest.mark.parametrize('manifest', [{}, {'args': None}, {'args': {}}, {'args': {'spec': ''}}, {'args': {'spec': 7}}])
def test_a_run_json_without_a_decision_record_gives_no_smoke_run_and_says_so(manifest):
    name, why = report_s3.decision_smoke_run(manifest)
    assert name is None and 'args.spec' in why


BAD_RECORDS = [
    None,  # the file is not there
    'not json',
    '[]',
    '{}',
    '{"decisions": []}',
    '{"decisions": {"S3_run": {}}}',
    '{"decisions": {"S3_run": {"smoke_run": null}}}',
    '{"decisions": {"S3_run": {"smoke_run": ""}}}',
    '{"decisions": {"S3_run": {"smoke_run": 20261022}}}',
    '{"decisions": {"S3_run": {"smoke_run": "..\\\\x"}}}',
    '{"decisions": {"S3_run": {"smoke_run": "a/b"}}}',
    '{"decisions": {"S3_run": {"smoke_run": ".."}}}',
]


@pytest.mark.parametrize('text', BAD_RECORDS)
def test_a_record_that_names_no_usable_smoke_run_gives_none_and_a_reason_that_names_the_file_and_not_its_folder(tmp_path, text):
    path = tmp_path / 'SPEC.json'
    if text is not None:
        path.write_text(text, encoding='utf-8')
    name, why = report_s3.decision_smoke_run({'args': {'spec': str(path)}})
    assert name is None and 'SPEC.json' in why and str(tmp_path) not in why


def with_decision_record(run: Path, record) -> None:
    """Make RUN.json of the run copy name ``record`` as its decision record."""
    set_json(run / 'RUN.json', lambda d: d['args'].update(spec=str(record)))


# resolve_smoke on its own: a fake S1 store, small folders, no toy run
def smoke_folder(root: Path, name: str, store, n=6, **changes) -> Path:
    """A folder with the RUN.json of a smoke run of ``n`` made on ``store`` (the S1 store of the manifests below: files hash 'h')."""
    doc = {'stage': 's3', 's1': {'files_sha256': 'h'}, 'smoke': run_s3.select_smoke(store, n), **changes}
    (root / name).mkdir(parents=True)
    (root / name / 'RUN.json').write_text(json.dumps(doc), encoding='utf-8')
    return root / name


def resolving(tmp_path: Path, smoke_run, *, own_smoke=None):
    """``(run, manifest)``: a run folder in ``tmp_path / 'runs'`` whose manifest names a decision record that names ``smoke_run``."""
    record = write_record(tmp_path / 'SPEC.json', smoke_run)
    (tmp_path / 'runs').mkdir(exist_ok=True)
    run = tmp_path / 'runs' / '20261022_c_ept_s3_3'
    return run, {'s1': {'files_sha256': 'h'}, 'smoke': own_smoke, 'args': {'spec': str(record)}}


def test_a_smoke_run_that_is_given_is_the_one_whose_dev_matches_are_left_out(tmp_path):
    store = FakeRoles()
    folder = smoke_folder(tmp_path, '20261022_c_ept_s3_1', store)
    run, manifest = resolving(tmp_path, 'not_used_when_given')
    got = report_s3.resolve_smoke(run, manifest, store, folder, None)
    assert got['requested'] is True and got['reason'] is None
    assert got['selection']['source'] == report_s3.SMOKE_SOURCE_ARGUMENT and got['selection']['run'] == '20261022_c_ept_s3_1'
    assert got['selection']['dev_matches'] == run_s3.select_smoke(store, 6)['matches']['DEV'] and got['selection']['seed'] == run_s3.SEED


def test_a_run_that_is_not_a_smoke_run_takes_the_smoke_run_the_decision_record_names_from_next_to_it_or_from_the_runs_root(tmp_path):
    store = FakeRoles()
    smoke_folder(tmp_path / 'runs', '20261022_c_ept_s3_1', store)  # next to the run folder
    run, manifest = resolving(tmp_path, '20261022_c_ept_s3_1')
    got = report_s3.resolve_smoke(run, manifest, store, None, tmp_path / 'nowhere')
    assert got['selection']['source'] == report_s3.SMOKE_SOURCE_DECISION and got['selection']['run'] == '20261022_c_ept_s3_1'
    other = tmp_path / 'other'
    smoke_folder(other, '20261022_c_ept_s3_7', store)  # in the runs root
    run, manifest = resolving(tmp_path, '20261022_c_ept_s3_7')
    got = report_s3.resolve_smoke(run, manifest, store, None, other)
    assert got['requested'] is True and got['selection']['run'] == '20261022_c_ept_s3_7'


def test_a_smoke_run_asks_for_nothing_and_a_name_that_is_not_there_is_a_missing_line_and_not_a_refusal(tmp_path):
    store = FakeRoles()
    for own_smoke in ({'n': 1}, {}):  # a run is a smoke run when its RUN.json has a smoke block, as the run section tells it
        run, manifest = resolving(tmp_path, '20261022_c_ept_s3_1', own_smoke=own_smoke)
        got = report_s3.resolve_smoke(run, manifest, store, None, None)
        assert got['requested'] is False and got['selection'] is None and 'smoke run' in got['reason']
    run, manifest = resolving(tmp_path, '20261022_c_ept_s3_1')  # nothing is there
    got = report_s3.resolve_smoke(run, manifest, store, None, tmp_path / 'nowhere')
    assert got['requested'] is True and got['selection'] is None and 'not next to this run folder' in got['reason']
    assert '20261022_c_ept_s3_1' in got['reason'] and str(tmp_path) not in got['reason']
    run, manifest = resolving(tmp_path, '..\\x')  # a name that is no folder name
    got = report_s3.resolve_smoke(run, manifest, store, None, None)
    assert got['requested'] is True and got['selection'] is None and 'decisions.S3_run.smoke_run' in got['reason']


def test_a_smoke_run_that_is_found_but_does_not_belong_to_this_store_or_this_selection_is_refused(tmp_path):
    store = FakeRoles()
    run, manifest = resolving(tmp_path, 'x')
    for name, changes, match in (
        ('other_store', {'s1': {'files_sha256': 'z'}}, 'another S1 store'),
        ('no_smoke_block', {'smoke': None}, 'not a smoke run'),
    ):
        folder = smoke_folder(tmp_path / 'given', name, store, **changes)
        with pytest.raises(report_s3.ReportInputError, match=match):
            report_s3.resolve_smoke(run, manifest, store, folder, None)
    smoke_folder(tmp_path / 'runs', 'bad_selection', store)
    set_json(tmp_path / 'runs' / 'bad_selection' / 'RUN.json', lambda d: d['smoke'].update(n=d['smoke']['n'] + 1))
    run, manifest = resolving(tmp_path, 'bad_selection')
    with pytest.raises(report_s3.ReportInputError, match='selection'):  # also through the decision record
        report_s3.resolve_smoke(run, manifest, store, None, None)
    with pytest.raises(report_s3.ReportInputError, match='no RUN.json'):  # a folder that is given and is not there is the caller's mistake
        report_s3.resolve_smoke(run, manifest, store, tmp_path / 'there_is_no_such_run', None)


def test_the_smoke_run_of_the_decision_record_is_the_default_for_a_run_that_is_not_a_smoke_run(toy_run, tmp_path, monkeypatch):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    smoke = fake_smoke_run(tmp_path, toy_run)  # next to the copy of the run, where the report looks for it
    run = copy_of(toy_run, tmp_path)
    with_decision_record(run, write_record(tmp_path / 'SPEC.json', smoke.name))
    default = read_json(ask(toy_run, run, report_dir=tmp_path / 'one'))  # no exclude_smoke
    given = read_json(ask(toy_run, run, exclude_smoke=smoke, report_dir=tmp_path / 'two'))
    sens, explicit = default['sensitivity_without_smoke_dev'], given['sensitivity_without_smoke_dev']
    assert sens['requested'] is True and sens['computed'] is True and sens['smoke_run'] == smoke.name
    assert sens['source'] == report_s3.SMOKE_SOURCE_DECISION and explicit['source'] == report_s3.SMOKE_SOURCE_ARGUMENT
    assert {k: v for k, v in sens.items() if k != 'source'} == {k: v for k, v in explicit.items() if k != 'source'}  # the same line
    assert default['inputs']['exclude_smoke']['source'] == report_s3.SMOKE_SOURCE_DECISION
    assert 'SENSITIVITY_NOT_COMPUTED' not in [f['code'] for f in default['flags']]


@pytest.mark.parametrize('how', ['the argument', 'the decision record'])
@pytest.mark.parametrize('damage', list(SELECTION_DAMAGE.values())[:4], ids=list(SELECTION_DAMAGE)[:4])
def test_a_smoke_run_whose_selection_is_not_the_one_of_its_n_is_refused_and_nothing_is_written(toy_run, tmp_path, damage, how):
    smoke = fake_smoke_run(tmp_path, toy_run)
    set_json(smoke / 'RUN.json', lambda d: damage(d['smoke']))
    run = copy_of(toy_run, tmp_path)
    kw = {}
    if how == 'the argument':
        kw['exclude_smoke'] = smoke
    else:
        with_decision_record(run, write_record(tmp_path / 'SPEC.json', smoke.name))
    with pytest.raises(report_s3.ReportInputError, match='selection'):
        ask(toy_run, run, **kw)
    assert not (run / 'report').exists()


def test_two_smoke_runs_of_the_same_n_on_the_same_store_have_the_same_selection(toy_run, tmp_path):
    first, second = fake_smoke_run(tmp_path / 'a', toy_run), fake_smoke_run(tmp_path / 'b', toy_run)
    assert read_json(first / 'RUN.json')['smoke']['selection_sha256'] == read_json(second / 'RUN.json')['smoke']['selection_sha256']


@pytest.mark.parametrize(
    'how, reason',
    [
        ('no spec', 'args.spec'),
        ('a record without S3_run', 'decisions.S3_run.smoke_run'),
        ('a record that is not there', 'cannot be read'),
        ('a smoke run that is not there', 'not next to this run folder'),
    ],
)
def test_a_full_run_whose_line_cannot_be_computed_gets_a_flag_with_the_reason(toy_run, tmp_path, monkeypatch, how, reason):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    if how == 'no spec':
        set_json(run / 'RUN.json', lambda d: d['args'].pop('spec'))
    elif how == 'a record without S3_run':
        with_decision_record(run, write_record(tmp_path / 'SPEC.json', 'x', key='S3_prices'))
    elif how == 'a record that is not there':
        with_decision_record(run, tmp_path / 'nothing_here.json')
    else:
        with_decision_record(run, write_record(tmp_path / 'SPEC.json', '20261022_c_ept_s3_8'))
    doc = read_json(ask(toy_run, run, report_dir=tmp_path / 'rep'))
    sens = doc['sensitivity_without_smoke_dev']
    assert sens['requested'] is True and sens['computed'] is False and reason in sens['reason']
    flags = {f['code']: f['message'] for f in doc['flags']}
    assert reason in flags['SENSITIVITY_NOT_COMPUTED']


# -- item 3: no band flag on a line fitted on a Monte Carlo mean
def calibrated_mc_world(n_matches=60, rows=20, S=100, seed=5):
    """``(mc_mean, mc_sd, S, realized, match, w)`` of a model that is exactly right and known only through a Monte Carlo mean of S draws:
    the mean is the truth plus the error sd / sqrt(S) of a mean of S draws of sd 800 (80 gold), and the realized value is the truth plus a
    noise of 50 gold. The slope of the realized value on the mean is the reliability var(truth) / var(mean) = 10,000 / 16,400 = 0.61."""
    from c_ept import features

    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(n_matches), rows)
    n = len(match)
    truth = rng.normal(0.0, 100.0, n)
    sd = np.full(n, 800.0)
    mc_mean = truth + rng.normal(size=n) * sd / np.sqrt(S)
    realized = truth + rng.normal(0.0, 50.0, n)
    return mc_mean, sd, S, realized, match, features.match_weights(match)


def test_a_value_line_made_without_judging_has_no_band_flag_and_the_same_numbers_otherwise():
    mc_mean, _, _, realized, match, w = calibrated_mc_world()
    full = report_s3.value_summary(mc_mean, realized, match, w, B=60)
    plain = report_s3.value_summary(mc_mean, realized, match, w, B=60, judged=False)
    assert set(full) - set(plain) == {'slope_in_band', 'bias_in_band'} and all(plain[k] == full[k] for k in plain)


def test_a_calibrated_monte_carlo_line_has_no_band_flag_and_reads_one_against_its_attenuated_reference():
    mc_mean, sd, S, realized, match, w = calibrated_mc_world()
    out = report_s3.mc_value_summary(mc_mean, sd, S, realized, match, w, B=60)
    assert not [k for k in out if k.endswith('_in_band')] and not any(v is False for v in out.values())  # no in-band false
    assert out['slope'] < 0.8  # the band [0.8, 1.2] of the gate would have said False for a model that is exactly right
    reference = report_s3.expected_slope_if_calibrated(mc_mean, sd, S, w)
    assert out['expected_slope_if_calibrated'] == reference == pytest.approx(0.61, abs=0.03)
    assert out['slope_ratio_to_expected'] == out['slope'] / reference and 0.9 < out['slope_ratio_to_expected'] < 1.1
    assert out['note'] == 'read against the attenuated reference, not against 1'
    base = report_s3.value_summary(mc_mean, realized, match, w, B=60, judged=False)
    assert all(out[k] == base[k] for k in base)  # the numbers are those of the gate function; only the reading differs


def test_the_slope_ratio_is_none_when_the_reference_slope_is_not_positive():
    from c_ept import features

    rng = np.random.default_rng(7)
    match, x = np.repeat(np.arange(8), 5), rng.normal(0.0, 1.0, 40)
    out = report_s3.mc_value_summary(x, np.full(40, 100.0), 1, 3.0 * x + rng.normal(size=40), match, features.match_weights(match), B=40)
    assert out['expected_slope_if_calibrated'] == 0.0 and out['slope_ratio_to_expected'] is None  # the noise of the mean swamps its spread


def test_the_composition_summary_has_no_band_flag_on_a_line_made_on_a_monte_carlo_mean_but_keeps_it_on_the_closed_form():
    b0, b1, realized, observed = composition_world()
    out = report_s3.b1_composition(b0, b1, realized, observed, sample='x', B=40)
    for block in (out['b1'], out['b0_same_rows']):
        line = block['value_simulated']
        assert not [k for k in line if k.endswith('_in_band')]
        reference = block['mc_se']['expected_slope_if_calibrated']
        assert reference > 0 and line['expected_slope_if_calibrated'] == reference
        assert line['slope_ratio_to_expected'] == line['slope'] / reference
        assert line['note'] == 'read against the attenuated reference, not against 1'
    closed = out['b0_same_rows']['value_closed_form']  # B0's closed form is no Monte Carlo mean: the band of the gate applies to it
    assert isinstance(closed['slope_in_band'], bool) and isinstance(closed['bias_in_band'], bool)
    assert 'value_simulated' not in closed and 'expected_slope_if_calibrated' not in closed
    for block in (
        out['b1'],
        out['b0_same_rows'],
    ):  # the expected onsets are sums of rate x exposure, not a mean of draws: their lines keep the band
        assert isinstance(block['occurrence_60']['oe_in_band'], bool) and isinstance(block['occurrence_60']['slope_in_band'], bool)


def test_the_report_of_the_toy_run_has_no_band_flag_on_a_line_fitted_on_a_monte_carlo_mean(reported):
    comp = reported.doc['b1_composition']
    for block in (comp['b1'], comp['b0_same_rows']):
        assert not [k for k in block['value_simulated'] if k.endswith('_in_band')] and 'slope_ratio_to_expected' in block['value_simulated']
    assert 'slope_in_band' in comp['b0_same_rows']['value_closed_form']


# -- items 4, 5 and 7: the cost record of a resumed run, of a run with a killed call, and of a run in the current layout
def progress_line(label, best, rounds):
    """A line of fit/PROGRESS.jsonl as the runner writes it for a B1 head. A fit stage that is resumed once B1 was saved reports each head
    with this line (no summary, the rounds at the top level)."""
    return {
        'best_iteration': best,
        'rounds_run': rounds,
        'peak_private_gib': 1.5,
        'peak_private_upper_gib': 60.0,
        'wall_s': 10.0,
        'unit': f'b1.{label}',
        't': 'T',
        'code_commit': 'c',
        'code_dirty': False,
    }


def test_the_b1_rounds_of_a_head_without_a_summary_come_from_the_progress_line_and_what_is_still_none_is_named():
    manifest, ledger, fit, compose = cost_docs()
    fit['b1']['heads']['masked.M2Rate'] = progress_line('masked.M2Rate', [185], [235])
    section, missing = report_s3.cost_section(manifest, ledger, fit, compose)
    line, whole = section['b1_heads']['masked.M2Rate'], section['b1_heads']['masked.M3Gold']
    assert (line['best_iteration'], line['rounds_run'], line['rounds_source']) == ([185], [235], 'progress_line')
    assert line['stopped_early'] is None and line['num_trees'] is None and line['wall_s'] == 10.0  # the line has no more than this
    assert (whole['rounds_source'], whole['best_iteration'], whole['num_trees']) == ('summary', [33, 25], [166, 150])
    assert section['b1_rounds_missing'] == {'stopped_early': ['masked.M2Rate'], 'num_trees': ['masked.M2Rate']}
    assert missing == []  # a resumed run is in the current layout: this is no case for OLD_RECORD_LAYOUT


def test_a_head_without_any_record_has_no_rounds_and_is_named_for_every_field():
    manifest, ledger, fit, compose = cost_docs()
    fit['b1']['heads']['masked.M3Gold'] = None
    section, _ = report_s3.cost_section(manifest, ledger, fit, compose)
    head = section['b1_heads']['masked.M3Gold']
    assert [head[k] for k in report_s3.B1_ROUND_FIELDS] == [None] * 4 and head['rounds_source'] is None
    assert section['b1_rounds_missing'] == {k: ['masked.M3Gold'] for k in report_s3.B1_ROUND_FIELDS}


def test_complete_head_summaries_leave_no_b1_round_missing():
    section, missing = report_s3.cost_section(*cost_docs())
    assert section['b1_rounds_missing'] == {} and missing == []
    assert {h['rounds_source'] for h in section['b1_heads'].values()} == {'summary'}


def test_a_head_summary_is_read_before_the_progress_line_when_a_record_has_both():
    manifest, ledger, fit, compose = cost_docs()
    both = {**progress_line('masked.M2Rate', [1], [2]), 'summary': fit['b1']['heads']['masked.M2Rate']['summary']}
    fit['b1']['heads']['masked.M2Rate'] = both
    head = report_s3.cost_section(manifest, ledger, fit, compose)[0]['b1_heads']['masked.M2Rate']
    assert (head['best_iteration'], head['rounds_run'], head['rounds_source']) == ([185], [235], 'summary')


def test_the_b1_rounds_finding_names_the_fields_and_the_number_of_heads():
    doc = quiet_doc()
    doc['cost']['b1_rounds_missing'] = {'stopped_early': ['a', 'b'], 'num_trees': ['a']}
    (flag,) = report_s3.findings(doc)
    assert flag['code'] == 'B1_ROUNDS_NOT_RECORDED'
    assert 'stopped_early of 2 heads' in flag['message'] and 'num_trees of 1 head)' in flag['message']


def test_only_a_record_with_the_sampled_peak_and_without_the_bound_counts_as_an_old_layout():
    manifest, ledger, fit, compose = cost_docs()
    del fit['tier']['peak_private_upper_gib']
    assert report_s3.cost_section(manifest, ledger, fit, compose)[1] == ['peak_private_upper_gib']
    manifest, ledger, fit, compose = cost_docs()
    fit['tier'] = {'wall_s': 2.0}  # neither peak: a record that was never finished, not an old layout
    assert report_s3.cost_section(manifest, ledger, fit, compose)[1] == []


def test_a_call_that_was_killed_is_reported_as_unfinished_and_is_no_case_for_the_old_layout():
    manifest, ledger, fit, compose = cost_docs()
    assert report_s3.cost_section(manifest, ledger, fit, compose)[0]['unfinished_calls'] == []
    killed = {  # the start fields of a call record: the call never wrote its end
        'started': 't2',
        'stages_requested': ['compose'],
        'workers': 8,
        'resumed': True,
        'smoke': None,
        'code_commit': 'c',
        'code_dirty': False,
    }
    ledger['calls'].append(killed)
    section, missing = report_s3.cost_section(manifest, ledger, fit, compose)
    assert missing == []
    assert section['unfinished_calls'] == [{'call': 1, 'started': 't2', 'stages_requested': ['compose']}]
    first, second = section['runner_calls']
    assert (first['started'], first['finished'], second['started'], second['finished']) == ('t0', 't1', 't2', None)
    assert second['peak_private_gib'] is None and second['lifetime_peak_bound_gib'] is None and second['resumed'] is True


# -- item 11: the compose module text
def test_the_compose_module_text_names_the_censoring_time_beside_the_t0_information_of_decision_p10():
    from c_ept import compose

    text = ' '.join(compose.__doc__.split())
    assert 'using the t0 information plus the censoring time T of the match (decision P10:' in text
    assert 'exposures and conversion windows are cut at T - 30 s and at T; censoring is observation design, not a model input' in text
    assert 'nothing but the information at t0' not in text and 'at t0 only' not in compose.__doc__.splitlines()[0]


# -- item 10: the traceback of an unexpected error is printed without the match keys and the game ids
def planted_key(toy_run) -> str:
    return toy_keys(toy_run)[3]


def raising(error):
    """A function that takes any arguments and raises ``error``."""

    def fail(*args, **kwargs):
        raise error

    return fail


SCRUB_KEYS = ['EUW1_7012345678', 'KR_7099999999']  # keys of the shape of the real ones: platform, underscore, a game id of ten digits
SCRUB_IDS = ['7012345678', '7099999999']


def test_a_traceback_is_printed_with_every_key_and_game_id_taken_out(monkeypatch):
    monkeypatch.setattr(report_s3, '_s1_keys', lambda a: SCRUB_KEYS)
    try:
        raise RuntimeError('match EUW1_7012345678 broke; id 7012345678; other KR_7099999999')
    except RuntimeError as exc:
        text = report_s3.scrubbed_traceback(types.SimpleNamespace(), exc)  # `a` is read by _s1_keys only
    assert text.startswith('Traceback') and 'RuntimeError' in text and 'broke' in text  # it is still a traceback
    assert '<key>' in text and '<id>' in text
    assert not any(k in text for k in SCRUB_KEYS + SCRUB_IDS)


def test_a_key_in_the_cause_of_an_error_is_taken_out_too():
    try:
        try:
            raise ValueError('inner EUW1_7012345678')
        except ValueError as inner:
            raise RuntimeError('outer') from inner
    except RuntimeError as exc:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(report_s3, '_s1_keys', lambda a: SCRUB_KEYS)
            text = report_s3.scrubbed_traceback(types.SimpleNamespace(), exc)
    assert 'ValueError' in text and 'RuntimeError' in text and 'inner <key>' in text
    assert not any(k in text for k in SCRUB_KEYS + SCRUB_IDS)


def test_a_traceback_is_not_printed_when_the_keys_cannot_be_read_and_the_type_of_the_error_is_named(monkeypatch):
    monkeypatch.setattr(report_s3, '_s1_keys', raising(OSError('no store')))
    try:
        raise RuntimeError('some error that quotes EUW1_7012345678')
    except RuntimeError as exc:
        text = report_s3.scrubbed_traceback(types.SimpleNamespace(), exc)
    assert text == (
        'RuntimeError: the traceback is not printed, because the match keys of the S1 store could not be read to take them out of it '
        '(give the S1 run with --s1)\n'
    )
    assert 'Traceback' not in text and 'EUW1' not in text and '7012345678' not in text


def test_an_unexpected_error_prints_its_traceback_without_the_match_keys_and_the_game_ids_and_returns_1(
    toy_run, tmp_path, monkeypatch, capsys
):
    key = planted_key(toy_run)
    game_id = key.rsplit('_', 1)[1]
    # inside report(): the S1 store is open and its keys are known; the message holds a whole key and, apart from it, the game id of the key
    monkeypatch.setattr(report_s3, 'build_report', raising(RuntimeError(f'the match {key} is broken, and its id is {game_id}')))
    run = copy_of(toy_run, tmp_path)
    assert (
        report_s3.main(['--run', str(run), '--s1', str(toy_run.s1), '--s3a', str(toy_run.s3a), '--report-dir', str(tmp_path / 'rep')]) == 1
    )
    err = capsys.readouterr().err
    assert 'Traceback' in err and 'RuntimeError' in err and 'is broken' in err  # the traceback is still the message
    assert key not in err and game_id not in err and 'the match <key> is broken, and its id is <id>' in err
    assert not (tmp_path / 'rep').exists()


def test_the_keys_for_the_scrub_are_found_where_the_report_finds_its_s1_run_when_s1_is_not_given(toy_run, tmp_path, monkeypatch, capsys):
    root = tmp_path / 'runs'
    root.mkdir()
    run = copy_of(toy_run, root)
    shutil.copytree(toy_run.world.s1, root / toy_run.world.s1.name)
    shutil.copytree(toy_run.s3a, root / toy_run.s3a.name)
    key = planted_key(toy_run)
    monkeypatch.setattr(report_s3, 'build_report', raising(ValueError(f'bad {key}')))
    assert report_s3.main(['--run', str(run)]) == 1
    err = capsys.readouterr().err
    assert 'ValueError' in err and key not in err and 'bad <key>' in err


def test_a_traceback_that_cannot_be_scrubbed_is_not_printed_but_named_by_the_type_of_the_error(toy_run, tmp_path, monkeypatch, capsys):
    key = planted_key(toy_run)  # no --s1 and no S1 run next to the run folder: the keys cannot be read
    monkeypatch.setattr(report_s3, 'report', raising(ValueError(f'bad {key}')))
    assert report_s3.main(['--run', str(copy_of(toy_run, tmp_path))]) == 1
    err = capsys.readouterr().err
    assert 'ValueError' in err and key not in err and 'not printed' in err and 'Traceback' not in err


def test_an_error_that_is_a_refusal_is_still_one_line_and_exit_code_2(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(report_s3, 'report', raising(report_s3.ReportInputError('the input is not the one')))
    assert report_s3.main(['--run', str(tmp_path)]) == 2
    assert capsys.readouterr().err == 'report refused: the input is not the one\n'


# -- item 8: the x-label of the window figure
def test_the_window_figure_writes_the_start_time_t0_as_mathtext_in_its_x_label(monkeypatch):
    expected_cum, observed, match, _ = window_world([10] * 12)
    lines = report_s3.window_lines(expected_cum, observed, match, B=40)
    sample = {'b0': {'rows': 120, 'matches': 12, **lines}, 'b1': {'rows': 120, 'matches': 12, **lines}}
    drawn = []
    monkeypatch.setattr(report_s3, '_render', lambda plt, fig: drawn.append(fig) or PNG_SIGNATURE)
    report_s3.fig_windows(report_s3._plt(), {'window_slopes': sample})
    label = drawn[0].axes[0].get_xlabel()
    assert (
        '$t_0$' in label and 't0' not in label
    )  # the same subscript as the title of the figure (the report draws it with the real renderer)
    report_s3._plt().close(drawn[0])


# -- item 9: the winner shares say which units they count
def test_the_winner_labels_take_the_threshold_of_the_sure_stratum_from_their_argument():
    labels = report_s3.winner_labels(200.0)
    assert set(labels) == {'winner_blue', 'winner_red', 'draw', 'sure'} and all(report_s3.WINNER_SCOPE in v for v in labels.values())
    assert '200' in labels['sure'] and '150' not in labels['sure']
    assert 'blue' in labels['winner_blue'] and 'red' in labels['winner_red'] and 'draw' in labels['draw']


def test_the_winner_shares_say_that_they_count_all_valued_units_of_fit_stop_and_dev_out_of_scope_units_included(reported):
    doc = reported.doc['winner']
    assert report_s3.WINNER_SCOPE == 'all valued units of FIT, STOP and DEV, out-of-scope units included'
    assert doc['scope'] == report_s3.WINNER_SCOPE and all(role in doc['scope'] for role in run_s3.LABEL_ROLES)
    assert set(doc['field_labels']) == {'winner_blue', 'winner_red', 'draw', 'sure'}
    assert all(report_s3.WINNER_SCOPE in text for text in doc['field_labels'].values())
    assert (
        'blue' in doc['field_labels']['winner_blue']
        and 'red' in doc['field_labels']['winner_red']
        and 'draw' in doc['field_labels']['draw']
    )
    assert '150' in doc['field_labels']['sure']
    raw = read_json(reported.run / 'labels' / 'LABELS_SUMMARY.json')  # the values are not changed by the label
    assert [doc[k] for k in ('winner_blue', 'winner_red', 'draw', 'sure')] == [
        raw[k] for k in ('winner_blue', 'winner_red', 'draw', 'sure')
    ]


def test_the_toy_run_has_the_current_layout_so_nothing_is_missing_and_nothing_is_old(reported):
    assert reported.doc['inputs']['missing_optional_fields'] == []
    assert 'OLD_RECORD_LAYOUT' not in [f['code'] for f in reported.doc['flags']]
    cost = reported.doc['cost']
    assert cost['unfinished_calls'] == [] and cost['b1_rounds_missing'] == {}
    assert {h['rounds_source'] for h in cost['b1_heads'].values()} == {'summary'}
    assert all(h[k] is not None for h in cost['b1_heads'].values() for k in report_s3.B1_ROUND_FIELDS)


def test_a_run_whose_fit_stage_was_resumed_after_b1_was_saved_still_reports_the_b1_rounds_and_flags_what_is_none(
    toy_run, tmp_path, monkeypatch
):
    monkeypatch.setattr(report_s3, 'BOOT_B', 50)
    run = copy_of(toy_run, tmp_path)
    log = run_s3._ProgressLog(run / 'fit' / 'PROGRESS.jsonl')

    def resumed(facts):  # what _fit_b1 returns when b1/ was saved: the head records are the lines of the log
        heads = {label: log.last(f'b1.{label}') for label in facts['b1']['heads']}
        return {**facts, 'b1': {**facts['b1'], **(log.last('b1') or {}), 'loaded': True, 'heads': heads}}

    original = read_json(toy_run.run / 'fit' / 'FIT_FACTS.json')['b1']['heads']
    rewrite_json(run / 'fit' / 'FIT_FACTS.json', resumed)
    rehash(run)
    doc = read_json(ask(toy_run, run, report_dir=tmp_path / 'rep'))
    heads = doc['cost']['b1_heads']
    assert set(heads) == set(original) and len(heads) == 13
    for label, head in heads.items():
        assert head['rounds_source'] == 'progress_line' and head['stopped_early'] is None and head['num_trees'] is None
        assert head['best_iteration'] == original[label]['summary']['best_iteration']
        assert head['rounds_run'] == original[label]['summary']['rounds_run']
    assert doc['cost']['b1_rounds_missing'] == {'stopped_early': sorted(heads), 'num_trees': sorted(heads)}
    codes = [f['code'] for f in doc['flags']]
    assert 'B1_ROUNDS_NOT_RECORDED' in codes and 'OLD_RECORD_LAYOUT' not in codes and doc['inputs']['missing_optional_fields'] == []
