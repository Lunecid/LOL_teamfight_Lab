"""Tests of ``c_ept.baselines_b0``: the B0 time-state GLM heads of M2, M3 and M4, masked and open (stage S3 Task 6).

No run folder, raw file or catalog is read. The data are synthetic: designs made here with a known truth (the head classes), and the "heads
world" of ``s3_toys`` (small S1, S2 and label stores with the real links between the tables; the loader and ``B0Heads``). Truth comes from
five places:

    brief tests    the five tests of the task brief come first, word for word (their helpers are defined right below them)
    oracles        plain-Python loops over the arrays of the world, written from the text of the brief, that give the training rows of every
                   head (which view a row has, its one-hot columns, its target and exposure); the code is compared with them exactly
    references     ``scipy``-free Newton iterations (IRLS with a log-exposure offset) for the Poisson fit, ``sklearn.linear_model.
                   LogisticRegression`` fitted on the same standardised design for the categorical heads, a planted truth for both
    rulings        R13 (every categorical output is (1 - 1e-6) p + 1e-6 / K over the full class set), R19 (one model per team for the
                   kill and the gold heads: team-specific coefficients for every column), R20 (tolerance 1e-8 for both families, an
                   honest convergence flag), R3 (the composer protocol), R17 (the zone label)
    refusals       inputs that cannot be trusted: counts that are not whole numbers, a view that is missing, an S2 of another run,
                   a TEST role, a changed file
"""
import ast
import json
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from c_ept.models import baselines_b0
from c_ept.preprocessing import features
from c_ept.data import head_rows
from c_ept.models import heads, nll
from c_ept import config as C
from c_ept.data.store_core import Store
from c_ept.tests.conftest import all_imports, module_level_imports
from c_ept.tests.s3_toys import KEY_PREFIX, WORLD_ZERO_CELL, heads_world  # noqa: F401  (the fixture is imported by ruling R1)

SOURCE = Path(baselines_b0.__file__)
# The heads of the tiny heads world have about as many columns as rows and the Poisson fit is nearly unpenalised (alpha = 1e-6), so some of
# its fits stop at the iteration limit; the head records that in ``converged_``. The warning is still raised (one test asserts it).
pytestmark = pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
MIX = 1e-6
HEAD_NAMES = ('M2Rate', 'M2Mark', 'M2Zone', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
OPEN_NAMES = ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')

# The one-hot blocks (value range lo .. lo + n - 1, and whether the lowest level is the reference with no column) in the order of the
# brief's table. Written out here by hand; the module's ``LAYOUTS`` must say the same.
Q, M, Z, S, LAYER = (0, 18), (0, 52), (-1, 11), (0, 4), (1, 6)
EXPECTED = {   # (head, kind) -> (width of the numeric part, ((block name, (lo, n), drop the reference level), ...))
    ('M2Rate', 'masked'): (39, (('q', Q, True),)),
    ('M2Mark', 'masked'): (39, (('q', Q, False),)),
    ('M2Zone', 'masked'): (39, (('q', Q, False), ('m', M, False))),
    ('M3Size', 'masked'): (39, (('q', Q, False), ('m', M, False), ('z', Z, False))),
    # ruling R19: the kill and the gold heads have a model for each team, so no team column in the design of either
    ('M3Kills', 'masked'): (39, (('q', Q, True), ('m', M, True), ('z', Z, True), ('s', S, True))),
    ('M3Gold', 'masked'): (39, (('q', Q, False), ('m', M, False), ('z', Z, False), ('s', S, False), ('layer', LAYER, False))),
    ('M3Obj', 'masked'): (39, (('q', Q, True), ('m', M, True), ('z', Z, True), ('s', S, True))),
    ('M4Conv', 'masked'): (39, (('q', Q, True), ('m', M, True), ('z', Z, True))),
    ('M3Size', 'open'): (39, ()),
    ('M3Kills', 'open'): (39, (('s', S, True),)),
    ('M3Gold', 'open'): (39, (('s', S, False), ('layer', LAYER, False))),
    ('M3Obj', 'open'): (43, (('s', S, True),)),                       # b0 and K (kills blue, red, gold blue, red in thousands)
    ('M4Conv', 'open'): (61, ()),                                     # b0, K and the 18 cells of O
}
WIDTHS = {('M2Rate', 'masked'): 56, ('M2Mark', 'masked'): 57, ('M2Zone', 'masked'): 109, ('M3Size', 'masked'): 120,
          ('M3Kills', 'masked'): 120, ('M3Gold', 'masked'): 130, ('M3Obj', 'masked'): 120, ('M4Conv', 'masked'): 117,
          ('M3Size', 'open'): 39, ('M3Kills', 'open'): 42, ('M3Gold', 'open'): 49, ('M3Obj', 'open'): 46, ('M4Conv', 'open'): 61}
BLUE, RED = 0, 1                                                      # the two teams: the cells of M3Kills, the members of M3Gold


# =====================================================================================================================================
# the helpers of the task brief and the five tests of the task brief
# =====================================================================================================================================
def synthetic_poisson_with_exposure(seed=1, n=20_000):
    """(X (n, 3), counts (n,), exposure (n,)) from a Poisson law with the rate exp(-2 + 0.4 x1 - 0.15 x2 + 0.7 x3) per second; the first
    50 rows have no exposure (and no count)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(3.0, 2.0, n), rng.integers(0, 2, n).astype(float)])
    e = rng.uniform(2.0, 20.0, n)
    e[:50] = 0.0
    y = rng.poisson(np.exp(-2.0 + X @ np.array([0.4, -0.15, 0.7])) * e).astype(float)
    return X, y, e


def poisson_offset_irls(X, y, e, with_intercept=False):
    """The maximum-likelihood slopes (and intercept) of a Poisson regression with the offset log(exposure), by Newton's method on the
    rows with exposure > 0. Written from the textbook (Cameron and Trivedi 2013), not from scikit-learn."""
    keep = e > 0
    X, y, e = X[keep], y[keep], e[keep]
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1])
    b[0] = np.log(y.sum() / e.sum())
    for _ in range(100):
        mu = np.exp(A @ b + np.log(e))
        step = np.linalg.solve((A * mu[:, None]).T @ A, A.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-13:
            break
    return b if with_intercept else b[1:]


def fitted_toy_m4():
    X, y, e = synthetic_poisson_with_exposure(seed=3, n=3_000)
    return baselines_b0.M4Conv().fit(X, y.reshape(-1, 1), e)             # one conversion cell, as a column


def toy_m4_input():
    return np.array([[0.2, 3.0, 1.0]])                                  # one row of the design of the toy (3 columns)


def toy_zone_data(n=500, missing=7, seed=2):
    """(X (n, 5), zone (n,)) with the ten zones, ``missing`` of the entries being -1 (the first kill has no position)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    zone = rng.integers(0, 10, n)
    zone[rng.permutation(n)[:missing]] = -1
    return X, zone


_WORLD = None


@pytest.fixture(scope='module', autouse=True)
def _the_world_of_the_brief_helpers(heads_world):
    """``fit_toy_b0`` is a plain function in the brief; it reads the module's heads world."""
    global _WORLD
    _WORLD = heads_world
    yield
    _WORLD = None


def fit_toy_b0():
    """The B0 heads fitted on the FIT matches of the heads world."""
    return baselines_b0.B0Heads.fit(_WORLD.s1, _WORLD.s2, _WORLD.labels, _WORLD.fit_means)


def toy_batch(n=12, seed=7) -> baselines_b0.HeadBatch:
    rng = np.random.default_rng(seed)
    K = rng.integers(0, 7, (n, 4)).astype(float) * [1, 1, 300, 300]                  # kills blue, red; kill gold blue, red
    return baselines_b0.HeadBatch(row_view=rng.normal(0.0, 3.0, (n, 39)), q=rng.integers(0, 18, n), m=rng.integers(0, 52, n),
                                  z=rng.integers(-1, 10, n), s=rng.integers(0, 4, n), K=K, O=rng.integers(0, 3, (n, 18)).astype(float))


TOY_X = toy_batch()


def test_offset_trick_equals_poisson_with_offset():                 # y/e 목표 + 가중 e = 오프셋 log e
    X, y, e = synthetic_poisson_with_exposure(seed=1)
    assert np.allclose(PlainRate().fit(X, y, e).coef_, poisson_offset_irls(X, y, e), atol=1e-3)


def test_m4_rate_scales_with_conversion_exposure():
    m = fitted_toy_m4()
    x = toy_m4_input()
    assert m.expected(x, exposure_s=30.0) == pytest.approx(0.5 * m.expected(x, exposure_s=60.0))
    assert m.expected(x, exposure_s=0.0) == 0.0


def test_missing_zone_is_left_out_of_rho_and_counted():
    z = baselines_b0.M2Zone().fit(*toy_zone_data(n=500, missing=7, seed=2))
    assert z.n_missing_ == 7 and z.n_fit_ == 493


def test_masked_heads_are_closed_form():
    h = fit_toy_b0().composer_heads('masked')
    assert h.analytic_ok and not h.obj_uses_K and not h.conv_uses_KO


def test_b0_is_deterministic():
    assert np.array_equal(fit_toy_b0().predict_all(TOY_X), fit_toy_b0().predict_all(TOY_X))


# =====================================================================================================================================
# the Poisson heads on synthetic designs
# =====================================================================================================================================
def onehot(values, k, drop_first=True):
    """Plain one-hot columns of the integers ``values`` in 0 .. k - 1 (the level 0 has no column when ``drop_first``)."""
    out = np.zeros((len(values), k))
    out[np.arange(len(values)), values] = 1.0
    return out[:, 1:] if drop_first else out


def poisson_design(seed=0, n=6_000):
    """(X, y, e, rate_truth): two numeric columns on very different scales and a 4-level one-hot block (level 0 dropped)."""
    rng = np.random.default_rng(seed)
    num = np.column_stack([rng.normal(40.0, 12.0, n), rng.normal(-3.0, 0.2, n)])
    level = rng.integers(0, 4, n)
    X = np.column_stack([num, onehot(level, 4)])
    e = rng.uniform(1.0, 30.0, n)
    rate = np.exp(-3.0 + 0.02 * (num[:, 0] - 40.0) + 0.5 * (num[:, 1] + 3.0) + np.array([0.0, 0.3, -0.4, 0.8])[level])
    return X, rng.poisson(rate * e).astype(float), e, rate


def test_poisson_head_agrees_with_the_newton_iteration_when_it_has_a_standardised_block_and_a_one_hot_block():
    X, y, e, _ = poisson_design()
    head = baselines_b0.M3Obj().fit(X, y, e, n_numeric=2)
    truth = poisson_offset_irls(X, y, e, with_intercept=True)             # [intercept, slopes]
    assert head.coef_ == pytest.approx(truth[1:], abs=2e-3)
    assert head.intercept_ == pytest.approx(truth[0], abs=5e-2)


def low_rate_cells(seed, n=20_000, intercept=-4.0):
    """(X, counts, exposure) of a Poisson law at the low rate exp(-4) = 0.018 per second, the order of the first-kill rate of the data."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(3.0, 2.0, n), rng.integers(0, 2, n).astype(float)])
    e = rng.uniform(5.0, 60.0, n)
    return X, rng.poisson(np.exp(intercept + X @ np.array([0.4, -0.15, 0.7])) * e).astype(float), e


def test_the_poisson_head_is_at_the_maximum_likelihood_fit_also_at_the_low_rates_of_the_data():
    """The gradient tolerance of scikit-learn is on the mean deviance, which is tiny at a rate of 0.02 per second: with its default 1e-4
    the solver stops after four iterations and the slopes are off by 1.4e-3 here; the head (tolerance 1e-8) is within 1e-4."""
    from sklearn.linear_model import PoissonRegressor
    X, y, e = low_rate_cells(seed=5)
    truth = poisson_offset_irls(X, y, e, with_intercept=True)
    head = PlainRate().fit(X, y, e)
    assert np.abs(head.coef_ - truth[1:]).max() < 1e-4 and abs(head.intercept_ - truth[0]) < 1e-3
    mean, sd = X.mean(axis=0), X.std(axis=0)
    default = PoissonRegressor(alpha=1e-6, max_iter=1000)                          # the call of the brief as it is, the default tolerance
    default.fit((X - mean) / sd, y / e, sample_weight=e)
    assert np.abs(default.coef_ / sd - truth[1:]).max() > 5e-4                    # the reason for the tighter tolerance (documented probe)


def test_an_intercept_only_poisson_head_is_the_total_count_over_the_total_exposure():
    # hand numbers: counts 0, 2, 1, 4 over 10, 20, 5, 25 seconds, plus a row with no exposure that is left out: 7 / 60 = 0.11666 per second
    X = np.ones((5, 1))                                       # a column with one value standardises to zeros: only the intercept is left
    head = baselines_b0.M4Conv().fit(X, np.array([[0], [2], [1], [4], [0]]), np.array([10.0, 20.0, 5.0, 25.0, 0.0]))
    assert head.predict(X[:1]) == pytest.approx(np.array([[7 / 60]]), rel=1e-6)
    assert head.n_fit_ == 4 and head.n_zero_exposure_ == 1 and head.n_events_.tolist() == [7]
    assert head.expected(X[:1], 30.0) == pytest.approx(np.array([[3.5]]), rel=1e-6)           # 30 s of exposure: 7 / 60 x 30


def test_an_intercept_only_logistic_head_is_the_class_frequencies_mixed_with_the_uniform_law():
    # hand numbers: the classes 0, 0, 0, 1, 1, 2 of M3Size (4 classes): 1/2, 1/3, 1/6 and 0, then (1 - 1e-6) p + 1e-6 / 4
    X = np.ones((6, 1))
    head = baselines_b0.M3Size().fit(X, np.array([0, 0, 0, 1, 1, 2]))
    expected = (1 - MIX) * np.array([1 / 2, 1 / 3, 1 / 6, 0.0]) + MIX / 4
    assert head.predict(X[:1])[0] == pytest.approx(expected, rel=1e-3, abs=1e-9)
    assert head.predict(X[:1])[0, 3] == pytest.approx(MIX / 4, rel=1e-12)                # the class 3 never occurred: exactly 1e-6 / 4


def test_poisson_head_prediction_is_exp_of_the_raw_scale_linear_predictor():
    X, y, e, _ = poisson_design(seed=1)
    head = PlainRate().fit(X, y, e, n_numeric=2)
    assert head.predict(X) == pytest.approx(np.exp(X @ head.coef_ + head.intercept_), rel=1e-10)
    assert head.rate(X) == pytest.approx(head.predict(X), rel=0, abs=0)


def test_poisson_head_standardises_the_numeric_columns_only_with_the_unweighted_statistics_of_the_training_rows():
    X, y, e, _ = poisson_design(seed=2)
    e[:30] = 0.0
    y[:30] = 0.0
    head = PlainRate().fit(X, y, e, n_numeric=2)
    kept = X[e > 0, :2]
    assert head.n_numeric_ == 2
    assert head.mean_ == pytest.approx(kept.mean(axis=0), rel=1e-12)
    assert head.scale_ == pytest.approx(kept.std(axis=0), rel=1e-12)
    assert head.mean_.shape == head.scale_.shape == (2,)            # the one-hot columns have no statistic


def test_poisson_head_gives_a_constant_column_the_scale_one():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=400), np.full(400, 0.1)])
    head = PlainRate().fit(X, rng.poisson(1.0, 400).astype(float), np.ones(400))
    assert head.scale_[1] == 1.0 and head.mean_[1] == 0.1
    assert head.coef_[1] == pytest.approx(0.0, abs=1e-9)


def test_poisson_head_leaves_zero_exposure_rows_out_of_the_fit_and_counts_them():
    X, y, e = synthetic_poisson_with_exposure(seed=4, n=3_000)
    a = PlainRate().fit(X, y, e)
    b = PlainRate().fit(X[50:], y[50:], e[50:])
    assert a.n_zero_exposure_ == 50 and a.n_fit_ == 2_950 and b.n_zero_exposure_ == 0
    assert np.array_equal(a.coef_, b.coef_) and a.intercept_ == b.intercept_


def test_poisson_head_refuses_a_count_in_a_row_without_exposure():
    X, y, e = synthetic_poisson_with_exposure(seed=4, n=500)
    y[3] = 1.0                                                         # row 3 has exposure 0
    with pytest.raises(ValueError, match='exposure'):
        PlainRate().fit(X, y, e)


@pytest.mark.parametrize('what', ['count_float', 'count_negative', 'x_nan', 'x_inf', 'exposure_negative', 'exposure_nan', 'shape'])
def test_poisson_head_refuses_data_that_cannot_be_trusted(what):
    X, y, e = synthetic_poisson_with_exposure(seed=5, n=300)
    if what == 'count_float':
        y[100] = 0.5
    elif what == 'count_negative':
        y[100] = -1.0
    elif what == 'x_nan':
        X[100, 1] = np.nan
    elif what == 'x_inf':
        X[100, 1] = np.inf
    elif what == 'exposure_negative':
        e[100] = -1.0
    elif what == 'exposure_nan':
        e[100] = np.nan
    elif what == 'shape':
        e = e[:-1]
    with pytest.raises(ValueError):
        PlainRate().fit(X, y, e)


def test_poisson_head_refuses_a_numeric_width_beyond_the_design():
    X, y, e = synthetic_poisson_with_exposure(seed=5, n=300)
    with pytest.raises(ValueError, match='n_numeric'):
        PlainRate().fit(X, y, e, n_numeric=4)


def test_poisson_head_with_cells_fits_each_cell_alone():
    X, y, e, _ = poisson_design(seed=6, n=2_000)
    rng = np.random.default_rng(6)
    Y = np.column_stack([y, rng.poisson(0.3, len(y)), rng.poisson(1.5, len(y))]).astype(float)
    many = baselines_b0.M3Obj().fit(X, Y, e, n_numeric=2)
    assert many.coef_.shape == (3, X.shape[1]) and many.intercept_.shape == (3,) and many.predict(X).shape == (len(X), 3)
    for c in range(3):
        one = baselines_b0.M3Obj().fit(X, Y[:, c], e, n_numeric=2)
        assert np.array_equal(many.coef_[c], one.coef_) and many.intercept_[c] == one.intercept_
        assert np.array_equal(many.predict(X)[:, c], one.predict(X))


def test_poisson_head_a_cell_without_any_event_gets_the_rate_half_an_event_over_the_exposure_and_a_finite_nll():
    X, y, e, _ = poisson_design(seed=7, n=1_000)
    Y = np.column_stack([y, np.zeros(len(y))])
    head = baselines_b0.M4Conv().fit(X, Y, e, n_numeric=2)
    assert head.n_events_.tolist() == [int(y.sum()), 0] and head.degenerate_.tolist() == [False, True]
    assert head.predict(X)[:, 1] == pytest.approx(0.5 / e.sum(), rel=1e-12)
    assert np.isfinite(head.nll(X, Y, e, cells=True)).all()
    assert head.coef_[1] == pytest.approx(0.0, abs=0.0)


def test_poisson_head_nll_is_the_poisson_nll_of_the_shared_module():
    X, y, e, _ = poisson_design(seed=8, n=800)
    e[:10], y[:10] = 0.0, 0.0
    head = PlainRate().fit(X, y, e, n_numeric=2)
    assert np.array_equal(head.nll(X, y, e), nll.poisson(y, head.predict(X), e))
    assert (head.nll(X, y, e)[:10] == 0.0).all()                       # Review Focus 3: a cell without exposure adds nothing


def test_poisson_head_nll_of_many_cells_is_the_sum_of_the_cells():
    X, y, e, _ = poisson_design(seed=9, n=500)
    Y = np.column_stack([y, y // 2]).astype(float)
    head = baselines_b0.M3Obj().fit(X, Y, e, n_numeric=2)
    cells = head.nll(X, Y, e, cells=True)
    assert cells.shape == (500, 2) and head.nll(X, Y, e) == pytest.approx(cells.sum(axis=1), rel=1e-14)


def test_poisson_head_expected_is_rate_times_exposure_and_refuses_a_negative_exposure():
    X, y, e, _ = poisson_design(seed=10, n=500)
    head = baselines_b0.M4Conv().fit(X, y.reshape(-1, 1), e, n_numeric=2)
    assert head.expected(X, 42.0) == pytest.approx(42.0 * head.predict(X), rel=1e-14)
    assert head.expected(X, e.reshape(-1, 1)) == pytest.approx(e.reshape(-1, 1) * head.predict(X), rel=1e-14)
    assert (head.expected(X, 0.0) == 0.0).all()
    with pytest.raises(ValueError, match='exposure'):
        head.expected(X, -1.0)


def test_poisson_head_is_deterministic_and_leaves_its_inputs_alone():
    X, y, e, _ = poisson_design(seed=11, n=1_000)
    X0, y0, e0 = X.copy(), y.copy(), e.copy()
    a = PlainRate().fit(X, y, e, n_numeric=2)
    b = PlainRate().fit(X, y, e, n_numeric=2)
    assert np.array_equal(a.coef_, b.coef_) and a.intercept_ == b.intercept_
    assert np.array_equal(X, X0) and np.array_equal(y, y0) and np.array_equal(e, e0)


def test_poisson_head_overwrite_x_standardises_in_place_and_gives_the_same_fit():
    X, y, e, _ = poisson_design(seed=12, n=1_000)
    X2 = X.copy()
    a = PlainRate().fit(X, y, e, n_numeric=2)
    b = PlainRate().fit(X2, y, e, n_numeric=2, overwrite_X=True)
    assert np.array_equal(a.coef_, b.coef_) and a.intercept_ == b.intercept_
    assert not np.array_equal(X2[:, :2], X[:, :2]) and np.array_equal(X2[:, 2:], X[:, 2:])        # only the numeric columns changed


def test_poisson_head_knows_whether_the_optimiser_converged():
    X, y, e, _ = poisson_design(seed=13, n=1_000)
    head = PlainRate().fit(X, y, e, n_numeric=2)
    assert head.converged_.tolist() == [True] and 0 < head.n_iter_[0] < baselines_b0.MAX_ITER_POISSON


def test_poisson_head_uses_the_sklearn_estimator_of_the_brief(monkeypatch):
    from sklearn.linear_model import PoissonRegressor
    seen = []
    real = PoissonRegressor.fit

    def spy(self, X, y, sample_weight=None):
        seen.append((self.alpha, self.max_iter, self.solver, self.fit_intercept, self.tol, np.asarray(y).copy(),
                     np.asarray(sample_weight).copy()))
        return real(self, X, y, sample_weight)
    monkeypatch.setattr(PoissonRegressor, 'fit', spy)
    X, y, e = synthetic_poisson_with_exposure(seed=14, n=400)
    PlainRate().fit(X, y, e)
    alpha, max_iter, solver, intercept, tol, target, weight = seen[0]
    assert (alpha, max_iter, solver, intercept, tol) == (1e-6, 1000, 'lbfgs', True, 1e-8)               # tolerance 1e-8: ruling R20
    keep = e > 0
    assert np.array_equal(weight, e[keep]) and np.array_equal(target, y[keep] / e[keep])      # the target y / e, the weight e


def test_the_poisson_planted_truth_is_recovered_by_the_head():
    X, y, e, rate = poisson_design(seed=15, n=40_000)
    head = baselines_b0.M3Obj().fit(X, y, e, n_numeric=2)
    assert head.predict(X).mean() == pytest.approx(rate.mean(), rel=0.04)
    assert head.coef_[0] == pytest.approx(0.02, abs=0.004) and head.coef_[1] == pytest.approx(0.5, abs=0.12)
    assert head.coef_[2:] == pytest.approx([0.3, -0.4, 0.8], abs=0.1)


def test_poisson_blocks_decomposition_equals_the_matrix_path():
    rng = np.random.default_rng(16)
    n = 600
    num = rng.normal(size=(n, 3))
    a, b = rng.integers(0, 5, n), rng.integers(0, 3, n)
    X = np.column_stack([num, onehot(a, 5), onehot(b, 3)])
    y = rng.poisson(np.exp(0.3 * num[:, 0] + 0.2 * a), n).astype(float)
    blocks = (baselines_b0.Block('a', 0, 5, True), baselines_b0.Block('b', 0, 3, True))
    head = baselines_b0.M3Obj().fit(X, y, np.ones(n), n_numeric=3, blocks=blocks)
    assert head.predict_blocks(num, a=a, b=b) == pytest.approx(head.predict(X), rel=1e-12)
    with pytest.raises(ValueError, match='a'):
        head.predict_blocks(num, a=a + 5, b=b)
    with pytest.raises(ValueError, match='b'):
        head.predict_blocks(num, a=a)                                   # a block without codes


# =====================================================================================================================================
# the categorical heads on synthetic designs
# =====================================================================================================================================
def logit_design(seed=0, n=8_000, classes=(0, 2, 5)):
    """(X, y, probs): two numeric columns, a 3-level one-hot block, and labels from a multinomial logit over the three ``classes``."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 2)) * [1.0, 4.0] + [0.0, 10.0]
    level = rng.integers(0, 3, n)
    X = np.column_stack([num, onehot(level, 3, drop_first=False)])
    W = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -0.05, 0.5, -0.5, 0.0], [-1.0, 0.1, 0.0, 0.7, -0.7]])
    Z = (X - [0.0, 10.0, 0, 0, 0]) @ W.T
    P = np.exp(Z - Z.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    draw = (rng.random((n, 1)) > np.cumsum(P, axis=1)).sum(axis=1)
    return X, np.array(classes)[draw], P


def reference_probabilities(X, y, K, n_numeric):
    """What the head must give: scikit-learn's multinomial logistic regression on the design with the numeric columns standardised by hand,
    its probabilities put on all K classes and mixed with the uniform law (ruling R13)."""
    mean, sd = X[:, :n_numeric].mean(axis=0), X[:, :n_numeric].std(axis=0)
    Xs = X.copy()
    Xs[:, :n_numeric] = (X[:, :n_numeric] - mean) / sd
    lr = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8).fit(Xs, y)                  # tolerance 1e-8: ruling R20
    P = np.zeros((len(X), K))
    P[:, lr.classes_] = lr.predict_proba(Xs)
    return (1.0 - MIX) * P + MIX / K


def test_categorical_head_equals_the_reference_logistic_regression_with_r13_mixing():
    X, y, _ = logit_design()
    y4 = np.where(y == 5, 3, y) % 4                                    # keep the labels inside the four classes of M3Size
    head = baselines_b0.M3Size().fit(X, y4, n_numeric=2)
    assert head.N_CLASSES == 4
    assert head.predict(X) == pytest.approx(reference_probabilities(X, y4, 4, 2), rel=1e-9, abs=1e-12)


def test_categorical_head_probabilities_are_full_width_positive_and_add_up_to_one():
    X, y, _ = logit_design(seed=1, n=2_000)
    y = y % 4
    head = baselines_b0.M3Size().fit(X, y, n_numeric=2)
    P = head.predict(X)
    assert P.shape == (len(X), 4) and P.dtype == np.float64
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12, rtol=0) and (P > 0).all() and (P >= MIX / 4 - 1e-18).all()


def test_ruling_r13_a_class_absent_from_training_gets_one_millionth_over_k():
    X, y, _ = logit_design(seed=2, n=1_500)
    y = np.where(y == 5, 2, y)                                         # classes 0 and 2 only
    head = baselines_b0.M3Size().fit(X, y, n_numeric=2)
    assert head.classes_.tolist() == [0, 2]
    P = head.predict(X)
    assert P[:, [1, 3]] == pytest.approx(MIX / 4, rel=1e-12, abs=0)
    assert np.isfinite(head.nll(X, np.full(len(X), 1))).all()           # class 1 never seen in training: finite, 4 ln 4 ... 13.8 nats
    assert head.nll(X, np.full(len(X), 1)) == pytest.approx(-np.log(MIX / 4), rel=1e-12)


def test_ruling_r13_every_head_has_the_mixing_and_the_full_class_set():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 4))
    team = np.arange(600) % 2
    for cls in (baselines_b0.M2Mark, baselines_b0.M2Zone, baselines_b0.M3Size, baselines_b0.M3Gold):
        K = cls.N_CLASSES
        y = rng.integers(0, 2, 600)                                    # two classes only
        head = cls().fit(X, y, team) if cls is baselines_b0.M3Gold else cls().fit(X, y)         # the gold head has a model for each team
        P = head.predict(X, team) if cls is baselines_b0.M3Gold else head.predict(X)
        assert P.shape == (600, K) and np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert P[:, 2:] == pytest.approx(MIX / K, rel=1e-12, abs=0) and (P > 0).all()


def test_categorical_head_with_one_class_in_training_is_constant_and_mixed():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    head = baselines_b0.M2Zone().fit(X, np.full(100, 6))
    P = head.predict(X)
    assert head.classes_.tolist() == [6]
    assert P[:, 6] == pytest.approx((1 - MIX) + MIX / 10, rel=1e-14) and P[:, 0] == pytest.approx(MIX / 10, rel=1e-12)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-14)


def test_categorical_head_with_two_classes_matches_the_binary_logistic_regression():
    X, y, _ = logit_design(seed=5, n=3_000)
    y2 = np.where(y == 0, 1, 3)                                        # classes 1 and 3 of M3Size
    head = baselines_b0.M3Size().fit(X, y2, n_numeric=2)
    assert head.classes_.tolist() == [1, 3] and head.coef_.shape == (2, 5)
    assert head.predict(X) == pytest.approx(reference_probabilities(X, y2, 4, 2), rel=1e-9, abs=1e-12)


def test_categorical_head_nll_is_minus_log_probability_and_minus_one_scores_zero():
    X, y, _ = logit_design(seed=6, n=1_000)
    y = y % 4
    head = baselines_b0.M3Size().fit(X, y, n_numeric=2)
    P = head.predict(X)
    t = y.copy()
    t[:10] = -1
    v = head.nll(X, t)
    assert np.array_equal(v, nll.categorical(t, P)) and (v[:10] == 0.0).all()
    assert v[10:] == pytest.approx(-np.log(P[np.arange(10, 1_000), y[10:]]), rel=1e-12)


def test_gold_head_scores_with_the_31_bin_nll():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 3))
    y = rng.integers(0, 31, 400)
    team = rng.integers(0, 2, 400)
    head = baselines_b0.M3Gold().fit(X, y, team)
    assert head.N_CLASSES == 31 and head.predict(X, team).shape == (400, 31)
    assert np.array_equal(head.nll(X, y, team), nll.gold(y, head.predict(X, team)))
    y[:5] = -1                                                         # a team without a kill has no gold category: scores 0, row not read
    assert (head.nll(X, y, team)[:5] == 0.0).all()


def test_class_counts_of_the_heads_are_the_ones_of_the_model():
    assert (baselines_b0.M2Mark.N_CLASSES, baselines_b0.M2Zone.N_CLASSES, baselines_b0.M3Size.N_CLASSES, baselines_b0.M3Gold.N_CLASSES) \
        == (C.N_MARKS, features.N_ZONES, 4, nll.N_GOLD_BINS) == (52, 10, 4, 31)


def test_categorical_head_standardises_only_the_numeric_columns():
    X, y, _ = logit_design(seed=8, n=1_200)
    head = baselines_b0.M3Size().fit(X, y % 4, n_numeric=2)
    assert head.mean_ == pytest.approx(X[:, :2].mean(axis=0), rel=1e-12) and head.scale_ == pytest.approx(X[:, :2].std(axis=0), rel=1e-12)
    assert head.mean_.shape == (2,)


def test_categorical_head_raw_scale_coefficients_give_the_same_probabilities():
    X, y, _ = logit_design(seed=9, n=1_500)
    head = baselines_b0.M3Size().fit(X, y % 4, n_numeric=2)
    logits = X @ head.coef_.T + head.intercept_
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    full = np.zeros((len(X), 4))
    full[:, head.classes_] = p
    assert head.predict(X) == pytest.approx((1 - MIX) * full + MIX / 4, rel=1e-10, abs=1e-14)


def test_categorical_head_recovers_a_planted_truth():
    X, y, P = logit_design(seed=10, n=60_000)
    y4 = np.array([0, 1, 3])[np.searchsorted([0, 2, 5], y)]            # classes 0, 1, 3 of M3Size
    head = baselines_b0.M3Size().fit(X, y4, n_numeric=2)
    assert np.abs(head.predict(X)[:, [0, 1, 3]] - P).mean() < 0.01


def test_categorical_head_is_deterministic_and_leaves_its_inputs_alone():
    X, y, _ = logit_design(seed=11, n=1_000)
    y = y % 4
    X0, y0 = X.copy(), y.copy()
    a, b = baselines_b0.M3Size().fit(X, y, n_numeric=2), baselines_b0.M3Size().fit(X, y, n_numeric=2)
    assert np.array_equal(a.coef_, b.coef_) and np.array_equal(a.intercept_, b.intercept_)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    X2 = X.copy()
    c = baselines_b0.M3Size().fit(X2, y, n_numeric=2, overwrite_X=True)
    assert np.array_equal(c.coef_, a.coef_)


def test_categorical_head_uses_the_sklearn_estimator_of_the_brief(monkeypatch):
    seen = []
    real = LogisticRegression.fit

    def spy(self, X, y, sample_weight=None):
        seen.append((self.C, self.max_iter, self.solver, self.tol, sample_weight))
        return real(self, X, y, sample_weight)
    monkeypatch.setattr(LogisticRegression, 'fit', spy)
    X, y, _ = logit_design(seed=12, n=300)
    baselines_b0.M3Size().fit(X, y % 4, n_numeric=2)
    assert seen == [(1.0, 2000, 'lbfgs', 1e-8, None)]                  # equal weight for every row (P9); tolerance 1e-8 (ruling R20)


@pytest.mark.parametrize('what', ['y_float', 'y_too_big', 'y_below_minus_one', 'x_nan', 'shape', 'n_numeric'])
def test_categorical_head_refuses_data_that_cannot_be_trusted(what):
    rng = np.random.default_rng(13)
    X, y = rng.normal(size=(100, 3)), rng.integers(0, 4, 100)
    kwargs = {}
    if what == 'y_float':
        y = y + 0.5
    elif what == 'y_too_big':
        y[5] = 4
    elif what == 'y_below_minus_one':
        y[5] = -2
    elif what == 'x_nan':
        X[5, 0] = np.nan
    elif what == 'shape':
        y = y[:-1]
    else:
        kwargs['n_numeric'] = 9
    with pytest.raises((ValueError, TypeError)):
        baselines_b0.M3Size().fit(X, y, **kwargs)


def test_zone_head_nll_of_a_missing_zone_is_zero_and_the_fit_counts_it():
    X, zone = toy_zone_data(n=300, missing=5, seed=3)
    head = baselines_b0.M2Zone().fit(X, zone)
    assert head.n_missing_ == 5 and head.n_fit_ == 295
    v = head.nll(X, zone)
    assert (v[zone < 0] == 0.0).all() and (v[zone >= 0] > 0).all()


def test_categorical_blocks_decomposition_equals_the_matrix_path():
    rng = np.random.default_rng(14)
    n = 500
    num = rng.normal(size=(n, 3))
    a, b = rng.integers(0, 5, n), rng.integers(0, 3, n)
    X = np.column_stack([num, onehot(a, 5, False), onehot(b, 3, True)])
    y = (a + b) % 4
    blocks = (baselines_b0.Block('a', 0, 5, False), baselines_b0.Block('b', 0, 3, True))
    head = baselines_b0.M3Size().fit(X, y, n_numeric=3, blocks=blocks)
    assert head.predict_blocks(num, a=a, b=b) == pytest.approx(head.predict(X), rel=1e-12)


def test_a_head_without_any_usable_row_is_refused():
    X = np.ones((3, 2))
    with pytest.raises(ValueError, match='no row has any exposure'):
        PlainRate().fit(X, [0, 0, 0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='no row has a class'):
        baselines_b0.M3Size().fit(X, [-1, -1, -1])


def test_a_head_that_was_fitted_without_blocks_cannot_use_the_block_form_and_one_that_is_not_fitted_cannot_predict():
    X, y, e, _ = poisson_design(seed=17, n=300)
    head = baselines_b0.M3Obj().fit(X, y, e, n_numeric=2)                 # 5 columns, but only 2 of them described: no blocks
    with pytest.raises(ValueError, match='not described by blocks'):
        head.predict_blocks(X[:, :2])
    with pytest.raises(RuntimeError, match='not fitted'):
        baselines_b0.M3Obj().predict(X)
    with pytest.raises(ValueError, match='columns'):
        head.predict(X[:, :3])


def test_blocks_must_fill_the_design_exactly():
    X = np.zeros((10, 6))
    X[:, 0] = np.arange(10)
    with pytest.raises(ValueError, match='columns'):
        baselines_b0.M3Size().fit(X, np.arange(10) % 4, n_numeric=2, blocks=(baselines_b0.Block('a', 0, 3, False),))


# =====================================================================================================================================
# the team heads (ruling R19): a model for each team, team-specific coefficients for every column
# =====================================================================================================================================
GOLD_STATE = features.B0_NAMES.index('diff_totalGold')        # a b0 state column: the gold lead of the blue team over the red team
BLUE_FIRST_KILL = 25                                          # marks 0 .. 24: the first kill is the blue team's; 25 .. 49: red's


def planted_kills(seed=0, n=60_000):
    """(X, Y, e): the design of the masked M3Kills head (39 b0 columns, then q, m, z and s as one-hot blocks) and the kills of the two teams
    from a truth that is different for the two teams (g is the gold lead, a b0 column; a mark below 25 is a blue first kill)

        blue  exp(-0.6 + 0.8 [blue first kill] + 0.5 g)          red  exp(-0.6 + 0.8 [red first kill] - 0.5 g)

    so that the ratio red / blue is exp(0.8 - 1.6 [blue first kill] - g)."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 39))
    g = num[:, GOLD_STATE]
    q, z, s, m = rng.integers(0, 18, n), rng.integers(-1, 10, n), rng.integers(0, 4, n), rng.integers(0, 50, n)
    blue_first = (m < BLUE_FIRST_KILL).astype(float)
    Y = np.column_stack([rng.poisson(np.exp(-0.6 + 0.8 * blue_first + 0.5 * g)),
                         rng.poisson(np.exp(-0.6 + 0.8 * (1.0 - blue_first) - 0.5 * g))]).astype(float)
    return np.column_stack([num, onehot(q, 18), onehot(m, 52), onehot(z + 1, 11), onehot(s, 4)]), Y, np.ones(n)


def planted_gold(seed=0, n=8_000):
    """(X, y, team, codes): the design of the masked M3Gold head and a gold bin from a truth that is different for the two teams. The latent
    bin is 14 + a s + b g + c (layer - 3.5) + noise (sd 1.5), s = +1 for a blue-first-kill mark and -1 for a red-first-kill mark, g the gold
    lead (a b0 column); (a, b, c) = (3, 3, 1.2) for the blue and (-3, -3, 0) for the red team, n entries of each team. The two teams have
    the same average bin, so their laws overlap and one set of slopes with a team term cannot tell the two effects apart (its mark and
    state effects come out as 0). ``codes`` holds the integer q, m, z, s and layer of every row."""
    rng = np.random.default_rng(seed)
    designs, bins, teams, codes = [], [], [], {name: [] for name in ('q', 'm', 'z', 's', 'layer')}
    for team, (a, b, c) in enumerate(((3.0, 3.0, 1.2), (-3.0, -3.0, 0.0))):
        num = rng.normal(size=(n, 39))
        q, z, s = rng.integers(0, 18, n), rng.integers(-1, 10, n), rng.integers(0, 4, n)
        m, layer = rng.integers(0, 50, n), rng.integers(1, 7, n)
        sign = np.where(m < BLUE_FIRST_KILL, 1.0, -1.0)
        latent = 14.0 + a * sign + b * num[:, GOLD_STATE] + c * (layer - 3.5) + rng.normal(0.0, 1.5, n)
        designs.append(np.column_stack([num, onehot(q, 18, False), onehot(m, 52, False), onehot(z + 1, 11, False), onehot(s, 4, False),
                                        onehot(layer - 1, 6, False)]))
        bins.append(np.clip(np.rint(latent), 0, 30).astype(int))
        teams.append(np.full(n, team))
        for name, values in zip(codes, (q, m, z, s, layer)):
            codes[name].append(values)
    return np.vstack(designs), np.concatenate(bins), np.concatenate(teams), {k: np.concatenate(v) for k, v in codes.items()}


def test_the_kill_head_has_team_specific_effects_of_the_mark_and_of_a_state_column_r19():
    X, Y, e = planted_kills()
    layout = baselines_b0.LAYOUTS[('M3Kills', 'masked')]
    assert X.shape[1] == layout.width == 120
    head = baselines_b0.M3Kills().fit(X, Y, e, n_numeric=layout.n_numeric, blocks=layout.blocks)
    assert head.coef_.shape == (2, layout.width) and head.intercept_.shape == (2,)                  # a row of coefficients for each team

    def log_ratio(m, g):                                       # log of the expected red kills over the expected blue kills
        numeric = np.zeros((1, 39))
        numeric[0, GOLD_STATE] = g
        k = head.predict_blocks(numeric, q=[3], m=[m], z=[2], s=[1])[0]
        return np.log(k[RED] / k[BLUE])
    # the mark: a blue-first-kill mark against a red-first-kill mark, same state: the log ratio moves by 1.6 (planted)
    assert log_ratio(40, 0.0) - log_ratio(10, 0.0) == pytest.approx(1.6, abs=0.25)
    assert log_ratio(30, 0.0) - log_ratio(5, 0.0) == pytest.approx(1.6, abs=0.25)
    # the state column: a gold lead of +1 against -1, same mark: the log ratio moves by -2.0 (planted -1 per unit of the lead)
    assert log_ratio(10, 1.0) - log_ratio(10, -1.0) == pytest.approx(-2.0, abs=0.25)
    assert log_ratio(40, 1.0) - log_ratio(40, -1.0) == pytest.approx(-2.0, abs=0.25)


def test_the_gold_head_has_team_specific_effects_of_the_mark_of_a_state_column_and_of_the_layer_r19():
    X, y, team, _ = planted_gold()
    layout = baselines_b0.LAYOUTS[('M3Gold', 'masked')]
    assert X.shape[1] == layout.width == 130
    head = baselines_b0.M3Gold().fit(X, y, team, n_numeric=layout.n_numeric, blocks=layout.blocks)

    def expected_bin(t, m, g, layer):
        numeric = np.zeros((1, 39))
        numeric[0, GOLD_STATE] = g
        p = head.predict_blocks(numeric, np.array([t]), q=[3], m=[m], z=[2], s=[1], layer=[layer])[0]
        return float(p @ np.arange(31))
    # The mark: a blue-first-kill mark (10) against a red-first-kill mark (40); planted +6 for the blue team and -6 for the red team. The
    # ridge penalty shrinks a one-hot block of 50 levels: the fit gives about +4 and -4. One set of slopes for both teams gives 0.
    assert expected_bin(BLUE, 10, 0.0, 3) - expected_bin(BLUE, 40, 0.0, 3) > 2.5
    assert expected_bin(RED, 10, 0.0, 3) - expected_bin(RED, 40, 0.0, 3) < -2.5
    # the state column: a gold lead of +1 against -1; planted +6 for the blue team and -6 for the red team (fit: about +/-5.5)
    assert expected_bin(BLUE, 10, 1.0, 3) - expected_bin(BLUE, 10, -1.0, 3) > 4.0
    assert expected_bin(RED, 10, 1.0, 3) - expected_bin(RED, 10, -1.0, 3) < -4.0
    # the kill layer: layer 6 against layer 1; planted +6 for the blue team and nothing for the red team (fit: about +5 and 0; one set
    # of slopes gives the same +2.7 to both)
    assert expected_bin(BLUE, 10, 0.0, 6) - expected_bin(BLUE, 10, 0.0, 1) > 4.0
    assert abs(expected_bin(RED, 10, 0.0, 6) - expected_bin(RED, 10, 0.0, 1)) < 1.0


def test_the_kill_head_is_a_poisson_head_with_one_independent_cell_for_each_team_r19():
    X, Y, e = planted_kills(seed=3, n=3_000)
    layout = baselines_b0.LAYOUTS[('M3Kills', 'masked')]
    both = baselines_b0.M3Kills().fit(X, Y, e, n_numeric=layout.n_numeric, blocks=layout.blocks)
    assert both.predict(X).shape == (3_000, 2) and both.nll(X, Y, e).shape == (3_000,)
    for team in (BLUE, RED):
        alone = baselines_b0.M3Kills().fit(X, Y[:, team], e, n_numeric=layout.n_numeric, blocks=layout.blocks)      # this team's kills only
        assert np.array_equal(both.coef_[team], alone.coef_) and both.intercept_[team] == alone.intercept_


def test_the_gold_head_sends_every_row_to_the_model_of_its_team_r19():
    X, y, team, codes = planted_gold(seed=1, n=1_500)
    layout = baselines_b0.LAYOUTS[('M3Gold', 'masked')]
    head = baselines_b0.M3Gold().fit(X, y, team, n_numeric=layout.n_numeric, blocks=layout.blocks)
    assert len(head.members) == 2 and head.member(BLUE) is head.members[0] and head.member(RED) is head.members[1]
    P = head.predict(X, team)
    for t in (BLUE, RED):
        rows = team == t
        assert np.array_equal(P[rows], head.member(t).predict(X[rows]))
        assert not np.allclose(P[rows], head.member(1 - t).predict(X[rows]), atol=1e-3)           # the other team's model gives other laws
    pick = np.random.default_rng(0).permutation(len(X))[:80]                                        # a mixed batch of both teams
    blocks = head.predict_blocks(X[pick, :39], team[pick], **{k: v[pick] for k, v in codes.items()})
    assert blocks == pytest.approx(head.predict(X[pick], team[pick]), rel=1e-12, abs=1e-15)
    assert head.nll(X, y, team) == pytest.approx(-np.log(P[np.arange(len(X)), y]), rel=1e-12)
    assert head.n_fit_ == len(X) and head.n_missing_ == 0 and head.converged_.shape == (2,) and head.n_iter_.shape == (2,)


def test_the_members_of_the_gold_head_are_independent_fits_on_the_rows_of_their_team_r19():
    X, y, team, _ = planted_gold(seed=2, n=1_200)
    head = baselines_b0.M3Gold().fit(X, y, team, n_numeric=39)                         # 39 numeric columns, the rest as they are
    for t in (BLUE, RED):
        rows = team == t
        expected = reference_probabilities(X[rows], y[rows], 31, 39)                                # scikit-learn on this team's rows alone
        assert head.member(t).predict(X[rows]) == pytest.approx(expected, rel=1e-6, abs=1e-9)
        assert head.member(t).n_fit_ == int(rows.sum())


def test_the_gold_head_refuses_a_missing_team_and_bad_team_codes_and_an_unfitted_use():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 3)), rng.integers(0, 31, 60)
    with pytest.raises(ValueError, match='red'):
        baselines_b0.M3Gold().fit(X, y, np.zeros(60, dtype=int))                                    # no row of the red team
    with pytest.raises(ValueError, match='blue'):
        baselines_b0.M3Gold().fit(X, y, np.ones(60, dtype=int))
    for bad in (np.full(60, 2), np.full(60, -1), np.zeros(59, dtype=int), np.full(60, 0.5), np.zeros((60, 1), dtype=int), None):
        with pytest.raises((ValueError, TypeError), match='team'):
            baselines_b0.M3Gold().fit(X, y, bad)
    head = baselines_b0.M3Gold().fit(X, y, np.arange(60) % 2)
    with pytest.raises(ValueError, match='team'):
        head.predict(X, np.full(60, 3))
    with pytest.raises(ValueError, match='team'):
        head.member(2)
    with pytest.raises(RuntimeError, match='not fitted'):
        baselines_b0.M3Gold().predict(X, np.zeros(60, dtype=int))


# =====================================================================================================================================
# G-S3b B2-a (author, 2026-10-22; ruling R36): the M2Rate head has one state coefficient vector for each gate window
#
#     log rate_q(x) = gamma_q + beta_w(q) . x~,     w(q) = q // 6 (the bins 0-5, 6-11 and 12-17),     x~ the standardised state row
#
# The design that a caller builds is the same as before ([state columns | the q block]); the head makes the window interaction itself.
# =====================================================================================================================================
class PlainRate(baselines_b0._PoissonHead):
    """The Poisson head as M2Rate was before B2-a: one coefficient vector for every row, any design. The unit tests of the shared Poisson
    behaviour above use it, and the B2-a claim below fits it as the single-vector alternative."""
    name = 'M2Rate'


WINDOW_OF_Q = np.repeat(np.arange(3), 6)                      # w(q) = q // 6
Q_BLOCK = baselines_b0.Block('q', 0, 18, True)                # the bin block of the M2Rate design: bin 0 is the reference level
K_WORLD = 3                                                   # state columns of the synthetic window world
WINDOWS_JSON = [[0, 6], [6, 12], [12, 18]]


def window_world(seed=0, n_rows=6_000, strong=0.6):
    """Rows x 18 bins of first-kill counts from a truth that has the shape of the real data: the state effect is strong in the bins 0-5 and
    weak after them. ``z`` is the standardised state the truth is written in, ``x`` what the head is given (columns on other scales). The
    cells are the (row, bin) pairs with exposure (the last bins of a row are cut at the end of its game, one of them partly). The bin
    intercepts vary little (0.03 in the log): the ridge of the head (alpha 1e-6, on the bin columns too) pulls the level of the reference
    bin by about alpha * (the exposure of a row / the expected count of its bin 0) * (the sum of the bin terms), 3e-4 here, and a
    comparison with an unpenalised fit at 1e-3 must leave room for that. Returns a namespace with the matrices (``x``, ``e``, ``y``,
    ``match``) and the cells (``cell_x``, ``cell_q``, ``cell_y``, ``cell_e``, ``X``)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_rows, K_WORLD))
    x = z * np.array([1.0, 4.0, 0.5]) + np.array([0.0, 30.0, -2.0])
    beta = np.array([[strong, -0.5 * strong, 0.3 * strong], [0.08, 0.05, -0.04], [0.05, -0.04, 0.03]])      # one vector for each window
    gamma = np.log(0.02) + 0.03 * np.cos(np.arange(18) / 3.0)                                               # the bin intercepts, per second
    last = rng.integers(8, 18, n_rows)                                                                        # the last bin with exposure
    q = np.arange(18)[None, :]
    e = np.where(q < last[:, None], 10.0, np.where(q == last[:, None], rng.uniform(1.0, 10.0, (n_rows, 1)), 0.0))
    y = rng.poisson(np.exp(gamma[None, :] + np.stack([z @ beta[w] for w in WINDOW_OF_Q], axis=1)) * e)
    row, cell_q = np.nonzero(e > 0)
    cell_x = x[row]
    return types.SimpleNamespace(
        x=x, e=e, y=y, match=np.arange(n_rows) // 10, row=row, cell_x=cell_x, cell_q=cell_q, cell_y=y[row, cell_q].astype(float),
        cell_e=e[row, cell_q], X=np.column_stack([cell_x, onehot(cell_q, 18)]))


def fit_window_head(w):
    return baselines_b0.M2Rate().fit(w.X, w.cell_y, w.cell_e, n_numeric=K_WORLD, blocks=(Q_BLOCK,))


def window_interacted(state, q):
    """The design of the window model written out by hand: [state * 1[w(q) = 0] | state * 1[w(q) = 1] | state * 1[w(q) = 2] | the 17 bin
    columns], 3 k + 17 columns."""
    win = WINDOW_OF_Q[q]
    return np.column_stack([*[state * (win == j)[:, None] for j in range(3)], onehot(q, 18)])


def standardised(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)               # the statistics of the head's own rows, unweighted


def test_the_windows_are_the_three_gate_horizons_in_bins():
    edges = [0] + [h // C.BIN_MS for h in C.EVAL_H_MS]
    assert baselines_b0.M2_WINDOWS == ((0, 6), (6, 12), (12, 18)) == tuple(zip(edges[:-1], edges[1:]))


def test_the_m2_rate_design_has_the_width_of_one_vector_for_each_window_and_the_caller_design_is_the_old_one():
    layout = baselines_b0.LAYOUTS[('M2Rate', 'masked')]
    assert layout.windows == baselines_b0.M2_WINDOWS and layout.width == 39 + 17 and layout.model_width == 3 * 39 + 17 == 134
    for key, other in baselines_b0.LAYOUTS.items():
        if key != ('M2Rate', 'masked'):
            assert other.windows == () and other.model_width == other.width, key


def test_the_m2_rate_head_is_a_poisson_irls_fit_of_the_window_interacted_design(monkeypatch):
    """The head against the textbook Newton iteration on the window-interacted design. The ridge of the head (alpha 1e-6) pulls the bin
    terms of the bins with the least exposure by up to 2e-3 here, so the comparison at 1e-3 is made with the ridge switched to 1e-9 (the
    fit that the reference is); the penalised equations of the real alpha are checked in the score-identity test below."""
    monkeypatch.setattr(baselines_b0, 'ALPHA', 1e-9)
    w = window_world(seed=1, n_rows=1_500)
    head = fit_window_head(w)
    Z = window_interacted(standardised(w.cell_x), w.cell_q)
    assert head.coef_std_.shape == (1, 3 * K_WORLD + 17) == (1, Z.shape[1]) and head.n_numeric_ == K_WORLD
    truth = poisson_offset_irls(Z, w.cell_y, w.cell_e, with_intercept=True)                         # [intercept, slopes]
    assert head.coef_std_[0] == pytest.approx(truth[1:], abs=1e-3)
    assert head.intercept_std_[0] == pytest.approx(truth[0], abs=1e-3)
    raw = poisson_offset_irls(window_interacted(w.cell_x, w.cell_q), w.cell_y, w.cell_e, with_intercept=True)
    assert head.coef_ == pytest.approx(raw[1:], abs=1e-3) and head.intercept_ == pytest.approx(raw[0], abs=1e-3)     # also on the raw scale


def test_the_standardisation_is_shared_by_the_windows_and_the_interacted_columns_are_not_standardised_again(monkeypatch):
    from sklearn.linear_model import PoissonRegressor
    seen = []
    real = PoissonRegressor.fit

    def spy(self, X, y, sample_weight=None):
        seen.append(np.array(X))
        return real(self, X, y, sample_weight)
    monkeypatch.setattr(PoissonRegressor, 'fit', spy)
    w = window_world(seed=4, n_rows=300)
    head = fit_window_head(w)
    expected = window_interacted(standardised(w.cell_x), w.cell_q)
    assert len(seen) == 1 and seen[0].shape == expected.shape and seen[0] == pytest.approx(expected, rel=0, abs=1e-12)
    for j in range(3):
        off = WINDOW_OF_Q[w.cell_q] != j
        assert (seen[0][off][:, j * K_WORLD:(j + 1) * K_WORLD] == 0.0).all()                       # zero outside the window, exactly
    assert head.mean_.shape == head.scale_.shape == (K_WORLD,)                                        # one statistic per state column
    assert head.mean_ == pytest.approx(w.cell_x.mean(axis=0), rel=1e-12) and head.scale_ == pytest.approx(w.cell_x.std(axis=0), rel=1e-12)


def test_the_window_head_satisfies_the_score_identities_of_each_window_in_sample():
    """For scikit-learn's objective (the mean half deviance plus alpha / 2 of the squared coefficients, the intercept not penalised) the
    score equation of a column j is sum_i (y_i - mu_i) x_ij = S alpha coef_j, S being the exposure that the mean is taken over; the solver
    meets it to S times its gradient tolerance. So in every window the observed and the expected counts add up to the same number, and
    so does every standardised state column weighted by the residual, apart from the ridge (under a part in a thousand of the counts)."""
    w = window_world(seed=3, n_rows=3_000)
    head = fit_window_head(w)
    resid = w.cell_y - head.predict(w.X) * w.cell_e                                              # observed - expected count of every cell
    state = (w.cell_x - head.mean_) / head.scale_
    S, alpha, coef = w.cell_e.sum(), baselines_b0.ALPHA, head.coef_std_[0]
    slack = S * 10 * baselines_b0.TOL_POISSON                                                    # ten gradient tolerances of the solver
    bin_term = np.concatenate([[0.0], coef[3 * K_WORLD:]])                                      # c_0 = 0 (the reference bin), c_1 .. c_17
    assert abs(resid.sum()) <= slack                                                            # the intercept is not penalised
    for win in range(3):
        inside = WINDOW_OF_Q[w.cell_q] == win
        count = w.cell_y[inside].sum()
        beta = coef[win * K_WORLD:(win + 1) * K_WORLD]
        total = resid[inside].sum()                                                             # sum (y - mu) over the window
        assert abs(total) <= slack + S * alpha * np.abs(bin_term).sum() and abs(total) < 1e-3 * count
        for j in range(K_WORLD):
            got = (resid[inside] * state[inside, j]).sum()                                       # sum (y - mu) x~_j over the window
            assert abs(got - S * alpha * beta[j]) <= slack and abs(got) < 1e-3 * count, (win, j)
        for q in (c for c in range(1, 18) if WINDOW_OF_Q[c] == win):                            # and each bin on its own
            assert abs(resid[w.cell_q == q].sum() - S * alpha * bin_term[q]) <= slack, q


def test_the_window_head_gives_the_bin_q_the_coefficients_of_the_window_q_div_6():
    k = 2
    mean, scale = np.array([1.0, -2.0]), np.array([2.0, 0.5])
    beta = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])                                    # one coefficient vector for each window
    c = 0.01 * np.arange(18)                                                                    # the bin terms relative to bin 0
    arrays = {'coef_std': np.concatenate([beta.ravel(), c[1:]])[None, :], 'intercept_std': np.array([-3.0]), 'mean': mean, 'scale': scale,
              'n_iter': np.array([5]), 'converged': np.array([True]), 'n_events': np.array([10]), 'degenerate': np.array([False])}
    meta = {'class': 'M2Rate', 'kind': 'masked', 'n_numeric': k, 'n_fit': 100, 'blocks': [['q', 0, 18, True]], 'width': 3 * k + 17,
            'squeeze': True, 'n_zero_exposure': 0, 'windows': WINDOWS_JSON}
    head = baselines_b0.M2Rate.from_state(arrays, meta)
    x = np.array([[3.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    rates = head.rate_bins(x)
    state = (x - mean) / scale
    assert rates.shape == (3, 18)
    for i in range(len(x)):
        for q in range(18):
            assert rates[i, q] == pytest.approx(np.exp(-3.0 + c[q] + state[i] @ beta[q // 6]), rel=1e-12), (i, q)
    for q in (0, 5, 6, 11, 12, 17):                                                             # the other two routes to the same rate
        assert head.predict_blocks(x, q=np.full(3, q)) == pytest.approx(rates[:, q], rel=1e-12)
        assert head.predict(np.column_stack([x, onehot(np.full(3, q), 18)])) == pytest.approx(rates[:, q], rel=1e-12)


def test_the_window_form_calibrates_the_60_s_counts_where_one_coefficient_vector_does_not():
    """The B2-a claim, in sample: with a state effect that is strong in the first minute and weak after it, one coefficient vector
    under-reaches in the first minute (the slope of observed on expected 60 s counts is far above 1) and the window form does not."""
    from c_ept.checks import gate_s3
    w = window_world(seed=7, n_rows=6_000)
    window = fit_window_head(w)
    single = PlainRate().fit(w.X, w.cell_y, w.cell_e, n_numeric=K_WORLD, blocks=(Q_BLOCK,))
    n_rows = len(w.x)
    rate_window = window.rate_bins(w.x)                                                         # (rows, 18)
    rate_single = single.predict_blocks(np.repeat(w.x, 18, axis=0), q=np.tile(np.arange(18), n_rows)).reshape(n_rows, 18)
    observed = w.y[:, :6].sum(axis=1)
    weights = features.match_weights(w.match)

    def slope(rate):
        expected = (rate * w.e)[:, :6].sum(axis=1)
        return gate_s3.occurrence_calibration(expected, observed, w.match, weights, B=20, seed=1)['slope']
    assert 0.95 <= slope(rate_window) <= 1.05
    assert slope(rate_single) > 1.1


def test_the_window_head_state_round_trips_with_its_windows_and_its_width():
    w = window_world(seed=2, n_rows=800)
    head = fit_window_head(w)
    arrays, meta = head.state()
    assert meta['windows'] == WINDOWS_JSON and meta['width'] == 3 * K_WORLD + 17 and meta['n_numeric'] == K_WORLD
    summary = head.summary()
    assert summary['windows'] == WINDOWS_JSON and summary['width'] == 3 * K_WORLD + 17 and len(summary['n_iter']) == 1
    again = baselines_b0.M2Rate.from_state({k: v.copy() for k, v in arrays.items()}, json.loads(json.dumps(meta)))
    assert np.array_equal(again.rate_bins(w.x), head.rate_bins(w.x)) and np.array_equal(again.predict(w.X), head.predict(w.X))
    assert np.array_equal(again.coef_, head.coef_) and again.intercept_ == head.intercept_ and again.n_numeric_ == K_WORLD


def test_the_raw_coefficients_of_the_window_head_give_its_rates_on_the_raw_interacted_design():
    w = window_world(seed=5, n_rows=1_000)
    head = fit_window_head(w)
    raw = window_interacted(w.cell_x, w.cell_q)
    assert raw.shape[1] == head.coef_.shape[0] == 3 * K_WORLD + 17
    assert np.exp(raw @ head.coef_ + head.intercept_) == pytest.approx(head.predict(w.X), rel=1e-9)


def test_a_m2_rate_saved_before_b2_a_is_refused_with_a_clear_message():
    w = window_world(seed=6, n_rows=300)
    arrays, meta = fit_window_head(w).state()
    old = dict(meta)
    del old['windows']                                                                          # a model of one coefficient vector has none
    with pytest.raises(ValueError, match='B2-a'):
        baselines_b0.M2Rate.from_state(arrays, old)
    other = dict(meta, windows=[[0, 9], [9, 18]])
    with pytest.raises(ValueError, match='windows'):
        baselines_b0.M2Rate.from_state(arrays, other)
    narrow = dict(arrays, coef_std=arrays['coef_std'][:, :K_WORLD + 17])                         # the width of the old form
    with pytest.raises(ValueError, match='layout'):
        baselines_b0.M2Rate.from_state(narrow, meta)


@pytest.mark.parametrize('what', ['no_block', 'other_block', 'small_q_block', 'kept_level', 'two_bins_in_a_row', 'two_cells'])
def test_the_window_head_needs_the_bin_block_one_bin_for_each_row_and_one_cell(what):
    w = window_world(seed=8, n_rows=200)
    X, blocks, y = w.X.copy(), (Q_BLOCK,), w.cell_y
    if what == 'no_block':
        X, blocks = w.cell_x, ()
    elif what == 'other_block':
        blocks = (baselines_b0.Block('m', 0, 18, True),)
    elif what == 'small_q_block':
        X, blocks = X[:, :K_WORLD + 11], (baselines_b0.Block('q', 0, 12, True),)
    elif what == 'kept_level':
        X, blocks = np.column_stack([w.cell_x, onehot(w.cell_q, 18, drop_first=False)]), (baselines_b0.Block('q', 0, 18, False),)
    elif what == 'two_bins_in_a_row':
        X[5, K_WORLD:K_WORLD + 2] = 1.0
    else:
        y = np.column_stack([y, y])
    with pytest.raises(ValueError):
        baselines_b0.M2Rate().fit(X, y, w.cell_e, n_numeric=K_WORLD, blocks=blocks)


# =====================================================================================================================================
# the layout of the designs
# =====================================================================================================================================
def test_the_layouts_are_those_of_the_brief_table():
    assert set(baselines_b0.LAYOUTS) == set(EXPECTED)
    for key, (n_numeric, blocks) in EXPECTED.items():
        layout = baselines_b0.LAYOUTS[key]
        assert layout.n_numeric == n_numeric, key
        assert [(b.name, (b.lo, b.n), b.drop_first) for b in layout.blocks] == [(name, rng, drop) for name, rng, drop in blocks], key


def test_the_design_widths():
    for key, width in WIDTHS.items():
        assert baselines_b0.LAYOUTS[key].width == width, key
    assert baselines_b0.LAYOUTS[('M3Kills', 'masked')].width == 39 + 17 + 51 + 10 + 3            # "about 120 columns": no team column (R19)
    assert baselines_b0.LAYOUTS[('M3Gold', 'masked')].width == 39 + 18 + 52 + 11 + 4 + 6


def test_no_design_has_a_team_column_because_the_team_heads_have_a_model_for_each_team_r19():
    for (name, kind), layout in baselines_b0.LAYOUTS.items():
        assert 'team' not in [b.name for b in layout.blocks], (name, kind)


def test_poisson_blocks_drop_the_reference_level_and_the_logistic_blocks_keep_every_level():
    for (name, kind), layout in baselines_b0.LAYOUTS.items():
        poisson = name in ('M2Rate', 'M3Kills', 'M3Obj', 'M4Conv')
        for b in layout.blocks:
            assert b.drop_first == poisson, (name, kind, b.name)


def test_the_zone_input_has_a_none_level_and_the_layer_starts_at_one():
    block = {b.name: b for b in baselines_b0.LAYOUTS[('M3Size', 'masked')].blocks}['z']
    assert (block.lo, block.n) == (-1, 11) == (-1, features.N_ZONES + 1)
    layer = {b.name: b for b in baselines_b0.LAYOUTS[('M3Gold', 'masked')].blocks}['layer']
    assert (layer.lo, layer.n) == (1, 6)


def test_the_head_names_and_kinds():
    assert baselines_b0.HEAD_NAMES == HEAD_NAMES and baselines_b0.OPEN_HEAD_NAMES == OPEN_NAMES
    for name in HEAD_NAMES:
        assert getattr(baselines_b0, name)().name == name == getattr(baselines_b0, name).__name__


def test_the_constants_of_the_brief():
    assert (baselines_b0.ALPHA, baselines_b0.MAX_ITER_POISSON, baselines_b0.TOL_POISSON) == (1e-6, 1000, 1e-8)
    assert (baselines_b0.C_LOGISTIC, baselines_b0.MAX_ITER_LOGISTIC, baselines_b0.TOL_LOGISTIC) == (1.0, 2000, 1e-8)       # ruling R20
    assert baselines_b0.MIX == 1e-6


# =====================================================================================================================================
# ruling R20: tolerance 1e-8 for both families, and a convergence flag that is honest
# =====================================================================================================================================
def warn_after_fit(monkeypatch, estimator):
    """Make ``estimator.fit`` raise scikit-learn's ConvergenceWarning after a normal fit that stopped far below the iteration limit (what
    lbfgs does on an abnormal termination of the line search)."""
    from sklearn.exceptions import ConvergenceWarning
    real = estimator.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        warnings.warn('lbfgs failed to converge (status=2): ABNORMAL', ConvergenceWarning)
        return out
    monkeypatch.setattr(estimator, 'fit', fit)


def test_a_convergence_warning_marks_a_poisson_cell_not_converged_also_below_the_iteration_limit(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    warn_after_fit(monkeypatch, PoissonRegressor)
    X, y, e = synthetic_poisson_with_exposure(seed=21, n=500)
    with pytest.warns(ConvergenceWarning, match='M2Rate'):
        head = PlainRate().fit(X, y, e)
    assert 0 < head.n_iter_[0] < baselines_b0.MAX_ITER_POISSON and head.converged_.tolist() == [False]


def test_a_convergence_warning_marks_a_logistic_head_not_converged_also_below_the_iteration_limit(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    warn_after_fit(monkeypatch, LogisticRegression)
    X, zone = toy_zone_data(n=300, missing=0, seed=6)
    with pytest.warns(ConvergenceWarning, match='M2Zone'):
        head = baselines_b0.M2Zone().fit(X, zone)
    assert 0 < head.n_iter_[0] < baselines_b0.MAX_ITER_LOGISTIC and head.converged_.tolist() == [False]


def test_reaching_the_iteration_limit_marks_a_cell_not_converged_even_when_no_warning_came(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    real_poisson, real_logistic = PoissonRegressor.fit, LogisticRegression.fit

    def poisson_fit(self, X, y, sample_weight=None):
        out = real_poisson(self, X, y, sample_weight)
        self.n_iter_ = self.max_iter                                    # the limit is reached and scikit-learn says nothing
        return out

    def logistic_fit(self, X, y, sample_weight=None):
        out = real_logistic(self, X, y, sample_weight)
        self.n_iter_ = np.array([self.max_iter])
        return out
    monkeypatch.setattr(PoissonRegressor, 'fit', poisson_fit)
    monkeypatch.setattr(LogisticRegression, 'fit', logistic_fit)
    X, y, e = synthetic_poisson_with_exposure(seed=22, n=500)
    Xz, zone = toy_zone_data(n=300, missing=0, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        poisson = PlainRate().fit(X, y, e)
        logistic = baselines_b0.M2Zone().fit(Xz, zone)
    assert poisson.converged_.tolist() == [False] and poisson.n_iter_.tolist() == [baselines_b0.MAX_ITER_POISSON]
    assert logistic.converged_.tolist() == [False] and logistic.n_iter_.tolist() == [baselines_b0.MAX_ITER_LOGISTIC]


def test_a_converged_fit_is_marked_converged_and_raises_no_convergence_warning():
    from sklearn.exceptions import ConvergenceWarning
    X, y, e, _ = poisson_design(seed=23, n=2_000)
    Xc, yc, _ = logit_design(seed=23, n=2_000)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        poisson = PlainRate().fit(X, y, e, n_numeric=2)
        logistic = baselines_b0.M3Size().fit(Xc, yc % 4, n_numeric=2)
    assert poisson.converged_.tolist() == [True] and logistic.converged_.tolist() == [True]
    assert 0 < poisson.n_iter_[0] < baselines_b0.MAX_ITER_POISSON and 0 < logistic.n_iter_[0] < baselines_b0.MAX_ITER_LOGISTIC


def test_every_cell_that_does_not_converge_gives_one_warning_that_names_the_head_and_the_cell(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    warn_after_fit(monkeypatch, PoissonRegressor)
    X, y, e, _ = poisson_design(seed=24, n=600)
    Y = np.column_stack([y, y + 1, y // 2 + 1]).astype(float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        head = baselines_b0.M3Obj().fit(X, Y, e, n_numeric=2)
    ours = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(ours) == 3 and all('M3Obj' in text for text in ours) and [f'cell {c}' in t for c, t in enumerate(ours)] == [True] * 3
    assert head.converged_.tolist() == [False] * 3


def test_a_gold_model_that_does_not_converge_warns_with_the_name_of_its_team(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    warn_after_fit(monkeypatch, LogisticRegression)
    X, y, team, _ = planted_gold(seed=4, n=300)
    layout = baselines_b0.LAYOUTS[('M3Gold', 'masked')]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        head = baselines_b0.M3Gold().fit(X, y, team, n_numeric=layout.n_numeric, blocks=layout.blocks)
    ours = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(ours) == 2 and 'M3Gold (blue team)' in ours[0] and 'M3Gold (red team)' in ours[1]
    assert head.converged_.tolist() == [False, False]


def test_the_gold_head_reports_the_convergence_of_each_team(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    real = LogisticRegression.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        if len(y) == 200:                                              # the red team has 200 rows here, the blue team 300
            warnings.warn('lbfgs failed to converge (status=2): ABNORMAL', ConvergenceWarning)
        return out
    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    X, y, team, _ = planted_gold(seed=5, n=300)
    keep = np.r_[0:300, 300:500]
    layout = baselines_b0.LAYOUTS[('M3Gold', 'masked')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        head = baselines_b0.M3Gold().fit(X[keep], y[keep], team[keep], n_numeric=layout.n_numeric, blocks=layout.blocks)
    summary = head.summary()
    assert head.converged_.tolist() == [True, False] and summary['converged'] == [True, False]
    assert summary['teams']['blue']['converged'] == [True] and summary['teams']['red']['converged'] == [False]
    assert (head.n_iter_ > 0).all() and summary['n_iter'] == head.n_iter_.tolist()


def test_a_warning_that_is_not_a_convergence_warning_goes_on_to_the_caller(monkeypatch):
    from sklearn.linear_model import PoissonRegressor
    real = PoissonRegressor.fit

    def fit(self, X, y, sample_weight=None):
        out = real(self, X, y, sample_weight)
        warnings.warn('something else about the estimator', UserWarning)
        return out
    monkeypatch.setattr(PoissonRegressor, 'fit', fit)
    X, y, e, _ = poisson_design(seed=23, n=2_000)
    with pytest.warns(UserWarning, match='something else about the estimator'):
        head = PlainRate().fit(X, y, e, n_numeric=2)
    assert head.converged_.tolist() == [True]                          # it is no ConvergenceWarning: the flag does not change


def test_a_cell_without_events_and_a_head_with_one_class_are_constant_models_that_count_as_converged():
    X, y, e, _ = poisson_design(seed=25, n=300)
    cells = baselines_b0.M4Conv().fit(X, np.column_stack([y, np.zeros(len(y))]), e, n_numeric=2)
    assert cells.degenerate_.tolist() == [False, True] and cells.converged_.tolist() == [True, True] and cells.n_iter_[1] == 0
    one = baselines_b0.M2Zone().fit(np.ones((20, 2)), np.full(20, 3))
    assert one.converged_.tolist() == [True] and one.n_iter_.tolist() == [0]


# =====================================================================================================================================
# oracles over the heads world: which rows every head is trained on
# =====================================================================================================================================
class World:
    """The arrays of the heads world as numpy arrays, and the plain-Python answers to 'which view', 'which zone', 'which row'."""

    def __init__(self, w):
        self.w = w
        g = lambda n: np.asarray(w.s1.array(n))                                                       # noqa: E731
        self.role, self.fr_match, self.expo = g('matches/role'), g('frame_rows/match'), g('frame_rows/exposure_ms')
        self.on_row, self.on_bin, self.on_unit = g('onsets/row'), g('onsets/bin'), g('onsets/unit')
        self.u_match, self.mark, self.size, self.K, self.O = g('units/match'), g('units/mark'), g('units/size'), g('units/K'), g('units/O')
        self.conv, self.cexp, self.in_scope = g('units/conv'), g('units/conv_exposure_ms'), g('units/in_scope')
        self.kill0 = g('units/kill0')
        self.kx, self.ky, self.kmiss = g('kills/x'), g('kills/y'), g('kills/pos_missing')
        self.ev_unit, self.ev_kind = g('event_rows/unit'), g('event_rows/kind')
        self.src, self.src_index = np.asarray(w.s2.array('views/src')), np.asarray(w.s2.array('views/src_index'))
        self.gold_bin, self.kill_gold = np.asarray(w.labels.array('labels/gold_bin')), np.asarray(w.labels.array('labels/kill_gold'))
        self._b0 = {}

    def role_of_match(self, m):
        return C.ROLES[int(self.role[m])]

    def view(self, src, index):
        """The one view with this src and src_index, found by a plain loop over all views."""
        hits = [v for v in range(len(self.src)) if self.src[v] == src and self.src_index[v] == index]
        assert len(hits) == 1, (src, index, hits)
        return hits[0]

    def b0(self, src, index):
        if (src, index) not in self._b0:
            self._b0[(src, index)] = features.b0(self.w.s2, [self.view(src, index)], self.w.fit_means)[0].astype(np.float64)
        return self._b0[(src, index)]

    def zone(self, u):
        from c_ept.preprocessing import terrain
        k = self.kill0[u]
        return -1 if self.kmiss[k] else int(terrain.zone_of(self.kx[k], self.ky[k]))

    def onsets(self, role):
        return [i for i in range(len(self.on_row)) if self.role_of_match(self.fr_match[self.on_row[i]]) == role]

    def units(self, role):
        return [u for u in range(len(self.u_match)) if self.role_of_match(self.u_match[u]) == role]

    def event_row(self, u, kind):
        hits = [e for e in range(len(self.ev_unit)) if self.ev_unit[e] == u and self.ev_kind[e] == kind]
        assert len(hits) <= 1
        return hits[0] if hits else None

    def kblock(self, u):
        return np.array([self.K[u, 0], self.K[u, 1], self.kill_gold[u, 0] / 1000.0, self.kill_gold[u, 1] / 1000.0])


def row_vector(numeric, blocks):
    """One design row: the numeric part, then for every (value, lo, n, drop_first) the one-hot columns."""
    parts = [np.asarray(numeric, dtype=np.float64)]
    for value, (lo, n), drop in blocks:
        assert lo <= value < lo + n, (value, lo, n)
        v = np.zeros(n)
        v[value - lo] = 1.0
        parts.append(v[1:] if drop else v)
    return np.concatenate(parts)


def blocks_of(key, **values):
    """The (value, (lo, n), drop_first) list of a head in the order of the brief's table, the values by block name."""
    return [(values[name], rng, drop) for name, rng, drop in EXPECTED[key][1]]


@pytest.fixture(scope='module')
def world(heads_world):
    return World(heads_world)


def rows_of(heads_world, head, kind, role='FIT'):
    return baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, role).data(head, kind)


def test_the_world_has_what_the_tests_below_need(world):
    fit = world.onsets('FIT')
    assert len(fit) > 60 and len(world.units('FIT')) > 30
    assert any(world.zone(world.on_unit[i]) < 0 for i in fit) and any(world.zone(world.on_unit[i]) >= 0 for i in fit)
    assert any(world.cexp[u] == 0 and world.in_scope[u] for u in world.units('FIT'))              # a unit with no conversion window
    assert (world.O[:, WORLD_ZERO_CELL] == 0).all() and (world.conv[:, WORLD_ZERO_CELL] == 0).all()
    assert max(world.gold_bin[world.units('FIT')].ravel()) == 30                                    # the open gold bin occurs
    assert set(world.src.tolist()) == {0, 1, 2}


def test_m2_rate_rows_are_the_row_bin_pairs_with_exposure(heads_world, world):
    d = rows_of(heads_world, 'M2Rate', 'masked')
    fit_rows = sorted({r for r in range(len(world.fr_match)) if world.role_of_match(world.fr_match[r]) == 'FIT'})
    pairs = [(r, q) for r in fit_rows for q in range(18) if world.expo[r, q] > 0]
    assert len(d.y) == len(d.e) == len(pairs) == d.X.shape[0]
    counts = {}
    for i in world.onsets('FIT'):
        counts[(world.on_row[i], world.on_bin[i])] = counts.get((world.on_row[i], world.on_bin[i]), 0) + 1
    for j, (r, q) in enumerate(pairs):
        expected = row_vector(world.b0(0, r), blocks_of(('M2Rate', 'masked'), q=q))
        assert np.array_equal(d.X[j], expected), j
        assert d.y[j] == counts.get((r, q), 0) and d.e[j] == world.expo[r, q] / 1000.0 and d.e[j] > 0
        assert d.row[j] == r and d.match[j] == world.fr_match[r]
    assert d.y.sum() == len(world.onsets('FIT'))                      # every FIT onset is counted in exactly one cell


def test_m2_rate_has_no_pair_in_a_bin_without_exposure(heads_world, world):
    d = rows_of(heads_world, 'M2Rate', 'masked')
    assert (d.e > 0).all()
    all_fit_cells = sum(1 for r in range(len(world.fr_match)) if world.role_of_match(world.fr_match[r]) == 'FIT' for q in range(18))
    assert len(d.e) < all_fit_cells                                    # the bins after the cut at the game end are not rows


def test_masked_rows_use_the_frame_clock_view_of_the_onset_row_not_an_event_view(heads_world, world):
    d = rows_of(heads_world, 'M2Mark', 'masked')
    fit = world.onsets('FIT')
    assert d.X.shape == (len(fit), 57) and len(d.y) == len(fit)
    for j, i in enumerate(fit):
        row = world.on_row[i]
        expected = row_vector(world.b0(0, row), blocks_of(('M2Mark', 'masked'), q=world.on_bin[i]))
        assert np.array_equal(d.X[j], expected), j
        assert d.y[j] == world.mark[world.on_unit[i]] and d.row[j] == i and d.match[j] == world.fr_match[row]
    numbers = [[v for v in range(len(world.src)) if world.src_index[v] == world.on_row[i]] for i in fit]
    assert any(len(views) > 1 for views in numbers)                                       # the src_index numbers do collide


def test_masked_mark_target_is_the_mark_of_the_unit_and_agrees_with_the_onset_table(heads_world, world):
    d = rows_of(heads_world, 'M2Mark', 'masked')
    fit = world.onsets('FIT')
    on_mark = np.asarray(heads_world.s1.array('onsets/mark'))
    assert np.array_equal(d.y, on_mark[fit])


def test_m2_zone_keeps_the_missing_zones_as_minus_one_for_the_head_to_count(heads_world, world):
    d = rows_of(heads_world, 'M2Zone', 'masked')
    fit = world.onsets('FIT')
    assert d.X.shape == (len(fit), 109)
    for j, i in enumerate(fit):
        u = world.on_unit[i]
        expected = row_vector(world.b0(0, world.on_row[i]), blocks_of(('M2Zone', 'masked'), q=world.on_bin[i], m=world.mark[u]))
        assert np.array_equal(d.X[j], expected) and d.y[j] == world.zone(u)
    assert (d.y < 0).sum() > 0 and (d.y >= 0).sum() > 0


def test_m3_size_masked_has_the_none_level_of_the_zone(heads_world, world):
    d = rows_of(heads_world, 'M3Size', 'masked')
    fit = world.onsets('FIT')
    assert d.X.shape == (len(fit), 120)
    for j, i in enumerate(fit):
        u = world.on_unit[i]
        expected = row_vector(world.b0(0, world.on_row[i]), blocks_of(('M3Size', 'masked'), q=world.on_bin[i], m=world.mark[u],
                                                                      z=world.zone(u)))
        assert np.array_equal(d.X[j], expected) and d.y[j] == world.size[u]
    none_column = 39 + 18 + 52                                          # z = -1 is the first level of the z block
    assert d.X[:, none_column].sum() == sum(1 for i in fit if world.zone(world.on_unit[i]) < 0) > 0


def test_m3_kills_masked_has_one_row_per_onset_and_one_cell_for_each_team_r19(heads_world, world):
    d = rows_of(heads_world, 'M3Kills', 'masked')
    fit = world.onsets('FIT')
    assert d.X.shape == (len(fit), 120) and d.y.shape == (len(fit), 2) and (d.e == 1.0).all() and d.team is None
    for j, i in enumerate(fit):
        u = world.on_unit[i]
        expected = row_vector(world.b0(0, world.on_row[i]), blocks_of(('M3Kills', 'masked'), q=world.on_bin[i], m=world.mark[u],
                                                                      z=world.zone(u), s=world.size[u]))
        assert np.array_equal(d.X[j], expected) and d.y[j].tolist() == [world.K[u, BLUE], world.K[u, RED]] and d.row[j] == i


def test_m3_gold_masked_has_only_the_team_entries_with_kills_the_layer_and_the_team_of_each_r19(heads_world, world):
    d = rows_of(heads_world, 'M3Gold', 'masked')
    fit = world.onsets('FIT')
    entries = [(i, team) for team in (0, 1) for i in fit if world.K[world.on_unit[i], team] >= 1]
    assert len(d.y) == d.X.shape[0] == len(entries) and d.e is None and d.X.shape[1] == 130
    assert len(entries) < 2 * len(fit)                                   # teams with no kill are no entries
    assert d.team.tolist() == [team for _, team in entries] and set(d.team.tolist()) == {BLUE, RED}      # blue entries first, then red
    for j, (i, team) in enumerate(entries):
        u = world.on_unit[i]
        layer = min(world.K[u, team], 6)
        expected = row_vector(world.b0(0, world.on_row[i]), blocks_of(('M3Gold', 'masked'), q=world.on_bin[i], m=world.mark[u],
                                                                      z=world.zone(u), s=world.size[u], layer=layer))
        assert np.array_equal(d.X[j], expected) and d.y[j] == world.gold_bin[u, team] >= 0 and d.row[j] == i


def test_m3_gold_layer_is_capped_at_six(heads_world, world):
    d = rows_of(heads_world, 'M3Gold', 'masked')
    layer_block = d.X[:, 39 + 18 + 52 + 11 + 4:39 + 18 + 52 + 11 + 4 + 6]
    assert (layer_block.sum(axis=1) == 1).all()
    kills = np.array([world.K[world.on_unit[i], t] for t in (0, 1) for i in world.onsets('FIT') if world.K[world.on_unit[i], t] >= 1])
    assert np.array_equal(layer_block.argmax(axis=1) + 1, np.minimum(kills, 6)) and (kills > 6).any()


def test_m3_obj_masked_has_no_kills_column_p6(heads_world, world):
    d = rows_of(heads_world, 'M3Obj', 'masked')
    fit = world.onsets('FIT')
    assert d.X.shape == (len(fit), 120) and d.y.shape == (len(fit), 18) and (d.e == 1.0).all()
    for j, i in enumerate(fit):
        u = world.on_unit[i]
        blocks = blocks_of(('M3Obj', 'masked'), q=world.on_bin[i], m=world.mark[u], z=world.zone(u), s=world.size[u])
        expected = row_vector(world.b0(0, world.on_row[i]), blocks)
        assert np.array_equal(d.X[j], expected) and np.array_equal(d.y[j], world.O[u])


def test_m4_conv_masked_has_neither_k_nor_o_p6_and_the_conversion_exposure(heads_world, world):
    d = rows_of(heads_world, 'M4Conv', 'masked')
    fit = [i for i in world.onsets('FIT') if world.cexp[world.on_unit[i]] > 0]
    assert d.X.shape == (len(fit), 117) and d.y.shape == (len(fit), 18) and (d.e > 0).all() and d.n_zero_exposure > 0
    assert d.n_zero_exposure == len(world.onsets('FIT')) - len(fit)
    for j, i in enumerate(fit):
        u = world.on_unit[i]
        blocks = blocks_of(('M4Conv', 'masked'), q=world.on_bin[i], m=world.mark[u], z=world.zone(u))
        expected = row_vector(world.b0(0, world.on_row[i]), blocks)
        assert np.array_equal(d.X[j], expected) and np.array_equal(d.y[j], world.conv[u]) and d.e[j] == world.cexp[u] / 1000.0


def test_the_masked_m4_input_is_built_with_the_blessed_constructor(heads_world, monkeypatch):
    seen = []
    real = features.m4_masked_inputs

    def spy(*args, **kwargs):
        seen.append(len(args[0]))
        return real(*args, **kwargs)
    monkeypatch.setattr(features, 'm4_masked_inputs', spy)
    rows_of(heads_world, 'M4Conv', 'masked')
    assert len(seen) >= 1                                              # the masked M4 design goes through features.m4_masked_inputs


def test_open_m3_rows_use_the_src_1_view_of_the_unit_and_m4_rows_the_src_2_view(heads_world, world):
    units = [u for u in world.units('FIT') if world.in_scope[u]]
    d3 = rows_of(heads_world, 'M3Size', 'open')
    assert d3.X.shape == (len(units), 39) and len(d3.y) == len(units)
    for j, u in enumerate(units):
        assert np.array_equal(d3.X[j], world.b0(1, world.event_row(u, 1))) and d3.y[j] == world.size[u] and d3.row[j] == u
    d4 = rows_of(heads_world, 'M4Conv', 'open')
    with_view = [u for u in units if world.event_row(u, 2) is not None]
    assert len(with_view) < len(units) and [world.cexp[u] > 0 for u in units] == [world.event_row(u, 2) is not None for u in units]
    assert d4.X.shape == (len(with_view), 61) and d4.n_zero_exposure == len(units) - len(with_view)
    for j, u in enumerate(with_view):
        expected = np.concatenate([world.b0(2, world.event_row(u, 2)), world.kblock(u), world.O[u]])
        assert np.array_equal(d4.X[j], expected) and np.array_equal(d4.y[j], world.conv[u]) and d4.e[j] == world.cexp[u] / 1000.0


def test_open_m3_kills_gold_and_obj_designs(heads_world, world):
    units = [u for u in world.units('FIT') if world.in_scope[u]]
    n = len(units)
    d = rows_of(heads_world, 'M3Kills', 'open')
    assert d.X.shape == (n, 42) and d.y.shape == (n, 2) and d.team is None                  # one row per unit, one cell for each team (R19)
    for j, u in enumerate(units):
        expected = row_vector(world.b0(1, world.event_row(u, 1)), blocks_of(('M3Kills', 'open'), s=world.size[u]))
        assert np.array_equal(d.X[j], expected) and d.y[j].tolist() == [world.K[u, BLUE], world.K[u, RED]]
    g = rows_of(heads_world, 'M3Gold', 'open')
    entries = [(u, team) for team in (0, 1) for u in units if world.K[u, team] >= 1]
    assert g.X.shape == (len(entries), 49) and g.team.tolist() == [team for _, team in entries]
    for j, (u, team) in enumerate(entries):
        blocks = blocks_of(('M3Gold', 'open'), s=world.size[u], layer=min(world.K[u, team], 6))
        expected = row_vector(world.b0(1, world.event_row(u, 1)), blocks)
        assert np.array_equal(g.X[j], expected) and g.y[j] == world.gold_bin[u, team]
    o = rows_of(heads_world, 'M3Obj', 'open')
    assert o.X.shape == (n, 46) and o.y.shape == (n, 18)
    for j, u in enumerate(units):
        numeric = np.concatenate([world.b0(1, world.event_row(u, 1)), world.kblock(u)])
        expected = row_vector(numeric, blocks_of(('M3Obj', 'open'), s=world.size[u]))
        assert np.array_equal(o.X[j], expected) and np.array_equal(o.y[j], world.O[u])


def test_the_open_k_block_is_the_k_of_the_masked_m4_input(heads_world, world):
    u = world.units('FIT')[5]
    K = np.array([[world.K[u, 0], world.K[u, 1], world.kill_gold[u, 0], world.kill_gold[u, 1]]])
    m4 = features.m4_masked_inputs(np.zeros((1, 39)), [0], [0], [0], K, np.zeros((1, 18)))
    assert np.array_equal(m4[0, 42:46], world.kblock(u).astype(np.float32))


@pytest.mark.parametrize('key', list(EXPECTED))
def test_changing_the_arrays_of_one_call_does_not_change_the_next_call(heads_world, key):
    """A fit with ``overwrite_X`` standardises ``HeadData.X`` in place; whatever a caller does to the arrays of a call, the next call of
    the same ``HeadRows`` gives the same rows."""
    rows = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT')
    first = rows.data(*key)
    kept = [None if a is None else np.array(a) for a in (first.X, first.y, first.e, first.match, first.row, first.team)]
    for a in (first.X, first.y, first.e, first.match, first.row, first.team):
        if a is not None and a.flags.writeable:
            a[...] = 0                                                    # what an in-place fit does to X; the cached arrays refuse it
    again = rows.data(*key)
    for before, after in zip(kept, (again.X, again.y, again.e, again.match, again.row, again.team)):
        assert (before is None and after is None) or np.array_equal(before, after)


def test_the_training_rows_are_of_the_asked_role_only(heads_world, world):
    d = rows_of(heads_world, 'M2Mark', 'masked', role='DEV')
    dev = world.onsets('DEV')
    assert len(dev) > 0 and d.X.shape[0] == len(dev) and set(d.match.tolist()) <= {m for m in range(len(world.role)) if world.role[m] == 3}
    fit = rows_of(heads_world, 'M2Mark', 'masked')
    assert set(d.row.tolist()).isdisjoint(set(fit.row.tolist()))


def test_a_holdout_role_and_a_role_that_s3_does_not_use_are_refused_before_any_array_is_read(heads_world):
    for role in ('TEST', 'EXTERNAL', 'CAL', 'AUDIT'):
        with pytest.raises(ValueError):
            baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, role)


def test_head_rows_refuse_an_s2_store_of_another_run(heads_world, tmp_path):
    other = build_other_s2(heads_world, tmp_path)
    with pytest.raises(ValueError, match='match') as raised:
        baselines_b0.HeadRows(heads_world.s1, other, heads_world.labels, heads_world.fit_means, 'FIT').data('M2Mark', 'masked')
    assert KEY_PREFIX not in str(raised.value)


def build_other_s2(heads_world, tmp_path):
    """A copy of the S2 views with the match numbers shifted by one: the views are of other matches than the S1 rows."""
    s = Store(tmp_path / 'other_s2')
    for name in ('views/src', 'views/src_index', 'views/ref', 'views/t0', 'views/frame_k', 'x/summary', 'x/players_num', 'x/geo_pp'):
        a = np.asarray(heads_world.s2.array(name))
        m = s.allocate(name, a.shape, a.dtype)
        m[...] = a
        m.flush()
        del m
    match = np.asarray(heads_world.s2.array('views/match'))
    m = s.allocate('views/match', match.shape, match.dtype)
    m[...] = (match + 1) % 8
    m.flush()
    del m
    s.finalize()
    return s


def test_head_rows_refuse_labels_of_another_store(heads_world, tmp_path):
    s = Store(tmp_path / 'short_labels')
    for name, dtype in (('labels/kill_gold', 'f8'), ('labels/gold_bin', 'i1')):
        a = s.allocate(name, (3, 2), dtype)
        a[...] = 0
        a.flush()
        del a
    s.finalize()
    with pytest.raises(ValueError, match='labels'):
        baselines_b0.HeadRows(heads_world.s1, heads_world.s2, s, heads_world.fit_means, 'FIT').data('M3Gold', 'masked')


def test_head_rows_refuse_bad_fit_means(heads_world):
    for means in (np.zeros(38), np.full(39, np.nan), None):
        with pytest.raises(ValueError, match='fit_means'):
            baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, means, 'FIT')


def test_head_rows_name_a_head_and_a_kind_that_exist(heads_world):
    rows = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT')
    for head, kind in (('M2Rate', 'open'), ('M2Mark', 'open'), ('M5', 'masked'), ('M3Size', 'both')):
        with pytest.raises(ValueError):
            rows.data(head, kind)


# ---- stores that do not fit together are refused ---------------------------------------------------------------------------------------
class Copy:
    """A writable copy of the three stores of the heads world, to break one thing at a time."""

    def __init__(self, heads_world, tmp_path):
        import shutil
        for part in ('s1', 's2', 'labels'):
            shutil.copytree(heads_world.s1.root.parent / part, tmp_path / part)
        self.root, self.world = tmp_path, heads_world

    def edit(self, part, name):
        """The array ``name`` of the copied store ``part`` opened for writing (flush and delete it before reading the store again)."""
        return np.load(Store(self.root / part).path(name), mmap_mode='r+')

    def rows(self, role='FIT'):
        stores = (Store(self.root / 's1'), Store(self.root / 's2'), Store(self.root / 'labels'))
        return baselines_b0.HeadRows(*stores, self.world.fit_means, role)


def first_fit(world, predicate):
    """The first FIT unit for which ``predicate(unit)`` holds."""
    return next(u for u in world.units('FIT') if predicate(u))


def test_an_s2_store_with_other_reference_times_is_refused(heads_world, tmp_path):
    c = Copy(heads_world, tmp_path)
    ref = c.edit('s2', 'views/ref')
    ref[:] += 1
    ref.flush()
    del ref
    with pytest.raises(ValueError, match='reference times|does not belong'):
        c.rows().data('M2Mark', 'masked')


def test_onsets_whose_mark_is_not_the_mark_of_their_unit_are_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    mark = c.edit('s1', 'onsets/mark')
    mark[world.onsets('FIT')[0]] = (mark[world.onsets('FIT')[0]] + 1) % 52
    mark.flush()
    del mark
    with pytest.raises(ValueError, match='onsets/mark'):
        c.rows().data('M2Mark', 'masked')


def test_a_unit_with_two_event_rows_of_one_kind_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    ev = c.edit('s1', 'event_rows/unit')
    m3 = [e for e in range(len(world.ev_unit)) if world.ev_kind[e] == 1 and world.role_of_match(world.u_match[world.ev_unit[e]]) == 'FIT']
    ev[m3[1]] = ev[m3[0]]
    ev.flush()
    del ev
    with pytest.raises(ValueError, match='two event rows'):
        c.rows().data('M3Size', 'open')


def test_an_m4_event_row_of_a_unit_without_a_conversion_window_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    u = first_fit(world, lambda u: world.in_scope[u] and world.event_row(u, 2) is not None)
    exposure = c.edit('s1', 'units/conv_exposure_ms')
    exposure[u] = 0
    exposure.flush()
    del exposure
    with pytest.raises(ValueError, match='conversion window'):
        c.rows().data('M4Conv', 'open')


def test_two_frame_clock_views_with_one_src_index_are_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    frame = [v for v in range(len(world.src)) if world.src[v] == 0]
    index = c.edit('s2', 'views/src_index')
    index[frame[1]] = index[frame[0]]
    index.flush()
    del index
    with pytest.raises(ValueError, match='same src_index'):
        c.rows().data('M2Mark', 'masked')


def test_a_src_index_beyond_the_rows_of_s1_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    index = c.edit('s2', 'views/src_index')
    index[[v for v in range(len(world.src)) if world.src[v] == 0][0]] = 10_000
    index.flush()
    del index
    with pytest.raises(ValueError, match='src_index'):
        c.rows().data('M2Mark', 'masked')


def test_a_frame_row_without_a_frame_clock_view_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    fit_row = world.onsets('FIT')[0]
    view = world.view(0, world.on_row[fit_row])
    src = c.edit('s2', 'views/src')
    src[view] = 3                                                         # no longer a frame-clock view (and not an event view either)
    src.flush()
    del src
    with pytest.raises(ValueError, match='no frame-clock view'):
        c.rows().data('M2Mark', 'masked')


def test_an_event_view_of_another_kind_than_its_event_row_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    u = first_fit(world, lambda u: world.in_scope[u])
    src = c.edit('s2', 'views/src')
    src[world.view(1, world.event_row(u, 1))] = 2
    src.flush()
    del src
    with pytest.raises(ValueError, match='kind'):
        c.rows().data('M3Size', 'open')


def test_an_onset_in_a_bin_without_exposure_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    i = world.onsets('FIT')[0]
    exposure = c.edit('s1', 'frame_rows/exposure_ms')
    exposure[world.on_row[i], world.on_bin[i]] = 0
    exposure.flush()
    del exposure
    with pytest.raises(ValueError, match='without exposure'):
        c.rows().data('M2Rate', 'masked')


def test_a_team_with_kills_but_without_a_gold_bin_is_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    u = first_fit(world, lambda u: world.K[u, 0] >= 1 and any(world.on_unit[i] == u for i in world.onsets('FIT')))
    gold = c.edit('labels', 'labels/gold_bin')
    gold[u, 0] = -1
    gold.flush()
    del gold
    with pytest.raises(ValueError, match='gold_bin'):
        c.rows().data('M3Gold', 'masked')


def test_labels_that_are_not_finite_for_a_unit_of_the_role_are_refused(heads_world, world, tmp_path):
    c = Copy(heads_world, tmp_path)
    u = first_fit(world, lambda u: world.in_scope[u])
    gold = c.edit('labels', 'labels/kill_gold')
    gold[u, 1] = np.nan
    gold.flush()
    del gold
    with pytest.raises(ValueError, match='kill_gold'):
        c.rows().data('M3Obj', 'open')


def test_an_s1_store_without_an_array_the_heads_need_is_refused_by_name(heads_world, tmp_path):
    c = Copy(heads_world, tmp_path)
    (c.root / 's1' / 'units' / 'size.npy').unlink()
    with pytest.raises(ValueError, match='units/size') as raised:
        c.rows().data('M3Size', 'masked')
    assert KEY_PREFIX not in str(raised.value)


def test_a_role_without_rows_is_refused(heads_world, tmp_path):
    import gc
    import shutil
    from c_ept.tests.s3_toys import build_heads_world
    stop = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'STOP')
    assert stop.data('M2Mark', 'masked').X.shape[0] > 0                    # the STOP match has onsets in the heads world
    world = build_heads_world(tmp_path / 'no_dev', roles=('FIT', 'FIT', 'STOP'))
    rows = baselines_b0.HeadRows(world.s1, world.s2, world.labels, world.fit_means, 'DEV')
    with pytest.raises(ValueError, match='no onset'):
        rows.data('M2Mark', 'masked')
    with pytest.raises(ValueError, match='no unit in scope'):
        rows.data('M3Size', 'open')
    with pytest.raises(ValueError, match='no frame row'):
        rows.data('M2Rate', 'masked')
    del rows, world
    gc.collect()
    shutil.rmtree(tmp_path / 'no_dev', ignore_errors=True)


# =====================================================================================================================================
# B0Heads: fit
# =====================================================================================================================================
@pytest.fixture(scope='module')
def fitted(heads_world):
    return baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means)


def test_fit_makes_the_eight_masked_and_the_five_open_heads(fitted):
    assert tuple(fitted.masked) == HEAD_NAMES and tuple(fitted.open) == OPEN_NAMES
    for kind, group in (('masked', fitted.masked), ('open', fitted.open)):
        for name, head in group.items():
            assert type(head) is getattr(baselines_b0, name) and head.kind == kind and head.n_fit_ > 0


def test_fit_stores_the_fit_means_and_the_roles(fitted, heads_world):
    assert np.array_equal(fitted.fit_means, heads_world.fit_means) and fitted.fit_roles == ('FIT',)


def same_parameters(a, b) -> bool:
    """Whether two heads have bit-identical fitted arrays (coefficients, intercepts, standardisation, iterations, ...); ``state()`` has them
    for every head, also for the gold head that holds a model for each team."""
    arrays_a, arrays_b = a.state()[0], b.state()[0]
    return set(arrays_a) == set(arrays_b) and all(np.array_equal(arrays_a[k], arrays_b[k]) for k in arrays_a)


def test_every_head_is_what_a_fresh_head_gives_on_the_rows_of_the_oracle(fitted, heads_world):
    """B0Heads.fit only wires rows to heads: a fresh head fitted on the same rows has the same coefficients."""
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            d = rows_of(heads_world, name, kind)
            layout = baselines_b0.LAYOUTS[(name, kind)]
            fresh = getattr(baselines_b0, name)(kind)
            args = (d.X, d.y) if d.e is None else (d.X, d.y, d.e)
            extra = {} if d.team is None else {'team': d.team}                                  # the gold head: the team of every row
            fresh.fit(*args, **extra, n_numeric=layout.n_numeric, blocks=layout.blocks)
            assert same_parameters(fresh, head), (kind, name)


def test_the_zone_head_counts_the_onsets_without_a_position(fitted, world):
    fit = world.onsets('FIT')
    missing = sum(1 for i in fit if world.zone(world.on_unit[i]) < 0)
    assert fitted.masked['M2Zone'].n_missing_ == missing > 0 and fitted.masked['M2Zone'].n_fit_ == len(fit) - missing


def test_the_zero_exposure_conversions_are_left_out_of_the_m4_fit_and_counted(fitted, world):
    fit = world.onsets('FIT')
    zero = sum(1 for i in fit if world.cexp[world.on_unit[i]] == 0)
    assert zero > 0 and fitted.masked['M4Conv'].n_zero_exposure_ == zero and fitted.masked['M4Conv'].n_fit_ == len(fit) - zero
    units = [u for u in world.units('FIT') if world.in_scope[u]]
    assert fitted.open['M4Conv'].n_fit_ == sum(1 for u in units if world.cexp[u] > 0)
    assert fitted.open['M4Conv'].n_zero_exposure_ == sum(1 for u in units if world.cexp[u] == 0) > 0
    cells = sum(1 for r in range(len(world.fr_match)) if world.role_of_match(world.fr_match[r]) == 'FIT' for q in range(18)
                if world.expo[r, q] == 0)
    assert fitted.masked['M2Rate'].n_zero_exposure_ == cells > 0                # the bins beyond the cut at the game end


def test_the_cell_that_never_happens_has_the_half_event_rate(fitted):
    for head in (fitted.masked['M3Obj'], fitted.open['M3Obj'], fitted.masked['M4Conv'], fitted.open['M4Conv']):
        assert head.degenerate_[WORLD_ZERO_CELL] and head.n_events_[WORLD_ZERO_CELL] == 0
        assert np.array_equal(head.degenerate_, head.n_events_ == 0)           # a cell is degenerate exactly when it has no event
        assert head.degenerate_.sum() < len(head.degenerate_)                  # the other cells of the world do happen


def test_fit_summary_of_the_heads_is_exposed_for_the_report(fitted):
    summary = fitted.summary()
    assert set(summary) == {'masked', 'open'} and tuple(summary['masked']) == HEAD_NAMES
    one = summary['masked']['M2Zone']
    zone = fitted.masked['M2Zone']
    assert one['n_missing'] == zone.n_missing_ and one['kind'] == 'masked' and one['n_fit'] == zone.n_fit_
    json.dumps(summary, allow_nan=False)                                # plain JSON values


def test_the_summary_reports_iterations_and_convergence_of_every_head_and_every_cell_r20(fitted):
    # M3Kills and M3Gold: a cell (a model) for each team
    cells = {'M2Rate': 1, 'M2Mark': 1, 'M2Zone': 1, 'M3Size': 1, 'M3Kills': 2, 'M3Gold': 2, 'M3Obj': 18, 'M4Conv': 18}
    summary = fitted.summary()
    for kind in ('masked', 'open'):
        for name, one in summary[kind].items():
            head = getattr(fitted, kind)[name]
            assert len(one['n_iter']) == len(one['converged']) == cells[name], (kind, name)
            assert all(type(c) is bool for c in one['converged']) and all(type(i) is int and i >= 0 for i in one['n_iter']), (kind, name)
            assert one['n_iter'] == [int(i) for i in head.n_iter_] and one['converged'] == [bool(c) for c in head.converged_], (kind, name)
    for kind in ('masked', 'open'):
        gold = summary[kind]['M3Gold']
        assert set(gold['teams']) == {'blue', 'red'}
        assert [gold['teams'][t]['converged'][0] for t in ('blue', 'red')] == gold['converged']        # one entry for each team, blue first
        assert [gold['teams'][t]['n_iter'][0] for t in ('blue', 'red')] == gold['n_iter']
        assert len(summary[kind]['M3Kills']['n_events']) == 2                                  # the kills of the blue and of the red team
    for kind in ('masked', 'open'):
        for name in ('M3Obj', 'M4Conv'):
            one = summary[kind][name]
            for n_iter, converged, degenerate in zip(one['n_iter'], one['converged'], one['degenerate']):
                assert (n_iter == 0 and converged) if degenerate else n_iter > 0              # a cell without events is a constant model


def test_fitting_uses_the_fit_matches_only(heads_world, fitted, tmp_path):
    """Scramble every row of S1, S2 and the labels that belongs to a match that is not FIT: the heads do not change."""
    import shutil
    copy = tmp_path / 'scrambled'
    for part in ('s1', 's2', 'labels'):
        shutil.copytree(heads_world.s1.root.parent / part, copy / part)
    s1, s2, labels = Store(copy / 's1'), Store(copy / 's2'), Store(copy / 'labels')
    rng = np.random.default_rng(0)
    role = np.asarray(s1.array('matches/role'))
    fit_unit = role[np.asarray(s1.array('units/match'))] == 0
    fit_row = role[np.asarray(s1.array('frame_rows/match'))] == 0
    fit_view = role[np.asarray(s2.array('views/match'))] == 0
    fit_onset = fit_row[np.asarray(s1.array('onsets/row'))]
    for name, keep in (('units/mark', fit_unit), ('units/size', fit_unit), ('units/K', fit_unit), ('units/O', fit_unit),
                       ('units/conv', fit_unit), ('units/conv_exposure_ms', fit_unit), ('frame_rows/exposure_ms', fit_row),
                       ('onsets/bin', fit_onset), ('onsets/mark', fit_onset), ('labels/gold_bin', fit_unit), ('labels/kill_gold', fit_unit),
                       ('x/summary', fit_view), ('x/players_num', fit_view), ('x/geo_pp', fit_view)):
        store = labels if name.startswith('labels/') else (s2 if name.startswith('x/') else s1)
        a = np.load(store.path(name), mmap_mode='r+')
        junk = rng.permutation(a[~keep].reshape(-1)).reshape(a[~keep].shape) if a.dtype.kind != 'f' else a[~keep][::-1] + 1.0
        a[~keep] = junk
        a.flush()
        del a
    again = baselines_b0.B0Heads.fit(s1, s2, labels, heads_world.fit_means)
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            assert same_parameters(head, getattr(again, kind)[name]), (kind, name)


def test_fit_standardises_with_the_statistics_of_the_fit_rows(fitted, heads_world):
    d = rows_of(heads_world, 'M2Mark', 'masked')
    head = fitted.masked['M2Mark']
    assert head.mean_ == pytest.approx(d.X[:, :39].mean(axis=0), rel=1e-10) and head.n_numeric_ == 39
    m2 = rows_of(heads_world, 'M2Rate', 'masked')
    assert fitted.masked['M2Rate'].mean_ == pytest.approx(m2.X[:, :39].mean(axis=0), rel=1e-10)      # over the (row, bin) pairs


def test_fit_calls_the_e6_check_before_the_reads_and_before_each_head(heads_world):
    calls = []
    baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, check=lambda: calls.append(1))
    assert len(calls) == 1 + len(HEAD_NAMES) + len(OPEN_NAMES)


def test_an_e6_refusal_stops_the_fit_before_anything_is_read(heads_world, monkeypatch):
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])

    def refuse():
        raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, check=refuse)
    assert reads == []


def test_fit_refuses_a_holdout_role_and_a_wrong_roles_argument(heads_world):
    args = (heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means)
    for roles in ({'fit': 'TEST'}, {'fit': 'EXTERNAL'}, {'fit': 'CAL'}):
        with pytest.raises(ValueError) as raised:
            baselines_b0.B0Heads.fit(*args, roles=roles)
        assert KEY_PREFIX not in str(raised.value)
    for roles in ({'train': 'FIT'}, {}, 'FIT', ['FIT']):
        with pytest.raises((ValueError, TypeError)):
            baselines_b0.B0Heads.fit(*args, roles=roles)


def test_fit_accepts_the_roles_argument_of_the_brief(heads_world, fitted):
    again = baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, roles={'fit': 'FIT'})
    assert np.array_equal(again.masked['M3Size'].coef_, fitted.masked['M3Size'].coef_) and again.fit_roles == ('FIT',)


def test_fit_leaks_no_match_key(heads_world):
    with pytest.raises(ValueError) as raised:
        baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means[:5])
    assert KEY_PREFIX not in str(raised.value)


# =====================================================================================================================================
# B0Heads: scoring (row-wise NLL)
# =====================================================================================================================================
@pytest.fixture(scope='module')
def scores(fitted, heads_world):
    return fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')


def test_nll_gives_a_score_object_for_every_masked_head(scores):
    assert tuple(scores) == HEAD_NAMES
    for name, sc in scores.items():
        assert len(sc.nll) == len(sc.match) == len(sc.row) > 0 and np.isfinite(sc.nll).all() and (sc.nll >= 0).all(), name


def test_the_zone_scores_leave_out_the_missing_zones_and_count_them(scores, world):
    dev = world.onsets('DEV')
    missing = sum(1 for i in dev if world.zone(world.on_unit[i]) < 0)
    zone = scores['M2Zone']
    assert zone.n_unscored == missing and len(zone.nll) == len(dev) - missing
    assert scores['M2Mark'].n_unscored == 0 and len(scores['M2Mark'].nll) == len(dev)


def test_rows_and_cells_without_exposure_are_not_scored_and_are_counted(scores, world):
    dev = world.onsets('DEV')
    zero_conversions = sum(1 for i in dev if world.cexp[world.on_unit[i]] == 0)
    assert scores['M4Conv'].n_unscored == zero_conversions and len(scores['M4Conv'].nll) == len(dev) - zero_conversions
    frame_rows = [r for r in range(len(world.fr_match)) if world.role_of_match(world.fr_match[r]) == 'DEV']
    cells = sum(1 for r in frame_rows for q in range(18) if world.expo[r, q] == 0)
    assert scores['M2Rate'].n_unscored == cells > 0 and len(scores['M2Rate'].nll) == 18 * len(frame_rows) - cells
    assert all(scores[name].n_unscored == 0 for name in ('M2Mark', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj'))


def test_the_m2_rate_scores_are_per_frame_row_sums_over_the_bins(scores, fitted, heads_world, world):
    d = rows_of(heads_world, 'M2Rate', 'masked', role='DEV')
    cell = fitted.masked['M2Rate'].nll(d.X, d.y, d.e)
    values, match = scores['M2Rate'].per_row()
    expected = {}
    for r, v in zip(d.row, cell):
        expected[int(r)] = expected.get(int(r), 0.0) + v
    assert len(values) == len(expected) == len(set(d.row.tolist()))
    assert sorted(values.tolist()) == pytest.approx(sorted(expected.values()), rel=1e-12)
    assert set(match.tolist()) <= {m for m in range(len(world.role)) if world.role[m] == 3}


def test_the_gold_scores_have_the_team_entries_with_kills_only(scores, world):
    entries = sum(1 for i in world.onsets('DEV') for t in (0, 1) if world.K[world.on_unit[i], t] >= 1)
    assert len(scores['M3Gold'].nll) == entries and scores['M3Gold'].n_unscored == 0


def test_the_gold_scores_come_from_the_model_of_the_team_of_each_entry_r19(fitted, heads_world):
    for kind in ('masked', 'open'):
        d = rows_of(heads_world, 'M3Gold', kind, role='DEV')
        gold = getattr(fitted, kind)['M3Gold']
        got = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind=kind)['M3Gold']
        assert set(d.team.tolist()) == {BLUE, RED}, kind
        expected = np.empty(len(d.y))
        for t in (BLUE, RED):
            sel = np.flatnonzero(d.team == t)
            expected[sel] = -np.log(gold.member(t).predict(d.X[sel])[np.arange(len(sel)), d.y[sel]])
        assert got.nll == pytest.approx(expected, rel=1e-12), kind
        assert np.array_equal(got.row, d.row) and np.array_equal(got.match, d.match), kind


def test_the_kill_scores_sum_the_two_teams_of_an_onset(scores, fitted, heads_world, world):
    d = rows_of(heads_world, 'M3Kills', 'masked', role='DEV')
    cells = fitted.masked['M3Kills'].nll(d.X, d.y, d.e, cells=True)                 # (onsets, 2): the NLL of the blue and of the red kills
    values, match = scores['M3Kills'].per_row()
    assert len(values) == len(scores['M3Kills'].nll) == len(world.onsets('DEV')) and cells.shape == (len(values), 2)
    assert values == pytest.approx(cells.sum(axis=1), rel=1e-12)


def test_scores_means_are_match_weighted_and_finite(scores):
    for name, sc in scores.items():
        v, match = sc.per_row()
        assert sc.mean() == nll.weighted_mean(v, features.match_weights(match)) and np.isfinite(sc.mean()), name
        boot = sc.bootstrap(B=20)
        assert boot.shape == (20,) and np.isfinite(boot).all()


def test_the_zone_mean_passes_only_rows_with_a_zone(scores):
    zone = scores['M2Zone']
    v, match = zone.per_row()
    assert len(v) == len(zone.nll)                                       # no row without a category is in the mean


def test_a_row_without_exposure_cell_adds_nothing_to_the_m2_rate_score(fitted, heads_world):
    d = rows_of(heads_world, 'M2Rate', 'masked', role='DEV')
    cells = fitted.masked['M2Rate'].nll(d.X, d.y, d.e)
    assert np.isfinite(cells).all() and (d.e > 0).all()                  # cells without exposure are not rows at all, so nothing is added


def test_open_scores_exist_for_the_five_open_heads(fitted, heads_world):
    sc = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind='open')
    assert tuple(sc) == OPEN_NAMES and all(np.isfinite(s.nll).all() and len(s.nll) > 0 for s in sc.values())


def test_scoring_refuses_a_holdout_role(fitted, heads_world):
    with pytest.raises(ValueError):
        fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'TEST')


def test_scoring_does_not_change_the_heads(fitted, heads_world):
    before = {n: {k: v.copy() for k, v in h.state()[0].items()} for n, h in fitted.masked.items()}
    fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')
    for n, h in fitted.masked.items():
        after = h.state()[0]
        assert set(after) == set(before[n]) and all(np.array_equal(before[n][k], after[k]) for k in after), n


# =====================================================================================================================================
# B0Heads: save and load
# =====================================================================================================================================
def test_save_writes_an_npz_and_a_json_and_nothing_else(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    assert sorted(p.name for p in (tmp_path / 'b0').iterdir()) == ['b0_heads.json', 'b0_heads.npz']
    with np.load(tmp_path / 'b0' / 'b0_heads.npz', allow_pickle=False) as z:
        assert 'fit_means' in z.files and all(z[k].dtype.kind in 'fiub' for k in z.files)
    doc = json.loads((tmp_path / 'b0' / 'b0_heads.json').read_text(encoding='utf-8'))
    assert doc['schema'] == 'b0_heads_v3' and set(doc['heads']) == {'masked', 'open'}       # v3: the M2Rate windows of G-S3b B2-a (v2: R19)


def test_load_gives_back_the_same_heads_bit_for_bit(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    again = baselines_b0.B0Heads.load(tmp_path / 'b0')
    assert np.array_equal(again.fit_means, fitted.fit_means) and again.fit_roles == fitted.fit_roles
    assert tuple(again.masked) == tuple(fitted.masked) == HEAD_NAMES and tuple(again.open) == tuple(fitted.open) == OPEN_NAMES   # the order
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            other = getattr(again, kind)[name]
            assert type(other) is type(head) and other.kind == kind
            assert same_parameters(head, other), (kind, name)
            assert other.n_fit_ == head.n_fit_ and other.blocks_ == head.blocks_ and other.n_numeric_ == head.n_numeric_
    assert np.array_equal(again.predict_all(TOY_X), fitted.predict_all(TOY_X))
    assert again.summary() == fitted.summary()
    assert list(again.summary()['masked']) == list(fitted.summary()['masked']) == list(HEAD_NAMES)       # == on dicts ignores the order


def test_load_gives_the_heads_in_the_order_of_head_names_not_in_the_sorted_order_of_the_json(fitted, heads_world, tmp_path):
    """A resumed run writes its outputs in the order of a fresh run: ``masked``, ``open``, ``summary()`` and ``nll()`` follow HEAD_NAMES."""
    fitted.save(tmp_path / 'b0')
    doc = json.loads((tmp_path / 'b0' / 'b0_heads.json').read_text(encoding='utf-8'))
    assert list(doc['heads']['masked']) == sorted(HEAD_NAMES) != list(HEAD_NAMES)               # the file itself is sorted
    again = baselines_b0.B0Heads.load(tmp_path / 'b0')
    assert tuple(again.masked) == HEAD_NAMES and tuple(again.open) == OPEN_NAMES
    summary = again.summary()
    assert tuple(summary['masked']) == HEAD_NAMES and tuple(summary['open']) == OPEN_NAMES
    scores = again.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')
    assert tuple(scores) == HEAD_NAMES
    assert tuple(again.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind='open')) == OPEN_NAMES
    assert list(json.loads(json.dumps(fitted.summary()))['masked']) == list(HEAD_NAMES)       # and a fresh fit says the same


def test_the_constructor_puts_the_heads_in_the_order_of_head_names(fitted):
    shuffled = baselines_b0.B0Heads(dict(reversed(list(fitted.masked.items()))), dict(sorted(fitted.open.items())), fitted.fit_means)
    assert tuple(shuffled.masked) == HEAD_NAMES and tuple(shuffled.open) == OPEN_NAMES
    assert all(shuffled.masked[n] is fitted.masked[n] for n in HEAD_NAMES)


def test_the_constructor_refuses_a_head_name_that_does_not_exist_for_its_kind(fitted):
    with pytest.raises(ValueError, match='unknown head'):
        baselines_b0.B0Heads({**fitted.masked, 'M9Nope': fitted.masked['M2Rate']}, fitted.open, fitted.fit_means)
    with pytest.raises(ValueError, match='unknown head'):
        baselines_b0.B0Heads(fitted.masked, {**fitted.open, 'M2Rate': fitted.masked['M2Rate']}, fitted.fit_means)     # M2 has no open head


def test_the_saved_files_do_not_depend_on_the_time_of_saving(fitted, tmp_path):
    fitted.save(tmp_path / 'a')
    fitted.save(tmp_path / 'b')
    for name in ('b0_heads.json', 'b0_heads.npz'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_save_never_writes_over_a_saved_set(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    with pytest.raises(FileExistsError):
        fitted.save(tmp_path / 'b0')


def test_load_refuses_a_changed_array_file(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    path = tmp_path / 'b0' / 'b0_heads.npz'
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='sha256|changed|hash'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')


def test_load_refuses_a_missing_file_and_a_wrong_schema(fitted, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        baselines_b0.B0Heads.load(tmp_path / 'nothing')
    fitted.save(tmp_path / 'b0')
    doc = json.loads((tmp_path / 'b0' / 'b0_heads.json').read_text(encoding='utf-8'))
    doc['schema'] = 'something_else'
    (tmp_path / 'b0' / 'b0_heads.json').write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')


def test_load_refuses_a_design_that_is_not_the_layout_of_the_module_and_a_set_that_misses_a_head(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    path = tmp_path / 'b0' / 'b0_heads.json'
    original = json.loads(path.read_text(encoding='utf-8'))
    doc = json.loads(json.dumps(original))
    doc['heads']['masked']['M2Rate']['n_numeric'] = 38
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='layout'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')
    doc = json.loads(json.dumps(original))
    doc['heads']['masked']['M3Gold']['members']['red']['n_numeric'] = 38                          # the red model of the gold head
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='layout'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')
    doc = json.loads(json.dumps(original))
    del doc['heads']['open']['M3Obj']
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='five open heads'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')
    doc = json.loads(json.dumps(original))
    doc['heads']['open']['M3Obj']['class'] = 'M3Kills'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='names the class'):
        baselines_b0.B0Heads.load(tmp_path / 'b0')
    path.write_text(json.dumps(original), encoding='utf-8')
    assert baselines_b0.B0Heads.load(tmp_path / 'b0').summary() == fitted.summary()                # the untouched JSON still loads


def test_the_saved_json_names_the_design_columns_and_holds_no_key(fitted, tmp_path):
    fitted.save(tmp_path / 'b0')
    text = (tmp_path / 'b0' / 'b0_heads.json').read_text(encoding='utf-8')
    assert KEY_PREFIX not in text
    doc = json.loads(text)
    head = doc['heads']['masked']['M3Kills']
    assert head['n_numeric'] == 39 and len(head['columns']) == 120 and head['columns'][:39] == list(features.B0_NAMES)
    assert head['columns'][39] == 'q=1' and head['columns'][-1] == 's=3'                          # no team column (ruling R19)
    gold = doc['heads']['masked']['M3Gold']
    assert len(gold['columns']) == 130 and gold['columns'][-1] == 'layer=6' and 'team=1' not in gold['columns']
    assert set(gold['members']) == {'blue', 'red'} and all(m['class'] == 'M3Gold' for m in gold['members'].values())


def test_the_fitted_m2_rate_has_one_state_vector_for_each_window_and_the_saved_json_records_them(fitted, tmp_path):
    head = fitted.masked['M2Rate']
    assert head.coef_std_.shape == (1, 3 * 39 + 17) and head.n_numeric_ == 39 and head.mean_.shape == head.scale_.shape == (39,)
    summary = fitted.summary()['masked']['M2Rate']
    assert summary['width'] == 134 and summary['windows'] == WINDOWS_JSON and len(summary['n_iter']) == len(summary['converged']) == 1
    fitted.save(tmp_path / 'b0')
    doc = json.loads((tmp_path / 'b0' / 'b0_heads.json').read_text(encoding='utf-8'))
    saved = doc['heads']['masked']['M2Rate']
    assert saved['windows'] == WINDOWS_JSON and saved['width'] == 134 and saved['n_numeric'] == 39
    names = list(features.B0_NAMES)
    assert saved['columns'] == [f'w{j}:{n}' for j in range(3) for n in names] + [f'q={q}' for q in range(1, 18)]
    assert all('windows' not in doc['heads']['masked'][other] for other in HEAD_NAMES if other != 'M2Rate')
    again = baselines_b0.B0Heads.load(tmp_path / 'b0')
    x = np.asarray(TOY_X.row_view)
    assert np.array_equal(again.masked['M2Rate'].rate_bins(x), head.rate_bins(x))


def test_the_module_uses_no_pickle():
    text = SOURCE.read_text(encoding='utf-8')
    assert 'allow_pickle=True' not in text and 'import pickle' not in text and 'import joblib' not in text
    assert not {n for n in all_imports(SOURCE) if any(w in n for w in ('pickle', 'joblib', 'dill', 'shelve', 'marshal'))}
    calls = {node.func.attr for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert 'dump' not in calls and not calls & {'save', 'savez', 'savez_compressed'}       # arrays: _write_npz_bytes; text: json.dumps
    assert {'loads', 'dumps'} <= calls                                                        # and the text is JSON


# =====================================================================================================================================
# the composer heads (masked): ruling R3 and the rows of a bound set of views
# =====================================================================================================================================
@pytest.fixture(scope='module')
def chain(fitted, heads_world):
    h = fitted.composer_heads('masked')
    views = np.flatnonzero((np.asarray(heads_world.s2.array('views/src')) == 0) &
                           features.split_views(heads_world.s1, heads_world.s2, 'DEV'))
    h.bind(heads_world.s2, views)
    h.bound_views = views
    return h


def test_the_composer_heads_follow_the_protocol_of_ruling_r3(fitted):
    h = fitted.composer_heads('masked')
    assert isinstance(h, heads.ComposerHeads)
    assert h.feature_kind == 'b0' and h.analytic_ok is True and h.obj_uses_K is False and h.conv_uses_KO is False


def test_the_open_heads_are_scored_not_composed(fitted):
    with pytest.raises(ValueError, match='open'):
        fitted.composer_heads('open')
    with pytest.raises(ValueError):
        fitted.composer_heads('something')


def test_the_chain_refuses_every_call_before_bind(fitted):
    h = fitted.composer_heads('masked')
    with pytest.raises(RuntimeError, match='bind'):
        h.rate(np.array([0]))
    with pytest.raises(RuntimeError, match='bind'):
        h.row_view(np.array([0]))


def test_bind_computes_the_b0_of_the_views_in_order(chain, heads_world):
    expected = features.b0(heads_world.s2, chain.bound_views, heads_world.fit_means)
    assert np.array_equal(chain.row_view(np.arange(len(chain.bound_views))), expected)
    rows = np.array([3, 0, 3, 2])
    assert np.array_equal(chain.row_view(rows), expected[rows]) and chain.row_view(rows).dtype == np.float32


def test_bind_reads_only_the_three_x_arrays_a3(fitted, heads_world):
    seen = []
    real = Store.array
    h = fitted.composer_heads('masked')
    views = np.flatnonzero(np.asarray(heads_world.s2.array('views/src')) == 0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Store, 'array', lambda self, name: (seen.append(name), real(self, name))[1])
        h.bind(heads_world.s2, views)
        bound = list(seen)
        h.rate(np.arange(3))
        h.size_prob(np.arange(3), np.zeros(3, int), np.zeros(3, int), np.zeros(3, int))
    assert set(bound) == {'x/summary', 'x/players_num', 'x/geo_pp', 'views/src'}     # views/src: only frame-clock views are bound
    assert seen == bound                                                           # after bind the heads read no store at all


def test_bind_refuses_event_views_a3(fitted, heads_world):
    src = np.asarray(heads_world.s2.array('views/src'))
    h = fitted.composer_heads('masked')
    for views in (np.flatnonzero(src != 0), np.flatnonzero(src >= 0)[:60], src != 0):
        with pytest.raises(ValueError, match='frame-clock') as raised:
            h.bind(heads_world.s2, views)
        assert KEY_PREFIX not in str(raised.value)
    with pytest.raises(RuntimeError, match='bind'):
        h.rate(np.array([0]))                                                      # a refused bind binds nothing


def test_bind_accepts_a_mask_and_replaces_the_bound_views(fitted, heads_world):
    h = fitted.composer_heads('masked')
    mask = features.split_views(heads_world.s1, heads_world.s2, 'DEV') & (np.asarray(heads_world.s2.array('views/src')) == 0)
    h.bind(heads_world.s2, mask)
    n = int(mask.sum())
    assert h.row_view(np.arange(n)).shape == (n, 39)
    h.bind(heads_world.s2, np.flatnonzero(mask)[:2])
    assert h.row_view(np.arange(2)).shape == (2, 39)
    with pytest.raises(ValueError):
        h.row_view(np.array([2]))


def test_the_outputs_have_the_documented_shapes_and_laws(chain):
    n = 7
    rows = np.arange(n) % len(chain.bound_views)
    rng = np.random.default_rng(0)
    q, m, z, s = rng.integers(0, 18, n), rng.integers(0, 52, n), rng.integers(-1, 10, n), rng.integers(0, 4, n)
    layer = rng.integers(1, 7, (n, 2))
    K = np.column_stack([rng.integers(0, 5, (n, 2)), rng.integers(0, 2000, (n, 2))]).astype(float)
    out = {'rate': chain.rate(rows), 'mark_prob': chain.mark_prob(rows, q), 'zone_prob': chain.zone_prob(rows, q, m),
           'size_prob': chain.size_prob(rows, q, m, z), 'kill_rate': chain.kill_rate(rows, q, m, z, s),
           'gold_prob': chain.gold_prob(rows, q, m, z, s, layer), 'obj_rate': chain.obj_rate(rows, q, m, z, s, K)}
    out['conv_rate'] = chain.conv_rate(features.m4_masked_inputs(chain.row_view(rows), q, m, z, K, np.zeros((n, 18))))
    for name, array in out.items():
        assert array.shape == (n,) + heads.OUTPUT_WIDTHS[name] and array.dtype == np.float64 and np.isfinite(array).all(), name
    for name in ('mark_prob', 'zone_prob', 'size_prob', 'gold_prob'):
        floor = MIX / out[name].shape[-1] * (1 - 1e-9)
        assert np.allclose(out[name].sum(axis=-1), 1.0, atol=1e-12) and (out[name] > 0).all() and (out[name] >= floor).all()
    assert (out['rate'] >= 0).all() and (out['kill_rate'] > 0).all() and (out['obj_rate'] >= 0).all() and (out['conv_rate'] >= 0).all()


def one_output(fitted, key, b0_row, **values):
    """The output of the masked head ``key`` for ONE row, on a design built with ``row_vector`` (a second implementation of the layout)."""
    return fitted.masked[key[0]].predict(row_vector(b0_row, blocks_of(key, **values))[None, :])[0]


def one_gold(fitted, key, b0_row, team, **values):
    """The gold law of the model of ``team`` for ONE row (the gold head has a model for each team, ruling R19)."""
    return fitted.masked['M3Gold'].member(team).predict(row_vector(b0_row, blocks_of(key, **values))[None, :])[0]


def oracle_probe(b0, fitted, q, m, z, s, layer):
    """Every composer output evaluated row by row on designs built with ``row_vector``; ``b0`` is the (n, 39) matrix of row features."""
    b0 = np.asarray(b0, dtype=np.float64)
    n = range(len(b0))
    out = {}
    out['rate'] = np.array([[one_output(fitted, ('M2Rate', 'masked'), b0[i], q=qq) for qq in range(18)] for i in n])
    out['mark_prob'] = np.array([one_output(fitted, ('M2Mark', 'masked'), b0[i], q=q[i]) for i in n])
    out['zone_prob'] = np.array([one_output(fitted, ('M2Zone', 'masked'), b0[i], q=q[i], m=m[i]) for i in n])
    out['size_prob'] = np.array([one_output(fitted, ('M3Size', 'masked'), b0[i], q=q[i], m=m[i], z=z[i]) for i in n])
    out['kill_rate'] = np.array([one_output(fitted, ('M3Kills', 'masked'), b0[i], q=q[i], m=m[i], z=z[i], s=s[i]) for i in n])   # (n, 2)
    out['gold_prob'] = np.array([[one_gold(fitted, ('M3Gold', 'masked'), b0[i], t, q=q[i], m=m[i], z=z[i], s=s[i], layer=layer[i, t])
                                  for t in (BLUE, RED)] for i in n])
    out['obj_rate'] = np.array([one_output(fitted, ('M3Obj', 'masked'), b0[i], q=q[i], m=m[i], z=z[i], s=s[i]) for i in n])
    out['conv_rate'] = np.array([one_output(fitted, ('M4Conv', 'masked'), b0[i], q=q[i], m=m[i], z=z[i]) for i in n])
    return out


def test_the_chain_equals_the_heads_on_designs_built_row_by_row(chain, fitted):
    n = 9
    rows = np.array([0, 1, 2, 3, 0, 5, 4, 2, 1]) % len(chain.bound_views)
    rng = np.random.default_rng(1)
    q, m, z, s = rng.integers(0, 18, n), rng.integers(0, 52, n), rng.integers(-1, 10, n), rng.integers(0, 4, n)
    layer = rng.integers(1, 7, (n, 2))
    K = np.column_stack([rng.integers(0, 5, (n, 2)), rng.integers(0, 2000, (n, 2))]).astype(float)
    expected = oracle_probe(chain.row_view(rows), fitted, q, m, z, s, layer)
    got = {'rate': chain.rate(rows), 'mark_prob': chain.mark_prob(rows, q), 'zone_prob': chain.zone_prob(rows, q, m),
           'size_prob': chain.size_prob(rows, q, m, z), 'kill_rate': chain.kill_rate(rows, q, m, z, s),
           'gold_prob': chain.gold_prob(rows, q, m, z, s, layer), 'obj_rate': chain.obj_rate(rows, q, m, z, s, K),
           'conv_rate': chain.conv_rate(features.m4_masked_inputs(chain.row_view(rows), q, m, z, K, rng.integers(0, 3, (n, 18))))}
    for name in expected:
        assert got[name] == pytest.approx(expected[name], rel=1e-9, abs=1e-14), name


def test_obj_rate_does_not_read_k_and_conv_rate_does_not_read_k_or_o_p6(chain):
    n = 6
    rows = np.arange(n) % len(chain.bound_views)
    rng = np.random.default_rng(2)
    q, m, z, s = rng.integers(0, 18, n), rng.integers(0, 52, n), rng.integers(-1, 10, n), rng.integers(0, 4, n)
    K1, K2 = rng.integers(0, 5, (n, 4)).astype(float), rng.integers(5, 20, (n, 4)).astype(float) * 1000
    assert np.array_equal(chain.obj_rate(rows, q, m, z, s, K1), chain.obj_rate(rows, q, m, z, s, K2))
    O1, O2 = rng.integers(0, 3, (n, 18)).astype(float), rng.integers(3, 9, (n, 18)).astype(float)
    rv = chain.row_view(rows)
    first = chain.conv_rate(features.m4_masked_inputs(rv, q, m, z, K1, O1))
    assert np.array_equal(first, chain.conv_rate(features.m4_masked_inputs(rv, q, m, z, K2, O2)))


def test_the_kill_rate_ratio_of_the_teams_moves_with_the_row_the_bin_the_mark_the_zone_and_the_size_r19(chain):
    """One set of slopes for both teams and a team term gives the same ratio red / blue for every input (a probe of the first version, on
    this world, gave min = max = 0.980383377980). With a model for each team the log ratio moves with every input of the head."""
    n_rows = len(chain.bound_views)
    rows = np.arange(n_rows)
    zero = np.zeros(n_rows, dtype=int)

    def spread(k):
        return np.ptp(np.log(k[:, RED]) - np.log(k[:, BLUE]))
    assert spread(chain.kill_rate(rows, zero, zero, zero, zero)) > 0.05                              # the b0 state of the row
    ones = np.zeros(52, dtype=int)
    assert spread(chain.kill_rate(ones, ones, np.arange(52), ones, ones)) > 0.05                    # the mark
    ones = np.zeros(18, dtype=int)
    assert spread(chain.kill_rate(ones, np.arange(18), ones, ones, ones)) > 0.05                    # the bin
    ones = np.zeros(11, dtype=int)
    assert spread(chain.kill_rate(ones, ones, ones, np.arange(-1, 10), ones)) > 0.05                # the zone
    ones = np.zeros(4, dtype=int)
    assert spread(chain.kill_rate(ones, ones, ones, ones, np.arange(4))) > 0.05                     # the size class


def test_the_gold_laws_of_the_teams_differ_by_more_than_one_shift_of_the_logits_r19(chain):
    """With one set of slopes and a team term, log p_blue - log p_red is the same vector for every input up to a constant per input; with a
    model for each team the part that is not a per-input constant moves."""
    n_rows = len(chain.bound_views)
    rows = np.arange(n_rows)
    zero = np.zeros(n_rows, dtype=int)

    def moves(g):
        d = np.log(g[:, BLUE]) - np.log(g[:, RED])
        d -= d.mean(axis=1, keepdims=True)                                # the per-input constant (the normalisers) is no information
        return np.ptp(d, axis=0).max()
    layer = np.full((n_rows, 2), 3)
    assert moves(chain.gold_prob(rows, zero, zero, zero, zero, layer)) > 0.05                       # the b0 state of the row
    marks = np.zeros(52, dtype=int)
    assert moves(chain.gold_prob(marks, marks, np.arange(52), marks, marks, np.full((52, 2), 3))) > 0.05              # the mark
    layers = np.stack([np.arange(1, 7)] * 2, axis=1)
    six = np.zeros(6, dtype=int)
    assert moves(chain.gold_prob(six, six, six, six, six, layers)) > 0.05                                             # the kill layer


def test_the_gold_head_reads_the_layer_of_its_own_team(chain):
    rows = np.zeros(4, dtype=int)
    args = (np.zeros(4, int), np.zeros(4, int), np.zeros(4, int), np.zeros(4, int))
    base = chain.gold_prob(rows, *args, np.full((4, 2), 3))
    mixed = chain.gold_prob(rows, *args, np.array([[3, 1], [3, 6], [1, 3], [6, 3]]))
    assert np.array_equal(mixed[0, 0], base[0, 0]) and not np.allclose(mixed[0, 1], base[0, 1])
    assert np.array_equal(mixed[2, 1], base[2, 1]) and not np.allclose(mixed[2, 0], base[2, 0])


def test_the_chain_refuses_rows_and_categories_that_are_not_integers(chain):
    with pytest.raises(TypeError, match='rows'):
        chain.rate(np.array([0.5]))
    with pytest.raises(TypeError, match='q'):
        chain.mark_prob(np.array([0]), np.array([0.0]))
    with pytest.raises(TypeError, match='rows'):
        chain.rate(np.array([True]))


@pytest.mark.parametrize('call', [
    lambda h: h.rate(np.array([99])), lambda h: h.rate(np.array([-1])), lambda h: h.rate(np.zeros((2, 2), int)),
    lambda h: h.mark_prob(np.array([0]), np.array([18])), lambda h: h.mark_prob(np.array([0]), np.array([-1])),
    lambda h: h.zone_prob(np.array([0]), np.array([0]), np.array([52])),
    lambda h: h.size_prob(np.array([0]), np.array([0]), np.array([0]), np.array([10])),
    lambda h: h.size_prob(np.array([0]), np.array([0]), np.array([0]), np.array([-2])),
    lambda h: h.kill_rate(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([4])),
    lambda h: h.gold_prob(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([[0, 1]])),
    lambda h: h.gold_prob(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([[1, 7]])),
    lambda h: h.gold_prob(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([1, 2])),
    lambda h: h.mark_prob(np.array([0, 1]), np.array([0])), lambda h: h.conv_rate(np.zeros((1, 10)))])
def test_the_chain_refuses_categories_and_rows_out_of_range(chain, call):
    with pytest.raises(ValueError):
        call(chain)


def test_the_chain_is_deterministic_and_changes_no_head(chain, fitted):
    rows = np.arange(4)
    before = fitted.masked['M2Rate'].coef_.copy()
    a, b = chain.rate(rows), chain.rate(rows)
    assert np.array_equal(a, b) and np.array_equal(before, fitted.masked['M2Rate'].coef_)


def test_the_rate_of_a_row_is_the_rate_head_on_the_18_bins(chain, fitted):
    rate = chain.rate(np.array([0]))[0]
    b0 = chain.row_view(np.array([0])).astype(np.float64)[0]
    key = ('M2Rate', 'masked')
    again = [fitted.masked['M2Rate'].predict(row_vector(b0, blocks_of(key, q=q))[None, :])[0] for q in range(18)]
    assert rate == pytest.approx(again, rel=1e-10)


# =====================================================================================================================================
# predict_all
# =====================================================================================================================================
def test_predict_all_is_every_masked_head_on_the_rows_of_a_batch(fitted):
    out = fitted.predict_all(TOY_X)
    assert list(baselines_b0.PREDICT_ALL_WIDTHS) == ['rate', 'mark_prob', 'zone_prob', 'size_prob', 'kill_rate', 'gold_prob', 'obj_rate',
                                                     'conv_rate']
    assert baselines_b0.PREDICT_ALL_WIDTHS == {'rate': 18, 'mark_prob': 52, 'zone_prob': 10, 'size_prob': 4, 'kill_rate': 2,
                                               'gold_prob': 62, 'obj_rate': 18, 'conv_rate': 18}
    assert out.shape == (12, 184) and out.dtype == np.float64 and np.isfinite(out).all()


def test_predict_all_equals_the_heads_on_designs_built_row_by_row(fitted):
    b = TOY_X
    n = len(b.q)
    layer = np.clip(b.K[:, :2], 1, 6).astype(int)                       # the kill layer of each team: 1 .. 6 (a team with no kill is 1)
    expected = oracle_probe(b.row_view.astype(np.float32), fitted, b.q, b.m, b.z, b.s, layer)         # the rows are float32, as b0 is
    pieces = [expected['rate'], expected['mark_prob'], expected['zone_prob'], expected['size_prob'], expected['kill_rate'],
              expected['gold_prob'].reshape(n, -1), expected['obj_rate'], expected['conv_rate']]
    assert fitted.predict_all(b) == pytest.approx(np.hstack(pieces), rel=1e-9, abs=1e-14)


def test_a_batch_is_checked(fitted):
    b = TOY_X
    with pytest.raises(ValueError):
        fitted.predict_all(b._replace(q=b.q + 18))
    with pytest.raises(ValueError):
        fitted.predict_all(b._replace(row_view=b.row_view[:, :38]))
    with pytest.raises(TypeError):
        fitted.predict_all(b.row_view)


# =====================================================================================================================================
# structure
# =====================================================================================================================================
def test_the_module_imports_no_runner_no_cohort_no_raw_reader_no_value_at_module_level():
    imported = module_level_imports(SOURCE)
    forbidden = {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.cohort', 'c_ept.runlog', 'c_ept.terrain', 'c_ept.store_build', 'c_ept.reader',
                 'c_ept.tokens', 'c_ept.value', 'c_ept.compose', 'c_ept.units', 'c_ept.targets', 'c_ept.towertier'}
    forbidden |= {'c_ept.workflows.run_s1', 'c_ept.workflows.run_s2', 'c_ept.preprocessing.terrain',
                  'c_ept.data.store_build', 'c_ept.data.reader', 'c_ept.preprocessing.tokens',
                  'c_ept.valuation.value', 'c_ept.valuation.compose', 'c_ept.domain.units',
                  'c_ept.domain.targets', 'c_ept.valuation.towertier'}
    assert not imported & forbidden, imported & forbidden


ROWS_SOURCE = Path(head_rows.__file__)          # the rows of the heads are chosen in the shared module (ruling R18): both files are scanned


def test_the_module_names_only_the_arrays_it_is_documented_to_read():
    names = set()
    for source in (SOURCE, ROWS_SOURCE):
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names |= {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and node.value.count('/') == 1 and ' ' not in node.value and node.value.split('/')[0].isidentifier()}
    documented = {'matches/role', 'frame_rows/match', 'frame_rows/t0', 'frame_rows/exposure_ms', 'onsets/row', 'onsets/bin', 'onsets/unit',
                  'onsets/mark', 'units/match', 'units/mark', 'units/size', 'units/K', 'units/O', 'units/conv', 'units/conv_exposure_ms',
                  'units/in_scope', 'event_rows/unit', 'event_rows/kind', 'event_rows/ref', 'views/src', 'views/src_index', 'views/match',
                  'views/ref', 'labels/gold_bin', 'labels/kill_gold'}
    assert names <= documented, names - documented
    assert {'views/src', 'views/src_index', 'onsets/row', 'units/K', 'labels/gold_bin'} <= names           # the scan looks at the row code
    for source in (SOURCE, ROWS_SOURCE):
        text = source.read_text(encoding='utf-8')
        assert 'matches/key' not in text and 'allow_holdout' not in text


def test_the_module_does_not_read_the_kills_or_the_objectives_itself():
    for source in (SOURCE, ROWS_SOURCE):
        text = source.read_text(encoding='utf-8')
        for forbidden in ("'kills/", "'objectives/", "'frames/", "'units/f'", "'units/delta'", "'units/last'", "'units/kill0'"):
            assert forbidden not in text, forbidden                      # the zone comes from features.unit_zone


def test_a_wrong_role_and_a_bad_means_never_reach_a_key(heads_world):
    for call in (lambda: baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'TEST'),
                 lambda: baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, np.zeros(3), 'FIT')):
        with pytest.raises(ValueError) as raised:
            call()
        assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)


def test_the_logistic_heads_of_the_world_converge_and_say_so(heads_world, fitted):
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        d = rows_of(heads_world, 'M3Size', 'masked')
        baselines_b0.M3Size().fit(d.X, d.y, n_numeric=39)
    for name in ('M2Mark', 'M2Zone', 'M3Size', 'M3Gold'):
        assert fitted.masked[name].converged_.tolist() == [True] * (2 if name == 'M3Gold' else 1)          # M3Gold: a model for each team


def test_a_fit_that_hits_the_iteration_limit_warns_and_is_marked_not_converged(monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    monkeypatch.setattr(baselines_b0, 'MAX_ITER_POISSON', 2)
    monkeypatch.setattr(baselines_b0, 'MAX_ITER_LOGISTIC', 2)
    X, y, e = synthetic_poisson_with_exposure(seed=21, n=500)
    with pytest.warns(ConvergenceWarning):
        poisson = PlainRate().fit(X, y, e)
    assert poisson.converged_.tolist() == [False] and poisson.n_iter_.tolist() == [2]
    X, zone = toy_zone_data(n=300, missing=0, seed=5)
    with pytest.warns(ConvergenceWarning):
        logistic = baselines_b0.M2Zone().fit(X, zone)
    assert logistic.converged_.tolist() == [False]
    assert all(s['converged'] == [False] for s in (poisson.summary(), logistic.summary()))


# =====================================================================================================================================
# S4b preparation M11 (decision D16): B0 on the rows of a match subset (matches=), the per-entry NLL and the rows of the set
# =====================================================================================================================================
M11_SUBSET = [1, 3]                  # two of the four FIT matches of the heads world (as in tests/test_head_rows)


def test_b0_headrows_accepts_matches_subset(heads_world):
    """B0 is scored on exactly the matches= rows of the other models (``head_rows.HeadRows(..., matches=)``, B1 has it already): the design, the targets, the exposure, the match and the row of every
    entry of a subset are the entries of the full rows that belong to the chosen matches, head by head; the subset is proper and nonempty for every head."""
    args = (heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT')
    full, sub = baselines_b0.HeadRows(*args), baselines_b0.HeadRows(*args, matches=M11_SUBSET)
    assert sub.matches is not None and list(sub.matches) == M11_SUBSET
    for head, kind in baselines_b0.LAYOUTS:
        f, s = full.data(head, kind), sub.data(head, kind)
        mask = np.isin(f.match, M11_SUBSET)
        assert mask.any() and not mask.all() and len(s.match) == mask.sum(), (head, kind)
        for name in ('X', 'y', 'e', 'match', 'row', 'team', 'first_team'):
            a, b = getattr(f, name), getattr(s, name)
            if a is None or b is None:
                assert a is None and b is None, (head, kind, name)
            else:
                assert np.array_equal(np.asarray(a)[mask], np.asarray(b), equal_nan=True), (head, kind, name)


def test_b0_nll_on_a_match_subset_is_the_full_nll_of_those_matches(heads_world, fitted):
    """``B0Heads.nll(..., matches=)`` scores the rows of the chosen matches only, with the numbers of the full call."""
    for kind in ('masked', 'open'):
        full = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'FIT', kind)
        sub = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'FIT', kind, matches=M11_SUBSET)
        assert tuple(full) == tuple(sub)
        for name in full:
            mask = np.isin(full[name].match, M11_SUBSET)
            assert mask.any() and not mask.all(), (kind, name)
            for field in ('match', 'row'):
                assert np.array_equal(getattr(full[name], field)[mask], getattr(sub[name], field)), (kind, name, field)
            # the same heads on fewer rows: the linear predictor is a matrix product whose last bits can depend on the number of rows, so the scores agree to the rounding of float64, not bit for bit
            np.testing.assert_allclose(sub[name].nll, full[name].nll[mask], rtol=1e-12, atol=1e-12, err_msg=f'{kind} {name}')


def test_the_entry_scores_of_b0_cover_every_design_entry_and_equal_nll_on_the_scored_ones(heads_world, fitted):
    """``entry_nll(head, kind, d)``: one score for each design entry of ``d`` (a zone entry without a zone and a gold entry without a bin score 0, as the head says); the scores are those of ``nll``
    (M2Zone: nll leaves the entries without a zone out, they are the zeros here). ``head_rows`` builds the rows of the set with the condition and the means of the set."""
    rows = fitted.head_rows(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')
    assert isinstance(rows, baselines_b0.HeadRows) and rows.k_condition == fitted.k_condition and rows.kill_law == fitted.kill_law
    assert np.array_equal(rows.means, fitted.fit_means)
    for kind in ('masked', 'open'):
        scores = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind)
        for name in scores:
            d = rows.data(name, kind)
            entry = fitted.entry_nll(name, kind, d)
            assert entry.shape == (len(d.y),) and np.isfinite(entry).all(), (kind, name)
            if name == 'M2Zone':
                scored = d.y >= 0
                assert (~scored).any() and (entry[~scored] == 0).all()
                np.testing.assert_allclose(entry[scored], scores[name].nll, rtol=1e-12, atol=0, err_msg='M2Zone')       # ``nll`` scores the scored entries alone: another number of rows for the product
            else:
                assert np.array_equal(entry, scores[name].nll), (kind, name)


def test_the_saved_identity_of_a_b0_set_is_the_sha256_of_the_files_that_save_writes(fitted, tmp_path):
    """``saved_sha256()`` (S4b preparation M11 review, m4): the sha256 of the array file and of the JSON that ``save`` writes, computed without writing anything (the run provenance records it)."""
    import hashlib
    ident = fitted.saved_sha256()
    assert set(ident) == {'npz_sha256', 'json_sha256'}
    out = fitted.save(tmp_path / 'saved')                         # nothing was written before: save does not refuse
    assert ident['npz_sha256'] == hashlib.sha256((out / baselines_b0.NPZ_NAME).read_bytes()).hexdigest()
    assert ident['json_sha256'] == hashlib.sha256((out / baselines_b0.JSON_NAME).read_bytes()).hexdigest()
    assert fitted.saved_sha256() == ident and len(ident['npz_sha256']) == 64
