"""Tests of ``c_ept.baselines_b1``: the B1 LightGBM heads of M2, M3 and M4 (masked and open) and the B2 next-minute onset classifier
(stage S3 Task 7).

No run folder, raw file or catalog is read. The data are synthetic: designs made here with a known truth (the head classes), the "heads
world" of ``s3_toys`` (small S1, S2 and label stores with the real links between the tables; the rows and ``B1Heads``) and the "onset
toy" (a tiny S1 with the bins of its onsets written out by hand; the target of B2). Truth comes from five places:

    brief tests    the five tests of the task brief come first, word for word (their helpers are defined right below them)
    oracles        plain-Python loops over the arrays of the world, written from the text of the brief, that give the training rows of every
                   head (which view a row has, its categories and its K and O columns); the code is compared with them exactly
    planted truth  a known rate or law that the boosters must recover: the exposure is an offset, the teams have laws of their own (R19)
    rulings        R13 (every categorical output is (1 - 1e-6) p + 1e-6 / K over the full class set), R18 (B0 and B1 train on the
                   same rows), R19 (a model for each team), R21 (the signature of fit), R22 (left-out rows are counted, a cell without
                   events gets the Jeffreys rate), P6 (the composer flags), P9 (the fixed parameters and the early stopping on the STOP
                   rows)
    refusals       inputs that cannot be trusted: counts that are not whole numbers, a bad role, a changed file
"""
import ast
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from c_ept import baselines_b0, baselines_b1, features, head_rows, heads, nll
from c_ept import config as C
from c_ept.store_core import Store
from c_ept.tests.conftest import all_imports, module_level_imports
from c_ept.tests.s3_toys import (KEY_PREFIX, TOY_FIRST_ONSET_BIN, TOY_N_POSITIVE, TOY_N_STOP_ROWS,  # noqa: F401
                                 WORLD_ZERO_CELL, HeadsOracle, heads_world)
# the onset toy under the names of the brief's tests (``TOY_N_ROWS`` is the value toy's row count in ``s3_toys``): a fixture and a number
from c_ept.tests.s3_toys import TOY_ONSET_N_ROWS as TOY_N_ROWS  # noqa: F401
from c_ept.tests.s3_toys import onset_toy_s1 as toy_s1  # noqa: F401

lgb = baselines_b1.lgb
SOURCE = Path(baselines_b1.__file__)
HEAD_NAMES = ('M2Rate', 'M2Mark', 'M2Zone', 'M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
OPEN_NAMES = ('M3Size', 'M3Kills', 'M3Gold', 'M3Obj', 'M4Conv')
ALL_HEADS = [(name, 'masked') for name in HEAD_NAMES] + [(name, 'open') for name in OPEN_NAMES]
MIX = 1e-6
BLUE, RED = 0, 1
# small trees on small data: min_data_in_leaf and min_data_per_group are lowered so that the toys can be learnt at all; everything else
# (the objective, the exposure offset, the early stopping, the categorical columns) is the code of the real fit
SMALL = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 30, 'min_data_per_group': 5, 'cat_smooth': 1.0}


class Spy:
    def __init__(self):
        self.calls = []

    @property
    def count(self) -> int:
        return len(self.calls)


def spy(monkeypatch, module, name) -> Spy:
    """Replace ``module.name`` by a function that records its calls and calls the real one."""
    real = getattr(module, name)
    s = Spy()

    def wrapper(*args, **kwargs):
        s.calls.append((args, kwargs))
        return real(*args, **kwargs)
    monkeypatch.setattr(module, name, wrapper)
    return s


def spy_dataset(monkeypatch) -> list:
    """Record the design and the keyword arguments of every ``lightgbm.Dataset`` that the code makes (the training one, then the
    validation one)."""
    seen = []
    real = lgb.Dataset

    def dataset(data, *args, **kwargs):
        seen.append({'data': np.array(data), 'kwargs': dict(kwargs)})
        return real(data, *args, **kwargs)
    monkeypatch.setattr(lgb, 'Dataset', dataset)
    return seen


# =====================================================================================================================================
# the helpers of the task brief and the five tests of the task brief
# =====================================================================================================================================
def synthetic_with_exposure(seed=1, n=4000):
    """(X (n, 4) float32, counts (n,), exposure (n,)): x0, x1 numeric, x2 a 0/1 flag, x3 a category 0 .. 3 (a categorical column). The rate
    is exp(-3 + 0.5 x0 + 0.4 [x3 == 2]) per second and the counts are Poisson with the mean rate x exposure, exposure 2 .. 40 s."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(3.0, 2.0, n), rng.integers(0, 2, n), rng.integers(0, 4, n)]).astype(np.float32)
    e = rng.uniform(2.0, 40.0, n)
    y = rng.poisson(np.exp(-3.0 + 0.5 * X[:, 0] + 0.4 * (X[:, 3] == 2)) * e)
    return X, y, e


def one_row():
    return np.array([[0.2, 3.0, 1.0, 2.0]], dtype=np.float32)


def train_toy_b1(seed=1):
    X, y, e = synthetic_with_exposure(seed=1)
    return baselines_b1.M2Rate(params={**SMALL, 'seed': seed}).fit(X, y, e, categorical=(3,))


TOY_X = np.column_stack([np.linspace(-1.5, 1.5, 7), np.linspace(0.0, 6.0, 7), np.arange(7) % 2, np.arange(7) % 4]).astype(np.float32)

_WORLD = None


@pytest.fixture(scope='module', autouse=True)
def _the_world_of_the_brief_helpers(heads_world):
    """``fit_toy_b1_m4`` is a plain function in the brief; it reads the module's heads world."""
    global _WORLD
    _WORLD = heads_world
    yield
    _WORLD = None


def dropped_of(w):
    return features.dropped_columns(w.s2, w.fit_views)


def fit_toy_b1_m4(masked=True):
    """The M4 conversion head (masked or open) fitted on the FIT matches of the heads world with the STOP matches for the early stopping."""
    w = _WORLD
    dropped = dropped_of(w)
    fit_rows = baselines_b1.HeadRows(w.s1, w.s2, w.labels, dropped, 'FIT')
    stop_rows = baselines_b1.HeadRows(w.s1, w.s2, w.labels, dropped, 'STOP')
    return baselines_b1.fit_head(fit_rows, stop_rows, 'M4Conv', 'masked' if masked else 'open', params=SMALL)


def test_poisson_offset_is_used():
    m = baselines_b1.M2Rate(params=SMALL).fit(*synthetic_with_exposure(seed=1))
    x = one_row()
    assert m.expected(x, exposure_s=20.0) == pytest.approx(2 * m.expected(x, exposure_s=10.0), rel=1e-6)


def test_b1_bit_reproducible():
    assert np.array_equal(train_toy_b1(seed=1).predict(TOY_X), train_toy_b1(seed=1).predict(TOY_X))


def test_b2_target_is_onset_within_60s(toy_s1):
    y = baselines_b1.b2_target(toy_s1, rows=np.arange(TOY_N_ROWS))
    assert np.array_equal(y, TOY_FIRST_ONSET_BIN < 6)


def test_early_stopping_uses_stop_rows_only(toy_s1):
    assert baselines_b1.valid_rows(toy_s1).sum() == TOY_N_STOP_ROWS


def test_masked_m4_training_uses_the_input_builder(monkeypatch):
    calls = spy(monkeypatch, features, 'm4_masked_inputs')
    fit_toy_b1_m4(masked=True)
    assert calls.count > 0


# =====================================================================================================================================
# the input builder of the masked M4 head: its output IS the training design (the deferred point of the Task 6 review)
# =====================================================================================================================================
MARK_FEAT, MARK_K, MARK_O = -123.0, -456.0, -789.0


def mark_the_input_builder(monkeypatch):
    """Make ``features.m4_masked_inputs`` return its matrix with three marks: one cell of the b1 columns, one of the kills of the blue team
    and one of the objectives. The marks are values that no real column holds."""
    real = features.m4_masked_inputs

    def marked(*args, **kwargs):
        out = real(*args, **kwargs).copy()
        d = out.shape[1] - len(features.M4_EXTRA_NAMES) - features.N_O
        out[:, 5] = MARK_FEAT
        out[:, d + 3] = MARK_K
        out[:, d + 7] = MARK_O
        return out
    monkeypatch.setattr(features, 'm4_masked_inputs', marked)


def test_the_marks_of_the_input_builder_reach_the_design_that_the_masked_m4_booster_is_trained_on(heads_world, monkeypatch):
    mark_the_input_builder(monkeypatch)
    seen = spy_dataset(monkeypatch)
    fit_toy_b1_m4(masked=True)
    d = len(features.b1_columns(dropped_of(heads_world)))
    train, valid = seen[0]['data'], seen[1]['data']                         # the Dataset of the FIT rows, then that of the STOP rows
    for X in (train, valid):
        assert X.shape[1] == d + 25 and (X[:, 5] == MARK_FEAT).all() and (X[:, d + 3] == MARK_K).all() and (X[:, d + 7] == MARK_O).all()


def test_the_marks_of_the_input_builder_reach_the_masked_m4_rows(heads_world, monkeypatch):
    mark_the_input_builder(monkeypatch)
    dropped = dropped_of(heads_world)
    d = len(features.b1_columns(dropped))
    X = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT').data('M4Conv', 'masked').X
    assert (X[:, 5] == MARK_FEAT).all() and (X[:, d + 3] == MARK_K).all() and (X[:, d + 7] == MARK_O).all()


def test_b0_does_not_read_the_k_and_o_columns_of_the_masked_m4_input_but_b1_does(heads_world, monkeypatch):
    """Decision P6: the masked M4 head of B0 is conditioned on [b0, q, m, z] only; the one of B1 on the whole matrix."""
    mark_the_input_builder(monkeypatch)
    x0 = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT').data('M4Conv', 'masked').X
    assert not np.isin(x0, (MARK_K, MARK_O)).any()
    dropped = dropped_of(heads_world)
    x1 = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT').data('M4Conv', 'masked').X
    assert np.isin(x1, (MARK_K, MARK_O)).any()


def test_the_masked_m4_design_is_the_builder_matrix_with_the_zone_shifted_by_one(heads_world):
    dropped = dropped_of(heads_world)
    rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    p = rows.parts('M4Conv', 'masked')
    d = len(features.b1_columns(dropped))
    X = rows.data('M4Conv', 'masked').X
    assert X.dtype == np.float32 and X.shape == p.m4.shape == (len(p.y), d + 25)
    assert np.array_equal(X[:, :d + 2], p.m4[:, :d + 2], equal_nan=True) and np.array_equal(X[:, d + 3:], p.m4[:, d + 3:])
    assert np.array_equal(X[:, d + 2], p.codes['z'] + 1) and X[:, d + 2].min() == 0           # "no position" is the level 0


# =====================================================================================================================================
# the fixed parameters of decision P9
# =====================================================================================================================================
def test_the_fixed_parameters_are_those_of_decision_p9():
    assert baselines_b1.FIXED_PARAMS == {'num_leaves': 63, 'learning_rate': 0.1, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
                                         'bagging_fraction': 1.0, 'lambda_l2': 1.0, 'max_bin': 255, 'deterministic': True,
                                         'force_row_wise': True, 'num_threads': 8, 'seed': 1, 'verbosity': -1}
    assert (baselines_b1.MAX_ROUNDS, baselines_b1.EARLY_STOP) == (1000, 50)


def capture_training(monkeypatch) -> dict:
    """Record the arguments of every ``lightgbm.train`` and ``lightgbm.early_stopping`` call (and let them run)."""
    seen = {'train': [], 'stop': []}
    real_train, real_stop = lgb.train, lgb.early_stopping

    def train(params, train_set, num_boost_round=100, **kwargs):
        seen['train'].append((dict(params), num_boost_round, dict(kwargs)))
        return real_train(params, train_set, num_boost_round=num_boost_round, **kwargs)

    def early_stopping(stopping_rounds, **kwargs):
        seen['stop'].append((stopping_rounds, dict(kwargs)))
        return real_stop(stopping_rounds, **kwargs)
    monkeypatch.setattr(lgb, 'train', train)
    monkeypatch.setattr(lgb, 'early_stopping', early_stopping)
    return seen


def test_a_poisson_head_with_the_default_parameters_trains_with_the_fixed_ones_for_1000_rounds_and_50_of_patience(monkeypatch):
    seen = capture_training(monkeypatch)
    X, y, e = synthetic_with_exposure(seed=3, n=3000)
    baselines_b1.M2Rate().fit(X[:2000], y[:2000], e[:2000], categorical=(3,), valid=(X[2000:], y[2000:], e[2000:]))
    (params, rounds, kwargs), = seen['train']
    assert {k: v for k, v in params.items() if k not in ('objective', 'metric')} == baselines_b1.FIXED_PARAMS
    assert (params['objective'], params['metric'], rounds) == ('poisson', 'poisson', 1000)
    assert seen['stop'] == [(50, {'first_metric_only': True, 'verbose': False})] and len(kwargs['valid_sets']) == 1


def test_a_categorical_head_trains_a_multiclass_model_over_the_classes_that_occur(monkeypatch):
    seen = capture_training(monkeypatch)
    X, y = planted_classes(seed=3, n=1500)
    baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,), valid=(X[:200], y[:200]))
    (params, rounds, _), = seen['train']
    # 3 of the 4 classes of M3Size occur
    assert (params['objective'], params['metric'], params['num_class']) == ('multiclass', 'multi_logloss', 3)


def test_b2_trains_a_binary_model(monkeypatch):
    seen = capture_training(monkeypatch)
    X, y, _ = synthetic_with_exposure(seed=3, n=1500)
    baselines_b1.B2(params=SMALL).fit(X, y > 0)
    (params, _, _), = seen['train']
    assert (params['objective'], params['metric']) == ('binary', 'binary_logloss') and 'num_class' not in params


def test_params_replace_the_fixed_ones_and_the_ones_that_the_head_sets_are_refused():
    head = baselines_b1.M2Rate(params={'num_leaves': 5, 'num_iterations': 7, 'early_stopping_round': 3, 'seed': 9})
    assert (head.params['num_leaves'], head.params['seed'], head.params['learning_rate'], head.rounds, head.patience) == (5, 9, 0.1, 7, 3)
    for key in ('objective', 'metric', 'num_class', 'init_score', 'categorical_feature'):
        with pytest.raises(ValueError, match=key):
            baselines_b1.M2Rate(params={key: 'x'})
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            baselines_b1.M2Rate(params={'num_iterations': bad})
    with pytest.raises(TypeError):
        baselines_b1.M2Rate(params=['num_leaves'])
    with pytest.raises(ValueError):
        baselines_b1.M2Rate('both')


# =====================================================================================================================================
# the Poisson heads: the exposure is the offset
# =====================================================================================================================================
def test_the_init_score_is_the_log_exposure_plus_the_average_log_rate_of_the_cell(monkeypatch):
    seen = spy_dataset(monkeypatch)
    X, y, e = synthetic_with_exposure(seed=4, n=2500)
    Xv, yv, ev = synthetic_with_exposure(seed=5, n=800)
    baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,), valid=(Xv, yv, ev))
    train, valid = seen
    base = np.log(y.sum() / e.sum())
    assert np.array_equal(train['kwargs']['init_score'], np.log(e) + base)
    assert np.array_equal(valid['kwargs']['init_score'], np.log(ev) + base)           # the validation loss is the one of the same offset
    assert np.array_equal(np.asarray(train['kwargs']['label']), y) and np.array_equal(np.asarray(valid['kwargs']['label']), yv)
    assert train['kwargs']['categorical_feature'] == [3] and valid['kwargs']['reference'] is not None


def test_the_rate_is_per_second_whatever_the_exposure_of_the_rows():
    """Half of the rows are watched for 3 s and half for 60 s; the rate is the same. A booster that fitted the counts without the offset
    would be off by the average exposure (about 30 times)."""
    rng = np.random.default_rng(11)
    n = 20_000
    x0 = rng.normal(size=n)
    e = np.where(rng.random(n) < 0.5, rng.uniform(2, 4, n), rng.uniform(40, 80, n))
    truth = lambda x: np.exp(-3.5 + 0.8 * x)                                                      # noqa: E731
    y = rng.poisson(truth(x0) * e)
    m = baselines_b1.M2Rate(params={**SMALL, 'num_leaves': 15, 'num_iterations': 100}).fit(np.column_stack([x0]).astype(np.float32), y, e)
    grid = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    assert m.predict(grid) == pytest.approx(truth(grid[:, 0]), rel=0.15)


def test_the_expected_counts_add_up_to_the_observed_counts_of_fresh_rows_that_were_watched_for_another_time():
    """Calibration in the large: fitted on exposures of 2 .. 40 s, the expected counts of fresh rows that were watched three times as long
    add up to the counts of those rows (the offset makes the output a rate, not a count)."""
    X, y, e = synthetic_with_exposure(seed=11, n=8000)
    m = baselines_b1.M2Rate(params={**SMALL, 'num_leaves': 15, 'num_iterations': 60}).fit(X, y, e, categorical=(3,))
    Xf, _, ef = synthetic_with_exposure(seed=12, n=8000)
    ef = 3.0 * ef
    yf = np.random.default_rng(3).poisson(np.exp(-3.0 + 0.5 * Xf[:, 0] + 0.4 * (Xf[:, 3] == 2)) * ef)
    assert m.expected(Xf, ef).sum() == pytest.approx(yf.sum(), rel=0.06)
    assert m.expected(X, e).sum() == pytest.approx(y.sum(), rel=0.03)


def test_the_expected_count_is_the_rate_times_the_exposure_and_a_bad_exposure_is_refused():
    m = baselines_b1.M2Rate(params=SMALL).fit(*synthetic_with_exposure(seed=2), categorical=(3,))
    x = one_row()
    assert m.expected(x, exposure_s=10.0) == pytest.approx(m.predict(x) * 10.0, rel=1e-12)
    assert m.expected(x, exposure_s=0.0) == 0.0
    assert m.expected(TOY_X, np.arange(1, 8)).tolist() == (m.predict(TOY_X) * np.arange(1, 8)).tolist()
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            m.expected(x, exposure_s=bad)


def test_rows_without_exposure_are_left_out_of_the_fit_and_counted_and_a_count_in_one_is_refused():
    X, y, e = synthetic_with_exposure(seed=2, n=1500)
    y = y.copy()
    e = e.copy()
    e[:30] = 0.0
    y[:30] = 0
    m = baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))
    assert m.n_zero_exposure_ == 30 and m.n_fit_ == 1470
    base = np.log(y.sum() / e.sum())                                                   # the zero-exposure rows weigh nothing
    assert m.base_[0] == pytest.approx(base, rel=1e-12)
    y[0] = 1
    with pytest.raises(ValueError, match='count but no exposure'):
        baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))


def test_the_nll_is_the_poisson_nll_of_the_shared_module_and_a_cell_without_exposure_adds_nothing():
    X, y, e = synthetic_with_exposure(seed=2, n=1500)
    m = baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))
    got = m.nll(X, y, e)
    assert np.array_equal(got, nll.poisson(y, m.predict(X), e))
    e0 = e.copy()
    y0 = y.copy()
    e0[:5] = 0.0
    y0[:5] = 0
    got0 = m.nll(X, y0, e0)
    assert (got0[:5] == 0).all() and np.array_equal(got0[5:], got[5:])                  # the rate of an unobserved cell is not even read


def test_the_nll_of_many_cells_is_the_sum_of_the_cells():
    X, y, e = synthetic_with_exposure(seed=2, n=1200)
    Y = np.column_stack([y, y % 2, (y > 1).astype(int)])
    m = baselines_b1.M3Obj(params=SMALL).fit(X, Y, e, categorical=(3,))
    cells = m.nll(X, Y, e, cells=True)
    assert cells.shape == (1200, 3) and np.array_equal(m.nll(X, Y, e), cells.sum(axis=1))


def test_a_cell_without_any_event_has_no_booster_and_gets_the_jeffreys_rate():
    X, y, e = synthetic_with_exposure(seed=6, n=1500)
    Y = np.column_stack([y, np.zeros_like(y), y])                                       # the middle cell never happens
    m = baselines_b1.M3Obj(params=SMALL).fit(X, Y, e, categorical=(3,))
    assert m.degenerate_.tolist() == [False, True, False] and m.boosters_[1] is None and m.n_events_.tolist() == [y.sum(), 0, y.sum()]
    assert m.predict(X)[:, 1] == pytest.approx(np.full(1500, 0.5 / e.sum()), rel=1e-12)       # the posterior mean under the Jeffreys prior
    s = m.summary()
    assert s['degenerate'] == [False, True, False] and s['num_trees'][1] == 0 and s['num_trees'][0] > 0
    assert np.isfinite(m.nll(X, Y, e)).all()


def test_the_cells_of_a_head_are_independent_boosters_on_one_binned_matrix(monkeypatch):
    seen = spy_dataset(monkeypatch)
    X, y, e = synthetic_with_exposure(seed=7, n=2500)
    rng = np.random.default_rng(3)
    Y = np.column_stack([y, rng.poisson(0.02 * e), rng.poisson(np.exp(-4 + 0.7 * X[:, 1] / 3) * e)])
    Xv, yv, ev = synthetic_with_exposure(seed=8, n=700)
    Yv = np.column_stack([yv, rng.poisson(0.02 * ev), rng.poisson(np.exp(-4 + 0.7 * Xv[:, 1] / 3) * ev)])
    m = baselines_b1.M3Obj(params=SMALL).fit(X, Y, e, categorical=(3,), valid=(Xv, Yv, ev))
    assert len(seen) == 2                                       # one Dataset for the FIT rows and one for the STOP rows
    for c in range(3):                                          # ... and each cell is what a head of its own would give
        single = baselines_b1.M4Conv(params=SMALL).fit(X, Y[:, c], e, categorical=(3,), valid=(Xv, Yv[:, c], ev))
        assert single.boosters_[0].model_to_string() == m.boosters_[c].model_to_string()
    assert len({b.model_to_string() for b in m.boosters_}) == 3


def test_a_one_dimensional_target_gives_a_vector_and_a_one_column_target_a_matrix():
    X, y, e = synthetic_with_exposure(seed=2, n=1000)
    a = baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))
    b = baselines_b1.M2Rate(params=SMALL).fit(X, y.reshape(-1, 1), e, categorical=(3,))
    assert a.predict(X).shape == (1000,) and b.predict(X).shape == (1000, 1) and np.array_equal(a.predict(X), b.predict(X)[:, 0])


def test_the_kill_head_has_a_booster_for_each_team_with_laws_of_its_own_r19():
    """The blue team's kills rise with the state x0 and the red team's fall: the ratio of the two rates is not one number for every
    input."""
    rng = np.random.default_rng(5)
    n = 20_000
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 3, n)]).astype(np.float32)
    Y = np.column_stack([rng.poisson(np.exp(-0.5 + 0.8 * X[:, 0])), rng.poisson(np.exp(-0.5 - 0.8 * X[:, 0]))])
    m = baselines_b1.M3Kills(params={**SMALL, 'num_leaves': 15, 'num_iterations': 80}).fit(X, Y, np.ones(n), categorical=(1,))
    assert len(m.boosters_) == 2 and m.predict(X[:3]).shape == (3, 2)
    low, high = m.predict(np.array([[-1.0, 0], [1.0, 0]], dtype=np.float32))
    assert high[0] / high[1] > 3.0 and low[0] / low[1] < 1 / 3.0                      # truth: exp(1.6) = 4.95 and exp(-1.6) = 0.20
    assert m.boosters_[0].model_to_string() != m.boosters_[1].model_to_string()


def test_a_poisson_head_refuses_data_that_cannot_be_trusted():
    X, y, e = synthetic_with_exposure(seed=2, n=500)
    bad_x = X.copy()
    bad_x[0, 0] = np.inf
    for what, args in (('count_float', (X, y + 0.5, e)), ('count_negative', (X, y - 5, e)), ('x_inf', (bad_x, y, e)),
                       ('exposure_negative', (X, y, -e)), ('exposure_nan', (X, y, np.where(np.arange(500) == 0, np.nan, e))),
                       ('shape', (X, y[:-1], e)), ('exposure_shape', (X, y, e[:-1])), ('no_exposure', (X, y * 0, e * 0))):
        with pytest.raises(ValueError):
            baselines_b1.M2Rate(params=SMALL).fit(*args)
    with pytest.raises(TypeError):
        baselines_b1.M2Rate(params=SMALL).fit(X.astype(str), y, e)
    with pytest.raises(TypeError):
        baselines_b1.M2Rate(params=SMALL).fit(X, y.astype(str), e)
    with pytest.raises(ValueError, match='categorical column 9'):
        baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(9,))
    with pytest.raises(ValueError, match='whole numbers'):
        baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(0,))                 # x0 is a number with decimals


def test_a_categorical_column_must_hold_whole_numbers_from_zero_up_each_condition_on_its_own():
    X, y, e = synthetic_with_exposure(seed=2, n=500)
    baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))                          # the category 0 .. 3 as it is: fine
    missing = np.where(np.arange(500) == 3, np.nan, X[:, 3])
    for what, column in (('decimals', X[:, 3] + 0.5), ('negative', X[:, 3] - 1), ('missing', missing)):
        bad = X.copy()
        bad[:, 3] = column
        with pytest.raises(ValueError, match='whole numbers from 0 up'):
            baselines_b1.M2Rate(params=SMALL).fit(bad, y, e, categorical=(3,))
    with pytest.raises(ValueError, match='distinct'):
        baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3, 3))


def test_a_head_that_is_not_fitted_cannot_predict_and_a_design_of_another_width_is_refused():
    with pytest.raises(RuntimeError, match='not fitted'):
        baselines_b1.M2Rate().predict(one_row())
    m = baselines_b1.M2Rate(params=SMALL).fit(*synthetic_with_exposure(seed=2, n=800), categorical=(3,))
    with pytest.raises(ValueError, match='4 columns'):
        m.predict(one_row()[:, :3])


def test_the_validation_rows_of_a_poisson_head_must_fit_the_design():
    X, y, e = synthetic_with_exposure(seed=2, n=500)
    for valid in ((X[:50], y[:50]), (X[:50, :3], y[:50], e[:50]), (X[:50], np.column_stack([y[:50], y[:50]]), e[:50])):
        with pytest.raises(ValueError):
            baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,), valid=valid)


# =====================================================================================================================================
# early stopping on the STOP rows
# =====================================================================================================================================
EARLY = {'num_leaves': 15, 'min_data_in_leaf': 5, 'num_iterations': 200, 'early_stopping_round': 10, 'learning_rate': 0.3}


def noise_rows(seed, n):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    e = rng.uniform(5, 30, n)
    return X, rng.poisson(0.03 * e), e                                                 # the counts do not depend on the features


def test_early_stopping_keeps_the_rounds_up_to_the_best_loss_on_the_stop_rows():
    X, y, e = noise_rows(1, 1500)
    Xv, yv, ev = noise_rows(2, 1500)
    m = baselines_b1.M2Rate(params=EARLY).fit(X, y, e, valid=(Xv, yv, ev))
    info = m.info_[0]
    assert info['stopped_early'] and info['best_iteration'] < info['rounds_run'] < 200
    assert info['rounds_run'] - info['best_iteration'] == 10 and m.boosters_[0].num_trees() == info['best_iteration'] == info['num_trees']
    # the loss of every round on the STOP rows, from a model that was not stopped: the kept rounds are the first minimum of it
    full = baselines_b1.M2Rate(params={**EARLY, 'early_stopping_round': 1000}).fit(X, y, e)
    assert full.info_[0]['best_iteration'] == 200 and not full.info_[0]['stopped_early']
    base = full.base_[0]
    loss = []
    for k in range(1, info['rounds_run'] + 1):
        raw = full.boosters_[0].predict(Xv, raw_score=True, num_iteration=k) + base + np.log(ev)
        loss.append(np.mean(np.exp(raw) - yv * raw))                                   # the 'poisson' metric of LightGBM
    assert int(np.argmin(loss)) + 1 == info['best_iteration']


def test_without_stop_rows_every_round_is_run_and_kept():
    X, y, e = noise_rows(1, 800)
    m = baselines_b1.M2Rate(params={**EARLY, 'num_iterations': 25}).fit(X, y, e)
    assert m.info_[0] == {'best_iteration': 25, 'num_trees': 25, 'rounds_run': 25, 'stopped_early': False, 'best_score': None}


def test_the_stop_rows_choose_the_number_of_rounds_and_change_no_tree():
    X, y, e = noise_rows(1, 1500)
    Xv, yv, ev = noise_rows(2, 1500)
    stopped = baselines_b1.M2Rate(params=EARLY).fit(X, y, e, valid=(Xv, yv, ev))
    full = baselines_b1.M2Rate(params={**EARLY, 'early_stopping_round': 1000}).fit(X, y, e)
    k = stopped.info_[0]['best_iteration']
    assert k < 200
    a = stopped.boosters_[0].predict(Xv, raw_score=True)
    b = full.boosters_[0].predict(Xv, raw_score=True, num_iteration=k)
    assert np.array_equal(a, b)                                  # the same trees: the STOP rows only cut the sequence
    other = noise_rows(3, 1500)
    again = baselines_b1.M2Rate(params=EARLY).fit(X, y, e, valid=other)
    kk = again.info_[0]['best_iteration']
    n = min(k, kk)
    assert np.array_equal(stopped.boosters_[0].predict(Xv, raw_score=True, num_iteration=n),
                          again.boosters_[0].predict(Xv, raw_score=True, num_iteration=n))


def test_the_summary_gives_the_rounds_kept_the_trees_and_the_rounds_run_of_every_model():
    X, y, e = noise_rows(1, 1200)
    Xv, yv, ev = noise_rows(2, 600)
    Y = np.column_stack([y, rng_counts(4, e), rng_counts(5, e)])
    Yv = np.column_stack([yv, rng_counts(6, ev), rng_counts(7, ev)])
    m = baselines_b1.M3Obj(params=EARLY).fit(X, Y, e, valid=(Xv, Yv, ev))
    s = m.summary()
    assert len(s['best_iteration']) == len(s['num_trees']) == len(s['rounds_run']) == len(s['stopped_early']) == 3
    assert s['best_iteration'] == s['num_trees'] == [i['best_iteration'] for i in m.info_]
    assert all(0 < k <= 200 for k in s['best_iteration'])
    assert all(r >= k for r, k in zip(s['rounds_run'], s['best_iteration']))
    json.dumps(s, allow_nan=False)


def rng_counts(seed, e):
    return np.random.default_rng(seed).poisson(0.02 * e)


# =====================================================================================================================================
# the multiclass heads
# =====================================================================================================================================
TRUTH = np.array([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.10, 0.20, 0.70]])           # law of the class given the category of column 2


def planted_classes(seed=0, n=6000):
    """(X (n, 3), y (n,)): x0, x1 noise and the category g of column 2; the class (0, 1 or 2) has the law TRUTH[g]."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    y = np.array([rng.choice(3, p=TRUTH[k]) for k in g])
    return np.column_stack([rng.normal(size=n), rng.normal(size=n), g]).astype(np.float32), y


def test_a_categorical_head_recovers_a_planted_law():
    """On fresh rows the mean of the predicted law of the rows of a category is the planted law of that category (the two noise columns are
    there to be ignored)."""
    X, y = planted_classes(seed=1, n=9000)
    Xt, _ = planted_classes(seed=2, n=6000)
    m = baselines_b1.M3Size(params={**SMALL, 'min_data_in_leaf': 100, 'num_iterations': 40}).fit(X, y, categorical=(2,))
    p = m.predict(Xt)
    for g in range(3):
        assert p[Xt[:, 2] == g][:, :3].mean(axis=0) == pytest.approx(TRUTH[g], abs=0.03)


def test_a_categorical_output_is_full_width_positive_and_adds_up_to_one_with_the_mixing_of_r13():
    X, y = planted_classes(seed=1, n=2000)
    m = baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,))
    p = m.predict(X)
    assert p.shape == (2000, 4) and (p > 0).all() and p.sum(axis=1) == pytest.approx(np.ones(2000), abs=1e-12)
    assert (p[:, 3] == MIX / 4).all()                                                  # the class 3 never occurred: exactly 1e-6 / K
    assert (p[:, :3] >= MIX / 4).all()


def test_ruling_r13_every_categorical_head_has_the_full_class_set():
    X, y = planted_classes(seed=1, n=1500)
    for cls, k in ((baselines_b1.M2Mark, 52), (baselines_b1.M2Zone, 10), (baselines_b1.M3Size, 4), (baselines_b1.M3Gold, 31)):
        assert cls.N_CLASSES == k
    m = baselines_b1.M2Mark(params=SMALL).fit(X, y, categorical=(2,))
    p = m.predict(X[:5])
    assert p.shape == (5, 52) and (p[:, 3:] == MIX / 52).all() and (p > 0).all()


def test_a_head_with_one_class_in_training_is_a_constant_mixed_model_without_a_booster():
    X, y = planted_classes(seed=1, n=600)
    m = baselines_b1.M3Size(params=SMALL).fit(X, np.full(600, 2), categorical=(2,))
    p = m.predict(X)
    assert m.boosters_ == [None] and m.summary()['num_trees'] == [0] and (p[:, 2] == (1 - MIX) + MIX / 4).all()
    assert (p[:, [0, 1, 3]] == MIX / 4).all()


def test_two_classes_in_training_make_a_two_class_model():
    X, y = planted_classes(seed=1, n=2000)
    y = np.where(y == 0, 1, 3)
    m = baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,))
    assert m.classes_.tolist() == [1, 3] and m.boosters_[0].num_model_per_iteration() == 2
    p = m.predict(X[:4])
    assert (p[:, [0, 2]] == MIX / 4).all() and p.sum(axis=1) == pytest.approx(np.ones(4), abs=1e-12)


def test_a_class_minus_one_is_left_out_of_the_fit_and_counted_and_scores_zero():
    X, y = planted_classes(seed=2, n=1000)
    y = y.copy()
    y[:7] = -1
    m = baselines_b1.M2Zone(params=SMALL).fit(X, y, categorical=(2,))
    assert m.n_missing_ == 7 and m.n_fit_ == 993 and m.summary()['n_missing'] == 7
    s = m.nll(X, y)
    assert (s[:7] == 0).all() and (s[7:] > 0).all()


def test_the_nll_of_a_categorical_head_is_minus_the_log_probability_of_the_class():
    X, y = planted_classes(seed=3, n=800)
    m = baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,))
    p = m.predict(X)
    assert m.nll(X, y) == pytest.approx(-np.log(p[np.arange(800), y]), rel=1e-12)
    assert np.array_equal(m.nll(X, y), nll.categorical(y, p))


def test_the_gold_head_scores_with_the_31_bin_nll():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=900), rng.integers(0, 6, 900)]).astype(np.float32)
    y = rng.choice([2, 10, 20, 30], 900)
    team = rng.integers(0, 2, 900)
    m = baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,))
    p = m.predict(X, team)
    assert p.shape == (900, 31) and np.array_equal(m.nll(X, y, team), nll.gold(y, p))


def test_a_validation_class_that_the_training_rows_lack_is_left_out_of_the_early_stopping_and_counted():
    X, y = planted_classes(seed=4, n=1500)
    yv = y[:300].copy()
    yv[:20] = 3                                                                         # the class 3 never occurs in the training rows
    yv[20:25] = -1
    m = baselines_b1.M3Size(params={**SMALL, 'early_stopping_round': 5}).fit(X, y, categorical=(2,), valid=(X[:300], yv))
    assert m.n_valid_unseen_ == 20 and m.n_valid_missing_ == 5 and m.n_valid_ == 275
    assert m.summary()['n_valid_unseen'] == 20
    with pytest.raises(ValueError, match='no validation row'):
        baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,), valid=(X[:30], np.full(30, 3)))


def test_categorical_early_stopping_keeps_the_rounds_up_to_the_best_loss():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 5)).astype(np.float32)
    y = rng.integers(0, 3, 1500)                                                       # noise
    Xv, yv = rng.normal(size=(1500, 5)).astype(np.float32), rng.integers(0, 3, 1500)
    m = baselines_b1.M3Size(params={'num_leaves': 15, 'min_data_in_leaf': 5, 'num_iterations': 200, 'early_stopping_round': 10,
                                    'learning_rate': 0.3}).fit(X, y, valid=(Xv, yv))
    info = m.info_[0]
    assert info['stopped_early'] and info['rounds_run'] - info['best_iteration'] == 10
    assert m.boosters_[0].num_trees() == 3 * info['best_iteration'] and info['num_trees'] == 3 * info['best_iteration']


def test_a_categorical_head_refuses_data_that_cannot_be_trusted():
    X, y = planted_classes(seed=2, n=300)
    bad_x = X.copy()
    bad_x[0, 0] = np.inf
    for args in ((X, y + 1.0), (X, y + 10), (X, y - 2), (bad_x, y), (X, y[:-1])):
        with pytest.raises((ValueError, TypeError)):
            baselines_b1.M3Size(params=SMALL).fit(*args, categorical=(2,))
    with pytest.raises(ValueError, match='no row has a class'):
        baselines_b1.M3Size(params=SMALL).fit(X, np.full(300, -1), categorical=(2,))
    with pytest.raises(ValueError, match='valid must be'):
        baselines_b1.M3Size(params=SMALL).fit(X, y, categorical=(2,), valid=(X,))


# =====================================================================================================================================
# the gold head: a model for each team (ruling R19)
# =====================================================================================================================================
def planted_gold(seed=0, n=12_000):
    """(X (n, 2), gold bin (n,), team (n,)): the state x0 raises the chance of the high bin (20) of the blue team and lowers that of the red
    team; the column 1 is the kill layer 0 .. 5."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    team = rng.integers(0, 2, n)
    high = rng.random(n) < 1.0 / (1.0 + np.exp(-2.5 * np.where(team == 0, x0, -x0)))
    return np.column_stack([x0, rng.integers(0, 6, n)]).astype(np.float32), np.where(high, 20, 5), team


def test_the_gold_laws_of_the_teams_move_in_opposite_directions_with_the_state_r19():
    X, y, team = planted_gold(seed=1)
    m = baselines_b1.M3Gold(params={**SMALL, 'num_leaves': 15, 'num_iterations': 60}).fit(X, y, team, categorical=(1,))
    grid = np.array([[2.0, 0], [-2.0, 0]], dtype=np.float32)
    blue = m.predict(grid, np.array([BLUE, BLUE]))[:, 20]
    red = m.predict(grid, np.array([RED, RED]))[:, 20]
    assert blue[0] > 0.8 > 0.2 > blue[1] and red[0] < 0.2 < 0.8 < red[1]               # the odds flip with the sign of the state


def test_each_member_of_the_gold_head_is_fitted_on_the_rows_of_its_team_alone_r19():
    X, y, team = planted_gold(seed=2, n=3000)
    m = baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,))
    assert [mem.n_fit_ for mem in m.members] == [(team == 0).sum(), (team == 1).sum()] and m.n_fit_ == 3000
    for t in (0, 1):
        alone = baselines_b1._GoldModel('masked', SMALL).fit(X[team == t], y[team == t], categorical=(1,))
        assert alone.boosters_[0].model_to_string() == m.member(t).boosters_[0].model_to_string()
    assert m.members[0].boosters_[0].model_to_string() != m.members[1].boosters_[0].model_to_string()
    assert baselines_b1.TEAMS == ('blue', 'red') and m.member(0) is m.members[0] and m.member(np.int64(1)) is m.members[1]


def test_the_gold_head_sends_every_row_to_the_model_of_its_team_in_the_order_of_the_rows_r19():
    X, y, team = planted_gold(seed=3, n=1500)
    m = baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,))
    p = m.predict(X, team)
    for t in (0, 1):
        assert np.array_equal(p[team == t], m.member(t).predict(X[team == t]))
    shuffled = np.random.default_rng(0).permutation(1500)
    assert np.array_equal(m.predict(X[shuffled], team[shuffled]), p[shuffled])


def test_the_gold_head_refuses_a_missing_team_and_bad_team_codes_and_an_unfitted_use():
    X, y, team = planted_gold(seed=3, n=400)
    with pytest.raises(ValueError, match='no row of the red team'):
        baselines_b1.M3Gold(params=SMALL).fit(X, y, np.zeros(400, dtype=int), categorical=(1,))
    for bad in (np.full(400, 2), np.full(400, -1), team[:-1], team.astype(float)):
        with pytest.raises((ValueError, TypeError)):
            baselines_b1.M3Gold(params=SMALL).fit(X, y, bad, categorical=(1,))
    for t in (2, -1, True, 0.0, None):
        with pytest.raises(ValueError):
            baselines_b1.M3Gold(params=SMALL).member(t)
    with pytest.raises(RuntimeError, match='not fitted'):
        baselines_b1.M3Gold(params=SMALL).summary()
    with pytest.raises(ValueError, match='valid must be'):
        baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,), valid=(X, y))
    with pytest.raises(ValueError, match='no validation row of the red team'):
        baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,), valid=(X[team == 0], y[team == 0], team[team == 0]))


def test_the_gold_summary_has_an_entry_for_each_team_blue_first():
    X, y, team = planted_gold(seed=3, n=900)
    m = baselines_b1.M3Gold(params=SMALL).fit(X, y, team, categorical=(1,), valid=(X[:300], y[:300], team[:300]))
    s = m.summary()
    assert set(s['teams']) == {'blue', 'red'} and len(s['best_iteration']) == len(s['num_trees']) == 2
    assert [s['teams'][t]['best_iteration'][0] for t in ('blue', 'red')] == s['best_iteration']
    assert s['n_valid'] == 300 and s['n_fit'] == 900
    json.dumps(s, allow_nan=False)


# =====================================================================================================================================
# the designs
# =====================================================================================================================================
# the table of the brief, written out by hand: the pieces of every design in the order of its columns
EXPECTED_SLOTS = {
    ('M2Rate', 'masked'): ('feat', 'q'),
    ('M2Mark', 'masked'): ('feat', 'q'),
    ('M2Zone', 'masked'): ('feat', 'q', 'm'),
    ('M3Size', 'masked'): ('feat', 'q', 'm', 'z'),
    ('M3Kills', 'masked'): ('feat', 'q', 'm', 'z', 's'),
    ('M3Gold', 'masked'): ('feat', 'q', 'm', 'z', 's', 'layer'),
    ('M3Obj', 'masked'): ('feat', 'K', 'q', 'm', 'z', 's'),
    ('M4Conv', 'masked'): ('feat', 'q', 'm', 'z', 'K', 'O'),
    ('M3Size', 'open'): ('feat',),
    ('M3Kills', 'open'): ('feat', 's'),
    ('M3Gold', 'open'): ('feat', 's', 'layer'),
    ('M3Obj', 'open'): ('feat', 'K', 's'),
    ('M4Conv', 'open'): ('feat', 'K', 'O'),
}
# (lowest raw value, number of levels) of every category
CATEGORY_LEVELS = {'q': (0, 18), 'm': (0, 52), 'z': (-1, 11), 's': (0, 4), 'layer': (1, 6)}


def test_the_slots_of_the_designs_are_those_of_the_brief_table():
    assert baselines_b1.SLOTS == EXPECTED_SLOTS


def test_the_categories_of_b1_are_the_blocks_of_b0_and_k_and_o_are_read_where_decision_p6_says():
    for key, slots in EXPECTED_SLOTS.items():
        blocks = [b.name for b in baselines_b0.LAYOUTS[key].blocks]
        assert [s for s in slots if s in CATEGORY_LEVELS] != [] or blocks == []
        assert sorted(s for s in slots if s in CATEGORY_LEVELS) == sorted(blocks), key
    with_k = {key for key, slots in EXPECTED_SLOTS.items() if 'K' in slots}
    with_o = {key for key, slots in EXPECTED_SLOTS.items() if 'O' in slots}
    assert with_k == {('M3Obj', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open'), ('M4Conv', 'open')}
    assert with_o == {('M4Conv', 'masked'), ('M4Conv', 'open')}
    # B0 reads K and O in the open heads only (P6): the masked columns of B0 have neither
    assert baselines_b0.LAYOUTS[('M3Obj', 'masked')].n_numeric == 39 and baselines_b0.LAYOUTS[('M4Conv', 'masked')].n_numeric == 39


def test_the_levels_of_the_categories_are_those_of_the_spec():
    assert baselines_b1._LEVELS == CATEGORY_LEVELS
    assert (head_rows.N_Q, head_rows.N_M, head_rows.N_ZONE + 1, head_rows.N_SIZE, head_rows.N_LAYER) == (18, 52, 11, 4, 6)


def test_the_masked_m4_design_has_the_columns_of_the_input_builder_in_its_order():
    cols = features.b1_columns(('t_s',))
    layout = baselines_b1.make_layout('M4Conv', 'masked', cols)
    assert layout.names == features.m4_columns(cols) and layout.width == len(cols) + 25
    assert layout.categorical == (len(cols), len(cols) + 1, len(cols) + 2) and layout.d == len(cols)


def test_the_layout_names_the_columns_and_the_categorical_ones():
    cols = features.b1_columns(('t_s',))
    d = len(cols)
    lay = baselines_b1.make_layout('M3Obj', 'masked', cols)
    assert lay.names == cols + ('kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k', 'q', 'm', 'z', 's')
    assert lay.categorical == (d + 4, d + 5, d + 6, d + 7) and lay.width == d + 8 and lay.slots == EXPECTED_SLOTS[('M3Obj', 'masked')]
    assert baselines_b1.make_layout('M3Size', 'open', cols).categorical == ()
    assert baselines_b1.make_layout('M4Conv', 'open', cols).names == cols + features.M4_EXTRA_NAMES[3:] + features.O_NAMES
    for key in EXPECTED_SLOTS:
        assert len(set(baselines_b1.make_layout(*key, cols).names)) == baselines_b1.make_layout(*key, cols).width
    with pytest.raises(ValueError, match='there is no'):
        baselines_b1.make_layout('M2Rate', 'open', cols)
    with pytest.raises(ValueError, match='twice'):
        baselines_b1.make_layout('M2Rate', 'masked', cols + ('q',))


def test_assemble_makes_the_design_by_hand():
    feat = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
    X = baselines_b1._assemble(('feat', 'K', 'q', 'z', 's', 'layer'), feat, np.array([2, 0, 2]),
                               {'q': [17, 0, 3], 'z': [-1, 9, 4], 's': [3, 0, 1], 'layer': [1, 6, 3]},
                               K=[[2, 1, 1.5, 0.5], [0, 0, 0, 0], [8, 8, 2.0, 3.0]])
    expected = np.array([[3.0, 6.0, 2, 1, 1.5, 0.5, 17, 0, 3, 0],                           # z = -1 -> 0, layer 1 -> 0
                         [1.0, np.nan, 0, 0, 0, 0, 0, 10, 0, 5],                           # z = 9 -> 10, layer 6 -> 5
                         [3.0, 6.0, 8, 8, 2.0, 3.0, 3, 5, 1, 2]], dtype=np.float32)
    assert X.dtype == np.float32 and X.flags.c_contiguous and np.array_equal(X, expected, equal_nan=True)
    plain = baselines_b1._assemble(('feat',), feat, None, {})
    assert np.array_equal(plain, feat, equal_nan=True) and plain is not feat


def test_assemble_is_the_input_builder_for_the_slots_of_the_masked_m4_head():
    """The slot order of M4Conv is the column order of ``features.m4_masked_inputs`` (the zone shifted by one)."""
    rng = np.random.default_rng(3)
    n, d = 40, 9
    row = rng.normal(size=(n, d)).astype(np.float32)
    q, m, z = rng.integers(0, 18, n), rng.integers(0, 52, n), rng.integers(-1, 10, n)
    K = np.column_stack([rng.integers(0, 8, (n, 2)), rng.integers(0, 6000, (n, 2))]).astype(float)
    O = rng.integers(0, 3, (n, 18))
    m4 = features.m4_masked_inputs(row, q, m, z, K, O)
    design = baselines_b1._m4_design(m4.copy(), d)
    Kt = K.copy()
    Kt[:, 2:] /= 1000.0
    by_slots = baselines_b1._assemble(baselines_b1.SLOTS[('M4Conv', 'masked')], row, None, {'q': q, 'm': m, 'z': z}, K=Kt, O=O)
    assert np.array_equal(design, by_slots) and design.dtype == np.float32
    assert np.array_equal(m4[:, d + 2], z) and np.array_equal(design[:, d + 2], z + 1)       # the builder's matrix is not the design


def test_assemble_refuses_what_cannot_be_a_design():
    feat = np.zeros((4, 2), dtype=np.float32)
    ok = {'q': [0, 1, 2, 3]}
    assert baselines_b1._assemble(('feat', 'q'), feat, None, ok).shape == (4, 3)
    for slots, codes, kwargs, error in (
            (('feat', 'q'), {'q': [0, 1, 2, 18]}, {}, 'q must lie in 0 .. 17'),
            (('feat', 'z'), {'z': [0, 1, 2, 10]}, {}, 'z must lie in -1 .. 9'), (('feat', 'z'), {'z': [-2, 0, 0, 0]}, {}, 'z must lie'),
            (('feat', 'layer'), {'layer': [0, 1, 1, 1]}, {}, 'layer must lie in 1 .. 6'),
            (('feat', 'q'), {'q': [0, 1, 2]}, {}, 'q has 3 entries'), (('feat', 'q'), {}, {}, 'needs the category q'),
            (('feat', 'K'), {}, {}, 'needs the K block'), (('feat', 'K'), {}, {'K': np.zeros((4, 3))}, 'K must have shape'),
            (('feat', 'K'), {}, {'K': -np.ones((4, 4))}, 'not negative'), (('feat', 'O'), {}, {'O': np.full((4, 18), np.nan)}, 'finite'),
            (('feat', 'm'), {'m': [0, 52, 0, 0]}, {}, 'm must lie in 0 .. 51')):
        with pytest.raises(ValueError, match=error):
            baselines_b1._assemble(slots, feat, None, codes, **kwargs)
    with pytest.raises(TypeError):
        baselines_b1._assemble(('feat', 'q'), feat, None, {'q': [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='take must lie'):
        baselines_b1._assemble(('feat',), feat, np.array([0, 4]), {})
    with pytest.raises(ValueError, match='2-d'):
        baselines_b1._assemble(('feat',), np.zeros(4), None, {})


def test_m4_design_checks_the_matrix_and_leaves_the_callers_float32_matrix_alone_unless_it_owns_it():
    d = 3
    m4 = np.zeros((2, d + 25), dtype=np.float32)
    m4[:, d + 2] = [-1, 9]
    shifted = baselines_b1._m4_design(m4.copy(), d)
    assert shifted[:, d + 2].tolist() == [0, 10]
    readonly = m4.copy()
    readonly.setflags(write=False)
    out = baselines_b1._m4_design(readonly, d)                                           # a read-only matrix is copied, not changed
    assert out[:, d + 2].tolist() == [0, 10] and readonly[:, d + 2].tolist() == [-1, 9]
    as64 = baselines_b1._m4_design(m4.astype(np.float64), d)
    assert as64.dtype == np.float32
    for bad, error in ((m4[:, :-1], 'columns'), (np.zeros(5), 'columns')):
        with pytest.raises(ValueError, match=error):
            baselines_b1._m4_design(bad, d)
    for column, value, name in ((d + 2, 10, 'z'), (d + 2, 0.5, 'z'), (d + 2, -2, 'z'), (d, 18, 'q'), (d, -1, 'q'), (d + 1, 52, 'm'),
                                (d + 1, 1.5, 'm'), (d + 1, np.nan, 'm')):
        worse = m4.copy()
        worse[0, column] = value
        with pytest.raises(ValueError, match=f'the {name} column of the masked M4 input'):
            baselines_b1._m4_design(worse, d)


# =====================================================================================================================================
# the rows of the heads (decision R18: the same rows as B0, with the b1 features and the designs of this module)
# =====================================================================================================================================
@pytest.fixture(scope='module')
def oracle(heads_world):
    return HeadsOracle(heads_world)


@pytest.fixture(scope='module')
def dropped(heads_world):
    return dropped_of(heads_world)


def rows_of(heads_world, dropped, head, kind, role='FIT'):
    return baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, role).data(head, kind)


def b1_of(heads_world, dropped, src, index):
    return features.b1(heads_world.s2, [HeadsOracle(heads_world).view(src, index)], dropped)[0]


def design_row(slots, feat, codes=None, K=None, O=None):
    """One design row by plain concatenation: the b1 row, the K block, the O block and the categories as value - lowest value."""
    parts = []
    for slot in slots:
        if slot == 'feat':
            parts.append(np.asarray(feat, dtype=np.float32))
        elif slot == 'K':
            parts.append(np.asarray(K, dtype=np.float32))
        elif slot == 'O':
            parts.append(np.asarray(O, dtype=np.float32))
        else:
            parts.append(np.array([codes[slot] - CATEGORY_LEVELS[slot][0]], dtype=np.float32))
    return np.concatenate(parts)


def test_the_rows_of_b1_are_the_rows_of_b0_with_other_designs(heads_world, dropped):
    """Ruling R18: one selection of rows, targets and exposures for both models."""
    b0 = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT')
    b1 = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    for key in ALL_HEADS:
        a, b = b0.data(*key), b1.data(*key)
        for field in ('y', 'e', 'match', 'row', 'team'):
            x, y = getattr(a, field), getattr(b, field)
            assert (x is None) == (y is None), (key, field)
            if x is not None:
                assert np.array_equal(x, y), (key, field)
        assert a.n_zero_exposure == b.n_zero_exposure and a.X.shape[0] == b.X.shape[0], key


def test_the_designs_are_float32_matrices_of_the_width_of_the_layout(heads_world, dropped):
    cols = features.b1_columns(dropped)
    for key in ALL_HEADS:
        d = rows_of(heads_world, dropped, *key)
        width = baselines_b1.make_layout(*key, cols).width
        assert d.X.dtype == np.float32 and d.X.flags.c_contiguous and d.X.shape[1] == d.layout.width == width
        assert d.layout.slots == EXPECTED_SLOTS[key] and d.layout.d == len(cols), key
        for c in d.layout.categorical:
            col = d.X[:, c]
            assert (col == np.floor(col)).all() and col.min() >= 0, key


def test_m2_rate_designs_are_the_b1_row_and_the_bin(heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M2Rate', 'masked')
    pairs = [(r, q) for r in oracle.frame_rows('FIT') for q in range(18) if oracle.expo[r, q] > 0]
    assert d.X.shape[0] == len(pairs) == len(d.y)
    counts = {}
    for i in oracle.onsets('FIT'):
        counts[(oracle.on_row[i], oracle.on_bin[i])] = counts.get((oracle.on_row[i], oracle.on_bin[i]), 0) + 1
    for j in range(0, len(pairs), 7):
        r, q = pairs[j]
        expected = design_row(('feat', 'q'), b1_of(heads_world, dropped, 0, r), {'q': q})
        assert np.array_equal(d.X[j], expected, equal_nan=True), j
        assert d.y[j] == counts.get((r, q), 0) and d.e[j] == oracle.expo[r, q] / 1000.0 and d.row[j] == r


def test_masked_designs_use_the_frame_clock_view_of_the_onset_row_and_the_categories_of_the_unit(heads_world, dropped, oracle):
    fit = oracle.onsets('FIT')
    for head, slots in (('M2Zone', ('feat', 'q', 'm')), ('M3Size', ('feat', 'q', 'm', 'z')), ('M3Kills', ('feat', 'q', 'm', 'z', 's'))):
        d = rows_of(heads_world, dropped, head, 'masked')
        assert d.X.shape[0] == len(fit), head
        for j, i in enumerate(fit):
            u = oracle.on_unit[i]
            codes = {'q': oracle.on_bin[i], 'm': oracle.mark[u], 'z': oracle.zone(u), 's': oracle.size[u]}
            expected = design_row(slots, b1_of(heads_world, dropped, 0, oracle.on_row[i]), codes)
            assert np.array_equal(d.X[j], expected, equal_nan=True), (head, j)
    assert any(oracle.zone(oracle.on_unit[i]) < 0 for i in fit)                           # the level "no position" is exercised


def test_the_masked_object_head_of_b1_is_conditioned_on_the_true_k_decision_p6(heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M3Obj', 'masked')
    fit = oracle.onsets('FIT')
    assert d.X.shape[0] == len(fit) and d.y.shape == (len(fit), 18)
    for j, i in enumerate(fit):
        u = oracle.on_unit[i]
        codes = {'q': oracle.on_bin[i], 'm': oracle.mark[u], 'z': oracle.zone(u), 's': oracle.size[u]}
        b1 = b1_of(heads_world, dropped, 0, oracle.on_row[i])
        expected = design_row(EXPECTED_SLOTS[('M3Obj', 'masked')], b1, codes, K=oracle.kblock(u))
        assert np.array_equal(d.X[j], expected, equal_nan=True) and np.array_equal(d.y[j], oracle.O[u]), j


def test_the_masked_conversion_head_of_b1_is_conditioned_on_the_true_k_and_o_decision_p6(heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M4Conv', 'masked')
    fit = [i for i in oracle.onsets('FIT') if oracle.cexp[oracle.on_unit[i]] > 0]
    assert d.X.shape[0] == len(fit) and d.n_zero_exposure == len(oracle.onsets('FIT')) - len(fit) > 0
    for j, i in enumerate(fit):
        u = oracle.on_unit[i]
        codes = {'q': oracle.on_bin[i], 'm': oracle.mark[u], 'z': oracle.zone(u)}
        b1 = b1_of(heads_world, dropped, 0, oracle.on_row[i])
        expected = design_row(EXPECTED_SLOTS[('M4Conv', 'masked')], b1, codes, K=oracle.kblock(u), O=oracle.O[u])
        assert np.array_equal(d.X[j], expected, equal_nan=True), j
        assert np.array_equal(d.y[j], oracle.conv[u]) and d.e[j] == oracle.cexp[u] / 1000.0


def test_the_gold_designs_have_the_layer_of_the_team_of_the_entry(heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M3Gold', 'masked')
    fit = oracle.onsets('FIT')
    entries = [(i, t) for t in (0, 1) for i in fit if oracle.K[oracle.on_unit[i], t] >= 1]
    assert d.X.shape[0] == len(entries) and d.team.tolist() == [t for _, t in entries]
    for j, (i, t) in enumerate(entries):
        u = oracle.on_unit[i]
        codes = {'q': oracle.on_bin[i], 'm': oracle.mark[u], 'z': oracle.zone(u), 's': oracle.size[u], 'layer': min(oracle.K[u, t], 6)}
        expected = design_row(EXPECTED_SLOTS[('M3Gold', 'masked')], b1_of(heads_world, dropped, 0, oracle.on_row[i]), codes)
        assert np.array_equal(d.X[j], expected, equal_nan=True) and d.y[j] == oracle.gold_bin[u, t], j
    assert (np.array([min(oracle.K[oracle.on_unit[i], t], 6) for i, t in entries]) == 6).any()        # the cap at six kills occurs


def test_open_designs_use_the_event_views_of_the_unit(heads_world, dropped, oracle):
    units = oracle.units('FIT')
    d = rows_of(heads_world, dropped, 'M3Obj', 'open')
    assert d.X.shape[0] == len(units)
    for j, u in enumerate(units):
        b1 = b1_of(heads_world, dropped, 1, oracle.event_row(u, 1))
        expected = design_row(EXPECTED_SLOTS[('M3Obj', 'open')], b1, {'s': oracle.size[u]}, K=oracle.kblock(u))
        assert np.array_equal(d.X[j], expected, equal_nan=True) and d.row[j] == u, j
    d4 = rows_of(heads_world, dropped, 'M4Conv', 'open')
    with_view = [u for u in units if oracle.event_row(u, 2) is not None]
    assert d4.X.shape[0] == len(with_view) < len(units) and d4.n_zero_exposure == len(units) - len(with_view)
    for j, u in enumerate(with_view):
        b1 = b1_of(heads_world, dropped, 2, oracle.event_row(u, 2))
        expected = design_row(EXPECTED_SLOTS[('M4Conv', 'open')], b1, K=oracle.kblock(u), O=oracle.O[u])
        assert np.array_equal(d4.X[j], expected, equal_nan=True) and d4.e[j] == oracle.cexp[u] / 1000.0, j
    g = rows_of(heads_world, dropped, 'M3Gold', 'open')
    entries = [(u, t) for t in (0, 1) for u in units if oracle.K[u, t] >= 1]
    assert g.X.shape[0] == len(entries)
    for j, (u, t) in enumerate(entries):
        expected = design_row(EXPECTED_SLOTS[('M3Gold', 'open')], b1_of(heads_world, dropped, 1, oracle.event_row(u, 1)),
                              {'s': oracle.size[u], 'layer': min(oracle.K[u, t], 6)})
        assert np.array_equal(g.X[j], expected, equal_nan=True), j


def test_a_nan_of_b1_stays_a_nan_in_the_design(heads_world, dropped, oracle):
    """Three event views of the heads world (src 2) have no distance between the teams: b1 keeps the NaN of ``geo_min`` and ``geo_mean``
    (LightGBM reads it as missing), b0 fills it with the FIT mean."""
    d = rows_of(heads_world, dropped, 'M4Conv', 'open')
    d0 = baselines_b0.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means, 'FIT').data('M4Conv', 'open')
    nan_rows = np.flatnonzero(np.isnan(d.X).any(axis=1))
    assert len(nan_rows) == 3 and np.isfinite(d0.X).all()
    names = d.layout.names
    assert np.isnan(d.X[nan_rows][:, [names.index('geo_min'), names.index('geo_mean')]]).all()
    assert np.isnan(d.X[:, names.index('geo_min')]).sum() == 3 and np.isfinite(d.X[:, names.index('kills_blue')]).all()


def test_the_rows_of_a_role_are_those_of_the_role_and_a_bad_role_or_column_is_refused(heads_world, dropped):
    d = rows_of(heads_world, dropped, 'M2Mark', 'masked', role='DEV')
    assert set(d.match.tolist()) <= {m for m in range(8) if np.asarray(heads_world.s1.array('matches/role'))[m] == C.ROLES.index('DEV')}
    for role in ('TEST', 'EXTERNAL', 'CAL', 'AUDIT'):
        with pytest.raises(ValueError):
            baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, role)
    with pytest.raises(ValueError, match='not columns'):
        baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, ('nothing',), 'FIT')
    with pytest.raises(TypeError):
        baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, 't_s', 'FIT')
    rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    for head, kind in (('M2Rate', 'open'), ('M5', 'masked'), ('M3Size', 'both')):
        with pytest.raises(ValueError, match='there is no'):
            rows.data(head, kind)


def test_the_arrays_of_one_call_can_be_changed_without_changing_the_next_call(heads_world, dropped):
    rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    for key in (('M2Rate', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open')):
        first = rows.data(*key)
        kept = np.array(first.X)
        first.X[...] = 0.0                                                                # what a caller that owns its design may do
        assert np.array_equal(rows.data(*key).X, kept, equal_nan=True), key


# =====================================================================================================================================
# B1Heads: fit
# =====================================================================================================================================
@pytest.fixture(scope='module')
def fitted(heads_world, dropped):
    return baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=SMALL)


def texts_of(head):
    return head.state()[0]


def test_fit_makes_the_eight_masked_and_the_five_open_heads(fitted, dropped):
    assert tuple(fitted.masked) == HEAD_NAMES and tuple(fitted.open) == OPEN_NAMES
    for kind, group in (('masked', fitted.masked), ('open', fitted.open)):
        for name, head in group.items():
            assert type(head) is getattr(baselines_b1, name) and head.kind == kind and head.n_fit_ > 0, (kind, name)
            for part in head.parts:                                                          # the parameters of the call reach every model
                assert part.rounds == SMALL['num_iterations'] and part.params['num_leaves'] == SMALL['num_leaves'], (kind, name)
                assert part.layout_ == baselines_b1.make_layout(name, kind, fitted.columns), (kind, name)
    assert fitted.dropped == tuple(dropped) and fitted.columns == features.b1_columns(dropped)
    assert fitted.fit_roles == ('FIT',) and fitted.stop_roles == ('STOP',)


def test_every_head_is_what_a_fresh_fit_head_gives_on_the_rows_of_the_oracle(fitted, heads_world, dropped):
    """B1Heads.fit only wires rows to heads: a head fitted on the same rows with the same parameters is the same model."""
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            d = rows_of(heads_world, dropped, name, kind)
            v = rows_of(heads_world, dropped, name, kind, role='STOP')
            fresh = getattr(baselines_b1, name)(kind, SMALL)
            if name == 'M3Gold':
                fresh.fit(d.X, d.y, d.team, categorical=d.layout.categorical, valid=(v.X, v.y, v.team))
            elif d.e is None:
                fresh.fit(d.X, d.y, categorical=d.layout.categorical, valid=(v.X, v.y))
            else:
                fresh.fit(d.X, d.y, d.e, categorical=d.layout.categorical, valid=(v.X, v.y, v.e))
            assert texts_of(fresh) == texts_of(head), (kind, name)


def test_the_heads_are_fitted_on_the_fit_rows_and_stopped_on_the_stop_rows(fitted, heads_world, dropped, oracle):
    fit_onsets, stop_onsets = oracle.onsets('FIT'), oracle.onsets('STOP')
    mark = fitted.masked['M2Mark']
    # a STOP onset whose mark no FIT onset has cannot be scored by the booster: it is counted, not used
    assert mark.n_fit_ == len(fit_onsets) and mark.n_valid_ + mark.n_valid_unseen_ + mark.n_valid_missing_ == len(stop_onsets)
    fit_marks = {oracle.mark[oracle.on_unit[j]] for j in fit_onsets}
    assert mark.n_valid_unseen_ == sum(1 for i in stop_onsets if oracle.mark[oracle.on_unit[i]] not in fit_marks)
    zone = fitted.masked['M2Zone']
    assert zone.n_missing_ == sum(1 for i in fit_onsets if oracle.zone(oracle.on_unit[i]) < 0) > 0
    cells = sum(1 for r in oracle.frame_rows('FIT') for q in range(18) if oracle.expo[r, q] > 0)
    stop_cells = sum(1 for r in oracle.frame_rows('STOP') for q in range(18) if oracle.expo[r, q] > 0)
    assert fitted.masked['M2Rate'].n_fit_ == cells and fitted.masked['M2Rate'].n_valid_ == stop_cells
    units, stop_units = oracle.units('FIT'), oracle.units('STOP')
    assert fitted.open['M3Size'].n_fit_ == len(units) and fitted.open['M3Size'].n_valid_ == len(stop_units)


def test_the_validation_data_of_every_head_are_the_stop_rows_and_the_training_data_the_fit_rows(heads_world, dropped, monkeypatch):
    seen = spy_dataset(monkeypatch)
    fit_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    stop_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'STOP')
    for key in (('M2Rate', 'masked'), ('M3Obj', 'masked'), ('M4Conv', 'open'), ('M3Size', 'open')):
        del seen[:]
        baselines_b1.fit_head(fit_rows, stop_rows, *key, params=SMALL)
        train, valid = seen[0], seen[1]
        assert np.array_equal(train['data'], fit_rows.data(*key).X, equal_nan=True), key
        width = len(fit_rows.columns)
        assert np.array_equal(valid['data'][:, :width], stop_rows.data(*key).X[:, :width], equal_nan=True), key
        assert 'reference' in valid['kwargs'] and 'reference' not in train['kwargs']


def test_the_categories_of_the_designs_are_the_categorical_features_of_the_boosters(heads_world, dropped, monkeypatch):
    seen = spy_dataset(monkeypatch)
    fit_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    stop_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'STOP')
    for key, expected in ((('M2Rate', 'masked'), ['q']), (('M3Obj', 'masked'), ['q', 'm', 'z', 's']),
                          (('M4Conv', 'masked'), ['q', 'm', 'z']), (('M3Gold', 'open'), ['s', 'layer']), (('M3Size', 'open'), None)):
        del seen[:]
        baselines_b1.fit_head(fit_rows, stop_rows, *key, params=SMALL)
        names = baselines_b1.make_layout(*key, fit_rows.columns).names
        given = seen[0]['kwargs']['categorical_feature']
        assert (given == 'auto' and expected is None) or [names[i] for i in given] == expected, key        # the others have no category


def test_the_float32_design_is_freed_as_soon_as_lightgbm_has_binned_it(heads_world, dropped, monkeypatch):
    """Memory (the 14 GiB design of M2Rate): when the first booster starts to train, nothing holds the float32 matrix any more -- LightGBM
    keeps the binned one, the head and ``fit_head`` have let go. A weak reference to the design of every kind of head is dead by then."""
    import weakref
    designs, alive = [], []
    real_dataset, real_train = lgb.Dataset, lgb.train

    def dataset(data, *args, **kwargs):
        designs.append(weakref.ref(data))
        return real_dataset(data, *args, **kwargs)

    def train(params, train_set, *args, **kwargs):
        alive.append(designs[0]() is not None)                                            # the first Dataset is the one of the FIT rows
        return real_train(params, train_set, *args, **kwargs)
    monkeypatch.setattr(lgb, 'Dataset', dataset)
    monkeypatch.setattr(lgb, 'train', train)
    fit_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    stop_rows = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'STOP')
    for key in (('M2Rate', 'masked'), ('M3Size', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open')):
        del designs[:], alive[:]
        baselines_b1.fit_head(fit_rows, stop_rows, *key, params=SMALL)
        assert alive and not any(alive), key                                       # every booster of the head, the first one included
        assert designs[0]() is None, key                                           # ... and after the fit as well


def test_a_design_that_a_head_owns_is_taken_once():
    holder = baselines_b1._Design(np.zeros((2, 2)))
    assert holder.take().shape == (2, 2)
    with pytest.raises(RuntimeError, match='taken already'):
        holder.take()
    X, y, e = synthetic_with_exposure(seed=3, n=600)
    a = baselines_b1.M2Rate(params=SMALL).fit(baselines_b1._Design(X), y, e, categorical=(3,))
    b = baselines_b1.M2Rate(params=SMALL).fit(X, y, e, categorical=(3,))
    assert a.state()[0] == b.state()[0]                                                   # a design in a holder is the same design


def test_fitting_uses_the_fit_and_stop_matches_only(heads_world, fitted, dropped, tmp_path):
    """Scramble every row of S1, S2 and the labels of a match that is neither FIT nor STOP: the heads do not change."""
    copy = tmp_path / 'scrambled'
    for part in ('s1', 's2', 'labels'):
        shutil.copytree(heads_world.s1.root.parent / part, copy / part)
    s1, s2, labels = Store(copy / 's1'), Store(copy / 's2'), Store(copy / 'labels')
    rng = np.random.default_rng(0)
    role = np.asarray(s1.array('matches/role'))
    used = lambda r: np.isin(r, (C.ROLES.index('FIT'), C.ROLES.index('STOP')))                  # noqa: E731
    used_unit = used(role[np.asarray(s1.array('units/match'))])
    used_row = used(role[np.asarray(s1.array('frame_rows/match'))])
    used_view = used(role[np.asarray(s2.array('views/match'))])
    used_onset = used_row[np.asarray(s1.array('onsets/row'))]
    for name, keep in (('units/mark', used_unit), ('units/size', used_unit), ('units/K', used_unit), ('units/O', used_unit),
                       ('units/conv', used_unit), ('units/conv_exposure_ms', used_unit), ('frame_rows/exposure_ms', used_row),
                       ('onsets/bin', used_onset), ('onsets/mark', used_onset), ('labels/gold_bin', used_unit),
                       ('labels/kill_gold', used_unit), ('x/summary', used_view), ('x/players_num', used_view), ('x/geo_pp', used_view)):
        store = labels if name.startswith('labels/') else (s2 if name.startswith('x/') else s1)
        a = np.load(store.path(name), mmap_mode='r+')
        a[~keep] = rng.permutation(a[~keep].reshape(-1)).reshape(a[~keep].shape) if a.dtype.kind != 'f' else a[~keep][::-1] + 1.0
        a.flush()
        del a
    again = baselines_b1.B1Heads.fit(s1, s2, labels, dropped, params=SMALL)
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            assert texts_of(head) == texts_of(getattr(again, kind)[name]), (kind, name)
    del s1, s2, labels, again
    gc.collect()


def test_fit_calls_the_e6_check_before_the_reads_before_each_head_and_before_each_booster(heads_world, dropped):
    calls = []
    fit = baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=SMALL, check=lambda: calls.append(1))
    boosters = sum(1 for kind in ('masked', 'open') for head in getattr(fit, kind).values() for part in head.parts
                   for b in part.boosters_ if b is not None)
    assert len(calls) == 1 + len(HEAD_NAMES) + len(OPEN_NAMES) + boosters


def test_fit_head_calls_the_e6_check_first_before_it_reads_a_single_array(heads_world, dropped, monkeypatch):
    """A runner that fits head by head (``run_s3``) calls ``fit_head`` alone: the guard must come before the reads of that head."""
    w = heads_world
    fit_rows = baselines_b1.HeadRows(w.s1, w.s2, w.labels, dropped, 'FIT')
    stop_rows = baselines_b1.HeadRows(w.s1, w.s2, w.labels, dropped, 'STOP')
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])

    def refuse():
        raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        baselines_b1.fit_head(fit_rows, stop_rows, 'M2Rate', 'masked', params=SMALL, check=refuse)
    assert reads == []
    seen = []
    head = baselines_b1.fit_head(fit_rows, stop_rows, 'M2Mark', 'masked', params=SMALL, check=lambda: seen.append(len(reads)))
    assert head.name == 'M2Mark' and seen[0] == 0 and len(seen) >= 2 and reads      # the first call before any read, then one per booster


def test_an_e6_refusal_stops_the_fit_before_anything_is_read(heads_world, dropped, monkeypatch):
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])

    def refuse():
        raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=SMALL, check=refuse)
    assert reads == []


def test_an_e6_refusal_between_two_boosters_stops_the_fit(heads_world, dropped):
    calls = []

    def refuse_late():
        calls.append(1)
        if len(calls) == 30:
            raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=SMALL, check=refuse_late)
    assert len(calls) == 30


def test_fit_refuses_a_holdout_role_the_same_role_twice_and_a_wrong_roles_argument(heads_world, dropped):
    args = (heads_world.s1, heads_world.s2, heads_world.labels, dropped)
    for roles in ({'fit': 'TEST', 'stop': 'STOP'}, {'fit': 'FIT', 'stop': 'EXTERNAL'}, {'fit': 'CAL', 'stop': 'STOP'},
                  {'fit': 'FIT', 'stop': 'CAL'}):
        with pytest.raises(ValueError) as raised:
            baselines_b1.B1Heads.fit(*args, roles=roles, params=SMALL)
        assert KEY_PREFIX not in str(raised.value)
    for roles in ({'fit': 'FIT'}, {'fit': 'FIT', 'stop': 'STOP', 'extra': 1}, {'train': 'FIT', 'stop': 'STOP'}, {}):
        with pytest.raises(ValueError, match='exactly the keys'):
            baselines_b1.B1Heads.fit(*args, roles=roles, params=SMALL)
    for roles in ('FIT', ['FIT', 'STOP']):
        with pytest.raises(TypeError):
            baselines_b1.B1Heads.fit(*args, roles=roles, params=SMALL)
    for roles in ({'fit': 'FIT', 'stop': 'FIT'}, {'fit': ('FIT', 'STOP'), 'stop': 'STOP'}):
        with pytest.raises(ValueError, match='must not be one of the'):
            baselines_b1.B1Heads.fit(*args, roles=roles, params=SMALL)


def test_a_holdout_role_is_refused_before_any_array_is_read(heads_world, dropped, monkeypatch):
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])
    for roles in ({'fit': 'TEST', 'stop': 'STOP'}, {'fit': 'FIT', 'stop': 'EXTERNAL'}):
        with pytest.raises(ValueError):
            baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, roles=roles, params=SMALL)
    assert not [name for name in reads if name not in ('matches/role', 'frame_rows/match', 'onsets/row', 'units/match')]


def test_fit_accepts_the_roles_argument_of_ruling_r21(heads_world, dropped, fitted):
    again = baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, roles={'fit': 'FIT', 'stop': 'STOP'},
                                     params=SMALL)
    assert texts_of(again.masked['M3Size']) == texts_of(fitted.masked['M3Size']) and again.fit_roles == ('FIT',)


def test_fit_leaks_no_match_key_and_refuses_a_bad_drop_list(heads_world, dropped):
    for bad in (('nothing',), 'no'):
        with pytest.raises((ValueError, TypeError)) as raised:
            baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, bad, params=SMALL)
        assert KEY_PREFIX not in str(raised.value)


def test_fit_head_refuses_rows_made_with_different_drop_lists(heads_world, dropped):
    a = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, dropped, 'FIT')
    b = baselines_b1.HeadRows(heads_world.s1, heads_world.s2, heads_world.labels, tuple(dropped) + ('rift_element_air',), 'STOP')
    with pytest.raises(ValueError, match='same dropped columns'):
        baselines_b1.fit_head(a, b, 'M3Size', 'open', params=SMALL)


def test_b1_is_bit_reproducible_for_all_the_heads(heads_world, dropped, fitted):
    again = baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=SMALL)
    rng = np.random.default_rng(7)
    batch = random_batch(rng, 12, len(fitted.columns))
    assert np.array_equal(fitted.predict_all(batch), again.predict_all(batch))
    for kind in ('masked', 'open'):
        for name, head in getattr(fitted, kind).items():
            assert texts_of(head) == texts_of(getattr(again, kind)[name]), (kind, name)


def test_the_seed_changes_the_trees(heads_world, dropped, fitted):
    other = baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params={**SMALL, 'seed': 2})
    assert any(texts_of(h) != texts_of(other.masked[name]) for name, h in fitted.masked.items())


def test_the_summary_of_the_fit_is_plain_json_with_the_rounds_and_trees_of_every_model(fitted):
    summary = fitted.summary()
    assert set(summary) == {'masked', 'open'} and tuple(summary['masked']) == HEAD_NAMES and tuple(summary['open']) == OPEN_NAMES
    cells = {'M2Rate': 1, 'M2Mark': 1, 'M2Zone': 1, 'M3Size': 1, 'M3Kills': 2, 'M3Gold': 2, 'M3Obj': 18, 'M4Conv': 18}
    for kind in ('masked', 'open'):
        for name, one in summary[kind].items():
            assert len(one['best_iteration']) == len(one['num_trees']) == len(one['rounds_run']) == cells[name], (kind, name)
            assert all(type(i) is int for i in one['best_iteration'] + one['num_trees']), (kind, name)
            assert all(k <= SMALL['num_iterations'] for k in one['best_iteration']), (kind, name)
    zone = summary['masked']['M2Zone']
    assert zone['n_missing'] == fitted.masked['M2Zone'].n_missing_ and zone['kind'] == 'masked' and zone['n_valid'] > 0
    json.dumps(summary, allow_nan=False)


def test_the_cells_that_never_happen_have_no_booster_and_the_half_event_rate(fitted):
    for head in (fitted.masked['M3Obj'], fitted.open['M3Obj'], fitted.masked['M4Conv'], fitted.open['M4Conv']):
        assert head.degenerate_[WORLD_ZERO_CELL] and head.n_events_[WORLD_ZERO_CELL] == 0 and head.boosters_[WORLD_ZERO_CELL] is None
        assert np.array_equal(head.degenerate_, head.n_events_ == 0)
        assert head.degenerate_.sum() < len(head.degenerate_)


def test_the_zero_exposure_rows_are_left_out_of_the_fit_and_counted(fitted, oracle):
    fit = oracle.onsets('FIT')
    zero = sum(1 for i in fit if oracle.cexp[oracle.on_unit[i]] == 0)
    assert zero > 0 and fitted.masked['M4Conv'].n_zero_exposure_ == zero and fitted.masked['M4Conv'].n_fit_ == len(fit) - zero
    units = oracle.units('FIT')
    assert fitted.open['M4Conv'].n_fit_ == sum(1 for u in units if oracle.cexp[u] > 0)
    cells = sum(1 for r in oracle.frame_rows('FIT') for q in range(18) if oracle.expo[r, q] == 0)
    assert fitted.masked['M2Rate'].n_zero_exposure_ == cells > 0                          # the bins beyond the cut at the game end


# =====================================================================================================================================
# B1Heads: scoring (row-wise NLL)
# =====================================================================================================================================
@pytest.fixture(scope='module')
def scores(fitted, heads_world):
    return fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')


def test_nll_gives_a_score_object_for_every_masked_head(scores):
    assert tuple(scores) == HEAD_NAMES
    for name, sc in scores.items():
        assert len(sc.nll) == len(sc.match) == len(sc.row) > 0 and np.isfinite(sc.nll).all() and (sc.nll >= 0).all(), name
        assert type(sc) is head_rows.HeadScores


def test_the_zone_scores_leave_out_the_missing_zones_and_count_them(scores, oracle):
    dev = oracle.onsets('DEV')
    missing = sum(1 for i in dev if oracle.zone(oracle.on_unit[i]) < 0)
    assert scores['M2Zone'].n_unscored == missing and len(scores['M2Zone'].nll) == len(dev) - missing
    assert scores['M2Mark'].n_unscored == 0 and len(scores['M2Mark'].nll) == len(dev)


def test_rows_and_cells_without_exposure_are_not_scored_and_are_counted(scores, oracle):
    dev = oracle.onsets('DEV')
    zero = sum(1 for i in dev if oracle.cexp[oracle.on_unit[i]] == 0)
    assert scores['M4Conv'].n_unscored == zero and len(scores['M4Conv'].nll) == len(dev) - zero
    cells = sum(1 for r in oracle.frame_rows('DEV') for q in range(18) if oracle.expo[r, q] == 0)
    assert scores['M2Rate'].n_unscored == cells > 0 and len(scores['M2Rate'].nll) == 18 * len(oracle.frame_rows('DEV')) - cells


def test_the_m2_rate_scores_are_per_frame_row_sums_over_the_bins(scores, fitted, heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M2Rate', 'masked', role='DEV')
    per_cell = fitted.masked['M2Rate'].nll(d.X, d.y, d.e)
    values, match = scores['M2Rate'].per_row()
    assert len(values) == len(oracle.frame_rows('DEV'))
    for k, r in enumerate(sorted(set(d.row.tolist()))):
        assert values[k] == pytest.approx(per_cell[d.row == r].sum(), rel=1e-12)


def test_the_gold_scores_come_from_the_model_of_the_team_of_each_entry_r19(fitted, heads_world, dropped):
    d = rows_of(heads_world, dropped, 'M3Gold', 'masked', role='DEV')
    sc = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')['M3Gold']
    gold = fitted.masked['M3Gold']
    expected = np.array([nll.gold(d.y[j:j + 1], gold.member(int(d.team[j])).predict(d.X[j:j + 1]))[0] for j in range(len(d.y))])
    assert sc.nll == pytest.approx(expected, rel=1e-12) and len(sc.nll) == len(d.y)


def test_the_kill_scores_sum_the_two_teams_of_an_onset(scores, fitted, heads_world, dropped, oracle):
    d = rows_of(heads_world, dropped, 'M3Kills', 'masked', role='DEV')
    cells = fitted.masked['M3Kills'].nll(d.X, d.y, d.e, cells=True)
    assert scores['M3Kills'].nll == pytest.approx(cells.sum(axis=1), rel=1e-12) and cells.shape == (len(oracle.onsets('DEV')), 2)


def test_scores_means_are_match_weighted_and_finite(scores):
    for name, sc in scores.items():
        assert np.isfinite(sc.mean()), name
        boot = sc.bootstrap(B=20)
        assert boot.shape == (20,) and np.isfinite(boot).all(), name


def test_open_scores_exist_for_the_five_open_heads(fitted, heads_world):
    open_scores = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind='open')
    assert tuple(open_scores) == OPEN_NAMES and all(np.isfinite(s.nll).all() and len(s.nll) > 0 for s in open_scores.values())
    with pytest.raises(ValueError, match='kind'):
        fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', kind='both')


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_b0_and_b1_score_the_same_rows_so_that_their_nll_can_be_paired(fitted, heads_world):
    """Ruling R18 at the end of the line: the evaluation rows, the matches and the left-out counts of every head are the same for the two
    models, so a paired bootstrap of the difference of two NLL means compares the same rows."""
    b0 = baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means)
    stores = (heads_world.s1, heads_world.s2, heads_world.labels)
    for kind in ('masked', 'open'):
        a, b = b0.nll(*stores, 'DEV', kind), fitted.nll(*stores, 'DEV', kind)
        assert tuple(a) == tuple(b)
        for name in a:
            assert np.array_equal(a[name].row, b[name].row) and np.array_equal(a[name].match, b[name].match), (kind, name)
            assert a[name].n_unscored == b[name].n_unscored and a[name].per_row()[0].shape == b[name].per_row()[0].shape, (kind, name)


def test_scoring_refuses_a_holdout_role_and_changes_no_head(fitted, heads_world):
    before = {name: texts_of(h) for name, h in fitted.masked.items()}
    for role in ('TEST', 'EXTERNAL'):
        with pytest.raises(ValueError):
            fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, role)
    fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'STOP')
    assert before == {name: texts_of(h) for name, h in fitted.masked.items()}


def test_the_e6_check_is_called_before_scoring(fitted, heads_world):
    calls = []
    fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV', check=lambda: calls.append(1))
    assert len(calls) == 1 + len(HEAD_NAMES)


# =====================================================================================================================================
# save and load
# =====================================================================================================================================
def random_batch(rng, n, d):
    K = rng.integers(0, 7, (n, 4)).astype(float) * [1, 1, 300, 300]
    return baselines_b1.HeadBatch(row_view=rng.normal(0.0, 3.0, (n, d)), q=rng.integers(0, 18, n), m=rng.integers(0, 52, n),
                                  z=rng.integers(-1, 10, n), s=rng.integers(0, 4, n), K=K, O=rng.integers(0, 3, (n, 18)).astype(float))


def test_save_writes_a_json_a_folder_for_each_head_with_its_json_and_one_text_file_for_each_booster_and_nothing_else(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    names = sorted(p.relative_to(tmp_path / 'b1').as_posix() for p in (tmp_path / 'b1').rglob('*') if p.is_file())
    doc = json.loads((tmp_path / 'b1' / 'b1_heads.json').read_text(encoding='utf-8'))
    n_models = sum(len(meta['files']) for kind in doc['heads'].values() for meta in kind.values())
    n_heads = len(HEAD_NAMES) + len(OPEN_NAMES)
    assert names[0] == 'b1_heads.json' and len(names) == 1 + n_heads + n_models
    assert all(n.startswith('models/') and n.endswith(('.txt', '/head.json')) for n in names[1:])
    assert sum(n.endswith('/head.json') for n in names) == n_heads
    assert n_models == sum(1 for kind in ('masked', 'open') for h in getattr(fitted, kind).values() for p in h.parts for b in p.boosters_
                           if b is not None)
    assert doc['schema'] == 'b1_heads_v2' and doc['lightgbm'] == lgb.__version__
    assert doc['fit_roles'] == ['FIT'] and doc['stop_roles'] == ['STOP']
    assert doc['fixed_params'] == baselines_b1.FIXED_PARAMS and doc['max_rounds'] == 1000 and doc['early_stop'] == 50


def test_load_gives_back_the_same_heads_bit_for_bit(fitted, heads_world, dropped, tmp_path):
    fitted.save(tmp_path / 'b1')
    back = baselines_b1.B1Heads.load(tmp_path / 'b1')
    assert back.dropped == fitted.dropped and back.fit_roles == fitted.fit_roles and back.stop_roles == fitted.stop_roles
    for kind in ('masked', 'open'):
        assert tuple(getattr(back, kind)) == tuple(getattr(fitted, kind))
        for name, head in getattr(fitted, kind).items():
            assert texts_of(head) == texts_of(getattr(back, kind)[name]), (kind, name)
            assert getattr(back, kind)[name].summary() == head.summary(), (kind, name)
    batch = random_batch(np.random.default_rng(1), 20, len(fitted.columns))
    assert np.array_equal(back.predict_all(batch), fitted.predict_all(batch))
    stores = (heads_world.s1, heads_world.s2, heads_world.labels)
    for kind in ('masked', 'open'):                                                      # the scores of a role too
        a, b = fitted.nll(*stores, 'DEV', kind), back.nll(*stores, 'DEV', kind)
        assert all(np.array_equal(a[k].nll, b[k].nll) and a[k].n_unscored == b[k].n_unscored for k in a)


def test_load_gives_the_heads_in_the_order_of_head_names_not_in_the_sorted_order_of_the_json(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    path = tmp_path / 'b1' / 'b1_heads.json'
    assert list(json.loads(path.read_text(encoding='utf-8'))['heads']['masked']) == sorted(HEAD_NAMES)       # the file itself is sorted
    back = baselines_b1.B1Heads.load(tmp_path / 'b1')
    assert tuple(back.masked) == HEAD_NAMES and tuple(back.open) == OPEN_NAMES
    assert list(back.summary()['masked']) == list(HEAD_NAMES)


def test_the_constructor_puts_the_heads_in_the_order_of_head_names(fitted):
    shuffled = baselines_b1.B1Heads(dict(reversed(list(fitted.masked.items()))), dict(reversed(list(fitted.open.items()))), fitted.dropped)
    assert tuple(shuffled.masked) == HEAD_NAMES and tuple(shuffled.open) == OPEN_NAMES
    with pytest.raises(ValueError, match='unknown head'):
        baselines_b1.B1Heads({'M2Rate': fitted.masked['M2Rate'], 'M9': fitted.masked['M2Mark']}, {}, fitted.dropped)
    with pytest.raises(ValueError, match='unknown head'):
        baselines_b1.B1Heads({}, {'M2Rate': fitted.masked['M2Rate']}, fitted.dropped)


def test_the_saved_files_do_not_depend_on_the_time_of_saving(fitted, tmp_path):
    fitted.save(tmp_path / 'a')
    fitted.save(tmp_path / 'b')
    files = lambda root: {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}        # noqa: E731
    assert files(tmp_path / 'a') == files(tmp_path / 'b')


def test_save_never_writes_over_a_saved_set(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    with pytest.raises(FileExistsError):
        fitted.save(tmp_path / 'b1')


def a_model_text(root: Path) -> Path:
    """The first model text of a saved set (a head folder holds its JSON and one text for each booster)."""
    return sorted((root / 'models').rglob('*.txt'))[0]


def test_load_refuses_a_changed_model_file(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    model = a_model_text(tmp_path / 'b1')
    model.write_bytes(model.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='sha256'):
        baselines_b1.B1Heads.load(tmp_path / 'b1')


def test_load_refuses_a_missing_file_and_a_wrong_schema(fitted, tmp_path):
    with pytest.raises(FileNotFoundError):
        baselines_b1.B1Heads.load(tmp_path / 'nothing')
    fitted.save(tmp_path / 'b1')
    a_model_text(tmp_path / 'b1').unlink()
    with pytest.raises(FileNotFoundError):
        baselines_b1.B1Heads.load(tmp_path / 'b1')
    fitted.save(tmp_path / 'c')
    path = tmp_path / 'c' / 'b1_heads.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['schema'] = 'b1_heads_v0'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        baselines_b1.B1Heads.load(tmp_path / 'c')


def test_load_refuses_a_design_that_is_not_the_design_of_the_module_and_a_set_that_misses_a_head(fitted, tmp_path):
    fitted.save(tmp_path / 'a')
    path = tmp_path / 'a' / 'b1_heads.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['heads']['masked']['M2Mark']['columns'][0] = 'something_else'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='not the design'):
        baselines_b1.B1Heads.load(tmp_path / 'a')
    fitted.save(tmp_path / 'b')
    path = tmp_path / 'b' / 'b1_heads.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    del doc['heads']['open']['M3Kills']
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='eight masked and the five open'):
        baselines_b1.B1Heads.load(tmp_path / 'b')
    fitted.save(tmp_path / 'c')
    path = tmp_path / 'c' / 'b1_heads.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['dropped'] = doc['dropped'] + ['nothing']
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError):
        baselines_b1.B1Heads.load(tmp_path / 'c')


def test_the_saved_json_names_the_design_columns_the_dropped_columns_and_holds_no_key(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    text = (tmp_path / 'b1' / 'b1_heads.json').read_text(encoding='utf-8')
    doc = json.loads(text)
    cols = baselines_b1.make_layout('M3Obj', 'masked', features.b1_columns(fitted.dropped)).names
    assert doc['heads']['masked']['M3Obj']['columns'] == list(cols) and doc['dropped'] == list(fitted.dropped)
    assert doc['b1_columns'] == list(fitted.columns) and KEY_PREFIX not in text
    assert doc['heads']['masked']['M3Gold']['members']['blue']['class'] == 'M3Gold'


# =====================================================================================================================================
# G-S3b B3 item 1: a head saved and loaded on its own (the runner's per-head checkpoint of the B1 fit)
# =====================================================================================================================================
HEAD_RECORD = {'code_commit': 'a' * 40, 'code_dirty': False, 'inputs_sha256': 'b' * 64, 'wall_s': 1.5}
ONE_HEADS = [('masked', 'M2Rate'), ('masked', 'M3Gold'), ('open', 'M4Conv')]      # a Poisson head, a gold head, 18 cells


@pytest.mark.parametrize('kind, name', ONE_HEADS)
def test_a_head_saved_on_its_own_loads_back_bit_for_bit_with_the_record_it_was_saved_with(fitted, tmp_path, kind, name):
    head = getattr(fitted, kind)[name]
    folder = baselines_b1.save_head(head, kind, name, fitted.columns, tmp_path / 'h', record=HEAD_RECORD)
    assert Path(folder) == tmp_path / 'h' and baselines_b1.read_head_record(tmp_path / 'h') == HEAD_RECORD
    back = baselines_b1.load_head(tmp_path / 'h', kind, name, fitted.columns)
    assert type(back) is type(head) and back.kind == kind and texts_of(back) == texts_of(head) and back.summary() == head.summary()
    names = sorted(p.name for p in (tmp_path / 'h').iterdir())
    assert baselines_b1.HEAD_JSON in names and len(names) == 1 + len(texts_of(head))              # the JSON and one text for each booster
    assert baselines_b1.read_head_record(folder) == HEAD_RECORD


def test_a_head_saved_without_a_record_has_none(fitted, tmp_path):
    baselines_b1.save_head(fitted.masked['M2Rate'], 'masked', 'M2Rate', fitted.columns, tmp_path / 'h')
    assert baselines_b1.read_head_record(tmp_path / 'h') is None


def test_the_json_of_a_head_is_written_last_so_that_a_cut_save_has_none(fitted, tmp_path, monkeypatch):
    from c_ept import store_core
    written = []
    real = store_core.write_atomic
    monkeypatch.setattr(store_core, 'write_atomic', lambda path, data: (written.append(Path(path).name), real(path, data))[1])
    baselines_b1.save_head(fitted.masked['M3Gold'], 'masked', 'M3Gold', fitted.columns, tmp_path / 'h')
    assert len(written) >= 3 and written[-1] == baselines_b1.HEAD_JSON and baselines_b1.HEAD_JSON not in written[:-1]


def test_a_head_is_never_saved_over_a_saved_head(fitted, tmp_path):
    baselines_b1.save_head(fitted.masked['M2Rate'], 'masked', 'M2Rate', fitted.columns, tmp_path / 'h')
    with pytest.raises(FileExistsError):
        baselines_b1.save_head(fitted.masked['M2Rate'], 'masked', 'M2Rate', fitted.columns, tmp_path / 'h')


def test_a_saved_head_is_refused_when_a_text_changed_the_kind_or_name_or_columns_differ_or_the_schema_is_another(fitted, tmp_path):
    def save(where):
        baselines_b1.save_head(fitted.masked['M2Rate'], 'masked', 'M2Rate', fitted.columns, tmp_path / where)
    save('text')
    model = next(p for p in (tmp_path / 'text').iterdir() if p.suffix == '.txt')
    model.write_bytes(model.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='sha256'):
        baselines_b1.load_head(tmp_path / 'text', 'masked', 'M2Rate', fitted.columns)
    save('who')
    for kind, name in (('open', 'M2Rate'), ('masked', 'M2Mark'), ('open', 'M4Conv')):
        with pytest.raises(ValueError, match='head'):
            baselines_b1.load_head(tmp_path / 'who', kind, name, fitted.columns)
    with pytest.raises(ValueError, match='not the design'):
        baselines_b1.load_head(tmp_path / 'who', 'masked', 'M2Rate', fitted.columns[:-1])
    save('schema')
    path = tmp_path / 'schema' / baselines_b1.HEAD_JSON
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['schema'] = 'b1_head_v0'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        baselines_b1.load_head(tmp_path / 'schema', 'masked', 'M2Rate', fitted.columns)
    with pytest.raises(FileNotFoundError):
        baselines_b1.load_head(tmp_path / 'nothing', 'masked', 'M2Rate', fitted.columns)


def test_a_set_of_heads_is_saved_and_loaded_by_the_functions_that_save_and_load_one_head(fitted, tmp_path, monkeypatch):
    saved, loaded = [], []
    real_save, real_load = baselines_b1.save_head, baselines_b1.load_head

    def spy_save(head, kind, name, *a, **k):
        saved.append((kind, name))
        return real_save(head, kind, name, *a, **k)

    def spy_load(folder, kind, name, *a, **k):
        loaded.append((kind, name))
        return real_load(folder, kind, name, *a, **k)
    monkeypatch.setattr(baselines_b1, 'save_head', spy_save)
    monkeypatch.setattr(baselines_b1, 'load_head', spy_load)
    fitted.save(tmp_path / 'b1')
    baselines_b1.B1Heads.load(tmp_path / 'b1')
    everything = [(kind, name) for kind in ('masked', 'open') for name in getattr(fitted, kind)]
    assert saved == everything == loaded                                                # one call for each head, in the order of the heads


def test_the_json_of_a_set_names_the_folder_and_the_hash_of_the_json_of_every_head(fitted, tmp_path):
    from c_ept import store_core
    fitted.save(tmp_path / 'b1')
    doc = json.loads((tmp_path / 'b1' / 'b1_heads.json').read_text(encoding='utf-8'))
    assert doc['schema'] == 'b1_heads_v2'
    for kind in ('masked', 'open'):
        for name, entry in doc['heads'][kind].items():
            folder = tmp_path / 'b1' / entry['dir']
            assert entry['dir'] == f'models/{kind}.{name}' and (folder / baselines_b1.HEAD_JSON).is_file()
            assert entry['head_sha256'] == store_core.sha(folder / baselines_b1.HEAD_JSON)
            assert all((tmp_path / 'b1' / f['file']).is_file() for f in entry['files'].values())      # paths from the root of the set
    assert baselines_b1.read_head_record(tmp_path / 'b1' / 'models' / 'masked.M2Rate') is None         # a set carries no record of a run


def test_a_set_whose_head_json_changed_is_refused(fitted, tmp_path):
    fitted.save(tmp_path / 'b1')
    path = tmp_path / 'b1' / 'models' / 'masked.M2Mark' / baselines_b1.HEAD_JSON
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['record'] = {'code_commit': 'other'}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='sha256'):
        baselines_b1.B1Heads.load(tmp_path / 'b1')


def test_the_module_uses_no_pickle():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {a.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    names |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not names & {'pickle', 'joblib', 'dill', 'cloudpickle', 'shelve', 'marshal'}
    assert not {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} & {'dump', 'dumps', 'load_model', 'save_model'} - {'dumps'}


# =====================================================================================================================================
# the composer
# =====================================================================================================================================
@pytest.fixture(scope='module')
def bound(fitted, heads_world):
    chain = fitted.composer_heads('masked')
    src = np.asarray(heads_world.s2.array('views/src'))
    views = np.flatnonzero(src == 0)[::-1][:30]                                           # frame-clock views, in a scrambled order
    chain.bind(heads_world.s2, views)
    return chain, views


def test_the_composer_heads_follow_the_protocol_of_ruling_r3_with_the_flags_of_decision_p6(fitted):
    chain = fitted.composer_heads('masked')
    assert isinstance(chain, heads.ComposerHeads)
    assert chain.feature_kind == 'b1' and chain.feature_kind in heads.FEATURE_KINDS
    assert chain.analytic_ok is False and chain.obj_uses_K is True and chain.conv_uses_KO is True


def test_the_flags_are_the_opposite_of_those_of_b0(fitted):
    b0 = baselines_b0._MaskedChain
    b1 = fitted.composer_heads('masked')
    assert (b0.analytic_ok, b0.obj_uses_K, b0.conv_uses_KO) == (True, False, False)
    assert (b1.analytic_ok, b1.obj_uses_K, b1.conv_uses_KO) == (False, True, True) and b0.feature_kind != b1.feature_kind


def test_the_open_heads_are_scored_not_composed(fitted):
    with pytest.raises(ValueError, match='open'):
        fitted.composer_heads('open')
    with pytest.raises(ValueError, match='masked'):
        fitted.composer_heads('both')


def test_the_chain_refuses_every_call_before_bind(fitted):
    chain = fitted.composer_heads('masked')
    for call in (lambda: chain.rate([0]), lambda: chain.mark_prob([0], [0]), lambda: chain.zone_prob([0], [0], [0]),
                 lambda: chain.size_prob([0], [0], [0], [0]), lambda: chain.kill_rate([0], [0], [0], [0], [0]),
                 lambda: chain.gold_prob([0], [0], [0], [0], [0], [[1, 1]]),
                 lambda: chain.obj_rate([0], [0], [0], [0], [0], np.ones((1, 4))), lambda: chain.row_view([0])):
        with pytest.raises(RuntimeError, match='bind'):
            call()


def test_bind_computes_the_b1_rows_of_the_views_in_order(bound, heads_world, dropped):
    chain, views = bound
    assert np.array_equal(chain.row_view(np.arange(len(views))), features.b1(heads_world.s2, views, dropped), equal_nan=True)
    assert np.array_equal(chain.row_view([2, 0]), features.b1(heads_world.s2, views[[2, 0]], dropped), equal_nan=True)


def test_bind_accepts_a_mask_and_replaces_the_bound_views(fitted, heads_world, dropped):
    chain = fitted.composer_heads('masked')
    src = np.asarray(heads_world.s2.array('views/src'))
    mask = src == 0
    chain.bind(heads_world.s2, mask)
    assert len(chain.row_view(np.arange(int(mask.sum())))) == mask.sum()
    chain.bind(heads_world.s2, np.flatnonzero(mask)[:3])
    assert len(chain._b1) == 3
    with pytest.raises(ValueError, match='rows must lie'):
        chain.rate([3])


def test_bind_refuses_event_views_a3(fitted, heads_world):
    chain = fitted.composer_heads('masked')
    src = np.asarray(heads_world.s2.array('views/src'))
    event = np.flatnonzero(src != 0)[:2]
    with pytest.raises(ValueError, match='frame-clock views'):
        chain.bind(heads_world.s2, event)
    with pytest.raises(ValueError, match='frame-clock views'):
        chain.bind(heads_world.s2, np.concatenate([np.flatnonzero(src == 0)[:2], event]))


def test_bind_reads_only_the_three_x_arrays_and_the_view_kind_a3(fitted, heads_world, monkeypatch):
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])
    chain = fitted.composer_heads('masked')
    chain.bind(heads_world.s2, np.flatnonzero(np.asarray(heads_world.s2.array('views/src')) == 0)[:5])
    assert set(reads) <= {'x/summary', 'x/players_num', 'x/geo_pp', 'views/src'}
    reads.clear()
    chain.rate([0, 1])
    chain.mark_prob([0], [3])
    chain.obj_rate([0], [3], [4], [0], [1], np.ones((1, 4)))
    assert reads == []                                                                   # after bind a head reads the bound rows only


def rows_and_codes(n, seed=5, n_rows=30):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_rows, n), rng.integers(0, 18, n), rng.integers(0, 52, n), rng.integers(-1, 10, n), rng.integers(0, 4, n),
            rng.integers(1, 7, (n, 2)), np.column_stack([rng.integers(0, 8, (n, 2)), rng.integers(0, 6000, (n, 2))]).astype(float))


def test_the_outputs_have_the_documented_shapes_and_laws(bound):
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(17)
    out = {'rate': chain.rate(rows), 'mark_prob': chain.mark_prob(rows, q), 'zone_prob': chain.zone_prob(rows, q, m),
           'size_prob': chain.size_prob(rows, q, m, z), 'kill_rate': chain.kill_rate(rows, q, m, z, s),
           'gold_prob': chain.gold_prob(rows, q, m, z, s, layer), 'obj_rate': chain.obj_rate(rows, q, m, z, s, K)}
    for name, a in out.items():
        assert a.shape == (17,) + heads.OUTPUT_WIDTHS[name] and np.isfinite(a).all(), name
    for name in ('mark_prob', 'zone_prob', 'size_prob', 'gold_prob'):
        assert (out[name] > 0).all() and out[name].sum(axis=-1) == pytest.approx(np.ones(out[name].shape[:-1]), abs=1e-12), name
    for name in ('rate', 'kill_rate', 'obj_rate'):
        assert (out[name] >= 0).all(), name
    m4 = features.m4_masked_inputs(chain.row_view(rows), q, m, z, K, np.zeros((17, 18)))
    conv = chain.conv_rate(m4)
    assert conv.shape == (17, 18) and (conv >= 0).all()


def design_oracle(fitted_head, view_row, **pieces):
    return design_row(baselines_b1.SLOTS[(fitted_head.name, fitted_head.kind)], view_row, **pieces)


def test_the_chain_equals_the_heads_on_designs_built_row_by_row(bound, fitted):
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(19)
    base = chain.row_view(np.arange(len(views)))
    h = fitted.masked
    for j in range(19):
        b, codes = base[rows[j]], {'q': q[j], 'm': m[j], 'z': z[j], 's': s[j]}
        one = lambda head, **kw: head.predict(design_oracle(head, b, **kw)[None, :])[0]                      # noqa: E731
        assert np.array_equal(chain.rate(rows[j:j + 1])[0], h['M2Rate'].predict(np.stack(
            [design_oracle(h['M2Rate'], b, codes={'q': k}) for k in range(18)])))
        assert np.array_equal(chain.mark_prob(rows[j:j + 1], q[j:j + 1])[0], one(h['M2Mark'], codes=codes))
        assert np.array_equal(chain.zone_prob(rows[j:j + 1], q[j:j + 1], m[j:j + 1])[0], one(h['M2Zone'], codes=codes))
        assert np.array_equal(chain.size_prob(rows[j:j + 1], q[j:j + 1], m[j:j + 1], z[j:j + 1])[0], one(h['M3Size'], codes=codes))
        kills = chain.kill_rate(rows[j:j + 1], q[j:j + 1], m[j:j + 1], z[j:j + 1], s[j:j + 1])[0]
        assert np.array_equal(kills, one(h['M3Kills'], codes=codes))
        gold = chain.gold_prob(rows[j:j + 1], q[j:j + 1], m[j:j + 1], z[j:j + 1], s[j:j + 1], layer[j:j + 1])[0]
        for t in (0, 1):
            member = h['M3Gold'].member(t)
            assert np.array_equal(gold[t], member.predict(design_row(baselines_b1.SLOTS[('M3Gold', 'masked')], b,
                                                                     codes={**codes, 'layer': layer[j, t]})[None, :])[0])
        k_thousands = K[j].copy()
        k_thousands[2:] /= 1000.0
        assert np.array_equal(chain.obj_rate(rows[j:j + 1], q[j:j + 1], m[j:j + 1], z[j:j + 1], s[j:j + 1], K[j:j + 1])[0],
                              one(h['M3Obj'], codes=codes, K=k_thousands))


def test_the_object_head_reads_k_and_the_conversion_head_reads_k_and_o_decision_p6(bound, fitted, monkeypatch):
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(6)
    d = len(chain._columns)
    captured = []
    for name in ('M3Obj', 'M4Conv'):
        real = fitted.masked[name].predict
        monkeypatch.setattr(fitted.masked[name], 'predict',
                            lambda X, real=real, name=name: (captured.append((name, np.array(X))), real(X))[1])
    chain.obj_rate(rows, q, m, z, s, K)
    chain.obj_rate(rows, q, m, z, s, K * 2)
    (_, a), (_, b) = captured
    assert np.array_equal(a[:, d:d + 2], K[:, :2]) and np.allclose(a[:, d + 2:d + 4], K[:, 2:] / 1000.0, rtol=1e-6)
    assert not np.array_equal(a, b) and np.array_equal(b[:, d:d + 2], 2 * K[:, :2])         # K goes into the design: the head reads it
    del captured[:]
    O = np.random.default_rng(3).integers(0, 3, (6, 18))
    m4 = features.m4_masked_inputs(chain.row_view(rows), q, m, z, K, O)
    chain.conv_rate(m4)
    (_, c), = captured
    assert np.array_equal(c[:, d + 7:], O.astype(np.float32)) and np.array_equal(c[:, d + 3:d + 5], K[:, :2].astype(np.float32))
    assert np.array_equal(c[:, d + 2], z + 1)


def test_the_conversion_head_depends_on_k_and_o_through_its_trees(bound, fitted):
    """A booster fitted on the world has a split on a K or an O column somewhere; the answer of the chain moves with them."""
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(40, seed=9)
    O1, O2 = np.zeros((40, 18)), np.full((40, 18), 5)
    a = chain.conv_rate(features.m4_masked_inputs(chain.row_view(rows), q, m, z, K, O1))
    b = chain.conv_rate(features.m4_masked_inputs(chain.row_view(rows), q, m, z, K * 3, O2))
    d = len(chain._columns)
    used = set()
    for booster in fitted.masked['M4Conv'].boosters_:
        if booster is not None:
            used |= {i for i, v in enumerate(booster.feature_importance('split')) if v > 0}
    assert any(i >= d + 3 for i in used) and not np.array_equal(a, b)


def test_the_chain_is_deterministic_and_changes_no_head(bound, fitted):
    chain, views = bound
    before = {name: texts_of(h) for name, h in fitted.masked.items()}
    rows, q, m, z, s, layer, K = rows_and_codes(8)
    assert np.array_equal(chain.rate(rows), chain.rate(rows))
    assert np.array_equal(chain.obj_rate(rows, q, m, z, s, K), chain.obj_rate(rows, q, m, z, s, K))
    assert before == {name: texts_of(h) for name, h in fitted.masked.items()}


def test_the_chunks_of_the_design_fill_and_of_the_prediction_change_no_number(heads_world, dropped, fitted, bound, monkeypatch):
    """The designs are filled and the boosters asked a chunk of rows at a time (the 8.2 M rows of M2Rate are never gathered at once); a
    chunk of 5 rows and a prediction of 7 design rows give exactly what one chunk gives."""
    keys = (('M2Rate', 'masked'), ('M3Gold', 'masked'), ('M4Conv', 'masked'), ('M3Obj', 'open'))
    whole = {key: rows_of(heads_world, dropped, *key).X for key in keys}
    batch = random_batch(np.random.default_rng(2), 12, len(fitted.columns))
    answer = fitted.predict_all(batch)
    team = rows_of(heads_world, dropped, 'M3Gold', 'masked').team
    one_shot = (fitted.masked['M2Rate'].predict(whole[keys[0]]), fitted.masked['M3Gold'].predict(whole[keys[1]], team))
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(16)
    expected = (chain.rate(rows), chain.size_prob(rows, q, m, z), chain.gold_prob(rows, q, m, z, s, layer))
    monkeypatch.setattr(baselines_b1, '_CHUNK', 5)
    monkeypatch.setattr(baselines_b1, '_PREDICT_CHUNK', 7)
    for key in keys:
        assert np.array_equal(rows_of(heads_world, dropped, *key).X, whole[key], equal_nan=True), key
    assert np.array_equal(fitted.predict_all(batch), answer)
    assert np.array_equal(fitted.masked['M2Rate'].predict(whole[keys[0]]), one_shot[0])
    assert np.array_equal(fitted.masked['M3Gold'].predict(whole[keys[1]], team), one_shot[1])
    got = (chain.rate(rows), chain.size_prob(rows, q, m, z), chain.gold_prob(rows, q, m, z, s, layer))
    assert all(np.array_equal(a, b) for a, b in zip(expected, got))


def test_the_chain_answers_no_rows_with_empty_arrays_of_the_right_shape(bound):
    chain, views = bound
    e = np.empty(0, dtype=np.int64)
    assert chain.rate(e).shape == (0, 18) and chain.mark_prob(e, e).shape == (0, 52) and chain.zone_prob(e, e, e).shape == (0, 10)
    assert chain.size_prob(e, e, e, e).shape == (0, 4) and chain.kill_rate(e, e, e, e, e).shape == (0, 2)
    assert chain.gold_prob(e, e, e, e, e, np.empty((0, 2), dtype=np.int64)).shape == (0, 2, 31)
    assert chain.obj_rate(e, e, e, e, e, np.empty((0, 4))).shape == (0, 18)
    assert chain.conv_rate(np.empty((0, len(chain._columns) + 25))).shape == (0, 18)


def test_the_chain_refuses_rows_and_categories_that_are_not_integers_or_out_of_range(bound):
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(4)
    with pytest.raises(TypeError):
        chain.mark_prob(rows.astype(float), q)
    with pytest.raises(TypeError):
        chain.mark_prob(rows, q.astype(float))
    for call in (lambda: chain.rate([len(views)]), lambda: chain.rate([-1]), lambda: chain.mark_prob(rows, q + 18),
                 lambda: chain.zone_prob(rows, q, m + 52), lambda: chain.size_prob(rows, q, m, z + 10),
                 lambda: chain.kill_rate(rows, q, m, z, s + 4), lambda: chain.gold_prob(rows, q, m, z, s, layer + 6),
                 lambda: chain.mark_prob(rows, q[:2]),
                 lambda: chain.obj_rate(rows, q, m, z, s, K[:, :3]), lambda: chain.obj_rate(rows, q, m, z, s, -K),
                 lambda: chain.gold_prob(rows, q, m, z, s, layer[:, :1])):
        with pytest.raises(ValueError):
            call()


def test_conv_rate_checks_its_input(bound):
    chain, views = bound
    d = len(chain._columns)
    good = np.zeros((2, d + 25), dtype=np.float32)
    assert chain.conv_rate(good).shape == (2, 18)
    with pytest.raises(ValueError, match='shape'):
        chain.conv_rate(good[:, :-1])
    bad = good.copy()
    bad[0, d] = 0.5
    with pytest.raises(ValueError, match='whole numbers'):
        chain.conv_rate(bad)


def test_predict_all_is_every_masked_head_on_the_rows_of_a_batch(fitted):
    batch = random_batch(np.random.default_rng(7), 12, len(fitted.columns))
    out = fitted.predict_all(batch)
    assert out.shape == (12, sum(baselines_b1.PREDICT_ALL_WIDTHS.values())) == (12, 184) and np.isfinite(out).all()
    assert np.array_equal(out, fitted.predict_all(batch))
    start = 0
    parts = {}
    for name, width in baselines_b1.PREDICT_ALL_WIDTHS.items():
        parts[name] = out[:, start:start + width]
        start += width
    chain = fitted.composer_heads('masked')
    chain._set_rows(batch.row_view)
    rows = np.arange(12)
    assert np.array_equal(parts['rate'], chain.rate(rows)) and np.array_equal(parts['mark_prob'], chain.mark_prob(rows, batch.q))
    assert np.array_equal(parts['obj_rate'], chain.obj_rate(rows, batch.q, batch.m, batch.z, batch.s, batch.K))


def test_a_batch_is_checked(fitted):
    b = random_batch(np.random.default_rng(1), 5, len(fitted.columns))
    with pytest.raises(ValueError):
        fitted.predict_all(b._replace(K=b.K[:, :3]))
    with pytest.raises(ValueError):
        fitted.predict_all(b._replace(K=-b.K))
    with pytest.raises(ValueError):
        fitted.predict_all(b._replace(row_view=b.row_view[:, :38]))
    with pytest.raises(TypeError):
        fitted.predict_all(b.row_view)


# =====================================================================================================================================
# the heads as blocks: the numeric part and the raw categories (the signature of B0 for the composer)
# =====================================================================================================================================
def test_predict_blocks_is_predict_on_the_design_of_the_pieces(fitted, bound):
    chain, views = bound
    rows, q, m, z, s, layer, K = rows_and_codes(9)
    feat = chain.row_view(rows)
    h = fitted.masked
    by_hand = baselines_b1._assemble(('feat', 'q'), feat, None, {'q': q})
    assert np.array_equal(h['M2Mark'].predict_blocks(feat, q=q), h['M2Mark'].predict(by_hand))
    assert np.array_equal(h['M3Kills'].predict_blocks(feat, q=q, m=m, z=z, s=s), chain.kill_rate(rows, q, m, z, s))
    k = K.copy()
    k[:, 2:] /= 1000.0
    assert np.array_equal(h['M3Obj'].predict_blocks(np.hstack([feat, k]), q=q, m=m, z=z, s=s), chain.obj_rate(rows, q, m, z, s, K))
    O = np.random.default_rng(1).integers(0, 3, (9, 18))
    m4 = features.m4_masked_inputs(feat, q, m, z, K, O)
    d = feat.shape[1]
    numeric = np.hstack([m4[:, :d], m4[:, d + 3:]])                                       # the b1 row, then K and O
    assert np.array_equal(h['M4Conv'].predict_blocks(numeric, q=q, m=m, z=z), chain.conv_rate(m4))
    gold = h['M3Gold'].predict_blocks(feat, np.zeros(9, dtype=int), q=q, m=m, z=z, s=s, layer=layer[:, 0])
    assert np.array_equal(gold, chain.gold_prob(rows, q, m, z, s, layer)[:, 0])


def test_predict_blocks_checks_what_it_is_given(fitted, bound):
    chain, views = bound
    feat = chain.row_view(np.arange(3))
    head = fitted.masked['M2Mark']
    for call in (lambda: head.predict_blocks(feat[:, :-1], q=[0, 1, 2]), lambda: head.predict_blocks(feat, q=[0, 1, 2], m=[0, 0, 0]),
                 lambda: head.predict_blocks(feat), lambda: head.predict_blocks(feat, q=[0, 1]),
                 lambda: head.predict_blocks(feat[0], q=[0])):
        with pytest.raises(ValueError):
            call()
    raw = baselines_b1.M2Mark(params=SMALL).fit(*planted_classes(seed=2, n=300), categorical=(2,))
    with pytest.raises(ValueError, match='without a layout'):
        raw.predict_blocks(np.zeros((1, 3)), q=[0])


# =====================================================================================================================================
# B2: the next-minute onset classifier
# =====================================================================================================================================
def test_b2_target_boundaries_by_hand(toy_s1):
    y = baselines_b1.b2_target(toy_s1, np.arange(TOY_N_ROWS))
    assert y.tolist() == [True, False, False, True, True, False, True, False, False, True, False] and int(y.sum()) == TOY_N_POSITIVE
    assert not y[1] and y[3]                                                       # bin 6 is the second minute, bin 5 the last of the first
    assert baselines_b1.b2_target(toy_s1, np.array([4, 1, 9])).tolist() == [True, False, True]       # any order, any subset
    assert baselines_b1.b2_target(toy_s1, np.array([], dtype=np.int64)).shape == (0,) and baselines_b1.B2_BINS == 6
    for bad in (np.array([TOY_N_ROWS]), np.array([-1])):
        with pytest.raises(ValueError, match='rows must lie'):
            baselines_b1.b2_target(toy_s1, bad)
    with pytest.raises(TypeError):
        baselines_b1.b2_target(toy_s1, np.array([0.0]))


def test_b2_valid_rows_are_the_stop_frame_rows_and_a_holdout_role_is_refused(toy_s1):
    mask = baselines_b1.valid_rows(toy_s1)
    assert mask.shape == (TOY_N_ROWS,) and np.flatnonzero(mask).tolist() == [6, 7, 8]
    assert np.flatnonzero(baselines_b1.valid_rows(toy_s1, 'DEV')).tolist() == [9, 10]
    for role in ('TEST', 'EXTERNAL', 'CAL'):
        with pytest.raises(ValueError):
            baselines_b1.valid_rows(toy_s1, role)


def planted_binary(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.integers(0, 3, n)]).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0] - 0.4)))
    return X, rng.random(n) < p


def test_b2_recovers_a_planted_probability():
    X, y = planted_binary(seed=1, n=10_000)
    m = baselines_b1.B2(params={**SMALL, 'num_leaves': 15, 'num_iterations': 80}).fit(X, y, categorical=(2,))
    grid = np.array([[-2.0, 0, 0], [0.0, 0, 1], [2.0, 0, 2]], dtype=np.float32)
    truth = 1.0 / (1.0 + np.exp(-(1.2 * grid[:, 0] - 0.4)))
    assert m.predict(grid) == pytest.approx(truth, abs=0.08)


def test_b2_is_a_probability_mixed_with_the_uniform_law_and_scores_by_the_nll_and_the_brier_score():
    X, y = planted_binary(seed=2, n=3000)
    m = baselines_b1.B2(params=SMALL).fit(X, y, categorical=(2,))
    p = m.predict(X)
    assert p.shape == (3000,) and (p >= MIX / 2).all() and (p <= 1 - MIX / 2).all()
    assert m.nll(X, y) == pytest.approx(-np.log(np.where(y, p, 1 - p)), rel=1e-12)
    assert m.brier(X, y) == pytest.approx((p - y) ** 2, rel=1e-12) and 0.0 <= m.brier(X, y).mean() < 0.25
    for bad in (np.arange(3000) % 3, np.ones(2999), y.astype(float) + 0.5):
        with pytest.raises(ValueError):
            m.nll(X, bad)


def test_a_b2_target_with_one_value_gives_the_smoothed_frequency_and_no_booster():
    X, _ = planted_binary(seed=2, n=200)
    for value, expected in ((False, 0.5 / 201), (True, 200.5 / 201)):
        m = baselines_b1.B2(params=SMALL).fit(X, np.full(200, value))
        p = m.predict(X)
        assert m.boosters_ == [None] and p == pytest.approx(np.full(200, (1 - MIX) * expected + MIX / 2), rel=1e-12) and (p > 0).all()


def test_b2_early_stopping_uses_the_validation_rows_and_is_bit_reproducible():
    rng = np.random.default_rng(1)
    X, Xv = rng.normal(size=(1500, 4)).astype(np.float32), rng.normal(size=(1500, 4)).astype(np.float32)
    y, yv = rng.random(1500) < 0.3, rng.random(1500) < 0.3                               # noise
    params = {'num_leaves': 15, 'min_data_in_leaf': 5, 'num_iterations': 200, 'early_stopping_round': 10, 'learning_rate': 0.3}
    a = baselines_b1.B2(params=params).fit(X, y, valid=(Xv, yv))
    b = baselines_b1.B2(params=params).fit(X, y, valid=(Xv, yv))
    info = a.info_[0]
    assert info['stopped_early'] and info['rounds_run'] - info['best_iteration'] == 10
    assert a.summary()['num_trees'] == [info['best_iteration']]
    assert a.state()[0] == b.state()[0] and np.array_equal(a.predict(Xv), b.predict(Xv))
    assert a.n_valid_ == 1500 and a.n_fit_ == 1500


def test_b2_refuses_what_cannot_be_trusted():
    X, y = planted_binary(seed=2, n=300)
    for bad in (np.arange(300) % 3, y[:-1], y.astype(str)):
        with pytest.raises(ValueError):
            baselines_b1.B2(params=SMALL).fit(X, bad)
    with pytest.raises(ValueError, match='valid must be'):
        baselines_b1.B2(params=SMALL).fit(X, y, valid=(X,))
    with pytest.raises(ValueError, match='validation rows'):
        baselines_b1.B2(params=SMALL).fit(X, y, valid=(X[:, :2], y))
    with pytest.raises(RuntimeError, match='not fitted'):
        baselines_b1.B2().predict(X)
    with pytest.raises(ValueError, match='dropped columns'):
        baselines_b1.B2(params=SMALL).fit(X, y).score_role(None, None, 'DEV')


def oracle_b2_target(oracle, rows):
    return [any(oracle.on_row[i] == r and oracle.on_bin[i] < 6 for i in range(len(oracle.on_row))) for r in rows]


@pytest.fixture(scope='module')
def b2(heads_world, dropped):
    return baselines_b1.B2.fit_stores(heads_world.s1, heads_world.s2, dropped, params=SMALL)


def test_b2_is_fitted_on_the_frame_rows_of_the_fit_matches_and_stopped_on_the_stop_frame_rows(heads_world, dropped, oracle, monkeypatch):
    seen = spy_dataset(monkeypatch)
    model = baselines_b1.B2.fit_stores(heads_world.s1, heads_world.s2, dropped, params=SMALL)
    fit_rows, stop_rows = oracle.frame_rows('FIT'), oracle.frame_rows('STOP')
    train, valid = seen
    d = len(features.b1_columns(dropped))
    assert train['data'].shape == (len(fit_rows), d) and valid['data'].shape == (len(stop_rows), d)       # frame rows only: no bin column
    assert valid['data'].shape[0] == baselines_b1.valid_rows(heads_world.s1).sum()              # the early-stopping rows of the module
    assert train['kwargs']['label'].tolist() == [int(v) for v in oracle_b2_target(oracle, fit_rows)]
    assert valid['kwargs']['label'].tolist() == [int(v) for v in oracle_b2_target(oracle, stop_rows)]
    first = rows_b1_frame(heads_world, dropped, fit_rows)
    assert np.array_equal(train['data'], first, equal_nan=True)
    assert model.fit_roles_ == ('FIT',) and model.stop_roles_ == ('STOP',) and model.dropped_ == tuple(dropped)
    assert model.n_fit_ == len(fit_rows) and model.n_valid_ == len(stop_rows)


def rows_b1_frame(heads_world, dropped, frame_rows):
    oracle = HeadsOracle(heads_world)
    return features.b1(heads_world.s2, [oracle.view(0, r) for r in frame_rows], dropped)


def test_b2_scores_the_frame_rows_of_a_role(b2, fitted, heads_world, oracle, dropped):
    out = b2.score_role(heads_world.s1, heads_world.s2, 'DEV')
    rows = oracle.frame_rows('DEV')
    m2 = fitted.nll(heads_world.s1, heads_world.s2, heads_world.labels, 'DEV')['M2Rate']
    assert sorted(set(m2.row.tolist())) == sorted(out.row.tolist()) == rows               # M2 and B2 forecast the same frame rows
    assert out.row.tolist() == rows and out.y.tolist() == oracle_b2_target(oracle, rows) and out.match.tolist() == [
        int(oracle.fr_match[r]) for r in rows]
    assert np.array_equal(out.p, b2.predict(rows_b1_frame(heads_world, dropped, rows))) and ((out.p > 0) & (out.p < 1)).all()
    assert out.nll() == pytest.approx(-np.log(np.where(out.y, out.p, 1 - out.p)), rel=1e-12)
    assert out.brier() == pytest.approx((out.p - out.y) ** 2)
    sc = out.scores()
    assert type(sc) is head_rows.HeadScores and np.isfinite(sc.mean()) and sc.n_unscored == 0 and len(sc.nll) == len(rows)
    for role in ('TEST', 'EXTERNAL', 'CAL'):
        with pytest.raises(ValueError):
            b2.score_role(heads_world.s1, heads_world.s2, role)


def test_b2_fit_calls_the_e6_check_and_refuses_roles_before_any_read(heads_world, dropped, monkeypatch):
    calls = []
    baselines_b1.B2.fit_stores(heads_world.s1, heads_world.s2, dropped, params=SMALL, check=lambda: calls.append(1))
    assert len(calls) == 3                                                         # before the reads, before the booster, in the cell loop
    reads = []
    real = Store.array
    monkeypatch.setattr(Store, 'array', lambda self, name: (reads.append(name), real(self, name))[1])

    def refuse():
        raise RuntimeError('E6: stop')
    with pytest.raises(RuntimeError, match='E6'):
        baselines_b1.B2.fit_stores(heads_world.s1, heads_world.s2, dropped, params=SMALL, check=refuse)
    assert reads == []
    for roles in ({'fit': 'TEST', 'stop': 'STOP'}, {'fit': 'FIT', 'stop': 'FIT'}, {'fit': 'FIT'}):
        with pytest.raises(ValueError):
            baselines_b1.B2.fit_stores(heads_world.s1, heads_world.s2, dropped, roles=roles, params=SMALL)


def test_b2_uses_the_fit_and_stop_matches_only(heads_world, b2, dropped, tmp_path):
    copy = tmp_path / 'scrambled'
    for part in ('s1', 's2'):
        shutil.copytree(heads_world.s1.root.parent / part, copy / part)
    s1, s2 = Store(copy / 's1'), Store(copy / 's2')
    rng = np.random.default_rng(0)
    role = np.asarray(s1.array('matches/role'))
    used = lambda r: np.isin(r, (C.ROLES.index('FIT'), C.ROLES.index('STOP')))                  # noqa: E731
    used_row, used_view = used(role[np.asarray(s1.array('frame_rows/match'))]), used(role[np.asarray(s2.array('views/match'))])
    used_onset = used_row[np.asarray(s1.array('onsets/row'))]
    for name, keep in (('onsets/bin', used_onset), ('x/summary', used_view), ('x/players_num', used_view), ('x/geo_pp', used_view)):
        store = s2 if name.startswith('x/') else s1
        a = np.load(store.path(name), mmap_mode='r+')
        a[~keep] = rng.permutation(a[~keep].reshape(-1)).reshape(a[~keep].shape) if a.dtype.kind != 'f' else a[~keep][::-1] + 1.0
        a.flush()
        del a
    again = baselines_b1.B2.fit_stores(s1, s2, dropped, params=SMALL)
    assert again.state()[0] == b2.state()[0]
    del s1, s2, again
    gc.collect()


def test_b2_save_and_load_give_the_same_model_bit_for_bit(b2, heads_world, tmp_path):
    b2.save(tmp_path / 'b2')
    written = sorted(p.relative_to(tmp_path / 'b2').as_posix() for p in (tmp_path / 'b2').rglob('*') if p.is_file())
    assert written == ['b2.json', 'models/b2.txt']
    back = baselines_b1.B2.load(tmp_path / 'b2')
    a, b = b2.score_role(heads_world.s1, heads_world.s2, 'DEV'), back.score_role(heads_world.s1, heads_world.s2, 'DEV')
    assert np.array_equal(a.p, b.p) and np.array_equal(a.y, b.y) and back.state()[0] == b2.state()[0] and back.summary() == b2.summary()
    assert back.dropped_ == b2.dropped_ and back.fit_roles_ == ('FIT',) and back.stop_roles_ == ('STOP',)
    with pytest.raises(FileExistsError):
        b2.save(tmp_path / 'b2')
    baselines_b1.B2.save(b2, tmp_path / 'again')
    assert (tmp_path / 'again' / 'models' / 'b2.txt').read_bytes() == (tmp_path / 'b2' / 'models' / 'b2.txt').read_bytes()
    (tmp_path / 'b2' / 'models' / 'b2.txt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sha256'):
        baselines_b1.B2.load(tmp_path / 'b2')


def test_a_constant_b2_survives_save_and_load(tmp_path):
    X, _ = planted_binary(seed=2, n=100)
    m = baselines_b1.B2(params=SMALL).fit(X, np.zeros(100, dtype=bool))
    m.save(tmp_path / 'c')
    back = baselines_b1.B2.load(tmp_path / 'c')
    assert np.array_equal(back.predict(X), m.predict(X)) and back.boosters_ == [None] and back.dropped_ is None


def test_load_refuses_a_b2_file_that_is_not_one(fitted, b2, tmp_path):
    b2.save(tmp_path / 'b2')
    path = tmp_path / 'b2' / 'b2.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['model'] = 'B1'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        baselines_b1.B2.load(tmp_path / 'b2')


# =====================================================================================================================================
# structure
# =====================================================================================================================================
def test_the_module_imports_no_runner_no_cohort_no_raw_reader_no_value_and_no_other_model_at_module_level():
    imported = module_level_imports(SOURCE)
    forbidden = {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.cohort', 'c_ept.runlog', 'c_ept.terrain', 'c_ept.store_build', 'c_ept.reader',
                 'c_ept.tokens', 'c_ept.value', 'c_ept.compose', 'c_ept.units', 'c_ept.targets', 'c_ept.towertier', 'c_ept.baselines_b0',
                 'sklearn'}
    assert not imported & forbidden, imported & forbidden
    assert {'lightgbm', 'c_ept.head_rows', 'c_ept.features'} <= imported
    assert not all_imports(SOURCE) & {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.reader', 'c_ept.baselines_b0', 'pickle'}


def test_the_module_names_only_the_arrays_it_is_documented_to_read():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.count('/') == 1 and ' ' not in node.value and node.value.split('/')[0].isidentifier()}
    assert names == {'onsets/row', 'onsets/bin', 'frame_rows/t0', 'views/src'}               # b2_target and the frame-clock check of bind
    text = SOURCE.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text
    for forbidden in ("'kills/", "'objectives/", "'frames/", "'units/", "'labels/", "'event_rows/"):
        assert forbidden not in text, forbidden                                              # the rows come from head_rows, not from here


def test_the_training_rows_call_the_input_builder_through_head_rows_only():
    """The masked M4 training design is made by ``head_rows`` from ``features.m4_masked_inputs``; this module calls that function in
    ``B1Heads.predict_all`` alone (the batch of rows that the composer would hand over) and otherwise only shifts the zone of its matrix."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    callers = []
    for scope in ast.walk(tree):
        if isinstance(scope, ast.FunctionDef):
            for node in ast.walk(scope):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'm4_masked_inputs':
                    callers.append(scope.name)
    assert set(callers) == {'predict_all'}


def test_a_wrong_role_and_a_bad_drop_list_never_reach_a_key(heads_world, dropped):
    s1, s2, labels = heads_world.s1, heads_world.s2, heads_world.labels
    for call in (lambda: baselines_b1.HeadRows(s1, s2, labels, dropped, 'TEST'),
                 lambda: baselines_b1.HeadRows(s1, s2, labels, ('nope',), 'FIT')):
        with pytest.raises(ValueError) as raised:
            call()
        assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)
