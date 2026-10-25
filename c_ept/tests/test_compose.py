"""Tests of ``c_ept.compose``: the composition of V_start from the heads at t0 only (stage S3 Task 9).

No run folder, raw file or catalog is read. The stores are the toy S1 of the value toys (made by the real extraction) with a toy S2, the
heads world of ``s3_toys`` for the real B0 and B1 heads, and arrays kept in memory. Truth comes from six places:

    brief tests    the six tests of the task brief, with their names and assertions as written there
    note 03 E      the hand-worked row of math-control note 03 section E (32.16, and 22.12 when the game ends at 1,075 s) and the formula
                   behind it, written out again here for every toy row and horizon
    oracle         ``oracle_expect``: the closed form computed a second way (a Python loop over the bin and the mark, the zone and the size
                   asked for as one block, the Poisson law written with ``math``) for heads whose laws differ in every argument
    hand models    a tier law and prices chosen by hand, so that the price of a predicted tower is known for each team, bin and state
    statistics     the draws of ``simulate_onsets`` compared with the laws that the heads themselves give (chi-square for the categories,
                   z for the Poisson counts), with fixed seeds
    rulings        A3 (the table holds t0 information, the source names no event array, the stores are not kept), P10 (conversion
                   exposure), Review Focus 3 (no onsets and no conversions without exposure), R3 (the protocol), the determinism rule
                   (a row's draws depend on the seed and its global row number only)
"""
import ast
import gc
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from c_ept.valuation import compose
from c_ept.preprocessing import features
from c_ept.valuation import flowprice
from c_ept.models import heads
from c_ept.valuation import towertier
from c_ept import config as C
from c_ept.tests.conftest import all_imports, module_level_imports
from c_ept.tests.s3_toys import (KEY_PREFIX, TOY_B1_LIKE_HEADS, TOY_COMPOSE_PRICES, TOY_EXPECTED_V60, TOY_EXPECTED_V60_CUT,  # noqa: F401
                                 TOY_GOLD_REPS, TOY_ROW_SPECS, TOY_ROWS, TOY_T0_S, CountingTier, CountingToyHeads, EventToyHeads,
                                 FileLogToyHeads, KOToyHeads, ProcessProbeHeads, RichToyHeads, ToyHeads, build_toy_s2, heads_world,
                                 rows_with_zero_tail_from_bin, toy_composer, toy_row_table, toy_tier_model, toy_value_s1)

SOURCE = Path(compose.__file__)
FORBIDDEN = ('units/', 'onsets/', 'kills/', 'objectives/', 'event_rows/')
BLUE, RED = 0, 1
TOWER_CELL = 2 * C.O_KINDS.index('tower')                   # the blue tower cell; the red one is the next
H_BINS = (6, 12, 18)


# =====================================================================================================================================
# the helpers of the task brief
# =====================================================================================================================================
def simulated_onset_bins(c):
    """The bin of every simulated onset of every simulation of every row of the composer ``c``."""
    return np.concatenate([c.simulate_onsets(i).q for i in range(len(c.rows))])


def run_blocks(rows, block, workers):
    """Everything the composer says about ``rows`` (closed form and simulation), as one vector, for a block size and a worker count."""
    c = toy_composer(S=300, seed=3)
    e = c.expect(rows, block=block, workers=workers)
    r = c.simulate(rows, block=block, workers=workers)
    return np.concatenate([e.ravel(), r.mean.ravel(), r.sd.ravel(), r.q19.ravel()])


# =====================================================================================================================================
# the six tests of the task brief
# =====================================================================================================================================
def test_one_row_expectation_matches_toy_note():                       # 검산 노트 03
    c = toy_composer(S=1000, seed=1)
    assert c.expect([0])[0, 0] == pytest.approx(TOY_EXPECTED_V60)


def test_simulation_mean_agrees_with_expectation():
    c = toy_composer(S=20_000, seed=2)
    r = c.simulate(TOY_ROWS)
    assert (np.abs(r.mean[:, 0] - c.expect(TOY_ROWS)[:, 0]) < 3 * r.sd[:, 0] / np.sqrt(20_000)).all()


def test_no_onsets_in_zero_exposure_bins():
    assert simulated_onset_bins(toy_composer(rows=rows_with_zero_tail_from_bin(12))).max() < 12


def test_conversion_exposure_is_cut_at_game_end():
    assert compose.conv_exposure_s(1_000_000, 5, 1_075_000) == 0.0
    assert compose.conv_exposure_s(1_000_000, 0, 1_075_000) == pytest.approx(50.0)


def test_results_do_not_depend_on_block_size_or_workers():
    assert np.array_equal(run_blocks(TOY_ROWS, block=3, workers=1), run_blocks(TOY_ROWS, block=1000, workers=2))


def test_expect_refuses_heads_without_closed_form():
    with pytest.raises(ValueError):
        toy_composer(heads=TOY_B1_LIKE_HEADS).expect(TOY_ROWS)


# =====================================================================================================================================
# note 03 section E: the closed form of the toy rows, by the formula of the note
# =====================================================================================================================================
def note_expected(t0_s, T_s, n_bins):
    """E[V_H] of the toy of note 03 E for the first ``n_bins`` bins: sum of 0.01 x (seconds of the bin) x 0.2 x (220 + 0.8 c)."""
    total = 0.0
    for q in range(n_bins):
        seconds = min(max(min(t0_s + 180.0, T_s - 30.0) - (t0_s + 10.0 * q), 0.0), 10.0)
        c = max(0.0, min(60.0, T_s - (t0_s + (q + 0.5) * 10.0 + 20.0)))
        total += 0.01 * seconds * 0.2 * (220.0 + 0.8 * c)
    return total


def test_the_two_numbers_of_note_03_are_the_formula_of_the_note():
    assert note_expected(1000.0, 5000.0, 6) == pytest.approx(TOY_EXPECTED_V60)
    assert note_expected(1000.0, 1075.0, 6) == pytest.approx(TOY_EXPECTED_V60_CUT)
    assert TOY_T0_S == 1000.0


def test_the_cut_row_of_note_03_gives_22_12():
    assert toy_composer().expect([1])[0, 0] == pytest.approx(TOY_EXPECTED_V60_CUT)


def test_every_toy_row_and_horizon_follows_the_formula_of_the_note():
    got = toy_composer().expect(TOY_ROWS)
    want = np.array([[note_expected(t0, T, n) for n in H_BINS] for t0, T in TOY_ROW_SPECS])
    assert got.shape == (len(TOY_ROWS), 3)
    assert got == pytest.approx(want, rel=1e-12)


def test_the_far_row_is_the_same_in_every_bin_so_the_horizons_scale_with_the_number_of_bins():
    row = toy_composer().expect([0])[0]
    assert row == pytest.approx(TOY_EXPECTED_V60 * np.array([1.0, 2.0, 3.0]), rel=1e-12)


def test_the_cut_row_adds_nothing_after_its_last_bin():
    row = toy_composer().expect([1])[0]
    assert row[0] == pytest.approx(row[1], rel=1e-14) and row[1] == pytest.approx(row[2], rel=1e-14)       # bins 5 .. 17 have no exposure


def test_equal_marks_make_the_expectation_zero():
    """Note 03 E, 'try it yourself': with pi = (0.5, 0.5) the bracket is 0, so the expectation is 0."""
    class Even(ToyHeads):
        def mark_prob(self, rows, q):
            p = np.zeros((len(self._rows(rows)), C.N_MARKS))
            p[:, 0] = p[:, 1] = 0.5
            return p
    assert toy_composer(heads=Even()).expect([0])[0] == pytest.approx(0.0, abs=1e-12)


# =====================================================================================================================================
# conv_exposure_s (decision P10)
# =====================================================================================================================================
def test_conversion_exposure_of_the_bins_of_note_03():
    got = [compose.conv_exposure_s(1_000_000, q, 1_075_000) for q in range(6)]
    assert got == [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]


def test_conversion_exposure_is_capped_at_the_conversion_window():
    assert compose.conv_exposure_s(1_000_000, 0, 5_000_000) == 60.0
    assert compose.conv_exposure_s(1_000_000, 0, 1_085_000) == 60.0                      # delta_hat + 60 s = T exactly
    assert compose.conv_exposure_s(1_000_000, 0, 1_084_999) == pytest.approx(59.999)


def test_conversion_exposure_is_never_negative():
    assert compose.conv_exposure_s(1_000_000, 17, 1_000_000) == 0.0
    assert compose.conv_exposure_s(1_000_000, 0, 0) == 0.0


def test_conversion_exposure_uses_the_middle_of_the_bin_and_twenty_seconds():
    assert compose.conv_exposure_s(0, 3, 100_000) == pytest.approx(100.0 - (35.0 + 20.0))
    assert compose.conv_exposure_s(0, 0, 80_000) == pytest.approx(80.0 - (5.0 + 20.0))


def test_conversion_exposure_broadcasts_over_arrays():
    t0, q, T = np.array([[1_000_000], [2_000_000]]), np.arange(18)[None, :], np.array([[1_075_000], [5_000_000]])
    out = compose.conv_exposure_s(t0, q, T)
    assert out.shape == (2, 18) and out.dtype == np.float64
    assert out[0, :6].tolist() == [50.0, 40.0, 30.0, 20.0, 10.0, 0.0] and (out[0, 6:] == 0.0).all()
    assert (out[1] == 60.0).all()
    assert out[0, 2] == compose.conv_exposure_s(1_000_000, 2, 1_075_000)


def test_conversion_exposure_returns_a_plain_float_for_numbers():
    assert isinstance(compose.conv_exposure_s(1_000_000, 0, 1_075_000), float)


@pytest.mark.parametrize('args', [(1_000_000, -1, 1_075_000), (1_000_000, 18, 1_075_000), (np.nan, 0, 1_075_000), (1_000_000, 0, np.inf),
                                  (1_000_000, 1.5, 1_075_000)])
def test_conversion_exposure_refuses_a_bad_bin_and_non_finite_times(args):
    with pytest.raises(ValueError):
        compose.conv_exposure_s(*args)


# =====================================================================================================================================
# RowTable
# =====================================================================================================================================
def row_arrays(n=3, **over):
    d = dict(frame_row=np.arange(n), match=np.arange(n) // 2, t0_ms=np.full(n, 120_300), exposure_ms=np.full((n, C.N_BINS), 10_000),
             T_ms=np.full(n, 900_000), views=np.arange(n) + 5, destroyed=np.zeros((n, 2, 4), dtype=np.int64))
    d.update(over)
    return d


def test_a_row_table_keeps_read_only_int64_copies():
    src = row_arrays(frame_row=np.arange(3, dtype=np.int32), destroyed=np.ones((3, 2, 4), dtype=np.int16))
    t = compose.RowTable(**src)
    for name in ('frame_row', 'match', 't0_ms', 'exposure_ms', 'T_ms', 'views', 'destroyed'):
        a = getattr(t, name)
        assert a.dtype == np.int64 and not a.flags.writeable, name
    assert len(t) == 3 and t.roles == ()
    src['T_ms'][0] = 1
    assert t.T_ms[0] == 900_000                                                    # a copy, not a view of the caller's array


@pytest.mark.parametrize('field,value', [
    ('exposure_ms', np.full((3, 17), 10_000)), ('exposure_ms', np.full((3, 18), 10_001)), ('exposure_ms', np.full((3, 18), -1)),
    ('destroyed', np.zeros((3, 4), dtype=np.int64)), ('destroyed', np.full((3, 2, 4), -1)), ('views', np.array([5, 6, -1])),
    ('frame_row', np.array([0, 1, -2])), ('T_ms', np.array([900_000, 0, 900_000])), ('t0_ms', np.array([120_300, -5, 5])),
    ('views', np.arange(4))])
def test_a_row_table_refuses_arrays_that_do_not_describe_rows(field, value):
    with pytest.raises(ValueError):
        compose.RowTable(**row_arrays(**{field: value}))


@pytest.mark.parametrize('field', ['frame_row', 'match', 't0_ms', 'T_ms', 'views'])
def test_a_row_table_refuses_floats_and_booleans_for_whole_numbers(field):
    for bad in (np.array([0.0, 1.0, 2.0]), np.array([True, False, True])):
        with pytest.raises(TypeError):
            compose.RowTable(**row_arrays(**{field: bad}))


def test_a_row_table_refuses_holdout_roles_and_unknown_names():
    for roles in (('TEST',), ('FIT', 'EXTERNAL'), ('NOT_A_ROLE',), 'DEV'):
        with pytest.raises((ValueError, TypeError)):
            compose.RowTable(**row_arrays(), roles=roles)
    assert compose.RowTable(**row_arrays(), roles=('FIT', 'DEV')).roles == ('FIT', 'DEV')


# =====================================================================================================================================
# row_table: the t0 information of the frame-clock rows of some roles
# =====================================================================================================================================
@pytest.fixture(scope='module')
def value_s2(tmp_path_factory, toy_value_s1):
    """The toy S2 of the value toys: one frame-clock view per frame row and two event views per match whose ``src_index`` collide."""
    folder = tmp_path_factory.mktemp('value_s2')
    s = build_toy_s2(folder / 's2', toy_value_s1)
    yield s
    del s
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


S1_NEEDS = ('matches/role', 'matches/T_ms', 'matches/obj_off', 'frame_rows/match', 'frame_rows/t0', 'frame_rows/exposure_ms',
            'objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker')
S2_NEEDS = ('views/src', 'views/src_index', 'views/match', 'views/ref')


class DictStore:
    """A store that holds only the arrays it is given (copies) and records every read; any other array is a FileNotFoundError."""

    def __init__(self, arrays):
        self.arrays = {k: np.array(v) for k, v in arrays.items()}
        self.reads = []

    @classmethod
    def of(cls, store, names):
        return cls({n: store.array(n) for n in names})

    def array(self, name):
        self.reads.append(name)
        try:
            return self.arrays[name]
        except KeyError:
            raise FileNotFoundError(f'no array {name}') from None


class Untouchable:
    def array(self, name):
        raise AssertionError(f'a store was read: {name}')


# the frame rows of the value toys: match 0 (FIT) rows 0 .. 9, match 1 (STOP) rows 10 .. 16, match 2 (DEV) rows 17 .. 21, match 3 (CAL)
# rows 22 .. 24. Match 0 has t0 = 120,300 + 60,000 j (j = 0 .. 9). Its towers with a tier and a taker: the inner tower of blue taken by red
# at 168,000 (blue has lost an inner tower), the outer tower of red taken by blue at 330,000 and the inner tower of red taken by blue at
# 505,000; the tower without a tier at 405,000 counts for nothing. So the lost towers (team, tier) before t0 are
#     j = 0        none
#     j = 1 .. 3   blue: one inner
#     j = 4 .. 6   blue: one inner;  red: one outer                (t0 = 360,300 is after 330,000)
#     j = 7 .. 9   blue: one inner;  red: one outer and one inner  (t0 = 540,300 is after 505,000)
DESTROYED_M0 = np.zeros((10, 2, 4), dtype=np.int64)
DESTROYED_M0[1:, BLUE, 1] = 1
DESTROYED_M0[4:, RED, 0] = 1
DESTROYED_M0[7:, RED, 1] = 1


def test_row_table_of_one_role_lists_its_frame_rows_in_s1_order(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, 'DEV')
    assert t.frame_row.tolist() == [17, 18, 19, 20, 21] and (t.match == 2).all() and t.roles == ('DEV',)
    assert np.array_equal(t.t0_ms, np.asarray(toy_value_s1.array('frame_rows/t0'))[17:22])
    assert np.array_equal(t.exposure_ms, np.asarray(toy_value_s1.array('frame_rows/exposure_ms'))[17:22])
    assert (t.T_ms == 400_000).all() and t.exposure_ms.shape == (5, 18) and t.destroyed.shape == (5, 2, 4)
    assert len(t) == 5


def test_row_table_of_several_roles_keeps_the_order_of_s1_and_names_the_roles_in_the_order_of_config(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, ('STOP', 'FIT'))
    assert t.frame_row.tolist() == list(range(17)) and t.roles == ('FIT', 'STOP')
    assert t.match.tolist() == [0] * 10 + [1] * 7
    assert t.T_ms.tolist() == [700_000] * 10 + [520_000] * 7
    three = compose.row_table(toy_value_s1, value_s2, ('DEV', 'STOP', 'FIT'))                 # config order, not alphabetical
    assert three.roles == ('FIT', 'STOP', 'DEV') and three.frame_row.tolist() == list(range(22))
    assert three.match.tolist() == [0] * 10 + [1] * 7 + [2] * 5


def test_the_view_of_a_row_is_the_frame_clock_view_with_its_row_number_not_an_event_view(toy_value_s1, value_s2):
    src, index = np.asarray(value_s2.array('views/src')), np.asarray(value_s2.array('views/src_index'))
    t = compose.row_table(toy_value_s1, value_s2, ('FIT', 'STOP', 'DEV'))
    for r, v in zip(t.frame_row, t.views):
        hits = [w for w in range(len(src)) if src[w] == 0 and index[w] == r]          # the one view of src 0 and src_index r
        assert hits == [v], (r, v)
    assert (src[t.views] == 0).all()
    event_row_numbers = set(index[src != 0].tolist())
    assert event_row_numbers & set(t.frame_row.tolist())                              # the event views do collide with the frame rows


def test_the_view_has_the_match_and_the_reference_time_of_its_row(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, ('FIT', 'STOP', 'DEV'))
    assert np.array_equal(np.asarray(value_s2.array('views/match'))[t.views], t.match)
    assert np.array_equal(np.asarray(value_s2.array('views/ref'))[t.views], t.t0_ms)


def test_the_tower_state_is_the_one_before_t0_worked_out_by_hand(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, 'FIT')
    assert np.array_equal(t.destroyed, DESTROYED_M0)
    other = compose.row_table(toy_value_s1, value_s2, ('STOP', 'DEV'))
    assert (other.destroyed == 0).all()                                               # no tower falls in the matches 1 and 2


def test_the_tower_state_is_what_destroyed_before_says(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, ('FIT', 'STOP', 'DEV'))
    assert np.array_equal(t.destroyed, towertier.destroyed_before(toy_value_s1, t.match, t.t0_ms))


def test_row_table_arrays_are_read_only(toy_value_s1, value_s2):
    t = compose.row_table(toy_value_s1, value_s2, 'FIT')
    with pytest.raises(ValueError):
        t.destroyed[0, 0, 0] = 5


def test_row_table_refuses_holdout_roles_and_roles_s3_does_not_use_before_any_read():
    for roles in ('TEST', ('EXTERNAL',), ('FIT', 'TEST'), 'CAL', ('AUDIT',), 'UNUSED', 'NOT_A_ROLE', ()):
        with pytest.raises(ValueError):
            compose.row_table(Untouchable(), Untouchable(), roles)


def test_row_table_needs_only_these_arrays_and_reads_nothing_about_the_engagements(toy_value_s1, value_s2):
    s1, s2 = DictStore.of(toy_value_s1, S1_NEEDS), DictStore.of(value_s2, S2_NEEDS)
    t = compose.row_table(s1, s2, ('FIT', 'DEV'))                                     # any other array would be a FileNotFoundError
    assert set(s1.reads) <= set(S1_NEEDS) and set(s2.reads) <= set(S2_NEEDS)
    assert 'matches/key' not in s1.reads
    assert not [name for name in s1.reads + s2.reads if name.split('/')[0] in ('units', 'onsets', 'kills', 'event_rows')]
    assert np.array_equal(t.frame_row, compose.row_table(toy_value_s1, value_s2, ('FIT', 'DEV')).frame_row)


def test_events_at_or_after_t0_do_not_change_the_tower_state_of_a_row(toy_value_s1, value_s2):
    base = compose.row_table(toy_value_s1, value_s2, 'FIT')
    s1 = DictStore.of(toy_value_s1, S1_NEEDS)
    when = s1.arrays['objectives/t']
    j = 4                                                                              # the row at t0 = 360,300
    late = when >= base.t0_ms[j]
    s1.arrays['objectives/kind'][late] = C.O_KINDS.index('dragon')                      # whatever happens from t0 on is no tower any more
    changed = compose.row_table(s1, DictStore.of(value_s2, S2_NEEDS), 'FIT')
    assert np.array_equal(changed.destroyed[:j + 1], base.destroyed[:j + 1])
    assert not np.array_equal(changed.destroyed[7:], base.destroyed[7:])               # the rows after those events do change


def corrupt_s2(value_s2, **edits):
    """A DictStore of the view arrays with some entries changed: ``edits`` = {array name: {view: new value}}."""
    s2 = DictStore.of(value_s2, S2_NEEDS)
    for name, changes in edits.items():
        for view, value in changes.items():
            s2.arrays[name][view] = value
    return s2


def frame_view_of(value_s2, row):
    src, index = np.asarray(value_s2.array('views/src')), np.asarray(value_s2.array('views/src_index'))
    return int(np.flatnonzero((src == 0) & (index == row))[0])


@pytest.mark.parametrize('what', ['ref', 'match', 'duplicate', 'missing', 'out_of_range'])
def test_row_table_refuses_an_s2_store_that_does_not_belong_to_s1_without_a_key_in_the_message(toy_value_s1, value_s2, what):
    v = frame_view_of(value_s2, 18)
    event = int(np.flatnonzero(np.asarray(value_s2.array('views/src')) != 0)[0])
    edits = {'ref': {'views/ref': {v: 999}}, 'match': {'views/match': {v: 0}},
             'duplicate': {'views/src': {event: 0}, 'views/src_index': {event: 17}},    # an extra frame-clock view of the row 17
             'missing': {'views/src': {v: 1}},                                           # the view of the row 18 is an event view now
             'out_of_range': {'views/src_index': {v: 10_000}}}[what]
    with pytest.raises(ValueError) as raised:
        compose.row_table(DictStore.of(toy_value_s1, S1_NEEDS), corrupt_s2(value_s2, **edits), 'DEV')
    assert KEY_PREFIX not in str(raised.value)


def test_row_table_refuses_a_store_without_the_view_arrays(toy_value_s1):
    with pytest.raises(ValueError) as raised:
        compose.row_table(DictStore.of(toy_value_s1, S1_NEEDS), DictStore({}), 'DEV')
    assert KEY_PREFIX not in str(raised.value)


# =====================================================================================================================================
# the composer: construction
# =====================================================================================================================================
def test_the_composer_binds_the_heads_once_to_the_views_of_the_row_table_and_keeps_no_store():
    h, sentinel = ToyHeads(), object()
    c = toy_composer(heads=h, s2=sentinel)
    assert h.n_bind == 1 and h.bound_with == id(sentinel)
    assert np.array_equal(h.bound_views, c.rows.views) and (h.bound_views == 100 + np.arange(len(TOY_ROWS))).all()
    assert not [k for k, v in vars(c).items() if v is sentinel]                           # the store is dropped after the bind
    c.expect([0])
    c.simulate([0])
    assert h.n_bind == 1


def test_the_composer_keeps_its_own_copy_of_the_prices_as_floats():
    prices = {name: int(value) for name, value in TOY_COMPOSE_PRICES.items()}
    c = toy_composer(prices=prices)
    assert all(type(v) is float for v in c.prices.values()) and c.prices == {k: float(v) for k, v in prices.items()}
    prices['dragon'] = 1
    assert c.prices['dragon'] == 200.0 and c.expect([0])[0, 0] == pytest.approx(TOY_EXPECTED_V60)
    with pytest.raises(ValueError):
        c.gold_reps[0] = 1.0                                                           # the representatives are read-only


def test_a_composer_notices_that_its_heads_were_bound_again_to_other_views():
    import dataclasses
    h = ToyHeads()
    first = toy_composer(heads=h)
    assert first.expect([0]).shape == (1, 3)
    other = dataclasses.replace(toy_row_table(), views=np.arange(len(TOY_ROWS)) + 400)
    toy_composer(heads=h, rows=other)                                                  # binds the same chain to other views
    calls = (lambda: first.expect([0]), lambda: first.simulate([0]), lambda: first.simulate_onsets(0), lambda: first.expected_onsets([0]))
    for call in calls:
        with pytest.raises(RuntimeError, match='bound again'):
            call()
    toy_composer(heads=h, rows=toy_row_table(TOY_ROW_SPECS[:2]))                      # a shorter table: the probe row is gone
    with pytest.raises(RuntimeError, match='bound again'):
        first.expect([0])
    assert toy_composer(heads=ToyHeads()).expect([0]).shape == (1, 3)                    # one chain for each composer is fine


def test_the_composer_exposes_its_parts_and_its_numbers():
    c = toy_composer(S=77, seed=5)
    assert c.S == 77 and c.seed == 5 and c.heads is not None and len(c.rows) == len(TOY_ROWS)
    assert c.gold_reps.shape == (31,) and c.prices['dragon'] == 200.0


def test_the_composer_has_the_default_sample_size_and_seed_of_the_decisions():
    assert compose.DEFAULT_S == 1000 and compose.DEFAULT_SEED == 20261021
    c = compose.Composer(ToyHeads(), TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, toy_row_table(), None)
    assert c.S == 1000 and c.seed == 20261021


def test_composer_heads_is_the_protocol_of_ruling_r3_re_exported():
    assert compose.ComposerHeads is heads.ComposerHeads
    assert isinstance(ToyHeads(), compose.ComposerHeads)


def test_the_composer_refuses_what_is_not_a_set_of_heads(monkeypatch):
    class NoHeads:
        analytic_ok = True
    with pytest.raises(TypeError):
        toy_composer(heads=NoHeads())


@pytest.mark.parametrize('S', [0, 1, -3, 2.5, True, '10', None])
def test_the_composer_refuses_a_sample_size_below_two_and_non_integers(S):
    with pytest.raises((ValueError, TypeError)):
        toy_composer(S=S)


@pytest.mark.parametrize('seed', [-1, 1.5, True, 'x', None])
def test_the_composer_refuses_a_seed_that_is_no_non_negative_integer(seed):
    with pytest.raises((ValueError, TypeError)):
        toy_composer(seed=seed)


@pytest.mark.parametrize('reps', [np.arange(30.0), np.full(31, np.nan), -np.arange(31.0), np.ones((31, 1)), np.full(31, np.inf)])
def test_the_composer_refuses_gold_representatives_that_are_not_31_finite_non_negative_numbers(reps):
    with pytest.raises(ValueError):
        toy_composer(gold_reps=reps)


def test_the_composer_refuses_prices_that_miss_a_group_or_are_not_numbers():
    for group in flowprice.GROUPS:
        bad = {k: v for k, v in TOY_COMPOSE_PRICES.items() if k != group}
        with pytest.raises(ValueError):
            toy_composer(prices=bad)
        with pytest.raises(ValueError):
            toy_composer(prices={**TOY_COMPOSE_PRICES, group: float('nan')})
    with pytest.raises(TypeError):
        toy_composer(prices=[1.0] * 12)


def test_the_composer_refuses_a_row_table_that_is_none_and_a_tier_model_without_a_law():
    with pytest.raises(TypeError):
        toy_composer(rows=np.arange(3))
    with pytest.raises(TypeError):
        toy_composer(tier_model=object())


def test_the_composer_is_picklable_and_the_copy_gives_the_same_answers():
    c = toy_composer(S=200, seed=9)
    d = pickle.loads(pickle.dumps(c))
    assert np.array_equal(c.expect(TOY_ROWS), d.expect(TOY_ROWS))
    assert np.array_equal(c.simulate(TOY_ROWS).q19, d.simulate(TOY_ROWS).q19)


# =====================================================================================================================================
# the closed form: a second computation of the same number
# =====================================================================================================================================
def hand_tier_model():
    """A tier law with weights chosen by hand: the logits move with the towers lost by the victim, with the side and with the minute."""
    k = towertier.N_FEATURES
    coef = np.zeros((4, k))
    coef[0, 0] = -0.9                       # the more outer towers the victim has lost, the less likely the next one is an outer tower
    coef[1, 0], coef[1, 1] = 0.5, -0.3
    coef[2, 4] = 0.7                        # red takes more base towers
    coef[3, 5] = 0.5                        # the later it is (minutes), the more likely the nexus towers
    return towertier.TierModel([0, 1, 2, 3], coef, np.array([0.3, 0.1, -0.2, -0.8]), np.zeros(k), np.ones(k))


TIER_PRICES = {'tower_outer': 520.0, 'tower_inner': 610.0, 'tower_base': 400.0, 'tower_nexus': 900.0}
RICH_PRICES = {**TOY_COMPOSE_PRICES, **TIER_PRICES}
# the state of three rows: blue has lost one outer tower and red three outer and an inner one / nothing / two inners and a base tower of
# blue and two nexus towers of red
STATE = np.zeros((3, 2, 4), dtype=np.int64)
STATE[0, BLUE], STATE[0, RED] = (1, 0, 0, 0), (3, 1, 0, 0)
STATE[2, BLUE], STATE[2, RED] = (0, 2, 1, 0), (0, 0, 0, 2)
RICH_SPECS = ((1000.0, 5000.0), (1500.0, 1585.0), (900.0, 2400.0))        # all bins / five full bins and one of 5 s / all bins
SHORT_SPECS = ((1000.0, 1075.0), (1500.0, 1585.0), (900.0, 1000.0))      # 5 + 5 s / 5 + 5 s / 7 full bins: few bins to evaluate


def rich_composer(S=1000, seed=7, heads_=None, specs=RICH_SPECS, state=STATE, **kw):
    return toy_composer(S=S, seed=seed, rows=toy_row_table(specs, state), heads=RichToyHeads() if heads_ is None else heads_,
                        prices=RICH_PRICES, tier_model=hand_tier_model(), **kw)


def tower_price(c, i, q, taker):
    """The expected price of a tower taken by ``taker`` in the bin q of the row i: the tier law at the state of the victim and at the middle
    of the bin, times the four tier prices (written out, not through ``expected_tower_price``)."""
    t_s = (int(c.rows.t0_ms[i]) + (q + 0.5) * 10_000) / 1000.0
    p = c.tier_model.prob(c.rows.destroyed[i, 1 - taker][None, :], np.array([taker]), np.array([t_s]))[0]
    return float(p @ np.array([c.prices[g] for g in towertier.TIER_GROUPS]))


def cell_values(c, i, q):
    """(18,) the value of one object of each cell in the bin q of the row i: blue +, red -, the price of the group, the tower by the law."""
    out = np.zeros(18)
    for kind, name in enumerate(C.O_KINDS):
        for team in (BLUE, RED):
            price = tower_price(c, i, q, team) if name == 'tower' else c.prices[name]
            out[2 * kind + team] = price if team == BLUE else -price
    return out


def oracle_expect(c, i):
    """E[V_H] of row i for H = 60, 120, 180 s by a second route: a Python loop over the bin and the mark, the zone and the size asked for as
    one block each, the Poisson law written with ``math`` and the tail as one minus the sum. Bins without exposure are skipped (weight
    0)."""
    h, rows = c.heads, c.rows
    t0, T = int(rows.t0_ms[i]), int(rows.T_ms[i])
    reps = np.asarray(c.gold_reps, dtype=np.float64)
    one = np.array([i])
    lam = h.rate(one)[0]
    zz, ss = np.repeat(np.arange(10), 4), np.tile(np.arange(4), 10)
    r10, r40 = np.full(10, i), np.full(40, i)
    fact = np.array([math.factorial(k) for k in range(6)], dtype=np.float64)
    total = np.zeros(3)
    for q in range(18):
        a = lam[q] * rows.exposure_ms[i, q] / 1000.0
        if a == 0.0:
            continue
        cexp = max(0.0, min(60.0, (T - (t0 + (q + 0.5) * 10_000 + 20_000)) / 1000.0))
        value = cell_values(c, i, q)
        pi = h.mark_prob(one, np.array([q]))[0]
        e_q = 0.0
        for m in range(52):
            rho = h.zone_prob(one, np.array([q]), np.array([m]))[0]
            size = h.size_prob(r10, np.full(10, q), np.full(10, m), np.arange(10))                     # (10, 4)
            q40, m40 = np.full(40, q), np.full(40, m)
            mu = h.kill_rate(r40, q40, m40, zz, ss)                                                    # (40, 2)
            G = np.stack([h.gold_prob(r40, q40, m40, zz, ss, np.full((40, 2), layer)) @ reps for layer in range(1, 7)], axis=-1)  # (40,2,6)
            pk = np.exp(-mu)[..., None] * mu[..., None] ** np.arange(6) / fact                          # P(k) for k = 0 .. 5
            tail = 1.0 - pk.sum(axis=-1)
            gold_mean = (pk[..., 1:] * G[..., :5]).sum(axis=-1) + tail * G[..., 5]                      # (40, 2)
            zeros = np.zeros((40, 4))
            obj = h.obj_rate(r40, q40, m40, zz, ss, zeros)                                             # (40, 18)
            m4 = features.m4_masked_inputs(h.row_view(r40), q40, m40, zz, zeros, np.zeros((40, 18)))
            conv = h.conv_rate(m4)                                                                     # (40, 18)
            g = gold_mean[:, 0] - gold_mean[:, 1] + obj @ value + (conv @ value) * cexp
            e_q += pi[m] * float((rho[zz] * size[zz, ss] * g).sum())
        for j, n_bins in enumerate(H_BINS):
            if q < n_bins:
                total[j] += a * e_q
    return total


def test_the_closed_form_equals_the_brute_force_oracle_on_heads_that_differ_in_every_argument():
    c = rich_composer()
    got = c.expect([0, 1, 2])
    want = np.array([oracle_expect(c, i) for i in range(3)])
    assert got == pytest.approx(want, rel=1e-10)
    assert (np.abs(got) > 1.0).all()                                                  # the answer is not accidentally near zero


def test_the_rich_row_that_ends_early_uses_only_its_bins_with_exposure():
    c = rich_composer()
    assert c.rows.exposure_ms[1].tolist() == [10_000] * 5 + [5_000] + [0] * 12
    got = c.expect([1])[0]
    assert got == pytest.approx(oracle_expect(c, 1), rel=1e-10)
    assert got[0] == got[1] == got[2]                                                  # the six bins with exposure are all below the bin 6


def test_the_closed_form_moves_with_the_row_state_the_prices_and_the_gold_representatives():
    base = rich_composer().expect([0])[0, 0]
    assert rich_composer(state=np.zeros((3, 2, 4), dtype=np.int64)).expect([0])[0, 0] != pytest.approx(base, rel=1e-6)    # the tower state
    other = {**RICH_PRICES, 'dragon': 999.0}
    assert toy_composer(rows=toy_row_table(RICH_SPECS, STATE), heads=RichToyHeads(), prices=other, tier_model=hand_tier_model()
                        ).expect([0])[0, 0] != pytest.approx(base, rel=1e-6)
    assert rich_composer(gold_reps=TOY_GOLD_REPS * 2).expect([0])[0, 0] != pytest.approx(base, rel=1e-6)


def test_expect_is_the_same_for_a_row_alone_in_another_order_and_with_repeats():
    c = rich_composer()
    all_rows = c.expect([0, 1, 2])
    assert np.array_equal(c.expect([2, 0, 0, 1]), all_rows[[2, 0, 0, 1]])
    assert np.array_equal(c.expect([1]), all_rows[[1]])


def test_expect_of_no_rows_is_an_empty_table():
    assert toy_composer().expect([]).shape == (0, 3)
    assert toy_composer().simulate([]).mean.shape == (0, 3)
    assert toy_composer().expected_onsets([]).shape == (0, 3)


@pytest.mark.parametrize('idx', [[-1], [5], [0, 99], [[0, 1]], [0.5], [True], 'a'])
def test_the_row_numbers_must_be_a_flat_list_of_rows_of_the_table(idx):
    with pytest.raises((ValueError, TypeError)):
        toy_composer().expect(idx)
    with pytest.raises((ValueError, TypeError)):
        toy_composer().simulate(idx)


def test_row_numbers_are_checked_before_any_head_is_asked():
    h = RecordingHeads()
    c = rich_composer(heads_=h)
    for bad in ([3], [-1], [0, 3]):                                                    # the table has the rows 0, 1, 2
        for call in (c.expect, c.simulate, c.expected_onsets):
            with pytest.raises(ValueError, match='row numbers'):
                call(bad)
    with pytest.raises(ValueError, match='row numbers'):
        c.simulate_onsets(3)
    assert h.log == {}


def test_expected_onsets_is_the_rate_times_the_exposure_summed_over_the_bins_of_the_horizon():
    c = rich_composer()
    got = c.expected_onsets([0, 1, 2])
    lam = c.heads.rate(np.arange(3))
    a = lam * c.rows.exposure_ms / 1000.0
    want = np.stack([a[:, :n].sum(axis=1) for n in H_BINS], axis=1)
    assert got == pytest.approx(want, rel=1e-14)
    assert toy_composer().expected_onsets([0])[0] == pytest.approx([0.6, 1.2, 1.8])


# ---- the full grid is evaluated ---------------------------------------------------------------------------------------------------------
class RecordingHeads(RichToyHeads):
    """RichToyHeads that notes every call (the arguments of the categories) in ``log``."""

    def __init__(self):
        super().__init__()
        self.log = {}

    def _note(self, name, **arrays):
        self.log.setdefault(name, []).append({k: np.array(v) for k, v in arrays.items()})

    def mark_prob(self, rows, q):
        self._note('mark_prob', q=q)
        return super().mark_prob(rows, q)

    def zone_prob(self, rows, q, m):
        self._note('zone_prob', q=q, m=m)
        return super().zone_prob(rows, q, m)

    def size_prob(self, rows, q, m, z):
        self._note('size_prob', q=q, m=m, z=z)
        return super().size_prob(rows, q, m, z)

    def kill_rate(self, rows, q, m, z, s):
        self._note('kill_rate', rows=rows, q=q, m=m, z=z, s=s)
        return super().kill_rate(rows, q, m, z, s)

    def gold_prob(self, rows, q, m, z, s, layer):
        self._note('gold_prob', q=q, m=m, z=z, s=s, layer=layer)
        return super().gold_prob(rows, q, m, z, s, layer)

    def obj_rate(self, rows, q, m, z, s, K):
        self._note('obj_rate', q=q, m=m, z=z, s=s, K=K)
        return super().obj_rate(rows, q, m, z, s, K)

    def conv_rate(self, m4_inputs):
        self._note('conv_rate', x=m4_inputs)
        return super().conv_rate(m4_inputs)


def asked(calls, *names):
    """The tuples (q, m, ...) of the inputs of every call in ``calls`` (the logs of ``RecordingHeads``), in one list."""
    return [t for d in calls for t in zip(*[d[n].tolist() for n in names])]


def test_the_closed_form_asks_the_heads_about_every_combination_and_every_layer_nothing_is_pruned():
    h = RecordingHeads()
    c = rich_composer(heads_=h)
    c.expect([0])                                                                      # row 0 has exposure in all 18 bins
    log = h.log
    grid = [(q, m, z, s) for q in range(18) for m in range(52) for z in range(10) for s in range(4)]       # sorted
    assert len(log['mark_prob']) == 1 and sorted(asked(log['mark_prob'], 'q')) == [(q,) for q in range(18)]
    assert sorted(asked(log['zone_prob'], 'q', 'm')) == [(q, m) for q in range(18) for m in range(52)]
    assert sorted(asked(log['size_prob'], 'q', 'm', 'z')) == [(q, m, z) for q in range(18) for m in range(52) for z in range(10)]
    assert sorted(asked(log['kill_rate'], 'q', 'm', 'z', 's')) == grid                         # every combination, once
    assert all((d['rows'] == 0).all() for d in log['kill_rate'])
    assert sorted(asked(log['obj_rate'], 'q', 'm', 'z', 's')) == grid
    for layer in range(1, 7):                                                                  # every layer: every combination, once
        calls = [d for d in log['gold_prob'] if (d['layer'] == layer).all()]
        assert sorted(asked(calls, 'q', 'm', 'z', 's')) == grid, layer
    assert all(((d['layer'] >= 1) & (d['layer'] <= 6)).all() and (d['layer'][:, 0] == d['layer'][:, 1]).all() for d in log['gold_prob'])
    x = np.concatenate([d['x'] for d in log['conv_rate']])
    assert sorted((int(a), int(b), int(d)) for a, b, d in x[:, 2:5]) == [(q, m, z) for q in range(18) for m in range(52) for z in range(10)]
    sizes = [len(d['q']) for name in ('zone_prob', 'size_prob', 'kill_rate', 'gold_prob', 'obj_rate') for d in log[name]]
    sizes += [len(d['x']) for d in log['conv_rate']]
    assert max(sizes) == compose.CALL_ROWS == 1024                                              # a head gets 1,024 inputs at a time
    assert len(log['kill_rate']) == len(grid) // 1024 + 1


def test_the_closed_form_passes_no_k_and_no_o_to_the_heads_that_ignore_them():
    h = RecordingHeads()
    rich_composer(heads_=h).expect([0])
    assert len(h.log['obj_rate']) > 1 and all((d['K'] == 0).all() and d['K'].shape == (len(d['q']), 4) for d in h.log['obj_rate'])
    assert all((d['x'][:, 5:] == 0).all() for d in h.log['conv_rate'])                      # kills, kill gold and the 18 cells of O: zeros


def test_the_size_of_a_call_does_not_change_the_closed_form_of_heads_without_a_library(monkeypatch):
    c = rich_composer(specs=SHORT_SPECS)
    ref = c.expect([0, 1, 2])
    for rows in (7, 100, 10 ** 9):
        monkeypatch.setattr(compose, 'CALL_ROWS', rows)
        assert np.array_equal(c.expect([0, 1, 2]), ref), rows


# ---- heads that return something the composer cannot use ---------------------------------------------------------------------------------
class BadHeads(RichToyHeads):
    def __init__(self, which, how):
        super().__init__()
        self.which, self.how = which, how

    def _spoil(self, name, out):
        if name != self.which:
            return out
        out = np.array(out, dtype=np.float64)
        if self.how == 'nan':
            out.flat[0] = np.nan
        elif self.how == 'negative':
            out.flat[0] = -0.5
        elif self.how == 'scale':
            out = out * 0.5
        elif self.how == 'shape':
            out = out[..., :-1]
        return out

    def rate(self, rows): return self._spoil('rate', super().rate(rows))                                          # noqa: E704
    def mark_prob(self, rows, q): return self._spoil('mark_prob', super().mark_prob(rows, q))                    # noqa: E704
    def zone_prob(self, rows, q, m): return self._spoil('zone_prob', super().zone_prob(rows, q, m))             # noqa: E704
    def size_prob(self, rows, q, m, z): return self._spoil('size_prob', super().size_prob(rows, q, m, z))      # noqa: E704
    def kill_rate(self, rows, q, m, z, s): return self._spoil('kill_rate', super().kill_rate(rows, q, m, z, s))   # noqa: E704
    def gold_prob(self, rows, q, m, z, s, layer): return self._spoil('gold_prob', super().gold_prob(rows, q, m, z, s, layer))   # noqa: E704
    def obj_rate(self, rows, q, m, z, s, K): return self._spoil('obj_rate', super().obj_rate(rows, q, m, z, s, K))   # noqa: E704
    def conv_rate(self, m4_inputs): return self._spoil('conv_rate', super().conv_rate(m4_inputs))               # noqa: E704


BAD = [('rate', 'nan'), ('rate', 'negative'), ('rate', 'shape'), ('mark_prob', 'scale'), ('mark_prob', 'nan'), ('zone_prob', 'scale'),
       ('size_prob', 'negative'), ('size_prob', 'shape'), ('kill_rate', 'nan'), ('kill_rate', 'negative'), ('gold_prob', 'scale'),
       ('gold_prob', 'shape'), ('obj_rate', 'nan'), ('obj_rate', 'negative'), ('conv_rate', 'nan'), ('conv_rate', 'shape')]


@pytest.mark.parametrize('which,how', BAD)
def test_heads_that_return_a_wrong_shape_a_bad_number_or_probabilities_that_do_not_add_up_are_refused_by_name(which, how):
    c = rich_composer(S=40, heads_=BadHeads(which, how))
    with pytest.raises(ValueError, match=which):
        c.expect([0])
    with pytest.raises(ValueError, match=which):
        c.simulate([0])


# ---- heads whose answer is a function of the row alone ---------------------------------------------------------------------------------
class RowCodedHeads(ToyHeads):
    """Every head answers with a point mass, or a Poisson mean, that is a function of the ROW (and of what the chain has drawn so far): a
    head that the composer asks about another row is seen at once, in the simulation and in the closed form. Row r:

        mark  3 + r         zone  (r + m) mod 10        size  (r + z) mod 4          kills  blue 1 + 2 r, red 0.5 + r
        gold bin of a team t at the kill layer L: (r + 2 L + t) mod 31          objects: 2 in the cell r mod 12 (never a tower cell)
        conversions: 3 per second in the cell (r + 5) mod 12 (read from the first column of the row view, the number of the row)
    """

    @staticmethod
    def _one_hot(width, at):
        p = np.zeros((len(at), width))
        p[np.arange(len(at)), at] = 1.0
        return p

    def mark_prob(self, rows, q):
        return self._one_hot(C.N_MARKS, (3 + self._rows(rows)) % C.N_MARKS)

    def zone_prob(self, rows, q, m):
        return self._one_hot(10, (self._rows(rows) + np.asarray(m)) % 10)

    def size_prob(self, rows, q, m, z):
        return self._one_hot(4, (self._rows(rows) + np.asarray(z)) % 4)

    def kill_rate(self, rows, q, m, z, s):
        r = self._rows(rows)
        return np.column_stack([1.0 + 2.0 * r, 0.5 + 1.0 * r])

    def gold_prob(self, rows, q, m, z, s, layer):
        r, layer = self._rows(rows), np.asarray(layer)
        p = np.zeros((len(r), 2, 31))
        for t in (BLUE, RED):
            p[np.arange(len(r)), t, (r + 2 * layer[:, t] + t) % 31] = 1.0
        return p

    def obj_rate(self, rows, q, m, z, s, K):
        r = self._rows(rows)
        out = np.zeros((len(r), 18))
        out[np.arange(len(r)), r % 12] = 2.0
        return out

    def conv_rate(self, m4_inputs):
        A = np.asarray(m4_inputs, dtype=np.float64)
        out = np.zeros((len(A), 18))
        out[np.arange(len(A)), (A[:, 0].astype(np.int64) + 5) % 12] = 3.0
        return out


def row_coded_expected(c, i):
    """E[V_H] of the row i for ``RowCodedHeads`` worked out from the laws above with ``math`` and plain loops."""
    t0, T = int(c.rows.t0_ms[i]), int(c.rows.T_ms[i])
    total = np.zeros(3)
    for q in range(18):
        a = 0.01 * c.rows.exposure_ms[i, q] / 1000.0
        if a == 0.0:
            continue
        cexp = max(0.0, min(60.0, (T - (t0 + (q + 0.5) * 10_000 + 20_000)) / 1000.0))
        gold = 0.0
        for t, mu in enumerate((1.0 + 2.0 * i, 0.5 + i)):
            pk = [math.exp(-mu) * mu ** k / math.factorial(k) for k in range(6)]
            G = lambda layer: float(TOY_GOLD_REPS[(i + 2 * layer + t) % 31])                  # noqa: E731
            e = sum(pk[k] * G(k) for k in range(1, 6)) + (1.0 - sum(pk)) * G(6)
            gold += e if t == BLUE else -e
        value = lambda cell: (1.0 if cell % 2 == 0 else -1.0) * c.prices[C.O_KINDS[cell // 2]]     # noqa: E731
        per_onset = gold + 2.0 * value(i % 12) + 3.0 * cexp * value((i + 5) % 12)
        for j, n_bins in enumerate(H_BINS):
            if q < n_bins:
                total[j] += a * per_onset
    return total


def row_coded_composer(S=400, seed=5):
    return toy_composer(S=S, seed=seed, heads=RowCodedHeads(), rows=toy_row_table(RICH_SPECS))


def test_every_head_is_asked_about_the_row_that_the_simulation_composes():
    c = row_coded_composer()
    for i in range(3):
        d = c.simulate_onsets(i)
        assert len(d.q) > 50
        assert (d.m == (3 + i) % 52).all() and (d.z == (i + d.m) % 10).all() and (d.s == (i + d.z) % 4).all(), i
        layer = np.clip(d.k, 1, 6)
        for t, mean in ((BLUE, 1.0 + 2.0 * i), (RED, 0.5 + i)):
            has = d.k[:, t] >= 1
            assert (d.gold_bin[has, t] == (i + 2 * layer[has, t] + t) % 31).all(), (i, t)
            assert abs(z_test_sum(d.k[:, t], np.full(len(d.q), mean))) < 4.5, (i, t)
        o_cell, c_cell = i % 12, (i + 5) % 12
        assert (np.delete(d.O, o_cell, axis=1) == 0).all() and (np.delete(d.conv, c_cell, axis=1) == 0).all(), i
        assert abs(z_test_sum(d.O[:, o_cell], np.full(len(d.q), 2.0))) < 4.5, i
        cexp = compose.conv_exposure_s(c.rows.t0_ms[i], d.q, c.rows.T_ms[i])
        assert abs(z_test_sum(d.conv[:, c_cell], 3.0 * cexp)) < 4.5, i


def test_every_head_is_asked_about_the_row_that_the_closed_form_composes():
    c = row_coded_composer()
    got = c.expect([0, 1, 2])
    want = np.array([row_coded_expected(c, i) for i in range(3)])
    assert got == pytest.approx(want, rel=1e-11)
    assert not np.allclose(want[0], want[1]) and not np.allclose(want[1], want[2])
    r = row_coded_composer(S=4000, seed=9).simulate([0, 1, 2])
    z = (r.mean - want) / (r.sd / np.sqrt(4000))
    assert np.abs(z).max() < 4.5, z


# ---- the prices of the cells ------------------------------------------------------------------------------------------------------------
class TowerHeads(ToyHeads):
    """No kills; every onset brings one object in the blue tower cell and two in the red one, and a conversion rate of 0.003 per second
    in the red tower cell only."""

    def kill_rate(self, rows, q, m, z, s):
        return np.zeros((len(self._rows(rows)), 2))

    def obj_rate(self, rows, q, m, z, s, K):
        out = np.zeros((len(self._rows(rows)), 18))
        out[:, TOWER_CELL], out[:, TOWER_CELL + 1] = 1.0, 2.0
        return out

    def conv_rate(self, m4_inputs):
        out = np.zeros((len(np.asarray(m4_inputs)), 18))
        out[:, TOWER_CELL + 1] = 0.003
        return out


def test_a_predicted_tower_is_priced_by_the_tier_law_at_the_state_of_the_victim_and_the_middle_of_the_bin():
    c = rich_composer(heads_=TowerHeads())
    got = c.expect([0, 1, 2])
    want = np.zeros((3, 3))
    for i in range(3):
        t0, T = int(c.rows.t0_ms[i]), int(c.rows.T_ms[i])
        for q in range(18):
            a = 0.01 * c.rows.exposure_ms[i, q] / 1000.0
            cexp = max(0.0, min(60.0, (T - (t0 + (q + 0.5) * 10_000 + 20_000)) / 1000.0))
            blue, red = tower_price(c, i, q, BLUE), tower_price(c, i, q, RED)
            for j, n_bins in enumerate(H_BINS):
                if q < n_bins:
                    want[i, j] += a * (1.0 * blue - 2.0 * red - 0.003 * cexp * red)
    assert got == pytest.approx(want, rel=1e-10)


def test_the_victim_is_the_other_team_so_swapping_the_two_teams_of_the_state_changes_the_answer():
    swapped = STATE[:, ::-1].copy()
    a = rich_composer(heads_=TowerHeads()).expect([0, 2])
    b = rich_composer(heads_=TowerHeads(), state=swapped).expect([0, 2])
    assert not np.allclose(a, b, rtol=1e-6)


def test_the_tower_price_moves_with_the_bin_through_the_minute_of_the_law():
    c = rich_composer(heads_=TowerHeads())
    prices = [tower_price(c, 0, q, RED) for q in range(18)]
    assert max(prices) - min(prices) > 0.3                                              # the hand law has a minute term, so q matters


class OneCellHeads(ToyHeads):
    """One object per onset in one cell (``what='obj'``) or a conversion rate of 0.01 per second in one cell (``what='conv'``); no kills."""

    def __init__(self, cell, what):
        super().__init__()
        self.cell, self.what = cell, what

    def kill_rate(self, rows, q, m, z, s):
        return np.zeros((len(self._rows(rows)), 2))

    def obj_rate(self, rows, q, m, z, s, K):
        out = np.zeros((len(self._rows(rows)), 18))
        if self.what == 'obj':
            out[:, self.cell] = 1.0
        return out

    def conv_rate(self, m4_inputs):
        out = np.zeros((len(np.asarray(m4_inputs)), 18))
        if self.what == 'conv':
            out[:, self.cell] = 0.01
        return out


DISTINCT_PRICES = {name: 100.0 + 13.0 * i for i, name in enumerate(flowprice.GROUPS)}


@pytest.mark.parametrize('what', ['obj', 'conv'])
@pytest.mark.parametrize('kind', [k for k, name in enumerate(C.O_KINDS) if name != 'tower'])
@pytest.mark.parametrize('team', [BLUE, RED])
def test_every_cell_is_priced_by_its_own_group_blue_plus_and_red_minus(kind, team, what):
    cell = 2 * kind + team
    c = toy_composer(heads=OneCellHeads(cell, what), prices=DISTINCT_PRICES, rows=toy_row_table(TOY_ROW_SPECS[:3]))
    price = DISTINCT_PRICES[C.O_KINDS[kind]] * (1.0 if team == BLUE else -1.0)
    got = c.expect([0, 1, 2])
    for i, (t0, T) in enumerate(TOY_ROW_SPECS[:3]):
        for j, n_bins in enumerate(H_BINS):
            total = 0.0
            for q in range(n_bins):
                seconds = min(max(min(t0 + 180.0, T - 30.0) - (t0 + 10.0 * q), 0.0), 10.0)
                cexp = max(0.0, min(60.0, T - (t0 + (q + 0.5) * 10.0 + 20.0)))
                total += 0.01 * seconds * (1.0 if what == 'obj' else 0.01 * cexp)
            assert got[i, j] == pytest.approx(price * total, rel=1e-12), (i, j)


class BigKillsHeads(ToyHeads):
    """Blue makes about 40 kills in every onset, red none; the gold bin of blue is 10 + (its layer), so the layer 6 has the bin 16."""

    def kill_rate(self, rows, q, m, z, s):
        n = len(self._rows(rows))
        return np.column_stack([np.full(n, 40.0), np.zeros(n)])

    def gold_prob(self, rows, q, m, z, s, layer):
        p = np.zeros((len(self._rows(rows)), 2, 31))
        p[np.arange(len(p)), 0, 10 + np.asarray(layer)[:, 0]] = 1.0
        p[:, 1, 3] = 1.0
        return p

    def obj_rate(self, rows, q, m, z, s, K):
        return np.zeros((len(self._rows(rows)), 18))

    def conv_rate(self, m4_inputs):
        return np.zeros((len(np.asarray(m4_inputs)), 18))


def test_the_kill_gold_of_a_team_without_a_kill_is_zero_and_the_tail_of_six_kills_is_the_layer_six():
    """With about 40 kills P(six or more) = 1, so the gold of blue is the mean of the layer 6 bin (1,600); red has no kill, so no gold."""
    got = toy_composer(heads=BigKillsHeads()).expect([0])[0, 0]
    assert got == pytest.approx(6 * 0.01 * 10.0 * 1600.0, rel=1e-9)
    r = toy_composer(S=200, heads=BigKillsHeads()).simulate([0])
    assert r.mean[0, 0] == pytest.approx(960.0, rel=0.05)


# =====================================================================================================================================
# the refusals of the closed form
# =====================================================================================================================================
@pytest.mark.parametrize('flags', [dict(analytic_ok=False), dict(obj_uses_K=True), dict(conv_uses_KO=True),
                                   dict(analytic_ok=False, obj_uses_K=True, conv_uses_KO=True)])
def test_expect_refuses_every_head_set_that_is_not_closed_form(flags):
    with pytest.raises(ValueError, match='closed form'):
        toy_composer(heads=ToyHeads(**flags)).expect([0])


def test_the_simulation_accepts_the_heads_that_the_closed_form_refuses():
    r = toy_composer(S=50, heads=KOToyHeads()).simulate([0, 1])
    assert r.mean.shape == (2, 3) and np.isfinite(r.mean).all()


# =====================================================================================================================================
# the simulation
# =====================================================================================================================================
def test_the_simulation_is_a_function_of_the_seed_and_changes_with_it():
    a, b = toy_composer(S=300, seed=1).simulate(TOY_ROWS), toy_composer(S=300, seed=1).simulate(TOY_ROWS)
    other = toy_composer(S=300, seed=2).simulate(TOY_ROWS)
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sd, b.sd) and np.array_equal(a.q19, b.q19)
    assert not np.array_equal(a.mean, other.mean)


def test_the_draws_of_a_row_depend_on_the_seed_and_the_global_row_number_only():
    c = toy_composer(S=300, seed=4)
    all_rows = c.simulate(TOY_ROWS)
    alone = c.simulate([3])
    assert np.array_equal(alone.mean[0], all_rows.mean[3]) and np.array_equal(alone.q19[0], all_rows.q19[3])
    twice = c.simulate([2, 2])
    assert np.array_equal(twice.mean[0], twice.mean[1]) and np.array_equal(twice.mean[0], all_rows.mean[2])
    d1, d2 = c.simulate_onsets(3), c.simulate_onsets(3)
    assert np.array_equal(d1.q, d2.q) and np.array_equal(d1.value, d2.value)


def test_the_row_stream_is_the_generator_of_the_seed_and_the_global_row_number():
    """The first draw of a row is the Poisson count of its onsets, from default_rng([seed, global row]): the global row is the row of
    the S1 store (``frame_row``), here 10 + the place in the table."""
    c = toy_composer(S=50, seed=11, rows=toy_row_table(frame_row=10 + np.arange(len(TOY_ROW_SPECS))))
    rng = np.random.default_rng([11, 13])
    a = c.heads.rate(np.array([3]))[0] * (c.rows.exposure_ms[3] / 1000.0)
    counts = rng.poisson(a, size=(50, 18))
    d = c.simulate_onsets(3)
    assert np.array_equal(np.bincount(d.sim * 18 + d.q, minlength=50 * 18).reshape(50, 18), counts)


def test_the_same_row_in_another_table_gets_the_same_draws_whatever_its_place_and_its_neighbours():
    """The generator is keyed by the global row number, not by the place in the table: a table of other rows, in another order, gives
    the same simulation for a row that both hold (heads whose laws do not depend on the place of the row)."""
    full = toy_composer(S=200, seed=6, rows=toy_row_table(frame_row=[40, 41, 42, 43, 44]))
    order = [3, 0, 4]
    sub = toy_composer(S=200, seed=6, rows=toy_row_table([TOY_ROW_SPECS[i] for i in order], frame_row=[40 + i for i in order]))
    a, b = full.simulate(order), sub.simulate([0, 1, 2])
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sd, b.sd) and np.array_equal(a.q19, b.q19)
    d1, d2 = full.simulate_onsets(4), sub.simulate_onsets(2)
    assert np.array_equal(d1.q, d2.q) and np.array_equal(d1.m, d2.m) and np.array_equal(d1.value, d2.value)
    other = toy_composer(S=200, seed=6, rows=toy_row_table(frame_row=[50, 51, 52, 53, 54]))             # other global numbers: other draws
    assert not np.array_equal(full.simulate([0]).mean, other.simulate([0]).mean)


def test_the_summary_is_the_mean_the_sample_sd_and_the_19_quantiles_of_the_values_of_the_horizon():
    c = rich_composer(S=400, seed=3)
    r = c.simulate([0, 2])
    assert r.S == 400 and r.mean.shape == (2, 3) and r.sd.shape == (2, 3) and r.q19.shape == (2, 3, 19)
    assert np.array_equal(compose.Q_LEVELS, np.arange(1, 20) / 20) and compose.HORIZON_BINS == (6, 12, 18)
    assert not compose.Q_LEVELS.flags.writeable
    for k, i in enumerate((0, 2)):
        d = c.simulate_onsets(i)
        V = np.zeros((400, 3))
        for j, n_bins in enumerate(H_BINS):
            sel = d.q < n_bins
            V[:, j] = np.bincount(d.sim[sel], weights=d.value[sel], minlength=400)
        assert r.mean[k] == pytest.approx(V.mean(axis=0), rel=1e-12, abs=1e-9)
        assert r.sd[k] == pytest.approx(V.std(axis=0, ddof=1), rel=1e-12, abs=1e-9)
        assert r.q19[k] == pytest.approx(np.quantile(V, np.arange(1, 20) / 20, axis=0).T, rel=1e-12, abs=1e-9)
        assert (np.diff(r.q19[k], axis=1) >= 0).all()


def test_the_value_of_an_onset_is_its_gold_plus_its_valued_objects_and_conversions():
    c = rich_composer(S=60, seed=5)
    d = c.simulate_onsets(0)
    gold = np.where(d.k >= 1, c.gold_reps[d.gold_bin], 0.0)
    assert np.array_equal(d.gold, gold)
    for j in range(0, len(d.q), max(1, len(d.q) // 25)):
        v = cell_values(c, 0, int(d.q[j]))
        assert d.value[j] == pytest.approx(d.gold[j, 0] - d.gold[j, 1] + (d.O[j] + d.conv[j]) @ v, rel=1e-12, abs=1e-9)


def test_the_onsets_are_ordered_by_simulation_and_bin_and_have_the_documented_columns():
    d = rich_composer(S=60, seed=5).simulate_onsets(0)
    n = len(d.q)
    assert n > 10
    shapes = (('sim', (n,)), ('q', (n,)), ('m', (n,)), ('z', (n,)), ('s', (n,)), ('k', (n, 2)), ('gold_bin', (n, 2)), ('gold', (n, 2)),
              ('O', (n, 18)), ('conv', (n, 18)), ('value', (n,)))
    for name, shape in shapes:
        assert getattr(d, name).shape == shape, name
    assert (np.diff(d.sim) >= 0).all() and ((d.q[1:] >= d.q[:-1]) | (d.sim[1:] > d.sim[:-1])).all()
    assert d.sim.max() < 60 and d.q.max() < 18 and d.m.max() < 52 and d.z.max() < 10 and d.s.max() < 4 and d.z.min() >= 0
    assert (d.gold_bin[d.k == 0] == -1).all() and (d.gold_bin[d.k >= 1] >= 0).all()


def test_no_simulation_has_an_onset_in_a_bin_without_exposure_even_when_the_rate_is_huge():
    c = toy_composer(S=200, heads=ToyHeads(rate=0.5), rows=rows_with_zero_tail_from_bin(12))
    bins = simulated_onset_bins(c)
    assert len(bins) > 1000 and bins.max() == 11 and bins.min() == 0


def test_a_row_without_any_exposure_has_no_onset_and_the_value_zero():
    t = toy_row_table([(1000.0, 5000.0)])
    zero = compose.RowTable(frame_row=t.frame_row, match=t.match, t0_ms=t.t0_ms, exposure_ms=np.zeros((1, 18), dtype=np.int64), T_ms=t.T_ms,
                            views=t.views, destroyed=t.destroyed)
    c = toy_composer(S=30, rows=zero, heads=ToyHeads(rate=5.0))
    r, e = c.simulate([0]), c.expect([0])
    assert len(c.simulate_onsets(0).q) == 0
    assert (r.mean == 0).all() and (r.sd == 0).all() and (r.q19 == 0).all() and (e == 0).all()


def z_test_sum(counts, means):
    counts, means = np.asarray(counts, dtype=np.float64).ravel(), np.asarray(means, dtype=np.float64).ravel()
    assert (counts[means == 0] == 0).all()
    keep = means > 0
    return float((counts[keep] - means[keep]).sum() / np.sqrt(means[keep].sum()))


def z_test_dispersion(counts, means):
    counts, means = np.asarray(counts, dtype=np.float64).ravel(), np.asarray(means, dtype=np.float64).ravel()
    keep = means > 0
    c, mu = counts[keep], means[keep]
    return float((((c - mu) ** 2 / mu).sum() - len(c)) / np.sqrt((2.0 + 1.0 / mu).sum()))


def chi2_p(observed, expected, floor=5.0):
    """p-value of Pearson's statistic; the categories whose expected count is below ``floor`` are pooled into one."""
    observed, expected = np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    small = expected < floor
    observed = np.append(observed[~small], observed[small].sum())
    expected = np.append(expected[~small], expected[small].sum())
    keep = expected > 0
    observed, expected = observed[keep], expected[keep]
    if len(expected) < 2:
        return 1.0
    return float(stats.chi2.sf(((observed - expected) ** 2 / expected).sum(), len(expected) - 1))


def pooled_chi2_p(codes, laws, groups):
    """The p-value of the chi-square statistics of all the groups added up (the degrees of freedom added up too): in a group the drawn codes
    are compared with the sum of the laws that the heads give the onsets of the group (conservative when the laws differ inside a group).
    Adding the groups up finds a small departure that is spread over many groups."""
    stat, df = 0.0, 0
    for g in np.unique(groups):
        sel = groups == g
        observed = np.bincount(codes[sel], minlength=laws.shape[1]).astype(np.float64)
        expected = laws[sel].sum(axis=0)
        small = expected < 5.0
        observed, expected = np.append(observed[~small], observed[small].sum()), np.append(expected[~small], expected[small].sum())
        keep = expected > 0
        observed, expected = observed[keep], expected[keep]
        if len(expected) >= 2:
            stat += float(((observed - expected) ** 2 / expected).sum())
            df += len(expected) - 1
    return float(stats.chi2.sf(stat, df)) if df else 1.0


@pytest.fixture(scope='module')
def rich_draws():
    c = rich_composer(S=3000, seed=21)
    return c, {i: c.simulate_onsets(i) for i in range(3)}


def test_the_number_of_onsets_of_a_bin_is_poisson_with_the_rate_times_the_exposure(rich_draws):
    c, draws = rich_draws
    for i, d in draws.items():
        counts = np.bincount(d.sim * 18 + d.q, minlength=3000 * 18).reshape(3000, 18)
        a = c.heads.rate(np.array([i]))[0] * c.rows.exposure_ms[i] / 1000.0
        assert (counts[:, a == 0] == 0).all()
        expected = a
        assert abs(z_test_sum(counts.sum(axis=0), np.full(18, 3000.0) * expected)) < 4.5
        assert abs(z_test_dispersion(counts, np.tile(expected, (3000, 1)))) < 4.5


def test_the_mark_the_zone_and_the_size_are_drawn_from_the_laws_of_the_heads(rich_draws):
    c, draws = rich_draws
    h = c.heads
    for i, d in draws.items():
        r = np.full(len(d.q), i)
        assert pooled_chi2_p(d.m, h.mark_prob(r, d.q), d.q) > 1e-6
        assert pooled_chi2_p(d.z, h.zone_prob(r, d.q, d.m), d.m) > 1e-6
        assert pooled_chi2_p(d.s, h.size_prob(r, d.q, d.m, d.z), d.z) > 1e-6


def test_the_kills_are_poisson_with_the_means_of_the_kill_head(rich_draws):
    c, draws = rich_draws
    for i, d in draws.items():
        mu = c.heads.kill_rate(np.full(len(d.q), i), d.q, d.m, d.z, d.s)
        for team in (BLUE, RED):
            assert abs(z_test_sum(d.k[:, team], mu[:, team])) < 4.5
            assert abs(z_test_dispersion(d.k[:, team], mu[:, team])) < 4.5
    assert max(d.k.max() for d in draws.values()) >= 6                                      # the layer cap is used


def test_the_gold_bin_of_a_team_is_drawn_at_the_layer_of_its_kills_capped_at_six(rich_draws):
    c, draws = rich_draws
    for i, d in draws.items():
        r = np.full(len(d.q), i)
        layer = np.clip(d.k, 1, 6)
        pg = c.heads.gold_prob(r, d.q, d.m, d.z, d.s, layer)
        for team in (BLUE, RED):
            has = d.k[:, team] >= 1
            assert pooled_chi2_p(d.gold_bin[has, team], pg[has, team], layer[has, team]) > 1e-6, (i, team)


def mid_pit(p, codes):
    """The middle of the cumulative-probability interval of each drawn category: uniform on (0, 1) when the draw follows the law,
    whatever the law; two draws that are independent have uncorrelated mid-points."""
    cdf = np.cumsum(p, axis=1)
    at = np.arange(len(codes))
    lo = np.where(codes > 0, cdf[at, np.maximum(codes - 1, 0)], 0.0)
    return 0.5 * (lo + cdf[at, codes])


def test_the_two_teams_draw_their_kills_and_their_gold_bins_independently_of_each_other(rich_draws):
    c, draws = rich_draws
    for i, d in draws.items():
        r = np.full(len(d.q), i)
        both = (d.k >= 1).all(axis=1)
        pg = c.heads.gold_prob(r, d.q, d.m, d.z, d.s, np.clip(d.k, 1, 6))
        a, b = mid_pit(pg[both, BLUE], d.gold_bin[both, BLUE]), mid_pit(pg[both, RED], d.gold_bin[both, RED])
        assert abs(np.corrcoef(a, b)[0, 1]) < 4.5 / np.sqrt(both.sum()), i
        mu = c.heads.kill_rate(r, d.q, d.m, d.z, d.s)
        ra, rb = (d.k[:, BLUE] - mu[:, BLUE]) / np.sqrt(mu[:, BLUE]), (d.k[:, RED] - mu[:, RED]) / np.sqrt(mu[:, RED])
        assert abs(np.corrcoef(ra, rb)[0, 1]) < 4.5 / np.sqrt(len(d.q)), i


def test_the_objects_and_the_conversions_are_poisson_with_the_means_of_their_heads(rich_draws):
    c, draws = rich_draws
    h = c.heads
    for i, d in draws.items():
        r = np.full(len(d.q), i)
        K = np.column_stack([d.k, d.gold])
        lam_o = h.obj_rate(r, d.q, d.m, d.z, d.s, K)
        lam_c = h.conv_rate(features.m4_masked_inputs(h.row_view(r), d.q, d.m, d.z, K, d.O))
        cexp = compose.conv_exposure_s(c.rows.t0_ms[i], d.q, c.rows.T_ms[i])
        assert abs(z_test_sum(d.O, lam_o)) < 4.5 and abs(z_test_sum(d.conv, lam_c * cexp[:, None])) < 4.5


class RecordingKO(KOToyHeads):
    def __init__(self):
        super().__init__()
        self.K_seen, self.x_seen = [], []

    def obj_rate(self, rows, q, m, z, s, K):
        self.K_seen.append(np.array(K))
        return super().obj_rate(rows, q, m, z, s, K)

    def conv_rate(self, m4_inputs):
        self.x_seen.append(np.array(m4_inputs))
        return super().conv_rate(m4_inputs)


def test_the_object_head_gets_the_sampled_kills_and_gold_and_the_conversion_head_the_masked_input_of_the_draw():
    h = RecordingKO()
    c = toy_composer(S=300, seed=8, heads=h)
    d = c.simulate_onsets(0)
    K = np.column_stack([d.k, d.gold])
    assert len(h.K_seen) == 1 and np.array_equal(h.K_seen[0], K)                          # the sampled kills and the gold of the drawn bins
    assert set(np.unique(d.gold)) <= set(TOY_GOLD_REPS.tolist()) | {0.0}
    want = features.m4_masked_inputs(h.row_view(np.zeros(len(d.q), dtype=np.int64)), d.q, d.m, d.z, K, d.O)
    assert len(h.x_seen) == 1 and np.array_equal(h.x_seen[0], want) and h.x_seen[0].dtype == np.float32      # only m4_masked_inputs
    assert (h.x_seen[0][:, 5:9] == np.column_stack([d.k, d.gold / 1000.0]).astype(np.float32)).all() and (h.x_seen[0][:, 9:] == d.O).all()


def ko_expected(t0_s, T_s, n_bins):
    """E[V_H] of KOToyHeads by hand (note the layout of the section text of s3_toys): per mark the kill gold, the dragons that follow the
    kills and the tower conversions that follow the dragons, the red kills and the blue gold."""
    total = 0.0
    for q in range(n_bins):
        seconds = min(max(min(t0_s + 180.0, T_s - 30.0) - (t0_s + 10.0 * q), 0.0), 10.0)
        cexp = max(0.0, min(60.0, T_s - (t0_s + (q + 0.5) * 10.0 + 20.0)))
        per_mark = 0.0
        for weight, blue_first in ((0.6, True), (0.4, False)):
            mu_b, mu_r = (math.log(2.0), math.log(4 / 3)) if blue_first else (math.log(4 / 3), math.log(2.0))
            rep_b, rep_r = (600.0, 400.0) if blue_first else (400.0, 600.0)
            gold_b, gold_r = (1 - math.exp(-mu_b)) * rep_b, (1 - math.exp(-mu_r)) * rep_r
            o0, o1 = 0.1 * mu_b, 0.05 * mu_r
            conv12 = cexp * (0.001 * o0 + 0.0005 * mu_r)
            conv13 = cexp * 0.002 * gold_b / 1000.0
            per_mark += weight * (gold_b - gold_r + 200.0 * (o0 - o1) + 400.0 * (conv12 - conv13))
        total += 0.01 * seconds * per_mark
    return total


def test_a_simulation_through_heads_that_read_k_and_o_agrees_with_the_mean_worked_out_by_hand():
    c = toy_composer(S=12_000, seed=13, heads=KOToyHeads())
    r = c.simulate(TOY_ROWS)
    for i, (t0, T) in enumerate(TOY_ROW_SPECS):
        for j, n_bins in enumerate(H_BINS):
            want = ko_expected(t0, T, n_bins)
            assert abs(r.mean[i, j] - want) < 4.0 * r.sd[i, j] / np.sqrt(12_000) + 1e-9, (i, j)


class ConvOnlyHeads(ToyHeads):
    """No kills and no objects; the conversion rate is 5 per second in the blue dragon cell, so that the conversions show the exposure."""

    def kill_rate(self, rows, q, m, z, s):
        return np.zeros((len(self._rows(rows)), 2))

    def obj_rate(self, rows, q, m, z, s, K):
        return np.zeros((len(self._rows(rows)), 18))

    def conv_rate(self, m4_inputs):
        out = np.zeros((len(np.asarray(m4_inputs)), 18))
        out[:, 0] = 5.0
        return out


def test_the_conversions_of_an_onset_are_poisson_with_the_rate_times_the_exposure_of_its_bin():
    c = toy_composer(S=2000, seed=17, heads=ConvOnlyHeads(), rows=toy_row_table([(1000.0, 1100.0)]))
    d = c.simulate_onsets(0)
    for q in range(8):
        sel = d.q == q
        if not sel.any():
            continue
        cexp = compose.conv_exposure_s(1_000_000, q, 1_100_000)
        assert abs(z_test_sum(d.conv[sel, 0], np.full(int(sel.sum()), 5.0 * cexp))) < 4.5, q
    assert (d.conv[:, 1:] == 0).all()


def test_a_conversion_without_exposure_is_zero_whatever_the_rate():
    """A hand-made row whose game ends one second after t0: every conversion window is empty, although the occurrence bins are full."""
    t = toy_row_table([(1000.0, 5000.0)])
    short = compose.RowTable(frame_row=t.frame_row, match=t.match, t0_ms=t.t0_ms, exposure_ms=t.exposure_ms, T_ms=t.t0_ms + 1000,
                             views=t.views, destroyed=t.destroyed)
    c = toy_composer(S=100, heads=ConvOnlyHeads(), rows=short)
    assert (c.simulate_onsets(0).conv == 0).all() and len(c.simulate_onsets(0).q) > 100
    assert c.simulate([0]).mean[0, 0] == 0.0 and c.expect([0])[0, 0] == 0.0


def test_the_simulation_agrees_with_the_closed_form_on_heads_that_differ_in_every_argument():
    c = rich_composer(S=6000, seed=31)
    r, e = c.simulate([0, 1, 2]), c.expect([0, 1, 2])
    z = (r.mean - e) / (r.sd / np.sqrt(6000))
    assert np.abs(z).max() < 4.0, z
    assert (np.abs(e) > 1.0).all()


# =====================================================================================================================================
# block sizes and workers
# =====================================================================================================================================
def test_the_block_size_does_not_change_a_bit_for_heads_that_differ_in_every_argument():
    c = rich_composer(S=120, seed=2, specs=SHORT_SPECS)
    ref_e, ref_s = c.expect([0, 1, 2], block=100), c.simulate([0, 1, 2], block=100)
    for block in (1, 2):
        assert np.array_equal(c.expect([0, 1, 2], block=block), ref_e)
        s = c.simulate([0, 1, 2], block=block)
        assert np.array_equal(s.mean, ref_s.mean) and np.array_equal(s.sd, ref_s.sd) and np.array_equal(s.q19, ref_s.q19)


def test_worker_processes_give_the_same_bits_as_one_process():
    rows = np.arange(5)
    ref = run_blocks(rows, block=1000, workers=1)
    assert np.array_equal(ref, run_blocks(rows, block=1, workers=3))
    assert np.array_equal(ref, run_blocks(rows, block=2, workers=2))


def test_the_number_of_workers_is_limited_to_what_a_process_pool_of_windows_can_hold():
    assert compose.MAX_WORKERS == 61
    with pytest.raises(ValueError, match='at most'):
        toy_composer(S=20).expect([0], workers=compose.MAX_WORKERS + 1)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True, '2'])
def test_blocks_and_workers_must_be_positive_integers(bad):
    c = toy_composer(S=20)
    with pytest.raises((ValueError, TypeError)):
        c.expect([0], block=bad)
    with pytest.raises((ValueError, TypeError)):
        c.simulate([0], workers=bad)


class EnvHeads(ToyHeads):
    """ToyHeads whose onset rate is three times higher in a process without the worker environment (a probe of what a process inherits)."""

    def rate(self, rows):
        ours = all(os.environ.get(k) == v for k, v in compose.WORKER_ENVIRONMENT.items())
        return super().rate(rows) * (1.0 if ours else 3.0)


def test_the_worker_environment_is_one_thread_and_the_compatible_branch_of_mkl():
    assert compose.WORKER_ENVIRONMENT['MKL_CBWR'] == 'COMPATIBLE' and compose.WORKER_ENVIRONMENT['MKL_NUM_THREADS'] == '1'
    assert compose.WORKER_ENVIRONMENT['OMP_NUM_THREADS'] == '1' and compose.WORKER_ENVIRONMENT['OPENBLAS_NUM_THREADS'] == '1'


def test_the_worker_environment_is_set_inside_and_put_back_outside(monkeypatch):
    monkeypatch.delenv('MKL_CBWR', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    with compose._worker_environment():
        assert all(os.environ[k] == v for k, v in compose.WORKER_ENVIRONMENT.items())
    assert 'MKL_CBWR' not in os.environ and os.environ['OMP_NUM_THREADS'] == '7'
    with pytest.raises(RuntimeError):
        with compose._worker_environment():
            raise RuntimeError('stop')
    assert 'MKL_CBWR' not in os.environ and os.environ['OMP_NUM_THREADS'] == '7'                # also when the block fails


def test_worker_processes_inherit_the_worker_environment_and_the_calling_process_keeps_its_own(monkeypatch):
    for k in compose.WORKER_ENVIRONMENT:
        monkeypatch.delenv(k, raising=False)
    c = toy_composer(heads=EnvHeads())
    in_parent = c.expect([0, 1])                                                       # no variables here: the rate factor is 3
    in_workers = c.expect([0, 1], block=1, workers=2)
    assert not [k for k in compose.WORKER_ENVIRONMENT if k in os.environ]               # nothing is left behind in the parent
    for k, v in compose.WORKER_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    want = c.expect([0, 1])                                                            # with the variables: the factor is 1
    assert np.array_equal(in_workers, want) and not np.allclose(in_parent, want)


def test_a_row_whose_onset_rate_is_absurd_is_refused_before_its_draws_but_has_a_closed_form():
    c = toy_composer(S=1000, heads=ToyHeads(rate=50.0))
    with pytest.raises(ValueError, match='rate'):
        c.simulate([0])
    with pytest.raises(ValueError, match='rate'):
        c.simulate_onsets(0)
    assert np.isfinite(c.expect([0])).all()                                            # the closed form has no memory problem
    assert compose.MAX_ONSETS_PER_ROW == 1_000_000


# =====================================================================================================================================
# the structure of the source (A3 and the rules of the existing code)
# =====================================================================================================================================
def test_the_source_names_none_of_the_event_arrays():
    text = SOURCE.read_text(encoding='utf-8')
    assert [w for w in FORBIDDEN if w in text] == []


def test_the_source_has_no_match_key_no_holdout_switch_and_no_pickle():
    text = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(text)
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert 'matches/key' not in text and 'allow_holdout' not in text
    assert not {n for n in all_imports(SOURCE) if any(w in n for w in ('pickle', 'joblib', 'dill', 'shelve', 'marshal'))}
    assert not calls & {'dump', 'load', 'save', 'savez'}


def test_the_module_level_imports_are_light_and_do_not_include_the_raw_readers():
    imports = module_level_imports(SOURCE)
    assert not imports & {'c_ept.run_s1', 'c_ept.reader', 'c_ept.units', 'c_ept.terrain', 'c_ept.value', 'c_ept.targets', 'c_ept.cohort'}
    assert {'c_ept.features', 'c_ept.towertier', 'c_ept.heads'} <= imports


def test_the_source_gives_its_references():
    assert '# Ref:' in SOURCE.read_text(encoding='utf-8')


# =====================================================================================================================================
# the real B0 heads on the heads world
# =====================================================================================================================================
def world_row_table(w, role):
    """The row table of the frame rows of ``role`` of the heads world (it has no tower events: nothing destroyed)."""
    s1, s2 = w.s1, w.s2
    rows = np.flatnonzero(features.split_rows(s1, role))
    match = np.asarray(s1.array('frame_rows/match'))[rows].astype(np.int64)
    src, index = np.asarray(s2.array('views/src')), np.asarray(s2.array('views/src_index'))
    view_of = {int(i): v for v, (s, i) in enumerate(zip(src, index)) if s == 0}
    return compose.RowTable(frame_row=rows, match=match, t0_ms=np.asarray(s1.array('frame_rows/t0'))[rows],
                            exposure_ms=np.asarray(s1.array('frame_rows/exposure_ms'))[rows],
                            T_ms=np.asarray(s1.array('matches/T_ms'))[match], views=[view_of[int(r)] for r in rows],
                            destroyed=np.zeros((len(rows), 2, 4), dtype=np.int64), roles=(role,))


@pytest.fixture(scope='module')
def b0_composer(heads_world):
    from c_ept.models import baselines_b0
    fitted = baselines_b0.B0Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, heads_world.fit_means)
    rows = world_row_table(heads_world, 'DEV')
    return compose.Composer(fitted.composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, rows, heads_world.s2,
                            S=4000, seed=20261021)


def test_the_b0_heads_are_a_closed_form_head_set_and_the_composer_takes_them(b0_composer):
    h = b0_composer.heads
    assert isinstance(h, compose.ComposerHeads) and h.analytic_ok and not h.obj_uses_K and not h.conv_uses_KO
    assert len(b0_composer.rows) >= 8


def test_the_b0_closed_form_is_finite_and_follows_the_sum_of_its_onset_rates(b0_composer):
    e = b0_composer.expect([0, 1])
    assert e.shape == (2, 3) and np.isfinite(e).all()
    n = b0_composer.expected_onsets([0, 1, 2])
    assert (n > 0).all() and (np.diff(n, axis=1) >= 0).all()


def sane_rows(c, k=3):
    """The first ``k`` rows of the table whose heads give between 0.3 and 15 expected onsets in 180 s: the B0 heads that are fitted on the
    random heads world give rates between 0.001 and 600,000 onsets for its rows (extrapolation of a log-linear model on random features),
    and a simulation of the latter would need gigabytes."""
    eo = c.expected_onsets(np.arange(len(c.rows)))[:, 2]
    rows = np.flatnonzero((eo > 0.3) & (eo < 15.0))
    assert len(rows) >= k
    return rows[:k].tolist()


def test_the_b0_closed_form_equals_the_brute_force_oracle_on_rows_with_few_bins(b0_composer):
    """The second route of ``oracle_expect`` with the real B0 heads (39 columns of features, the real M4 input) on the rows at the end of a
    game, which have a few bins with exposure."""
    c = b0_composer
    few = [i for i in range(len(c.rows)) if 0 < (c.rows.exposure_ms[i] > 0).sum() <= 5]
    assert len(few) >= 2
    for i in few[:2]:
        assert c.expect([i])[0] == pytest.approx(oracle_expect(c, i), rel=1e-9), i


def test_the_size_of_a_call_does_not_change_the_b0_closed_form_beyond_rounding(b0_composer, monkeypatch):
    idx = sane_rows(b0_composer, 2)
    ref = b0_composer.expect(idx)
    for rows in (97, 10 ** 9):
        monkeypatch.setattr(compose, 'CALL_ROWS', rows)
        assert b0_composer.expect(idx) == pytest.approx(ref, rel=1e-12), rows


def test_the_b0_simulation_agrees_with_the_b0_closed_form(b0_composer):
    idx = sane_rows(b0_composer, 3)
    r, e = b0_composer.simulate(idx), b0_composer.expect(idx)
    # G-S3b B2-a: with a state vector for each window the B0 heads of the random world give some rows a first-minute rate of 5e-9: no draw
    # has an onset in 60 s, the simulation has sd 0 there and no standard error, and the closed form is the value of an event that
    # almost never happens (-1e-5 gold). Such a cell is held to 0.01 gold; the others to four standard errors, as before.
    flat = r.sd == 0.0
    z = (r.mean - e) / np.where(flat, 1.0, r.sd / np.sqrt(b0_composer.S))
    assert np.abs(z[~flat]).max(initial=0.0) < 4.0, z
    assert (np.abs(r.mean - e)[flat] < 0.01).all(), (r.mean, e)


def test_the_b0_closed_form_does_not_depend_on_the_block_size(b0_composer):
    idx = sane_rows(b0_composer, 3)
    a, b = b0_composer.expect(idx, block=1), b0_composer.expect(idx, block=50)
    assert np.array_equal(a, b)


def test_the_b0_simulation_does_not_depend_on_the_block_size(b0_composer):
    idx = sane_rows(b0_composer, 3)
    a, b = b0_composer.simulate(idx, block=1), b0_composer.simulate(idx, block=50)
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.q19, b.q19)


def test_the_b0_closed_form_is_the_same_bit_for_bit_for_every_block_size_and_number_of_workers(b0_composer):
    """B0 calls MKL, whose products depend on the thread count and the alignment of the operands: every block, also a single one, runs in
    a worker (``compose.WORKER_ENVIRONMENT``), so the blocks agree with each other to the last bit; the calling process agrees to
    rounding."""
    idx = sane_rows(b0_composer, 4)
    per_row = b0_composer.expect(idx, block=1, workers=2)
    assert np.array_equal(per_row, b0_composer.expect(idx, block=len(idx), workers=2))          # one block, with workers: still a worker
    assert np.array_equal(per_row, b0_composer.expect(idx, block=2, workers=3))
    assert per_row == pytest.approx(b0_composer.expect(idx), rel=1e-12)


def test_the_b0_simulation_is_the_same_with_worker_processes(b0_composer):
    idx = sane_rows(b0_composer, 4)
    one, two = b0_composer.simulate(idx), b0_composer.simulate(idx, block=1, workers=2)
    assert np.array_equal(one.mean, two.mean) and np.array_equal(one.sd, two.sd) and np.array_equal(one.q19, two.q19)


# =====================================================================================================================================
# the real B1 heads (LightGBM) on the heads world: the path that reads K and O
# =====================================================================================================================================
B1_SMALL = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 30, 'min_data_per_group': 5, 'cat_smooth': 1.0}


@pytest.fixture(scope='module')
def b1_composer(heads_world):
    from c_ept.models import baselines_b1
    dropped = features.dropped_columns(heads_world.s2, heads_world.fit_views)
    fitted = baselines_b1.B1Heads.fit(heads_world.s1, heads_world.s2, heads_world.labels, dropped, params=B1_SMALL)
    rows = world_row_table(heads_world, 'DEV')
    return compose.Composer(fitted.composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, rows, heads_world.s2,
                            S=30, seed=20261021)


def test_the_b1_heads_have_no_closed_form_and_the_composer_refuses_the_expectation(b1_composer):
    h = b1_composer.heads
    assert isinstance(h, compose.ComposerHeads) and h.feature_kind == 'b1'
    assert not h.analytic_ok and h.obj_uses_K and h.conv_uses_KO
    with pytest.raises(ValueError, match='closed form'):
        b1_composer.expect([0])


def test_the_b1_heads_are_simulated_through_the_sampled_kills_and_objects(b1_composer):
    idx = [0, 1]
    r = b1_composer.simulate(idx)
    assert r.mean.shape == (2, 3) and r.sd.shape == (2, 3) and r.q19.shape == (2, 3, 19) and r.S == 30
    assert np.isfinite(r.mean).all() and np.isfinite(r.sd).all() and np.isfinite(r.q19).all()
    d = b1_composer.simulate_onsets(0)
    assert d.O.shape[1] == 18 and d.conv.shape[1] == 18 and len(d.q) > 0
    again = b1_composer.simulate(idx)
    assert np.array_equal(r.mean, again.mean) and np.array_equal(r.q19, again.q19)
    assert np.array_equal(r.mean, b1_composer.simulate(idx, block=1).mean)


# =====================================================================================================================================
# fix round 1 of Task 9: every block in the pool, the tower prices of each row, the E6 hook
# =====================================================================================================================================
def test_with_workers_every_block_runs_in_the_pool_also_a_single_one():
    """The rate of ``ProcessProbeHeads`` is three times higher in the process that made the heads: a block that is computed here has the
    factor 3, a block in a worker the factor 1 (R30: with workers the calling process computes no value)."""
    c = toy_composer(S=60, heads=ProcessProbeHeads())
    here = c.expect([0, 1])
    one_block = c.expect([0, 1], block=1000, workers=2)                                     # a single block, with two workers
    assert one_block == pytest.approx(here / 3.0, rel=1e-12)
    sim_here, sim_worker = c.simulate([0, 1]), c.simulate([0, 1], block=1000, workers=2)
    assert not np.array_equal(sim_here.mean, sim_worker.mean)
    onsets_here, onsets_worker = c.expected_onsets([0, 1]), c.expected_onsets([0, 1], block=1000, workers=2)
    assert onsets_here == pytest.approx(3.0 * onsets_worker, rel=1e-12)


def test_the_b0_expected_onsets_are_the_same_bit_for_bit_for_one_block_and_one_block_per_row_with_workers(b0_composer):
    idx = list(range(6))
    one = b0_composer.expected_onsets(idx, block=len(idx), workers=2)
    assert np.array_equal(one, b0_composer.expected_onsets(idx, block=1, workers=2))
    assert one == pytest.approx(b0_composer.expected_onsets(idx), rel=1e-12)


def test_the_b1_simulation_is_the_same_bit_for_bit_for_one_block_and_one_block_per_row_with_workers(b1_composer):
    idx = [0, 1, 2, 3]
    per_row, one = b1_composer.simulate(idx, block=1, workers=2), b1_composer.simulate(idx, block=len(idx), workers=2)
    assert np.array_equal(per_row.mean, one.mean) and np.array_equal(per_row.sd, one.sd) and np.array_equal(per_row.q19, one.q19)


def test_a_row_priced_alone_is_valued_exactly_as_in_a_larger_table_bit_for_bit():
    """The tier law is asked for each row on its own, two calls (one for each team) with the 18 bins of that row, whatever the table is. The
    law of ``CountingTier`` changes with the number of its inputs, as the last bit of a library product can: a table-wide call would give
    a row alone and the same row in a table of three different values."""
    for batch_effect in (1e-9, 0.0):
        tier = CountingTier(hand_tier_model(), batch_effect)
        big = toy_composer(S=300, seed=7, rows=toy_row_table(RICH_SPECS, STATE), prices=RICH_PRICES, tier_model=tier)
        e_big, s_big = big.expect([0, 1, 2]), big.simulate([0, 1, 2])
        for i in range(3):
            alone = toy_composer(S=300, seed=7, rows=toy_row_table([RICH_SPECS[i]], STATE[i:i + 1], frame_row=[i]), prices=RICH_PRICES,
                                 tier_model=tier)
            a = alone.simulate([0])
            assert np.array_equal(alone.expect([0]), e_big[[i]]), (batch_effect, i)
            assert np.array_equal(a.mean, s_big.mean[[i]]) and np.array_equal(a.sd, s_big.sd[[i]]), (batch_effect, i)
            assert np.array_equal(a.q19, s_big.q19[[i]]), (batch_effect, i)


def test_the_tier_law_is_asked_twice_for_each_row_with_the_18_bins_and_the_parent_asks_nothing_when_workers_compute():
    spy = CountingTier(hand_tier_model())
    heads = CountingToyHeads()
    c = toy_composer(S=60, heads=heads, rows=toy_row_table(RICH_SPECS, STATE), prices=RICH_PRICES, tier_model=spy)
    assert spy.n_calls == 0 and not heads.n_calls                                         # making the composer computes no value
    c.expect([0, 1, 2])                                                                  # one worker: in this process
    assert spy.n_calls == 6 and spy.sizes == [18] * 6 and heads.n_calls['rate'] == 3     # two calls of 18 inputs for each of the three rows
    c.simulate([0, 1, 2])
    assert spy.n_calls == 12 and spy.sizes == [18] * 12
    rates = heads.n_calls['rate']
    c.expected_onsets([0, 1, 2], block=3)                                  # the rate head is asked for each row, not for the block
    assert heads.n_calls['rate'] == rates + 3
    before, counted = spy.n_calls, dict(heads.n_calls)
    c.expect([0, 1, 2], block=2, workers=2)                                          # with workers: the calling process computes nothing
    assert spy.n_calls == before and dict(heads.n_calls) == counted


# ---- the E6 hook -------------------------------------------------------------------------------------------------------------------------
class FakeCheck:
    """A fake E6 check: it counts its calls, notes each of them in ``events`` and raises on the call number ``stop_at``."""

    def __init__(self, stop_at=None, events=None):
        self.calls, self.stop_at = 0, stop_at
        self.events = [] if events is None else events

    def __call__(self):
        self.calls += 1
        self.events.append('check')
        if self.calls == self.stop_at:
            raise RuntimeError('E6: a model server runs')


@pytest.mark.parametrize('method', ['expect', 'simulate', 'expected_onsets'])
def test_the_e6_check_runs_before_every_block_in_this_process(method):
    events = []
    c = toy_composer(S=20, heads=EventToyHeads(events))
    check = FakeCheck(events=events)
    getattr(c, method)([0, 1, 2, 3, 4], block=2, check=check)
    assert events == ['check', 'row0', 'row1', 'check', 'row2', 'row3', 'check', 'row4'] and check.calls == 3


@pytest.mark.parametrize('method', ['expect', 'simulate', 'expected_onsets'])
def test_a_refusal_of_the_e6_check_stops_the_run_and_no_further_block_is_computed_in_this_process(method):
    events = []
    c = toy_composer(S=20, heads=EventToyHeads(events))
    with pytest.raises(RuntimeError, match='E6'):
        getattr(c, method)([0, 1, 2, 3, 4], block=1, check=FakeCheck(stop_at=3, events=events))
    assert events == ['check', 'row0', 'check', 'row1', 'check']


def test_the_e6_check_runs_before_every_block_that_is_dispatched_to_the_workers():
    check = FakeCheck()
    c = toy_composer(S=20)
    got = c.expect([0, 1, 2, 3, 4], block=2, workers=2, check=check)
    assert check.calls == 3 and np.array_equal(got, c.expect([0, 1, 2, 3, 4]))


def test_a_refusal_of_the_e6_check_dispatches_no_further_block_to_the_workers(tmp_path):
    log = tmp_path / 'rows.log'
    c = toy_composer(S=20, heads=FileLogToyHeads(log))
    check = FakeCheck(stop_at=3)
    with pytest.raises(RuntimeError, match='E6'):
        c.expect([0, 1, 2, 3, 4], block=1, workers=2, check=check)
    done = {int(x) for x in log.read_text().split()} if log.exists() else set()
    assert check.calls == 3 and done and done <= {0, 1}                                  # at most the two blocks that were in flight


def test_the_e6_check_runs_before_each_block_is_submitted_to_a_worker_and_a_refusal_submits_no_more(monkeypatch):
    """The order of the events in the calling process, seen by a pool that logs its submissions (the workers still compute): check, then
    submit, for each block; a refusal at the third check ends the run before the third submission. A check after the submission would
    show as 'submit', 'check' here; the rows that the workers computed cannot tell it (a block that has not started is cancelled)."""
    events = []

    class LoggingPool(compose.cf.ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            events.append('submit')
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(compose.cf, 'ProcessPoolExecutor', LoggingPool)
    c = toy_composer(S=20)
    c.expect([0, 1, 2, 3, 4], block=2, workers=2, check=FakeCheck(events=events))
    assert events == ['check', 'submit', 'check', 'submit', 'check', 'submit']
    del events[:]
    with pytest.raises(RuntimeError, match='E6'):
        c.expect([0, 1, 2, 3, 4], block=1, workers=2, check=FakeCheck(stop_at=3, events=events))
    assert events == ['check', 'submit', 'check', 'submit', 'check']


def test_the_e6_check_must_be_a_function_or_none_and_compose_does_not_import_run_s1():
    c = toy_composer(S=20)
    for bad in (5, 'check', [lambda: None]):
        with pytest.raises(TypeError, match='check'):
            c.expect([0], check=bad)
    assert c.expect([0], check=None).shape == (1, 3) and c.expected_onsets([0], check=lambda: None).shape == (1, 3)
    assert 'c_ept.run_s1' not in all_imports(SOURCE)                                      # the runner passes run_s1.require_no_strata
