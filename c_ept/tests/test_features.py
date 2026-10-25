"""Tests of ``c_ept.features``: the data splits and match weights, the columns to drop, the shared baseline features b0 (39 columns) and b1,
and the only builder of the masked M4 input (stage S3 Task 5). The NLL scores of the same task are tested in ``test_nll.py``.

No run folder, raw file or catalog is read. The stores are toy ones: the S1 and S2 stores of ``s3_toys`` (``toy_s1``, ``toy_s2``: 49 views,
a constant summary column, a constant player column, players without a position; the ``TOY_*`` numbers are worked out by hand there), the
S1 store of the value toys (it has units and onsets) and small stores of explicit arrays (``build_feature_store``). Truth comes from four
places:

    hand numbers   a one-view store whose b0 vector is worked out below column by column, the geodesic numbers of ``TOY_GEO_HAND``
    oracles        plain-Python implementations written here from the text of the brief (a loop per view and per column, no numpy
                   reduction over teams), compared with the code on every view of the toy
    flowprice      the window controls of ``flowprice.window_table`` are the same blue-minus-red differences, phase indicators and buff
                   indicators that b0 holds; the two modules were written independently
    refusals       stores and arguments that cannot be trusted: TEST and EXTERNAL and the roles stage S3 does not use, a name that is no
                   column, a masked-M4 argument out of its range, a NaN in b0 without the FIT means

The four tests of the task brief that concern this module come first, word for word (the first, ``test_drops_constant_columns_and_t_s``,
passes the FIT views of the toy, as ruling R14 amends it: ``dropped_columns`` has no default for its views).
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.preprocessing import features
from c_ept.valuation import flowprice
from c_ept.domain import targets
from c_ept.preprocessing import tokens
from c_ept.data.store_core import Store
from c_ept.tests.conftest import module_level_imports
from c_ept.tests.s3_toys import (   # noqa: F401  (the fixtures are imported by ruling R1)
    JUNK, KEY_PREFIX, TOY_ALL_VIEWS, TOY_CONSTANT_COLUMN, TOY_CONSTANT_PLAYER_COLUMN, TOY_DEV_MATCHES, TOY_DROPPED_ALL, TOY_DROPPED_FIT,
    TOY_EVENT_VIEWS, TOY_FIT_CONSTANT_COLUMN, TOY_FIT_MATCHES, TOY_FIT_VIEWS, TOY_FRAME_VIEWS, TOY_GEO_HAND, TOY_MISSING,
    TOY_N_FIT_FRAME_ROWS, TOY_N_VIEWS, TOY_NAN_FIT_VIEWS, TOY_NAN_VIEWS, TOY_ROW_OFF, TOY_STOP_MATCHES, TOY_VIEWS, blank_feature_arrays,
    build_feature_store, toy_s1, toy_s2, toy_value_s1)

SOURCE = Path(features.__file__)
SUM_I = {name: i for i, name in enumerate(tokens.SUMMARY)}
PLR_I = {name: i for i, name in enumerate(tokens.PLAYER_NUM)}
X_ARRAYS = {'x/summary', 'x/players_num', 'x/geo_pp'}
EXPECTED_B0_NAMES = (
    'game_time_min', 'phase_1', 'phase_2', 'phase_3', 'phase_4', 'phase_5', 'phase_6', 'phase_7',
    'diff_totalGold', 'diff_xp', 'diff_level', 'diff_minionsKilled', 'diff_jungleMinionsKilled', 'diff_alive', 'diff_respawn_s',
    'diff_item_gold_owned', 'alive_total', 'geo_min', 'geo_mean',
    'diff_towers_outer', 'diff_towers_inner', 'diff_towers_base', 'diff_towers_nexus', 'diff_inhibs_down', 'diff_plates', 'diff_dragons',
    'diff_elders', 'diff_barons', 'diff_heralds', 'diff_atakhans', 'diff_voidgrubs', 'diff_has_soul', 'diff_baron_buff', 'diff_elder_buff',
    'dragon_up', 'baron_up', 'herald_up', 'atakhan_up', 'voidgrub_up')
# the cells that b0 reads, listed here by hand: the summary columns (game time, blue/red pairs, buff seconds, flags) and the player columns
PAIRS = ('towers_outer', 'towers_inner', 'towers_base', 'towers_nexus', 'inhibs_down', 'plates_top', 'plates_mid', 'plates_bot', 'dragons',
         'elders', 'barons', 'heralds', 'atakhans', 'voidgrubs', 'has_soul')
BUFFS = ('blue_baron_buff_s', 'red_baron_buff_s', 'blue_elder_buff_s', 'red_elder_buff_s')
USED_SUMMARY = {'game_time_s', 'dragon_up', 'baron_up', 'herald_up', 'atakhan_up', 'voidgrub_up', *BUFFS,
                *(f'{side}_{name}' for side in ('blue', 'red') for name in PAIRS)}
USED_PLAYER = {'totalGold', 'xp', 'level', 'minionsKilled', 'jungleMinionsKilled', 'alive', 'respawn_s', 'item_gold_owned'}
PHASE_EDGES = (120, 300, 600, 900, 1200, 1500, 1800, 2100)
GEO_MIN, GEO_MEAN = EXPECTED_B0_NAMES.index('geo_min'), EXPECTED_B0_NAMES.index('geo_mean')


def message_of(raised) -> str:
    """The text of a caught error. The toy stores hold sentinel match keys (``KEY_PREFIX``); none may get into a message."""
    assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)
    return str(raised.value)


def arrays_read(fn, *args, **kwargs):
    """``(list of (store folder, array name) that fn read through Store.array, fn's result)``."""
    seen = []
    real = Store.array

    def spy(self, name):
        seen.append((self.root.name, name))
        return real(self, name)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Store, 'array', spy)
        result = fn(*args, **kwargs)
    return seen, result


def names_in(seen) -> set:
    return {name for _, name in seen}


def fit_views(s1, s2) -> np.ndarray:
    return np.flatnonzero(features.split_views(s1, s2, 'FIT'))


def toy_drop_list(s2) -> tuple:
    """The drop list of the toy judged over all its 49 views, passed explicitly: the b1 tests need some list of dropped names and this
    one is the fixed 3-name ``TOY_DROPPED_ALL``. The runner never does this: it passes the FIT views (ruling R14)."""
    return features.dropped_columns(s2, TOY_ALL_VIEWS)


# ---- oracles, written from the text of the brief -----------------------------------------------------------------------------------------
def oracle_phase(game_time_s: float) -> int:
    """The phase bin of a time: the number of edges (120, 300, ..., 2100 s) that the time has reached, minus 1, not below 0."""
    reached = sum(1 for edge in PHASE_EDGES if round(game_time_s * 1000) >= edge * 1000)
    return max(reached - 1, 0)


def oracle_geo(G):
    """(minimum, mean) of the blue x red block of a (10, 10) geodesic matrix, the NaN pairs left out; (NaN, NaN) without a pair."""
    pairs = [float(G[i, j]) for i in range(5) for j in range(5, 10) if not np.isnan(G[i, j])]
    return (min(pairs), sum(pairs) / len(pairs)) if pairs else (np.nan, np.nan)


def oracle_b0(s2, v) -> np.ndarray:
    """The 39 columns of view v, NaN where a value is missing: one plain loop per column group, in the order of the brief's table."""
    S = np.asarray(s2.array('x/summary')[v], dtype=np.float64)
    P = np.asarray(s2.array('x/players_num')[v], dtype=np.float64)
    G = np.asarray(s2.array('x/geo_pp')[v], dtype=np.float64)
    s = lambda name: S[SUM_I[name]]                                                                    # noqa: E731
    out = [s('game_time_s') / 60.0]
    phase = oracle_phase(s('game_time_s'))
    out += [1.0 if phase == k else 0.0 for k in range(1, 8)]
    for name in ('totalGold', 'xp', 'level', 'minionsKilled', 'jungleMinionsKilled', 'alive', 'respawn_s', 'item_gold_owned'):
        c = PLR_I[name]
        out.append(sum(P[p, c] for p in range(5)) - sum(P[p, c] for p in range(5, 10)))
    out.append(sum(P[p, PLR_I['alive']] for p in range(10)))
    out += list(oracle_geo(G))
    for name in ('towers_outer', 'towers_inner', 'towers_base', 'towers_nexus', 'inhibs_down'):
        out.append(s('blue_' + name) - s('red_' + name))
    lanes = ('top', 'mid', 'bot')
    out.append(sum(s('blue_plates_' + lane) for lane in lanes) - sum(s('red_plates_' + lane) for lane in lanes))
    for name in ('dragons', 'elders', 'barons', 'heralds', 'atakhans', 'voidgrubs', 'has_soul'):
        out.append(s('blue_' + name) - s('red_' + name))
    for buff in ('baron', 'elder'):
        out.append(float(s(f'blue_{buff}_buff_s') > 0) - float(s(f'red_{buff}_buff_s') > 0))
    out += [s(f'{name}_up') for name in ('dragon', 'baron', 'herald', 'atakhan', 'voidgrub')]
    assert len(out) == 39
    return np.array(out)


def oracle_b0_all(s2, views) -> np.ndarray:
    return np.array([oracle_b0(s2, int(v)) for v in views]).reshape(len(views), 39)


def oracle_b1(s2, v, dropped) -> np.ndarray:
    """summary (minus dropped), then per kept player column the sum / the maximum over blue (players 0-4) and red (5-9) in the order
    sum blue, sum red, max blue, max red (NaN if a player of the team has NaN there), then the cross-team geodesic minimum and mean."""
    S = np.asarray(s2.array('x/summary')[v], dtype=np.float64)
    P = np.asarray(s2.array('x/players_num')[v], dtype=np.float64)
    G = np.asarray(s2.array('x/geo_pp')[v], dtype=np.float64)
    out = [S[i] for i, name in enumerate(tokens.SUMMARY) if name not in dropped]
    keep = [i for i, name in enumerate(tokens.PLAYER_NUM) if name not in dropped]

    def team(players, c, how):
        cells = [P[p, c] for p in players]
        if any(np.isnan(x) for x in cells):
            return np.nan
        return sum(cells) if how == 'sum' else max(cells)
    for how in ('sum', 'max'):
        for players in (range(5), range(5, 10)):
            out += [team(players, c, how) for c in keep]
    out += list(oracle_geo(G))
    return np.array(out)


def same(a, b, rtol=1e-6) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=1e-6, equal_nan=True)


# ---- the tests of the task brief that concern this module --------------------------------------------------------------------------------
def test_drops_constant_columns_and_t_s(toy_s1, toy_s2):            # 장난감 S2에 상수 열 하나 (R14: the FIT views are passed)
    d = features.dropped_columns(toy_s2, np.flatnonzero(features.split_views(toy_s1, toy_s2, 'FIT')))
    assert 't_s' in d and TOY_CONSTANT_COLUMN in d and 'game_time_s' not in d


def test_b0_has_no_ids_and_fixed_width(toy_s2):
    assert features.b0(toy_s2, TOY_VIEWS).shape[1] == len(features.B0_NAMES) == 39


def test_m4_masked_inputs_signature_is_fixed():
    assert list(inspect.signature(features.m4_masked_inputs).parameters) == ['row_view', 'q', 'm', 'z', 'K', 'O']


def test_weights_are_one_over_matches():
    w = features.match_weights(np.array([0, 0, 1, 2, 2, 2]))
    assert np.allclose(np.bincount([0, 0, 1, 2, 2, 2], weights=w), 1.0)


# ---- the toy itself: the hand numbers of s3_toys hold ------------------------------------------------------------------------------------
def test_the_toy_s2_has_the_layout_the_hand_numbers_assume(toy_s1, toy_s2):
    src = np.asarray(toy_s2.array('views/src'))
    assert len(src) == TOY_N_VIEWS and np.flatnonzero(src == 0).tolist() == list(TOY_FRAME_VIEWS)
    assert np.flatnonzero(src > 0).tolist() == list(TOY_EVENT_VIEWS) and TOY_VIEWS.tolist() == list(TOY_FRAME_VIEWS)
    role = np.asarray(toy_s1.array('matches/role'))[np.asarray(toy_s2.array('views/match'))]
    assert np.flatnonzero(role == C.ROLES.index('FIT')).tolist() == list(TOY_FIT_VIEWS) and len(TOY_FIT_VIEWS) == 30
    assert toy_s2.array('x/players_num').shape == (TOY_N_VIEWS, 10, 81) and toy_s2.array('x/geo_pp').shape == (TOY_N_VIEWS, 10, 10)
    assert toy_s2.array('x/summary').shape == (TOY_N_VIEWS, 107) and X_ARRAYS <= set(toy_s2.manifest['arrays'])


def test_the_toy_event_views_keep_junk_in_every_summary_cell(toy_s2):
    summary = np.asarray(toy_s2.array('x/summary'))
    assert (summary[list(TOY_EVENT_VIEWS)] == JUNK).all()                       # a frame-clock lookup that picked one would show it


# ---- splits and weights ------------------------------------------------------------------------------------------------------------------
def test_split_rows_marks_the_frame_rows_of_the_matches_of_one_role(toy_s1):
    fit, stop, dev = (features.split_rows(toy_s1, r) for r in ('FIT', 'STOP', 'DEV'))
    match = np.asarray(toy_s1.array('frame_rows/match'))
    assert fit.dtype == bool and len(fit) == len(match) == 37
    assert fit.sum() == TOY_N_FIT_FRAME_ROWS == 22 and set(match[fit].tolist()) == TOY_FIT_MATCHES - {4}      # match 4 has no frame row
    assert set(match[stop]) == TOY_STOP_MATCHES and stop.sum() == 7 and set(match[dev]) == TOY_DEV_MATCHES and dev.sum() == 8
    assert (fit | stop | dev).all() and not (fit & stop).any() and not (fit & dev).any() and not (stop & dev).any()


def test_split_rows_takes_several_roles(toy_s1):
    both = features.split_rows(toy_s1, ('FIT', 'STOP'))
    assert np.array_equal(both, features.split_rows(toy_s1, 'FIT') | features.split_rows(toy_s1, 'STOP')) and both.sum() == 29


@pytest.mark.parametrize('role, text', [('TEST', 'holdout'), ('EXTERNAL', 'holdout'), ('CAL', 'not used'), ('AUDIT', 'not used'),
                                        ('UNUSED', 'not used'), ('NOPE', 'unknown role'), ((), 'no role')])
def test_a_holdout_role_or_one_that_stage_s3_does_not_use_is_refused_before_anything_is_read(toy_s1, toy_s2, role, text):
    for call in (lambda: features.split_rows(toy_s1, role), lambda: features.role_mask(toy_s1, 'frame_rows', role),
                 lambda: features.split_views(toy_s1, toy_s2, role)):
        seen = []
        real = Store.array

        def spy(self, name):
            seen.append(name)
            return real(self, name)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Store, 'array', spy)
            with pytest.raises(ValueError, match=text) as raised:
                call()
        assert seen == [] and KEY_PREFIX not in message_of(raised)


def test_role_mask_of_the_unit_table(toy_value_s1):
    unit_match = np.asarray(toy_value_s1.array('units/match'))
    assert unit_match.tolist() == [0] * 6 + [1] * 3 + [3]                       # value toy: FIT (6 units), STOP (3), DEV (none), CAL (1)
    assert features.role_mask(toy_value_s1, 'units', 'FIT').tolist() == [True] * 6 + [False] * 4
    assert features.role_mask(toy_value_s1, 'units', 'STOP').tolist() == [False] * 6 + [True] * 3 + [False]
    assert not features.role_mask(toy_value_s1, 'units', 'DEV').any()
    assert features.role_mask(toy_value_s1, 'units', ('FIT', 'STOP', 'DEV')).tolist() == [True] * 9 + [False]


def test_role_mask_of_the_frame_row_table_is_split_rows(toy_value_s1):
    for role in ('FIT', 'STOP', 'DEV'):
        assert np.array_equal(features.role_mask(toy_value_s1, 'frame_rows', role), features.split_rows(toy_value_s1, role))
    fit = features.split_rows(toy_value_s1, 'FIT')
    assert fit.sum() == TOY_ROW_OFF[1] - TOY_ROW_OFF[0] == 10 and fit[:10].all()


def test_role_mask_of_the_onsets_goes_through_the_frame_row_of_each_onset(toy_value_s1):
    row = np.asarray(toy_value_s1.array('onsets/row')).astype(np.int64)
    row_match = np.asarray(toy_value_s1.array('frame_rows/match')).astype(np.int64)
    role = np.asarray(toy_value_s1.array('matches/role')).astype(np.int64)
    assert len(row) > 0
    for name in ('FIT', 'STOP', 'DEV'):
        want = [bool(role[row_match[r]] == C.ROLES.index(name)) for r in row]        # one onset at a time
        got = features.role_mask(toy_value_s1, 'onsets', name)
        assert got.dtype == bool and got.tolist() == want
    assert features.role_mask(toy_value_s1, 'onsets', 'FIT').any()


def test_role_mask_reads_only_the_documented_arrays(toy_value_s1):
    seen, _ = arrays_read(features.split_rows, toy_value_s1, 'FIT')
    assert names_in(seen) == {'matches/role', 'frame_rows/match'}
    seen, _ = arrays_read(features.role_mask, toy_value_s1, 'units', 'FIT')
    assert names_in(seen) == {'matches/role', 'units/match'}
    seen, _ = arrays_read(features.role_mask, toy_value_s1, 'onsets', 'FIT')
    assert names_in(seen) == {'matches/role', 'onsets/row', 'frame_rows/match'}


@pytest.mark.parametrize('table', ['../matches', 'a/b', '', 'Units', 'units ', 'frame-rows', 'matches/key'])
def test_a_table_name_that_is_not_a_plain_name_is_refused_before_any_array_is_read(toy_value_s1, table):
    seen = []
    real = Store.array

    def spy(self, name):
        seen.append(name)
        return real(self, name)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Store, 'array', spy)
        with pytest.raises(ValueError, match='match column'):
            features.role_mask(toy_value_s1, table, 'FIT')
    assert seen == []                                                              # no path is built from such a name


@pytest.mark.parametrize('table', ['kills', 'objectives', 'nope', 'matches'])
def test_role_mask_refuses_a_table_without_a_match_column(toy_value_s1, table):
    with pytest.raises(ValueError, match='match column') as raised:
        features.role_mask(toy_value_s1, table, 'FIT')
    assert KEY_PREFIX not in message_of(raised)


def test_role_mask_refuses_a_store_that_lacks_the_table(toy_s1):
    with pytest.raises(ValueError, match='match column'):
        features.role_mask(toy_s1, 'units', 'FIT')                              # the S1 toy has no engagement units


def small_s1(root, roles, onsets_row=None, **tables):
    """An S1 store of ``matches/role``, the given ``<table>/match`` arrays and, if asked, ``onsets/row`` only."""
    s1 = Store(root)
    arrays = {'matches/role': roles, **{f'{t}/match': v for t, v in tables.items()}}
    if onsets_row is not None:
        arrays['onsets/row'] = onsets_row
    for name, values in arrays.items():
        a = s1.allocate(name, (len(values),), 'i1' if name == 'matches/role' else 'i4')
        a[:] = values
        a.flush()
        del a
    s1.finalize()
    return s1


def test_role_mask_refuses_a_match_number_the_store_does_not_have(tmp_path):
    s1 = small_s1(tmp_path / 's1', [0, 2], frame_rows=[0, 1, 2], units=[-1, 0])
    for table in ('frame_rows', 'units'):
        with pytest.raises(ValueError, match='match number'):
            features.role_mask(s1, table, 'FIT')
    ok = small_s1(tmp_path / 's1b', [0, 2], frame_rows=[0, 1, 1, 0])
    assert features.role_mask(ok, 'frame_rows', 'FIT').tolist() == [True, False, False, True]


def test_role_mask_refuses_an_onset_whose_frame_row_the_store_does_not_have(tmp_path):
    for i, bad in enumerate(([0, 3], [-1, 0])):                                 # past the last frame row; before the first (it would wrap)
        s1 = small_s1(tmp_path / f's1_{i}', [0, 2], onsets_row=bad, frame_rows=[0, 1, 1])
        with pytest.raises(ValueError, match='frame row'):
            features.role_mask(s1, 'onsets', 'FIT')
    ok = small_s1(tmp_path / 'ok', [0, 2], onsets_row=[2, 0, 0, 1], frame_rows=[0, 1, 1])     # onsets of the frame rows 2, 0, 0, 1
    assert features.role_mask(ok, 'onsets', 'FIT').tolist() == [False, True, True, False]      # = matches 1, 0, 0, 1 = STOP, FIT, FIT, STOP


def test_split_views_marks_the_views_of_the_matches_of_one_role(toy_s1, toy_s2):
    assert np.flatnonzero(features.split_views(toy_s1, toy_s2, 'FIT')).tolist() == list(TOY_FIT_VIEWS)
    assert np.flatnonzero(features.split_views(toy_s1, toy_s2, 'STOP')).tolist() == list(range(22, 31))
    assert np.flatnonzero(features.split_views(toy_s1, toy_s2, 'DEV')).tolist() == list(range(31, 41))
    both = features.split_views(toy_s1, toy_s2, ('STOP', 'DEV'))
    assert both.sum() == 19 and both.dtype == bool and len(both) == TOY_N_VIEWS


def test_split_views_reads_the_match_of_each_view_and_the_roles_only(toy_s1, toy_s2):
    seen, _ = arrays_read(features.split_views, toy_s1, toy_s2, 'FIT')
    assert names_in(seen) == {'matches/role', 'views/match'}


def test_split_views_refuses_a_view_of_a_match_the_s1_store_does_not_have(toy_s2, tmp_path):
    s1 = small_s1(tmp_path / 's1', [0, 0])                                       # the toy S2 names the matches 0-5
    with pytest.raises(ValueError, match='match number'):
        features.split_views(s1, toy_s2, 'FIT')


def test_weights_of_unsorted_negative_and_sparse_match_numbers():
    match = np.array([7, 7, -3, 100, 100, 100, 7])
    w = features.match_weights(match)
    assert w.dtype == np.float64 and w.tolist() == pytest.approx([1 / 3, 1 / 3, 1.0, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    assert np.isclose(w[match == 7].sum(), 1.0) and np.isclose(w[match == 100].sum(), 1.0) and w[match == -3][0] == 1.0


def test_weights_depend_on_the_rows_passed_not_on_the_store():
    rows = np.array([5, 5, 5, 5, 9, 9])
    assert features.match_weights(rows).tolist() == pytest.approx([0.25] * 4 + [0.5] * 2)
    assert features.match_weights(rows[:3]).tolist() == pytest.approx([1 / 3] * 3)         # fewer rows passed: more weight per row


def test_weights_of_nothing_and_of_the_wrong_kind():
    assert features.match_weights(np.array([], dtype=np.int64)).shape == (0,) and features.match_weights([]).dtype == np.float64
    with pytest.raises(TypeError, match='integer'):
        features.match_weights(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match='1-d'):
        features.match_weights(np.zeros((2, 2), dtype=np.int64))


def test_weights_of_the_toy_frame_rows_add_up_to_the_number_of_matches(toy_s1):
    rows = np.flatnonzero(features.split_rows(toy_s1, ('FIT', 'STOP', 'DEV')))
    match = np.asarray(toy_s1.array('frame_rows/match'))[rows]
    w = features.match_weights(match)
    assert w.sum() == pytest.approx(5.0) and len(set(match.tolist())) == 5                  # match 4 has no frame row, the other 5 have


# ---- the columns to drop -----------------------------------------------------------------------------------------------------------------
def test_dropped_columns_of_the_toy_are_exactly_the_constant_ones_and_t_s(toy_s2):
    d = features.dropped_columns(toy_s2, TOY_ALL_VIEWS)                         # all 49 views, named by the caller
    assert isinstance(d, tuple) and all(type(name) is str for name in d)
    assert set(d) == set(TOY_DROPPED_ALL) and len(d) == len(set(d)) == 3
    assert TOY_CONSTANT_PLAYER_COLUMN in d and 'game_time_s' not in d and TOY_FIT_CONSTANT_COLUMN not in d


def test_t_s_is_dropped_by_name_also_when_it_varies(toy_s1, toy_s2):
    summary = np.asarray(toy_s2.array('x/summary'))
    fit = fit_views(toy_s1, toy_s2)
    assert len(np.unique(summary[fit, SUM_I['t_s']])) > 1                       # in the FIT views of the toy t_s is not constant ...
    assert 't_s' in features.dropped_columns(toy_s2, fit)                        # ... and it goes anyway: it is game_time_s twice


def test_dropped_columns_judge_the_views_they_are_given(toy_s1, toy_s2):
    fit = fit_views(toy_s1, toy_s2)
    assert fit.tolist() == list(TOY_FIT_VIEWS)
    d_fit = features.dropped_columns(toy_s2, fit)
    assert set(d_fit) == set(TOY_DROPPED_FIT) and TOY_FIT_CONSTANT_COLUMN in d_fit            # constant over the FIT views only
    assert set(features.dropped_columns(toy_s2, np.array(TOY_FIT_VIEWS))) == set(d_fit)
    assert set(features.dropped_columns(toy_s2, features.split_views(toy_s1, toy_s2, 'FIT'))) == set(d_fit)       # a mask works too
    assert TOY_FIT_CONSTANT_COLUMN not in features.dropped_columns(toy_s2, TOY_ALL_VIEWS)       # another set of views, another list


def test_the_views_of_dropped_columns_are_required_and_the_old_default_is_gone(toy_s2):       # ruling R14
    parameters = inspect.signature(features.dropped_columns).parameters
    assert list(parameters) == ['s2', 'fit_views'] and parameters['fit_views'].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match='fit_views'):
        features.dropped_columns(toy_s2)                                        # the default judged every role together
    seen = []
    real = Store.array

    def spy(self, name):
        seen.append(name)
        return real(self, name)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Store, 'array', spy)
        with pytest.raises(ValueError, match='FIT views') as raised:
            features.dropped_columns(toy_s2, None)                              # None is refused before the store is touched ...
    assert 'split_views' in message_of(raised) and seen == []                   # ... and the message says what to pass
    for nothing in ([], np.array([], dtype=np.int64), np.zeros(TOY_N_VIEWS, dtype=bool)):      # no view at all, also as a mask
        with pytest.raises(ValueError, match='FIT views') as raised:
            features.dropped_columns(toy_s2, nothing)
        assert 'split_views' in message_of(raised), nothing


def test_dropped_columns_come_in_the_order_of_the_summary_and_then_the_player_columns(toy_s2):
    d = features.dropped_columns(toy_s2, np.array(TOY_FIT_VIEWS))
    summary_part = [n for n in d if n in SUM_I]
    player_part = [n for n in d if n in PLR_I]
    assert list(d) == summary_part + player_part
    assert summary_part == sorted(summary_part, key=SUM_I.get) and player_part == sorted(player_part, key=PLR_I.get)
    assert summary_part == ['t_s', TOY_CONSTANT_COLUMN, TOY_FIT_CONSTANT_COLUMN] and player_part == [TOY_CONSTANT_PLAYER_COLUMN]


def test_summary_and_player_names_are_disjoint():
    assert set(tokens.SUMMARY).isdisjoint(tokens.PLAYER_NUM)


def test_dropped_columns_are_plain_names_that_b1_understands(toy_s2):
    d = toy_drop_list(toy_s2)
    assert set(d) <= set(tokens.SUMMARY) | set(tokens.PLAYER_NUM)
    assert len(features.b1_columns(d)) == features.b1(toy_s2, TOY_VIEWS, d).shape[1]


def test_variance_zero_means_min_equals_max_ignoring_nan(tmp_path):
    summary, players, geo = blank_feature_arrays(6)
    summary[:, SUM_I['game_time_s']] = np.arange(6)                              # varies: kept
    players[:, :, PLR_I['x']] = np.arange(60).reshape(6, 10)                      # varies: kept
    players[:, :, PLR_I['prev_x']] = 100.0
    players[0, 3, PLR_I['prev_x']] = np.nan                                       # NaN and one value: zero variance among the values
    players[:, :, PLR_I['prev_y']] = np.nan                                       # no value at all: nothing to learn from
    players[:, :, PLR_I['xp']] = 7.0
    players[5, 9, PLR_I['xp']] = 7.5                                              # one cell of one view differs: kept
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    d = features.dropped_columns(store, np.arange(6))
    assert 'prev_x' in d and 'prev_y' in d and 'xp' not in d and 'x' not in d and 'game_time_s' not in d
    assert 'rift_element_known' in d and 'flag_stasis' in d and 'level' in d       # all-zero columns are constant


def test_a_column_that_changes_between_chunks_is_not_constant(tmp_path, monkeypatch):
    monkeypatch.setattr(features, '_CHUNK', 5)
    summary, players, geo = blank_feature_arrays(10)
    summary[:, SUM_I['frame_age_s']] = np.repeat([3.0, 4.0], 5)                  # constant inside each chunk, another value in the other
    summary[:, SUM_I['old_ward_kill_b']] = 2.0                                   # constant everywhere
    summary[9, SUM_I['old_ward_kill_r']] = 1.0                                   # only the last view of the last chunk differs
    summary[0, SUM_I['old_ward_placed_r']] = 1.0                                 # only the first view of the first chunk differs
    players[:, :, PLR_I['level']] = 5.0
    players[9, 9, PLR_I['level']] = 6.0                                          # a player cell of the last chunk
    players[:, :, PLR_I['xp']] = 7.0
    players[0, 3, PLR_I['xp']] = 8.0                                             # a player cell of the first chunk
    players[:, :, PLR_I['lvl_armor']] = np.repeat([1.0, 2.0], 5)[:, None]        # constant inside each chunk, another value in the other
    players[:, :, PLR_I['respawn_s']] = 4.0                                      # constant everywhere
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    d = features.dropped_columns(store, np.arange(10))
    kept = {'frame_age_s', 'old_ward_kill_r', 'old_ward_placed_r', 'level', 'xp', 'lvl_armor'}
    assert not kept & set(d), kept & set(d)
    assert 'old_ward_kill_b' in d and 'respawn_s' in d                          # constant in every chunk and between chunks


def test_dropped_columns_do_not_depend_on_the_chunk_size(toy_s2, monkeypatch):
    want = features.dropped_columns(toy_s2, np.array(TOY_FIT_VIEWS))
    for chunk in (1, 3, 7, 1000):
        monkeypatch.setattr(features, '_CHUNK', chunk)
        assert features.dropped_columns(toy_s2, np.array(TOY_FIT_VIEWS)) == want


def test_dropped_columns_read_the_summary_and_the_players_only(toy_s2):
    seen, _ = arrays_read(features.dropped_columns, toy_s2, np.array(TOY_FIT_VIEWS))
    assert names_in(seen) == {'x/summary', 'x/players_num'}


def test_dropped_columns_refuse_no_views_and_bad_view_numbers(toy_s2):
    for bad, error in ((np.array([], dtype=np.int64), ValueError), (np.zeros(TOY_N_VIEWS - 1, dtype=bool), ValueError), (None, ValueError),
                       ([], ValueError), (np.zeros(TOY_N_VIEWS, dtype=bool), ValueError),
                       (np.array([0, TOY_N_VIEWS]), ValueError), (np.array([-1, 0]), ValueError), (np.array([0.5, 1.5]), TypeError),
                       (np.zeros((2, 2), dtype=np.int64), ValueError)):
        with pytest.raises(error):
            features.dropped_columns(toy_s2, bad)


def test_dropped_columns_refuse_a_store_that_is_not_an_s2_store(tmp_path):
    summary, players, geo = blank_feature_arrays(3)
    with pytest.raises(ValueError, match='107'):
        features.dropped_columns(build_feature_store(tmp_path / 'a', summary[:, :100], players, geo), [0])     # 100 summary columns
    with pytest.raises(ValueError, match='10, 81'):
        features.dropped_columns(build_feature_store(tmp_path / 'b', summary, players[:, :, :80], geo), [0])
    with pytest.raises(ValueError, match='views'):
        features.dropped_columns(build_feature_store(tmp_path / 'c', summary, players[:2], geo), [0])


# ---- b0: names and width -----------------------------------------------------------------------------------------------------------------
def test_b0_names_are_the_39_columns_of_the_brief_in_order():
    assert features.B0_NAMES == EXPECTED_B0_NAMES and len(set(features.B0_NAMES)) == 39
    assert all(type(n) is str for n in features.B0_NAMES)


def test_b0_has_no_champion_item_or_patch_columns():
    trait_or_id = ('tag_', 'lvl_', 'item_', 'is_ranged', 'partype')
    assert not any(('champ' in n or 'patch' in n or n.startswith(trait_or_id)) for n in features.B0_NAMES)


def test_b0_signature_takes_two_arguments_and_the_fill_options():
    parameters = inspect.signature(features.b0).parameters
    assert list(parameters) == ['s2', 'views', 'fit_means', 'fill_counts']
    assert all(p.default is None for p in list(parameters.values())[2:])


def test_b0_is_float32_with_one_row_per_view_in_the_order_given(toy_s2):
    x = features.b0(toy_s2, TOY_VIEWS)
    assert x.dtype == np.float32 and x.shape == (len(TOY_VIEWS), 39) and x.flags['C_CONTIGUOUS'] and np.isfinite(x).all()
    order = np.array([5, 0, 5, 36, 2])
    assert np.array_equal(features.b0(toy_s2, TOY_VIEWS[order]), x[order])        # duplicates and any order are fine


def test_b0_of_no_view_has_no_row(toy_s2):
    assert features.b0(toy_s2, np.array([], dtype=np.int64)).shape == (0, 39) and features.b0(toy_s2, []).dtype == np.float32


def test_b0_takes_a_mask_over_the_views_too(toy_s2):
    mask = np.zeros(TOY_N_VIEWS, dtype=bool)
    mask[[1, 4, 8]] = True
    assert np.array_equal(features.b0(toy_s2, mask), features.b0(toy_s2, np.array([1, 4, 8])))


# ---- b0: values --------------------------------------------------------------------------------------------------------------------------
def hand_store(tmp_path):
    """One view whose 39 columns are worked out by hand (the working is above ``HAND_B0``)."""
    summary, players, geo = blank_feature_arrays(1)
    summary[0, SUM_I['game_time_s']] = 750.0
    for name, blue, red in (('towers_outer', 3, 1), ('towers_inner', 2, 2), ('towers_base', 1, 0), ('towers_nexus', 0, 0),
                            ('inhibs_down', 0, 1), ('plates_top', 1, 0), ('plates_mid', 2, 1), ('plates_bot', 0, 1), ('dragons', 2, 1),
                            ('elders', 0, 0), ('barons', 1, 0), ('heralds', 0, 1), ('atakhans', 0, 0), ('voidgrubs', 3, 0),
                            ('has_soul', 1, 0)):
        summary[0, SUM_I['blue_' + name]], summary[0, SUM_I['red_' + name]] = blue, red
    summary[0, SUM_I['blue_baron_buff_s']], summary[0, SUM_I['red_baron_buff_s']] = 120.5, 0.0
    summary[0, SUM_I['blue_elder_buff_s']], summary[0, SUM_I['red_elder_buff_s']] = 0.0, 30.0
    for name, flag in (('dragon_up', 1), ('baron_up', 0), ('herald_up', 1), ('atakhan_up', 0), ('voidgrub_up', 1)):
        summary[0, SUM_I[name]] = flag
    for name, blue, red in (('totalGold', (1000, 2000, 3000, 4000, 5000), (1500,) * 5), ('xp', (100,) * 5, (80,) * 5),
                            ('level', (5, 6, 7, 8, 9), (6,) * 5), ('minionsKilled', (10,) * 5, (12,) * 5),
                            ('jungleMinionsKilled', (1, 2, 3, 4, 5), (0,) * 5), ('alive', (1,) * 5, (1, 1, 0, 0, 1)),
                            ('respawn_s', (0,) * 5, (0, 0, 20, 30, 0)), ('item_gold_owned', (500,) * 5, (300,) * 5)):
        players[0, :5, PLR_I[name]], players[0, 5:, PLR_I[name]] = blue, red
    geo[0, :5, 5:] = 100.0
    geo[0, 0, 5], geo[0, 4, 9] = 40.0, 160.0
    return build_feature_store(tmp_path / 's2', summary, players, geo)


# time: 750 s = 12.5 min, in [600, 900) s = phase 2; players (blue - red): gold 15,000 - 7,500, xp 500 - 400, level 35 - 30, minions
# 50 - 60, jungle 15 - 0, alive 5 - 3, respawn 0 - 50, item gold 2,500 - 1,500; alive in all 5 + 3; geodesics: 23 pairs of 100, one of
# 40 and one of 160, minimum 40 and mean 2,500 / 25 = 100; summary: towers 3-1, 2-2, 1-0, 0-0; inhibitors down 0-1; plates (1 + 2 + 0) -
# (0 + 1 + 1); dragons 2-1, elders 0, barons 1-0, heralds 0-1, atakhans 0, grubs 3-0, soul 1-0; baron buff on for blue only, elder buff
# on for red only; flags 1, 0, 1, 0, 1
HAND_B0 = [12.5,
           0, 1, 0, 0, 0, 0, 0,
           7500, 100, 5, -10, 15, 2, -50, 1000,
           8,
           40, 100,
           2, 0, 1, 0, -1, 1, 1, 0, 1, -1, 0, 3, 1, 1, -1,
           1, 0, 1, 0, 1]


def test_b0_hand_example_column_by_column(tmp_path):
    x = features.b0(hand_store(tmp_path), [0])
    assert len(HAND_B0) == 39
    for name, got, want in zip(features.B0_NAMES, x[0].tolist(), HAND_B0):
        assert got == want, name


def test_b0_equals_the_oracle_on_every_view_of_the_toy(toy_s2):
    raw = oracle_b0_all(toy_s2, TOY_ALL_VIEWS)
    nan_views = np.isnan(raw).any(axis=1)
    assert np.flatnonzero(nan_views).tolist() == list(TOY_NAN_VIEWS)
    assert same(features.b0(toy_s2, TOY_VIEWS), raw[TOY_VIEWS])
    events = [v for v in TOY_EVENT_VIEWS if v not in TOY_NAN_VIEWS]                  # event views are views too (those with a distance)
    assert len(events) == 8 and same(features.b0(toy_s2, np.array(events)), raw[events])


def test_b0_geodesic_columns_are_the_hand_numbers_of_the_toy(toy_s2):
    views = np.array(sorted(TOY_GEO_HAND))
    x = features.b0(toy_s2, views)
    for row, view in enumerate(views):
        assert (x[row, GEO_MIN], x[row, GEO_MEAN]) == TOY_GEO_HAND[int(view)], view


def test_b0_geodesic_ignores_the_pairs_without_a_position_and_is_nan_without_any(toy_s2):
    geo = np.asarray(toy_s2.array('x/geo_pp')).astype(np.float64)
    for view in TOY_MISSING:
        assert np.isnan(geo[view, :5, 5:]).all() == (view in TOY_NAN_VIEWS)       # the toy: a whole team missing leaves no pair
    raw = oracle_b0_all(toy_s2, TOY_ALL_VIEWS)
    assert np.isnan(raw[list(TOY_NAN_VIEWS)][:, [GEO_MIN, GEO_MEAN]]).all()
    assert not np.isnan(np.delete(raw[list(TOY_NAN_VIEWS)], [GEO_MIN, GEO_MEAN], axis=1)).any()


def test_b0_phases_are_one_hot_with_phase_0_as_the_reference(toy_s2):
    x = features.b0(toy_s2, TOY_VIEWS)
    phases = x[:, 1:8]
    assert set(np.unique(phases)) <= {0.0, 1.0} and (phases.sum(axis=1) <= 1).all()
    t = np.asarray(toy_s2.array('x/summary'))[TOY_VIEWS, SUM_I['game_time_s']]
    want = np.array([oracle_phase(float(s)) for s in t])
    for k in range(1, 8):
        assert np.array_equal(phases[:, k - 1], (want == k).astype(np.float32))
    assert (phases.sum(axis=1) == (want > 0)).all()                              # all seven zero = phase 0
    assert set(want.tolist()) == {0, 1, 2}                                       # the toy times, 120.3 .. 660.3 s, reach phases 0, 1 and 2


@pytest.mark.parametrize('seconds, phase', [
    (0.0, 0), (90.0, 0), (119.999, 0), (120.0, 0), (299.999, 0), (300.0, 1), (300.001, 1), (599.999, 1), (600.0, 2), (899.999, 2),
    (900.0, 3),
    (1199.999, 3), (1200.0, 4), (1500.0, 5), (1799.999, 5), (1800.0, 6), (2099.999, 6), (2100.0, 7), (2400.0, 7), (7777.0, 7)])
def test_b0_phase_edges(tmp_path, seconds, phase):
    summary, players, geo = blank_feature_arrays(1)
    summary[0, SUM_I['game_time_s']] = seconds
    x = features.b0(build_feature_store(tmp_path / 's2', summary, players, geo), [0])[0]
    assert x[0] == pytest.approx(np.float32(seconds) / 60, rel=1e-6)
    assert x[1:8].tolist() == [1.0 if k == phase else 0.0 for k in range(1, 8)]
    assert phase == oracle_phase(float(np.float32(seconds))) == int(flowprice.phase_of(round(seconds * 1000)))      # as flowprice


def test_b0_agrees_with_the_window_controls_of_flowprice(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    src = np.asarray(toy_s2.array('views/src'))
    src_index = np.asarray(toy_s2.array('views/src_index'))
    frame_views = np.flatnonzero(src == 0)
    view_of = np.full(len(toy_s1.array('frame_rows/k')), -1)
    view_of[src_index[frame_views]] = frame_views
    views = view_of[w.row]
    assert (views >= 0).all() and len(views) == 20
    x = features.b0(toy_s2, views).astype(np.float64)
    first, last = features.B0_NAMES.index('diff_towers_outer'), features.B0_NAMES.index('diff_elder_buff')
    assert np.array_equal(x[:, 1:8], w.ctrl[:, 9:16])                              # phase indicators 1 .. 7 (ctrl 8 is phase 0)
    assert np.array_equal(1.0 - x[:, 1:8].sum(axis=1), w.ctrl[:, 8])
    assert np.array_equal(x[:, first:last + 1], w.ctrl[:, 18:33]) and last - first + 1 == 15       # the 13 differences and the 2 buffs


def test_b0_reads_only_the_cells_of_its_table(tmp_path, toy_s2):
    summary = np.array(toy_s2.array('x/summary'))
    players = np.array(toy_s2.array('x/players_num'))
    geo = np.array(toy_s2.array('x/geo_pp'))
    rng = np.random.default_rng(0)
    for name, i in SUM_I.items():
        if name not in USED_SUMMARY:
            summary[:, i] = rng.random(len(summary)) * 1000
    for name, i in PLR_I.items():
        if name not in USED_PLAYER:
            players[:, :, i] = rng.random(players.shape[:2]) * 1000          # champion, item, trait columns, positions, the rest
    geo[:, :5, :5] = 5.0                                                      # the blue-blue and red-red blocks are not read either
    geo[:, 5:, 5:] = 6.0
    other = build_feature_store(tmp_path / 's2', summary, players, geo)
    means = np.zeros(39)                                                      # the NaN views need some mean
    assert np.array_equal(features.b0(toy_s2, TOY_ALL_VIEWS, means), features.b0(other, TOY_ALL_VIEWS, means))


def test_b0_reads_the_three_x_arrays_and_nothing_else(toy_s2):
    seen, _ = arrays_read(features.b0, toy_s2, TOY_VIEWS)
    assert names_in(seen) == X_ARRAYS


def test_b0_does_not_depend_on_the_chunk_size(toy_s2, monkeypatch):
    want = features.b0(toy_s2, TOY_VIEWS)
    for chunk in (1, 4, 1000):
        monkeypatch.setattr(features, '_CHUNK', chunk)
        assert np.array_equal(features.b0(toy_s2, TOY_VIEWS), want)


def test_b0_of_a_view_does_not_depend_on_the_other_views(toy_s2):
    all_ = features.b0(toy_s2, TOY_VIEWS)
    for i in (0, 7, 36):
        assert np.array_equal(features.b0(toy_s2, TOY_VIEWS[i:i + 1])[0], all_[i])


def test_b0_buff_indicator_is_active_for_a_positive_number_of_seconds(tmp_path):
    summary, players, geo = blank_feature_arrays(4)
    for v, (bb, rb, be, re_) in enumerate(((10.0, 20.0, 0.0, 0.0), (0.5, 0.0, 0.0, 1e-3), (0.0, 0.0, 5.0, 5.0), (0.0, 7.0, 3.0, 0.0))):
        summary[v, SUM_I['blue_baron_buff_s']], summary[v, SUM_I['red_baron_buff_s']] = bb, rb
        summary[v, SUM_I['blue_elder_buff_s']], summary[v, SUM_I['red_elder_buff_s']] = be, re_
    x = features.b0(build_feature_store(tmp_path / 's2', summary, players, geo), np.arange(4))
    assert x[:, features.B0_NAMES.index('diff_baron_buff')].tolist() == [0.0, 1.0, 0.0, -1.0]      # both on; blue only; neither; red only
    assert x[:, features.B0_NAMES.index('diff_elder_buff')].tolist() == [0.0, -1.0, 0.0, 1.0]


def test_b0_swapping_the_teams_flips_the_differences_and_nothing_else(tmp_path):
    summary, players, geo = blank_feature_arrays(2)
    rng = np.random.default_rng(1)
    for name in USED_SUMMARY - {'game_time_s'}:
        summary[0, SUM_I[name]] = rng.integers(0, 6)
    for name in USED_PLAYER:
        players[0, :, PLR_I[name]] = rng.integers(0, 9, 10)
    summary[0, SUM_I['game_time_s']] = 700.0
    swap = [SUM_I[n.replace('blue_', 'tmp_').replace('red_', 'blue_').replace('tmp_', 'red_')] for n in tokens.SUMMARY]
    summary[1] = summary[0][swap]                                                  # blue_x <- red_x and red_x <- blue_x
    players[1] = np.concatenate([players[0, 5:], players[0, :5]])
    x = features.b0(build_feature_store(tmp_path / 's2', summary, players, geo), [0, 1])
    names = list(features.B0_NAMES)
    flipped = [i for i, n in enumerate(names) if n.startswith('diff_')]
    assert len(flipped) == 8 + 15 and (x[0, flipped] != 0).any()
    assert np.array_equal(x[1, flipped], -x[0, flipped])
    kept = [i for i, n in enumerate(names) if not n.startswith('diff_')]
    assert np.array_equal(x[1, kept], x[0, kept])                                  # time, phases, the number alive, geodesics, flags


# ---- b0: missing values are filled with the FIT mean (ruling R2) -------------------------------------------------------------------------
def test_b0_with_a_nan_needs_the_fit_means(toy_s2):
    with pytest.raises(ValueError, match='FIT mean') as raised:
        features.b0(toy_s2, TOY_ALL_VIEWS)                                       # views 10, 29, 41 and 42 have no cross-team geodesic pair
    assert 'b0_fit_means' in message_of(raised)


def test_fit_means_are_the_nan_ignoring_means_of_the_fit_views(toy_s1, toy_s2):
    fit = fit_views(toy_s1, toy_s2)
    means = features.b0_fit_means(toy_s2, fit)
    raw = oracle_b0_all(toy_s2, fit)
    assert means.shape == (39,) and means.dtype == np.float64 and np.isfinite(means).all()
    assert np.allclose(means, np.nanmean(raw, axis=0), rtol=1e-12, atol=0)
    assert np.flatnonzero(np.isnan(raw).any(axis=0)).tolist() == [GEO_MIN, GEO_MEAN]            # the two geodesic columns have NaN in FIT
    assert np.isnan(raw[:, GEO_MIN]).sum() == len(TOY_NAN_FIT_VIEWS) == 3


def test_the_fit_mean_is_the_mean_of_the_views_asked_for_not_of_all_of_them(toy_s1, toy_s2):
    fit_mean = features.b0_fit_means(toy_s2, fit_views(toy_s1, toy_s2))
    other_mean = features.b0_fit_means(toy_s2, np.array(TOY_FRAME_VIEWS))
    assert not np.allclose(fit_mean, other_mean, rtol=1e-9)


def test_nan_cells_take_the_fit_mean_and_nothing_else_changes(toy_s1, toy_s2):
    means = features.b0_fit_means(toy_s2, fit_views(toy_s1, toy_s2))
    x = features.b0(toy_s2, TOY_ALL_VIEWS, means)
    raw = oracle_b0_all(toy_s2, TOY_ALL_VIEWS)
    want = np.where(np.isnan(raw), means[None, :], raw)
    assert x.shape == (TOY_N_VIEWS, 39) and x.dtype == np.float32 and np.isfinite(x).all()
    assert same(x, want)
    for view in TOY_NAN_VIEWS:
        assert x[view, GEO_MIN] == np.float32(means[GEO_MIN]) and x[view, GEO_MEAN] == np.float32(means[GEO_MEAN])
    assert np.array_equal(x[TOY_VIEWS], features.b0(toy_s2, TOY_VIEWS))          # a view without a NaN is the same with and without means


def test_fill_counts_are_the_filled_cells_per_column_and_add_up_over_calls(toy_s1, toy_s2):
    means = features.b0_fit_means(toy_s2, fit_views(toy_s1, toy_s2))
    counts = np.zeros(39, dtype=np.int64)
    features.b0(toy_s2, TOY_ALL_VIEWS, means, fill_counts=counts)
    want = np.zeros(39, dtype=np.int64)
    want[[GEO_MIN, GEO_MEAN]] = len(TOY_NAN_VIEWS)
    assert np.array_equal(counts, want)
    features.b0(toy_s2, np.array(TOY_FIT_VIEWS), means, fill_counts=counts)
    want[[GEO_MIN, GEO_MEAN]] += len(TOY_NAN_FIT_VIEWS)
    assert np.array_equal(counts, want)
    features.b0(toy_s2, TOY_VIEWS, means, fill_counts=counts)                      # no NaN there: nothing is added
    assert np.array_equal(counts, want)


def test_there_are_no_missing_value_indicator_columns(toy_s1, toy_s2):
    means = features.b0_fit_means(toy_s2, fit_views(toy_s1, toy_s2))
    assert features.b0(toy_s2, TOY_ALL_VIEWS, means).shape[1] == 39 and not any('miss' in n or 'nan' in n for n in features.B0_NAMES)


def test_fit_means_do_not_depend_on_the_chunk_size(toy_s1, toy_s2, monkeypatch):
    fit = fit_views(toy_s1, toy_s2)
    want = features.b0_fit_means(toy_s2, fit)
    for chunk in (1, 5, 1000):
        monkeypatch.setattr(features, '_CHUNK', chunk)
        assert np.allclose(features.b0_fit_means(toy_s2, fit), want, rtol=1e-12, atol=0)


def test_fit_means_read_the_three_x_arrays_only(toy_s2):
    seen, _ = arrays_read(features.b0_fit_means, toy_s2, np.array(TOY_FIT_VIEWS))
    assert names_in(seen) == X_ARRAYS


def test_fit_means_of_a_column_with_no_value_are_refused(toy_s2):
    with pytest.raises(ValueError, match='geo_min') as raised:
        features.b0_fit_means(toy_s2, np.array(TOY_NAN_FIT_VIEWS))                # in these views no geodesic pair has a value
    assert 'geo_mean' in message_of(raised)
    with pytest.raises(ValueError, match='no views'):
        features.b0_fit_means(toy_s2, np.array([], dtype=np.int64))


@pytest.mark.parametrize('means', [np.zeros(38), np.zeros((39, 1)), np.full(39, np.nan), np.full(39, np.inf)])
def test_bad_fit_means_are_refused(toy_s2, means):
    with pytest.raises(ValueError, match='fit_means'):
        features.b0(toy_s2, TOY_VIEWS, means)


@pytest.mark.parametrize('counts', [np.zeros(38, dtype=np.int64), np.zeros(39), [0] * 39, np.zeros((39, 1), dtype=np.int64)])
def test_bad_fill_counts_are_refused(toy_s2, counts):
    with pytest.raises((TypeError, ValueError), match='fill_counts'):
        features.b0(toy_s2, TOY_VIEWS, np.zeros(39), fill_counts=counts)


def test_b0_refuses_a_time_that_is_not_finite_and_a_value_that_is_infinite(tmp_path):
    summary, players, geo = blank_feature_arrays(2)
    summary[:, SUM_I['game_time_s']] = [100.0, np.nan]
    with pytest.raises(ValueError, match='game_time_s'):
        features.b0(build_feature_store(tmp_path / 'a', summary, players, geo), [0, 1], np.zeros(39))
    summary[:, SUM_I['game_time_s']] = 100.0
    players[1, 0, PLR_I['totalGold']] = np.inf
    with pytest.raises(ValueError, match='infinite') as raised:
        features.b0(build_feature_store(tmp_path / 'b', summary, players, geo), [0, 1], np.zeros(39))
    assert 'diff_totalGold' in message_of(raised)


def test_a_nan_in_a_player_column_of_b0_is_filled_like_any_other(tmp_path):
    summary, players, geo = blank_feature_arrays(3)
    summary[:, SUM_I['game_time_s']] = 200.0
    geo[:, :5, 5:] = 100.0
    players[:, :5, PLR_I['xp']] = [[10] * 5, [20] * 5, [30] * 5]
    players[1, 2, PLR_I['xp']] = np.nan                                           # a value that the table never leaves out in real data
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    xp = features.B0_NAMES.index('diff_xp')
    raw = oracle_b0_all(store, [0, 1, 2])
    assert np.isnan(raw[1, xp]) and not np.isnan(raw[[0, 2], xp]).any()
    means = features.b0_fit_means(store, [0, 1, 2])
    counts = np.zeros(39, dtype=np.int64)
    x = features.b0(store, [0, 1, 2], means, fill_counts=counts)
    assert means[xp] == pytest.approx((50 + 150) / 2) and x[1, xp] == np.float32(means[xp]) and counts[xp] == 1 and counts.sum() == 1


def test_a_nan_in_a_summary_cell_makes_the_difference_and_the_buff_indicator_nan_and_they_are_filled(tmp_path):
    summary, players, geo = blank_feature_arrays(3)
    summary[:, SUM_I['game_time_s']] = 200.0
    geo[:, :5, 5:] = 100.0
    summary[:, SUM_I['blue_towers_outer']] = [4.0, np.nan, 2.0]                   # blue minus red with a NaN part is NaN
    summary[:, SUM_I['red_towers_outer']] = 1.0
    summary[:, SUM_I['blue_baron_buff_s']] = [5.0, 5.0, np.nan]                   # a NaN number of seconds is not "buff off": NaN
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    towers, baron = features.B0_NAMES.index('diff_towers_outer'), features.B0_NAMES.index('diff_baron_buff')
    means = features.b0_fit_means(store, [0, 1, 2])
    assert means[towers] == pytest.approx((3 + 1) / 2) and means[baron] == 1.0         # (4 - 1, NaN, 2 - 1) and (1 - 0, 1 - 0, NaN)
    counts = np.zeros(39, dtype=np.int64)
    x = features.b0(store, [0, 1, 2], means, fill_counts=counts)
    assert x[1, towers] == 2.0 and x[2, baron] == 1.0 and x[0, towers] == 3.0 and x[1, baron] == 1.0
    assert counts[towers] == 1 and counts[baron] == 1 and counts.sum() == 2
    with pytest.raises(ValueError, match='FIT mean'):
        features.b0(store, [0, 1, 2])


# ---- the views argument ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('views, error, text', [
    (np.array([0, TOY_N_VIEWS]), ValueError, 'view numbers'), (np.array([-1]), ValueError, 'view numbers'),
    (np.array([0.5]), TypeError, 'integer'), (np.zeros(3, dtype=bool), ValueError, 'mask'),
    (np.zeros((2, 2), dtype=np.int64), ValueError, '1-d'),
    (['a'], TypeError, 'integer')])
def test_bad_view_numbers_are_refused(toy_s2, views, error, text):
    for call in (lambda: features.b0(toy_s2, views, np.zeros(39)), lambda: features.b1(toy_s2, views, ()),
                 lambda: features.dropped_columns(toy_s2, views), lambda: features.b0_fit_means(toy_s2, views)):
        with pytest.raises(error, match=text):
            call()


def test_a_store_without_the_arrays_or_with_other_shapes_is_refused(tmp_path):
    summary, players, geo = blank_feature_arrays(2)
    for i, (kwargs, text) in enumerate((
            (dict(summary=summary[:, :106], players_num=players, geo_pp=geo), '107'),
            (dict(summary=summary, players_num=players[:, :9], geo_pp=geo), '10, 81'),
            (dict(summary=summary, players_num=players, geo_pp=geo[:, :9]), '10, 10'),
            (dict(summary=summary, players_num=players[:1], geo_pp=geo), 'views'))):
        store = build_feature_store(tmp_path / f's{i}', **kwargs)
        with pytest.raises(ValueError, match=text):
            features.b0(store, [0], np.zeros(39))
    empty = Store(tmp_path / 'empty')
    empty.finalize()
    with pytest.raises(ValueError, match='x/summary'):
        features.b0(empty, [0], np.zeros(39))


# ---- b1 ----------------------------------------------------------------------------------------------------------------------------------
def test_b1_columns_are_the_kept_summary_the_team_aggregates_and_the_geodesics():
    n_summary, n_players = 107, 81
    names = features.b1_columns(())
    assert len(names) == len(set(names)) == n_summary + 4 * n_players + 2 == 433
    assert names[:n_summary] == tokens.SUMMARY and names[-2:] == ('geo_min', 'geo_mean')
    assert names[n_summary:n_summary + 3] == ('sum_blue_x', 'sum_blue_y', 'sum_blue_level')
    assert names[n_summary + n_players] == 'sum_red_x' and names[n_summary + 2 * n_players] == 'max_blue_x'
    assert names[n_summary + 3 * n_players] == 'max_red_x'
    assert names[n_summary + 4 * n_players - 1] == 'max_red_unknown_item_count'


def test_b1_columns_leave_out_the_dropped_names_everywhere():
    dropped = ('t_s', 'rift_element_known', 'flag_stasis', 'x')
    names = features.b1_columns(dropped)
    assert len(names) == 433 - 2 - 4 * 2
    assert 't_s' not in names and 'rift_element_known' not in names and 'sum_blue_flag_stasis' not in names and 'max_red_x' not in names
    assert 'game_time_s' in names and 'sum_blue_y' in names
    assert names == features.b1_columns(list(dropped)) == features.b1_columns(set(dropped)) == features.b1_columns(tuple(reversed(dropped)))


def test_b1_columns_refuse_a_name_that_is_no_column():
    with pytest.raises(ValueError, match='no_such_column'):
        features.b1_columns(('t_s', 'no_such_column'))
    with pytest.raises(ValueError, match='sum_blue_x'):
        features.b1_columns(('sum_blue_x',))                                      # a b1 name is not a column of the view


def test_b1_equals_the_oracle_on_every_view_of_the_toy(toy_s2):
    for dropped in (toy_drop_list(toy_s2), (), ('t_s', 'x', 'y', 'dragon_up', 'level', 'tag_Mage', 'blue_towers_outer')):
        x = features.b1(toy_s2, TOY_ALL_VIEWS, dropped)
        want = np.array([oracle_b1(toy_s2, int(v), dropped) for v in TOY_ALL_VIEWS])
        assert x.dtype == np.float32 and x.shape == (TOY_N_VIEWS, len(features.b1_columns(dropped)))
        assert same(x, want), dropped


@pytest.mark.parametrize('what', ['summary', 'players', 'both'])
def test_b1_when_a_whole_table_of_columns_is_dropped(toy_s2, what):
    dropped = {'summary': tuple(tokens.SUMMARY), 'players': tuple(tokens.PLAYER_NUM),
               'both': tuple(tokens.SUMMARY) + tuple(tokens.PLAYER_NUM)}[what]
    x = features.b1(toy_s2, TOY_ALL_VIEWS, dropped)
    names = features.b1_columns(dropped)
    assert x.shape == (TOY_N_VIEWS, len(names)) and same(x, np.array([oracle_b1(toy_s2, int(v), dropped) for v in TOY_ALL_VIEWS]))
    assert len(names) == {'summary': 4 * 81 + 2, 'players': 107 + 2, 'both': 2}[what]
    if what == 'both':
        assert names == ('geo_min', 'geo_mean')                                       # only the distances are left


def test_b1_hand_numbers_for_one_view(tmp_path):
    summary, players, geo = blank_feature_arrays(1)
    summary[0, SUM_I['game_time_s']] = 750.0
    summary[0, SUM_I['blue_dragons']] = 2.0
    players[0, :5, PLR_I['totalGold']] = (1000, 2000, 3000, 4000, 5000)
    players[0, 5:, PLR_I['totalGold']] = (1500, 1500, 1500, 1500, 1500)
    players[0, :, PLR_I['x']] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    geo[0, :5, 5:] = 100.0
    geo[0, 1, 7] = 30.0
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    names = list(features.b1_columns(()))
    x = features.b1(store, [0], ())[0]
    at = lambda n: x[names.index(n)]                                                        # noqa: E731
    assert at('game_time_s') == 750.0 and at('blue_dragons') == 2.0
    assert at('sum_blue_totalGold') == 15000 and at('sum_red_totalGold') == 7500 and at('max_blue_totalGold') == 5000
    assert at('max_red_totalGold') == 1500
    assert at('sum_blue_x') == 15 and at('sum_red_x') == 40 and at('max_blue_x') == 5 and at('max_red_x') == 10
    assert at('geo_min') == 30 and at('geo_mean') == pytest.approx((24 * 100 + 30) / 25)


def test_b1_leaves_nan_as_nan_and_fills_nothing(toy_s2):
    d = toy_drop_list(toy_s2)
    x = features.b1(toy_s2, TOY_ALL_VIEWS, d)
    names = list(features.b1_columns(d))
    nan = np.isnan(x)
    assert nan.any() and not np.isinf(x).any()
    # view 3: blue player 2 has no position: the blue team aggregates of x, y and wall_fraction are NaN, the red ones and the others are not
    for name in ('x', 'y', 'wall_fraction'):
        assert nan[3, names.index(f'sum_blue_{name}')] and nan[3, names.index(f'max_blue_{name}')]
        assert not nan[3, names.index(f'sum_red_{name}')] and not nan[3, names.index(f'max_red_{name}')]
    assert not nan[3, names.index('sum_blue_totalGold')] and not nan[3, names.index('max_blue_level')]
    # view 25: a blue and a red player are missing; view 42: everyone is
    assert nan[25, names.index('sum_blue_x')] and nan[25, names.index('sum_red_x')]
    assert nan[42, names.index('sum_blue_x')] and nan[42, names.index('max_red_y')] and nan[42, names.index('geo_min')]
    # the geodesics: a cross-team pair is left, or none (NaN)
    assert not nan[3, names.index('geo_min')] and nan[10, names.index('geo_min')] and nan[10, names.index('geo_mean')]
    # the NaN cells are exactly the ones the oracle expects (the oracle propagates NaN through a team's sum and maximum)
    assert np.array_equal(nan, np.isnan(np.array([oracle_b1(toy_s2, int(v), d) for v in TOY_ALL_VIEWS])))
    n_summary = len(tokens.SUMMARY) - len([n for n in d if n in SUM_I])
    assert not nan[:, :n_summary].any()                                          # the summary part has no NaN in the toy


def test_b1_is_float32_in_the_order_given_and_of_no_view_empty(toy_s2):
    d = toy_drop_list(toy_s2)
    x = features.b1(toy_s2, TOY_VIEWS, d)
    assert x.dtype == np.float32 and x.flags['C_CONTIGUOUS'] and len(x) == len(TOY_VIEWS)
    order = np.array([3, 0, 3, 20])
    assert np.array_equal(features.b1(toy_s2, TOY_VIEWS[order], d), x[order], equal_nan=True)
    assert features.b1(toy_s2, [], d).shape == (0, len(features.b1_columns(d)))


def test_b1_reads_the_three_x_arrays_and_nothing_else(toy_s2):
    seen, _ = arrays_read(features.b1, toy_s2, TOY_VIEWS, ())
    assert names_in(seen) == X_ARRAYS


def test_b1_does_not_depend_on_the_chunk_size(toy_s2, monkeypatch):
    d = toy_drop_list(toy_s2)
    want = features.b1(toy_s2, TOY_ALL_VIEWS, d)
    for chunk in (1, 6, 1000):
        monkeypatch.setattr(features, '_CHUNK', chunk)
        assert np.array_equal(features.b1(toy_s2, TOY_ALL_VIEWS, d), want, equal_nan=True)


def test_b1_and_b0_do_not_change_the_store(toy_s2):
    before = {name: np.array(toy_s2.array(name)) for name in X_ARRAYS}
    features.b1(toy_s2, TOY_ALL_VIEWS, ())
    features.b0(toy_s2, TOY_VIEWS)
    assert all(np.array_equal(before[name], np.asarray(toy_s2.array(name)), equal_nan=True) for name in X_ARRAYS)


def test_b1_team_sum_and_max_of_every_player_column_on_random_data(tmp_path):
    rng = np.random.default_rng(5)
    summary, players, geo = blank_feature_arrays(40)
    players[:] = rng.integers(-20, 60, players.shape).astype(np.float32) / 4          # quarters: exact in float32 and float64
    players[rng.random(players.shape) < 0.02] = np.nan
    geo[:] = rng.integers(1, 90, geo.shape) * 10
    geo[rng.random(geo.shape) < 0.1] = np.nan
    summary[:] = rng.integers(0, 9, summary.shape)
    store = build_feature_store(tmp_path / 's2', summary, players, geo)
    dropped = ('t_s', 'level', 'tag_Mage')
    assert same(features.b1(store, np.arange(40), dropped), np.array([oracle_b1(store, v, dropped) for v in range(40)]))


# ---- the masked M4 input -----------------------------------------------------------------------------------------------------------------
def m4_example():
    row = np.array([[1.5, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    q, m, z = np.array([0, 5, 17, 3]), np.array([0, 10, 51, 7]), np.array([-1, 0, 9, 4])
    K = np.array([[1, 0, 403.125, 0.0], [2, 1, 1186.875, 450.0], [0, 0, 0.0, 0.0], [5, 5, 3650.0, 2500.0]])
    O = np.arange(4 * 18, dtype=np.float64).reshape(4, 18) % 3
    return row, q, m, z, K, O


def test_m4_inputs_are_the_row_then_q_m_z_then_the_kills_and_gold_in_thousands_then_o():
    row, q, m, z, K, O = m4_example()
    x = features.m4_masked_inputs(row, q, m, z, K, O)
    assert x.dtype == np.float32 and x.shape == (4, 3 + 3 + 4 + 18) and x.flags['C_CONTIGUOUS']
    assert np.array_equal(x[:, :3], row, equal_nan=True) and np.isnan(x[0, 1])                  # NaN of a b1 row stays NaN
    assert x[:, 3].tolist() == q.tolist() and x[:, 4].tolist() == m.tolist() and x[:, 5].tolist() == z.tolist()
    assert x[:, 6].tolist() == [1, 2, 0, 5] and x[:, 7].tolist() == [0, 1, 0, 5]                # kills of blue and of red
    assert x[:, 8].tolist() == [np.float32(0.403125), np.float32(1.186875), 0.0, np.float32(3.65)]       # gold / 1,000
    assert x[:, 9].tolist() == [0.0, np.float32(0.45), 0.0, 2.5]
    assert np.array_equal(x[:, 10:], O.astype(np.float32))


def test_m4_does_not_change_its_arguments():
    args = m4_example()
    before = [a.copy() for a in args]
    features.m4_masked_inputs(*args)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(args, before))


def test_m4_accepts_any_integer_dtype_for_q_m_z_and_lists():
    row, q, m, z, K, O = m4_example()
    a = features.m4_masked_inputs(row, q, m, z, K, O)
    b = features.m4_masked_inputs(row, q.astype(np.int8), m.astype(np.uint8), z.astype(np.int16), K.astype(np.float32), O.astype(np.int64))
    assert np.array_equal(a, b, equal_nan=True)
    as_lists = features.m4_masked_inputs(row.tolist(), q.tolist(), m.tolist(), z.tolist(), K.tolist(), O.tolist())
    assert np.array_equal(as_lists, a, equal_nan=True)


def test_m4_works_with_the_b0_width_and_with_a_wide_b1_row():
    for d in (39, 427):
        n = 3
        x = features.m4_masked_inputs(np.zeros((n, d), np.float32), [0, 1, 2], [3, 4, 5], [-1, 0, 1], np.zeros((n, 4)), np.zeros((n, 18)))
        assert x.shape == (n, d + 25) and len(features.m4_columns([f'c{i}' for i in range(d)])) == d + 25


def test_m4_of_no_row():
    x = features.m4_masked_inputs(np.zeros((0, 5)), [], [], [], np.zeros((0, 4)), np.zeros((0, 18)))
    assert x.shape == (0, 30) and x.dtype == np.float32


def test_m4_columns_name_every_column_of_the_input():
    names = features.m4_columns(('a', 'b', 'c'))
    assert names[:3] == ('a', 'b', 'c') and names[3:6] == ('q', 'm', 'z')
    assert names[6:10] == ('kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k')
    assert len(names) == len(set(names)) == 3 + 3 + 4 + 18
    cells = names[10:]
    assert len(cells) == targets.N_CELLS == 18
    for kind, name in enumerate(C.O_KINDS):                                          # the cell of S1 is 2 kind + taker (0 blue, 1 red)
        assert cells[2 * kind] == f'obj_{name}_blue' and cells[2 * kind + 1] == f'obj_{name}_red'
    assert features.m4_columns(features.B0_NAMES)[:39] == features.B0_NAMES and len(features.m4_columns(features.B0_NAMES)) == 64


@pytest.mark.parametrize('row_names', [('a', 'q'), ('a', 'kills_red'), ('a', 'obj_baron_blue'), ('a', 'a')])
def test_m4_columns_refuse_a_row_name_that_is_also_an_m4_name(row_names):
    with pytest.raises(ValueError, match='more than once'):
        features.m4_columns(row_names)


@pytest.mark.parametrize('which, bad, error', [
    ('q', [-1, 0, 0, 0], ValueError), ('q', [0, 0, 0, 18], ValueError), ('m', [0, 0, 0, 52], ValueError), ('m', [-1, 0, 0, 0], ValueError),
    ('z', [-2, 0, 0, 0], ValueError), ('z', [0, 0, 0, 10], ValueError), ('q', [0.0, 1.0, 2.0, 3.0], TypeError),
    ('m', [True, False, True, False], TypeError), ('z', np.zeros((4, 1), dtype=int), ValueError), ('q', [0, 1, 2], ValueError),
    ('K', np.zeros((4, 3)), ValueError), ('K', np.full((4, 4), np.nan), ValueError), ('K', np.full((4, 4), -1.0), ValueError),
    ('K', np.zeros(4), ValueError), ('O', np.zeros((4, 17)), ValueError), ('O', np.full((4, 18), -1.0), ValueError),
    ('O', np.full((4, 18), np.inf), ValueError), ('row_view', np.zeros(4), ValueError), ('row_view', np.zeros((3, 3)), ValueError),
    ('row_view', np.array([['a'] * 3] * 4), TypeError)])
def test_m4_refuses_an_argument_out_of_its_range(which, bad, error):
    row, q, m, z, K, O = m4_example()
    args = dict(row_view=row, q=q, m=m, z=z, K=K, O=O)
    args[which] = bad
    with pytest.raises(error, match=rf'\b{which}\b'):
        features.m4_masked_inputs(**args)


def test_m4_ranges_come_from_the_config():
    row, q, m, z, K, O = m4_example()
    ok = features.m4_masked_inputs(row, [C.N_BINS - 1] * 4, [C.N_MARKS - 1] * 4, [9] * 4, K, O)            # the largest values pass
    assert ok.shape[0] == 4 and C.N_BINS == 18 and C.N_MARKS == 52
    from c_ept.preprocessing import terrain
    assert len(terrain.ZONES) == features.N_ZONES == 10


def test_the_masked_input_does_not_depend_on_anything_but_its_arguments():
    row, q, m, z, K, O = m4_example()
    again = features.m4_masked_inputs(row.copy(), q.copy(), m.copy(), z.copy(), K.copy(), O.copy())
    assert np.array_equal(features.m4_masked_inputs(row, q, m, z, K, O), again, equal_nan=True)


# ---- the zone of the first kill of a unit (ruling R17) -----------------------------------------------------------------------------------
# The zone label of the engagement heads: ``terrain.zone_of`` of the position of the first kill of each unit, -1 where the kill has no
# position. The positions below were worked out by hand from the rules in ``terrain.ZONE_RULES`` (the map has the side W = 14,875, its
# centre is (7,437.5, 7,437.5); with a = x - W / 2, b = y - W / 2, s = a + b and d = a - b a point is: base_blue if s < -9,520, base_red if
# s > 9,520, top if a < -5,503.75 or b > 5,503.75, bot if b < -5,503.75 or a > 5,503.75, mid if |d| < 1,041.25, river if |s| < 1,041.25,
# else jungle: blue / red by the sign of s, top / bot by the sign of d, d < 0 being top):
#     (0, 0)               s = -14,875                      base_blue 0
#     (14,875, 14,875)     s = +14,875                      base_red 1
#     (7,437.5, 14,875)    a = 0, b = 7,437.5               top 2
#     (7,437.5, 7,437.5)   s = 0, d = 0                     mid 3
#     (14,875, 7,437.5)    a = 7,437.5, b = 0               bot 4
#     (6,000, 9,000)       s = 125, d = -3,000              river 5
#     (3,000, 7,000)       s = -4,875, d = -4,000           jg_blue_top 6
#     (4,000, 2,000)       s = -8,875, d = +2,000           jg_blue_bot 7
#     (9,000, 11,500)      s = 5,625, d = -2,500            jg_red_top 8
#     (11,500, 9,000)      s = 5,625, d = +2,500            jg_red_bot 9
ZONE_HAND = (((0.0, 0.0), 0), ((14_875.0, 14_875.0), 1), ((7_437.5, 14_875.0), 2), ((7_437.5, 7_437.5), 3), ((14_875.0, 7_437.5), 4),
             ((6_000.0, 9_000.0), 5), ((3_000.0, 7_000.0), 6), ((4_000.0, 2_000.0), 7), ((9_000.0, 11_500.0), 8), ((11_500.0, 9_000.0), 9))


def kill_store(root, x, y, missing, kill0) -> Store:
    """A store of just the four arrays that ``unit_zone`` reads: the kills (x, y, pos_missing) and the first kill of each unit."""
    s = Store(root)
    for name, values, dtype in (('kills/x', x, 'f8'), ('kills/y', y, 'f8'), ('kills/pos_missing', missing, '?'),
                                ('units/kill0', kill0, 'i4')):
        a = s.allocate(name, np.shape(values), dtype)
        a[...] = values
        a.flush()
        del a
    s.finalize()
    return s


def zone_world(tmp_path):
    """Ten kills at the hand positions, then five kills that no unit starts with or that have no position; the units start (in a scrambled
    order, one kill twice) at kills 4, 0, 3, 9, 2, 1, 5, 6, 7, 8, then at the kills without a position. Returns (store, expected zones)."""
    xy = [p for p, _ in ZONE_HAND]
    x = [p[0] for p in xy] + [0.0, np.nan, 100.0, 7_437.5, 5_000.0]            # kill 10: flagged (it is at the corner), 11: NaN, no flag
    y = [p[1] for p in xy] + [0.0, 7_437.5, np.nan, 7_437.5, 5_000.0]          # 12: NaN y, flagged; 13: a good position nobody starts with
    missing = [False] * 10 + [True, False, True, False, False]
    kill0 = [4, 0, 3, 9, 2, 1, 5, 6, 7, 8, 10, 11, 12, 4]
    expected = [dict(ZONE_HAND_BY_KILL)[k] for k in kill0[:10]] + [-1, -1, -1, 4]
    return kill_store(tmp_path / 's1', x, y, missing, kill0), np.array(expected, dtype=np.int8)


ZONE_HAND_BY_KILL = tuple(enumerate(zone for _, zone in ZONE_HAND))


def test_unit_zone_is_the_zone_of_the_first_kill_with_minus_one_where_there_is_no_position(tmp_path):
    store, expected = zone_world(tmp_path)
    z = features.unit_zone(store)
    assert z.dtype == np.int8 and z.shape == (14,)
    assert z.tolist() == expected.tolist()


def test_unit_zone_hand_positions_are_what_terrain_says():
    from c_ept.preprocessing import terrain                                            # the hand numbers above and the code agree
    for (x, y), zone in ZONE_HAND:
        assert int(terrain.zone_of(x, y)) == zone
    assert terrain.ZONES == ('base_blue', 'base_red', 'top', 'mid', 'bot', 'river', 'jg_blue_top', 'jg_blue_bot', 'jg_red_top',
                             'jg_red_bot')


def test_unit_zone_the_flag_beats_the_position_and_a_nan_position_is_missing_without_it(tmp_path):
    store, _ = zone_world(tmp_path)
    z = features.unit_zone(store)
    assert z[10] == -1                                   # kill 10: flagged, at the corner (0, 0) that would be base_blue (0)
    assert z[11] == -1                                   # kill 11: x is NaN and there is no flag
    assert z[12] == -1                                   # kill 12: flagged, y is NaN


def test_unit_zone_uses_the_first_kill_of_the_unit_not_the_unit_number(tmp_path):
    store, expected = zone_world(tmp_path)
    z = features.unit_zone(store)
    # the zone of kill k is k in this world, and the units start at the kills 4, 0, 3, 9 ...: the unit number would give 0, 1, 2, 3
    assert z[:4].tolist() == [4, 0, 3, 9] and z[13] == 4 == expected[13]


def test_unit_zone_ignores_the_kills_that_start_no_unit(tmp_path):
    x = [0.0, 14_875.0, 7_437.5]
    store = kill_store(tmp_path / 's1', x, [0.0, 14_875.0, 7_437.5], [False, True, True], [0])
    assert features.unit_zone(store).tolist() == [0]      # only kill 0 is a first kill; kills 1 and 2 are flagged and do not matter


def test_unit_zone_reads_exactly_four_arrays(tmp_path):
    store, _ = zone_world(tmp_path)
    seen, z = arrays_read(features.unit_zone, store)
    assert names_in(seen) == {'units/kill0', 'kills/x', 'kills/y', 'kills/pos_missing'}
    assert len(z) == 14


def test_unit_zone_equals_a_loop_on_the_value_toys(toy_value_s1):
    from c_ept.preprocessing import terrain
    x, y = np.asarray(toy_value_s1.array('kills/x')), np.asarray(toy_value_s1.array('kills/y'))
    missing, kill0 = np.asarray(toy_value_s1.array('kills/pos_missing')), np.asarray(toy_value_s1.array('units/kill0'))
    expected = [-1 if missing[k] else int(terrain.zone_of(x[k], y[k])) for k in kill0]
    z = features.unit_zone(toy_value_s1)
    assert z.tolist() == expected and len(z) == len(kill0) == 10
    assert z.min() >= 0 and set(z.tolist()) <= set(range(10))        # the value toys have a position for every first kill


def test_unit_zone_of_no_unit(tmp_path):
    store = kill_store(tmp_path / 's1', [1.0], [1.0], [False], np.empty(0, dtype=np.int32))
    z = features.unit_zone(store)
    assert z.dtype == np.int8 and z.shape == (0,)


@pytest.mark.parametrize('kill0', [[3], [-1], [0, 4]])
def test_unit_zone_refuses_a_first_kill_that_is_not_a_kill(tmp_path, kill0):
    store = kill_store(tmp_path / 's1', [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [False] * 3, kill0)
    with pytest.raises(ValueError, match='units/kill0'):
        features.unit_zone(store)


def test_unit_zone_refuses_a_store_without_the_arrays(tmp_path, toy_s1):
    with pytest.raises(ValueError, match='units/kill0') as raised:
        features.unit_zone(toy_s1)
    assert KEY_PREFIX not in str(raised.value)
    s = Store(tmp_path / 'partial')
    a = s.allocate('units/kill0', (1,), 'i4')
    a[...] = 0
    a.flush()
    del a
    s.finalize()
    with pytest.raises(ValueError, match='kills/'):
        features.unit_zone(s)


def test_unit_zone_is_the_ruling_r17_signature():
    assert list(inspect.signature(features.unit_zone).parameters) == ['s1']


def test_unit_zone_is_deterministic_and_changes_nothing(tmp_path):
    store, _ = zone_world(tmp_path)
    before = {n: np.array(store.array(n)) for n in ('kills/x', 'kills/y', 'kills/pos_missing', 'units/kill0')}
    assert np.array_equal(features.unit_zone(store), features.unit_zone(store))
    for n, a in before.items():
        assert np.array_equal(a, np.asarray(store.array(n)), equal_nan=True)


# ---- structure ---------------------------------------------------------------------------------------------------------------------------
def test_the_module_imports_no_runner_no_cohort_no_raw_reader_at_module_level():
    imported = module_level_imports(SOURCE)
    forbidden = {'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.cohort', 'c_ept.runlog', 'c_ept.terrain', 'c_ept.store_build', 'c_ept.reader',
                 'c_ept.tokens', 'c_ept.value', 'c_ept.compose', 'c_ept.nll'}
    assert not imported & forbidden, imported & forbidden


def test_the_module_names_only_the_arrays_it_is_documented_to_read():
    text = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(text)
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.count('/') == 1 and ' ' not in node.value and node.value.split('/')[0].isidentifier()}
    documented = {'matches/role', 'frame_rows/match', 'onsets/row', 'views/match', 'x/summary', 'x/players_num', 'x/geo_pp', 'units/match',
                  'units/kill0', 'kills/x', 'kills/y', 'kills/pos_missing'}                  # the last four: ``unit_zone`` (ruling R17)
    assert names <= documented, names
    assert 'matches/key' not in text and 'allow_holdout' not in text


def test_the_module_does_not_read_the_labels_the_units_the_kills_or_the_event_rows():
    text = SOURCE.read_text(encoding='utf-8')
    # "'kills/" is not on the list: ``unit_zone`` (ruling R17) reads kills/x, kills/y and kills/pos_missing, and the test above pins that
    # these are the only kill arrays the module names
    for forbidden in ("'objectives/", "'event_rows/", "'labels/", "'units/f", "'units/K", "'units/O", "'units/conv", "'frames/"):
        assert forbidden not in text, forbidden


def test_a_wrong_role_never_reaches_a_key(toy_s1):
    with pytest.raises(ValueError) as raised:
        features.split_rows(toy_s1, 'TEST')
    assert KEY_PREFIX not in message_of(raised)
