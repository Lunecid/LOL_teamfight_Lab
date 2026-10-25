"""Tests of the minute gold-lead panel, the event table and the window table of the D7 estimator (``c_ept.flowprice``, stage S3 Task 1).

Every store is a toy one under pytest's folders (``s3_toys``); no run folder, no raw file and no catalog is read. Three kinds of check:
numbers worked out by hand for the toy matches (the ``TOY_*`` constants of ``s3_toys``, with the working in its comments), a brute-force
second implementation written from the text of the rules here (``oracle_*``: one loop per event, no shared code with the module) that
is compared on the toy matches and on random ones, and one run of the real S1 extraction on the same matches (``toy_record``), whose arrays
and tables must equal the toy store's. The module is also pinned to the arrays it is allowed to read.

The second half (from "the estimators") tests stage S3 Task 2: the within-event regression of the flow price, the window regression of the
immediate value, the match bootstrap, the price table and the ``flowprice`` command line. Truth comes from three places: the hand numbers of
note 03 sections A and B, a second implementation written out in the test (explicit event dummies, explicit row weights, ``lstsq``), and
tables made from a known truth (synthetic ones, and toy matches with a planted baron that go through the S1 and S2 stores).
"""
import ast
import dataclasses
import gc
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.valuation import flowprice
from c_ept.data.store_core import Store
from c_ept.tests import synth
from c_ept.tests.conftest import PROTECTED_IMPORT_PRELUDE, import_watch_names, module_level_imports
from c_ept.tests.s3_toys import (   # noqa: F401  (toy_s1 and toy_s2 are fixtures, imported by ruling R1)
    BLUE5, GROUP_NAMES, JUNK, KEY_PREFIX, KIND_SUB, NOBODY, RED5, SUMMARY_PAIRS, TOY_BETA7, TOY_CAL_BIAS, TOY_CAL_E, TOY_CAL_SLOPE,
    TOY_CAL_V, TOY_CAL_W, TOY_CAL_WSLOPE, TOY_CO_BARON_M0, TOY_CO_DRAGON_M1, TOY_CO_ELDER_M0, TOY_CO_RED_DRAGON, TOY_CTRL_ROWS_M0,
    TOY_DEV_MATCHES, TOY_EVENT_COMPLETE, TOY_EVENT_Y, TOY_EVENTS_BY_GROUP, TOY_EXP, TOY_EXPECTED_V60, TOY_EXPECTED_V60_CUT,
    TOY_FIT_MATCHES, TOY_GOLD, TOY_GOLD_BIN3_MEAN, TOY_GOLD_BINS, TOY_GOLD_OVERFLOW_MEAN, TOY_IMMEDIATE, TOY_INCOMPLETE_BY_GROUP,
    TOY_K_RED_DRAGON, TOY_LEAD_M0, TOY_MATCHES, TOY_MINUTES, TOY_N_DROPPED_LAST, TOY_N_EVENTS, TOY_N_FIT_FRAME_ROWS,
    TOY_N_FIT_WINDOW_ROWS, TOY_N_FRAMES, TOY_N_INCOMPLETE, TOY_N_UNKNOWN, TOY_OBJ_UNWINDOWED, TOY_PHI, TOY_PHI_Y, TOY_POISSON_NLL,
    TOY_PRETREND, TOY_PRICE, TOY_RATE, TOY_STOP_MATCHES, TOY_T0_S, TOY_T_DRAGON, TOY_T_END_CUT_S, TOY_T_END_FAR_S, TOY_T_PLATE,
    TOY_T_RED_DRAGON, TOY_TS_M0, TOY_WINDOW_M0, TOY_WINDOW_TREAT, TOY_WINDOW_Y, TOY_Y, TOY_Y_BARON_M0, TOY_Y_DRAGON_M1, TOY_Y_ELDER_M0,
    TOY_Y_RED_DRAGON, ToyMatch, build_toy_s1, build_toy_s2, frame_times, kill, obj, summary_values, toy_record, toy_s1, toy_s2)
from c_ept.tests.s3_toys import (   # the synthetic tables and the planted matches of the estimator tests (Task 2)
    BETA_FLAT, BETA_RAMP, DEFAULT_WINDOW_EFFECT, N_MIXED_MATCHES, PLANT_EFFECT, PLANT_PRICE, RAMP_PRETREND, TOY_BETA7_TRY,
    TOY_EVENT_Y_TRY, TOY_IMMEDIATE_TRY, TOY_PRETREND_TRY, TOY_WINDOW_Y_TRY, TRUE_BETA, TRUE_GAMMA, TRUE_WINDOW_CO, concat_events,
    event_table_from_y, mixed_events, mixed_windows, planted_baron_matches, random_summary_values, synthetic_events, synthetic_windows,
    window_design, with_event_level_shift)

REPO = Path(C.__file__).parents[1]
GROUPS = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower_outer', 'tower_inner', 'tower_base', 'tower_nexus', 'inhibitor',
          'plate')
EDGES = (120, 300, 600, 900, 1200, 1500, 1800, 2100)
CTRL = ([f'lead_x_phase{i}' for i in range(8)] + [f'phase{i}' for i in range(8)] + ['momentum', 'cum_kill_diff']
        + ['towers_outer', 'towers_inner', 'towers_base', 'towers_nexus', 'inhibs_down', 'plates', 'dragons', 'elders', 'barons',
           'heralds', 'atakhans', 'voidgrubs', 'has_soul', 'baron_buff', 'elder_buff'])
CO = ('kill_gold',) + GROUPS


def co_col(name: str) -> int:
    return CO.index(name)


def message_of(raised) -> str:
    """The text of a caught error. The toy stores hold sentinel match keys (``KEY_PREFIX``); none may get into a message."""
    assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)
    return str(raised.value)


# ---- names and small functions ---------------------------------------------------------------------------------------------------------
def test_names_of_the_groups_the_controls_and_the_constants():
    assert flowprice.GROUPS == GROUPS == GROUP_NAMES
    assert flowprice.FIXED_IMMEDIATE == ('tower_base', 'tower_nexus', 'inhibitor')
    assert flowprice.PHASE_EDGES_S == EDGES
    assert flowprice.CO_NAMES == CO
    assert flowprice.CTRL_NAMES == tuple(CTRL) and len(flowprice.CTRL_NAMES) == 33
    assert flowprice.REL_MINUTES == (-4, -3, -2, -1, 0, 1, 2)
    assert flowprice.WINDOW_CO_NAMES == ('kill_gold', 'kill_count', 'assist_count')


def test_group_of_maps_the_kinds_and_the_tower_tiers():
    names = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower', 'inhibitor', 'plate')
    assert C.O_KINDS == names                                                          # the kind numbers below follow the config
    kind = np.arange(9)
    assert flowprice.group_of(kind, np.full(9, -1)).tolist() == [0, 1, 2, 3, 4, 5, -1, 10, 11]      # a tower without a tier: unknown
    assert flowprice.group_of(np.full(5, 6), np.array([0, 1, 2, 3, -1])).tolist() == [6, 7, 8, 9, -1]
    assert flowprice.group_of(np.array([0, 4, 8, 7]), np.array([3, 3, 3, 3])).tolist() == [0, 4, 11, 10]      # only a tower reads the tier
    assert flowprice.group_of(np.array([-1, 9, 100]), np.array([0, 0, 0])).tolist() == [-1, -1, -1]            # no such kind
    assert flowprice.group_of(np.array([6]), np.array([4])).tolist() == [-1]                                    # no such tier
    out = flowprice.group_of(np.array([0, 6], dtype=np.int8), np.array([-1, 2], dtype=np.int8))
    assert out.dtype == np.int8 and out.tolist() == [0, 8]
    assert int(flowprice.group_of(6, 1)) == 7 and int(flowprice.group_of(-1, -1)) == -1        # scalars work too


def test_phase_bins_follow_the_edges_and_the_last_one_is_open():
    ms = lambda s: int(round(s * 1000))                                                # noqa: E731
    edges = [ms(e) for e in EDGES]
    assert [int(flowprice.phase_of(t)) for t in edges] == list(range(8))               # an edge belongs to the bin that starts there
    assert [int(flowprice.phase_of(t - 1)) for t in edges[1:]] == list(range(7))       # one ms before it, to the bin before
    assert int(flowprice.phase_of(ms(2100))) == 7 and int(flowprice.phase_of(ms(9000))) == 7
    assert int(flowprice.phase_of(ms(60))) == 0 and int(flowprice.phase_of(119_999)) == 0   # before the first edge: bin 0
    got = flowprice.phase_of(np.array([120_300, 420_300, 3_000_000]))
    assert got.dtype == np.int8 and got.tolist() == [0, 1, 7]


# ---- the minute panel -------------------------------------------------------------------------------------------------------------------
def test_lead_is_blue_minus_red_total_gold(toy_s1):
    ts, lead = flowprice.lead_series(toy_s1, 0)
    assert lead[3] == pytest.approx(TOY_LEAD_M0[3])
    assert ts.dtype == np.int64 and lead.dtype == np.float64
    assert ts.tolist() == list(TOY_TS_M0) and lead.tolist() == list(TOY_LEAD_M0)


def test_lead_series_holds_the_regular_frames_only_and_works_for_every_toy_match(toy_s1):
    for m, spec in enumerate(TOY_MATCHES):
        ts, lead = flowprice.lead_series(toy_s1, m)
        assert len(ts) == len(lead) == TOY_N_FRAMES[m] == len(spec.lead)                    # R5: no game-end frame, none is stored
        assert ts.tolist() == list(frame_times(len(spec.lead))) and lead.tolist() == list(spec.lead)


def test_lead_series_reads_the_team_array_not_the_participant_order(toy_s1):
    spec = TOY_MATCHES[2]
    assert spec.team != BLUE5 + RED5                                                       # blue is participants 1, 3, 5, 7, 9
    assert flowprice.lead_series(toy_s1, 2)[1].tolist() == list(spec.lead)


def test_a_frame_without_total_gold_is_refused_naming_the_match_row_and_never_the_key(tmp_path):
    s1 = build_toy_s1(tmp_path / 's1')
    a = np.load(s1.path('frames/num'), mmap_mode='r+')
    a[3, 0, C.FRAME_FIELDS.index('totalGold')] = np.nan                                    # match 0, frame 3, participant 1
    a.flush()
    del a
    with pytest.raises(ValueError, match=r'match row 0: totalGold is missing in 1 of 12 frames') as raised:
        flowprice.lead_series(s1, 0)
    message_of(raised)
    with pytest.raises(ValueError, match='totalGold') as raised:
        flowprice.event_table(Store(tmp_path / 's1'))
    message_of(raised)
    assert flowprice.lead_series(s1, 1)[1].tolist() == list(TOY_MATCHES[1].lead)           # the other matches are fine
    assert len(flowprice.event_table(Store(tmp_path / 's1'), roles=('STOP',)).t) == 2        # and a role that does not include it


def test_a_team_id_that_is_neither_blue_nor_red_is_refused(tmp_path):
    s1 = build_toy_s1(tmp_path / 's1')
    a = np.load(s1.path('matches/team'), mmap_mode='r+')
    a[0, 3] = 300
    a.flush()
    del a
    with pytest.raises(ValueError, match='team') as raised:
        flowprice.event_table(Store(tmp_path / 's1'))
    message_of(raised)


# ---- the event table --------------------------------------------------------------------------------------------------------------------
def test_event_minute_and_taker_signed_changes(toy_s1):          # a dragon of red: the signs flip
    t = flowprice.event_table(toy_s1)
    i = t.index_of(match=0, t=TOY_T_RED_DRAGON)
    assert t.k[i] == TOY_K_RED_DRAGON and np.allclose(t.y[i], TOY_Y_RED_DRAGON)
    assert t.group[i] == GROUPS.index('dragon') and t.taker[i] == 1 and bool(t.complete[i])
    for name, want in TOY_CO_RED_DRAGON.items():                                         # the other events of the seven minutes
        assert t.co[i, :, co_col(name)].tolist() == list(want), name
    others = [c for c in CO if c not in TOY_CO_RED_DRAGON]
    assert not t.co[i][:, [co_col(c) for c in others]].any()                              # nothing else, the dragon itself included


def test_a_dragon_of_blue_keeps_the_signs_and_the_other_events_of_its_minutes(toy_s1):
    t = flowprice.event_table(toy_s1)
    i = t.index_of(match=1, t=TOY_T_DRAGON)
    assert t.k[i] == 6 and t.taker[i] == 0 and np.allclose(t.y[i], TOY_Y_DRAGON_M1)
    for name, want in TOY_CO_DRAGON_M1.items():
        assert t.co[i, :, co_col(name)].tolist() == list(want), name
    others = [c for c in CO if c not in TOY_CO_DRAGON_M1]
    assert not t.co[i][:, [co_col(c) for c in others]].any()


def test_the_event_minute_is_the_frame_that_ends_it_and_a_frame_time_belongs_to_its_own_minute(toy_s1):
    t = flowprice.event_table(toy_s1)
    for (m, when), minute in TOY_MINUTES.items():
        assert t.k[t.index_of(m, when)] == minute, (m, when)
    assert t.k.dtype == np.int32


def test_incomplete_windows_are_excluded_and_counted(toy_s1):
    t = flowprice.event_table(toy_s1)
    assert (~t.complete).sum() == TOY_N_INCOMPLETE and np.isnan(t.y[~t.complete]).all()
    assert not np.isnan(t.y[t.complete]).any()
    assert np.isnan(t.co[~t.complete]).all() and np.isfinite(t.co[t.complete]).all()      # NaN, never a made-up 0
    counts = t.counts_by_group()
    assert set(counts) == {'events', 'complete', 'incomplete'}
    assert counts['events'].tolist() == list(TOY_EVENTS_BY_GROUP) and counts['events'].sum() == TOY_N_EVENTS == len(t.t)
    assert counts['incomplete'].tolist() == list(TOY_INCOMPLETE_BY_GROUP) and counts['incomplete'].sum() == TOY_N_INCOMPLETE
    assert (counts['complete'] + counts['incomplete'] == counts['events']).all()
    assert counts['events'].shape == (12,) and counts['events'].dtype == np.int64


def test_a_window_is_complete_from_minute_5_to_the_third_last_frame(toy_s1):
    """k - 5 >= 0 and k + 2 <= the last regular frame: match 0 (12 frames) is complete for 5 <= k <= 9, match 1 (11 frames) for
    5 <= k <= 8."""
    t = flowprice.event_table(toy_s1)
    got = {(int(m), int(k)): bool(c) for m, k, c in zip(t.match, t.k, t.complete)}
    assert got == TOY_EVENT_COMPLETE
    assert 4 in {k for m, k in got if m == 0} and 5 in {k for m, k in got if m == 0}        # both sides of the first edge are tested ...
    assert 9 in {k for m, k in got if m == 0} and 10 in {k for m, k in got if m == 0}       # ... and of the last
    assert 8 in {k for m, k in got if m == 1} and 9 in {k for m, k in got if m == 1}


def test_a_baron_in_the_first_complete_minute_uses_the_first_frame(toy_s1):
    t = flowprice.event_table(toy_s1)
    i = t.index_of(0, 250_000)
    assert t.k[i] == 5 and bool(t.complete[i]) and np.allclose(t.y[i], TOY_Y_BARON_M0)         # y[r = -4] is lead[1] - lead[0]
    for name, want in TOY_CO_BARON_M0.items():
        assert t.co[i, :, co_col(name)].tolist() == list(want), name
    assert not t.co[i][:, [co_col(c) for c in CO if c not in TOY_CO_BARON_M0]].any()


def test_an_elder_in_the_last_complete_minute_uses_the_last_regular_frame(toy_s1):
    t = flowprice.event_table(toy_s1)
    i = t.index_of(0, 540_000)
    assert t.k[i] == 9 and bool(t.complete[i]) and np.allclose(t.y[i], TOY_Y_ELDER_M0)         # y[r = +2] is lead[11] - lead[10]
    for name, want in TOY_CO_ELDER_M0.items():
        assert t.co[i, :, co_col(name)].tolist() == list(want), name
    assert not t.co[i][:, [co_col(c) for c in CO if c not in TOY_CO_ELDER_M0]].any()


def test_same_minute_objectives_are_separate_events(toy_s1):     # the same team gets a dragon and a plate in one minute
    t = flowprice.event_table(toy_s1)
    a, b = t.index_of(match=1, t=TOY_T_DRAGON), t.index_of(match=1, t=TOY_T_PLATE)
    co = flowprice.CO_NAMES.index
    assert a != b and t.co[a, 4, co('plate')] == 1 and t.co[b, 4, co('dragon')] == 1
    assert t.k[a] == t.k[b] == 6 and bool(t.complete[a]) and bool(t.complete[b])


def test_an_event_leaves_itself_out_of_its_own_co_but_not_a_second_event_of_its_group(toy_s1):
    t = flowprice.event_table(toy_s1)
    first, second = t.index_of(1, 440_000), t.index_of(1, 445_000)                          # two plates of blue in minute 8
    third = t.index_of(1, 480_300)                                                          # the atakhan on the frame time, same minute
    assert t.co[first, 4, co_col('plate')] == 1 and t.co[second, 4, co_col('plate')] == 1   # only the other plate
    assert t.co[third, 4, co_col('plate')] == 2 and t.co[third, 4, co_col('atakhan')] == 0
    assert t.co[first, 4, co_col('atakhan')] == 1
    other = t.index_of(1, 355_000)                                          # the inhibitor of red, same minute as the dragon and the plate
    assert t.co[other, 4, co_col('dragon')] == -1 and t.co[other, 4, co_col('plate')] == -1 and t.co[other, 4, co_col('inhibitor')] == 0


def test_kill_gold_is_signed_by_the_taker_and_executions_pay_nobody(toy_s1):
    """Minute 7 of match 0 holds a red kill of 300, a blue kill of 150 on the frame time and an execution of 300 whose assister is blue:
    from the red dragon's side the minute is worth 300 - 150, from the blue inner tower's side (same minute) the opposite sign."""
    t = flowprice.event_table(toy_s1)
    red_dragon, blue_tower = t.index_of(0, 400_000), t.index_of(0, 401_000)
    kg = co_col('kill_gold')
    assert t.co[red_dragon, 4, kg] == 150 and t.co[blue_tower, 4, kg] == -150
    assert (t.co[red_dragon, :, kg] == -t.co[blue_tower, :, kg]).all()                     # the same minutes, seen from the two teams


def test_only_fit_matches(toy_s1):
    assert set(flowprice.event_table(toy_s1).match) <= TOY_FIT_MATCHES
    assert set(flowprice.event_table(toy_s1).match) == {0, 1, 4}                           # match 5 is FIT too, but has no objective


def test_only_the_asked_roles_are_in_the_event_table(toy_s1):
    assert set(flowprice.event_table(toy_s1, roles=('STOP',)).match) == TOY_STOP_MATCHES
    assert set(flowprice.event_table(toy_s1, roles=('DEV',)).match) == TOY_DEV_MATCHES
    both = flowprice.event_table(toy_s1, roles=('STOP', 'DEV'))
    assert set(both.match) == TOY_STOP_MATCHES | TOY_DEV_MATCHES and both.n_matches == 2
    every = flowprice.event_table(toy_s1, roles=('FIT', 'STOP', 'DEV'))
    assert set(every.match) == {0, 1, 2, 3, 4} and every.n_matches == 6                  # match 5 is read but has no objective: no event
    assert flowprice.event_table(toy_s1, roles='FIT').n_matches == 4                     # a bare name is one role


@pytest.mark.parametrize('roles, message', [(('TEST',), 'holdout'), (('EXTERNAL',), 'holdout'), (('FIT', 'TEST'), 'holdout'),
                                            (('CAL',), 'not used'), (('AUDIT',), 'not used'), (('UNUSED',), 'not used'),
                                            (('NOPE',), 'unknown'), ((), 'no role')])
def test_roles_the_stage_must_not_read_are_refused_before_anything_is_read(roles, message):
    with pytest.raises(ValueError, match=message):
        flowprice.event_table(None, roles=roles)                                           # None: a store that would fail on first use
    with pytest.raises(ValueError, match=message):
        flowprice.window_table(None, None, roles=roles)


def test_objectives_without_a_kind_a_taker_or_a_tower_tier_are_left_out_and_counted(toy_s1):
    t = flowprice.event_table(toy_s1)
    assert t.n_unknown == TOY_N_UNKNOWN == 5
    gone = [(0, 350_000), (0, 450_000), (0, 455_000), (1, 250_000), (1, 330_000)]
    for m, when in gone:
        with pytest.raises(KeyError):
            t.index_of(m, when)
    assert flowprice.event_table(toy_s1, roles=('STOP',)).n_unknown == 0
    assert flowprice.event_table(toy_s1, roles=('FIT', 'STOP', 'DEV')).n_unknown == 5


def test_the_table_is_ordered_by_match_then_time_and_has_the_documented_types(toy_s1):
    t = flowprice.event_table(toy_s1)
    n = TOY_N_EVENTS
    assert (t.match.dtype, t.group.dtype, t.taker.dtype, t.t.dtype, t.k.dtype, t.complete.dtype) == (
        np.int32, np.int8, np.int8, np.int64, np.int32, np.bool_)
    assert t.y.shape == (n, 7) and t.y.dtype == np.float64 and t.co.shape == (n, 7, 13) and t.co.dtype == np.float64
    assert (np.diff(t.match) >= 0).all() and all((np.diff(t.t[t.match == m]) >= 0).all() for m in set(t.match))
    assert t.match.tolist() == [0] * 9 + [1] * 11 + [4] * 2 and set(t.taker.tolist()) == {0, 1}
    assert t.n_matches == 4


def test_index_of_names_the_missing_and_the_ambiguous(tmp_path):
    twice = ToyMatch('FIT', (0, 100, 200, 300, 400, 500, 600, 700, 800, 900), 700_000,
                     (obj(250_000, 'plate', 0), obj(250_000, 'tower_outer', 1), obj(300_000, 'dragon', 0)))
    t = flowprice.event_table(build_toy_s1(tmp_path / 's1', (twice,)))
    with pytest.raises(KeyError, match='no event'):
        t.index_of(0, 251_000)
    with pytest.raises(KeyError, match='no event'):
        t.index_of(1, 250_000)                                                              # there is no match 1
    with pytest.raises(ValueError, match='2 events'):
        t.index_of(0, 250_000)
    assert t.group[t.index_of(0, 250_000, group='plate')] == 11
    assert t.group[t.index_of(0, 250_000, group=6)] == 6
    assert t.group[t.index_of(0, 300_000)] == 0


# ---- the window table -------------------------------------------------------------------------------------------------------------------
def test_window_rows_are_the_fit_frame_rows(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    assert len(w.y) + w.n_dropped_last == TOY_N_FIT_FRAME_ROWS and w.ctrl.shape[1] == len(flowprice.CTRL_NAMES)
    assert len(w.y) == TOY_N_FIT_WINDOW_ROWS and w.n_dropped_last == TOY_N_DROPPED_LAST


def test_the_window_table_holds_only_the_asked_roles(toy_s1, toy_s2):
    assert set(flowprice.window_table(toy_s1, toy_s2).match) <= TOY_FIT_MATCHES
    assert set(flowprice.window_table(toy_s1, toy_s2, roles=('STOP',)).match) == TOY_STOP_MATCHES
    assert set(flowprice.window_table(toy_s1, toy_s2, roles=('DEV',)).match) == TOY_DEV_MATCHES
    assert len(flowprice.window_table(toy_s1, toy_s2, roles=('FIT', 'STOP', 'DEV')).y) == 20 + 6 + 7


def test_the_last_frame_row_without_a_next_regular_frame_is_dropped_and_counted(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    assert w.n_dropped_last == 2
    assert (w.match == 0).sum() == 9 and w.k[w.match == 0].tolist() == list(range(2, 11))      # row k = 11 is the last regular frame
    assert (w.match == 1).sum() == 8 and w.k[w.match == 1].tolist() == list(range(2, 10))      # its game end leaves room for k = 9
    assert (w.match == 4).sum() == 0                                                            # two frames: no frame row at all
    assert w.k[w.match == 5].tolist() == [2, 3, 4]
    assert flowprice.window_table(toy_s1, toy_s2, roles=('STOP',)).n_dropped_last == 1
    assert flowprice.window_table(toy_s1, toy_s2, roles=('FIT', 'STOP', 'DEV')).n_dropped_last == 4
    assert not hasattr(w, 'censored')                                                           # R5: nothing is carried past the last frame


def test_objectives_in_no_kept_window_are_counted_by_group(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    assert w.n_obj_unwindowed.tolist() == list(TOY_OBJ_UNWINDOWED) and w.n_obj_unwindowed.shape == (12,)
    assert w.n_unknown == TOY_N_UNKNOWN


def test_window_y_treat_and_co_are_blue_minus_red_of_the_minute_after_the_frame(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    first = np.flatnonzero(w.match == 0)
    assert w.row[first].tolist() == list(range(9)) and w.k[first].tolist() == list(range(2, 11))
    for k, (y, treat, (gold, count, assists)) in TOY_WINDOW_M0.items():
        i = first[k - 2]
        assert w.y[i] == y, k
        assert w.treat[i].tolist() == [treat.get(g, 0) for g in GROUPS], k
        assert w.co[i].tolist() == [gold, count, assists], k


def test_window_controls_are_lead_phase_momentum_kills_summary_differences_and_buffs(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    for row, want in TOY_CTRL_ROWS_M0.items():
        assert w.ctrl[row].tolist() == [float(v) for v in want], (row, dict(zip(CTRL, w.ctrl[row])))


def test_controls_use_the_frame_clock_views_only(toy_s1, toy_s2):
    views_src = np.asarray(toy_s2.array('views/src'))
    assert (views_src > 0).any() and (np.asarray(toy_s2.array('x/summary'))[views_src > 0] == JUNK).all()      # the toy has event views
    w = flowprice.window_table(toy_s1, toy_s2)
    summary = w.ctrl[:, flowprice.CTRL_NAMES.index('towers_outer'):]
    assert summary.shape[1] == 15 and not (np.abs(summary) > 100).any()                         # no value of an event view got in
    # the event views reuse the numbers 0, 1, 2, ... as src_index, which are frame-row numbers too: rows 0 and 1 are still the rows of
    # the frame-clock views
    assert w.ctrl[0].tolist() == [float(v) for v in TOY_CTRL_ROWS_M0[0]]


def test_the_window_table_has_the_documented_types_and_order(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    n = TOY_N_FIT_WINDOW_ROWS
    assert (w.match.dtype, w.row.dtype, w.k.dtype) == (np.int32, np.int64, np.int32)
    assert w.y.shape == (n,) and w.treat.shape == (n, 12) and w.co.shape == (n, 3) and w.ctrl.shape == (n, 33)
    assert all(a.dtype == np.float64 for a in (w.y, w.treat, w.co, w.ctrl)) and np.isfinite(w.ctrl).all()
    assert (np.diff(w.row) > 0).all() and (np.diff(w.match) >= 0).all() and w.n_matches == 4
    assert np.asarray(toy_s1.array('frame_rows/match'))[w.row].tolist() == w.match.tolist()


def test_a_window_row_is_one_phase_and_one_indicator(toy_s1, toy_s2):
    w = flowprice.window_table(toy_s1, toy_s2)
    phase = w.ctrl[:, 8:16]
    assert (phase.sum(axis=1) == 1).all() and set(np.unique(phase)) == {0.0, 1.0}
    lead = w.ctrl[:, :8].sum(axis=1)                                          # lead x phase: only the phase of the row is non-zero
    for i in range(len(w.y)):
        _, series = flowprice.lead_series(toy_s1, int(w.match[i]))
        assert lead[i] == series[w.k[i]] and w.ctrl[i, 16] == series[w.k[i]] - series[w.k[i] - 1]


def test_a_selection_without_events_or_window_rows_gives_empty_tables_of_the_right_types(tmp_path):
    quiet = ToyMatch('FIT', (0, 20, 40, 60, 80, 100), 420_000)                              # no objective, no kill
    short = ToyMatch('FIT', (0, 50), 100_000)                                               # two frames: no frame row
    only_dev = ToyMatch('DEV', (0, 20, 40, 60, 80, 100), 420_000)
    s1 = build_toy_s1(tmp_path / 's1', (quiet, short, only_dev))
    t = flowprice.event_table(s1)                                                           # FIT: two matches, no objective at all
    assert (len(t.t), t.n_matches, t.n_unknown) == (0, 2, 0)
    assert (t.y.shape, t.co.shape) == ((0, 7), (0, 7, 13)) and (t.match.dtype, t.group.dtype, t.k.dtype) == (np.int32, np.int8, np.int32)
    assert t.counts_by_group()['events'].tolist() == [0] * 12
    gone = flowprice.event_table(s1, roles=('STOP',))                                       # no STOP match at all
    assert (len(gone.t), gone.n_matches) == (0, 0)
    w = flowprice.window_table(s1, build_toy_s2(tmp_path / 's2', s1))
    assert len(w.y) == 3 and w.n_dropped_last == 1 and w.n_matches == 2 and not w.treat.any() and not w.co.any()   # the quiet match only
    none = flowprice.window_table(build_toy_s1(tmp_path / 's1b', (short,)), None)           # nothing to read from S2: it may be absent
    assert len(none.y) == 0 and none.n_dropped_last == 0 and none.n_matches == 1
    assert (none.treat.shape, none.co.shape, none.ctrl.shape) == ((0, 12), (0, 3), (0, 33))
    assert (none.match.dtype, none.row.dtype, none.k.dtype) == (np.int32, np.int64, np.int32)
    assert none.n_obj_unwindowed.tolist() == [0] * 12


# ---- the S2 side ------------------------------------------------------------------------------------------------------------------------
def test_a_frame_without_a_frame_clock_view_is_refused_and_counted(tmp_path):
    s1 = build_toy_s1(tmp_path / 's1')
    s2 = build_toy_s2(tmp_path / 's2', s1)
    src = np.load(s2.path('views/src'), mmap_mode='r+')
    src[3] = 1                                                                # the view of frame row 3 is now an event view
    src.flush()
    del src
    with pytest.raises(ValueError, match=r'no frame-clock view for 1 of the 20 window rows') as raised:
        flowprice.window_table(s1, Store(tmp_path / 's2'))
    message_of(raised)
    assert len(flowprice.window_table(s1, Store(tmp_path / 's2'), roles=('DEV',)).y) == 7      # other matches do not need that view


@pytest.mark.parametrize('array, value', [('views/match', 3), ('views/ref', 1), ('views/frame_k', 9)])
def test_an_s2_store_of_another_s1_is_refused(tmp_path, array, value):
    s1 = build_toy_s1(tmp_path / 's1')
    s2 = build_toy_s2(tmp_path / 's2', s1)
    a = np.load(s2.path(array), mmap_mode='r+')
    a[2] = value
    a.flush()
    del a
    with pytest.raises(ValueError, match=f'{array} differs from the S1 frame rows in 1 of the 20 window rows') as raised:
        flowprice.window_table(s1, Store(tmp_path / 's2'))
    message_of(raised)


def test_two_frame_clock_views_for_one_row_are_refused(tmp_path):
    s1 = build_toy_s1(tmp_path / 's1')
    s2 = build_toy_s2(tmp_path / 's2', s1)
    a = np.load(s2.path('views/src_index'), mmap_mode='r+')
    a[4] = 3
    a.flush()
    del a
    with pytest.raises(ValueError, match='more than one') as raised:
        flowprice.window_table(s1, Store(tmp_path / 's2'))
    message_of(raised)


def test_a_summary_that_is_not_finite_is_refused(tmp_path):
    from c_ept.preprocessing import tokens
    s1 = build_toy_s1(tmp_path / 's1')
    s2 = build_toy_s2(tmp_path / 's2', s1)
    a = np.load(s2.path('x/summary'), mmap_mode='r+')
    a[5, tokens.SUMMARY.index('red_barons')] = np.nan
    a.flush()
    del a
    with pytest.raises(ValueError, match=r'x/summary holds 1 cells that are not finite') as raised:
        flowprice.window_table(s1, Store(tmp_path / 's2'))
    message_of(raised)


def test_the_s2_store_may_cover_more_than_the_window_rows(toy_s1, toy_s2):
    all_roles = flowprice.window_table(toy_s1, toy_s2, roles=('FIT', 'STOP', 'DEV'))
    fit = flowprice.window_table(toy_s1, toy_s2)
    keep = np.isin(all_roles.match, sorted(TOY_FIT_MATCHES))
    assert np.array_equal(all_roles.ctrl[keep], fit.ctrl) and np.array_equal(all_roles.y[keep], fit.y)


# ---- the arrays the module may read -----------------------------------------------------------------------------------------------------
S1_ARRAYS = {'matches/frame_off', 'matches/kill_off', 'matches/obj_off', 'matches/row_off', 'matches/role', 'matches/team',
             'frames/ts', 'frames/num', 'objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker', 'kills/t', 'kills/bounty',
             'kills/shutdown', 'kills/killer', 'kills/amask', 'frame_rows/match', 'frame_rows/k', 'frame_rows/t0'}
S2_ARRAYS = {'views/src', 'views/src_index', 'views/match', 'views/ref', 'views/frame_k', 'x/summary'}


def test_the_tables_read_only_the_documented_arrays(toy_s1, toy_s2, monkeypatch):
    seen = []
    real = Store.array

    def spy(self, name):
        seen.append((self.root.name, name))
        return real(self, name)
    monkeypatch.setattr(Store, 'array', spy)
    flowprice.event_table(toy_s1)
    assert {name for _, name in seen} <= S1_ARRAYS and not any(root == toy_s2.root.name for root, _ in seen)
    del seen[:]
    flowprice.window_table(toy_s1, toy_s2)
    s1_names = {name for root, name in seen if root == toy_s1.root.name}
    s2_names = {name for root, name in seen if root == toy_s2.root.name}
    assert s1_names <= S1_ARRAYS and s2_names == S2_ARRAYS
    assert not any(n.startswith(('units/', 'onsets/', 'event_rows/')) or n == 'matches/key' for _, n in seen)


# ---- a second implementation -----------------------------------------------------------------------------------------------------------
def side_of(spec, pid):
    return 1 if spec.team[pid - 1] == 100 else -1


def known_objectives(spec):
    """([(t, group index, +1 blue / -1 red)], number of objectives left out) of a toy match."""
    out, unknown = [], 0
    for t, name, taker in spec.objectives:
        if name in GROUPS and taker in (0, 1):
            out.append((t, GROUPS.index(name), 1 if taker == 0 else -1))
        else:
            unknown += 1
    return out, unknown


def champion_kills(spec):
    """[(t, +1 blue / -1 red, bounty + shutdown, number of assisters 1-10)] of the kills with a champion killer."""
    return [(t, side_of(spec, killer), bounty + shutdown, len({a for a in assisters if 1 <= a <= 10}))
            for t, killer, bounty, shutdown, assisters in spec.kills if 1 <= killer <= 10]


def event_minute(ts, t):
    """The smallest k with t <= ts[k], and the number of frames when there is none."""
    return next((k for k, f in enumerate(ts) if t <= f), len(ts))


def oracle_events(specs, roles):
    rows, unknown, n_matches = [], 0, 0
    for m, spec in enumerate(specs):
        if spec.role not in roles:
            continue
        n_matches += 1
        ts, lead, F = spec.ts, spec.lead, len(spec.lead)
        objs, n_unknown = known_objectives(spec)
        unknown += n_unknown
        kills = champion_kills(spec)
        for i, (t, g, s) in enumerate(objs):
            k = event_minute(ts, t)
            complete = k - 5 >= 0 and k + 2 <= F - 1
            y, co = np.full(7, np.nan), np.full((7, 13), np.nan)                       # an incomplete window: nothing is made up
            if complete:
                co = np.zeros((7, 13))
                for ri, r in enumerate(range(-4, 3)):
                    j = k + r
                    lo, hi = ts[j - 1], ts[j]                                                  # the window of relative minute r: (lo, hi]
                    y[ri] = s * (lead[j] - lead[j - 1])
                    co[ri, 0] = s * sum(sk * gold for kt, sk, gold, _ in kills if lo < kt <= hi)
                    for i2, (t2, g2, s2) in enumerate(objs):
                        if i2 != i and lo < t2 <= hi:
                            co[ri, 1 + g2] += s * s2
            rows.append(dict(match=m, group=g, taker=0 if s == 1 else 1, t=t, k=k, complete=complete, y=y, co=co))
    return rows, unknown, n_matches


def oracle_summary_ctrl(row):
    v = summary_values(row)
    d = lambda name: v['blue_' + name] - v['red_' + name]                                      # noqa: E731
    plates = sum(d(p) for p in ('plates_top', 'plates_mid', 'plates_bot'))
    on = lambda name: float(v[name] > 0)                                                       # noqa: E731
    return [d('towers_outer'), d('towers_inner'), d('towers_base'), d('towers_nexus'), d('inhibs_down'), plates, d('dragons'), d('elders'),
            d('barons'), d('heralds'), d('atakhans'), d('voidgrubs'), d('has_soul'),
            on('blue_baron_buff_s') - on('red_baron_buff_s'), on('blue_elder_buff_s') - on('red_elder_buff_s')]


def oracle_windows(specs, roles):
    rows = dict(match=[], row=[], k=[], y=[], treat=[], co=[], ctrl=[])
    dropped, unknown, n_matches = 0, 0, 0
    unwindowed = np.zeros(12, dtype=np.int64)
    first_row = 0
    for m, spec in enumerate(specs):
        ts, lead, F = spec.ts, spec.lead, len(spec.lead)
        frame_rows = [k for k in range(2, F) if ts[k] < spec.T_ms - 30_000]
        if spec.role in roles:
            n_matches += 1
            objs, n_unknown = known_objectives(spec)
            unknown += n_unknown
            kills = champion_kills(spec)
            kept = []
            for pos, k in enumerate(frame_rows):
                if k + 1 > F - 1:
                    dropped += 1
                    continue
                kept.append(k)
                lo, hi = ts[k], ts[k + 1]
                treat = np.zeros(12)
                for t, g, s in objs:
                    if lo < t <= hi:
                        treat[g] += s
                in_window = [(sk, gold, a) for kt, sk, gold, a in kills if lo < kt <= hi]
                phase = max([i for i, e in enumerate(EDGES) if ts[k] / 1000 >= e], default=0)
                ctrl = np.zeros(33)
                ctrl[phase], ctrl[8 + phase] = lead[k], 1
                ctrl[16] = lead[k] - lead[k - 1]
                ctrl[17] = sum(sk for kt, sk, _, _ in kills if kt <= ts[k])
                ctrl[18:33] = oracle_summary_ctrl(first_row + pos)
                for key, value in (('match', m), ('row', first_row + pos), ('k', k), ('y', lead[k + 1] - lead[k]), ('treat', treat),
                                   ('co', [sum(sk * gold for sk, gold, _ in in_window), sum(sk for sk, _, _ in in_window),
                                           sum(sk * a for sk, _, a in in_window)]), ('ctrl', ctrl)):
                    rows[key].append(value)
            for t, g, s in objs:
                if event_minute(ts, t) - 1 not in kept:                                        # row k holds the minute k + 1
                    unwindowed[g] += 1
        first_row += len(frame_rows)
    return rows, dropped, unwindowed, unknown, n_matches


def assert_events_equal_oracle(table, specs, roles):
    rows, unknown, n_matches = oracle_events(specs, roles)
    assert len(table.t) == len(rows) and table.n_unknown == unknown and table.n_matches == n_matches
    for field in ('match', 'group', 'taker', 't', 'k', 'complete'):
        assert getattr(table, field).tolist() == [r[field] for r in rows], field
    y = np.array([r['y'] for r in rows]).reshape(-1, 7)
    co = np.array([r['co'] for r in rows]).reshape(-1, 7, 13)
    assert np.array_equal(table.y, y, equal_nan=True) and np.array_equal(table.co, co, equal_nan=True)


def assert_windows_equal_oracle(table, specs, roles):
    rows, dropped, unwindowed, unknown, n_matches = oracle_windows(specs, roles)
    assert table.n_dropped_last == dropped and table.n_unknown == unknown and table.n_matches == n_matches
    assert table.n_obj_unwindowed.tolist() == unwindowed.tolist()
    for field in ('match', 'row', 'k'):
        assert getattr(table, field).tolist() == rows[field], field
    n = len(rows['y'])
    assert np.array_equal(table.y, np.array(rows['y'], dtype=float))
    assert np.array_equal(table.treat, np.array(rows['treat']).reshape(n, 12))
    assert np.array_equal(table.co, np.array(rows['co'], dtype=float).reshape(n, 3))
    assert np.array_equal(table.ctrl, np.array(rows['ctrl']).reshape(n, 33))


@pytest.mark.parametrize('roles', [('FIT',), ('STOP',), ('DEV',), ('FIT', 'STOP', 'DEV')])
def test_the_toy_tables_equal_a_brute_force_implementation(toy_s1, toy_s2, roles):
    assert_events_equal_oracle(flowprice.event_table(toy_s1, roles=roles), TOY_MATCHES, roles)
    assert_windows_equal_oracle(flowprice.window_table(toy_s1, toy_s2, roles=roles), TOY_MATCHES, roles)


def random_specs(seed, n_matches=5):
    """Random toy matches: frame counts from 0 to 14 (the short ones have no complete window), objectives and kills of every kind,
    some exactly on a frame time or one ms after it, some unknown, some executions, interleaved teams, a random walk of the lead."""
    rng = np.random.default_rng(seed)
    names = list(KIND_SUB)
    specs = []
    for m in range(n_matches):
        F = int(rng.choice([0, 1, 2, 3, 4, 6, 8, 9, 11, 14, 14, 18]))
        ts = frame_times(F)
        lead = tuple(int(v) for v in 5 * np.cumsum(rng.integers(-40, 60, F)))
        T = (ts[-1] if F else 0) + int(rng.integers(1_000, 150_000))

        def draw_time():
            if F and rng.random() < 0.3:                                                     # on a frame time, one ms before or after it
                return int(ts[int(rng.integers(0, F))] + int(rng.choice([0, 0, 1, -1])))
            if F >= 8 and rng.random() < 0.7:                                                # in the middle, where windows are complete
                return int(rng.integers(ts[3], ts[F - 2] + 1))
            return int(rng.integers(0, T + 1))

        objectives = sorted((obj(max(0, min(T, draw_time())), str(rng.choice(names)), int(rng.choice([0, 1, 0, 1, 0, 1, NOBODY]))))
                            for _ in range(int(rng.integers(0, 26))))
        kills = sorted(kill(max(0, min(T, draw_time())), int(rng.choice([0, 0] + list(range(1, 11)))), int(rng.choice([0, 150, 300, 450])),
                            int(rng.choice([0, 0, 100, 300])), rng.choice(range(1, 11), int(rng.integers(0, 5)), replace=False).tolist())
                       for _ in range(int(rng.integers(0, 41))))
        team = tuple(rng.permutation(BLUE5 + RED5).tolist()) if rng.random() < 0.3 else BLUE5 + RED5
        specs.append(ToyMatch(('FIT', 'STOP', 'DEV', 'FIT')[m % 4], lead, T, tuple(objectives), tuple(kills), team))
    return tuple(specs)


@pytest.mark.parametrize('seed', range(16))
def test_random_toy_stores_equal_a_brute_force_implementation(tmp_path, seed):
    specs = random_specs(seed)
    s1 = build_toy_s1(tmp_path / 's1', specs)
    s2 = build_toy_s2(tmp_path / 's2', s1)
    for roles in (('FIT',), ('STOP', 'DEV'), ('FIT', 'STOP', 'DEV')):
        assert_events_equal_oracle(flowprice.event_table(s1, roles=roles), specs, roles)
        assert_windows_equal_oracle(flowprice.window_table(s1, s2, roles=roles), specs, roles)


def test_the_random_matches_reach_the_cases_they_are_there_for():
    """The generator is not vacuous: over the seeds it makes complete and incomplete events, frame-time ties and dropped last rows."""
    complete = incomplete = dropped = unknown = ties = other_objectives = other_kills = 0
    for seed in range(16):
        specs = random_specs(seed)
        rows, n_unknown, _ = oracle_events(specs, ('FIT', 'STOP', 'DEV'))
        complete += sum(r['complete'] for r in rows)
        incomplete += sum(not r['complete'] for r in rows)
        unknown += n_unknown
        dropped += oracle_windows(specs, ('FIT', 'STOP', 'DEV'))[1]
        ties += sum(1 for r in rows if r['t'] in specs[r['match']].ts)
        other_objectives += sum(1 for r in rows if r['complete'] and r['co'][:, 1:].any())
        other_kills += sum(1 for r in rows if r['complete'] and r['co'][:, 0].any())
    assert complete > 100 and incomplete > 100 and dropped > 10 and unknown > 20 and ties > 10
    assert other_objectives > 30 and other_kills > 30                    # the controls of the events are not all zero


# ---- the real S1 extraction -------------------------------------------------------------------------------------------------------------
def test_the_toy_s1_equals_the_real_s1_extraction_of_the_same_matches(tmp_path):
    from c_ept.workflows import run_s1
    specs = tuple(TOY_MATCHES[i] for i in (0, 1, 3, 4, 5))                                      # the standard team layout
    matches = [('15.18', f'SYN_{i:04d}', spec.role) for i, spec in enumerate(specs)]
    records = {key: toy_record(spec, key) for (_, key, _), spec in zip(matches, specs)}
    run_s1.build_store(matches, tmp_path / 's1', tmp_path / 'blocks', workers=1, records=lambda key, patch: records[key],
                       role_of=synth.roles_of(matches))
    real = run_s1.load_s1(tmp_path / 's1')
    toy = build_toy_s1(tmp_path / 'toy', specs)
    gold = C.FRAME_FIELDS.index('totalGold')
    for name in sorted(S1_ARRAYS):
        a, b = np.asarray(real.array(name)), np.asarray(toy.array(name))
        if name == 'frames/num':
            a, b = a[:, :, gold], b[:, :, gold]
        assert a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b), name        # the toy encodings are S1's
    toy_views = build_toy_s2(tmp_path / 'toy_s2', toy)
    real_views = build_toy_s2(tmp_path / 'real_s2', real)
    for roles in (('FIT',), ('FIT', 'STOP', 'DEV')):
        te, re_ = flowprice.event_table(toy, roles=roles), flowprice.event_table(real, roles=roles)
        for field in ('match', 'group', 'taker', 't', 'k', 'complete', 'y', 'co'):
            assert np.array_equal(getattr(te, field), getattr(re_, field), equal_nan=True), field
        assert (te.n_unknown, te.n_matches) == (re_.n_unknown, re_.n_matches)
        tw, rw = flowprice.window_table(toy, toy_views, roles=roles), flowprice.window_table(real, real_views, roles=roles)
        for field in ('match', 'row', 'k', 'y', 'treat', 'co', 'ctrl', 'n_obj_unwindowed'):
            assert np.array_equal(getattr(tw, field), getattr(rw, field)), field
        assert (tw.n_dropped_last, tw.n_unknown) == (rw.n_dropped_last, rw.n_unknown)


# ---- the numbers of note 03 --------------------------------------------------------------------------------------------------------------
def fe_beta(y):
    """OLS of y[e, r] on event dummies and relative-minute dummies with r = -1 left out (beta_{-1} = 0): note 03 section A."""
    n_e, n_r = y.shape
    rows = [np.r_[np.eye(n_e)[e], np.delete(np.eye(n_r)[j], 3)] for e in range(n_e) for j in range(n_r)]
    coef = np.linalg.lstsq(np.array(rows), y.ravel(), rcond=None)[0]
    return np.insert(coef[n_e:], 3, 0.0)


def test_the_toy_numbers_of_note_03_recompute_from_their_inputs():
    # A. flow price and pre-trend of two events
    beta = fe_beta(np.array(TOY_EVENT_Y, dtype=float))
    assert beta.tolist() == pytest.approx(list(TOY_BETA7))
    assert beta[4:].sum() - 3 * beta[:4].mean() == pytest.approx(TOY_PRICE)
    assert (beta[2] + beta[3]) - (beta[0] + beta[1]) == pytest.approx(TOY_PRETREND)
    # B. the slope of the window regression
    design = np.c_[np.ones(6), np.array(TOY_WINDOW_TREAT, dtype=float)]
    assert np.linalg.lstsq(design, np.array(TOY_WINDOW_Y, dtype=float), rcond=None)[0][1] == pytest.approx(TOY_IMMEDIATE)
    # C. Poisson NLL with exposure, log y! included, an unobserved cell adds nothing
    mu = np.array(TOY_RATE) * np.array(TOY_EXP)
    nll = sum(mu[i] - TOY_Y[i] * math.log(mu[i]) + math.lgamma(TOY_Y[i] + 1) for i in range(4) if TOY_EXP[i] > 0)
    assert nll == pytest.approx(TOY_POISSON_NLL, abs=1e-6)
    # D. gold bins and bin means
    gold = np.array(TOY_GOLD, dtype=float)
    bins = np.minimum(gold // 100, 30).astype(int)
    assert bins.tolist() == list(TOY_GOLD_BINS)
    assert gold[bins == 3].mean() == pytest.approx(TOY_GOLD_BIN3_MEAN) and gold[bins == 30].mean() == pytest.approx(TOY_GOLD_OVERFLOW_MEAN)
    # E. one composed row (onsets 0.01 per second, marks 0.6 / 0.4, f0(K, O) of the first-kill team +-220, towers 0.002 per second at 400)

    def expected_v(T):
        hi = min(TOY_T0_S + 180.0, T - 30.0)
        exposure = np.array([min(max(hi - (TOY_T0_S + 10.0 * q), 0.0), 10.0) for q in range(6)])
        delta_hat = TOY_T0_S + (np.arange(6) + 0.5) * 10.0 + 20.0
        towers = 0.002 * np.maximum(0.0, np.minimum(60.0, T - delta_hat)) * 400.0
        return float((0.01 * exposure * (0.6 * (220.0 + towers) + 0.4 * (-220.0 - towers))).sum())
    assert expected_v(TOY_T_END_FAR_S) == pytest.approx(TOY_EXPECTED_V60)
    assert expected_v(TOY_T_END_CUT_S) == pytest.approx(TOY_EXPECTED_V60_CUT)
    # F. calibration: slope, mean bias and the weighted slope
    e, v, w = (np.array(a, dtype=float) for a in (TOY_CAL_E, TOY_CAL_V, TOY_CAL_W))
    assert np.polyfit(e, v, 1)[0] == pytest.approx(TOY_CAL_SLOPE) and (v - e).mean() == pytest.approx(TOY_CAL_BIAS)
    ew, vw = np.average(e, weights=w), np.average(v, weights=w)
    assert np.sum(w * (e - ew) * (v - vw)) / np.sum(w * (e - ew) ** 2) == pytest.approx(TOY_CAL_WSLOPE)
    # G. Pearson dispersion at mu = 1
    assert np.mean((np.array(TOY_PHI_Y, dtype=float) - 1.0) ** 2) == pytest.approx(TOY_PHI)


def test_the_hand_worked_toy_counts_agree_with_the_toy_matches():
    """The constants of the toy matches are checked against the matches themselves, so that a changed match cannot leave stale numbers."""
    assert [len(sp.lead) for sp in TOY_MATCHES] == list(TOY_N_FRAMES)
    assert {m for m, sp in enumerate(TOY_MATCHES) if sp.role == 'FIT'} == TOY_FIT_MATCHES
    assert TOY_MATCHES[0].lead == TOY_LEAD_M0 and TOY_MATCHES[0].ts == TOY_TS_M0
    rows, unknown, _ = oracle_events(TOY_MATCHES, ('FIT',))
    assert (len(rows), unknown, sum(not r['complete'] for r in rows)) == (TOY_N_EVENTS, TOY_N_UNKNOWN, TOY_N_INCOMPLETE)
    assert np.bincount([r['group'] for r in rows], minlength=12).tolist() == list(TOY_EVENTS_BY_GROUP)
    assert np.bincount([r['group'] for r in rows if not r['complete']], minlength=12).tolist() == list(TOY_INCOMPLETE_BY_GROUP)
    _, dropped, unwindowed, _, _ = oracle_windows(TOY_MATCHES, ('FIT',))
    assert dropped == TOY_N_DROPPED_LAST and unwindowed.tolist() == list(TOY_OBJ_UNWINDOWED)
    assert sum(len([k for k in range(2, len(sp.lead)) if sp.ts[k] < sp.T_ms - 30_000]) for sp in TOY_MATCHES if sp.role == 'FIT') == (
        TOY_N_FIT_FRAME_ROWS)
    assert len(SUMMARY_PAIRS) == 15


# =====================================================================================================================================
# the estimators (stage S3 Task 2)
# =====================================================================================================================================
def solved(table, group='baron'):
    return flowprice.solve(flowprice.flow_stats(table, group))


def subset_events(table, keep):
    fields = ('match', 'group', 'taker', 't', 'k', 'complete', 'y', 'co')
    return dataclasses.replace(table, **{f: getattr(table, f)[keep] for f in fields})


def design_stats(columns, y, match):
    return flowprice.stats_from_design(np.asarray(columns, dtype=float), np.asarray(y, dtype=float), np.asarray(match))


# ---- the formulas of the price and of the pre-trend --------------------------------------------------------------------------------------
def test_price_and_pretrend_formula():                            # note 03 section A
    assert flowprice.price_from_beta(TOY_BETA7) == pytest.approx(TOY_PRICE)
    assert flowprice.pretrend_from_beta(TOY_BETA7) == pytest.approx(TOY_PRETREND)


def test_price_and_pretrend_do_not_depend_on_the_level_of_the_reference_minute():
    """Both are contrasts (their weights add up to 0), so the level of beta_{-1} can be anything: TRUE_BETA still has 30 in it."""
    assert TRUE_BETA[3] != 0
    assert flowprice.price_from_beta(TRUE_BETA) == pytest.approx(TOY_PRICE)
    assert flowprice.pretrend_from_beta(TRUE_BETA) == pytest.approx(TOY_PRETREND)
    assert flowprice.price_from_beta(np.asarray(TOY_BETA7) + 1234.5) == pytest.approx(TOY_PRICE)


def test_price_and_pretrend_work_on_a_stack_and_pass_nan_on():
    stack = np.array([TOY_BETA7, TOY_BETA7_TRY, [np.nan] * 7], dtype=float)
    price, trend = flowprice.price_from_beta(stack), flowprice.pretrend_from_beta(stack)
    assert price.shape == trend.shape == (3,)
    assert price[0] == pytest.approx(TOY_PRICE) and trend[0] == pytest.approx(TOY_PRETREND)
    assert trend[1] == pytest.approx(TOY_PRETREND_TRY) and price[1] == pytest.approx((670 + 120 - 50) + 3 * 65)      # 740 + 195
    assert np.isnan([price[2], trend[2]]).all()
    for bad in (np.zeros(6), np.zeros((4, 8))):
        with pytest.raises(ValueError, match='7'):
            flowprice.price_from_beta(bad)
        with pytest.raises(ValueError, match='7'):
            flowprice.pretrend_from_beta(bad)


def test_flow_beta_puts_the_zero_of_the_reference_minute_in_the_fourth_place():
    theta = np.arange(1.0, 20.0)
    assert flowprice.flow_beta(theta).tolist() == [1, 2, 3, 0, 4, 5, 6]
    stack = flowprice.flow_beta(np.stack([theta, 2 * theta]))
    assert stack.shape == (2, 7) and stack[1].tolist() == [2, 4, 6, 0, 8, 10, 12]
    with pytest.raises(ValueError, match='6'):
        flowprice.flow_beta(np.zeros(5))


def test_the_design_columns_are_named_and_counted():
    assert len(flowprice.FLOW_COLUMNS) == 19 and flowprice.FLOW_COLUMNS[:6] == ('r-4', 'r-3', 'r-2', 'r+0', 'r+1', 'r+2')
    assert flowprice.FLOW_COLUMNS[6:] == tuple('co_' + n for n in CO)
    assert len(flowprice.WINDOW_COLUMNS) == 48
    assert flowprice.WINDOW_COLUMNS[:12] == tuple('treat_' + g for g in GROUPS)
    assert flowprice.WINDOW_COLUMNS[12:15] == ('co_kill_gold', 'co_kill_count', 'co_assist_count')
    assert flowprice.WINDOW_COLUMNS[15:] == tuple(CTRL)


# ---- the within-event regression of the flow price ---------------------------------------------------------------------------------------
def test_recovers_known_flow_betas_with_event_effects():        # the level of every event is different and beta is known
    t = synthetic_events(beta=TRUE_BETA, n=3000, event_sd=500.0, seed=1)
    beta = flowprice.flow_beta(flowprice.solve(flowprice.flow_stats(t, 'baron')))
    assert np.allclose(beta, TRUE_BETA - TRUE_BETA[3], atol=5.0)


def test_the_synthetic_events_would_expose_an_estimator_without_the_controls():
    """The recovery test above is not vacuous: without the 13 controls beta_0 would be too high by the kill gold of the event minute."""
    t = synthetic_events(beta=TRUE_BETA, n=3000, event_sd=500.0, seed=1)
    naive = (t.y - t.y[:, [3]]).mean(axis=0)                      # the change from r = -1 in each event, which takes the event level out
    assert naive[4] - (TRUE_BETA[4] - TRUE_BETA[3]) > 200.0


def test_event_level_shifts_do_not_change_beta():
    t = synthetic_events(beta=TRUE_BETA, n=500, seed=2)
    a = flowprice.solve(flowprice.flow_stats(t, 'baron'))
    b = flowprice.solve(flowprice.flow_stats(with_event_level_shift(t, seed=3), 'baron'))
    assert np.isfinite(a).all() and np.allclose(a, b)


def test_the_flow_estimator_reproduces_the_hand_numbers_of_note_03():
    beta = flowprice.flow_beta(solved(event_table_from_y(TOY_EVENT_Y)))
    assert beta.tolist() == pytest.approx(list(TOY_BETA7))
    assert flowprice.price_from_beta(beta) == pytest.approx(TOY_PRICE)
    assert flowprice.pretrend_from_beta(beta) == pytest.approx(TOY_PRETREND)
    beta = flowprice.flow_beta(solved(event_table_from_y(TOY_EVENT_Y_TRY)))                    # "try it yourself"
    assert beta.tolist() == pytest.approx(list(TOY_BETA7_TRY))
    assert flowprice.pretrend_from_beta(beta) == pytest.approx(TOY_PRETREND_TRY)


def test_flow_stats_equal_the_explicit_event_dummy_regression():
    """A second implementation: one dummy per event, one per relative minute (r = -1 left out), the 13 controls, and lstsq."""
    n = 40
    t = synthetic_events(TRUE_BETA, n=n, event_sd=300.0, seed=11)
    events = np.kron(np.eye(n), np.ones((7, 1)))
    minutes = np.tile(np.delete(np.eye(7), 3, axis=1), (n, 1))
    coef = np.linalg.lstsq(np.column_stack([events, minutes, t.co.reshape(-1, 13)]), t.y.reshape(-1), rcond=None)[0]
    assert np.allclose(solved(t), coef[n:], rtol=1e-7, atol=1e-7)


def test_incomplete_events_do_not_enter_the_flow_statistics():
    t = synthetic_events(TRUE_BETA, n=300, event_sd=200.0, seed=21, n_incomplete=40)
    assert np.isnan(t.y).any() and np.isnan(t.co).any() and (~t.complete).sum() == 40         # the table does hold the NaN rows
    s = flowprice.flow_stats(t, 'baron')
    assert s.n_rows.sum() == 300 and np.isfinite(s.xtx).all() and np.isfinite(s.xty).all()
    clean = flowprice.flow_stats(subset_events(t, t.complete), 'baron')
    assert np.array_equal(s.xtx, clean.xtx) and np.array_equal(s.xty, clean.xty) and np.array_equal(s.match, clean.match)
    assert np.isfinite(flowprice.solve(s)).all()


def test_flow_stats_use_only_the_events_of_the_group_and_name_it_by_name_or_index():
    dragons = synthetic_events(TRUE_BETA, 60, seed=1, group='dragon')
    barons = synthetic_events(TRUE_BETA, 80, seed=2, group='baron', n_incomplete=10)
    t = concat_events([dragons, barons])
    d, b = flowprice.flow_stats(t, 'dragon'), flowprice.flow_stats(t, 'baron')
    assert d.n_rows.sum() == 60 and b.n_rows.sum() == 80 and d.columns == b.columns == flowprice.FLOW_COLUMNS
    assert d.xtx.shape == (len(d.match), 19, 19) and d.xty.shape == (len(d.match), 19) and d.match.dtype == np.int64
    assert flowprice.flow_stats(t, GROUPS.index('dragon')).n_rows.sum() == 60
    assert np.array_equal(d.match, np.unique(t.match[(t.group == GROUPS.index('dragon')) & t.complete]))
    for bad in ('nope', 12, -1):
        with pytest.raises(ValueError, match='group'):
            flowprice.flow_stats(t, bad)


def test_flow_stats_accept_a_table_that_is_not_ordered_by_match():
    t = synthetic_events(TRUE_BETA, 200, seed=22, event_sd=100.0)
    order = np.random.default_rng(23).permutation(len(t.match))
    shuffled = subset_events(t, order)
    a, b = flowprice.flow_stats(t, 'baron'), flowprice.flow_stats(shuffled, 'baron')
    assert np.array_equal(a.match, b.match) and np.allclose(a.xtx, b.xtx) and np.allclose(a.xty, b.xty)


def test_a_non_finite_value_in_a_complete_event_is_refused():
    t = synthetic_events(TRUE_BETA, 50, seed=24)
    y = t.y.copy()
    y[3, 2] = np.nan
    with pytest.raises(ValueError, match='not finite'):
        flowprice.flow_stats(dataclasses.replace(t, y=y), 'baron')


def test_a_control_column_without_variation_is_not_estimable_and_the_others_are():
    """The column of the group itself is zero when the group cannot happen twice in a minute (a baron, a dragon, an atakhan): its
    coefficient is NaN, the six beta and the other 12 controls are estimated."""
    t = synthetic_events(TRUE_BETA, n=3000, event_sd=400.0, seed=12, own_group_co=False)
    s = flowprice.flow_stats(t, 'baron')
    theta = flowprice.solve(s)
    own = s.columns.index('co_baron')
    assert np.isnan(theta[own]) and np.isfinite(np.delete(theta, own)).all()
    assert np.allclose(flowprice.flow_beta(theta), TRUE_BETA - TRUE_BETA[3], atol=5.0)
    assert np.allclose(theta[6:][np.arange(13) != 5], TRUE_GAMMA[np.arange(13) != 5], atol=4.0)      # co_baron is the sixth control


def test_a_group_without_complete_events_and_a_singular_system_have_no_estimate():
    t = synthetic_events(n=500, seed=13, group='baron')
    empty = flowprice.flow_stats(t, 'dragon')
    assert len(empty.match) == 0 and empty.xtx.shape == (0, 19, 19) and empty.xty.shape == (0, 19) and empty.n_rows.shape == (0,)
    assert flowprice.solve(empty).shape == (19,) and np.isnan(flowprice.solve(empty)).all()
    boot = flowprice.bootstrap(empty, B=5, seed=1)
    assert boot.shape == (5, 19) and np.isnan(boot).all()
    one = flowprice.flow_stats(synthetic_events(n=1, seed=14), 'baron')                      # one event: rank 6 of 19
    assert np.isnan(flowprice.solve(one)).all()


# ---- the window regression of the immediate value ----------------------------------------------------------------------------------------
def test_sufficient_statistics_equal_direct_ols():
    w = synthetic_windows(n=4000, seed=4)
    s = flowprice.window_stats(w)
    assert np.allclose(np.linalg.solve(s.xtx.sum(0), s.xty.sum(0)), np.linalg.lstsq(window_design(w), w.y, rcond=None)[0])
    assert np.allclose(flowprice.solve(s), np.linalg.lstsq(window_design(w), w.y, rcond=None)[0], rtol=1e-8, atol=1e-8)
    assert s.columns == flowprice.WINDOW_COLUMNS and s.n_rows.sum() == 4000 and s.xtx.shape == (len(s.match), 48, 48)
    assert np.array_equal(s.match, np.unique(w.match)) and (np.diff(s.match) > 0).all()
    assert np.array_equal(s.n_rows, np.bincount(w.match)[s.match])


def test_window_regression_recovers_immediate_effect():
    w = synthetic_windows(n=20_000, effect={'baron': 1500.0}, seed=5)
    assert flowprice.solve(flowprice.window_stats(w))[flowprice.GROUPS.index('baron')] == pytest.approx(1500.0, abs=30)


def test_window_regression_recovers_every_group_and_the_kill_controls():
    w = synthetic_windows(n=30_000, effect=DEFAULT_WINDOW_EFFECT, seed=25)
    theta = flowprice.solve(flowprice.window_stats(w))
    assert np.allclose(theta[:12], [DEFAULT_WINDOW_EFFECT[g] for g in GROUPS], atol=25.0)
    assert np.allclose(theta[12:15], TRUE_WINDOW_CO, atol=[0.3, 3.0, 2.0])


def test_the_window_estimator_reproduces_the_hand_slopes_of_note_03():
    """Six minutes, an intercept and the baron count: the slope is 1,525 (and 1,575 in the "try it yourself" variant)."""
    design = np.column_stack([np.ones(6), TOY_WINDOW_TREAT])
    theta = flowprice.solve(design_stats(design, TOY_WINDOW_Y, np.arange(6) // 3))
    assert theta.tolist() == pytest.approx([250 / 6, TOY_IMMEDIATE])
    theta = flowprice.solve(design_stats(design, TOY_WINDOW_Y_TRY, np.arange(6) // 3))
    assert theta.tolist() == pytest.approx([25.0, TOY_IMMEDIATE_TRY])


def test_a_statistics_table_is_the_same_whatever_the_grouping_into_matches():
    """The per-match sums add up to the same normal equations whatever the matches are (here one match against six)."""
    rng = np.random.default_rng(26)
    x, y = rng.normal(size=(60, 4)), rng.normal(size=60)
    one, six = design_stats(x, y, np.zeros(60, dtype=int)), design_stats(x, y, np.arange(60) // 10)
    assert len(one.match) == 1 and len(six.match) == 6 and np.allclose(one.xtx.sum(0), six.xtx.sum(0))
    assert np.allclose(flowprice.solve(one), flowprice.solve(six))
    assert np.allclose(flowprice.solve(one), np.linalg.lstsq(x, y, rcond=None)[0])


def test_a_window_row_that_is_not_finite_is_refused_and_an_empty_table_has_no_estimate():
    w = synthetic_windows(n=100, seed=27)
    bad = w.ctrl.copy()
    bad[5, 3] = np.inf
    with pytest.raises(ValueError, match='not finite'):
        flowprice.window_stats(dataclasses.replace(w, ctrl=bad))
    empty = flowprice.window_stats(dataclasses.replace(w, match=w.match[:0], row=w.row[:0], k=w.k[:0], y=w.y[:0], treat=w.treat[:0],
                                                       co=w.co[:0], ctrl=w.ctrl[:0]))
    assert len(empty.match) == 0 and empty.xtx.shape == (0, 48, 48) and np.isnan(flowprice.solve(empty)).all()


def test_window_stats_accept_a_table_that_is_not_ordered_by_match():
    w = synthetic_windows(n=800, seed=34, n_matches=20)
    o = np.random.default_rng(35).permutation(len(w.y))
    shuffled = dataclasses.replace(w, match=w.match[o], row=w.row[o], k=w.k[o], y=w.y[o], treat=w.treat[o], co=w.co[o], ctrl=w.ctrl[o])
    a, b = flowprice.window_stats(w), flowprice.window_stats(shuffled)
    assert np.array_equal(a.match, b.match) and np.array_equal(a.n_rows, b.n_rows)
    assert np.allclose(a.xtx, b.xtx) and np.allclose(a.xty, b.xty)


# ---- solving and weights ----------------------------------------------------------------------------------------------------------------
def test_a_weighted_solve_is_weighted_least_squares_on_the_rows():
    w = synthetic_windows(n=600, seed=15, n_matches=12)
    s = flowprice.window_stats(w)
    weights = np.random.default_rng(16).integers(1, 4, len(s.match)).astype(float)
    weights[0] = 0.0                                                              # a match that is left out
    per_row = weights[np.searchsorted(s.match, w.match)]
    reference = np.linalg.lstsq(np.sqrt(per_row)[:, None] * window_design(w), np.sqrt(per_row) * w.y, rcond=None)[0]
    assert np.allclose(flowprice.solve(s, weights), reference, rtol=1e-8, atol=1e-8)
    assert np.allclose(flowprice.solve(s, np.ones(len(s.match))), flowprice.solve(s))
    assert np.allclose(flowprice.solve(s, None), flowprice.solve(s))


@pytest.mark.parametrize('make', [lambda m: np.ones(m + 1), lambda m: np.ones((m, 1)), lambda m: -np.ones(m), lambda m: np.full(m, np.nan)])
def test_weights_that_do_not_fit_are_refused(make):
    s = flowprice.flow_stats(synthetic_events(n=100, seed=28), 'baron')
    with pytest.raises(ValueError, match='weights'):
        flowprice.solve(s, make(len(s.match)))


def test_solve_is_accurate_when_the_columns_have_very_different_scales():
    """The normal equations are equilibrated: a column of units next to one of 10^6 (the lead times a phase next to an indicator)."""
    rng = np.random.default_rng(29)
    x = rng.normal(size=(2000, 3)) * np.array([1e-3, 1.0, 1e6])
    y = x @ np.array([5e3, -2.0, 3e-4]) + rng.normal(scale=0.005, size=2000)
    theta = flowprice.solve(design_stats(x, y, np.arange(2000) // 20))
    assert np.allclose(theta, [5e3, -2.0, 3e-4], rtol=1e-3)
    assert np.allclose(theta, np.linalg.lstsq(x, y, rcond=None)[0], rtol=1e-5)


def test_a_design_with_highly_correlated_columns_is_still_estimated_and_exactly_collinear_columns_are_not():
    """Two columns that differ by a thousandth of their size are far from collinear for double precision: the fit is made (and is the
    least-squares fit); two equal columns are collinear: nothing is estimated."""
    rng = np.random.default_rng(36)
    x1 = rng.normal(size=2000)
    x = np.column_stack([x1, x1 + 1e-3 * rng.normal(size=2000), rng.normal(size=2000)])
    y = x @ np.array([2.0, 3.0, -1.0]) + 0.01 * rng.normal(size=2000)
    theta = flowprice.solve(design_stats(x, y, np.arange(2000) // 10))
    assert np.isfinite(theta).all() and np.allclose(theta, np.linalg.lstsq(x, y, rcond=None)[0], rtol=1e-5, atol=1e-5)
    twin = np.column_stack([x1, x1, x[:, 2]])
    assert np.isnan(flowprice.solve(design_stats(twin, y, np.arange(2000) // 10))).all()


def test_rank_of_counts_the_columns_that_vary_and_their_rank():
    w = synthetic_windows(n=1500, seed=37)
    assert flowprice.rank_of(flowprice.window_stats(w)) == {'live': 48, 'rank': 48}
    no_baron = dataclasses.replace(w, treat=np.where(np.arange(12) == GROUPS.index('baron'), 0.0, w.treat))
    assert flowprice.rank_of(flowprice.window_stats(no_baron)) == {'live': 47, 'rank': 47}
    x = np.random.default_rng(38).normal(size=(300, 3))
    twin = np.column_stack([x, x[:, 0]])                                            # a fourth column equal to the first
    assert flowprice.rank_of(design_stats(twin, x.sum(axis=1), np.arange(300) // 5)) == {'live': 4, 'rank': 3}
    assert flowprice.rank_of(flowprice.flow_stats(synthetic_events(n=3, seed=39), 'dragon')) == {'live': 0, 'rank': 0}
    one = flowprice.flow_stats(synthetic_events(n=1, seed=14), 'baron')             # one event: its seven rows have rank 6
    assert flowprice.rank_of(one)['rank'] == 6 < flowprice.rank_of(one)['live']
    weights = np.zeros(len(one.match))
    assert flowprice.rank_of(one, weights) == {'live': 0, 'rank': 0}                # every match left out: nothing varies


def test_rank_of_equals_live_exactly_when_solve_can_estimate_the_columns_that_vary():
    rng = np.random.default_rng(40)
    for i in range(30):
        x = rng.normal(size=(120, 4))
        if i % 3 == 0:
            x[:, 3] = 2.0 * x[:, 0] - x[:, 1]                                       # a linear relation among columns that vary
        if i % 5 == 0:
            x[:, 2] = 0.0                                                           # a column that does not vary
        s = design_stats(x, rng.normal(size=120), np.arange(120) // 6)
        rank, theta, live = flowprice.rank_of(s), flowprice.solve(s), x.any(axis=0)
        assert rank['live'] == live.sum() and (rank['rank'] == rank['live']) == np.isfinite(theta[live]).all(), i
        assert np.isnan(theta[~live]).all()


# ---- the match bootstrap -----------------------------------------------------------------------------------------------------------------
def test_bootstrap_is_deterministic():
    s = flowprice.flow_stats(synthetic_events(beta=TRUE_BETA, n=300, seed=7), 'baron')
    a = flowprice.bootstrap(s, B=50, seed=7)
    assert np.array_equal(a, flowprice.bootstrap(s, B=50, seed=7)) and a.shape == (50, 19)
    assert not np.array_equal(a, flowprice.bootstrap(s, B=50, seed=8))
    assert not np.isnan(a[:, :6]).any()


def test_bootstrap_draws_whole_matches_from_the_multinomial_stream_of_default_rng():
    """The replicate b uses the b-th draw of ``default_rng(seed).multinomial(M, 1/M)`` as the weights of the matches (n = M)."""
    s = flowprice.flow_stats(synthetic_events(n=200, seed=17), 'baron')
    m, boot = len(s.match), flowprice.bootstrap(s, B=4, seed=18)
    rng = np.random.default_rng(18)
    for b in range(4):
        weights = rng.multinomial(m, np.full(m, 1.0 / m)).astype(float)
        assert weights.sum() == m
        assert np.allclose(boot[b], flowprice.solve(s, weights), rtol=1e-8, atol=1e-8), b


def test_bootstrap_solves_in_chunks_of_the_memory_budget_and_does_not_depend_on_them(monkeypatch):
    s = flowprice.flow_stats(synthetic_events(n=200, seed=30), 'baron')
    sizes, real = [], flowprice._solve_normal

    def spy(a, b):
        sizes.append(len(b))
        return real(a, b)
    monkeypatch.setattr(flowprice, '_solve_normal', spy)
    whole = flowprice.bootstrap(s, B=11, seed=3)
    assert sizes == [11]                                                                      # a small table: one chunk
    del sizes[:]
    monkeypatch.setattr(flowprice, '_BOOT_BYTES', 8 * len(s.match) * 3)                      # the weights of three replicates fit
    chunked = flowprice.bootstrap(s, B=11, seed=3)
    assert sizes == [3, 3, 3, 2] and np.allclose(whole, chunked, rtol=1e-9, atol=1e-9)


def test_singular_replicates_are_nan_rows_and_an_empty_column_is_nan_in_its_replicate():
    """Two matches: column 0 is 1 in every row, column 1 is 1 in match 0 and 0 in match 1. Of the replicates (weights of match 0 and 1)
    (2, 0) is singular (the columns are equal in the rows that are left), (0, 2) has no variation in column 1, (1, 1) is fine."""
    s = design_stats([[1, 1], [1, 1], [1, 0], [1, 0]], [1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    boot = flowprice.bootstrap(s, B=400, seed=3)
    singular = np.isnan(boot).all(axis=1)
    half = np.isnan(boot).any(axis=1) & ~singular
    fine = ~np.isnan(boot).any(axis=1)
    assert 60 < singular.sum() < 140 and 60 < half.sum() < 140 and 150 < fine.sum() < 250         # 1/4, 1/4 and 1/2 of 400
    assert singular.sum() + half.sum() + fine.sum() == 400
    assert np.allclose(boot[half][:, 0], 3.5) and np.isnan(boot[half][:, 1]).all()                 # only the rows of match 1 are left
    assert np.allclose(boot[fine], [3.5, -2.0])                                                     # all four rows, in any doubling
    assert flowprice.singular_count(boot) == singular.sum()


def test_bootstrap_spread_is_the_standard_error_of_the_mean():
    v = np.random.default_rng(19).normal(size=200)
    boot = flowprice.bootstrap(design_stats(np.ones((200, 1)), v, np.arange(200)), B=2000, seed=20)
    assert boot.shape == (2000, 1) and np.var(boot[:, 0]) == pytest.approx(np.var(v) / 200, rel=0.12)
    assert np.mean(boot[:, 0]) == pytest.approx(np.mean(v), abs=4 * np.std(v) / math.sqrt(200 * 2000))


def test_bootstrap_refuses_nonsense_sizes_and_works_with_one_replicate():
    s = flowprice.flow_stats(synthetic_events(n=100, seed=31), 'baron')
    for bad in (0, -3, 2.5, True, '5'):
        with pytest.raises(ValueError, match='B'):
            flowprice.bootstrap(s, B=bad, seed=1)
    assert flowprice.bootstrap(s, B=1, seed=1).shape == (1, 19)


def test_an_event_level_control_changes_nothing_because_every_event_has_its_own_level():
    """P2: with a free level for each event, a control that is constant inside an event is taken out with it. Here a whole-number level
    per event (as the real ``co`` values are) is put into the column of 'dragon': the column is then without variation, and everything
    else is as with that column at 0."""
    t = synthetic_events(TRUE_BETA, n=400, event_sd=200.0, seed=32)
    level = 100.0 * np.random.default_rng(33).integers(-5, 6, (400, 1))
    with_level, without = t.co.copy(), t.co.copy()
    with_level[:, :, 1], without[:, :, 1] = level, 0.0
    a, b = solved(dataclasses.replace(t, co=with_level)), solved(dataclasses.replace(t, co=without))
    assert np.isnan(a[6 + 1]) and np.allclose(a, b, equal_nan=True) and np.isfinite(np.delete(a, 6 + 1)).all()


def test_percentile_intervals_cover_the_true_price_and_pretrend_at_about_the_nominal_rate():
    """Calibration of the whole chain (statistics, weights, solve, price and pre-trend per replicate, percentiles) on 60 simulated groups
    of 200 events with the truth known: the 95% intervals cover it about 95% of the time; a planted pre-trend is always found."""
    price_ok = trend_ok = ramp_found = ramp_ok = 0
    for i in range(60):
        for beta, true_trend in ((BETA_FLAT, 0.0), (BETA_RAMP, RAMP_PRETREND)):
            t = synthetic_events(beta, 200, event_sd=300.0, seed=1000 + i, noise_sd=60.0, per_match=1)
            s = flowprice.flow_stats(t, 'baron')
            beta_b = flowprice.flow_beta(flowprice.bootstrap(s, B=100, seed=i)[:, :6])
            price_lo, price_hi = np.percentile(flowprice.price_from_beta(beta_b), [2.5, 97.5])
            trend_lo, trend_hi = np.percentile(flowprice.pretrend_from_beta(beta_b), [2.5, 97.5])
            true_price = float(flowprice.price_from_beta(beta))
            if beta is BETA_FLAT:
                price_ok += price_lo <= true_price <= price_hi
                trend_ok += trend_lo <= true_trend <= trend_hi
            else:
                ramp_found += trend_lo > 0.0
                ramp_ok += trend_lo <= true_trend <= trend_hi
    assert price_ok >= 51 and trend_ok >= 51 and ramp_ok >= 51          # at least 85% of 60 (the nominal 95% would be 57)
    assert ramp_found == 60


# ---- the price table ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def mixed_table():
    return flowprice.price_table(mixed_events(seed=6), mixed_windows(seed=6), B=200, seed=6)


def test_adoption_rule_and_fixed_immediate():                    # herald alone has a trend before the event
    pt = flowprice.price_table(mixed_events(seed=6), mixed_windows(seed=6), B=200, seed=6)
    assert pt['dragon']['used'] == 'flow' and pt['herald']['used'] == 'immediate'
    assert all(pt[g]['used'] == 'immediate' and pt[g]['fixed_immediate'] for g in flowprice.FIXED_IMMEDIATE)


def test_the_planted_pretrend_is_found_and_the_flat_groups_are_adopted(mixed_table):
    herald, dragon = mixed_table['herald'], mixed_table['dragon']
    assert herald['pretrend'] == pytest.approx(RAMP_PRETREND, abs=40) and herald['pretrend_ci'][0] > 0 and not herald['adopted']
    assert herald['value'] == herald['immediate'] == pytest.approx(DEFAULT_WINDOW_EFFECT['herald'], abs=60)
    assert dragon['adopted'] and dragon['pretrend_ci'][0] <= 0 <= dragon['pretrend_ci'][1]
    assert dragon['value'] == dragon['price_flow'] == pytest.approx(950.0, abs=40)                         # 700 + 200 + 50, no trend
    assert dragon['beta'][:4] == pytest.approx([0, 0, 0, 0], abs=15) and dragon['beta'][4:] == pytest.approx([700, 200, 50], abs=15)
    assert [g for g in GROUPS if mixed_table[g]['used'] == 'flow' and mixed_table[g]['fixed_immediate']] == []


def test_the_fields_of_a_group_and_of_the_table(mixed_table):
    assert set(mixed_table) == set(GROUPS) | {'n_matches', 'n_window_rows', 'n_dropped_last', 'n_unknown_objectives', 'kill_gold_coef',
                                              'kill_gold_coef_ci', 'window_controls_without_variation', 'window_rank', 'B', 'seed'}
    assert (mixed_table['n_matches'], mixed_table['B'], mixed_table['seed'], mixed_table['n_dropped_last']) == (N_MIXED_MATCHES, 200, 6, 0)
    assert mixed_table['n_window_rows'] == 6000 and mixed_table['kill_gold_coef'] == pytest.approx(TRUE_WINDOW_CO[0], abs=0.2)
    lo, hi = mixed_table['kill_gold_coef_ci']
    assert lo < mixed_table['kill_gold_coef'] < hi
    for g in GROUPS:
        rec = mixed_table[g]
        assert set(rec) == {'fixed_immediate', 'n_events', 'n_excluded', 'n_event_matches', 'beta', 'beta_ci', 'price_flow',
                            'price_flow_ci', 'pretrend', 'pretrend_ci', 'adopted', 'n_boot_flow', 'flow_controls_without_variation',
                            'flow_rank', 'immediate', 'immediate_ci', 'n_boot_immediate', 'used', 'value', 'n_obj_unwindowed',
                            'obj_unwindowed_share'}, g
        assert rec['fixed_immediate'] == (g in flowprice.FIXED_IMMEDIATE) and rec['used'] in ('flow', 'immediate')
        assert rec['value'] == (rec['price_flow'] if rec['used'] == 'flow' else rec['immediate'])
        assert len(rec['beta']) == 7 and rec['beta'][3] == 0.0 and len(rec['beta_ci']) == 7 and rec['beta_ci'][3] == [0.0, 0.0]
        assert rec['price_flow_ci'][0] < rec['price_flow'] < rec['price_flow_ci'][1]
        assert rec['immediate_ci'][0] < rec['immediate'] < rec['immediate_ci'][1]
        assert rec['adopted'] == (rec['pretrend_ci'][0] <= 0 <= rec['pretrend_ci'][1])
        assert 150 <= rec['n_boot_flow'] <= 200 and 150 <= rec['n_boot_immediate'] <= 200
        assert rec['immediate'] == pytest.approx(DEFAULT_WINDOW_EFFECT[g], abs=60), g
    assert mixed_table['dragon']['flow_controls_without_variation'] == ['co_dragon']
    assert mixed_table['baron']['flow_controls_without_variation'] == ['co_baron']
    assert mixed_table['plate']['flow_controls_without_variation'] == [] and mixed_table['window_controls_without_variation'] == []
    assert mixed_table['window_rank'] == {'live': 48, 'rank': 48} and mixed_table['plate']['flow_rank'] == {'live': 19, 'rank': 19}
    assert mixed_table['dragon']['flow_rank'] == mixed_table['baron']['flow_rank'] == {'live': 18, 'rank': 18}   # no co column of its own


def test_the_table_is_plain_json(mixed_table):
    text = json.dumps(mixed_table, allow_nan=False)                      # no NaN, no numpy number
    assert json.loads(text) == mixed_table


def test_intervals_are_percentiles_of_quantities_recomputed_on_each_replicate(mixed_table):
    e, w = mixed_events(seed=6), mixed_windows(seed=6)
    rec = mixed_table['herald']
    beta_b = flowprice.flow_beta(flowprice.bootstrap(flowprice.flow_stats(e, 'herald'), B=200, seed=6))
    assert rec['price_flow_ci'] == pytest.approx(np.percentile(flowprice.price_from_beta(beta_b), [2.5, 97.5]).tolist())
    assert rec['pretrend_ci'] == pytest.approx(np.percentile(flowprice.pretrend_from_beta(beta_b), [2.5, 97.5]).tolist())
    assert rec['beta_ci'][4] == pytest.approx(np.percentile(beta_b[:, 4], [2.5, 97.5]).tolist())
    lo, hi = np.percentile(beta_b, [2.5, 97.5], axis=0)          # the price at the ends of the beta intervals is a different, wider thing
    assert not np.isclose(rec['price_flow_ci'][0], flowprice.price_from_beta(lo))
    assert not np.isclose(rec['price_flow_ci'][1], flowprice.price_from_beta(hi))
    window_b = flowprice.bootstrap(flowprice.window_stats(w), B=200, seed=6)
    assert rec['immediate_ci'] == pytest.approx(np.percentile(window_b[:, GROUPS.index('herald')], [2.5, 97.5]).tolist())
    assert mixed_table['kill_gold_coef_ci'] == pytest.approx(np.percentile(window_b[:, 12], [2.5, 97.5]).tolist())
    assert rec['beta'] == pytest.approx(flowprice.flow_beta(solved(e, 'herald')).tolist())
    assert rec['immediate'] == pytest.approx(float(flowprice.solve(flowprice.window_stats(w))[GROUPS.index('herald')]))


def test_events_excluded_and_objectives_in_no_window_are_counted_by_group():
    e = mixed_events(seed=7, n=100, n_incomplete=10)
    unwindowed = 3 * np.arange(12)
    pt = flowprice.price_table(e, mixed_windows(seed=7, n_obj_unwindowed=unwindowed, n_matches=e.n_matches), B=20, seed=7)
    for i, g in enumerate(GROUPS):
        assert (pt[g]['n_events'], pt[g]['n_excluded'], pt[g]['n_obj_unwindowed']) == (100, 10, 3 * i), g
        assert pt[g]['obj_unwindowed_share'] == pytest.approx(3 * i / 110)                    # of all the known objectives of the group
        assert pt[g]['n_event_matches'] <= N_MIXED_MATCHES and pt[g]['n_event_matches'] >= 50


def test_a_group_with_too_few_events_has_no_flow_values_and_the_others_continue():
    e = mixed_events(seed=8, n=100, n_incomplete=5)
    keep = np.ones(len(e.t), dtype=bool)
    atakhan = np.flatnonzero((e.group == GROUPS.index('atakhan')) & e.complete)
    keep[atakhan[1:]] = False                                                      # one complete event is left
    keep[e.group == GROUPS.index('elder')] = False                                 # none, incomplete ones included
    pt = flowprice.price_table(subset_events(e, keep), mixed_windows(seed=8, n_matches=e.n_matches), B=30, seed=8)
    for g, n_events, n_excluded in (('atakhan', 1, 5), ('elder', 0, 0)):
        rec = pt[g]
        assert (rec['n_events'], rec['n_excluded']) == (n_events, n_excluded), g
        for key in ('beta', 'beta_ci', 'price_flow', 'price_flow_ci', 'pretrend', 'pretrend_ci'):
            assert rec[key] is None, (g, key)
        assert rec['adopted'] is False and rec['used'] == 'immediate' and rec['n_boot_flow'] == 0
        assert rec['value'] == rec['immediate'] and rec['immediate'] is not None             # the window fit does not need the events
    assert pt['dragon']['price_flow'] is not None and pt['herald']['price_flow'] is not None and pt['baron']['n_events'] == 100
    assert pt['elder']['flow_rank'] == {'live': 0, 'rank': 0}
    assert pt['atakhan']['flow_rank']['rank'] == 6 < pt['atakhan']['flow_rank']['live']     # one event: rank 6, however many vary


def test_a_group_that_never_occurs_in_a_window_has_no_immediate_value():
    e = mixed_events(seed=9, n=100)
    w = mixed_windows(seed=9, n_matches=e.n_matches)
    treat = w.treat.copy()
    treat[:, GROUPS.index('inhibitor')] = 0.0
    pt = flowprice.price_table(e, dataclasses.replace(w, treat=treat), B=30, seed=9)
    rec = pt['inhibitor']
    assert rec['immediate'] is None and rec['immediate_ci'] is None and rec['n_boot_immediate'] == 0
    assert rec['used'] == 'immediate' and rec['value'] is None and rec['price_flow'] is not None          # fixed: the flow price is shown
    assert pt['plate']['immediate'] == pytest.approx(DEFAULT_WINDOW_EFFECT['plate'], abs=60)              # the others are estimated
    assert pt['window_controls_without_variation'] == ['treat_inhibitor']


def test_price_table_is_deterministic_and_the_seed_moves_the_intervals_only():
    e = mixed_events(seed=10, n=120)
    w = mixed_windows(seed=10, n_matches=e.n_matches)
    a, b, c = (flowprice.price_table(e, w, B=40, seed=s) for s in (1, 1, 2))
    assert a == b
    assert a['dragon']['price_flow'] == c['dragon']['price_flow'] and a['dragon']['immediate'] == c['dragon']['immediate']
    assert a['dragon']['price_flow_ci'] != c['dragon']['price_flow_ci'] and a['seed'] == 1 and c['seed'] == 2


def test_price_table_calls_its_check_between_the_stages():
    calls = []
    e = mixed_events(seed=11, n=60)
    w = mixed_windows(seed=11, n=3000, n_matches=e.n_matches)
    flowprice.price_table(e, w, B=10, seed=11, check=lambda: calls.append(1))
    assert len(calls) >= 13                                                  # before the window fit and before each of the 12 groups
    boom = RuntimeError('stop')

    def stop():
        raise boom
    with pytest.raises(RuntimeError, match='stop'):
        flowprice.price_table(e, w, B=10, seed=11, check=stop)


def test_price_table_refuses_tables_of_different_matches_and_bad_sizes():
    e = mixed_events(seed=12, n=60)
    w = mixed_windows(seed=12, n=3000, n_matches=e.n_matches)
    with pytest.raises(ValueError, match='matches'):
        flowprice.price_table(dataclasses.replace(e, n_matches=e.n_matches + 1), w, B=10, seed=1)
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match='B'):
            flowprice.price_table(e, w, B=bad, seed=1)
    with pytest.raises(ValueError, match='seed'):
        flowprice.price_table(e, w, B=10, seed=-1)


def test_a_flow_fit_that_is_not_estimable_is_not_used_whatever_its_interval(monkeypatch):
    """The rule as written: 'immediate' when the flow fit is not estimable, even if an interval of the pre-trend that holds 0 is there.
    (A singular point fit makes every replicate singular, so this cannot happen by itself; the rule does not depend on that.)"""
    real = flowprice._flow_fit

    def no_price(stats, B, seed):
        return {**real(stats, B, seed), 'price_flow': None, 'pretrend_ci': [-1.0, 1.0]}
    monkeypatch.setattr(flowprice, '_flow_fit', no_price)
    e = mixed_events(seed=13, n=60)
    pt = flowprice.price_table(e, mixed_windows(seed=13, n=3000, n_matches=e.n_matches), B=10, seed=13)
    assert all(pt[g]['adopted'] and pt[g]['used'] == 'immediate' and pt[g]['value'] == pt[g]['immediate'] for g in GROUPS)


# ---- the comparison with the prototype ---------------------------------------------------------------------------------------------------
def fake_probe(path, shift=0.0):
    def entry(i):
        return {'point': [100.0 * i + shift, 1.0], 'lo': [100.0 * i - 5, 0.0], 'hi': [100.0 * i + 5, 2.0]}      # horizons 0 and 1
    probe = {'rows': 1234, 'matches': 56, 'price': {g: entry(i) for i, g in enumerate(GROUPS)},
             'kill_gold_coef': {'point': [1.1, 1.6], 'lo': [1.0, 1.5], 'hi': [1.2, 1.7]}}
    Path(path).write_text(json.dumps(probe), encoding='utf-8')
    return probe


def test_the_probe_block_sets_our_immediate_value_beside_the_prototype_and_asserts_nothing(tmp_path, mixed_table):
    fake_probe(tmp_path / 'probe.json', shift=7.0)
    block = flowprice.probe_block(mixed_table, tmp_path / 'probe.json')
    assert block['found'] is True and block['path'] == str(tmp_path / 'probe.json')
    assert (block['probe_rows'], block['probe_matches']) == (1234, 56)
    for i, g in enumerate(GROUPS):
        rec = block['groups'][g]
        assert rec['immediate'] == mixed_table[g]['immediate'] and rec['probe'] == 100.0 * i + 7.0
        assert rec['probe_ci'] == [100.0 * i - 5, 100.0 * i + 5]
        assert rec['difference'] == pytest.approx(mixed_table[g]['immediate'] - (100.0 * i + 7.0))
    assert block['kill_gold_coef'] == {'kill_gold_coef': mixed_table['kill_gold_coef'], 'probe': 1.1, 'probe_ci': [1.0, 1.2],
                                       'difference': pytest.approx(mixed_table['kill_gold_coef'] - 1.1)}
    json.dumps(block, allow_nan=False)


def test_the_probe_block_survives_a_missing_a_broken_and_an_incomplete_probe(tmp_path, mixed_table):
    missing = flowprice.probe_block(mixed_table, tmp_path / 'nothing.json')
    assert missing == {'path': str(tmp_path / 'nothing.json'), 'found': False}
    (tmp_path / 'broken.json').write_text('{ not json', encoding='utf-8')
    broken = flowprice.probe_block(mixed_table, tmp_path / 'broken.json')
    assert broken['found'] is True and 'error' in broken and 'groups' not in broken
    (tmp_path / 'part.json').write_text(json.dumps({'price': {'baron': {'point': [5.0], 'lo': [4.0], 'hi': [6.0]}}}), encoding='utf-8')
    part = flowprice.probe_block(mixed_table, tmp_path / 'part.json')
    assert part['groups']['baron']['probe'] == 5.0 and part['groups']['dragon'] is None and part['kill_gold_coef'] is None


def test_the_default_probe_is_the_prototype_of_ruling_r4():
    assert flowprice.DEFAULT_PROBE == Path(r'D:\LOL_Data\runs\20261020_flow_price_probe_1\flow_price_probe.json')
    assert (flowprice.BOOT_B, flowprice.BOOT_SEED) == (2000, 20261021)


# ---- how the module is imported ----------------------------------------------------------------------------------------------------------
def test_no_module_level_import_reaches_the_s1_runner():
    imports = module_level_imports(Path(flowprice.__file__))
    chain = import_watch_names(('c_ept.run_s1', 'c_ept.run_s2', 'c_ept.store_build', 'c_ept.cohort', 'c_ept.targets', 'c_ept.runlog', 'c_ept.store_core'))
    assert not [i for i in imports if i in chain or i.startswith(chain)], sorted(imports)


def test_the_e6_guard_is_imported_inside_the_command_line():
    tree = ast.parse(Path(flowprice.__file__).read_text(encoding='utf-8'))
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'main')
    inner = [n for n in ast.walk(main) if isinstance(n, ast.ImportFrom) and n.module == 'c_ept.workflows.run_s1']
    assert inner and any(a.name == 'require_no_strata' for n in inner for a in n.names)


@pytest.mark.parametrize('module', ['c_ept.flowprice', 'c_ept.valuation.flowprice'])
def test_importing_the_module_loads_no_s1_module(module):
    chain = import_watch_names(('c_ept.run_s1', 'c_ept.run_s2', 'c_ept.store_build', 'c_ept.cohort',
                                'c_ept.targets', 'c_ept.runlog', 'c_ept.store_core'))
    code = (PROTECTED_IMPORT_PRELUDE + f'import sys, {module}\n'
            f'print(sorted(m for m in {chain!r} if m in sys.modules))')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip() == '[]', done.stderr[-500:]


# ---- toy matches with a planted baron effect, through the S1 and S2 stores ---------------------------------------------------------------
@pytest.fixture(scope='module')
def planted(tmp_path_factory):
    folder = tmp_path_factory.mktemp('planted')
    s1_run, s2_run = folder / '20261020_c_ept_s1_1', folder / '20261021_c_ept_s2_1'
    # Seed 3: the sampled pre-trend of this dataset is 0.9 gold (true value 0, standard error about 6) and its interval holds 0 with a
    # margin of about 10 at every bootstrap seed tried, so the baron is adopted. Seed 0 gives -13.6: one draw in 20 has an interval that
    # excludes the true 0, which is what a 95% interval means. The adoption rule itself is tested on the tables of ``mixed_events``.
    s1 = build_toy_s1(s1_run / 's1', planted_baron_matches(seed=3))
    s2 = build_toy_s2(s2_run / 's2', s1, values=random_summary_values)    # the patterned toy controls are collinear: see build_toy_s2
    yield types.SimpleNamespace(runs=(s1_run, s2_run), s1=s1, s2=s2, events=flowprice.event_table(s1),
                                windows=flowprice.window_table(s1, s2))
    del s1, s2
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


def test_a_planted_flow_effect_is_recovered_through_the_stores(planted):
    e = planted.events
    assert e.complete.all() and len(e.t) == 400 and set(e.group.tolist()) == {GROUPS.index('baron')} and set(e.taker.tolist()) == {0, 1}
    beta = flowprice.flow_beta(solved(e, 'baron'))
    assert np.allclose(beta, [0, 0, 0, 0, *PLANT_EFFECT], atol=30.0), beta       # both teams: the sign of the taker is in y
    assert flowprice.price_from_beta(beta) == pytest.approx(PLANT_PRICE, abs=70.0)
    assert abs(flowprice.pretrend_from_beta(beta)) < 40.0


def test_a_planted_immediate_value_is_recovered_through_the_stores(planted):
    w = planted.windows
    assert w.n_dropped_last == 400 and len(w.y) == 400 * 13 and w.n_obj_unwindowed.sum() == 0
    theta = flowprice.solve(flowprice.window_stats(w))
    assert theta[GROUPS.index('baron')] == pytest.approx(PLANT_EFFECT[0], abs=70.0)
    assert np.isnan(theta[[i for i, g in enumerate(GROUPS) if g != 'baron']]).all()          # no other group occurs: nothing to estimate


def test_the_planted_table_runs_through_price_table(planted):
    pt = flowprice.price_table(planted.events, planted.windows, B=100, seed=5)
    baron = pt['baron']
    assert baron['n_events'] == 400 and baron['n_excluded'] == 0 and baron['n_obj_unwindowed'] == 0 and baron['obj_unwindowed_share'] == 0.0
    assert baron['price_flow'] == pytest.approx(PLANT_PRICE, abs=70.0) and baron['adopted'] and baron['used'] == 'flow'
    assert baron['value'] == baron['price_flow'] and baron['immediate'] == pytest.approx(PLANT_EFFECT[0], abs=70.0)
    assert pt['dragon']['n_events'] == 0 and pt['dragon']['used'] == 'immediate' and pt['dragon']['value'] is None
    assert pt['n_dropped_last'] == 400 and pt['n_matches'] == 400


def test_the_patterned_toy_controls_are_collinear_and_the_rank_says_so(tmp_path):
    """Why ``build_toy_s2`` has ``values=``: with its periodic default summary some window controls add up to a constant."""
    s1 = build_toy_s1(tmp_path / 's1', planted_baron_matches(60, seed=3))
    for name, values in (('patterned', None), ('random', random_summary_values)):
        s2 = build_toy_s2(tmp_path / name, s1, values=values)
        rank = flowprice.rank_of(flowprice.window_stats(flowprice.window_table(s1, s2)))
        assert (rank['rank'] < rank['live']) == (values is None), (name, rank)


# ---- the command line --------------------------------------------------------------------------------------------------------------------
def cli(tmp_path, planted_stores, *extra, probe=None):
    s1_run, s2_run = planted_stores.runs
    return flowprice.main(['--s1', str(s1_run), '--s2', str(s2_run), '--probe', str(tmp_path / 'no_probe.json' if probe is None else probe),
                           '--B', '60', '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite'), *extra])


def catalog_rows(tmp_path):
    db = tmp_path / 'catalog.sqlite'
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        return con.execute('SELECT run, stage FROM runs').fetchall()
    finally:
        con.close()


def only_run(tmp_path) -> Path:
    runs = sorted((tmp_path / 'runs').iterdir())
    assert len(runs) == 1
    return runs[0]


def test_cli_writes_the_prices_and_the_provenance_and_registers_the_run(planted, tmp_path, capsys):
    from c_ept import runlog
    from c_ept.data import store_core
    assert cli(tmp_path, planted) == 0
    run = only_run(tmp_path)
    assert re.fullmatch(r'\d{8}_c_ept_s3a_1', run.name) and run.parent == tmp_path / 'runs'
    path = run / 'prices' / 'PRICES.json'
    prices, manifest = json.loads(path.read_text(encoding='utf-8')), json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    assert manifest['stage'] == 's3a' and manifest['run'] == run.name
    assert manifest['args']['B'] == 60 and manifest['args']['seed'] == 20261021
    s1_hash, s2_hash = runlog.sha256_json(planted.s1.manifest['files']), runlog.sha256_json(planted.s2.manifest['files'])
    assert manifest['s1'] == {'run': planted.runs[0].name, 'files_sha256': s1_hash}
    assert manifest['s2'] == {'run': planted.runs[1].name, 'files_sha256': s2_hash}
    assert prices['provenance'] == {'s1_files_sha256': s1_hash, 's2_files_sha256': s2_hash, 'code_commit': manifest['code_commit'],
                                    'code_dirty': manifest['code_dirty']}
    result = manifest['s3a_result']
    assert result['complete'] is True and result['prices_sha256'] == store_core.sha(path) and result['files_sha256'] == {
        'prices/PRICES.json': store_core.sha(path)}
    assert result['wall_s'] >= 0 and result['n_matches'] == 400 and 'peak_private_gib' in result and 'memory_samples' in result
    assert result['code_state_at_end'] == {'code_commit': manifest['code_commit'], 'code_dirty': manifest['code_dirty']}
    assert result['n_events'] == 400 and result['n_window_rows'] == 400 * 13 and result['n_dropped_last'] == 400
    assert result['used_flow'] == ['baron'] and len(result['used_immediate']) == 11
    for key in ('code_commit', 'code_dirty', 'config_sha256', 'spec_sha256', 'strata_running'):
        assert key in manifest
    assert prices['baron']['price_flow'] == pytest.approx(PLANT_PRICE, abs=80.0) and prices['baron']['used'] == 'flow'
    assert prices['probe'] == {'path': str(tmp_path / 'no_probe.json'), 'found': False}
    assert prices['B'] == 60 and prices['seed'] == 20261021
    assert catalog_rows(tmp_path) == [(run.name, 's3a')]
    out = capsys.readouterr()
    assert str(run) in out.out and KEY_PREFIX not in out.out + out.err
    assert KEY_PREFIX not in path.read_text(encoding='utf-8') + (run / 'RUN.json').read_text(encoding='utf-8')
    assert not (run / flowprice.LOCK_NAME).exists() and sorted(p.name for p in run.iterdir()) == ['RUN.json', 'prices']


def test_cli_prices_are_the_same_bytes_in_every_run_of_the_same_inputs(planted, tmp_path):
    other = tmp_path / 'again'
    other.mkdir()
    probe = tmp_path / 'no_probe.json'                      # the path is part of the file, so it is the same in both runs
    assert cli(tmp_path, planted, probe=probe) == 0 and cli(other, planted, probe=probe) == 0
    a, b = only_run(tmp_path), only_run(other)
    assert a.name == b.name and (a / 'prices' / 'PRICES.json').read_bytes() == (b / 'prices' / 'PRICES.json').read_bytes()


def test_cli_puts_the_probe_block_into_the_prices(planted, tmp_path):
    fake_probe(tmp_path / 'probe.json')
    assert cli(tmp_path, planted, probe=tmp_path / 'probe.json') == 0
    prices = json.loads((only_run(tmp_path) / 'prices' / 'PRICES.json').read_text(encoding='utf-8'))
    block = prices['probe']
    assert block['found'] is True and block['groups']['baron']['probe'] == 400.0
    assert block['groups']['baron']['immediate'] == prices['baron']['immediate']


def test_cli_accepts_the_store_folders_as_well_as_the_run_folders(planted, tmp_path):
    s1_run, s2_run = planted.runs
    assert flowprice.main(['--s1', str(s1_run / 's1'), '--s2', str(s2_run / 's2'), '--probe', str(tmp_path / 'no_probe.json'), '--B', '20',
                           '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')]) == 0
    manifest = json.loads((only_run(tmp_path) / 'RUN.json').read_text(encoding='utf-8'))
    assert manifest['s1']['run'] == s1_run.name and manifest['s2']['run'] == s2_run.name


def test_cli_uses_the_run_folder_it_is_given_and_never_overwrites_prices(planted, tmp_path, capsys):
    run = tmp_path / 'mine' / '20261021_c_ept_s3a_7'
    assert cli(tmp_path, planted, '--run', str(run)) == 0
    assert (run / 'prices' / 'PRICES.json').is_file() and (run / 'RUN.json').is_file() and not (tmp_path / 'runs').exists()
    assert catalog_rows(tmp_path) == [(run.name, 's3a')]
    before = (run / 'prices' / 'PRICES.json').read_bytes()
    assert cli(tmp_path, planted, '--run', str(run)) == 2
    assert 'PRICES.json' in capsys.readouterr().err and (run / 'prices' / 'PRICES.json').read_bytes() == before
    assert catalog_rows(tmp_path) == [(run.name, 's3a')]


def test_cli_stops_at_e6_before_a_run_folder_is_made(planted, tmp_path, fake_processes, capsys):
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE])
    assert cli(tmp_path, planted) == 2
    err = capsys.readouterr().err
    assert 'E6' in err and 'does not stop' in err and f'pid {synth.STRATA_SERVER[0]}' in err and 'strata_worker_mcp' not in err
    assert not (tmp_path / 'runs').exists() and catalog_rows(tmp_path) == []


def test_cli_checks_e6_again_between_the_stages_and_stops_cleanly(planted, tmp_path, monkeypatch, capsys):
    from c_ept.workflows import run_s1
    real, calls = run_s1.require_no_strata, []

    def fails_on_the_fourth_call():
        calls.append(1)
        if len(calls) == 4:
            raise run_s1.StrataRunning('E6: a Strata process started meanwhile')
        return real()
    monkeypatch.setattr(run_s1, 'require_no_strata', fails_on_the_fourth_call)
    assert cli(tmp_path, planted) == 2
    run = only_run(tmp_path)
    assert 'E6' in capsys.readouterr().err and not (run / 'prices').exists() and catalog_rows(tmp_path) == []
    assert not (run / flowprice.LOCK_NAME).exists()                                           # the lock is released on a failure too


def test_cli_on_the_toy_stores_reads_the_fit_matches_only_and_counts_what_it_leaves_out(toy_s1, toy_s2, tmp_path):
    assert flowprice.main(['--s1', str(toy_s1.root), '--s2', str(toy_s2.root), '--probe', str(tmp_path / 'none.json'), '--B', '5',
                           '--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')]) == 0
    prices = json.loads((only_run(tmp_path) / 'prices' / 'PRICES.json').read_text(encoding='utf-8'))
    assert prices['n_matches'] == len(TOY_FIT_MATCHES) == 4 and prices['n_window_rows'] == TOY_N_FIT_WINDOW_ROWS == 20
    assert prices['n_dropped_last'] == TOY_N_DROPPED_LAST == 2 and prices['n_unknown_objectives'] == TOY_N_UNKNOWN == 5
    for i, g in enumerate(GROUPS):                                           # complete + incomplete = the known objectives of the group
        rec = prices[g]
        assert rec['n_events'] + rec['n_excluded'] == TOY_EVENTS_BY_GROUP[i] and rec['n_excluded'] == TOY_INCOMPLETE_BY_GROUP[i], g
        assert rec['n_obj_unwindowed'] == TOY_OBJ_UNWINDOWED[i], g
        assert rec['obj_unwindowed_share'] == pytest.approx(TOY_OBJ_UNWINDOWED[i] / TOY_EVENTS_BY_GROUP[i]), g
    assert sum(prices[g]['n_events'] + prices[g]['n_excluded'] for g in GROUPS) == TOY_N_EVENTS      # the STOP and DEV objectives: not


def test_cli_continues_a_run_after_an_e6_stop_and_refuses_other_stores_b_or_seed_for_it(planted, tmp_path, monkeypatch, capsys):
    from c_ept.workflows import run_s1
    real, calls = run_s1.require_no_strata, []

    def fails_on_the_fourth_call_only():
        calls.append(1)
        if len(calls) == 4:
            raise run_s1.StrataRunning('E6: a Strata process started meanwhile')
        return real()
    monkeypatch.setattr(run_s1, 'require_no_strata', fails_on_the_fourth_call_only)
    run = tmp_path / '20261021_c_ept_s3a_1'
    assert cli(tmp_path, planted, '--run', str(run)) == 2
    err = capsys.readouterr().err
    assert str(run) in err and (run / 'RUN.json').is_file() and not (run / 'prices').exists() and catalog_rows(tmp_path) == []
    other_s1 = build_toy_s1(tmp_path / 'other' / '20261020_c_ept_s1_1' / 's1', planted_baron_matches(30, seed=5))
    build_toy_s2(tmp_path / 'other' / '20261021_c_ept_s2_1' / 's2', other_s1, values=random_summary_values)
    other = types.SimpleNamespace(runs=(tmp_path / 'other' / '20261020_c_ept_s1_1', tmp_path / 'other' / '20261021_c_ept_s2_1'))
    assert cli(tmp_path, other, '--run', str(run)) == 2
    assert 'other input stores' in capsys.readouterr().err and not (run / 'prices').exists() and catalog_rows(tmp_path) == []
    for flag, value in (('--B', '61'), ('--seed', '7')):                                      # the same stores, another B or seed
        assert cli(tmp_path, planted, '--run', str(run), flag, value) == 2
        assert 'same --B and --seed' in capsys.readouterr().err and not (run / 'prices').exists() and catalog_rows(tmp_path) == []
    assert cli(tmp_path, planted, '--run', str(run)) == 0                                     # the same stores: the run goes on
    manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    assert (run / 'prices' / 'PRICES.json').is_file() and manifest['s3a_result']['complete'] is True
    assert catalog_rows(tmp_path) == [(run.name, 's3a')] and not (run / flowprice.LOCK_NAME).exists()


@pytest.mark.parametrize('flag, value', [('--B', '0'), ('--B', '-5'), ('--seed', '-1')])
def test_cli_refuses_bad_numbers_before_a_run_folder_is_made(planted, tmp_path, flag, value, capsys):
    assert cli(tmp_path, planted, flag, value) == 2
    assert flag in capsys.readouterr().err and not (tmp_path / 'runs').exists()


def test_cli_refuses_a_missing_or_incomplete_store_before_a_run_folder_is_made(planted, tmp_path, capsys):
    s1_run, s2_run = planted.runs
    where = ['--runs-root', str(tmp_path / 'runs'), '--catalog-db', str(tmp_path / 'catalog.sqlite')]
    assert flowprice.main(['--s1', str(tmp_path / 'nope'), '--s2', str(s2_run), *where]) == 2
    assert 'S1' in capsys.readouterr().err
    assert flowprice.main(['--s1', str(s1_run), '--s2', str(tmp_path / 'nope'), *where]) == 2
    assert 'S2' in capsys.readouterr().err
    half = build_toy_s2(tmp_path / 'half' / 's2', planted.s1)
    half.manifest['complete'] = False
    half.save()
    assert flowprice.main(['--s1', str(s1_run), '--s2', str(tmp_path / 'half'), *where]) == 2
    assert 'not complete' in capsys.readouterr().err and not (tmp_path / 'runs').exists() and catalog_rows(tmp_path) == []


def test_cli_verify_checks_the_files_of_both_stores(planted, tmp_path, capsys):
    assert cli(tmp_path, planted, '--verify') == 0
    s1 = build_toy_s1(tmp_path / 'bad' / 's1', planted_baron_matches(20, seed=1))
    s2 = build_toy_s2(tmp_path / 'bad' / 's2', s1)
    del s1, s2
    gc.collect()
    victim = tmp_path / 'bad' / 's1' / 'kills' / 'killer.npy'
    victim.write_bytes(victim.read_bytes() + b'\0')
    assert flowprice.main(['--s1', str(tmp_path / 'bad' / 's1'), '--s2', str(tmp_path / 'bad' / 's2'), '--verify', '--runs-root',
                           str(tmp_path / 'runs2'), '--catalog-db', str(tmp_path / 'catalog.sqlite')]) == 2
    assert 'does not match STORE.json' in capsys.readouterr().err and not (tmp_path / 'runs2').exists()


def test_cli_refuses_a_run_folder_that_another_call_holds(planted, tmp_path, capsys):
    run = tmp_path / '20261021_c_ept_s3a_1'
    run.mkdir()
    (run / flowprice.LOCK_NAME).write_text(json.dumps({'pid': 4242, 'created': '2026-10-21T00:00:00+00:00'}), encoding='utf-8')
    assert cli(tmp_path, planted, '--run', str(run)) == 2
    err = capsys.readouterr().err
    assert 'lock' in err and '4242' in err and (run / flowprice.LOCK_NAME).is_file()          # it is not deleted for the user
    assert not (run / 'prices').exists() and not (run / 'RUN.json').exists() and catalog_rows(tmp_path) == []


def test_cli_reads_only_the_documented_arrays_and_never_the_match_keys(planted, tmp_path, monkeypatch):
    seen = []
    real = Store.array

    def spy(self, name):
        seen.append(name)
        return real(self, name)
    monkeypatch.setattr(Store, 'array', spy)
    assert cli(tmp_path, planted) == 0
    assert set(seen) <= S1_ARRAYS | S2_ARRAYS and 'matches/key' not in seen
