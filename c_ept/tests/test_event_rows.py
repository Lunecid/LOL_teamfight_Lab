"""Tests of c_ept.event_rows: the open views of M3 (at the first kill f) and of M4 (at delta). Synthetic data only.

The regular frames of ``synth`` lie at 0 and then at 60,000 k + 300 ms. A match of T = 440,000 ms has the frames k = 0 .. 7 (0, 60,300, ...,
420,300) and the frame-clock rows k = 2 .. 6: the frame at 420,300 is not before T - END_CUT_MS = 410,000, so it is no row.
"""
import dataclasses

import numpy as np
import pytest

from c_ept import config as C
from c_ept import event_rows, frame_rows, units
from c_ept.frame_rows import FIRST_ROW_FRAME
from c_ept.tests import synth

M3, M4 = 1, 2                      # the kind codes of the contract, written out so that the tests do not read them from the module
FIELDS = ('unit', 'kind', 'ref', 'frame_k', 't0', 'age_ms', 'frame_row')
DTYPES = (np.int32, np.int8, np.int64, np.int16, np.int64, np.int64, np.int32)


# ---- helpers ----------------------------------------------------------------------------------------------------------------
def kill(t):
    """A cross-team kill at one place: kills less than 15 s apart chain into one unit."""
    return dict(t=t, x=1_000.0, y=1_000.0, killer=1, victim=6)


def case(f=180_300, kills_t=None, T=600_000, n_frames=None):
    """``(rec, u, rows)`` of a synthetic match of ``T`` ms with kills at ``kills_t`` (default: the one kill ``f``) and the regular frames
    of ``synth`` (``n_frames`` of them; default: all before T)."""
    rec = synth.make_record(T_ms=T, n_frames=n_frames, kills=[kill(t) for t in ([f] if kills_t is None else kills_t)])
    return rec, units.build_units(units.kill_table(rec), rec.team, rec.T_ms), frame_rows.frame_rows(rec)


def case_scope():
    """Four units; the first (f = 100,000, before 2:00) and the last (f = 455,000, after T - 30 s = 450,000) are out of scope."""
    return case(kills_t=[100_000, 150_000, 300_000, 455_000], T=480_000)


def case_many():
    """Seven units in a match of T = 440,000 ms (frames k = 0 .. 7, rows k = 2 .. 6); the views of every unit are in ``TABLE``.

    Unit 0 (f = 100,000) is out of scope. Unit 1 has f = 120,000, so its M3 view is frame 1, which is no row (k < 2). Units 2 and 4 have f
    at a frame time (180,300 and 300,300), which the strict rule does not use. Unit 3 is two kills 500 ms apart (delta = f + 20 s). Unit 6
    (f = 405,000) has delta = 425,000, whose last frame is k = 7 at 420,300: a regular frame, but at or after T - 30 s, so no row.
    """
    return case(kills_t=[100_000, 120_000, 180_300, 262_000, 262_500, 300_300, 355_000, 405_000], T=440_000)


# (unit, kind, ref, frame_k, t0, age_ms, frame_row) of case_many, worked out by hand from the frames above and rows k = 2 .. 6 -> 0 .. 4
TABLE = [
    (1, M3, 120_000, 1, 60_300, 59_700, -1), (1, M4, 140_000, 2, 120_300, 19_700, 0),
    (2, M3, 180_300, 2, 120_300, 60_000, 0), (2, M4, 200_300, 3, 180_300, 20_000, 1),
    (3, M3, 262_000, 4, 240_300, 21_700, 2), (3, M4, 282_000, 4, 240_300, 41_700, 2),
    (4, M3, 300_300, 4, 240_300, 60_000, 2), (4, M4, 320_300, 5, 300_300, 20_000, 3),
    (5, M3, 355_000, 5, 300_300, 54_700, 3), (5, M4, 375_000, 6, 360_300, 14_700, 4),
    (6, M3, 405_000, 6, 360_300, 44_700, 4), (6, M4, 425_000, 7, 420_300, 4_700, -1),
]


def as_tuples(e):
    return list(zip(*(getattr(e, name).tolist() for name in FIELDS)))


def rows_with_k(rec, ks):
    """FrameRows with the frame indices ``ks`` (a set that ``frame_rows`` would not necessarily make); event_rows reads only ``k``."""
    k = np.array(ks, dtype=np.int16)
    t0 = rec.frame_ts[k]
    return frame_rows.FrameRows(k=k, t0=t0, hi=t0 + C.H_OCC_MS, exposure_ms=np.zeros((len(k), C.N_BINS), dtype=np.int32))


def assert_empty_events(e):
    assert all(len(getattr(e, name)) == 0 for name in FIELDS)
    assert tuple(getattr(e, name).dtype for name in FIELDS) == tuple(np.dtype(d) for d in DTYPES)


def brute_force(rec, u, rows):
    """The rows by explicit loops: for every in-scope unit its M3 row and, when delta < T, its M4 row (a unit with no frame before the
    reference time gets none), as (unit, kind, ref, frame_k, t0, age_ms, frame_row)."""
    ts = [int(t) for t in rec.frame_ts]
    row_of = {int(k): r for r, k in enumerate(rows.k)}
    out = []
    for i in range(len(u.f)):
        if not u.in_scope[i]:
            continue
        for kind, ref in ((M3, int(u.f[i])), (M4, int(u.delta[i]))):
            before = [j for j, t in enumerate(ts) if t < ref]
            if (kind == M4 and ref >= rec.T_ms) or not before:
                continue
            k = before[-1]
            out.append((i, kind, ref, k, ts[k], ref - ts[k], row_of.get(k, -1)))
    return out


# ---- the five tests of the brief (the second with Ruling 20 applied to its last assertion, and its converse added) -------------
def test_m3_row_uses_last_frame_strictly_before_f():
    e = event_rows.event_rows(*case(f=180_300))                 # the frame 180,300 is at the same time as f
    m3 = e.kind == 1
    assert e.t0[m3][0] == 120_300 and e.age_ms[m3][0] == 60_000


def test_frame_row_is_open_view_frame():
    rec, u, rows = case_many()
    e = event_rows.event_rows(rec, u, rows)
    m = e.frame_row >= 0
    assert np.array_equal(rows.k[e.frame_row[m]], e.frame_k[m]) and np.all(rows.t0[e.frame_row[m]] < e.ref[m])
    # a frame that is no row: frames 0 and 1, or a regular frame at or after T - 30 s (a late M4 view)  [Ruling 20]
    assert np.all((e.frame_k[~m] < FIRST_ROW_FRAME) | (rec.frame_ts[e.frame_k[~m]] >= rec.T_ms - C.END_CUT_MS))
    # the converse: a frame that is a row is frame 2 or later and before T - 30 s
    assert np.all((e.frame_k[m] >= FIRST_ROW_FRAME) & (rec.frame_ts[e.frame_k[m]] < rec.T_ms - C.END_CUT_MS))
    assert m.any() and (e.frame_k[~m] < FIRST_ROW_FRAME).any() and (e.frame_k[~m] >= FIRST_ROW_FRAME).any()      # both kinds of -1 occur


def test_m4_row_skipped_when_delta_at_or_after_T():
    e = event_rows.event_rows(*case(kills_t=[550_000, 564_999, 579_999], T=580_000))   # f = T - 30 s (in scope), delta = 580,000 = T
    assert np.any(e.kind == 1) and not np.any(e.kind == 2)


def test_only_in_scope_units_get_rows():
    e = event_rows.event_rows(*case_scope())
    assert set(e.unit) == {1, 2}


def test_age_bounds():
    rec, u, rows = case_many()
    e = event_rows.event_rows(rec, u, rows)
    assert np.all((e.age_ms > 0) & (e.age_ms < 61_000))


# ---- hand-checked rows ------------------------------------------------------------------------------------------------------
def test_case_many_rows_equal_the_hand_worked_table():
    rec, u, rows = case_many()
    assert list(rows.k) == [2, 3, 4, 5, 6] and len(u.f) == 7 and list(u.in_scope) == [False] + [True] * 6
    assert as_tuples(event_rows.event_rows(rec, u, rows)) == TABLE


def test_late_m4_view_frame_is_a_regular_frame_but_no_row():
    rec, u, rows = case(f=405_000, T=440_000)                   # delta = 425,000; the last frame before it is k = 7 at 420,300 >= 410,000
    e = event_rows.event_rows(rec, u, rows)
    assert as_tuples(e) == [(0, M3, 405_000, 6, 360_300, 44_700, 4), (0, M4, 425_000, 7, 420_300, 4_700, -1)]
    assert 7 not in rows.k and rec.frame_ts[7] >= rec.T_ms - C.END_CUT_MS


@pytest.mark.parametrize('f, k, t0', [(180_299, 2, 120_300), (180_300, 2, 120_300), (180_301, 3, 180_300)])
def test_m3_frame_is_strictly_before_f(f, k, t0):
    e = event_rows.event_rows(*case(f=f))
    m3 = e.kind == M3
    assert (e.ref[m3][0], e.frame_k[m3][0], e.t0[m3][0], e.age_ms[m3][0]) == (f, k, t0, f - t0)


@pytest.mark.parametrize('delta, k, t0', [(180_299, 2, 120_300), (180_300, 2, 120_300), (180_301, 3, 180_300)])
def test_m4_frame_is_strictly_before_delta(delta, k, t0):
    e = event_rows.event_rows(*case(f=delta - C.DELTA_AFTER_F_MS))      # one kill: delta = f + 20 s
    m4 = e.kind == M4
    assert (e.ref[m4][0], e.frame_k[m4][0], e.t0[m4][0], e.age_ms[m4][0]) == (delta, k, t0, delta - t0)


def test_ref_is_f_for_m3_and_delta_for_m4():
    rec, u, rows = case_many()
    e = event_rows.event_rows(rec, u, rows)
    assert np.array_equal(e.ref[e.kind == M3], u.f[e.unit[e.kind == M3]])
    assert np.array_equal(e.ref[e.kind == M4], u.delta[e.unit[e.kind == M4]])
    assert np.array_equal(e.t0, rec.frame_ts[e.frame_k]) and np.array_equal(e.age_ms, e.ref - e.t0)


# ---- the delta < T rule ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kills_t, delta, has_m4', [
    ([550_000, 564_999, 579_998], 579_999, True),                       # delta = T - 1
    ([550_000, 564_999, 579_999], 580_000, False),                      # delta = T
    ([550_000, 564_999, 579_999, 580_000], 580_001, False),             # delta = T + 1 (the last kill is at the game end)
])
def test_m4_row_exists_iff_delta_is_before_T(kills_t, delta, has_m4):
    rec, u, rows = case(kills_t=kills_t, T=580_000)                     # one chain (gaps <= 15 s): f = 550,000 = T - 30 s, delta = last + 1
    assert len(u.f) == 1 and u.in_scope[0] and u.delta[0] == delta
    e = event_rows.event_rows(rec, u, rows)
    assert list(e.kind) == ([M3, M4] if has_m4 else [M3])
    if has_m4:                                                           # the last regular frame is k = 9 at 540,300, the last row
        assert (e.ref[1], e.frame_k[1], e.t0[1], e.age_ms[1], e.frame_row[1]) == (579_999, 9, 540_300, 39_699, 7)


def test_m3_row_is_made_for_every_in_scope_unit_and_m4_when_delta_before_T():
    rec, u, rows = case_many()
    e = event_rows.event_rows(rec, u, rows)
    assert (e.kind == M3).sum() == u.in_scope.sum() == 6
    assert (e.kind == M4).sum() == (u.in_scope & (u.delta < rec.T_ms)).sum() == 6


# ---- scope, order -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('f, in_scope', [(119_999, False), (120_000, True), (450_000, True), (450_001, False)])
def test_scope_boundaries_follow_the_unit_table(f, in_scope):
    rec, u, rows = case(f=f, T=480_000)                                 # T - 30 s = 450,000, both ends included
    assert bool(u.in_scope[0]) == in_scope
    assert list(event_rows.event_rows(rec, u, rows).unit) == ([0, 0] if in_scope else [])


def test_rows_are_ordered_by_unit_then_kind():
    e = event_rows.event_rows(*case_many())
    key = e.unit.astype(np.int64) * 3 + e.kind
    assert np.all(np.diff(key) > 0)                                     # strictly: one row per (unit, kind)
    assert list(e.unit) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] and list(e.kind) == [M3, M4] * 6


def test_kind_codes():
    assert (event_rows.KIND_M3, event_rows.KIND_M4) == (M3, M4)


# ---- frame_row --------------------------------------------------------------------------------------------------------------
def test_frame_row_searches_the_rows_and_is_not_k_minus_2():
    rec, u, _ = case_many()                                             # views at frames k = 1, 2, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7
    ks = [3, 5, 6]                                                      # rows with gaps (frame_rows makes none): not k - 2
    e = event_rows.event_rows(rec, u, rows_with_k(rec, ks))
    assert list(e.frame_k) == [1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7]
    assert list(e.frame_row) == [ks.index(k) if k in ks else -1 for k in e.frame_k] == [-1, -1, -1, 0, -1, -1, -1, 1, 1, 2, 2, -1]


@pytest.mark.parametrize('args, want', [
    # two regular frames: frame_rows gives no rows at all, but the views exist
    (dict(kills_t=[200_000], n_frames=2), [(0, M3, 200_000, 1, 60_300, 139_700, -1), (0, M4, 220_000, 1, 60_300, 159_700, -1)]),
    # frame 0 is all there is to view
    (dict(kills_t=[200_000], n_frames=1), [(0, M3, 200_000, 0, 0, 200_000, -1), (0, M4, 220_000, 0, 0, 220_000, -1)]),
    # frame 2 at 120,300 = T - 30 s is no row (the cut is strict), so there are no rows; it is the M4 view of f = 120,000 all the same
    (dict(kills_t=[120_000], T=150_300), [(0, M3, 120_000, 1, 60_300, 59_700, -1), (0, M4, 140_000, 2, 120_300, 19_700, -1)]),
])
def test_without_frame_clock_rows_every_frame_row_is_minus_one(args, want):
    rec, u, rows = case(**args)
    assert len(rows.k) == 0 and u.in_scope.all()
    assert as_tuples(event_rows.event_rows(rec, u, rows)) == want


def test_rows_out_of_frame_order_are_refused():
    rec, u, _ = case_many()
    for ks in ([3, 2], [2, 2, 3]):
        with pytest.raises(ValueError):
            event_rows.event_rows(rec, u, rows_with_k(rec, ks))


# ---- empty inputs, dtypes, ownership ----------------------------------------------------------------------------------------
@pytest.mark.parametrize('args', [
    dict(kills_t=[]),                                   # no units
    dict(kills_t=[100_000, 590_000]),                   # units, none in scope (before 2:00, after T - 30 s = 570,000)
    dict(kills_t=[200_000], n_frames=0),                # an in-scope unit, but no regular frame to view: no row
])
def test_empty_inputs_give_empty_arrays_with_their_dtypes(args):
    assert_empty_events(event_rows.event_rows(*case(**args)))


def test_dtypes_and_shapes():
    rec, u, rows = case_many()
    e = event_rows.event_rows(rec, u, rows)
    assert tuple(getattr(e, name).dtype for name in FIELDS) == tuple(np.dtype(d) for d in DTYPES)
    assert all(getattr(e, name).shape == (12,) for name in FIELDS)


def test_dtypes_do_not_depend_on_the_input_dtypes():
    rec, u, rows = case_many()
    narrow_u = dataclasses.replace(u, f=u.f.astype(np.int32), delta=u.delta.astype(np.int32))      # tables made by hand may be narrower
    e = event_rows.event_rows(rec, narrow_u, dataclasses.replace(rows, k=rows.k.astype(np.int32)))
    assert as_tuples(e) == TABLE
    assert tuple(getattr(e, name).dtype for name in FIELDS) == tuple(np.dtype(d) for d in DTYPES)


def test_arrays_are_new_and_writable_not_views_of_the_inputs():
    rec, u, rows = case_many()
    before = [a.copy() for a in (rec.frame_ts, rows.k, rows.t0, u.f, u.delta)]
    e = event_rows.event_rows(rec, u, rows)
    for name in FIELDS:
        a = getattr(e, name)
        assert a.flags.writeable and not any(np.shares_memory(a, src) for src in (rec.frame_ts, rows.k, rows.t0, u.f, u.delta))
        a[0] += 1                                                       # the caller may edit its rows; the inputs stay as they were
    assert all(np.array_equal(a, b) for a, b in zip((rec.frame_ts, rows.k, rows.t0, u.f, u.delta), before))


# ---- random matches ---------------------------------------------------------------------------------------------------------
def test_random_matches_equal_brute_force():
    n_m3 = n_m4 = n_late = 0
    for seed in range(25):
        rec = synth.records_from_seed(f'SYN_{seed}', '15.18')
        u = units.build_units(units.kill_table(rec), rec.team, rec.T_ms)
        rows = frame_rows.frame_rows(rec)
        e = event_rows.event_rows(rec, u, rows)
        assert as_tuples(e) == brute_force(rec, u, rows)
        is_row = (e.frame_k >= FIRST_ROW_FRAME) & (rec.frame_ts[e.frame_k] < rec.T_ms - C.END_CUT_MS)     # the row rule, without rows.k
        assert np.array_equal(e.frame_row >= 0, is_row)
        assert np.all((e.age_ms > 0) & (e.age_ms < 61_000))
        n_m3 += int((e.kind == M3).sum())
        n_m4 += int((e.kind == M4).sum())
        n_late += int(((e.frame_k >= FIRST_ROW_FRAME) & ~is_row).sum())
    assert n_m3 > 300 and n_m4 > 300 and n_late > 0                    # the comparison is not vacuous: it covers late frames too
