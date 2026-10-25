"""Tests of c_ept.frame_rows: the frame-clock rows with their exposure per occurrence bin, and the onsets table (row, bin, mark, unit).

Synthetic data only. The frames of ``synth`` lie at 0 and then at 60,000 k + 300 ms, so a match of T = 425,000 ms has the regular frames
0, 60,300, 120,300, ..., 420,300.
"""
import dataclasses

import numpy as np
import pytest

from c_ept import config as C
from c_ept.data import frame_rows
from c_ept.domain import units
from c_ept.tests import synth
from c_ept.tests.synth import TEAM, kt

LONG_T = 1_200_000     # game end of the hand-built rows: far enough that no window is cut by T - END_CUT_MS


# ---- helpers ----------------------------------------------------------------------------------------------------------------
def rec_with_frames(T, n_frames=None):
    """A synthetic match of T ms with the regular frames 0, 60,300, 120,300, ... (``n_frames`` of them; default: all before T)."""
    return synth.make_record(T_ms=T, n_frames=n_frames)


def rows_at(t0s, T=LONG_T):
    """FrameRows with one row per time in ``t0s`` (frames 2, 3, ...); hi is what the rule gives for a match of T ms. ``onsets`` does not
    read the exposure, so it is left at 0 here."""
    t0 = np.array(t0s, dtype=np.int64)
    return frame_rows.FrameRows(k=np.arange(2, 2 + len(t0), dtype=np.int16), t0=t0, hi=np.minimum(t0 + C.H_OCC_MS, T - C.END_CUT_MS),
                                exposure_ms=np.zeros((len(t0), C.N_BINS), dtype=np.int32))


def rows_one(t0):
    return rows_at([t0])


def rows_three(t0s):
    """Three rows whose 180 s windows overlap."""
    assert len(t0s) == 3 and t0s[2] - t0s[0] < C.H_OCC_MS
    return rows_at(t0s)


def units_at(fs, pairs=None):
    """UnitTable with one unit per first-kill time in ``fs`` (ascending; equal times give separate units). Every unit is one kill and
    the kills lie 5,000 map units apart, so none links to another. ``pairs``: (killer, victim) of each kill; default killer 1, victim 6
    (mark 0)."""
    pairs = pairs or [(1, 6)] * len(fs)
    u = units.build_units(kt([(f, 5_000 * i, kr, vc) for i, (f, (kr, vc)) in enumerate(zip(fs, pairs))]), TEAM, LONG_T)
    assert list(u.f) == list(fs)
    return u


def bins_for(t0, fs):
    """The occurrence bins of the units whose first kills are at ``fs``, in the one row at ``t0`` (the order of the onsets)."""
    return [int(b) for b in frame_rows.onsets(rows_one(t0), units_at(sorted(fs))).bin]


def assert_empty_rows(r):
    assert len(r.k) == len(r.t0) == len(r.hi) == 0
    assert (r.k.dtype, r.t0.dtype, r.hi.dtype, r.exposure_ms.dtype) == (np.int16, np.int64, np.int64, np.int32)
    assert r.exposure_ms.shape == (0, C.N_BINS)


def assert_empty_onsets(o):
    assert len(o.row) == len(o.bin) == len(o.mark) == len(o.unit) == 0
    assert (o.row.dtype, o.bin.dtype, o.mark.dtype, o.unit.dtype) == (np.int32, np.int8, np.int16, np.int32)


# ---- rows and exposure ------------------------------------------------------------------------------------------------------
def test_rows_start_at_frame_2_and_stop_before_end_cut():
    r = frame_rows.frame_rows(rec_with_frames(T=425_000))      # frames 0, 60,300, 120,300, ..., last regular frame 420,300
    assert r.k[0] == 2 and r.t0[0] == 120_300 and r.t0[-1] == 360_300 and r.t0[-1] < 425_000 - 30_000


def test_exposure_cut_at_end():
    r = frame_rows.frame_rows(rec_with_frames(T=425_000))
    assert r.hi[0] == 300_300 and list(r.exposure_ms[0]) == [10_000] * 18
    assert r.hi[-1] == 395_000 and list(r.exposure_ms[-1]) == [10_000, 10_000, 10_000, 4_700] + [0] * 14
    assert list(r.exposure_ms[2]) == [10_000] * 15 + [4_700, 0, 0]          # t0 = 240,300


def test_onset_bins_boundaries():
    t0 = 120_300
    assert bins_for(t0, [t0 + 1, t0 + 10_000, t0 + 10_001, t0 + 60_000, t0 + 60_001, t0 + 180_000]) == [0, 0, 1, 5, 6, 17]
    assert bins_for(t0, [t0]) == [] and bins_for(t0, [t0 + 180_001]) == []


def test_unit_appears_in_every_row_whose_window_holds_f():
    o = frame_rows.onsets(rows_three(t0s=[120_300, 180_300, 240_300]), units_at([250_000]))
    assert list(o.row) == [0, 1, 2] and list(o.bin) == [12, 6, 0]


def test_onset_counts_every_unit_in_window():
    o = frame_rows.onsets(rows_one(t0=120_300), units_at([130_000, 131_000, 175_000]))
    assert len(o.row) == 3 and list(o.bin) == [0, 1, 5]


def test_rows_are_the_frames_from_2_before_the_cut():
    rec = rec_with_frames(T=425_000)
    r = frame_rows.frame_rows(rec)
    assert list(r.k) == [2, 3, 4, 5, 6] and np.array_equal(r.t0, rec.frame_ts[2:7])        # the frame at 420,300 >= 395,000 is no row
    assert list(r.t0) == [120_300, 180_300, 240_300, 300_300, 360_300]
    assert list(r.hi) == [300_300, 360_300, 395_000, 395_000, 395_000]


def test_end_cut_is_strict():
    assert_empty_rows(frame_rows.frame_rows(rec_with_frames(T=150_300)))        # frame 2 at 120,300 = T - 30,000 is not before it
    r = frame_rows.frame_rows(rec_with_frames(T=150_301))                       # one ms later it is a row, with a window of 1 ms
    assert list(r.k) == [2] and r.t0[0] == 120_300 and r.hi[0] == 120_301 and list(r.exposure_ms[0]) == [1] + [0] * 17


@pytest.mark.parametrize('T, n_frames, n_regular', [
    (425_000, 0, 0), (425_000, 1, 1), (425_000, 2, 2),         # fewer than three regular frames
    (140_000, None, 3), (150_300, None, 3),                    # three frames, none of them before T - 30 s
])
def test_no_usable_frame_gives_no_rows(T, n_frames, n_regular):
    rec = rec_with_frames(T=T, n_frames=n_frames)
    assert len(rec.frame_ts) == n_regular
    assert_empty_rows(frame_rows.frame_rows(rec))


@pytest.mark.parametrize('T', [140_000, 150_301, 200_000, 331_000, 425_000, 600_000, 1_234_567, 2_500_000])
def test_exposure_is_the_part_of_each_bin_inside_the_window(T):
    r = frame_rows.frame_rows(rec_with_frames(T=T))
    assert (len(r.k) > 0) == (T > 150_300)
    for i in range(len(r.k)):
        t0, hi = int(r.t0[i]), int(r.hi[i])
        assert t0 < T - C.END_CUT_MS and hi == min(t0 + C.H_OCC_MS, T - C.END_CUT_MS) and hi > t0
        for q in range(C.N_BINS):                                  # bin q is (t0 + BIN q, t0 + BIN (q + 1)]: how much of it is before hi
            assert r.exposure_ms[i, q] == max(0, min(t0 + C.BIN_MS * (q + 1), hi) - (t0 + C.BIN_MS * q))
        assert r.exposure_ms[i].sum() == hi - t0


def test_dtypes_and_shapes():
    r = frame_rows.frame_rows(rec_with_frames(T=425_000))
    assert (r.k.dtype, r.t0.dtype, r.hi.dtype, r.exposure_ms.dtype) == (np.int16, np.int64, np.int64, np.int32)
    assert r.k.shape == r.t0.shape == r.hi.shape == (5,) and r.exposure_ms.shape == (5, C.N_BINS)
    o = frame_rows.onsets(r, units_at([200_000, 250_000]))
    assert (o.row.dtype, o.bin.dtype, o.mark.dtype, o.unit.dtype) == (np.int32, np.int8, np.int16, np.int32)
    assert len(o.row) == len(o.bin) == len(o.mark) == len(o.unit) == 5


def test_rows_are_new_arrays_not_views_of_the_record():
    rec = rec_with_frames(T=425_000)
    before = rec.frame_ts.copy()
    r = frame_rows.frame_rows(rec)
    assert all(a.flags.writeable and not np.shares_memory(a, rec.frame_ts) for a in (r.k, r.t0, r.hi, r.exposure_ms))
    r.t0[0] += 1                                                                # the caller may edit its rows; the record stays as it was
    assert np.array_equal(rec.frame_ts, before)


# ---- onsets -----------------------------------------------------------------------------------------------------------------
def test_onsets_in_row_order_then_unit_order_with_marks_and_local_unit_index():
    pairs = [(1, 6), (2, 7), (6, 1), (3, 8), (0, 3)]                    # marks 0, 6, 25, 12 and 50 (an execution on a blue victim)
    u = units_at([130_000, 200_000, 250_000, 250_000, 400_000], pairs)  # two units at the same ms: unit order
    assert list(u.mark) == [0, 6, 25, 12, 50]
    o = frame_rows.onsets(rows_three(t0s=[120_300, 180_300, 240_300]), u)
    assert list(o.row) == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert list(o.unit) == [0, 1, 2, 3, 1, 2, 3, 2, 3, 4]               # local indices; unit 0 (f = 130,000) is in progress at rows 1 and 2
    assert list(o.bin) == [0, 7, 12, 12, 1, 6, 6, 0, 0, 15]
    assert list(o.mark) == [0, 6, 25, 12, 6, 25, 12, 25, 12, 50] and np.array_equal(o.mark, u.mark[o.unit])


def test_a_unit_in_progress_at_t0_is_no_onset_of_that_row():
    t0 = 120_300
    assert bins_for(t0, [t0 - 60_000, t0 - 1, t0]) == []                # f <= t0: the engagement has begun, the event clock handles it
    assert bins_for(t0, [t0 - 1, t0 + 1]) == [0]


def test_end_cut_limits_the_onsets_like_the_exposure():
    rec = rec_with_frames(T=425_000)                                    # T - 30,000 = 395,000
    r = frame_rows.frame_rows(rec)
    o = frame_rows.onsets(r, units_at([395_000, 395_001]))
    assert list(o.row) == [2, 3, 4] and list(o.bin) == [15, 9, 3] and set(o.unit) == {0}
    assert np.all(r.exposure_ms[o.row, o.bin] > 0)                      # an onset never falls into a bin with no exposure


def test_onsets_are_not_filtered_by_scope():
    u = units_at([100_000])                                             # a first kill before 2:00 is out of scope
    assert not u.in_scope[0]
    assert list(frame_rows.onsets(rows_one(60_300), u).bin) == [3]      # a row before frame 2 (not made by frame_rows) still counts it


@pytest.mark.parametrize('rows_args, fs', [
    (dict(T=425_000), []),                                              # no units
    (dict(T=425_000), [100_000, 120_300, 395_001, 420_000]),            # units, but none inside a window (before or at t0, past T - 30 s)
    (dict(T=425_000, n_frames=2), [200_000]),                           # no rows
    (dict(T=425_000, n_frames=0), []),                                  # neither
])
def test_onsets_of_empty_inputs_are_empty(rows_args, fs):
    rows = frame_rows.frame_rows(rec_with_frames(**rows_args))
    assert (len(rows.k) == 0) == ('n_frames' in rows_args)             # the cases that cut the frames have no rows, the others have some
    assert_empty_onsets(frame_rows.onsets(rows, units_at(fs)))


def test_unsorted_first_kills_are_refused():
    u = units_at([200_000, 250_000])
    with pytest.raises(ValueError):
        frame_rows.onsets(rows_one(120_300), dataclasses.replace(u, f=u.f[::-1].copy()))


def test_random_matches_equal_brute_force():
    n_rows = n_onsets = 0
    for seed in range(25):
        rec = synth.records_from_seed(f'SYN_{seed}', '15.18')
        u = units.build_units(units.kill_table(rec), rec.team, rec.T_ms)
        rows = frame_rows.frame_rows(rec)
        o = frame_rows.onsets(rows, u)
        want = [(r, i, (int(u.f[i]) - int(rows.t0[r]) - 1) // C.BIN_MS, int(u.mark[i]))
                for r in range(len(rows.k)) for i in range(len(u.f)) if rows.t0[r] < u.f[i] <= rows.hi[r]]
        assert list(zip(o.row.tolist(), o.unit.tolist(), o.bin.tolist(), o.mark.tolist())) == want
        assert np.all(u.in_scope[o.unit])                                          # rows from frame 2, windows before T - 30 s: in scope
        assert np.all(rows.exposure_ms[o.row, o.bin] > 0)
        assert np.all(np.bincount(o.unit, minlength=len(u.f)) <= 3)                # 180 s windows, rows 60 s apart: at most three rows
        n_rows += len(rows.k)
        n_onsets += len(o.row)
    assert n_rows > 300 and n_onsets > 300                                         # the comparison is not vacuous
