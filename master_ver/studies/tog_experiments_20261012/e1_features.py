"""T1.2 E1 matching features of a window [start, start + D] (PRESPEC_E1_KO.md section 3 as amended 2026-10-13).

One function gives the features of an engagement (start = tau, D = its h90 endpoint - tau) and of a placebo
candidate (start = s, with the engagement's D), from one match's bank arrays:

  frames_ms      the match's frame clock (bank ``minute_ts``), strictly increasing
  kills_ms       its CHAMPION_KILL times, sorted
  gold_by_frame  per frame the cached team gold difference gold_team[F, 0] - gold_team[F, 1] (FIELD_MANIFEST
                 team_order [100, 200]: blue - red), as float64 (``team_gold_difference``)

All times are ms from the game start; F(t) is the last frame at or before t.

  start_age_bin  start - F(start) <= 5,000 -> 0, in (5,000, 10,000] -> 1; outside [0, 10,000] or no frame at or before
                 the start -> -1 (not a candidate: reject key start_age_out)
  n_bound        frames f with start < f <= start + D (F_pre = F_post is n_bound = 0)
  end_age_bin    (start + D) - F(start + D) in [0, 20,000) -> 0, [20,000, 40,000) -> 1, >= 40,000 -> 2 (open above:
                 frame gaps exceed 60 s)
  time_bin       start / 60,000 < 15 -> 0, [15, 25) -> 1, >= 25 -> 2 (controls.py ``_game_bin``)
  kill_gap_bin   start - (last kill before the start) < 30,000 -> 0, [30,000, 60,000) -> 1, [60,000, 90,000) -> 2,
                 [90,000, 180,000) -> 3, >= 180,000 or no earlier kill -> 4 (a kill at the start itself is not before
                 it); the five bins of the 2026-10-13 amendment replace the 10-12 three bins (edges 90 and 180 s)
  gold_bin       np.searchsorted(gold_edges[time_bin], gold at F(start), side="right"), 0-4

Gold edges (``gold_edges``): per time_bin the 20/40/60/80 percentiles (numpy.quantile, linear) of the F(tau) gold
difference of the TRAIN q rows, a 3 x 4 array fixed before any window is made (PRESPEC_E1 Appendix A).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

FEATURES = ("start_age_bin", "n_bound", "end_age_bin", "time_bin", "kill_gap_bin", "gold_bin")
START_AGE_SPLIT_MS = 5_000           # start age <= 5 s -> bin 0
START_AGE_MAX_MS = 10_000            # start age beyond 10 s: not a candidate (start_age_out)
END_AGE_EDGES_MS = (20_000, 40_000)  # [0, 20) -> 0, [20, 40) -> 1, >= 40 s -> 2
TIME_EDGES_MIN = (15, 25)            # < 15 -> 0, [15, 25) -> 1, >= 25 min -> 2
KILL_GAP_EDGES_MS = (30_000, 60_000, 90_000, 180_000)   # five bins; no earlier kill -> the last bin
GOLD_QUANTILES = (0.2, 0.4, 0.6, 0.8)
N_TIME_BINS = len(TIME_EDGES_MIN) + 1
N_KILL_GAP_BINS = len(KILL_GAP_EDGES_MS) + 1


def _int_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size and array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers (ms)")
    return array.astype(np.int64, copy=False).reshape(-1)


def _frames(frames_ms) -> np.ndarray:
    frames = _int_array(frames_ms, "frames_ms")
    if not len(frames):
        raise ValueError("a match needs at least one frame")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frame times must be strictly increasing")
    return frames


def _kills(kills_ms) -> np.ndarray:
    kills = _int_array(kills_ms, "kills_ms")
    if np.any(np.diff(kills) < 0):
        raise ValueError("kill times must be sorted")
    return kills


def frame_at(frames_ms, t_ms) -> np.ndarray:
    """Index of F(t), the last frame at or before t; -1 when no frame is at or before t."""

    return np.searchsorted(_frames(frames_ms), _int_array(t_ms, "t_ms"), side="right").astype(np.int64) - 1


def team_gold_difference(gold_team) -> np.ndarray:
    """Per frame blue - red cached team gold, gold_team[F, 0] - gold_team[F, 1], computed in float64."""

    gold = np.asarray(gold_team)
    if gold.ndim != 2 or gold.shape[1] != 2:
        raise ValueError("gold_team must be [frames, 2] (team_order 100, 200)")
    diff = gold[:, 0].astype(np.float64) - gold[:, 1].astype(np.float64)
    if not np.isfinite(diff).all():
        raise ValueError("a frame has no finite team gold")
    return diff


def window_values(start_ms, D_ms, frames_ms, kills_ms, gold_by_frame) -> dict[str, np.ndarray]:
    """The continuous quantities behind the features of windows [start, start + D].

    start_frame (index of F(start), -1 none), start_age_ms (-1 without a frame), n_bound, end_age_ms (-1 without a
    frame), kill_gap_ms (start - last kill before the start; -1 without an earlier kill) and gold (at F(start); NaN
    without a frame).
    """

    start, D = _int_array(start_ms, "start_ms"), _int_array(D_ms, "D_ms")
    if start.shape != D.shape:
        raise ValueError("one D per start")
    if np.any(D < 0):
        raise ValueError("a window length is negative")
    frames, kills = _frames(frames_ms), _kills(kills_ms)
    gold = np.asarray(gold_by_frame, dtype=np.float64).reshape(-1)
    if len(gold) != len(frames):
        raise ValueError("one gold difference per frame")
    end = start + D
    first = np.searchsorted(frames, start, side="right").astype(np.int64) - 1
    last = np.searchsorted(frames, end, side="right").astype(np.int64) - 1
    held = first >= 0
    start_age = np.where(held, start - frames[np.maximum(first, 0)], -1)
    end_age = np.where(last >= 0, end - frames[np.maximum(last, 0)], -1)
    if len(kills):
        previous = np.searchsorted(kills, start, side="left").astype(np.int64) - 1
        kill_gap = np.where(previous >= 0, start - kills[np.maximum(previous, 0)], -1)
    else:
        kill_gap = np.full(len(start), -1, dtype=np.int64)
    return {"start_frame": first, "start_age_ms": start_age, "n_bound": last - first, "end_age_ms": end_age,
            "kill_gap_ms": kill_gap, "gold": np.where(held, gold[np.maximum(first, 0)], np.nan)}


def start_age_bin(start_age_ms) -> np.ndarray:
    age = np.asarray(start_age_ms, dtype=np.int64)
    inside = (age >= 0) & (age <= START_AGE_MAX_MS)
    return np.where(inside, (age > START_AGE_SPLIT_MS).astype(np.int64), -1)


def end_age_bin(end_age_ms) -> np.ndarray:
    age = np.asarray(end_age_ms, dtype=np.int64)
    return np.where(age < 0, -1, np.searchsorted(np.asarray(END_AGE_EDGES_MS, np.int64), age, side="right"))


def time_bin(start_ms, time_edges_min: Sequence[float] = TIME_EDGES_MIN) -> np.ndarray:
    return np.digitize(np.asarray(start_ms) / 60000.0, time_edges_min).astype(np.int64)


def kill_gap_bin(kill_gap_ms) -> np.ndarray:
    gap = np.asarray(kill_gap_ms, dtype=np.int64)
    return np.where(gap < 0, len(KILL_GAP_EDGES_MS),
                    np.searchsorted(np.asarray(KILL_GAP_EDGES_MS, np.int64), gap, side="right")).astype(np.int64)


def _edges(gold_edges) -> np.ndarray:
    edges = np.asarray(gold_edges, dtype=np.float64)
    if edges.shape != (N_TIME_BINS, len(GOLD_QUANTILES)):
        raise ValueError(f"gold edges must be a {N_TIME_BINS} x {len(GOLD_QUANTILES)} array (four per time bin)")
    if not np.isfinite(edges).all() or np.any(np.diff(edges, axis=1) < 0):
        raise ValueError("gold edges must be finite and non-decreasing within a time bin")
    return edges


def gold_bin(gold, tbin, gold_edges) -> np.ndarray:
    """np.searchsorted(edges[time_bin], gold, side="right") per window; -1 where the gold value is missing."""

    edges, gold, tbin = _edges(gold_edges), np.asarray(gold, dtype=np.float64), np.asarray(tbin, dtype=np.int64)
    if gold.shape != tbin.shape:
        raise ValueError("one time bin per gold value")
    if np.any((tbin < 0) | (tbin >= N_TIME_BINS)):
        raise ValueError("time bin outside 0-2")
    out = np.full(gold.shape, -1, dtype=np.int64)
    for b in range(N_TIME_BINS):
        rows = (tbin == b) & np.isfinite(gold)
        out[rows] = np.searchsorted(edges[b], gold[rows], side="right")
    return out


def keys_from_values(values: Mapping[str, np.ndarray], start_ms, gold_edges,
                     time_edges_min: Sequence[float] = TIME_EDGES_MIN) -> dict[str, np.ndarray]:
    """The six matching features from ``window_values`` (same order as FEATURES)."""

    tbin = time_bin(start_ms, time_edges_min)
    return {"start_age_bin": start_age_bin(values["start_age_ms"]), "n_bound": np.asarray(values["n_bound"], np.int64),
            "end_age_bin": end_age_bin(values["end_age_ms"]), "time_bin": tbin,
            "kill_gap_bin": kill_gap_bin(values["kill_gap_ms"]), "gold_bin": gold_bin(values["gold"], tbin, gold_edges)}


def window_keys(start_ms, D_ms, frames_ms, kills_ms, gold_by_frame, gold_edges,
                time_edges_min: Sequence[float] = TIME_EDGES_MIN) -> dict[str, np.ndarray]:
    """start_age_bin, n_bound, end_age_bin, time_bin, kill_gap_bin, gold_bin of windows [start, start + D]."""

    values = window_values(start_ms, D_ms, frames_ms, kills_ms, gold_by_frame)
    return keys_from_values(values, _int_array(start_ms, "start_ms"), gold_edges, time_edges_min)


def gold_edges(gold, tbin, quantiles: Sequence[float] = GOLD_QUANTILES) -> np.ndarray:
    """Per time bin (rows 0-2) the quantiles of the gold differences (numpy.quantile, method="linear")."""

    gold, tbin = np.asarray(gold, dtype=np.float64).reshape(-1), np.asarray(tbin, dtype=np.int64).reshape(-1)
    if gold.shape != tbin.shape:
        raise ValueError("one time bin per gold value")
    if not np.isfinite(gold).all():
        raise ValueError("a gold difference is not finite")
    out = np.empty((N_TIME_BINS, len(quantiles)), dtype=np.float64)
    for b in range(N_TIME_BINS):
        values = gold[tbin == b]
        if not len(values):
            raise ValueError(f"time bin {b} has no rows")
        out[b] = np.quantile(values, list(quantiles), method="linear")
    return out
