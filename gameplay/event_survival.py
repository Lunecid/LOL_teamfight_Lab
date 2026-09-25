"""Kill-event survival: who is dead at time t, from CHAMPION_KILL / LEVEL_UP events only.

Ported from outputs/diag_survival_dbscan_20260925/scripts/p3_detection_arms.py:67-120
(death_intervals, event_alive, respawned_since_frame) with the module-level state `_S` removed:
the patch is an argument and the death timer comes from gameplay.respawn_rules.

A death interval is [t0, t1) in milliseconds: t0 = kill timestamp, t1 = t0 + patch death timer
(level = highest LEVEL_UP level with timestamp <= t0, 1 if none; game time = t0). A later death of
the same champion ends the earlier interval (early revive, e.g. timer over-estimate or a revive
effect): t1 = min(t1, t0_next).

Exactness. No frame 'alive'/status column, no position data and no 5-s grid is read.
* event_alive(intervals, t) and respawned_since(intervals, t0, t1) give the same answer whether the
  intervals were built from all events or only from events with timestamp <= t (resp. <= t1): an
  early-revive cut by a death after t only moves an end that is already > t.
* respawn_remaining_s(intervals, t) is NOT invariant: a cut by a future death would shorten the
  remaining time. For causal use build the intervals with death_intervals(events, patch, t=t).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from gameplay.respawn_rules import death_timer_s

Intervals = Dict[int, List[List[float]]]


def _check_patch(patch: str) -> str:
    """respawn_rules maps only Match-V5 majors 15/16 to public 25/26; a Match-V5 major > 16
    (public 27+) would be read as public 17-26 and silently accepted there, so refuse it here."""
    major = int(str(patch).split(".")[0])
    if major > 16:
        raise KeyError(f"no verified respawn rule for Match-V5 patch {patch}")
    return str(patch)


def death_intervals(events: Iterable[Mapping], patch: str, t: Optional[float] = None) -> Intervals:
    """{participantId 1..10: [[t0_ms, t1_ms], ...]} sorted by t0.

    events: Match-V5 timeline events (dicts with 'type', 'timestamp', ...), any order.
    patch: Match-V5 major.minor ('15.14'); out-of-range patches raise KeyError (respawn_rules).
    t: if given, only events with timestamp <= t are used (causal view at time t).
    """
    patch = _check_patch(patch)
    lvl: Dict[int, list] = {p: [] for p in range(1, 11)}
    kills = []
    for e in events or []:
        et = e.get("type")
        if et not in ("LEVEL_UP", "CHAMPION_KILL"):
            continue
        ts = int(e["timestamp"])
        if t is not None and ts > t:
            continue
        if et == "LEVEL_UP":
            p = int(e.get("participantId", 0) or 0)
            if 1 <= p <= 10:
                lvl[p].append((ts, int(e.get("level", 0) or 0)))
        else:
            v = int(e.get("victimId", 0) or 0)
            if 1 <= v <= 10:
                kills.append((ts, v))
    for p in lvl:
        lvl[p].sort()
    out: Intervals = {p: [] for p in range(1, 11)}
    for tk, v in sorted(kills):
        level = 1
        for tl, lv in lvl[v]:
            if tl <= tk:
                level = max(level, lv)
            else:
                break
        dur_s = death_timer_s(patch, level, tk / 1000.0)
        out[v].append([tk, tk + dur_s * 1000.0])
    for iv in out.values():  # a later death of the same champion ends the earlier interval
        for a, b in zip(iv[:-1], iv[1:]):
            a[1] = min(a[1], b[0])
    return out


def event_alive(intervals: Mapping[int, list], t: float) -> np.ndarray:
    """(10,) float: 0 for players with a death interval t0 <= t < t1, else 1 (index = pid - 1)."""
    a = np.ones(10, dtype=float)
    for p, iv in intervals.items():
        for t0, t1 in iv:
            if t0 <= t < t1:
                a[int(p) - 1] = 0.0
                break
    return a


def respawned_since(intervals: Mapping[int, list], t0: float, t1: float) -> np.ndarray:
    """(10,) float: 1 for players whose death interval ended in (t0, t1], else 0.

    With t0 = last minute-frame timestamp <= t1 (or -1 if none) this equals p3
    respawned_since_frame: players whose frame position predates their respawn.
    """
    out = np.zeros(10, dtype=float)
    for p, iv in intervals.items():
        for _a, end in iv:
            if t0 < end <= t1:
                out[int(p) - 1] = 1.0
                break
    return out


def respawn_remaining_s(intervals: Mapping[int, list], t: float) -> np.ndarray:
    """(10,) float seconds until respawn at t (0 for alive players).

    Causal only when `intervals` were built with death_intervals(..., t=t) (see module docstring).
    """
    out = np.zeros(10, dtype=float)
    for p, iv in intervals.items():
        for a, end in iv:
            if a <= t < end:
                out[int(p) - 1] = (end - t) / 1000.0
                break
    return out
