from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(math.sqrt(dx * dx + dy * dy))


def count_events_in_window(ts_sorted: np.ndarray, t0: int, t1_inclusive: int) -> int:
    """Count events in inclusive interval [t0, t1] for sorted timestamps."""
    if ts_sorted.size == 0:
        return 0
    l = int(np.searchsorted(ts_sorted, t0, side="left"))
    r = int(np.searchsorted(ts_sorted, t1_inclusive, side="right"))
    return max(0, r - l)


def label_end_ts(fight: dict, horizon_ms: int) -> int:
    """Compute label-end timestamp (exclusive)."""
    t0 = int(fight.get("engage_ts", -1))
    if t0 < 0:
        return int(horizon_ms)
    fallback = int(t0 + int(horizon_ms))
    try:
        hend = int(fight.get("horizon_end_ts", fallback))
    except Exception:
        hend = fallback
    if hend <= t0:
        return fallback
    return hend


def fight_priority_score(
    f: dict,
    *,
    kill_ts: np.ndarray,
    horizon_ms: int,
) -> float:
    """Observe-time-only priority for conflict resolution."""
    try:
        t0 = int(f.get("engage_ts", -1))
    except Exception:
        t0 = -1
    if t0 < 0:
        return -1e18

    prox = int(f.get("det_prox_pairs", 0) or 0)
    seg = int(f.get("n_segments", 1) or 1)
    anchor = int(f.get("det_anchor", 0) or 0)
    back = int(f.get("det_backtracked", 0) or 0)
    return float(100 * prox + 50 * anchor + 20 * seg + 10 * back)


def enforce_postmerge_spacing_and_nonoverlap(
    fights: List[dict],
    *,
    horizon_ms: int,
    fight_min_gap_ms: int,
    kill_ts: np.ndarray,
    location_radius: float = 0.0,
    diag: Optional[dict] = None,
) -> List[dict]:
    """Apply minimum start-gap and label-window non-overlap constraints post-merge.

    [FIX-6.2] When two fights overlap but are spatially distant, instead of
    blindly keeping both (which causes label overlap for sequential models),
    resolve by priority:
      - Higher-priority fight: kept unchanged.
      - Lower-priority (smaller) fight:
          * If smaller is the earlier fight: clip its horizon_end_ts to the
            later fight's engage_ts (eliminates temporal overlap).
          * If smaller is the later fight: drop it (its engage falls inside
            the bigger fight's label window, cannot be cleanly clipped).
    """
    if not fights:
        if diag is not None:
            diag.setdefault("postmerge_conflicts", 0)
            diag.setdefault("postmerge_removed", 0)
            diag.setdefault("postmerge_replaced", 0)
            diag.setdefault("postmerge_overlap_clipped", 0)
            diag.setdefault("postmerge_overlap_dropped", 0)
        return fights

    fs = sorted(fights, key=lambda x: int(x.get("engage_ts", -1)))
    kept: List[dict] = []

    conflicts = 0
    removed = 0
    replaced = 0
    overlap_clipped = 0
    overlap_dropped = 0

    for f in fs:
        t0 = int(f.get("engage_ts", -1))
        if t0 < 0:
            continue
        if not kept:
            kept.append(f)
            continue

        prev = kept[-1]
        p0 = int(prev.get("engage_ts", -1))
        if p0 < 0:
            kept[-1] = f
            continue

        prev_label_end = label_end_ts(prev, horizon_ms)
        gap_from_prev_start = t0 - p0

        overlap = (t0 < prev_label_end)
        too_close = (gap_from_prev_start < int(fight_min_gap_ms))

        if overlap or too_close:
            # [FIX-6.2] Spatially distant overlap → clip smaller fight
            if overlap and float(location_radius) > 0:
                try:
                    pcx = float(prev.get("centroid_x", float("nan")))
                    pcy = float(prev.get("centroid_y", float("nan")))
                    ccx = float(f.get("centroid_x", float("nan")))
                    ccy = float(f.get("centroid_y", float("nan")))
                    if np.isfinite(pcx) and np.isfinite(pcy) and np.isfinite(ccx) and np.isfinite(ccy):
                        if distance_2d((pcx, pcy), (ccx, ccy)) > float(location_radius):
                            sp = fight_priority_score(prev, kill_ts=kill_ts, horizon_ms=horizon_ms)
                            sc = fight_priority_score(f, kill_ts=kill_ts, horizon_ms=horizon_ms)
                            if sc > sp:
                                # f is bigger — clip prev (smaller, earlier)
                                clipped_end = int(min(prev_label_end, t0))
                                if clipped_end > p0:
                                    prev["horizon_end_ts"] = clipped_end
                                    prev["det_overlap_clipped"] = 1
                                    overlap_clipped += 1
                                kept.append(f)
                            else:
                                # prev is bigger — drop f (smaller, later)
                                overlap_dropped += 1
                            continue
                except Exception:
                    pass

            conflicts += 1
            sp = fight_priority_score(prev, kill_ts=kill_ts, horizon_ms=horizon_ms)
            sc = fight_priority_score(f, kill_ts=kill_ts, horizon_ms=horizon_ms)
            if sc > sp:
                kept[-1] = f
                replaced += 1
            else:
                removed += 1
            continue

        kept.append(f)

    if diag is not None:
        diag["postmerge_conflicts"] = int(diag.get("postmerge_conflicts", 0) or 0) + int(conflicts)
        diag["postmerge_removed"] = int(diag.get("postmerge_removed", 0) or 0) + int(removed)
        diag["postmerge_replaced"] = int(diag.get("postmerge_replaced", 0) or 0) + int(replaced)
        diag["postmerge_overlap_clipped"] = int(diag.get("postmerge_overlap_clipped", 0) or 0) + int(overlap_clipped)
        diag["postmerge_overlap_dropped"] = int(diag.get("postmerge_overlap_dropped", 0) or 0) + int(overlap_dropped)

    return kept
