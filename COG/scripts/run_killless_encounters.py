"""How much of the phenomenon does kill-anchored detection miss?

The CoG 2026 meta-reviewer asked what happens to teamfights that produce no
kills -- fights won by draining cooldowns or taking map control.  The corpus
is kill-anchored by construction, so those fights are absent.  This script
counts *proximity encounters* with and without a kill.  It computes no label
and fits no model.

What the code does:

* Matches: the ids of every ``*.meta.json`` in ``core.config.CACHE_DIR``,
  sorted.  When ``--n-matches`` is non-zero and smaller than that list,
  ``random.Random(--seed).sample`` draws that many ids and the draw is sorted
  again.  A match whose cache does not load, that has fewer than three minute
  frames or an empty team, or whose preparation or scan raises, is skipped
  silently and not counted in ``matches``.
* Positions: ``gameplay.fight_clustering.build_5s_position_grid`` with the
  process ``cfg``.  Minute-frame positions are interpolated onto a grid of
  ``cfg.TF2_GRID_STEP_MS`` (5,000 ms by default), and while
  ``cfg.TF2_USE_KILL_TRAJECTORY_INTERP`` is on (the default) every kill
  participant is moved towards its kill position.  Encounters around kills are
  therefore scanned on kill-adjusted positions, kill-less ones mostly on frame
  interpolation.
* Alive: the ``alive`` node feature of the last minute frame at or before each
  grid timestamp.  If the node layout has no ``alive`` column, or reading it
  fails, every champion counts as alive.
* Active grid point: every alive champion of either team is tried as the
  anchor, and the point is active when some anchor has at least
  ``--min-per-team`` alive champions of EACH team, itself included, within
  ``--radius`` game units of its own position.  There is no joint centroid.
  ``--radius`` is a flag (default 1,800 u; the v3.3 grid passes 1,200, 1,600
  or 2,000 u); when the cached coordinates are detected as normalised it is
  divided by the scale that ``detect_coordinate_scale`` returns.
* Encounter: a run of consecutive active grid points holding at least
  ``max(1, round(--min-duration * 1000 / GRID_STEP_MS))`` points, with
  ``GRID_STEP_MS`` = 5,000 ms: 3 points for 13.7 s, 4 for 20 s, 2 for the
  default 10 s.  ``start_ms`` and ``end_ms`` are the first and last active grid
  timestamps, so a 3-point run spans 10 s.
* Kill test: ``has_kill`` is true when any CHAMPION_KILL of the match, anywhere
  on the map, has a timestamp in ``[start_ms - --grace-ms, end_ms + --grace-ms]``.
  The grace window extends the run on both sides, not only after it
  (default 10,000 ms; the v3.3 grid passes 10,000 or 15,000 ms).

This is not the detector's presence gate, which counts alive champions within
R of the first kill's position at the cutoff tau = first kill - B.

Reported (JSON at ``--output``): the four flags, ``matches``, ``encounters``,
``with_kill``, ``killless``, ``killless_share`` (kill-less share of
encounters), ``encounters_per_match`` (per scanned match) and
``killless_median_duration_s`` (median ``end_ms - start_ms`` of kill-less
encounters).  The share is a share of proximity encounters, not of corpus
engagements.  Proximity is not commitment, and the kill-adjusted positions
favour encounters with a kill, so the share bounds the fights the corpus omits
in neither direction.  ``scripts/run_killless_v33.py`` wraps this scanner for
the v3.3 analyses (frame-only track, match bootstrap intervals, corpus
comparison).

Example, one row of the v3.3 grid
(``D:/LOL_Project/fusion_2615/features/tog_revision/killless_grid/summary.json``):

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_killless_encounters.py ^
        --n-matches 20000 --seed 7 --radius 1600 --min-per-team 4 ^
        --min-duration 13.7 --grace-ms 15000 ^
        --output D:/LOL_Project/fusion_2615/features/tog_revision/killless_grid/r1600_t4_dG_g15.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GRID_STEP_MS = 5000


def encounters_for_match(
    xy_dense: np.ndarray,
    dense_ts: np.ndarray,
    alive: np.ndarray | None,
    blue: np.ndarray,
    red: np.ndarray,
    kill_ts: np.ndarray,
    radius: float,
    min_per_team: int,
    min_duration_s: float,
    grace_ms: int,
) -> list[dict]:
    """Contiguous runs of frames holding a localized both-team cluster.

    The detector anchors its validity test at the first kill's position.  A
    kill-less encounter has no such anchor, so every alive champion is tried
    as one and the best is taken: a frame is active when some champion has at
    least ``min_per_team`` alive champions of *each* side within ``radius``.
    Using the centroid of all survivors instead would place the anchor near
    mid-lane whenever the teams are spread across the map.
    """
    n_frames = len(dense_ts)
    active = np.zeros(n_frames, dtype=bool)
    r_sq = radius * radius
    for t in range(n_frames):
        blue_pts = np.array(
            [xy_dense[t, int(p), :2] for p in blue
             if alive is None or bool(alive[t, int(p)])], dtype=float
        ).reshape(-1, 2)
        red_pts = np.array(
            [xy_dense[t, int(p), :2] for p in red
             if alive is None or bool(alive[t, int(p)])], dtype=float
        ).reshape(-1, 2)
        if len(blue_pts) < min_per_team or len(red_pts) < min_per_team:
            continue
        anchors = np.vstack([blue_pts, red_pts])
        d_blue = ((anchors[:, None, :] - blue_pts[None, :, :]) ** 2).sum(axis=2)
        d_red = ((anchors[:, None, :] - red_pts[None, :, :]) ** 2).sum(axis=2)
        blue_near = (d_blue <= r_sq).sum(axis=1)
        red_near = (d_red <= r_sq).sum(axis=1)
        active[t] = bool(
            np.any((blue_near >= min_per_team) & (red_near >= min_per_team))
        )

    out: list[dict] = []
    t = 0
    min_frames = max(1, int(round(min_duration_s * 1000 / GRID_STEP_MS)))
    while t < n_frames:
        if not active[t]:
            t += 1
            continue
        start = t
        while t < n_frames and active[t]:
            t += 1
        if t - start < min_frames:
            continue
        t0, t1 = int(dense_ts[start]), int(dense_ts[t - 1])
        has_kill = bool(np.any((kill_ts >= t0 - grace_ms) & (kill_ts <= t1 + grace_ms)))
        out.append({"start_ms": t0, "end_ms": t1, "has_kill": has_kill})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-matches", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--radius", type=float, default=1800.0)
    parser.add_argument("--min-per-team", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=10.0)
    parser.add_argument("--grace-ms", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    from core.contract import NODE_IDX
    from data.cache_io import load_match_cache
    from gameplay.fight_clustering import build_5s_position_grid
    from gameplay.fights import _extract_kill_events, detect_coordinate_scale

    mids = sorted(p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json"))
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))

    alive_idx = NODE_IDX.get("alive", None)
    totals = {"matches": 0, "encounters": 0, "with_kill": 0, "killless": 0}
    per_match: list[int] = []
    durations_killless: list[float] = []

    for done, mid in enumerate(mids, 1):
        pack = load_match_cache(mid)
        if not pack:
            continue
        try:
            xy = pack.get("xy_raw_minute")
            if xy is None:
                xy = pack["node_minute"][:, :, [NODE_IDX.get("x_norm", 0), NODE_IDX.get("y_norm", 1)]]
            minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
            if len(minute_ts) < 3:
                continue
            team_map = pack["meta"]["team_map"]
            blue = np.array([p - 1 for p, t in team_map.items() if int(t) == 100], dtype=int)
            red = np.array([p - 1 for p, t in team_map.items() if int(t) == 200], dtype=int)
            if len(blue) == 0 or len(red) == 0:
                continue
            events = pack.get("events", [])
            kill_events = _extract_kill_events(events)
            kill_ts = np.array([int(k["timestamp"]) for k in kill_events], dtype=np.int64)
            dense_ts, xy_dense = build_5s_position_grid(xy, minute_ts, kill_events, team_map, cfg_obj=cfg)

            is_norm = bool(pack.get("meta", {}).get("anchor_is_norm", False))
            scale = 1.0
            if not is_norm:
                is_norm, scale = detect_coordinate_scale(xy)
            radius = args.radius / scale if (is_norm and scale > 0) else args.radius

            alive = None
            if alive_idx is not None:
                try:
                    alive_minute = np.asarray(pack["node_minute"][:, :, alive_idx]) > 0.5
                    idx = np.clip(np.searchsorted(minute_ts, dense_ts, side="right") - 1,
                                  0, len(minute_ts) - 1)
                    alive = alive_minute[idx]
                except Exception:
                    alive = None

            found = encounters_for_match(
                xy_dense, dense_ts, alive, blue, red, kill_ts,
                radius, args.min_per_team, args.min_duration, args.grace_ms,
            )
        except Exception:
            continue

        totals["matches"] += 1
        totals["encounters"] += len(found)
        killless = [e for e in found if not e["has_kill"]]
        totals["with_kill"] += len(found) - len(killless)
        totals["killless"] += len(killless)
        per_match.append(len(found))
        durations_killless.extend((e["end_ms"] - e["start_ms"]) / 1000.0 for e in killless)
        if done % 200 == 0:
            share = totals["killless"] / max(1, totals["encounters"])
            print(f"{done}/{len(mids)} matches | encounters={totals['encounters']}"
                  f" killless={share*100:.1f}%", flush=True)

    share = totals["killless"] / max(1, totals["encounters"])
    results = {
        "radius": args.radius,
        "min_per_team": args.min_per_team,
        "min_duration_s": args.min_duration,
        "grace_ms": args.grace_ms,
        **totals,
        "killless_share": share,
        "encounters_per_match": float(np.mean(per_match)) if per_match else 0.0,
        "killless_median_duration_s": float(np.median(durations_killless)) if durations_killless else 0.0,
    }
    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
