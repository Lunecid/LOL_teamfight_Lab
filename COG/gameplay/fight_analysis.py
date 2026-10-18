from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


LabelEndFn = Callable[[dict, int], int]
GetHorizonFn = Callable[[], int]
MapTsToMinuteIdxFn = Callable[[np.ndarray, int], int]
TeamOfPidFn = Callable[[int, Dict[int, int]], int]
GoldAtMsFn = Callable[[Optional[Dict[str, Any]], int], Optional[np.ndarray]]
WindowResourceChangesFn = Callable[[List[dict], Dict[int, int], int, int], Dict[str, Any]]


def compute_fight_outcome_impl(
    fight: dict,
    kill_events: List[dict],
    tm: Dict[int, int],
    *,
    label_end_ts_fn: LabelEndFn,
    get_horizon_ms_fn: GetHorizonFn,
    node_idx: Dict[str, int],
    map_ts_to_minute_idx_fn: MapTsToMinuteIdxFn,
    team_of_pid_fn: TeamOfPidFn,
    gold_team_at_ms_fn: GoldAtMsFn,
    window_resource_changes_fn: WindowResourceChangesFn,
    cache: Optional[Dict[str, Any]] = None,
    events: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    engage_ts = int(fight["engage_ts"])
    horizon_end = label_end_ts_fn(fight, get_horizon_ms_fn())

    kills_in_fight = [k for k in kill_events if engage_ts <= int(k["timestamp"]) < horizon_end]

    blue_kills = red_kills = blue_deaths = red_deaths = blue_assists = red_assists = 0
    blue_dead_unique: set = set()
    red_dead_unique: set = set()

    for kill in kills_in_fight:
        killer_team = tm.get(kill.get("killer_id", 0), 0)
        victim_team = tm.get(kill.get("victim_id", 0), 0)

        if killer_team == 100:
            blue_kills += 1
        elif killer_team == 200:
            red_kills += 1

        if victim_team == 100:
            blue_deaths += 1
            blue_dead_unique.add(int(kill.get("victim_id", 0) or 0))
        elif victim_team == 200:
            red_deaths += 1
            red_dead_unique.add(int(kill.get("victim_id", 0) or 0))

        for assist_id in kill.get("assisting_ids", []) or []:
            assist_team = tm.get(assist_id, 0)
            if assist_team == 100:
                blue_assists += 1
            elif assist_team == 200:
                red_assists += 1

    kill_diff = blue_kills - red_kills
    winner = "blue" if kill_diff > 0 else ("red" if kill_diff < 0 else "draw")

    blue_survivors = int(max(0, 5 - len(blue_dead_unique)))
    red_survivors = int(max(0, 5 - len(red_dead_unique)))
    blue_alive_end = int(blue_survivors)
    red_alive_end = int(red_survivors)

    try:
        if isinstance(cache, dict):
            minute_ts = np.asarray(cache.get("minute_ts", []), dtype=np.int64)
            nm = cache.get("node_minute", None)
            alive_idx = node_idx.get("alive", None)
            if (
                alive_idx is not None
                and isinstance(nm, np.ndarray)
                and nm.ndim == 3
                and minute_ts.ndim == 1
                and minute_ts.size > 0
                and nm.shape[0] == minute_ts.size
                and nm.shape[1] >= 10
                and int(alive_idx) < nm.shape[2]
            ):
                m_idx = map_ts_to_minute_idx_fn(minute_ts, max(0, int(horizon_end) - 1))
                alive_vec = nm[m_idx, :, int(alive_idx)].astype(np.float32)
                b_alive = 0.0
                r_alive = 0.0
                for pid in range(1, 11):
                    tid = team_of_pid_fn(pid, tm)
                    if tid == 100:
                        b_alive += float(alive_vec[pid - 1])
                    elif tid == 200:
                        r_alive += float(alive_vec[pid - 1])
                blue_alive_end = int(np.clip(round(b_alive), 0, 5))
                red_alive_end = int(np.clip(round(r_alive), 0, 5))
                blue_survivors = int(blue_alive_end)
                red_survivors = int(red_alive_end)
    except Exception:
        pass

    gold_blue_delta = 0.0
    gold_red_delta = 0.0
    gold_diff = 0.0
    g0 = gold_team_at_ms_fn(cache, int(engage_ts))
    g1 = gold_team_at_ms_fn(cache, max(int(engage_ts), int(horizon_end) - 1))
    if g0 is not None and g1 is not None:
        gold_blue_delta = float(g1[0] - g0[0])
        gold_red_delta = float(g1[1] - g0[1])
        gold_diff = float(gold_blue_delta - gold_red_delta)

    res = window_resource_changes_fn(events or [], tm, int(engage_ts), int(horizon_end))

    return {
        "winner": winner,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "blue_deaths": blue_deaths,
        "red_deaths": red_deaths,
        "kill_diff": kill_diff,
        "total_kills": len(kills_in_fight),
        "assists": {"blue": blue_assists, "red": red_assists},
        "blue_unique_deaths": int(len(blue_dead_unique)),
        "red_unique_deaths": int(len(red_dead_unique)),
        "blue_survivors": int(blue_survivors),
        "red_survivors": int(red_survivors),
        "blue_alive_end": int(blue_alive_end),
        "red_alive_end": int(red_alive_end),
        "gold_blue_delta": float(gold_blue_delta),
        "gold_red_delta": float(gold_red_delta),
        "gold_diff": float(gold_diff),
        "tower_blue": int(res.get("tower_blue", 0)),
        "tower_red": int(res.get("tower_red", 0)),
        "tower_diff": int(res.get("tower_diff", 0)),
        "plate_blue": int(res.get("plate_blue", 0)),
        "plate_red": int(res.get("plate_red", 0)),
        "plate_diff": int(res.get("plate_diff", 0)),
        "inhib_blue": int(res.get("inhib_blue", 0)),
        "inhib_red": int(res.get("inhib_red", 0)),
        "inhib_diff": int(res.get("inhib_diff", 0)),
        "objective_blue": int(res.get("objective_blue", 0)),
        "objective_red": int(res.get("objective_red", 0)),
        "objective_diff": int(res.get("objective_diff", 0)),
        "objective_by_type": res.get("objective_by_type", {}),
    }


def compute_player_engagement_impl(
    fight: dict,
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    R: float,
    b: np.ndarray,
    r: np.ndarray,
    *,
    label_end_ts_fn: LabelEndFn,
    get_horizon_ms_fn: GetHorizonFn,
) -> List[Dict[str, Any]]:
    engage_ts = int(fight["engage_ts"])
    horizon_end = label_end_ts_fn(fight, get_horizon_ms_fn())

    Td = len(dense_ts)
    start_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="left"), 0, Td - 1))
    end_idx = int(np.clip(np.searchsorted(dense_ts, horizon_end, side="left"), start_idx + 1, Td))

    end_idx = min(end_idx, len(dists))
    n_frames = max(1, end_idx - start_idx)

    dist_slice = dists[start_idx:end_idx]

    min_dist_b = np.min(dist_slice, axis=2)
    frames_engaged_b = np.sum(min_dist_b <= R, axis=0)
    avg_dist_b = np.mean(min_dist_b, axis=0)

    min_dist_r = np.min(dist_slice, axis=1)
    frames_engaged_r = np.sum(min_dist_r <= R, axis=0)
    avg_dist_r = np.mean(min_dist_r, axis=0)

    result: List[Dict[str, Any]] = []
    for i, player_idx in enumerate(b):
        result.append(
            {
                "player_idx": int(player_idx),
                "team": "blue",
                "engagement_ratio": float(frames_engaged_b[i]) / n_frames,
                "avg_dist_to_enemy": float(avg_dist_b[i]),
                "frames_engaged": int(frames_engaged_b[i]),
                "total_frames": n_frames,
            }
        )
    for j, player_idx in enumerate(r):
        result.append(
            {
                "player_idx": int(player_idx),
                "team": "red",
                "engagement_ratio": float(frames_engaged_r[j]) / n_frames,
                "avg_dist_to_enemy": float(avg_dist_r[j]),
                "frames_engaged": int(frames_engaged_r[j]),
                "total_frames": n_frames,
            }
        )
    return result


def compute_fight_importance_impl(
    fight: dict,
    outcome: Dict[str, Any],
    fight_type: str,
    game_duration_ms: int,
    *,
    type_scores: Dict[str, float],
) -> float:
    score = 0.0
    score += min(int(outcome.get("total_kills", 0) or 0) * 6, 30)

    if game_duration_ms > 0:
        progress = float(int(fight["engage_ts"])) / float(game_duration_ms)
        if progress > 0.7:
            score += 20
        elif progress > 0.5:
            score += 10

    score += type_scores.get(fight_type, 0)
    score += min(abs(int(outcome.get("kill_diff", 0) or 0)) * 5, 25)
    return float(min(score, 100.0))


def generate_fight_visualization_impl(
    fight: dict,
    xy_dense: np.ndarray,
    dists: np.ndarray,
    dense_ts: np.ndarray,
    prox_pairs: np.ndarray,
    kill_events: List[dict],
    b: np.ndarray,
    r: np.ndarray,
    R: float,
    *,
    label_end_ts_fn: LabelEndFn,
    get_horizon_ms_fn: GetHorizonFn,
    sample_interval: int = 5,
) -> Dict[str, Any]:
    engage_ts = int(fight["engage_ts"])
    horizon_end = label_end_ts_fn(fight, get_horizon_ms_fn())

    Td = len(dense_ts)
    start_idx = int(np.clip(np.searchsorted(dense_ts, engage_ts, side="left"), 0, Td - 1))
    end_idx = int(np.clip(np.searchsorted(dense_ts, horizon_end, side="left"), start_idx + 1, Td))

    trajectory: List[Dict[str, Any]] = []
    for d_idx in range(start_idx, end_idx, sample_interval):
        if d_idx >= Td:
            break
        positions: Dict[str, Any] = {}
        for i, player_idx in enumerate(b):
            positions[f"blue_{i}"] = {
                "x": float(xy_dense[d_idx, player_idx, 0]),
                "y": float(xy_dense[d_idx, player_idx, 1]),
            }
        for j, player_idx in enumerate(r):
            positions[f"red_{j}"] = {
                "x": float(xy_dense[d_idx, player_idx, 0]),
                "y": float(xy_dense[d_idx, player_idx, 1]),
            }
        trajectory.append(
            {
                "timestamp": int(dense_ts[d_idx]),
                "positions": positions,
                "prox_pairs": int(prox_pairs[d_idx]) if d_idx < len(prox_pairs) else 0,
            }
        )

    heatmap_points = [
        (float(xy_dense[d_idx, p, 0]), float(xy_dense[d_idx, p, 1]))
        for d_idx in range(start_idx, min(end_idx, Td))
        for p in range(10)
    ]

    engagement_timeline = [
        {
            "timestamp": int(dense_ts[d_idx]),
            "intensity": int(prox_pairs[d_idx]) if d_idx < len(prox_pairs) else 0,
            "normalized": float(prox_pairs[d_idx]) / 25.0 if d_idx < len(prox_pairs) else 0.0,
        }
        for d_idx in range(start_idx, min(end_idx, Td), sample_interval)
    ]

    kill_markers = [
        {
            "timestamp": int(kill["timestamp"]),
            "killer_id": kill.get("killer_id"),
            "victim_id": kill.get("victim_id"),
            "position": {"x": kill["position"][0], "y": kill["position"][1]} if kill.get("position") else None,
        }
        for kill in kill_events
        if engage_ts <= int(kill["timestamp"]) < horizon_end
    ]

    return {
        "trajectory": trajectory,
        "heatmap_points": heatmap_points,
        "engagement_timeline": engagement_timeline,
        "kill_markers": kill_markers,
    }
