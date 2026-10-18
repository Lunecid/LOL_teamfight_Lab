from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def team_of_pid(pid: int, tm: Dict[int, int]) -> int:
    pid = int(pid or 0)
    tid = int(tm.get(pid, 0) or 0) if isinstance(tm, dict) else 0
    if tid in (100, 200):
        return tid
    if 1 <= pid <= 5:
        return 100
    if 6 <= pid <= 10:
        return 200
    return 0


def gold_team_at_ms(cache: Optional[Dict[str, Any]], q_ms: int, *, cfg_obj: Optional[Any] = None) -> Optional[np.ndarray]:
    if not isinstance(cache, dict):
        return None

    ts_raw = cache.get("minute_ts", None)
    g_raw = cache.get("gold_team_minute", None)

    try:
        ts = np.asarray(ts_raw, dtype=np.int64)
        g = np.asarray(g_raw, dtype=np.float32)
    except Exception:
        return None

    if ts.ndim != 1 or ts.size <= 0:
        return None
    if g.ndim != 2 or g.shape[0] != ts.size or g.shape[1] < 2:
        return None

    if ts.size == 1:
        return g[0, :2].astype(np.float32, copy=False)

    q = int(q_ms)
    idx = int(np.searchsorted(ts, q, side="right") - 1)
    i = int(np.clip(idx, 0, ts.size - 1))
    j = int(np.clip(i + 1, 0, ts.size - 1))

    method = "ffill"
    if cfg_obj is not None:
        method = str(
            getattr(
                cfg_obj,
                "LABEL_GOLD_METHOD",
                getattr(cfg_obj, "INTERP_SCALARS_METHOD", "ffill"),
            )
        ).lower().strip()

    if method in ("bfill",):
        return g[j, :2].astype(np.float32, copy=False)

    if method in ("linear",) and j > i and int(ts[j]) != int(ts[i]):
        alpha = float(q - int(ts[i])) / float(int(ts[j]) - int(ts[i]))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return ((1.0 - alpha) * g[i, :2] + alpha * g[j, :2]).astype(np.float32)

    return g[i, :2].astype(np.float32, copy=False)


def window_resource_changes(events: List[dict], tm: Dict[int, int], t0: int, t1_exclusive: int) -> Dict[str, Any]:
    tower_blue = tower_red = 0
    plate_blue = plate_red = 0
    inhib_blue = inhib_red = 0
    obj_by_type = {
        "dragon": {"blue": 0, "red": 0},
        "baron": {"blue": 0, "red": 0},
        "herald": {"blue": 0, "red": 0},
        "atakhan": {"blue": 0, "red": 0},
        "horde": {"blue": 0, "red": 0},
        "other": {"blue": 0, "red": 0},
    }

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        try:
            ts = int(ev.get("timestamp", ev.get("ts", -1)) or -1)
        except Exception:
            ts = -1
        if ts < int(t0) or ts >= int(t1_exclusive):
            continue

        et = str(ev.get("type", ev.get("eventType", ""))).upper()
        if et == "ELITE_MONSTER_KILL":
            team = int(ev.get("killerTeamId", 0) or 0)
            if team not in (100, 200):
                team = team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if team not in (100, 200):
                continue

            mt = str(ev.get("monsterType", "")).upper()
            key = "other"
            if mt == "DRAGON":
                key = "dragon"
            elif mt == "BARON_NASHOR":
                key = "baron"
            elif mt == "RIFTHERALD":
                key = "herald"
            elif mt == "ATAKHAN":
                key = "atakhan"
            elif mt == "HORDE":
                key = "horde"

            side = "blue" if team == 100 else "red"
            obj_by_type[key][side] += 1

        elif et == "BUILDING_KILL":
            bt = str(ev.get("buildingType", "")).upper()
            victim_team = int(ev.get("teamId", 0) or 0)
            if victim_team in (100, 200):
                taker_team = 100 if victim_team == 200 else 200
            else:
                taker_team = team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if taker_team not in (100, 200):
                continue

            if "TOWER" in bt:
                if taker_team == 100:
                    tower_blue += 1
                else:
                    tower_red += 1
            elif "INHIBITOR" in bt:
                if taker_team == 100:
                    inhib_blue += 1
                else:
                    inhib_red += 1

        elif et == "TURRET_PLATE_DESTROYED":
            victim_team = int(ev.get("teamId", 0) or 0)
            if victim_team in (100, 200):
                taker_team = 100 if victim_team == 200 else 200
            else:
                taker_team = team_of_pid(
                    int(ev.get("killerId", 0) or ev.get("participantId", 0) or 0),
                    tm,
                )
            if taker_team == 100:
                plate_blue += 1
            elif taker_team == 200:
                plate_red += 1

    obj_blue = int(sum(int(v["blue"]) for v in obj_by_type.values()))
    obj_red = int(sum(int(v["red"]) for v in obj_by_type.values()))

    return {
        "tower_blue": int(tower_blue),
        "tower_red": int(tower_red),
        "tower_diff": int(tower_blue - tower_red),
        "plate_blue": int(plate_blue),
        "plate_red": int(plate_red),
        "plate_diff": int(plate_blue - plate_red),
        "inhib_blue": int(inhib_blue),
        "inhib_red": int(inhib_red),
        "inhib_diff": int(inhib_blue - inhib_red),
        "objective_blue": int(obj_blue),
        "objective_red": int(obj_red),
        "objective_diff": int(obj_blue - obj_red),
        "objective_by_type": {
            k: {
                "blue": int(v["blue"]),
                "red": int(v["red"]),
                "diff": int(int(v["blue"]) - int(v["red"])),
            }
            for k, v in obj_by_type.items()
        },
    }
