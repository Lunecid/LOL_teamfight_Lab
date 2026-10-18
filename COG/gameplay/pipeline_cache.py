from __future__ import annotations

import hashlib

from core.config import (
    cfg,
    CHAMPION_STATS_KEYS,
    CS_DENOM,
    DAMAGE_STATS_KEYS,
    DS_DENOM,
    CHAMPION_STATS_DIV100_KEYS,
    F_GLOBAL,
    F_NODE,
    MAP_MAX,
    NODE_IDX,
    GLOBAL_IDX,
    DRAGON_SOUL_TYPES,
)
from core.common import Any, Dict, List, Optional, Tuple, np, safe_float


def _sum_feat(nm_t: np.ndarray, idxs: np.ndarray, name: str) -> float:
    j = NODE_IDX.get(name, None)
    if j is None:
        return 0.0
    return float(nm_t[idxs, j].sum())


def _mean_feat(nm_t: np.ndarray, idxs: np.ndarray, name: str) -> float:
    j = NODE_IDX.get(name, None)
    if j is None:
        return 0.0
    return float(nm_t[idxs, j].mean())


def build_global_minute_vector(
    t: int,
    T: int,
    nm_t: np.ndarray,
    team_map: Dict[int, int],
    history: Dict[str, np.ndarray],
    bans: Optional[Dict[str, List[int]]] = None,
) -> np.ndarray:
    tids = np.array([team_map.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
    b_idx = np.where(tids == 100)[0]
    r_idx = np.where(tids == 200)[0]

    time_norm = float(t / max(1.0, T - 1))
    gold_diff = _sum_feat(nm_t, b_idx, "totalGold_norm") - _sum_feat(nm_t, r_idx, "totalGold_norm")
    xp_diff = _sum_feat(nm_t, b_idx, "xp_norm") - _sum_feat(nm_t, r_idx, "xp_norm")
    lvl_diff = _mean_feat(nm_t, b_idx, "level_norm") - _mean_feat(nm_t, r_idx, "level_norm")
    lane_cs_diff = _sum_feat(nm_t, b_idx, "laneCS_norm") - _sum_feat(nm_t, r_idx, "laneCS_norm")
    jg_cs_diff = _sum_feat(nm_t, b_idx, "jgCS_norm") - _sum_feat(nm_t, r_idx, "jgCS_norm")
    cs_total_diff = lane_cs_diff + jg_cs_diff
    alive_diff = _sum_feat(nm_t, b_idx, "alive") - _sum_feat(nm_t, r_idx, "alive")

    h = {
        k: float(history.get(k, np.zeros((T,), np.float32))[t])
        for k in ["kill", "tower", "inhib", "dragon", "baron", "herald", "atakhan", "plate", "horde"]
    }

    alias = {
        "time_norm": time_norm,
        "t_norm": time_norm,
        "minute_norm": time_norm,
        "gold_diff": gold_diff,
        "goldDiff": gold_diff,
        "totalGold_diff": gold_diff,
        "totalGoldDiff": gold_diff,
        "xp_diff": xp_diff,
        "xpDiff": xp_diff,
        "level_diff": lvl_diff,
        "levelDiff": lvl_diff,
        "avgLevelDiff": lvl_diff,
        "laneCS_diff": lane_cs_diff,
        "laneCSDiff": lane_cs_diff,
        "jgCS_diff": jg_cs_diff,
        "jgCSDiff": jg_cs_diff,
        "csJgDiff": jg_cs_diff,
        "csDiff_total": cs_total_diff,
        "alive_diff": alive_diff,
        "aliveDiff": alive_diff,
        "kill_diff": h["kill"],
        "killDiff": h["kill"],
        "killDiff_cum": h["kill"],
        "tower_diff": h["tower"],
        "towerDiff": h["tower"],
        "towerDiff_cum": h["tower"],
        "inhib_diff": h["inhib"],
        "inhibDiff": h["inhib"],
        "inhibDiff_cum": h["inhib"],
        "dragon_diff": h["dragon"],
        "dragonDiff": h["dragon"],
        "dragonDiff_cum": h["dragon"],
        "baron_diff": h["baron"],
        "baronDiff": h["baron"],
        "baronDiff_cum": h["baron"],
        "herald_diff": h["herald"],
        "heraldDiff": h["herald"],
        "heraldDiff_cum": h["herald"],
        "atakhan_diff": h["atakhan"],
        "atakhanDiff": h["atakhan"],
        "atakhanDiff_cum": h["atakhan"],
        "plate_diff": h["plate"],
        "plateDiff": h["plate"],
        "plateDiff_cum": h["plate"],
        "horde_diff": h["horde"],
        "hordeDiff": h["horde"],
        "hordeDiff_cum": h["horde"],
    }

    out = np.zeros((F_GLOBAL,), dtype=np.float32)
    for name, j in GLOBAL_IDX.items():
        if name in alias:
            out[j] = float(alias[name])
            continue
        n = name.lower()
        if "time" in n and "norm" in n:
            out[j] = time_norm
        elif ("gold" in n) and ("diff" in n):
            out[j] = gold_diff
        elif ("xp" in n) and ("diff" in n):
            out[j] = xp_diff
        elif ("level" in n) and ("diff" in n):
            out[j] = lvl_diff
        elif (("lane" in n) or ("minion" in n)) and ("diff" in n):
            out[j] = lane_cs_diff
        elif ("jungle" in n or "jg" in n) and ("diff" in n):
            out[j] = jg_cs_diff
        elif ("cs" in n) and ("diff" in n):
            out[j] = cs_total_diff
        elif ("alive" in n) and ("diff" in n):
            out[j] = alive_diff
        else:
            for hk in ["kill", "tower", "inhib", "dragon", "baron", "herald", "atakhan", "plate", "horde"]:
                if hk in n and "diff" in n:
                    out[j] = float(h[hk])
                    break

    if isinstance(bans, dict):
        blue = bans.get("blue", [0, 0, 0, 0, 0])
        red = bans.get("red", [0, 0, 0, 0, 0])
        for k in range(5):
            jj = GLOBAL_IDX.get(f"blue_ban_{k}", None)
            if jj is not None:
                out[jj] = float(blue[k] if k < len(blue) else 0)
        for k in range(5):
            jj = GLOBAL_IDX.get(f"red_ban_{k}", None)
            if jj is not None:
                out[jj] = float(red[k] if k < len(red) else 0)

    return out.astype(np.float32)


def _dragon_subtype_to_soul(sub: str) -> Optional[str]:
    s = str(sub or "").upper()
    if not s:
        return None
    if "HEXTECH" in s:
        return "hextech"
    if "CHEMTECH" in s:
        return "chemtech"
    if "INFERNAL" in s or "FIRE" in s:
        return "infernal"
    if "OCEAN" in s or "WATER" in s:
        return "ocean"
    if "MOUNTAIN" in s or "EARTH" in s:
        return "mountain"
    if "CLOUD" in s or "WIND" in s or "AIR" in s:
        return "cloud"
    return None


def _is_elder_dragon(e: dict) -> bool:
    if not isinstance(e, dict):
        return False
    mt = str(e.get("monsterType", "")).upper()
    if mt != "DRAGON":
        return False
    sub = str(e.get("monsterSubType", "")).upper()
    return ("ELDER" in sub) or ("ELDER_DRAGON" in sub)


def _pf_xy(pf: dict) -> Tuple[float, float]:
    if not isinstance(pf, dict):
        return 0.0, 0.0
    pos = pf.get("position", None)
    if isinstance(pos, dict):
        return float(safe_float(pos.get("x", 0.0))), float(safe_float(pos.get("y", 0.0)))
    if "x" in pf and "y" in pf:
        return float(safe_float(pf.get("x", 0.0))), float(safe_float(pf.get("y", 0.0)))
    return 0.0, 0.0


def _stable_name_id(name: Any, vocab_size: int) -> int:
    s = str(name or "").strip().lower()
    if not s:
        return 0
    v = int(hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return int(v % max(1, int(vocab_size) - 1)) + 1


def _normalize_cs_raw(key: str, raw_value: float) -> float:
    v = float(raw_value)
    if key in CHAMPION_STATS_DIV100_KEYS and abs(v) > 2.0:
        return v / 100.0
    return v


def _extract_champ_runes_bans_from_detail(detail):
    from data.cache_io import _extract_static_meta_from_detail

    return _extract_static_meta_from_detail(detail)


def parse_timeline_to_minute_cache(
    timeline: dict,
    team_map: Dict[int, int],
    *,
    detail: Optional[Dict[str, Any]] = None,
    cfg_in: Any = None,
) -> Dict[str, Any]:
    if not isinstance(timeline, dict):
        return {}

    frames = timeline.get("info", {}).get("frames", []) or []
    if not isinstance(frames, list) or len(frames) == 0:
        return {}

    _cfg = cfg_in if cfg_in is not None else cfg

    T0 = len(frames)
    T = T0
    if getattr(_cfg, "MAX_T", None):
        T = min(T0, int(_cfg.MAX_T))
    if T <= 0:
        return {}

    minute_ts = np.zeros((T,), dtype=np.int64)
    node_minute = np.zeros((T, 10, F_NODE), dtype=np.float32)
    gold_team = np.zeros((T, 2), dtype=np.float32)
    global_minute = np.zeros((T, F_GLOBAL), dtype=np.float32)
    xy_raw_minute = np.zeros((T, 10, 2), dtype=np.float32)

    counters = {k: 0 for k in ["kill", "tower", "inhib", "dragon", "elder", "baron", "herald", "atakhan", "plate", "horde"]}
    history = {k: np.zeros(T, dtype=np.float32) for k in counters}

    all_events: List[dict] = []

    frame_ms = int(getattr(_cfg, "FRAME_MS", 60000))
    coord_div = float(getattr(_cfg, "COORD_NORM_DIV", MAP_MAX))

    use_status = bool(getattr(_cfg, "USE_STATUS_FEATURES", True))
    use_buff = bool(getattr(_cfg, "USE_BUFF_STATUS", True))
    use_soul = bool(getattr(_cfg, "USE_DRAGON_SOUL_STATUS", True))
    use_ult = bool(getattr(_cfg, "USE_ULT_LEVEL", True))

    buff_dur = dict(getattr(_cfg, "BUFF_DUR_SEC", {})) if hasattr(_cfg, "BUFF_DUR_SEC") else {}
    baron_dur_ms = int(buff_dur.get("baron", 180)) * 1000
    elder_dur_ms = int(buff_dur.get("elder", 150)) * 1000

    baron_end = np.zeros(11, dtype=np.int64)
    elder_end = np.zeros(11, dtype=np.int64)
    ult_lv = np.zeros(11, dtype=np.int32)

    team_dragon_cnt = {100: 0, 200: 0}
    team_soul = {100: None, 200: None}

    _meta = _extract_champ_runes_bans_from_detail(detail)
    champ_by_pid: Dict[int, int] = {int(k): int(v) for k, v in _meta.get("champion_by_pid", {}).items()}
    runes_by_pid: Dict[int, Dict[str, int]] = {int(k): v for k, v in _meta.get("runes_by_pid", {}).items()}
    bans: Dict[str, List[int]] = _meta.get("bans", {"blue": [0, 0, 0, 0, 0], "red": [0, 0, 0, 0, 0]})

    def _set(vec: np.ndarray, name: str, val: float):
        j = NODE_IDX.get(name, None)
        if j is not None:
            vec[j] = float(val)

    DEN_LEVEL = 18.0
    DEN_XP = 20000.0
    DEN_CUR_G = 4000.0
    DEN_TOT_G = 25000.0
    DEN_GPS = 30.0
    DEN_LANECS = 400.0
    DEN_JGCS = 250.0
    DEN_CCT = 600.0

    team_pids = {
        100: [pid for pid, tid in team_map.items() if tid == 100],
        200: [pid for pid, tid in team_map.items() if tid == 200],
    }

    for t in range(T):
        fr = frames[t] if isinstance(frames[t], dict) else {}
        t_ms = int(fr.get("timestamp", t * frame_ms) or (t * frame_ms))
        minute_ts[t] = t_ms

        pframes = fr.get("participantFrames", {}) or {}
        if not isinstance(pframes, dict):
            pframes = {}

        bg, rg = 0.0, 0.0

        for pid in range(1, 11):
            pf = pframes.get(str(pid), {}) or {}
            if not isinstance(pf, dict):
                pf = {}

            x_raw, y_raw = _pf_xy(pf)
            xy_raw_minute[t, pid - 1, 0] = float(x_raw)
            xy_raw_minute[t, pid - 1, 1] = float(y_raw)

            level = float(safe_float(pf.get("level", 0.0)))
            xp = float(safe_float(pf.get("xp", 0.0)))
            cur_g = float(safe_float(pf.get("currentGold", pf.get("currentgold", 0.0))))
            tot_g = float(safe_float(pf.get("totalGold", 0.0)))
            lane_cs = float(safe_float(pf.get("minionsKilled", 0.0)))
            jg_cs = float(safe_float(pf.get("jungleMinionsKilled", 0.0)))
            cc_t = float(safe_float(pf.get("timeEnemySpentControlled", 0.0)))

            cs = pf.get("championStats", {}) or {}
            ds = pf.get("damageStats", {}) or {}
            if not isinstance(cs, dict):
                cs = {}
            if not isinstance(ds, dict):
                ds = {}

            health = float(safe_float(cs.get("health", 0.0)))
            health_max = float(safe_float(cs.get("healthMax", 1.0)))
            power = float(safe_float(cs.get("power", 0.0)))
            power_max = float(safe_float(cs.get("powerMax", 1.0)))

            alive = 1.0 if health > 0.0 else 0.0
            hp_pct = 0.0 if health_max <= 0 else float(np.clip(health / health_max, 0.0, 1.0))
            mp_pct = 0.0 if power_max <= 0 else float(np.clip(power / power_max, 0.0, 1.0))
            time_sec = max(1.0, float(t_ms) / 1000.0)
            gps = float(np.clip(tot_g / time_sec, 0.0, 200.0))

            vec = np.zeros((F_NODE,), dtype=np.float32)
            if "champion_id" in NODE_IDX:
                _set(vec, "champion_id", float(champ_by_pid.get(pid, 0)))

            rmeta = runes_by_pid.get(pid, None)
            if isinstance(rmeta, dict):
                for rk, rv in rmeta.items():
                    if rk in NODE_IDX:
                        _set(vec, rk, float(int(rv)))

            _set(vec, "x_norm", float(np.clip(x_raw / max(1e-6, coord_div), 0.0, 2.0)))
            _set(vec, "y_norm", float(np.clip(y_raw / max(1e-6, coord_div), 0.0, 2.0)))
            _set(vec, "level_norm", float(np.clip(level / DEN_LEVEL, 0.0, 2.0)))
            _set(vec, "xp_norm", float(np.clip(xp / DEN_XP, 0.0, 5.0)))
            _set(vec, "curGold_norm", float(np.clip(cur_g / DEN_CUR_G, 0.0, 5.0)))
            _set(vec, "totalGold_norm", float(np.clip(tot_g / DEN_TOT_G, 0.0, 5.0)))
            _set(vec, "gps_norm", float(np.clip(gps / DEN_GPS, 0.0, 5.0)))
            _set(vec, "laneCS_norm", float(np.clip(lane_cs / DEN_LANECS, 0.0, 5.0)))
            _set(vec, "jgCS_norm", float(np.clip(jg_cs / DEN_JGCS, 0.0, 5.0)))
            _set(vec, "ccTime_norm", float(np.clip(cc_t / DEN_CCT, 0.0, 5.0)))
            _set(vec, "hp_pct", hp_pct)
            _set(vec, "mp_pct", mp_pct)
            _set(vec, "alive", alive)

            for k in CHAMPION_STATS_KEYS:
                denom = float(CS_DENOM.get(k, 1.0))
                raw_cs = _normalize_cs_raw(k, float(safe_float(cs.get(k, 0.0))))
                val = raw_cs / max(1e-6, denom)
                _set(vec, f"cs_{k}", float(np.clip(val, -10.0, 10.0)))

            for k in DAMAGE_STATS_KEYS:
                denom = float(DS_DENOM.get(k, 1.0))
                val = float(safe_float(ds.get(k, 0.0))) / max(1e-6, denom)
                _set(vec, f"ds_{k}", float(np.clip(val, -10.0, 10.0)))

            if use_status:
                tid = int(team_map.get(pid, 100 if pid <= 5 else 200))
                if use_buff:
                    hb = 1.0 if t_ms < int(baron_end[pid]) else 0.0
                    he = 1.0 if t_ms < int(elder_end[pid]) else 0.0
                    _set(vec, "has_baron", hb)
                    _set(vec, "has_elder", he)

                    def _remain(end_ts: int, dur_ms: int) -> float:
                        if dur_ms <= 0:
                            return 0.0
                        rem = float(max(0, int(end_ts) - int(t_ms))) / float(dur_ms)
                        return float(np.clip(rem, 0.0, 1.0))

                    _set(vec, "baron_remain_norm", _remain(int(baron_end[pid]), baron_dur_ms))
                    _set(vec, "elder_remain_norm", _remain(int(elder_end[pid]), elder_dur_ms))

                if use_soul:
                    soul = team_soul.get(tid, None)
                    for st in DRAGON_SOUL_TYPES:
                        _set(vec, f"soul_{st}", 1.0 if soul == st else 0.0)

                if use_ult:
                    _set(vec, "ult_level_norm", float(np.clip(float(ult_lv[pid]) / 3.0, 0.0, 1.0)))

            node_minute[t, pid - 1] = vec.astype(np.float32)

            if team_map.get(pid, 100 if pid <= 5 else 200) == 100:
                bg += tot_g
            else:
                rg += tot_g

        gold_team[t] = [bg, rg]
        evs = fr.get("events", []) or []
        if not isinstance(evs, list):
            evs = []

        for e in evs:
            if not isinstance(e, dict):
                continue
            if "timestamp" in e and "type" in e:
                all_events.append(e)

            et = str(e.get("type", "")).upper()
            if et == "CHAMPION_KILL":
                killer = int(e.get("killerId", 0) or 0)
                if killer in team_map:
                    counters["kill"] += (1 if team_map[killer] == 100 else -1)
            elif et == "ELITE_MONSTER_KILL":
                mt = str(e.get("monsterType", "")).upper()
                team = int(e.get("killerTeamId", 0) or 0)
                val = 1 if team == 100 else -1
                if mt == "DRAGON":
                    counters["dragon"] += val
                    if _is_elder_dragon(e):
                        counters["elder"] += val
                elif mt == "BARON_NASHOR":
                    counters["baron"] += val
                elif mt == "RIFTHERALD":
                    counters["herald"] += val
                elif mt == "ATAKHAN":
                    counters["atakhan"] += val
                elif mt == "HORDE":
                    counters["horde"] += val
            elif et == "BUILDING_KILL":
                bt = str(e.get("buildingType", "")).upper()
                lt = int(e.get("teamId", 0) or 0)
                if lt in (100, 200):
                    val = 1 if lt == 200 else -1
                    if "TOWER" in bt:
                        counters["tower"] += val
                    elif "INHIBITOR" in bt:
                        counters["inhib"] += val
            elif et == "TURRET_PLATE_DESTROYED":
                lt = int(e.get("teamId", 0) or 0)
                if lt in (100, 200):
                    counters["plate"] += (1 if lt == 200 else -1)

            if not use_status:
                continue

            try:
                ev_ts = int(e.get("timestamp", -1) or -1)
            except Exception:
                ev_ts = -1
            if ev_ts < 0:
                continue

            if et == "ELITE_MONSTER_KILL":
                mt = str(e.get("monsterType", "")).upper()
                team = int(e.get("killerTeamId", 0) or 0)
                if use_buff and mt == "BARON_NASHOR" and team in (100, 200):
                    end_ts = int(ev_ts) + int(baron_dur_ms)
                    for p in team_pids.get(team, []):
                        baron_end[p] = max(int(baron_end[p]), end_ts)

                if use_buff and _is_elder_dragon(e):
                    if team in (100, 200):
                        end_ts = int(ev_ts) + int(elder_dur_ms)
                        for p in team_pids.get(team, []):
                            elder_end[p] = max(int(elder_end[p]), end_ts)

                if use_soul and mt == "DRAGON" and (not _is_elder_dragon(e)) and team in (100, 200):
                    sub = str(e.get("monsterSubType", "") or "")
                    soul_type = _dragon_subtype_to_soul(sub)
                    if soul_type is not None:
                        team_dragon_cnt[team] = int(team_dragon_cnt.get(team, 0)) + 1
                        if team_soul.get(team, None) is None and team_dragon_cnt[team] >= 4:
                            team_soul[team] = soul_type

            elif et == "DRAGON_SOUL_GIVEN" and use_soul:
                team = int(e.get("teamId", 0) or 0)
                soul = str(e.get("dragonSoul", "") or e.get("soulType", "") or "").lower()
                if team in (100, 200) and soul in DRAGON_SOUL_TYPES:
                    team_soul[team] = soul

            if et == "SKILL_LEVEL_UP" and use_ult:
                pid = int(e.get("participantId", 0) or 0)
                if 1 <= pid <= 10:
                    try:
                        slot = int(e.get("skillSlot", 0) or 0)
                    except Exception:
                        slot = 0
                    if slot == 4:
                        ult_lv[pid] = int(min(3, int(ult_lv[pid]) + 1))

        for k in counters:
            history[k][t] = float(counters[k])

        global_minute[t] = build_global_minute_vector(t, T, node_minute[t], team_map, history, bans=bans)

    return {
        "minute_ts": minute_ts.astype(np.int64),
        "node_minute": node_minute.astype(np.float32),
        "global_minute": global_minute.astype(np.float32),
        "gold_team_minute": gold_team.astype(np.float32),
        "xy_raw_minute": xy_raw_minute.astype(np.float32),
        "events": all_events,
    }

