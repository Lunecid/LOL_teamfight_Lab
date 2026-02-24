from __future__ import annotations
import hashlib
import random
from core.config import (
    cfg, CHAMPION_STATS_KEYS, CS_DENOM, DAMAGE_STATS_KEYS, DS_DENOM,
    CHAMPION_STATS_DIV100_KEYS,
    F_EVENT, F_GLOBAL, F_NODE, GLOBAL_FEATURE_NAMES, MAP_MAX, NODE_IDX,
    GLOBAL_IDX, EVENT_IDX, DRAGON_SOUL_TYPES, OBJ_SCORE, ITEM_HASH_DIM,
)
from core.common import Any, Dict, List, Optional, Tuple, np, math, safe_float, log1p_norm
from core.contract import _BOOL_NODE_COLS

from data.events_index import _events_in_window
from core.timeutils import gold_at_ms, _get_horizon_ms, _get_context_ms, _get_bin_ms, ts_to_minute_idx, validate_engage_ts
from gameplay.fights import _event_xy


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
        bans: Optional[Dict[str, List[int]]] = None,   # ✅ NEW
) -> np.ndarray:
    """
    Build vector aligned to GLOBAL_FEATURE_NAMES.
    Unknown names -> 0.
    """
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

    h = {k: float(history.get(k, np.zeros((T,), np.float32))[t]) for k in
         ["kill", "tower", "inhib", "dragon", "baron", "herald", "atakhan", "plate", "horde"]}

    # ─────────────────────────────────────────────────────────────────
    # [P1-LOGIC-6] Explicit alias mapping — order-independent resolution.
    #
    # EVERY key in GLOBAL_FEATURE_NAMES must appear in this dict so that
    # the mapping M: name → ℝ resolves via O(1) hash lookup:
    #
    #   M(n) = alias[n]            if n ∈ A  (O(1), order-independent)
    #        = fallback_chain(n)   if n ∉ A  (defense-in-depth only)
    #
    # The fallback elif-chain below is retained ONLY for future feature
    # names that haven't been explicitly registered yet.
    # Currently |A| ⊇ |GLOBAL_FEATURE_NAMES \ ban_features|, so the
    # fallback chain's predicate ordering never affects existing features.
    # ─────────────────────────────────────────────────────────────────
    alias = {
        # ── Time ──
        "time_norm": time_norm, "t_norm": time_norm, "minute_norm": time_norm,

        # ── Team-level resource diffs ──
        "gold_diff": gold_diff, "goldDiff": gold_diff,
        "totalGold_diff": gold_diff, "totalGoldDiff": gold_diff,
        "xp_diff": xp_diff, "xpDiff": xp_diff,
        "level_diff": lvl_diff, "levelDiff": lvl_diff,
        "avgLevelDiff": lvl_diff,           # [P1-LOGIC-6] ← was fallback: ("level" ∈ n) ∧ ("diff" ∈ n)

        # ── CS diffs ──
        "laneCS_diff": lane_cs_diff, "laneCSDiff": lane_cs_diff,
        "jgCS_diff": jg_cs_diff, "jgCSDiff": jg_cs_diff,
        "csJgDiff": jg_cs_diff,             # [P1-LOGIC-6] ← was fallback: ("jg" ∈ n) ∧ ("diff" ∈ n) — order-sensitive
        "csDiff_total": cs_total_diff,       # [P1-LOGIC-6] ← was fallback: ("cs" ∈ n) ∧ ("diff" ∈ n) — order-sensitive

        # ── Alive ──
        "alive_diff": alive_diff, "aliveDiff": alive_diff,

        # ── Cumulative objective/kill diffs (history-based) ──
        "kill_diff": h["kill"], "killDiff": h["kill"],
        "killDiff_cum": h["kill"],           # [P1-LOGIC-6] ← was fallback: history loop
        "tower_diff": h["tower"], "towerDiff": h["tower"],
        "towerDiff_cum": h["tower"],         # [P1-LOGIC-6] ← was fallback: history loop
        "inhib_diff": h["inhib"], "inhibDiff": h["inhib"],
        "inhibDiff_cum": h["inhib"],         # [P1-LOGIC-6] ← was fallback: history loop
        "dragon_diff": h["dragon"], "dragonDiff": h["dragon"],
        "dragonDiff_cum": h["dragon"],       # [P1-LOGIC-6] ← was fallback: history loop
        "baron_diff": h["baron"], "baronDiff": h["baron"],
        "baronDiff_cum": h["baron"],         # [P1-LOGIC-6] ← was fallback: history loop
        "herald_diff": h["herald"], "heraldDiff": h["herald"],
        "heraldDiff_cum": h["herald"],       # [P1-LOGIC-6] ← was fallback: history loop
        "atakhan_diff": h["atakhan"], "atakhanDiff": h["atakhan"],
        "atakhanDiff_cum": h["atakhan"],     # [P1-LOGIC-6] ← was fallback: history loop
        "plate_diff": h["plate"], "plateDiff": h["plate"],
        "plateDiff_cum": h["plate"],         # [P1-LOGIC-6] ← was fallback: history loop
        "horde_diff": h["horde"], "hordeDiff": h["horde"],
        "hordeDiff_cum": h["horde"],         # [P1-LOGIC-6] ← was fallback: history loop
    }

    out = np.zeros((F_GLOBAL,), dtype=np.float32)
    for name, j in GLOBAL_IDX.items():
        if name in alias:
            out[j] = float(alias[name])
            continue
        # ── Fallback chain (defense-in-depth for future feature names only) ──
        # [P1-LOGIC-6] All current GLOBAL_FEATURE_NAMES are in alias above.
        # This fallback is NEVER reached for existing features.
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

    # ✅ NEW: bans are static per match (team-level metadata)
    if isinstance(bans, dict):
        blue = bans.get("blue", [0, 0, 0, 0, 0])
        red = bans.get("red", [0, 0, 0, 0, 0])
        for k in range(5):
            nm = f"blue_ban_{k}"
            jj = GLOBAL_IDX.get(nm, None)
            if jj is not None:
                out[jj] = float(blue[k] if k < len(blue) else 0)
        for k in range(5):
            nm = f"red_ban_{k}"
            jj = GLOBAL_IDX.get(nm, None)
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


def _detect_red_blue(monster_type: str, monster_sub: str) -> Optional[str]:
    s = (str(monster_type or "") + " " + str(monster_sub or "")).upper()
    if ("RED" in s) or ("BRAMBLEBACK" in s) or ("REDBUFF" in s):
        return "red"
    if ("BLUE" in s) or ("SENTINEL" in s) or ("BLUEBUFF" in s):
        return "blue"
    return None


def _is_flash_spell(e: dict) -> bool:
    if not isinstance(e, dict):
        return False
    for k in ("spellId", "summonerSpellId"):
        v = e.get(k, None)
        try:
            if int(v) == 4:
                return True
        except Exception:
            pass
    nm = str(e.get("summonerSpellName", "") or e.get("spellName", "") or "")
    return "FLASH" in nm.upper()


def _pf_xy(pf: dict) -> Tuple[float, float]:
    if not isinstance(pf, dict):
        return 0.0, 0.0
    pos = pf.get("position", None)
    if isinstance(pos, dict):
        return float(safe_float(pos.get("x", 0.0))), float(safe_float(pos.get("y", 0.0)))
    if "x" in pf and "y" in pf:
        return float(safe_float(pf.get("x", 0.0))), float(safe_float(pf.get("y", 0.0)))
    return 0.0, 0.0


def _count_near_points(points: List[Tuple[float, float, int]], px: float, py: float,
                       radius: float, t_ms: int, window_ms: int) -> int:
    if not points:
        return 0
    r2 = float(radius) * float(radius)
    lo = int(t_ms) - int(window_ms)
    c = 0
    for (x, y, ts) in points:
        if ts < lo:
            continue
        dx = float(x) - float(px)
        dy = float(y) - float(py)
        if dx * dx + dy * dy <= r2:
            c += 1
    return int(c)


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


# =========================
# ✅ NEW: detail(meta)에서 champion/runes/bans 추출
# =========================
def _extract_champ_runes_bans_from_detail(detail):
    """Delegate to canonical implementation in data.cache_io."""
    from data.cache_io import _extract_static_meta_from_detail
    return _extract_static_meta_from_detail(detail)


# =========================
# ✅ signature 확장 (detail/cfg 대응)
# =========================
def parse_timeline_to_minute_cache(
    timeline: dict,
    team_map: Dict[int, int],
    *,
    detail: Optional[Dict[str, Any]] = None,    # ✅ NEW
    cfg_in: Any = None,                          # ✅ NEW (호환)
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

    counters = {k: 0 for k in
                ["kill", "tower", "inhib", "dragon", "elder", "baron", "herald", "atakhan", "plate", "horde"]}
    history = {k: np.zeros(T, dtype=np.float32) for k in counters}

    all_events: List[dict] = []

    frame_ms = int(getattr(_cfg, "FRAME_MS", 60000))
    coord_div = float(getattr(_cfg, "COORD_NORM_DIV", MAP_MAX))

    # ---- status switches ----
    use_status = bool(getattr(_cfg, "USE_STATUS_FEATURES", True))
    use_buff = bool(getattr(_cfg, "USE_BUFF_STATUS", True))
    use_soul = bool(getattr(_cfg, "USE_DRAGON_SOUL_STATUS", True))
    use_ult = bool(getattr(_cfg, "USE_ULT_LEVEL", True))
    use_flash = bool(getattr(_cfg, "USE_FLASH_READY", True))
    use_vision = bool(getattr(_cfg, "USE_LOCAL_VISION", True))

    buff_dur = dict(getattr(_cfg, "BUFF_DUR_SEC", {})) if hasattr(_cfg, "BUFF_DUR_SEC") else {}
    baron_dur_ms = int(buff_dur.get("baron", 180)) * 1000
    elder_dur_ms = int(buff_dur.get("elder", 150)) * 1000
    red_dur_ms = int(buff_dur.get("red", 120)) * 1000
    blue_dur_ms = int(buff_dur.get("blue", 120)) * 1000

    flash_cd_ms = int(getattr(_cfg, "FLASH_CD_SEC", 300)) * 1000

    vision_radius = float(getattr(_cfg, "VISION_RADIUS", 1200.0))
    vision_recent_ms = int(getattr(_cfg, "VISION_RECENT_SEC", 90)) * 1000
    vision_denom = float(getattr(_cfg, "VISION_CNT_DENOM", 10.0))

    baron_end = np.zeros(11, dtype=np.int64)
    elder_end = np.zeros(11, dtype=np.int64)
    red_end = np.zeros(11, dtype=np.int64)
    blue_end = np.zeros(11, dtype=np.int64)

    ult_lv = np.zeros(11, dtype=np.int32)
    last_flash = np.full(11, -10 ** 18, dtype=np.int64)

    team_dragon_cnt = {100: 0, 200: 0}
    team_soul = {100: None, 200: None}

    ward_place: Dict[int, List[Tuple[float, float, int]]] = {100: [], 200: []}
    ward_kill: Dict[int, List[Tuple[float, float, int]]] = {100: [], 200: []}

    # ✅ NEW: metadata (champ/runes/bans)
    _meta = _extract_champ_runes_bans_from_detail(detail)
    champ_by_pid: Dict[int, int] = {
        int(k): int(v) for k, v in _meta.get("champion_by_pid", {}).items()
    }
    runes_by_pid: Dict[int, Dict[str, int]] = {
        int(k): v for k, v in _meta.get("runes_by_pid", {}).items()
    }
    bans: Dict[str, List[int]] = _meta.get("bans", {"blue": [0, 0, 0, 0, 0], "red": [0, 0, 0, 0, 0]})

    def _set(vec: np.ndarray, name: str, val: float):
        j = NODE_IDX.get(name, None)
        if j is not None:
            vec[j] = float(val)

    # base denoms (kept local)
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

        if use_status and use_vision:
            lo = t_ms - vision_recent_ms
            ward_place[100] = [(x, y, ts) for (x, y, ts) in ward_place[100] if ts >= lo]
            ward_place[200] = [(x, y, ts) for (x, y, ts) in ward_place[200] if ts >= lo]
            ward_kill[100] = [(x, y, ts) for (x, y, ts) in ward_kill[100] if ts >= lo]
            ward_kill[200] = [(x, y, ts) for (x, y, ts) in ward_kill[200] if ts >= lo]

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

            # ✅ NEW: champion_id + runes (static metadata)
            if "champion_id" in NODE_IDX:
                _set(vec, "champion_id", float(champ_by_pid.get(pid, 0)))

            rmeta = runes_by_pid.get(pid, None)
            if isinstance(rmeta, dict):
                for rk, rv in rmeta.items():
                    if rk in NODE_IDX:
                        _set(vec, rk, float(int(rv)))

            # coords (deterministic)
            _set(vec, "x_norm", float(np.clip(x_raw / max(1e-6, coord_div), 0.0, 2.0)))
            _set(vec, "y_norm", float(np.clip(y_raw / max(1e-6, coord_div), 0.0, 2.0)))

            # base scalars (deterministic)
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

            # champion stats + damage stats
            for k in CHAMPION_STATS_KEYS:
                denom = float(CS_DENOM.get(k, 1.0))
                raw_cs = _normalize_cs_raw(k, float(safe_float(cs.get(k, 0.0))))
                val = raw_cs / max(1e-6, denom)
                _set(vec, f"cs_{k}", float(np.clip(val, -10.0, 10.0)))

            for k in DAMAGE_STATS_KEYS:
                denom = float(DS_DENOM.get(k, 1.0))
                val = float(safe_float(ds.get(k, 0.0))) / max(1e-6, denom)
                _set(vec, f"ds_{k}", float(np.clip(val, -10.0, 10.0)))

            # status
            if use_status:
                tid = int(team_map.get(pid, 100 if pid <= 5 else 200))

                if use_buff:
                    hb = 1.0 if t_ms < int(baron_end[pid]) else 0.0
                    he = 1.0 if t_ms < int(elder_end[pid]) else 0.0
                    hr = 1.0 if t_ms < int(red_end[pid]) else 0.0
                    hu = 1.0 if t_ms < int(blue_end[pid]) else 0.0

                    _set(vec, "has_baron", hb)
                    _set(vec, "has_elder", he)
                    _set(vec, "has_red", hr)
                    _set(vec, "has_blue", hu)

                    def _remain(end_ts: int, dur_ms: int) -> float:
                        if dur_ms <= 0:
                            return 0.0
                        rem = float(max(0, int(end_ts) - int(t_ms))) / float(dur_ms)
                        return float(np.clip(rem, 0.0, 1.0))

                    _set(vec, "baron_remain_norm", _remain(int(baron_end[pid]), baron_dur_ms))
                    _set(vec, "elder_remain_norm", _remain(int(elder_end[pid]), elder_dur_ms))
                    _set(vec, "red_remain_norm", _remain(int(red_end[pid]), red_dur_ms))
                    _set(vec, "blue_remain_norm", _remain(int(blue_end[pid]), blue_dur_ms))

                if use_soul:
                    soul = team_soul.get(tid, None)
                    for st in DRAGON_SOUL_TYPES:
                        _set(vec, f"soul_{st}", 1.0 if soul == st else 0.0)

                if use_ult:
                    _set(vec, "ult_level_norm", float(np.clip(float(ult_lv[pid]) / 3.0, 0.0, 1.0)))

                if use_flash:
                    lf = int(last_flash[pid])
                    if lf < -10 ** 17:
                        _set(vec, "flash_ready", 1.0)
                        _set(vec, "flash_remain_norm", 0.0)
                    else:
                        dt = int(t_ms) - lf
                        if dt >= flash_cd_ms:
                            _set(vec, "flash_ready", 1.0)
                            _set(vec, "flash_remain_norm", 0.0)
                        else:
                            rem = float(flash_cd_ms - dt) / float(max(1, flash_cd_ms))
                            _set(vec, "flash_ready", 0.0)
                            _set(vec, "flash_remain_norm", float(np.clip(rem, 0.0, 1.0)))

                if use_vision:
                    ally_cnt = _count_near_points(ward_place.get(tid, []), x_raw, y_raw,
                                                  vision_radius, t_ms, vision_recent_ms)
                    kill_cnt = _count_near_points(ward_kill.get(tid, []), x_raw, y_raw,
                                                  vision_radius, t_ms, vision_recent_ms)
                    ally_norm = float(np.clip(float(ally_cnt) / max(1e-6, vision_denom), 0.0, 1.0))
                    kill_norm = float(np.clip(float(kill_cnt) / max(1e-6, vision_denom), 0.0, 1.0))
                    score = float(np.clip(0.5 * ally_norm + 0.5 * kill_norm, 0.0, 1.0))

                    _set(vec, "vision_ally_ward_cnt_norm", ally_norm)
                    _set(vec, "vision_ward_kill_recent_norm", kill_norm)
                    _set(vec, "vision_nearby_score_norm", score)

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

            # ----- status updates (event-driven) -----
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

            if et == "MONSTER_KILL" and use_buff:
                mt = str(e.get("monsterType", "") or "")
                sub = str(e.get("monsterSubType", "") or "")
                which = _detect_red_blue(mt, sub)
                if which is not None:
                    killer = int(e.get("killerId", 0) or e.get("participantId", 0) or 0)
                    if 1 <= killer <= 10:
                        if which == "red":
                            red_end[killer] = max(int(red_end[killer]), int(ev_ts) + int(red_dur_ms))
                        else:
                            blue_end[killer] = max(int(blue_end[killer]), int(ev_ts) + int(blue_dur_ms))

            if et == "SKILL_LEVEL_UP" and use_ult:
                pid = int(e.get("participantId", 0) or 0)
                if 1 <= pid <= 10:
                    try:
                        slot = int(e.get("skillSlot", 0) or 0)
                    except Exception:
                        slot = 0
                    if slot == 4:
                        ult_lv[pid] = int(min(3, int(ult_lv[pid]) + 1))

            if et in ("SUMMONER_SPELL_USED", "SUMMONER_SPELL_CAST") and use_flash:
                if _is_flash_spell(e):
                    pid = int(e.get("participantId", 0) or e.get("casterId", 0) or 0)
                    if 1 <= pid <= 10:
                        last_flash[pid] = int(ev_ts)

            if et in ("WARD_PLACED", "WARD_KILL") and use_vision:
                xy = _event_xy(e)
                if xy is None:
                    continue
                x, y = xy
                if not (0 <= x <= 20000 and 0 <= y <= 20000):
                    continue

                pid = int(e.get("creatorId", 0) or e.get("killerId", 0) or 0)
                tid = int(team_map.get(pid, 0))
                if tid not in (100, 200):
                    continue
                if et == "WARD_PLACED":
                    ward_place[tid].append((float(x), float(y), int(ev_ts)))
                else:
                    ward_kill[tid].append((float(x), float(y), int(ev_ts)))

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


# =========================================================
# ✅ NEW: XY linear + discontinuity guard
# =========================================================
def _interp_xy_guarded(
    cache: Dict[str, Any],
    i: int,
    j: int,
    alpha: float,
    *,
    coord_div: float,
) -> Optional[np.ndarray]:
    """
    Returns xy_norm (10,2) after guarded interpolation.
    - Uses cache['xy_raw_minute'] if available, otherwise returns None.
    - If jump/discontinuity detected, uses midstep snapping by default.
    """
    xy = cache.get("xy_raw_minute", None)
    if not isinstance(xy, np.ndarray) or xy.ndim != 3 or xy.shape[2] != 2:
        return None

    xi = xy[i].astype(np.float32)   # (10,2) raw
    xj = xy[j].astype(np.float32)

    # config knobs
    th = float(getattr(cfg, "XY_DISCONT_DIST_RAW", 7000.0))
    th2 = th * th
    guard_mode = str(getattr(cfg, "XY_GUARD_MODE", "midstep")).lower()
    use_alive = bool(getattr(cfg, "XY_DISCONT_USE_ALIVE", True))

    alive_idx = NODE_IDX.get("alive", None)
    alive_i = None
    alive_j = None
    if use_alive and alive_idx is not None:
        alive_i = cache["node_minute"][i][:, alive_idx].astype(np.float32)
        alive_j = cache["node_minute"][j][:, alive_idx].astype(np.float32)

    out = np.zeros((10, 2), np.float32)

    for p in range(10):
        dx = float(xj[p, 0] - xi[p, 0])
        dy = float(xj[p, 1] - xi[p, 1])
        dist2 = dx * dx + dy * dy

        disc = False
        if dist2 > th2:
            disc = True
        if use_alive and alive_i is not None and alive_j is not None:
            # alive가 바뀌는 구간은 순간이동/리콜/부활/데스 등의 불연속 가능성이 높으니 보수적으로 처리
            if (alive_i[p] > 0.5) != (alive_j[p] > 0.5):
                disc = True

        if disc:
            # midstep: alpha<0.5면 i, 아니면 j
            if guard_mode == "hold":
                xr, yr = xi[p, 0], xi[p, 1]
            else:
                xr, yr = (xi[p, 0], xi[p, 1]) if alpha < 0.5 else (xj[p, 0], xj[p, 1])
        else:
            xr = (1.0 - alpha) * xi[p, 0] + alpha * xj[p, 0]
            yr = (1.0 - alpha) * xi[p, 1] + alpha * xj[p, 1]

        out[p, 0] = float(np.clip(xr / max(1e-6, coord_div), 0.0, 2.0))
        out[p, 1] = float(np.clip(yr / max(1e-6, coord_div), 0.0, 2.0))

    return out


def interpolate_node_global(cache: Dict[str, Any], q_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    ts = cache["minute_ts"]
    nm = cache["node_minute"]
    gm = cache["global_minute"]

    coord_div = float(getattr(cfg, "COORD_NORM_DIV", MAP_MAX))

    if len(ts) == 1:
        node = nm[0].astype(np.float32)
        glob = gm[0].astype(np.float32)
    else:
        idx = int(np.searchsorted(ts, q_ms) - 1)
        i = max(0, idx)
        j = min(len(ts) - 1, idx + 1)

        # ---- time fraction ----
        if ts[j] == ts[i]:
            alpha = 0.0
        else:
            alpha = float(q_ms - ts[i]) / float(ts[j] - ts[i])
            alpha = float(np.clip(alpha, 0.0, 1.0))

        # ✅ 핵심: scalars는 hold(ffill), xy만 guarded linear
        scalars_method = str(getattr(cfg, "INTERP_SCALARS_METHOD", "ffill")).lower()
        if scalars_method in ("ffill", "hold", "step", "zero_order"):
            node = nm[i].copy()
            glob = gm[i].copy()
        else:
            # legacy fallback: full linear (원하면 유지)
            node = ((1.0 - alpha) * nm[i] + alpha * nm[j]).astype(np.float32)
            glob = ((1.0 - alpha) * gm[i] + alpha * gm[j]).astype(np.float32)

        # ---- XY interpolation (guarded) ----
        if bool(getattr(cfg, "INTERP_XY", True)):
            xy_norm = _interp_xy_guarded(cache, i, j, alpha, coord_div=coord_div)
            if xy_norm is not None:
                xj_idx = NODE_IDX.get("x_norm", None)
                yj_idx = NODE_IDX.get("y_norm", None)
                if xj_idx is not None:
                    node[:, xj_idx] = xy_norm[:, 0]
                if yj_idx is not None:
                    node[:, yj_idx] = xy_norm[:, 1]

    # ---- Zero out x,y node features when configured ----
    # Position data is at 60s frame resolution; too stale for prediction input.
    if bool(getattr(cfg, "ZERO_XY_NODE_FEATURES", False)):
        xj_idx = NODE_IDX.get("x_norm", None)
        yj_idx = NODE_IDX.get("y_norm", None)
        if xj_idx is not None:
            node[:, xj_idx] = 0.0
        if yj_idx is not None:
            node[:, yj_idx] = 0.0

        # ---- optional: continuous time_norm in global ----
        tj = GLOBAL_IDX.get("time_norm", None)
        if tj is not None:
            t0 = float(ts[0])
            t1 = float(ts[-1])
            if t1 > t0:
                glob[tj] = float(np.clip((float(q_ms) - t0) / (t1 - t0), 0.0, 1.0))

    alive_idx = NODE_IDX.get("alive", None)
    if alive_idx is not None:
        node[:, alive_idx] = np.clip(node[:, alive_idx], 0.0, 1.0)

    if _BOOL_NODE_COLS:
        node[:, _BOOL_NODE_COLS] = np.clip(node[:, _BOOL_NODE_COLS], 0.0, 1.0)

    for k in ("baron_remain_norm", "elder_remain_norm", "red_remain_norm", "blue_remain_norm", "flash_remain_norm"):
        j = NODE_IDX.get(k, None)
        if j is not None:
            node[:, j] = np.clip(node[:, j], 0.0, 1.0)

    return node.astype(np.float32), glob.astype(np.float32)


def _prev_snapshot_idx(ts: np.ndarray, ref_ms: int, *, strict_before: bool = True) -> int:
    """Index of nearest previous timestamp in ts for ref_ms."""
    if ts is None or len(ts) == 0:
        return -1
    side = "left" if strict_before else "right"
    idx = int(np.searchsorted(ts, int(ref_ms), side=side) - 1)
    if idx < 0:
        return -1
    return int(min(idx, len(ts) - 1))


def global_from_prev_snapshot(cache: Dict[str, Any], ref_ms: int, *, strict_before: bool = True) -> Tuple[np.ndarray, int]:
    """Return global feature vector from nearest previous available timestamp."""
    ts = cache["minute_ts"]
    gm = cache["global_minute"]
    if ts is None or len(ts) == 0 or gm is None or len(gm) == 0:
        return np.zeros((F_GLOBAL,), dtype=np.float32), -1
    idx = _prev_snapshot_idx(ts, int(ref_ms), strict_before=bool(strict_before))
    if idx < 0:
        return np.zeros((F_GLOBAL,), dtype=np.float32), -1
    return gm[int(idx)].astype(np.float32, copy=True), int(ts[int(idx)])


def aggregate_events(events_or_pack: Any, tm: Dict[int, int], s_ms: int, e_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast aggregation with event indexing:
      - If events_or_pack is a cache pack(dict) with events_ts, use binary search slicing.
      - Else fallback to scanning a list (backward compatible).
    """
    ev = np.zeros(F_EVENT, np.float32)
    h = np.zeros(ITEM_HASH_DIM, np.float32)

    # fast slice
    if isinstance(events_or_pack, dict):
        evs = _events_in_window(events_or_pack, s_ms, e_ms)
    else:
        # fallback: assume it's a list of events
        events = events_or_pack if isinstance(events_or_pack, list) else []
        evs = []
        for e in events:
            if not isinstance(e, dict):
                continue
            try:
                t = int(e.get("timestamp", -1) or -1)
            except Exception:
                continue
            if s_ms <= t < e_ms:
                evs.append(e)

    for e in evs:
        if not isinstance(e, dict):
            continue

        et = str(e.get("type", "")).upper()

        if et == "CHAMPION_KILL":
            tid = tm.get(int(e.get("killerId", 0) or 0), 0)
            if tid in (100, 200):
                k = f"kills_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                b = f"bounty_t{tid}"
                shutdown = safe_float(e.get("shutdownBounty", 0.0))
                if b in EVENT_IDX:
                    ev[EVENT_IDX[b]] += safe_float(e.get("bounty", 0.0)) + shutdown

                if shutdown > 0.0:
                    ks = f"shutdown_kill_t{tid}"
                    if ks in EVENT_IDX:
                        ev[EVENT_IDX[ks]] += 1.0

                streak = max(0.0, safe_float(e.get("killStreakLength", 0.0)))
                if streak > 0.0:
                    kk = f"killstreak_t{tid}"
                    if kk in EVENT_IDX:
                        ev[EVENT_IDX[kk]] += streak

        elif et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            tid = int(e.get("killerTeamId", 0) or 0)
            tag = {
                "DRAGON": "dragon",
                "BARON_NASHOR": "baron",
                "RIFTHERALD": "herald",
                "ATAKHAN": "atakhan",
                "HORDE": "horde",
            }.get(mt)
            if tag and tid in (100, 200):
                k = f"{tag}_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                kb = f"obj_bounty_t{tid}"
                if kb in EVENT_IDX:
                    ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "BUILDING_KILL":
            bt = str(e.get("buildingType", "")).upper()
            victim_team = int(e.get("teamId", 0) or 0)
            if victim_team not in (100, 200):
                continue
            tid = 200 if victim_team == 100 else 100
            if "TOWER" in bt:
                k = f"tower_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
            elif "INHIBITOR" in bt:
                k = f"inhib_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
            kb = f"obj_bounty_t{tid}"
            if kb in EVENT_IDX:
                ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "TURRET_PLATE_DESTROYED":
            victim_team = int(e.get("teamId", 0) or 0)
            if victim_team not in (100, 200):
                continue
            tid = 200 if victim_team == 100 else 100
            k = f"plate_t{tid}"
            if k in EVENT_IDX:
                ev[EVENT_IDX[k]] += 1.0
            kb = f"obj_bounty_t{tid}"
            if kb in EVENT_IDX:
                ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "CHAMPION_SPECIAL_KILL":
            tid = tm.get(int(e.get("killerId", 0) or 0), 0)
            if tid in (100, 200):
                kt = str(e.get("killType", "")).upper()
                if "MULTI" in kt:
                    km = f"multikill_t{tid}"
                    if km in EVENT_IDX:
                        mk = max(1.0, safe_float(e.get("multiKillLength", 1.0)))
                        ev[EVENT_IDX[km]] += mk
                if "ACE" in kt:
                    ka = f"ace_t{tid}"
                    if ka in EVENT_IDX:
                        ev[EVENT_IDX[ka]] += 1.0

        elif et in ("WARD_PLACED", "WARD_KILL"):
            pid = int(e.get("creatorId", 0) or e.get("killerId", 0) or 0)
            tid = tm.get(pid, 0)
            if tid in (100, 200):
                k = ("ward_placed_t" if et == "WARD_PLACED" else "ward_kill_t") + str(tid)
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                wt = str(e.get("wardType", "")).upper()
                is_control = ("CONTROL" in wt)
                if is_control:
                    kc = ("control_ward_placed_t" if et == "WARD_PLACED" else "control_ward_kill_t") + str(tid)
                    if kc in EVENT_IDX:
                        ev[EVENT_IDX[kc]] += 1.0

        elif et in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"):
            pid = int(e.get("participantId", 0) or 0)
            tid = tm.get(pid, 0)
            if tid in (100, 200):
                item_id = int(e.get("itemId", 0) or 0)
                if item_id > 0:
                    h[int(item_id * 2654435761) % ITEM_HASH_DIM] += 1.0
                tag = {
                    "ITEM_PURCHASED": "pur",
                    "ITEM_SOLD": "sold",
                    "ITEM_DESTROYED": "sold",
                    "ITEM_UNDO": "undo",
                }.get(et)
                if tag:
                    k = f"item_{tag}_t{tid}"
                    if k in EVENT_IDX:
                        ev[EVENT_IDX[k]] += 1.0

    return ev, h


import zlib

def _hash32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def build_event_tokens_for_xattn(
    pack: Dict[str, Any],
    tm: Dict[int, int],
    s_ms: int,
    e_ms: int,
    *,
    max_tokens: int = 64,
) -> Dict[str, np.ndarray]:

    from data.events_index import _events_in_window, _event_ts_safe

    evs = _events_in_window(pack, int(s_ms), int(e_ms))
    if not evs:
        return {
            "event_type": np.zeros((max_tokens,), np.int64),
            "event_actor": np.zeros((max_tokens,), np.int64),
            "event_team": np.full((max_tokens,), 2, np.int64),  # 0 blue, 1 red, 2 unk
            "event_cont": np.zeros((max_tokens, 5), np.float32),
            "event_mask": np.zeros((max_tokens,), np.float32),
        }

    evs2 = []
    for e in evs:
        ts = _event_ts_safe(e)
        if ts is None:
            continue
        evs2.append((int(ts), e))
    evs2.sort(key=lambda x: x[0])
    evs2 = evs2[-int(max_tokens):]

    K = int(max_tokens)
    type_vocab = int(getattr(cfg, "EVENT_TYPE_VOCAB", 128))
    coord_div = float(getattr(cfg, "COORD_NORM_DIV", 16000.0))
    denom = max(1.0, float(int(e_ms) - int(s_ms)))

    event_type = np.zeros((K,), np.int64)
    event_actor = np.zeros((K,), np.int64)
    event_team = np.full((K,), 2, np.int64)
    event_cont = np.zeros((K, 5), np.float32)
    event_mask = np.zeros((K,), np.float32)

    def _pick_pid(e: dict) -> int:
        for kk in ("killerId", "participantId", "creatorId", "victimId"):
            v = e.get(kk, None)
            if v is None:
                continue
            try:
                vi = int(v)
                if 0 <= vi <= 10:
                    return vi
            except Exception:
                pass
        return 0

    for i, (ts, e) in enumerate(evs2):
        et = str(e.get("type", "")).upper()
        event_type[i] = int(_hash32(et) % max(1, type_vocab))

        pid = _pick_pid(e)
        event_actor[i] = int(pid)

        tid = tm.get(pid, 0)
        if tid not in (100, 200):
            try:
                tid = int(e.get("killerTeamId", 0) or e.get("teamId", 0) or 0)
            except Exception:
                tid = 0
        if tid == 100:
            event_team[i] = 0
        elif tid == 200:
            event_team[i] = 1
        else:
            event_team[i] = 2

        t_rel = float(ts - int(s_ms)) / denom
        dt_end = float(int(e_ms) - ts) / denom
        pos = e.get("position", {}) if isinstance(e.get("position", {}), dict) else {}
        x = safe_float(pos.get("x", 0.0)) / coord_div
        y = safe_float(pos.get("y", 0.0)) / coord_div

        val = 0.0
        for kk in ("bounty", "shutdownBounty", "goldGain", "xpGain"):
            if kk in e:
                val += safe_float(e.get(kk, 0.0))
        val = float(np.log1p(max(0.0, val)))

        event_cont[i] = np.asarray([t_rel, dt_end, x, y, val], np.float32)
        event_mask[i] = 1.0

    return {
        "event_type": event_type,
        "event_actor": event_actor,
        "event_team": event_team,
        "event_cont": event_cont,
        "event_mask": event_mask,
    }



def compute_label(
        cache: Dict[str, Any],
        tm: Dict[int, int],
        t_start: int,
        *,
        engage_ts: Optional[int] = None,
        label_end_ts: Optional[int] = None,
        horizon_ms: Optional[int] = None,
) -> Optional[int]:
    if horizon_ms is None:
        horizon_ms = _get_horizon_ms()

    if engage_ts is not None and engage_ts >= 0:
        s_ms = int(engage_ts)
        e_ms = s_ms + int(horizon_ms)
        if label_end_ts is not None:
            try:
                cand_end = int(label_end_ts)
            except Exception:
                cand_end = -1
            if cand_end > s_ms:
                e_ms = cand_end
    else:
        if t_start < 0 or t_start >= len(cache["minute_ts"]):
            return None
        s_ms = int(cache["minute_ts"][t_start])
        e_ms = s_ms + int(horizon_ms)

    if e_ms > int(cache["minute_ts"][-1]):
        return None

    evs = _events_in_window(cache, s_ms, e_ms)

    if getattr(cfg, "REQUIRE_SIGNAL_IN_HORIZON", False):
        has_sig = any(
            str(e.get("type", "")).upper() in ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL")
            for e in evs
        )
        if not has_sig:
            return None

    label_type = str(getattr(cfg, "LABEL_TYPE", "micro_win")).lower()

    if label_type == "micro_win":
        return _compute_label_micro_win(evs, tm)
    elif label_type == "kill_survival":
        return _compute_label_kill_survival(evs, tm, cache, e_ms)
    else:
        return _compute_label_weighted(evs, tm, cache, s_ms, e_ms)


def _compute_label_micro_win(evs: List[dict], tm: Dict[int, int]) -> Optional[int]:
    """
    Micro-win label: pure kill differential.
        S_mw = K_blue - K_red

    [P1-LOGIC-1b FIX] Tie policy (S_mw == 0) now delegates to
    LABEL_TIE_STRATEGY, consistent with kill_survival and weighted.
    Previously ties were hard-coded to 0 (red win).
    """
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY",
                             getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

    blue_kills = 0
    red_kills = 0
    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                blue_kills += 1
            elif tm.get(killer, 0) == 200:
                red_kills += 1

    if blue_kills == red_kills:
        if tie_policy in ("drop", "exclude", "none"):
            return None
        if tie_policy in ("random", "stochastic", "coinflip"):
            return int(random.random() < 0.5)
        if tie_policy == "blue":
            return 1
        if tie_policy == "red":
            return 0
        return None

    return 1 if blue_kills > red_kills else 0


def _compute_label_kill_survival(
        evs: List[dict],
        tm: Dict[int, int],
        cache: Dict[str, Any],
        e_ms: int
) -> Optional[int]:

    w_kill = float(getattr(cfg, "LABEL_W_KILL", 1.0))
    w_alive = float(getattr(cfg, "LABEL_W_ALIVE", 0.3))

    eps = float(getattr(cfg, "LABEL_TIE_EPS", 1e-8))
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY",
                             getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

    blue_kills = 0
    red_kills = 0

    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                blue_kills += 1
            elif tm.get(killer, 0) == 200:
                red_kills += 1

    kill_diff = blue_kills - red_kills

    node_end, _ = interpolate_node_global(cache, e_ms)
    alive_idx = NODE_IDX.get("alive", None)

    alive_diff = 0.0
    if alive_idx is not None:
        tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
        b_idx = np.where(tids == 100)[0]
        r_idx = np.where(tids == 200)[0]
        blue_alive = float(node_end[b_idx, alive_idx].sum())
        red_alive = float(node_end[r_idx, alive_idx].sum())
        alive_diff = blue_alive - red_alive

    score = w_kill * kill_diff + w_alive * alive_diff


    if abs(score) < eps:
        if tie_policy in ("drop", "exclude", "none"):
            return None

        if tie_policy in ("random", "stochastic", "coinflip"):
            return int(random.random() < 0.5)

        return None

    return 1 if score > 0 else 0



def _compute_label_weighted(
        evs: List[dict],
        tm: Dict[int, int],
        cache: Dict[str, Any],
        s_ms: int,
        e_ms: int
) -> Optional[int]:
    """
    Weighted composite label:
        S_w = w_k · Δ_kill + w_g · (Δ_gold / G_norm) + w_o · Δ_obj

    [P1-LOGIC-1 FIX] Tie policy (|S_w| < ε) now delegates to
    LABEL_TIE_STRATEGY, consistent with kill_survival.
    Previously ties were hard-coded to 0 (red win), introducing
    systematic class-prior bias in ablation comparisons.
    """
    # ── read tie policy (same fallback chain as kill_survival) ──
    eps = float(getattr(cfg, "LABEL_TIE_EPS", 1e-8))
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY",
                             getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

    kd = 0
    obj = 0.0

    for e in evs:
        et = str(e.get("type", "")).upper()

        if et == "CHAMPION_KILL":
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                kd += 1
            elif tm.get(killer, 0) == 200:
                kd -= 1

        elif et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            team = int(e.get("killerTeamId", 0) or 0)
            key = mt.replace("_NASHOR", "")
            w = float(OBJ_SCORE.get(key, 0.0))
            obj += w * (1.0 if team == 100 else -1.0)

        elif et == "BUILDING_KILL":
            bt = str(e.get("buildingType", "")).upper()
            victim = int(e.get("teamId", 0) or 0)
            sign = 1.0 if victim == 200 else -1.0
            if "TOWER" in bt:
                obj += float(OBJ_SCORE.get("TOWER", 0.0)) * sign
            elif "INHIBITOR" in bt:
                obj += float(OBJ_SCORE.get("INHIBITOR", 0.0)) * sign

    gold_method = str(getattr(cfg, "LABEL_GOLD_METHOD", "linear")).lower()
    g0 = gold_at_ms(cache, s_ms, method=gold_method)
    g1 = gold_at_ms(cache, e_ms, method=gold_method)
    gd = (g1[0] - g0[0]) - (g1[1] - g0[1])

    score = (
            float(cfg.W_KILL) * float(kd)
            + float(cfg.W_GOLD) * (float(gd) / float(cfg.GOLD_NORM))
            + float(cfg.W_OBJ) * float(obj)
    )

    # ── [P1-LOGIC-1 FIX] unified tie handling ──────────────────
    if abs(score) < eps:
        if tie_policy in ("drop", "exclude", "none"):
            return None
        if tie_policy in ("random", "stochastic", "coinflip"):
            return int(random.random() < 0.5)
        if tie_policy == "blue":
            return 1
        if tie_policy == "red":
            return 0
        # unknown policy → safe default: exclude
        return None

    return 1 if score > 0 else 0


def _compute_window_targets(
        evs: List[dict],
        tm: Dict[int, int],
        cache: Dict[str, Any],
        s_ms: int,
        e_ms: int,
) -> Dict[str, float]:
    kill_diff = 0.0
    obj_diff = 0.0
    summoner_spells = 0.0

    for e in evs:
        et = str(e.get("type", "")).upper()

        if et == "CHAMPION_KILL":
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                kill_diff += 1.0
            elif tm.get(killer, 0) == 200:
                kill_diff -= 1.0

        elif et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            team = int(e.get("killerTeamId", 0) or 0)
            key = mt.replace("_NASHOR", "")
            w = float(OBJ_SCORE.get(key, 0.0))
            obj_diff += w * (1.0 if team == 100 else -1.0)

        elif et == "BUILDING_KILL":
            bt = str(e.get("buildingType", "")).upper()
            victim = int(e.get("teamId", 0) or 0)
            sign = 1.0 if victim == 200 else -1.0
            if "TOWER" in bt:
                obj_diff += float(OBJ_SCORE.get("TOWER", 0.0)) * sign
            elif "INHIBITOR" in bt:
                obj_diff += float(OBJ_SCORE.get("INHIBITOR", 0.0)) * sign

        elif et in ("SUMMONER_SPELL_USED", "SUMMONER_SPELL_CAST"):
            summoner_spells += 1.0

    gold_method = str(getattr(cfg, "LABEL_GOLD_METHOD", "linear")).lower()
    g0 = gold_at_ms(cache, s_ms, method=gold_method)
    g1 = gold_at_ms(cache, e_ms, method=gold_method)
    gold_diff = float((g1[0] - g0[0]) - (g1[1] - g0[1]))

    # Alive diff at horizon end (for diagnostic / auxiliary target)
    alive_diff = 0.0
    try:
        node_end, _ = interpolate_node_global(cache, e_ms)
        alive_idx = NODE_IDX.get("alive", None)
        if alive_idx is not None:
            tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
            b_idx = np.where(tids == 100)[0]
            r_idx = np.where(tids == 200)[0]
            alive_diff = float(node_end[b_idx, alive_idx].sum() - node_end[r_idx, alive_idx].sum())
    except Exception:
        alive_diff = 0.0

    return {
        "kill_diff": float(kill_diff),
        "gold_diff": float(gold_diff),
        "obj_diff": float(obj_diff),
        "alive_diff": float(alive_diff),
        "summoner_spells": float(summoner_spells),
    }


def compute_label_targets(
        cache: Dict[str, Any],
        tm: Dict[int, int],
        t_start: int,
        *,
        engage_ts: Optional[int] = None,
        label_end_ts: Optional[int] = None,
        horizon_ms: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    if horizon_ms is None:
        horizon_ms = _get_horizon_ms()

    if engage_ts is not None and engage_ts >= 0:
        s_ms = int(engage_ts)
        e_ms = s_ms + int(horizon_ms)
        if label_end_ts is not None:
            try:
                cand_end = int(label_end_ts)
            except Exception:
                cand_end = -1
            if cand_end > s_ms:
                e_ms = cand_end
    else:
        if t_start < 0 or t_start >= len(cache["minute_ts"]):
            return None
        s_ms = int(cache["minute_ts"][t_start])
        e_ms = s_ms + int(horizon_ms)

    if e_ms > int(cache["minute_ts"][-1]):
        return None

    y = compute_label(
        cache,
        tm,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=e_ms,
        horizon_ms=horizon_ms,
    )
    if y is None:
        return None

    evs = _events_in_window(cache, s_ms, e_ms)
    raw = _compute_window_targets(evs, tm, cache, s_ms, e_ms)

    gold_norm = float(max(1e-6, float(getattr(cfg, "GOLD_NORM", 1000.0))))
    kill_norm = float(max(1e-6, float(getattr(cfg, "MTL_KILL_NORM", 5.0))))
    obj_norm = float(max(1e-6, float(getattr(cfg, "MTL_OBJ_NORM", 5.0))))

    out = {
        "y": float(int(y)),
        "kill_diff": float(raw["kill_diff"]),
        "gold_diff": float(raw["gold_diff"]),
        "obj_diff": float(raw["obj_diff"]),
        "alive_diff": float(raw["alive_diff"]),
        "summoner_spells": float(raw["summoner_spells"]),
        "kill_diff_norm": float(raw["kill_diff"] / kill_norm),
        "gold_diff_norm": float(raw["gold_diff"] / gold_norm),
        "obj_diff_norm": float(raw["obj_diff"] / obj_norm),
        "label_start_ms": float(s_ms),
        "label_end_ms": float(e_ms),
    }
    return out


def build_ms_sequence(
        cache: Dict[str, Any],
        tm: Dict[int, int],
        t_start: int,
        *,
        engage_ts: Optional[int] = None,
        label_end_ts: Optional[int] = None,
        ctx_ms: Optional[int] = None,
        bin_ms: Optional[int] = None,
        horizon_ms: Optional[int] = None,
        prediction_gap_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if ctx_ms is None:
        ctx_ms = _get_context_ms()
    if bin_ms is None:
        bin_ms = _get_bin_ms()
    if horizon_ms is None:
        horizon_ms = _get_horizon_ms()
    if prediction_gap_ms is None:
        prediction_gap_ms = int(getattr(cfg, "PREDICTION_GAP_MS", 0))
    prediction_gap_ms = max(0, int(prediction_gap_ms))

    # ── compute observation window [start_ms, end_ms] ────────────
    t_min = int(cache["minute_ts"][0])
    t_max = int(cache["minute_ts"][-1])

    label_start_ms: int
    label_end_ms: int
    if engage_ts is not None and engage_ts >= 0:
        label_start_ms = int(engage_ts)
        label_end_ms = label_start_ms + int(horizon_ms)
        if label_end_ts is not None:
            try:
                cand_end = int(label_end_ts)
            except Exception:
                cand_end = -1
            if cand_end > label_start_ms:
                label_end_ms = cand_end
        end_ms = int(label_start_ms - prediction_gap_ms)
        start_ms = end_ms - int(ctx_ms)
    else:
        if t_start < 0 or t_start >= len(cache["minute_ts"]):
            return None
        label_start_ms = int(cache["minute_ts"][t_start])
        label_end_ms = label_start_ms + int(horizon_ms)
        end_ms = int(label_start_ms - prediction_gap_ms)
        start_ms = end_ms - int(ctx_ms)

    if end_ms < t_min:
        return None
    if start_ms < t_min:
        return None
    if label_end_ms > t_max:
        return None

    L = int(ctx_ms // bin_ms)
    if L <= 0:
        return None

    glob_seq, node_seq, ev_seq, item_seq = [], [], [], []
    glob_snap_ts_seq: List[int] = []

    for i in range(L):
        b0 = start_ms + i * bin_ms
        b1 = start_ms + (i + 1) * bin_ms
        q = b0 + bin_ms // 2

        node_i, glob_i = interpolate_node_global(cache, q)
        # Enforce "previous global snapshot" for input globals:
        # use nearest available timestamp strictly before the reference ms.
        g_ref_ms = int(q)
        if engage_ts is not None and engage_ts >= 0:
            g_ref_ms = min(int(g_ref_ms), int(label_start_ms) - 1)
        glob_i, g_ts = global_from_prev_snapshot(cache, g_ref_ms, strict_before=True)
        glob_snap_ts_seq.append(int(g_ts))
        ev_i, it_i = aggregate_events(cache, tm, b0, b1)

        node_seq.append(node_i)
        glob_seq.append(glob_i)
        ev_seq.append(ev_i)
        item_seq.append(it_i)

    if engage_ts is not None and engage_ts >= 0:
        y_pack = compute_label_targets(
            cache,
            tm,
            -1,
            engage_ts=label_start_ms,
            label_end_ts=label_end_ms,
            horizon_ms=horizon_ms,
        )
    else:
        y_pack = compute_label_targets(
            cache,
            tm,
            t_start,
            engage_ts=label_start_ms,
            label_end_ts=label_end_ms,
            horizon_ms=horizon_ms,
        )

    if y_pack is None:
        return None

    y_s = int(y_pack.get("label_start_ms", label_start_ms))
    y_e = int(y_pack.get("label_end_ms", label_end_ms))
    y_dur = max(0, y_e - y_s)

    sample = {
        "node_seq": np.stack(node_seq, axis=0).astype(np.float32),
        "glob_seq": np.stack(glob_seq, axis=0).astype(np.float32),
        "ev_seq": np.stack(ev_seq, axis=0).astype(np.float32),
        "item_seq": np.stack(item_seq, axis=0).astype(np.float32),
        "y": int(y_pack["y"]),
        "y_kill_diff": float(y_pack.get("kill_diff_norm", 0.0)),
        "y_gold_diff": float(y_pack.get("gold_diff_norm", 0.0)),
        "y_obj_diff": float(y_pack.get("obj_diff_norm", 0.0)),
        "label_kill_diff_raw": float(y_pack.get("kill_diff", 0.0)),
        "label_gold_diff_raw": float(y_pack.get("gold_diff", 0.0)),
        "label_obj_diff_raw": float(y_pack.get("obj_diff", 0.0)),
        "label_alive_diff_raw": float(y_pack.get("alive_diff", 0.0)),
        "label_summoner_spells_raw": float(y_pack.get("summoner_spells", 0.0)),
        "engage_ts": int(label_start_ms),
        "obs_end_ts": int(end_ms),
        "ctx_ms": int(ctx_ms),
        "bin_ms": int(bin_ms),
        "horizon_ms": int(y_dur),
        "label_duration_ms": int(y_dur),
        "prediction_gap_ms": int(prediction_gap_ms),
        "label_start_ts": int(y_s),
        "label_end_ts": int(y_e),
        "global_snap_last_ts": int(glob_snap_ts_seq[-1]) if glob_snap_ts_seq else -1,
        "game_duration_min": float(max(1.0, (int(cache["minute_ts"][-1]) - int(cache["minute_ts"][0])) / 60000.0)),
    }

    if bool(getattr(cfg, "USE_EVENT_TOKENS", True)):
        tok = build_event_tokens_for_xattn(
            cache,
            tm,
            start_ms,
            end_ms,
            max_tokens=int(getattr(cfg, "MAX_EVENT_TOKENS", 64)),
        )
        sample.update(tok)

    anchors = cache.get("meta", {}).get("anchors", None)
    if isinstance(anchors, dict):
        sample["anchors"] = anchors
    return sample
