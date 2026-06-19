from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import cfg, NODE_IDX, OBJ_SCORE
from core.common import np, safe_float
from core.timeutils import _get_horizon_ms, gold_at_ms
from data.events_index import _events_in_window

InterpNodeGlobalFn = Callable[[Dict[str, Any], int], Tuple[np.ndarray, np.ndarray]]


_TIE_EVENT_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "type",
    "killerId",
    "victimId",
    "killerTeamId",
    "teamId",
    "monsterType",
    "monsterSubType",
    "buildingType",
    "towerType",
    "laneType",
    "shutdownBounty",
    "killStreakLength",
    "bounty",
    "multiKillLength",
    "assistingParticipantIds",
)


def _tie_repr(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((str(k), _tie_repr(val)) for k, val in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_tie_repr(x) for x in v)
    return v


def _seeded_tie_coin(evs: List[dict], tm: Dict[int, int], tie_key: str = "") -> int:
    """Deterministic replacement for the paper's seeded tie coin flip."""
    try:
        seed = int(getattr(cfg, "LABEL_TIE_SEED", 7))
    except Exception:
        seed = 7

    h = hashlib.blake2b(digest_size=8)
    h.update(f"{seed}|{tie_key}".encode("utf-8", errors="ignore"))
    for e in evs:
        if not isinstance(e, dict):
            continue
        sign = _label_event_team_sign(e, tm)
        payload = tuple((k, _tie_repr(e.get(k, None))) for k in _TIE_EVENT_FIELDS)
        h.update(repr((sign, payload)).encode("utf-8", errors="backslashreplace"))
    return int(int.from_bytes(h.digest(), byteorder="little", signed=False) & 1)


def _label_lane_tag(e: dict) -> str:
    lane = str(e.get("laneType", "")).upper()
    if "MID" in lane:
        return "MID"
    if "BOT" in lane or "BOTTOM" in lane:
        return "BOT"
    if "TOP" in lane:
        return "TOP"
    if "RIVER" in lane:
        return "RIVER"
    if "JUNGLE" in lane:
        return "JUNGLE"

    pos = e.get("position", {})
    if not isinstance(pos, dict):
        return "UNKNOWN"
    x = safe_float(pos.get("x", 0.0))
    y = safe_float(pos.get("y", 0.0))
    if x <= 0.0 and y <= 0.0:
        return "UNKNOWN"

    d = y - x
    if abs(d) <= 2200.0:
        return "MID"
    if d > 0.0:
        return "TOP"
    return "BOT"


def _label_lane_priority(et: str, e: dict) -> float:
    lane = _label_lane_tag(e)
    if lane == "MID":
        return 1.00
    if lane == "BOT":
        return 0.80
    if lane == "TOP":
        return 0.70
    if lane == "RIVER":
        return 0.90
    if lane == "JUNGLE":
        return 0.75

    mt = str(e.get("monsterType", "")).upper()
    if mt in ("BARON_NASHOR", "DRAGON", "RIFTHERALD", "ATAKHAN", "HORDE"):
        return 0.95
    if et == "CHAMPION_KILL":
        if safe_float(e.get("shutdownBounty", 0.0)) > 0.0:
            return 0.95
        return 0.60
    return 0.50


def _label_objective_tier(et: str, e: dict) -> float:
    if et == "ELITE_MONSTER_KILL":
        mt = str(e.get("monsterType", "")).upper()
        sub = str(e.get("monsterSubType", "")).upper()
        if mt == "BARON_NASHOR":
            return 1.0
        if mt == "DRAGON":
            return 1.0 if ("ELDER" in sub) else 0.75
        if mt == "RIFTHERALD":
            return 0.70
        if mt == "ATAKHAN":
            return 0.85
        if mt == "HORDE":
            return 0.60
    elif et == "BUILDING_KILL":
        bt = str(e.get("buildingType", "")).upper()
        tt = str(e.get("towerType", "")).upper()
        if "NEXUS" in bt:
            return 1.0
        if "INHIBITOR" in bt:
            return 0.85
        if "TOWER" in bt:
            if "BASE" in tt:
                return 0.75
            if "INNER" in tt:
                return 0.60
            if "OUTER" in tt:
                return 0.45
            return 0.55
    elif et == "TURRET_PLATE_DESTROYED":
        return 0.35
    elif et == "DRAGON_SOUL_GIVEN":
        return 1.0
    elif et == "CHAMPION_KILL":
        return 0.40
    return 0.0


def _label_special_kill_bonus(et: str, e: dict) -> float:
    if et != "CHAMPION_SPECIAL_KILL":
        return 0.0
    score = 0.0
    kt = str(e.get("killType", "")).upper()
    if "ACE" in kt:
        score += 0.45
    elif "MULTI" in kt:
        score += 0.25
    elif "FIRST_BLOOD" in kt:
        score += 0.20
    mk = float(np.clip(safe_float(e.get("multiKillLength", 0.0)) / 4.0, 0.0, 1.0))
    score += 0.20 * mk
    return float(np.clip(score, 0.0, 1.0))


def _label_event_team_sign(e: dict, tm: Dict[int, int]) -> int:
    et = str(e.get("type", "")).upper()
    tid = 0
    try:
        if et in ("CHAMPION_KILL", "CHAMPION_SPECIAL_KILL"):
            tid = int(tm.get(int(e.get("killerId", 0) or 0), 0) or 0)
        elif et == "ELITE_MONSTER_KILL":
            tid = int(e.get("killerTeamId", 0) or 0)
        elif et in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            victim_team = int(e.get("teamId", 0) or 0)
            if victim_team == 100:
                tid = 200
            elif victim_team == 200:
                tid = 100
            else:
                tid = 0
        elif et == "DRAGON_SOUL_GIVEN":
            tid = int(e.get("teamId", 0) or 0)
    except Exception:
        tid = 0

    if tid == 100:
        return 1
    if tid == 200:
        return -1
    return 0


def _compute_label_attention_value_win(evs: List[dict], tm: Dict[int, int], *, tie_key: str = "") -> Optional[int]:
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()
    eps = float(getattr(cfg, "LABEL_TIE_EPS", 1e-8))

    beta = float(max(1e-6, float(getattr(cfg, "LABEL_ATTN_BETA", 2.0))))
    w_kill = float(getattr(cfg, "LABEL_ATTN_W_KILL", 1.0))
    w_shutdown = float(getattr(cfg, "LABEL_ATTN_W_SHUTDOWN", 1.6))
    w_streak = float(getattr(cfg, "LABEL_ATTN_W_STREAK", 0.35))
    w_assist = float(getattr(cfg, "LABEL_ATTN_W_ASSIST", 0.20))
    w_bounty = float(getattr(cfg, "LABEL_ATTN_W_BOUNTY", 0.30))
    w_obj = float(getattr(cfg, "LABEL_ATTN_W_OBJECTIVE", 1.10))
    w_lane = float(getattr(cfg, "LABEL_ATTN_W_LANE", 0.25))

    values: List[float] = []
    priors: List[float] = []
    signs: List[float] = []

    for e in evs:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type", "")).upper()
        sign = _label_event_team_sign(e, tm)
        if sign == 0:
            continue

        shutdown = max(0.0, safe_float(e.get("shutdownBounty", 0.0)))
        shutdown_norm = float(np.clip(np.log1p(shutdown) / np.log1p(1500.0), 0.0, 1.0))
        streak_norm = float(np.clip(safe_float(e.get("killStreakLength", 0.0)) / 10.0, 0.0, 1.0))
        assists = e.get("assistingParticipantIds", [])
        assist_cnt = len(assists) if isinstance(assists, list) else 0
        assist_norm = float(np.clip(float(assist_cnt) / 4.0, 0.0, 1.0))
        bounty_raw = max(0.0, safe_float(e.get("bounty", 0.0))) + shutdown
        bounty_norm = float(np.clip(np.log1p(bounty_raw) / np.log1p(1500.0), 0.0, 1.0))
        obj_tier = float(_label_objective_tier(et, e))
        lane_pri = float(_label_lane_priority(et, e))
        special_bonus = float(_label_special_kill_bonus(et, e))
        is_kill = 1.0 if et in ("CHAMPION_KILL", "CHAMPION_SPECIAL_KILL") else 0.0

        value_e = (
            w_kill * is_kill
            + w_shutdown * shutdown_norm
            + w_streak * streak_norm
            + w_assist * assist_norm
            + w_bounty * bounty_norm
            + w_obj * obj_tier
            + w_lane * lane_pri
            + special_bonus
        )
        prior_e = (
            0.25 * is_kill
            + 0.30 * shutdown_norm
            + 0.15 * streak_norm
            + 0.10 * assist_norm
            + 0.20 * bounty_norm
            + 0.35 * obj_tier
            + 0.15 * lane_pri
            + special_bonus
        )

        values.append(float(max(0.0, value_e)))
        priors.append(float(prior_e))
        signs.append(float(sign))

    if len(values) == 0:
        return None

    pri = np.asarray(priors, dtype=np.float64)
    logits = beta * pri
    logits = logits - float(np.max(logits))
    w = np.exp(logits)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return None
    alpha = w / w_sum

    val = np.asarray(values, dtype=np.float64)
    sgn = np.asarray(signs, dtype=np.float64)
    score = float(np.sum(alpha * sgn * val))

    if abs(score) < eps:
        if tie_policy in ("drop", "exclude", "none"):
            return None
        if tie_policy in ("random", "stochastic", "coinflip"):
            return _seeded_tie_coin(evs, tm, tie_key)
        if tie_policy == "blue":
            return 1
        if tie_policy == "red":
            return 0
        return None

    return 1 if score > 0.0 else 0


def _resolve_label_window(
    cache: Dict[str, Any],
    t_start: int,
    *,
    engage_ts: Optional[int],
    label_end_ts: Optional[int],
    horizon_ms: Optional[int],
) -> Optional[Tuple[int, int, int]]:
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
    return s_ms, e_ms, int(horizon_ms)


def compute_label(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    t_start: int,
    *,
    engage_ts: Optional[int] = None,
    label_end_ts: Optional[int] = None,
    horizon_ms: Optional[int] = None,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    interp_node_global: InterpNodeGlobalFn,
) -> Optional[int]:
    win = _resolve_label_window(
        cache,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=label_end_ts,
        horizon_ms=horizon_ms,
    )
    if win is None:
        return None
    s_ms, e_ms, _ = win

    evs = _events_in_window(cache, s_ms, e_ms)

    if getattr(cfg, "REQUIRE_SIGNAL_IN_HORIZON", False):
        has_sig = any(
            str(e.get("type", "")).upper() in ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL")
            for e in evs
        )
        if not has_sig:
            return None

    label_type = str(getattr(cfg, "LABEL_TYPE", "micro_win")).lower()
    tie_key = f"{s_ms}:{e_ms}:{first_kill_ts if first_kill_ts is not None else -1}:{last_kill_ts if last_kill_ts is not None else -1}"

    if label_type == "micro_win":
        return _compute_label_micro_win(
            evs, tm,
            first_kill_ts=first_kill_ts,
            last_kill_ts=last_kill_ts,
            tie_key=tie_key,
        )
    if label_type == "kill_survival":
        return _compute_label_kill_survival(
            evs, tm, cache, e_ms,
            interp_node_global=interp_node_global,
            first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts,
            tie_key=tie_key,
        )
    if label_type in ("attention_value_win", "attention_value", "attn_value", "attn"):
        return _compute_label_attention_value_win(evs, tm, tie_key=tie_key)
    if label_type in ("weighted", "composite", "weight"):
        return _compute_label_weighted(evs, tm, cache, s_ms, e_ms, tie_key=tie_key)
    return _compute_label_kill_survival(
        evs, tm, cache, e_ms,
        interp_node_global=interp_node_global,
        first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts,
    )


def _compute_label_micro_win(
    evs: List[dict],
    tm: Dict[int, int],
    *,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    tie_key: str = "",
) -> Optional[int]:
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

    blue_kills = 0
    red_kills = 0
    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            # Cluster-scoped kill filtering
            if first_kill_ts is not None and last_kill_ts is not None:
                kill_ts = int(e.get("timestamp", 0) or 0)
                if kill_ts < first_kill_ts or kill_ts > last_kill_ts:
                    continue
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                blue_kills += 1
            elif tm.get(killer, 0) == 200:
                red_kills += 1

    if blue_kills == red_kills:
        if tie_policy in ("drop", "exclude", "none"):
            return None
        if tie_policy in ("random", "stochastic", "coinflip"):
            return _seeded_tie_coin(evs, tm, tie_key)
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
    e_ms: int,
    *,
    interp_node_global: InterpNodeGlobalFn,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    tie_key: str = "",
) -> Optional[int]:
    w_kill = float(getattr(cfg, "LABEL_W_KILL", 1.0))
    w_alive = float(getattr(cfg, "LABEL_W_ALIVE", 0.3))

    eps = float(getattr(cfg, "LABEL_TIE_EPS", 1e-8))
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

    blue_kills = 0
    red_kills = 0

    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            # Cluster-scoped kill filtering
            if first_kill_ts is not None and last_kill_ts is not None:
                kill_ts = int(e.get("timestamp", 0) or 0)
                if kill_ts < first_kill_ts or kill_ts > last_kill_ts:
                    continue
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                blue_kills += 1
            elif tm.get(killer, 0) == 200:
                red_kills += 1

    kill_diff = blue_kills - red_kills

    # Measure alive state at last_kill_ts if available (cluster end)
    alive_measure_ts = e_ms
    if last_kill_ts is not None and last_kill_ts > 0:
        alive_measure_ts = last_kill_ts

    node_end, _ = interp_node_global(cache, alive_measure_ts)
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
            return _seeded_tie_coin(evs, tm, tie_key)

        return None

    return 1 if score > 0 else 0


def _compute_label_weighted(
    evs: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    s_ms: int,
    e_ms: int,
    *,
    tie_key: str = "",
) -> Optional[int]:
    eps = float(getattr(cfg, "LABEL_TIE_EPS", 1e-8))
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()

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

    if abs(score) < eps:
        if tie_policy in ("drop", "exclude", "none"):
            return None
        if tie_policy in ("random", "stochastic", "coinflip"):
            return _seeded_tie_coin(evs, tm, tie_key)
        if tie_policy == "blue":
            return 1
        if tie_policy == "red":
            return 0
        return None

    return 1 if score > 0 else 0


def _compute_window_targets(
    evs: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    s_ms: int,
    e_ms: int,
    *,
    interp_node_global: InterpNodeGlobalFn,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
) -> Dict[str, float]:
    kill_diff = 0.0
    obj_diff = 0.0
    summoner_spells = 0.0

    for e in evs:
        et = str(e.get("type", "")).upper()

        if et == "CHAMPION_KILL":
            # Cluster-scoped kill filtering
            if first_kill_ts is not None and last_kill_ts is not None:
                kill_ts = int(e.get("timestamp", 0) or 0)
                if kill_ts < first_kill_ts or kill_ts > last_kill_ts:
                    continue
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

    alive_diff = 0.0
    try:
        node_end, _ = interp_node_global(cache, e_ms)
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
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    interp_node_global: InterpNodeGlobalFn,
) -> Optional[Dict[str, float]]:
    win = _resolve_label_window(
        cache,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=label_end_ts,
        horizon_ms=horizon_ms,
    )
    if win is None:
        return None
    s_ms, e_ms, horizon_ms = win

    y = compute_label(
        cache,
        tm,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=e_ms,
        horizon_ms=horizon_ms,
        first_kill_ts=first_kill_ts,
        last_kill_ts=last_kill_ts,
        interp_node_global=interp_node_global,
    )
    if y is None:
        return None

    evs = _events_in_window(cache, s_ms, e_ms)
    raw = _compute_window_targets(
        evs, tm, cache, s_ms, e_ms,
        interp_node_global=interp_node_global,
        first_kill_ts=first_kill_ts,
        last_kill_ts=last_kill_ts,
    )

    gold_norm = float(max(1e-6, float(getattr(cfg, "GOLD_NORM", 500.0))))
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
