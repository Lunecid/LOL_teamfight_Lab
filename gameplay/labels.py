from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

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
        if et in ("CHAMPION_KILL", "CHAMPION_SPECIAL_KILL", "WARD_KILL"):
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

        # [B1] Special-kill markers (ace / multi-kill / first-blood) are a BONUS
        # on the underlying CHAMPION_KILL (paper Eq. 3, s(u)), not a second kill.
        # Riot emits a CHAMPION_SPECIAL_KILL alongside the CHAMPION_KILL for the
        # same death; scoring it as is_kill=1.0 double-counted the kill. Score
        # the special marker as the bonus only.
        if et == "CHAMPION_SPECIAL_KILL":
            sb = float(_label_special_kill_bonus(et, e))
            values.append(float(max(0.0, sb)))
            priors.append(float(sb))
            signs.append(float(sign))
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
        is_kill = 1.0 if et == "CHAMPION_KILL" else 0.0

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
            float(getattr(cfg, "LABEL_ATTN_PRIOR_W_KILL", 0.25)) * is_kill
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_SHUTDOWN", 0.30)) * shutdown_norm
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_STREAK", 0.15)) * streak_norm
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_ASSIST", 0.10)) * assist_norm
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_BOUNTY", 0.20)) * bounty_norm
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_OBJECTIVE", 0.35)) * obj_tier
            + float(getattr(cfg, "LABEL_ATTN_PRIOR_W_LANE", 0.15)) * lane_pri
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


def _event_xy(e: dict):
    pos = e.get("position", None)
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        return float(pos["x"]), float(pos["y"])
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _split_label_type(label_type: str):
    """'market_event@window' -> ('market_event', 'window'); no suffix -> cfg default."""
    lt = str(label_type or "").lower()
    if "@" in lt:
        base, attr = lt.split("@", 1)
        return base.strip(), attr.strip()
    return lt, str(getattr(cfg, "LABEL_EVENT_ATTRIBUTION", "engagement")).lower()


def attribute_events(evs: List[dict], anchor_xy, attribution: str) -> List[dict]:
    """Keep the events the engagement may claim.

    "engagement": events with a position within the attribution radius of the fight centre
    (kills, buildings, plates, monsters); events without a position (ward kills) are dropped.
    "window": every event.  Without an anchor the filter cannot be applied and all events are kept.
    """
    if str(attribution).lower() != "engagement" or anchor_xy is None:
        return list(evs)
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    if ax < 0 or ay < 0:
        return list(evs)
    radius = float(getattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 0.0) or 0.0)
    if radius <= 0:
        radius = float(getattr(cfg, "CLUSTER_MAX_DIAMETER", 4000.0) or 4000.0)
    out = []
    for e in evs:
        xy = _event_xy(e)
        if xy is None:
            continue
        if (xy[0] - ax) ** 2 + (xy[1] - ay) ** 2 <= radius * radius:
            out.append(e)
    return out


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
    anchor_xy=None,
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
    label_type, attribution = _split_label_type(str(getattr(cfg, "LABEL_TYPE", "micro_win")))
    evs = attribute_events(evs, anchor_xy, attribution)

    if getattr(cfg, "REQUIRE_SIGNAL_IN_HORIZON", False):
        has_sig = any(
            str(e.get("type", "")).upper() in ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL")
            for e in evs
        )
        if not has_sig:
            return None

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
    if label_type in ("market_lex", "market_lexicographic", "gold_lex"):
        return _compute_label_market_lex(
            evs, tm, cache, s_ms, e_ms,
            interp_node_global=interp_node_global,
            first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts,
            tie_key=tie_key,
        )
    if label_type in ("market_event", "event_market", "kill_bounty_lex"):
        return _compute_label_market_event(
            evs, tm, cache, s_ms, e_ms,
            interp_node_global=interp_node_global,
            first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts,
            tie_key=tie_key,
        )
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


def _compute_label_market_lex(
    evs: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    s_ms: int,
    e_ms: int,
    *,
    interp_node_global: InterpNodeGlobalFn,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    tie_key: str = "",
) -> Optional[int]:
    """Market verdict first, lexicographic refinement inside the dead zone.

    The window team-gold-swing difference decides whenever it exceeds
    LABEL_GOLD_DEADZONE (default 300 g, one base kill bounty): kills,
    turrets, plates and monsters are already priced in gold by the game,
    so no researcher weights are involved.  When the market is silent
    (|swing| within the dead zone), discrete material facts refine in a
    fixed order -- cluster kills, then survivors at the last kill, then
    structure events in the window -- and an engagement even on all of
    them is a genuine draw handled by the tie policy.
    """
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()
    deadzone = float(getattr(cfg, "LABEL_GOLD_DEADZONE", 300.0))

    gold_method = str(getattr(cfg, "LABEL_GOLD_METHOD", "linear")).lower()
    g0 = gold_at_ms(cache, s_ms, method=gold_method)
    g1 = gold_at_ms(cache, e_ms, method=gold_method)
    gd = float((g1[0] - g0[0]) - (g1[1] - g0[1]))
    if gd > deadzone:
        return 1
    if gd < -deadzone:
        return 0

    return _lex_refine(evs, tm, cache, e_ms, interp_node_global=interp_node_global,
                       first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts, tie_key=tie_key, tie_policy=tie_policy)


def _lex_refine(
    evs: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    e_ms: int,
    *,
    interp_node_global: InterpNodeGlobalFn,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    tie_key: str = "",
    tie_policy: str = "drop",
) -> Optional[int]:
    """Refinement inside the market's dead zone: cluster kills -> survivors at the
    last kill -> structure events in the window -> tie policy (shared by the
    market_lex and market_event labels)."""
    kd = 0
    struct = 0
    for e in evs:
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            if first_kill_ts is not None and last_kill_ts is not None:
                kill_ts = int(e.get("timestamp", 0) or 0)
                if kill_ts < first_kill_ts or kill_ts > last_kill_ts:
                    continue
            killer = int(e.get("killerId", 0) or 0)
            if tm.get(killer, 0) == 100:
                kd += 1
            elif tm.get(killer, 0) == 200:
                kd -= 1
        elif et in ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            struct += _label_event_team_sign(e, tm)
    if kd != 0:
        return 1 if kd > 0 else 0

    alive_measure_ts = e_ms if not (last_kill_ts and last_kill_ts > 0) else last_kill_ts
    node_end, _ = interp_node_global(cache, alive_measure_ts)
    alive_idx = NODE_IDX.get("alive", None)
    if alive_idx is not None:
        tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
        blue_alive = float(node_end[np.where(tids == 100)[0], alive_idx].sum())
        red_alive = float(node_end[np.where(tids == 200)[0], alive_idx].sum())
        if blue_alive != red_alive:
            return 1 if blue_alive > red_alive else 0

    if struct != 0:
        return 1 if struct > 0 else 0

    if tie_policy in ("random", "stochastic", "coinflip"):
        return _seeded_tie_coin(evs, tm, tie_key)
    if tie_policy == "blue":
        return 1
    if tie_policy == "red":
        return 0
    return None



_EVENT_PRICE_CACHE: Dict[str, Any] = {}


def _event_price_table() -> Dict[str, float]:
    """Team gold per priced event (pooled regression table), loaded once; {} = kills only."""
    rel = str(getattr(cfg, "LABEL_EVENT_PRICE_TABLE", "") or "").strip()
    if not rel:
        return {}
    if rel in _EVENT_PRICE_CACHE:
        return _EVENT_PRICE_CACHE[rel]
    from pathlib import Path as _P
    path = _P(rel)
    if not path.is_absolute():
        path = _P(__file__).resolve().parents[1] / rel
    table: Dict[str, float] = {}
    if path.exists():
        import json as _json
        blob = _json.load(open(path, encoding="utf-8"))
        table = {str(k): float(v) for k, v in (blob.get("pooled") or {}).items()}
    _EVENT_PRICE_CACHE[rel] = table
    return table


# Price-table entries paid per kill / per assist on top of the event's own bounty; every other
# entry (plates, towers, inhibitor, monsters, ward kills, first-tower bonus) is a non-kill price.
_PRICE_KILL_KEYS: Tuple[str, ...] = ("kills", "assists")


def _apply_price_table_variant(table: Dict[str, float]) -> Dict[str, float]:
    """Price-table perturbation for the label-family sensitivity (R2); the identity under the defaults.

    Every non-kill entry is multiplied by ``LABEL_EVENT_PRICE_NONKILL_SCALE``; afterwards
    ``LABEL_EVENT_PRICE_OVERRIDES`` (dict, or a JSON object string) replaces single entries
    verbatim, unscaled.  The kill gold read from the event (bounty + shutdownBounty) is never
    touched, and the loaded (cached) table is never mutated.
    """
    scale = float(getattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 1.0))
    overrides = getattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", None) or {}
    if isinstance(overrides, str):
        import json as _json
        overrides = _json.loads(overrides) if overrides.strip() else {}
    if scale == 1.0 and not overrides:
        return table
    out = {str(k): (float(v) if str(k) in _PRICE_KILL_KEYS else float(v) * scale) for k, v in table.items()}
    for k, v in dict(overrides).items():
        out[str(k)] = float(v)
    return out


def _priced_event_gold(e: dict, table: Dict[str, float], first_tower_ts: Optional[int]) -> float:
    """Team gold the game paid for a non-kill event, from the price table (0 if unpriced)."""
    if not table:
        return 0.0
    et = str(e.get("type", "")).upper()
    if et == "TURRET_PLATE_DESTROYED":
        return float(table.get("plates", 0.0))
    if et == "BUILDING_KILL":
        if str(e.get("buildingType", "")).upper() == "INHIBITOR_BUILDING":
            return float(table.get("inhibitor", 0.0))
        g = float(table.get(f"tower_{str(e.get('towerType', '')).lower()}", 0.0))
        if first_tower_ts is not None and int(e.get("timestamp", -1) or -1) == int(first_tower_ts):
            g += float(table.get("first_tower", 0.0))
        return g
    if et == "ELITE_MONSTER_KILL":
        mt = str(e.get("monsterType", "")).upper()
        st = str(e.get("monsterSubType", "")).upper()
        key = "elder_dragon" if (mt == "DRAGON" and st == "ELDER_DRAGON") else mt.lower()
        return float(table.get(key, 0.0))
    if et == "WARD_KILL":
        return float(table.get("ward_kills", 0.0))
    return 0.0


def _first_tower_ts(cache: Dict[str, Any]) -> Optional[int]:
    best = None
    for e in cache.get("events") or []:
        if str(e.get("type", "")).upper() == "BUILDING_KILL" and str(e.get("buildingType", "")).upper() != "INHIBITOR_BUILDING":
            ts = int(e.get("timestamp", 0) or 0)
            best = ts if best is None or ts < best else best
    return best


def _compute_label_market_event(
    evs: List[dict],
    tm: Dict[int, int],
    cache: Dict[str, Any],
    s_ms: int,
    e_ms: int,
    *,
    interp_node_global: InterpNodeGlobalFn,
    first_kill_ts: Optional[int] = None,
    last_kill_ts: Optional[int] = None,
    tie_key: str = "",
) -> Optional[int]:
    """Market verdict priced by the events themselves, millisecond-exact.

    Kill events carry the gold the game actually paid (``bounty`` +
    ``shutdownBounty``); structure and monster events carry no usable bounty
    in Match-V5 and stay in the refinement tier.  Same dead zone and same
    order as market_lex, so the only difference is the currency: paid kill
    gold at event resolution instead of minute-frame team gold interpolated
    linearly across the window.
    """
    tie_policy = str(getattr(cfg, "LABEL_TIE_POLICY", getattr(cfg, "LABEL_TIE_STRATEGY", "drop"))).lower()
    deadzone = float(getattr(cfg, "LABEL_GOLD_DEADZONE", 300.0))
    table = _apply_price_table_variant(_event_price_table())
    first_tower = _first_tower_ts(cache) if table else None
    gd = 0.0
    for e in evs:
        et = str(e.get("type", "")).upper()
        sign = _label_event_team_sign(e, tm)
        if sign == 0:
            continue
        if et == "CHAMPION_KILL":
            g = max(0.0, safe_float(e.get("bounty", 0.0))) + max(0.0, safe_float(e.get("shutdownBounty", 0.0)))
            assists = e.get("assistingParticipantIds", [])
            g += float(table.get("kills", 0.0)) + float(table.get("assists", 0.0)) * (len(assists) if isinstance(assists, list) else 0)
            gd += float(sign) * g
        else:
            gd += float(sign) * _priced_event_gold(e, table, first_tower)
    if gd > deadzone:
        return 1
    if gd < -deadzone:
        return 0
    return _lex_refine(evs, tm, cache, e_ms, interp_node_global=interp_node_global,
                       first_kill_ts=first_kill_ts, last_kill_ts=last_kill_ts, tie_key=tie_key, tie_policy=tie_policy)


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
    anchor_xy=None,
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
        anchor_xy=anchor_xy,
    )
    if y is None:
        return None

    evs = _events_in_window(cache, s_ms, e_ms)
    evs = attribute_events(evs, anchor_xy, _split_label_type(str(getattr(cfg, "LABEL_TYPE", "micro_win")))[1])
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


# =====================================================================================================
# Label-variant registry (ToG revision of CoG 2026 submission 118; reviewer R2, major comment on labels)
# =====================================================================================================
# R2: the engagement outcome aggregates events with hand-set weights, so a model may learn the labelling
# heuristic rather than the game; R2 asked for alternative labelling schemes (e.g. raw kill advantage or
# learned weights) and a sensitivity analysis on the weighting parameters.
#
# A variant is (label type, event attribution, cfg overrides).  The overrides hold ONLY while that variant
# is computed -- variant_cfg restores every cfg value afterwards, also when the computation raises -- so
# scripts/build_label_sidecars_v33.py computes the whole family in one pass on the same rows.  The tie
# policy is not part of a variant: the caller's LABEL_TIE_POLICY applies (the sidecars use "drop", so a
# draw is stored as -1).  Nothing below changes a default, and "market_event" carries no overrides.
#
# Definitions followed (no scheme is re-implemented here; every variant calls compute_label):
#   docs/ENGAGEMENT_WINNER_DEFINITION.md, "What each scheme computes": micro_win, kill_survival,
#       attention_value_win (CoG 2026 Eq. 3), weighted, market_lex;
#   docs/DEFINITION_EVIDENCE.md sections 18, 20, 21 and 22: market_event (event-priced gold swing, fitted
#       price table, 300 g dead zone, lexicographic refinement, engagement attribution) and the v3.3 preset.

DRAGON_RULE_TEAM_GOLD: float = 25.0
DRAGON_RULE_SOURCE: str = (
    "League of Legends Wiki, 'Dragon pit' (https://wiki.leagueoflegends.com/en-us/Dragon_pit) and "
    "'Elemental drake' (https://wiki.leagueoflegends.com/en-us/Elemental_drake), patch history V26.01: "
    "elemental drakes 'Kill gold increased to 75 from 25' (retrieved 2026-09-11). 25 g is the value V26.01 replaced."
)
DRAGON_RULE_VERIFICATION: str = (
    "UNVERIFIED for patches 15.14-15.16: the wiki dates only the V26.01 change, not the value in force in "
    "15.14-15.16, and does not say whether the kill gold is paid to the killer alone or to every member of the "
    "killing team. DRAGON_RULE_TEAM_GOLD reads it as 25 g of team gold (killer only); the per-member reading "
    "(5 x 25 = 125 g) is the variant market_event_dragon_rule_per_member. Supporting the killer-only reading "
    "(re-checked 2026-09-14): the wiki's 'Baron Nashor' V26.01 entry separates 'kill gold' (25 -> 100) from "
    "'global gold' (300 -> 150), and the fitted team price of Baron Nashor (1,555 g) is close to 5 x 300 + 25 = "
    "1,525 g, i.e. global gold is paid to every member and kill gold once; the elemental drake carries kill gold "
    "only. Still no dated source for 15.14-15.16."
)
# D / 2 for the v3.3 cluster diameter D = CLUSTER_MAX_DIAMETER = 4,264 u.  attribute_events uses D itself as the
# disc RADIUS when LABEL_ATTRIBUTION_RADIUS_U is 0, so the half-radius disc is the disc whose diameter is D.
ATTRIBUTION_RADIUS_HALF_V33_U: float = 2132.0

_VARIANT_LABEL_TYPES: Tuple[str, ...] = (
    "micro_win", "kill_survival", "attention_value_win", "market_lex", "market_event", "weighted",
)
_VARIANT_ATTRIBUTIONS: Tuple[str, ...] = ("engagement", "window")
_CFG_FIELD_NAMES: Optional[frozenset] = None
_MISSING = object()

_PRICES_SOURCE = ("config/game_rules/event_prices.json: regression-estimated average team gold per event "
                  "(24,000 matches, patches 15.14-15.16; docs/DEFINITION_EVIDENCE.md section 20)")


class _FrozenMap(tuple):
    """Immutable stand-in for a dict-valued override: sorted (key, value) pairs, thawed by overrides_dict."""


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMap(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, _FrozenMap):
        return {k: _thaw(v) for k, v in value}
    return value


def _cfg_field_names() -> frozenset:
    global _CFG_FIELD_NAMES
    if _CFG_FIELD_NAMES is None:
        import dataclasses as _dc
        _CFG_FIELD_NAMES = frozenset(f.name for f in _dc.fields(type(cfg)))
    return _CFG_FIELD_NAMES


@dataclass(frozen=True)
class LabelVariant:
    """One member of the label family.

    ``label_type`` is a compute_label scheme, ``attribution`` "engagement" or "window" (applied as
    ``LABEL_TYPE = "<label_type>@<attribution>"``), ``overrides`` (key, value) pairs of CFG fields set only
    while the variant is computed.  ``stored``: corpus_shards_v33 stores ``y_<name>``, so the sidecar build
    checks this variant row for row.  ``required``: one of the R2 label family scored by default.
    """

    name: str
    label_type: str
    attribution: str = "engagement"
    overrides: Tuple[Tuple[str, Any], ...] = ()
    description: str = ""
    source: str = ""
    verification: str = ""
    stored: bool = False
    required: bool = False

    @property
    def cfg_label_type(self) -> str:
        return f"{self.label_type}@{self.attribution}"

    def overrides_dict(self) -> Dict[str, Any]:
        return {k: _thaw(v) for k, v in self.overrides}

    def validate(self) -> None:
        if self.label_type not in _VARIANT_LABEL_TYPES:
            raise ValueError(f"variant {self.name!r}: unknown label type {self.label_type!r}; known {_VARIANT_LABEL_TYPES}")
        if self.attribution not in _VARIANT_ATTRIBUTIONS:
            raise ValueError(f"variant {self.name!r}: attribution {self.attribution!r} not in {_VARIANT_ATTRIBUTIONS}")
        fields = _cfg_field_names()
        bad = [k for k, _ in self.overrides if k not in fields or k == "LABEL_TYPE"]
        if bad:
            raise KeyError(f"variant {self.name!r}: overrides {bad} are not overridable CFG fields "
                           "(LABEL_TYPE is set from label_type/attribution)")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "label_type": self.label_type, "attribution": self.attribution,
            "cfg_label_type": self.cfg_label_type, "overrides": self.overrides_dict(),
            "description": self.description, "source": self.source, "verification": self.verification,
            "stored": bool(self.stored), "required": bool(self.required),
        }


def _variant(name: str, label_type: str, attribution: str = "engagement",
             overrides: Optional[Dict[str, Any]] = None, **meta: Any) -> LabelVariant:
    return LabelVariant(name=name, label_type=label_type, attribution=attribution,
                        overrides=tuple((str(k), _freeze(v)) for k, v in (overrides or {}).items()), **meta)


_ME = ("market_event: team gold swing over the label window from the attributed events -- kill gold read from the "
       "event (bounty + shutdownBounty) plus the fitted per-kill and per-assist prices, plates / turrets / inhibitor / "
       "monsters / ward kills at the fitted price table (elemental dragon clamped to 0 g); beyond the dead zone the sign "
       "decides, inside it cluster kills, then survivors at the last kill, then structure events refine")

_VARIANT_LIST: List[LabelVariant] = [
    # --- reference and the columns corpus_shards_v33 stores (checked row for row by the sidecar build) ---
    _variant("market_event", "market_event", stored=True, required=True,
             description=_ME + ". v3.3 headline: 300 g dead zone, events within CLUSTER_MAX_DIAMETER (4,264 u, used as "
                               "the radius) of the anchor.",
             source=_PRICES_SOURCE),
    _variant("market_event@window", "market_event", "window", stored=True,
             description="market_event over every event in the time window (no spatial attribution).",
             source=_PRICES_SOURCE),
    _variant("market_lex", "market_lex", stored=True,
             description="Minute-frame team gold swing (linear interpolation at the window ends) beyond the 300 g dead "
                         "zone, refined by attributed cluster kills -> survivors -> structures."),
    _variant("market_lex@window", "market_lex", "window", stored=True,
             description="market_lex with every event of the window in the refinement."),
    _variant("attention_value_win", "attention_value_win", stored=True,
             description="CoG 2026 Eq. 3 attention-weighted event value (hand-set LABEL_ATTN_* weights) over the "
                         "attributed events, as stored in corpus v3.3.",
             verification="hand-set weights (the heuristic R2 questioned)"),
    # --- R2 label family: alternative schemes ---
    _variant("micro_win", "micro_win", required=True,
             description="Raw kill advantage (R2's named alternative): blue minus red champion kills timestamped in "
                         "[first cluster kill, last cluster kill] among the attributed events; equal counts are draws. "
                         "No gold, objectives or weights."),
    _variant("kill_survival", "kill_survival", required=True,
             description="LABEL_W_KILL (1.0) x kill difference (cluster kills) + LABEL_W_ALIVE (0.3) x alive difference "
                         "at the last cluster kill; |score| < 1e-8 is a draw.",
             verification="hand-set weights 1.0 / 0.3"),
    # --- R2 sensitivity: dead zone ---
    *[_variant(f"market_event_dz{int(dz)}", "market_event", overrides={"LABEL_GOLD_DEADZONE": float(dz)}, required=True,
               description=f"market_event with LABEL_GOLD_DEADZONE = {int(dz)} g (v3.3: 300 g, one base kill bounty)"
                           + ("; any non-zero swing decides and only an exactly zero swing is refined." if dz == 0 else "."),
               source=_PRICES_SOURCE)
      for dz in (0.0, 150.0, 600.0, 900.0)],
    # --- R2 sensitivity: prices ---
    _variant("market_event_kills_only", "market_event", overrides={"LABEL_EVENT_PRICE_TABLE": ""}, required=True,
             description="market_event without the price table: the swing is the kill gold read from the events alone "
                         "(bounty + shutdownBounty; no fitted per-kill / per-assist prices, every other event 0 g). "
                         "Structures still enter the refinement tier."),
    _variant("market_event_prices_x0.5", "market_event", overrides={"LABEL_EVENT_PRICE_NONKILL_SCALE": 0.5}, required=True,
             description="market_event with every NON-kill price (plates, turrets, inhibitor, monsters, ward kills, "
                         "first-tower bonus) x 0.5; kill gold and the per-kill / per-assist prices unchanged.",
             source=_PRICES_SOURCE),
    _variant("market_event_prices_x2", "market_event", overrides={"LABEL_EVENT_PRICE_NONKILL_SCALE": 2.0}, required=True,
             description="market_event with every NON-kill price x 2; kill gold and the per-kill / per-assist prices "
                         "unchanged.",
             source=_PRICES_SOURCE),
    _variant("market_event_dragon_rule", "market_event",
             overrides={"LABEL_EVENT_PRICE_OVERRIDES": {"dragon": DRAGON_RULE_TEAM_GOLD}}, required=True,
             description=f"market_event with an elemental dragon priced at the rule payout {DRAGON_RULE_TEAM_GOLD:g} g of team "
                         "gold instead of the fitted, clamped 0 g (Elder Dragon keeps its fitted price).",
             source=DRAGON_RULE_SOURCE, verification=DRAGON_RULE_VERIFICATION),
    # --- R2 sensitivity: attribution disc ---
    _variant("market_event_attr_radius_half", "market_event",
             overrides={"LABEL_ATTRIBUTION_RADIUS_U": ATTRIBUTION_RADIUS_HALF_V33_U}, required=True,
             description="market_event with the attribution disc radius D/2 = 2,132 u instead of D = 4,264 u (the default "
                         "disc uses the cluster diameter as its radius).",
             source=_PRICES_SOURCE),
    # --- optional extras (computed by the sidecars, scored on request) ---
    _variant("market_event_dragon_rule_per_member", "market_event",
             overrides={"LABEL_EVENT_PRICE_OVERRIDES": {"dragon": 5.0 * DRAGON_RULE_TEAM_GOLD}},
             description=f"market_event with an elemental dragon priced at {5.0 * DRAGON_RULE_TEAM_GOLD:g} g: the per-member "
                         f"reading of the {DRAGON_RULE_TEAM_GOLD:g} g kill gold (five members).",
             source=DRAGON_RULE_SOURCE, verification=DRAGON_RULE_VERIFICATION),
    _variant("attention_value_win@window", "attention_value_win", "window",
             description="The CoG 2026 label as published: Eq. 3 over every event in the window, on the v3.3 rows.",
             verification="hand-set weights (the heuristic R2 questioned)"),
    _variant("weighted", "weighted",
             description="Pre-CoG composite: W_KILL (1.0) x kill difference + W_GOLD (0.5) x frame gold swing / GOLD_NORM "
                         "(500) + W_OBJ (0.25) x OBJ_SCORE-weighted objectives, over the attributed events.",
             verification="hand-set weights"),
]

LABEL_VARIANTS: Dict[str, LabelVariant] = {v.name: v for v in _VARIANT_LIST}
if len(LABEL_VARIANTS) != len(_VARIANT_LIST):
    raise RuntimeError("duplicate label-variant names")
REQUIRED_LABEL_VARIANTS: Tuple[str, ...] = tuple(v.name for v in _VARIANT_LIST if v.required)
STORED_LABEL_VARIANTS: Tuple[str, ...] = tuple(v.name for v in _VARIANT_LIST if v.stored)


def get_label_variant(variant: Union[str, LabelVariant]) -> LabelVariant:
    """Registry lookup by name (a LabelVariant passes through); KeyError for unknown names."""
    if isinstance(variant, LabelVariant):
        return variant
    try:
        return LABEL_VARIANTS[str(variant)]
    except KeyError:
        raise KeyError(f"unknown label variant {variant!r}; known: {sorted(LABEL_VARIANTS)}") from None


@contextmanager
def cfg_override(values: Dict[str, Any]) -> Iterator[None]:
    """Set cfg attributes for the duration of the block.

    On exit -- also when the block raises -- every value is restored and attributes the instance did not
    hold before are removed again.  Any attribute name is accepted (e.g. LABEL_TIE_POLICY, which is not a
    CFG field); variant_cfg validates variant overrides against the CFG fields first.
    """
    saved: List[Tuple[str, Any]] = []
    try:
        for k, v in dict(values).items():
            saved.append((k, cfg.__dict__.get(k, _MISSING)))
            setattr(cfg, k, v)
        yield
    finally:
        for k, old in reversed(saved):
            if old is _MISSING:
                try:
                    delattr(cfg, k)
                except AttributeError:
                    pass
            else:
                setattr(cfg, k, old)


@contextmanager
def variant_cfg(variant: Union[str, LabelVariant]) -> Iterator[LabelVariant]:
    """cfg as the variant computes it: LABEL_TYPE = "<label_type>@<attribution>" plus the overrides; restored after."""
    v = get_label_variant(variant)
    v.validate()
    values: Dict[str, Any] = {"LABEL_TYPE": v.cfg_label_type}
    values.update(v.overrides_dict())
    with cfg_override(values):
        yield v


def compute_label_variant(variant: Union[str, LabelVariant], cache: Dict[str, Any], tm: Dict[int, int],
                          t_start: int, **kwargs: Any) -> Optional[int]:
    """compute_label under one registry variant (same keyword arguments as compute_label)."""
    with variant_cfg(variant):
        return compute_label(cache, tm, t_start, **kwargs)
