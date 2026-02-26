from __future__ import annotations

import zlib
from typing import Any, Dict

from core.config import ITEM_HASH_DIM, cfg
from core.common import np, safe_float


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
    from data.events_index import _event_ts_safe, _events_in_window

    cont_dim = max(5, int(getattr(cfg, "EVENT_CONT_DIM", 12)))

    evs = _events_in_window(pack, int(s_ms), int(e_ms))
    if not evs:
        return {
            "event_type": np.zeros((max_tokens,), np.int64),
            "event_actor": np.zeros((max_tokens,), np.int64),
            "event_team": np.full((max_tokens,), 2, np.int64),
            "event_cont": np.zeros((max_tokens, cont_dim), np.float32),
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

    k_tok = int(max_tokens)
    type_vocab = int(getattr(cfg, "EVENT_TYPE_VOCAB", 128))
    coord_div = float(getattr(cfg, "COORD_NORM_DIV", 16000.0))
    denom = max(1.0, float(int(e_ms) - int(s_ms)))

    event_type = np.zeros((k_tok,), np.int64)
    event_actor = np.zeros((k_tok,), np.int64)
    event_team = np.full((k_tok,), 2, np.int64)
    event_cont = np.zeros((k_tok, cont_dim), np.float32)
    event_mask = np.zeros((k_tok,), np.float32)

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

    def _lane_priority(et: str, e: dict) -> float:
        lane = str(e.get("laneType", "")).upper()
        if "MID" in lane:
            return 1.0
        if "BOT" in lane or "BOTTOM" in lane:
            return 0.80
        if "TOP" in lane:
            return 0.70
        if "RIVER" in lane:
            return 0.90
        if "JUNGLE" in lane:
            return 0.75

        mt = str(e.get("monsterType", "")).upper()
        if mt in ("BARON_NASHOR", "DRAGON", "RIFTHERALD", "ATAKHAN", "HORDE"):
            return 0.95
        if et == "CHAMPION_KILL":
            if safe_float(e.get("shutdownBounty", 0.0)) > 0.0:
                return 0.95
            return 0.60
        return 0.50

    def _objective_tier(et: str, e: dict) -> float:
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
        elif et == "CHAMPION_KILL":
            return 0.40
        elif et == "DRAGON_SOUL_GIVEN":
            return 1.0
        return 0.0

    def _importance_prior(
        et: str,
        e: dict,
        *,
        shutdown_flag: float,
        shutdown_norm: float,
        streak_norm: float,
        assist_norm: float,
        obj_tier: float,
        lane_pri: float,
    ) -> float:
        score = 0.10
        score += 0.55 * float(shutdown_flag)
        score += 0.20 * float(streak_norm)
        score += 0.15 * float(assist_norm)
        score += 0.45 * float(obj_tier)
        score += 0.20 * float(lane_pri)
        score += 0.15 * float(shutdown_norm)

        if et == "CHAMPION_SPECIAL_KILL":
            kt = str(e.get("killType", "")).upper()
            if "ACE" in kt:
                score += 0.45
            elif "MULTI" in kt:
                score += 0.25
            elif "FIRST_BLOOD" in kt:
                score += 0.20
            mk = float(np.clip(safe_float(e.get("multiKillLength", 0.0)) / 4.0, 0.0, 1.0))
            score += 0.20 * mk

        if et in ("GAME_END", "DRAGON_SOUL_GIVEN"):
            score = max(score, 1.0)

        return float(np.clip(score, 0.0, 1.0))

    for i, (ts, e) in enumerate(evs2):
        et = str(e.get("type", "")).upper()
        event_type[i] = int(_hash32(et) % max(1, type_vocab))

        pid = _pick_pid(e)
        event_actor[i] = int(pid)

        tid = tm.get(pid, 0)
        if tid not in (100, 200):
            try:
                if et in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                    victim_team = int(e.get("teamId", 0) or 0)
                    if victim_team == 100:
                        tid = 200
                    elif victim_team == 200:
                        tid = 100
                    else:
                        tid = 0
                else:
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

        shutdown = max(0.0, safe_float(e.get("shutdownBounty", 0.0)))
        shutdown_flag = 1.0 if shutdown > 0.0 else 0.0
        shutdown_norm = float(np.clip(np.log1p(shutdown) / np.log1p(1500.0), 0.0, 1.0))
        streak_norm = float(np.clip(safe_float(e.get("killStreakLength", 0.0)) / 10.0, 0.0, 1.0))
        assists = e.get("assistingParticipantIds", [])
        assist_cnt = len(assists) if isinstance(assists, list) else 0
        assist_norm = float(np.clip(float(assist_cnt) / 4.0, 0.0, 1.0))
        obj_tier = _objective_tier(et, e)
        lane_pri = _lane_priority(et, e)

        val = 0.0
        for kk in ("bounty", "shutdownBounty", "goldGain", "xpGain"):
            if kk in e:
                val += safe_float(e.get(kk, 0.0))
        val = float(np.log1p(max(0.0, val)))
        imp_prior = _importance_prior(
            et,
            e,
            shutdown_flag=shutdown_flag,
            shutdown_norm=shutdown_norm,
            streak_norm=streak_norm,
            assist_norm=assist_norm,
            obj_tier=obj_tier,
            lane_pri=lane_pri,
        )

        feat = np.asarray(
            [
                t_rel,
                dt_end,
                x,
                y,
                val,
                shutdown_flag,
                shutdown_norm,
                streak_norm,
                assist_norm,
                obj_tier,
                lane_pri,
                imp_prior,
            ],
            np.float32,
        )
        d = min(cont_dim, int(feat.shape[0]))
        if d > 0:
            event_cont[i, :d] = feat[:d]
        event_mask[i] = 1.0

    return {
        "event_type": event_type,
        "event_actor": event_actor,
        "event_team": event_team,
        "event_cont": event_cont,
        "event_mask": event_mask,
    }
