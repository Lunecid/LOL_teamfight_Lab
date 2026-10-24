"""Object-local parsers for ELITE_MONSTER_KILL and BUILDING_KILL timing events."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from . import config as C


def parse_timestamp_ms(value: Any) -> tuple[Optional[int], str]:
    """Strict millisecond timestamp. Reject bools and fractional floats (no truncation)."""
    if isinstance(value, bool):
        return None, "boolean_timestamp"
    if isinstance(value, int):
        if value < 0:
            return None, "negative_timestamp"
        return int(value), ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, "nonfinite_timestamp"
        if value != math.trunc(value):
            return None, "fractional_timestamp"
        if value < 0:
            return None, "negative_timestamp"
        return int(value), ""
    return None, "non_integral_timestamp"


def parse_team_id(value: Any) -> tuple[int, str]:
    """Return (team, reason). Unknown stays 0; never invent 100/200."""
    if isinstance(value, bool):
        return C.TEAM_UNKNOWN, "boolean_team"
    if isinstance(value, int):
        return int(value), ""
    if isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            return C.TEAM_UNKNOWN, "non_integral_team"
        return int(value), ""
    if value is None:
        return C.TEAM_UNKNOWN, "missing_team"
    return C.TEAM_UNKNOWN, "non_integral_team"


def killer_id_to_team(killer_id: Any, blue_pids: Sequence[int]) -> int:
    """Map killer participant id to 100/200 when roster known; else 0."""
    tid, reason = parse_team_id(killer_id)
    if reason or tid <= 0:
        return C.TEAM_UNKNOWN
    blue = {int(p) for p in blue_pids if 1 <= int(p) <= 10}
    if 1 <= tid <= 10:
        return C.TEAM_BLUE if tid in blue else C.TEAM_RED
    return C.TEAM_UNKNOWN


def resolve_monster_acquisition(
    event: Mapping[str, Any], blue_pids: Sequence[int]
) -> dict[str, Any]:
    """Resolve monster acquisition evidence without promoting explicit neutral 300.

    Rules:
    - killerTeamId 100/200 → confirmed (roster conflict → contradictory)
    - killerTeamId 300 → neutral; never roster-promote
    - missing/invalid (not 300) → roster fallback only for killerId in 1..10
    """
    raw_kt, kt_reason = parse_team_id(event.get("killerTeamId"))
    raw_kid, kid_reason = parse_team_id(event.get("killerId"))
    roster = killer_id_to_team(raw_kid if not kid_reason else None, blue_pids)

    out = {
        "team": C.TEAM_UNKNOWN,
        "acq_status": C.ACQ_UNKNOWN,
        "team_source": "unknown_monster_team",
        "raw_killerTeamId": int(raw_kt),
        "raw_killerId": int(raw_kid) if not kid_reason else C.SENTINEL_MISSING,
        "raw_building_owner": C.SENTINEL_MISSING,
    }

    if raw_kt == C.TEAM_NEUTRAL and not kt_reason:
        out["acq_status"] = C.ACQ_NEUTRAL
        out["team_source"] = "explicit_neutral_300"
        return out

    if raw_kt in (C.TEAM_BLUE, C.TEAM_RED) and not kt_reason:
        if roster in (C.TEAM_BLUE, C.TEAM_RED) and roster != raw_kt:
            out["acq_status"] = C.ACQ_CONTRADICTORY
            out["team_source"] = "contradictory_killerTeamId_vs_roster"
            out["team"] = C.TEAM_UNKNOWN
            return out
        out["team"] = int(raw_kt)
        out["acq_status"] = C.ACQ_CONFIRMED
        out["team_source"] = "killerTeamId"
        return out

    # Missing / invalid / nonstandard (not explicit 300): roster fallback only.
    if roster in (C.TEAM_BLUE, C.TEAM_RED):
        out["team"] = int(roster)
        out["acq_status"] = C.ACQ_CONFIRMED
        out["team_source"] = "killerId_roster"
        return out

    if kt_reason:
        out["team_source"] = kt_reason
    elif raw_kt not in (C.TEAM_BLUE, C.TEAM_RED, C.TEAM_UNKNOWN, 0, C.TEAM_NEUTRAL):
        out["team_source"] = f"nonstandard_killerTeamId:{raw_kt}"
    else:
        out["team_source"] = "unknown_monster_team"
    out["acq_status"] = C.ACQ_UNKNOWN
    return out


def acquiring_team_monster(event: Mapping[str, Any], blue_pids: Sequence[int]) -> tuple[int, str]:
    """Back-compat wrapper: (team, team_source). Confirmed teams only when acq confirmed."""
    r = resolve_monster_acquisition(event, blue_pids)
    return int(r["team"]), str(r["team_source"])


def resolve_building_acquisition(event: Mapping[str, Any]) -> dict[str, Any]:
    """BUILDING_KILL.teamId is destroyed owner; acquirer is opponent when owner is 100/200."""
    lost, reason = parse_team_id(event.get("teamId"))
    out = {
        "team": C.TEAM_UNKNOWN,
        "acq_status": C.ACQ_UNKNOWN,
        "team_source": "missing_building_owner",
        "raw_killerTeamId": C.SENTINEL_MISSING,
        "raw_killerId": C.SENTINEL_MISSING,
        "raw_building_owner": int(lost) if not reason else C.SENTINEL_MISSING,
    }
    if reason:
        out["team_source"] = reason
        return out
    if lost in (C.TEAM_BLUE, C.TEAM_RED):
        out["team"] = 300 - int(lost)
        out["acq_status"] = C.ACQ_CONFIRMED
        out["team_source"] = "owner_inverted"
        out["raw_building_owner"] = int(lost)
        return out
    if lost in (C.TEAM_UNKNOWN, 0):
        out["team_source"] = "missing_building_owner"
        return out
    out["team_source"] = f"nonstandard_building_owner:{lost}"
    out["raw_building_owner"] = int(lost)
    return out


def acquiring_team_building(event: Mapping[str, Any]) -> tuple[int, str]:
    r = resolve_building_acquisition(event)
    return int(r["team"]), str(r["team_source"])


def classify_monster(event: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return (category, monsterType_str, monsterSubType_str).

    Missing/unknown Dragon subtype is NOT asserted elemental.
    """
    raw_m = event.get("monsterType")
    raw_s = event.get("monsterSubType")
    m = "" if raw_m is None else str(raw_m)
    s = "" if raw_s is None else str(raw_s)
    if m == "DRAGON":
        su = s.upper()
        if su == "ELDER_DRAGON":
            return "dragon_elder", m, s
        if su in C.KNOWN_ELEMENTAL_DRAGON_SUBTYPES:
            return "dragon_elemental", m, s
        return "dragon_unknown_subtype", m, s
    mapping = {
        "BARON_NASHOR": "baron",
        "RIFTHERALD": "herald",
        "HORDE": "horde",
        "ATAKHAN": "atakhan",
    }
    if m in mapping:
        return mapping[m], m, s
    return "monster_other", m, s


def classify_building(event: Mapping[str, Any]) -> tuple[str, str, str, bool]:
    """Return (category, buildingType, towerType, is_terminal_structure)."""
    raw_b = event.get("buildingType")
    raw_t = event.get("towerType")
    b = "" if raw_b is None else str(raw_b)
    t = "" if raw_t is None else str(raw_t)
    is_terminal = b == "NEXUS_BUILDING"
    if is_terminal:
        return "building_terminal", b, t, True
    if b == "TOWER_BUILDING":
        return "tower", b, t, False
    if b == "INHIBITOR_BUILDING":
        return "inhibitor", b, t, False
    if b == "":
        return "building_other", b, t, False
    return "building_other", b, t, False


def event_categories(leaf: str, is_monster: bool, is_terminal_building: bool) -> list[str]:
    """All reporting categories an event contributes to (including aggregates)."""
    cats = ["any_objective", leaf]
    if is_monster:
        cats.append("any_epic_monster")
    else:
        cats.append("any_building")
        if not is_terminal_building:
            cats.append("any_nonterminal_building")
    seen: set[str] = set()
    out: list[str] = []
    for c in cats:
        if c not in seen and c in C.CATEGORY_TO_CODE:
            seen.add(c)
            out.append(c)
    return out


def parse_objective_events(
    events: Sequence[Mapping[str, Any] | Any],
    blue_pids: Sequence[int],
    last_frame_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse top-level event objects into objective records + schema census.

    Every source objective event is retained with raw team evidence and acq_status.
    Primary acquisition timing uses confirmed 100/200 rows only (see query layer).
    """
    rows: list[dict[str, Any]] = []
    census: dict[str, Any] = {
        "n_events_total": 0,
        "n_elite_monster_kill": 0,
        "n_building_kill": 0,
        "n_malformed": 0,
        "n_invalid_objective_timestamp": 0,
        "observation_support_unknown": False,
        "malformed_reasons": {},
        "monster_type_counts": {},
        "monster_subtype_counts": {},
        "building_type_counts": {},
        "tower_type_counts": {},
        "raw_type_counts": {},
        "n_after_last_frame": 0,
        "n_acq_confirmed": 0,
        "n_acq_neutral": 0,
        "n_acq_unknown": 0,
        "n_acq_contradictory": 0,
        "team_source_counts": {},
    }

    def _bump(d: dict[str, int], key: str) -> None:
        d[key] = int(d.get(key, 0)) + 1

    for ord_i, e in enumerate(events):
        census["n_events_total"] += 1
        if not isinstance(e, Mapping):
            census["n_malformed"] += 1
            _bump(census["malformed_reasons"], "non_mapping_event")
            continue
        typ = e.get("type")
        typ_s = "" if typ is None else str(typ)
        _bump(census["raw_type_counts"], typ_s if typ_s else "<missing_type>")
        if typ_s not in ("ELITE_MONSTER_KILL", "BUILDING_KILL"):
            continue

        if "timestamp" not in e:
            census["n_malformed"] += 1
            census["n_invalid_objective_timestamp"] += 1
            census["observation_support_unknown"] = True
            _bump(census["malformed_reasons"], "missing_timestamp")
            # Do not invent NO_ACQ from a dropped objective; support is unknown.
            continue
        ts, ts_reason = parse_timestamp_ms(e["timestamp"])
        if ts_reason or ts is None:
            census["n_malformed"] += 1
            census["n_invalid_objective_timestamp"] += 1
            census["observation_support_unknown"] = True
            _bump(census["malformed_reasons"], ts_reason or "invalid_timestamp")
            continue

        after_lf = bool(last_frame_ms >= 0 and ts > int(last_frame_ms))
        if after_lf:
            census["n_after_last_frame"] += 1

        if typ_s == "ELITE_MONSTER_KILL":
            census["n_elite_monster_kill"] += 1
            leaf, mtype, msub = classify_monster(e)
            acq = resolve_monster_acquisition(e, blue_pids)
            _bump(census["monster_type_counts"], mtype if mtype else "<missing>")
            _bump(census["monster_subtype_counts"], msub if msub else "<missing>")
            cats = event_categories(leaf, is_monster=True, is_terminal_building=False)
            row = {
                "event_ord": int(ord_i),
                "t_ms": int(ts),
                "raw_type": typ_s,
                "leaf_category": leaf,
                "categories": cats,
                "category_mask": _mask_from_cats(cats),
                "team": int(acq["team"]),
                "team_source": str(acq["team_source"]),
                "acq_status": int(acq["acq_status"]),
                "raw_killerTeamId": int(acq["raw_killerTeamId"]),
                "raw_killerId": int(acq["raw_killerId"]),
                "raw_building_owner": int(acq["raw_building_owner"]),
                "monster_type": mtype,
                "monster_subtype": msub,
                "building_type": "",
                "tower_type": "",
                "is_terminal_building": False,
                "after_last_frame": after_lf,
                "same_ms_at_L_placeholder": False,
            }
        else:
            census["n_building_kill"] += 1
            leaf, btype, ttype, is_term = classify_building(e)
            acq = resolve_building_acquisition(e)
            _bump(census["building_type_counts"], btype if btype else "<missing>")
            _bump(census["tower_type_counts"], ttype if ttype else "<missing>")
            cats = event_categories(leaf, is_monster=False, is_terminal_building=is_term)
            row = {
                "event_ord": int(ord_i),
                "t_ms": int(ts),
                "raw_type": typ_s,
                "leaf_category": leaf,
                "categories": cats,
                "category_mask": _mask_from_cats(cats),
                "team": int(acq["team"]),
                "team_source": str(acq["team_source"]),
                "acq_status": int(acq["acq_status"]),
                "raw_killerTeamId": int(acq["raw_killerTeamId"]),
                "raw_killerId": int(acq["raw_killerId"]),
                "raw_building_owner": int(acq["raw_building_owner"]),
                "monster_type": "",
                "monster_subtype": "",
                "building_type": btype,
                "tower_type": ttype,
                "is_terminal_building": bool(is_term),
                "after_last_frame": after_lf,
                "same_ms_at_L_placeholder": False,
            }

        st = int(row["acq_status"])
        if st == C.ACQ_CONFIRMED:
            census["n_acq_confirmed"] += 1
        elif st == C.ACQ_NEUTRAL:
            census["n_acq_neutral"] += 1
        elif st == C.ACQ_CONTRADICTORY:
            census["n_acq_contradictory"] += 1
        else:
            census["n_acq_unknown"] += 1
        _bump(census["team_source_counts"], str(row["team_source"]))
        rows.append(row)

    rows.sort(key=lambda r: (int(r["t_ms"]), int(r["event_ord"])))
    return rows, census


def _mask_from_cats(cats: Sequence[str]) -> int:
    mask = 0
    for c in cats:
        code = C.CATEGORY_TO_CODE.get(c)
        if code is not None:
            mask |= 1 << int(code)
    return int(mask)


def category_in_mask(mask: int, category: str) -> bool:
    code = C.CATEGORY_TO_CODE[category]
    return bool(int(mask) & (1 << int(code)))


def rows_to_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert parsed rows to numpy-friendly parallel arrays (sorted by t, ord)."""
    import numpy as np

    n = len(rows)
    t = np.zeros(n, dtype=np.int64)
    ord_ = np.zeros(n, dtype=np.int64)
    mask = np.zeros(n, dtype=np.int64)
    team = np.zeros(n, dtype=np.int16)
    acq = np.zeros(n, dtype=np.int8)
    after = np.zeros(n, dtype=np.int8)
    raw_kt = np.full(n, C.SENTINEL_MISSING, dtype=np.int32)
    raw_kid = np.full(n, C.SENTINEL_MISSING, dtype=np.int32)
    raw_owner = np.full(n, C.SENTINEL_MISSING, dtype=np.int32)
    leaf_code = np.full(n, C.SENTINEL_MISSING, dtype=np.int16)
    for i, r in enumerate(rows):
        t[i] = int(r["t_ms"])
        ord_[i] = int(r["event_ord"])
        mask[i] = int(r["category_mask"])
        team[i] = int(r["team"])
        acq[i] = int(r["acq_status"])
        after[i] = 1 if r["after_last_frame"] else 0
        raw_kt[i] = int(r["raw_killerTeamId"])
        raw_kid[i] = int(r["raw_killerId"])
        raw_owner[i] = int(r["raw_building_owner"])
        leaf = str(r["leaf_category"])
        leaf_code[i] = int(C.CATEGORY_TO_CODE.get(leaf, C.SENTINEL_MISSING))
    return {
        "t_ms": t,
        "event_ord": ord_,
        "category_mask": mask,
        "team": team,
        "acq_status": acq,
        "after_last_frame": after,
        "raw_killerTeamId": raw_kt,
        "raw_killerId": raw_kid,
        "raw_building_owner": raw_owner,
        "leaf_code": leaf_code,
    }
