"""Filesystem / raw-cache source audit for the frozen objective_timing baseline.

Adapted from reports/.../objective_timing_review_20261002/verify_source_and_queries.py.
Independent of production parse/query helpers; uses strict integer timestamps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .. import config as C
from ..parse import parse_timestamp_ms


def _leaf_and_mask_building(kind: str) -> tuple[int, int]:
    if kind == "TOWER_BUILDING":
        leaf = 12
    elif kind == "INHIBITOR_BUILDING":
        leaf = 13
    elif kind == "NEXUS_BUILDING":
        leaf = 15
    else:
        leaf = 14
    mask = (1 << 0) | (1 << 2) | (1 << leaf)
    if leaf != 15:
        mask |= 1 << 3
    return leaf, mask


def _leaf_and_mask_monster(kind: str, sub: str) -> tuple[int, int]:
    if kind == "DRAGON":
        if sub == "ELDER_DRAGON":
            leaf = 5
        elif sub in C.KNOWN_ELEMENTAL_DRAGON_SUBTYPES:
            leaf = 4
        else:
            leaf = 6
    else:
        leaf = {"BARON_NASHOR": 7, "RIFTHERALD": 8, "HORDE": 9, "ATAKHAN": 10}.get(kind, 11)
    mask = (1 << 0) | (1 << 1) | (1 << leaf)
    return leaf, mask


def classify_raw_objective(
    typ: str, event: Mapping[str, Any]
) -> tuple[int, int, int]:
    """Return (credited_team, acq_status, category_mask) using independent review rules.

    Neutral 300 has no credited saved team (team 0), excludes acquisition and uncertainty.
    Unknown/nonstandard evidence → acq_status unknown (2), never credited as confirmed.
    """
    if typ == "BUILDING_KILL":
        owner = event.get("teamId", 0)
        if isinstance(owner, bool) or not isinstance(owner, (int, float)) or (
            isinstance(owner, float) and owner != int(owner)
        ):
            owner_i = 0
        else:
            owner_i = int(owner)
        team = 300 - owner_i if owner_i in (100, 200) else 0
        kind = str(event.get("buildingType", ""))
        _, mask = _leaf_and_mask_building(kind)
        status = C.ACQ_CONFIRMED if team in (100, 200) else C.ACQ_UNKNOWN
        return team, status, mask

    team_raw = event.get("killerTeamId", 0)
    if isinstance(team_raw, bool) or not isinstance(team_raw, (int, float)) or (
        isinstance(team_raw, float) and team_raw != int(team_raw)
    ):
        team_i = 0
    else:
        team_i = int(team_raw)
    kid = event.get("killerId", 0)
    if isinstance(kid, bool) or not isinstance(kid, (int, float)) or (
        isinstance(kid, float) and kid != int(kid)
    ):
        kid_i = -1
    else:
        kid_i = int(kid)
    kind = str(event.get("monsterType", ""))
    sub = str(event.get("monsterSubType", ""))
    _, mask = _leaf_and_mask_monster(kind, sub)
    if team_i in (100, 200):
        status = C.ACQ_CONFIRMED
        team = team_i
    elif team_i == 300 and kid_i == 0:
        status = C.ACQ_NEUTRAL
        team = 0  # credited team field; raw 300 not stored as credited team
    else:
        status = C.ACQ_UNKNOWN
        team = 0
    return team, status, mask


def audit_raw_sources(
    *,
    match_ids: Sequence[str],
    terminals: Sequence[Mapping[str, Any]],
    kill_match_local: np.ndarray,
    kill_t_ms: np.ndarray,
    events: Mapping[str, np.ndarray],
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Reconcile flat raw events.json against kill ledger, terminals, and objective_events."""
    t0 = time.perf_counter()
    mids = list(match_ids)
    km = np.asarray(kill_match_local, dtype=np.int64)
    kt = np.asarray(kill_t_ms, dtype=np.int64)
    order = np.lexsort((kt, km))
    km = km[order]
    kt = kt[order]
    kb = np.searchsorted(km, np.arange(len(mids) + 1))
    eb = np.searchsorted(events["match_local"], np.arange(len(mids) + 1))

    failures: list[Any] = []
    counts = {
        "n_flat_lists": 0,
        "n_events": 0,
        "n_kills": 0,
        "n_game_end": 0,
        "n_objectives": 0,
        "n_non_object": 0,
        "n_bad_timestamp": 0,
    }

    for i, mid in enumerate(mids):
        path = cache_dir / f"{mid}.events.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            failures.append([i, f"cache_read:{type(exc).__name__}"])
            continue
        if not isinstance(raw, list):
            failures.append([i, "not_flat_list"])
            continue
        counts["n_flat_lists"] += 1
        counts["n_events"] += len(raw)
        kills: list[int] = []
        ends: list[int] = []
        objects: list[tuple[int, int, str, Mapping[str, Any]]] = []
        for ordinal, e in enumerate(raw):
            if not isinstance(e, dict):
                counts["n_non_object"] += 1
                failures.append([i, "non_object"])
                continue
            typ = e.get("type")
            if typ == "CHAMPION_KILL":
                ts, reason = parse_timestamp_ms(e.get("timestamp"))
                if reason or ts is None:
                    counts["n_bad_timestamp"] += 1
                    failures.append([i, "kill_timestamp", reason or "invalid"])
                    continue
                kills.append(int(ts))
            elif typ == "GAME_END":
                ts, reason = parse_timestamp_ms(e.get("timestamp"))
                if reason or ts is None:
                    counts["n_bad_timestamp"] += 1
                    failures.append([i, "game_end_timestamp", reason or "invalid"])
                    continue
                ends.append(int(ts))
            elif typ in ("ELITE_MONSTER_KILL", "BUILDING_KILL"):
                ts, reason = parse_timestamp_ms(e.get("timestamp"))
                if reason or ts is None:
                    counts["n_bad_timestamp"] += 1
                    failures.append([i, "objective_timestamp", reason or "invalid"])
                    continue
                objects.append((ordinal, int(ts), str(typ), e))
        counts["n_kills"] += len(kills)
        counts["n_game_end"] += len(ends)
        counts["n_objectives"] += len(objects)

        if not np.array_equal(np.sort(kills), kt[kb[i] : kb[i + 1]]):
            failures.append([i, "kill_times"])
        term = terminals[i]
        if len(kills) != int(term["n_kills"]):
            failures.append([i, "terminal_kill_count"])
        if ends != list(term["game_end_all"]) or len(ends) != 1 or ends[0] != int(term["game_end_ms"]):
            failures.append([i, "game_end"])

        rows = range(int(eb[i]), int(eb[i + 1]))
        if len(objects) != len(rows):
            failures.append([i, "objective_count"])
            continue
        by_ord = {int(events["event_ord"][j]): j for j in rows}
        for ordinal, ts, typ, e in objects:
            j = by_ord.get(ordinal)
            if j is None or int(events["t_ms"][j]) != ts:
                failures.append([i, "objective_identity"])
                continue
            team, status, mask = classify_raw_objective(typ, e)
            if (
                int(events["team"][j]) != team
                or int(events["acq_status"][j]) != status
                or int(events["category_mask"][j]) != mask
            ):
                failures.append([i, "team_or_category_or_evidence"])
        if progress and (i + 1) % 10000 == 0:
            progress(f"source_matches_checked {i + 1}")

    all_ok = (
        not failures
        and counts["n_flat_lists"] == C.EXPECTED_TRAIN_MATCHES
        and counts["n_kills"] == C.EXPECTED_TRAIN_KILLS
        and counts["n_bad_timestamp"] == 0
    )
    return {
        "n_matches": len(mids),
        "counts": counts,
        "n_failures": len(failures),
        "failures_head": failures[:30],
        "all_ok": all_ok,
        "wall_s": float(time.perf_counter() - t0),
    }
