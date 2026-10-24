"""Pure all-role engagement population and endpoint construction.

This module consumes one already validated :class:`ObservableMatch`.  It does
not read source files, write artifacts, fit a model, or construct V/q inputs.
The event normalizer deliberately keeps malformed ``CHAMPION_KILL`` records in
the audit surface while only records with a valid timestamp and position enter
the space-time component graph.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import numpy as np


LINK_TIME_MS = 14_000
LINK_RADIUS_UNITS = 2_000.0
MIN_KILLS = 1
PRIMARY_RADIUS_UNITS = 1_800.0
SENSITIVITY_RADIUS_UNITS = 2_000.0
FRAME_LAG_MIN_MS = 10_000
FRAME_LAG_MAX_MS = 20_000
ANCHOR_OFFSET_MS = 10_000
PRIMARY_PRE_MIN_MS = 120_000
HORIZONS_S = (60, 90, 120)

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

# The integers are part of the compact audit surface.  Text remains available
# in ``all_reasons`` so downstream columnizers do not need to reverse a bitmask.
REASON_CODES = {
    "normalization_issue": 1,
    "component_source_issue": 2,
    "pre_frame_support_unknown": 4,
    "firstkill_frame_lag_outside_inclusive_10000_20000": 8,
    "global_isolation_failed_closed_pre_lastkill": 16,
    "global_isolation_support_unknown": 32,
    "firstkill_credit_block_association_unverified": 64,
    "no_confirmed_opposing_player_credit": 128,
    "environment_killer_zero_not_player": 256,
    "minimum_team_credit_not_met": 512,
    "next_anchor_overlap_component": 1024,
    "h60_endpoint_or_frame_support_invalid": 2048,
    "h90_endpoint_or_frame_support_invalid": 4096,
    "h120_endpoint_or_frame_support_invalid": 8192,
    "critical_source_invalid": 16384,
    "critical_source_unknown": 32768,
    "early_pre_cohort_preserved_separately": 65536,
    "unknown_starting_scale_retained_for_overall_candidate": 131072,
    "no_valid_component_members": 262144,
}


def _strict_int(value: Any, *, nonnegative: bool = False) -> int | None:
    """Parse an integer without accepting numeric strings or integral floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    if nonnegative and result < 0:
        return None
    return result


def _strict_number(value: Any) -> float | None:
    """Parse a finite raw coordinate while rejecting strings and booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _event_items(match: Any) -> Iterable[tuple[int, Mapping[str, Any], str]]:
    """Yield ``(source ordinal, raw payload, type)`` without reordering source.

    ``ObservableMatch.typed_events`` is preferred because ``match.events`` is a
    timestamp-sorted projection.  The payload in a typed event is the original
    event mapping and ``source_index`` is its original list ordinal.
    """

    typed = getattr(match, "typed_events", None)
    if typed is not None:
        for fallback, event in enumerate(typed):
            event_type_value = getattr(event, "event_type", None)
            if event_type_value is None:
                probe = getattr(event, "payload", None)
                event_type_value = probe.get("type", "") if isinstance(probe, Mapping) else ""
            event_type = str(event_type_value)
            if event_type.strip().upper() != "CHAMPION_KILL":
                continue
            payload = getattr(event, "payload", None)
            if isinstance(payload, Mapping):
                # ``ObservableMatch`` keeps the validated timestamp on the
                # typed record and strips it from the immutable payload.
                # Copy only when needed so normalization never mutates source
                # event mappings while retaining the original timestamp.
                if "timestamp" not in payload and "timestamp_ms" not in payload:
                    timestamp = getattr(event, "timestamp_ms", None)
                    if timestamp is not None:
                        payload = dict(payload)
                        payload["timestamp"] = timestamp
                ordinal = _strict_int(getattr(event, "source_index", fallback))
                yield fallback if ordinal is None else ordinal, payload, event_type
        return

    events = getattr(match, "events", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return
    for ordinal, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            kind = str(event.get("event_type", payload.get("type", "")))
            if kind.strip().upper() != "CHAMPION_KILL":
                continue
            if "timestamp" not in payload and "timestamp_ms" not in payload:
                timestamp = event.get("timestamp_ms", event.get("timestamp"))
                if timestamp is not None:
                    payload = dict(payload)
                    payload["timestamp"] = timestamp
            yield ordinal, payload, kind
        else:
            kind = str(event.get("event_type", event.get("type", "")))
            if kind.strip().upper() != "CHAMPION_KILL":
                continue
            yield ordinal, event, kind


def _position(payload: Mapping[str, Any]) -> tuple[float | None, float | None, bool, list[str]]:
    issues: list[str] = []
    position_present = "position" in payload or ("x" in payload and "y" in payload)
    source = payload.get("position") if "position" in payload else payload
    if not isinstance(source, Mapping):
        return None, None, position_present, ["position_invalid"]
    x = _strict_number(source.get("x"))
    y = _strict_number(source.get("y"))
    if x is None or y is None:
        issues.append("position_missing" if not position_present else "position_invalid")
    return x, y, position_present, issues


def normalize_champion_kills(match: Any) -> list[dict[str, Any]]:
    """Normalize every recorded ``CHAMPION_KILL`` in stable ``(t, ordinal)`` order.

    The returned records are audit-friendly dictionaries.  ``assist_field_present``
    distinguishes an explicitly empty assist list from an absent raw field;
    absence is retained as ``assist_status='unknown'`` rather than converted to
    an empty list with a false certainty claim.
    """

    records: list[dict[str, Any]] = []
    for fallback, payload, event_type in _event_items(match):
        if str(event_type).strip().upper() != "CHAMPION_KILL":
            continue
        ordinal = _strict_int(fallback, nonnegative=True)
        if ordinal is None:
            ordinal = len(records)
        issues: list[str] = []

        raw_ts = payload.get("timestamp") if "timestamp" in payload else payload.get("timestamp_ms")
        timestamp_ms = _strict_int(raw_ts, nonnegative=True)
        if timestamp_ms is None:
            issues.append("timestamp_invalid")

        raw_killer = payload.get("killerId")
        killer_id = _strict_int(raw_killer)
        if killer_id is None or not 0 <= killer_id <= 10:
            killer_id = None
            issues.append("killer_invalid")

        raw_victim = payload.get("victimId")
        victim_id = _strict_int(raw_victim)
        if victim_id is None or not 1 <= victim_id <= 10:
            victim_id = None
            issues.append("victim_invalid")

        assist_field_present = "assistingParticipantIds" in payload
        assist_status = "valid"
        assist_ids: list[int] = []
        if not assist_field_present:
            assist_status = UNKNOWN
            issues.append("assist_field_missing")
        else:
            raw_assists = payload.get("assistingParticipantIds")
            if not isinstance(raw_assists, Sequence) or isinstance(raw_assists, (str, bytes, bytearray)):
                assist_status = UNKNOWN
                issues.append("assist_invalid_container")
            else:
                seen: set[int] = set()
                for value in raw_assists:
                    participant = _strict_int(value)
                    if participant is None or not 1 <= participant <= 10:
                        assist_status = UNKNOWN
                        issues.append("assist_invalid_id")
                        continue
                    if participant in seen:
                        issues.append("assist_duplicate_id")
                        continue
                    seen.add(participant)
                    assist_ids.append(participant)

        x, y, position_present, position_issues = _position(payload)
        issues.extend(position_issues)
        if timestamp_ms is None or x is None or y is None:
            valid_st = False
        else:
            valid_st = True

        records.append({
            "source_ordinal": int(ordinal),
            "source_index": int(ordinal),
            "timestamp_ms": timestamp_ms,
            "killer_id": killer_id,
            "victim_id": victim_id,
            "assisting_participant_ids": tuple(assist_ids),
            "assist_field_present": bool(assist_field_present),
            "assist_status": assist_status,
            "position_present": bool(position_present),
            "x": x,
            "y": y,
            "timestamp_valid": timestamp_ms is not None,
            "killer_valid": killer_id is not None,
            "victim_valid": victim_id is not None,
            "xy_valid": x is not None and y is not None,
            "valid_st": bool(valid_st),
            "environmental_killer": killer_id == 0,
            "raw_presence_unknown": True,
            "issues": tuple(dict.fromkeys(issues)),
        })

    records.sort(key=lambda row: (
        0 if row["timestamp_ms"] is not None else 1,
        row["timestamp_ms"] if row["timestamp_ms"] is not None else 0,
        int(row["source_ordinal"]),
    ))
    return records


# Short alias used by some producers and tests.
normalize_kills = normalize_champion_kills


def _meta(match: Any) -> Mapping[str, Any]:
    value = getattr(match, "meta", None)
    if isinstance(value, Mapping):
        return value
    value = getattr(match, "_meta", None)
    return value if isinstance(value, Mapping) else {}


def _team_ids(match: Any) -> tuple[np.ndarray, bool]:
    values = np.asarray(getattr(match, "team_ids", ()), dtype=np.int64)
    valid = values.shape == (10,) and set(values.tolist()) == {100, 200}
    if valid:
        valid = int(np.sum(values == 100)) == 5 and int(np.sum(values == 200)) == 5
    return values, bool(valid)


def _team_for_pid(team_ids: np.ndarray, pid: int | None) -> int | None:
    if pid is None or not 1 <= int(pid) <= 10 or team_ids.shape != (10,):
        return None
    team = int(team_ids[int(pid) - 1])
    return team if team in (100, 200) else None


def _bits(participants: Iterable[int]) -> int:
    bits = 0
    for pid in participants:
        if 1 <= int(pid) <= 10:
            bits |= 1 << (int(pid) - 1)
    return int(bits)


def _ordinal_ranges(ordinals: Iterable[int]) -> tuple[tuple[int, int], ...]:
    values = sorted({int(value) for value in ordinals})
    if not values:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return tuple(ranges)


def _stable_component_key(members: Sequence[Mapping[str, Any]]) -> str:
    material = [
        {
            "source_ordinal": int(row["source_ordinal"]),
            "timestamp_ms": row["timestamp_ms"],
            "killer_id": row["killer_id"],
            "victim_id": row["victim_id"],
            "assisting_participant_ids": list(row["assisting_participant_ids"]),
            "x": row["x"],
            "y": row["y"],
        }
        for row in members
    ]
    encoded = json.dumps(material, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _st_components_windowed():
    """Load the exact adopted ST implementation without importing legacy config."""

    root = Path(__file__).resolve().parents[2]
    path = root / "worktrees" / "engagement-state-value" / "studies" / "st_dbscan_20261001" / "kernel.py"
    if not path.is_file():
        raise RuntimeError(f"adopted ST kernel is unavailable: {path}")
    module_name = "_observable_state_20261002_st_kernel"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing._components_windowed
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load adopted ST kernel")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module._components_windowed


def _frame_index(minute_ts: np.ndarray, query_ms: int | None) -> int:
    if query_ms is None or minute_ts.ndim != 1 or minute_ts.size == 0:
        return -1
    return int(np.searchsorted(minute_ts, int(query_ms), side="right") - 1)


def _proxy_at_frame(
    match: Any,
    frame_index: int,
    locations: Sequence[tuple[float, float]],
    c0_bits: int,
    radius: float,
) -> tuple[bool, int, int, int, int]:
    """Return cache-only proxy support, lower/upper counts and bitsets.

    Credit is a lower bound.  A finite nonzero cached position within the
    radius contributes to the geometric upper bound; zero/nonfinite positions
    remain ambiguous and therefore also enter the upper bound.
    """

    xy = np.asarray(getattr(match, "xy", ()))
    if frame_index < 0 or xy.ndim != 3 or frame_index >= xy.shape[0] or xy.shape[1:] != (10, 2) or not locations:
        return False, -1, -1, int(c0_bits), (1 << 10) - 1
    frame = np.asarray(xy[frame_index], dtype=np.float64)
    near_bits = 0
    unknown_bits = 0
    radius2 = float(radius) * float(radius)
    for index in range(10):
        x, y = float(frame[index, 0]), float(frame[index, 1])
        if not (math.isfinite(x) and math.isfinite(y)) or x == 0.0 or y == 0.0:
            unknown_bits |= 1 << index
            continue
        if any((x - lx) * (x - lx) + (y - ly) * (y - ly) <= radius2 for lx, ly in locations):
            near_bits |= 1 << index
    lower_bits = int(c0_bits)
    upper_bits = int(c0_bits) | int(near_bits) | int(unknown_bits)
    return True, int(lower_bits.bit_count()), int(upper_bits.bit_count()), lower_bits, upper_bits


def _endpoint(
    last_kill_ms: int,
    horizon_s: int,
    next_kill_ms: int | None,
    next_anchor_ms: int | None,
    end_ms: int | None,
) -> tuple[int | None, tuple[str, ...]]:
    candidates: list[tuple[str, int]] = [("horizon", int(last_kill_ms) + int(horizon_s) * 1000)]
    if next_kill_ms is not None:
        candidates.append(("next_kill_minus_1", int(next_kill_ms) - 1))
    if next_anchor_ms is not None:
        candidates.append(("next_anchor_minus_1", int(next_anchor_ms) - 1))
    if end_ms is not None:
        candidates.append(("game_end_minus_1", int(end_ms) - 1))
    if not candidates:
        return None, ()
    endpoint = min(value for _, value in candidates)
    reasons = tuple(sorted(name for name, value in candidates if value == endpoint))
    return int(endpoint), reasons


def _reason_code(reason: str) -> int:
    if reason in REASON_CODES:
        return REASON_CODES[reason]
    # Stable code for a source-normalization detail that is not a population
    # gate.  The reserved high range keeps it distinct from fixed policy bits.
    digest = hashlib.sha256(reason.encode("utf-8")).digest()
    return 1_000_000 + int.from_bytes(digest[:4], "big")


def build_population(
    match: Any,
    *,
    kills: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build every valid ST component and its population/endpoint audit row.

    Rows are returned sorted by first kill timestamp and source ordinal.  All
    rows, including early and unsupported candidates, are retained for audit;
    downstream code decides which supported primary h90 rows enter model prep.
    """

    # ``kills`` is an optional normalized-kill handoff for callers that already
    # performed source normalization.  It is read-only here and is never
    # cached on ``match``; the default path preserves the one-argument API.
    if kills is None:
        kills = normalize_champion_kills(match)
    if not kills:
        return []

    valid_indices = [index for index, row in enumerate(kills) if row["valid_st"]]
    if not valid_indices:
        return []
    t = np.asarray([int(kills[index]["timestamp_ms"]) for index in valid_indices], dtype=np.int64)
    x = np.asarray([float(kills[index]["x"]) for index in valid_indices], dtype=np.float64)
    y = np.asarray([float(kills[index]["y"]) for index in valid_indices], dtype=np.float64)
    labels = np.asarray(_st_components_windowed()(t, x, y, LINK_TIME_MS, LINK_RADIUS_UNITS), dtype=np.int64)

    groups: dict[int, list[int]] = {}
    for local_index, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(valid_indices[local_index])
    component_members = sorted(
        [
            sorted(
                indices,
                key=lambda index: (
                    int(kills[index]["timestamp_ms"]),
                    int(kills[index]["source_ordinal"]),
                ),
            )
            for indices in groups.values()
        ],
        key=lambda indices: (
            int(kills[indices[0]]["timestamp_ms"]),
            int(kills[indices[0]]["source_ordinal"]),
        ),
    )

    minute_ts = np.asarray(getattr(match, "minute_ts", ()))
    if minute_ts.ndim != 1 or minute_ts.dtype.kind not in "iu" or minute_ts.size == 0:
        minute_ts = np.asarray([], dtype=np.int64)
    else:
        minute_ts = minute_ts.astype(np.int64, copy=False)
    team_ids, team_map_ok = _team_ids(match)
    end_value = _strict_int(getattr(match, "end_ms", getattr(match, "end_ms_value", None)), nonnegative=True)
    winner = _strict_int(getattr(match, "winning_team", getattr(match, "winning_team_value", None)))
    terminal_support = bool(end_value is not None and winner in (100, 200))
    meta = _meta(match)
    canonical_match_key = str(
        _mapping_value(meta, "canonical_match_key", "anon_match_key", "match_key", "match_id", "original_match_id") or ""
    )
    patch = str(getattr(match, "patch", meta.get("patch", "")) or "")

    valid_kills = [kills[index] for index in valid_indices]
    valid_kills.sort(key=lambda row: (int(row["timestamp_ms"]), int(row["source_ordinal"])))
    valid_times = np.asarray([int(row["timestamp_ms"]) for row in valid_kills], dtype=np.int64)
    # Missing optional assist fields are retained as provenance uncertainty but
    # do not invalidate event chronology, components, or closed-interval
    # isolation.  Boundary-critical issues are limited to fields needed to
    # place a kill in the ST graph or identify its two participants.
    critical_issue_names = {
        "timestamp_invalid",
        "killer_invalid",
        "victim_invalid",
        "position_missing",
        "position_invalid",
    }
    source_invalid_ordinals = tuple(
        int(row["source_ordinal"])
        for row in kills
        if any(issue in critical_issue_names for issue in row["issues"])
    )
    optional_issue_ordinals = tuple(
        int(row["source_ordinal"])
        for row in kills
        if row["issues"] and not any(issue in critical_issue_names for issue in row["issues"])
    )
    source_issues = tuple(sorted({issue for row in kills for issue in row["issues"]}))

    component_firsts: list[int] = []
    for indices in component_members:
        component_firsts.append(int(kills[indices[0]]["timestamp_ms"]))
    component_first_array = np.asarray(component_firsts, dtype=np.int64)
    component_anchor_array = component_first_array - ANCHOR_OFFSET_MS

    rows: list[dict[str, Any]] = []
    for component_id, indices in enumerate(component_members):
        members = [kills[index] for index in indices]
        f = int(members[0]["timestamp_ms"])
        last = int(members[-1]["timestamp_ms"])
        member_ordinals = tuple(int(row["source_ordinal"]) for row in members)
        member_ranges = _ordinal_ranges(member_ordinals)
        component_key = _stable_component_key(members)
        pre = f - ANCHOR_OFFSET_MS
        frame_index = _frame_index(minute_ts, pre)
        if (
            frame_index < 0
            or minute_ts.size == 0
            or pre < int(minute_ts[0])
        ):
            frame_index = -1
        frame_ms = int(minute_ts[frame_index]) if frame_index >= 0 else None
        lag = f - frame_ms if frame_ms is not None else None
        frame_lag_candidate = bool(lag is not None and FRAME_LAG_MIN_MS <= lag <= FRAME_LAG_MAX_MS)
        pre_support = bool(
            frame_index >= 0
            and frame_ms is not None
            and pre >= 0
            and (minute_ts.size == 0 or pre <= int(minute_ts[-1]))
        )

        first_block = [row for row in members if int(row["timestamp_ms"]) == f]
        credited: set[int] = set()
        relation_kind = "none"
        direct = False
        direct_unknown = False
        env_count = 0
        for kill in first_block:
            killer = kill["killer_id"]
            victim = kill["victim_id"]
            if killer == 0:
                env_count += 1
            elif killer is not None:
                credited.add(int(killer))
            if victim is not None:
                credited.add(int(victim))
            credited.update(int(pid) for pid in kill["assisting_participant_ids"])

            victim_team = _team_for_pid(team_ids, victim)
            killer_team = _team_for_pid(team_ids, killer) if killer not in (None, 0) else None
            if victim_team is None or killer is None or victim is None:
                direct_unknown = True
            elif killer_team is not None and killer_team != victim_team:
                direct = True
                relation_kind = "killer"
            if kill["assist_status"] != "valid":
                direct_unknown = True
            for assist in kill["assisting_participant_ids"]:
                assist_team = _team_for_pid(team_ids, assist)
                if victim_team is None or assist_team is None:
                    direct_unknown = True
                elif assist_team != victim_team:
                    direct = True
                    if relation_kind == "none":
                        relation_kind = "assist"

        credit_bits = _bits(credited)
        blue_credit = sum(1 for pid in credited if _team_for_pid(team_ids, pid) == 100)
        red_credit = sum(1 for pid in credited if _team_for_pid(team_ids, pid) == 200)
        unknown_credit = len(credited) - blue_credit - red_credit
        if direct:
            direct_status = PASS
        elif direct_unknown or not team_map_ok:
            direct_status = UNKNOWN
            relation_kind = "unknown"
        else:
            direct_status = FAIL
        if direct_status == PASS and blue_credit >= 1 and red_credit >= 1:
            opposition = "confirmed_two_team_firstkill_credit"
            minimum_status = PASS
        elif direct_status == UNKNOWN:
            opposition = "opposing_player_credit_unverified"
            minimum_status = UNKNOWN
        elif blue_credit == 0 or red_credit == 0:
            opposition = "no_confirmed_player_credit"
            minimum_status = FAIL
        else:
            opposition = "opposing_player_credit_unconfirmed"
            minimum_status = FAIL

        own_ordinals = set(member_ordinals)
        lo = pre
        hi = last
        foreign_ordinals = tuple(
            int(row["source_ordinal"])
            for row in valid_kills
            if lo <= int(row["timestamp_ms"]) <= hi and int(row["source_ordinal"]) not in own_ordinals
        )
        # Isolation uses the other component's observation anchor (first kill
        # minus 10 seconds), with a same-first-kill conflict retained
        # explicitly.  The latter is redundant for the closed interval when
        # anchors are valid, but keeping it in the audit makes same-ms
        # conflicts visible even if a source later reports an invalid anchor.
        anchor_left = int(np.searchsorted(component_anchor_array, lo, side="left"))
        anchor_right = int(np.searchsorted(component_anchor_array, hi, side="right"))
        anchor_values = component_anchor_array[anchor_left:anchor_right].astype(np.int64, copy=False).tolist()
        if anchor_left <= component_id < anchor_right:
            del anchor_values[component_id - anchor_left]
        component_anchor_values = tuple(int(value) for value in anchor_values)
        first_left = int(np.searchsorted(component_first_array, f, side="left"))
        first_right = int(np.searchsorted(component_first_array, f, side="right"))
        same_first_kill_components = tuple(
            int(value) for index, value in enumerate(component_first_array[first_left:first_right].tolist(), start=first_left)
            if index != component_id
        )
        component_anchors = component_anchor_values
        isolation_unknown = bool(source_invalid_ordinals)
        isolated = not foreign_ordinals and not component_anchors and not isolation_unknown

        next_kill_index = int(np.searchsorted(valid_times, last, side="right"))
        next_kill = int(valid_times[next_kill_index]) if next_kill_index < valid_times.size else None
        next_component_index = int(np.searchsorted(component_first_array, f, side="right"))
        if next_component_index < component_first_array.size:
            next_first_kill = int(component_first_array[next_component_index])
            later_first_kills = [next_first_kill]
            next_anchor = next_first_kill - ANCHOR_OFFSET_MS
        else:
            later_first_kills = []
            next_anchor = None
        next_anchor_overlap = bool(next_anchor is not None and next_anchor <= last)

        locations = [(float(row["x"]), float(row["y"])) for row in first_block if row["xy_valid"]]
        proxy_values: dict[int, tuple[bool, int, int, int, int]] = {}
        for radius in (1800, 2000):
            proxy_values[radius] = _proxy_at_frame(match, frame_index, locations, credit_bits, float(radius))

        endpoint_values: dict[int, tuple[int | None, tuple[str, ...], bool, int | None, int]] = {}
        for horizon in HORIZONS_S:
            endpoint, truncation = _endpoint(last, horizon, next_kill, next_anchor, end_value)
            post_index = _frame_index(minute_ts, endpoint)
            if (
                endpoint is None
                or minute_ts.size == 0
                or endpoint < int(minute_ts[0])
                or endpoint > int(minute_ts[-1])
            ):
                post_index = -1
            post_frame = int(minute_ts[post_index]) if post_index >= 0 else None
            frame_ok = bool(endpoint is not None and minute_ts.size and minute_ts[0] <= endpoint <= minute_ts[-1])
            endpoint_ok = bool(
                endpoint is not None
                and endpoint >= last
                and endpoint > pre
                and frame_ok
                and terminal_support
                and end_value is not None
                and endpoint < end_value
            )
            endpoint_values[horizon] = (endpoint, truncation, endpoint_ok, post_frame, post_index)

        critical_member_unknown = any(
            any(issue in critical_issue_names for issue in row["issues"])
            for row in members
        )
        terminal_component_order_ok = bool(terminal_support and end_value is not None and last < end_value)
        if not terminal_component_order_ok or not team_map_ok:
            critical_status = FAIL
        elif critical_member_unknown:
            critical_status = UNKNOWN
        else:
            critical_status = PASS
        events_support_ok = not bool(source_invalid_ordinals)
        h90_valid = bool(endpoint_values[90][2])
        frame_support_ok = bool(pre_support and h90_valid)
        # The inclusive candidate flag is a timestamp-lag classification.  A
        # query beyond the last actual frame remains a candidate for audit, but
        # cannot become a fully supported row because ``pre_support`` is false.
        timing_status = PASS if frame_lag_candidate else UNKNOWN if frame_index < 0 else FAIL
        candidate_eligible = bool(
            critical_status == PASS
            and timing_status == PASS
            and isolated
            and direct_status == PASS
            and minimum_status == PASS
        )
        supported_candidate_eligible = bool(
            candidate_eligible
            and events_support_ok
            and terminal_support
            and frame_support_ok
            and h90_valid
        )

        reasons: set[str] = set()
        if source_issues:
            reasons.add("normalization_issue")
        if critical_status == FAIL:
            reasons.add("critical_source_invalid")
        elif critical_status == UNKNOWN:
            reasons.add("critical_source_unknown")
        if not pre_support:
            reasons.add("pre_frame_support_unknown")
        elif not frame_lag_candidate:
            reasons.add("firstkill_frame_lag_outside_inclusive_10000_20000")
        if foreign_ordinals or component_anchors:
            reasons.add("global_isolation_failed_closed_pre_lastkill")
        elif isolation_unknown:
            reasons.add("global_isolation_support_unknown")
        if same_first_kill_components:
            reasons.add("global_isolation_failed_closed_pre_lastkill")
        if direct_status == UNKNOWN:
            reasons.add("firstkill_credit_block_association_unverified")
        elif direct_status == FAIL:
            reasons.add("no_confirmed_opposing_player_credit")
        if env_count:
            reasons.add("environment_killer_zero_not_player")
        if minimum_status == FAIL:
            reasons.add("minimum_team_credit_not_met")
        if next_anchor_overlap:
            reasons.add("next_anchor_overlap_component")
        if f - ANCHOR_OFFSET_MS < PRIMARY_PRE_MIN_MS:
            reasons.add("early_pre_cohort_preserved_separately")
        reasons.add("unknown_starting_scale_retained_for_overall_candidate")
        for horizon in HORIZONS_S:
            if not endpoint_values[horizon][2]:
                reasons.add(f"h{horizon}_endpoint_or_frame_support_invalid")

        all_reasons = sorted(reasons)
        reason_codes = sorted({_reason_code(reason) for reason in all_reasons})
        time_role = "primary_pre_ge_120000" if pre >= PRIMARY_PRE_MIN_MS else "early_pre_lt_120000"
        if candidate_eligible and supported_candidate_eligible:
            eligibility = "candidate_primary_pending_label" if time_role.startswith("primary") else "candidate_early_pending_label"
        elif candidate_eligible:
            eligibility = "candidate_pending_source_support"
        elif critical_status != PASS:
            eligibility = "unverified_critical_source"
        elif timing_status != PASS or not isolated:
            eligibility = "excluded_timing_or_isolation"
        elif minimum_status == UNKNOWN:
            eligibility = "unverified_critical_source"
        elif minimum_status == FAIL:
            eligibility = "review_candidate_no_confirmed_opposition"
        else:
            eligibility = "excluded_minimum_team_credit"

        row: dict[str, Any] = {
            "canonical_match_key": canonical_match_key,
            "patch": patch,
            "component_id": int(component_id),
            "component_key": component_key,
            "member_key": component_key,
            "component_key_sha256": component_key,
            "event_member_list": list(member_ordinals),
            "event_member_list_tuple": tuple(member_ordinals),
            "member_source_ordinals": list(member_ordinals),
            "member_source_index_ranges": [list(pair) for pair in member_ranges],
            "member_ordinal_ranges": [list(pair) for pair in member_ranges],
            "n_members": int(len(members)),
            "same_ms_first_n": int(len(first_block)),
            "n_source_kills": int(len(kills)),
            "n_valid_kills": int(len(valid_kills)),
            "unassigned_kill_ordinals": list(source_invalid_ordinals),
            "optional_issue_kill_ordinals": list(optional_issue_ordinals),
            "normalization_issues": list(source_issues),
            "f": f,
            "first_kill_ms": f,
            "firstkill_ms": f,
            "first_kill_ts": f,
            "L": last,
            "last_kill_ms": last,
            "lastkill_ms": last,
            "last_kill_ts": last,
            "pre": pre,
            "pre_ms": pre,
            "tau": pre,
            "cohort_time_role": time_role,
            "F_pre_ms": frame_ms if frame_ms is not None else -1,
            "F_tau": frame_ms if frame_ms is not None else -1,
            "F_pre_index": int(frame_index),
            "pre_frame_index": int(frame_index),
            "pre_frame_support": bool(pre_support),
            "support_pre": bool(pre_support),
            "pre_frame_age_ms": int(pre - frame_ms) if frame_ms is not None else -1,
            "first_kill_frame_lag_ms": int(lag) if lag is not None else -1,
            "lag_f_F_tau": int(lag) if lag is not None else -1,
            "cand_10_20_tau": bool(frame_lag_candidate),
            "timing_candidate": bool(frame_lag_candidate),
            "timing_status": timing_status,
            "firstkill_credit_bits": int(credit_bits),
            "c0_bits": int(credit_bits),
            "firstkill_credit_pids": sorted(int(pid) for pid in credited),
            "firstkill_credit_blue_pids": sorted(
                int(pid) for pid in credited if _team_for_pid(team_ids, pid) == 100
            ),
            "firstkill_credit_red_pids": sorted(
                int(pid) for pid in credited if _team_for_pid(team_ids, pid) == 200
            ),
            "firstkill_credit_blue": int(blue_credit),
            "firstkill_credit_red": int(red_credit),
            "c0_n_blue": int(blue_credit),
            "c0_n_red": int(red_credit),
            "c0_n_unknown_team": int(unknown_credit),
            "c0_n_credited": int(len(credited)),
            "c0_n": int(len(credited)),
            "c0_n_env_killer": int(env_count),
            "firstkill_direct_opposition_status": direct_status,
            "firstkill_direct_opposition": bool(direct_status == PASS),
            "direct_opposition_confirmed": bool(direct_status == PASS),
            "firstkill_relation_kind": relation_kind,
            "opposition_evidence": opposition,
            "minimum_team_status": minimum_status,
            "team_map_ok": bool(team_map_ok),
            "critical_source_status": critical_status,
            "critical_invalid": bool(critical_status == FAIL),
            "events_support_ok": bool(events_support_ok),
            "terminal_support_authoritative": bool(terminal_support),
            "terminal_component_order_ok": bool(terminal_component_order_ok),
            "game_end_ms": end_value if end_value is not None else -1,
            "winning_team": winner if winner is not None else 0,
            "next_kill_after_L": next_kill if next_kill is not None else -1,
            "next_kill_strict_after_L": next_kill if next_kill is not None else -1,
            "next_component_anchor": next_anchor if next_anchor is not None else -1,
            "next_anchor_ms": next_anchor if next_anchor is not None else -1,
            "next_component_tau": next_anchor if next_anchor is not None else -1,
            "next_anchor_of_component_firstkill": next_anchor if next_anchor is not None else -1,
            "next_component_first_kill_ms": (
                min(later_first_kills) if later_first_kills else -1
            ),
            "overlap_next_anchor_le_L": bool(next_anchor_overlap),
            "overlap_next_tau_le_L": bool(next_anchor_overlap),
            "foreign_kills_in_pre_L": int(len(foreign_ordinals)),
            "foreign_kills_in_tau_L": int(len(foreign_ordinals)),
            "foreign_kill_ordinals": list(foreign_ordinals),
            "other_component_anchors_in_pre_L": int(len(component_anchors)),
            "other_anchors_in_tau_L": int(len(component_anchors)),
            "other_component_anchor_values": list(component_anchors),
            "other_component_firstkill_values": [
                int(value)
                for index, value in enumerate(
                    component_first_array[
                        int(np.searchsorted(component_first_array, lo, side="left")):
                        int(np.searchsorted(component_first_array, hi, side="right"))
                    ].tolist(),
                    start=int(np.searchsorted(component_first_array, lo, side="left")),
                )
                if index != component_id
            ],
            "global_isolation_verified": bool(isolated),
            "isolated": bool(isolated),
            "proxy_participation_mode": "cache_only_proxy",
            "participation_mode": "cache_only_proxy",
            "starting_scale_group": "unknown_scale",
            "actual_participation_verified": False,
            "candidate_eligible": bool(candidate_eligible),
            "supported_candidate_eligible": bool(supported_candidate_eligible),
            "full_supported_eligibility": bool(supported_candidate_eligible),
            "full_supported_eligible": bool(supported_candidate_eligible),
            "isolated_candidate": bool(isolated and frame_lag_candidate),
            "primary_candidate_eligible": bool(candidate_eligible and pre >= PRIMARY_PRE_MIN_MS),
            "early_candidate_eligible": bool(candidate_eligible and pre < PRIMARY_PRE_MIN_MS),
            "overall_retained_pending_label": bool(supported_candidate_eligible),
            "eligibility_status": eligibility,
            "all_reasons": all_reasons,
            "reason_codes": reason_codes,
            "final_population_ready": False,
        }

        for radius in (1800, 2000):
            support, lower, upper, lower_bits, upper_bits = proxy_values[radius]
            row[f"support_ok_tau_r{radius}"] = bool(support)
            row[f"p_lower_n_tau_r{radius}"] = int(lower)
            row[f"p_upper_n_tau_r{radius}"] = int(upper)
            row[f"p_lower_bits_tau_r{radius}"] = int(lower_bits)
            row[f"p_upper_bits_tau_r{radius}"] = int(upper_bits)
            row[f"proxy_P0_lower_r{radius}"] = int(lower)
            row[f"proxy_P0_upper_r{radius}"] = int(upper)
            row[f"proxy_P0_lower_upper_r{radius}"] = (int(lower), int(upper))

        for horizon in HORIZONS_S:
            endpoint, truncation, endpoint_ok, post_frame, post_index = endpoint_values[horizon]
            row[f"e_h{horizon}"] = endpoint if endpoint is not None else -1
            # ``e_h*`` keeps the compact numeric candidate for shard storage;
            # the public endpoint aliases follow the contract and remain NA
            # when support fails instead of silently clamping to a frame.
            row[f"endpoint_ms_h{horizon}"] = endpoint if endpoint_ok else None
            row[f"post_ms_h{horizon}"] = endpoint if endpoint_ok else None
            row[f"valid_h{horizon}"] = bool(endpoint_ok)
            row[f"endpoint_valid_h{horizon}"] = bool(endpoint_ok)
            row[f"h{horizon}_failure"] = bool(not endpoint_ok)
            row[f"frame_support_h{horizon}"] = bool(endpoint is not None and minute_ts.size and minute_ts[0] <= endpoint <= minute_ts[-1])
            row[f"post_frame_support_h{horizon}"] = bool(row[f"frame_support_h{horizon}"])
            row[f"post_frame_index_h{horizon}"] = int(post_index)
            row[f"F_post_h{horizon}"] = post_frame if post_frame is not None else -1
            row[f"truncation_reasons_h{horizon}"] = list(truncation)
            row[f"endpoint_truncation_reasons_h{horizon}"] = list(truncation)
            row[f"reason_codes_h{horizon}"] = list(
                _reason_code(f"h{horizon}_endpoint_or_frame_support_invalid")
                for _ in [0]
                if not endpoint_ok
            )

        rows.append(row)

    return rows


__all__ = [
    "ANCHOR_OFFSET_MS",
    "FRAME_LAG_MAX_MS",
    "FRAME_LAG_MIN_MS",
    "HORIZONS_S",
    "LINK_RADIUS_UNITS",
    "LINK_TIME_MS",
    "PRIMARY_PRE_MIN_MS",
    "PRIMARY_RADIUS_UNITS",
    "REASON_CODES",
    "build_population",
    "normalize_champion_kills",
    "normalize_kills",
]
