"""Pure, cache-only observable state kernel.

The kernel keeps recorded frame values and timestamped records separate.  It
does not estimate survival, buffs, rewards, cooldowns, roles, or inventory
transforms.  Source values (including numeric zero) are copied and retained;
raw-field presence is deliberately represented as unknown provenance.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STATE_VERSION = "observable_state_v1_20261002"
TEAM_BLUE = 100
TEAM_RED = 200
PARTICIPANTS = 10
FRAME_WIDTH = 46
STATIC_WIDTH = 14
BAN_WIDTH = 5

STATIC_NAMES = (
    "champion_id", "summoner_spell_1_id", "summoner_spell_2_id",
    "primary_style_id", "sub_style_id", "primary_rune_1", "primary_rune_2",
    "primary_rune_3", "primary_rune_4", "sub_rune_1", "sub_rune_2",
    "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
)
E_NAMES = ("level_norm", "xp_norm", "curGold_norm", "totalGold_norm", "laneCS_norm", "jgCS_norm")
# These names are the ordered B-history categories.  Source event variants
# stay separate below: an unrecognised monster/building subtype remains in
# history with category ``None`` and never receives an inferred count.
OBJECTIVE_TYPES = (
    "dragon", "baron", "herald", "atakhan", "horde",
    "tower", "inhibitor", "plate", "dragon_soul",
)
OBJECTIVE_SOURCE_TYPES = (
    "ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "DRAGON_SOUL_GIVEN",
)
ITEM_TYPES = ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO", "ITEM_DESTROYED")
HISTORY_GROUPS = ("B", "C", "E", "R", "D")
EVENT_TYPE_NAMES = (
    "BUILDING_KILL", "CHAMPION_KILL", "CHAMPION_SPECIAL_KILL", "DRAGON_SOUL_GIVEN",
    "ELITE_MONSTER_KILL", "FEAT_UPDATE", "GAME_END", "ITEM_DESTROYED", "ITEM_PURCHASED",
    "ITEM_SOLD", "ITEM_UNDO", "LEVEL_UP", "PAUSE_END", "SKILL_LEVEL_UP", "TURRET_PLATE_DESTROYED",
    "WARD_KILL", "WARD_PLACED", "ASCENDED_EVENT", "CAPTURE_POINT", "PORO_KING_SUMMON", "GAME_START", "UNKNOWN",
)
EVENT_PAYLOAD_FIELDS = (
    "participantId", "itemId", "beforeId", "afterId", "killerId", "victimId",
    "killerTeamId", "teamId", "level", "skillSlot",
    "winningTeam", "goldGain", "bounty", "shutdownBounty", "killStreakLength",
)
EVENT_PAYLOAD_FIELD_INDEX = MappingProxyType({name: index for index, name in enumerate(EVENT_PAYLOAD_FIELDS)})
EVENT_STRING_FIELDS = (
    "monsterType", "monsterSubType", "buildingType", "towerType", "laneType",
    "levelUpType", "dragonSoul", "soulType", "name",
)
MONSTER_TYPE_NAMES = (
    "UNKNOWN", "DRAGON", "BARON_NASHOR", "RIFTHERALD", "RIFT_HERALD", "ATAKHAN", "HORDE",
)
MONSTER_SUBTYPE_NAMES = (
    "UNKNOWN", "INFERNAL_DRAGON", "MOUNTAIN_DRAGON", "OCEAN_DRAGON", "CLOUD_DRAGON",
    "HEXTECH_DRAGON", "CHEMTECH_DRAGON", "ELDER_DRAGON",
)
BUILDING_TYPE_NAMES = ("UNKNOWN", "TOWER_BUILDING", "INHIBITOR_BUILDING", "TOWER", "INHIBITOR")
TOWER_TYPE_NAMES = ("UNKNOWN", "OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET", "UNDEFINED_TURRET")
LANE_TYPE_NAMES = ("UNKNOWN", "TOP_LANE", "MID_LANE", "BOT_LANE", "TOP", "MID", "BOT")
LEVEL_UP_TYPE_NAMES = ("UNKNOWN", "NORMAL", "ULTIMATE", "Q", "W", "E", "R")
DRAGON_SOUL_NAMES = ("UNKNOWN", "INFERNAL", "OCEAN", "MOUNTAIN", "CLOUD", "HEXTECH", "CHEMTECH", "ELDER")
SOUL_TYPE_NAMES = DRAGON_SOUL_NAMES
EVENT_STRING_ENUMS = MappingProxyType({
    "monsterType": MONSTER_TYPE_NAMES,
    "monsterSubType": MONSTER_SUBTYPE_NAMES,
    "buildingType": BUILDING_TYPE_NAMES,
    "towerType": TOWER_TYPE_NAMES,
    "laneType": LANE_TYPE_NAMES,
    "levelUpType": LEVEL_UP_TYPE_NAMES,
    "dragonSoul": DRAGON_SOUL_NAMES,
    "soulType": SOUL_TYPE_NAMES,
    # Soul ``name`` is a small verified token in the source contract.  An
    # unrecognised name remains valid-but-unknown in the bank.
    "name": ("UNKNOWN", "DRAGON_SOUL", "SOUL"),
})
EVENT_STRING_BANK_FIELDS = {
    "monsterType": "monster_type", "monsterSubType": "monster_subtype",
    "buildingType": "building_type", "towerType": "tower_type",
    "laneType": "lane_type", "levelUpType": "level_up_type",
    "dragonSoul": "dragon_soul", "soulType": "soul_type", "name": "name",
}
EVENT_BANK_KEYS = (
    "event_timestamp", "event_source_index", "event_identity_code",
    "event_identity_explicit", "event_type_code", "event_category_code",
    "event_identity_raw_rows", "event_identity_raw_offsets", "event_identity_raw_values",
    "event_field_values", "event_field_validity", "event_owner_team",
    "event_owner_validity", "event_acquirer_team", "event_acquirer_validity",
    "event_assist_offsets",
    "event_assist_values", "event_assist_validity", "event_position_xy",
    "event_assist_presence", "event_position_validity", "event_position_presence", "event_string_codes", "event_string_validity",
    "event_string_unknown_rows", "event_string_unknown_cols",
    "event_string_unknown_offsets", "event_string_unknown_values",
    "event_raw_rows", "event_raw_cols", "event_raw_offsets", "event_raw_values",
    "item_event_indices",
)


def _readonly(value: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Copy an array and make the copy immutable."""
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _compact_int(value: Any) -> np.ndarray:
    """Use int32 for the normal event range and retain int64 when required."""
    arr = np.asarray(value, dtype=np.int64)
    info = np.iinfo(np.int32)
    if arr.size == 0 or (int(np.min(arr)) >= info.min and int(np.max(arr)) <= info.max):
        return arr.astype(np.int32)
    return arr


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _int_value(value: Any, *, nonnegative: bool = False) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.startswith(("+", "-")):
            digits = text[1:]
        else:
            digits = text
        if not digits.isdigit():
            return None
        result = int(text)
    else:
        return None
    return result if (not nonnegative or result >= 0) else None


def _timestamp(value: Any) -> int | None:
    return _int_value(value, nonnegative=True)


def _freeze_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # ``_event_payload`` already copied the allow-listed source values.  Freeze
    # that projection recursively without paying for a second deep copy.
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        if isinstance(item, set):
            return frozenset(freeze(child) for child in item)
        return item

    return MappingProxyType({key: freeze(child) for key, child in value.items()})


def _thaw_value(value: Any) -> Any:
    """Create a mutable query projection from an immutable typed value."""
    if isinstance(value, Mapping):
        return {key: _thaw_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(child) for child in value]
    if isinstance(value, frozenset):
        return {_thaw_value(child) for child in value}
    return value


def _valid_pid(value: Any) -> int | None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        return None
    pid = _int_value(value)
    return pid if pid is not None and 1 <= pid <= PARTICIPANTS else None


def _valid_team(value: Any) -> int | None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        return None
    team = _int_value(value)
    return team if team in (TEAM_BLUE, TEAM_RED) else None


def _array_alias(arrays: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in arrays:
            return arrays[name]
    return None


def _frame_schema() -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """Load the adopted order through the side-effect-free schema helper."""
    # The import is local by design: importing this kernel never imports old
    # production config modules.  schema.frame_spec reads the allow-listed AST
    # extractor and the adopted policy JSON.
    try:
        from .schema import frame_spec

        names, indices, groups = frame_spec()
        return tuple(names), tuple(int(i) for i in indices), tuple(groups)
    except Exception as exc:  # pragma: no cover - only used if the policy is unavailable
        raise RuntimeError("adopted frame schema could not be loaded") from exc


@dataclass(frozen=True)
class TypedEvent:
    """Timestamp-ordered projection of one recorded event."""

    timestamp_ms: int | None
    event_type: str
    source_index: int
    source_identity: str
    timestamp_valid: bool
    duplicate_identity: bool
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "event_type": self.event_type,
            "source_index": self.source_index,
            "source_identity": self.source_identity,
            "timestamp_valid": self.timestamp_valid,
            "duplicate_identity": self.duplicate_identity,
            "payload": _thaw_value(self.payload),
        }


def _readonly_event_view(event: TypedEvent) -> Mapping[str, Any]:
    """Build the legacy mapping view without copying the typed payload."""
    return MappingProxyType({
        "timestamp_ms": event.timestamp_ms,
        "event_type": event.event_type,
        "source_index": event.source_index,
        "source_identity": event.source_identity,
        "timestamp_valid": event.timestamp_valid,
        "duplicate_identity": event.duplicate_identity,
        "payload": event.payload,
    })


def _source_identity(event: Mapping[str, Any], ordinal: int) -> tuple[str, bool]:
    """Return an explicit source identity when one was recorded.

    The list ordinal remains the stable source index.  An explicit identity is
    used only to detect identity reuse; equal payloads at different ordinals
    are retained as distinct records because payload equality does not prove a
    duplicate event.
    """
    for key in ("source_id", "sourceId", "source_index", "sourceIndex", "event_index", "eventIndex"):
        if key in event:
            value = event.get(key)
            if isinstance(value, (str, int, np.integer)) and not isinstance(value, bool):
                text = str(value)
                return (text if key in {"source_id", "sourceId"} and ":" in text else f"{key}:{text}"), True
    return f"ordinal:{ordinal}", False


def _normal_token(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def _enum_code(value: Any, names: Sequence[str]) -> tuple[int, bool]:
    token = _normal_token(value)
    if token is None:
        return 0, False
    try:
        code = int(tuple(names).index(token))
        # Keep noncanonical source spelling in the compact unknown-token
        # side table so export/reload remains a typed projection, not a
        # silently normalised replacement.
        return (code, True) if str(value) == str(names[code]) else (0, True)
    except ValueError:
        return 0, True


def _enum_value(code: Any, names: Sequence[str]) -> str | None:
    parsed = _int_value(code, nonnegative=True)
    if parsed is None or parsed <= 0 or parsed >= len(names):
        return None
    return str(names[parsed])


def _objective_category(event_type: str, payload: Mapping[str, Any]) -> str | None:
    """Map only verified objective variants to the ordered B categories."""
    recorded = _normal_token(payload.get("_objective_category"))
    if recorded is not None and recorded.lower() in OBJECTIVE_TYPES:
        return recorded.lower()
    if event_type == "ELITE_MONSTER_KILL":
        return {
            "DRAGON": "dragon",
            "BARON_NASHOR": "baron",
            "RIFTHERALD": "herald",
            "RIFT_HERALD": "herald",
            "ATAKHAN": "atakhan",
            "HORDE": "horde",
        }.get(_normal_token(payload.get("monsterType")))
    if event_type == "BUILDING_KILL":
        return {
            "TOWER_BUILDING": "tower",
            "TOWER": "tower",
            "INHIBITOR_BUILDING": "inhibitor",
            "INHIBITOR": "inhibitor",
        }.get(_normal_token(payload.get("buildingType")))
    if event_type == "TURRET_PLATE_DESTROYED":
        return "plate"
    if event_type == "DRAGON_SOUL_GIVEN":
        return "dragon_soul"
    return None


def _position_pair(value: Any) -> tuple[float, float] | None:
    source = value if isinstance(value, Mapping) else None
    if source is None:
        return None
    left, right = source.get("x"), source.get("y")
    if not isinstance(left, (int, float, np.integer, np.floating)) or isinstance(left, (bool, np.bool_)):
        return None
    if not isinstance(right, (int, float, np.integer, np.floating)) or isinstance(right, (bool, np.bool_)):
        return None
    left_float, right_float = float(left), float(right)
    if not math.isfinite(left_float) or not math.isfinite(right_float):
        return None
    return left_float, right_float


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the compact, declared typed projection of one source event."""
    payload: dict[str, Any] = {}
    for name in EVENT_PAYLOAD_FIELDS:
        if name in event:
            value = event.get(name)
            # Explicit invalid values cannot trigger a fallback or become
            # valid IDs after serialization.
            payload[name] = deepcopy(value)
    for name in EVENT_STRING_FIELDS:
        value = event.get(name)
        if isinstance(value, str):
            payload[name] = value
    if "assistingParticipantIds" in event:
        values = event.get("assistingParticipantIds")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            payload["assistingParticipantIds"] = [deepcopy(value) for value in values]
        else:
            # Preserve field presence while avoiding an arbitrary raw object.
            payload["assistingParticipantIds"] = None
    if "position" in event or ("x" in event and "y" in event):
        position = event.get("position") if "position" in event else event
        pair = _position_pair(position)
        if pair is not None:
            payload["position"] = {"x": pair[0], "y": pair[1]}
        else:
            # Keep source presence without retaining an invalid raw object.
            payload["position"] = {}
    # A bank reload may carry the already verified category as a compact
    # marker.  It is never read from the original raw event.
    if "_objective_category" in event and isinstance(event.get("_objective_category"), str):
        payload["_objective_category"] = str(event["_objective_category"])
    return payload


def _event_records(events: Sequence[Any]) -> tuple[tuple[TypedEvent, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        raise ValueError("events must be a sequence of recorded mappings")
    seen: set[str] = set()
    records: list[TypedEvent] = []
    reasons: list[str] = []
    for ordinal, item in enumerate(events):
        if not isinstance(item, Mapping):
            reasons.append(f"event_{ordinal}_not_mapping")
            records.append(TypedEvent(None, "UNKNOWN", ordinal, f"ordinal:{ordinal}", False, False, MappingProxyType({"invalid_record": True})))
            continue
        source_ordinal = _int_value(item.get("_source_ordinal"))
        if source_ordinal is None or source_ordinal < 0:
            source_ordinal = ordinal
        event_type = item.get("type")
        event_type = str(event_type) if isinstance(event_type, str) and event_type else "UNKNOWN"
        timestamp = _timestamp(item.get("timestamp"))
        if timestamp is None:
            reasons.append(f"event_{ordinal}_invalid_timestamp")
        identity, _explicit = _source_identity(item, source_ordinal)
        duplicate = identity in seen
        if duplicate:
            reasons.append(f"duplicate_source_identity:{identity}")
        seen.add(identity)
        records.append(TypedEvent(timestamp, event_type, source_ordinal, identity, timestamp is not None, duplicate, _freeze_payload(_event_payload(item))))
    records.sort(key=lambda record: (0, int(record.timestamp_ms), int(record.source_index)) if record.timestamp_valid else (1, 0, int(record.source_index)))
    return tuple(records), tuple(sorted(set(reasons))), tuple(sorted(seen))


def _map_team_ids(meta: Mapping[str, Any]) -> np.ndarray:
    raw = meta.get("team_map")
    if raw is None:
        raw = meta.get("team_ids")
    values: list[Any]
    if isinstance(raw, Mapping):
        values = [raw.get(str(pid), raw.get(pid)) for pid in range(1, PARTICIPANTS + 1)]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = []
    if len(values) != PARTICIPANTS:
        raise ValueError("team_map must contain participant IDs 1 through 10")
    result = np.asarray([_valid_team(value) or 0 for value in values], dtype=np.int16)
    if set(result.tolist()) != {TEAM_BLUE, TEAM_RED} or int(np.sum(result == TEAM_BLUE)) != 5 or int(np.sum(result == TEAM_RED)) != 5:
        raise ValueError("team_map must be an exact 5v5 partition of participant IDs 1 through 10")
    result.setflags(write=False)
    return result


def _recorded_id(value: Any) -> tuple[int, bool]:
    parsed = _int_value(value)
    if parsed is None:
        return 0, False
    # Preserve nonpositive recorded integers while keeping their meaning unknown.
    return parsed, parsed > 0


def _pid_value(container: Any, pid: int) -> Any:
    if isinstance(container, Mapping):
        return container.get(str(pid), container.get(pid))
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)) and len(container) >= pid:
        return container[pid - 1]
    return None


def _nested_id(container: Any, pid: int, key: str) -> Any:
    value = _pid_value(container, pid)
    if isinstance(value, Mapping):
        return value.get(key)
    if key.endswith("_1_id") and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 1:
        return value[0]
    if key.endswith("_2_id") and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        return value[1]
    return None


def _static_arrays(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    static = meta.get("static_meta")
    static = static if isinstance(static, Mapping) else {}
    champions = static.get("champion_by_pid")
    spells = static.get("summoner_spells_by_pid")
    runes = static.get("runes_by_pid")
    unknown: list[str] = []
    result = np.zeros((PARTICIPANTS, STATIC_WIDTH), dtype=np.int64)
    for row, pid in enumerate(range(1, PARTICIPANTS + 1)):
        values = (
            _pid_value(champions, pid),
            _nested_id(spells, pid, "summoner_spell_1_id"),
            _nested_id(spells, pid, "summoner_spell_2_id"),
            _nested_id(runes, pid, "primary_style_id"),
            _nested_id(runes, pid, "sub_style_id"),
            _nested_id(runes, pid, "primary_rune_1"),
            _nested_id(runes, pid, "primary_rune_2"),
            _nested_id(runes, pid, "primary_rune_3"),
            _nested_id(runes, pid, "primary_rune_4"),
            _nested_id(runes, pid, "sub_rune_1"),
            _nested_id(runes, pid, "sub_rune_2"),
            _nested_id(runes, pid, "stat_perk_offense"),
            _nested_id(runes, pid, "stat_perk_flex"),
            _nested_id(runes, pid, "stat_perk_defense"),
        )
        for col, value in enumerate(values):
            result[row, col], valid = _recorded_id(value)
            if not valid:
                unknown.append(f"C.{STATIC_NAMES[col]}:pid={pid}")
    bans_raw = static.get("bans")
    bans = np.zeros((2, BAN_WIDTH), dtype=np.int64)
    for team_row, team_name in enumerate(("blue", "red")):
        values = bans_raw.get(team_name) if isinstance(bans_raw, Mapping) else None
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            unknown.append(f"C.bans.{team_name}")
            values = []
        for col in range(BAN_WIDTH):
            value = values[col] if col < len(values) else None
            bans[team_row, col], valid = _recorded_id(value)
            if not valid:
                unknown.append(f"C.bans.{team_name}[{col}]")
    result = _readonly(result, np.int64)
    bans = _readonly(bans, np.int64)
    return result, bans, tuple(sorted(set(unknown)))


def _direct_integer_block(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Validate an already projected categorical block without remapping it."""
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype.kind not in "iu" or arr.dtype.kind == "b":
        raise ValueError(f"{name} must have integer shape {shape}")
    out = np.asarray(arr, dtype=np.int64)
    return _readonly(out, np.int64)


def _terminal(events: Sequence[TypedEvent], meta: Mapping[str, Any]) -> tuple[int | None, int | None, tuple[str, ...]]:
    reasons: list[str] = []
    candidates: list[tuple[int, int]] = []
    marker_identities: set[str] = set()
    malformed_marker = False
    for event in events:
        if event.event_type != "GAME_END":
            continue
        if event.duplicate_identity:
            continue
        if not event.timestamp_valid:
            malformed_marker = True
            continue
        winner = _valid_team(event.payload.get("winningTeam"))
        if winner is not None:
            marker_identities.add(event.source_identity)
            candidates.append((int(event.timestamp_ms), int(winner)))
        else:
            malformed_marker = True
    if malformed_marker:
        reasons.append("game_end_malformed_or_unknown")
    if len(marker_identities) != 1 or malformed_marker:
        if not marker_identities:
            reasons.append("game_end_missing_or_unknown")
        else:
            reasons.append("game_end_nonunique_or_ambiguous")
        # A metadata copy cannot replace the independently recorded terminal
        # marker.  The sole exception is a bank that explicitly records its
        # verified terminal provenance during numeric reload.
        if meta.get("_verified_end_from_bank") is not True:
            return None, None, tuple(sorted(set(reasons)))
    unique = sorted(set(candidates))
    if not unique and meta.get("_verified_end_from_bank") is not True:
        reasons.append("game_end_missing_or_unknown")
        return None, None, tuple(sorted(set(reasons)))
    if len(unique) != 1:
        reasons.append("game_end_nonunique_or_ambiguous")
        if meta.get("_verified_end_from_bank") is not True:
            return None, None, tuple(sorted(set(reasons)))
    if unique:
        return unique[0][0], unique[0][1], tuple(sorted(set(reasons)))
    explicit_end = _timestamp(meta.get("end_ms"))
    explicit_winner = _valid_team(meta.get("winning_team"))
    if explicit_end is None or explicit_winner is None:
        reasons.append("terminal_metadata_partial")
        return None, None, tuple(sorted(set(reasons)))
    return explicit_end, explicit_winner, tuple(sorted(set(reasons)))


def _event_team(event: TypedEvent, team_ids: np.ndarray) -> int | None:
    # Prefer an explicitly recorded killer team.  An explicitly present but
    # invalid team must remain unknown; falling through would manufacture
    # player credit from a conflicting or neutral record.
    if "killerTeamId" in event.payload:
        team = _valid_team(event.payload.get("killerTeamId"))
        if team is None:
            return None
        killer = _valid_pid(event.payload.get("killerId"))
        if killer is not None and int(team_ids[killer - 1]) != team:
            return None
        return team
    killer = _valid_pid(event.payload.get("killerId"))
    if killer is not None:
        return int(team_ids[killer - 1])
    team = _valid_team(event.payload.get("teamId"))
    if team is not None and event.event_type in {"BUILDING_KILL", "TURRET_PLATE_DESTROYED"}:
        # Riot's structure ``teamId`` is the destroyed owner's team.  The
        # acquiring team is the opposing side when no killer ID is recorded.
        return TEAM_RED if team == TEAM_BLUE else TEAM_BLUE
    if team is not None and event.event_type == "DRAGON_SOUL_GIVEN":
        return team
    return None


def _record_dict(event: TypedEvent, **extra: Any) -> dict[str, Any]:
    result = {
        "timestamp_ms": int(event.timestamp_ms) if event.timestamp_valid else None,
        "event_type": event.event_type,
        "source_index": int(event.source_index),
        "source_identity": event.source_identity,
        "duplicate_identity": bool(event.duplicate_identity),
    }
    result.update(extra)
    return result


def _nan_or_delta(left: float, right: float) -> float:
    return float(left - right) if math.isfinite(left) and math.isfinite(right) else float("nan")


def _aggregate(values: np.ndarray, *, mean: bool) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite) if mean else np.sum(finite))


def _history_fields() -> list[dict[str, Any]]:
    fields = [
        {"id": "B.elapsed_minutes", "group": "B", "source_path": "query.timestamp_ms", "source_timestamp_rule": "query cutoff", "parent_fields": ["query_ms"], "unit": "minutes", "dtype": "float64", "transform": "query_ms/60000", "missingness_or_presence_unknown": "not applicable", "reconstruction_policy": "recorded clock only"},
        {"id": "B.frame_age_seconds", "group": "B", "source_path": "minute_ts[F_query]", "source_timestamp_rule": "latest supported frame F<=query", "parent_fields": ["query_ms", "F_query_ms"], "unit": "seconds", "dtype": "float64", "transform": "(query_ms-F_query_ms)/1000", "missingness_or_presence_unknown": "unknown before first supported frame", "reconstruction_policy": "held frame age"},
        {"id": "B.team_growth", "group": "B", "source_path": "node_frame[F,E] and gold_team[F]", "source_timestamp_rule": "latest supported frame F<=query", "parent_fields": ["E", "team_map", "gold_team"], "unit": "cache normalized units", "dtype": "float64", "transform": "per-team level mean and accumulation sums", "missingness_or_presence_unknown": "raw presence unknown; zero retained", "reconstruction_policy": "held observed frame aggregate"},
        {"id": "B.team_kills", "group": "B", "source_path": "events[].CHAMPION_KILL", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["killerId", "team_map"], "unit": "recorded count", "dtype": "int64", "transform": "deduplicated source identity; killerId=0 has no team credit", "missingness_or_presence_unknown": "malformed credit remains unknown", "reconstruction_policy": "history prefix only"},
        {"id": "B.objective_structure_history", "group": "B", "source_path": "events[].objective types", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["timestamp", "type", "monsterType", "monsterSubType", "buildingType", "killerTeamId", "teamId", "position"], "unit": "recorded count and age", "dtype": "typed records", "transform": "verified subtype categories only; no current buff/state inference", "missingness_or_presence_unknown": "unknown subtype/team remains in history but uncredited", "reconstruction_policy": "retain raw subtype, ownership, acquirer, and position fields"},
        {"id": "C.item_transaction_history", "group": "C", "source_path": "events[].ITEM_*", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["participantId", "itemId", "beforeId", "afterId"], "unit": "recorded transaction", "dtype": "typed records", "transform": "identity; no automatic grants/transforms", "missingness_or_presence_unknown": "malformed payload retained as unknown", "reconstruction_policy": "source identity and ordinal retained"},
        {"id": "C.current_item_ledger", "group": "C", "source_path": "events[].ITEM_* replay", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["C.item_transaction_history"], "unit": "item multiset", "dtype": "ragged integer", "transform": "direct replay only", "missingness_or_presence_unknown": "NaN payload plus verified reason when unknown", "reconstruction_policy": "no inferred grants/transforms; underflow and same-ms ambiguity unknown"},
        {"id": "E.confirmed_level", "group": "E", "source_path": "events[].LEVEL_UP", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["participantId", "level"], "unit": "recorded level", "dtype": "typed records", "transform": "identity; no inferred growth", "missingness_or_presence_unknown": "malformed payload retained", "reconstruction_policy": "history only"},
        {"id": "R.confirmed_skill_history", "group": "R", "source_path": "events[].SKILL_LEVEL_UP", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["participantId", "skillSlot", "levelUpType"], "unit": "recorded skill event", "dtype": "typed records", "transform": "identity; duplicate identity does not add rank", "missingness_or_presence_unknown": "ambiguous duplicate retained with reason", "reconstruction_policy": "history only"},
        {"id": "D.player_kills_deaths_assists", "group": "D", "source_path": "events[].CHAMPION_KILL", "source_timestamp_rule": "all recorded event timestamps<=query", "parent_fields": ["killerId", "victimId", "assistingParticipantIds", "team_map"], "unit": "recorded cumulative count", "dtype": "int64", "transform": "player credit only; environmental killer 0 has no kill credit", "missingness_or_presence_unknown": "invalid IDs retain unknown diagnostics", "reconstruction_policy": "deduplicated source identity history"},
    ]
    return fields


def history_feature_description() -> list[dict[str, Any]]:
    """Return ordered history fields for the root schema manifest."""
    return deepcopy(_history_fields())


def ordered_feature_description() -> dict[str, Any]:
    """Return the executable ordered feature card without audit metadata."""
    names, indices, groups = _frame_schema()
    fields: list[dict[str, Any]] = []
    for index, (name, source_index, group) in enumerate(zip(names, indices, groups)):
        canonical = "last_frame_power_fraction" if name == "mp_pct" else name
        fields.append({
            "id": f"{group}.{canonical}", "group": group, "block": "node_frame", "index": index,
            "source_index": int(source_index), "source_path": f"npz.node_minute[F,participantId-1,{source_index}]",
            "source_timestamp_rule": "latest supported frame F<=query", "parent_fields": [f"cache.node.{name}"],
            "unit": "cache normalized value", "dtype": "float32", "transform": "retain cached value",
            "missingness_or_presence_unknown": "raw presence unknown; zero retained",
            "reconstruction_policy": "held observation; no interpolation or inverse clipping",
        })
    for index, name in enumerate(STATIC_NAMES):
        fields.append({
            "id": f"C.{name}", "group": "C", "block": "static_ids", "index": index,
            "source_path": "meta.static_meta[participantId]", "source_timestamp_rule": "fixed start selection only",
            "parent_fields": [name], "unit": "recorded categorical ID", "dtype": "int64", "transform": "identity",
            "missingness_or_presence_unknown": "raw presence unknown; zero retained",
            "reconstruction_policy": "preserve recorded selection; no name hash or role slots",
        })
    for index, axis in enumerate(("x", "y")):
        fields.append({
            "id": f"S.{axis}", "group": "S", "block": "xy", "index": index,
            "source_path": "npz.xy_raw_minute[F,participantId-1]", "source_timestamp_rule": "latest supported frame F<=query",
            "parent_fields": ["xy_raw_minute"], "unit": "map coordinate units", "dtype": "float32", "transform": "identity",
            "missingness_or_presence_unknown": "raw presence unknown; zero retained",
            "reconstruction_policy": "no interpolation or first-kill centering",
        })
    fields.extend(_history_fields())
    return {
        "state_version": STATE_VERSION,
        "ordered_fields": fields,
        "frame_width": FRAME_WIDTH,
        "static_width": STATIC_WIDTH,
        "participant_order": list(range(1, PARTICIPANTS + 1)),
        "main_V_groups": ["B", "E", "C", "R", "D"],
        "main_q_common_groups": ["B", "E", "C", "R"],
        "raw_presence_verified": False,
        "inferred_zero_masks": False,
        "audit_metadata_is_model_input": False,
        "training_ready": False,
    }


class ObservableMatch:
    """Immutable observed match inputs and timestamp-cutoff state queries."""

    def __init__(self, arrays: Mapping[str, Any], meta: Mapping[str, Any], events: Sequence[Any], patch: str):
        if not isinstance(arrays, Mapping):
            raise TypeError("arrays must be a mapping")
        if not isinstance(meta, Mapping):
            raise TypeError("meta must be a mapping")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("patch must be a non-empty string")
        self.patch = patch.strip()
        self._meta = deepcopy(dict(meta))

        minute_ts = _array_alias(arrays, "minute_ts")
        node = _array_alias(arrays, "node_frame", "node_minute")
        xy = _array_alias(arrays, "xy", "xy_raw_minute")
        gold = _array_alias(arrays, "gold_team", "gold_team_minute")
        if minute_ts is None or node is None or xy is None or gold is None:
            raise ValueError("arrays must include minute_ts, node_frame, xy, and gold_team")
        ts = np.asarray(minute_ts)
        if ts.ndim != 1 or ts.dtype.kind not in "iu" or ts.dtype.kind == "b" or ts.size == 0:
            raise ValueError("minute_ts must be a non-empty integer vector")
        ts64 = np.asarray(ts, dtype=np.int64)
        if np.any(ts64 < 0) or np.any(ts64[1:] <= ts64[:-1]):
            raise ValueError("minute_ts must be nonnegative and strictly increasing")
        node_arr = np.asarray(node)
        xy_arr = np.asarray(xy)
        gold_arr = np.asarray(gold)
        if node_arr.ndim != 3 or node_arr.shape != (ts64.size, PARTICIPANTS, FRAME_WIDTH):
            raise ValueError(f"node_frame must have shape ({ts64.size},10,46)")
        if xy_arr.ndim != 3 or xy_arr.shape != (ts64.size, PARTICIPANTS, 2):
            raise ValueError(f"xy must have shape ({ts64.size},10,2)")
        if gold_arr.ndim != 2 or gold_arr.shape != (ts64.size, 2):
            raise ValueError(f"gold_team must have shape ({ts64.size},2)")
        for name, arr in (("node_frame", node_arr), ("xy", xy_arr), ("gold_team", gold_arr)):
            if arr.dtype.kind != "f" or arr.dtype.itemsize != 4:
                raise ValueError(f"{name} must be float32")
        self.minute_ts = _readonly(ts64, np.int64)
        self.node_frame = _readonly(node_arr, np.float32)
        self.xy = _readonly(xy_arr, np.float32)
        self.gold_team = _readonly(gold_arr, np.float32)
        # Aliases make source/storage integration explicit while sharing the
        # same immutable blocks.
        self.node_minute = self.node_frame
        self.xy_raw_minute = self.xy
        self.gold_team_minute = self.gold_team

        meta_for_team = dict(self._meta)
        if "team_ids" in arrays:
            meta_for_team["team_ids"] = arrays["team_ids"]
        self.team_ids = _map_team_ids(meta_for_team)
        self.static_ids, self.bans, static_unknown = _static_arrays(self._meta)
        if "static_ids" in arrays:
            self.static_ids = _direct_integer_block(arrays["static_ids"], (PARTICIPANTS, STATIC_WIDTH), "static_ids")
            static_unknown = tuple()
            static_unknown = tuple(
                sorted(set(static_unknown) | {
                    f"C.{STATIC_NAMES[col]}:pid={pid}"
                    for pid in range(1, PARTICIPANTS + 1)
                    for col in range(STATIC_WIDTH)
                    if int(self.static_ids[pid - 1, col]) <= 0
                })
            )
        if "bans" in arrays:
            self.bans = _direct_integer_block(arrays["bans"], (2, BAN_WIDTH), "bans")
            static_unknown = tuple(
                sorted(set(static_unknown) | {
                    f"C.bans.{team}[{col}]"
                    for team, row in (("blue", 0), ("red", 1))
                    for col in range(BAN_WIDTH)
                    if int(self.bans[row, col]) <= 0
                })
            )
        self.static_unknown_fields = tuple(sorted(set(static_unknown)))
        terminal_meta = dict(self._meta)
        if "end_ms" in arrays:
            terminal_meta["end_ms"] = arrays["end_ms"]
        if "winning_team" in arrays:
            terminal_meta["winning_team"] = arrays["winning_team"]
        typed_events, event_reasons, _ = _event_records(events)
        self.typed_events = typed_events
        self._events_projection: tuple[Mapping[str, Any], ...] | None = None
        self._event_reasons = tuple(sorted(set(event_reasons + self.static_unknown_fields)))
        self.end_ms, self.winning_team, terminal_reasons = _terminal(typed_events, terminal_meta)
        self._event_reasons = tuple(sorted(set(self._event_reasons + terminal_reasons)))
        self.end_ms_value = self.end_ms
        self.winning_team_value = self.winning_team
        self.raw_presence_unknown = True

        self._valid_events = tuple(event for event in typed_events if event.timestamp_valid)
        self._event_timestamps = _readonly(np.asarray([int(event.timestamp_ms) for event in self._valid_events], dtype=np.int64), np.int64)
        self._duplicate_identity = np.asarray([event.duplicate_identity for event in self._valid_events], dtype=bool)
        self._duplicate_identity.setflags(write=False)
        self._valid_index_by_source = {event.source_index: i for i, event in enumerate(self._valid_events)}
        self._frame_indices_by_team = {
            TEAM_BLUE: np.flatnonzero(np.asarray(self.team_ids) == TEAM_BLUE),
            TEAM_RED: np.flatnonzero(np.asarray(self.team_ids) == TEAM_RED),
        }
        for value in self._frame_indices_by_team.values():
            value.setflags(write=False)
        self._prepare_event_bank()

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Lazily expose the immutable legacy event-mapping projection."""
        if self._events_projection is None:
            self._events_projection = tuple(_readonly_event_view(event) for event in self.typed_events)
        return self._events_projection

    def _prepare_event_bank(self) -> None:
        n = len(self._valid_events)
        kill_delta = np.zeros((n, 2 + PARTICIPANTS * 3 + 1), dtype=np.int64)
        objective_delta = np.zeros((n, len(OBJECTIVE_TYPES), 3), dtype=np.int64)
        self._kill_records: list[TypedEvent] = []
        self._objective_records: list[tuple[TypedEvent, int | None, str | None]] = []
        unknown_objectives: list[str] = []
        self._level_records: list[TypedEvent] = []
        self._skill_records: list[TypedEvent] = []
        self._item_records: list[TypedEvent] = []
        seen_kill_identity: set[str] = set()
        seen_objective_identity: set[str] = set()
        seen_level_identity: set[str] = set()
        seen_skill_identity: set[str] = set()
        seen_item_identity: set[str] = set()
        item_event_indices: list[int] = []
        for index, event in enumerate(self._valid_events):
            duplicate_identity = event.duplicate_identity
            identity = event.source_identity
            countable = not duplicate_identity
            if event.event_type == "CHAMPION_KILL":
                self._kill_records.append(event)
                if countable and identity not in seen_kill_identity:
                    seen_kill_identity.add(identity)
                    killer = _valid_pid(event.payload.get("killerId"))
                    victim = _valid_pid(event.payload.get("victimId"))
                    team = int(self.team_ids[killer - 1]) if killer is not None else None
                    if killer is not None:
                        kill_delta[index, 2 + (killer - 1) * 3] = 1
                        kill_delta[index, 0 if team == TEAM_BLUE else 1] = 1
                    elif _int_value(event.payload.get("killerId")) == 0:
                        kill_delta[index, -1] = 1
                    if victim is not None:
                        kill_delta[index, 2 + (victim - 1) * 3 + 1] = 1
                    assists = event.payload.get("assistingParticipantIds")
                    assist_ids: set[int] = set()
                    if isinstance(assists, Sequence) and not isinstance(assists, (str, bytes, bytearray)):
                        for value in assists:
                            pid = _valid_pid(value)
                            if pid is not None:
                                assist_ids.add(pid)
                    for pid in assist_ids:
                        kill_delta[index, 2 + (pid - 1) * 3 + 2] = 1
            elif event.event_type in OBJECTIVE_SOURCE_TYPES:
                team = _event_team(event, self.team_ids)
                category = _objective_category(event.event_type, event.payload)
                self._objective_records.append((event, team, category))
                if category is None:
                    unknown_objectives.append(f"unknown_objective_category:{event.event_type}:source={event.source_index}")
                if category is not None and countable and identity not in seen_objective_identity:
                    seen_objective_identity.add(identity)
                    type_index = OBJECTIVE_TYPES.index(category)
                    objective_delta[index, type_index, 0] = 1
                    if team == TEAM_BLUE:
                        objective_delta[index, type_index, 1] = 1
                    elif team == TEAM_RED:
                        objective_delta[index, type_index, 2] = 1
            elif event.event_type == "LEVEL_UP":
                self._level_records.append(event)
                if countable and identity not in seen_level_identity:
                    seen_level_identity.add(identity)
            elif event.event_type == "SKILL_LEVEL_UP":
                self._skill_records.append(event)
                if countable and identity not in seen_skill_identity:
                    seen_skill_identity.add(identity)
            elif event.event_type in ITEM_TYPES:
                self._item_records.append(event)
                item_event_indices.append(index)
                if countable and identity not in seen_item_identity:
                    seen_item_identity.add(identity)
        self._kill_prefix = np.concatenate((np.zeros((1, kill_delta.shape[1]), dtype=np.int64), np.cumsum(kill_delta, axis=0)), axis=0)
        self._objective_prefix = np.concatenate((np.zeros((1, len(OBJECTIVE_TYPES), 3), dtype=np.int64), np.cumsum(objective_delta, axis=0)), axis=0)
        self._kill_prefix.setflags(write=False)
        self._objective_prefix.setflags(write=False)
        self._item_event_indices = _readonly(np.asarray(item_event_indices, dtype=np.int64), np.int64)
        self._unknown_objective_categories = tuple(sorted(set(unknown_objectives)))
        self._event_reasons = tuple(sorted(set(self._event_reasons) | set(self._unknown_objective_categories)))
        self._item_snapshots, self._item_unknown_at = self._prepare_item_snapshots()
        last = np.full(33, np.nan, dtype=np.float64)
        timing = [last.copy()]
        for event in self._kill_records:
            if not event.duplicate_identity:
                timestamp = float(event.timestamp_ms)
                last[0] = timestamp
                killer = _valid_pid(event.payload.get("killerId"))
                victim = _valid_pid(event.payload.get("victimId"))
                if killer is not None:
                    last[1 if self.team_ids[killer - 1] == TEAM_BLUE else 2] = timestamp
                    last[3 + (killer - 1) * 3] = timestamp
                if victim is not None:
                    last[3 + (victim - 1) * 3 + 1] = timestamp
                values = event.payload.get("assistingParticipantIds")
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                    for pid in set(pid for pid in map(_valid_pid, values) if pid is not None):
                        last[3 + (pid - 1) * 3 + 2] = timestamp
            timing.append(last.copy())
        self._kill_timing_prefix = _readonly(np.asarray(timing), np.float64)
        self._kill_event_indices = _readonly(np.asarray([self._valid_index_by_source[e.source_index] for e in self._kill_records]), np.int64)

    def _prepare_item_snapshots(self) -> tuple[tuple[dict[int, dict[int, int]], ...], dict[int, str]]:
        state: dict[int, dict[int, int]] = {pid: {} for pid in range(1, PARTICIPANTS + 1)}
        snapshots: list[dict[int, dict[int, int]]] = []
        unknown_at: dict[int, str] = {}
        prior_count_by_pid_timestamp: Counter[tuple[int, int]] = Counter()
        seen_identity: set[str] = set()
        for event_index in self._item_event_indices.tolist():
            event = self._valid_events[int(event_index)]
            pid = _valid_pid(event.payload.get("participantId"))
            reason: str | None = None
            if pid is None:
                reason = "item_participant_unknown"
            timestamp = int(event.timestamp_ms)
            if pid is not None:
                key = (pid, timestamp)
                prior_count_by_pid_timestamp[key] += 1
                if prior_count_by_pid_timestamp[key] > 1:
                    reason = "same_timestamp_item_order_ambiguous"
            if event.duplicate_identity or event.source_identity in seen_identity:
                reason = reason or "duplicate_item_source_identity"
            seen_identity.add(event.source_identity)
            item_id = _int_value(event.payload.get("itemId"), nonnegative=True)
            before_id = _int_value(event.payload.get("beforeId"), nonnegative=True)
            after_id = _int_value(event.payload.get("afterId"), nonnegative=True)
            if event.event_type != "ITEM_UNDO" and (item_id is None or item_id <= 0):
                reason = reason or "item_id_unknown"
            if event.event_type == "ITEM_UNDO":
                if before_id is None or after_id is None or (before_id <= 0 and after_id <= 0):
                    reason = reason or "undo_before_after_id_unknown"
            if reason is None and pid is not None:
                ledger = state[pid]
                if event.event_type == "ITEM_PURCHASED":
                    ledger[item_id] = ledger.get(item_id, 0) + 1
                elif event.event_type in {"ITEM_SOLD", "ITEM_DESTROYED"}:
                    if ledger.get(item_id, 0) <= 0:
                        reason = "item_replay_underflow"
                    else:
                        ledger[item_id] -= 1
                        if ledger[item_id] == 0:
                            del ledger[item_id]
                elif event.event_type == "ITEM_UNDO":
                    if before_id > 0 and ledger.get(before_id, 0) <= 0:
                        reason = "item_replay_underflow"
                    else:
                        if before_id > 0:
                            ledger[before_id] -= 1
                            if ledger[before_id] == 0:
                                del ledger[before_id]
                        if after_id and after_id > 0:
                            ledger[after_id] = ledger.get(after_id, 0) + 1
            if reason is not None:
                unknown_at[int(event_index)] = reason
            snapshots.append({pid_: dict(items) for pid_, items in state.items()})
        return tuple(snapshots), unknown_at

    def _cutoff(self, timestamp_ms: Any) -> tuple[int, int]:
        query = _timestamp(timestamp_ms)
        if query is None:
            raise ValueError("query timestamp must be a nonnegative integer millisecond value")
        if query < int(self.minute_ts[0]) or query > int(self.minute_ts[-1]):
            raise ValueError("query timestamp is outside supported frame range")
        frame_index = int(np.searchsorted(self.minute_ts, query, side="right") - 1)
        event_index = int(np.searchsorted(self._event_timestamps, query, side="right"))
        return query, frame_index if frame_index >= 0 else -1

    def _held(self, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None, int | None]:
        if frame_index < 0:
            return (
                np.full((PARTICIPANTS, FRAME_WIDTH), np.nan, dtype=np.float32),
                np.full((PARTICIPANTS, 2), np.nan, dtype=np.float32),
                np.full((2,), np.nan, dtype=np.float32),
                None,
                None,
            )
        return self.node_frame[frame_index], self.xy[frame_index], self.gold_team[frame_index], int(frame_index), int(self.minute_ts[frame_index])

    def _team_growth(self, node: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
        channels: dict[str, Any] = {}
        for team, label in ((TEAM_BLUE, "blue"), (TEAM_RED, "red")):
            rows = self._frame_indices_by_team[team]
            team_values = np.asarray(node[rows], dtype=np.float64)
            values = {
                "level_mean": _aggregate(team_values[:, 0], mean=True),
                "xp_sum": _aggregate(team_values[:, 1], mean=False),
                "curGold_sum": _aggregate(team_values[:, 2], mean=False),
                "totalGold_sum": _aggregate(team_values[:, 3], mean=False),
                "laneCS_sum": _aggregate(team_values[:, 4], mean=False),
                "jgCS_sum": _aggregate(team_values[:, 5], mean=False),
            }
            channels[label] = values
        blue = channels["blue"]
        red = channels["red"]
        diff = {name: _nan_or_delta(float(blue[name]), float(red[name])) for name in blue}
        return {
            "channel_names": list(E_NAMES),
            "blue": channels["blue"], "red": channels["red"], "blue_minus_red": diff,
            "cached_gold": np.asarray(gold, dtype=np.float32),
        }

    def _history(self, records: Iterable[TypedEvent], event_limit: int, *, include_duplicates: bool = True) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for event in records:
            if event.source_index >= len(self.typed_events):
                continue
            index = self._valid_index_by_source.get(event.source_index)
            if index is None or index >= event_limit:
                continue
            if not include_duplicates and event.duplicate_identity:
                continue
            output.append(event.as_dict())
        return output

    def _kill_history(self, event_limit: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for event in self._kill_records:
            index = self._valid_index_by_source.get(event.source_index)
            if index is None or index >= event_limit:
                continue
            killer = _valid_pid(event.payload.get("killerId"))
            victim = _valid_pid(event.payload.get("victimId"))
            assists: list[int] = []
            values = event.payload.get("assistingParticipantIds")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                for value in values:
                    pid = _valid_pid(value)
                    if pid is not None and pid not in assists:
                        assists.append(pid)
            output.append(_record_dict(
                event,
                killer_id=killer,
                victim_id=victim,
                assisting_participant_ids=assists,
                killer_team=(int(self.team_ids[killer - 1]) if killer else None),
                environmental_killer=_int_value(event.payload.get("killerId")) == 0,
                position=_thaw_value(event.payload.get("position")),
            ))
        return output

    def _kill_timing(self, event_limit: int, query: int) -> dict[str, Any]:
        count = int(np.searchsorted(self._kill_event_indices, event_limit, side="left"))
        last = self._kill_timing_prefix[count]
        last_any, last_team, last_player = float(last[0]), last[1:3], last[3:].reshape(10, 3)
        age_any = float(query - last_any) / 1000.0 if math.isfinite(last_any) else float("nan")
        age_team = np.asarray([float(query - value) / 1000.0 if math.isfinite(value) else float("nan") for value in last_team], dtype=np.float64)
        age_player = np.asarray([[float(query - value) / 1000.0 if math.isfinite(value) else float("nan") for value in row] for row in last_player], dtype=np.float64)
        return {
            "last_kill_timestamp_ms": last_any,
            "age_seconds": age_any,
            "last_team_kill_timestamp_ms": last_team,
            "team_age_seconds": age_team,
            "last_player_timestamp_ms": last_player,
            "player_age_seconds": age_player,
        }

    def _objective_state(self, event_limit: int, query: int) -> dict[str, Any]:
        counts = self._objective_prefix[event_limit]
        by_type = {OBJECTIVE_TYPES[i]: {"total": int(counts[i, 0]), "blue": int(counts[i, 1]), "red": int(counts[i, 2])} for i in range(len(OBJECTIVE_TYPES))}
        histories: list[dict[str, Any]] = []
        last = np.full((len(OBJECTIVE_TYPES),), np.nan, dtype=np.float64)
        for event, team, category in self._objective_records:
            index = self._valid_index_by_source.get(event.source_index)
            if index is None or index >= event_limit:
                continue
            if category is not None and not event.duplicate_identity:
                type_index = OBJECTIVE_TYPES.index(category)
                last[type_index] = float(event.timestamp_ms)
            histories.append(_record_dict(
                event,
                objective_category=category,
                team_id=team,
                acquirer_team_id=team,
                killer_team_id=_valid_team(event.payload.get("killerTeamId")),
                owner_team_id=_valid_team(event.payload.get("teamId")),
                monster_type=event.payload.get("monsterType"),
                monster_subtype=event.payload.get("monsterSubType"),
                building_type=event.payload.get("buildingType"),
                tower_type=event.payload.get("towerType"),
                lane_type=event.payload.get("laneType"),
                position=_thaw_value(event.payload.get("position")),
            ))
        age = np.asarray([float(query - value) / 1000.0 if math.isfinite(value) else float("nan") for value in last], dtype=np.float64)
        return {
            "types": list(OBJECTIVE_TYPES), "counts": by_type,
            "last_timestamp_ms": last, "age_seconds": age,
            "unknown_category_count": len(self._unknown_objectives_before(event_limit)),
            "unknown_category_reasons": self._unknown_objectives_before(event_limit),
            "history": histories,
        }

    def _unknown_objectives_before(self, event_limit: int) -> list[str]:
        return [f"unknown_objective_category:{e.event_type}:source={e.source_index}"
                for e, _team, category in self._objective_records
                if category is None and self._valid_index_by_source[e.source_index] < event_limit]

    def _item_state(self, event_limit: int, query: int) -> dict[str, Any]:
        transactions: list[dict[str, Any]] = []
        for event in self._item_records:
            index = self._valid_index_by_source.get(event.source_index)
            if index is None or index >= event_limit:
                continue
            transactions.append(_record_dict(
                event,
                participant_id=_valid_pid(event.payload.get("participantId")),
                item_id=_int_value(event.payload.get("itemId"), nonnegative=True),
                before_id=_int_value(event.payload.get("beforeId"), nonnegative=True),
                after_id=_int_value(event.payload.get("afterId"), nonnegative=True),
                gold_gain=_int_value(event.payload.get("goldGain")),
            ))
        unknown_reason: str | None = None
        snapshot_index = -1
        for snapshot_no, event_index in enumerate(self._item_event_indices.tolist()):
            if int(event_index) >= event_limit:
                break
            snapshot_index = snapshot_no
            if int(event_index) in self._item_unknown_at:
                unknown_reason = self._item_unknown_at[int(event_index)]
                break
        if unknown_reason is not None:
            payload = np.full((PARTICIPANTS, 6, 2), np.nan, dtype=np.float64)
            empty_offsets = np.zeros((PARTICIPANTS + 1,), dtype=np.int64)
            return {
                "known": False, "valid": False, "reason": unknown_reason, "payload": payload,
                "item_ids": np.empty((0,), dtype=np.int64), "counts": np.empty((0,), dtype=np.int64),
                "offsets": empty_offsets, "transactions": transactions,
            }
        if snapshot_index < 0:
            states = {pid: {} for pid in range(1, PARTICIPANTS + 1)}
        else:
            states = self._item_snapshots[snapshot_index]
        ids: list[int] = []
        counts: list[int] = []
        offsets = [0]
        payload = np.full((PARTICIPANTS, 6, 2), np.nan, dtype=np.float64)
        for row, pid in enumerate(range(1, PARTICIPANTS + 1)):
            for slot, (item_id, count) in enumerate(sorted(states[pid].items())):
                ids.append(int(item_id)); counts.append(int(count))
                if slot < 6:
                    payload[row, slot] = (float(item_id), float(count))
            offsets.append(len(ids))
        return {
            "known": True, "valid": True, "reason": None, "payload": payload,
            "item_ids": np.asarray(ids, dtype=np.int64), "counts": np.asarray(counts, dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64), "transactions": transactions,
        }

    def query(self, timestamp_ms: Any) -> dict[str, Any]:
        query, frame_index = self._cutoff(timestamp_ms)
        event_limit = int(np.searchsorted(self._event_timestamps, query, side="right"))
        node, xy, gold, _, frame_timestamp = self._held(frame_index)
        frame_age = None if frame_timestamp is None else int(query - frame_timestamp)
        node_out = _readonly(node, np.float32)
        xy_out = _readonly(xy, np.float32)
        gold_out = _readonly(gold, np.float32)
        static_out = self.static_ids
        bans_out = self.bans
        kill_prefix = self._kill_prefix[event_limit]
        kda = {
            "kills": _readonly(kill_prefix[2:2 + PARTICIPANTS * 3:3], np.int64),
            "deaths": _readonly(kill_prefix[3:2 + PARTICIPANTS * 3:3], np.int64),
            "assists": _readonly(kill_prefix[4:2 + PARTICIPANTS * 3:3], np.int64),
            "environmental_deaths": _readonly(kill_prefix[-1:], np.int64),
        }
        kills = self._kill_history(event_limit)
        kill_timing = self._kill_timing(event_limit, query)
        objectives = self._objective_state(event_limit, query)
        items = self._item_state(event_limit, query)
        level_history = self._history(self._level_records, event_limit)
        skill_history = self._history(self._skill_records, event_limit)
        team_kills = _readonly(kill_prefix[:2], np.int64)
        team_growth = self._team_growth(node, gold)
        team_growth["cached_gold"] = _readonly(team_growth["cached_gold"], np.float32)
        histories = {
            "kills": kills,
            "objectives": objectives["history"],
            "level_ups": level_history,
            "skill_level_ups": skill_history,
            "item_transactions": items["transactions"],
        }
        audit = {
            "state_version": STATE_VERSION,
            "raw_presence_unknown": True,
            "source_event_count": len(self.typed_events),
            "valid_event_count": len(self._valid_events),
            "event_cutoff_count": event_limit,
            "event_reasons": list(self._event_reasons),
            "static_unknown_fields": list(self.static_unknown_fields),
            "same_timestamp_cutoff_inclusive": True,
            "excluded_inferred_fields": ["alive", "status", "champion_name_id", "time_norm", "gps", "role_slots", "first_kill", "person_selection", "legacy_globals"],
        }
        state = {
            "timestamp_ms": int(query), "query_timestamp_ms": int(query),
            "frame_index": int(frame_index), "frame_timestamp_ms": frame_timestamp, "frame_age_ms": frame_age,
            "node_frame": node_out, "xy": xy_out, "gold_team": gold_out,
            "static_ids": static_out, "bans": bans_out, "team_ids": self.team_ids,
            "B": {
                "elapsed_minutes": float(query) / 60000.0,
                "frame_age_seconds": float(frame_age) / 1000.0 if frame_age is not None else float("nan"),
                "team_growth": team_growth,
                "cached_gold": team_growth["cached_gold"],
                "team_kills": team_kills,
                "team_kill_history": kills,
                "team_kill_age_seconds": kill_timing["team_age_seconds"],
                "objectives": objectives,
            },
            "team_growth": team_growth, "cached_gold": team_growth["cached_gold"],
            "team_kills": team_kills, "team_kill_history": kills, "kill_timing": kill_timing,
            "kda": kda, "kills": kills,
            "objectives": objectives,
            "level_ups": level_history, "skill_level_ups": skill_history,
            "items": items, "item_transactions": items["transactions"], "item_ledger": items,
            "histories": histories, "audit": audit,
        }
        return state

    def query_numeric(self, timestamp_ms: Any) -> dict[str, Any]:
        """Return prefix summaries and references without materializing histories.

        This is the production path for support/count validation.  Rich
        source records remain available through ``query`` and
        ``iter_typed_events`` when an audit needs them.
        """
        query, frame_index = self._cutoff(timestamp_ms)
        event_limit = int(np.searchsorted(self._event_timestamps, query, side="right"))
        node, xy, gold, _, frame_timestamp = self._held(frame_index)
        frame_age = None if frame_timestamp is None else int(query - frame_timestamp)
        kill_prefix = self._kill_prefix[event_limit]
        kda = np.asarray([
            kill_prefix[2:2 + PARTICIPANTS * 3:3],
            kill_prefix[3:2 + PARTICIPANTS * 3:3],
            kill_prefix[4:2 + PARTICIPANTS * 3:3],
        ], dtype=np.int64)
        team_growth = self._team_growth(node, gold)
        growth_names = ("level_mean", "xp_sum", "curGold_sum", "totalGold_sum", "laneCS_sum", "jgCS_sum")
        growth_values = np.asarray([[team_growth[side][name] for name in growth_names] for side in ("blue", "red")], dtype=np.float64)
        growth_delta = np.asarray([_nan_or_delta(growth_values[0, i], growth_values[1, i]) for i in range(len(growth_names))], dtype=np.float64)
        item_cutoff = int(np.searchsorted(self._item_event_indices, event_limit, side="left"))
        unknown_item_events = sorted((index, reason) for index, reason in self._item_unknown_at.items() if index < event_limit)
        kill_timing = self._kill_timing(event_limit, query)
        objective_counts = np.asarray(self._objective_prefix[event_limit], dtype=np.int64)
        objective_last = np.full((len(OBJECTIVE_TYPES),), np.nan, dtype=np.float64)
        for event, _team, category in self._objective_records:
            event_index = self._valid_index_by_source.get(event.source_index)
            if event_index is None or event_index >= event_limit or event.duplicate_identity or category is None:
                continue
            objective_last[OBJECTIVE_TYPES.index(category)] = float(event.timestamp_ms)
        objective_age = np.asarray([float(query - value) / 1000.0 if math.isfinite(value) else float("nan") for value in objective_last], dtype=np.float64)
        return {
            "timestamp_ms": int(query),
            "frame_index": int(frame_index),
            "frame_timestamp_ms": frame_timestamp,
            "frame_age_ms": frame_age,
            "node_frame": _readonly(node, np.float32),
            "xy": _readonly(xy, np.float32),
            "gold_team": _readonly(gold, np.float32),
            "static_ids": self.static_ids,
            "bans": self.bans,
            "team_ids": self.team_ids,
            "event_cutoff_index": int(event_limit),
            "item_event_cutoff_index": item_cutoff,
            "team_growth_values": _readonly(growth_values, np.float64),
            "team_growth_delta": _readonly(growth_delta, np.float64),
            "cached_gold": _readonly(np.asarray(gold, dtype=np.float32), np.float32),
            "team_kills": _readonly(kill_prefix[:2], np.int64),
            "kda": _readonly(kda, np.int64),
            "environmental_deaths": int(kill_prefix[-1]),
            "kill_timing": kill_timing,
            "objective_counts": _readonly(objective_counts, np.int64),
            "objective_last_timestamp_ms": _readonly(objective_last, np.float64),
            "objective_age_seconds": _readonly(objective_age, np.float64),
            "unknown_objective_category_count": len(self._unknown_objectives_before(event_limit)),
            "item_ledger_known": not bool(unknown_item_events),
            "item_ledger_unknown_reason": unknown_item_events[0][1] if unknown_item_events else None,
            "audit": {
                "state_version": STATE_VERSION,
                "same_timestamp_cutoff_inclusive": True,
                "history_materialized": False,
                "raw_presence_unknown": True,
            },
        }

    def validate_queries(self, timestamps_ms: Sequence[Any]) -> dict[str, Any]:
        """Validate a batch of supported timestamps with compact numeric output."""
        values = np.asarray([_timestamp(value) for value in timestamps_ms], dtype=object)
        if values.ndim != 1 or any(value is None for value in values.tolist()):
            raise ValueError("timestamps_ms must be a one-dimensional sequence of nonnegative integers")
        queries = np.asarray([int(value) for value in values.tolist()], dtype=np.int64)
        rows = [self.query_numeric(int(value)) for value in queries.tolist()]
        return {
            "timestamp_ms": _readonly(queries, np.int64),
            "frame_index": _readonly(np.asarray([row["frame_index"] for row in rows], dtype=np.int64), np.int64),
            "frame_timestamp_ms": _readonly(np.asarray([row["frame_timestamp_ms"] for row in rows], dtype=np.int64), np.int64),
            "frame_age_ms": _readonly(np.asarray([row["frame_age_ms"] for row in rows], dtype=np.int64), np.int64),
            "event_cutoff_index": _readonly(np.asarray([row["event_cutoff_index"] for row in rows], dtype=np.int64), np.int64),
            "item_event_cutoff_index": _readonly(np.asarray([row["item_event_cutoff_index"] for row in rows], dtype=np.int64), np.int64),
            "team_kills": _readonly(np.asarray([row["team_kills"] for row in rows], dtype=np.int64), np.int64),
            "kda": _readonly(np.asarray([row["kda"] for row in rows], dtype=np.int64), np.int64),
            "objective_counts": _readonly(np.asarray([row["objective_counts"] for row in rows], dtype=np.int64), np.int64),
            "item_ledger_known": _readonly(np.asarray([row["item_ledger_known"] for row in rows], dtype=np.bool_), np.bool_),
        }

    def iter_typed_events(self, timestamp_ms: Any | None = None) -> Iterable[TypedEvent]:
        """Yield immutable timestamp-ordered records through an inclusive cutoff."""
        limit = len(self._valid_events) if timestamp_ms is None else int(np.searchsorted(self._event_timestamps, self._cutoff(timestamp_ms)[0], side="right"))
        return iter(self._valid_events[:limit])

    def state(self, timestamp_ms: Any) -> dict[str, Any]:
        """Compatibility alias for callers that name the operation ``state``."""
        return self.query(timestamp_ms)

    def feature_description(self) -> dict[str, Any]:
        """Return the ordered executable feature card for this state version."""
        return ordered_feature_description()

    def typed_projection(self) -> dict[str, Any]:
        """Return immutable source-oriented typed arrays and records."""
        return {
            "state_version": STATE_VERSION,
            "patch": self.patch,
            "minute_ts": self.minute_ts,
            "node_frame": self.node_frame,
            "xy": self.xy,
            "gold_team": self.gold_team,
            "team_ids": self.team_ids,
            "static_ids": self.static_ids,
            "bans": self.bans,
            "end_ms": self.end_ms,
            "winning_team": self.winning_team,
            "raw_presence_unknown": True,
            "events": self.typed_events,
            "event_payload_fields": EVENT_PAYLOAD_FIELDS,
            "event_string_fields": EVENT_STRING_FIELDS,
            "event_string_enums": EVENT_STRING_ENUMS,
            "event_bank_keys": EVENT_BANK_KEYS,
            "objective_types": OBJECTIVE_TYPES,
            "unknown_objective_categories": self._unknown_objective_categories,
            "event_reasons": self._event_reasons,
            "static_unknown_fields": self.static_unknown_fields,
        }

    def export_bank(self) -> dict[str, Any]:
        """Return compact non-object arrays for numeric shard storage.

        Raw event JSON is intentionally not copied into the bank.  The source
        triplet remains the immutable provenance record; this bank stores only
        fields consumed by the observable kernel plus validity bits.
        """
        n = len(self._valid_events)
        type_codes = {name: index for index, name in enumerate(EVENT_TYPE_NAMES)}
        category_codes = {name: index + 1 for index, name in enumerate(OBJECTIVE_TYPES)}
        field_values = np.zeros((n, len(EVENT_PAYLOAD_FIELDS)), dtype=np.int64)
        field_valid = np.zeros((n, len(EVENT_PAYLOAD_FIELDS)), dtype=np.bool_)
        assist_offsets = np.zeros((n + 1,), dtype=np.int64)
        assist_values: list[int] = []
        assist_validity = np.zeros((n,), dtype=np.bool_)
        assist_presence = np.zeros((n,), dtype=np.bool_)
        identity_codes = np.zeros((n,), dtype=np.uint64)
        identity_explicit = np.zeros((n,), dtype=np.bool_)
        identity_raw_rows: list[int] = []
        identity_raw_offsets = [0]
        identity_raw_values: list[int] = []
        category_codes_out = np.zeros((n,), dtype=np.int16)
        owner_team = np.zeros((n,), dtype=np.int16)
        owner_validity = np.zeros((n,), dtype=np.bool_)
        acquirer_team = np.zeros((n,), dtype=np.int16)
        acquirer_validity = np.zeros((n,), dtype=np.bool_)
        string_codes = np.zeros((n, len(EVENT_STRING_FIELDS)), dtype=np.int16)
        string_validity = np.zeros((n, len(EVENT_STRING_FIELDS)), dtype=np.bool_)
        unknown_string_rows: list[int] = []
        unknown_string_cols: list[int] = []
        unknown_string_offsets = [0]
        unknown_string_values: list[int] = []
        raw_rows, raw_cols, raw_values, raw_offsets = [], [], [], [0]
        position_xy = np.full((n, 2), np.nan, dtype=np.float64)
        position_validity = np.zeros((n,), dtype=np.bool_)
        position_presence = np.zeros((n,), dtype=np.bool_)
        for row, event in enumerate(self._valid_events):
            for name, value in event.payload.items():
                col = EVENT_PAYLOAD_FIELD_INDEX.get(name)
                if col is None:
                    continue
                parsed = _int_value(value)
                if parsed is not None:
                    field_values[row, col] = int(parsed)
                    field_valid[row, col] = True
                if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                    encoded = json.dumps(_thaw_value(value), ensure_ascii=False, separators=(",", ":"),
                                         default=lambda x: x.item()).encode("utf-8")
                    raw_rows.append(row)
                    raw_cols.append(col)
                    raw_values.extend(encoded)
                    raw_offsets.append(len(raw_values))
            values = event.payload.get("assistingParticipantIds")
            assist_presence[row] = "assistingParticipantIds" in event.payload
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                assist_validity[row] = all(_valid_pid(value) is not None for value in values)
                for value in values:
                    parsed = _int_value(value)
                    if parsed is not None:
                        assist_values.append(int(parsed))
                if not assist_validity[row]:
                    encoded = json.dumps(_thaw_value(values), ensure_ascii=False, separators=(",", ":"),
                                         default=lambda x: x.item()).encode("utf-8")
                    raw_rows.append(row)
                    raw_cols.append(len(EVENT_PAYLOAD_FIELDS))
                    raw_values.extend(encoded)
                    raw_offsets.append(len(raw_values))
            assist_offsets[row + 1] = len(assist_values)
            category = _objective_category(event.event_type, event.payload)
            category_codes_out[row] = category_codes.get(category, 0)
            owner = _valid_team(event.payload.get("teamId"))
            if owner is not None:
                owner_team[row] = owner
                owner_validity[row] = True
            acquirer = _event_team(event, self.team_ids)
            if acquirer is not None:
                acquirer_team[row] = acquirer
                acquirer_validity[row] = True
            for col, name in enumerate(EVENT_STRING_FIELDS):
                code, present = _enum_code(event.payload.get(name), EVENT_STRING_ENUMS[name])
                string_codes[row, col] = code
                string_validity[row, col] = present
                if present and code == 0:
                    encoded = str(event.payload.get(name)).encode("utf-8")
                    unknown_string_rows.append(row)
                    unknown_string_cols.append(col)
                    unknown_string_values.extend(encoded)
                    unknown_string_offsets.append(len(unknown_string_values))
            pair = _position_pair(event.payload.get("position"))
            position_presence[row] = "position" in event.payload
            if pair is not None:
                position_xy[row] = pair
                position_validity[row] = True
            if event.source_identity.startswith("ordinal:"):
                # Source ordinals already form the stable default identity;
                # hashing them wastes work and loses their direct provenance.
                identity_codes[row] = np.uint64(event.source_index)
            else:
                digest = hashlib.sha256(event.source_identity.encode("utf-8")).digest()
                identity_codes[row] = int.from_bytes(digest[:8], "big", signed=False)
                identity_explicit[row] = True
                identity_raw_rows.append(row)
                identity_raw_values.extend(event.source_identity.encode("utf-8"))
                identity_raw_offsets.append(len(identity_raw_values))
        return {
            "minute_ts": self.minute_ts,
            "node_frame": self.node_frame,
            "xy": self.xy,
            "gold_team": self.gold_team,
            "team_ids": self.team_ids,
            "static_ids": self.static_ids,
            "bans": self.bans,
            "end_ms": np.asarray(-1 if self.end_ms is None else int(self.end_ms), dtype=np.int64),
            "winning_team": np.asarray(0 if self.winning_team is None else int(self.winning_team), dtype=np.int16),
            "event_timestamp": _readonly(_compact_int(self._event_timestamps)),
            "event_source_index": _readonly(_compact_int([event.source_index for event in self._valid_events])),
            "event_identity_code": _readonly(identity_codes, np.uint64),
            "event_identity_explicit": _readonly(identity_explicit, np.bool_),
            "event_identity_raw_rows": _readonly(_compact_int(identity_raw_rows)),
            "event_identity_raw_offsets": _readonly(_compact_int(identity_raw_offsets)),
            "event_identity_raw_values": _readonly(np.asarray(identity_raw_values, dtype=np.uint8), np.uint8),
            "event_type_code": _readonly(np.asarray([type_codes.get(event.event_type, type_codes["UNKNOWN"]) for event in self._valid_events], dtype=np.int16), np.int16),
            "event_category_code": _readonly(category_codes_out, np.int16),
            "event_field_values": _readonly(_compact_int(field_values)),
            "event_field_validity": _readonly(field_valid, np.bool_),
            "event_owner_team": _readonly(owner_team, np.int16),
            "event_owner_validity": _readonly(owner_validity, np.bool_),
            "event_acquirer_team": _readonly(acquirer_team, np.int16),
            "event_acquirer_validity": _readonly(acquirer_validity, np.bool_),
            "event_assist_offsets": _readonly(_compact_int(assist_offsets)),
            "event_assist_values": _readonly(_compact_int(assist_values)),
            "event_assist_validity": _readonly(assist_validity, np.bool_),
            "event_assist_presence": _readonly(assist_presence, np.bool_),
            "event_position_xy": _readonly(position_xy, np.float64),
            "event_position_validity": _readonly(position_validity, np.bool_),
            "event_position_presence": _readonly(position_presence, np.bool_),
            "event_string_codes": _readonly(string_codes, np.int16),
            "event_string_validity": _readonly(string_validity, np.bool_),
            "event_string_unknown_rows": _readonly(_compact_int(unknown_string_rows)),
            "event_string_unknown_cols": _readonly(_compact_int(unknown_string_cols)),
            "event_string_unknown_offsets": _readonly(_compact_int(unknown_string_offsets)),
            "event_string_unknown_values": _readonly(np.asarray(unknown_string_values, dtype=np.uint8), np.uint8),
            "event_raw_rows": _readonly(_compact_int(raw_rows)),
            "event_raw_cols": _readonly(_compact_int(raw_cols)),
            "event_raw_offsets": _readonly(_compact_int(raw_offsets)),
            "event_raw_values": _readonly(np.asarray(raw_values, dtype=np.uint8), np.uint8),
            "item_event_indices": _readonly(_compact_int(self._item_event_indices)),
        }

    @classmethod
    def from_bank(cls, bank: Mapping[str, Any], patch: str | None = None) -> "ObservableMatch":
        """Reopen a compact bank written by :meth:`export_bank`.

        The bank intentionally contains valid timestamped records only; source
        preflight rejects malformed clocks before storage.  Event payloads are
        decoded from their stable JSON projection and retain original source
        ordinals for query identity checks.
        """
        if not isinstance(bank, Mapping):
            raise TypeError("bank must be a mapping")
        timestamps = np.asarray(bank.get("event_timestamp"))
        source_indices = np.asarray(bank.get("event_source_index"))
        type_codes = np.asarray(bank.get("event_type_code"))
        identity_codes = np.asarray(bank.get("event_identity_code", np.zeros_like(timestamps, dtype=np.uint64)))
        identity_explicit = np.asarray(bank.get("event_identity_explicit", np.zeros_like(timestamps, dtype=np.bool_)))
        identity_raw_rows = np.asarray(bank.get("event_identity_raw_rows", np.empty((0,), dtype=np.int32)))
        identity_raw_offsets = np.asarray(bank.get("event_identity_raw_offsets", np.zeros((identity_raw_rows.size + 1,), dtype=np.int64)))
        identity_raw_values = np.asarray(bank.get("event_identity_raw_values", np.empty((0,), dtype=np.uint8)))
        category_codes = np.asarray(bank.get("event_category_code", np.zeros_like(timestamps, dtype=np.int16)))
        field_values = np.asarray(bank.get("event_field_values"))
        field_valid = np.asarray(bank.get("event_field_validity"))
        assist_offsets = np.asarray(bank.get("event_assist_offsets", np.zeros((timestamps.size + 1,), dtype=np.int64)))
        assist_values = np.asarray(bank.get("event_assist_values", np.empty((0,), dtype=np.int64)))
        assist_validity = np.asarray(bank.get("event_assist_validity", np.zeros((timestamps.size,), dtype=np.bool_)))
        assist_presence = np.asarray(bank.get("event_assist_presence", assist_validity.copy()))
        position_xy = np.asarray(bank.get("event_position_xy", np.full((timestamps.size, 2), np.nan, dtype=np.float32)))
        position_validity = np.asarray(bank.get("event_position_validity", np.zeros((timestamps.size,), dtype=np.bool_)))
        position_presence = np.asarray(bank.get("event_position_presence", position_validity.copy()))
        string_codes = np.asarray(bank.get("event_string_codes", np.zeros((timestamps.size, len(EVENT_STRING_FIELDS)), dtype=np.int16)))
        string_validity = np.asarray(bank.get("event_string_validity", np.zeros((timestamps.size, len(EVENT_STRING_FIELDS)), dtype=np.bool_)))
        unknown_string_rows = np.asarray(bank.get("event_string_unknown_rows", np.empty((0,), dtype=np.int32)))
        unknown_string_cols = np.asarray(bank.get("event_string_unknown_cols", np.empty((0,), dtype=np.int16)))
        unknown_string_offsets = np.asarray(bank.get("event_string_unknown_offsets", np.zeros((unknown_string_rows.size + 1,), dtype=np.int64)))
        unknown_string_values = np.asarray(bank.get("event_string_unknown_values", np.empty((0,), dtype=np.uint8)))
        if (timestamps.ndim != 1 or source_indices.shape != timestamps.shape or type_codes.shape != timestamps.shape
                or identity_codes.shape != timestamps.shape or identity_explicit.shape != timestamps.shape
                or category_codes.shape != timestamps.shape
                or identity_raw_offsets.shape != (identity_raw_rows.size + 1,)
                or identity_raw_values.ndim != 1
                or field_values.shape != (timestamps.size, len(EVENT_PAYLOAD_FIELDS))
                or field_valid.shape != field_values.shape or assist_offsets.shape != (timestamps.size + 1,)
                or assist_validity.shape != (timestamps.size,) or assist_presence.shape != (timestamps.size,)
                or position_xy.shape != (timestamps.size, 2) or position_validity.shape != (timestamps.size,)
                or position_presence.shape != (timestamps.size,)
                or string_codes.shape != (timestamps.size, len(EVENT_STRING_FIELDS))
                or string_validity.shape != string_codes.shape
                or unknown_string_cols.shape != unknown_string_rows.shape
                or unknown_string_offsets.shape != (unknown_string_rows.size + 1,)
                or unknown_string_values.ndim != 1):
            raise ValueError("event bank arrays have inconsistent lengths")
        identity_raw: dict[int, str] = {}
        for identity_no, row_value in enumerate(identity_raw_rows.tolist()):
            row = int(row_value)
            if row < 0 or row >= timestamps.size:
                raise ValueError("event bank identity rows are invalid")
            begin, end = int(identity_raw_offsets[identity_no]), int(identity_raw_offsets[identity_no + 1])
            if begin < 0 or end < begin or end > identity_raw_values.size:
                raise ValueError("event bank identity offsets are invalid")
            try:
                identity_raw[row] = bytes(identity_raw_values[begin:end].tolist()).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("event bank identity bytes are invalid UTF-8") from exc
        unknown_strings: dict[tuple[int, int], str] = {}
        for unknown_no, (row, col) in enumerate(zip(unknown_string_rows.tolist(), unknown_string_cols.tolist())):
            row, col = int(row), int(col)
            if row < 0 or row >= timestamps.size or col < 0 or col >= len(EVENT_STRING_FIELDS):
                raise ValueError("event bank unknown string coordinates are invalid")
            begin, end = int(unknown_string_offsets[unknown_no]), int(unknown_string_offsets[unknown_no + 1])
            if begin < 0 or end < begin or end > unknown_string_values.size:
                raise ValueError("event bank unknown string offsets are invalid")
            try:
                unknown_strings[(row, col)] = bytes(unknown_string_values[begin:end].tolist()).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("event bank unknown string bytes are invalid UTF-8") from exc
        raw_payloads = {}
        raw_rows = np.asarray(bank.get("event_raw_rows", []))
        raw_cols = np.asarray(bank.get("event_raw_cols", []))
        raw_offsets = np.asarray(bank.get("event_raw_offsets", [0]))
        raw_values = np.asarray(bank.get("event_raw_values", []))
        if len(raw_rows) != len(raw_cols) or len(raw_offsets) != len(raw_rows) + 1:
            raise ValueError("event raw payload lengths disagree")
        for i, (row, col) in enumerate(zip(raw_rows, raw_cols)):
            a, b = int(raw_offsets[i]), int(raw_offsets[i + 1])
            if not 0 <= row < len(timestamps) or not 0 <= col <= len(EVENT_PAYLOAD_FIELDS) or not 0 <= a <= b <= len(raw_values):
                raise ValueError("event raw payload index is invalid")
            names = (*EVENT_PAYLOAD_FIELDS, "assistingParticipantIds")
            raw_payloads.setdefault(int(row), {})[names[int(col)]] = json.loads(bytes(raw_values[a:b].tolist()).decode("utf-8"))
        events: list[dict[str, Any]] = []
        for event_no, (timestamp, source_index, type_code) in enumerate(zip(timestamps.tolist(), source_indices.tolist(), type_codes.tolist())):
            record: dict[str, Any] = {}
            for col, name in enumerate(EVENT_PAYLOAD_FIELDS):
                if bool(field_valid[event_no, col]):
                    record[name] = int(field_values[event_no, col])
            begin, end = int(assist_offsets[event_no]), int(assist_offsets[event_no + 1])
            if begin < 0 or end < begin or end > assist_values.size:
                raise ValueError("event assist offsets are invalid")
            if bool(assist_validity[event_no]):
                record["assistingParticipantIds"] = [int(value) for value in assist_values[begin:end].tolist()]
            elif bool(assist_presence[event_no]):
                record["assistingParticipantIds"] = None
            record.update(raw_payloads.get(event_no, {}))
            record["type"] = EVENT_TYPE_NAMES[int(type_code)] if 0 <= int(type_code) < len(EVENT_TYPE_NAMES) else "UNKNOWN"
            record["timestamp"] = int(timestamp)
            for col, name in enumerate(EVENT_STRING_FIELDS):
                if bool(string_validity[event_no, col]):
                    value = _enum_value(string_codes[event_no, col], EVENT_STRING_ENUMS[name])
                    if value is None:
                        value = unknown_strings.get((event_no, col))
                    if value is not None:
                        record[name] = value
            if bool(position_validity[event_no]):
                x, y = position_xy[event_no].tolist()
                if math.isfinite(float(x)) and math.isfinite(float(y)):
                    record["position"] = {"x": float(x), "y": float(y)}
            elif bool(position_presence[event_no]):
                record["position"] = {}
            if bool(identity_explicit[event_no]):
                record["source_id"] = identity_raw.get(event_no, f"code:{int(identity_codes[event_no])}")
            record["_source_ordinal"] = int(source_index)
            events.append(record)
        team_ids = np.asarray(bank.get("team_ids"))
        meta = {
            "team_ids": team_ids,
            "end_ms": int(np.asarray(bank.get("end_ms", -1)).item()) if int(np.asarray(bank.get("end_ms", -1)).item()) >= 0 else None,
            "winning_team": int(np.asarray(bank.get("winning_team", 0)).item()) if int(np.asarray(bank.get("winning_team", 0)).item()) in (TEAM_BLUE, TEAM_RED) else None,
            "_verified_end_from_bank": True,
        }
        arrays = {
            "minute_ts": bank["minute_ts"], "node_frame": bank["node_frame"], "xy": bank["xy"],
            "gold_team": bank["gold_team"], "team_ids": team_ids,
            "static_ids": bank["static_ids"], "bans": bank["bans"],
            "end_ms": meta["end_ms"], "winning_team": meta["winning_team"],
        }
        return cls(arrays, meta, events, patch or str(bank.get("patch", "unknown")) or "unknown")


__all__ = [
    "ObservableMatch", "TypedEvent", "STATE_VERSION", "history_feature_description",
    "ordered_feature_description", "STATIC_NAMES", "OBJECTIVE_TYPES", "ITEM_TYPES",
    "OBJECTIVE_SOURCE_TYPES", "EVENT_TYPE_NAMES", "EVENT_PAYLOAD_FIELDS", "EVENT_PAYLOAD_FIELD_INDEX",
    "EVENT_STRING_FIELDS", "EVENT_STRING_ENUMS", "EVENT_BANK_KEYS",
    "MONSTER_TYPE_NAMES", "MONSTER_SUBTYPE_NAMES", "BUILDING_TYPE_NAMES",
    "TOWER_TYPE_NAMES", "LANE_TYPE_NAMES", "LEVEL_UP_TYPE_NAMES",
    "DRAGON_SOUL_NAMES", "SOUL_TYPE_NAMES",
]
