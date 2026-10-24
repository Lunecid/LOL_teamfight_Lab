"""Deterministic common non-item event history counters for V input.

This module consumes the immutable record shape emitted by
``v_input_projection_20261002.SemanticHistory``.  It intentionally keeps the
scope narrow: the seven non-item event types below are counted, while item
transactions remain owned by the accepted item-history v3 path.

The accumulator has no model vocabulary or fitting state.  A counter key is a
compact JSON tuple ``[event_type, field_role, measure, typed_value]``.  The
``typed_value`` component is itself ``[type_name, value]`` so that, for
example, integer ``1`` and string ``"1"`` cannot collide.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from functools import lru_cache
import json
import math
from numbers import Integral, Real
from typing import Any, Final, TypeAlias


VERSION: Final = "v_model_history_p2a_20261003"
PARTICIPANTS: Final = 10
TEAM_IDS: Final = (100, 200)

SUPPORTED_EVENT_TYPES: Final = (
    "CHAMPION_KILL",
    "ELITE_MONSTER_KILL",
    "BUILDING_KILL",
    "TURRET_PLATE_DESTROYED",
    "DRAGON_SOUL_GIVEN",
    "LEVEL_UP",
    "SKILL_LEVEL_UP",
)
SUPPORTED_EVENT_SET: Final = frozenset(SUPPORTED_EVENT_TYPES)
ITEM_EVENT_TYPES: Final = frozenset(
    {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"}
)

ADD_ACCEPTED: Final = "accepted"
ADD_SKIPPED_ITEM: Final = "skipped_item"
ADD_SKIPPED_UNSUPPORTED: Final = "skipped_unsupported"

_GLOBAL_LANE: Final[tuple[str, int]] = ("global", -1)
_NODE_LANE_KIND: Final[str] = "node"
_TEAM_LANE_KIND: Final[str] = "team"
LaneIndex: TypeAlias = tuple[str, int]
Delta: TypeAlias = tuple[LaneIndex, str, int]

_ID_FIELDS: Final = frozenset(
    {"participantId", "killerId", "victimId", "assistingParticipantIds"}
)
_TEAM_FIELDS: Final = frozenset({"teamId", "killerTeamId"})
_ASSIST_FIELD: Final[str] = "assistingParticipantIds"

_ALLOWED_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "CHAMPION_KILL": (
        "killerId",
        "victimId",
        "killerTeamId",
        "teamId",
        _ASSIST_FIELD,
    ),
    "ELITE_MONSTER_KILL": (
        "killerId",
        "killerTeamId",
        "teamId",
        "monsterType",
        "monsterSubType",
        "buildingType",
        "towerType",
        "laneType",
        "dragonSoul",
        "soulType",
        "name",
    ),
    "BUILDING_KILL": (
        "killerId",
        "killerTeamId",
        "teamId",
        "monsterType",
        "monsterSubType",
        "buildingType",
        "towerType",
        "laneType",
        "dragonSoul",
        "soulType",
        "name",
    ),
    "TURRET_PLATE_DESTROYED": (
        "killerId",
        "killerTeamId",
        "teamId",
        "monsterType",
        "monsterSubType",
        "buildingType",
        "towerType",
        "laneType",
        "dragonSoul",
        "soulType",
        "name",
    ),
    "DRAGON_SOUL_GIVEN": (
        "killerId",
        "killerTeamId",
        "teamId",
        "monsterType",
        "monsterSubType",
        "buildingType",
        "towerType",
        "laneType",
        "dragonSoul",
        "soulType",
        "name",
    ),
    "LEVEL_UP": ("participantId", "level", "levelUpType"),
    "SKILL_LEVEL_UP": ("participantId", "level", "skillSlot", "levelUpType"),
}

_STRING_FIELDS: Final = frozenset(
    {
        "monsterType",
        "monsterSubType",
        "buildingType",
        "towerType",
        "laneType",
        "levelUpType",
        "dragonSoul",
        "soulType",
        "name",
        "objective_category",
    }
)
_NUMERIC_FIELDS: Final = frozenset(
    {
        "participantId",
        "killerId",
        "victimId",
        "killerTeamId",
        "teamId",
        "level",
        "skillSlot",
    }
)


class _FrozenDict(dict[str, Any]):
    """A JSON-compatible immutable dictionary.

    ``MappingProxyType`` is immutable but is not accepted by the standard
    JSON encoder.  A frozen ``dict`` subclass keeps the snapshot both
    serializable and safe to mutate through the returned object.
    """

    __slots__ = ()

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("snapshot is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __ior__(self, other: Any) -> "_FrozenDict":
        self._immutable(other)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        # Consumers such as the focal-view swapper deliberately make a
        # mutable working copy before relabelling lanes.
        copied = {key: _deepcopy(child, memo) for key, child in self.items()}
        memo[id(self)] = copied
        return copied


class _FrozenList(list[Any]):
    """A JSON-compatible immutable list."""

    __slots__ = ()

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("snapshot is immutable")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = clear = extend = insert = pop = remove = reverse = sort = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        copied = [_deepcopy(child, memo) for child in self]
        memo[id(self)] = copied
        return copied


def _deepcopy(value: Any, memo: dict[int, Any]) -> Any:
    """Small recursive deepcopy used without importing a model/runtime stack."""

    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value.__deepcopy__(memo)
    if isinstance(value, dict):
        copied = {key: _deepcopy(child, memo) for key, child in value.items()}
        memo[id(value)] = copied
        return copied
    if isinstance(value, list):
        copied = [_deepcopy(child, memo) for child in value]
        memo[id(value)] = copied
        return copied
    return value


def _typed_value(value: Any) -> list[Any] | None:
    """Return a JSON-safe type-tagged scalar, or ``None`` if it is unsupported."""

    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", bool(value)]
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return ["float", number]
    if isinstance(value, str):
        return ["str", value]
    return None


def encode_token(event_type: str, field_role: str, measure: str, value: Any) -> str:
    """Encode one canonical counter token.

    ``value`` is tagged before compact JSON encoding.  The function rejects
    non-scalar values because raw IDs, lists, mappings, and source metadata
    are never model tokens.
    """

    typed = _typed_value(value)
    if typed is None:
        raise ValueError("counter token value must be a finite JSON scalar")
    return _encode_typed_token(str(event_type), str(field_role), str(measure), typed)


@lru_cache(maxsize=65_536)
def _encode_cached(
    event_type: str, field_role: str, measure: str, type_name: str, value: Any
) -> str:
    return json.dumps(
        [event_type, field_role, measure, [type_name, value]],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _encode_typed_token(
    event_type: str, field_role: str, measure: str, typed: list[Any]
) -> str:
    """Encode a value already validated and type-tagged."""
    if len(typed) != 2 or not isinstance(typed[0], str):
        raise ValueError("typed counter value must be a [type_name, value] pair")
    return _encode_cached(event_type, field_role, measure, typed[0], typed[1])


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _flag(value: Any) -> bool:
    """Read a P1 boolean without Python's truthiness promotion of strings."""

    if isinstance(value, bool):
        return value
    if _is_integer(value) and int(value) in (0, 1):
        return bool(value)
    return False


def _timestamp(record: Mapping[str, Any]) -> int:
    value = record.get("timestamp_ms")
    if not _is_integer(value) or int(value) < 0:
        raise ValueError("record timestamp_ms must be a nonnegative integer")
    return int(value)


def _event_type(record: Mapping[str, Any]) -> str:
    value = record.get("event_type")
    if not isinstance(value, str) or not value:
        raise ValueError("record event_type must be a nonempty string")
    return value


def _maps(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    payload = record.get("payload")
    presence = record.get("field_presence")
    validity = record.get("field_validity")
    if not isinstance(payload, Mapping) or not isinstance(presence, Mapping) or not isinstance(validity, Mapping):
        raise ValueError("P1 history records require payload, field_presence, and field_validity mappings")
    return payload, presence, validity


def _field_state(
    field: str,
    payload: Mapping[str, Any],
    presence: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> str:
    """Map P1 presence/validity to one stable status token value."""

    present = _flag(presence.get(field, False))
    valid = _flag(validity.get(field, False))
    if not present:
        return "absent"
    if not valid or field not in payload:
        return "malformed"
    if field == _ASSIST_FIELD:
        value = payload.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return "malformed"
        # A present, valid empty list is retained separately from missing and
        # malformed assists; repeated members remain repeated credits below.
        return "valid_empty" if len(value) == 0 else "valid"
    # Source validity and routing validity are separate.  A source-valid
    # integer outside the participant/team range remains a valid unassigned
    # diagnostic, while a bool/float/string cannot be treated as an ID.
    if field in _ID_FIELDS or field in _TEAM_FIELDS or field in _NUMERIC_FIELDS:
        if not _is_integer(payload[field]):
            return "malformed"
    elif field in _STRING_FIELDS and not isinstance(payload[field], str):
        return "malformed"
    return "valid"


def _valid_pid(value: Any) -> int | None:
    return int(value) if _is_integer(value) and 1 <= int(value) <= PARTICIPANTS else None


def _valid_team(value: Any) -> int | None:
    return int(value) if _is_integer(value) and int(value) in TEAM_IDS else None


def _raw_valid_id(
    field: str,
    payload: Mapping[str, Any],
    presence: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> tuple[Any, bool]:
    """Return raw value and source-valid flag without treating range as validity."""

    present = _flag(presence.get(field, False))
    source_valid = _flag(validity.get(field, False))
    return payload.get(field), bool(present and source_valid and field in payload)


def _primary_lane(event_type: str, payload: Mapping[str, Any], presence: Mapping[str, Any], validity: Mapping[str, Any]) -> LaneIndex:
    if event_type not in {"LEVEL_UP", "SKILL_LEVEL_UP"}:
        return _GLOBAL_LANE
    raw, source_valid = _raw_valid_id("participantId", payload, presence, validity)
    pid = _valid_pid(raw) if source_valid else None
    return (_NODE_LANE_KIND, pid - 1) if pid is not None else _GLOBAL_LANE


def _delta_builder() -> dict[LaneIndex, dict[str, int]]:
    return {}


def _add_token(
    lanes: dict[LaneIndex, dict[str, int]], lane: LaneIndex, token: str, count: int = 1
) -> None:
    if count < 0:
        raise ValueError("counter increments must be nonnegative")
    if count == 0:
        return
    lanes.setdefault(lane, {})[token] = lanes.setdefault(lane, {}).get(token, 0) + int(count)


def _status_token(event_type: str, field: str, state: str) -> str:
    return _encode_typed_token(event_type, field, "status", ["str", state])


def _source_status_token(
    event_type: str,
    field: str,
    presence: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> str:
    """Encode the raw P1 presence/validity pair without payload inspection."""

    present = _flag(presence.get(field, False))
    valid = _flag(validity.get(field, False))
    state = (
        "present_valid" if present and valid
        else "present_invalid" if present
        else "absent_valid" if valid
        else "absent_invalid"
    )
    return _encode_typed_token(event_type, field, "source_status", ["str", state])


def _value_token(event_type: str, field: str, value: Any) -> str | None:
    typed = _typed_value(value)
    if typed is None:
        return None
    return _encode_typed_token(event_type, field, "value", typed)


def _event_token(event_type: str) -> str:
    return _encode_typed_token(event_type, "event", "count", ["int", 1])


def _credit_token(event_type: str, role: str) -> str:
    return _encode_typed_token(event_type, role, "credit", ["int", 1])


def _team_event_token(event_type: str, field: str) -> str:
    return _encode_typed_token(event_type, field, "event", ["int", 1])


def _team_category_token(event_type: str, field: str, category: Any) -> str | None:
    typed = _typed_value(category)
    if typed is None:
        return None
    return _encode_typed_token(event_type, field, "category", typed)


def _assist_values(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return tuple(value)


def _non_id_value_is_valid(field: str, value: Any) -> bool:
    if field in _STRING_FIELDS:
        return isinstance(value, str)
    if field in _NUMERIC_FIELDS:
        return _is_integer(value)
    return False


def _event_deltas(record: Mapping[str, Any]) -> tuple[str, tuple[Delta, ...]]:
    """Validate one P1 record and return ``(status, aggregated_deltas)``."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    _timestamp(record)  # timestamp is a chronological guard, never a token
    event_type = _event_type(record)
    payload, presence, validity = _maps(record)
    if event_type in ITEM_EVENT_TYPES:
        return ADD_SKIPPED_ITEM, ()
    if event_type not in SUPPORTED_EVENT_SET:
        return ADD_SKIPPED_UNSUPPORTED, ()

    allowed = _ALLOWED_FIELDS[event_type]
    primary = _primary_lane(event_type, payload, presence, validity)
    lanes = _delta_builder()

    # Exactly one event count is emitted at the primary lane.  For kills and
    # objectives that lane is global; assigned level/skill records are routed
    # to their participant node, while unassigned records remain global.
    _add_token(lanes, primary, _event_token(event_type))

    # P1 field status is a three-/four-state diagnostic.  IDs are represented
    # only by this status and optional assignment status, never by raw IDs.
    for field in allowed:
        state = _field_state(field, payload, presence, validity)
        _add_token(lanes, primary, _status_token(event_type, field, state))
        _add_token(lanes, primary, _source_status_token(event_type, field, presence, validity))

    # Every valid non-ID allowlisted field is a typed value at the primary
    # lane.  Invalid values contribute only the status above.
    for field in allowed:
        if field in _ID_FIELDS or field in _TEAM_FIELDS:
            continue
        if field not in payload:
            continue
        state = _field_state(field, payload, presence, validity)
        if state != "valid":
            continue
        if not _non_id_value_is_valid(field, payload[field]):
            continue
        token = _value_token(event_type, field, payload[field])
        if token is not None:
            _add_token(lanes, primary, token)

    # Objective category is an allowlisted derived P1 value outside payload.
    # It is counted only when explicitly supplied by the P1 record; no subtype
    # or owner inference is performed here.
    if event_type in {"ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "DRAGON_SOUL_GIVEN"}:
        category = record.get("objective_category")
        if isinstance(category, str):
            token = _value_token(event_type, "objective_category", category)
            if token is not None:
                _add_token(lanes, primary, token)

    # Participant credits are a separate node-local D/E/R path.  A source
    # valid but unassigned ID (notably 0) remains a global diagnostic and does
    # not manufacture a participant node.
    if event_type in {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "DRAGON_SOUL_GIVEN"}:
        for field, role in (("killerId", "killer"), ("victimId", "victim")):
            if field not in allowed:
                continue
            raw, source_valid = _raw_valid_id(field, payload, presence, validity)
            pid = _valid_pid(raw) if source_valid else None
            if pid is not None:
                _add_token(lanes, (_NODE_LANE_KIND, pid - 1), _credit_token(event_type, role))
        if event_type == "CHAMPION_KILL":
            assist_raw, assist_source_valid = _raw_valid_id(_ASSIST_FIELD, payload, presence, validity)
            assist_state = _field_state(_ASSIST_FIELD, payload, presence, validity)
            assists = _assist_values(assist_raw) if assist_source_valid else None
            if assists is not None and assist_state in {"valid", "valid_empty"}:
                for raw in assists:
                    pid = _valid_pid(raw)
                    if pid is not None:
                        _add_token(
                            lanes,
                            (_NODE_LANE_KIND, pid - 1),
                            _credit_token(event_type, "assist"),
                        )

    # Explicit team fields are credited only to their recorded lane.  In
    # particular, BUILDING_KILL teamId=100 remains Blue even when killerId=0;
    # no destroyed-owner inversion or other owner inference is applied.
    for field in _TEAM_FIELDS.intersection(allowed):
        raw, source_valid = _raw_valid_id(field, payload, presence, validity)
        team = _valid_team(raw) if source_valid else None
        if team is None:
            continue
        lane = (_TEAM_LANE_KIND, 0 if team == TEAM_IDS[0] else 1)
        _add_token(lanes, lane, _team_event_token(event_type, field))
        category = record.get("objective_category")
        if isinstance(category, str):
            token = _team_category_token(event_type, field, category)
            if token is not None:
                _add_token(lanes, lane, token)

    # Assignment status is intentionally metadata-like and contains no raw
    # participant number.  It is useful for distinguishing valid participant
    # ownership from source-valid unassigned/invalid IDs.
    if event_type in {"LEVEL_UP", "SKILL_LEVEL_UP"}:
        raw, source_valid = _raw_valid_id("participantId", payload, presence, validity)
        state = _field_state("participantId", payload, presence, validity)
        if state in {"valid", "valid_empty"} and source_valid:
            assignment = "assigned" if _valid_pid(raw) is not None else "unassigned"
            _add_token(
                lanes,
                primary,
                _encode_typed_token(event_type, "participantId", "assignment", ["str", assignment]),
            )

    flattened: list[Delta] = []
    for lane, counter in lanes.items():
        for token, count in counter.items():
            flattened.append((lane, token, int(count)))
    flattened.sort(key=lambda item: (0 if item[0][0] == "global" else 1 if item[0][0] == "team" else 2, item[0][1], item[1]))
    return ADD_ACCEPTED, tuple(flattened)


def event_delta(record: Mapping[str, Any]) -> tuple[Delta, ...]:
    """Return the aggregated lane/token/count contribution of one record.

    Item and unsupported records return an empty tuple.  ``HistoryAccumulator``
    exposes the explicit skip status through :meth:`add`.
    """

    _status, deltas = _event_deltas(record)
    return deltas


def _empty_counters() -> tuple[dict[str, int], list[dict[str, int]], list[dict[str, int]]]:
    return {}, [dict() for _ in range(PARTICIPANTS)], [dict() for _ in range(2)]


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _frozen(child) for key, child in value.items()})
    if isinstance(value, list):
        return _FrozenList([_frozen(child) for child in value])
    return value


def _canonical_snapshot(
    global_counter: Mapping[str, int],
    node_counters: Sequence[Mapping[str, int]],
    team_counters: Sequence[Mapping[str, int]],
) -> Mapping[str, Any]:
    def ordered(counter: Mapping[str, int]) -> dict[str, int]:
        return {str(token): int(counter[token]) for token in sorted(counter)}

    result = {
        "version": VERSION,
        "node": [ordered(counter) for counter in node_counters],
        "team": [ordered(counter) for counter in team_counters],
        "global": ordered(global_counter),
    }
    return _frozen(result)


class HistoryAccumulator:
    """Chronological common non-item history counter accumulator."""

    def __init__(self) -> None:
        self._global, self._node, self._team = _empty_counters()
        self._timestamps: list[int] = []
        self._deltas: list[tuple[Delta, ...]] = []
        self._last_timestamp: int | None = None
        self._last_query_ms: int | None = None
        self._query_cursor = 0
        self._query_global, self._query_node, self._query_team = _empty_counters()

    @property
    def last_timestamp_ms(self) -> int | None:
        return self._last_timestamp

    def add(self, record: Mapping[str, Any]) -> str:
        """Add one source record, enforcing nondecreasing integer timestamps."""

        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        if self._last_query_ms is not None:
            raise ValueError("cannot add records after snapshot_at has started")
        timestamp = _timestamp(record)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("history records must be added in nondecreasing timestamp order")
        status, deltas = _event_deltas(record)
        self._last_timestamp = timestamp
        self._timestamps.append(timestamp)
        self._deltas.append(deltas)
        for lane, token, count in deltas:
            self._apply(self._global, self._node, self._team, lane, token, count)
        return status

    @staticmethod
    def _apply(
        global_counter: dict[str, int],
        node_counters: list[dict[str, int]],
        team_counters: list[dict[str, int]],
        lane: LaneIndex,
        token: str,
        count: int,
    ) -> None:
        if count < 0:
            raise ValueError("counter increments must be nonnegative")
        kind, index = lane
        if kind == "global":
            target = global_counter
        elif kind == _NODE_LANE_KIND and 0 <= index < PARTICIPANTS:
            target = node_counters[index]
        elif kind == _TEAM_LANE_KIND and 0 <= index < 2:
            target = team_counters[index]
        else:
            raise ValueError(f"invalid history lane: {lane!r}")
        target[token] = target.get(token, 0) + int(count)

    def snapshot(self) -> Mapping[str, Any]:
        """Return the immutable full-prefix snapshot."""

        return _canonical_snapshot(self._global, self._node, self._team)

    def snapshot_at(self, query_ms: Any) -> Mapping[str, Any]:
        """Return an inclusive snapshot at a monotonic, already-seen cutoff."""

        if not _is_integer(query_ms) or int(query_ms) < 0:
            raise ValueError("query_ms must be a nonnegative integer")
        query = int(query_ms)
        if self._last_query_ms is not None and query < self._last_query_ms:
            raise ValueError("snapshot_at query must be nondecreasing")
        target = bisect_right(self._timestamps, query)
        while self._query_cursor < target:
            for lane, token, count in self._deltas[self._query_cursor]:
                self._apply(self._query_global, self._query_node, self._query_team, lane, token, count)
            self._query_cursor += 1
        self._last_query_ms = query
        return _canonical_snapshot(self._query_global, self._query_node, self._query_team)


__all__ = [
    "ADD_ACCEPTED",
    "ADD_SKIPPED_ITEM",
    "ADD_SKIPPED_UNSUPPORTED",
    "HistoryAccumulator",
    "ITEM_EVENT_TYPES",
    "SUPPORTED_EVENT_TYPES",
    "VERSION",
    "encode_token",
    "event_delta",
]
