"""Lossless, cutoff-safe projection of the approved primary V inputs."""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np

from studies.observable_state_20261002.kernel import (
    EVENT_PAYLOAD_FIELDS,
    EVENT_STRING_ENUMS,
    EVENT_STRING_FIELDS,
    EVENT_TYPE_NAMES,
    ITEM_TYPES,
    OBJECTIVE_TYPES,
    TEAM_BLUE,
    TEAM_RED,
)

from .schema import (
    BANS_SHAPE,
    FRAME_WIDTH,
    HISTORY_ALLOWED_EVENT_TYPES,
    HISTORY_ASSIST_FIELD,
    HISTORY_NUMERIC_FIELDS,
    HISTORY_PARTITIONS,
    HISTORY_STRING_FIELDS,
    PARTICIPANTS,
    QUERY_BLOCK_SPECS,
    STATIC_BLOCK_SPECS,
)


_NUMERIC_INDEX = {name: index for index, name in enumerate(EVENT_PAYLOAD_FIELDS)}
_STRING_INDEX = {name: index for index, name in enumerate(EVENT_STRING_FIELDS)}
_OBJECTIVE_EVENT_TYPES = {
    "ELITE_MONSTER_KILL",
    "BUILDING_KILL",
    "TURRET_PLATE_DESTROYED",
    "DRAGON_SOUL_GIVEN",
}
_ITEM_EVENT_TYPES = frozenset(ITEM_TYPES)
_OBJECTIVE_CATEGORY_BY_EVENT = {
    "ELITE_MONSTER_KILL": {
        "DRAGON": "dragon",
        "BARON_NASHOR": "baron",
        "RIFTHERALD": "herald",
        "RIFT_HERALD": "herald",
        "ATAKHAN": "atakhan",
        "HORDE": "horde",
    },
    "BUILDING_KILL": {
        "TOWER_BUILDING": "tower",
        "TOWER": "tower",
        "INHIBITOR_BUILDING": "inhibitor",
        "INHIBITOR": "inhibitor",
    },
}


def _array(bank: Mapping[str, Any], name: str, *aliases: str, default: Any = None) -> np.ndarray | None:
    value: Any = None
    found = False
    for candidate in (name, *aliases):
        if candidate in bank:
            value = bank[candidate]
            found = True
            break
    if not found:
        if default is None:
            return None
        value = default
    result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"{name} has object dtype")
    return result


def _integer_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.hasobject or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return result


def _scalar_int(value: Any, name: str) -> int:
    result = np.asarray(value)
    if result.size != 1 or result.dtype.hasobject:
        raise ValueError(f"{name} must be a scalar integer")
    item = result.reshape(()).item()
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
        raise ValueError(f"{name} must be a scalar integer")
    return int(item)


def _valid_int(value: Any, *, nonnegative: bool = False) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        parsed = int(value)
    else:
        return None
    if nonnegative and parsed < 0:
        return None
    return parsed


def _valid_pid(value: Any) -> int | None:
    parsed = _valid_int(value)
    return parsed if parsed is not None and 1 <= parsed <= PARTICIPANTS else None


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


def _decode_sparse_numeric(bank: Mapping[str, Any], n_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_values = _array(bank, "event_field_values")
    direct_validity = _array(bank, "event_field_validity")
    width = len(EVENT_PAYLOAD_FIELDS)
    if direct_values is not None or direct_validity is not None:
        if direct_values is None or direct_validity is None:
            raise ValueError("event field values and validity must be supplied together")
        values = np.asarray(direct_values)
        validity = np.asarray(direct_validity, dtype=bool)
        if values.shape != (n_events, width) or validity.shape != values.shape:
            raise ValueError("event field matrices have the wrong shape")
        if values.dtype.hasobject or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("event field values must be integer typed")
        explicit_presence = _array(bank, "event_field_presence")
        if explicit_presence is None:
            presence = validity.copy()
        else:
            presence = np.asarray(explicit_presence, dtype=bool)
            if presence.shape != validity.shape:
                raise ValueError("event field presence has the wrong shape")
        return values.astype(np.int64, copy=False), validity, presence

    width_value = _array(bank, "event_field_width")
    if width_value is None:
        empty = np.zeros((n_events, width), dtype=bool)
        return np.zeros((n_events, width), dtype=np.int64), empty, empty.copy()
    width = _scalar_int(width_value, "event_field_width")
    if width != len(EVENT_PAYLOAD_FIELDS):
        raise ValueError("event field width differs from the accepted source schema")
    rows = _integer_array(_array(bank, "event_field_row", default=np.empty(0, dtype=np.int32)), "event_field_row", ndim=1)
    cols = _integer_array(_array(bank, "event_field_col", default=np.empty(0, dtype=np.int16)), "event_field_col", ndim=1)
    values_sparse = _integer_array(
        _array(bank, "event_field_nonzero", default=np.empty(0, dtype=np.int64)),
        "event_field_nonzero", ndim=1,
    )
    if not (len(rows) == len(cols) == len(values_sparse)):
        raise ValueError("sparse event field arrays have different lengths")
    if len(rows):
        if np.any(rows < 0) or np.any(rows >= n_events) or np.any(cols < 0) or np.any(cols >= width):
            raise ValueError("sparse event field coordinate is outside the bank")
        keys = rows.astype(np.int64) * width + cols.astype(np.int64)
        if len(keys) > 1 and np.any(np.diff(keys) <= 0):
            raise ValueError("sparse event field coordinates are not unique and ordered")
    values = np.zeros((n_events, width), dtype=np.int64)
    values[rows, cols] = values_sparse
    mask = _integer_array(
        _array(bank, "event_field_valid_mask", default=np.zeros(n_events, dtype=np.uint32)),
        "event_field_valid_mask", ndim=1,
    )
    if mask.shape != (n_events,):
        raise ValueError("event field validity mask has the wrong shape")
    validity = ((mask.astype(np.uint64)[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(bool)
    presence = validity.copy()
    explicit_presence = _array(bank, "event_field_presence")
    if explicit_presence is not None:
        explicit_presence = np.asarray(explicit_presence, dtype=bool)
        if explicit_presence.shape != validity.shape:
            raise ValueError("event field presence has the wrong shape")
        presence = presence | explicit_presence
    return values, validity, presence


def _decode_assists(bank: Mapping[str, Any], n_events: int) -> tuple[list[list[int] | None], np.ndarray, np.ndarray]:
    offsets = _array(bank, "event_assist_offsets")
    if offsets is None:
        counts = _array(bank, "event_assist_counts")
        if counts is None:
            offsets = np.zeros(n_events + 1, dtype=np.int64)
        else:
            counts = _integer_array(counts, "event_assist_counts", ndim=1)
            if counts.shape != (n_events,) or np.any(counts < 0):
                raise ValueError("event assist counts have the wrong shape")
            offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    offsets = _integer_array(offsets, "event_assist_offsets", ndim=1)
    if offsets.shape != (n_events + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("event assist offsets are invalid")
    values = _integer_array(_array(bank, "event_assist_values", default=np.empty(0, dtype=np.int64)), "event_assist_values", ndim=1)
    if int(offsets[-1]) != len(values):
        raise ValueError("event assist offsets do not cover the stored values")
    validity = np.asarray(_array(bank, "event_assist_validity", default=np.zeros(n_events, dtype=bool)), dtype=bool)
    presence = np.asarray(_array(bank, "event_assist_presence", default=validity.copy()), dtype=bool)
    if validity.shape != (n_events,) or presence.shape != (n_events,):
        raise ValueError("event assist validity/presence has the wrong shape")
    decoded: list[list[int] | None] = []
    for index in range(n_events):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        if bool(validity[index]):
            decoded.append([int(value) for value in values[begin:end].tolist()])
        elif bool(presence[index]):
            decoded.append(None)
        else:
            decoded.append(None)
    return decoded, presence | validity, validity


def _decode_strings(bank: Mapping[str, Any], n_events: int) -> tuple[list[dict[str, str]], np.ndarray]:
    width = len(EVENT_STRING_FIELDS)
    direct_codes = _array(bank, "event_string_codes")
    direct_validity = _array(bank, "event_string_validity")
    if direct_codes is not None or direct_validity is not None:
        if direct_codes is None or direct_validity is None:
            raise ValueError("event string codes and validity must be supplied together")
        codes = np.asarray(direct_codes)
        validity = np.asarray(direct_validity, dtype=bool)
        if codes.shape != (n_events, width) or validity.shape != codes.shape:
            raise ValueError("event string matrices have the wrong shape")
    else:
        width_value = _array(bank, "event_string_width")
        if width_value is None:
            return [{} for _ in range(n_events)], np.zeros((n_events, width), dtype=bool)
        encoded_width = _scalar_int(width_value, "event_string_width")
        if encoded_width != width:
            raise ValueError("event string width differs from accepted source schema")
        rows = _integer_array(_array(bank, "event_string_row", default=np.empty(0, dtype=np.int32)), "event_string_row", ndim=1)
        cols = _integer_array(_array(bank, "event_string_col", default=np.empty(0, dtype=np.int16)), "event_string_col", ndim=1)
        nonzero = _integer_array(_array(bank, "event_string_nonzero", default=np.empty(0, dtype=np.int16)), "event_string_nonzero", ndim=1)
        if not (len(rows) == len(cols) == len(nonzero)):
            raise ValueError("sparse event string arrays have different lengths")
        if len(rows):
            if np.any(rows < 0) or np.any(rows >= n_events) or np.any(cols < 0) or np.any(cols >= width):
                raise ValueError("sparse event string coordinate is outside the bank")
            keys = rows.astype(np.int64) * width + cols.astype(np.int64)
            if len(keys) > 1 and np.any(np.diff(keys) <= 0):
                raise ValueError("sparse event string coordinates are not unique and ordered")
        codes = np.zeros((n_events, width), dtype=np.int64)
        codes[rows, cols] = nonzero
        mask = _integer_array(
            _array(bank, "event_string_valid_mask", default=np.zeros(n_events, dtype=np.uint16)),
            "event_string_valid_mask", ndim=1,
        )
        if mask.shape != (n_events,):
            raise ValueError("event string validity mask has the wrong shape")
        validity = ((mask.astype(np.uint64)[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(bool)
    unknown: dict[tuple[int, int], str] = {}
    unknown_rows = _integer_array(_array(bank, "event_string_unknown_rows", default=np.empty(0, dtype=np.int32)), "event_string_unknown_rows", ndim=1)
    unknown_cols = _integer_array(_array(bank, "event_string_unknown_cols", default=np.empty(0, dtype=np.int16)), "event_string_unknown_cols", ndim=1)
    unknown_offsets = _integer_array(
        _array(bank, "event_string_unknown_offsets", default=np.zeros(len(unknown_rows) + 1, dtype=np.int64)),
        "event_string_unknown_offsets", ndim=1,
    )
    unknown_values = _array(bank, "event_string_unknown_values", default=np.empty(0, dtype=np.uint8))
    if len(unknown_cols) != len(unknown_rows) or len(unknown_offsets) != len(unknown_rows) + 1:
        raise ValueError("unknown event string arrays have different lengths")
    if unknown_offsets[0] != 0 or np.any(np.diff(unknown_offsets) < 0) or int(unknown_offsets[-1]) > len(unknown_values):
        raise ValueError("unknown event string offsets are invalid")
    for index, (row, col) in enumerate(zip(unknown_rows.tolist(), unknown_cols.tolist())):
        if not 0 <= int(row) < n_events or not 0 <= int(col) < width:
            raise ValueError("unknown event string coordinate is outside the bank")
        begin, end = int(unknown_offsets[index]), int(unknown_offsets[index + 1])
        try:
            text = bytes(np.asarray(unknown_values[begin:end], dtype=np.uint8).tolist()).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("unknown event string is not valid UTF-8") from exc
        unknown[(int(row), int(col))] = text
    decoded = []
    for row in range(n_events):
        record: dict[str, str] = {}
        for col, name in enumerate(EVENT_STRING_FIELDS):
            if not bool(validity[row, col]):
                continue
            code = int(codes[row, col])
            names = EVENT_STRING_ENUMS[name]
            if 0 < code < len(names):
                record[name] = str(names[code])
            elif (row, col) in unknown:
                record[name] = unknown[(row, col)]
            else:
                # Code zero is a present but unknown token.  Preserve that
                # fact through field_presence without inventing a value.
                record[name] = "UNKNOWN"
        decoded.append(record)
    return decoded, validity


def _decode_raw_payload(bank: Mapping[str, Any], n_events: int) -> dict[int, dict[str, Any]]:
    rows = _integer_array(_array(bank, "event_raw_rows", default=np.empty(0, dtype=np.int32)), "event_raw_rows", ndim=1)
    cols = _integer_array(_array(bank, "event_raw_cols", default=np.empty(0, dtype=np.int16)), "event_raw_cols", ndim=1)
    offsets = _integer_array(
        _array(bank, "event_raw_offsets", default=np.zeros(len(rows) + 1, dtype=np.int64)),
        "event_raw_offsets", ndim=1,
    )
    values = _array(bank, "event_raw_values", default=np.empty(0, dtype=np.uint8))
    if len(rows) != len(cols) or len(offsets) != len(rows) + 1:
        raise ValueError("event raw arrays have different lengths")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or int(offsets[-1]) > len(values):
        raise ValueError("event raw offsets are invalid")
    names = (*EVENT_PAYLOAD_FIELDS, HISTORY_ASSIST_FIELD)
    output: dict[int, dict[str, Any]] = {}
    for index, (row_value, col_value) in enumerate(zip(rows.tolist(), cols.tolist())):
        row, col = int(row_value), int(col_value)
        if not 0 <= row < n_events or not 0 <= col < len(names):
            raise ValueError("event raw coordinate is outside the accepted payload")
        begin, end = int(offsets[index]), int(offsets[index + 1])
        raw_bytes = bytes(np.asarray(values[begin:end], dtype=np.uint8).tolist())
        try:
            decoded = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("event raw payload is not valid JSON/UTF-8") from exc
        output.setdefault(row, {})[names[col]] = decoded
    return output


def _objective_category(event_type: str, payload: Mapping[str, Any], category_code: int) -> str | None:
    if 1 <= category_code <= len(OBJECTIVE_TYPES):
        return str(OBJECTIVE_TYPES[category_code - 1])
    if event_type == "ELITE_MONSTER_KILL":
        return _OBJECTIVE_CATEGORY_BY_EVENT[event_type].get(str(payload.get("monsterType", "")).upper())
    if event_type == "BUILDING_KILL":
        return _OBJECTIVE_CATEGORY_BY_EVENT[event_type].get(str(payload.get("buildingType", "")).upper())
    if event_type == "TURRET_PLATE_DESTROYED":
        return "plate"
    if event_type == "DRAGON_SOUL_GIVEN":
        return "dragon_soul"
    return None


def _event_types(bank: Mapping[str, Any], n_events: int) -> tuple[str, ...]:
    direct = bank.get("event_type")
    if direct is not None:
        values = tuple(str(value) for value in direct)
        if len(values) != n_events:
            raise ValueError("event_type has the wrong length")
        return values
    codes = _integer_array(_array(bank, "event_type_code", default=np.empty(0, dtype=np.int16)), "event_type_code", ndim=1)
    if codes.shape != (n_events,):
        raise ValueError("event_type_code has the wrong shape")
    return tuple(EVENT_TYPE_NAMES[int(code)] if 0 <= int(code) < len(EVENT_TYPE_NAMES) else "UNKNOWN" for code in codes.tolist())


def _decode_events(bank: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], np.ndarray, np.ndarray]:
    timestamps = _integer_array(_array(bank, "event_timestamp", "event_timestamps", default=np.empty(0, dtype=np.int64)), "event_timestamp", ndim=1)
    if len(timestamps) > 1 and np.any(np.diff(timestamps.astype(np.int64)) < 0):
        raise ValueError("event timestamps must be nondecreasing")
    n_events = len(timestamps)
    types = _event_types(bank, n_events)
    values, valid, presence = _decode_sparse_numeric(bank, n_events)
    assists, assist_presence, assist_validity = _decode_assists(bank, n_events)
    strings, string_validity = _decode_strings(bank, n_events)
    raw = _decode_raw_payload(bank, n_events)
    categories = _integer_array(_array(bank, "event_category_code", default=np.zeros(n_events, dtype=np.int16)), "event_category_code", ndim=1)
    if categories.shape != (n_events,):
        raise ValueError("event_category_code has the wrong shape")
    records: list[Mapping[str, Any]] = []
    for row, (timestamp, event_type) in enumerate(zip(timestamps.tolist(), types)):
        all_payload: dict[str, Any] = {}
        all_presence: dict[str, bool] = {}
        all_validity: dict[str, bool] = {}
        for col, name in enumerate(EVENT_PAYLOAD_FIELDS):
            if bool(presence[row, col]):
                all_presence[name] = True
            if bool(valid[row, col]):
                all_payload[name] = int(values[row, col])
                all_validity[name] = True
        for name, value in raw.get(row, {}).items():
            all_payload[name] = value
            all_presence[name] = True
            # A raw sidecar records the source's malformed/noncanonical value;
            # a typed validity bit, when present, remains authoritative.
            if name != HISTORY_ASSIST_FIELD:
                all_validity.setdefault(name, bool(valid[row, _NUMERIC_INDEX[name]]))
        if bool(assist_presence[row]) or bool(assist_validity[row]):
            all_presence[HISTORY_ASSIST_FIELD] = True
            all_validity[HISTORY_ASSIST_FIELD] = bool(assist_validity[row])
            if bool(assist_validity[row]):
                all_payload[HISTORY_ASSIST_FIELD] = assists[row]
            elif HISTORY_ASSIST_FIELD not in all_payload:
                all_payload[HISTORY_ASSIST_FIELD] = None
        for col, name in enumerate(EVENT_STRING_FIELDS):
            if bool(string_validity[row, col]):
                all_presence[name] = True
                all_validity[name] = True
                all_payload[name] = strings[row].get(name, "UNKNOWN")
        category = _objective_category(event_type, all_payload, int(categories[row]))
        if category is not None:
            all_payload["objective_category"] = category
        allowed = _allowed_fields(event_type)
        payload = {name: all_payload[name] for name in allowed if name in all_payload}
        field_presence = {name: bool(all_presence.get(name, False)) for name in allowed}
        field_validity = {name: bool(all_validity.get(name, False)) for name in allowed}
        if event_type in HISTORY_ALLOWED_EVENT_TYPES:
            records.append(
                _freeze(
                    {
                        "timestamp_ms": int(timestamp),
                        "event_type": event_type,
                        "payload": payload,
                        "field_presence": field_presence,
                        "field_validity": field_validity,
                        **({"objective_category": category} if category is not None and event_type in _OBJECTIVE_EVENT_TYPES else {}),
                    }
                )
            )
        else:
            # Keep no excluded event records in the semantic history.  The
            # timestamp/type arrays remain available to the numeric cutoff
            # validator and are never model inputs.
            continue
    return tuple(records), timestamps.astype(np.int64, copy=False), np.asarray(types, dtype="U32")


def _allowed_fields(event_type: str) -> tuple[str, ...]:
    if event_type == "CHAMPION_KILL":
        return ("killerId", "victimId", "killerTeamId", "teamId", HISTORY_ASSIST_FIELD)
    if event_type in _OBJECTIVE_EVENT_TYPES:
        return (
            "killerId", "killerTeamId", "teamId",
            "monsterType", "monsterSubType", "buildingType", "towerType", "laneType",
            "dragonSoul", "soulType", "name",
        )
    if event_type == "LEVEL_UP":
        return ("participantId", "level", "levelUpType")
    if event_type == "SKILL_LEVEL_UP":
        return ("participantId", "level", "skillSlot", "levelUpType")
    if event_type in _ITEM_EVENT_TYPES:
        return ("participantId", "itemId", "beforeId", "afterId")
    return ()


class _RecordView(Sequence[Mapping[str, Any]]):
    __slots__ = ("_records", "_stop")

    def __init__(self, records: tuple[Mapping[str, Any], ...], stop: int):
        self._records = records
        self._stop = int(stop)

    def __len__(self) -> int:
        return self._stop

    def __getitem__(self, index: int | slice) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
        if isinstance(index, slice):
            return self._records[: self._stop][index]
        if index < 0:
            index += self._stop
        if index < 0 or index >= self._stop:
            raise IndexError(index)
        return self._records[index]

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._records[: self._stop])


class HistoryPrefix(Mapping[str, Sequence[Mapping[str, Any]]]):
    """Lazy typed history partition views for one inclusive query cutoff."""

    __slots__ = ("query_ms", "_partitions")

    def __init__(
        self,
        query_ms: int,
        partitions: Mapping[str, tuple[tuple[Mapping[str, Any], ...], np.ndarray]],
    ):
        self.query_ms = int(query_ms)
        self._partitions = dict(partitions)

    def __getitem__(self, key: str) -> Sequence[Mapping[str, Any]]:
        if key not in self._partitions:
            raise KeyError(key)
        records, timestamps = self._partitions[key]
        return _RecordView(records, bisect_right(timestamps, self.query_ms))

    def __iter__(self) -> Iterator[str]:
        return iter(self._partitions)

    def __len__(self) -> int:
        return len(self._partitions)

    def to_dict(self) -> dict[str, tuple[Mapping[str, Any], ...]]:
        return {key: tuple(self[key]) for key in self._partitions}


def _columnar(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records_tuple = tuple(records)
    event_types = [str(record["event_type"]) for record in records_tuple]
    width = max((len(value) for value in event_types), default=1)
    return {
        "timestamp_ms": np.asarray([int(record["timestamp_ms"]) for record in records_tuple], dtype=np.int64),
        "event_type": np.asarray(event_types, dtype=f"U{width}"),
        "payload": tuple(record["payload"] for record in records_tuple),
        "field_presence": tuple(record["field_presence"] for record in records_tuple),
        "field_validity": tuple(record["field_validity"] for record in records_tuple),
        "records": records_tuple,
    }


class SemanticHistory:
    """One decoded match history with lazy inclusive timestamp prefixes."""

    def __init__(self, bank: Mapping[str, Any]):
        if not isinstance(bank, Mapping):
            raise TypeError("bank must be a mapping")
        records, timestamps, _types = _decode_events(bank)
        self._timestamps = timestamps
        self._records = records
        self._record_timestamps = np.asarray([int(record["timestamp_ms"]) for record in records], dtype=np.int64)
        partition_lists: dict[str, list[Mapping[str, Any]]] = {name: [] for name in HISTORY_PARTITIONS}
        for record in records:
            event_type = str(record["event_type"])
            for name, event_types in HISTORY_PARTITIONS.items():
                if event_type in event_types:
                    partition_lists[name].append(record)
        partitions = {name: tuple(values) for name, values in partition_lists.items()}
        self._partitions = {
            name: (values, np.asarray([int(record["timestamp_ms"]) for record in values], dtype=np.int64))
            for name, values in partitions.items()
        }

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return self._records

    def prefix(self, query_ms: Any) -> HistoryPrefix:
        value = _valid_int(query_ms, nonnegative=True)
        if value is None:
            raise ValueError("query_ms must be a nonnegative integer")
        return HistoryPrefix(value, self._partitions)

    def iter_prefix(self, query_ms: Any) -> Iterator[Mapping[str, Any]]:
        value = _valid_int(query_ms, nonnegative=True)
        if value is None:
            raise ValueError("query_ms must be a nonnegative integer")
        stop = bisect_right(self._record_timestamps, value)
        for index in range(stop):
            yield self._records[index]

    def prefix_columnar(self, query_ms: Any) -> dict[str, dict[str, Any]]:
        prefix = self.prefix(query_ms)
        return {name: _columnar(records) for name, records in prefix.items()}

    # Short aliases used by source readers that choose a columnar history view.
    columnar = prefix_columnar

    def prefix_view(self, query_ms: Any) -> HistoryPrefix:
        return self.prefix(query_ms)


def _validate_common(bank: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minute_ts = _integer_array(_array(bank, "minute_ts"), "minute_ts", ndim=1)
    if not len(minute_ts) or (len(minute_ts) > 1 and np.any(np.diff(minute_ts.astype(np.int64)) <= 0)):
        raise ValueError("minute_ts must be nonempty and strictly increasing")
    node_frame = _array(bank, "node_frame")
    if node_frame is None or node_frame.ndim != 3 or node_frame.shape[1:] != (PARTICIPANTS, FRAME_WIDTH):
        raise ValueError("node_frame must have shape (F, 10, 46)")
    if not np.issubdtype(node_frame.dtype, np.floating):
        raise ValueError("node_frame must be floating point")
    gold_team = _array(bank, "gold_team")
    if gold_team is None or gold_team.shape != (len(minute_ts), 2) or not np.issubdtype(gold_team.dtype, np.floating):
        raise ValueError("gold_team must have shape (F, 2) and floating dtype")
    team_ids = _integer_array(_array(bank, "team_ids"), "team_ids", ndim=1)
    if team_ids.shape != (PARTICIPANTS,) or np.sum(team_ids == TEAM_BLUE) != 5 or np.sum(team_ids == TEAM_RED) != 5:
        raise ValueError("team_ids must contain exactly five Blue and five Red participants")
    static_ids = _integer_array(_array(bank, "static_ids"), "static_ids")
    bans = _integer_array(_array(bank, "bans"), "bans")
    if static_ids.shape != (PARTICIPANTS, 14) or bans.shape != BANS_SHAPE:
        raise ValueError("static_ids or bans has the wrong shape")
    return minute_ts.astype(np.int64, copy=False), node_frame, gold_team, team_ids, static_ids, bans


def _query_array(bank: Mapping[str, Any], name: str, q: int, shape_tail: tuple[int, ...], dtype: Any) -> np.ndarray | None:
    value = _array(bank, name)
    if value is None:
        return None
    result = np.asarray(value, dtype=dtype)
    if result.shape != (q, *shape_tail):
        raise ValueError(f"{name} must have shape {(q, *shape_tail)}")
    return result


def _growth(node: np.ndarray, team_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate all query frames at once, preserving per-channel unknowns."""

    blue_rows = np.flatnonzero(team_ids == TEAM_BLUE)
    red_rows = np.flatnonzero(team_ids == TEAM_RED)
    blue = np.asarray(node[:, blue_rows, :6], dtype=np.float64)
    red = np.asarray(node[:, red_rows, :6], dtype=np.float64)
    output = np.full((len(node), 2, 6), np.nan, dtype=np.float64)
    for side, values in enumerate((blue, red)):
        valid = np.all(np.isfinite(values), axis=1)
        output[:, side, 0] = np.where(valid[:, 0], np.mean(values[:, :, 0], axis=1), np.nan)
        sums = np.sum(values[:, :, 1:], axis=1)
        output[:, side, 1:] = np.where(valid[:, 1:], sums, np.nan)
    delta = np.full((len(node), 6), np.nan, dtype=np.float64)
    valid_delta = np.isfinite(output[:, 0]) & np.isfinite(output[:, 1])
    delta = np.where(valid_delta, output[:, 0] - output[:, 1], np.nan)
    return output, delta


def project_match(bank: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Project one accepted match bank into query-leading raw blocks.

    Query cutoffs are inclusive and are validated against the source clocks.
    Audit, target, coordinate, terminal, source-identity, and inferred-owner
    fields are intentionally absent from the returned model blocks.
    """

    if not isinstance(bank, Mapping):
        raise TypeError("bank must be a mapping")
    minute_ts, node_frame, gold_team, team_ids, static_ids, bans = _validate_common(bank)
    event_timestamps = _integer_array(
        _array(bank, "event_timestamp", "event_timestamps", default=np.empty(0, dtype=np.int64)),
        "event_timestamp", ndim=1,
    ).astype(np.int64, copy=False)
    if len(event_timestamps) > 1 and np.any(np.diff(event_timestamps) < 0):
        raise ValueError("event timestamps must be nondecreasing")
    event_types_array = np.asarray(_event_types(bank, len(event_timestamps)), dtype="U32")
    event_types = tuple(str(value) for value in event_types_array.tolist())
    q_values = _integer_array(_array(bank, "query_ms"), "query_ms", ndim=1)
    q = len(q_values)
    query_ms = q_values.astype(np.int64, copy=False)
    if q and np.any(query_ms < minute_ts[0]) or q and np.any(query_ms > minute_ts[-1]):
        raise ValueError("query_ms is outside supported frame range")
    expected_frame = np.searchsorted(minute_ts, query_ms, side="right").astype(np.int64) - 1
    frame_index = _query_array(bank, "query_frame_index", q, (), np.int64)
    if frame_index is None:
        frame_index = expected_frame
    if not np.array_equal(frame_index, expected_frame) or np.any(frame_index < 0) or np.any(frame_index >= len(minute_ts)):
        raise ValueError("query_frame_index is not the exact supported-frame cutoff")
    expected_event_limit = np.searchsorted(event_timestamps, query_ms, side="right").astype(np.int64)
    event_limit = _query_array(bank, "query_event_limit", q, (), np.int64)
    if event_limit is None:
        event_limit = expected_event_limit
    if not np.array_equal(event_limit, expected_event_limit) or np.any(event_limit < 0) or np.any(event_limit > len(event_timestamps)):
        raise ValueError("query_event_limit is not the inclusive event cutoff")
    item_indices = _array(bank, "item_event_indices")
    if item_indices is None:
        item_indices = np.asarray(
            [index for index, event_type in enumerate(event_types) if event_type in _ITEM_EVENT_TYPES], dtype=np.int64
        )
    item_indices = _integer_array(item_indices, "item_event_indices", ndim=1).astype(np.int64, copy=False)
    if len(item_indices) and (np.any(item_indices < 0) or np.any(item_indices >= len(event_timestamps))):
        raise ValueError("item_event_indices contains an out-of-range event")
    if len(item_indices) > 1 and np.any(np.diff(item_indices) <= 0):
        raise ValueError("item_event_indices must be strictly increasing")
    expected_item_limit = np.searchsorted(item_indices, event_limit, side="left").astype(np.int64)
    item_limit = _query_array(bank, "query_item_limit", q, (), np.int64)
    if item_limit is None:
        item_limit = expected_item_limit
    if not np.array_equal(item_limit, expected_item_limit) or np.any(item_limit < 0) or np.any(item_limit > len(item_indices)):
        raise ValueError("query_item_limit is not the item prefix cutoff")

    team_kills = _query_array(bank, "query_team_kills", q, (2,), np.int32)
    kda = _query_array(bank, "query_kda", q, (3, PARTICIPANTS), np.int32)
    if team_kills is None or kda is None:
        raise ValueError("accepted bank must provide query_team_kills and query_kda")
    objective_source = _array(bank, "query_objective_counts")
    if objective_source is None:
        raise ValueError("accepted bank must provide query_objective_counts")
    objective_source = np.asarray(objective_source)
    if objective_source.shape == (q, len(OBJECTIVE_TYPES), 3):
        objective_counts = objective_source[:, :, 0].astype(np.int32, copy=False)
    elif objective_source.shape == (q, len(OBJECTIVE_TYPES)):
        objective_counts = objective_source.astype(np.int32, copy=False)
    else:
        raise ValueError("query_objective_counts must have shape (Q, 9, 3) or (Q, 9)")
    objective_last = _query_array(bank, "query_objective_last_ms", q, (len(OBJECTIVE_TYPES),), np.float64)
    if objective_last is None:
        raise ValueError("accepted bank must provide query_objective_last_ms")
    finite_last = np.isfinite(objective_last)
    if np.any(finite_last & (objective_last > query_ms[:, None])):
        raise ValueError("objective last timestamp contains future information")
    held_node = np.asarray(node_frame[frame_index], dtype=np.float32)
    growth, growth_delta = _growth(held_node, team_ids)
    objective_age = np.full((q, len(OBJECTIVE_TYPES)), np.nan, dtype=np.float64)
    objective_age[finite_last] = (query_ms[:, None] - objective_last)[finite_last] / 1000.0
    result: dict[str, np.ndarray] = {
        "node_frame": held_node,
        "elapsed_minutes": query_ms.astype(np.float64) / 60000.0,
        "frame_age_seconds": (query_ms - minute_ts[frame_index]).astype(np.float64) / 1000.0,
        "team_growth": growth,
        "team_growth_delta": growth_delta,
        "cached_gold": np.asarray(gold_team[frame_index], dtype=np.float32),
        "team_kills": np.asarray(team_kills, dtype=np.int32),
        "player_kda": np.asarray(kda.transpose(0, 2, 1), dtype=np.int32),
        "objective_counts": np.asarray(objective_counts, dtype=np.int32),
        "objective_age_seconds": objective_age,
        "static_ids": np.asarray(static_ids, dtype=np.int64).copy(),
        "bans": np.asarray(bans, dtype=np.int64).copy(),
        "team_ids": np.asarray(team_ids, dtype=np.int64).copy(),
    }
    return result


__all__ = ["HistoryPrefix", "SemanticHistory", "project_match"]
