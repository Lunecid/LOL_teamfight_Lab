"""Read the accepted observable banks for the item-ledger study.

The observable-state banks are immutable inputs to this study.  This module
does not reopen ``ObservableMatch`` instances: it validates the compact bank
members that carry item transactions, slices one match at a time, and keeps
the source ordinal/identity and malformed-payload provenance available to the
ledger and audit layers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from studies.observable_state_20261002.bank import match_view


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "outputs" / "observable_state_20261002" / "full_static_preserved_20261002_1"
DEFAULT_PREPARATION_ROOT = ROOT / "outputs" / "v_preparation_20261002" / "full_nominal_refs_20261002_1"
DEFAULT_ACCEPTANCE_PATH = ROOT / "reports" / "observable_state_20261002" / "ACCEPTANCE.json"

ITEM_TYPE_CODES = {
    7: "ITEM_DESTROYED",
    8: "ITEM_PURCHASED",
    9: "ITEM_SOLD",
    10: "ITEM_UNDO",
}
ITEM_TYPE_NAMES = frozenset(ITEM_TYPE_CODES.values())
ITEM_PAYLOAD_FIELDS = ("participantId", "itemId", "beforeId", "afterId")
ITEM_PAYLOAD_COLUMNS = {name: index for index, name in enumerate(ITEM_PAYLOAD_FIELDS)}

# These are the packed members needed for the direct ledger.  Static/team
# members are retained only because the caller may use them for node ownership
# diagnostics; they never enter the ledger feature surface.
CORE_MATCH_FIELDS = (
    "event_timestamp", "event_source_index", "event_type_code", "event_category_code",
    "event_field_width", "event_field_row", "event_field_col", "event_field_nonzero",
    "event_field_valid_mask", "event_explicit_rows", "event_explicit_identity",
    "event_identity_raw_rows", "event_identity_raw_offsets", "event_identity_raw_values",
    "event_raw_rows", "event_raw_cols", "event_raw_offsets", "event_raw_values",
    "item_event_indices", "query_ms", "query_item_ledger_known", "query_event_limit", "v_nominal",
)
NODE_AUDIT_FIELDS = ("team_ids", "static_ids")
OPTIONAL_MATCH_FIELDS = frozenset({"event_raw_rows", "event_raw_cols", "event_raw_offsets", "event_raw_values"})


class SourceValidationError(ValueError):
    """Raised when an accepted bank cannot be interpreted losslessly."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceValidationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SourceValidationError(f"JSON root is not an object: {path}")
    return value


def _integer_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise SourceValidationError(f"{name} must have ndim={ndim}, got {array.shape}")
    if array.dtype.kind not in "iu":
        raise SourceValidationError(f"{name} must have an integer dtype, got {array.dtype}")
    return array


def _bool_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise SourceValidationError(f"{name} must have boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise SourceValidationError(f"{name} shape {array.shape} != {shape}")
    return array


def _scalar_int(value: Any, *, name: str) -> int:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.size != 1:
        raise SourceValidationError(f"{name} must be one integer scalar")
    return int(array.reshape(-1)[0])


def _validate_offsets(value: Any, *, name: str, n_matches: int, n_values: int) -> np.ndarray:
    offsets = _integer_array(value, name=name, ndim=1)
    if offsets.shape != (n_matches + 1,):
        raise SourceValidationError(f"{name} must have shape {(n_matches + 1,)}, got {offsets.shape}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_values:
        raise SourceValidationError(f"{name} does not span its packed member")
    if np.any(offsets[1:] < offsets[:-1]) or np.any(offsets < 0):
        raise SourceValidationError(f"{name} is not nondecreasing and nonnegative")
    return offsets


def _validate_ragged_reference_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    rows_name: str,
    cols_name: str,
    offsets_name: str,
    values_name: str,
    n_events: int,
    width: int,
) -> None:
    rows = _integer_array(arrays[rows_name], name=rows_name, ndim=1)
    cols = _integer_array(arrays[cols_name], name=cols_name, ndim=1)
    offsets = _integer_array(arrays[offsets_name], name=offsets_name, ndim=1)
    values = np.asarray(arrays[values_name])
    if rows.shape != cols.shape or offsets.shape != (len(rows) + 1,):
        raise SourceValidationError(f"{rows_name}/{cols_name}/{offsets_name} lengths disagree")
    if values.ndim != 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(values):
        raise SourceValidationError(f"{offsets_name} does not span {values_name}")
    if values.dtype.kind not in "biu" or (len(values) and (np.any(values < 0) or np.any(values > 255))):
        raise SourceValidationError(f"{values_name} must contain byte values")
    if np.any(offsets[1:] < offsets[:-1]) or np.any(offsets < 0):
        raise SourceValidationError(f"{offsets_name} is not monotone")
    if len(rows):
        if np.any(rows < 0) or np.any(rows >= n_events):
            raise SourceValidationError(f"{rows_name} references an event outside the match")
        if np.any(cols < 0) or np.any(cols > width):
            raise SourceValidationError(f"{cols_name} references a field outside the payload width")


def _decode_raw_value(raw: bytes) -> tuple[Any, bool]:
    """Decode an exported JSON raw marker without treating bad bytes as absent."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"__invalid_raw_utf8__": raw.hex()}, False
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {"__invalid_raw_json__": text}, False


def _raw_values(
    rows: np.ndarray,
    cols: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    selected_rows: set[int] | None = None,
) -> dict[tuple[int, int], tuple[Any, bool, bytes]]:
    result: dict[tuple[int, int], tuple[Any, bool, bytes]] = {}
    for index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
        if selected_rows is not None and int(row) not in selected_rows:
            continue
        begin, end = int(offsets[index]), int(offsets[index + 1])
        raw = bytes(np.asarray(values[begin:end], dtype=np.uint8).tolist())
        decoded, valid = _decode_raw_value(raw)
        result[(int(row), int(col))] = (decoded, valid, raw)
    return result


def _identity_values(
    rows: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
) -> dict[int, tuple[str | None, bool, bytes]]:
    result: dict[int, tuple[str | None, bool, bytes]] = {}
    for index, row in enumerate(rows.tolist()):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        raw = bytes(np.asarray(values[begin:end], dtype=np.uint8).tolist())
        try:
            text = raw.decode("utf-8")
            result[int(row)] = (text, True, raw)
        except UnicodeDecodeError:
            result[int(row)] = (None, False, raw)
    return result


@dataclass(frozen=True)
class ItemEvent:
    """Compact item event with source and raw-payload provenance."""

    event_index: int
    timestamp_ms: int
    source_ordinal: int
    source_identity: Any
    source_identity_valid: bool
    source_identity_raw: bytes | None
    event_type: str
    participant_id: Any = None
    item_id: Any = None
    before_id: Any = None
    after_id: Any = None
    field_presence: tuple[bool, bool, bool, bool] = (False, False, False, False)
    field_validity: tuple[bool, bool, bool, bool] = (False, False, False, False)
    raw_fields: Mapping[str, Any] = field(default_factory=dict)
    raw_field_validity: Mapping[str, bool] = field(default_factory=dict)
    raw_field_bytes: Mapping[str, bytes] = field(default_factory=dict)
    source_ordinal_valid: bool = True
    duplicate_source_identity: bool = False

    @property
    def payload_valid(self) -> bool:
        if self.event_type in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
            return bool(self.field_validity[0] and self.field_validity[1])
        return bool(self.field_validity[0] and (self.field_validity[2] or self.field_validity[3]))

    def to_mapping(self) -> dict[str, Any]:
        """Return the mapping accepted by :class:`ItemTransactionLedger`."""
        result: dict[str, Any] = {
            "timestamp": int(self.timestamp_ms),
            "source_ordinal": int(self.source_ordinal),
            "source_identity": self.source_identity,
            "duplicate_source_identity": bool(self.duplicate_source_identity),
            "type": self.event_type,
        }
        names = ITEM_PAYLOAD_FIELDS
        values = (self.participant_id, self.item_id, self.before_id, self.after_id)
        for index, name in enumerate(names):
            if self.field_presence[index]:
                result[name] = values[index]
        return result

    def to_transaction(self) -> Any:
        """Build the ledger dataclass lazily to keep the source reader standalone."""
        from .ledger import ItemTransaction

        return ItemTransaction(
            timestamp_ms=int(self.timestamp_ms),
            source_ordinal=int(self.source_ordinal),
            source_identity=self.source_identity,
            participant_id=self.participant_id if isinstance(self.participant_id, (int, np.integer)) and not isinstance(self.participant_id, (bool, np.bool_)) else None,
            item_id=self.item_id if isinstance(self.item_id, (int, np.integer)) and not isinstance(self.item_id, (bool, np.bool_)) else None,
            before_id=self.before_id if isinstance(self.before_id, (int, np.integer)) and not isinstance(self.before_id, (bool, np.bool_)) else None,
            after_id=self.after_id if isinstance(self.after_id, (int, np.integer)) and not isinstance(self.after_id, (bool, np.bool_)) else None,
            type=self.event_type,
            duplicate_source_identity=bool(self.duplicate_source_identity),
            raw_item_id=self.item_id,
            raw_payload=self.to_mapping(),
        )


@dataclass
class AcceptedMatch:
    """One match view sliced from a validated compact shard."""

    arrays: Mapping[str, np.ndarray]
    layout: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    shard_index: int = -1
    local_index: int = -1
    _item_event_cache: tuple[ItemEvent, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.arrays = {key: np.asarray(value) for key, value in self.arrays.items()}
        self._validate()

    @classmethod
    def from_bank(
        cls,
        arrays: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        shard_index: int = -1,
        local_index: int = -1,
    ) -> "AcceptedMatch":
        """Construct a match from an already sliced bank for focused tests."""
        layout = {"scalar_fields": [], "match_fields": list(arrays)}
        return cls(arrays, layout, metadata or {}, shard_index, local_index)

    @property
    def canonical_match_key(self) -> str | None:
        value = self.metadata.get("canonical_match_key")
        return str(value) if value is not None else None

    @property
    def patch(self) -> str | None:
        value = self.metadata.get("patch")
        return str(value) if value is not None else None

    @property
    def n_events(self) -> int:
        return int(len(self.arrays["event_timestamp"]))

    @property
    def n_queries(self) -> int:
        return int(len(self.arrays["query_ms"]))

    @property
    def item_event_indices(self) -> np.ndarray:
        return np.asarray(self.arrays["item_event_indices"], dtype=np.int64)

    @property
    def query_ms(self) -> np.ndarray:
        return np.asarray(self.arrays["query_ms"], dtype=np.int64)

    @property
    def query_event_limit(self) -> np.ndarray:
        return np.asarray(self.arrays["query_event_limit"], dtype=np.int64)

    @property
    def query_item_ledger_known(self) -> np.ndarray:
        return np.asarray(self.arrays["query_item_ledger_known"], dtype=bool)

    @property
    def v_nominal(self) -> np.ndarray:
        return np.asarray(self.arrays["v_nominal"])

    def _validate(self) -> None:
        arrays = self.arrays
        missing = [name for name in CORE_MATCH_FIELDS if name not in arrays and name not in OPTIONAL_MATCH_FIELDS]
        if missing:
            raise SourceValidationError("match bank is missing required members: " + ", ".join(missing))
        timestamps = _integer_array(arrays["event_timestamp"], name="event_timestamp", ndim=1)
        n_events = len(timestamps)
        if np.any(timestamps < 0) or (n_events > 1 and np.any(timestamps[1:] < timestamps[:-1])):
            raise SourceValidationError("event_timestamp must be nonnegative and nondecreasing")
        for name in ("event_source_index", "event_type_code", "event_category_code", "event_field_valid_mask"):
            value = _integer_array(arrays[name], name=name, ndim=1)
            if value.shape != (n_events,):
                raise SourceValidationError(f"{name} length differs from event_timestamp")
        source_index = np.asarray(arrays["event_source_index"])
        source_valid = source_index >= 0
        # The ordinal remains available even if a malformed negative source
        # value is present; the event is quarantined by ``source_identity_valid``.
        # Invalid source ordinals are represented as quarantined item events;
        # avoid mutating caller-owned ledger metadata during validation.
        width = _scalar_int(arrays["event_field_width"], name="event_field_width")
        if not 4 <= width <= 32:
            raise SourceValidationError("event_field_width must cover item fields and be at most 32")
        rows = _integer_array(arrays["event_field_row"], name="event_field_row", ndim=1)
        cols = _integer_array(arrays["event_field_col"], name="event_field_col", ndim=1)
        values = _integer_array(arrays["event_field_nonzero"], name="event_field_nonzero", ndim=1)
        if rows.shape != cols.shape or rows.shape != values.shape:
            raise SourceValidationError("sparse event payload arrays have different lengths")
        if len(rows):
            if np.any(rows < 0) or np.any(rows >= n_events) or np.any(cols < 0) or np.any(cols >= width):
                raise SourceValidationError("sparse event payload references an invalid coordinate")
            linear = rows.astype(np.int64) * width + cols.astype(np.int64)
            if np.any(linear[1:] <= linear[:-1]):
                raise SourceValidationError("sparse event payload coordinates are not unique and ordered")
        mask = np.asarray(arrays["event_field_valid_mask"])
        if np.any(mask < 0) or np.any(mask.astype(object) >= (1 << width)):
            raise SourceValidationError("event_field_valid_mask has bits outside event_field_width")
        for name in ("event_explicit_rows", "event_explicit_identity"):
            if name not in arrays:
                raise SourceValidationError("missing explicit source identity member: " + name)
        explicit_rows = _integer_array(arrays["event_explicit_rows"], name="event_explicit_rows", ndim=1)
        explicit_identity = _integer_array(arrays["event_explicit_identity"], name="event_explicit_identity", ndim=1)
        if explicit_rows.shape != explicit_identity.shape:
            raise SourceValidationError("explicit identity rows and codes disagree")
        if len(explicit_rows):
            if np.any(explicit_rows < 0) or np.any(explicit_rows >= n_events):
                raise SourceValidationError("explicit identity row is outside events")
            if np.any(explicit_rows[1:] <= explicit_rows[:-1]):
                raise SourceValidationError("explicit identity rows are not unique and ordered")
        self._validate_raw_members(n_events, width)
        item_indices = _integer_array(arrays["item_event_indices"], name="item_event_indices", ndim=1)
        if len(item_indices):
            if np.any(item_indices < 0) or np.any(item_indices >= n_events):
                raise SourceValidationError("item_event_indices references an event outside the match")
            if np.any(item_indices[1:] <= item_indices[:-1]):
                raise SourceValidationError("item_event_indices are not unique and ordered")
            type_codes = np.asarray(arrays["event_type_code"])[item_indices]
            if any(int(code) not in ITEM_TYPE_CODES for code in type_codes.tolist()):
                raise SourceValidationError("item_event_indices contains a non-item event")
        all_item_indices = np.flatnonzero(np.isin(np.asarray(arrays["event_type_code"]), tuple(ITEM_TYPE_CODES))).astype(item_indices.dtype, copy=False)
        if not np.array_equal(item_indices, all_item_indices):
            raise SourceValidationError("item_event_indices is not the complete recognized item-event index")
        query = _integer_array(arrays["query_ms"], name="query_ms", ndim=1)
        if np.any(query < 0) or (len(query) > 1 and np.any(query[1:] <= query[:-1])):
            raise SourceValidationError("query_ms must be nonnegative and strictly increasing")
        limits = _integer_array(arrays["query_event_limit"], name="query_event_limit", ndim=1)
        if limits.shape != query.shape or np.any(limits < 0) or np.any(limits > n_events):
            raise SourceValidationError("query_event_limit shape or bounds are invalid")
        expected_limits = np.searchsorted(timestamps, query, side="right").astype(limits.dtype, copy=False)
        if not np.array_equal(limits, expected_limits):
            raise SourceValidationError("query_event_limit is not an inclusive full timestamp cutoff")
        known = _bool_array(arrays["query_item_ledger_known"], name="query_item_ledger_known", shape=query.shape)
        nominal = _integer_array(arrays["v_nominal"], name="v_nominal", ndim=2)
        if nominal.shape[1] != 7:
            raise SourceValidationError("v_nominal must have seven integer columns")
        self._validate_nominal(nominal, query)
        if "team_ids" in arrays:
            team_ids = _integer_array(arrays["team_ids"], name="team_ids", ndim=1)
            if team_ids.shape != (10,):
                raise SourceValidationError("team_ids must contain participant IDs 1..10")
        if "static_ids" in arrays:
            static_ids = _integer_array(arrays["static_ids"], name="static_ids", ndim=2)
            if static_ids.shape != (10, 14):
                raise SourceValidationError("static_ids must have shape (10,14)")

    def _validate_raw_members(self, n_events: int, width: int) -> None:
        arrays = self.arrays
        if "event_identity_raw_rows" not in arrays or "event_identity_raw_offsets" not in arrays or "event_identity_raw_values" not in arrays:
            raise SourceValidationError("missing raw source identity members")
        identity_rows = _integer_array(arrays["event_identity_raw_rows"], name="event_identity_raw_rows", ndim=1)
        identity_offsets = _integer_array(arrays["event_identity_raw_offsets"], name="event_identity_raw_offsets", ndim=1)
        identity_values = np.asarray(arrays["event_identity_raw_values"])
        if identity_values.ndim != 1 or identity_values.dtype.kind not in "biu":
            raise SourceValidationError("event_identity_raw_values must be a one-dimensional byte array")
        if identity_offsets.shape != (len(identity_rows) + 1,):
            raise SourceValidationError("event_identity_raw_offsets length is invalid")
        if int(identity_offsets[0]) != 0 or int(identity_offsets[-1]) != len(identity_values) or np.any(identity_offsets[1:] < identity_offsets[:-1]):
            raise SourceValidationError("event_identity_raw_offsets are invalid")
        if len(identity_rows) and (np.any(identity_rows < 0) or np.any(identity_rows >= n_events)):
            raise SourceValidationError("event_identity_raw_rows references an invalid event")
        if len(identity_rows) and np.any(identity_rows[1:] <= identity_rows[:-1]):
            raise SourceValidationError("event_identity_raw_rows are not unique and ordered")
        if "event_raw_rows" in arrays or "event_raw_cols" in arrays or "event_raw_offsets" in arrays or "event_raw_values" in arrays:
            required = ("event_raw_rows", "event_raw_cols", "event_raw_offsets", "event_raw_values")
            if any(name not in arrays for name in required):
                raise SourceValidationError("event raw payload members are incomplete")
            _validate_ragged_reference_arrays(
                arrays, rows_name="event_raw_rows", cols_name="event_raw_cols", offsets_name="event_raw_offsets",
                values_name="event_raw_values", n_events=n_events, width=width,
            )

    @staticmethod
    def _validate_nominal(nominal: np.ndarray, query: np.ndarray) -> None:
        if len(nominal) == 0:
            return
        bucket, lo, hi, requested, actual, frame, reference = (nominal[:, index] for index in range(7))
        if np.any(bucket < 0) or np.any(lo < 0) or np.any(hi < lo) or np.any(requested < lo) or np.any(requested > hi):
            raise SourceValidationError("v_nominal contains an invalid bucket or query")
        supported = actual >= 0
        if np.any(actual[supported] != requested[supported]) or np.any(reference[supported] < 0) or np.any(reference[supported] >= len(query)):
            raise SourceValidationError("supported v_nominal rows disagree with their query references")
        if np.any(supported):
            supported_refs = reference[supported]
            if np.any(query[supported_refs] != actual[supported]):
                raise SourceValidationError("v_nominal query reference timestamp differs from actual_ms")
        if np.any(~supported & (reference != -1)):
            raise SourceValidationError("unsupported v_nominal row retains a query reference")
        if np.any(~supported & (frame != -1)):
            raise SourceValidationError("unsupported v_nominal row retains a frame reference")

    def _field_maps(self) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[Any, bool, bytes]]]:
        arrays = self.arrays
        rows = np.asarray(arrays["event_field_row"])
        cols = np.asarray(arrays["event_field_col"])
        values = np.asarray(arrays["event_field_nonzero"])
        item_set = set(self.item_event_indices.tolist())
        numeric = {
            (int(row), int(col)): int(value)
            for row, col, value in zip(rows.tolist(), cols.tolist(), values.tolist())
            if int(row) in item_set and int(col) < len(ITEM_PAYLOAD_FIELDS)
        }
        raw = _raw_values(
            np.asarray(arrays.get("event_raw_rows", np.empty(0, dtype=np.int32))),
            np.asarray(arrays.get("event_raw_cols", np.empty(0, dtype=np.int16))),
            np.asarray(arrays.get("event_raw_offsets", np.zeros(1, dtype=np.int64))),
            np.asarray(arrays.get("event_raw_values", np.empty(0, dtype=np.uint8))),
            item_set,
        )
        return numeric, raw

    def item_events(self) -> tuple[ItemEvent, ...]:
        if self._item_event_cache is not None:
            return self._item_event_cache
        arrays = self.arrays
        timestamps = np.asarray(arrays["event_timestamp"])
        source_indices = np.asarray(arrays["event_source_index"])
        type_codes = np.asarray(arrays["event_type_code"])
        masks = np.asarray(arrays["event_field_valid_mask"])
        numeric, raw = self._field_maps()
        explicit_rows = np.asarray(arrays["event_explicit_rows"], dtype=np.int64)
        explicit_codes = np.asarray(arrays["event_explicit_identity"], dtype=np.uint64)
        explicit = {int(row): int(code) for row, code in zip(explicit_rows.tolist(), explicit_codes.tolist())}
        identity_raw = _identity_values(
            np.asarray(arrays["event_identity_raw_rows"], dtype=np.int64),
            np.asarray(arrays["event_identity_raw_offsets"], dtype=np.int64),
            np.asarray(arrays["event_identity_raw_values"], dtype=np.uint8),
        )
        events: list[ItemEvent] = []
        for event_index in self.item_event_indices.tolist():
            event_index = int(event_index)
            source_ordinal = int(source_indices[event_index])
            source_ordinal_valid = source_ordinal >= 0
            explicit_code = explicit.get(event_index)
            source_raw = identity_raw.get(event_index)
            if explicit_code is None:
                source_identity: Any = f"ordinal:{source_ordinal if source_ordinal_valid else event_index}"
                identity_valid = source_ordinal_valid
                identity_bytes = None
            elif source_raw is None or not source_raw[1] or source_raw[0] is None:
                source_identity = f"invalid-source:{event_index}"
                identity_valid = False
                identity_bytes = source_raw[2] if source_raw is not None else None
            else:
                source_identity = source_raw[0]
                identity_valid = source_ordinal_valid
                identity_bytes = source_raw[2]
            values: list[Any] = []
            presence: list[bool] = []
            validity: list[bool] = []
            raw_fields: dict[str, Any] = {}
            raw_field_validity: dict[str, bool] = {}
            raw_field_bytes: dict[str, bytes] = {}
            mask = int(masks[event_index])
            for col, name in enumerate(ITEM_PAYLOAD_FIELDS):
                marker = raw.get((event_index, col))
                if marker is not None:
                    value, raw_valid, raw_bytes = marker
                    values.append(value)
                    presence.append(True)
                    validity.append(False)
                    raw_fields[name] = value
                    raw_field_validity[name] = raw_valid
                    raw_field_bytes[name] = raw_bytes
                elif mask & (1 << col):
                    values.append(numeric.get((event_index, col), 0))
                    presence.append(True)
                    validity.append(True)
                else:
                    values.append(None)
                    presence.append(False)
                    validity.append(False)
            events.append(ItemEvent(
                event_index=event_index,
                timestamp_ms=int(timestamps[event_index]),
                source_ordinal=source_ordinal if source_ordinal_valid else event_index,
                source_identity=source_identity,
                source_identity_valid=identity_valid,
                source_identity_raw=identity_bytes,
                event_type=ITEM_TYPE_CODES[int(type_codes[event_index])],
                participant_id=values[0], item_id=values[1], before_id=values[2], after_id=values[3],
                field_presence=tuple(presence), field_validity=tuple(validity), raw_fields=raw_fields,
                raw_field_validity=raw_field_validity, raw_field_bytes=raw_field_bytes,
                source_ordinal_valid=source_ordinal_valid,
                duplicate_source_identity=not identity_valid,
            ))
        self._item_event_cache = tuple(events)
        return self._item_event_cache

    def item_transactions(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.to_mapping() for event in self.item_events())

    def item_transaction_objects(self) -> tuple[Any, ...]:
        return tuple(event.to_transaction() for event in self.item_events())

    def nominal_query_refs(self) -> np.ndarray:
        nominal = np.asarray(self.v_nominal, dtype=np.int64)
        return nominal[:, 6].copy()


def _load_selected_bank(directory: Path, layout: Mapping[str, Any]) -> dict[str, np.ndarray]:
    scalar_fields = set(layout.get("scalar_fields", ()))
    match_fields = set(layout.get("match_fields", ()))
    names = set(CORE_MATCH_FIELDS) | set(NODE_AUDIT_FIELDS)
    names = {name for name in names if name in match_fields}
    for name in tuple(names):
        names.add("offset_" + name)
    with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
        missing = sorted(name for name in names if name not in archive)
        if missing:
            raise SourceValidationError(f"BANK.npz is missing selected members: {', '.join(missing)}")
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    offsets = {name: arrays["offset_" + name] for name in match_fields if "offset_" + name in arrays}
    n_matches = None
    for name, value in offsets.items():
        if n_matches is None:
            n_matches = len(value) - 1
        if len(value) - 1 != n_matches:
            raise SourceValidationError("shard offsets disagree on match cardinality")
        _validate_offsets(value, name="offset_" + name, n_matches=n_matches, n_values=len(arrays[name]))
        if name in scalar_fields:
            widths = np.diff(value)
            if np.any(widths != 1):
                raise SourceValidationError(f"scalar bank field {name} has non-scalar match slices")
    if n_matches is None:
        raise SourceValidationError("shard layout contains no selected match fields")
    local_scalar = [name for name in scalar_fields if name in names]
    local_match = [name for name in match_fields if name in names]
    arrays["__local_layout__"] = np.asarray([], dtype=np.uint8)
    arrays["__scalar_fields__"] = np.asarray(local_scalar, dtype=object)
    arrays["__match_fields__"] = np.asarray(local_match, dtype=object)
    return arrays


def _read_match_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SourceValidationError(f"cannot read {path}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceValidationError(f"invalid match ledger JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise SourceValidationError(f"match ledger row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _local_view(arrays: Mapping[str, np.ndarray], layout: Mapping[str, Any], local_index: int) -> dict[str, np.ndarray]:
    scalar_fields = [str(value) for value in np.asarray(arrays["__scalar_fields__"], dtype=object).tolist()]
    match_fields = [str(value) for value in np.asarray(arrays["__match_fields__"], dtype=object).tolist()]
    return match_view(arrays, {"scalar_fields": scalar_fields, "match_fields": match_fields}, local_index)


@dataclass
class AcceptedSourceReader:
    """Stream validated matches from the accepted source bank."""

    source_root: Path = DEFAULT_SOURCE_ROOT
    preparation_root: Path | None = DEFAULT_PREPARATION_ROOT
    verify_hashes: bool = True
    expected_links: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root)
        self.preparation_root = Path(self.preparation_root) if self.preparation_root is not None else None
        if self.expected_links is None and self.preparation_root is not None:
            links_path = self.preparation_root / "SOURCE_SHARD_LINKS.json"
            if links_path.exists():
                links = _json(links_path)
                self.expected_links = tuple(links.get("shards", ()))
        if self.expected_links is not None:
            self.expected_links = tuple(self.expected_links)

    def verify_source_bindings(self) -> dict[str, Any]:
        if self.expected_links is None:
            directories = sorted(path.name for path in self.source_root.iterdir() if path.is_dir() and (path / "BANK.npz").exists())
            records = [{"directory": name} for name in directories]
        else:
            records = [dict(row) for row in self.expected_links]
        if len({row.get("directory") for row in records}) != len(records):
            raise SourceValidationError("source shard bindings contain duplicate directories")
        actual_directories = {
            path.name for path in self.source_root.iterdir()
            if path.is_dir() and (path / "BANK.npz").is_file()
        }
        expected_directories = {str(row.get("directory")) for row in records}
        if actual_directories != expected_directories:
            raise SourceValidationError("source shard directory set differs from accepted preparation bindings")
        actual: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda row: str(row.get("directory", ""))):
            directory_name = record.get("directory")
            if not isinstance(directory_name, str) or not directory_name:
                raise SourceValidationError("source shard binding has no directory")
            directory = self.source_root / directory_name
            file_names = {
                "archive_sha256": "BANK.npz",
                "layout_sha256": "LAYOUT.json",
                "shard_manifest_sha256": "SHARD_MANIFEST.json",
                "match_ledger_sha256": "MATCH_LEDGER.jsonl",
                "static_repair_sha256": "STATIC_REPAIR.json",
            }
            actual_record = {"directory": directory_name}
            for binding_name, filename in file_names.items():
                path = directory / filename
                if not path.is_file():
                    raise SourceValidationError(f"missing source shard file: {path}")
                expected = record.get(binding_name)
                if self.verify_hashes:
                    digest = file_sha256(path)
                    actual_record[binding_name] = digest
                    if expected is not None and digest != expected:
                        raise SourceValidationError(f"source shard hash changed: {path}")
                else:
                    actual_record[binding_name] = expected
            actual.append(actual_record)
        return {
            "source_root": str(self.source_root),
            "n_shards": len(actual),
            "records": actual,
            "records_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }

    def _records(self) -> list[dict[str, Any]]:
        if self.expected_links is not None:
            return sorted((dict(row) for row in self.expected_links), key=lambda row: str(row["directory"]))
        names = sorted(path.name for path in self.source_root.iterdir() if path.is_dir() and (path / "BANK.npz").exists())
        return [{"directory": name} for name in names]

    def iter_matches(self, *, max_matches: int | None = None) -> Iterator[AcceptedMatch]:
        if max_matches is not None and max_matches < 0:
            raise ValueError("max_matches must be nonnegative")
        yielded = 0
        if self.verify_hashes:
            self.verify_source_bindings()
        for shard_index, record in enumerate(self._records()):
            directory = self.source_root / str(record["directory"])
            layout = _json(directory / "LAYOUT.json")
            ledgers = _read_match_ledger(directory / "MATCH_LEDGER.jsonl")
            selected = _load_selected_bank(directory, layout)
            try:
                n_matches = len(np.asarray(selected["offset_event_timestamp"])) - 1
                if n_matches != len(ledgers):
                    raise SourceValidationError(f"match ledger and bank cardinality differ in {directory.name}")
                for local_index, metadata in enumerate(ledgers):
                    view = _local_view(selected, layout, local_index)
                    yield AcceptedMatch(view, layout, metadata, shard_index, local_index)
                    yielded += 1
                    if max_matches is not None and yielded >= max_matches:
                        return
            finally:
                del selected

    def __iter__(self) -> Iterator[AcceptedMatch]:
        return self.iter_matches()


def load_accepted_source(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    preparation_root: Path | None = DEFAULT_PREPARATION_ROOT,
    verify_hashes: bool = True,
) -> AcceptedSourceReader:
    return AcceptedSourceReader(source_root, preparation_root, verify_hashes)


__all__ = [
    "AcceptedMatch", "AcceptedSourceReader", "CORE_MATCH_FIELDS", "DEFAULT_ACCEPTANCE_PATH",
    "DEFAULT_PREPARATION_ROOT", "DEFAULT_SOURCE_ROOT", "ITEM_PAYLOAD_FIELDS", "ITEM_TYPE_CODES",
    "ITEM_TYPE_NAMES", "ItemEvent", "NODE_AUDIT_FIELDS", "SourceValidationError", "file_sha256",
    "load_accepted_source",
]
