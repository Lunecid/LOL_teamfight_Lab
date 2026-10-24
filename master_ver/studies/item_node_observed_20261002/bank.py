"""Streaming recipient-confirmed observed-item inputs.

The accepted observable bank contains item events as sparse payloads.  This
module turns one accepted match into compact, query-local observed ledgers
without reconstructing an actual starting inventory.  It deliberately keeps
the raw typed event history separate from the recipient-confirmed replay:
events whose recipient is not a strict participant ID in ``1..10`` remain
observable context and are never assigned to a node.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from studies.item_node_encoding_20261002.source import (
    AcceptedMatch,
    DEFAULT_PREPARATION_ROOT,
    DEFAULT_SOURCE_ROOT,
    ITEM_PAYLOAD_FIELDS,
    ITEM_TYPE_NAMES,
    SourceValidationError,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "item_node_observed_20261002"
CONTRACT_PATH = REPORT / "CONTRACT.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "item_node_observed_20261002" / "full_bank_20261002_1"
EXPECTED_MATCHES = 209_999
EXPECTED_QUERIES = 4_384_026
EXPECTED_NOMINAL_ROWS = 2_618_905
EXPECTED_SHARDS = 411
PARTICIPANTS = 10
CONTEXT_NAMES = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
SCOPES = ("fit", "stop", "refit", "dev", "cal", "target")

# The first 63 bits are stable across runs.  Unknown future replay reasons are
# retained in the JSON reason table and use the final ``other`` bit in the
# compact mask instead of silently disappearing.
REASON_BITS: dict[str, int] = {
    "duplicate_source_identity": 0,
    "duplicate_source_identity_flag": 1,
    "invalid_item_transaction_type": 2,
    "invalid_item_id": 3,
    "invalid_undo_payload": 4,
    "same_timestamp_block_not_order_independent": 5,
    "item_replay_underflow": 6,
    "invalid_participant_id": 7,
    "invalid_source_identity": 8,
    "other": 63,
}
UNASSIGNED_REASON_BITS: dict[str, int] = {
    "missing_participant": 0,
    "malformed_participant": 1,
    "zero_participant": 2,
    "out_of_range_participant": 3,
    "invalid_source_identity": 4,
    "other": 63,
}

MATCH_DTYPE = np.dtype([
    ("canonical_key", "S64"), ("shard", "<u2"), ("local_match", "<u2"),
    ("original_index", "<u4"), ("patch", "S8"), ("role", "S16"),
    ("context_membership", "u1"), ("first_query", "<i8"), ("n_queries", "<i4"),
    ("first_nominal", "<i8"), ("n_nominal", "<i4"),
    ("first_assigned_history", "<i8"), ("n_assigned_history", "<i4"),
    ("first_unassigned_history", "<i8"), ("n_unassigned_history", "<i4"),
])
QUERY_DTYPE = np.dtype([
    ("match_index", "<u4"), ("query_index", "<u4"), ("query_ms", "<i8"), ("event_limit", "<i8"),
    ("node_offset_row", "<i8"), ("assigned_history_end", "<i8"),
    ("unassigned_history_end", "<i8"), ("unassigned_observed", "?"),
    ("unassigned_reason_mask", "<u8"),
])
NOMINAL_DTYPE = np.dtype([
    ("match_index", "<u4"), ("nominal_index", "<u4"), ("query_ref", "<i4"),
    ("supported", "?"), ("actual_ms", "<i8"), ("nominal_ms", "<i8"),
])
FORMAT_VERSION = "observed_item_bank_v3_20261002"


class ObservedBankError(SourceValidationError):
    """Raised when observed-bank inputs or persisted outputs are invalid."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _identity_key(value: Any) -> tuple[str, str, Any]:
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__qualname__, repr(value))
    return ("value", type(value).__qualname__, value)


def _strict_participant(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    value = int(value)
    return value if 1 <= value <= PARTICIPANTS else None


def _unassigned_reason(value: Any) -> str:
    if value is None:
        return "missing_participant"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return "malformed_participant"
    value = int(value)
    if value == 0:
        return "zero_participant"
    if value < 1 or value > PARTICIPANTS:
        return "out_of_range_participant"
    return "other"


def _mask(reasons: Iterable[str], bits: Mapping[str, int]) -> int:
    result = 0
    for reason in reasons:
        result |= 1 << int(bits.get(str(reason), bits["other"]))
    return result


@dataclass(frozen=True)
class ObservedEvent:
    """One typed source event retained in its original audit form."""

    event_index: int
    timestamp_ms: int
    source_ordinal: int
    source_identity: Any
    source_identity_valid: bool
    event_type: str
    participant_id: Any = None
    item_id: Any = None
    before_id: Any = None
    after_id: Any = None
    field_presence: tuple[bool, bool, bool, bool] = (True, True, False, False)
    raw_fields: Mapping[str, Any] = field(default_factory=dict, repr=False)
    duplicate_source_identity: bool = False
    raw_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)
    unassigned_reason: str | None = None

    @property
    def recipient_id(self) -> int | None:
        return _strict_participant(self.participant_id)

    @property
    def assigned(self) -> bool:
        return self.recipient_id is not None

    @property
    def history_reason(self) -> str | None:
        if self.assigned:
            return "invalid_source_identity" if not self.source_identity_valid else None
        return self.unassigned_reason or _unassigned_reason(self.participant_id)

    def transaction_mapping(self) -> dict[str, Any]:
        """Return the mapping accepted by the versioned item ledger."""
        result = {
            "timestamp_ms": int(self.timestamp_ms),
            "source_ordinal": int(self.source_ordinal),
            "source_identity": _json_value(self.source_identity),
            "type": str(self.event_type),
            "duplicate_source_identity": bool(self.duplicate_source_identity),
            "raw_itemId": _json_value(self.item_id),
            "raw_payload": _json_value(self.raw_payload),
        }
        names = ("participantId", "itemId", "beforeId", "afterId")
        values = (self.participant_id, self.item_id, self.before_id, self.after_id)
        for index, (name, value) in enumerate(zip(names, values)):
            if self.field_presence[index]:
                # Keep the raw recipient value.  The v3 ledger performs the
                # strict recipient partition itself; replacing 0/malformed
                # values with ``None`` would destroy source provenance.
                result[name] = _json_value(self.raw_fields.get(name, value))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": int(self.event_index),
            "timestamp_ms": int(self.timestamp_ms),
            "source_ordinal": int(self.source_ordinal),
            "source_identity": _json_value(self.source_identity),
            "source_identity_valid": bool(self.source_identity_valid),
            "event_type": str(self.event_type),
            "participant_id": _json_value(self.participant_id),
            "item_id": _json_value(self.item_id),
            "before_id": _json_value(self.before_id),
            "after_id": _json_value(self.after_id),
            "field_presence": [bool(value) for value in self.field_presence],
            "duplicate_source_identity": bool(self.duplicate_source_identity),
            "unassigned_reason": self.unassigned_reason,
            "raw_payload": _json_value(self.raw_payload),
        }


@dataclass(frozen=True)
class ObservedQueryState:
    """Compact current observed ledgers and lazy history references at one query."""

    query_timestamp_ms: int
    query_event_limit: int
    item_ids: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    supported: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    reason_mask: np.ndarray
    unassigned_observed: bool
    unassigned_reason_mask: int
    assigned_history_end: int
    unassigned_history_end: int
    actual_inventory_complete: str = "unknown"

    @property
    def known(self) -> np.ndarray:
        """Compatibility spelling used by the frozen direct ledger audit."""
        return self.supported

    @property
    def recorded_ledger_supported(self) -> np.ndarray:
        return self.supported


@dataclass(frozen=True)
class ObservedMatch:
    """All query endpoints for one match; histories remain event references."""

    metadata: Mapping[str, Any]
    queries: np.ndarray
    query_event_limits: np.ndarray
    states: tuple[ObservedQueryState, ...]
    assigned_events: tuple[ObservedEvent, ...]
    unassigned_events: tuple[ObservedEvent, ...]
    nominal_query_refs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    nominal_supported: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    context_membership: int = 0
    context_memberships: tuple[int, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def history_prefix(self, query_index: int) -> dict[str, tuple[dict[str, Any], ...]]:
        index = int(query_index)
        if index < 0 or index >= len(self.states):
            raise IndexError("query_index is outside this match")
        state = self.states[index]
        return {
            "assigned": tuple(event.to_dict() for event in self.assigned_events[:state.assigned_history_end]),
            "unassigned": tuple(event.to_dict() for event in self.unassigned_events[:state.unassigned_history_end]),
        }

    def iter_history_prefix(self, query_index: int) -> Iterator[ObservedEvent]:
        index = int(query_index)
        if index < 0 or index >= len(self.states):
            raise IndexError("query_index is outside this match")
        state = self.states[index]
        yield from self.assigned_events[:state.assigned_history_end]
        yield from self.unassigned_events[:state.unassigned_history_end]


def _event_from_value(value: Any, index: int) -> ObservedEvent:
    if hasattr(value, "event_index") and hasattr(value, "event_type"):
        raw = dict(getattr(value, "raw_fields", {}) or {})
        raw_payload = dict(getattr(value, "raw_payload", {}) or {})
        if not raw_payload:
            # ``ItemEvent`` keeps malformed and explicit-null values in
            # ``raw_fields`` even when its typed attributes are ``None``.
            # Preserve those markers when the v3 ledger receives the event.
            raw_payload = dict(raw)
            if not raw_payload:
                raw_payload = {
                    name: getattr(value, name.replace("Id", "_id"), None)
                    for name in ITEM_PAYLOAD_FIELDS
                    if getattr(value, name.replace("Id", "_id"), None) is not None
                }
        return ObservedEvent(
            event_index=int(getattr(value, "event_index", index)),
            timestamp_ms=int(getattr(value, "timestamp_ms")),
            source_ordinal=int(getattr(value, "source_ordinal", index)),
            source_identity=getattr(value, "source_identity", f"ordinal:{index}"),
            source_identity_valid=bool(getattr(value, "source_identity_valid", True)),
            event_type=str(getattr(value, "event_type", "")),
            participant_id=getattr(value, "participant_id", None),
            item_id=getattr(value, "item_id", None),
            before_id=getattr(value, "before_id", None),
            after_id=getattr(value, "after_id", None),
            field_presence=tuple(bool(item) for item in getattr(value, "field_presence", (True, True, False, False))),
            raw_fields=raw,
            duplicate_source_identity=bool(getattr(value, "duplicate_source_identity", False)),
            raw_payload=raw_payload or raw,
        )
    if not isinstance(value, Mapping):
        raise TypeError("events must be ItemEvent values or mappings")
    def first(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in value:
                return value[name]
        return default
    event_type = first("event_type", "type", default="")
    present = tuple(any(name in value for name in aliases) for aliases in (
        ("participant_id", "participantId", "pid"),
        ("item_id", "itemId"), ("before_id", "beforeId"), ("after_id", "afterId"),
    ))
    supplied_presence = value.get("field_presence")
    if isinstance(supplied_presence, (tuple, list)) and len(supplied_presence) == 4:
        present = tuple(bool(item) for item in supplied_presence)
    raw_fields = dict(value)
    raw_payload_value = value.get("raw_payload")
    if isinstance(raw_payload_value, Mapping):
        raw_fields.update(raw_payload_value)
    return ObservedEvent(
        event_index=int(first("event_index", default=index)),
        timestamp_ms=int(first("timestamp_ms", "timestamp")),
        source_ordinal=int(first("source_ordinal", "source_index", "ordinal", default=index)),
        source_identity=first("source_identity", "source_id", "identity", default=f"ordinal:{index}"),
        source_identity_valid=bool(first("source_identity_valid", default=True)),
        event_type=str(event_type).strip().upper(),
        participant_id=first("participant_id", "participantId", "pid"),
        item_id=first("item_id", "itemId"),
        before_id=first("before_id", "beforeId"),
        after_id=first("after_id", "afterId"),
        field_presence=present,
        raw_fields=raw_fields,
        duplicate_source_identity=bool(first("duplicate_source_identity", "duplicate_identity", default=False)),
        raw_payload=dict(value),
    )


def partition_item_events(events: Iterable[Any]) -> tuple[tuple[ObservedEvent, ...], tuple[ObservedEvent, ...]]:
    """Split events into recipient-confirmed and unassigned histories.

    Duplicate source identities are detected chronologically over the union of
    both partitions.  Every event in a duplicate same-timestamp block is
    tainted, while an earlier endpoint remains unaffected by a later duplicate.
    """
    normalized = [_event_from_value(value, index) for index, value in enumerate(events)]
    normalized.sort(key=lambda event: (int(event.timestamp_ms), int(event.source_ordinal), int(event.event_index)))
    by_time: dict[int, list[ObservedEvent]] = defaultdict(list)
    for event in normalized:
        by_time[int(event.timestamp_ms)].append(event)
    seen: set[tuple[str, str, Any]] = set()
    assigned: list[ObservedEvent] = []
    unassigned: list[ObservedEvent] = []
    for timestamp in sorted(by_time):
        block = by_time[timestamp]
        counts: Counter[tuple[str, str, Any]] = Counter(_identity_key(event.source_identity) for event in block)
        for event in block:
            identity = _identity_key(event.source_identity)
            duplicate = bool(event.duplicate_source_identity or identity in seen or counts[identity] > 1)
            reason = None if event.recipient_id is not None else _unassigned_reason(event.participant_id)
            current = ObservedEvent(**{**event.__dict__, "duplicate_source_identity": duplicate, "unassigned_reason": reason})
            if current.recipient_id is None:
                unassigned.append(current)
            else:
                assigned.append(current)
        seen.update(counts)
    return tuple(assigned), tuple(unassigned)


def _ledger_class() -> type:
    from .ledger import RecipientConfirmedItemLedger
    return RecipientConfirmedItemLedger


def _build_ledger(events: Sequence[ObservedEvent]) -> Any:
    ledger_type = _ledger_class()
    return ledger_type(event.transaction_mapping() for event in events)


def _iter_replay_states(
    ledger: Any,
    queries: Sequence[int] | np.ndarray,
    assigned_events: Sequence[ObservedEvent],
    unassigned_events: Sequence[ObservedEvent],
    query_event_limits: Sequence[int] | np.ndarray | None = None,
) -> Iterator[ObservedQueryState]:
    values = [int(value) for value in np.asarray(queries).reshape(-1).tolist()]
    limits = [int(value) for value in (np.asarray(query_event_limits).reshape(-1).tolist() if query_event_limits is not None else [0] * len(values))]
    if len(limits) != len(values) or any(value < 0 for value in limits):
        raise ObservedBankError("query event limits must align with source queries")
    if any(value < 0 for value in values) or any(left > right for left, right in zip(values, values[1:])):
        raise ValueError("queries must be nonnegative and sorted")
    assigned_timestamps = [int(event.timestamp_ms) for event in assigned_events]
    unassigned_timestamps = [int(event.timestamp_ms) for event in unassigned_events]
    # The companion v3 ledger owns replay semantics.  In particular, this
    # call is what applies chronological duplicate taints across both the
    # identified and unassigned partitions while keeping unknown-owner rows
    # out of node bags.
    compact_states = ledger.iter_compact_observed_states(values)
    for state, event_limit in zip(compact_states, limits):
        query = int(state["query_timestamp_ms"])
        supported = np.asarray(state.get("observed_ledger_supported", state.get("recorded_ledger_supported", state["known"])), dtype=bool)
        reason_rows = tuple(tuple(str(reason) for reason in row) for row in state.get("reasons", ()))
        if len(reason_rows) != PARTICIPANTS:
            raise ObservedBankError("v3 ledger returned an invalid reason matrix")
        reason_masks = np.asarray([_mask(row, REASON_BITS) for row in reason_rows], dtype=np.uint64)
        item_ids = np.asarray(state["item_ids"], dtype=np.int64)
        item_counts = np.asarray(state["counts"], dtype=np.int64)
        offsets = np.asarray(state["offsets"], dtype=np.int64)
        unassigned_end = int(state.get("unassigned_count", bisect_right(unassigned_timestamps, query)))
        if unassigned_end < 0 or unassigned_end > len(unassigned_events):
            raise ObservedBankError("v3 ledger returned an invalid unassigned history reference")
        assigned_end = bisect_right(assigned_timestamps, query)
        unassigned_reasons = [event.history_reason for event in unassigned_events[:unassigned_end] if event.history_reason]
        yield ObservedQueryState(
            query_timestamp_ms=int(query),
            query_event_limit=int(event_limit),
            item_ids=item_ids,
            counts=item_counts,
            offsets=offsets,
            supported=supported,
            reasons=reason_rows,
            reason_mask=reason_masks,
            unassigned_observed=bool(unassigned_end),
            unassigned_reason_mask=_mask(unassigned_reasons, UNASSIGNED_REASON_BITS),
            assigned_history_end=int(assigned_end),
            unassigned_history_end=int(unassigned_end),
            actual_inventory_complete=str(state.get("full_inventory_complete", state.get("actual_inventory_complete", "unknown"))),
        )


def _source_metadata(match: AcceptedMatch) -> dict[str, Any]:
    metadata = dict(match.metadata)
    key = metadata.get("canonical_match_key")
    if isinstance(key, bytes):
        key = key.decode("ascii")
    return {
        "canonical_match_key": str(key) if key is not None else None,
        "patch": str(metadata.get("patch")) if metadata.get("patch") is not None else None,
        "role": str(metadata.get("role")) if metadata.get("role") is not None else None,
        "original_index": int(metadata["original_index"]) if metadata.get("original_index") is not None else None,
        "shard": int(match.shard_index),
        "local_match": int(match.local_index),
    }


def _nominal_refs(nominal_rows: Any, n_queries: int, queries: Sequence[int] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if nominal_rows is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    rows = np.asarray(nominal_rows)
    if rows.dtype.names:
        refs = np.asarray(rows["query_ref"], dtype=np.int64)
        supported = np.asarray(rows["supported"], dtype=bool)
        actual = np.asarray(rows["actual_ms"], dtype=np.int64)
        nominal = np.asarray(rows["nominal_ms"], dtype=np.int64)
        if np.any(supported & ((refs < 0) | (refs >= n_queries))):
            raise ObservedBankError("nominal row query_ref is outside source query array")
        if np.any(supported & (actual != nominal)):
            raise ObservedBankError("supported nominal row actual_ms differs from nominal_ms")
        if np.any(~supported & (refs != -1)):
            raise ObservedBankError("unsupported nominal row retains a query reference")
        if queries is not None and np.any(supported & (np.asarray(queries, dtype=np.int64)[refs] != actual)):
            raise ObservedBankError("supported nominal row query_ref timestamp differs from source query_ms")
        return refs, supported
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ObservedBankError("nominal query references must be one-dimensional")
    supported = rows >= 0
    if np.any(rows[supported] >= n_queries):
        raise ObservedBankError("nominal query reference is outside source query array")
    return rows, supported


def observe_match(
    match: AcceptedMatch | Any,
    nominal_rows: Any = None,
    *,
    context_membership: int | Mapping[str, bool] = 0,
    context_memberships: Sequence[int] | None = None,
) -> ObservedMatch:
    """Replay one accepted match and retain lazy assigned/unassigned histories."""
    if hasattr(match, "item_events"):
        source_events = match.item_events()
        queries = np.asarray(match.query_ms, dtype=np.int64)
        if hasattr(match, "query_event_limit"):
            query_event_limits = np.asarray(match.query_event_limit, dtype=np.int64)
        else:
            event_times = np.asarray([
                int(event.get("timestamp_ms", event.get("timestamp"))) if isinstance(event, Mapping) else int(event.timestamp_ms)
                for event in source_events
            ], dtype=np.int64)
            query_event_limits = np.searchsorted(event_times, queries, side="right").astype(np.int64)
    else:
        source_events = tuple(match)
        queries = np.empty(0, dtype=np.int64)
        query_event_limits = np.empty(0, dtype=np.int64)
    if queries.ndim != 1 or (len(queries) > 1 and np.any(queries[1:] <= queries[:-1])) or np.any(queries < 0):
        raise ObservedBankError("source query_ms must be nonnegative and strictly increasing")
    assigned, unassigned = partition_item_events(source_events)
    all_events = tuple(sorted(assigned + unassigned, key=lambda event: (int(event.timestamp_ms), int(event.source_ordinal), int(event.event_index))))
    ledger = _build_ledger(all_events)
    if len(query_event_limits) != len(queries) or np.any(query_event_limits < 0):
        raise ObservedBankError("source query_event_limit is invalid")
    states = tuple(_iter_replay_states(ledger, queries, assigned, unassigned, query_event_limits))
    refs, nominal_supported = _nominal_refs(nominal_rows, len(queries), queries)
    if isinstance(context_membership, Mapping):
        membership = sum(1 << index for index, scope in enumerate(SCOPES) if bool(context_membership.get(scope, False)))
    else:
        membership = int(context_membership)
    if context_memberships is None:
        memberships = (membership,)
    else:
        memberships = tuple(int(value) for value in context_memberships)
        if len(memberships) != 6 or any(value < 0 or value > 63 for value in memberships):
            raise ObservedBankError("context_memberships must contain six u1 masks")
        membership = memberships[0]
    metadata = _source_metadata(match) if hasattr(match, "metadata") else {}
    event_type_counts = Counter(event.event_type for event in assigned + unassigned)
    item_id_counts: Counter[str] = Counter()
    for event in assigned + unassigned:
        for value in (event.item_id, event.before_id, event.after_id):
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and int(value) > 0:
                item_id_counts[str(int(value))] += 1
    stats = {
        "n_item_events": len(assigned) + len(unassigned),
        "n_assigned_events": len(assigned),
        "n_unassigned_events": len(unassigned),
        "n_unique_queries": len(queries),
        "n_supported_player_queries": int(sum(int(state.supported.sum()) for state in states)),
        "n_unassigned_observed_queries": int(sum(int(state.unassigned_observed) for state in states)),
        "support_by_participant": [int(sum(bool(state.supported[row]) for state in states)) for row in range(PARTICIPANTS)],
        "reason_counts": dict(sorted(Counter(reason for state in states for row in state.reasons for reason in row).items())),
        "unassigned_reason_counts": dict(sorted(Counter(event.history_reason for event in unassigned if event.history_reason).items())),
        "item_event_counts": dict(sorted(event_type_counts.items())),
        "item_id_event_counts": dict(sorted(item_id_counts.items(), key=lambda row: int(row[0]))),
        "actual_inventory_complete": "unknown",
        "initial_state": "initial_empty_recorded_transaction_ledger_assumption",
    }
    return ObservedMatch(
        metadata=metadata,
        queries=queries,
        query_event_limits=query_event_limits,
        states=states,
        assigned_events=assigned,
        unassigned_events=unassigned,
        nominal_query_refs=refs,
        nominal_supported=nominal_supported,
        context_membership=membership,
        context_memberships=memberships,
        stats=stats,
    )


def iter_observed_matches(reader: Iterable[AcceptedMatch], *, max_matches: int | None = None) -> Iterator[ObservedMatch]:
    """Yield observed matches from an accepted source reader."""
    if max_matches is not None and max_matches < 0:
        raise ValueError("max_matches must be nonnegative")
    for index, match in enumerate(reader):
        if max_matches is not None and index >= max_matches:
            return
        yield observe_match(match)


def file_digest(path: Path) -> str:
    return file_sha256(Path(path))


class _SourceReferenceResolver:
    """Resolve compact event references back to one or two source shards.

    The observed bank stores only source event indices.  A small LRU keeps the
    source decoder bounded while retaining the accepted source reader's exact
    event and field-presence semantics for public lazy snapshots.
    """

    def __init__(
        self, metadata: Mapping[str, Any], *, source_bindings: Sequence[Mapping[str, Any]],
        verify_hashes: bool, max_cached_shards: int = 2,
    ) -> None:
        source_value = metadata.get("source_root")
        if not isinstance(source_value, str) or not source_value:
            raise ObservedBankError("bank metadata has no source_root for event references")
        self.source_root = Path(source_value)
        preparation_value = metadata.get("preparation_root")
        self.preparation_root = Path(preparation_value) if isinstance(preparation_value, str) and preparation_value else None
        if not self.source_root.is_dir():
            raise ObservedBankError(f"source_root for event references is unavailable: {self.source_root}")
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.verify_hashes = bool(verify_hashes)
        self._bindings = tuple(dict(row) for row in source_bindings)
        if not self._bindings or any(not isinstance(row.get("directory"), str) for row in self._bindings):
            raise ObservedBankError("SOURCE_BINDINGS.json has no complete source directory order")
        self._directories = self._read_directories()
        self._cache: OrderedDict[int, tuple[Mapping[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]] = OrderedDict()
        self._event_cache: OrderedDict[tuple[int, int], dict[int, ObservedEvent]] = OrderedDict()

    def _read_directories(self) -> tuple[str, ...]:
        candidates = [str(row["directory"]) for row in self._bindings]
        if not candidates or len(set(candidates)) != len(candidates):
            raise ObservedBankError("source event reference shard order is unavailable")
        return tuple(candidates)

    def _load_shard(self, shard_index: int) -> tuple[Mapping[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
        index = int(shard_index)
        if index < 0 or index >= len(self._directories):
            raise ObservedBankError(f"source shard reference is outside accepted source coverage: {index}")
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        # These are the frozen source reader's selected-bank and match-view
        # helpers; no source reconstruction or alternate replay is introduced.
        from studies.item_node_encoding_20261002.source import (
            _json as source_json,
            _load_selected_bank,
            _local_view,
            _read_match_ledger,
            AcceptedMatch,
        )
        directory = self.source_root / self._directories[index]
        expected = self._bindings[index]
        filenames = {
            "archive_sha256": "BANK.npz", "layout_sha256": "LAYOUT.json",
            "shard_manifest_sha256": "SHARD_MANIFEST.json", "match_ledger_sha256": "MATCH_LEDGER.jsonl",
            "static_repair_sha256": "STATIC_REPAIR.json",
        }
        for binding, filename in filenames.items():
            path = directory / filename
            expected_hash = expected.get(binding)
            if not path.is_file() or not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ObservedBankError(f"source reference shard binding is incomplete: {path}")
            if self.verify_hashes and file_sha256(path) != expected_hash:
                raise ObservedBankError(f"source reference shard hash differs: {path}")
        layout = source_json(directory / "LAYOUT.json")
        selected = _load_selected_bank(directory, layout)
        ledgers = _read_match_ledger(directory / "MATCH_LEDGER.jsonl")
        if len(np.asarray(selected["offset_event_timestamp"])) - 1 != len(ledgers):
            raise ObservedBankError(f"source shard cardinality differs while resolving references: {directory}")
        cached = (layout, selected, ledgers)
        self._cache[index] = cached
        self._cache.move_to_end(index)
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return cached

    def events_for(self, shard_index: int, local_index: int) -> dict[int, ObservedEvent]:
        cache_key = (int(shard_index), int(local_index))
        cached_events = self._event_cache.get(cache_key)
        if cached_events is not None:
            self._event_cache.move_to_end(cache_key)
            return cached_events
        layout, selected, ledgers = self._load_shard(shard_index)
        local = int(local_index)
        if local < 0 or local >= len(ledgers):
            raise ObservedBankError("source match reference is outside shard coverage")
        from studies.item_node_encoding_20261002.source import _local_view, AcceptedMatch
        view = _local_view(selected, layout, local)
        match = AcceptedMatch(view, layout, ledgers[local], int(shard_index), local)
        events = {
            int(event.event_index): _event_from_value(event, int(event.event_index))
            for event in match.item_events()
        }
        self._event_cache[cache_key] = events
        self._event_cache.move_to_end(cache_key)
        while len(self._event_cache) > self.max_cached_shards:
            self._event_cache.popitem(last=False)
        return events


class ObservedBankLoader:
    """Read persisted query inputs and source histories without prefix copies."""

    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / "RUN_MANIFEST.json"
        metadata_path = self.root / "BANK_METADATA.json"
        if not manifest_path.is_file() or not metadata_path.is_file():
            raise ObservedBankError("observed bank is missing RUN_MANIFEST.json or BANK_METADATA.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.verify_hashes = bool(verify_hashes)
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ObservedBankError("unsupported observed bank format")
        self.matches = np.load(self.root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.queries = np.load(self.root / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.node_offsets = np.load(self.root / "NODE_OFFSETS.npy", mmap_mode="r", allow_pickle=False)
        self.support = np.load(self.root / "NODE_SUPPORT.npy", mmap_mode="r", allow_pickle=False)
        self.reason_mask = np.load(self.root / "NODE_REASON_MASK.npy", mmap_mode="r", allow_pickle=False)
        self.nominal = np.load(self.root / "NOMINAL_TABLE.npy", mmap_mode="r", allow_pickle=False)
        contexts_path = self.root / "CONTEXT_MEMBERSHIP.npy"
        self.contexts = np.load(contexts_path, mmap_mode="r", allow_pickle=False) if contexts_path.is_file() else None
        if self.contexts is not None and self.contexts.shape != (6, len(self.matches)):
            raise ObservedBankError("persisted context membership shape differs from match coverage")
        self._mapped_arrays = tuple(value for value in (
            self.matches, self.queries, self.node_offsets, self.support, self.reason_mask, self.nominal, self.contexts
        ) if value is not None)
        expected_shapes = {
            "matches": (len(self.matches),), "queries": (len(self.queries),),
            "node_offsets": (len(self.queries), PARTICIPANTS + 1),
            "support": (len(self.queries), PARTICIPANTS),
            "reason_mask": (len(self.queries), PARTICIPANTS),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ObservedBankError(f"{name} shape differs from query coverage")
        if self.node_offsets.dtype.kind not in "iu" or np.any(self.node_offsets < 0):
            raise ObservedBankError("NODE_OFFSETS.npy is invalid")
        self._ids = (self.root / "ITEM_IDS.bin").open("rb")
        self._counts = (self.root / "ITEM_COUNTS.bin").open("rb")
        def load_source_index(name: str) -> np.ndarray:
            path = self.root / name
            if not path.is_file() or path.stat().st_size % 4:
                raise ObservedBankError(f"source event reference file is missing or truncated: {path}")
            if path.stat().st_size == 0:
                return np.empty(0, dtype="<i4")
            return np.memmap(path, dtype="<i4", mode="r")
        self._assigned_sources = load_source_index("EVENT_SOURCE_INDEX_ASSIGNED.bin")
        self._unassigned_sources = load_source_index("EVENT_SOURCE_INDEX_UNASSIGNED.bin")
        self._source_resolver: _SourceReferenceResolver | None = None
        bindings_path = self.root / "SOURCE_BINDINGS.json"
        self._source_bindings: tuple[Mapping[str, Any], ...] | None = None
        if bindings_path.is_file():
            bindings_payload = json.loads(bindings_path.read_text(encoding="utf-8"))
            binding_rows = bindings_payload.get("records") if isinstance(bindings_payload, Mapping) else None
            if isinstance(binding_rows, list):
                self._source_bindings = tuple(dict(row) for row in binding_rows if isinstance(row, Mapping))
        if verify_hashes:
            for name, record in self.manifest.get("saved_files", {}).items():
                path = self.root / name
                if not path.is_file() or not isinstance(record, Mapping) or file_sha256(path) != record.get("sha256"):
                    raise ObservedBankError(f"persisted bank artifact hash differs: {path}")

    def close(self) -> None:
        for handle in (self._ids, self._counts):
            handle.close()
        for source_refs in (self._assigned_sources, self._unassigned_sources):
            mmap = getattr(source_refs, "_mmap", None)
            if mmap is not None:
                mmap.close()
        for array in self._mapped_arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "ObservedBankLoader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _read_int64(self, handle: Any, start: int, end: int) -> np.ndarray:
        if end < start:
            raise ObservedBankError("ragged item offsets are decreasing")
        handle.seek(int(start) * 8)
        raw = handle.read((int(end) - int(start)) * 8)
        if len(raw) != (int(end) - int(start)) * 8:
            raise ObservedBankError("ragged item binary member is truncated")
        return np.frombuffer(raw, dtype="<i8").copy()

    def query(self, query_index: int) -> dict[str, Any]:
        index = int(query_index)
        if index < 0 or index >= len(self.queries):
            raise IndexError("query_index is outside persisted bank")
        row = self.queries[index]
        global_offsets = np.asarray(self.node_offsets[index], dtype=np.int64)
        ids = self._read_int64(self._ids, int(global_offsets[0]), int(global_offsets[-1]))
        counts = self._read_int64(self._counts, int(global_offsets[0]), int(global_offsets[-1]))
        offsets = global_offsets - int(global_offsets[0])
        return {
            "query_ms": int(row["query_ms"]), "query_event_limit": int(row["event_limit"]), "item_ids": ids, "counts": counts,
            "offsets": offsets.copy(), "supported": np.asarray(self.support[index], dtype=bool).copy(),
            "known": np.asarray(self.support[index], dtype=bool).copy(),
            "recorded_ledger_supported": np.asarray(self.support[index], dtype=bool).copy(),
            "observed_ledger_supported": np.asarray(self.support[index], dtype=bool).copy(),
            "global_offsets": global_offsets.copy(),
            "reason_mask": np.asarray(self.reason_mask[index], dtype=np.uint64).copy(),
            "unassigned_observed": bool(row["unassigned_observed"]),
            "unassigned_reason_mask": int(row["unassigned_reason_mask"]),
            "assigned_history_end": int(row["assigned_history_end"]),
            "unassigned_history_end": int(row["unassigned_history_end"]),
        }

    def query_for_match(self, match_index: int, query_index: int) -> dict[str, Any]:
        match = self.matches[int(match_index)]
        local = int(query_index)
        if local < 0 or local >= int(match["n_queries"]):
            raise IndexError("query_index is outside match")
        return self.query(int(match["first_query"]) + local)

    def history_prefix(self, match_index: int, query_index: int) -> dict[str, tuple[dict[str, Any], ...]]:
        state = self.query_for_match(match_index, query_index)
        if self._source_resolver is None:
            if self._source_bindings is None:
                raise ObservedBankError("observed bank has no immutable SOURCE_BINDINGS.json for history references")
            if self.verify_hashes:
                binding_record = self.manifest.get("saved_files", {}).get("SOURCE_BINDINGS.json")
                if not isinstance(binding_record, Mapping) or file_sha256(self.root / "SOURCE_BINDINGS.json") != binding_record.get("sha256"):
                    raise ObservedBankError("SOURCE_BINDINGS.json is not hash-bound by RUN_MANIFEST.json")
            self._source_resolver = _SourceReferenceResolver(
                self.metadata, source_bindings=self._source_bindings, verify_hashes=self.verify_hashes,
            )
        match = self.matches[int(match_index)]
        events = self._source_resolver.events_for(int(match["shard"]), int(match["local_match"]))
        events_by_partition = {"assigned": events, "unassigned": events}
        refs_by_partition = {
            "assigned": self._assigned_sources,
            "unassigned": self._unassigned_sources,
        }
        result: dict[str, list[dict[str, Any]]] = {"assigned": [], "unassigned": []}
        for partition in result:
            end = int(state[f"{partition}_history_end"])
            first = int(match[f"first_{partition}_history"])
            count = int(match[f"n_{partition}_history"])
            if end < 0 or end > count:
                raise ObservedBankError("query history prefix exceeds persisted source reference count")
            refs = refs_by_partition[partition][first:first + end]
            if len(refs) != end:
                raise ObservedBankError("persisted source reference slice is truncated")
            for source_event_index in refs.tolist():
                event = events_by_partition[partition].get(int(source_event_index))
                if event is None:
                    raise ObservedBankError("persisted source event reference is absent from accepted source match")
                result[partition].append(event.to_dict())
        return {key: tuple(value) for key, value in result.items()}

    def node_input(self, match_index: int, query_index: int) -> dict[str, Any]:
        payload = self.query_for_match(match_index, query_index)
        history = self.history_prefix(match_index, query_index)
        payload["history"] = history
        from .ledger import RecipientConfirmedItemLedger
        source_events = []
        for partition in ("assigned", "unassigned"):
            for index, value in enumerate(history[partition]):
                source_events.append(_event_from_value(value, index).transaction_mapping())
        ledger = RecipientConfirmedItemLedger(source_events)
        snapshot = ledger.query(int(payload["query_ms"]))
        for key, expected in (
            ("item_ids", np.asarray(snapshot["item_ids"], dtype=np.int64)),
            ("counts", np.asarray(snapshot["counts"], dtype=np.int64)),
            ("offsets", np.asarray(snapshot["offsets"], dtype=np.int64)),
            ("observed_ledger_supported", np.asarray(snapshot["observed_ledger_supported"], dtype=bool)),
        ):
            if not np.array_equal(np.asarray(payload[key]), expected):
                raise ObservedBankError(f"persisted query differs from source-reference replay: {key}")
        payload.update({
            "snapshot": snapshot,
            "version": snapshot.get("version", ledger.version),
            "recorded_ledger_supported": snapshot["recorded_ledger_supported"],
            "observed_ledger_supported": snapshot["observed_ledger_supported"],
            "unknown": snapshot["unknown"],
            "assigned_history_context": snapshot["assigned_history_context"],
            "assigned_history": snapshot["assigned_history"],
            "assigned_history_by_participant": snapshot["assigned_history_by_participant"],
            "unassigned_history_context": snapshot["unassigned_history_context"],
            "unassigned_context": snapshot["unassigned_context"],
            "unassigned_history_observed": snapshot["unassigned_history_observed"],
            "has_unassigned_history": snapshot["has_unassigned_history"],
            "unassigned_count": snapshot["unassigned_count"],
            "full_inventory_complete": "unknown",
            "actual_inventory_complete": "unknown",
            "whole_actual_inventory": "unknown",
            "full_inventory_supported": False,
            "whole_inventory_supported": False,
            "full_inventory_verification": 0.0,
        })
        if self.contexts is not None:
            payload["context_memberships"] = tuple(int(value) for value in self.contexts[:, int(match_index)].tolist())
            payload["context_membership"] = int(self.matches[int(match_index)]["context_membership"])
        return payload


__all__ = [
    "CONTEXT_NAMES", "CONTRACT_PATH", "DEFAULT_OUTPUT", "DEFAULT_PREPARATION_ROOT",
    "DEFAULT_SOURCE_ROOT", "EXPECTED_MATCHES", "EXPECTED_NOMINAL_ROWS", "EXPECTED_QUERIES",
    "EXPECTED_SHARDS", "FORMAT_VERSION", "MATCH_DTYPE", "NOMINAL_DTYPE",
    "QUERY_DTYPE", "ObservedBankError", "ObservedBankLoader", "ObservedEvent", "ObservedMatch", "ObservedQueryState",
    "PARTICIPANTS", "REASON_BITS", "REPORT", "SCOPES", "UNASSIGNED_REASON_BITS",
    "file_digest", "iter_observed_matches", "observe_match", "partition_item_events",
]
