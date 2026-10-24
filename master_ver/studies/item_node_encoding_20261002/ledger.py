"""Direct, certifiable replay of recorded item transactions.

This module deliberately models only the transactions present in the source
event stream.  It does not infer starting inventory, automatic grants, item
transforms, or later consumption.  A same-millisecond block is accepted only
when every permutation has a non-negative replay from the block's starting
counts; otherwise the affected participant remains unknown for all later
queries.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


PARTICIPANTS = 10
ITEM_TRANSACTION_TYPES = frozenset(
    {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"}
)
INITIAL_EMPTY_PROVENANCE = "initial_empty_recorded_transaction_ledger_assumption"


def _strict_int(value: Any, *, name: str, nonnegative: bool = False) -> int:
    """Parse an integer source field without accepting booleans or fractions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _optional_int(value: Any, *, name: str, nonnegative: bool = False) -> Optional[int]:
    if value is None:
        return None
    return _strict_int(value, name=name, nonnegative=nonnegative)


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _identity_key(value: Any) -> tuple[str, str, Any]:
    """Create a stable, hashable identity key while retaining the raw value."""
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__qualname__, repr(value))
    return ("value", type(value).__qualname__, value)


@dataclass(frozen=True)
class ItemTransaction:
    """Lossless typed projection of one recorded item transaction."""

    timestamp_ms: int
    source_ordinal: int
    source_identity: Any
    participant_id: Optional[int]
    item_id: Optional[int]
    before_id: Optional[int]
    after_id: Optional[int]
    type: str
    duplicate_source_identity: bool = False
    raw_item_id: Any = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, (bool, np.bool_)) or not isinstance(self.timestamp_ms, (int, np.integer)):
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if int(self.timestamp_ms) < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        if isinstance(self.source_ordinal, (bool, np.bool_)) or not isinstance(self.source_ordinal, (int, np.integer)):
            raise ValueError("source_ordinal must be a nonnegative integer")
        if int(self.source_ordinal) < 0:
            raise ValueError("source_ordinal must be a nonnegative integer")
        object.__setattr__(self, "source_ordinal", int(self.source_ordinal))
        if self.source_identity is None:
            raise ValueError("source_identity is required for a typed transaction")
        if self.participant_id is not None and (
            isinstance(self.participant_id, (bool, np.bool_))
            or not isinstance(self.participant_id, (int, np.integer))
        ):
            raise ValueError("participant_id must be an integer or None")
        if self.participant_id is not None:
            object.__setattr__(self, "participant_id", int(self.participant_id))
        for name, value in (("item_id", self.item_id), ("before_id", self.before_id), ("after_id", self.after_id)):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
            if value is not None:
                object.__setattr__(self, name, int(value))
        if not isinstance(self.type, str):
            raise ValueError("type must be a string")
        object.__setattr__(self, "type", self.type.strip().upper())
        if not isinstance(self.duplicate_source_identity, (bool, np.bool_)):
            raise ValueError("duplicate_source_identity must be boolean")

    @property
    def participantId(self) -> Optional[int]:  # pragma: no cover - compatibility spelling
        return self.participant_id

    @property
    def itemId(self) -> Optional[int]:  # pragma: no cover - compatibility spelling
        return self.item_id

    def to_dict(self) -> dict[str, Any]:
        """Return an audit record retaining source identity and raw item ID."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "source_ordinal": self.source_ordinal,
            "source_identity": self.source_identity,
            "duplicate_source_identity": self.duplicate_source_identity,
            "participantId": self.participant_id,
            "itemId": self.item_id,
            "raw_itemId": self.raw_item_id,
            "beforeId": self.before_id,
            "afterId": self.after_id,
            "type": self.type,
        }


def _normalize_transaction(value: ItemTransaction | Mapping[str, Any], ordinal: int) -> ItemTransaction:
    if isinstance(value, ItemTransaction):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each transaction must be an ItemTransaction or mapping")

    timestamp = _strict_int(
        _first(value, "timestamp_ms", "timestamp"),
        name="timestamp_ms",
        nonnegative=True,
    )
    source_ordinal = _optional_int(
        _first(value, "source_ordinal", "source_index", "ordinal", default=ordinal),
        name="source_ordinal",
        nonnegative=True,
    )
    if source_ordinal is None:  # defensive; default above makes this unreachable
        source_ordinal = ordinal
    supplied_identity = _first(
        value,
        "source_identity",
        "source_id",
        "identity",
        default=None,
    )
    source_identity = (
        supplied_identity if supplied_identity is not None else f"ordinal:{source_ordinal}"
    )
    duplicate = _first(
        value,
        "duplicate_source_identity",
        "duplicate_identity",
        "duplicate",
        default=False,
    )
    if not isinstance(duplicate, (bool, np.bool_)):
        raise ValueError("duplicate_source_identity must be boolean")

    participant = _first(value, "participantId", "participant_id", "pid", default=None)
    try:
        participant_id = _optional_int(participant, name="participantId", nonnegative=True)
    except ValueError:
        participant_id = None
    item_raw = _first(value, "itemId", "item_id", default=None)
    before_raw = _first(value, "beforeId", "before_id", default=None)
    after_raw = _first(value, "afterId", "after_id", default=None)
    try:
        item_id = _optional_int(item_raw, name="itemId", nonnegative=True)
    except ValueError:
        item_id = None
    try:
        before_id = _optional_int(before_raw, name="beforeId", nonnegative=True)
    except ValueError:
        before_id = None
    try:
        after_id = _optional_int(after_raw, name="afterId", nonnegative=True)
    except ValueError:
        after_id = None
    event_type = _first(value, "type", "event_type", default="")
    event_type = event_type.strip().upper() if isinstance(event_type, str) else ""
    return ItemTransaction(
        timestamp_ms=timestamp,
        source_ordinal=int(source_ordinal),
        source_identity=source_identity,
        participant_id=participant_id,
        item_id=item_id,
        before_id=before_id,
        after_id=after_id,
        type=event_type,
        duplicate_source_identity=bool(duplicate),
        raw_item_id=item_raw,
        raw_payload=dict(value),
    )


@dataclass(frozen=True)
class _Block:
    timestamp_ms: int
    records: tuple[ItemTransaction, ...]
    deltas: Mapping[int, Mapping[int, int]]
    unknown_participants: Mapping[int, tuple[str, ...]]
    global_reasons: tuple[str, ...]


def _copy_counts(source: Mapping[int, Mapping[int, int]]) -> dict[int, dict[int, int]]:
    return {pid: dict(items) for pid, items in source.items()}


def _validate_initial_inventory(
    initial_inventory: Optional[Mapping[Any, Mapping[Any, Any]]],
) -> dict[int, dict[int, int]]:
    result = {pid: {} for pid in range(1, PARTICIPANTS + 1)}
    if initial_inventory is None:
        return result
    if not isinstance(initial_inventory, Mapping):
        raise TypeError("initial_inventory must map participant IDs to item counts")
    for raw_pid, items in initial_inventory.items():
        pid = _strict_int(raw_pid, name="initial participantId", nonnegative=True)
        if pid not in result:
            raise ValueError("initial participantId must be in 1..10")
        if not isinstance(items, Mapping):
            raise TypeError("each initial participant inventory must be a mapping")
        for raw_item, raw_count in items.items():
            item_id = _strict_int(raw_item, name="initial itemId", nonnegative=True)
            count = _strict_int(raw_count, name="initial item count", nonnegative=True)
            if item_id <= 0:
                raise ValueError("initial itemId must be positive")
            if count:
                result[pid][item_id] = count
    return result


def _record_reason(reasons: MutableMapping[int, list[str]], pid: int, reason: str) -> None:
    if reason not in reasons[pid]:
        reasons[pid].append(reason)


class ItemTransactionLedger:
    """Replay item transactions with same-millisecond certification.

    The constructor validates and processes sorted timestamp blocks once.  It
    stores sparse block deltas rather than a full ten-player snapshot per
    event.  ``query_many`` is the efficient path for sorted endpoint queries;
    ``query`` remains convenient for a single endpoint.
    """

    def __init__(
        self,
        transactions: Iterable[ItemTransaction | Mapping[str, Any]],
        *,
        initial_inventory: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
        initial_counts: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
    ) -> None:
        if initial_inventory is not None and initial_counts is not None:
            raise ValueError("provide only one of initial_inventory or initial_counts")
        if initial_counts is not None:
            initial_inventory = initial_counts
        normalized = [_normalize_transaction(value, i) for i, value in enumerate(transactions)]
        normalized.sort(key=lambda row: (row.timestamp_ms, row.source_ordinal))
        self._transactions = tuple(normalized)
        self._initial_counts = _validate_initial_inventory(initial_inventory)
        self.initial_inventory_provenance = (
            INITIAL_EMPTY_PROVENANCE if initial_inventory is None else "explicit_initial_inventory"
        )
        # This ledger cannot prove full actual inventory coverage from replay.
        # Completeness is an epistemic status, not a successful-replay bit.
        # A caller with independent coverage evidence may assess it later;
        # this ledger never infers that assessment.
        self.actual_inventory_complete = "unknown"
        self._blocks = self._build_blocks()
        self._block_timestamps = tuple(block.timestamp_ms for block in self._blocks)
        self._transaction_timestamps = tuple(row.timestamp_ms for row in self._transactions)

    @property
    def transactions(self) -> tuple[ItemTransaction, ...]:
        return self._transactions

    @property
    def blocks(self) -> tuple[_Block, ...]:
        return self._blocks

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_inventory_provenance,
            "actual_inventory_complete": "unknown",
            "recorded_transaction_ledger": True,
            "automatic_grants_or_transforms_inferred": False,
        }

    @staticmethod
    def _payload_reasons(row: ItemTransaction) -> list[str]:
        reasons: list[str] = []
        if row.type not in ITEM_TRANSACTION_TYPES:
            reasons.append("invalid_item_transaction_type")
            return reasons
        if row.type in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
            if row.item_id is None or row.item_id <= 0:
                reasons.append("invalid_item_id")
        else:
            if row.before_id is None or row.after_id is None:
                reasons.append("invalid_undo_payload")
            elif row.before_id <= 0 and row.after_id <= 0:
                reasons.append("invalid_undo_payload")
        return reasons

    @staticmethod
    def _delta(row: ItemTransaction) -> tuple[dict[int, int], dict[int, int]]:
        additions: dict[int, int] = defaultdict(int)
        removals: dict[int, int] = defaultdict(int)
        if row.type == "ITEM_PURCHASED":
            additions[int(row.item_id)] += 1  # payload was validated first
        elif row.type in {"ITEM_SOLD", "ITEM_DESTROYED"}:
            removals[int(row.item_id)] += 1
        elif row.type == "ITEM_UNDO":
            if row.before_id and row.before_id > 0:
                removals[int(row.before_id)] += 1
            if row.after_id and row.after_id > 0:
                additions[int(row.after_id)] += 1
        return dict(additions), dict(removals)

    def _build_blocks(self) -> tuple[_Block, ...]:
        by_timestamp: dict[int, list[ItemTransaction]] = defaultdict(list)
        for row in self._transactions:
            by_timestamp[row.timestamp_ms].append(row)
        working = _copy_counts(self._initial_counts)
        persistent_unknown = {pid: False for pid in range(1, PARTICIPANTS + 1)}
        persistent_reasons: dict[int, list[str]] = {pid: [] for pid in range(1, PARTICIPANTS + 1)}
        global_unknown = False
        global_reasons: list[str] = []
        seen_identities: set[tuple[str, str, Any]] = set()
        blocks: list[_Block] = []

        for timestamp in sorted(by_timestamp):
            records = tuple(sorted(by_timestamp[timestamp], key=lambda row: row.source_ordinal))
            identities_in_block: dict[tuple[str, str, Any], int] = defaultdict(int)
            for row in records:
                identities_in_block[_identity_key(row.source_identity)] += 1
            by_pid: dict[int, list[ItemTransaction]] = defaultdict(list)
            invalid_pid_records: list[ItemTransaction] = []
            for row in records:
                if row.participant_id is None or not (1 <= row.participant_id <= PARTICIPANTS):
                    invalid_pid_records.append(row)
                else:
                    by_pid[row.participant_id].append(row)

            block_global_reasons: list[str] = []
            if invalid_pid_records:
                block_global_reasons.append("invalid_participant_id")
            block_deltas: dict[int, dict[int, int]] = {}
            block_unknown: dict[int, tuple[str, ...]] = {}

            if block_global_reasons and not global_unknown:
                global_unknown = True
                for reason in block_global_reasons:
                    if reason not in global_reasons:
                        global_reasons.append(reason)
            # Once a global invalid participant is observed, no later event can
            # restore the all-player certainty state.  We still retain every
            # transaction in the block for audit output.
            if not global_unknown:
                for pid, pid_records in by_pid.items():
                    if persistent_unknown[pid]:
                        continue
                    reasons: list[str] = []
                    for row in pid_records:
                        identity_key = _identity_key(row.source_identity)
                        # A source identity is duplicate only when repeated in
                        # this atomic block or already observed in an earlier
                        # block.  Looking at the complete future stream would
                        # leak later evidence into an earlier query endpoint.
                        if identities_in_block[identity_key] > 1 or identity_key in seen_identities:
                            reasons.append("duplicate_source_identity")
                        if row.duplicate_source_identity:
                            reasons.append("duplicate_source_identity_flag")
                        reasons.extend(self._payload_reasons(row))
                    if reasons:
                        for reason in reasons:
                            _record_reason(persistent_reasons, pid, reason)
                        persistent_unknown[pid] = True
                        block_unknown[pid] = tuple(persistent_reasons[pid])
                        continue

                    additions: dict[int, int] = defaultdict(int)
                    removals: dict[int, int] = defaultdict(int)
                    for row in pid_records:
                        row_add, row_remove = self._delta(row)
                        for item_id, count in row_add.items():
                            additions[item_id] += count
                        for item_id, count in row_remove.items():
                            removals[item_id] += count
                    # The criterion is deliberately based on the block's
                    # starting count.  In-block purchases never rescue an
                    # ambiguous sell/undo permutation.
                    insufficient = sorted(
                        item_id
                        for item_id, removed in removals.items()
                        if working[pid].get(item_id, 0) < removed
                    )
                    if insufficient:
                        _record_reason(persistent_reasons, pid, "same_timestamp_block_not_order_independent")
                        _record_reason(persistent_reasons, pid, "item_replay_underflow")
                        persistent_unknown[pid] = True
                        block_unknown[pid] = tuple(persistent_reasons[pid])
                        continue

                    delta: dict[int, int] = {}
                    for item_id in set(additions) | set(removals):
                        net = additions.get(item_id, 0) - removals.get(item_id, 0)
                        if net:
                            delta[int(item_id)] = int(net)
                    if delta:
                        block_deltas[pid] = delta
                        for item_id, net in delta.items():
                            next_count = working[pid].get(item_id, 0) + net
                            # The certification above proves this cannot be
                            # negative; retain a defensive guard against bad
                            # future edits rather than silently clamping.
                            if next_count < 0:
                                raise AssertionError("certified item replay produced a negative count")
                            if next_count:
                                working[pid][item_id] = next_count
                            else:
                                working[pid].pop(item_id, None)
            blocks.append(
                _Block(
                    timestamp_ms=int(timestamp),
                    records=records,
                    deltas=MappingProxyType({pid: MappingProxyType(dict(delta)) for pid, delta in block_deltas.items()}),
                    unknown_participants=MappingProxyType(dict(block_unknown)),
                    global_reasons=tuple(block_global_reasons),
                )
            )
            seen_identities.update(_identity_key(row.source_identity) for row in records)
        return tuple(blocks)

    @staticmethod
    def _validate_queries(queries: Sequence[Any]) -> list[int]:
        values: list[int] = []
        for value in queries:
            values.append(_strict_int(value, name="query timestamp", nonnegative=True))
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("query timestamps must be sorted nondecreasing integers")
        return values

    def _snapshot(
        self,
        query: int,
        counts: Mapping[int, Mapping[int, int]],
        known: Sequence[bool],
        reasons: Sequence[Sequence[str]],
        global_reason_values: Sequence[str],
    ) -> dict[str, Any]:
        all_ids: list[int] = []
        all_counts: list[int] = []
        offsets = [0]
        players: list[dict[str, Any]] = []
        for row, pid in enumerate(range(1, PARTICIPANTS + 1)):
            player_ids: list[int] = []
            player_counts: list[int] = []
            if known[row]:
                for item_id, count in sorted(counts[pid].items()):
                    if count > 0:
                        player_ids.append(int(item_id))
                        player_counts.append(int(count))
            all_ids.extend(player_ids)
            all_counts.extend(player_counts)
            offsets.append(len(all_ids))
            player_reasons = tuple(dict.fromkeys(tuple(reasons[row]) + tuple(global_reason_values)))
            players.append(
                {
                    "participantId": pid,
                    "known": bool(known[row]),
                    "recorded_ledger_supported": bool(known[row]),
                    "unknown": not bool(known[row]),
                    "reason": player_reasons[0] if player_reasons else None,
                    "reasons": player_reasons,
                    "item_ids": np.asarray(player_ids, dtype=np.int64),
                    "counts": np.asarray(player_counts, dtype=np.int64),
                }
            )
        item_ids = np.asarray(all_ids, dtype=np.int64)
        item_counts = np.asarray(all_counts, dtype=np.int64)
        offset_array = np.asarray(offsets, dtype=np.int64)
        known_array = np.asarray(known, dtype=bool)
        reasons_tuple = tuple(
            tuple(dict.fromkeys(tuple(reasons[row]) + tuple(global_reason_values)))
            for row in range(PARTICIPANTS)
        )
        transactions = [row.to_dict() for row in self._transactions[: bisect_right(self._transaction_timestamps, query)]]
        return {
            "query_timestamp_ms": int(query),
            "known": known_array,
            "recorded_ledger_supported": known_array.copy(),
            "unknown": np.logical_not(known_array),
            "reasons": reasons_tuple,
            "players": tuple(players),
            "item_ids": item_ids,
            "counts": item_counts,
            "offsets": offset_array,
            "endpoint_item_ids": item_ids.copy(),
            "endpoint_counts": item_counts.copy(),
            "endpoint_offsets": offset_array.copy(),
            "transactions": transactions,
            "item_history": transactions,
            "provenance": self.provenance,
            "actual_inventory_complete": "unknown",
        }

    def query_many(self, queries: Sequence[Any]) -> list[dict[str, Any]]:
        """Return endpoints for sorted integer timestamps, including full blocks."""
        values = self._validate_queries(queries)
        if not values:
            return []
        counts = _copy_counts(self._initial_counts)
        known = [True] * PARTICIPANTS
        reasons: list[list[str]] = [[] for _ in range(PARTICIPANTS)]
        global_reason_values: list[str] = []
        global_unknown = False
        block_index = 0
        output: list[dict[str, Any]] = []
        for query in values:
            while block_index < len(self._blocks) and self._blocks[block_index].timestamp_ms <= query:
                block = self._blocks[block_index]
                if block.global_reasons and not global_unknown:
                    global_unknown = True
                    for reason in block.global_reasons:
                        if reason not in global_reason_values:
                            global_reason_values.append(reason)
                    known = [False] * PARTICIPANTS
                if not global_unknown:
                    for pid, delta in block.deltas.items():
                        row = pid - 1
                        if not known[row]:
                            continue
                        for item_id, net in delta.items():
                            next_count = counts[pid].get(item_id, 0) + net
                            if next_count < 0:
                                # This cannot occur after construction's
                                # certificate.  Keep query behavior fail-closed.
                                known[row] = False
                                reasons[row].append("item_replay_underflow")
                                counts[pid].clear()
                                continue
                            if next_count:
                                counts[pid][item_id] = next_count
                            else:
                                counts[pid].pop(item_id, None)
                    for pid, pid_reasons in block.unknown_participants.items():
                        row = pid - 1
                        known[row] = False
                        for reason in pid_reasons:
                            if reason not in reasons[row]:
                                reasons[row].append(reason)
                        counts[pid].clear()
                block_index += 1
            output.append(self._snapshot(query, counts, known, reasons, global_reason_values))
        return output

    def query(self, timestamp_ms: Any) -> dict[str, Any]:
        return self.query_many([timestamp_ms])[0]


# Short names make the bounded API easy to discover while retaining the
# explicit class name for callers that want to state the provenance boundary.
ItemLedger = ItemTransactionLedger
DirectItemLedger = ItemTransactionLedger


def replay_item_transactions(
    transactions: Iterable[ItemTransaction | Mapping[str, Any]],
    queries: Sequence[Any],
    *,
    initial_inventory: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for one sorted query batch."""
    return ItemTransactionLedger(transactions, initial_inventory=initial_inventory).query_many(queries)


__all__ = [
    "DirectItemLedger",
    "INITIAL_EMPTY_PROVENANCE",
    "ITEM_TRANSACTION_TYPES",
    "ItemLedger",
    "ItemTransaction",
    "ItemTransactionLedger",
    "PARTICIPANTS",
    "replay_item_transactions",
]
