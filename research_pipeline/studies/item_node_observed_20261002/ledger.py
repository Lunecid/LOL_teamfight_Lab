"""Recipient-confirmed observed item transactions.

The v3 interface is a narrow wrapper around the strict v2 replay ledger.  Only
transactions whose recipient is an integer participant ID in ``1..10`` enter a
player ledger.  Other records remain in a separate, timestamp-cutoff history
and therefore cannot poison a known player's replay.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from studies.item_node_encoding_20261002.ledger import (
    INITIAL_EMPTY_PROVENANCE,
    ITEM_TRANSACTION_TYPES,
    PARTICIPANTS,
    ItemTransaction,
    ItemTransactionLedger as _StrictItemTransactionLedger,
    _identity_key,
    _normalize_transaction,
)


OBSERVED_LEDGER_VERSION = "recipient_confirmed_item_observation_v3_20261002"
ITEM_EVENT_TYPES = tuple(sorted(ITEM_TRANSACTION_TYPES))
ITEM_FIELD_ROLES = ("itemId", "beforeId", "afterId")
FULL_INVENTORY_COMPLETE = "unknown"


def _pid_is_recipient(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and 1 <= int(value) <= PARTICIPANTS


def _normalize(value: ItemTransaction | Mapping[str, Any], ordinal: int) -> ItemTransaction:
    """Use the v2 normalizer while retaining malformed source payloads."""
    return _normalize_transaction(value, ordinal)


def _record(row: ItemTransaction, *, partition: str) -> dict[str, Any]:
    """Return a lossless source-oriented row with normalized audit fields."""
    raw = dict(row.raw_payload) if isinstance(row.raw_payload, Mapping) else {}
    result = row.to_dict()
    # Preserve the source spelling/value for malformed fields.  Normalized
    # fields remain available under snake_case and raw_* aliases.
    for canonical, aliases in (
        ("participantId", ("participantId", "participant_id", "pid")),
        ("itemId", ("itemId", "item_id")),
        ("beforeId", ("beforeId", "before_id")),
        ("afterId", ("afterId", "after_id")),
    ):
        for alias in aliases:
            if alias in raw:
                result[canonical] = raw[alias]
                result[f"raw_{canonical}"] = raw[alias]
                break
    result.update(
        {
            "participant_id": row.participant_id,
            "item_id": row.item_id,
            "before_id": row.before_id,
            "after_id": row.after_id,
            "partition": partition,
            "recipient_confirmed": partition == "identified",
            "raw_payload": raw,
        }
    )
    return result


def _positive(value: Any) -> Optional[int]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    value = int(value)
    return value if value > 0 else None


def _field_present(row: ItemTransaction, role: str) -> bool:
    aliases = {
        "itemId": ("itemId", "item_id"),
        "beforeId": ("beforeId", "before_id"),
        "afterId": ("afterId", "after_id"),
    }[role]
    if isinstance(row.raw_payload, Mapping) and row.raw_payload:
        return any(alias in row.raw_payload for alias in aliases)
    return getattr(row, {"itemId": "item_id", "beforeId": "before_id", "afterId": "after_id"}[role]) is not None


def _history_context(rows: Sequence[ItemTransaction]) -> dict[str, Any]:
    """Aggregate typed item history without assigning unknown ownership."""
    event_index = {name: i for i, name in enumerate(ITEM_EVENT_TYPES)}
    field_index = {name: i for i, name in enumerate(ITEM_FIELD_ROLES)}
    vectors: list[list[list[int]]] = [[[] for _ in ITEM_FIELD_ROLES] for _ in ITEM_EVENT_TYPES]
    event_counts = np.zeros(len(ITEM_EVENT_TYPES), dtype=np.int64)
    event_malformed_counts = np.zeros(len(ITEM_EVENT_TYPES), dtype=np.int64)
    field_presence_counts = np.zeros((len(ITEM_EVENT_TYPES), len(ITEM_FIELD_ROLES)), dtype=np.int64)
    field_validity_counts = np.zeros((len(ITEM_EVENT_TYPES), len(ITEM_FIELD_ROLES)), dtype=np.int64)
    field_malformed_counts = np.zeros_like(field_validity_counts)
    for row in rows:
        event_row = event_index.get(row.type)
        if event_row is None:
            continue
        event_counts[event_row] += 1
        values_by_role: dict[str, Optional[int]] = {}
        for role in ITEM_FIELD_ROLES:
            value = getattr(row, {"itemId": "item_id", "beforeId": "before_id", "afterId": "after_id"}[role])
            if _field_present(row, role):
                field_presence_counts[event_row, field_index[role]] += 1
            item_id = _positive(value)
            values_by_role[role] = item_id
            if item_id is not None:
                vectors[event_row][field_index[role]].append(item_id)
                field_validity_counts[event_row, field_index[role]] += 1
        if row.type == "ITEM_UNDO":
            # Both endpoint fields are required by the strict replay payload
            # certificate.  Zero is a valid present no-ID endpoint, but an
            # absent/null/malformed endpoint is a field-level failure.  A
            # positive opposite endpoint still contributes its history token.
            endpoint_invalid = False
            for role in ("beforeId", "afterId"):
                value = getattr(row, {"beforeId": "before_id", "afterId": "after_id"}[role])
                present = _field_present(row, role)
                if not present or value is None:
                    endpoint_invalid = True
                    field_malformed_counts[event_row, field_index[role]] += 1
                elif value == 0:
                    continue
            if endpoint_invalid or not any(values_by_role[role] is not None for role in ("beforeId", "afterId")):
                event_malformed_counts[event_row] += 1
        elif row.type in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
            if values_by_role["itemId"] is None:
                event_malformed_counts[event_row] += 1
                field_malformed_counts[event_row, field_index["itemId"]] += 1
        # Non-applicable fields are retained as presence/validity evidence but
        # do not make an otherwise valid event malformed.
    field_validity = field_validity_counts > 0
    field_presence = field_presence_counts > 0
    event_presence = event_counts > 0
    arrays = tuple(
        tuple(np.asarray(values, dtype=np.int64) for values in event_values)
        for event_values in vectors
    )
    context: dict[str, Any] = {
        "event_types": ITEM_EVENT_TYPES,
        "field_roles": ITEM_FIELD_ROLES,
        "item_id_vectors": arrays,
        "item_ids": arrays,
        "event_counts": event_counts,
        "event_presence": event_presence,
        "field_validity": field_validity,
        "field_presence": field_presence,
        "field_presence_counts": field_presence_counts,
        "field_validity_counts": field_validity_counts,
        "field_malformed_counts": field_malformed_counts,
        "event_malformed_counts": event_malformed_counts,
        "malformed_payload_count": int(event_malformed_counts.sum()),
        "valid_item_id_count": int(field_validity_counts.sum()),
        "event_count": int(event_counts.sum()),
        "history_observed": bool(len(rows)),
        "source_event_count": int(len(rows)),
    }
    return context


def _copy_counts(source: Mapping[int, Mapping[int, int]]) -> dict[int, dict[int, int]]:
    return {pid: dict(items) for pid, items in source.items()}


@dataclass(frozen=True)
class CompactObservedState:
    """Prefix state used by streaming bank writers.

    It intentionally excludes raw history lists, so iterating millions of
    query endpoints does not copy every history prefix.
    """

    query_timestamp_ms: int
    known: np.ndarray
    recorded_ledger_supported: np.ndarray
    unknown: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    item_ids: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    unassigned_history_observed: bool
    unassigned_count: int
    assigned_event_counts: np.ndarray
    unassigned_event_counts: np.ndarray
    full_inventory_complete: str = FULL_INVENTORY_COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_timestamp_ms": self.query_timestamp_ms,
            "known": self.known,
            "recorded_ledger_supported": self.recorded_ledger_supported,
            "observed_ledger_supported": self.recorded_ledger_supported.copy(),
            "recipient_confirmed_supported": self.recorded_ledger_supported.copy(),
            "unknown": self.unknown,
            "reasons": self.reasons,
            "item_ids": self.item_ids,
            "counts": self.counts,
            "offsets": self.offsets,
            "observed_item_ids": self.item_ids.copy(),
            "observed_counts": self.counts.copy(),
            "observed_offsets": self.offsets.copy(),
            "endpoint_item_ids": self.item_ids.copy(),
            "endpoint_counts": self.counts.copy(),
            "endpoint_offsets": self.offsets.copy(),
            "unassigned_history_observed": self.unassigned_history_observed,
            "unassigned_observed": self.unassigned_history_observed,
            "unassigned_item_history_observed": self.unassigned_history_observed,
            "has_unassigned_history": self.unassigned_history_observed,
            "unassigned_count": self.unassigned_count,
            "assigned_event_counts": self.assigned_event_counts,
            "unassigned_event_counts": self.unassigned_event_counts,
            "type_counts": self.assigned_event_counts.copy(),
            "unassigned_type_counts": self.unassigned_event_counts.copy(),
            "item_event_types": ITEM_EVENT_TYPES,
            "item_field_roles": ITEM_FIELD_ROLES,
            "full_inventory_complete": self.full_inventory_complete,
            "actual_inventory_complete": self.full_inventory_complete,
            "full_inventory_verification": 0.0,
            "whole_inventory_verification": 0.0,
            "full_inventory_verified": False,
            "full_inventory_supported": False,
            "whole_actual_inventory": self.full_inventory_complete,
        }


class RecipientConfirmedItemLedger:
    """Strict recipient-confirmed replay with separate unassigned history."""

    version = OBSERVED_LEDGER_VERSION
    actual_inventory_complete = FULL_INVENTORY_COMPLETE
    full_inventory_complete = FULL_INVENTORY_COMPLETE

    def __init__(
        self,
        transactions: Iterable[ItemTransaction | Mapping[str, Any]],
        *,
        initial_inventory: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
        initial_counts: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
    ) -> None:
        normalized = [_normalize(value, i) for i, value in enumerate(transactions)]
        normalized_with_order = list(enumerate(normalized))
        normalized_with_order.sort(key=lambda pair: (pair[1].timestamp_ms, pair[1].source_ordinal, pair[0]))
        self._transactions = tuple(row for _, row in normalized_with_order)
        self._timestamps = tuple(row.timestamp_ms for row in self._transactions)
        self._identified = tuple(row for row in self._transactions if _pid_is_recipient(row.participant_id))
        self._unassigned = tuple(row for row in self._transactions if not _pid_is_recipient(row.participant_id))
        self._identified_timestamps = tuple(row.timestamp_ms for row in self._identified)
        self._unassigned_timestamps = tuple(row.timestamp_ms for row in self._unassigned)
        self._assigned_event_counts_prefix = self._event_count_prefix(self._identified)
        self._unassigned_event_counts_prefix = self._event_count_prefix(self._unassigned)
        self._ledger = _StrictItemTransactionLedger(
            self._identified,
            initial_inventory=initial_inventory,
            initial_counts=initial_counts,
        )
        self._duplicate_taints, self._duplicate_taint_reasons = self._build_duplicate_taints()

    @staticmethod
    def _event_count_prefix(rows: Sequence[ItemTransaction]) -> np.ndarray:
        event_index = {name: i for i, name in enumerate(ITEM_EVENT_TYPES)}
        prefix = np.zeros((len(rows) + 1, len(ITEM_EVENT_TYPES)), dtype=np.int64)
        for index, row in enumerate(rows, start=1):
            prefix[index] = prefix[index - 1]
            if row.type in event_index:
                prefix[index, event_index[row.type]] += 1
        prefix.setflags(write=False)
        return prefix

    @property
    def transactions(self) -> tuple[ItemTransaction, ...]:
        """All normalized source transactions in chronological order."""
        return self._transactions

    @property
    def identified_transactions(self) -> tuple[ItemTransaction, ...]:
        return self._identified

    @property
    def unassigned_transactions(self) -> tuple[ItemTransaction, ...]:
        return self._unassigned

    @property
    def identified_history(self) -> tuple[ItemTransaction, ...]:
        return self._identified

    @property
    def unassigned_history(self) -> tuple[ItemTransaction, ...]:
        return self._unassigned

    @property
    def blocks(self):
        """Expose v2 compact blocks for downstream replay writers."""
        return self._ledger.blocks

    @property
    def provenance(self) -> dict[str, Any]:
        base = dict(self._ledger.provenance)
        base.update(
            {
                "version": self.version,
                "recipient_confirmed_only": True,
                "unassigned_history_preserved": True,
                "full_inventory_complete": FULL_INVENTORY_COMPLETE,
                "actual_inventory_complete": FULL_INVENTORY_COMPLETE,
                "actual_inventory_inferred": False,
                "assigned_history_consumed": True,
                "unassigned_history_is_graph_context": True,
            }
        )
        return base

    def _build_duplicate_taints(self) -> tuple[dict[int, frozenset[int]], dict[int, tuple[str, ...]]]:
        by_timestamp: dict[int, set[int]] = defaultdict(set)
        reason_by_timestamp: dict[int, list[str]] = defaultdict(list)
        seen_pids: dict[tuple[str, str, Any], set[int]] = defaultdict(set)
        seen_identities: set[tuple[str, str, Any]] = set()
        block_rows: dict[int, list[ItemTransaction]] = defaultdict(list)
        for row in self._transactions:
            block_rows[row.timestamp_ms].append(row)
        for timestamp in sorted(block_rows):
            rows = block_rows[timestamp]
            current: dict[tuple[str, str, Any], list[ItemTransaction]] = defaultdict(list)
            for row in rows:
                current[_identity_key(row.source_identity)].append(row)
            for identity, identity_rows in current.items():
                current_pids = {int(row.participant_id) for row in identity_rows if _pid_is_recipient(row.participant_id)}
                duplicate = len(identity_rows) > 1 or identity in seen_identities or any(
                    row.duplicate_source_identity for row in identity_rows
                )
                if duplicate:
                    linked = set(seen_pids.get(identity, set())) | current_pids
                    if linked:
                        by_timestamp[timestamp].update(linked)
                        if "duplicate_source_identity" not in reason_by_timestamp[timestamp]:
                            reason_by_timestamp[timestamp].append("duplicate_source_identity")
            for identity, identity_rows in current.items():
                seen_pids[identity].update(
                    int(row.participant_id) for row in identity_rows if _pid_is_recipient(row.participant_id)
                )
                seen_identities.add(identity)
        return (
            {timestamp: frozenset(pids) for timestamp, pids in by_timestamp.items()},
            {timestamp: tuple(reasons) for timestamp, reasons in reason_by_timestamp.items()},
        )

    @staticmethod
    def _validate_queries(queries: Iterable[Any]) -> Iterator[int]:
        previous: Optional[int] = None
        for value in queries:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError("query timestamps must be sorted nondecreasing integers")
            current = int(value)
            if previous is not None and current < previous:
                raise ValueError("query timestamps must be sorted nondecreasing integers")
            previous = current
            yield current

    def _apply_blocks(
        self,
        queries: Iterable[int],
    ) -> Iterator[CompactObservedState]:
        counts = _copy_counts(self._ledger._initial_counts)
        known = [True] * PARTICIPANTS
        reasons: list[list[str]] = [[] for _ in range(PARTICIPANTS)]
        block_index = 0
        taint_timestamps = tuple(sorted(self._duplicate_taints))
        taint_index = 0
        for query in queries:
            while True:
                block_timestamp = (
                    self._ledger.blocks[block_index].timestamp_ms
                    if block_index < len(self._ledger.blocks)
                    else None
                )
                taint_timestamp = (
                    taint_timestamps[taint_index]
                    if taint_index < len(taint_timestamps)
                    else None
                )
                candidates = [value for value in (block_timestamp, taint_timestamp) if value is not None]
                if not candidates or min(candidates) > query:
                    break
                timestamp = min(candidates)
                if block_timestamp == timestamp:
                    block = self._ledger.blocks[block_index]
                    for pid, delta in block.deltas.items():
                        row = pid - 1
                        if not known[row]:
                            continue
                        for item_id, net in delta.items():
                            next_count = counts[pid].get(item_id, 0) + net
                            if next_count < 0:
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
                if taint_timestamp == timestamp:
                    for pid in self._duplicate_taints[timestamp]:
                        row = pid - 1
                        known[row] = False
                        for reason in self._duplicate_taint_reasons.get(timestamp, ("duplicate_source_identity",)):
                            if reason not in reasons[row]:
                                reasons[row].append(reason)
                        counts[pid].clear()
                    taint_index += 1
            all_ids: list[int] = []
            all_counts: list[int] = []
            offsets = [0]
            for pid in range(1, PARTICIPANTS + 1):
                if known[pid - 1]:
                    for item_id, count in sorted(counts[pid].items()):
                        if count > 0:
                            all_ids.append(int(item_id))
                            all_counts.append(int(count))
                offsets.append(len(all_ids))
            reason_tuple = tuple(tuple(values) for values in reasons)
            unassigned_count = bisect_right(self._unassigned_timestamps, query)
            yield CompactObservedState(
                query_timestamp_ms=query,
                known=np.asarray(known, dtype=bool),
                recorded_ledger_supported=np.asarray(known, dtype=bool),
                unknown=np.logical_not(np.asarray(known, dtype=bool)),
                reasons=reason_tuple,
                item_ids=np.asarray(all_ids, dtype=np.int64),
                counts=np.asarray(all_counts, dtype=np.int64),
                offsets=np.asarray(offsets, dtype=np.int64),
                unassigned_history_observed=bool(unassigned_count),
                unassigned_count=unassigned_count,
                assigned_event_counts=self._assigned_event_counts_prefix[
                    bisect_right(self._identified_timestamps, query)
                ].copy(),
                unassigned_event_counts=self._unassigned_event_counts_prefix[unassigned_count].copy(),
            )

    def iter_compact_observed_states(self, queries: Iterable[Any]) -> Iterator[dict[str, Any]]:
        """Yield support/count endpoints without copying raw history prefixes."""
        for state in self._apply_blocks(self._validate_queries(queries)):
            yield state.to_dict()

    iter_compact_states = iter_compact_observed_states
    iter_compact_ledger_states = iter_compact_observed_states

    def _snapshot(self, compact: CompactObservedState) -> dict[str, Any]:
        query = compact.query_timestamp_ms
        end = bisect_right(self._timestamps, query)
        identified_rows = self._identified[: bisect_right(self._identified_timestamps, query)]
        unassigned_rows = self._unassigned[: bisect_right(self._unassigned_timestamps, query)]
        identified = [_record(row, partition="identified") for row in identified_rows]
        unassigned = [_record(row, partition="unassigned") for row in unassigned_rows]
        all_history = [_record(row, partition="identified" if _pid_is_recipient(row.participant_id) else "unassigned") for row in self._transactions[:end]]
        players: list[dict[str, Any]] = []
        for row, pid in enumerate(range(1, PARTICIPANTS + 1)):
            start, stop = int(compact.offsets[row]), int(compact.offsets[row + 1])
            player_ids = compact.item_ids[start:stop].copy()
            player_counts = compact.counts[start:stop].copy()
            player_reasons = compact.reasons[row]
            players.append(
                {
                    "participantId": pid,
                    "participant_id": pid,
                    "known": bool(compact.known[row]),
                    "recorded_ledger_supported": bool(compact.known[row]),
                    "observed_ledger_supported": bool(compact.known[row]),
                    "unknown": bool(compact.unknown[row]),
                    "reason": player_reasons[0] if player_reasons else None,
                    "reasons": player_reasons,
                    "item_ids": player_ids,
                    "counts": player_counts,
                }
            )
        assigned_context = _history_context(identified_rows)
        unassigned_context = _history_context(unassigned_rows)
        by_player = tuple(
            _history_context(tuple(row for row in identified_rows if int(row.participant_id) == pid))
            for pid in range(1, PARTICIPANTS + 1)
        )
        unassigned_identity_counts: dict[tuple[str, str, Any], int] = defaultdict(int)
        unassigned_duplicate_observed = False
        for row in unassigned_rows:
            identity = _identity_key(row.source_identity)
            unassigned_identity_counts[identity] += 1
            if unassigned_identity_counts[identity] > 1 or row.duplicate_source_identity:
                unassigned_duplicate_observed = True
        result = compact.to_dict()
        result.update(
            {
                "version": self.version,
                "query_timestamp_ms": query,
                "known": compact.known.copy(),
                "recorded_ledger_supported": compact.recorded_ledger_supported.copy(),
                "observed_ledger_supported": compact.recorded_ledger_supported.copy(),
                "recipient_confirmed_supported": compact.recorded_ledger_supported.copy(),
                "unknown": compact.unknown.copy(),
                "reasons": compact.reasons,
                "players": tuple(players),
                "item_ids": compact.item_ids.copy(),
                "counts": compact.counts.copy(),
                "offsets": compact.offsets.copy(),
                "observed_item_ids": compact.item_ids.copy(),
                "observed_counts": compact.counts.copy(),
                "observed_offsets": compact.offsets.copy(),
                "endpoint_item_ids": compact.item_ids.copy(),
                "endpoint_counts": compact.counts.copy(),
                "endpoint_offsets": compact.offsets.copy(),
                "transactions": identified,
                "item_history": identified,
                "identified_transactions": identified,
                "identified_history": identified,
                "raw_identified_history": identified,
                "unassigned_transactions": unassigned,
                "unassigned_history": unassigned,
                "raw_unassigned_history": unassigned,
                "raw_item_history": all_history,
                "unassigned_history_observed": bool(unassigned),
                "unassigned_observed": bool(unassigned),
                "unassigned_item_history_observed": bool(unassigned),
                "has_unassigned_history": bool(unassigned),
                "unassigned_count": len(unassigned),
                "unassigned_query_flags": {
                    "observed": bool(unassigned),
                    "count": len(unassigned),
                    "duplicate_identity_observed": unassigned_duplicate_observed,
                    "duplicate_source_identity_observed": unassigned_duplicate_observed,
                },
                "assigned_history_context": assigned_context,
                "assigned_history": assigned_context,
                "history_context": assigned_context,
                "assigned_history_by_participant": by_player,
                "unassigned_history_context": unassigned_context,
                "unassigned_context": unassigned_context,
                "unassigned_graph_context": unassigned_context,
                "full_inventory_complete": FULL_INVENTORY_COMPLETE,
                "actual_inventory_complete": FULL_INVENTORY_COMPLETE,
                "full_inventory_verification": 0.0,
                "whole_inventory_verification": 0.0,
                "full_inventory_verified": False,
                "full_inventory_supported": False,
                "whole_actual_inventory": FULL_INVENTORY_COMPLETE,
                "whole_inventory_supported": False,
                "provenance": self.provenance,
            }
        )
        return result

    def query_many(self, queries: Sequence[Any]) -> list[dict[str, Any]]:
        return [self._snapshot(CompactObservedState(**{
            "query_timestamp_ms": state["query_timestamp_ms"],
            "known": state["known"],
            "recorded_ledger_supported": state["recorded_ledger_supported"],
            "unknown": state["unknown"],
            "reasons": state["reasons"],
            "item_ids": state["item_ids"],
            "counts": state["counts"],
            "offsets": state["offsets"],
            "unassigned_history_observed": state["unassigned_history_observed"],
            "unassigned_count": state["unassigned_count"],
            "assigned_event_counts": state["assigned_event_counts"],
            "unassigned_event_counts": state["unassigned_event_counts"],
        })) for state in self.iter_compact_observed_states(queries)]

    def query(self, timestamp_ms: Any) -> dict[str, Any]:
        return self.query_many([timestamp_ms])[0]


# Discoverable aliases used by downstream adapters.
ObservedItemLedger = RecipientConfirmedItemLedger
ItemObservationLedger = RecipientConfirmedItemLedger
ObservedTransactionLedger = RecipientConfirmedItemLedger
ItemLedger = RecipientConfirmedItemLedger
ItemTransactionLedger = RecipientConfirmedItemLedger


def replay_observed_item_transactions(
    transactions: Iterable[ItemTransaction | Mapping[str, Any]],
    queries: Sequence[Any],
    *,
    initial_inventory: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
) -> list[dict[str, Any]]:
    return RecipientConfirmedItemLedger(transactions, initial_inventory=initial_inventory).query_many(queries)


replay_item_transactions = replay_observed_item_transactions


def iter_compact_observed_states(
    ledger: RecipientConfirmedItemLedger,
    queries: Iterable[Any],
) -> Iterator[dict[str, Any]]:
    """Top-level adapter for streaming bank writers."""
    return ledger.iter_compact_observed_states(queries)


iter_compact_ledger_states = iter_compact_observed_states


__all__ = [
    "CompactObservedState",
    "FULL_INVENTORY_COMPLETE",
    "INITIAL_EMPTY_PROVENANCE",
    "ITEM_EVENT_TYPES",
    "ITEM_FIELD_ROLES",
    "ITEM_TRANSACTION_TYPES",
    "ItemObservationLedger",
    "ItemLedger",
    "ItemTransaction",
    "ItemTransactionLedger",
    "ObservedItemLedger",
    "ObservedTransactionLedger",
    "OBSERVED_LEDGER_VERSION",
    "PARTICIPANTS",
    "RecipientConfirmedItemLedger",
    "iter_compact_observed_states",
    "iter_compact_ledger_states",
    "replay_observed_item_transactions",
    "replay_item_transactions",
]
