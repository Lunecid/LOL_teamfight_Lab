"""Independent literal checks for the recipient-confirmed item contract.

The small replay below intentionally does not import production replay or
reason helpers.  The integration checks near the end bind the worker APIs
only after comparing against that oracle.  The oracle covers edge cases that
are easy to accidentally hide behind a compact implementation:
unknown owners, chronological identity duplicates, same-ms certification,
typed histories, and the separation between observed support and complete
actual inventory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import permutations
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pytest


TYPES = {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"}
ITEM_FIELDS = ("itemId", "beforeId", "afterId")
FIELD_NAMES = ("participantId", "itemId", "beforeId", "afterId")


@dataclass(frozen=True)
class Event:
    t: int
    ordinal: int
    identity: Any
    payload: Mapping[str, Any]


@dataclass
class _State:
    bags: dict[int, Counter[int]] = field(
        default_factory=lambda: {pid: Counter() for pid in range(1, 11)}
    )
    unknown: dict[int, bool] = field(
        default_factory=lambda: {pid: False for pid in range(1, 11)}
    )
    reasons: dict[int, list[str]] = field(
        default_factory=lambda: {pid: [] for pid in range(1, 11)}
    )
    linked: dict[Any, set[int]] = field(default_factory=lambda: defaultdict(set))
    seen: set[Any] = field(default_factory=set)
    assigned_history: dict[int, dict[str, Any]] = field(default_factory=dict)
    unassigned_history: dict[str, Any] = field(default_factory=dict)


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_pid(value: Any) -> bool:
    return _strict_int(value) and 1 <= value <= 10


def _valid_positive(value: Any) -> bool:
    return _strict_int(value) and value > 0


def _valid_nonnegative(value: Any) -> bool:
    return _strict_int(value) and value >= 0


def _key(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _event_type(payload: Mapping[str, Any]) -> str:
    value = payload.get("type", payload.get("event_type", ""))
    return value.strip().upper() if isinstance(value, str) else ""


def _field_stats() -> dict[str, Any]:
    return {
        "event_count": Counter(),
        "presence": Counter(),
        "validity": Counter(),
        "malformed": Counter(),
        "event_malformed": Counter(),
        "positive_item_tokens": Counter(),
    }


def _merge_history(dst: dict[str, Any], payload: Mapping[str, Any]) -> None:
    stats = dst.setdefault("stats", _field_stats())
    kind = _event_type(payload)
    stats["event_count"][kind] += 1
    for field_name in FIELD_NAMES:
        if field_name not in payload:
            continue
        stats["presence"][field_name] += 1
        value = payload[field_name]
        valid = _valid_pid(value) if field_name == "participantId" else _valid_positive(value)
        stats["validity"][field_name] += int(valid)
        stats["malformed"][field_name] += int(not valid)
        if field_name in ITEM_FIELDS and valid:
            stats["positive_item_tokens"][(kind, field_name, int(value))] += 1
    if kind in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
        if not _valid_positive(payload.get("itemId")):
            stats["event_malformed"][kind] += 1
    elif kind == "ITEM_UNDO":
        before = payload.get("beforeId")
        after = payload.get("afterId")
        if not (_valid_nonnegative(before) and _valid_nonnegative(after) and (before > 0 or after > 0)):
            stats["event_malformed"][kind] += 1


def _delta(payload: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, int]]:
    kind = _event_type(payload)
    add: Counter[int] = Counter()
    remove: Counter[int] = Counter()
    if kind == "ITEM_PURCHASED":
        add[int(payload["itemId"])] += 1
    elif kind in {"ITEM_SOLD", "ITEM_DESTROYED"}:
        remove[int(payload["itemId"])] += 1
    elif kind == "ITEM_UNDO":
        if payload.get("beforeId", 0) > 0:
            remove[int(payload["beforeId"])] += 1
        if payload.get("afterId", 0) > 0:
            add[int(payload["afterId"])] += 1
    return dict(add), dict(remove)


def _payload_valid(payload: Mapping[str, Any]) -> bool:
    kind = _event_type(payload)
    if kind not in TYPES or not _valid_pid(payload.get("participantId")):
        return False
    if kind in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
        return _valid_positive(payload.get("itemId"))
    before = payload.get("beforeId")
    after = payload.get("afterId")
    return _valid_nonnegative(before) and _valid_nonnegative(after) and (before > 0 or after > 0)


def _payload_reason(payload: Mapping[str, Any]) -> str:
    kind = _event_type(payload)
    if kind not in TYPES:
        return "invalid_item_transaction_type"
    if kind in {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"}:
        return "invalid_item_id"
    return "invalid_undo_payload"


def _reason(state: _State, pid: int, value: str) -> None:
    if value not in state.reasons[pid]:
        state.reasons[pid].append(value)
    state.unknown[pid] = True
    # Once a player ledger is unsupported, expose no stale current bag as if
    # it were observed.  The source history remains available separately.
    state.bags[pid].clear()


def literal_replay(rows: Iterable[Mapping[str, Any]], cutoff: int | None = None) -> dict[str, Any]:
    """Replay source rows with only rules written in the v3 contract."""
    events = [
        Event(
            t=int(row["timestamp_ms"]),
            ordinal=int(row.get("source_ordinal", index)),
            identity=row.get("source_identity", f"ordinal:{index}"),
            payload=dict(row),
        )
        for index, row in enumerate(rows)
        if cutoff is None or int(row["timestamp_ms"]) <= cutoff
    ]
    events.sort(key=lambda event: (event.t, event.ordinal))
    state = _State()
    if not events:
        return _snapshot(state, False)

    by_t: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_t[event.t].append(event)

    unassigned_observed = False
    for timestamp in sorted(by_t):
        block = by_t[timestamp]
        groups: dict[Any, list[Event]] = defaultdict(list)
        current_valid_pids: dict[Any, set[int]] = defaultdict(set)
        current_events_by_pid: dict[int, list[Event]] = defaultdict(list)
        for event in block:
            identity = _key(event.identity)
            groups[identity].append(event)
            pid = event.payload.get("participantId")
            assigned = _valid_pid(pid)
            target = (
                state.assigned_history.setdefault(int(pid), {})
                if assigned
                else state.unassigned_history
            )
            _merge_history(target, event.payload)
            if not assigned:
                unassigned_observed = True
            else:
                current_valid_pids[identity].add(int(pid))
                state.linked[identity].add(int(pid))
                current_events_by_pid[int(pid)].append(event)

        # Duplicate identities are detected chronologically across assigned
        # and unassigned partitions.  The prior snapshot remains untouched.
        duplicate_ids = {
            identity
            for identity, grouped in groups.items()
            if len(grouped) > 1
            or identity in state.seen
            or any(bool(event.payload.get("duplicate_source_identity", False)) for event in grouped)
        }
        was_unknown = {pid: state.unknown[pid] for pid in current_events_by_pid}
        for identity in duplicate_ids:
            for pid in sorted(state.linked[identity] | current_valid_pids[identity]):
                grouped = groups[identity]
                if len(grouped) > 1 or identity in state.seen or any(bool(event.payload.get("duplicate_source_identity", False)) for event in grouped):
                    _reason(state, pid, "duplicate_source_identity")
                if any(bool(event.payload.get("duplicate_source_identity", False)) for event in grouped):
                    _reason(state, pid, "duplicate_source_identity_flag")

        # An unassigned row is retained in history but cannot invalidate a
        # different recipient-confirmed ledger.
        # Certification is per participant over the complete same-ms block,
        # even when that block contains multiple source identities.
        for pid, pid_events in sorted(current_events_by_pid.items()):
            invalid_reasons = tuple(
                dict.fromkeys(
                    _payload_reason(event.payload)
                    for event in pid_events
                    if not _payload_valid(event.payload)
                )
            )
            if invalid_reasons:
                if not was_unknown[pid]:
                    for reason in invalid_reasons:
                        _reason(state, pid, reason)
                continue
            if state.unknown[pid]:
                continue
            additions: Counter[int] = Counter()
            removals: Counter[int] = Counter()
            for event in pid_events:
                add, remove = _delta(event.payload)
                additions.update(add)
                removals.update(remove)
            if any(state.bags[pid][item] < count for item, count in removals.items()):
                _reason(state, pid, "same_timestamp_block_not_order_independent")
                _reason(state, pid, "item_replay_underflow")
                continue
            state.bags[pid].update(additions)
            state.bags[pid].subtract(removals)
            state.bags[pid] += Counter()
        # Keep every identity in chronological history. A later confirmed
        # reuse of an earlier unassigned identity taints only that later
        # endpoint; the earlier endpoint was already emitted unchanged.
        state.seen.update(groups)

    return _snapshot(state, unassigned_observed)


def _snapshot(state: _State, unassigned_observed: bool) -> dict[str, Any]:
    bags = {pid: dict(sorted(counter.items())) for pid, counter in state.bags.items()}
    support = {pid: not state.unknown[pid] for pid in range(1, 11)}
    return {
        "bags": bags,
        "recorded_ledger_supported": support,
        "unknown": dict(state.unknown),
        "reasons": {pid: tuple(values) for pid, values in state.reasons.items()},
        "assigned_history": state.assigned_history,
        "unassigned_history": state.unassigned_history,
        "unassigned_item_history_observed": bool(unassigned_observed),
        "whole_inventory_verification": 0,
        "actual_inventory_complete": "unknown",
        "initial_state": "initial_empty_recorded_transaction_ledger_assumption",
    }


def _row(t: int, ordinal: int, identity: str, **payload: Any) -> dict[str, Any]:
    return {"timestamp_ms": t, "source_ordinal": ordinal, "source_identity": identity, **payload}


def test_source_zero_is_unassigned_and_does_not_poison_known_recipient():
    rows = [
        _row(0, 0, "u0", type="ITEM_PURCHASED", participantId=0, itemId=3865),
        _row(10, 1, "p1", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(10, 2, "p2", type="ITEM_PURCHASED", participantId=2, itemId=1001),
    ]
    result = literal_replay(rows, 10)
    assert result["bags"][1] == {1001: 1}
    assert result["bags"][2] == {1001: 1}
    assert all(result["recorded_ledger_supported"].values())
    assert result["unassigned_item_history_observed"] is True
    assert result["unassigned_history"]["stats"]["positive_item_tokens"][("ITEM_PURCHASED", "itemId", 3865)] == 1


def test_invalid_assigned_payload_is_local_and_persistent():
    rows = [
        _row(10, 0, "bad", type="ITEM_PURCHASED", participantId=1, itemId=0),
        _row(10, 1, "good", type="ITEM_PURCHASED", participantId=2, itemId=2002),
        _row(20, 2, "later", type="ITEM_PURCHASED", participantId=1, itemId=3003),
    ]
    result = literal_replay(rows, 20)
    assert result["recorded_ledger_supported"][1] is False
    assert result["unknown"][1] is True
    assert result["bags"][1] == {}
    assert result["recorded_ledger_supported"][2] is True
    assert result["bags"][2] == {2002: 1}
    stats = result["assigned_history"][1]["stats"]
    assert stats["presence"]["itemId"] == 2
    assert stats["validity"]["itemId"] == 1
    assert stats["malformed"]["itemId"] == 1
    assert stats["positive_item_tokens"][("ITEM_PURCHASED", "itemId", 3003)] == 1


def test_same_ms_certificate_accepts_all_permutations_when_starting_count_covers_removals():
    rows = [
        _row(100, 0, "s1", type="ITEM_SOLD", participantId=1, itemId=1001),
        _row(100, 1, "s2", type="ITEM_DESTROYED", participantId=1, itemId=1001),
        _row(100, 2, "s3", type="ITEM_PURCHASED", participantId=1, itemId=2002),
    ]
    # Prefix supplies two units, so no ordering of the atomic block can underflow.
    prefix = [
        _row(50, 0, "seed1", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(50, 1, "seed2", type="ITEM_PURCHASED", participantId=1, itemId=1001),
    ]
    expected = {1001: 0, 2002: 1}
    for permutation in permutations(rows):
        result = literal_replay(prefix + list(permutation), 100)
        assert result["recorded_ledger_supported"][1] is True
        assert result["bags"][1] == {item: count for item, count in expected.items() if count}


def test_same_ms_purchase_cannot_rescue_ambiguous_removal_from_empty_start():
    rows = [
        _row(100, 0, "buy", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(100, 1, "sell", type="ITEM_SOLD", participantId=1, itemId=1001),
        _row(100, 2, "other", type="ITEM_PURCHASED", participantId=2, itemId=2002),
    ]
    result = literal_replay(rows, 100)
    assert result["recorded_ledger_supported"][1] is False
    assert result["bags"][1] == {}
    assert result["recorded_ledger_supported"][2] is True
    assert result["bags"][2] == {2002: 1}


def test_cross_partition_duplicate_taints_at_duplicate_timestamp_only():
    rows = [
        _row(100, 0, "shared", type="ITEM_PURCHASED", participantId=1, itemId=1111),
        _row(150, 1, "clean", type="ITEM_PURCHASED", participantId=2, itemId=2222),
        _row(200, 2, "shared", type="ITEM_PURCHASED", participantId=0, itemId=3333),
        _row(250, 3, "later", type="ITEM_PURCHASED", participantId=1, itemId=4444),
    ]
    early = literal_replay(rows, 150)
    late = literal_replay(rows, 250)
    assert early["recorded_ledger_supported"][1] is True
    assert early["bags"][1] == {1111: 1}
    assert late["recorded_ledger_supported"][1] is False
    assert late["bags"][1] == {}
    assert late["recorded_ledger_supported"][2] is True
    assert late["bags"][2] == {2222: 1}
    assert late["unassigned_item_history_observed"] is True


def test_unassigned_only_duplicate_stays_in_context_and_does_not_taint_players():
    rows = [
        _row(100, 0, "u", type="ITEM_PURCHASED", participantId=0, itemId=5001),
        _row(200, 1, "u", type="ITEM_PURCHASED", participantId=None, itemId=5001),
        _row(300, 2, "p", type="ITEM_PURCHASED", participantId=1, itemId=6001),
    ]
    result = literal_replay(rows, 300)
    assert result["recorded_ledger_supported"][1] is True
    assert result["bags"][1] == {6001: 1}
    assert result["unassigned_item_history_observed"] is True


def test_prefix_replay_is_equal_to_full_stream_at_same_cutoff():
    prefix = [
        _row(100, 0, "p", type="ITEM_PURCHASED", participantId=1, itemId=7001),
        _row(100, 1, "p2", type="ITEM_PURCHASED", participantId=2, itemId=7002),
    ]
    future = [
        _row(200, 2, "p", type="ITEM_PURCHASED", participantId=0, itemId=7003),
        _row(300, 3, "new", type="ITEM_PURCHASED", participantId=1, itemId=7004),
    ]
    left = literal_replay(prefix, 100)
    right = literal_replay(prefix + future, 100)
    assert left["bags"] == right["bags"]
    assert left["recorded_ledger_supported"] == right["recorded_ledger_supported"]
    assert left["unknown"] == right["unknown"]
    assert left["reasons"] == right["reasons"]
    assert left["unassigned_item_history_observed"] == right["unassigned_item_history_observed"]


def test_typed_histories_count_presence_and_positive_ids_without_pushing_unknown_item():
    rows = [
        _row(1, 0, "u", type="ITEM_UNDO", participantId=0, beforeId=0, afterId=1234),
        _row(2, 1, "a", type="ITEM_PURCHASED", participantId=1, itemId=0),
        _row(3, 2, "b", type="ITEM_UNDO", participantId=1, beforeId=2222, afterId=0),
        _row(4, 3, "negative", type="ITEM_PURCHASED", participantId=1, itemId=-5),
        _row(5, 4, "fraction", type="ITEM_PURCHASED", participantId=1, itemId=1.5),
        _row(6, 5, "string", type="ITEM_PURCHASED", participantId=1, itemId="1004"),
    ]
    result = literal_replay(rows, 6)
    assert result["bags"][1] == {}
    assert result["unassigned_history"]["stats"]["presence"]["beforeId"] == 1
    assert result["unassigned_history"]["stats"]["validity"]["beforeId"] == 0
    assert result["unassigned_history"]["stats"]["positive_item_tokens"][("ITEM_UNDO", "afterId", 1234)] == 1
    assigned_stats = result["assigned_history"][1]["stats"]
    assert assigned_stats["presence"]["itemId"] == 4
    assert assigned_stats["presence"]["beforeId"] == 1
    assert assigned_stats["presence"]["afterId"] == 1
    assert assigned_stats["positive_item_tokens"][("ITEM_UNDO", "beforeId", 2222)] == 1
    assert assigned_stats["event_malformed"]["ITEM_PURCHASED"] == 4
    assert all(key[2] not in {-5, 1.5, 1004} for key in assigned_stats["positive_item_tokens"])
    assert all(item_id != 1234 for item_id in result["bags"][1])


def test_known_observed_empty_differs_from_unknown_and_full_inventory_is_never_promoted():
    known_empty = literal_replay(
        [
            _row(1, 0, "buy", type="ITEM_PURCHASED", participantId=1, itemId=1001),
            _row(2, 1, "sell", type="ITEM_SOLD", participantId=1, itemId=1001),
        ],
        2,
    )
    unknown = literal_replay(
        [_row(1, 0, "bad", type="ITEM_SOLD", participantId=1, itemId=1001)],
        1,
    )
    assert known_empty["bags"][1] == {}
    assert known_empty["recorded_ledger_supported"][1] is True
    assert unknown["bags"][1] == {}
    assert unknown["recorded_ledger_supported"][1] is False
    for result in (known_empty, unknown):
        assert result["whole_inventory_verification"] == 0
        assert result["actual_inventory_complete"] == "unknown"
        assert result["initial_state"] == "initial_empty_recorded_transaction_ledger_assumption"


def test_undo_delta_uses_positive_before_and_after_ids_only():
    rows = [
        _row(1, 0, "seed", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(2, 1, "undo", type="ITEM_UNDO", participantId=1, beforeId=1001, afterId=2002),
    ]
    result = literal_replay(rows, 2)
    assert result["bags"][1] == {2002: 1}


def test_all_nodes_and_full_uncertainty_have_explicit_contract_values():
    result = literal_replay([], 0)
    assert set(result["bags"]) == set(range(1, 11))
    assert set(result["recorded_ledger_supported"]) == set(range(1, 11))
    assert result["whole_inventory_verification"] == 0
    assert result["actual_inventory_complete"] == "unknown"


def _production_bag(snapshot: Mapping[str, Any], pid: int) -> dict[int, int]:
    offsets = np.asarray(snapshot["offsets"], dtype=np.int64)
    ids = np.asarray(snapshot["item_ids"], dtype=np.int64)
    counts = np.asarray(snapshot["counts"], dtype=np.int64)
    start, stop = int(offsets[pid - 1]), int(offsets[pid])
    return {int(item): int(count) for item, count in zip(ids[start:stop], counts[start:stop])}


def test_production_v3_ledger_matches_literal_partition_and_prefix_oracle():
    """Exercise the worker API against the independent source oracle."""
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(0, 0, "u0", type="ITEM_PURCHASED", participantId=0, itemId=3865),
        _row(10, 1, "p1", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(20, 2, "p2", type="ITEM_PURCHASED", participantId=2, itemId=1001),
        _row(30, 3, "p1", type="ITEM_SOLD", participantId=1, itemId=1001),
        _row(40, 4, "p0", type="ITEM_PURCHASED", participantId=0, itemId=2002),
    ]
    expected = literal_replay(rows, 40)
    actual = RecipientConfirmedItemLedger(rows).query(40)
    for pid in range(1, 11):
        assert _production_bag(actual, pid) == expected["bags"][pid]
        assert bool(actual["recorded_ledger_supported"][pid - 1]) == expected["recorded_ledger_supported"][pid]
    assert bool(actual["unassigned_history_observed"]) is expected["unassigned_item_history_observed"]
    assert actual["actual_inventory_complete"] == "unknown"
    assert actual["full_inventory_supported"] is False


def test_production_v3_cross_partition_duplicate_is_prefix_safe():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(100, 0, "shared", type="ITEM_PURCHASED", participantId=1, itemId=1111),
        _row(150, 1, "clean", type="ITEM_PURCHASED", participantId=2, itemId=2222),
        _row(200, 2, "shared", type="ITEM_PURCHASED", participantId=0, itemId=3333),
        _row(250, 3, "later", type="ITEM_PURCHASED", participantId=1, itemId=4444),
    ]
    ledger = RecipientConfirmedItemLedger(rows)
    for cutoff in (150, 250):
        expected = literal_replay(rows, cutoff)
        actual = ledger.query(cutoff)
        for pid in range(1, 11):
            assert _production_bag(actual, pid) == expected["bags"][pid]
            assert bool(actual["recorded_ledger_supported"][pid - 1]) == expected["recorded_ledger_supported"][pid]


def test_production_v3_duplicate_flag_and_identity_reason_union_is_literal():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(100, 0, "same", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(200, 1, "same", type="ITEM_PURCHASED", participantId=1, itemId=1002, duplicate_source_identity=True),
    ]
    expected = literal_replay(rows, 200)
    actual = RecipientConfirmedItemLedger(rows).query(200)
    assert expected["reasons"][1] == ("duplicate_source_identity", "duplicate_source_identity_flag")
    assert tuple(actual["reasons"][0]) == expected["reasons"][1]
    assert bool(actual["recorded_ledger_supported"][0]) is False


def test_literal_single_duplicate_flag_synthesizes_base_and_flag_reasons():
    rows = [
        _row(100, 0, "flag-only", type="ITEM_PURCHASED", participantId=1, itemId=1001, duplicate_source_identity=True),
    ]
    expected = literal_replay(rows, 100)
    assert set(expected["reasons"][1]) == {"duplicate_source_identity", "duplicate_source_identity_flag"}


def test_production_v3_same_ms_invalid_undo_reason_is_not_first_row_reason():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(100, 0, "purchase", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(100, 1, "bad-undo", type="ITEM_UNDO", participantId=1, afterId=1002),
    ]
    expected = literal_replay(rows, 100)
    actual = RecipientConfirmedItemLedger(rows).query(100)
    assert expected["reasons"][1] == ("invalid_undo_payload",)
    assert tuple(actual["reasons"][0]) == expected["reasons"][1]
    assert bool(actual["recorded_ledger_supported"][0]) is False


def test_production_v3_unassigned_duplicate_then_assigned_same_identity_taints_later_endpoint_only():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(100, 0, "shared", type="ITEM_PURCHASED", participantId=0, itemId=3865),
        _row(200, 1, "shared", type="ITEM_PURCHASED", participantId=1, itemId=1001),
    ]
    ledger = RecipientConfirmedItemLedger(rows)
    early, late = ledger.query_many((100, 200))
    assert bool(early["recorded_ledger_supported"][0]) is True
    assert _production_bag(early, 1) == {}
    assert bool(late["recorded_ledger_supported"][0]) is False
    assert _production_bag(late, 1) == {}
    assert "duplicate_source_identity" in late["reasons"][0]


def test_production_v3_history_keeps_zero_presence_separate_from_absence():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(1, 0, "p", type="ITEM_UNDO", participantId=1, beforeId=0, afterId=1001),
        _row(2, 1, "p2", type="ITEM_UNDO", participantId=1, afterId=1002),
        _row(3, 2, "bad", type="ITEM_PURCHASED", participantId=1, itemId=0),
        _row(4, 3, "negative", type="ITEM_PURCHASED", participantId=1, itemId=-5),
        _row(5, 4, "fraction", type="ITEM_PURCHASED", participantId=1, itemId=1.5),
        _row(6, 5, "string", type="ITEM_PURCHASED", participantId=1, itemId="1004"),
    ]
    snapshot = RecipientConfirmedItemLedger(rows).query(6)
    context = snapshot["assigned_history_by_participant"][0]
    event_index = {name: index for index, name in enumerate(context["event_types"])}
    field_index = {name: index for index, name in enumerate(context["field_roles"])}
    undo = event_index["ITEM_UNDO"]
    item = field_index["itemId"]
    before = field_index["beforeId"]
    after = field_index["afterId"]
    assert int(context["field_presence_counts"][undo, before]) == 1
    assert int(context["field_validity_counts"][undo, before]) == 0
    assert int(context["field_presence_counts"][undo, after]) == 2
    assert int(context["field_validity_counts"][undo, after]) == 2
    assert int(context["field_presence_counts"][event_index["ITEM_PURCHASED"], item]) == 4
    assert int(context["field_validity_counts"][event_index["ITEM_PURCHASED"], item]) == 0
    assert int(context["event_malformed_counts"][event_index["ITEM_PURCHASED"]]) == 4
    assert all(0 not in np.asarray(values).tolist() for row in context["item_id_vectors"] for values in row)


def test_bank_wrapper_preserves_absent_fields_and_explicit_zero_presence():
    from studies.item_node_observed_20261002.bank import _build_ledger, _event_from_value

    event = _event_from_value(
        {"timestamp_ms": 1, "source_ordinal": 0, "source_identity": "p", "type": "ITEM_PURCHASED", "participantId": 1, "itemId": 1001},
        0,
    )
    mapped = event.transaction_mapping()
    assert "beforeId" not in mapped and "afterId" not in mapped
    # The wrapper must carry raw presence separately, including an explicit
    # zero endpoint that is present but not a positive item token.
    undo = _event_from_value(
        {"timestamp_ms": 2, "source_ordinal": 1, "source_identity": "u", "type": "ITEM_UNDO", "participantId": 1, "beforeId": 0, "afterId": 1002},
        1,
    )
    ledger = _build_ledger((event,))
    context = ledger.query(1)["assigned_history_by_participant"][0]
    item = context["field_roles"].index("itemId")
    before = context["field_roles"].index("beforeId")
    after = context["field_roles"].index("afterId")
    purchased = context["event_types"].index("ITEM_PURCHASED")
    assert context["field_presence_counts"][purchased, item] == 1
    assert context["field_presence_counts"][purchased, before] == 0
    assert context["field_presence_counts"][purchased, after] == 0
    undo_context = _build_ledger((undo,)).query(2)["assigned_history_by_participant"][0]
    undo_row = undo_context["event_types"].index("ITEM_UNDO")
    assert undo_context["field_presence_counts"][undo_row, before] == 1
    assert undo_context["field_validity_counts"][undo_row, before] == 0
    assert undo_context["field_malformed_counts"][undo_row, before] == 0
    assert undo_context["field_presence_counts"][undo_row, after] == 1
    assert undo_context["field_validity_counts"][undo_row, after] == 1
    assert undo_context["event_malformed_counts"][undo_row] == 0

    malformed_undo = _build_ledger(
        (_event_from_value({
            "timestamp_ms": 3, "source_ordinal": 2, "source_identity": "bad-undo",
            "type": "ITEM_UNDO", "participantId": 1, "afterId": 1003,
        }, 2),)
    ).query(3)["assigned_history_by_participant"][0]
    malformed_row = malformed_undo["event_types"].index("ITEM_UNDO")
    assert malformed_undo["event_malformed_counts"][malformed_row] == 1
    assert malformed_undo["field_malformed_counts"][malformed_row, before] == 1


def test_bank_observe_match_keeps_local_ragged_offsets_and_full_context_masks():
    from types import SimpleNamespace

    from studies.item_node_observed_20261002.bank import observe_match

    match = SimpleNamespace(
        item_events=lambda: (
            {"timestamp_ms": 1, "source_ordinal": 0, "source_identity": "a", "type": "ITEM_PURCHASED", "participantId": 1, "itemId": 1001},
            {"timestamp_ms": 2, "source_ordinal": 1, "source_identity": "b", "type": "ITEM_PURCHASED", "participantId": 2, "itemId": 2002},
        ),
        query_ms=np.asarray([1, 2], dtype=np.int64),
        metadata={},
        shard_index=0,
        local_index=0,
    )
    memberships = (1, 2, 4, 8, 16, 32)
    observed = observe_match(match, context_memberships=memberships)
    assert observed.context_memberships == memberships
    assert observed.states[0].offsets.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert observed.states[1].offsets.tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert observed.states[1].item_ids.tolist() == [1001, 2002]


def test_bank_nominal_reference_rejects_inbounds_future_timestamp():
    from studies.item_node_observed_20261002.bank import ObservedBankError, _nominal_refs

    rows = np.asarray(
        [(1, True, 100, 100)],
        dtype=[("query_ref", "i4"), ("supported", "?"), ("actual_ms", "i8"), ("nominal_ms", "i8")],
    )
    with pytest.raises(ObservedBankError, match="timestamp"):
        _nominal_refs(rows, 2, np.asarray([50, 200], dtype=np.int64))


def test_production_v3_known_empty_and_unknown_remain_distinct():
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    known = RecipientConfirmedItemLedger(
        [
            _row(1, 0, "buy", type="ITEM_PURCHASED", participantId=1, itemId=1001),
            _row(2, 1, "sell", type="ITEM_SOLD", participantId=1, itemId=1001),
        ]
    ).query(2)
    unknown = RecipientConfirmedItemLedger(
        [_row(1, 0, "bad-sell", type="ITEM_SOLD", participantId=1, itemId=1001)]
    ).query(1)
    assert _production_bag(known, 1) == {}
    assert _production_bag(unknown, 1) == {}
    assert bool(known["recorded_ledger_supported"][0]) is True
    assert bool(unknown["recorded_ledger_supported"][0]) is False
    assert known["full_inventory_supported"] is False
    assert unknown["full_inventory_supported"] is False


def test_production_v3_encoder_routes_player_history_and_unassigned_context():
    torch = pytest.importorskip("torch")
    from studies.item_node_observed_20261002.embedding import ObservedItemMultisetEmbedding, ItemVocabulary
    from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger

    rows = [
        _row(1, 0, "u", type="ITEM_PURCHASED", participantId=0, itemId=3865),
        _row(2, 1, "p1", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(3, 2, "p1-sold", type="ITEM_SOLD", participantId=1, itemId=1001),
    ]
    snapshot = RecipientConfirmedItemLedger(rows).query(3)
    vocabulary = ItemVocabulary.fit([1001], scope_keys=["fit"], provenance="FIT")
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=3)
    result = encoder.encode_snapshot(snapshot)
    assert tuple(result["node_features"].shape[:2]) == (1, 10)
    assert result["full_inventory_complete"] == 0.0
    assert result["full_inventory_unknown"] == 1.0
    assert result["unassigned_history_observed"] is True
    assert bool(torch.any(result["node_history_features"][0, 0]))
    assert torch.equal(result["node_history_features"][0, 1], torch.zeros_like(result["node_history_features"][0, 1]))
    assert torch.equal(result["current_node_features"][0, 1, :3], torch.zeros(3))
    assert 3865 not in np.asarray(snapshot["item_ids"]).tolist()


def test_persisted_bank_node_input_roundtrips_typed_player_histories(tmp_path):
    """A reloaded bank payload must be directly consumable by the v3 encoder."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from studies.item_node_observed_20261002.audit import _BankWriter
    from studies.item_node_observed_20261002.bank import FORMAT_VERSION, ObservedBankLoader, file_digest, observe_match
    from studies.item_node_observed_20261002.embedding import ItemVocabulary, ObservedItemMultisetEmbedding

    match = SimpleNamespace(
        item_events=lambda: (
            _row(1, 0, "p1", type="ITEM_PURCHASED", participantId=1, itemId=1001),
            _row(2, 1, "p2", type="ITEM_PURCHASED", participantId=2, itemId=2002),
            _row(3, 2, "unknown-owner", type="ITEM_PURCHASED", participantId=0, itemId=3865),
        ),
        query_ms=np.asarray([3], dtype=np.int64),
        metadata={},
        shard_index=0,
        local_index=0,
    )
    observed = observe_match(match, context_memberships=(1,) * 6)
    output = tmp_path / "bank"
    writer = _BankWriter(output, capacities=None)
    writer.write(0, observed, None)
    writer.finalize()
    (output / "BANK_METADATA.json").write_text(json.dumps({"format_version": FORMAT_VERSION}), encoding="utf-8")
    names = [
        "MATCH_TABLE.npy", "QUERY_TABLE.npy", "NODE_OFFSETS.npy", "NODE_SUPPORT.npy", "NODE_REASON_MASK.npy",
        "NOMINAL_TABLE.npy", "CONTEXT_MEMBERSHIP.npy", "ITEM_IDS.bin", "ITEM_COUNTS.bin",
        "EVENT_SOURCE_INDEX_ASSIGNED.bin", "EVENT_SOURCE_INDEX_UNASSIGNED.bin", "MATCH_STATS.jsonl",
    ]
    manifest = {"format_version": FORMAT_VERSION, "saved_files": {name: {"sha256": file_digest(output / name)} for name in names}}
    (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    with ObservedBankLoader(output) as loader:
        # This synthetic fixture has no accepted source shard.  Bind the
        # loader's reference resolver to the immutable events just written so
        # the test still exercises persisted event-index references, replay,
        # and the public encoder payload together.
        source_events = {
            int(event.event_index): event
            for event in (*observed.assigned_events, *observed.unassigned_events)
        }

        class _FixtureResolver:
            def events_for(self, shard_index, local_index):
                del shard_index, local_index
                return source_events

        loader._source_resolver = _FixtureResolver()
        payload = loader.node_input(0, 0)
        assert len(payload["assigned_history_by_participant"]) == 10
        assert payload["assigned_history_by_participant"][0]["valid_item_id_count"] == 1
        assert payload["assigned_history_by_participant"][1]["valid_item_id_count"] == 1
        assert payload["assigned_history_by_participant"][2]["valid_item_id_count"] == 0
        assert payload["unassigned_history_context"]["valid_item_id_count"] == 1
        assert payload["full_inventory_complete"] == "unknown"
        vocabulary = ItemVocabulary.fit([1001, 2002], scope_keys=["fit"], provenance="FIT")
        encoded = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=3).encode_snapshot(payload)
        assert tuple(encoded["node_history_features"].shape[:2]) == (1, 10)
        assert bool(torch.any(encoded["node_history_features"][0, 0]))
        assert bool(torch.any(encoded["node_history_features"][0, 1]))
        assert torch.equal(encoded["node_history_features"][0, 2], torch.zeros_like(encoded["node_history_features"][0, 2]))
        assert encoded["unassigned_history_observed"] is True


def test_independent_vocab_audit_loads_pointer_and_each_artifact(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from reports.item_node_observed_20261002.independent_verify import _BuiltinObservedBankAdapter

    contexts = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
    scopes = ("FIT", "REFIT_TRAIN")
    metadata = {}
    for context in contexts:
        for scope in scopes:
            key = f"{context}:{scope}"
            payload = {
                "version": "test-vocabulary",
                "pad_index": 0,
                "unk_index": 1,
                "item_ids": [1001],
                "scope_keys": [key],
                "fit_provenance": {key: scope},
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            path = tmp_path / f"VOCABULARY_{context}_{scope}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metadata[key] = {
                "context": context,
                "scope": scope,
                "path": path.name,
                "item_ids": [1001],
                "item_ids_sha256": hashlib.sha256(b"[1001]").hexdigest(),
                "vocabulary_fingerprint": hashlib.sha256(encoded).hexdigest(),
                "cutoff_sha256": "0" * 64,
                "included_match_count": 1,
            }
    pointer = tmp_path / "VOCABULARY_SCOPE_META.json"
    pointer.write_text(json.dumps(metadata), encoding="utf-8")
    bank = SimpleNamespace(
        root=tmp_path,
        metadata={"vocabulary_scope_metadata": pointer.name},
    )
    adapter = _BuiltinObservedBankAdapter()
    audited = adapter.vocabulary_audit(bank, tmp_path)
    assert set(audited) == {f"{context}:{scope}" for context in contexts for scope in scopes}
    assert audited["final:FIT"]["item_ids"] == [1001]
    assert audited["final:FIT"]["fit_provenance"] == {"final:FIT": "FIT"}
    assert len(audited["final:FIT"]["artifact_sha256"]) == 64

    tampered = json.loads((tmp_path / "VOCABULARY_final_FIT.json").read_text(encoding="utf-8"))
    tampered["item_ids"] = [2002]
    (tmp_path / "VOCABULARY_final_FIT.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AssertionError, match="IDs differ|fingerprint"):
        adapter.vocabulary_audit(bank, tmp_path)


def test_production_v3_cpu_forward_backward_reload_is_deterministic(tmp_path):
    torch = pytest.importorskip("torch")
    from studies.item_node_observed_20261002.embedding import ObservedItemMultisetEmbedding, ItemVocabulary

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    vocabulary = ItemVocabulary.fit([1001, 2002], scope_keys=["fit"], provenance="FIT")
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=4)
    indices = torch.tensor([2, 3], dtype=torch.long)
    counts = torch.tensor([2, 1], dtype=torch.long)
    offsets = torch.tensor([[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.long)
    support = torch.ones(1, 10, dtype=torch.bool)
    first = encoder(indices, counts, offsets, support)
    loss = first[0, 0, :4].sum()
    loss.backward()
    state_path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), state_path)
    restored = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=4)
    restored.load_state_dict(torch.load(state_path, map_location="cpu"))
    second = restored(indices, counts, offsets, support)
    assert torch.equal(first.detach(), second)
    assert torch.isfinite(first).all()
    assert torch.isfinite(encoder.item_embeddings.weight.grad).all()


@pytest.mark.parametrize("rows", [
    [
        _row(100, 0, "same", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(200, 1, "same", type="ITEM_PURCHASED", participantId=1, itemId=1002),
    ],
    [
        _row(100, 0, "same", type="ITEM_SOLD", participantId=1, itemId=1001),
        _row(200, 1, "same", type="ITEM_PURCHASED", participantId=1, itemId=1001),
    ],
    [
        _row(100, 0, "same", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(200, 1, "same", type="ITEM_PURCHASED", participantId=0, itemId=1002),
        _row(300, 2, "new", type="ITEM_UNDO", participantId=1, beforeId=1001, afterId=None),
    ],
    [
        _row(100, 0, "valid", type="ITEM_PURCHASED", participantId=1, itemId=1001),
        _row(100, 1, "invalid", type="ITEM_UNDO", participantId=1, beforeId=1001, afterId=None),
    ],
    [
        _row(100, 0, "flag", type="ITEM_PURCHASED", participantId=1, itemId=1001, duplicate_source_identity=True),
    ],
])
def test_full_verifier_reason_oracle_matches_bank_layers(rows):
    from types import SimpleNamespace
    from reports.item_node_observed_20261002.independent_verify import stream_literal_oracle
    from studies.item_node_observed_20261002.bank import REASON_BITS, observe_match

    queries = np.asarray(sorted({row["timestamp_ms"] for row in rows}), dtype=np.int64)
    match = SimpleNamespace(
        query_ms=queries,
        query_event_limit=np.searchsorted([row["timestamp_ms"] for row in rows], queries, side="right"),
        item_events=lambda: rows,
        metadata={"canonical_match_key": "a" * 64},
        shard_index=0, local_index=0,
    )
    production = observe_match(match, context_memberships=(1,) * 6)
    independent = list(stream_literal_oracle(rows, queries))
    for observed, (_, _, expected) in zip(production.states, independent, strict=True):
        for index, player in enumerate(expected["players"]):
            actual_reasons = {name for name, bit in REASON_BITS.items() if int(observed.reason_mask[index]) & (1 << bit)}
            assert actual_reasons == set(player["reasons"])
            assert bool(observed.supported[index]) == player["recorded_ledger_supported"]
            start, stop = int(observed.offsets[index]), int(observed.offsets[index + 1])
            assert observed.item_ids[start:stop].tolist() == player["item_ids"]
            assert observed.counts[start:stop].tolist() == player["counts"]
