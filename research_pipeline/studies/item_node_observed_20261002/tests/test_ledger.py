from __future__ import annotations

import numpy as np
import pytest

from studies.item_node_observed_20261002.ledger import (
    ITEM_EVENT_TYPES,
    ITEM_FIELD_ROLES,
    RecipientConfirmedItemLedger,
)


def _tx(timestamp, identity, pid, kind="ITEM_PURCHASED", *, item=None, before=None, after=None):
    return {
        "timestamp_ms": timestamp,
        "source_ordinal": timestamp,
        "source_identity": identity,
        "participantId": pid,
        "itemId": item,
        "beforeId": before,
        "afterId": after,
        "type": kind,
    }


def _items(snapshot, pid):
    player = snapshot["players"][pid - 1]
    return dict(zip(player["item_ids"].tolist(), player["counts"].tolist()))


def test_unassigned_source_zero_is_retained_without_global_poison_and_full_inventory_is_unknown():
    ledger = RecipientConfirmedItemLedger(
        [
            _tx(100, "known", 1, item=1001),
            _tx(200, "unknown", 0, item=1002),
            _tx(300, "known2", 2, item=1003),
        ]
    )
    before = ledger.query(100)
    after = ledger.query(300)
    assert before["known"].all()
    assert after["known"].all()
    assert _items(after, 1) == {1001: 1}
    assert _items(after, 2) == {1003: 1}
    assert after["unassigned_history_observed"] is True
    assert after["unassigned_history"][0]["participantId"] == 0
    assert after["full_inventory_complete"] == "unknown"
    assert after["actual_inventory_complete"] == "unknown"


def test_cross_partition_duplicate_taints_only_at_duplicate_prefix_and_taints_both_recipients():
    rows = [
        _tx(100, "cross", 1, item=1001),
        _tx(150, "unassigned", 0, item=1002),
        _tx(200, "cross", 2, item=1003),
    ]
    ledger = RecipientConfirmedItemLedger(rows)
    early = ledger.query(199)
    late = ledger.query(200)
    assert early["known"].all()
    assert _items(early, 1) == {1001: 1}
    assert late["known"][0] is np.bool_(False) or not bool(late["known"][0])
    assert not bool(late["known"][1])
    assert all("duplicate_source_identity" in reasons for reasons in late["reasons"][:2])
    assert late["item_ids"].size == 0
    prefix = RecipientConfirmedItemLedger(rows[:2]).query(150)
    assert bool(prefix["known"][0])
    assert _items(prefix, 1) == {1001: 1}


def test_identity_seen_only_in_unassigned_history_taints_later_recipient_prefix():
    rows = [
        _tx(100, "shared", 0, item=1001),
        _tx(200, "shared", 1, item=1002),
    ]
    ledger = RecipientConfirmedItemLedger(rows)
    early = ledger.query(100)
    late = ledger.query(200)
    assert bool(early["known"][0])
    assert not bool(late["known"][0])
    assert "duplicate_source_identity" in late["reasons"][0]
    assert late["item_ids"].size == 0


def test_same_ms_underflow_is_player_local_and_observed_history_remains_available():
    ledger = RecipientConfirmedItemLedger(
        [
            _tx(100, "p1buy", 1, item=1001),
            _tx(100, "p1sell", 1, "ITEM_SOLD", item=1001),
            _tx(100, "p2buy", 2, item=2001),
        ]
    )
    snapshot = ledger.query(100)
    assert not bool(snapshot["known"][0])
    assert bool(snapshot["known"][1])
    assert _items(snapshot, 2) == {2001: 1}
    assert len(snapshot["identified_history"]) == 3
    assert snapshot["assigned_history_context"]["event_counts"].sum() == 3


def test_compact_prefix_contains_arrays_and_no_history_copy():
    ledger = RecipientConfirmedItemLedger(
        [_tx(100, "a", 1, item=1001), _tx(200, "u", 0, item=1002)]
    )
    compact = list(ledger.iter_compact_observed_states([100, 200]))
    full = ledger.query_many([100, 200])
    assert [row["query_timestamp_ms"] for row in compact] == [100, 200]
    for row, expected in zip(compact, full):
        assert np.array_equal(row["item_ids"], expected["item_ids"])
        assert np.array_equal(row["counts"], expected["counts"])
        assert np.array_equal(row["offsets"], expected["offsets"])
        assert row["unassigned_count"] == len(expected["unassigned_history"])
        assert "identified_history" not in row


def test_history_context_has_declared_type_and_field_order_and_positive_id_vectors():
    snapshot = RecipientConfirmedItemLedger(
        [
            _tx(100, "buy", 1, "ITEM_PURCHASED", item=1001),
            _tx(200, "undo", 1, "ITEM_UNDO", before=1001, after=1002),
            _tx(300, "bad", 0, "ITEM_SOLD", item="bad"),
        ]
    ).query(300)
    context = snapshot["assigned_history_context"]
    assert context["event_types"] == ITEM_EVENT_TYPES
    assert context["field_roles"] == ITEM_FIELD_ROLES
    buy = ITEM_EVENT_TYPES.index("ITEM_PURCHASED")
    undo = ITEM_EVENT_TYPES.index("ITEM_UNDO")
    assert context["item_id_vectors"][buy][ITEM_FIELD_ROLES.index("itemId")].tolist() == [1001]
    assert context["item_id_vectors"][undo][ITEM_FIELD_ROLES.index("beforeId")].tolist() == [1001]
    assert context["item_id_vectors"][undo][ITEM_FIELD_ROLES.index("afterId")].tolist() == [1002]
    assert snapshot["unassigned_history_context"]["event_counts"].sum() == 1


def test_history_context_separates_presence_from_zero_and_malformed_fields():
    snapshot = RecipientConfirmedItemLedger(
        [
            _tx(100, "undo", 1, "ITEM_UNDO", before=0, after=1001),
            _tx(200, "bad", 1, "ITEM_PURCHASED", item=0),
        ]
    ).query(200)
    context = snapshot["assigned_history_by_participant"][0]
    undo = ITEM_EVENT_TYPES.index("ITEM_UNDO")
    purchased = ITEM_EVENT_TYPES.index("ITEM_PURCHASED")
    before = ITEM_FIELD_ROLES.index("beforeId")
    item = ITEM_FIELD_ROLES.index("itemId")
    assert context["field_presence_counts"][undo, before] == 1
    assert context["field_validity_counts"][undo, before] == 0
    assert context["field_malformed_counts"][undo, before] == 0
    assert context["field_presence_counts"][purchased, item] == 1
    assert context["field_validity_counts"][purchased, item] == 0
    assert context["field_malformed_counts"][purchased, item] == 1
    assert context["event_malformed_counts"][undo] == 0


@pytest.mark.parametrize(
    ("before", "remove_before"),
    [(None, False), ("bad", False), (None, True)],
)
def test_undo_missing_or_malformed_endpoint_marks_event_and_preserves_positive_token(before, remove_before):
    row = _tx(100, "undo", 1, "ITEM_UNDO", before=before, after=1001)
    if remove_before:
        row.pop("beforeId")
    snapshot = RecipientConfirmedItemLedger([row]).query(100)
    context = snapshot["assigned_history_by_participant"][0]
    undo = ITEM_EVENT_TYPES.index("ITEM_UNDO")
    before_index = ITEM_FIELD_ROLES.index("beforeId")
    after_index = ITEM_FIELD_ROLES.index("afterId")
    assert not bool(snapshot["known"][0])
    assert "invalid_undo_payload" in snapshot["reasons"][0]
    assert context["event_malformed_counts"][undo] == 1
    assert context["field_malformed_counts"][undo, before_index] == 1
    assert context["item_id_vectors"][undo][after_index].tolist() == [1001]
