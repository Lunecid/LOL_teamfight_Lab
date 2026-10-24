from __future__ import annotations

import pytest

from studies.item_node_encoding_20261002.ledger import ItemTransaction, ItemTransactionLedger


def _tx(timestamp, identity, pid, kind, *, item=None, before=None, after=None):
    return {
        "timestamp_ms": timestamp,
        "source_ordinal": int(identity.replace("s", "")) if identity.startswith("s") and identity[1:].isdigit() else timestamp,
        "source_identity": identity,
        "participantId": pid,
        "itemId": item,
        "beforeId": before,
        "afterId": after,
        "type": kind,
    }


def _player_items(snapshot, pid):
    row = snapshot["players"][pid - 1]
    return dict(zip(row["item_ids"].tolist(), row["counts"].tolist()))


def test_same_ms_permutations_and_mixed_block_are_certified_from_start_counts():
    rows = [
        _tx(100, "s1", 1, "ITEM_SOLD", item=1001),
        _tx(100, "s2", 1, "ITEM_UNDO", before=1002, after=1003),
        _tx(100, "s3", 1, "ITEM_PURCHASED", item=1004),
    ]
    initial = {1: {1001: 1, 1002: 1}}
    expected = None
    for permutation in (rows, [rows[2], rows[0], rows[1]], [rows[1], rows[2], rows[0]]):
        snapshot = ItemTransactionLedger(rows if permutation is rows else permutation, initial_inventory=initial).query(100)
        assert bool(snapshot["known"][0])
        actual = _player_items(snapshot, 1)
        if expected is None:
            expected = actual
        assert actual == expected == {1003: 1, 1004: 1}


def test_pure_purchases_qualify_and_duplicate_payloads_need_distinct_identities():
    snapshot = ItemTransactionLedger(
        [_tx(100, "s1", 1, "ITEM_PURCHASED", item=1001), _tx(100, "s2", 1, "ITEM_PURCHASED", item=1001)]
    ).query(100)
    assert bool(snapshot["known"][0])
    assert _player_items(snapshot, 1) == {1001: 2}

    repeated = ItemTransactionLedger(
        [_tx(100, "same", 1, "ITEM_PURCHASED", item=1001), _tx(200, "same", 1, "ITEM_PURCHASED", item=1001)]
    )
    assert bool(repeated.query(100)["known"][0])
    assert not bool(repeated.query(200)["known"][0])
    distinct = ItemTransactionLedger(
        [_tx(100, "s1", 1, "ITEM_PURCHASED", item=1001), _tx(200, "s2", 1, "ITEM_PURCHASED", item=1001)]
    ).query(200)
    assert bool(distinct["known"][0])
    assert _player_items(distinct, 1) == {1001: 2}

    # A later repeated identity cannot retroactively invalidate the earlier
    # prefix; the second occurrence quarantines the state from its endpoint.
    future_duplicate = ItemTransactionLedger(
        [_tx(100, "same", 1, "ITEM_PURCHASED", item=1001), _tx(200, "same", 1, "ITEM_PURCHASED", item=1001)]
    )
    assert future_duplicate.query(100)["known"][0]
    assert not future_duplicate.query(200)["known"][0]


def test_same_ms_underflow_is_persistent_and_isolated_by_participant():
    ledger = ItemTransactionLedger(
        [
            _tx(100, "s1", 1, "ITEM_PURCHASED", item=1001),
            _tx(100, "s2", 1, "ITEM_SOLD", item=1001),
            _tx(200, "s3", 1, "ITEM_PURCHASED", item=1002),
            _tx(200, "s4", 2, "ITEM_PURCHASED", item=1003),
        ]
    )
    at_first = ledger.query(100)
    at_later = ledger.query(200)
    assert not bool(at_first["known"][0])
    assert "same_timestamp_block_not_order_independent" in at_first["reasons"][0]
    assert not bool(at_later["known"][0])
    assert bool(at_later["known"][1])
    assert _player_items(at_later, 2) == {1003: 1}


def test_invalid_pid_marks_all_players_unknown_without_future_recovery():
    ledger = ItemTransactionLedger(
        [_tx(100, "s1", 1, "ITEM_PURCHASED", item=1001), _tx(200, "s2", 0, "ITEM_PURCHASED", item=1002)]
    )
    before = ledger.query(100)
    after = ledger.query(200)
    assert before["known"].all()
    assert before["actual_inventory_complete"] == "unknown"
    assert not after["known"].any()
    assert all("invalid_participant_id" in reasons for reasons in after["reasons"])


def test_cutoff_includes_full_block_and_ragged_endpoint_is_uncapped_and_auditable():
    rows = [_tx(100, f"s{i}", 1, "ITEM_PURCHASED", item=2000 + i) for i in range(7)]
    rows.append(_tx(200, "s9", 1, "ITEM_PURCHASED", item=9999))
    ledger = ItemTransactionLedger(rows)
    before = ledger.query(99)
    at_block = ledger.query(100)
    assert _player_items(before, 1) == {}
    assert _player_items(at_block, 1) == {2000 + i: 1 for i in range(7)}
    assert at_block["offsets"].tolist()[1] == 7
    assert len(at_block["transactions"]) == 7
    assert _player_items(ledger.query(200), 1)[9999] == 1
    with pytest.raises(ValueError, match="sorted"):
        ledger.query_many([200, 100])


def test_typed_transactions_validate_timestamp_and_undo_endpoints():
    with pytest.raises(ValueError, match="timestamp_ms"):
        ItemTransaction(1.5, 0, "s1", 1, 1001, None, None, "ITEM_PURCHASED")
    with pytest.raises(ValueError, match="before_id"):
        ItemTransaction(1, 0, "s1", 1, None, -1, 1001, "ITEM_UNDO")
