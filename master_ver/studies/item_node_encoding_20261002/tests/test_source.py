from __future__ import annotations

import json

import numpy as np
import pytest

from studies.item_node_encoding_20261002.audit import _accepted_audit_status, iter_compact_ledger_states
from studies.item_node_encoding_20261002.ledger import ItemTransactionLedger
from studies.item_node_encoding_20261002.source import AcceptedMatch, SourceValidationError


def _match_arrays(*, bad_mask: bool = False, raw_item: bool = False) -> dict[str, np.ndarray]:
    timestamps = np.asarray([0, 100, 100, 200], dtype=np.int32)
    type_codes = np.asarray([8, 8, 10, 7], dtype=np.uint8)
    # participantId/itemId for the first two purchases, participantId plus
    # beforeId/afterId for undo, and participantId/itemId for destroy.
    values = {
        (0, 0): 1, (0, 1): 1001,
        (1, 0): 1, (1, 1): 1001,
        (2, 0): 1, (2, 2): 1001, (2, 3): 2003,
        (3, 0): 1, (3, 1): 2003,
    }
    rows = np.asarray([row for row, _ in values], dtype=np.int32)
    cols = np.asarray([col for _, col in values], dtype=np.uint8)
    nonzero = np.asarray(list(values.values()), dtype=np.int32)
    masks = np.asarray([3, 3, 13, 3], dtype=np.uint32)
    raw_rows = np.empty(0, dtype=np.int32)
    raw_cols = np.empty(0, dtype=np.int16)
    raw_offsets = np.zeros(1, dtype=np.int32)
    raw_values = np.empty(0, dtype=np.uint8)
    if raw_item:
        raw = json.dumps("bad-item").encode("utf-8")
        raw_rows = np.asarray([0], dtype=np.int32)
        raw_cols = np.asarray([1], dtype=np.int16)
        raw_offsets = np.asarray([0, len(raw)], dtype=np.int32)
        raw_values = np.frombuffer(raw, dtype=np.uint8).copy()
    if bad_mask:
        masks[0] = 1 << 15
    return {
        "event_timestamp": timestamps,
        "event_source_index": np.arange(4, dtype=np.int32),
        "event_type_code": type_codes,
        "event_category_code": np.zeros(4, dtype=np.int16),
        "event_field_width": np.asarray(15, dtype=np.int16),
        "event_field_row": rows,
        "event_field_col": cols,
        "event_field_nonzero": nonzero,
        "event_field_valid_mask": masks,
        "event_explicit_rows": np.empty(0, dtype=np.int32),
        "event_explicit_identity": np.empty(0, dtype=np.uint64),
        "event_identity_raw_rows": np.empty(0, dtype=np.int32),
        "event_identity_raw_offsets": np.zeros(1, dtype=np.int32),
        "event_identity_raw_values": np.empty(0, dtype=np.uint8),
        "event_raw_rows": raw_rows,
        "event_raw_cols": raw_cols,
        "event_raw_offsets": raw_offsets,
        "event_raw_values": raw_values,
        "item_event_indices": np.arange(4, dtype=np.int32),
        "query_ms": np.asarray([0, 100, 200], dtype=np.int32),
        "query_item_ledger_known": np.zeros(3, dtype=bool),
        "query_event_limit": np.asarray([1, 3, 4], dtype=np.int32),
        "v_nominal": np.asarray([[0, 0, 99, 0, 0, 0, 0]], dtype=np.int32),
    }


def test_source_preserves_masks_and_raw_payload_markers() -> None:
    match = AcceptedMatch.from_bank(_match_arrays(raw_item=True))
    first, second = match.item_events()[:2]
    assert first.item_id == "bad-item"
    assert first.field_validity[1] is False
    assert first.raw_field_validity["itemId"] is True
    assert second.item_id == 1001
    assert first.to_mapping()["itemId"] == "bad-item"


def test_invalid_mask_bits_and_incomplete_item_index_are_rejected() -> None:
    with pytest.raises(SourceValidationError, match="valid_mask"):
        AcceptedMatch.from_bank(_match_arrays(bad_mask=True))
    arrays = _match_arrays()
    arrays["item_event_indices"] = np.asarray([0, 1, 2], dtype=np.int32)
    with pytest.raises(SourceValidationError, match="complete recognized"):
        AcceptedMatch.from_bank(arrays)


def test_compact_replay_matches_public_ledger_queries() -> None:
    match = AcceptedMatch.from_bank(_match_arrays())
    ledger = ItemTransactionLedger(match.item_transactions())
    compact = list(iter_compact_ledger_states(ledger, match.query_ms))
    public = ledger.query_many(match.query_ms.tolist())
    assert len(compact) == len(public)
    for fast, reference in zip(compact, public):
        assert np.array_equal(fast.known, reference["known"])
        assert np.array_equal(fast.item_ids, reference["item_ids"])
        assert np.array_equal(fast.counts, reference["counts"])
        assert np.array_equal(fast.offsets, reference["offsets"])
        assert fast.reasons == reference["reasons"]


def test_compact_replay_rejects_nonempty_initial_ledger() -> None:
    ledger = ItemTransactionLedger([], initial_inventory={1: {1001: 1}})
    with pytest.raises(ValueError, match="nonempty"):
        list(iter_compact_ledger_states(ledger, [0]))


@pytest.mark.parametrize(
    ("complete", "hashes", "strict", "accepted"),
    ((True, True, True, True), (True, False, True, False), (True, True, False, False), (False, True, True, False)),
)
def test_only_strict_hash_bound_full_audit_is_accepted(complete, hashes, strict, accepted) -> None:
    assert _accepted_audit_status(complete, hashes, strict) is accepted
