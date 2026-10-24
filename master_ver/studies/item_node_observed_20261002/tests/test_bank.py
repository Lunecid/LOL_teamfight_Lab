from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from studies.item_node_observed_20261002.audit import _BankWriter, _vocabulary_scope_update
from studies.item_node_observed_20261002.bank import FORMAT_VERSION, ObservedBankError, ObservedBankLoader, file_digest, observe_match, partition_item_events


class _Match:
    metadata = {
        "canonical_match_key": "a" * 64,
        "patch": "15.14",
        "role": "TRAIN",
        "original_index": 0,
    }
    shard_index = 0
    local_index = 0

    def __init__(self, events, queries=(100, 200, 300)):
        self._events = tuple(events)
        self.query_ms = np.asarray(queries, dtype=np.int64)

    def item_events(self):
        return self._events


def _row(t, ordinal, identity, **payload):
    return {
        "timestamp_ms": t,
        "source_ordinal": ordinal,
        "source_identity": identity,
        **payload,
    }


def test_source_zero_is_unassigned_and_does_not_poison_recipient_ledger():
    match = _Match(
        [
            _row(100, 1, "identified", participantId=1, itemId=101, type="ITEM_PURCHASED"),
            _row(100, 2, "source-zero", participantId=0, itemId=202, type="ITEM_PURCHASED"),
        ],
        queries=(100,),
    )
    result = observe_match(match)
    state = result.states[0]
    assert state.supported.all()
    assert state.item_ids.tolist() == [101]
    assert state.unassigned_observed
    assert len(result.unassigned_events) == 1
    assert result.unassigned_events[0].unassigned_reason == "zero_participant"


def test_duplicate_identity_across_partitions_taints_only_at_duplicate_cutoff():
    match = _Match(
        [
            _row(100, 1, "shared", participantId=1, itemId=101, type="ITEM_PURCHASED"),
            _row(200, 2, "shared", participantId=0, itemId=202, type="ITEM_PURCHASED"),
        ]
    )
    result = observe_match(match)
    assert result.states[0].supported[0]
    assert not result.states[1].supported[0]
    assert "duplicate_source_identity" in result.states[1].reasons[0]
    assert result.states[0].item_ids.tolist() == [101]
    assert result.states[1].item_ids.size == 0


def test_prefix_endpoint_matches_full_replay_prefix_and_keeps_typed_history_refs():
    events = [
        _row(100, 1, "buy", participantId=1, itemId=101, type="ITEM_PURCHASED"),
        _row(300, 2, "undo", participantId=1, beforeId=101, afterId=202, type="ITEM_UNDO"),
    ]
    early = observe_match(_Match(events, queries=(100,))).__dict__
    full = observe_match(_Match(events, queries=(100, 300)))
    assert full.states[0].item_ids.tolist() == early["states"][0].item_ids.tolist()
    assert full.states[0].counts.tolist() == early["states"][0].counts.tolist()
    assert full.states[0].assigned_history_end == 1
    assert full.states[1].assigned_history_end == 2
    history = full.history_prefix(1)
    assert history["assigned"][1]["event_type"] == "ITEM_UNDO"
    assert history["assigned"][1]["after_id"] == 202


def test_partition_marks_same_ms_identity_duplicate_in_both_partitions():
    assigned, unassigned = partition_item_events(
        [
            _row(100, 1, "same", participantId=1, itemId=101, type="ITEM_PURCHASED"),
            _row(100, 2, "same", participantId=0, itemId=101, type="ITEM_PURCHASED"),
        ]
    )
    assert assigned[0].duplicate_source_identity
    assert unassigned[0].duplicate_source_identity


def test_transaction_mapping_preserves_missing_and_explicit_payload_fields():
    missing = partition_item_events([_row(100, 1, "missing", participantId=1, type="ITEM_PURCHASED")])[0][0]
    explicit = partition_item_events([_row(100, 1, "explicit", participantId=1, itemId=0, type="ITEM_PURCHASED")])[0][0]
    assert "itemId" not in missing.transaction_mapping()
    assert explicit.transaction_mapping()["itemId"] == 0


def test_writer_and_public_loader_keep_ragged_inputs_and_history_refs():
    observed = observe_match(_Match([_row(100, 1, "buy", participantId=1, itemId=101, type="ITEM_PURCHASED")], queries=(100, 200)), context_memberships=(1,) * 6)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        output = Path(temporary) / "bank"
        writer = _BankWriter(output, capacities=None)
        writer.write(0, observed, None)
        writer.finalize()
        (output / "BANK_METADATA.json").write_text(json.dumps({"format_version": FORMAT_VERSION}), encoding="utf-8")
        names = [
            "MATCH_TABLE.npy", "QUERY_TABLE.npy", "NODE_OFFSETS.npy", "NODE_SUPPORT.npy", "NODE_REASON_MASK.npy",
            "NOMINAL_TABLE.npy", "CONTEXT_MEMBERSHIP.npy", "ITEM_IDS.bin", "ITEM_COUNTS.bin",
            "EVENT_SOURCE_INDEX_ASSIGNED.bin", "EVENT_SOURCE_INDEX_UNASSIGNED.bin",
        ]
        manifest = {"format_version": FORMAT_VERSION, "saved_files": {name: {"sha256": file_digest(output / name)} for name in names}}
        (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with ObservedBankLoader(output) as loader:
            first = loader.query(0)
            second = loader.query(1)
            assert first["item_ids"].tolist() == [101]
            assert first["offsets"][0] == 0
            assert second["offsets"][0] == 0
            assert second["offsets"][-1] == len(second["item_ids"])
            assert first["supported"].all()
            assert np.fromfile(output / "EVENT_SOURCE_INDEX_ASSIGNED.bin", dtype="<i4").tolist() == [0]


def test_writer_persists_each_context_membership_column():
    observed = observe_match(
        _Match([_row(100, 1, "buy", participantId=1, itemId=101, type="ITEM_PURCHASED")], queries=(100,)),
        context_memberships=(1, 2, 4, 8, 16, 32),
    )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        output = Path(temporary) / "bank"
        writer = _BankWriter(output, capacities=None)
        writer.write(0, observed, None)
        writer.finalize()
        assert np.load(output / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False).tolist() == [[1], [2], [4], [8], [16], [32]]


def test_incomplete_context_memberships_cannot_be_written_or_fit(tmp_path):
    observed = observe_match(_Match([], queries=(100,)))
    incomplete = replace(observed, context_memberships=(1,))
    writer = _BankWriter(tmp_path / "bank", capacities=None)
    with pytest.raises(ObservedBankError, match="six context"):
        writer.write(0, incomplete, None)
    assert writer.finalize()["n_matches"] == 0
    with pytest.raises(ObservedBankError, match="six context"):
        _vocabulary_scope_update({}, incomplete, {}, {})
