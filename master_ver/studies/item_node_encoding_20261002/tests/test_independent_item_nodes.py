from __future__ import annotations

import copy
import itertools
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pytest
import torch

torch.set_num_threads(1)

from studies.item_node_encoding_20261002.embedding import (
    ItemMultisetEmbedding,
    ItemVocabulary,
    PAD_INDEX,
    UNK_INDEX,
    attach_item_embedding_to_nodes,
    fit_item_vocabulary,
)
from studies.item_node_encoding_20261002.ledger import ItemTransaction, ItemTransactionLedger


def _tx(timestamp: int, identity: str, pid: int, kind: str, *, item=None, before=None, after=None):
    return {
        "timestamp_ms": timestamp,
        "source_ordinal": int(identity[1:]) if identity[1:].isdigit() else timestamp,
        "source_identity": identity,
        "participantId": pid,
        "itemId": item,
        "beforeId": before,
        "afterId": after,
        "type": kind,
    }


def _player_multiset(snapshot: dict, participant_id: int) -> dict[int, int]:
    player = snapshot["players"][participant_id - 1]
    return dict(zip(player["item_ids"].tolist(), player["counts"].tolist()))


def _single_batch_offsets(*ends: int) -> torch.Tensor:
    """Build [1, 11] offsets from ten explicit node endpoints."""
    assert len(ends) == 10
    return torch.tensor([[0, *ends]], dtype=torch.long)


def _fixed_encoder() -> tuple[ItemVocabulary, ItemMultisetEmbedding]:
    vocabulary = ItemVocabulary.fit(
        [101, 202, 303],
        scope_keys=["patch_15_14_train"],
        provenance="FIT",
    )
    encoder = ItemMultisetEmbedding(vocabulary, embedding_dim=3)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[1] = torch.tensor([50.0, 50.0, 50.0])
        encoder.item_embeddings.weight[2] = torch.tensor([1.0, 2.0, 3.0])
        encoder.item_embeddings.weight[3] = torch.tensor([10.0, 20.0, 30.0])
        encoder.item_embeddings.weight[4] = torch.tensor([100.0, 200.0, 300.0])
        encoder.unknown_ledger_embedding.copy_(torch.tensor([7.0, 8.0, 9.0]))
    return vocabulary, encoder


def test_every_same_ms_permutation_has_the_literal_certified_endpoint():
    rows = [
        _tx(100, "s1", 1, "ITEM_SOLD", item=1001),
        _tx(100, "s2", 1, "ITEM_DESTROYED", item=1001),
        _tx(100, "s3", 1, "ITEM_UNDO", before=1002, after=1003),
        _tx(100, "s4", 1, "ITEM_PURCHASED", item=1004),
    ]

    # Starting counts dominate every removal in the block.  The expected
    # endpoint is calculated from the source deltas, not from ledger output.
    expected = {1003: 1, 1004: 1}
    for permutation in itertools.permutations(rows):
        snapshot = ItemTransactionLedger(
            permutation,
            initial_inventory={1: {1001: 2, 1002: 1}},
        ).query(100)
        assert bool(snapshot["known"][0])
        assert _player_multiset(snapshot, 1) == expected


def test_distinct_same_ms_identical_buys_preserve_count_two():
    snapshot = ItemTransactionLedger(
        [
            _tx(100, "s1", 1, "ITEM_PURCHASED", item=5001),
            _tx(100, "s2", 1, "ITEM_PURCHASED", item=5001),
        ]
    ).query(100)

    assert bool(snapshot["known"][0])
    assert _player_multiset(snapshot, 1) == {5001: 2}
    assert [row["source_identity"] for row in snapshot["transactions"]] == ["s1", "s2"]


def test_same_ms_duplicate_source_records_are_retained_in_full_audit():
    snapshot = ItemTransactionLedger(
        [
            _tx(100, "s10", 1, "ITEM_PURCHASED", item=5050),
            _tx(100, "s10", 1, "ITEM_PURCHASED", item=5050),
        ]
    ).query(100)

    assert len(snapshot["transactions"]) == 2
    assert [row["source_identity"] for row in snapshot["transactions"]] == ["s10", "s10"]
    assert not bool(snapshot["known"][0])


def test_future_duplicate_source_does_not_retroactively_change_early_state_or_embedding():
    early_row = _tx(100, "s11", 1, "ITEM_PURCHASED", item=5151)
    with_future_duplicate = ItemTransactionLedger(
        [early_row, _tx(200, "s11", 1, "ITEM_PURCHASED", item=5252)]
    )
    clean = ItemTransactionLedger([early_row])
    early = with_future_duplicate.query(100)
    clean_early = clean.query(100)
    late = with_future_duplicate.query(200)

    assert bool(early["known"][0])
    assert bool(early["recorded_ledger_supported"][0])
    assert _player_multiset(early, 1) == _player_multiset(clean_early, 1) == {5151: 1}
    assert [row["source_identity"] for row in early["transactions"]] == ["s11"]
    assert not bool(late["known"][0])
    assert bool(late["unknown"][0])

    vocabulary = ItemVocabulary.fit([5151], scope_keys=["fit"], provenance="FIT")
    encoder = ItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[2] = torch.tensor([4.0, 5.0])
        encoder.unknown_ledger_embedding.copy_(torch.tensor([8.0, 9.0]))

    def encode_snapshot(snapshot):
        indices = torch.as_tensor(vocabulary.transform(snapshot["item_ids"].tolist()), dtype=torch.long)
        counts = torch.as_tensor(snapshot["counts"], dtype=torch.long)
        offsets = torch.as_tensor(snapshot["offsets"], dtype=torch.long).reshape(1, 11)
        support = torch.as_tensor(snapshot["recorded_ledger_supported"], dtype=torch.bool).reshape(1, 10)
        return encoder(indices, counts, offsets, support)

    assert torch.equal(encode_snapshot(early), encode_snapshot(clean_early))
    assert torch.equal(encode_snapshot(late)[0, 0], torch.tensor([8.0, 9.0, 0.0, 0.0]))


def test_item_ownership_is_player_local_even_when_item_ids_match():
    ledger = ItemTransactionLedger(
        [
            _tx(100, "s1", 1, "ITEM_PURCHASED", item=6001),
            _tx(100, "s2", 2, "ITEM_PURCHASED", item=6001),
            _tx(200, "s3", 1, "ITEM_SOLD", item=6001),
        ]
    )

    snapshot = ledger.query(200)
    assert bool(snapshot["known"][0]) and bool(snapshot["known"][1])
    assert _player_multiset(snapshot, 1) == {}
    assert _player_multiset(snapshot, 2) == {6001: 1}
    assert snapshot["offsets"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_unknown_is_persistent_and_distinct_from_known_empty():
    ledger = ItemTransactionLedger(
        [
            # Participant 1 starts empty, so this same-ms sell cannot be
            # certified even though the purchase is in the same block.
            _tx(100, "s1", 1, "ITEM_PURCHASED", item=7001),
            _tx(100, "s2", 1, "ITEM_SOLD", item=7001),
            _tx(200, "s3", 1, "ITEM_PURCHASED", item=7002),
            _tx(100, "s4", 2, "ITEM_PURCHASED", item=8001),
            _tx(200, "s5", 2, "ITEM_SOLD", item=8001),
        ]
    )
    snapshot = ledger.query(200)

    assert not bool(snapshot["known"][0])
    assert bool(snapshot["unknown"][0])
    assert _player_multiset(snapshot, 1) == {}
    assert snapshot["players"][0]["recorded_ledger_supported"] is False

    assert bool(snapshot["known"][1])
    assert not bool(snapshot["unknown"][1])
    assert _player_multiset(snapshot, 2) == {}
    assert snapshot["players"][1]["recorded_ledger_supported"] is True
    assert snapshot["players"][1]["item_ids"].size == 0
    assert snapshot["actual_inventory_complete"] == "unknown"
    assert snapshot["provenance"]["actual_inventory_complete"] == "unknown"


def test_endpoint_cutoff_is_exact_and_ragged_inventory_keeps_all_items():
    rows = [_tx(100, f"s{i}", 1, "ITEM_PURCHASED", item=9000 + i) for i in range(1, 8)]
    rows.extend(
        [
            _tx(200, "s8", 1, "ITEM_PURCHASED", item=9010),
            _tx(300, "s9", 1, "ITEM_PURCHASED", item=9020),
        ]
    )
    ledger = ItemTransactionLedger(rows)

    before = ledger.query(99)
    at_first = ledger.query(100)
    between = ledger.query(250)

    assert _player_multiset(before, 1) == {}
    assert _player_multiset(at_first, 1) == {9000 + i: 1 for i in range(1, 8)}
    assert at_first["offsets"].tolist()[1] == 7
    assert [row["source_identity"] for row in at_first["transactions"]] == [
        f"s{i}" for i in range(1, 8)
    ]
    assert _player_multiset(between, 1) == {
        **{9000 + i: 1 for i in range(1, 8)},
        9010: 1,
    }
    assert 9020 not in _player_multiset(between, 1)
    assert all(row["timestamp_ms"] <= 250 for row in between["transactions"])


def test_vocabulary_fit_excludes_test_oof_and_keeps_oov_explicit():
    vocabulary = fit_item_vocabulary(
        {"patch_15_14_train": [101, 202], "patch_15_15_validation": [303]},
        provenance={"patch_15_14_train": "FIT", "patch_15_15_validation": "REFIT_TRAIN"},
    )
    assert vocabulary.item_ids == (101, 202, 303)
    assert vocabulary.transform([101, 404]).tolist() == [2, UNK_INDEX]
    audit = vocabulary.transform_with_audit([404])
    assert audit["raw_item_ids"].tolist() == [404]
    assert audit["indices"].tolist() == [UNK_INDEX]

    with pytest.raises(ValueError, match="TEST"):
        fit_item_vocabulary([999], scope_keys=["patch_15_16_TEST"], provenance="TEST")
    with pytest.raises(ValueError, match="excluded"):
        fit_item_vocabulary(
            {"patch_15_14_train": [101], "patch_15_16_oof": [404]},
            provenance={"patch_15_14_train": "FIT", "patch_15_16_oof": "OOF"},
            excluded_scope_keys=["patch_15_16_oof"],
        )


@pytest.mark.parametrize("label", ["V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "STOP", "", "TRAIN", "OOF", "arbitrary"])
def test_vocabulary_rejects_noncanonical_fit_labels(label):
    with pytest.raises(ValueError):
        fit_item_vocabulary([101], scope_keys=["fit_scope"], provenance=label)


def test_typed_fractional_timestamp_and_negative_undo_cannot_bypass_normalization():
    with pytest.raises(ValueError):
        typed = ItemTransaction(
            timestamp_ms=100.5,
            source_ordinal=1,
            source_identity="typed",
            participant_id=1,
            item_id=1234,
            before_id=None,
            after_id=None,
            type="ITEM_PURCHASED",
        )
        ItemTransactionLedger([typed])

    with pytest.raises(ValueError):
        ItemTransaction(
            timestamp_ms=100,
            source_ordinal=2,
            source_identity="typed-undo",
            participant_id=1,
            item_id=None,
            before_id=-1,
            after_id=2345,
            type="ITEM_UNDO",
        )

    negative_undo = ItemTransactionLedger(
        [_tx(100, "s1", 1, "ITEM_UNDO", before=-1, after=2345)]
    ).query(100)
    assert not bool(negative_undo["known"][0])
    assert "invalid_undo_payload" in negative_undo["reasons"][0]
    assert _player_multiset(negative_undo, 1) == {}


def test_fixed_weights_distinguish_known_empty_oov_and_unknown_ledger():
    _, encoder = _fixed_encoder()
    # Node 0: 2x item 101 and 1x item 202; node 1: known empty;
    # node 2: unknown ledger; node 3: one fitted-vocabulary OOV item.
    indices = torch.tensor([2, 3, UNK_INDEX], dtype=torch.long)
    counts = torch.tensor([2, 1, 1], dtype=torch.long)
    offsets = _single_batch_offsets(2, 2, 2, 3, 3, 3, 3, 3, 3, 3)
    support = torch.ones(1, 10, dtype=torch.bool)
    support[0, 2] = False

    encoded = encoder(indices, counts, offsets, support)
    assert encoded.shape == (1, 10, 5)
    assert torch.equal(encoded[0, 0], torch.tensor([12.0, 24.0, 36.0, 3.0, 1.0]))
    assert torch.equal(encoded[0, 1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(encoded[0, 2], torch.tensor([7.0, 8.0, 9.0, 0.0, 0.0]))
    assert torch.equal(encoded[0, 3], torch.tensor([50.0, 50.0, 50.0, 1.0, 1.0]))


def test_attach_preserves_batch_and_routes_only_to_matching_participant_axis():
    _, encoder = _fixed_encoder()
    # Batch 0 has items at participant 1 and 10. Batch 1 has one item at
    # participant 7. Every other row must retain only its own node features
    # and the supported-empty marker.
    indices = torch.tensor([2, 3, 4], dtype=torch.long)
    counts = torch.tensor([1, 2, 1], dtype=torch.long)
    offsets = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    node_features = torch.arange(20, dtype=torch.float32).reshape(2, 10, 1)
    attached = attach_item_embedding_to_nodes(
        node_features,
        encoder,
        indices,
        counts,
        offsets,
    )

    assert attached.shape == (2, 10, 6)
    assert torch.equal(attached[..., 0], node_features[..., 0])
    assert torch.equal(attached[0, 0, 1:], torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0]))
    assert torch.equal(attached[0, 9, 1:], torch.tensor([20.0, 40.0, 60.0, 2.0, 1.0]))
    assert torch.equal(attached[1, 6, 1:], torch.tensor([100.0, 200.0, 300.0, 1.0, 1.0]))
    for batch, node in ((0, 1), (0, 8), (1, 0), (1, 9)):
        assert torch.equal(attached[batch, node, 1:], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_gradients_and_vocab_model_reload_preserve_fixed_outputs(tmp_path):
    vocabulary, encoder = _fixed_encoder()
    indices = torch.tensor([2, 3], dtype=torch.long)
    counts = torch.tensor([2, 1], dtype=torch.long)
    offsets = _single_batch_offsets(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    support = torch.ones(1, 10, dtype=torch.bool)
    support[0, 1] = False

    encoded = encoder(indices, counts, offsets, support)
    loss = encoded[0, 0, :3].sum() + encoded[0, 1, :3].sum()
    loss.backward()
    assert torch.equal(encoder.item_embeddings.weight.grad[PAD_INDEX], torch.zeros(3))
    assert torch.equal(encoder.item_embeddings.weight.grad[2], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.equal(encoder.item_embeddings.weight.grad[3], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(encoder.item_embeddings.weight.grad[UNK_INDEX], torch.zeros(3))
    assert torch.equal(encoder.unknown_ledger_embedding.grad, torch.ones(3))

    vocab_path = tmp_path / "item_vocab.json"
    state_path = tmp_path / "item_encoder.pt"
    vocabulary.save(vocab_path)
    torch.save(copy.deepcopy(encoder.state_dict()), state_path)
    restored_vocab = ItemVocabulary.load(vocab_path)
    restored = ItemMultisetEmbedding(restored_vocab, embedding_dim=3)
    restored.load_state_dict(torch.load(state_path, map_location="cpu"))

    assert restored_vocab.to_dict() == vocabulary.to_dict()
    assert restored_vocab.fingerprint == vocabulary.fingerprint
    assert torch.equal(
        restored(indices, counts, offsets, support),
        encoded.detach(),
    )
