from __future__ import annotations

import torch
import pytest

from studies.item_node_observed_20261002 import (
    ObservedItemMultisetEmbedding,
    RecipientConfirmedItemLedger,
    UNK_INDEX,
    fit_item_vocabulary,
)


def test_observed_encoder_reuses_fit_refit_vocab_and_explicit_dimension():
    vocabulary = fit_item_vocabulary(
        {"train": [1002, 1001], "refit": [1003]},
        provenance={"train": "FIT", "refit": "REFIT_TRAIN"},
    )
    assert vocabulary.transform([1001, 9999]).tolist() == [2, UNK_INDEX]
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=3)
    assert encoder.embedding_dim == 3
    assert encoder.output_dim == 5
    assert encoder.current_output_dim == 5
    assert encoder.context_output_dim == 3 * 12 + 4 + 4 + 12 + 12 + 12 + 4 + 1


def test_unknown_ledger_known_empty_and_oov_are_distinct():
    vocabulary = fit_item_vocabulary([1001], scope_keys=["train"], provenance="FIT")
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[UNK_INDEX].fill_(3.0)
        encoder.unknown_ledger_embedding.fill_(7.0)
    empty = encoder(torch.empty(0, dtype=torch.long), torch.empty(0), torch.zeros((1, 11), dtype=torch.long), torch.ones(1, 10, dtype=torch.bool))
    unknown = encoder(torch.empty(0, dtype=torch.long), torch.empty(0), torch.zeros((1, 11), dtype=torch.long), torch.zeros(1, 10, dtype=torch.bool))
    oov = encoder(torch.tensor([UNK_INDEX]), torch.tensor([1]), torch.tensor([[0, 1] + [1] * 9]), torch.ones(1, 10, dtype=torch.bool))
    assert torch.equal(empty[0, 0, :2], torch.zeros(2))
    assert torch.equal(empty[0, 0, 2:], torch.tensor([0.0, 1.0]))
    assert torch.equal(unknown[0, 0, :2], torch.tensor([7.0, 7.0]))
    assert torch.equal(unknown[0, 0, 2:], torch.tensor([0.0, 0.0]))
    assert torch.equal(oov[0, 0, :2], torch.tensor([3.0, 3.0]))


def test_snapshot_embedding_keeps_unassigned_context_graph_level_and_routes_bag_to_exact_player():
    vocabulary = fit_item_vocabulary([1001, 1002], scope_keys=["train"], provenance="FIT")
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[2] = torch.tensor([1.0, 0.0])
        encoder.item_embeddings.weight[3] = torch.tensor([0.0, 1.0])
    snapshot = RecipientConfirmedItemLedger(
        [
            {"timestamp_ms": 100, "source_identity": "p1", "participantId": 1, "itemId": 1001, "type": "ITEM_PURCHASED"},
            {"timestamp_ms": 100, "source_identity": "u", "participantId": 0, "itemId": 1002, "type": "ITEM_PURCHASED"},
        ]
    ).query(100)
    encoded = encoder.encode_snapshot(snapshot)
    assert encoded["node_features"].shape == (1, 10, encoder.combined_output_dim)
    assert encoded["current_node_features"].shape == (1, 10, 4)
    assert torch.equal(encoded["current_node_features"][0, 0, :2], torch.tensor([1.0, 0.0]))
    assert torch.equal(encoded["current_node_features"][0, 1, :2], torch.zeros(2))
    assert encoded["node_history_features"].shape == (1, 10, encoder.context_output_dim)
    assert not torch.equal(encoded["node_history_features"][0, 0], encoded["node_history_features"][0, 1])
    assert encoded["unassigned_history_observed"] is True
    assert encoded["unassigned_context"].ndim == 1
    # The unassigned token is represented in graph context only and cannot
    # change any champion node's current observed bag.
    assert torch.equal(encoded["current_node_features"][0, 1, :2], torch.zeros(2))


def test_snapshot_embedding_requires_personal_history_and_unassigned_context():
    vocabulary = fit_item_vocabulary([1001], scope_keys=["train"], provenance="FIT")
    encoder = ObservedItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with pytest.raises(ValueError, match="assigned_history_by_participant"):
        encoder.encode_snapshot({"item_ids": [], "counts": [], "offsets": [0] * 11, "recorded_ledger_supported": [True] * 10, "unassigned_history_context": {}})
