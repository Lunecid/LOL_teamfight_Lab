from __future__ import annotations

import pytest
import torch

from studies.item_node_encoding_20261002.embedding import (
    ItemMultisetEmbedding,
    ItemVocabulary,
    PAD_INDEX,
    UNK_INDEX,
    attach_item_embedding_to_nodes,
    fit_item_vocabulary,
)


def _empty_offsets(batch=1):
    return torch.zeros((batch, 11), dtype=torch.long)


def test_fit_vocab_is_sorted_fit_scoped_and_reload_stable(tmp_path):
    vocabulary = fit_item_vocabulary(
        {"train_b": [1002, 1001], "train_a": [1003, 1001]},
        provenance={"train_a": "FIT", "train_b": "REFIT_TRAIN"},
    )
    assert vocabulary.item_ids == (1001, 1002, 1003)
    assert vocabulary.item_id_to_index == {1001: 2, 1002: 3, 1003: 4}
    assert vocabulary.transform([1001, 9999]).tolist() == [2, UNK_INDEX]
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    restored = ItemVocabulary.load(path)
    assert restored.to_dict() == vocabulary.to_dict()
    assert restored.fingerprint == vocabulary.fingerprint
    with pytest.raises(ValueError, match="TEST"):
        fit_item_vocabulary([1001], scope_keys=["test"], provenance={"test": "TEST"})
    with pytest.raises(ValueError, match="excluded"):
        fit_item_vocabulary([1001], scope_keys=["train"], provenance="FIT", excluded_scope_keys=["train"])
    for label in ("V_DEV", "V_CAL", "STOP", "Q_CAL", "TRAIN", ""):
        with pytest.raises(ValueError, match="FIT or REFIT_TRAIN"):
            fit_item_vocabulary([1001], scope_keys=["train"], provenance=label)


def test_unknown_ledger_is_distinct_from_known_empty_and_oov_item():
    vocabulary = ItemVocabulary.fit([1001], scope_keys=["train"], provenance="FIT")
    encoder = ItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[UNK_INDEX].fill_(3.0)
        encoder.unknown_ledger_embedding.fill_(7.0)
    empty = encoder(torch.empty(0, dtype=torch.long), torch.empty(0), _empty_offsets(), torch.ones(1, 10, dtype=torch.bool))
    unknown = encoder(torch.empty(0, dtype=torch.long), torch.empty(0), _empty_offsets(), torch.zeros(1, 10, dtype=torch.bool))
    oov_indices = torch.tensor([UNK_INDEX], dtype=torch.long)
    oov_offsets = torch.tensor([[0, 1] + [1] * 9], dtype=torch.long)
    oov = encoder(oov_indices, torch.tensor([1]), oov_offsets, torch.ones(1, 10, dtype=torch.bool))
    assert torch.equal(empty[0, 0, :2], torch.zeros(2))
    assert torch.equal(empty[0, 0, 2:], torch.tensor([0.0, 1.0]))
    assert torch.equal(unknown[0, 0, :2], torch.tensor([7.0, 7.0]))
    assert torch.equal(unknown[0, 0, 2:], torch.tensor([0.0, 0.0]))
    assert torch.equal(oov[0, 0, :2], torch.tensor([3.0, 3.0]))


def test_count_weighted_pooling_is_permutation_invariant_and_routes_same_slot():
    vocabulary = ItemVocabulary.fit([1001, 1002], scope_keys=["train"], provenance="FIT")
    encoder = ItemMultisetEmbedding(vocabulary, embedding_dim=2)
    with torch.no_grad():
        encoder.item_embeddings.weight.zero_()
        encoder.item_embeddings.weight[2] = torch.tensor([1.0, 0.0])
        encoder.item_embeddings.weight[3] = torch.tensor([0.0, 1.0])
    offsets = torch.tensor([[0, 2] + [2] * 9], dtype=torch.long)
    forward = encoder(torch.tensor([2, 3]), torch.tensor([2, 1]), offsets)
    permuted = encoder(torch.tensor([3, 2]), torch.tensor([1, 2]), offsets)
    assert torch.allclose(forward, permuted)
    assert torch.equal(forward[0, 0, :2], torch.tensor([2.0, 1.0]))
    assert torch.equal(forward[0, 0, 2:], torch.tensor([3.0, 1.0]))

    nodes = torch.zeros(1, 10, 4)
    attached = attach_item_embedding_to_nodes(nodes, encoder, torch.tensor([2, 3]), torch.tensor([2, 1]), offsets)
    assert attached.shape == (1, 10, 8)
    assert torch.equal(attached[0, 0, 4:6], torch.tensor([2.0, 1.0]))
    assert torch.equal(attached[0, 1, 4:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
