"""Focused P2b adapter contract checks; no corpus training or label fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

torch.set_num_threads(1)

from studies.v_model_fit_input_20261003.models import (
    BagInput,
    FAMILY_CARDS,
    LightGBMAdapter,
    LogisticAdapter,
    PlainMLP,
    SharedSlotMLP,
    StructuredBatch,
    StructuredFeatureEncoder,
    TeamSetMLP,
    build_model,
    family_cards,
)


def _batch() -> tuple[StructuredBatch, dict[str, int], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    torch.manual_seed(7)
    batch_size = 3
    specs = {"static:champion_id": 8, "static:spell": 7, "item": 10, "nonitem": 6}
    category_layout = {"node": ("champion_id", "spell"), "team": ("ban0",), "global": ()}
    bag_layout = {
        "node": ("current_item", "history0"),
        "team": ("nonitem",),
        "global": ("history0", "nonitem"),
    }
    category_namespaces = {
        "champion_id": "static:champion_id",
        "spell": "static:spell",
        "ban0": "static:champion_id",
    }
    bag_namespaces = {"current_item": "item", "history0": "item", "nonitem": "nonitem"}
    node_numeric = torch.randn(batch_size, 10, 3)
    team_numeric = torch.randn(batch_size, 2, 2)
    global_numeric = torch.randn(batch_size, 4)
    player_team_ids = torch.tensor([[100, 200] * 5] * batch_size)
    node_categories = {
        "champion_id": torch.randint(2, 8, (batch_size, 10)),
        "spell": torch.randint(2, 7, (batch_size, 10)),
    }
    team_categories = {"ban0": torch.randint(2, 8, (batch_size, 2))}
    dense_node = torch.randint(2, 10, (batch_size, 10, 2))
    dense_node_counts = torch.ones(batch_size, 10, 2)
    dense_history = torch.randint(2, 10, (batch_size, 10, 1))
    dense_team = torch.randint(2, 6, (batch_size, 2, 1))
    dense_global_history = torch.randint(2, 10, (batch_size, 1, 1))
    dense_global_nonitem = torch.randint(2, 6, (batch_size, 1, 1))
    node_bags = {
        "current_item": BagInput(dense_node, dense_node_counts, observed_count=torch.full((batch_size, 10), 2.0)),
        "history0": BagInput(dense_history, torch.ones_like(dense_history, dtype=torch.float32)),
    }
    team_bags = {"nonitem": BagInput(dense_team, torch.ones_like(dense_team, dtype=torch.float32))}
    global_bags = {
        "history0": BagInput(dense_global_history, torch.ones_like(dense_global_history, dtype=torch.float32)),
        "nonitem": BagInput(dense_global_nonitem, torch.ones_like(dense_global_nonitem, dtype=torch.float32)),
    }
    batch = StructuredBatch(
        node_numeric=node_numeric,
        team_numeric=team_numeric,
        global_numeric=global_numeric,
        player_team_ids=player_team_ids,
        side_flag=torch.ones(batch_size),
        current_item_supported=torch.tensor([[True] * 10, [True] * 9 + [False], [True] * 10]),
        node_categories=node_categories,
        team_categories=team_categories,
        node_bags=node_bags,
        team_bags=team_bags,
        global_bags=global_bags,
        category_namespaces=category_namespaces,
        bag_namespaces=bag_namespaces,
    )
    return batch, specs, category_layout, bag_layout


def _structured_model(cls, *, card=None):
    batch, specs, category_layout, bag_layout = _batch()
    model = cls(
        node_numeric_dim=3,
        team_numeric_dim=2,
        global_numeric_dim=4,
        vocab_sizes=specs,
        category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces,
        category_layout=category_layout,
        bag_layout=bag_layout,
        card=card,
    )
    return batch, model


def test_each_family_has_the_declared_four_card_grid():
    assert set(FAMILY_CARDS) == {"logistic", "lightgbm", "plain_mlp", "shared_slot_mlp", "team_set_mlp"}
    assert all(len(family_cards(family)) == 4 for family in FAMILY_CARDS)
    assert family_cards("plain_mlp")[0]["hidden"] == (128, 64)
    assert family_cards("plain_mlp")[2]["hidden"] == (256, 128)


def test_structured_widths_preserve_each_group_even_when_namespaces_are_shared():
    batch, specs, category_layout, bag_layout = _batch()
    encoder = StructuredFeatureEncoder(
        node_numeric_dim=3,
        team_numeric_dim=2,
        global_numeric_dim=4,
        vocab_sizes=specs,
        category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces,
        category_layout=category_layout,
        bag_layout=bag_layout,
    )
    assert encoder.player_feature_dim == 3 + 2 * 16 + 2 * 17 + 1
    assert encoder.context_feature_dim == 2 * (2 + 16 + 17) + 4 + 1 + 2 * 17
    players, context, ids = encoder.encode_components(batch)
    assert players.shape == (3, 10, 70)
    assert context.shape == (3, 109)
    assert torch.equal(ids, batch.player_team_ids)


def test_structured_layout_and_routing_metadata_are_strict():
    batch, specs, category_layout, bag_layout = _batch()
    encoder = StructuredFeatureEncoder(
        node_numeric_dim=3,
        team_numeric_dim=2,
        global_numeric_dim=4,
        vocab_sizes=specs,
        category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces,
        category_layout=category_layout,
        bag_layout=bag_layout,
    )
    missing_category = replace(batch, node_categories={"champion_id": batch.node_categories["champion_id"]})
    with pytest.raises(ValueError, match="exactly match"):
        encoder.encode_components(missing_category)
    missing_support = replace(batch, current_item_supported=None)
    with pytest.raises(ValueError, match="current_item_supported"):
        encoder.encode_components(missing_support)
    fractional_ids = replace(batch, player_team_ids=batch.player_team_ids.to(torch.float32) + 0.5)
    with pytest.raises(ValueError, match="integer 100/200"):
        encoder.encode_components(fractional_ids)
    wrong_team_ids = batch.player_team_ids.clone()
    wrong_team_ids[:, 0] = 200
    wrong_team_count = replace(batch, player_team_ids=wrong_team_ids)
    with pytest.raises(ValueError, match="exactly five"):
        encoder.encode_components(wrong_team_count)


def test_current_unsupported_ledger_uses_dedicated_vector_and_zero_count():
    batch, specs, category_layout, bag_layout = _batch()
    encoder = StructuredFeatureEncoder(
        node_numeric_dim=3,
        team_numeric_dim=2,
        global_numeric_dim=4,
        vocab_sizes=specs,
        category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces,
        category_layout=category_layout,
        bag_layout=bag_layout,
    )
    with torch.no_grad():
        encoder.unsupported_current_item_embedding.fill_(9.0)
    players, _, _ = encoder.encode_components(batch)
    current_start = 3 + 2 * 16
    assert torch.equal(players[1, 9, current_start : current_start + 16], torch.full((16,), 9.0))
    assert players[1, 9, current_start + 16].item() == 0.0


def test_slot_and_team_set_models_have_expected_outputs_and_team_set_is_invariant():
    batch, slot = _structured_model(SharedSlotMLP, card=family_cards("shared_slot_mlp")[0])
    _, team_set = _structured_model(TeamSetMLP, card=family_cards("team_set_mlp")[0])
    slot.eval()
    team_set.eval()
    assert slot(batch).shape == (3, 1)
    assert team_set(batch).shape == (3, 1)
    perm = torch.tensor([6, 0, 9, 3, 2, 5, 4, 7, 8, 1])
    permuted = StructuredBatch(
        node_numeric=batch.node_numeric[:, perm],
        team_numeric=batch.team_numeric,
        global_numeric=batch.global_numeric,
        player_team_ids=batch.player_team_ids[:, perm],
        side_flag=batch.side_flag,
        current_item_supported=batch.current_item_supported[:, perm],
        node_categories={key: value[:, perm] for key, value in batch.node_categories.items()},
        team_categories=batch.team_categories,
        node_bags={
            key: BagInput(value.indices[:, perm], value.counts[:, perm], observed_count=value.observed_count[:, perm] if value.observed_count is not None else None)
            for key, value in batch.node_bags.items()
        },
        team_bags=batch.team_bags,
        global_bags=batch.global_bags,
        category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces,
    )
    assert torch.allclose(team_set(batch), team_set(permuted))
    assert not torch.allclose(slot(batch), slot(permuted))


def test_plain_mlp_requires_batch_dense_input_and_uses_declared_cards():
    plain = PlainMLP(input_dim=5, card=family_cards("plain_mlp")[2])
    assert plain(torch.zeros(4, 5)).shape == (4, 1)
    with pytest.raises(ValueError, match="dense_features"):
        plain(StructuredBatch(torch.zeros(4, 1, 1), torch.zeros(4, 2, 1), torch.zeros(4, 1)))


def test_table_adapters_are_sparse_input_only_and_keep_card_parameters():
    logistic = LogisticAdapter(card=family_cards("logistic")[0])
    lightgbm = LightGBMAdapter(card=family_cards("lightgbm")[0])
    assert logistic.C == 0.001
    assert logistic.max_iter == 4000
    assert logistic.ftol == 1e-12
    assert logistic.gtol == 1e-7
    assert lightgbm.n_estimators == 2000
    assert lightgbm.num_leaves == 31
    assert lightgbm.min_data_in_leaf == 200
    with pytest.raises(ValueError, match="table_csr"):
        logistic.fit(StructuredBatch(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1)), np.array([0]))


def test_logistic_adapter_matches_weighted_objective_and_gradient():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    x_zero = scipy_sparse.csr_matrix(np.zeros((4, 1), dtype=np.float64))
    y_zero = np.array([0.0, 1.0, 1.0, 0.0])
    weights_zero = np.array([1.0, 3.0, 2.0, 1.0])
    intercept_only = LogisticAdapter(C=0.5).fit(x_zero, y_zero, sample_weight=weights_zero)
    expected_probability = float(np.dot(weights_zero, y_zero) / weights_zero.sum())
    expected_intercept = np.log(expected_probability / (1.0 - expected_probability))
    assert np.allclose(intercept_only.coef_, 0.0, atol=1e-8)
    assert np.isclose(intercept_only.intercept_, expected_intercept, atol=1e-7)

    x = scipy_sparse.csr_matrix(np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64))
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.array([1.0, 2.0, 1.0, 2.0])
    C = 0.5
    fitted = LogisticAdapter(C=C).fit(x, y, sample_weight=weights)
    beta = fitted.coef_.reshape(-1)
    intercept = float(fitted.intercept_)
    logits = np.asarray(x @ beta).reshape(-1) + intercept
    probabilities = fitted._sigmoid(logits)
    total_weight = weights.sum()
    residual = weights * (probabilities - y) / total_weight
    gradient = np.concatenate((np.asarray(x.T @ residual).reshape(-1) + beta / (C * len(y)), [residual.sum()]))
    objective = float(np.dot(weights, np.logaddexp(0.0, logits) - y * logits) / total_weight)
    objective += float(np.dot(beta, beta) / (2.0 * C * len(y)))
    assert np.linalg.norm(gradient, ord=np.inf) < 2e-6
    assert np.isclose(fitted.convergence_["fun"], objective, atol=1e-10)


def test_ragged_bag_embedding_bag_matches_dense_sum_and_backpropagates():
    specs = {"item": 8}
    bank_model = StructuredFeatureEncoder(
        node_numeric_dim=1,
        team_numeric_dim=1,
        global_numeric_dim=1,
        vocab_sizes=specs,
        category_layout={"node": (), "team": (), "global": ()},
        bag_layout={"node": ("history",), "team": (), "global": ()},
        bag_namespaces={"history": "item"},
    )
    embedding = bank_model.bank.table("item")
    indices = torch.tensor([2, 3, 4], dtype=torch.long)
    counts = torch.tensor([1.0, 2.0, 3.0])
    offsets = torch.tensor([[0, 1, 1], [1, 3, 3]], dtype=torch.long)
    encoded = bank_model.bag_encoder.encode("item", BagInput(indices, counts, offsets), batch=2, nodes=2)
    expected = torch.stack(
        (
            embedding.weight[2],
            torch.zeros_like(embedding.weight[2]),
            2.0 * embedding.weight[3] + 3.0 * embedding.weight[4],
            torch.zeros_like(embedding.weight[2]),
        )
    ).reshape(2, 2, -1)
    assert torch.allclose(encoded[:, :, :-1], expected)
    assert torch.allclose(encoded[:, :, -1], torch.tensor([[1.0, 0.0], [5.0, 0.0]]))
    encoded.sum().backward()
    assert embedding.weight.grad is not None
    assert torch.allclose(embedding.weight.grad[2], torch.ones_like(embedding.weight.grad[2]))
    assert torch.allclose(embedding.weight.grad[3], torch.full_like(embedding.weight.grad[3], 2.0))
    assert torch.allclose(embedding.weight.grad[4], torch.full_like(embedding.weight.grad[4], 3.0))


def test_build_model_constructs_all_families_without_fitting():
    _, specs, category_layout, bag_layout = _batch()
    kwargs = {
        "node_numeric_dim": 3,
        "team_numeric_dim": 2,
        "global_numeric_dim": 4,
        "vocab_sizes": specs,
        "category_namespaces": {"champion_id": "static:champion_id", "spell": "static:spell", "ban0": "static:champion_id"},
        "bag_namespaces": {"current_item": "item", "history0": "item", "nonitem": "nonitem"},
        "category_layout": category_layout,
        "bag_layout": bag_layout,
    }
    models = [build_model(family, card=0, **kwargs) for family in ("shared_slot_mlp", "team_set_mlp")]
    models.append(build_model("plain_mlp", card=0, input_dim=5))
    models.extend((build_model("logistic", card=0), build_model("lightgbm", card=0)))
    assert [model.family for model in models] == ["shared_slot_mlp", "team_set_mlp", "plain_mlp", "logistic", "lightgbm"]
