"""q graph models and FT-Transformer: graph contract, S0 constants, permutation invariance, width matching."""
import hashlib
import json

import numpy as np
import pytest
import torch

from studies.q_production_20261005 import features as qf
from studies.q_production_20261005 import qmodels as qm
from studies.v_model_fit_input_20261003.common import BAG_LAYOUT, BAG_NAMESPACES, CATEGORY_LAYOUT, CATEGORY_NAMESPACES

from tests.test_q_features import _row  # synthetic semantic rows


def _batch(tmp_path, n=6):
    rows = [_row(seed) for seed in range(n)]
    writer = qf.RawStoreWriter(tmp_path / "s", np.arange(n, dtype=np.int64))
    for i, row in enumerate(rows):
        writer.add(i, row)
    writer.close()
    store = qf.RawStore(tmp_path / "s")
    keys = [hashlib.sha256(f"g{i}".encode()).hexdigest() for i in range(n)]
    pre = qf.fit_scope(store, np.arange(n), "REFIT_TRAIN", keys)
    return qf.Transformer(store, pre)(np.arange(n)), pre


def _encoder_kwargs(pre):
    return {"node_numeric_dim": 102, "team_numeric_dim": 8, "global_numeric_dim": 76,
            "vocab_sizes": {name: vocab.num_embeddings for name, vocab in pre.categorical_map.items()},
            "category_namespaces": dict(CATEGORY_NAMESPACES), "bag_namespaces": dict(BAG_NAMESPACES),
            "category_layout": dict(CATEGORY_LAYOUT), "bag_layout": dict(BAG_LAYOUT)}


def _xy(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 15000, size=(n, 10, 2))


def test_graph_contract_and_s0_constants():
    teams = np.tile(np.asarray([100, 200] * 5), (3, 1))
    g = qm.graph_inputs(_xy(3), teams, 1500.0, spatial=True)
    assert g.edge.shape == (3, 10, 9, 3) and g.ally.sum(axis=2).tolist() == [[4] * 10] * 3
    assert (~g.ally).sum() == 3 * 50 and g.ally.sum() == 3 * 40
    i, k = 2, 5
    j = qm.SENDERS[i, k]
    np.testing.assert_allclose(g.edge[0, i, k, :2], (_xy(3)[0, j] - _xy(3)[0, i]) / 1500.0, rtol=1e-6)
    np.testing.assert_allclose(g.weight[0, i, k], np.exp(-(np.linalg.norm(_xy(3)[0, j] - _xy(3)[0, i]) / 1500.0) ** 2), rtol=1e-5)
    s0 = qm.graph_inputs(None, teams, None, spatial=False)
    assert not s0.edge.any() and (s0.weight == 1).all() and np.array_equal(s0.ally, g.ally)
    alpha = torch.softmax(torch.as_tensor(s0.sage_log_weight), dim=-1)
    torch.testing.assert_close(alpha, torch.full_like(alpha, 1 / 9))


@pytest.mark.parametrize("family", ["team_set_pair_readout", "relational_mpnn", "tied_message", "graphsage"])
def test_graph_models_are_invariant_to_within_team_permutation(tmp_path, family):
    torch.manual_seed(0)
    batch, pre = _batch(tmp_path)
    model = qm.QGraphModel(family, hidden_width=16, encoder_kwargs=_encoder_kwargs(pre), g0_width=64, tied_width=24).eval()
    xy = _xy(6, 1)
    out = model(batch, qm.graph_inputs(xy, batch.player_team_ids, 2000.0, spatial=True))
    assert out.shape == (6,) and torch.isfinite(out).all()
    order = np.asarray([2, 1, 0, 3, 4, 5, 6, 7, 8, 9])      # slots 0 and 2 are both Blue (100)
    assert batch.player_team_ids[0, 0] == batch.player_team_ids[0, 2]
    permuted = qf.StructuredBatch(
        node_numeric=batch.node_numeric[:, order], team_numeric=batch.team_numeric, global_numeric=batch.global_numeric,
        player_team_ids=batch.player_team_ids[:, order], current_item_supported=batch.current_item_supported[:, order],
        node_categories={k: v[:, order] for k, v in batch.node_categories.items()}, team_categories=batch.team_categories,
        global_categories=batch.global_categories, node_bags={k: _permute_bag(v, order, 10) for k, v in batch.node_bags.items()},
        team_bags=batch.team_bags, global_bags=batch.global_bags, category_namespaces=batch.category_namespaces,
        bag_namespaces=batch.bag_namespaces)
    out_p = model(permuted, qm.graph_inputs(xy[:, order], permuted.player_team_ids, 2000.0, spatial=True))
    torch.testing.assert_close(out_p, out, atol=1e-5, rtol=1e-5)


def _permute_bag(bag, order, slots):
    offsets = np.asarray(bag.offsets)
    indices, counts = np.asarray(bag.indices), np.asarray(bag.counts)
    new_idx, new_cnt, new_off, observed = [], [], [], []
    cursor = 0
    for r in range(len(offsets)):
        row_off = [cursor]
        for s in order:
            a, z = offsets[r, s], offsets[r, s + 1]
            new_idx.extend(indices[a:z]); new_cnt.extend(counts[a:z]); cursor += z - a
            row_off.append(cursor)
        new_off.append(row_off)
    return qf.BagInput(np.asarray(new_idx, np.int64), np.asarray(new_cnt, np.float32), np.asarray(new_off, np.int64),
                       np.asarray(bag.observed_count)[:, order])


def test_g0_width_matches_g1_parameters_within_five_percent(tmp_path):
    _, pre = _batch(tmp_path)
    for hidden in (64, 128):
        g1 = qm.parameter_count(qm.QGraphModel("relational_mpnn", hidden_width=hidden, encoder_kwargs=_encoder_kwargs(pre)))
        start = qm.g0_start_width(hidden)
        assert start == {64: 320, 128: 640}[hidden]
        width, gap = qm.matched_width(hidden, lambda k: qm.QGraphModel("team_set_pair_readout", hidden_width=hidden, encoder_kwargs=_encoder_kwargs(pre), g0_width=k), g1, start)
        assert gap <= 0.05 and width % 8 == 0


def test_ft_transformer_counts_one_token_per_field_and_runs(tmp_path):
    batch, pre = _batch(tmp_path, 4)
    sizes = {name: vocab.num_embeddings for name, vocab in pre.categorical_map.items()}
    numeric = np.concatenate((batch.node_numeric.reshape(4, -1), batch.team_numeric.reshape(4, -1), batch.global_numeric), axis=1)
    category_fields, category_values = [], []
    for lane, groups in (("node", batch.node_categories), ("team", batch.team_categories)):
        for group, values in groups.items():
            for slot in range(values.shape[1]):
                category_fields.append((CATEGORY_NAMESPACES[group], sizes[CATEGORY_NAMESPACES[group]]))
                category_values.append(values[:, slot])
    bag_fields, bag_values = [], []
    for lane, groups, slots in (("node", batch.node_bags, 10), ("team", batch.team_bags, 2), ("global", batch.global_bags, 1)):
        for group, bag in groups.items():
            namespace = BAG_NAMESPACES[group]
            bag_fields.extend([(namespace, sizes[namespace])] * slots)
            bag_values.extend(qm.ft_bag_inputs(bag, slots, torch.device("cpu")))
    model = qm.FTTransformer(numeric_fields=numeric.shape[1], category_fields=category_fields, bag_fields=bag_fields, d_token=8, layers=1, heads=2)
    assert model.token_count == 1 + numeric.shape[1] + len(category_fields) + len(bag_fields)
    out = model(torch.as_tensor(numeric, dtype=torch.float32), torch.as_tensor(np.stack(category_values, axis=1)), bag_values)
    assert out.shape == (4,) and torch.isfinite(out).all()
