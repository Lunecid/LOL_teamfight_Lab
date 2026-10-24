"""Independent checks for the bounded common semantic surface."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from studies.v_model_fit_input_20261003.common import (
    BAG_LAYOUT,
    BAG_NAMESPACES,
    CATEGORY_LAYOUT,
    CATEGORY_NAMESPACES,
    ITEM_GROUPS,
    NUMERIC_DIMS,
    fit_common,
    semantic_row,
    transform_rows,
)
from studies.v_model_fit_input_20261003.models import StructuredFeatureEncoder
from studies.v_model_fit_input_20261003.tests.test_common import fixture, scope


_LANES = (("node", 10), ("team", 2), ("global", 1))
_EXPECTED_NUMERIC_DIMS = {"node": 102, "team": 8, "global": 75}
_EXPECTED_NODE_CATEGORIES = (
    "champion_id", "summoner_spell_1_id", "summoner_spell_2_id",
    "primary_style_id", "sub_style_id", "primary_rune_1", "primary_rune_2",
    "primary_rune_3", "primary_rune_4", "sub_rune_1", "sub_rune_2",
    "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
)
_EXPECTED_TEAM_CATEGORIES = tuple(f"ban{i}" for i in range(5))
_EXPECTED_CATEGORIES = {"node": _EXPECTED_NODE_CATEGORIES, "team": _EXPECTED_TEAM_CATEGORIES, "global": ()}
_EXPECTED_ITEM_GROUPS = (
    "history:ITEM_DESTROYED:itemId", "history:ITEM_DESTROYED:beforeId", "history:ITEM_DESTROYED:afterId",
    "history:ITEM_PURCHASED:itemId", "history:ITEM_PURCHASED:beforeId", "history:ITEM_PURCHASED:afterId",
    "history:ITEM_SOLD:itemId", "history:ITEM_SOLD:beforeId", "history:ITEM_SOLD:afterId",
    "history:ITEM_UNDO:itemId", "history:ITEM_UNDO:beforeId", "history:ITEM_UNDO:afterId",
)
_EXPECTED_NODE_BAGS = ("current_item", *_EXPECTED_ITEM_GROUPS, "nonitem")
_EXPECTED_TEAM_BAGS = ("nonitem",)
_EXPECTED_GLOBAL_BAGS = (*_EXPECTED_ITEM_GROUPS, "nonitem")
_EXPECTED_BAGS = {"node": _EXPECTED_NODE_BAGS, "team": _EXPECTED_TEAM_BAGS, "global": _EXPECTED_GLOBAL_BAGS}
_EXPECTED_CATEGORY_NAMESPACES = {
    **{name: f"static:{name}" for name in _EXPECTED_NODE_CATEGORIES},
    **{name: "static:champion_id" for name in _EXPECTED_TEAM_CATEGORIES},
}
_EXPECTED_BAG_NAMESPACES = {
    **{name: "item" for name in ("current_item", *_EXPECTED_ITEM_GROUPS)},
    "nonitem": "nonitem",
}


def _probe_rows():
    """Make a FIT row and a held-out semantic probe from the shared fixture."""

    fit_view = fixture()
    probe_view = copy.deepcopy(fit_view)
    probe_view["features"]["static_ids"][0, 0] = 999

    items = probe_view["features"]["items"]
    items["item_ids"] = np.asarray([9001], dtype=np.int64)
    items["counts"] = np.asarray([3], dtype=np.int64)
    items["offsets"] = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    items["recorded_ledger_supported"] = np.asarray(
        [False, True, True, True, True, True, True, True, True, True], dtype=bool
    )
    return [semantic_row(fit_view), semantic_row(probe_view)]


def _fit_probe():
    rows = _probe_rows()
    fitted = fit_common(rows, [0], **scope(["a" * 64, "b" * 64]))
    return rows, fitted, transform_rows(rows, fitted)


def _vocab_index(vocabulary, raw_value):
    """Map one raw semantic token with the documented PAD/UNK reservation."""

    return vocabulary.mapping.get(int(raw_value), 1)


def _expected_bag(rows, fitted, lane, group):
    namespace = _EXPECTED_BAG_NAMESPACES[group]
    vocabulary = fitted.categorical_map[namespace]
    nodes = dict(_LANES)[lane]
    flat_indices, flat_counts, offsets, observed = [], [], [], []
    for row in rows:
        bounds = [len(flat_indices)]
        row_observed = []
        for bag in row["bags"][lane][group]:
            for raw_value, count in bag.items():
                flat_indices.append(_vocab_index(vocabulary, raw_value))
                flat_counts.append(float(count))
            row_observed.append(float(sum(bag.values())))
            bounds.append(len(flat_indices))
        assert len(bounds) == nodes + 1
        offsets.append(bounds)
        observed.append(row_observed)
    return (
        np.asarray(flat_indices, dtype=np.int64),
        np.asarray(flat_counts, dtype=np.float32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(observed, dtype=np.float32),
    )


def _expected_table(rows, fitted):
    """Rebuild the table columns without calling common.table_layout/CSR code."""

    numeric = {
        lane: np.asarray(
            fitted.numeric_map[lane].transform(
                np.stack([row["numeric"][lane] for row in rows])
            ),
            dtype=np.float32,
        ).reshape(len(rows), -1)
        for lane, _ in _LANES
    }
    width = sum(_EXPECTED_NUMERIC_DIMS[lane] * size for lane, size in _LANES) + 10 + 1
    for lane, size in _LANES:
        for group in _EXPECTED_CATEGORIES[lane]:
            width += size * (fitted.categorical_map[_EXPECTED_CATEGORY_NAMESPACES[group]].size + 2)
        for group in _EXPECTED_BAGS[lane]:
            width += size * (fitted.categorical_map[_EXPECTED_BAG_NAMESPACES[group]].size + 2) + size

    expected = np.zeros((len(rows), width), dtype=np.float32)
    cursor = 0
    for lane, size in _LANES:
        block = numeric[lane]
        block_width = _EXPECTED_NUMERIC_DIMS[lane] * size
        assert block.shape == (len(rows), block_width)
        expected[:, cursor : cursor + block_width] = block
        cursor += block_width

    teams = np.stack([row["team_ids"] for row in rows])
    expected[:, cursor : cursor + 10] = (teams == 200).astype(np.float32)
    cursor += 10
    expected[:, cursor] = [float(row["focal_is_blue"]) for row in rows]
    cursor += 1

    for lane, size in _LANES:
        for group in _EXPECTED_CATEGORIES[lane]:
            vocabulary = fitted.categorical_map[_EXPECTED_CATEGORY_NAMESPACES[group]]
            ncat = vocabulary.size + 2
            block_width = size * ncat
            for row_index, row in enumerate(rows):
                values = np.asarray(row["categories"][lane][group]).reshape(-1)
                assert values.shape == (size,)
                for slot, raw_value in enumerate(values):
                    expected[row_index, cursor + slot * ncat + _vocab_index(vocabulary, raw_value)] += 1.0
            cursor += block_width

        for group in _EXPECTED_BAGS[lane]:
            vocabulary = fitted.categorical_map[_EXPECTED_BAG_NAMESPACES[group]]
            ncat = vocabulary.size + 2
            block_width = size * ncat
            for row_index, row in enumerate(rows):
                bags = row["bags"][lane][group]
                assert len(bags) == size
                for slot, bag in enumerate(bags):
                    for raw_value, count in bag.items():
                        expected[row_index, cursor + slot * ncat + _vocab_index(vocabulary, raw_value)] += float(count)
            cursor += block_width
            for row_index, row in enumerate(rows):
                for slot, bag in enumerate(row["bags"][lane][group]):
                    expected[row_index, cursor + slot] = float(sum(bag.values()))
            cursor += size

    assert cursor == width
    return expected


def test_common_structured_and_table_surfaces_match_an_independent_rebuild():
    rows, fitted, batch = _fit_probe()

    assert NUMERIC_DIMS == _EXPECTED_NUMERIC_DIMS
    assert len(_EXPECTED_NODE_CATEGORIES) == 14
    assert len(ITEM_GROUPS) == 12
    assert tuple(ITEM_GROUPS) == _EXPECTED_ITEM_GROUPS
    assert CATEGORY_LAYOUT == _EXPECTED_CATEGORIES
    assert BAG_LAYOUT == _EXPECTED_BAGS
    assert CATEGORY_NAMESPACES == _EXPECTED_CATEGORY_NAMESPACES
    assert BAG_NAMESPACES == _EXPECTED_BAG_NAMESPACES
    assert batch.global_numeric.shape == (2, 76)

    champion_vocab = fitted.categorical_map["static:champion_id"]
    item_vocab = fitted.categorical_map["item"]
    assert champion_vocab.mapping[1] >= 2
    assert _vocab_index(champion_vocab, 999) == 1
    assert _vocab_index(item_vocab, 9001) == 1

    for lane, _ in _LANES:
        for group in _EXPECTED_CATEGORIES[lane]:
            vocabulary = fitted.categorical_map[_EXPECTED_CATEGORY_NAMESPACES[group]]
            expected = np.asarray(
                [
                    [_vocab_index(vocabulary, value) for value in row["categories"][lane][group]]
                    for row in rows
                ],
                dtype=np.int64,
            )
            actual = getattr(batch, f"{lane}_categories")[group]
            np.testing.assert_array_equal(actual, expected)

        for group in _EXPECTED_BAGS[lane]:
            expected_indices, expected_counts, expected_offsets, expected_observed = _expected_bag(
                rows, fitted, lane, group
            )
            actual = getattr(batch, f"{lane}_bags")[group]
            np.testing.assert_array_equal(actual.indices, expected_indices)
            np.testing.assert_allclose(actual.counts, expected_counts, rtol=0, atol=0)
            np.testing.assert_array_equal(actual.offsets, expected_offsets)
            np.testing.assert_allclose(actual.observed_count, expected_observed, rtol=0, atol=0)

    np.testing.assert_array_equal(
        batch.current_item_supported,
        np.asarray([[True] * 10, [False, True, True, True, True, True, True, True, True, True]], dtype=bool),
    )
    np.testing.assert_allclose(batch.node_bags["current_item"].observed_count[1, [0, 2]], 0.0, rtol=0, atol=0)
    assert batch.node_bags["current_item"].observed_count[1, 1] == 3.0

    actual = batch.table_csr.toarray()
    expected = _expected_table(rows, fitted)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_structured_encoder_accepts_all_declared_groups_and_rejects_extras():
    rows, fitted, transformed = _fit_probe()
    model_batch = replace(
        transformed,
        global_numeric=np.asarray(transformed.global_numeric)[:, :75],
        side_flag=np.asarray(transformed.global_numeric)[:, 75],
    )
    assert set(model_batch.node_categories) == set(_EXPECTED_CATEGORIES["node"])
    assert set(model_batch.team_categories) == set(_EXPECTED_CATEGORIES["team"])
    assert set(model_batch.global_categories) == set(_EXPECTED_CATEGORIES["global"])
    assert set(model_batch.node_bags) == set(_EXPECTED_BAGS["node"])
    assert set(model_batch.team_bags) == set(_EXPECTED_BAGS["team"])
    assert set(model_batch.global_bags) == set(_EXPECTED_BAGS["global"])

    encoder = StructuredFeatureEncoder(
        vocab_sizes={name: vocabulary.num_embeddings for name, vocabulary in fitted.categorical_map.items()},
        category_namespaces=_EXPECTED_CATEGORY_NAMESPACES,
        bag_namespaces=_EXPECTED_BAG_NAMESPACES,
        category_layout=_EXPECTED_CATEGORIES,
        bag_layout=_EXPECTED_BAGS,
        embedding_dim=4,
    )
    players, context, ids = encoder.encode_components(model_batch)
    assert players.shape == (len(rows), 10, encoder.player_feature_dim)
    assert context.shape == (len(rows), encoder.context_feature_dim)
    np.testing.assert_array_equal(ids.detach().cpu().numpy(), np.stack([row["team_ids"] for row in rows]))

    cases = (
        ("node_categories", {**model_batch.node_categories, "extra": np.zeros((2, 10), dtype=np.int64)}),
        ("team_categories", {**model_batch.team_categories, "extra": np.zeros((2, 2), dtype=np.int64)}),
        ("global_bags", {**model_batch.global_bags, "extra": model_batch.global_bags["nonitem"]}),
    )
    for field, value in cases:
        with pytest.raises(ValueError, match="declared layout"):
            encoder.encode_components(replace(model_batch, **{field: value}))
