import copy

import numpy as np
import pytest

from studies.item_node_observed_20261002.ledger import _history_context, ItemTransaction
from studies.v_input_projection_20261002.schema import QUERY_BLOCK_SPECS, STATIC_BLOCK_SPECS
from studies.v_model_input_20261003.history import HistoryAccumulator
from studies.v_model_input_20261003.view import make_view, permute_players
from studies.v_preparation_20261002.partition import context_membership
from studies.v_model_fit_input_20261003.common import semantic_row, fit_common, transform_rows, ITEM_GROUPS


def fixture():
    features = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in (*QUERY_BLOCK_SPECS.items(), *STATIC_BLOCK_SPECS.items())}
    features["team_ids"] = np.array([100, 200] * 5)
    features["node_frame"][0, 0] = np.nan
    features["static_ids"][:, 0] = np.arange(10)
    features["items"] = {"item_ids": np.array([1001]), "counts": np.array([2]),
                         "offsets": np.array([0, 1] + [1] * 9),
                         "recorded_ledger_supported": np.ones(10, dtype=bool), "reason_mask": np.zeros(10, dtype=np.uint64),
                         "unassigned_observed": True, "whole_inventory_verified": False}
    empty = _history_context(())
    purchase = ItemTransaction(timestamp_ms=1, source_ordinal=0, source_identity="x", type="ITEM_PURCHASED", participant_id=1, item_id=1001, before_id=None, after_id=None)
    unknown = ItemTransaction(timestamp_ms=1, source_ordinal=1, source_identity="y", type="ITEM_PURCHASED", participant_id=0, item_id=3001, before_id=None, after_id=None)
    contexts = {"node": [_history_context((purchase, purchase))] + [copy.deepcopy(empty) for _ in range(9)],
                "unassigned": _history_context((unknown,))}
    accumulator = HistoryAccumulator()
    return make_view(features, accumulator.snapshot(), item_history=contexts)


def scope(keys):
    flags = [context_membership("15.14", key) for key in keys]
    return {"context": "final", "scope": "REFIT_TRAIN", "game_keys": keys, "roles": ["TRAIN"] * len(keys),
            "membership": {name: [row[name] for row in flags] for name in flags[0]}, "source_hash": "a" * 64}


def test_item_role_multiplicity_and_unknown_owner_are_separate():
    row = semantic_row(fixture())
    group = "history:ITEM_PURCHASED:itemId"
    assert row["bags"]["node"][group][0] == {1001: 2}
    assert row["bags"]["global"][group][0] == {3001: 1}
    assert all(3001 not in bag for bag in row["bags"]["node"][group])
    assert row["numeric"]["node"].shape == (10, 102)
    assert row["numeric"]["global"].shape == (75,)
    assert len(ITEM_GROUPS) == 12


def test_joint_permutation_and_shared_scaler_vocabulary():
    view = fixture()
    order = np.array([6, 0, 9, 3, 2, 5, 4, 7, 8, 1])
    rows = [semantic_row(view), semantic_row(permute_players(view, order))]
    fitted = fit_common(rows, [0], **scope(["a" * 64, "b" * 64]))
    batch = transform_rows(rows, fitted)
    np.testing.assert_allclose(batch.node_numeric[1], batch.node_numeric[0, order])
    np.testing.assert_equal(batch.node_categories["champion_id"][1], batch.node_categories["champion_id"][0, order])
    assert fitted.categorical_map["static:champion_id"].mapping[0] >= 2
    assert fitted.categorical_map["static:champion_id"].mapping[1] >= 2
    assert batch.table_csr.dtype == np.float32
    assert batch.global_numeric.shape == (2, 76)


def test_missing_item_history_and_unsupported_nonempty_current_fail():
    view = fixture()
    view["item_history"] = None
    with pytest.raises(ValueError, match="history is required"):
        semantic_row(view)
    view = fixture()
    view["features"]["items"]["recorded_ledger_supported"][0] = False
    with pytest.raises(ValueError, match="masked empty"):
        semantic_row(view)


def test_partition_metadata_cannot_authorize_heldout_fit():
    rows = [semantic_row(fixture())]
    supplied = scope(["a" * 64])
    supplied["roles"] = ["TEST"]
    with pytest.raises(ValueError, match="membership differs"):
        fit_common(rows, [0], **supplied)
    supplied = scope(["a" * 64])
    supplied["membership"] = {}
    with pytest.raises(ValueError, match="six protocol"):
        fit_common(rows, [0], **supplied)
