"""Boundary checks independent of source archive size or learned models."""
from types import SimpleNamespace

import numpy as np
import pytest

from studies.v_input_projection_20261002.bank import RawProjectionLoader, QUERY_SPECS, STATIC_SPECS, QUERY_DTYPE


def loader_fixture():
    loader = RawProjectionLoader.__new__(RawProjectionLoader)
    loader.features = {name: np.zeros((2, *tail), dtype=dtype) for name, (dtype, tail) in QUERY_SPECS.items()}
    loader.features.update({name: np.zeros((1, *tail), dtype=dtype) for name, (dtype, tail) in STATIC_SPECS.items()})
    loader.queries = np.asarray([(0, 0, 100, 0, 1, 0), (0, 1, 200, 1, 2, 1)], dtype=QUERY_DTYPE)
    loader.matches = np.asarray([(True,)], dtype=[("target_Blue_win", "?")])
    loader.match_queries = np.asarray([(0, 2)], dtype=[("first_query", "i8"), ("n_queries", "i4")])
    loader.nominal = np.asarray([(0, True, 1, 200), (0, False, -1, -1)],
                                dtype=[("match_index", "i4"), ("supported", "?"), ("query_ref", "i4"), ("actual_ms", "i8")])

    def item_query(index):
        return {"query_ms": (100, 200)[index], "query_event_limit": index + 1,
                "item_ids": np.asarray([1001]), "counts": np.asarray([1]), "offsets": np.asarray([0, 1] + [1] * 9),
                "recorded_ledger_supported": np.ones(10, dtype=bool), "reason_mask": np.zeros(10, dtype="u8"),
                "unassigned_observed": True, "future_audit_marker": 999, "source_event_count": 9000}

    loader._item_loader = lambda: SimpleNamespace(query=item_query)
    loader._history = lambda _: SimpleNamespace(prefix=lambda time: ({"timestamp_ms": time, "event_type": "LEVEL_UP"},))
    return loader


def test_feature_allowlist_separates_target_and_audit_even_if_item_reader_adds_fields():
    row = loader_fixture().query_input(0)
    assert row["target"]["final_blue_win"] is True
    assert set(row["features"]) == set(QUERY_SPECS) | set(STATIC_SPECS) | {"items", "history"}
    assert "target_Blue_win" not in row["features"]
    assert "query_ms" not in row["features"]
    assert "future_audit_marker" not in row["features"]["items"]
    assert "source_event_count" not in row["features"]["items"]
    assert row["features"]["items"]["whole_inventory_verified"] is False


def test_nominal_exclusion_never_indexes_last_query_and_nominal_mapping_is_exact():
    loader = loader_fixture()
    with pytest.raises(ValueError, match="unsupported nominal"):
        loader.nominal_input(1)
    row = loader.nominal_input(0)
    assert row["audit"]["query_index"] == 1
    assert row["audit"]["query_ms"] == 200
    loader.nominal[0]["actual_ms"] = 199
    with pytest.raises(ValueError, match="original query"):
        loader.nominal_input(0)


@pytest.mark.parametrize("index", [-1, 2])
def test_negative_or_outside_queries_rejected(index):
    with pytest.raises(IndexError):
        loader_fixture().query_input(index)


def test_bool_query_rejected_and_debug_omission_of_history_is_explicit():
    loader = loader_fixture()
    with pytest.raises(TypeError):
        loader.query_input(True)
    row = loader.query_input(0, include_histories=False)
    assert row["audit"]["history_included"] is False
    assert "history" not in row["features"]


def test_item_cutoff_disagreement_rejected():
    loader = loader_fixture()
    loader.queries[0]["event_limit"] = 3
    with pytest.raises(ValueError, match="timing differs"):
        loader.query_input(0)
