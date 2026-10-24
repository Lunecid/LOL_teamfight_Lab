from __future__ import annotations

import numpy as np
import pytest

from studies.v_model_fit_input_20261003.common import fit_common, table_layout, transform_rows
from studies.v_model_fit_input_20261003.tests.test_common import fixture, scope
from studies.v_model_fit_input_20261003.common import semantic_row
from studies.v_production_20261004.cache import (
    CacheWriter,
    InputCache,
    independent_csr,
    estimate_cache_bytes,
    transform_batch,
)


def _probe():
    first = semantic_row(fixture())
    second = semantic_row(fixture())
    bundle = fit_common([first, second], [0], **scope(["a" * 64, "b" * 64]))
    return [first, second], bundle


def _metadata(n: int):
    return {
        "nominal_index": np.arange(10, 10 + n, dtype=np.int64),
        "match_index": np.arange(n, dtype=np.int32),
        "query_index": np.arange(100, 100 + n, dtype=np.int64),
        "query_ms": np.arange(1000, 1000 + n, dtype=np.int64),
        "train_ordinal": np.arange(n, dtype=np.int64),
        "game_key": np.asarray([f"{i + 1:064x}".encode("ascii") for i in range(n)], dtype="S64"),
        "role": np.asarray([b"TRAIN"] * n, dtype="S8"),
        "patch": np.asarray([b"15.14"] * n, dtype="S5"),
        "swapped": np.zeros(n, dtype=bool),
        "source_w": np.ones(n, dtype=bool),
        "label_w": np.ones(n, dtype=bool),
        "winning_team": np.full(n, 100, dtype=np.int16),
        "context_membership": np.full(n, 5, dtype=np.uint8),
    }


def test_independent_csr_matches_common_surface():
    rows, bundle = _probe()
    actual = transform_rows(rows, bundle).table_csr
    expected = independent_csr(rows, bundle)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0


def test_cache_roundtrip_keeps_typed_structured_and_sparse_inputs(tmp_path):
    rows, bundle = _probe()
    transformed = transform_batch(rows, _metadata(2), bundle, check_reference=True)
    _blocks, width = table_layout(bundle)
    root = tmp_path / "final_FIT_fit_eval"
    with CacheWriter(
        root,
        total_rows=2,
        table_width=width,
        plan={"context": "final", "scope": "FIT", "rows": 2},
        source_bindings={"fixture": {"sha256": "a" * 64}},
    ) as writer:
        writer.append(transformed)
        manifest = writer.finalize()
    assert manifest["complete"] is True
    with InputCache(root) as restored:
        batch = next(restored.iter_batches())
        np.testing.assert_array_equal(batch.metadata["nominal_index"], [10, 11])
        np.testing.assert_allclose(batch.structured.node_numeric, transformed.structured.node_numeric)
        np.testing.assert_array_equal(batch.structured.node_categories["champion_id"], transformed.structured.node_categories["champion_id"])
        assert (batch.table_csr != transformed.table_csr).nnz == 0
        assert batch.structured.global_bags["nonitem"].offsets.shape == transformed.structured.global_bags["nonitem"].offsets.shape


def test_cache_writer_rejects_out_of_order_rows(tmp_path):
    rows, bundle = _probe()
    _blocks, width = table_layout(bundle)
    first = transform_batch(rows[:1], {key: value[:1] for key, value in _metadata(2).items()}, bundle)
    second_meta = _metadata(1)
    second_meta["nominal_index"][:] = 0
    second = transform_batch(rows[:1], second_meta, bundle)
    with CacheWriter(
        tmp_path / "ordered",
        total_rows=2,
        table_width=width,
        plan={"rows": 2},
    ) as writer:
        writer.append(first)
        with pytest.raises(ValueError, match="globally ordered"):
            writer.append(second)


def test_cache_estimate_is_sparse_and_retains_headroom():
    estimate = estimate_cache_bytes(923_485, csr_nnz_per_row=4096)
    assert estimate["numeric_bytes"] < 5_000_000_000
    assert estimate["csr_bytes"] < 40_000_000_000
    assert estimate["required_free_bytes"] > estimate["estimated_bytes"]
