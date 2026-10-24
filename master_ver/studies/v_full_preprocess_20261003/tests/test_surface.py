from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import studies.v_full_preprocess_20261003.surface as surface_module
from studies.v_model_fit_input_20261003.tests.test_common import fixture
from studies.v_preparation_20261002.partition import context_membership, inner_stop, train_fold


MATCH_DTYPE = np.dtype([
    ("key", "S64"), ("shard", "u2"), ("local_match", "u2"), ("original_index", "u4"),
    ("patch_role", "u1"), ("role", "u1"), ("fold", "i1"), ("stop", "?"),
    ("target_Blue_win", "?"), ("first_nominal_row", "i8"), ("n_nominal_rows", "i4"),
])
ROW_DTYPE = np.dtype([
    ("match_index", "u4"), ("shard", "u2"), ("local_match", "u2"), ("nominal_index", "u4"),
    ("bucket_id", "i4"), ("bucket_lo_ms", "i4"), ("bucket_hi_ms", "i4"),
    ("nominal_ms", "i4"), ("actual_ms", "i4"), ("query_ref", "i4"),
    ("frame_index", "i4"), ("event_limit", "i4"), ("supported", "?"),
    ("support_reason_mask", "u2"),
])


class _Loader:
    def __init__(self, *, match_keys=("a" * 64,), query_ms=(100, 200)):
        self.manifest = {"source": "synthetic"}
        self._view = fixture()
        n_matches = len(match_keys)
        self.matches = np.zeros(n_matches, dtype=MATCH_DTYPE)
        self.match_queries = np.zeros(n_matches, dtype=[("first_query", "i8"), ("n_queries", "i8")])
        self.queries = np.zeros(len(query_ms), dtype=[("match_index", "i8"), ("query_ms", "i8"), ("local_query", "i8")])
        self.nominal = np.zeros(len(query_ms), dtype=ROW_DTYPE)
        for index, key in enumerate(match_keys):
            self.matches[index] = (
                key.encode("ascii"), 0, index, index, 0, 0, train_fold(key), inner_stop(key),
                True, index, 1,
            )
            self.match_queries[index] = (index, 1)
            self.queries[index] = (index, int(query_ms[index]), 0)
            self.nominal[index] = (index, 0, index, 0, index, 0, 1, int(query_ms[index]), int(query_ms[index]), 0, 0, 0, True, 0)

    def query_input(self, query_index, *, include_histories):
        assert include_histories is False
        query_index = int(query_index)
        return {
            "features": copy.deepcopy(self._view["features"]),
            "audit": {"match_index": 0, "query_ms": int(self.queries[query_index]["query_ms"])},
        }

    def close(self):
        self.closed = True


class _Reader:
    instances = []
    records = {}

    def __init__(self, directory, *, identity):
        self.directory = directory
        self.identity = identity
        self.calls = []
        self.__class__.instances.append(self)

    def match_records(self, match_index):
        self.calls.append(int(match_index))
        return copy.deepcopy(self.records[int(match_index)])


def _surface(monkeypatch, loader, records, *, seed=20261001):
    _Reader.instances.clear()
    _Reader.records = records
    monkeypatch.setattr(surface_module, "ShardReader", _Reader)
    surface = surface_module.BankSurface.__new__(surface_module.BankSurface)
    surface._loader = loader
    surface._closed = False
    surface.seed = seed
    surface.plan = {"synthetic": True}
    surface._reader = None
    surface._reader_directory = None
    surface._receipt_by_match = {index: (Path("synthetic"), {}) for index in range(len(loader.matches))}
    return surface


def test_iter_nominal_rows_uses_one_shard_read_and_no_raw_history(monkeypatch):
    loader = _Loader(query_ms=(100,))
    view = loader._view
    loader.nominal[0]["nominal_index"] = 0
    records = {0: [{
        "query_id": 0,
        "query_ms": 100,
        "nonitem": copy.deepcopy(view["nonitem_history"]),
        "item": copy.deepcopy(view["item_history"]),
    }]}
    surface = _surface(monkeypatch, loader, records, seed=7)

    rows = list(surface.iter_nominal_rows())

    assert [index for index, _row, _meta in rows] == [0]
    assert rows[0][2]["query_index"] == 0
    assert rows[0][2]["swapped"] == surface_module.replacement_swap("a" * 64, 100, 7)
    assert "target" not in rows[0][1]
    assert _Reader.instances[0].calls == [0]


def test_iter_nominal_rows_rejects_missing_query_record(monkeypatch):
    loader = _Loader(query_ms=(100,))
    surface = _surface(monkeypatch, loader, {0: []})

    with pytest.raises(ValueError, match="shard query records differ|missing query"):
        list(surface.iter_nominal_rows())


def _write_projection(root: Path, keys: tuple[str, ...]):
    root.mkdir()
    matches = np.zeros(len(keys), dtype=MATCH_DTYPE)
    rows = np.zeros(len(keys), dtype=ROW_DTYPE)
    starts = np.zeros(len(keys), dtype=[("first_query", "i8"), ("n_queries", "i8")])
    queries = np.zeros(len(keys), dtype=[("match_index", "i8"), ("local_query", "i8"), ("query_ms", "i8"), ("frame_index", "i8"), ("event_limit", "i8"), ("item_query", "?")])
    contexts = np.zeros((6, len(keys)), dtype=np.uint8)
    for index, key in enumerate(keys):
        matches[index] = (key.encode("ascii"), 0, index, index, 0, 0, train_fold(key), inner_stop(key), True, index, 1)
        rows[index] = (index, 0, index, 0, index, 0, 1, 100 + index, 100 + index, 0, 0, 0, True, 0)
        starts[index] = (index, 1)
        queries[index] = (index, 0, 100 + index, 0, 0, False)
        for context_index, context in enumerate(surface_module.CONTEXTS):
            flags = context_membership("15.14", key, surface_module._scope_fold(context))
            contexts[context_index, index] = sum(surface_module.MEMBERSHIP_BITS[name] for name, enabled in flags.items() if enabled)
    for name, value in {
        "MATCH_TABLE.npy": matches,
        "V_ROWS.npy": rows,
        "CONTEXT_MEMBERSHIP.npy": contexts,
        "MATCH_QUERIES.npy": starts,
        "QUERY_TABLE.npy": queries,
    }.items():
        np.save(root / name, value)
    saved = {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in (
        "MATCH_TABLE.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy", "MATCH_QUERIES.npy", "QUERY_TABLE.npy")}
    (root / "RUN_MANIFEST.json").write_text(json.dumps({"complete": True, "saved_files": saved}), encoding="utf-8")
    return matches, rows, contexts


def test_build_scope_plan_derives_and_binds_all_twelve_scopes(tmp_path):
    projection = tmp_path / "projection"
    census_root = tmp_path / "census"
    census_root.mkdir()
    keys = ("a" * 64, "b" * 64)
    matches, rows, contexts = _write_projection(projection, keys)
    bindings = {
        name: hashlib.sha256((projection / name).read_bytes()).hexdigest()
        for name in surface_module._BINDING_FILES
    }
    census_scopes = []
    for context_index, context in enumerate(surface_module.CONTEXTS):
        for scope, bit in (("FIT", 1), ("REFIT_TRAIN", 4)):
            indices = np.flatnonzero((contexts[context_index] & bit) != 0).astype(np.int64)
            filename = f"{context}_{scope}_nominal_rows.npy"
            np.save(census_root / filename, indices)
            census_scopes.append({
                "context": context,
                "scope": scope,
                "rows": int(len(indices)),
                "rows_sha256": hashlib.sha256((census_root / filename).read_bytes()).hexdigest(),
            })
    census = {
        "status": "passed",
        "full_nominal_scope_census": True,
        "matches": len(matches),
        "nominal_rows": len(rows),
        "queries": len(rows),
        "upstream_bindings": bindings,
        "scopes": census_scopes,
    }
    (census_root / "SCOPE_CENSUS.json").write_text(json.dumps(census), encoding="utf-8")

    plan = surface_module.build_scope_plan(projection, census_root)

    assert plan["train_nominal_indices"].dtype == np.int64
    np.testing.assert_array_equal(plan["train_nominal_indices"], np.arange(2, dtype=np.int64))
    assert len(plan["scopes"]) == 12
    assert all(set(("context", "scope", "nominal_indices", "local_indices", "match_indices", "excluded_game_keys")) <= set(item) for item in plan["scopes"])
