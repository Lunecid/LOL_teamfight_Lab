"""Read-only raw feature blocks with explicitly separated reference sidecars."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from studies.item_node_encoding_20261002.source import file_sha256
from studies.item_node_observed_20261002.bank import ObservedBankLoader
from studies.observable_state_20261002.bank import match_view

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "v_primary_raw_projection_v2_20261002"
QUERY_SPECS = {
    "node_frame": ("<f4", (10, 46)),
    "elapsed_minutes": ("<f8", ()),
    "frame_age_seconds": ("<f8", ()),
    "team_growth": ("<f8", (2, 6)),
    "team_growth_delta": ("<f8", (6,)),
    "cached_gold": ("<f4", (2,)),
    "team_kills": ("<i4", (2,)),
    "player_kda": ("<i4", (10, 3)),
    "objective_counts": ("<i4", (9,)),
    "objective_age_seconds": ("<f8", (9,)),
}
STATIC_SPECS = {"static_ids": ("<i8", (10, 14)), "bans": ("<i8", (2, 5)), "team_ids": ("<i8", (10,))}
QUERY_DTYPE = np.dtype([
    ("match_index", "<u4"), ("local_query", "<i4"), ("query_ms", "<i8"),
    ("frame_index", "<i4"), ("event_limit", "<i4"), ("item_query", "<i8"),
])
MATCH_QUERY_DTYPE = np.dtype([("first_query", "<i8"), ("n_queries", "<i4")])
BASE_FIELDS = frozenset((
    "node_frame", "minute_ts", "gold_team", "static_ids", "bans", "team_ids",
    "query_ms", "query_frame_index", "query_event_limit", "query_item_limit", "query_team_kills",
    "query_kda", "query_objective_counts", "query_objective_last_ms", "v_nominal", "item_event_indices",
))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_fields(layout: dict) -> list[str]:
    return [name for name in layout["match_fields"]
            if name in BASE_FIELDS or (name.startswith("event_") and not name.startswith("event_position"))]


def load_source_shard(directory: Path) -> tuple[dict, dict]:
    layout = read_json(directory / "LAYOUT.json")
    fields = source_fields(layout)
    selected_layout = {"match_fields": fields,
                       "scalar_fields": [name for name in layout["scalar_fields"] if name in fields]}
    with np.load(directory / "BANK.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for field in fields for name in (field, "offset_" + field)}
    return arrays, selected_layout


def local_source(arrays: dict, layout: dict, local_index: int) -> dict:
    return match_view(arrays, layout, int(local_index))


class RawProjectionLoader:
    """Single-owner reader. Query features never include target or audit sidecars."""

    def __init__(self, root: Path, *, verify_hashes: bool = True):
        self.root = Path(root)
        self.manifest = read_json(self.root / "RUN_MANIFEST.json")
        self.metadata = read_json(self.root / "BANK_METADATA.json")
        if self.manifest.get("complete") is not True or self.metadata.get("format_version") != FORMAT:
            raise ValueError("raw projection is incomplete or has the wrong format")
        self.verify_hashes = bool(verify_hashes)
        if verify_hashes:
            saved = self.manifest.get("saved_files", {})
            required = {name.upper() + ".npy" for name in (*QUERY_SPECS, *STATIC_SPECS)}
            required |= {"MATCH_TABLE.npy", "QUERY_TABLE.npy", "MATCH_QUERIES.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy",
                         "FIELD_MANIFEST.json", "BANK_METADATA.json"}
            if not required <= set(saved):
                raise ValueError("raw projection is missing required file bindings")
            for name, record in saved.items():
                if file_sha256(self.root / name) != record["sha256"]:
                    raise ValueError(f"raw projection file hash differs: {name}")
        self.features = {name: np.load(self.root / (name.upper() + ".npy"), mmap_mode="r", allow_pickle=False)
                         for name in (*QUERY_SPECS, *STATIC_SPECS)}
        self.matches = np.load(self.root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.queries = np.load(self.root / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.match_queries = np.load(self.root / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
        self.nominal = np.load(self.root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        self.contexts = np.load(self.root / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
        for name, (dtype, tail) in QUERY_SPECS.items():
            if self.features[name].shape != (len(self.queries), *tail) or self.features[name].dtype != np.dtype(dtype):
                raise ValueError(f"wrong query block shape/dtype: {name}")
        for name, (dtype, tail) in STATIC_SPECS.items():
            if self.features[name].shape != (len(self.matches), *tail) or self.features[name].dtype != np.dtype(dtype):
                raise ValueError(f"wrong static block shape/dtype: {name}")
        self._items = None
        self._shard_cache: OrderedDict[int, tuple[dict, dict]] = OrderedDict()
        self._history_cache: OrderedDict[int, Any] = OrderedDict()
        self._verified_shards: set[int] = set()

    def _item_loader(self) -> ObservedBankLoader:
        if self._items is None:
            root = Path(self.metadata["item_root"])
            if self.verify_hashes and file_sha256(root / "RUN_MANIFEST.json") != self.metadata["item_manifest_sha256"]:
                raise ValueError("bound item bank manifest changed")
            self._items = ObservedBankLoader(root, verify_hashes=self.verify_hashes)
        return self._items

    def _history(self, match_index: int):
        from .projection import SemanticHistory
        if match_index in self._history_cache:
            self._history_cache.move_to_end(match_index)
            return self._history_cache[match_index]
        record = self.matches[match_index]
        shard = int(record["shard"])
        if shard not in self._shard_cache:
            binding = self.metadata["source_shards"][shard]
            directory = Path(self.metadata["source_root"]) / binding["directory"]
            if self.verify_hashes and shard not in self._verified_shards:
                for key, filename in (("archive_sha256", "BANK.npz"), ("layout_sha256", "LAYOUT.json"),
                                      ("shard_manifest_sha256", "SHARD_MANIFEST.json"),
                                      ("match_ledger_sha256", "MATCH_LEDGER.jsonl"),
                                      ("static_repair_sha256", "STATIC_REPAIR.json")):
                    if file_sha256(directory / filename) != binding[key]:
                        raise ValueError(f"bound source changed: {directory.name}/{filename}")
                self._verified_shards.add(shard)
            self._shard_cache[shard] = load_source_shard(directory)
            while len(self._shard_cache) > 2:
                self._shard_cache.popitem(last=False)
        self._shard_cache.move_to_end(shard)
        arrays, layout = self._shard_cache[shard]
        result = SemanticHistory(local_source(arrays, layout, int(record["local_match"])))
        self._history_cache[match_index] = result
        while len(self._history_cache) > 8:
            self._history_cache.popitem(last=False)
        return result

    def query_input(self, query_index: int, *, include_histories: bool = True) -> dict:
        if isinstance(query_index, bool) or not isinstance(query_index, (int, np.integer)):
            raise TypeError("query index must be an integer")
        index = int(query_index)
        if not 0 <= index < len(self.queries):
            raise IndexError("query index outside raw projection")
        ref = self.queries[index]
        match_index = int(ref["match_index"])
        feature = {name: self.features[name][index] for name in QUERY_SPECS}
        feature.update({name: self.features[name][match_index] for name in STATIC_SPECS})
        item = self._item_loader().query(int(ref["item_query"]))
        if item["query_ms"] != int(ref["query_ms"]) or item["query_event_limit"] != int(ref["event_limit"]):
            raise ValueError("item query timing differs from raw projection")
        feature["items"] = {name: item[name] for name in (
            "item_ids", "counts", "offsets", "recorded_ledger_supported", "reason_mask", "unassigned_observed",
        )}
        feature["items"]["whole_inventory_verified"] = False
        if include_histories:
            feature["history"] = self._history(match_index).prefix(int(ref["query_ms"]))
        return {"features": feature,
                "audit": {"query_index": index, "match_index": match_index,
                          "query_ms": int(ref["query_ms"]), "history_included": include_histories},
                "target": {"final_blue_win": bool(self.matches[match_index]["target_Blue_win"])}}

    def nominal_input(self, row_index: int, *, include_histories: bool = True) -> dict:
        if isinstance(row_index, bool) or not isinstance(row_index, (int, np.integer)):
            raise TypeError("nominal row index must be an integer")
        index = int(row_index)
        if not 0 <= index < len(self.nominal):
            raise IndexError("nominal row outside raw projection")
        row = self.nominal[index]
        if not bool(row["supported"]) or int(row["query_ref"]) < 0:
            raise ValueError("unsupported nominal row has no model input; preserve its exclusion sidecar")
        match_index = int(row["match_index"])
        query_index = int(self.match_queries[match_index]["first_query"]) + int(row["query_ref"])
        ref = self.queries[query_index]
        if int(ref["match_index"]) != match_index or int(ref["query_ms"]) != int(row["actual_ms"]):
            raise ValueError("nominal row is not bound to its original query")
        result = self.query_input(query_index, include_histories=include_histories)
        result["audit"]["nominal_row_index"] = index
        return result

    def close(self):
        if self._items is not None:
            self._items.close()
        self._history_cache.clear()
        self._shard_cache.clear()
        for value in (*self.features.values(), self.matches, self.queries, self.match_queries, self.nominal, self.contexts):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
