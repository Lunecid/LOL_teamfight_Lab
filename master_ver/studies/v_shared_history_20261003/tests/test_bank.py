from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from studies.item_node_observed_20261002.bank import observe_match
from studies.v_input_projection_20261002.schema import QUERY_BLOCK_SPECS, STATIC_BLOCK_SPECS
from studies.v_shared_history_20261003.bank import (
    MatchHistoryCache,
    iter_match_views,
    loader_fingerprint,
    save_match_cache,
)


SOURCE_HASH = "c" * 64


def _history_record(timestamp_ms: int, *, level: int = 2) -> dict:
    payload = {"participantId": 1, "level": level}
    return {
        "timestamp_ms": timestamp_ms,
        "event_type": "LEVEL_UP",
        "payload": payload,
        "field_presence": {name: True for name in payload},
        "field_validity": {name: True for name in payload},
    }


class _Match:
    metadata = {
        "canonical_match_key": "a" * 64,
        "patch": "15.14",
        "role": "TRAIN",
        "original_index": 0,
    }
    shard_index = 0
    local_index = 0

    def __init__(self, events, query_ms=(100, 200, 300)):
        self._events = tuple(events)
        self.query_ms = np.asarray(query_ms, dtype=np.int64)

    def item_events(self):
        return self._events


class _History:
    def __init__(self, records):
        self.records = tuple(records)

    def iter_prefix(self, cutoff):
        yield from (record for record in self.records if int(record["timestamp_ms"]) <= int(cutoff))


class _ItemLoader:
    def __init__(self, observed):
        self.observed = observed

    def history_prefix(self, match_index, query_index):
        assert int(match_index) == 0
        return self.observed.history_prefix(int(query_index))


class _Loader:
    def __init__(self, item_events, *, manifest_name="source-a", query_ms=(100, 200, 300)):
        self.manifest = {"format_version": "fake-source-v1", "source_identity": manifest_name}
        self.matches = (object(),)
        self.match_queries = np.asarray([(0, len(query_ms))], dtype=[("first_query", "<i8"), ("n_queries", "<i8")])
        self.queries = np.asarray(
            [(0, int(timestamp), index) for index, timestamp in enumerate(query_ms)],
            dtype=[("match_index", "<i8"), ("query_ms", "<i8"), ("local_query", "<i8")],
        )
        self._observed = observe_match(_Match(item_events, query_ms=query_ms), context_memberships=(1,) * 6)
        self._history_records = (_history_record(50),)

    def _history(self, match_index):
        assert int(match_index) == 0
        return _History(self._history_records)

    def _item_loader(self):
        return _ItemLoader(self._observed)

    def query_input(self, query_id, *, include_histories):
        assert include_histories is False
        query_id = int(query_id)
        state = self._observed.states[query_id]
        features = {
            name: np.zeros(shape, dtype=np.dtype(dtype))
            for name, (dtype, shape) in (*QUERY_BLOCK_SPECS.items(), *STATIC_BLOCK_SPECS.items())
        }
        features["team_ids"] = np.asarray([100] * 5 + [200] * 5, dtype=np.int64)
        features["items"] = {
            "item_ids": np.asarray(state.item_ids, dtype=np.int64).copy(),
            "counts": np.asarray(state.counts, dtype=np.int64).copy(),
            "offsets": np.asarray(state.offsets, dtype=np.int64).copy(),
            "recorded_ledger_supported": np.asarray(state.recorded_ledger_supported, dtype=bool).copy(),
            "reason_mask": np.asarray(state.reason_mask, dtype=np.uint64).copy(),
            "unassigned_observed": bool(state.unassigned_observed),
            "whole_inventory_verified": False,
        }
        return {
            "features": features,
            "audit": {"match_index": 0, "query_ms": int(self.queries[query_id]["query_ms"])},
        }


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _purchase(timestamp=100, *, identity="purchase", item_id=101, participant_id=1):
    return {
        "timestamp_ms": timestamp,
        "source_ordinal": timestamp,
        "source_identity": identity,
        "participantId": participant_id,
        "itemId": item_id,
        "type": "ITEM_PURCHASED",
    }


def _cache(tmp_path: Path, loader: _Loader):
    records = list(iter_match_views(loader, 0))
    directory = tmp_path / "cache"
    manifest = save_match_cache(
        directory,
        records,
        match_index=0,
        source_hash=SOURCE_HASH,
        loader_manifest_sha256=loader_fingerprint(loader),
    )
    return records, directory, manifest, MatchHistoryCache(directory, source_hash=SOURCE_HASH)


def test_incremental_views_equal_cached_reference_views_and_gzip_preserves_payload(tmp_path):
    loader = _Loader([_purchase()])
    records, directory, manifest, cache = _cache(tmp_path, loader)

    assert [record["query_id"] for record in records] == [0, 1, 2]
    for record in records:
        restored = cache.query_view(loader, record["query_id"])
        assert _plain(restored) == _plain(record["view"])

    assert manifest["source_hash"] == SOURCE_HASH
    assert manifest["loader_manifest_sha256"] == loader_fingerprint(loader)
    with gzip.open(directory / "history.jsonl.gz", "rt", encoding="utf-8") as stream:
        raw = stream.read()
    assert "LEVEL_UP" in raw
    assert "101" in raw
    assert '"query_id":0' in raw


@pytest.mark.parametrize(
    "future_events",
    [
        [_purchase(250, identity="future-underflow", item_id=999, participant_id=1)],
        [
            _purchase(250, identity="future-duplicate", item_id=102, participant_id=1),
            _purchase(250, identity="future-duplicate", item_id=103, participant_id=0),
        ],
    ],
)
def test_future_underflow_or_duplicate_does_not_change_earlier_queries(future_events):
    if future_events[0]["type"] == "ITEM_PURCHASED" and "underflow" in future_events[0]["source_identity"]:
        future_events = [
            {
                "timestamp_ms": 250,
                "source_ordinal": 250,
                "source_identity": "future-underflow",
                "participantId": 1,
                "beforeId": 999,
                "afterId": 1000,
                "type": "ITEM_UNDO",
            }
        ]
    baseline = _Loader([_purchase()])
    future = _Loader([_purchase(), *future_events])
    baseline_views = list(iter_match_views(baseline, 0))
    future_views = list(iter_match_views(future, 0))
    for index in (0, 1):
        assert _plain(future_views[index]["view"]) == _plain(baseline_views[index]["view"])


def test_corrupt_source_hash_query_ids_and_loader_manifest_are_rejected(tmp_path):
    loader = _Loader([_purchase()])
    records, directory, _manifest, cache = _cache(tmp_path, loader)

    with pytest.raises(ValueError, match="source"):
        MatchHistoryCache(directory, source_hash="d" * 64)

    other_loader = _Loader([_purchase()], manifest_name="source-b")
    with pytest.raises(ValueError, match="loader manifest differs"):
        cache.query_view(other_loader, 0)

    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["query_ids"] = [100, 101, 102]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="query identity"):
        MatchHistoryCache(directory, source_hash=SOURCE_HASH)

    corrupt_dir = tmp_path / "corrupt"
    save_match_cache(
        corrupt_dir,
        records,
        match_index=0,
        source_hash=SOURCE_HASH,
        loader_manifest_sha256=loader_fingerprint(loader),
    )
    path = corrupt_dir / "history.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        MatchHistoryCache(corrupt_dir, source_hash=SOURCE_HASH)


def test_cache_directory_is_exclusive(tmp_path):
    loader = _Loader([_purchase()])
    records = list(iter_match_views(loader, 0))
    directory = tmp_path / "cache"
    kwargs = {
        "match_index": 0,
        "source_hash": SOURCE_HASH,
        "loader_manifest_sha256": loader_fingerprint(loader),
    }
    save_match_cache(directory, records, **kwargs)
    with pytest.raises(FileExistsError):
        save_match_cache(directory, records, **kwargs)
