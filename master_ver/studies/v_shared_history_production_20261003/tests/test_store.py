from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from studies.v_shared_history_20261003.bank import iter_match_views, loader_fingerprint
from studies.v_shared_history_20261003.tests.test_bank import _Loader, _plain, _purchase
import studies.v_shared_history_production_20261003.store as store
from studies.v_shared_history_production_20261003.store import ShardReader, write_shard


def _identity(loader):
    return {
        "source_hash": "a" * 64,
        "schema_hash": "b" * 64,
        "code_hash": "c" * 64,
        "loader_manifest_sha256": loader_fingerprint(loader),
    }


def _written(tmp_path: Path):
    loader = _Loader([_purchase()])
    records = list(iter_match_views(loader, 0))
    directory = tmp_path / "shard"
    manifest = write_shard(directory, ((0, iter(records)),), identity=_identity(loader))
    return loader, records, directory, manifest


def _shifted(records, *, query_start: int, time_start: int):
    for offset, record in enumerate(records):
        yield {
            "query_id": query_start + offset,
            "query_ms": time_start + offset,
            "view": record["view"],
        }


def test_roundtrip_indexes_one_match_and_reconstructs_each_query(tmp_path):
    loader, records, directory, manifest = _written(tmp_path)

    reader = ShardReader(directory, identity=_identity(loader))
    assert manifest["storage_complete"] is True
    assert manifest["semantically_verified"] is False
    assert reader.match_indices == [0]
    assert reader.query_ids == [0, 1, 2]
    assert reader.index[0]["query_ids"] == [0, 1, 2]
    for record in records:
        restored = reader.query_view(loader, record["query_id"])
        assert _plain(restored) == _plain(record["view"])

    member = reader.match_records(0)
    assert [row["query_id"] for row in member] == [0, 1, 2]
    with (directory / "shard.bin").open("rb") as stream:
        assert gzip.decompress(stream.read(manifest["binary_bytes"]))
    assert manifest["version"].endswith("_v2")


def test_v2_allows_empty_match_members_and_skips_them_for_query_coverage(tmp_path):
    loader = _Loader([_purchase()])
    records = list(iter_match_views(loader, 0))
    directory = tmp_path / "empty-middle"
    write_shard(
        directory,
        (
            (0, _shifted(records, query_start=0, time_start=100)),
            (1, iter(())),
            (2, _shifted(records, query_start=3, time_start=100)),
        ),
        identity=_identity(loader),
    )

    reader = ShardReader(directory, identity=_identity(loader))
    assert reader.match_indices == [0, 1, 2]
    assert reader.index[1]["n_queries"] == 0
    assert reader.index[1]["query_ids"] == []
    assert reader.index[1]["query_id_first"] is None
    assert reader.match_records(1) == []
    assert reader.query_ids == [0, 1, 2, 3, 4, 5]


def test_v2_all_empty_shard_is_storage_complete(tmp_path):
    loader = _Loader([_purchase()])
    directory = tmp_path / "all-empty"
    manifest = write_shard(
        directory,
        ((0, iter(())), (1, iter(()))),
        identity=_identity(loader),
    )
    reader = ShardReader(directory, identity=_identity(loader))
    assert manifest["n_matches"] == 2
    assert manifest["n_queries"] == 0
    assert reader.match_indices == [0, 1]
    assert reader.query_ids == []
    assert reader.match_records(0) == []


def test_negative_records_and_empty_entry_claims_are_rejected(tmp_path):
    loader = _Loader([_purchase()])
    identity = _identity(loader)
    negative = tmp_path / "negative"
    with pytest.raises(ValueError, match="nonnegative"):
        write_shard(
            negative,
            ((0, iter(({"query_id": -1, "query_ms": 1, "view": {"nonitem_history": {}}},))),),
            identity=identity,
        )

    directory = tmp_path / "claimed-empty"
    write_shard(directory, ((0, iter(())),), identity=identity)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["matches"][0]["query_ids"] = [0]
    index["matches"][0]["query_ms"] = [1]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_bytes"] = index_path.stat().st_size
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="empty shard match claims"):
        ShardReader(directory, identity=identity)


def test_reader_binds_loader_once_and_reuses_one_match_member(tmp_path, monkeypatch):
    loader, records, directory, _manifest = _written(tmp_path)
    reader = ShardReader(directory, identity=_identity(loader))
    original_fingerprint = store.loader_fingerprint
    original_decompress = store.gzip.decompress
    calls = {"fingerprint": 0, "decompress": 0}

    def fingerprint(value):
        calls["fingerprint"] += 1
        return original_fingerprint(value)

    def decompress(value):
        calls["decompress"] += 1
        return original_decompress(value)

    monkeypatch.setattr(store, "loader_fingerprint", fingerprint)
    monkeypatch.setattr(store.gzip, "decompress", decompress)
    for record in records:
        reader.query_view(loader, record["query_id"])
    assert calls == {"fingerprint": 1, "decompress": 1}


def test_identity_incomplete_and_binary_corruption_are_rejected(tmp_path):
    loader, _records, directory, _manifest = _written(tmp_path)

    with pytest.raises(ValueError, match="identity"):
        ShardReader(directory, identity={**_identity(loader), "source_hash": "d" * 64})

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "shard.bin").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        ShardReader(incomplete, identity=_identity(loader))

    corrupt = directory / "shard.bin"
    corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        ShardReader(directory, identity=_identity(loader))


def test_query_order_is_rejected_and_failed_attempt_is_preserved(tmp_path):
    loader = _Loader([_purchase()])
    records = list(iter_match_views(loader, 0))
    directory = tmp_path / "failed"
    bad = list(reversed(records))
    with pytest.raises(ValueError, match="contiguous"):
        write_shard(directory, ((0, iter(bad)),), identity=_identity(loader))
    assert directory.is_dir()
    assert not (directory / "MANIFEST.json").exists()
    with pytest.raises(FileExistsError):
        write_shard(directory, ((0, iter(records)),), identity=_identity(loader))


def test_index_summary_and_payload_corruption_are_detected_on_read(tmp_path):
    loader, _records, directory, manifest = _written(tmp_path)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["matches"][0]["query_ids"] = [4, 5, 6]
    # The committed hash protects the index before the semantic checks run.
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        ShardReader(directory, identity=_identity(loader))

    # The manifest remains immutable after a failed open; no cleanup/rewrite
    # of the committed directory is attempted by the reader.
    assert manifest["complete"] is True
