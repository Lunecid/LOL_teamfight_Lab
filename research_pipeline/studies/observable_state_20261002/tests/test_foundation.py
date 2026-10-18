import json

import numpy as np
import pytest

from studies.observable_state_20261002.categorical import lookups, static_lookup_counts
from studies.observable_state_20261002.schema import frame_spec, make_field_manifest
from studies.observable_state_20261002.storage import close_shard, open_shard, write_shard


def test_held_field_registry_excludes_future_and_inferred_channels():
    names, indices, groups = frame_spec()
    assert len(names) == len(set(indices)) == 46
    assert [groups.count(g) for g in ("E", "R", "D")] == [6, 27, 13]
    assert not {"alive", "gps_norm", "champion_name_id", "time_norm"} & set(names)
    fields = make_field_manifest([])["ordered_fields"]
    power = next(row for row in fields if row["id"] == "R.last_frame_power_fraction")
    assert power["source_index"] == 15
    assert "zero retained" in power["missingness_or_presence_unknown"]


def test_canonical_lookup_and_unverified_zero_do_not_mutate_input():
    values = np.zeros((10, 14), dtype=np.int64)
    values[:, 0] = 1
    values[:, 1:3] = 4
    before = values.copy()
    counts = static_lookup_counts(values, np.zeros((2, 5), dtype=np.int64), "15.14")
    assert counts["champion_id"] == {"canonical_ID_known": 10}
    assert counts["primary_rune_1"] == {"nonpositive_cache_ID_presence_unknown": 10}
    values[:, -1] = 5001
    counts = static_lookup_counts(values, np.zeros((2, 5), dtype=np.int64), "15.14")
    assert counts["stat_perk_defense"] == {"recorded_ID_map_unavailable": 10}
    assert np.array_equal(before[:, :-1], values[:, :-1])
    assert all(len(lookups(p)["styles"]) == 5 for p in ("15.14", "15.15", "15.16"))


def test_numeric_bank_complete_reload_zero_nan_and_read_only(tmp_path):
    source = {"node_frame": np.array([[[0.0, np.nan, 3.0]]], dtype=np.float32),
              "query_ms": np.array([120000, 240001], dtype=np.int64)}
    directory = tmp_path / "shard"
    manifest = write_shard(directory, source)
    assert manifest["complete"] and manifest["reload_mismatches"] == 0
    saved = open_shard(directory)
    try:
        assert np.array_equal(saved["node_frame"], source["node_frame"], equal_nan=True)
        assert saved["query_ms"].dtype == np.int64
        with pytest.raises(ValueError):
            saved["query_ms"][0] = 0
    finally:
        close_shard(saved)
    with pytest.raises(FileExistsError):
        write_shard(directory, source)


def test_numeric_bank_corruption_fails_hash_gate(tmp_path):
    directory = tmp_path / "shard"
    write_shard(directory, {"values": np.array([1.0, 2.0], dtype=np.float32)})
    path = directory / "values.npy"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash changed"):
        open_shard(directory)


def test_numeric_bank_rejects_objects_and_records_failure(tmp_path):
    directory = tmp_path / "shard"
    with pytest.raises(ValueError, match="Python objects"):
        write_shard(directory, {"values": np.array([{}], dtype=object)})
    manifest = json.loads((directory / "SHARD_MANIFEST.json").read_text())
    assert manifest["complete"] is False
    assert "error" in manifest


def test_parallel_writes_share_reserve_check_and_do_not_overcommit(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from studies.observable_state_20261002 import storage

    monkeypatch.setattr(storage, "MIN_FREE_RESERVE", 100)
    def remaining(path):
        used = sum(p.stat().st_size for p in tmp_path.glob("*/BANK.npz"))
        return SimpleNamespace(free=4317 - used)
    monkeypatch.setattr(storage.shutil, "disk_usage", remaining)
    def attempt(index):
        try:
            write_shard(tmp_path / str(index), {"values": np.zeros(100, dtype=np.int8)}, compressed=True)
            return True
        except OSError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, range(2))) == 1
