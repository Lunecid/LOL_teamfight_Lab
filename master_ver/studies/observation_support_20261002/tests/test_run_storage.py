"""Failure-path tests for the direct runner's provenance and bounded storage."""
import json

import numpy as np
import pytest

from studies.observation_support_20261002 import extract_match as EX
from studies.observation_support_20261002 import io_source as IO
from studies.observation_support_20261002 import run_diagnostic as RUN
from studies.observation_support_20261002 import storage as STORE


def test_reload_catches_value_and_schema_corruption(tmp_path):
    expected = {"values": np.array([1.0, np.nan, 3.0]), "keys": np.array([1, 2, 3])}
    path = tmp_path / "data.npz"
    np.savez(path, **expected)
    sha = RUN.file_sha256(path)
    assert STORE.verify_archive(path, expected, sha, RUN.file_sha256, 1)["all_ok"]
    np.savez(path, values=np.array([1.0, np.nan, 4.0]), extra=np.array([1]))
    result = STORE.verify_archive(path, expected, RUN.file_sha256(path), RUN.file_sha256, 1)
    assert not result["all_ok"]
    assert result["missing_fields"] == ["keys"]
    assert result["extra_fields"] == ["extra"]
    assert result["fields"]["values"]["n_mismatches"] == 1


def test_run_local_storage_supports_vector_and_matrix_shapes(tmp_path):
    vector = STORE.allocate_array(tmp_path, "vector", 3, np.int64, -1)
    matrix = STORE.allocate_array(tmp_path, "matrix", (3, 2), np.float64, np.nan)
    assert isinstance(vector, np.memmap) and vector.shape == (3,)
    assert matrix.shape == (3, 2)
    assert np.array_equal(vector, np.array([-1, -1, -1]))
    assert np.isnan(matrix).all()


def test_equal_first_times_preserve_component_identity_and_ragged_evidence():
    records = [
        dict(match_local=7, component_id=13, f=764886, same_ms_first_n=1,
             first_kill_x=[10.], first_kill_y=[20.]),
        dict(match_local=7, component_id=12, f=764886, same_ms_first_n=2,
             first_kill_x=[30., 40.], first_kill_y=[50., 60.]),
    ]
    ordered = RUN._order_component_records(records, 7, np.array([12, 13]))
    table = {key: np.zeros(2, dtype=np.int64)
             for key in ["match_local", "component_id", "f", "same_ms_first_n"]}
    evidence = RUN._allocate_evidence(2, first_counts=np.array([2, 1]))
    RUN._fill_rows(table, evidence, 0, ordered)
    assert table["component_id"].tolist() == [12, 13]
    assert table["same_ms_first_n"].tolist() == [2, 1]
    assert evidence["first_loc_x"].tolist() == [30., 40., 10.]
    assert evidence["first_loc_y"].tolist() == [50., 60., 20.]


@pytest.mark.parametrize("keys", [[(7, 12), (7, 12)], [(7, 12)], [(7, 12), (8, 13)]])
def test_component_storage_rejects_duplicate_missing_and_wrong_match_keys(keys):
    rows = [dict(match_local=ml, component_id=cid) for ml, cid in keys]
    with pytest.raises(RuntimeError, match="component identity"):
        RUN._order_component_records(rows, 7, np.array([12, 13]))


@pytest.mark.parametrize("mutate", [False, True])
def test_worker_pins_sources_before_production(tmp_path, monkeypatch, mutate):
    monkeypatch.setattr(RUN.C, "CACHE_DIR", tmp_path)
    for suffix in (".npz", ".meta.json", ".events.json"):
        (tmp_path / f"fixture{suffix}").write_bytes(b"before")
    def production(*args, **kwargs):
        if mutate:
            (tmp_path / "fixture.npz").write_bytes(b"after")
        return {"components": [], "match": {}}
    def reference(**kwargs):
        hashes = {key: RUN.file_sha256(tmp_path / f"fixture{suffix}")
                  for key, suffix in (("npz", ".npz"), ("meta", ".meta.json"), ("events", ".events.json"))}
        return {"components": [], "source_hashes": hashes}
    monkeypatch.setattr(EX, "process_match", production)
    monkeypatch.setattr(RUN.REF, "replay_match_components", reference)
    payload = {key: [] for key in ("t", "x", "y", "killer", "victim", "assist", "labels", "blue_pids")}
    payload.update(match_local=0, match_id="fixture")
    result = RUN._worker(payload)
    assert result["ok"] is (not mutate)
    assert result["input_versions"]["ok"] is (not mutate)
    if mutate:
        assert result["input_versions"]["before"] != result["input_versions"]["after_production"]


def test_startup_exception_keeps_failure_status(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("fixture startup failure")
    monkeypatch.setattr(RUN, "_run", fail)
    directory = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="fixture startup failure"):
        RUN.run(directory, max_matches=1)
    status = json.loads((directory / "STATUS.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and not status["complete"]
    assert "run_exception" in status["failed_gates"]
    with pytest.raises(RuntimeError, match="non-empty"):
        RUN.run(directory, max_matches=1)


@pytest.mark.parametrize("invalid", ["fractional_time", "short_node", "unequal_frames"])
def test_cache_schema_does_not_coerce_invalid_observations(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(RUN.C, "CACHE_DIR", tmp_path)
    arrays = {"minute_ts": np.array([0, 60000]), "xy_raw_minute": np.ones((2, 10, 2)),
              "node_minute": np.ones((2, 10, 76)), "gold_team_minute": np.ones((2, 2))}
    if invalid == "fractional_time":
        arrays["minute_ts"] = np.array([0.0, 60000.5])
    elif invalid == "short_node":
        arrays["node_minute"] = np.ones((2, 10, 17))
    else:
        arrays["gold_team_minute"] = np.ones((1, 2))
    np.savez(tmp_path / "fixture.npz", **arrays)
    pack, reason = IO.load_cache_pack("fixture")
    assert pack is None and reason


def test_final_version_check_detects_change_after_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN.C, "CACHE_DIR", tmp_path)
    suffixes = {"npz": ".npz", "meta": ".meta.json", "events": ".events.json"}
    for suffix in suffixes.values():
        (tmp_path / f"fixture{suffix}").write_bytes(b"worker-pinned")
    pinned = {key: RUN.file_sha256(tmp_path / f"fixture{suffix}") for key, suffix in suffixes.items()}
    rows = [{"match_local": 0, "match_id": "fixture", "before": pinned}]
    assert RUN._verify_final_cache_versions(rows, 1)["all_ok"]
    (tmp_path / "fixture.meta.json").write_bytes(b"changed-after-worker")
    report = RUN._verify_final_cache_versions(rows, 1)
    assert not report["all_ok"] and report["n_mismatches"] == 1


def test_source_snapshot_excludes_generated_test_outputs(tmp_path, monkeypatch):
    study = tmp_path / "study"
    (study / "tests").mkdir(parents=True)
    (study / "main.py").write_text("pass", encoding="utf-8")
    (study / "tests" / "fixture.py").write_text("pass", encoding="utf-8")
    generated = study / "pytest_basetemp" / "code_snapshot"
    generated.mkdir(parents=True)
    (generated / "main.py").write_text("pass", encoding="utf-8")
    monkeypatch.setattr(RUN.C, "STUDY", study)
    assert {p.relative_to(study).as_posix() for p in RUN._source_files()} == {"main.py", "tests/fixture.py"}


def test_resource_preflight_rejects_insufficient_ram(tmp_path, monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=8 << 30, available=1 << 30))
    with pytest.raises(RuntimeError, match="insufficient execution resources"):
        STORE.resource_preflight(tmp_path, 1024)


def test_zero_kill_sentinel_preserves_exact_full_event_coverage():
    starts, ends = np.array([0, -1, 2]), np.array([2, -1, 3])
    keys = np.array([0, 0, 2])
    IO.validate_ledger_ranges(starts, ends, keys, 3, 3, 1)
    with pytest.raises(RuntimeError, match="match keys"):
        IO.validate_ledger_ranges(starts, ends, np.array([0, 0, 1]), 3, 3, 1)
    with pytest.raises(RuntimeError, match="partition"):
        IO.validate_ledger_ranges(np.array([0, -1, 1]), ends, keys, 3, 3, 1)
