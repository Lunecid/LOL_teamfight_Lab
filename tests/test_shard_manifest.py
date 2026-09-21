"""The decomposition must refuse an incomplete or mixed shard directory."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sd", REPO / "scripts" / "run_scale_decomposition.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)


def _dir(tmp_path, n_files, manifest):
    for i in range(n_files):
        np.savez_compressed(tmp_path / f"shard_{i:03d}.npz", X=np.zeros((1, 2), np.float32))
    if manifest is not None:
        json.dump(manifest, open(tmp_path / "manifest.json", "w"))
    return tmp_path


def test_complete_manifest_passes(tmp_path):
    m = {"num_shards": 2, "complete": True, "shards": {"0": {"rc": 0}, "1": {"rc": 0}}}
    assert sd.verify_shard_manifest(_dir(tmp_path, 2, m))["num_shards"] == 2


def test_missing_shard_is_refused(tmp_path):
    m = {"num_shards": 3, "complete": True, "shards": {"0": {"rc": 0}, "1": {"rc": 0}, "2": {"rc": 0}}}
    with pytest.raises(SystemExit):
        sd.verify_shard_manifest(_dir(tmp_path, 2, m))


def test_failed_child_is_refused(tmp_path):
    m = {"num_shards": 2, "complete": False, "shards": {"0": {"rc": 0}, "1": {"rc": 1}}}
    with pytest.raises(SystemExit):
        sd.verify_shard_manifest(_dir(tmp_path, 2, m))


def test_no_manifest_is_refused_unless_allowed(tmp_path):
    d = _dir(tmp_path, 2, None)
    with pytest.raises(SystemExit):
        sd.verify_shard_manifest(d)
    assert sd.verify_shard_manifest(d, allow_partial=True) == {}
