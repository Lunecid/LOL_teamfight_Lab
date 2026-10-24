"""Run-local typed storage and bounded reload verification."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def allocate_array(directory: Path | None, key: str, shape, dtype, fill=0) -> np.ndarray:
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    if directory is None:
        return np.full(shape, fill, dtype=dtype)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npy"
    if path.exists():
        raise RuntimeError(f"refuse existing backing array: {path}")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = fill
    return arr


def flush_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for arr in arrays.values():
        if isinstance(arr, np.memmap):
            arr.flush()


def resource_preflight(directory: Path, projected_bytes: int) -> dict[str, Any]:
    """Reserve space for backing arrays, archives, logs and transient writes."""
    free = int(shutil.disk_usage(directory).free)
    required = int(projected_bytes * 2.5) + (1 << 30)
    result = {"projected_backing_bytes": projected_bytes, "required_free_bytes": required,
              "disk_free_bytes": free, "disk_ok": free >= required}
    try:
        import psutil
        result.update(ram_total_bytes=int(psutil.virtual_memory().total),
                      ram_available_bytes=int(psutil.virtual_memory().available))
    except ImportError:
        raise RuntimeError("RAM evidence unavailable for full resource preflight")
    # Allow all mapped pages resident, plus immutable baseline/ledger arrays,
    # three workers, one reloaded evidence column, aggregate sorts and margin.
    result["transient_and_source_ram_reserve_bytes"] = 8 << 30
    result["required_ram_bytes"] = projected_bytes + (8 << 30)
    result["ram_ok"] = result["ram_available_bytes"] >= result["required_ram_bytes"]
    result["ok"] = result["disk_ok"] and result["ram_ok"]
    if not result["ok"]:
        raise RuntimeError(f"insufficient execution resources: {result}")
    return result


def verify_archive(path: Path, expected: Mapping[str, np.ndarray], sha: str,
                   hash_file, chunk_rows: int = 32768) -> dict[str, Any]:
    """Reload one member at a time and compare every element to verified storage."""
    result: dict[str, Any] = {"sha256_match": hash_file(path) == sha, "fields": {},
                              "n_checks": 0, "n_mismatches": 0}
    with np.load(path, allow_pickle=False) as saved:
        result["schema_ok"] = set(saved.files) == set(expected)
        result["missing_fields"] = sorted(set(expected) - set(saved.files))
        result["extra_fields"] = sorted(set(saved.files) - set(expected))
        for key, original in expected.items():
            if key not in saved.files:
                continue
            got = saved[key]
            shape_ok = got.shape == original.shape
            dtype_ok = got.dtype == original.dtype
            checked = mismatches = 0
            if shape_ok and dtype_ok:
                for begin in range(0, len(original), chunk_rows):
                    a = np.asarray(original[begin:begin + chunk_rows])
                    b = got[begin:begin + chunk_rows]
                    equal = a == b
                    if np.issubdtype(a.dtype, np.inexact):
                        equal |= np.isnan(a) & np.isnan(b)
                    checked += int(equal.size)
                    mismatches += int(np.count_nonzero(~equal))
            else:
                mismatches += 1
            result["fields"][key] = {"shape_ok": shape_ok, "dtype_ok": dtype_ok,
                                     "n_checked": checked, "n_mismatches": mismatches}
            result["n_checks"] += checked
            result["n_mismatches"] += mismatches
            del got
    result["all_ok"] = bool(result["schema_ok"] and result["sha256_match"]
                            and result["n_mismatches"] == 0)
    return result
