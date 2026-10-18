"""Immutable numeric shards with complete payload reload and disk guards."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import zipfile
from pathlib import Path

import numpy as np

from .source import file_sha256

MIN_FREE_RESERVE = 2 << 30
_WRITE_LOCK = threading.RLock()


def configure_write_lock(lock) -> None:
    """Install the parent's cross-process lock before any worker writes."""
    global _WRITE_LOCK
    _WRITE_LOCK = lock


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    dtype = array.dtype.descr if array.dtype.fields else array.dtype.str
    digest.update(json.dumps({"shape": list(array.shape), "dtype": dtype}, sort_keys=True).encode())
    if array.ndim == 0:
        digest.update(array.tobytes())
    else:
        for begin in range(0, len(array), 16384):
            digest.update(np.ascontiguousarray(array[begin:begin + 16384]).tobytes())
    return digest.hexdigest()


def write_shard(directory: Path, arrays: dict[str, np.ndarray], *, compressed: bool = False) -> dict:
    with _WRITE_LOCK:
        return _write_shard(directory, arrays, compressed=compressed)


def _write_shard(directory: Path, arrays: dict[str, np.ndarray], *, compressed: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"complete": False, "arrays": {}, "logical_bytes": 0, "reload_mismatches": 0,
                "storage": "npz_deflate_level1" if compressed else "npy_mmap"}
    try:
        projected = sum(value.nbytes for value in arrays.values())
        free = shutil.disk_usage(directory).free
        overhead = (projected + 99) // 100 + 4096 * len(arrays)
        if free < projected + overhead + MIN_FREE_RESERVE:
            raise OSError("numeric shard would violate the 2GiB free-space reserve")
        for key, array in arrays.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError("unsafe numeric bank field name")
            if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                raise ValueError("numeric shards cannot store Python objects")
        if compressed:
            archive_path = directory / "BANK.npz"
            with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for key, array in arrays.items():
                    with archive.open(key + ".npy", "w", force_zip64=True) as handle:
                        np.lib.format.write_array(handle, array, allow_pickle=False)
            with np.load(archive_path, allow_pickle=False) as archive:
                for key, array in arrays.items():
                    digest = array_digest(array)
                    if array_digest(archive[key]) != digest:
                        manifest["reload_mismatches"] += 1
                        raise ValueError("complete compressed payload reload mismatch")
                    dtype = array.dtype.descr if array.dtype.fields else array.dtype.str
                    manifest["arrays"][key] = {"shape": list(array.shape), "dtype": dtype,
                                               "payload_sha256": digest}
                    manifest["logical_bytes"] += int(array.nbytes)
            manifest["archive_sha256"] = file_sha256(archive_path)
            manifest["archive_bytes"] = archive_path.stat().st_size
            manifest["complete"] = True
            return manifest
        for key, array in arrays.items():
            path = directory / (key + ".npy")
            before = array_digest(array)
            with path.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            reloaded = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if array_digest(reloaded) != before:
                    manifest["reload_mismatches"] += 1
                    raise ValueError("complete numeric payload reload mismatch")
            finally:
                reloaded._mmap.close()
            dtype = array.dtype.descr if array.dtype.fields else array.dtype.str
            manifest["arrays"][key] = {"shape": list(array.shape), "dtype": dtype,
                                      "bytes": path.stat().st_size, "sha256": file_sha256(path),
                                      "payload_sha256": before}
            manifest["logical_bytes"] += int(array.nbytes)
        manifest["complete"] = True
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (directory / "SHARD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def open_shard(directory: Path, *, verify_hashes: bool = True) -> dict[str, np.memmap]:
    manifest = json.loads((directory / "SHARD_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("reload_mismatches") != 0:
        raise ValueError("numeric shard was not completed and reloaded successfully")
    arrays = {}
    try:
        if manifest.get("storage") == "npz_deflate_level1":
            path = directory / "BANK.npz"
            if verify_hashes and file_sha256(path) != manifest["archive_sha256"]:
                raise ValueError("immutable compressed bank hash changed")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(manifest["arrays"]):
                    raise ValueError("compressed bank field coverage changed")
                for key, record in manifest["arrays"].items():
                    value = archive[key]
                    dtype = value.dtype.descr if value.dtype.fields else value.dtype.str
                    if list(value.shape) != record["shape"] or json.dumps(dtype) != json.dumps(record["dtype"]):
                        raise ValueError("compressed shard header differs from manifest")
                    value.setflags(write=False)
                    arrays[key] = value
            return arrays
        for key, record in manifest["arrays"].items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError("unsafe stored field name")
            path = directory / (key + ".npy")
            if verify_hashes and file_sha256(path) != record["sha256"]:
                raise ValueError("immutable numeric bank hash changed")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[key] = value
            dtype = value.dtype.descr if value.dtype.fields else value.dtype.str
            if list(value.shape) != record["shape"] or json.dumps(dtype) != json.dumps(record["dtype"]):
                raise ValueError("numeric shard header differs from manifest")
        return arrays
    except Exception:
        close_shard(arrays)
        raise


def close_shard(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        if getattr(array, "_mmap", None) is not None:
            array._mmap.close()
