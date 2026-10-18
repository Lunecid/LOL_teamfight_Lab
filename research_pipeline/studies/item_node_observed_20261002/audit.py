"""Streaming observed-item bank construction and binding audit.

This stage writes query-local observed node inputs and lazy typed histories.  It
does not fit vocabularies or models and it never marks the bank accepted by
itself; the reload/independent replay gate remains a separate check.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from studies.item_node_encoding_20261002.source import (
    AcceptedSourceReader,
    DEFAULT_PREPARATION_ROOT,
    DEFAULT_SOURCE_ROOT,
    SourceValidationError,
    file_sha256,
)
from studies.v_preparation_20261002.partition import context_membership

from .bank import (
    CONTRACT_PATH,
    CONTEXT_NAMES,
    DEFAULT_OUTPUT,
    EXPECTED_MATCHES,
    EXPECTED_NOMINAL_ROWS,
    EXPECTED_QUERIES,
    EXPECTED_SHARDS,
    FORMAT_VERSION,
    MATCH_DTYPE,
    NOMINAL_DTYPE,
    PARTICIPANTS,
    QUERY_DTYPE,
    REASON_BITS,
    REPORT,
    SCOPES,
    ObservedBankError,
    ObservedMatch,
    UNASSIGNED_REASON_BITS,
    file_digest,
    observe_match,
    ObservedBankLoader,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ACCEPTANCE_PATH = ROOT / "reports" / "observable_state_20261002" / "ACCEPTANCE.json"
PREPARATION_ACCEPTANCE_PATH = ROOT / "reports" / "v_preparation_20261002" / "ACCEPTANCE.json"
PREPARATION_CONTRACT_PATH = ROOT / "reports" / "v_preparation_20261002" / "CONTRACT.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedBankError(f"cannot read JSON: {path}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _decode_key(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii")
    return str(value)


def _set_numeric_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _set_process_affinity(cpu: int | None = None) -> list[int] | None:
    """Pin this process when the host exposes Windows process affinity."""
    try:
        import psutil
        process = psutil.Process()
        if cpu is not None:
            process.cpu_affinity([int(cpu)])
        return [int(value) for value in process.cpu_affinity()]
    except (ImportError, OSError, ValueError):
        return None


def _worker_initializer() -> None:
    _set_numeric_threads()
    # ProcessPool worker identities are one-based and stable for the pool.
    worker_number = int(getattr(__import__("multiprocessing").current_process(), "_identity", (1,))[0])
    _set_process_affinity((2, 4, 6)[(worker_number - 1) % 3])


def _worker_affinity() -> list[int] | None:
    try:
        import psutil
        return [int(value) for value in psutil.Process().cpu_affinity()]
    except (ImportError, OSError, ValueError):
        return None


def _code_bindings() -> dict[str, str]:
    relative_paths = (
        "studies/item_node_observed_20261002/bank.py",
        "studies/item_node_observed_20261002/audit.py",
        "studies/item_node_observed_20261002/__main__.py",
        "studies/item_node_observed_20261002/__init__.py",
        "studies/item_node_observed_20261002/ledger.py",
        "studies/item_node_observed_20261002/embedding.py",
        "studies/item_node_encoding_20261002/source.py",
        "studies/item_node_encoding_20261002/ledger.py",
        "studies/item_node_encoding_20261002/embedding.py",
        "studies/item_node_encoding_20261002/audit.py",
        "studies/item_node_encoding_20261002/__init__.py",
        "studies/observable_state_20261002/bank.py",
        "studies/v_preparation_20261002/partition.py",
        "reports/item_node_observed_20261002/CONTRACT.json",
        "reports/v_preparation_20261002/CONTRACT.json",
    )
    bindings: dict[str, str] = {}
    for name in relative_paths:
        path = ROOT / name
        if not path.is_file():
            raise ObservedBankError(f"required code binding is missing: {path}")
        bindings[name] = file_sha256(path)
    return bindings


def _accepted_source(source_root: Path) -> dict[str, Any]:
    accepted = _json(SOURCE_ACCEPTANCE_PATH)
    if accepted.get("phase_accepted") is not True:
        raise ObservedBankError("observable source phase is not accepted")
    accepted_run = Path(str(accepted.get("accepted_run", "")))
    if accepted_run.resolve() != Path(source_root).resolve():
        raise ObservedBankError("source root differs from accepted observable run")
    manifest_path = Path(source_root) / "RUN_MANIFEST.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != accepted.get("run_manifest_sha256"):
        raise ObservedBankError("accepted observable RUN_MANIFEST hash differs")
    manifest = _json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("accepted") is not True:
        raise ObservedBankError("observable source manifest is not complete and accepted")
    return {
        "acceptance_path": str(SOURCE_ACCEPTANCE_PATH),
        "acceptance_sha256": file_sha256(SOURCE_ACCEPTANCE_PATH),
        "run_manifest_sha256": file_sha256(manifest_path),
        "counts": manifest.get("counts", accepted.get("counts", {})),
    }


def _required_source_links(preparation_root: Path) -> list[dict[str, Any]]:
    path = Path(preparation_root) / "SOURCE_SHARD_LINKS.json"
    payload = _json(path)
    links = payload.get("shards")
    if not isinstance(links, list) or not links:
        raise ObservedBankError("V preparation SOURCE_SHARD_LINKS.json has no shard list")
    required = ("archive_sha256", "layout_sha256", "shard_manifest_sha256", "match_ledger_sha256", "static_repair_sha256")
    result: list[dict[str, Any]] = []
    for row in links:
        if not isinstance(row, Mapping) or not isinstance(row.get("directory"), str):
            raise ObservedBankError("V preparation source link has an invalid directory")
        for key in required:
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or not all(char in "0123456789abcdef" for char in value):
                raise ObservedBankError(f"source link {row['directory']} has no complete {key} hash")
        result.append(dict(row))
    if len({row["directory"] for row in result}) != len(result):
        raise ObservedBankError("source shard links contain duplicate directories")
    return sorted(result, key=lambda row: row["directory"])


def verify_source_bindings(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    preparation_root: Path = DEFAULT_PREPARATION_ROOT,
    *,
    verify_hashes: bool = True,
    strict_full: bool = True,
) -> dict[str, Any]:
    """Verify all five accepted files for every source shard."""
    source_root, preparation_root = Path(source_root), Path(preparation_root)
    source_acceptance = _accepted_source(source_root)
    links = _required_source_links(preparation_root)
    if strict_full and len(links) != EXPECTED_SHARDS:
        raise ObservedBankError(f"expected {EXPECTED_SHARDS} source shards, got {len(links)}")
    actual_dirs = {path.name for path in source_root.iterdir() if path.is_dir() and (path / "BANK.npz").is_file()}
    if actual_dirs != {row["directory"] for row in links}:
        raise ObservedBankError("source shard directory set differs from accepted links")
    filenames = {
        "archive_sha256": "BANK.npz", "layout_sha256": "LAYOUT.json",
        "shard_manifest_sha256": "SHARD_MANIFEST.json", "match_ledger_sha256": "MATCH_LEDGER.jsonl",
        "static_repair_sha256": "STATIC_REPAIR.json",
    }
    records: list[dict[str, Any]] = []
    for link in links:
        record = {"directory": link["directory"]}
        directory = source_root / link["directory"]
        for binding, filename in filenames.items():
            path = directory / filename
            if not path.is_file():
                raise ObservedBankError(f"accepted source shard is missing {path}")
            digest = file_sha256(path) if verify_hashes else link[binding]
            if verify_hashes and digest != link[binding]:
                raise ObservedBankError(f"accepted source shard hash differs: {path}")
            record[binding] = digest
        records.append(record)
    return {
        "source_root": str(source_root), "n_shards": len(records), "records": records,
        "records_sha256": _json_digest(records), "acceptance": source_acceptance,
    }


class PreparationIndex:
    """Memory-mapped V references plus exact six-context membership."""

    def __init__(self, root: Path, *, strict_full: bool = True) -> None:
        self.root = Path(root)
        self.match_table = np.load(self.root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.v_rows = np.load(self.root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        self.contexts = np.load(self.root / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
        if self.match_table.ndim != 1 or self.v_rows.ndim != 1 or self.contexts.shape != (6, len(self.match_table)):
            raise ObservedBankError("V preparation reference or context shapes are invalid")
        required_match = {"key", "shard", "local_match", "first_nominal_row", "n_nominal_rows", "patch_role"}
        required_rows = {"match_index", "shard", "local_match", "nominal_index", "nominal_ms", "actual_ms", "query_ref", "supported"}
        if not required_match.issubset(self.match_table.dtype.names or ()) or not required_rows.issubset(self.v_rows.dtype.names or ()):
            raise ObservedBankError("V preparation dtype is missing source-reference fields")
        self._by_source: dict[tuple[int, int], int] = {}
        self._by_key: dict[str, int] = {}
        for index, row in enumerate(self.match_table):
            source = (int(row["shard"]), int(row["local_match"]))
            key = _decode_key(row["key"])
            if source in self._by_source or key in self._by_key:
                raise ObservedBankError("V preparation match identity is not one-to-one")
            self._by_source[source] = index
            self._by_key[key] = index
        if strict_full and len(self.match_table) != EXPECTED_MATCHES or strict_full and len(self.v_rows) != EXPECTED_NOMINAL_ROWS:
            raise ObservedBankError("V preparation coverage differs from accepted full corpus")

    @staticmethod
    def _patch_from_role(value: int) -> str:
        return ("15.14", "15.15", "15.16")[int(value)]

    def rows_for(self, shard: int, local: int) -> tuple[int, np.ndarray, tuple[int, ...]]:
        source = (int(shard), int(local))
        if source not in self._by_source:
            raise ObservedBankError(f"source match {source} is absent from V preparation")
        index = self._by_source[source]
        row = self.match_table[index]
        first, size = int(row["first_nominal_row"]), int(row["n_nominal_rows"])
        if first < 0 or size < 0 or first + size > len(self.v_rows):
            raise ObservedBankError("V preparation nominal slice is outside V_ROWS")
        selected = self.v_rows[first:first + size]
        if len(selected) != size or np.any(selected["match_index"] != index):
            raise ObservedBankError("V preparation nominal ownership differs from MATCH_TABLE")
        if len(selected) and np.any(selected["nominal_index"] != np.arange(size, dtype=selected["nominal_index"].dtype)):
            raise ObservedBankError("V preparation nominal indices are not complete")
        return index, selected, tuple(int(value) for value in np.asarray(self.contexts[:, index], dtype=np.uint8).tolist())

    def context_mask(self, index: int, context_index: int | None = None) -> int | tuple[int, ...]:
        values = np.asarray(self.contexts[:, int(index)], dtype=np.uint8)
        if np.any(values > 63):
            raise ObservedBankError("V preparation context membership contains an invalid mask")
        if context_index is None:
            return tuple(int(value) for value in values.tolist())
        return int(values[int(context_index)])

    def verify_context_membership(self) -> dict[str, Any]:
        checked = 0
        for index, row in enumerate(self.match_table):
            key = _decode_key(row["key"])
            patch = self._patch_from_role(int(row["patch_role"]))
            for context_index in range(6):
                expected_flags = context_membership(patch, key, excluded_fold=None if context_index == 0 else context_index - 1)
                expected = sum(1 << bit for bit, scope in enumerate(SCOPES) if expected_flags[scope])
                if int(self.contexts[context_index, index]) != expected:
                    raise ObservedBankError(f"V preparation context membership differs at match {index}, context {context_index}")
                checked += 1
        return {"exact_context_membership": True, "n_match_context_cells": checked}

    def metadata(self) -> dict[str, Any]:
        keys = [_decode_key(value) for value in self.match_table["key"]]
        return {
            "n_matches": len(self.match_table), "n_nominal_rows": len(self.v_rows),
            "match_key_sha256": hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest(),
            "context_membership_sha256": file_sha256(self.root / "CONTEXT_MEMBERSHIP.npy"),
            "source_identity_sha256": hashlib.sha256(self.match_table[["shard", "local_match"]].tobytes()).hexdigest(),
        }


def verify_preparation_bindings(
    preparation_root: Path = DEFAULT_PREPARATION_ROOT,
    *,
    strict_full: bool = True,
) -> dict[str, Any]:
    """Verify accepted V references, all saved hashes and exact memberships."""
    root = Path(preparation_root)
    accepted = _json(PREPARATION_ACCEPTANCE_PATH)
    if accepted.get("preparation_accepted") is not True or accepted.get("full_corpus_verified") is not True:
        raise ObservedBankError("V preparation is not accepted as a full reference source")
    if Path(str(accepted.get("accepted_run", ""))).resolve() != root.resolve():
        raise ObservedBankError("V preparation accepted_run differs from requested root")
    files = {
        "RUN_MANIFEST.json": root / "RUN_MANIFEST.json",
        "SOURCE_SHARD_LINKS.json": root / "SOURCE_SHARD_LINKS.json",
        "MATCH_TABLE.npy": root / "MATCH_TABLE.npy",
        "V_ROWS.npy": root / "V_ROWS.npy",
        "CONTEXT_MEMBERSHIP.npy": root / "CONTEXT_MEMBERSHIP.npy",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ObservedBankError("V preparation is missing: " + ", ".join(missing))
    manifest_path = files["RUN_MANIFEST.json"]
    if file_sha256(manifest_path) != accepted.get("manifest_sha256"):
        raise ObservedBankError("V preparation RUN_MANIFEST hash differs from acceptance")
    manifest = _json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("accepted_preparation") is not True:
        raise ObservedBankError("V preparation manifest is not complete and accepted")
    if strict_full and (manifest.get("counts", {}).get("n_matches") != EXPECTED_MATCHES or manifest.get("counts", {}).get("n_nominal_rows") != EXPECTED_NOMINAL_ROWS):
        raise ObservedBankError("V preparation manifest coverage differs from accepted full corpus")
    saved = manifest.get("saved_files", {})
    records: dict[str, Any] = {}
    for name, path in files.items():
        digest = file_sha256(path)
        expected = accepted.get("manifest_sha256") if name == "RUN_MANIFEST.json" else saved.get(name, {}).get("sha256")
        if not isinstance(expected, str) or digest != expected:
            raise ObservedBankError(f"V preparation saved hash differs or is missing: {path}")
        records[name] = {"sha256": digest, "bytes": path.stat().st_size}
    index = PreparationIndex(root, strict_full=strict_full)
    context_audit = index.verify_context_membership()
    return {
        "acceptance_path": str(PREPARATION_ACCEPTANCE_PATH), "acceptance_sha256": file_sha256(PREPARATION_ACCEPTANCE_PATH),
        "manifest_sha256": file_sha256(manifest_path), "files": records,
        "metadata": index.metadata(), **context_audit,
    }


def _source_task(match: Any, nominal_rows: Any, context_masks: Sequence[int]) -> ObservedMatch:
    return _compact_for_writer(observe_match(match, nominal_rows, context_memberships=context_masks))


@dataclass(frozen=True, slots=True)
class _EventSourceRef:
    """Small worker-to-writer event view; raw payload stays in source banks."""

    event_index: int
    timestamp_ms: int
    item_id: Any = None
    before_id: Any = None
    after_id: Any = None
    history_reason: str | None = None


def _compact_for_writer(observed: ObservedMatch) -> ObservedMatch:
    def compact(event: Any) -> _EventSourceRef:
        return _EventSourceRef(
            event_index=int(event.event_index), timestamp_ms=int(event.timestamp_ms),
            item_id=event.item_id, before_id=event.before_id, after_id=event.after_id,
            history_reason=event.history_reason,
        )
    return replace(
        observed,
        assigned_events=tuple(compact(event) for event in observed.assigned_events),
        unassigned_events=tuple(compact(event) for event in observed.unassigned_events),
        stats={**dict(observed.stats), "worker_cpu_affinity": _worker_affinity()},
    )


class _BankWriter:
    def __init__(self, output: Path, *, capacities: tuple[int, int, int] | None) -> None:
        self.output = Path(output)
        # ``run_bank`` creates the run directory before its initial manifest;
        # direct focused tests may hand the writer a new directory.
        self.output.mkdir(parents=True, exist_ok=True)
        self.full = capacities is not None
        self._ids = (self.output / "ITEM_IDS.bin").open("xb")
        self._counts = (self.output / "ITEM_COUNTS.bin").open("xb")
        # Histories are source references, not copied JSON prefixes.  The
        # accepted source banks remain the lossless typed history authority.
        self._assigned_sources = (self.output / "EVENT_SOURCE_INDEX_ASSIGNED.bin").open("xb")
        self._unassigned_sources = (self.output / "EVENT_SOURCE_INDEX_UNASSIGNED.bin").open("xb")
        self._match_log = (self.output / "MATCH_STATS.jsonl").open("x", encoding="utf-8")
        self._progress = (self.output / "PROGRESS.jsonl").open("x", encoding="utf-8")
        self._ids_cursor = 0
        self._history_counts = {"assigned": 0, "unassigned": 0}
        self._match_count = self._query_count = self._nominal_count = 0
        if self.full:
            n_matches, n_queries, n_nominal = capacities
            self.matches = np.lib.format.open_memmap(self.output / "MATCH_TABLE.npy", mode="w+", dtype=MATCH_DTYPE, shape=(n_matches,))
            self.queries = np.lib.format.open_memmap(self.output / "QUERY_TABLE.npy", mode="w+", dtype=QUERY_DTYPE, shape=(n_queries,))
            self.node_offsets = np.lib.format.open_memmap(self.output / "NODE_OFFSETS.npy", mode="w+", dtype="<i8", shape=(n_queries, PARTICIPANTS + 1))
            self.support = np.lib.format.open_memmap(self.output / "NODE_SUPPORT.npy", mode="w+", dtype="?", shape=(n_queries, PARTICIPANTS))
            self.reason_mask = np.lib.format.open_memmap(self.output / "NODE_REASON_MASK.npy", mode="w+", dtype="<u8", shape=(n_queries, PARTICIPANTS))
            self.nominal = np.lib.format.open_memmap(self.output / "NOMINAL_TABLE.npy", mode="w+", dtype=NOMINAL_DTYPE, shape=(n_nominal,))
            self.contexts = np.lib.format.open_memmap(self.output / "CONTEXT_MEMBERSHIP.npy", mode="w+", dtype="u1", shape=(6, n_matches))
        else:
            self.matches_list: list[Any] = []
            self.queries_list: list[Any] = []
            self.node_offsets_list: list[np.ndarray] = []
            self.support_list: list[np.ndarray] = []
            self.reason_mask_list: list[np.ndarray] = []
            self.nominal_list: list[Any] = []
            self.context_list: list[tuple[int, ...]] = []

    def _write_history(self, partition: str, match_index: int, events: Sequence[Any]) -> tuple[int, int]:
        handle = self._assigned_sources if partition == "assigned" else self._unassigned_sources
        first = self._history_counts[partition]
        source_indices = np.asarray([int(event.event_index) for event in events], dtype="<i4")
        if np.any((source_indices < 0) | (source_indices > np.iinfo(np.int32).max)):
            raise ObservedBankError("negative source event index cannot be persisted")
        source_indices.tofile(handle)
        self._history_counts[partition] += len(events)
        return first, len(events)

    @staticmethod
    def _nominal_value(rows: Any, name: str, index: int, default: Any) -> Any:
        if rows is not None and getattr(rows, "dtype", None) is not None and rows.dtype.names and name in rows.dtype.names:
            return rows[name][index]
        return default

    def write(self, match_index: int, observed: ObservedMatch, nominal_rows: Any) -> None:
        if len(observed.context_memberships) != 6:
            raise ObservedBankError("writer requires all six context memberships")
        query_start = self._query_count
        nominal_start = self._nominal_count
        assigned_start, assigned_count = self._write_history("assigned", match_index, observed.assigned_events)
        unassigned_start, unassigned_count = self._write_history("unassigned", match_index, observed.unassigned_events)
        for query_index, state in enumerate(observed.states):
            ids_start = self._ids_cursor
            np.asarray(state.item_ids, dtype="<i8").tofile(self._ids)
            np.asarray(state.counts, dtype="<i8").tofile(self._counts)
            self._ids_cursor += len(state.item_ids)
            shifted_offsets = np.asarray(state.offsets, dtype=np.int64) + ids_start
            query_row = np.asarray((match_index, query_index, int(state.query_timestamp_ms), int(state.query_event_limit), self._query_count,
                                    int(state.assigned_history_end), int(state.unassigned_history_end),
                                    bool(state.unassigned_observed), int(state.unassigned_reason_mask)), dtype=QUERY_DTYPE)
            if self.full:
                self.queries[self._query_count] = query_row
                self.node_offsets[self._query_count] = shifted_offsets
                self.support[self._query_count] = state.supported
                self.reason_mask[self._query_count] = state.reason_mask
            else:
                self.queries_list.append(query_row)
                self.node_offsets_list.append(shifted_offsets)
                self.support_list.append(np.asarray(state.supported, dtype=bool))
                self.reason_mask_list.append(np.asarray(state.reason_mask, dtype=np.uint64))
            self._query_count += 1
        rows = np.asarray(nominal_rows) if nominal_rows is not None else None
        if rows is not None and rows.dtype.names:
            for index in range(len(rows)):
                value = np.asarray((match_index, index, int(rows["query_ref"][index]), bool(rows["supported"][index]),
                                    int(rows["actual_ms"][index]), int(rows["nominal_ms"][index])), dtype=NOMINAL_DTYPE)
                if self.full:
                    self.nominal[self._nominal_count] = value
                else:
                    self.nominal_list.append(value)
                self._nominal_count += 1
        row = np.asarray((str(observed.metadata.get("canonical_match_key") or "").encode("ascii"),
                          int(observed.metadata.get("shard", 0)), int(observed.metadata.get("local_match", 0)),
                          int(observed.metadata.get("original_index") or 0),
                          str(observed.metadata.get("patch") or "").encode("ascii"),
                          str(observed.metadata.get("role") or "").encode("ascii"), int(observed.context_membership),
                          query_start, len(observed.states), nominal_start, len(rows) if rows is not None else 0,
                          assigned_start, assigned_count, unassigned_start, unassigned_count), dtype=MATCH_DTYPE)
        memberships = observed.context_memberships
        if self.full:
            self.matches[self._match_count] = row
            self.contexts[:, self._match_count] = np.asarray(memberships, dtype=np.uint8)
        else:
            self.matches_list.append(row)
            self.context_list.append(tuple(int(value) for value in memberships))
        self._match_count += 1
        self._match_log.write(json.dumps({"match_index": int(match_index), **dict(observed.stats), "metadata": dict(observed.metadata)}, sort_keys=True, ensure_ascii=False) + "\n")

    def write_progress(self, *, final: bool = False) -> None:
        output_bytes = sum(path.stat().st_size for path in self.output.iterdir() if path.is_file() and path.name != "PROGRESS.jsonl")
        free_bytes = shutil.disk_usage(self.output).free
        self._progress.write(json.dumps({
            "matches": int(self._match_count), "queries": int(self._query_count),
            "nominal_rows": int(self._nominal_count), "output_bytes": int(output_bytes),
            "free_bytes": int(free_bytes), "final": bool(final),
        }, sort_keys=True) + "\n")
        self._progress.flush()

    def finalize(self) -> dict[str, Any]:
        self.write_progress(final=True)
        for handle in (self._ids, self._counts, self._assigned_sources, self._unassigned_sources, self._match_log, self._progress):
            handle.flush()
            handle.close()
        if self.full:
            if self._match_count != len(self.matches) or self._query_count != len(self.queries) or self._nominal_count != len(self.nominal):
                raise ObservedBankError(f"bank capacity mismatch: matches={self._match_count}, queries={self._query_count}, nominal={self._nominal_count}")
            arrays = (self.matches, self.queries, self.node_offsets, self.support, self.reason_mask, self.nominal, self.contexts)
            for value in arrays:
                value.flush()
                value._mmap.close()
        else:
            np.save(self.output / "MATCH_TABLE.npy", np.asarray(self.matches_list, dtype=MATCH_DTYPE), allow_pickle=False)
            np.save(self.output / "QUERY_TABLE.npy", np.asarray(self.queries_list, dtype=QUERY_DTYPE), allow_pickle=False)
            np.save(self.output / "NODE_OFFSETS.npy", np.asarray(self.node_offsets_list, dtype="<i8").reshape((-1, PARTICIPANTS + 1)), allow_pickle=False)
            np.save(self.output / "NODE_SUPPORT.npy", np.asarray(self.support_list, dtype="?").reshape((-1, PARTICIPANTS)), allow_pickle=False)
            np.save(self.output / "NODE_REASON_MASK.npy", np.asarray(self.reason_mask_list, dtype="<u8").reshape((-1, PARTICIPANTS)), allow_pickle=False)
            np.save(self.output / "NOMINAL_TABLE.npy", np.asarray(self.nominal_list, dtype=NOMINAL_DTYPE), allow_pickle=False)
            context_array = np.asarray(self.context_list, dtype="u1").reshape((-1, 6)).T
            np.save(self.output / "CONTEXT_MEMBERSHIP.npy", context_array, allow_pickle=False)
        return {"n_matches": self._match_count, "n_unique_queries": self._query_count, "n_nominal_rows": self._nominal_count, "n_item_values": self._ids_cursor, **{f"n_{key}_history": value for key, value in self._history_counts.items()}}


def _merge_counter(target: Counter[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, np.integer)):
            target[str(key)] += int(value)


def _merge_stats(aggregate: dict[str, Any], observed: ObservedMatch) -> None:
    stats = observed.stats
    aggregate["n_item_events"] += int(stats.get("n_item_events", 0))
    aggregate["n_assigned_events"] += int(stats.get("n_assigned_events", 0))
    aggregate["n_unassigned_events"] += int(stats.get("n_unassigned_events", 0))
    aggregate["n_supported_player_queries"] += int(stats.get("n_supported_player_queries", 0))
    aggregate["n_unassigned_observed_queries"] += int(stats.get("n_unassigned_observed_queries", 0))
    aggregate["support_by_participant"] += np.asarray(stats.get("support_by_participant", [0] * PARTICIPANTS), dtype=np.int64)
    _merge_counter(aggregate["reason_counts"], stats.get("reason_counts", {}))
    _merge_counter(aggregate["unassigned_reason_counts"], stats.get("unassigned_reason_counts", {}))
    _merge_counter(aggregate["item_event_counts"], stats.get("item_event_counts", {}))
    _merge_counter(aggregate["item_id_event_counts"], stats.get("item_id_event_counts", {}))
    affinity = stats.get("worker_cpu_affinity")
    if affinity is not None:
        marker = ",".join(str(int(value)) for value in affinity)
        aggregate["worker_affinity_masks"][marker] += 1
    role = str(observed.metadata.get("role") or "UNKNOWN")
    patch = str(observed.metadata.get("patch") or "UNKNOWN")
    for label, key in (("role", role), ("patch", patch)):
        entry = aggregate[f"by_{label}"].setdefault(key, {"matches": 0, "queries": 0, "nominal_rows": 0, "supported_player_queries": 0, "unassigned_events": 0, "reason_counts": Counter()})
        entry["matches"] += 1
        entry["queries"] += len(observed.states)
        entry["nominal_rows"] += len(observed.nominal_query_refs)
        entry["supported_player_queries"] += int(stats.get("n_supported_player_queries", 0))
        entry["unassigned_events"] += int(stats.get("n_unassigned_events", 0))
        _merge_counter(entry["reason_counts"], stats.get("reason_counts", {}))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _saved_file_records(output: Path) -> dict[str, dict[str, Any]]:
    names = (
        "MATCH_TABLE.npy", "QUERY_TABLE.npy", "NODE_OFFSETS.npy", "NODE_SUPPORT.npy", "NODE_REASON_MASK.npy",
        "NOMINAL_TABLE.npy", "CONTEXT_MEMBERSHIP.npy", "ITEM_IDS.bin", "ITEM_COUNTS.bin",
        "EVENT_SOURCE_INDEX_ASSIGNED.bin", "EVENT_SOURCE_INDEX_UNASSIGNED.bin", "MATCH_STATS.jsonl", "PROGRESS.jsonl", "SUMMARY.json",
        "SOURCE_BINDINGS.json", "PREPARATION_BINDINGS.json", "VOCABULARY_SCOPE_IDS.json", "VOCABULARY_SCOPE_META.json",
        *(f"VOCABULARY_{context}_{scope}.json" for context in CONTEXT_NAMES for scope in ("FIT", "REFIT_TRAIN")),
        "BANK_METADATA.json",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = output / name
        if path.is_file():
            result[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return result


def _estimated_output_bytes() -> dict[str, Any]:
    """Conservative preflight estimate from accepted event/reference counts."""
    estimate_path = REPORT / "SOURCE_SIZE_ESTIMATE.json"
    payload = _json(estimate_path)
    source_events = payload.get("recorded_item_events")
    if not isinstance(source_events, int) or source_events < 0:
        raise ObservedBankError("SOURCE_SIZE_ESTIMATE.json has no valid recorded_item_events")
    fixed = (
        EXPECTED_MATCHES * MATCH_DTYPE.itemsize
        + EXPECTED_QUERIES * (QUERY_DTYPE.itemsize + (PARTICIPANTS + 1) * 8 + PARTICIPANTS + PARTICIPANTS * 8)
        + EXPECTED_NOMINAL_ROWS * NOMINAL_DTYPE.itemsize
        + EXPECTED_MATCHES * 6
    )
    # Four int64 arrays (IDs/counts) at a deliberately high 256 values/query
    # bound leave room for malformed/tainted endpoints while preserving C:8GiB.
    ragged_bound = EXPECTED_QUERIES * 256 * 16
    total = int(fixed + ragged_bound + source_events * 4 + (1 << 20))
    return {
        "path": str(estimate_path), "sha256": file_sha256(estimate_path),
        "recorded_item_events": int(source_events), "estimated_output_bytes": total,
        "estimated_output_GiB": total / float(1 << 30), "ragged_values_bound_per_query": 256,
    }


def _vocabulary_scope_update(
    scope_ids: dict[str, set[int]], observed: ObservedMatch,
    scope_audit: dict[str, dict[str, Any]], scope_hashers: dict[str, Any],
) -> None:
    if len(observed.context_memberships) != 6:
        raise ObservedBankError("vocabulary fitting requires all six context memberships")
    supported_refs = observed.nominal_query_refs[observed.nominal_supported] if observed.nominal_supported.size else np.empty(0, dtype=np.int64)
    cutoff_query_ref = int(np.max(supported_refs)) if len(supported_refs) else -1
    memberships = observed.context_memberships
    shard = int(observed.metadata.get("shard", -1))
    local_match = int(observed.metadata.get("local_match", -1))
    ids_cache: dict[int, set[int]] = {}

    def ids_for(query_ref: int) -> set[int]:
        cached = ids_cache.get(query_ref)
        if cached is not None:
            return cached
        state = observed.states[query_ref]
        values: set[int] = set(int(value) for value in state.item_ids.tolist())
        for event in observed.assigned_events[:state.assigned_history_end] + observed.unassigned_events[:state.unassigned_history_end]:
            for value in (event.item_id, event.before_id, event.after_id):
                if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and int(value) > 0:
                    values.add(int(value))
        ids_cache[query_ref] = values
        return values

    for context_index, context_name in enumerate(CONTEXT_NAMES):
        membership = int(memberships[context_index])
        for bit, scope_name in ((0, "FIT"), (2, "REFIT_TRAIN")):
            key = f"{context_name}:{scope_name}"
            cutoff = int(observed.query_event_limits[cutoff_query_ref]) if cutoff_query_ref >= 0 and membership & (1 << bit) else -1
            scope_hashers[key].update(f"{shard},{local_match},{cutoff}\n".encode("ascii"))
            if cutoff < 0:
                continue
            ids = ids_for(cutoff_query_ref)
            state = observed.states[cutoff_query_ref]
            scope_ids[key].update(ids)
            audit = scope_audit[key]
            audit["included_match_count"] += 1
            audit["last_supported_query_ref_max"] = max(audit["last_supported_query_ref_max"], cutoff_query_ref)
            audit["last_supported_actual_ms_max"] = max(audit["last_supported_actual_ms_max"], int(observed.queries[cutoff_query_ref]))
            canonical = str(observed.metadata.get("canonical_match_key") or "")
            scope_hashers.setdefault(f"{key}:match_keys", hashlib.sha256()).update((canonical + "\n").encode("ascii"))


def _write_vocabulary_artifacts(
    output: Path, scope_ids: Mapping[str, set[int]], scope_audit: dict[str, dict[str, Any]],
    scope_hashers: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit and persist the twelve loadable FIT/REFIT vocabulary objects."""
    from studies.item_node_observed_20261002.embedding import fit_item_vocabulary

    result: dict[str, Any] = {}
    for key in sorted(scope_ids):
        context, scope = key.split(":", 1)
        path = output / f"VOCABULARY_{context}_{scope}.json"
        vocabulary = fit_item_vocabulary(
            sorted(scope_ids[key]), scope_keys=[key], provenance={key: scope}, excluded_scope_keys=(),
        )
        vocabulary.save(path)
        record = dict(scope_audit[key])
        record.update({
            "scope_key": key, "scope_keys": [key], "fit_provenance": {key: scope},
            "item_ids": [int(value) for value in vocabulary.raw_item_ids],
            "item_ids_sha256": hashlib.sha256(json.dumps(list(vocabulary.raw_item_ids), separators=(",", ":")).encode("ascii")).hexdigest(),
            "item_count": int(vocabulary.size), "vocabulary_fingerprint": vocabulary.fingerprint,
            "path": path.name, "cutoff_sha256": scope_hashers[key].hexdigest(),
            "included_match_keys_sha256": scope_hashers[f"{key}:match_keys"].hexdigest(),
            "excluded_scope_checks": {
                "fit_only": True,
                "excluded_scopes": (["STOP", "DEV", "CAL", "TARGET", "TEST"] if scope == "FIT" else ["DEV", "CAL", "TARGET", "TEST"]),
                "excluded_oof_fold": context if context.startswith("oof") else None,
            },
        })
        result[key] = record
    if len(result) != 12:
        raise ObservedBankError(f"expected twelve vocabulary artifacts, wrote {len(result)}")
    _atomic_json(output / "VOCABULARY_SCOPE_META.json", result)
    _atomic_json(output / "VOCABULARY_SCOPE_IDS.json", {key: sorted(values) for key, values in scope_ids.items()})
    return result


def load_observed_vocabulary(output: Path, scope_key: str) -> Any:
    """Load one persisted FIT/REFIT vocabulary by its exact scope key."""
    if not isinstance(scope_key, str) or scope_key.count(":") != 1:
        raise ValueError("scope_key must be one exact context:FIT or context:REFIT_TRAIN key")
    context, scope = scope_key.split(":", 1)
    if context not in CONTEXT_NAMES or scope not in {"FIT", "REFIT_TRAIN"}:
        raise ValueError("scope_key is outside the twelve accepted context scopes")
    from studies.item_node_observed_20261002.embedding import ObservedItemVocabulary
    return ObservedItemVocabulary.load(Path(output) / f"VOCABULARY_{context}_{scope}.json")


def independent_reload_check(output: Path, *, max_queries: int | None = None) -> dict[str, Any]:
    """Reload persisted arrays and verify ragged references in a fresh pass."""
    with ObservedBankLoader(output) as loader:
        n_queries = len(loader.queries) if max_queries is None else min(int(max_queries), len(loader.queries))
        ids_end = 0
        for index in range(n_queries):
            row = loader.queries[index]
            offsets = np.asarray(loader.node_offsets[index], dtype=np.int64)
            if offsets.shape != (PARTICIPANTS + 1,) or np.any(offsets[1:] < offsets[:-1]):
                raise ObservedBankError("reloaded query offsets are invalid")
            ids_end = max(ids_end, int(offsets[-1]))
            if bool(row["unassigned_observed"]) != (int(row["unassigned_history_end"]) > 0):
                raise ObservedBankError("reloaded unassigned observation flag differs from history reference")
        return {"reloaded": True, "checked_queries": n_queries, "max_item_value_offset": ids_end, "independent_replay": False}


def run_bank(
    output: Path = DEFAULT_OUTPUT,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    preparation_root: Path = DEFAULT_PREPARATION_ROOT,
    max_matches: int | None = None,
    workers: int = 0,
    verify_hashes: bool = True,
    strict_full: bool = True,
    disk_reserve_bytes: int = 8 << 30,
) -> dict[str, Any]:
    """Build a deterministic streaming observed bank into a new directory."""
    _set_numeric_threads()
    reader_affinity = _set_process_affinity(0)
    if max_matches is not None and max_matches < 0:
        raise ValueError("max_matches must be nonnegative")
    if workers < 0 or workers > 3:
        raise ValueError("workers must be between 0 and 3")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"bank output already exists: {output}")
    size_estimate = _estimated_output_bytes()
    available_bytes = shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free
    if available_bytes < int(disk_reserve_bytes) + int(size_estimate["estimated_output_bytes"]):
        raise OSError(
            f"observed bank preflight needs {size_estimate['estimated_output_bytes']} bytes plus "
            f"{disk_reserve_bytes} reserve; available={available_bytes}"
        )
    full_gate = bool(strict_full and max_matches is None)
    if full_gate:
        source_before = verify_source_bindings(source_root, preparation_root, verify_hashes=verify_hashes, strict_full=True)
        preparation_before = verify_preparation_bindings(preparation_root, strict_full=True)
        capacities = (EXPECTED_MATCHES, EXPECTED_QUERIES, EXPECTED_NOMINAL_ROWS)
    else:
        source_before = verify_source_bindings(source_root, preparation_root, verify_hashes=verify_hashes, strict_full=False)
        preparation_before = verify_preparation_bindings(preparation_root, strict_full=False)
        capacities = None
    index = PreparationIndex(Path(preparation_root), strict_full=full_gate)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "complete": False, "accepted_bank": False,
        "full_corpus_verified": False, "training_ready": False, "new_models_fitted": False,
        "new_SVI_labels_produced": False, "actual_inventory_complete": "unknown",
        "source_root": str(Path(source_root)), "preparation_root": str(Path(preparation_root)),
        "started_utc": started, "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "CPU_budget": 4,
            "reader_writer_processes": 1, "replay_workers": int(workers), "numeric_threads": 1,
            "bounded_inflight": 12, "GPU_used": False,
            "reader_writer_affinity": reader_affinity, "replay_worker_affinity_targets": [2, 4, 6] if workers else [],
        },
        "code_bindings_before": _code_bindings(), "contract_sha256": file_sha256(CONTRACT_PATH),
        "build_gates": {
            "verify_hashes": bool(verify_hashes), "strict_full": bool(strict_full),
            "max_matches": max_matches, "full_gate": bool(full_gate),
        },
        "source_bindings_before": source_before, "preparation_bindings_before": preparation_before,
        "source_size_estimate": size_estimate,
        "disk_preflight": {"available_bytes": int(available_bytes), "reserve_bytes": int(disk_reserve_bytes)},
    }
    _atomic_json(output / "RUN_MANIFEST.json", manifest)
    writer: _BankWriter | None = None
    aggregate: dict[str, Any] = {
        "n_matches": 0, "n_unique_queries": 0, "n_nominal_rows": 0, "n_item_events": 0,
        "n_assigned_events": 0, "n_unassigned_events": 0, "n_supported_player_queries": 0,
        "n_unassigned_observed_queries": 0, "support_by_participant": np.zeros(PARTICIPANTS, dtype=np.int64),
        "reason_counts": Counter(), "unassigned_reason_counts": Counter(), "item_event_counts": Counter(),
        "item_id_event_counts": Counter(), "by_role": {}, "by_patch": {}, "worker_affinity_masks": Counter(),
    }
    scope_ids = {f"{context}:{scope}": set() for context in CONTEXT_NAMES for scope in ("FIT", "REFIT_TRAIN")}
    scope_audit = {
        f"{context}:{scope}": {
            "context_index": int(context_index), "context": context, "scope": scope,
            "membership_matrix_sha256": preparation_before["metadata"]["context_membership_sha256"],
            "included_match_count": 0, "last_supported_query_ref_max": -1,
            "last_supported_actual_ms_max": -1,
        }
        for context_index, context in enumerate(CONTEXT_NAMES)
        for scope in ("FIT", "REFIT_TRAIN")
    }
    scope_hashers = {
        key: hashlib.sha256()
        for key in (*scope_ids.keys(), *(f"{key}:match_keys" for key in scope_ids))
    }
    started_perf = time.perf_counter()
    try:
        writer = _BankWriter(output, capacities=capacities)
        reader = AcceptedSourceReader(Path(source_root), Path(preparation_root), verify_hashes=False)
        source_iter = iter(()) if max_matches == 0 else reader.iter_matches(max_matches=max_matches)

        def tasks() -> Iterator[tuple[int, Any, Any, tuple[int, ...]]]:
            for match_index, match in enumerate(source_iter):
                prep_index, nominal_rows, context_masks = index.rows_for(match.shard_index, match.local_index)
                if full_gate and len(context_masks) != 6:
                    raise ObservedBankError("strict full observed bank requires all six context memberships")
                if match.canonical_match_key != _decode_key(index.match_table[prep_index]["key"]):
                    raise ObservedBankError("source canonical match key differs from V preparation")
                yield match_index, match, nominal_rows, context_masks

        if workers:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer) as pool:
                pending: deque[tuple[int, Any, Any]] = deque()
                iterator = iter(tasks())
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < 12:
                        try:
                            match_index, match, nominal_rows, context_masks = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.append((match_index, nominal_rows, pool.submit(_source_task, match, nominal_rows, context_masks)))
                    if pending:
                        match_index, nominal_rows, future = pending.popleft()
                        observed = future.result()
                        writer.write(match_index, observed, nominal_rows)
                        if writer._match_count % 128 == 0:
                            writer.write_progress()
                            if shutil.disk_usage(output).free < int(disk_reserve_bytes):
                                raise OSError("observed bank fell below the configured free-space reserve")
                        _merge_stats(aggregate, observed)
                        _vocabulary_scope_update(scope_ids, observed, scope_audit, scope_hashers)
        else:
            for match_index, match, nominal_rows, context_masks in tasks():
                observed = _compact_for_writer(observe_match(match, nominal_rows, context_memberships=context_masks))
                writer.write(match_index, observed, nominal_rows)
                if writer._match_count % 128 == 0:
                    writer.write_progress()
                    if shutil.disk_usage(output).free < int(disk_reserve_bytes):
                        raise OSError("observed bank fell below the configured free-space reserve")
                _merge_stats(aggregate, observed)
                _vocabulary_scope_update(scope_ids, observed, scope_audit, scope_hashers)
        writer_counts = writer.finalize()
        aggregate["n_matches"] = writer_counts["n_matches"]
        aggregate["n_unique_queries"] = writer_counts["n_unique_queries"]
        aggregate["n_nominal_rows"] = writer_counts["n_nominal_rows"]
        if full_gate and (aggregate["n_matches"], aggregate["n_unique_queries"], aggregate["n_nominal_rows"]) != (EXPECTED_MATCHES, EXPECTED_QUERIES, EXPECTED_NOMINAL_ROWS):
            raise ObservedBankError("full observed bank coverage differs from accepted counts")
        _atomic_json(output / "SOURCE_BINDINGS.json", source_before)
        _atomic_json(output / "PREPARATION_BINDINGS.json", preparation_before)
        vocabulary_meta = _write_vocabulary_artifacts(output, scope_ids, scope_audit, scope_hashers)
        summary = _jsonable(aggregate)
        summary["complete_full_corpus_coverage"] = bool(full_gate)
        summary["scope_membership"] = {key: len(values) for key, values in scope_ids.items()}
        summary["vocabulary_scope_audit"] = vocabulary_meta
        _atomic_json(output / "SUMMARY.json", summary)
        _atomic_json(output / "BANK_METADATA.json", {
            "format_version": FORMAT_VERSION,
            "node_order": "participant IDs 1..10",
            "binary_members": {"ITEM_IDS.bin": "little-endian signed int64", "ITEM_COUNTS.bin": "little-endian signed int64"},
            "ragged_offsets": "NODE_OFFSETS.npy rows are per-query ten-node global element offsets",
            "support": "NODE_SUPPORT.npy is recipient-confirmed observed-ledger support; actual inventory completeness remains unknown",
            "reason_bits": {
                "assigned": {str(key): int(value) for key, value in sorted(REASON_BITS.items(), key=lambda row: row[1])},
                "unassigned": {str(key): int(value) for key, value in sorted(UNASSIGNED_REASON_BITS.items(), key=lambda row: row[1])},
            },
            "history_framing": "EVENT_SOURCE_INDEX_* are little-endian int32 source event indices; accepted source banks retain typed fields",
            "history_refs": "MATCH_TABLE first/n fields point into source-index arrays; QUERY_TABLE ends are lazy per-match prefix references",
            "outcome_fields_in_tensors": False,
            "source_root": str(Path(source_root)),
            "preparation_root": str(Path(preparation_root)),
            "vocabulary_fit_performed": True,
            "vocabulary_fit_models": False,
            "vocabulary_scope_metadata": "VOCABULARY_SCOPE_META.json",
            "vocabularies": vocabulary_meta,
        })
        saved_files = _saved_file_records(output)
        manifest.update({
            "complete": True, "accepted_bank": False, "full_corpus_verified": False,
            "vocabulary_fit_performed": True, "vocabulary_fit_models": False,
            "counts": _jsonable({**writer_counts, "n_shards": source_before["n_shards"]}),
            "elapsed_seconds": time.perf_counter() - started_perf, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_bindings_after": verify_source_bindings(source_root, preparation_root, verify_hashes=verify_hashes, strict_full=full_gate),
            "preparation_bindings_after": verify_preparation_bindings(preparation_root, strict_full=full_gate),
            "source_size_estimate_after": _estimated_output_bytes(),
            "vocabulary_scope_audit": vocabulary_meta,
            "code_bindings_after": _code_bindings(), "saved_files": saved_files,
            "independent_reload_required": True,
        })
        if manifest["source_bindings_before"]["records_sha256"] != manifest["source_bindings_after"]["records_sha256"]:
            raise ObservedBankError("source shard bindings changed during observed bank run")
        if manifest["source_size_estimate"]["sha256"] != manifest["source_size_estimate_after"]["sha256"]:
            raise ObservedBankError("accepted source size estimate changed during observed bank run")
        if manifest["preparation_bindings_before"] != manifest["preparation_bindings_after"]:
            raise ObservedBankError("V preparation bindings changed during observed bank run")
        if manifest["code_bindings_before"] != manifest["code_bindings_after"]:
            raise ObservedBankError("code bindings changed during observed bank run")
        _atomic_json(output / "RUN_MANIFEST.json", manifest)
        return manifest
    except Exception as exc:
        if writer is not None:
            for handle_name in ("_ids", "_counts", "_assigned_sources", "_unassigned_sources", "_match_log", "_progress"):
                handle = getattr(writer, handle_name, None)
                if handle is not None and not handle.closed:
                    handle.close()
        manifest.update({"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started_perf})
        _atomic_json(output / "RUN_MANIFEST.json", manifest)
        _atomic_json(output / "FAILED.json", {"failed": True, "error": manifest["error"]})
        raise


__all__ = [
    "DEFAULT_OUTPUT", "PreparationIndex", "independent_reload_check", "load_observed_vocabulary", "run_bank",
    "verify_preparation_bindings", "verify_source_bindings",
]
