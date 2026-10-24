"""Full-corpus independent audit for the direct item transaction ledger.

The audit consumes the accepted observable banks and the already-saved V
reference table.  It never writes to either source tree and never treats the
legacy ``query_item_ledger_known`` flag as evidence for the new ledger.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .ledger import INITIAL_EMPTY_PROVENANCE, ItemTransactionLedger
from .source import (
    AcceptedMatch,
    AcceptedSourceReader,
    DEFAULT_ACCEPTANCE_PATH,
    DEFAULT_PREPARATION_ROOT,
    DEFAULT_SOURCE_ROOT,
    ITEM_TYPE_NAMES,
    SourceValidationError,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "item_node_encoding_20261002"
CONTRACT_PATH = REPORT / "CONTRACT.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "item_node_encoding_20261002" / "full_audit_20261002_1"
EXPECTED_MATCHES = 209_999
EXPECTED_QUERIES = 4_384_026
EXPECTED_NOMINAL_ROWS = 2_618_905
EXPECTED_SHARDS = 411


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _append_bytes(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(value)


def _state_bytes(query_ms: int, known: np.ndarray, reasons: Sequence[Sequence[str]], offsets: np.ndarray,
                item_ids: np.ndarray, counts: np.ndarray) -> bytes:
    parts = [
        np.asarray([query_ms], dtype="<i8").tobytes(),
        np.asarray(known, dtype=np.uint8).reshape(-1).tobytes(),
        np.asarray(offsets, dtype="<i8").reshape(-1).tobytes(),
        np.asarray(item_ids, dtype="<i8").reshape(-1).tobytes(),
        np.asarray(counts, dtype="<i8").reshape(-1).tobytes(),
        json.dumps(tuple(tuple(reason) for reason in reasons), separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
    ]
    result = bytearray()
    for value in parts:
        result.extend(len(value).to_bytes(8, "little", signed=False))
        result.extend(value)
    return bytes(result)


@dataclass(frozen=True)
class CompactLedgerState:
    """A query endpoint without the ledger's repeated transaction history."""

    query_timestamp_ms: int
    known: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    item_ids: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray


def iter_compact_ledger_states(
    ledger: ItemTransactionLedger,
    queries: Sequence[int] | np.ndarray,
) -> Iterator[CompactLedgerState]:
    """Replay sorted queries using ledger blocks, without copying histories."""
    if getattr(ledger, "initial_inventory_provenance", INITIAL_EMPTY_PROVENANCE) != INITIAL_EMPTY_PROVENANCE:
        raise ValueError("compact audit replay only supports the explicit empty initial ledger; nonempty initial state is rejected")
    initial_counts = getattr(ledger, "_initial_counts", {})
    if any(dict(value) for value in initial_counts.values()):
        raise ValueError("compact audit replay cannot silently discard nonempty initial inventory")
    values = [int(value) for value in np.asarray(queries).reshape(-1).tolist()]
    if any(value < 0 for value in values) or any(left > right for left, right in zip(values, values[1:])):
        raise ValueError("queries must be nonnegative and sorted")
    counts: dict[int, dict[int, int]] = {pid: {} for pid in range(1, 11)}
    known = [True] * 10
    reasons: list[list[str]] = [[] for _ in range(10)]
    global_reasons: list[str] = []
    global_unknown = False
    block_index = 0
    blocks = ledger.blocks
    for query in values:
        while block_index < len(blocks) and int(blocks[block_index].timestamp_ms) <= query:
            block = blocks[block_index]
            if block.global_reasons and not global_unknown:
                global_unknown = True
                global_reasons.extend(reason for reason in block.global_reasons if reason not in global_reasons)
                known = [False] * 10
                for row in reasons:
                    for reason in global_reasons:
                        if reason not in row:
                            row.append(reason)
            if not global_unknown:
                for pid, delta in block.deltas.items():
                    row = int(pid) - 1
                    if not known[row]:
                        continue
                    for item_id, net in delta.items():
                        next_count = counts[int(pid)].get(int(item_id), 0) + int(net)
                        if next_count < 0:
                            known[row] = False
                            counts[int(pid)].clear()
                            if "item_replay_underflow" not in reasons[row]:
                                reasons[row].append("item_replay_underflow")
                        elif next_count:
                            counts[int(pid)][int(item_id)] = next_count
                        else:
                            counts[int(pid)].pop(int(item_id), None)
                for pid, block_reasons in block.unknown_participants.items():
                    row = int(pid) - 1
                    known[row] = False
                    counts[int(pid)].clear()
                    for reason in block_reasons:
                        if reason not in reasons[row]:
                            reasons[row].append(reason)
            block_index += 1
        item_ids: list[int] = []
        item_counts: list[int] = []
        offsets = [0]
        state_reasons: list[tuple[str, ...]] = []
        for row, pid in enumerate(range(1, 11)):
            if known[row]:
                for item_id, item_count in sorted(counts[pid].items()):
                    if item_count > 0:
                        item_ids.append(int(item_id))
                        item_counts.append(int(item_count))
            offsets.append(len(item_ids))
            state_reasons.append(tuple(dict.fromkeys(tuple(reasons[row]) + tuple(global_reasons))))
        yield CompactLedgerState(
            query_timestamp_ms=query,
            known=np.asarray(known, dtype=bool),
            reasons=tuple(state_reasons),
            item_ids=np.asarray(item_ids, dtype=np.int64),
            counts=np.asarray(item_counts, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
        )


class NominalReferenceIndex:
    """Memory-mapped independent V references keyed by source shard/local row."""

    def __init__(self, preparation_root: Path):
        self.root = Path(preparation_root)
        self.match_table = np.load(self.root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.v_rows = np.load(self.root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        if self.match_table.ndim != 1 or self.v_rows.ndim != 1:
            raise SourceValidationError("preparation reference arrays must be one-dimensional structured arrays")
        required_match = {"key", "shard", "local_match", "first_nominal_row", "n_nominal_rows"}
        required_rows = {"match_index", "shard", "local_match", "nominal_index", "nominal_ms", "actual_ms", "query_ref", "supported"}
        if not required_match.issubset(self.match_table.dtype.names or ()) or not required_rows.issubset(self.v_rows.dtype.names or ()):
            raise SourceValidationError("preparation reference dtype is missing required identity fields")
        self._by_source: dict[tuple[int, int], int] = {}
        for index, row in enumerate(self.match_table):
            key = (int(row["shard"]), int(row["local_match"]))
            if key in self._by_source:
                raise SourceValidationError("MATCH_TABLE has duplicate source shard/local identity")
            self._by_source[key] = index
        if len(self._by_source) != len(self.match_table):
            raise SourceValidationError("MATCH_TABLE source identity is not one-to-one")

    def rows_for(self, shard_index: int, local_index: int) -> tuple[int, np.ndarray]:
        key = (int(shard_index), int(local_index))
        if key not in self._by_source:
            raise SourceValidationError(f"source match {key} is absent from MATCH_TABLE")
        match_index = self._by_source[key]
        row = self.match_table[match_index]
        first = int(row["first_nominal_row"])
        size = int(row["n_nominal_rows"])
        if first < 0 or size < 0 or first + size > len(self.v_rows):
            raise SourceValidationError("MATCH_TABLE nominal slice is outside V_ROWS")
        selected = self.v_rows[first:first + size]
        if len(selected) != size or np.any(selected["match_index"] != match_index):
            raise SourceValidationError("V_ROWS nominal ownership differs from MATCH_TABLE")
        if len(selected) and np.any(selected["nominal_index"] != np.arange(size, dtype=selected["nominal_index"].dtype)):
            raise SourceValidationError("V_ROWS nominal indices are not complete per match")
        return match_index, selected

    def metadata(self) -> dict[str, Any]:
        keys = [bytes(value).decode("ascii") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in self.match_table["key"]]
        if len(set(keys)) != len(keys):
            raise SourceValidationError("MATCH_TABLE has duplicate canonical keys")
        return {
            "n_matches": int(len(self.match_table)),
            "n_nominal_rows": int(len(self.v_rows)),
            "match_key_sha256": hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest(),
            "source_identity_sha256": hashlib.sha256(self.match_table[["shard", "local_match"]].tobytes()).hexdigest(),
        }


def load_item_catalogs(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load patch-pinned Data Dragon item IDs for unresolved-ID reporting."""
    base = Path(root) / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw"
    catalogs: dict[str, dict[str, Any]] = {}
    for patch in ("15.14", "15.15", "15.16"):
        path = base / f"item_{patch}.1.json"
        if not path.is_file():
            continue
        payload = _json(path)
        if payload.get("version") != patch + ".1" or not isinstance(payload.get("data"), dict):
            raise SourceValidationError(f"item catalog version or data changed: {path}")
        catalogs[patch] = {
            "version": payload["version"],
            "path": str(path),
            "sha256": file_sha256(path),
            "item_ids": frozenset(int(value) for value in payload["data"]),
        }
    return catalogs


def _source_metadata(match: AcceptedMatch) -> dict[str, Any]:
    metadata = dict(match.metadata)
    key = metadata.get("canonical_match_key")
    if isinstance(key, bytes):
        key = key.decode("ascii")
    return {
        "canonical_match_key": str(key) if key is not None else None,
        "patch": metadata.get("patch"),
        "role": metadata.get("role"),
        "original_index": metadata.get("original_index"),
        "shard": int(match.shard_index),
        "local_match": int(match.local_index),
    }


def audit_match(
    match: AcceptedMatch,
    nominal_rows: np.ndarray | None = None,
    *,
    item_catalogs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit one source match and return JSON-safe aggregate metadata."""
    transactions = match.item_transactions()
    ledger = ItemTransactionLedger(transactions)
    states = list(iter_compact_ledger_states(ledger, match.query_ms))
    query_digest = hashlib.sha256()
    key = _source_metadata(match).get("canonical_match_key") or f"shard:{match.shard_index}/local:{match.local_index}"
    _append_bytes(query_digest, str(key).encode("utf-8"))
    support_by_player = np.zeros(10, dtype=np.int64)
    reason_counts: Counter[str] = Counter()
    item_event_counts: Counter[str] = Counter()
    item_id_counts: Counter[str] = Counter()
    item_id_inventory_units: Counter[str] = Counter()
    observed_ids: set[int] = set()
    for event in match.item_events():
        item_event_counts[event.event_type] += 1
        for name, value in (("itemId", event.item_id), ("beforeId", event.before_id), ("afterId", event.after_id)):
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and int(value) > 0:
                item_id_counts[str(int(value))] += 1
                observed_ids.add(int(value))
    unknown_query_count = 0
    all_supported_query_count = 0
    for state in states:
        _append_bytes(query_digest, _state_bytes(state.query_timestamp_ms, state.known, state.reasons, state.offsets, state.item_ids, state.counts))
        support_by_player += state.known.astype(np.int64)
        if bool(np.all(state.known)):
            all_supported_query_count += 1
        else:
            unknown_query_count += 1
        for row, row_reasons in enumerate(state.reasons):
            if not bool(state.known[row]):
                for reason in row_reasons:
                    reason_counts[reason] += 1
        for row in range(10):
            start, end = int(state.offsets[row]), int(state.offsets[row + 1])
            for item_id, count in zip(state.item_ids[start:end].tolist(), state.counts[start:end].tolist()):
                item_id_inventory_units[str(int(item_id))] += int(count)
                observed_ids.add(int(item_id))
    nominal_digest = hashlib.sha256()
    nominal_count = 0
    nominal_supported_count = 0
    nominal_supported_nodes = 0
    if nominal_rows is not None:
        nominal_rows = np.asarray(nominal_rows)
        by_query = {index: state for index, state in enumerate(states)}
        for nominal in nominal_rows:
            query_ref = int(nominal["query_ref"])
            if bool(nominal["supported"]):
                if query_ref < 0 or query_ref >= len(states):
                    raise SourceValidationError("nominal V row references an invalid query")
                if int(nominal["actual_ms"]) != int(match.query_ms[query_ref]) or int(nominal["actual_ms"]) != int(nominal["nominal_ms"]):
                    raise SourceValidationError("nominal V row timestamp differs from source query reference")
                state = by_query[query_ref]
                nominal_supported_count += 1
                nominal_supported_nodes += int(state.known.sum())
                payload = _state_bytes(state.query_timestamp_ms, state.known, state.reasons, state.offsets, state.item_ids, state.counts)
            else:
                if query_ref != -1:
                    raise SourceValidationError("unsupported nominal V row retains a query reference")
                payload = b"unsupported"
            nominal_digest.update(np.asarray([int(nominal["nominal_index"])], dtype="<i8").tobytes())
            nominal_digest.update(np.asarray([query_ref], dtype="<i8").tobytes())
            _append_bytes(nominal_digest, payload)
            nominal_count += 1
    unresolved: list[int] = []
    patch = _source_metadata(match).get("patch")
    catalog = item_catalogs.get(str(patch)) if item_catalogs is not None and patch is not None else None
    if catalog is not None:
        allowed = catalog["item_ids"]
        unresolved = sorted(item_id for item_id in observed_ids if item_id not in allowed)
    reasons = dict(sorted(reason_counts.items()))
    return {
        **_source_metadata(match),
        "n_events": int(match.n_events),
        "n_item_events": int(len(transactions)),
        "n_unique_queries": int(len(states)),
        "n_recorded_ledger_supported_player_queries": int(support_by_player.sum()),
        "n_unknown_player_queries": int(len(states) * 10 - support_by_player.sum()),
        "n_all_players_supported_queries": int(all_supported_query_count),
        "n_any_player_unknown_queries": int(unknown_query_count),
        "support_by_participant": [int(value) for value in support_by_player.tolist()],
        "reason_counts": reasons,
        "item_event_counts": dict(sorted(item_event_counts.items())),
        "item_id_event_counts": dict(sorted(item_id_counts.items(), key=lambda row: int(row[0]))),
        "item_id_inventory_units": dict(sorted(item_id_inventory_units.items(), key=lambda row: int(row[0]))),
        "observed_item_ids": sorted(observed_ids),
        "unresolved_item_ids": unresolved,
        "per_query_inventory_digest_sha256": query_digest.hexdigest(),
        "n_nominal_rows": nominal_count,
        "n_supported_nominal_rows": nominal_supported_count,
        "n_recorded_ledger_supported_nominal_player_queries": nominal_supported_nodes,
        "per_nominal_query_inventory_digest_sha256": nominal_digest.hexdigest(),
        "legacy_query_item_ledger_known_count": int(match.query_item_ledger_known.sum()),
        "actual_inventory_complete": "unknown",
        "initial_state": "initial_empty_recorded_transaction_ledger_assumption",
    }


def _merge_counter(target: Counter[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, np.integer)):
            target[str(key)] += int(value)


def _catalog_manifest(catalogs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        patch: {key: value for key, value in record.items() if key != "item_ids"}
        | {"n_item_ids": len(record["item_ids"])}
        for patch, record in catalogs.items()
    }


def _code_bindings() -> dict[str, str]:
    files = {
        "studies/item_node_encoding_20261002/source.py": Path(__file__).with_name("source.py"),
        "studies/item_node_encoding_20261002/audit.py": Path(__file__),
        "studies/item_node_encoding_20261002/__main__.py": Path(__file__).with_name("__main__.py"),
        "studies/item_node_encoding_20261002/ledger.py": Path(__file__).with_name("ledger.py"),
        "studies/observable_state_20261002/bank.py": ROOT / "studies" / "observable_state_20261002" / "bank.py",
        "reports/item_node_encoding_20261002/CONTRACT.json": CONTRACT_PATH,
    }
    return {name: file_sha256(path) for name, path in files.items() if path.is_file()}


def verify_preparation_bindings(preparation_root: Path) -> dict[str, Any]:
    root = Path(preparation_root)
    acceptance_path = REPORT.parent / "v_preparation_20261002" / "ACCEPTANCE.json"
    files = {
        "RUN_MANIFEST.json": root / "RUN_MANIFEST.json",
        "SOURCE_SHARD_LINKS.json": root / "SOURCE_SHARD_LINKS.json",
        "MATCH_TABLE.npy": root / "MATCH_TABLE.npy",
        "V_ROWS.npy": root / "V_ROWS.npy",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise SourceValidationError("preparation output is missing: " + ", ".join(missing))
    if not acceptance_path.is_file():
        raise SourceValidationError("accepted preparation record is missing")
    acceptance = _json(acceptance_path)
    manifest = _json(files["RUN_MANIFEST.json"])
    if acceptance.get("preparation_accepted") is not True or acceptance.get("full_corpus_verified") is not True:
        raise SourceValidationError("V preparation is not accepted as a full reference source")
    accepted_run = Path(str(acceptance.get("accepted_run", "")))
    if accepted_run.resolve() != root.resolve():
        raise SourceValidationError("V preparation accepted_run differs from requested source")
    if acceptance.get("manifest_sha256") != file_sha256(files["RUN_MANIFEST.json"]):
        raise SourceValidationError("V preparation RUN_MANIFEST hash differs from its acceptance record")
    if manifest.get("complete") is not True or manifest.get("accepted_preparation") is not True:
        raise SourceValidationError("V preparation RUN_MANIFEST is not complete and accepted")
    if manifest.get("counts", {}).get("n_matches") != EXPECTED_MATCHES or manifest.get("counts", {}).get("n_nominal_rows") != EXPECTED_NOMINAL_ROWS:
        raise SourceValidationError("V preparation RUN_MANIFEST coverage differs from the accepted full corpus")
    saved = manifest.get("saved_files", {})
    records: dict[str, Any] = {}
    for name, path in files.items():
        digest = file_sha256(path)
        expected = saved.get(name, {}).get("sha256")
        if name != "RUN_MANIFEST.json" and not isinstance(expected, str):
            raise SourceValidationError(f"V preparation artifact has no saved SHA binding: {path}")
        if isinstance(expected, str) and digest != expected:
            raise SourceValidationError(f"V preparation artifact hash changed: {path}")
        records[name] = {"sha256": digest, "bytes": path.stat().st_size}
    if _json(files["SOURCE_SHARD_LINKS.json"]).get("source") != manifest.get("source"):
        raise SourceValidationError("V preparation source link does not match its RUN_MANIFEST source")
    return {
        "acceptance_path": str(acceptance_path),
        "acceptance_sha256": file_sha256(acceptance_path),
        "manifest_sha256": records["RUN_MANIFEST.json"]["sha256"],
        "files": records,
    }


def verify_accepted_source_binding(source_root: Path) -> dict[str, Any]:
    """Anchor the source directory to the immutable observable acceptance."""
    root = Path(source_root)
    if not DEFAULT_ACCEPTANCE_PATH.is_file():
        raise SourceValidationError("accepted observable-state record is missing")
    acceptance = _json(DEFAULT_ACCEPTANCE_PATH)
    if acceptance.get("phase_accepted") is not True:
        raise SourceValidationError("observable-state source phase is not accepted")
    accepted_run = Path(str(acceptance.get("accepted_run", "")))
    if accepted_run.resolve() != root.resolve():
        raise SourceValidationError("observable-state accepted_run differs from requested source")
    manifest_path = root / "RUN_MANIFEST.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != acceptance.get("run_manifest_sha256"):
        raise SourceValidationError("observable-state RUN_MANIFEST differs from its acceptance record")
    manifest = _json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("accepted") is not True:
        raise SourceValidationError("observable-state RUN_MANIFEST is not complete and accepted")
    return {
        "acceptance_path": str(DEFAULT_ACCEPTANCE_PATH),
        "acceptance_sha256": file_sha256(DEFAULT_ACCEPTANCE_PATH),
        "run_manifest_sha256": file_sha256(manifest_path),
        "n_matches": acceptance.get("counts", {}).get("n_matches"),
        "n_unique_queries": acceptance.get("counts", {}).get("n_unique_queries"),
    }


def _accepted_audit_status(complete_expected: bool, verify_hashes: bool, strict_full: bool) -> bool:
    """Only a strict, hash-bound full run can be marked accepted."""
    return bool(complete_expected and verify_hashes and strict_full)


def run_audit(
    output: Path = DEFAULT_OUTPUT,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    preparation_root: Path = DEFAULT_PREPARATION_ROOT,
    max_matches: int | None = None,
    verify_hashes: bool = True,
    strict_full: bool = True,
) -> dict[str, Any]:
    """Run the source audit into a new, never-overwritten output directory."""
    if max_matches is not None and max_matches < 0:
        raise ValueError("max_matches must be nonnegative")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "AUDIT.log"
    manifest: dict[str, Any] = {
        "version": "item_node_direct_ledger_audit_v2_20261002",
        "complete": False,
        "accepted_audit": False,
        "training_ready": False,
        "new_models_fitted": False,
        "new_SVI_labels_produced": False,
        "actual_inventory_complete": "unknown",
        "source_root": str(Path(source_root)),
        "preparation_root": str(Path(preparation_root)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "CPU_budget": 4,
            "audit_numeric_threads": 1, "GPU_used": False,
            "numeric_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "bindings": _code_bindings(),
    }
    _atomic_json(output / "RUN_MANIFEST.json", manifest)
    start = time.perf_counter()
    source_before: dict[str, Any] | None = None
    source_after: dict[str, Any] | None = None
    source_acceptance_before: dict[str, Any] | None = None
    source_acceptance_after: dict[str, Any] | None = None
    preparation_before: dict[str, Any] | None = None
    preparation_after: dict[str, Any] | None = None
    code_before: dict[str, str] = dict(manifest["bindings"])
    matches_seen = queries_seen = nominal_seen = nominal_supported_seen = 0
    aggregate = Counter()
    support_by_participant = np.zeros(10, dtype=np.int64)
    reasons = Counter()
    event_types = Counter()
    item_ids = Counter()
    inventory_units = Counter()
    unresolved = Counter()
    catalog: dict[str, dict[str, Any]] = {}
    try:
        prep_bindings = verify_preparation_bindings(Path(preparation_root))
        preparation_before = prep_bindings
        source_acceptance_before = verify_accepted_source_binding(Path(source_root))
        references = NominalReferenceIndex(Path(preparation_root))
        # The explicit before-binding pass below is the one source hash pass
        # for the run.  Do not repeat it for every shard iterator.
        source_reader = AcceptedSourceReader(Path(source_root), Path(preparation_root), verify_hashes=verify_hashes)
        source_before = source_reader.verify_source_bindings()
        if source_before["n_shards"] != EXPECTED_SHARDS and strict_full:
            raise SourceValidationError(f"expected {EXPECTED_SHARDS} source shards, got {source_before['n_shards']}")
        catalog = load_item_catalogs(ROOT)
        _atomic_json(output / "SOURCE_BINDINGS.json", {"before": source_before, "source_acceptance_before": source_acceptance_before,
                                                         "preparation_before": prep_bindings, "catalogs": _catalog_manifest(catalog)})
        _atomic_json(output / "PREPARATION_REFERENCE_METADATA.json", references.metadata())
        aggregate["n_shards"] = source_before["n_shards"]
        # Hashes were checked in the explicit before pass.  Iteration only
        # validates bank shapes and source references; it must not reread all
        # 411 archives before every shard.
        source_reader.verify_hashes = False
        match_path = output / "MATCH_AUDIT.jsonl"
        with match_path.open("x", encoding="utf-8") as handle:
            for match in source_reader.iter_matches(max_matches=max_matches):
                match_index, nominal_rows = references.rows_for(match.shard_index, match.local_index)
                table_key = references.match_table[match_index]["key"]
                table_key = table_key.decode("ascii") if isinstance(table_key, (bytes, np.bytes_)) else str(table_key)
                source_key = _source_metadata(match)["canonical_match_key"]
                if source_key is None or source_key != table_key:
                    raise SourceValidationError("source match key differs from MATCH_TABLE metadata")
                row = audit_match(match, nominal_rows, item_catalogs=catalog)
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                matches_seen += 1
                queries_seen += int(row["n_unique_queries"])
                nominal_seen += int(row["n_nominal_rows"])
                nominal_supported_seen += int(row["n_supported_nominal_rows"])
                support_by_participant += np.asarray(row["support_by_participant"], dtype=np.int64)
                _merge_counter(reasons, row["reason_counts"])
                _merge_counter(event_types, row["item_event_counts"])
                _merge_counter(item_ids, row["item_id_event_counts"])
                _merge_counter(inventory_units, row["item_id_inventory_units"])
                _merge_counter(unresolved, {str(value): 1 for value in row["unresolved_item_ids"]})
                aggregate["n_item_events"] += int(row["n_item_events"])
                aggregate["n_recorded_ledger_supported_player_queries"] += int(row["n_recorded_ledger_supported_player_queries"])
                aggregate["n_unknown_player_queries"] += int(row["n_unknown_player_queries"])
                aggregate["n_all_players_supported_queries"] += int(row["n_all_players_supported_queries"])
                aggregate["n_any_player_unknown_queries"] += int(row["n_any_player_unknown_queries"])
                if matches_seen % 32 == 0:
                    log_path.open("a", encoding="utf-8").write(json.dumps({"matches": matches_seen, "queries": queries_seen}) + "\n")
        source_reader.verify_hashes = verify_hashes
        source_after = source_reader.verify_source_bindings()
        source_acceptance_after = verify_accepted_source_binding(Path(source_root))
        if source_before["records_sha256"] != source_after["records_sha256"]:
            raise SourceValidationError("source shard hashes changed during audit")
        if source_acceptance_before != source_acceptance_after:
            raise SourceValidationError("observable-state acceptance bindings changed during audit")
        preparation_after = verify_preparation_bindings(Path(preparation_root))
        if preparation_before != preparation_after:
            raise SourceValidationError("V preparation reference hashes changed during audit")
        code_after = _code_bindings()
        if code_before != code_after:
            raise SourceValidationError("audit code bindings changed during audit")
        complete_expected = (matches_seen == EXPECTED_MATCHES and queries_seen == EXPECTED_QUERIES and
                             nominal_seen == EXPECTED_NOMINAL_ROWS and nominal_supported_seen == EXPECTED_NOMINAL_ROWS)
        if strict_full and not complete_expected:
            raise SourceValidationError(
                f"full coverage mismatch: matches={matches_seen}, queries={queries_seen}, nominal={nominal_seen}, "
                f"supported_nominal={nominal_supported_seen}"
            )
        _atomic_json(output / "SOURCE_BINDINGS.json", {"before": source_before, "after": source_after,
                                                         "source_acceptance_before": source_acceptance_before,
                                                         "source_acceptance_after": source_acceptance_after,
                                                         "preparation_before": preparation_before,
                                                         "preparation_after": preparation_after,
                                                         "catalogs": _catalog_manifest(catalog)})
        aggregate.update(n_matches=matches_seen, n_unique_queries=queries_seen, n_nominal_rows=nominal_seen,
                         n_supported_nominal_rows=nominal_supported_seen, n_unique_observed_item_ids=len(item_ids),
                         n_unresolved_item_ids=len(unresolved))
        aggregate_json = {
            "complete_full_corpus_coverage": bool(complete_expected),
            "source_coverage": {"n_matches": matches_seen, "n_unique_query_timestamps": queries_seen,
                                "n_nominal_V_rows": nominal_seen, "n_supported_nominal_V_rows": nominal_supported_seen},
            "recorded_ledger_supported": {"per_participant_query_counts": [int(value) for value in support_by_participant.tolist()],
                                           "n_player_queries": int(aggregate["n_recorded_ledger_supported_player_queries"])},
            "unknown_reasons": dict(sorted(reasons.items())),
            "item_event_counts": dict(sorted(event_types.items())),
            "item_id_event_counts": dict(sorted(item_ids.items(), key=lambda row: int(row[0]))),
            "item_id_inventory_units": dict(sorted(inventory_units.items(), key=lambda row: int(row[0]))),
            "unresolved_item_ids": sorted((int(value), int(count)) for value, count in unresolved.items()),
            "actual_inventory_complete": "unknown",
            "initial_state": "initial_empty_recorded_transaction_ledger_assumption",
        }
        _atomic_json(output / "AUDIT_AGGREGATE.json", aggregate_json)
        _atomic_json(output / "ITEM_ID_INVENTORY.json", {"catalogs": _catalog_manifest(catalog), "observed_event_ids": dict(sorted(item_ids.items(), key=lambda row: int(row[0]))), "unresolved_item_ids": sorted((int(value), int(count)) for value, count in unresolved.items())})
        manifest.update({
            "complete": True,
            "accepted_audit": _accepted_audit_status(complete_expected, verify_hashes, strict_full),
            "full_corpus_verified": _accepted_audit_status(complete_expected, verify_hashes, strict_full),
            "counts": dict(aggregate),
            "elapsed_seconds": time.perf_counter() - start,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_before_records_sha256": source_before["records_sha256"],
            "source_after_records_sha256": source_after["records_sha256"],
            "source_acceptance_before": source_acceptance_before,
            "source_acceptance_after": source_acceptance_after,
            "code_bindings_before": code_before,
            "code_bindings_after": code_after,
            "preparation_bindings_before": preparation_before,
            "preparation_bindings_after": preparation_after,
            "saved_files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                            for name in ("MATCH_AUDIT.jsonl", "SOURCE_BINDINGS.json", "PREPARATION_REFERENCE_METADATA.json", "AUDIT_AGGREGATE.json", "ITEM_ID_INVENTORY.json")},
        })
        _atomic_json(output / "RUN_MANIFEST.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update({"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - start})
        _atomic_json(output / "RUN_MANIFEST.json", manifest)
        _atomic_json(output / "FAILED.json", {"failed": True, "error": manifest["error"], "manifest": manifest})
        raise


__all__ = [
    "CompactLedgerState", "DEFAULT_OUTPUT", "NominalReferenceIndex", "audit_match",
    "iter_compact_ledger_states", "load_item_catalogs", "run_audit", "verify_accepted_source_binding",
    "verify_preparation_bindings",
]
