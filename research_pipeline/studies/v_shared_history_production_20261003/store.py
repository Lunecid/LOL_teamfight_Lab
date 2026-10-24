"""Bounded, immutable shared-history shard storage.

Each match is one independent gzip member in ``shard.bin``.  The index keeps
the byte range and query identity for every member, so a reader can seek to a
single match without materialising the rest of a shard.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from numbers import Integral
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from studies.v_model_input_20261003.view import make_view
from studies.v_shared_history_20261003.bank import loader_fingerprint


VERSION = "shared_history_shard_store_20261003_v2"
INDEX_VERSION = "shared_history_shard_index_20261003_v2"
DEFAULT_MAX_MATCHES = 256
_MANIFEST_NAME = "MANIFEST.json"
_INDEX_NAME = "index.json"
_BINARY_NAME = "shard.bin"


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _canonical_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe identity copy whose structure can be compared exactly."""

    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    try:
        encoded = json.dumps(
            dict(identity),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("identity must be JSON serialisable without NaN") from exc
    if not isinstance(result, dict):  # pragma: no cover - json object guard
        raise ValueError("identity must be a JSON object")
    # The loader fingerprint is part of the immutable identity used by the
    # query reader.  Keeping this check at the boundary prevents a shard from
    # being written without enough information to validate its source later.
    fingerprint = result.get("loader_manifest_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("identity loader_manifest_sha256 must be a SHA256 hex digest")
    return result


def _plain(value: Any) -> Any:
    """Convert view values, including NumPy scalars/arrays, to JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    # Importing numpy at module import would make the storage layer needlessly
    # expensive for readers that only inspect manifests.  NumPy values expose
    # the two conversion methods used here without requiring a hard import.
    if hasattr(value, "tolist") and callable(value.tolist):
        return _plain(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a new JSON file atomically, refusing an existing destination."""

    pending = path.with_name(f".{path.name}.pending")
    if pending.exists() or path.exists():
        raise FileExistsError(path)
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        pending.rename(path)
    except BaseException:
        # Preserve the pending file as evidence of an interrupted attempt.
        raise


def _record_payload(record: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise TypeError("match records must be mappings")
    query_id = _nonnegative_integer(record.get("query_id"), "query_id")
    query_ms = _nonnegative_integer(record.get("query_ms"), "query_ms")
    view = record.get("view")
    if not isinstance(view, Mapping):
        raise ValueError("match records require a view mapping")
    if "nonitem_history" in view:
        nonitem = view["nonitem_history"]
    elif "nonitem" in view:
        nonitem = view["nonitem"]
    else:
        raise ValueError("view requires nonitem_history")
    item = view.get("item_history", view.get("item"))
    payload = {
        "query_id": query_id,
        "query_ms": query_ms,
        "nonitem": _plain(nonitem),
        "item": _plain(item),
    }
    # Validate the payload before touching the shard stream.  This makes
    # malformed values fail while the attempt directory remains inspectable.
    json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return query_id, query_ms, payload


def _write_match_member(stream, match_index: int, records: Iterable[Mapping[str, Any]]) -> tuple[int, int, list[int], list[int]]:
    offset = int(stream.tell())
    gzip_stream = gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=6, mtime=0)
    query_ids: list[int] = []
    query_ms: list[int] = []
    first = True
    try:
        gzip_stream.write(b'{"match_index":')
        gzip_stream.write(str(match_index).encode("ascii"))
        gzip_stream.write(b',"records":[')
        for record in records:
            query_id, cutoff, payload = _record_payload(record)
            if query_ids:
                if query_id != query_ids[-1] + 1 or cutoff <= query_ms[-1]:
                    raise ValueError("match queries must be contiguous and time ordered")
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if not first:
                gzip_stream.write(b",")
            gzip_stream.write(encoded)
            first = False
            query_ids.append(query_id)
            query_ms.append(cutoff)
        gzip_stream.write(b"]}")
    finally:
        gzip_stream.close()
    return offset, int(stream.tell()) - offset, query_ids, query_ms


def write_shard(
    directory: str | os.PathLike[str],
    matches: Iterable[tuple[int, Iterable[Mapping[str, Any]]]],
    identity: Mapping[str, Any],
    *,
    max_matches: int = DEFAULT_MAX_MATCHES,
) -> dict[str, Any]:
    """Write one immutable shard from ``(match_index, records)`` iterators.

    The directory is created exclusively.  Data and index files are written
    before the completed manifest, so an interrupted attempt has no loadable
    manifest and remains available for diagnosis.  At most ``max_matches``
    match members are accepted, bounding writer state and shard size.
    """

    identity_copy = _canonical_identity(identity)
    max_matches = _nonnegative_integer(max_matches, "max_matches")
    if max_matches == 0:
        raise ValueError("max_matches must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    binary_path = directory / _BINARY_NAME
    index_path = directory / _INDEX_NAME
    manifest_path = directory / _MANIFEST_NAME
    entries: list[dict[str, Any]] = []
    previous_match: int | None = None
    previous_query: int | None = None
    try:
        with binary_path.open("xb") as stream:
            for match_index, records in matches:
                if len(entries) >= max_matches:
                    raise ValueError(f"shard exceeds max_matches={max_matches}")
                match_index = _nonnegative_integer(match_index, "match_index")
                if previous_match is not None and match_index != previous_match + 1:
                    raise ValueError("match indices must be contiguous and ordered")
                offset, length, query_ids, query_ms = _write_match_member(stream, match_index, records)
                if query_ids and previous_query is not None and query_ids[0] != previous_query + 1:
                    raise ValueError("query ids must be contiguous across matches")
                entries.append(
                    {
                        "match_index": match_index,
                        "offset": offset,
                        "length": length,
                        "n_queries": len(query_ids),
                        "query_ids": query_ids,
                        "query_ms": query_ms,
                        "query_id_first": query_ids[0] if query_ids else None,
                        "query_id_last": query_ids[-1] if query_ids else None,
                        "query_ms_first": query_ms[0] if query_ms else None,
                        "query_ms_last": query_ms[-1] if query_ms else None,
                    }
                )
                previous_match = match_index
                if query_ids:
                    previous_query = query_ids[-1]
            stream.flush()
            os.fsync(stream.fileno())
        if not entries:
            raise ValueError("empty shards are prohibited")
        index = {
            "version": INDEX_VERSION,
            "matches": entries,
            "n_matches": len(entries),
            "n_queries": sum(entry["n_queries"] for entry in entries),
        }
        _write_json(index_path, index)
        binary_hash = _sha256(binary_path)
        index_hash = _sha256(index_path)
        manifest = {
            "version": VERSION,
            "complete": True,
            "storage_complete": True,
            "semantically_verified": False,
            "training_ready": False,
            "identity": copy.deepcopy(identity_copy),
            "binary": _BINARY_NAME,
            "index": _INDEX_NAME,
            "binary_bytes": binary_path.stat().st_size,
            "index_bytes": index_path.stat().st_size,
            "binary_sha256": binary_hash,
            "bytes": binary_path.stat().st_size,
            "sha256": binary_hash,
            "index_sha256": index_hash,
            "n_matches": len(entries),
            "n_queries": index["n_queries"],
            "match_indices": [entry["match_index"] for entry in entries],
        }
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException:
        # The directory and every completed prefix are intentionally retained;
        # callers can inspect or record the failed attempt and choose a new
        # unique directory for a retry.
        raise


def _validate_index(index: Mapping[str, Any], binary_bytes: int) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if index.get("version") != INDEX_VERSION:
        raise ValueError("shard index version differs")
    entries = index.get("matches")
    if not isinstance(entries, list) or not entries:
        raise ValueError("shard index has no matches")
    if index.get("n_matches") != len(entries):
        raise ValueError("shard index match count differs")
    parsed: list[dict[str, Any]] = []
    query_owner: dict[int, dict[str, Any]] = {}
    previous_match: int | None = None
    previous_query: int | None = None
    previous_end = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("shard index entry is not an object")
        match_index = _nonnegative_integer(entry.get("match_index"), "match_index")
        offset = _nonnegative_integer(entry.get("offset"), "offset")
        length = _nonnegative_integer(entry.get("length"), "length")
        if length == 0 or offset != previous_end or offset + length > binary_bytes:
            raise ValueError("shard index byte ranges are not contiguous")
        if previous_match is not None and match_index != previous_match + 1:
            raise ValueError("shard index match order differs")
        query_ids = entry.get("query_ids")
        query_ms = entry.get("query_ms")
        n_queries = entry.get("n_queries")
        if not isinstance(query_ids, list) or not isinstance(query_ms, list) or len(query_ids) != len(query_ms):
            raise ValueError("shard index query identity is invalid")
        if isinstance(n_queries, bool) or not isinstance(n_queries, int) or n_queries < 0:
            raise ValueError("shard index query count differs")
        if n_queries == 0:
            if query_ids or query_ms or any(
                entry.get(name) is not None
                for name in ("query_id_first", "query_id_last", "query_ms_first", "query_ms_last")
            ):
                raise ValueError("empty shard match claims query identity")
            ids: list[int] = []
            times: list[int] = []
        elif not query_ids:
            raise ValueError("shard index query identity is invalid")
        else:
            ids = [_nonnegative_integer(value, "query_id") for value in query_ids]
            times = [_nonnegative_integer(value, "query_ms") for value in query_ms]
        if any(right != left + 1 for left, right in zip(ids, ids[1:])):
            raise ValueError("shard index query ids are not contiguous")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("shard index query times are not ordered")
        if ids and previous_query is not None and ids[0] != previous_query + 1:
            raise ValueError("shard index query ids are not contiguous across matches")
        if n_queries != len(ids):
            raise ValueError("shard index query count differs")
        if ids and (
            entry.get("query_id_first") != ids[0]
            or entry.get("query_id_last") != ids[-1]
            or entry.get("query_ms_first") != times[0]
            or entry.get("query_ms_last") != times[-1]
        ):
            raise ValueError("shard index query summary differs")
        current = dict(entry)
        current["match_index"] = match_index
        current["offset"] = offset
        current["length"] = length
        current["query_ids"] = ids
        current["query_ms"] = times
        parsed.append(current)
        for query_id in ids:
            if query_id in query_owner:
                raise ValueError("shard index query id is duplicated")
            query_owner[query_id] = current
        previous_match = match_index
        if ids:
            previous_query = ids[-1]
        previous_end = offset + length
    if previous_end != binary_bytes:
        raise ValueError("shard index does not cover the shard binary")
    expected_queries = sum(entry["n_queries"] for entry in parsed)
    if index.get("n_queries") != expected_queries:
        raise ValueError("shard index query count differs")
    return parsed, query_owner


class ShardReader:
    """Read one committed shard and reconstruct one query view at a time."""

    def __init__(self, directory: str | os.PathLike[str], identity: Mapping[str, Any]):
        self.directory = Path(directory)
        self.manifest = self._load_manifest(identity)
        self.identity = copy.deepcopy(self.manifest["identity"])
        binary_name = self.manifest.get("binary")
        index_name = self.manifest.get("index")
        if binary_name != _BINARY_NAME or index_name != _INDEX_NAME:
            raise ValueError("shard file names differ")
        self.binary_path = self.directory / _BINARY_NAME
        self.index_path = self.directory / _INDEX_NAME
        if not self.binary_path.is_file() or not self.index_path.is_file():
            raise ValueError("committed shard files are incomplete")
        if self.binary_path.stat().st_size != self.manifest.get("binary_bytes") or _sha256(self.binary_path) != self.manifest.get("binary_sha256"):
            raise ValueError("shard binary hash differs")
        if self.index_path.stat().st_size != self.manifest.get("index_bytes") or _sha256(self.index_path) != self.manifest.get("index_sha256"):
            raise ValueError("shard index hash differs")
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("shard index is unreadable") from exc
        entries, query_owner = _validate_index(index, self.binary_path.stat().st_size)
        if self.manifest.get("n_matches") != len(entries) or self.manifest.get("n_queries") != len(query_owner) or self.manifest.get("match_indices") != [entry["match_index"] for entry in entries]:
            raise ValueError("shard manifest/index summary differs")
        # Keep the on-disk JSON available for diagnostics while exposing the
        # bounded lookup mapping used by coverage checks and readers.
        self.index_manifest = index
        self.index = {entry["match_index"]: entry for entry in entries}
        self.entries = tuple(entries)
        self._entries_by_match = {entry["match_index"]: entry for entry in entries}
        self._entries_by_query = query_owner
        self.match_indices = [entry["match_index"] for entry in entries]
        self.query_ids = sorted(query_owner)
        self._cached_match_index: int | None = None
        self._cached_match_records: list[dict[str, Any]] | None = None
        # Loaders are an immutable source binding for a reader.  Keep the
        # object itself alongside its fingerprint so a distinct loader object
        # cannot accidentally reuse the first binding's validation result.
        self._loader_fingerprints: list[tuple[Any, str]] = []

    def _load_manifest(self, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
        path = self.directory / _MANIFEST_NAME
        if not path.is_file():
            raise ValueError("shard is incomplete: completed manifest is missing")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("shard manifest is unreadable") from exc
        if not isinstance(manifest, dict) or manifest.get("version") != VERSION or manifest.get("complete") is not True or manifest.get("storage_complete") is not True:
            raise ValueError("shard is incomplete or has the wrong version")
        expected = _canonical_identity(expected_identity)
        if manifest.get("identity") != expected:
            raise ValueError("shard identity differs")
        if not isinstance(manifest.get("semantically_verified"), bool):
            raise ValueError("shard semantic verification flag is invalid")
        return manifest

    def _records_for_match(self, match_index: int) -> list[dict[str, Any]]:
        match_index = _nonnegative_integer(match_index, "match_index")
        if match_index == self._cached_match_index and self._cached_match_records is not None:
            return self._cached_match_records
        try:
            entry = self._entries_by_match[match_index]
        except KeyError as exc:
            raise KeyError(match_index) from exc
        with self.binary_path.open("rb") as stream:
            stream.seek(entry["offset"])
            raw = stream.read(entry["length"])
        if len(raw) != entry["length"]:
            raise ValueError("shard match member is truncated")
        try:
            payload = gzip.decompress(raw)
            value = json.loads(payload.decode("utf-8"))
        except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("shard match member is corrupt") from exc
        if not isinstance(value, dict) or value.get("match_index") != match_index or not isinstance(value.get("records"), list):
            raise ValueError("shard match payload identity differs")
        records = value["records"]
        if len(records) != entry["n_queries"]:
            raise ValueError("shard match query count differs")
        ids: list[int] = []
        times: list[int] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("shard match record is invalid")
            query_id = _nonnegative_integer(record.get("query_id"), "query_id")
            query_ms = _nonnegative_integer(record.get("query_ms"), "query_ms")
            if not {"query_id", "query_ms", "nonitem", "item"} <= set(record):
                raise ValueError("shard match record fields differ")
            ids.append(query_id)
            times.append(query_ms)
        if ids != entry["query_ids"] or times != entry["query_ms"]:
            raise ValueError("shard match query identity differs")
        self._cached_match_index = match_index
        self._cached_match_records = records
        return records

    def match_records(self, match_index: int) -> list[dict[str, Any]]:
        """Load and validate exactly one compressed match member.

        The reader retains only the most recently requested match.  A copy is
        returned so callers cannot mutate that bounded cache.
        """

        return copy.deepcopy(self._records_for_match(match_index))

    def _check_loader(self, loader: Any) -> None:
        expected_fingerprint = self.identity["loader_manifest_sha256"]
        for bound_loader, fingerprint in self._loader_fingerprints:
            if bound_loader is loader:
                if fingerprint != expected_fingerprint:
                    raise ValueError("shard/source loader manifest differs")
                return
        fingerprint = loader_fingerprint(loader)
        if fingerprint != expected_fingerprint:
            raise ValueError("shard/source loader manifest differs")
        self._loader_fingerprints.append((loader, fingerprint))

    def query_view(self, loader: Any, query_id: int) -> dict[str, Any]:
        query_id = _nonnegative_integer(query_id, "query_id")
        self._check_loader(loader)
        try:
            entry = self._entries_by_query[query_id]
        except KeyError as exc:
            raise KeyError(query_id) from exc
        records = self._records_for_match(entry["match_index"])
        row = next(record for record in records if record["query_id"] == query_id)
        supplied = loader.query_input(query_id, include_histories=False)
        audit = supplied.get("audit") if isinstance(supplied, Mapping) else None
        try:
            audit_match = _nonnegative_integer(audit["match_index"], "audit match_index") if isinstance(audit, Mapping) else -1
            audit_time = _nonnegative_integer(audit["query_ms"], "audit query_ms") if isinstance(audit, Mapping) else -1
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("shard/source query identity differs") from exc
        if audit_match != entry["match_index"] or audit_time != row["query_ms"]:
            raise ValueError("shard/source query identity differs")
        queries = getattr(loader, "queries", None)
        if queries is not None:
            try:
                reference = queries[query_id]
                if int(reference["match_index"]) != entry["match_index"] or int(reference["query_ms"]) != row["query_ms"]:
                    raise ValueError("shard/source query identity differs")
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc) == "shard/source query identity differs":
                    raise
                raise ValueError("shard/source query identity differs") from exc
        return make_view(supplied["features"], row["nonitem"], item_history=row["item"])


__all__ = ["DEFAULT_MAX_MATCHES", "INDEX_VERSION", "ShardReader", "VERSION", "write_shard"]
