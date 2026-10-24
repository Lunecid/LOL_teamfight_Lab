"""Verified, bounded reader for the full V preprocessing surface.

The accepted shared-history bank is the only history source used here.  A
match member is decompressed once, nominal rows are resolved through the
accepted projection references, and the resulting view is converted to the
common semantic row without exposing a target.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np

from studies.v_input_projection_20261002.bank import RawProjectionLoader
from studies.v_model_fit_input_20261003.common import semantic_row
from studies.v_model_input_20261003.view import make_view, replacement_swap, swap_view
from studies.v_preparation_20261002.partition import (
    assigned_role,
    context_membership,
    inner_stop,
    train_fold,
)
from studies.v_shared_history_20261003.bank import loader_fingerprint
from studies.v_shared_history_production_20261003.store import ShardReader


VERSION = "v_full_preprocess_surface_20261003_v1"
PATCHES = ("15.14", "15.15", "15.16")
ROLES = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")
CONTEXTS = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
MEMBERSHIP_BITS = {"fit": 1, "stop": 2, "refit": 4, "dev": 8, "cal": 16, "target": 32}
SCOPE_NAMES = tuple((context, scope) for context in CONTEXTS for scope in ("FIT", "REFIT_TRAIN"))
_SCOPE_FILES = tuple(f"{context}_{scope}_nominal_rows.npy" for context, scope in SCOPE_NAMES)
_BINDING_FILES = ("MATCH_TABLE.npy", "V_ROWS.npy", "CONTEXT_MEMBERSHIP.npy", "MATCH_QUERIES.npy", "QUERY_TABLE.npy")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("identity is not JSON serialisable") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hex_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _key(value: Any) -> str:
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("canonical match key is not ASCII") from exc
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("canonical match key must be lowercase SHA256 hex64")
    return value


def _field(record: Any, name: str) -> Any:
    names = getattr(getattr(record, "dtype", None), "names", None)
    if names is not None:
        if name not in names:
            raise ValueError(f"projection record is missing {name}")
        return record[name]
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"projection record is missing {name}")
        return record[name]
    try:
        return record[name]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"projection record is missing {name}") from exc


def _record_names(record: Any) -> tuple[str, ...]:
    names = getattr(getattr(record, "dtype", None), "names", None)
    if names is not None:
        return tuple(names)
    if isinstance(record, Mapping):
        return tuple(record)
    return ()


def _record_patch(record: Any) -> str:
    return _patch(_field(record, "patch") if "patch" in _record_names(record) else _field(record, "patch_role"))


def _inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a relative path")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its root") from exc
    return candidate


def _patch(value: Any) -> str:
    if isinstance(value, str):
        if value not in PATCHES:
            raise ValueError(f"unknown patch: {value}")
        return value
    index = _integer(value, "patch_role")
    try:
        return PATCHES[index]
    except IndexError as exc:
        raise ValueError(f"unknown patch role: {index}") from exc


def _stored_role(value: Any) -> str:
    if isinstance(value, str):
        if value not in ROLES:
            raise ValueError(f"unknown stored role: {value}")
        return value
    index = _integer(value, "role")
    try:
        return ROLES[index]
    except IndexError as exc:
        raise ValueError(f"unknown stored role: {index}") from exc


def _match_role(record: Any) -> tuple[str, str, bool, int | None]:
    key = _key(_field(record, "key"))
    patch = _record_patch(record)
    role = assigned_role(patch, key)
    stored = _stored_role(_field(record, "role"))
    if stored != role:
        raise ValueError("stored match role differs from canonical assigned_role")
    fold: int | None = None
    if patch == "15.14":
        fold = train_fold(key)
        if "fold" in _record_names(record):
            if _integer(_field(record, "fold"), "fold", nonnegative=False) != fold:
                raise ValueError("stored train fold differs from canonical train_fold")
        if "stop" in _record_names(record):
            if bool(_field(record, "stop")) != inner_stop(key):
                raise ValueError("stored STOP flag differs from canonical inner_stop")
    return key, role, patch == "15.14", fold


def _normal_range(match_range: Sequence[int] | None, n_matches: int) -> tuple[int, int]:
    if match_range is None:
        return 0, int(n_matches)
    if isinstance(match_range, (str, bytes)) or len(match_range) != 2:
        raise ValueError("match_range must be a (first, last) pair")
    first = _integer(match_range[0], "match_range.first")
    last = _integer(match_range[1], "match_range.last")
    if first > last or last > n_matches:
        raise ValueError("match_range is outside the projection")
    return first, last


def _scope_fold(context: str) -> int | None:
    if context == "final":
        return None
    if context not in CONTEXTS:
        raise ValueError(f"unknown scope context: {context}")
    return int(context[-1])


def _projection_bindings(projection_root: Path) -> dict[str, str]:
    manifest = _read_json(projection_root / "RUN_MANIFEST.json", "projection manifest")
    if manifest.get("complete") is not True:
        raise ValueError("projection manifest is incomplete")
    saved = manifest.get("saved_files")
    if not isinstance(saved, Mapping):
        raise ValueError("projection manifest has no saved file bindings")
    result: dict[str, str] = {}
    for name in _BINDING_FILES:
        record = saved.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"projection manifest is missing {name}")
        expected = _hex_digest(record.get("sha256"), f"projection {name} hash")
        actual = _sha256(projection_root / name)
        if actual != expected:
            raise ValueError(f"projection file hash differs: {name}")
        result[name] = actual
    return result


def _validate_partition(
    matches: Any,
    contexts: Any,
    *,
    require_nominal: bool = False,
) -> tuple[np.ndarray, list[str]]:
    if np.shape(contexts) != (6, len(matches)):
        raise ValueError("context membership has the wrong shape")
    match_keys: list[str] = []
    for index, match in enumerate(matches):
        key, role, is_train_patch, fold = _match_role(match)
        match_keys.append(key)
        patch = _record_patch(match)
        for context_index in range(6):
            expected = context_membership(patch, key, _scope_fold(CONTEXTS[context_index]))
            mask = sum(MEMBERSHIP_BITS[name] for name, enabled in expected.items() if enabled)
            if int(contexts[context_index, index]) != mask:
                raise ValueError(f"context membership differs for match {index}, {CONTEXTS[context_index]}")
        if require_nominal and is_train_patch and role != "TRAIN":  # pragma: no cover - role check above is authoritative
            raise ValueError("train patch has a non-TRAIN canonical role")
        if fold is not None and not is_train_patch:  # pragma: no cover - defensive branch
            raise ValueError("non-training patch unexpectedly has a train fold")
    return np.asarray(match_keys, dtype=object), match_keys


def build_scope_plan(projection_root: str | Path, census_root: str | Path) -> dict[str, Any]:
    """Recompute and hash-check all twelve accepted nominal preprocessing scopes."""

    projection_root = Path(projection_root)
    census_root = Path(census_root)
    census_path = census_root / "SCOPE_CENSUS.json"
    census = _read_json(census_path, "scope census")
    if census.get("status") != "passed" or census.get("full_nominal_scope_census") is not True:
        raise ValueError("scope census is not an accepted full census")
    projection_bindings = _projection_bindings(projection_root)
    upstream = census.get("upstream_bindings")
    if not isinstance(upstream, Mapping):
        raise ValueError("scope census has no upstream bindings")
    for name, expected in upstream.items():
        if name not in _BINDING_FILES:
            continue
        if _hex_digest(expected, f"census binding {name}") != projection_bindings[name]:
            raise ValueError(f"scope census binding differs: {name}")

    try:
        matches = np.load(projection_root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        nominal = np.load(projection_root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
        contexts = np.load(projection_root / "CONTEXT_MEMBERSHIP.npy", mmap_mode="r", allow_pickle=False)
        starts = np.load(projection_root / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
        queries = np.load(projection_root / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
        if census.get("matches") != len(matches) or census.get("nominal_rows") != len(nominal) or census.get("queries") != len(queries):
            raise ValueError("scope census counts differ from projection")
        if len(starts) != len(matches):
            raise ValueError("match/query ownership counts differ")
        _, match_keys = _validate_partition(matches, contexts, require_nominal=True)
        match_refs = np.asarray(nominal["match_index"], dtype=np.int64)
        if np.any(match_refs < 0) or np.any(match_refs >= len(matches)):
            raise ValueError("nominal row references an invalid match")
        if not np.array_equal(np.bincount(match_refs, minlength=len(matches)), matches["n_nominal_rows"]):
            raise ValueError("nominal per-match row coverage differs")
        for match_index, match in enumerate(matches):
            first = _integer(_field(match, "first_nominal_row"), "first_nominal_row")
            count = _integer(_field(match, "n_nominal_rows"), "n_nominal_rows")
            rows = nominal[first:first + count]
            if len(rows) != count or not np.all(rows["match_index"] == match_index):
                raise ValueError(f"nominal ownership differs for match {match_index}")
            if len(rows) and not np.array_equal(rows["nominal_index"], np.arange(count, dtype=rows["nominal_index"].dtype)):
                raise ValueError(f"nominal local indices differ for match {match_index}")
            start = _integer(starts[match_index]["first_query"], "first_query")
            n_queries = _integer(starts[match_index]["n_queries"], "n_queries")
            if start < 0 or start + n_queries > len(queries):
                raise ValueError(f"query ownership is outside QUERY_TABLE for match {match_index}")
            supported = np.asarray(rows["supported"], dtype=bool)
            refs = np.asarray(rows["query_ref"], dtype=np.int64)
            if np.any((~supported) & (refs >= 0)):
                raise ValueError(f"unsupported nominal row retains a query reference for match {match_index}")
            selected = refs[supported]
            if np.any(selected < 0) or np.any(selected >= n_queries):
                raise ValueError(f"nominal query reference is outside its match for match {match_index}")
            if len(selected):
                source = queries[start + selected]
                if not np.all(source["match_index"] == match_index) or not np.array_equal(source["query_ms"], rows["actual_ms"][supported]):
                    raise ValueError(f"nominal query timing differs for match {match_index}")
        role_codes = np.asarray(matches["role"][match_refs], dtype=np.int64)
        train_nominal_indices = np.flatnonzero(role_codes == 0).astype(np.int64, copy=False)

        census_scopes = census.get("scopes")
        if not isinstance(census_scopes, list):
            raise ValueError("scope census has no scope records")
        census_by_scope = {(item.get("context"), item.get("scope")): item for item in census_scopes if isinstance(item, Mapping)}
        records: list[dict[str, Any]] = []
        for context_index, context in enumerate(CONTEXTS):
            fold = _scope_fold(context)
            excluded = [
                match_keys[index]
                for index, match in enumerate(matches)
                if _stored_role(_field(match, "role")) == "TRAIN" and fold is not None and train_fold(match_keys[index]) == fold
            ]
            for scope, bit in (("FIT", MEMBERSHIP_BITS["fit"]), ("REFIT_TRAIN", MEMBERSHIP_BITS["refit"])):
                authorized = (np.asarray(contexts[context_index], dtype=np.uint8) & bit) != 0
                selected = np.flatnonzero(authorized[match_refs]).astype(np.int64, copy=False)
                selected_matches = np.flatnonzero(authorized).astype(np.int64, copy=False)
                if scope == "FIT" and np.any(np.asarray(matches["stop"][match_refs[selected]], dtype=bool)):
                    raise ValueError(f"STOP rows entered {context}:{scope}")
                if np.any(np.asarray(matches["role"][match_refs[selected]], dtype=np.int64) != 0):
                    raise ValueError(f"non-TRAIN rows entered {context}:{scope}")
                if fold is not None and len(selected_matches) and np.any(np.asarray(matches["fold"][selected_matches], dtype=np.int64) == fold):
                    raise ValueError(f"OOF held-out matches entered {context}:{scope}")
                name = f"{context}_{scope}_nominal_rows.npy"
                path = census_root / name
                expected_record = census_by_scope.get((context, scope))
                if expected_record is None:
                    raise ValueError(f"scope census record is missing: {context}:{scope}")
                if _sha256(path) != _hex_digest(expected_record.get("rows_sha256"), f"{name} hash"):
                    raise ValueError(f"scope census file hash differs: {name}")
                accepted = np.load(path, mmap_mode="r", allow_pickle=False)
                if accepted.ndim != 1 or not np.array_equal(np.asarray(accepted, dtype=np.int64), selected):
                    raise ValueError(f"derived scope indices differ: {context}:{scope}")
                records.append({
                    "context": context,
                    "scope": scope,
                    "nominal_indices": selected,
                    "local_indices": np.asarray(nominal["nominal_index"][selected], dtype=np.int64),
                    "match_indices": np.asarray(nominal["match_index"][selected], dtype=np.int64),
                    "excluded_game_keys": tuple(excluded),
                    "matches": selected_matches,
                    "rows": int(len(selected)),
                })
        census_bindings = {"SCOPE_CENSUS.json": _sha256(census_path)}
        census_bindings.update({name: _sha256(census_root / name) for name in _SCOPE_FILES})
        return {
            "version": VERSION,
            "train_nominal_indices": train_nominal_indices,
            "scopes": records,
            "source_bindings": {"projection": projection_bindings, "census": census_bindings},
            "counts": {"matches": int(len(matches)), "nominal_rows": int(len(nominal)), "queries": int(len(queries))},
        }
    finally:
        for value in (locals().get("matches"), locals().get("nominal"), locals().get("contexts"), locals().get("starts"), locals().get("queries")):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


class BankSurface:
    """Read accepted shared-history rows with bounded per-match memory."""

    def __init__(
        self,
        bank_root: str | Path,
        projection_root: str | Path,
        seed: int = 20261001,
        *,
        verify_projection_hashes: bool = True,
    ) -> None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.bank_root = Path(bank_root).resolve()
        self.projection_root = Path(projection_root).resolve()
        self.seed = int(seed)
        self._closed = False
        self._reader: ShardReader | None = None
        self._reader_directory: Path | None = None
        self._receipt_by_match: dict[int, tuple[Path, dict[str, Any]]] = {}
        self.acceptance = _read_json(self.bank_root / "ACCEPTANCE.json", "bank acceptance")
        self.plan = _read_json(self.bank_root / "PLAN.json", "bank plan")
        if self.acceptance.get("complete") is not True or self.acceptance.get("full_bank_produced") is not True:
            raise ValueError("bank acceptance is incomplete or not full")
        identity = _hex_digest(self.acceptance.get("identity_sha256"), "bank acceptance identity")
        if identity != _fingerprint(self.plan):
            raise ValueError("bank acceptance identity differs from PLAN.json")
        if self.plan.get("match_start") != 0:
            raise ValueError("bank plan must start at match zero")
        try:
            self._loader = RawProjectionLoader(self.projection_root, verify_hashes=bool(verify_projection_hashes))
            if self.plan.get("loader_manifest_sha256") != loader_fingerprint(self._loader):
                raise ValueError("bank plan loader identity differs from projection")
            self._validate_projection_counts()
            self._validate_receipts()
        except BaseException:
            loader = getattr(self, "_loader", None)
            if loader is not None:
                loader.close()
            raise

    @property
    def loader(self) -> RawProjectionLoader:
        if self._closed:
            raise RuntimeError("bank surface is closed")
        return self._loader

    def _validate_projection_counts(self) -> None:
        matches = int(len(self._loader.matches))
        queries = int(len(self._loader.queries))
        if self.acceptance.get("matches") != matches or self.acceptance.get("queries") != queries:
            raise ValueError("bank/projection coverage counts differ")
        if self.plan.get("match_end") != matches or self.plan.get("query_count") != queries:
            raise ValueError("bank plan coverage counts differ")
        if len(self._loader.match_queries) != matches:
            raise ValueError("projection match/query table counts differ")

    def _validate_receipts(self) -> None:
        receipts = self.acceptance.get("shards")
        if not isinstance(receipts, list) or not receipts:
            raise ValueError("bank acceptance has no shard receipts")
        expected_first = 0
        total_queries = 0
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise ValueError("bank shard receipt is malformed")
            span = receipt.get("match_range")
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("bank shard receipt range is malformed")
            first, last = _integer(span[0], "receipt.match_range.first"), _integer(span[1], "receipt.match_range.last")
            if first != expected_first or last <= first:
                raise ValueError("bank shard receipts have a gap or overlap")
            if receipt.get("identity_sha256") != self.acceptance.get("identity_sha256") or receipt.get("all_queries_reference_and_reload_equal") is not True:
                raise ValueError("bank shard receipt identity/equivalence differs")
            manifest_hash = _hex_digest(receipt.get("manifest_sha256"), "receipt manifest hash")
            receipt_path = self.bank_root / f"accepted_{first:06d}.json"
            if not receipt_path.is_file() or _read_json(receipt_path, "accepted receipt") != dict(receipt):
                raise ValueError(f"accepted receipt differs: {receipt_path.name}")
            directory = _inside(self.bank_root, receipt.get("directory"), "receipt directory")
            manifest_path = directory / "MANIFEST.json"
            if _sha256(manifest_path) != manifest_hash:
                raise ValueError("receipt manifest hash differs")
            manifest = _read_json(manifest_path, "shard manifest")
            if manifest.get("complete") is not True or manifest.get("storage_complete") is not True or manifest.get("identity") != self.plan:
                raise ValueError("shard manifest identity/completion differs")
            if manifest.get("binary") != "shard.bin" or manifest.get("index") != "index.json":
                raise ValueError("shard file names differ")
            expected_queries = int(np.sum(self._loader.match_queries["n_queries"][first:last], dtype=np.int64))
            if receipt.get("queries") != expected_queries or manifest.get("n_matches") != last - first or manifest.get("n_queries") != expected_queries:
                raise ValueError("shard receipt coverage differs from projection")
            if manifest.get("match_indices") != list(range(first, last)):
                raise ValueError("shard manifest match coverage differs")
            self._receipt_by_match.update({index: (directory, dict(receipt)) for index in range(first, last)})
            total_queries += expected_queries
            expected_first = last
        if expected_first != len(self._loader.matches) or total_queries != len(self._loader.queries):
            raise ValueError("bank shard coverage is incomplete")

    def _reader_for(self, match_index: int) -> ShardReader:
        try:
            directory, _receipt = self._receipt_by_match[match_index]
        except KeyError as exc:
            raise ValueError(f"no accepted shard covers match {match_index}") from exc
        if self._reader is None or self._reader_directory != directory:
            self._reader = ShardReader(directory, identity=self.plan)
            self._reader_directory = directory
        return self._reader

    def _records_for_match(self, match_index: int) -> tuple[Any, str, str, bool, int | None, list[dict[str, Any]]]:
        match = self.loader.matches[match_index]
        key, role, is_train_patch, fold = _match_role(match)
        patch = _record_patch(match)
        records = self._reader_for(match_index).match_records(match_index)
        count = _integer(self.loader.match_queries[match_index]["n_queries"], "n_queries")
        first_query = _integer(self.loader.match_queries[match_index]["first_query"], "first_query")
        expected_ids = list(range(first_query, first_query + count))
        actual_ids = [
            _integer(record.get("query_id"), "shard query_id")
            for record in records
            if isinstance(record, Mapping)
        ]
        if len(actual_ids) != len(records) or actual_ids != expected_ids:
            raise ValueError(f"shard query records differ for match {match_index}")
        return match, key, role, is_train_patch, fold, records

    def iter_nominal_rows(
        self,
        match_range: Sequence[int] | None = None,
        *,
        train_only: bool = True,
    ) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        """Yield supported nominal rows in global nominal-index order.

        The iterator uses only the accepted shard record for a match.  It does
        not call the raw-history reader or reconstruct history prefixes.
        """

        if self._closed:
            raise RuntimeError("bank surface is closed")
        if not isinstance(train_only, bool):
            raise TypeError("train_only must be a boolean")
        first_match, last_match = _normal_range(match_range, len(self.loader.matches))
        previous_nominal = -1
        for match_index in range(first_match, last_match):
            match = self.loader.matches[match_index]
            key, role, _is_train_patch, _fold = _match_role(match)
            if train_only and role != "TRAIN":
                continue
            first_nominal = _integer(_field(match, "first_nominal_row"), "first_nominal_row")
            n_nominal = _integer(_field(match, "n_nominal_rows"), "n_nominal_rows")
            rows = self.loader.nominal[first_nominal:first_nominal + n_nominal]
            if len(rows) != n_nominal or (len(rows) and not np.all(rows["match_index"] == match_index)):
                raise ValueError(f"nominal ownership differs for match {match_index}")
            supported = np.asarray(rows["supported"], dtype=bool)
            if not np.any(supported):
                continue
            _match, key, role, _is_train_patch, _fold, records = self._records_for_match(match_index)
            record_by_query = {int(record["query_id"]): record for record in records}
            first_query = _integer(self.loader.match_queries[match_index]["first_query"], "first_query")
            for local_index, nominal in enumerate(rows):
                if not bool(nominal["supported"]):
                    if int(nominal["query_ref"]) >= 0:
                        raise ValueError(f"unsupported nominal row retains a query reference at {first_nominal + local_index}")
                    continue
                query_ref = _integer(nominal["query_ref"], "nominal query_ref")
                query_index = first_query + query_ref
                if not 0 <= query_index < len(self.loader.queries):
                    raise ValueError(f"nominal query reference is outside QUERY_TABLE at {first_nominal + local_index}")
                query = self.loader.queries[query_index]
                query_ms = _integer(query["query_ms"], "query_ms")
                if int(query["match_index"]) != match_index or query_ms != _integer(nominal["actual_ms"], "actual_ms"):
                    raise ValueError(f"nominal query identity differs at {first_nominal + local_index}")
                try:
                    record = record_by_query[query_index]
                except KeyError as exc:
                    raise ValueError(f"shard is missing query {query_index} for match {match_index}") from exc
                if _integer(record.get("query_ms"), "shard query_ms") != query_ms:
                    raise ValueError(f"shard/query timestamp differs for query {query_index}")
                supplied = self.loader.query_input(query_index, include_histories=False)
                audit = supplied.get("audit") if isinstance(supplied, Mapping) else None
                if not isinstance(audit, Mapping) or int(audit.get("match_index", -1)) != match_index or int(audit.get("query_ms", -1)) != query_ms:
                    raise ValueError(f"projection query identity differs for query {query_index}")
                nonitem = record.get("nonitem", record.get("nonitem_history"))
                item = record.get("item", record.get("item_history"))
                if nonitem is None or item is None:
                    raise ValueError(f"shard history record is incomplete for query {query_index}")
                view = make_view(supplied["features"], nonitem, item_history=item)
                swapped = bool(role == "TRAIN" and replacement_swap(key, query_ms, self.seed))
                if swapped:
                    view = swap_view(view)
                row = semantic_row(view)
                nominal_index = first_nominal + local_index
                if nominal_index <= previous_nominal:
                    raise ValueError("nominal rows are not globally sorted")
                previous_nominal = nominal_index
                metadata = {
                    "match_index": match_index,
                    "nominal_index": nominal_index,
                    "local_nominal_index": local_index,
                    "query_index": query_index,
                    "query_ms": query_ms,
                    "query_ref": query_ref,
                    "game_key": key,
                    "patch": _record_patch(match),
                    "role": role,
                    "swapped": swapped,
                }
                for name in ("bucket_id", "bucket_lo_ms", "bucket_hi_ms", "nominal_ms", "frame_index", "event_limit"):
                    if name in getattr(rows.dtype, "names", ()):
                        metadata[name] = int(nominal[name])
                yield nominal_index, row, metadata

    @contextmanager
    def open_context(self) -> Iterator["BankSurface"]:
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._reader = None
            self._reader_directory = None
            self._loader.close()

    def __enter__(self) -> "BankSurface":
        if self._closed:
            raise RuntimeError("bank surface is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["BankSurface", "CONTEXTS", "ROLES", "SCOPE_NAMES", "VERSION", "build_scope_plan"]
