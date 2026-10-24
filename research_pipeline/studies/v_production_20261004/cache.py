"""Streaming V input cache for the 2026-10-04 production gate.

The cache is deliberately batch oriented.  A fitted preprocessing bundle is
applied to at most :data:`MAX_BATCH_ROWS` semantic rows at a time; transformed
numeric arrays are stored in typed memmaps, while sparse CSR and the variable
length structured inputs are stored in immutable per-batch shards.  No caller
ever needs to materialise the full dense table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import time
from typing import Any, Iterator

import numpy as np
from scipy import sparse

from studies.execution_progress_20261003.progress import _atomic_write_json
from studies.v_model_fit_input_20261003.common import (
    BAG_LAYOUT,
    BAG_NAMESPACES,
    CATEGORY_LAYOUT,
    CATEGORY_NAMESPACES,
    NUMERIC_DIMS,
    transform_rows,
)


VERSION = "v_production_input_cache_20261004_v1"
MAX_BATCH_ROWS = 1024
LANE_SHAPES = {"node": (10, 102), "team": (2, 8), "global": (76,)}
ROLE_CODES = {name: index for index, name in enumerate(("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST"))}
META_DTYPES = {
    "nominal_index": np.dtype("<i8"),
    "match_index": np.dtype("<i4"),
    "query_index": np.dtype("<i8"),
    "query_ms": np.dtype("<i8"),
    "train_ordinal": np.dtype("<i8"),
    "game_key": np.dtype("S64"),
    "role": np.dtype("S8"),
    "patch": np.dtype("S5"),
    "swapped": np.dtype("?") ,
    "source_w": np.dtype("?") ,
    "label_w": np.dtype("?") ,
    "winning_team": np.dtype("<i2"),
    "context_membership": np.dtype("u1"),
}
META_NAMES = tuple(META_DTYPES)


def sha256_file(path: str | Path, *, block_size: int = 1 << 20) -> str:
    """Hash one source or cache file without reading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    """Return a stable digest for JSON-compatible provenance values."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_retry(source: Path, destination: Path) -> None:
    """Atomically publish a shard, retrying transient Windows sharing errors."""

    for attempt in range(41):
        try:
            os.replace(str(source), str(destination))
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32} or attempt == 40:
                raise
            time.sleep(0.05)


def _token(value: Any) -> int | str | None:
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        value = int(value)
        return value if value >= 0 else None
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value if value else None
    return None


def _vocab_index(vocabulary: Any, value: Any) -> int:
    token = _token(value)
    return int(vocabulary.mapping.get(token, 1)) if token is not None else 1


def independent_csr(rows: Sequence[Mapping[str, Any]], preprocessor: Any) -> sparse.csr_matrix:
    """Rebuild the table CSR independently of ``common.transform_rows``.

    This is intentionally a second implementation of the column layout used
    by the production cache.  It is used for the first-batch gate and focused
    tests; it is not used to produce the cache itself.
    """

    n_rows = len(rows)
    if not 0 < n_rows <= MAX_BATCH_ROWS:
        raise ValueError("independent CSR reference requires 1..1024 rows")
    numeric = {
        lane: np.asarray(
            preprocessor.numeric_map[lane].transform(np.stack([row["numeric"][lane] for row in rows])),
            dtype=np.float32,
        ).reshape(n_rows, -1)
        for lane in NUMERIC_DIMS
    }
    lanes = (("node", 10), ("team", 2), ("global", 1))
    width = sum(NUMERIC_DIMS[lane] * size for lane, size in lanes) + 10 + 1
    for lane, size in lanes:
        for group in CATEGORY_LAYOUT[lane]:
            width += size * (preprocessor.categorical_map[CATEGORY_NAMESPACES[group]].size + 2)
        for group in BAG_LAYOUT[lane]:
            width += size * (preprocessor.categorical_map[BAG_NAMESPACES[group]].size + 2) + size

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    def add_dense(array: Any, start: int) -> None:
        dense = np.asarray(array, dtype=np.float32).reshape(n_rows, -1)
        rr, cc = np.nonzero(dense)
        row_indices.extend(rr.tolist())
        column_indices.extend((cc + start).tolist())
        values.extend(dense[rr, cc].astype(np.float32, copy=False).tolist())

    cursor = 0
    for lane, size in lanes:
        add_dense(numeric[lane], cursor)
        cursor += NUMERIC_DIMS[lane] * size
    teams = np.stack([np.asarray(row["team_ids"]) for row in rows])
    if teams.shape != (n_rows, 10):
        raise ValueError("team_ids must have shape [rows,10]")
    add_dense((teams == 200).astype(np.float32), cursor)
    cursor += 10
    add_dense(np.asarray([float(row["focal_is_blue"]) for row in rows], dtype=np.float32).reshape(n_rows, 1), cursor)
    cursor += 1

    for lane, size in lanes:
        for group in CATEGORY_LAYOUT[lane]:
            vocabulary = preprocessor.categorical_map[CATEGORY_NAMESPACES[group]]
            n_categories = vocabulary.size + 2
            for row_index, row in enumerate(rows):
                raw = np.asarray(row["categories"][lane][group]).reshape(-1)
                if raw.shape != (size,):
                    raise ValueError(f"category group {lane}/{group} has the wrong shape")
                for slot, value in enumerate(raw):
                    encoded = _vocab_index(vocabulary, value)
                    if encoded:
                        row_indices.append(row_index)
                        column_indices.append(cursor + slot * n_categories + encoded)
                        values.append(1.0)
            cursor += size * n_categories

        for group in BAG_LAYOUT[lane]:
            vocabulary = preprocessor.categorical_map[BAG_NAMESPACES[group]]
            n_categories = vocabulary.size + 2
            for row_index, row in enumerate(rows):
                bags = row["bags"][lane][group]
                if len(bags) != size:
                    raise ValueError(f"bag group {lane}/{group} has the wrong number of slots")
                for slot, bag in enumerate(bags):
                    for raw_value, count in bag.items():
                        row_indices.append(row_index)
                        column_indices.append(cursor + slot * n_categories + _vocab_index(vocabulary, raw_value))
                        values.append(float(count))
            cursor += size * n_categories
            for row_index, row in enumerate(rows):
                for slot, bag in enumerate(row["bags"][lane][group]):
                    count = float(sum(bag.values()))
                    if count:
                        row_indices.append(row_index)
                        column_indices.append(cursor + slot)
                        values.append(count)
            cursor += size
    if cursor != width:
        raise ValueError("independent CSR layout cursor differs from width")
    result = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
        shape=(n_rows, width),
        dtype=np.float32,
    )
    result.sum_duplicates()
    result.sort_indices()
    return result


def transform_batch(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Sequence[Any] | np.ndarray],
    preprocessor: Any,
    *,
    check_reference: bool = False,
) -> "CacheBatch":
    """Transform one bounded semantic batch and attach typed provenance."""

    if not 0 < len(rows) <= MAX_BATCH_ROWS:
        raise ValueError("model input must be transformed in batches of at most 1024 rows")
    structured = transform_rows(rows, preprocessor)
    if check_reference:
        expected = independent_csr(rows, preprocessor)
        if structured.table_csr.shape != expected.shape or (structured.table_csr != expected).nnz:
            raise ValueError("production CSR differs from the independent reference")
    normalised = _normalise_metadata(metadata, len(rows))
    return CacheBatch(structured=structured, metadata=normalised)


@dataclass(frozen=True)
class CacheBatch:
    """One in-memory bounded input batch and its row provenance."""

    structured: Any
    metadata: Mapping[str, np.ndarray]

    @property
    def rows(self) -> int:
        return int(self.metadata["nominal_index"].shape[0])

    @property
    def table_csr(self) -> sparse.csr_matrix:
        return self.structured.table_csr

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.metadata["label_w"], dtype=np.float32)


def _normalise_metadata(metadata: Mapping[str, Sequence[Any] | np.ndarray], rows: int) -> dict[str, np.ndarray]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    missing = set(META_NAMES) - set(metadata)
    extra = set(metadata) - set(META_NAMES)
    if missing or extra:
        raise ValueError(f"metadata fields differ (missing={sorted(missing)}, extra={sorted(extra)})")
    result: dict[str, np.ndarray] = {}
    for name, dtype in META_DTYPES.items():
        array = np.asarray(metadata[name], dtype=dtype)
        if array.ndim != 1 or len(array) != rows:
            raise ValueError(f"metadata field {name!r} must have one value per row")
        result[name] = array.copy()
    if np.any(result["nominal_index"] < 0) or np.any(result["match_index"] < 0):
        raise ValueError("metadata indices must be nonnegative")
    if np.any(~np.isin(result["winning_team"], (100, 200))):
        raise ValueError("winning_team must use 100/200")
    if np.any(~np.isin(result["role"].astype("U8"), tuple(ROLE_CODES))):
        raise ValueError("metadata contains an unknown role")
    return result


def make_row_metadata(
    surface: Any,
    records: Sequence[Mapping[str, Any]],
    train_nominal_indices: Any,
    *,
    context_membership: Sequence[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build labels and source bindings from a ``BankSurface`` batch.

    ``source_w`` always remains the raw Blue-win outcome.  ``label_w`` follows
    the training replacement swap applied by ``BankSurface``.
    """

    n_rows = len(records)
    if context_membership is None:
        context_membership = np.zeros(n_rows, dtype=np.uint8)
    membership = np.asarray(context_membership, dtype=np.uint8)
    if membership.shape != (n_rows,):
        raise ValueError("context_membership must align with records")
    nominal_values = np.asarray([int(record["nominal_index"]) for record in records], dtype=np.int64)
    train_indices = np.asarray(train_nominal_indices, dtype=np.int64)
    positions = np.searchsorted(train_indices, nominal_values)
    is_train = np.asarray([str(record["role"]) == "TRAIN" for record in records], dtype=bool)
    train_ordinal = np.full(n_rows, -1, dtype=np.int64)
    valid_positions = is_train & (positions < len(train_indices))
    if np.any(valid_positions):
        exact = train_indices[positions[valid_positions]] == nominal_values[valid_positions]
        if not np.all(exact):
            raise ValueError("TRAIN surface nominal index is absent from TRAIN_NOMINAL.npy")
        train_ordinal[valid_positions] = positions[valid_positions]
    if np.any(is_train & (train_ordinal < 0)):
        raise ValueError("TRAIN rows require a valid numeric-cache ordinal")

    raw_w, label_w, winning_team, swapped = [], [], [], []
    for record in records:
        match_index = int(record["match_index"])
        value = bool(surface.loader.matches[match_index]["target_Blue_win"])
        did_swap = bool(record.get("swapped", False))
        raw_w.append(value)
        label_w.append(bool(value) ^ did_swap)
        winning_team.append(100 if value else 200)
        swapped.append(did_swap)
    return _normalise_metadata(
        {
            "nominal_index": nominal_values,
            "match_index": np.asarray([int(record["match_index"]) for record in records], dtype=np.int32),
            "query_index": np.asarray([int(record["query_index"]) for record in records], dtype=np.int64),
            "query_ms": np.asarray([int(record["query_ms"]) for record in records], dtype=np.int64),
            "train_ordinal": train_ordinal,
            "game_key": np.asarray([str(record["game_key"]).encode("ascii") for record in records], dtype="S64"),
            "role": np.asarray([str(record["role"]).encode("ascii") for record in records], dtype="S8"),
            "patch": np.asarray([str(record["patch"]).encode("ascii") for record in records], dtype="S5"),
            "swapped": np.asarray(swapped, dtype=bool),
            "source_w": np.asarray(raw_w, dtype=bool),
            "label_w": np.asarray(label_w, dtype=bool),
            "winning_team": np.asarray(winning_team, dtype=np.int16),
            "context_membership": membership,
        },
        n_rows,
    )


def _structured_payload(batch: CacheBatch) -> dict[str, np.ndarray]:
    structured = batch.structured
    payload: dict[str, np.ndarray] = {
        "player_team_ids": np.asarray(structured.player_team_ids, dtype=np.int16),
        "current_item_supported": np.asarray(structured.current_item_supported, dtype=bool),
    }
    for lane in ("node", "team", "global"):
        categories = getattr(structured, f"{lane}_categories")
        for group, values in categories.items():
            payload[f"category::{lane}::{group}"] = np.asarray(values, dtype=np.int32)
        bags = getattr(structured, f"{lane}_bags")
        for group, bag in bags.items():
            payload[f"bag::{lane}::{group}::indices"] = np.asarray(bag.indices, dtype=np.int64)
            payload[f"bag::{lane}::{group}::counts"] = np.asarray(bag.counts, dtype=np.float32)
            payload[f"bag::{lane}::{group}::offsets"] = np.asarray(bag.offsets, dtype=np.int64)
            payload[f"bag::{lane}::{group}::observed"] = np.asarray(bag.observed_count, dtype=np.float32)
    return payload


def _write_npz(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_structured(path: Path, numeric: Mapping[str, np.ndarray], csr: sparse.csr_matrix) -> Any:
    from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    categories: dict[str, dict[str, np.ndarray]] = {lane: {} for lane in ("node", "team", "global")}
    bags: dict[str, dict[str, Any]] = {lane: {} for lane in ("node", "team", "global")}
    for name, values in payload.items():
        parts = name.split("::")
        if len(parts) == 3 and parts[0] == "category":
            categories[parts[1]][parts[2]] = values
        elif len(parts) == 4 and parts[0] == "bag":
            lane, group, field = parts[1], parts[2], parts[3]
            bags[lane].setdefault(group, {})[field] = values
    bag_values = {
        lane: {
            group: BagInput(fields["indices"], fields["counts"], fields["offsets"], fields["observed"])
            for group, fields in groups.items()
        }
        for lane, groups in bags.items()
    }
    return StructuredBatch(
        node_numeric=numeric["node"],
        team_numeric=numeric["team"],
        global_numeric=numeric["global"],
        player_team_ids=payload["player_team_ids"],
        current_item_supported=payload["current_item_supported"],
        node_categories=categories["node"],
        team_categories=categories["team"],
        global_categories=categories["global"],
        node_bags=bag_values["node"],
        team_bags=bag_values["team"],
        global_bags=bag_values["global"],
        category_namespaces=CATEGORY_NAMESPACES,
        bag_namespaces=BAG_NAMESPACES,
        table_csr=csr,
    )


def estimate_cache_bytes(
    rows: int,
    *,
    csr_nnz_per_row: int = 4096,
    reserve_bytes: int = 8 << 30,
) -> dict[str, int]:
    """Return a conservative preflight estimate for one cache.

    The CSR estimate is explicit so callers can replace it with a measured
    pilot.  It is an estimate, never a permission to densify the table.
    """

    if isinstance(rows, (bool, np.bool_)) or int(rows) < 0:
        raise ValueError("rows must be a nonnegative integer")
    if int(csr_nnz_per_row) < 0:
        raise ValueError("csr_nnz_per_row must be nonnegative")
    n = int(rows)
    numeric = n * (10 * 102 + 2 * 8 + 76) * 4
    categories = n * (14 * 10 + 5 * 2) * 4
    metadata = n * sum(dtype.itemsize for dtype in META_DTYPES.values())
    csr = n * int(csr_nnz_per_row) * (4 + 4) + (n + 1) * 8
    overhead = max(1 << 20, n // MAX_BATCH_ROWS * (1 << 15))
    estimated = numeric + categories + metadata + csr + overhead
    return {
        "rows": n,
        "numeric_bytes": int(numeric),
        "category_bytes": int(categories),
        "metadata_bytes": int(metadata),
        "csr_bytes": int(csr),
        "overhead_bytes": int(overhead),
        "estimated_bytes": int(estimated),
        "reserve_bytes": int(reserve_bytes),
        "required_free_bytes": int(estimated + reserve_bytes),
    }


class CacheWriter:
    """Append immutable bounded batches to a production input cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        total_rows: int,
        table_width: int,
        plan: Mapping[str, Any],
        source_bindings: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.total_rows = int(total_rows)
        self.table_width = int(table_width)
        if self.total_rows < 0 or self.table_width <= 0:
            raise ValueError("invalid cache dimensions")
        self.plan = dict(plan)
        self.source_bindings = dict(source_bindings or {})
        self._batches: list[dict[str, Any]] = []
        self._closed = False
        self.root.mkdir(parents=True, exist_ok=resume)
        if (self.root / "MANIFEST.json").exists() and not resume:
            raise FileExistsError(f"cache already has a manifest: {self.root}")
        (self.root / "structured").mkdir(exist_ok=resume)
        (self.root / "csr").mkdir(exist_ok=resume)
        plan_path = self.root / "PLAN.json"
        if plan_path.exists():
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
            if existing != self.plan:
                raise ValueError("cache resume plan differs from the existing cache")
        else:
            _atomic_write_json(plan_path, self.plan)
        self._numeric = {}
        for lane, shape in LANE_SHAPES.items():
            path = self.root / f"{lane}_numeric.f32.npy"
            if path.exists():
                array = np.load(path, mmap_mode="r+")
                if tuple(array.shape) != (self.total_rows, *shape) or array.dtype != np.dtype("float32"):
                    raise ValueError(f"numeric cache shape/dtype differs for {lane}")
            else:
                array = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(self.total_rows, *shape))
            self._numeric[lane] = array
        self._metadata = {}
        for name, dtype in META_DTYPES.items():
            path = self.root / f"metadata_{name}.npy"
            if path.exists():
                array = np.load(path, mmap_mode="r+")
                if tuple(array.shape) != (self.total_rows,) or array.dtype != dtype:
                    raise ValueError(f"metadata cache shape/dtype differs for {name}")
            else:
                array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(self.total_rows,))
            self._metadata[name] = array
        batches_path = self.root / "BATCHES.json"
        if batches_path.exists():
            raw = json.loads(batches_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("BATCHES.json must be a list")
            self._batches = [dict(item) for item in raw]
            self._validate_receipts()
        self.completed = sum(int(item["rows"]) for item in self._batches)

    def _validate_receipts(self) -> None:
        current = 0
        for index, receipt in enumerate(self._batches):
            if int(receipt.get("batch_index", -1)) != index or int(receipt.get("row_start", -1)) != current:
                raise ValueError("cache batch receipts are not contiguous")
            rows = int(receipt.get("rows", -1))
            if not 0 < rows <= MAX_BATCH_ROWS:
                raise ValueError("cache batch receipt has invalid row count")
            for key in ("structured_file", "csr_file"):
                path = self.root / str(receipt.get(key, ""))
                if not path.is_file() or receipt.get(f"{key}_sha256") != sha256_file(path):
                    raise ValueError(f"cache batch receipt does not verify {key}")
            current += rows
        if current > self.total_rows:
            raise ValueError("cache receipts exceed the planned row count")

    @property
    def batches(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._batches)

    def append(self, batch: CacheBatch) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cache writer is closed")
        if not isinstance(batch, CacheBatch):
            raise TypeError("append expects CacheBatch")
        rows = batch.rows
        if not 0 < rows <= MAX_BATCH_ROWS:
            raise ValueError("cache batch must contain 1..1024 rows")
        if self.completed + rows > self.total_rows:
            raise ValueError("cache batch exceeds planned row count")
        metadata = _normalise_metadata(batch.metadata, rows)
        nominal = metadata["nominal_index"]
        if len(nominal) > 1 and np.any(np.diff(nominal) <= 0):
            raise ValueError("cache nominal indices must be strictly increasing within a batch")
        if self.completed and int(nominal[0]) <= int(self._metadata["nominal_index"][self.completed - 1]):
            raise ValueError("cache nominal indices must remain globally ordered")
        for lane, expected_shape in LANE_SHAPES.items():
            values = np.asarray(getattr(batch.structured, f"{lane}_numeric"), dtype=np.float32)
            if values.shape != (rows, *expected_shape):
                raise ValueError(f"{lane}_numeric has the wrong shape")
            self._numeric[lane][self.completed:self.completed + rows] = values
        for name in META_NAMES:
            self._metadata[name][self.completed:self.completed + rows] = metadata[name]
        batch_index = len(self._batches)
        structured_name = f"structured/batch_{batch_index:08d}.npz"
        csr_name = f"csr/batch_{batch_index:08d}.npz"
        structured_path = self.root / structured_name
        csr_path = self.root / csr_name
        if structured_path.exists() or csr_path.exists():
            raise FileExistsError("cache batch shard already exists; use resume with its receipt")
        _write_npz(structured_path, _structured_payload(batch))
        csr = batch.table_csr.tocsr().astype(np.float32, copy=False)
        if csr.shape != (rows, self.table_width):
            raise ValueError("CSR shape differs from the planned table width")
        _write_npz(csr_path, {"data": csr.data, "indices": csr.indices, "indptr": csr.indptr})
        receipt = {
            "version": VERSION,
            "batch_index": batch_index,
            "row_start": self.completed,
            "rows": rows,
            "nominal_first": int(nominal[0]),
            "nominal_last": int(nominal[-1]),
            "nnz": int(csr.nnz),
            "structured_file": structured_name,
            "csr_file": csr_name,
            "structured_file_sha256": sha256_file(structured_path),
            "csr_file_sha256": sha256_file(csr_path),
        }
        self._batches.append(receipt)
        _atomic_write_json(self.root / "BATCHES.json", self._batches)
        self.completed += rows
        for array in (*self._numeric.values(), *self._metadata.values()):
            array.flush()
        return dict(receipt)

    def finalize(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cache writer is closed")
        if self.completed != self.total_rows:
            raise ValueError(f"cache is incomplete: {self.completed}/{self.total_rows} rows")
        for array in (*self._numeric.values(), *self._metadata.values()):
            array.flush()
        manifest: dict[str, Any] = {
            "version": VERSION,
            "complete": True,
            "rows": self.total_rows,
            "table_width": self.table_width,
            "batch_rows_max": MAX_BATCH_ROWS,
            "batches": self._batches,
            "plan": self.plan,
            "plan_sha256": canonical_digest(self.plan),
            "source_bindings": self.source_bindings,
            "numeric_files": {lane: f"{lane}_numeric.f32.npy" for lane in LANE_SHAPES},
            "metadata_files": {name: f"metadata_{name}.npy" for name in META_NAMES},
            "semantic_layout": {
                "node_numeric": [10, 102],
                "team_numeric": [2, 8],
                "global_numeric": [76],
                "side_flag": "included as the final global numeric channel",
            },
        }
        if extra:
            manifest.update(dict(extra))
        _atomic_write_json(self.root / "MANIFEST.json", manifest)
        self._closed = True
        return manifest

    def close(self) -> None:
        if not self._closed:
            for array in (*self._numeric.values(), *self._metadata.values()):
                array.flush()
            self._closed = True

    def __enter__(self) -> "CacheWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class InputCache:
    """Read a completed cache one bounded batch at a time."""

    def __init__(self, root: str | Path, *, verify: bool = True) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError("cache is missing MANIFEST.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("version") != VERSION or self.manifest.get("complete") is not True:
            raise ValueError("cache manifest is incomplete or has the wrong version")
        self.rows = int(self.manifest["rows"])
        self.table_width = int(self.manifest["table_width"])
        self._numeric = {lane: np.load(self.root / self.manifest["numeric_files"][lane], mmap_mode="r") for lane in LANE_SHAPES}
        self._metadata = {name: np.load(self.root / self.manifest["metadata_files"][name], mmap_mode="r") for name in META_NAMES}
        self._batches = tuple(dict(item) for item in self.manifest.get("batches", ()))
        if verify:
            self._verify()

    def _verify(self) -> None:
        current = 0
        for index, receipt in enumerate(self._batches):
            if int(receipt.get("batch_index", -1)) != index or int(receipt.get("row_start", -1)) != current:
                raise ValueError("cache batch order is not contiguous")
            rows = int(receipt["rows"])
            if not 0 < rows <= MAX_BATCH_ROWS:
                raise ValueError("cache batch exceeds the bounded reader size")
            for key in ("structured_file", "csr_file"):
                path = self.root / str(receipt.get(key, ""))
                if not path.is_file() or receipt.get(f"{key}_sha256") != sha256_file(path):
                    raise ValueError(f"cache batch receipt does not verify {key}")
            current += rows
        if current != self.rows:
            raise ValueError("cache manifest row count differs from batch receipts")
        nominal = np.asarray(self._metadata["nominal_index"])
        if len(nominal) and np.any(np.diff(nominal) <= 0):
            raise ValueError("cache nominal rows are not globally ordered")

    @property
    def metadata(self) -> Mapping[str, np.ndarray]:
        return self._metadata

    def batch(self, batch_index: int) -> CacheBatch:
        index = int(batch_index)
        if not 0 <= index < len(self._batches):
            raise IndexError("cache batch index outside the manifest")
        receipt = self._batches[index]
        start, rows = int(receipt["row_start"]), int(receipt["rows"])
        numeric = {lane: np.asarray(self._numeric[lane][start:start + rows]) for lane in LANE_SHAPES}
        metadata = {name: np.asarray(self._metadata[name][start:start + rows]) for name in META_NAMES}
        with np.load(self.root / receipt["csr_file"], allow_pickle=False) as csr_archive:
            csr = sparse.csr_matrix(
                (csr_archive["data"], csr_archive["indices"], csr_archive["indptr"]),
                shape=(rows, self.table_width),
                dtype=np.float32,
            )
        structured = _load_structured(self.root / receipt["structured_file"], numeric, csr)
        return CacheBatch(structured=structured, metadata=metadata)

    def iter_batches(self) -> Iterator[CacheBatch]:
        for index in range(len(self._batches)):
            yield self.batch(index)

    def close(self) -> None:
        for value in (*self._numeric.values(), *self._metadata.values()):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "InputCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "VERSION",
    "MAX_BATCH_ROWS",
    "LANE_SHAPES",
    "META_NAMES",
    "CacheBatch",
    "CacheWriter",
    "InputCache",
    "canonical_digest",
    "estimate_cache_bytes",
    "independent_csr",
    "make_row_metadata",
    "sha256_file",
    "transform_batch",
]
