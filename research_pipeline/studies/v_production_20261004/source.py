"""Replayable split sources backed by the bounded production input cache.

``InputCache`` stores rows in source order and in shards of at most 1024 rows.
This adapter applies the locked membership/role split to those shards and
rebatches the surviving rows so the runner sees full 1024-row batches.  The
cache is verified once when it is opened; iteration only reads its typed
arrays and immutable shards.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import sparse

from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch
from studies.v_production_20261004.cache import (
    LANE_SHAPES,
    META_NAMES,
    CacheBatch,
    InputCache,
    MAX_BATCH_ROWS,
)


VERSION = "v_production_batch_source_20261004_v1"
MEMBERSHIP_BITS = {"fit": 1, "stop": 2, "refit": 4, "dev": 8, "cal": 16}
SPLIT_MASK = sum(MEMBERSHIP_BITS.values())  # bit 32 is the OOF target fold, never a served split
SPLIT_ROLES = {
    "fit": "TRAIN",
    "stop": "TRAIN",
    "refit": "TRAIN",
    "dev": "V_DEV",
    "cal": "V_CAL",
}
SPLIT_ALIASES = {"development": "dev", "calibration": "cal", "reload": "dev", "gate": "gate"}
ROLE_ORDER = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")
FAMILIES = {"logistic", "lightgbm", "plain_mlp", "shared_slot_mlp", "team_set_mlp"}


def _decode(value: Any) -> str:
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    return str(value)


def _decode_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("metadata text fields must be one-dimensional")
    return np.asarray([_decode(item) for item in array], dtype=object)


def _row_indices(indices: Sequence[int] | np.ndarray, rows: int) -> np.ndarray:
    value = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(value < 0) or np.any(value >= rows):
        raise IndexError("row slice is outside the cache batch")
    return value


def _slice_bag(bag: BagInput, indices: np.ndarray, source_rows: int) -> BagInput:
    """Slice a dense or flattened :class:`BagInput` without losing offsets."""

    raw_indices = np.asarray(bag.indices)
    raw_counts = np.asarray(bag.counts)
    offsets = None if bag.offsets is None else np.asarray(bag.offsets)
    observed = None if bag.observed_count is None else np.asarray(bag.observed_count)
    if offsets is None:
        if raw_indices.ndim == 0 or raw_indices.shape[0] != source_rows:
            raise ValueError("dense bag indices do not align with the batch")
        new_indices = raw_indices[indices].copy()
        new_counts = raw_counts[indices].copy()
        new_observed = None if observed is None else observed[indices].copy()
        return BagInput(new_indices, new_counts, None, new_observed)
    if offsets.ndim != 2 or offsets.shape[0] != source_rows or offsets.shape[1] < 1:
        raise ValueError("ragged bag offsets do not align with the batch")
    if raw_indices.ndim != 1 or raw_counts.ndim != 1 or raw_indices.shape != raw_counts.shape:
        raise ValueError("ragged bag indices/counts must be aligned vectors")
    if observed is not None and (observed.ndim == 0 or observed.shape[0] != source_rows):
        raise ValueError("ragged bag observed counts do not align with the batch")
    selected_offsets = offsets[indices]
    new_offsets = np.empty_like(selected_offsets)
    pieces_indices: list[np.ndarray] = []
    pieces_counts: list[np.ndarray] = []
    cursor = 0
    for row_offset, target in zip(selected_offsets, new_offsets):
        starts = int(row_offset[0])
        ends = int(row_offset[-1])
        if starts < 0 or ends < starts or ends > len(raw_indices):
            raise ValueError("ragged bag offset is outside its value vector")
        target[:] = row_offset - starts + cursor
        pieces_indices.append(raw_indices[starts:ends])
        pieces_counts.append(raw_counts[starts:ends])
        cursor += ends - starts
    new_indices = np.concatenate(pieces_indices) if pieces_indices else raw_indices[:0].copy()
    new_counts = np.concatenate(pieces_counts) if pieces_counts else raw_counts[:0].copy()
    new_observed = None if observed is None else observed[indices].copy()
    return BagInput(new_indices, new_counts, new_offsets, new_observed)


def _concat_bags(parts: Sequence[BagInput]) -> BagInput:
    if not parts:
        raise ValueError("cannot concatenate an empty bag sequence")
    first = parts[0]
    offsets_present = first.offsets is not None
    if any((part.offsets is not None) != offsets_present for part in parts):
        raise ValueError("cannot concatenate dense and ragged bag representations")
    indices = np.concatenate([np.asarray(part.indices) for part in parts])
    counts = np.concatenate([np.asarray(part.counts) for part in parts])
    observed = None
    if any(part.observed_count is not None for part in parts):
        if any(part.observed_count is None for part in parts):
            raise ValueError("bag observed_count is missing from one concatenated part")
        observed = np.concatenate([np.asarray(part.observed_count) for part in parts], axis=0)
    if not offsets_present:
        return BagInput(indices, counts, None, observed)
    offset_parts: list[np.ndarray] = []
    cursor = 0
    for part in parts:
        value = np.asarray(part.offsets)
        offset_parts.append(value + cursor)
        cursor += len(np.asarray(part.indices))
    return BagInput(indices, counts, np.concatenate(offset_parts, axis=0), observed)


def _slice_structured(structured: StructuredBatch, indices: Sequence[int] | np.ndarray) -> StructuredBatch:
    """Return a row-safe copy of one structured cache batch."""

    source_rows = next(
        int(np.asarray(value).shape[0])
        for value in (structured.node_numeric, structured.team_numeric, structured.global_numeric, structured.dense_features)
        if value is not None
    )
    selected = _row_indices(indices, source_rows)

    def rows(value: Any) -> Any:
        return None if value is None else np.asarray(value)[selected].copy()

    def groups(values: Mapping[str, Any]) -> dict[str, Any]:
        return {str(name): np.asarray(value)[selected].copy() for name, value in values.items()}

    def bags(values: Mapping[str, BagInput]) -> dict[str, BagInput]:
        return {str(name): _slice_bag(BagInput.from_value(value), selected, source_rows) for name, value in values.items()}

    table = structured.table_csr
    sliced_table = None if table is None else table[selected].tocsr().copy()
    return replace(
        structured,
        node_numeric=rows(structured.node_numeric),
        team_numeric=rows(structured.team_numeric),
        global_numeric=rows(structured.global_numeric),
        player_team_ids=rows(structured.player_team_ids),
        side_flag=rows(structured.side_flag),
        current_item_supported=rows(structured.current_item_supported),
        node_categories=groups(structured.node_categories),
        team_categories=groups(structured.team_categories),
        global_categories=groups(structured.global_categories),
        node_bags=bags(structured.node_bags),
        team_bags=bags(structured.team_bags),
        global_bags=bags(structured.global_bags),
        dense_features=rows(structured.dense_features),
        table_csr=sliced_table,
    )


def _concat_structured(parts: Sequence[StructuredBatch]) -> StructuredBatch:
    if not parts:
        raise ValueError("cannot concatenate an empty structured batch sequence")
    first = parts[0]

    def concat(value_name: str) -> Any:
        values = [getattr(part, value_name) for part in parts]
        if any(value is None for value in values):
            if any(value is not None for value in values):
                raise ValueError(f"structured field {value_name} is missing from one part")
            return None
        return np.concatenate([np.asarray(value) for value in values], axis=0)

    def concat_groups(name: str) -> dict[str, Any]:
        groups = [getattr(part, name) for part in parts]
        names = set(groups[0])
        if any(set(group) != names for group in groups[1:]):
            raise ValueError(f"structured group keys differ for {name}")
        return {key: np.concatenate([np.asarray(group[key]) for group in groups], axis=0) for key in sorted(names)}

    def concat_bags(name: str) -> dict[str, BagInput]:
        groups = [getattr(part, name) for part in parts]
        names = set(groups[0])
        if any(set(group) != names for group in groups[1:]):
            raise ValueError(f"structured bag keys differ for {name}")
        return {key: _concat_bags([BagInput.from_value(group[key]) for group in groups]) for key in sorted(names)}

    tables = [part.table_csr for part in parts]
    table = None if all(value is None for value in tables) else sparse.vstack(tables, format="csr")
    return replace(
        first,
        node_numeric=concat("node_numeric"),
        team_numeric=concat("team_numeric"),
        global_numeric=concat("global_numeric"),
        player_team_ids=concat("player_team_ids"),
        side_flag=concat("side_flag"),
        current_item_supported=concat("current_item_supported"),
        node_categories=concat_groups("node_categories"),
        team_categories=concat_groups("team_categories"),
        global_categories=concat_groups("global_categories"),
        node_bags=concat_bags("node_bags"),
        team_bags=concat_bags("team_bags"),
        global_bags=concat_bags("global_bags"),
        dense_features=concat("dense_features"),
        table_csr=table,
    )


def _concat_batches(parts: Sequence[CacheBatch]) -> CacheBatch:
    if not parts:
        raise ValueError("cannot concatenate empty cache batches")
    if len(parts) == 1:
        return parts[0]
    metadata = {name: np.concatenate([np.asarray(part.metadata[name]) for part in parts]) for name in META_NAMES}
    return CacheBatch(structured=_concat_structured([part.structured for part in parts]), metadata=metadata)


def _split_mask(metadata: Mapping[str, Any], split: str) -> np.ndarray:
    key = SPLIT_ALIASES.get(str(split).lower(), str(split).lower())
    if key == "gate":
        # The full-corpus gate consumes the refit TRAIN membership.  This is
        # FIT+STOP for final and excludes the held-out OOF fold by construction.
        bit = MEMBERSHIP_BITS["refit"]
        expected_role = "TRAIN"
    elif key in MEMBERSHIP_BITS:
        bit = MEMBERSHIP_BITS[key]
        expected_role = SPLIT_ROLES[key]
    else:
        raise ValueError(f"unknown source split: {split!r}")
    membership = np.asarray(metadata["context_membership"], dtype=np.uint8)
    roles = np.asarray([_decode(value).upper() for value in metadata["role"]], dtype=object)
    selected = (membership & bit) != 0
    if np.any(selected & (roles != expected_role)):
        raise ValueError(f"membership bit for {key} is attached to a non-{expected_role} row")
    return selected


def _metadata_for_runner(metadata: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Decode fixed-width cache strings and mark labels as already supervised."""

    result: dict[str, Any] = {name: np.asarray(metadata[name]).copy() for name in META_NAMES}
    roles = tuple(_decode(value).upper() for value in metadata["role"])
    game_keys = tuple(_decode(value) for value in metadata["game_key"])
    patches = tuple(_decode(value) for value in metadata["patch"])
    result.update(
        {
            "role": roles,
            "roles": roles,
            "game_key": game_keys,
            "game_keys": game_keys,
            "patch": patches,
            "target_is_supervised": True,
            "split": str(split),
            "nominal_indices": result["nominal_index"],
            "query_indices": result["query_index"],
            "membership": result["context_membership"],
        }
    )
    return result


class CacheSource:
    """Replayable FIT/STOP/REFIT/DEV/CAL source over one or two caches.

    A cache path is verified once at construction by :class:`InputCache`.
    Supplying an already-open ``InputCache`` preserves that verification and
    avoids reopening or hashing shards during every epoch.
    """

    def __init__(
        self,
        cache: InputCache | str | Path | None = None,
        family: str | None = None,
        *,
        fit_cache: InputCache | str | Path | None = None,
        refit_cache: InputCache | str | Path | None = None,
        dimensions: Mapping[str, Any] | int | None = None,
        excluded_game_keys: Sequence[str] = (),
        verify: bool = True,
        expected_train_rows: int | None = None,
    ) -> None:
        if cache is not None and fit_cache is not None:
            raise ValueError("pass cache or fit_cache, not both")
        if cache is not None:
            fit_cache = cache
        if fit_cache is None and refit_cache is not None:
            fit_cache = refit_cache
        if fit_cache is None:
            raise ValueError("a FIT cache is required")
        self.family = None if family is None else str(family).lower()
        if self.family is not None and self.family not in FAMILIES:
            raise ValueError(f"unknown V model family: {family!r}")
        self._owned: list[InputCache] = []
        self.fit_cache = self._coerce_cache(fit_cache, verify=verify)
        self.refit_cache = self._coerce_cache(refit_cache, verify=verify) if refit_cache is not None else self.fit_cache
        self.excluded_game_keys = tuple(str(key) for key in excluded_game_keys)
        if len(set(self.excluded_game_keys)) != len(self.excluded_game_keys):
            raise ValueError("excluded_game_keys must be unique")
        self._excluded = frozenset(self.excluded_game_keys)
        self._check_exclusions()
        self.dimensions = self._dimensions(dimensions)
        self.row_counts = self._row_counts()
        self.counts = dict(self.row_counts)
        self.train_rows = self.row_counts["fit"] + self.row_counts["stop"]
        if expected_train_rows is not None and self.train_rows != int(expected_train_rows):
            raise ValueError(f"TRAIN FIT+STOP rows differ: {self.train_rows}/{int(expected_train_rows)}")

    def _check_exclusions(self) -> None:
        """Reject a cache in which an excluded OOF game has fit/STOP/REFIT/DEV/CAL membership.

        The held-out fold's own rows legitimately carry only the OOF target bit
        (partition.context_membership ``target``, value 32): they are scored for
        labels later and never enter any split this source serves.
        """

        if not self._excluded:
            return
        excluded = np.asarray([key.encode("ascii") for key in self._excluded], dtype="S64")
        for cache in {id(value): value for value in (self.fit_cache, self.refit_cache)}.values():
            metadata = cache.metadata
            active = (np.asarray(metadata["context_membership"], dtype=np.uint8) & SPLIT_MASK) != 0
            if np.any(active & np.isin(np.asarray(metadata["game_key"], dtype="S64"), excluded)):
                raise ValueError("an excluded OOF game has active split membership in the cache")

    def _coerce_cache(self, value: InputCache | str | Path, *, verify: bool) -> InputCache:
        if isinstance(value, InputCache):
            return value
        result = InputCache(value, verify=verify)
        self._owned.append(result)
        return result

    def _dimensions(self, supplied: Mapping[str, Any] | int | None) -> dict[str, Any]:
        if supplied is None:
            result: dict[str, Any] = {}
        elif isinstance(supplied, Mapping):
            result = dict(supplied)
        elif isinstance(supplied, (int, np.integer)) and not isinstance(supplied, (bool, np.bool_)):
            result = {"table_width": int(supplied)}
        else:
            raise TypeError("dimensions must be a mapping, table width, or None")
        requested_width = result.get("table_width")
        if requested_width is not None and int(requested_width) not in {
            int(self.fit_cache.table_width),
            int(self.refit_cache.table_width),
        }:
            raise ValueError("dimensions table_width differs from both bound caches")
        # FIT and REFIT vocabularies are fitted independently; adding STOP
        # rows can legitimately change the REFIT sparse width.  Each emitted
        # batch therefore keeps the width bound to the cache it came from.
        if "refit_table_width" in result and int(result["refit_table_width"]) != self.refit_cache.table_width:
            raise ValueError("dimensions refit_table_width differs from the REFIT cache")
        result.setdefault("table_width", int(self.fit_cache.table_width))
        result.setdefault("fit_table_width", int(self.fit_cache.table_width))
        result.setdefault("refit_table_width", int(self.refit_cache.table_width))
        result.setdefault("node_numeric_dim", LANE_SHAPES["node"][1])
        result.setdefault("team_numeric_dim", LANE_SHAPES["team"][1])
        result.setdefault("global_numeric_dim", LANE_SHAPES["global"][0])
        vocab = result.get("vocab_sizes", {})
        if not isinstance(vocab, Mapping):
            raise TypeError("dimensions.vocab_sizes must be a mapping")
        result["vocab_sizes"] = dict(vocab)
        return result

    @property
    def table_width(self) -> int:
        return int(self.fit_cache.table_width)

    def _row_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in (*MEMBERSHIP_BITS, "gate")}
        cache_names: dict[int, tuple[InputCache, set[str]]] = {}
        for cache, names in (
            (self.fit_cache, ("fit", "stop")),
            (self.refit_cache, ("refit", "dev", "cal")),
        ):
            key = id(cache)
            if key not in cache_names:
                cache_names[key] = (cache, set())
            cache_names[key][1].update(names)
        for cache, names in cache_names.values():
            metadata = cache.metadata
            membership = np.asarray(metadata["context_membership"], dtype=np.uint8)
            roles = np.asarray([_decode(value).upper() for value in metadata["role"]], dtype=object)
            for name in names:
                bit = MEMBERSHIP_BITS[name]
                expected = SPLIT_ROLES[name]
                wrong = ((membership & bit) != 0) & (roles != expected)
                if np.any(wrong):
                    raise ValueError(f"cache membership bit {name} has an invalid role")
                counts[name] = int(np.count_nonzero((membership & bit) != 0))
        gate_metadata = self.refit_cache.metadata
        gate_membership = np.asarray(gate_metadata["context_membership"], dtype=np.uint8)
        gate_roles = np.asarray([_decode(value).upper() for value in gate_metadata["role"]], dtype=object)
        gate_mask = (gate_membership & MEMBERSHIP_BITS["refit"]) != 0
        if np.any(gate_mask & (gate_roles != "TRAIN")):
            raise ValueError("refit membership contains a non-TRAIN row")
        counts["gate"] = int(np.count_nonzero(gate_mask))
        return counts

    def _cache_for(self, split: str) -> InputCache:
        key = SPLIT_ALIASES.get(str(split).lower(), str(split).lower())
        if key in {"fit", "stop"}:
            return self.fit_cache
        if key in {"gate", "refit", "dev", "cal"}:
            return self.refit_cache
        raise ValueError(f"unknown source split: {split!r}")

    def _family_inputs(self, structured: StructuredBatch) -> Any:
        if self.family == "plain_mlp":
            if structured.table_csr is None:
                raise ValueError("plain_mlp source requires a table CSR")
            return replace(structured, dense_features=np.asarray(structured.table_csr.toarray(), dtype=np.float32))
        if self.family in {"logistic", "lightgbm"}:
            return {"table_csr": structured.table_csr}
        return structured

    def iter_batches(self, split: str, epoch: int = 0, batch_size: int = MAX_BATCH_ROWS) -> Iterator[dict[str, Any]]:
        """Yield deterministic source-order batches for one locked split."""

        if isinstance(epoch, (bool, np.bool_)) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if isinstance(batch_size, (bool, np.bool_)) or not 0 < int(batch_size) <= MAX_BATCH_ROWS:
            raise ValueError("batch_size must be between 1 and 1024")
        batch_size = int(batch_size)
        key = SPLIT_ALIASES.get(str(split).lower(), str(split).lower())
        cache = self._cache_for(key)
        pending: list[CacheBatch] = []
        pending_rows = 0
        for source_batch in cache.iter_batches():
            keep = _split_mask(source_batch.metadata, key)
            if not np.any(keep):
                continue
            piece = _slice_cache_batch(source_batch, np.flatnonzero(keep))
            pending.append(piece)
            pending_rows += piece.rows
            while pending_rows >= batch_size:
                combined = _concat_batches(pending)
                selected = _slice_cache_batch(combined, np.arange(batch_size, dtype=np.int64))
                remainder = combined.rows - batch_size
                pending = [] if remainder == 0 else [_slice_cache_batch(combined, np.arange(batch_size, combined.rows, dtype=np.int64))]
                pending_rows = remainder
                yield self._envelope(selected, key)
        if pending_rows:
            combined = _concat_batches(pending)
            yield self._envelope(combined, key)

    def _envelope(self, batch: CacheBatch, split: str) -> dict[str, Any]:
        metadata = _metadata_for_runner(batch.metadata, split)
        if self._excluded and any(key in self._excluded for key in metadata["game_keys"]):
            raise ValueError(f"{split} batch contains a game excluded from this OOF context")
        labels = np.asarray(batch.metadata["label_w"], dtype=np.float32)
        return {"inputs": self._family_inputs(batch.structured), "target": labels, "metadata": metadata}

    def close(self) -> None:
        for cache in self._owned:
            cache.close()
        self._owned.clear()

    def __enter__(self) -> "CacheSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _slice_cache_batch(batch: CacheBatch, indices: Sequence[int] | np.ndarray) -> CacheBatch:
    selected = _row_indices(indices, batch.rows)
    metadata = {name: np.asarray(batch.metadata[name])[selected].copy() for name in META_NAMES}
    return CacheBatch(structured=_slice_structured(batch.structured, selected), metadata=metadata)


CacheBatchSource = CacheSource
InputCacheSource = CacheSource
ProductionBatchSource = CacheSource


__all__ = [
    "VERSION",
    "MEMBERSHIP_BITS",
    "CacheSource",
    "CacheBatchSource",
    "InputCacheSource",
    "ProductionBatchSource",
]
