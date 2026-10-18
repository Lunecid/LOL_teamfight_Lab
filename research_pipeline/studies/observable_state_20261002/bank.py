"""Lossless sparse event payloads and factorized per-match shard views."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def pack_bank(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    packed = dict(bank)
    values = packed.pop("event_field_values")
    valid = packed.pop("event_field_validity")
    if values.shape != valid.shape or values.ndim != 2 or values.shape[1] > 32:
        raise ValueError("unsupported event payload width")
    row, col = np.nonzero(values)
    packed["event_field_row"] = row.astype(np.int32)
    packed["event_field_col"] = col.astype(np.uint8)
    selected = values[row, col]
    small = not selected.size or (selected.min() >= -(1 << 31) and selected.max() < (1 << 31))
    packed["event_field_nonzero"] = selected.astype(np.int32 if small else np.int64)
    mask = np.zeros(len(values), dtype=np.uint32)
    for index in range(valid.shape[1]):
        mask |= valid[:, index].astype(np.uint32) << index
    packed["event_field_valid_mask"] = mask
    packed["event_field_width"] = np.asarray(values.shape[1], dtype=np.int16)
    explicit = packed.pop("event_identity_explicit")
    identity = packed.pop("event_identity_code")
    packed["event_explicit_rows"] = np.flatnonzero(explicit).astype(np.int32)
    packed["event_explicit_identity"] = identity[explicit]
    offsets = packed.pop("event_assist_offsets")
    counts = np.diff(offsets)
    if np.any(counts < 0) or np.any(counts > 65535):
        raise ValueError("assist list length cannot be encoded")
    packed["event_assist_counts"] = counts.astype(np.uint16)
    packed["event_type_code"] = packed["event_type_code"].astype(np.uint8)
    if "event_string_codes" in packed:
        codes = packed.pop("event_string_codes")
        validity = packed.pop("event_string_validity")
        rows, cols = np.nonzero(codes)
        packed["event_string_row"] = rows.astype(np.int32)
        packed["event_string_col"] = cols.astype(np.uint8)
        packed["event_string_nonzero"] = codes[rows, cols].astype(np.uint8)
        packed["event_string_width"] = np.asarray(codes.shape[1], dtype=np.int16)
        mask = np.zeros(len(codes), dtype=np.uint16)
        for index in range(codes.shape[1]):
            mask |= validity[:, index].astype(np.uint16) << index
        packed["event_string_valid_mask"] = mask
    if "event_position_xy" in packed:
        xy = packed.pop("event_position_xy")
        validity = packed.pop("event_position_validity")
        rows = np.flatnonzero(validity)
        selected = xy[rows]
        reduced = selected.astype(np.float32)
        packed["event_position_rows"] = rows.astype(np.int32)
        packed["event_position_values"] = reduced if np.array_equal(reduced, selected, equal_nan=True) else selected
        packed["event_position_validity"] = validity
    return packed


def unpack_bank(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    bank = dict(packed)
    n = len(bank["event_timestamp"])
    width = int(bank.pop("event_field_width"))
    rows, cols, vals = (bank.pop(key) for key in
                        ("event_field_row", "event_field_col", "event_field_nonzero"))
    if len(rows) != len(cols) or len(cols) != len(vals):
        raise ValueError("sparse event payload length mismatch")
    if len(rows) and (rows.min() < 0 or rows.max() >= n or cols.max() >= width):
        raise ValueError("sparse event payload index out of range")
    if len(rows) > 1 and np.any(np.diff(rows.astype(np.int64) * width + cols) <= 0):
        raise ValueError("sparse payload indices are not unique and ordered")
    values = np.zeros((n, width), dtype=np.int64)
    values[rows, cols] = vals
    mask = bank.pop("event_field_valid_mask")
    bank["event_field_values"] = values
    bank["event_field_validity"] = ((mask[:, None] >> np.arange(width, dtype=np.uint32)) & 1).astype(bool)
    explicit_rows = bank.pop("event_explicit_rows")
    explicit_identity = bank.pop("event_explicit_identity")
    identity = np.zeros(n, dtype=np.uint64)
    explicit = np.zeros(n, dtype=bool)
    identity[explicit_rows] = explicit_identity
    explicit[explicit_rows] = True
    bank["event_identity_code"] = identity
    bank["event_identity_explicit"] = explicit
    counts = bank.pop("event_assist_counts")
    bank["event_assist_offsets"] = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    if int(bank["event_assist_offsets"][-1]) != len(bank["event_assist_values"]):
        raise ValueError("ragged assist payload mismatch")
    if "event_string_width" in bank:
        width = int(bank.pop("event_string_width"))
        rows, cols, vals = (bank.pop(key) for key in ("event_string_row", "event_string_col", "event_string_nonzero"))
        codes = np.zeros((n, width), dtype=np.uint8)
        codes[rows, cols] = vals
        mask = bank.pop("event_string_valid_mask")
        bank["event_string_codes"] = codes
        bank["event_string_validity"] = ((mask[:, None] >> np.arange(width, dtype=np.uint16)) & 1).astype(bool)
    if "event_position_rows" in bank:
        rows = bank.pop("event_position_rows")
        vals = bank.pop("event_position_values")
        validity = bank["event_position_validity"]
        xy = np.full((n, 2), np.nan, dtype=vals.dtype)
        xy[rows] = vals
        bank["event_position_xy"] = xy
        bank["event_position_validity"] = validity
    return bank


class ShardBuilder:
    """Concatenate each array independently; offsets preserve match boundaries."""

    def __init__(self):
        self.parts = defaultdict(list)
        self.offsets = defaultdict(lambda: [0])
        self.scalars = set()
        self.keys = None

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        if self.keys is None:
            self.keys = set(arrays)
        if set(arrays) != self.keys:
            raise ValueError("match bank schema changed inside shard")
        for key, value in arrays.items():
            value = np.asarray(value)
            if value.dtype.hasobject:
                raise ValueError("object payload is forbidden")
            if value.ndim == 0:
                self.scalars.add(key)
                value = value.reshape(1)
            if self.parts[key] and value.shape[1:] != self.parts[key][0].shape[1:]:
                raise ValueError("trailing bank shape changed")
            self.parts[key].append(value)
            self.offsets[key].append(self.offsets[key][-1] + len(value))

    def finish(self) -> tuple[dict[str, np.ndarray], dict]:
        arrays = {}
        for key in sorted(self.parts):
            arrays[key] = np.concatenate(self.parts[key], axis=0)
            arrays["offset_" + key] = np.asarray(self.offsets[key], dtype=np.int64)
        return arrays, {"scalar_fields": sorted(self.scalars), "match_fields": sorted(self.parts)}


def match_view(arrays: dict[str, np.ndarray], layout: dict, local_index: int) -> dict[str, np.ndarray]:
    result = {}
    scalars = set(layout["scalar_fields"])
    for key in layout["match_fields"]:
        offsets = arrays["offset_" + key]
        if not 0 <= local_index < len(offsets) - 1:
            raise IndexError("match index is outside this shard")
        value = arrays[key][int(offsets[local_index]):int(offsets[local_index + 1])]
        if key in scalars:
            if value.size != 1:
                raise ValueError("scalar match field has incorrect stored size")
            value = value.reshape(())
        result[key] = value
    return result
