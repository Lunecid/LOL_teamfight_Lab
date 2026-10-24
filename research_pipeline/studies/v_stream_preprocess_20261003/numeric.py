"""Bounded, exact numeric preprocessing for selected fit rows.

The accepted V-model preprocessor stores the fit state in
``NumericPreprocessor``.  This module supplies the same state through a
channel reader so a caller never has to materialize a full rows-by-channels
matrix.  A reader returns either one one-dimensional selected channel or an
iterable of one-dimensional chunks.  The caller supplies the exact number of
pooled values for each channel.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from studies.v_model_fit_input_20261003.preprocess import (
    FitScopeEvidence,
    NumericFieldSpec,
    NumericPreprocessor,
    _channel_names,
)
from studies.v_training_gate_20261003.guard import validate_scope_evidence


_INTERNAL_SCRATCH_BYTES_PER_VALUE = 32
_AXIS_WIDTH = {"row": 1, "player": 10, "team": 2}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coerce_spec(
    spec: NumericFieldSpec | Mapping[str, Any] | None,
    *,
    name: str | None,
    axis: str | None,
    channels: Sequence[str],
) -> NumericFieldSpec:
    supplied_channels = tuple(str(value) for value in channels)
    if spec is None:
        return NumericFieldSpec(
            name=name or "numeric",
            axis=axis or "row",
            channels=supplied_channels,
        )
    if isinstance(spec, NumericFieldSpec):
        if name is not None and spec.name != name:
            raise ValueError(f"field spec name {spec.name!r} does not match {name!r}")
        if axis is not None and spec.normalized_axis != NumericFieldSpec(spec.name, axis=axis).normalized_axis:
            raise ValueError("axis conflicts with field spec")
        if supplied_channels and tuple(spec.channels) != supplied_channels:
            raise ValueError("channels conflict with field spec")
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a NumericFieldSpec or mapping")
    spec_name = str(spec.get("name", name or "numeric"))
    if name is not None and spec_name != name:
        raise ValueError(f"field spec name {spec_name!r} does not match {name!r}")
    spec_axis = str(spec.get("axis", axis or "row"))
    if axis is not None and NumericFieldSpec(spec_name, axis=spec_axis).normalized_axis != NumericFieldSpec(spec_name, axis=axis).normalized_axis:
        raise ValueError("axis conflicts with field spec")
    raw_channels = tuple(str(value) for value in spec.get("channels", ()))
    if supplied_channels and raw_channels != supplied_channels:
        raise ValueError("channels conflict with field spec")
    return NumericFieldSpec(name=spec_name, axis=spec_axis, channels=raw_channels or supplied_channels)


def _as_float_vector(value: Any, *, channel: int) -> np.ndarray | None:
    """Return a one-dimensional float64 array, or ``None`` for a chunk stream."""

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"channel {channel} must be one-dimensional")
        if np.iscomplexobj(value):
            raise TypeError(f"channel {channel} must contain real numeric values")
        try:
            return np.array(value, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"channel {channel} must contain numeric values") from exc

    try:
        candidate = np.asarray(value)
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None and candidate.ndim == 1:
        if np.iscomplexobj(candidate):
            raise TypeError(f"channel {channel} must contain real numeric values")
        try:
            return np.array(candidate, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"channel {channel} must contain numeric values") from exc
    if candidate is not None and candidate.ndim == 0:
        if not isinstance(value, (str, bytes)):
            try:
                iter(value)
            except TypeError:
                pass
            else:
                return None
        raise ValueError(f"channel {channel} must be one-dimensional or an iterable of chunks")
    return None


def _read_channel(
    channel_reader: Callable[[int], Any],
    channel: int,
    *,
    count: int,
) -> np.ndarray:
    """Read one channel into one bounded float64 working vector."""

    try:
        raw = channel_reader(channel)
    except Exception:
        raise
    vector = _as_float_vector(raw, channel=channel)
    if vector is not None:
        if vector.size != count:
            raise ValueError(f"channel {channel} returned {vector.size} values; expected {count}")
        if np.any(np.isinf(vector)):
            raise ValueError(f"channel {channel} contains infinity; use NaN for missing values")
        return vector

    try:
        chunks = iter(raw)
    except TypeError as exc:
        raise TypeError(f"channel {channel} must contain numeric values") from exc
    vector = np.empty(count, dtype=np.float64)
    offset = 0
    for chunk in chunks:
        if isinstance(chunk, np.ndarray) and chunk.ndim != 1:
            raise ValueError(f"channel {channel} chunks must be one-dimensional")
        try:
            chunk_array = np.asarray(chunk)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"channel {channel} chunks must contain numeric values") from exc
        if chunk_array.ndim != 1:
            raise ValueError(f"channel {channel} chunks must be one-dimensional")
        if np.iscomplexobj(chunk_array):
            raise TypeError(f"channel {channel} must contain real numeric values")
        try:
            chunk_array = np.asarray(chunk_array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"channel {channel} chunks must contain numeric values") from exc
        end = offset + int(chunk_array.size)
        if end > count:
            raise ValueError(f"channel {channel} returned more than {count} values")
        vector[offset:end] = chunk_array
        offset = end
    if offset != count:
        raise ValueError(f"channel {channel} returned {offset} values; expected {count}")
    if np.any(np.isinf(vector)):
        raise ValueError(f"channel {channel} contains infinity; use NaN for missing values")
    return vector


def _channel_statistics(values: np.ndarray) -> tuple[float, float, float, bool, int]:
    """Compute the accepted fitter's exact finite/median-imputation policy."""

    finite = np.isfinite(values)
    fit_count = int(np.sum(finite, dtype=np.int64))
    if fit_count == 0:
        return 0.0, 0.0, 1.0, False, 0

    median = float(np.median(values[finite]))
    values[~finite] = median
    mean = float(np.mean(values, dtype=np.float64))
    scale = float(np.std(values, dtype=np.float64))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return median, mean, scale, True, fit_count


def fit_numeric_stream(
    channel_reader: Callable[[int], Any],
    *,
    count: int,
    channel_count: int | None = None,
    n_channels: int | None = None,
    spec: NumericFieldSpec | Mapping[str, Any] | None = None,
    name: str | None = None,
    axis: str | None = None,
    channels: Sequence[str] = (),
    evidence: FitScopeEvidence | Mapping[str, Any],
    expected_indices: Sequence[int],
    expected_game_keys: Sequence[str],
    expected_roles: Sequence[str],
    expected_source_hash: str,
    max_scratch_bytes: int,
) -> NumericPreprocessor:
    """Fit exact channel statistics from a bounded selected-row reader.

    ``count`` is the exact number of pooled values returned for every channel:
    it must equal the number of trusted fit rows times the semantic axis width
    (one for ``row``, ten for ``player``, and two for ``team``).  The declared
    scratch budget covers a conservative internal estimate of 32 bytes per
    pooled value for one channel; memory owned by the reader's callback is
    additional.  RSS accounting for the complete caller belongs to the
    run-level resource gate.
    """

    if not callable(channel_reader):
        raise TypeError("channel_reader must be callable")
    pooled_count = _positive_int(count, "count")
    if channel_count is not None and n_channels is not None and int(channel_count) != int(n_channels):
        raise ValueError("channel_count and n_channels must agree")
    resolved_channel_count = channel_count if channel_count is not None else n_channels
    field_spec = _coerce_spec(spec, name=name, axis=axis, channels=channels)
    if resolved_channel_count is None:
        if field_spec.channels:
            resolved_channel_count = len(field_spec.channels)
        else:
            raise ValueError("channel_count is required when spec declares no channels")
    resolved_channel_count = _positive_int(resolved_channel_count, "channel_count")
    if field_spec.channels and len(field_spec.channels) != resolved_channel_count:
        raise ValueError(
            f"field {field_spec.name!r} declares {len(field_spec.channels)} channels but has {resolved_channel_count}"
        )

    trusted = validate_scope_evidence(
        evidence,
        expected_game_keys=expected_game_keys,
        expected_roles=expected_roles,
        expected_indices=expected_indices,
        expected_source_hash=expected_source_hash,
    )
    expected_count = len(trusted.row_indices) * _AXIS_WIDTH[field_spec.normalized_axis]
    if pooled_count != expected_count:
        raise ValueError(
            f"count {pooled_count} does not match {field_spec.normalized_axis} fit rows: expected {expected_count}"
        )
    budget = _positive_int(max_scratch_bytes, "max_scratch_bytes")
    required_scratch_bytes = pooled_count * _INTERNAL_SCRATCH_BYTES_PER_VALUE
    if required_scratch_bytes > budget:
        raise ValueError(
            f"max_scratch_bytes={budget} is below the {required_scratch_bytes}-byte channel working estimate"
        )

    median = np.zeros(resolved_channel_count, dtype=np.float64)
    mean = np.zeros(resolved_channel_count, dtype=np.float64)
    scale = np.ones(resolved_channel_count, dtype=np.float64)
    usable = np.zeros(resolved_channel_count, dtype=bool)
    fit_count = np.zeros(resolved_channel_count, dtype=np.int64)
    for channel in range(resolved_channel_count):
        values = _read_channel(channel_reader, channel, count=pooled_count)
        median[channel], mean[channel], scale[channel], usable[channel], fit_count[channel] = _channel_statistics(values)
        del values

    return NumericPreprocessor(
        spec=field_spec,
        channel_names=_channel_names(field_spec, resolved_channel_count),
        median=median,
        mean=mean,
        scale=scale,
        usable=usable,
        fit_count=fit_count,
        evidence=trusted,
    )


__all__ = [
    "FitScopeEvidence",
    "NumericFieldSpec",
    "NumericPreprocessor",
    "fit_numeric_stream",
]
