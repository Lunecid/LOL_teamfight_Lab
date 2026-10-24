"""Full-corpus V preprocessing artifacts and bounded channel-wise fitting.

The public preprocessing classes live in ``v_model_fit_input_20261003``.
This module only supplies the full-corpus boundary: fitted state is written
as compact JSON with one shared fit-evidence record, and numeric statistics
are computed from read-only source arrays one channel at a time.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from studies.v_model_fit_input_20261003.preprocess import (
    CategoricalFieldSpec,
    CategoricalVocabulary,
    FitPreprocessor,
    FitScopeEvidence,
    NumericFieldSpec,
    _category_sort_key,
    _channel_names,
    VERSION as FIT_PREPROCESS_VERSION,
)
from studies.v_stream_preprocess_20261003.numeric import NumericPreprocessor


ARTIFACT_VERSION = "v_full_preprocess_artifacts_20261003_v1"
MANIFEST_NAME = "MANIFEST.json"
EVIDENCE_NAME = "evidence.json"
NUMERIC_NAME = "numeric.json"
VOCAB_NAME = "vocab.json"

_LANE_SPECS: dict[str, tuple[tuple[int, ...], NumericFieldSpec]] = {
    "node": ((-1, 10, 102), NumericFieldSpec("node", axis="player")),
    "team": ((-1, 2, 8), NumericFieldSpec("team", axis="team")),
    "global": ((-1, 75), NumericFieldSpec("global", axis="row")),
}
_LANE_WIDTHS = {name: spec_shape[-1] for name, (spec_shape, _spec) in _LANE_SPECS.items()}
_TOTAL_CHANNELS = sum(_LANE_WIDTHS.values())
_NUMERIC_SOURCE = {
    "kind": "NumericPreprocessor",
    "module": "studies.v_model_fit_input_20261003.preprocess",
    "version": FIT_PREPROCESS_VERSION,
}
_VOCAB_SOURCE = {
    "kind": "CategoricalVocabulary",
    "module": "studies.v_model_fit_input_20261003.preprocess",
    "version": FIT_PREPROCESS_VERSION,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _strict_indices(values: Sequence[Any], *, name: str, upper_bound: int | None = None) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of sorted unique integer indices")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of sorted unique integer indices") from exc
    if not raw:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError(f"{name} must contain integers")
    result = tuple(int(value) for value in raw)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted and unique")
    if upper_bound is not None and any(value >= upper_bound for value in result):
        raise ValueError(f"{name} contains an index outside the source rows")
    return result


def _checked_evidence(value: FitScopeEvidence | Mapping[str, Any]) -> FitScopeEvidence:
    """Validate serialized evidence hashes once and preserve object identity."""

    if isinstance(value, FitScopeEvidence):
        # Reparse the canonical payload so manually constructed objects receive
        # the same hash checks as load_bundle and mapping inputs.
        FitScopeEvidence.from_dict(value.to_dict())
        return value
    if isinstance(value, Mapping):
        # FitScopeEvidence.from_dict intentionally normalizes JSON scalars;
        # reject non-integer index tokens before that normalization can turn a
        # float or boolean into a different row reference.
        _strict_indices(value.get("row_indices", ()), name="evidence.row_indices")
        return FitScopeEvidence.from_dict(value)
    raise TypeError("evidence must be FitScopeEvidence or a mapping")


def _validate_source_array(value: Any, *, lane: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"numeric_arrays[{lane!r}] must be a numeric array") from exc
    if array.dtype.kind not in "biuf":
        raise TypeError(f"numeric_arrays[{lane!r}] must contain real numeric values")
    expected_rank = len(_LANE_SPECS[lane][0])
    if array.ndim != expected_rank:
        raise ValueError(f"numeric_arrays[{lane!r}] has the wrong rank")
    expected_tail = _LANE_SPECS[lane][0][1:]
    if tuple(array.shape[1:]) != expected_tail:
        raise ValueError(
            f"numeric_arrays[{lane!r}] must have shape (N, {', '.join(map(str, expected_tail))})"
        )
    if array.shape[0] <= 0:
        raise ValueError(f"numeric_arrays[{lane!r}] must contain at least one row")
    return array


def _selected_channel(array: np.ndarray, local_rows: np.ndarray, channel: int, *, lane: str) -> np.ndarray:
    """Copy only one selected pooled channel into a writable float64 vector."""

    if lane == "node":
        selected = array[local_rows, :, channel]
    elif lane == "team":
        selected = array[local_rows, :, channel]
    else:
        selected = array[local_rows, channel]
    try:
        values = np.array(selected, dtype=np.float64, copy=True, order="C").reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"numeric_arrays[{lane!r}] contains nonnumeric values") from exc
    if np.any(np.isinf(values)):
        raise ValueError(f"numeric_arrays[{lane!r}] channel {channel} contains infinity")
    return values


def _full_channel_statistics(values: np.ndarray) -> tuple[float, float, float, bool, int]:
    """Match ``fit_numeric``'s non-fast-axis float64 reduction order.

    ``_channel_statistics`` is the accepted stream policy, but NumPy's
    reduction can use a different summation path for a contiguous 1-D vector
    than for ``np.mean/std(imputed, axis=0)`` on the canonical row-by-channel
    matrix.  The full-corpus fitter keeps one channel only and reproduces the
    latter path with sequential ``cumsum`` reductions.
    """

    finite = np.isfinite(values)
    fit_count = int(np.sum(finite, dtype=np.int64))
    if fit_count == 0:
        return 0.0, 0.0, 1.0, False, 0

    median = float(np.median(values[finite]))
    values[~finite] = median
    denominator = values.size
    mean = float(np.cumsum(values, dtype=np.float64)[-1] / denominator)
    # Reuse the one-channel buffer so the exact path does not require a
    # second retained channel-sized working matrix.
    np.subtract(values, mean, out=values)
    np.square(values, out=values)
    scale = float(np.sqrt(np.cumsum(values, dtype=np.float64)[-1] / denominator))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return median, mean, scale, True, fit_count


def _category_token(value: Any) -> int | str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return value if value >= 0 else None
    if isinstance(value, str) and value:
        return value
    return None


def _sorted_categories(namespace: str, values: Any) -> tuple[int | str, ...]:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("categorical namespace names must be nonempty strings")
    if isinstance(values, (str, bytes)):
        raise TypeError(f"vocab_categories[{namespace!r}] must be a set of categories")
    try:
        normalized = {
            token
            for value in values
            if (token := _category_token(value)) is not None
        }
    except TypeError as exc:
        raise TypeError(f"vocab_categories[{namespace!r}] must be iterable") from exc
    return tuple(sorted(normalized, key=_category_sort_key))


def _serialized_categories(namespace: str, values: Any) -> tuple[int | str, ...]:
    """Validate categories already persisted in an artifact.

    Fitting follows the canonical tokenizer's ignore policy.  Persisted
    vocabularies are stricter: a malformed or negative token indicates a
    corrupt artifact and must not be silently filtered during reload.
    """

    if isinstance(values, (str, bytes)):
        raise ValueError(f"vocabulary artifact field {namespace!r} categories are malformed")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"vocabulary artifact field {namespace!r} categories are malformed") from exc
    categories: list[int | str] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"vocabulary artifact field {namespace!r} contains an invalid category")
        if isinstance(value, (int, np.integer)):
            token = int(value)
            if token < 0:
                raise ValueError(f"vocabulary artifact field {namespace!r} contains a negative category")
        elif isinstance(value, str) and value:
            token = value
        else:
            raise ValueError(f"vocabulary artifact field {namespace!r} contains an invalid category")
        categories.append(token)
    result = tuple(categories)
    if tuple(sorted(set(result), key=_category_sort_key)) != result:
        raise ValueError(f"vocabulary artifact field {namespace!r} categories are not sorted and unique")
    return result


def fit_bundle(
    numeric_arrays: Mapping[str, Any],
    selected_local_rows: Sequence[int],
    evidence: FitScopeEvidence | Mapping[str, Any],
    vocab_categories: Mapping[str, Any],
    *,
    on_channel: Callable[[str, int, int], Any] | None = None,
) -> FitPreprocessor:
    """Fit the full V preprocessing bundle from selected rows.

    ``selected_local_rows`` index the local row order of every lane array;
    ``evidence.row_indices`` retain the original nominal/global references.
    Arrays are read without mutation and one pooled channel is copied at a
    time, so a full ``N x channels`` matrix is never constructed.
    """

    if not isinstance(numeric_arrays, Mapping):
        raise TypeError("numeric_arrays must be a mapping of lane names to arrays")
    if set(numeric_arrays) != set(_LANE_SPECS):
        raise ValueError("numeric_arrays must contain exactly node, team, and global lanes")
    if on_channel is not None and not callable(on_channel):
        raise TypeError("on_channel must be callable")
    arrays = {lane: _validate_source_array(numeric_arrays[lane], lane=lane) for lane in _LANE_SPECS}
    n_rows = int(arrays["global"].shape[0])
    if any(int(array.shape[0]) != n_rows for array in arrays.values()):
        raise ValueError("numeric lane arrays must have the same row count")

    local_rows = _strict_indices(selected_local_rows, name="selected_local_rows", upper_bound=n_rows)
    trusted = _checked_evidence(evidence)
    nominal_rows = _strict_indices(trusted.row_indices, name="evidence.row_indices")
    if len(nominal_rows) != len(local_rows):
        raise ValueError("evidence.row_indices must align with selected_local_rows")
    if trusted.context.startswith("oof") and not trusted.excluded_game_keys:
        raise ValueError("OOF fit evidence requires explicit excluded_game_keys")
    if not isinstance(vocab_categories, Mapping):
        raise TypeError("vocab_categories must be a namespace mapping")

    row_array = np.asarray(local_rows, dtype=np.intp)
    numeric_fits: list[tuple[str, NumericPreprocessor]] = []
    completed = 0
    for lane in ("global", "node", "team"):
        array = arrays[lane]
        shape, spec = _LANE_SPECS[lane]
        width = _LANE_WIDTHS[lane]
        median = np.zeros(width, dtype=np.float64)
        mean = np.zeros(width, dtype=np.float64)
        scale = np.ones(width, dtype=np.float64)
        usable = np.zeros(width, dtype=bool)
        fit_count = np.zeros(width, dtype=np.int64)
        for channel in range(width):
            values = _selected_channel(array, row_array, channel, lane=lane)
            median[channel], mean[channel], scale[channel], usable[channel], fit_count[channel] = _full_channel_statistics(values)
            del values
            completed += 1
            if on_channel is not None:
                on_channel(lane, channel, _TOTAL_CHANNELS)
        for output in (median, mean, scale, usable, fit_count):
            output.setflags(write=False)
        # ``shape`` is retained in the lane table as an input contract; the
        # field spec and channel names are the accepted preprocessor state.
        del shape
        numeric_fits.append(
            (
                lane,
                NumericPreprocessor(
                    spec=spec,
                    channel_names=_channel_names(spec, width),
                    median=median,
                    mean=mean,
                    scale=scale,
                    usable=usable,
                    fit_count=fit_count,
                    evidence=trusted,
                ),
            )
        )
    assert completed == _TOTAL_CHANNELS

    categorical_fits = tuple(
        (
            namespace,
            CategoricalVocabulary(
                spec=CategoricalFieldSpec(namespace),
                categories=_sorted_categories(namespace, values),
                evidence=trusted,
            ),
        )
        for namespace, values in sorted(vocab_categories.items(), key=lambda item: item[0])
    )
    return FitPreprocessor(
        context=trusted.context,
        scope=trusted.scope,
        numeric=tuple(numeric_fits),
        categorical=categorical_fits,
        evidence=trusted,
    )


def _numeric_payload(preprocessor: FitPreprocessor) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, fitted in sorted(preprocessor.numeric, key=lambda item: item[0]):
        fields[name] = {
            "kind": "numeric",
            "name": name,
            "source": dict(_NUMERIC_SOURCE),
            "spec": fitted.spec.to_dict(),
            "channel_names": list(fitted.channel_names),
            "median": np.asarray(fitted.median, dtype=np.float64).tolist(),
            "mean": np.asarray(fitted.mean, dtype=np.float64).tolist(),
            "scale": np.asarray(fitted.scale, dtype=np.float64).tolist(),
            "usable": np.asarray(fitted.usable, dtype=bool).tolist(),
            "fit_count": np.asarray(fitted.fit_count, dtype=np.int64).tolist(),
        }
    return {"version": ARTIFACT_VERSION, "source": dict(_NUMERIC_SOURCE), "fields": fields}


def _vocab_payload(preprocessor: FitPreprocessor) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, fitted in sorted(preprocessor.categorical, key=lambda item: item[0]):
        fields[name] = {
            "kind": "categorical",
            "name": name,
            "source": dict(_VOCAB_SOURCE),
            "spec": fitted.spec.to_dict(),
            "pad_index": 0,
            "unk_index": 1,
            "categories": list(fitted.categories),
        }
    return {"version": ARTIFACT_VERSION, "source": dict(_VOCAB_SOURCE), "fields": fields}


def _manifest_base(
    preprocessor: FitPreprocessor,
    files: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "version": ARTIFACT_VERSION,
        "kind": "full_v_preprocessor_bundle",
        "source": {
            "module": "studies.v_model_fit_input_20261003.preprocess",
            "version": FIT_PREPROCESS_VERSION,
            "semantics": "numeric and vocabulary fields reference one evidence.json record",
        },
        "context": preprocessor.context,
        "scope": preprocessor.scope,
        "evidence": {"file": EVIDENCE_NAME, "version": preprocessor.evidence.version},
        "numeric": {"file": NUMERIC_NAME, "fields": [name for name, _fit in sorted(preprocessor.numeric)]},
        "vocab": {"file": VOCAB_NAME, "namespaces": [name for name, _fit in sorted(preprocessor.categorical)]},
        "files": dict(sorted(files.items())),
    }


def save_bundle(directory: str | Path, preprocessor: FitPreprocessor) -> Path:
    """Write a compact immutable bundle into a new exclusive directory.

    The manifest is written last and hashes the three data files.  An existing
    destination is refused so an accepted artifact cannot be overwritten.
    """

    if not isinstance(preprocessor, FitPreprocessor):
        raise TypeError("preprocessor must be a FitPreprocessor")
    root = Path(directory)
    evidence_payload = preprocessor.evidence.to_dict()
    numeric_payload = _numeric_payload(preprocessor)
    vocab_payload = _vocab_payload(preprocessor)
    encoded = {
        EVIDENCE_NAME: _canonical_json(evidence_payload) + "\n",
        NUMERIC_NAME: _canonical_json(numeric_payload) + "\n",
        VOCAB_NAME: _canonical_json(vocab_payload) + "\n",
    }
    root.mkdir(parents=True, exist_ok=False)
    try:
        for name, text in encoded.items():
            (root / name).write_text(text, encoding="utf-8")
        # Hash the bytes that reached disk.  ``Path.write_text`` uses the
        # platform newline convention, so hashing the pre-write strings would
        # be wrong on Windows.
        files = {name: _sha256_file(root / name) for name in encoded}
        manifest = _manifest_base(preprocessor, files)
        manifest["manifest_fingerprint"] = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
        # Write the manifest last and publish it atomically.  If interrupted
        # before the replace, the directory has no accepted manifest.
        pending_manifest = root / "MANIFEST.pending.json"
        _write_json(pending_manifest, manifest)
        pending_manifest.replace(root / MANIFEST_NAME)
    except Exception:
        # Preserve any partial evidence for diagnosis; the missing/invalid
        # manifest keeps the directory unusable as an accepted artifact.
        raise
    return root


def _safe_relative_file(root: Path, raw: Any, *, expected: str) -> Path:
    if not isinstance(raw, str) or raw != expected:
        raise ValueError(f"manifest file reference for {expected} is invalid")
    path = root / raw
    if path.parent != root or not path.is_file():
        raise ValueError(f"bundle is missing {expected}")
    return path


def _check_source(payload: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(payload) != dict(expected):
        raise ValueError(f"{label} source semantics differ from the accepted preprocessor")


def _load_numeric(payload: Mapping[str, Any], evidence: FitScopeEvidence) -> tuple[tuple[str, NumericPreprocessor], ...]:
    if payload.get("version") != ARTIFACT_VERSION or payload.get("source") != _NUMERIC_SOURCE:
        raise ValueError("numeric artifact version/source mismatch")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise ValueError("numeric artifact fields must be a mapping")
    fits: list[tuple[str, NumericPreprocessor]] = []
    for name, raw in sorted(raw_fields.items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("numeric artifact field is malformed")
        if raw.get("kind") != "numeric" or raw.get("name") != name or raw.get("source") != _NUMERIC_SOURCE:
            raise ValueError(f"numeric artifact field {name!r} is malformed")
        spec_payload = raw.get("spec")
        if not isinstance(spec_payload, Mapping):
            raise ValueError(f"numeric artifact field {name!r} spec is malformed")
        spec = NumericFieldSpec(
            name=str(spec_payload.get("name", "")),
            axis=str(spec_payload.get("axis", "row")),
            channels=tuple(spec_payload.get("channels", ())),
        )
        channel_names = tuple(str(value) for value in raw.get("channel_names", ()))
        size = len(channel_names)
        if not size:
            raise ValueError(f"numeric artifact field {name!r} has no channels")
        try:
            median = np.asarray(raw["median"], dtype=np.float64)
            mean = np.asarray(raw["mean"], dtype=np.float64)
            scale = np.asarray(raw["scale"], dtype=np.float64)
            usable = np.asarray(raw["usable"], dtype=bool)
            fit_count = np.asarray(raw["fit_count"], dtype=np.int64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"numeric artifact field {name!r} statistics are malformed") from exc
        for label, array in (("median", median), ("mean", mean), ("scale", scale), ("usable", usable), ("fit_count", fit_count)):
            if array.shape != (size,):
                raise ValueError(f"numeric artifact field {name!r} {label} width mismatch")
            array.setflags(write=False)
        fits.append(
            (
                name,
                NumericPreprocessor(
                    spec=spec,
                    channel_names=channel_names,
                    median=median,
                    mean=mean,
                    scale=scale,
                    usable=usable,
                    fit_count=fit_count,
                    evidence=evidence,
                ),
            )
        )
    return tuple(fits)


def _load_vocab(payload: Mapping[str, Any], evidence: FitScopeEvidence) -> tuple[tuple[str, CategoricalVocabulary], ...]:
    if payload.get("version") != ARTIFACT_VERSION or payload.get("source") != _VOCAB_SOURCE:
        raise ValueError("vocabulary artifact version/source mismatch")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise ValueError("vocabulary artifact fields must be a mapping")
    fits: list[tuple[str, CategoricalVocabulary]] = []
    for name, raw in sorted(raw_fields.items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("vocabulary artifact field is malformed")
        if raw.get("kind") != "categorical" or raw.get("name") != name or raw.get("source") != _VOCAB_SOURCE:
            raise ValueError(f"vocabulary artifact field {name!r} is malformed")
        if raw.get("pad_index") != 0 or raw.get("unk_index") != 1:
            raise ValueError(f"vocabulary artifact field {name!r} reserves invalid indices")
        spec_payload = raw.get("spec")
        if not isinstance(spec_payload, Mapping):
            raise ValueError(f"vocabulary artifact field {name!r} spec is malformed")
        spec = CategoricalFieldSpec(
            name=str(spec_payload.get("name", "")),
            axis=str(spec_payload.get("axis", "row")),
            channels=tuple(spec_payload.get("channels", ())),
        )
        categories = _serialized_categories(name, raw.get("categories", ()))
        fits.append((name, CategoricalVocabulary(spec=spec, categories=categories, evidence=evidence)))
    return tuple(fits)


def load_bundle(directory: str | Path) -> FitPreprocessor:
    """Load a bundle after verifying its manifest, file hashes, and evidence."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"bundle directory does not exist: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("bundle is missing MANIFEST.json")
    manifest = _read_json(manifest_path, "MANIFEST.json")
    if manifest.get("version") != ARTIFACT_VERSION or manifest.get("kind") != "full_v_preprocessor_bundle":
        raise ValueError("bundle manifest version/kind mismatch")
    fingerprint = manifest.get("manifest_fingerprint")
    base = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    if fingerprint != _sha256_bytes(_canonical_json(base).encode("utf-8")):
        raise ValueError("bundle manifest fingerprint mismatch")
    _check_source(manifest.get("source", {}), {
        "module": "studies.v_model_fit_input_20261003.preprocess",
        "version": FIT_PREPROCESS_VERSION,
        "semantics": "numeric and vocabulary fields reference one evidence.json record",
    }, "bundle")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping) or set(raw_files) != {EVIDENCE_NAME, NUMERIC_NAME, VOCAB_NAME}:
        raise ValueError("bundle manifest file list is invalid")
    paths = {
        EVIDENCE_NAME: _safe_relative_file(root, manifest.get("evidence", {}).get("file") if isinstance(manifest.get("evidence"), Mapping) else None, expected=EVIDENCE_NAME),
        NUMERIC_NAME: _safe_relative_file(root, manifest.get("numeric", {}).get("file") if isinstance(manifest.get("numeric"), Mapping) else None, expected=NUMERIC_NAME),
        VOCAB_NAME: _safe_relative_file(root, manifest.get("vocab", {}).get("file") if isinstance(manifest.get("vocab"), Mapping) else None, expected=VOCAB_NAME),
    }
    for name, path in paths.items():
        expected_hash = raw_files.get(name)
        if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
            raise ValueError(f"bundle file hash mismatch: {name}")
    evidence = FitScopeEvidence.from_dict(_read_json(paths[EVIDENCE_NAME], EVIDENCE_NAME))
    if manifest.get("evidence", {}).get("version") != evidence.version:
        raise ValueError("bundle evidence version mismatch")
    numeric = _load_numeric(_read_json(paths[NUMERIC_NAME], NUMERIC_NAME), evidence)
    categorical = _load_vocab(_read_json(paths[VOCAB_NAME], VOCAB_NAME), evidence)
    expected_numeric = manifest.get("numeric", {}).get("fields")
    expected_categorical = manifest.get("vocab", {}).get("namespaces")
    if expected_numeric != [name for name, _fit in numeric] or expected_categorical != [name for name, _fit in categorical]:
        raise ValueError("bundle manifest field list mismatch")
    if manifest.get("context") != evidence.context or manifest.get("scope") != evidence.scope:
        raise ValueError("bundle manifest scope differs from evidence")
    return FitPreprocessor(
        context=evidence.context,
        scope=evidence.scope,
        numeric=numeric,
        categorical=categorical,
        evidence=evidence,
    )


__all__ = [
    "ARTIFACT_VERSION",
    "EVIDENCE_NAME",
    "MANIFEST_NAME",
    "NUMERIC_NAME",
    "VOCAB_NAME",
    "fit_bundle",
    "load_bundle",
    "save_bundle",
]
