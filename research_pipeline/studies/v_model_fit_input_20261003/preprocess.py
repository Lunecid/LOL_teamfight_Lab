"""Fit-scoped preprocessing for the V model input phase.

The module deliberately knows nothing about the V semantic bank or a model
family.  A caller supplies ordered named numeric/categorical fields and the
exact row membership used for fitting.  Numeric statistics are pooled over
the declared semantic axis (one shared statistic for all ten players or both
observed teams), while categorical vocabularies are fitted from the same
rows and reserve PAD/UNK indices.

Only NumPy and the Python standard library are used here.  The persisted
objects contain their fit evidence, so a reload cannot silently acquire a
different row set or scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VERSION = "v_model_fit_preprocess_20261003_v1"
PAD_INDEX = 0
UNK_INDEX = 1
CONTEXTS = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
FIT_SCOPES = ("FIT", "REFIT_TRAIN")
FORBIDDEN_ROLES = frozenset({"STOP", "DEV", "V_DEV", "CAL", "V_CAL", "TEST", "TARGET", "Q_DEV", "Q_CAL"})


def _json_value(value: Any) -> Any:
    """Convert NumPy/dataclass-adjacent values to canonical JSON values."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_indices(indices: Sequence[int], *, n_rows: int | None = None) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise TypeError("fit_indices must be a sequence of integer row indices")
    values = tuple(indices)
    if not values:
        raise ValueError("fit_indices must not be empty")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("fit_indices must contain integers")
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("fit_indices must be nonnegative")
    if len(set(result)) != len(result):
        raise ValueError("fit_indices must be unique")
    if n_rows is not None and any(value >= n_rows for value in result):
        raise IndexError("fit_indices contains a row outside the source")
    return result


def _sequence_for_rows(value: Sequence[Any] | np.ndarray | None, rows: tuple[int, ...], n_rows: int, name: str) -> tuple[Any, ...]:
    if value is None:
        return tuple(None for _ in rows)
    values = tuple(value.tolist() if isinstance(value, np.ndarray) else value)
    if len(values) == n_rows:
        return tuple(values[index] for index in rows)
    if len(values) == len(rows):
        return values
    raise ValueError(f"{name} must have one value per source row or fit row")


def _bool_rows(value: Sequence[bool] | np.ndarray | None, rows: tuple[int, ...], n_rows: int, name: str) -> tuple[bool, ...] | None:
    if value is None:
        return None
    selected = _sequence_for_rows(value, rows, n_rows, name)
    if any(isinstance(item, (bool, np.bool_)) is False for item in selected):
        raise TypeError(f"{name} must contain booleans")
    return tuple(bool(item) for item in selected)


def _normal_role(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("source roles must be non-empty strings")
    return value.strip().upper()


@dataclass(frozen=True)
class FitScopeEvidence:
    """Immutable proof of the rows consumed by one fitted artifact.

    ``row_indices`` preserve caller order; the order is part of the evidence
    hash even though the statistics themselves are order-independent.  The
    optional membership flags are selected to those rows and allow a caller
    to prove that the rows were explicitly authorized for FIT or
    REFIT_TRAIN.  ``game_keys`` may repeat because a match contributes many
    nominal rows.
    """

    context: str
    scope: str
    row_indices: tuple[int, ...]
    game_keys: tuple[str | None, ...] = ()
    roles: tuple[str | None, ...] = ()
    membership: tuple[tuple[str, tuple[bool, ...]], ...] = ()
    excluded_game_keys: tuple[str, ...] = ()
    source_hash: str | None = None
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise ValueError("unsupported fit evidence version")
        if self.context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}")
        if self.scope not in FIT_SCOPES:
            raise ValueError(f"scope must be one of {FIT_SCOPES}")
        if not self.row_indices:
            raise ValueError("fit evidence must contain at least one row")
        if len(set(self.row_indices)) != len(self.row_indices) or any(index < 0 for index in self.row_indices):
            raise ValueError("fit evidence row_indices must be unique nonnegative integers")
        if self.game_keys and len(self.game_keys) != len(self.row_indices):
            raise ValueError("game_keys must align with row_indices")
        if self.roles and len(self.roles) != len(self.row_indices):
            raise ValueError("roles must align with row_indices")
        if len(set(self.excluded_game_keys)) != len(self.excluded_game_keys):
            raise ValueError("excluded_game_keys must be unique")
        excluded = set(self.excluded_game_keys)
        if excluded and any(key in excluded for key in self.game_keys if key is not None):
            raise ValueError("fit rows overlap excluded game keys")
        normalized_roles = {role for role in self.roles if role is not None}
        forbidden = normalized_roles & FORBIDDEN_ROLES
        if forbidden:
            raise ValueError(f"forbidden source roles in fit rows: {sorted(forbidden)}")
        allowed = {"TRAIN", "FIT", "REFIT_TRAIN"}
        unsupported = normalized_roles - allowed
        if unsupported:
            raise ValueError(f"unsupported source roles in fit rows: {sorted(unsupported)}")
        membership_names = {name for name, _ in self.membership}
        if len(membership_names) != len(self.membership):
            raise ValueError("membership names must be unique")
        for name, flags in self.membership:
            if len(flags) != len(self.row_indices):
                raise ValueError(f"membership flag {name!r} must align with row_indices")
            if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
                raise TypeError(f"membership flag {name!r} must contain booleans")
        required = "fit" if self.scope == "FIT" else "refit"
        required_flags = dict(self.membership).get(required)
        if self.membership and required_flags is None:
            raise ValueError(f"membership must include authorization flag {required!r}")
        if required_flags is not None and not all(required_flags):
            raise ValueError(f"all rows must be authorized by membership[{required!r}]")
        forbidden_memberships = ("dev", "cal", "target", "test")
        if self.scope == "FIT":
            forbidden_memberships = (*forbidden_memberships, "stop")
        for name in forbidden_memberships:
            flags = dict(self.membership).get(name)
            if flags is not None and any(flags):
                raise ValueError(f"membership[{name!r}] cannot be used for preprocessing FIT")
        if self.context.startswith("oof"):
            target_flags = dict(self.membership).get("target")
            if target_flags is not None and any(target_flags):
                raise ValueError("OOF target rows cannot enter preprocessing FIT")

    @property
    def consumed_indices(self) -> tuple[int, ...]:
        return self.row_indices

    @property
    def consumed_indices_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(list(self.row_indices)).encode("ascii")).hexdigest()

    @property
    def consumed_game_keys_sha256(self) -> str | None:
        if not self.game_keys:
            return None
        return hashlib.sha256(_canonical_json(list(self.game_keys)).encode("utf-8")).hexdigest()

    @property
    def membership_sha256(self) -> str | None:
        if not self.membership:
            return None
        return hashlib.sha256(_canonical_json(dict(self.membership)).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "context": self.context,
            "scope": self.scope,
            "row_indices": list(self.row_indices),
            "game_keys": list(self.game_keys),
            "roles": list(self.roles),
            "membership": {name: list(flags) for name, flags in self.membership},
            "excluded_game_keys": list(self.excluded_game_keys),
            "source_hash": self.source_hash,
            "consumed_indices_sha256": self.consumed_indices_sha256,
            "consumed_game_keys_sha256": self.consumed_game_keys_sha256,
            "membership_sha256": self.membership_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FitScopeEvidence":
        if not isinstance(payload, Mapping):
            raise TypeError("fit evidence payload must be a mapping")
        raw_membership = payload.get("membership", {})
        if not isinstance(raw_membership, Mapping):
            raise ValueError("membership must be a mapping")
        evidence = cls(
            context=str(payload.get("context", "")),
            scope=str(payload.get("scope", "")),
            row_indices=tuple(int(value) for value in payload.get("row_indices", ())),
            game_keys=tuple(payload.get("game_keys", ())),
            roles=tuple(payload.get("roles", ())),
            membership=tuple((str(name), tuple(bool(flag) for flag in flags)) for name, flags in sorted(raw_membership.items())),
            excluded_game_keys=tuple(str(value) for value in payload.get("excluded_game_keys", ())),
            source_hash=payload.get("source_hash"),
            version=str(payload.get("version", "")),
        )
        for key, expected in (
            ("consumed_indices_sha256", evidence.consumed_indices_sha256),
            ("consumed_game_keys_sha256", evidence.consumed_game_keys_sha256),
            ("membership_sha256", evidence.membership_sha256),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"fit evidence {key} mismatch")
        return evidence


def make_fit_evidence(
    fit_indices: Sequence[int],
    *,
    n_rows: int,
    context: str = "final",
    scope: str = "FIT",
    game_keys: Sequence[str | None] | np.ndarray | None = None,
    roles: Sequence[str | None] | np.ndarray | None = None,
    membership: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    excluded_game_keys: Sequence[str] = (),
    source_hash: str | None = None,
) -> FitScopeEvidence:
    """Validate and capture exact source membership for a fit operation."""

    rows = _strict_indices(fit_indices, n_rows=n_rows)
    selected_keys = _sequence_for_rows(game_keys, rows, n_rows, "game_keys") if game_keys is not None else ()
    if selected_keys and any(key is not None and not isinstance(key, str) for key in selected_keys):
        raise TypeError("game_keys must contain strings or None")
    selected_roles = _sequence_for_rows(roles, rows, n_rows, "roles") if roles is not None else ()
    normalized_roles = tuple(_normal_role(value) for value in selected_roles) if selected_roles else ()
    selected_membership: list[tuple[str, tuple[bool, ...]]] = []
    if membership is not None:
        for name in sorted(membership):
            if not isinstance(name, str) or not name:
                raise TypeError("membership names must be non-empty strings")
            flags = _bool_rows(membership[name], rows, n_rows, f"membership[{name!r}]")
            assert flags is not None
            selected_membership.append((name, flags))
    return FitScopeEvidence(
        context=context,
        scope=scope,
        row_indices=rows,
        game_keys=tuple(selected_keys),
        roles=normalized_roles,
        membership=tuple(selected_membership),
        excluded_game_keys=tuple(str(value) for value in excluded_game_keys),
        source_hash=source_hash,
    )


@dataclass(frozen=True)
class NumericFieldSpec:
    """Semantic shape metadata for one numeric field.

    ``axis='player'`` pools all ten participant rows into one statistic per
    channel, and ``axis='team'`` pools both observed-team rows.  Neither axis
    creates slot-specific statistics.
    """

    name: str
    axis: str = "row"
    channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        axis = self.normalized_axis
        if not self.name:
            raise ValueError("numeric field name must be non-empty")
        if axis not in {"row", "player", "team"}:
            raise ValueError("numeric field axis must be row, player, or team")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("numeric channel names must be unique")

    @property
    def normalized_axis(self) -> str:
        aliases = {"scalar": "row", "shared_player": "player", "paired_team": "team", "pairedteam": "team"}
        return aliases.get(self.axis.strip().lower(), self.axis.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "axis": self.normalized_axis, "channels": list(self.channels)}


@dataclass(frozen=True)
class CategoricalFieldSpec:
    name: str
    axis: str = "row"
    channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        axis = self.normalized_axis
        if not self.name:
            raise ValueError("categorical field name must be non-empty")
        if axis not in {"row", "player", "team"}:
            raise ValueError("categorical field axis must be row, player, or team")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("categorical channel names must be unique")

    @property
    def normalized_axis(self) -> str:
        aliases = {"scalar": "row", "shared_player": "player", "paired_team": "team", "pairedteam": "team"}
        return aliases.get(self.axis.strip().lower(), self.axis.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "axis": self.normalized_axis, "channels": list(self.channels)}


def _coerce_spec(spec: NumericFieldSpec | CategoricalFieldSpec | Mapping[str, Any] | None, name: str, *, categorical: bool) -> NumericFieldSpec | CategoricalFieldSpec:
    if spec is None:
        return (CategoricalFieldSpec if categorical else NumericFieldSpec)(name=name)
    if isinstance(spec, (NumericFieldSpec, CategoricalFieldSpec)):
        if spec.name != name:
            raise ValueError(f"field spec name {spec.name!r} does not match {name!r}")
        return spec
    if isinstance(spec, Mapping):
        cls = CategoricalFieldSpec if categorical else NumericFieldSpec
        return cls(name=name, axis=str(spec.get("axis", "row")), channels=tuple(str(value) for value in spec.get("channels", ())))
    raise TypeError("field specs must be field-spec objects or mappings")


def _field_matrix(values: Any, spec: NumericFieldSpec | CategoricalFieldSpec, *, rows: Sequence[int] | None = None) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"field {spec.name!r} must have a leading row dimension")
    if rows is not None:
        array = array[np.asarray(rows, dtype=np.int64)]
    axis = spec.normalized_axis
    if axis == "row":
        shape = array.shape
        matrix = array.reshape((shape[0], -1))
        semantic_shape = shape[1:]
    else:
        if array.ndim < 2:
            raise ValueError(f"{axis} field {spec.name!r} must include its participant/team axis")
        expected = 10 if axis == "player" else 2
        if array.shape[1] != expected:
            raise ValueError(f"{axis} field {spec.name!r} must have {expected} values on axis 1")
        semantic_shape = array.shape[1:]
        matrix = array.reshape((array.shape[0] * array.shape[1], -1))
    channel_count = matrix.shape[1]
    if spec.channels and len(spec.channels) != channel_count:
        raise ValueError(f"field {spec.name!r} declares {len(spec.channels)} channels but has {channel_count}")
    return matrix, tuple(int(value) for value in semantic_shape), tuple(int(value) for value in array.shape)


def _channel_names(spec: NumericFieldSpec | CategoricalFieldSpec, count: int) -> tuple[str, ...]:
    if spec.channels:
        return spec.channels
    if count == 1:
        return (spec.name,)
    return tuple(f"{spec.name}[{index}]" for index in range(count))


def _numeric_array(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"numeric field {name!r} must contain numeric values") from exc
    if result.ndim == 0:
        raise ValueError(f"numeric field {name!r} must have a leading row dimension")
    return result


@dataclass(frozen=True)
class NumericPreprocessor:
    spec: NumericFieldSpec
    channel_names: tuple[str, ...]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    usable: np.ndarray
    fit_count: np.ndarray
    evidence: FitScopeEvidence
    version: str = VERSION

    def __post_init__(self) -> None:
        size = len(self.channel_names)
        for name, value in (("median", self.median), ("mean", self.mean), ("scale", self.scale), ("usable", self.usable), ("fit_count", self.fit_count)):
            array = np.asarray(value)
            if array.shape != (size,):
                raise ValueError(f"numeric {name} must have one value per channel")
        if np.any(~np.isfinite(np.asarray(self.median, dtype=np.float64))) or np.any(~np.isfinite(np.asarray(self.mean, dtype=np.float64))):
            raise ValueError("numeric statistics must be finite")
        if np.any(np.asarray(self.scale, dtype=np.float64) <= 0) or np.any(~np.isfinite(np.asarray(self.scale, dtype=np.float64))):
            raise ValueError("numeric scale must be finite and positive")
        if np.any(np.asarray(self.fit_count) < 0):
            raise ValueError("numeric fit_count cannot be negative")
        if self.version != VERSION:
            raise ValueError("unsupported numeric preprocessor version")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def consumed_indices(self) -> tuple[int, ...]:
        return self.evidence.consumed_indices

    @property
    def sd(self) -> np.ndarray:
        """Alias matching the execution-card terminology for the scale."""

        return self.scale

    @property
    def all_missing(self) -> np.ndarray:
        return ~self.usable

    @classmethod
    def fit(cls, values: Any, **kwargs: Any) -> "NumericPreprocessor":
        return fit_numeric(values, **kwargs)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "kind": "numeric",
            "spec": self.spec.to_dict(),
            "channel_names": list(self.channel_names),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "usable": self.usable.tolist(),
            "fit_count": self.fit_count.tolist(),
            "evidence": self.evidence.to_dict(),
        }
        if include_fingerprint:
            payload["fingerprint"] = _fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NumericPreprocessor":
        if payload.get("fingerprint") != _fingerprint({key: value for key, value in payload.items() if key != "fingerprint"}):
            raise ValueError("numeric preprocessor fingerprint mismatch")
        if payload.get("kind") != "numeric":
            raise ValueError("payload is not a numeric preprocessor")
        spec = NumericFieldSpec(name=str(payload["spec"]["name"]), axis=str(payload["spec"]["axis"]), channels=tuple(payload["spec"].get("channels", ())))
        return cls(
            spec=spec,
            channel_names=tuple(str(value) for value in payload["channel_names"]),
            median=np.asarray(payload["median"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            usable=np.asarray(payload["usable"], dtype=bool),
            fit_count=np.asarray(payload["fit_count"], dtype=np.int64),
            evidence=FitScopeEvidence.from_dict(payload["evidence"]),
            version=str(payload.get("version", "")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_canonical_json(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NumericPreprocessor":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, values: Any) -> np.ndarray:
        raw = _numeric_array(values, self.spec.name)
        matrix, semantic_shape, original_shape = _field_matrix(raw, self.spec)
        if matrix.shape[1] != len(self.channel_names):
            raise ValueError(f"field {self.spec.name!r} has a different channel width")
        finite = np.isfinite(matrix)
        clean = np.where(finite, matrix, self.median.reshape((1, -1)))
        transformed = np.where(
            self.usable.reshape((1, -1)),
            (clean - self.mean.reshape((1, -1))) / self.scale.reshape((1, -1)),
            0.0,
        )
        return transformed.reshape(original_shape).astype(np.float32, copy=False)


def fit_numeric(
    values: Any,
    *,
    fit_indices: Sequence[int],
    name: str | None = None,
    axis: str | None = None,
    channels: Sequence[str] = (),
    spec: NumericFieldSpec | Mapping[str, Any] | None = None,
    context: str = "final",
    scope: str = "FIT",
    game_keys: Sequence[str | None] | np.ndarray | None = None,
    roles: Sequence[str | None] | np.ndarray | None = None,
    membership: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    excluded_game_keys: Sequence[str] = (),
    source_hash: str | None = None,
) -> NumericPreprocessor:
    """Fit one field using only ``fit_indices`` and return its immutable state."""

    raw = _numeric_array(values, "numeric")
    field_name = name or (spec.name if isinstance(spec, NumericFieldSpec) else "numeric")
    if spec is None and axis is not None:
        spec = NumericFieldSpec(field_name, axis=axis, channels=tuple(channels))
    field_spec = _coerce_spec(spec, field_name, categorical=False)
    assert isinstance(field_spec, NumericFieldSpec)
    evidence = make_fit_evidence(
        fit_indices,
        n_rows=raw.shape[0],
        context=context,
        scope=scope,
        game_keys=game_keys,
        roles=roles,
        membership=membership,
        excluded_game_keys=excluded_game_keys,
        source_hash=source_hash,
    )
    # Select the authorized rows before flattening.  A full nominal bank must
    # never be copied merely to fit one scope's statistics.
    fit_matrix, _semantic_shape, _original_shape = _field_matrix(raw, field_spec, rows=evidence.row_indices)
    finite = np.isfinite(fit_matrix)
    count = finite.sum(axis=0).astype(np.int64)
    usable = count > 0
    median = np.zeros(fit_matrix.shape[1], dtype=np.float64)
    for channel in range(fit_matrix.shape[1]):
        observed = fit_matrix[finite[:, channel], channel]
        if observed.size:
            median[channel] = float(np.median(observed))
    imputed = np.where(finite, fit_matrix, median.reshape((1, -1)))
    mean = np.mean(imputed, axis=0, dtype=np.float64)
    scale = np.std(imputed, axis=0, dtype=np.float64)
    mean[~usable] = 0.0
    scale[~usable] = 1.0
    scale[(usable) & ((scale == 0) | ~np.isfinite(scale))] = 1.0
    return NumericPreprocessor(
        spec=field_spec,
        channel_names=_channel_names(field_spec, fit_matrix.shape[1]),
        median=median,
        mean=mean,
        scale=scale,
        usable=usable,
        fit_count=count,
        evidence=evidence,
    )


def fit_numeric_preprocessor(*args: Any, **kwargs: Any) -> NumericPreprocessor:
    """Compatibility alias for callers naming the fitted object explicitly."""

    return fit_numeric(*args, **kwargs)


def _category_token(value: Any) -> int | str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return value if value >= 0 else None
    if isinstance(value, str) and value:
        return value
    return None


def _category_sort_key(value: int | str) -> tuple[int, Any]:
    return (0, value) if isinstance(value, int) else (1, value)


@dataclass(frozen=True)
class CategoricalVocabulary:
    spec: CategoricalFieldSpec
    categories: tuple[int | str, ...]
    evidence: FitScopeEvidence
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise ValueError("unsupported categorical vocabulary version")
        if tuple(sorted(set(self.categories), key=_category_sort_key)) != self.categories:
            raise ValueError("categorical categories must be sorted and unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def mapping(self) -> dict[int | str, int]:
        return {value: index + 2 for index, value in enumerate(self.categories)}

    @property
    def raw_categories(self) -> tuple[int | str, ...]:
        return self.categories

    @property
    def category_to_index(self) -> dict[int | str, int]:
        return self.mapping

    @property
    def size(self) -> int:
        return len(self.categories)

    @property
    def num_embeddings(self) -> int:
        return self.size + 2

    @property
    def consumed_indices(self) -> tuple[int, ...]:
        return self.evidence.consumed_indices

    @classmethod
    def fit(cls, values: Any, **kwargs: Any) -> "CategoricalVocabulary":
        return fit_vocabulary(values, **kwargs)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "kind": "categorical",
            "spec": self.spec.to_dict(),
            "pad_index": PAD_INDEX,
            "unk_index": UNK_INDEX,
            "categories": list(self.categories),
            "evidence": self.evidence.to_dict(),
        }
        if include_fingerprint:
            payload["fingerprint"] = _fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CategoricalVocabulary":
        if payload.get("fingerprint") != _fingerprint({key: value for key, value in payload.items() if key != "fingerprint"}):
            raise ValueError("categorical vocabulary fingerprint mismatch")
        if payload.get("kind") != "categorical" or payload.get("pad_index") != PAD_INDEX or payload.get("unk_index") != UNK_INDEX:
            raise ValueError("invalid categorical vocabulary payload")
        spec = CategoricalFieldSpec(name=str(payload["spec"]["name"]), axis=str(payload["spec"]["axis"]), channels=tuple(payload["spec"].get("channels", ())))
        categories = tuple(_category_token(value) for value in payload.get("categories", ()))
        if any(value is None for value in categories):
            raise ValueError("categorical vocabulary contains an invalid raw category")
        return cls(spec=spec, categories=categories, evidence=FitScopeEvidence.from_dict(payload["evidence"]), version=str(payload.get("version", "")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_canonical_json(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CategoricalVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, values: Any, *, valid_mask: Any | None = None, pad_mask: Any | None = None) -> np.ndarray:
        array = np.asarray(values, dtype=object)
        mapping = self.mapping
        result = np.full(array.shape, UNK_INDEX, dtype=np.int64)
        valid = np.ones(array.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.shape != array.shape:
            raise ValueError("valid_mask must have the same shape as categorical values")
        for index in np.ndindex(array.shape):
            if valid[index]:
                token = _category_token(array[index])
                if token is not None:
                    result[index] = mapping.get(token, UNK_INDEX)
        if pad_mask is not None:
            padding = np.asarray(pad_mask, dtype=bool)
            if padding.shape != array.shape:
                raise ValueError("pad_mask must have the same shape as categorical values")
            result[padding] = PAD_INDEX
        return result

    def transform_ragged(self, rows: Sequence[Iterable[int | str]]) -> tuple[np.ndarray, ...]:
        """Transform variable-length namespace bags without rectangular padding."""

        transformed: list[np.ndarray] = []
        mapping = self.mapping
        for row in rows:
            transformed.append(np.asarray([mapping.get(token, UNK_INDEX) if token is not None else UNK_INDEX for token in (_category_token(value) for value in row)], dtype=np.int64))
        return tuple(transformed)


def fit_vocabulary(
    values: Any,
    *,
    fit_indices: Sequence[int],
    name: str | None = None,
    axis: str | None = None,
    channels: Sequence[str] = (),
    spec: CategoricalFieldSpec | Mapping[str, Any] | None = None,
    valid_mask: Any | None = None,
    context: str = "final",
    scope: str = "FIT",
    game_keys: Sequence[str | None] | np.ndarray | None = None,
    roles: Sequence[str | None] | np.ndarray | None = None,
    membership: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    excluded_game_keys: Sequence[str] = (),
    source_hash: str | None = None,
) -> CategoricalVocabulary:
    """Fit one shared vocabulary; raw integer ``0`` is a valid category."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        raise ValueError("categorical values must have a leading row dimension")
    field_name = name or (spec.name if isinstance(spec, CategoricalFieldSpec) else "categorical")
    if spec is None and axis is not None:
        spec = CategoricalFieldSpec(field_name, axis=axis, channels=tuple(channels))
    field_spec = _coerce_spec(spec, field_name, categorical=True)
    assert isinstance(field_spec, CategoricalFieldSpec)
    evidence = make_fit_evidence(
        fit_indices,
        n_rows=array.shape[0],
        context=context,
        scope=scope,
        game_keys=game_keys,
        roles=roles,
        membership=membership,
        excluded_game_keys=excluded_game_keys,
        source_hash=source_hash,
    )
    matrix, _semantic_shape, _original_shape = _field_matrix(array, field_spec, rows=None)
    valid = np.ones(array.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != array.shape:
        raise ValueError("valid_mask must have the same shape as categorical values")
    fit_rows = np.asarray(evidence.row_indices, dtype=np.int64)
    if field_spec.normalized_axis == "row":
        valid_matrix = valid[fit_rows].reshape((len(fit_rows), -1))
        fit_matrix = matrix[fit_rows]
    else:
        width = 10 if field_spec.normalized_axis == "player" else 2
        fit_matrix = matrix.reshape((array.shape[0], width, -1))[fit_rows].reshape((-1, matrix.shape[1]))
        valid_matrix = valid.reshape((array.shape[0], width, -1))[fit_rows].reshape((-1, matrix.shape[1]))
    tokens: set[int | str] = set()
    for value, is_valid in zip(fit_matrix.reshape(-1), valid_matrix.reshape(-1)):
        if is_valid:
            token = _category_token(value)
            if token is not None:
                tokens.add(token)
    categories = tuple(sorted(tokens, key=_category_sort_key))
    return CategoricalVocabulary(spec=field_spec, categories=categories, evidence=evidence)


def fit_ragged_vocabulary(
    rows: Sequence[Iterable[int | str]],
    *,
    fit_indices: Sequence[int],
    name: str | None = None,
    axis: str | None = None,
    channels: Sequence[str] = (),
    spec: CategoricalFieldSpec | Mapping[str, Any] | None = None,
    context: str = "final",
    scope: str = "FIT",
    game_keys: Sequence[str | None] | np.ndarray | None = None,
    roles: Sequence[str | None] | np.ndarray | None = None,
    membership: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    excluded_game_keys: Sequence[str] = (),
    source_hash: str | None = None,
) -> CategoricalVocabulary:
    """Fit one namespace vocabulary from variable-length per-row bags.

    The selected rows are iterated directly.  No padded or rectangular array
    is allocated, and duplicate values within a bag do not affect the
    vocabulary.  Counts remain owned by the semantic bank and are irrelevant
    to vocabulary membership.
    """

    if isinstance(rows, (str, bytes)):
        raise TypeError("ragged rows must be a sequence of iterables")
    n_rows = len(rows)
    field_name = name or (spec.name if isinstance(spec, CategoricalFieldSpec) else "categorical")
    if spec is None and axis is not None:
        spec = CategoricalFieldSpec(field_name, axis=axis, channels=tuple(channels))
    field_spec = _coerce_spec(spec, field_name, categorical=True)
    assert isinstance(field_spec, CategoricalFieldSpec)
    if field_spec.normalized_axis != "row":
        raise ValueError("ragged vocabulary fields must use axis='row'")
    evidence = make_fit_evidence(
        fit_indices,
        n_rows=n_rows,
        context=context,
        scope=scope,
        game_keys=game_keys,
        roles=roles,
        membership=membership,
        excluded_game_keys=excluded_game_keys,
        source_hash=source_hash,
    )
    categories: set[int | str] = set()
    for index in evidence.row_indices:
        try:
            values = rows[index]
            iterator = iter(values)
        except TypeError as exc:
            raise TypeError(f"ragged row {index} must be iterable") from exc
        for value in iterator:
            token = _category_token(value)
            if token is not None:
                categories.add(token)
    return CategoricalVocabulary(spec=field_spec, categories=tuple(sorted(categories, key=_category_sort_key)), evidence=evidence)


def transform_ragged_vocabulary(vocabulary: CategoricalVocabulary, rows: Sequence[Iterable[int | str]]) -> tuple[np.ndarray, ...]:
    """Transform variable-length rows with a fitted namespace vocabulary."""

    if not isinstance(vocabulary, CategoricalVocabulary):
        raise TypeError("vocabulary must be a CategoricalVocabulary")
    return vocabulary.transform_ragged(rows)


def fit_categorical_vocabulary(*args: Any, **kwargs: Any) -> CategoricalVocabulary:
    """Compatibility alias for the fit-scoped vocabulary helper."""

    return fit_vocabulary(*args, **kwargs)


@dataclass(frozen=True)
class FitPreprocessor:
    """A context/scope bundle of numeric preprocessors and vocabularies."""

    context: str
    scope: str
    numeric: tuple[tuple[str, NumericPreprocessor], ...]
    categorical: tuple[tuple[str, CategoricalVocabulary], ...]
    evidence: FitScopeEvidence
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS or self.scope not in FIT_SCOPES:
            raise ValueError("invalid fit preprocessor context or scope")
        if self.evidence.context != self.context or self.evidence.scope != self.scope:
            raise ValueError("bundle evidence does not match context/scope")
        numeric_names = [name for name, _ in self.numeric]
        category_names = [name for name, _ in self.categorical]
        if len(set(numeric_names)) != len(numeric_names) or len(set(category_names)) != len(category_names):
            raise ValueError("bundle field names must be unique")
        if any(fit.evidence != self.evidence for _, fit in (*self.numeric, *self.categorical)):
            raise ValueError("all bundle fields must share exact fit evidence")

    @property
    def numeric_map(self) -> dict[str, NumericPreprocessor]:
        return dict(self.numeric)

    @property
    def categorical_map(self) -> dict[str, CategoricalVocabulary]:
        return dict(self.categorical)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def consumed_indices(self) -> tuple[int, ...]:
        return self.evidence.consumed_indices

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "kind": "bundle",
            "context": self.context,
            "scope": self.scope,
            "evidence": self.evidence.to_dict(),
            "numeric": {name: fit.to_dict() for name, fit in self.numeric},
            "categorical": {name: vocab.to_dict() for name, vocab in self.categorical},
        }
        if include_fingerprint:
            payload["fingerprint"] = _fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FitPreprocessor":
        if payload.get("fingerprint") != _fingerprint({key: value for key, value in payload.items() if key != "fingerprint"}):
            raise ValueError("fit preprocessor fingerprint mismatch")
        if payload.get("kind") != "bundle":
            raise ValueError("payload is not a fit preprocessor")
        evidence = FitScopeEvidence.from_dict(payload["evidence"])
        numeric = tuple((str(name), NumericPreprocessor.from_dict(value)) for name, value in sorted(payload.get("numeric", {}).items()))
        categorical = tuple((str(name), CategoricalVocabulary.from_dict(value)) for name, value in sorted(payload.get("categorical", {}).items()))
        return cls(context=str(payload["context"]), scope=str(payload["scope"]), numeric=numeric, categorical=categorical, evidence=evidence, version=str(payload.get("version", "")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_canonical_json(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FitPreprocessor":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(
        self,
        numeric: Mapping[str, Any],
        categorical: Mapping[str, Any],
        *,
        categorical_valid_masks: Mapping[str, Any] | None = None,
        categorical_pad_masks: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        numeric_map = self.numeric_map
        category_map = self.categorical_map
        if set(numeric) != set(numeric_map) or set(categorical) != set(category_map):
            raise ValueError("transform fields must exactly match the fitted bundle")
        valid_masks = categorical_valid_masks or {}
        pad_masks = categorical_pad_masks or {}
        return {
            "numeric": {name: numeric_map[name].transform(numeric[name]) for name in numeric_map},
            "categorical": {name: category_map[name].transform(categorical[name], valid_mask=valid_masks.get(name), pad_mask=pad_masks.get(name)) for name in category_map},
        }


def fit_preprocessor(
    numeric: Mapping[str, Any],
    categorical: Mapping[str, Any],
    *,
    fit_indices: Sequence[int],
    context: str = "final",
    scope: str = "FIT",
    numeric_specs: Mapping[str, NumericFieldSpec | Mapping[str, Any]] | None = None,
    categorical_specs: Mapping[str, CategoricalFieldSpec | Mapping[str, Any]] | None = None,
    categorical_valid_masks: Mapping[str, Any] | None = None,
    game_keys: Sequence[str | None] | np.ndarray | None = None,
    roles: Sequence[str | None] | np.ndarray | None = None,
    membership: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    excluded_game_keys: Sequence[str] = (),
    source_hash: str | None = None,
) -> FitPreprocessor:
    """Fit all fields under one exact source membership contract."""

    if not isinstance(numeric, Mapping) or not isinstance(categorical, Mapping):
        raise TypeError("numeric and categorical fields must be mappings")
    all_fields = [*numeric.values(), *categorical.values()]
    n_rows = None
    for field in all_fields:
        shape = np.shape(field)
        if not shape:
            raise ValueError("all fields must have a leading row dimension")
        if n_rows is None:
            n_rows = int(shape[0])
        elif int(shape[0]) != n_rows:
            raise ValueError("all fields must have the same row count")
    if n_rows is None:
        raise ValueError("at least one numeric or categorical field is required")
    evidence = make_fit_evidence(
        fit_indices,
        n_rows=n_rows,
        context=context,
        scope=scope,
        game_keys=game_keys,
        roles=roles,
        membership=membership,
        excluded_game_keys=excluded_game_keys,
        source_hash=source_hash,
    )
    numeric_fits = []
    for name, values in numeric.items():
        spec = (numeric_specs or {}).get(name)
        numeric_fits.append((name, fit_numeric(values, fit_indices=evidence.row_indices, name=name, spec=spec, context=context, scope=scope, game_keys=game_keys, roles=roles, membership=membership, excluded_game_keys=excluded_game_keys, source_hash=source_hash)))
    category_fits = []
    for name, values in categorical.items():
        spec = (categorical_specs or {}).get(name)
        valid_mask = (categorical_valid_masks or {}).get(name)
        category_fits.append((name, fit_vocabulary(values, fit_indices=evidence.row_indices, name=name, spec=spec, valid_mask=valid_mask, context=context, scope=scope, game_keys=game_keys, roles=roles, membership=membership, excluded_game_keys=excluded_game_keys, source_hash=source_hash)))
    return FitPreprocessor(context=context, scope=scope, numeric=tuple(numeric_fits), categorical=tuple(category_fits), evidence=evidence)


__all__ = [
    "CONTEXTS",
    "FIT_SCOPES",
    "FORBIDDEN_ROLES",
    "FitPreprocessor",
    "FitScopeEvidence",
    "CategoricalFieldSpec",
    "CategoricalVocabulary",
    "NumericFieldSpec",
    "NumericPreprocessor",
    "PAD_INDEX",
    "UNK_INDEX",
    "VERSION",
    "fit_categorical_vocabulary",
    "fit_numeric",
    "fit_numeric_preprocessor",
    "fit_preprocessor",
    "fit_ragged_vocabulary",
    "fit_vocabulary",
    "make_fit_evidence",
    "transform_ragged_vocabulary",
]
