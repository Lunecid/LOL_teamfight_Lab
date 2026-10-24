"""Fail-closed scientific V candidate training.

The production cache and preprocessing stages own population membership and
feature construction.  This module consumes their already transformed
``table_csr``/``StructuredBatch`` batches and owns only candidate fitting,
raw V_DEV selection, positive-slope V_CAL calibration, checkpoints, and
progress evidence.  The default model path is deliberately bound to
``build_locked_model``; a caller cannot turn an incomplete acceptance record
into a training run by setting a local flag.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from typing import Any, Protocol

# NumPy/SciPy and the packaged Torch wheel may load separate Intel OpenMP
# runtimes on Windows.  This process-local setting prevents an import-time
# abort; it does not alter machine-wide configuration or numerical policy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# The execution card requires deterministic algorithms; cuBLAS reads this
# workspace setting when its first handle is created, so it must precede any
# CUDA work in the process (EXECUTION_CARDS optimizer.neural).
CUBLAS_WORKSPACE = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)

import numpy as np

from studies.v_model_fit_input_20261003.models import (
    BagInput,
    FAMILIES,
    LogisticNonconvergenceError,
    StructuredBatch,
    build_model,
    family_cards,
)
from studies.v_training_gate_20261003.guard import build_locked_model, validate_indexed_batch


VERSION = "v_production_runner_20261004_v3"
FIXED_SEEDS = (20261001, 20261002, 20261003)
# V family tie tolerance (research_execution_20261003/PROTOCOL_KO.md, V .0001;
# 5e-4 is the q tolerance and must not be used for V).
V_FAMILY_TIE_TOLERANCE = 0.0001
PHASE1_SEED = 20261001
BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 1e-5
NEURAL_LR = 1e-3
GRADIENT_CLIP = 1.0
LGB_ROUNDS = 2000
LGB_PATIENCE = 100
FAMILY_PRIORITY = ("logistic", "lightgbm", "plain_mlp", "shared_slot_mlp", "team_set_mlp")


class BatchSource(Protocol):
    def iter_batches(self, split: str, **kwargs: Any) -> Iterable[Any]: ...


@dataclass(frozen=True)
class RunnerConfig:
    """Locked optimization settings for one candidate job."""

    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    learning_rate: float = NEURAL_LR
    gradient_clip: float = GRADIENT_CLIP
    lgb_rounds: int = LGB_ROUNDS
    lgb_patience: int = LGB_PATIENCE
    cpu_threads: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if int(self.batch_size) != BATCH_SIZE:
            raise ValueError("V production batch_size is fixed at 1024")
        if not 1 <= int(self.max_epochs) <= MAX_EPOCHS:
            raise ValueError("max_epochs must be in 1..30")
        if int(self.patience) != PATIENCE or float(self.min_delta) != MIN_DELTA:
            raise ValueError("V production STOP settings are fixed at patience=5 and min_delta=1e-5")
        if float(self.learning_rate) != NEURAL_LR or float(self.gradient_clip) != GRADIENT_CLIP:
            raise ValueError("V production AdamW settings are fixed")
        if int(self.lgb_rounds) != LGB_ROUNDS or int(self.lgb_patience) != LGB_PATIENCE:
            raise ValueError("V production LightGBM settings are fixed")
        if int(self.cpu_threads) < 1 or int(self.cpu_threads) > 3:
            raise ValueError("cpu_threads must be in 1..3")


@dataclass(frozen=True)
class CandidateJob:
    """One family/card/context candidate and its replayable input sources.

    Sources must be replayable iterables, callables, or objects exposing
    ``iter_batches(split, ...)``.  Each yielded item is either a
    :class:`Batch`/mapping or ``(inputs, target[, metadata])``.  Targets are
    blue-side ``W`` unless marked as already supervised; TRAIN rows with the
    deterministic replacement-swap flag are flipped here.
    """

    family: str
    card_id: str
    seed: int
    acceptance_path: str | Path
    output_dir: str | Path
    fit_source: Any | None = None
    stop_source: Any | None = None
    refit_source: Any | None = None
    dev_source: Any | None = None
    context: str = "final"
    fold: int | None = None
    dimensions: Mapping[str, Any] | int | None = None
    # REFIT may have a separately fitted vocabulary/statistics width.  The
    # production cache owner supplies this explicitly when it differs from
    # the FIT bundle; omitting it preserves the canonical shared dimensions.
    refit_dimensions: Mapping[str, Any] | int | None = None
    config: RunnerConfig = RunnerConfig()
    expected_fit_rows: int | None = None
    expected_stop_rows: int | None = None
    expected_refit_rows: int | None = None
    expected_dev_rows: int | None = None
    excluded_game_keys: tuple[str, ...] = ()
    job_id: str | None = None

    def __post_init__(self) -> None:
        family = str(self.family).lower()
        if family not in FAMILIES:
            raise ValueError(f"unknown V family: {self.family!r}")
        object.__setattr__(self, "family", family)
        if self.seed != PHASE1_SEED:
            raise ValueError(f"phase-1 V runner is locked to seed {PHASE1_SEED}")
        if self.context != "final" and self.context not in {f"oof{i}" for i in range(5)}:
            raise ValueError("context must be final or oof0..oof4")
        if self.context == "final" and self.fold is not None:
            raise ValueError("final candidate cannot declare an OOF fold")
        if self.context != "final" and int(self.fold if self.fold is not None else -1) != int(self.context[-1]):
            raise ValueError("OOF fold must match context")
        if self.context != "final" and not self.excluded_game_keys:
            raise ValueError("OOF candidates require nonempty excluded_game_keys")
        cards = family_cards(family)
        if not any(card["id"] == self.card_id for card in cards):
            raise ValueError(f"unknown canonical {family} card: {self.card_id!r}")
        for name in ("expected_fit_rows", "expected_stop_rows", "expected_refit_rows", "expected_dev_rows"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        excluded = tuple(str(key) for key in self.excluded_game_keys)
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded_game_keys must be unique")
        object.__setattr__(self, "excluded_game_keys", excluded)


@dataclass(frozen=True)
class Batch:
    """Explicit source envelope accepted by the runner."""

    inputs: Any
    target: Any | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CandidateResult:
    """Serializable summary of one candidate attempt."""

    job_id: str
    family: str
    card_id: str
    seed: int
    context: str
    status: str
    output_dir: str
    fit_rows: int
    stop_rows: int
    refit_rows: int
    dev_rows: int
    chosen_epoch: int | None
    fit_metrics: Mapping[str, Any]
    stop_metrics: Mapping[str, Any]
    refit_metrics: Mapping[str, Any]
    dev_metrics: Mapping[str, Any]
    reload_metrics: Mapping[str, Any]
    checkpoint: str | None
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            **{field.name: _jsonable(getattr(self, field.name)) for field in fields(self)},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateResult":
        """Rebuild a persisted terminal result written by :meth:`to_dict`."""

        names = {field.name for field in fields(cls)}
        missing = sorted(name for name in names if name not in payload and name != "error")
        if missing:
            raise ValueError(f"candidate result is missing fields: {missing}")
        return cls(**{name: payload.get(name) for name in names})


@dataclass(frozen=True)
class PositiveSlopeCalibrator:
    """V_CAL positive-slope sigmoid fitted with float64 L-BFGS-B."""

    a: float
    b: float
    loss: float
    projected_gradient_inf: float
    converged: bool
    class_counts: Mapping[str, int]

    def predict(self, raw_probability: Any) -> np.ndarray:
        probability = np.asarray(raw_probability, dtype=np.float64)
        if not np.isfinite(probability).all():
            raise ValueError("raw probabilities must be finite")
        logit = np.log(np.clip(probability, 1e-12, 1.0 - 1e-12)) - np.log1p(-np.clip(probability, 1e-12, 1.0 - 1e-12))
        return _sigmoid(self.a * logit + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "formula": "sigmoid(a * logit(clip(raw_probability,1e-12,1-1e-12)) + b)",
            "a": self.a,
            "b": self.b,
            "loss": self.loss,
            "projected_gradient_inf": self.projected_gradient_inf,
            "converged": self.converged,
            "class_counts": dict(self.class_counts),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_retry(source: Path, destination: Path) -> None:
    """Publish one file, retrying a transient Windows sharing denial."""

    for attempt in range(41):
        try:
            os.replace(str(source), str(destination))
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32} or attempt == 40:
                raise
            time.sleep(0.05)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_retry(pending, path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_retry(pending, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ resumption
# A candidate interrupted by a power loss or reboot continues from its last
# verified epoch checkpoint (neural FIT/REFIT) or completed phase (FIT_DONE /
# REFIT_DONE).  Restoring model, optimizer and every RNG state reproduces the
# uninterrupted trajectory exactly; anything that does not verify (identity,
# device, file hash) is ignored and that work is recomputed.

def _device_name(device: str) -> str:
    if str(device).startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() else "cuda-unavailable"
    return "cpu"


def _job_identity(job: "CandidateJob", job_id: str) -> dict[str, Any]:
    """What a resumable checkpoint must match: job, locked config, acceptance bytes, host and device."""

    import platform

    acceptance = Path(job.acceptance_path)
    return _jsonable({
        "version": VERSION, "job_id": job_id, "family": job.family, "card_id": job.card_id, "seed": job.seed,
        "context": job.context, "fold": job.fold, "config": {field.name: getattr(job.config, field.name) for field in fields(job.config)},
        "dimensions": job.dimensions, "refit_dimensions": job.refit_dimensions,
        "expected_rows": [job.expected_fit_rows, job.expected_stop_rows, job.expected_refit_rows, job.expected_dev_rows],
        "excluded_game_keys_sha256": _sha256_bytes(json.dumps(sorted(job.excluded_game_keys)).encode("utf-8")),
        "acceptance_sha256": _file_sha256(acceptance) if acceptance.is_file() else None,
        "host": platform.node(), "device": _device_name(job.config.device),
    })


def _rng_state() -> dict[str, Any]:
    """Python and NumPy global RNG states in exact JSON form (torch states live in checkpoints)."""

    version, internal, gauss = random.getstate()
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": [version, list(internal), gauss], "numpy": [name, [int(value) for value in keys], int(position), int(has_gauss), float(cached)]}


def _set_rng_state(state: Mapping[str, Any]) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), int(position), int(has_gauss), float(cached)))


def _restore_checkpoint_rng(payload: Mapping[str, Any]) -> None:
    import torch

    # Generator states are CPU ByteTensors; a checkpoint loaded with map_location="cuda" moved them.
    torch.set_rng_state(payload["rng_state"].cpu())
    if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])


def _log_resume(output_dir: Path, event: Mapping[str, Any]) -> None:
    path = Path(output_dir) / "RESUME_LOG.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable({"utc": _utc_now(), "pid": os.getpid(), **event}), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_phase(output_dir: Path, name: str, identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _atomic_json(Path(output_dir) / name, {"version": VERSION, "identity": identity, **payload, "utc": _utc_now()})


def _read_phase(output_dir: Path, name: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """A completed-phase record, only when its identity and every file it names verify."""

    path = Path(output_dir) / name
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _log_resume(output_dir, {"ignored": name, "reason": f"unreadable: {error}"})
        return None
    if record.get("identity") != identity:
        _log_resume(output_dir, {"ignored": name, "reason": "identity differs"})
        return None
    for file_name, digest in (record.get("files") or {}).items():
        target = Path(output_dir) / file_name
        if not target.is_file() or _file_sha256(target) != digest:
            _log_resume(output_dir, {"ignored": name, "reason": f"{file_name} missing or changed"})
            return None
    return record


def _resume_entries(output_dir: Path, name: str, identity: Mapping[str, Any], *, epochs: int | None = None) -> list[dict[str, Any]]:
    """Epoch entries up to the last checkpoint whose bytes verify (a torn last file falls back one epoch)."""

    path = Path(output_dir) / name
    if not path.is_file():
        return []
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _log_resume(output_dir, {"ignored": name, "reason": f"unreadable: {error}"})
        return []
    if record.get("identity") != identity or (epochs is not None and record.get("epochs") != int(epochs)):
        _log_resume(output_dir, {"ignored": name, "reason": "identity or epoch target differs"})
        return []
    entries = list(record.get("entries") or [])
    if any(int(entry.get("epoch", -1)) != index for index, entry in enumerate(entries, start=1)):
        _log_resume(output_dir, {"ignored": name, "reason": "entries are not contiguous"})
        return []
    for count in range(len(entries), 0, -1):
        entry = entries[count - 1]
        target = Path(output_dir) / entry["checkpoint"]
        if target.is_file() and _file_sha256(target) == entry.get("checkpoint_sha256"):
            if count < len(entries):
                _log_resume(output_dir, {"phase": name, "fallback_from_epoch": len(entries), "to_epoch": count})
            return entries[:count]
    return []


class _LedgerSummary:
    """Consumption record of an epoch finished before a resume: its rows, batches and finish()."""

    def __init__(self, consumed: Mapping[str, Any]) -> None:
        self._consumed = dict(consumed)
        self.rows = int(consumed["rows"])
        self.batches = int(consumed["batches"])

    def finish(self, *, validate: bool = True) -> dict[str, Any]:
        return dict(self._consumed)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def raw_logloss_from_logits(logits: Any, target: Any) -> float:
    """Stable unit-row binary log loss from logits."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.shape != labels.shape or not np.isfinite(values).all() or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("logits/target must be aligned finite binary data")
    if not len(values):
        raise ValueError("logloss requires at least one row")
    return float(np.mean(np.logaddexp(0.0, values) - labels * values))


def raw_logloss_from_probability(probability: Any, target: Any) -> float:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        values = np.clip(values, 1e-12, 1.0 - 1e-12)
    return raw_logloss_from_logits(np.log(values) - np.log1p(-values), target)


def configure_runtime(seed: int, device: str = "cpu") -> None:
    """Configure deterministic local state without changing machine settings."""

    if seed != PHASE1_SEED:
        raise ValueError(f"phase-1 V runner is locked to seed {PHASE1_SEED}")
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts")
    import torch

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # ``highest`` selects IEEE float32 matmul where the installed torch API
    # supports it; the backend flags above cover older releases.
    try:
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, RuntimeError):
        pass


def _source_iter(source: Any, split: str, *, epoch: int, batch_size: int) -> Iterator[Any]:
    if source is None:
        return iter(())
    if hasattr(source, "iter_batches"):
        method = source.iter_batches
        attempts = (
            lambda: method(split=split, epoch=epoch, batch_size=batch_size),
            lambda: method(split, epoch=epoch, batch_size=batch_size),
            lambda: method(split, epoch=epoch),
            lambda: method(split),
        )
    elif hasattr(source, "iter_split"):
        method = source.iter_split
        attempts = (
            lambda: method(split=split, epoch=epoch, batch_size=batch_size),
            lambda: method(split, epoch=epoch, batch_size=batch_size),
            lambda: method(split),
        )
    elif callable(source):
        attempts = (
            lambda: source(split=split, epoch=epoch, batch_size=batch_size),
            lambda: source(split, epoch=epoch, batch_size=batch_size),
            lambda: source(split),
        )
    else:
        if epoch > 0 and iter(source) is source:
            raise ValueError(f"{split} source is a one-shot iterator and cannot support repeated epochs")
        return iter(source)
    last: Exception | None = None
    for attempt in attempts:
        try:
            return iter(attempt())
        except TypeError as exc:
            last = exc
    assert last is not None
    raise last


def _as_rows(value: Any, count: int, name: str, *, default: Any = None) -> tuple[Any, ...]:
    if value is None:
        return tuple(default for _ in range(count))
    if isinstance(value, str):
        return tuple(value for _ in range(count))
    array = np.asarray(value, dtype=object).reshape(-1)
    if len(array) == 1 and count != 1:
        return tuple(array[0] for _ in range(count))
    if len(array) != count:
        raise ValueError(f"metadata {name!r} must have one value per row")
    return tuple(array.tolist())


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _unpack_item(item: Any) -> tuple[Any, Any | None, Mapping[str, Any]]:
    if isinstance(item, Batch):
        return item.inputs, item.target, dict(item.metadata or {})
    if isinstance(item, Mapping):
        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("batch metadata must be a mapping")
        explicit = any(key in item for key in ("inputs", "batch", "model_input", "features"))
        if explicit:
            inputs = next(item[key] for key in ("inputs", "batch", "model_input", "features") if key in item)
            target_key = next((key for key in ("target", "y", "label", "W", "winning_team") if key in item), None)
            target = item[target_key] if target_key is not None else None
            metadata = dict(metadata)
            if target_key == "winning_team":
                metadata.setdefault("target_kind", "winning_team")
            elif target_key == "W":
                metadata.setdefault("target_kind", "W")
            return inputs, target, metadata
        target_key = next((key for key in ("target", "y", "label", "W", "winning_team") if key in item), None)
        target = item[target_key] if target_key is not None else None
        clean = dict(item)
        for key in ("target", "y", "label", "W", "winning_team", "metadata"):
            clean.pop(key, None)
        metadata = dict(metadata)
        if target_key == "winning_team":
            metadata.setdefault("target_kind", "winning_team")
        elif target_key == "W":
            metadata.setdefault("target_kind", "W")
        return clean, target, metadata
    if isinstance(item, tuple) and len(item) in (2, 3):
        inputs, target = item[:2]
        metadata = item[2] if len(item) == 3 else {}
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("batch metadata must be a mapping")
        return inputs, target, dict(metadata)
    raise TypeError("batches must be Batch, mapping, or (inputs,target[,metadata])")


def _normalise_target(target: Any, metadata: Mapping[str, Any], count: int) -> tuple[np.ndarray, dict[str, Any]]:
    source_kind = str(metadata.get("target_kind", "supervised" if target is not None else "")).lower()
    if target is None:
        target = _metadata_value(metadata, "target", "y", "label", "W", "winning_team")
        if target is None:
            raise ValueError("every training batch must provide a target or W/winning_team metadata")
        if "winning_team" in metadata and "target_kind" not in metadata:
            source_kind = "winning_team"
    labels = np.asarray(target)
    if labels.ndim == 0:
        labels = np.repeat(labels.reshape(1), count)
    labels = labels.reshape(-1)
    if len(labels) != count:
        raise ValueError("target must have one value per input row")
    if source_kind in {"winning_team", "team_id", "winner"}:
        labels = (labels.astype(np.int64, copy=False) == 100).astype(np.float64)
    else:
        labels = labels.astype(np.float64, copy=False)
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("targets must be finite binary W values")
    roles = tuple(str(value).upper() if value is not None else "" for value in _as_rows(_metadata_value(metadata, "roles", "role"), count, "role", default=""))
    swaps = tuple(bool(value) for value in _as_rows(_metadata_value(metadata, "swapped", "replacement_swap"), count, "swapped", default=False))
    if not bool(metadata.get("target_is_supervised", False)):
        labels = labels.copy()
        for index, (role, swapped) in enumerate(zip(roles, swaps)):
            if role == "TRAIN" and swapped:
                labels[index] = 1.0 - labels[index]
    result_metadata = dict(metadata)
    result_metadata["roles"] = roles
    result_metadata["swapped"] = swaps
    result_metadata["target_is_supervised"] = True
    return labels, result_metadata


def _normalise_batch(item: Any) -> tuple[Any, np.ndarray, dict[str, Any]]:
    inputs, target, metadata = _unpack_item(item)
    if isinstance(inputs, StructuredBatch):
        count = inputs.batch_size
    elif isinstance(inputs, Mapping):
        for candidate in ("table_csr", "table", "csr", "dense_features", "dense", "node_numeric", "node"):
            if candidate in inputs:
                count = int(np.shape(inputs[candidate])[0])
                break
        else:
            raise ValueError("cannot infer batch size from model inputs")
    else:
        shape = getattr(inputs, "shape", None)
        if shape is None or not shape:
            raise ValueError("cannot infer batch size from model inputs")
        count = int(shape[0])
    labels, metadata = _normalise_target(target, metadata, count)
    return inputs, labels, metadata


def _move_value(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, dtype=torch.float32) if value.is_floating_point() else value.to(device)
    if isinstance(value, np.ndarray):
        dtype = torch.float32 if value.dtype.kind == "f" else None
        return torch.as_tensor(value, device=device, dtype=dtype)
    if isinstance(value, BagInput):
        return BagInput(
            _move_value(value.indices, device),
            _move_value(value.counts, device),
            _move_value(value.offsets, device) if value.offsets is not None else None,
            _move_value(value.observed_count, device) if value.observed_count is not None else None,
        )
    if isinstance(value, StructuredBatch):
        updates = {field.name: _move_value(getattr(value, field.name), device) for field in fields(value)}
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return {key: _move_value(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(child, device) for child in value)
    if isinstance(value, list):
        return [_move_value(child, device) for child in value]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: _move_value(getattr(value, field.name), device) for field in fields(value)})
    return value


def _table_from_inputs(inputs: Any) -> Any:
    if isinstance(inputs, StructuredBatch):
        value = inputs.table_csr
    elif isinstance(inputs, Mapping):
        value = inputs.get("table_csr", inputs.get("table", inputs.get("csr")))
    else:
        value = inputs
    if value is None or getattr(value, "ndim", 2) != 2:
        raise ValueError("table family requires a two-dimensional CSR/table input")
    return value


def _is_neural(family: str) -> bool:
    return family in {"plain_mlp", "shared_slot_mlp", "team_set_mlp"}


def _model_logits(model: Any, family: str, inputs: Any, device: Any) -> Any:
    if _is_neural(family):
        import torch

        if family == "plain_mlp":
            # PlainMLP consumes root-provided dense features.  If the cache
            # exposes only CSR, densify this bounded batch at the model edge;
            # the full corpus is never materialised as one dense matrix.
            inputs = _probe_plain_inputs(inputs)
        moved = _move_value(inputs, device)
        result = model(moved)
        return result.reshape(-1)
    return np.asarray(model.predict_logits(_table_from_inputs(inputs)), dtype=np.float64).reshape(-1)


class _Ledger:
    def __init__(self, *, expected_rows: int | None, excluded_game_keys: Sequence[str], split: str, track_evidence: bool = True) -> None:
        self.expected_rows = expected_rows
        self.excluded_game_keys = frozenset(str(key) for key in excluded_game_keys)
        self.split = split
        self.track_evidence = bool(track_evidence)
        self.rows = 0
        self.batches = 0
        self.keys: list[Any] = []
        self.game_keys: list[str] = []
        self.roles: list[str] = []
        self.target_sum = 0.0

    def update(self, metadata: Mapping[str, Any], labels: np.ndarray) -> None:
        count = len(labels)
        raw_game_keys = _metadata_value(metadata, "game_keys", "game_key")
        raw_roles = _metadata_value(metadata, "roles", "role")
        if raw_game_keys is None or raw_roles is None:
            raise ValueError(f"{self.split} batches must carry per-row game_keys and roles")
        game_keys = tuple(str(value) for value in _as_rows(raw_game_keys, count, "game_key"))
        roles = tuple(str(value).upper() for value in _as_rows(raw_roles, count, "role"))
        if any(not key or key.lower() == "none" for key in game_keys):
            raise ValueError(f"{self.split} batches contain an empty game_key")
        if any(not role or role.lower() == "none" for role in roles):
            raise ValueError(f"{self.split} batches contain an empty role")
        if self.excluded_game_keys and any(key in self.excluded_game_keys for key in game_keys):
            raise ValueError(f"{self.split} source contains an excluded game key")
        self.rows += count
        self.batches += 1
        self.target_sum += float(np.sum(labels, dtype=np.float64))
        if self.track_evidence:
            self.game_keys.extend(key for key in game_keys if key)
        if self.track_evidence:
            self.roles.extend(roles)
        row_keys = _metadata_value(metadata, "row_keys", "nominal_indices", "nominal_index", "query_indices", "query_index")
        if self.track_evidence and row_keys is not None:
            self.keys.extend(_as_rows(row_keys, count, "row_key"))

    def finish(self, *, validate: bool = True) -> dict[str, Any]:
        if validate and self.expected_rows is not None and self.rows != int(self.expected_rows):
            raise ValueError(f"{self.split} consumed {self.rows} rows; expected {self.expected_rows}")
        payload = [str(value) for value in self.keys]
        return {
            "rows": self.rows,
            "batches": self.batches,
            "target_sum": self.target_sum,
            "key_count": len(payload),
            "keys_sha256": _sha256_bytes(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) if payload else None,
            "game_keys_sha256": _sha256_bytes(json.dumps(self.game_keys, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) if self.game_keys else None,
            "roles": sorted(set(self.roles)),
        }


class _Progress:
    def __init__(self, path: Path, *, job_id: str, total_rows: int | None) -> None:
        self.path = path
        self.job_id = job_id
        self.total_rows = total_rows
        self.started = time.monotonic()
        self.rows = 0
        self.batches = 0
        self.overall_rows = 0
        self.overall_batches = 0
        self.stage = "starting"
        self.status = "running"
        self.stage_started = self.started
        # Phase time already spent before a resume ({"FIT": s, "REFIT": s}); the
        # scheduler adds it so a resumed fit keeps one per-fit time limit.
        self.resumed: dict[str, float] = {}
        self._write()

    def _write(self) -> None:
        _atomic_json(self.path, {
            "version": VERSION,
            "job_id": self.job_id,
            "stage": self.stage,
            "epoch": None,
            "rows": self.rows,
            "batches": self.batches,
            "overall_rows": self.overall_rows,
            "overall_batches": self.overall_batches,
            "current": self.rows,
            "total": self.total_rows,
            "total_rows": self.total_rows,
            "elapsed_seconds": 0.0,
            "rate_rows_per_second": None,
            "eta_seconds": None,
            "eta_basis": None,
            "unmeasured_stages": ["fit", "stop", "refit", "selection"],
            "status": self.status,
            "last_error": None,
            "phase_resumed_seconds": dict(self.resumed),
            "updated_utc": _utc_now(),
        })

    def update(self, *, stage: str, epoch: int | None = None, rows: int = 0, batches: int = 0, total_rows: int | None = None, last_error: str | None = None) -> None:
        if stage != self.stage:
            self.stage_started = time.monotonic()
            self.rows = 0
            self.batches = 0
            self.total_rows = None
        self.stage = stage
        self.rows += int(rows)
        self.batches += int(batches)
        self.overall_rows += int(rows)
        self.overall_batches += int(batches)
        if total_rows is not None:
            self.total_rows = int(total_rows)
        elapsed = max(0.0, time.monotonic() - self.stage_started)
        rate = self.rows / elapsed if elapsed > 0 and self.rows else None
        eta = ((self.total_rows - self.rows) / rate) if rate and self.total_rows is not None and self.total_rows >= self.rows else None
        payload = {
            "version": VERSION,
            "job_id": self.job_id,
            "stage": stage,
            "epoch": epoch,
            "rows": self.rows,
            "batches": self.batches,
            "overall_rows": self.overall_rows,
            "overall_batches": self.overall_batches,
            "current": self.rows,
            "total": self.total_rows,
            "total_rows": self.total_rows,
            "elapsed_seconds": elapsed,
            "rate_rows_per_second": rate,
            "eta_seconds": eta,
            "eta_basis": "observed_rows" if eta is not None else None,
            "unmeasured_stages": [] if stage in {"done", "failed"} else ["future_refit", "future_selection"],
            "status": self.status,
            "last_error": last_error,
            "phase_resumed_seconds": dict(self.resumed),
            "updated_utc": _utc_now(),
        }
        _atomic_json(self.path, payload)

    def finish(self, status: str, *, error: str | None = None) -> None:
        self.status = status
        self.update(stage="done" if status == "completed" else "failed", last_error=error)


def _validate_scope_metadata(metadata: Mapping[str, Any], split: str, count: int) -> None:
    raw_roles = _metadata_value(metadata, "roles", "role")
    if raw_roles is None:
        raise ValueError(f"{split} batches must carry per-row roles")
    roles = tuple(str(value).upper() for value in _as_rows(raw_roles, count, "role"))
    if any(not role or role.lower() == "none" for role in roles):
        raise ValueError(f"{split} batches contain an empty role")
    expected = {
        "fit": {"TRAIN"},
        "stop": {"TRAIN"},
        "refit": {"TRAIN"},
        "dev": {"V_DEV"},
        "cal": {"V_CAL"},
        "test": {"TEST"},
    }.get(split)
    if expected is not None and any(role not in expected for role in roles):
        raise ValueError(f"{split} source has roles outside {sorted(expected)}")
    if any(role == "TEST" for role in roles) and split != "test":
        raise ValueError("TEST rows cannot be consumed for fitting, STOP, REFIT, DEV, or calibration")


def _fit_neural(
    model: Any,
    family: str,
    source: Any,
    stop_source: Any,
    *,
    config: RunnerConfig,
    progress: _Progress,
    output_dir: Path,
    seed: int,
    expected_fit_rows: int | None,
    expected_stop_rows: int | None,
    excluded_game_keys: Sequence[str],
    identity: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any], _Ledger]:
    import torch

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=float(getattr(model, "weight_decay", 0.0)), betas=(0.9, 0.999), eps=1e-8)
    best_loss = math.inf
    best_epoch = 0
    bad_epochs = 0
    epoch_metrics: list[dict[str, Any]] = []
    fit_ledger: _Ledger | _LedgerSummary | None = None
    stop_ledger: _Ledger | _LedgerSummary | None = None
    stop_total = expected_stop_rows
    if stop_source is None:
        raise ValueError("neural candidates require a replayable STOP source")
    entries = _resume_entries(output_dir, "RESUME_FIT.json", identity) if identity is not None else []
    phase_seconds = 0.0
    if entries:
        # Continue after the last verified epoch: weights, AdamW moments and all
        # RNG states are restored, and STOP bookkeeping is replayed from the
        # recorded epoch metrics with the same rule as below.
        payload = torch.load(output_dir / entries[-1]["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _restore_checkpoint_rng(payload)
        del payload
        for entry in entries:
            metric = entry["metric"]
            epoch_metrics.append(metric)
            if metric["stop_logloss"] < best_loss - config.min_delta:
                best_loss, best_epoch, bad_epochs = float(metric["stop_logloss"]), int(metric["epoch"]), 0
            else:
                bad_epochs += 1
        fit_ledger = _LedgerSummary(epoch_metrics[-1]["fit_consumed"])
        stop_ledger = _LedgerSummary(epoch_metrics[-1]["stop_consumed"])
        phase_seconds = float(entries[-1]["phase_seconds"])
        _restore_best_checkpoint(output_dir, entries, best_epoch, device)
        progress.resumed["FIT"] = phase_seconds
        _log_resume(output_dir, {"phase": "fit", "resumed_after_epoch": len(entries), "phase_seconds": phase_seconds})
    segment = time.monotonic()
    first_epoch = len(entries) + 1 if bad_epochs < config.patience else config.max_epochs + 1
    for epoch in range(first_epoch, config.max_epochs + 1):
        model.train()
        epoch_rows = 0
        epoch_batches = 0
        epoch_ledger = _Ledger(expected_rows=expected_fit_rows, excluded_game_keys=excluded_game_keys, split="fit")
        for item in _source_iter(source, "fit", epoch=epoch - 1, batch_size=config.batch_size):
            inputs, labels, metadata = _normalise_batch(item)
            if len(labels) > config.batch_size:
                raise ValueError("a batch exceeds the locked batch size 1024")
            _validate_scope_metadata(metadata, "fit", len(labels))
            epoch_ledger.update(metadata, labels)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model, family, inputs, device)
            if logits.shape[0] != target.shape[0] or not bool(torch.isfinite(logits).all()):
                raise ValueError("model produced invalid training logits")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite neural training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_rows += len(labels)
            epoch_batches += 1
            progress.update(stage="fit", epoch=epoch, rows=len(labels), batches=1, total_rows=(expected_fit_rows * config.max_epochs if expected_fit_rows is not None else None))
        if epoch_batches == 0:
            raise ValueError("FIT source produced no batches")
        epoch_consumed = epoch_ledger.finish()
        stop_loss, stop_ledger = _score_source(model, family, stop_source, "stop", device, config, expected_rows=stop_total, excluded_game_keys=excluded_game_keys, progress=progress, epoch=epoch)
        stop_consumed = stop_ledger.finish(validate=False)
        metric = {"epoch": epoch, "fit_rows": epoch_rows, "fit_batches": epoch_batches, "stop_logloss": stop_loss, "stop_rows": stop_ledger.rows, "fit_consumed": epoch_consumed, "stop_consumed": stop_consumed}
        epoch_metrics.append(metric)
        fit_ledger = epoch_ledger
        checkpoint = f"checkpoint_epoch_{epoch:02d}.pt"
        _save_neural_checkpoint(output_dir / checkpoint, model, optimizer, seed=seed, epoch=epoch, chosen_epoch=best_epoch, metrics=metric, consumed=epoch_consumed)
        if stop_loss < best_loss - config.min_delta:
            best_loss = stop_loss
            best_epoch = epoch
            bad_epochs = 0
            _save_neural_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, seed=seed, epoch=epoch, chosen_epoch=epoch, metrics=metric, consumed=epoch_consumed)
        else:
            bad_epochs += 1
        now = time.monotonic()
        phase_seconds, segment = phase_seconds + (now - segment), now
        if identity is not None:
            # Commit point of the epoch: written only after its checkpoint is durable.
            entries.append({"epoch": epoch, "metric": metric, "checkpoint": checkpoint,
                            "checkpoint_sha256": _file_sha256(output_dir / checkpoint), "phase_seconds": phase_seconds})
            _atomic_json(output_dir / "RESUME_FIT.json", {"version": VERSION, "identity": identity, "phase": "fit", "entries": entries})
        if bad_epochs >= config.patience:
            break
    if best_epoch <= 0:
        raise RuntimeError("STOP did not produce a finite improving epoch")
    if fit_ledger is None:
        raise RuntimeError("FIT source produced no epochs")
    return best_epoch, {"epochs": epoch_metrics, "best_logloss": best_loss, "rows_per_epoch": fit_ledger.rows, "batches_per_epoch": fit_ledger.batches}, {"rows": stop_ledger.rows, "logloss": best_loss, "consumed": stop_ledger.finish(validate=False)}, fit_ledger


def _score_source(
    model: Any,
    family: str,
    source: Any,
    split: str,
    device: Any,
    config: RunnerConfig,
    *,
    expected_rows: int | None,
    excluded_game_keys: Sequence[str],
    progress: _Progress | None = None,
    epoch: int | None = None,
) -> tuple[float, _Ledger]:
    import torch

    model.eval() if _is_neural(family) else None
    ledger = _Ledger(expected_rows=expected_rows, excluded_game_keys=excluded_game_keys, split=split)
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    with torch.no_grad() if _is_neural(family) else _nullcontext():
        for item in _source_iter(source, split, epoch=0, batch_size=config.batch_size):
            inputs, labels, metadata = _normalise_batch(item)
            if len(labels) > config.batch_size:
                raise ValueError("a batch exceeds the locked batch size 1024")
            _validate_scope_metadata(metadata, split, len(labels))
            ledger.update(metadata, labels)
            values = _model_logits(model, family, inputs, device)
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(values) != len(labels) or not np.isfinite(values).all():
                raise ValueError(f"{split} model output is invalid")
            logits_parts.append(values)
            labels_parts.append(labels)
            if progress is not None:
                progress.update(stage=split, epoch=epoch, rows=len(labels), batches=1, total_rows=expected_rows)
    if not logits_parts:
        raise ValueError(f"{split} source produced no batches")
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(labels_parts)
    ledger.finish()
    return raw_logloss_from_logits(logits, labels), ledger


class _nullcontext:
    def __enter__(self) -> "_nullcontext":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _save_neural_checkpoint(path: Path, model: Any, optimizer: Any, *, seed: int, epoch: int, chosen_epoch: int, metrics: Mapping[str, Any], consumed: Mapping[str, Any]) -> None:
    import torch

    payload = {
        "version": VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "seed": seed,
        "epoch": epoch,
        "chosen_epoch": chosen_epoch,
        "metrics": dict(metrics),
        "consumed": dict(consumed),
    }
    _write_torch(path, payload)


def _write_torch(path: Path, payload: Mapping[str, Any]) -> None:
    """torch.save made durable before it is published (survives a power loss)."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_retry(pending, path)


def _restore_best_checkpoint(output_dir: Path, entries: Sequence[Mapping[str, Any]], best_epoch: int, device: Any) -> None:
    """After a fallback resume, ``checkpoint_best.pt`` must again hold the best verified epoch."""

    import torch

    if best_epoch <= 0:
        return
    path = Path(output_dir) / "checkpoint_best.pt"
    if path.is_file():
        try:
            if int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1)) == int(best_epoch):
                return
        except Exception:  # noqa: BLE001 - a torn file is rewritten below
            pass
    entry = entries[int(best_epoch) - 1]
    source = Path(output_dir) / entry["checkpoint"]
    if not source.is_file() or _file_sha256(source) != entry.get("checkpoint_sha256"):
        _log_resume(output_dir, {"phase": "fit", "best_checkpoint": "not restorable", "best_epoch": best_epoch})
        return
    payload = torch.load(source, weights_only=False)  # tensors back on the devices they were saved from
    payload["chosen_epoch"] = int(best_epoch)  # the only field the best copy differs in
    _write_torch(path, payload)
    _log_resume(output_dir, {"phase": "fit", "best_checkpoint": "restored", "best_epoch": best_epoch})


def _table_dtype(family: str) -> Any:
    return np.float64 if family == "logistic" else np.float32


def _stack_csr(parts: list[Any], dtype: Any) -> Any:
    """Row-stack CSR batches into one matrix, releasing each batch once copied.

    Equivalent to ``sparse.vstack(parts, format="csr").astype(dtype)`` but the
    resident peak stays near one matrix instead of batches + stack + cast.
    """
    from scipy import sparse

    if not parts:
        raise ValueError("no CSR batches to stack")
    columns = int(parts[0].shape[1])
    if any(int(part.shape[1]) != columns for part in parts):
        raise ValueError("CSR batches disagree on table width")
    rows = sum(int(part.shape[0]) for part in parts)
    nnz = sum(int(part.nnz) for part in parts)
    index_dtype = np.int32 if max(nnz, columns) < np.iinfo(np.int32).max else np.int64
    data = np.empty(nnz, dtype=dtype)
    indices = np.empty(nnz, dtype=index_dtype)
    indptr = np.empty(rows + 1, dtype=index_dtype)
    indptr[0] = 0
    row = value = 0
    parts.reverse()
    while parts:
        part = parts.pop()
        count, length = int(part.shape[0]), int(part.nnz)
        data[value:value + length] = part.data
        indices[value:value + length] = part.indices
        indptr[row + 1:row + count + 1] = part.indptr[1:] + value
        row, value = row + count, value + length
        del part
    return sparse.csr_matrix((data, indices, indptr), shape=(rows, columns), copy=False)


def _fit_table(model: Any, family: str, source: Any, stop_source: Any | None, *, config: RunnerConfig, progress: _Progress, output_dir: Path, expected_fit_rows: int | None, expected_stop_rows: int | None, excluded_game_keys: Sequence[str]) -> tuple[int | None, dict[str, Any], dict[str, Any], _Ledger]:
    from scipy import sparse

    fit_ledger = _Ledger(expected_rows=expected_fit_rows, excluded_game_keys=excluded_game_keys, split="fit")
    matrices: list[Any] = []
    labels: list[np.ndarray] = []
    for item in _source_iter(source, "fit", epoch=0, batch_size=config.batch_size):
        inputs, target, metadata = _normalise_batch(item)
        if len(target) > config.batch_size:
            raise ValueError("a batch exceeds the locked batch size 1024")
        _validate_scope_metadata(metadata, "fit", len(target))
        matrix = _table_from_inputs(inputs)
        if int(matrix.shape[0]) != len(target):
            raise ValueError("table rows and target rows differ")
        fit_ledger.update(metadata, target)
        matrices.append(sparse.csr_matrix(matrix, dtype=np.float32))
        labels.append(target)
        progress.update(stage="fit", epoch=None, rows=len(target), batches=1, total_rows=expected_fit_rows)
    if not matrices:
        raise ValueError("FIT source produced no batches")
    # Row/exclusion accounting is checked before the long solver call, not after it.
    fit_ledger.finish()
    x_fit = _stack_csr(matrices, _table_dtype(family))
    y_fit = np.concatenate(labels)
    if family == "lightgbm":
        if stop_source is None:
            raise ValueError("LightGBM candidates require a replayable STOP source")
        stop_ledger = _Ledger(expected_rows=expected_stop_rows, excluded_game_keys=excluded_game_keys, split="stop")
        stop_matrices: list[Any] = []
        stop_labels: list[np.ndarray] = []
        for item in _source_iter(stop_source, "stop", epoch=0, batch_size=config.batch_size):
            inputs, target, metadata = _normalise_batch(item)
            _validate_scope_metadata(metadata, "stop", len(target))
            stop_ledger.update(metadata, target)
            stop_matrices.append(sparse.csr_matrix(_table_from_inputs(inputs), dtype=np.float32))
            stop_labels.append(target)
            progress.update(stage="stop", rows=len(target), batches=1, total_rows=expected_stop_rows)
        if not stop_matrices:
            raise ValueError("STOP source produced no batches")
        stop_consumed = stop_ledger.finish()
        x_stop, y_stop = _stack_csr(stop_matrices, _table_dtype(family)), np.concatenate(stop_labels)
        progress.update(stage="fit_solver", total_rows=None)
        try:
            from lightgbm import early_stopping

            model.estimator.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)], callbacks=[early_stopping(config.lgb_patience, verbose=False)])
        except TypeError:
            # A test double or older adapter may expose only the base sklearn
            # signature; scientific runs use the LightGBM branch above.
            model.estimator.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)])
        best_iteration = getattr(model.estimator, "best_iteration_", None)
        chosen = int(best_iteration) if best_iteration is not None else None
        best_score = None
        best_score_payload = getattr(model.estimator, "best_score_", None)
        if isinstance(best_score_payload, Mapping):
            for values in best_score_payload.values():
                if isinstance(values, Mapping) and values:
                    best_score = next((float(value) for key, value in values.items() if str(key).lower() in {"binary_logloss", "logloss"}), None)
                    if best_score is not None:
                        break
        fit_metrics = {"fit_rows": fit_ledger.rows, "fit_batches": fit_ledger.batches, "stop_rows": stop_ledger.rows, "best_iteration": chosen, "stop_logloss": best_score, "consumed": fit_ledger.finish(validate=False)}
        stop_metrics = {"rows": stop_ledger.rows, "best_iteration": chosen, "consumed": stop_consumed}
    else:
        progress.update(stage="fit_solver", total_rows=None)
        model.fit(x_fit, y_fit)
        chosen = None
        fit_metrics = {"fit_rows": fit_ledger.rows, "fit_batches": fit_ledger.batches, "consumed": fit_ledger.finish(validate=False)}
        stop_metrics = {}
    model_path = output_dir / "model_fit.pkl"
    fit_ledger.finish()
    _atomic_pickle(model_path, model)
    return chosen, fit_metrics, stop_metrics, fit_ledger


def _fit_refit_neural(model: Any, family: str, source: Any, *, epochs: int, config: RunnerConfig, progress: _Progress, expected_rows: int | None, excluded_game_keys: Sequence[str], output_dir: Path, seed: int, identity: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], _Ledger]:
    import torch

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=float(getattr(model, "weight_decay", 0.0)), betas=(0.9, 0.999), eps=1e-8)
    ledger = _Ledger(expected_rows=None, excluded_game_keys=excluded_game_keys, split="refit", track_evidence=False)
    metrics: list[dict[str, Any]] = []
    entries = _resume_entries(output_dir, "RESUME_REFIT.json", identity, epochs=int(epochs)) if identity is not None else []
    phase_seconds = 0.0
    if entries:
        payload = torch.load(output_dir / entries[-1]["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _restore_checkpoint_rng(payload)
        del payload
        metrics = [entry["metric"] for entry in entries]
        # The cross-epoch ledger keeps counts and the exact running target sum.
        state = entries[-1]["ledger"]
        ledger.rows, ledger.batches, ledger.target_sum = int(state["rows"]), int(state["batches"]), float(state["target_sum"])
        phase_seconds = float(entries[-1]["phase_seconds"])
        progress.resumed["REFIT"] = phase_seconds
        _log_resume(output_dir, {"phase": "refit", "resumed_after_epoch": len(entries), "phase_seconds": phase_seconds})
    segment = time.monotonic()
    for epoch in range(len(entries) + 1, int(epochs) + 1):
        model.train()
        epoch_rows = 0
        epoch_loss = 0.0
        batches = 0
        epoch_ledger = _Ledger(expected_rows=expected_rows, excluded_game_keys=excluded_game_keys, split="refit")
        for item in _source_iter(source, "refit", epoch=epoch - 1, batch_size=config.batch_size):
            inputs, labels, metadata = _normalise_batch(item)
            _validate_scope_metadata(metadata, "refit", len(labels))
            ledger.update(metadata, labels)
            epoch_ledger.update(metadata, labels)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model, family, inputs, device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite REFIT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_rows += len(labels)
            epoch_loss += float(loss.detach().cpu()) * len(labels)
            batches += 1
            progress.update(stage="refit", epoch=epoch, rows=len(labels), batches=1, total_rows=(expected_rows * epochs if expected_rows is not None else None))
        if batches == 0:
            raise ValueError("REFIT source produced no batches")
        epoch_consumed = epoch_ledger.finish()
        metrics.append({"epoch": epoch, "rows": epoch_rows, "batches": batches, "fit_logloss": epoch_loss / epoch_rows, "consumed": epoch_consumed})
        checkpoint = f"refit_checkpoint_epoch_{epoch:02d}.pt"
        _save_neural_checkpoint(output_dir / checkpoint, model, optimizer, seed=seed, epoch=epoch, chosen_epoch=epochs, metrics=metrics[-1], consumed=epoch_consumed)
        now = time.monotonic()
        phase_seconds, segment = phase_seconds + (now - segment), now
        if identity is not None:
            entries.append({"epoch": epoch, "metric": metrics[-1], "checkpoint": checkpoint,
                            "checkpoint_sha256": _file_sha256(output_dir / checkpoint), "phase_seconds": phase_seconds,
                            "ledger": {"rows": ledger.rows, "batches": ledger.batches, "target_sum": ledger.target_sum}})
            _atomic_json(output_dir / "RESUME_REFIT.json", {"version": VERSION, "identity": identity, "phase": "refit",
                                                             "epochs": int(epochs), "entries": entries})
    _save_neural_checkpoint(output_dir / "model_refit.pt", model, optimizer, seed=seed, epoch=epochs, chosen_epoch=epochs, metrics={"epochs": metrics}, consumed=ledger.finish(validate=False))
    if expected_rows is not None and ledger.rows != int(expected_rows) * int(epochs):
        raise ValueError(f"refit consumed {ledger.rows} rows across {epochs} epochs; expected {int(expected_rows) * int(epochs)}")
    return {"epochs": metrics, "rows": ledger.rows}, ledger


def _fit_refit_table(model: Any, source: Any, *, family: str, chosen_epoch: int | None, config: RunnerConfig, progress: _Progress, expected_rows: int | None, excluded_game_keys: Sequence[str], output_dir: Path) -> tuple[dict[str, Any], _Ledger]:
    # Logistic/LGBM fit has no epoch-valued STOP; REFIT is a fresh estimator
    # fit on the REFIT_TRAIN cache and therefore remains a distinct artifact.
    from scipy import sparse

    ledger = _Ledger(expected_rows=expected_rows, excluded_game_keys=excluded_game_keys, split="refit")
    matrices: list[Any] = []
    labels: list[np.ndarray] = []
    for item in _source_iter(source, "refit", epoch=0, batch_size=config.batch_size):
        inputs, target, metadata = _normalise_batch(item)
        _validate_scope_metadata(metadata, "refit", len(target))
        ledger.update(metadata, target)
        matrices.append(sparse.csr_matrix(_table_from_inputs(inputs), dtype=np.float32))
        labels.append(target)
        progress.update(stage="refit", rows=len(target), batches=1, total_rows=expected_rows)
    if not matrices:
        raise ValueError("REFIT source produced no batches")
    # Fail on a truncated or contaminated REFIT source before the solver runs.
    ledger.finish()
    matrix = _stack_csr(matrices, _table_dtype(family))
    if family == "lightgbm":
        if chosen_epoch is None or int(chosen_epoch) <= 0:
            raise ValueError("LightGBM REFIT requires a positive STOP-selected round count")
        model.estimator.set_params(n_estimators=int(chosen_epoch))
    progress.update(stage="refit_solver", total_rows=None)
    model.fit(matrix, np.concatenate(labels))
    consumed = ledger.finish()
    _atomic_pickle(output_dir / "model_refit.pkl", model)
    return {"rows": ledger.rows, "batches": ledger.batches, "consumed": consumed}, ledger


def _fresh_model(job: CandidateJob, *, dimensions: Mapping[str, Any] | int | None = None) -> Any:
    return build_locked_model(
        job.family,
        job.card_id,
        acceptance_path=job.acceptance_path,
        seed=job.seed,
        dimensions=job.dimensions if dimensions is None else dimensions,
        cpu_threads=job.config.cpu_threads,
    )


def _probe_model(family: str, card_id: str, *, dimensions: Mapping[str, Any] | int | None, seed: int) -> Any:
    """Build an unfit canonical probe model without claiming readiness."""

    import torch

    torch.manual_seed(int(seed))
    key = str(family).lower()
    if key == "plain_mlp":
        if dimensions is None or isinstance(dimensions, Mapping):
            raise ValueError("plain_mlp probe requires the canonical table width")
        return build_model(key, card=card_id, input_dim=int(dimensions))
    if key in {"shared_slot_mlp", "team_set_mlp"}:
        if not isinstance(dimensions, Mapping) or not isinstance(dimensions.get("vocab_sizes"), Mapping):
            raise ValueError("structured probe requires canonical vocab_sizes")
        from studies.v_model_fit_input_20261003.common import BAG_LAYOUT, BAG_NAMESPACES, CATEGORY_LAYOUT, CATEGORY_NAMESPACES

        return build_model(
            key,
            card=card_id,
            node_numeric_dim=102,
            team_numeric_dim=8,
            global_numeric_dim=76,
            vocab_sizes=dict(dimensions["vocab_sizes"]),
            category_namespaces=dict(CATEGORY_NAMESPACES),
            bag_namespaces=dict(BAG_NAMESPACES),
            category_layout=dict(CATEGORY_LAYOUT),
            bag_layout=dict(BAG_LAYOUT),
            include_side_flag=False,
        )
    if key == "lightgbm":
        return build_model(key, card=card_id, random_state=seed, num_threads=1)
    return build_model(key, card=card_id, random_state=seed)


def _family_dimensions(dimensions: Mapping[str, Any] | int | None, family: str) -> Mapping[str, Any] | int | None:
    """Resolve a shared CLI dimensions object to one family contract."""

    if not isinstance(dimensions, Mapping):
        return dimensions
    key = str(family).lower()
    if key in dimensions and isinstance(dimensions[key], (Mapping, int, np.integer)):
        return dimensions[key]
    if key == "plain_mlp" and "plain_input_dim" in dimensions:
        return int(dimensions["plain_input_dim"])
    if key == "plain_mlp" and "table_width" in dimensions:
        return int(dimensions["table_width"])
    if key in {"shared_slot_mlp", "team_set_mlp"} and isinstance(dimensions.get("structured"), Mapping):
        return dimensions["structured"]
    return dimensions


def _probe_plain_inputs(inputs: Any) -> Any:
    """Batch-densify only the current PlainMLP batch when the cache supplies CSR."""

    if isinstance(inputs, StructuredBatch):
        if inputs.dense_features is not None:
            return inputs
        table = inputs.table_csr
        if table is None:
            raise ValueError("plain_mlp probe requires dense_features or table_csr")
        return replace(inputs, dense_features=np.asarray(table.toarray(), dtype=np.float32))
    if isinstance(inputs, Mapping):
        if inputs.get("dense_features", inputs.get("dense")) is not None:
            return inputs
        table = inputs.get("table_csr", inputs.get("table", inputs.get("csr")))
        if table is None:
            raise ValueError("plain_mlp probe requires dense_features or table_csr")
        result = dict(inputs)
        result["dense_features"] = np.asarray(table.toarray(), dtype=np.float32)
        return result
    return inputs


def _resource_sample(phase: str, *, process: Any, limits: Mapping[str, int], device: str) -> dict[str, Any]:
    import shutil

    try:
        import psutil

        memory = psutil.virtual_memory()
        rss = int(process.memory_info().rss)
        available = int(memory.available)
    except ImportError:  # pragma: no cover - psutil is part of the locked runtime
        rss = 0
        available = 0
    free_disk = int(shutil.disk_usage(Path.cwd()).free)
    sample = {"phase": phase, "device": device, "rss_bytes": rss, "available_ram_bytes": available, "free_disk_bytes": free_disk}
    if rss and rss > int(limits.get("rss_bytes", 16 << 30)):
        raise RuntimeError(f"resource limit exceeded at {phase}: RSS={rss}")
    if free_disk < int(limits.get("free_disk_bytes", 8 << 30)):
        raise RuntimeError(f"resource limit exceeded at {phase}: free disk={free_disk}")
    if available and available < int(limits.get("available_ram_bytes", 8 << 30)):
        raise RuntimeError(f"resource limit exceeded at {phase}: available RAM={available}")
    if str(device).startswith("cuda"):
        import torch

        free_vram, total_vram = torch.cuda.mem_get_info(torch.device(device))
        sample.update({"free_vram_bytes": int(free_vram), "total_vram_bytes": int(total_vram)})
        if free_vram < int(limits.get("free_vram_bytes", 2 << 30)):
            raise RuntimeError(f"resource limit exceeded at {phase}: free VRAM={free_vram}")
    return sample


def run_full_input_consumption_gate(
    source: Any,
    *,
    output_dir: str | Path,
    dimensions: Mapping[str, Any] | int | None,
    expected_rows: int | None = None,
    families: Sequence[str] = FAMILIES,
    card_id: str | int = 0,
    seed: int = 20261001,
    device: str = "cpu",
    require_full_batch: bool = True,
    backward_probe: bool = True,
    resource_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Consume a replayable full-input source through every V family.

    This is a probe/resource gate.  It performs real forward checks over every
    supplied row and one bounded optimizer/reload check for neural families,
    but it writes ``training_ready=false`` and never calls
    :func:`build_locked_model`.  The guarded candidate runner remains the only
    path that can create production fitted models.
    """

    if seed != PHASE1_SEED:
        raise ValueError(f"phase-1 full-input gate is locked to seed {PHASE1_SEED}")
    if not families:
        raise ValueError("at least one family is required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    limits = dict(resource_limits or {})
    progress = _Progress(root / "progress.json", job_id=f"consumption_gate_seed{seed}", total_rows=expected_rows)
    try:
        import torch

        configure_runtime(seed, device)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for full-input gate but CUDA is unavailable")
        try:
            import psutil
        except ImportError:  # pragma: no cover - locked runtime includes psutil
            process = None
        else:
            process = psutil.Process()
        if process is None:
            raise RuntimeError("full-input resource gate requires psutil")
        resource_samples = []
        resource_samples.append(_resource_sample("start", process=process, limits=limits, device=device))
        records: list[dict[str, Any]] = []
        for family in families:
            key = str(family).lower()
            card = family_cards(key)[int(card_id)] if isinstance(card_id, (int, np.integer)) else next(card for card in family_cards(key) if card["id"] == str(card_id))
            family_dimensions = _family_dimensions(dimensions, key)
            model = _probe_model(key, card["id"], dimensions=family_dimensions, seed=seed)
            model_device = torch.device(device)
            if _is_neural(key):
                model.to(model_device)
            first: tuple[Any, np.ndarray, dict[str, Any]] | None = None
            requested_card = card["id"]
            nonconverged: list[dict[str, Any]] = []
            ledger = _Ledger(expected_rows=expected_rows, excluded_game_keys=tuple(getattr(source, "excluded_game_keys", ())), split="gate")
            logits: list[np.ndarray] = []
            started = time.perf_counter()
            optimizer = None
            for item in _source_iter(source, "gate", epoch=0, batch_size=BATCH_SIZE):
                inputs, labels, metadata = _normalise_batch(item)
                if key in {"shared_slot_mlp", "team_set_mlp"} and isinstance(family_dimensions, Mapping):
                    if isinstance(inputs, Mapping):
                        inputs = StructuredBatch.coerce(inputs)
                    if not isinstance(inputs, StructuredBatch):
                        raise TypeError(f"{key} full-input gate requires StructuredBatch inputs")
                    validate_indexed_batch(inputs, family_dimensions.get("vocab_sizes", {}))
                if len(labels) > BATCH_SIZE:
                    raise ValueError("full-input gate received a batch larger than 1024")
                if require_full_batch and first is None and len(labels) != BATCH_SIZE:
                    raise ValueError("full-input gate requires an initial batch of exactly 1024 rows")
                if first is None and set(np.unique(labels).tolist()) != {0.0, 1.0}:
                    raise ValueError(f"{key} full-input probe requires both target classes in the initial batch")
                ledger.update(metadata, labels)
                if first is None:
                    first = (inputs, labels, metadata)
                    if key == "plain_mlp":
                        inputs = _probe_plain_inputs(inputs)
                    if _is_neural(key):
                        model.train()
                        initial_logits = _model_logits(model, key, inputs, model_device)
                        if not bool(torch.isfinite(initial_logits).all()):
                            raise ValueError(f"nonfinite {key} initial logits")
                        if backward_probe:
                            optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL_LR, weight_decay=float(getattr(model, "weight_decay", 0.0)), betas=(0.9, 0.999), eps=1e-8)
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(initial_logits, torch.as_tensor(labels, dtype=torch.float32, device=model_device))
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                            optimizer.step()
                        model.eval()
                    else:
                        matrix = _table_from_inputs(inputs)
                        try:
                            model.fit(matrix, labels)
                        except LogisticNonconvergenceError as error:
                            # The probe only has to consume the input.  A 1,024-row logistic probe can be
                            # too ill-conditioned to converge (oof0, 2026-10-05); the next canonical card
                            # that converges is used instead and the substitution is recorded.
                            nonconverged.append({"card_id": card["id"], "convergence": error.convergence})
                            model, card = _converging_logistic_probe(key, card, family_dimensions, seed, matrix, labels, nonconverged)
                    resource_samples.append(_resource_sample(f"family:{key}:first_batch", process=process, limits=limits, device=device))
                probe_inputs = _probe_plain_inputs(inputs) if key == "plain_mlp" else inputs
                with torch.no_grad() if _is_neural(key) else _nullcontext():
                    values = _model_logits(model, key, probe_inputs, model_device)
                if hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                values = np.asarray(values, dtype=np.float64).reshape(-1)
                if len(values) != len(labels) or not np.isfinite(values).all():
                    raise ValueError(f"{key} produced nonfinite or misaligned full-input outputs")
                logits.append(values)
                progress.update(stage=f"consumption:{key}:{card['id']}", rows=len(labels), batches=1, total_rows=expected_rows)
            if first is None:
                raise ValueError(f"{key} full-input source produced no rows")
            ledger.finish()
            elapsed = max(0.0, time.perf_counter() - started)
            payload = {"family": key, "card_id": card["id"], "rows": ledger.rows, "batches": ledger.batches, "elapsed_seconds": elapsed, "rate_rows_per_second": ledger.rows / elapsed if elapsed else None, "input_output_finite": True, "probe_only": True, "reload_equal": None}
            if nonconverged:
                payload.update(requested_card_id=requested_card, probe_nonconverged=nonconverged)
            if expected_rows is not None and ledger.rows != int(expected_rows):
                raise ValueError(f"{key} consumed {ledger.rows} rows; expected {expected_rows}")
            # Reload equality is deliberately checked on the first cached
            # batch only; the source was already fully consumed above.
            if _is_neural(key):
                state = copy.deepcopy(model.state_dict())
                restored = _probe_model(key, card["id"], dimensions=family_dimensions, seed=seed)
                restored.to(model_device)
                # PlainMLP initializes lazily only when dimensions were omitted;
                # the probe dimensions are explicit in production.
                if key == "plain_mlp":
                    probe_input = _probe_plain_inputs(first[0])
                else:
                    probe_input = first[0]
                restored.eval()
                with torch.no_grad():
                    _model_logits(restored, key, probe_input, model_device)
                restored.load_state_dict(state, strict=True)
                model.eval()
                with torch.no_grad():
                    left = _model_logits(model, key, probe_input, model_device)
                    right = _model_logits(restored, key, probe_input, model_device)
                left = left.detach().cpu().numpy() if hasattr(left, "detach") else np.asarray(left)
                right = right.detach().cpu().numpy() if hasattr(right, "detach") else np.asarray(right)
                payload["reload_equal"] = bool(np.array_equal(left, right))
                torch.save({"version": VERSION, "family": key, "card_id": card["id"], "probe_only": True, "state_dict": state}, root / f"PROBE_{key}.pt")
            else:
                _atomic_pickle(root / f"PROBE_{key}.pkl", model)
                with (root / f"PROBE_{key}.pkl").open("rb") as stream:
                    restored = pickle.load(stream)
                probe_input = first[0]
                payload["reload_equal"] = bool(np.array_equal(model.predict_logits(_table_from_inputs(probe_input)), restored.predict_logits(_table_from_inputs(probe_input))))
            records.append(payload)
            resource_samples.append(_resource_sample(f"family:{key}", process=process, limits=limits, device=device))
        result = {"version": VERSION, "status": "passed", "probe_only": True, "models_trained": False, "training_ready": False, "full_corpus_model_consumption_verified": True, "full_batch_resource_verified": True, "scientific_runner_verified": False, "families": records, "resource_samples": resource_samples, "device": device, "precision": "neural float32; AMP/TF32 disabled; table/logistic adapter precision preserved", "updated_utc": _utc_now()}
        _atomic_json(root / "CONSUMPTION_GATE.json", result)
        progress.finish("completed")
        return result
    except Exception as exc:
        failure = {"version": VERSION, "status": "failed", "probe_only": True, "training_ready": False, "error": {"type": type(exc).__name__, "message": str(exc)}, "updated_utc": _utc_now()}
        _atomic_json(root / "FAILED.json", failure)
        progress.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _converging_logistic_probe(family: str, failed: Mapping[str, Any], dimensions: Any, seed: int, matrix: Any, labels: np.ndarray,
                               nonconverged: list[dict[str, Any]]) -> tuple[Any, Mapping[str, Any]]:
    """Fit the gate's logistic probe with the other canonical cards in order until one converges.

    Gate probes never become candidates; candidate convergence rules are untouched.
    """

    for card in family_cards(family):
        if card["id"] == failed["id"] or any(row["card_id"] == card["id"] for row in nonconverged):
            continue
        model = _probe_model(family, card["id"], dimensions=dimensions, seed=seed)
        try:
            model.fit(matrix, labels)
        except LogisticNonconvergenceError as error:
            nonconverged.append({"card_id": card["id"], "convergence": error.convergence})
            continue
        return model, card
    raise LogisticNonconvergenceError({"probe": "no canonical logistic card converged on the first gate batch", "cards": nonconverged})


verify_full_input_consumption = run_full_input_consumption_gate
run_resource_consumption_gate = run_full_input_consumption_gate


def _independent_reload(job: CandidateJob, model: Any, *, source: Any, output_dir: Path, expected_rows: int | None, progress: _Progress) -> dict[str, Any]:
    reload_source = source
    if _is_neural(job.family):
        import torch

        reloaded = _fresh_model(job, dimensions=job.refit_dimensions)
        device = torch.device(job.config.device)
        reloaded.to(device)
        state_path = output_dir / "model_refit.pt"
        payload = torch.load(state_path, map_location=device, weights_only=False)
        reloaded.load_state_dict(payload["model_state"], strict=True)
        reloaded.eval()
        model.eval()
    else:
        with (output_dir / "model_refit.pkl").open("rb") as stream:
            reloaded = pickle.load(stream)
    original_logits: list[np.ndarray] = []
    reload_logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ledger = _Ledger(expected_rows=expected_rows, excluded_game_keys=job.excluded_game_keys, split="reload")
    device = None
    if _is_neural(job.family):
        import torch

        device = torch.device(job.config.device)
    with _nullcontext():
        for item in _source_iter(reload_source, "reload", epoch=0, batch_size=job.config.batch_size):
            inputs, target, metadata = _normalise_batch(item)
            _validate_scope_metadata(metadata, "dev", len(target))
            ledger.update(metadata, target)
            left = _model_logits(model, job.family, inputs, device)
            right = _model_logits(reloaded, job.family, inputs, device)
            if hasattr(left, "detach"):
                left = left.detach().cpu().numpy()
            if hasattr(right, "detach"):
                right = right.detach().cpu().numpy()
            original_logits.append(np.asarray(left, dtype=np.float64).reshape(-1))
            reload_logits.append(np.asarray(right, dtype=np.float64).reshape(-1))
            labels.append(target)
            progress.update(stage="reload", rows=len(target), batches=1, total_rows=expected_rows)
    if not labels:
        raise ValueError("reload source produced no batches")
    left = np.concatenate(original_logits)
    right = np.concatenate(reload_logits)
    y = np.concatenate(labels)
    consumed = ledger.finish()
    prediction_equal = bool(np.array_equal(left, right))
    max_abs_diff = float(np.max(np.abs(left - right)))
    if not prediction_equal:
        raise RuntimeError(f"checkpoint reload changed predictions (max_abs_logit_diff={max_abs_diff})")
    return {"rows": ledger.rows, "logloss": raw_logloss_from_logits(right, y), "max_abs_logit_diff": max_abs_diff, "prediction_equal": prediction_equal, "consumed": consumed}


def _validate_reload_against_dev(dev_metrics: Mapping[str, Any], reload_metrics: Mapping[str, Any]) -> None:
    """Require the independent reload pass to use the exact V_DEV rows."""

    expected = dev_metrics.get("consumed")
    actual = reload_metrics.get("consumed")
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise ValueError("reload evidence is missing V_DEV consumption metadata")
    for key in ("rows", "key_count", "keys_sha256", "game_keys_sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"reload consumption differs from V_DEV for {key}")
    if not np.isclose(float(actual.get("target_sum", np.nan)), float(expected.get("target_sum", np.nan)), rtol=0.0, atol=1e-12):
        raise ValueError("reload consumption differs from V_DEV for target_sum")
    if not np.isclose(float(reload_metrics.get("logloss", np.nan)), float(dev_metrics.get("raw_logloss", np.nan)), rtol=0.0, atol=1e-12):
        raise ValueError("reload logloss differs from V_DEV")


def run_candidate_job(job: CandidateJob) -> CandidateResult:
    """Fit one frozen-card candidate and persist all checkpoints/evidence."""

    output_dir = Path(job.output_dir)
    if (output_dir / "candidate_result.json").exists():
        raise FileExistsError(f"candidate output already has a terminal result: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = job.job_id or f"{job.context}_{job.family}_{job.card_id}_seed{job.seed}"
    progress = _Progress(output_dir / "progress.json", job_id=job_id, total_rows=job.expected_fit_rows)
    fit_rows = stop_rows = refit_rows = dev_rows = 0
    chosen_epoch: int | None = None
    fit_metrics: dict[str, Any] = {}
    stop_metrics: dict[str, Any] = {}
    refit_metrics: dict[str, Any] = {}
    dev_metrics: dict[str, Any] = {}
    reload_metrics: dict[str, Any] = {}
    try:
        configure_runtime(job.seed, job.config.device)
        identity = _job_identity(job, job_id)
        if job.family == "logistic":
            # The locked protocol gives logistic candidates one REFIT_TRAIN
            # fit.  There is no preliminary FIT/STOP fit whose sample count
            # could alter the L-BFGS regularization scale.
            chosen_epoch = None
            fit_metrics = {"skipped": True, "reason": "logistic_refit_only"}
            stop_metrics = {"skipped": True, "reason": "logistic_refit_only"}
            fit_rows = 0
        else:
            fit_done = _read_phase(output_dir, "FIT_DONE.json", identity)
            if fit_done is not None:
                # FIT/STOP finished before an interruption: reuse its verified record
                # and the RNG state it ended with.
                chosen_epoch = fit_done["chosen_epoch"]
                fit_metrics, stop_metrics, fit_rows = fit_done["fit_metrics"], fit_done["stop_metrics"], int(fit_done["fit_rows"])
                _set_rng_state(fit_done["rng"])
                _log_resume(output_dir, {"phase": "fit", "skipped": "FIT_DONE.json"})
            else:
                model = _fresh_model(job)
                if _is_neural(job.family):
                    chosen_epoch, fit_metrics, stop_metrics, fit_ledger = _fit_neural(
                        model, job.family, job.fit_source, job.stop_source,
                        config=job.config, progress=progress, output_dir=output_dir,
                        seed=job.seed, expected_fit_rows=job.expected_fit_rows,
                        expected_stop_rows=job.expected_stop_rows,
                        excluded_game_keys=job.excluded_game_keys, identity=identity,
                    )
                else:
                    chosen_epoch, fit_metrics, stop_metrics, fit_ledger = _fit_table(
                        model, job.family, job.fit_source, job.stop_source,
                        config=job.config, progress=progress, output_dir=output_dir,
                        expected_fit_rows=job.expected_fit_rows,
                        expected_stop_rows=job.expected_stop_rows,
                        excluded_game_keys=job.excluded_game_keys,
                    )
                fit_rows = fit_ledger.rows
                evidence = "checkpoint_best.pt" if _is_neural(job.family) else "model_fit.pkl"
                _write_phase(output_dir, "FIT_DONE.json", identity, {
                    "chosen_epoch": chosen_epoch, "fit_metrics": fit_metrics, "stop_metrics": stop_metrics, "fit_rows": fit_rows,
                    "files": {evidence: _file_sha256(output_dir / evidence)}, "rng": _rng_state()})
        stop_rows = int(stop_metrics.get("rows", stop_metrics.get("stop_rows", 0)))
        if job.refit_source is None:
            raise ValueError("a fresh REFIT_TRAIN source is required after STOP")
        refit_model = _fresh_model(job, dimensions=job.refit_dimensions)
        # Table REFIT is resumed only once its V_DEV scoring and the independent
        # reload check have passed on the in-memory model (REFIT_SCORED.json);
        # resuming earlier would leave the reload check nothing to compare.  A
        # neural REFIT resumes per epoch, and the reload check then compares
        # the model restored from the last verified epoch with model_refit.pt.
        scored = None if _is_neural(job.family) else _read_phase(output_dir, "REFIT_SCORED.json", identity)
        if scored is not None and scored.get("epochs") == chosen_epoch:
            refit_metrics, refit_rows = scored["refit_metrics"], int(scored["refit_rows"])
            dev_metrics, reload_metrics, dev_rows = scored["dev_metrics"], scored["reload_metrics"], int(scored["dev_rows"])
            _set_rng_state(scored["rng"])
            _log_resume(output_dir, {"phase": "refit", "skipped": "REFIT_SCORED.json"})
        else:
            if _is_neural(job.family):
                # A fresh model/optimizer is trained for exactly the selected STOP
                # epoch; no STOP weights or optimizer state are reused.
                refit_metrics, refit_ledger = _fit_refit_neural(
                    refit_model, job.family, job.refit_source, epochs=int(chosen_epoch or 1),
                    config=job.config, progress=progress, expected_rows=job.expected_refit_rows,
                    excluded_game_keys=job.excluded_game_keys, output_dir=output_dir, seed=job.seed, identity=identity,
                )
            else:
                refit_metrics, refit_ledger = _fit_refit_table(
                    refit_model, job.refit_source, family=job.family, chosen_epoch=chosen_epoch, config=job.config, progress=progress,
                    expected_rows=job.expected_refit_rows, excluded_game_keys=job.excluded_game_keys,
                    output_dir=output_dir,
                )
            refit_rows = refit_ledger.rows
            if job.dev_source is None:
                raise ValueError("V_DEV source is required for raw candidate selection")
            dev_loss, dev_ledger = _score_source(refit_model, job.family, job.dev_source, "dev", __import__("torch").device(job.config.device) if _is_neural(job.family) else None, job.config, expected_rows=job.expected_dev_rows, excluded_game_keys=job.excluded_game_keys, progress=progress)
            dev_rows = dev_ledger.rows
            dev_metrics = {"rows": dev_rows, "raw_logloss": dev_loss, "consumed": dev_ledger.finish(validate=False)}
            reload_metrics = _independent_reload(job, refit_model, source=job.dev_source, output_dir=output_dir, expected_rows=job.expected_dev_rows, progress=progress)
            _validate_reload_against_dev(dev_metrics, reload_metrics)
            if not _is_neural(job.family):
                _write_phase(output_dir, "REFIT_SCORED.json", identity, {
                    "epochs": chosen_epoch, "refit_metrics": refit_metrics, "refit_rows": refit_rows, "dev_rows": dev_rows,
                    "dev_metrics": dev_metrics, "reload_metrics": reload_metrics,
                    "files": {"model_refit.pkl": _file_sha256(output_dir / "model_refit.pkl")}, "rng": _rng_state()})
        result = CandidateResult(job_id, job.family, job.card_id, job.seed, job.context, "completed", str(output_dir), fit_rows, stop_rows, refit_rows, dev_rows, chosen_epoch, fit_metrics, stop_metrics, refit_metrics, dev_metrics, reload_metrics, str(output_dir / ("model_refit.pt" if _is_neural(job.family) else "model_refit.pkl")))
        _atomic_json(output_dir / "candidate_result.json", result.to_dict())
        progress.finish("completed")
        return result
    except LogisticNonconvergenceError as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "convergence": exc.convergence, "reason": "nonconverged"}
        failed = CandidateResult(job_id, job.family, job.card_id, job.seed, job.context, "failed", str(output_dir), fit_rows, stop_rows, refit_rows, dev_rows, chosen_epoch, fit_metrics, stop_metrics, refit_metrics, dev_metrics, reload_metrics, None, error)
        _atomic_json(output_dir / "candidate_result.json", failed.to_dict())
        progress.finish("failed", error=f"{type(exc).__name__}: {exc}")
        return failed
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed = CandidateResult(job_id, job.family, job.card_id, job.seed, job.context, "failed", str(output_dir), fit_rows, stop_rows, refit_rows, dev_rows, chosen_epoch, fit_metrics, stop_metrics, refit_metrics, dev_metrics, reload_metrics, None, error)
        _atomic_json(output_dir / "candidate_result.json", failed.to_dict())
        progress.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def run_candidate_search(base_job: CandidateJob, *, families: Sequence[str] = FAMILIES, require_all: bool = True) -> list[CandidateResult]:
    """Run the frozen four-card search for each requested family."""

    results: list[CandidateResult] = []
    failures: list[dict[str, Any]] = []
    root = Path(base_job.output_dir)
    for family in families:
        key = str(family).lower()
        for card in family_cards(key):
            job = replace(base_job, family=key, card_id=card["id"], output_dir=root / key / card["id"])
            try:
                results.append(run_candidate_job(job))
            except Exception as exc:
                failures.append({"family": key, "card_id": card["id"], "error": f"{type(exc).__name__}: {exc}"})
                if require_all:
                    raise
    _atomic_json(root / "search_summary.json", {"version": VERSION, "expected_candidates": sum(len(family_cards(str(f).lower())) for f in families), "completed_candidates": len(results), "failed_candidates": failures, "status": "completed" if not failures else "failed"})
    return results


def select_v_dev(results: Sequence[CandidateResult], *, tie_tolerance: float = V_FAMILY_TIE_TOLERANCE, family_priority: Sequence[str] = FAMILY_PRIORITY) -> CandidateResult:
    """Select the lowest raw V_DEV row logloss with inherited tie ordering."""

    candidates = [result for result in results if result.status == "completed" and np.isfinite(result.dev_metrics.get("raw_logloss", np.nan))]
    if not candidates:
        raise ValueError("no completed finite V_DEV candidates")
    dev_hashes = [
        result.dev_metrics.get("consumed", {}).get("keys_sha256")
        for result in candidates
        if isinstance(result.dev_metrics.get("consumed"), Mapping)
    ]
    if dev_hashes and (len(dev_hashes) != len(candidates) or any(value is None for value in dev_hashes) or len(set(dev_hashes)) != 1):
        raise ValueError("V_DEV candidates do not share one complete row-key set")
    priority = {str(name).lower(): index for index, name in enumerate(family_priority)}
    family_winners: list[CandidateResult] = []
    for family in sorted({result.family for result in candidates}):
        within = [result for result in candidates if result.family == family]
        card_order = {str(card["id"]): index for index, card in enumerate(family_cards(family))}
        family_winners.append(min(within, key=lambda result: (float(result.dev_metrics["raw_logloss"]), card_order.get(result.card_id, len(card_order)), result.seed)))
    best_value = min(float(result.dev_metrics["raw_logloss"]) for result in family_winners)
    tied = [result for result in family_winners if float(result.dev_metrics["raw_logloss"]) <= best_value + float(tie_tolerance)]
    return min(tied, key=lambda result: (priority.get(result.family, len(priority)), result.family, result.card_id, result.seed))


def fit_positive_slope_calibrator(raw_probability: Any, target: Any) -> PositiveSlopeCalibrator:
    """Fit the locked V_CAL positive-slope sigmoid in float64."""

    probabilities = np.asarray(raw_probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if probabilities.shape != labels.shape or len(labels) == 0 or not np.isfinite(probabilities).all() or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("calibration inputs must be aligned finite binary arrays")
    counts = {"0": int(np.sum(labels == 0.0)), "1": int(np.sum(labels == 1.0))}
    if not counts["0"] or not counts["1"]:
        raise ValueError("V_CAL requires both classes")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped) - np.log1p(-clipped)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = float(parameters[0]), float(parameters[1])
        z = a * logits + b
        p = _sigmoid(z)
        loss = float(np.mean(np.logaddexp(0.0, z) - labels * z))
        residual = p - labels
        return loss, np.asarray([np.mean(residual * logits), np.mean(residual)], dtype=np.float64)

    from scipy.optimize import minimize

    result = minimize(objective, np.asarray([1.0, 0.0], dtype=np.float64), jac=True, method="L-BFGS-B", bounds=((1e-6, None), (None, None)), options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
    if not bool(result.success) or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"positive-slope calibration failed: {result.message}")
    gradient = np.asarray(objective(np.asarray(result.x))[1], dtype=np.float64)
    projected = gradient.copy()
    if float(result.x[0]) <= 1e-6 + 1e-12:
        projected[0] = min(0.0, projected[0])
    projected_inf = float(np.max(np.abs(projected)))
    if projected_inf > 1e-6:
        raise RuntimeError(f"positive-slope calibration KKT check failed: {projected_inf}")
    return PositiveSlopeCalibrator(float(result.x[0]), float(result.x[1]), float(result.fun), projected_inf, True, counts)


def _load_cli_object(spec: str) -> Any:
    """Load a replayable source or factory named ``module:attribute``."""

    if ":" not in spec:
        raise ValueError("source must use module:attribute syntax")
    module_name, attribute = spec.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not hasattr(value, "iter_batches") and not hasattr(value, "iter_split") and callable(value):
        try:
            value = value()
        except TypeError:
            # A callable source may require the split/epoch/batch_size
            # arguments supplied by _source_iter; leave it callable.
            pass
    return value


def _load_cli_json(value: str | None) -> Any:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8-sig"))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"expected JSON value or readable path: {value!r}") from exc


def _load_cli_card(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V production candidate runner")
    commands = parser.add_subparsers(dest="command", required=True)

    gate = commands.add_parser("gate", help="consume the full transformed source in probe mode")
    gate.add_argument("--source", required=True, help="replayable module:attribute source or factory")
    gate.add_argument("--output-dir", required=True)
    gate.add_argument("--dimensions-json", help="JSON object/path or integer model width")
    gate.add_argument("--expected-rows", type=int)
    gate.add_argument("--families", default=",".join(FAMILIES))
    gate.add_argument("--card-id", default="0")
    gate.add_argument("--seed", type=int, default=20261001)
    gate.add_argument("--device", default="cpu")
    gate.add_argument("--allow-short-first-batch", action="store_true")
    gate.add_argument("--no-backward", action="store_true")
    gate.add_argument("--resource-limits-json")

    candidate = commands.add_parser("candidate", help="fit one guarded family/card candidate")
    candidate.add_argument("--family", required=True, choices=FAMILIES)
    candidate.add_argument("--card-id", required=True)
    candidate.add_argument("--seed", type=int, default=20261001)
    candidate.add_argument("--acceptance-path", required=True)
    candidate.add_argument("--output-dir", required=True)
    candidate.add_argument("--fit-source", help="module:attribute source; omitted for logistic REFIT-only candidates")
    candidate.add_argument("--stop-source")
    candidate.add_argument("--refit-source", required=True)
    candidate.add_argument("--dev-source", required=True)
    candidate.add_argument("--context", default="final")
    candidate.add_argument("--fold", type=int)
    candidate.add_argument("--dimensions-json")
    candidate.add_argument("--refit-dimensions-json")
    candidate.add_argument("--expected-fit-rows", type=int)
    candidate.add_argument("--expected-stop-rows", type=int)
    candidate.add_argument("--expected-refit-rows", type=int)
    candidate.add_argument("--expected-dev-rows", type=int)
    candidate.add_argument("--excluded-game-key", action="append", default=[])
    candidate.add_argument("--device", default="cpu")
    candidate.add_argument("--cpu-threads", type=int, default=1)
    candidate.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the supervisor and bounded local checks."""

    args = _cli_parser().parse_args(argv)
    if args.command == "gate":
        dimensions = _load_cli_json(args.dimensions_json)
        limits = _load_cli_json(args.resource_limits_json) or {}
        families = tuple(part.strip() for part in str(args.families).split(",") if part.strip())
        result = run_full_input_consumption_gate(
            _load_cli_object(args.source),
            output_dir=args.output_dir,
            dimensions=dimensions,
            expected_rows=args.expected_rows,
            families=families,
            card_id=_load_cli_card(args.card_id),
            seed=args.seed,
            device=args.device,
            require_full_batch=not args.allow_short_first_batch,
            backward_probe=not args.no_backward,
            resource_limits=limits,
        )
    else:
        config = RunnerConfig(max_epochs=args.max_epochs, cpu_threads=args.cpu_threads, device=args.device)
        job = CandidateJob(
            family=args.family,
            card_id=args.card_id,
            seed=args.seed,
            acceptance_path=args.acceptance_path,
            output_dir=args.output_dir,
            fit_source=_load_cli_object(args.fit_source) if args.fit_source else None,
            stop_source=_load_cli_object(args.stop_source) if args.stop_source else None,
            refit_source=_load_cli_object(args.refit_source),
            dev_source=_load_cli_object(args.dev_source),
            context=args.context,
            fold=args.fold,
            dimensions=_load_cli_json(args.dimensions_json),
            refit_dimensions=_load_cli_json(args.refit_dimensions_json),
            config=config,
            expected_fit_rows=args.expected_fit_rows,
            expected_stop_rows=args.expected_stop_rows,
            expected_refit_rows=args.expected_refit_rows,
            expected_dev_rows=args.expected_dev_rows,
            excluded_game_keys=tuple(args.excluded_game_key),
        )
        result = run_candidate_job(job).to_dict()
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BATCH_SIZE", "Batch", "BatchSource", "CandidateJob", "CandidateResult", "FIXED_SEEDS", "PHASE1_SEED", "PositiveSlopeCalibrator", "RunnerConfig", "VERSION", "configure_runtime", "fit_positive_slope_calibrator", "main", "raw_logloss_from_logits", "raw_logloss_from_probability", "run_candidate_job", "run_candidate_search", "run_full_input_consumption_gate", "run_resource_consumption_gate", "select_v_dev", "verify_full_input_consumption",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke checks
    raise SystemExit(main())
