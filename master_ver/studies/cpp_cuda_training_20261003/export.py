"""Portable TorchScript transport for the bounded C++ training smoke.

The exporter deliberately accepts an already-created model head and already
materialized dense tensors.  It does not read the research corpus, fit
preprocessors, or claim that the resulting bundle is ready for full training.
"""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


EXPORT_VERSION = "cpp_cuda_dense_transport_20261003_v1"


class DenseJobData(nn.Module):
    """TorchScript-readable data bundle with stable, named tensor buffers."""

    def __init__(
        self,
        features: Tensor,
        targets: Tensor,
        weights: Tensor,
        order: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("features", features)
        self.register_buffer("targets", targets)
        self.register_buffer("weights", weights)
        self.register_buffer("order", order)

    def forward(self) -> Tensor:
        """Return the dense feature matrix; labels and order stay named buffers."""

        return self.features


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_script_module(module: torch.jit.ScriptModule, path: Path) -> None:
    # The Windows TorchScript filename overload does not reliably accept the
    # Korean workspace path; the file-object overload preserves the same
    # archive while avoiding that narrow path conversion.
    with path.open("wb") as stream:
        torch.jit.save(module, stream)


def _tensor_sha256(value: Tensor) -> str:
    cpu_value = value.detach().to(device="cpu").contiguous()
    # Include shape and dtype so an accidental schema change cannot retain the
    # same byte hash merely because the underlying storage is unchanged.
    header = _canonical_json({"dtype": str(cpu_value.dtype), "shape": list(cpu_value.shape)})
    return _sha256_bytes(header + b"\0" + cpu_value.numpy().tobytes(order="C"))


def _require_tensor(value: Any, name: str, dtype: torch.dtype) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _validate_inputs(
    features: Tensor,
    targets: Tensor,
    weights: Tensor,
    order: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    features = _require_tensor(features, "features", torch.float32)
    targets = _require_tensor(targets, "targets", torch.float32)
    weights = _require_tensor(weights, "weights", torch.float32)
    order = _require_tensor(order, "order", torch.int64)

    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError("features must have non-empty shape [N,D]")
    n_rows = int(features.shape[0])
    if targets.shape != (n_rows, 1):
        raise ValueError(f"targets must have shape [{n_rows},1], got {tuple(targets.shape)}")
    if weights.shape != (n_rows, 1):
        raise ValueError(f"weights must have shape [{n_rows},1], got {tuple(weights.shape)}")
    if order.ndim != 1 or int(order.shape[0]) != n_rows:
        raise ValueError(f"order must have shape [{n_rows}], got {tuple(order.shape)}")

    if not bool(torch.isfinite(features).all()):
        raise ValueError("features contains nonfinite values")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets contains nonfinite values")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("targets must contain only binary labels 0 or 1")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights contains nonfinite values")
    if not bool((weights >= 0).all()):
        raise ValueError("weights must be nonnegative")
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("weights must have a positive global sum")

    sorted_order = torch.sort(order.detach().to(device="cpu").contiguous()).values
    expected = torch.arange(n_rows, dtype=torch.int64)
    if not torch.equal(sorted_order, expected):
        raise ValueError("order must be a permutation of 0..N-1")

    # The transport format is CPU-portable.  Cloning also prevents later
    # caller mutations from changing the tensors after this function returns.
    return tuple(
        value.detach().to(device="cpu").contiguous().clone()
        for value in (features, targets, weights, order)
    )  # type: ignore[return-value]


def _validate_model(model_head: nn.Module) -> None:
    if not isinstance(model_head, nn.Module):
        raise TypeError("model_head must be a torch.nn.Module")
    parameters = list(model_head.parameters())
    if not parameters:
        raise ValueError("model_head must contain trainable parameters")
    for parameter in parameters:
        if parameter.dtype != torch.float32:
            raise TypeError(f"model parameters must be float32, got {parameter.dtype}")
    for buffer in model_head.buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.float32:
            raise TypeError(f"floating model buffers must be float32, got {buffer.dtype}")


def export_dense_job(
    model_head: nn.Module,
    features: Tensor,
    targets: Tensor,
    weights: Tensor,
    order: Tensor,
    out_dir: str | Path,
    *,
    synthetic: bool,
) -> dict[str, Path]:
    """Export one dense training job into a new, exclusive directory.

    The model is scripted rather than traced so GELU and Dropout retain their
    train/eval switch.  Data tensors are saved as named CPU buffers for the C++
    side to load without Python pickling.
    """

    _validate_model(model_head)
    features, targets, weights, order = _validate_inputs(features, targets, weights, order)

    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"export directory must be new and exclusive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    model_training = bool(model_head.training)
    # Script a detached copy: moving the saved module to CPU must not move or
    # otherwise mutate a caller-owned model head.
    scripted_model = torch.jit.script(copy.deepcopy(model_head))
    scripted_model = scripted_model.cpu()
    scripted_model.train(model_training)
    model_path = output / "model.pt"
    _save_script_module(scripted_model, model_path)

    data_module = DenseJobData(features, targets, weights, order)
    scripted_data = torch.jit.script(data_module).cpu()
    data_path = output / "data.pt"
    _save_script_module(scripted_data, data_path)

    n_rows, feature_dim = int(features.shape[0]), int(features.shape[1])
    schema: dict[str, Any] = {
        "format": EXPORT_VERSION,
        "model": {
            "input": {"dtype": "torch.float32", "shape": [n_rows, feature_dim]},
            "output": {"dtype": "torch.float32", "shape": [n_rows, 1]},
            "scripted": True,
            "trace_used": False,
            "initial_training": model_training,
        },
        "data": {
            "features": {"dtype": "torch.float32", "shape": [n_rows, feature_dim]},
            "targets": {"dtype": "torch.float32", "shape": [n_rows, 1], "values": [0, 1]},
            "weights": {"dtype": "torch.float32", "shape": [n_rows, 1], "minimum": 0.0},
            "order": {"dtype": "torch.int64", "shape": [n_rows], "permutation": True},
        },
        "objective": "mean(weights * binary_cross_entropy_with_logits)",
        "batch_order": "fixed order buffer; no shuffle",
        "gradient_clip_norm": 1.0,
    }
    schema_sha256 = _sha256_bytes(_canonical_json(schema))
    metadata: dict[str, Any] = {
        "format": EXPORT_VERSION,
        "schema": schema,
        "schema_sha256": schema_sha256,
        "model_sha256": _sha256_file(model_path),
        "data_sha256": _sha256_file(data_path),
        "input_counts": {
            "rows": n_rows,
            "features": feature_dim,
            "targets": int(targets.numel()),
            "weights": int(weights.numel()),
            "order": int(order.numel()),
        },
        "input_sha256": {
            "features": _tensor_sha256(features),
            "targets": _tensor_sha256(targets),
            "weights": _tensor_sha256(weights),
            "order": _tensor_sha256(order),
        },
        "synthetic": bool(synthetic),
        "synthetic_marker": "synthetic_only" if synthetic else "caller_supplied",
        "caller_explicit": True,
        "caller": "explicit_synthetic_smoke" if synthetic else "explicit_input",
        "full_training_ready": False,
        "leakage_audit": {
            "status": "excluded",
            "performed": False,
            "required_for_full_training": True,
            "full_training_ready": False,
            "scope": "synthetic transport smoke only" if synthetic else "caller-supplied transport only",
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": model_path, "data": data_path, "metadata": metadata_path}


__all__ = ["DenseJobData", "EXPORT_VERSION", "export_dense_job"]
