from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import cfg

logger = logging.getLogger(__name__)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _safe_auc_ap(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auc": float(roc_auc_score(y_true, y_prob)),
            "ap": float(average_precision_score(y_true, y_prob)),
        }
    except Exception as e:
        logger.debug("AUC/AP computation failed: %s", e)
        return {"auc": -1.0, "ap": -1.0}


def _resolve_amp_dtype() -> torch.dtype:
    pref = str(getattr(cfg, "AMP_DTYPE", "auto")).strip().lower()
    if pref in ("bfloat16", "bf16"):
        return torch.bfloat16
    if pref in ("float16", "fp16", "half"):
        return torch.float16

    try:
        if torch.cuda.is_available() and bool(torch.cuda.is_bf16_supported()):
            return torch.bfloat16
    except Exception as e:
        logger.debug("bf16 support check failed: %s", e)
    return torch.float16


def _autocast_ctx(use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_resolve_amp_dtype())
    return contextlib.nullcontext()


def _extract_main_logit(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, dict):
        for k in ("fight_logit", "logit", "y_logit"):
            v = model_out.get(k, None)
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


class _LogitAuxHead(nn.Module):
    """Auxiliary regression head on top of the main fight logit."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, logit: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = logit.view(logit.shape[0], -1)
        o = self.mlp(z)
        return {
            "gold_pred": o[:, 0:1],
            "kill_pred": o[:, 1:2],
            "obj_pred": o[:, 2:3],
        }


def _eval_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    ys: List[np.ndarray] = []
    ls: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            batch = _to_device(batch, device)

            y = batch.get("y", None)
            if y is None:
                continue

            with _autocast_ctx(use_amp, device):
                logit = _extract_main_logit(model(batch))

            y_np = y.detach().float().cpu().numpy().reshape(-1)
            logit_np = logit.detach().float().cpu().numpy().reshape(-1)

            ys.append(y_np)
            ls.append(logit_np)

    if not ys:
        return {
            "acc": -1.0,
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0,
            "auc": -1.0,
            "ap": -1.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "n": 0.0,
        }

    y_true = np.concatenate(ys).astype(np.int64)
    y_logit = np.concatenate(ls).astype(np.float32)
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_logit, -30, 30)))
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    acc = float((y_pred == y_true).mean())

    auc_ap = _safe_auc_ap(y_true, y_prob)

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc_ap["auc"]),
        "ap": float(auc_ap["ap"]),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "n": float(len(y_true)),
    }


@torch.inference_mode()
def _predict_logit_map_for_refs(
    model: nn.Module,
    loader: DataLoader,
    refs: List[Any],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}

    for batch in loader:
        if batch is None:
            continue
        batch = _to_device(batch, device)

        with _autocast_ctx(use_amp, device):
            logit = _extract_main_logit(model(batch))

        logit_np = logit.detach().float().cpu().numpy().reshape(-1)
        bsz = int(logit_np.shape[0])

        keys = None
        if "ref_key" in batch:
            rk = batch["ref_key"]
            if isinstance(rk, (list, tuple)):
                keys = [str(x) for x in rk]
        elif "ref_keys" in batch:
            rk = batch["ref_keys"]
            if isinstance(rk, (list, tuple)):
                keys = [str(x) for x in rk]

        if keys is None or len(keys) != bsz:
            raise RuntimeError(
                "[PRED_MAP] Batch missing ref_key/ref_keys (or length mismatch). "
                "Cannot safely align logits to FightRef. "
                "Ensure dataset/collate_batch injects 'ref_key' list."
            )

        for k, l in zip(keys, logit_np):
            out[str(k)] = float(l)

    return out


def _predict_logit_and_label_maps_for_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    model.eval()
    logit_map: Dict[str, float] = {}
    label_map: Dict[str, int] = {}

    for batch in loader:
        if batch is None:
            continue
        batch = _to_device(batch, device)

        y = batch.get("y", None)
        with _autocast_ctx(use_amp, device):
            logit = _extract_main_logit(model(batch))

        logit_np = logit.detach().float().cpu().numpy().reshape(-1)
        bsz = int(logit_np.shape[0])

        if y is not None:
            y_np = y.detach().float().cpu().numpy().reshape(-1)
        else:
            y_np = np.full(bsz, -1, dtype=np.float32)

        keys = None
        if "ref_key" in batch:
            rk = batch["ref_key"]
            if isinstance(rk, (list, tuple)):
                keys = [str(x) for x in rk]
        elif "ref_keys" in batch:
            rk = batch["ref_keys"]
            if isinstance(rk, (list, tuple)):
                keys = [str(x) for x in rk]

        if keys is None or len(keys) != bsz:
            raise RuntimeError(
                "[PRED_LABEL_MAP] Batch missing ref_key/ref_keys (or length mismatch). "
                "Cannot safely align logits+labels to FightRef."
            )

        for k, l_val, y_val in zip(keys, logit_np, y_np):
            logit_map[str(k)] = float(l_val)
            label_map[str(k)] = int(round(float(y_val)))

    return logit_map, label_map
