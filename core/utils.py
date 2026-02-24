# utils.py
from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

_write_log_logger = logging.getLogger("lol_teamfight")


def _pairwise_dists_xy(xy_t: np.ndarray, b_idx: np.ndarray, r_idx: np.ndarray) -> np.ndarray:
    """xy_t: (10,2) raw coords. return (5,5) distances."""
    d = xy_t[b_idx][:, None, :] - xy_t[r_idx][None, :, :]
    return np.sqrt((d * d).sum(axis=-1))

def _min_cross_team_dist_mean(dists: np.ndarray) -> float:
    """dists: (5,5)"""
    # each blue -> nearest red
    return float(dists.min(axis=1).mean())

def _count_prox_pairs(dists: np.ndarray, R: float) -> int:
    return int((dists <= R).sum())

def _first_kill_ts_in_window(events: list, s: int, e: int) -> Optional[int]:
    best = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ts = int(ev.get("timestamp", -1))
        if ts < s or ts >= e:
            continue
        if str(ev.get("type", "")).upper() == "CHAMPION_KILL":
            best = ts if best is None else min(best, ts)
    return best


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Logging / IO
# =========================================================
def write_log(msg: str, fp: Optional[Path] = None):
    _write_log_logger.info(msg)
    if fp is not None:
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def setup_logging(level: int = logging.INFO):
    """Configure root logging for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# utils.py
import json
from pathlib import Path


def _json_default(o):
    # Path -> str
    if isinstance(o, Path):
        return str(o)

    # numpy scalars -> python scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)

    # numpy arrays -> list
    if isinstance(o, np.ndarray):
        return o.tolist()

    # fallback (optional): uncomment if you want to be very permissive
    # return str(o)

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_json(path: Path, obj, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_json_default)



def save_csv_rows(path: Path, fieldnames: List[str], rows: List[dict]):
    """
    Safer CSV writer:
    - expands fieldnames with any keys found in rows (stable order)
    - ignores extra keys if still present
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    base = list(fieldnames or [])
    seen = set(base)

    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k not in seen:
                base.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if isinstance(r, dict):
                w.writerow(r)


# =========================================================
# Numeric safety helpers
# =========================================================
def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def log1p_norm(x, denom: float) -> float:
    """
    Deterministic log(1+x) normalization.
    - Used in features.py for various counts/values.
    """
    x = max(0.0, safe_float(x))
    d = max(1.0, float(denom))
    return math.log1p(x) / math.log1p(d)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, safe_float(x))))


# =========================================================
# Torch helpers
# =========================================================
def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensors inside a dict batch to target device.
    Non-tensors are passed through.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_t(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


# =========================================================
# Metrics (AUC / Confusion Matrix / Precision@K)
# =========================================================
def confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    """
    y_true: {0,1}
    y_prob: [0,1]
    """
    y_true = y_true.astype(int)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pure-numpy ROC-AUC (Mann–Whitney U) with proper tie handling (average ranks).
    Deterministic and correct for tied scores (e.g., tree leaf outputs).
    """
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # concat: first all positives, then all negatives
    all_scores = np.concatenate([pos, neg], axis=0)
    n = int(all_scores.shape[0])
    n_pos = int(len(pos))
    n_neg = int(len(neg))

    # stable sort for determinism
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]

    # assign average ranks for ties (1-indexed ranks)
    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        # exact tie group (good for leaf-based identical floats)
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1

        # ranks i..j-1 correspond to (i+1) .. j in 1-indexed rank
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j

    # map back to original order
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted

    # Mann–Whitney U using ranks of positives
    r_pos = float(ranks[:n_pos].sum())
    u = r_pos - (n_pos * (n_pos + 1.0) / 2.0)

    auc = u / (float(n_pos) * float(n_neg))
    return float(auc)


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Simple Average Precision (area under PR) without sklearn.
    """
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    if y_true.sum() == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)
    # sum precision at each positive instance / num positives
    ap = (prec * (y_sorted == 1)).sum() / max(1, (y_true == 1).sum())
    return float(ap)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of probabilistic predictions.

    BS = (1/N) * sum((p_i - y_i)^2)

    Lower is better. Range [0, 1].
    """
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.95) -> float:
    """Compute recall at a given minimum precision threshold.

    Finds the highest recall achievable while maintaining precision >= min_precision.
    Useful for high-stakes predictions where false positives are costly.
    """
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    if y_true.sum() == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1).astype(float)
    fp_cum = np.cumsum(y_sorted == 0).astype(float)
    prec_at_k = tp_cum / (tp_cum + fp_cum)

    total_pos = float(y_true.sum())
    recall_at_k = tp_cum / total_pos

    # Find the maximum recall where precision >= min_precision
    valid = prec_at_k >= min_precision
    if not valid.any():
        return 0.0
    return float(recall_at_k[valid].max())


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Returns (mean_predicted, fraction_positive) for each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_pred = []
    frac_pos = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))

    return np.array(mean_pred), np.array(frac_pos)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    cm = confusion_from_probs(y_true, y_prob, threshold=threshold)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    auc = _auc_rank(y_true, y_prob)
    ap = average_precision(y_true, y_prob)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
        "brier": brier_score(y_true, y_prob),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    k = int(k)
    if k <= 0:
        return float("nan")
    order = np.argsort(-y_prob)
    top = y_true[order[: min(k, len(y_true))]]
    return float(top.mean()) if len(top) > 0 else float("nan")


def precision_at_frac(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> float:
    n = len(y_true)
    k = int(max(1, round(n * float(frac))))
    return precision_at_k(y_true, y_prob, k)


# =========================================================
# Feature-name sanitization (for LightGBM / logs / saving)
# =========================================================
def sanitize_feature_name(name: str) -> str:
    bad = ['{', '}', '[', ']', ':', '"', "'", '\\', '\n', '\r', '\t']
    out = str(name)
    for b in bad:
        out = out.replace(b, "_")
    return out


def sanitize_feature_names(names: Iterable[str]) -> List[str]:
    return [sanitize_feature_name(n) for n in names]


# =========================================================
# Sequence Scaler (fit on train only), excluding coordinate-like features
# - x/y should NOT be fitted-scaled, so we exclude by prefix (config)
# - used by data_loader: scaler.transform(node_ts, extra_ts)
# =========================================================
@dataclass
class SeqScaler:
    """
    Fits feature-wise mean/std on flattened sequences.
    - node_ts: (L,10,F_NODE) or (B,L,10,F_NODE)
    - extra_ts: (L,D) or (B,L,D)
    Exclusion:
    - node features with prefixes in cfg.SCALER_EXCLUDE_PREFIXES will be left unchanged.
    - macro features can also be excluded similarly if names provided.
    """
    scaler_type: Optional[str] = "standard"  # standard|minmax|None
    exclude_prefixes: Tuple[str, ...] = ("x_", "y_", "pos_", "dist_", "angle_")
    # stats
    node_mean: Optional[torch.Tensor] = None
    node_std: Optional[torch.Tensor] = None
    extra_mean: Optional[torch.Tensor] = None
    extra_std: Optional[torch.Tensor] = None
    fitted: bool = False

    def fit(self, node_samples: List[torch.Tensor], extra_samples: List[torch.Tensor],
            node_feature_names: Optional[List[str]] = None,
            extra_feature_names: Optional[List[str]] = None):
        """
        node_samples: list of (L,10,F)
        extra_samples: list of (L,D)
        """
        if self.scaler_type is None or str(self.scaler_type).lower() == "none":
            self.fitted = True
            return self

        st = str(self.scaler_type).lower()
        if st not in ("standard",):
            raise ValueError(f"SeqScaler: unsupported scaler_type={self.scaler_type}")

        # stack -> (N_total, F)
        if node_samples:
            Xn = torch.cat([x.reshape(-1, x.shape[-1]) for x in node_samples], dim=0)  # (S, F_node)
            mean = Xn.mean(dim=0)
            std = Xn.std(dim=0).clamp_min(1e-6)

            # exclude by name prefixes (default: x_norm/y_norm)
            if node_feature_names is not None:
                keep_mask = torch.ones_like(mean, dtype=torch.bool)
                for i, nm in enumerate(node_feature_names):
                    for pfx in self.exclude_prefixes:
                        if str(nm).startswith(pfx):
                            keep_mask[i] = False
                            break
                # for excluded dims: mean=0 std=1 (no-op)
                mean = torch.where(keep_mask, mean, torch.zeros_like(mean))
                std = torch.where(keep_mask, std, torch.ones_like(std))

            self.node_mean = mean
            self.node_std = std

        if extra_samples:
            Xe = torch.cat([x.reshape(-1, x.shape[-1]) for x in extra_samples], dim=0)  # (S, D_extra)
            mean = Xe.mean(dim=0)
            std = Xe.std(dim=0).clamp_min(1e-6)

            # exclude by name prefixes if provided
            if extra_feature_names is not None:
                keep_mask = torch.ones_like(mean, dtype=torch.bool)
                for i, nm in enumerate(extra_feature_names):
                    for pfx in self.exclude_prefixes:
                        if str(nm).startswith(pfx):
                            keep_mask[i] = False
                            break
                mean = torch.where(keep_mask, mean, torch.zeros_like(mean))
                std = torch.where(keep_mask, std, torch.ones_like(std))

            self.extra_mean = mean
            self.extra_std = std

        self.fitted = True
        return self

    def transform(self, node_ts: Optional[torch.Tensor], extra_ts: Optional[torch.Tensor]):
        """
        Accepts:
          - node_ts: (L,10,F) or (B,L,10,F)
          - extra_ts: (L,D) or (B,L,D)
        Returns transformed tensors (same shape).
        """
        if not self.fitted or (self.scaler_type is None) or str(self.scaler_type).lower() == "none":
            return node_ts, extra_ts

        if node_ts is not None and self.node_mean is not None and self.node_std is not None:
            mean = self.node_mean.to(node_ts.device)
            std = self.node_std.to(node_ts.device)
            node_ts = (node_ts - mean) / std

        if extra_ts is not None and self.extra_mean is not None:
            mean = self.extra_mean.to(extra_ts.device)
            std = self.extra_std.to(extra_ts.device)
            if int(extra_ts.shape[-1]) == int(mean.shape[0]):
                extra_ts = (extra_ts - mean) / std

        return node_ts, extra_ts

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scaler state for checkpoint persistence."""
        state = {
            "scaler_type": self.scaler_type,
            "exclude_prefixes": self.exclude_prefixes,
            "fitted": self.fitted,
        }
        if self.node_mean is not None:
            state["node_mean"] = self.node_mean.cpu()
        if self.node_std is not None:
            state["node_std"] = self.node_std.cpu()
        if self.extra_mean is not None:
            state["extra_mean"] = self.extra_mean.cpu()
        if self.extra_std is not None:
            state["extra_std"] = self.extra_std.cpu()
        return state

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SeqScaler":
        """Restore scaler from checkpoint state dict."""
        scaler = cls(
            scaler_type=state.get("scaler_type", "standard"),
            exclude_prefixes=tuple(state.get("exclude_prefixes", ("x_", "y_", "pos_", "dist_", "angle_"))),
        )
        scaler.fitted = state.get("fitted", False)
        scaler.node_mean = state.get("node_mean", None)
        scaler.node_std = state.get("node_std", None)
        scaler.extra_mean = state.get("extra_mean", None)
        scaler.extra_std = state.get("extra_std", None)
        return scaler


# =========================================================
# Minor helpers for debugging / shape assertions
# =========================================================
def assert_finite(t: torch.Tensor, name: str = "tensor"):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise ValueError(f"[NaN/Inf] {name} has {bad} non-finite values")


def pretty_cm(cm: Dict[str, int]) -> str:
    return f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}"
