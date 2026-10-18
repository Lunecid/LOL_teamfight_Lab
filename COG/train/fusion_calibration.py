from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.fight_types import ref_key
from core.utils import write_log


def _nll_temperature(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    T = max(float(T), 1e-6)
    z = logits / T
    loss_pos = np.log1p(np.exp(-np.clip(z, -30, 30)))
    loss_neg = np.log1p(np.exp(np.clip(z, -30, 30)))
    nll = float(np.mean(labels * loss_pos + (1 - labels) * loss_neg))
    return nll


def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    T_init: float = 1.0,
    bounds: Tuple[float, float] = (0.1, 10.0),
) -> float:
    try:
        from scipy.optimize import minimize_scalar  # type: ignore

        result = minimize_scalar(
            _nll_temperature,
            bounds=bounds,
            args=(logits, labels),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 200},
        )
        return float(result.x)
    except ImportError:
        best_T, best_nll = T_init, _nll_temperature(T_init, logits, labels)
        for T_cand in np.linspace(bounds[0], bounds[1], 100):
            nll = _nll_temperature(T_cand, logits, labels)
            if nll < best_nll:
                best_T, best_nll = float(T_cand), nll
        return best_T


def calibrate_logits_by_patch(
    logit_map: Dict[str, float],
    refs: List,
    labels: Dict[str, int],
    log_fp: Optional[Path] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    from collections import defaultdict

    patch_groups: Dict[str, Tuple[List[str], List[float], List[int]]] = defaultdict(lambda: ([], [], []))

    for r in refs:
        k = ref_key(r) if callable(ref_key) else str(r)
        if k not in logit_map or k not in labels:
            continue
        patch_str = str(getattr(r, "patch", getattr(r, "patch_str", "unknown")))
        keys_list, logits_list, labels_list = patch_groups[patch_str]
        keys_list.append(k)
        logits_list.append(logit_map[k])
        labels_list.append(labels[k])

    calibrated_map: Dict[str, float] = {}
    T_by_patch: Dict[str, float] = {}

    for patch_str, (keys_list, logits_list, labels_list) in patch_groups.items():
        z = np.array(logits_list, dtype=np.float64)
        y = np.array(labels_list, dtype=np.float64)

        if len(z) < 50 or len(np.unique(y)) < 2:
            T_star = 1.0
        else:
            n_cal = max(20, len(z) // 2)
            T_star = find_optimal_temperature(z[:n_cal], y[:n_cal])

        T_by_patch[patch_str] = T_star

        for k, logit_val in zip(keys_list, logits_list):
            calibrated_map[k] = float(logit_val / T_star)

        if log_fp:
            write_log(
                f"[TEMP_SCALE] patch={patch_str} n={len(z)} T*={T_star:.4f}",
                log_fp,
            )

    return calibrated_map, T_by_patch


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc_bin = float(labels[mask].mean())
        conf_bin = float(probs[mask].mean())
        ece += (n_bin / n) * abs(acc_bin - conf_bin)

    return float(ece)
