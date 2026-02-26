from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval via percentile method."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        m = float(arr.mean()) if n == 1 else 0.0
        return m, m, m

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(arr.mean())
    return mean, ci_low, ci_high


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """DeLong's test for comparing two correlated ROC-AUC values."""
    try:
        from scipy import stats as sp_stats
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[WARN] scipy/sklearn not available for DeLong's test")
        return float("nan")

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    def _compute_structural(pred: np.ndarray, pos: np.ndarray, neg: np.ndarray):
        pred_pos = pred[pos]
        pred_neg = pred[neg]

        v10 = np.zeros(len(pos))
        for j, pp in enumerate(pred_pos):
            v10[j] = np.mean((pred_neg < pp).astype(float) + 0.5 * (pred_neg == pp).astype(float))

        v01 = np.zeros(len(neg))
        for i, pn in enumerate(pred_neg):
            v01[i] = np.mean((pred_pos > pn).astype(float) + 0.5 * (pred_pos == pn).astype(float))

        return v10, v01

    v10_a, v01_a = _compute_structural(pred_a, pos_idx, neg_idx)
    v10_b, v01_b = _compute_structural(pred_b, pos_idx, neg_idx)

    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)

    s10 = np.cov(np.stack([v10_a, v10_b]))[0, 1] if n_pos > 1 else 0
    s01 = np.cov(np.stack([v01_a, v01_b]))[0, 1] if n_neg > 1 else 0

    var_a = np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg if (n_pos > 1 and n_neg > 1) else 1e-10
    var_b = np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg if (n_pos > 1 and n_neg > 1) else 1e-10
    cov_ab = s10 / n_pos + s01 / n_neg

    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        return 1.0 if abs(auc_a - auc_b) < 1e-10 else 0.0

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(p_value)


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """McNemar's test for paired classification comparison."""
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan")

    y_a = (pred_a >= threshold).astype(int)
    y_b = (pred_b >= threshold).astype(int)

    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)

    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))

    if b + c == 0:
        return 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
    return float(p_value)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni multiple-testing correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m

    for rank, (original_idx, p_val) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p_val <= adjusted_alpha:
            results[original_idx] = True
        else:
            break

    return results


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)

    return float(ece)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score."""
    return float(np.mean((y_prob - y_true) ** 2))


def safe_mean(values: List[float], default: float = -1.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(values))


def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))
