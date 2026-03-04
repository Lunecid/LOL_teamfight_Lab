from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────

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


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for AUC by resampling (y_true, y_prob) pairs.

    Uses stratified resampling to maintain the positive/negative class ratio
    in each bootstrap sample, preventing degenerate samples.

    Returns:
        (auc_point, ci_low, ci_high)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0 or n < 2:
        return float("nan"), float("nan"), float("nan")

    auc_point = float(roc_auc_score(y_true, y_prob))

    rng = np.random.RandomState(seed)
    boot_aucs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        # Stratified resampling: resample pos and neg independently
        idx_p = pos_idx[rng.randint(0, n_pos, size=n_pos)]
        idx_n = neg_idx[rng.randint(0, n_neg, size=n_neg)]
        idx = np.concatenate([idx_p, idx_n])
        boot_aucs[i] = roc_auc_score(y_true[idx], y_prob[idx])

    ci_low = float(np.percentile(boot_aucs, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return auc_point, ci_low, ci_high


def bootstrap_auc_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for paired AUC difference (AUC_A - AUC_B).

    Uses stratified resampling on the *same* bootstrap indices so the
    correlation between the two models' predictions is preserved.

    Returns:
        (delta_point, ci_low, ci_high)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=np.int32)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan")

    delta_point = float(roc_auc_score(y_true, pred_a) - roc_auc_score(y_true, pred_b))

    rng = np.random.RandomState(seed)
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx_p = pos_idx[rng.randint(0, n_pos, size=n_pos)]
        idx_n = neg_idx[rng.randint(0, n_neg, size=n_neg)]
        idx = np.concatenate([idx_p, idx_n])
        auc_a_i = roc_auc_score(y_true[idx], pred_a[idx])
        auc_b_i = roc_auc_score(y_true[idx], pred_b[idx])
        boot_deltas[i] = auc_a_i - auc_b_i

    ci_low = float(np.percentile(boot_deltas, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    return delta_point, ci_low, ci_high


# ──────────────────────────────────────────────────────────────
# DeLong's method — shared structural component computation
# ──────────────────────────────────────────────────────────────

def _delong_structural_components(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised DeLong structural components (placement values).

    Reference:
        DeLong et al. (1988), "Comparing the Areas under Two or More
        Correlated Receiver Operating Characteristic Curves".

    Args:
        y_true:      1-D array of binary labels.
        predictions: 2-D array of shape (K, N) where K is the number of
                     models and N the number of samples.

    Returns:
        (v10, v01, pos_idx, neg_idx)
        v10 — shape (K, n_pos): placement values for positive samples.
        v01 — shape (K, n_neg): placement values for negative samples.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    predictions = np.atleast_2d(np.asarray(predictions, dtype=np.float64))

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    k = predictions.shape[0]  # number of models

    v10 = np.empty((k, n_pos), dtype=np.float64)
    v01 = np.empty((k, n_neg), dtype=np.float64)

    for m in range(k):
        pred_pos = predictions[m, pos_idx]  # (n_pos,)
        pred_neg = predictions[m, neg_idx]  # (n_neg,)

        # v10[j] = fraction of negatives scored below positive j
        # Broadcasting: pred_pos[:, None] vs pred_neg[None, :]
        cmp = pred_pos[:, None] - pred_neg[None, :]  # (n_pos, n_neg)
        v10[m] = np.mean((cmp > 0).astype(np.float64) + 0.5 * (cmp == 0).astype(np.float64), axis=1)

        # v01[i] = fraction of positives scored above negative i
        cmp_t = pred_neg[:, None] - pred_pos[None, :]  # (n_neg, n_pos)
        v01[m] = np.mean((cmp_t < 0).astype(np.float64) + 0.5 * (cmp_t == 0).astype(np.float64), axis=1)

    return v10, v01, pos_idx, neg_idx


def _delong_variance(
    v10: np.ndarray,
    v01: np.ndarray,
    n_pos: int,
    n_neg: int,
) -> np.ndarray:
    """Covariance matrix of AUC estimates via DeLong's method.

    Args:
        v10: (K, n_pos) structural components for positives.
        v01: (K, n_neg) structural components for negatives.
        n_pos, n_neg: class counts.

    Returns:
        (K, K) covariance matrix of AUC estimates.
    """
    k = v10.shape[0]

    # S10: covariance among positive placement values
    s10 = np.cov(v10) if n_pos > 1 else np.zeros((k, k))
    # S01: covariance among negative placement values
    s01 = np.cov(v01) if n_neg > 1 else np.zeros((k, k))

    # Ensure 2-D even when k=1
    s10 = np.atleast_2d(s10)
    s01 = np.atleast_2d(s01)

    return s10 / n_pos + s01 / n_neg


# ──────────────────────────────────────────────────────────────
# DeLong's test — paired AUC comparison (two models)
# ──────────────────────────────────────────────────────────────

def delong_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> float:
    """DeLong's test for comparing two correlated ROC-AUC values.

    H0: AUC_A = AUC_B   (two-sided)

    Returns:
        p-value (float). NaN if inputs are degenerate.
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("[WARN] scipy not available for DeLong's test")
        return float("nan")

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    predictions = np.stack([pred_a, pred_b], axis=0)  # (2, N)
    v10, v01, _, _ = _delong_structural_components(y_true, predictions)

    auc_a = float(v10[0].mean())
    auc_b = float(v10[1].mean())

    cov_mat = _delong_variance(v10, v01, n_pos, n_neg)  # (2, 2)
    var_diff = cov_mat[0, 0] + cov_mat[1, 1] - 2 * cov_mat[0, 1]

    if var_diff <= 0:
        return 1.0 if abs(auc_a - auc_b) < 1e-10 else 0.0

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(p_value)


# ──────────────────────────────────────────────────────────────
# DeLong confidence interval — single AUC
# ──────────────────────────────────────────────────────────────

def delong_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Analytical DeLong confidence interval for a single AUC.

    Based on the asymptotic normality of the Mann-Whitney U statistic.

    Returns:
        (auc, ci_low, ci_high)
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan")

    predictions = y_prob[np.newaxis, :]  # (1, N)
    v10, v01, _, _ = _delong_structural_components(y_true, predictions)

    auc = float(v10[0].mean())
    var_auc = _delong_variance(v10, v01, n_pos, n_neg)[0, 0]

    if var_auc <= 0:
        return auc, auc, auc

    se = math.sqrt(var_auc)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_low = max(0.0, auc - z_crit * se)
    ci_high = min(1.0, auc + z_crit * se)
    return auc, float(ci_low), float(ci_high)


# ──────────────────────────────────────────────────────────────
# DeLong confidence interval — paired AUC difference
# ──────────────────────────────────────────────────────────────

def delong_paired_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Analytical DeLong CI for paired AUC difference (AUC_A - AUC_B).

    Returns:
        (delta, ci_low, ci_high, p_value)
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan"), float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    predictions = np.stack([pred_a, pred_b], axis=0)  # (2, N)
    v10, v01, _, _ = _delong_structural_components(y_true, predictions)

    auc_a = float(v10[0].mean())
    auc_b = float(v10[1].mean())
    delta = auc_a - auc_b

    cov_mat = _delong_variance(v10, v01, n_pos, n_neg)  # (2, 2)
    var_diff = cov_mat[0, 0] + cov_mat[1, 1] - 2 * cov_mat[0, 1]

    if var_diff <= 0:
        p_value = 1.0 if abs(delta) < 1e-10 else 0.0
        return delta, delta, delta, p_value

    se = math.sqrt(var_diff)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_low = delta - z_crit * se
    ci_high = delta + z_crit * se

    z = delta / se
    p_value = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    return delta, float(ci_low), float(ci_high), p_value


# ──────────────────────────────────────────────────────────────
# McNemar's test
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Multiple-testing correction
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def safe_mean(values: List[float], default: float = -1.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(values))


def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))
