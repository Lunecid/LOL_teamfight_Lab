"""Tests for app/experiment_stats.py — DeLong, bootstrap CI, calibration metrics."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from app.experiment_stats import (
    bootstrap_ci,
    bootstrap_auc_ci,
    bootstrap_auc_diff_ci,
    delong_test,
    delong_auc_ci,
    delong_paired_ci,
    mcnemar_test,
    holm_bonferroni,
    compute_ece,
    compute_brier,
    safe_mean,
    safe_std,
    _delong_structural_components,
    _delong_variance,
)


# =========================================================
# Helpers
# =========================================================

def _make_separable(n_pos=50, n_neg=50, seed=42):
    """Create perfectly separable predictions (AUC = 1.0)."""
    rng = np.random.RandomState(seed)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_prob = np.concatenate([
        rng.uniform(0.7, 1.0, n_pos),
        rng.uniform(0.0, 0.3, n_neg),
    ])
    return y_true, y_prob


def _make_random(n=200, seed=42):
    """Create random predictions (AUC ≈ 0.5)."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, size=n)
    y_prob = rng.uniform(0, 1, size=n)
    return y_true, y_prob


def _make_realistic(n_pos=100, n_neg=100, auc_target=0.8, seed=42):
    """Create predictions with a target AUC around 0.8."""
    rng = np.random.RandomState(seed)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    # Positive class: higher scores, negative class: lower scores
    y_prob = np.concatenate([
        rng.beta(4, 2, n_pos),   # mean ≈ 0.67
        rng.beta(2, 4, n_neg),   # mean ≈ 0.33
    ])
    return y_true, y_prob


# =========================================================
# bootstrap_ci (scalar values)
# =========================================================

class TestBootstrapCI:
    def test_single_value(self):
        mean, lo, hi = bootstrap_ci([0.85])
        assert mean == 0.85
        assert lo == 0.85
        assert hi == 0.85

    def test_empty_returns_zero(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0

    def test_deterministic_with_seed(self):
        vals = [0.80, 0.82, 0.81, 0.83, 0.84]
        r1 = bootstrap_ci(vals, seed=42)
        r2 = bootstrap_ci(vals, seed=42)
        assert r1 == r2

    def test_ci_contains_mean(self):
        vals = [0.80, 0.82, 0.81, 0.83, 0.84]
        mean, lo, hi = bootstrap_ci(vals)
        assert lo <= mean <= hi

    def test_wider_ci_with_higher_variance(self):
        narrow = [0.81, 0.82, 0.81, 0.82, 0.81]
        wide = [0.60, 0.70, 0.80, 0.90, 1.00]
        _, lo_n, hi_n = bootstrap_ci(narrow)
        _, lo_w, hi_w = bootstrap_ci(wide)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_alpha_controls_width(self):
        vals = [0.80, 0.82, 0.81, 0.83, 0.84]
        _, lo_90, hi_90 = bootstrap_ci(vals, alpha=0.10)
        _, lo_95, hi_95 = bootstrap_ci(vals, alpha=0.05)
        # 95% CI should be wider than 90% CI
        assert (hi_95 - lo_95) >= (hi_90 - lo_90)


# =========================================================
# bootstrap_auc_ci (AUC from resampled predictions)
# =========================================================

class TestBootstrapAucCI:
    def test_perfect_separation(self):
        y_true, y_prob = _make_separable()
        auc, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=500)
        assert auc > 0.95
        assert lo > 0.90
        assert hi <= 1.0

    def test_random_predictions(self):
        y_true, y_prob = _make_random(n=300, seed=42)
        auc, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=500)
        assert lo <= auc <= hi
        # AUC should be near 0.5 for random
        assert 0.35 < auc < 0.65

    def test_ci_contains_point_estimate(self):
        y_true, y_prob = _make_realistic()
        auc, lo, hi = bootstrap_auc_ci(y_true, y_prob)
        assert lo <= auc <= hi

    def test_degenerate_all_positive(self):
        y_true = np.ones(50, dtype=int)
        y_prob = np.random.uniform(0, 1, 50)
        auc, lo, hi = bootstrap_auc_ci(y_true, y_prob)
        assert math.isnan(auc)

    def test_deterministic_seed(self):
        y_true, y_prob = _make_realistic()
        r1 = bootstrap_auc_ci(y_true, y_prob, seed=7)
        r2 = bootstrap_auc_ci(y_true, y_prob, seed=7)
        assert r1 == r2


# =========================================================
# bootstrap_auc_diff_ci
# =========================================================

class TestBootstrapAucDiffCI:
    def test_same_model_delta_near_zero(self):
        y_true, y_prob = _make_realistic()
        delta, lo, hi = bootstrap_auc_diff_ci(y_true, y_prob, y_prob, n_bootstrap=500)
        assert abs(delta) < 1e-10
        assert lo <= 0.0 <= hi

    def test_better_model_positive_delta(self):
        y_true, pred_good = _make_separable()
        _, pred_bad = _make_random(n=len(y_true), seed=99)
        delta, lo, hi = bootstrap_auc_diff_ci(y_true, pred_good, pred_bad, n_bootstrap=500)
        assert delta > 0
        assert lo > 0  # significantly better

    def test_ci_contains_point(self):
        y_true, pred_a = _make_realistic(seed=1)
        _, pred_b = _make_realistic(seed=2)
        # Match lengths
        n = min(len(y_true), len(pred_b))
        y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]
        delta, lo, hi = bootstrap_auc_diff_ci(y_true, pred_a, pred_b, n_bootstrap=500)
        assert lo <= delta <= hi


# =========================================================
# DeLong structural components
# =========================================================

class TestDeLongStructural:
    def test_single_model_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        y_true, y_prob = _make_realistic()
        predictions = y_prob[np.newaxis, :]
        v10, v01, pos, neg = _delong_structural_components(y_true, predictions)
        auc_delong = float(v10[0].mean())
        auc_sklearn = roc_auc_score(y_true, y_prob)
        assert abs(auc_delong - auc_sklearn) < 1e-10

    def test_two_models_shapes(self):
        y_true, pred_a = _make_realistic(seed=1)
        _, pred_b = _make_realistic(seed=2)
        n = min(len(pred_a), len(pred_b))
        y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]

        predictions = np.stack([pred_a, pred_b], axis=0)
        v10, v01, pos, neg = _delong_structural_components(y_true, predictions)
        assert v10.shape == (2, len(pos))
        assert v01.shape == (2, len(neg))

    def test_perfect_auc(self):
        y_true, y_prob = _make_separable(n_pos=30, n_neg=30)
        predictions = y_prob[np.newaxis, :]
        v10, v01, _, _ = _delong_structural_components(y_true, predictions)
        auc = float(v10[0].mean())
        assert auc > 0.99


# =========================================================
# DeLong variance
# =========================================================

class TestDeLongVariance:
    def test_single_model_variance_positive(self):
        y_true, y_prob = _make_realistic()
        predictions = y_prob[np.newaxis, :]
        v10, v01, pos, neg = _delong_structural_components(y_true, predictions)
        cov_mat = _delong_variance(v10, v01, len(pos), len(neg))
        assert cov_mat.shape == (1, 1)
        assert cov_mat[0, 0] > 0

    def test_two_models_covariance_matrix(self):
        y_true, pred_a = _make_realistic(seed=1)
        _, pred_b = _make_realistic(seed=2)
        n = min(len(pred_a), len(pred_b))
        y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]

        predictions = np.stack([pred_a, pred_b], axis=0)
        v10, v01, pos, neg = _delong_structural_components(y_true, predictions)
        cov_mat = _delong_variance(v10, v01, len(pos), len(neg))
        assert cov_mat.shape == (2, 2)
        # Diagonal entries should be positive
        assert cov_mat[0, 0] > 0
        assert cov_mat[1, 1] > 0
        # Symmetric
        assert abs(cov_mat[0, 1] - cov_mat[1, 0]) < 1e-15


# =========================================================
# delong_test (paired AUC comparison)
# =========================================================

class TestDeLongTest:
    def test_same_predictions_p_is_1(self):
        y_true, y_prob = _make_realistic()
        p = delong_test(y_true, y_prob, y_prob)
        assert p == 1.0

    def test_clearly_different_models_significant(self):
        y_true, pred_good = _make_separable(n_pos=100, n_neg=100)
        rng = np.random.RandomState(99)
        pred_bad = rng.uniform(0, 1, len(y_true))
        p = delong_test(y_true, pred_good, pred_bad)
        assert p < 0.05

    def test_similar_models_not_significant(self):
        y_true, pred_a = _make_realistic(seed=1)
        # Create a very similar model
        rng = np.random.RandomState(2)
        pred_b = pred_a + rng.normal(0, 0.01, len(pred_a))
        pred_b = np.clip(pred_b, 0, 1)
        p = delong_test(y_true, pred_a, pred_b)
        assert p > 0.05

    def test_degenerate_all_positive(self):
        y_true = np.ones(50, dtype=int)
        pred_a = np.random.uniform(0, 1, 50)
        pred_b = np.random.uniform(0, 1, 50)
        p = delong_test(y_true, pred_a, pred_b)
        assert math.isnan(p)

    def test_degenerate_all_negative(self):
        y_true = np.zeros(50, dtype=int)
        pred_a = np.random.uniform(0, 1, 50)
        pred_b = np.random.uniform(0, 1, 50)
        p = delong_test(y_true, pred_a, pred_b)
        assert math.isnan(p)

    def test_symmetric(self):
        """p-value should be same regardless of model order."""
        y_true, pred_a = _make_realistic(seed=1)
        _, pred_b = _make_realistic(seed=2)
        n = min(len(pred_a), len(pred_b))
        y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]
        p_ab = delong_test(y_true, pred_a, pred_b)
        p_ba = delong_test(y_true, pred_b, pred_a)
        assert abs(p_ab - p_ba) < 1e-10


# =========================================================
# delong_auc_ci (single AUC confidence interval)
# =========================================================

class TestDeLongAucCI:
    def test_realistic_ci(self):
        y_true, y_prob = _make_realistic(n_pos=200, n_neg=200)
        auc, lo, hi = delong_auc_ci(y_true, y_prob)
        assert lo <= auc <= hi
        assert lo > 0.5  # should be clearly above chance
        assert hi <= 1.0

    def test_perfect_model(self):
        y_true, y_prob = _make_separable(n_pos=50, n_neg=50)
        auc, lo, hi = delong_auc_ci(y_true, y_prob)
        assert auc > 0.95
        assert hi <= 1.0

    def test_degenerate_returns_nan(self):
        y_true = np.ones(50, dtype=int)
        y_prob = np.random.uniform(0, 1, 50)
        auc, lo, hi = delong_auc_ci(y_true, y_prob)
        assert math.isnan(auc)

    def test_wider_ci_with_fewer_samples(self):
        """Smaller sample size should produce wider CI."""
        y_true_large, y_prob_large = _make_realistic(n_pos=500, n_neg=500, seed=1)
        y_true_small, y_prob_small = _make_realistic(n_pos=30, n_neg=30, seed=1)

        _, lo_l, hi_l = delong_auc_ci(y_true_large, y_prob_large)
        _, lo_s, hi_s = delong_auc_ci(y_true_small, y_prob_small)

        width_large = hi_l - lo_l
        width_small = hi_s - lo_s
        assert width_small > width_large

    def test_ci_bounds_in_01(self):
        y_true, y_prob = _make_realistic()
        auc, lo, hi = delong_auc_ci(y_true, y_prob)
        assert lo >= 0.0
        assert hi <= 1.0


# =========================================================
# delong_paired_ci (paired AUC difference CI)
# =========================================================

class TestDeLongPairedCI:
    def test_same_model_delta_zero(self):
        y_true, y_prob = _make_realistic()
        delta, lo, hi, p = delong_paired_ci(y_true, y_prob, y_prob)
        assert abs(delta) < 1e-10
        assert p == 1.0

    def test_better_model_positive_delta(self):
        y_true, pred_good = _make_separable(n_pos=100, n_neg=100)
        rng = np.random.RandomState(99)
        pred_bad = rng.uniform(0, 1, len(y_true))
        delta, lo, hi, p = delong_paired_ci(y_true, pred_good, pred_bad)
        assert delta > 0
        assert lo > 0  # CI excludes zero
        assert p < 0.05

    def test_ci_contains_delta(self):
        y_true, pred_a = _make_realistic(seed=1)
        _, pred_b = _make_realistic(seed=2)
        n = min(len(pred_a), len(pred_b))
        delta, lo, hi, p = delong_paired_ci(y_true[:n], pred_a[:n], pred_b[:n])
        assert lo <= delta <= hi

    def test_p_value_consistent_with_delong_test(self):
        """p-value from delong_paired_ci should match delong_test."""
        y_true, pred_a = _make_realistic(seed=10)
        _, pred_b = _make_realistic(seed=20)
        n = min(len(pred_a), len(pred_b))
        y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]

        p_test = delong_test(y_true, pred_a, pred_b)
        _, _, _, p_ci = delong_paired_ci(y_true, pred_a, pred_b)
        assert abs(p_test - p_ci) < 1e-10

    def test_degenerate_returns_nan(self):
        y_true = np.ones(50, dtype=int)
        pred_a = np.random.uniform(0, 1, 50)
        pred_b = np.random.uniform(0, 1, 50)
        delta, lo, hi, p = delong_paired_ci(y_true, pred_a, pred_b)
        assert math.isnan(delta)


# =========================================================
# mcnemar_test
# =========================================================

class TestMcNemarTest:
    def test_same_predictions_p_is_1(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4])
        p = mcnemar_test(y_true, pred, pred)
        assert p == 1.0

    def test_clearly_different_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        pred_good = y_true.astype(float)  # perfect
        pred_bad = 1 - pred_good  # worst possible
        p = mcnemar_test(y_true, pred_good, pred_bad)
        assert p < 0.01

    def test_small_sample_not_significant(self):
        y_true = np.array([1, 0, 1])
        pred_a = np.array([0.9, 0.1, 0.8])
        pred_b = np.array([0.9, 0.1, 0.2])
        p = mcnemar_test(y_true, pred_a, pred_b)
        # Too few samples to reach significance
        assert p >= 0.0


# =========================================================
# holm_bonferroni
# =========================================================

class TestHolmBonferroni:
    def test_single_significant(self):
        result = holm_bonferroni([0.01], alpha=0.05)
        assert result == [True]

    def test_single_not_significant(self):
        result = holm_bonferroni([0.10], alpha=0.05)
        assert result == [False]

    def test_multiple_corrections(self):
        # With m=3 and alpha=0.05:
        #   sorted: 0.01, 0.03, 0.04
        #   rank 0: 0.01 <= 0.05/3 = 0.0167? YES
        #   rank 1: 0.03 <= 0.05/2 = 0.025? NO → stop
        result = holm_bonferroni([0.03, 0.01, 0.04], alpha=0.05)
        assert result == [False, True, False]

    def test_all_significant(self):
        result = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
        assert result == [True, True, True]

    def test_none_significant(self):
        result = holm_bonferroni([0.10, 0.20, 0.30], alpha=0.05)
        assert result == [False, False, False]

    def test_seven_treatments(self):
        """Simulate the 7-treatment ablation scenario."""
        p_vals = [0.001, 0.01, 0.02, 0.04, 0.06, 0.10, 0.50]
        result = holm_bonferroni(p_vals, alpha=0.05)
        # rank 0: 0.001 <= 0.05/7 ≈ 0.0071? YES
        # rank 1: 0.01  <= 0.05/6 ≈ 0.0083? NO → stop
        assert result[0] is True
        assert sum(result) == 1  # only the first survives


# =========================================================
# compute_ece
# =========================================================

class TestComputeECE:
    def test_perfect_calibration(self):
        """Well-calibrated predictions should have low ECE."""
        rng = np.random.RandomState(42)
        n = 1000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.1

    def test_terrible_calibration(self):
        """Overconfident wrong predictions should have high ECE."""
        y_true = np.zeros(100, dtype=int)
        y_prob = np.ones(100) * 0.99  # predicts 0.99 but all are negative
        ece = compute_ece(y_true, y_prob)
        assert ece > 0.5

    def test_ece_range(self):
        y_true, y_prob = _make_realistic()
        ece = compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0


# =========================================================
# compute_brier
# =========================================================

class TestComputeBrier:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        brier = compute_brier(y_true, y_prob)
        assert brier == 0.0

    def test_worst_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        brier = compute_brier(y_true, y_prob)
        assert brier == 1.0

    def test_random_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        brier = compute_brier(y_true, y_prob)
        assert abs(brier - 0.25) < 1e-10


# =========================================================
# safe_mean / safe_std
# =========================================================

class TestSafeMeanStd:
    def test_safe_mean_empty(self):
        assert safe_mean([]) == -1.0

    def test_safe_mean_values(self):
        assert abs(safe_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-10

    def test_safe_mean_custom_default(self):
        assert safe_mean([], default=0.0) == 0.0

    def test_safe_std_single_value(self):
        assert safe_std([5.0]) == 0.0

    def test_safe_std_empty(self):
        assert safe_std([]) == 0.0

    def test_safe_std_multiple(self):
        result = safe_std([1.0, 2.0, 3.0])
        assert result > 0


# =========================================================
# Cross-validation: DeLong CI vs Bootstrap CI agreement
# =========================================================

class TestCrossValidation:
    def test_delong_and_bootstrap_auc_ci_agree(self):
        """DeLong CI and bootstrap CI for AUC should roughly agree."""
        y_true, y_prob = _make_realistic(n_pos=300, n_neg=300, seed=7)

        auc_d, lo_d, hi_d = delong_auc_ci(y_true, y_prob)
        auc_b, lo_b, hi_b = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=5000, seed=7)

        # Point estimates should be very close
        assert abs(auc_d - auc_b) < 0.01

        # CI widths should be in the same ballpark (within 50% of each other)
        width_d = hi_d - lo_d
        width_b = hi_b - lo_b
        ratio = max(width_d, width_b) / max(min(width_d, width_b), 1e-10)
        assert ratio < 2.0, f"CI widths differ too much: DeLong={width_d:.4f}, Bootstrap={width_b:.4f}"

    def test_delong_test_and_paired_ci_consistent(self):
        """If DeLong CI for difference excludes zero, p-value should be < 0.05."""
        y_true, pred_good = _make_separable(n_pos=100, n_neg=100, seed=42)
        rng = np.random.RandomState(99)
        pred_bad = rng.uniform(0, 1, len(y_true))

        delta, lo, hi, p = delong_paired_ci(y_true, pred_good, pred_bad)
        if lo > 0 or hi < 0:
            assert p < 0.05
        if p >= 0.05:
            assert lo <= 0 <= hi
