"""Tests for core/utils.py -- numeric helpers, metrics, scaler, sanitization."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np
import torch

from core.utils import (
    safe_float,
    log1p_norm,
    clamp01,
    _auc_rank,
    average_precision,
    brier_score,
    recall_at_precision,
    metrics_from_probs,
    sanitize_feature_name,
    SeqScaler,
)


# =========================================================
# safe_float
# =========================================================
class TestSafeFloat:
    def test_valid_int(self):
        assert safe_float(42) == 42.0

    def test_valid_float(self):
        assert safe_float(3.14) == 3.14

    def test_valid_string(self):
        assert safe_float("2.5") == 2.5

    def test_none_returns_default(self):
        # utils.safe_float does not special-case None -- it goes through
        # the try/except. float(None) raises TypeError, so default is returned.
        assert safe_float(None) == 0.0

    def test_abc_returns_default(self):
        assert safe_float("abc") == 0.0

    def test_custom_default(self):
        assert safe_float("abc", default=-1.0) == -1.0

    def test_inf(self):
        # [BUG FIX] safe_float now correctly rejects inf, returning default.
        result = safe_float(float("inf"))
        assert result == 0.0

    def test_nan(self):
        # [BUG FIX] safe_float now correctly rejects nan, returning default.
        result = safe_float(float("nan"))
        assert result == 0.0


# =========================================================
# log1p_norm
# =========================================================
class TestLog1pNorm:
    def test_positive_value(self):
        result = log1p_norm(100, 100)
        # log1p(100) / log1p(100) == 1.0
        assert abs(result - 1.0) < 1e-9

    def test_zero(self):
        result = log1p_norm(0, 100)
        assert result == 0.0

    def test_negative_preserves_sign(self):
        # [BUG FIX] log1p_norm now preserves sign of negative values
        # sign(x) * log1p(|x|) / log1p(denom)
        result = log1p_norm(-50, 100)
        assert result < 0.0
        assert result > -1.0

    def test_output_range(self):
        result = log1p_norm(50, 100)
        assert 0.0 <= result <= 1.0


# =========================================================
# clamp01
# =========================================================
class TestClamp01:
    def test_within_range(self):
        assert clamp01(0.5) == 0.5

    def test_below_zero(self):
        assert clamp01(-0.5) == 0.0

    def test_above_one(self):
        assert clamp01(1.5) == 1.0

    def test_exact_boundaries(self):
        assert clamp01(0.0) == 0.0
        assert clamp01(1.0) == 1.0

    def test_non_numeric_returns_zero(self):
        # safe_float("abc") returns 0.0, then clamped to [0,1]
        assert clamp01("abc") == 0.0


# =========================================================
# _auc_rank
# =========================================================
class TestAucRank:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _auc_rank(y_true, y_score)
        assert abs(auc - 1.0) < 1e-9

    def test_worst_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _auc_rank(y_true, y_score)
        assert abs(auc - 0.0) < 1e-9

    def test_random_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        auc = _auc_rank(y_true, y_score)
        assert abs(auc - 0.5) < 1e-9

    def test_tied_scores_handled(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.3, 0.3, 0.7, 0.7, 0.7, 0.3])
        auc = _auc_rank(y_true, y_score)
        # Should not crash and should return a valid float
        assert 0.0 <= auc <= 1.0

    def test_no_positives_returns_nan(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        auc = _auc_rank(y_true, y_score)
        assert math.isnan(auc)

    def test_no_negatives_returns_nan(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3])
        auc = _auc_rank(y_true, y_score)
        assert math.isnan(auc)


# =========================================================
# average_precision
# =========================================================
class TestAveragePrecision:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        ap = average_precision(y_true, y_prob)
        assert abs(ap - 1.0) < 1e-9

    def test_no_positives_returns_nan(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.5, 0.9])
        ap = average_precision(y_true, y_prob)
        assert math.isnan(ap)

    def test_all_positive(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.3, 0.5, 0.7])
        ap = average_precision(y_true, y_prob)
        assert abs(ap - 1.0) < 1e-9


# =========================================================
# brier_score
# =========================================================
class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        bs = brier_score(y_true, y_prob)
        assert abs(bs - 0.0) < 1e-9

    def test_worst_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        bs = brier_score(y_true, y_prob)
        assert abs(bs - 1.0) < 1e-9

    def test_midpoint(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        bs = brier_score(y_true, y_prob)
        assert abs(bs - 0.25) < 1e-9


# =========================================================
# recall_at_precision
# =========================================================
class TestRecallAtPrecision:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        recall = recall_at_precision(y_true, y_prob, min_precision=0.95)
        assert recall > 0.0

    def test_no_positives_returns_nan(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.5, 0.9])
        recall = recall_at_precision(y_true, y_prob, min_precision=0.95)
        assert math.isnan(recall)

    def test_all_correct_recall_is_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        recall = recall_at_precision(y_true, y_prob, min_precision=0.99)
        assert abs(recall - 1.0) < 1e-9


# =========================================================
# metrics_from_probs
# =========================================================
class TestMetricsFromProbs:
    def test_returns_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        m = metrics_from_probs(y_true, y_prob)
        expected_keys = {"acc", "precision", "recall", "f1", "auc", "ap", "brier",
                         "tp", "tn", "fp", "fn"}
        assert expected_keys.issubset(set(m.keys()))

    def test_values_are_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        m = metrics_from_probs(y_true, y_prob)
        for k, v in m.items():
            assert isinstance(v, float), f"Key {k} is not float: {type(v)}"


# =========================================================
# sanitize_feature_name
# =========================================================
class TestSanitizeFeatureName:
    def test_removes_braces(self):
        assert "{" not in sanitize_feature_name("feat{1}")
        assert "}" not in sanitize_feature_name("feat{1}")

    def test_removes_brackets(self):
        assert "[" not in sanitize_feature_name("feat[0]")
        assert "]" not in sanitize_feature_name("feat[0]")

    def test_removes_colon(self):
        assert ":" not in sanitize_feature_name("key:value")

    def test_removes_quotes(self):
        assert '"' not in sanitize_feature_name('feat"name')
        assert "'" not in sanitize_feature_name("feat'name")

    def test_clean_name_unchanged(self):
        assert sanitize_feature_name("gold_diff_norm") == "gold_diff_norm"

    def test_removes_whitespace_chars(self):
        result = sanitize_feature_name("feat\nname\twith\rspaces")
        assert "\n" not in result
        assert "\t" not in result
        assert "\r" not in result


# =========================================================
# SeqScaler round-trip
# =========================================================
class TestSeqScaler:
    def test_state_dict_roundtrip(self):
        scaler = SeqScaler(scaler_type="standard")
        scaler.fitted = True
        scaler.node_mean = torch.tensor([1.0, 2.0, 3.0])
        scaler.node_std = torch.tensor([0.5, 0.5, 0.5])
        scaler.extra_mean = torch.tensor([10.0, 20.0])
        scaler.extra_std = torch.tensor([2.0, 3.0])

        state = scaler.state_dict()
        restored = SeqScaler.from_state_dict(state)

        assert restored.fitted is True
        assert restored.scaler_type == "standard"
        assert torch.allclose(restored.node_mean, scaler.node_mean)
        assert torch.allclose(restored.node_std, scaler.node_std)
        assert torch.allclose(restored.extra_mean, scaler.extra_mean)
        assert torch.allclose(restored.extra_std, scaler.extra_std)

    def test_state_dict_unfitted(self):
        scaler = SeqScaler(scaler_type="standard")
        state = scaler.state_dict()
        restored = SeqScaler.from_state_dict(state)
        assert restored.fitted is False
        assert restored.node_mean is None

    def test_exclude_prefixes_preserved(self):
        scaler = SeqScaler(exclude_prefixes=("x_", "y_", "custom_"))
        state = scaler.state_dict()
        restored = SeqScaler.from_state_dict(state)
        assert "custom_" in restored.exclude_prefixes
