"""Tests for core/common.py -- parsing, sigmoid, logit, safe_float, log1p_norm."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from core.common import (
    parse_csv_nums,
    parse_csv_str,
    sigmoid_np,
    logit,
    safe_float,
    log1p_norm,
)


# =========================================================
# parse_csv_nums
# =========================================================
class TestParseCsvNums:
    def test_valid_csv(self):
        result = parse_csv_nums("1, 2, 3.5")
        assert result == [1.0, 2.0, 3.5]

    def test_semicolons(self):
        result = parse_csv_nums("10;20;30")
        assert result == [10.0, 20.0, 30.0]

    def test_empty_string(self):
        assert parse_csv_nums("") == []

    def test_none(self):
        assert parse_csv_nums(None) == []

    def test_whitespace_only(self):
        assert parse_csv_nums("   ") == []

    def test_mixed_valid_invalid(self):
        result = parse_csv_nums("1, abc, 3")
        assert result == [1.0, 3.0]

    def test_cast_int(self):
        result = parse_csv_nums("1, 2, 3", cast=int)
        assert result == [1, 2, 3]


# =========================================================
# parse_csv_str
# =========================================================
class TestParseCsvStr:
    def test_valid_csv(self):
        result = parse_csv_str("hello, world, test")
        assert result == ["hello", "world", "test"]

    def test_semicolons(self):
        result = parse_csv_str("a;b;c")
        assert result == ["a", "b", "c"]

    def test_empty_string(self):
        assert parse_csv_str("") == []

    def test_none(self):
        assert parse_csv_str(None) == []

    def test_whitespace_stripped(self):
        result = parse_csv_str("  foo ,  bar  ")
        assert result == ["foo", "bar"]


# =========================================================
# sigmoid_np
# =========================================================
class TestSigmoidNp:
    def test_zero(self):
        result = sigmoid_np(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-9

    def test_positive(self):
        result = sigmoid_np(np.array([10.0]))
        assert result[0] > 0.99

    def test_negative(self):
        result = sigmoid_np(np.array([-10.0]))
        assert result[0] < 0.01

    def test_extreme_positive_no_overflow(self):
        result = sigmoid_np(np.array([1000.0]))
        assert np.isfinite(result[0])
        assert abs(result[0] - 1.0) < 1e-6

    def test_extreme_negative_no_overflow(self):
        result = sigmoid_np(np.array([-1000.0]))
        assert np.isfinite(result[0])
        assert abs(result[0] - 0.0) < 1e-6

    def test_vectorized(self):
        x = np.array([-5.0, 0.0, 5.0])
        result = sigmoid_np(x)
        assert result.shape == (3,)
        assert result[0] < 0.5
        assert abs(result[1] - 0.5) < 1e-9
        assert result[2] > 0.5


# =========================================================
# logit (inverse of sigmoid)
# =========================================================
class TestLogit:
    def test_half(self):
        result = logit(0.5)
        assert abs(result - 0.0) < 1e-5

    def test_inverse_of_sigmoid(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            sig_val = float(sigmoid_np(np.array([logit(p)]))[0])
            assert abs(sig_val - p) < 1e-4

    def test_near_zero_clipped(self):
        result = logit(0.0)
        assert np.isfinite(result)

    def test_near_one_clipped(self):
        result = logit(1.0)
        assert np.isfinite(result)


# =========================================================
# safe_float (common.py version -- different from utils.py)
# =========================================================
class TestSafeFloatCommon:
    def test_valid_int(self):
        assert safe_float(42) == 42.0

    def test_valid_float(self):
        assert safe_float(3.14) == 3.14

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_abc_returns_default(self):
        assert safe_float("abc") == 0.0

    def test_inf_returns_default(self):
        # common.safe_float specifically filters inf -> default
        assert safe_float(float("inf")) == 0.0

    def test_nan_returns_default(self):
        # common.safe_float specifically filters nan -> default
        assert safe_float(float("nan")) == 0.0

    def test_custom_default(self):
        assert safe_float("abc", default=-1.0) == -1.0


# =========================================================
# log1p_norm (common.py version -- preserves sign)
# =========================================================
class TestLog1pNormCommon:
    def test_positive_value(self):
        result = log1p_norm(100.0, 100.0)
        assert abs(result - 1.0) < 1e-9

    def test_zero(self):
        result = log1p_norm(0.0, 100.0)
        assert result == 0.0

    def test_negative_value_preserves_sign(self):
        # common.log1p_norm preserves sign, unlike utils.log1p_norm
        result = log1p_norm(-50.0, 100.0)
        assert result < 0.0

    def test_range_for_positive(self):
        result = log1p_norm(50.0, 100.0)
        assert 0.0 <= result <= 1.0

    def test_range_for_negative(self):
        result = log1p_norm(-50.0, 100.0)
        assert -1.0 <= result <= 0.0
