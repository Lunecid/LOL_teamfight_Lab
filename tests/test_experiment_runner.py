"""Tests for experiment_runner.py -- bootstrap_ci() function."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from experiment_runner import bootstrap_ci


class TestBootstrapCi:
    """Test bootstrap_ci() with known values, single values, and CI coverage."""

    def test_known_values(self):
        """Mean of [1, 2, 3, 4, 5] should be 3.0."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=2000, seed=42)
        assert abs(mean - 3.0) < 1e-9

    def test_ci_order(self):
        """ci_low <= mean <= ci_high."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=2000, seed=42)
        assert ci_low <= mean <= ci_high

    def test_single_value_returns_same(self):
        """With a single value, mean, ci_low, and ci_high should all be that value."""
        values = [0.85]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert abs(mean - 0.85) < 1e-9
        assert abs(ci_low - 0.85) < 1e-9
        assert abs(ci_high - 0.85) < 1e-9

    def test_ci_covers_mean(self):
        """The CI should contain the sample mean."""
        values = [0.70, 0.72, 0.75, 0.73, 0.71]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=5000, seed=42)
        assert ci_low <= mean <= ci_high

    def test_identical_values(self):
        """If all values are the same, CI should collapse."""
        values = [0.80, 0.80, 0.80, 0.80, 0.80]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=1000, seed=42)
        assert abs(mean - 0.80) < 1e-9
        assert abs(ci_low - 0.80) < 1e-9
        assert abs(ci_high - 0.80) < 1e-9

    def test_wide_spread_gives_wider_ci(self):
        """More spread in the data should give a wider CI."""
        narrow = [4.9, 5.0, 5.1, 5.0, 5.0]
        wide = [1.0, 3.0, 5.0, 7.0, 9.0]

        _, n_low, n_high = bootstrap_ci(narrow, n_bootstrap=2000, seed=42)
        _, w_low, w_high = bootstrap_ci(wide, n_bootstrap=2000, seed=42)

        narrow_width = n_high - n_low
        wide_width = w_high - w_low
        assert wide_width > narrow_width

    def test_deterministic_with_same_seed(self):
        """Same seed should give same results."""
        values = [0.70, 0.72, 0.75, 0.73, 0.71]
        r1 = bootstrap_ci(values, n_bootstrap=1000, seed=123)
        r2 = bootstrap_ci(values, n_bootstrap=1000, seed=123)
        assert r1 == r2

    def test_empty_list(self):
        """Empty list should return (0.0, 0.0, 0.0) based on n<=1 branch."""
        mean, ci_low, ci_high = bootstrap_ci([], n_bootstrap=100, seed=42)
        assert mean == 0.0
        assert ci_low == 0.0
        assert ci_high == 0.0

    def test_two_values(self):
        """Two values should produce a valid CI."""
        values = [0.70, 0.90]
        mean, ci_low, ci_high = bootstrap_ci(values, n_bootstrap=2000, seed=42)
        assert abs(mean - 0.80) < 1e-9
        assert ci_low <= mean <= ci_high
