"""Tests for core/interpolation.py — alpha-curve functions."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from core.interpolation import (
    alpha_linear,
    alpha_cosine,
    alpha_exponential,
    alpha_cubic,
    remap_alpha,
    ALPHA_CURVES,
)


# =========================================================
# Boundary conditions: all curves must satisfy f(0)=0, f(1)=1
# =========================================================
class TestBoundaryConditions:
    """Every curve must map 0->0 and 1->~1."""

    @pytest.mark.parametrize("name", list(ALPHA_CURVES.keys()))
    def test_zero(self, name):
        fn = ALPHA_CURVES[name]
        assert fn(0.0, k=3.0) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("name", list(ALPHA_CURVES.keys()))
    def test_one(self, name):
        fn = ALPHA_CURVES[name]
        # exponential at t=1 with k=3 gives ~0.95, not 1.0
        if name in ("exponential", "exp"):
            assert fn(1.0, k=3.0) == pytest.approx(1.0 - math.exp(-3.0), abs=1e-9)
        else:
            assert fn(1.0, k=3.0) == pytest.approx(1.0, abs=1e-9)


# =========================================================
# Monotonicity: all curves should be non-decreasing on [0,1]
# =========================================================
class TestMonotonicity:
    @pytest.mark.parametrize("name", list(ALPHA_CURVES.keys()))
    def test_monotone(self, name):
        fn = ALPHA_CURVES[name]
        ts = np.linspace(0.0, 1.0, 200)
        vals = [fn(float(t), k=3.0) for t in ts]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-12, (
                f"{name}: not monotone at t={ts[i]:.4f}"
            )


# =========================================================
# Individual curve tests
# =========================================================
class TestLinear:
    def test_midpoint(self):
        assert alpha_linear(0.5) == pytest.approx(0.5)

    def test_quarter(self):
        assert alpha_linear(0.25) == pytest.approx(0.25)


class TestCosine:
    def test_midpoint(self):
        assert alpha_cosine(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_quarter_below_linear(self):
        # cosine ease-in starts slow, so at t=0.25 alpha < 0.25
        assert alpha_cosine(0.25) < 0.25

    def test_three_quarter_above_linear(self):
        # cosine ease-out ends slow, so at t=0.75 alpha > 0.75
        assert alpha_cosine(0.75) > 0.75


class TestExponential:
    def test_known_values(self):
        """Verify the documented key points for k=3."""
        assert alpha_exponential(0.0, k=3.0) == pytest.approx(0.0, abs=1e-9)
        assert alpha_exponential(0.3, k=3.0) == pytest.approx(
            1.0 - math.exp(-0.9), abs=1e-6
        )
        assert alpha_exponential(0.5, k=3.0) == pytest.approx(
            1.0 - math.exp(-1.5), abs=1e-6
        )
        assert alpha_exponential(1.0, k=3.0) == pytest.approx(
            1.0 - math.exp(-3.0), abs=1e-6
        )

    def test_higher_k_faster(self):
        """Higher k should produce larger alpha at any t in (0,1)."""
        t = 0.5
        assert alpha_exponential(t, k=5.0) > alpha_exponential(t, k=3.0)

    def test_accelerates_toward_target(self):
        """At t=0.5 with k=3, should be ~78% toward target (above linear)."""
        a = alpha_exponential(0.5, k=3.0)
        assert a > 0.75  # 78% > 75%
        assert a < 0.85


class TestCubic:
    def test_midpoint(self):
        assert alpha_cubic(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_symmetry(self):
        """Smoothstep is symmetric: f(t) + f(1-t) = 1."""
        for t in [0.1, 0.2, 0.3, 0.4]:
            assert alpha_cubic(t) + alpha_cubic(1.0 - t) == pytest.approx(
                1.0, abs=1e-12
            )

    def test_zero_derivative_endpoints(self):
        """f'(0) = f'(1) = 0 for smoothstep 3t^2 - 2t^3."""
        # Numerical derivative at endpoints
        eps = 1e-8
        d0 = (alpha_cubic(eps) - alpha_cubic(0.0)) / eps
        d1 = (alpha_cubic(1.0) - alpha_cubic(1.0 - eps)) / eps
        assert d0 == pytest.approx(0.0, abs=1e-4)
        assert d1 == pytest.approx(0.0, abs=1e-4)


# =========================================================
# remap_alpha integration
# =========================================================
class TestRemapAlpha:
    def test_clamp_below(self):
        # negative t should clamp to 0
        assert remap_alpha(-0.5, curve="linear") == pytest.approx(0.0, abs=1e-9)

    def test_unknown_curve_falls_back_to_linear(self):
        assert remap_alpha(0.5, curve="nonexistent_method") == pytest.approx(0.5)

    def test_aliases(self):
        assert remap_alpha(0.5, curve="exp", k=3.0) == pytest.approx(
            remap_alpha(0.5, curve="exponential", k=3.0)
        )
        assert remap_alpha(0.5, curve="smoothstep") == pytest.approx(
            remap_alpha(0.5, curve="cubic")
        )

    def test_all_curves_in_unit_range(self):
        """For t in [0,1], all curves must return alpha in [0,1]."""
        ts = np.linspace(0.0, 1.0, 100)
        for name in ALPHA_CURVES:
            for t in ts:
                a = remap_alpha(float(t), curve=name, k=3.0)
                assert 0.0 <= a <= 1.0, f"{name} at t={t}: alpha={a}"
