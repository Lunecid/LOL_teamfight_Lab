"""Tests for feature ablation analysis utilities.

Covers:
  - Static feature identification (feature_contract helpers)
  - Static temporal noise filtering
  - is_static_temporal_noise edge cases
"""
from __future__ import annotations

import numpy as np
import pytest

from core.feature_contract import (
    STATIC_NODE_FEATURE_PREFIXES,
    STATIC_GLOBAL_FEATURE_PREFIXES,
    TEMPORAL_NOISE_SUFFIXES,
    TABULAR_SUFFIXES,
    is_static_temporal_noise,
    filter_static_temporal_noise,
    tabular_feature_names,
)


# ─────────────────────────────────────────────────────────────
# is_static_temporal_noise
# ─────────────────────────────────────────────────────────────
class TestIsStaticTemporalNoise:
    """Test static feature identification in tabular feature names."""

    def test_rune_delta_is_noise(self):
        assert is_static_temporal_noise("bJNG_primary_rune_3__delta") is True

    def test_rune_std_is_noise(self):
        assert is_static_temporal_noise("rTOP_sub_rune_1__std") is True

    def test_rune_slope_is_noise(self):
        assert is_static_temporal_noise("bMID_primary_rune_2__slope") is True

    def test_rune_mean_is_not_noise(self):
        # mean is a constant suffix, not noise
        assert is_static_temporal_noise("bJNG_sub_rune_2__mean") is False

    def test_rune_last_is_not_noise(self):
        assert is_static_temporal_noise("bJNG_primary_rune_1__last") is False

    def test_champion_id_delta_is_noise(self):
        assert is_static_temporal_noise("bTOP_champion_id__delta") is True

    def test_champion_name_id_std_is_noise(self):
        assert is_static_temporal_noise("rBOT_champion_name_id__std") is True

    def test_summoner_spell_slope_is_noise(self):
        assert is_static_temporal_noise("bSUP_summoner_spell_1_id__slope") is True

    def test_stat_perk_delta_is_noise(self):
        assert is_static_temporal_noise("rMID_stat_perk_offense__delta") is True

    def test_dynamic_feature_delta_is_not_noise(self):
        # cs_cooldownReduction changes over time (items), NOT static
        assert is_static_temporal_noise("bJNG_cs_cooldownReduction__delta") is False

    def test_dynamic_feature_std_is_not_noise(self):
        assert is_static_temporal_noise("bJNG_hp_pct__std") is False

    def test_ban_delta_is_noise(self):
        assert is_static_temporal_noise("blue_ban_0__delta") is True

    def test_ban_std_is_noise(self):
        assert is_static_temporal_noise("red_ban_3__std") is True

    def test_no_suffix_returns_false(self):
        assert is_static_temporal_noise("bJNG_primary_rune_3") is False

    def test_empty_string(self):
        assert is_static_temporal_noise("") is False

    def test_primary_style_id(self):
        assert is_static_temporal_noise("bBOT_primary_style_id__slope") is True
        assert is_static_temporal_noise("rJNG_sub_style_id__delta") is True

    def test_gold_diff_is_not_noise(self):
        assert is_static_temporal_noise("goldDiff__delta") is False

    def test_time_norm_is_not_noise(self):
        assert is_static_temporal_noise("time_norm__slope") is False


# ─────────────────────────────────────────────────────────────
# filter_static_temporal_noise
# ─────────────────────────────────────────────────────────────
class TestFilterStaticTemporalNoise:
    """Test the filter that removes static temporal noise features."""

    def test_basic_filtering(self):
        names = [
            "bJNG_cs_cooldownReduction__max",   # keep (dynamic)
            "bJNG_primary_rune_3__delta",        # drop (static noise)
            "time_norm__last",                    # keep (dynamic)
            "rTOP_champion_id__std",              # drop (static noise)
            "bMID_hp_pct__mean",                  # keep (dynamic)
        ]
        keep_idx, dropped = filter_static_temporal_noise(names)
        assert keep_idx == (0, 2, 4)
        assert "bJNG_primary_rune_3__delta" in dropped
        assert "rTOP_champion_id__std" in dropped
        assert len(dropped) == 2

    def test_all_kept_when_no_static(self):
        names = ["goldDiff__mean", "hp_pct__std", "level_norm__slope"]
        keep_idx, dropped = filter_static_temporal_noise(names)
        assert keep_idx == (0, 1, 2)
        assert len(dropped) == 0

    def test_empty_input(self):
        keep_idx, dropped = filter_static_temporal_noise([])
        assert keep_idx == ()
        assert dropped == ()

    def test_all_slots_covered(self):
        """All 10 slot prefixes should be recognized."""
        slots = [
            "bTOP", "bJNG", "bMID", "bBOT", "bSUP",
            "rTOP", "rJNG", "rMID", "rBOT", "rSUP",
        ]
        for slot in slots:
            name = f"{slot}_champion_id__delta"
            assert is_static_temporal_noise(name), f"Failed for {name}"


# ─────────────────────────────────────────────────────────────
# Integration: tabular_feature_names × static filter
# ─────────────────────────────────────────────────────────────
class TestTabularStaticIntegration:
    """Verify that tabular_feature_names output correctly filters."""

    def test_node_features_generate_expected_noise(self):
        """When we generate tabular names for node features with slot prefix,
        static features should produce identifiable noise columns."""
        base_names = [
            "bJNG_champion_id",
            "bJNG_primary_rune_1",
            "bJNG_hp_pct",
        ]
        tab_names = list(tabular_feature_names(base_names))

        # Should have 3 * 7 = 21 features
        assert len(tab_names) == 21

        # Check noise identification
        noise_names = [n for n in tab_names if is_static_temporal_noise(n)]
        # champion_id has 3 noise suffixes (__std, __delta, __slope)
        # primary_rune_1 has 3 noise suffixes
        # hp_pct has 0 (not static)
        assert len(noise_names) == 6

        # Verify specific names
        assert "bJNG_champion_id__std" in noise_names
        assert "bJNG_champion_id__delta" in noise_names
        assert "bJNG_champion_id__slope" in noise_names
        assert "bJNG_primary_rune_1__std" in noise_names
        assert "bJNG_hp_pct__std" not in noise_names  # dynamic, not noise


# ─────────────────────────────────────────────────────────────
# Static attribute expected behavior validation
# ─────────────────────────────────────────────────────────────
_HAS_TORCH = False
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    pass


class TestStaticAttributeExpectedBehavior:
    """Validate expected behavior of seq_to_tabular on constant sequences."""

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_constant_sequence_produces_zero_noise(self):
        """A constant feature across all timesteps should yield
        std=0, delta=0, slope=0."""
        from gameplay.features import seq_to_tabular

        L, D = 12, 3
        x_seq = np.ones((L, D), dtype=np.float32) * 42.0

        result = seq_to_tabular(x_seq)
        # result layout: [last, mean, std, min, max, delta, slope]
        # Each has D elements
        last = result[0:D]
        mean = result[D:2*D]
        std = result[2*D:3*D]
        mn = result[3*D:4*D]
        mx = result[4*D:5*D]
        delta = result[5*D:6*D]
        slope = result[6*D:7*D]

        np.testing.assert_allclose(last, 42.0, atol=1e-6)
        np.testing.assert_allclose(mean, 42.0, atol=1e-6)
        np.testing.assert_allclose(std, 0.0, atol=1e-6)
        np.testing.assert_allclose(mn, 42.0, atol=1e-6)
        np.testing.assert_allclose(mx, 42.0, atol=1e-6)
        np.testing.assert_allclose(delta, 0.0, atol=1e-6)
        np.testing.assert_allclose(slope, 0.0, atol=1e-6)

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_varying_sequence_has_nonzero_stats(self):
        """A linearly increasing feature should have non-zero std/delta/slope."""
        from gameplay.features import seq_to_tabular

        L, D = 12, 1
        x_seq = np.arange(L, dtype=np.float32).reshape(L, D)

        result = seq_to_tabular(x_seq)
        std = result[2*D:3*D]
        delta = result[5*D:6*D]
        slope = result[6*D:7*D]

        assert float(std[0]) > 0.0, "Varying sequence should have non-zero std"
        assert float(delta[0]) > 0.0, "Increasing sequence should have positive delta"
        assert float(slope[0]) > 0.0, "Increasing sequence should have positive slope"
