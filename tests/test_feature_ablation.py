"""Tests for feature ablation analysis utilities.

Covers:
  - Static feature identification (feature_contract helpers)
  - Static temporal noise filtering (legacy)
  - Constant/quasi-constant redundant feature filtering (comprehensive)
  - is_static_temporal_noise edge cases
  - classify_feature_constancy
"""
from __future__ import annotations

import numpy as np
import pytest

from core.feature_contract import (
    STATIC_NODE_FEATURE_PREFIXES,
    STATIC_GLOBAL_FEATURE_PREFIXES,
    TEMPORAL_NOISE_SUFFIXES,
    REDUNDANT_SUFFIXES_FOR_CONSTANT,
    QUASI_CONSTANT_NODE_FEATURE_PREFIXES,
    QUASI_CONSTANT_EXTRA_FEATURE_PREFIXES,
    WITHIN_FIGHT_CONSTANT_NODE_FEATURE_PREFIXES,
    TABULAR_SUFFIXES,
    is_static_temporal_noise,
    is_constant_redundant,
    is_quasi_constant_redundant,
    is_within_fight_constant_redundant,
    classify_feature_constancy,
    filter_static_temporal_noise,
    filter_constant_and_quasi_constant,
    tabular_feature_names,
)


# ─────────────────────────────────────────────────────────────
# is_static_temporal_noise (legacy — std/delta/slope only)
# ─────────────────────────────────────────────────────────────
class TestIsStaticTemporalNoise:
    """Test static feature identification in tabular feature names."""

    def test_rune_delta_is_noise(self):
        assert is_static_temporal_noise("bJNG_primary_rune_3__delta") is True

    def test_rune_std_is_noise(self):
        assert is_static_temporal_noise("rTOP_sub_rune_1__std") is True

    def test_rune_slope_is_noise(self):
        assert is_static_temporal_noise("bMID_primary_rune_2__slope") is True

    def test_rune_mean_is_not_noise_legacy(self):
        # Legacy function: __mean is NOT in TEMPORAL_NOISE_SUFFIXES
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
# is_constant_redundant (comprehensive — all 6 non-last suffixes)
# ─────────────────────────────────────────────────────────────
class TestIsConstantRedundant:
    """Test comprehensive redundant constant feature identification."""

    def test_champion_id_mean_is_redundant(self):
        assert is_constant_redundant("bJNG_champion_id__mean") is True

    def test_champion_id_std_is_redundant(self):
        assert is_constant_redundant("bJNG_champion_id__std") is True

    def test_champion_id_min_is_redundant(self):
        assert is_constant_redundant("bTOP_champion_id__min") is True

    def test_champion_id_max_is_redundant(self):
        assert is_constant_redundant("rBOT_champion_id__max") is True

    def test_champion_id_delta_is_redundant(self):
        assert is_constant_redundant("bTOP_champion_id__delta") is True

    def test_champion_id_slope_is_redundant(self):
        assert is_constant_redundant("rMID_champion_id__slope") is True

    def test_champion_id_last_is_kept(self):
        assert is_constant_redundant("bJNG_champion_id__last") is False

    def test_rune_all_suffixes_redundant(self):
        for sfx in ("__mean", "__std", "__min", "__max", "__delta", "__slope"):
            name = f"bMID_primary_rune_2{sfx}"
            assert is_constant_redundant(name) is True, f"Expected {name} to be redundant"

    def test_rune_last_is_kept(self):
        assert is_constant_redundant("bMID_primary_rune_2__last") is False

    def test_ban_mean_is_redundant(self):
        assert is_constant_redundant("blue_ban_0__mean") is True

    def test_ban_max_is_redundant(self):
        assert is_constant_redundant("red_ban_4__max") is True

    def test_ban_last_is_kept(self):
        assert is_constant_redundant("blue_ban_0__last") is False

    def test_dynamic_feature_not_redundant(self):
        assert is_constant_redundant("bJNG_hp_pct__mean") is False
        assert is_constant_redundant("goldDiff__std") is False

    def test_stat_perk_all_suffixes_redundant(self):
        for sfx in ("__mean", "__std", "__min", "__max", "__delta", "__slope"):
            assert is_constant_redundant(f"bSUP_stat_perk_offense{sfx}") is True

    def test_summoner_spell_min_is_redundant(self):
        assert is_constant_redundant("rTOP_summoner_spell_2_id__min") is True

    def test_no_suffix_returns_false(self):
        assert is_constant_redundant("bJNG_champion_id") is False

    def test_empty_string(self):
        assert is_constant_redundant("") is False


# ─────────────────────────────────────────────────────────────
# is_quasi_constant_redundant
# ─────────────────────────────────────────────────────────────
class TestIsQuasiConstantRedundant:
    """Test quasi-constant feature identification.

    NOTE: Dragon soul features were reclassified from quasi-constant to
    within-fight-constant (sparse binary).  See TestIsWithinFightConstantRedundant.
    """

    def test_soul_is_NOT_quasi_constant(self):
        # Dragon soul is now within-fight-constant, not quasi-constant
        assert is_quasi_constant_redundant("bTOP_soul_infernal__delta") is False
        assert is_quasi_constant_redundant("rJNG_soul_ocean__std") is False
        assert is_quasi_constant_redundant("bMID_soul_mountain__mean") is False
        assert is_quasi_constant_redundant("bBOT_soul_cloud__min") is False
        assert is_quasi_constant_redundant("rSUP_soul_hextech__max") is False
        assert is_quasi_constant_redundant("rBOT_soul_chemtech__slope") is False

    def test_itemhash_delta_is_redundant(self):
        assert is_quasi_constant_redundant("bTOP_itemhash0__delta") is True

    def test_itemhash_std_is_redundant(self):
        assert is_quasi_constant_redundant("bTOP_itemhash15__std") is True

    def test_itemhash_mean_is_redundant(self):
        assert is_quasi_constant_redundant("bTOP_itemhash31__mean") is True

    def test_itemhash_last_is_kept(self):
        assert is_quasi_constant_redundant("bTOP_itemhash0__last") is False

    def test_zone_delta_is_redundant(self):
        assert is_quasi_constant_redundant("zone_top_lane__delta") is True
        assert is_quasi_constant_redundant("zone_mid_lane__std") is True
        assert is_quasi_constant_redundant("zone_bot_lane__mean") is True
        assert is_quasi_constant_redundant("zone_river__min") is True
        assert is_quasi_constant_redundant("zone_jungle__max") is True

    def test_zone_last_is_kept(self):
        assert is_quasi_constant_redundant("zone_top_lane__last") is False

    def test_pos_fight_delta_is_redundant(self):
        assert is_quasi_constant_redundant("pos_fight_x_norm__delta") is True
        assert is_quasi_constant_redundant("pos_fight_y_norm__slope") is True

    def test_pos_fight_last_is_kept(self):
        assert is_quasi_constant_redundant("pos_fight_x_norm__last") is False

    def test_dynamic_feature_not_quasi_constant(self):
        assert is_quasi_constant_redundant("bJNG_hp_pct__delta") is False
        assert is_quasi_constant_redundant("goldDiff__std") is False

    def test_strictly_constant_not_quasi_constant(self):
        # champion_id is strictly constant, not quasi-constant
        assert is_quasi_constant_redundant("bJNG_champion_id__delta") is False

    def test_no_suffix_returns_false(self):
        assert is_quasi_constant_redundant("bTOP_itemhash0") is False


# ─────────────────────────────────────────────────────────────
# is_within_fight_constant_redundant (dragon soul — sparse binary)
# ─────────────────────────────────────────────────────────────
class TestIsWithinFightConstantRedundant:
    """Test within-fight-constant (sparse binary) feature identification.

    Dragon soul is a sparse binary: 0 in ~70% of games, 1 in ~30%.
    Within a fight it never changes, so non-__last aggregations are
    mathematically trivial.
    """

    def test_soul_delta_is_redundant(self):
        assert is_within_fight_constant_redundant("bTOP_soul_infernal__delta") is True

    def test_soul_std_is_redundant(self):
        assert is_within_fight_constant_redundant("rJNG_soul_ocean__std") is True

    def test_soul_mean_is_redundant(self):
        assert is_within_fight_constant_redundant("bMID_soul_mountain__mean") is True

    def test_soul_min_max_are_redundant(self):
        assert is_within_fight_constant_redundant("bBOT_soul_cloud__min") is True
        assert is_within_fight_constant_redundant("rSUP_soul_hextech__max") is True

    def test_soul_last_is_kept(self):
        assert is_within_fight_constant_redundant("bTOP_soul_infernal__last") is False

    def test_soul_chemtech_is_within_fight_constant(self):
        assert is_within_fight_constant_redundant("rBOT_soul_chemtech__slope") is True

    def test_all_10_slots_covered(self):
        """All 10 slot prefixes for within-fight-constant node features."""
        slots = [
            "bTOP", "bJNG", "bMID", "bBOT", "bSUP",
            "rTOP", "rJNG", "rMID", "rBOT", "rSUP",
        ]
        for slot in slots:
            for soul in ("soul_infernal", "soul_ocean", "soul_mountain",
                         "soul_cloud", "soul_hextech", "soul_chemtech"):
                name = f"{slot}_{soul}__delta"
                assert is_within_fight_constant_redundant(name), f"Failed for {name}"

    def test_non_soul_not_within_fight_constant(self):
        assert is_within_fight_constant_redundant("bJNG_hp_pct__delta") is False
        assert is_within_fight_constant_redundant("bTOP_itemhash0__delta") is False
        assert is_within_fight_constant_redundant("zone_river__std") is False

    def test_no_suffix_returns_false(self):
        assert is_within_fight_constant_redundant("bTOP_soul_infernal") is False


# ─────────────────────────────────────────────────────────────
# classify_feature_constancy
# ─────────────────────────────────────────────────────────────
class TestClassifyFeatureConstancy:
    """Test comprehensive constancy classification."""

    def test_strictly_constant(self):
        assert classify_feature_constancy("bJNG_champion_id__delta") == "strictly_constant"
        assert classify_feature_constancy("blue_ban_0__mean") == "strictly_constant"
        assert classify_feature_constancy("rTOP_stat_perk_defense__slope") == "strictly_constant"

    def test_within_fight_constant(self):
        assert classify_feature_constancy("bTOP_soul_infernal__delta") == "within_fight_constant"
        assert classify_feature_constancy("rJNG_soul_ocean__std") == "within_fight_constant"

    def test_quasi_constant(self):
        assert classify_feature_constancy("bTOP_itemhash5__std") == "quasi_constant"
        assert classify_feature_constancy("zone_river__mean") == "quasi_constant"

    def test_time_varying(self):
        assert classify_feature_constancy("goldDiff__delta") == "time_varying"
        assert classify_feature_constancy("bJNG_hp_pct__std") == "time_varying"
        assert classify_feature_constancy("time_norm__slope") == "time_varying"

    def test_last_suffix_classification(self):
        # Even with __last suffix, classification should work
        assert classify_feature_constancy("bJNG_champion_id__last") == "strictly_constant"
        assert classify_feature_constancy("bTOP_soul_infernal__last") == "within_fight_constant"
        assert classify_feature_constancy("bTOP_itemhash0__last") == "quasi_constant"
        assert classify_feature_constancy("goldDiff__last") == "time_varying"

    def test_no_suffix(self):
        assert classify_feature_constancy("bJNG_champion_id") == "time_varying"

    def test_empty_string(self):
        assert classify_feature_constancy("") == "time_varying"


# ─────────────────────────────────────────────────────────────
# filter_static_temporal_noise (legacy)
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
# filter_constant_and_quasi_constant (comprehensive)
# ─────────────────────────────────────────────────────────────
class TestFilterConstantAndQuasiConstant:
    """Test the comprehensive redundant feature filter."""

    def test_drops_all_constant_suffixes(self):
        names = [
            "bJNG_champion_id__last",     # keep
            "bJNG_champion_id__mean",     # drop (constant)
            "bJNG_champion_id__std",      # drop (constant)
            "bJNG_champion_id__min",      # drop (constant)
            "bJNG_champion_id__max",      # drop (constant)
            "bJNG_champion_id__delta",    # drop (constant)
            "bJNG_champion_id__slope",    # drop (constant)
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(names)
        assert keep_idx == (0,)  # only __last kept
        assert len(dc) == 6
        assert len(dq) == 0

    def test_drops_quasi_constant_suffixes(self):
        names = [
            "bTOP_itemhash0__last",     # keep
            "bTOP_itemhash0__mean",     # drop (quasi)
            "bTOP_itemhash0__std",      # drop (quasi)
            "bTOP_itemhash0__delta",    # drop (quasi)
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(names)
        assert keep_idx == (0,)
        assert len(dc) == 0
        assert len(dq) == 3

    def test_keeps_time_varying(self):
        names = [
            "goldDiff__last",
            "goldDiff__mean",
            "goldDiff__std",
            "goldDiff__delta",
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(names)
        assert keep_idx == (0, 1, 2, 3)
        assert len(dc) == 0
        assert len(dq) == 0

    def test_mixed_features(self):
        names = [
            "bJNG_champion_id__last",      # keep (constant, last)
            "bJNG_champion_id__delta",     # drop (constant)
            "bTOP_itemhash5__last",              # keep (quasi, last)
            "bTOP_itemhash5__std",              # drop (quasi)
            "goldDiff__delta",             # keep (time-varying)
            "blue_ban_0__mean",            # drop (constant)
            "zone_river__slope",           # drop (quasi)
            "bJNG_hp_pct__std",            # keep (time-varying)
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(names)
        assert keep_idx == (0, 2, 4, 7)
        assert len(dc) == 2  # champion_id__delta, blue_ban_0__mean
        assert len(dq) == 2  # itemhash5__std, zone_river__slope

    def test_disable_constant_drop(self):
        names = [
            "bJNG_champion_id__delta",     # kept when constant drop disabled
            "bTOP_itemhash5__std",              # still dropped (quasi)
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(
            names, drop_strictly_constant=False, drop_quasi_constant=True,
        )
        assert keep_idx == (0,)
        assert len(dc) == 0
        assert len(dq) == 1

    def test_disable_quasi_drop(self):
        names = [
            "bJNG_champion_id__delta",     # dropped (constant)
            "bTOP_itemhash5__std",              # kept when quasi drop disabled
        ]
        keep_idx, dc, dq = filter_constant_and_quasi_constant(
            names, drop_strictly_constant=True, drop_quasi_constant=False,
        )
        assert keep_idx == (1,)
        assert len(dc) == 1
        assert len(dq) == 0

    def test_empty_input(self):
        keep_idx, dc, dq = filter_constant_and_quasi_constant([])
        assert keep_idx == ()
        assert dc == ()
        assert dq == ()

    def test_all_10_slots_constant(self):
        """All 10 slot prefixes should be recognized for constant features."""
        slots = [
            "bTOP", "bJNG", "bMID", "bBOT", "bSUP",
            "rTOP", "rJNG", "rMID", "rBOT", "rSUP",
        ]
        for slot in slots:
            assert is_constant_redundant(f"{slot}_champion_id__mean"), f"Failed for {slot}"
            assert is_constant_redundant(f"{slot}_stat_perk_flex__min"), f"Failed for {slot}"

    def test_all_10_slots_quasi_constant(self):
        """All 10 slot prefixes for quasi-constant extra features."""
        # NOTE: Dragon soul moved to within-fight-constant (tested separately).
        # Quasi-constant node features list is now empty; this test verifies
        # that non-slotted quasi-constants (itemhash, zone, pos) still work.
        for pfx in ("bTOP_itemhash0", "zone_top_lane", "pos_fight_x_norm"):
            name = f"{pfx}__delta"
            assert is_quasi_constant_redundant(name), f"Failed for {name}"


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

        # Check noise identification (legacy: only std/delta/slope)
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

    def test_comprehensive_filter_catches_more(self):
        """The comprehensive filter catches mean/min/max redundancy too."""
        base_names = [
            "bJNG_champion_id",
            "bJNG_primary_rune_1",
            "bJNG_hp_pct",
        ]
        tab_names = list(tabular_feature_names(base_names))

        # Comprehensive: champion_id has 6 redundant (all except last)
        # primary_rune_1 has 6 redundant
        # hp_pct has 0 (dynamic)
        redundant = [n for n in tab_names if is_constant_redundant(n)]
        assert len(redundant) == 12  # 2 static features × 6 suffixes

    def test_quasi_constant_in_tabular(self):
        """Quasi-constant features correctly identified in tabular names."""
        base_names = [
            "bTOP_itemhash0",
            "zone_river",
            "goldDiff",
        ]
        tab_names = list(tabular_feature_names(base_names))
        assert len(tab_names) == 21

        quasi = [n for n in tab_names if is_quasi_constant_redundant(n)]
        # itemhash0 has 6 quasi-redundant, zone_river has 6, goldDiff has 0
        assert len(quasi) == 12

    def test_full_filter_integration(self):
        """End-to-end: generate tabular names and apply comprehensive filter."""
        base_names = [
            "bTOP_champion_id",    # strictly constant
            "bTOP_soul_infernal",  # within-fight constant (sparse binary)
            "goldDiff",            # time-varying
        ]
        tab_names = list(tabular_feature_names(base_names))
        assert len(tab_names) == 21

        keep_idx, dc, dq = filter_constant_and_quasi_constant(tab_names)
        # champion_id: 1 kept (last), 6 dropped (constant)
        # soul_infernal: 1 kept (last), 6 dropped (within-fight-constant → reported in dq)
        # goldDiff: 7 kept (all)
        assert len(keep_idx) == 9  # 1 + 1 + 7
        assert len(dc) == 6
        assert len(dq) == 6  # within-fight-constant reported alongside quasi


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


# ─────────────────────────────────────────────────────────────
# Reduction estimate validation
# ─────────────────────────────────────────────────────────────
class TestReductionEstimate:
    """Validate that the filtering produces expected reductions
    on a realistic set of feature names."""

    def _build_full_tabular_names(self):
        """Build a representative set of tabular feature names
        similar to the 'full' feature set."""
        from core.config import (
            NODE_FEATURE_NAMES, SLOT_NAMES,
            GLOBAL_FEATURE_NAMES, EVENT_FEATURE_NAMES,
            ITEM_HASH_NAMES,
        )
        from gameplay.feature_spatial import SPATIAL_FEATURE_NAMES

        # node_flat base names (slotted), including per-player item hash (16-D
        # per player) — the real X_tab carries slotted per-player items, not the
        # legacy global 32-D ITEM_HASH_NAMES.
        node_base = [f"{slot}_{f}" for slot in SLOT_NAMES for f in NODE_FEATURE_NAMES]
        node_base += [f"{slot}_itemhash{i}" for slot in SLOT_NAMES for i in range(16)]
        # macro + spatial base names (non-slotted)
        macro_base = (
            list(GLOBAL_FEATURE_NAMES)
            + list(EVENT_FEATURE_NAMES)
            + list(SPATIAL_FEATURE_NAMES)
        )
        all_base = node_base + macro_base
        return list(tabular_feature_names(all_base)), len(all_base)

    def test_strictly_constant_reduction(self):
        """Verify that strictly constant filtering reduces feature count."""
        tab_names, n_base = self._build_full_tabular_names()
        keep_idx, dc, dq = filter_constant_and_quasi_constant(
            tab_names, drop_strictly_constant=True, drop_quasi_constant=False,
            drop_within_fight_constant=False,
        )
        n_total = len(tab_names)
        n_dropped = len(dc)
        # Should drop a substantial number of features
        assert n_dropped > 0, "Should drop strictly constant redundant features"
        # For 15 node static features × 10 slots × 6 suffixes = 900
        # + 10 ban features × 6 suffixes = 60
        # Total = 960
        assert n_dropped == 960, f"Expected 960 dropped, got {n_dropped}"
        reduction_pct = n_dropped / n_total * 100
        assert reduction_pct > 10, f"Expected >10% reduction, got {reduction_pct:.1f}%"

    def test_combined_reduction(self):
        """Verify total reduction with constant + quasi + within-fight-constant."""
        tab_names, n_base = self._build_full_tabular_names()
        keep_idx, dc, dq = filter_constant_and_quasi_constant(tab_names)
        n_total = len(tab_names)
        n_dropped_total = len(dc) + len(dq)
        reduction_pct = n_dropped_total / n_total * 100
        assert reduction_pct > 15, f"Expected >15% total reduction, got {reduction_pct:.1f}%"
        # dq combines quasi-constant + within-fight-constant drops:
        assert len(dq) > 0, "Should drop quasi + within-fight-constant features"
        # 6 soul × 10 slots × 6 suffixes = 360 (within-fight-constant)
        # + 16 per-player itemhash × 10 slots × 6 suffixes = 960 (quasi-constant)
        # + 5 zone × 6 suffixes = 30 (quasi-constant)
        # + 2 pos_fight × 6 suffixes = 12 (quasi-constant)
        # Total = 1362
        assert len(dq) == 1362, f"Expected 1362 quasi+wfc dropped, got {len(dq)}"
