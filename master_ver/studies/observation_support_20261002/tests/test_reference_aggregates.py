"""Independent aggregate reconstruction and strict comparison fixtures."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from studies.observation_support_20261002 import aggregate as production
from studies.observation_support_20261002 import reference_aggregates as reference


def _table() -> dict[str, np.ndarray]:
    n = 5
    out: dict[str, np.ndarray] = {
        "match_local": np.asarray([0, 0, 1, 1, 2], dtype=np.int64),
        "lag_f_F_tau": np.asarray([10_000, 12_000, 17_000, 25_000, -1], dtype=np.int64),
        "lag_f_F_tm1": np.asarray([10_000, 11_000, 22_000, 10_000, -1], dtype=np.int64),
        "cand_10_20_tm1": np.asarray([1, 1, 0, 1, 0], dtype=np.int8),
        "foreign_kills_in_tau_L": np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
        "other_anchors_in_tau_L": np.zeros(n, dtype=np.int64),
        "exact_10000_tau": np.asarray([1, 0, 0, 0, 0], dtype=np.int8),
        "strict_eligible": np.asarray([1, 1, 1, 0, 0], dtype=np.int8),
        "frame_differs": np.asarray([0, 1, 0, 0, 0], dtype=np.int8),
        "singleton": np.asarray([1, 0, 1, 0, 1], dtype=np.int8),
        "c0_bits": np.asarray([3, 3, 5, 5, 1], dtype=np.int64),
        "call_bits": np.asarray([7, 7, 13, 13, 1], dtype=np.int64),
    }
    for key, values in {
        "c0_n": [2, 2, 2, 2, 1],
        "c0_n_blue": [1, 1, 1, 1, 1],
        "c0_n_red": [1, 1, 1, 1, 0],
        "c0_n_unknown_team": [0, 0, 0, 0, 0],
        "c0_n_env_killer": [0, 0, 1, 1, 0],
        "c0_scale_max_side": [1, 1, 1, 1, 1],
        "call_n": [3, 3, 3, 3, 1],
        "call_n_blue": [2, 2, 2, 2, 1],
        "call_n_red": [1, 1, 1, 1, 0],
        "call_n_unknown_team": [0, 0, 0, 0, 0],
        "call_n_env_killer": [0, 0, 1, 1, 0],
        "call_scale_max_side": [2, 2, 2, 2, 1],
    }.items():
        out[key] = np.asarray(values, dtype=np.float64)
    for cut in ("tau", "tm1"):
        for radius, offset in ((1800, 0.0), (2000, 1.0)):
            tag = f"{cut}_r{radius}"
            out[f"support_ok_{tag}"] = np.asarray([1, 1, 0, 1, 1], dtype=np.int8)
            out[f"p_lower_n_{tag}"] = np.asarray([1, 3, -1, 2, 1], dtype=np.float64) + offset
            out[f"p_upper_n_{tag}"] = np.asarray([2, 4, -1, 3, 2], dtype=np.float64) + offset
            out[f"n_lower_{tag}"] = np.asarray([1, 2, -1, 1, 1], dtype=np.float64)
            out[f"n_upper_{tag}"] = np.asarray([2, 3, -1, 2, 2], dtype=np.float64)
            out[f"near_bits_{tag}"] = np.asarray([1, 3, -1, 5, 1], dtype=np.float64) + offset
            out[f"dist_min_{tag}"] = np.asarray([10, 20, np.nan, 30, 40], dtype=np.float64) + offset
            out[f"dist_p50_{tag}"] = out[f"dist_min_{tag}"] + 1
            out[f"dist_p90_{tag}"] = out[f"dist_min_{tag}"] + 2
    for horizon in (60, 90, 120):
        out[f"valid_h{horizon}"] = np.asarray([1, 1, 0, 1, 1], dtype=np.int8)
        out[f"e_h{horizon}"] = np.asarray([160_000, 160_000, -1, 160_000, 160_000], dtype=np.int64)
        for cut in ("tau", "tm1"):
            tag = f"{cut}_h{horizon}"
            out[f"same_as_Fpre_{tag}"] = np.asarray([1, 0, 0, 1, 0], dtype=np.int8)
            out[f"F_post_gt_L_{tag}"] = np.asarray([0, 1, 0, 0, 1], dtype=np.int8)
            out[f"stale_at_or_before_L_{tag}"] = np.asarray([1, 0, 0, 1, 0], dtype=np.int8)
            out[f"events_support_ok_{tag}"] = np.asarray([1, 1, 0, 1, 1], dtype=np.int8)
            out[f"F_post_{tag}"] = np.asarray([120_000, 130_000, -1, 140_000, 150_000], dtype=np.int64)
            out[f"F_post_idx_{tag}"] = np.asarray([2, 2, -1, 3, 3], dtype=np.int64)
            out[f"age_post_{tag}"] = np.asarray([5, 6, -1, 7, 8], dtype=np.int64)
            out[f"frame_idx_gap_{tag}"] = np.asarray([0, 1, -1, 1, 0], dtype=np.int64)
            out[f"timestamp_gap_{tag}"] = np.asarray([0, 60_000, -1, 60_000, 0], dtype=np.int64)
            out[f"async_events_Fpre_to_Fpost_{tag}"] = np.asarray([0, 2, -1, 1, 0], dtype=np.int64)
            out[f"async_events_cutoff_to_endpoint_{tag}"] = np.asarray([1, 2, -1, 0, 1], dtype=np.int64)
            for name, offset in (("d_gold_blue", 1), ("d_gold_red", 2), ("d_level_mean", 3), ("d_xp_mean", 4), ("d_hp_mean", 5)):
                out[f"{name}_{tag}"] = np.asarray([offset, offset + 1, np.nan, offset + 2, offset + 3], dtype=np.float64)
    return out


def _reports() -> tuple[dict, dict]:
    table = _table()
    prod = production.build_aggregates(table, 3, 0, n_matches_processed=3, n_zero_kill_processed=0)
    ref = reference.build_reference_aggregates(table, 3, 0, n_matches_processed=3, n_zero_kill_processed=0)
    return ref, prod


def test_reference_report_matches_production_tree_and_has_all_cells():
    ref, prod = _reports()
    comparison = reference.compare_aggregates(ref, prod)
    assert comparison["all_ok"] is True, comparison["mismatch_examples"]
    assert len(prod["participation_cells"]) == 4
    assert len(prod["post_frame"]) == 6
    for cell in prod["participation_cells"].values():
        assert {"full", "candidate", "isolated"} <= set(cell)
        assert {"c0", "c_all", "teams", "scale", "ambiguity"} <= set(cell["candidate"])
    for cell in prod["post_frame"].values():
        assert {"full", "candidate", "isolated"} <= set(cell)
        assert "async_events_cutoff_to_endpoint" in cell["candidate"]


def test_strict_compare_catches_statistics_support_none_post_population_isolation_and_schema():
    ref, prod = _reports()
    mutations = (
        ("median", lambda x: x["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_median"] + 1),
        ("mean", lambda x: x["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_mean"] + 1),
        ("support", lambda x: x["participation_cells"]["tau_r1800"]["candidate"]["n_support_ok"] + 1),
        ("none", lambda x: None),
        ("post", lambda x: x["post_frame"]["tau_h60"]["candidate"]["n_valid"] + 1),
        ("population", lambda x: x["post_frame"]["tau_h60"]["n_population_cand"] + 1),
        ("isolation", lambda x: x["participation_cells"]["tau_r1800"]["n_population_isolated"] + 1),
    )
    for name, mutate in mutations:
        bad = deepcopy(prod)
        if name == "median":
            bad["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_median"] = mutate(bad)
        elif name == "mean":
            bad["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_mean"] = mutate(bad)
        elif name == "support":
            bad["participation_cells"]["tau_r1800"]["candidate"]["n_support_ok"] = mutate(bad)
        elif name == "none":
            bad["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["equal_total_match_mean"] = None
        elif name == "post":
            bad["post_frame"]["tau_h60"]["candidate"]["n_valid"] = mutate(bad)
        elif name == "population":
            bad["post_frame"]["tau_h60"]["n_population_cand"] = mutate(bad)
        else:
            bad["participation_cells"]["tau_r1800"]["n_population_isolated"] = mutate(bad)
        result = reference.compare_aggregates(ref, bad)
        assert result["all_ok"] is False, name
        assert result["n_mismatches"] > 0, name
    schema_bad = deepcopy(prod)
    del schema_bad["gap_tau"]["cum_10_20"]
    schema_bad["extra_schema_field"] = 1
    result = reference.compare_aggregates(ref, schema_bad)
    assert result["schema_ok"] is False
    assert result["n_missing_fields"] >= 1
    assert result["n_extra_fields"] == 1


def test_reference_weighted_quantiles_use_inverse_cdf_and_required_columns_are_shared():
    values = np.asarray([1.0, 100.0, 2.0])
    weights = np.asarray([0.5, 0.5, 1.0])
    assert reference.inverse_weighted_cdf_quantile(values, weights) == 2.0
    assert production.inverse_weighted_cdf_quantile(values, np.ones(3)) == 2.0
    assert production.AGGREGATE_REQUIRED_FIELDS == reference.REFERENCE_REQUIRED_FIELDS
    assert "call_scale_max_side" in reference.REFERENCE_REQUIRED_FIELDS
    assert "async_events_cutoff_to_endpoint_tau_h60" in reference.REFERENCE_REQUIRED_FIELDS


def test_integer_counts_are_exact_and_reference_input_fails_closed():
    ref, prod = _reports()
    bad = deepcopy(prod)
    bad["n_components"] = int(bad["n_components"]) + 1
    result = reference.compare_aggregates(ref, bad)
    assert result["all_ok"] is False
    assert result["n_numeric_mismatches"] >= 1
    near = deepcopy(prod)
    value = near["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_mean"]
    near["participation_cells"]["tau_r1800"]["candidate"]["p_lower"]["component_equal_mean"] = value + 5e-8
    assert reference.compare_aggregates(ref, near)["all_ok"] is True
    incomplete = _table()
    del incomplete["d_gold_blue_tau_h60"]
    with pytest.raises(ValueError, match="missing required typed fields"):
        reference.build_reference_aggregates(incomplete, 3, 0)
