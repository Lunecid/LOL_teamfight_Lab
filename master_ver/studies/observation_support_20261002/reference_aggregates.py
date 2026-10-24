"""Independent aggregate reconstruction from typed reference arrays.

This module intentionally repeats the small statistical kernels instead of
importing the production aggregate, contract, extraction, or replay modules.
The caller supplies arrays filled by the independent source replay; this file
only interprets those arrays and compares the resulting report tree.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from . import config as C


PARTICIPATION_BASE_FIELDS = (
    "support_ok",
    "p_lower_n",
    "p_upper_n",
    "n_lower",
    "n_upper",
    "dist_min",
    "dist_p50",
    "dist_p90",
    "near_bits",
)
SOURCE_FIELDS = (
    "c0_n",
    "c0_n_blue",
    "c0_n_red",
    "c0_n_unknown_team",
    "c0_n_env_killer",
    "c0_scale_max_side",
    "call_n",
    "call_n_blue",
    "call_n_red",
    "call_n_unknown_team",
    "call_n_env_killer",
    "call_scale_max_side",
)
POST_BASE_FIELDS = (
    "F_post",
    "F_post_idx",
    "age_post",
    "frame_idx_gap",
    "timestamp_gap",
    "async_events_Fpre_to_Fpost",
    "async_events_cutoff_to_endpoint",
    "same_as_Fpre",
    "F_post_gt_L",
    "stale_at_or_before_L",
    "events_support_ok",
    "d_gold_blue",
    "d_gold_red",
    "d_level_mean",
    "d_xp_mean",
    "d_hp_mean",
)


def _required_fields() -> tuple[str, ...]:
    fields = {
        "match_local",
        "lag_f_F_tau",
        "lag_f_F_tm1",
        "foreign_kills_in_tau_L",
        "other_anchors_in_tau_L",
        "exact_10000_tau",
        "strict_eligible",
        "frame_differs",
        "singleton",
        *SOURCE_FIELDS,
    }
    for cut in ("tau", "tm1"):
        for radius in (1800, 2000):
            fields.update(f"{name}_{cut}_r{radius}" for name in PARTICIPATION_BASE_FIELDS)
    for horizon in C.HORIZONS_S:
        fields.update(f"{name}_h{horizon}" for name in ("valid",))
        for cut in ("tau", "tm1"):
            fields.update(f"{name}_{cut}_h{horizon}" for name in POST_BASE_FIELDS)
    return tuple(sorted(fields))


# Minimal typed columns needed by build_reference_aggregates.  The source
# replay may retain richer evidence arrays, but none are needed for this tree.
REFERENCE_REQUIRED_FIELDS = _required_fields()
AGGREGATE_REQUIRED_FIELDS = REFERENCE_REQUIRED_FIELDS
REQUIRED_FIELDS = REFERENCE_REQUIRED_FIELDS


def minimal_required_fields() -> tuple[str, ...]:
    """Return the typed columns needed for the independent aggregate tree."""
    return REFERENCE_REQUIRED_FIELDS


def _column(
    table: Mapping[str, np.ndarray],
    name: str,
    n: int,
    dtype: np.dtype | type,
    default: Any,
) -> np.ndarray:
    if name not in table:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(table[name])
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"reference column {name!r} has shape {arr.shape}, expected {(n,)}")
    return arr.astype(dtype, copy=False)


def inverse_weighted_cdf_quantile(
    values: np.ndarray, weights: np.ndarray, p: float = 0.5
) -> float | None:
    """Protocol inverse-CDF quantile, including component-equal summaries."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return None
    values, weights = values[ok], weights[ok]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights, dtype=np.float64)
    total = float(cumulative[-1])
    return _quantile_from_sorted_cdf(values, cumulative, total, p)


def _quantile_from_sorted_cdf(
    values: np.ndarray, cumulative: np.ndarray, total: float, p: float
) -> float | None:
    """Inverse-CDF lookup for an already sorted weighted sample."""
    if total <= 0.0 or not np.isfinite(total):
        return None
    idx = int(np.searchsorted(cumulative, float(p) * total, side="left"))
    return float(values[min(max(idx, 0), values.size - 1)])


def _stats(values: np.ndarray, pop: np.ndarray, ml: np.ndarray, support: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    pop = np.asarray(pop, dtype=bool)
    if values.shape != pop.shape:
        raise ValueError("reference values and population mask have different shapes")
    accepted = pop if support is None else pop & np.asarray(support, dtype=bool)
    known = accepted & np.isfinite(values)
    n_population = int(pop.sum())
    n_known = int(known.sum())
    out: dict[str, Any] = {
        "n_population": n_population,
        "n_known": n_known,
        "n_unknown": n_population - n_known if n_population else 0,
        "n_unknown_excluded": n_population - n_known if n_population else 0,
        "n_matches_known": 0,
        "support_rate": float(n_known / n_population) if n_population else None,
    }
    vals = values[known]
    if vals.size:
        ones = np.ones(vals.size, dtype=np.float64)
        component_order = np.argsort(vals, kind="mergesort")
        component_sorted = vals[component_order]
        component_cdf = np.cumsum(ones[component_order], dtype=np.float64)
        component_total = float(component_cdf[-1])
        comp_med = _quantile_from_sorted_cdf(
            component_sorted, component_cdf, component_total, 0.5
        )
        comp_p90 = _quantile_from_sorted_cdf(
            component_sorted, component_cdf, component_total, 0.9
        )
        used = np.asarray(ml, dtype=np.int64)[known]
        _, inverse, counts = np.unique(used, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].astype(np.float64)
        match_mean = float(np.sum(weights * vals) / np.sum(weights))
        match_order = np.argsort(vals, kind="mergesort")
        match_sorted = vals[match_order]
        match_cdf = np.cumsum(weights[match_order], dtype=np.float64)
        match_total = float(match_cdf[-1])
        match_med = _quantile_from_sorted_cdf(
            match_sorted, match_cdf, match_total, 0.5
        )
        match_p90 = _quantile_from_sorted_cdf(
            match_sorted, match_cdf, match_total, 0.9
        )
        out.update(
            {
                "component_equal_mean": float(np.mean(vals)),
                "component_equal_median": comp_med,
                "component_equal_p50": comp_med,
                "component_equal_p90": comp_p90,
                "equal_total_match_mean": match_mean,
                "equal_total_match_median": match_med,
                "equal_total_match_p50": match_med,
                "equal_total_match_p90": match_p90,
                "n_matches_known": int(counts.size),
            }
        )
    else:
        out.update(
            {
                "component_equal_mean": None,
                "component_equal_median": None,
                "component_equal_p50": None,
                "component_equal_p90": None,
                "equal_total_match_mean": None,
                "equal_total_match_median": None,
                "equal_total_match_p50": None,
                "equal_total_match_p90": None,
            }
        )
    return out


def _support_info(pop: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    pop = np.asarray(pop, dtype=bool)
    ok = pop & np.asarray(support, dtype=bool)
    n_pop, n_ok = int(pop.sum()), int(ok.sum())
    return {
        "n_population": n_pop,
        "n_supported": n_ok,
        "n_known": n_ok,
        "n_unknown": n_pop - n_ok if n_pop else 0,
        "support_rate": float(n_ok / n_pop) if n_pop else None,
    }


def _bit_count(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.uint16)
    lookup = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.int8)
    return lookup[vals & 255].astype(np.int64) + lookup[(vals >> 8) & 255].astype(np.int64)


def _source_columns(table: Mapping[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    c0_bits = _column(table, "c0_bits", n, np.uint16, 0)
    call_bits = _column(table, "call_bits", n, np.uint16, 0)
    out = {
        "c0_n": _column(table, "c0_n", n, np.float64, np.nan),
        "call_n": _column(table, "call_n", n, np.float64, np.nan),
        "c0_n_blue": _column(table, "c0_n_blue", n, np.float64, np.nan),
        "c0_n_red": _column(table, "c0_n_red", n, np.float64, np.nan),
        "c0_n_unknown_team": _column(table, "c0_n_unknown_team", n, np.float64, np.nan),
        "c0_n_env_killer": _column(table, "c0_n_env_killer", n, np.float64, np.nan),
        "c0_scale_max_side": _column(table, "c0_scale_max_side", n, np.float64, np.nan),
        "call_n_blue": _column(table, "call_n_blue", n, np.float64, np.nan),
        "call_n_red": _column(table, "call_n_red", n, np.float64, np.nan),
        "call_n_unknown_team": _column(table, "call_n_unknown_team", n, np.float64, np.nan),
        "call_n_env_killer": _column(table, "call_n_env_killer", n, np.float64, np.nan),
        "call_scale_max_side": _column(table, "call_scale_max_side", n, np.float64, np.nan),
    }
    out["c0_n"] = np.where(np.isfinite(out["c0_n"]), out["c0_n"], _bit_count(c0_bits))
    out["call_n"] = np.where(np.isfinite(out["call_n"]), out["call_n"], _bit_count(call_bits))
    return out


def _source_metrics(source: Mapping[str, np.ndarray], pop: np.ndarray, ml: np.ndarray) -> dict[str, Any]:
    def count_stats(values: np.ndarray) -> dict[str, Any]:
        return _stats(values, pop, ml, np.isfinite(values) & (values >= 0))

    def group(prefix: str) -> dict[str, Any]:
        return {
            "n": count_stats(source[f"{prefix}_n"]),
            "n_blue": count_stats(source[f"{prefix}_n_blue"]),
            "n_red": count_stats(source[f"{prefix}_n_red"]),
            "n_unknown_team": count_stats(source[f"{prefix}_n_unknown_team"]),
            "n_env_killer": count_stats(source[f"{prefix}_n_env_killer"]),
            "scale_max_side": count_stats(source[f"{prefix}_scale_max_side"]),
        }
    c0, call = group("c0"), group("call")
    scale_diff = source["call_scale_max_side"] - source["c0_scale_max_side"]
    scale_support = (
        np.isfinite(source["call_scale_max_side"])
        & (source["call_scale_max_side"] >= 0)
        & np.isfinite(source["c0_scale_max_side"])
        & (source["c0_scale_max_side"] >= 0)
    )
    teams = {
        "c0": {"blue": c0["n_blue"], "red": c0["n_red"], "unknown": c0["n_unknown_team"]},
        "c_all": {"blue": call["n_blue"], "red": call["n_red"], "unknown": call["n_unknown_team"]},
    }
    scale = {
        "c0": c0["scale_max_side"],
        "c_all": call["scale_max_side"],
        "c_all_minus_c0": _stats(scale_diff, pop, ml, scale_support),
    }
    return {
        "c0": c0,
        "c_all": call,
        "teams": teams,
        "scale": scale,
        "c0_stats": c0,
        "c_all_stats": call,
        "team_stats": teams,
        "scale_stats": scale,
    }


def _part_metrics(table: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray], tag: str, pop: np.ndarray, ml: np.ndarray, singleton: np.ndarray) -> dict[str, Any]:
    n = pop.shape[0]
    support = _column(table, f"support_ok_{tag}", n, bool, False)
    pl = _column(table, f"p_lower_n_{tag}", n, np.float64, np.nan)
    pu = _column(table, f"p_upper_n_{tag}", n, np.float64, np.nan)
    nl = _column(table, f"n_lower_{tag}", n, np.float64, np.nan)
    nu = _column(table, f"n_upper_{tag}", n, np.float64, np.nan)
    known_pl = support & np.isfinite(pl) & (pl >= 0)
    known_pu = support & np.isfinite(pu) & (pu >= 0)
    known_nl = support & np.isfinite(nl) & (nl >= 0)
    known_nu = support & np.isfinite(nu) & (nu >= 0)
    both = known_pl & known_pu
    n_both = known_nl & known_nu
    distance: dict[str, Any] = {}
    for name in ("dist_min", "dist_p50", "dist_p90"):
        values = _column(table, f"{name}_{tag}", n, np.float64, np.nan)
        distance[name] = _stats(values, pop, ml, support & np.isfinite(values) & (values >= 0))
    out: dict[str, Any] = {
        "n_population": int(pop.sum()),
        "n_support_ok": int((pop & support).sum()),
        "n_support_unknown": int(pop.sum() - (pop & support).sum()),
        "support": _support_info(pop, support),
        "p_lower": _stats(pl, pop, ml, known_pl),
        "p_upper": _stats(pu, pop, ml, known_pu),
        "n_lower": _stats(nl, pop, ml, known_nl),
        "n_upper": _stats(nu, pop, ml, known_nu),
        "distance": distance,
        "ambiguity": {
            "p_width": _stats(pu - pl, pop, ml, both),
            "n_width": _stats(nu - nl, pop, ml, n_both),
            "p_ambiguous": _stats((pu > pl).astype(np.float64), pop, ml, both),
            "n_ambiguous": _stats((nu > nl).astype(np.float64), pop, ml, n_both),
        },
        "source": _source_metrics(source, pop, ml),
        "singleton": _stats(np.asarray(singleton, dtype=np.float64), pop, ml, support),
    }
    out["singleton_rate_known"] = out["singleton"]["component_equal_mean"]
    out["c0"], out["c_all"] = out["source"]["c0"], out["source"]["c_all"]
    out["teams"], out["scale"] = out["source"]["teams"], out["source"]["scale"]
    return out


def _radius_difference(table: Mapping[str, np.ndarray], cut: str, pop: np.ndarray, ml: np.ndarray) -> dict[str, Any]:
    n = pop.shape[0]
    support = _column(table, f"support_ok_{cut}_r1800", n, bool, False) & _column(table, f"support_ok_{cut}_r2000", n, bool, False)
    metrics: dict[str, Any] = {}
    for field in ("p_lower_n", "p_upper_n", "n_lower", "n_upper"):
        small = _column(table, f"{field}_{cut}_r1800", n, np.float64, np.nan)
        large = _column(table, f"{field}_{cut}_r2000", n, np.float64, np.nan)
        metrics[field] = _stats(large - small, pop, ml, support & np.isfinite(small) & np.isfinite(large) & (small >= 0) & (large >= 0))
    small = _column(table, f"near_bits_{cut}_r1800", n, np.float64, np.nan)
    large = _column(table, f"near_bits_{cut}_r2000", n, np.float64, np.nan)
    metrics["near_changed"] = _stats((large != small).astype(np.float64), pop, ml, support & np.isfinite(small) & np.isfinite(large) & (small >= 0) & (large >= 0))
    return {"from_radius": 1800, "to_radius": 2000, "support": _support_info(pop, support), "metrics": metrics}


def _post_metrics(table: Mapping[str, np.ndarray], pop: np.ndarray, ml: np.ndarray, tag: str, horizon: int) -> dict[str, Any]:
    n = pop.shape[0]
    valid = _column(table, f"valid_h{horizon}", n, bool, False)
    valid_pop = pop & valid
    events_ok = _column(table, f"events_support_ok_{tag}", n, bool, False)
    event_pop = valid_pop & events_ok

    def numeric(field: str, support: np.ndarray, nonnegative: bool = False) -> dict[str, Any]:
        values = _column(table, f"{field}_{tag}", n, np.float64, np.nan)
        accepted = support & np.isfinite(values)
        if nonnegative:
            accepted &= values >= 0
        return _stats(values, pop, ml, accepted)

    def status(field: str) -> dict[str, Any]:
        values = _column(table, f"{field}_{tag}", n, bool, False)
        known = pop & valid_pop
        return {
            "n_population": int(pop.sum()),
            "n_known": int(known.sum()),
            "n_unknown": int(pop.sum() - known.sum()),
            "n_true": int((known & values).sum()),
            "rate": float((known & values).sum() / known.sum()) if known.any() else None,
            "stats": _stats(values.astype(np.float64), pop, ml, valid_pop),
        }

    channels = {
        name: numeric(name, valid_pop)
        for name in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean")
    }
    events = {
        "Fpre_to_Fpost": numeric("async_events_Fpre_to_Fpost", event_pop, True),
        "cutoff_to_endpoint": numeric("async_events_cutoff_to_endpoint", event_pop, True),
    }
    same = _column(table, f"same_as_Fpre_{tag}", n, bool, False)
    fgt = _column(table, f"F_post_gt_L_{tag}", n, bool, False)
    stale = _column(table, f"stale_at_or_before_L_{tag}", n, bool, False)
    cutoff_events = _column(table, f"async_events_cutoff_to_endpoint_{tag}", n, np.float64, np.nan)
    frame_events = _column(table, f"async_events_Fpre_to_Fpost_{tag}", n, np.float64, np.nan)
    same_with_events = pop & valid & same & events_ok & np.isfinite(cutoff_events) & (cutoff_events > 0)
    out: dict[str, Any] = {
        "n_population": int(pop.sum()),
        "n_valid": int(valid_pop.sum()),
        "n_invalid": int(pop.sum() - valid_pop.sum()),
        "valid_rate": float(valid_pop.sum() / pop.sum()) if pop.any() else None,
        "n_event_support_ok": int(event_pop.sum()),
        "n_event_unknown": int(valid_pop.sum() - event_pop.sum()),
        "event_support_rate": float(event_pop.sum() / valid_pop.sum()) if valid_pop.any() else None,
        "same_as_Fpre": status("same_as_Fpre"),
        "F_post_gt_L": status("F_post_gt_L"),
        "stale_at_or_before_L": status("stale_at_or_before_L"),
        "F_post": numeric("F_post", valid_pop, True),
        "F_post_idx": numeric("F_post_idx", valid_pop, True),
        "age_post": numeric("age_post", valid_pop, True),
        "frame_idx_gap": numeric("frame_idx_gap", valid_pop, True),
        "timestamp_gap": numeric("timestamp_gap", valid_pop, True),
        "async_events_Fpre_to_Fpost": events["Fpre_to_Fpost"],
        "async_events_cutoff_to_endpoint": events["cutoff_to_endpoint"],
        "events": events,
        "channels": channels,
        "same_frame_with_async_events": int(same_with_events.sum()),
        "n_async_events_Fpre_to_Fpost_positive": int((pop & event_pop & np.isfinite(frame_events) & (frame_events > 0)).sum()),
        "n_async_events_cutoff_to_endpoint_positive": int((pop & event_pop & np.isfinite(cutoff_events) & (cutoff_events > 0)).sum()),
    }
    out.update(
        {
            "n_cand_valid": out["n_valid"],
            "n_same_as_Fpre": out["same_as_Fpre"]["n_true"],
            "n_F_post_gt_L": out["F_post_gt_L"]["n_true"],
            "n_stale_at_or_before_L": out["stale_at_or_before_L"]["n_true"],
            "d_gold_blue": channels["d_gold_blue"],
            "d_gold_red": channels["d_gold_red"],
            "d_level_mean": channels["d_level_mean"],
            "d_xp_mean": channels["d_xp_mean"],
            "d_hp_mean": channels["d_hp_mean"],
        }
    )
    return out


def build_reference_aggregates(
    reference_table: Mapping[str, np.ndarray],
    n_matches_source: int,
    n_zero_kill: int,
    *,
    n_matches_processed: int | None = None,
    n_zero_kill_processed: int | None = None,
) -> dict[str, Any]:
    """Build the same report tree from independently filled typed arrays."""
    missing = tuple(field for field in REFERENCE_REQUIRED_FIELDS if field not in reference_table)
    if missing:
        raise ValueError(
            "reference aggregate input is missing required typed fields: "
            + ", ".join(missing)
        )
    n = int(np.asarray(reference_table["match_local"]).shape[0])
    tau = _column(reference_table, "lag_f_F_tau", n, np.int64, -1)
    tm1 = _column(reference_table, "lag_f_F_tm1", n, np.int64, -1)
    ml = _column(reference_table, "match_local", n, np.int64, -1)
    foreign = _column(reference_table, "foreign_kills_in_tau_L", n, np.int64, 0)
    other = _column(reference_table, "other_anchors_in_tau_L", n, np.int64, 0)
    singleton = _column(reference_table, "singleton", n, bool, False)
    full = np.ones(n, dtype=bool)
    cand_tau = (tau >= 10_000) & (tau <= 20_000)
    # Reconstruct tm1 candidates from the independent tm1 lags rather than
    # trusting a copied candidate flag.
    cand_tm1 = (tm1 >= 10_000) & (tm1 <= 20_000)
    isol_tau = cand_tau & (foreign == 0) & (other == 0)
    isol_tm1 = cand_tm1 & (foreign == 0) & (other == 0)
    source = _source_columns(reference_table, n)
    gaps_tau = {
        "exact_10000": int((tau >= 0).astype(bool).astype(np.int8)[tau == 10_000].sum()),
        "gt10_le15": int(((tau > 10_000) & (tau <= 15_000)).sum()),
        "gt15_le20": int(((tau > 15_000) & (tau <= 20_000)).sum()),
        "cum_10_15": int(((tau >= 10_000) & (tau <= 15_000)).sum()),
        "cum_10_20": int(cand_tau.sum()),
        "n_known_lag": int((tau >= 0).sum()),
        "n_unknown_lag": int((tau < 0).sum()),
        "n_negative_lag": int((tau < 0).sum()),
    }
    gaps_tm1 = {
        "exact_10000": int(((tm1 >= 0) & (tm1 == 10_000)).sum()),
        "gt10_le15": int(((tm1 > 10_000) & (tm1 <= 15_000)).sum()),
        "gt15_le20": int(((tm1 > 15_000) & (tm1 <= 20_000)).sum()),
        "cum_10_15": int(((tm1 >= 10_000) & (tm1 <= 15_000)).sum()),
        "cum_10_20": int(((tm1 >= 10_000) & (tm1 <= 20_000)).sum()),
        "n_known_lag": int((tm1 >= 0).sum()),
        "n_unknown_lag": int((tm1 < 0).sum()),
        "n_negative_lag": int((tm1 < 0).sum()),
    }
    out: dict[str, Any] = {
        "aggregate_schema": "observation_support_aggregates_v3",
        "n_components": n,
        "n_matches_source": int(n_matches_source),
        "n_zero_kill_matches_source": int(n_zero_kill),
        "n_matches_processed": int(n_matches_processed if n_matches_processed is not None else n_matches_source),
        "n_zero_kill_processed": int(n_zero_kill_processed if n_zero_kill_processed is not None else n_zero_kill),
        "n_matches_with_components": int(np.unique(ml).size) if n else 0,
        "gap_tau": gaps_tau,
        "gap_tm1": gaps_tm1,
        "gap_groups": {"tau": gaps_tau, "tm1": gaps_tm1},
        "n_cand_10_20_tau": int(cand_tau.sum()),
        "n_isolated_cand_10_20_tau": int(isol_tau.sum()),
        "n_cand_10_20_tm1": int(cand_tm1.sum()),
        "n_isolated_cand_10_20_tm1": int(isol_tm1.sum()),
        "n_exact_10000_tau": int(((tau >= 0) & (tau == 10_000)).sum()),
        "n_strict_eligible": int(_column(reference_table, "strict_eligible", n, bool, False).sum()),
        "n_frame_differs": int(_column(reference_table, "frame_differs", n, bool, False).sum()),
        "reconcile_expected": {
            "inclusive_10_20": C.EXPECTED_INCLUSIVE_10_20,
            "strict_eligible": C.EXPECTED_STRICT_ELIGIBLE,
            "exact_10000": C.EXPECTED_EXACT_10000,
            "n_components": C.EXPECTED_MIN1_COMPONENTS,
        },
    }
    out["reconcile_ok"] = bool(
        out["n_cand_10_20_tau"] == C.EXPECTED_INCLUSIVE_10_20
        and out["n_strict_eligible"] == C.EXPECTED_STRICT_ELIGIBLE
        and out["n_exact_10000_tau"] == C.EXPECTED_EXACT_10000
        and out["n_components"] == C.EXPECTED_MIN1_COMPONENTS
    )
    masks = {"tau": (cand_tau, isol_tau), "tm1": (cand_tm1, isol_tm1)}
    cells: dict[str, Any] = {}
    radius_differences: dict[str, Any] = {}
    for cut, (candidate, isolated) in masks.items():
        populations = {"full": full, "candidate": candidate, "isolated": isolated}
        radius_differences[cut] = {key: _radius_difference(reference_table, cut, mask, ml) for key, mask in populations.items()}
        for radius in (1800, 2000):
            tag = f"{cut}_r{radius}"
            cell = {
                "n_population_full": int(full.sum()),
                "n_population_cand": int(candidate.sum()),
                "n_population_isolated": int(isolated.sum()),
                "full": _part_metrics(reference_table, source, tag, full, ml, singleton),
                "candidate": _part_metrics(reference_table, source, tag, candidate, ml, singleton),
                "isolated": _part_metrics(reference_table, source, tag, isolated, ml, singleton),
                "radius_difference": radius_differences[cut],
            }
            candidate_cell = cell["candidate"]
            cell.update({
                "n_support_ok": candidate_cell["n_support_ok"],
                "p_lower": candidate_cell["p_lower"],
                "p_upper": candidate_cell["p_upper"],
                "n_lower": candidate_cell["n_lower"],
                "n_upper": candidate_cell["n_upper"],
            })
            cells[tag] = cell
    out["participation_cells"] = cells
    out["participation_radius_differences"] = radius_differences
    out["population_cells"] = {
        cut: {
            key: _source_metrics(source, mask, ml)
            for key, mask in {"full": full, "candidate": candidate, "isolated": isolated}.items()
        }
        for cut, (candidate, isolated) in masks.items()
    }
    post: dict[str, Any] = {}
    for horizon in C.HORIZONS_S:
        for cut, (candidate, isolated) in masks.items():
            tag = f"{cut}_h{horizon}"
            populations = {"full": full, "candidate": candidate, "isolated": isolated}
            post_cells = {key: _post_metrics(reference_table, mask, ml, tag, horizon) for key, mask in populations.items()}
            candidate_cell = post_cells["candidate"]
            post[tag] = {
                "n_population_full": int(full.sum()),
                "n_population_cand": int(candidate.sum()),
                "n_population_isolated": int(isolated.sum()),
                "full": post_cells["full"],
                "candidate": candidate_cell,
                "isolated": post_cells["isolated"],
                "n_cand_valid": candidate_cell["n_valid"],
                "n_same_as_Fpre": candidate_cell["n_same_as_Fpre"],
                "n_F_post_gt_L": candidate_cell["n_F_post_gt_L"],
                "n_stale_at_or_before_L": candidate_cell["n_stale_at_or_before_L"],
                "async_events_Fpre_to_Fpost": candidate_cell["async_events_Fpre_to_Fpost"],
                "async_events_cutoff_to_endpoint": candidate_cell["async_events_cutoff_to_endpoint"],
                "d_gold_blue": candidate_cell["d_gold_blue"],
                "d_gold_red": candidate_cell["d_gold_red"],
                "d_level_mean": candidate_cell["d_level_mean"],
                "d_xp_mean": candidate_cell["d_xp_mean"],
                "d_hp_mean": candidate_cell["d_hp_mean"],
                "same_frame_with_async_events": candidate_cell["same_frame_with_async_events"],
                "n_async_events_Fpre_to_Fpost_positive": candidate_cell["n_async_events_Fpre_to_Fpost_positive"],
                "n_async_events_cutoff_to_endpoint_positive": candidate_cell["n_async_events_cutoff_to_endpoint_positive"],
            }
    out["post_frame"] = post
    out["channel_units"] = {
        "gold_team_minute": "cache gold_team_minute float32 units (not claimed raw gold)",
        "level_norm": "node_minute[:, :, level_norm] normalized /18-style cache units",
        "xp_norm": "node_minute[:, :, xp_norm] normalized cache units",
        "hp_pct": "node_minute[:, :, hp_pct] fraction cache units; 0 is ambiguous_zero",
    }
    out["raw_presence_verified"] = False
    out["definitive_n0"] = False
    out["note"] = (
        "P_lower/P_upper are CACHE-ONLY proxies; not actual participants or definitive N0/P0 sample counts. "
        "Geometric near_bits are not participation near-only; use part_*_lower/upper decompositions."
    )
    return out


def recompute_aggregates_reference(reference_table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compatibility wrapper for older callers that did not pass totals."""
    n = int(np.asarray(reference_table["match_local"]).shape[0])
    n_matches = int(np.unique(np.asarray(reference_table["match_local"], dtype=np.int64)).size)
    return build_reference_aggregates(reference_table, n_matches, 0)


def compare_aggregates(
    ref: Any,
    prod: Any,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
    max_examples: int = 50,
) -> dict[str, Any]:
    """Strict recursive comparison of two report trees.

    Missing/extra keys, shape/count differences, ``None`` asymmetry, and every
    leaf mismatch are counted.  Examples are bounded only for output size; no
    mismatch is skipped from the totals.
    """
    if atol < 0 or rtol < 0 or max_examples < 0:
        raise ValueError("comparison tolerances and max_examples must be nonnegative")
    state: dict[str, Any] = {
        "n_compared": 0,
        "n_mismatches": 0,
        "mismatch_counts": {},
        "mismatch_examples": [],
        "n_missing_fields": 0,
        "n_extra_fields": 0,
        "n_shape_mismatches": 0,
        "n_count_mismatches": 0,
        "n_type_mismatches": 0,
        "n_none_asymmetry": 0,
        "n_numeric_mismatches": 0,
    }

    def record(path: str, kind: str, expected: Any, actual: Any) -> None:
        state["n_mismatches"] += 1
        state["mismatch_counts"][kind] = state["mismatch_counts"].get(kind, 0) + 1
        if kind == "missing_key":
            state["n_missing_fields"] += 1
        elif kind == "extra_key":
            state["n_extra_fields"] += 1
        elif kind == "shape_mismatch":
            state["n_shape_mismatches"] += 1
        elif kind == "count_mismatch":
            state["n_count_mismatches"] += 1
        elif kind == "none_asymmetry":
            state["n_none_asymmetry"] += 1
        elif kind == "type_mismatch":
            state["n_type_mismatches"] += 1
        elif kind == "numeric_mismatch":
            state["n_numeric_mismatches"] += 1
        if len(state["mismatch_examples"]) < max_examples:
            state["mismatch_examples"].append(
                {"path": path, "kind": kind, "expected": _preview(expected), "actual": _preview(actual)}
            )

    def walk(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, Mapping) or isinstance(actual, Mapping):
            if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
                record(path, "type_mismatch", expected, actual)
                return
            expected_keys, actual_keys = set(expected), set(actual)
            for key in sorted(expected_keys - actual_keys, key=str):
                record(f"{path}.{key}" if path else str(key), "missing_key", expected[key], None)
            for key in sorted(actual_keys - expected_keys, key=str):
                record(f"{path}.{key}" if path else str(key), "extra_key", None, actual[key])
            for key in sorted(expected_keys & actual_keys, key=str):
                walk(expected[key], actual[key], f"{path}.{key}" if path else str(key))
            return
        expected_array = isinstance(expected, np.ndarray)
        actual_array = isinstance(actual, np.ndarray)
        if expected_array or actual_array:
            if not expected_array or not actual_array:
                record(path, "type_mismatch", expected, actual)
                return
            if expected.shape != actual.shape:
                record(path, "shape_mismatch", expected.shape, actual.shape)
                return
            if expected.ndim == 0:
                walk(expected.item(), actual.item(), path)
            else:
                for index in np.ndindex(expected.shape):
                    walk(expected[index], actual[index], f"{path}[{','.join(map(str, index))}]")
            return
        if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
            if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
                record(path, "type_mismatch", expected, actual)
                return
            if len(expected) != len(actual):
                record(path, "count_mismatch", len(expected), len(actual))
            for i, (ev, av) in enumerate(zip(expected, actual)):
                walk(ev, av, f"{path}[{i}]")
            return
        state["n_compared"] += 1
        if expected is None or actual is None:
            if expected is not actual:
                record(path, "none_asymmetry", expected, actual)
            return
        if isinstance(expected, (bool, np.bool_)) or isinstance(actual, (bool, np.bool_)):
            if bool(expected) != bool(actual) or type(expected) is not type(actual):
                record(path, "value_mismatch", expected, actual)
            return
        if isinstance(expected, Real) and isinstance(actual, Real):
            ev, av = float(expected), float(actual)
            equal = (np.isnan(ev) and np.isnan(av)) or (np.isinf(ev) and np.isinf(av) and ev == av)
            # Counts and masks are integral scientific fields.  A one-unit
            # count corruption must fail even when a relative float tolerance
            # would otherwise hide it at full-corpus scale.
            if isinstance(expected, Integral) or isinstance(actual, Integral):
                equal = expected == actual
            elif not equal:
                equal = bool(np.isclose(ev, av, rtol=rtol, atol=atol, equal_nan=False))
            if not equal:
                record(path, "numeric_mismatch", expected, actual)
            return
        if expected != actual:
            record(path, "value_mismatch", expected, actual)

    def _preview(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            return {"keys": sorted((str(k) for k in value), key=str)[:20]}
        if isinstance(value, (list, tuple)):
            return {"count": len(value)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Real) or value is None or isinstance(value, (str, bool)):
            return value
        return repr(value)

    walk(ref, prod, "")
    state["comparison_tolerance"] = {"atol": float(atol), "rtol": float(rtol)}
    state["schema_ok"] = bool(
        state["n_missing_fields"] == 0
        and state["n_extra_fields"] == 0
        and state["n_shape_mismatches"] == 0
        and state["n_count_mismatches"] == 0
        and state["n_type_mismatches"] == 0
    )
    state["coverage_ok"] = bool(state["n_compared"] > 0)
    state["all_ok"] = bool(state["n_mismatches"] == 0 and state["schema_ok"] and state["coverage_ok"])
    return state
