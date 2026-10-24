"""Population aggregates for observation-support diagnostics.

The aggregate tree is deliberately made from row masks and typed columns.  A
population mask is retained separately from each channel's support mask so an
unsupported observation contributes to the population denominator without
being turned into a zero.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import config as C
from . import contracts as CT


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


# Parent allocation code can use this tuple without importing the aggregate
# implementation's private helpers.  It includes only columns used below.
AGGREGATE_REQUIRED_FIELDS = _required_fields()
REQUIRED_FIELDS = AGGREGATE_REQUIRED_FIELDS


def minimal_required_fields() -> tuple[str, ...]:
    """Return the typed columns needed for aggregate memmap allocation."""
    return AGGREGATE_REQUIRED_FIELDS


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
        raise ValueError(f"aggregate column {name!r} has shape {arr.shape}, expected {(n,)}")
    return arr.astype(dtype, copy=False)


def _safe_mean(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.mean(x))


def inverse_weighted_cdf_quantile(
    values: np.ndarray, weights: np.ndarray, p: float = 0.5
) -> float | None:
    """Return the first value whose cumulative positive weight reaches ``p``.

    This is the protocol's inverse-CDF definition.  It is also used for the
    component-equal median (unit weights), so no NumPy interpolation median is
    used anywhere in the aggregate tree.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return None
    values = values[ok]
    weights = weights[ok]
    order = np.argsort(values, kind="mergesort")
    values, cumulative = values[order], np.cumsum(weights[order], dtype=np.float64)
    total = float(cumulative[-1])
    return _quantile_from_sorted_cdf(values, cumulative, total, p)


def _quantile_from_sorted_cdf(
    values: np.ndarray, cumulative: np.ndarray, total: float, p: float
) -> float | None:
    """Inverse-CDF lookup for an already sorted weighted sample."""
    if total <= 0.0 or not np.isfinite(total):
        return None
    index = int(np.searchsorted(cumulative, float(p) * total, side="left"))
    index = min(max(index, 0), values.size - 1)
    return float(values[index])


def _stats_from_known(
    values: np.ndarray,
    known: np.ndarray,
    match_local: np.ndarray,
) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)[known]
    if vals.size == 0:
        component_mean = None
        component_median = None
        component_p90 = None
        match_mean = None
        match_median = None
        match_p90 = None
        n_matches = 0
    else:
        component_mean = float(np.mean(vals))
        unit_weights = np.ones(vals.shape[0], dtype=np.float64)
        component_order = np.argsort(vals, kind="mergesort")
        component_sorted = vals[component_order]
        component_cdf = np.cumsum(unit_weights[component_order], dtype=np.float64)
        component_total = float(component_cdf[-1])
        component_median = _quantile_from_sorted_cdf(
            component_sorted, component_cdf, component_total, p=0.5
        )
        component_p90 = _quantile_from_sorted_cdf(
            component_sorted, component_cdf, component_total, p=0.9
        )
        used = np.asarray(match_local, dtype=np.int64)[known]
        _, inverse, counts = np.unique(used, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].astype(np.float64)
        match_mean = float(np.sum(weights * vals) / np.sum(weights))
        match_order = np.argsort(vals, kind="mergesort")
        match_sorted = vals[match_order]
        match_cdf = np.cumsum(weights[match_order], dtype=np.float64)
        match_total = float(match_cdf[-1])
        match_median = _quantile_from_sorted_cdf(
            match_sorted, match_cdf, match_total, p=0.5
        )
        match_p90 = _quantile_from_sorted_cdf(
            match_sorted, match_cdf, match_total, p=0.9
        )
        n_matches = int(counts.size)
    return {
        "component_equal_mean": component_mean,
        "component_equal_median": component_median,
        "component_equal_p50": component_median,
        "component_equal_p90": component_p90,
        "equal_total_match_mean": match_mean,
        "equal_total_match_median": match_median,
        "equal_total_match_p50": match_median,
        "equal_total_match_p90": match_p90,
        "n_matches_known": n_matches,
    }


def _support_info(population: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    pop = np.asarray(population, dtype=bool)
    supported = pop & np.asarray(support, dtype=bool)
    n_population = int(pop.sum())
    n_supported = int(supported.sum())
    n_unknown = n_population - n_supported if n_population else 0
    return {
        "n_population": n_population,
        "n_supported": n_supported,
        "n_known": n_supported,
        "n_unknown": n_unknown,
        "support_rate": float(n_supported / n_population) if n_population else None,
    }


def weighted_stats(
    values: np.ndarray,
    population_mask: np.ndarray,
    match_local: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Component-equal and equal-total-match summaries.

    ``population_mask`` controls the denominator.  ``support_mask`` controls
    the conditional numerator; non-finite values are always unsupported.
    Grouped inverse counts create equal-total-match weights in one vectorized
    operation and avoid scanning the full table once per match.
    """
    values = np.asarray(values, dtype=np.float64)
    pop = np.asarray(population_mask, dtype=bool)
    if values.shape != pop.shape:
        raise ValueError("values and population_mask must have identical shapes")
    if support_mask is None:
        support = pop.copy()
    else:
        support = pop & np.asarray(support_mask, dtype=bool)
    known = support & np.isfinite(values)
    n_population = int(pop.sum())
    n_known = int(known.sum())
    n_unknown = n_population - n_known if n_population else 0
    result = {
        "n_population": n_population,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "n_unknown_excluded": n_unknown,
        "n_matches_known": 0,
        "support_rate": float(n_known / n_population) if n_population else None,
    }
    result.update(_stats_from_known(values, known, match_local))
    return result


def gap_denominators(lags: np.ndarray) -> dict[str, Any]:
    lags = np.asarray(lags, dtype=np.int64)
    masks = CT.candidate_masks(lags)
    out = {k: int(v.sum()) for k, v in masks.items()}
    out["n_known_lag"] = int((lags >= 0).sum())
    out["n_unknown_lag"] = int((lags < 0).sum())
    out["n_negative_lag"] = out["n_unknown_lag"]
    return out


def _bit_count(values: np.ndarray) -> np.ndarray:
    """Count the ten low bits without a Python loop over rows."""
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
    # Older snapshots did not carry c0_n/call_n in the derived table.  Bit
    # counts are exact for those rows; team decomposition remains unknown.
    out["c0_n"] = np.where(np.isfinite(out["c0_n"]), out["c0_n"], _bit_count(c0_bits))
    out["call_n"] = np.where(np.isfinite(out["call_n"]), out["call_n"], _bit_count(call_bits))
    return out


def _source_metrics(
    source: Mapping[str, np.ndarray], population: np.ndarray, match_local: np.ndarray
) -> dict[str, Any]:
    def count_stats(values: np.ndarray) -> dict[str, Any]:
        return weighted_stats(
            values,
            population,
            match_local,
            support_mask=np.isfinite(values) & (values >= 0),
        )

    def group(prefix: str) -> dict[str, Any]:
        return {
            "n": count_stats(source[f"{prefix}_n"]),
            "n_blue": count_stats(source[f"{prefix}_n_blue"]),
            "n_red": count_stats(source[f"{prefix}_n_red"]),
            "n_unknown_team": count_stats(source[f"{prefix}_n_unknown_team"]),
            "n_env_killer": count_stats(source[f"{prefix}_n_env_killer"]),
            "scale_max_side": count_stats(source[f"{prefix}_scale_max_side"]),
        }

    c0 = group("c0")
    call = group("call")
    scale_diff = source["call_scale_max_side"] - source["c0_scale_max_side"]
    scale_diff_support = (
        np.isfinite(source["call_scale_max_side"])
        & (source["call_scale_max_side"] >= 0)
        & np.isfinite(source["c0_scale_max_side"])
        & (source["c0_scale_max_side"] >= 0)
    )
    result = {
        "c0": c0,
        "c_all": call,
        "teams": {
            "c0": {
                "blue": c0["n_blue"],
                "red": c0["n_red"],
                "unknown": c0["n_unknown_team"],
            },
            "c_all": {
                "blue": call["n_blue"],
                "red": call["n_red"],
                "unknown": call["n_unknown_team"],
            },
        },
        "scale": {
            "c0": c0["scale_max_side"],
            "c_all": call["scale_max_side"],
            "c_all_minus_c0": weighted_stats(
                scale_diff, population, match_local, support_mask=scale_diff_support
            ),
        },
    }
    # Explicit aliases make the C0/C_all/team/scale evidence easy to locate in
    # reports while preserving the grouped form above.
    result["c0_stats"] = c0
    result["c_all_stats"] = call
    result["team_stats"] = result["teams"]
    result["scale_stats"] = result["scale"]
    return result


def _part_metrics(
    table: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    tag: str,
    population: np.ndarray,
    match_local: np.ndarray,
    singleton: np.ndarray,
) -> dict[str, Any]:
    n = population.shape[0]
    support = _column(table, f"support_ok_{tag}", n, bool, False)
    pl = _column(table, f"p_lower_n_{tag}", n, np.float64, np.nan)
    pu = _column(table, f"p_upper_n_{tag}", n, np.float64, np.nan)
    nl = _column(table, f"n_lower_{tag}", n, np.float64, np.nan)
    nu = _column(table, f"n_upper_{tag}", n, np.float64, np.nan)
    known_pl = support & np.isfinite(pl) & (pl >= 0)
    known_pu = support & np.isfinite(pu) & (pu >= 0)
    known_nl = support & np.isfinite(nl) & (nl >= 0)
    known_nu = support & np.isfinite(nu) & (nu >= 0)
    known_both = known_pl & known_pu
    singleton_values = np.asarray(singleton, dtype=np.float64)
    distance: dict[str, Any] = {}
    for name in ("dist_min", "dist_p50", "dist_p90"):
        values = _column(table, f"{name}_{tag}", n, np.float64, np.nan)
        distance[name] = weighted_stats(
            values,
            population,
            match_local,
            support_mask=support & np.isfinite(values) & (values >= 0),
        )
    result = {
        "n_population": int(population.sum()),
        "n_support_ok": int((population & support).sum()),
        "n_support_unknown": int(population.sum() - (population & support).sum()),
        "support": _support_info(population, support),
        "p_lower": weighted_stats(pl, population, match_local, support_mask=known_pl),
        "p_upper": weighted_stats(pu, population, match_local, support_mask=known_pu),
        "n_lower": weighted_stats(nl, population, match_local, support_mask=known_nl),
        "n_upper": weighted_stats(nu, population, match_local, support_mask=known_nu),
        "distance": distance,
        "ambiguity": {
            "p_width": weighted_stats(pu - pl, population, match_local, support_mask=known_both),
            "n_width": weighted_stats(
                nu - nl,
                population,
                match_local,
                support_mask=support & np.isfinite(nl) & np.isfinite(nu) & (nl >= 0) & (nu >= 0),
            ),
            "p_ambiguous": weighted_stats(
                (pu > pl).astype(np.float64),
                population,
                match_local,
                support_mask=known_both,
            ),
            "n_ambiguous": weighted_stats(
                (nu > nl).astype(np.float64),
                population,
                match_local,
                support_mask=support & np.isfinite(nl) & np.isfinite(nu) & (nl >= 0) & (nu >= 0),
            ),
        },
        "source": _source_metrics(source, population, match_local),
        "singleton": weighted_stats(
            singleton_values, population, match_local, support_mask=support
        ),
    }
    result["singleton_rate_known"] = result["singleton"]["component_equal_mean"]
    result["c0"] = result["source"]["c0"]
    result["c_all"] = result["source"]["c_all"]
    result["teams"] = result["source"]["teams"]
    result["scale"] = result["source"]["scale"]
    return result


def _radius_difference(
    table: Mapping[str, np.ndarray],
    cut: str,
    population: np.ndarray,
    match_local: np.ndarray,
) -> dict[str, Any]:
    n = population.shape[0]
    s_small = _column(table, f"support_ok_{cut}_r1800", n, bool, False)
    s_large = _column(table, f"support_ok_{cut}_r2000", n, bool, False)
    support = s_small & s_large
    metrics: dict[str, Any] = {}
    for field in ("p_lower_n", "p_upper_n", "n_lower", "n_upper"):
        small = _column(table, f"{field}_{cut}_r1800", n, np.float64, np.nan)
        large = _column(table, f"{field}_{cut}_r2000", n, np.float64, np.nan)
        metrics[field] = weighted_stats(
            large - small,
            population,
            match_local,
            support_mask=support & np.isfinite(small) & np.isfinite(large) & (small >= 0) & (large >= 0),
        )
    small_bits = _column(table, f"near_bits_{cut}_r1800", n, np.float64, np.nan)
    large_bits = _column(table, f"near_bits_{cut}_r2000", n, np.float64, np.nan)
    metrics["near_changed"] = weighted_stats(
        (large_bits != small_bits).astype(np.float64),
        population,
        match_local,
        support_mask=support & np.isfinite(small_bits) & np.isfinite(large_bits) & (small_bits >= 0) & (large_bits >= 0),
    )
    return {
        "from_radius": 1800,
        "to_radius": 2000,
        "support": _support_info(population, support),
        "metrics": metrics,
    }


def _post_metrics(
    table: Mapping[str, np.ndarray],
    population: np.ndarray,
    match_local: np.ndarray,
    tag: str,
    horizon: int,
    cut: str,
) -> dict[str, Any]:
    n = population.shape[0]
    valid = _column(table, f"valid_h{horizon}", n, bool, False)
    valid_pop = population & valid
    events_ok = _column(table, f"events_support_ok_{tag}", n, bool, False)
    event_pop = valid_pop & events_ok

    def numeric(field: str, *, support: np.ndarray, nonnegative: bool = False) -> dict[str, Any]:
        values = _column(table, f"{field}_{tag}", n, np.float64, np.nan)
        accepted = support & np.isfinite(values)
        if nonnegative:
            accepted &= values >= 0
        return weighted_stats(values, population, match_local, support_mask=accepted)

    def status(field: str) -> dict[str, Any]:
        values = _column(table, f"{field}_{tag}", n, bool, False)
        summary = weighted_stats(
            values.astype(np.float64), population, match_local, support_mask=valid_pop
        )
        known = population & valid_pop
        return {
            "n_population": int(population.sum()),
            "n_known": int(known.sum()),
            "n_unknown": int(population.sum() - known.sum()),
            "n_true": int((known & values).sum()),
            "rate": float((known & values).sum() / known.sum()) if known.any() else None,
            "stats": summary,
        }

    channels = {
        name: numeric(name, support=valid_pop)
        for name in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean")
    }
    events = {
        "Fpre_to_Fpost": numeric(
            "async_events_Fpre_to_Fpost", support=event_pop, nonnegative=True
        ),
        "cutoff_to_endpoint": numeric(
            "async_events_cutoff_to_endpoint", support=event_pop, nonnegative=True
        ),
    }
    same = _column(table, f"same_as_Fpre_{tag}", n, bool, False)
    fgt = _column(table, f"F_post_gt_L_{tag}", n, bool, False)
    stale = _column(table, f"stale_at_or_before_L_{tag}", n, bool, False)
    cutoff_events = _column(
        table, f"async_events_cutoff_to_endpoint_{tag}", n, np.float64, np.nan
    )
    frame_events = _column(
        table, f"async_events_Fpre_to_Fpost_{tag}", n, np.float64, np.nan
    )
    same_with_events = (
        population
        & valid
        & same
        & events_ok
        & np.isfinite(cutoff_events)
        & (cutoff_events > 0)
    )
    result = {
        "n_population": int(population.sum()),
        "n_valid": int(valid_pop.sum()),
        "n_invalid": int(population.sum() - valid_pop.sum()),
        "valid_rate": float(valid_pop.sum() / population.sum()) if population.any() else None,
        "n_event_support_ok": int(event_pop.sum()),
        "n_event_unknown": int(valid_pop.sum() - event_pop.sum()),
        "event_support_rate": float(event_pop.sum() / valid_pop.sum()) if valid_pop.any() else None,
        "same_as_Fpre": status("same_as_Fpre"),
        "F_post_gt_L": status("F_post_gt_L"),
        "stale_at_or_before_L": status("stale_at_or_before_L"),
        "F_post": numeric("F_post", support=valid_pop, nonnegative=True),
        "F_post_idx": numeric("F_post_idx", support=valid_pop, nonnegative=True),
        "age_post": numeric("age_post", support=valid_pop, nonnegative=True),
        "frame_idx_gap": numeric("frame_idx_gap", support=valid_pop, nonnegative=True),
        "timestamp_gap": numeric("timestamp_gap", support=valid_pop, nonnegative=True),
        "async_events_Fpre_to_Fpost": events["Fpre_to_Fpost"],
        "async_events_cutoff_to_endpoint": events["cutoff_to_endpoint"],
        "events": events,
        "channels": channels,
        "same_frame_with_async_events": int(same_with_events.sum()),
        "n_async_events_Fpre_to_Fpost_positive": int(
            (population & event_pop & np.isfinite(frame_events) & (frame_events > 0)).sum()
        ),
        "n_async_events_cutoff_to_endpoint_positive": int(
            (population & event_pop & np.isfinite(cutoff_events) & (cutoff_events > 0)).sum()
        ),
    }
    # Existing verifier and reports consume these candidate-cell aliases.
    result["n_cand_valid"] = result["n_valid"]
    result["n_same_as_Fpre"] = result["same_as_Fpre"]["n_true"]
    result["n_F_post_gt_L"] = result["F_post_gt_L"]["n_true"]
    result["n_stale_at_or_before_L"] = result["stale_at_or_before_L"]["n_true"]
    result["d_gold_blue"] = channels["d_gold_blue"]
    result["d_gold_red"] = channels["d_gold_red"]
    result["d_level_mean"] = channels["d_level_mean"]
    result["d_xp_mean"] = channels["d_xp_mean"]
    result["d_hp_mean"] = channels["d_hp_mean"]
    return result


def build_aggregates(
    table: Mapping[str, np.ndarray],
    n_matches_source: int,
    n_zero_kill: int,
    *,
    n_matches_processed: int | None = None,
    n_zero_kill_processed: int | None = None,
) -> dict[str, Any]:
    """Build the complete full/candidate/isolated aggregate report tree."""
    n = int(np.asarray(table["match_local"]).shape[0])
    lags_tau = _column(table, "lag_f_F_tau", n, np.int64, C.SENTINEL_MISSING)
    lags_tm1 = _column(table, "lag_f_F_tm1", n, np.int64, C.SENTINEL_MISSING)
    match_local = _column(table, "match_local", n, np.int64, -1)
    foreign = _column(table, "foreign_kills_in_tau_L", n, np.int64, 0)
    other = _column(table, "other_anchors_in_tau_L", n, np.int64, 0)
    singleton = _column(table, "singleton", n, bool, False)
    full = np.ones(n, dtype=bool)
    cand_tau = (lags_tau >= 10_000) & (lags_tau <= 20_000)
    # Reconstruct the second cutoff from its own lag column.  The stored flag
    # remains evidence for row-level verification, but cannot silently choose
    # the post population when it disagrees with the cutoff timestamp.
    cand_tm1 = (lags_tm1 >= 10_000) & (lags_tm1 <= 20_000)
    isol_tau = cand_tau & (foreign == 0) & (other == 0)
    isol_tm1 = cand_tm1 & (foreign == 0) & (other == 0)
    source = _source_columns(table, n)
    gaps_tau = gap_denominators(lags_tau)
    gaps_tm1 = gap_denominators(lags_tm1)

    out: dict[str, Any] = {
        "aggregate_schema": "observation_support_aggregates_v3",
        "n_components": n,
        "n_matches_source": int(n_matches_source),
        "n_zero_kill_matches_source": int(n_zero_kill),
        "n_matches_processed": int(
            n_matches_processed if n_matches_processed is not None else n_matches_source
        ),
        "n_zero_kill_processed": int(
            n_zero_kill_processed if n_zero_kill_processed is not None else n_zero_kill
        ),
        "n_matches_with_components": int(np.unique(match_local).size) if n else 0,
        "gap_tau": gaps_tau,
        "gap_tm1": gaps_tm1,
        "gap_groups": {"tau": gaps_tau, "tm1": gaps_tm1},
        "n_cand_10_20_tau": int(cand_tau.sum()),
        "n_isolated_cand_10_20_tau": int(isol_tau.sum()),
        "n_cand_10_20_tm1": int(cand_tm1.sum()),
        "n_isolated_cand_10_20_tm1": int(isol_tm1.sum()),
        "n_exact_10000_tau": int(_column(table, "exact_10000_tau", n, bool, False).sum()),
        "n_strict_eligible": int(_column(table, "strict_eligible", n, bool, False).sum()),
        "n_frame_differs": int(_column(table, "frame_differs", n, bool, False).sum()),
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

    cut_masks = {"tau": (cand_tau, isol_tau), "tm1": (cand_tm1, isol_tm1)}
    cells: dict[str, Any] = {}
    radius_differences: dict[str, Any] = {}
    for cut, (candidate, isolated) in cut_masks.items():
        population_masks = {"full": full, "candidate": candidate, "isolated": isolated}
        radius_differences[cut] = {
            key: _radius_difference(table, cut, mask, match_local)
            for key, mask in population_masks.items()
        }
        for radius in (1800, 2000):
            tag = f"{cut}_r{radius}"
            cell = {
                "n_population_full": int(full.sum()),
                "n_population_cand": int(candidate.sum()),
                "n_population_isolated": int(isolated.sum()),
                "full": _part_metrics(table, source, tag, full, match_local, singleton),
                "candidate": _part_metrics(
                    table, source, tag, candidate, match_local, singleton
                ),
                "isolated": _part_metrics(
                    table, source, tag, isolated, match_local, singleton
                ),
                "radius_difference": radius_differences[cut],
            }
            # Backward-compatible aliases point at the already-computed
            # candidate summaries; no duplicate calculations are performed.
            candidate_cell = cell["candidate"]
            cell.update(
                {
                    "n_support_ok": candidate_cell["n_support_ok"],
                    "p_lower": candidate_cell["p_lower"],
                    "p_upper": candidate_cell["p_upper"],
                    "n_lower": candidate_cell["n_lower"],
                    "n_upper": candidate_cell["n_upper"],
                }
            )
            cells[tag] = cell
    out["participation_cells"] = cells
    out["participation_radius_differences"] = radius_differences
    out["population_cells"] = {
        cut: {
            key: _source_metrics(source, mask, match_local)
            for key, mask in {
                "full": full,
                "candidate": candidate,
                "isolated": isolated,
            }.items()
        }
        for cut, (candidate, isolated) in cut_masks.items()
    }

    post: dict[str, Any] = {}
    for horizon in C.HORIZONS_S:
        for cut, (candidate, isolated) in cut_masks.items():
            tag = f"{cut}_h{horizon}"
            populations = {"full": full, "candidate": candidate, "isolated": isolated}
            post_cells = {
                key: _post_metrics(table, mask, match_local, tag, horizon, cut)
                for key, mask in populations.items()
            }
            candidate_cell = post_cells["candidate"]
            post[tag] = {
                "n_population_full": int(full.sum()),
                "n_population_cand": int(candidate.sum()),
                "n_population_isolated": int(isolated.sum()),
                "full": post_cells["full"],
                "candidate": candidate_cell,
                "isolated": post_cells["isolated"],
                # Existing report/verifier aliases use the candidate cell.
                "n_cand_valid": candidate_cell["n_valid"],
                "n_same_as_Fpre": candidate_cell["n_same_as_Fpre"],
                "n_F_post_gt_L": candidate_cell["n_F_post_gt_L"],
                "n_stale_at_or_before_L": candidate_cell["n_stale_at_or_before_L"],
                "async_events_Fpre_to_Fpost": candidate_cell[
                    "async_events_Fpre_to_Fpost"
                ],
                "async_events_cutoff_to_endpoint": candidate_cell[
                    "async_events_cutoff_to_endpoint"
                ],
                "d_gold_blue": candidate_cell["d_gold_blue"],
                "d_gold_red": candidate_cell["d_gold_red"],
                "d_level_mean": candidate_cell["d_level_mean"],
                "d_xp_mean": candidate_cell["d_xp_mean"],
                "d_hp_mean": candidate_cell["d_hp_mean"],
                "same_frame_with_async_events": candidate_cell[
                    "same_frame_with_async_events"
                ],
                "n_async_events_Fpre_to_Fpost_positive": candidate_cell[
                    "n_async_events_Fpre_to_Fpost_positive"
                ],
                "n_async_events_cutoff_to_endpoint_positive": candidate_cell[
                    "n_async_events_cutoff_to_endpoint_positive"
                ],
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
