"""Cohort aggregation with event-weighted and equal-match-weighted delay stats."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import config as C


def equal_match_weights(match_local: np.ndarray) -> np.ndarray:
    """w_i = 1/n_obs_in_match(i); caller may normalize globally.

    Used only for conditional delay summaries over observed acquisitions.
    """
    ml = np.asarray(match_local, dtype=np.int64)
    if ml.size == 0:
        return np.asarray([], dtype=np.float64)
    mx = int(ml.max()) + 1
    counts = np.bincount(ml, minlength=mx).astype(np.float64)
    w = 1.0 / counts[ml]
    return w


def equal_match_candidate_weights(match_local: np.ndarray) -> np.ndarray:
    """w_i = 1/n_candidates_in_match(i) over the eligible cohort rows."""
    ml = np.asarray(match_local, dtype=np.int64)
    if ml.size == 0:
        return np.asarray([], dtype=np.float64)
    mx = int(ml.max()) + 1
    counts = np.bincount(ml, minlength=mx).astype(np.float64)
    return 1.0 / counts[ml]


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if s <= 0 or w.size == 0:
        return w
    return w / s


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    return float(np.dot(x, w) / sw)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile via sorted cumulative weight (q in [0,1])."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    cw = np.cumsum(ws)
    total = float(cw[-1])
    if total <= 0:
        return float("nan")
    target = float(q) * total
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])


def delay_summary(
    delay_ms: np.ndarray,
    match_local: np.ndarray,
) -> dict[str, Any]:
    """Conditional delay summaries on observed acquisitions only (seconds).

    prop_le_* here is the conditional CDF of observed delays.
    Cohort-denominator threshold coverage is computed separately.
    Zero observed → all delay stats NA.
    """
    d = np.asarray(delay_ms, dtype=np.float64)
    ml = np.asarray(match_local, dtype=np.int64)
    n = int(d.size)
    if n == 0:
        return {
            "n_observed": 0,
            "n_matches_observed": 0,
            "event_weighted": _empty_delay_block(),
            "equal_match_weighted": _empty_delay_block(),
        }
    delay_s = d / 1000.0
    ew = np.ones(n, dtype=np.float64)
    mw = normalize_weights(equal_match_weights(ml))

    def block(weights: np.ndarray) -> dict[str, Any]:
        w = weights
        return {
            "mean_s": weighted_mean(delay_s, w),
            "median_s": weighted_quantile(delay_s, w, 0.5),
            "p25_s": weighted_quantile(delay_s, w, 0.25),
            "p75_s": weighted_quantile(delay_s, w, 0.75),
            "p90_s": weighted_quantile(delay_s, w, 0.90),
            # Conditional CDF among observed delays only (explicit label).
            "prop_le_conditional_on_observed": {
                str(th): float(np.average(delay_s <= float(th), weights=w))
                for th in C.DELAY_THRESHOLDS_S
            },
            # Back-compat alias for prior report readers (same conditional values).
            "prop_le": {
                str(th): float(np.average(delay_s <= float(th), weights=w))
                for th in C.DELAY_THRESHOLDS_S
            },
        }

    return {
        "n_observed": n,
        "n_matches_observed": int(np.unique(ml).size),
        "event_weighted": block(ew),
        "equal_match_weighted": block(mw),
    }


def _empty_delay_block() -> dict[str, Any]:
    nan_prop = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
    return {
        "mean_s": float("nan"),
        "median_s": float("nan"),
        "p25_s": float("nan"),
        "p75_s": float("nan"),
        "p90_s": float("nan"),
        "prop_le_conditional_on_observed": dict(nan_prop),
        "prop_le": dict(nan_prop),
    }


def threshold_coverage(
    status: np.ndarray,
    delay_ms: np.ndarray,
    match_local: np.ndarray,
) -> dict[str, Any]:
    """Acquisition-within-threshold coverage over the ENTIRE eligible cohort.

    Denominator definitions:
    - event_weighted: each eligible candidate weight 1; denom = n_candidates
    - equal_match_weighted: each eligible candidate weight 1/n_cand_in_match;
      denom is the sum of those weights (equals n_matches)

    Numerator: status==OBSERVED and delay_s <= threshold.
    NO_ACQ, UNKNOWN_OBS, NO_INTERVAL count in the denominator but not numerator
    (not a confirmed acquisition within the threshold). Terminal-ending intervals
    that yield NO_ACQ/OBSERVED are treated like any other status above.
    Zero candidates → NA.
    """
    st = np.asarray(status, dtype=np.int64)
    delay = np.asarray(delay_ms, dtype=np.float64)
    ml = np.asarray(match_local, dtype=np.int64)
    n = int(st.size)
    if n == 0:
        empty = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
        return {
            "denominator": "entire_eligible_cohort",
            "n_candidates": 0,
            "n_matches": 0,
            "event_weighted": dict(empty),
            "equal_match_weighted": dict(empty),
            "unknown_no_interval_treatment": (
                "included_in_denominator_excluded_from_numerator"
            ),
        }

    delay_s = np.where(st == C.ST_OBSERVED, delay / 1000.0, np.inf)
    hit = {
        str(th): (st == C.ST_OBSERVED) & (delay_s <= float(th))
        for th in C.DELAY_THRESHOLDS_S
    }
    ew = np.ones(n, dtype=np.float64)
    mw = equal_match_candidate_weights(ml)

    def block(weights: np.ndarray) -> dict[str, float]:
        w = normalize_weights(weights)
        return {
            th: float(np.dot(mask.astype(np.float64), w)) for th, mask in hit.items()
        }

    return {
        "denominator": "entire_eligible_cohort",
        "n_candidates": n,
        "n_matches": int(np.unique(ml).size),
        "event_weighted": block(ew),
        "equal_match_weighted": block(mw),
        "unknown_no_interval_treatment": (
            "included_in_denominator_excluded_from_numerator"
        ),
    }


def cohort_mask(components: dict[str, np.ndarray], cohort: str) -> np.ndarray:
    n = int(components["L"].shape[0])
    if cohort == C.COHORT_ALL:
        return np.ones(n, dtype=bool)
    delta = components["delta"]
    if cohort == C.COHORT_FULL_WINDOW:
        return (delta >= 10_000) & (delta <= 20_000)
    if cohort == C.COHORT_FULL_ISOL:
        return (
            (delta >= 10_000)
            & (delta <= 20_000)
            & (components["foreign_kills_in_tau_L"] == 0)
            & (components["other_anchors_in_tau_L"] == 0)
        )
    if cohort == C.COHORT_10_15:
        return (delta >= 10_000) & (delta <= 15_000)
    if cohort == C.COHORT_STRICT:
        return components["strict_eligible"].astype(bool)
    raise KeyError(cohort)


def summarize_query_slice(
    status: np.ndarray,
    delay_ms: np.ndarray,
    match_local: np.ndarray,
    event_ord: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    neutral_in_interval: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate one cohort×category×interval slice."""
    m = np.asarray(eligible_mask, dtype=bool)
    st = np.asarray(status, dtype=np.int64)[m]
    delay = np.asarray(delay_ms, dtype=np.int64)[m]
    ml = np.asarray(match_local, dtype=np.int64)[m]
    eord = np.asarray(event_ord, dtype=np.int64)[m]

    n_cand = int(m.sum())
    n_matches = int(np.unique(ml).size) if n_cand else 0
    n_obs = int((st == C.ST_OBSERVED).sum())
    n_no = int((st == C.ST_NO_ACQ).sum())
    n_unk = int((st == C.ST_UNKNOWN_OBS).sum())
    n_noint = int((st == C.ST_NO_INTERVAL).sum())
    n_neutral_iv = 0
    if neutral_in_interval is not None and n_cand:
        n_neutral_iv = int(np.asarray(neutral_in_interval, dtype=np.int8)[m].astype(bool).sum())

    obs = st == C.ST_OBSERVED
    delay_sum = delay_summary(delay[obs], ml[obs])
    cov_thr = threshold_coverage(st, delay, ml)

    uniq_events = 0
    multiplicity_mean = float("nan")
    if n_obs:
        keys = ml[obs].astype(np.int64) * 10_000_000 + eord[obs].astype(np.int64)
        uniq_events = int(np.unique(keys).size)
        multiplicity_mean = float(n_obs) / float(uniq_events) if uniq_events else float("nan")

    return {
        "n_candidates": n_cand,
        "n_matches": n_matches,
        "n_observed_acquisition": n_obs,
        "n_no_acquisition": n_no,
        "n_unknown_observation": n_unk,
        "n_no_interval": n_noint,
        "n_neutral_in_interval": n_neutral_iv,
        "coverage_observed": float(n_obs) / float(n_cand) if n_cand else float("nan"),
        "coverage_no_acquisition": float(n_no) / float(n_cand) if n_cand else float("nan"),
        "coverage_unknown": float(n_unk) / float(n_cand) if n_cand else float("nan"),
        "coverage_no_interval": float(n_noint) / float(n_cand) if n_cand else float("nan"),
        "coverage_neutral_in_interval": float(n_neutral_iv) / float(n_cand) if n_cand else float("nan"),
        "unique_first_objective_events": uniq_events,
        "query_count_over_unique_events": multiplicity_mean,
        "delays": delay_sum,
        "threshold_coverage_cohort": {
            "event_weighted": cov_thr["event_weighted"],
            "equal_match_weighted": cov_thr["equal_match_weighted"],
            "denominator": cov_thr["denominator"],
            "unknown_no_interval_treatment": cov_thr["unknown_no_interval_treatment"],
        },
        "definitions": {
            "delay_mean_median_quantiles_prop_le": "conditional_on_observed_acquisition",
            "threshold_coverage_cohort": "entire_eligible_cohort_denominator",
            "equal_match_delay_weights": "1_over_n_observed_in_match",
            "equal_match_coverage_weights": "1_over_n_candidates_in_match",
            "zero_observed": "conditional_delay_stats_NA_coverage_still_defined",
        },
    }


def aggregate_all(
    components: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    categories: Sequence[str] = C.CATEGORY_ORDER,
    cohorts: Sequence[str] | None = None,
    intervals: Sequence[str] | None = None,
) -> dict[str, Any]:
    if cohorts is None:
        cohorts = (
            C.COHORT_ALL,
            C.COHORT_FULL_WINDOW,
            C.COHORT_FULL_ISOL,
            C.COHORT_10_15,
            C.COHORT_STRICT,
        )
    if intervals is None:
        intervals = (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT)

    out: dict[str, Any] = {"cohorts": {}, "cohort_sizes": {}}
    for cohort in cohorts:
        mask = cohort_mask(components, cohort)
        out["cohort_sizes"][cohort] = {
            "n_components": int(mask.sum()),
            "n_matches": int(np.unique(components["match_local"][mask]).size) if mask.any() else 0,
        }
        out["cohorts"][cohort] = {}
        for interval in intervals:
            out["cohorts"][cohort][interval] = {}
            for cat in categories:
                key = f"{interval}:{cat}"
                st = query[f"status__{key}"]
                delay = query[f"delay_ms__{key}"]
                eord = query[f"event_ord__{key}"]
                neut_key = f"neutral_in_interval__{key}"
                neut = query.get(neut_key)
                out["cohorts"][cohort][interval][cat] = summarize_query_slice(
                    st,
                    delay,
                    components["match_local"],
                    eord,
                    mask,
                    neutral_in_interval=neut,
                )
    return out
