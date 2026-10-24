"""Independent persisted-table verification (does NOT call production aggregate helpers)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import config as C


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    if a != a and b != b:
        return True
    return abs(float(a) - float(b)) <= tol


def _eq_weights_obs(match_local: np.ndarray) -> np.ndarray:
    ml = np.asarray(match_local, dtype=np.int64)
    if ml.size == 0:
        return np.zeros(0, dtype=np.float64)
    counts = np.bincount(ml, minlength=int(ml.max()) + 1).astype(np.float64)
    w = 1.0 / counts[ml]
    s = float(w.sum())
    return w / s if s > 0 else w


def _eq_weights_cand(match_local: np.ndarray) -> np.ndarray:
    ml = np.asarray(match_local, dtype=np.int64)
    if ml.size == 0:
        return np.zeros(0, dtype=np.float64)
    counts = np.bincount(ml, minlength=int(ml.max()) + 1).astype(np.float64)
    w = 1.0 / counts[ml]
    s = float(w.sum())
    return w / s if s > 0 else w


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    sw = float(w.sum())
    return float(np.dot(x, w) / sw) if sw > 0 else float("nan")


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    cw = np.cumsum(ws)
    total = float(cw[-1])
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(cw, float(q) * total, side="left"))
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])


def _cohort_mask(components: dict[str, np.ndarray], cohort: str) -> np.ndarray:
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


def _slice_stats(
    status: np.ndarray,
    delay_ms: np.ndarray,
    match_local: np.ndarray,
) -> dict[str, Any]:
    st = np.asarray(status, dtype=np.int64)
    delay = np.asarray(delay_ms, dtype=np.float64)
    ml = np.asarray(match_local, dtype=np.int64)
    n = int(st.size)
    obs = st == C.ST_OBSERVED
    n_obs = int(obs.sum())
    out: dict[str, Any] = {
        "n_candidates": n,
        "n_observed_acquisition": n_obs,
        "n_no_acquisition": int((st == C.ST_NO_ACQ).sum()),
        "n_unknown_observation": int((st == C.ST_UNKNOWN_OBS).sum()),
        "n_no_interval": int((st == C.ST_NO_INTERVAL).sum()),
    }
    if n_obs == 0:
        out["event_mean_s"] = float("nan")
        out["event_median_s"] = float("nan")
        out["event_p25_s"] = float("nan")
        out["event_p75_s"] = float("nan")
        out["event_p90_s"] = float("nan")
        out["eq_mean_s"] = float("nan")
        out["eq_median_s"] = float("nan")
        out["prop_le_cond"] = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
        out["eq_prop_le_cond"] = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
    else:
        ds = delay[obs] / 1000.0
        ml_o = ml[obs]
        ew = np.ones(n_obs, dtype=np.float64)
        mw = _eq_weights_obs(ml_o)
        out["event_mean_s"] = _wmean(ds, ew)
        out["event_median_s"] = _wquantile(ds, ew, 0.5)
        out["event_p25_s"] = _wquantile(ds, ew, 0.25)
        out["event_p75_s"] = _wquantile(ds, ew, 0.75)
        out["event_p90_s"] = _wquantile(ds, ew, 0.90)
        out["eq_mean_s"] = _wmean(ds, mw)
        out["eq_median_s"] = _wquantile(ds, mw, 0.5)
        out["prop_le_cond"] = {
            str(th): float(np.average(ds <= float(th), weights=ew)) for th in C.DELAY_THRESHOLDS_S
        }
        out["eq_prop_le_cond"] = {
            str(th): float(np.average(ds <= float(th), weights=mw)) for th in C.DELAY_THRESHOLDS_S
        }

    # Cohort-denominator threshold coverage
    if n == 0:
        out["cov_le_event"] = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
        out["cov_le_eq"] = {str(th): float("nan") for th in C.DELAY_THRESHOLDS_S}
    else:
        delay_s = np.where(obs, delay / 1000.0, np.inf)
        cw = _eq_weights_cand(ml)
        out["cov_le_event"] = {}
        out["cov_le_eq"] = {}
        for th in C.DELAY_THRESHOLDS_S:
            hit = (obs & (delay_s <= float(th))).astype(np.float64)
            out["cov_le_event"][str(th)] = float(hit.mean())
            out["cov_le_eq"][str(th)] = float(np.dot(hit, cw))
    return out


def verify_aggregates_independent(
    components: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    aggregates: dict[str, Any],
    *,
    categories: Sequence[str] = C.CATEGORY_ORDER,
    cohorts: Sequence[str] | None = None,
    intervals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Recompute every cohort×interval×category metric without production AGG.*."""
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

    mismatches: list[str] = []
    n_checked = 0

    for cohort in cohorts:
        mask = _cohort_mask(components, cohort)
        n_comp = int(mask.sum())
        n_matches = int(np.unique(components["match_local"][mask]).size) if n_comp else 0
        saved_size = aggregates["cohort_sizes"][cohort]
        if int(saved_size["n_components"]) != n_comp:
            mismatches.append(f"{cohort}:n_components {saved_size['n_components']}!={n_comp}")
        if int(saved_size["n_matches"]) != n_matches:
            mismatches.append(f"{cohort}:n_matches {saved_size['n_matches']}!={n_matches}")
        n_checked += 2

        for interval in intervals:
            for cat in categories:
                key = f"{interval}:{cat}"
                st = np.asarray(query[f"status__{key}"], dtype=np.int64)[mask]
                delay = np.asarray(query[f"delay_ms__{key}"], dtype=np.int64)[mask]
                ml = np.asarray(components["match_local"], dtype=np.int64)[mask]
                ind = _slice_stats(st, delay, ml)
                saved = aggregates["cohorts"][cohort][interval][cat]
                pairs = [
                    ("n_candidates", ind["n_candidates"], saved["n_candidates"]),
                    ("n_observed_acquisition", ind["n_observed_acquisition"], saved["n_observed_acquisition"]),
                    ("n_no_acquisition", ind["n_no_acquisition"], saved["n_no_acquisition"]),
                    ("n_unknown_observation", ind["n_unknown_observation"], saved["n_unknown_observation"]),
                    ("n_no_interval", ind["n_no_interval"], saved["n_no_interval"]),
                    (
                        "event_mean",
                        ind["event_mean_s"],
                        saved["delays"]["event_weighted"]["mean_s"],
                    ),
                    (
                        "event_median",
                        ind["event_median_s"],
                        saved["delays"]["event_weighted"]["median_s"],
                    ),
                    (
                        "event_p25",
                        ind["event_p25_s"],
                        saved["delays"]["event_weighted"]["p25_s"],
                    ),
                    (
                        "event_p75",
                        ind["event_p75_s"],
                        saved["delays"]["event_weighted"]["p75_s"],
                    ),
                    (
                        "event_p90",
                        ind["event_p90_s"],
                        saved["delays"]["event_weighted"]["p90_s"],
                    ),
                    (
                        "eq_mean",
                        ind["eq_mean_s"],
                        saved["delays"]["equal_match_weighted"]["mean_s"],
                    ),
                    (
                        "eq_median",
                        ind["eq_median_s"],
                        saved["delays"]["equal_match_weighted"]["median_s"],
                    ),
                ]
                for name, a, b in pairs:
                    n_checked += 1
                    if isinstance(a, (int, np.integer)):
                        if int(a) != int(b):
                            mismatches.append(f"{cohort}/{interval}/{cat}:{name} {a}!={b}")
                    elif not _close(float(a), float(b)):
                        mismatches.append(f"{cohort}/{interval}/{cat}:{name} {a}!={b}")

                for th in C.DELAY_THRESHOLDS_S:
                    n_checked += 4
                    sth = str(th)
                    cond_e = saved["delays"]["event_weighted"]["prop_le"][sth]
                    cond_m = saved["delays"]["equal_match_weighted"]["prop_le"][sth]
                    if not _close(ind["prop_le_cond"][sth], float(cond_e)):
                        mismatches.append(
                            f"{cohort}/{interval}/{cat}:prop_le_cond_e@{sth}"
                        )
                    if not _close(ind["eq_prop_le_cond"][sth], float(cond_m)):
                        mismatches.append(
                            f"{cohort}/{interval}/{cat}:prop_le_cond_m@{sth}"
                        )
                    cov_e = saved["threshold_coverage_cohort"]["event_weighted"][sth]
                    cov_m = saved["threshold_coverage_cohort"]["equal_match_weighted"][sth]
                    if not _close(ind["cov_le_event"][sth], float(cov_e)):
                        mismatches.append(
                            f"{cohort}/{interval}/{cat}:cov_le_e@{sth}"
                        )
                    if not _close(ind["cov_le_eq"][sth], float(cov_m)):
                        mismatches.append(
                            f"{cohort}/{interval}/{cat}:cov_le_m@{sth}"
                        )

    return {
        "n_checked": n_checked,
        "n_mismatches": len(mismatches),
        "mismatches_head": mismatches[:50],
        "all_ok": len(mismatches) == 0,
        "method": "independent_direct_calculation_no_production_aggregate_helpers",
    }


def verify_events_and_query_support(
    components: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    *,
    expect_full: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    n_comp = int(components["L"].shape[0])
    checks["n_components"] = n_comp
    checks["component_keys_present"] = all(
        k in components
        for k in (
            "match_local",
            "component_id",
            "L",
            "delta",
            "game_end_ms",
            "game_end_known",
            "next_kill_after_L",
            "next_component_tau",
            "foreign_kills_in_tau_L",
            "other_anchors_in_tau_L",
            "strict_eligible",
        )
    )
    checks["match_local_sorted"] = bool(np.all(np.diff(components["match_local"]) >= 0))
    checks["n_events"] = int(events["t_ms"].shape[0])
    acq = events["acq_status"]
    checks["n_acq_confirmed"] = int((acq == C.ACQ_CONFIRMED).sum())
    checks["n_acq_neutral"] = int((acq == C.ACQ_NEUTRAL).sum())
    checks["n_acq_unknown"] = int((acq == C.ACQ_UNKNOWN).sum())
    checks["n_acq_contradictory"] = int((acq == C.ACQ_CONTRADICTORY).sum())
    checks["neutral_524_ok"] = (checks["n_acq_neutral"] == 524) if expect_full else True
    checks["unknown_zero_ok"] = (checks["n_acq_unknown"] == 0) if expect_full else True
    checks["contradictory_zero_ok"] = (
        (checks["n_acq_contradictory"] == 0) if expect_full else True
    )

    # Query key completeness
    missing_keys = []
    for cat in C.CATEGORY_ORDER:
        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
            for field in (
                "status",
                "delay_ms",
                "event_ord",
                "t_ms",
                "uncertain_in_interval",
                "neutral_in_interval",
            ):
                k = f"{field}__{interval}:{cat}"
                if k not in query:
                    missing_keys.append(k)
                elif int(np.asarray(query[k]).shape[0]) != n_comp:
                    missing_keys.append(f"{k}:len")
    checks["query_keys_complete"] = len(missing_keys) == 0
    checks["missing_query_keys_head"] = missing_keys[:20]

    # Strict-after-L: observed delay > 0
    st = query[f"status__{C.INTERVAL_UNCAPED}:any_objective"]
    delay = query[f"delay_ms__{C.INTERVAL_UNCAPED}:any_objective"]
    obs = st == C.ST_OBSERVED
    checks["strict_after_L_ok"] = bool(np.all(delay[obs] > 0)) if obs.any() else True
    checks["nonneg_delay_ok"] = bool(np.all(delay[obs] >= 0)) if obs.any() else True

    # GAME_END support: unknown game_end → UNKNOWN_OBS on both intervals
    gek = components["game_end_known"].astype(bool)
    if (~gek).any():
        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
            st_u = query[f"status__{interval}:any_objective"]
            bad = (~gek) & (st_u != C.ST_UNKNOWN_OBS)
            checks[f"unknown_game_end_{interval}_ok"] = not bool(bad.any())
    else:
        checks["unknown_game_end_uncapped_to_game_end_ok"] = True
        checks["unknown_game_end_pre_next_fight_ok"] = True

    if expect_full:
        band = (components["delta"] >= 10_000) & (components["delta"] <= 20_000)
        isol = band & (components["foreign_kills_in_tau_L"] == 0) & (
            components["other_anchors_in_tau_L"] == 0
        )
        checks["full_window_count_ok"] = int(band.sum()) == C.EXPECTED_FULL_WINDOW
        checks["full_isol_count_ok"] = int(isol.sum()) == C.EXPECTED_FULL_ISOL
        checks["n_components_ok"] = n_comp == C.EXPECTED_MIN1_COMPONENTS
    else:
        checks["full_window_count_ok"] = True
        checks["full_isol_count_ok"] = True
        checks["n_components_ok"] = True

    checks["all_ok"] = bool(
        checks["component_keys_present"]
        and checks["match_local_sorted"]
        and checks["neutral_524_ok"]
        and checks["unknown_zero_ok"]
        and checks["contradictory_zero_ok"]
        and checks["query_keys_complete"]
        and checks["strict_after_L_ok"]
        and checks["nonneg_delay_ok"]
        and checks["unknown_game_end_uncapped_to_game_end_ok"]
        and checks["unknown_game_end_pre_next_fight_ok"]
        and checks["full_window_count_ok"]
        and checks["full_isol_count_ok"]
        and checks["n_components_ok"]
    )
    return checks
