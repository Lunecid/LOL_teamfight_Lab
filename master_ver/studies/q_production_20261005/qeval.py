"""q evaluation: match-weighted metrics, strata, per-match losses, paired match bootstrap, P1/P2.

Specification (line by line in reports/q_production_20261005/MEMO_eval_KO.md):
``reports/feature_control_study_20261001/EXECUTION_CARDS.json`` q.weighting (568-574) and statistics
(1016-1062); ``FULL_STUDY_PROTOCOL.md`` 177 and 190-202; ``CARTESIAN_COMPARISON_MATRIX.json``
primary_raw_Brier_G_by_S_contrast (9554-9581); ``reports/architecture_20261005/ARCHITECTURE_KO.md``
5.1-5.2 with ``DECISIONS_20261005.json`` D-V1 and D-Q11.  The one-seed contract removes seed
averaging (``reports/research_execution_one_seed_20261003/CONTRACT.json`` phase1.q_metrics).

Every q metric weights each match equally: a match with m eligible rows gives each row 1/m, so
Brier = (1/M) sum_matches (1/m) sum_rows (q - Y)^2.  Metric functions take an explicit ``weights``
array; ``weights=None`` with ``match_ids`` means these match weights, and ``np.ones(n)`` is the
unit-row sensitivity.  Nothing here reads predictions from disk.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

VERSION = "q_eval_v1_20261005"

LOGLOSS_EPS = 1e-15  # the cards give no clip for the q logloss metric (memo, open item)
LOGIT_CLIP = 1e-6  # EC calibration.q.clip / FSP 192; reused for the slope/intercept metric
SOLVER = {"method": "L-BFGS-B", "x0": (1.0, 0.0), "maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10}  # EC calibration.solver
ACCURACY_THRESHOLD = 0.5  # D-Q11: p >= 0.5 -> 1
RELIABILITY_BINS = 10
RELIABILITY_EDGES = np.arange(RELIABILITY_BINS + 1, dtype=np.float64) / RELIABILITY_BINS  # k/10 correctly rounded
TIME_MINUTE_BOUNDARIES = (15.0, 25.0)  # EC statistics.time_minute_boundaries
GAME_TIME_ORDER = ("[0,15)", "[15,25)", "[25,inf)")
MINUTE_FIRST, MINUTE_LAST = 2, 40  # D-V1
MINUTE_ORDER = ("[0,2)",) + tuple(f"[{m},{m + 1})" for m in range(MINUTE_FIRST, MINUTE_LAST)) + ("[40,inf)",)
DV1_MIN_ROWS, DV1_MIN_CLASS = 100, 10  # D-V1: empty metric when a bin has <100 rows or <10 of a class
BALANCED_P = (0.4, 0.6)  # EC statistics.balanced_p, closed interval
BALANCED_ORDER = ("[0.4,0.6]", "outside")
FRAME_AGE_SECONDS = (0.0, 5.0, 10.0)  # EC frame_age_second_boundaries; D-Q11 strata [0,5] and (5,10]
FRAME_AGE_ORDER = ("[0,5]", "(5,10]")
ENDPOINT_ORDER = ("full90", "truncated")  # FSP 202, ARCHITECTURE 5.2
KILL_COUNT_ORDER = ("1", "2-3", "4+")  # D-Q17: exploratory scale strata, added before any q result
CREDIT_ORDER = ("1", "2-3", "4+")  # D-Q17: first-kill credit on the scoring side (killer + assists)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261011
PRIMARY_LEVEL = 0.975  # P1, P2
EXPLORATORY_LEVEL = 0.95
P2_WEIGHT = 0.5  # MAT quads weight per D
P2_QUADS = {
    d: {f"G{g}_S{s}": f"N_G{g}_S{s}_A0_M0_E1_C1_R1_D{d}" for g in (0, 1) for s in (0, 1)} for d in (0, 1)
}  # CARTESIAN_COMPARISON_MATRIX.json primary_raw_Brier_G_by_S_contrast.quads
METRIC_KEYS = ("brier", "logloss", "auc", "accuracy", "calibration_slope", "calibration_intercept",
               "calibration_status", "reliability")
_ID_KINDS = ("i", "u", "S", "U")


# ---------------------------------------------------------------- validation and weights

def _labels(y: Any) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    values = values.astype(np.float64)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("labels must be binary 0/1")
    return values


def _probabilities(p: Any, n: int) -> np.ndarray:
    values = np.asarray(p, dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(f"probabilities must have shape ({n},)")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return values


def _match_ids(match_ids: Any, n: int | None = None) -> np.ndarray:
    ids = np.asarray(match_ids)
    if ids.ndim != 1 or ids.size == 0 or (n is not None and ids.shape != (n,)):
        raise ValueError("match ids must be a non-empty 1-D array aligned with the rows")
    if ids.dtype.kind not in _ID_KINDS:
        raise ValueError("match ids must be an integer, str or bytes array")
    return ids


def _explicit_weights(weights: Any, n: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all() or (values < 0.0).any() or not values.sum() > 0.0:
        raise ValueError("weights must be finite, non-negative, aligned with the rows and have a positive sum")
    return values


def _weights(weights: Any, match_ids: Any, n: int) -> np.ndarray:
    if weights is None:
        if match_ids is None:
            raise ValueError("pass match_ids (match weights, the default) or an explicit weights array")
        return match_weights(_match_ids(match_ids, n))
    if match_ids is not None:
        raise ValueError("pass either weights or match_ids, not both")
    return _explicit_weights(weights, n)


def _inputs(y: Any, p: Any, weights: Any, match_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = _labels(y)
    return labels, _probabilities(p, labels.size), _weights(weights, match_ids, labels.size)


def match_weights(match_ids: Any) -> np.ndarray:
    """Raw row weights 1/m_g (EC 569): each match sums to 1, the total is the number of matches."""

    _, inverse, counts = np.unique(_match_ids(match_ids), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def normalized_match_weights(match_ids: Any) -> np.ndarray:
    """Training and calibration weights n_rows/(n_matches*m_g) (EC 570): mean 1, total n_rows."""

    ids = _match_ids(match_ids)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return ids.size / (unique.size * counts[inverse].astype(np.float64))


# ---------------------------------------------------------------- metrics

def _brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def _logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    q = np.clip(p, LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    return float(np.sum(w * -(y * np.log(q) + (1.0 - y) * np.log1p(-q))) / np.sum(w))


def _auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    positive = np.where(y == 1.0, w, 0.0)
    negative = np.where(y == 0.0, w, 0.0)
    total_positive, total_negative = positive.sum(), negative.sum()
    if not (total_positive > 0.0 and total_negative > 0.0):
        return None
    order = np.argsort(p, kind="stable")
    values = p[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    tied_positive = np.add.reduceat(positive[order], starts)
    tied_negative = np.add.reduceat(negative[order], starts)
    below = np.concatenate(([0.0], np.cumsum(tied_negative)[:-1]))
    return float(np.sum(tied_positive * (below + 0.5 * tied_negative)) / (total_positive * total_negative))


def _accuracy(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    predicted = (p >= ACCURACY_THRESHOLD).astype(np.float64)
    return float(np.sum(w * (predicted == y)) / np.sum(w))


def _reliability(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> list[dict[str, Any]]:
    index = np.minimum(np.searchsorted(RELIABILITY_EDGES, p, side="right") - 1, RELIABILITY_BINS - 1)
    rows = np.bincount(index, minlength=RELIABILITY_BINS)
    weight = np.bincount(index, weights=w, minlength=RELIABILITY_BINS)
    weighted_p = np.bincount(index, weights=w * p, minlength=RELIABILITY_BINS)
    weighted_y = np.bincount(index, weights=w * y, minlength=RELIABILITY_BINS)
    table = []
    for k in range(RELIABILITY_BINS):
        closing = "]" if k == RELIABILITY_BINS - 1 else ")"
        defined = weight[k] > 0.0
        table.append({"bin": k, "interval": f"[{RELIABILITY_EDGES[k]:.1f},{RELIABILITY_EDGES[k + 1]:.1f}{closing}",
                      "rows": int(rows[k]), "weight_sum": float(weight[k]),
                      "mean_p": float(weighted_p[k] / weight[k]) if defined else None,
                      "mean_y": float(weighted_y[k] / weight[k]) if defined else None})
    return table


def logit_clipped(p: Any, clip: float = LOGIT_CLIP) -> np.ndarray:
    """logit(clip(p, clip, 1 - clip)) in float64."""

    q = np.clip(np.asarray(p, dtype=np.float64), clip, 1.0 - clip)
    return np.log(q) - np.log1p(-q)


def separation_status(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> str | None:
    """Why the MLE of sigmoid(a*x + b) does not exist or is not unique, else None (rows with w > 0)."""

    keep = w > 0.0
    x1, x0 = x[keep & (y == 1.0)], x[keep & (y == 0.0)]
    if not x1.size or not x0.size:
        return "one_class"
    if x[keep].min() == x[keep].max():
        return "constant_input"
    if x0.max() < x1.min() or x1.max() < x0.min():
        return "complete_separation"
    if x0.max() <= x1.min() or x1.max() <= x0.min():
        return "quasi_complete_separation"
    return None


def fit_logit_regression(x: Any, y: Any, w: Any) -> dict[str, Any]:
    """Unpenalized MLE of P(Y=1) = sigmoid(a*x + b): minimize mean(w * BCE) with the cards' L-BFGS-B settings.

    The weights are used as given (their scale only moves the optimizer's stopping point, never the minimizer).
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = theta[0] * x + theta[1]
        residual = w * (expit(z) - y)
        return float(np.mean(w * (np.logaddexp(0.0, z) - y * z))), np.asarray([np.mean(residual * x), np.mean(residual)])

    result = minimize(objective, np.asarray(SOLVER["x0"], dtype=np.float64), jac=True, method=SOLVER["method"],
                      options={"maxiter": SOLVER["maxiter"], "ftol": SOLVER["ftol"], "gtol": SOLVER["gtol"]})
    gradient = objective(np.asarray(result.x, dtype=np.float64))[1]
    return {"a": float(result.x[0]), "b": float(result.x[1]), "fun": float(result.fun), "success": bool(result.success),
            "message": str(result.message), "nit": int(result.nit), "nfev": int(result.nfev),
            "grad_max_abs": float(np.max(np.abs(gradient)))}


def _slope_intercept(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    x = logit_clipped(p)
    status = separation_status(x, y, w)
    if status is not None:
        return {"slope": None, "intercept": None, "status": status}
    fit = fit_logit_regression(x, y, w * (w.size / w.sum()))  # weights rescaled to mean 1, as the calibrator
    by_kkt = not fit["success"] and kkt_accepted(fit)
    if not ((fit["success"] or by_kkt) and math.isfinite(fit["a"]) and math.isfinite(fit["b"])):
        return {"slope": None, "intercept": None, "status": "optimizer_failed", "fit": fit}
    return {"slope": fit["a"], "intercept": fit["b"], "status": "fitted", "accepted_by_kkt": by_kkt, "fit": fit}


KKT_TOL = 1e-6


def kkt_accepted(fit: Mapping[str, Any]) -> bool:
    """A fit scipy did not flag successful is accepted when it is stationary: max |gradient| <= 1e-6.

    scipy's L-BFGS-B can end with ABNORMAL_TERMINATION_IN_LNSRCH at a converged point (1 of 300 synthetic
    Q_CAL-sized fits, gradient 4.7e-10; MEMO_eval_KO.md B8).  The author adopted the stationarity test,
    the same threshold as the V calibrator's KKT check, before any q result (DECISIONS D-Q12).
    """

    grad = fit.get("grad_max_abs")
    return grad is not None and math.isfinite(grad) and grad <= KKT_TOL and all(math.isfinite(fit[k]) for k in ("a", "b", "fun"))


def brier(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> float:
    """sum w (p - y)^2 / sum w; with match weights = (1/M) sum_g (1/m_g) sum_j (p - y)^2 (EC 572)."""

    return _brier(*_inputs(y, p, weights, match_ids))


def logloss(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> float:
    """Weighted binary cross-entropy on clip(p, 1e-15, 1 - 1e-15)."""

    return _logloss(*_inputs(y, p, weights, match_ids))


def auc(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> float:
    """Weighted Mann-Whitney AUC, ties count 1/2 (D-Q11); raises when a class has zero weight."""

    value = _auc(*_inputs(y, p, weights, match_ids))
    if value is None:
        raise ValueError("AUC is undefined: a class has zero weight")
    return value


def accuracy(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> float:
    """Weighted accuracy of the rule p >= 0.5 -> 1 (D-Q11)."""

    return _accuracy(*_inputs(y, p, weights, match_ids))


def reliability_table(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> list[dict[str, Any]]:
    """10 fixed equal-width bins [k/10, (k+1)/10), the last one [0.9, 1.0] closed (EC 1043, D-Q11)."""

    return _reliability(*_inputs(y, p, weights, match_ids))


def calibration_slope_intercept(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> dict[str, Any]:
    """Weighted logistic regression of Y on logit(clip(p, 1e-6, 1 - 1e-6)) (D-Q11); None when the MLE fails."""

    return _slope_intercept(*_inputs(y, p, weights, match_ids))


def _summary(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    calibration = _slope_intercept(y, p, w)
    return {"brier": _brier(y, p, w), "logloss": _logloss(y, p, w), "auc": _auc(y, p, w), "accuracy": _accuracy(y, p, w),
            "calibration_slope": calibration["slope"], "calibration_intercept": calibration["intercept"],
            "calibration_status": calibration["status"], "reliability": _reliability(y, p, w)}


def summarize(y: Any, p: Any, weights: Any = None, *, match_ids: Any = None) -> dict[str, Any]:
    """All secondary metrics (EC 1039-1046); None where a metric is undefined."""

    return _summary(*_inputs(y, p, weights, match_ids))


def evaluate(y: Any, p: Any, match_ids: Any) -> dict[str, Any]:
    """Counts plus match-weighted metrics and the unit-row sensitivity (EC 1044)."""

    labels = _labels(y)
    probabilities = _probabilities(p, labels.size)
    ids = _match_ids(match_ids, labels.size)
    positives = int(labels.sum())
    return {"rows": int(labels.size), "matches": int(np.unique(ids).size), "positives": positives,
            "negatives": int(labels.size) - positives,
            "match_weighted": _summary(labels, probabilities, match_weights(ids)),
            "unit_row": _summary(labels, probabilities, np.ones(labels.size))}


# ---------------------------------------------------------------- strata

def metrics_by_stratum(y: Any, p: Any, weights: Any, strata_labels: Any, *, match_ids: Any, min_rows: int | None = None,
                       min_class: int | None = None, order: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Counts for every stratum and metrics where the stratum passes the thresholds.

    ``weights=None`` recomputes the match weights 1/m_g inside each stratum, so every match present in
    the stratum weighs equally there: "mean over matches of the mean case loss" applied to the stratum's
    sub-population (DECISIONS D-Q13, author 2026-10-05).  Explicit ``weights`` are used as given.
    Metrics are all None when rows < min_rows or the smaller class has < min_class rows; the defaults are
    the D-V1 thresholds 100 and 10, applied to q strata as well (D-Q14).  Metrics are individually None
    where undefined.  ``order`` fixes the output order and reports empty strata.
    """

    min_rows = DV1_MIN_ROWS if min_rows is None else int(min_rows)
    min_class = DV1_MIN_CLASS if min_class is None else int(min_class)

    labels = _labels(y)
    n = labels.size
    probabilities = _probabilities(p, n)
    ids = _match_ids(match_ids, n)
    explicit = None if weights is None else _explicit_weights(weights, n)
    strata = np.asarray(strata_labels).astype(str)
    if strata.shape != (n,):
        raise ValueError("strata labels must be aligned with the rows")
    present = [str(name) for name in np.unique(strata)]
    names = present if order is None else [str(name) for name in order]
    if len(set(names)) != len(names):
        raise ValueError("duplicate stratum names in order")
    unknown = sorted(set(present) - set(names))
    if unknown:
        raise ValueError(f"strata missing from order: {unknown}")
    table: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = strata == name
        count = int(rows.sum())
        positives = int(labels[rows].sum())
        w = np.zeros(n)
        if count:
            w[rows] = match_weights(ids[rows]) if explicit is None else explicit[rows]
        entry: dict[str, Any] = {"rows": count, "matches": int(np.unique(ids[rows]).size), "positives": positives,
                                 "negatives": count - positives, "weight_sum": float(w[rows].sum())}
        reasons = []
        if count == 0:
            reasons.append("no_rows")
        if count < min_rows:
            reasons.append(f"rows<{min_rows}")
        if min(positives, count - positives) < min_class:
            reasons.append(f"class<{min_class}")
        if count and not entry["weight_sum"] > 0.0:
            reasons.append("zero_weight")
        entry["suppressed"] = reasons
        entry["metrics"] = dict.fromkeys(METRIC_KEYS) if reasons else _summary(labels[rows], probabilities[rows], w[rows])
        table[name] = entry
    return table


def _nonnegative(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must be a finite, non-negative 1-D array")
    return array


def game_time_strata(elapsed_minutes: Any) -> np.ndarray:
    """[0,15), [15,25), [25,inf) minutes (EC 1048-1051, FSP 202)."""

    t = _nonnegative(elapsed_minutes, "elapsed minutes")
    first, second = TIME_MINUTE_BOUNDARIES
    return np.asarray(GAME_TIME_ORDER)[np.where(t < first, 0, np.where(t < second, 1, 2))]


def minute_strata(elapsed_minutes: Any) -> np.ndarray:
    """D-V1 bins [2,3), ..., [39,40), [40,inf); rows below 2 minutes get their own label "[0,2)"."""

    t = _nonnegative(elapsed_minutes, "elapsed minutes")
    middle = np.floor(np.minimum(t, MINUTE_LAST)).astype(np.int64) - MINUTE_FIRST + 1
    index = np.where(t < MINUTE_FIRST, 0, np.where(t >= MINUTE_LAST, len(MINUTE_ORDER) - 1, middle))
    return np.asarray(MINUTE_ORDER)[index]


def balanced_p_strata(p_pre: Any) -> np.ndarray:
    """p_pre in the closed interval [0.4, 0.6] versus outside (EC 1052-1055)."""

    p = _nonnegative(p_pre, "p_pre")
    if (p > 1.0).any():
        raise ValueError("p_pre must be in [0, 1]")
    low, high = BALANCED_P
    return np.where((p >= low) & (p <= high), BALANCED_ORDER[0], BALANCED_ORDER[1])


def frame_age_strata(frame_age_seconds: Any) -> np.ndarray:
    """[0,5] and (5,10] seconds (D-Q11); ages outside [0, 10] are outside the population and raise."""

    age = _nonnegative(frame_age_seconds, "frame age")
    if (age > FRAME_AGE_SECONDS[-1]).any():
        raise ValueError("frame age above 10 s is outside the population (PSL time.pre_frame_age_ms)")
    return np.where(age <= FRAME_AGE_SECONDS[1], FRAME_AGE_ORDER[0], FRAME_AGE_ORDER[1])


def endpoint_strata(truncated: Any) -> np.ndarray:
    """Full 90 s follow-up versus a truncated endpoint (FSP 202)."""

    flags = np.asarray(truncated)
    if flags.ndim != 1 or flags.dtype != np.bool_:
        raise ValueError("truncated must be a 1-D boolean array")
    return np.where(flags, ENDPOINT_ORDER[1], ENDPOINT_ORDER[0])


def kill_count_strata(n_kills: Any) -> np.ndarray:
    """Kills in the engagement: 1 (pick-off), 2-3, 4+ (D-Q17; evaluation strata only, never a q input)."""

    k = np.asarray(n_kills)
    if k.ndim != 1 or k.dtype.kind not in "iu" or (k < 1).any():
        raise ValueError("kill counts must be positive integers")
    return np.asarray(KILL_COUNT_ORDER)[np.where(k == 1, 0, np.where(k <= 3, 1, 2))]


def credit_strata(credit_blue: Any, credit_red: Any) -> np.ndarray:
    """First-kill credit on the larger side (killer + assists): 1, 2-3, 4+ (D-Q17; evaluation strata only).

    The smaller side is the victim's team and is 1 in 99.8% of Q_DEV rows, so it cannot separate scales."""

    blue, red = np.asarray(credit_blue), np.asarray(credit_red)
    if blue.shape != red.shape or blue.ndim != 1 or blue.dtype.kind not in "iu" or red.dtype.kind not in "iu":
        raise ValueError("credit counts must be aligned 1-D integer arrays")
    if (np.minimum(blue, red) < 1).any():
        raise ValueError("every engagement has first-kill credit on both teams (opposition evidence)")
    larger = np.maximum(blue, red)
    return np.asarray(CREDIT_ORDER)[np.where(larger == 1, 0, np.where(larger <= 3, 1, 2))]


# ---------------------------------------------------------------- per-match losses and contrasts

def per_match_loss(y: Any, p: Any, match_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    """Ascending unique match ids and each match's mean Brier; their mean is the match-weighted Brier."""

    labels = _labels(y)
    probabilities = _probabilities(p, labels.size)
    unique, inverse, counts = np.unique(_match_ids(match_ids, labels.size), return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=(probabilities - labels) ** 2, minlength=unique.size)
    return unique, sums / counts


def _ascending_ids(match_ids: Any) -> np.ndarray:
    ids = _match_ids(match_ids)
    if ids.size > 1 and not (ids[1:] > ids[:-1]).all():
        raise ValueError("match ids must be unique and ascending (np.unique order)")
    return ids


def align_cells(cells: Mapping[str, tuple[Any, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Check that every cell's (match ids, per-match losses) covers the identical match set."""

    if not cells:
        raise ValueError("no cells")
    reference_name, reference = None, None
    aligned: dict[str, np.ndarray] = {}
    for name, (ids, losses) in cells.items():
        ids = _ascending_ids(ids)
        values = np.asarray(losses, dtype=np.float64)
        if values.shape != ids.shape or not np.isfinite(values).all():
            raise ValueError(f"cell {name!r}: losses must be finite and aligned with its match ids")
        if reference is None:
            reference_name, reference = name, ids
        elif ids.shape != reference.shape or not np.array_equal(ids, reference):
            raise ValueError(f"cell {name!r} covers a different match set than {reference_name!r}")
        aligned[str(name)] = values
    return reference, aligned


def difference(minuend: str, subtrahend: str) -> dict[str, float]:
    """Coefficients of minuend - subtrahend."""

    if minuend == subtrahend:
        raise ValueError("a difference needs two different cells")
    return {minuend: 1.0, subtrahend: -1.0}


def _combine(terms: Sequence[tuple[float, Mapping[str, float]]]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for weight, contrast in terms:
        for cell, coefficient in contrast.items():
            combined[cell] = combined.get(cell, 0.0) + weight * coefficient
    return combined


def difference_of_differences(a1: str, b1: str, a0: str, b0: str) -> dict[str, float]:
    """Coefficients of (a1 - b1) - (a0 - b0)."""

    return _combine([(1.0, difference(a1, b1)), (-1.0, difference(a0, b0))])


def p1_contrast(selected_full_input: str, pt_flex: str = "PT_flex") -> dict[str, float]:
    """P1 = Brier(selected full-input q) - Brier(PT_flex) (EC 1019-1022, FSP 196)."""

    return difference(selected_full_input, pt_flex)


def p2_contrast(quads: Mapping[int, Mapping[str, str]] = P2_QUADS) -> dict[str, float]:
    """P2 = mean over D in {0,1} of (G1S1 - G0S1) - (G1S0 - G0S0) at E=C=R=1, A=M=0 (EC 1023-1026, MAT)."""

    if set(quads) != {0, 1}:
        raise ValueError("P2 needs the D=0 and D=1 quads")
    return _combine([(P2_WEIGHT, difference_of_differences(quads[d]["G1_S1"], quads[d]["G0_S1"],
                                                           quads[d]["G1_S0"], quads[d]["G0_S0"])) for d in (0, 1)])


def contrast_per_match(cell_losses: Mapping[str, Any], contrast: Mapping[str, float]) -> np.ndarray:
    """sum_c coefficient_c * per-match losses_c (the per-match contrast vector)."""

    total = None
    for cell, coefficient in contrast.items():
        term = float(coefficient) * np.asarray(cell_losses[cell], dtype=np.float64)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("empty contrast")
    return total


def contrast_point(cell_losses: Mapping[str, Any], contrast: Mapping[str, float]) -> float:
    """sum_c coefficient_c * mean_g L_c[g], the full-sample contrast of match-weighted Briers."""

    if not contrast:
        raise ValueError("empty contrast")
    point = 0.0
    for cell, coefficient in contrast.items():
        point += float(coefficient) * float(np.mean(np.asarray(cell_losses[cell], dtype=np.float64)))
    return point


# ---------------------------------------------------------------- paired match bootstrap

def bootstrap_quantiles(level: float) -> tuple[float, float]:
    """Percentile-interval quantiles: 0.975 -> (0.0125, 0.9875), 0.95 -> (0.025, 0.975)."""

    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("interval level must be in (0, 1)")
    lower = round((1.0 - level) / 2.0, 12)
    return lower, round(1.0 - lower, 12)


def bootstrap_indices(n_matches: int, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> Iterator[np.ndarray]:
    """Draws b = 0, 1, ...: n_matches positions with replacement from one PCG64(seed) stream, in this order.

    The stream equals ``Generator(PCG64(seed)).integers(0, M, size=(resamples, M))`` row by row, without
    allocating that matrix.
    """

    if int(n_matches) < 1 or int(resamples) < 1:
        raise ValueError("need at least one match and one resample")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return (generator.integers(0, int(n_matches), size=int(n_matches)) for _ in range(int(resamples)))


def paired_match_bootstrap(match_ids: Any, cell_losses: Mapping[str, Any], contrasts: Mapping[str, Mapping[str, float]],
                           *, levels: float | Mapping[str, float] = PRIMARY_LEVEL,
                           resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                           return_draws: bool = False) -> dict[str, Any]:
    """Paired match bootstrap with draws shared by every cell (EC 1028-1036, FSP 198, D-Q11).

    ``match_ids`` are the ascending ids shared by all cells; ``cell_losses[c]`` holds cell c's per-match
    Brier losses in that order (``per_match_loss``/``align_cells``).  Draw b resamples positions once and
    every cell's resampled Brier is the mean of its losses at those positions; each contrast draw is the
    coefficient-weighted sum of cell draws.  Point = full-sample contrast; interval = percentile at
    ``levels`` (one level or one per contrast).  Memory is O(cells * M): no resamples x M matrix.
    """

    ids = _ascending_ids(match_ids)
    m = ids.size
    if not cell_losses:
        raise ValueError("no cells")
    names = [str(name) for name in cell_losses]
    losses = np.empty((len(names), m), dtype=np.float64)
    for row, name in enumerate(cell_losses):
        values = np.asarray(cell_losses[name], dtype=np.float64)
        if values.shape != (m,) or not np.isfinite(values).all():
            raise ValueError(f"cell {name!r}: per-match losses must be finite and aligned with match_ids")
        losses[row] = values
    column = {name: row for row, name in enumerate(names)}
    plan: dict[str, tuple[dict[str, float], float]] = {}
    for contrast_name, contrast in contrasts.items():
        if not contrast:
            raise ValueError(f"contrast {contrast_name!r} is empty")
        coefficients = {}
        for cell, coefficient in contrast.items():
            if cell not in column:
                raise ValueError(f"contrast {contrast_name!r} uses unknown cell {cell!r}")
            coefficients[cell] = float(coefficient)
            if not math.isfinite(coefficients[cell]):
                raise ValueError(f"contrast {contrast_name!r}: non-finite coefficient")
        if isinstance(levels, Mapping):
            if contrast_name not in levels:
                raise ValueError(f"no interval level for contrast {contrast_name!r}")
            level = float(levels[contrast_name])
        else:
            level = float(levels)
        bootstrap_quantiles(level)
        plan[str(contrast_name)] = (coefficients, level)

    cell_draws = np.empty((int(resamples), len(names)), dtype=np.float64)
    buffer = np.empty_like(losses)
    for b, positions in enumerate(bootstrap_indices(m, resamples, seed)):
        np.take(losses, positions, axis=1, out=buffer, mode="clip")
        cell_draws[b] = buffer.mean(axis=1)

    by_cell = {name: losses[row] for row, name in enumerate(names)}
    results: dict[str, Any] = {}
    for contrast_name, (coefficients, level) in plan.items():
        draws = np.zeros(int(resamples), dtype=np.float64)
        for cell, coefficient in coefficients.items():
            draws += coefficient * cell_draws[:, column[cell]]
        quantiles = bootstrap_quantiles(level)
        low, high = np.quantile(draws, quantiles)
        results[contrast_name] = {"coefficients": coefficients, "point": contrast_point(by_cell, coefficients),
                                  "level": level, "quantiles": list(quantiles), "interval": [float(low), float(high)]}
        if return_draws:
            results[contrast_name]["draws"] = draws
    out: dict[str, Any] = {
        "version": VERSION, "unit": "match", "paired": True, "shared_draws": True, "resamples": int(resamples),
        "seed": int(seed), "generator": "numpy.random.Generator(numpy.random.PCG64(seed)).integers(0, M, size=M) per draw",
        "numpy_version": np.__version__, "matches": int(m), "match_order": "ascending match id",
        "interval_method": "percentile; numpy.quantile method='linear'; point = full-sample contrast",
        "cells": {name: {"point": float(np.mean(losses[row]))} for row, name in enumerate(names)},
        "contrasts": results,
    }
    if return_draws:
        out["cell_names"] = names
        out["cell_draws"] = cell_draws
    return out
