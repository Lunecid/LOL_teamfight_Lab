"""q calibrator on Q_CAL (EXECUTION_CARDS.json calibration.q 989-998, solver 999-1012, on_failure 1013;
FULL_STUDY_PROTOCOL.md 192; ARCHITECTURE_KO.md 5.2; DECISIONS_20261005.json D-Q11).

p_cal = sigmoid(a * logit(clip(p_raw, 1e-6, 1 - 1e-6)) + b), with (a, b) the unpenalized maximum-likelihood
estimate under the match-weighted binary cross-entropy: minimize mean_i(w_i * BCE_i) with the normalized
match weights w_i = n_rows / (n_matches * m_g(i)) of the Q_CAL rows passed (EC 570 and 991; mean 1).  a is
unconstrained (no V-style lower bound), b is free, no penalty; L-BFGS-B from (1, 0), maxiter 1000,
ftol 1e-15, gtol 1e-10, other options at the scipy defaults.

Weights and scale: mean(w_norm * BCE) = (n / M) * mean(w_raw * BCE) = (1/M) sum_g mean_j BCE_gj, so the raw
weights 1/m_g, the normalized weights, or their sum instead of the mean all have the same minimizer.  Only
the stopping tests see the scale: ftol tests (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) and gtol bounds the
absolute projected gradient, so the cards' tolerances act on this per-row-mean objective (value below 1).

Gates: optimizer success, finite a, b and loss, both classes (EC 1009-1011); FSP 192 also names complete
separation.  A fit scipy did not flag successful but that is stationary (max |gradient| <= 1e-6) is
accepted and recorded as ``accepted_by_kkt`` (DECISIONS D-Q12, author 2026-10-05, before any q result).  Complete and quasi-complete separation (the MLE does not exist in either) fail before the
optimizer runs.  A failed record keeps its diagnostics, its predict() raises, and no other calibrator
replaces it: only raw metrics are reported (EC 1013, D-Q11).  The constant reference model is never
calibrated (EC 866, 997).  Constant raw input (slope not identifiable) is flagged, not failed (memo).
Invalid inputs (misaligned, non-finite, outside [0, 1], non-binary labels) raise ValueError.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import math
from typing import Any

import numpy as np
from scipy.special import expit

from studies.q_production_20261005 import qeval

VERSION = "q_calibrator_v1_20261005"
CLIP = 1e-6  # EC calibration.q.clip
FORMULA = "sigmoid(a*logit(clip(p_raw,1e-6,1-1e-6))+b)"
WEIGHTS = "normalized match weights n_rows/(n_matches*m_g) over the Q_CAL rows passed (EC 570, 991)"
FAILING_SEPARATION = ("one_class", "complete_separation", "quasi_complete_separation")


@dataclass(frozen=True)
class QCalibrator:
    """A fitted, failed or not-calibrated (constant reference) q calibrator record."""

    status: str  # "fitted" | "failed" | "not_calibrated"
    a: float | None
    b: float | None
    fun: float | None  # objective at the solution = Q_CAL match-weighted calibrated BCE
    success: bool | None  # scipy OptimizeResult.success; None when the optimizer was not run
    message: str
    nit: int | None
    nfev: int | None
    grad_max_abs: float | None  # max |gradient| of the objective at (a, b), reported, not a gate
    class_counts: Mapping[str, int]
    rows: int
    matches: int
    weight_sum: float
    reasons: tuple[str, ...]
    input_constant: bool
    clipped_low: int
    clipped_high: int
    accepted_by_kkt: bool = False  # scipy not successful but stationary (max |grad| <= 1e-6), D-Q12
    weights: str = WEIGHTS
    formula: str = FORMULA
    clip: float = CLIP
    solver: Mapping[str, Any] = field(default_factory=lambda: dict(qeval.SOLVER))
    version: str = VERSION

    @property
    def fitted(self) -> bool:
        return self.status == "fitted"

    def predict(self, p_raw: Any) -> np.ndarray:
        """Calibrated probabilities; raises unless fitted (no replacement calibrator, EC 1013)."""

        if self.status != "fitted":
            raise RuntimeError(f"q calibrator is {self.status} ({', '.join(self.reasons)}): report raw metrics only, "
                               "no replacement calibrator (EC calibration.on_failure)")
        p = np.asarray(p_raw, dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
            raise ValueError("raw probabilities must be finite and in [0, 1]")
        return expit(self.a * qeval.logit_clipped(p, self.clip) + self.b)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                value = {key: (list(child) if isinstance(child, tuple) else child) for key, child in value.items()}
            elif isinstance(value, tuple):
                value = list(value)
            out[item.name] = value
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QCalibrator":
        names = {item.name for item in fields(cls)}
        missing = sorted(names - set(payload))
        if missing:
            raise ValueError(f"calibrator record is missing fields: {missing}")
        values = {name: payload[name] for name in names}
        values["reasons"] = tuple(values["reasons"])
        values["class_counts"] = dict(values["class_counts"])
        values["solver"] = {key: (tuple(child) if isinstance(child, list) else child) for key, child in values["solver"].items()}
        return cls(**values)


def fit_q_calibrator(p_raw: Any, y: Any, match_ids: Any, *, constant_model: bool = False) -> QCalibrator:
    """Fit the q calibrator of one selected model on its Q_CAL rows; never substitutes another method."""

    labels = qeval._labels(y)
    n = labels.size
    probabilities = qeval._probabilities(p_raw, n)
    ids = qeval._match_ids(match_ids, n)
    weights = qeval.normalized_match_weights(ids)
    x = qeval.logit_clipped(probabilities, CLIP)
    record = {"class_counts": {"0": int((labels == 0.0).sum()), "1": int((labels == 1.0).sum())}, "rows": int(n),
              "matches": int(np.unique(ids).size), "weight_sum": float(weights.sum()),
              "input_constant": bool(x.min() == x.max()), "clipped_low": int((probabilities < CLIP).sum()),
              "clipped_high": int((probabilities > 1.0 - CLIP).sum())}
    not_run = {"a": None, "b": None, "fun": None, "success": None, "nit": None, "nfev": None, "grad_max_abs": None}
    if constant_model:
        return QCalibrator(status="not_calibrated", message="constant reference model is never calibrated (EC 866, 997; FSP 192)",
                           reasons=("constant_model",), **not_run, **record)
    separation = qeval.separation_status(x, labels, weights)
    if separation in FAILING_SEPARATION:
        return QCalibrator(status="failed", message=f"maximum-likelihood estimate does not exist: {separation}",
                           reasons=(separation,), **not_run, **record)
    fit = qeval.fit_logit_regression(x, labels, weights)
    reasons = []
    by_kkt = not fit["success"] and qeval.kkt_accepted(fit)
    if not fit["success"] and not by_kkt:
        reasons.append("optimizer_not_successful")
    if not all(math.isfinite(fit[key]) for key in ("a", "b", "fun")):
        reasons.append("non_finite")
    return QCalibrator(status="failed" if reasons else "fitted", a=fit["a"], b=fit["b"], fun=fit["fun"],
                       success=fit["success"], message=fit["message"], nit=fit["nit"], nfev=fit["nfev"],
                       grad_max_abs=fit["grad_max_abs"], reasons=tuple(reasons), accepted_by_kkt=by_kkt, **record)
