"""q references (EXECUTION_CARDS.json ``q.references``; DECISIONS D-Q8).

  - constant: TRAIN match-weighted positive rate; never calibrated;
  - time_only: elapsed minutes (pre_ms / 60000), standardized; logistic over C in {0.001, 0.01, 0.1, 1};
  - p_only: logit(clip(p_pre, 1e-6)), standardized; same C grid;
  - PT_linear: standardized logit p and time with intercept; same C grid;
  - PT_flex: cubic B-spline bases (scikit-learn SplineTransformer, knots at TRAIN unit-row quantiles,
    linear extrapolation, no bias column) of logit p (4 / 6 / 8 knots) and time (4 knots), every
    univariate basis plus the full p x time tensor product, columns standardized by TRAIN mean/sd,
    logistic with intercept; duplicate or degenerate knots fail the card.
Logistic-type references are fitted once on all TRAIN with normalized match weights (no STOP).
p_pre is the label's calibrated pre probability (OOF V for TRAIN rows); only these references read it.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import sparse

CLIP = 1e-6


def logit_p(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    return np.log(p) - np.log1p(-p)


class Standardizer:
    def __init__(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.sd


class Design:
    """Feature map of one reference fitted on TRAIN rows."""

    def __init__(self, name: str, card: Mapping[str, Any], p_train: np.ndarray, t_train: np.ndarray) -> None:
        self.name, self.card = name, dict(card)
        z, t = logit_p(p_train), np.asarray(t_train, dtype=np.float64)
        if name == "PT_flex":
            from sklearn.preprocessing import SplineTransformer

            self.spline_p = SplineTransformer(n_knots=int(card["p_knots"]), degree=int(card["degree"]), knots="quantile",
                                              extrapolation="linear", include_bias=False).fit(z.reshape(-1, 1))
            self.spline_t = SplineTransformer(n_knots=int(card["time_knots"]), degree=int(card["degree"]), knots="quantile",
                                              extrapolation="linear", include_bias=False).fit(t.reshape(-1, 1))
            for spline in (self.spline_p, self.spline_t):
                knots = np.asarray(spline.bsplines_[0].t)
                interior = knots[spline.degree:len(knots) - spline.degree]
                if np.any(np.diff(interior) <= 0):
                    raise ValueError("duplicate or degenerate spline knots")
        self.scale = Standardizer(self._raw(z, t))

    def _raw(self, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.name == "time_only":
            return t.reshape(-1, 1)
        if self.name == "p_only":
            return z.reshape(-1, 1)
        if self.name == "PT_linear":
            return np.column_stack((z, t))
        if self.name == "PT_flex":
            bp, bt = self.spline_p.transform(z.reshape(-1, 1)), self.spline_t.transform(t.reshape(-1, 1))
            tensor = (bp[:, :, None] * bt[:, None, :]).reshape(len(z), -1)
            return np.column_stack((bp, bt, tensor))
        raise ValueError(f"unknown reference {self.name!r}")

    def __call__(self, p: np.ndarray, t: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.scale(self._raw(logit_p(p), np.asarray(t, dtype=np.float64))))


def constant_rate(y_train: np.ndarray, w_raw_train: np.ndarray) -> float:
    y, w = np.asarray(y_train, dtype=np.float64), np.asarray(w_raw_train, dtype=np.float64)
    return float((w * y).sum() / w.sum())


def fit_reference(name: str, card: Mapping[str, Any], train: Mapping[str, np.ndarray]) -> tuple[Any, Design]:
    """Fit one logistic-type reference on all TRAIN rows; raises LogisticNonconvergenceError when it does not converge."""

    from studies.v_model_fit_input_20261003.models import LogisticAdapter

    design = Design(name, card, train["p"], train["t"])
    model = LogisticAdapter(C=float(card["C"]), card=dict(card))
    model.fit(design(train["p"], train["t"]), np.asarray(train["y"], dtype=np.float64), sample_weight=train["w_norm"])
    return model, design
