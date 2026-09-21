#!/usr/bin/env python3
"""Shared forecast diagnostics for review-response (CORP + triad).

CORP Brier decomposition (Dimitriadis–Gneiting–Jordan 2021):
  Fit isotonic recalibration p* = iso(p) on the *evaluation* sample for DIAGNOSIS only.
  With match weights ⟨·⟩_w and climatology ȳ_w:

    BS      = ⟨(p − y)²⟩_w
    BS_iso  = ⟨(p* − y)²⟩_w
    UNC     = ⟨(ȳ_w − y)²⟩_w          (= Brier of constant ȳ_w)
    MCB     = BS − BS_iso             (score gap vs isotonic recalibration)
    DSC     = UNC − BS_iso            (discrimination of recalibrated probs)

  Exact identity: BS = MCB − DSC + UNC.
  Do NOT use ⟨(p − p*)²⟩ as MCB — that drops the cross term and breaks the identity.

Never apply TEST-fit iso curve to claim improved frozen-model transfer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return weighted_mean((np.asarray(p, float) - np.asarray(y, float)) ** 2, w)


def logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    eps = 1e-15
    pc = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return weighted_mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), w)


def auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, float)
    if y.min() == y.max():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        return float("nan")


def corp_brier_decomposition(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """CORP Brier decomposition via isotonic recalibration (diagnostic only)."""
    from sklearn.isotonic import IsotonicRegression

    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, float)
    bs = brier(y, p, w)
    ybar = weighted_mean(y, w)
    # Isotonic: nondecreasing map p -> E[y|p]; sample_weight required (fail closed)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p, y, sample_weight=w)
    p_star = np.clip(iso.predict(p), 0.0, 1.0)
    bs_iso = brier(y, p_star, w)
    unc = brier(y, np.full_like(y, ybar, dtype=float), w)
    mcb = float(bs - bs_iso)
    dsc = float(unc - bs_iso)
    recon = mcb - dsc + unc
    gap = float(bs - recon)
    if not np.isclose(bs, recon, atol=1e-12, rtol=0.0):
        raise AssertionError(
            f"CORP identity failed: BS={bs:.16g} recon={recon:.16g} gap={gap:.16g} "
            "(MCB/DSC must be score gaps, not ⟨(p−p*)²⟩ / ⟨(p*−ȳ)²⟩)"
        )
    # optional: squared-distance MCB for diagnostics of the cross-term (not CORP MCB)
    mcb_sqdist = weighted_mean((p - p_star) ** 2, w)
    # reliability curve knots from isotonic
    order = np.argsort(p)
    p_sorted = p[order]
    ps_sorted = p_star[order]
    xs, ys, ns = [], [], []
    i = 0
    n = len(p_sorted)
    while i < n:
        j = i + 1
        while j < n and abs(ps_sorted[j] - ps_sorted[i]) < 1e-15:
            j += 1
        xs.append(float(np.average(p_sorted[i:j], weights=w[order][i:j])))
        ys.append(float(ps_sorted[i]))
        ns.append(int(j - i))
        i = j
    return dict(
        n=int(len(y)),
        brier=bs,
        brier_iso=bs_iso,
        UNC=unc,
        DSC=dsc,
        MCB=mcb,
        MCB_sqdist_not_CORP=mcb_sqdist,
        reconstructed_brier=recon,
        reconstruction_gap=gap,
        mean_y=ybar,
        mean_p=weighted_mean(p, w),
        reliability_curve=dict(p_hat=xs, p_star=ys, n=ns),
        note=(
            "CORP: MCB=BS−BS_iso, DSC=UNC−BS_iso; exact BS=MCB−DSC+UNC. "
            "Evaluation-sample isotonic — diagnostic only; do not use as frozen-model fix."
        ),
    )

def fixed_bin_reliability(
    y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None, n_bins: int = 10
) -> List[Dict[str, Any]]:
    """Classic equal-width reliability (secondary; CORP is preferred).

    Bin means are match-weighted; also stores ``w_sum`` for match-weighted ECE.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if w is None:
        w = np.ones(len(y), float)
    else:
        w = np.asarray(w, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            rows.append(
                dict(bin=i, lo=lo, hi=hi, n=0, w_sum=0.0, mean_p=float("nan"), mean_y=float("nan"))
            )
            continue
        rows.append(
            dict(
                bin=i,
                lo=float(lo),
                hi=float(hi),
                n=int(m.sum()),
                w_sum=float(np.sum(w[m])),
                mean_p=weighted_mean(p[m], w[m]),
                mean_y=weighted_mean(y[m], w[m]),
            )
        )
    return rows

def calib_logit_slope_intercept(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Diagnostic logit(y) ~ a + b logit(p). Not applied to rewrite predictions."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if w is None:
        w = np.ones(len(y), float)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    try:
        lr.fit(logit, y, sample_weight=w)
        return dict(ok=True, intercept=float(lr.intercept_[0]), slope=float(lr.coef_.ravel()[0]))
    except Exception as e:
        return dict(ok=False, error=str(e))


def triad_by_p_bins(
    delta: np.ndarray,
    p_pre: np.ndarray,
    g: np.ndarray,
    edges: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Christoffersen–Diebold style triad: direction rate, mean change, scale."""
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    delta = np.asarray(delta, float)
    p_pre = np.asarray(p_pre, float)
    g = np.asarray(g)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (p_pre >= lo) & (p_pre < hi) if i < len(edges) - 2 else (p_pre >= lo) & (p_pre <= hi)
        if not m.any():
            rows.append(dict(lo=lo, hi=hi, n=0))
            continue
        w = match_weights(g[m])
        d = delta[m]
        rows.append(
            dict(
                lo=lo,
                hi=hi,
                n=int(m.sum()),
                n_matches=int(len(np.unique(g[m].astype(str)))),
                P_delta_gt0=weighted_mean((d > 0).astype(float), w),
                E_delta=weighted_mean(d, w),
                E_abs_delta=weighted_mean(np.abs(d), w),
                median_abs_delta=float(np.median(np.abs(d))),
            )
        )
    return rows


def metrics_bundle(y, p, g) -> Dict[str, Any]:
    w = match_weights(g)
    return dict(
        n=int(len(y)),
        n_matches=int(len(np.unique(np.asarray(g).astype(str)))),
        brier=brier(y, p, w),
        logloss=logloss(y, p, w),
        auc=auc(y, p, w),
        corp=corp_brier_decomposition(y, p, w),
        calib_logit=calib_logit_slope_intercept(y, p, w),
        fixed_bin_reliability=fixed_bin_reliability(y, p, w, 10),
    )
