#!/usr/bin/env python3
"""Review-response RR1 + RR2: flexible p/time baselines vs frozen q.

Locks:
  - V / SVI / frozen logit_state weights unchanged
  - New work: b_spline, PT_flex, optional increasing-sigmoid g_q
  - Select baseline settings + identity/calibrated on Q_SELECT (all T Brier)
  - Report all-T || B40 || narrow p_pre bins; H = D_B40 - D_outside

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
QDIR = REPO / "outputs" / "q_newv_fit85_20260920"
OUT = REPO / "outputs" / "review_response_rr12_20260920"
ROLE = "EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y, p, g) -> Dict[str, Any]:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    g = np.asarray(g)
    if len(y) == 0:
        return dict(n=0, n_matches=0, brier=float("nan"), logloss=float("nan"), auc=float("nan"), p_pos=float("nan"))
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        auc = float("nan")
    else:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            auc = float("nan")
    return dict(
        n=int(len(y)),
        n_matches=int(len(np.unique(g.astype(str)))),
        brier=br,
        logloss=ll,
        auc=auc,
        p_pos=float(np.average(y, weights=w)),
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def bootstrap_delta_brier(y, p_a, p_b, g, reps=2000, seed=7):
    """ΔBrier = Brier(a) - Brier(b); negative ⇒ a better."""
    g = np.asarray(g).astype(str)
    y = np.asarray(y).astype(float)
    p_a = np.asarray(p_a).astype(float)
    p_b = np.asarray(p_b).astype(float)
    if len(y) == 0:
        return dict(estimate=float("nan"), ci95=[float("nan"), float("nan")], p_gt0=float("nan"))
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = (1.0 / counts[inv]).astype(np.float64)
    ea = w * (p_a - y) ** 2
    eb = w * (p_b - y) ** 2
    sum_a = np.zeros(n_m)
    sum_b = np.zeros(n_m)
    np.add.at(sum_a, inv, ea)
    np.add.at(sum_b, inv, eb)
    obs = float((sum_a.sum() - sum_b.sum()) / n_m)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        samp = rng.integers(0, n_m, size=n_m)
        draws[i] = float((sum_a[samp].sum() - sum_b[samp].sum()) / n_m)
    return dict(
        estimate=obs,
        ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        p_gt0=float(np.mean(draws > 0)),
    )


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def expanded_pre(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    return X[:, keep].astype(np.float64, copy=False), cols


def load_role_pack(lab_path: Path, D, L, roles):
    lab = np.load(lab_path, allow_pickle=False)
    E = D.load_engagements(L, "MAIN", roles, states=True, counts=False)
    names = list(E["names"])
    e_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(E["match"].astype(str).tolist(), E["s"].astype(np.int64).tolist()))
    }
    idx_e, idx_l = [], []
    for i, (a, b) in enumerate(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist())):
        j = e_key.get((a, int(b)))
        if j is not None:
            idx_l.append(i)
            idx_e.append(j)
    idx_l = np.asarray(idx_l, int)
    idx_e = np.asarray(idx_e, int)
    ok = lab["Y_SVI"][idx_l] >= 0
    if "missing_score" in lab.files:
        ok &= lab["missing_score"][idx_l] == 0
    idx_l, idx_e = idx_l[ok], idx_e[ok]
    X, cols = expanded_pre(E["X_pre"][idx_e], names)
    return dict(
        X=X,
        cols=cols,
        p_pre=lab["p_pre"][idx_l].astype(float),
        delta_V=lab["delta_V"][idx_l].astype(float),
        y=lab["Y_SVI"][idx_l].astype(np.int8),
        g=lab["match"][idx_l].astype(str),
        s=lab["s"][idx_l].astype(np.int64),
        tmin=lab["s"][idx_l].astype(float) / 60000.0,
        B40=lab["B40"][idx_l].astype(bool) if "B40" in lab.files else None,
    )


class PosSlopeSigmoid:
    """Increasing sigmoid on probability logits (g_q only; not g_V)."""

    def __init__(self):
        self.coef_ = 1.0
        self.intercept_ = 0.0
        self.ok = False

    def fit(self, p, y, w):
        from sklearn.linear_model import LogisticRegression

        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(logit, y, sample_weight=w)
        coef = float(lr.coef_.ravel()[0])
        if coef <= 0:
            self.ok = False
            return self
        self.ok, self.coef_, self.intercept_ = True, coef, float(lr.intercept_.ravel()[0])
        return self

    def transform(self, p):
        if not self.ok:
            return np.asarray(p, dtype=float)
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        return 1.0 / (1.0 + np.exp(-(self.coef_ * logit + self.intercept_)))

    def to_dict(self):
        return dict(coef=self.coef_, intercept=self.intercept_, ok=self.ok)


class PTFlexFeatures:
    """f_p(p) + f_t(t) + tensor-product f_pt(p,t) via cubic SplineTransformer."""

    def __init__(self, n_knots_p: int = 6, n_knots_t: int = 4, degree: int = 3):
        self.n_knots_p = int(n_knots_p)
        self.n_knots_t = int(n_knots_t)
        self.degree = int(degree)
        self.sp_p = None
        self.sp_t = None

    def fit(self, X, y=None):
        from sklearn.preprocessing import SplineTransformer

        X = np.asarray(X, dtype=float)
        self.sp_p = SplineTransformer(
            n_knots=self.n_knots_p,
            degree=self.degree,
            include_bias=False,
            extrapolation="constant",
        ).fit(X[:, [0]])
        self.sp_t = SplineTransformer(
            n_knots=self.n_knots_t,
            degree=self.degree,
            include_bias=False,
            extrapolation="constant",
        ).fit(X[:, [1]])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Bp = self.sp_p.transform(X[:, [0]])
        Bt = self.sp_t.transform(X[:, [1]])
        inter = (Bp[:, :, None] * Bt[:, None, :]).reshape(X.shape[0], -1)
        return np.hstack([Bp, Bt, inter])


def fit_pt_flex(X, y, w, n_knots_p: int, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("feat", PTFlexFeatures(n_knots_p=n_knots_p, n_knots_t=4, degree=3)),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=4000)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y, clf__sample_weight=w)
    return pipe


def fit_b_spline(p, y, w, n_knots: int, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    pipe = Pipeline(
        [
            (
                "spl",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="constant",
                ),
            ),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=4000)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(p.reshape(-1, 1), y, clf__sample_weight=w)
    return pipe


def predict_q_frozen(logit_obj, pack):
    pipe = logit_obj["pipe"] if isinstance(logit_obj, dict) else logit_obj.pipe
    num_ix = logit_obj["num_ix"]
    X = np.column_stack([pack["X"][:, num_ix], pack["p_pre"].reshape(-1, 1)])
    return pipe.predict_proba(X)[:, 1]


def p_bin_edges() -> np.ndarray:
    return np.linspace(0.0, 1.0, 21)


def assign_p_bin(p: np.ndarray) -> np.ndarray:
    edges = p_bin_edges()
    # [0,0.05), ..., [0.95,1] with 1.0 in last bin
    idx = np.digitize(p, edges[1:-1], right=False)
    idx = np.clip(idx, 0, 19)
    # design: exactly 0.5 -> [0.50, 0.55) which is bin 10
    return idx


def bin_label(i: int) -> str:
    edges = p_bin_edges()
    lo, hi = edges[i], edges[i + 1]
    if i == 19:
        return f"[{lo:.2f},{hi:.2f}]"
    return f"[{lo:.2f},{hi:.2f})"


def cell_low_support(sc: Dict[str, Any], y: np.ndarray) -> bool:
    if sc["n_matches"] < 200:
        return True
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return pos < 30 or neg < 30


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    for need in (
        LAB / "TRAIN_oof_h90.npz",
        LAB / "Q_CAL_h90.npz",
        LAB / "Q_SELECT_h90.npz",
        LAB / "TEST_h90.npz",
        QDIR / "models" / "logit_state.joblib",
    ):
        if not need.is_file():
            raise SystemExit(f"missing {need}")

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load packs…", flush=True)
    TR = load_role_pack(LAB / "TRAIN_oof_h90.npz", D, L, train_roles)
    CA = load_role_pack(LAB / "Q_CAL_h90.npz", D, L, ["Q_CAL"])
    SE = load_role_pack(LAB / "Q_SELECT_h90.npz", D, L, ["Q_SELECT"])
    TE = load_role_pack(LAB / "TEST_h90.npz", D, L, ["TEST"])
    assert TR["cols"] == CA["cols"] == SE["cols"] == TE["cols"]

    w_tr = match_weights(TR["g"])
    w_ca = match_weights(CA["g"])
    logit_obj = joblib.load(QDIR / "models" / "logit_state.joblib")

    # --- raw predictors ---
    raw_models: Dict[str, Any] = {}
    print("fit constant / b_linear / PT_linear…", flush=True)
    base_rate = float(np.average(TR["y"], weights=w_tr))
    raw_models["constant"] = dict(kind="constant", p=base_rate)

    b_lin = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    b_lin.fit(TR["p_pre"].reshape(-1, 1), TR["y"], logisticregression__sample_weight=w_tr)
    raw_models["b_linear"] = dict(kind="sklearn_p", pipe=b_lin)

    pt_lin = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pt_lin.fit(
        np.column_stack([TR["p_pre"], TR["tmin"]]),
        TR["y"],
        logisticregression__sample_weight=w_tr,
    )
    raw_models["PT_linear"] = dict(kind="sklearn_pt", pipe=pt_lin)

    # b_spline grid → select on Q_SELECT raw Brier
    print("fit b_spline grid…", flush=True)
    b_spline_cands = []
    for nk in (4, 6):
        for Ccoef in (0.01, 0.1, 1.0):
            pipe = fit_b_spline(TR["p_pre"], TR["y"], w_tr, nk, Ccoef)
            pred_se = pipe.predict_proba(SE["p_pre"].reshape(-1, 1))[:, 1]
            sc = metrics(SE["y"], pred_se, SE["g"])
            b_spline_cands.append(dict(n_knots=nk, C=Ccoef, pipe=pipe, select_brier=sc["brier"], select_logloss=sc["logloss"]))
    b_spline_cands.sort(key=lambda d: (d["select_brier"], d["select_logloss"]))
    best_bs = b_spline_cands[0]
    raw_models["b_spline"] = dict(
        kind="sklearn_p",
        pipe=best_bs["pipe"],
        config=dict(n_knots=best_bs["n_knots"], C=best_bs["C"]),
    )
    print(f"  b_spline winner knots={best_bs['n_knots']} C={best_bs['C']} Brier={fmt(best_bs['select_brier'])}", flush=True)

    print("fit PT_flex grid…", flush=True)
    pt_flex_cands = []
    Xtr = np.column_stack([TR["p_pre"], TR["tmin"]])
    Xse = np.column_stack([SE["p_pre"], SE["tmin"]])
    for nk in (4, 6):
        for Ccoef in (0.01, 0.1, 1.0):
            pipe = fit_pt_flex(Xtr, TR["y"], w_tr, nk, Ccoef)
            pred_se = pipe.predict_proba(Xse)[:, 1]
            sc = metrics(SE["y"], pred_se, SE["g"])
            pt_flex_cands.append(
                dict(n_knots_p=nk, C=Ccoef, pipe=pipe, select_brier=sc["brier"], select_logloss=sc["logloss"])
            )
    pt_flex_cands.sort(key=lambda d: (d["select_brier"], d["select_logloss"]))
    best_pf = pt_flex_cands[0]
    raw_models["PT_flex"] = dict(
        kind="sklearn_pt",
        pipe=best_pf["pipe"],
        config=dict(n_knots_p=best_pf["n_knots_p"], n_knots_t=4, C=best_pf["C"], degree=3),
    )
    print(
        f"  PT_flex winner knots_p={best_pf['n_knots_p']} C={best_pf['C']} Brier={fmt(best_pf['select_brier'])}",
        flush=True,
    )

    raw_models["q_base"] = dict(kind="frozen_q")

    def predict_raw(name: str, pack: dict) -> np.ndarray:
        m = raw_models[name]
        if m["kind"] == "constant":
            return np.full(len(pack["y"]), m["p"], dtype=float)
        if m["kind"] == "sklearn_p":
            return m["pipe"].predict_proba(pack["p_pre"].reshape(-1, 1))[:, 1]
        if m["kind"] == "sklearn_pt":
            return m["pipe"].predict_proba(np.column_stack([pack["p_pre"], pack["tmin"]]))[:, 1]
        if m["kind"] == "frozen_q":
            return predict_q_frozen(logit_obj, pack)
        raise KeyError(name)

    # --- Q_CAL calibrators ---
    print("fit g_q (identity vs PosSlopeSigmoid) on Q_CAL…", flush=True)
    calib: Dict[str, Any] = {}
    selected: Dict[str, str] = {}
    for name in raw_models:
        praw_ca = predict_raw(name, CA)
        praw_se = predict_raw(name, SE)
        gq = PosSlopeSigmoid().fit(praw_ca, CA["y"], w_ca)
        pcal_se = gq.transform(praw_se)
        sc_id = metrics(SE["y"], praw_se, SE["g"])
        sc_cal = metrics(SE["y"], pcal_se, SE["g"])
        # choose identity if cal not ok or worse Brier; tie → logloss
        use_cal = False
        if gq.ok:
            if sc_cal["brier"] < sc_id["brier"] - 1e-15:
                use_cal = True
            elif abs(sc_cal["brier"] - sc_id["brier"]) <= 1e-15 and sc_cal["logloss"] < sc_id["logloss"]:
                use_cal = True
        mode = "sigmoid" if use_cal else "identity"
        selected[name] = mode
        calib[name] = dict(
            g=gq,
            mode=mode,
            select_identity=sc_id,
            select_sigmoid=sc_cal if gq.ok else None,
            g_dict=gq.to_dict(),
        )
        print(
            f"  {name}: choose {mode}  idBrier={fmt(sc_id['brier'])} "
            f"calBrier={fmt(sc_cal['brier']) if gq.ok else 'NA'}",
            flush=True,
        )

    def predict_rr(name: str, pack: dict) -> np.ndarray:
        p = predict_raw(name, pack)
        if selected[name] == "sigmoid":
            return calib[name]["g"].transform(p)
        return p

    # rename for reporting
    report_names = ["constant", "b_linear", "b_spline", "PT_linear", "PT_flex", "q_base"]
    # q_RR is q_base under selected calibrator
    alias = {n: ("q_RR" if n == "q_base" and selected[n] == "sigmoid" else n) for n in report_names}

    print("score TEST…", flush=True)
    test_pred = {n: predict_rr(n, TE) for n in report_names}
    # also keep raw q_base for side-by-side
    test_pred["q_base_raw"] = predict_raw("q_base", TE)

    b40 = TE["B40"] if TE["B40"] is not None else ((TE["p_pre"] >= 0.4) & (TE["p_pre"] <= 0.6))
    outside = ~b40

    def score_mask(mask: np.ndarray) -> Dict[str, Any]:
        out = {}
        for n in report_names + ["q_base_raw"]:
            if n == "q_base_raw":
                p = test_pred["q_base_raw"][mask]
            else:
                p = test_pred[n][mask]
            out[n] = metrics(TE["y"][mask], p, TE["g"][mask])
        return out

    all_mask = np.ones(len(TE["y"]), dtype=bool)
    scores_all = score_mask(all_mask)
    scores_b40 = score_mask(b40)
    scores_out = score_mask(outside)

    q_key = "q_base"  # RR predictions already include chosen calibrator
    pt_flex_key = "PT_flex"

    boot = {
        "all_q_minus_PT_flex": bootstrap_delta_brier(
            TE["y"], test_pred[q_key], test_pred[pt_flex_key], TE["g"], args.boot_reps, args.seed
        ),
        "all_q_minus_PT_linear": bootstrap_delta_brier(
            TE["y"], test_pred[q_key], test_pred["PT_linear"], TE["g"], args.boot_reps, args.seed
        ),
        "all_q_minus_b_spline": bootstrap_delta_brier(
            TE["y"], test_pred[q_key], test_pred["b_spline"], TE["g"], args.boot_reps, args.seed
        ),
        "B40_q_minus_PT_flex": bootstrap_delta_brier(
            TE["y"][b40], test_pred[q_key][b40], test_pred[pt_flex_key][b40], TE["g"][b40], args.boot_reps, args.seed
        ),
        "B40_q_minus_PT_linear": bootstrap_delta_brier(
            TE["y"][b40], test_pred[q_key][b40], test_pred["PT_linear"][b40], TE["g"][b40], args.boot_reps, args.seed
        ),
        "outside_q_minus_PT_flex": bootstrap_delta_brier(
            TE["y"][outside],
            test_pred[q_key][outside],
            test_pred[pt_flex_key][outside],
            TE["g"][outside],
            args.boot_reps,
            args.seed,
        ),
    }
    # H = D_B40 - D_outside; D = Brier(q)-Brier(PT_flex) with cell-local match weights.
    # Per-match losses then joint match bootstrap (with replacement).
    def match_cell_deltas(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-match ΔBrier = Brier_q - Brier_PT_flex inside cell; aligned to full TEST matches."""
        g = TE["g"].astype(str)
        y = TE["y"].astype(float)
        pq = test_pred[q_key]
        pf = test_pred[pt_flex_key]
        matches_all = np.unique(g)
        d_m = np.full(len(matches_all), np.nan, dtype=float)
        has = np.zeros(len(matches_all), dtype=bool)
        if not mask.any():
            return has, d_m
        gg = g[mask]
        yy = y[mask]
        a = pq[mask]
        b = pf[mask]
        cell_matches, inv, c = np.unique(gg, return_inverse=True, return_counts=True)
        w = 1.0 / c[inv]
        sum_q = np.zeros(len(cell_matches))
        sum_p = np.zeros(len(cell_matches))
        np.add.at(sum_q, inv, w * (a - yy) ** 2)
        np.add.at(sum_p, inv, w * (b - yy) ** 2)
        # each match's weights sum to 1 → sum_q[j] is that match's mean Brier
        d_cell = sum_q - sum_p
        # map into full match list
        idx_map = {m: i for i, m in enumerate(matches_all.tolist())}
        for j, cm in enumerate(cell_matches.tolist()):
            gi = idx_map[cm]
            d_m[gi] = float(d_cell[j])
            has[gi] = True
        return has, d_m

    has_b40, d_b40_m = match_cell_deltas(b40)
    has_out, d_out_m = match_cell_deltas(outside)

    def mean_where(has, vals, idx):
        pick = has[idx]
        if not np.any(pick):
            return float("nan")
        return float(np.mean(vals[idx][pick]))

    n_m_all = len(has_b40)
    idx_all = np.arange(n_m_all)
    D_B40_obs = mean_where(has_b40, d_b40_m, idx_all)
    D_out_obs = mean_where(has_out, d_out_m, idx_all)
    H_obs = float(D_B40_obs - D_out_obs)
    rng = np.random.default_rng(args.seed)
    H_draws = np.empty(args.boot_reps, dtype=float)
    for i in range(args.boot_reps):
        samp = rng.integers(0, n_m_all, size=n_m_all)
        H_draws[i] = float(mean_where(has_b40, d_b40_m, samp) - mean_where(has_out, d_out_m, samp))
    boot["H_B40_minus_outside"] = dict(
        estimate=H_obs,
        D_B40=D_B40_obs,
        D_outside=D_out_obs,
        ci95=[float(np.quantile(H_draws, 0.025)), float(np.quantile(H_draws, 0.975))],
        p_gt0=float(np.mean(H_draws > 0)),
    )

    # RR2 narrow bins
    print("RR2 narrow bins…", flush=True)
    bins = []
    bin_ix = assign_p_bin(TE["p_pre"])
    for i in range(20):
        mask = bin_ix == i
        y_m = TE["y"][mask]
        sc_q = metrics(y_m, test_pred[q_key][mask], TE["g"][mask])
        sc_pf = metrics(y_m, test_pred[pt_flex_key][mask], TE["g"][mask])
        sc_pl = metrics(y_m, test_pred["PT_linear"][mask], TE["g"][mask])
        dlt = TE["delta_V"][mask]
        w = match_weights(TE["g"][mask]) if mask.any() else None
        mean_dv = float(np.average(dlt, weights=w)) if mask.any() else float("nan")
        p_pos = float(np.average(y_m, weights=w)) if mask.any() else float("nan")
        d_boot = (
            bootstrap_delta_brier(
                y_m, test_pred[q_key][mask], test_pred[pt_flex_key][mask], TE["g"][mask], args.boot_reps, args.seed
            )
            if int(mask.sum()) > 50
            else dict(estimate=float("nan"), ci95=[float("nan"), float("nan")], p_gt0=float("nan"))
        )
        row = dict(
            bin=bin_label(i),
            bin_index=i,
            n=int(mask.sum()),
            n_matches=int(len(np.unique(TE["g"][mask]))) if mask.any() else 0,
            n_pos=int(y_m.sum()) if mask.any() else 0,
            n_neg=int((y_m == 0).sum()) if mask.any() else 0,
            p_pos=p_pos,
            mean_delta_V=mean_dv,
            q=sc_q,
            PT_flex=sc_pf,
            PT_linear=sc_pl,
            delta_brier_q_minus_PT_flex=d_boot,
            low_support=cell_low_support(sc_q, y_m) if mask.any() else True,
        )
        bins.append(row)

    # balance neighborhood detail
    balance_bins = []
    for lo, hi, lab in (
        (0.40, 0.45, "[0.40,0.45)"),
        (0.45, 0.50, "[0.45,0.50)"),
        (0.50, 0.55, "[0.50,0.55)"),
        (0.55, 0.60, "[0.55,0.60)"),
    ):
        mask = (TE["p_pre"] >= lo) & (TE["p_pre"] < hi)
        y_m = TE["y"][mask]
        d_boot = bootstrap_delta_brier(
            y_m, test_pred[q_key][mask], test_pred[pt_flex_key][mask], TE["g"][mask], args.boot_reps, args.seed
        )
        balance_bins.append(
            dict(
                bin=lab,
                n=int(mask.sum()),
                n_matches=int(len(np.unique(TE["g"][mask]))) if mask.any() else 0,
                q=metrics(y_m, test_pred[q_key][mask], TE["g"][mask]),
                PT_flex=metrics(y_m, test_pred[pt_flex_key][mask], TE["g"][mask]),
                PT_linear=metrics(y_m, test_pred["PT_linear"][mask], TE["g"][mask]),
                delta_brier_q_minus_PT_flex=d_boot,
                low_support=cell_low_support(metrics(y_m, test_pred[q_key][mask], TE["g"][mask]), y_m)
                if mask.any()
                else True,
            )
        )

    OUT.mkdir(parents=True, exist_ok=True)
    mdir = OUT / "models"
    mdir.mkdir(exist_ok=True)
    joblib.dump(raw_models["b_linear"]["pipe"], mdir / "b_linear.joblib")
    joblib.dump(raw_models["b_spline"]["pipe"], mdir / "b_spline.joblib")
    joblib.dump(raw_models["PT_linear"]["pipe"], mdir / "PT_linear.joblib")
    joblib.dump(raw_models["PT_flex"]["pipe"], mdir / "PT_flex.joblib")

    selection = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md",
        frozen_q="outputs/q_newv_fit85_20260920/models/logit_state.joblib",
        b_spline=raw_models["b_spline"]["config"],
        PT_flex=raw_models["PT_flex"]["config"],
        calibrator_choice={n: calib[n]["mode"] for n in report_names},
        calibrator_params={n: calib[n]["g_dict"] for n in report_names},
        q_cal_note="g_q only; g_V untouched. Selection on Q_SELECT all-T Brier (not B40).",
        aliases=alias,
    )
    (OUT / "baseline_selection.json").write_text(json.dumps(scrub(selection), indent=2) + "\n", encoding="utf-8")

    # prediction table (TEST keys)
    pred_path = OUT / "prediction_table.npz"
    np.savez_compressed(
        pred_path,
        match=TE["g"],
        s=TE["s"],
        y=TE["y"],
        p_pre=TE["p_pre"],
        delta_V=TE["delta_V"],
        B40=b40.astype(np.int8),
        **{f"p_{n}": test_pred[n] for n in report_names},
        p_q_base_raw=test_pred["q_base_raw"],
    )

    payload = dict(
        generated=selection["generated"],
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md",
        census=dict(train=len(TR["y"]), q_cal=len(CA["y"]), q_select=len(SE["y"]), test=len(TE["y"]), test_B40=int(b40.sum())),
        selection=selection,
        TEST_all=scores_all,
        TEST_B40=scores_b40,
        TEST_outside_B40=scores_out,
        bootstrap=boot,
        narrow_bins=bins,
        balance_bins=balance_bins,
        note=(
            "Primary RR contrast: Brier(q_RR)-Brier(PT_flex) on identical rows; "
            "q_RR = frozen logit_state + optional Q_CAL PosSlopeSigmoid chosen on Q_SELECT. "
            "Original PT_linear primary table preserved separately."
        ),
    )
    (OUT / "paired_ci.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    # markdown report
    def table_block(title: str, scores: Dict[str, Any]) -> List[str]:
        lines = [
            f"## {title}",
            "",
            "| Model | n | matches | Brier | logloss | AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        order = ["q_base", "PT_flex", "PT_linear", "b_spline", "b_linear", "constant", "q_base_raw"]
        for n in order:
            if n not in scores:
                continue
            sc = scores[n]
            label = n
            if n == "q_base":
                label = f"q_RR ({selected['q_base']})"
            lines.append(
                f"| {label} | {sc['n']} | {sc['n_matches']} | {fmt(sc['brier'])} | {fmt(sc['logloss'])} | {fmt(sc['auc'])} |"
            )
        return lines

    d_all = boot["all_q_minus_PT_flex"]
    d_b40 = boot["B40_q_minus_PT_flex"]
    d_lin = boot["all_q_minus_PT_linear"]
    H = boot["H_B40_minus_outside"]

    lines = [
        "# Review response RR1+RR2 — flexible baselines vs frozen q",
        "",
        f"Generated: {payload['generated']}",
        f"**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)",
        f"**Epistemic:** {ROLE} (post-TEST review response; not confirmatory preregistration)",
        "",
        "## Locks",
        "",
        "- V / SVI / engagement T / frozen `logit_state` weights: **unchanged**",
        "- New: `b_spline`, `PT_flex`, optional `g_q` (PosSlopeSigmoid) selected on Q_SELECT all-T",
        f"- Selected calibrators: `{json.dumps({n: selected[n] for n in report_names})}`",
        f"- PT_flex config: `{json.dumps(raw_models['PT_flex']['config'])}`",
        f"- b_spline config: `{json.dumps(raw_models['b_spline']['config'])}`",
        "",
    ]
    lines += table_block("TEST 15.16 T — all", scores_all)
    lines += [""]
    lines += table_block("TEST 15.16 T — B40", scores_b40)
    lines += [
        "",
        "### Primary RR contrast: ΔBrier(q_RR − PT_flex)",
        "",
        f"- **All T:** estimate={fmt(d_all['estimate'], 5)}  CI95=[{fmt(d_all['ci95'][0], 5)}, {fmt(d_all['ci95'][1], 5)}]  P(Δ>0)={fmt(d_all['p_gt0'], 4)}",
        f"- **B40:** estimate={fmt(d_b40['estimate'], 5)}  CI95=[{fmt(d_b40['ci95'][0], 5)}, {fmt(d_b40['ci95'][1], 5)}]  P(Δ>0)={fmt(d_b40['p_gt0'], 4)}",
        f"- All T vs PT_linear (continuity): estimate={fmt(d_lin['estimate'], 5)}  CI95=[{fmt(d_lin['ci95'][0], 5)}, {fmt(d_lin['ci95'][1], 5)}]",
        "",
        "### Heterogeneity H = D_B40 − D_outside (D = Brier(q)−Brier(PT_flex))",
        "",
        f"- D_B40={fmt(H['D_B40'], 5)}  D_outside={fmt(H['D_outside'], 5)}  H={fmt(H['estimate'], 5)}",
        f"- 95% CI=[{fmt(H['ci95'][0], 5)}, {fmt(H['ci95'][1], 5)}]  P(H>0)={fmt(H['p_gt0'], 4)}",
        "",
        "## Balance neighborhood (0.05-width)",
        "",
        "| Bin | n | matches | q Brier | PT_flex Brier | ΔBrier | CI95 | low-support |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in balance_bins:
        db = row["delta_brier_q_minus_PT_flex"]
        lines.append(
            f"| {row['bin']} | {row['n']} | {row['n_matches']} | {fmt(row['q']['brier'])} | "
            f"{fmt(row['PT_flex']['brier'])} | {fmt(db['estimate'], 5)} | "
            f"[{fmt(db['ci95'][0], 5)}, {fmt(db['ci95'][1], 5)}] | {row['low_support']} |"
        )
    lines += [
        "",
        "## Narrow p_pre bins (diagnostic; do not cherry-merge)",
        "",
        "| Bin | n | matches | P(Y=1) | E[ΔV] | ΔBrier(q−PT_flex) | low-support |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in bins:
        db = row["delta_brier_q_minus_PT_flex"]
        lines.append(
            f"| {row['bin']} | {row['n']} | {row['n_matches']} | {fmt(row['p_pos'])} | "
            f"{fmt(row['mean_delta_V'], 4)} | {fmt(db['estimate'], 5)} | {row['low_support']} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Negative ΔBrier ⇒ q better than that baseline on this cell.",
        "- If B40 CI includes 0: **additional lift not clearly confirmed** here — not 'no information' / study failure.",
        "- Surviving PT_flex ⇒ 'beyond the *tested* p/time summaries'; disappearing ⇒ function form of initial edge explained part of the prior linear gap.",
        "- Original primary table vs PT_linear remains the historical result; this pack is review-response.",
        "",
        f"Artifacts: `{OUT.as_posix()}/`",
        "",
    ]
    md = REPO / "docs" / "REVIEW_RESPONSE_RR12_RESULTS_20260920.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # public JSON copy (small)
    public = {
        k: payload[k]
        for k in (
            "generated",
            "epistemic",
            "design",
            "census",
            "TEST_all",
            "TEST_B40",
            "bootstrap",
            "balance_bins",
            "note",
        )
    }
    public["calibrator_choice"] = selection["calibrator_choice"]
    public["PT_flex"] = selection["PT_flex"]
    public["b_spline"] = selection["b_spline"]
    (REPO / "docs" / "REVIEW_RESPONSE_RR12_RESULTS_20260920.json").write_text(
        json.dumps(scrub(public), indent=2) + "\n", encoding="utf-8"
    )

    print("wrote", OUT / "paired_ci.json", md, flush=True)
    print(
        f"ALL  ΔBrier(q-PT_flex)={fmt(d_all['estimate'], 5)} CI=[{fmt(d_all['ci95'][0], 5)}, {fmt(d_all['ci95'][1], 5)}]",
        flush=True,
    )
    print(
        f"B40  ΔBrier(q-PT_flex)={fmt(d_b40['estimate'], 5)} CI=[{fmt(d_b40['ci95'][0], 5)}, {fmt(d_b40['ci95'][1], 5)}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
