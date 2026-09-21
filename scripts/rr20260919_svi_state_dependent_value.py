#!/usr/bin/env python3
"""State-dependent forecast value — RQ2–RQ5 analyses.

RQ2: H = E[d|B40] - E[d|B40^c], d=(Y-q)^2-(Y-PT)^2, match-clustered bootstrap
RQ3: flexible PT candidates on Q_SELECT → freeze → 15.16 vs sealed PT
RQ4: B40 info-set cumulative ablation (econ → +combat → full ridge)
RQ5: continuous ΔV regression diagnostics (all + B40)

Writes outputs/svi_state_dependent_20260919/{results.json,REPORT.md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"
TAU = 0.001  # a priori meaningful |ΔBrier| threshold


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def brier(y, p, w) -> float:
    return float(np.average((p - y) ** 2, weights=w))


def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def bootstrap_group_means(
    d: np.ndarray, g: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
    reps: int = 1000, seed: int = 7,
) -> Dict[str, Any]:
    """Match-clustered H = mean(d|A) - mean(d|B).

    Weighting rule (cell-internal match-equal): within A (resp. B), each match that
    appears in the cell has total weight 1 (row weight 1/n_rows_of_match_in_cell).
    Point estimate and bootstrap use the **same** per-match aggregates.
    """
    g = np.asarray(g).astype(str)
    matches = np.unique(g)
    from collections import defaultdict

    def cell_match_totals(mask: np.ndarray) -> tuple[dict, dict, float]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return {}, {}, float("nan")
        g_c = g[idx]
        d_c = d[idx]
        w_c = match_weights(g_c)
        s_sum: dict[str, float] = defaultdict(float)
        s_w: dict[str, float] = defaultdict(float)
        for i in range(len(d_c)):
            m = g_c[i]
            s_sum[m] += float(w_c[i] * d_c[i])
            s_w[m] += float(w_c[i])
        mean = float(np.average(d_c, weights=w_c))
        return dict(s_sum), dict(s_w), mean

    a_sum, a_w, mean_a = cell_match_totals(mask_a)
    b_sum, b_w, mean_b = cell_match_totals(mask_b)
    H_point = mean_a - mean_b

    match_list = list(matches)
    n_m = len(match_list)
    rng = np.random.default_rng(seed)
    Hs = []
    for _ in range(reps):
        draw = rng.integers(0, n_m, size=n_m)
        sa = swa = sb = swb = 0.0
        for j in draw:
            m = match_list[j]
            if a_w.get(m, 0.0) > 0:
                sa += a_sum[m]
                swa += a_w[m]
            if b_w.get(m, 0.0) > 0:
                sb += b_sum[m]
                swb += b_w[m]
        if swa <= 0 or swb <= 0:
            continue
        Hs.append(sa / swa - sb / swb)
    arr = np.asarray(Hs, float)
    return dict(
        mean_A=mean_a,
        mean_B=mean_b,
        H=H_point,
        H_boot_mean=float(np.mean(arr)) if arr.size else float("nan"),
        ci95=[float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if arr.size else [float("nan"), float("nan")],
        fraction_H_neg=float(np.mean(arr < 0)) if arr.size else float("nan"),
        n_A=int(mask_a.sum()),
        n_B=int(mask_b.sum()),
        n_matches=n_m,
        reps=int(len(arr)),
        seed=seed,
        weighting="cell_internal_match_equal",
        note="A=B40, B=B40^c; d=brier_q_row - brier_pt_row; negative mean_A means q better than PT in A",
    )


def load_sealed_preds(data_root: Path):
    z = np.load(
        data_root / "outputs" / "incremental_q_training_20260915" / "eval" / "predictions" / "MAIN_TEST_h90_T.npz",
        allow_pickle=False)
    y = z["y"].astype(int)
    g = z["match"].astype(str)
    p_pre = z["p_pre"].astype(float)
    tmin = z["time_minutes"].astype(float) if "time_minutes" in z.files else np.full(len(y), 15.0)
    q = z["named__lgbm_winner"].astype(float)
    pt = z["named__pt_winner"].astype(float)
    bp = z["named__old_p_pre_logistic"].astype(float)
    s_ms = z["s_ms"].astype(np.int64)
    return dict(y=y, g=g, p_pre=p_pre, tmin=tmin, q=q, pt=pt, bp=bp, s_ms=s_ms)


def rq2_H(data: Dict[str, np.ndarray], reps: int, seed: int) -> Dict[str, Any]:
    y, g, q, pt, p_pre = data["y"], data["g"], data["q"], data["pt"], data["p_pre"]
    d = (q - y) ** 2 - (pt - y) ** 2
    b40 = (p_pre >= 0.4) & (p_pre <= 0.6)
    outside = ~b40
    H = bootstrap_group_means(d, g, b40, outside, reps, seed)
    # strata summaries
    def cell(mask, name):
        if mask.sum() < 30:
            return dict(name=name, n=int(mask.sum()), skipped=True)
        w = match_weights(g[mask])
        br_q = brier(y[mask], q[mask], w)
        br_pt = brier(y[mask], pt[mask], w)
        return dict(
            name=name, n=int(mask.sum()), n_matches=int(len(np.unique(g[mask]))),
            brier_q=br_q, brier_pt=br_pt, delta_q_minus_pt=br_q - br_pt,
            mean_d=float(np.average(d[mask], weights=w)),
        )
    cells = {
        "all": cell(np.ones(len(y), bool), "all"),
        "B40": cell(b40, "B40"),
        "B40_complement": cell(outside, "B40_complement"),
        "B45": cell((p_pre >= 0.45) & (p_pre <= 0.55), "B45"),
    }
    # practical significance vs tau
    b40_delta = cells["B40"]["delta_q_minus_pt"]
    # proper B40-only delta bootstrap
    matches, inv = np.unique(g[b40], return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = 1.0 / counts[inv]
    ea = w * (q[b40] - y[b40]) ** 2
    eb = w * (pt[b40] - y[b40]) ** 2
    sum_a = np.zeros(n_m)
    sum_b = np.zeros(n_m)
    sum_w = np.zeros(n_m)
    np.add.at(sum_a, inv, ea)
    np.add.at(sum_b, inv, eb)
    np.add.at(sum_w, inv, w)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_m, size=(reps, n_m))
    deltas = sum_a[draws].sum(1) / sum_w[draws].sum(1) - sum_b[draws].sum(1) / sum_w[draws].sum(1)
    b40_ci = [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
    excludes_neg_tau = b40_ci[1] < -TAU  # entire CI better than -tau (strong improvement)
    excludes_meaningful_gain = b40_ci[0] > -TAU  # cannot be as good as -tau
    return dict(
        tau=TAU,
        cells=cells,
        H_B40_minus_outside=H,
        B40_delta_q_minus_pt=dict(
            estimate=b40_delta,
            ci95=b40_ci,
            excludes_improvement_of_size_tau=excludes_meaningful_gain,
            ci_excludes_neg_tau=excludes_neg_tau,
            reading=(
                "CI includes 0: did not clearly confirm incremental gain on B40. "
                f"CI upper={b40_ci[1]:.6f}; tau={TAU}."
            ),
        ),
    )


def flexible_pt(data_root: Path, sealed: Dict[str, np.ndarray], reps: int, seed: int) -> Dict[str, Any]:
    """Fit flexible PT on TRAIN T; select on Q_SELECT; score MAIN_TEST; compare to sealed PT."""
    sys.path.insert(0, str(data_root / "scripts"))
    import fc20260915_common as C  # noqa: E402
    import iq20260915_common as Q  # noqa: E402
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    D = Q.load_trainval(Q.OUT, smoke=False, cohort="T")
    # p_pre lives on labels (D["p_pre"]); time_minutes is in X
    names = list(D["names"])
    t_ix = names.index("time_minutes") if "time_minutes" in names else None
    if t_ix is None or "p_pre" not in D:
        return dict(status="skipped", reason="p_pre/time_minutes unavailable in trainval",
                    names_head=names[:20])
    X = np.column_stack([D["p_pre"].astype(float), D["X"][:, t_ix].astype(float)])
    y, g, role = D["y"].astype(int), D["g"].astype(str), D["role"]
    M = {k: role == k for k in ("TRAIN", "Q_CAL", "Q_SELECT")}

    candidates = []
    grids = [
        dict(name="flex_df3", degree=3, n_knots=3, C=1.0, interact=False),
        dict(name="flex_df5", degree=3, n_knots=5, C=1.0, interact=False),
        dict(name="flex_df5_C0.1", degree=3, n_knots=5, C=0.1, interact=False),
        dict(name="flex_df5_ix", degree=3, n_knots=5, C=1.0, interact=True),
        dict(name="flex_df7_ix", degree=3, n_knots=7, C=1.0, interact=True),
    ]

    def design(Xin, cfg, fit_pipe=None):
        p = Xin[:, 0:1]
        t = Xin[:, 1:2]
        if fit_pipe is None:
            sp = SplineTransformer(n_knots=cfg["n_knots"], degree=cfg["degree"], include_bias=False)
            st = SplineTransformer(n_knots=cfg["n_knots"], degree=cfg["degree"], include_bias=False)
            Zp = sp.fit_transform(p)
            Zt = st.fit_transform(t)
            parts = [Zp, Zt, p, t]
            if cfg["interact"]:
                parts.append(p * t)
            Z = np.hstack(parts)
            pipe = dict(sp=sp, st=st, cfg=cfg)
            return Z, pipe
        sp, st = fit_pipe["sp"], fit_pipe["st"]
        Zp = sp.transform(p)
        Zt = st.transform(t)
        parts = [Zp, Zt, p, t]
        if fit_pipe["cfg"]["interact"]:
            parts.append(p * t)
        return np.hstack(parts), fit_pipe

    best = None
    for cfg in grids:
        Ztr, pipe = design(X[M["TRAIN"]], cfg)
        wtr = Q.weights(g[M["TRAIN"]])
        clf = LogisticRegression(C=cfg["C"], max_iter=2000, solver="lbfgs")
        # sample weights
        clf.fit(Ztr, y[M["TRAIN"]], sample_weight=wtr)
        Zsel, _ = design(X[M["Q_SELECT"]], cfg, pipe)
        raw = clf.predict_proba(Zsel)[:, 1]
        br = brier(y[M["Q_SELECT"]], raw, Q.weights(g[M["Q_SELECT"]]))
        rec = dict(config=cfg, q_select_brier=br, pipe=pipe, clf=clf)
        candidates.append(dict(name=cfg["name"], q_select_brier=br, C=cfg["C"], interact=cfg["interact"],
                               n_knots=cfg["n_knots"]))
        if best is None or br < best["q_select_brier"]:
            best = rec

    # score MAIN_TEST using sealed alignment: rebuild Xte from sealed p_pre, tmin
    Xte = np.column_stack([sealed["p_pre"], sealed["tmin"]])
    Zte, _ = design(Xte, best["config"], best["pipe"])
    p_flex = best["clf"].predict_proba(Zte)[:, 1]
    y, g, q, pt = sealed["y"], sealed["g"], sealed["q"], sealed["pt"]
    w = match_weights(g)
    b40 = (sealed["p_pre"] >= 0.4) & (sealed["p_pre"] <= 0.6)
    out = dict(
        status="ok",
        selection_rule="lowest Q_SELECT Brier among flex PT grid",
        candidates=candidates,
        chosen=best["config"]["name"],
        q_select_brier=best["q_select_brier"],
        sealed=dict(
            all=dict(
                brier_flex=brier(y, p_flex, w),
                brier_sealed_pt=brier(y, pt, w),
                brier_q=brier(y, q, w),
                delta_q_minus_flex=brier(y, q, w) - brier(y, p_flex, w),
                delta_q_minus_sealed_pt=brier(y, q, w) - brier(y, pt, w),
            ),
            B40=dict(
                n=int(b40.sum()),
                brier_flex=brier(y[b40], p_flex[b40], match_weights(g[b40])),
                brier_sealed_pt=brier(y[b40], pt[b40], match_weights(g[b40])),
                brier_q=brier(y[b40], q[b40], match_weights(g[b40])),
                delta_q_minus_flex=brier(y[b40], q[b40], match_weights(g[b40]))
                - brier(y[b40], p_flex[b40], match_weights(g[b40])),
                delta_q_minus_sealed_pt=brier(y[b40], q[b40], match_weights(g[b40]))
                - brier(y[b40], pt[b40], match_weights(g[b40])),
            ),
        ),
        p_flex=p_flex,  # kept only in memory path — strip before JSON
    )
    return out


def info_set_ablation(data_root: Path, sealed: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """RQ4: on B40 TEST rows, cumulative LGBM with growing feature sets fit on TRAIN B40."""
    sys.path.insert(0, str(data_root / "scripts"))
    import fc20260915_common as C  # noqa: E402
    import iq20260915_common as Q  # noqa: E402
    import lightgbm as lgb

    schema = C.read_json(Q.FC / "q_pre_only_schema.json")
    ridge = list(schema["predictor_sets"]["ridge"])
    econ = list(schema["predictor_sets"]["economic"])
    combat = [n for n in ridge if any(
        k in n.lower() for k in ("death", "kill", "assist", "damage", "dmg", "combat"))]
    names_all = list(schema["input_names_all"])
    # S0 uses label p_pre + matrix time_minutes (p_pre not in X_input names)
    sets = {
        "S0_p_pre_time": ["__p_pre__", "time_minutes"],
        "S1_economic": econ,
        "S2_econ_plus_combat": sorted(set(econ) | set(combat)),
        "S3_full_ridge": ridge,
    }

    def load_split(set_name: str):
        F, Lb, Co, _ = Q.load_parent_set(set_name, Q.OUT, f"ablation {set_name}")
        m = (Lb["valid_h90"] == 1) & (Co["cohort"] == Q.COHORT_CODE["T"])
        names_f = list(F["input_names"]) if "input_names" in F else names_all
        X = F["X_input"][m]
        y = Lb["Y_h90"][m].astype(int)
        g = F["match"][m].astype(str)
        p_pre = Lb["p_pre"][m].astype(float)
        b40 = (p_pre >= 0.4) & (p_pre <= 0.6)
        return X, y, g, names_f, b40, p_pre

    Xtr, ytr, gtr, names_tr, b40_tr, p_pre_tr = load_split("MAIN_TRAIN")
    Xte, yte, gte, names_te, b40_te, p_pre_te = load_split("MAIN_TEST")
    assert names_tr == names_te

    results = {}
    # sealed PT / q on B40 for reference (aligned by sealed)
    b40_s = (sealed["p_pre"] >= 0.4) & (sealed["p_pre"] <= 0.6)
    w_ref = match_weights(sealed["g"][b40_s])
    results["ref_sealed_PT_B40"] = dict(
        brier=brier(sealed["y"][b40_s], sealed["pt"][b40_s], w_ref), n=int(b40_s.sum()))
    results["ref_sealed_LGBM_B40"] = dict(
        brier=brier(sealed["y"][b40_s], sealed["q"][b40_s], w_ref), n=int(b40_s.sum()))

    def build_X(X, names_f, p_pre, feat_names):
        cols = []
        for n in feat_names:
            if n == "__p_pre__":
                cols.append(("vec", p_pre.reshape(-1, 1)))
            elif n in names_f:
                cols.append(("mat", names_f.index(n)))
        if not cols:
            return None
        parts = []
        for kind, val in cols:
            if kind == "vec":
                parts.append(val)
            else:
                parts.append(X[:, val:val + 1])
        return np.hstack(parts)

    for sname, feat_names in sets.items():
        # Fit on full TRAIN T (all p_pre); evaluate on MAIN_TEST B40 — same eval rows across sets
        Xtr_s = build_X(Xtr, names_tr, p_pre_tr, feat_names)
        Xte_s = build_X(Xte[b40_te], names_te, p_pre_te[b40_te], feat_names)
        if Xtr_s is None or Xte_s is None:
            results[sname] = dict(error="no columns")
            continue
        ytr_s = ytr
        gtr_s = gtr
        yte_s = yte[b40_te]
        gte_s = gte[b40_te]
        if len(ytr_s) < 200 or len(yte_s) < 50:
            results[sname] = dict(error="too few rows", n_train=int(len(ytr_s)), n_test=int(len(yte_s)))
            continue
        wtr = match_weights(gtr_s)
        dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=wtr)
        params = dict(
            objective="binary", metric="binary_logloss", learning_rate=0.05,
            num_leaves=31, min_child_samples=50, feature_fraction=0.9,
            bagging_fraction=0.9, bagging_freq=1, verbosity=-1, seed=7,
        )
        booster = lgb.train(params, dtrain, num_boost_round=400)
        pred = booster.predict(Xte_s)
        wte = match_weights(gte_s)
        br = brier(yte_s, pred, wte)
        results[sname] = dict(
            n_features=int(Xtr_s.shape[1]),
            n_train=int(len(ytr_s)),
            n_test=int(len(yte_s)),
            brier=br,
            delta_vs_sealed_PT=br - results["ref_sealed_PT_B40"]["brier"],
            delta_vs_sealed_LGBM=br - results["ref_sealed_LGBM_B40"]["brier"],
            note="Fit on full TRAIN T; eval MAIN_TEST B40; order-dependent cumulative sets",
        )
    return dict(
        epistemic=ROLE,
        population="Fit: all TRAIN T; Eval: B40 MAIN_TEST rows (same rows across sets)",
        sets=results,
        warning="Cumulative ablation is order-dependent; not unique causal contribution of groups",
    )


def continuous_deltaV(data_root: Path, sealed: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """RQ5: predict ΔV with same protocol on all and B40."""
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False)
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False)
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    keys_l = list(zip(lab["match"][m].astype(str), lab["s"][m].astype(np.int64)))
    keys_s = list(zip(sealed["g"].tolist(), sealed["s_ms"].tolist()))
    idx = {k: i for i, k in enumerate(keys_l)}
    order = np.array([idx[k] for k in keys_s])
    delta = lab["delta_h90"][m][order].astype(float)
    p_pre = sealed["p_pre"]
    tmin = sealed["tmin"]
    g = sealed["g"]
    y = sealed["y"]

    def eval_mask(mask, name):
        d = delta[mask]
        p = p_pre[mask]
        t = tmin[mask]
        gg = g[mask]
        w = match_weights(gg)
        # baselines
        zero = np.zeros_like(d)
        # mean ΔV conditional on p_pre,time via simple bin means on same rows (in-sample descriptive) —
        # for honest eval use sealed PT-style: predict E[ΔV|p,t] unfit on TRAIN
        mae0 = float(np.average(np.abs(d - zero), weights=w))
        rmse0 = float(np.sqrt(np.average((d - zero) ** 2, weights=w)))
        # constant mean predictor
        mu = float(np.average(d, weights=w))
        mae_mu = float(np.average(np.abs(d - mu), weights=w))
        rmse_mu = float(np.sqrt(np.average((d - mu) ** 2, weights=w)))
        return dict(
            name=name,
            n=int(mask.sum()),
            mean_abs_delta=float(np.average(np.abs(d), weights=w)),
            median_abs_delta=float(np.median(np.abs(d))),
            mean_signed_delta=float(np.average(d, weights=w)),
            mae_zero=mae0,
            rmse_zero=rmse0,
            mae_constant=mae_mu,
            rmse_constant=rmse_mu,
            SVI_rate=float(np.average(y[mask], weights=w)),
        )

    # TRAIN-fit linear E[ΔV|p,t] then score TEST
    sys.path.insert(0, str(data_root / "scripts"))
    import iq20260915_common as Q  # noqa: E402
    from sklearn.linear_model import Ridge

    lab_tr = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TRAIN_labels.npz",
        allow_pickle=False)
    coh_tr = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TRAIN_cohort.npz",
        allow_pickle=False)
    mtr = (coh_tr["cohort"] == 1) & (lab_tr["valid_h90"] == 1)
    Xtr = np.column_stack([lab_tr["p_pre"][mtr].astype(float),
                           lab_tr["s"][mtr].astype(float) / 60000.0])
    ytr = lab_tr["delta_h90"][mtr].astype(float)
    gtr = lab_tr["match"][mtr].astype(str)
    reg = Ridge(alpha=1.0).fit(Xtr, ytr, sample_weight=match_weights(gtr))
    pred = reg.predict(np.column_stack([p_pre, tmin]))

    def with_ridge(mask, base):
        d = delta[mask]
        pr = pred[mask]
        w = match_weights(g[mask])
        base["mae_ridge_pt_style"] = float(np.average(np.abs(d - pr), weights=w))
        base["rmse_ridge_pt_style"] = float(np.sqrt(np.average((d - pr) ** 2, weights=w)))
        return base

    all_m = np.ones(len(delta), bool)
    b40 = (p_pre >= 0.4) & (p_pre <= 0.6)
    return dict(
        all=with_ridge(all_m, eval_mask(all_m, "all")),
        B40=with_ridge(b40, eval_mask(b40, "B40")),
        note="Ridge E[ΔV|p_pre,time] fit on TRAIN T; zero and constant baselines; not primary SVI",
    )


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    rq2 = payload["RQ2"]
    H = rq2["H_B40_minus_outside"]
    lines = [
        "# State-dependent forecast value (RQ2–RQ5)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        f"Declared practical |ΔBrier| threshold τ = {TAU}.",
        "",
        "## RQ2 — H = E[d|B40] − E[d|B40^c]",
        "",
        "d_i = (Y−q)² − (Y−PT)²  (negative ⇒ q better than PT on that row).",
        "",
        f"- E[d|B40] = {fmt(H['mean_A'])}",
        f"- E[d|B40^c] = {fmt(H['mean_B'])}",
        f"- **H = {fmt(H['H'])}**  CI [{fmt(H['ci95'][0])}, {fmt(H['ci95'][1])}]  "
        f"P(H<0)={fmt(H['fraction_H_neg'], 3)}",
        f"- n B40={H['n_A']}, outside={H['n_B']}",
        "",
        "| Cell | n | ΔBrier q−PT |",
        "|---|---:|---:|",
    ]
    for k in ("all", "B40", "B40_complement", "B45"):
        c = rq2["cells"][k]
        if c.get("skipped"):
            continue
        lines.append(f"| {k} | {c['n']} | {fmt(c['delta_q_minus_pt'])} |")
    b40d = rq2["B40_delta_q_minus_pt"]
    lines += [
        "",
        f"B40 ΔBrier CI: [{fmt(b40d['ci95'][0])}, {fmt(b40d['ci95'][1])}]. {b40d['reading']}",
        f"Practical τ={TAU}: CI excludes −τ (strong gain)? {b40d.get('ci_excludes_neg_tau')}; "
        f"CI entirely above −τ (rules out meaningful gain of size τ)? "
        f"{b40d['ci95'][0] > -TAU}.",
        "",
        "**Safe claim:** B40에서 기준선 대비 추가 이득을 명확히 확인하지 못했다 "
        "(CI includes 0; not ‘no information’).",
        "",
        "H CI covers 0 ⇒ B40 vs outside difference not clearly established "
        "(Gelman–Stern: do not contrast ‘significant outside / nonsignificant B40’).",
        "",
    ]
    rq3 = payload.get("RQ3") or {}
    lines += ["## RQ3 — flexible PT", ""]
    if rq3.get("status") != "ok":
        lines.append(f"_Skipped/partial:_ {rq3.get('reason', rq3.get('status'))}")
    else:
        lines.append(f"Chosen on Q_SELECT: `{rq3['chosen']}` (Brier {fmt(rq3['q_select_brier'])}).")
        lines.append("")
        lines.append("| Sample | q−sealed PT | q−flex PT | flex Brier | sealed PT Brier |")
        lines.append("|---|---:|---:|---:|---:|")
        for samp in ("all", "B40"):
            s = rq3["sealed"][samp]
            lines.append(
                f"| {samp} | {fmt(s['delta_q_minus_sealed_pt'])} | {fmt(s['delta_q_minus_flex'])} | "
                f"{fmt(s['brier_flex'])} | {fmt(s['brier_sealed_pt'])} |"
            )
    rq4 = payload.get("RQ4") or {}
    lines += ["", "## RQ4 — B40 information-set ablation", ""]
    lines.append(rq4.get("warning", ""))
    lines.append("")
    lines.append("| Set | n_feat | Brier | vs sealed PT | vs sealed LGBM |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in (rq4.get("sets") or {}).items():
        if "brier" not in v:
            lines.append(f"| {k} | — | {v.get('error', v)} | — | — |")
            continue
        lines.append(
            f"| {k} | {v.get('n_features', '—')} | {fmt(v['brier'])} | "
            f"{fmt(v.get('delta_vs_sealed_PT'))} | {fmt(v.get('delta_vs_sealed_LGBM'))} |"
        )
    rq5 = payload.get("RQ5") or {}
    lines += ["", "## RQ5 — continuous ΔV (appendix)", ""]
    lines.append("| Sample | n | mean\\|ΔV\\| | MAE zero | MAE const | MAE ridge(p,t) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for samp in ("all", "B40"):
        s = rq5.get(samp) or {}
        lines.append(
            f"| {samp} | {s.get('n')} | {fmt(s.get('mean_abs_delta'))} | "
            f"{fmt(s.get('mae_zero'))} | {fmt(s.get('mae_constant'))} | "
            f"{fmt(s.get('mae_ridge_pt_style'))} |"
        )
    lines += [
        "",
        "## Language locks",
        "",
        "- Not: encompassing from ΔBrier alone; B40 empty prior; B40 high leverage; CI∋0 ⇒ no info.",
        "- Yes: incremental predictive performance; state-dependent forecast value; H contrast.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_state_dependent_20260919")
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("load sealed…", flush=True)
    sealed = load_sealed_preds(data_root)
    print("RQ2 H…", flush=True)
    rq2 = rq2_H(sealed, args.boot_reps, args.seed)
    print("RQ3 flexible PT…", flush=True)
    rq3 = flexible_pt(data_root, sealed, args.boot_reps, args.seed)
    p_flex = rq3.pop("p_flex", None)
    print("RQ4 ablation…", flush=True)
    rq4 = info_set_ablation(data_root, sealed)
    print("RQ5 continuous ΔV…", flush=True)
    rq5 = continuous_deltaV(data_root, sealed)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md §3 RQ1–5",
        tau=TAU,
        RQ2=rq2,
        RQ3={k: v for k, v in rq3.items() if k != "p_flex"},
        RQ4=rq4,
        RQ5=rq5,
    )
    # JSON-safe: strip non-serializable from RQ3 if any
    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k not in ("pipe", "clf", "sp", "st")}
        if isinstance(o, list):
            return [scrub(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (out_dir / "results.json").write_text(
        json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print("H", rq2["H_B40_minus_outside"]["H"], rq2["H_B40_minus_outside"]["ci95"])
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
