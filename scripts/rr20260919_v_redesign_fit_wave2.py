#!/usr/bin/env python3
"""V redesign wave-2 Tier 0–1 horse-race (same 210k cohort).

Candidates (docs/V2_CANDIDATE_MATRIX_20260919.md):
  A0  shared logistic (re-fit expanded StateV2)     — Maymin / Hodge LR
  A1  shared Random Forest                          — Hodge RF
  A2  shared LightGBM wave-1 (load if present)      — Hodge / CoG tab
  A2b shared LightGBM CoG paper hyperparameters
  A3  shared MLP + PosSlopeSigmoid                  — Kim / CoG matched MLP
  B0  per_band_lgbm (load if present)               — ablation
  legacy_logistic_raw                               — compare only

Selection: L_time on V_SELECT (alpha=1/4); tie-break overall logloss.
Writes: outputs/v_redesign_wave2_20260919/{results.json,REPORT.md,models/}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
BANDS = ((0.0, 10.0, "t_0_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf"))
ALPHA = {b[2]: 0.25 for b in BANDS}
WAVE1 = REPO / "outputs" / "v_redesign_20260919"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    # worktree `train` must precede REPO (REPO has a different train package)
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
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(n=int(len(y)), n_matches=int(len(np.unique(g))), brier=br, logloss=ll, auc=auc)


def band_mask(tmin, lo, hi):
    return (tmin >= lo) & (tmin < hi)


def time_balanced_brier(y, p, g, tmin) -> Dict[str, Any]:
    parts, acc, missing = {}, 0.0, []
    for lo, hi, name in BANDS:
        m = band_mask(tmin, lo, hi)
        if int(m.sum()) < 50:
            parts[name] = dict(skipped=True, n=int(m.sum()))
            missing.append(name)
            continue
        sc = metrics(y[m], p[m], g[m])
        parts[name] = sc
        acc += ALPHA[name] * sc["brier"]
    if missing:
        return dict(L_time=float("inf"), bands=parts, ineligible=True, missing_bands=missing)
    return dict(L_time=float(acc), bands=parts, ineligible=False)


def expanded_X(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float64, copy=False), cols, champ_ix


class PosSlopeSigmoid:
    def __init__(self):
        self.ok = False
        self.coef_ = 1.0
        self.intercept_ = 0.0

    def fit(self, p, y, w):
        from sklearn.linear_model import LogisticRegression
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y, sample_weight=w)
        coef = float(lr.coef_.ravel()[0])
        if coef <= 0:
            self.ok = False
            return self
        self.ok, self.coef_, self.intercept_ = True, coef, float(lr.intercept_.ravel()[0])
        return self

    def transform(self, p):
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        return 1.0 / (1.0 + np.exp(-(self.coef_ * logit + self.intercept_)))


def predict_lgbm(clf, X, cols, champ_ix):
    import pandas as pd
    df = pd.DataFrame(X, columns=list(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in champ_ix:
            df[cols[j]] = df[cols[j]].astype("category")
        return clf.predict_proba(df)[:, 1]


def fit_lgbm_wave1(X, y, g, champ_ix, cols, seed=7):
    import lightgbm as lgb
    import pandas as pd
    w = match_weights(g)
    df = pd.DataFrame(X, columns=list(cols))
    for j in champ_ix:
        df[cols[j]] = df[cols[j]].astype("category")
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbose=-1,
    )
    clf.fit(df, y, sample_weight=w)
    return clf


def fit_lgbm_cog(X, y, g, X_val, y_val, g_val, champ_ix, cols, seed=7):
    """CoG paper-ish LightGBM (MODELS.md / CoG §4.2.1) with early stop on V_CAL."""
    import lightgbm as lgb
    import pandas as pd
    w = match_weights(g)
    wv = match_weights(g_val)
    df = pd.DataFrame(X, columns=list(cols))
    dfv = pd.DataFrame(X_val, columns=list(cols))
    for j in champ_ix:
        df[cols[j]] = df[cols[j]].astype("category")
        dfv[cols[j]] = dfv[cols[j]].astype("category")
    clf = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_samples=200,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(
        df, y, sample_weight=w,
        eval_set=[(dfv, y_val)],
        eval_sample_weight=[wv],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    return clf


def fit_logistic(X_raw, y, g, names, C=0.01):
    from train.state_value_experiment import logistic
    from train.temporal_winprob import feature_matrix
    Xf, cols = feature_matrix(X_raw, names, "expanded")
    model = logistic(cols, C=C)
    w = match_weights(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xf, y, model__sample_weight=w)
    keep = [names.index(c) for c in cols]
    return model, keep


def predict_logistic(model, X_raw, keep):
    return model.predict_proba(X_raw[:, keep])[:, 1]


def fit_rf(X, y, g, seed=7):
    """Hodge-style RF; subsample rows for tractability on ~400k."""
    from sklearn.ensemble import RandomForestClassifier
    w = match_weights(g)
    # sklearn RF: sample_weight supported
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=50,
        max_features="sqrt",
        max_samples=0.5,
        n_jobs=4,
        random_state=seed,
        verbose=0,
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def fit_mlp(X, y, g, X_val, y_val, g_val, seed=7, max_epochs=40, batch=8192, hidden=(256, 128)):
    """Kim/CoG-style MLP on standardized StateV2 (torch)."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = StandardScaler()
    Xtr = sc.fit_transform(X).astype(np.float32)
    Xva = sc.transform(X_val).astype(np.float32)
    wtr = match_weights(g).astype(np.float32)
    wva = match_weights(g_val).astype(np.float32)

    layers: List[nn.Module] = []
    d = Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def batches(Xm, ym, wm):
        n = len(ym)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        for i in range(0, n, batch):
            sl = idx[i : i + batch]
            yield (
                torch.from_numpy(Xm[sl]).to(device),
                torch.from_numpy(ym[sl].astype(np.float32)).to(device),
                torch.from_numpy(wm[sl]).to(device),
            )

    best_state, best_brier, patience, bad = None, 1e9, 6, 0
    yva_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    wva_t = torch.from_numpy(wva).to(device)

    for epoch in range(max_epochs):
        net.train()
        for xb, yb, wb in batches(Xtr, y, wtr):
            opt.zero_grad()
            logit = net(xb).squeeze(-1)
            loss = (loss_fn(logit, yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva_t).squeeze(-1))
            br = float((((pv - yva_t) ** 2) * wva_t).sum() / wva_t.sum())
        if br < best_brier - 1e-5:
            best_brier, bad = br, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return dict(net=net.cpu(), scaler=sc, device="cpu", hidden=list(hidden), best_val_brier=best_brier)


def predict_mlp(pack, X):
    import torch
    Xs = pack["scaler"].transform(X).astype(np.float32)
    net = pack["net"]
    net.eval()
    with torch.no_grad():
        t = torch.from_numpy(Xs)
        # chunk for memory
        outs = []
        for i in range(0, len(Xs), 65536):
            outs.append(torch.sigmoid(net(t[i : i + 65536]).squeeze(-1)).numpy())
    return np.concatenate(outs)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "v_redesign_wave2_20260919")
    ap.add_argument("--skip-rf", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-cog-lgbm", action="store_true")
    ap.add_argument("--skip-logistic", action="store_true")
    ap.add_argument("--smoke-frac", type=float, default=0.0)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load TRAIN / V_CAL / V_SELECT / TEST…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    CA = D.load_v_rows(L, "MAIN", ["V_CAL"], bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TR["names"])
    W = D.load_outcomes(
        L, "MAIN", train_roles + ["V_CAL", "V_SELECT", "TEST"],
        purpose="V redesign wave2 Tier0-1 horse-race",
    )

    def y_of(R):
        return np.asarray([W[m][0] for m in R["match"].tolist()], dtype=np.int64)

    if args.smoke_frac and args.smoke_frac < 1.0:
        rng = np.random.default_rng(7)
        matches = np.unique(TR["match"])
        keep_m = set(rng.choice(matches, size=max(200, int(len(matches) * args.smoke_frac)), replace=False).tolist())
        m = np.isin(TR["match"], list(keep_m))
        for k in ("match", "query_ms", "X"):
            TR[k] = TR[k][m]
        print(f"smoke TRAIN matches={len(keep_m)} rows={len(TR['match'])}", flush=True)

    yTR, yCA, ySE, yTE = y_of(TR), y_of(CA), y_of(SE), y_of(TE)
    tTR = TR["query_ms"].astype(float) / 60000.0
    tCA = CA["query_ms"].astype(float) / 60000.0
    tSE = SE["query_ms"].astype(float) / 60000.0
    tTE = TE["query_ms"].astype(float) / 60000.0
    Xtr, cols, champ_ix = expanded_X(TR["X"], names)
    Xca, _, _ = expanded_X(CA["X"], names)
    Xse, _, _ = expanded_X(SE["X"], names)
    Xte, _, _ = expanded_X(TE["X"], names)
    print(f"rows TRAIN={len(yTR)} CAL={len(yCA)} SEL={len(ySE)} TEST={len(yTE)} feats={len(cols)}", flush=True)

    raw_pred: Dict[str, Dict[str, np.ndarray]] = {}
    meta: Dict[str, Any] = {}

    # --- A2 load wave-1 shared_lgbm if present
    p_w1 = WAVE1 / "models" / "shared_lgbm.joblib"
    if p_w1.is_file():
        print("load wave-1 shared_lgbm…", flush=True)
        pack = joblib.load(p_w1)
        clf = pack["model"]
        raw_pred["A2_shared_lgbm_wave1"] = {
            "cal": predict_lgbm(clf, Xca, cols, champ_ix),
            "sel": predict_lgbm(clf, Xse, cols, champ_ix),
            "test": predict_lgbm(clf, Xte, cols, champ_ix),
        }
        meta["A2_shared_lgbm_wave1"] = dict(source=str(p_w1), precedent="Hodge/CoG LGBM wave-1")
    else:
        print("fit A2 shared_lgbm wave-1 settings…", flush=True)
        t0 = time.time()
        clf = fit_lgbm_wave1(Xtr, yTR, TR["match"], champ_ix, cols)
        print(f"  done {time.time()-t0:.1f}s", flush=True)
        joblib.dump({"model": clf, "cols": cols, "champ_ix": champ_ix, "kind": "A2"}, out / "models" / "A2_shared_lgbm.joblib")
        raw_pred["A2_shared_lgbm_wave1"] = {
            "cal": predict_lgbm(clf, Xca, cols, champ_ix),
            "sel": predict_lgbm(clf, Xse, cols, champ_ix),
            "test": predict_lgbm(clf, Xte, cols, champ_ix),
        }

    # --- B0 per-band
    p_b = WAVE1 / "models" / "per_band_lgbm.joblib"
    if p_b.is_file():
        print("score wave-1 per_band_lgbm…", flush=True)
        band_pack = joblib.load(p_b)

        def score_band(X, t):
            outp = np.full(len(t), np.nan)
            for key, clf_b in band_pack["models"].items():
                if key == "t_0_10":
                    m = t < 10
                elif key == "t_10_20":
                    m = (t >= 10) & (t < 20)
                elif key == "t_20_30":
                    m = (t >= 20) & (t < 30)
                else:
                    m = t >= 30
                if m.any():
                    outp[m] = predict_lgbm(clf_b, X[m], cols, champ_ix)
            return outp

        raw_pred["B0_per_band_lgbm"] = {
            "cal": score_band(Xca, tCA),
            "sel": score_band(Xse, tSE),
            "test": score_band(Xte, tTE),
        }
        meta["B0_per_band_lgbm"] = dict(source=str(p_b), precedent="Choice B ablation")

    # --- A0 logistic
    a0_path = out / "models" / "A0_shared_logistic.joblib"
    if not args.skip_logistic:
        if a0_path.is_file():
            print("load A0 shared logistic…", flush=True)
            pack = joblib.load(a0_path)
            logit_m, keep = pack["model"], pack["keep"]
        else:
            print("fit A0 shared logistic…", flush=True)
            t0 = time.time()
            logit_m, keep = fit_logistic(TR["X"], yTR, TR["match"], names, C=getattr(C, "V_C", 0.01))
            print(f"  done {time.time()-t0:.1f}s", flush=True)
            joblib.dump({"model": logit_m, "keep": keep, "kind": "A0"}, a0_path)
        raw_pred["A0_shared_logistic"] = {
            "cal": predict_logistic(logit_m, CA["X"], keep),
            "sel": predict_logistic(logit_m, SE["X"], keep),
            "test": predict_logistic(logit_m, TE["X"], keep),
        }
        meta["A0_shared_logistic"] = dict(precedent="Maymin; Hodge LR")

    # --- A1 RF
    a1_path = out / "models" / "A1_shared_rf.joblib"
    if not args.skip_rf:
        if a1_path.is_file():
            print("load A1 Random Forest…", flush=True)
            rf = joblib.load(a1_path)["model"]
        else:
            print("fit A1 Random Forest…", flush=True)
            t0 = time.time()
            rf = fit_rf(Xtr, yTR, TR["match"])
            print(f"  done {time.time()-t0:.1f}s", flush=True)
            joblib.dump({"model": rf, "cols": cols, "kind": "A1"}, a1_path)
        raw_pred["A1_shared_rf"] = {
            "cal": rf.predict_proba(Xca)[:, 1],
            "sel": rf.predict_proba(Xse)[:, 1],
            "test": rf.predict_proba(Xte)[:, 1],
        }
        meta["A1_shared_rf"] = dict(precedent="Hodge RF", n_estimators=300, max_depth=20)

    # --- A2b CoG LGBM
    a2b_path = out / "models" / "A2b_shared_lgbm_cog.joblib"
    if not args.skip_cog_lgbm:
        if a2b_path.is_file():
            print("load A2b CoG-hyper LightGBM…", flush=True)
            clf_c = joblib.load(a2b_path)["model"]
        else:
            print("fit A2b CoG-hyper LightGBM…", flush=True)
            t0 = time.time()
            clf_c = fit_lgbm_cog(Xtr, yTR, TR["match"], Xca, yCA, CA["match"], champ_ix, cols)
            print(f"  done {time.time()-t0:.1f}s best_iter={getattr(clf_c, 'best_iteration_', None)}", flush=True)
            joblib.dump({"model": clf_c, "cols": cols, "champ_ix": champ_ix, "kind": "A2b"}, a2b_path)
        raw_pred["A2b_shared_lgbm_cog"] = {
            "cal": predict_lgbm(clf_c, Xca, cols, champ_ix),
            "sel": predict_lgbm(clf_c, Xse, cols, champ_ix),
            "test": predict_lgbm(clf_c, Xte, cols, champ_ix),
        }
        meta["A2b_shared_lgbm_cog"] = dict(
            precedent="CoG MODELS.md LightGBM",
            best_iteration=int(getattr(clf_c, "best_iteration_", 0) or 0),
        )

    # --- A3 MLP
    a3_path = out / "models" / "A3_shared_mlp.joblib"
    if not args.skip_mlp:
        if a3_path.is_file():
            print("load A3 shared MLP…", flush=True)
            mlp = joblib.load(a3_path)
        else:
            print("fit A3 shared MLP…", flush=True)
            t0 = time.time()
            mlp = fit_mlp(Xtr, yTR, TR["match"], Xca, yCA, CA["match"])
            print(f"  done {time.time()-t0:.1f}s val_brier={mlp['best_val_brier']:.4f}", flush=True)
            joblib.dump(mlp, a3_path)
        raw_pred["A3_shared_mlp"] = {
            "cal": predict_mlp(mlp, Xca),
            "sel": predict_mlp(mlp, Xse),
            "test": predict_mlp(mlp, Xte),
        }
        meta["A3_shared_mlp"] = dict(precedent="Kim CoG WP; CoG matched MLP", hidden=mlp["hidden"])

    # --- legacy
    legacy_path = data_root / "outputs" / "full_corpus_training_20260915" / "models" / "v" / "v_final_raw.joblib"
    if legacy_path.is_file():
        print("score legacy logistic…", flush=True)
        legacy = joblib.load(legacy_path)
        raw_pred["legacy_logistic_raw"] = {
            "cal": legacy.predict_matrix(CA["X"], names, C.STATE_VERSION),
            "sel": legacy.predict_matrix(SE["X"], names, C.STATE_VERSION),
            "test": legacy.predict_matrix(TE["X"], names, C.STATE_VERSION),
        }
        meta["legacy_logistic_raw"] = dict(precedent="old sealed V", eligible=False)

    # --- calibrate + select
    print("calibrate + select…", flush=True)
    selection, calibrated_test = {}, {}
    for name, preds in raw_pred.items():
        fin_cal = np.isfinite(preds["cal"])
        if int(fin_cal.sum()) < 1000:
            selection[name] = dict(skipped=True, reason="too few finite cal rows")
            continue
        sig = PosSlopeSigmoid().fit(
            preds["cal"][fin_cal], yCA[fin_cal], match_weights(CA["match"][fin_cal])
        )
        sel_c = np.where(np.isfinite(preds["sel"]), sig.transform(preds["sel"]), np.nan)
        te_c = np.where(np.isfinite(preds["test"]), sig.transform(preds["test"]), np.nan)
        fin_s = np.isfinite(sel_c)
        tb = time_balanced_brier(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s], tSE[fin_s])
        overall = metrics(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s])
        selection[name] = dict(
            L_time=tb["L_time"],
            select_overall=overall,
            select_bands=tb["bands"],
            ineligible=bool(tb.get("ineligible")),
            calib_ok=sig.ok,
            calib=dict(coef=sig.coef_, intercept=sig.intercept_) if sig.ok else None,
            meta=meta.get(name),
        )
        calibrated_test[name] = te_c
        print(f"  {name}: L_time={tb['L_time']:.6f} Brier={overall['brier']:.6f} AUC={overall['auc']:.4f}", flush=True)

    eligible = {
        k: v for k, v in selection.items()
        if "L_time" in v and not str(k).startswith("legacy_") and not v.get("ineligible")
        and np.isfinite(v.get("L_time", float("inf")))
    }
    winner = min(eligible, key=lambda k: (eligible[k]["L_time"], eligible[k]["select_overall"]["logloss"], k))
    print("MECHANICAL WINNER", winner, flush=True)

    test_ledger = {}
    for name, p in calibrated_test.items():
        fin = np.isfinite(p)
        test_ledger[name] = dict(all=metrics(yTE[fin], p[fin], TE["match"][fin]), bands={})
        for lo, hi, bname in BANDS:
            m = band_mask(tTE, lo, hi) & fin
            test_ledger[name]["bands"][bname] = (
                metrics(yTE[m], p[m], TE["match"][m]) if int(m.sum()) >= 50
                else dict(skipped=True, n=int(m.sum()))
            )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V2_CANDIDATE_MATRIX_20260919.md Tier 0–1",
        query_sampling="bucket_only=True",
        selection_rule="min L_time on V_SELECT; tie-break overall logloss; legacy ineligible",
        mechanical_winner=winner,
        a_priori_note="Choice A still default for ΔV until continuity; winner here is WP quality only",
        census=dict(train=len(yTR), cal=len(yCA), select=len(ySE), test=len(yTE), n_features=len(cols)),
        selection=selection,
        test_ledger=test_ledger,
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

    (out / "results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V redesign wave-2 — Tier 0–1 horse-race",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        f"**Mechanical V_SELECT winner:** `{winner}`",
        "",
        "Choice A remains the ΔV default until continuity; this table is **WP quality** only.",
        "",
        "## V_SELECT",
        "",
        "| Candidate | L_time | Brier | logloss | AUC | Precedent |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, sc in sorted(selection.items(), key=lambda kv: kv[1].get("L_time", 9e9) if "L_time" in kv[1] else 9e9):
        if "L_time" not in sc:
            continue
        o = sc["select_overall"]
        prec = (sc.get("meta") or {}).get("precedent", "")
        mark = " **" if name == winner else ""
        lines.append(
            f"| {name}{mark} | {fmt(sc['L_time'], 6)} | {fmt(o['brier'], 6)} | "
            f"{fmt(o['logloss'], 6)} | {fmt(o['auc'], 4)} | {prec} |"
        )
    lines += ["", "## TEST ledger (calibrated) — all rows", "", "| Candidate | Brier | AUC |", "|---|---:|---:|"]
    for name, blk in test_ledger.items():
        a = blk["all"]
        lines.append(f"| {name} | {fmt(a['brier'], 6)} | {fmt(a['auc'], 4)} |")
    lines += [
        "",
        "## Reading",
        "",
        "- Tier 0–1 complete for this run; Tier 2–3 (history/sequence) and Tier 4 (GNN) still open.",
        "- Do not freeze on mechanical winner alone — run DeltaV continuity on top contenders under g o f.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
