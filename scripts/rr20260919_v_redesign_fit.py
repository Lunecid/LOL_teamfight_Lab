#!/usr/bin/env python3
"""V redesign wave-1 fit on the same 210k full-corpus cohort.

Shared time-conditional models (Choice A) + per-band LightGBM comparator (Choice B).
Selection: time-balanced Brier on V_SELECT (alpha_b=1/4); logloss tie-break.
Sealed TEST band ledger + comparison to legacy frozen logistic V.

Data: 문서/LOL_Teamfight outputs/full_corpus_training_20260915 (bucket queries, grid from 120s).
Contract: docs/V_REDESIGN_CONTRACT_20260919.md, docs/V1_TASK_CONTRACT_20260919.md

Writes: outputs/v_redesign_20260919/{results.json,REPORT.md,models/}
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


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup_paths(data_root: Path) -> None:
    scripts = data_root / "scripts"
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(wt))
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("LOL_OUTPUT_ROOT", str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"))


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
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


def band_mask(tmin: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (tmin >= lo) & (tmin < hi)


def time_balanced_brier(y, p, g, tmin) -> Dict[str, Any]:
    """L_time = sum_b alpha_b Brier_b. Missing required band → ineligible (L_time=inf)."""
    parts = {}
    acc = 0.0
    missing = []
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
        return dict(
            L_time=float("inf"),
            alpha=ALPHA,
            bands=parts,
            ineligible=True,
            missing_bands=missing,
            note="required band too small — do not drop alpha_b silently",
        )
    return dict(L_time=float(acc), alpha=ALPHA, bands=parts, ineligible=False)


def expanded_X(X: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float64, copy=False), cols, champ_ix


def fit_lgbm(X, y, g, champ_ix, cols, seed=7):
    import lightgbm as lgb
    import pandas as pd
    w = match_weights(g)
    df = pd.DataFrame(X, columns=list(cols))
    for j in champ_ix:
        df.iloc[:, j] = df.iloc[:, j].astype("category")
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(df, y, sample_weight=w)
    return clf


def fit_logistic_expanded(X_raw, y, g, names, C=0.01):
    """Reuse corpus logistic pipeline (champion one-hot + scaled numerics)."""
    from train.state_value_experiment import logistic
    from train.temporal_winprob import feature_matrix
    Xf, cols = feature_matrix(X_raw, names, "expanded")
    model = logistic(cols, C=C)
    w = match_weights(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xf, y, model__sample_weight=w)
    keep = [names.index(c) for c in cols]
    return model, cols, keep


class PosSlopeSigmoid:
    """Positive-slope 1-d sigmoid on logit(p); falls back to identity if unfit."""

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
        self.ok = True
        self.coef_ = coef
        self.intercept_ = float(lr.intercept_.ravel()[0])
        return self

    def transform(self, p):
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        z = self.coef_ * logit + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))


def predict_lgbm(clf, X, cols, champ_ix):
    import pandas as pd
    df = pd.DataFrame(X, columns=list(cols))
    for j in champ_ix:
        df.iloc[:, j] = df.iloc[:, j].astype("category")
    return clf.predict_proba(df)[:, 1]


def predict_logistic(model, X_raw, keep):
    return model.predict_proba(X_raw[:, keep])[:, 1]


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "v_redesign_20260919")
    ap.add_argument("--skip-logistic", action="store_true", help="LGBM only (faster)")
    ap.add_argument("--skip-band", action="store_true", help="skip Choice B per-band LGBM")
    ap.add_argument("--smoke-frac", type=float, default=0.0, help="if >0, subsample TRAIN matches")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup_paths(data_root)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    import fc20260915_common as C  # noqa: E402
    import fc20260915_data as D  # noqa: E402
    import joblib

    L = D.Layout(False)
    print("load TRAIN…", flush=True)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    print("load V_CAL / V_SELECT / TEST…", flush=True)
    CA = D.load_v_rows(L, "MAIN", ["V_CAL"], bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TR["names"])
    W = D.load_outcomes(
        L, "MAIN", train_roles + ["V_CAL", "V_SELECT", "TEST"],
        purpose="V redesign fit/select/test ledger (prior TEST exposure; exploratory)",
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
    print(f"features={len(cols)} champ_slots={len(champ_ix)} TRAIN rows={len(yTR)} matches={len(np.unique(TR['match']))}", flush=True)

    candidates: Dict[str, Any] = {}
    raw_pred: Dict[str, Dict[str, np.ndarray]] = {}

    # --- Choice A: shared LightGBM
    print("fit shared LightGBM…", flush=True)
    t0 = time.time()
    lgbm = fit_lgbm(Xtr, yTR, TR["match"], champ_ix, cols)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    joblib.dump({"model": lgbm, "cols": cols, "champ_ix": champ_ix, "kind": "shared_lgbm"}, out_dir / "models" / "shared_lgbm.joblib")
    raw_pred["shared_lgbm"] = {
        "cal": predict_lgbm(lgbm, Xca, cols, champ_ix),
        "sel": predict_lgbm(lgbm, Xse, cols, champ_ix),
        "test": predict_lgbm(lgbm, Xte, cols, champ_ix),
    }

    # --- Choice A: shared logistic (legacy family)
    if not args.skip_logistic:
        print("fit shared logistic expanded…", flush=True)
        t0 = time.time()
        logit_m, logit_cols, logit_keep = fit_logistic_expanded(TR["X"], yTR, TR["match"], names, C=C.V_C)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        joblib.dump(
            {"model": logit_m, "cols": logit_cols, "keep": logit_keep, "kind": "shared_logistic"},
            out_dir / "models" / "shared_logistic.joblib",
        )
        raw_pred["shared_logistic"] = {
            "cal": predict_logistic(logit_m, CA["X"], logit_keep),
            "sel": predict_logistic(logit_m, SE["X"], logit_keep),
            "test": predict_logistic(logit_m, TE["X"], logit_keep),
        }

    # --- Choice B: per-band LightGBM
    if not args.skip_band:
        print("fit per-band LightGBM…", flush=True)
        band_models = {}
        p_cal = np.full(len(yCA), np.nan)
        p_sel = np.full(len(ySE), np.nan)
        p_te = np.full(len(yTE), np.nan)
        for lo, hi, name in BANDS:
            mtr = band_mask(tTR, lo, hi)
            print(f"  {name} n={int(mtr.sum())}", flush=True)
            if int(mtr.sum()) < 500:
                continue
            clf = fit_lgbm(Xtr[mtr], yTR[mtr], TR["match"][mtr], champ_ix, cols)
            band_models[name] = clf
            for tag, X, t, dest in (
                ("cal", Xca, tCA, p_cal),
                ("sel", Xse, tSE, p_sel),
                ("test", Xte, tTE, p_te),
            ):
                m = band_mask(t, lo, hi)
                if m.any():
                    dest[m] = predict_lgbm(clf, X[m], cols, champ_ix)
        joblib.dump({"models": band_models, "cols": cols, "kind": "per_band_lgbm"}, out_dir / "models" / "per_band_lgbm.joblib")
        # drop rows without band model
        raw_pred["per_band_lgbm"] = {"cal": p_cal, "sel": p_sel, "test": p_te}

    # --- Legacy frozen V on TEST/SELECT (compare)
    legacy_path = data_root / "outputs" / "full_corpus_training_20260915" / "models" / "v" / "v_final_raw.joblib"
    legacy_scores = {}
    if legacy_path.is_file():
        print("score legacy frozen V…", flush=True)
        legacy = joblib.load(legacy_path)
        # FullCorpusWinProbV2 API
        try:
            p_leg_se = legacy.predict_matrix(SE["X"], names, C.STATE_VERSION)
            p_leg_te = legacy.predict_matrix(TE["X"], names, C.STATE_VERSION)
            legacy_scores = {
                "select": time_balanced_brier(ySE, p_leg_se, SE["match"], tSE),
                "test": {name: metrics(yTE[band_mask(tTE, lo, hi)], p_leg_te[band_mask(tTE, lo, hi)], TE["match"][band_mask(tTE, lo, hi)])
                         if int(band_mask(tTE, lo, hi).sum()) >= 50 else dict(skipped=True)
                         for lo, hi, name in BANDS},
                "test_all": metrics(yTE, p_leg_te, TE["match"]),
            }
            raw_pred["legacy_logistic_raw"] = {"sel": p_leg_se, "test": p_leg_te, "cal": legacy.predict_matrix(CA["X"], names, C.STATE_VERSION)}
        except Exception as e:
            legacy_scores = {"error": repr(e)}

    # --- calibrate + select
    print("calibrate + select…", flush=True)
    selection = {}
    calibrated_test: Dict[str, np.ndarray] = {}
    for name, preds in raw_pred.items():
        if "cal" not in preds or not np.isfinite(preds["cal"]).all():
            # per-band may have nan — mask
            ok = np.isfinite(preds["cal"]) & np.isfinite(preds["sel"])
            if ok.sum() < 1000:
                selection[name] = dict(skipped=True, reason="too many non-finite")
                continue
            cal_p, sel_p = preds["cal"].copy(), preds["sel"].copy()
            # fill nan with 0.5 for calib fit only on finite
            fin = np.isfinite(cal_p)
            sig = PosSlopeSigmoid().fit(cal_p[fin], yCA[fin], match_weights(CA["match"][fin]))
            sel_c = np.where(np.isfinite(sel_p), sig.transform(sel_p), np.nan)
            te_c = np.where(np.isfinite(preds["test"]), sig.transform(preds["test"]), np.nan)
        else:
            sig = PosSlopeSigmoid().fit(preds["cal"], yCA, match_weights(CA["match"]))
            sel_c = sig.transform(preds["sel"])
            te_c = sig.transform(preds["test"])
        # selection on finite sel rows
        fin_s = np.isfinite(sel_c)
        tb = time_balanced_brier(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s], tSE[fin_s])
        overall = metrics(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s])
        selection[name] = dict(
            L_time=tb["L_time"],
            select_overall=overall,
            select_bands=tb["bands"],
            ineligible=bool(tb.get("ineligible")),
            missing_bands=tb.get("missing_bands"),
            calib_ok=sig.ok,
            calib=dict(coef=sig.coef_, intercept=sig.intercept_) if sig.ok else None,
        )
        calibrated_test[name] = te_c
        print(f"  {name}: L_time={tb['L_time']:.6f} Brier={overall['brier']:.6f} AUC={overall['auc']:.4f}", flush=True)

    eligible = {
        k: v for k, v in selection.items()
        if "L_time" in v and not str(k).startswith("legacy_") and not v.get("ineligible")
        and np.isfinite(v.get("L_time", float("inf")))
    }
    if not eligible:
        raise SystemExit("no non-legacy candidates to select")
    winner = min(eligible, key=lambda k: (eligible[k]["L_time"], eligible[k]["select_overall"]["logloss"], k))
    print("WINNER", winner, flush=True)

    # --- TEST ledger
    test_ledger = {}
    for name, p in calibrated_test.items():
        fin = np.isfinite(p)
        test_ledger[name] = dict(
            all=metrics(yTE[fin], p[fin], TE["match"][fin]),
            bands={},
        )
        for lo, hi, bname in BANDS:
            m = band_mask(tTE, lo, hi) & fin
            if int(m.sum()) < 50:
                test_ledger[name]["bands"][bname] = dict(skipped=True, n=int(m.sum()))
            else:
                test_ledger[name]["bands"][bname] = metrics(yTE[m], p[m], TE["match"][m])

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_REDESIGN_CONTRACT_20260919.md",
        cohort="full_corpus_training_20260915 / same 210k",
        query="bucket samples; grid from 120s (pre-2 excluded by design)",
        alpha=ALPHA,
        selection_rule=(
            "min L_time on V_SELECT; tie-break = overall match-weighted logloss on V_SELECT "
            "(not time-balanced logloss — documented wave-1; see V1 §5)"
        ),
        query_sampling="bucket_only=True (corpus is_bucket_sample); not every minute frame",
        winner=winner,
        census=dict(
            train_rows=int(len(yTR)), train_matches=int(len(np.unique(TR["match"]))),
            v_cal_rows=int(len(yCA)), v_select_rows=int(len(ySE)), test_rows=int(len(yTE)),
            n_features=len(cols),
        ),
        selection=selection,
        test_ledger=test_ledger,
        legacy=legacy_scores,
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

    (out_dir / "results.json").write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "winner_manifest.json").write_text(
        json.dumps(dict(winner=winner, generated=payload["generated"], epistemic=ROLE, alpha=ALPHA), indent=2) + "\n",
        encoding="utf-8",
    )

    # REPORT
    lines = [
        "# V redesign wave-1 — same 210k cohort",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        "Choice A = shared time-conditional; Choice B = per-band LGBM.",
        "Select on **V_SELECT** by time-balanced Brier ($\\alpha_b=1/4$); TEST is ledger only.",
        f"**Winner: `{winner}`**",
        "",
        "## Selection (V_SELECT)",
        "",
        "| Candidate | L_time | Brier | logloss | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, sc in selection.items():
        if "L_time" not in sc:
            lines.append(f"| {name} | SKIP | | | |")
            continue
        o = sc["select_overall"]
        mark = " **" if name == winner else ""
        lines.append(
            f"| {name}{mark} | {fmt(sc['L_time'], 6)} | {fmt(o['brier'], 6)} | {fmt(o['logloss'], 6)} | {fmt(o['auc'], 4)} |"
        )
    lines += ["", "## TEST ledger by band (calibrated)", ""]
    for name, block in test_ledger.items():
        lines += [f"### {name}", "", "| Band | n | Brier | logloss | AUC |", "|---|---:|---:|---:|---:|"]
        allb = block["all"]
        lines.append(f"| all | {allb['n']} | {fmt(allb['brier'], 6)} | {fmt(allb['logloss'], 6)} | {fmt(allb['auc'], 4)} |")
        for bname, sc in block["bands"].items():
            if sc.get("skipped"):
                continue
            lines.append(f"| {bname} | {sc['n']} | {fmt(sc['brier'], 6)} | {fmt(sc['logloss'], 6)} | {fmt(sc['auc'], 4)} |")
        lines.append("")
    if legacy_scores and "test_all" in legacy_scores:
        la = legacy_scores["test_all"]
        lines += [
            "## Legacy frozen logistic (raw) on TEST",
            "",
            f"all: Brier {fmt(la['brier'], 6)} AUC {fmt(la['auc'], 4)}",
            "",
        ]
    lines += [
        "## Reading",
        "",
        "- Winner is frozen for downstream ΔV checks only after V-4 continuity suite.",
        "- Do not rebuild SVI/q until V-3/V-4 complete.",
        "- Pre-2 minutes excluded (corpus grid starts at 120s) — same as ToG/full-corpus V.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines).replace("$\\alpha_b=1/4$", "alpha_b=1/4"), encoding="utf-8")
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
