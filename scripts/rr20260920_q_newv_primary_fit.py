#!/usr/bin/env python3
"""Primary q table on new-V OOF labels (fit85 MLP lineage).

Train on TRAIN_oof_h90; select on Q_SELECT (match-weighted Brier);
freeze winner + rules; score TEST. Baselines: constant, b(p), PT, logit, LGBM.

Contract: docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
OUT = REPO / "outputs" / "q_newv_fit85_20260920"
ROLE = "EXPLORATORY_Q_NEWV_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"


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


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def bootstrap_delta_brier(y, p_q, p_pt, g, reps=800, seed=7):
    """ΔBrier = Brier(q) - Brier(PT); negative ⇒ q better."""
    g = np.asarray(g).astype(str)
    matches, inv = np.unique(g, return_inverse=True)
    n_m = len(matches)
    _, counts = np.unique(inv, return_counts=True)
    w = (1.0 / counts[inv]).astype(np.float64)
    eq = w * (p_q - y) ** 2
    ep = w * (p_pt - y) ** 2
    sum_q = np.zeros(n_m)
    sum_p = np.zeros(n_m)
    np.add.at(sum_q, inv, eq)
    np.add.at(sum_p, inv, ep)
    obs = float((sum_q.sum() - sum_p.sum()) / n_m)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        samp = rng.integers(0, n_m, size=n_m)
        draws.append(float((sum_q[samp].sum() - sum_p[samp].sum()) / n_m))
    draws = np.asarray(draws)
    return dict(
        estimate=obs,
        ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        p_gt0=float(np.mean(draws > 0)),
    )


def expanded_pre(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    return X[:, keep].astype(np.float64, copy=False), cols


def load_role_pack(lab_path: Path, D, L, roles, data_root: Path):
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
        y=lab["Y_SVI"][idx_l].astype(np.int8),
        g=lab["match"][idx_l].astype(str),
        tmin=lab["s"][idx_l].astype(float) / 60000.0,
        B40=lab["B40"][idx_l].astype(bool) if "B40" in lab.files else None,
        names=names,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-reps", type=int, default=800)
    args = ap.parse_args(argv)

    train_lab = LAB / "TRAIN_oof_h90.npz"
    if not train_lab.is_file():
        raise SystemExit(f"missing {train_lab}")
    for r in ("Q_CAL", "Q_SELECT", "TEST"):
        if not (LAB / f"{r}_h90.npz").is_file():
            raise SystemExit(f"missing {LAB / f'{r}_h90.npz'}")

    data_root = _data_root()
    _setup(data_root)
    import joblib
    import fc20260915_common as C
    import fc20260915_data as D
    import lightgbm as lgb
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load packs…", flush=True)
    TR = load_role_pack(train_lab, D, L, train_roles, data_root)
    CA = load_role_pack(LAB / "Q_CAL_h90.npz", D, L, ["Q_CAL"], data_root)
    SE = load_role_pack(LAB / "Q_SELECT_h90.npz", D, L, ["Q_SELECT"], data_root)
    TE = load_role_pack(LAB / "TEST_h90.npz", D, L, ["TEST"], data_root)
    assert TR["cols"] == SE["cols"] == TE["cols"] == CA["cols"]
    cols = TR["cols"]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    print(
        f"  TRAIN={len(TR['y'])} Q_CAL={len(CA['y'])} Q_SELECT={len(SE['y'])} TEST={len(TE['y'])}",
        flush=True,
    )

    w_tr = match_weights(TR["g"])
    models: Dict[str, Any] = {}

    base_rate = float(np.average(TR["y"], weights=w_tr))
    models["constant"] = dict(kind="constant", p=base_rate)

    print("fit b(p)…", flush=True)
    bp = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    bp.fit(TR["p_pre"].reshape(-1, 1), TR["y"], logisticregression__sample_weight=w_tr)
    models["b_p"] = dict(kind="sklearn", pipe=bp)

    print("fit PT…", flush=True)
    pt = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pt.fit(
        np.column_stack([TR["p_pre"], TR["tmin"]]),
        TR["y"],
        logisticregression__sample_weight=w_tr,
    )
    models["PT"] = dict(kind="sklearn", pipe=pt)

    print("fit logit_state…", flush=True)
    # numeric-only for logit stability (drop raw champ ids)
    num_ix = [j for j, c in enumerate(cols) if not c.endswith("_champion_id")]
    Xtr_num = np.column_stack([TR["X"][:, num_ix], TR["p_pre"].reshape(-1, 1)])
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    logit.fit(Xtr_num, TR["y"], logisticregression__sample_weight=w_tr)
    models["logit_state"] = dict(kind="logit_num", pipe=logit, num_ix=num_ix)

    print("fit lgbm_state…", flush=True)
    feat_cols = cols + ["p_pre_V"]

    def make_df(X, p):
        df = pd.DataFrame(np.column_stack([X, p.reshape(-1, 1)]), columns=feat_cols)
        for j in champ_ix:
            df[feat_cols[j]] = df[feat_cols[j]].astype("category")
        return df

    lgbm = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=7,
        n_jobs=4,
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm.fit(
            make_df(TR["X"], TR["p_pre"]),
            TR["y"],
            sample_weight=w_tr,
            eval_set=[(make_df(CA["X"], CA["p_pre"]), CA["y"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    models["lgbm_state"] = dict(kind="lgbm", model=lgbm)

    def predict(name, pack):
        m = models[name]
        if m["kind"] == "constant":
            return np.full(len(pack["y"]), m["p"], dtype=float)
        if name == "b_p":
            return m["pipe"].predict_proba(pack["p_pre"].reshape(-1, 1))[:, 1]
        if name == "PT":
            return m["pipe"].predict_proba(np.column_stack([pack["p_pre"], pack["tmin"]]))[:, 1]
        if name == "logit_state":
            X = np.column_stack([pack["X"][:, m["num_ix"]], pack["p_pre"].reshape(-1, 1)])
            return m["pipe"].predict_proba(X)[:, 1]
        if name == "lgbm_state":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return m["model"].predict_proba(make_df(pack["X"], pack["p_pre"]))[:, 1]
        raise KeyError(name)

    print("select on Q_SELECT…", flush=True)
    sel_scores = {n: metrics(SE["y"], predict(n, SE), SE["g"]) for n in models}
    winner = min(sel_scores, key=lambda n: (sel_scores[n]["brier"], sel_scores[n]["logloss"], n))
    print(f"  winner={winner} Brier={fmt(sel_scores[winner]['brier'])}", flush=True)

    # Freeze selection (do not peek TEST for choice)
    freeze = dict(
        selected_q=winner,
        selection_metric="match_weighted_Brier_on_Q_SELECT_T",
        selection_scores=sel_scores,
        label_train="TRAIN_oof_h90.npz",
        label_eval="frozen_fit85 for Q_CAL/Q_SELECT/TEST",
        engagement_def="T_h90",
        direction_threshold="Y=1[delta_V>0]; exact0 -> Y=0",
        primary_contrast="DeltaBrier = Brier(q)-Brier(PT) on identical TEST rows",
        tau=0.001,
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    (OUT / "selection_freeze.json").write_text(json.dumps(scrub(freeze), indent=2) + "\n", encoding="utf-8")

    print("score TEST (frozen)…", flush=True)
    test_pred = {n: predict(n, TE) for n in models}
    test_scores = {n: metrics(TE["y"], test_pred[n], TE["g"]) for n in models}
    b40 = TE["B40"] if TE["B40"] is not None else ((TE["p_pre"] >= 0.4) & (TE["p_pre"] <= 0.6))
    b40_scores = {
        n: metrics(TE["y"][b40], test_pred[n][b40], TE["g"][b40]) for n in models if int(b40.sum()) > 50
    }

    boot = {
        f"{winner}_minus_PT": bootstrap_delta_brier(
            TE["y"], test_pred[winner], test_pred["PT"], TE["g"], args.boot_reps
        ),
        f"{winner}_minus_b_p": bootstrap_delta_brier(
            TE["y"], test_pred[winner], test_pred["b_p"], TE["g"], args.boot_reps
        ),
    }
    if int(b40.sum()) > 50:
        boot[f"B40_{winner}_minus_PT"] = bootstrap_delta_brier(
            TE["y"][b40], test_pred[winner][b40], test_pred["PT"][b40], TE["g"][b40], args.boot_reps
        )

    mdir = OUT / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["b_p"]["pipe"], mdir / "bp.joblib")
    joblib.dump(models["PT"]["pipe"], mdir / "pt.joblib")
    joblib.dump(models["logit_state"], mdir / "logit_state.joblib")
    joblib.dump({"model": lgbm, "feat_cols": feat_cols, "champ_ix": champ_ix}, mdir / "lgbm_state.joblib")

    payload = dict(
        generated=freeze["generated"],
        epistemic=ROLE,
        contract="docs/Q_PREDICTION_DESIGN_CONTRACT_20260920.md",
        census=dict(
            train=len(TR["y"]),
            q_cal=len(CA["y"]),
            q_select=len(SE["y"]),
            test=len(TE["y"]),
            test_B40=int(b40.sum()),
        ),
        Q_SELECT=sel_scores,
        selected_q=winner,
        TEST=test_scores,
        TEST_B40=b40_scores,
        bootstrap_delta_brier=boot,
        note=(
            "Selected on Q_SELECT only; TEST scored after freeze. "
            "OOF TRAIN labels; eval labels from frozen fit85. "
            "DeltaBrier = Brier(q)-Brier(PT)."
        ),
    )
    (OUT / "primary_table.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    # markdown report
    lines = [
        "# q primary table — new V OOF labels (fit85 MLP lineage)",
        "",
        f"Generated: {payload['generated']}",
        f"**Selected q (Q_SELECT Brier):** `{winner}`",
        "",
        "Contract: [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)",
        "",
        "## Q_SELECT (selection)",
        "",
        "| Model | Brier | logloss | AUC |",
        "|---|---:|---:|---:|",
    ]
    for n, sc in sorted(sel_scores.items(), key=lambda kv: kv[1]["brier"]):
        mark = " ← selected" if n == winner else ""
        lines.append(f"| {n}{mark} | {fmt(sc['brier'])} | {fmt(sc['logloss'])} | {fmt(sc['auc'])} |")
    lines += [
        "",
        "## TEST 15.16 T (after freeze)",
        "",
        "| Model | Brier | logloss | AUC |",
        "|---|---:|---:|---:|",
    ]
    for n, sc in sorted(test_scores.items(), key=lambda kv: kv[1]["brier"]):
        lines.append(f"| {n} | {fmt(sc['brier'])} | {fmt(sc['logloss'])} | {fmt(sc['auc'])} |")
    d = boot[f"{winner}_minus_PT"]
    lines += [
        "",
        f"### Primary contrast: ΔBrier(`{winner}` − PT)",
        "",
        f"- estimate={fmt(d['estimate'], 5)}  (negative ⇒ q better)",
        f"- 95% CI=[{fmt(d['ci95'][0], 5)}, {fmt(d['ci95'][1], 5)}]",
        f"- τ=0.001 interpretation threshold (a priori)",
        "",
        "## B40 (new p_pre)",
        "",
    ]
    if b40_scores:
        lines += ["| Model | Brier | AUC |", "|---|---:|---:|"]
        for n, sc in sorted(b40_scores.items(), key=lambda kv: kv[1]["brier"]):
            lines.append(f"| {n} | {fmt(sc['brier'])} | {fmt(sc['auc'])} |")
    md = REPO / "docs" / "Q_NEWV_FIT85_PRIMARY_20260920.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "primary_table.json", md, flush=True)
    print(
        f"TEST {winner} Brier={fmt(test_scores[winner]['brier'])} "
        f"PT={fmt(test_scores['PT']['brier'])} "
        f"ΔBrier={fmt(d['estimate'], 5)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
