#!/usr/bin/env python3
"""Lean primary SVI table under NEW shared_lgbm V lineage.

Fits on new-V MAIN_TRAIN OOF labels; selects lightly on VAL; seals on 15.16 T.
Baselines: constant, b(p)=logistic(p_pre), PT=logistic([p_pre,time]), LightGBM(state+p_pre).

Does not touch outputs/svi_*_20260919 (old V). Writes outputs/svi_newv_primary_20260919/.
"""
from __future__ import annotations

import argparse
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
ROLE = "EXPLORATORY_NEW_V_LINEAGE_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"
V_DIR = REPO / "outputs" / "v_redesign_20260919"


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(data_root / "worktrees" / "engagement-state-value"))
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


def expanded_pre(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    return X[:, keep].astype(np.float64, copy=False), cols


def load_pack(lab_path: Path, data_root: Path, L, D, set_id: str, roles: List[str], cohort_path: Optional[Path] = None):
    """Align labels with engagement X_pre; optional cohort filter (1=T)."""
    lab = np.load(lab_path, allow_pickle=False)
    E = D.load_engagements(L, set_id, roles, states=True, counts=False)
    names = list(E["names"])
    e_key = {(a, int(b)): i for i, (a, b) in enumerate(zip(E["match"].astype(str).tolist(), E["s"].astype(np.int64).tolist()))}
    idx_e, idx_l = [], []
    for i, (a, b) in enumerate(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist())):
        j = e_key.get((a, int(b)))
        if j is not None:
            idx_l.append(i)
            idx_e.append(j)
    idx_l = np.asarray(idx_l, int)
    idx_e = np.asarray(idx_e, int)
    v = lab["valid_h90"][idx_l] == 1
    idx_l, idx_e = idx_l[v], idx_e[v]
    if cohort_path and cohort_path.is_file():
        coh = np.load(cohort_path, allow_pickle=False)
        # cohort aligned to OLD label order — map via keys
        c_map = {(a, int(b)): int(c) for a, b, c in zip(
            coh["match"].astype(str).tolist() if "match" in coh.files else lab["match"].astype(str).tolist(),
            # prefer s from cohort file if present
            (coh["s"] if "s" in coh.files else lab["s"]).astype(np.int64).tolist(),
            coh["cohort"].tolist(),
        )}
        # If cohort has no match, assume same order as official MAIN_TEST_labels
        if "match" not in coh.files:
            old = np.load(
                data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
                allow_pickle=False,
            )
            c_map = {(a, int(b)): int(c) for a, b, c in zip(
                old["match"].astype(str).tolist(), old["s"].astype(np.int64).tolist(), coh["cohort"].tolist()
            )}
        keep = np.array([c_map.get((a, int(b)), 0) == 1 for a, b in zip(
            lab["match"][idx_l].astype(str).tolist(), lab["s"][idx_l].astype(np.int64).tolist()
        )], dtype=bool)
        idx_l, idx_e = idx_l[keep], idx_e[keep]

    Xpre, cols = expanded_pre(E["X_pre"][idx_e], names)
    p_pre = lab["p_pre"][idx_l].astype(float)
    y = lab["Y_h90"][idx_l].astype(np.int8)
    g = lab["match"][idx_l].astype(str)
    tmin = lab["s"][idx_l].astype(float) / 60000.0
    return dict(X=Xpre, cols=cols, p_pre=p_pre, y=y, g=g, tmin=tmin, names=names)


def bootstrap_delta(y, p_a, p_b, g, reps=800, seed=7):
    g = np.asarray(g).astype(str)
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
    draws = []
    for _ in range(reps):
        samp = rng.integers(0, n_m, size=n_m)
        draws.append(float((sum_a[samp].sum() - sum_b[samp].sum()) / n_m))
    draws = np.asarray(draws)
    return dict(
        estimate=obs,
        boot_mean=float(draws.mean()),
        ci95=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        p_gt0=float(np.mean(draws > 0)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_newv_primary_20260919")
    ap.add_argument("--boot-reps", type=int, default=800)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import pandas as pd

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load TRAIN pack…", flush=True)
    TR = load_pack(V_DIR / "labels" / "MAIN_TRAIN_labels.npz", data_root, L, D, "MAIN", train_roles)
    print("load VAL pack…", flush=True)
    # Prefer Q_SELECT for light selection; fall back to full VAL file if present
    val_lab = V_DIR / "labels" / "MAIN_VAL_labels.npz"
    if not val_lab.is_file():
        raise SystemExit("missing MAIN_VAL_labels.npz — run relabel --group validation first")
    VA = load_pack(val_lab, data_root, L, D, "MAIN", list(C.VAL_ROLES))
    print("load TEST T pack…", flush=True)
    coh = data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz"
    TE = load_pack(
        V_DIR / "labels" / "MAIN_TEST_labels.npz", data_root, L, D, "MAIN", ["TEST"],
        cohort_path=coh if coh.is_file() else None,
    )
    print(f"  TRAIN n={len(TR['y'])} VAL n={len(VA['y'])} TEST_T n={len(TE['y'])}", flush=True)

    # Align feature cols
    assert TR["cols"] == TE["cols"] == VA["cols"]
    cols = TR["cols"]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]

    def add_p(X, p):
        return np.column_stack([X, p.reshape(-1, 1)])

    models = {}
    # constant
    base_rate = float(np.average(TR["y"], weights=match_weights(TR["g"])))
    models["constant"] = dict(kind="constant", p=base_rate)

    # b(p)
    print("fit b(p)…", flush=True)
    bp = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    bp.fit(TR["p_pre"].reshape(-1, 1), TR["y"], logisticregression__sample_weight=match_weights(TR["g"]))
    models["b(p)"] = dict(kind="sklearn", pipe=bp, x="p_pre")

    # PT
    print("fit PT…", flush=True)
    pt = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    Xpt_tr = np.column_stack([TR["p_pre"], TR["tmin"]])
    pt.fit(Xpt_tr, TR["y"], logisticregression__sample_weight=match_weights(TR["g"]))
    models["PT"] = dict(kind="sklearn", pipe=pt, x="p_t")

    # LightGBM on state + p_pre
    print("fit LightGBM…", flush=True)
    feat_cols = cols + ["p_pre_V"]
    def make_df(X, p):
        df = pd.DataFrame(add_p(X, p), columns=feat_cols)
        for j in champ_ix:
            df[feat_cols[j]] = df[feat_cols[j]].astype("category")
        return df
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=7, n_jobs=4, verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm.fit(make_df(TR["X"], TR["p_pre"]), TR["y"], sample_weight=match_weights(TR["g"]))
    models["LightGBM"] = dict(kind="lgbm", model=lgbm)

    def predict(name, pack):
        m = models[name]
        if m["kind"] == "constant":
            return np.full(len(pack["y"]), m["p"])
        if name == "b(p)":
            return m["pipe"].predict_proba(pack["p_pre"].reshape(-1, 1))[:, 1]
        if name == "PT":
            return m["pipe"].predict_proba(np.column_stack([pack["p_pre"], pack["tmin"]]))[:, 1]
        if name == "LightGBM":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return m["model"].predict_proba(make_df(pack["X"], pack["p_pre"]))[:, 1]
        raise KeyError(name)

    # VAL selection by Brier
    print("select on VAL…", flush=True)
    val_scores = {n: metrics(VA["y"], predict(n, VA), VA["g"]) for n in models}
    winner = min(val_scores, key=lambda n: (val_scores[n]["brier"], val_scores[n]["logloss"], n))

    # TEST T ledger
    print("score TEST T…", flush=True)
    test_pred = {n: predict(n, TE) for n in models}
    test_scores = {n: metrics(TE["y"], test_pred[n], TE["g"]) for n in models}
    b40 = (TE["p_pre"] >= 0.4) & (TE["p_pre"] <= 0.6)
    b40_scores = {n: metrics(TE["y"][b40], test_pred[n][b40], TE["g"][b40]) for n in models}

    # bootstrap key contrasts
    boot = {
        "LightGBM_minus_PT": bootstrap_delta(TE["y"], test_pred["LightGBM"], test_pred["PT"], TE["g"], args.boot_reps),
        "LightGBM_minus_bp": bootstrap_delta(TE["y"], test_pred["LightGBM"], test_pred["b(p)"], TE["g"], args.boot_reps),
        "PT_minus_bp": bootstrap_delta(TE["y"], test_pred["PT"], test_pred["b(p)"], TE["g"], args.boot_reps),
        "B40_LightGBM_minus_PT": bootstrap_delta(
            TE["y"][b40], test_pred["LightGBM"][b40], test_pred["PT"][b40], TE["g"][b40], args.boot_reps
        ),
    }

    # save models
    mdir = out / "models"
    mdir.mkdir(exist_ok=True)
    joblib.dump(models["b(p)"]["pipe"], mdir / "bp.joblib")
    joblib.dump(models["PT"]["pipe"], mdir / "pt.joblib")
    joblib.dump({"model": lgbm, "feat_cols": feat_cols, "champ_ix": champ_ix}, mdir / "lgbm.joblib")

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        v_lineage="shared_lgbm_freeze_provisional",
        freeze_manifest="outputs/v_redesign_20260919/freeze_manifest.json",
        label_dir="outputs/v_redesign_20260919/labels",
        horizon="h90",
        sample="15.16 T (cohort==1) on new-V labels",
        census=dict(train=len(TR["y"]), val=len(VA["y"]), test_T=len(TE["y"]), test_T_B40=int(b40.sum())),
        val_selection=val_scores,
        val_winner=winner,
        test_T=test_scores,
        test_T_B40=b40_scores,
        bootstrap_delta_brier=boot,
        note="Lean slate (constant/b(p)/PT/LGBM). Deep models deferred. Old svi_* not comparable as same-V evidence.",
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
        "# Primary SVI table — NEW V lineage (shared_lgbm)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        f"**V freeze:** `shared_lgbm` — do not cite with old `svi_*_20260919` as one table.",
        "",
        f"VAL winner (min Brier): **{winner}**",
        "",
        f"TEST T n={payload['census']['test_T']} · B40 n={payload['census']['test_T_B40']}",
        "",
        "## 15.16 T (all)",
        "",
        "| Model | Brier | logloss | AUC |",
        "|---|---:|---:|---:|",
    ]
    for n in ["constant", "b(p)", "PT", "LightGBM"]:
        s = test_scores[n]
        mark = " **" if n == winner else ""
        lines.append(f"| {n}{mark} | {fmt(s['brier'])} | {fmt(s['logloss'])} | {fmt(s['auc'], 4)} |")
    lines += [
        "",
        "## B40 (p_pre ∈ [0.4, 0.6])",
        "",
        "| Model | Brier | logloss | AUC |",
        "|---|---:|---:|---:|",
    ]
    for n in ["constant", "b(p)", "PT", "LightGBM"]:
        s = b40_scores[n]
        lines.append(f"| {n} | {fmt(s['brier'])} | {fmt(s['logloss'])} | {fmt(s['auc'], 4)} |")
    lines += ["", "## Bootstrap ΔBrier (row − col; negative ⇒ row better)", ""]
    for k, v in boot.items():
        lines.append(
            f"- `{k}`: est={fmt(v['estimate'])} CI95=[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}] "
            f"P(Δ>0)={fmt(v['p_gt0'], 3)}"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Labels from new-V OOF TRAIN + final TEST (`v_redesign_20260919/labels`).",
        "- This is a lean restart slate; MLP/TabM/Tier-B not re-fit yet.",
        "- Compare magnitudes to old primary table only as *lineage shift*, not as a horse-race.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "REPORT.md")
    print("TEST T LightGBM Brier", test_scores["LightGBM"]["brier"], "PT", test_scores["PT"]["brier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
