#!/usr/bin/env python3
"""MLP fit85 evaluator continuity + TEST band ledger (freeze-prep P1).

Primary scorer: A_MLP_expanded evaluator_bundle (g∘f∘T).
Peers: wave-2 A0; wave-4 corrected Logistic Expanded (if model present).

Checks (shared-model friendly — no D_switch band swap required):
  1) TEST consecutive bucket |ΔV| interior vs minute-boundary crossings
  2) Engagement pre→post h90: ΔV summary + sign agree vs peers; V→W quality at pre/end
  3) TEST time-band Brier / logloss / AUC on the same bundle

Writes: outputs/v_redesign_wave4_corrected_20260919/continuity_*
        docs/CONTINUITY_LEDGER_MLP_FIT85_20260920.md
        docs/TEST_BAND_LEDGER_MLP_FIT85_20260920.md
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
sys.path.insert(0, str(REPO / "scripts"))
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
BAND_EDGES = (10.0, 20.0, 30.0)
BANDS = ((0.0, 10.0, "t_0_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
WAVE2 = REPO / "outputs" / "v_redesign_wave2_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"

from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402
from v_redesign_feature_adapters import (  # noqa: E402
    FeatureSchema,
    ProfileBundle,
    mean_one_match_weights,
    match_holdout_mask,
)


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


def crosses_boundary(t0: float, t1: float) -> bool:
    for e in BAND_EDGES:
        if (t0 < e) != (t1 < e):
            return True
    return False


def summarize_abs(x: np.ndarray) -> Dict[str, Any]:
    if len(x) == 0:
        return dict(n=0)
    return dict(
        n=int(len(x)),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p90=float(np.quantile(x, 0.9)),
        p99=float(np.quantile(x, 0.99)),
        frac_gt_0_02=float(np.mean(x > 0.02)),
        frac_gt_0_05=float(np.mean(x > 0.05)),
    )


def consecutive_deltas(match, tmin, p) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(match)
    t = np.asarray(tmin, dtype=float)
    pr = np.asarray(p, dtype=float)
    _, codes = np.unique(m.astype(str), return_inverse=True)
    order = np.lexsort((t, codes))
    t, pr, codes = t[order], pr[order], codes[order]
    interior, boundary = [], []
    i, n = 0, len(t)
    while i < n:
        j = i + 1
        while j < n and codes[j] == codes[i]:
            j += 1
        for k in range(i, j - 1):
            d = abs(float(pr[k + 1] - pr[k]))
            if crosses_boundary(float(t[k]), float(t[k + 1])):
                boundary.append(d)
            else:
                interior.append(d)
        i = j
    return np.asarray(interior, float), np.asarray(boundary, float)


def jump_block(g, tmin, p) -> Dict[str, Any]:
    fin = np.isfinite(p)
    interior, boundary = consecutive_deltas(g[fin], tmin[fin], p[fin])
    return dict(
        interior=summarize_abs(interior),
        boundary=summarize_abs(boundary),
        boundary_minus_interior_mean=(
            float(np.mean(boundary) - np.mean(interior)) if len(boundary) and len(interior) else None
        ),
    )


def metrics(y, p, g) -> Dict[str, Any]:
    w = mean_one_match_weights(g)
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


def band_metrics(y, p, g, tmin) -> Dict[str, Any]:
    out = {}
    for lo, hi, name in BANDS:
        m = (tmin >= lo) & (tmin < hi) & np.isfinite(p)
        if int(m.sum()) < 50:
            out[name] = dict(skipped=True, n=int(m.sum()))
            continue
        out[name] = metrics(y[m], p[m], g[m])
    return out


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def predict_a0(data_root, X_raw, names):
    import joblib
    import fc20260915_common as C

    a0 = joblib.load(WAVE2 / "models" / "A0_shared_logistic.joblib")
    w2 = json.loads((WAVE2 / "results.json").read_text(encoding="utf-8"))
    cal = w2["selection"]["A0_shared_logistic"]["calib"]
    from v_redesign_evaluator_bundle import PosSlopeSigmoid

    raw = a0["model"].predict_proba(X_raw[:, a0["keep"]])[:, 1]
    return PosSlopeSigmoid.from_dict(cal).transform(raw)


def predict_lr_expanded(X_raw, bun, model):
    from v_redesign_evaluator_bundle import PosSlopeSigmoid

    w4 = json.loads((WAVE4 / "results.json").read_text(encoding="utf-8"))
    cal = w4["selection"]["A_LR_expanded"]["calib"]
    raw = model.predict_proba(bun.matrix_onehot(X_raw))[:, 1]
    return PosSlopeSigmoid.from_dict(cal).transform(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    print("load evaluator bundle…", flush=True)
    ev = load_evaluator(BUNDLE)
    L = D.Layout(False)
    print("load TEST buckets + outcomes…", flush=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TE["names"])
    W = D.load_outcomes(L, "MAIN", ["TEST"], purpose="MLP fit85 continuity")
    yTE = np.asarray([W[m][0] for m in TE["match"].tolist()], dtype=np.int64)
    tTE = TE["query_ms"].astype(float) / 60000.0
    gTE = TE["match"].astype(str)

    print("score MLP bundle on TEST…", flush=True)
    p_mlp = predict_calibrated(ev, TE["X"])
    jumps = jump_block(gTE, tTE, p_mlp)
    bands = band_metrics(yTE, p_mlp, gTE, tTE)
    overall = metrics(yTE[np.isfinite(p_mlp)], p_mlp[np.isfinite(p_mlp)], gTE[np.isfinite(p_mlp)])
    print("  TEST overall Brier", fmt(overall["brier"]), "AUC", fmt(overall["auc"]), flush=True)
    print("  jump excess", jumps.get("boundary_minus_interior_mean"), flush=True)

    # peers
    peers: Dict[str, np.ndarray] = {}
    print("score peers…", flush=True)
    peers["A0_wave2"] = predict_a0(data_root, TE["X"], names)
    lr_path = WAVE4 / "models" / "A_LR_expanded.joblib"
    if lr_path.is_file():
        train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
        TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
        schema = FeatureSchema.from_names(TR["names"])
        stop = match_holdout_mask(TR["match"], 0.15, seed=7)
        bun = ProfileBundle(schema, "expanded").fit(TR["X"][~stop])
        lr = joblib.load(lr_path)["model"]
        peers["A_LR_expanded_fit85"] = predict_lr_expanded(TE["X"], bun, lr)

    # engagements
    print("engagement continuity…", flush=True)
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False,
    )
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    m_lab = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    pre_ok = E["pre_ok"] == 1 if "pre_ok" in E else np.ones(len(em), dtype=bool)
    valid = E["valid_h90"] == 1 if "valid_h90" in E else np.ones(len(em), dtype=bool)
    lab_keys = set(
        zip(lab["match"][m_lab].astype(str).tolist(), lab["s"][m_lab].astype(np.int64).tolist())
    )
    keep = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= pre_ok & valid
    Xpre, Xpost = E["X_pre"][keep], E["X_post_h90"][keep]
    m_eng = em[keep]
    y_eng = np.asarray([W[m][0] for m in m_eng.tolist()], dtype=np.int64)

    p_pre = predict_calibrated(ev, Xpre)
    p_post = predict_calibrated(ev, Xpost)
    dV = p_post - p_pre
    eng = {
        "n": int(len(dV)),
        "deltaV": summarize_abs(np.abs(dV)),
        "mean_deltaV": float(np.mean(dV)),
        "frac_exact_zero": float(np.mean(dV == 0)),
        "frac_abs_lt_1e-3": float(np.mean(np.abs(dV) < 1e-3)),
        "V_pre_vs_W": metrics(y_eng, p_pre, m_eng),
        "V_post_vs_W": metrics(y_eng, p_post, m_eng),
    }

    def signs(pre, post):
        return np.sign(post - pre)

    s_mlp = signs(p_pre, p_post)
    sign_agree = {}
    for name, p_peer_all in peers.items():
        # peers scored on TEST buckets only — need engagement scores
        if name == "A0_wave2":
            pre_p = predict_a0(data_root, Xpre, names)
            post_p = predict_a0(data_root, Xpost, names)
        else:
            pre_p = predict_lr_expanded(Xpre, bun, lr)
            post_p = predict_lr_expanded(Xpost, bun, lr)
        s = signs(pre_p, post_p)
        m = np.isfinite(s_mlp) & np.isfinite(s) & (s_mlp != 0) & (s != 0)
        sign_agree[f"MLP_vs_{name}"] = float(np.mean(s_mlp[m] == s[m])) if m.any() else None
        print(f"  sign MLP vs {name}: {fmt(sign_agree[f'MLP_vs_{name}'])}", flush=True)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        evaluator="A_MLP_expanded_evaluator.joblib",
        fit_scope=ev.get("fit_scope"),
        evaluation_map="V = g(f(T(X))) via evaluator_bundle",
        role_of_test="DIAGNOSTIC_AND_REPORTING_NOT_SELECTION",
        consecutive_jumps_calibrated={"A_MLP_expanded": jumps},
        test_overall=overall,
        test_bands=bands,
        engagement=eng,
        sign_agree=sign_agree,
        reading=(
            "Shared-model continuity: boundary excess is a warning signal on stored queries, "
            "not D_switch. Engagement ΔV measures change suitability; matching A0 is not the goal."
        ),
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

    out_json = WAVE4 / "continuity_mlp_fit85_results.json"
    out_json.write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # docs
    cont_md = REPO / "docs" / "CONTINUITY_LEDGER_MLP_FIT85_20260920.md"
    lines = [
        "# Continuity ledger — A_MLP_expanded fit85 bundle",
        "",
        f"Generated: {payload['generated']}",
        f"**Evaluator:** `{payload['evaluator']}`",
        f"**Fit scope:** `{payload['fit_scope']}`",
        "",
        "Primary: change-measurement suitability of \(\\widehat{V}_{\\mathrm{fit85}}\). "
        "Peers (A0, corrected LR) are comparison only — not selection targets.",
        "",
        "## 1. Consecutive TEST |ΔV| (calibrated)",
        "",
        f"| interior mean | boundary mean | excess (b−i) | P(|Δ|>0.05) boundary |",
        f"|---:|---:|---:|---:|",
        f"| {fmt(jumps['interior'].get('mean'))} | {fmt(jumps['boundary'].get('mean'))} | "
        f"{fmt(jumps.get('boundary_minus_interior_mean'))} | {fmt(jumps['boundary'].get('frac_gt_0_05'))} |",
        "",
        f"## 2. Engagement pre→post (h90, n={eng['n']})",
        "",
        f"- mean |ΔV|={fmt(eng['deltaV'].get('mean'))}; median={fmt(eng['deltaV'].get('median'))}; "
        f"P(|ΔV|>0.05)={fmt(eng['deltaV'].get('frac_gt_0_05'))}",
        f"- frac exact 0={fmt(eng['frac_exact_zero'])}; frac |ΔV|<1e-3={fmt(eng['frac_abs_lt_1e-3'])}",
        f"- V_pre→W: Brier={fmt(eng['V_pre_vs_W']['brier'])} AUC={fmt(eng['V_pre_vs_W']['auc'])}",
        f"- V_post→W: Brier={fmt(eng['V_post_vs_W']['brier'])} AUC={fmt(eng['V_post_vs_W']['auc'])}",
        "",
        "### Sign(ΔV) agree (nonzero)",
        "",
    ]
    for k, v in sign_agree.items():
        lines.append(f"- {k}: {fmt(v)}")
    lines += [
        "",
        "## Reading",
        "",
        "- Boundary excess on bucket queries ≠ proof of artifact; document as diagnostic.",
        "- Engagement V→W links training queries to application times.",
        "- Do not prefer a peer merely for higher sign agree with A0.",
        "",
    ]
    cont_md.write_text("\n".join(lines), encoding="utf-8")

    band_md = REPO / "docs" / "TEST_BAND_LEDGER_MLP_FIT85_20260920.md"
    bl = [
        "# TEST band ledger — A_MLP_expanded fit85",
        "",
        f"Generated: {payload['generated']}",
        "",
        f"**Overall:** Brier={fmt(overall['brier'])} AUC={fmt(overall['auc'])} "
        f"n={overall['n']} matches={overall['n_matches']}",
        "",
        "| Band | n | matches | Brier | logloss | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, _, name in BANDS:
        b = bands[name]
        if b.get("skipped"):
            bl.append(f"| {name} | {b.get('n', 0)} | — | SKIP | — | — |")
        else:
            bl.append(
                f"| {name} | {b['n']} | {b['n_matches']} | {fmt(b['brier'])} | "
                f"{fmt(b['logloss'])} | {fmt(b['auc'])} |"
            )
    bl += [
        "",
        "Reporting only — do not retune per-band models from TEST.",
        "",
    ]
    band_md.write_text("\n".join(bl), encoding="utf-8")

    print("wrote", cont_md)
    print("wrote", band_md)
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
