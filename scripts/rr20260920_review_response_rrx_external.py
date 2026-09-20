#!/usr/bin/env python3
"""RRX — external dual-stage diagnostics: frozen V→W and q→SVI (score-only).

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §11
Locks:
  - No EXT refit of V/q/PT_flex; constant = MAIN TRAIN prior
  - Fail-closed if cohort file missing (do not treat all engagements as T)
  - Separate V→W vs q→SVI; do not infer EXT failure mode from 15.16 CORP
  - Main claim cohorts: KR/NA1 16.13; pilots reported separately
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from forecast_diagnostics import metrics_bundle  # noqa: E402

WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
QOUT = REPO / "outputs" / "q_newv_fit85_20260920"
RR12 = REPO / "outputs" / "review_response_rr12_20260920"
OUT = REPO / "outputs" / "review_response_rrx_external_20260920"
ROLE = "TRANSFER_DUAL_STAGE_SCORE_ONLY_NO_REFIT"

COHORTS = (
    ("KR_16.13", "EXT_KR_16.13", "KR 16.13", False),
    ("NA1_16.13", "EXT_NA1_16.13", "NA1 16.13", False),
    ("KR_16.15", "EXT_KR_16.15", "KR 16.15", False),
    ("KR_16.14_pilot", "EXT_KR_16.14_pilot", "KR 16.14 pilot", True),
)

TIME_BANDS = (
    (2.0, 10.0, "t_2_10"),
    (10.0, 20.0, "t_10_20"),
    (20.0, 30.0, "t_20_30"),
    (30.0, 1e9, "t_30_inf"),
)

from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402


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


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


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


def load_ext_t_failclosed(data_root: Path, D, L, set_id: str, coh_name: str) -> Dict[str, Any]:
    coh_path = (
        data_root
        / "outputs"
        / "cohort_role_training_20260915"
        / "cohorts"
        / f"{coh_name}_cohort.npz"
    )
    if not coh_path.is_file():
        return dict(ok=False, reason=f"missing_cohort:{coh_path.name}", set_id=set_id)
    E = D.load_engagements(L, set_id, None, states=True, counts=False)
    coh = np.load(coh_path, allow_pickle=False)
    if "match" not in coh.files or "s" not in coh.files:
        return dict(ok=False, reason="cohort_missing_match_s_keys", set_id=set_id)
    m = coh["cohort"] == 1
    lab_keys = set(zip(coh["match"][m].astype(str).tolist(), coh["s"][m].astype(np.int64).tolist()))
    em, es = E["match"].astype(str), E["s"].astype(np.int64)
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    keep &= np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    if int(keep.sum()) == 0:
        return dict(ok=False, reason="empty_after_cohort_filter", set_id=set_id)
    X, cols = expanded_pre(E["X_pre"][keep], list(E["names"]))
    return dict(
        ok=True,
        set_id=set_id,
        X=X,
        cols=cols,
        X_pre_raw=E["X_pre"][keep],
        X_post=E["X_post_h90"][keep],
        match=em[keep],
        s=es[keep],
        tmin=es[keep].astype(float) / 60000.0,
        names=list(E["names"]),
        n_raw=int(len(em)),
        n_kept=int(keep.sum()),
        cohort_path=str(coh_path),
    )


def cell_corp(y, p, g, label: str) -> Dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    g = np.asarray(g)
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return dict(label=label, n=0, skipped=True)
    bun = metrics_bundle(y[ok], p[ok], g[ok])
    return dict(
        label=label,
        n=bun["n"],
        n_matches=bun["n_matches"],
        brier=bun["brier"],
        logloss=bun["logloss"],
        auc=bun["auc"],
        CORP=dict(MCB=bun["corp"]["MCB"], DSC=bun["corp"]["DSC"], UNC=bun["corp"]["UNC"], gap=bun["corp"]["reconstruction_gap"]),
    )


def by_time(y, p, g, tmin) -> Dict[str, Any]:
    out = {}
    for lo, hi, name in TIME_BANDS:
        m = (tmin >= lo) & (tmin < hi)
        out[name] = cell_corp(y[m], p[m], g[m], name)
    return out


def predict_q_frozen(logit_obj, X, p_pre, cols_expected=None):
    num_ix = logit_obj["num_ix"]
    Xnum = np.column_stack([X[:, num_ix], np.asarray(p_pre, float).reshape(-1, 1)])
    return logit_obj["pipe"].predict_proba(Xnum)[:, 1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import joblib

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    for need in (BUNDLE, QOUT / "models" / "logit_state.joblib", RR12 / "models" / "PT_flex.joblib"):
        if not Path(need).is_file():
            raise SystemExit(f"missing {need}")

    # PT_flex was pickled under __main__.PTFlexFeatures when rr12 ran as a script
    import rr20260920_review_response_rr12 as rr12

    sys.modules["__main__"].PTFlexFeatures = rr12.PTFlexFeatures

    OUT.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    ev = load_evaluator(BUNDLE)
    logit = joblib.load(QOUT / "models" / "logit_state.joblib")
    pt_flex = joblib.load(RR12 / "models" / "PT_flex.joblib")
    # TRAIN prior for constant (not EXT rate)
    train_meta = json.loads((QOUT / "labels" / "TRAIN_oof_h90_meta.json").read_text(encoding="utf-8"))
    train_prior = float(train_meta.get("P_SVI", train_meta.get("p_pos", 0.5)))

    # MAIN 15.16 reference row from RR12 prediction table (for side-by-side)
    main_ref = None
    pred_path = RR12 / "prediction_table.npz"
    if pred_path.is_file():
        pred = np.load(pred_path, allow_pickle=False)
        y = pred["y"].astype(float)
        g = pred["match"].astype(str)
        q = pred["p_q_base"].astype(float)
        pt = pred["p_PT_flex"].astype(float)
        main_ref = dict(
            q=cell_corp(y, q, g, "MAIN_TEST_q"),
            PT_flex=cell_corp(y, pt, g, "MAIN_TEST_PT_flex"),
            delta_brier_q_minus_PT=float(cell_corp(y, q, g, "q")["brier"] - cell_corp(y, pt, g, "pt")["brier"]),
        )

    results = {}
    rows = []
    for set_id, coh_name, label, pilot in COHORTS:
        print(f"RRX {label}…", flush=True)
        pack = load_ext_t_failclosed(data_root, D, L, set_id, coh_name)
        if not pack.get("ok"):
            results[set_id] = dict(label=label, pilot=pilot, ok=False, reason=pack.get("reason"))
            print(f"  FAIL-CLOSED: {pack.get('reason')}", flush=True)
            continue

        # --- V → W ---
        Wmap = None
        try:
            W_raw = D.load_outcomes(L, set_id, None, purpose=f"RRX V→W {set_id}")
            Wmap = {str(k): int(v[0]) if isinstance(v, (tuple, list, np.ndarray)) else int(v) for k, v in W_raw.items()}
        except Exception as e:
            results[set_id] = dict(label=label, pilot=pilot, ok=False, reason=f"outcomes_load:{e}")
            print(f"  outcomes fail: {e}", flush=True)
            continue

        g = pack["match"].astype(str)
        yW = np.asarray([Wmap.get(m, -1) for m in g.tolist()], dtype=float)
        sealed = yW >= 0
        p_pre = predict_calibrated(ev, pack["X_pre_raw"])
        p_post = predict_calibrated(ev, pack["X_post"])
        tmin = pack["tmin"]

        V_pre = cell_corp(yW[sealed], p_pre[sealed], g[sealed], "V_pre") if sealed.any() else dict(n=0)
        V_post = cell_corp(yW[sealed], p_post[sealed], g[sealed], "V_post") if sealed.any() else dict(n=0)
        V_pre_time = by_time(yW[sealed], p_pre[sealed], g[sealed], tmin[sealed]) if sealed.any() else {}

        # --- q → SVI ---
        dV = p_post - p_pre
        yS = (dV > 0).astype(float)
        b40 = (p_pre >= 0.40) & (p_pre <= 0.60)
        q = predict_q_frozen(logit, pack["X"], p_pre)
        pt = pt_flex.predict_proba(np.column_stack([p_pre, tmin]))[:, 1]
        const = np.full(len(yS), train_prior)

        q_all = cell_corp(yS, q, g, "q")
        pt_all = cell_corp(yS, pt, g, "PT_flex")
        c_all = cell_corp(yS, const, g, "constant_TRAIN_prior")
        q_b40 = cell_corp(yS[b40], q[b40], g[b40], "q_B40") if b40.any() else dict(n=0)
        pt_b40 = cell_corp(yS[b40], pt[b40], g[b40], "PT_B40") if b40.any() else dict(n=0)

        d_brier = float(q_all["brier"] - pt_all["brier"]) if q_all.get("n") and pt_all.get("n") else float("nan")

        block = dict(
            label=label,
            pilot=pilot,
            ok=True,
            n=int(len(yS)),
            n_matches=int(len(np.unique(g))),
            n_sealed_W=int(sealed.sum()),
            P_SVI=float(np.average(yS, weights=match_weights(g))),
            V_to_W=dict(pre=V_pre, post=V_post, pre_by_time=V_pre_time),
            q_to_SVI=dict(
                constant_TRAIN_prior=c_all,
                q=q_all,
                PT_flex=pt_all,
                q_B40=q_b40,
                PT_flex_B40=pt_b40,
                by_time_q=by_time(yS, q, g, tmin),
                by_time_PT_flex=by_time(yS, pt, g, tmin),
                delta_brier_q_minus_PT_flex=d_brier,
            ),
            reading=(
                "Dual-stage: V→W quality and q→SVI lift are separate. "
                "Do not attribute ΔBrier sign to calibration alone without CORP components on THIS cohort."
            ),
        )
        results[set_id] = block
        rows.append(
            dict(
                cohort=label,
                pilot=pilot,
                n=block["n"],
                P_SVI=block["P_SVI"],
                V_pre_brier=V_pre.get("brier"),
                V_pre_auc=V_pre.get("auc"),
                V_pre_MCB=V_pre.get("CORP", {}).get("MCB"),
                V_pre_DSC=V_pre.get("CORP", {}).get("DSC"),
                q_brier=q_all.get("brier"),
                PT_flex_brier=pt_all.get("brier"),
                delta_brier_q_minus_PT=d_brier,
                q_MCB=q_all.get("CORP", {}).get("MCB"),
                q_DSC=q_all.get("CORP", {}).get("DSC"),
                PT_MCB=pt_all.get("CORP", {}).get("MCB"),
                PT_DSC=pt_all.get("CORP", {}).get("DSC"),
                q_auc=q_all.get("auc"),
            )
        )
        print(
            f"  n={block['n']} V_pre Brier={fmt(V_pre.get('brier'))} "
            f"dBrier(q-PT_flex)={fmt(d_brier)} qMCB={fmt(q_all.get('CORP',{}).get('MCB'))} "
            f"qDSC={fmt(q_all.get('CORP',{}).get('DSC'))}",
            flush=True,
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §11",
        train_constant_prior=train_prior,
        MAIN_15_16_q_vs_PT_flex_ref=main_ref,
        cohorts=results,
        reading=(
            "Preserve KR/NA1 16.13 loss of q lift. Pilots/16.15 not pooled into success. "
            "15.16 CORP does not explain EXT; components reported per cohort."
        ),
    )
    (OUT / "rrx_external_results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    with (OUT / "transfer_dual_stage.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md = [
        "# RRX — external dual-stage (V→W and q→SVI)",
        "",
        f"Generated: {payload['generated']}",
        f"**TRAIN constant prior (for EXT constant baseline):** {fmt(train_prior, 4)}",
        "",
        "Score-only. Fail-closed without cohort keys. "
        "**Do not** infer external failure mode from 15.16 MAIN CORP alone.",
        "",
        "## Summary",
        "",
        "| Cohort | pilot | n | V_pre Brier | V_pre AUC | ΔBrier(q−PT_flex) | q MCB | q DSC | PT MCB | PT DSC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['cohort']} | {r['pilot']} | {r['n']} | {fmt(r['V_pre_brier'])} | {fmt(r['V_pre_auc'])} | "
            f"{fmt(r['delta_brier_q_minus_PT'])} | {fmt(r['q_MCB'])} | {fmt(r['q_DSC'])} | "
            f"{fmt(r['PT_MCB'])} | {fmt(r['PT_DSC'])} |"
        )
    failed = [k for k, v in results.items() if not v.get("ok")]
    if failed:
        md += ["", "### Fail-closed", ""]
        for k in failed:
            md.append(f"- `{k}`: {results[k].get('reason')}")
    md += [
        "",
        "## Reading",
        "",
        "- Main external claim cohorts remain KR/NA1 16.13 (q lift lost vs PT).",
        "- Compare V→W CORP vs q→SVI CORP **on the same external rows** before attributing cause.",
        f"- Artifacts: `{OUT.relative_to(REPO).as_posix()}/`",
        "",
    ]
    doc = REPO / "docs" / "REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md"
    doc.write_text("\n".join(md), encoding="utf-8")
    (OUT / "REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "rrx_external_results.json", doc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
