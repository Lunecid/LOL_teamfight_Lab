#!/usr/bin/env python3
"""RR6a — CORP / reliability diagnostics for V→W and q→SVI (fit85 frozen).

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §10
Lit: docs/ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md (Dimitriadis–Gneiting–Jordan CORP)

Locks:
  - Same fit85 MLP evaluator bundle; no new per-band V or TEST recalibrator
  - q = frozen logit_state (RR12 identity g_q); CORP is DIAGNOSTIC ONLY
  - Do not apply TEST-fit isotonic as improved frozen-model transfer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from forecast_diagnostics import (  # noqa: E402
    metrics_bundle,
)

BUNDLE = (
    REPO
    / "outputs"
    / "v_redesign_wave4_corrected_20260919"
    / "evaluators"
    / "A_MLP_expanded_evaluator.joblib"
)
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
RR12 = REPO / "outputs" / "review_response_rr12_20260920"
OUT = REPO / "outputs" / "review_response_rr6a_corp_20260920"
ROLE = "EXPLORATORY_RR6A_CORP_DIAGNOSTIC_PRIOR_TEST_EXPOSURE"

# Design §10: [2,10), [10,20), [20,30), [30,∞)
TIME_BANDS = (
    (2.0, 10.0, "t_2_10"),
    (10.0, 20.0, "t_10_20"),
    (20.0, 30.0, "t_20_30"),
    (30.0, 1e9, "t_30_inf"),
)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


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


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def ece_from_bins(rows: List[Dict[str, Any]]) -> float:
    """Match-weighted ECE: sum (w_b / W) |mean_p - mean_y| using bin w_sum when present."""
    tot_w = sum(float(r.get("w_sum", r.get("n", 0)) or 0.0) for r in rows)
    if tot_w <= 0:
        return float("nan")
    acc = 0.0
    for r in rows:
        wb = float(r.get("w_sum", r.get("n", 0)) or 0.0)
        if wb <= 0:
            continue
        mp, my = r.get("mean_p"), r.get("mean_y")
        if mp is None or my is None or (isinstance(mp, float) and math.isnan(mp)):
            continue
        acc += (wb / tot_w) * abs(float(mp) - float(my))
    return float(acc)


def cell_report(y: np.ndarray, p: np.ndarray, g: np.ndarray, label: str) -> Dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    g = np.asarray(g)
    ok = np.isfinite(p) & np.isfinite(y)
    y, p, g = y[ok], p[ok], g[ok]
    if len(y) == 0:
        return dict(label=label, n=0, skipped=True)
    bun = metrics_bundle(y, p, g)
    bins = bun["fixed_bin_reliability"]
    return dict(
        label=label,
        n=bun["n"],
        n_matches=bun["n_matches"],
        brier=bun["brier"],
        logloss=bun["logloss"],
        auc=bun["auc"],
        ECE=ece_from_bins(bins),
        calib_logit=bun["calib_logit"],
        CORP=dict(
            UNC=bun["corp"]["UNC"],
            DSC=bun["corp"]["DSC"],
            MCB=bun["corp"]["MCB"],
            brier=bun["corp"]["brier"],
            reconstruction_gap=bun["corp"]["reconstruction_gap"],
            mean_y=bun["corp"]["mean_y"],
            mean_p=bun["corp"]["mean_p"],
        ),
        reliability_curve=bun["corp"]["reliability_curve"],
        fixed_bin_reliability=bins,
        note=bun["corp"]["note"],
    )


def by_time_bands(y, p, g, tmin) -> Dict[str, Any]:
    out = {}
    tmin = np.asarray(tmin, float)
    for lo, hi, name in TIME_BANDS:
        m = (tmin >= lo) & (tmin < hi)
        out[name] = cell_report(y[m], p[m], g[m], name)
    # also overall on band-eligible rows (t>=2)
    m2 = tmin >= 2.0
    out["overall_t_ge_2"] = cell_report(y[m2], p[m2], g[m2], "overall_t_ge_2")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-timeline", action="store_true")
    ap.add_argument("--skip-engagement", action="store_true")
    ap.add_argument("--skip-q", action="store_true")
    args = ap.parse_args(argv)

    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")
    pred_path = RR12 / "prediction_table.npz"
    if not pred_path.is_file() and not args.skip_q:
        raise SystemExit(f"missing {pred_path} (run RR12 first)")

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    OUT.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    print("load evaluator…", flush=True)
    ev = load_evaluator(BUNDLE)
    bundle_sha = sha16(BUNDLE)

    payload: Dict[str, Any] = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §10 RR6a",
        lit="docs/ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md",
        bundle=str(BUNDLE.relative_to(REPO)).replace("\\", "/"),
        bundle_sha16=bundle_sha,
        fit_scope=ev.get("fit_scope"),
        time_bands=[dict(lo=a, hi=b, name=n) for a, b, n in TIME_BANDS],
        warning=(
            "CORP isotonic is fit on the evaluation sample for diagnosis only. "
            "Do not claim TEST-fit recalibration improves frozen V/q generalization."
        ),
    )

    # --- V → W timeline ---
    if not args.skip_timeline:
        print("score TEST timeline buckets V→W…", flush=True)
        TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
        Wmap = D.load_outcomes(L, "MAIN", ["TEST"], purpose="RR6a CORP V→W")
        y = np.asarray([Wmap[m][0] for m in TE["match"].tolist()], dtype=np.float64)
        g = TE["match"].astype(str)
        tmin = TE["query_ms"].astype(float) / 60000.0
        p = predict_calibrated(ev, TE["X"])
        payload["V_to_W_timeline"] = dict(
            overall=cell_report(y, p, g, "overall"),
            by_time=by_time_bands(y, p, g, tmin),
            n_raw=int(len(y)),
        )
        print(
            f"  timeline overall Brier={fmt(payload['V_to_W_timeline']['overall']['brier'])} "
            f"MCB={fmt(payload['V_to_W_timeline']['overall']['CORP']['MCB'])} "
            f"DSC={fmt(payload['V_to_W_timeline']['overall']['CORP']['DSC'])}",
            flush=True,
        )

    # --- V → W engagement pre / post ---
    if not args.skip_engagement:
        print("score TEST engagement pre/post V→W…", flush=True)
        lab = np.load(LAB / "TEST_h90.npz", allow_pickle=False)
        E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
        Wmap = D.load_outcomes(L, "MAIN", ["TEST"], purpose="RR6a eng V→W")
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
        ok &= E["pre_ok"][idx_e] == 1
        ok &= E["valid_h90"][idx_e] == 1
        idx_l, idx_e = idx_l[ok], idx_e[ok]

        g = lab["match"][idx_l].astype(str)
        yW = np.asarray([Wmap[m][0] for m in g.tolist()], dtype=np.float64)
        t_pre = E["s"][idx_e].astype(float) / 60000.0
        t_post = E["endpoint_h90"][idx_e].astype(float) / 60000.0
        # prefer stored label probs to avoid rescoring drift; still score for CORP path consistency
        p_pre = lab["p_pre"][idx_l].astype(float)
        p_post = lab["p_post"][idx_l].astype(float)
        # verify vs live bundle on a sample
        live_pre = predict_calibrated(ev, E["X_pre"][idx_e])
        live_post = predict_calibrated(ev, E["X_post_h90"][idx_e])
        parity = dict(
            mean_abs_pre=float(np.mean(np.abs(live_pre - p_pre))),
            mean_abs_post=float(np.mean(np.abs(live_post - p_post))),
            p95_abs_pre=float(np.quantile(np.abs(live_pre - p_pre), 0.95)),
            p95_abs_post=float(np.quantile(np.abs(live_post - p_post), 0.95)),
        )
        payload["V_to_W_engagement"] = dict(
            n=int(len(yW)),
            label_vs_live_parity=parity,
            pre_overall=cell_report(yW, p_pre, g, "pre_overall"),
            post_overall=cell_report(yW, p_post, g, "post_overall"),
            pre_by_time=by_time_bands(yW, p_pre, g, t_pre),
            post_by_time=by_time_bands(yW, p_post, g, t_post),
            note=(
                "post time-band uses actual endpoint time; "
                "paired pre→post comparison is separate from band-pooled tables."
            ),
        )
        print(
            f"  eng pre Brier={fmt(payload['V_to_W_engagement']['pre_overall']['brier'])} "
            f"post Brier={fmt(payload['V_to_W_engagement']['post_overall']['brier'])} "
            f"parity mean|Δpre|={fmt(parity['mean_abs_pre'])}",
            flush=True,
        )

    # --- q → SVI (and PT_flex for contrast) ---
    if not args.skip_q:
        print("CORP q→SVI / PT_flex→SVI…", flush=True)
        pred = np.load(pred_path, allow_pickle=False)
        y = pred["y"].astype(float)
        g = pred["match"].astype(str)
        tmin = pred["s"].astype(float) / 60000.0
        b40 = pred["B40"].astype(bool)
        q = pred["p_q_base"].astype(float)
        pt = pred["p_PT_flex"].astype(float)

        def slice_block(mask, tag):
            return dict(
                q=cell_report(y[mask], q[mask], g[mask], f"q_{tag}"),
                PT_flex=cell_report(y[mask], pt[mask], g[mask], f"PT_flex_{tag}"),
            )

        payload["q_to_SVI"] = dict(
            source="outputs/review_response_rr12_20260920/prediction_table.npz",
            calibrator="identity (RR12 baseline_selection)",
            all_T=slice_block(np.ones(len(y), bool), "all_T"),
            B40=slice_block(b40, "B40"),
            outside_B40=slice_block(~b40, "outside_B40"),
            by_time_q=by_time_bands(y, q, g, tmin),
            by_time_PT_flex=by_time_bands(y, pt, g, tmin),
        )
        print(
            f"  q all-T Brier={fmt(payload['q_to_SVI']['all_T']['q']['brier'])} "
            f"MCB={fmt(payload['q_to_SVI']['all_T']['q']['CORP']['MCB'])} "
            f"DSC={fmt(payload['q_to_SVI']['all_T']['q']['CORP']['DSC'])}",
            flush=True,
        )

    (OUT / "rr6a_corp_results.json").write_text(
        json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8"
    )

    # CSV-ish summaries
    rows = []

    def add_row(stage, cell, block):
        if not block or block.get("skipped"):
            return
        rows.append(
            dict(
                stage=stage,
                cell=cell,
                n=block["n"],
                n_matches=block["n_matches"],
                brier=block["brier"],
                logloss=block["logloss"],
                auc=block["auc"],
                ECE=block["ECE"],
                MCB=block["CORP"]["MCB"],
                DSC=block["CORP"]["DSC"],
                UNC=block["CORP"]["UNC"],
                calib_slope=(block["calib_logit"].get("slope") if block["calib_logit"].get("ok") else None),
                calib_intercept=(block["calib_logit"].get("intercept") if block["calib_logit"].get("ok") else None),
            )
        )

    if "V_to_W_timeline" in payload:
        add_row("V_to_W_timeline", "overall", payload["V_to_W_timeline"]["overall"])
        for k, v in payload["V_to_W_timeline"]["by_time"].items():
            add_row("V_to_W_timeline", k, v)
    if "V_to_W_engagement" in payload:
        add_row("V_to_W_eng_pre", "overall", payload["V_to_W_engagement"]["pre_overall"])
        add_row("V_to_W_eng_post", "overall", payload["V_to_W_engagement"]["post_overall"])
        for k, v in payload["V_to_W_engagement"]["pre_by_time"].items():
            add_row("V_to_W_eng_pre", k, v)
        for k, v in payload["V_to_W_engagement"]["post_by_time"].items():
            add_row("V_to_W_eng_post", k, v)
    if "q_to_SVI" in payload:
        for scope in ("all_T", "B40", "outside_B40"):
            add_row("q_to_SVI", f"q_{scope}", payload["q_to_SVI"][scope]["q"])
            add_row("q_to_SVI", f"PT_flex_{scope}", payload["q_to_SVI"][scope]["PT_flex"])
        for k, v in payload["q_to_SVI"]["by_time_q"].items():
            add_row("q_to_SVI", f"q_{k}", v)
        for k, v in payload["q_to_SVI"]["by_time_PT_flex"].items():
            add_row("q_to_SVI", f"PT_flex_{k}", v)

    # write calibration_by_time.csv
    import csv

    csv_path = OUT / "calibration_by_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "cell",
                "n",
                "n_matches",
                "brier",
                "logloss",
                "auc",
                "ECE",
                "MCB",
                "DSC",
                "UNC",
                "calib_slope",
                "calib_intercept",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown report
    md = [
        "# RR6a — CORP calibration diagnostics (V→W and q→SVI)",
        "",
        f"Generated: {payload['generated']}",
        f"**Bundle:** `{payload['bundle']}` sha16=`{bundle_sha}`",
        "",
        "Diagnostic only (Dimitriadis–Gneiting–Jordan CORP + Gneiting–Raftery scoring). "
        "**Do not** treat TEST-fit isotonic as a frozen-model upgrade.",
        "",
        "## Summary table",
        "",
        "| Stage | Cell | n | Brier | MCB | DSC | UNC | ECE | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['stage']} | {r['cell']} | {r['n']} | {fmt(r['brier'])} | {fmt(r['MCB'])} | "
            f"{fmt(r['DSC'])} | {fmt(r['UNC'])} | {fmt(r['ECE'])} | {fmt(r['auc'])} |"
        )
    md += [
        "",
        "## CORP definition (corrected)",
        "",
        r"MCB \(=\) BS − BS\(_\mathrm{iso}\), DSC \(=\) UNC − BS\(_\mathrm{iso}\). "
        "Exact identity BS = MCB − DSC + UNC (asserted in code). "
        r"Do **not** use \(\langle(p-p^*)^2\rangle\) as MCB.",
        "",
        "## Reading",
        "",
        "- Report **relative** MCB vs DSC contributions; do not claim “calibration is fine” from small absolute MCB alone "
        r"(ΔBrier vs PT is \(O(10^{-3})\), so MCB differences of that order matter).",
        "- 15.16 MAIN diagnostics do **not** explain 16.x external transfer; RRX must score V→W and q→SVI externally.",
        "- V→W and q→SVI are **separate** stages — never pooled into one reliability diagram.",
        "- Artifacts: `outputs/review_response_rr6a_corp_20260920/`",
        "",
    ]
    (REPO / "docs" / "REVIEW_RESPONSE_RR6A_CORP_20260920.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "REVIEW_RESPONSE_RR6A_CORP_20260920.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "rr6a_corp_results.json", csv_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
