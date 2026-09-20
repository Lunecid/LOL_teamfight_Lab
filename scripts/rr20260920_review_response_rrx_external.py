#!/usr/bin/env python3
"""RRX — external dual-stage diagnostics: frozen V→W and q→SVI (score-only).

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §11
Locks:
  - No EXT refit of V/q/PT_flex; constant = MAIN TRAIN prior
  - Fail-closed if cohort file missing (do not treat all engagements as T)
  - V→W and q→SVI on identical common-valid rows
  - Separate stages; do not infer EXT failure from 15.16 CORP
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
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
LAB = QOUT / "labels"
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
        CORP=dict(
            MCB=bun["corp"]["MCB"],
            DSC=bun["corp"]["DSC"],
            UNC=bun["corp"]["UNC"],
            gap=bun["corp"]["reconstruction_gap"],
        ),
    )


def by_time(y, p, g, tmin) -> Dict[str, Any]:
    out = {}
    for lo, hi, name in TIME_BANDS:
        m = (tmin >= lo) & (tmin < hi)
        out[name] = cell_corp(y[m], p[m], g[m], name)
    return out


def predict_q_frozen(logit_obj, X, p_pre):
    num_ix = logit_obj["num_ix"]
    Xnum = np.column_stack([X[:, num_ix], np.asarray(p_pre, float).reshape(-1, 1)])
    return logit_obj["pipe"].predict_proba(Xnum)[:, 1]


def reference_main_cols(D, L) -> List[str]:
    """MAIN TEST expanded_pre column order used when fitting frozen q."""
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    _, cols = expanded_pre(E["X_pre"][:1], list(E["names"]))
    return cols


def check_feature_order(ref_cols: List[str], ext_cols: List[str], num_ix: Sequence[int]) -> Dict[str, Any]:
    ok_len = len(ref_cols) == len(ext_cols)
    ok_names = list(ref_cols) == list(ext_cols)
    ok_ix = bool(num_ix) and max(num_ix) < len(ext_cols)
    return dict(
        ok=bool(ok_len and ok_names and ok_ix),
        n_ref=len(ref_cols),
        n_ext=len(ext_cols),
        names_match=ok_names,
        num_ix_in_range=ok_ix,
        ref_sha16=hashlib.sha256("|".join(ref_cols).encode()).hexdigest()[:16],
        ext_sha16=hashlib.sha256("|".join(ext_cols).encode()).hexdigest()[:16],
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import joblib

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    for need in (BUNDLE, QOUT / "models" / "logit_state.joblib", RR12 / "models" / "PT_flex.joblib"):
        if not Path(need).is_file():
            raise SystemExit(f"missing {need}")

    import rr20260920_review_response_rr12 as rr12

    sys.modules["__main__"].PTFlexFeatures = rr12.PTFlexFeatures

    OUT.mkdir(parents=True, exist_ok=True)
    L = D.Layout(False)
    print("resolve MAIN reference feature order…", flush=True)
    ref_cols = reference_main_cols(D, L)

    ev = load_evaluator(BUNDLE)
    logit = joblib.load(QOUT / "models" / "logit_state.joblib")
    pt_flex = joblib.load(RR12 / "models" / "PT_flex.joblib")
    train_meta = json.loads((LAB / "TRAIN_oof_h90_meta.json").read_text(encoding="utf-8"))
    train_prior = float(train_meta.get("P_SVI", train_meta.get("p_pos", 0.5)))

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
            delta_brier_q_minus_PT=float(
                cell_corp(y, q, g, "q")["brier"] - cell_corp(y, pt, g, "pt")["brier"]
            ),
        )

    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for set_id, coh_name, label, pilot in COHORTS:
        print(f"RRX {label}…", flush=True)
        pack = load_ext_t_failclosed(data_root, D, L, set_id, coh_name)
        if not pack.get("ok"):
            results[set_id] = dict(label=label, pilot=pilot, ok=False, reason=pack.get("reason"))
            print(f"  FAIL-CLOSED: {pack.get('reason')}", flush=True)
            continue

        feat_check = check_feature_order(ref_cols, list(pack["cols"]), logit["num_ix"])
        if not feat_check["ok"]:
            results[set_id] = dict(
                label=label, pilot=pilot, ok=False, reason="feature_order_mismatch", feature_check=feat_check
            )
            print("  FAIL-CLOSED: feature_order_mismatch", flush=True)
            continue

        try:
            W_raw = D.load_outcomes(L, set_id, None, purpose=f"RRX V→W {set_id}")
            Wmap = {
                str(k): int(v[0]) if isinstance(v, (tuple, list, np.ndarray)) else int(v)
                for k, v in W_raw.items()
            }
        except Exception as e:
            results[set_id] = dict(label=label, pilot=pilot, ok=False, reason=f"outcomes_load:{e}")
            print(f"  outcomes fail: {e}", flush=True)
            continue

        g_all = pack["match"].astype(str)
        yW_all = np.asarray([Wmap.get(m, -1) for m in g_all.tolist()], dtype=float)
        p_pre_all = predict_calibrated(ev, pack["X_pre_raw"])
        p_post_all = predict_calibrated(ev, pack["X_post"])
        tmin_all = pack["tmin"]
        q_raw = predict_q_frozen(logit, pack["X"], p_pre_all)
        pt_raw = pt_flex.predict_proba(np.column_stack([p_pre_all, tmin_all]))[:, 1]

        common = (
            (yW_all >= 0)
            & np.isfinite(p_pre_all)
            & np.isfinite(p_post_all)
            & np.isfinite(q_raw)
            & np.isfinite(pt_raw)
        )
        align = dict(
            n_loaded=int(len(g_all)),
            n_sealed_W=int((yW_all >= 0).sum()),
            n_finite_V=int((np.isfinite(p_pre_all) & np.isfinite(p_post_all)).sum()),
            n_finite_q_pt=int((np.isfinite(q_raw) & np.isfinite(pt_raw)).sum()),
            n_common=int(common.sum()),
            n_sealed_equals_n_common=bool(int((yW_all >= 0).sum()) == int(common.sum())),
            n_common_equals_n_loaded=bool(int(common.sum()) == int(len(g_all))),
            feature_check=feat_check,
        )
        if not common.any():
            results[set_id] = dict(label=label, pilot=pilot, ok=False, reason="empty_common_valid", align=align)
            continue

        g = g_all[common]
        yW = yW_all[common]
        p_pre = p_pre_all[common]
        p_post = p_post_all[common]
        tmin = tmin_all[common]
        q = q_raw[common]
        pt = pt_raw[common]
        yS = ((p_post - p_pre) > 0).astype(float)
        b40 = (p_pre >= 0.40) & (p_pre <= 0.60)
        const = np.full(len(yS), train_prior)

        V_pre = cell_corp(yW, p_pre, g, "V_pre")
        V_post = cell_corp(yW, p_post, g, "V_post")
        q_all = cell_corp(yS, q, g, "q")
        pt_all = cell_corp(yS, pt, g, "PT_flex")
        c_all = cell_corp(yS, const, g, "constant_TRAIN_prior")
        q_b40 = cell_corp(yS[b40], q[b40], g[b40], "q_B40") if b40.any() else dict(n=0)
        pt_b40 = cell_corp(yS[b40], pt[b40], g[b40], "PT_B40") if b40.any() else dict(n=0)

        d_brier = float(q_all["brier"] - pt_all["brier"])
        d_mcb = float(q_all["CORP"]["MCB"] - pt_all["CORP"]["MCB"])
        d_dsc = float(q_all["CORP"]["DSC"] - pt_all["CORP"]["DSC"])

        block = dict(
            label=label,
            pilot=pilot,
            ok=True,
            n=int(len(yS)),
            n_matches=int(len(np.unique(g))),
            n_sealed_W=align["n_sealed_W"],
            alignment=align,
            P_SVI=float(np.average(yS, weights=match_weights(g))),
            V_to_W=dict(
                pre=V_pre,
                post=V_post,
                pre_by_time=by_time(yW, p_pre, g, tmin),
                post_by_time=by_time(yW, p_post, g, tmin),
            ),
            q_to_SVI=dict(
                constant_TRAIN_prior=c_all,
                q=q_all,
                PT_flex=pt_all,
                q_B40=q_b40,
                PT_flex_B40=pt_b40,
                by_time_q=by_time(yS, q, g, tmin),
                by_time_PT_flex=by_time(yS, pt, g, tmin),
                delta_brier_q_minus_PT_flex=d_brier,
                delta_MCB_q_minus_PT=d_mcb,
                delta_DSC_q_minus_PT=d_dsc,
            ),
            reading=(
                "Same common-valid rows for V→W and q→SVI. "
                "On main EXT cohorts q DSC can still exceed PT while larger MCB dominates ΔBrier. "
                "EXT CORP is diagnostic only."
            ),
        )
        results[set_id] = block
        rows.append(
            dict(
                cohort=label,
                pilot=pilot,
                n=block["n"],
                n_common_eq_loaded=align["n_common_equals_n_loaded"],
                P_SVI=block["P_SVI"],
                V_pre_brier=V_pre["brier"],
                V_pre_auc=V_pre["auc"],
                V_pre_MCB=V_pre["CORP"]["MCB"],
                V_pre_DSC=V_pre["CORP"]["DSC"],
                V_post_brier=V_post["brier"],
                V_post_auc=V_post["auc"],
                V_post_MCB=V_post["CORP"]["MCB"],
                V_post_DSC=V_post["CORP"]["DSC"],
                q_brier=q_all["brier"],
                PT_flex_brier=pt_all["brier"],
                delta_brier_q_minus_PT=d_brier,
                delta_MCB=d_mcb,
                delta_DSC=d_dsc,
                q_MCB=q_all["CORP"]["MCB"],
                q_DSC=q_all["CORP"]["DSC"],
                PT_MCB=pt_all["CORP"]["MCB"],
                PT_DSC=pt_all["CORP"]["DSC"],
                q_auc=q_all["auc"],
            )
        )
        print(
            f"  n={block['n']} align_eq={align['n_common_equals_n_loaded']} "
            f"V_pre={fmt(V_pre['brier'])} V_post={fmt(V_post['brier'])} "
            f"dBrier={fmt(d_brier)} dMCB={fmt(d_mcb)} dDSC={fmt(d_dsc)}",
            flush=True,
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §11",
        train_constant_prior=train_prior,
        MAIN_feature_cols_sha16=hashlib.sha256("|".join(ref_cols).encode()).hexdigest()[:16],
        MAIN_15_16_q_vs_PT_flex_ref=main_ref,
        cohorts=results,
        reading=(
            "Preserve KR/NA1 16.13 loss of q lift. Pilots not pooled. "
            "15.16 CORP does not explain EXT; report per-cohort components on common-valid rows."
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
        f"**TRAIN constant prior:** {fmt(train_prior, 4)}",
        f"**MAIN feature-order sha16:** `{payload['MAIN_feature_cols_sha16']}`",
        "",
        "Score-only. Fail-closed without cohort keys / feature-order mismatch. "
        "V→W and q→SVI use **identical common-valid rows** (sealed W + finite V/q/PT).",
        "",
        "## Summary",
        "",
        "| Cohort | n | align | V_pre Brier | V_post Brier | ΔBrier(q−PT) | ΔMCB | ΔDSC | q MCB | q DSC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['cohort']} | {r['n']} | {r['n_common_eq_loaded']} | {fmt(r['V_pre_brier'])} | "
            f"{fmt(r['V_post_brier'])} | {fmt(r['delta_brier_q_minus_PT'])} | {fmt(r['delta_MCB'])} | "
            f"{fmt(r['delta_DSC'])} | {fmt(r['q_MCB'])} | {fmt(r['q_DSC'])} |"
        )
    md += ["", "### Alignment / fail-closed", ""]
    for set_id, block in results.items():
        if not block.get("ok"):
            md.append(f"- `{set_id}`: FAIL — {block.get('reason')}")
            continue
        a = block["alignment"]
        md.append(
            f"- `{block['label']}`: n_loaded={a['n_loaded']}, n_common={a['n_common']}, "
            f"n_common==n_loaded={a['n_common_equals_n_loaded']}, "
            f"feature_ok={a['feature_check']['ok']}"
        )
    md += [
        "",
        "## Reading",
        "",
        "- KR/NA1 16.13: q DSC still > PT_flex, but larger ΔMCB → net ΔBrier > 0 "
        "(signal not fully gone; calibration component dominates the loss gap).",
        "- V_pre Brier ~0.15 does not by itself validate EXT ΔV labels.",
        "- EXT CORP is diagnostic — not a fitted EXT recalibrator.",
        f"- Detail: `{OUT.relative_to(REPO).as_posix()}/rrx_external_results.json`",
        "",
    ]
    doc = REPO / "docs" / "REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md"
    doc.write_text("\n".join(md), encoding="utf-8")
    (OUT / doc.name).write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "rrx_external_results.json", doc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
