#!/usr/bin/env python3
"""C04: horizon sensitivity for ΔV̂ / SVI (h60, h90, h120) on 15.16 T.

Does not retrain q. Reports label flips, ΔV agreement, and V̂(S_end)→W by horizon.
Contract: docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md §4.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"
HS = (60, 90, 120)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics_g(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    w = match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(n=int(len(y)), n_matches=int(len(np.unique(g))), brier=br, auc=auc)


def load_match_w(data_root: Path) -> Dict[str, int]:
    d = data_root / "outputs" / "full_corpus_training_20260915" / "extract" / "MAIN" / "outcomes_SEALED"
    out: Dict[str, int] = {}
    for p in sorted(d.glob("chunk_*.npz")):
        z = np.load(p, allow_pickle=False)
        for m, w in zip(z["o_match"].astype(str), z["o_winner_blue"].astype(int)):
            out[m] = int(w)
    return out


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_horizon_sensitivity_20260919")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False)
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False)

    t_mask = coh["cohort"] == 1
    g_all = lab["match"][t_mask].astype(str)
    p_pre = lab["p_pre"][t_mask].astype(float)
    match_w = load_match_w(data_root)
    yw = np.array([match_w.get(x, -1) for x in g_all], dtype=int)
    sealed = yw >= 0

    # Intersection: valid at all three horizons + sealed W
    valid = sealed.copy()
    for h in HS:
        valid &= lab[f"valid_h{h}"][t_mask] == 1
    idx = np.where(valid)[0]
    g = g_all[idx]
    y_w = yw[idx]
    p0 = p_pre[idx]

    Y = {h: lab[f"Y_h{h}"][t_mask][idx].astype(int) for h in HS}
    delta = {h: lab[f"delta_h{h}"][t_mask][idx].astype(float) for h in HS}
    p_post = {h: lab[f"p_post_h{h}"][t_mask][idx].astype(float) for h in HS}

    # Pairwise SVI flip rates (among intersection)
    flips = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        disagree = int((Y[a] != Y[b]).sum())
        flips[f"Y_h{a}_vs_h{b}"] = dict(
            n=int(len(idx)),
            n_disagree=disagree,
            rate=float(disagree / max(len(idx), 1)),
            agree=float(1.0 - disagree / max(len(idx), 1)),
        )

    # Continuous ΔV agreement
    corr = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        corr[f"delta_h{a}_vs_h{b}"] = float(np.corrcoef(delta[a], delta[b])[0, 1])

    # Sign agreement including zeros? Y already is 1[delta>0] typically
    sign_agree = {k: flips[k]["agree"] for k in flips}

    # V̂(S_end) → W by horizon
    v_end = {f"h{h}": metrics_g(y_w, p_post[h], g) for h in HS}
    v_pre = metrics_g(y_w, p0, g)

    # Also report valid counts per horizon (not only intersection)
    per_h = {}
    for h in HS:
        m = t_mask & (lab[f"valid_h{h}"] == 1)
        gg = lab["match"][m].astype(str)
        yy = np.array([match_w.get(x, -1) for x in gg], dtype=int)
        ok = yy >= 0
        per_h[f"h{h}"] = dict(
            n_valid_T=int(m.sum()),
            n_sealed=int(ok.sum()),
            Y_rate=float(lab[f"Y_h{h}"][m][ok].mean()) if ok.any() else None,
            v_end=metrics_g(yy[ok], lab[f"p_post_h{h}"][m][ok].astype(float), gg[ok]),
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md §4 (C04)",
        sample="MAIN_TEST T ∩ valid_h60∩h90∩h120 ∩ sealed W",
        n_intersection=int(len(idx)),
        n_matches=int(len(np.unique(g))),
        Y_positive_rate={f"h{h}": float(Y[h].mean()) for h in HS},
        flip_rates=flips,
        delta_pearson=corr,
        sign_agree=sign_agree,
        v_pre_intersection=v_pre,
        v_end_intersection=v_end,
        per_horizon_valid=per_h,
        note="Primary paper label remains h90; this is sensitivity, not reselection.",
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

    (out_dir / "results.json").write_text(
        json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Horizon sensitivity (C04) — ΔV̂ / SVI",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        "Primary label stays **h90**. This table is sensitivity only.",
        f"Intersection sample: n={payload['n_intersection']} / matches={payload['n_matches']}.",
        "",
        "## 1. SVI flip rates (intersection)",
        "",
        "| Pair | agree | disagree rate | n_disagree |",
        "|---|---:|---:|---:|",
    ]
    for k, v in flips.items():
        lines.append(f"| {k} | {fmt(v['agree'], 4)} | {fmt(v['rate'], 4)} | {v['n_disagree']} |")
    lines += [
        "",
        "## 2. Continuous ΔV Pearson (intersection)",
        "",
    ]
    for k, v in corr.items():
        lines.append(f"- {k}: {fmt(v, 4)}")
    lines += [
        "",
        "## 3. V(S_end) -> W on intersection",
        "",
        f"Pre: Brier {fmt(v_pre['brier'], 6)} AUC {fmt(v_pre['auc'], 4)}",
        "",
        "| Horizon | Brier | AUC |",
        "|---|---:|---:|",
    ]
    for h in HS:
        r = v_end[f"h{h}"]
        lines.append(f"| h{h} | {fmt(r['brier'], 6)} | {fmt(r['auc'], 4)} |")
    lines += [
        "",
        "## 4. Per-horizon valid T (not intersection)",
        "",
        "| Horizon | n_valid_T | Y rate | post Brier | post AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in HS:
        r = per_h[f"h{h}"]
        ve = r["v_end"]
        lines.append(
            f"| h{h} | {r['n_valid_T']} | {fmt(r['Y_rate'], 4)} | "
            f"{fmt(ve['brier'], 6)} | {fmt(ve['auc'], 4)} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- High agree between h60/h90/h120 => primary h90 choice is not fragile to +/-30s endpoint.",
        "- Large flips => Methods must treat horizon as part of the estimand definition.",
        "- Does **not** recompute sealed q-PT (q trained for h90 only).",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_dir)
    print("flips", flips)
    print("corr", corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
