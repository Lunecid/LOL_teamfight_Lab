#!/usr/bin/env python3
"""V-4 partial continuity checks for redesign candidates (same 210k cohort).

PRIMARY outputs use the frozen evaluation map
  V_final(x) = g_frozen( f_frozen( T(x) ) )
matching relabel / SVI (PosSlopeSigmoid from freeze_manifest).

Raw f(x) jumps are reported as a separate DIAGNOSTIC (magnitude can differ under g).

Checks (partial V-4 only):
  1) Consecutive bucket-query |Δp| within matches (interior vs band-boundary minutes)
  2) Engagement pre→post h90 sign(ΔV) agreement across specs (15.16 T)

NOT implemented here: event-matched quiet windows, frame-refresh-only, D_switch(x)
same-state band-model swap. Boundary excess is a WARNING SIGNAL, not proof of
model-switch artifact. Choice A remains the a priori default — TEST diagnostics
do not select the freeze winner.

Contract: docs/V_REDESIGN_CONTRACT_20260919.md §7
Writes: outputs/v_redesign_20260919/continuity_{results.json,REPORT.md}
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
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
BAND_EDGES = (10.0, 20.0, 30.0)  # minute boundaries


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


class PosSlopeSigmoid:
    def __init__(self, coef: float, intercept: float, ok: bool = True):
        self.coef_, self.intercept_, self.ok = coef, intercept, ok

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        z = self.coef_ * logit + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))


def expanded_X(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float64, copy=False), cols, champ_ix


def predict_lgbm(clf, X, cols, champ_ix):
    import pandas as pd
    df = pd.DataFrame(X, columns=list(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in champ_ix:
            df[cols[j]] = df[cols[j]].astype("category")
        return clf.predict_proba(df)[:, 1]


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
    """|Δp| for interior vs boundary-crossing consecutive steps (sort by match, t)."""
    m = np.asarray(match)
    t = np.asarray(tmin, dtype=float)
    pr = np.asarray(p, dtype=float)
    _, match_codes = np.unique(m.astype(str), return_inverse=True)
    order = np.lexsort((t, match_codes))
    t, pr, codes = t[order], pr[order], match_codes[order]
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
            float(np.mean(boundary) - np.mean(interior))
            if len(boundary) and len(interior) else None
        ),
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def load_calib(v_dir: Path, results: dict, name: str) -> PosSlopeSigmoid:
    freeze = json.loads((v_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    if name == "shared_lgbm":
        c = freeze["calibration"]
        return PosSlopeSigmoid(float(c["coef"]), float(c["intercept"]), bool(c["ok"]))
    # per-band: use its own V_CAL map from results.json (not the shared freeze g)
    blk = results.get("selection", {}).get(name, {})
    cal = blk.get("calib") or {}
    if cal and blk.get("calib_ok", True):
        return PosSlopeSigmoid(float(cal["coef"]), float(cal["intercept"]), True)
    return PosSlopeSigmoid(1.0, 0.0, False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--v-dir", type=Path, default=REPO / "outputs" / "v_redesign_20260919")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    v_dir = args.v_dir
    out_json = v_dir / "continuity_results.json"
    out_md = v_dir / "CONTINUITY_REPORT.md"
    results = json.loads((v_dir / "results.json").read_text(encoding="utf-8"))

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    print("load TEST bucket rows…", flush=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TE["names"])
    X, cols, champ_ix = expanded_X(TE["X"], names)
    g = TE["match"].astype(str)
    tmin = TE["query_ms"].astype(float) / 60000.0

    shared_pack = joblib.load(v_dir / "models" / "shared_lgbm.joblib")
    band_pack = joblib.load(v_dir / "models" / "per_band_lgbm.joblib")
    legacy = joblib.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "models" / "v" / "v_final_raw.joblib"
    )
    g_shared = load_calib(v_dir, results, "shared_lgbm")
    g_band = load_calib(v_dir, results, "per_band_lgbm")

    print("score shared (raw + calibrated)…", flush=True)
    raw_shared = predict_lgbm(shared_pack["model"], X, cols, champ_ix)
    p_shared = g_shared.transform(raw_shared)

    print("score per-band (raw + calibrated)…", flush=True)
    raw_band = np.full(len(tmin), np.nan)

    def fill_band(dest, Xmat, t):
        for lo_hi, clf in band_pack["models"].items():
            if lo_hi == "t_0_10":
                m = t < 10
            elif lo_hi == "t_10_20":
                m = (t >= 10) & (t < 20)
            elif lo_hi == "t_20_30":
                m = (t >= 20) & (t < 30)
            else:
                m = t >= 30
            if m.any():
                dest[m] = predict_lgbm(clf, Xmat[m], cols, champ_ix)

    fill_band(raw_band, X, tmin)
    p_band = np.where(np.isfinite(raw_band), g_band.transform(raw_band), np.nan)

    print("score legacy…", flush=True)
    p_leg = legacy.predict_matrix(TE["X"], names, C.STATE_VERSION)

    jump_cal = {}
    jump_raw = {}
    for name, p_cal, p_raw in (
        ("shared_lgbm", p_shared, raw_shared),
        ("per_band_lgbm", p_band, raw_band),
        ("legacy_logistic_raw", p_leg, p_leg),
    ):
        jump_cal[name] = jump_block(g, tmin, p_cal)
        jump_raw[name] = jump_block(g, tmin, p_raw)
        print(
            name, "CAL excess", jump_cal[name]["boundary_minus_interior_mean"],
            "RAW excess", jump_raw[name]["boundary_minus_interior_mean"], flush=True,
        )

    # Engagement sign agreement — calibrated for redesign models
    print("engagement sign agree (calibrated redesign)…", flush=True)
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
    lab_keys = set(zip(lab["match"][m_lab].astype(str).tolist(), lab["s"][m_lab].astype(np.int64).tolist()))
    keep = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= pre_ok & valid
    Xpre = E["X_pre"][keep]
    Xpost = E["X_post_h90"][keep]
    Xp, cols2, _ = expanded_X(Xpre, names)
    Xq, _, _ = expanded_X(Xpost, names)
    assert cols2 == cols
    t_pre = E["s"][keep].astype(float) / 60000.0
    t_post = E["endpoint_h90"][keep].astype(float) / 60000.0

    def score_band_raw(Xmat, t):
        out = np.full(len(t), np.nan)
        fill_band(out, Xmat, t)
        return out

    p_pre_s = g_shared.transform(predict_lgbm(shared_pack["model"], Xp, cols, champ_ix))
    p_post_s = g_shared.transform(predict_lgbm(shared_pack["model"], Xq, cols, champ_ix))
    p_pre_b = g_band.transform(score_band_raw(Xp, t_pre))
    p_post_b = g_band.transform(score_band_raw(Xq, t_post))
    p_pre_l = legacy.predict_matrix(Xpre, names, C.STATE_VERSION)
    p_post_l = legacy.predict_matrix(Xpost, names, C.STATE_VERSION)

    def signs(pre, post):
        return np.sign(post - pre)

    s_s, s_b, s_l = signs(p_pre_s, p_post_s), signs(p_pre_b, p_post_b), signs(p_pre_l, p_post_l)

    def agree(a, b):
        m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        if not m.any():
            return None
        return float(np.mean(a[m] == b[m]))

    sign_agree = dict(
        n=int(len(s_s)),
        shared_vs_per_band=agree(s_s, s_b),
        shared_vs_legacy=agree(s_s, s_l),
        per_band_vs_legacy=agree(s_b, s_l),
        shared_vs_per_band_incl0=float(np.mean(s_s == s_b)),
        shared_vs_legacy_incl0=float(np.mean(s_s == s_l)),
        note="redesign models use calibrated probs; legacy uses sealed logistic map",
    )

    b_excess = jump_cal["per_band_lgbm"]["boundary_minus_interior_mean"]
    s_excess = jump_cal["shared_lgbm"]["boundary_minus_interior_mean"]
    # Soft diagnostic — Choice A is a priori; do not treat TEST as selection
    rec = (
        "Choice A shared_lgbm remains the a priori provisional freeze. "
        "TEST consecutive-query boundary excess is a WARNING SIGNAL only "
        "(not event-matched; not D_switch same-state). "
        f"Calibrated excess: per_band={fmt(b_excess)} vs shared={fmt(s_excess)}. "
        "Further model-switch isolation on DEV is still required before claiming "
        "per-band is unsuitable for ΔV."
    )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_REDESIGN_CONTRACT_20260919.md §7",
        evaluation_map="V_final = g(f(T(X))); primary tables use calibrated probs",
        sample="MAIN TEST bucket queries + MAIN_TEST T∩valid_h90 (exploratory; prior TEST exposure)",
        role_of_test="DIAGNOSTIC_LEDGER_NOT_SELECTION — Choice A locked a priori",
        consecutive_jumps_calibrated=jump_cal,
        consecutive_jumps_raw_diagnostic=jump_raw,
        engagement_sign_agree=sign_agree,
        recommendation=rec,
        provisional_freeze="shared_lgbm",
        v4_coverage="PARTIAL: consecutive bucket jumps + multi-spec sign agree; "
                    "missing quiet / frame-refresh / D_switch(x)",
        threshold_note="No formal Δ-excess tolerance; 0.005 was an ad-hoc script flag (removed as decision rule)",
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

    out_json.write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# V-4 continuity — PARTIAL (calibrated evaluation map)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        f"**Provisional freeze (a priori Choice A):** `{payload['provisional_freeze']}`",
        "",
        rec,
        "",
        "## Evaluation map",
        "",
        "Primary numbers use **calibrated** $g\\circ f$ (same as relabel / SVI).",
        "Raw $f$ tables are diagnostic only — magnitudes need not match under monotone $g$.",
        "TEST here is **ledger / warning**, not the freeze selection rule.",
        "",
        "## 1. Consecutive bucket-query |Δp| — CALIBRATED (TEST)",
        "",
        "| Model | interior mean | boundary mean | excess (b−i) | boundary P(>|0.05|) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, blk in jump_cal.items():
        i, b = blk["interior"], blk["boundary"]
        lines.append(
            f"| {name} | {fmt(i.get('mean'), 4)} | {fmt(b.get('mean'), 4)} | "
            f"{fmt(blk.get('boundary_minus_interior_mean'), 4)} | {fmt(b.get('frac_gt_0_05'), 4)} |"
        )
    lines += [
        "",
        "### Diagnostic — RAW (uncalibrated redesign)",
        "",
        "| Model | excess (b−i) |",
        "|---|---:|",
    ]
    for name, blk in jump_raw.items():
        lines.append(f"| {name} | {fmt(blk.get('boundary_minus_interior_mean'), 4)} |")
    sa = sign_agree
    lines += [
        "",
        "## 2. Engagement sign(ΔV) agree (15.16 T, h90; calibrated redesign)",
        "",
        f"n={sa['n']}",
        f"- shared vs per_band (nonzero): {fmt(sa['shared_vs_per_band'], 4)}",
        f"- shared vs legacy (nonzero): {fmt(sa['shared_vs_legacy'], 4)}",
        f"- per_band vs legacy (nonzero): {fmt(sa['per_band_vs_legacy'], 4)}",
        "",
        "## Reading (narrow)",
        "",
        "- Excess on boundary-crossing **stored queries** ≠ proof of model-switch artifact",
        "  (events / spacing / refresh not matched).",
        "- Needed next: D_switch(x)=|V_{b+1}(x)-V_b(x)| on DEV + quiet/event strata.",
        "- Coverage: **partial V-4** only.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    man = dict(
        provisional_freeze="shared_lgbm",
        status="FROZEN_PROVISIONAL_A_PRIORI_CHOICE_A",
        mechanical_select_winner_on_V_SELECT="per_band_lgbm",
        reason=(
            "Choice A a priori; tiny V_SELECT L_time gap; TEST boundary excess is "
            "diagnostic warning only — see CONTINUITY_REPORT.md"
        ),
        freeze_manifest="outputs/v_redesign_20260919/freeze_manifest.json",
        generated=payload["generated"],
        epistemic=ROLE,
    )
    (v_dir / "winner_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_md)
    print("recommendation:", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
