#!/usr/bin/env python3
"""V-4 continuity checks for redesign candidates (same 210k cohort).

Compares shared_lgbm vs per_band_lgbm vs legacy logistic on:
  1) Consecutive bucket-query Δp within matches (interior vs band-boundary minutes)
  2) Engagement pre→post h90 sign(ΔV) agreement across specs (15.16 T)

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


def band_name(tmin: float) -> str:
    if tmin < 10:
        return "t_0_10"
    if tmin < 20:
        return "t_10_20"
    if tmin < 30:
        return "t_20_30"
    return "t_30_inf"


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
    """Return |Δp| arrays for interior steps and boundary-crossing steps.

    Sort by (match, t) so consecutive rows stay within the same match.
    Global time-only argsort interleaves matches and yields empty groups.
    """
    m = np.asarray(match)
    t = np.asarray(tmin, dtype=float)
    pr = np.asarray(p, dtype=float)
    # Stable within-match time order: primary key match, secondary t
    # Encode match as factor for numeric lexsort
    _, match_codes = np.unique(m.astype(str), return_inverse=True)
    order = np.lexsort((t, match_codes))
    t = t[order]
    pr = pr[order]
    codes = match_codes[order]
    interior, boundary = [], []
    i = 0
    n = len(t)
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


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


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

    print("score shared…", flush=True)
    p_shared = predict_lgbm(shared_pack["model"], X, cols, champ_ix)
    print("score per-band…", flush=True)
    p_band = np.full(len(tmin), np.nan)
    for lo_hi, clf in band_pack["models"].items():
        # keys are t_0_10 etc.
        if lo_hi == "t_0_10":
            m = tmin < 10
        elif lo_hi == "t_10_20":
            m = (tmin >= 10) & (tmin < 20)
        elif lo_hi == "t_20_30":
            m = (tmin >= 20) & (tmin < 30)
        else:
            m = tmin >= 30
        if m.any():
            p_band[m] = predict_lgbm(clf, X[m], cols, champ_ix)
    print("score legacy…", flush=True)
    p_leg = legacy.predict_matrix(TE["X"], names, C.STATE_VERSION)

    jump = {}
    for name, p in (("shared_lgbm", p_shared), ("per_band_lgbm", p_band), ("legacy_logistic_raw", p_leg)):
        fin = np.isfinite(p)
        interior, boundary = consecutive_deltas(g[fin], tmin[fin], p[fin])
        jump[name] = dict(
            interior=summarize_abs(interior),
            boundary=summarize_abs(boundary),
            boundary_minus_interior_mean=(
                float(np.mean(boundary) - np.mean(interior))
                if len(boundary) and len(interior) else None
            ),
        )
        print(name, "interior mean", jump[name]["interior"].get("mean"),
              "boundary mean", jump[name]["boundary"].get("mean"), flush=True)

    # Engagement sign agreement on MAIN_TEST T h90
    print("engagement sign agree…", flush=True)
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False,
    )
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    # Need pre/post state matrices from extract — use sealed p_pre / p_post for legacy;
    # for new models score e_X_pre / e_X_post from chunks is heavy. Use label p_* for legacy
    # and re-score engagement X from a sample of state chunks via D.load_engagements.
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    # filter T + valid h90
    # E has arrays aligned; cohort codes from match+s join to labels
    # Simpler: use MAIN_TEST labels p_pre/p_post for legacy Δ; for new models score e_X
    m_lab = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    # Build key map from engagement extract
    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    # only rows with pre ok and valid h90 flags if present
    pre_ok = E["pre_ok"] == 1 if "pre_ok" in E else np.ones(len(em), dtype=bool)
    # valid_h90 in engagement extract
    valid = E["valid_h90"] == 1 if "valid_h90" in E else np.ones(len(em), dtype=bool)
    # Restrict to teamfight: join label keys
    lab_keys = set(zip(lab["match"][m_lab].astype(str).tolist(), lab["s"][m_lab].astype(np.int64).tolist()))
    keep = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= pre_ok & valid
    em, es = em[keep], es[keep]
    Xpre = E["X_pre"][keep]
    Xpost = E["X_post_h90"][keep]
    Xp, cols2, cix2 = expanded_X(Xpre, names)
    Xq, _, _ = expanded_X(Xpost, names)
    assert cols2 == cols

    p_pre_s = predict_lgbm(shared_pack["model"], Xp, cols, champ_ix)
    p_post_s = predict_lgbm(shared_pack["model"], Xq, cols, champ_ix)
    # per-band uses onset band for both ends? Contract: same frozen map — for Choice B use time at each query
    t_pre = es.astype(float) / 60000.0
    t_post = E["endpoint_h90"][keep].astype(float) / 60000.0

    def score_band(Xmat, t):
        out = np.full(len(t), np.nan)
        for key, clf in band_pack["models"].items():
            if key == "t_0_10":
                m = t < 10
            elif key == "t_10_20":
                m = (t >= 10) & (t < 20)
            elif key == "t_20_30":
                m = (t >= 20) & (t < 30)
            else:
                m = t >= 30
            if m.any():
                out[m] = predict_lgbm(clf, Xmat[m], cols, champ_ix)
        return out

    p_pre_b = score_band(Xp, t_pre)
    p_post_b = score_band(Xq, t_post)
    p_pre_l = legacy.predict_matrix(Xpre, names, C.STATE_VERSION)
    p_post_l = legacy.predict_matrix(Xpost, names, C.STATE_VERSION)

    def signs(pre, post):
        return np.sign(post - pre)

    s_s, s_b, s_l = signs(p_pre_s, p_post_s), signs(p_pre_b, p_post_b), signs(p_pre_l, p_post_l)
    # drop zeros for agree rate optional — count nonzero
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
        # also include zeros as disagree with nonzero
        shared_vs_per_band_incl0=float(np.mean(s_s == s_b)),
        shared_vs_legacy_incl0=float(np.mean(s_s == s_l)),
    )

    # Recommendation
    b_excess = jump["per_band_lgbm"]["boundary_minus_interior_mean"]
    s_excess = jump["shared_lgbm"]["boundary_minus_interior_mean"]
    rec = (
        "Keep Choice A shared_lgbm as provisional freeze: per-band L_time gain is tiny and "
        "boundary jump excess / sign flips must not dominate."
    )
    if b_excess is not None and s_excess is not None and b_excess > s_excess + 0.005:
        rec = (
            "Reject per_band for ΔV until boundary jumps shrink: per-band boundary |Δp| excess "
            f"({b_excess:.4f}) exceeds shared ({s_excess:.4f}). Prefer shared_lgbm."
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_REDESIGN_CONTRACT_20260919.md §7",
        sample="MAIN TEST bucket queries + MAIN_TEST T∩valid_h90 engagements",
        consecutive_jumps=jump,
        engagement_sign_agree=sign_agree,
        recommendation=rec,
        provisional_freeze="shared_lgbm",
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
        "# V-4 continuity — redesign candidates",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        f"**Provisional freeze:** `{payload['provisional_freeze']}`",
        "",
        rec,
        "",
        "## 1. Consecutive bucket-query |Δp| (TEST)",
        "",
        "| Model | interior mean | boundary mean | excess (b−i) | boundary P(>|0.05|) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, blk in jump.items():
        i, b = blk["interior"], blk["boundary"]
        lines.append(
            f"| {name} | {fmt(i.get('mean'), 4)} | {fmt(b.get('mean'), 4)} | "
            f"{fmt(blk.get('boundary_minus_interior_mean'), 4)} | {fmt(b.get('frac_gt_0_05'), 4)} |"
        )
    sa = sign_agree
    lines += [
        "",
        "## 2. Engagement sign(ΔV) agree (15.16 T, h90)",
        "",
        f"n={sa['n']}",
        f"- shared vs per_band (nonzero): {fmt(sa['shared_vs_per_band'], 4)}",
        f"- shared vs legacy (nonzero): {fmt(sa['shared_vs_legacy'], 4)}",
        f"- per_band vs legacy (nonzero): {fmt(sa['per_band_vs_legacy'], 4)}",
        f"- shared vs per_band (incl 0): {fmt(sa['shared_vs_per_band_incl0'], 4)}",
        "",
        "## Reading",
        "",
        "- Boundary steps should not show a large excess |Δp| for per-band vs shared.",
        "- High sign agree with legacy supports using shared_lgbm for ΔV without rewriting history.",
        "- Next: optional history stack (K=3/5); then freeze + rebuild SVI only after V-3 manifest.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # Update winner manifest to provisional shared
    man = dict(
        provisional_freeze="shared_lgbm",
        mechanical_select_winner="per_band_lgbm",
        reason="Choice A default; tiny L_time gap; see CONTINUITY_REPORT.md",
        generated=payload["generated"],
        epistemic=ROLE,
    )
    (v_dir / "winner_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_md)
    print("recommendation:", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
