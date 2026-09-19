#!/usr/bin/env python3
"""V-4 partial continuity on wave-2 freeze contenders (A0 / A3 / A2b).

PRIMARY: calibrated map V = g∘f∘T (PosSlopeSigmoid from each model's V_CAL fit
stored in wave2 results.json). Raw f jumps are DIAGNOSTIC only.

Checks (partial V-4, same as wave-1 continuity):
  1) Consecutive TEST bucket-query |Δp| (interior vs band-boundary minutes)
  2) Engagement pre→post h90 sign(ΔV) agreement across specs (15.16 T)

Optional --with-h5: also score H5_lgbm jumps + sign vs A0 (history stacks from
TEST bucket queries at/before engagement times).

Choice A / A0 remains a priori default — TEST diagnostics do NOT select freeze.
Writes: outputs/v_redesign_wave2_20260919/continuity_{results.json,REPORT.md}
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
BAND_EDGES = (10.0, 20.0, 30.0)
WAVE2 = REPO / "outputs" / "v_redesign_wave2_20260919"
WAVE3 = REPO / "outputs" / "v_redesign_wave3_tier23_20260919"


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


def predict_logistic(model, X_raw, keep):
    return model.predict_proba(X_raw[:, keep])[:, 1]


def predict_mlp(pack, X):
    import torch

    Xs = pack["scaler"].transform(X).astype(np.float32)
    net = pack["net"]
    net.eval()
    outs = []
    with torch.no_grad():
        t = torch.from_numpy(Xs)
        for i in range(0, len(Xs), 65536):
            outs.append(torch.sigmoid(net(t[i : i + 65536]).squeeze(-1)).numpy())
    return np.concatenate(outs)


def predict_lgbm_flat(clf, Xflat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clf.predict_proba(Xflat)[:, 1]


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
            if len(boundary) and len(interior)
            else None
        ),
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def load_calib(results: dict, name: str) -> PosSlopeSigmoid:
    blk = results.get("selection", {}).get(name, {})
    cal = blk.get("calib") or {}
    if cal and blk.get("calib_ok", True):
        return PosSlopeSigmoid(float(cal["coef"]), float(cal["intercept"]), True)
    return PosSlopeSigmoid(1.0, 0.0, False)


def build_history_stacks(match, tmin, X, K):
    n, d = X.shape
    _, codes = np.unique(match.astype(str), return_inverse=True)
    order = np.lexsort((tmin, codes))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    codes_o = codes[order]
    X_o = X[order]
    seq = np.zeros((n, K, d), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and codes_o[j] == codes_o[i]:
            j += 1
        for t in range(i, j):
            start = max(i, t - K + 1)
            length = t - start + 1
            seq[t, K - length : K] = X_o[start : t + 1]
            mask[t, K - length : K] = 1.0
        i = j
    seq, mask = seq[inv], mask[inv]
    flat = np.concatenate([seq.reshape(n, K * d), mask], axis=1)
    return flat


def history_at_times(
    match_q: np.ndarray,
    tmin_q: np.ndarray,
    X_q: np.ndarray,
    match_t: np.ndarray,
    tmin_t: np.ndarray,
    K: int,
) -> np.ndarray:
    """For each target (match,t), flatten last-K bucket frames with t_q <= t in that match."""
    d = X_q.shape[1]
    # index queries by match
    from collections import defaultdict

    by = defaultdict(list)
    for i, m in enumerate(match_q.astype(str)):
        by[m].append(i)
    for m in by:
        ix = np.asarray(by[m], dtype=np.int64)
        order = np.argsort(tmin_q[ix])
        by[m] = ix[order]

    n = len(match_t)
    flat = np.zeros((n, K * d + K), dtype=np.float32)
    for i, (m, t) in enumerate(zip(match_t.astype(str), tmin_t)):
        ix = by.get(m)
        if ix is None or len(ix) == 0:
            continue
        ts = tmin_q[ix]
        # last index with ts <= t
        k = int(np.searchsorted(ts, t, side="right") - 1)
        if k < 0:
            continue
        start = max(0, k - K + 1)
        length = k - start + 1
        frames = X_q[ix[start : k + 1]]
        seq = np.zeros((K, d), dtype=np.float32)
        mask = np.zeros(K, dtype=np.float32)
        seq[K - length : K] = frames
        mask[K - length : K] = 1.0
        flat[i] = np.concatenate([seq.ravel(), mask])
    return flat


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--wave2-dir", type=Path, default=WAVE2)
    ap.add_argument("--wave3-dir", type=Path, default=WAVE3)
    ap.add_argument("--with-h5", action="store_true")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    v_dir = args.wave2_dir
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

    # Load models
    a0 = joblib.load(v_dir / "models" / "A0_shared_logistic.joblib")
    a2b = joblib.load(v_dir / "models" / "A2b_shared_lgbm_cog.joblib")
    a3 = joblib.load(v_dir / "models" / "A3_shared_mlp.joblib")
    legacy = joblib.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "models" / "v" / "v_final_raw.joblib"
    )
    g_a0 = load_calib(results, "A0_shared_logistic")
    g_a2b = load_calib(results, "A2b_shared_lgbm_cog")
    g_a3 = load_calib(results, "A3_shared_mlp")

    scorers_raw: Dict[str, np.ndarray] = {}
    scorers_cal: Dict[str, np.ndarray] = {}

    print("score A0 / A2b / A3 / legacy…", flush=True)
    raw_a0 = predict_logistic(a0["model"], TE["X"], a0["keep"])
    scorers_raw["A0_shared_logistic"] = raw_a0
    scorers_cal["A0_shared_logistic"] = g_a0.transform(raw_a0)

    raw_a2b = predict_lgbm(a2b["model"], X, cols, champ_ix)
    scorers_raw["A2b_shared_lgbm_cog"] = raw_a2b
    scorers_cal["A2b_shared_lgbm_cog"] = g_a2b.transform(raw_a2b)

    raw_a3 = predict_mlp(a3, X.astype(np.float32))
    scorers_raw["A3_shared_mlp"] = raw_a3
    scorers_cal["A3_shared_mlp"] = g_a3.transform(raw_a3)

    p_leg = legacy.predict_matrix(TE["X"], names, C.STATE_VERSION)
    scorers_raw["legacy_logistic_raw"] = p_leg
    scorers_cal["legacy_logistic_raw"] = p_leg  # sealed map already evaluation-ready

    h5_clf = None
    g_h5 = None
    if args.with_h5:
        w3res = json.loads((args.wave3_dir / "results.json").read_text(encoding="utf-8"))
        h5_clf = joblib.load(args.wave3_dir / "models" / "H5_lgbm.joblib")["model"]
        g_h5 = load_calib(w3res, "H5_lgbm")
        print("score H5 on TEST history stacks…", flush=True)
        Xf = X.astype(np.float32)
        flat = build_history_stacks(g, tmin, Xf, 5)
        raw_h5 = predict_lgbm_flat(h5_clf, flat)
        scorers_raw["H5_lgbm"] = raw_h5
        scorers_cal["H5_lgbm"] = g_h5.transform(raw_h5)

    jump_cal, jump_raw = {}, {}
    for name in scorers_cal:
        jump_cal[name] = jump_block(g, tmin, scorers_cal[name])
        jump_raw[name] = jump_block(g, tmin, scorers_raw[name])
        print(
            name,
            "CAL excess",
            jump_cal[name]["boundary_minus_interior_mean"],
            "RAW excess",
            jump_raw[name]["boundary_minus_interior_mean"],
            flush=True,
        )

    # Engagement sign agreement
    print("engagement sign agree…", flush=True)
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
    Xpre = E["X_pre"][keep]
    Xpost = E["X_post_h90"][keep]
    Xp, cols2, _ = expanded_X(Xpre, names)
    Xq, _, _ = expanded_X(Xpost, names)
    assert cols2 == cols
    t_pre = E["s"][keep].astype(float) / 60000.0
    t_post = E["endpoint_h90"][keep].astype(float) / 60000.0
    m_eng = em[keep]

    def signs(pre, post):
        return np.sign(post - pre)

    eng_raw_pre: Dict[str, np.ndarray] = {}
    eng_raw_post: Dict[str, np.ndarray] = {}
    eng_cal_pre: Dict[str, np.ndarray] = {}
    eng_cal_post: Dict[str, np.ndarray] = {}

    eng_raw_pre["A0_shared_logistic"] = predict_logistic(a0["model"], Xpre, a0["keep"])
    eng_raw_post["A0_shared_logistic"] = predict_logistic(a0["model"], Xpost, a0["keep"])
    eng_cal_pre["A0_shared_logistic"] = g_a0.transform(eng_raw_pre["A0_shared_logistic"])
    eng_cal_post["A0_shared_logistic"] = g_a0.transform(eng_raw_post["A0_shared_logistic"])

    eng_raw_pre["A2b_shared_lgbm_cog"] = predict_lgbm(a2b["model"], Xp, cols, champ_ix)
    eng_raw_post["A2b_shared_lgbm_cog"] = predict_lgbm(a2b["model"], Xq, cols, champ_ix)
    eng_cal_pre["A2b_shared_lgbm_cog"] = g_a2b.transform(eng_raw_pre["A2b_shared_lgbm_cog"])
    eng_cal_post["A2b_shared_lgbm_cog"] = g_a2b.transform(eng_raw_post["A2b_shared_lgbm_cog"])

    eng_raw_pre["A3_shared_mlp"] = predict_mlp(a3, Xp.astype(np.float32))
    eng_raw_post["A3_shared_mlp"] = predict_mlp(a3, Xq.astype(np.float32))
    eng_cal_pre["A3_shared_mlp"] = g_a3.transform(eng_raw_pre["A3_shared_mlp"])
    eng_cal_post["A3_shared_mlp"] = g_a3.transform(eng_raw_post["A3_shared_mlp"])

    eng_raw_pre["legacy_logistic_raw"] = legacy.predict_matrix(Xpre, names, C.STATE_VERSION)
    eng_raw_post["legacy_logistic_raw"] = legacy.predict_matrix(Xpost, names, C.STATE_VERSION)
    eng_cal_pre["legacy_logistic_raw"] = eng_raw_pre["legacy_logistic_raw"]
    eng_cal_post["legacy_logistic_raw"] = eng_raw_post["legacy_logistic_raw"]

    if args.with_h5 and h5_clf is not None and g_h5 is not None:
        print("score H5 on engagement via bucket history…", flush=True)
        Xf = X.astype(np.float32)
        flat_pre = history_at_times(g, tmin, Xf, m_eng, t_pre, 5)
        flat_post = history_at_times(g, tmin, Xf, m_eng, t_post, 5)
        # rows with no history mask (all-zero mask bits) → nan
        mask_ok_pre = flat_pre[:, -5:].sum(axis=1) > 0
        mask_ok_post = flat_post[:, -5:].sum(axis=1) > 0
        rp = np.full(len(m_eng), np.nan)
        rq = np.full(len(m_eng), np.nan)
        rp[mask_ok_pre] = predict_lgbm_flat(h5_clf, flat_pre[mask_ok_pre])
        rq[mask_ok_post] = predict_lgbm_flat(h5_clf, flat_post[mask_ok_post])
        eng_raw_pre["H5_lgbm"] = rp
        eng_raw_post["H5_lgbm"] = rq
        eng_cal_pre["H5_lgbm"] = np.where(np.isfinite(rp), g_h5.transform(rp), np.nan)
        eng_cal_post["H5_lgbm"] = np.where(np.isfinite(rq), g_h5.transform(rq), np.nan)

    names_eng = list(eng_cal_pre.keys())
    sign_map = {n: signs(eng_cal_pre[n], eng_cal_post[n]) for n in names_eng}

    def agree(a, b):
        m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        if not m.any():
            return None
        return float(np.mean(a[m] == b[m]))

    pairs = []
    for i, a in enumerate(names_eng):
        for b in names_eng[i + 1 :]:
            pairs.append((a, b, agree(sign_map[a], sign_map[b])))

    sign_agree = dict(
        n=int(len(next(iter(sign_map.values())))),
        pairs={f"{a}_vs_{b}": v for a, b, v in pairs},
        note="calibrated redesign; legacy = sealed logistic map; H5 uses bucket history ≤ t",
    )
    for a, b, v in pairs:
        print(f"  sign {a} vs {b}: {fmt(v, 4)}", flush=True)

    # Recommendation — diagnostic only
    excess = {n: jump_cal[n]["boundary_minus_interior_mean"] for n in jump_cal}
    a0_ex = excess.get("A0_shared_logistic")
    rec = (
        "A0_shared_logistic remains the a priori Choice-A freeze candidate "
        f"(mech. V_SELECT winner; CAL boundary excess={fmt(a0_ex)}). "
        "TEST consecutive-query excess and multi-spec sign(ΔV) agree are WARNING / "
        "DIAGNOSTIC only — not the freeze selection rule. "
        "Still missing for full V-4: D_switch(x), quiet/event strata, frame-refresh."
    )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_REDESIGN_CONTRACT_20260919.md §7; wave2 A0/A3/A2b",
        evaluation_map="V_final = g(f(T(X))); primary tables use calibrated probs",
        sample="MAIN TEST bucket queries + MAIN_TEST T∩valid_h90 (exploratory; prior TEST exposure)",
        role_of_test="DIAGNOSTIC_LEDGER_NOT_SELECTION — Choice A / A0 locked a priori",
        consecutive_jumps_calibrated=jump_cal,
        consecutive_jumps_raw_diagnostic=jump_raw,
        engagement_sign_agree=sign_agree,
        recommendation=rec,
        provisional_freeze_candidate="A0_shared_logistic",
        v4_coverage="PARTIAL: consecutive bucket jumps + multi-spec sign agree; "
        "missing quiet / frame-refresh / D_switch(x)",
        with_h5=bool(args.with_h5),
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

    out_json = v_dir / "continuity_results.json"
    out_md = v_dir / "CONTINUITY_REPORT.md"
    out_json.write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# V-4 continuity — wave-2 freeze contenders (PARTIAL)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        f"**Provisional freeze candidate (a priori Choice A):** `{payload['provisional_freeze_candidate']}`",
        "",
        rec,
        "",
        "## Evaluation map",
        "",
        "Primary numbers use **calibrated** $g\\circ f$ (PosSlopeSigmoid from each arm's V_CAL).",
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
        "### Diagnostic — RAW excess (b−i)",
        "",
        "| Model | excess |",
        "|---|---:|",
    ]
    for name, blk in jump_raw.items():
        lines.append(f"| {name} | {fmt(blk.get('boundary_minus_interior_mean'), 4)} |")

    lines += [
        "",
        "## 2. Engagement sign(ΔV) agree (15.16 T, h90; calibrated)",
        "",
        f"n={sign_agree['n']}",
        "",
        "| Pair | agree (nonzero) |",
        "|---|---:|",
    ]
    for a, b, v in pairs:
        lines.append(f"| {a} vs {b} | {fmt(v, 4)} |")
    lines += [
        "",
        "## Reading (narrow)",
        "",
        "- Boundary excess on stored queries ≠ proof of model-switch artifact.",
        "- High multi-spec sign agree supports SVI label stability across top WP arms.",
        "- Still needed for full V-4: $D_{\\mathrm{switch}}(x)$, quiet/event strata.",
        "- Coverage: **partial V-4** only; A0 remains freeze candidate pending those checks.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    man = dict(
        provisional_freeze_candidate="A0_shared_logistic",
        status="CONTINUITY_PARTIAL_WAVE2_CONTENDERS",
        mechanical_select_winner_on_V_SELECT="A0_shared_logistic",
        reason=(
            "Choice A a priori; A0 leads L_time; continuity is diagnostic — "
            "see CONTINUITY_REPORT.md"
        ),
        continuity_report=str(out_md.relative_to(REPO)).replace("\\", "/"),
        generated=payload["generated"],
        epistemic=ROLE,
    )
    (v_dir / "continuity_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_md)
    print("recommendation:", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
