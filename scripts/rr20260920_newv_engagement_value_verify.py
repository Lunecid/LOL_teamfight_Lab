#!/usr/bin/env python3
"""New-V engagement value-change verification (J-RQ1 part ② / M-RQ2).

Does NOT retrain V or start q. Engagement definition fixed (T, onset/cutoff, h90
primary). Only the evaluator is the frozen fit85 MLP Expanded bundle.

Writes: outputs/newv_engagement_value_verify_20260920/
        docs/NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md
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
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WAVE4 = REPO / "outputs" / "v_redesign_wave4_corrected_20260919"
WAVE2 = REPO / "outputs" / "v_redesign_wave2_20260919"
BUNDLE = WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.joblib"
OUT = REPO / "outputs" / "newv_engagement_value_verify_20260920"
ROLE = "J_RQ1_PART2_MEASUREMENT_NOT_Q_SELECTION"
HS = (60, 90, 120)
TIME_BANDS = ((0.0, 10.0, "t_0_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf"))
P_EDGES = np.array([0.0, 0.20, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.80, 1.01])
P_LABELS = ["0-20", "20-35", "35-40", "40-45", "45-55", "55-60", "60-65", "65-80", "80-100"]

from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402
from v_redesign_feature_adapters import FeatureSchema, ProfileBundle, match_holdout_mask  # noqa: E402


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    w = (1.0 / c[inv]).astype(np.float64)
    return w / w.mean()


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return "NA"
    return f"{v:.{nd}f}"


def scrub(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def summarize_delta(dV: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    w = match_weights(g)
    abs_d = np.abs(dV)
    y = (dV > 0).astype(np.float64)
    exact0 = dV == 0
    return dict(
        n=int(len(dV)),
        n_matches=int(len(np.unique(g))),
        P_SVI_pos=float(np.average(y, weights=w)),
        mean_deltaV=float(np.average(dV, weights=w)),
        mean_abs=float(np.average(abs_d, weights=w)),
        median_abs=float(np.median(abs_d)),
        p25_abs=float(np.quantile(abs_d, 0.25)),
        p75_abs=float(np.quantile(abs_d, 0.75)),
        p90_abs=float(np.quantile(abs_d, 0.90)),
        frac_exact_zero=float(np.mean(exact0)),
        frac_abs_lt_1e_3=float(np.mean(abs_d < 1e-3)),
        frac_abs_gt_0_05=float(np.mean(abs_d > 0.05)),
        n_exact_zero=int(exact0.sum()),
    )


def agreement_sign(y_svi: np.ndarray, material: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    s = np.sign(material).astype(int)
    y_pm = np.where(y_svi == 1, 1, -1)
    decided = s != 0
    out: Dict[str, Any] = dict(n=int(len(y_svi)), n_decided=int(decided.sum()), tie_share=float((~decided).mean()))
    if not decided.any():
        out.update(agreement_rate=None, matches_decided=0)
        return out
    w = match_weights(g[decided])
    agree = y_pm[decided] == s[decided]
    out["agreement_rate"] = float(np.average(agree, weights=w))
    out["matches_decided"] = int(len(np.unique(g[decided])))
    return out


def slot_diff(X: np.ndarray, names: Sequence[str], field: str) -> np.ndarray:
    ix = {n: i for i, n in enumerate(names)}
    blue = [ix[f"participant_slot{s}_{field}"] for s in range(0, 5)]
    red = [ix[f"participant_slot{s}_{field}"] for s in range(5, 10)]
    return X[:, blue].sum(axis=1) - X[:, red].sum(axis=1)


def count_net(during: np.ndarray, after: np.ndarray, keys: Sequence[str], blue: str, red: str) -> np.ndarray:
    ki = {str(k): i for i, k in enumerate(keys)}
    C = during + after
    return C[:, ki[blue]].astype(float) - C[:, ki[red]].astype(float)


def strata_block(dV: np.ndarray, p_pre: np.ndarray, tmin: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    by_time = []
    for lo, hi, name in TIME_BANDS:
        m = (tmin >= lo) & (tmin < hi)
        if m.sum() < 50:
            continue
        s = summarize_delta(dV[m], g[m])
        s.update(band=name, t_lo=lo, t_hi=hi)
        by_time.append(s)

    by_p = []
    bins = np.digitize(p_pre, P_EDGES) - 1
    for i, lab in enumerate(P_LABELS):
        m = bins == i
        if m.sum() < 50:
            continue
        s = summarize_delta(dV[m], g[m])
        s.update(p_bin=lab, E_deltaV=s["mean_deltaV"])
        by_p.append(s)

    b40 = (p_pre >= 0.40) & (p_pre <= 0.60)
    b45 = (p_pre >= 0.45) & (p_pre <= 0.55)
    return dict(
        overall=summarize_delta(dV, g),
        by_time=by_time,
        by_p_pre=by_p,
        B40=summarize_delta(dV[b40], g[b40]) if b40.any() else dict(n=0),
        B45=summarize_delta(dV[b45], g[b45]) if b45.any() else dict(n=0),
        B40_n=int(b40.sum()),
        B40_match_n=int(len(np.unique(g[b40]))) if b40.any() else 0,
    )


def predict_a0(data_root: Path, X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    import joblib
    from v_redesign_evaluator_bundle import PosSlopeSigmoid

    a0 = joblib.load(WAVE2 / "models" / "A0_shared_logistic.joblib")
    w2 = json.loads((WAVE2 / "results.json").read_text(encoding="utf-8"))
    cal = w2["selection"]["A0_shared_logistic"]["calib"]
    raw = a0["model"].predict_proba(X[:, a0["keep"]])[:, 1]
    return PosSlopeSigmoid.from_dict(cal).transform(raw)


def predict_lr_expanded(X: np.ndarray, bun: ProfileBundle, lr) -> np.ndarray:
    from v_redesign_evaluator_bundle import PosSlopeSigmoid

    w4 = json.loads((WAVE4 / "results.json").read_text(encoding="utf-8"))
    cal = w4["selection"]["A_LR_expanded"]["calib"]
    raw = lr.predict_proba(bun.matrix_onehot(X))[:, 1]
    return PosSlopeSigmoid.from_dict(cal).transform(raw)


def build_table(E, keep, p_pre, p_post, h, bundle_sha, fit_scope):
    dV = p_post - p_pre
    exact0 = dV == 0
    missing = ~(np.isfinite(p_pre) & np.isfinite(p_post))
    y = np.full(len(dV), -1, dtype=np.int8)
    y[~missing] = (dV[~missing] > 0).astype(np.int8)
    same_frame = E[f"post_snapshot_h{h}"][keep] == E["pre_snapshot"][keep]
    return dict(
        match=E["match"][keep].astype(str),
        s=E["s"][keep].astype(np.int64),
        sub_role=E["sub_role"][keep].astype(str),
        L=E["L"][keep].astype(np.int64),
        q_pre=E["q_pre"][keep].astype(np.int64),
        endpoint=E[f"endpoint_h{h}"][keep].astype(np.int64),
        pre_snapshot=E["pre_snapshot"][keep].astype(np.int64),
        post_snapshot=E[f"post_snapshot_h{h}"][keep].astype(np.int64),
        last_frame=E["last_frame"][keep].astype(np.int64),
        p_pre=p_pre.astype(np.float64),
        p_post=p_post.astype(np.float64),
        delta_V=dV.astype(np.float64),
        Y_SVI=y,
        exact_zero=exact0.astype(np.int8),
        missing_score=missing.astype(np.int8),
        same_frame=same_frame.astype(np.int8),
        B40=((p_pre >= 0.40) & (p_pre <= 0.60) & (~missing)).astype(np.int8),
        horizon_ms=np.full(len(dV), h * 1000, dtype=np.int32),
    )


def material_block(E, keep, y_svi, dV, g, p_pre, names, h):
    keys = [str(k) for k in E["count_keys"]]
    during = E["during"][keep]
    after = E[f"after_h{h}"][keep]

    def net(b, r):
        return count_net(during, after, keys, b, r)

    epic = (
        net("baron_blue", "baron_red") + net("dragon_blue", "dragon_red")
        + net("elder_blue", "elder_red") + net("herald_blue", "herald_red")
        + net("horde_blue", "horde_red") + net("atakhan_blue", "atakhan_red")
        + net("soul_owned_blue", "soul_owned_red")
    )
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    obj = epic + struct
    Xpre, Xpost = E["X_pre"][keep], E[f"X_post_h{h}"][keep]
    kill_diff = slot_diff(Xpost, names, "kills") - slot_diff(Xpre, names, "kills")
    alive_post = slot_diff(Xpost, names, "alive")

    ok = y_svi >= 0
    y, gg, pp, dd = y_svi[ok], g[ok], p_pre[ok], dV[ok]
    axes = {
        "epic_net": epic[ok],
        "structure_net": struct[ok],
        "objective_net": obj[ok],
        "kill_diff": kill_diff[ok],
        "alive_diff_post": alive_post[ok],
    }
    report = dict(
        n=int(ok.sum()),
        axes={},
        note="NEW Y_SVI from fit85 MLP — not old-label rates. Not independent fight-winner accuracy.",
    )
    for name, val in axes.items():
        block = agreement_sign(y, val, gg)
        if name == "kill_diff":
            s = np.sign(val).astype(int)
            y_pm = np.where(y == 1, 1, -1)
            decided = s != 0
            disagree = decided & (y_pm != s)
            so = np.sign(obj[ok]).astype(int)
            report["kill_disagreement"] = dict(
                n_decided=int(decided.sum()),
                n_disagree=int(disagree.sum()),
                disagree_share=float(disagree.sum() / max(1, decided.sum())),
                among_disagree_obj_agrees_SVI=int((disagree & (so != 0) & (so == y_pm)).sum()),
                among_disagree_obj_agrees_kill=int((disagree & (so != 0) & (so == s)).sum()),
                among_disagree_obj_tie=int((disagree & (so == 0)).sum()),
                SVI_blue_kill_red=int((disagree & (y == 1) & (s < 0)).sum()),
                SVI_red_kill_blue=int((disagree & (y == 0) & (s > 0)).sum()),
                mean_abs_dV_disagree=float(np.mean(np.abs(dd[disagree]))) if disagree.any() else None,
            )
        b40 = (pp >= 0.40) & (pp <= 0.60)
        block["B40"] = agreement_sign(y[b40], val[b40], gg[b40]) if b40.any() else dict(n=0)
        report["axes"][name] = block
    return report


def horizon_stability(E, lab_keys, ev, W):
    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    pre_ok = E["pre_ok"] == 1
    base = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    base &= pre_ok
    common = base.copy()
    for h in HS:
        common &= E[f"valid_h{h}"] == 1
    Xpre = E["X_pre"][common]
    p0 = predict_calibrated(ev, Xpre)
    g = em[common]
    per_h, Y, dVs = {}, {}, {}
    for h in HS:
        p1 = predict_calibrated(ev, E[f"X_post_h{h}"][common])
        dV = p1 - p0
        Y[h] = (dV > 0).astype(np.int8)
        dVs[h] = dV
        yw = np.asarray([W.get(m, -1) for m in g.tolist()], dtype=int)
        sealed = yw >= 0
        block = dict(n=int(common.sum()), summary=summarize_delta(dV, g), V_post_vs_W=None)
        if sealed.any():
            from sklearn.metrics import roc_auc_score
            w = match_weights(g[sealed])
            pp, yy = p1[sealed], yw[sealed].astype(float)
            block["V_post_vs_W"] = dict(
                brier=float(np.average((pp - yy) ** 2, weights=w)),
                auc=float(roc_auc_score(yy, pp, sample_weight=w)),
            )
        per_h[h] = block
    flips = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        m = (np.sign(dVs[a]) != 0) & (np.sign(dVs[b]) != 0)
        flips[f"h{a}_vs_h{b}"] = dict(
            svi_flip_rate=float(np.mean(Y[a] != Y[b])),
            mean_abs_delta_diff=float(np.mean(np.abs(dVs[a] - dVs[b]))),
            sign_agree_nonzero=float(np.mean(np.sign(dVs[a])[m] == np.sign(dVs[b])[m])) if m.any() else None,
        )
    h90_only = base & (E["valid_h90"] == 1)
    return dict(
        design="common_valid_h60_h90_h120_intersection",
        n_common=int(common.sum()),
        n_h90_only=int(h90_only.sum()),
        composition_note="Horizon comparisons use intersection of valid cases; separate from sample-composition change.",
        per_horizon=per_h,
        pairwise=flips,
        n_matches_common=int(len(np.unique(g))),
    )


def peer_sign_agree(data_root, E, keep, names, s_mlp):
    import joblib
    Xpre, Xpost = E["X_pre"][keep], E["X_post_h90"][keep]
    out: Dict[str, Any] = {}
    try:
        s = np.sign(predict_a0(data_root, Xpost, names) - predict_a0(data_root, Xpre, names))
        m = np.isfinite(s_mlp) & np.isfinite(s) & (s_mlp != 0) & (s != 0)
        out["MLP_vs_A0_wave2"] = float(np.mean(s_mlp[m] == s[m])) if m.any() else None
        out["MLP_vs_A0_n_compared"] = int(m.sum())
    except Exception as e:
        out["MLP_vs_A0_wave2"] = None
        out["MLP_vs_A0_error"] = str(e)

    lr_path = WAVE4 / "models" / "A_LR_expanded.joblib"
    if lr_path.is_file():
        try:
            import fc20260915_data as D
            import fc20260915_common as C
            L = D.Layout(False)
            TR = D.load_v_rows(L, "MAIN", [f"fold{k}" for k in range(C.N_FOLDS)], bucket_only=True)
            schema = FeatureSchema.from_names(TR["names"])
            stop = match_holdout_mask(TR["match"], 0.15, seed=7)
            bun = ProfileBundle(schema, "expanded").fit(TR["X"][~stop])
            lr = joblib.load(lr_path)["model"]
            s = np.sign(predict_lr_expanded(Xpost, bun, lr) - predict_lr_expanded(Xpre, bun, lr))
            m = np.isfinite(s_mlp) & np.isfinite(s) & (s_mlp != 0) & (s != 0)
            out["MLP_vs_A_LR_expanded_fit85"] = float(np.mean(s_mlp[m] == s[m])) if m.any() else None
            out["MLP_vs_LR_n_compared"] = int(m.sum())
        except Exception as e:
            out["MLP_vs_A_LR_expanded_fit85"] = None
            out["MLP_vs_LR_error"] = str(e)
    out["reading"] = "Stability diagnostic only; do not retune V to maximize peer agreement."
    return out


def observation_refresh(same_frame, dV, g):
    same = same_frame.astype(bool)
    out = dict(
        n_same_frame=int(same.sum()),
        n_refreshed=int((~same).sum()),
        same_frame_share=float(same.mean()),
        note="same_frame: pre_snapshot == post_snapshot_h90.",
    )
    if same.any():
        out["same_frame"] = summarize_delta(dV[same], g[same])
    if (~same).any():
        out["refreshed"] = summarize_delta(dV[~same], g[~same])
    return out


def write_report(payload, path):
    d, mat, hor, peer, ref = (
        payload["distribution_h90"],
        payload["material_h90"],
        payload["horizon_stability"],
        payload["peer_sign_agree"],
        payload["observation_refresh"],
    )
    lines = [
        "# New-V engagement value-change verification",
        "",
        f"Generated: {payload['generated']}",
        f"**Bundle SHA256:** `{payload['bundle_sha256']}`",
        f"**Fit scope:** `{payload['fit_scope']}`",
        f"**RQ:** Journal **J-RQ1** part ② / Master **M-RQ2** (not q / J-RQ2).",
        "",
        "V freeze close: [V_EVALUATOR_FREEZE_CLOSE_20260920.md](V_EVALUATOR_FREEZE_CLOSE_20260920.md)",
        "",
        "## 0. Design locks",
        "",
        "- Engagement definition unchanged (T, onset/cutoff, h90 primary).",
        "- Only V replaced by fit85 MLP Expanded.",
        "- B40 recomputed from **new** p_pre; old SVI/B40 not reused.",
        "- Exact ΔV=0 → Y_SVI=0; missing scores flagged, never coerced to 0/red.",
        "",
        "## 1. Primary table (h90)",
        "",
        f"- n={d['overall']['n']:,} / matches={d['overall']['n_matches']:,}",
        f"- P(SVI=1)={fmt(d['overall']['P_SVI_pos'], 3)}",
        f"- mean ΔV={fmt(d['overall']['mean_deltaV'])}; mean |ΔV|={fmt(d['overall']['mean_abs'])}; median |ΔV|={fmt(d['overall']['median_abs'])}",
        f"- exact 0 share={fmt(d['overall']['frac_exact_zero'], 4)} (n={d['overall']['n_exact_zero']})",
        f"- **B40 (new p_pre):** n={d['B40_n']:,} / matches={d['B40_match_n']:,}; P(SVI=1)={fmt(d['B40'].get('P_SVI_pos'), 3)}; mean |ΔV|={fmt(d['B40'].get('mean_abs'))}",
        "",
        "### By match time",
        "",
        "| Band | n | P(SVI=1) | E[ΔV] | mean |ΔV| | median |ΔV| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in d["by_time"]:
        lines.append(
            f"| {r['band']} | {r['n']} | {fmt(r['P_SVI_pos'], 3)} | {fmt(r['mean_deltaV'])} | {fmt(r['mean_abs'])} | {fmt(r['median_abs'])} |"
        )
    lines += [
        "",
        "### By new p_pre",
        "",
        "| p_pre bin | n | P(SVI=1) | E[ΔV] | mean |ΔV| |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in d["by_p_pre"]:
        lines.append(
            f"| {r['p_bin']} | {r['n']} | {fmt(r['P_SVI_pos'], 3)} | {fmt(r['E_deltaV'])} | {fmt(r['mean_abs'])} |"
        )
    lines += ["", "## 2. Material correspondence (new Y_SVI)", "", mat.get("note", ""), ""]
    lines += ["| Axis | n_decided | agree (match-wtd) | tie share | B40 agree |", "|---|---:|---:|---:|---:|"]
    for name, ax in mat.get("axes", {}).items():
        b40 = ax.get("B40") or {}
        lines.append(
            f"| {name} | {ax.get('n_decided')} | {fmt(ax.get('agreement_rate'), 3)} | {fmt(ax.get('tie_share'), 3)} | {fmt(b40.get('agreement_rate'), 3)} |"
        )
    kd = mat.get("kill_disagreement") or {}
    if kd:
        lines += [
            "",
            "### Kill-axis disagreements",
            "",
            f"- decided={kd.get('n_decided')}; disagree share={fmt(kd.get('disagree_share'), 3)}",
            f"- among disagree: obj↔SVI={kd.get('among_disagree_obj_agrees_SVI')}, obj↔kill={kd.get('among_disagree_obj_agrees_kill')}, obj tie={kd.get('among_disagree_obj_tie')}",
            f"- SVI blue & kill red={kd.get('SVI_blue_kill_red')}; SVI red & kill blue={kd.get('SVI_red_kill_blue')}",
        ]
    lines += [
        "",
        "## 3. Stability",
        "",
        f"### Horizons (common valid n={hor['n_common']:,}; h90-only n={hor['n_h90_only']:,})",
        "",
        hor["composition_note"],
        "",
        "| Pair | SVI flip | mean |ΔΔV| | nonzero sign agree |",
        "|---|---:|---:|---:|",
    ]
    for k, v in hor["pairwise"].items():
        lines.append(
            f"| {k} | {fmt(v['svi_flip_rate'], 3)} | {fmt(v['mean_abs_delta_diff'])} | {fmt(v.get('sign_agree_nonzero'), 3)} |"
        )
    lines += ["", "### Peer evaluators (h90)", ""]
    for k, v in peer.items():
        if k.startswith("MLP_vs_") and isinstance(v, float):
            lines.append(f"- {k}: {fmt(v, 3)}")
    lines.append(f"- {peer.get('reading', '')}")
    lines += [
        "",
        "### Observation refresh (h90)",
        "",
        f"- same_frame share={fmt(ref.get('same_frame_share'), 3)} (n_same={ref.get('n_same_frame')}, n_refreshed={ref.get('n_refreshed')})",
    ]
    if "same_frame" in ref:
        lines.append(
            f"- same_frame mean |ΔV|={fmt(ref['same_frame'].get('mean_abs'))}; P(SVI=1)={fmt(ref['same_frame'].get('P_SVI_pos'), 3)}"
        )
    if "refreshed" in ref:
        lines.append(
            f"- refreshed mean |ΔV|={fmt(ref['refreshed'].get('mean_abs'))}; P(SVI=1)={fmt(ref['refreshed'].get('P_SVI_pos'), 3)}"
        )
    lines += [
        "",
        "## 4. Quiet (no-kill) reference",
        "",
        payload.get("quiet", {}).get("status", "deferred"),
        "",
        "## 5. RQ1 completion bar",
        "",
        "Measurement table + three analysis axes are here. RQ1 closes when the manuscript can state which V, which window, what change, and how it corresponds to material outcomes / definition changes — not when SVI is proven correct.",
        "",
        "q (J-RQ2) only after this measurement story is written. TRAIN SVI needs OOF V — separate from this freeze bundle.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-quiet", action="store_true")
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    OUT.mkdir(parents=True, exist_ok=True)
    if not BUNDLE.is_file():
        raise SystemExit(f"missing bundle {BUNDLE}")

    bundle_sha = sha256_file(BUNDLE)
    print("bundle SHA256", bundle_sha, flush=True)
    ev = load_evaluator(BUNDLE)
    fit_scope = str(ev.get("fit_scope") or "")

    L = D.Layout(False)
    print("load MAIN TEST engagements…", flush=True)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=True)
    names = list(E["names"])
    W_raw = D.load_outcomes(L, "MAIN", ["TEST"], purpose="newv engagement verify")
    W = {str(k): int(v[0]) if isinstance(v, (tuple, list, np.ndarray)) else int(v) for k, v in W_raw.items()}

    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False,
    )
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False,
    )
    m_lab = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    lab_keys = set(zip(lab["match"][m_lab].astype(str).tolist(), lab["s"][m_lab].astype(np.int64).tolist()))

    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    keep = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    keep &= (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    print(f"primary h90 keep={int(keep.sum())}", flush=True)

    print("score MLP pre/post h90…", flush=True)
    p_pre = predict_calibrated(ev, E["X_pre"][keep])
    p_post = predict_calibrated(ev, E["X_post_h90"][keep])
    dV = p_post - p_pre
    g = em[keep]
    tmin = es[keep].astype(float) / 60000.0
    miss = ~(np.isfinite(p_pre) & np.isfinite(p_post))
    y_svi = np.full(len(dV), -1, dtype=np.int8)
    y_svi[~miss] = (dV[~miss] > 0).astype(np.int8)

    table = build_table(E, keep, p_pre, p_post, 90, bundle_sha, fit_scope)
    table_path = OUT / "engagement_table_h90_MAIN_TEST.npz"
    np.savez_compressed(table_path, **{k: v for k, v in table.items() if isinstance(v, np.ndarray)})
    meta = dict(bundle_sha256=bundle_sha, fit_scope=fit_scope, n=int(keep.sum()), evaluator="A_MLP_expanded")
    (OUT / "engagement_table_h90_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", table_path, flush=True)

    dist = strata_block(dV[~miss], p_pre[~miss], tmin[~miss], g[~miss])
    print("material…", flush=True)
    material = material_block(E, keep, y_svi, dV, g, p_pre, names, 90)
    print("horizon stability…", flush=True)
    horizon = horizon_stability(E, lab_keys, ev, W)
    print("peer sign agree…", flush=True)
    s_mlp = np.sign(dV)
    s_mlp[miss] = 0
    peers = peer_sign_agree(data_root, E, keep, names, s_mlp)
    same_frame = E["post_snapshot_h90"][keep] == E["pre_snapshot"][keep]
    refresh = observation_refresh(same_frame[~miss], dV[~miss], g[~miss])

    quiet = dict(
        status="Deferred. Quiet no-kill is a background-drift reference, not a causal control. Follow-up with --with-quiet / phase2 pattern on this bundle."
    )
    if args.with_quiet:
        quiet = dict(status="flag set; implement quiet smoke against MLP bundle in follow-up (not in this lean runner).")

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        bundle=str(BUNDLE.relative_to(REPO)).replace("\\", "/"),
        bundle_sha256=bundle_sha,
        fit_scope=fit_scope,
        table_path=str(table_path.relative_to(REPO)).replace("\\", "/"),
        distribution_h90=dist,
        material_h90=material,
        horizon_stability=horizon,
        peer_sign_agree=peers,
        observation_refresh=refresh,
        quiet=quiet,
    )
    (OUT / "results.json").write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = REPO / "docs" / "NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md"
    write_report(payload, md_path)
    (WAVE4 / "evaluators" / "A_MLP_expanded_evaluator.sha256").write_text(bundle_sha + "\n", encoding="utf-8")
    print("wrote", md_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
