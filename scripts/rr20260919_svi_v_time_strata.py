#!/usr/bin/env python3
"""Frame-aligned V̂ → W performance by time band (+ engagement pre/post on 15.16 T).

Design: docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md
V̂(S_t) is static parameters + dynamic state; report quality on the data's time grid.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_NOT_CONFIRMATORY"
TIME_BANDS = ((0, 10, "t_0_10"), (10, 20, "t_10_20"), (20, 30, "t_20_30"), (30, 1e9, "t_30_inf"))


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    br = float(np.average((p - y) ** 2, weights=w))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(n=int(len(y)), n_matches=int(len(np.unique(np.arange(len(y))))), brier=br, auc=auc)


def metrics_g(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    w = match_weights(g)
    out = metrics(y, p, w)
    out["n_matches"] = int(len(np.unique(g)))
    return out


def band_table(y: np.ndarray, p: np.ndarray, g: np.ndarray, tmin: np.ndarray) -> Dict[str, Any]:
    rows = {"all": metrics_g(y, p, g)}
    for lo, hi, name in TIME_BANDS:
        m = (tmin >= lo) & (tmin < hi)
        if int(m.sum()) < 50:
            rows[name] = dict(n=int(m.sum()), skipped=True)
            continue
        rows[name] = metrics_g(y[m], p[m], g[m])
    return rows


def load_match_w(data_root: Path) -> Dict[str, int]:
    """winner_blue by match from sealed outcomes chunks."""
    d = data_root / "outputs" / "full_corpus_training_20260915" / "extract" / "MAIN" / "outcomes_SEALED"
    out: Dict[str, int] = {}
    for p in sorted(d.glob("chunk_*.npz")):
        z = np.load(p, allow_pickle=False)
        for m, w in zip(z["o_match"].astype(str), z["o_winner_blue"].astype(int)):
            out[m] = int(w)
    return out


def eval_v_npz(path: Path, bucket_only: bool = False) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    y = z["winner_blue"].astype(int)
    g = z["match"].astype(str)
    p = z["p_sigmoid_pos"].astype(float) if "p_sigmoid_pos" in z.files else z["p_raw"].astype(float)
    tmin = z["query_ms"].astype(float) / 60000.0
    if bucket_only and "is_bucket_sample" in z.files:
        m = z["is_bucket_sample"].astype(bool)
        y, g, p, tmin = y[m], g[m], p[m], tmin[m]
    return dict(
        path=str(path),
        n=int(len(y)),
        sha=str(z["v_final_sha256"]) if "v_final_sha256" in z.files else None,
        bands=band_table(y, p, g, tmin),
    )


def eval_engagement_anchors(data_root: Path, match_w: Dict[str, int]) -> Dict[str, Any]:
    """15.16 T engagements: V̂ at pre and post (h90) vs match W."""
    lab = np.load(
        data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
        allow_pickle=False)
    coh = np.load(
        data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
        allow_pickle=False)
    # cohort code T == 1 in prior scripts
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    g = lab["match"][m].astype(str)
    tmin = lab["s"][m].astype(float) / 60000.0
    p_pre = lab["p_pre"][m].astype(float)
    p_post = lab["p_post_h90"][m].astype(float)
    y = np.array([match_w.get(x, -1) for x in g], dtype=int)
    ok = y >= 0
    n_miss = int((~ok).sum())
    g, tmin, p_pre, p_post, y = g[ok], tmin[ok], p_pre[ok], p_post[ok], y[ok]
    return dict(
        sample="MAIN_TEST T ∩ valid_h90 with sealed outcomes",
        n=int(len(y)),
        n_matches=int(len(np.unique(g))),
        missing_outcome_rows=n_miss,
        pre=band_table(y, p_pre, g, tmin),
        post_h90=band_table(y, p_post, g, tmin),
        note="Same frozen V̂; pre vs post frames on engagement clock (onset s)",
    )


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# V̂ → W by time band (frame-aligned)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        "",
        "Contract: `docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md`.",
        "V̂(S_t)=P̂(W=1|S_t); dynamics via changing S_t on the public frame grid.",
        "",
        "## 1. Timeline / bucket queries — MAIN_TEST `v_MAIN_TEST.npz`",
        "",
        "| Band | n | Brier | AUC |",
        "|---|---:|---:|---:|",
    ]
    for k, v in (payload["v_MAIN_TEST"]["bands"] or {}).items():
        if v.get("skipped"):
            continue
        lines.append(f"| {k} | {v['n']} | {fmt(v['brier'], 6)} | {fmt(v['auc'], 4)} |")

    if payload.get("v_MAIN_TEST_bucket"):
        lines += ["", "### Bucket-sample only", "", "| Band | n | Brier | AUC |", "|---|---:|---:|---:|"]
        for k, v in payload["v_MAIN_TEST_bucket"]["bands"].items():
            if v.get("skipped"):
                continue
            lines.append(f"| {k} | {v['n']} | {fmt(v['brier'], 6)} | {fmt(v['auc'], 4)} |")

    eng = payload["engagement_15_16_T"]
    lines += [
        "",
        "## 2. Engagement anchors — 15.16 T (SVI path)",
        "",
        f"n={eng['n']} / matches={eng['n_matches']}. Pre = p_pre; Post = p_post_h90.",
        "",
        "### Pre-fight V̂(S_pre)",
        "",
        "| Band | n | Brier | AUC |",
        "|---|---:|---:|---:|",
    ]
    for k, v in eng["pre"].items():
        if v.get("skipped"):
            continue
        lines.append(f"| {k} | {v['n']} | {fmt(v['brier'], 6)} | {fmt(v['auc'], 4)} |")
    lines += ["", "### Post-fight V̂(S_end) h90", "", "| Band | n | Brier | AUC |", "|---|---:|---:|---:|"]
    for k, v in eng["post_h90"].items():
        if v.get("skipped"):
            continue
        lines.append(f"| {k} | {v['n']} | {fmt(v['brier'], 6)} | {fmt(v['auc'], 4)} |")

    lines += [
        "",
        "## 3. External — by time band (same clock)",
        "",
    ]
    for name, block in (payload.get("external") or {}).items():
        if "bands" not in block:
            lines.append(f"- {name}: {block.get('error', block)}")
            continue
        lines += [
            f"### {name}",
            "",
            "| Band | n | Brier | AUC |",
            "|---|---:|---:|---:|",
        ]
        for k, v in block["bands"].items():
            if v.get("skipped"):
                continue
            lines.append(f"| {k} | {v['n']} | {fmt(v['brier'], 6)} | {fmt(v['auc'], 4)} |")
        lines.append("")

    lines += [
        "",
        "## Reading",
        "",
        "- **Band tables are primary;** pooled AUC/Brier is a summary only.",
        "- Expect large early→late skill gaps (more information in late frames).",
        "- Engagement pre/post tables justify reading ΔV̂ on fight-relevant frames.",
        "- Higher late-game V̂ AUC does not by itself prove SVI is fight-win truth.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "svi_v_time_strata_20260919")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v_dir = data_root / "outputs" / "full_corpus_training_20260915" / "eval" / "predictions"
    print("load outcomes…", flush=True)
    match_w = load_match_w(data_root)
    print("matches with W", len(match_w), flush=True)

    print("v_MAIN_TEST…", flush=True)
    v_main = eval_v_npz(v_dir / "v_MAIN_TEST.npz", bucket_only=False)
    print("v_MAIN_TEST bucket…", flush=True)
    v_bucket = eval_v_npz(v_dir / "v_MAIN_TEST.npz", bucket_only=True)

    print("engagement anchors…", flush=True)
    eng = eval_engagement_anchors(data_root, match_w)

    external = {}
    for name in ("v_EXT_KR_16.13.npz", "v_EXT_NA1_16.13.npz", "v_EXT_KR_16.15.npz"):
        p = v_dir / name
        if p.is_file():
            print(name, flush=True)
            external[name] = eval_v_npz(p)

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md",
        time_bands=[b[2] for b in TIME_BANDS],
        v_MAIN_TEST=v_main,
        v_MAIN_TEST_bucket=v_bucket,
        engagement_15_16_T=eng,
        external=external,
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
    write_report(out_dir, payload)
    print("wrote", out_dir)
    print("MAIN all", v_main["bands"]["all"])
    print("eng pre all", eng["pre"]["all"], "post", eng["post_h90"]["all"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
