#!/usr/bin/env python3
"""E3 — external paired match-cluster bootstrap for frozen q vs PT_flex.

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §6
Task: .ai/tasks/T017.md

Does not overwrite frozen REVIEW_RESPONSE_RRX_* artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rr20260920_review_response_rrx_external import (  # noqa: E402
    BUNDLE,
    COHORTS,
    _data_root,
    _setup,
    check_feature_order,
    load_ext_failclosed,
    predict_q_frozen,
    reference_main_cols,
)
from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated  # noqa: E402

MAIN_SETS = ("KR_16.13", "NA1_16.13")  # primary family
PILOT_SETS = ("KR_16.15", "KR_16.14_pilot")


def match_level_delta(
    match: np.ndarray, y: np.ndarray, q: np.ndarray, pt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-match mean of (q-y)^2 - (pt-y)^2; return (unique_matches, d_m)."""
    g = match.astype(str)
    y = y.astype(np.float64)
    q = q.astype(np.float64)
    pt = pt.astype(np.float64)
    row_d = (q - y) ** 2 - (pt - y) ** 2
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums, inv, row_d)
    np.add.at(counts, inv, 1.0)
    return uniq, sums / counts


def bootstrap_mean(
    d_m: np.ndarray, n_boot: int, seed: int
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(d_m)
    if n == 0:
        return dict(n_matches=0, estimate=None)
    point = float(np.mean(d_m))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d_m[idx].mean(axis=1)
    return dict(
        n_matches=int(n),
        estimate=point,
        ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        ci9875=[float(np.quantile(boots, 0.00625)), float(np.quantile(boots, 0.99375))],
        frac_boot_gt0=float(np.mean(boots > 0)),
        n_boot=int(n_boot),
        seed=int(seed),
    )


def score_cohort(
    data_root: Path,
    D,
    L,
    cohort_tag: str,
    q_path: Path,
    pt_path: Path,
    ref_cols: List[str],
    ev,
    set_id: str,
    coh_name: str,
) -> Dict[str, Any]:
    import joblib

    pack = load_ext_failclosed(data_root, D, L, set_id, coh_name, cohort_tag=cohort_tag)
    if not pack.get("ok"):
        return dict(ok=False, reason=pack.get("reason"), set_id=set_id, cohort=cohort_tag)

    logit = joblib.load(q_path)
    pt_flex = joblib.load(pt_path)
    feat_check = check_feature_order(ref_cols, list(pack["cols"]), logit["num_ix"])
    if not feat_check["ok"]:
        return dict(ok=False, reason="feature_order_mismatch", feature_check=feat_check)

    try:
        W_raw = D.load_outcomes(L, set_id, None, purpose=f"E3 {set_id}")
        Wmap = {
            str(k): int(v[0]) if isinstance(v, (tuple, list, np.ndarray)) else int(v)
            for k, v in W_raw.items()
        }
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, reason=f"outcomes_load:{e}")

    g_all = pack["match"].astype(str)
    # SVI labels from frozen V on pre/post
    p_pre = predict_calibrated(ev, pack["X_pre_raw"])
    p_post = predict_calibrated(ev, pack["X_post"])
    dV = p_post - p_pre
    miss = ~(np.isfinite(p_pre) & np.isfinite(p_post))
    y = np.full(len(dV), -1, dtype=np.int8)
    y[~miss] = (dV[~miss] > 0).astype(np.int8)

    q = predict_q_frozen(logit, pack["X"], p_pre)
    tmin = pack["tmin"]
    pt = pt_flex.predict_proba(np.column_stack([p_pre, tmin]))[:, 1]

    common = (y >= 0) & np.isfinite(q) & np.isfinite(pt) & np.isfinite(p_pre)
    # also require sealed W for dual-stage alignment with RRX (optional filter)
    yW = np.asarray([Wmap.get(m, -1) for m in g_all.tolist()], dtype=float)
    common &= yW >= 0

    match = g_all[common]
    y_s = y[common].astype(np.float64)
    q_s = q[common]
    pt_s = pt[common]
    uniq, d_m = match_level_delta(match, y_s, q_s, pt_s)

    # row-weighted Brier delta (same as RRX cell_corp style via match weights)
    _, inv, c = np.unique(match.astype(str), return_inverse=True, return_counts=True)
    w = 1.0 / c[inv]
    brier_q = float(np.sum(w * (q_s - y_s) ** 2) / np.sum(w))
    brier_pt = float(np.sum(w * (pt_s - y_s) ** 2) / np.sum(w))

    return dict(
        ok=True,
        set_id=set_id,
        cohort=cohort_tag,
        n_rows=int(common.sum()),
        n_matches=int(len(uniq)),
        brier_q=brier_q,
        brier_pt=brier_pt,
        delta_brier_point=brier_q - brier_pt,
        match=match,
        y=y_s,
        q=q_s,
        pt=pt_s,
        d_m=d_m,
        matches=uniq,
        feature_check=feat_check,
        q_model=str(q_path).replace("\\", "/"),
        pt_model=str(pt_path).replace("\\", "/"),
    )


def common_match_contrast(
    t_pack: Dict[str, Any], s_pack: Dict[str, Any], n_boot: int, seed: int
) -> Dict[str, Any]:
    if not (t_pack.get("ok") and s_pack.get("ok")):
        return dict(ok=False, reason="missing_cohort_pack")
    t_map = {m: d for m, d in zip(t_pack["matches"].tolist(), t_pack["d_m"].tolist())}
    s_map = {m: d for m, d in zip(s_pack["matches"].tolist(), s_pack["d_m"].tolist())}
    common = sorted(set(t_map) & set(s_map))
    if not common:
        return dict(ok=False, reason="no_common_matches", n_common=0)
    h = np.asarray([t_map[m] - s_map[m] for m in common], dtype=np.float64)
    boot = bootstrap_mean(h, n_boot, seed)
    return dict(
        ok=True,
        estimand="mean_m(d_mT - d_mS) on matches with both T and S rows",
        n_common_matches=len(common),
        **boot,
        reading="Not a mechanism for T/S difference (X-31 withdrawn).",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/supplementary_e3_20260921")
    ap.add_argument("--docs-md", type=Path, default=REPO / "docs/SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.md")
    ap.add_argument("--docs-json", type=Path, default=REPO / "docs/SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json")
    args = ap.parse_args(argv)

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D
    import rr20260920_review_response_rr12 as rr12

    sys.modules["__main__"].PTFlexFeatures = rr12.PTFlexFeatures

    L = D.Layout(False)
    print("reference cols…", flush=True)
    ref_cols = reference_main_cols(D, L)
    ev = load_evaluator(BUNDLE)

    paths = {
        "T": dict(
            q=REPO / "outputs/q_newv_fit85_20260920/models/logit_state.joblib",
            pt=REPO / "outputs/review_response_rr12_20260920/models/PT_flex.joblib",
        ),
        "S": dict(
            q=REPO / "outputs/q_newv_fit85_20260920_S/models/logit_state.joblib",
            pt=REPO / "outputs/review_response_rr12_20260920_S_qS_id/models/PT_flex.joblib",
        ),
    }
    for tag, p in paths.items():
        for k, path in p.items():
            if not path.is_file():
                raise SystemExit(f"missing {tag} {k}: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    family: Dict[str, Any] = {}
    pilots: Dict[str, Any] = {}
    packs_for_h: Dict[str, Dict[str, Any]] = {}

    for cohort_tag in ("T", "S"):
        for set_id, coh_name, label, pilot in COHORTS:
            print(f"E3 score {cohort_tag} {label}…", flush=True)
            pack = score_cohort(
                data_root,
                D,
                L,
                cohort_tag,
                paths[cohort_tag]["q"],
                paths[cohort_tag]["pt"],
                ref_cols,
                ev,
                set_id,
                coh_name,
            )
            key = f"{cohort_tag}|{set_id}"
            if not pack.get("ok"):
                print(f"  FAIL {pack.get('reason')}", flush=True)
                (family if set_id in MAIN_SETS else pilots)[key] = dict(
                    ok=False, label=label, pilot=pilot, reason=pack.get("reason")
                )
                continue

            boot = bootstrap_mean(pack["d_m"], args.n_boot, args.seed)
            # note: D_c = mean_m d_mc ; point from match means (design), also report row-weighted
            row = dict(
                ok=True,
                label=label,
                pilot=pilot,
                cohort=cohort_tag,
                set_id=set_id,
                n_rows=pack["n_rows"],
                n_matches=pack["n_matches"],
                brier_q=pack["brier_q"],
                brier_pt=pack["brier_pt"],
                delta_brier_row_weighted=pack["delta_brier_point"],
                delta_brier_match_mean=boot["estimate"],
                bootstrap=boot,
                q_model=pack["q_model"],
                pt_model=pack["pt_model"],
            )
            np.savez_compressed(
                args.out_dir / f"rows_{cohort_tag}_{set_id}.npz",
                match=pack["match"],
                y=pack["y"],
                q=pack["q"],
                pt=pack["pt"],
                d_m=pack["d_m"],
                matches=pack["matches"],
            )
            print(
                f"  n={pack['n_rows']}/{pack['n_matches']} "
                f"Δrow={pack['delta_brier_point']:+.4f} "
                f"Δmatch={boot['estimate']:+.4f} "
                f"CI95={boot['ci95']}",
                flush=True,
            )
            if set_id in MAIN_SETS:
                family[key] = row
                packs_for_h[key] = pack
            else:
                pilots[key] = {k: v for k, v in row.items() if k != "bootstrap"}
                pilots[key]["bootstrap"] = "omitted_for_pilot"

    h_blocks = {}
    for set_id in MAIN_SETS:
        h_blocks[set_id] = common_match_contrast(
            packs_for_h.get(f"T|{set_id}", {}),
            packs_for_h.get(f"S|{set_id}", {}),
            args.n_boot,
            args.seed + 1,
        )

    doc = dict(
        schema="SUPPLEMENTARY_E3_EXT_BOOTSTRAP_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#6",
        task=".ai/tasks/T017.md",
        epistemic="SUPPLEMENTARY_SCORE_ONLY_NO_REFIT",
        n_boot=args.n_boot,
        seed=args.seed,
        primary_family=["KR_16.13|T", "KR_16.13|S", "NA1_16.13|T", "NA1_16.13|S"],
        note_ci=(
            "ci95 = 95% percentile; ci9875 = Bonferroni 0.05/4 family. "
            "Conditional on frozen models; not model-fit uncertainty. "
            "frac_boot_gt0 is not a classical p-value."
        ),
        forbidden_readings=[
            "T/S mechanism",
            "recalibration will fix transfer",
            "replace frozen RRX headlines",
        ],
        primary=family,
        pilots=pilots,
        common_match_H=h_blocks,
        git_head=_git_head(),
    )
    args.docs_json.parent.mkdir(parents=True, exist_ok=True)
    args.docs_json.write_text(json.dumps(_scrub(doc), indent=2) + "\n", encoding="utf-8")
    args.docs_md.write_text(_to_md(doc), encoding="utf-8")
    print(f"wrote {args.docs_json}", flush=True)
    print(f"wrote {args.docs_md}", flush=True)
    return 0


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _scrub(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items() if k not in ("match", "y", "q", "pt", "d_m", "matches")}
    if isinstance(o, list):
        return [_scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _to_md(doc: Dict[str, Any]) -> str:
    lines = [
        "# Supplementary E3 — External paired match-cluster bootstrap",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**n_boot / seed:** {doc['n_boot']} / {doc['seed']}  ",
        f"**epistemic:** {doc['epistemic']}  ",
        "",
        doc["note_ci"],
        "",
        "## Primary family (KR/NA1 16.13 × T/S)",
        "",
        "| Cohort | Set | n_rows | matches | ΔBrier (match-mean) | CI95 | CI98.75 | frac>0 | Δ row-wtd |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for key in doc["primary_family"]:
        row = doc["primary"].get(key, {})
        if not row.get("ok"):
            lines.append(f"| {key} | — | — | — | FAIL | {row.get('reason')} | — | — | — |")
            continue
        b = row["bootstrap"]
        lines.append(
            f"| {row['cohort']} | {row['label']} | {row['n_rows']} | {row['n_matches']} | "
            f"{b['estimate']:+.5f} | [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] | "
            f"[{b['ci9875'][0]:+.5f}, {b['ci9875'][1]:+.5f}] | {b['frac_boot_gt0']:.3f} | "
            f"{row['delta_brier_row_weighted']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Common-match H = mean(d_T − d_S) (optional; not a mechanism)",
            "",
        ]
    )
    for set_id, block in doc["common_match_H"].items():
        if not block.get("ok"):
            lines.append(f"- **{set_id}:** {block.get('reason')}")
            continue
        lines.append(
            f"- **{set_id}:** n_common={block['n_common_matches']}; "
            f"H={block['estimate']:+.5f}; CI95={block['ci95']}; {block['reading']}"
        )
    lines.extend(["", "## Forbidden", ""])
    for f in doc["forbidden_readings"]:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
