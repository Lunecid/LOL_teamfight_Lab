#!/usr/bin/env python3
"""E2 §5.4 — frame-refresh stratification on frozen TEST rows (T/S).

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §5.4
Task: .ai/tasks/T019.md

§5.2 S_hold: recorded as INCOMPLETE unless a verified rebuild path is available
(do not mutate only time columns).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E2_FRAME_STRATA_20260921.md"
AGE_CUT_S = 30.0


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


def brier(y, p, g) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = match_weights(g)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def cohort_keys(data_root: Path, set_name: str, cohort: str) -> set:
    coh = np.load(
        data_root
        / "outputs"
        / "cohort_role_training_20260915"
        / "cohorts"
        / f"{set_name}_cohort.npz",
        allow_pickle=False,
    )
    if cohort == "T":
        m = coh["cohort"] == 1
    else:
        m = (coh["cohort"] == 0) & (coh["fine"] == 1)
    return set(
        zip(coh["match"][m].astype(str).tolist(), coh["s"][m].astype(np.int64).tolist())
    )


def load_pred_table(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return dict(
        match=z["match"].astype(str),
        s=z["s"].astype(np.int64),
        y=z["y"].astype(np.float64),
        q=z["p_q_base"].astype(np.float64),
        pt=z["p_PT_flex"].astype(np.float64),
        p_pre=z["p_pre"].astype(np.float64),
        delta_V=z["delta_V"].astype(np.float64),
    )


def cell_stats(
    mask: np.ndarray,
    match: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    pt: np.ndarray,
    dV: np.ndarray,
    p_pre: np.ndarray,
    tmin: np.ndarray,
    followup_s: np.ndarray,
) -> Dict[str, Any]:
    if not mask.any():
        return dict(n=0, n_matches=0)
    m = match[mask]
    yy = y[mask]
    qq = q[mask]
    ppt = pt[mask]
    dv = dV[mask]
    return dict(
        n=int(mask.sum()),
        n_matches=int(len(np.unique(m))),
        P_SVI=float(np.mean(yy > 0)),
        E_abs_dV=float(np.mean(np.abs(dv))),
        median_abs_dV=float(np.median(np.abs(dv))),
        brier_q=brier(yy, qq, m),
        brier_pt=brier(yy, ppt, m),
        delta_brier_q_minus_pt=brier(yy, qq, m) - brier(yy, ppt, m),
        mean_p_pre=float(np.mean(p_pre[mask])),
        mean_tmin=float(np.mean(tmin[mask])),
        mean_followup_s=float(np.mean(followup_s[mask])),
    )


def analyze_cohort(
    cohort: str,
    E: Dict[str, Any],
    keys: set,
    pred: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    em = E["match"].astype(str)
    es = E["s"].astype(np.int64) if "s" in E else E["q_pre"].astype(np.int64)
    # engagement key for cohort file uses cutoff s; prefer q_pre alignment with labels
    q_pre = E["q_pre"].astype(np.int64)
    # try both key conventions
    keep = (E["pre_ok"] == 1) & (E["valid_h90"] == 1)
    in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    if in_coh[keep].sum() == 0:
        in_coh = np.array([(a, int(b)) in keys for a, b in zip(em.tolist(), q_pre.tolist())], dtype=bool)
        key_s = q_pre
    else:
        key_s = es
    keep &= in_coh

    pre_s = E["pre_snapshot"][keep].astype(np.int64)
    post_s = E["post_snapshot_h90"][keep].astype(np.int64)
    qp = E["q_pre"][keep].astype(np.int64)
    ep = E["endpoint_h90"][keep].astype(np.int64)
    pre_age = (qp - pre_s).astype(np.float64) / 1000.0
    followup = (ep - qp).astype(np.float64) / 1000.0
    same = pre_s == post_s
    new = ~same
    young = pre_age < AGE_CUT_S
    old = ~young

    # join predictions
    pred_key = {(a, int(b)): i for i, (a, b) in enumerate(zip(pred["match"].tolist(), pred["s"].tolist()))}
    idx = []
    ok_rows = []
    for i, (a, b) in enumerate(zip(em[keep].tolist(), key_s[keep].tolist())):
        j = pred_key.get((a, int(b)))
        if j is not None:
            ok_rows.append(i)
            idx.append(j)
    ok_rows = np.asarray(ok_rows, int)
    idx = np.asarray(idx, int)
    if len(idx) == 0:
        return dict(ok=False, reason="no_pred_join", n_eng=int(keep.sum()))

    match = pred["match"][idx]
    y = pred["y"][idx]
    q = pred["q"][idx]
    pt = pred["pt"][idx]
    dV = pred["delta_V"][idx]
    p_pre = pred["p_pre"][idx]
    tmin = key_s[keep][ok_rows].astype(np.float64) / 60000.0
    followup_j = followup[ok_rows]
    same_j = same[ok_rows]
    young_j = young[ok_rows]

    cells = {
        "same_young": cell_stats(same_j & young_j, match, y, q, pt, dV, p_pre, tmin, followup_j),
        "same_old": cell_stats(same_j & (~young_j), match, y, q, pt, dV, p_pre, tmin, followup_j),
        "new_young": cell_stats((~same_j) & young_j, match, y, q, pt, dV, p_pre, tmin, followup_j),
        "new_old": cell_stats((~same_j) & (~young_j), match, y, q, pt, dV, p_pre, tmin, followup_j),
        "all": cell_stats(np.ones(len(y), dtype=bool), match, y, q, pt, dV, p_pre, tmin, followup_j),
    }
    return dict(
        ok=True,
        cohort=cohort,
        n_engagements=int(keep.sum()),
        n_joined=int(len(y)),
        join_rate=float(len(y) / max(keep.sum(), 1)),
        same_frame_rate=float(np.mean(same_j)),
        pre_age_lt30_rate=float(np.mean(young_j)),
        pre_age_median_s=float(np.median(pre_age[ok_rows])),
        followup_median_s=float(np.median(followup_j)),
        age_cut_s=AGE_CUT_S,
        cells=cells,
        reading=(
            "Cell differences are diagnostic of composition under frozen scores; "
            "not causal claims that frame refresh caused performance."
        ),
    )


def assess_s_hold() -> Dict[str, Any]:
    """§5.2: only run if a verified rebuild exists; else INCOMPLETE."""
    data_root = _data_root()
    cache = data_root / "outputs/full_corpus_training_20260915/runtime/cache"
    return dict(
        status="INCOMPLETE",
        reason=(
            "S_hold requires rebuilding StateV2 at endpoint_h90 from events/frames "
            "with timestamp ≤ q_pre (state_value_v2.StateBuilder.at), then scoring "
            "frozen V. A verified batch rebuild path over TEST was not executed in "
            "T019; mutating only time_minutes/time_minutes_sq on X_pre is forbidden "
            "by the contract. Leave probe incomplete rather than a partial clock hack."
        ),
        cache_dir_exists=cache.is_dir(),
        next_step=(
            "T020+: scripted StateBuilder hold rebuild on a fixed match subsample, "
            "assert d_clock+d_update≈ΔV, then full TEST."
        ),
    )


def main() -> int:
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    L = D.Layout(False)
    print("load TEST engagements…", flush=True)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=False)
    keys_T = cohort_keys(data_root, "MAIN_TEST", "T")
    keys_S = cohort_keys(data_root, "MAIN_TEST", "S")

    pred_T = load_pred_table(REPO / "outputs/review_response_rr12_20260920/prediction_table.npz")
    pred_S = load_pred_table(
        REPO / "outputs/review_response_rr12_20260920_S_qS_id/prediction_table.npz"
    )

    print("analyze T…", flush=True)
    block_T = analyze_cohort("T", E, keys_T, pred_T)
    print("analyze S…", flush=True)
    block_S = analyze_cohort("S", E, keys_S, pred_S)

    doc = dict(
        schema="SUPPLEMENTARY_E2_FRAME_STRATA_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#5",
        task=".ai/tasks/T019.md",
        section_5_4=dict(T=block_T, S=block_S),
        section_5_2_S_hold=assess_s_hold(),
        forbidden=[
            "causal frame-refresh effect",
            "put endpoint/frame-refresh into q inputs",
            "time-only X_pre mutation as S_hold",
            "replace journal headlines",
        ],
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    DOCS_MD.write_text(_to_md(doc), encoding="utf-8")
    print(f"wrote {DOCS_JSON}", flush=True)
    print(f"wrote {DOCS_MD}", flush=True)
    print(f"S_hold={doc['section_5_2_S_hold']['status']}", flush=True)
    return 0


def _to_md(doc: Dict[str, Any]) -> str:
    lines = [
        "# Supplementary E2 — Frame-refresh strata (§5.4) + S_hold status (§5.2)",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**task:** {doc['task']}  ",
        "",
        "## §5.2 Mechanical S_hold probe",
        "",
        f"**status:** `{doc['section_5_2_S_hold']['status']}`  ",
        "",
        doc["section_5_2_S_hold"]["reason"],
        "",
        f"Next: {doc['section_5_2_S_hold']['next_step']}",
        "",
        "## §5.4 Observation-refresh strata (TEST, frozen scores)",
        "",
        "Cells: same-frame / new-frame × pre_frame_age <30s / ≥30s. "
        "ΔBrier = q − PT_flex (match-weighted). Diagnostic only.",
        "",
    ]
    for cohort, block in doc["section_5_4"].items():
        lines.append(f"### Cohort {cohort}")
        if not block.get("ok"):
            lines.append(f"FAIL: {block.get('reason')}")
            lines.append("")
            continue
        lines.append(
            f"joined {block['n_joined']}/{block['n_engagements']} "
            f"(rate {block['join_rate']:.3f}); same-frame {block['same_frame_rate']:.3f}; "
            f"pre_age<30s {block['pre_age_lt30_rate']:.3f}; "
            f"median pre_age {block['pre_age_median_s']:.1f}s; "
            f"median followup {block['followup_median_s']:.1f}s"
        )
        lines.append("")
        lines.append(
            "| Cell | n | matches | P(SVI+) | E|ΔV| | ΔBrier(q−PT) | mean p_pre | mean t_min |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, c in block["cells"].items():
            if c.get("n", 0) == 0:
                lines.append(f"| {name} | 0 | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {name} | {c['n']} | {c['n_matches']} | {c['P_SVI']:.3f} | "
                f"{c['E_abs_dV']:.4f} | {c['delta_brier_q_minus_pt']:+.5f} | "
                f"{c['mean_p_pre']:.3f} | {c['mean_tmin']:.2f} |"
            )
        lines.append("")
        lines.append(block["reading"])
        lines.append("")
    lines.extend(["## Forbidden", ""])
    for f in doc["forbidden"]:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
