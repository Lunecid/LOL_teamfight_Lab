#!/usr/bin/env python3
"""G0 Phase B verification against frozen repo outputs (no refit).

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §3
Task: .ai/tasks/T016.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RR0_PATH = REPO / "docs" / "REVIEW_RESPONSE_RR0_MANIFEST_20260920.json"
PHASE_A_JSON = REPO / "docs" / "G0_INTEGRITY_STATUS_20260921.json"
OUT_JSON = REPO / "docs" / "G0_INTEGRITY_STATUS_20260921.json"
OUT_MD = REPO / "docs" / "G0_INTEGRITY_STATUS_20260921.md"
PHASE_B_JSON = REPO / "docs" / "G0_PHASE_B_VERIFY_20260921.json"

DIGEST_PATHS = {
    "rr12_prediction_table": "outputs/review_response_rr12_20260920/prediction_table.npz",
    "rr12_paired_ci": "outputs/review_response_rr12_20260920/paired_ci.json",
    "rr12_baseline_selection": "outputs/review_response_rr12_20260920/baseline_selection.json",
    "rr12_PT_flex": "outputs/review_response_rr12_20260920/models/PT_flex.joblib",
    "rr12_PT_linear": "outputs/review_response_rr12_20260920/models/PT_linear.joblib",
    "rr12_b_spline": "outputs/review_response_rr12_20260920/models/b_spline.joblib",
    "logit_state": "outputs/q_newv_fit85_20260920/models/logit_state.joblib",
    "primary_table": "outputs/q_newv_fit85_20260920/primary_table.json",
    "selection_freeze": "outputs/q_newv_fit85_20260920/selection_freeze.json",
    "TRAIN_oof_STATUS": "outputs/q_newv_fit85_20260920/labels/TRAIN_oof_STATUS.json",
    "TRAIN_oof_h90_meta": "outputs/q_newv_fit85_20260920/labels/TRAIN_oof_h90_meta.json",
    "TEST_h90": "outputs/q_newv_fit85_20260920/labels/TEST_h90.npz",
}

DOCS_DELTA_T = -0.00373
DOCS_DELTA_S = -0.00335
ATOL_HEADLINE = 5e-6  # point estimate print tolerance


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def match_weighted_brier(y: np.ndarray, p: np.ndarray, match: np.ndarray) -> float:
    _, inv, c = np.unique(match.astype(str), return_inverse=True, return_counts=True)
    w = (1.0 / c[inv]).astype(np.float64)
    return float(np.sum(w * (p.astype(np.float64) - y.astype(np.float64)) ** 2) / np.sum(w))


def reaggregate(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    match = z["match"]
    y = z["y"]
    q = z["p_q_base"]
    pt = z["p_PT_flex"]
    bq = match_weighted_brier(y, q, match)
    bpt = match_weighted_brier(y, pt, match)
    delta = bq - bpt
    n_matches = int(len(np.unique(match.astype(str))))
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "n_rows": int(len(y)),
        "n_matches": n_matches,
        "brier_q": bq,
        "brier_PT_flex": bpt,
        "delta_brier_q_minus_PT_flex": delta,
    }


def verify_digests() -> Dict[str, Any]:
    rr0 = json.loads(RR0_PATH.read_text(encoding="utf-8"))
    digests = rr0["digests"]
    rows: List[Dict[str, Any]] = []
    n_match = n_diff = n_missing_file = n_missing_key = 0
    for name, rel in DIGEST_PATHS.items():
        path = REPO / rel
        if name not in digests:
            n_missing_key += 1
            rows.append({"name": name, "status": "MISSING_KEY", "path": rel})
            continue
        if not path.is_file():
            n_missing_file += 1
            rows.append({"name": name, "status": "MISSING_FILE", "path": rel})
            continue
        exp = digests[name]["sha256"].lower()
        got = sha256_file(path)
        ok = got == exp
        if ok:
            n_match += 1
        else:
            n_diff += 1
        rows.append(
            {
                "name": name,
                "status": "MATCH" if ok else "DIFF",
                "path": rel,
                "expected_sha16": exp[:16],
                "got_sha16": got[:16],
                "bytes": path.stat().st_size,
            }
        )
    status = "PASS" if n_diff == 0 and n_missing_file == 0 and n_missing_key == 0 else (
        "FAIL" if n_diff else "PARTIAL"
    )
    return {
        "status": status,
        "n_match": n_match,
        "n_diff": n_diff,
        "n_missing_file": n_missing_file,
        "n_missing_key": n_missing_key,
        "rows": rows,
    }


def verify_oof_reload() -> Dict[str, Any]:
    import joblib

    oof_dir = REPO / "outputs/q_newv_fit85_20260920/oof_evaluators"
    folds: List[Dict[str, Any]] = []
    all_ok = True
    for k in range(5):
        path = oof_dir / f"V_oof_fold{k}_mlp_expanded.joblib"
        bun = oof_dir / f"bundle_oof_fold{k}.joblib"
        if not path.is_file():
            all_ok = False
            folds.append({"fold": k, "status": "MISSING_FILE"})
            continue
        obj = joblib.load(path)
        required = {"kind", "fold", "mlp", "calib", "n_eng_labeled"}
        keys_ok = isinstance(obj, dict) and required.issubset(obj.keys())
        n_eng = int(obj.get("n_eng_labeled", -1)) if isinstance(obj, dict) else -1
        bun_ok = bun.is_file()
        if bun_ok:
            _ = joblib.load(bun)
        row_ok = keys_ok and n_eng >= 0 and bun_ok
        all_ok = all_ok and row_ok
        folds.append(
            {
                "fold": k,
                "status": "PASS" if row_ok else "FAIL",
                "path": str(path.relative_to(REPO)).replace("\\", "/"),
                "sha16": hashlib.sha256(path.read_bytes()).hexdigest().upper()[:16],
                "n_eng_labeled": n_eng,
                "keys_ok": keys_ok,
                "bundle_present": bun_ok,
            }
        )
    return {"status": "PASS" if all_ok else "FAIL", "folds": folds}


def verify_headlines() -> Dict[str, Any]:
    t_path = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
    s_path = REPO / "outputs/review_response_rr12_20260920_S_qS_id/prediction_table.npz"
    out: Dict[str, Any] = {}
    for tag, path, expect, n_exp, m_exp in (
        ("T", t_path, DOCS_DELTA_T, 32981, 24020),
        ("S_identity", s_path, DOCS_DELTA_S, 101205, 49730),
    ):
        if not path.is_file():
            out[tag] = {"status": "MISSING_FILE", "path": str(path)}
            continue
        agg = reaggregate(path)
        close = abs(agg["delta_brier_q_minus_PT_flex"] - expect) <= ATOL_HEADLINE
        census_ok = agg["n_rows"] == n_exp and agg["n_matches"] == m_exp
        out[tag] = {
            **agg,
            "docs_delta_printed": expect,
            "abs_err_vs_docs_print": abs(agg["delta_brier_q_minus_PT_flex"] - expect),
            "census_ok": census_ok,
            "status": "PASS" if close and census_ok else "FAIL",
        }
    statuses = [v.get("status") for v in out.values()]
    overall = "PASS" if statuses and all(s == "PASS" for s in statuses) else (
        "FAIL" if "FAIL" in statuses else "PARTIAL"
    )
    return {"status": overall, "cohorts": out}


def merge_phase_a(phase_b: Dict[str, Any]) -> Dict[str, Any]:
    if PHASE_A_JSON.is_file():
        base = json.loads(PHASE_A_JSON.read_text(encoding="utf-8"))
    else:
        base = {"schema": "G0_INTEGRITY_STATUS_v1", "checks": {}}
    checks = dict(base.get("checks", {}))

    dig = phase_b["digests"]
    checks["G0.2_evaluator_hash_after_finalize"] = {
        "status": "PASS" if phase_b["oof_reload"]["status"] == "PASS" else "PARTIAL",
        "reason": "Phase B: on-disk OOF evaluator SHA16 recorded after load; finalize unit tests remain from Phase A.",
        "phase_b_oof": phase_b["oof_reload"],
    }
    # G0.6: structural reload of frozen OOF/bundles (not full predict parity)
    checks["G0.6_bundle_reload_parity"] = {
        "status": "PARTIAL" if phase_b["oof_reload"]["status"] == "PASS" else "FAIL",
        "reason": (
            "Loaded all five OOF evaluator+bundle joblibs and verified required keys / "
            "n_eng_labeled; full score-vs-saved prediction parity not run (needs engagement X)."
        ),
        "detail": phase_b["oof_reload"],
    }
    checks["G0.7_headline_reaggregate"] = {
        "status": phase_b["headlines"]["status"],
        "reason": "Recomputed match-weighted ΔBrier from frozen prediction_table.npz; compared to docs print.",
        "detail": phase_b["headlines"],
        "rr0_digests": dig,
    }
    # G0.5 still incomplete without fold match-set hash store
    checks.setdefault(
        "G0.5_fold_holdout_match_set",
        {
            "status": "INCOMPLETE",
            "reason": "No stored held-match set hashes in meta to compare in Phase B.",
        },
    )
    checks.setdefault(
        "G0.F_future_info_invariance",
        {
            "status": "INCOMPLETE",
            "reason": "Requires feature rebuild with post-query events stripped.",
        },
    )

    g_statuses = [c["status"] for k, c in checks.items() if k.startswith("G0.")]
    if "FAIL" in g_statuses or dig["status"] == "FAIL":
        overall = "FAIL"
    elif any(s == "INCOMPLETE" for s in g_statuses) or any(s == "PARTIAL" for s in g_statuses):
        overall = "PARTIAL_PHASE_B"
    else:
        overall = "COMPLETE_G0"

    base.update(
        {
            "schema": "G0_INTEGRITY_STATUS_v1",
            "phase": "A+B",
            "phase_b_task": ".ai/tasks/T016.md",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall,
            "checks": checks,
            "phase_b": {
                "digests": dig,
                "headlines": phase_b["headlines"],
                "oof_reload": phase_b["oof_reload"],
            },
        }
    )
    return base


def to_md(report: Dict[str, Any]) -> str:
    lines = [
        "# G0 Integrity Status — Phase A+B (2026-09-21)",
        "",
        f"**overall:** `{report['overall_status']}`  ",
        f"**generated_at_utc:** {report['generated_at_utc']}  ",
        f"**tasks:** T015 (Phase A) · T016 (Phase B)  ",
        "",
        "| ID | Status | Reason |",
        "|---|---|---|",
    ]
    for key, block in report["checks"].items():
        if not (key.startswith("G0.") or key == "unit_tests_provenance"):
            continue
        reason = str(block.get("reason", "")).replace("|", "/")
        lines.append(f"| `{key}` | **{block['status']}** | {reason} |")

    pb = report.get("phase_b", {})
    dig = pb.get("digests", {})
    lines.extend(["", "## Phase B digests (RR0)", "", f"status: **{dig.get('status')}**", ""])
    for row in dig.get("rows", []):
        lines.append(
            f"- `{row['name']}`: **{row['status']}** (`{row.get('path','')}`)"
        )

    hl = pb.get("headlines", {}).get("cohorts", {})
    lines.extend(["", "## Headline re-aggregation", ""])
    for tag, block in hl.items():
        if block.get("status") == "MISSING_FILE":
            lines.append(f"- **{tag}**: MISSING")
            continue
        lines.append(
            f"- **{tag}**: ΔBrier={block['delta_brier_q_minus_PT_flex']:.8f} "
            f"(docs print {block['docs_delta_printed']}); "
            f"n={block['n_rows']}/{block['n_matches']} matches; **{block['status']}**"
        )

    lines.extend(["", "## OOF evaluator reload", ""])
    for fold in pb.get("oof_reload", {}).get("folds", []):
        lines.append(
            f"- fold{fold.get('fold')}: {fold.get('status')} "
            f"sha16=`{fold.get('sha16','')}` n_eng_labeled={fold.get('n_eng_labeled')}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    phase_b = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "digests": verify_digests(),
        "headlines": verify_headlines(),
        "oof_reload": verify_oof_reload(),
    }
    PHASE_B_JSON.write_text(json.dumps(phase_b, indent=2) + "\n", encoding="utf-8")
    merged = merge_phase_a(phase_b)
    OUT_JSON.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(to_md(merged), encoding="utf-8")
    print(f"wrote {PHASE_B_JSON}")
    print(f"wrote {OUT_JSON}")
    print(f"overall_status={merged['overall_status']}")
    print(f"digests={phase_b['digests']['status']} headlines={phase_b['headlines']['status']} oof={phase_b['oof_reload']['status']}")
    if phase_b["digests"]["status"] == "FAIL" or phase_b["headlines"]["status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
