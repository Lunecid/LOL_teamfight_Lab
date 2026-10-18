#!/usr/bin/env python3
"""G0 Phase A integrity audit (repo-only; does not open outputs/ NPZ/joblib).

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §3
Task: .ai/tasks/T015.md

Writes a status JSON/MD with PASS / PARTIAL / INCOMPLETE per G0 item.
Never publishes FAIL data as a normal artifact: overall status is
COMPLETE_PHASE_A only when required Phase A checks pass; otherwise PARTIAL.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]

SCRIPTS_OF_INTEREST = [
    "scripts/rr20260920_q_train_oof_mlp_folds.py",
    "scripts/rr20260920_review_response_rr12.py",
    "scripts/rr20260920_q_build_newv_labels.py",
    "scripts/rr20260920_q_newv_primary_fit.py",
    "scripts/g0_integrity_audit.py",
]

WORKTREE_MARKERS = (
    "worktrees",
    "engagement-state-value",
    "sys.path.insert",
)


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            )
            .strip()
        )
    except Exception as e:  # noqa: BLE001
        return f"UNKNOWN:{e}"


def _file_sha16(rel: str) -> str:
    path = REPO / rel
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()[:16]


def _package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "sklearn", "lightgbm", "joblib", "torch"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "not_importable"
    return out


def _ast_has_names(path: Path, names: List[str]) -> Dict[str, bool]:
    found = {n: False for n in names}
    if not path.is_file():
        return found
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in found:
            found[node.name] = True
    return found


def _scan_worktree_inserts(rel: str) -> List[str]:
    path = REPO / rel
    hits: List[str] = []
    if not path.is_file():
        return [f"MISSING:{rel}"]
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "sys.path.insert" in line and (
            "worktree" in line.lower() or "data_root" in line or "wt" in line
        ):
            hits.append(f"{rel}:{i}:{line.strip()}")
        elif "worktrees" in line and "engagement-state-value" in line:
            hits.append(f"{rel}:{i}:{line.strip()}")
    return hits


def _run_pytest() -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_q_provenance_fixes.py",
        "-q",
        "--tb=no",
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-5:]),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-5:]),
        "PASS": proc.returncode == 0,
    }


def build_report() -> Dict[str, Any]:
    oof = REPO / "scripts/rr20260920_q_train_oof_mlp_folds.py"
    rr12 = REPO / "scripts/rr20260920_review_response_rr12.py"
    oof_fns = _ast_has_names(
        oof,
        [
            "validate_cohort_reuse",
            "assert_oof_evaluator_write_allowed",
            "finalize_fold_evaluator",
            "integrity_report",
            "resolve_data_root",
        ],
    )
    rr12_fns = _ast_has_names(rr12, ["join_label_engagement_indices", "resolve_data_root"])

    worktree_hits: List[str] = []
    for rel in SCRIPTS_OF_INTEREST:
        worktree_hits.extend(_scan_worktree_inserts(rel))

    pytest_result = _run_pytest()

    script_hashes = {rel: _file_sha16(rel) for rel in SCRIPTS_OF_INTEREST}

    rr0 = REPO / "docs/REVIEW_RESPONSE_RR0_MANIFEST_20260920.json"
    scale_rr0 = REPO / "docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json"
    docs_manifests = {
        "REVIEW_RESPONSE_RR0_MANIFEST": rr0.is_file(),
        "SCALE_SPLIT_RR0_MANIFEST": scale_rr0.is_file(),
    }

    checks: Dict[str, Any] = {}

    checks["G0.1_provenance_record"] = {
        "status": "PARTIAL",
        "reason": (
            "git HEAD, script sha16, and package versions recorded in this audit; "
            "runtime imported worktree modules are not hashed (scripts still "
            "sys.path-insert engagement-state-value — see worktree_inserts)."
        ),
        "git_head": _git_head(),
        "script_sha16": script_hashes,
        "packages": _package_versions(),
        "worktree_inserts": worktree_hits,
        "resolve_data_root_present": bool(
            oof_fns.get("resolve_data_root") and rr12_fns.get("resolve_data_root")
        ),
    }

    checks["G0.2_evaluator_hash_after_finalize"] = {
        "status": "PASS" if oof_fns.get("finalize_fold_evaluator") and pytest_result["PASS"] else "PARTIAL",
        "reason": "finalize_fold_evaluator unit-tested; on-disk frozen evaluator re-hash is Phase B.",
        "function_present": oof_fns.get("finalize_fold_evaluator"),
    }

    write_ok = oof_fns.get("assert_oof_evaluator_write_allowed") and oof_fns.get(
        "validate_cohort_reuse"
    )
    checks["G0.3_s_reuse_and_write_guard"] = {
        "status": "PASS" if write_ok and pytest_result["PASS"] else "PARTIAL",
        "reason": "S requires --reuse-evaluators; OOF dumps guarded for S/reuse.",
        "functions": {
            "validate_cohort_reuse": oof_fns.get("validate_cohort_reuse"),
            "assert_oof_evaluator_write_allowed": oof_fns.get(
                "assert_oof_evaluator_write_allowed"
            ),
        },
    }

    checks["G0.4_join_key_integrity"] = {
        "status": "PASS" if rr12_fns.get("join_label_engagement_indices") and pytest_result["PASS"] else "PARTIAL",
        "reason": "join_label_engagement_indices fails on missing/duplicate keys; full-corpus join is Phase B.",
        "function_present": rr12_fns.get("join_label_engagement_indices"),
    }

    checks["G0.5_fold_holdout_match_set"] = {
        "status": "INCOMPLETE",
        "reason": (
            "reuse mode loads V-bucket fold match sets in code, but verifying "
            "held-out match set hashes against stored OOF fits requires outputs/ "
            "(AGENTS.md: do not open in Phase A)."
        ),
    }

    checks["G0.6_bundle_reload_parity"] = {
        "status": "INCOMPLETE",
        "reason": "Requires loading frozen joblib bundles from outputs/.",
    }

    checks["G0.7_headline_reaggregate"] = {
        "status": "INCOMPLETE",
        "reason": (
            "Requires frozen weights + prediction tables under outputs/; "
            "docs RR0 manifests present for later digest compare only."
        ),
        "docs_manifests_present": docs_manifests,
    }

    checks["G0.8_missing_data_policy"] = {
        "status": "PASS",
        "reason": (
            "This audit marks data-dependent checks INCOMPLETE rather than "
            "emitting a fake PASS; FAIL artifacts are not written as normal results."
        ),
        "phase_a_opens_outputs": False,
    }

    checks["G0.F_future_info_invariance"] = {
        "status": "INCOMPLETE",
        "reason": "Needs fixed query metadata + feature rebuild with post-query events stripped (Phase B).",
    }

    checks["unit_tests_provenance"] = {
        "status": "PASS" if pytest_result["PASS"] else "FAIL",
        "detail": pytest_result,
    }

    statuses = [c["status"] for k, c in checks.items() if k.startswith("G0.")]
    if "FAIL" in statuses or checks["unit_tests_provenance"]["status"] == "FAIL":
        overall = "FAIL"
    elif all(s in ("PASS", "PARTIAL", "INCOMPLETE") for s in statuses) and any(
        s == "INCOMPLETE" for s in statuses
    ):
        overall = "PARTIAL_PHASE_A"
    else:
        overall = "COMPLETE_PHASE_A"

    return {
        "schema": "G0_INTEGRITY_STATUS_v1",
        "phase": "A",
        "contract": "docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#3",
        "task": ".ai/tasks/T015.md",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": checks,
        "phase_b_required_for": [
            "G0.5_fold_holdout_match_set",
            "G0.6_bundle_reload_parity",
            "G0.7_headline_reaggregate",
            "G0.F_future_info_invariance",
            "G0.1 runtime imported-module byte hashes",
            "G0.2 on-disk frozen evaluator re-hash",
        ],
    }


def _to_md(report: Dict[str, Any]) -> str:
    lines = [
        "# G0 Integrity Status — Phase A (2026-09-21)",
        "",
        f"**overall:** `{report['overall_status']}`  ",
        f"**generated_at_utc:** {report['generated_at_utc']}  ",
        f"**contract:** {report['contract']}  ",
        f"**task:** {report['task']}  ",
        "",
        "Phase A does **not** open `outputs/` or large NPZ/joblib (`AGENTS.md`). "
        "INCOMPLETE items require Phase B with an explicit data-root work order.",
        "",
        "| ID | Status | Reason |",
        "|---|---|---|",
    ]
    for key, block in report["checks"].items():
        if not key.startswith("G0.") and key != "unit_tests_provenance":
            continue
        reason = str(block.get("reason") or block.get("detail", "")).replace("|", "/")
        if isinstance(block.get("detail"), dict):
            reason = f"pytest rc={block['detail'].get('returncode')}; {block['detail'].get('stdout_tail', '')}"
        lines.append(f"| `{key}` | **{block['status']}** | {reason} |")
    lines.extend(
        [
            "",
            "## Phase B required for",
            "",
        ]
    )
    for item in report["phase_b_required_for"]:
        lines.append(f"- {item}")
    g01 = report["checks"]["G0.1_provenance_record"]
    lines.extend(
        [
            "",
            "## Provenance snapshot",
            "",
            f"- git_head: `{g01['git_head']}`",
            f"- packages: `{json.dumps(g01['packages'], ensure_ascii=False)}`",
            "",
            "### Script sha16",
            "",
        ]
    )
    for rel, h in g01["script_sha16"].items():
        lines.append(f"- `{rel}`: `{h}`")
    if g01["worktree_inserts"]:
        lines.extend(["", "### Worktree / path-insert hits (known)", ""])
        for hit in g01["worktree_inserts"]:
            lines.append(f"- `{hit}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "docs/G0_INTEGRITY_STATUS_20260921.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=REPO / "docs/G0_INTEGRITY_STATUS_20260921.md",
    )
    args = ap.parse_args()
    report = build_report()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_to_md(report), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"overall_status={report['overall_status']}")
    return 0 if report["overall_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
