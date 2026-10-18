#!/usr/bin/env python3
"""E6 Phase A — lock never-seen holdout protocol + Riot API probe (no scoring).

Must run BEFORE any E6 score-only results are computed.
Task: .ai/tasks/T024.md
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK_JSON = REPO / "docs/SUPPLEMENTARY_E6_LOCK_20260921.json"
LOCK_MD = REPO / "docs/SUPPLEMENTARY_E6_LOCK_20260921.md"
KEY_FILE = Path.home() / ".secrets" / "riot.env"
OUT_ROOT = Path(r"D:/LOL_Project/data/raw/e6_holdout_20260921")

# Patches already exposed in MAIN/EXT analyses — NOT eligible for confirmatory E6.
EXPOSED_API_PATCHES = ("15.14", "15.15", "15.16", "16.13", "16.14", "16.15")
MIN_API_PATCH = "16.16"  # strictly after last EXT patch used in RRX
TARGET_N_MATCHES = 500  # budget pilot; not a power guarantee
PLATFORM = "kr"
TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER"]
SEED = 20260921


def lock_payload(probe: dict) -> dict:
    rules = dict(
        epistemic_if_collected="CONFIRMATORY_CANDIDATE_SCORE_ONLY_NEVER_SEEN_PATCH",
        epistemic_forbidden=[
            "re-split MAIN 15.16 after TEST exposure",
            "re-split EXT 16.13/16.14/16.15 after RRX exposure",
            "score then choose which matches to keep",
            "stop collecting based on interim ΔBrier",
        ],
        collection=dict(
            platform=PLATFORM,
            tiers=TIERS,
            min_api_patch=MIN_API_PATCH,
            excluded_api_patches=list(EXPOSED_API_PATCHES),
            output_root=str(OUT_ROOT),
            key_file=str(KEY_FILE),
            target_n_complete_matches=TARGET_N_MATCHES,
            seed=SEED,
            ranking_rule=(
                "After collection, retain matches with api_patch >= min_api_patch, "
                "exclude EXPOSED_API_PATCHES, then keep first TARGET_N by "
                "sha256(match_id|seed) ascending. Do not peek at scores before lock finalize."
            ),
        ),
        scoring=dict(
            mode="score_only",
            V="frozen fit85 A_MLP_expanded ac459cc4397630a9",
            q="frozen logit_state from q_newv_fit85_20260920",
            PT="frozen PT_flex from review_response_rr12_20260920",
            no_refit=True,
            primary_contrast="match-weighted ΔBrier(q − PT_flex) with match-cluster bootstrap",
        ),
        sample_size_note=(
            f"TARGET_N={TARGET_N_MATCHES} is a compute budget for a confirmatory-candidate pilot; "
            "see design §9 for n≈(z s_d / h)^2 planning once s_d is measured on a pilot slice."
        ),
        prior_audit=dict(
            EXT_CSV_remainder_not_usable=(
                "Unused IDs in EXT CSVs have packs but detect_fights→0; cannot form T/S engagements."
            ),
            MAIN_cache_patches_only=("15.14", "15.15", "16.16"),  # filled below if known
        ),
    )
    blob = json.dumps(rules, sort_keys=True, ensure_ascii=False).encode()
    return dict(
        schema="SUPPLEMENTARY_E6_LOCK_v1",
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
        task=".ai/tasks/T024.md",
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#9",
        rules_sha256=hashlib.sha256(blob).hexdigest(),
        rules=rules,
        probe=probe,
        status="LOCKED_AWAITING_COLLECTION",
    )


def probe_api() -> dict:
    if not KEY_FILE.is_file():
        return dict(ok=False, reason=f"missing key file {KEY_FILE}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "collect_current_season.py"),
        "--key-file",
        str(KEY_FILE),
        "--output-root",
        str(OUT_ROOT),
        "--platform",
        PLATFORM,
        "--probe-only",
        "--log-level",
        "WARNING",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return dict(ok=False, reason=str(e))
    return dict(
        ok=(r.returncode == 0),
        returncode=r.returncode,
        stdout_tail=(r.stdout or "")[-500:],
        stderr_tail=(r.stderr or "")[-500:],
    )


def main() -> int:
    print("probe Riot API…", flush=True)
    probe = probe_api()
    print(f"probe ok={probe.get('ok')} rc={probe.get('returncode')}", flush=True)
    doc = lock_payload(probe)
    # correct MAIN cache note
    doc["rules"]["prior_audit"]["MAIN_cache_patches_only"] = ("15.14", "15.15", "15.16")
    # recompute sha after fix
    blob = json.dumps(doc["rules"], sort_keys=True, ensure_ascii=False).encode()
    doc["rules_sha256"] = hashlib.sha256(blob).hexdigest()

    doc["status"] = "LOCKED_AWAITING_COLLECTION" if probe.get("ok") else "LOCKED_AWAITING_KEY"
    if not probe.get("ok"):
        doc["blocked"] = dict(
            reason="Riot API probe failed (see probe.stderr_tail). Refresh RIOT_API_KEY then re-run.",
            key_file=str(KEY_FILE),
            next_step=(
                f"python scripts/collect_current_season.py --key-file {KEY_FILE} "
                f"--output-root {OUT_ROOT} --platform {PLATFORM} "
                f"--min-api-patch {MIN_API_PATCH} --max-complete-matches {TARGET_N_MATCHES} --once"
            ),
        )
    LOCK_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Supplementary E6 — Holdout lock (pre-score)",
        "",
        f"**locked_at:** {doc['locked_at_utc']}  ",
        f"**rules_sha256:** `{doc['rules_sha256']}`  ",
        f"**status:** `{doc['status']}`  ",
        f"**API probe:** `{'OK' if probe.get('ok') else 'FAIL'}`  ",
        "",
    ]
    if not probe.get("ok"):
        lines += [
            "## Blocked",
            "",
            doc["blocked"]["reason"],
            "",
            f"Next: refresh key → `{doc['blocked']['next_step']}`",
            "",
        ]
    lines += [
        "## Eligibility",
        "",
        f"- Collect `{PLATFORM}` Master+ matches with `api_patch >= {MIN_API_PATCH}`.",
        f"- Exclude already-exposed patches: `{', '.join(EXPOSED_API_PATCHES)}`.",
        f"- Target n={TARGET_N_MATCHES} after sha256(match_id|{SEED}) ranking.",
        "",
        "## Scoring (only after collection lock finalize)",
        "",
        "Frozen V/q/PT score-only; no refit; match-cluster ΔBrier.",
        "",
        "## Not confirmatory if",
        "",
        "- Re-using MAIN/EXT remainder after seeing those patches' results.",
        "- Choosing matches after peeking at scores.",
        "",
        f"Output root: `{OUT_ROOT}`",
        "",
    ]
    LOCK_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {LOCK_JSON}", flush=True)
    print(f"wrote {LOCK_MD}", flush=True)
    return 0 if probe.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
