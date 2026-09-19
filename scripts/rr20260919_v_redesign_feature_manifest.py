#!/usr/bin/env python3
"""Dump runtime StateV2 feature manifest for V redesign (Step 1).

Loads MAIN TEST bucket names from the same loader path as wave-2/3 fits,
drops snapshot_age_s like expanded_X, classifies columns, and diffs against
docs/V_MODEL_INPUT_DESIGN_20260919/STATEV2_REFERENCE_FEATURE_SET.json.

Writes: outputs/v_redesign_feature_manifest_20260919.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "docs" / "V_MODEL_INPUT_DESIGN_20260919" / "STATEV2_REFERENCE_FEATURE_SET.json"


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


def classify(name: str) -> Dict[str, str]:
    if name.endswith("_champion_id") or "champion_id" in name:
        return dict(group="champion", dtype="categorical")
    if name in ("time_minutes", "time_minutes_sq"):
        return dict(group="clock", dtype="numeric")
    if name == "unknown_objective_team_count":
        return dict(group="quality", dtype="numeric")
    if name == "snapshot_age_s":
        return dict(group="audit_excluded", dtype="numeric")
    if "_x_time" in name or name.endswith("_x_time_minutes") or "x_time" in name:
        return dict(group="team_time_interaction", dtype="numeric")
    if name.startswith("participant_slot") and any(
        k in name for k in ("totalGold", "curGold", "level", "xp", "hp", "mp", "alive", "laneCS", "jgCS")
    ):
        return dict(group="player_snapshot", dtype="numeric")
    if name.startswith("participant_slot"):
        return dict(group="player_event", dtype="numeric")
    if name.startswith(("blue_", "red_")):
        return dict(group="team_event", dtype="numeric")
    return dict(group="other", dtype="numeric")


def main(argv: Optional[Sequence[str]] = None) -> int:
    data_root = _data_root()
    _setup(data_root)
    import fc20260915_data as D

    L = D.Layout(False)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    raw_names: List[str] = list(TE["names"])
    keep = [n for n in raw_names if n != "snapshot_age_s"]
    excluded = [n for n in raw_names if n == "snapshot_age_s"]

    features = []
    groups: Dict[str, int] = {}
    for i, name in enumerate(keep):
        meta = classify(name)
        groups[meta["group"]] = groups.get(meta["group"], 0) + 1
        features.append(dict(index=i, name=name, **meta))

    n_cat = sum(1 for f in features if f["dtype"] == "categorical")
    n_num = len(features) - n_cat

    ref_names = []
    ref_status = None
    if REF.is_file():
        ref = json.loads(REF.read_text(encoding="utf-8"))
        ref_status = ref.get("status")
        ref_names = [f["name"] for f in ref.get("features", [])]

    runtime_set = set(keep)
    ref_set = set(ref_names)
    only_runtime = sorted(runtime_set - ref_set)
    only_ref = sorted(ref_set - runtime_set)
    order_match = keep == ref_names if ref_names else None

    # hash worktree state_value_v2 if present
    sv2 = data_root / "worktrees" / "engagement-state-value" / "gameplay" / "state_value_v2.py"
    src_hash = None
    if sv2.is_file():
        src_hash = hashlib.sha256(sv2.read_bytes()).hexdigest()[:16]

    payload: Dict[str, Any] = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic="RUNTIME_FEATURE_MANIFEST_STEP1",
        data_root=str(data_root),
        loader="fc20260915_data.load_v_rows MAIN TEST bucket_only",
        state_value_v2_sha256_16=src_hash,
        raw_width=len(raw_names),
        expanded_width=len(keep),
        excluded=excluded,
        numeric=n_num,
        categorical=n_cat,
        groups=groups,
        features=features,
        reference_compare=dict(
            reference_path=str(REF.relative_to(REPO)).replace("\\", "/"),
            reference_status=ref_status,
            reference_n=len(ref_names),
            order_exact_match=order_match,
            only_in_runtime_n=len(only_runtime),
            only_in_reference_n=len(only_ref),
            only_in_runtime_head=only_runtime[:30],
            only_in_reference_head=only_ref[:30],
        ),
        profiles=dict(
            Expanded361=dict(numeric=n_num, categorical=n_cat, native_width=len(keep)),
            Core267_note="drop team_time_interaction group if present; verify count==94 before claiming 267",
            team_time_interaction_count=groups.get("team_time_interaction", 0),
        ),
    )

    out = REPO / "outputs" / "v_redesign_feature_manifest_20260919.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # also write a small markdown summary into docs (committed)
    md = REPO / "docs" / "V_FEATURE_MANIFEST_RUNTIME_20260919.md"
    lines = [
        "# Runtime StateV2 feature manifest (Step 1)",
        "",
        f"Generated: {payload['generated']}",
        f"Source: `{out.as_posix()}` (local outputs; may be gitignored)",
        "",
        f"- Raw width: **{payload['raw_width']}**",
        f"- Expanded (drop `snapshot_age_s`): **{payload['expanded_width']}** "
        f"(numeric {n_num} + categorical {n_cat})",
        f"- Groups: `{groups}`",
        f"- `state_value_v2.py` hash16: `{src_hash}`",
        "",
        "## vs reference reconstruction",
        "",
        f"- Reference status: `{ref_status}`",
        f"- Order exact match: **{order_match}**",
        f"- Only in runtime: {len(only_runtime)} (head: {only_runtime[:10]})",
        f"- Only in reference: {len(only_ref)} (head: {only_ref[:10]})",
        "",
        "## Next",
        "",
        "- If order/names differ, treat reference JSON as **proposal** until reconciled.",
        "- Core267 ablation: drop `team_time_interaction` columns "
        f"(count={groups.get('team_time_interaction', 0)}).",
        "- Proceed to typed adapters (CAT-1) and history builder (HIST-1) per "
        "[V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md).",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)
    print("wrote", md)
    print("expanded", len(keep), "groups", groups, "order_match", order_match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
