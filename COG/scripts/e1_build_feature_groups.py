#!/usr/bin/env python3
"""Build E1 feature_groups.json from runtime manifest + design §4."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "outputs/v_redesign_feature_manifest_20260919.json"
OUT_DIR = REPO / "docs/supplementary_e1_20260921"
OUT = OUT_DIR / "feature_groups.json"

F_GROUPS = {"player_snapshot"}
E_GROUPS = {"player_event", "team_event", "team_time_interaction"}
C_CLOCK = {"time_minutes", "time_minutes_sq"}
Q0 = {"unknown_objective_team_count"}


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feats = [f for f in man["features"] if f["name"] != "snapshot_age_s"]
    by_group = {}
    for f in feats:
        by_group.setdefault(f["group"], []).append(f["name"])

    clock = [n for n in by_group.get("clock", []) if n in C_CLOCK]
    q0 = [n for n in by_group.get("quality", []) if n in Q0]
    F = list(by_group.get("player_snapshot", []))
    E = []
    for g in ("player_event", "team_event", "team_time_interaction"):
        E.extend(by_group.get(g, []))

    # p_pre is appended at train time (not in engagement X)
    arms = {
        "M-F0": {"columns_from_X": clock + q0, "include_p_pre": True, "expected_dim": 4},
        "M-F1": {"columns_from_X": clock + q0 + F, "include_p_pre": True, "expected_dim": 4 + len(F)},
        "M-F2": {"columns_from_X": clock + q0 + E, "include_p_pre": True, "expected_dim": 4 + len(E)},
        "M-F3": {
            "columns_from_X": clock + q0 + F + E,
            "include_p_pre": True,
            "expected_dim": 4 + len(F) + len(E),
        },
    }
    for arm, spec in arms.items():
        assert len(spec["columns_from_X"]) + (1 if spec["include_p_pre"] else 0) == spec["expected_dim"], (
            arm,
            len(spec["columns_from_X"]),
            spec["expected_dim"],
        )

    doc = {
        "schema": "E1_FEATURE_GROUPS_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#4",
        "manifest": str(MANIFEST.relative_to(REPO)).replace("\\", "/"),
        "excluded": ["snapshot_age_s", "champion_id columns"],
        "groups": {g: by_group[g] for g in sorted(by_group)},
        "C_clock": clock,
        "Q0": q0,
        "F_player_snapshot": F,
        "E_events": E,
        "arms": arms,
        "notes": [
            "p_pre is shared across arms (conditioning on win-prob summary).",
            "M-F0 is not identical to PT_flex.",
            "Champion IDs excluded to match current q numeric policy.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for arm, spec in arms.items():
        print(f"  {arm}: dim={spec['expected_dim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
