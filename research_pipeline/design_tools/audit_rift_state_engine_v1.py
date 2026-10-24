import os
"""Read source declarations and prior audit summaries; probe pure kernels on synthetic data.

No match cache reads, training, scoring, label generation or source edits.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ["LOL_RESEARCH_ROOT"])
OBS = ROOT / "studies/observable_state_20261002"
LEGACY = ROOT / "worktrees/engagement-state-value"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def declarations(path, names):
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    return result


def main():
    source_paths = [
        OBS / "schema.py", OBS / "kernel.py", OBS / "population.py",
        OBS / "tests/test_population.py",
        ROOT / "reports/input_population_specification_20261002/INPUT_POLICY.json",
        LEGACY / "gameplay/event_survival.py", LEGACY / "gameplay/respawn_rules.py",
        LEGACY / "gameplay/objective_timers.py", LEGACY / "gameplay/evidence_state.py",
        LEGACY / "config/game_rules/objective_rules.json",
        LEGACY / "config/game_rules/map_anchors.json",
        OUT / "objective_audit/audit_results.json",
        OUT / "objective_audit/semantic_checks.json",
    ]
    audit = json.loads((OUT / "objective_audit/audit_results.json").read_text(encoding="utf-8"))
    summary = {}
    for name in ["LEVEL_UP", "CHAMPION_KILL", "BUILDING_KILL", "WARD_PLACED", "WARD_KILL", "CHAMPION_TRANSFORM"]:
        count, fields = 0, {}
        for group in audit["groups"].values():
            count += group["events"].get(name, 0)
            for key, value in group["field_presence"].get(name, {}).items():
                fields[key] = fields.get(key, 0) + value
        summary[name] = {"count": count, "field_presence": fields}
    kernel = load("rift_audit_kernel", OBS / "kernel.py")
    population = load("rift_audit_population", OBS / "population.py")
    teams = {str(p): 100 if p <= 5 else 200 for p in range(1, 11)}
    arrays = {"minute_ts": np.array([0, 120000, 200000]),
              "node_frame": np.zeros((3, 10, 46), dtype=np.float32),
              "xy": np.ones((3, 10, 2), dtype=np.float32),
              "gold_team": np.zeros((3, 2), dtype=np.float32)}
    events = [
        {"type": "LEVEL_UP", "timestamp": 120001, "participantId": 1, "level": 11},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 120002, "killerId": 1,
         "killerTeamId": 100, "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"},
        {"type": "CHAMPION_KILL", "timestamp": 120003, "killerId": 1,
         "victimId": 6, "assistingParticipantIds": [], "position": {"x": 5000, "y": 5000}},
        {"type": "WARD_PLACED", "timestamp": 120004, "creatorId": 1, "wardType": "YELLOW_TRINKET"},
        {"type": "CHAMPION_KILL", "timestamp": 120005, "killerId": 2,
         "victimId": 7, "assistingParticipantIds": [], "position": {"x": 5000, "y": 5000}},
    ]
    meta = {"team_map": teams, "static_meta": {}}
    match = kernel.ObservableMatch(arrays, meta, events, "15.14")
    prefix = kernel.ObservableMatch(arrays, meta, [e for e in events if e["timestamp"] <= 120004], "15.14")
    q, p = match.query_numeric(120004), prefix.query_numeric(120004)
    checks = {}
    checks["synthetic_prefix_kda_invariant"] = bool(np.array_equal(q["kda"], p["kda"]))
    checks["synthetic_inclusive_same_ms_cutoff"] = len(list(match.iter_typed_events(120004))) == 4
    checks["synthetic_future_event_excluded"] = len(list(match.iter_typed_events(120003))) == 3
    bank_roundtrip = kernel.ObservableMatch.from_bank(match.export_bank(), patch="15.14")
    dragon = next(e for e in bank_roundtrip.iter_typed_events() if e.event_type == "ELITE_MONSTER_KILL")
    checks["synthetic_unknown_enum_raw_subtype_preserved"] = dragon.payload["monsterSubType"] == "FIRE_DRAGON"
    ward = next(e for e in match.iter_typed_events() if e.event_type == "WARD_PLACED")
    checks["synthetic_typed_ward_creator_and_type_not_preserved"] = "creatorId" not in ward.payload and "wardType" not in ward.payload
    population_ast = ast.parse((OBS / "population.py").read_text(encoding="utf-8"))
    lag_expression = next(node.value for node in ast.walk(population_ast)
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "frame_lag_candidate" for t in node.targets))
    lag_code = compile(ast.Expression(body=lag_expression), "population_frame_lag_expression", "eval")
    timing = []
    for first in [130000, 140000, 140001]:
        pre = first - population.ANCHOR_OFFSET_MS
        frame = int(arrays["minute_ts"][population._frame_index(arrays["minute_ts"], pre)])
        candidate = eval(lag_code, {"__builtins__": {}, "bool": bool},
                         {"lag": first-frame, "FRAME_LAG_MIN_MS": population.FRAME_LAG_MIN_MS,
                          "FRAME_LAG_MAX_MS": population.FRAME_LAG_MAX_MS})
        timing.append({"firstkill_ms": first, "pre_ms": pre, "frame_ms": frame,
                       "frame_age_ms": pre-frame, "timing_candidate": bool(candidate)})
    checks["synthetic_age_zero_and_ten_seconds_included"] = timing[0]["timing_candidate"] and timing[1]["timing_candidate"]
    checks["synthetic_age_ten_seconds_plus_one_ms_excluded"] = not timing[2]["timing_candidate"]
    rules = json.loads((LEGACY / "config/game_rules/objective_rules.json").read_text(encoding="utf-8"))
    geometry = json.loads((LEGACY / "config/game_rules/map_anchors.json").read_text(encoding="utf-8"))
    report = {
        "scope": "source declarations, prior 640-match audit summary, synthetic kernel probes only",
        "new_raw_matches_scanned": 0, "training_run": False, "new_labels_produced": False,
        "new_state_engine_implemented": False,
        "full_population_synthetic_probe": "NOT_RUN: first attempt failed because bundled Python lacks sklearn; no install or monkeypatch. Boundary checks use existing _frame_index and AST-extracted eligibility expression only.",
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "existing_sample_groups": list(audit["groups"]), "existing_sample_field_summary": summary,
        "projection": declarations(OBS / "kernel.py", {"EVENT_TYPE_NAMES", "EVENT_PAYLOAD_FIELDS", "EVENT_STRING_FIELDS", "MONSTER_SUBTYPE_NAMES"}),
        "population_timing": declarations(OBS / "population.py", {"FRAME_LAG_MIN_MS", "FRAME_LAG_MAX_MS", "ANCHOR_OFFSET_MS"}),
        "synthetic_timing_examples": timing, "synthetic_checks": checks,
        "rule_status_legend_from_existing_file": rules["status_legend"],
        "existing_rules": rules["rules"], "map_source_claim": geometry["source"],
        "map_tower_entries": len(geometry["towers"]),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "limitation": "PASS concerns these source/probe checks, not real-match state correctness or full-population coverage",
    }
    (OUT / "RIFT_STATE_ENGINE_AUDIT_V1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks, "timing_examples": timing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
