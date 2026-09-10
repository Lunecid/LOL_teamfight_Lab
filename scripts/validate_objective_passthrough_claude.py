"""Validation C: do objectives actually reach the inputs and the outputs?

Three separate questions that are easy to conflate:
  C1  What objectives were actually taken in each window, per team and per type, read from the
      SOURCE events rather than from the state vector - and does that reproduce the frozen
      immediate counts (Baron 225 / Dragon 1,060 / Elder 18 / Soul 63)?
  C2  Does the state built by the SAME StateBuilder at t-1ms and at t actually move when an
      objective event lands, and which fields move?
  C4  What does the representation NOT recover - buff ownership and its expiry.

The model's input response is not the objective's value.  A probability difference between
t-1ms and t is a statement about the fitted coefficients, not a causal effect of the objective,
and simultaneous events are reported as such rather than separated.

Read-only.  The source cache is opened for reading; index-cache writing and fight dumping are
disabled, and every artefact this script writes goes under --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WINDOWS = (("pre_to_end", 0, 1), ("end_to_plus30", 1, 2), ("plus30_to_plus60", 2, 3))
FAMILIES = ("maymin", "expanded")
SAMPLE_PER_CATEGORY = 20
DRAGONS = ("AIR", "EARTH", "FIRE", "WATER", "HEXTECH", "CHEMTECH", "OTHER")


def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def objective_events(pack, team_map):
    """Every objective acquisition in one match, as (timestamp, team, category, detail)."""
    out = []
    for e in pack["events"]:
        typ = e.get("type")
        ts = int(e["timestamp"])
        if typ == "ELITE_MONSTER_KILL":
            killer = int(e.get("killerId", 0) or 0)
            team = int(e.get("killerTeamId", 0) or team_map.get(killer, 0))
            if team not in (100, 200):
                out.append((ts, 0, "unknown_team", str(e.get("monsterType", ""))))
                continue
            monster = e.get("monsterType", "")
            sub = str(e.get("monsterSubType", "")).upper()
            if monster == "DRAGON" and sub != "ELDER_DRAGON":
                kind = sub[:-len("_DRAGON")] if sub.endswith("_DRAGON") else sub
                out.append((ts, team, "dragon", kind if kind in DRAGONS else "OTHER"))
            else:
                cat = {"BARON_NASHOR": "baron", "RIFTHERALD": "herald", "HORDE": "horde",
                       "ATAKHAN": "atakhan", "DRAGON": "elder"}.get(monster)
                if cat:
                    out.append((ts, team, cat, monster))
        elif typ == "DRAGON_SOUL_GIVEN":
            team = int(e.get("teamId", 0) or 0)
            soul = str(e.get("dragonSoul", e.get("soulType", "OTHER"))).upper()
            soul = {"INFERNAL": "FIRE", "CLOUD": "AIR", "OCEAN": "WATER", "MOUNTAIN": "EARTH"}.get(soul, soul)
            out.append((ts, team if team in (100, 200) else 0,
                        "soul" if team in (100, 200) else "unknown_team", soul))
    return sorted(out)


def run(args):
    started = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        (out / "status_c.json").write_text(json.dumps(
            dict(stage=stage, pid=os.getpid(),
                 updated_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), **extra),
            indent=2, ensure_ascii=False), encoding="utf-8")

    status("running", step="configure_source_cache")
    os.environ["LOL_OUTPUT_ROOT"] = str(out / "runtime_c")
    os.environ["LOL_CFG_PRESET"] = "v3.3"
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({
        "CACHE_DIRNAME": str(args.cache_dir.resolve()),
        "FIGHT_INDEX_CACHE_ENABLED": False, "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False})
    from core.config import CACHE_DIR, NODE_FEATURE_NAMES
    from data.cache_io import load_match_cache
    from gameplay.state_value import StateBuilder
    assert CACHE_DIR.resolve() == args.cache_dir.resolve()

    with np.load(args.rows, allow_pickle=False) as z:
        rows = {k: z[k] for k in z.files}
    ids = rows["id"].astype(str)
    match = rows["match"].astype(str)
    query = rows["query_ms"]
    valid = rows["valid"]
    scale = rows["scale"]

    by_match = defaultdict(list)
    for i, m in enumerate(match):
        by_match[m].append(i)

    # ---------------- C1: window census straight from the source events ----------------
    status("running", step="C1_event_census")
    census = {w: {"events": Counter(), "rows_with_any": Counter(),
                  "by_team": Counter(), "valid_rows": 0} for w, _, _ in WINDOWS}
    dragon_kinds = {w: Counter() for w, _, _ in WINDOWS}
    per_row_events = {}
    unknown_team_events = 0
    missing_matches = []
    for number, (mid, idxs) in enumerate(sorted(by_match.items()), 1):
        pack = load_match_cache(mid)
        if pack is None:
            missing_matches.append(mid)
            continue
        team_map = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        events = objective_events(pack, team_map)
        unknown_team_events += sum(1 for e in events if e[1] == 0)
        for i in idxs:
            found = {}
            for wname, a, b in WINDOWS:
                if not (valid[i, a] and valid[i, b]):
                    continue
                census[wname]["valid_rows"] += 1
                lo, hi = int(query[i, a]), int(query[i, b])
                hits = [e for e in events if lo < e[0] <= hi and e[1] in (100, 200)]
                found[wname] = hits
                seen = set()
                for ts, team, cat, detail in hits:
                    census[wname]["events"][cat] += 1
                    census[wname]["by_team"][f"{cat}:{'blue' if team == 100 else 'red'}"] += 1
                    if cat == "dragon":
                        dragon_kinds[wname][detail] += 1
                    seen.add(cat)
                for cat in seen:
                    census[wname]["rows_with_any"][cat] += 1
            if any(found.values()):
                per_row_events[ids[i]] = {w: [(t, tm, c, d) for t, tm, c, d in v] for w, v in found.items() if v}
        if number % 2000 == 0:
            status("running", step=f"C1_event_census {number}/{len(by_match)}")

    frozen = json.loads(io.open(args.v2_dir / "results.json", encoding="utf-8").read())
    frozen_windows = frozen["observed_objective_windows"]
    immediate = census["pre_to_end"]["rows_with_any"]
    c1 = {
        "source": "objective events read from the match cache, not from the state vector",
        "windows": {w: {"valid_rows": census[w]["valid_rows"],
                        "events_by_category": dict(census[w]["events"]),
                        "rows_with_any_by_category": dict(census[w]["rows_with_any"]),
                        "events_by_category_and_team": dict(census[w]["by_team"]),
                        "dragon_subtypes": dict(dragon_kinds[w])} for w, _, _ in WINDOWS},
        "unknown_team_events_in_corpus": unknown_team_events,
        "matches_missing_from_cache": missing_matches,
        "reproduction_of_frozen_immediate_counts": {
            "frozen": frozen_windows,
            "recomputed": {"baron": immediate.get("baron", 0), "dragons": immediate.get("dragon", 0),
                           "elder": immediate.get("elder", 0), "soul_event_recorded": immediate.get("soul", 0)},
        },
    }
    c1["reproduction_of_frozen_immediate_counts"]["matches"] = {
        k: bool(c1["reproduction_of_frozen_immediate_counts"]["recomputed"][k] == v)
        for k, v in frozen_windows.items()}

    # ---------------- C3: which events to inspect at the source ----------------
    status("running", step="C3_select_samples")
    catalogue = []
    for eid, windows in per_row_events.items():
        i = int(np.flatnonzero(ids == eid)[0])
        for wname, hits in windows.items():
            for ts, team, cat, detail in hits:
                catalogue.append(dict(engagement_id=eid, match=str(match[i]), row=i, window=wname,
                                      ts=int(ts), team=int(team), category=cat, detail=detail,
                                      scale=float(scale[i])))
    selected, per_bucket = [], Counter()
    # Every Elder in the immediate window is inspected; the handoff calls that group out by name.
    for rec in catalogue:
        if rec["category"] == "elder" and rec["window"] == "pre_to_end":
            selected.append({**rec, "selection": "all_elder_immediate"})
    chosen_ids = {(r["engagement_id"], r["ts"], r["category"]) for r in selected}
    # Everything else: a fixed hash of the identity, deliberately independent of delta size.
    for rec in sorted(catalogue, key=lambda r: stable_int(f"objsample:7:{r['engagement_id']}:{r['ts']}:{r['category']}")):
        key = (rec["engagement_id"], rec["ts"], rec["category"])
        if key in chosen_ids:
            continue
        bucket = f"{rec['window']}:{rec['category']}"
        if per_bucket[bucket] >= SAMPLE_PER_CATEGORY:
            continue
        per_bucket[bucket] += 1
        selected.append({**rec, "selection": f"hash_sample:{bucket}"})
    c3_note = {"selection_rule": "all immediate Elder windows, then a fixed sha256 rank per "
                                 "window-and-category bucket, capped at 20; the rank does not "
                                 "look at delta size", "bucket_counts": dict(per_bucket),
               "total_inspected": len(selected)}

    # ---------------- C2: t-1ms vs t through the same StateBuilder ----------------
    status("running", step="C2_state_transition")
    import joblib
    models = {f: joblib.load(args.v2_dir / f"{f}_model.joblib") for f in FAMILIES}
    names = json.loads(io.open(args.dataset / "schema.json", encoding="utf-8").read())["state_names"]
    inspected, builders = [], {}
    for rec in selected:
        mid = rec["match"]
        if mid not in builders:
            pack = load_match_cache(mid)
            if pack is None:
                continue
            team_map = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
            builders[mid] = (StateBuilder(pack, NODE_FEATURE_NAMES), objective_events(pack, team_map))
        builder, events = builders[mid]
        t = rec["ts"]
        try:
            before, after = builder.at(t - 1), builder.at(t)
        except ValueError as exc:
            inspected.append({**rec, "error": str(exc)})
            continue
        b = np.array([before.values[n] for n in names])
        a = np.array([after.values[n] for n in names])
        moved = [names[k] for k in np.flatnonzero(a != b)]
        simultaneous = [e for e in events if e[0] == t and not (
            e[1] == rec["team"] and e[2] == rec["category"] and e[3] == rec["detail"])]
        related = [n for n in moved if any(tag in n for tag in
                   ("baron", "elder", "dragon", "soul", "herald", "horde", "atakhan"))]
        probs = {f: (float(m.predict_proba(b.reshape(1, -1))[0, 1]),
                     float(m.predict_proba(a.reshape(1, -1))[0, 1])) for f, m in models.items()}
        inspected.append({
            **rec,
            "same_source_frame": bool(before.snapshot_ms == after.snapshot_ms),
            "fields_moved": len(moved),
            "objective_fields_moved": len(related),
            "objective_fields_sample": related[:12],
            "non_objective_fields_moved": len(moved) - len(related),
            "simultaneous_other_events": [{"ts": e[0], "team": e[1], "category": e[2], "detail": e[3]}
                                          for e in simultaneous],
            "model_input_response_pp": {f: round((p[1] - p[0]) * 100., 4) for f, p in probs.items()},
        })
    ok = [r for r in inspected if "error" not in r]
    c2 = {
        "meaning": "A probability move here is the fitted model's response to an input change at "
                   "the event timestamp. It is NOT the objective's causal value, and rows with "
                   "simultaneous events do not isolate one objective.",
        "inspected": len(inspected),
        "errors": [r for r in inspected if "error" in r],
        "every_event_moved_objective_fields": bool(ok) and all(r["objective_fields_moved"] > 0 for r in ok),
        "rows_with_no_field_change": sum(1 for r in ok if r["fields_moved"] == 0),
        "rows_with_simultaneous_events": sum(1 for r in ok if r["simultaneous_other_events"]),
        "same_source_frame_fraction": float(np.mean([r["same_source_frame"] for r in ok])) if ok else None,
        "by_category": {},
        "records": inspected,
    }
    for cat in sorted({r["category"] for r in ok}):
        sub = [r for r in ok if r["category"] == cat]
        for family in FAMILIES:
            moves = [r["model_input_response_pp"][family] for r in sub]
            c2["by_category"].setdefault(cat, {"n": len(sub)})[family] = dict(
                mean_pp=float(np.mean(moves)), median_pp=float(np.median(moves)),
                min_pp=float(np.min(moves)), max_pp=float(np.max(moves)))

    # ---------------- C4: what the representation cannot recover ----------------
    status("running", step="C4_buff_recovery")
    baron = next((r for r in ok if r["category"] == "baron"), None)
    c4 = {"claim": "Objective HISTORY is represented; buff ownership and its expiry are not.",
          "fields_used": ["<team>_<obj>_ever", "<team>_<obj>_age_minutes",
                          "<team>_<obj>_acquired_last_{60,120,180,300}s",
                          "slot<i>_<obj>_death_since_acquisition"],
          "not_represented": ["whether the buff is currently active", "buff expiry at 180s",
                              "buff loss on death (only a death-since-acquisition flag exists)"],
          "note": "gameplay/state_value.py deliberately ignores the cached buff flags: its docstring "
                  "records that their builder updates a frame late and does not clear on death."}
    if baron is not None:
        builder, _ = builders[baron["match"]]
        t = baron["ts"]
        team = "blue_" if baron["team"] == 100 else "red_"
        probe = {}
        for offset in (0, 60_000, 179_000, 181_000, 301_000):
            try:
                st = builder.at(t + offset)
            except ValueError:
                continue
            probe[f"t+{offset//1000}s"] = {k: st.values[k] for k in (
                team + "baron_ever", team + "baron_age_minutes",
                team + "baron_acquired_last_60s", team + "baron_acquired_last_180s",
                team + "baron_acquired_last_300s")}
        c4["worked_example"] = {"engagement_id": baron["engagement_id"], "event_ms": t,
                                "team": baron["team"], "probe": probe,
                                "reading": "The 180s flag is an age proxy that flips on a clock, "
                                           "not an observation that the buff ended."}

    status("running", step="write")
    results = {"question": "Do objectives actually reach the inputs and the outputs?",
               "interpretation_guard": (
                   "C reports pass-through, not value. The difference between including and "
                   "excluding an objective is not that objective's causal contribution."),
               "C1_event_census": c1, "C2_state_transition": c2,
               "C3_sampling": c3_note, "C4_representation_limits": c4,
               "elapsed_seconds": round(time.time() - started, 2), "status": "complete"}
    (out / "results_c.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    from importlib.metadata import version
    (out / "manifest_c.json").write_text(json.dumps({
        "command": " ".join([sys.executable, *sys.argv]), "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scikit-learn", "joblib")},
        "source_cache": str(args.cache_dir), "cache_written": False,
        "inputs": {"engagement_rows.npz": sha256(args.rows)},
        "source": {"scripts/validate_objective_passthrough_claude.py": sha256(Path(__file__)),
                   "gameplay/state_value.py": sha256(ROOT / "gameplay/state_value.py")},
        "elapsed_seconds": results["elapsed_seconds"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    status("complete", elapsed_seconds=results["elapsed_seconds"], results="results_c.json")
    print(json.dumps({"reproduction": c1["reproduction_of_frozen_immediate_counts"],
                      "inspected": c2["inspected"],
                      "all_moved": c2["every_event_moved_objective_fields"],
                      "elapsed_seconds": results["elapsed_seconds"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13"))
    ap.add_argument("--rows", type=Path,
                    default=ROOT / "outputs/temporal_winprob_claude_validation/engagement_rows.npz")
    ap.add_argument("--v2-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v2")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs/temporal_winprob_claude_validation")
    args = ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "status_c.json").write_text(
            json.dumps({"stage": "failed", "error": repr(exc), "pid": os.getpid()}, indent=2),
            encoding="utf-8")
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
