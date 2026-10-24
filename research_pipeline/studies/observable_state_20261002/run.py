"""Bounded CPU execution and complete independent stored-ledger acceptance."""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing
import platform
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bank import match_view
from .categorical import lookups
from .kernel import EVENT_PAYLOAD_FIELDS, EVENT_STRING_ENUMS, EVENT_STRING_FIELDS, EVENT_TYPE_NAMES, OBJECTIVE_TYPES, history_feature_description
from .producer import COMPONENT_COLUMNS, produce_shard, producer_layout
from .sampling import SAMPLER_FORMULA, SAMPLER_FORMULA_SHA256
from .schema import make_field_manifest
from .source import ACTIVE_ROLES, EXCLUDED_KEY, REPORT, ROOT, canonical_key, file_sha256, load_frozen_entries
from .storage import close_shard, configure_write_lock, open_shard


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _initialize_worker(lock) -> None:
    configure_write_lock(lock)


def bindings() -> dict:
    paths = list((ROOT / "studies" / "observable_state_20261002").glob("*.py"))
    paths += [ROOT / p for p in (
        "reports/post_state_label_contract_20261002/CONTRACT.json",
        "reports/input_population_specification_20261002/INPUT_POLICY.json",
        "reports/population_adoption_20261002/CONTRACT.json",
        "reports/observable_state_20261002/USER_EXCLUSION.json",
        "worktrees/engagement-state-value/studies/st_dbscan_20261001/kernel.py",
        "studies/cache_source_audit_20261002/audit.py", "studies/cache_source_audit_20261002/run_audit.py",
        "studies/cache_source_audit_20261002/config.py", "studies/input_specification_20261002/schema_extract.py",
        "studies/cache_source_audit_20261002/champion_map.py",
        "worktrees/engagement-state-value/core/config.py",
        "worktrees/engagement-state-value/gameplay/pipeline_cache.py",
        "worktrees/engagement-state-value/data/cache_io.py",
        "worktrees/engagement-state-value/gameplay/event_aggregation.py",
        "worktrees/engagement-state-value/core/feature_contract.py",
        "reports/feature_control_study_20261001/api_snapshot/official_dto_inventory.json",
        "reports/population_adoption_20261002/ACCEPTANCE.json",
        "outputs/observation_support_20261002/run_20261002T000935Z_full_direct_full_r3/RUN_MANIFEST.json",
        "outputs/anchor10_boundary_study_20261001/st_dbscan_20261001T131304Z/source/train_match_ids.json",
    )]
    result = {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in paths}
    result["categorical_maps"] = {patch: lookups(patch)["source_hashes"] for patch in ("15.14", "15.15", "15.16")}
    for path in (REPORT / "static_maps_verified_20261002").glob("*.json"):
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    for patch in ("15.14", "15.15", "15.16"):
        for kind in ("item", "championFull"):
            path = ROOT / "outputs/reest_exact_v4_20260925/datadragon_raw" / f"{kind}_{patch}.1.json"
            result[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return result


def executable_schema() -> dict:
    schema = make_field_manifest(history_feature_description())
    schema.update(producer_layout())
    schema.update({"event_type_names": list(EVENT_TYPE_NAMES), "event_numeric_field_order": list(EVENT_PAYLOAD_FIELDS),
                   "event_string_field_order": list(EVENT_STRING_FIELDS),
                   "event_string_enum_names": {k: list(v) for k, v in EVENT_STRING_ENUMS.items()},
                   "objective_category_order": list(OBJECTIVE_TYPES), "objective_count_team_order": ["total", "blue", "red"],
                   "KDA_order": ["kills", "deaths", "recorded_assists"], "team_order": [100, 200],
                   "team_growth_order": ["level_mean", "xp_sum", "curGold_sum", "totalGold_sum", "laneCS_sum", "jgCS_sum"],
                   "team_growth_derivation": "held E values per actual team; mean for level, sums otherwise; blue minus red; cached team gold held separately",
                   "bans_shape_order": {"shape": [2, 5], "team_order": [100, 200], "position_order": [0, 1, 2, 3, 4]},
                   "structured_event_fields": {"timestamp": "recorded ms <= query", "source_index": "original source ordinal",
                       "numeric_fields": "recorded integer plus validity; zero does not imply absent",
                       "strings": "fixed known enum codes plus valid unknown UTF8 tokens",
                       "assists": "ragged recorded IDs plus presence validity; missing remains unknown",
                       "position": "recorded event coordinate; audit and history only, never future centering for q"},
                   "source_units": "cached normalization retained for frames; no inverse clip or invented per-value raw masks",
                   "recorded_inventory": "derived only by supported transaction replay; does not assert actual game inventory",
                   "state_generator_complete": False})
    return schema


def _verify_shard(directory: Path, expected: list[dict]) -> dict:
    layout = json.loads((directory / "LAYOUT.json").read_text(encoding="utf-8"))
    ledgers = [json.loads(line) for line in (directory / "MATCH_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(ledgers) != len(expected):
        raise ValueError("stored match ledger cardinality mismatch")
    arrays = open_shard(directory)
    totals = Counter()
    columns = {name: index for index, name in enumerate(COMPONENT_COLUMNS)}
    train_rows = []
    support = Counter()
    try:
        for index, (entry, ledger) in enumerate(zip(expected, ledgers)):
            if ledger["canonical_match_key"] != entry["canonical_match_key"] or ledger["role"] != entry["role"]:
                raise ValueError("stored match identity or split changed")
            if ledger["canonical_match_key"] == EXCLUDED_KEY or not ledger["source_stable"]:
                raise ValueError("excluded or unstable source entered bank")
            if ledger["source_hashes"] != entry["source_hashes"]:
                raise ValueError("stored source binding changed")
            bank = match_view(arrays, layout, index)
            for i, value in enumerate(np.sum(bank["node_frame"] == 0, axis=(0, 1))):
                support[f"frame_zero:{i}"] += int(value)
            for i, value in enumerate(np.sum(~np.isfinite(bank["node_frame"]), axis=(0, 1))):
                support[f"frame_nonfinite:{i}"] += int(value)
            presence = [int(np.sum((bank["event_field_valid_mask"] >> i) & 1)) for i in range(len(EVENT_PAYLOAD_FIELDS))]
            if presence != ledger["support"]["event_field_presence"]:
                raise ValueError("saved event field support differs from producer")
            for name, value in zip(EVENT_PAYLOAD_FIELDS, presence):
                support["event_presence:" + name] += value
            for name, diagnoses in ledger["support"]["static_lookup_counts"].items():
                for status, value in diagnoses.items():
                    support[f"C.{name}:{status}"] += value
            for reason, value in ledger["support"]["component_reason_counts"].items():
                support["component_reason:" + reason] += value
            support["n_item_ledger_known_queries"] += int(bank["query_item_ledger_known"].sum())
            ts, query, frame = bank["minute_ts"], bank["query_ms"], bank["query_frame_index"]
            if not np.array_equal(frame, np.searchsorted(ts, query, side="right") - 1):
                raise ValueError("stored frame reference is not last supported frame")
            if query.size and (query.min() < ts[0] or query.max() > ts[-1]):
                raise ValueError("stored query outside support")
            limit = np.searchsorted(bank["event_timestamp"], query, side="right")
            if not np.array_equal(limit, bank["query_event_limit"]):
                raise ValueError("stored event prefix violates inclusive cutoff")
            if len(query) > 1 and np.any(query[1:] <= query[:-1]):
                raise ValueError("duplicate query identity")
            nominal = bank["v_nominal"]
            if len(nominal):
                lo, hi, t, actual, ref = (nominal[:, k] for k in (1, 2, 3, 4, 6))
                if np.any(lo != 120000 * (nominal[:, 0] + 1)) or np.any(hi != np.minimum(lo + 119999, int(bank["end_ms"]) - 1)):
                    raise ValueError("nominal bucket bounds changed")
                if np.any(t < lo) or np.any(t > hi):
                    raise ValueError("nominal query outside bucket")
                good = actual >= 0
                if np.any(actual[good] != t[good]) or np.any(ref[~good] != -1) or np.any(query[ref[good]] != actual[good]):
                    raise ValueError("nominal query redrawn, clamped, or reference mismatch")
            components = bank["component_values"]
            if len(components) and len(set(bank["component_key"].tolist())) != len(components):
                raise ValueError("duplicate component identity within match")
            references = bank["component_query_refs"]
            for row, refs in zip(components, references):
                primary = bool(row[columns["primary_candidate_eligible"]] and row[columns["supported_candidate_eligible"]])
                if primary and (refs[0] < 0 or query[refs[0]] != row[columns["pre"]]):
                    raise ValueError("supported primary pre query missing")
                if not primary and np.any(refs != -1):
                    raise ValueError("unsupported or early cohort entered primary query references")
                for h, ref in zip((60, 90, 120), refs[1:]):
                    valid = bool(row[columns[f"valid_h{h}"]])
                    if primary and valid and (ref < 0 or query[ref] != row[columns[f"e_h{h}"]]):
                        raise ValueError("supported post query missing")
                    if not (primary and valid) and ref != -1:
                        raise ValueError("invalid post query retained")
            count = {"n_matches": 1, "n_frames": len(ts), "n_events": len(bank["event_timestamp"]),
                     "n_kills": int(np.sum(bank["event_type_code"] == EVENT_TYPE_NAMES.index("CHAMPION_KILL"))),
                     "n_components": len(components), "n_nominal_V_queries": len(nominal),
                     "n_supported_V_queries": int(np.sum(nominal[:, 4] >= 0)), "n_unique_queries": len(query),
                     "n_primary_candidates": int(components[:, columns["primary_candidate_eligible"]].sum()),
                     "n_supported_primary": int(np.sum(components[:, columns["primary_candidate_eligible"]] & components[:, columns["supported_candidate_eligible"]]))}
            if any(ledger["counts"][k] != v for k, v in count.items()):
                raise ValueError("independent stored row counts differ from producer")
            totals.update(count)
            if entry["role"] == "TRAIN":
                train_rows.append((entry["canonical_match_key"], components[:, [columns[k] for k in
                    ("component_id", "f", "L", "pre", "candidate_eligible", "primary_candidate_eligible",
                     "firstkill_credit_blue", "firstkill_credit_red")]].copy()))
    finally:
        close_shard(arrays)
    return {"counts": dict(totals), "train_rows": train_rows, "support": dict(support)}


def _reconcile_train(rows: list[tuple[str, np.ndarray]], *, full: bool = True) -> dict:
    acceptance_path = ROOT / "reports/population_adoption_20261002/ACCEPTANCE.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    source_manifest_path = ROOT / "outputs/observation_support_20261002/run_20261002T000935Z_full_direct_full_r3/RUN_MANIFEST.json"
    if file_sha256(source_manifest_path) != acceptance["source_manifest_hashes"]["run_manifest_sha256"]:
        raise ValueError("accepted TRAIN parent manifest changed")
    ids_path = ROOT / "outputs/anchor10_boundary_study_20261001/st_dbscan_20261001T131304Z/source/train_match_ids.json"
    if file_sha256(ids_path) != "96cd7979537bda8abad74dfdc42163e3fc00fd426dad842794933caa282da7ff":
        raise ValueError("accepted TRAIN local match order changed")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(ids) != ACTIVE_ROLES["TRAIN"] or len(set(ids)) != len(ids):
        raise ValueError("accepted TRAIN source order is not one-to-one")
    legacy_keys = {canonical_key(mid): f"m{index:05d}" for index, mid in enumerate(ids)}
    path = Path(acceptance["accepted_run"]) / "population_projection.npz"
    if file_sha256(path) != acceptance["projection_sha256"]:
        raise ValueError("accepted TRAIN projection changed")
    with np.load(path, allow_pickle=False) as old:
        keys = old["canonical_match_key"]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        ends = np.r_[starts[1:], len(keys)]
        positions = {str(keys[a]): (int(a), int(b)) for a, b in zip(starts, ends)}
        old_columns = np.column_stack([old[k] for k in ("firstkill_ms", "lastkill_ms", "pre_ms", "candidate_eligible",
                                                       "firstkill_credit_blue", "firstkill_credit_red")])
        changes = Counter()
        diffs = []
        seen = set()
        for key, values in rows:
            if key not in legacy_keys:
                raise ValueError("new TRAIN match outside accepted source order")
            old_key = legacy_keys[key]
            if old_key in seen:
                raise ValueError("duplicate TRAIN match in independent reconciliation")
            seen.add(old_key)
            if old_key not in positions:
                if len(values):
                    raise ValueError("new TRAIN component outside accepted matches")
                continue
            a, b = positions[old_key]
            if len(values) != b - a or not np.array_equal(values[:, 1:4], old_columns[a:b, :3]):
                raise ValueError("accepted TRAIN component times/counts changed")
            if not np.array_equal(values[:, 6:8], old_columns[a:b, 4:6]):
                raise ValueError("accepted TRAIN first-block credit changed")
            previous = old_columns[a:b, 3].astype(bool)
            current = values[:, 4].astype(bool)
            changed = previous != current
            changes.update(n_rows=len(values), n_candidate_changed=int(changed.sum()),
                           n_candidate_removed=int(np.sum(previous & ~current)), n_candidate_added=int(np.sum(~previous & current)))
            for local in np.flatnonzero(changed):
                diffs.append({"canonical_match_key": key, "component_id": int(local), "previous_candidate": bool(previous[local]),
                              "current_candidate": bool(current[local]), "f": int(values[local, 1]), "L": int(values[local, 2])})
        if full and not set(positions) <= seen:
            raise ValueError("accepted TRAIN reconciliation omitted component-bearing matches")
    return {"accepted_projection_sha256": acceptance["projection_sha256"], "full_TRAIN_coverage": full,
            "all_component_times_and_credits_identical": True,
            "counts": dict(changes), "candidate_changes": diffs,
            "policy": "every candidate change preserved for review; never force older candidate counts"}


def run(output: Path, *, workers: int = 3, shard_matches: int = 512, pilot: bool = False, pilot_per_role: int = 4) -> dict:
    if workers not in (1, 2, 3) or not 1 <= shard_matches <= 1024:
        raise ValueError("CPU worker or shard size exceeds approved limits")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen, cohort = load_frozen_entries()
    entries = frozen
    if pilot:
        if not 1 <= pilot_per_role <= 1024:
            raise ValueError("resource pilot size is outside bounded scope")
        entries = [r for role in ACTIVE_ROLES for r in [x for x in frozen if x["role"] == role][:pilot_per_role]]
    code = bindings()
    schema = executable_schema()
    _json(output / "FIELD_MANIFEST.json", schema)
    manifest = {"complete": False, "accepted": False, "pilot": pilot, "cohort": cohort, "bindings": code,
                "field_manifest_sha256": file_sha256(output / "FIELD_MANIFEST.json"),
                "sampler_formula": SAMPLER_FORMULA, "sampler_sha256": SAMPLER_FORMULA_SHA256,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "workers": workers,
                                "CPU_budget": 4, "GPU_used": False, "numeric_threads": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
                "started_utc": datetime.now(timezone.utc).isoformat(), "new_models_fitted": False,
                "new_SVI_labels_produced": False, "training_ready": False, "shards": []}
    _json(output / "RUN_MANIFEST.json", manifest)
    tasks = []
    for role in ACTIVE_ROLES:
        group = [r for r in entries if r["role"] == role]
        for start in range(0, len(group), shard_matches):
            tasks.append((str(output / f"{role.lower()}_{start // shard_matches:04d}"), group[start:start + shard_matches]))
    totals = Counter()
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_initialize_worker, initargs=(context.Lock(),)) as pool:
            pending, iterator = {}, iter(tasks)
            for _ in range(workers):
                task = next(iterator, None)
                if task:
                    pending[pool.submit(produce_shard, task)] = task
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    result = future.result()
                    totals.update(result["counts"])
                    manifest["shards"].append(result)
                    manifest["counts"] = dict(totals)
                    manifest["elapsed_seconds"] = time.perf_counter() - started
                    _json(output / "RUN_MANIFEST.json", manifest)
                    print(json.dumps({"phase": "produce", "completed_matches": totals["n_matches"], "total_matches": len(entries),
                                      "shard": result["directory"], "payload_GiB": round(result["payload_bytes"] / (1 << 30), 3)}), flush=True)
                    next_task = next(iterator, None)
                    if next_task:
                        pending[pool.submit(produce_shard, next_task)] = next_task
        if bindings() != code:
            raise ValueError("code, maps or adopted contracts changed during execution")
        independently = Counter()
        roles = Counter()
        supports = {role: Counter() for role in ACTIVE_ROLES}
        train_rows = []
        for directory, group in tasks:
            verified = _verify_shard(Path(directory), group)
            independently.update(verified["counts"])
            roles[group[0]["role"]] += len(group)
            supports[group[0]["role"]].update(verified["support"])
            train_rows.extend(verified["train_rows"])
            print(json.dumps({"phase": "independent_reload", "verified_matches": independently["n_matches"]}), flush=True)
        if any(totals[k] != v for k, v in independently.items()):
            raise ValueError("independent full-bank counts disagree")
        if not pilot:
            baseline = json.loads((REPORT / "ACTIVE_COHORT_BASELINE.json").read_text(encoding="utf-8"))
            expected = {"n_matches": 209999, "n_frames": baseline["n_frames_expected"], "n_kills": baseline["n_kills_expected"],
                        "n_events": baseline["n_events_expected"], "n_nominal_V_queries": baseline["n_nominal_V_queries_expected_from_accepted_terminal_audit"]}
            if dict(roles) != ACTIVE_ROLES or any(totals[k] != v for k, v in expected.items()):
                raise ValueError("full frozen cohort/count acceptance mismatch")
            reconciled = _reconcile_train(train_rows)
            _json(output / "TRAIN_RECONCILIATION.json", reconciled)
            if reconciled["counts"]["n_candidate_added"] or reconciled["counts"]["n_candidate_removed"]:
                manifest["population_review_required"] = True
        _, final_cohort = load_frozen_entries()
        if final_cohort != cohort or bindings() != code:
            raise ValueError("final frozen evidence binding changed")
        _json(output / "INDEPENDENT_CHECKS.json", {"complete_full_coverage": not pilot, "counts": dict(independently),
                                                  "roles": dict(roles), "all_saved_arrays_reloaded_and_hashed": True,
                                                  "all_match_sparse_and_query_semantic_roundtrips": True,
                                                  "excluded_source_read": False})
        _json(output / "FIELD_SUPPORT_COUNTS.json", {"full_corpus": not pilot,
              "role_counts": {role: dict(counts) for role, counts in supports.items()},
              "raw_field_presence_verified": False, "cache_zero_retained": True,
              "missing_assist_field_is_unknown_not_zero_actual_assists": True})
        manifest.update(complete=True, accepted=not pilot and not manifest.get("population_review_required", False),
                        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
    except Exception as exc:
        manifest.update(error=f"{type(exc).__name__}: {exc}", failed_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        _json(output / "RUN_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shard-matches", type=int, default=512)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-per-role", type=int, default=4)
    args = parser.parse_args()
    result = run(args.output, workers=args.workers, shard_matches=args.shard_matches, pilot=args.pilot, pilot_per_role=args.pilot_per_role)
    print(json.dumps({k: result[k] for k in ("complete", "accepted", "pilot", "counts", "elapsed_seconds")}), flush=True)
