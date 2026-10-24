"""Full-corpus objective timing runner (CPU<=4, no GPU, no V/q fit)."""
from __future__ import annotations

from . import thread_limits as TL  # noqa: F401

import argparse
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import aggregate as AGG
from . import config as C
from . import independent_verify as IV
from . import io_util as IO
from .parse import parse_objective_events, rows_to_arrays
from .query import (
    classify_uncapped_vs_next_anchor,
    query_match_components_vectorized,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _process_tree_rss_bytes() -> dict[str, int | None]:
    try:
        import psutil

        parent = psutil.Process(os.getpid())
        parent_rss = int(parent.memory_info().rss)
        total = parent_rss
        for ch in parent.children(recursive=True):
            try:
                total += int(ch.memory_info().rss)
            except (psutil.Error, OSError):
                pass
        return {"parent_rss_bytes": parent_rss, "process_tree_rss_bytes": total}
    except Exception:
        return {"parent_rss_bytes": _peak_rss_bytes(), "process_tree_rss_bytes": None}


def _match_ranges(match_local: np.ndarray, n_matches: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(np.asarray(match_local, dtype=np.int64), minlength=n_matches)
    ends = np.cumsum(counts)
    starts = ends - counts
    return starts.astype(np.int64), ends.astype(np.int64)


def _empty_query_block(n: int) -> dict[str, np.ndarray]:
    return {
        "status": np.full(n, C.ST_NO_ACQ, dtype=np.int8),
        "t_ms": np.full(n, C.SENTINEL_MISSING, dtype=np.int64),
        "delay_ms": np.full(n, C.SENTINEL_MISSING, dtype=np.int64),
        "event_ord": np.full(n, C.SENTINEL_MISSING, dtype=np.int64),
        "team": np.full(n, C.TEAM_UNKNOWN, dtype=np.int16),
        "same_ms_at_L": np.zeros(n, dtype=np.int8),
        "after_last_frame": np.zeros(n, dtype=np.int8),
        "n_tie_events": np.zeros(n, dtype=np.int16),
        "tie_category_mask": np.zeros(n, dtype=np.int64),
        "tie_teams": np.zeros(n, dtype=np.int8),
        "uncertain_precedes_confirmed": np.zeros(n, dtype=np.int8),
        "uncertain_in_interval": np.zeros(n, dtype=np.int8),
        "neutral_precedes_confirmed": np.zeros(n, dtype=np.int8),
        "neutral_in_interval": np.zeros(n, dtype=np.int8),
    }


def _process_match_batch(payload: dict[str, Any]) -> dict[str, Any]:
    from . import thread_limits as _tl  # noqa: F401

    match_locals = [int(x) for x in payload["match_locals"]]
    match_ids = payload["match_ids"]
    blue_pids = np.asarray(payload["blue_pids"], dtype=np.int16)
    starts = np.asarray(payload["comp_starts"], dtype=np.int64)
    ends = np.asarray(payload["comp_ends"], dtype=np.int64)
    L = np.asarray(payload["L"], dtype=np.int64)
    ge = np.asarray(payload["game_end_ms"], dtype=np.int64)
    gek = np.asarray(payload["game_end_known"], dtype=np.int8)
    lf = np.asarray(payload["last_frame_ms"], dtype=np.int64)
    nk = np.asarray(payload["next_kill_after_L"], dtype=np.int64)
    nt = np.asarray(payload["next_component_tau"], dtype=np.int64)
    cid = np.asarray(payload["component_id"], dtype=np.int64)

    event_chunks: list[dict[str, np.ndarray]] = []
    # Per global component index within this batch's component arrays
    n_comp_batch = int(L.size)
    query_out: dict[str, dict[str, np.ndarray]] = {}
    for cat in C.CATEGORY_ORDER:
        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
            query_out[f"{interval}:{cat}"] = _empty_query_block(n_comp_batch)

    census_acc: dict[str, Any] = {
        "n_matches": 0,
        "n_missing_events": 0,
        "n_parse_errors": 0,
        "n_events_total": 0,
        "n_elite_monster_kill": 0,
        "n_building_kill": 0,
        "n_malformed": 0,
        "n_invalid_objective_timestamp": 0,
        "n_support_unknown_matches": 0,
        "n_acq_confirmed": 0,
        "n_acq_neutral": 0,
        "n_acq_unknown": 0,
        "n_acq_contradictory": 0,
        "n_after_last_frame": 0,
        "malformed_reasons": Counter(),
        "monster_type_counts": Counter(),
        "monster_subtype_counts": Counter(),
        "building_type_counts": Counter(),
        "tower_type_counts": Counter(),
        "team_source_counts": Counter(),
        "contamination_uncapped_any_objective": Counter(),
    }
    n_queries = 0
    t0 = time.perf_counter()
    errors: list[dict[str, Any]] = []

    for i, ml in enumerate(match_locals):
        mid = match_ids[i]
        census_acc["n_matches"] += 1
        try:
            events, err = IO.load_events(str(mid))
            support_unknown = bool(err)
            if err == "missing_events_json":
                census_acc["n_missing_events"] += 1
                errors.append({"match_local": ml, "error": err})
            elif err:
                census_acc["n_parse_errors"] += 1
                errors.append({"match_local": ml, "error": err})

            s = int(starts[i])
            e = int(ends[i])
            if e > s:
                last_frame = int(lf[s])
                game_end = int(ge[s])
                game_end_known = bool(gek[s])
                L_m = L[s:e]
                nk_m = nk[s:e]
                nt_m = nt[s:e]
            else:
                last_frame = C.SENTINEL_MISSING
                game_end = C.SENTINEL_MISSING
                game_end_known = False
                L_m = np.zeros(0, dtype=np.int64)
                nk_m = np.zeros(0, dtype=np.int64)
                nt_m = np.zeros(0, dtype=np.int64)

            if blue_pids.ndim == 2:
                blue = [int(x) for x in blue_pids[ml] if int(x) > 0]
            else:
                blue = [int(x) for x in blue_pids if int(x) > 0]

            # Schema failure: do not invent empty-timeline NO_ACQ.
            if support_unknown:
                census_acc["n_support_unknown_matches"] += 1
                if e > s:
                    for cat in C.CATEGORY_ORDER:
                        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
                            query_out[f"{interval}:{cat}"]["status"][s:e] = C.ST_UNKNOWN_OBS
                    n_queries += int(L_m.size) * len(C.CATEGORY_ORDER) * 2
                continue

            rows, census = parse_objective_events(events, blue, last_frame)
            for k in (
                "n_events_total",
                "n_elite_monster_kill",
                "n_building_kill",
                "n_malformed",
                "n_invalid_objective_timestamp",
                "n_acq_confirmed",
                "n_acq_neutral",
                "n_acq_unknown",
                "n_acq_contradictory",
                "n_after_last_frame",
            ):
                census_acc[k] += int(census.get(k, 0))
            for ck in (
                "malformed_reasons",
                "monster_type_counts",
                "monster_subtype_counts",
                "building_type_counts",
                "tower_type_counts",
                "team_source_counts",
            ):
                census_acc[ck].update(census.get(ck, {}))

            if census.get("observation_support_unknown"):
                support_unknown = True
                census_acc["n_support_unknown_matches"] += 1
                errors.append(
                    {
                        "match_local": ml,
                        "error": "invalid_objective_timestamp_or_malformed_objective",
                    }
                )

            arr = rows_to_arrays(rows)
            n_ev = int(arr["t_ms"].size)
            if n_ev and not support_unknown:
                event_chunks.append(
                    {
                        "match_local": np.full(n_ev, ml, dtype=np.int32),
                        **arr,
                    }
                )

            if e > s:
                if support_unknown:
                    for cat in C.CATEGORY_ORDER:
                        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
                            query_out[f"{interval}:{cat}"]["status"][s:e] = C.ST_UNKNOWN_OBS
                    n_queries += int(L_m.size) * len(C.CATEGORY_ORDER) * 2
                    continue

                q = query_match_components_vectorized(
                    arr["t_ms"],
                    arr["event_ord"],
                    arr["category_mask"],
                    arr["team"],
                    arr["after_last_frame"],
                    arr["acq_status"],
                    L_m,
                    game_end,
                    game_end_known,
                    nk_m,
                    nt_m,
                )
                n_queries += int(L_m.size) * len(C.CATEGORY_ORDER) * 2
                for key, block in q.items():
                    for field, vals in block.items():
                        query_out[key][field][s:e] = vals

                st = q[f"{C.INTERVAL_UNCAPED}:any_objective"]["status"]
                tms = q[f"{C.INTERVAL_UNCAPED}:any_objective"]["t_ms"]
                for j in range(L_m.size):
                    label = classify_uncapped_vs_next_anchor(
                        int(tms[j]),
                        int(st[j]),
                        int(L_m[j]),
                        int(nk_m[j]),
                        int(nt_m[j]),
                        game_end,
                        game_end_known,
                    )
                    census_acc["contamination_uncapped_any_objective"][label] += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "match_local": ml,
                    "error": f"{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    # Stack events
    if event_chunks:
        ev = {k: np.concatenate([c[k] for c in event_chunks]) for k in event_chunks[0]}
    else:
        ev = {
            "match_local": np.zeros(0, dtype=np.int32),
            "t_ms": np.zeros(0, dtype=np.int64),
            "event_ord": np.zeros(0, dtype=np.int64),
            "category_mask": np.zeros(0, dtype=np.int64),
            "team": np.zeros(0, dtype=np.int16),
            "acq_status": np.zeros(0, dtype=np.int8),
            "after_last_frame": np.zeros(0, dtype=np.int8),
            "raw_killerTeamId": np.zeros(0, dtype=np.int32),
            "raw_killerId": np.zeros(0, dtype=np.int32),
            "raw_building_owner": np.zeros(0, dtype=np.int32),
            "leaf_code": np.zeros(0, dtype=np.int16),
        }

    # Convert counters
    census_out = {
        k: (dict(v) if isinstance(v, Counter) else v) for k, v in census_acc.items()
    }
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "events": ev,
        "query": query_out,
        "comp_ids": cid,
        "n_comp": n_comp_batch,
        "n_queries": n_queries,
        "wall_s": float(time.perf_counter() - t0),
        "census": census_out,
        "match_locals": match_locals,
        "comp_starts": starts,
        "comp_ends": ends,
    }


def _allocate_full_query(n_comp: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for cat in C.CATEGORY_ORDER:
        for interval in (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT):
            out[f"{interval}:{cat}"] = _empty_query_block(n_comp)
    return out


def reload_verify(
    run_dir: Path,
    components: dict[str, np.ndarray],
    aggregates: dict[str, Any],
    *,
    expect_full_cohorts: bool = True,
) -> dict[str, Any]:
    """Reload persisted tables; independent verification for ALL cohort metrics.

    Does not call production AGG.aggregate_all / summarize_query_slice / delay_summary
    for the acceptance checks (those live in independent_verify.py).
    """
    ev_path = run_dir / "objective_events.npz"
    q_path = run_dir / "component_queries.npz"
    with np.load(ev_path, allow_pickle=False) as z:
        ev = {k: np.asarray(z[k]) for k in z.files}
    with np.load(q_path, allow_pickle=False) as z:
        query = {k: np.asarray(z[k]) for k in z.files}

    support = IV.verify_events_and_query_support(
        components, ev, query, expect_full=expect_full_cohorts
    )
    indep = IV.verify_aggregates_independent(components, query, aggregates)
    primary = aggregates["cohorts"][C.COHORT_FULL_ISOL][C.INTERVAL_UNCAPED]["any_objective"]

    checks: dict[str, Any] = {
        "n_events": int(ev["t_ms"].shape[0]),
        "n_components_query": int(components["L"].shape[0]),
        "events_match_local_max": int(ev["match_local"].max()) if ev["match_local"].size else -1,
        "n_acq_confirmed_events": support["n_acq_confirmed"],
        "n_acq_neutral_events": support["n_acq_neutral"],
        "n_acq_unknown_events": support["n_acq_unknown"],
        "n_acq_contradictory_events": support["n_acq_contradictory"],
        "support_checks": support,
        "independent_aggregate_checks": indep,
        "cohort_full_window": int(
            aggregates["cohort_sizes"][C.COHORT_FULL_WINDOW]["n_components"]
        ),
        "cohort_full_isol": int(
            aggregates["cohort_sizes"][C.COHORT_FULL_ISOL]["n_components"]
        ),
        "full_window_ok": support["full_window_count_ok"],
        "full_isol_ok": support["full_isol_count_ok"],
        "strict_after_L_ok": support["strict_after_L_ok"],
        "nonneg_delay_ok": support["nonneg_delay_ok"],
        "reload_n_obs": int(primary["n_observed_acquisition"]),
        "reload_mean_s": float(primary["delays"]["event_weighted"]["mean_s"]),
        "reload_median_s": float(primary["delays"]["event_weighted"]["median_s"]),
        "command": (
            "python -m studies.objective_timing_20261002 "
            "--reaggregate-from <source_run> --run-dir <this_run>"
        ),
        "all_ok": bool(support["all_ok"] and indep["all_ok"]),
        "saved_primary": {
            "n_observed_acquisition": primary["n_observed_acquisition"],
            "n_no_acquisition": primary["n_no_acquisition"],
            "n_unknown_observation": primary["n_unknown_observation"],
            "n_neutral_in_interval": primary.get("n_neutral_in_interval"),
            "event_mean_s": primary["delays"]["event_weighted"]["mean_s"],
            "event_median_s": primary["delays"]["event_weighted"]["median_s"],
            "threshold_coverage_cohort": primary.get("threshold_coverage_cohort"),
        },
    }
    # Back-compat keys used by older report readers
    checks["reload_n_obs_ok"] = True
    checks["reload_mean_ok"] = bool(indep["all_ok"])
    checks["reload_median_ok"] = bool(indep["all_ok"])
    checks["reload_eq_mean_ok"] = bool(indep["all_ok"])
    return checks


def run(
    run_dir: Path | None = None,
    max_matches: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    run_dir = Path(run_dir) if run_dir else C.new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    progress_path = run_dir / "progress.jsonl"

    def log(msg: str) -> None:
        line = f"{_utc()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"start run_dir={run_dir}")
    np.random.seed(C.SEED)

    man_checks = IO.verify_anchor_manifest()
    IO.write_json(run_dir / "anchor_manifest_checks.json", man_checks)
    if not man_checks["all_ok"]:
        raise RuntimeError(f"anchor manifest gates failed: {man_checks}")

    components_sha_before = man_checks["components_sha256"]
    terminals_sha_before = man_checks["terminals_sha256"]

    match_ids = IO.load_match_ids()
    components = IO.load_components()
    blue_pids = IO.load_meta_blue_pids()
    n_matches = len(match_ids)
    n_comp = int(components["L"].shape[0])
    starts_all, ends_all = _match_ranges(components["match_local"], n_matches)

    if max_matches is not None:
        n_use = min(int(max_matches), n_matches)
    else:
        n_use = n_matches

    n_workers = int(workers) if workers is not None else C.MAX_WORKER_PROCESSES
    n_workers = max(1, min(n_workers, C.MAX_WORKER_PROCESSES, C.CPU_BUDGET - 1 if C.CPU_BUDGET > 1 else 1))

    # Component mask for used matches
    used_ml = np.arange(n_use, dtype=np.int64)
    comp_keep = components["match_local"] < n_use
    # For smoke, still store full-size query arrays only for used components compacted
    # Full run uses all components.
    if n_use < n_matches:
        # Compact components to used matches for smoke outputs
        idx = np.flatnonzero(comp_keep)
        comp_used = {k: v[idx] for k, v in components.items()}
        # Remap batch placement via local index 0..len(idx)
        global_to_local = np.full(n_comp, -1, dtype=np.int64)
        global_to_local[idx] = np.arange(idx.size, dtype=np.int64)
        n_comp_out = int(idx.size)
    else:
        comp_used = components
        global_to_local = np.arange(n_comp, dtype=np.int64)
        n_comp_out = n_comp
        idx = np.arange(n_comp, dtype=np.int64)

    query_full = _allocate_full_query(n_comp_out)
    event_parts: list[dict[str, np.ndarray]] = []
    census_global: dict[str, Any] = {
        "n_matches": 0,
        "n_missing_events": 0,
        "n_parse_errors": 0,
        "n_events_total": 0,
        "n_elite_monster_kill": 0,
        "n_building_kill": 0,
        "n_malformed": 0,
        "n_invalid_objective_timestamp": 0,
        "n_support_unknown_matches": 0,
        "n_acq_confirmed": 0,
        "n_acq_neutral": 0,
        "n_acq_unknown": 0,
        "n_acq_contradictory": 0,
        "n_after_last_frame": 0,
        "malformed_reasons": Counter(),
        "monster_type_counts": Counter(),
        "monster_subtype_counts": Counter(),
        "building_type_counts": Counter(),
        "tower_type_counts": Counter(),
        "team_source_counts": Counter(),
        "contamination_uncapped_any_objective": Counter(),
    }
    all_errors: list[dict[str, Any]] = []
    total_queries = 0
    query_wall = 0.0
    peak_tree = 0
    peak_parent = 0

    # Build batches of match locals
    batches: list[list[int]] = []
    cur: list[int] = []
    for ml in range(n_use):
        cur.append(ml)
        if len(cur) >= C.BATCH_SIZE_MATCHES:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    log(f"n_use_matches={n_use} n_comp_out={n_comp_out} n_batches={len(batches)} workers={n_workers}")

    def _make_payload(batch_mls: list[int]) -> dict[str, Any]:
        # Gather component rows for these matches in order of match_local
        slices = []
        comp_starts = []
        comp_ends = []
        cursor = 0
        for ml in batch_mls:
            s = int(starts_all[ml])
            e = int(ends_all[ml])
            slices.append(np.arange(s, e, dtype=np.int64))
            n = e - s
            comp_starts.append(cursor)
            comp_ends.append(cursor + n)
            cursor += n
        if slices:
            gidx = np.concatenate(slices) if slices else np.zeros(0, dtype=np.int64)
        else:
            gidx = np.zeros(0, dtype=np.int64)
        return {
            "match_locals": batch_mls,
            "match_ids": [match_ids[ml] for ml in batch_mls],
            "blue_pids": blue_pids,
            "comp_starts": np.asarray(comp_starts, dtype=np.int64),
            "comp_ends": np.asarray(comp_ends, dtype=np.int64),
            "L": components["L"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "game_end_ms": components["game_end_ms"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "game_end_known": components["game_end_known"][gidx] if gidx.size else np.zeros(0, dtype=np.int8),
            "last_frame_ms": components["last_frame_ms"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "next_kill_after_L": components["next_kill_after_L"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "next_component_tau": components["next_component_tau"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "component_id": components["component_id"][gidx] if gidx.size else np.zeros(0, dtype=np.int64),
            "global_idx": gidx,
        }

    def _absorb(result: dict[str, Any], payload: dict[str, Any]) -> None:
        nonlocal total_queries, query_wall, peak_tree, peak_parent
        total_queries += int(result["n_queries"])
        query_wall += float(result["wall_s"])
        all_errors.extend(result.get("errors") or [])
        cen = result["census"]
        for k in (
            "n_matches",
            "n_missing_events",
            "n_parse_errors",
            "n_events_total",
            "n_elite_monster_kill",
            "n_building_kill",
            "n_malformed",
            "n_invalid_objective_timestamp",
            "n_support_unknown_matches",
            "n_acq_confirmed",
            "n_acq_neutral",
            "n_acq_unknown",
            "n_acq_contradictory",
            "n_after_last_frame",
        ):
            census_global[k] += int(cen.get(k, 0))
        for ck in (
            "malformed_reasons",
            "monster_type_counts",
            "monster_subtype_counts",
            "building_type_counts",
            "tower_type_counts",
            "team_source_counts",
            "contamination_uncapped_any_objective",
        ):
            census_global[ck].update(cen.get(ck, {}))

        ev = result["events"]
        if int(ev["t_ms"].size):
            event_parts.append(ev)

        gidx = np.asarray(payload["global_idx"], dtype=np.int64)
        if gidx.size == 0:
            return
        local_idx = global_to_local[gidx]
        q = result["query"]
        # Map batch component positions -> output positions
        # Batch arrays are ordered as concatenated per-match slices = gidx order
        for key, block in q.items():
            for field, vals in block.items():
                query_full[key][field][local_idx] = vals

        rss = _process_tree_rss_bytes()
        if rss["parent_rss_bytes"]:
            peak_parent = max(peak_parent, int(rss["parent_rss_bytes"]))
        if rss["process_tree_rss_bytes"]:
            peak_tree = max(peak_tree, int(rss["process_tree_rss_bytes"]))

    import json as _json

    def _write_progress(bi: int, n_batch: int, wall_s: float) -> None:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps({"batch": bi, "n_matches": n_batch, "wall_s": wall_s}) + "\n")

    # Execute
    t_query0 = time.perf_counter()
    if n_workers <= 1:
        for bi, batch in enumerate(batches):
            payload = _make_payload(batch)
            result = _process_match_batch(payload)
            _absorb(result, payload)
            _write_progress(bi, len(batch), float(result["wall_s"]))
            if bi % 5 == 0 or bi + 1 == len(batches):
                log(f"batch {bi+1}/{len(batches)} done")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {}
            for bi, batch in enumerate(batches):
                payload = _make_payload(batch)
                fut = ex.submit(_process_match_batch, payload)
                futs[fut] = (bi, payload)
            done_n = 0
            for fut in as_completed(futs):
                bi, payload = futs[fut]
                result = fut.result()
                _absorb(result, payload)
                done_n += 1
                _write_progress(bi, len(payload["match_locals"]), float(result["wall_s"]))
                if done_n % 5 == 0 or done_n == len(batches):
                    log(f"batches_done {done_n}/{len(batches)}")

    query_wall_total = float(time.perf_counter() - t_query0)
    log(f"query_pass_wall_s={query_wall_total:.3f} n_queries={total_queries}")

    # Persist events
    if event_parts:
        events = {k: np.concatenate([p[k] for p in event_parts]) for k in event_parts[0]}
    else:
        events = {
            "match_local": np.zeros(0, dtype=np.int32),
            "t_ms": np.zeros(0, dtype=np.int64),
            "event_ord": np.zeros(0, dtype=np.int64),
            "category_mask": np.zeros(0, dtype=np.int64),
            "team": np.zeros(0, dtype=np.int16),
            "acq_status": np.zeros(0, dtype=np.int8),
            "after_last_frame": np.zeros(0, dtype=np.int8),
            "raw_killerTeamId": np.zeros(0, dtype=np.int32),
            "raw_killerId": np.zeros(0, dtype=np.int32),
            "raw_building_owner": np.zeros(0, dtype=np.int32),
            "leaf_code": np.zeros(0, dtype=np.int16),
        }
    # Sort events by match_local, t, ord for stable reload
    order = np.lexsort((events["event_ord"], events["t_ms"], events["match_local"]))
    events = {k: v[order] for k, v in events.items()}
    ev_path = run_dir / "objective_events.npz"
    np.savez_compressed(ev_path, **events)
    ev_sha = IO.file_sha256(ev_path)

    # Persist queries as flat keys
    q_flat: dict[str, np.ndarray] = {}
    for key, block in query_full.items():
        for field, vals in block.items():
            q_flat[f"{field}__{key}"] = vals
    # Also store component identity for join
    q_flat["match_local"] = comp_used["match_local"]
    q_flat["component_id"] = comp_used["component_id"]
    q_flat["L"] = comp_used["L"]
    q_path = run_dir / "component_queries.npz"
    np.savez_compressed(q_path, **q_flat)
    q_sha = IO.file_sha256(q_path)

    # Aggregate
    aggregates = AGG.aggregate_all(comp_used, q_flat)
    agg_sha = IO.write_json(run_dir / "aggregates.json", aggregates)

    census_out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in census_global.items()}
    census_sha = IO.write_json(run_dir / "schema_census.json", census_out)

    # Source immutability
    components_sha_after = IO.file_sha256(C.COMPONENTS_NPZ)
    terminals_sha_after = IO.file_sha256(C.TERMINALS_JSON)

    reload = reload_verify(
        run_dir, comp_used, aggregates, expect_full_cohorts=(n_use == n_matches)
    )
    IO.write_json(run_dir / "reload_verification.json", reload)

    # Gates
    zero_kill_ok = True
    if n_use == n_matches:
        # matches with zero components
        n_zero = int((ends_all - starts_all == 0).sum())
        zero_kill_ok = n_zero == C.EXPECTED_ZERO_KILL
    else:
        n_zero = int(((ends_all[:n_use] - starts_all[:n_use]) == 0).sum())

    full_corpus = n_use == n_matches
    n_full_window = int(aggregates["cohort_sizes"][C.COHORT_FULL_WINDOW]["n_components"])
    n_full_isol = int(aggregates["cohort_sizes"][C.COHORT_FULL_ISOL]["n_components"])
    gates = {
        "anchor_manifest_ok": bool(man_checks["all_ok"]),
        "components_immutable": components_sha_before == components_sha_after,
        "terminals_immutable": terminals_sha_before == terminals_sha_after,
        "n_matches_processed": n_use,
        "n_matches_expected_full": C.EXPECTED_TRAIN_MATCHES,
        "full_corpus": full_corpus,
        "n_components_out": n_comp_out,
        "n_components_ok": (n_comp_out == C.EXPECTED_MIN1_COMPONENTS) if full_corpus else True,
        "zero_kill_matches": n_zero,
        "zero_kill_ok": zero_kill_ok if full_corpus else True,
        "n_full_window": n_full_window,
        "n_full_isol": n_full_isol,
        "full_window_ok": (n_full_window == C.EXPECTED_FULL_WINDOW) if full_corpus else True,
        "full_isol_ok": (n_full_isol == C.EXPECTED_FULL_ISOL) if full_corpus else True,
        "reload_ok": bool(reload.get("all_ok")),
        "n_errors": len(all_errors),
        "no_errors": len(all_errors) == 0,
        "n_missing_events": int(census_out["n_missing_events"]),
        "no_missing_events": int(census_out["n_missing_events"]) == 0,
        "n_support_unknown_matches": int(census_out.get("n_support_unknown_matches", 0)),
        "no_support_unknown": int(census_out.get("n_support_unknown_matches", 0)) == 0,
        "n_invalid_objective_timestamp": int(
            census_out.get("n_invalid_objective_timestamp", 0)
        ),
        "no_invalid_objective_timestamp": int(
            census_out.get("n_invalid_objective_timestamp", 0)
        )
        == 0,
    }
    gates["all_ok"] = bool(
        gates["anchor_manifest_ok"]
        and gates["components_immutable"]
        and gates["terminals_immutable"]
        and gates["n_components_ok"]
        and gates["zero_kill_ok"]
        and gates["full_window_ok"]
        and gates["full_isol_ok"]
        and gates["reload_ok"]
        and gates["no_errors"]
        and gates["no_missing_events"]
        and gates["no_support_unknown"]
        and gates["no_invalid_objective_timestamp"]
        and gates["full_corpus"]
    )

    rss = _process_tree_rss_bytes()
    manifest = {
        "study": "objective_timing_20261002",
        "complete": bool(gates["all_ok"]),
        "run_dir": str(run_dir),
        "started": True,
        "seed": C.SEED,
        "max_matches": max_matches,
        "n_matches": n_use,
        "n_components": n_comp_out,
        "n_queries": total_queries,
        "query_wall_s": query_wall_total,
        "worker_wall_sum_s": query_wall,
        "wall_s": float(time.perf_counter() - t_start),
        "workers": n_workers,
        "environment": IO.environment_record(),
        "machine": {
            "peak_parent_rss_bytes": peak_parent or rss.get("parent_rss_bytes"),
            "peak_process_tree_rss_bytes": peak_tree or rss.get("process_tree_rss_bytes"),
            "peak_rss_bytes_note": "process-tree via psutil when available",
        },
        "source": {
            "anchor_manifest_sha256": man_checks["sha256"],
            "components_sha256": components_sha_after,
            "terminals_sha256": terminals_sha_after,
        },
        "artifacts": {
            "objective_events_npz": str(ev_path),
            "objective_events_sha256": ev_sha,
            "component_queries_npz": str(q_path),
            "component_queries_sha256": q_sha,
            "aggregates_sha256": agg_sha,
            "schema_census_sha256": census_sha,
        },
        "census": {
            "n_objective_events": int(events["t_ms"].size),
            "n_acq_confirmed": int(census_out["n_acq_confirmed"]),
            "n_acq_neutral": int(census_out["n_acq_neutral"]),
            "n_acq_unknown": int(census_out["n_acq_unknown"]),
            "n_acq_contradictory": int(census_out["n_acq_contradictory"]),
        },
        "gates": gates,
        "reload_verification": reload,
        "errors_head": all_errors[:20],
        "finished_at": _utc(),
    }
    IO.write_json(run_dir / "RUN_MANIFEST.json", manifest)
    IO.write_json(run_dir / "status.json", {"complete": manifest["complete"], "run_dir": str(run_dir)})
    log(f"complete={manifest['complete']} gates_all_ok={gates['all_ok']}")
    return manifest


def _verify_source_events_schema_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """Read-only payload/schema check for a batch of match_ids (no query)."""
    from . import thread_limits as _tl  # noqa: F401
    from .parse import parse_timestamp_ms

    out = {
        "n_matches": 0,
        "n_missing": 0,
        "n_payload_errors": 0,
        "n_invalid_objective_timestamp": 0,
        "n_events_total": 0,
        "n_elite": 0,
        "n_building": 0,
        "errors": [],
        "game_end_known_ok": True,
    }
    for mid in payload["match_ids"]:
        out["n_matches"] += 1
        events, err = IO.load_events(str(mid))
        if err:
            if err == "missing_events_json":
                out["n_missing"] += 1
            else:
                out["n_payload_errors"] += 1
            out["errors"].append({"match_id_hash8": None, "error": err})
            continue
        out["n_events_total"] += len(events)
        for e in events:
            if not isinstance(e, dict):
                continue
            typ = str(e.get("type") or "")
            if typ not in ("ELITE_MONSTER_KILL", "BUILDING_KILL"):
                continue
            if typ == "ELITE_MONSTER_KILL":
                out["n_elite"] += 1
            else:
                out["n_building"] += 1
            if "timestamp" not in e:
                out["n_invalid_objective_timestamp"] += 1
                continue
            ts, reason = parse_timestamp_ms(e["timestamp"])
            if reason or ts is None:
                out["n_invalid_objective_timestamp"] += 1
    return out


def verify_full_source_schema(
    match_ids: list[str],
    *,
    workers: int | None = None,
    log=None,
) -> dict[str, Any]:
    """Full read-only schema verification of all events.json payloads."""
    n_workers = int(workers) if workers is not None else C.MAX_WORKER_PROCESSES
    n_workers = max(1, min(n_workers, C.MAX_WORKER_PROCESSES, C.CPU_BUDGET - 1 if C.CPU_BUDGET > 1 else 1))
    batches: list[list[str]] = []
    cur: list[str] = []
    for mid in match_ids:
        cur.append(mid)
        if len(cur) >= C.BATCH_SIZE_MATCHES:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    acc = {
        "n_matches": 0,
        "n_missing": 0,
        "n_payload_errors": 0,
        "n_invalid_objective_timestamp": 0,
        "n_events_total": 0,
        "n_elite": 0,
        "n_building": 0,
        "errors_head": [],
    }
    t0 = time.perf_counter()

    def absorb(r: dict[str, Any]) -> None:
        for k in (
            "n_matches",
            "n_missing",
            "n_payload_errors",
            "n_invalid_objective_timestamp",
            "n_events_total",
            "n_elite",
            "n_building",
        ):
            acc[k] += int(r.get(k, 0))
        if len(acc["errors_head"]) < 20:
            acc["errors_head"].extend(r.get("errors") or [])

    if n_workers <= 1:
        for bi, batch in enumerate(batches):
            absorb(_verify_source_events_schema_batch({"match_ids": batch}))
            if log and (bi % 20 == 0 or bi + 1 == len(batches)):
                log(f"schema_batch {bi+1}/{len(batches)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(_verify_source_events_schema_batch, {"match_ids": batch})
                for batch in batches
            ]
            done = 0
            for fut in as_completed(futs):
                absorb(fut.result())
                done += 1
                if log and (done % 20 == 0 or done == len(batches)):
                    log(f"schema_batches_done {done}/{len(batches)}")

    acc["wall_s"] = float(time.perf_counter() - t0)
    acc["all_ok"] = bool(
        acc["n_missing"] == 0
        and acc["n_payload_errors"] == 0
        and acc["n_invalid_objective_timestamp"] == 0
        and acc["n_matches"] == len(match_ids)
    )
    acc["n_matches_expected"] = len(match_ids)
    return acc


def reaggregate_from(
    source_run: Path,
    run_dir: Path | None = None,
    *,
    workers: int | None = None,
    skip_source_schema: bool = False,
) -> dict[str, Any]:
    """Reuse immutable objective_events.npz; recompute queries + corrected aggregates."""
    t_start = time.perf_counter()
    source_run = Path(source_run)
    run_dir = Path(run_dir) if run_dir else C.new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        line = f"{_utc()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"reaggregate_from source={source_run} run_dir={run_dir}")
    man_checks = IO.verify_anchor_manifest()
    IO.write_json(run_dir / "anchor_manifest_checks.json", man_checks)
    if not man_checks["all_ok"]:
        raise RuntimeError(f"anchor manifest gates failed: {man_checks}")

    # Fail closed before copy/requery on incomplete, smoke, or tampered sources.
    reuse_checks = IO.verify_reusable_source_run(source_run)
    IO.write_json(run_dir / "reuse_source_identity_checks.json", reuse_checks)
    if not reuse_checks.get("all_ok"):
        raise RuntimeError(
            "reusable source identity gate failed before copy/requery: "
            f"{reuse_checks.get('failures')}"
        )

    match_ids = IO.load_match_ids()
    components = IO.load_components()
    n_matches = len(match_ids)
    n_comp = int(components["L"].shape[0])
    starts_all, ends_all = _match_ranges(components["match_local"], n_matches)

    src_ev = source_run / "objective_events.npz"
    src_man = source_run / "RUN_MANIFEST.json"
    src_census = source_run / "schema_census.json"
    src_ev_sha = str(reuse_checks["objective_events_sha256_actual"])
    src_man_obj = IO.load_json(src_man)
    src_census_obj = IO.load_json(src_census) if src_census.is_file() else {}

    provenance = {
        "mode": "reaggregate_from",
        "source_run": str(source_run),
        "source_objective_events_sha256": src_ev_sha,
        "source_manifest_complete": bool(src_man_obj.get("complete")),
        "source_n_matches": src_man_obj.get("n_matches"),
        "source_n_components": src_man_obj.get("n_components"),
        "reuse_source_identity_checks": reuse_checks,
        "source_census_acq": src_man_obj.get("census") or {
            k: src_census_obj.get(k)
            for k in (
                "n_acq_confirmed",
                "n_acq_neutral",
                "n_acq_unknown",
                "n_acq_contradictory",
            )
        },
        "anchor_checks": {
            "components_sha256_ok": man_checks.get("components_sha256_ok"),
            "terminals_sha256_ok": man_checks.get("terminals_sha256_ok"),
            "match_ids_sha256_ok": man_checks.get("match_ids_sha256_ok"),
            "match_meta_sha256_ok": man_checks.get("match_meta_sha256_ok"),
        },
    }
    IO.write_json(run_dir / "reaggregate_provenance.json", provenance)

    # Reference/copy events bytes into new run without modifying source
    import shutil

    dst_ev = run_dir / "objective_events.npz"
    if not dst_ev.exists():
        shutil.copy2(src_ev, dst_ev)
    if IO.file_sha256(dst_ev) != src_ev_sha:
        raise RuntimeError("copied objective_events.npz hash mismatch")

    with np.load(dst_ev, allow_pickle=False) as z:
        events = {k: np.asarray(z[k]) for k in z.files}

    schema_verify: dict[str, Any]
    if skip_source_schema:
        schema_verify = {
            "skipped": True,
            "all_ok": False,
            "reason": "skip_source_schema requested",
        }
    else:
        log("full read-only source events schema verification start")
        schema_verify = verify_full_source_schema(match_ids, workers=workers, log=log)
        log(
            f"schema_verify all_ok={schema_verify['all_ok']} "
            f"payload_errors={schema_verify['n_payload_errors']} "
            f"bad_ts={schema_verify['n_invalid_objective_timestamp']} "
            f"wall_s={schema_verify['wall_s']:.2f}"
        )
    IO.write_json(run_dir / "source_schema_verification.json", schema_verify)

    # Recompute queries from immutable event table
    n_workers = int(workers) if workers is not None else C.MAX_WORKER_PROCESSES
    n_workers = max(1, min(n_workers, C.MAX_WORKER_PROCESSES, C.CPU_BUDGET - 1 if C.CPU_BUDGET > 1 else 1))
    query_full = _allocate_full_query(n_comp)
    total_queries = 0
    t_q0 = time.perf_counter()

    # Group event indices by match_local
    ml_ev = events["match_local"].astype(np.int64)
    order = np.argsort(ml_ev, kind="mergesort")
    ml_sorted = ml_ev[order]
    # Build per-match slices into sorted events
    if ml_sorted.size:
        uniq, start_idx, counts = np.unique(ml_sorted, return_index=True, return_counts=True)
        ev_starts = {int(u): int(s) for u, s in zip(uniq, start_idx)}
        ev_counts = {int(u): int(c) for u, c in zip(uniq, counts)}
    else:
        ev_starts, ev_counts = {}, {}

    contamination: Counter = Counter()
    log(f"requery n_comp={n_comp} n_matches={n_matches} workers={n_workers}")

    # Single-process requery is fine and avoids pickling large arrays; CPU budget ok.
    for ml in range(n_matches):
        s = int(starts_all[ml])
        e = int(ends_all[ml])
        if e <= s:
            continue
        es = ev_starts.get(ml)
        if es is None:
            arr = {
                "t_ms": np.zeros(0, dtype=np.int64),
                "event_ord": np.zeros(0, dtype=np.int64),
                "category_mask": np.zeros(0, dtype=np.int64),
                "team": np.zeros(0, dtype=np.int16),
                "after_last_frame": np.zeros(0, dtype=np.int8),
                "acq_status": np.zeros(0, dtype=np.int8),
            }
        else:
            sl = order[es : es + ev_counts[ml]]
            # Events already sorted globally by match,t,ord in source; re-sort locally
            sub = {k: events[k][sl] for k in (
                "t_ms",
                "event_ord",
                "category_mask",
                "team",
                "after_last_frame",
                "acq_status",
            )}
            loc = np.lexsort((sub["event_ord"], sub["t_ms"]))
            arr = {k: v[loc] for k, v in sub.items()}

        L_m = components["L"][s:e]
        nk_m = components["next_kill_after_L"][s:e]
        nt_m = components["next_component_tau"][s:e]
        game_end = int(components["game_end_ms"][s])
        game_end_known = bool(components["game_end_known"][s])
        q = query_match_components_vectorized(
            arr["t_ms"],
            arr["event_ord"],
            arr["category_mask"],
            arr["team"],
            arr["after_last_frame"],
            arr["acq_status"],
            L_m,
            game_end,
            game_end_known,
            nk_m,
            nt_m,
        )
        total_queries += int(L_m.size) * len(C.CATEGORY_ORDER) * 2
        for key, block in q.items():
            for field, vals in block.items():
                query_full[key][field][s:e] = vals
        st = q[f"{C.INTERVAL_UNCAPED}:any_objective"]["status"]
        tms = q[f"{C.INTERVAL_UNCAPED}:any_objective"]["t_ms"]
        for j in range(L_m.size):
            contamination[
                classify_uncapped_vs_next_anchor(
                    int(tms[j]),
                    int(st[j]),
                    int(L_m[j]),
                    int(nk_m[j]),
                    int(nt_m[j]),
                    game_end,
                    game_end_known,
                )
            ] += 1
        if ml % 5000 == 0:
            log(f"requery_match {ml}/{n_matches}")

    query_wall = float(time.perf_counter() - t_q0)
    log(f"requery_done wall_s={query_wall:.3f} n_queries={total_queries}")

    q_flat: dict[str, np.ndarray] = {}
    for key, block in query_full.items():
        for field, vals in block.items():
            q_flat[f"{field}__{key}"] = vals
    q_flat["match_local"] = components["match_local"]
    q_flat["component_id"] = components["component_id"]
    q_flat["L"] = components["L"]
    q_path = run_dir / "component_queries.npz"
    np.savez_compressed(q_path, **q_flat)
    q_sha = IO.file_sha256(q_path)

    aggregates = AGG.aggregate_all(components, q_flat)
    agg_sha = IO.write_json(run_dir / "aggregates.json", aggregates)

    census_out = dict(src_census_obj) if src_census_obj else {}
    census_out["contamination_uncapped_any_objective"] = dict(contamination)
    census_out["reaggregate_note"] = (
        "Event/acq census reused from source run after full schema verification; "
        "queries recomputed with corrected neutral/uncertainty semantics."
    )
    census_out["n_acq_confirmed"] = int((events["acq_status"] == C.ACQ_CONFIRMED).sum())
    census_out["n_acq_neutral"] = int((events["acq_status"] == C.ACQ_NEUTRAL).sum())
    census_out["n_acq_unknown"] = int((events["acq_status"] == C.ACQ_UNKNOWN).sum())
    census_out["n_acq_contradictory"] = int(
        (events["acq_status"] == C.ACQ_CONTRADICTORY).sum()
    )
    census_sha = IO.write_json(run_dir / "schema_census.json", census_out)

    reload = reload_verify(run_dir, components, aggregates, expect_full_cohorts=True)
    IO.write_json(run_dir / "reload_verification.json", reload)

    n_zero = int((ends_all - starts_all == 0).sum())
    n_full_window = int(aggregates["cohort_sizes"][C.COHORT_FULL_WINDOW]["n_components"])
    n_full_isol = int(aggregates["cohort_sizes"][C.COHORT_FULL_ISOL]["n_components"])
    components_sha = IO.file_sha256(C.COMPONENTS_NPZ)
    terminals_sha = IO.file_sha256(C.TERMINALS_JSON)

    gates = {
        "anchor_manifest_ok": bool(man_checks["all_ok"]),
        "components_immutable": components_sha == C.EXPECTED_COMPONENTS_SHA256,
        "terminals_immutable": terminals_sha == C.EXPECTED_TERMINALS_SHA256,
        "source_events_immutable": IO.file_sha256(dst_ev) == src_ev_sha,
        "source_schema_ok": bool(schema_verify.get("all_ok")),
        "n_matches_processed": n_matches,
        "full_corpus": True,
        "n_components_out": n_comp,
        "n_components_ok": n_comp == C.EXPECTED_MIN1_COMPONENTS,
        "zero_kill_matches": n_zero,
        "zero_kill_ok": n_zero == C.EXPECTED_ZERO_KILL,
        "n_full_window": n_full_window,
        "n_full_isol": n_full_isol,
        "full_window_ok": n_full_window == C.EXPECTED_FULL_WINDOW,
        "full_isol_ok": n_full_isol == C.EXPECTED_FULL_ISOL,
        "reload_ok": bool(reload.get("all_ok")),
        "n_acq_neutral_ok": int(census_out["n_acq_neutral"]) == 524,
        "n_acq_unknown_ok": int(census_out["n_acq_unknown"]) == 0,
        "n_acq_contradictory_ok": int(census_out["n_acq_contradictory"]) == 0,
    }
    gates["all_ok"] = bool(
        gates["anchor_manifest_ok"]
        and gates["components_immutable"]
        and gates["terminals_immutable"]
        and gates["source_events_immutable"]
        and gates["source_schema_ok"]
        and gates["n_components_ok"]
        and gates["zero_kill_ok"]
        and gates["full_window_ok"]
        and gates["full_isol_ok"]
        and gates["reload_ok"]
        and gates["n_acq_neutral_ok"]
        and gates["n_acq_unknown_ok"]
        and gates["n_acq_contradictory_ok"]
        and gates["full_corpus"]
    )

    wall_s = float(time.perf_counter() - t_start)
    manifest = {
        "study": "objective_timing_20261002",
        "mode": "reaggregate_from",
        "complete": bool(gates["all_ok"]),
        "run_dir": str(run_dir),
        "source_run": str(source_run),
        "started": True,
        "seed": C.SEED,
        "n_matches": n_matches,
        "n_components": n_comp,
        "n_queries": total_queries,
        "query_wall_s": query_wall,
        "wall_s": wall_s,
        "correction_wall_s": wall_s,
        "workers": n_workers,
        "environment": IO.environment_record(),
        "source": {
            "anchor_manifest_sha256": man_checks["sha256"],
            "components_sha256": components_sha,
            "terminals_sha256": terminals_sha,
            "match_ids_sha256": man_checks["match_ids_sha256"],
            "match_meta_sha256": man_checks["match_meta_sha256"],
            "reused_objective_events_sha256": src_ev_sha,
        },
        "artifacts": {
            "objective_events_npz": str(dst_ev),
            "objective_events_sha256": IO.file_sha256(dst_ev),
            "component_queries_npz": str(q_path),
            "component_queries_sha256": q_sha,
            "aggregates_sha256": agg_sha,
            "schema_census_sha256": census_sha,
        },
        "census": {
            "n_objective_events": int(events["t_ms"].size),
            "n_acq_confirmed": int(census_out["n_acq_confirmed"]),
            "n_acq_neutral": int(census_out["n_acq_neutral"]),
            "n_acq_unknown": int(census_out["n_acq_unknown"]),
            "n_acq_contradictory": int(census_out["n_acq_contradictory"]),
        },
        "gates": gates,
        "reload_verification": reload,
        "source_schema_verification": schema_verify,
        "provenance": provenance,
        "finished_at": _utc(),
    }
    IO.write_json(run_dir / "RUN_MANIFEST.json", manifest)
    IO.write_json(
        run_dir / "status.json",
        {"complete": manifest["complete"], "run_dir": str(run_dir), "mode": "reaggregate_from"},
    )
    log(f"complete={manifest['complete']} gates_all_ok={gates['all_ok']} wall_s={wall_s:.2f}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Objective timing 20261002")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--reaggregate-from",
        type=Path,
        default=None,
        help="Reuse immutable objective_events from a prior complete run",
    )
    p.add_argument(
        "--skip-source-schema",
        action="store_true",
        help="Skip full read-only events.json schema verification (not for acceptance)",
    )
    args = p.parse_args(argv)
    if args.reaggregate_from is not None:
        man = reaggregate_from(
            args.reaggregate_from,
            run_dir=args.run_dir,
            workers=args.workers,
            skip_source_schema=args.skip_source_schema,
        )
    else:
        man = run(run_dir=args.run_dir, max_matches=args.max_matches, workers=args.workers)
    print({"complete": man.get("complete"), "run_dir": man.get("run_dir")})
    if args.max_matches is not None:
        return 0
    return 0 if man.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
