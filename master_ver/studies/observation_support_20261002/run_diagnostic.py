"""Full-corpus observation-support diagnostic runner (CPU<=4, CACHE-ONLY if raw absent)."""
from __future__ import annotations

from . import thread_limits as TL  # noqa: F401

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import aggregate as AG
from . import config as C
from . import reference_replay as REF
from . import storage as STORE
from .io_source import (
    build_full_labels,
    census_raw_source_availability,
    component_match_ranges,
    file_sha256,
    load_baseline_components,
    load_ledger_and_ranges,
    load_match_ids,
    load_meta_blue_pids,
    verify_baseline_identity,
    write_json_atomic,
)


FIXED_INT_FIELDS = [
    "match_local",
    "component_id",
    "n_members",
    "same_ms_first_n",
    "f",
    "L",
    "tau",
    "t_in",
    "c0_bits",
    "call_bits",
    "c0_n",
    "call_n",
    "c0_n_env_killer",
    "call_n_env_killer",
    "c0_n_blue",
    "c0_n_red",
    "c0_n_unknown_team",
    "c0_scale_max_side",
    "call_n_blue",
    "call_n_red",
    "call_scale_max_side",
    "F_tau",
    "F_tau_idx",
    "lag_f_F_tau",
    "F_tm1",
    "F_tm1_idx",
    "lag_f_F_tm1",
    "a_strict",
    "ev_at_tau",
    "ev_between_Ftau_tau",
    "ev_between_Ftm1_tin",
    "ev_types_at_tau_n",
    "foreign_kills_in_tau_L",
    "other_anchors_in_tau_L",
    "call_n_unknown_team", "c0_n_credited", "call_n_credited", "foreign_near", "foreign_distant",
    "last_frame_ms", "support_start_ms", "q_pre", "game_end_ms", "next_component_tau", "next_kill_after_L",
]
FIXED_BOOL_FIELDS = [
    "singleton",
    "team_map_ok",
    "cand_10_20_tau",
    "cand_10_20_tm1",
    "exact_10000_tau",
    "strict_eligible",
    "frame_differs",
    "raw_presence_verified",
    "radius_mono_p_lower_tau",
    "radius_mono_p_upper_tau",
    "events_support_ok",
    "game_end_known", "overlap_next_tau_le_L", "missing_terminal_game_end",
    "frame_support_ok", "terminal_support_authoritative",
]
PART_TAGS = [f"{c}_r{int(r)}" for c in ("tau", "tm1") for r in C.DIAGNOSTIC_RADII]
PART_INT = [
    "n_lower",
    "n_upper",
    "p_lower_n",
    "p_upper_n",
    "lower_bits",
    "upper_bits",
    "near_bits",
    "geom_near_bits",
    "geom_credit_only_bits",
    "geom_near_only_bits",
    "geom_intersection_bits",
    "p_lower_bits",
    "p_upper_bits",
    "part_credit_only_lower_bits",
    "part_near_only_lower_bits",
    "part_intersection_lower_bits",
    "part_credit_only_upper_bits",
    "part_near_only_upper_bits",
    "part_intersection_upper_bits",
    "credit_only_bits",
    "near_only_bits",
    "intersection_bits",
]
PART_BOOL = ["support_ok", "lower_le_upper", "c0_subset_lower", "c0_subset_upper"]
PART_FLOAT = ["dist_min", "dist_p50", "dist_p90"]
POST_CUTS = ("tau", "tm1")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_files() -> list[Path]:
    return sorted(C.STUDY.glob("*.py")) + sorted((C.STUDY / "tests").glob("*.py"))


def _code_sha256() -> str:
    h = hashlib.sha256()
    for p in _source_files():
        h.update(p.relative_to(C.STUDY).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _protocol_sha256() -> str:
    return file_sha256(C.PROTOCOL) if C.PROTOCOL.is_file() else ""


def _assignment_versions():
    return {str(p.relative_to(C.ROOT)): file_sha256(p)
            for p in sorted(C.ST_BATCHES.glob("batch_*")) if p.suffix in (".json", ".npz")}


def _final_cache_hashes(row):
    final = {key: file_sha256(C.CACHE_DIR / f"{row['match_id']}{suffix}")
             for key, suffix in (("npz", ".npz"), ("meta", ".meta.json"), ("events", ".events.json"))}
    return {"match_local": int(row["match_local"]), "hashes": final, "ok": final == row["before"]}


def _verify_final_cache_versions(rows, workers):
    results = []
    if workers == 1:
        results = [_final_cache_hashes(row) for row in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for begin in range(0, len(rows), C.BATCH_SIZE_MATCHES):
                results.extend(pool.map(_final_cache_hashes, rows[begin:begin + C.BATCH_SIZE_MATCHES]))
    return {"n_expected": len(rows), "n_checked": len(results), "n_mismatches": sum(not r["ok"] for r in results),
            "all_ok": len(results) == len(rows) and all(r["ok"] for r in results), "rows": results}


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
                continue
        return {"parent_rss": parent_rss, "tree_rss": total}
    except Exception:
        return {"parent_rss": None, "tree_rss": None}


def _allocate(n: int, directory: Path | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    def new(key, fill, dtype):
        return STORE.allocate_array(directory, key, n, dtype, fill)
    for f in FIXED_INT_FIELDS:
        out[f] = new(f, C.SENTINEL_MISSING, np.int64)
    for f in FIXED_BOOL_FIELDS:
        out[f] = new(f, 0, np.int8)
    for key, width in (("anon_match_key", 16), ("member_key", 24), ("tau_reason", 32),
                       ("tm1_reason", 32), ("events_support_reason", 64),
                       ("participation_mode", 24), ("team_map_provenance", 48)):
        out[key] = new(key, "cache_only_proxy" if key == "participation_mode" else "", f"U{width}")
    for tag in PART_TAGS:
        for f in PART_INT:
            default = C.SENTINEL_MISSING if f.startswith(("n_", "p_")) and "bits" not in f else 0
            out[f"{f}_{tag}"] = new(f"{f}_{tag}", default, np.int64)
        for f in PART_BOOL:
            out[f"{f}_{tag}"] = new(f"{f}_{tag}", 0, np.int8)
        for f in PART_FLOAT:
            out[f"{f}_{tag}"] = new(f"{f}_{tag}", np.nan, np.float64)
    for h in C.HORIZONS_S:
        out[f"e_h{h}"] = new(f"e_h{h}", C.SENTINEL_MISSING, np.int64)
        out[f"e_h{h}_recomputed"] = new(f"e_h{h}_recomputed", C.SENTINEL_MISSING, np.int64)
        out[f"valid_h{h}"] = new(f"valid_h{h}", 0, np.int8)
        for cut in POST_CUTS:
            tag = f"{cut}_h{h}"
            for f in (
                "F_post",
                "F_post_idx",
                "age_post",
                "frame_idx_gap",
                "timestamp_gap",
                "async_events_Fpre_to_Fpost",
                "async_events_cutoff_to_endpoint",
            ):
                out[f"{f}_{tag}"] = new(f"{f}_{tag}", C.SENTINEL_MISSING, np.int64)
            for f in ("same_as_Fpre", "F_post_gt_L", "stale_at_or_before_L", "events_support_ok"):
                out[f"{f}_{tag}"] = new(f"{f}_{tag}", 0, np.int8)
            for f in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean"):
                out[f"{f}_{tag}"] = new(f"{f}_{tag}", np.nan, np.float64)
            out[f"post_support_reason_{tag}"] = new(f"post_support_reason_{tag}", "", "U48")
    return out


def _allocate_evidence(n: int, directory: Path | None = None,
                       first_counts: np.ndarray | None = None) -> dict[str, Any]:
    if first_counts is None:
        first_counts = np.ones(n, dtype=np.int64)
    first_counts = np.asarray(first_counts, dtype=np.int64)
    if first_counts.shape != (n,) or np.any(first_counts < 1):
        raise ValueError("every component needs a positive first-kill location count")
    offsets = STORE.allocate_array(directory, "first_loc_offsets", n + 1, np.int64)
    offsets[1:] = np.cumsum(first_counts)
    out = {"first_loc_offsets": offsets,
           "first_loc_x": STORE.allocate_array(directory, "first_loc_x", int(offsets[-1]), np.float64, np.nan),
           "first_loc_y": STORE.allocate_array(directory, "first_loc_y", int(offsets[-1]), np.float64, np.nan)}
    for cut in ("tau", "tm1"):
        for radius in C.DIAGNOSTIC_RADII:
            tag = f"{cut}_r{int(radius)}"
            out[f"dist_{tag}"] = STORE.allocate_array(directory, f"dist_{tag}", (n, 10), np.float64, np.nan)
            for prefix in ("pos_codes", "hp_codes"):
                out[f"{prefix}_{tag}"] = STORE.allocate_array(directory, f"{prefix}_{tag}", (n, 10), np.int8, -1)
        for prefix, shape in (("frame_xy", (n, 10, 2)), ("hp", (n, 10))):
            out[f"{prefix}_{cut}"] = STORE.allocate_array(directory, f"{prefix}_{cut}", shape, np.float64, np.nan)
        for prefix in ("frame_ts", "frame_idx"):
            out[f"{prefix}_{cut}"] = STORE.allocate_array(directory, f"{prefix}_{cut}", n, np.int64, -1)
        for horizon in C.HORIZONS_S:
            tag = f"{cut}_h{horizon}"
            for prefix in ("pre_gold", "post_gold", "pre_level_mean", "post_level_mean",
                           "pre_xp_mean", "post_xp_mean", "pre_hp_mean", "post_hp_mean"):
                shape = (n, 2) if prefix.endswith("gold") else n
                out[f"{prefix}_{tag}"] = STORE.allocate_array(directory, f"{prefix}_{tag}", shape, np.float64, np.nan)
    return out


def _fill_row(table: dict[str, np.ndarray], evidence: dict[str, Any], i: int, rec: dict[str, Any]) -> None:
    _fill_rows(table, evidence, i, [rec])


def _order_component_records(records, match_local, component_ids):
    expected = [(int(match_local), int(cid)) for cid in component_ids]
    actual = [(int(row["match_local"]), int(row["component_id"])) for row in records]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual):
        raise RuntimeError("duplicate component identity before storage")
    if set(actual) != set(expected):
        raise RuntimeError("component identity coverage differs from frozen baseline")
    by_key = dict(zip(actual, records))
    return [by_key[key] for key in expected]


def _slice_evidence(evidence, begin, end):
    offsets = evidence["first_loc_offsets"]
    a, b = int(offsets[begin]), int(offsets[end])
    out = {key: arr[begin:end] for key, arr in evidence.items()
           if key not in ("first_loc_offsets", "first_loc_x", "first_loc_y")}
    out.update(first_loc_offsets=np.asarray(offsets[begin:end + 1]) - a,
               first_loc_x=evidence["first_loc_x"][a:b], first_loc_y=evidence["first_loc_y"][a:b])
    return out


def _fill_rows(table, evidence, begin, records):
    """Write one bounded match by columns instead of millions of scattered writes."""
    end = begin + len(records)
    if not records:
        return
    for key, arr in table.items():
        values = [rec[key] for rec in records]
        if arr.dtype.kind == "U" and max(map(lambda v: len(str(v)), values)) > arr.dtype.itemsize // 4:
            raise RuntimeError(f"string truncation would occur: {key}")
        arr[begin:end] = values
    offsets = evidence["first_loc_offsets"]
    for axis in ("x", "y"):
        pieces = [np.asarray(rec[f"first_kill_{axis}"], dtype=np.float64) for rec in records]
        if any(piece.shape != (int(offsets[begin + j + 1] - offsets[begin + j]),)
               for j, piece in enumerate(pieces)):
            raise RuntimeError("first-kill location coverage mismatch")
        evidence[f"first_loc_{axis}"][int(offsets[begin]):int(offsets[end])] = np.concatenate(pieces)
    aliases = {f"dist_{tag}": f"_dists_{tag}" for tag in PART_TAGS}
    aliases.update({f"{prefix}_{tag}": f"_{prefix}_{tag}"
                   for tag in PART_TAGS for prefix in ("pos_codes", "hp_codes")})
    aliases.update({f"{prefix}_{cut}": f"_{prefix}_{cut}"
                   for cut in ("tau", "tm1") for prefix in ("frame_xy", "hp", "frame_ts", "frame_idx")})
    aliases.update({f"{prefix}_{cut}_h{h}": f"_{prefix}_{cut}_h{h}"
                   for cut in ("tau", "tm1") for h in C.HORIZONS_S
                   for prefix in ("pre_gold", "post_gold", "pre_level_mean", "post_level_mean",
                                  "pre_xp_mean", "post_xp_mean", "pre_hp_mean", "post_hp_mean")})
    for key, alias in aliases.items():
        evidence[key][begin:end] = [rec.get(alias, evidence[key][begin + j])
                                    for j, rec in enumerate(records)]


def _finalize_evidence(evidence: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    if evidence["first_loc_offsets"].shape != (n + 1,):
        raise RuntimeError("evidence row coverage mismatch")
    STORE.flush_arrays(evidence)
    return evidence


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    from . import thread_limits as _tl  # noqa: F401
    from .extract_match import process_match as _pm

    try:
        mid = str(payload["match_id"])
        def hashes():
            return {name: file_sha256(C.CACHE_DIR / f"{mid}{suffix}")
                    for name, suffix in (("npz", ".npz"), ("events", ".events.json"), ("meta", ".meta.json"))}
        before = hashes()
        bl = payload.get("baseline")
        out = _pm(
            int(payload["match_local"]),
            str(payload["match_id"]),
            np.asarray(payload["t"], dtype=np.int64),
            np.asarray(payload["x"], dtype=np.float64),
            np.asarray(payload["y"], dtype=np.float64),
            np.asarray(payload["killer"], dtype=np.int64),
            np.asarray(payload["victim"], dtype=np.int64),
            np.asarray(payload["assist"], dtype=np.int64),
            np.asarray(payload["labels"], dtype=np.int64),
            np.asarray(payload["blue_pids"], dtype=np.int16),
            baseline_rows=bl,
        )
        if bl is not None and "foreign_kills_in_tau_L" in bl:
            by = {int(c): i for i, c in enumerate(bl["component_id"].tolist())}
            for rec in out["components"]:
                j = by.get(int(rec["component_id"]))
                if j is not None:
                    rec["foreign_kills_in_tau_L"] = int(bl["foreign_kills_in_tau_L"][j])
                    rec["other_anchors_in_tau_L"] = int(bl["other_anchors_in_tau_L"][j])
                    for key in ("foreign_near", "foreign_distant", "game_end_ms", "game_end_known",
                                "next_component_tau", "next_kill_after_L", "overlap_next_tau_le_L",
                                "missing_terminal_game_end", "last_frame_ms"):
                        rec[key] = int(bl[key][j])
                    rec["call_n_unknown_team"] = int(bl["Call_n_unknown"][j])
                    rec["c0_n_credited"] = rec["c0_n"]
                    rec["call_n_credited"] = rec["call_n"]
                    rec["q_pre"] = rec["t_in"]
                    rec["terminal_support_authoritative"] = int(bl["terminal_support_authoritative"][j])
                    rec["frame_support_ok"] = not bool(out["match"].get("cache_pack_reason") or out["match"].get("frame_reason"))
                else:
                    rec["foreign_kills_in_tau_L"] = 0
                    rec["other_anchors_in_tau_L"] = 0
        after_production = hashes()
        ref = REF.replay_match_components(
            match_local=int(payload["match_local"]), match_id=mid,
            t=payload["t"], x=payload["x"], y=payload["y"], killer=payload["killer"],
            victim=payload["victim"], assist=payload["assist"], labels=payload["labels"],
            blue_pids=payload["blue_pids"], baseline_rows=bl)
        after_reference = hashes()
        hash_ok = before == after_production == after_reference == ref.get("source_hashes")
        if not hash_ok:
            raise RuntimeError("actual source versions differ before/after production/reference")
        return {"ok": True, "match_local": int(payload["match_local"]), "result": out,
                "reference": ref, "input_versions": {"before": before, "after_production": after_production,
                    "after_reference": after_reference, "reference_read": ref.get("source_hashes"), "ok": hash_ok}}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "match_local": int(payload["match_local"]),
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(limit=8),
            "input_versions": {"before": locals().get("before"),
                "after_production": locals().get("after_production"),
                "after_reference": locals().get("after_reference"), "ok": False},
        }


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return file_sha256(path)


def _run(
    run_dir: Path | None = None,
    max_matches: int | None = None,
    workers: int | None = None,
    tag: str = "",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    from . import reference_aggregates as RAG
    smoke = max_matches is not None
    if run_dir is None:
        run_dir = C.new_run_dir(("smoke" if smoke else "full") + (f"_{tag}" if tag else ""))
    else:
        run_dir = Path(run_dir)
    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        line = f"{_utc()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"start run_dir={run_dir} smoke={smoke} max_matches={max_matches}")
    np.random.seed(C.SEED)

    code_sha_start = _code_sha256()
    protocol_sha_start = _protocol_sha256()
    if protocol_sha_start != C.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("approved protocol SHA256 mismatch")
    write_json_atomic(
        run_dir / "hashes_start.json",
        {"code_sha256": code_sha_start, "protocol_sha256": protocol_sha_start, "utc": _utc()},
    )

    identity = verify_baseline_identity()
    write_json_atomic(run_dir / "identity_checks.json", identity)
    if not identity["all_ok"]:
        raise RuntimeError(f"baseline identity failed: {identity}")

    match_ids = load_match_ids()
    blue_pids = load_meta_blue_pids()
    ledger = load_ledger_and_ranges()
    assignments_start = _assignment_versions()
    write_json_atomic(run_dir / "assignment_input_versions.json", assignments_start)
    labels = build_full_labels(
        C.EXPECTED_TRAIN_KILLS, C.EXPECTED_TRAIN_MATCHES, ledger["starts"], ledger["ends"]
    )
    if _assignment_versions() != assignments_start:
        raise RuntimeError("assignment sources changed while labels were loaded")
    baseline = load_baseline_components()
    comp_starts, comp_ends = component_match_ranges(baseline["match_local"], C.EXPECTED_TRAIN_MATCHES)

    channel_map = REF.confirm_feature_channel_mapping()
    write_json_atomic(run_dir / "feature_channel_mapping.json", channel_map)
    if not channel_map.get("ok"):
        write_json_atomic(
            run_dir / "STATUS.json",
            {
                "complete": False,
                "smoke": bool(smoke),
                "raw_presence_verified": False,
                "failed_gates": ["feature_channel_mapping"],
                "utc": _utc(),
            },
        )
        raise RuntimeError(f"feature channel mapping confirmation failed: {channel_map}")

    log("raw source availability census begin")
    census = census_raw_source_availability(match_ids)
    # Persist FULL failure ledgers locally (no truncation); compact summary separately.
    write_json_atomic(
        run_dir / "raw_source_census_full.json",
        {
            "missing_rows": census.get("missing_rows", []),
            "mismatch_rows": census.get("mismatch_rows", []),
            "unsupported_rows": census.get("unsupported_rows", []),
            "found_but_unsupported_raw_rows": census.get("found_but_unsupported_raw_rows", []),
            "available_match_ids": census.get("available_match_ids", []),
        },
    )
    census_summary = {
        k: v
        for k, v in census.items()
        if k
        not in (
            "missing_rows",
            "mismatch_rows",
            "unsupported_rows",
            "found_but_unsupported_raw_rows",
        )
    }
    write_json_atomic(run_dir / "raw_source_census.json", census_summary)
    log(
        f"raw census: missing={census['n_missing']} found_unsupported={census['n_found_but_unsupported_raw']} "
        f"mismatch={census['n_mismatch']} mode={census['diagnostic_mode']} all_checked={census['all_keys_checked']}"
    )
    if census.get("hard_fail_found_unsupported_raw") or census.get("hard_fail_mismatch"):
        gate = (
            "found_originals_unsupported_by_implementation"
            if census.get("hard_fail_found_unsupported_raw")
            else "raw_source_identity_mismatch"
        )
        write_json_atomic(
            run_dir / "STATUS.json",
            {
                "complete": False,
                "smoke": bool(smoke),
                "raw_presence_verified": False,
                "failed_gates": [gate],
                "utc": _utc(),
            },
        )
        raise RuntimeError(
            "Found original detail/timeline pairs at recorded roots that mismatch or this "
            "implementation cannot support for raw presence claims; refusing silent absence/CACHE upgrade."
        )

    n_matches = C.EXPECTED_TRAIN_MATCHES
    n_use = min(int(max_matches), n_matches) if max_matches is not None else n_matches
    n_workers = int(workers) if workers is not None else C.MAX_WORKER_PROCESSES
    n_workers = max(1, min(n_workers, C.MAX_WORKER_PROCESSES, max(1, C.CPU_BUDGET - 1)))

    if n_use < n_matches:
        n_comp_out = int(np.sum(baseline["match_local"] < n_use))
        global_idx = np.flatnonzero(baseline["match_local"] < n_use)
    else:
        n_comp_out = C.EXPECTED_MIN1_COMPONENTS
        global_idx = np.arange(n_comp_out, dtype=np.int64)

    prototype = _allocate(1)
    missing_ref_fields = sorted(set(RAG.REFERENCE_REQUIRED_FIELDS) - set(prototype))
    if missing_ref_fields:
        raise RuntimeError(f"reference aggregate fields lack typed storage: {missing_ref_fields}")
    first_counts = baseline["same_ms_first_n"][global_idx]
    projected_bytes = n_comp_out * (sum(v.nbytes for v in prototype.values())
                + sum(v.nbytes for v in _allocate_evidence(1).values())
                + sum(prototype[k].nbytes for k in RAG.REFERENCE_REQUIRED_FIELDS))
    projected_bytes += int(np.sum(first_counts) - n_comp_out) * 16
    resource = STORE.resource_preflight(run_dir, projected_bytes)
    write_json_atomic(run_dir / "resource_preflight.json", resource)
    table = _allocate(n_comp_out, run_dir / "backing" / "production")
    evidence = _allocate_evidence(n_comp_out, run_dir / "backing" / "evidence", first_counts)
    reference_table = {key: STORE.allocate_array(run_dir / "backing" / "reference", key,
                      n_comp_out, prototype[key].dtype, np.nan if prototype[key].dtype.kind == "f" else 0)
                      for key in RAG.REFERENCE_REQUIRED_FIELDS}
    g2o = np.full(C.EXPECTED_MIN1_COMPONENTS, -1, dtype=np.int64)
    g2o[global_idx] = np.arange(n_comp_out, dtype=np.int64)

    table["foreign_kills_in_tau_L"][g2o[global_idx]] = baseline["foreign_kills_in_tau_L"][global_idx]
    table["other_anchors_in_tau_L"][g2o[global_idx]] = baseline["other_anchors_in_tau_L"][global_idx]

    errors: list[dict[str, Any]] = []
    match_ledger: list[dict[str, Any]] = []
    input_hash_ledger: list[dict[str, Any]] = []
    seen_matches: set[int] = set()
    ref_cmp = {"n_rows": 0, "n_reference_rows": 0, "n_field_comparisons": 0,
               "n_field_mismatches": 0, "missing_reference_rows": 0, "extra_reference_rows": 0,
               "compared_by_field": {}, "mismatches_by_field": {}, "expected_comparisons_by_field": {},
               "duplicate_table_keys": 0, "duplicate_reference_keys": 0, "unsupported_reference_rows": 0,
               "all_ok": True, "coverage_ok": True, "smoke": smoke,
               "verification_mode": "complete_per_match_source_replay",
               "bounded_records": True}
    reconcile_tot = {
        "cutoff_mismatches": 0,
        "endpoint_mismatches": 0,
        "c0_count_mismatches": 0,
        "n_components": 0,
    }
    peak_tree = 0
    n_done = 0
    n_zero_kill_processed = 0
    n_kills_processed = 0
    n_components_processed = 0
    n_source_errors = 0
    n_team_errors = 0
    n_event_errors = 0

    batches: list[list[int]] = []
    cur: list[int] = []
    for ml in range(n_use):
        cur.append(ml)
        if len(cur) >= C.BATCH_SIZE_MATCHES:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    log(f"n_use={n_use} n_comp_out={n_comp_out} batches={len(batches)} workers={n_workers}")

    def make_payload(ml: int) -> dict[str, Any]:
        s = int(ledger["starts"][ml])
        e = int(ledger["ends"][ml])
        cs = int(comp_starts[ml])
        ce = int(comp_ends[ml])
        bl = None
        if ce > cs:
            bl = {k: baseline[k][cs:ce] for k in baseline}
        return {
            "match_local": ml,
            "match_id": match_ids[ml],
            "t": ledger["t_ms"][s:e],
            "x": ledger["x"][s:e],
            "y": ledger["y"][s:e],
            "killer": ledger["killer"][s:e],
            "victim": ledger["victim"][s:e],
            "assist": ledger["assist_mask"][s:e],
            "labels": labels[s:e],
            "blue_pids": blue_pids[ml],
            "baseline": bl,
        }

    def _absorb(res: dict[str, Any]) -> None:
        nonlocal n_done, peak_tree, n_zero_kill_processed, n_kills_processed
        nonlocal n_components_processed, n_source_errors, n_team_errors, n_event_errors
        ml = int(res["match_local"])
        versions = res.get("input_versions", {})
        input_hash_ledger.append({"match_local": ml, "match_id": match_ids[ml], **versions})
        if not res["ok"]:
            errors.append({"match_local": ml, "error": res.get("error"), "traceback": res.get("traceback")})
            match_ledger.append({"match_local": ml, "status": "error", "error": res.get("error")})
            return
        if ml in seen_matches:
            raise RuntimeError(f"duplicate processed match key: {ml}")
        seen_matches.add(ml)
        result, ref = res["result"], res["reference"]
        recs, ref_rows = result["components"], ref["components"]
        meta, status = result.get("match") or {}, ref.get("source_status") or {}
        cs, ce = int(comp_starts[ml]), int(comp_ends[ml])
        if len(recs) != ce - cs or len(ref_rows) != ce - cs:
            raise RuntimeError(f"component coverage mismatch for match {ml}")
        recs = _order_component_records(recs, ml, baseline["component_id"][cs:ce])
        source_bad = bool(meta.get("cache_pack_reason") or meta.get("frame_reason")
                          or status.get("pack_reason") or status.get("frame_reason"))
        team_bad = not bool(meta.get("team_map_ok") and status.get("team_map_ok"))
        events_bad = not bool(meta.get("events_support_ok") and status.get("events_ok"))
        n_source_errors += int(source_bad)
        n_team_errors += int(team_bad)
        n_event_errors += int(events_bad)
        if source_bad or team_bad or events_bad or not versions.get("ok") or not ref.get("ok"):
            errors.append({"match_local": ml, "error": "source_support_or_version_gate",
                           "production_status": meta, "reference_status": status, "versions": versions})
        n_kills_m = int(ledger["ends"][ml] - ledger["starts"][ml])
        n_zero_kill_processed += int(n_kills_m == 0)
        n_kills_processed += n_kills_m
        n_components_processed += len(recs)
        br = result.get("baseline_reconcile") or {}
        for key in reconcile_tot:
            reconcile_tot[key] += int(br.get(key, 0))
        begin = int(g2o[cs]) if ce > cs else 0
        end = begin + len(recs)
        if ce > cs and not np.array_equal(g2o[cs:ce], np.arange(begin, end)):
            raise RuntimeError("noncontiguous component output key mapping")
        _fill_rows(table, evidence, begin, recs)
        cmp = REF.compare_table_to_reference({k: v[begin:end] for k, v in table.items()},
                    _slice_evidence(evidence, begin, end), ref_rows, smoke=smoke)
        for key in ("n_rows", "n_reference_rows", "n_field_comparisons", "n_field_mismatches",
                    "missing_reference_rows", "extra_reference_rows", "duplicate_table_keys",
                    "duplicate_reference_keys", "unsupported_reference_rows"):
            ref_cmp[key] += int(cmp.get(key, 0))
        for key in ("compared_by_field", "mismatches_by_field", "expected_comparisons_by_field"):
            for field, count in cmp.get(key, {}).items():
                ref_cmp[key][field] = ref_cmp[key].get(field, 0) + int(count)
        ref_cmp["all_ok"] &= bool(cmp.get("all_ok"))
        ref_cmp["coverage_ok"] &= bool(cmp.get("coverage_ok"))
        if not cmp.get("all_ok"):
            errors.append({"match_local": ml, "error": "independent_source_comparison", "comparison": cmp})
        with open(run_dir / "reference_match_checks.jsonl", "a", encoding="utf-8") as out:
            out.write(json.dumps({"match_local": ml, **cmp}, ensure_ascii=False) + "\n")
        reference_by_key = {(int(r["match_local"]), int(r["component_id"])): r for r in ref_rows}
        ordered_reference = [reference_by_key[(ml, int(rec["component_id"]))] for rec in recs]
        for key, arr in reference_table.items():
            arr[begin:end] = [row[key] for row in ordered_reference]
        match_ledger.append({"match_local": ml, "status": "ok" if cmp.get("all_ok") and not (source_bad or team_bad or events_bad) else "error",
            "n_kills": n_kills_m, "n_components": len(recs), "zero_kill": n_kills_m == 0,
            "source_hashes": ref.get("source_hashes"), "source_status": status, "input_versions_ok": bool(versions.get("ok"))})
        n_done += 1
        rss = _process_tree_rss_bytes()
        if rss["tree_rss"]:
            peak_tree = max(peak_tree, int(rss["tree_rss"]))
        if n_done % 500 == 0 or n_done == n_use:
            write_json_atomic(run_dir / "progress.json", {"n_done": n_done, "n_use": n_use,
                              "n_errors": len(errors), "utc": _utc()})
            write_json_atomic(run_dir / "failed_manifest.json", {"n": len(errors), "rows": errors})
            log(f"progress {n_done}/{n_use} errors={len(errors)}")

    def absorb(res):
        try:
            _absorb(res)
        except Exception as exc:
            errors.append({"match_local": int(res["match_local"]), "error": f"{type(exc).__name__}:{exc}",
                           "traceback": traceback.format_exc()})
            write_json_atomic(run_dir / "errors.json", {"n": len(errors), "rows": errors})
            write_json_atomic(run_dir / "failed_manifest.json", {"n": len(errors), "rows": errors})
            write_json_atomic(run_dir / "input_hash_ledger.json", {"n": len(input_hash_ledger), "rows": input_hash_ledger})
            write_json_atomic(run_dir / "match_ledger.json", {"n": len(match_ledger), "rows": match_ledger})
            raise

    if n_workers == 1:
        for batch in batches:
            for ml in batch:
                absorb(_worker(make_payload(ml)))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for batch in batches:
                local_futs = [ex.submit(_worker, make_payload(ml)) for ml in batch]
                for fut in as_completed(local_futs):
                    absorb(fut.result())

    # Preserve ALL errors without truncation.
    write_json_atomic(run_dir / "errors.json", {"n": len(errors), "rows": errors})
    write_json_atomic(run_dir / "failed_manifest.json", {"n": len(errors), "rows": errors})
    write_json_atomic(
        run_dir / "match_ledger.json",
        {
            "n": len(match_ledger),
            "n_use": n_use,
            "n_distinct_ok": sum(1 for r in match_ledger if r.get("status") == "ok"),
            "n_zero_kill": n_zero_kill_processed,
            "n_kills": n_kills_processed,
            "n_components": n_components_processed,
            "n_source_errors": n_source_errors,
            "n_team_errors": n_team_errors,
            "n_event_errors": n_event_errors,
            "rows": match_ledger,
        },
    )
    write_json_atomic(run_dir / "input_hash_ledger.json", {"n": len(input_hash_ledger), "rows": input_hash_ledger})
    if errors:
        write_json_atomic(
            run_dir / "STATUS.json",
            {
                "complete": False,
                "smoke": bool(smoke),
                "failed_gates": ["match_processing_error"],
                "n_errors": len(errors),
                "utc": _utc(),
            },
        )
        raise RuntimeError(f"{len(errors)} match errors; see errors.json")
    if len(match_ledger) != n_use:
        raise RuntimeError(f"match ledger coverage {len(match_ledger)} != n_use {n_use}")

    log("finalize evidence")
    evidence_arr = _finalize_evidence(evidence, n_comp_out)

    STORE.flush_arrays(table)
    STORE.flush_arrays(reference_table)
    aggregate_args = dict(n_matches_source=n_matches if not smoke else n_use,
        n_zero_kill=C.EXPECTED_ZERO_KILL if not smoke else n_zero_kill_processed,
        n_matches_processed=n_use, n_zero_kill_processed=n_zero_kill_processed)
    log("production aggregate computation")
    aggregates = AG.build_aggregates(table, **aggregate_args)
    log("independent aggregate computation from source-derived typed rows")
    ref_agg = RAG.build_reference_aggregates(reference_table, **aggregate_args)
    agg_cmp = RAG.compare_aggregates(ref_agg, aggregates)
    if smoke:
        for report in (aggregates, ref_agg):
            report.update(complete=False, smoke=True, reconcile_ok=False,
                          reconcile_full_corpus_excluded=True)
    write_json_atomic(run_dir / "aggregates.json", aggregates)
    write_json_atomic(run_dir / "reference_aggregates.json", {"reference": ref_agg, "compare": agg_cmp})
    write_json_atomic(run_dir / "reference_source_verification.json", ref_cmp)

    baseline_mapping = {"f": "f", "L": "L", "tau": "tau", "n_members": "n_members",
        "same_ms_first_n": "same_ms_first_n", "F_tau": "F_inclusive", "F_tm1": "F_strict",
        "lag_f_F_tau": "delta", "t_in": "t_in", "a_strict": "a_strict",
        "exact_10000_tau": "exact_10000", "strict_eligible": "strict_eligible",
        "c0_n": "C0_n_credited", "call_n": "Call_n_credited"}
    for h in C.HORIZONS_S:
        baseline_mapping[f"e_h{h}"] = f"e_h{h}"
        baseline_mapping[f"valid_h{h}"] = f"valid_h{h}"
    baseline_checks = {key: bool(np.array_equal(table[key], baseline[other][global_idx]))
                       for key, other in baseline_mapping.items()}
    indep_inmem = {"all_ok": all(baseline_checks.values()), "fields": baseline_checks,
                   "n_checks": n_comp_out * len(baseline_checks),
                   "n_mismatches": sum(not value for value in baseline_checks.values())}
    log("write verified typed outputs")
    components_sha = _save_npz_atomic(run_dir / "components.npz", table)
    evidence_sha = _save_npz_atomic(run_dir / "evidence.npz", evidence_arr)
    reference_sha = _save_npz_atomic(run_dir / "reference_fields.npz", reference_table)
    log("bounded reload verification of every saved field")
    reload_ver = {"components": STORE.verify_archive(run_dir / "components.npz", table, components_sha, file_sha256),
                  "evidence": STORE.verify_archive(run_dir / "evidence.npz", evidence_arr, evidence_sha, file_sha256),
                  "reference": STORE.verify_archive(run_dir / "reference_fields.npz", reference_table, reference_sha, file_sha256),
                  "mode": "per-field_reload_against_source_verified_typed_storage"}
    reload_ver["all_ok"] = all(reload_ver[key]["all_ok"] for key in ("components", "evidence", "reference"))
    expected_kills = int(np.sum(ledger["ends"][:n_use] - ledger["starts"][:n_use]))
    expected_zero = int(np.count_nonzero(ledger["ends"][:n_use] == ledger["starts"][:n_use]))
    coverage_checks = {"match_keys_exact": seen_matches == set(range(n_use)),
        "match_ledger_exact": len(match_ledger) == n_use == n_done,
        "match_statuses_ok": all(r["status"] == "ok" for r in match_ledger),
        "components_exact": n_components_processed == n_comp_out,
        "kills_exact": n_kills_processed == expected_kills,
        "zero_kill_exact": n_zero_kill_processed == expected_zero,
        "input_versions_exact": len(input_hash_ledger) == n_use and all(r.get("ok") for r in input_hash_ledger),
        "source_support_ok": n_source_errors == n_team_errors == n_event_errors == 0,
        "all_source_keys_censused": bool(census.get("all_keys_checked"))}
    coverage_gate = all(coverage_checks.values())
    indep = {"in_memory": indep_inmem, "reload": reload_ver, "reference_source": ref_cmp,
        "reference_aggregates": agg_cmp, "feature_channel_mapping": channel_map,
        "coverage": {"n_matches_processed": n_done, "n_match_ledger": len(match_ledger),
            "n_zero_kill_processed": n_zero_kill_processed, "n_zero_kill_expected": expected_zero,
            "n_kills_processed": n_kills_processed, "n_kills_expected": expected_kills,
            "n_components_processed": n_components_processed, "n_source_errors": n_source_errors,
            "n_team_errors": n_team_errors, "n_event_errors": n_event_errors,
            "checks": coverage_checks, "ok": coverage_gate}}
    indep["all_ok"] = bool(indep_inmem["all_ok"] and reload_ver["all_ok"] and ref_cmp["all_ok"]
                         and agg_cmp["all_ok"] and channel_map.get("ok") and coverage_gate)
    indep["n_checks"] = indep_inmem["n_checks"] + ref_cmp["n_field_comparisons"] + int(agg_cmp.get("n_checks", 0))
    indep["n_mismatches"] = indep_inmem["n_mismatches"] + ref_cmp["n_field_mismatches"] + int(agg_cmp.get("n_mismatches", 0))
    for key in ("components", "evidence", "reference"):
        indep["n_checks"] += reload_ver[key]["n_checks"]
        indep["n_mismatches"] += reload_ver[key]["n_mismatches"]
    write_json_atomic(run_dir / "independent_verification.json", indep)

    log("final cache and assignment source-version verification")
    final_versions = _verify_final_cache_versions(input_hash_ledger, n_workers)
    write_json_atomic(run_dir / "final_cache_versions.json", final_versions)
    assignments_unchanged = _assignment_versions() == assignments_start
    code_sha_end = _code_sha256()
    protocol_sha_end = _protocol_sha256()
    identity_end = verify_baseline_identity()
    channel_map_end = REF.confirm_feature_channel_mapping()
    channel_map_unchanged = bool(channel_map_end.get("ok") and channel_map.get("ok")
        and channel_map.get("source_sha256")
        and channel_map_end.get("source_sha256") == channel_map.get("source_sha256"))
    indep["final_input_versions"] = {"cache_all_ok": final_versions["all_ok"],
        "cache_n_checked": final_versions["n_checked"], "cache_n_mismatches": final_versions["n_mismatches"],
        "assignments_unchanged": assignments_unchanged, "feature_source_unchanged": channel_map_unchanged}
    indep["all_ok"] &= bool(final_versions["all_ok"] and assignments_unchanged and channel_map_unchanged)
    indep["n_checks"] += 3 * final_versions["n_checked"] + len(assignments_start) + 1
    indep["n_mismatches"] += final_versions["n_mismatches"] + int(not assignments_unchanged) + int(not channel_map_unchanged)
    write_json_atomic(run_dir / "independent_verification.json", indep)
    write_json_atomic(
        run_dir / "hashes_end.json",
        {
            "code_sha256": code_sha_end,
            "protocol_sha256": protocol_sha_end,
            "code_unchanged": code_sha_end == code_sha_start,
            "protocol_unchanged": protocol_sha_end == protocol_sha_start,
            "baseline_unchanged": bool(identity_end.get("all_ok")),
            "feature_channel_source_unchanged": channel_map_unchanged,
            "cache_sources_unchanged": final_versions["all_ok"],
            "assignment_sources_unchanged": assignments_unchanged,
            "utc": _utc(),
        },
    )

    wall = time.perf_counter() - t0
    rss = _process_tree_rss_bytes()
    out_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())

    failed_gates = []
    if smoke:
        failed_gates.append("smoke_incomplete_by_design")
    if not aggregates.get("reconcile_ok") and not smoke:
        failed_gates.append("baseline_gap_reconcile")
    if not indep.get("all_ok"):
        failed_gates.append("independent_verification")
    if not ref_cmp.get("all_ok"):
        failed_gates.append("reference_source_verification")
    if not agg_cmp.get("all_ok"):
        failed_gates.append("reference_aggregate_verification")
    if reconcile_tot["cutoff_mismatches"] or reconcile_tot["c0_count_mismatches"]:
        failed_gates.append("baseline_field_reconcile")
    if reconcile_tot["endpoint_mismatches"]:
        failed_gates.append("endpoint_formula_mismatch")
    if code_sha_end != code_sha_start or protocol_sha_end != protocol_sha_start:
        failed_gates.append("code_or_protocol_changed_during_run")
    if not identity_end.get("all_ok"):
        failed_gates.append("baseline_changed_during_run")
    if not channel_map_unchanged:
        failed_gates.append("feature_source_changed_during_run")
    if not final_versions["all_ok"] or not assignments_unchanged:
        failed_gates.append("input_source_changed_during_run")
    if not coverage_gate:
        failed_gates.append("processed_coverage")
    if len(match_ledger) != n_use:
        failed_gates.append("match_ledger_incomplete")

    # Smoke may exit 0 with smoke=true,complete=false when all smoke gates pass.
    smoke_blocking = [g for g in failed_gates if g != "smoke_incomplete_by_design"]
    complete = (
        (not smoke)
        and bool(aggregates.get("reconcile_ok"))
        and bool(indep.get("all_ok"))
        and len(errors) == 0
        and not failed_gates
    )
    smoke_ok = bool(smoke) and len(smoke_blocking) == 0 and bool(indep.get("all_ok"))

    manifest = {
        "study": "observation_support_20261002",
        "generated_utc": _utc(),
        "run_dir": str(run_dir),
        "complete": bool(complete),
        "smoke": bool(smoke),
        "max_matches": max_matches,
        "seed": C.SEED,
        "cpu_budget": C.CPU_BUDGET,
        "workers": n_workers,
        "thread_limits": TL.applied_settings(),
        "n_matches_processed": n_use,
        "n_match_ledger": len(match_ledger),
        "n_components": n_comp_out,
        "n_components_processed": n_components_processed,
        "n_kills_processed": n_kills_processed,
        "n_zero_kill_source": C.EXPECTED_ZERO_KILL,
        "n_zero_kill_processed": n_zero_kill_processed,
        "n_source_errors": n_source_errors,
        "n_team_errors": n_team_errors,
        "n_event_errors": n_event_errors,
        "smoke_ok": bool(smoke_ok) if smoke else None,
        "diagnostic_mode": census["diagnostic_mode"],
        "raw_presence_verified": False,
        "raw_presence_scope": {
            "status": "unavailable_at_checked_locations"
            if census["n_missing"] == census["n_train_keys"]
            else "partial_or_unsupported",
            "n_missing": census["n_missing"],
            "n_found_but_unsupported_raw": census["n_found_but_unsupported_raw"],
            "n_available_verified_supported": census["n_available_verified_supported"],
            "checked_locations": census["checked_locations_only"],
            "all_keys_checked": census["all_keys_checked"],
            "note": "Cache-only proxies must not be named actual participants or definitive N0",
        },
        "identity": identity,
        "source_census_sha256": file_sha256(run_dir / "raw_source_census.json"),
        "components_sha256": components_sha,
        "evidence_sha256": evidence_sha,
        "reference_fields_sha256": reference_sha,
        "reference_match_checks_sha256": file_sha256(run_dir / "reference_match_checks.jsonl"),
        "input_hash_ledger_sha256": file_sha256(run_dir / "input_hash_ledger.json"),
        "resource_preflight": resource,
        "aggregates_sha256": file_sha256(run_dir / "aggregates.json"),
        "independent_verification_sha256": file_sha256(run_dir / "independent_verification.json"),
        "code_sha256": code_sha_end,
        "protocol_sha256": protocol_sha_end,
        "code_unchanged_during_run": code_sha_end == code_sha_start,
        "baseline_hashes": {
            "manifest": C.EXPECTED_ANCHOR_MANIFEST_SHA256,
            "components": C.EXPECTED_COMPONENTS_SHA256,
            "terminals": C.EXPECTED_TERMINALS_SHA256,
        },
        "radius_source": {
            "path": str(C.CHAMPION_FULL),
            "sha256": C.EXPECTED_CHAMPION_FULL_SHA256,
            "bytes": C.EXPECTED_CHAMPION_FULL_BYTES,
            "diagnostic_radii": list(C.DIAGNOSTIC_RADII),
            "radius_adopted": False,
        },
        "channel_names": {
            "gold_team_minute": "gold_team_minute",
            "level_norm_index": C.IDX_LEVEL_NORM,
            "xp_norm_index": C.IDX_XP_NORM,
            "hp_pct_index": C.IDX_HP_PCT,
        },
        "reconcile": reconcile_tot,
        "aggregates_summary": {
            "n_cand_10_20_tau": aggregates.get("n_cand_10_20_tau"),
            "n_isolated_cand_10_20_tau": aggregates.get("n_isolated_cand_10_20_tau"),
            "n_exact_10000_tau": aggregates.get("n_exact_10000_tau"),
            "n_strict_eligible": aggregates.get("n_strict_eligible"),
            "reconcile_ok": aggregates.get("reconcile_ok"),
        },
        "independent_verification": {
            "all_ok": indep.get("all_ok"),
            "n_checks": indep.get("n_checks"),
            "n_mismatches": indep.get("n_mismatches"),
        },
        "failed_gates": failed_gates,
        "wall_s": wall,
        "peak_tree_rss_bytes": peak_tree or rss.get("tree_rss"),
        "output_bytes": out_bytes,
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
        },
        "scientific_limits": [
            "No V/SVI/q training or parameter adoption",
            "CACHE-ONLY participation is proxy bounds, not definitive N0/P0",
            "Raw participantFrames presence not verified for TRAIN keys at recorded roots",
            "Absence claims limited to checked recorded roots only",
            "complete means this bounded diagnostic passed, not training-ready",
        ],
    }
    write_json_atomic(run_dir / "RUN_MANIFEST.json", manifest)
    write_json_atomic(
        run_dir / "STATUS.json",
        {
            "complete": bool(complete),
            "smoke": bool(smoke),
            "smoke_ok": bool(smoke_ok) if smoke else None,
            "raw_presence_verified": False,
            "failed_gates": failed_gates,
            "utc": _utc(),
        },
    )
    log(f"done complete={complete} smoke_ok={smoke_ok if smoke else None} wall_s={wall:.1f} failed_gates={failed_gates}")
    manifest["_smoke_ok"] = bool(smoke_ok) if smoke else None
    manifest["_validation_failed"] = bool((smoke and not smoke_ok) or ((not smoke) and not complete))
    return manifest


def run(run_dir: Path | None = None, max_matches: int | None = None,
        workers: int | None = None, tag: str = "") -> dict[str, Any]:
    smoke = max_matches is not None
    if max_matches is not None and not 1 <= max_matches <= C.EXPECTED_TRAIN_MATCHES:
        raise ValueError("max_matches outside frozen TRAIN range")
    run_dir = (Path(run_dir) if run_dir else C.new_run_dir(("smoke" if smoke else "full") + (f"_{tag}" if tag else ""))).resolve()
    if not run_dir.is_relative_to(C.ROOT.resolve()):
        raise ValueError("run directory must remain inside project workspace")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refuse non-empty run_dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "STATUS.json", {"complete": False, "smoke": smoke,
                      "state": "running", "utc": _utc()})
    try:
        snapshot = run_dir / "code_snapshot"
        for source in _source_files():
            target = snapshot / source.relative_to(C.STUDY)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(C.PROTOCOL, snapshot / "PROTOCOL.md")
        write_json_atomic(snapshot / "SHA256.json", {str(p.relative_to(snapshot)): file_sha256(p)
                          for p in snapshot.rglob("*") if p.is_file()})
        return _run(run_dir, max_matches, workers, tag)
    except Exception as exc:
        status_path = run_dir / "STATUS.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(complete=False, state="failed", error=f"{type(exc).__name__}:{exc}",
                      traceback=traceback.format_exc(), utc=_utc())
        status.setdefault("failed_gates", []).append("run_exception")
        write_json_atomic(status_path, status)
        write_json_atomic(run_dir / "run_exception.json", status)
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="observation_support_20261002 diagnostic")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args(argv)
    try:
        manifest = run(
            run_dir=Path(args.run_dir) if args.run_dir else None,
            max_matches=args.max_matches,
            workers=args.workers,
            tag=args.tag,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    if manifest.get("_validation_failed"):
        return 1
    # Intentionally incomplete smoke with all smoke gates passing may exit 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
