"""Produce observable banks, sampler ledgers, and component audits without fitting."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bank import ShardBuilder, pack_bank, unpack_bank
from .categorical import static_lookup_counts
from .kernel import ObservableMatch
from .population import HORIZONS_S, REASON_CODES, build_population, normalize_champion_kills
from .sampling import query_support, sample_match_queries
from .schema import frame_spec
from .source import read_cached_triplet
from .storage import array_digest, write_shard

COMPONENT_COLUMNS = (
    "component_id", "n_members", "same_ms_first_n", "f", "L", "pre",
    "F_pre_ms", "F_pre_index", "pre_frame_age_ms", "first_kill_frame_lag_ms",
    "firstkill_credit_bits", "firstkill_credit_blue", "firstkill_credit_red", "c0_n_env_killer",
    "next_kill_after_L", "next_component_anchor", "foreign_kills_in_pre_L",
    "other_component_anchors_in_pre_L", "game_end_ms", "winning_team",
    "support_ok_tau_r1800", "p_lower_bits_tau_r1800", "p_upper_bits_tau_r1800",
    "support_ok_tau_r2000", "p_lower_bits_tau_r2000", "p_upper_bits_tau_r2000",
    "pre_frame_support", "team_map_ok", "events_support_ok", "terminal_support_authoritative",
    "terminal_component_order_ok", "firstkill_direct_opposition", "global_isolation_verified",
    "overlap_next_anchor_le_L", "candidate_eligible", "supported_candidate_eligible",
    "primary_candidate_eligible", "early_candidate_eligible",
) + tuple(f"{prefix}_h{h}" for h in HORIZONS_S for prefix in
          ("e", "valid", "post_frame_index", "F_post"))
TRUNCATION_NAMES = ("horizon", "next_kill", "next_anchor", "game_end")


def _equal_value(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal_value(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and np.isnan(left):
        return isinstance(right, float) and np.isnan(right)
    return left == right


def _endpoint_check(rows: list[dict], match: ObservableMatch, kills: list[dict]) -> None:
    """Independent direct closed-interval and strict endpoint checks for every row."""
    times = {k["source_ordinal"]: k["timestamp_ms"] for k in kills if k["valid_st"]}
    firsts = [r["f"] for r in rows]
    for row in rows:
        member = set(row["member_source_ordinals"])
        if not member or row["f"] != min(times[k] for k in member) or row["L"] != max(times[k] for k in member):
            raise ValueError("component member times differ from source")
        pre, last = row["pre"], row["L"]
        if pre != row["f"] - 10000:
            raise ValueError("pre anchor drift")
        foreign = sum(pre <= t <= last for k, t in times.items() if k not in member)
        others = sum(pre <= f - 10000 <= last for index, f in enumerate(firsts) if index != row["component_id"])
        if foreign != row["foreign_kills_in_pre_L"] or others != row["other_component_anchors_in_pre_L"]:
            raise ValueError("closed global isolation mismatch")
        later = [f - 10000 for f in firsts if f > row["f"]]
        next_anchor = min(later) if later else -1
        next_kills = [t for t in times.values() if t > last]
        next_kill = min(next_kills) if next_kills else -1
        if next_anchor != row["next_component_anchor"] or next_kill != row["next_kill_after_L"]:
            raise ValueError("strict next boundary mismatch")
        for h in HORIZONS_S:
            bounds = [last + 1000 * h, match.end_ms - 1]
            if next_anchor >= 0:
                bounds.append(next_anchor - 1)
            if next_kill >= 0:
                bounds.append(next_kill - 1)
            expected = min(bounds)
            if row[f"e_h{h}"] != expected:
                raise ValueError("endpoint minimum mismatch")
            index, reasons = query_support(match.minute_ts, expected)
            valid = expected >= last and expected > pre and expected < match.end_ms and not reasons
            if row[f"valid_h{h}"] != valid:
                raise ValueError("endpoint support mismatch")


def _queries(match: ObservableMatch, entry: dict, rows: list[dict]) -> tuple[dict, Counter]:
    sampler = sample_match_queries(entry["match_id"], int(match.end_ms), match.minute_ts)
    times = {r["actual_query_ms"] for r in sampler if r["query_support_valid"]}
    for row in rows:
        # Early components are audited separately and never expand primary V support.
        if row["primary_candidate_eligible"] and row["supported_candidate_eligible"]:
            times.add(row["pre"])
            times.update(row[f"e_h{h}"] for h in HORIZONS_S if row[f"valid_h{h}"])
    times = sorted(times)
    refs = {t: index for index, t in enumerate(times)}
    snapshots = [match.query_numeric(t) for t in times]
    result = {
        "query_ms": np.asarray(times, dtype=np.int32),
        "query_frame_index": np.asarray([s["frame_index"] for s in snapshots], dtype=np.int32),
        "query_event_limit": np.asarray([s["event_cutoff_index"] for s in snapshots], dtype=np.int32),
        "query_item_limit": np.asarray([s["item_event_cutoff_index"] for s in snapshots], dtype=np.int32),
        "query_team_kills": np.asarray([s["team_kills"] for s in snapshots], dtype=np.int32).reshape(-1, 2),
        "query_kda": np.asarray([s["kda"] for s in snapshots], dtype=np.int32).reshape(-1, 3, 10),
        "query_objective_counts": np.asarray([s["objective_counts"] for s in snapshots], dtype=np.int32),
        "query_objective_last_ms": np.asarray([s["objective_last_timestamp_ms"] for s in snapshots], dtype=np.float64),
        "query_item_ledger_known": np.asarray([s["item_ledger_known"] for s in snapshots], dtype=bool),
        "v_nominal": np.asarray([[r["bucket_id"], r["bucket_lo_ms"], r["bucket_hi_ms"],
                                  r["nominal_query_ms"], r["actual_query_ms"] if r["actual_query_ms"] is not None else -1,
                                  r["frame_index"] if r["frame_index"] is not None else -1,
                                  refs.get(r["actual_query_ms"], -1)] for r in sampler], dtype=np.int32).reshape(-1, 7),
        "component_query_refs": np.asarray([[refs.get(r["pre"], -1) if r["primary_candidate_eligible"] and r["supported_candidate_eligible"] else -1]
                                             + [refs.get(r[f"e_h{h}"], -1) if r["primary_candidate_eligible"] and r["supported_candidate_eligible"] and r[f"valid_h{h}"] else -1 for h in HORIZONS_S]
                                             for r in rows], dtype=np.int32).reshape(-1, 4),
    }
    from .kernel import OBJECTIVE_TYPES
    if not snapshots:
        result["query_objective_counts"] = np.empty((0, len(OBJECTIVE_TYPES), 3), dtype=np.int32)
        result["query_objective_last_ms"] = np.empty((0, len(OBJECTIVE_TYPES)), dtype=np.float64)
    counts = Counter(n_nominal_V_queries=len(sampler), n_supported_V_queries=sum(r["query_support_valid"] for r in sampler),
                     n_unique_queries=len(times), n_query_roundtrips=len(times))
    counts.update({"V_support_exclusion:" + reason: sum(reason in r["query_support_reasons"] for r in sampler)
                   for reason in {reason for r in sampler for reason in r["query_support_reasons"]}})
    return result, counts


def _independent_credit_check(events: list[dict], match: ObservableMatch, queries: dict) -> None:
    """Recount original source credits directly, independently of kernel prefixes."""
    times, increments, seen = [], [], set()
    identity_fields = ("source_id", "sourceId", "source_index", "sourceIndex", "event_index", "eventIndex")
    for ordinal, event in enumerate(events):
        identity = f"ordinal:{ordinal}"
        for name in identity_fields:
            value = event.get(name)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                text = str(value)
                identity = text if name in {"source_id", "sourceId"} and ":" in text else f"{name}:{text}"
                break
        first_identity = identity not in seen
        seen.add(identity)
        if event.get("type") != "CHAMPION_KILL":
            continue
        delta = np.zeros(32, dtype=np.int32)
        if first_identity:
            killer, victim = event.get("killerId"), event.get("victimId")
            if type(killer) is int and 1 <= killer <= 10:
                delta[0 if match.team_ids[killer - 1] == 100 else 1] += 1
                delta[2 + killer - 1] += 1
            if type(victim) is int and 1 <= victim <= 10:
                delta[12 + victim - 1] += 1
            assists = event.get("assistingParticipantIds", [])
            if isinstance(assists, list):
                for pid in set(v for v in assists if type(v) is int and 1 <= v <= 10):
                    delta[22 + pid - 1] += 1
        times.append(int(event["timestamp"]))
        increments.append(delta)
    order = np.argsort(times, kind="stable")
    prefix = np.vstack((np.zeros((1, 32), dtype=np.int32),
                        np.cumsum(np.asarray(increments, dtype=np.int32).reshape(-1, 32)[order], axis=0)))
    indices = np.searchsorted(np.asarray(times, dtype=np.int64)[order], queries["query_ms"], side="right")
    expected = prefix[indices]
    if not np.array_equal(expected[:, :2], queries["query_team_kills"]):
        raise ValueError("independent original-source team kill prefix mismatch")
    if not np.array_equal(expected[:, 2:].reshape(-1, 3, 10), queries["query_kda"]):
        raise ValueError("independent original-source player KDA prefix mismatch")


def _component_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    offsets, members, reasons, reason_offsets = [0], [], [], [0]
    for row in rows:
        members.extend(row["member_source_ordinals"])
        offsets.append(len(members))
        reasons.extend(row["reason_codes"])
        reason_offsets.append(len(reasons))
    return {
        "component_values": np.asarray([[int(r[k]) for k in COMPONENT_COLUMNS] for r in rows], dtype=np.int32).reshape(-1, len(COMPONENT_COLUMNS)),
        "component_key": np.asarray([r["component_key"] for r in rows], dtype="S64"),
        "component_member_offsets": np.asarray(offsets, dtype=np.int32),
        "component_member_ordinals": np.asarray(members, dtype=np.int32),
        "component_reason_offsets": np.asarray(reason_offsets, dtype=np.int32),
        "component_reason_codes": np.asarray(reasons, dtype=np.int32),
    }


def produce_match(entry: dict) -> tuple[dict, dict]:
    arrays, meta, events, info = read_cached_triplet(entry)
    _, indices, _ = frame_spec()
    projected = {"minute_ts": arrays["minute_ts"], "node_frame": arrays["node_minute"][:, :, indices],
                 "xy": arrays["xy_raw_minute"], "gold_team": arrays["gold_team_minute"]}
    meta = dict(meta, canonical_match_key=entry["canonical_match_key"])
    match = ObservableMatch(projected, meta, events, entry["patch"])
    audited = entry["audited_numbers"]
    kills = normalize_champion_kills(match)
    actual = {"n_frames": len(match.minute_ts), "n_events": len(events), "n_kills": len(kills),
              "game_end_ms": match.end_ms, "game_end_winning_team": match.winning_team}
    if actual != audited or len(match.typed_events) != len(events):
        raise ValueError("observable source counts or verified terminal differ from accepted full audit")
    rows = build_population(match, kills=kills)
    _endpoint_check(rows, match, kills)
    queries, counts = _queries(match, entry, rows)
    _independent_credit_check(events, match, queries)
    counts["n_independent_credit_queries"] = len(queries["query_ms"])
    original_bank = match.export_bank()
    packed = pack_bank(original_bank)
    decoded = unpack_bank(packed)
    for key in original_bank:
        # Default identities are ordinal-derived, never treated as explicit hashes.
        if key == "event_identity_code":
            explicit = original_bank["event_identity_explicit"]
            equal = np.array_equal(original_bank[key][explicit], decoded[key][explicit])
        else:
            equal = np.array_equal(original_bank[key], decoded[key], equal_nan=True)
        if not equal:
            raise ValueError("lossless sparse bank roundtrip mismatch: " + key)
    reopened = ObservableMatch.from_bank(decoded, entry["patch"])
    roundtrip_components = _component_arrays(build_population(reopened))
    for key, value in _component_arrays(rows).items():
        if not np.array_equal(value, roundtrip_components[key]):
            raise ValueError("component semantic reload mismatch: " + key)
    for timestamp in queries["query_ms"]:
        if not _equal_value(match.query_numeric(int(timestamp)), reopened.query_numeric(int(timestamp))):
            raise ValueError("every-query numeric semantic reload mismatch")
    counts.update(n_matches=1, n_frames=len(match.minute_ts), n_events=len(events), n_kills=len(kills),
                  n_components=len(rows), n_zero_kill_matches=int(not kills), n_match_roundtrips=1,
                  n_timing_candidates=sum(r["timing_candidate"] for r in rows),
                  n_candidate_eligible=sum(r["candidate_eligible"] for r in rows),
                  n_primary_candidates=sum(r["primary_candidate_eligible"] for r in rows),
                  n_supported_primary=sum(r["primary_candidate_eligible"] and r["supported_candidate_eligible"] for r in rows),
                  n_early_candidates=sum(r["early_candidate_eligible"] for r in rows),
                  n_actual_scale_verified=0)
    packed.update(queries)
    packed.update(_component_arrays(rows))
    support = {
        "frame_zero_counts": np.sum(match.node_frame == 0, axis=(0, 1)).tolist(),
        "frame_nonfinite_counts": np.sum(~np.isfinite(match.node_frame), axis=(0, 1)).tolist(),
        "event_field_presence": original_bank["event_field_validity"].sum(axis=0).tolist(),
        "static_lookup_counts": static_lookup_counts(match.static_ids, match.bans, entry["patch"]),
        "component_reason_counts": dict(Counter(reason for row in rows for reason in row["all_reasons"])),
    }
    ledger = {"canonical_match_key": entry["canonical_match_key"], "patch": entry["patch"], "role": entry["role"],
              "original_index": entry["original_index"], "counts": dict(counts), "source_hashes": entry["source_hashes"],
              "source_stable": all(info[k]["stable"] for k in ("npz", "meta", "events")), "support": support}
    return packed, ledger


def produce_shard(task: tuple[str, list[dict]]) -> dict:
    directory_text, entries = task
    directory = Path(directory_text)
    start = time.perf_counter()
    builder = ShardBuilder()
    ledgers = []
    counts = Counter()
    try:
        for entry in entries:
            bank, ledger = produce_match(entry)
            builder.append(bank)
            ledgers.append(ledger)
            counts.update(ledger["counts"])
        arrays, layout = builder.finish()
        manifest = write_shard(directory, arrays, compressed=True)
        (directory / "LAYOUT.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        with (directory / "MATCH_LEDGER.jsonl").open("x", encoding="utf-8") as handle:
            for row in ledgers:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        result = {"complete": True, "directory": directory.name, "role": entries[0]["role"],
                  "counts": dict(counts), "payload_bytes": manifest["logical_bytes"],
                  "stored_bytes": manifest["archive_bytes"], "elapsed_seconds": time.perf_counter() - start}
        if __import__("os").name == "nt":
            import ctypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong)] + [
                    (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "quota_peak_paged", "quota_paged",
                                                        "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = ctypes.c_void_p
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
            if get_info(get_process(), ctypes.byref(counters), counters.cb):
                result["worker_peak_working_set_bytes"] = int(counters.peak_working_set)
        (directory / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        detail = {"complete": False, "n_completed_matches": len(ledgers), "error": f"{type(exc).__name__}: {exc}"}
        if len(ledgers) < len(entries):
            detail["failed_key"] = entries[len(ledgers)]["canonical_match_key"]
        (directory / "FAILURE.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        raise


def producer_layout() -> dict:
    return {"component_columns": list(COMPONENT_COLUMNS), "component_query_refs": ["pre", "h60", "h90", "h120"],
            "nominal_V_columns": ["bucket_id", "lo_ms", "hi_ms", "nominal_ms", "actual_ms_or_minus1", "frame_index_or_minus1", "query_index_or_minus1"],
            "reason_codes": dict(REASON_CODES), "query_refs_local_to_match": True,
            "component_key_scope": "canonical_match_key plus component_key", "model_input_exclusions":
            ["component audit", "future membership", "firstkill location", "winner", "patch role", "source identifiers"],
            "stored_query_values": "held frame references and observed inclusive event-prefix summaries",
            "no_V_probabilities_or_SVI_labels": True, "seed": "deterministic SHA256 sampler; no stochastic fitting"}
