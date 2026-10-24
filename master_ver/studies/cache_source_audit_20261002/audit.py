"""Pure per-match checks and aggregate summaries for the immutable cache audit.

The audit deliberately reports support and missingness separately.  A cached
zero is counted as a value and is never converted into a fabricated presence
mask or a known absence claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from studies.anchor_frame_audit_20261002.contracts import (
    parse_game_end_events as _reference_game_end,
    parse_timestamp_ms as _reference_timestamp,
)

from . import config as C
from .champion_map import load_champion_map


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _json_object(data: bytes | None) -> tuple[Any, str | None]:
    if data is None:
        return None, "missing_bytes"
    try:
        return json.loads(data), None
    except Exception as exc:  # noqa: BLE001 - row-local parse evidence
        return None, f"json_parse_error:{type(exc).__name__}"


def _stat_signature(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _read_once(path: Path) -> tuple[bytes | None, dict[str, Any]]:
    """Read one immutable source file once and compare metadata around that read."""
    before = _stat_signature(path)
    if before is None:
        return None, {"path_exists": False, "stable": False, "error": "missing_file"}
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, {
            "path_exists": True,
            "stable": False,
            "error": f"read_error:{type(exc).__name__}",
            "before": before,
        }
    after = _stat_signature(path)
    stable = before == after and int(before["size"]) == len(data)
    info = {
        "path_exists": True,
        "stable": bool(stable),
        "before": before,
        "after": after,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if not stable:
        info["error"] = "file_metadata_changed_before_after"
    return data, info


def _field_counts(array: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_values": int(array.size),
        "n_finite": None,
        "n_nonfinite": None,
        "n_zero": None,
    }
    try:
        finite = np.isfinite(array)
        out["n_finite"] = int(np.count_nonzero(finite))
        out["n_nonfinite"] = int(finite.size - np.count_nonzero(finite))
        out["n_zero"] = int(np.count_nonzero(array == 0))
        if array.ndim == 1:
            out["channel_counts"] = {
                "n_values": [out["n_values"]],
                "n_finite": [out["n_finite"]],
                "n_nonfinite": [out["n_nonfinite"]],
                "n_zero": [out["n_zero"]],
            }
        elif array.ndim >= 2:
            axes = tuple(range(array.ndim - 1))
            out["channel_counts"] = {
                "n_values": np.ones(array.shape[-1], dtype=np.int64) * int(np.prod(array.shape[:-1], dtype=np.int64)),
                "n_finite": np.sum(finite, axis=axes, dtype=np.int64),
                "n_nonfinite": np.sum(~finite, axis=axes, dtype=np.int64),
                "n_zero": np.sum(array == 0, axis=axes, dtype=np.int64),
            }
            out["channel_counts"] = {
                metric: [int(value) for value in values]
                for metric, values in out["channel_counts"].items()
            }
    except (TypeError, ValueError):
        out["count_status"] = "unknown_non_numeric"
    return out


def _npz_audit(data: bytes | None, *, explicit_frame_ms: int | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    """Decode NPZ bytes without pickle and return (diagnostics, critical, unknown)."""
    out: dict[str, Any] = {
        "decode_ok": False,
        "keys": [],
        "missing_keys": list(C.EXPECTED_NPZ_KEYS),
        "extra_keys": [],
        "fields": {},
        "main_state_ok": False,
        "channel_alignment_ok": False,
        "frame_clock": {
            "present": False,
            "dtype_ok": False,
            "shape_ok": False,
            "n_frames": 0,
            "nonnegative": None,
            "unique": None,
            "strict_increasing": None,
            "integrity_ok": False,
            "explicit_frame_ms": explicit_frame_ms,
            "n_index_clock_mismatches": None,
            "index_clock_match_is_diagnostic_only": True,
        },
        "stored_positions": {"status": "unknown"},
        "gold_values": {"status": "unknown"},
    }
    critical: list[str] = []
    unknown: list[str] = []
    if data is None:
        return out, ["missing_npz_bytes"], ["npz_support_unknown"]

    arrays: dict[str, np.ndarray] = {}
    stream = BytesIO(data)
    try:
        with np.load(stream, allow_pickle=False) as archive:
            out["keys"] = sorted(str(key) for key in archive.files)
            out["missing_keys"] = sorted(set(C.EXPECTED_NPZ_KEYS) - set(out["keys"]))
            out["extra_keys"] = sorted(set(out["keys"]) - set(C.EXPECTED_NPZ_KEYS))
            out["decode_ok"] = True
            for key in C.EXPECTED_NPZ_KEYS:
                if key not in archive.files:
                    continue
                try:
                    arrays[key] = np.asarray(archive[key])
                except Exception as exc:  # noqa: BLE001 - preserve row failure
                    out["fields"][key] = {
                        "present": True,
                        "decode_ok": False,
                        "decode_error": type(exc).__name__,
                    }
                    critical.append(f"npz_field_decode_error:{key}")
    except Exception as exc:  # noqa: BLE001 - malformed ZIP/NPZ
        out["decode_error"] = type(exc).__name__
        critical.append(f"npz_decode_error:{type(exc).__name__}")
    finally:
        stream.close()

    if not out["decode_ok"]:
        return out, critical, unknown
    if out["missing_keys"] or out["extra_keys"]:
        critical.append("npz_key_schema_mismatch")

    minute = arrays.get("minute_ts")
    n_frames = int(minute.shape[0]) if minute is not None and minute.ndim >= 1 else None
    expected_shapes = {
        "minute_ts": None,
        "node_minute": (10, 76),
        "global_minute": (26,),
        "gold_team_minute": (2,),
        "xy_raw_minute": (10, 2),
    }
    shape_dtype_ok = True
    for key in C.EXPECTED_NPZ_KEYS:
        array = arrays.get(key)
        if array is None:
            out["fields"].setdefault(key, {"present": False})
            shape_dtype_ok = False
            continue
        expected_dtype = C.EXPECTED_NPZ_DTYPES[key]
        dtype_ok = str(array.dtype) == expected_dtype
        if key == "minute_ts":
            shape_ok = array.ndim == 1
        else:
            shape_ok = n_frames is not None and array.ndim == 3 - (key in {"global_minute", "gold_team_minute"}) and array.shape[0] == n_frames and tuple(array.shape[1:]) == expected_shapes[key]
        if key == "minute_ts":
            shape_ok = bool(shape_ok and (n_frames is not None))
        out["fields"][key] = {
            "present": True,
            "decode_ok": True,
            "dtype": str(array.dtype),
            "expected_dtype": expected_dtype,
            "dtype_ok": bool(dtype_ok),
            "shape": list(array.shape),
            "shape_ok": bool(shape_ok),
            **_field_counts(array),
        }
        shape_dtype_ok = bool(shape_dtype_ok and dtype_ok and shape_ok)
        if not dtype_ok:
            critical.append(f"npz_dtype_mismatch:{key}")
        if not shape_ok:
            critical.append(f"npz_shape_mismatch:{key}")
        if out["fields"][key].get("n_nonfinite", 0):
            unknown.append(f"npz_nonfinite_values:{key}")

    frame = out["frame_clock"]
    if minute is None:
        critical.append("missing_minute_ts")
    elif minute.ndim != 1 or str(minute.dtype) != C.EXPECTED_NPZ_DTYPES["minute_ts"]:
        critical.append("minute_ts_shape_or_dtype_invalid")
    elif minute.size == 0:
        frame.update({"present": True, "dtype_ok": True, "shape_ok": True, "n_frames": 0})
        critical.append("empty_minute_ts")
    else:
        values = np.asarray(minute, dtype=np.int64)
        nonnegative = bool(np.all(values >= 0))
        unique = bool(np.unique(values).size == values.size)
        strict = bool(values.size >= 2 and np.all(np.diff(values) > 0))
        frame.update({
            "present": True,
            "dtype_ok": True,
            "shape_ok": True,
            "n_frames": int(values.size),
            "min_ms": int(values.min()),
            "max_ms": int(values.max()),
            "nonnegative": nonnegative,
            "unique": unique,
            "strict_increasing": strict,
            "integrity_ok": bool(nonnegative and unique and strict),
        })
        if explicit_frame_ms is not None and explicit_frame_ms > 0:
            expected = np.arange(values.size, dtype=np.int64) * int(explicit_frame_ms)
            frame["n_index_clock_mismatches"] = int(np.count_nonzero(values != expected))
        if not nonnegative:
            critical.append("minute_ts_negative")
        if not unique:
            critical.append("minute_ts_duplicate")
        if not strict:
            critical.append("minute_ts_not_strictly_increasing")
    if minute is not None and minute.ndim == 1:
        n_frames = int(minute.shape[0])
        aligned = all(
            key in arrays and arrays[key].ndim >= 1 and int(arrays[key].shape[0]) == n_frames
            for key in C.EXPECTED_NPZ_KEYS
        )
        out["channel_alignment_ok"] = bool(aligned)
        if not aligned:
            critical.append("npz_channel_alignment_mismatch")

    xy = arrays.get("xy_raw_minute")
    if xy is not None and xy.ndim == 3 and xy.shape[1:] == (10, 2):
        finite_pair = np.isfinite(xy).all(axis=2)
        zero_pair = (xy == 0).all(axis=2)
        out["stored_positions"] = {
            "status": "observed_cache_values",
            "n_pairs": int(finite_pair.size),
            "n_pairs_finite": int(np.count_nonzero(finite_pair)),
            "n_pairs_nonfinite": int(finite_pair.size - np.count_nonzero(finite_pair)),
            "n_pairs_zero": int(np.count_nonzero(zero_pair)),
            "n_pairs_nonzero_finite": int(np.count_nonzero(finite_pair & ~zero_pair)),
            "zero_is_not_missingness_proof": True,
        }
    gold = arrays.get("gold_team_minute")
    if gold is not None and gold.ndim == 2 and gold.shape[1:] == (2,):
        finite_row = np.isfinite(gold).all(axis=1)
        zero_row = (gold == 0).all(axis=1)
        out["gold_values"] = {
            "status": "observed_cache_values",
            "n_frames": int(gold.shape[0]),
            "n_values": int(gold.size),
            "n_values_finite": int(np.count_nonzero(np.isfinite(gold))),
            "n_values_nonfinite": int(gold.size - np.count_nonzero(np.isfinite(gold))),
            "n_frames_finite": int(np.count_nonzero(finite_row)),
            "n_frames_zero": int(np.count_nonzero(zero_row)),
            "zero_is_not_missingness_proof": True,
        }

    out["main_state_ok"] = bool(
        out["decode_ok"]
        and not out["missing_keys"]
        and not out["extra_keys"]
        and shape_dtype_ok
        and out["frame_clock"]["integrity_ok"]
        and out["channel_alignment_ok"]
    )
    if not out["main_state_ok"]:
        critical.append("npz_main_state_invalid")
    return out, sorted(set(critical)), sorted(set(unknown))


def _event_array(payload: Any) -> tuple[list[Any] | None, str, list[str]]:
    """Accept only the observed flat list or documented timeline wrappers."""
    if isinstance(payload, list):
        return payload, "flat_list", []
    if not isinstance(payload, Mapping):
        return None, "unsupported", ["events_payload_not_flat_or_documented_wrapper"]
    events = payload.get("events")
    if isinstance(events, list):
        return events, "root_events_field", []
    info = payload.get("info")
    frames = info.get("frames") if isinstance(info, Mapping) else None
    if frames is None:
        frames = payload.get("frames")
    if isinstance(frames, list):
        flattened: list[Any] = []
        for frame in frames:
            if not isinstance(frame, Mapping) or not isinstance(frame.get("events"), list):
                return None, "timeline_frames", ["events_frame_missing_events_list"]
            flattened.extend(frame["events"])
        return flattened, "timeline_frames", []
    return None, "unsupported", ["events_payload_schema_unsupported"]


def _id_value_status(value: Any) -> str:
    number = _strict_int(value)
    if number is None:
        return "unknown"
    if number == 0:
        return "zero_or_environment"
    if 1 <= number <= 10:
        return "valid_player_id"
    return "out_of_roster"


def _kill_support(events: Sequence[Any]) -> dict[str, Any]:
    kills = [e for e in events if isinstance(e, Mapping) and e.get("type") == "CHAMPION_KILL"]
    counts = Counter()
    for event in kills:
        for field in ("killerId", "victimId"):
            if field not in event:
                counts[f"{field}_missing"] += 1
            else:
                counts[f"{field}_{_id_value_status(event.get(field))}"] += 1
        assists = event.get("assistingParticipantIds")
        if assists is None:
            counts["assists_missing"] += 1
        elif not isinstance(assists, list):
            counts["assists_malformed"] += 1
        else:
            counts["assists_present"] += 1
            counts["assist_ids_total"] += len(assists)
            counts["assist_ids_valid"] += sum(_id_value_status(value) == "valid_player_id" for value in assists)
            counts["assist_ids_unknown"] += sum(_id_value_status(value) == "unknown" for value in assists)
        position = event.get("position")
        if position is None:
            counts["position_missing"] += 1
        elif not isinstance(position, Mapping):
            counts["position_malformed"] += 1
        else:
            x, y = position.get("x"), position.get("y")
            if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool) and math.isfinite(float(x)) and math.isfinite(float(y)):
                counts["position_valid"] += 1
            else:
                counts["position_unknown"] += 1
    return {
        "n_kills": len(kills),
        "status": "pass" if kills and counts["killerId_valid_player_id"] == len(kills) and counts["victimId_valid_player_id"] == len(kills) else "unknown",
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "missingness_is_unknown": True,
    }


def _event_audit(payload: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    events, variant, schema_reasons = _event_array(payload)
    out: dict[str, Any] = {
        "decode_ok": events is not None,
        "schema_variant": variant,
        "n_events": 0 if events is None else len(events),
        "flat_records": False,
        "type_counts": {},
        "timestamp_valid": 0,
        "timestamp_invalid": 0,
        "timestamp_order_ok": None,
        "real_timestamp_present": 0,
        "real_timestamp_invalid": 0,
        "game_end": {},
        "terminal_frame_order": {"status": "unknown"},
        "kill_support": {"n_kills": 0, "status": "unknown", "counts": {}, "missingness_is_unknown": True},
        "record_keys": [],
    }
    critical = list(schema_reasons)
    unknown: list[str] = []
    if events is None:
        return out, sorted(set(critical)), ["events_support_unknown"]

    out["flat_records"] = all(isinstance(event, Mapping) for event in events)
    key_union: set[str] = set()
    for event in events:
        if isinstance(event, Mapping):
            key_union.update(str(key) for key in event)
    out["record_keys"] = sorted(key_union)
    if not out["flat_records"]:
        critical.append("events_contains_nonrecord")
    type_counts: Counter[str] = Counter()
    timestamps: list[int] = []
    timestamp_errors: Counter[str] = Counter()
    valid_type = 0
    for event in events:
        if not isinstance(event, Mapping):
            timestamp_errors["nonrecord"] += 1
            continue
        event_type = event.get("type")
        if isinstance(event_type, str) and event_type:
            valid_type += 1
            type_counts[event_type] += 1
        else:
            critical.append("event_type_missing_or_invalid")
        if "timestamp" not in event:
            timestamp_errors["missing_timestamp"] += 1
        else:
            timestamp, reason = _reference_timestamp(event.get("timestamp"))
            if reason or timestamp is None:
                timestamp_errors[reason or "invalid_timestamp"] += 1
            else:
                timestamps.append(int(timestamp))
        if "realTimestamp" in event:
            out["real_timestamp_present"] += 1
            real, reason = _reference_timestamp(event.get("realTimestamp"))
            if reason or real is None:
                out["real_timestamp_invalid"] += 1
    out["type_counts"] = {key: int(value) for key, value in sorted(type_counts.items())}
    out["timestamp_valid"] = len(timestamps)
    out["timestamp_invalid"] = int(sum(timestamp_errors.values()))
    out["timestamp_error_counts"] = {key: int(value) for key, value in sorted(timestamp_errors.items())}
    out["timestamp_min"] = min(timestamps) if timestamps else None
    out["timestamp_max"] = max(timestamps) if timestamps else None
    out["timestamp_order_ok"] = bool(not timestamp_errors and all(a <= b for a, b in zip(timestamps, timestamps[1:])))
    if timestamp_errors:
        critical.append("event_timestamp_invalid")
    if not out["timestamp_order_ok"]:
        critical.append("event_timestamp_order_invalid")
    if valid_type != len(events):
        critical.append("event_type_coverage_incomplete")

    end_objects: list[tuple[int, int | None, int | None]] = []
    end_malformed_timestamp = 0
    end_malformed_winner = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("type") != "GAME_END":
            continue
        timestamp, timestamp_reason = _reference_timestamp(event.get("timestamp")) if "timestamp" in event else (None, "missing_timestamp")
        winner = _strict_int(event.get("winningTeam"))
        winner_ok = winner in (100, 200)
        if timestamp_reason or timestamp is None:
            end_malformed_timestamp += 1
        if not winner_ok:
            end_malformed_winner += 1
        end_objects.append((index, timestamp if not timestamp_reason else None, winner if winner_ok else None))
    valid_ends = [entry for entry in end_objects if entry[1] is not None and entry[2] is not None]
    valid_times = sorted({int(entry[1]) for entry in valid_ends})
    single_end = end_objects[0] if len(end_objects) == 1 else None
    game_end_status = "pass" if len(end_objects) == 1 and len(valid_ends) == 1 and len(valid_times) == 1 else "unknown"
    if not end_objects:
        game_end_status = "fail"
    elif len(end_objects) > 1 or len(valid_times) > 1:
        game_end_status = "fail"
    elif single_end is not None and single_end[1] is not None and single_end[2] is None:
        game_end_status = "fail"
    out["game_end"] = {
        "n_objects": len(end_objects),
        "n_valid": len(valid_ends),
        "n_malformed_timestamp": end_malformed_timestamp,
        "n_malformed_winner": end_malformed_winner,
        "valid_timestamps": valid_times,
        "timestamp_ms": (
            valid_times[0]
            if game_end_status == "pass"
            else (int(single_end[1]) if single_end is not None and single_end[1] is not None else None)
        ),
        "winning_team": int(valid_ends[0][2]) if game_end_status == "pass" else None,
        "status": game_end_status,
        "unique_valid": bool(game_end_status == "pass"),
        "last_event_index": int(valid_ends[0][0]) if valid_ends else None,
        "event_is_last": bool(valid_ends and valid_ends[0][0] == len(events) - 1),
        "missingness_is_unknown": True,
    }
    reference = _reference_game_end(events)
    out["game_end"]["reference_parser"] = {
        "n_valid_timestamps": int(reference.get("n_game_end", 0)),
        "timestamp_ms": reference.get("game_end_ms"),
        "known": bool(reference.get("game_end_known", False)),
    }
    if game_end_status == "fail":
        if not end_objects:
            critical.append("game_end_missing")
        elif single_end is not None and single_end[1] is not None and single_end[2] is None:
            critical.append("game_end_winner_invalid")
        else:
            critical.append("game_end_nonunique_or_ambiguous")
    elif game_end_status != "pass":
        unknown.append("game_end_missing_malformed_or_winner_unknown")
    if game_end_status == "pass" and not out["game_end"]["event_is_last"]:
        critical.append("game_end_not_last_event")
    if out["real_timestamp_invalid"]:
        unknown.append("real_timestamp_invalid")
    out["kill_support"] = _kill_support(events)
    if out["kill_support"]["n_kills"] == 0:
        unknown.append("kill_support_no_champion_kills")
    return out, sorted(set(critical)), sorted(set(unknown))


def _team_map_summary(value: Any) -> dict[str, Any]:
    expected = {str(i) for i in range(1, 11)}
    if not isinstance(value, Mapping):
        return {"status": "unknown", "key_set_ok": False, "values_ok": False, "team_counts": {}, "n_keys": 0}
    keys = {str(key) for key in value}
    key_set_ok = keys == expected and all(str(key) == key for key in value)
    normalized: dict[str, int] = {}
    invalid_values = 0
    for key, raw in value.items():
        number = _strict_int(raw)
        if number not in (100, 200):
            invalid_values += 1
        elif str(key) in expected:
            normalized[str(key)] = int(number)
    counts = Counter(str(team) for team in normalized.values())
    values_ok = invalid_values == 0 and set(normalized) == expected and counts["100"] == 5 and counts["200"] == 5
    return {
        "status": "pass" if key_set_ok and values_ok else "fail",
        "key_set_ok": bool(key_set_ok),
        "values_ok": bool(values_ok),
        "n_keys": len(value),
        "team_counts": {key: int(value) for key, value in sorted(counts.items())},
        "invalid_value_count": int(invalid_values),
        "expected_pid_range": [1, 10],
        "expected_team_values": [100, 200],
    }


def _nested_id_counts(value: Any) -> dict[str, int]:
    counts = Counter()
    def visit(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for child_key, child_value in node.items():
                visit(child_value, str(child_key))
        elif isinstance(node, list):
            for child in node:
                visit(child, key)
        elif "id" in key.lower():
            number = _strict_int(node)
            if number is None:
                counts["invalid"] += 1
            elif number == 0:
                counts["zero"] += 1
            else:
                counts["nonzero"] += 1
    visit(value)
    return {key: int(value) for key, value in sorted(counts.items())}


def _direct_id_counts(value: Mapping[Any, Any]) -> dict[str, int]:
    """Count ID-valued entries whose parent mapping is the ID schema."""
    counts = Counter()
    for raw in value.values():
        number = _strict_int(raw)
        if number is None:
            counts["invalid"] += 1
        elif number == 0:
            counts["zero"] += 1
        else:
            counts["nonzero"] += 1
    return {key: int(value) for key, value in sorted(counts.items())}


def _bans_summary(value: Any) -> dict[str, Any]:
    expected = {"blue", "red"}
    out: dict[str, Any] = {
        "kind": type(value).__name__,
        "n_sides": 0,
        "n_bans": 0,
        "side_key_set_ok": False,
        "side_lists_ok": False,
        "structure_ok": False,
    }
    if not isinstance(value, Mapping):
        return out
    keys = {str(key) for key in value}
    key_types_ok = all(isinstance(key, str) for key in value)
    side_key_set_ok = key_types_ok and keys == expected
    side_lists_ok = side_key_set_ok and all(isinstance(value.get(side), list) for side in expected)
    out.update({
        "kind": "mapping",
        "n_sides": len(value),
        "n_bans": sum(len(value.get(side, [])) for side in expected if isinstance(value.get(side), list)),
        "side_key_set_ok": bool(side_key_set_ok),
        "side_lists_ok": bool(side_lists_ok),
        "structure_ok": bool(side_key_set_ok and side_lists_ok),
    })
    return out


def _static_meta_summary(value: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    required = ("champion_by_pid", "champion_name_by_pid", "summoner_spells_by_pid", "runes_by_pid", "bans")
    mapping_fields = required[:4]
    out: dict[str, Any] = {"status": "unknown", "schema_keys": [], "fields": {}}
    if not isinstance(value, Mapping):
        return out, ["static_meta_missing_or_invalid"], ["static_meta_support_unknown"]
    out["schema_keys"] = sorted(str(key) for key in value)
    missing = sorted(set(required) - set(out["schema_keys"]))
    if missing:
        out["missing_required_keys"] = missing
    for key in mapping_fields:
        field = value.get(key)
        if isinstance(field, Mapping):
            pid_keys = {str(item) for item in field}
            expected = {str(i) for i in range(1, 11)}
            pid_key_set_ok = all(isinstance(item, str) for item in field) and pid_keys == expected
            out["fields"][key] = {
                "kind": "mapping",
                "n_keys": len(field),
                "pid_key_count": len(pid_keys & expected),
                "pid_key_set_ok": bool(pid_key_set_ok),
                "id_counts": (
                    _direct_id_counts(field)
                    if key == "champion_by_pid"
                    else _nested_id_counts(field)
                ),
            }
        else:
            out["fields"][key] = {
                "kind": type(field).__name__,
                "n_keys": 0,
                "pid_key_count": 0,
                "pid_key_set_ok": False,
                "id_counts": _nested_id_counts(field) if isinstance(field, list) else {},
                "status": "invalid_mapping_schema",
            }
    out["fields"]["bans"] = _bans_summary(value.get("bans"))
    out["fields"]["bans"]["id_counts"] = _nested_id_counts(value.get("bans"))
    schema_ok = not missing and all(
        out["fields"].get(key, {}).get("pid_key_set_ok") is True for key in mapping_fields
    ) and out["fields"]["bans"].get("structure_ok") is True
    out["status"] = "pass" if schema_ok else "unknown"
    if not schema_ok:
        return out, [], ["static_meta_schema_or_pid_support_unknown"]
    return out, [], []


def _champion_consistency(static_meta: Any, patch: str) -> dict[str, Any]:
    """Compare cached participant IDs/names to the current API key/id map."""
    numeric_keys = (
        "n_participants", "n_pairs_present", "n_exact", "n_mismatch",
        "n_unknown_id", "n_unknown_name", "n_missing_id", "n_missing_name",
        "n_map_unavailable", "n_api_map_entries",
    )
    out: dict[str, Any] = {key: 0 for key in numeric_keys}
    out.update({"status": "unknown"})
    out["n_participants"] = 10
    mapping = load_champion_map(str(patch))
    out["n_api_map_entries"] = int(mapping.get("n_valid_id_entries", 0))
    id_to_names = mapping.get("id_to_names", {})
    name_to_ids = mapping.get("name_to_ids", {})
    if not isinstance(id_to_names, Mapping) or not isinstance(name_to_ids, Mapping) or mapping.get("load_status") != "pass":
        out["n_map_unavailable"] = 1

    ids = static_meta.get("champion_by_pid") if isinstance(static_meta, Mapping) else None
    names = static_meta.get("champion_name_by_pid") if isinstance(static_meta, Mapping) else None
    ids = ids if isinstance(ids, Mapping) else {}
    names = names if isinstance(names, Mapping) else {}
    for pid in range(1, 11):
        pid_key = str(pid)
        raw_id = ids.get(pid_key) if pid_key in ids else None
        raw_name = names.get(pid_key) if pid_key in names else None
        id_missing = pid_key not in ids or raw_id is None
        name_missing = pid_key not in names or not isinstance(raw_name, str) or not raw_name.strip()
        if id_missing:
            out["n_missing_id"] += 1
        if name_missing:
            out["n_missing_name"] += 1
        if id_missing or name_missing:
            continue
        out["n_pairs_present"] += 1
        champion_id = _strict_int(raw_id)
        canonical_name = raw_name.strip().casefold() if isinstance(raw_name, str) else None
        known_id = champion_id is not None and champion_id in id_to_names
        known_name = canonical_name is not None and canonical_name in name_to_ids
        if not known_id:
            out["n_unknown_id"] += 1
        if not known_name:
            out["n_unknown_name"] += 1
        if known_id and known_name:
            if canonical_name in set(id_to_names.get(champion_id, ())):
                out["n_exact"] += 1
            else:
                out["n_mismatch"] += 1
    out["status"] = "pass" if all(out[key] == 0 for key in (
        "n_mismatch", "n_unknown_id", "n_unknown_name", "n_missing_id", "n_missing_name", "n_map_unavailable"
    )) else "unknown"
    return out


def _source_clock_summary(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Recognize provenance only when a producer explicitly records it."""
    for key in ("source_clock_provenance", "clock_provenance", "source_clock_status"):
        value = meta.get(key)
        if isinstance(value, Mapping):
            explicit = any(value.get(flag) is True for flag in ("explicit", "recorded", "verified"))
            status = value.get("status")
            if explicit and isinstance(status, str) and status.strip():
                return {"status": "known", "field": key, "explicit": True, "value": status.strip()}
        elif isinstance(value, str) and value.strip().lower() in {"recorded", "explicit", "verified", "known"}:
            return {"status": "known", "field": key, "explicit": True, "value": value.strip()}
    return {"status": "unknown", "explicit": False, "reason": "source_clock_provenance_not_explicitly_recorded"}


def audit_match_bytes(
    match_id: str,
    expected_patch: str,
    *,
    npz_bytes: bytes | None,
    meta_bytes: bytes | None,
    events_bytes: bytes | None,
    source_stability: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit one row from already-read bytes, without a second source read."""
    canonical_id = str(match_id).strip()
    row: dict[str, Any] = {
        "match_key_sha256": sha256_bytes(canonical_id.encode("utf-8")),
        "patch": str(expected_patch),
        "role": C.PATCH_ROLES.get(str(expected_patch), "UNKNOWN"),
        "status": "unknown",
        "reasons": [],
        "source_hashes": {
            "npz": sha256_bytes(npz_bytes) if npz_bytes is not None else "",
            "meta": sha256_bytes(meta_bytes) if meta_bytes is not None else "",
            "events": sha256_bytes(events_bytes) if events_bytes is not None else "",
        },
        "training_ready": False,
        "final_ready": False,
    }
    critical: list[str] = []
    unknown: list[str] = []
    stability = dict(source_stability or {})
    if source_stability is not None:
        row["source_files_stable"] = bool(all(info.get("stable") is True for info in source_stability.values()))
        for label, info in source_stability.items():
            if info.get("stable") is not True:
                critical.append(f"source_file_unstable:{label}")
            if info.get("error") == "missing_file":
                critical.append(f"missing_{label}_file")
    else:
        row["source_files_stable"] = None

    meta, meta_error = _json_object(meta_bytes)
    if meta_error or not isinstance(meta, Mapping):
        critical.append(meta_error or "meta_payload_not_object")
        meta = {}
    row["meta_top_keys"] = sorted(str(key) for key in meta)
    meta_match_id = meta.get("match_id")
    row["metadata_identity"] = {
        "match_id_present": "match_id" in meta,
        "match_id_matches": bool(isinstance(meta_match_id, str) and meta_match_id == canonical_id),
        "expected_patch": str(expected_patch),
        "patch": meta.get("patch"),
        "patch_matches": bool(meta.get("patch") == expected_patch),
        "patch_full": meta.get("patch_full"),
        "patch_full_matches": bool(isinstance(meta.get("patch_full"), str) and (meta.get("patch_full") == expected_patch or meta.get("patch_full").startswith(expected_patch + "."))),
    }
    for name, ok in (
        ("metadata_match_id_mismatch", row["metadata_identity"]["match_id_matches"]),
        ("metadata_patch_mismatch", row["metadata_identity"]["patch_matches"]),
        ("metadata_patch_full_mismatch", row["metadata_identity"]["patch_full_matches"]),
    ):
        if not ok:
            critical.append(name)
    row["team_map"] = _team_map_summary(meta.get("team_map"))
    if row["team_map"]["status"] != "pass":
        critical.append("team_map_not_exact_5v5_pid_partition")
    row["static_metadata"], static_critical, static_unknown = _static_meta_summary(meta.get("static_meta"))
    critical.extend(static_critical)
    unknown.extend(static_unknown)
    row["static_champion_consistency"] = _champion_consistency(meta.get("static_meta"), str(expected_patch))
    if row["static_champion_consistency"]["status"] != "pass":
        unknown.append("static_champion_consistency_unknown")
    row["feature_version"] = meta.get("feature_version") if isinstance(meta.get("feature_version"), str) else None
    if row["feature_version"] is None:
        unknown.append("feature_version_missing_or_invalid")
    interp = meta.get("interp_policy")
    row["interp_policy"] = interp if isinstance(interp, Mapping) else None
    row["interp_policy_signature"] = sha256_bytes(_canonical_json(interp).encode("utf-8")) if isinstance(interp, Mapping) else None
    if row["interp_policy_signature"] is None:
        unknown.append("interp_policy_missing_or_invalid")
    row["source_clock"] = _source_clock_summary(meta)
    if row["source_clock"]["status"] != "known":
        unknown.append("source_clock_provenance_unknown")

    explicit_frame_ms = None
    if isinstance(interp, Mapping):
        frame_ms = _strict_int(interp.get("frame_ms"))
        explicit_frame_ms = frame_ms if frame_ms is not None and frame_ms > 0 else None
    row["npz"], npz_critical, npz_unknown = _npz_audit(npz_bytes, explicit_frame_ms=explicit_frame_ms)
    events_payload, events_error = _json_object(events_bytes) if events_bytes is not None else (None, "missing_events_bytes")
    row["events"], event_critical, event_unknown = _event_audit(events_payload)
    if events_error:
        event_critical.append(events_error)
    critical.extend(npz_critical)
    critical.extend(event_critical)
    unknown.extend(npz_unknown)
    unknown.extend(event_unknown)

    game_end = row["events"].get("game_end", {})
    frame = row["npz"].get("frame_clock", {})
    game_end_ms = game_end.get("timestamp_ms")
    last_frame_ms = frame.get("max_ms")
    terminal_gap_ms = (
        int(game_end_ms) - int(last_frame_ms)
        if isinstance(game_end_ms, int) and not isinstance(game_end_ms, bool)
        and isinstance(last_frame_ms, int) and not isinstance(last_frame_ms, bool)
        else None
    )
    terminal_details = {
        "first_frame_ms": frame.get("min_ms"),
        "last_frame_ms": last_frame_ms,
        "game_end_timestamp_ms": game_end_ms,
        "winning_team": game_end.get("winning_team"),
        "terminal_gap_ms": terminal_gap_ms,
    }
    if game_end.get("status") == "pass":
        ge_ms = int(game_end["timestamp_ms"])
        event_max = row["events"].get("timestamp_max")
        ge_order_ok = bool(event_max is None or event_max <= ge_ms)
        frame_order_ok = bool(frame.get("integrity_ok") and int(frame.get("max_ms", -1)) <= ge_ms)
        if ge_order_ok and frame_order_ok and row["events"].get("timestamp_order_ok"):
            row["events"]["terminal_frame_order"] = {
                "status": "pass", "game_end_ge_events": True, "game_end_ge_last_frame": True,
                **terminal_details,
            }
        else:
            row["events"]["terminal_frame_order"] = {
                "status": "fail", "game_end_ge_events": ge_order_ok, "game_end_ge_last_frame": frame_order_ok,
                **terminal_details,
            }
            critical.append("event_terminal_frame_order_invalid")
    else:
        row["events"]["terminal_frame_order"] = {
            "status": "unknown", "game_end_ge_events": None, "game_end_ge_last_frame": None,
            **terminal_details,
        }
        unknown.append("event_terminal_frame_order_unknown")

    if critical:
        row["status"] = "fail"
    elif unknown:
        row["status"] = "unknown"
    else:
        row["status"] = "pass"
    row["reasons"] = sorted(set(critical + unknown))
    row["critical_reason_count"] = len(set(critical))
    row["unknown_reason_count"] = len(set(unknown))
    row["canonical_key"] = f"{row['role']}:{row['match_key_sha256']}"
    return row


def audit_match_files(match_id: str, expected_patch: str, cache_dir: Path) -> dict[str, Any]:
    """Read exactly the three source files once and audit their decoded bytes."""
    base = Path(cache_dir)
    payloads: dict[str, bytes | None] = {}
    infos: dict[str, dict[str, Any]] = {}
    for label, suffix in (("npz", ".npz"), ("meta", ".meta.json"), ("events", ".events.json")):
        payloads[label], infos[label] = _read_once(base / f"{match_id}{suffix}")
    return audit_match_bytes(
        match_id,
        expected_patch,
        npz_bytes=payloads["npz"],
        meta_bytes=payloads["meta"],
        events_bytes=payloads["events"],
        source_stability=infos,
    )


def _counter_add(counter: Counter[str], value: Any) -> None:
    counter["unknown" if value is None else str(value)] += 1


def _channel_add(
    target: dict[str, dict[str, list[int]]],
    field_name: str,
    field: Mapping[str, Any],
    width_errors: Counter[str] | None = None,
    expected_widths: Mapping[str, int] | None = None,
) -> None:
    raw = field.get("channel_counts")
    if not isinstance(raw, Mapping):
        return
    provided = {
        metric: values
        for metric in ("n_values", "n_finite", "n_nonfinite", "n_zero")
        if isinstance(values := raw.get(metric), list)
    }
    widths = {len(values) for values in provided.values()}
    if len(widths) > 1:
        if width_errors is not None:
            width_errors[field_name] += 1
        return
    width = next(iter(widths), 0)
    expected_width = expected_widths.get(field_name) if expected_widths is not None else None
    if expected_width is not None and width != int(expected_width):
        if width_errors is not None:
            width_errors[field_name] += 1
        return
    destination = target.get(field_name, {})
    if any(metric in destination and len(destination[metric]) != width for metric in provided):
        if width_errors is not None:
            width_errors[field_name] += 1
        return
    for metric, values in provided.items():
        sums = destination.setdefault(metric, [0] * len(values))
        for index, value in enumerate(values):
            sums[index] += int(value)
    target[field_name] = destination


def _channel_names(field_name: str, width: int, feature_binding: Mapping[str, Any] | None = None) -> list[str]:
    if feature_binding:
        names = (feature_binding.get("name_lists") or {}).get(field_name)
        if isinstance(names, list) and len(names) == width:
            return [str(name) for name in names]
    fixed = {
        "minute_ts": ["frame_timestamp_ms"],
        "gold_team_minute": ["team_100", "team_200"],
        "xy_raw_minute": ["x_raw", "y_raw"],
    }.get(field_name)
    if fixed is not None and len(fixed) == width:
        return fixed
    return [f"{field_name}[{index}]" for index in range(width)]


def _named_channel_counts(
    channel_counts: Mapping[str, Mapping[str, list[int]]],
    feature_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name, metrics in sorted(channel_counts.items()):
        width = len(metrics.get("n_values", []))
        out[field_name] = {
            "channel_names": _channel_names(field_name, width, feature_binding),
            **{metric: list(values) for metric, values in sorted(metrics.items())},
        }
    return out


_CHAMPION_NUMERIC_KEYS = (
    "n_participants", "n_pairs_present", "n_exact", "n_mismatch",
    "n_unknown_id", "n_unknown_name", "n_missing_id", "n_missing_name",
    "n_map_unavailable", "n_api_map_entries",
)


def _champion_add(
    numeric: Counter[str],
    statuses: Counter[str],
    diagnostic: Mapping[str, Any],
) -> None:
    statuses[str(diagnostic.get("status", "unknown"))] += 1
    for key in _CHAMPION_NUMERIC_KEYS:
        value = diagnostic.get(key)
        if type(value) is int:
            numeric[key] += value


def _champion_payload(numeric: Counter[str], statuses: Counter[str]) -> dict[str, Any]:
    return {
        "status_counts": {key: int(value) for key, value in sorted(statuses.items())},
        "numeric_counts": {key: int(value) for key, value in sorted(numeric.items())},
    }


def _extrema_add(target: dict[str, int | None], value: Any) -> None:
    if type(value) is not int:
        return
    target["n"] = int(target.get("n") or 0) + 1
    target["min"] = value if target.get("min") is None else min(int(target["min"]), value)
    target["max"] = value if target.get("max") is None else max(int(target["max"]), value)


def _terminal_add(
    status_counts: Counter[str],
    winner_counts: Counter[str],
    gap_counts: Counter[str],
    frame_extrema: dict[str, dict[str, int | None]],
    kill_statuses: Counter[str],
    kill_counts: Counter[str],
    row: Mapping[str, Any],
) -> None:
    events = row.get("events", {}) or {}
    terminal = events.get("terminal_frame_order", {}) or {}
    status_counts[str(terminal.get("status", "unknown"))] += 1
    winner = terminal.get("winning_team")
    if type(winner) is int:
        winner_counts[str(winner)] += 1
    gap = terminal.get("terminal_gap_ms")
    if type(gap) is int:
        gap_counts[str(gap)] += 1
    frame = row.get("npz", {}).get("frame_clock", {}) or {}
    _extrema_add(frame_extrema["first_frame_ms"], frame.get("min_ms"))
    _extrema_add(frame_extrema["last_frame_ms"], frame.get("max_ms"))
    kill = events.get("kill_support", {}) or {}
    kill_statuses[str(kill.get("status", "unknown"))] += 1
    if type(kill.get("n_kills")) is int:
        kill_counts["n_kills"] += int(kill["n_kills"])
    for key, value in (kill.get("counts", {}) or {}).items():
        if type(value) is int:
            kill_counts[str(key)] += int(value)


def _terminal_payload(
    status_counts: Counter[str],
    winner_counts: Counter[str],
    gap_counts: Counter[str],
    frame_extrema: Mapping[str, Mapping[str, int | None]],
    kill_statuses: Counter[str],
    kill_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "status_counts": {key: int(value) for key, value in sorted(status_counts.items())},
        "winner_counts": {key: int(value) for key, value in sorted(winner_counts.items())},
        "terminal_gap_ms_counts": {key: int(value) for key, value in sorted(gap_counts.items())},
        "frame_extrema": {key: dict(value) for key, value in sorted(frame_extrema.items())},
        "kill_support_status_counts": {key: int(value) for key, value in sorted(kill_statuses.items())},
        "kill_support_numeric_counts": {key: int(value) for key, value in sorted(kill_counts.items())},
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_index_counts: Mapping[str, int] | None = None,
    input_support: Mapping[str, Any] | None = None,
    feature_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic aggregate diagnostics from row-local audit records."""
    status_counts: Counter[str] = Counter()
    role_status: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    patch_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    interp_counts: Counter[str] = Counter()
    source_clock_counts: Counter[str] = Counter()
    static_schema_counts: Counter[str] = Counter()
    static_id_counts: dict[str, Counter[str]] = defaultdict(Counter)
    champion_numeric: Counter[str] = Counter()
    champion_statuses: Counter[str] = Counter()
    champion_by_patch_numeric: dict[str, Counter[str]] = defaultdict(Counter)
    champion_by_patch_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    terminal_statuses: Counter[str] = Counter()
    terminal_winners: Counter[str] = Counter()
    terminal_gaps: Counter[str] = Counter()
    frame_extrema: dict[str, dict[str, int | None]] = {
        "first_frame_ms": {"n": 0, "min": None, "max": None},
        "last_frame_ms": {"n": 0, "min": None, "max": None},
    }
    kill_statuses: Counter[str] = Counter()
    kill_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    channel_counts: dict[str, dict[str, list[int]]] = {}
    channel_width_errors: Counter[str] = Counter()
    source_hash_presence: Counter[str] = Counter()
    key_roles: dict[str, set[str]] = defaultdict(set)
    key_counts: Counter[str] = Counter()
    samples: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        role = str(row.get("role", "UNKNOWN"))
        status = str(row.get("status", "unknown"))
        status_counts[status] += 1
        role_status[role][status] += 1
        patch_counts[str(row.get("patch"))] += 1
        for reason in row.get("reasons", []):
            reason_counts[str(reason)] += 1
        _counter_add(feature_counts, row.get("feature_version"))
        _counter_add(interp_counts, row.get("interp_policy_signature"))
        _counter_add(source_clock_counts, (row.get("source_clock") or {}).get("status"))
        static_schema_counts[str((row.get("static_metadata") or {}).get("status", "unknown"))] += 1
        for field_name, field in (row.get("static_metadata", {}).get("fields", {}) or {}).items():
            for metric, value in (field.get("id_counts", {}) or {}).items():
                static_id_counts[str(field_name)][str(metric)] += int(value)
        champion_diagnostic = row.get("static_champion_consistency") or {}
        _champion_add(champion_numeric, champion_statuses, champion_diagnostic)
        _champion_add(champion_by_patch_numeric[str(row.get("patch"))], champion_by_patch_statuses[str(row.get("patch"))], champion_diagnostic)
        _terminal_add(terminal_statuses, terminal_winners, terminal_gaps, frame_extrema, kill_statuses, kill_counts, row)
        for event_type, count in (row.get("events", {}).get("type_counts", {}) or {}).items():
            event_type_counts[str(event_type)] += int(count)
        for name, field in (row.get("npz", {}).get("fields", {}) or {}).items():
            for metric in ("n_values", "n_finite", "n_nonfinite", "n_zero"):
                value = field.get(metric)
                if isinstance(value, int):
                    field_counts[str(name)][metric] += int(value)
            _channel_add(channel_counts, str(name), field, channel_width_errors, C.EXPECTED_CHANNEL_WIDTHS)
        for label, digest in (row.get("source_hashes", {}) or {}).items():
            source_hash_presence[label] += bool(digest)
        key = str(row.get("match_key_sha256", ""))
        if key:
            key_roles[key].add(role)
            key_counts[key] += 1
        if role not in samples:
            samples[role] = {
                "match_key_sha256": key,
                "npz_keys": row.get("npz", {}).get("keys", []),
                "npz_fields": {
                    name: {"shape": field.get("shape"), "dtype": field.get("dtype")}
                    for name, field in (row.get("npz", {}).get("fields", {}) or {}).items()
                },
                "event_schema_variant": row.get("events", {}).get("schema_variant"),
                "event_record_keys": row.get("events", {}).get("record_keys"),
                "meta_top_keys": row.get("meta_top_keys", []),
            }
    duplicate_keys = sum(1 for count in key_counts.values() if count > 1)
    multi_role_keys = sum(1 for roles in key_roles.values() if len(roles) > 1)
    unique_key_count = len(key_roles)
    expected = dict(expected_index_counts or {})
    split_counts = {role: int(patch_counts.get(patch, 0)) for patch, role in C.PATCH_ROLES.items()}
    split_ok = not expected or all(patch_counts.get(patch, 0) == int(count) for patch, count in expected.items())
    return {
        "n_rows": len(rows),
        "status_counts": {key: int(value) for key, value in sorted(status_counts.items())},
        "role_status_counts": {role: {key: int(value) for key, value in sorted(counter.items())} for role, counter in sorted(role_status.items())},
        "patch_counts": {key: int(value) for key, value in sorted(patch_counts.items())},
        "expected_patch_counts": {key: int(value) for key, value in sorted(expected.items())},
        "split_counts": split_counts,
        "split_counts_match_expected": bool(split_ok),
        "disjointness": {
            "canonical_key_scheme": "sha256(strip(match_id)) with role partitions",
            "n_unique_match_keys": unique_key_count,
            "n_duplicate_canonical_keys": duplicate_keys,
            "n_keys_in_multiple_roles": multi_role_keys,
            "ok": bool(duplicate_keys == 0),
        },
        "reason_counts": {key: int(value) for key, value in sorted(reason_counts.items())},
        "feature_version_counts": {key: int(value) for key, value in sorted(feature_counts.items())},
        "interp_policy_signature_counts": {key: int(value) for key, value in sorted(interp_counts.items())},
        "source_clock_provenance_counts": {key: int(value) for key, value in sorted(source_clock_counts.items())},
        "static_metadata_schema_counts": {key: int(value) for key, value in sorted(static_schema_counts.items())},
        "static_metadata_id_counts": {key: {metric: int(value) for metric, value in sorted(counter.items())} for key, counter in sorted(static_id_counts.items())},
        "static_champion_consistency": {
            "overall": _champion_payload(champion_numeric, champion_statuses),
            "by_patch": {
                patch: _champion_payload(champion_by_patch_numeric[patch], champion_by_patch_statuses[patch])
                for patch in sorted(champion_by_patch_numeric)
            },
        },
        "terminal_and_kill_diagnostics": _terminal_payload(
            terminal_statuses, terminal_winners, terminal_gaps, frame_extrema, kill_statuses, kill_counts
        ),
        "event_type_counts": {key: int(value) for key, value in sorted(event_type_counts.items())},
        "npz_field_counts": {key: {metric: int(value) for metric, value in sorted(counter.items())} for key, counter in sorted(field_counts.items())},
        "npz_channel_counts": _named_channel_counts(channel_counts, feature_binding),
        "npz_channel_width_mismatch_counts": {key: int(value) for key, value in sorted(channel_width_errors.items())},
        "source_hash_presence_counts": {key: int(value) for key, value in sorted(source_hash_presence.items())},
        "schema_samples_one_per_role": samples,
        "input_support": dict(input_support or {}),
        "zero_is_not_missingness_proof": True,
        "training_ready": False,
        "final_ready": False,
    }


class AggregateAccumulator:
    """Bounded in-memory aggregate state for the full-corpus runner."""

    def __init__(self) -> None:
        self.n_rows = 0
        self.status_counts: Counter[str] = Counter()
        self.role_status: dict[str, Counter[str]] = defaultdict(Counter)
        self.reason_counts: Counter[str] = Counter()
        self.patch_counts: Counter[str] = Counter()
        self.feature_counts: Counter[str] = Counter()
        self.interp_counts: Counter[str] = Counter()
        self.source_clock_counts: Counter[str] = Counter()
        self.static_schema_counts: Counter[str] = Counter()
        self.static_id_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.champion_numeric: Counter[str] = Counter()
        self.champion_statuses: Counter[str] = Counter()
        self.champion_by_patch_numeric: dict[str, Counter[str]] = defaultdict(Counter)
        self.champion_by_patch_statuses: dict[str, Counter[str]] = defaultdict(Counter)
        self.terminal_statuses: Counter[str] = Counter()
        self.terminal_winners: Counter[str] = Counter()
        self.terminal_gaps: Counter[str] = Counter()
        self.frame_extrema: dict[str, dict[str, int | None]] = {
            "first_frame_ms": {"n": 0, "min": None, "max": None},
            "last_frame_ms": {"n": 0, "min": None, "max": None},
        }
        self.kill_statuses: Counter[str] = Counter()
        self.kill_counts: Counter[str] = Counter()
        self.event_type_counts: Counter[str] = Counter()
        self.field_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.channel_counts: dict[str, dict[str, list[int]]] = {}
        self.channel_width_errors: Counter[str] = Counter()
        self.source_hash_presence: Counter[str] = Counter()
        self.key_roles: dict[str, set[str]] = defaultdict(set)
        self.key_counts: Counter[str] = Counter()
        self.samples: dict[str, Mapping[str, Any]] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        self.n_rows += 1
        role = str(row.get("role", "UNKNOWN"))
        status = str(row.get("status", "unknown"))
        self.status_counts[status] += 1
        self.role_status[role][status] += 1
        self.patch_counts[str(row.get("patch"))] += 1
        for reason in row.get("reasons", []):
            self.reason_counts[str(reason)] += 1
        _counter_add(self.feature_counts, row.get("feature_version"))
        _counter_add(self.interp_counts, row.get("interp_policy_signature"))
        _counter_add(self.source_clock_counts, (row.get("source_clock") or {}).get("status"))
        self.static_schema_counts[str((row.get("static_metadata") or {}).get("status", "unknown"))] += 1
        for field_name, field in (row.get("static_metadata", {}).get("fields", {}) or {}).items():
            for metric, value in (field.get("id_counts", {}) or {}).items():
                self.static_id_counts[str(field_name)][str(metric)] += int(value)
        champion_diagnostic = row.get("static_champion_consistency") or {}
        _champion_add(self.champion_numeric, self.champion_statuses, champion_diagnostic)
        _champion_add(self.champion_by_patch_numeric[str(row.get("patch"))], self.champion_by_patch_statuses[str(row.get("patch"))], champion_diagnostic)
        _terminal_add(
            self.terminal_statuses, self.terminal_winners, self.terminal_gaps,
            self.frame_extrema, self.kill_statuses, self.kill_counts, row,
        )
        for event_type, count in (row.get("events", {}).get("type_counts", {}) or {}).items():
            self.event_type_counts[str(event_type)] += int(count)
        for name, field in (row.get("npz", {}).get("fields", {}) or {}).items():
            for metric in ("n_values", "n_finite", "n_nonfinite", "n_zero"):
                value = field.get(metric)
                if isinstance(value, int):
                    self.field_counts[str(name)][metric] += int(value)
            _channel_add(self.channel_counts, str(name), field, self.channel_width_errors, C.EXPECTED_CHANNEL_WIDTHS)
        for label, digest in (row.get("source_hashes", {}) or {}).items():
            self.source_hash_presence[label] += bool(digest)
        key = str(row.get("match_key_sha256", ""))
        if key:
            self.key_roles[key].add(role)
            self.key_counts[key] += 1
        if role not in self.samples:
            self.samples[role] = {
                "match_key_sha256": key,
                "npz_keys": row.get("npz", {}).get("keys", []),
                "npz_fields": {
                    name: {"shape": field.get("shape"), "dtype": field.get("dtype")}
                    for name, field in (row.get("npz", {}).get("fields", {}) or {}).items()
                },
                "event_schema_variant": row.get("events", {}).get("schema_variant"),
                "event_record_keys": row.get("events", {}).get("record_keys", []),
                "meta_top_keys": row.get("meta_top_keys", []),
            }

    def finish(
        self,
        *,
        expected_index_counts: Mapping[str, int] | None = None,
        input_support: Mapping[str, Any] | None = None,
        feature_binding: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected = dict(expected_index_counts or {})
        duplicate_keys = sum(1 for count in self.key_counts.values() if count > 1)
        multi_role_keys = sum(1 for roles in self.key_roles.values() if len(roles) > 1)
        split_counts = {role: int(self.patch_counts.get(patch, 0)) for patch, role in C.PATCH_ROLES.items()}
        split_ok = not expected or all(self.patch_counts.get(patch, 0) == int(count) for patch, count in expected.items())
        return {
            "n_rows": int(self.n_rows),
            "status_counts": {key: int(value) for key, value in sorted(self.status_counts.items())},
            "role_status_counts": {role: {key: int(value) for key, value in sorted(counter.items())} for role, counter in sorted(self.role_status.items())},
            "patch_counts": {key: int(value) for key, value in sorted(self.patch_counts.items())},
            "expected_patch_counts": {key: int(value) for key, value in sorted(expected.items())},
            "split_counts": split_counts,
            "split_counts_match_expected": bool(split_ok),
            "disjointness": {
                "canonical_key_scheme": "sha256(strip(match_id)) with role partitions",
                "n_unique_match_keys": len(self.key_roles),
                "n_duplicate_canonical_keys": duplicate_keys,
                "n_keys_in_multiple_roles": multi_role_keys,
                "ok": bool(duplicate_keys == 0),
            },
            "reason_counts": {key: int(value) for key, value in sorted(self.reason_counts.items())},
            "feature_version_counts": {key: int(value) for key, value in sorted(self.feature_counts.items())},
            "interp_policy_signature_counts": {key: int(value) for key, value in sorted(self.interp_counts.items())},
            "source_clock_provenance_counts": {key: int(value) for key, value in sorted(self.source_clock_counts.items())},
            "static_metadata_schema_counts": {key: int(value) for key, value in sorted(self.static_schema_counts.items())},
            "static_metadata_id_counts": {key: {metric: int(value) for metric, value in sorted(counter.items())} for key, counter in sorted(self.static_id_counts.items())},
            "static_champion_consistency": {
                "overall": _champion_payload(self.champion_numeric, self.champion_statuses),
                "by_patch": {
                    patch: _champion_payload(self.champion_by_patch_numeric[patch], self.champion_by_patch_statuses[patch])
                    for patch in sorted(self.champion_by_patch_numeric)
                },
            },
            "terminal_and_kill_diagnostics": _terminal_payload(
                self.terminal_statuses, self.terminal_winners, self.terminal_gaps,
                self.frame_extrema, self.kill_statuses, self.kill_counts,
            ),
            "event_type_counts": {key: int(value) for key, value in sorted(self.event_type_counts.items())},
            "npz_field_counts": {key: {metric: int(value) for metric, value in sorted(counter.items())} for key, counter in sorted(self.field_counts.items())},
            "npz_channel_counts": _named_channel_counts(self.channel_counts, feature_binding),
            "npz_channel_width_mismatch_counts": {key: int(value) for key, value in sorted(self.channel_width_errors.items())},
            "source_hash_presence_counts": {key: int(value) for key, value in sorted(self.source_hash_presence.items())},
            "schema_samples_one_per_role": self.samples,
            "input_support": dict(input_support or {}),
            "zero_is_not_missingness_proof": True,
            "training_ready": False,
            "final_ready": False,
        }
