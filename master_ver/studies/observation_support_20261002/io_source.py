"""Read-only loaders, identity checks, and raw-source availability census."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import config as C
from . import contracts as CT


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return file_sha256(path)


def anonymous_match_key(match_local: int) -> str:
    return f"m{int(match_local):05d}"


def verify_baseline_identity() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    man_sha = file_sha256(C.ANCHOR_MANIFEST)
    checks["manifest_sha256"] = man_sha
    checks["manifest_sha256_ok"] = man_sha == C.EXPECTED_ANCHOR_MANIFEST_SHA256
    man = load_json(C.ANCHOR_MANIFEST)
    checks["manifest_complete"] = bool(man.get("complete"))
    checks["gates_all_ok"] = bool(man.get("identity", {}).get("all_ok"))

    for label, path, expected in (
        ("components", C.COMPONENTS_NPZ, C.EXPECTED_COMPONENTS_SHA256),
        ("terminals", C.TERMINALS_JSON, C.EXPECTED_TERMINALS_SHA256),
        ("match_ids", C.ST_SOURCE / "train_match_ids.json", C.EXPECTED_MATCH_IDS_SHA256),
        ("match_meta", C.ST_SOURCE / "train_match_meta.npz", C.EXPECTED_MATCH_META_SHA256),
        ("ledger", C.ST_SOURCE / "event_ledger_train.npz", C.EXPECTED_LEDGER_SHA256),
        ("ranges", C.ST_SOURCE / "ledger_match_ranges.npz", C.EXPECTED_RANGES_SHA256),
        ("champion_full", C.CHAMPION_FULL, C.EXPECTED_CHAMPION_FULL_SHA256),
    ):
        exists = path.is_file()
        sha = file_sha256(path) if exists else ""
        checks[f"{label}_exists"] = exists
        checks[f"{label}_sha256"] = sha
        checks[f"{label}_sha256_ok"] = sha == expected
        if label == "champion_full" and exists:
            checks["champion_full_bytes"] = path.stat().st_size
            checks["champion_full_bytes_ok"] = path.stat().st_size == C.EXPECTED_CHAMPION_FULL_BYTES

    # Cross-check against anchor-recorded source hashes when present.
    src = man.get("source_hashes") or {}
    for key, field in (
        ("match_ids_sha256", "match_ids_sha256"),
        ("match_meta_sha256", "match_meta_sha256"),
        ("ledger_sha256", "ledger_sha256"),
        ("ranges_sha256", "ranges_sha256"),
    ):
        expected_src = str(src.get(field) or "")
        actual = checks.get(key.replace("match_meta", "match_meta").split("_sha")[0] + "_sha256", "")
        # remap
    checks["anchor_match_ids_sha_match"] = checks["match_ids_sha256"] == str(src.get("match_ids_sha256") or "")
    checks["anchor_match_meta_sha_match"] = checks["match_meta_sha256"] == str(src.get("match_meta_sha256") or "")
    checks["anchor_ledger_sha_match"] = checks["ledger_sha256"] == str(src.get("ledger_sha256") or "")
    checks["anchor_ranges_sha_match"] = checks["ranges_sha256"] == str(src.get("ranges_sha256") or "")
    checks["anchor_components_sha_match"] = checks["components_sha256"] == str(man.get("components_sha256") or "")
    checks["anchor_terminals_sha_match"] = checks["terminals_sha256"] == str(man.get("terminals_sha256") or "")

    bool_keys = [k for k, v in checks.items() if isinstance(v, bool) and k.endswith("_ok") or k.endswith("_match") or k in ("manifest_complete", "gates_all_ok")]
    # Explicit all_ok
    checks["all_ok"] = all(
        bool(checks[k])
        for k in (
            "manifest_sha256_ok",
            "manifest_complete",
            "gates_all_ok",
            "components_sha256_ok",
            "terminals_sha256_ok",
            "match_ids_sha256_ok",
            "match_meta_sha256_ok",
            "ledger_sha256_ok",
            "ranges_sha256_ok",
            "champion_full_sha256_ok",
            "champion_full_bytes_ok",
            "anchor_match_ids_sha_match",
            "anchor_match_meta_sha_match",
            "anchor_ledger_sha_match",
            "anchor_ranges_sha_match",
            "anchor_components_sha_match",
            "anchor_terminals_sha_match",
        )
    )
    return checks


def load_match_ids() -> list[str]:
    ids = load_json(C.ST_SOURCE / "train_match_ids.json")
    if len(ids) != C.EXPECTED_TRAIN_MATCHES:
        raise RuntimeError(f"match id count {len(ids)} != {C.EXPECTED_TRAIN_MATCHES}")
    if len(set(map(str, ids))) != len(ids):
        raise RuntimeError("duplicate frozen match IDs")
    return [str(x) for x in ids]


def load_meta_blue_pids() -> np.ndarray:
    with np.load(C.ST_SOURCE / "train_match_meta.npz", allow_pickle=False) as z:
        return np.asarray(z["blue_pids"], dtype=np.int16)


def load_ledger_and_ranges() -> dict[str, Any]:
    ledger = np.load(C.ST_SOURCE / "event_ledger_train.npz", allow_pickle=False)
    ranges = np.load(C.ST_SOURCE / "ledger_match_ranges.npz", allow_pickle=False)
    out = {
        "t_ms": np.asarray(ledger["t_ms"], dtype=np.int64),
        "x": np.asarray(ledger["x"], dtype=np.float64),
        "y": np.asarray(ledger["y"], dtype=np.float64),
        "killer": np.asarray(ledger["killer"], dtype=np.int64),
        "victim": np.asarray(ledger["victim"], dtype=np.int64),
        "assist_mask": np.asarray(ledger["assist_mask"], dtype=np.int64),
        "match_local": np.asarray(ledger["match_local"], dtype=np.int64),
        "starts": np.asarray(ranges["starts"], dtype=np.int64),
        "ends": np.asarray(ranges["ends"], dtype=np.int64),
    }
    if out["t_ms"].shape[0] != C.EXPECTED_TRAIN_KILLS:
        raise RuntimeError("ledger kill count mismatch")
    if out["starts"].shape[0] != C.EXPECTED_TRAIN_MATCHES:
        raise RuntimeError("ranges match count mismatch")
    validate_ledger_ranges(out["starts"], out["ends"], out["match_local"],
                           C.EXPECTED_TRAIN_KILLS, C.EXPECTED_TRAIN_MATCHES, C.EXPECTED_ZERO_KILL)
    return out


def validate_ledger_ranges(starts, ends, event_match_local, n_kills, n_matches, n_zero):
    """Frozen ledger stores zero-kill ranges as (-1,-1), with no event slice."""
    if starts.shape != (n_matches,) or ends.shape != starts.shape or event_match_local.shape != (n_kills,):
        raise RuntimeError("ledger/range schema mismatch")
    nonempty = ends > starts
    empty = ~nonempty
    if (int(empty.sum()) != n_zero or np.any(starts[empty] != -1) or np.any(ends[empty] != -1)):
        raise RuntimeError("zero-kill ranges must retain the source (-1,-1) sentinel")
    positive_starts, positive_ends = starts[nonempty], ends[nonempty]
    if (not positive_starts.size or positive_starts[0] != 0 or positive_ends[-1] != n_kills
            or not np.array_equal(positive_starts[1:], positive_ends[:-1])
            or np.any(positive_starts < 0) or np.any(positive_ends > n_kills)):
        raise RuntimeError("nonempty ledger ranges do not partition every frozen event row")
    expected_keys = np.repeat(np.flatnonzero(nonempty), positive_ends - positive_starts)
    if not np.array_equal(expected_keys, event_match_local):
        raise RuntimeError("source event match keys differ from range assignments")


def load_baseline_components(keys: list[str] | None = None) -> dict[str, np.ndarray]:
    default = [
        "match_local",
        "component_id",
        "n_members",
        "f",
        "L",
        "tau",
        "span_ms",
        "same_ms_first_n",
        "F_inclusive",
        "delta",
        "t_in",
        "F_strict",
        "a_strict",
        "exact_10000",
        "strict_eligible",
        "negative_tau",
        "singleton",
        "C0_n_blue",
        "C0_n_red",
        "C0_n_unknown",
        "C0_n_credited",
        "C0_n_env_killer",
        "C0_scale_max_side",
        "Call_n_blue",
        "Call_n_red",
        "Call_n_unknown",
        "Call_n_credited",
        "Call_n_env_killer",
        "Call_scale_max_side",
        "foreign_kills_in_tau_L",
        "foreign_near",
        "foreign_distant",
        "other_anchors_in_tau_L",
        "last_frame_ms",
        "game_end_ms",
        "game_end_known",
        "e_h60",
        "valid_h60",
        "followup_ms_h60",
        "e_h90",
        "valid_h90",
        "followup_ms_h90",
        "e_h120",
        "valid_h120",
        "followup_ms_h120",
        "next_component_tau",
        "next_kill_after_L",
        "overlap_next_tau_le_L",
        "missing_terminal_game_end",
        "terminal_support_authoritative",
        "anon_match_key",
        "member_key",
    ]
    use = keys or default
    # Trusted local research artifact; string identity fields are object arrays.
    with np.load(C.COMPONENTS_NPZ, allow_pickle=True) as z:
        missing = [k for k in use if k not in z.files]
        if missing:
            raise RuntimeError(f"components missing fields: {missing}")
        out = {}
        for k in use:
            arr = np.asarray(z[k])
            if arr.dtype == object:
                out[k] = arr.astype("U64")
            else:
                out[k] = arr
    if int(out["match_local"].shape[0]) != C.EXPECTED_MIN1_COMPONENTS:
        raise RuntimeError("component count mismatch")
    if not bool(np.all(np.diff(out["match_local"]) >= 0)):
        raise RuntimeError("components not sorted by match_local")
    return out


def component_match_ranges(match_local: np.ndarray, n_matches: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.zeros(n_matches, dtype=np.int64)
    ends = np.zeros(n_matches, dtype=np.int64)
    if match_local.size == 0:
        return starts, ends
    uniq, idx, counts = np.unique(match_local, return_index=True, return_counts=True)
    for u, s, c in zip(uniq.tolist(), idx.tolist(), counts.tolist()):
        u = int(u)
        if 0 <= u < n_matches:
            starts[u] = int(s)
            ends[u] = int(s) + int(c)
    return starts, ends


def list_batch_ids() -> list[int]:
    ids = []
    for p in C.ST_BATCHES.glob("batch_*.json"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def _read_batch_sidecar_header(js_path: Path) -> dict[str, Any]:
    import re

    with open(js_path, "r", encoding="utf-8") as f:
        head = f.read(8192)

    def _grab(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', head)
        return m.group(1) if m else None

    def _grab_int(key: str) -> int | None:
        m = re.search(rf'"{key}"\s*:\s*(-?\d+)', head)
        return int(m.group(1)) if m else None

    out = {
        "npz_sha256": _grab("npz_sha256"),
        "n_event_rows": _grab_int("n_event_rows"),
        "identity_source_kills": _grab("source_kills_sha256"),
    }
    if out["npz_sha256"] is None or out["n_event_rows"] is None or out["identity_source_kills"] is None:
        js = load_json(js_path)
        return {
            "npz_sha256": js["npz_sha256"],
            "n_event_rows": int(js["n_event_rows"]),
            "identity_source_kills": js["identity"]["source_kills_sha256"],
        }
    return out


def load_adopted_labels_for_batch(batch_id: int) -> dict[str, Any]:
    npz_path = C.ST_BATCHES / f"batch_{batch_id:04d}.npz"
    js_path = C.ST_BATCHES / f"batch_{batch_id:04d}.json"
    js = _read_batch_sidecar_header(js_path)
    before = file_sha256(npz_path)
    if before != js["npz_sha256"]:
        raise RuntimeError(f"batch {batch_id} changed before reading")
    with np.load(npz_path, allow_pickle=False) as z:
        match_locals = np.asarray(z["match_locals"], dtype=np.int32)
        labels = np.asarray(z[f"labels_{C.ADOPTED_CFG}"], dtype=np.int64)
    if file_sha256(npz_path) != before:
        raise RuntimeError(f"batch {batch_id} changed while reading")
    return {
        "batch_id": batch_id,
        "match_locals": match_locals,
        "labels": labels,
        "n_event_rows": int(js["n_event_rows"]),
        "npz_sha256": js["npz_sha256"],
        "identity_source_kills": js["identity_source_kills"],
    }


def build_full_labels(n_kills: int, n_matches: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    labels = np.full(n_kills, -3, dtype=np.int64)
    seen = np.zeros(n_matches, dtype=np.int8)
    for bid in list_batch_ids():
        b = load_adopted_labels_for_batch(bid)
        if b["identity_source_kills"] != C.EXPECTED_KILLS_SHA256:
            raise RuntimeError(f"batch {bid} kills sha mismatch")
        actual = file_sha256(C.ST_BATCHES / f"batch_{bid:04d}.npz")
        if actual != b["npz_sha256"]:
            raise RuntimeError(f"batch {bid} npz sha mismatch")
        labs = b["labels"]
        mls = b["match_locals"]
        cursor = 0
        for ml in mls.tolist():
            ml = int(ml)
            if seen[ml]:
                raise RuntimeError(f"duplicate match_local in batches: {ml}")
            seen[ml] = 1
            s = int(starts[ml])
            e = int(ends[ml])
            n = e - s
            chunk = labs[cursor : cursor + n]
            if chunk.shape[0] != n:
                raise RuntimeError(f"batch label slice short for match {ml}")
            labels[s:e] = chunk
            cursor += n
        if cursor != labs.shape[0]:
            raise RuntimeError(f"batch {bid} unused label rows")
    if int(seen.sum()) != n_matches or int((labels == -3).sum()) != 0:
        raise RuntimeError("label coverage incomplete")
    return labels


def load_cache_pack(match_id: str) -> tuple[dict[str, Any] | None, str]:
    path = C.CACHE_DIR / f"{match_id}.npz"
    if not path.is_file():
        return None, "missing_npz"
    try:
        with np.load(path, allow_pickle=False) as z:
            need = ("minute_ts", "xy_raw_minute", "node_minute", "gold_team_minute")
            missing = [k for k in need if k not in z.files]
            if missing:
                return None, f"missing_members:{','.join(missing)}"
            original = {k: z[k] for k in need}
            if original["minute_ts"].dtype.kind not in "iu":
                return None, "noninteger_minute_ts"
            if any(original[k].dtype.kind not in "iuf" for k in need[1:]):
                return None, "nonnumeric_cache_channel"
            pack = {
                "minute_ts": np.asarray(original["minute_ts"], dtype=np.int64),
                "xy_raw_minute": np.asarray(original["xy_raw_minute"], dtype=np.float64),
                "node_minute": np.asarray(original["node_minute"], dtype=np.float64),
                "gold_team_minute": np.asarray(original["gold_team_minute"], dtype=np.float64),
                "npz_keys": list(z.files),
            }
        if pack["xy_raw_minute"].ndim != 3 or pack["xy_raw_minute"].shape[1:] != (10, 2):
            return None, "bad_xy_shape"
        if pack["minute_ts"].ndim != 1:
            return None, "bad_ts_shape"
        if pack["node_minute"].ndim != 3 or pack["node_minute"].shape[1:] != (10, 76):
            return None, "bad_node_shape"
        if pack["gold_team_minute"].ndim != 2 or pack["gold_team_minute"].shape[1] != 2:
            return None, "bad_gold_shape"
        if any(pack[k].shape[0] != pack["minute_ts"].size for k in need[1:]):
            return None, "frame_channel_len_mismatch"
        return pack, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"npz_read_error:{type(exc).__name__}"


def load_cache_meta_team_map(match_id: str) -> tuple[dict[int, int] | None, str]:
    path = C.CACHE_DIR / f"{match_id}.meta.json"
    if not path.is_file():
        return None, "missing_meta_json"
    try:
        meta = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"meta_parse_error:{type(exc).__name__}"
    tm = meta.get("team_map")
    if not isinstance(tm, dict):
        return None, "missing_team_map"
    out: dict[int, int] = {}
    for k, v in tm.items():
        try:
            pid = int(k)
            team = int(v)
        except (TypeError, ValueError):
            return None, "malformed_team_map_entry"
        if not (1 <= pid <= 10):
            return None, "team_map_pid_out_of_range"
        if team not in (C.TEAM_BLUE, C.TEAM_RED):
            return None, "team_map_unknown_team_id"
        out[pid] = team
    if len(out) != 10:
        return None, "team_map_incomplete"
    return out, ""


def load_event_timestamps(match_id: str) -> tuple[np.ndarray | None, list[str] | None, str]:
    """Load flat event timestamps and types from cache events.json (not raw presence)."""
    path = C.CACHE_DIR / f"{match_id}.events.json"
    if not path.is_file():
        return None, None, "missing_events_json"
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"events_parse_error:{type(exc).__name__}"
    if not isinstance(data, list):
        # Prefer top-level list; framed schemas are unsupported for this diagnostic channel.
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            events = data["events"]
        else:
            return None, None, "unsupported_events_schema"
    else:
        events = data
    ts_list: list[int] = []
    types: list[str] = []
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            return None, None, f"non_object_event_at_{i}"
        if "timestamp" not in e:
            return None, None, f"missing_timestamp_at_{i}"
        t, reason = CT.parse_timestamp_ms(e["timestamp"])
        if reason or t is None:
            return None, None, f"malformed_timestamp_at_{i}:{reason}"
        ts_list.append(int(t))
        types.append(str(e.get("type") or ""))
    return np.asarray(ts_list, dtype=np.int64), types, ""


def _detail_timeline_paths(root: Path, match_id: str) -> tuple[Path, Path]:
    return root / "detail" / f"{match_id}.json", root / "timeline" / f"{match_id}.json"


def _extract_detail_identity(detail: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Pull matchId/gameVersion/team map from a detail JSON object."""
    info = detail.get("info") if isinstance(detail, dict) else None
    meta = detail.get("metadata") if isinstance(detail, dict) else None
    if not isinstance(info, dict):
        return None, "detail_missing_info"
    match_id = None
    if isinstance(meta, dict):
        match_id = meta.get("matchId") or meta.get("match_id")
    if match_id is None:
        match_id = info.get("gameId") or detail.get("matchId")
    if match_id is None:
        return None, "detail_missing_match_id"
    game_version = info.get("gameVersion") or info.get("game_version")
    if game_version is None:
        return None, "detail_missing_game_version"
    participants = info.get("participants")
    if not isinstance(participants, list) or not participants:
        return None, "detail_missing_participants"
    team_map: dict[int, int] = {}
    for p in participants:
        if not isinstance(p, dict):
            return None, "detail_malformed_participant"
        try:
            pid = int(p.get("participantId"))
            team = int(p.get("teamId"))
        except (TypeError, ValueError):
            return None, "detail_bad_participant_ids"
        if not (1 <= pid <= 10) or team not in (C.TEAM_BLUE, C.TEAM_RED):
            return None, "detail_participant_out_of_range"
        team_map[pid] = team
    if len(team_map) != 10:
        return None, "detail_team_map_incomplete"
    return {
        "match_id": str(match_id),
        "game_version": str(game_version),
        "team_map": team_map,
    }, ""


def _timeline_metadata_match_id(timeline: dict[str, Any]) -> tuple[str | None, str]:
    meta = timeline.get("metadata") if isinstance(timeline, dict) else None
    if not isinstance(meta, dict):
        return None, "timeline_missing_metadata"
    mid = meta.get("matchId") or meta.get("match_id")
    if mid is None:
        return None, "timeline_missing_match_id"
    return str(mid), ""


def _extract_timeline_frame_ts(timeline: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    info = timeline.get("info") if isinstance(timeline, dict) else None
    if not isinstance(info, dict):
        return None, "timeline_missing_info"
    frames = info.get("frames")
    if not isinstance(frames, list):
        return None, "timeline_missing_frames"
    ts_list: list[int] = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict) or "timestamp" not in fr:
            return None, f"timeline_bad_frame_at_{i}"
        t, reason = CT.parse_timestamp_ms(fr["timestamp"])
        if reason or t is None:
            return None, f"timeline_bad_timestamp_at_{i}:{reason}"
        ts_list.append(int(t))
    return np.asarray(ts_list, dtype=np.int64), ""


def _match_id_equals(raw_id: str, expected: str) -> bool:
    mid_raw = str(raw_id)
    return mid_raw == expected or mid_raw == expected.replace("KR_", "")


def verify_raw_pair_identity(
    match_id: str,
    root: Path,
    *,
    cache_minute_ts: np.ndarray | None = None,
    cache_team_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Verify found original detail/timeline against match identity/version/team/frames.

    This implementation does not yet extract raw participantFrames for presence claims.
    Verified identity alone is insufficient for raw_presence_verified=True here.
    Both detail and timeline metadata.matchId are checked.
    """
    d_path, t_path = _detail_timeline_paths(root, match_id)
    out: dict[str, Any] = {
        "match_id": match_id,
        "root": str(root),
        "ok": False,
        "reason": "",
        "raw_fields_supported": False,
        "scope": "failure",
    }
    if not d_path.is_file() or not t_path.is_file():
        out["reason"] = "pair_files_missing"
        return out
    try:
        detail = load_json(d_path)
        timeline = load_json(t_path)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"raw_json_parse_error:{type(exc).__name__}"
        return out
    ident, ireason = _extract_detail_identity(detail)
    if ireason or ident is None:
        out["reason"] = ireason or "detail_identity_failed"
        return out
    mid_raw = str(ident["match_id"])
    if not _match_id_equals(mid_raw, match_id):
        out["reason"] = "detail_match_id_mismatch"
        out["detail_match_id"] = mid_raw
        out["scope"] = "mismatch"
        return out
    tl_mid, tl_reason = _timeline_metadata_match_id(timeline)
    if tl_reason or tl_mid is None:
        out["reason"] = tl_reason or "timeline_match_id_failed"
        out["scope"] = "mismatch"
        return out
    if not _match_id_equals(tl_mid, match_id):
        out["reason"] = "timeline_match_id_mismatch"
        out["timeline_match_id"] = tl_mid
        out["scope"] = "mismatch"
        return out
    if not _match_id_equals(tl_mid, mid_raw) and not (
        _match_id_equals(tl_mid, match_id) and _match_id_equals(mid_raw, match_id)
    ):
        out["reason"] = "detail_timeline_match_id_disagree"
        out["detail_match_id"] = mid_raw
        out["timeline_match_id"] = tl_mid
        out["scope"] = "mismatch"
        return out
    gv = str(ident["game_version"])
    # TRAIN frozen patch family 15.14 — require 15.14 prefix in gameVersion string.
    if not gv.startswith("15.14"):
        out["reason"] = f"game_version_unsupported:{gv}"
        out["game_version"] = gv
        out["scope"] = "unsupported"
        return out
    if cache_team_map is not None and ident["team_map"] != cache_team_map:
        out["reason"] = "team_map_mismatch_vs_cache"
        out["scope"] = "mismatch"
        return out
    frame_ts, freason = _extract_timeline_frame_ts(timeline)
    if freason or frame_ts is None:
        out["reason"] = freason or "timeline_frames_failed"
        out["scope"] = "mismatch"
        return out
    if cache_minute_ts is not None:
        cms = np.asarray(cache_minute_ts, dtype=np.int64)
        if cms.shape != frame_ts.shape or not np.array_equal(cms, frame_ts):
            out["reason"] = "frame_timestamp_sequence_mismatch_vs_cache"
            out["n_raw_frames"] = int(frame_ts.size)
            out["n_cache_frames"] = int(cms.size)
            out["scope"] = "mismatch"
            return out
    # Identity/version/team/frame verified, but this diagnostic does not implement
    # raw participantFrames presence extraction — fail closed for raw claims.
    out["ok"] = True
    out["game_version"] = gv
    out["n_raw_frames"] = int(frame_ts.size)
    out["raw_fields_supported"] = False
    out["reason"] = "identity_ok_but_raw_participantframes_extraction_unsupported"
    out["scope"] = "unsupported"
    return out


def census_raw_source_availability(match_ids: list[str]) -> dict[str, Any]:
    """Full TRAIN key census against recorded historical raw roots. No fetching."""
    roots_info = []
    pair_sets: list[set[str]] = []
    for root in C.RAW_SOURCE_ROOTS:
        detail = root / "detail"
        timeline = root / "timeline"
        info: dict[str, Any] = {
            "root": str(root),
            "exists": root.is_dir(),
            "detail_exists": detail.is_dir(),
            "timeline_exists": timeline.is_dir(),
            "n_detail": 0,
            "n_timeline": 0,
            "n_pairs": 0,
        }
        pairs: set[str] = set()
        if detail.is_dir() and timeline.is_dir():
            det = {n[:-5] if n.endswith(".json") else n for n in os.listdir(detail) if n.endswith(".json")}
            tim = {n[:-5] if n.endswith(".json") else n for n in os.listdir(timeline) if n.endswith(".json")}
            pairs = det & tim
            info["n_detail"] = len(det)
            info["n_timeline"] = len(tim)
            info["n_pairs"] = len(pairs)
        roots_info.append(info)
        pair_sets.append(pairs)

    available_verified: list[str] = []
    missing: list[dict[str, str]] = []
    mismatch: list[dict[str, str]] = []
    unsupported: list[dict[str, str]] = []
    found_but_unsupported_raw: list[dict[str, str]] = []

    for mid in match_ids:
        found_roots = [i for i, s in enumerate(pair_sets) if mid in s]
        if not found_roots:
            missing.append({"match_id": mid, "reason": "no_detail_timeline_pair_at_recorded_roots"})
            continue
        # Distinct frozen keys determine counts; multiple-root hits are unsupported, never absent.
        if len(found_roots) > 1:
            row = {
                "match_id": mid,
                "reason": "multiple_root_hits_unresolved",
                "n_roots": len(found_roots),
                "roots": [str(C.RAW_SOURCE_ROOTS[i]) for i in found_roots],
            }
            unsupported.append(row)
            found_but_unsupported_raw.append(row)
            continue
        root = C.RAW_SOURCE_ROOTS[found_roots[0]]
        # Load cache side for identity gates when present (optional; absence is cache miss, not raw absence).
        pack, pack_reason = load_cache_pack(mid)
        team_map, tm_reason = load_cache_meta_team_map(mid)
        cache_ts = pack["minute_ts"] if pack is not None else None
        # Mandatory cache comparisons when cache exists; cache miss does not relabel raw as absent.
        if pack is None and pack_reason:
            # Still verify raw identity; cache comparison skipped with explicit note.
            ver = verify_raw_pair_identity(mid, root, cache_minute_ts=None, cache_team_map=None)
            if ver.get("ok"):
                ver["cache_compare_skipped"] = pack_reason
        else:
            ver = verify_raw_pair_identity(
                mid, root, cache_minute_ts=cache_ts, cache_team_map=team_map
            )
            if tm_reason and team_map is None:
                ver.setdefault("cache_team_map_reason", tm_reason)
        scope = str(ver.get("scope") or "")
        if not ver.get("ok"):
            row = {"match_id": mid, "reason": str(ver.get("reason") or "identity_failed"), "scope": scope or "mismatch"}
            if scope == "unsupported":
                unsupported.append(row)
                found_but_unsupported_raw.append(row)
            else:
                mismatch.append(row)
            continue
        if not ver.get("raw_fields_supported"):
            # Found originals that this implementation cannot support for raw presence claims.
            # Never relabel as absent.
            row = {
                "match_id": mid,
                "reason": str(ver.get("reason") or "raw_fields_unsupported"),
                "scope": "unsupported",
            }
            found_but_unsupported_raw.append(row)
            unsupported.append(row)
            continue
        available_verified.append(mid)

    n = len(match_ids)
    n_found_at_roots = n - len(missing)
    hard_fail_found_unsupported = len(found_but_unsupported_raw) > 0
    hard_fail_mismatch = len(mismatch) > 0
    summary = {
        "n_train_keys": n,
        "n_available_pair_candidates": int(n_found_at_roots),
        "n_available_verified_supported": len(available_verified),
        "n_found_but_unsupported_raw": len(found_but_unsupported_raw),
        "n_missing": len(missing),
        "n_mismatch": len(mismatch),
        "n_unsupported": len(unsupported),
        "roots": roots_info,
        "raw_presence_verified": False,
        "hard_fail_found_unsupported_raw": hard_fail_found_unsupported,
        "hard_fail_mismatch": hard_fail_mismatch,
        "diagnostic_mode": (
            "CACHE_ONLY"
            if len(available_verified) == 0
            and not hard_fail_found_unsupported
            and not hard_fail_mismatch
            else (
                "RAW_FOUND_UNSUPPORTED"
                if hard_fail_found_unsupported
                else ("RAW_MISMATCH" if hard_fail_mismatch else "RAW_CANDIDATES_PRESENT")
            )
        ),
        "checked_locations_only": [str(r) for r in C.RAW_SOURCE_ROOTS],
        "note": (
            "Full TRAIN census against recorded historical roots only. "
            "Absence claims are limited to these checked locations. "
            "Found/mismatched/unsupported originals fail explicitly and are never relabeled absent. "
            "Cache fields are not raw presence proof. "
            "Full failure ledgers are retained locally; samples below are report-only."
        ),
    }
    reason_hist: dict[str, int] = {}
    for row in missing:
        reason_hist[row["reason"]] = reason_hist.get(row["reason"], 0) + 1
    summary["missing_reason_histogram"] = reason_hist
    mismatch_hist: dict[str, int] = {}
    for row in mismatch:
        mismatch_hist[row["reason"]] = mismatch_hist.get(row["reason"], 0) + 1
    summary["mismatch_reason_histogram"] = mismatch_hist
    unsupported_hist: dict[str, int] = {}
    for row in unsupported:
        unsupported_hist[row["reason"]] = unsupported_hist.get(row["reason"], 0) + 1
    summary["unsupported_reason_histogram"] = unsupported_hist
    # Full local ledgers (no 50-case truncation). Short samples for human reports only.
    summary["available_match_ids"] = available_verified
    summary["missing_rows"] = missing
    summary["mismatch_rows"] = mismatch
    summary["unsupported_rows"] = unsupported
    summary["found_but_unsupported_raw_rows"] = found_but_unsupported_raw
    summary["mismatch_rows_sample"] = mismatch[:50]
    summary["unsupported_rows_sample"] = unsupported[:50]
    summary["found_but_unsupported_raw_sample"] = found_but_unsupported_raw[:50]
    summary["missing_rows_sample"] = missing[:50]
    h = hashlib.sha256()
    for mid in match_ids:
        h.update(mid.encode("utf-8"))
        h.update(b"\n")
    summary["train_match_ids_sha256"] = h.hexdigest()
    summary["all_keys_checked"] = n == C.EXPECTED_TRAIN_MATCHES
    summary["n_keys_checked"] = n
    return summary


def census_raw_source_availability_from_pair_sets(
    match_ids: list[str],
    pair_sets: list[set[str]],
    roots: Sequence[Path],
    *,
    verify_fn=None,
) -> dict[str, Any]:
    """Testable census over injected pair sets (no real corpus writes)."""
    verify_fn = verify_fn or (lambda mid, root, **kw: {"ok": False, "reason": "stub", "raw_fields_supported": False, "scope": "mismatch"})
    available_verified: list[str] = []
    missing: list[dict[str, Any]] = []
    mismatch: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    found_but_unsupported_raw: list[dict[str, Any]] = []
    for mid in match_ids:
        found_roots = [i for i, s in enumerate(pair_sets) if mid in s]
        if not found_roots:
            missing.append({"match_id": mid, "reason": "no_detail_timeline_pair_at_recorded_roots"})
            continue
        if len(found_roots) > 1:
            row = {
                "match_id": mid,
                "reason": "multiple_root_hits_unresolved",
                "n_roots": len(found_roots),
            }
            unsupported.append(row)
            found_but_unsupported_raw.append(row)
            continue
        root = roots[found_roots[0]]
        ver = verify_fn(mid, root)
        if not ver.get("ok"):
            row = {"match_id": mid, "reason": str(ver.get("reason") or "identity_failed")}
            if str(ver.get("scope") or "") == "unsupported":
                unsupported.append(row)
                found_but_unsupported_raw.append(row)
            else:
                mismatch.append(row)
            continue
        if not ver.get("raw_fields_supported"):
            row = {"match_id": mid, "reason": str(ver.get("reason") or "raw_fields_unsupported")}
            found_but_unsupported_raw.append(row)
            unsupported.append(row)
            continue
        available_verified.append(mid)
    return {
        "n_train_keys": len(match_ids),
        "n_missing": len(missing),
        "n_mismatch": len(mismatch),
        "n_unsupported": len(unsupported),
        "n_found_but_unsupported_raw": len(found_but_unsupported_raw),
        "n_available_verified_supported": len(available_verified),
        "missing_rows": missing,
        "mismatch_rows": mismatch,
        "unsupported_rows": unsupported,
        "found_but_unsupported_raw_rows": found_but_unsupported_raw,
        "available_match_ids": available_verified,
    }



def verify_team_map_vs_blue_pids(team_map: dict[int, int], blue_pids: Sequence[int]) -> bool:
    blue_from_map = set(CT.blue_pids_from_team_map(team_map))
    blue_meta = {int(p) for p in blue_pids if 1 <= int(p) <= 10}
    return blue_from_map == blue_meta
