"""Independent reference computation from immutable ST ledger + cache parents.

Does NOT import or call production contracts/extract_match/aggregate science helpers.
Generic hashing and config path/constants may be shared.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import config as C

# Local observation codes (duplicated; do not import contracts).
_POS_KNOWN_NONZERO = 0
_POS_AMBIGUOUS_ZERO = 1
_POS_UNKNOWN = 2
_POS_NONFINITE = 3
_HP_CACHED_POSITIVE = 0
_HP_AMBIGUOUS_ZERO = 1
_SENTINEL = -1
_ANCHOR = 10_000

# The replay is consumed by more than one runner.  Keep the schema here rather
# than importing runner/aggregation code so a missing stored field cannot be
# hidden by a production default.  The tuples are public on purpose: callers
# use them when allocating typed per-match storage and when checking coverage.
REFERENCE_FIXED_INT_FIELDS = (
    "match_local", "component_id", "n_members", "same_ms_first_n", "f", "L",
    "tau", "t_in", "c0_bits", "call_bits", "c0_n", "call_n",
    "c0_n_env_killer", "call_n_env_killer", "c0_n_blue", "c0_n_red",
    "c0_n_unknown_team", "c0_scale_max_side", "call_n_blue", "call_n_red",
    "call_n_unknown_team", "call_scale_max_side", "c0_n_credited", "call_n_credited",
    "F_tau", "F_tau_idx", "lag_f_F_tau", "F_tm1", "F_tm1_idx", "lag_f_F_tm1",
    "a_strict", "ev_at_tau", "ev_between_Ftau_tau", "ev_between_Ftm1_tin",
    "ev_types_at_tau_n", "foreign_kills_in_tau_L", "foreign_near", "foreign_distant",
    "other_anchors_in_tau_L", "last_frame_ms", "support_start_ms", "q_pre",
    "game_end_ms", "next_component_tau", "next_kill_after_L",
)
REFERENCE_FIXED_BOOL_FIELDS = (
    "singleton", "team_map_ok", "cand_10_20_tau", "cand_10_20_tm1", "exact_10000_tau",
    "strict_eligible", "frame_differs", "raw_presence_verified", "radius_mono_p_lower_tau",
    "radius_mono_p_upper_tau", "events_support_ok", "game_end_known",
    "overlap_next_tau_le_L", "missing_terminal_game_end", "frame_support_ok",
    "terminal_support_authoritative",
)
REFERENCE_STRING_FIELDS = (
    "anon_match_key", "member_key", "tau_reason", "tm1_reason", "events_support_reason",
    "participation_mode", "team_map_provenance",
)
REFERENCE_PARTICIPATION_INT_FIELDS = (
    "n_lower", "n_upper", "p_lower_n", "p_upper_n", "lower_bits", "upper_bits", "near_bits",
    "geom_near_bits", "geom_credit_only_bits", "geom_near_only_bits", "geom_intersection_bits",
    "p_lower_bits", "p_upper_bits", "part_credit_only_lower_bits", "part_near_only_lower_bits",
    "part_intersection_lower_bits", "part_credit_only_upper_bits", "part_near_only_upper_bits",
    "part_intersection_upper_bits", "credit_only_bits", "near_only_bits", "intersection_bits",
)
REFERENCE_PARTICIPATION_BOOL_FIELDS = (
    "support_ok", "lower_le_upper", "c0_subset_lower", "c0_subset_upper",
)
REFERENCE_PARTICIPATION_FLOAT_FIELDS = ("dist_min", "dist_p50", "dist_p90")
REFERENCE_PARTICIPATION_TAGS = tuple(
    f"{cut}_r{int(radius)}" for cut in ("tau", "tm1") for radius in (1800, 2000)
)
REFERENCE_POST_INT_FIELDS = (
    "F_post", "F_post_idx", "age_post", "frame_idx_gap", "timestamp_gap",
    "async_events_Fpre_to_Fpost", "async_events_cutoff_to_endpoint",
)
REFERENCE_POST_BOOL_FIELDS = (
    "same_as_Fpre", "F_post_gt_L", "stale_at_or_before_L", "events_support_ok",
)
REFERENCE_POST_FLOAT_FIELDS = ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean")
REFERENCE_POST_TAGS = tuple(
    f"{cut}_h{h}" for h in (60, 90, 120) for cut in ("tau", "tm1")
)
REFERENCE_ENDPOINT_FIELDS = tuple(
    [f"e_h{h}" for h in (60, 90, 120)]
    + [f"e_h{h}_recomputed" for h in (60, 90, 120)]
    + [f"valid_h{h}" for h in (60, 90, 120)]
)
REFERENCE_TABLE_FIELDS = tuple(
    REFERENCE_FIXED_INT_FIELDS
    + REFERENCE_FIXED_BOOL_FIELDS
    + REFERENCE_STRING_FIELDS
    + tuple(f"{f}_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS for f in REFERENCE_PARTICIPATION_INT_FIELDS)
    + tuple(f"{f}_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS for f in REFERENCE_PARTICIPATION_BOOL_FIELDS)
    + tuple(f"{f}_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS for f in REFERENCE_PARTICIPATION_FLOAT_FIELDS)
    + REFERENCE_ENDPOINT_FIELDS
    + tuple(
        f"{f}_{tag}"
        for tag in REFERENCE_POST_TAGS
        for f in REFERENCE_POST_INT_FIELDS + REFERENCE_POST_BOOL_FIELDS + REFERENCE_POST_FLOAT_FIELDS
    )
    + tuple(f"post_support_reason_{tag}" for tag in REFERENCE_POST_TAGS)
)
REFERENCE_EVIDENCE_FIELDS = (
    "first_loc_offsets", "first_loc_x", "first_loc_y",
    "dist_tau_r1800", "dist_tau_r2000", "dist_tm1_r1800", "dist_tm1_r2000",
    "frame_xy_tau", "frame_xy_tm1", "hp_tau", "hp_tm1",
    "pos_codes_tau_r1800", "pos_codes_tau_r2000", "pos_codes_tm1_r1800", "pos_codes_tm1_r2000",
    "hp_codes_tau_r1800", "hp_codes_tau_r2000", "hp_codes_tm1_r1800", "hp_codes_tm1_r2000",
    "frame_ts_tau", "frame_ts_tm1", "frame_idx_tau", "frame_idx_tm1",
)
REFERENCE_EVIDENCE_FIELDS += tuple(
        f"{prefix}_{tag}"
        for tag in REFERENCE_POST_TAGS
        for prefix in (
            "pre_gold", "post_gold", "pre_level_mean", "post_level_mean",
            "pre_xp_mean", "post_xp_mean", "pre_hp_mean", "post_hp_mean",
        )
)
REFERENCE_ROW_ARRAY_ALIASES = tuple(
    [f"_dists_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS]
    + [f"_pos_codes_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS]
    + [f"_hp_codes_{tag}" for tag in REFERENCE_PARTICIPATION_TAGS]
    + ["_frame_xy_tau", "_frame_xy_tm1", "_hp_tau", "_hp_tm1", "first_kill_x", "first_kill_y"]
)
REFERENCE_FIELD_METADATA = {
    **{f: {"kind": "int", "required": True} for f in REFERENCE_FIXED_INT_FIELDS},
    **{f: {"kind": "bool", "required": True} for f in REFERENCE_FIXED_BOOL_FIELDS},
    **{f: {"kind": "string", "required": True} for f in REFERENCE_STRING_FIELDS},
    **{f: {"kind": "int", "required": True} for f in REFERENCE_ENDPOINT_FIELDS},
}
for _tag in REFERENCE_PARTICIPATION_TAGS:
    for _f in REFERENCE_PARTICIPATION_INT_FIELDS:
        REFERENCE_FIELD_METADATA[f"{_f}_{_tag}"] = {"kind": "int", "required": True}
    for _f in REFERENCE_PARTICIPATION_BOOL_FIELDS:
        REFERENCE_FIELD_METADATA[f"{_f}_{_tag}"] = {"kind": "bool", "required": True}
    for _f in REFERENCE_PARTICIPATION_FLOAT_FIELDS:
        REFERENCE_FIELD_METADATA[f"{_f}_{_tag}"] = {"kind": "float", "required": True}
for _tag in REFERENCE_POST_TAGS:
    for _f in REFERENCE_POST_INT_FIELDS + REFERENCE_POST_BOOL_FIELDS + REFERENCE_POST_FLOAT_FIELDS:
        REFERENCE_FIELD_METADATA[f"{_f}_{_tag}"] = {
            "kind": "float" if _f in REFERENCE_POST_FLOAT_FIELDS else "int",
            "required": True,
        }
    REFERENCE_FIELD_METADATA[f"post_support_reason_{_tag}"] = {"kind": "string", "required": True}

# Backward-compatible aliases used by external runners.
REQUIRED_REFERENCE_TABLE_FIELDS = REFERENCE_TABLE_FIELDS
REQUIRED_REFERENCE_EVIDENCE_FIELDS = REFERENCE_EVIDENCE_FIELDS
REQUIRED_REFERENCE_ROW_ARRAY_ALIASES = REFERENCE_ROW_ARRAY_ALIASES
REQUIRED_TABLE_FIELDS = REFERENCE_TABLE_FIELDS
REQUIRED_EVIDENCE_FIELDS = REFERENCE_EVIDENCE_FIELDS
FIELD_METADATA = REFERENCE_FIELD_METADATA
REFERENCE_EVIDENCE_METADATA: dict[str, dict[str, Any]] = {
    "first_loc_offsets": {"kind": "int", "shape": "n_rows+1"},
    "first_loc_x": {"kind": "float", "shape": "variable_per_row"},
    "first_loc_y": {"kind": "float", "shape": "variable_per_row"},
}
for _name in (
    "dist_tau_r1800", "dist_tau_r2000", "dist_tm1_r1800", "dist_tm1_r2000",
    "hp_tau", "hp_tm1", "pos_codes_tau_r1800", "pos_codes_tau_r2000",
    "pos_codes_tm1_r1800", "pos_codes_tm1_r2000", "hp_codes_tau_r1800",
    "hp_codes_tau_r2000", "hp_codes_tm1_r1800", "hp_codes_tm1_r2000",
):
    REFERENCE_EVIDENCE_METADATA[_name] = {
        "kind": "float" if _name.startswith("dist_") or _name in ("hp_tau", "hp_tm1") else "int",
        "shape": "n_rows,10",
    }
for _name in ("frame_xy_tau", "frame_xy_tm1"):
    REFERENCE_EVIDENCE_METADATA[_name] = {"kind": "float", "shape": "n_rows,10,2"}
for _name in ("frame_ts_tau", "frame_ts_tm1", "frame_idx_tau", "frame_idx_tm1"):
    REFERENCE_EVIDENCE_METADATA[_name] = {"kind": "int", "shape": "n_rows"}
for _name in REFERENCE_EVIDENCE_FIELDS:
    if _name.startswith(("pre_gold_", "post_gold_")):
        REFERENCE_EVIDENCE_METADATA[_name] = {"kind": "float", "shape": "n_rows,2"}
    elif _name not in REFERENCE_EVIDENCE_METADATA:
        REFERENCE_EVIDENCE_METADATA[_name] = {"kind": "float", "shape": "n_rows"}
REQUIRED_EVIDENCE_METADATA = REFERENCE_EVIDENCE_METADATA


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ts(value: Any) -> tuple[int | None, str]:
    if isinstance(value, bool):
        return None, "boolean_timestamp"
    if isinstance(value, int):
        return (None, "negative_timestamp") if value < 0 else (int(value), "")
    if isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value) or value < 0:
            return None, "bad_float_timestamp"
        return int(value), ""
    return None, "non_integral_timestamp"


def _preceding_frame(minute_ts: np.ndarray, t: int) -> tuple[int | None, int | None, int | None, str]:
    ts = np.asarray(minute_ts, dtype=np.int64)
    if ts.size == 0:
        return None, None, None, "empty_minute_ts"
    if np.any(ts < 0):
        return None, None, None, "negative_minute_ts"
    if ts.size >= 2 and np.any(np.diff(ts) < 0):
        return None, None, None, "nonmonotone_minute_ts"
    i = int(np.searchsorted(ts, int(t), side="right")) - 1
    if i < 0:
        return None, None, None, "no_preceding_frame"
    F = int(ts[i])
    return F, i, int(t) - F, ""


def _credit_bitset(killer: int, victim: int, assist_mask: int) -> int:
    bits = 0
    if 1 <= int(killer) <= 10:
        bits |= 1 << (int(killer) - 1)
    if 1 <= int(victim) <= 10:
        bits |= 1 << (int(victim) - 1)
    mask = int(assist_mask)
    for pid in range(1, 11):
        if mask & (1 << (pid - 1)):
            bits |= 1 << (pid - 1)
    return int(bits)


def _count_events_between(event_ts: np.ndarray, lo_exclusive: int, hi_inclusive: int) -> int:
    ts = np.asarray(event_ts, dtype=np.int64)
    if ts.size == 0:
        return 0
    return int(np.sum((ts > int(lo_exclusive)) & (ts <= int(hi_inclusive))))


def _endpoint_ms(
    L: int,
    h_s: int,
    next_kill_after_L: int,
    next_component_tau: int,
    game_end_ms: int,
    game_end_known: bool,
) -> int:
    horizon = int(L) + 1000 * int(h_s)
    cand = [horizon]
    nk = int(next_kill_after_L)
    nt = int(next_component_tau)
    ge = int(game_end_ms) if game_end_known and int(game_end_ms) >= 0 else None
    if nk >= 0:
        cand.append(nk - 1)
    if nt >= 0:
        if ge is None or nt < ge:
            cand.append(nt - 1)
    if ge is not None:
        cand.append(ge - 1)
    return int(min(cand))


def _cache_pos_state(x: float, y: float) -> int:
    if not (math.isfinite(x) and math.isfinite(y)):
        return _POS_NONFINITE
    if x == 0.0 or y == 0.0:
        return _POS_AMBIGUOUS_ZERO
    return _POS_KNOWN_NONZERO


def _cache_hp_state(hp: float) -> int:
    if not math.isfinite(hp):
        return 2
    if hp < 0:
        return 4
    if hp == 0.0:
        return _HP_AMBIGUOUS_ZERO
    return _HP_CACHED_POSITIVE


def _participation_from_frame(
    frame_xy: np.ndarray,
    frame_hp: np.ndarray,
    locs_x: np.ndarray,
    locs_y: np.ndarray,
    radius: float,
    c0_bits: int,
    *,
    unsupported: bool = False,
) -> dict[str, int]:
    c0 = int(c0_bits) & ((1 << 10) - 1)
    if unsupported:
        upper = (1 << 10) - 1
        return {
            "n_lower": 0,
            "n_upper": 10,
            "lower_bits": 0,
            "upper_bits": upper,
            "p_lower_bits": c0,
            "p_upper_bits": upper,
            "p_lower_n": int(c0.bit_count()),
            "p_upper_n": 10,
            "support_ok": 0,
            "lower_le_upper": 1,
            "c0_subset_lower": 1,
            "c0_subset_upper": 1,
            "near_bits": 0,
            "geom_near_bits": 0,
            "geom_credit_only_bits": c0,
            "geom_near_only_bits": 0,
            "geom_intersection_bits": 0,
            "part_credit_only_lower_bits": c0,
            "part_near_only_lower_bits": 0,
            "part_intersection_lower_bits": 0,
            "part_credit_only_upper_bits": 0,
            "part_near_only_upper_bits": upper & ~c0,
            "part_intersection_upper_bits": c0,
            "credit_only_bits": c0,
            "near_only_bits": 0,
            "intersection_bits": 0,
            "dists": np.full(10, np.nan, dtype=np.float64),
            "pos_codes": np.full(10, _POS_UNKNOWN, dtype=np.int8),
            "hp_codes": np.full(10, 2, dtype=np.int8),
        }
    lower_bits = 0
    upper_bits = 0
    near_bits = 0
    dists = np.full(10, np.nan, dtype=np.float64)
    pos_codes = np.full(10, _POS_UNKNOWN, dtype=np.int8)
    hp_codes = np.full(10, 2, dtype=np.int8)
    for j in range(10):
        px = float(frame_xy[j, 0])
        py = float(frame_xy[j, 1])
        hpv = float(frame_hp[j])
        pos_c = _cache_pos_state(px, py)
        hp_c = _cache_hp_state(hpv)
        pos_codes[j] = pos_c
        hp_codes[j] = hp_c
        in_radius = False
        if pos_c == _POS_KNOWN_NONZERO and locs_x.size > 0:
            d = float(np.sqrt((locs_x - px) ** 2 + (locs_y - py) ** 2).min())
            dists[j] = d
            in_radius = d <= float(radius)
            if in_radius:
                near_bits |= 1 << j
        if in_radius and hp_c == _HP_CACHED_POSITIVE:
            lower_bits |= 1 << j
        if in_radius or pos_c in (_POS_UNKNOWN, _POS_AMBIGUOUS_ZERO, _POS_NONFINITE):
            upper_bits |= 1 << j
    p_lower = int(c0 | lower_bits)
    p_upper = int(c0 | upper_bits)
    geom_credit_only = int(c0 & ~near_bits)
    geom_near_only = int(near_bits & ~c0)
    geom_intersection = int(c0 & near_bits)
    return {
        "n_lower": int(bin(lower_bits).count("1")),
        "n_upper": int(bin(upper_bits).count("1")),
        "lower_bits": int(lower_bits),
        "upper_bits": int(upper_bits),
        "p_lower_bits": p_lower,
        "p_upper_bits": p_upper,
        "p_lower_n": int(p_lower.bit_count()),
        "p_upper_n": int(p_upper.bit_count()),
        "support_ok": 1,
        "lower_le_upper": int(lower_bits.bit_count() <= upper_bits.bit_count()),
        "c0_subset_lower": int((c0 & p_lower) == c0),
        "c0_subset_upper": int((c0 & p_upper) == c0),
        "near_bits": int(near_bits),
        "geom_near_bits": int(near_bits),
        "geom_credit_only_bits": geom_credit_only,
        "geom_near_only_bits": geom_near_only,
        "geom_intersection_bits": geom_intersection,
        "part_credit_only_lower_bits": int(c0 & ~lower_bits),
        "part_near_only_lower_bits": int(lower_bits & ~c0),
        "part_intersection_lower_bits": int(c0 & lower_bits),
        "part_credit_only_upper_bits": int(c0 & ~upper_bits),
        "part_near_only_upper_bits": int(upper_bits & ~c0),
        "part_intersection_upper_bits": int(c0 & upper_bits),
        "credit_only_bits": geom_credit_only,
        "near_only_bits": geom_near_only,
        "intersection_bits": geom_intersection,
        "dists": dists,
        "pos_codes": pos_codes,
        "hp_codes": hp_codes,
    }


def _load_cache_direct(match_id: str) -> tuple[dict[str, Any] | None, str, str]:
    path = C.CACHE_DIR / f"{match_id}.npz"
    if not path.is_file():
        return None, "missing_npz", ""
    sha = _sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            need = ("minute_ts", "xy_raw_minute", "node_minute", "gold_team_minute")
            missing = [k for k in need if k not in z.files]
            if missing:
                return None, f"missing_members:{','.join(missing)}", sha
            raw_ts = np.asarray(z["minute_ts"])
            raw_xy = np.asarray(z["xy_raw_minute"])
            raw_node = np.asarray(z["node_minute"])
            raw_gold = np.asarray(z["gold_team_minute"])
            if not np.issubdtype(raw_ts.dtype, np.integer):
                return None, "bad_dtype:minute_ts_not_integer", sha
            if not np.issubdtype(raw_xy.dtype, np.floating):
                return None, "bad_dtype:xy_raw_minute_not_float", sha
            if not np.issubdtype(raw_node.dtype, np.floating):
                return None, "bad_dtype:node_minute_not_float", sha
            if not np.issubdtype(raw_gold.dtype, np.floating):
                return None, "bad_dtype:gold_team_minute_not_float", sha
            pack = {
                "minute_ts": raw_ts.astype(np.int64, copy=False),
                "xy_raw_minute": raw_xy.astype(np.float64, copy=False),
                "node_minute": raw_node.astype(np.float64, copy=False),
                "gold_team_minute": raw_gold.astype(np.float64, copy=False),
            }
            ts = pack["minute_ts"]
            xy = pack["xy_raw_minute"]
            node = pack["node_minute"]
            gold = pack["gold_team_minute"]
            if ts.ndim != 1:
                return None, "bad_shape:minute_ts", sha
            if xy.ndim != 3 or xy.shape[1:] != (10, 2):
                return None, "bad_shape:xy_raw_minute", sha
            if node.ndim != 3 or node.shape[1:] != (10, 76):
                return None, "bad_shape:node_minute", sha
            if gold.ndim != 2 or gold.shape[1] != 2:
                return None, "bad_shape:gold_team_minute", sha
            n_frames = int(ts.shape[0])
            if xy.shape[0] != n_frames or node.shape[0] != n_frames or gold.shape[0] != n_frames:
                return None, "bad_shape:frame_count", sha
        return pack, "", sha
    except Exception as exc:  # noqa: BLE001
        return None, f"npz_read_error:{type(exc).__name__}", sha


def _load_meta_team_map_direct(match_id: str) -> tuple[dict[int, int] | None, str, str]:
    path = C.CACHE_DIR / f"{match_id}.meta.json"
    if not path.is_file():
        return None, "missing_meta_json", ""
    sha = _sha256_file(path)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"meta_parse_error:{type(exc).__name__}", sha
    tm = meta.get("team_map")
    if not isinstance(tm, dict):
        return None, "missing_team_map", sha
    out: dict[int, int] = {}
    for k, v in tm.items():
        try:
            pid = int(k)
            team = int(v)
        except (TypeError, ValueError):
            return None, "malformed_team_map_entry", sha
        if not (1 <= pid <= 10) or team not in (C.TEAM_BLUE, C.TEAM_RED):
            return None, "team_map_bad", sha
        out[pid] = team
    if len(out) != 10:
        return None, "team_map_incomplete", sha
    return out, "", sha


def _load_events_direct(match_id: str) -> tuple[np.ndarray | None, list[str] | None, str, str]:
    path = C.CACHE_DIR / f"{match_id}.events.json"
    if not path.is_file():
        return None, None, "missing_events_json", ""
    sha = _sha256_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, None, f"events_parse_error:{type(exc).__name__}", sha
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and isinstance(data.get("events"), list):
        events = data["events"]
    else:
        return None, None, "unsupported_events_schema", sha
    ts_list: list[int] = []
    types: list[str] = []
    for i, e in enumerate(events):
        if not isinstance(e, dict) or "timestamp" not in e:
            return None, None, f"bad_event_at_{i}", sha
        t, reason = _parse_ts(e["timestamp"])
        if reason or t is None:
            return None, None, f"malformed_timestamp_at_{i}:{reason}", sha
        ts_list.append(int(t))
        types.append(str(e.get("type") or ""))
    return np.asarray(ts_list, dtype=np.int64), types, "", sha


def _game_end_from_events(event_ts: np.ndarray, event_types: list[str]) -> tuple[int, bool, str]:
    ends = [int(event_ts[i]) for i, ty in enumerate(event_types) if ty == "GAME_END"]
    if not ends:
        return _SENTINEL, False, "missing_game_end"
    if len(set(ends)) > 1:
        return _SENTINEL, False, "ambiguous_game_end_timestamps"
    return int(ends[0]), True, ""


def _inverse_weighted_cdf_quantile(values: np.ndarray, weights: np.ndarray, p: float = 0.5) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return None
    v = values[ok]
    w = weights[ok]
    order = np.argsort(v, kind="mergesort")
    v_s = v[order]
    cdf = np.cumsum(w[order])
    total = float(cdf[-1])
    if total <= 0:
        return None
    idx = int(np.searchsorted(cdf, float(p) * total, side="left"))
    idx = min(max(idx, 0), v_s.size - 1)
    return float(v_s[idx])


def confirm_feature_channel_mapping() -> dict[str, Any]:
    """Confirm pinned feature indices against the independent source config.

    A local tuple is not evidence of provenance.  If the source config cannot be
    opened or parsed this function deliberately returns ``ok=False``.
    """
    out: dict[str, Any] = {
        "pinned_IDX_LEVEL_NORM": C.IDX_LEVEL_NORM,
        "pinned_IDX_XP_NORM": C.IDX_XP_NORM,
        "pinned_IDX_HP_PCT": C.IDX_HP_PCT,
        "ok": False,
    }
    try:
        import importlib.util

        source_path = C.ROOT / "worktrees" / "engagement-state-value" / "core" / "config.py"
        out["source_path"] = str(source_path)
        if not source_path.is_file():
            out["error"] = "source_config_missing"
            return out
        out["source_sha256"] = _sha256_file(source_path)
        spec = importlib.util.spec_from_file_location("reference_feature_source_config", source_path)
        if spec is None or spec.loader is None:
            out["error"] = "source_config_spec_unavailable"
            return out
        import sys

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        cfg_root = str(source_path.parent.parent)
        original_path = sys.path[:]
        try:
            if cfg_root not in sys.path:
                sys.path.insert(0, cfg_root)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        names = list(getattr(module, "NODE_FEATURE_NAMES"))
        out["n_features"] = len(names)
        out["level_norm_index"] = names.index("level_norm")
        out["xp_norm_index"] = names.index("xp_norm")
        out["hp_pct_index"] = names.index("hp_pct")
        out["ok"] = (
            out["level_norm_index"] == C.IDX_LEVEL_NORM
            and out["xp_norm_index"] == C.IDX_XP_NORM
            and out["hp_pct_index"] == C.IDX_HP_PCT
            and len(names) == 76
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def _array_sha256(value: np.ndarray) -> str:
    """Hash caller-provided source arrays with dtype and shape included."""
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(repr(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _team_counts(bits: int, team_map: Mapping[int, int] | None, ok: bool) -> dict[str, Any]:
    pids = [pid for pid in range(1, 11) if int(bits) & (1 << (pid - 1))]
    if not ok or team_map is None:
        return {
            "n_blue": -1, "n_red": -1, "n_unknown_team": -1,
            "n_pids": len(pids), "scale_max_side": -1,
            "blue": [], "red": [], "unknown_team": [],
        }
    blue = [pid for pid in pids if int(team_map.get(pid, -1)) == C.TEAM_BLUE]
    red = [pid for pid in pids if int(team_map.get(pid, -1)) == C.TEAM_RED]
    unknown = [pid for pid in pids if pid not in blue and pid not in red]
    return {
        "n_blue": len(blue), "n_red": len(red), "n_unknown_team": len(unknown),
        "n_pids": len(pids), "scale_max_side": max(len(blue), len(red)),
        "blue": blue, "red": red, "unknown_team": unknown,
    }


def _endpoint_validity(
    e: int,
    L: int,
    tau: int,
    support_start: int,
    last_frame: int,
    overlap_next_tau_le_L: bool,
    game_end_known: bool,
) -> tuple[bool, str]:
    """Inline the pinned endpoint validity contract without importing helpers."""
    q_pre = int(tau) - 1
    flags = {
        "endpoint_ge_L": int(e) >= int(L),
        "endpoint_gt_q_pre": int(e) > q_pre,
        "endpoint_le_last_frame": int(last_frame) >= 0 and int(e) <= int(last_frame),
        "q_pre_ge_support": int(support_start) >= 0 and q_pre >= int(support_start),
        "q_pre_le_last_frame": int(last_frame) >= 0 and q_pre <= int(last_frame),
        "terminal_support_authoritative": bool(game_end_known),
    }
    reasons = [name for name, passed in flags.items() if not passed]
    if overlap_next_tau_le_L:
        reasons.insert(0, "same_match_overlap_next_start_le_L")
    return len(reasons) == 0, "|".join(reasons)


def _empty_post(rec: dict[str, Any], tag: str, reason: str) -> None:
    for f in REFERENCE_POST_INT_FIELDS:
        rec[f"{f}_{tag}"] = _SENTINEL
    for f in REFERENCE_POST_BOOL_FIELDS:
        rec[f"{f}_{tag}"] = False
    for f in REFERENCE_POST_FLOAT_FIELDS:
        rec[f"{f}_{tag}"] = float("nan")
    rec[f"post_support_reason_{tag}"] = str(reason or "unsupported")
    rec[f"_pre_gold_{tag}"] = np.full(2, np.nan, dtype=np.float64)
    rec[f"_post_gold_{tag}"] = np.full(2, np.nan, dtype=np.float64)
    for prefix in (
        "pre_level_mean", "post_level_mean", "pre_xp_mean", "post_xp_mean",
        "pre_hp_mean", "post_hp_mean",
    ):
        rec[f"_{prefix}_{tag}"] = float("nan")


def _set_participation_record(rec: dict[str, Any], tag: str, part: Mapping[str, Any]) -> None:
    for f in REFERENCE_PARTICIPATION_INT_FIELDS + REFERENCE_PARTICIPATION_BOOL_FIELDS:
        if f in part:
            rec[f"{f}_{tag}"] = int(part[f])
    for f in REFERENCE_PARTICIPATION_FLOAT_FIELDS:
        d = np.asarray(part.get("dists", np.full(10, np.nan)), dtype=np.float64)
        known = np.isfinite(d)
        if not np.any(known):
            rec[f"{f}_{tag}"] = float("nan")
        elif f == "dist_min":
            rec[f"{f}_{tag}"] = float(np.min(d[known]))
        elif f == "dist_p50":
            rec[f"{f}_{tag}"] = float(np.median(d[known]))
        else:
            rec[f"{f}_{tag}"] = float(np.quantile(d[known], 0.9))


def replay_match_components(
    *,
    match_local: int,
    match_id: str,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    killer: np.ndarray,
    victim: np.ndarray,
    assist: np.ndarray,
    labels: np.ndarray,
    blue_pids: Sequence[int],
    baseline_rows: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Replay one match directly from caller arrays and immutable cache parents.

    ``baseline_rows`` is used only for an auditable comparison target.  Component
    keys, cutoff frames, endpoint candidates/validity, and post-frame queries are
    reconstructed from the supplied ledger/labels and the direct cache parents.
    """
    arrays = {
        "t": np.asarray(t, dtype=np.int64),
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "killer": np.asarray(killer, dtype=np.int64),
        "victim": np.asarray(victim, dtype=np.int64),
        "assist": np.asarray(assist, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    n = int(arrays["t"].shape[0])
    if any(int(v.shape[0]) != n for v in arrays.values()):
        raise ValueError("ledger arrays and labels must have equal first dimension")
    input_hashes = {name: _array_sha256(value) for name, value in arrays.items()}

    pack, pack_reason, npz_sha = _load_cache_direct(match_id)
    team_map, tm_reason, meta_sha = _load_meta_team_map_direct(match_id)
    event_ts, event_types, ev_reason, ev_sha = _load_events_direct(match_id)
    events_ok = event_ts is not None and event_types is not None and not ev_reason

    blue_meta = {int(p) for p in blue_pids if 1 <= int(p) <= 10}
    blue_from_map = {pid for pid, team in (team_map or {}).items() if team == C.TEAM_BLUE}
    team_map_ok = bool(team_map is not None and not tm_reason and blue_from_map == blue_meta)
    if team_map is not None and blue_from_map != blue_meta:
        tm_reason = tm_reason or "team_map_blue_pids_mismatch"

    usable: np.ndarray | None = None
    frame_reason = pack_reason
    if pack is not None:
        ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        if ts.size == 0:
            frame_reason = "empty_minute_ts"
        elif np.any(ts < 0):
            frame_reason = "negative_minute_ts"
        elif ts.size >= 2 and np.any(np.diff(ts) < 0):
            frame_reason = "nonmonotone_minute_ts"
        else:
            usable = ts
            frame_reason = ""

    ge_ms, ge_known, ge_reason = (_SENTINEL, False, "events_unavailable")
    if events_ok and event_ts is not None and event_types is not None:
        ge_ms, ge_known, ge_reason = _game_end_from_events(event_ts, event_types)

    source_status = {
        "pack_reason": pack_reason,
        "frame_reason": frame_reason,
        "team_map_reason": tm_reason,
        "events_reason": ev_reason,
        "game_end_reason": ge_reason,
        "team_map_ok": team_map_ok,
        "events_ok": events_ok,
        "game_end_known": bool(ge_known),
        "input_array_hashes": input_hashes,
        "n_kills": n,
        "n_components": 0,
        "zero_kill": n == 0,
        "source_hashes_complete": bool(npz_sha and meta_sha and ev_sha),
    }
    source_hashes = {"npz": npz_sha, "meta": meta_sha, "events": ev_sha}
    if n == 0:
        return {
            "match_local": int(match_local), "match_id": str(match_id), "n_kills": 0,
            "n_components": 0, "components": [], "source_hashes": source_hashes,
            "source_status": source_status, "input_hashes": input_hashes, "ok": True,
        }

    pos = arrays["labels"] >= 0
    unique_labs = np.unique(arrays["labels"][pos]) if np.any(pos) else np.zeros(0, dtype=np.int64)
    comp_meta: list[tuple[int, int, int, np.ndarray]] = []
    for lab in unique_labs.tolist():
        memb = np.flatnonzero(arrays["labels"] == int(lab))
        if memb.size == 0:
            continue
        times = arrays["t"][memb]
        comp_meta.append((int(lab), int(times.min()), int(times.max()), memb))
    comp_meta.sort(key=lambda row: (row[1], row[2], row[0]))
    next_tau_by_lab: dict[int, int] = {}
    next_tau_present_by_lab: dict[int, bool] = {}
    for i, (lab, f, _L, _memb) in enumerate(comp_meta):
        nxt = _SENTINEL
        has_next = False
        for lab2, f2, _L2, _memb2 in comp_meta[i + 1 :]:
            if int(f2) > int(f):
                nxt = int(f2) - _ANCHOR
                has_next = True
                break
        next_tau_by_lab[int(lab)] = int(nxt)
        next_tau_present_by_lab[int(lab)] = bool(has_next)

    baseline_by_lab: dict[int, int] = {}
    baseline_duplicate_keys: list[int] = []
    if baseline_rows is not None and "component_id" in baseline_rows:
        for i, lab in enumerate(np.asarray(baseline_rows["component_id"]).tolist()):
            ilab = int(lab)
            if ilab in baseline_by_lab:
                baseline_duplicate_keys.append(ilab)
            baseline_by_lab[ilab] = i
    source_status["baseline_duplicate_component_ids"] = baseline_duplicate_keys

    all_t = arrays["t"]
    all_x = arrays["x"]
    all_y = arrays["y"]
    components: list[dict[str, Any]] = []
    support_start = int(usable[0]) if usable is not None and usable.size else _SENTINEL
    last_frame = int(usable[-1]) if usable is not None and usable.size else _SENTINEL

    for lab, f, L, memb in comp_meta:
        memb0 = memb[arrays["t"][memb] == int(f)]
        locs_x = all_x[memb0].astype(np.float64, copy=True)
        locs_y = all_y[memb0].astype(np.float64, copy=True)
        c0 = 0
        call = 0
        c0_env = 0
        call_env = 0
        for i in memb0.tolist():
            c0 |= _credit_bitset(int(arrays["killer"][i]), int(arrays["victim"][i]), int(arrays["assist"][i]))
            c0_env += int(arrays["killer"][i]) == 0
        for i in memb.tolist():
            call |= _credit_bitset(int(arrays["killer"][i]), int(arrays["victim"][i]), int(arrays["assist"][i]))
            call_env += int(arrays["killer"][i]) == 0
        c0_teams = _team_counts(c0, team_map, team_map_ok)
        call_teams = _team_counts(call, team_map, team_map_ok)
        tau = int(f) - _ANCHOR
        t_in = int(tau) - 1

        rec: dict[str, Any] = {
            "anon_match_key": f"m{int(match_local):05d}",
            "match_local": int(match_local), "component_id": int(lab),
            "member_key": f"m{int(match_local):05d}:c{int(lab)}",
            "n_members": int(memb.size), "same_ms_first_n": int(memb0.size),
            "f": int(f), "L": int(L), "tau": tau, "t_in": t_in,
            "singleton": bool(memb.size == 1), "c0_bits": int(c0), "call_bits": int(call),
            "c0_n": int(c0.bit_count()), "call_n": int(call.bit_count()),
            "c0_n_credited": int(c0.bit_count()), "call_n_credited": int(call.bit_count()),
            "c0_n_env_killer": int(c0_env), "call_n_env_killer": int(call_env),
            "c0_n_blue": int(c0_teams["n_blue"]), "c0_n_red": int(c0_teams["n_red"]),
            "c0_n_unknown_team": int(c0_teams["n_unknown_team"]),
            "c0_scale_max_side": int(c0_teams["scale_max_side"]),
            "call_n_blue": int(call_teams["n_blue"]), "call_n_red": int(call_teams["n_red"]),
            "call_n_unknown_team": int(call_teams["n_unknown_team"]),
            "call_scale_max_side": int(call_teams["scale_max_side"]),
            "team_map_ok": bool(team_map_ok),
            "team_map_provenance": "cache_meta_team_map" if team_map_ok else (tm_reason or "unavailable"),
            "raw_presence_verified": False, "participation_mode": "cache_only_proxy",
            "events_support_ok": bool(events_ok), "events_support_reason": ev_reason or "",
            "a_strict": _SENTINEL, "ev_at_tau": _SENTINEL,
            "ev_between_Ftau_tau": _SENTINEL, "ev_between_Ftm1_tin": _SENTINEL,
            "ev_types_at_tau_n": _SENTINEL,
            "game_end_ms": int(ge_ms), "game_end_known": bool(ge_known),
            "frame_support_ok": bool(usable is not None),
            "terminal_support_authoritative": bool(ge_known),
            "next_component_tau": int(next_tau_by_lab[int(lab)]),
            "next_kill_after_L": int(np.min(all_t[all_t > int(L)])) if np.any(all_t > int(L)) else _SENTINEL,
            "last_frame_ms": int(last_frame), "support_start_ms": int(support_start), "q_pre": int(t_in),
            "overlap_next_tau_le_L": bool(
                next_tau_present_by_lab[int(lab)] and next_tau_by_lab[int(lab)] <= int(L)
            ),
            "missing_terminal_game_end": bool(not ge_known),
            "first_kill_x": locs_x.copy(), "first_kill_y": locs_y.copy(),
        }

        member_set = set(int(i) for i in memb.tolist())
        member_x = all_x[memb].astype(np.float64, copy=False)
        member_y = all_y[memb].astype(np.float64, copy=False)
        foreign = foreign_near = foreign_distant = 0
        for i in range(n):
            if i in member_set or not (tau <= int(all_t[i]) <= int(L)):
                continue
            foreign += 1
            if member_x.size:
                d = float(np.sqrt((member_x - all_x[i]) ** 2 + (member_y - all_y[i]) ** 2).min())
                if d <= 2000.0:
                    foreign_near += 1
                else:
                    foreign_distant += 1
        other_anchors = 0
        for lab2, f2, _L2, _memb2 in comp_meta:
            if int(lab2) == int(lab):
                continue
            tau2 = int(f2) - _ANCHOR
            if tau <= tau2 <= int(L):
                other_anchors += 1
        rec["foreign_kills_in_tau_L"] = int(foreign)
        rec["foreign_near"] = int(foreign_near)
        rec["foreign_distant"] = int(foreign_distant)
        rec["other_anchors_in_tau_L"] = int(other_anchors)

        # Both independently selected cutoff frames and all boundary events.
        for cut_name, cutoff in (("tau", tau), ("tm1", t_in)):
            if usable is None:
                F = idx = age = None
                reason = frame_reason or "missing_frames"
            else:
                F, idx, age, reason = _preceding_frame(usable, cutoff)
            if reason or F is None or idx is None:
                rec[f"F_{cut_name}"] = _SENTINEL
                rec[f"F_{cut_name}_idx"] = _SENTINEL
                rec[f"lag_f_F_{cut_name}"] = _SENTINEL
                rec[f"{cut_name}_reason"] = reason or "no_preceding_frame"
            else:
                lag = int(f) - int(F)
                rec[f"F_{cut_name}"] = int(F)
                rec[f"F_{cut_name}_idx"] = int(idx)
                rec[f"lag_f_F_{cut_name}"] = lag
                rec[f"{cut_name}_reason"] = ""
                rec[f"cand_10_20_{cut_name}"] = bool(10_000 <= lag <= 20_000)
                if cut_name == "tau":
                    rec["exact_10000_tau"] = bool(lag == _ANCHOR and int(F) == tau)
                else:
                    rec["a_strict"] = int(age if age is not None else _SENTINEL)
                    rec["strict_eligible"] = bool(age is not None and 0 <= int(age) < _ANCHOR)
            if events_ok and event_ts is not None:
                at = np.flatnonzero(event_ts == int(cutoff))
                if cut_name == "tau":
                    rec["ev_at_tau"] = int(at.size)
                    rec["ev_types_at_tau_n"] = int(at.size)
                    rec["ev_between_Ftau_tau"] = (
                        _count_events_between(event_ts, int(rec["F_tau"]), tau)
                        if int(rec["F_tau"]) >= 0 else _SENTINEL
                    )
                else:
                    rec["ev_between_Ftm1_tin"] = (
                        _count_events_between(event_ts, int(rec["F_tm1"]), t_in)
                        if int(rec["F_tm1"]) >= 0 else _SENTINEL
                    )
            else:
                if cut_name == "tau":
                    rec["ev_at_tau"] = _SENTINEL
                    rec["ev_types_at_tau_n"] = _SENTINEL
                    rec["ev_between_Ftau_tau"] = _SENTINEL
                else:
                    rec["ev_between_Ftm1_tin"] = _SENTINEL
        rec["cand_10_20_tau"] = bool(rec.get("cand_10_20_tau", False))
        rec["cand_10_20_tm1"] = bool(rec.get("cand_10_20_tm1", False))
        rec["exact_10000_tau"] = bool(rec.get("exact_10000_tau", False))
        rec["strict_eligible"] = bool(rec.get("strict_eligible", False))
        rec["frame_differs"] = bool(
            int(rec.get("F_tau", _SENTINEL)) >= 0 and int(rec.get("F_tm1", _SENTINEL)) >= 0
            and int(rec["F_tau"]) != int(rec["F_tm1"])
        )

        # Every cutoff/radius receives either a cache-supported proxy or explicit bounds.
        for cut_name, idx_key in (("tau", "F_tau_idx"), ("tm1", "F_tm1_idx")):
            fi = int(rec.get(idx_key, _SENTINEL))
            unsupported = pack is None or usable is None or fi < 0
            xy = hp = None
            if not unsupported:
                xy = np.asarray(pack["xy_raw_minute"][fi], dtype=np.float64)
                hp = np.asarray(pack["node_minute"][fi, :, C.IDX_HP_PCT], dtype=np.float64)
            for radius in (1800.0, 2000.0):
                tag = f"{cut_name}_r{int(radius)}"
                part = _participation_from_frame(
                    np.zeros((10, 2)) if xy is None else xy,
                    np.zeros(10) if hp is None else hp,
                    locs_x, locs_y, radius, c0, unsupported=unsupported,
                )
                _set_participation_record(rec, tag, part)
                rec[f"_dists_{tag}"] = np.asarray(part["dists"], dtype=np.float64).copy()
                rec[f"_pos_codes_{tag}"] = np.asarray(part["pos_codes"], dtype=np.int8).copy()
                rec[f"_hp_codes_{tag}"] = np.asarray(part["hp_codes"], dtype=np.int8).copy()
                if cut_name == "tau" and int(radius) == 1800:
                    rec["_frame_xy_tau"] = np.full((10, 2), np.nan) if xy is None else xy.copy()
                    rec["_hp_tau"] = np.full(10, np.nan) if hp is None else hp.copy()
                    rec["_frame_ts_tau"] = int(rec.get("F_tau", _SENTINEL))
                    rec["_frame_idx_tau"] = int(rec.get("F_tau_idx", _SENTINEL))
                if cut_name == "tm1" and int(radius) == 1800:
                    rec["_frame_xy_tm1"] = np.full((10, 2), np.nan) if xy is None else xy.copy()
                    rec["_hp_tm1"] = np.full(10, np.nan) if hp is None else hp.copy()
                    rec["_frame_ts_tm1"] = int(rec.get("F_tm1", _SENTINEL))
                    rec["_frame_idx_tm1"] = int(rec.get("F_tm1_idx", _SENTINEL))
        rec["radius_mono_p_lower_tau"] = bool(
            rec["support_ok_tau_r1800"] and rec["support_ok_tau_r2000"]
            and int(rec["p_lower_n_tau_r1800"]) <= int(rec["p_lower_n_tau_r2000"])
        )
        rec["radius_mono_p_upper_tau"] = bool(
            rec["support_ok_tau_r1800"] and rec["support_ok_tau_r2000"]
            and int(rec["p_upper_n_tau_r1800"]) <= int(rec["p_upper_n_tau_r2000"])
        )

        # Source endpoint candidates and validity.  Baseline rows only appear as
        # comparison aliases and never select the endpoint used below.
        bi = baseline_by_lab.get(int(lab))
        for h in (60, 90, 120):
            e_ref = _endpoint_ms(
                int(L), h, int(rec["next_kill_after_L"]), int(rec["next_component_tau"]), int(ge_ms), bool(ge_known)
            )
            valid_ref, invalid_reason = _endpoint_validity(
                e_ref, int(L), tau, support_start, last_frame,
                bool(rec["overlap_next_tau_le_L"]), bool(ge_known),
            )
            rec[f"e_h{h}"] = int(e_ref)
            rec[f"e_h{h}_recomputed"] = int(e_ref)
            rec[f"valid_h{h}"] = int(valid_ref)
            rec[f"endpoint_valid_h{h}"] = bool(valid_ref)
            rec[f"endpoint_invalid_reason_h{h}"] = invalid_reason
            rec[f"e_h{h}_ref"] = int(e_ref)
            rec[f"valid_h{h}_ref"] = int(valid_ref)
            if bi is not None and baseline_rows is not None:
                rec[f"e_h{h}_baseline"] = int(baseline_rows[f"e_h{h}"][bi]) if f"e_h{h}" in baseline_rows else None
                rec[f"valid_h{h}_baseline"] = int(baseline_rows[f"valid_h{h}"][bi]) if f"valid_h{h}" in baseline_rows else None
            for cut_name, F_key, idx_key, cutoff_key in (
                ("tau", "F_tau", "F_tau_idx", "tau"), ("tm1", "F_tm1", "F_tm1_idx", "t_in"),
            ):
                tag = f"{cut_name}_h{h}"
                F_pre = int(rec[F_key]); i_pre = int(rec[idx_key]); cutoff_ms = int(rec[cutoff_key])
                _empty_post(rec, tag, "")
                if not valid_ref:
                    _empty_post(rec, tag, "invalid_endpoint")
                    continue
                F_post, i_post, age_post, post_reason = _preceding_frame(usable, e_ref) if usable is not None else (None, None, None, "missing_frames")
                if post_reason or F_post is None or i_post is None:
                    _empty_post(rec, tag, post_reason or "no_post_frame")
                    continue
                rec[f"F_post_{tag}"] = int(F_post)
                rec[f"F_post_idx_{tag}"] = int(i_post)
                rec[f"age_post_{tag}"] = int(age_post if age_post is not None else _SENTINEL)
                rec[f"same_as_Fpre_{tag}"] = bool(F_pre >= 0 and int(F_post) == F_pre)
                rec[f"frame_idx_gap_{tag}"] = int(i_post - i_pre) if F_pre >= 0 and i_pre >= 0 else _SENTINEL
                rec[f"timestamp_gap_{tag}"] = int(F_post - F_pre) if F_pre >= 0 else _SENTINEL
                rec[f"F_post_gt_L_{tag}"] = bool(int(F_post) > int(L))
                rec[f"stale_at_or_before_L_{tag}"] = bool(int(F_post) <= int(L))
                if events_ok and event_ts is not None:
                    rec[f"events_support_ok_{tag}"] = True
                    rec[f"async_events_Fpre_to_Fpost_{tag}"] = _count_events_between(event_ts, F_pre, int(F_post)) if F_pre >= 0 else _SENTINEL
                    rec[f"async_events_cutoff_to_endpoint_{tag}"] = _count_events_between(event_ts, cutoff_ms, e_ref)
                    rec[f"post_support_reason_{tag}"] = ""
                else:
                    rec[f"events_support_ok_{tag}"] = False
                    rec[f"async_events_Fpre_to_Fpost_{tag}"] = _SENTINEL
                    rec[f"async_events_cutoff_to_endpoint_{tag}"] = _SENTINEL
                    rec[f"post_support_reason_{tag}"] = ev_reason or "events_unavailable"
                if i_pre >= 0 and i_post >= 0:
                    g0 = np.asarray(pack["gold_team_minute"][i_pre], dtype=np.float64)
                    g1 = np.asarray(pack["gold_team_minute"][i_post], dtype=np.float64)
                    lv0 = np.asarray(pack["node_minute"][i_pre, :, C.IDX_LEVEL_NORM], dtype=np.float64)
                    lv1 = np.asarray(pack["node_minute"][i_post, :, C.IDX_LEVEL_NORM], dtype=np.float64)
                    xp0 = np.asarray(pack["node_minute"][i_pre, :, C.IDX_XP_NORM], dtype=np.float64)
                    xp1 = np.asarray(pack["node_minute"][i_post, :, C.IDX_XP_NORM], dtype=np.float64)
                    hp0 = np.asarray(pack["node_minute"][i_pre, :, C.IDX_HP_PCT], dtype=np.float64)
                    hp1 = np.asarray(pack["node_minute"][i_post, :, C.IDX_HP_PCT], dtype=np.float64)
                    rec[f"d_gold_blue_{tag}"] = float(g1[0] - g0[0]); rec[f"d_gold_red_{tag}"] = float(g1[1] - g0[1])
                    rec[f"d_level_mean_{tag}"] = float(np.mean(lv1 - lv0)); rec[f"d_xp_mean_{tag}"] = float(np.mean(xp1 - xp0)); rec[f"d_hp_mean_{tag}"] = float(np.mean(hp1 - hp0))
                    rec[f"_pre_gold_{tag}"] = g0.copy(); rec[f"_post_gold_{tag}"] = g1.copy()
                    rec[f"_pre_level_mean_{tag}"] = float(np.mean(lv0)); rec[f"_post_level_mean_{tag}"] = float(np.mean(lv1))
                    rec[f"_pre_xp_mean_{tag}"] = float(np.mean(xp0)); rec[f"_post_xp_mean_{tag}"] = float(np.mean(xp1))
                    rec[f"_pre_hp_mean_{tag}"] = float(np.mean(hp0)); rec[f"_post_hp_mean_{tag}"] = float(np.mean(hp1))
                else:
                    for field in REFERENCE_POST_FLOAT_FIELDS:
                        rec[f"{field}_{tag}"] = float("nan")
        components.append(rec)

    source_status["n_components"] = len(components)
    return {
        "match_local": int(match_local), "match_id": str(match_id), "n_kills": n,
        "n_components": len(components), "components": components,
        "source_hashes": source_hashes, "source_status": source_status,
        "input_hashes": input_hashes, "ok": True,
    }


def compare_table_to_reference(
    table: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
    reference_rows: list[dict[str, Any]],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Compare one production match/batch against its independent flat rows.

    The caller must pass only the current match/batch.  A local key map is used
    for that bounded scope and is discarded with the returned report.  Missing
    fields and unsupported rows are still counted as expected comparisons.
    """
    def _eq(got: Any, exp: Any) -> bool:
        if got is None or exp is None:
            return got is None and exp is None
        ga = np.asarray(got) if isinstance(got, (list, tuple, np.ndarray)) else None
        ea = np.asarray(exp) if isinstance(exp, (list, tuple, np.ndarray)) else None
        if ga is not None or ea is not None:
            if ga is None or ea is None or ga.shape != ea.shape:
                return False
            if np.issubdtype(ga.dtype, np.floating) or np.issubdtype(ea.dtype, np.floating):
                try:
                    return bool(np.allclose(ga.astype(float), ea.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True))
                except (TypeError, ValueError):
                    return False
            return bool(np.array_equal(ga, ea))
        if isinstance(got, (float, np.floating)) or isinstance(exp, (float, np.floating)):
            try:
                gf, ef = float(got), float(exp)
                return bool((math.isnan(gf) and math.isnan(ef)) or math.isclose(gf, ef, rel_tol=1e-9, abs_tol=1e-9))
            except (TypeError, ValueError):
                return False
        return bool(got == exp)

    n = int(np.asarray(table.get("match_local", np.zeros(0))).shape[0])
    errors: list[dict[str, Any]] = []
    compared: dict[str, int] = {}
    mismatches: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    missing_table_fields: list[str] = []
    missing_reference_fields: dict[str, int] = {}

    def bump(mapping: dict[str, int], field: str, amount: int = 1) -> None:
        mapping[field] = int(mapping.get(field, 0)) + int(amount)

    def fail(payload: dict[str, Any]) -> None:
        if len(errors) < 100:
            errors.append(payload)

    table_keys: list[tuple[int, int]] = []
    if "match_local" in table and "component_id" in table:
        table_keys = [
            (int(np.asarray(table["match_local"])[i]), int(np.asarray(table["component_id"])[i]))
            for i in range(n)
        ]
    table_key_set = set(table_keys)
    table_key_counts: dict[tuple[int, int], int] = {}
    for key in table_keys:
        table_key_counts[key] = table_key_counts.get(key, 0) + 1
    table_duplicates = sorted(key for key, count in table_key_counts.items() if count > 1)
    ref_keys: list[tuple[int, int]] = []
    ref_duplicate_keys: list[tuple[int, int]] = []
    ref_map: dict[tuple[int, int], dict[str, Any]] = {}
    for row in reference_rows:
        if not isinstance(row, Mapping) or "match_local" not in row or "component_id" not in row:
            fail({"kind": "reference_row_missing_key"})
            continue
        key = (int(row["match_local"]), int(row["component_id"]))
        ref_keys.append(key)
        if key in ref_map:
            ref_duplicate_keys.append(key)
        ref_map[key] = dict(row)
    ref_key_set = set(ref_keys)
    missing_ref = sorted(table_key_set - ref_key_set)
    extra_ref = sorted(ref_key_set - table_key_set)
    if table_duplicates:
        fail({"kind": "duplicate_table_keys", "keys": table_duplicates[:20], "n": len(table_duplicates)})
    if ref_duplicate_keys:
        fail({"kind": "duplicate_reference_keys", "keys": ref_duplicate_keys[:20], "n": len(ref_duplicate_keys)})
    if missing_ref:
        fail({"kind": "missing_reference_rows", "n": len(missing_ref), "sample": missing_ref[:20]})
    if extra_ref:
        fail({"kind": "extra_reference_rows", "n": len(extra_ref), "sample": extra_ref[:20]})

    def compare_field(field: str, row_index: int, ref: Mapping[str, Any] | None, key: tuple[int, int]) -> None:
        expected_counts[field] = expected_counts.get(field, 0) + 1
        if field not in table:
            if field not in missing_table_fields:
                missing_table_fields.append(field)
            fail({"kind": "missing_table_field", "field": field, "key": key, "row": row_index})
            return
        if ref is None or field not in ref:
            missing_reference_fields[field] = missing_reference_fields.get(field, 0) + 1
            fail({"kind": "missing_reference_field", "field": field, "key": key, "row": row_index})
            return
        got = np.asarray(table[field])[row_index]
        exp = ref[field]
        bump(compared, field)
        if not _eq(got, exp):
            bump(mismatches, field)
            fail({"kind": "field_mismatch", "field": field, "row": row_index, "key": key, "got": got, "ref": exp})

    row_refs = [ref_map.get(key) for key in table_keys]

    def compare_scalar_column(field: str) -> None:
        expected_counts[field] = expected_counts.get(field, 0) + n
        if field not in table:
            if field not in missing_table_fields:
                missing_table_fields.append(field)
            fail({"kind": "missing_table_field", "field": field, "n": n})
            return
        present = np.asarray([i for i, ref in enumerate(row_refs) if ref is not None and field in ref], dtype=np.int64)
        absent = n - int(present.size)
        if absent:
            missing_reference_fields[field] = missing_reference_fields.get(field, 0) + absent
            fail({"kind": "missing_reference_field", "field": field, "n": absent})
        if present.size == 0:
            return
        got = np.asarray(table[field])[present]
        exp_values = [row_refs[int(i)][field] for i in present.tolist()]  # type: ignore[index]
        has_none = any(v is None for v in exp_values)
        if has_none:
            eq = np.asarray([_eq(got[j], exp_values[j]) for j in range(len(exp_values))], dtype=bool)
        else:
            try:
                exp = np.asarray(exp_values)
                if np.issubdtype(got.dtype, np.floating) or np.issubdtype(exp.dtype, np.floating):
                    eq = np.isclose(got.astype(float), exp.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
                else:
                    eq = np.asarray(got == exp, dtype=bool)
            except (TypeError, ValueError):
                eq = np.asarray([_eq(got[j], exp_values[j]) for j in range(len(exp_values))], dtype=bool)
        bump(compared, field, int(present.size))
        bad = np.flatnonzero(~eq)
        if bad.size:
            bump(mismatches, field, int(bad.size))
            for j in bad[:10].tolist():
                row_i = int(present[j])
                fail({"kind": "field_mismatch", "field": field, "row": row_i, "key": table_keys[row_i], "got": got[j], "ref": exp_values[j]})

    for field in REFERENCE_TABLE_FIELDS:
        compare_scalar_column(field)
    # Compare additional scalar fields when a newer storage schema exposes them.
    extra_fields: set[str] = set()
    for ref in row_refs:
        if ref is not None:
            extra_fields.update(str(k) for k in ref if not str(k).startswith("_") and k not in REFERENCE_TABLE_FIELDS)
    for field in sorted(extra_fields.intersection(table)):
        compare_scalar_column(field)

    # Evidence arrays are compared per row and per player.  This catches a
    # same-size bitset mutation even when scalar population counts are unchanged.
    def evidence_row_value(name: str, i: int, ref: Mapping[str, Any] | None) -> Any:
        if ref is None:
            return None
        if name == "first_loc_x":
            return ref.get("first_kill_x", ref.get("first_loc_x"))
        if name == "first_loc_y":
            return ref.get("first_kill_y", ref.get("first_loc_y"))
        aliases = {
            "dist_": "_dists_", "pos_codes_": "_pos_codes_", "hp_codes_": "_hp_codes_",
            "frame_xy_": "_frame_xy_", "hp_": "_hp_",
        }
        if name in ("frame_xy_tau", "frame_xy_tm1", "hp_tau", "hp_tm1"):
            return ref.get(f"_{name}")
        if name in ("frame_ts_tau", "frame_ts_tm1", "frame_idx_tau", "frame_idx_tm1"):
            return ref.get(f"_{name}")
        if name.startswith(("pre_gold_", "post_gold_", "pre_level_mean_", "post_level_mean_", "pre_xp_mean_", "post_xp_mean_", "pre_hp_mean_", "post_hp_mean_")):
            return ref.get(f"_{name}")
        for prefix, alias in aliases.items():
            if name.startswith(prefix):
                return ref.get(alias + name[len(prefix):])
        return ref.get(name)

    offsets = np.asarray(evidence.get("first_loc_offsets", np.zeros(n + 1, dtype=np.int64)))
    offsets_ok = (
        offsets.ndim == 1 and offsets.size == n + 1 and offsets.size > 0
        and int(offsets[0]) == 0 and np.all(offsets >= 0)
        and np.all(offsets[:-1] <= offsets[1:])
    )
    if not offsets_ok:
        fail({"kind": "invalid_first_location_offsets", "shape": tuple(offsets.shape), "n_rows": n})
    for i in range(n):
        key = table_keys[i] if i < len(table_keys) else (_SENTINEL, _SENTINEL)
        ref = ref_map.get(key)
        if offsets_ok:
            a, b = int(offsets[i]), int(offsets[i + 1])
        else:
            a = b = 0
        for name in ("first_loc_x", "first_loc_y"):
            expected_counts[name] = expected_counts.get(name, 0) + 1
            if name not in evidence:
                fail({"kind": "missing_evidence_field", "field": name, "key": key, "row": i})
                continue
            got = np.asarray(evidence[name])[a:b]
            exp = evidence_row_value(name, i, ref)
            if exp is None:
                missing_reference_fields[name] = missing_reference_fields.get(name, 0) + 1
                fail({"kind": "missing_reference_evidence", "field": name, "key": key, "row": i})
                continue
            bump(compared, name)
            if not _eq(got, exp):
                bump(mismatches, name)
                fail({"kind": "evidence_mismatch", "field": name, "key": key, "row": i})
        for name in REFERENCE_EVIDENCE_FIELDS:
            if name in ("first_loc_offsets", "first_loc_x", "first_loc_y"):
                continue
            expected_counts[name] = expected_counts.get(name, 0) + 1
            if name not in evidence:
                fail({"kind": "missing_evidence_field", "field": name, "key": key, "row": i})
                continue
            try:
                got = np.asarray(evidence[name])[i]
            except (IndexError, TypeError):
                fail({"kind": "invalid_evidence_shape", "field": name, "key": key, "row": i})
                continue
            exp = evidence_row_value(name, i, ref)
            if exp is None:
                missing_reference_fields[name] = missing_reference_fields.get(name, 0) + 1
                fail({"kind": "missing_reference_evidence", "field": name, "key": key, "row": i})
                continue
            bump(compared, name)
            if not _eq(got, exp):
                bump(mismatches, name)
                fail({"kind": "evidence_mismatch", "field": name, "key": key, "row": i})
    for name in REFERENCE_EVIDENCE_FIELDS:
        if name in ("first_loc_offsets", "first_loc_x", "first_loc_y") or name not in evidence:
            continue
        arr = np.asarray(evidence[name])
        if arr.ndim == 0 or int(arr.shape[0]) != n:
            fail({"kind": "invalid_evidence_row_coverage", "field": name, "shape": tuple(arr.shape), "n_rows": n})
    expected_counts["first_loc_offsets"] = n + 1
    if "first_loc_offsets" in evidence and np.asarray(evidence["first_loc_offsets"]).shape == (n + 1,):
        expected_offsets = np.zeros(n + 1, dtype=np.int64)
        for i, ref in enumerate(row_refs):
            if ref is not None:
                xs = ref.get("first_kill_x", ref.get("first_loc_x", []))
                expected_offsets[i + 1] = expected_offsets[i] + int(np.asarray(xs).size)
            else:
                expected_offsets[i + 1] = expected_offsets[i]
        expected_total = int(expected_offsets[-1])
        bump(compared, "first_loc_offsets")
        actual_x = np.asarray(evidence.get("first_loc_x", np.zeros(0)))
        actual_y = np.asarray(evidence.get("first_loc_y", np.zeros(0)))
        if (
            not offsets_ok
            or int(offsets[-1]) != expected_total
            or actual_x.size != expected_total
            or actual_y.size != expected_total
            or not np.array_equal(np.asarray(evidence["first_loc_offsets"], dtype=np.int64), expected_offsets)
        ):
            bump(mismatches, "first_loc_offsets")
            fail({"kind": "first_location_coverage_mismatch", "got": np.asarray(evidence["first_loc_offsets"]).tolist(), "ref": expected_offsets.tolist()})

    unsupported_rows = 0
    for row in reference_rows:
        if isinstance(row, Mapping) and not any(bool(row.get(f"support_ok_{tag}", False)) for tag in REFERENCE_PARTICIPATION_TAGS):
            unsupported_rows += 1
    n_compared = int(sum(compared.values()))
    n_mismatch = int(sum(mismatches.values()))
    required_missing = bool(missing_table_fields or missing_reference_fields or missing_ref or extra_ref or table_duplicates or ref_duplicate_keys)
    coverage_ok = bool(
        not required_missing
        and not any(expected_counts.get(field, 0) != compared.get(field, 0) for field in expected_counts if field != "first_loc_offsets")
        and (expected_counts.get("first_loc_offsets", 0) == 0 or compared.get("first_loc_offsets", 0) == 1)
    )
    return {
        "n_rows": n,
        "n_reference_rows": len(reference_rows),
        "n_field_comparisons": n_compared,
        "n_field_mismatches": n_mismatch,
        "compared_by_field": compared,
        "mismatches_by_field": mismatches,
        "expected_comparisons_by_field": expected_counts,
        "missing_table_fields": sorted(missing_table_fields),
        "missing_reference_fields": dict(sorted(missing_reference_fields.items())),
        "n_error_examples": len(errors), "error_examples": errors,
        "missing_reference_rows": len(missing_ref), "extra_reference_rows": len(extra_ref),
        "duplicate_table_keys": len(table_duplicates), "duplicate_reference_keys": len(ref_duplicate_keys),
        "unsupported_reference_rows": int(unsupported_rows),
        "coverage_ok": coverage_ok,
        "all_ok": bool(coverage_ok and n_mismatch == 0 and not errors),
        "smoke": bool(smoke),
        "verification_mode": "parent_dependent_source_replay",
        "note": "Reference reopen of immutable ST ledger/assignments and per-match cache NPZ/events/meta; baseline is comparison-only.",
    }


def _reference_arrays_from_rows(reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Materialize only typed scalar reference arrays for aggregate recompute."""
    rows = list(reference_rows)
    n = len(rows)
    out: dict[str, np.ndarray] = {}
    for field in REFERENCE_TABLE_FIELDS:
        meta = REFERENCE_FIELD_METADATA.get(field, {})
        kind = meta.get("kind", "float")
        vals = [row.get(field) for row in rows]
        if kind == "string":
            out[field] = np.asarray(["" if v is None else str(v) for v in vals], dtype="U96")
        elif kind in ("int", "bool"):
            out[field] = np.asarray(
                [int(bool(v)) if kind == "bool" and v is not None else (-1 if v is None else int(v)) for v in vals],
                dtype=np.int64,
            )
        else:
            out[field] = np.asarray([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
    return out


def reference_rows_to_arrays(reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Public alias used by the bounded runner before aggregate recomputation."""
    return _reference_arrays_from_rows(reference_rows)


def _array_or_default(table: Mapping[str, np.ndarray], name: str, n: int, dtype: Any, default: Any) -> np.ndarray:
    if name not in table:
        return np.full(n, default, dtype=dtype)
    return np.asarray(table[name], dtype=dtype)


def _reference_weighted_stats(
    values: np.ndarray, population: np.ndarray, match_local: np.ndarray, support: np.ndarray | None = None
) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    pop = np.asarray(population, dtype=bool)
    known = pop & np.isfinite(v) if support is None else pop & np.asarray(support, dtype=bool) & np.isfinite(v)
    n_population = int(pop.sum())
    n_known = int(known.sum())
    n_unknown = int(n_population - n_known) if n_population else 0
    comp_mean = float(np.mean(v[known])) if n_known else None
    comp_median = float(np.median(v[known])) if n_known else None
    n_matches = 0
    match_mean = match_median = None
    if n_known:
        used = np.asarray(match_local, dtype=np.int64)[known]
        vals = v[known]
        _, inv, counts = np.unique(used, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inv].astype(np.float64)
        n_matches = int(np.unique(used).size)
        match_mean = float(np.sum(weights * vals) / np.sum(weights))
        match_median = _inverse_weighted_cdf_quantile(vals, weights, 0.5)
    return {
        "n_population": n_population, "n_known": n_known, "n_unknown_excluded": n_unknown,
        "n_matches_known": n_matches,
        "support_rate": (float(n_known) / float(n_population)) if n_population else None,
        "component_equal_mean": comp_mean, "component_equal_median": comp_median,
        "equal_total_match_mean": match_mean, "equal_total_match_median": match_median,
    }


def _reference_gap(lags: np.ndarray) -> dict[str, Any]:
    d = np.asarray(lags, dtype=np.int64)
    known = d >= 0
    return {
        "exact_10000": int(np.sum(known & (d == 10_000))),
        "cum_10_15": int(np.sum(known & (d >= 10_000) & (d <= 15_000))),
        "gt15_le20": int(np.sum(known & (d > 15_000) & (d <= 20_000))),
        "cum_10_20": int(np.sum(known & (d >= 10_000) & (d <= 20_000))),
        "n_known_lag": int(np.sum(known)), "n_unknown_lag": int(np.sum(~known)),
    }


def recompute_aggregates_reference(
    table_or_rows: Mapping[str, np.ndarray] | Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
    *, n_matches_source: int | None = None, n_zero_kill: int | None = None,
    n_matches_processed: int | None = None, n_zero_kill_processed: int | None = None,
) -> dict[str, Any]:
    """Recompute the complete report tree from independent typed reference arrays.

    ``reference_rows`` is the preferred path.  The mapping-only form remains for
    compatibility with the original focused fixtures and is marked as a
    deprecated table fallback by the returned metadata.
    """
    from_rows = reference_rows is not None or not isinstance(table_or_rows, Mapping)
    if from_rows:
        rows = list(reference_rows if reference_rows is not None else table_or_rows)  # type: ignore[arg-type]
        table = _reference_arrays_from_rows(rows)
        mode = "independent_reference_rows"
    else:
        table = {k: np.asarray(v) for k, v in table_or_rows.items()}  # type: ignore[union-attr]
        mode = "deprecated_table_fallback"
    n = int(np.asarray(table.get("match_local", np.zeros(0))).shape[0])
    ml = _array_or_default(table, "match_local", n, np.int64, -1)
    lag_tau = _array_or_default(table, "lag_f_F_tau", n, np.int64, -1)
    lag_tm1 = _array_or_default(table, "lag_f_F_tm1", n, np.int64, -1)
    foreign = _array_or_default(table, "foreign_kills_in_tau_L", n, np.int64, 0)
    other = _array_or_default(table, "other_anchors_in_tau_L", n, np.int64, 0)
    cand_tau = (lag_tau >= 10_000) & (lag_tau <= 20_000)
    cand_tm1 = _array_or_default(table, "cand_10_20_tm1", n, bool, False)
    isol_tau = cand_tau & (foreign == 0) & (other == 0)
    isol_tm1 = cand_tm1 & (foreign == 0) & (other == 0)
    out: dict[str, Any] = {
        "n_components": n,
        "n_matches_source": int(n_matches_source if n_matches_source is not None else np.unique(ml[ml >= 0]).size),
        "n_zero_kill_matches_source": int(n_zero_kill or 0),
        "n_matches_processed": int(n_matches_processed if n_matches_processed is not None else np.unique(ml[ml >= 0]).size),
        "n_zero_kill_processed": int(n_zero_kill_processed if n_zero_kill_processed is not None else (n_zero_kill or 0)),
        "n_matches_with_components": int(np.unique(ml[ml >= 0]).size) if n else 0,
        "gap_tau": _reference_gap(lag_tau), "gap_tm1": _reference_gap(lag_tm1),
        "n_cand_10_20_tau": int(cand_tau.sum()), "n_isolated_cand_10_20_tau": int(isol_tau.sum()),
        "n_exact_10000_tau": int(_array_or_default(table, "exact_10000_tau", n, np.int8, 0).sum()),
        "n_strict_eligible": int(_array_or_default(table, "strict_eligible", n, np.int8, 0).sum()),
        "n_frame_differs": int(_array_or_default(table, "frame_differs", n, np.int8, 0).sum()),
        "reconcile_expected": {
            "inclusive_10_20": C.EXPECTED_INCLUSIVE_10_20,
            "strict_eligible": C.EXPECTED_STRICT_ELIGIBLE,
            "exact_10000": C.EXPECTED_EXACT_10000,
            "n_components": C.EXPECTED_MIN1_COMPONENTS,
        },
    }
    out["reconcile_ok"] = bool(
        out["n_cand_10_20_tau"] == C.EXPECTED_INCLUSIVE_10_20
        and out["n_strict_eligible"] == C.EXPECTED_STRICT_ELIGIBLE
        and out["n_exact_10000_tau"] == C.EXPECTED_EXACT_10000
        and n == C.EXPECTED_MIN1_COMPONENTS
    )
    singleton = _array_or_default(table, "singleton", n, np.int8, 0)

    def part_metrics(tag: str, pop: np.ndarray) -> dict[str, Any]:
        support = _array_or_default(table, f"support_ok_{tag}", n, bool, False)
        pl = _array_or_default(table, f"p_lower_n_{tag}", n, np.float64, np.nan)
        pu = _array_or_default(table, f"p_upper_n_{tag}", n, np.float64, np.nan)
        nl = _array_or_default(table, f"n_lower_{tag}", n, np.float64, np.nan)
        nu = _array_or_default(table, f"n_upper_{tag}", n, np.float64, np.nan)
        return {
            "n_population": int(pop.sum()), "n_support_ok": int((pop & support).sum()),
            "p_lower": _reference_weighted_stats(pl, pop, ml, support & (pl >= 0)),
            "p_upper": _reference_weighted_stats(pu, pop, ml, support & (pu >= 0)),
            "n_lower": _reference_weighted_stats(nl, pop, ml, support & (nl >= 0)),
            "n_upper": _reference_weighted_stats(nu, pop, ml, support & (nu >= 0)),
            "singleton_rate_known": float(np.mean(singleton[pop & support])) if np.any(pop & support) else None,
        }

    cells: dict[str, Any] = {}
    full = np.ones(n, dtype=bool)
    for cut, cand, isol in (("tau", cand_tau, isol_tau), ("tm1", cand_tm1, isol_tm1)):
        for radius in (1800, 2000):
            tag = f"{cut}_r{radius}"
            candidate_metrics = part_metrics(tag, cand)
            cells[tag] = {
                "n_population_full": int(full.sum()), "n_population_cand": int(cand.sum()),
                "n_population_isolated": int(isol.sum()),
                "full": part_metrics(tag, full), "candidate": candidate_metrics, "isolated": part_metrics(tag, isol),
                "n_support_ok": candidate_metrics["n_support_ok"],
                "p_lower": candidate_metrics["p_lower"], "p_upper": candidate_metrics["p_upper"],
                "n_lower": candidate_metrics["n_lower"], "n_upper": candidate_metrics["n_upper"],
            }
    out["participation_cells"] = cells
    post: dict[str, Any] = {}
    for h in (60, 90, 120):
        valid = _array_or_default(table, f"valid_h{h}", n, bool, False)
        for cut, cand in (("tau", cand_tau), ("tm1", cand_tm1)):
            tag = f"{cut}_h{h}"
            base = cand & valid
            same = _array_or_default(table, f"same_as_Fpre_{tag}", n, bool, False)
            gt_l = _array_or_default(table, f"F_post_gt_L_{tag}", n, bool, False)
            stale = _array_or_default(table, f"stale_at_or_before_L_{tag}", n, bool, False)
            ev_ok = _array_or_default(table, f"events_support_ok_{tag}", n, bool, False)
            async_f = _array_or_default(table, f"async_events_Fpre_to_Fpost_{tag}", n, np.float64, np.nan)
            async_c = _array_or_default(table, f"async_events_cutoff_to_endpoint_{tag}", n, np.float64, np.nan)
            d_gold = _array_or_default(table, f"d_gold_blue_{tag}", n, np.float64, np.nan)
            d_level = _array_or_default(table, f"d_level_mean_{tag}", n, np.float64, np.nan)
            d_hp = _array_or_default(table, f"d_hp_mean_{tag}", n, np.float64, np.nan)
            post[tag] = {
                "n_cand_valid": int(base.sum()), "n_same_as_Fpre": int((base & same).sum()),
                "n_F_post_gt_L": int((base & gt_l).sum()), "n_stale_at_or_before_L": int((base & stale).sum()),
                "async_events_Fpre_to_Fpost": _reference_weighted_stats(async_f, base, ml, ev_ok & (async_f >= 0)),
                "async_events_cutoff_to_endpoint": _reference_weighted_stats(async_c, base, ml, ev_ok & (async_c >= 0)),
                "d_gold_blue": _reference_weighted_stats(d_gold, base, ml),
                "d_level_mean": _reference_weighted_stats(d_level, base, ml),
                "d_hp_mean": _reference_weighted_stats(d_hp, base, ml),
                "same_frame_with_async_events": int((base & same & ev_ok & np.isfinite(async_c) & (async_c > 0)).sum()),
                "n_async_events_Fpre_to_Fpost_positive": int((base & ev_ok & np.isfinite(async_f) & (async_f > 0)).sum()),
            }
    out["post_frame"] = post
    out["channel_units"] = {
        "gold_team_minute": "cache gold_team_minute float32 units (not claimed raw gold)",
        "level_norm": "node_minute[:, :, level_norm] normalized /18-style cache units",
        "xp_norm": "node_minute[:, :, xp_norm] normalized cache units",
        "hp_pct": "node_minute[:, :, hp_pct] fraction cache units; 0 is ambiguous_zero",
    }
    out["raw_presence_verified"] = False
    out["definitive_n0"] = False
    out["note"] = (
        "P_lower/P_upper are CACHE-ONLY proxies; not actual participants or definitive N0/P0 sample counts. "
        "Geometric near_bits are not participation near-only; use part_*_lower/upper decompositions."
    )
    if mode != "deprecated_table_fallback":
        out["verification_mode"] = mode
    return out


def compare_aggregates(ref_agg: Mapping[str, Any], prod_agg: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the full reference report tree, including None/number symmetry."""
    mismatches: list[dict[str, Any]] = []
    compared: dict[str, int] = {}

    def walk(ref: Any, prod: Any, path: str) -> None:
        if isinstance(ref, Mapping):
            if not isinstance(prod, Mapping):
                mismatches.append({"field": path, "got": prod, "ref": ref})
                return
            for key, value in ref.items():
                child = f"{path}.{key}" if path else str(key)
                if key not in prod:
                    mismatches.append({"field": child, "got": None, "ref": value})
                else:
                    walk(value, prod[key], child)
            return
        if isinstance(ref, (list, tuple)):
            if not isinstance(prod, (list, tuple)) or len(ref) != len(prod):
                mismatches.append({"field": path, "got": prod, "ref": ref})
                return
            for i, (rv, pv) in enumerate(zip(ref, prod)):
                walk(rv, pv, f"{path}[{i}]")
            return
        compared[path] = compared.get(path, 0) + 1
        if ref is None or prod is None:
            same = ref is None and prod is None
        elif isinstance(ref, (float, np.floating)) or isinstance(prod, (float, np.floating)):
            try:
                rf, pf = float(ref), float(prod)
                same = (math.isnan(rf) and math.isnan(pf)) or math.isclose(rf, pf, rel_tol=1e-9, abs_tol=1e-9)
            except (TypeError, ValueError):
                same = False
        else:
            same = ref == prod
        if not same:
            mismatches.append({"field": path, "got": prod, "ref": ref})

    walk(ref_agg, prod_agg, "")
    return {
        "n_compared_fields": int(sum(compared.values())),
        "compared_by_field": compared,
        "n_mismatches": len(mismatches), "mismatches": mismatches[:100],
        "all_ok": len(mismatches) == 0,
    }
