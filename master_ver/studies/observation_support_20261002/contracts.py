"""Pure contracts for observation-support diagnostics (no I/O)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import config as C


def preceding_frame(minute_ts: np.ndarray, t: int) -> tuple[Optional[int], Optional[int], Optional[int], str]:
    """F = max minute_ts <= t. Returns (F, frame_index, age=t-F, reason)."""
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


def cutoff_diagnostics(minute_ts: np.ndarray, f: int, anchor_offset_ms: int = C.ANCHOR_OFFSET_MS) -> dict[str, Any]:
    """Both τ−1 and τ cutoffs with lag and candidate flags."""
    f = int(f)
    tau = f - int(anchor_offset_ms)
    t_in = tau - 1
    out: dict[str, Any] = {
        "f": f,
        "tau": tau,
        "t_in": t_in,
        "negative_tau": bool(tau < 0),
        "F_tau": C.SENTINEL_MISSING,
        "F_tau_idx": C.SENTINEL_MISSING,
        "lag_f_F_tau": C.SENTINEL_MISSING,
        "F_tm1": C.SENTINEL_MISSING,
        "F_tm1_idx": C.SENTINEL_MISSING,
        "lag_f_F_tm1": C.SENTINEL_MISSING,
        "a_strict": C.SENTINEL_MISSING,
        "cand_10_20_tau": False,
        "cand_10_20_tm1": False,
        "exact_10000_tau": False,
        "strict_eligible": False,
        "tau_reason": "",
        "tm1_reason": "",
        "same_F": False,
        "frame_differs": False,
    }
    F_tau, i_tau, age_tau, r_tau = preceding_frame(minute_ts, tau)
    if r_tau:
        out["tau_reason"] = r_tau
    else:
        assert F_tau is not None and i_tau is not None
        lag = f - F_tau
        out["F_tau"] = int(F_tau)
        out["F_tau_idx"] = int(i_tau)
        out["lag_f_F_tau"] = int(lag)
        out["cand_10_20_tau"] = bool(10_000 <= lag <= 20_000)
        out["exact_10000_tau"] = bool(lag == int(anchor_offset_ms) and F_tau == tau)

    F_tm1, i_tm1, age_tm1, r_tm1 = preceding_frame(minute_ts, t_in)
    if r_tm1:
        out["tm1_reason"] = r_tm1
    else:
        assert F_tm1 is not None and i_tm1 is not None and age_tm1 is not None
        lag = f - F_tm1
        out["F_tm1"] = int(F_tm1)
        out["F_tm1_idx"] = int(i_tm1)
        out["lag_f_F_tm1"] = int(lag)
        out["a_strict"] = int(age_tm1)
        out["cand_10_20_tm1"] = bool(10_000 <= lag <= 20_000)
        out["strict_eligible"] = bool(0 <= age_tm1 < int(anchor_offset_ms))

    if out["F_tau"] >= 0 and out["F_tm1"] >= 0:
        out["same_F"] = bool(out["F_tau"] == out["F_tm1"])
        out["frame_differs"] = not out["same_F"]
    return out


def parse_timestamp_ms(value: Any) -> tuple[Optional[int], str]:
    if isinstance(value, bool):
        return None, "boolean_timestamp"
    if isinstance(value, int):
        if value < 0:
            return None, "negative_timestamp"
        return int(value), ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, "nonfinite_timestamp"
        if value != math.trunc(value):
            return None, "fractional_timestamp"
        if value < 0:
            return None, "negative_timestamp"
        return int(value), ""
    return None, "non_integral_timestamp"


def credit_bitset(killer: int, victim: int, assist_mask: int) -> int:
    """Bit (pid-1) for valid pids 1..10. Environmental killer 0 contributes no kill bit."""
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


def union_credit_bitset(
    killers: np.ndarray, victims: np.ndarray, assists: np.ndarray, member_idx: np.ndarray
) -> tuple[int, int]:
    """Return (bitset, n_env_killer) over member rows."""
    bits = 0
    n_env = 0
    for i in np.asarray(member_idx, dtype=np.int64).tolist():
        k = int(killers[i])
        if k == 0:
            n_env += 1
        bits |= credit_bitset(k, int(victims[i]), int(assists[i]))
    return int(bits), int(n_env)


def bitset_pids(bits: int) -> list[int]:
    return [pid for pid in range(1, 11) if int(bits) & (1 << (pid - 1))]


def team_counts_from_bitset(bits: int, team_map: Mapping[int, int]) -> dict[str, Any]:
    blue, red, unknown = [], [], []
    for pid in bitset_pids(bits):
        team = team_map.get(int(pid))
        if team == C.TEAM_BLUE:
            blue.append(pid)
        elif team == C.TEAM_RED:
            red.append(pid)
        else:
            unknown.append(pid)
    return {
        "n_blue": len(blue),
        "n_red": len(red),
        "n_unknown_team": len(unknown),
        "n_pids": len(blue) + len(red) + len(unknown),
        "scale_max_side": max(len(blue), len(red)),
        "blue": blue,
        "red": red,
        "unknown_team": unknown,
    }


def blue_pids_from_team_map(team_map: Mapping[Any, Any]) -> list[int]:
    out = []
    for k, v in team_map.items():
        try:
            pid = int(k)
            team = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= pid <= 10 and team == C.TEAM_BLUE:
            out.append(pid)
    return sorted(out)


def cache_position_state(x: float, y: float) -> int:
    """Cache XY tri-state. Actual numeric 0 is ambiguous (zero vs missing)."""
    if isinstance(x, bool) or isinstance(y, bool):
        return C.POS_UNKNOWN
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return C.POS_UNKNOWN
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return C.POS_NONFINITE
    if xf == 0.0 or yf == 0.0:
        return C.POS_AMBIGUOUS_ZERO
    return C.POS_KNOWN_NONZERO


def cache_health_state(hp_pct: float) -> int:
    if isinstance(hp_pct, bool):
        return C.HP_UNKNOWN
    try:
        v = float(hp_pct)
    except (TypeError, ValueError):
        return C.HP_UNKNOWN
    if not math.isfinite(v):
        return C.HP_NONFINITE
    if v < 0:
        return C.HP_NEGATIVE
    if v == 0.0:
        return C.HP_AMBIGUOUS_ZERO
    if v > 0:
        return C.HP_CACHED_POSITIVE
    return C.HP_UNKNOWN


def raw_health_state(value: Any) -> int:
    """PROTOCOL raw health tri-state from championStats.health."""
    if isinstance(value, bool) or value is None:
        return C.RAW_HP_UNKNOWN
    # Reject strings / nonnumeric before conversion (bool already excluded).
    if type(value) not in (int, float):
        return C.RAW_HP_UNKNOWN
    v = float(value)
    if not math.isfinite(v):
        return C.RAW_HP_UNKNOWN
    if v < 0:
        return C.RAW_HP_UNKNOWN
    if v == 0.0:
        return C.RAW_HP_OBSERVED_ZERO
    return C.RAW_HP_POSITIVE


def raw_position_state(x: Any, y: Any) -> int:
    """Raw finite numeric contract: reject bool/str/nonnumeric BEFORE conversion."""
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None:
        return C.RAW_POS_UNKNOWN
    # Strings (incl. numeric strings) and other non-int/float types are unknown.
    if type(x) not in (int, float) or type(y) not in (int, float):
        return C.RAW_POS_UNKNOWN
    xf = float(x)
    yf = float(y)
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return C.RAW_POS_UNKNOWN
    return C.RAW_POS_KNOWN


def min_distance_to_locations(
    px: float, py: float, locs_x: np.ndarray, locs_y: np.ndarray
) -> float:
    """float64 Euclidean min distance; inf if no locations."""
    if locs_x.size == 0:
        return float("inf")
    dx = locs_x.astype(np.float64) - float(px)
    dy = locs_y.astype(np.float64) - float(py)
    return float(np.sqrt(dx * dx + dy * dy).min())


def participation_proxy_cache(
    frame_xy: np.ndarray,
    frame_hp: np.ndarray,
    locs_x: np.ndarray,
    locs_y: np.ndarray,
    radius: float,
    c0_bits: int,
) -> dict[str, Any]:
    """CACHE-ONLY N_lower/N_upper/P bounds. Never definitive N0."""
    radius = float(radius)
    n_lower = 0
    n_upper = 0
    near_bits = 0
    lower_bits = 0
    upper_bits = 0
    dists = np.full(10, np.nan, dtype=np.float64)
    pos_codes = np.full(10, C.POS_UNKNOWN, dtype=np.int8)
    hp_codes = np.full(10, C.HP_UNKNOWN, dtype=np.int8)

    for pid in range(1, 11):
        j = pid - 1
        x = float(frame_xy[j, 0])
        y = float(frame_xy[j, 1])
        hp = float(frame_hp[j])
        pos_c = cache_position_state(x, y)
        hp_c = cache_health_state(hp)
        pos_codes[j] = pos_c
        hp_codes[j] = hp_c
        # Cache-only: ambiguous zero health is NOT observed exclusion.
        excluded_by_observed_zero = False  # raw-only path; false here
        in_radius = False
        if pos_c == C.POS_KNOWN_NONZERO:
            d = min_distance_to_locations(x, y, locs_x, locs_y)
            dists[j] = d
            in_radius = bool(d <= radius)
            if in_radius:
                near_bits |= 1 << j
        elif pos_c == C.POS_AMBIGUOUS_ZERO:
            dists[j] = float("nan")
        else:
            dists[j] = float("nan")

        # N_lower: known nonzero position in radius AND cached positive health
        if in_radius and hp_c == C.HP_CACHED_POSITIVE:
            n_lower += 1
            lower_bits |= 1 << j
        # N_upper: not raw-observed-zero-excluded; known-in-radius OR unknown/ambiguous position
        if not excluded_by_observed_zero:
            if in_radius or pos_c in (C.POS_UNKNOWN, C.POS_AMBIGUOUS_ZERO, C.POS_NONFINITE):
                n_upper += 1
                upper_bits |= 1 << j

    p_lower = int(c0_bits) | int(lower_bits)
    p_upper = int(c0_bits) | int(upper_bits)
    # Geometric-only naming (position in radius; ignores health).
    geom_near_bits = int(near_bits)
    geom_credit_only_bits = int(c0_bits) & ~geom_near_bits
    geom_near_only_bits = geom_near_bits & ~int(c0_bits)
    geom_intersection_bits = int(c0_bits) & geom_near_bits
    # Participation decompositions for N_lower / N_upper (not geometric near-only).
    part_credit_only_lower = int(c0_bits) & ~int(lower_bits)
    part_near_only_lower = int(lower_bits) & ~int(c0_bits)
    part_intersection_lower = int(c0_bits) & int(lower_bits)
    part_credit_only_upper = int(c0_bits) & ~int(upper_bits)
    part_near_only_upper = int(upper_bits) & ~int(c0_bits)
    part_intersection_upper = int(c0_bits) & int(upper_bits)
    return {
        "n_lower": int(n_lower),
        "n_upper": int(n_upper),
        "lower_bits": int(lower_bits),
        "upper_bits": int(upper_bits),
        "near_bits": geom_near_bits,
        "geom_near_bits": geom_near_bits,
        "geom_credit_only_bits": int(geom_credit_only_bits),
        "geom_near_only_bits": int(geom_near_only_bits),
        "geom_intersection_bits": int(geom_intersection_bits),
        "p_lower_bits": int(p_lower),
        "p_upper_bits": int(p_upper),
        "p_lower_n": int(bin(p_lower).count("1")),
        "p_upper_n": int(bin(p_upper).count("1")),
        "part_credit_only_lower_bits": int(part_credit_only_lower),
        "part_near_only_lower_bits": int(part_near_only_lower),
        "part_intersection_lower_bits": int(part_intersection_lower),
        "part_credit_only_upper_bits": int(part_credit_only_upper),
        "part_near_only_upper_bits": int(part_near_only_upper),
        "part_intersection_upper_bits": int(part_intersection_upper),
        # Deprecated aliases kept as geometric for backward field names in runners that still map them.
        "credit_only_bits": int(geom_credit_only_bits),
        "near_only_bits": int(geom_near_only_bits),
        "intersection_bits": int(geom_intersection_bits),
        "c0_subset_lower": bool((int(c0_bits) & int(p_lower)) == int(c0_bits)),
        "c0_subset_upper": bool((int(c0_bits) & int(p_upper)) == int(c0_bits)),
        "lower_le_upper": bool(n_lower <= n_upper and bin(p_lower).count("1") <= bin(p_upper).count("1")),
        "dists": dists,
        "pos_codes": pos_codes,
        "hp_codes": hp_codes,
        "mode": "cache_only_proxy",
        "definitive_n0": False,
    }


def radius_monotonicity(p_n_small: int, p_n_large: int) -> bool:
    return int(p_n_small) <= int(p_n_large)


def events_at_or_before(
    event_ts: np.ndarray, cutoff: int, *, preserve_order_idx: np.ndarray | None = None
) -> np.ndarray:
    """Indices of events with timestamp <= cutoff, original ordinal among equals preserved."""
    ts = np.asarray(event_ts, dtype=np.int64)
    if ts.size == 0:
        return np.zeros(0, dtype=np.int64)
    if preserve_order_idx is None:
        # stable: among equal timestamps, increasing original index
        order = np.argsort(ts, kind="mergesort")
    else:
        order = np.asarray(preserve_order_idx, dtype=np.int64)
    # For already-sorted inputs, searchsorted is enough; still filter via order for ties.
    if np.all(np.diff(ts) >= 0):
        end = int(np.searchsorted(ts, int(cutoff), side="right"))
        return np.arange(end, dtype=np.int64)
    # unsorted: take ordered filter
    ordered_ts = ts[order]
    end = int(np.searchsorted(ordered_ts, int(cutoff), side="right"))
    return order[:end]


def count_events_between(event_ts: np.ndarray, lo_exclusive: int, hi_inclusive: int) -> int:
    """Count events with lo < t <= hi (async support between frames)."""
    ts = np.asarray(event_ts, dtype=np.int64)
    if ts.size == 0:
        return 0
    return int(np.sum((ts > int(lo_exclusive)) & (ts <= int(hi_inclusive))))


def count_events_at(event_ts: np.ndarray, t: int) -> int:
    ts = np.asarray(event_ts, dtype=np.int64)
    if ts.size == 0:
        return 0
    return int(np.sum(ts == int(t)))


def unknown_event_count() -> int:
    """Sentinel for unknown integer event counts (never coerce missing support to 0)."""
    return int(C.SENTINEL_MISSING)


def recompute_endpoint_ms(
    L: int,
    h_s: int,
    next_kill_after_L: int,
    next_component_tau: int,
    game_end_ms: int,
    game_end_known: bool,
) -> int:
    """Pure DR endpoint formula (inline; no I/O). Returns sentinel on unbounded."""
    L = int(L)
    horizon = L + 1000 * int(h_s)
    nk = int(next_kill_after_L)
    nt = int(next_component_tau)
    ge = int(game_end_ms) if game_end_known and int(game_end_ms) >= 0 else None
    cand = [horizon]
    if nk >= 0:
        cand.append(nk - 1)
    if nt >= 0:
        if ge is None:
            cand.append(nt - 1)
        elif nt < ge:
            cand.append(nt - 1)
    if ge is not None:
        cand.append(ge - 1)
    if not cand:
        return int(C.SENTINEL_MISSING)
    return int(min(cand))


def gap_band(lag: int) -> str:
    if lag < 0:
        return "negative"
    if lag == 10_000:
        return "exact_10000"
    if 10_000 < lag <= 15_000:
        return "gt10_le15"
    if 15_000 < lag <= 20_000:
        return "gt15_le20"
    if 10_000 <= lag <= 20_000:
        return "in_10_20"
    return "outside_10_20"


def candidate_masks(lags: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(lags, dtype=np.int64)
    ok = d >= 0
    return {
        "exact_10000": ok & (d == 10_000),
        "gt10_le15": ok & (d > 10_000) & (d <= 15_000),
        "gt15_le20": ok & (d > 15_000) & (d <= 20_000),
        "cum_10_15": ok & (d >= 10_000) & (d <= 15_000),
        "cum_10_20": ok & (d >= 10_000) & (d <= 20_000),
    }
