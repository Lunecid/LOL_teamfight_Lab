"""Pure time/endpoint contracts for the anchor-frame audit (no I/O)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import config as C

# Re-export DR pure rules without side effects beyond import.
import importlib.util
from pathlib import Path


def _load_dr():
    p = C.DR_RULES
    spec = importlib.util.spec_from_file_location("engagement_labels_v3_rules_audit", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_DR = None


def dr():
    global _DR
    if _DR is None:
        _DR = _load_dr()
    return _DR


def preceding_frame(minute_ts: np.ndarray, t: int) -> Tuple[Optional[int], Optional[int], str]:
    """F = max minute_ts <= t. Returns (F, age_or_delta_base, reason).

    reason '' if found; else 'no_preceding_frame'.
    Does not interpolate, round, or take nearest future frame.
    """
    ts = np.asarray(minute_ts, dtype=np.int64)
    if ts.size == 0:
        return None, None, "empty_minute_ts"
    if not np.all(np.diff(ts) >= 0):
        return None, None, "nonmonotone_minute_ts"
    if np.any(ts < 0):
        return None, None, "negative_minute_ts"
    i = int(np.searchsorted(ts, int(t), side="right")) - 1
    if i < 0:
        return None, None, "no_preceding_frame"
    F = int(ts[i])
    return F, int(t) - F, ""


def inclusive_alignment(
    minute_ts: np.ndarray, f: int, anchor_offset_ms: int = C.ANCHOR_OFFSET_MS
) -> dict[str, Any]:
    """User-requested: τ=f-offset (unclamped), F_inclusive=max ts<=τ, δ=f-F."""
    f = int(f)
    tau = f - int(anchor_offset_ms)
    out: dict[str, Any] = {
        "f": f,
        "tau": tau,
        "negative_tau": bool(tau < 0),
        "F_inclusive": C.SENTINEL_MISSING,
        "delta": C.SENTINEL_MISSING,
        "exact_10000": False,
        "inclusive_reason": "",
    }
    F, _, reason = preceding_frame(minute_ts, tau)
    if reason:
        out["inclusive_reason"] = reason
        return out
    assert F is not None
    delta = f - F
    out["F_inclusive"] = int(F)
    out["delta"] = int(delta)
    out["exact_10000"] = bool(delta == int(anchor_offset_ms) and F == tau)
    return out


def strict_alignment(
    minute_ts: np.ndarray, f: int, anchor_offset_ms: int = C.ANCHOR_OFFSET_MS
) -> dict[str, Any]:
    """Historical: t_in=τ-1, F_strict=max ts<=t_in, a=t_in-F; eligible iff 0<=a<10000."""
    f = int(f)
    tau = f - int(anchor_offset_ms)
    t_in = tau - 1
    out: dict[str, Any] = {
        "t_in": t_in,
        "F_strict": C.SENTINEL_MISSING,
        "a": C.SENTINEL_MISSING,
        "strict_eligible": False,
        "strict_reason": "",
    }
    F, age, reason = preceding_frame(minute_ts, t_in)
    if reason:
        out["strict_reason"] = reason
        return out
    assert F is not None and age is not None
    out["F_strict"] = int(F)
    out["a"] = int(age)
    out["strict_eligible"] = bool(0 <= age < int(anchor_offset_ms))
    return out


def cumulative_inclusive_eligible(delta: int, width_ms: int, anchor_offset_ms: int = C.ANCHOR_OFFSET_MS) -> bool:
    """Component in cumulative inclusive group [offset, offset+w] when delta in that closed range."""
    if delta < 0:
        return False
    lo = int(anchor_offset_ms)
    hi = lo + int(width_ms)
    return lo <= int(delta) <= hi


def credit_pids(
    killer: int, victim: int, assist_mask: int
) -> list[int]:
    cred: set[int] = set()
    if 1 <= killer <= 10:
        cred.add(int(killer))
    if 1 <= victim <= 10:
        cred.add(int(victim))
    mask = int(assist_mask)
    for pid in range(1, 11):
        if mask & (1 << (pid - 1)):
            cred.add(pid)
    return sorted(cred)


def team_credit_counts(
    killers: np.ndarray,
    victims: np.ndarray,
    assists: np.ndarray,
    member_idx: np.ndarray,
    blue_pids: Sequence[int],
) -> dict[str, Any]:
    blue_set = {int(p) for p in blue_pids if 1 <= int(p) <= 10}
    cred: set[int] = set()
    n_env_killer = 0
    for i in member_idx.tolist():
        k = int(killers[i])
        if k == 0:
            n_env_killer += 1
        for p in credit_pids(k, int(victims[i]), int(assists[i])):
            cred.add(p)
    blue = sorted(p for p in cred if p in blue_set)
    red = sorted(p for p in cred if p not in blue_set and 1 <= p <= 10)
    unknown = sorted(p for p in cred if p < 1 or p > 10)
    # Environmental / team0 does NOT auto-classify as 1v1.
    scale = max(len(blue), len(red))
    return {
        "n_blue": len(blue),
        "n_red": len(red),
        "n_unknown": len(unknown),
        "n_credited": len(cred),
        "n_env_killer": n_env_killer,
        "scale_max_side": scale,
        "blue": blue,
        "red": red,
        "unknown": unknown,
    }


def diameter_xy(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    dx = xx[:, None] - xx[None, :]
    dy = yy[:, None] - yy[None, :]
    return float(np.sqrt(dx * dx + dy * dy).max())


def parse_timestamp_ms(value: Any) -> tuple[Optional[int], str]:
    """Strict millisecond timestamp. Reject bools and fractional floats (no truncation)."""
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


def parse_game_end_events(events: Sequence[Mapping]) -> dict[str, Any]:
    """Record GAME_END from top-level event objects only. Absent stays unknown.

    Distinguishes multiple GAME_END events from duplicate identical timestamps.
    Authoritative known terminal requires >=1 valid timestamp and a single unique time.
    Nested values inside a GAME_END object are ignored for the timestamp field.
    """
    ends: list[int] = []
    malformed = 0
    malformed_reasons: dict[str, int] = {}
    for e in events:
        if not isinstance(e, Mapping) or e.get("type") != "GAME_END":
            continue
        if "timestamp" not in e:
            malformed += 1
            malformed_reasons["missing_timestamp"] = malformed_reasons.get("missing_timestamp", 0) + 1
            continue
        ts, reason = parse_timestamp_ms(e["timestamp"])
        if reason or ts is None:
            malformed += 1
            key = reason or "invalid_timestamp"
            malformed_reasons[key] = malformed_reasons.get(key, 0) + 1
            continue
        ends.append(int(ts))
    ends_sorted = sorted(ends)
    unique = sorted(set(ends_sorted))
    multiple_events = len(ends_sorted) > 1
    duplicate_ts = len(ends_sorted) > len(unique)
    ambiguous_distinct = len(unique) > 1
    # Authoritative known only when all valid GAME_END timestamps agree.
    known = bool(unique) and not ambiguous_distinct
    return {
        "n_game_end": len(ends_sorted),
        "game_end_ms": int(unique[0]) if known else C.SENTINEL_MISSING,
        "game_end_all": ends_sorted,
        "game_end_unique": unique,
        "game_end_multiple_events": bool(multiple_events),
        "game_end_duplicate_ts": bool(duplicate_ts),
        "game_end_ambiguous_distinct": bool(ambiguous_distinct),
        "game_end_malformed": int(malformed),
        "game_end_malformed_reasons": malformed_reasons,
        "game_end_known": bool(known),
    }


def parse_game_end_events_reference(events: Sequence[Mapping]) -> dict[str, Any]:
    """Independent full-array reference path (same contract, separate implementation)."""
    ends: list[int] = []
    malformed = 0
    for e in events:
        if not isinstance(e, Mapping):
            continue
        if e.get("type") != "GAME_END":
            continue
        if "timestamp" not in e:
            malformed += 1
            continue
        raw = e["timestamp"]
        if isinstance(raw, bool):
            malformed += 1
            continue
        if isinstance(raw, float):
            if (not math.isfinite(raw)) or raw != float(int(raw)) or raw < 0:
                malformed += 1
                continue
            ends.append(int(raw))
            continue
        if isinstance(raw, int):
            if raw < 0:
                malformed += 1
                continue
            ends.append(int(raw))
            continue
        malformed += 1
    ends_sorted = sorted(ends)
    unique = sorted(set(ends_sorted))
    known = bool(unique) and len(unique) == 1
    return {
        "n_game_end": len(ends_sorted),
        "game_end_ms": int(unique[0]) if known else C.SENTINEL_MISSING,
        "game_end_all": ends_sorted,
        "game_end_unique": unique,
        "game_end_multiple_events": len(ends_sorted) > 1,
        "game_end_duplicate_ts": len(ends_sorted) > len(unique),
        "game_end_ambiguous_distinct": len(unique) > 1,
        "game_end_malformed": int(malformed),
        "game_end_known": bool(known),
    }


def endpoint_bundle(
    L: int,
    kill_times_sorted: Sequence[int],
    next_component_tau: Optional[int],
    game_end_ms: Optional[int],
    horizons_s: Sequence[int] = C.HORIZONS_S,
) -> dict[str, Any]:
    """Pure e_h candidates using DR rules. Missing GAME_END -> INF (not last_frame)."""
    R = dr()
    L = int(L)
    kts = [int(t) for t in kill_times_sorted]
    nk = R.next_kill_after(kts, L)
    ge = R.INF if game_end_ms is None or int(game_end_ms) < 0 else int(game_end_ms)
    if next_component_tau is None or int(next_component_tau) < 0:
        ns_eff = R.INF
    else:
        ns_eff = R.effective_next_start(int(next_component_tau), ge if ge != R.INF else int(next_component_tau) + 1)
        # When GAME_END unknown, still treat next_component_tau as a candidate (no sentinel collapse to INF via ge).
        if ge == R.INF:
            ns_eff = int(next_component_tau)
    out: dict[str, Any] = {}
    for h in horizons_s:
        e, reasons, cand = R.endpoint_rule(L, int(h), nk, ns_eff, ge)
        out[f"e_h{h}"] = int(e)
        out[f"e_h{h}_reasons"] = "|".join(reasons)
        out[f"e_h{h}_cand_horizon"] = int(cand["horizon"]) if cand["horizon"] != R.INF else C.SENTINEL_INF
        out[f"e_h{h}_cand_next_kill"] = (
            int(cand["next_kill"]) if cand["next_kill"] != R.INF else C.SENTINEL_INF
        )
        out[f"e_h{h}_cand_next_eng"] = (
            int(cand["next_engagement_start"])
            if cand["next_engagement_start"] != R.INF
            else C.SENTINEL_INF
        )
        out[f"e_h{h}_cand_game_end"] = (
            int(cand["game_end"]) if cand["game_end"] != R.INF else C.SENTINEL_INF
        )
    out["next_kill_after_L"] = int(nk) if nk != R.INF else C.SENTINEL_MISSING
    out["next_component_tau"] = (
        int(next_component_tau) if next_component_tau is not None else C.SENTINEL_MISSING
    )
    out["overlap_next_tau_le_L"] = bool(
        next_component_tau is not None and int(next_component_tau) <= L
    )
    out["missing_terminal_game_end"] = bool(ge == R.INF)
    return out


def endpoint_validity_flags(
    e: int, L: int, tau: int, support_start: int, last_frame: int, overlap_next_tau_le_L: bool
) -> dict[str, Any]:
    """DR validity + overlap flag. Invalid endpoints must not become negative labels."""
    R = dr()
    q_pre = int(tau) - 1
    flags = R.endpoint_validity(int(e), int(L), q_pre, int(support_start), int(last_frame))
    inval = [k for k, ok in flags.items() if not ok]
    if overlap_next_tau_le_L:
        inval.insert(0, "same_match_overlap_next_start_le_L")
    return {
        "q_pre": q_pre,
        "valid": len(inval) == 0,
        "invalid_reasons": "|".join(inval),
        "flags": flags,
        "followup_ms": int(e) - int(L) if len(inval) == 0 else C.SENTINEL_MISSING,
    }


def foreign_distance_class(
    fx: float, fy: float, member_x: np.ndarray, member_y: np.ndarray, radius: float = C.FOREIGN_NEAR_RADIUS
) -> str:
    """Diagnostic only: foreign kill near if min Euclidean distance to any member <= radius."""
    if member_x.size == 0:
        return "unknown"
    d = np.sqrt((member_x.astype(np.float64) - fx) ** 2 + (member_y.astype(np.float64) - fy) ** 2)
    return "near" if float(d.min()) <= float(radius) else "distant"
