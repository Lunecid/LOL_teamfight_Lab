"""Per-match component audit using completed labels + original cache frames/events."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import config as C
from . import contracts as CT
from .io_source import anonymous_match_key, load_events_game_end, load_minute_ts


def _first_kill_band(f: int) -> str:
    # Diagnostic bands on first-kill match time (minutes)
    m = f / 60_000.0
    edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 90, 1e9]
    for i in range(len(edges) - 1):
        if edges[i] <= m < edges[i + 1]:
            return f"[{edges[i]:g},{edges[i+1]:g})"
    return "other"


def _minute_ts_integrity(minute_ts: np.ndarray | None, ts_reason: str) -> dict[str, Any]:
    """Report frame integrity without silently normalizing invalid/nonmonotone timestamps."""
    out = {
        "minute_ts_reason": ts_reason,
        "n_frames": 0 if minute_ts is None else int(minute_ts.size),
        "has_negative_ts": False,
        "nonmonotone": False,
        "integrity_ok": False,
    }
    if minute_ts is None:
        return out
    ts = np.asarray(minute_ts, dtype=np.int64)
    out["n_frames"] = int(ts.size)
    if ts.size == 0:
        out["minute_ts_reason"] = ts_reason or "empty_minute_ts"
        return out
    out["has_negative_ts"] = bool(np.any(ts < 0))
    out["nonmonotone"] = bool(ts.size >= 2 and np.any(np.diff(ts) < 0))
    out["integrity_ok"] = (not out["has_negative_ts"]) and (not out["nonmonotone"]) and not bool(ts_reason)
    return out


def process_match(
    match_local: int,
    match_id: str,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    killer: np.ndarray,
    victim: np.ndarray,
    assist: np.ndarray,
    labels: np.ndarray,
    blue_pids: np.ndarray,
    duration_ms_meta: int,
) -> dict[str, Any]:
    """Audit one match. labels/t/... are match-local slices (aligned)."""
    anon = anonymous_match_key(match_local)
    n = int(t.shape[0])
    minute_ts, ts_reason = load_minute_ts(match_id)
    ge_info, ge_reason = load_events_game_end(match_id)
    frame_info = _minute_ts_integrity(minute_ts, ts_reason)

    # Use frames for F/last_frame only when integrity is OK; otherwise keep explicit unavailable.
    usable_frames = minute_ts if frame_info["integrity_ok"] else None
    last_frame = int(usable_frames[-1]) if usable_frames is not None and usable_frames.size else C.SENTINEL_MISSING
    first_frame = int(usable_frames[0]) if usable_frames is not None and usable_frames.size else C.SENTINEL_MISSING
    support_start = max(0, first_frame) if first_frame >= 0 else 0
    # Raw last observed frame (even if integrity failed) for independent GE comparison only.
    raw_last_frame = int(minute_ts[-1]) if minute_ts is not None and minute_ts.size else C.SENTINEL_MISSING

    ge_ms = int(ge_info["game_end_ms"])
    ge_known = bool(ge_info["game_end_known"])
    ge_minus_last = (ge_ms - raw_last_frame) if ge_known and raw_last_frame >= 0 else C.SENTINEL_MISSING
    meta_duration = int(duration_ms_meta)
    terminal = {
        "match_local": int(match_local),
        "anon_match_key": anon,
        "n_kills": n,
        "last_frame_ms": last_frame,
        "raw_last_frame_ms": raw_last_frame,
        "meta_duration_ms": meta_duration,
        "game_end_ms": ge_ms if ge_known else C.SENTINEL_MISSING,
        "game_end_known": ge_known,
        "n_game_end": int(ge_info["n_game_end"]),
        "game_end_multiple_events": bool(ge_info.get("game_end_multiple_events", False)),
        "game_end_duplicate_ts": bool(ge_info["game_end_duplicate_ts"]),
        "game_end_ambiguous_distinct": bool(ge_info.get("game_end_ambiguous_distinct", False)),
        "game_end_malformed": int(ge_info["game_end_malformed"]),
        "ge_minus_last_frame": ge_minus_last,
        "ge_minus_last_frame_negative": bool(ge_minus_last != C.SENTINEL_MISSING and ge_minus_last < 0),
        "last_frame_eq_meta_duration": bool(last_frame == meta_duration) if last_frame >= 0 else False,
        "meta_duration_disagrees_last_frame": bool(
            raw_last_frame >= 0 and meta_duration >= 0 and raw_last_frame != meta_duration
        ),
        "meta_duration_disagrees_game_end": bool(
            ge_known and meta_duration >= 0 and ge_ms != meta_duration
        ),
        "minute_ts_reason": frame_info["minute_ts_reason"],
        "frame_integrity_ok": bool(frame_info["integrity_ok"]),
        "frame_has_negative_ts": bool(frame_info["has_negative_ts"]),
        "frame_nonmonotone": bool(frame_info["nonmonotone"]),
        "n_frames": int(frame_info["n_frames"]),
        "events_reason": ge_reason,
        "terminal_crosscheck_match": bool(ge_info.get("terminal_crosscheck_match", True)),
        "terminal_support_authoritative": bool(ge_known and not ge_info.get("game_end_ambiguous_distinct", False)),
        "cache_zero_not_raw_absence": True,
        # Candidate earliest GAME_END for diagnosis even when not authoritative
        "game_end_candidate_ms": (
            int(ge_info["game_end_all"][0]) if ge_info.get("game_end_all") else C.SENTINEL_MISSING
        ),
        "game_end_all": list(ge_info.get("game_end_all") or []),
    }

    components: list[dict[str, Any]] = []
    if n == 0:
        return {"terminal": terminal, "components": components, "n_components": 0}

    if labels.shape[0] != n:
        raise RuntimeError(f"{anon}: label/event length mismatch {labels.shape[0]} vs {n}")

    pos = labels >= 0
    unique_labs = np.unique(labels[pos]) if np.any(pos) else np.zeros(0, dtype=np.int64)

    comp_meta: list[tuple[int, int, int, np.ndarray]] = []  # lab, f, L, memb
    for lab in unique_labs:
        memb = np.flatnonzero(labels == lab)
        tm = t[memb].astype(np.int64)
        f = int(tm.min())
        L = int(tm.max())
        comp_meta.append((int(lab), f, L, memb))
    comp_meta.sort(key=lambda r: (r[1], r[2], r[0]))  # by f, L, lab

    next_tau_by_lab: dict[int, Optional[int]] = {}
    for i, (lab, f, L, memb) in enumerate(comp_meta):
        nxt = None
        for j in range(i + 1, len(comp_meta)):
            f2 = comp_meta[j][1]
            if f2 > f:
                nxt = f2 - C.ANCHOR_OFFSET_MS
                break
        next_tau_by_lab[lab] = nxt

    kill_times_sorted = sorted(int(x) for x in t.tolist())
    terminal_ok = bool(terminal["terminal_support_authoritative"])

    for lab, f, L, memb in comp_meta:
        span = L - f
        xx = x[memb].astype(np.float64)
        yy = y[memb].astype(np.float64)
        diam = CT.diameter_xy(xx, yy)
        tau = f - C.ANCHOR_OFFSET_MS

        t0_mask = t[memb] == f
        memb0 = memb[t0_mask]
        c0 = CT.team_credit_counts(killer, victim, assist, memb0, blue_pids)
        call = CT.team_credit_counts(killer, victim, assist, memb, blue_pids)

        if usable_frames is None:
            reason = ts_reason or (
                "nonmonotone_minute_ts"
                if frame_info["nonmonotone"]
                else ("negative_minute_ts" if frame_info["has_negative_ts"] else "missing_minute_ts")
            )
            incl = {
                "f": f,
                "tau": tau,
                "negative_tau": tau < 0,
                "F_inclusive": C.SENTINEL_MISSING,
                "delta": C.SENTINEL_MISSING,
                "exact_10000": False,
                "inclusive_reason": reason,
            }
            strict = {
                "t_in": tau - 1,
                "F_strict": C.SENTINEL_MISSING,
                "a": C.SENTINEL_MISSING,
                "strict_eligible": False,
                "strict_reason": reason,
            }
        else:
            incl = CT.inclusive_alignment(usable_frames, f)
            strict = CT.strict_alignment(usable_frames, f)

        member_set = set(int(i) for i in memb.tolist())
        foreign_in_window = 0
        foreign_near = 0
        foreign_distant = 0
        for ii in range(n):
            if ii in member_set:
                continue
            ti = int(t[ii])
            if tau <= ti <= L:
                foreign_in_window += 1
                cls = CT.foreign_distance_class(float(x[ii]), float(y[ii]), xx, yy)
                if cls == "near":
                    foreign_near += 1
                elif cls == "distant":
                    foreign_distant += 1
        other_anchors_in_window = 0
        overlap_span = 0
        for lab2, f2, L2, _memb2 in comp_meta:
            if lab2 == lab:
                continue
            tau2 = f2 - C.ANCHOR_OFFSET_MS
            if tau <= tau2 <= L:
                other_anchors_in_window += 1
            if not (L2 < f or L < f2):
                overlap_span += 1

        same_ms_first = int(memb0.shape[0])
        nxt_tau = next_tau_by_lab[lab]
        ge_for_ep = ge_ms if ge_known else None
        ep = CT.endpoint_bundle(L, kill_times_sorted, nxt_tau, ge_for_ep)
        rec: dict[str, Any] = {
            "anon_match_key": anon,
            "match_local": int(match_local),
            "component_id": int(lab),
            "member_key": f"{anon}:c{int(lab)}",
            "n_members": int(memb.shape[0]),
            "f": int(f),
            "L": int(L),
            "tau": int(tau),
            "span_ms": int(span),
            "diameter": float(diam),
            "singleton": bool(memb.shape[0] == 1),
            "same_ms_first_n": same_ms_first,
            "first_kill_band": _first_kill_band(f),
            "C0_n_blue": c0["n_blue"],
            "C0_n_red": c0["n_red"],
            "C0_n_unknown": c0["n_unknown"],
            "C0_n_credited": c0["n_credited"],
            "C0_n_env_killer": c0["n_env_killer"],
            "C0_scale_max_side": c0["scale_max_side"],
            "Call_n_blue": call["n_blue"],
            "Call_n_red": call["n_red"],
            "Call_n_unknown": call["n_unknown"],
            "Call_n_credited": call["n_credited"],
            "Call_n_env_killer": call["n_env_killer"],
            "Call_scale_max_side": call["scale_max_side"],
            "foreign_kills_in_tau_L": foreign_in_window,
            "foreign_near": foreign_near,
            "foreign_distant": foreign_distant,
            "other_anchors_in_tau_L": other_anchors_in_window,
            "n_overlapping_spans": overlap_span,
            "F_inclusive": incl["F_inclusive"],
            "delta": incl["delta"],
            "exact_10000": bool(incl["exact_10000"]),
            "inclusive_reason": incl["inclusive_reason"],
            "negative_tau": bool(incl["negative_tau"]),
            "t_in": strict["t_in"],
            "F_strict": strict["F_strict"],
            "a_strict": strict["a"],
            "strict_eligible": bool(strict["strict_eligible"]),
            "strict_reason": strict["strict_reason"],
            "last_frame_ms": last_frame,
            "game_end_known": ge_known,
            "game_end_ms": ge_ms if ge_known else C.SENTINEL_MISSING,
            "terminal_support_authoritative": terminal_ok,
        }
        for h in C.HORIZONS_S:
            e = int(ep[f"e_h{h}"])
            val = CT.endpoint_validity_flags(
                e,
                L,
                tau,
                support_start,
                last_frame if last_frame >= 0 else -10**15,
                bool(ep["overlap_next_tau_le_L"]),
            )
            # Retain candidate endpoint even when terminal support is missing/invalid.
            rec[f"e_h{h}_candidate"] = e
            if last_frame < 0:
                val["valid"] = False
                val["invalid_reasons"] = (
                    (val["invalid_reasons"] + "|" if val["invalid_reasons"] else "")
                    + "missing_last_frame_support"
                )
                val["followup_ms"] = C.SENTINEL_MISSING
            if not terminal_ok:
                val["valid"] = False
                extra = "missing_or_invalid_terminal_support"
                val["invalid_reasons"] = (
                    (val["invalid_reasons"] + "|" if val["invalid_reasons"] else "") + extra
                )
                val["followup_ms"] = C.SENTINEL_MISSING
            rec[f"e_h{h}"] = e
            rec[f"e_h{h}_reasons"] = ep[f"e_h{h}_reasons"]
            rec[f"valid_h{h}"] = int(val["valid"])
            rec[f"invalid_h{h}"] = val["invalid_reasons"]
            rec[f"followup_ms_h{h}"] = val["followup_ms"]
            if val["valid"]:
                rec[f"clip_h{h}"] = ep[f"e_h{h}_reasons"]
            else:
                rec[f"clip_h{h}"] = val["invalid_reasons"]
        rec["overlap_next_tau_le_L"] = bool(ep["overlap_next_tau_le_L"])
        rec["next_component_tau"] = ep["next_component_tau"]
        rec["next_kill_after_L"] = ep["next_kill_after_L"]
        rec["missing_terminal_game_end"] = bool(ep["missing_terminal_game_end"])
        components.append(rec)

    return {"terminal": terminal, "components": components, "n_components": len(components)}
