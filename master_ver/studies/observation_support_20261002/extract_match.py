"""Per-match observation-support extraction (CACHE-ONLY when raw absent)."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from . import config as C
from . import contracts as CT
from .io_source import (
    anonymous_match_key,
    load_cache_meta_team_map,
    load_cache_pack,
    load_event_timestamps,
    verify_team_map_vs_blue_pids,
)


def _channel_at(node_minute: np.ndarray, frame_idx: int, feat_idx: int) -> np.ndarray:
    return np.asarray(node_minute[frame_idx, :, feat_idx], dtype=np.float64)


def _summarize_types(types: list[str], indices: np.ndarray) -> dict[str, int]:
    c: Counter[str] = Counter()
    for i in indices.tolist():
        c[types[int(i)] if types[int(i)] else "EMPTY_TYPE"] += 1
    return dict(c)


def _empty_cut(f: int, frame_reason: str) -> dict[str, Any]:
    return {
        "f": f,
        "tau": f - C.ANCHOR_OFFSET_MS,
        "t_in": f - C.ANCHOR_OFFSET_MS - 1,
        "negative_tau": (f - C.ANCHOR_OFFSET_MS) < 0,
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
        "tau_reason": frame_reason or "missing_frames",
        "tm1_reason": frame_reason or "missing_frames",
        "same_F": False,
        "frame_differs": False,
    }


def _unsupported_participation_bounds(c0_bits: int) -> dict[str, Any]:
    """Absent input-frame/cache support: preserve C0; N_lower empty; N_upper all ten."""
    c0 = int(c0_bits) & ((1 << 10) - 1)
    upper_bits = (1 << 10) - 1  # all ten players unknown-position-eligible
    p_lower = c0
    p_upper = upper_bits  # C0 ∪ N_upper = all ten
    return {
        "n_lower": 0,
        "n_upper": 10,
        "lower_bits": 0,
        "upper_bits": int(upper_bits),
        "near_bits": 0,
        "geom_near_bits": 0,
        "geom_credit_only_bits": int(c0),
        "geom_near_only_bits": 0,
        "geom_intersection_bits": 0,
        "p_lower_bits": int(p_lower),
        "p_upper_bits": int(p_upper),
        "p_lower_n": int(bin(p_lower).count("1")),
        "p_upper_n": int(bin(p_upper).count("1")),
        "part_credit_only_lower_bits": int(c0),
        "part_near_only_lower_bits": 0,
        "part_intersection_lower_bits": 0,
        "part_credit_only_upper_bits": 0,
        "part_near_only_upper_bits": int(upper_bits) & ~int(c0),
        "part_intersection_upper_bits": int(c0),
        "credit_only_bits": int(c0),
        "near_only_bits": 0,
        "intersection_bits": 0,
        "lower_le_upper": True,
        "c0_subset_lower": True,
        "c0_subset_upper": True,
        "dists": np.full(10, np.nan, dtype=np.float64),
        "pos_codes": np.full(10, C.POS_UNKNOWN, dtype=np.int8),
        "hp_codes": np.full(10, C.HP_UNKNOWN, dtype=np.int8),
        "mode": "unsupported_input_frame",
        "definitive_n0": False,
    }


def _fill_participation(
    rec: dict[str, Any],
    tag: str,
    part: dict[str, Any] | None,
    c0_bits: int,
) -> None:
    if part is None:
        # Explicit unsupported: coherent C0-preserving bounds; conditional metrics exclude via support_ok=False.
        part = _unsupported_participation_bounds(c0_bits)
        rec[f"n_lower_{tag}"] = int(part["n_lower"])
        rec[f"n_upper_{tag}"] = int(part["n_upper"])
        rec[f"p_lower_n_{tag}"] = int(part["p_lower_n"])
        rec[f"p_upper_n_{tag}"] = int(part["p_upper_n"])
        for key in (
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
        ):
            rec[f"{key}_{tag}"] = int(part[key])
        rec[f"lower_le_upper_{tag}"] = True
        rec[f"c0_subset_lower_{tag}"] = True
        rec[f"c0_subset_upper_{tag}"] = True
        rec[f"support_ok_{tag}"] = False
        rec[f"dist_min_{tag}"] = float("nan")
        rec[f"dist_p50_{tag}"] = float("nan")
        rec[f"dist_p90_{tag}"] = float("nan")
        rec[f"_dists_{tag}"] = part["dists"]
        rec[f"_pos_codes_{tag}"] = part["pos_codes"]
        rec[f"_hp_codes_{tag}"] = part["hp_codes"]
        return
    rec[f"n_lower_{tag}"] = int(part["n_lower"])
    rec[f"n_upper_{tag}"] = int(part["n_upper"])
    rec[f"p_lower_n_{tag}"] = int(part["p_lower_n"])
    rec[f"p_upper_n_{tag}"] = int(part["p_upper_n"])
    for key in (
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
    ):
        rec[f"{key}_{tag}"] = int(part[key])
    rec[f"lower_le_upper_{tag}"] = bool(part["lower_le_upper"])
    rec[f"c0_subset_lower_{tag}"] = bool(part["c0_subset_lower"])
    rec[f"c0_subset_upper_{tag}"] = bool(part["c0_subset_upper"])
    rec[f"support_ok_{tag}"] = True
    d = part["dists"]
    known = np.isfinite(d)
    if np.any(known):
        rec[f"dist_min_{tag}"] = float(np.min(d[known]))
        rec[f"dist_p50_{tag}"] = float(np.median(d[known]))
        rec[f"dist_p90_{tag}"] = float(np.quantile(d[known], 0.9))
    else:
        rec[f"dist_min_{tag}"] = float("nan")
        rec[f"dist_p50_{tag}"] = float("nan")
        rec[f"dist_p90_{tag}"] = float("nan")
    rec[f"_dists_{tag}"] = d.astype(np.float64)
    rec[f"_pos_codes_{tag}"] = part["pos_codes"].astype(np.int8)
    rec[f"_hp_codes_{tag}"] = part["hp_codes"].astype(np.int8)


def _post_for_cutoff(
    rec: dict[str, Any],
    *,
    cut_name: str,
    h: int,
    e: int,
    valid: int,
    L: int,
    F_pre: int,
    i_pre: int,
    cutoff_ms: int,
    pack: dict[str, Any] | None,
    usable_frames: np.ndarray | None,
    event_ts: np.ndarray | None,
    events_ok: bool,
    frame_reason: str,
) -> None:
    tag = f"{cut_name}_h{h}"
    if pack is None or usable_frames is None or valid != 1 or e < 0:
        rec[f"F_post_{tag}"] = C.SENTINEL_MISSING
        rec[f"F_post_idx_{tag}"] = C.SENTINEL_MISSING
        rec[f"age_post_{tag}"] = C.SENTINEL_MISSING
        rec[f"same_as_Fpre_{tag}"] = False
        rec[f"frame_idx_gap_{tag}"] = C.SENTINEL_MISSING
        rec[f"timestamp_gap_{tag}"] = C.SENTINEL_MISSING
        rec[f"F_post_gt_L_{tag}"] = False
        rec[f"stale_at_or_before_L_{tag}"] = False
        rec[f"async_events_Fpre_to_Fpost_{tag}"] = C.SENTINEL_MISSING
        rec[f"async_events_cutoff_to_endpoint_{tag}"] = C.SENTINEL_MISSING
        rec[f"events_support_ok_{tag}"] = False
        for f in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean"):
            rec[f"{f}_{tag}"] = float("nan")
        rec[f"post_support_reason_{tag}"] = (
            "invalid_endpoint" if valid != 1 else (frame_reason or "missing_frames")
        )
        return

    F_post, i_post, age_post, reason = CT.preceding_frame(usable_frames, e)
    if reason or F_post is None or i_post is None:
        rec[f"F_post_{tag}"] = C.SENTINEL_MISSING
        rec[f"F_post_idx_{tag}"] = C.SENTINEL_MISSING
        rec[f"age_post_{tag}"] = C.SENTINEL_MISSING
        rec[f"same_as_Fpre_{tag}"] = False
        rec[f"frame_idx_gap_{tag}"] = C.SENTINEL_MISSING
        rec[f"timestamp_gap_{tag}"] = C.SENTINEL_MISSING
        rec[f"F_post_gt_L_{tag}"] = False
        rec[f"stale_at_or_before_L_{tag}"] = False
        rec[f"async_events_Fpre_to_Fpost_{tag}"] = C.SENTINEL_MISSING
        rec[f"async_events_cutoff_to_endpoint_{tag}"] = C.SENTINEL_MISSING
        rec[f"events_support_ok_{tag}"] = False
        for f in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean"):
            rec[f"{f}_{tag}"] = float("nan")
        rec[f"post_support_reason_{tag}"] = reason or "no_post_frame"
        return

    rec[f"F_post_{tag}"] = int(F_post)
    rec[f"F_post_idx_{tag}"] = int(i_post)
    rec[f"age_post_{tag}"] = int(age_post) if age_post is not None else C.SENTINEL_MISSING
    same = bool(F_pre >= 0 and F_post == F_pre)
    rec[f"same_as_Fpre_{tag}"] = same
    if F_pre >= 0 and i_pre >= 0:
        rec[f"frame_idx_gap_{tag}"] = int(i_post - i_pre)
        rec[f"timestamp_gap_{tag}"] = int(F_post - F_pre)
    else:
        rec[f"frame_idx_gap_{tag}"] = C.SENTINEL_MISSING
        rec[f"timestamp_gap_{tag}"] = C.SENTINEL_MISSING
    rec[f"F_post_gt_L_{tag}"] = bool(F_post > L)
    rec[f"stale_at_or_before_L_{tag}"] = bool(F_post <= L)
    rec[f"post_support_reason_{tag}"] = ""

    if events_ok and event_ts is not None:
        rec[f"events_support_ok_{tag}"] = True
        if F_pre >= 0:
            # Precise frame-to-frame interval (F_pre, F_post]; same minute frame may still have async events.
            rec[f"async_events_Fpre_to_Fpost_{tag}"] = CT.count_events_between(event_ts, F_pre, F_post)
        else:
            rec[f"async_events_Fpre_to_Fpost_{tag}"] = C.SENTINEL_MISSING
        # Separate: cutoff < event_timestamp <= endpoint
        rec[f"async_events_cutoff_to_endpoint_{tag}"] = CT.count_events_between(
            event_ts, int(cutoff_ms), int(e)
        )
    else:
        rec[f"events_support_ok_{tag}"] = False
        rec[f"async_events_Fpre_to_Fpost_{tag}"] = C.SENTINEL_MISSING
        rec[f"async_events_cutoff_to_endpoint_{tag}"] = C.SENTINEL_MISSING

    if i_pre >= 0 and i_post >= 0:
        g0 = pack["gold_team_minute"][i_pre]
        g1 = pack["gold_team_minute"][i_post]
        rec[f"d_gold_blue_{tag}"] = float(g1[0] - g0[0])
        rec[f"d_gold_red_{tag}"] = float(g1[1] - g0[1])
        lv0 = _channel_at(pack["node_minute"], i_pre, C.IDX_LEVEL_NORM)
        lv1 = _channel_at(pack["node_minute"], i_post, C.IDX_LEVEL_NORM)
        xp0 = _channel_at(pack["node_minute"], i_pre, C.IDX_XP_NORM)
        xp1 = _channel_at(pack["node_minute"], i_post, C.IDX_XP_NORM)
        hp0 = _channel_at(pack["node_minute"], i_pre, C.IDX_HP_PCT)
        hp1 = _channel_at(pack["node_minute"], i_post, C.IDX_HP_PCT)
        rec[f"d_level_mean_{tag}"] = float(np.mean(lv1 - lv0))
        rec[f"d_xp_mean_{tag}"] = float(np.mean(xp1 - xp0))
        rec[f"d_hp_mean_{tag}"] = float(np.mean(hp1 - hp0))
        # Compact pre/post channel snapshot for independent checks (means + gold).
        rec[f"_pre_gold_{tag}"] = g0.astype(np.float64)
        rec[f"_post_gold_{tag}"] = g1.astype(np.float64)
        rec[f"_pre_level_mean_{tag}"] = float(np.mean(lv0))
        rec[f"_post_level_mean_{tag}"] = float(np.mean(lv1))
        rec[f"_pre_xp_mean_{tag}"] = float(np.mean(xp0))
        rec[f"_post_xp_mean_{tag}"] = float(np.mean(xp1))
        rec[f"_pre_hp_mean_{tag}"] = float(np.mean(hp0))
        rec[f"_post_hp_mean_{tag}"] = float(np.mean(hp1))
    else:
        for f in ("d_gold_blue", "d_gold_red", "d_level_mean", "d_xp_mean", "d_hp_mean"):
            rec[f"{f}_{tag}"] = float("nan")


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
    baseline_rows: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Extract cutoff / participation / post-frame diagnostics for one match."""
    anon = anonymous_match_key(match_local)
    n = int(t.shape[0])
    pack, pack_reason = load_cache_pack(match_id)
    team_map, tm_reason = load_cache_meta_team_map(match_id)
    event_ts, event_types, ev_reason = load_event_timestamps(match_id)
    events_ok = event_ts is not None and event_types is not None and not ev_reason

    team_map_ok = False
    if team_map is not None:
        team_map_ok = verify_team_map_vs_blue_pids(team_map, [int(p) for p in blue_pids.tolist()])
        if not team_map_ok:
            tm_reason = tm_reason or "team_map_blue_pids_mismatch"
    else:
        team_map = {}

    usable_frames = None
    frame_reason = pack_reason
    if pack is not None:
        ts = pack["minute_ts"]
        if ts.size == 0:
            frame_reason = "empty_minute_ts"
        elif np.any(ts < 0):
            frame_reason = "negative_minute_ts"
        elif ts.size >= 2 and np.any(np.diff(ts) < 0):
            frame_reason = "nonmonotone_minute_ts"
        else:
            usable_frames = ts
            frame_reason = ""

    match_meta = {
        "match_local": int(match_local),
        "anon_match_key": anon,
        "n_kills": n,
        "cache_pack_reason": pack_reason,
        "frame_reason": frame_reason,
        "team_map_reason": tm_reason,
        "team_map_ok": bool(team_map_ok and not tm_reason),
        "events_reason": ev_reason,
        "events_support_ok": bool(events_ok),
        "n_frames": int(usable_frames.size) if usable_frames is not None else 0,
        "n_events": int(event_ts.size) if event_ts is not None else 0,
        "raw_presence_verified": False,
        "diagnostic_mode": "CACHE_ONLY",
        "npz_keys": list(pack["npz_keys"]) if pack is not None else [],
        "team_map_provenance": "cache_meta_team_map" if team_map_ok else (tm_reason or "unavailable"),
    }

    components: list[dict[str, Any]] = []
    if n == 0:
        return {"match": match_meta, "components": components, "n_components": 0, "baseline_reconcile": {}}

    if labels.shape[0] != n:
        raise RuntimeError(f"{anon}: label/event length mismatch")

    pos = labels >= 0
    unique_labs = np.unique(labels[pos]) if np.any(pos) else np.zeros(0, dtype=np.int64)
    comp_meta: list[tuple[int, int, int, np.ndarray]] = []
    for lab in unique_labs:
        memb = np.flatnonzero(labels == lab)
        tm = t[memb].astype(np.int64)
        f = int(tm.min())
        L = int(tm.max())
        comp_meta.append((int(lab), f, L, memb))
    comp_meta.sort(key=lambda r: (r[1], r[2], r[0]))

    next_tau_by_lab: dict[int, int] = {}
    for i, (lab, f, L, memb) in enumerate(comp_meta):
        nxt = C.SENTINEL_MISSING
        for j in range(i + 1, len(comp_meta)):
            f2 = comp_meta[j][1]
            if f2 > f:
                nxt = f2 - C.ANCHOR_OFFSET_MS
                break
        next_tau_by_lab[lab] = int(nxt)

    baseline_by_lab: dict[int, int] = {}
    if baseline_rows is not None and baseline_rows.get("component_id") is not None:
        for i, lab in enumerate(baseline_rows["component_id"].tolist()):
            baseline_by_lab[int(lab)] = i

    reconcile = {
        "n_components": len(comp_meta),
        "n_baseline_rows": int(baseline_rows["component_id"].shape[0]) if baseline_rows else 0,
        "cutoff_mismatches": 0,
        "endpoint_mismatches": 0,
        "c0_count_mismatches": 0,
    }

    for lab, f, L, memb in comp_meta:
        t0_mask = t[memb] == f
        memb0 = memb[t0_mask]
        locs_x = x[memb0].astype(np.float64)
        locs_y = y[memb0].astype(np.float64)
        c0_bits, c0_env = CT.union_credit_bitset(killer, victim, assist, memb0)
        call_bits, call_env = CT.union_credit_bitset(killer, victim, assist, memb)
        c0_teams = (
            CT.team_counts_from_bitset(c0_bits, team_map)
            if team_map_ok
            else {
                "n_blue": -1,
                "n_red": -1,
                "n_unknown_team": -1,
                "n_pids": bin(c0_bits).count("1"),
                "scale_max_side": -1,
                "blue": [],
                "red": [],
                "unknown_team": [],
            }
        )
        call_teams = (
            CT.team_counts_from_bitset(call_bits, team_map)
            if team_map_ok
            else {
                "n_blue": -1,
                "n_red": -1,
                "n_unknown_team": -1,
                "n_pids": bin(call_bits).count("1"),
                "scale_max_side": -1,
                "blue": [],
                "red": [],
                "unknown_team": [],
            }
        )

        cut = _empty_cut(f, frame_reason) if usable_frames is None else CT.cutoff_diagnostics(usable_frames, f)

        # Event boundary counts — unknown when events unsupported (never coerce to observed 0).
        if events_ok and event_ts is not None and event_types is not None:
            at_idx = np.flatnonzero(event_ts == int(cut["tau"]))
            ev_at_tau = int(at_idx.size)
            ev_types_at_tau = _summarize_types(event_types, at_idx)
            ev_between_Ftau_tau = (
                CT.count_events_between(event_ts, int(cut["F_tau"]), int(cut["tau"]))
                if cut["F_tau"] >= 0
                else C.SENTINEL_MISSING
            )
            ev_between_Ftm1_tin = (
                CT.count_events_between(event_ts, int(cut["F_tm1"]), int(cut["t_in"]))
                if cut["F_tm1"] >= 0
                else C.SENTINEL_MISSING
            )
            ev_support_ok = True
            ev_support_reason = ""
        else:
            ev_at_tau = C.SENTINEL_MISSING
            ev_types_at_tau = {}
            ev_between_Ftau_tau = C.SENTINEL_MISSING
            ev_between_Ftm1_tin = C.SENTINEL_MISSING
            ev_support_ok = False
            ev_support_reason = ev_reason or "events_unavailable"

        rec: dict[str, Any] = {
            "anon_match_key": anon,
            "match_local": int(match_local),
            "component_id": int(lab),
            "member_key": f"{anon}:c{int(lab)}",
            "n_members": int(memb.shape[0]),
            "same_ms_first_n": int(memb0.shape[0]),
            "f": int(f),
            "L": int(L),
            "tau": int(cut["tau"]),
            "t_in": int(cut["t_in"]),
            "singleton": bool(memb.shape[0] == 1),
            "c0_bits": int(c0_bits),
            "call_bits": int(call_bits),
            "c0_n": int(bin(c0_bits).count("1")),
            "call_n": int(bin(call_bits).count("1")),
            "c0_n_env_killer": int(c0_env),
            "call_n_env_killer": int(call_env),
            "c0_n_blue": int(c0_teams["n_blue"]),
            "c0_n_red": int(c0_teams["n_red"]),
            "c0_n_unknown_team": int(c0_teams["n_unknown_team"]),
            "c0_scale_max_side": int(c0_teams["scale_max_side"]),
            "call_n_blue": int(call_teams["n_blue"]),
            "call_n_red": int(call_teams["n_red"]),
            "call_scale_max_side": int(call_teams["scale_max_side"]),
            "team_map_ok": bool(team_map_ok),
            "team_map_provenance": match_meta["team_map_provenance"],
            "F_tau": int(cut["F_tau"]),
            "F_tau_idx": int(cut["F_tau_idx"]),
            "lag_f_F_tau": int(cut["lag_f_F_tau"]),
            "F_tm1": int(cut["F_tm1"]),
            "F_tm1_idx": int(cut["F_tm1_idx"]),
            "lag_f_F_tm1": int(cut["lag_f_F_tm1"]),
            "a_strict": int(cut["a_strict"]),
            "cand_10_20_tau": bool(cut["cand_10_20_tau"]),
            "cand_10_20_tm1": bool(cut["cand_10_20_tm1"]),
            "exact_10000_tau": bool(cut["exact_10000_tau"]),
            "strict_eligible": bool(cut["strict_eligible"]),
            "tau_reason": str(cut["tau_reason"]),
            "tm1_reason": str(cut["tm1_reason"]),
            "frame_differs": bool(cut["frame_differs"]),
            "ev_at_tau": int(ev_at_tau),
            "ev_between_Ftau_tau": int(ev_between_Ftau_tau),
            "ev_between_Ftm1_tin": int(ev_between_Ftm1_tin),
            "ev_types_at_tau_n": int(sum(ev_types_at_tau.values())) if ev_support_ok else C.SENTINEL_MISSING,
            "events_support_ok": bool(ev_support_ok),
            "events_support_reason": str(ev_support_reason)[:64],
            "raw_presence_verified": False,
            "participation_mode": "cache_only_proxy",
            "support_start_ms": max(0, int(usable_frames[0])) if usable_frames is not None else 0,
        }

        # Participation at both cutoffs × both radii
        for cut_name, F_idx_key in (("tau", "F_tau_idx"), ("tm1", "F_tm1_idx")):
            fi = int(cut[F_idx_key])
            for radius in C.DIAGNOSTIC_RADII:
                tag = f"{cut_name}_r{int(radius)}"
                if pack is None or fi < 0 or usable_frames is None:
                    _fill_participation(rec, tag, None, c0_bits)
                    continue
                xy = pack["xy_raw_minute"][fi]
                hp = _channel_at(pack["node_minute"], fi, C.IDX_HP_PCT)
                part = CT.participation_proxy_cache(xy, hp, locs_x, locs_y, radius, c0_bits)
                _fill_participation(rec, tag, part, c0_bits)
                if cut_name == "tau" and int(radius) == 1800:
                    rec["_frame_xy_tau"] = xy.astype(np.float64)
                    rec["_hp_tau"] = hp.astype(np.float64)
                    rec["_frame_ts_tau"] = int(cut["F_tau"])
                    rec["_frame_idx_tau"] = int(fi)
                if cut_name == "tm1" and int(radius) == 1800:
                    rec["_frame_xy_tm1"] = xy.astype(np.float64)
                    rec["_hp_tm1"] = hp.astype(np.float64)
                    rec["_frame_ts_tm1"] = int(cut["F_tm1"])
                    rec["_frame_idx_tm1"] = int(fi)

        if rec.get("support_ok_tau_r1800") and rec.get("support_ok_tau_r2000"):
            rec["radius_mono_p_lower_tau"] = CT.radius_monotonicity(
                rec["p_lower_n_tau_r1800"], rec["p_lower_n_tau_r2000"]
            )
            rec["radius_mono_p_upper_tau"] = CT.radius_monotonicity(
                rec["p_upper_n_tau_r1800"], rec["p_upper_n_tau_r2000"]
            )
        else:
            rec["radius_mono_p_lower_tau"] = False
            rec["radius_mono_p_upper_tau"] = False

        # Post-frame support for BOTH tau and tm1 cutoffs × h60/h90/h120
        bi = baseline_by_lab.get(int(lab))
        for h in C.HORIZONS_S:
            if bi is not None and baseline_rows is not None:
                e = int(baseline_rows[f"e_h{h}"][bi])
                valid = int(baseline_rows[f"valid_h{h}"][bi])
            else:
                e = C.SENTINEL_MISSING
                valid = 0
            rec[f"e_h{h}"] = e
            rec[f"valid_h{h}"] = int(valid)

            # Actual endpoint mismatch vs current pure formula using baseline boundary inputs.
            if bi is not None and baseline_rows is not None:
                nk = int(baseline_rows["next_kill_after_L"][bi])
                nt = int(baseline_rows["next_component_tau"][bi])
                ge = int(baseline_rows["game_end_ms"][bi])
                ge_known = bool(int(baseline_rows["game_end_known"][bi]))
                e_re = CT.recompute_endpoint_ms(L, h, nk, nt, ge, ge_known)
                if e >= 0 and e_re >= 0 and e_re != e:
                    reconcile["endpoint_mismatches"] += 1
                rec[f"e_h{h}_recomputed"] = int(e_re)

            for cut_name, F_key, Fi_key, cutoff_key in (
                ("tau", "F_tau", "F_tau_idx", "tau"),
                ("tm1", "F_tm1", "F_tm1_idx", "t_in"),
            ):
                _post_for_cutoff(
                    rec,
                    cut_name=cut_name,
                    h=h,
                    e=e,
                    valid=valid,
                    L=L,
                    F_pre=int(cut[F_key]),
                    i_pre=int(cut[Fi_key]),
                    cutoff_ms=int(cut[cutoff_key]),
                    pack=pack,
                    usable_frames=usable_frames,
                    event_ts=event_ts,
                    events_ok=events_ok,
                    frame_reason=frame_reason,
                )

        # Baseline reconcile (cutoff / C0)
        if bi is not None and baseline_rows is not None:
            if int(baseline_rows["f"][bi]) != f or int(baseline_rows["L"][bi]) != L:
                reconcile["cutoff_mismatches"] += 1
            base_F = int(baseline_rows["F_inclusive"][bi])
            got_F = int(cut["F_tau"])
            if not ((base_F < 0 and got_F < 0) or base_F == got_F):
                reconcile["cutoff_mismatches"] += 1
            base_d = int(baseline_rows["delta"][bi])
            got_d = int(cut["lag_f_F_tau"])
            if not ((base_d < 0 and got_d < 0) or base_d == got_d):
                reconcile["cutoff_mismatches"] += 1
            if int(baseline_rows["exact_10000"][bi]) != int(bool(cut["exact_10000_tau"])):
                reconcile["cutoff_mismatches"] += 1
            if int(baseline_rows["strict_eligible"][bi]) != int(bool(cut["strict_eligible"])):
                reconcile["cutoff_mismatches"] += 1
            if int(baseline_rows["C0_n_credited"][bi]) != int(bin(c0_bits).count("1")):
                reconcile["c0_count_mismatches"] += 1

        rec["first_kill_x"] = locs_x.copy()
        rec["first_kill_y"] = locs_y.copy()
        components.append(rec)

    match_meta["n_components"] = len(components)
    return {
        "match": match_meta,
        "components": components,
        "n_components": len(components),
        "baseline_reconcile": reconcile,
    }
