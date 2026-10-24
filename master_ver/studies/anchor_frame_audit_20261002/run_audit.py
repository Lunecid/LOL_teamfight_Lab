"""Full-corpus anchor-frame diagnostic runner (CPU<=4, no V/q fit)."""
from __future__ import annotations

# Thread caps must precede numpy in workers; parent imports thread_limits first via __main__.
from . import thread_limits as TL  # noqa: F401

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C
from . import contracts as CT
from .io_source import (
    file_sha256,
    list_batch_ids,
    load_adopted_labels_for_batch,
    load_json,
    load_source_bundle,
    verify_identity,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return file_sha256(path)


def build_full_labels(n_kills: int, n_matches: int, log) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.full(n_kills, -3, dtype=np.int64)  # -3 = unset
    batch_ids = list_batch_ids()
    info: dict[str, Any] = {"n_batches": len(batch_ids), "batches": [], "identity_ok": True}
    for bid in batch_ids:
        b = load_adopted_labels_for_batch(bid)
        if b["identity_source_kills"] != C.EXPECTED_KILLS_SHA256:
            info["identity_ok"] = False
            raise RuntimeError(f"batch {bid} identity kills sha mismatch")
        actual = file_sha256(C.ST_BATCHES / f"batch_{bid:04d}.npz")
        if actual != b["npz_sha256"]:
            raise RuntimeError(f"batch {bid} npz sha mismatch")
        info["batches"].append({"batch_id": bid, "n_event_rows": b["n_event_rows"], "npz_sha256": actual})
        info.setdefault("_raw", []).append(b)
    return labels, info


def place_labels(labels: np.ndarray, ranges: dict[str, np.ndarray], raw_batches: list[dict], n_matches: int) -> dict[str, Any]:
    seen = np.zeros(n_matches, dtype=np.int8)
    n_placed = 0
    for b in raw_batches:
        labs = b["labels"]
        mls = b["match_locals"]
        cursor = 0
        for ml in mls.tolist():
            ml = int(ml)
            if seen[ml]:
                raise RuntimeError(f"duplicate match_local in batches: {ml}")
            seen[ml] = 1
            s = int(ranges["starts"][ml])
            e = int(ranges["ends"][ml])
            n = e - s
            chunk = labs[cursor : cursor + n]
            if chunk.shape[0] != n:
                raise RuntimeError(f"batch label slice short for match {ml}")
            labels[s:e] = chunk
            cursor += n
            n_placed += n
        if cursor != labs.shape[0]:
            raise RuntimeError(f"batch {b['batch_id']} unused label rows")
    unset = int((labels == -3).sum())
    return {
        "n_matches_seen": int(seen.sum()),
        "n_matches_expected": n_matches,
        "n_label_rows_placed": n_placed,
        "n_unset": unset,
        "coverage_ok": int(seen.sum()) == n_matches and unset == 0 and n_placed == labels.shape[0],
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    from . import thread_limits as _tl  # noqa: F401

    import numpy as _np

    from .audit_match import process_match as _pm

    t = _np.asarray(payload["t"], dtype=_np.int64)
    x = _np.asarray(payload["x"], dtype=_np.float32)
    y = _np.asarray(payload["y"], dtype=_np.float32)
    killer = _np.asarray(payload["killer"], dtype=_np.int64)
    victim = _np.asarray(payload["victim"], dtype=_np.int64)
    assist = _np.asarray(payload["assist"], dtype=_np.int64)
    labels = _np.asarray(payload["labels"], dtype=_np.int64)
    blue = _np.asarray(payload["blue_pids"], dtype=_np.int16)
    try:
        out = _pm(
            int(payload["match_local"]),
            str(payload["match_id"]),
            t, x, y, killer, victim, assist, labels, blue,
            int(payload["duration_ms"]),
        )
        return {"ok": True, "match_local": int(payload["match_local"]), "result": out}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "match_local": int(payload["match_local"]),
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _process_tree_rss_bytes() -> dict[str, int | None]:
    """Distinguish parent-only RSS from observed process-tree RSS."""
    try:
        import psutil

        parent = psutil.Process(os.getpid())
        parent_rss = int(parent.memory_info().rss)
        total = parent_rss
        children = 0
        for ch in parent.children(recursive=True):
            try:
                total += int(ch.memory_info().rss)
                children += 1
            except (psutil.Error, OSError):
                continue
        return {
            "parent_rss_bytes": parent_rss,
            "process_tree_rss_bytes": total,
            "n_child_processes_sampled": children,
        }
    except Exception:
        return {
            "parent_rss_bytes": _peak_rss_bytes(),
            "process_tree_rss_bytes": None,
            "n_child_processes_sampled": None,
        }


COMP_KEYS_INT = [
    "match_local", "component_id", "n_members", "f", "L", "tau", "span_ms",
    "same_ms_first_n", "C0_n_blue", "C0_n_red", "C0_n_unknown", "C0_n_credited",
    "C0_n_env_killer", "C0_scale_max_side", "Call_n_blue", "Call_n_red",
    "Call_n_unknown", "Call_n_credited", "Call_n_env_killer", "Call_scale_max_side",
    "foreign_kills_in_tau_L", "foreign_near", "foreign_distant", "other_anchors_in_tau_L",
    "n_overlapping_spans", "F_inclusive", "delta", "t_in", "F_strict", "a_strict",
    "last_frame_ms", "game_end_ms",
    "e_h60", "valid_h60", "followup_ms_h60",
    "e_h90", "valid_h90", "followup_ms_h90",
    "e_h120", "valid_h120", "followup_ms_h120",
    "next_component_tau", "next_kill_after_L",
]
COMP_KEYS_BOOL = [
    "singleton", "exact_10000", "negative_tau", "strict_eligible",
    "game_end_known", "overlap_next_tau_le_L", "missing_terminal_game_end",
    "terminal_support_authoritative",
]
COMP_KEYS_FLOAT = ["diameter"]
COMP_KEYS_STR = [
    "anon_match_key", "member_key", "first_kill_band", "inclusive_reason",
    "strict_reason", "e_h60_reasons", "invalid_h60", "clip_h60",
    "e_h90_reasons", "invalid_h90", "clip_h90",
    "e_h120_reasons", "invalid_h120", "clip_h120",
]


def components_to_columns(comps: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    n = len(comps)
    out: dict[str, np.ndarray] = {}
    for k in COMP_KEYS_INT:
        out[k] = np.asarray([int(c.get(k, C.SENTINEL_MISSING)) for c in comps], dtype=np.int64)
    for k in COMP_KEYS_BOOL:
        out[k] = np.asarray([1 if c.get(k) else 0 for c in comps], dtype=np.int8)
    for k in COMP_KEYS_FLOAT:
        out[k] = np.asarray([float(c.get(k, np.nan)) for c in comps], dtype=np.float64)
    for k in COMP_KEYS_STR:
        out[k] = np.asarray([str(c.get(k, "")) for c in comps], dtype=object)
    return out


def _hist_int(arr: np.ndarray) -> dict[str, int]:
    h: dict[str, int] = {}
    for v in arr.tolist():
        h[str(int(v))] = h.get(str(int(v)), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def _hist_str(arr: np.ndarray) -> dict[str, int]:
    h: dict[str, int] = {}
    for v in arr.tolist():
        h[str(v)] = h.get(str(v), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))


def _match_equal_weight_share(mask: np.ndarray, match_local: np.ndarray) -> float | None:
    """Equal-match-weighted share: mean over matches of within-match event share."""
    if not mask.any():
        return 0.0 if match_local.size else None
    shares = []
    for ml in np.unique(match_local):
        m = match_local == ml
        denom = int(m.sum())
        if denom == 0:
            continue
        shares.append(float(mask[m].sum()) / float(denom))
    return float(np.mean(shares)) if shares else None


def _window_contracted_summary(cols: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    """Contracted per-window distributions (C0/C_all are recorded contributors, not P0)."""
    if not mask.any():
        return {
            "n_components": 0,
            "n_matches": 0,
            "note": "C0/C_all are recorded kill-credit contributors, not proximity-based P0 or simultaneous participation",
        }
    sub = {k: v[mask] for k, v in cols.items()}
    delta = sub["delta"].astype(np.float64)
    valid_delta = delta >= 0
    out: dict[str, Any] = {
        "n_components": int(mask.sum()),
        "n_matches": int(np.unique(sub["match_local"]).size),
        "weighting": {
            "event_weighted": "each component one unit",
            "equal_match_weighted": "mean over matches of within-match event share / mean",
        },
        "note_C0_Call": (
            "C0/C_all are recorded contributors at earliest/all member kills; "
            "not proximity-based P0 or actual simultaneous participation"
        ),
        "C0_scale_max_side_hist_event": _hist_int(sub["C0_scale_max_side"]),
        "Call_scale_max_side_hist_event": _hist_int(sub["Call_scale_max_side"]),
        "C0_team_count_hist_event": {
            "n_blue": _hist_int(sub["C0_n_blue"]),
            "n_red": _hist_int(sub["C0_n_red"]),
        },
        "Call_team_count_hist_event": {
            "n_blue": _hist_int(sub["Call_n_blue"]),
            "n_red": _hist_int(sub["Call_n_red"]),
        },
        "first_kill_band_hist_event": _hist_str(sub["first_kill_band"]),
        "singleton_share_event": float(sub["singleton"].mean()),
        "singleton_share_equal_match": _match_equal_weight_share(
            sub["singleton"].astype(bool), sub["match_local"]
        ),
        "lag_delta_summary_event": {
            "n_valid": int(valid_delta.sum()),
            "mean": float(delta[valid_delta].mean()) if valid_delta.any() else None,
            "p50": float(np.median(delta[valid_delta])) if valid_delta.any() else None,
            "p90": float(np.quantile(delta[valid_delta], 0.9)) if valid_delta.any() else None,
            "min": int(delta[valid_delta].min()) if valid_delta.any() else None,
            "max": int(delta[valid_delta].max()) if valid_delta.any() else None,
            "primary_requested_lag": "first_kill_minus_frame in [10000,20000] closed; 25000/30000 supplementary",
        },
        "foreign_kill_any_share_event": float((sub["foreign_kills_in_tau_L"] > 0).mean()),
        "other_anchor_any_share_event": float((sub["other_anchors_in_tau_L"] > 0).mean()),
        "overlap_span_any_share_event": float((sub["n_overlapping_spans"] > 0).mean()),
        "foreign_kill_any_share_equal_match": _match_equal_weight_share(
            sub["foreign_kills_in_tau_L"] > 0, sub["match_local"]
        ),
    }
    for h in C.HORIZONS_S:
        v = sub[f"valid_h{h}"].astype(bool)
        fu = sub[f"followup_ms_h{h}"]
        out[f"h{h}"] = {
            "n_valid": int(v.sum()),
            "n_invalid": int((~v).sum()),
            "valid_share_event": float(v.mean()),
            "followup_ms_mean_valid": float(fu[v].mean()) if v.any() else None,
            "followup_ms_p50_valid": float(np.median(fu[v])) if v.any() else None,
            "n_full_horizon": int(((fu == h * 1000) & v).sum()) if v.any() else 0,
            "clip_reason_top": dict(Counter(sub[f"clip_h{h}"].tolist()).most_common(15)),
        }
    return out


def aggregate_from_columns(cols: dict[str, np.ndarray], terminals: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(cols["f"].shape[0])
    delta = cols["delta"]
    exact = cols["exact_10000"].astype(bool)
    strict = cols["strict_eligible"].astype(bool)
    valid_delta = delta >= 0

    def count_cum(w: int) -> dict[str, Any]:
        m = valid_delta & (delta >= C.ANCHOR_OFFSET_MS) & (delta <= C.ANCHOR_OFFSET_MS + w)
        base = {
            "n_components": int(m.sum()),
            "n_matches": int(np.unique(cols["match_local"][m]).size) if m.any() else 0,
            "n_exact_subset": int((m & exact).sum()),
            "n_strict_eligible": int((m & strict).sum()),
            "closed_interval": f"[{C.ANCHOR_OFFSET_MS},{C.ANCHOR_OFFSET_MS + w}]",
            "exact_integer_equality": True,
        }
        base["contracted"] = _window_contracted_summary(cols, m)
        return base

    cum = {f"w_{w}": count_cum(w) for w in C.INCLUSIVE_WIDTHS_MS}
    widths = list(C.INCLUSIVE_WIDTHS_MS)
    for i, w in enumerate(widths):
        if i == 0:
            cum[f"w_{w}"]["added_components"] = cum[f"w_{w}"]["n_components"]
        else:
            prev = widths[i - 1]
            cum[f"w_{w}"]["added_components"] = (
                cum[f"w_{w}"]["n_components"] - cum[f"w_{prev}"]["n_components"]
            )

    bands = {}
    for lo, hi in C.DISJOINT_DELTA_BANDS:
        m = valid_delta & (delta >= lo) & (delta < hi)
        bands[f"[{lo},{hi})"] = {
            "n_components": int(m.sum()),
            "n_matches": int(np.unique(cols["match_local"][m]).size) if m.any() else 0,
        }
    upper = {}
    for g in C.UPPER_GAP_MS:
        m = valid_delta & (delta >= C.ANCHOR_OFFSET_MS) & (delta <= g)
        upper[f"inclusive_delta_le_{g}"] = {
            "n_components": int(m.sum()),
            "n_matches": int(np.unique(cols["match_local"][m]).size) if m.any() else 0,
            "note": "supplementary; primary requested lag is closed [10000,20000]",
        }

    reasons = _hist_str(cols["inclusive_reason"]) if n else {}
    band_hist = _hist_str(cols["first_kill_band"]) if n else {}

    both = int((exact & strict).sum())
    incl_not_strict = int((exact & ~strict).sum())
    strict_among_delta_ok = int((valid_delta & strict).sum())
    inclusive_exact = int(exact.sum())

    term_agg = {
        "n_matches": len(terminals),
        "n_game_end_known": sum(1 for t in terminals if t.get("game_end_known")),
        "n_game_end_unknown": sum(1 for t in terminals if not t.get("game_end_known")),
        "n_game_end_multiple_events": sum(1 for t in terminals if t.get("game_end_multiple_events")),
        "n_game_end_duplicate_ts": sum(1 for t in terminals if t.get("game_end_duplicate_ts")),
        "n_game_end_ambiguous_distinct": sum(1 for t in terminals if t.get("game_end_ambiguous_distinct")),
        "n_game_end_malformed_any": sum(1 for t in terminals if int(t.get("game_end_malformed") or 0) > 0),
        "n_terminal_crosscheck_mismatch": sum(1 for t in terminals if not t.get("terminal_crosscheck_match", True)),
        "n_ge_minus_last_frame_negative": sum(1 for t in terminals if t.get("ge_minus_last_frame_negative")),
        "n_meta_duration_disagrees_last_frame": sum(
            1 for t in terminals if t.get("meta_duration_disagrees_last_frame")
        ),
        "n_meta_duration_disagrees_game_end": sum(
            1 for t in terminals if t.get("meta_duration_disagrees_game_end")
        ),
        "n_frame_integrity_fail": sum(1 for t in terminals if not t.get("frame_integrity_ok", True)),
        "n_frame_nonmonotone": sum(1 for t in terminals if t.get("frame_nonmonotone")),
        "n_frame_negative_ts": sum(1 for t in terminals if t.get("frame_has_negative_ts")),
        "n_missing_npz": sum(1 for t in terminals if t.get("minute_ts_reason")),
        "n_missing_events": sum(1 for t in terminals if t.get("events_reason")),
        "ge_minus_last_frame_hist": {},
    }
    for t in terminals:
        d = t.get("ge_minus_last_frame", C.SENTINEL_MISSING)
        if d == C.SENTINEL_MISSING:
            continue
        key = str(int(d))
        term_agg["ge_minus_last_frame_hist"][key] = term_agg["ge_minus_last_frame_hist"].get(key, 0) + 1

    follow = {}
    for h in C.HORIZONS_S:
        v = cols[f"valid_h{h}"].astype(bool) if n else np.zeros(0, dtype=bool)
        fu = cols[f"followup_ms_h{h}"] if n else np.zeros(0, dtype=np.int64)
        follow[f"h{h}"] = {
            "n_valid": int(v.sum()),
            "n_invalid": int((~v).sum()) if n else 0,
            "followup_ms_mean_valid": float(fu[v].mean()) if v.any() else None,
            "followup_ms_p50_valid": float(np.median(fu[v])) if v.any() else None,
            "n_full_horizon": int(((fu == h * 1000) & v).sum()) if v.any() else 0,
            "n_overlap_flag": int(cols["overlap_next_tau_le_L"].sum()) if n else 0,
            "clip_reason_top": dict(Counter(cols[f"clip_h{h}"].tolist()).most_common(20)) if n else {},
        }

    # Boundary diagnostics: span>14000 or diameter>2000 are chaining spans / neighbor-condition
    # extents, not detector errors. Radius 2000 is a neighbor condition.
    chain = {
        "n_span_gt_14000": int((cols["span_ms"] > C.EPS_T_MS).sum()) if n else 0,
        "n_diameter_gt_2000": int((cols["diameter"] > C.EPS_S).sum()) if n else 0,
        "note": (
            "span>14000 or diameter>2000 summarize chaining under neighbor conditions; "
            "not labeled detector errors. Radius 2000 is a neighbor condition, not a cluster cap."
        ),
    }

    return {
        "n_components": n,
        "n_matches_with_components": int(np.unique(cols["match_local"]).size) if n else 0,
        "n_singletons": int(cols["singleton"].sum()) if n else 0,
        "singleton_share": float(cols["singleton"].mean()) if n else None,
        "exact_10000": {
            "n_components": inclusive_exact,
            "n_matches": int(np.unique(cols["match_local"][exact]).size) if exact.any() else 0,
            "note": "F_inclusive==tau; includes anchor-time frame; incompatible with historical strict t_in=tau-1 cutoff",
        },
        "strict_eligible": {
            "n_components": int(strict.sum()),
            "n_matches": int(np.unique(cols["match_local"][strict]).size) if strict.any() else 0,
        },
        "inclusive_vs_strict": {
            "exact_and_strict": both,
            "exact_not_strict": incl_not_strict,
            "strict_among_valid_delta": strict_among_delta_ok,
            "discrepancy_note": "exact inclusive cases can fail strict because F==tau is excluded by t_in=tau-1",
        },
        "cumulative_inclusive": cum,
        "disjoint_delta_bands": bands,
        "upper_gap_diagnostics": upper,
        "inclusive_reasons": reasons,
        "C0_scale_max_side_hist": _hist_int(cols["C0_scale_max_side"]) if n else {},
        "Call_scale_max_side_hist": _hist_int(cols["Call_scale_max_side"]) if n else {},
        "first_kill_band_hist": band_hist,
        "span_ms_max": int(cols["span_ms"].max()) if n else None,
        "diameter_max": float(cols["diameter"].max()) if n else None,
        "n_foreign_any": int((cols["foreign_kills_in_tau_L"] > 0).sum()) if n else 0,
        "n_overlap_span_any": int((cols["n_overlapping_spans"] > 0).sum()) if n else 0,
        "n_same_ms_first_gt1": int((cols["same_ms_first_n"] > 1).sum()) if n else 0,
        "chaining_neighbor_summaries": chain,
        "terminal": term_agg,
        "followup": follow,
        "primary_lag_window": {
            "interval": "[10000,20000]",
            "closed": True,
            "n_components": int(
                (valid_delta & (delta >= 10_000) & (delta <= 20_000)).sum()
            ),
            "n_matches": int(
                np.unique(cols["match_local"][valid_delta & (delta >= 10_000) & (delta <= 20_000)]).size
            )
            if n
            else 0,
        },
    }


def reconcile_gates(cols: dict[str, np.ndarray], agg: dict[str, Any]) -> dict[str, Any]:
    delta = cols["delta"]
    exact = cols["exact_10000"].astype(bool)
    gates: dict[str, Any] = {}
    band_sum = sum(v["n_components"] for v in agg["disjoint_delta_bands"].values())
    n_valid_delta = int((delta >= 0).sum())
    n_missing_delta = int((delta < 0).sum())
    gates["disjoint_bands_cover_valid_delta"] = band_sum == n_valid_delta
    gates["band_sum"] = band_sum
    gates["n_valid_delta"] = n_valid_delta
    gates["n_missing_delta"] = n_missing_delta
    exact_n = int(exact.sum())
    subset_ok = True
    for w in C.INCLUSIVE_WIDTHS_MS:
        m = (delta >= C.ANCHOR_OFFSET_MS) & (delta <= C.ANCHOR_OFFSET_MS + w) & (delta >= 0)
        if exact_n != int((m & exact).sum()):
            subset_ok = False
        if exact_n > int(m.sum()):
            subset_ok = False
    gates["exact_subset_of_all_cumulative"] = subset_ok and all(
        agg["cumulative_inclusive"][f"w_{w}"]["n_exact_subset"] == exact_n for w in C.INCLUSIVE_WIDTHS_MS
    )
    F = cols["F_strict"]
    t_in = cols["t_in"]
    a = cols["a_strict"]
    recomputed = np.zeros(len(a), dtype=bool)
    ok_mask = (F >= 0) & (a >= 0)
    recomputed[ok_mask] = (a[ok_mask] >= 0) & (a[ok_mask] < C.ANCHOR_OFFSET_MS)
    a_check = np.full(len(a), C.SENTINEL_MISSING, dtype=np.int64)
    a_check[ok_mask] = t_in[ok_mask] - F[ok_mask]
    gates["strict_recomputed_matches_flag"] = bool(
        np.array_equal(recomputed.astype(np.int8), cols["strict_eligible"])
    )
    gates["a_equals_tin_minus_F"] = bool(np.all(a[ok_mask] == a_check[ok_mask])) if ok_mask.any() else True
    gates["span_max_matches_st"] = agg["span_ms_max"] == C.EXPECTED_SPAN_MAX
    gates["diameter_max_close"] = abs(float(agg["diameter_max"] or 0) - C.EXPECTED_DIAMETER_MAX) < 1e-3
    gates["n_components_expected"] = agg["n_components"] == C.EXPECTED_MIN1_COMPONENTS
    gates["all_ok"] = all(
        bool(gates[k]) for k in gates if k not in ("band_sum", "n_valid_delta", "n_missing_delta")
    )
    return gates


def _member_case_detail(
    memb: np.ndarray,
    f: int,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    killer: np.ndarray,
    victim: np.ndarray,
    assist: np.ndarray,
    blue_pids: np.ndarray,
) -> dict[str, Any]:
    """Actual member lists from assignments (not reconstructed from f/L alone)."""
    members = []
    credited_sets = []
    for ii in memb.tolist():
        ii = int(ii)
        cred = CT.team_credit_counts(
            killer, victim, assist, np.array([ii], dtype=np.int64), blue_pids
        )
        members.append({
            "source_ordinal": ii,
            "rel_t_ms": int(t[ii]) - int(f),
            "t_ms": int(t[ii]),
            "x": float(x[ii]),
            "y": float(y[ii]),
            "C_n_blue": cred["n_blue"],
            "C_n_red": cred["n_red"],
            "C_n_credited": cred["n_credited"],
            "C_n_env_killer": cred["n_env_killer"],
            "credited_blue": cred["blue"],
            "credited_red": cred["red"],
        })
        credited_sets.append(set(cred["blue"]) | set(cred["red"]))
    # Connecting temporal/spatial edges under neighbor condition
    edges = []
    for a_i in range(len(memb)):
        for b_i in range(a_i + 1, len(memb)):
            ia, ib = int(memb[a_i]), int(memb[b_i])
            dt = abs(int(t[ia]) - int(t[ib]))
            dist = float(np.hypot(float(x[ia]) - float(x[ib]), float(y[ia]) - float(y[ib])))
            if dt <= C.EPS_T_MS and dist <= C.EPS_S:
                edges.append({
                    "a_source_ordinal": ia,
                    "b_source_ordinal": ib,
                    "dt_ms": int(dt),
                    "dist": dist,
                })
    # Credited-player continuity: Jaccard across consecutive members in time order
    order = np.argsort(t[memb], kind="mergesort")
    continuity = []
    prev = None
    for oi in order.tolist():
        cur = credited_sets[int(oi)]
        if prev is not None:
            union = prev | cur
            inter = prev & cur
            continuity.append({
                "intersection": sorted(inter),
                "union_n": len(union),
                "jaccard": (len(inter) / len(union)) if union else None,
            })
        prev = cur
    return {
        "members": members,
        "connecting_edges_neighbor": edges,
        "credited_player_continuity": continuity,
        "note": "members from completed assignments; radius 2000 is neighbor condition",
    }


def build_casebook(
    cols: dict[str, np.ndarray],
    ledger: dict[str, np.ndarray],
    ranges: dict,
    labels_full: np.ndarray,
    meta: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Deterministic compact anonymous casebook using actual assignment members."""
    n = cols["f"].shape[0]
    if n == 0:
        return {"selection_rule": "stable_sorted_keys", "cases": [], "summaries": {}}

    # Stable sort key: (anon_match_key, component_id, member_key index)
    order = np.lexsort((cols["component_id"], cols["match_local"]))
    picks: dict[str, int] = {}

    def first_where(mask: np.ndarray, name: str) -> None:
        for oi in order.tolist():
            if bool(mask[oi]):
                picks[name] = int(oi)
                return

    first_where(cols["singleton"].astype(bool) & cols["exact_10000"].astype(bool), "singleton_exact")
    first_where(cols["singleton"].astype(bool) & ~cols["exact_10000"].astype(bool), "singleton_typical")
    first_where((cols["n_members"] > 1) & (cols["n_members"] <= 4), "typical_multi_kill")
    first_where(cols["span_ms"] == cols["span_ms"].max(), "long_chain_max_span")
    first_where(cols["diameter"] == cols["diameter"].max(), "wide_diameter_max")
    first_where(cols["same_ms_first_n"] > 1, "same_ms_first")
    first_where(cols["n_overlapping_spans"] > 0, "overlapping_spans_concurrent")
    first_where(cols["exact_10000"].astype(bool) & (cols["n_members"] > 1), "exact_multi")
    first_where(cols["delta"] == 20_000, "delta_exact_20000")
    first_where(cols["overlap_next_tau_le_L"].astype(bool), "next_tau_overlap")
    first_where(~cols["game_end_known"].astype(bool), "game_end_unknown_comp")
    first_where(cols["span_ms"] > C.EPS_T_MS, "span_gt_14000_chaining")
    first_where(cols["diameter"] > C.EPS_S, "diameter_gt_2000_neighbor_extent")

    cases = []
    for name in sorted(picks.keys()):
        i = picks[name]
        ml = int(cols["match_local"][i])
        lab = int(cols["component_id"][i])
        s, e = int(ranges["starts"][ml]), int(ranges["ends"][ml])
        t = ledger["t_ms"][s:e]
        x = ledger["x"][s:e]
        y = ledger["y"][s:e]
        killer = ledger["killer"][s:e]
        victim = ledger["victim"][s:e]
        assist = ledger["assist_mask"][s:e]
        labs = labels_full[s:e]
        memb = np.flatnonzero(labs == lab)
        blue = meta["blue_pids"][ml]
        detail = _member_case_detail(
            memb, int(cols["f"][i]), t, x, y, killer, victim, assist, blue
        )
        cases.append({
            "selection_rule": f"first_stable_sorted_where:{name}",
            "category": name,
            "anon_match_key": str(cols["anon_match_key"][i]),
            "member_key": str(cols["member_key"][i]),
            "n_members": int(cols["n_members"][i]),
            "f": int(cols["f"][i]),
            "L": int(cols["L"][i]),
            "tau": int(cols["tau"][i]),
            "span_ms": int(cols["span_ms"][i]),
            "diameter": float(cols["diameter"][i]),
            "delta": int(cols["delta"][i]),
            "exact_10000": bool(cols["exact_10000"][i]),
            "strict_eligible": bool(cols["strict_eligible"][i]),
            "C0_scale_max_side": int(cols["C0_scale_max_side"][i]),
            "Call_n_blue": int(cols["Call_n_blue"][i]),
            "Call_n_red": int(cols["Call_n_red"][i]),
            "foreign_kills_in_tau_L": int(cols["foreign_kills_in_tau_L"][i]),
            "n_overlapping_spans": int(cols["n_overlapping_spans"][i]),
            "same_ms_first_n": int(cols["same_ms_first_n"][i]),
            "source_provenance": {
                "st_run": str(C.ST_OUT),
                "adopted_cfg": C.ADOPTED_CFG,
                "cache_dir": str(C.CACHE_DIR),
                "match_local_on_disk": ml,
                "component_id_on_disk": lab,
                "event_source_ordinals_on_disk": [int(x) for x in memb.tolist()],
            },
            "member_detail": detail,
            "match_n_kills": int(e - s),
            "detector_error_label": False,
            "interpretation": (
                "span>14000 / diameter>2000 are chaining/neighbor-extent diagnostics, "
                "not automatic detector errors"
                if name.startswith("span_") or name.startswith("diameter_")
                else ""
            ),
        })
    summaries = {
        "n_span_gt_14000": int((cols["span_ms"] > C.EPS_T_MS).sum()),
        "n_diameter_gt_2000": int((cols["diameter"] > C.EPS_S).sum()),
        "note": "Not labeled detector errors; radius 2000 is neighbor condition",
    }
    return {
        "selection_rule": "deterministic first index under stable sort (match_local, component_id)",
        "neighbor_radius_diagnostic": C.FOREIGN_NEAR_RADIUS,
        "summaries": summaries,
        "cases": cases,
    }


def estimate_resource_storage(cols: dict[str, np.ndarray]) -> dict[str, Any]:
    """Unique (anonymous match key, query time) counts; no feature matrices / no V eval."""
    # Prefer declared dims from existing artifacts when present; else documented approximations.
    dims: dict[str, Any] = {
        "bytes_per_float32": 4,
        "source": {},
        "assumptions": [],
    }
    # q / PT from frozen bundles if available
    try:
        if C.Q_BUNDLE.is_file():
            qb = json.loads(C.Q_BUNDLE.read_text(encoding="utf-8"))
            dims["q_direct_declared"] = int(qb.get("n_features") or qb.get("n_columns") or 1152)
            dims["source"]["q_bundle"] = str(C.Q_BUNDLE)
        else:
            dims["q_direct_declared"] = 1152
            dims["assumptions"].append("q_direct_declared fallback 1152; q bundle missing")
    except Exception:
        dims["q_direct_declared"] = 1152
        dims["assumptions"].append("q_direct_declared fallback 1152; bundle unreadable")
    try:
        if C.PT_FLEX_BUNDLE.is_file():
            pb = json.loads(C.PT_FLEX_BUNDLE.read_text(encoding="utf-8"))
            dims["pt_flex_raw"] = int(pb.get("n_columns") or 2)
            dims["pt_flex_expanded"] = int((pb.get("model") or {}).get("fit_info", {}).get("n_features") or 47)
            dims["source"]["pt_flex_bundle"] = str(C.PT_FLEX_BUNDLE)
        else:
            dims["pt_flex_raw"] = 2
            dims["pt_flex_expanded"] = 47
            dims["assumptions"].append("pt_flex dims fallback; bundle missing")
    except Exception:
        dims["pt_flex_raw"] = 2
        dims["pt_flex_expanded"] = 47
        dims["assumptions"].append("pt_flex dims fallback; bundle unreadable")
    dims["q_M0"] = 2
    dims["state_v3_nonspatial_approx"] = 995
    dims["assumptions"].append(
        "V-only snapshot age means V and direct-q dimensions may differ; "
        "state_v3_nonspatial_approx=995 is an estimate, not a newly measured matrix width"
    )
    dims["assumptions"].append("No feature matrices built; storage is float32 nbytes estimate only")

    out: dict[str, Any] = {"dimensions": dims, "windows": {}}
    delta = cols["delta"]
    valid_delta = delta >= 0
    anon = cols["anon_match_key"]
    for w in C.INCLUSIVE_WIDTHS_MS:
        m = valid_delta & (delta >= C.ANCHOR_OFFSET_MS) & (delta <= C.ANCHOR_OFFSET_MS + w)
        if not m.any():
            out["windows"][f"w_{w}"] = {
                "n_components": 0,
                "unique_pre_strict_tau_minus_1": 0,
                "unique_pre_inclusive_tau": 0,
                "unique_post_h60_valid": 0,
                "unique_post_h90_valid": 0,
                "unique_post_h120_valid": 0,
                "unique_pre_post_union_all": 0,
            }
            continue
        # Strict pre uses t_in = tau-1; inclusive diagnostic uses tau. Keep distinct.
        pre_strict_keys = set(zip(anon[m].tolist(), cols["t_in"][m].tolist()))
        pre_incl_keys = set(zip(anon[m].tolist(), cols["tau"][m].tolist()))
        post = {}
        for h in C.HORIZONS_S:
            vm = m & cols[f"valid_h{h}"].astype(bool)
            post[h] = set(zip(anon[vm].tolist(), cols[f"e_h{h}"][vm].tolist())) if vm.any() else set()
        union = set(pre_strict_keys) | set(pre_incl_keys)
        for h in C.HORIZONS_S:
            union |= post[h]
        row = {
            "n_components": int(m.sum()),
            "n_matches": int(np.unique(cols["match_local"][m]).size),
            "unique_pre_strict_tau_minus_1": len(pre_strict_keys),
            "unique_pre_inclusive_tau": len(pre_incl_keys),
            "unique_post_h60_valid": len(post[60]),
            "unique_post_h90_valid": len(post[90]),
            "unique_post_h120_valid": len(post[120]),
            "unique_pre_post_union_all": len(union),
            "note": (
                "Counts are unique (anonymous match key, query time). "
                "Strict pre=tau-1 and inclusive pre=tau are distinct diagnostics."
            ),
            "float32_storage_estimates": {},
        }
        for name, d in {
            "state_v3_nonspatial_approx": dims["state_v3_nonspatial_approx"],
            "q_direct_declared": dims["q_direct_declared"],
            "q_M0": dims["q_M0"],
            "pt_flex_raw": dims["pt_flex_raw"],
            "pt_flex_expanded": dims["pt_flex_expanded"],
        }.items():
            row["float32_storage_estimates"][name] = {
                "dim": d,
                "bytes_unique_union_keys": len(union) * int(d) * 4,
                "bytes_pre_strict_plus_post_h90": (
                    (len(pre_strict_keys) + len(post[90])) * int(d) * 4
                ),
                "label": "estimate_not_measured_matrix",
            }
        out["windows"][f"w_{w}"] = row
    return out


def reload_and_reconcile(
    comp_path: Path,
    term_path: Path,
    agg: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Independently reload written outputs and reconcile cumulative/disjoint counts."""
    with np.load(comp_path, allow_pickle=True) as z:
        cols = {k: z[k] for k in z.files}
    terminals = load_json(term_path)
    re_agg = aggregate_from_columns(cols, terminals)
    re_gates = reconcile_gates(cols, re_agg)
    # Unique component keys
    keys = list(zip(cols["match_local"].tolist(), cols["component_id"].tolist())) if cols["f"].size else []
    n_unique_keys = len(set(keys))
    out = {
        "n_component_rows_reloaded": int(cols["f"].shape[0]),
        "n_unique_component_keys": n_unique_keys,
        "unique_keys_equal_rows": n_unique_keys == int(cols["f"].shape[0]),
        "n_matches_terminals": len(terminals),
        "n_kills_from_terminals": int(sum(int(t.get("n_kills", 0)) for t in terminals)),
        "cumulative_match_original": {
            f"w_{w}": agg["cumulative_inclusive"][f"w_{w}"]["n_components"] for w in C.INCLUSIVE_WIDTHS_MS
        },
        "cumulative_match_reloaded": {
            f"w_{w}": re_agg["cumulative_inclusive"][f"w_{w}"]["n_components"] for w in C.INCLUSIVE_WIDTHS_MS
        },
        "cumulative_equal": all(
            agg["cumulative_inclusive"][f"w_{w}"]["n_components"]
            == re_agg["cumulative_inclusive"][f"w_{w}"]["n_components"]
            for w in C.INCLUSIVE_WIDTHS_MS
        ),
        "disjoint_equal": agg["disjoint_delta_bands"] == re_agg["disjoint_delta_bands"],
        "gates_all_ok_reloaded": bool(re_gates.get("all_ok")),
        "terminal_known_reloaded": re_agg["terminal"]["n_game_end_known"],
        "terminal_crosscheck_mismatch_reloaded": re_agg["terminal"]["n_terminal_crosscheck_mismatch"],
        "horizon_valid_reloaded": {f"h{h}": re_agg["followup"][f"h{h}"]["n_valid"] for h in C.HORIZONS_S},
        "ok": True,
    }
    out["ok"] = bool(
        out["unique_keys_equal_rows"]
        and out["cumulative_equal"]
        and out["disjoint_equal"]
        and out["n_component_rows_reloaded"] == agg["n_components"]
        and out["terminal_crosscheck_mismatch_reloaded"] == 0
    )
    out["reloaded_gates"] = re_gates
    return out


def run(run_dir: Path | None = None, max_matches: int | None = None, workers: int | None = None) -> dict[str, Any]:
    run_dir = run_dir or C.new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    status_path = run_dir / "STATUS.json"
    t0 = time.perf_counter()
    t_extract0: float | None = None
    peak_parent = _peak_rss_bytes()
    peak_tree = None

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    workers = min(C.MAX_WORKER_PROCESSES, int(workers or C.MAX_WORKER_PROCESSES))
    status = {
        "complete": False,
        "started_utc": _utc(),
        "run_dir": str(run_dir),
        "adopted_cfg": C.ADOPTED_CFG,
        "workers": workers,
        "cpu_budget": C.CPU_BUDGET,
        "thread_settings": TL.applied_settings(),
        "seed": C.SEED,
        "restart_resume_supported": False,
    }
    _write_json(status_path, status)
    log(f"start run_dir={run_dir} workers={workers}")

    bundle = load_source_bundle()
    identity = verify_identity(bundle)
    _write_json(run_dir / "identity_checks.json", identity)
    if not identity["all_ok"]:
        status.update(complete=False, failed_gate="identity", finished_utc=_utc())
        _write_json(status_path, status)
        raise RuntimeError(f"identity checks failed: {identity}")

    ledger = bundle["ledger"]
    ranges = bundle["ranges"]
    meta = bundle["meta"]
    match_ids = bundle["match_ids"]
    n_matches = len(match_ids)
    n_kills = int(ledger["t_ms"].shape[0])

    labels = np.full(n_kills, -3, dtype=np.int64)
    _, lab_info = build_full_labels(n_kills, n_matches, log)
    place = place_labels(labels, ranges, lab_info.pop("_raw"), n_matches)
    lab_info["placement"] = place
    _write_json(run_dir / "label_load.json", lab_info)
    if not place["coverage_ok"]:
        status.update(complete=False, failed_gate="label_coverage", finished_utc=_utc())
        _write_json(status_path, status)
        raise RuntimeError(f"label coverage failed: {place}")

    n_assigned = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())
    n_inv = int((labels == -2).sum())
    cons = {
        "assigned": n_assigned,
        "noise": n_noise,
        "invalid": n_inv,
        "total": n_assigned + n_noise + n_inv,
        "expected_kills": C.EXPECTED_TRAIN_KILLS,
        "ok": n_assigned + n_noise + n_inv == C.EXPECTED_TRAIN_KILLS and n_noise == 0 and n_inv == 0,
    }
    _write_json(run_dir / "conservation_labels.json", cons)
    if not cons["ok"]:
        raise RuntimeError(f"label conservation failed: {cons}")

    match_indices = list(range(n_matches))
    if max_matches is not None:
        match_indices = match_indices[: int(max_matches)]
        log(f"WARNING max_matches={max_matches} - not a full-corpus completion")

    col_chunks: list[dict[str, np.ndarray]] = []
    terminals: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    n_files = 0
    n_comps = 0
    chunk_size = C.BATCH_SIZE_MATCHES
    done_matches = 0
    t_extract0 = time.perf_counter()

    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)

    try:
        for chunk_start in range(0, len(match_indices), chunk_size):
            chunk = match_indices[chunk_start : chunk_start + chunk_size]
            payloads = []
            for ml in chunk:
                s, e = int(ranges["starts"][ml]), int(ranges["ends"][ml])
                payloads.append({
                    "match_local": ml,
                    "match_id": match_ids[ml],
                    "t": ledger["t_ms"][s:e],
                    "x": ledger["x"][s:e],
                    "y": ledger["y"][s:e],
                    "killer": ledger["killer"][s:e],
                    "victim": ledger["victim"][s:e],
                    "assist": ledger["assist_mask"][s:e],
                    "labels": labels[s:e],
                    "blue_pids": meta["blue_pids"][ml],
                    "duration_ms": int(meta["duration_ms"][ml]),
                })
                n_files += 2  # npz + events.json
            if pool is None:
                results = [_worker(p) for p in payloads]
            else:
                futs = {pool.submit(_worker, p): int(p["match_local"]) for p in payloads}
                results = []
                for fut in as_completed(futs):
                    results.append(fut.result())
            # Stable order by match_local
            results.sort(key=lambda r: int(r["match_local"]))
            chunk_comps: list[dict[str, Any]] = []
            for r in results:
                if not r["ok"]:
                    errors.append(r)
                    continue
                res = r["result"]
                terminals.append(res["terminal"])
                chunk_comps.extend(res["components"])
            if chunk_comps:
                chunk_comps.sort(key=lambda c: (int(c["match_local"]), int(c["component_id"])))
                col_chunks.append(components_to_columns(chunk_comps))
                n_comps += len(chunk_comps)
            done_matches += len(chunk)
            mem = _process_tree_rss_bytes()
            if mem["parent_rss_bytes"] and (peak_parent is None or mem["parent_rss_bytes"] > peak_parent):
                peak_parent = mem["parent_rss_bytes"]
            if mem["process_tree_rss_bytes"] is not None and (
                peak_tree is None or mem["process_tree_rss_bytes"] > peak_tree
            ):
                peak_tree = mem["process_tree_rss_bytes"]
            _write_json(run_dir / "progress.json", {
                "done_matches": done_matches,
                "n_matches_target": len(match_indices),
                "n_components_so_far": n_comps,
                "n_errors": len(errors),
                "parent_rss_bytes": mem["parent_rss_bytes"],
                "process_tree_rss_bytes": mem["process_tree_rss_bytes"],
                "utc": _utc(),
            })
            log(
                f"progress {done_matches}/{len(match_indices)} components={n_comps} "
                f"errors={len(errors)} parent_rss={mem['parent_rss_bytes']} tree_rss={mem['process_tree_rss_bytes']}"
            )
            if errors:
                _write_json(run_dir / "errors.json", errors[:50])
                raise RuntimeError(f"{len(errors)} match failures; first={errors[0]}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    extract_wall_s = time.perf_counter() - t_extract0
    wall = time.perf_counter() - t0
    if col_chunks:
        cols = {k: np.concatenate([c[k] for c in col_chunks], axis=0) for k in col_chunks[0]}
        # Global stable sort
        order = np.lexsort((cols["component_id"], cols["match_local"]))
        cols = {k: v[order] for k, v in cols.items()}
    else:
        cols = components_to_columns([])
    # Stable terminal order
    terminals.sort(key=lambda t: int(t["match_local"]))

    comp_path = run_dir / "components.npz"
    np.savez_compressed(comp_path, **cols)
    term_path = run_dir / "terminals.json"
    _write_json(term_path, terminals)

    n_zero = sum(1 for t in terminals if t["n_kills"] == 0)
    zero_ok = n_zero == C.EXPECTED_ZERO_KILL if max_matches is None else None

    agg = aggregate_from_columns(cols, terminals)
    gates = reconcile_gates(cols, agg)
    gates["zero_kill_matches"] = n_zero
    gates["zero_kill_ok"] = zero_ok if zero_ok is not None else False
    gates["terminal_crosscheck_mismatches"] = agg["terminal"]["n_terminal_crosscheck_mismatch"]
    gates["terminal_crosscheck_ok"] = agg["terminal"]["n_terminal_crosscheck_mismatch"] == 0
    if max_matches is None:
        gates["all_ok"] = bool(
            gates["all_ok"]
            and zero_ok
            and agg["n_components"] == C.EXPECTED_MIN1_COMPONENTS
            and gates["terminal_crosscheck_ok"]
        )

    casebook = build_casebook(cols, ledger, ranges, labels, meta)
    _write_json(run_dir / "casebook.json", casebook)

    storage = estimate_resource_storage(cols)
    reload = reload_and_reconcile(comp_path, term_path, agg, gates)
    _write_json(run_dir / "reload_reconcile.json", reload)
    if max_matches is None:
        gates["reload_reconcile_ok"] = bool(reload.get("ok"))
        gates["all_ok"] = bool(gates["all_ok"] and reload.get("ok"))

    machine = {
        "python": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "numpy": np.__version__,
        "thread_settings": TL.applied_settings(),
        "cpu_budget": C.CPU_BUDGET,
        "workers": workers,
        "peak_parent_rss_bytes": peak_parent,
        "peak_process_tree_rss_bytes": peak_tree,
        "peak_rss_bytes_note": (
            "parent_rss is parent-only; process_tree_rss includes sampled children. "
            "Do not describe parent-only RSS as total peak."
        ),
        "wall_s": wall,
        "extraction_wall_s": extract_wall_s,
        "n_files_touched_approx": n_files,
        "n_component_rows": n_comps,
        "output_bytes_components": comp_path.stat().st_size,
        "restart_resume_supported": False,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        machine.update({
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": vm.total,
            "ram_available_bytes": vm.available,
        })
    except Exception as exc:  # noqa: BLE001
        machine["psutil_error"] = str(exc)
    try:
        import subprocess

        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        machine["nvidia_smi"] = smi.stdout.strip() if smi.returncode == 0 else smi.stderr.strip()
        machine["gpu_used_this_run"] = False
    except Exception as exc:  # noqa: BLE001
        machine["nvidia_smi_error"] = str(exc)

    code_files = sorted(C.STUDY.rglob("*.py"))
    code_hash = hashlib.sha256()
    for p in code_files:
        if p.name.startswith("_probe"):
            continue
        code_hash.update(p.read_bytes())
    # Source / assignment hashes already in identity; record again for completion gate
    source_hashes = {
        "kills_sha256_expected": C.EXPECTED_KILLS_SHA256,
        "ledger_sha256": file_sha256(C.ST_SOURCE / "event_ledger_train.npz"),
        "ranges_sha256": file_sha256(C.ST_SOURCE / "ledger_match_ranges.npz"),
        "match_ids_sha256": file_sha256(C.ST_SOURCE / "train_match_ids.json"),
        "match_meta_sha256": file_sha256(C.ST_SOURCE / "train_match_meta.npz"),
    }
    manifest = {
        "generated_utc": _utc(),
        "complete": bool(gates.get("all_ok")) and max_matches is None,
        "run_dir": str(run_dir),
        "adopted_cfg": C.ADOPTED_CFG,
        "thresholds": {
            "eps_t_ms": C.EPS_T_MS,
            "eps_s": C.EPS_S,
            "min_samples": C.MIN_SAMPLES,
            "anchor_offset_ms": C.ANCHOR_OFFSET_MS,
            "foreign_near_radius": C.FOREIGN_NEAR_RADIUS,
            "inclusive_widths_ms": list(C.INCLUSIVE_WIDTHS_MS),
            "upper_gap_ms": list(C.UPPER_GAP_MS),
            "primary_lag_closed": [10_000, 20_000],
            "upper_gap_supplementary": True,
        },
        "identity": identity,
        "source_hashes": source_hashes,
        "conservation_labels": cons,
        "aggregates": agg,
        "gates": gates,
        "reload_reconcile": reload,
        "machine": machine,
        "storage_estimate": storage,
        "code_sha256": code_hash.hexdigest(),
        "components_sha256": file_sha256(comp_path),
        "terminals_sha256": file_sha256(term_path),
        "tests_note": "targeted pytest under studies/anchor_frame_audit_20261002/tests",
        "seed": C.SEED,
        "max_matches": max_matches,
        "scientific_limits": [
            "Observed-cache coverage only; cache zeros do not prove raw field absence.",
            "No V/SVI/direction labels generated; no V/q training or test scoring.",
            "No new final population or participation-radius choice authorized.",
            "Foreign near/distant is a documented distance diagnostic using radius 2000, not isolation semantics.",
            "Inclusive exact (F=tau) is incompatible with historical strict t_in=tau-1 eligibility.",
            "Absent/ambiguous GAME_END recorded as unknown; not substituted by last_frame.",
            "Endpoints are not authoritatively valid without terminal support; candidates retained.",
            "C0/C_all are recorded contributors, not proximity-based P0 or simultaneous participation.",
            "span>14000 / diameter>2000 summarize chaining under neighbor conditions, not detector errors.",
            "Primary requested lag is closed [10000,20000]; 25000/30000 are supplementary.",
            "Storage estimates use unique (match key, query time); V vs q dims may differ by snapshot age.",
            "q538/V180/1531 optimizer passes remain a planned budget, not timed here.",
            "No restart/resume support in this runner.",
        ],
    }
    if max_matches is not None:
        manifest["complete"] = False
        manifest["partial_reason"] = f"max_matches={max_matches}"
    man_hash = _write_json(run_dir / "RUN_MANIFEST.json", manifest)
    status.update({
        "complete": manifest["complete"],
        "finished_utc": _utc(),
        "manifest_sha256": man_hash,
        "gates_all_ok": gates.get("all_ok"),
        "n_components": agg["n_components"],
        "n_exact_10000": agg["exact_10000"]["n_components"],
        "wall_s": wall,
        "extraction_wall_s": extract_wall_s,
        "terminal_crosscheck_ok": gates.get("terminal_crosscheck_ok"),
        "reload_reconcile_ok": gates.get("reload_reconcile_ok"),
    })
    _write_json(status_path, status)
    log(
        f"finished complete={manifest['complete']} components={agg['n_components']} "
        f"exact={agg['exact_10000']['n_components']} extract_s={extract_wall_s:.1f}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Anchor frame audit 20261002")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    man = run(run_dir=args.run_dir, max_matches=args.max_matches, workers=args.workers)
    return 0 if man.get("complete") or args.max_matches is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
