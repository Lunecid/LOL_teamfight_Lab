"""Engagement population from kill events only (v4-exact stage 1, task B1).

detect_engagements_exact(pack, tm, params, mode) -> (records, diag)

Reads only pack['events'], pack['minute_ts'] and pack['meta'] (patch, match_id).  It never reads
'xy_raw_minute', 'node_minute' (no frame alive / status column) and never builds the 5-s position
grid (fight_clustering.build_5s_position_grid).  Alive status comes from kill events and the patch
respawn formula (gameplay.event_survival).  Kill positions (CHAMPION_KILL.position) are exact event
data and are used, in raw map units, for the spatial split, the anchor and the merge distance.

Pipeline shared by both modes (same steps and order as fights.detect_fights_teamfight_v2,
fights.py:1264-1414, with the presence gate replaced by the event-alive rule):
  1. kills = fights._extract_kill_events(events)                       (sorted by timestamp, stable)
  2. clusters = fight_clustering.cluster_kills_temporal(kills, G), then
     fight_clustering.split_kill_cluster_spatial(cluster, D); sorted by first kill      (G, D = params)
  3. per cluster: tau = max(t0, first kill - pre_kill_ms) (first kill - 1 if that is not before the
     first kill); guards in order: context (tau - context_ms < t0), start offset
     (tau - t0 < start_offset_ms), horizon (tau + horizon_ms > last frame ts); then
     alive: >= min_alive_per_team event-alive champions per team at tau; then the duration cap
     (last kill - tau > max_duration_ms).  The frame-alive check of fights.py:1386 is not applied.

mode 'v4' (plan section 2):
  * participants = kill credit (killer / victim / assisters) of the candidate's kills
    (gameplay.cohorts_exact); alive at tau from death_intervals(events, patch, t=tau) (events <= tau).
  * merge (A6 fix): candidates in tau order; b merges into the MOST RECENT earlier engagement a with
    b.tau - a.horizon_end <= merge_max_gap_ms, anchor distance <= merge_radius and
    max(a.last_kill, b.last_kill) - a.tau <= max_duration_ms (the unfilled window, not the
    horizon-filled one).  Kill sets are unioned and the participants recounted.
  * no minimum-gap, no ACE truncation, no overlap drop / replace: every merged engagement is kept.
  * flags: isolated = no kill outside its own kill set and no other engagement's tau in
    [tau, last_kill] (both ends inclusive); clean = frame age at tau < clean_max_age_ms
    (frame age = tau - last minute_ts <= tau; clean_max_age_ms <= 0 turns the filter off).

mode 'r2_repro': reproduces the R2 prototype
  outputs/reest_exact_v4_20260925/records/baselines/r2_pipeline.py, mode 'r2_kill' (gate r2, parts
  kill): the same candidates, the line-for-line merge copy of fights._merge_adjacent_candidates
  (fights.py:904-949, immediate predecessor only, span on the horizon-filled window) with the kill-set
  union, recount (det_cluster_blue/red, det_prox_pairs = nb * nr), then the v3.3 minimum-gap filter,
  ACE truncation (fights._truncate_fights_at_ace) and the legacy post-merge overlap resolution
  (fight_postmerge.enforce_postmerge_spacing_and_nonoverlap, priority 100 * nb * nr + 20 * segments
  + 10, location radius = D).  The R2 'present_blue/red' columns are position features computed
  from xy_raw_minute by the prototype's wrapper; they are not decisions and are not produced here
  (None).  Every other column of finals_r2_kill_full.tsv is reproduced.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gameplay import fight_clustering as FC
from gameplay import fights as F
from gameplay.cohorts_exact import cohort_from_counts, credited_participants, team_sets
from gameplay.event_survival import death_intervals, event_alive
from gameplay.fight_postmerge import enforce_postmerge_spacing_and_nonoverlap

MODES = ("v4", "r2_repro")

R2_DIAG_KEYS = ("accepted", "merged_candidates", "rejected_max_duration", "rejected_gap",
                "rejected_too_few_per_team", "rejected_startctx", "rejected_start_offset",
                "rejected_horizon", "rejected_alive", "postmerge_conflicts", "postmerge_removed",
                "postmerge_replaced", "postmerge_overlap_clipped", "postmerge_overlap_dropped",
                "ace_end_truncated")

# finals_r2_kill_full.tsv columns; present_blue / present_red are position features (not reproduced)
R2_TSV_COLUMNS = ("match_id", "first_kill_ts", "killer", "victim", "n_min", "engage_ts", "present_blue",
                  "present_red", "n_segments", "prox", "last_kill_ts", "parts", "context")
R2_COMPARE_COLUMNS = tuple(c for c in R2_TSV_COLUMNS if c not in ("present_blue", "present_red"))


# ------------------------------------------------------------------ parameters
@dataclass(frozen=True)
class ExactParams:
    """Detector constants.  gap_ms (G) and diameter (D) have no default on purpose."""
    gap_ms: int
    diameter: float
    pre_kill_ms: int = 15000
    min_alive_per_team: int = 2
    context_ms: int = 30000
    start_offset_ms: int = 120000
    horizon_ms: int = 35000
    max_duration_ms: int = 60000
    merge: bool = True
    merge_max_gap_ms: int = 15000
    merge_radius: float = 2000.0
    min_gap_ms: int = 0                                  # r2_repro only (v3.3 FIGHT_MIN_GAP_MS)
    tail_buffer_ms: int = 0
    clean_max_age_ms: int = 10000                        # v4 flag; <= 0 = off
    postmerge_location_radius: Optional[float] = None    # r2_repro only; None = diameter

    @classmethod
    def from_cfg(cls, c: Any, mode: Optional[str] = "v4", *, require_locked: bool = False) -> "ExactParams":
        """Read the constants from a CFG object (after apply_preset).

        mode: the detector mode the parameters are for.  The cfg's ENG_* definition switches must
        match what that mode hard-codes, else ValueError (a cfg / mode mismatch never runs silently):
          'v4'        ENG_ALIVE_SOURCE='event', ENG_PARTICIPATION='kill_credit', ENG_OVERLAP_RULE='none',
                      ENG_MERGE_A6=True, ENG_ISOLATION=True, and CONTINUOUS_FIGHT_MERGE=True
                      (preset 'v4-exact')
          'r2_repro'  ENG_ALIVE_SOURCE='event', ENG_PARTICIPATION='kill_credit',
                      ENG_OVERLAP_RULE='legacy_priority', ENG_MERGE_A6=False, ENG_ISOLATION=False
          None        no check (explicit opt-out, e.g. for sensitivity scans that set params by hand)
        require_locked: also require ENG_BOUNDARIES_LOCKED=True (G / D written after E1).
        """
        required = {
            "v4": {"ENG_ALIVE_SOURCE": "event", "ENG_PARTICIPATION": "kill_credit", "ENG_OVERLAP_RULE": "none",
                   "ENG_MERGE_A6": True, "ENG_ISOLATION": True, "CONTINUOUS_FIGHT_MERGE": True},
            "r2_repro": {"ENG_ALIVE_SOURCE": "event", "ENG_PARTICIPATION": "kill_credit",
                         "ENG_OVERLAP_RULE": "legacy_priority", "ENG_MERGE_A6": False, "ENG_ISOLATION": False},
        }
        if mode is not None:
            if mode not in required:
                raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(required)} or None")
            bad = []
            for key, want in required[mode].items():
                if not hasattr(c, key):
                    bad.append(f"{key}=<missing> (need {want!r})")
                    continue
                got = getattr(c, key)
                if isinstance(want, bool):
                    ok = isinstance(got, (bool, int, np.integer)) and bool(got) is want
                else:
                    ok = str(got).strip().lower() == want
                if not ok:
                    bad.append(f"{key}={got!r} (need {want!r})")
            if bad:
                raise ValueError(f"cfg is inconsistent with detector mode {mode!r}: " + "; ".join(bad))
        if require_locked and not bool(getattr(c, "ENG_BOUNDARIES_LOCKED", False)):
            raise ValueError("ENG_BOUNDARIES_LOCKED is False: G / D are not yet fixed by E1")
        ctx_sec = int(getattr(c, "FIGHT_CONTEXT_SEC", 30) or 0)
        ctx_ms = ctx_sec * 1000 if ctx_sec > 0 else int(getattr(c, "FIGHT_CONTEXT_MIN", 1)) * 60000
        if hasattr(c, "FIGHT_HORIZON_SEC"):
            horizon = int(getattr(c, "FIGHT_HORIZON_SEC")) * 1000
        else:
            horizon = int(getattr(c, "FIGHT_HORIZON_MIN", 1)) * 60000
        return cls(
            gap_ms=int(getattr(c, "TF2_KILL_CLUSTER_GAP_MS")),
            diameter=float(getattr(c, "CLUSTER_MAX_DIAMETER")),
            pre_kill_ms=int(getattr(c, "TF2_ENGAGE_PRE_KILL_MS", 15000)),
            min_alive_per_team=int(getattr(c, "TF2_MIN_PER_TEAM", 2)),
            context_ms=ctx_ms,
            start_offset_ms=int(getattr(c, "START_OFFSET_MIN", 2)) * 60000,
            horizon_ms=horizon,
            max_duration_ms=int(getattr(c, "MAX_MERGED_FIGHT_DURATION_MS", 60000)),
            merge=bool(getattr(c, "CONTINUOUS_FIGHT_MERGE", True)),
            merge_max_gap_ms=int(getattr(c, "CONTINUOUS_FIGHT_MAX_GAP_MS", 15000)),
            merge_radius=float(getattr(c, "CONTINUOUS_FIGHT_MERGE_RADIUS", 2000.0)),
            min_gap_ms=int(getattr(c, "FIGHT_MIN_GAP_MS", 0)),
            tail_buffer_ms=int(getattr(c, "TF2_TAIL_BUFFER_MS", 0)),
            clean_max_age_ms=int(getattr(c, "ENG_CLEAN_MAX_AGE_MS", 0) or 0),
        )


def _coerce_params(params: Any) -> ExactParams:
    if isinstance(params, ExactParams):
        return params
    if isinstance(params, Mapping):
        known = {f.name for f in fields(ExactParams)}
        bad = set(params) - known
        if bad:
            raise KeyError(f"unknown ExactParams fields: {sorted(bad)}")
        return ExactParams(**dict(params))
    raise TypeError("params must be ExactParams or a mapping with at least gap_ms and diameter")


def _patch_of(meta: Mapping) -> str:
    raw = str(meta.get("patch", "") or meta.get("patch_full", "") or "")
    nums = re.findall(r"\d+", raw)
    if len(nums) < 2:
        raise KeyError(f"pack meta has no usable patch: {raw!r}")
    return f"{int(nums[0])}.{int(nums[1])}"


# ------------------------------------------------------------------ minimal exact loader
DEFAULT_CACHE_DIR = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")


def load_exact_pack(match_id: str, cache_dir: Path = DEFAULT_CACHE_DIR,
                    allowed_patches: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """{'meta', 'events', 'minute_ts'} of one cached match; nothing else is read from disk.

    The npz is opened lazily and only its 'minute_ts' member is decompressed (no positions, no node
    features).  If allowed_patches is given and the meta patch is not in it, returns None before
    the events are read.
    """
    cache_dir = Path(cache_dir)
    mp, ep, npz = (cache_dir / f"{match_id}.meta.json", cache_dir / f"{match_id}.events.json",
                   cache_dir / f"{match_id}.npz")
    if not (mp.exists() and ep.exists() and npz.exists()):
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        return None
    meta = dict(meta)
    meta["patch"] = _patch_of(meta)
    if allowed_patches is not None and meta["patch"] not in set(allowed_patches):
        return None
    meta["team_map"] = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
    events = json.loads(ep.read_text(encoding="utf-8"))
    events = [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []
    with np.load(npz, allow_pickle=False) as z:
        minute_ts = np.asarray(z["minute_ts"]).astype(np.int64)
    return {"meta": meta, "events": events, "minute_ts": minute_ts}


# ------------------------------------------------------------------ shared candidate stage
def _new_diag() -> Dict[str, int]:
    d = {k: 0 for k in R2_DIAG_KEYS}
    d.update(clusters_total=0, clusters_after_spatial=0, kills=0)
    return d


def _candidates(pack: Mapping, tm: Mapping[int, int], p: ExactParams, mode: str, diag: Dict[str, int]):
    """Shared steps 1-3.  Returns (kills, candidates, minute_ts) or None when the match is too short."""
    meta = pack.get("meta") or {}
    patch = _patch_of(meta)
    minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    events = pack.get("events") or []
    if len(minute_ts) < 3:
        return None
    kills = F._extract_kill_events(events)
    diag["kills"] = len(kills)
    if not kills:
        return kills, [], minute_ts
    kidx = {id(k): i for i, k in enumerate(kills)}
    blue_set, red_set = team_sets(tm)
    b_idx = np.asarray(sorted(x - 1 for x in blue_set), dtype=int)
    r_idx = np.asarray(sorted(x - 1 for x in red_set), dtype=int)
    t_min, t_max = int(minute_ts[0]), int(minute_ts[-1])

    clusters = FC.cluster_kills_temporal(kills, int(p.gap_ms))
    diag["clusters_total"] = len(clusters)
    if float(p.diameter) > 0.0:
        split: List[dict] = []
        for cl in clusters:
            split.extend(FC.split_kill_cluster_spatial(cl, max_diameter=float(p.diameter)))
        clusters = sorted(split, key=lambda c: int(c.get("first_kill_ts", 0)))
    diag["clusters_after_spatial"] = len(clusters)

    iv_full = death_intervals(events, patch) if mode == "r2_repro" else None
    cands: List[dict] = []
    for cl in clusters:
        fk, lk = int(cl["first_kill_ts"]), int(cl["last_kill_ts"])
        tau = int(max(t_min, fk - int(p.pre_kill_ms)))
        if tau >= fk:
            tau = int(max(t_min, fk - 1))
        if tau - int(p.context_ms) < t_min:
            diag["rejected_startctx"] += 1
            continue
        if tau - t_min < int(p.start_offset_ms):
            diag["rejected_start_offset"] += 1
            continue
        if tau + int(p.horizon_ms) > t_max:
            diag["rejected_horizon"] += 1
            continue
        # alive at tau from kill events only; v4 builds the intervals from events <= tau
        iv = iv_full if iv_full is not None else death_intervals(events, patch, t=tau)
        alive = event_alive(iv, tau)
        ab, ar = int(alive[b_idx].sum()), int(alive[r_idx].sum())
        need = int(p.min_alive_per_team)
        if not (ab >= need and ar >= need):
            diag["rejected_too_few_per_team"] += 1
            continue
        fight_end = lk + int(p.tail_buffer_ms)
        horizon_end = F.label_window_end_ts(fight_end, tau, int(p.horizon_ms))
        if fight_end - tau > int(p.max_duration_ms):
            diag["rejected_max_duration"] += 1
            continue
        parts = set(cl["participants"])
        nb, nr = len(parts & blue_set), len(parts & red_set)
        cx, cy = float(cl["fight_center"][0]), float(cl["fight_center"][1])
        cands.append({
            "engage_ts": tau, "first_kill_ts": fk, "last_kill_ts": lk,
            "centroid_x": cx, "centroid_y": cy, "horizon_end_ts": int(horizon_end), "n_segments": 1,
            "det_prox_pairs": int(nb * nr), "det_anchor": 0, "det_backtracked": 1,
            "det_event_score": float(cl["n_kills"]), "det_event_count": int(cl["n_kills"]),
            "det_kill_count_window": int(cl["n_kills"]), "det_cluster_participants": len(parts),
            "det_cluster_blue": nb, "det_cluster_red": nr, "det_interaction_count": 0,
            "det_cluster_duration_ms": int(lk - fk),
            "_parts": parts, "_kill_idx": [kidx[id(k)] for k in cl["kills"]],
            "_alive": (ab, ar), "_fks": [fk],
        })
    diag["accepted"] = len(cands)
    return kills, cands, minute_ts


# ------------------------------------------------------------------ r2_repro
def _merge_r2(cands: List[dict], max_gap_ms: int, merge_radius: float, max_duration_ms: int,
              blue_set, red_set) -> List[dict]:
    """Line-for-line copy of the R2 prototype merge (r2_pipeline.make_merge, itself a copy of
    fights._merge_adjacent_candidates, fights.py:904-949) with the kill-set union and recount."""
    if not cands:
        return cands
    ordered = sorted(cands, key=lambda c: int(c.get("engage_ts", 0)))
    merged = [dict(ordered[0])]
    r2 = float(merge_radius) * float(merge_radius)
    for b in ordered[1:]:
        a = merged[-1]
        gap = int(b.get("engage_ts", 0)) - int(a.get("horizon_end_ts", 0))
        dx = float(b.get("centroid_x", 0.0)) - float(a.get("centroid_x", 0.0))
        dy = float(b.get("centroid_y", 0.0)) - float(a.get("centroid_y", 0.0))
        close = (dx * dx + dy * dy) <= r2
        new_end = max(int(a.get("horizon_end_ts", 0)), int(b.get("horizon_end_ts", 0)))
        span = new_end - int(a.get("engage_ts", 0))
        if gap <= int(max_gap_ms) and close and span <= int(max_duration_ms):
            a["first_kill_ts"] = min(int(a.get("first_kill_ts", 0)), int(b.get("first_kill_ts", 0)))
            a["last_kill_ts"] = max(int(a.get("last_kill_ts", 0)), int(b.get("last_kill_ts", 0)))
            a["horizon_end_ts"] = new_end
            a["n_segments"] = int(a.get("n_segments", 1)) + int(b.get("n_segments", 1))
            a["det_event_count"] = int(a.get("det_event_count", 0)) + int(b.get("det_event_count", 0))
            a["det_kill_count_window"] = int(a.get("det_kill_count_window", 0)) + int(b.get("det_kill_count_window", 0))
            a["det_event_score"] = float(a.get("det_event_score", 0.0)) + float(b.get("det_event_score", 0.0))
            a["det_interaction_count"] = int(a.get("det_interaction_count", 0)) + int(b.get("det_interaction_count", 0))
            a["det_prox_pairs"] = max(int(a.get("det_prox_pairs", 0)), int(b.get("det_prox_pairs", 0)))
            a["det_cluster_participants"] = max(int(a.get("det_cluster_participants", 0)), int(b.get("det_cluster_participants", 0)))
            a["det_cluster_duration_ms"] = int(a["last_kill_ts"]) - int(a["first_kill_ts"])
            a["_parts"] = a["_parts"] | b["_parts"]
            a["_kill_idx"] = a["_kill_idx"] + b["_kill_idx"]
            a["_fks"] = a["_fks"] + b["_fks"]
        else:
            merged.append(dict(b))
    for a in merged:  # parts = kill: recount from the union
        U = a["_parts"]
        nb, nr = len(U & blue_set), len(U & red_set)
        a["det_cluster_blue"], a["det_cluster_red"] = nb, nr
        a["det_cluster_participants"] = len(U)
        a["det_prox_pairs"] = nb * nr
    return merged


def _earliest_kill_key(events: Iterable[Mapping], fk: int) -> Optional[Tuple[int, int, int]]:
    """p3_detection_arms.earliest_kill_key: first CHAMPION_KILL in event order with timestamp fk."""
    for e in events or []:
        if e.get("type") == "CHAMPION_KILL" and int(e.get("timestamp", -1) or -1) == fk:
            return (fk, int(e.get("killerId", 0) or 0), int(e.get("victimId", 0) or 0))
    return None


def _detect_r2(pack: Mapping, tm: Mapping[int, int], p: ExactParams):
    diag = _new_diag()
    got = _candidates(pack, tm, p, "r2_repro", diag)
    if got is None:
        return [], diag
    kills, cands, minute_ts = got
    if not cands:
        return [], diag
    events = pack.get("events") or []
    blue_set, red_set = team_sets(tm)
    if p.merge:
        n0 = len(cands)
        cands = _merge_r2(cands, p.merge_max_gap_ms, p.merge_radius, p.max_duration_ms, blue_set, red_set)
        diag["merged_candidates"] = n0 - len(cands)
    cands.sort(key=lambda f: int(f["engage_ts"]))
    fights: List[dict] = []
    last_ts = -(10 ** 18)
    for f in cands:
        if int(f["engage_ts"]) - int(last_ts) < int(p.min_gap_ms):
            diag["rejected_gap"] += 1
            continue
        fights.append(f)
        last_ts = int(f["engage_ts"])
    kill_ts = np.array([int(k["timestamp"]) for k in kills], dtype=np.int64)
    F._truncate_fights_at_ace(fights, F._extract_ace_ts(events), horizon_ms=int(p.horizon_ms), diag=diag)
    loc = float(p.diameter if p.postmerge_location_radius is None else p.postmerge_location_radius)
    fights = enforce_postmerge_spacing_and_nonoverlap(
        fights, horizon_ms=int(p.horizon_ms), fight_min_gap_ms=int(p.min_gap_ms), kill_ts=kill_ts,
        location_radius=float(max(0.0, loc)), diag=diag)
    anchors = F.build_anchors_from_events(events)
    mid = str((pack.get("meta") or {}).get("match_id", ""))
    out = []
    for f in fights:
        try:
            ctx = F.classify_fight_context(f, anchors, False, 1.0)
        except Exception:  # the detector logs and leaves the field unset
            ctx = ""
        key = _earliest_kill_key(events, int(f.get("first_kill_ts") or -1))
        if key is None:
            continue
        cb, cr = int(f["det_cluster_blue"]), int(f["det_cluster_red"])
        rec = {
            "match_id": mid, "first_kill_ts": key[0], "killer": key[1], "victim": key[2],
            "n_min": min(cb, cr), "engage_ts": int(f["engage_ts"]), "present_blue": None, "present_red": None,
            "n_segments": int(f.get("n_segments", 1)), "prox": int(f.get("det_prox_pairs", 0)),
            "last_kill_ts": int(f["last_kill_ts"]), "parts": sorted(f["_parts"]), "context": str(ctx),
            "horizon_end_ts": int(f["horizon_end_ts"]), "kill_idx": sorted(f["_kill_idx"]),
            "fks": list(f["_fks"]), "centroid_x": float(f["centroid_x"]), "centroid_y": float(f["centroid_y"]),
        }
        rec.update(cohort_from_counts(cb, cr))
        out.append(rec)
    diag["finals"] = len(out)
    return out, diag


def r2_tsv_row(rec: Mapping, columns: Sequence[str] = R2_COMPARE_COLUMNS) -> Tuple[str, ...]:
    """A record as the string fields of finals_r2_kill_full.tsv (default: all but present_*)."""
    out = []
    for c in columns:
        v = rec[c]
        out.append(",".join(map(str, v)) if c == "parts" else str(v))
    return tuple(out)


# ------------------------------------------------------------------ v4
def _merge_v4(cands: List[dict], p: ExactParams) -> List[dict]:
    """A6 merge: each candidate (tau order) joins the most recent earlier engagement that satisfies
    gap (b.tau - a.horizon_end <= merge_max_gap_ms), place (anchor distance <= merge_radius) and the
    unfilled-window cap (max(a.last_kill, b.last_kill) - a.tau <= max_duration_ms)."""
    ordered = sorted(cands, key=lambda c: int(c["engage_ts"]))
    r2 = float(p.merge_radius) ** 2
    merged: List[dict] = []
    for b in ordered:
        target = None
        for a in reversed(merged):
            gap = int(b["engage_ts"]) - int(a["horizon_end_ts"])
            dx = float(b["centroid_x"]) - float(a["centroid_x"])
            dy = float(b["centroid_y"]) - float(a["centroid_y"])
            span = max(int(a["last_kill_ts"]), int(b["last_kill_ts"])) - int(a["engage_ts"])
            if gap <= int(p.merge_max_gap_ms) and (dx * dx + dy * dy) <= r2 and span <= int(p.max_duration_ms):
                target = a
                break
        if target is None:
            merged.append(dict(b))
            continue
        a = target
        a["first_kill_ts"] = min(int(a["first_kill_ts"]), int(b["first_kill_ts"]))
        a["last_kill_ts"] = max(int(a["last_kill_ts"]), int(b["last_kill_ts"]))
        a["horizon_end_ts"] = max(int(a["horizon_end_ts"]), int(b["horizon_end_ts"]))
        a["n_segments"] = int(a["n_segments"]) + int(b["n_segments"])
        a["_parts"] = a["_parts"] | b["_parts"]
        a["_kill_idx"] = sorted(set(a["_kill_idx"]) | set(b["_kill_idx"]))
        a["_fks"] = a["_fks"] + b["_fks"]
    return merged


def frame_age_ms(minute_ts: np.ndarray, t: int) -> Optional[int]:
    """t - (last minute frame timestamp <= t); None if no frame is at or before t."""
    j = int(np.searchsorted(np.asarray(minute_ts, dtype=np.int64), int(t), side="right")) - 1
    if j < 0:
        return None
    return int(t) - int(minute_ts[j])


def isolation_counts(tau: int, last_kill: int, own_kill_idx: Iterable[int], kill_ts: np.ndarray,
                     other_taus: Iterable[int]) -> Tuple[int, int]:
    """(foreign kills, other engagements' tau) inside [tau, last_kill], both ends inclusive."""
    own = set(int(i) for i in own_kill_idx)
    lo = int(np.searchsorted(kill_ts, int(tau), side="left"))
    hi = int(np.searchsorted(kill_ts, int(last_kill), side="right"))
    n_foreign = sum(1 for i in range(lo, hi) if i not in own)
    n_tau = sum(1 for t2 in other_taus if int(tau) <= int(t2) <= int(last_kill))
    return n_foreign, n_tau


def _detect_v4(pack: Mapping, tm: Mapping[int, int], p: ExactParams):
    diag = _new_diag()
    diag.update(finals=0, isolated=0, clean=0, clean_isolated=0)
    got = _candidates(pack, tm, p, "v4", diag)
    if got is None:
        return [], diag
    kills, cands, minute_ts = got
    if not cands:
        return [], diag
    blue_set, red_set = team_sets(tm)
    n0 = len(cands)
    if p.merge:
        cands = _merge_v4(cands, p)
    diag["merged_candidates"] = n0 - len(cands)
    cands.sort(key=lambda f: int(f["engage_ts"]))
    kill_ts = np.array([int(k["timestamp"]) for k in kills], dtype=np.int64)
    taus = [int(c["engage_ts"]) for c in cands]
    mid = str((pack.get("meta") or {}).get("match_id", ""))
    out = []
    for i, c in enumerate(cands):
        tau, lk = int(c["engage_ts"]), int(c["last_kill_ts"])
        kidx = sorted(c["_kill_idx"])
        own_kills = [kills[j] for j in kidx]
        U = credited_participants(own_kills)
        assert U == c["_parts"], "kill-credit union mismatch"
        nb, nr = len(U & blue_set), len(U & red_set)
        n_foreign, n_tau = isolation_counts(tau, lk, kidx, kill_ts, taus[:i] + taus[i + 1:])
        age = frame_age_ms(minute_ts, tau)
        clean = bool(age is not None and (int(p.clean_max_age_ms) <= 0 or age < int(p.clean_max_age_ms)))
        k0 = own_kills[0]
        rec = {
            "match_id": mid, "engage_ts": tau, "tau": tau,
            "first_kill_ts": int(c["first_kill_ts"]), "last_kill_ts": lk,
            "killer": int(k0.get("killer_id", 0) or 0), "victim": int(k0.get("victim_id", 0) or 0),
            "centroid_x": float(c["centroid_x"]), "centroid_y": float(c["centroid_y"]),
            "horizon_end_ts": int(c["horizon_end_ts"]), "n_segments": int(c["n_segments"]),
            "n_kills": len(kidx), "kill_idx": kidx, "kill_ts": [int(kills[j]["timestamp"]) for j in kidx],
            "fks": list(c["_fks"]), "parts": sorted(U), "blue_parts": sorted(U & blue_set),
            "red_parts": sorted(U & red_set),
            "alive_blue": int(c["_alive"][0]), "alive_red": int(c["_alive"][1]),
            "frame_age_ms": age, "clean": int(clean),
            "n_foreign_kills": int(n_foreign), "n_other_tau": int(n_tau),
            "isolated": int(n_foreign == 0 and n_tau == 0),
        }
        rec.update(cohort_from_counts(nb, nr))
        out.append(rec)
        diag["isolated"] += rec["isolated"]
        diag["clean"] += rec["clean"]
        diag["clean_isolated"] += int(rec["isolated"] and rec["clean"])
    diag["finals"] = len(out)
    return out, diag


# ------------------------------------------------------------------ entry point
def detect_engagements_exact(pack: Mapping, tm: Mapping[int, int], params: Any,
                             mode: str = "v4") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Engagements of one match from kill events.  Returns (records sorted by tau, diagnostics).

    pack: mapping with 'events', 'minute_ts' and 'meta' (patch, match_id); no other key is read.
    tm: {participantId: teamId (100 / 200)}.
    params: ExactParams or a mapping of its fields (gap_ms and diameter are required).
    mode: 'v4' or 'r2_repro' (see the module docstring).
    """
    p = _coerce_params(params)
    tm = {int(k): int(v) for k, v in (tm or {}).items()}
    if mode == "v4":
        return _detect_v4(pack, tm, p)
    if mode == "r2_repro":
        return _detect_r2(pack, tm, p)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
