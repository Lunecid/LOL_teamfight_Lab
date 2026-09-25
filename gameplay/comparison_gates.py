"""Comparison-only presence gates for the B6 comparison rows (v4-exact plan, stage 1 table).

This is the ONLY module allowed to build the legacy 5-s position grid (gameplay.grid_guard.allow_grid);
the v4 population, StateV3 and setup features never read it.  The two gates are ported from
outputs/diag_survival_dbscan_20260925/p4_gate_decomp/gate_decomp.py (and p3_detection_arms.py), with
the module-level state `_S` replaced by an explicit GateContext:

  G1  the 'evr' gate (post-hoc frame gate): respawn-aware 5-s grid (frame interpolation + kill
      overwrite, then every respawn segment re-drawn from the own spawn point to the next
      observation), read at the last grid point <= tau; event alive.  Pass iff >= min_per_team
      event-alive champions of each team lie within R of the anchor AND each team has >= min_per_team
      event-alive champions.  NOTE: the grid interpolates towards the NEXT minute frame and kill
      positions after tau, so G1 is not causal; it is a comparison row only.
  G5  the tau-state gate: event alive; position = latest observation <= tau among (last minute
      frame <= tau, own kill <= tau: killer / victim / assister of a CHAMPION_KILL with a position);
      a champion that respawned after that observation and at or before tau is placed at its own
      spawn point (frame 0 position).  Pass iff >= min_per_team alive champions of each team lie
      within R of the anchor.  Uses data <= tau only (kill positions and frames), but still reads
      xy_raw_minute, so it is also a comparison row only.

R = TF2_VALIDITY_RADIUS (v3.3: 1,600); anchor = the candidate's fight_center (the kill-cluster
centre); both divided by the coordinate scale when the frames are normalised, as in
fights._validate_teamfight_at_engage.  Death intervals come from gameplay.event_survival over the
match's events (identical to p3 death_intervals; the respawn segments of the G1 grid use the ends of
those intervals, and event_alive at tau is the same with all events or events <= tau).

legacy_gate_candidates(pack, tm, cfg_obj) enumerates the presence-gate calls of the v3.3
teamfight_v2 detector in its 'evr' arm (REQUIRE_ALIVE_PER_TEAM = 0: no frame-alive pre-check):
temporal clusters (G), greedy diameter split (D), engage tau and the context / start-offset /
horizon guards, in detector order.  gates_for_records applies the gates to any (tau, anchor) rows,
e.g. gameplay.exact_population records (centroid_x / centroid_y).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gameplay import fights as F
from gameplay.event_survival import death_intervals, event_alive
from gameplay.grid_guard import allow_grid

GATES = ("G1", "G5")
DEFAULT_MIN_PER_TEAM = 2


def _patch_of(meta: Mapping) -> str:
    import re
    raw = str((meta or {}).get("patch", "") or (meta or {}).get("patch_full", "") or "")
    nums = re.findall(r"\d+", raw)
    if len(nums) < 2:
        raise KeyError(f"pack meta has no usable patch: {raw!r}")
    return f"{int(nums[0])}.{int(nums[1])}"


def _kill_ids(k: Mapping) -> set:
    ids = {int(k.get("killer_id", 0) or 0), int(k.get("victim_id", 0) or 0)}
    ids |= {int(a) for a in (k.get("assisting_ids") or [])}
    return {p for p in ids if 1 <= p <= 10}


@dataclass
class GateContext:
    """Per-match inputs of the comparison gates (see gate_context)."""
    match_id: str
    patch: str
    tm: Dict[int, int]
    b: np.ndarray                      # blue player indices (0-based)
    r: np.ndarray
    xy: np.ndarray                     # xy_raw_minute (T, 10, >= 2)
    minute_ts: np.ndarray
    kill_events: List[dict]            # fights._extract_kill_events(events)
    intervals: Dict[int, list]         # event_survival.death_intervals(events, patch)
    is_norm: bool
    scale_factor: float
    validity_radius: float
    min_per_team: int = DEFAULT_MIN_PER_TEAM
    grid_step_ms: int = 5000
    obs: Dict[int, List[Tuple[int, float, float, int]]] = field(default_factory=dict)
    _grid: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def div(self) -> float:
        return float(self.scale_factor) if (self.is_norm and self.scale_factor > 0) else 1.0


def gate_context(pack: Mapping, tm: Optional[Mapping] = None, cfg_obj: Any = None, *,
                 validity_radius: Optional[float] = None, min_per_team: int = DEFAULT_MIN_PER_TEAM) -> GateContext:
    """GateContext of one match.  pack needs 'xy_raw_minute', 'minute_ts', 'events', 'meta'."""
    if cfg_obj is None:
        from core.config import cfg as cfg_obj  # noqa: N813
    meta = pack.get("meta") or {}
    tm = {int(k): int(v) for k, v in (tm if tm is not None else meta.get("team_map") or {}).items()}
    b, r = F.validate_team_mapping(tm)
    xy = np.asarray(pack["xy_raw_minute"])
    minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    events = pack.get("events") or []
    patch = _patch_of(meta)
    is_norm = bool(meta.get("anchor_is_norm", False))
    scale = float(getattr(cfg_obj, "COORD_NORM_DIV", 16000.0))
    if not is_norm:
        is_norm, scale = F.detect_coordinate_scale(xy)
    kill_events = F._extract_kill_events(events)
    obs: Dict[int, list] = {p: [] for p in range(1, 11)}
    for ki, k in enumerate(kill_events):
        pos = k.get("position")
        if pos is None:
            continue
        for p in _kill_ids(k):
            obs[p].append((int(k["timestamp"]), float(pos[0]), float(pos[1]), ki))
    for p in obs:
        obs[p].sort()
    R = float(validity_radius if validity_radius is not None else getattr(cfg_obj, "TF2_VALIDITY_RADIUS", 1800.0))
    step = int(getattr(cfg_obj, "TF2_GRID_STEP_MS", 5000) or 5000)
    return GateContext(match_id=str(meta.get("match_id", "")), patch=patch, tm=tm, b=np.asarray(b), r=np.asarray(r),
                       xy=xy, minute_ts=minute_ts, kill_events=kill_events,
                       intervals=death_intervals(events, patch), is_norm=bool(is_norm), scale_factor=float(scale),
                       validity_radius=R, min_per_team=int(min_per_team), grid_step_ms=step, obs=obs)


# ------------------------------------------------------------------ G1 grid (p3 respawn_aware_grid)
def respawn_aware_grid(ctx: GateContext) -> Tuple[np.ndarray, np.ndarray]:
    """(dense_ts, xy_dense): the legacy 5-s grid with frame interpolation and kill overwrite, then
    every respawn segment re-drawn: the champion is at its spawn point (frame-0 position) at the
    respawn time and moves in a straight line to its next observation (next minute frame or next
    kill it takes part in, whichever is first).  Cached on the context."""
    if ctx._grid is not None:
        return ctx._grid
    grid_cfg = SimpleNamespace(TF2_GRID_STEP_MS=int(ctx.grid_step_ms), TF2_USE_FRAME_INTERP=True,
                               TF2_USE_KILL_TRAJECTORY_INTERP=True)
    with allow_grid():
        dense_ts, xy = F._build_5s_position_grid(xy_minute=ctx.xy, minute_ts=ctx.minute_ts,
                                                 kill_events=ctx.kill_events, tm=ctx.tm, cfg_obj=grid_cfg)
    mt = ctx.minute_ts
    spawn = np.asarray(ctx.xy[0, :, :2], dtype=np.float64)
    obs = {p: sorted((o[0], o[1], o[2]) for o in ctx.obs.get(p, [])) for p in range(1, 11)}
    for p, iv in ctx.intervals.items():
        ko = obs[p]
        for _t0, t1 in iv:
            t1 = int(np.ceil(t1))
            jf = int(np.searchsorted(mt, t1, side="right"))
            cand = []
            if jf < len(mt):
                cand.append((int(mt[jf]), float(ctx.xy[jf, p - 1, 0]), float(ctx.xy[jf, p - 1, 1])))
            nk = [o for o in ko if o[0] > t1]
            if nk:
                cand.append(nk[0])
            if not cand:
                continue
            tn, xn, yn = min(cand)
            a = int(np.searchsorted(dense_ts, t1, side="left"))
            bb = int(np.searchsorted(dense_ts, tn, side="left"))
            if bb <= a:
                continue
            w = ((dense_ts[a:bb] - t1) / max(tn - t1, 1)).astype(np.float64)
            xy[a:bb, p - 1, 0] = spawn[p - 1, 0] + w * (xn - spawn[p - 1, 0])
            xy[a:bb, p - 1, 1] = spawn[p - 1, 1] + w * (yn - spawn[p - 1, 1])
    ctx._grid = (dense_ts, xy)
    return ctx._grid


# ------------------------------------------------------------------ counting
def _count(pos: np.ndarray, ctr: Tuple[float, float], R2: float, team: np.ndarray, alive: np.ndarray) -> int:
    n = 0
    for i in team:
        i = int(i)
        if float(alive[i]) <= 0.5:
            continue
        dx = float(pos[i, 0]) - ctr[0]
        dy = float(pos[i, 1]) - ctr[1]
        n += int(dx * dx + dy * dy <= R2)
    return n


def _frame(ctx: GateContext, tau: int) -> Tuple[int, int]:
    j = int(np.searchsorted(ctx.minute_ts, int(tau), side="right")) - 1
    if j < 0:
        raise ValueError(f"no minute frame at or before tau={tau}")
    return j, int(ctx.minute_ts[j])


def _geom(ctx: GateContext, center: Sequence[float]) -> Tuple[Tuple[float, float], float]:
    div = ctx.div
    R = float(ctx.validity_radius) / div
    return (float(center[0]) / div, float(center[1]) / div), R * R


def gate_g1(ctx: GateContext, tau: int, center: Sequence[float]) -> Dict[str, int]:
    """G1 ('evr'): {'pass', 'nb', 'nr'} (in-radius event-alive counts on the respawn-aware grid)."""
    dense_ts, xyd = respawn_aware_grid(ctx)
    d = int(np.clip(np.searchsorted(dense_ts, int(tau), side="right") - 1, 0, len(dense_ts) - 1))
    ctr, R2 = _geom(ctx, center)
    ev = event_alive(ctx.intervals, int(tau))
    nb = _count(xyd[d], ctr, R2, ctx.b, ev)
    nr = _count(xyd[d], ctr, R2, ctx.r, ev)
    m = int(ctx.min_per_team)
    need_ok = ev[ctx.b].sum() >= m and ev[ctx.r].sum() >= m
    return {"pass": int(nb >= m and nr >= m and need_ok), "nb": nb, "nr": nr}


def tau_state_positions(ctx: GateContext, tau: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pos (10, 2), from_own_kill (10,), from_spawn (10,)): the G5 position of every champion at tau."""
    j, tf = _frame(ctx, tau)
    div = ctx.div
    pos = np.zeros((10, 2))
    src_own = np.zeros(10, dtype=int)
    src_spawn = np.zeros(10, dtype=int)
    for p in range(1, 11):
        t_obs, x, y = tf, float(ctx.xy[j, p - 1, 0]), float(ctx.xy[j, p - 1, 1])
        ko = [o for o in ctx.obs.get(p, []) if o[0] <= tau]
        if ko and ko[-1][0] >= t_obs:
            t_obs, x, y = ko[-1][0], ko[-1][1] / div, ko[-1][2] / div
            src_own[p - 1] = 1
        if any(t_obs < t1 <= tau for _t0, t1 in ctx.intervals.get(p, [])):
            x, y = float(ctx.xy[0, p - 1, 0]), float(ctx.xy[0, p - 1, 1])
            src_spawn[p - 1] = 1
            src_own[p - 1] = 0
        pos[p - 1] = (x, y)
    return pos, src_own, src_spawn


def gate_g5(ctx: GateContext, tau: int, center: Sequence[float]) -> Dict[str, int]:
    """G5 (tau-state): {'pass', 'nb', 'nr'} from the latest observation <= tau with spawn reset."""
    pos, _, _ = tau_state_positions(ctx, int(tau))
    ctr, R2 = _geom(ctx, center)
    ev = event_alive(ctx.intervals, int(tau))
    nb = _count(pos, ctr, R2, ctx.b, ev)
    nr = _count(pos, ctr, R2, ctx.r, ev)
    m = int(ctx.min_per_team)
    return {"pass": int(nb >= m and nr >= m), "nb": nb, "nr": nr}


def gate_decisions(ctx: GateContext, tau: int, center: Sequence[float]) -> Dict[str, Any]:
    """Both gates for one (tau, anchor): {'G1_pass', 'G1_nb', 'G1_nr', 'G5_pass', 'G5_nb', 'G5_nr'}."""
    out: Dict[str, Any] = {}
    for name, fn in (("G1", gate_g1), ("G5", gate_g5)):
        g = fn(ctx, tau, center)
        out.update({f"{name}_pass": g["pass"], f"{name}_nb": g["nb"], f"{name}_nr": g["nr"]})
    return out


def gates_for_records(ctx: GateContext, records: Iterable[Mapping]) -> List[Dict[str, Any]]:
    """Gate decisions for rows with 'tau' (or 'engage_ts') and an anchor ('fight_center' or
    'centroid_x' / 'centroid_y'), e.g. exact_population records.  Returns one dict per row."""
    out = []
    for rec in records:
        tau = int(rec["tau"] if rec.get("tau") is not None else rec["engage_ts"])
        c = rec.get("fight_center")
        if c is None:
            c = (float(rec["centroid_x"]), float(rec["centroid_y"]))
        out.append(dict(gate_decisions(ctx, tau, c), tau=tau))
    return out


# ------------------------------------------------------------------ legacy (v3.3 'evr') gate calls
def legacy_gate_candidates(pack: Mapping, tm: Mapping, cfg_obj: Any = None) -> List[Dict[str, Any]]:
    """The presence-gate calls of fights.detect_fights_teamfight_v2 in the p3 'evr' arm
    (REQUIRE_ALIVE_PER_TEAM = 0), in detector order: [{'tau', 'first_kill_ts', 'last_kill_ts',
    'fight_center', 'cluster'}].  Reads events and minute_ts only (no grid, no positions)."""
    if cfg_obj is None:
        from core.config import cfg as cfg_obj  # noqa: N813
    config = F.FightDetectorConfig.from_cfg(cfg_obj)
    minute_ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    if len(minute_ts) < 3:
        return []
    kill_events = F._extract_kill_events(pack.get("events") or [])
    if not kill_events:
        return []
    gap = int(getattr(cfg_obj, "TF2_KILL_CLUSTER_GAP_MS", 18000))
    pre = int(getattr(cfg_obj, "TF2_ENGAGE_PRE_KILL_MS", 10000))
    if hasattr(cfg_obj, "FIGHT_HORIZON_SEC"):
        horizon_ms = int(getattr(cfg_obj, "FIGHT_HORIZON_SEC", 30)) * 1000
    else:
        horizon_ms = int(getattr(cfg_obj, "FIGHT_HORIZON_MIN", 1)) * 60000
    ctx_ms = (int(config.fight_context_sec) * 1000 if int(config.fight_context_sec) > 0
              else int(config.fight_context_min) * 60000)
    start_offset_ms = int(getattr(cfg_obj, "START_OFFSET_MIN", 2)) * 60000
    t_min, t_max = int(minute_ts[0]), int(minute_ts[-1])
    clusters = F._cluster_kills_temporal(kill_events, gap)
    diam = float(config.cluster_max_diameter or 0.0)
    if diam > 0.0:
        split: List[dict] = []
        for cl in clusters:
            split.extend(F._split_kill_cluster_spatial(cl, max_diameter=diam))
        clusters = sorted(split, key=lambda c: int(c.get("first_kill_ts", 0)))
    out = []
    for cl in clusters:
        fk, lk = int(cl["first_kill_ts"]), int(cl["last_kill_ts"])
        tau = int(max(t_min, fk - pre))
        if tau >= fk:
            tau = int(max(t_min, fk - 1))
        if tau - ctx_ms < t_min or tau - t_min < start_offset_ms or tau + horizon_ms > t_max:
            continue
        out.append({"tau": tau, "first_kill_ts": fk, "last_kill_ts": lk,
                    "fight_center": (float(cl["fight_center"][0]), float(cl["fight_center"][1])), "cluster": cl})
    return out
