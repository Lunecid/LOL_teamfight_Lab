"""Anchor-free positional 'engagement setup' features at an exact ms t (v4-exact plan, section 3).

For q only, and only for clean engagements (frame age < 10 s): the positions come from the last
minute frame with timestamp <= t, which the caller has checked is recent.  There is deliberately
no anchor / first-kill position argument: distances or bearings to the first kill would point at
the first victim and leak the outcome.  ``build_setup(pack, t, roles)`` takes nothing else.

Positions: node_minute x_norm / y_norm * COORD_NORM_DIV (gameplay.role_inference.frame_positions;
xy_raw_minute only when node_minute is absent) of the frame <= t.  Players dead at t by kill-event
survival (gameplay.event_survival, events <= t) are masked (present = 0, position and zone 0, tower
distances FAR_NORM, excluded from every aggregate).  A player alive at t whose death interval ended
after the frame (respawned since the frame) is placed at his team's fountain (at_fountain = 1).
Towers standing at t: gameplay.anchors.standing_towers from BUILDING_KILL events <= t, a nexus
turret counting as standing again once its respawn time (objective_rules nexus_turret_respawn_ms)
has passed.

Features (``SETUP_COLUMNS``, name-order hash ``SETUP_NAME_HASH``); coordinates and distances are
divided by COORD_NORM_DIV (16 000):
  per player, '<side>_<role>_' in role-slot order (blue TOP..UTILITY, red TOP..UTILITY):
    x, y, present, at_fountain,
    zone_own_base, zone_enemy_base, zone_lane, zone_river, zone_own_jungle, zone_enemy_jungle
      (exclusive, in this priority: base, lane, river, jungle; all 0 when masked),
    near_dragon, near_baron (within PIT_RADIUS = 3 000 of the pit; not exclusive),
    dist_own_tower, dist_enemy_tower (nearest standing tower; FAR_NORM if none / masked)
  per team '<side>_': n_present, n_fountain, centroid_x, centroid_y (team fountain if nobody is
    present), spread (RMS distance to the centroid), largest_group (max over present players of the
    number of present teammates within GROUP_RADIUS = 2 000, self included)
  between teams: pair_min_dist, pair_mean_dist (FAR_NORM if no present pair), pairs_within_1600,
    pairs_within_3000, n_pairs
Every value is finite.

Zones (game units; blue base is bottom-left):
  base    x <= 4 700 and y <= 4 700 (blue) / x >= 10 100 and y >= 10 100 (red), past the base turrets
  lane    distance to a lane polyline (role_inference.lane_polylines) <= LANE_HALF_WIDTH (1 000)
  river   perpendicular distance to the river line x + y = 14 800 <= RIVER_HALF_WIDTH (1 000)
  jungle  the rest; blue side is x + y < 14 800
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from gameplay.anchors import load_static_anchors, standing_towers
from gameplay.event_survival import death_intervals, event_alive, respawned_since
from gameplay.role_inference import ROLES, frame_positions, lane_polylines, point_polyline_distance

TEAMS = (100, 200)
SIDE = {100: "blue", 200: "red"}
SLOT_PREFIXES: Tuple[str, ...] = tuple(f"{SIDE[t]}_{r.lower()}_" for t in TEAMS for r in ROLES)

COORD_NORM_DIV = 16000.0            # == core.config cfg.COORD_NORM_DIV (MAP_MAX); checked in the tests
FOUNTAIN = {100: (400.0, 420.0), 200: (14340.0, 14390.0)}   # fountain platforms (approximate)
BASE_EDGE = {100: 4700.0, 200: 10100.0}
RIVER_SUM = 14800.0
RIVER_HALF_WIDTH = 1000.0
LANE_HALF_WIDTH = 1000.0
PIT_RADIUS = 3000.0
GROUP_RADIUS = 2000.0
PAIR_RADII = (1600.0, 3000.0)
FAR_NORM = math.hypot(15000.0, 15000.0) / COORD_NORM_DIV   # larger than any distance on the map

PLAYER_SETUP_FIELDS: Tuple[str, ...] = (
    "x", "y", "present", "at_fountain",
    "zone_own_base", "zone_enemy_base", "zone_lane", "zone_river", "zone_own_jungle", "zone_enemy_jungle",
    "near_dragon", "near_baron", "dist_own_tower", "dist_enemy_tower",
)
TEAM_SETUP_FIELDS: Tuple[str, ...] = ("n_present", "n_fountain", "centroid_x", "centroid_y", "spread",
                                      "largest_group")
PAIR_SETUP_FIELDS: Tuple[str, ...] = ("pair_min_dist", "pair_mean_dist", "pairs_within_1600",
                                      "pairs_within_3000", "n_pairs")
SETUP_COLUMNS: Tuple[str, ...] = (
    tuple(pre + f for pre in SLOT_PREFIXES for f in PLAYER_SETUP_FIELDS)
    + tuple(f"{SIDE[t]}_{f}" for t in TEAMS for f in TEAM_SETUP_FIELDS)
    + PAIR_SETUP_FIELDS
)
SETUP_NAME_HASH = "5595d565c2780e6930494a75f58c91db1645119c5dd48da379a2d870f0436c57"

Roles = Union[Mapping[int, int], Sequence[int], object]


def _slot_by_pid(roles: Roles) -> Dict[int, int]:
    """Accepts a RoleAssignment (.slot_by_pid), {pid: slot} or a pid_by_slot list of 10."""
    sbp = getattr(roles, "slot_by_pid", None)
    if sbp is not None:
        out = {int(p): int(s) for p, s in sbp.items()}
    elif isinstance(roles, Mapping):
        out = {int(p): int(s) for p, s in roles.items()}
    else:
        seq = [int(p) for p in roles]
        out = {p: s for s, p in enumerate(seq)}
    if sorted(out) != list(range(1, 11)) or sorted(out.values()) != list(range(10)):
        raise ValueError(f"roles must map participants 1..10 onto slots 0..9, got {out}")
    return out


def _zone(x: float, y: float, team: int, polys: Mapping[str, np.ndarray]) -> str:
    if x <= BASE_EDGE[100] and y <= BASE_EDGE[100]:
        return "zone_own_base" if team == 100 else "zone_enemy_base"
    if x >= BASE_EDGE[200] and y >= BASE_EDGE[200]:
        return "zone_own_base" if team == 200 else "zone_enemy_base"
    pt = np.array([[x, y]])
    if min(float(point_polyline_distance(pt, polys[k])[0]) for k in ("TOP", "MIDDLE", "BOTTOM")) <= LANE_HALF_WIDTH:
        return "zone_lane"
    if abs(x + y - RIVER_SUM) / math.sqrt(2.0) <= RIVER_HALF_WIDTH:
        return "zone_river"
    blue_side = (x + y) < RIVER_SUM
    return "zone_own_jungle" if blue_side == (team == 100) else "zone_enemy_jungle"


def _standing_towers_at(events: Sequence[Mapping], t: int, patch: str) -> Dict[int, np.ndarray]:
    """{team: (k, 2) positions of that team's towers standing at t}; nexus turrets respawn."""
    try:
        from gameplay.objective_timers import rule
        nexus_resp = float(rule("nexus_turret_respawn_ms", patch).value)
    except KeyError:
        nexus_resp = float("inf")
    kept = []
    for e in events:
        if int(e.get("timestamp", 0)) > t:
            continue
        if (e.get("type") == "BUILDING_KILL" and str(e.get("towerType", "")).upper() == "NEXUS_TURRET"
                and int(e["timestamp"]) + nexus_resp <= t):
            continue  # respawned by t
        kept.append(e)
    st = standing_towers(kept, t, load_static_anchors())
    return {tm: np.asarray(st[f"TOWER_T{tm}"], dtype=float).reshape(-1, 2) for tm in TEAMS}


def _nearest(p: np.ndarray, pts: np.ndarray) -> float:
    if not len(pts):
        return FAR_NORM
    return float(np.sqrt(((pts - p) ** 2).sum(1)).min()) / COORD_NORM_DIV


def build_setup(pack: Mapping, t: int, roles: Roles) -> Dict[str, float]:
    """SETUP_COLUMNS at ms t from the last frame <= t and events <= t (see module docstring).

    pack: cache pack ('minute_ts', 'node_minute' (or 'xy_raw_minute'), 'events', 'meta' with
    'team_map' and 'patch').  roles: RoleAssignment, {pid: slot} or pid_by_slot.
    """
    q = int(t)
    meta = pack.get("meta") or {}
    tm = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
    if set(tm) != set(range(1, 11)) or sorted(tm.values()).count(100) != 5:
        raise ValueError("invalid team map")
    slot = _slot_by_pid(roles)
    ts, xy = frame_positions(pack)
    i = int(np.searchsorted(ts, q, side="right") - 1)
    if i < 0:
        raise ValueError("query before the first frame")
    snap = int(ts[i])
    events = [e for e in pack.get("events") or [] if int(e.get("timestamp", 0)) <= q]
    patch = str(meta.get("patch", ""))
    iv = death_intervals(events, patch, t=q)
    alive = event_alive(iv, q)
    resp = respawned_since(iv, snap, q)
    towers = _standing_towers_at(events, q, patch)
    polys = lane_polylines()
    static = load_static_anchors()
    dragon = np.asarray(static["objectives"]["DRAGON"], dtype=float)
    baron = np.asarray(static["objectives"]["BARON"], dtype=float)

    pos: Dict[int, np.ndarray] = {}
    out: Dict[str, float] = {}
    pid_by_slot = {s: p for p, s in slot.items()}
    for s, pre in enumerate(SLOT_PREFIXES):
        p = pid_by_slot[s]
        team = tm[p]
        rec = {f: 0.0 for f in PLAYER_SETUP_FIELDS}
        rec["dist_own_tower"] = rec["dist_enemy_tower"] = FAR_NORM
        if alive[p - 1] > 0.5:
            if resp[p - 1] > 0.5:
                xy_p = np.asarray(FOUNTAIN[team], dtype=float)
                rec["at_fountain"] = 1.0
            else:
                xy_p = np.asarray(xy[i, p - 1], dtype=float)
            pos[p] = xy_p
            x, y = float(xy_p[0]), float(xy_p[1])
            rec["x"], rec["y"], rec["present"] = x / COORD_NORM_DIV, y / COORD_NORM_DIV, 1.0
            rec[_zone(x, y, team, polys)] = 1.0
            rec["near_dragon"] = float(_nearest(xy_p, dragon) * COORD_NORM_DIV <= PIT_RADIUS)
            rec["near_baron"] = float(_nearest(xy_p, baron) * COORD_NORM_DIV <= PIT_RADIUS)
            rec["dist_own_tower"] = _nearest(xy_p, towers[team])
            rec["dist_enemy_tower"] = _nearest(xy_p, towers[300 - team])
        for f in PLAYER_SETUP_FIELDS:
            out[pre + f] = float(rec[f])

    for team in TEAMS:
        s = SIDE[team]
        members = [p for p in range(1, 11) if tm[p] == team and p in pos]
        out[f"{s}_n_present"] = float(len(members))
        out[f"{s}_n_fountain"] = float(sum(resp[p - 1] > 0.5 for p in members))
        if members:
            pts = np.stack([pos[p] for p in members])
            c = pts.mean(0)
            spread = float(np.sqrt(((pts - c) ** 2).sum(1).mean()))
            d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
            group = float((d <= GROUP_RADIUS).sum(1).max())
        else:
            c, spread, group = np.asarray(FOUNTAIN[team], dtype=float), 0.0, 0.0
        out[f"{s}_centroid_x"] = float(c[0]) / COORD_NORM_DIV
        out[f"{s}_centroid_y"] = float(c[1]) / COORD_NORM_DIV
        out[f"{s}_spread"] = spread / COORD_NORM_DIV
        out[f"{s}_largest_group"] = group

    blue = [pos[p] for p in range(1, 11) if tm[p] == 100 and p in pos]
    red = [pos[p] for p in range(1, 11) if tm[p] == 200 and p in pos]
    if blue and red:
        d = np.sqrt(((np.stack(blue)[:, None] - np.stack(red)[None]) ** 2).sum(-1)).ravel()
        out["pair_min_dist"] = float(d.min()) / COORD_NORM_DIV
        out["pair_mean_dist"] = float(d.mean()) / COORD_NORM_DIV
        out["pairs_within_1600"] = float((d <= PAIR_RADII[0]).sum())
        out["pairs_within_3000"] = float((d <= PAIR_RADII[1]).sum())
        out["n_pairs"] = float(d.size)
    else:
        out.update(pair_min_dist=FAR_NORM, pair_mean_dist=FAR_NORM, pairs_within_1600=0.0,
                   pairs_within_3000=0.0, n_pairs=0.0)

    if tuple(out) != SETUP_COLUMNS:
        raise RuntimeError("setup column order drifted from SETUP_COLUMNS")
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("non-finite setup feature")
    return out


def setup_vector(values: Mapping[str, float], names: Sequence[str] = SETUP_COLUMNS) -> np.ndarray:
    return np.asarray([values[n] for n in names], dtype=float)


def name_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
