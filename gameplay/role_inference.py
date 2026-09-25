"""Role inference (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY) from exact match data only.

Plan: REESTIMATION_PLAN_V4_EXACT_20260925 section 3 ('역할 추론(정확 자료)').  The cache's
``meta['role_slots']`` is participant order, so roles are inferred from evidence instead; this
module never reads participant order as evidence (the result is equivariant under a permutation
of participant ids within a team, up to exact cost ties, which are flagged).

Evidence (all at or before the horizon H = min(t, horizon_ms); t = None means H = horizon_ms):
  smite    summoner spell 11 (static; meta static_meta summoner_spells_by_pid, else runes_by_pid,
           else node_minute summoner_spell_*_id at frame 0)                         -> JUNGLE
  support  an ITEM_* event credited to the player (participantId 1..10) whose itemId / beforeId /
           afterId is in the support line 3865 3866 3867 3869 3870 3871 3876 3877  -> UTILITY
           (Match-V5 credits the World Atlas *purchase* to participantId 0, so in practice the
           evidence is the 3865 -> 3866 upgrade, ITEM_DESTROYED 3865, typically 7-9 min.)
  position frames whose minute index floor(ts / 60 s) is in [3, 8] (ts <= H), players dead at
           the frame timestamp by kill-event survival (gameplay.event_survival) excluded.  For
           each frame the distance to the three lane polylines (tower chains from
           config/game_rules/map_anchors.json plus the lane corner) is taken; a player's lane
           distance is the median over frames.  With fewer than MIN_FRAMES (2) usable window frames,
           frames with 60 s <= ts <= H are used ('fallback_frames'); if none, the player has no
           position ('no_position'), with one, 'few_frames'.
  tie-break (only matters when support evidence is absent or ambiguous) separates the two
           bottom-lane players:
           start  gold value of the starting inventory, gameplay.item_state at min(H, 90 s).  The
                  World Atlas purchase is logged for participant 0 and dropped before the replay
                  (item_state would give it to participants 5 / 10 by participant order; _inventory_gold),
                  so a support shows only
                  potions / trinkets (<= 250 gold) while other roles show a Doran's item, jungle pet
                  or similar (>= 400 gold): 'low_start' = start gold <= LOW_START_GOLD.
           tags   Data Dragon class tags (Support / Marksman) of the champion.
           gold_mode='inventory' reproduces the first (v1) rule, which used the inventory value at
           H instead of 'low_start'; it is misled when fighter supports face mage bottom laners.

Assignment: per team a 5 x 5 cost matrix (players x ROLE_ORDER), solved with
scipy.optimize.linear_sum_assignment:
  JUNGLE   P_RULE * (1 - smite) + max(0, JUNGLE_REF - median min-lane distance)
  TOP      P_RULE * (smite + support) + d_top
  MIDDLE   P_RULE * (smite + support) + d_mid
  BOTTOM   P_RULE * (smite + support) + d_bot + W_TAG * (1 - marksman)
  UTILITY  P_RULE * (smite + 1 - support) + d_bot + W_TAG * (1 - support_tag) + W_START * (1 - low_start)
           (gold_mode='inventory': W_GOLD * inventory gold at H / 1000 replaces the W_START term)
With exactly one Smite and one support-item player the rules decide JUNGLE and UTILITY and the
positions decide TOP/MIDDLE/BOTTOM (Hungarian on the remaining three).  With zero or two such
players the penalty is the same for every assignment and the costs decide; the team is flagged.

Confidence: margin = (second-best assignment cost) - (best), by enumerating the 120 permutations.
A team is 'ambiguous' when smite count != 1, support count > 1 (support-item players without Smite),
a player has fewer than MIN_FRAMES usable frames, summoner spells are missing, or
margin < MARGIN_MIN.  support count == 0 is reported ('no_support_item') but only makes the team
ambiguous through the margin.

Exactness: only frames with timestamp <= H and events with timestamp <= H are read; no
interpolation, no 5-s grid, no frame alive / status / cs / ds columns (positions come from the
node_minute x_norm / y_norm columns, which equal xy_raw_minute / COORD_NORM_DIV; xy_raw_minute is
read only when node_minute is absent).
"""
from __future__ import annotations

import ast
import itertools
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

ROLES: Tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")  # == core.config.ROLE_ORDER
TEAMS: Tuple[int, int] = (100, 200)
SMITE_SPELL_ID = 11
SUPPORT_ITEM_IDS = frozenset({3865, 3866, 3867, 3869, 3870, 3871, 3876, 3877})
ITEM_EVENT_TYPES = frozenset({"ITEM_PURCHASED", "ITEM_DESTROYED", "ITEM_SOLD", "ITEM_UNDO"})

WINDOW_MINUTES: Tuple[int, int] = (3, 8)          # inclusive minute indices floor(ts / 60 s)
HORIZON_MS = (WINDOW_MINUTES[1] + 1) * 60_000 - 1  # 539 999: last ms of minute 8
FALLBACK_START_MS = 60_000
MIN_FRAMES = 2         # fewer usable window frames -> use fallback frames; fewer still -> ambiguous

P_RULE = 100_000.0     # rule penalty (Smite / support item), >> any distance on the map
JUNGLE_REF = 3_000.0   # a player whose median distance to every lane is >= this costs 0 as JUNGLE
W_TAG = 500.0          # tie-break: champion class tag (Support for UTILITY, Marksman for BOTTOM)
W_START = 2_000.0      # tie-break: UTILITY cost of a player whose starting inventory is not support-like
START_MS = 90_000      # starting inventory = inventory at min(H, START_MS)
LOW_START_GOLD = 250.0 # starting inventory gold <= this -> support-like ('low_start')
W_GOLD = 400.0         # gold_mode='inventory' (v1): per 1000 gold of inventory value at H, added to UTILITY
GOLD_MODES = ("start", "inventory")
MARGIN_MIN = 500.0     # second-best minus best total cost below this -> ambiguous

ANCHORS_PATH = Path(__file__).resolve().parents[1] / "config" / "game_rules" / "map_anchors.json"
_TOWER_ORDER = ("INHIBITOR", "BASE_TURRET", "INNER_TURRET", "OUTER_TURRET")


# --------------------------------------------------------------------------------------------
# Lane geometry
# --------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def lane_polylines(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """{'TOP'|'MIDDLE'|'BOTTOM': (k, 2) polyline} in game units, from blue inhibitor to red inhibitor.

    Points are the median tower / inhibitor positions of map_anchors.json in the order inhibitor,
    base, inner, outer (blue), then (top / bottom) the lane corner, then outer ... inhibitor (red).
    The corner is the square corner of the two tower lines: TOP (median x of the blue top chain,
    median y of the red top chain); BOTTOM (median x of the red bottom chain, median y of the blue
    bottom chain).
    """
    d = json.loads(Path(path or ANCHORS_PATH).read_text(encoding="utf-8"))
    chain: Dict[Tuple[int, str], List[Tuple[float, float]]] = {}
    for lane in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
        for team in TEAMS:
            pts = {}
            for t in d["towers"]:
                if int(t["team"]) != team or t["lane"] != lane:
                    continue
                key = "INHIBITOR" if t["building"] == "INHIBITOR_BUILDING" else str(t["tower"])
                if key in _TOWER_ORDER:
                    pts[key] = (float(t["x"]), float(t["y"]))
            missing = [k for k in _TOWER_ORDER if k not in pts]
            if missing:
                raise ValueError(f"map_anchors: {team} {lane} lacks {missing}")
            chain[(team, lane)] = [pts[k] for k in _TOWER_ORDER]
    out = {}
    for lane, name in (("TOP_LANE", "TOP"), ("MID_LANE", "MIDDLE"), ("BOT_LANE", "BOTTOM")):
        blue, red = chain[(100, lane)], chain[(200, lane)]
        mid = []
        if name == "TOP":
            mid = [(float(np.median([p[0] for p in blue])), float(np.median([p[1] for p in red])))]
        elif name == "BOTTOM":
            mid = [(float(np.median([p[0] for p in red])), float(np.median([p[1] for p in blue])))]
        out[name] = np.asarray(blue + mid + red[::-1], dtype=float)
    return out


def point_polyline_distance(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Euclidean distance of each point in xy (n, 2) to the polyline poly (k, 2)."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = xy[:, None, :] - a[None, :, :]
    u = np.clip((ap * ab[None]).sum(2) / denom[None], 0.0, 1.0)
    proj = a[None] + u[..., None] * ab[None]
    return np.sqrt(((xy[:, None, :] - proj) ** 2).sum(2)).min(1)


# --------------------------------------------------------------------------------------------
# Evidence extraction
# --------------------------------------------------------------------------------------------
def _static_meta(meta: Mapping) -> dict:
    sm = (meta or {}).get("static_meta") or {}
    return ast.literal_eval(sm) if isinstance(sm, str) else dict(sm)


def _pid_map(d: Optional[Mapping]) -> Dict[int, object]:
    return {int(k): v for k, v in (d or {}).items()}


def team_map_of(pack: Mapping) -> Dict[int, int]:
    tm = _pid_map((pack.get("meta") or {}).get("team_map"))
    tm = {p: int(v) for p, v in tm.items() if 1 <= p <= 10}
    if sorted(tm) != list(range(1, 11)) or sorted(tm.values()).count(100) != 5:
        raise ValueError(f"team_map must give 5 players per team, got {tm}")
    return tm


def smite_by_pid(pack: Mapping) -> Dict[int, Optional[bool]]:
    """{pid: True/False, or None when no summoner spell source has the player}."""
    sm = _static_meta(pack.get("meta") or {})
    out: Dict[int, Optional[bool]] = {p: None for p in range(1, 11)}
    for src in ("summoner_spells_by_pid", "runes_by_pid"):
        for p, rec in _pid_map(sm.get(src)).items():
            if 1 <= p <= 10 and out[p] is None and isinstance(rec, Mapping):
                ids = [rec.get("summoner_spell_1_id"), rec.get("summoner_spell_2_id")]
                if any(i is not None for i in ids):
                    out[p] = SMITE_SPELL_ID in {int(i) for i in ids if i is not None}
    nm = pack.get("node_minute")
    if any(v is None for v in out.values()) and nm is not None and len(nm):
        from core.config import NODE_IDX
        cols = [NODE_IDX.get("summoner_spell_1_id"), NODE_IDX.get("summoner_spell_2_id")]
        if None not in cols:
            for p in range(1, 11):
                if out[p] is None:
                    out[p] = SMITE_SPELL_ID in {int(round(float(nm[0, p - 1, c]))) for c in cols}
    return out


def support_item_first_ms(events: Iterable[Mapping], horizon_ms: float) -> Dict[int, Optional[int]]:
    """{pid: first timestamp (<= horizon) of an ITEM_* event credited to pid with a support-line id}."""
    out: Dict[int, Optional[int]] = {p: None for p in range(1, 11)}
    for e in events or ():
        if e.get("type") not in ITEM_EVENT_TYPES:
            continue
        ts = int(e.get("timestamp", 0) or 0)
        if ts > horizon_ms:
            continue
        p = int(e.get("participantId", 0) or 0)
        if not 1 <= p <= 10:
            continue
        ids = {int(e.get(k) or 0) for k in ("itemId", "beforeId", "afterId")}
        if ids & SUPPORT_ITEM_IDS and (out[p] is None or ts < out[p]):
            out[p] = ts
    return out


def frame_positions(pack: Mapping) -> Tuple[np.ndarray, np.ndarray]:
    """(minute_ts (T,), xy (T, 10, 2) game units) from node_minute x_norm / y_norm."""
    ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    nm = pack.get("node_minute")
    if nm is not None:
        from core.config import NODE_IDX, cfg
        div = float(getattr(cfg, "COORD_NORM_DIV"))
        nm = np.asarray(nm)
        xy = np.stack([nm[:, :, NODE_IDX["x_norm"]], nm[:, :, NODE_IDX["y_norm"]]], -1).astype(float) * div
    else:
        xy = np.asarray(pack["xy_raw_minute"], dtype=float)
    if xy.shape[0] != ts.shape[0] or xy.shape[1:] != (10, 2):
        raise ValueError(f"position array {xy.shape} does not match minute_ts {ts.shape}")
    return ts, xy


def _dead_mask(pack: Mapping, ts: np.ndarray, horizon: float) -> Tuple[np.ndarray, Optional[str]]:
    """(T, 10) bool: player dead at the frame timestamp by kill-event survival (events <= horizon)."""
    dead = np.zeros((len(ts), 10), dtype=bool)
    patch = str((pack.get("meta") or {}).get("patch", ""))
    try:
        from gameplay.event_survival import death_intervals, event_alive
        iv = death_intervals(pack.get("events") or [], patch, t=horizon)
    except (KeyError, ValueError) as exc:
        return dead, f"no_survival_rule:{patch}:{exc.__class__.__name__}"
    for j, t in enumerate(ts):
        if t <= horizon:
            dead[j] = event_alive(iv, float(t)) < 0.5
    return dead, None


def lane_distances(pack: Mapping, horizon: float, window: Tuple[int, int] = WINDOW_MINUTES,
                   polylines: Optional[Mapping[str, np.ndarray]] = None) -> Dict[int, dict]:
    """{pid: {'TOP','MIDDLE','BOTTOM': median distance, 'min_lane': median of min distance,
    'n_frames': frames used, 'source': 'window'|'fallback'|'none', 'mean_xy': (x, y)}}."""
    return _lane_distances(pack, horizon, window, polylines)[0]


def _lane_distances(pack, horizon, window=WINDOW_MINUTES, polylines=None):
    polys = polylines or lane_polylines()
    ts, xy = frame_positions(pack)
    dead, note = _dead_mask(pack, ts, horizon)
    minute = ts // 60_000
    in_win = (minute >= window[0]) & (minute <= window[1]) & (ts <= horizon)
    in_fb = (ts >= FALLBACK_START_MS) & (ts <= horizon)
    out = {}
    for p in range(1, 11):
        src, sel = "window", in_win & ~dead[:, p - 1]
        if sel.sum() < MIN_FRAMES:
            src, sel = "fallback", in_fb & ~dead[:, p - 1]
        if not sel.any():
            out[p] = {"TOP": 0.0, "MIDDLE": 0.0, "BOTTOM": 0.0, "min_lane": 0.0, "n_frames": 0,
                      "source": "none", "mean_xy": (float("nan"), float("nan"))}
            continue
        pts = xy[sel, p - 1]
        d = {ln: point_polyline_distance(pts, polys[ln]) for ln in ("TOP", "MIDDLE", "BOTTOM")}
        mn = np.minimum(np.minimum(d["TOP"], d["MIDDLE"]), d["BOTTOM"])
        out[p] = {**{ln: float(np.median(v)) for ln, v in d.items()}, "min_lane": float(np.median(mn)),
                  "n_frames": int(sel.sum()), "source": src,
                  "mean_xy": (float(pts[:, 0].mean()), float(pts[:, 1].mean()))}
    return out, note


def _champion_tags(pack: Mapping) -> Tuple[Dict[int, Tuple[float, float]], Optional[str]]:
    """{pid: (Support tag, Marksman tag)}; (0, 0) if the table or champion is unavailable."""
    meta = pack.get("meta") or {}
    out = {p: (0.0, 0.0) for p in range(1, 11)}
    try:
        from gameplay.champion_attributes import champion_ids_from_meta, champion_vector, load_champion_table_v2
        tbl = load_champion_table_v2(str(meta.get("patch", "")))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return out, f"no_champion_table:{exc.__class__.__name__}"
    for p, cid in champion_ids_from_meta(meta).items():
        v = champion_vector(cid, tbl)
        if v["champ_unknown"] == 0.0:
            out[p] = (v["tag_Support"], v["tag_Marksman"])
    return out, None


def _inventory_gold(pack: Mapping, horizon: float) -> Tuple[Dict[int, float], Optional[str]]:
    """{pid: gold value of the inventory at the horizon} (gameplay.item_state, events <= horizon).

    The World Atlas purchases Match-V5 logs with participantId 0 are dropped before the replay: the exact item
    replay gives them to participants 5 and 10, i.e. by participant order, which this module never reads as
    evidence (with them every slot-5 / 10 start inventory is worth >= 400 gold and the 'low_start' tie-break would
    follow participant order).  The later 3865 -> 3866 upgrade is still replayed (3866 is added on the destroy)."""
    try:
        from gameplay.item_state import WORLD_ATLAS_ID, ItemStateIndex
        events = [e for e in (pack.get("events") or [])
                  if not (e.get("type") == "ITEM_PURCHASED" and int(e.get("participantId", 0) or 0) == 0
                          and int(e.get("itemId", 0) or 0) == WORLD_ATLAS_ID)]
        idx = ItemStateIndex.from_pack({**pack, "events": events})
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {p: 0.0 for p in range(1, 11)}, f"no_item_table:{exc.__class__.__name__}"
    return {p: float(idx.vector(p, int(horizon))["item_gold_owned"]) for p in range(1, 11)}, None


# --------------------------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------------------------
@dataclass
class TeamRoles:
    team: int
    pids: List[int]                  # row order of `cost`
    roles: Dict[int, str]            # pid -> role
    cost: np.ndarray                 # (5, 5) players x ROLES
    total_cost: float
    margin: float                    # second-best total cost - best (inf if unique by construction)
    n_smite: int
    n_support: int
    ambiguous: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class RoleAssignment:
    roles: Dict[int, str]            # pid -> role
    slot_by_pid: Dict[int, int]      # pid -> 0..9 in [blue ROLES, red ROLES]
    pid_by_slot: List[int]           # slot -> pid
    teams: Dict[int, TeamRoles]
    horizon_ms: float
    evidence: Dict[int, dict]        # pid -> evidence used

    @property
    def ambiguous(self) -> bool:
        return any(tr.ambiguous for tr in self.teams.values())

    @property
    def confident(self) -> bool:
        return not self.ambiguous


def cost_matrix(ev: Sequence[Mapping]) -> np.ndarray:
    """(n, 5) cost of each player (evidence dict) for each role in ROLES."""
    c = np.zeros((len(ev), len(ROLES)))
    for i, e in enumerate(ev):
        s, u = float(bool(e["smite"])), float(bool(e["support_item"]))
        d = e["lane"]
        lane_pen = P_RULE * (s + u)
        c[i, ROLES.index("TOP")] = lane_pen + d["TOP"]
        c[i, ROLES.index("MIDDLE")] = lane_pen + d["MIDDLE"]
        c[i, ROLES.index("BOTTOM")] = lane_pen + d["BOTTOM"] + W_TAG * (1.0 - e["tag_marksman"])
        mode = e.get("gold_mode", "none")
        gold_term = (W_START * (1.0 - float(e["low_start"])) if mode == "start"
                     else W_GOLD * float(e["gold"]) / 1000.0 if mode == "inventory" else 0.0)
        c[i, ROLES.index("UTILITY")] = (P_RULE * (s + 1.0 - u) + d["BOTTOM"] + W_TAG * (1.0 - e["tag_support"])
                                        + gold_term)
        c[i, ROLES.index("JUNGLE")] = P_RULE * (1.0 - s) + max(0.0, JUNGLE_REF - d["min_lane"])
    return c


def solve_team(cost: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """(role index per row, best total, margin to the second-best permutation)."""
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(cost.shape[0], dtype=int)
    assign[rows] = cols
    best = float(cost[rows, cols].sum())
    totals = sorted(float(cost[np.arange(cost.shape[0]), list(perm)].sum())
                    for perm in itertools.permutations(range(cost.shape[1]), cost.shape[0]))
    second = totals[1] if len(totals) > 1 else float("inf")
    return assign, best, second - best


def infer_roles(pack: Mapping, t: Optional[float] = None, *, horizon_ms: float = HORIZON_MS,
                use_tiebreak: bool = True, gold_mode: str = "start") -> RoleAssignment:
    """Roles of the 10 players from data with timestamp <= H = min(t, horizon_ms).

    pack: cache pack with 'minute_ts', 'node_minute' (or 'xy_raw_minute'), 'events' and 'meta'
    ('team_map', 'patch', 'static_meta').  t: query time in ms (None: H = horizon_ms, i.e. the
    whole 3-8 min window).  use_tiebreak=False drops the champion-tag and gold terms;
    gold_mode='inventory' is the v1 tie-break (inventory value at H instead of 'low_start').
    """
    if gold_mode not in GOLD_MODES:
        raise ValueError(f"gold_mode must be one of {GOLD_MODES}, got {gold_mode!r}")
    H = float(horizon_ms if t is None else min(float(t), float(horizon_ms)))
    tm = team_map_of(pack)
    smite = smite_by_pid(pack)
    sup_ms = support_item_first_ms(pack.get("events") or [], H)
    lanes, n0 = _lane_distances(pack, H)
    notes: List[str] = [n0] if n0 else []
    gold_ok = False
    if use_tiebreak:
        tags, n1 = _champion_tags(pack)
        gold, n2 = _inventory_gold(pack, H if gold_mode == "inventory" else min(H, START_MS))
        notes += [n for n in (n1, n2) if n]
        gold_ok = n2 is None
    else:
        tags, gold = {p: (1.0, 1.0) for p in range(1, 11)}, {p: 0.0 for p in range(1, 11)}
    mode = gold_mode if (use_tiebreak and gold_ok) else "none"
    evidence = {p: {"team": tm[p], "smite": bool(smite[p]), "smite_known": smite[p] is not None,
                    "support_item": sup_ms[p] is not None, "support_item_ms": sup_ms[p],
                    "lane": lanes[p], "tag_support": tags[p][0], "tag_marksman": tags[p][1],
                    "gold": gold[p], "gold_mode": mode,
                    "low_start": float(mode == "start" and gold[p] <= LOW_START_GOLD)}
                for p in range(1, 11)}
    roles: Dict[int, str] = {}
    teams: Dict[int, TeamRoles] = {}
    for team in TEAMS:
        pids = sorted(p for p in range(1, 11) if tm[p] == team)
        ev = [evidence[p] for p in pids]
        cost = cost_matrix(ev)
        assign, best, margin = solve_team(cost)
        tr_roles = {p: ROLES[assign[i]] for i, p in enumerate(pids)}
        n_s = sum(e["smite"] for e in ev)
        # support evidence on a Smite player (seen: a jungler credited with ITEM_DESTROYED 3865) is
        # overruled by the Smite rule; only non-Smite support-item players count as UTILITY candidates
        n_u = sum(e["support_item"] and not e["smite"] for e in ev)
        reasons = list(notes)
        if n_s != 1:
            reasons.append(f"smite_count={n_s}")
        if not all(e["smite_known"] for e in ev):
            reasons.append("summoner_spells_missing")
        if n_u == 0:
            reasons.append("no_support_item")
        elif n_u > 1:
            reasons.append(f"support_item_count={n_u}")
        if any(e["lane"]["source"] == "none" for e in ev):
            reasons.append("no_position")
        elif any(e["lane"]["n_frames"] < MIN_FRAMES for e in ev):
            reasons.append("few_frames")
        if any(e["lane"]["source"] == "fallback" for e in ev):
            reasons.append("fallback_frames")
        if margin < MARGIN_MIN:
            reasons.append(f"margin={margin:.0f}")
        amb = (n_s != 1 or n_u > 1 or margin < MARGIN_MIN or "no_position" in reasons
               or "few_frames" in reasons or "summoner_spells_missing" in reasons)
        teams[team] = TeamRoles(team, pids, tr_roles, cost, best, margin, int(n_s), int(n_u), bool(amb), reasons)
        roles.update(tr_roles)
    slot_by_pid = {p: TEAMS.index(tm[p]) * 5 + ROLES.index(r) for p, r in roles.items()}
    pid_by_slot = [0] * 10
    for p, s in slot_by_pid.items():
        pid_by_slot[s] = p
    return RoleAssignment(roles, slot_by_pid, pid_by_slot, teams, H, evidence)


def role_slots(pack: Mapping, t: Optional[float] = None, **kw) -> Dict[int, int]:
    """pid -> slot 0..9 ([blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY, red ...]), the layout of
    meta['role_slots'] / core.roles.get_role_slots_from_detail."""
    return infer_roles(pack, t, **kw).slot_by_pid
