"""Outcome-side labels for v4-exact engagements (stage 1, task B3).

These are OUTCOME functions: they may read events after tau, up to the label end e_h (and, for the
next-objective label, up to e_h + 180 s).  They never read frames, the 5-s grid, node alive/status
columns or positions other than the event positions of Match-V5 timeline events.

Label coding (same as gameplay/labels.py): 1 = blue (team 100) side won, 0 = red (team 200),
None = excluded.  A tie is ALWAYS excluded; there is no coin flip (labels._seeded_tie_coin is never
imported or called) and no cfg tie policy is read.

Engagement input (a mapping, e.g. a gameplay.exact_population record):
  'tau' or 'engage_ts'   engagement start tau (ms)
  'last_kill_ts'         L, the last kill of the engagement's own kill set (ms)
  own kill set           'kills' (raw CHAMPION_KILL event dicts) or 'kill_idx' (indices into the
                         timestamp-sorted kill list of fights._extract_kill_events(events), the
                         index used by exact_population records)
  optional              'anchor_xy' (x, y); else the position of the first own kill (earliest
                         timestamp, event order); else ('centroid_x', 'centroid_y')
  optional              'team_map', 'patch' (else pass them as keyword arguments)

exchange_label (priced exchange, lexicographic, ties excluded):
  gold = sum over OWN kills of sign(killer team) * (max(0, bounty) + max(0, shutdownBounty)
         + prices['kills'] + prices['assists'] * n_assisters)                    [labels.py rule]
       + sum over non-kill events with timestamp in [tau, e_h] whose position lies within D of the
         anchor (distance <= D) of sign(acquiring team) * labels._priced_event_gold(e)
         (plates, towers (+ first_tower), inhibitors, elite monsters; unpositioned events such as
         WARD_KILL cannot be placed and are skipped, as labels.attribute_events does)
  1. gold >  +deadzone -> 1;  gold < -deadzone -> 0   (deadzone 300 g)
  2. own-kill difference (blue kills - red kills).  An execution (killerId 0: minion, turret,
     monster) counts as a LOSS for the victim's team, i.e. as a kill for the other team (author
     decision 2026-09-25, records/stage1_decisions_20260925T103104Z.json); its gold is 0 in tier 1
  3. kill-event alive difference at L: event_survival.event_alive(death_intervals(events <= L), L),
     blue alive - red alive over all ten players
  4. structures: sum of signs of ELITE_MONSTER_KILL / BUILDING_KILL / TURRET_PLATE_DESTROYED events
     in [tau, e_h] within D of the anchor (the labels._lex_refine set)
  5. None (excluded)
kill_diff_label: sign of the own-kill difference (executions as in tier 2); 0 -> None.
next_objective_label: team of the first elite objective (baron, dragon incl. elder, herald,
  voidgrubs, Atakhan) in (e_h, e_h + 180 s]; none / game end / simultaneous-different-teams -> None.
  A camp despawn (killerTeamId 300, no acquiring team) is not an acquisition and is skipped by
  default (skip_despawn=True); skip_despawn=False reproduces the legacy classifier of
  scripts/t023_market_event_r0r1.py (a despawn then reads as 'tie_ambiguous').
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from gameplay.event_survival import death_intervals, event_alive
from gameplay.labels import _label_event_team_sign, _priced_event_gold

DEADZONE_GOLD = 300.0
NEXT_OBJECTIVE_WINDOW_MS = 180_000
ELIGIBLE_OBJECTIVES = ("BARON_NASHOR", "DRAGON", "RIFTHERALD", "HORDE", "ATAKHAN")
PRICED_NON_KILL_TYPES = ("TURRET_PLATE_DESTROYED", "BUILDING_KILL", "ELITE_MONSTER_KILL", "WARD_KILL")
STRUCTURE_TYPES = ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED")
DECIDED_BY = ("gold", "kills", "alive", "structures", "excluded")
NEXT_OBJECTIVE_OUTCOMES = ("Blue", "Red", "none", "tie_ambiguous", "game_ended_before_objective",
                           "game_ended_in_window_no_objective", "observation_censored")

_WT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_TABLE = _WT / "config" / "game_rules" / "event_prices.json"
# DR = C:/Users/todtj/문서/LOL_Teamfight/scripts (worktree parent's scripts folder); imported unchanged.
DEFAULT_DR_RULES = _WT.parents[1] / "scripts" / "engagement_labels_v3_rules.py"


# ------------------------------------------------------------------ small helpers
def _ts(e: Mapping) -> Optional[int]:
    v = e.get("timestamp", e.get("ts"))
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _etype(e: Mapping) -> str:
    return str(e.get("type", e.get("eventType", ""))).upper()


def _event_xy(e: Mapping) -> Optional[Tuple[float, float]]:
    pos = e.get("position", None)
    if isinstance(pos, Mapping) and "x" in pos and "y" in pos:
        return float(pos["x"]), float(pos["y"])
    if isinstance(pos, (list, tuple)) and len(pos) >= 2 and pos[0] is not None and pos[1] is not None:
        return float(pos[0]), float(pos[1])
    return None


def _team_map(engagement: Mapping, team_map: Optional[Mapping]) -> Dict[int, int]:
    tm = team_map if team_map is not None else engagement.get("team_map")
    if not tm:  # Match-V5: participants 1-5 are team 100, 6-10 team 200
        return {i: (100 if i <= 5 else 200) for i in range(1, 11)}
    return {int(k): int(v) for k, v in tm.items()}


def _tau(engagement: Mapping) -> int:
    for k in ("tau", "engage_ts"):
        if engagement.get(k) is not None:
            return int(engagement[k])
    raise KeyError("engagement needs 'tau' or 'engage_ts'")


def load_price_table(path: Optional[Path] = None, patch: Optional[str] = None) -> Dict[str, float]:
    """Price table {key: team gold per event}.  patch=None -> 'pooled'; else per_patch[patch]."""
    blob = json.loads(Path(path or DEFAULT_PRICE_TABLE).read_text(encoding="utf-8"))
    tab = blob["pooled"] if patch is None else blob["per_patch"][str(patch)]
    return {str(k): float(v) for k, v in tab.items()}


def raw_kill_events(events: Iterable[Mapping]) -> List[Mapping]:
    """Raw CHAMPION_KILL events in the order of fights._extract_kill_events (stable sort by
    timestamp), so that exact_population 'kill_idx' indexes this list."""
    out = [e for e in events or [] if isinstance(e, Mapping) and _etype(e) == "CHAMPION_KILL" and _ts(e) is not None]
    out.sort(key=lambda e: _ts(e))
    return out


def own_kill_events(engagement: Mapping, events: Sequence[Mapping]) -> List[Mapping]:
    """The engagement's own kills as raw CHAMPION_KILL events, sorted by timestamp (stable)."""
    if engagement.get("kills") is not None:
        ks = list(engagement["kills"])
        bad = [k for k in ks if "killerId" not in k or _ts(k) is None or _etype(k) not in ("", "CHAMPION_KILL")]
        if bad:
            raise ValueError("'kills' must be raw Match-V5 CHAMPION_KILL events (killerId, timestamp, bounty, ...)")
        return sorted(ks, key=lambda e: _ts(e))
    if engagement.get("kill_idx") is not None:
        allk = raw_kill_events(events)
        return [allk[int(i)] for i in sorted(int(j) for j in engagement["kill_idx"])]
    raise KeyError("engagement needs its own kill set: 'kills' (raw events) or 'kill_idx'")


def engagement_anchor(engagement: Mapping, own_kills: Sequence[Mapping]) -> Optional[Tuple[float, float]]:
    """'anchor_xy' if given, else the first own kill's position, else (centroid_x, centroid_y)."""
    a = engagement.get("anchor_xy")
    if a is not None:
        return float(a[0]), float(a[1])
    for k in own_kills:
        xy = _event_xy(k)
        if xy is not None:
            return xy
    if engagement.get("centroid_x") is not None and engagement.get("centroid_y") is not None:
        return float(engagement["centroid_x"]), float(engagement["centroid_y"])
    return None


def first_tower_ts(events: Iterable[Mapping], upto: Optional[int] = None) -> Optional[int]:
    """Timestamp of the match's first tower (non-inhibitor BUILDING_KILL), using events <= upto."""
    best = None
    for e in events or []:
        if _etype(e) != "BUILDING_KILL" or str(e.get("buildingType", "")).upper() == "INHIBITOR_BUILDING":
            continue
        ts = _ts(e)
        if ts is None or (upto is not None and ts > upto):
            continue
        best = ts if best is None or ts < best else best
    return best


def priced_kill_gold(kill: Mapping, prices: Mapping[str, float]) -> float:
    """labels._compute_label_market_event kill price: paid bounty + shutdown + kill/assist prices."""
    def f(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if math.isfinite(x) else 0.0
    g = max(0.0, f(kill.get("bounty", 0.0))) + max(0.0, f(kill.get("shutdownBounty", 0.0)))
    assists = kill.get("assistingParticipantIds", [])
    n_ast = len(assists) if isinstance(assists, list) else 0
    return g + float(prices.get("kills", 0.0)) + float(prices.get("assists", 0.0)) * n_ast


def _within(xy: Optional[Tuple[float, float]], anchor: Tuple[float, float], radius: float) -> bool:
    if xy is None:
        return False
    return (xy[0] - anchor[0]) ** 2 + (xy[1] - anchor[1]) ** 2 <= radius * radius


def is_execution(kill: Mapping) -> bool:
    """A CHAMPION_KILL without a champion killer (killerId 0: minion, turret or monster)."""
    try:
        return int(kill.get("killerId", 0) or 0) == 0
    except (TypeError, ValueError):
        return False


def kill_team_sign(kill: Mapping, team_map: Mapping[int, int]) -> int:
    """+1 if the kill counts for blue, -1 for red, 0 otherwise.

    A champion killer counts for its team (labels._label_event_team_sign).  An execution (killerId 0)
    counts as a loss for the victim's team: a blue victim gives -1, a red victim +1 (author decision
    2026-09-25).  A kill whose killer / victim team is unknown counts for nobody."""
    if is_execution(kill):
        try:
            vteam = int(team_map.get(int(kill.get("victimId", 0) or 0), 0) or 0)
        except (TypeError, ValueError):
            vteam = 0
        return -1 if vteam == 100 else 1 if vteam == 200 else 0
    return _label_event_team_sign(dict(kill, type="CHAMPION_KILL"), team_map)


def kill_difference(own_kills: Iterable[Mapping], team_map: Mapping[int, int]) -> int:
    """Blue kills - red kills over the given kills (kill_team_sign: killer's team; an execution
    counts against the victim's team)."""
    return sum(kill_team_sign(k, team_map) for k in own_kills)


def alive_difference_at(events: Sequence[Mapping], t: int, patch: str, team_map: Mapping[int, int]) -> int:
    """Blue alive - red alive at t from kill events <= t (event_survival; no frame data)."""
    al = event_alive(death_intervals(events, patch, t=t), t)
    d = 0
    for pid in range(1, 11):
        team = int(team_map.get(pid, 100 if pid <= 5 else 200))
        if team == 100:
            d += int(al[pid - 1])
        elif team == 200:
            d -= int(al[pid - 1])
    return d


def _sign_label(v: float) -> Optional[int]:
    if v > 0:
        return 1
    if v < 0:
        return 0
    return None


# ------------------------------------------------------------------ exchange label
def exchange_outcome(engagement: Mapping, events: Sequence[Mapping], prices: Mapping[str, float], e_h: int, *,
                     radius: Optional[float] = None, team_map: Optional[Mapping] = None,
                     patch: Optional[str] = None, deadzone: float = DEADZONE_GOLD) -> Dict[str, Any]:
    """Full priced-exchange outcome.  See the module docstring for the tiers.

    radius: D (raw map units) for non-kill event attribution (pipeline: the locked CLUSTER_MAX_DIAMETER);
            may also come from engagement['radius'].
    Only events with timestamp <= e_h are read.  Raises ValueError if e_h < L (an invalid endpoint
    blocks the row; it is never silently labelled).
    """
    tm = _team_map(engagement, team_map)
    patch = patch if patch is not None else engagement.get("patch")
    if patch is None:
        raise ValueError("patch is required (keyword or engagement['patch']) for the alive tier")
    radius = radius if radius is not None else engagement.get("radius")
    if radius is None or not (float(radius) > 0):
        raise ValueError("radius D (> 0) is required (keyword or engagement['radius'])")
    radius = float(radius)
    tau = _tau(engagement)
    e_h = int(e_h)
    evs = [e for e in events or [] if isinstance(e, Mapping) and _ts(e) is not None and _ts(e) <= e_h]
    own = own_kill_events(engagement, events)  # kill_idx indexes the match's full kill list
    if not own:
        raise ValueError("engagement has no own kills")
    L = int(engagement["last_kill_ts"]) if engagement.get("last_kill_ts") is not None else int(_ts(own[-1]))
    if int(_ts(own[-1])) != L:
        raise ValueError(f"last_kill_ts {L} != last own kill {_ts(own[-1])}")
    if e_h < L:
        raise ValueError(f"invalid endpoint e_h={e_h} < L={L}")
    if int(_ts(own[0])) < tau:
        raise ValueError("own kill before tau")
    anchor = engagement_anchor(engagement, own)

    gold_k = 0.0
    for k in own:  # an execution (killer 0) has sign 0 here: its gold contribution is 0
        s = _label_event_team_sign(dict(k, type="CHAMPION_KILL"), tm)
        if s:
            gold_k += s * priced_kill_gold(k, prices)
    ft = first_tower_ts(evs, upto=e_h) if prices else None
    gold_o, struct, n_priced, n_struct, n_unplaced = 0.0, 0, 0, 0, 0
    for e in evs:
        et = _etype(e)
        if et not in PRICED_NON_KILL_TYPES:
            continue
        ts = _ts(e)
        if ts < tau:
            continue
        xy = _event_xy(e)
        if xy is None or anchor is None:
            n_unplaced += 1
            continue
        if not _within(xy, anchor, radius):
            continue
        s = _label_event_team_sign(e, tm)
        if s == 0:
            continue
        g = _priced_event_gold(dict(e), dict(prices), ft)
        if g:
            gold_o += s * g
            n_priced += 1
        if et in STRUCTURE_TYPES:
            struct += s
            n_struct += 1
    gold = gold_k + gold_o
    kd = kill_difference(own, tm)
    out: Dict[str, Any] = {
        "tau": tau, "L": L, "e_h": e_h, "anchor": anchor, "radius": radius,
        "gold_diff": gold, "gold_kills": gold_k, "gold_other": gold_o, "n_own_kills": len(own),
        "n_priced_other": n_priced, "n_structure_events": n_struct, "n_unplaced_skipped": n_unplaced,
        "n_executions": sum(1 for k in own if is_execution(k)),
        "kill_diff": kd, "alive_diff": None, "structure_diff": struct, "label": None, "decided_by": "excluded",
    }
    if gold > deadzone:
        out.update(label=1, decided_by="gold")
        return out
    if gold < -deadzone:
        out.update(label=0, decided_by="gold")
        return out
    if kd != 0:
        out.update(label=_sign_label(kd), decided_by="kills")
        return out
    ad = alive_difference_at(evs, L, str(patch), tm)
    out["alive_diff"] = ad
    if ad != 0:
        out.update(label=_sign_label(ad), decided_by="alive")
        return out
    if struct != 0:
        out.update(label=_sign_label(struct), decided_by="structures")
        return out
    return out  # tie on every tier -> excluded, never random


def exchange_label(engagement: Mapping, events: Sequence[Mapping], prices: Mapping[str, float], e_h: int,
                   **kw: Any) -> Optional[int]:
    """1 (blue), 0 (red) or None (excluded tie).  Keywords: radius, team_map, patch, deadzone."""
    return exchange_outcome(engagement, events, prices, e_h, **kw)["label"]


# ------------------------------------------------------------------ kill-difference label
def kill_diff_label(engagement: Mapping, events: Sequence[Mapping], team_map: Optional[Mapping] = None) -> Optional[int]:
    """Sign of the own-kill difference: 1 blue, 0 red, None when equal."""
    own = own_kill_events(engagement, events)
    return _sign_label(kill_difference(own, _team_map(engagement, team_map)))


# ------------------------------------------------------------------ next objective
def game_end_ms(events: Iterable[Mapping]) -> Optional[int]:
    """GAME_END event timestamp (None if absent)."""
    for e in events or []:
        if _etype(e) == "GAME_END":
            return _ts(e)
    return None


def next_objective_outcome(events: Iterable[Mapping], e_h: int, *, team_map: Optional[Mapping] = None,
                           game_end: Optional[int] = None, window_ms: int = NEXT_OBJECTIVE_WINDOW_MS,
                           skip_despawn: bool = True) -> str:
    """Category of the first elite objective in (e_h, e_h + window]; see NEXT_OBJECTIVE_OUTCOMES."""
    tm = {int(k): int(v) for k, v in (team_map or {i: (100 if i <= 5 else 200) for i in range(1, 11)}).items()}
    evs = list(events or [])
    if game_end is None:
        game_end = game_end_ms(evs)
    e_h = int(e_h)
    t1 = e_h + int(window_ms)
    if game_end is not None and game_end <= e_h:
        return "game_ended_before_objective"
    cands = []
    for e in evs:
        ts = _ts(e)
        if ts is None or not (e_h < ts <= t1) or _etype(e) != "ELITE_MONSTER_KILL":
            continue
        if str(e.get("monsterType", "") or "") not in ELIGIBLE_OBJECTIVES:
            continue
        team = int(e.get("killerTeamId", 0) or 0)
        if team not in (100, 200):
            if skip_despawn and team == 300:
                continue
            team = int(tm.get(int(e.get("killerId", 0) or 0), 0))
        cands.append((ts, team))
    if not cands:
        if game_end is None:
            return "observation_censored"
        if e_h < game_end < t1:
            return "game_ended_in_window_no_objective"
        return "none"
    cands.sort()
    teams = {c[1] for c in cands if c[0] == cands[0][0]}
    if len(teams) != 1 or 0 in teams:
        return "tie_ambiguous"
    return "Blue" if next(iter(teams)) == 100 else "Red"


def next_objective_label(events: Iterable[Mapping], e_h: int, **kw: Any) -> Optional[int]:
    """1 if blue takes the first elite objective in (e_h, e_h + 180 s], 0 if red, else None."""
    o = next_objective_outcome(events, e_h, **kw)
    return 1 if o == "Blue" else 0 if o == "Red" else None


# ------------------------------------------------------------------ label end e_h (DR rule, unchanged)
_DR_MOD: Dict[str, Any] = {}


def load_dr_rules(path: Optional[Path] = None):
    """Import DR/engagement_labels_v3_rules.py unchanged (cached by path)."""
    p = Path(path or DEFAULT_DR_RULES).resolve()
    key = str(p)
    if key not in _DR_MOD:
        if not p.is_file():
            raise FileNotFoundError(f"endpoint rule module not found: {p}")
        spec = importlib.util.spec_from_file_location("engagement_labels_v3_rules", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DR_MOD[key] = mod
    return _DR_MOD[key]


def label_endpoint(last_kill_ms: int, events: Sequence[Mapping], *, h_s: int = 90,
                   next_start: Optional[int] = None, game_end: Optional[int] = None,
                   rules_path: Optional[Path] = None) -> Tuple[int, Tuple[str, ...]]:
    """e_h = min(L + h, next_kill - 1, next_start - 1, game_end - 1) via the DR rule module.

    next_start: next eligible engagement's tau (None -> absent); game_end: GAME_END timestamp
    (None -> read from events).  Returns (e_h, tied reasons)."""
    R = load_dr_rules(rules_path)
    kts = sorted(_ts(e) for e in raw_kill_events(events))
    ge = game_end if game_end is not None else game_end_ms(events)
    ge = R.INF if ge is None else int(ge)
    ns = R.INF if next_start is None else R.effective_next_start(int(next_start), ge)
    e, reasons, _ = R.endpoint_rule(int(last_kill_ms), int(h_s), R.next_kill_after(kts, int(last_kill_ms)), ns, ge)
    return int(e), tuple(reasons)
