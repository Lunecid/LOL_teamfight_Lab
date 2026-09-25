"""Team-level objective state at an exact time t, from Match-V5 events with timestamp <= t.

What it gives at t (milliseconds of game time):
  * seconds until the next Dragon (elemental or Elder), Rift Herald, Baron, Atakhan and Voidgrub
    spawn: 0.0 while the monster is up, NaN when it will not spawn again in this game;
  * Baron and Elder buff remaining seconds per player and per team.  A buff goes to the members of
    the slaying team who are alive at the kill (gameplay.event_survival.event_alive) and ends at
    min(kill + duration, the holder's next death);
  * inhibitor respawn remaining per inhibitor, nexus-turret respawn remaining, towers alive by tier,
    outer-turret plates left;
  * dragon counts per team and element, the Rift (soul) element and the soul holder.

Rules (spawn / respawn / despawn / buff durations) come from config/game_rules/objective_rules.json,
one entry per rule and patch range, each with sources and a status.  A rule with status
"uncertain" is still applied, but `uncertain_rules(patch)` lists it and the feature block carries
`rule_uncertain_<name>` = 1 for every timer that depends on it.

DRAGON_SOUL_GIVEN.  In 15.14 (500 matches) the event comes twice: teamId 0, 0.5 s after the 2nd
elemental drake of the game (the Rift transformation: `name` is the soul element for the rest of
the game), and teamId 100/200, 0.5 s after that team's 4th drake (the soul itself).  Older code
read every DRAGON_SOUL_GIVEN as a soul award (gameplay/labels.py prices it 1.0; tid 0 when teamId
is 0).  Here the element is taken from `name` of any DRAGON_SOUL_GIVEN and the holder from the
ELITE_MONSTER_KILL drake count reaching 4; the event's teamId is only used as a diagnostic.

Exactness.  Every query uses only events with timestamp <= t.  No minute frame, frame 'alive' /
node status column, position interpolation or 5-s grid is read.  `ObjectiveTimeline(events)`
built from all events and from events truncated at t give the same `state(t)` (tested).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from gameplay.event_survival import death_intervals, event_alive

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config/game_rules/objective_rules.json"

TEAMS = (100, 200)
SIDE = {100: "blue", 200: "red"}
LANES = ("TOP_LANE", "MID_LANE", "BOT_LANE")
LANE_SHORT = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}
TIERS = ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET")
TIER_SHORT = {"OUTER_TURRET": "outer", "INNER_TURRET": "inner", "BASE_TURRET": "base", "NEXUS_TURRET": "nexus"}
TIER_TOTAL = {"OUTER_TURRET": 3, "INNER_TURRET": 3, "BASE_TURRET": 3, "NEXUS_TURRET": 2}
ELEMENTS = ("AIR", "CHEMTECH", "EARTH", "FIRE", "HEXTECH", "WATER")
# DRAGON_SOUL_GIVEN `name` -> monsterSubType element (15.14: names always one of these six)
SOUL_NAME_TO_ELEMENT = {"cloud": "AIR", "chemtech": "CHEMTECH", "mountain": "EARTH", "infernal": "FIRE",
                        "hextech": "HEXTECH", "ocean": "WATER"}
OBJECTIVES = ("dragon", "herald", "baron", "atakhan", "voidgrub")


# ---------------------------------------------------------------------------- rule table
def _pkey(patch: str) -> Tuple[int, int]:
    major, minor = (int(x) for x in str(patch).split(".")[:2])
    return major, minor


@dataclass(frozen=True)
class Rule:
    name: str
    value: Optional[float]
    status: str
    patch_range: Tuple[str, str]
    sources: Tuple[str, ...]
    extra: Mapping

    @property
    def uncertain(self) -> bool:
        return self.status == "uncertain"


_RULES_CACHE: Dict[str, dict] = {}


def load_objective_rules(path: Optional[Path] = None) -> dict:
    p = str(path or RULES_PATH)
    if p not in _RULES_CACHE:
        _RULES_CACHE[p] = json.loads(Path(p).read_text(encoding="utf-8"))
    return _RULES_CACHE[p]


def rule(name: str, patch: str, rules: Optional[dict] = None) -> Rule:
    """The entry of rule `name` whose patch_range contains `patch` (Match-V5 major.minor).
    KeyError when no entry covers the patch: never extrapolate a rule to an undocumented patch."""
    table = (rules or load_objective_rules())["rules"]
    if name not in table:
        raise KeyError(f"unknown objective rule {name!r}")
    pk = _pkey(patch)
    for ent in table[name]:
        lo, hi = ent["patch_range"]
        if _pkey(lo) <= pk <= _pkey(hi):
            extra = {k: v for k, v in ent.items() if k not in ("patch_range", "value", "status", "sources")}
            return Rule(name, ent["value"], ent["status"], (lo, hi), tuple(ent.get("sources", ())), extra)
    raise KeyError(f"no objective rule {name!r} for patch {patch}")


def uncertain_rules(patch: str, rules: Optional[dict] = None) -> List[str]:
    table = (rules or load_objective_rules())["rules"]
    out = []
    for name in table:
        try:
            if rule(name, patch, rules).uncertain:
                out.append(name)
        except KeyError:
            continue
    return sorted(out)


# rules each objective timer depends on (for the rule_uncertain_<objective> flags)
TIMER_RULES = {
    "dragon": ("dragon_first_spawn_ms", "dragon_respawn_ms", "dragon_soul_count",
               "elder_first_spawn_after_soul_ms", "elder_respawn_ms"),
    "herald": ("herald_spawn_ms", "herald_despawn_ms", "herald_count"),
    "baron": ("baron_first_spawn_ms", "baron_respawn_ms"),
    "atakhan": ("atakhan_spawn_ms", "atakhan_count"),
    "voidgrub": ("voidgrub_spawn_ms", "voidgrub_despawn_ms", "voidgrub_count"),
    "baron_buff": ("baron_buff_ms",),
    "elder_buff": ("elder_buff_ms",),
    "structures": ("inhibitor_respawn_ms", "nexus_turret_respawn_ms", "turret_plates_per_outer",
                   "turret_plates_fall_ms"),
}


# ---------------------------------------------------------------------------- timeline
def _team_of(e: Mapping, key_team: str, team_map: Mapping[int, int]) -> Optional[int]:
    tid = int(e.get(key_team, 0) or 0)
    if tid in TEAMS:
        return tid
    kid = int(e.get("killerId", 0) or 0)
    return team_map.get(kid)


class ObjectiveTimeline:
    """Index a match's events once; query `state(t)` / `features(t)` at any ms t."""

    def __init__(self, events: Iterable[Mapping], patch: str, team_map: Optional[Mapping] = None,
                 rules: Optional[dict] = None):
        self.patch = str(patch)
        self.rules = rules or load_objective_rules()
        tm = {int(k): int(v) for k, v in (team_map or {}).items()}
        self.team_map = tm or {p: (100 if p <= 5 else 200) for p in range(1, 11)}
        R = lambda n: rule(n, self.patch, self.rules)  # noqa: E731
        self.r = {n: R(n) for n in self.rules["rules"]}
        self.events = sorted((dict(e) for e in events or []), key=lambda e: int(e.get("timestamp", 0)))
        # per-kind (ts, ...) lists, sorted by ts
        self.drakes: List[Tuple[int, Optional[int], str]] = []     # elemental: (ts, team, ELEMENT)
        self.elders: List[Tuple[int, Optional[int]]] = []
        self.monsters: Dict[str, List[Tuple[int, Optional[int]]]] = {k: [] for k in
                                                                    ("BARON_NASHOR", "RIFTHERALD", "ATAKHAN", "HORDE")}
        self.soul_events: List[Tuple[int, int, str]] = []           # (ts, teamId as given, name)
        self.building: List[Tuple[int, tuple]] = []                  # (ts, key)
        self.plates: List[Tuple[int, int, str]] = []                 # (ts, owner team, lane)
        self.deaths: Dict[int, List[int]] = {p: [] for p in range(1, 11)}
        self.unknown_team_elite = 0
        self.neutral_elite: List[Tuple[int, str]] = []               # killerTeamId 300 (no team credit)
        for e in self.events:
            et, ts = e.get("type"), int(e.get("timestamp", 0))
            if et == "ELITE_MONSTER_KILL":
                mt = str(e.get("monsterType", "")).upper()
                team = _team_of(e, "killerTeamId", self.team_map)
                if team is None:
                    self.unknown_team_elite += 1
                    if int(e.get("killerTeamId", 0) or 0) == 300:   # 15.14: logged despawn (14:45 grubs, 24:45 herald)
                        self.neutral_elite.append((ts, mt))
                if mt == "DRAGON":
                    sub = str(e.get("monsterSubType", "")).upper()
                    if sub == "ELDER_DRAGON":
                        self.elders.append((ts, team))
                    else:
                        self.drakes.append((ts, team, sub.replace("_DRAGON", "")))
                elif mt in self.monsters:
                    self.monsters[mt].append((ts, team))
            elif et == "DRAGON_SOUL_GIVEN":
                self.soul_events.append((ts, int(e.get("teamId", 0) or 0), str(e.get("name", ""))))
            elif et == "BUILDING_KILL":
                owner = int(e.get("teamId", 0) or 0)
                bt = str(e.get("buildingType", "")).upper()
                lane = str(e.get("laneType", "")).upper()
                if bt == "INHIBITOR_BUILDING":
                    key = ("INHIB", owner, lane)
                elif bt == "TOWER_BUILDING":
                    tt = str(e.get("towerType", "")).upper()
                    if tt == "NEXUS_TURRET":
                        pos = e.get("position") or {}
                        key = ("TOWER", owner, "NEXUS", (int(pos.get("x", 0)), int(pos.get("y", 0))))
                    else:
                        key = ("TOWER", owner, lane, tt)
                else:
                    continue
                self.building.append((ts, key))
            elif et == "TURRET_PLATE_DESTROYED":
                self.plates.append((ts, int(e.get("teamId", 0) or 0), str(e.get("laneType", "")).upper()))
            elif et == "CHAMPION_KILL":
                v = int(e.get("victimId", 0) or 0)
                if 1 <= v <= 10:
                    self.deaths[v].append(ts)
        self._ts_drakes = [x[0] for x in self.drakes]
        self._ts_elders = [x[0] for x in self.elders]
        self._ts_mon = {k: [x[0] for x in v] for k, v in self.monsters.items()}
        self._ts_soul = [x[0] for x in self.soul_events]
        self._ts_bld = [x[0] for x in self.building]
        self._ts_plates = [x[0] for x in self.plates]
        # soul drake: the kill that gives a team its N-th drake (derived from kills, not the event)
        need = int(self.r["dragon_soul_count"].value)
        cnt = {100: 0, 200: 0}
        self.soul_kill: Optional[Tuple[int, int]] = None
        for ts, team, _el in self.drakes:
            if team in cnt:
                cnt[team] += 1
                if cnt[team] == need and self.soul_kill is None:
                    self.soul_kill = (ts, team)
        # buff grants: (kill ts, team, holders), holders alive at the kill (events <= kill ts only)
        self.grants: Dict[str, List[Tuple[int, int, Tuple[int, ...]]]] = {"baron": [], "elder": []}
        for kind, kills in (("baron", self.monsters["BARON_NASHOR"]), ("elder", self.elders)):
            for ts, team in kills:
                if team not in TEAMS:
                    continue
                alive = event_alive(death_intervals(self.events, self.patch, t=ts), ts)
                holders = tuple(p for p in range(1, 11) if self.team_map.get(p) == team and alive[p - 1] > 0)
                self.grants[kind].append((ts, team, holders))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _n(ts_list: List[int], t: float) -> int:
        return bisect_right(ts_list, t)

    def _val(self, name: str) -> Optional[float]:
        return self.r[name].value

    # ------------------------------------------------------------------ spawn timers
    def timers(self, t: float) -> Dict[str, float]:
        """Seconds until next spawn (0 = up, NaN = never again) and up flags, from events <= t."""
        out: Dict[str, float] = {}
        nan = float("nan")

        # Dragon pit: elemental until a soul is claimed, Elder afterwards
        nd, ne = self._n(self._ts_drakes, t), self._n(self._ts_elders, t)
        last_d = self.drakes[nd - 1][0] if nd else None
        last_e = self.elders[ne - 1][0] if ne else None
        last = max((x for x in (last_d, last_e) if x is not None), default=None)
        soul_claimed = self.soul_kill is not None and self.soul_kill[0] <= t
        if last is None:
            spawn = self._val("dragon_first_spawn_ms")
        elif soul_claimed:
            elder_last = last_e is not None and last_e >= (last_d if last_d is not None else -1)
            spawn = last + (self._val("elder_respawn_ms") if elder_last else self._val("elder_first_spawn_after_soul_ms"))
        else:
            spawn = last + self._val("dragon_respawn_ms")
        out["dragon_next_spawn_s"] = max(0.0, (spawn - t) / 1000.0)
        out["dragon_up"] = float(t >= spawn)
        out["dragon_next_is_elder"] = float(soul_claimed)

        # Baron
        nb = self._n(self._ts_mon["BARON_NASHOR"], t)
        spawn = self._val("baron_first_spawn_ms") if nb == 0 else self.monsters["BARON_NASHOR"][nb - 1][0] + self._val("baron_respawn_ms")
        out["baron_next_spawn_s"] = max(0.0, (spawn - t) / 1000.0)
        out["baron_up"] = float(t >= spawn)

        # Rift Herald: once, between spawn and despawn
        nh = self._n(self._ts_mon["RIFTHERALD"], t)
        hs, hd = self._val("herald_spawn_ms"), self._val("herald_despawn_ms")
        if nh >= int(self._val("herald_count")) or t >= hd:
            out["herald_next_spawn_s"], out["herald_up"] = nan, 0.0
        else:
            out["herald_next_spawn_s"] = max(0.0, (hs - t) / 1000.0)
            out["herald_up"] = float(t >= hs)
        grace = float(self.r["herald_despawn_ms"].extra.get("combat_grace_ms", 0) or 0)
        out["herald_in_despawn_grace"] = float(nh == 0 and hd <= t < hd + grace)

        # Atakhan: once (absent from 16.1)
        na = self._n(self._ts_mon["ATAKHAN"], t)
        a_sp, a_n = self._val("atakhan_spawn_ms"), int(self._val("atakhan_count") or 0)
        if a_sp is None or na >= a_n:
            out["atakhan_next_spawn_s"], out["atakhan_up"] = nan, 0.0
        else:
            out["atakhan_next_spawn_s"] = max(0.0, (a_sp - t) / 1000.0)
            out["atakhan_up"] = float(t >= a_sp)
        out["atakhan_in_patch"] = float(a_n > 0)

        # Voidgrubs: one wave of `voidgrub_count`
        ng = self._n(self._ts_mon["HORDE"], t)
        gs, gd, gn = self._val("voidgrub_spawn_ms"), self._val("voidgrub_despawn_ms"), int(self._val("voidgrub_count"))
        if ng >= gn or t >= gd:
            out["voidgrub_next_spawn_s"], out["voidgrub_up"], out["voidgrub_left"] = nan, 0.0, 0.0
        else:
            out["voidgrub_next_spawn_s"] = max(0.0, (gs - t) / 1000.0)
            out["voidgrub_up"] = float(t >= gs)
            out["voidgrub_left"] = float(gn - ng) if t >= gs else 0.0
        grace = float(self.r["voidgrub_despawn_ms"].extra.get("combat_grace_ms", 0) or 0)
        out["voidgrub_in_despawn_grace"] = float(0 < gn - ng and gd <= t < gd + grace)
        return out

    # ------------------------------------------------------------------ buffs
    def buff_remaining_by_player(self, kind: str, t: float) -> np.ndarray:
        """(10,) seconds of Baron ('baron') or Elder ('elder') buff left at t (index = pid - 1)."""
        dur = float(self._val(f"{kind}_buff_ms"))
        out = np.zeros(10, dtype=float)
        for ts, _team, holders in self.grants[kind]:
            if ts > t:
                break
            for p in holders:
                end = ts + dur
                d = self.deaths[p]
                i = bisect_right(d, ts)             # first death strictly after the grant
                if i < len(d) and d[i] <= t:        # only deaths already known at t
                    end = min(end, d[i])
                out[p - 1] = max(out[p - 1], max(0.0, (end - t) / 1000.0))
        return out

    def buffs(self, t: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for kind in ("baron", "elder"):
            rem = self.buff_remaining_by_player(kind, t)
            for team in TEAMS:
                idx = [p - 1 for p in range(1, 11) if self.team_map.get(p) == team]
                r = rem[idx]
                out[f"{SIDE[team]}_{kind}_buff_s"] = float(r.max()) if len(r) else 0.0
                out[f"{SIDE[team]}_{kind}_buff_n"] = float((r > 0).sum())
        return out

    # ------------------------------------------------------------------ structures
    def structures(self, t: float) -> Dict[str, float]:
        inhib_resp = float(self._val("inhibitor_respawn_ms"))
        nexus_resp = float(self._val("nexus_turret_respawn_ms"))
        last_kill: Dict[tuple, int] = {}
        for ts, key in self.building[: self._n(self._ts_bld, t)]:
            last_kill[key] = ts
        out: Dict[str, float] = {}
        for team in TEAMS:
            s = SIDE[team]
            dead = {tt: 0 for tt in TIERS}
            for key, ts in last_kill.items():
                if key[0] != "TOWER" or key[1] != team:
                    continue
                if key[2] == "NEXUS":
                    dead["NEXUS_TURRET"] += int(t < ts + nexus_resp)
                else:
                    dead[key[3]] += 1
            for tt in TIERS:
                out[f"{s}_towers_{TIER_SHORT[tt]}"] = float(max(0, TIER_TOTAL[tt] - dead[tt]))
            nex_rem = [max(0.0, (ts + nexus_resp - t) / 1000.0) for key, ts in last_kill.items()
                       if key[0] == "TOWER" and key[1] == team and key[2] == "NEXUS"]
            out[f"{s}_nexus_turret_respawn_max_s"] = float(max(nex_rem, default=0.0))
            n_down = 0
            for lane in LANES:
                ts = last_kill.get(("INHIB", team, lane))
                rem = 0.0 if ts is None else max(0.0, (ts + inhib_resp - t) / 1000.0)
                out[f"{s}_inhib_{LANE_SHORT[lane]}_respawn_s"] = rem
                n_down += int(rem > 0)
            out[f"{s}_inhibs_down"] = float(n_down)
            # outer-turret plates left (0 once the outer turret falls or the plates fall off)
            fall = self._val("turret_plates_fall_ms")
            per = int(self._val("turret_plates_per_outer"))
            lost = {lane: 0 for lane in LANES}
            for ts, owner, lane in self.plates[: self._n(self._ts_plates, t)]:
                if owner == team and lane in lost:
                    lost[lane] += 1
            for lane in LANES:
                gone = ("TOWER", team, lane, "OUTER_TURRET") in last_kill or (fall is not None and t >= fall)
                out[f"{s}_plates_{LANE_SHORT[lane]}"] = 0.0 if gone else float(max(0, per - lost[lane]))
        return out

    # ------------------------------------------------------------------ dragons, soul
    def soul(self, t: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        nd, ne = self._n(self._ts_drakes, t), self._n(self._ts_elders, t)
        for team in TEAMS:
            s = SIDE[team]
            mine = [el for ts, tm, el in self.drakes[:nd] if tm == team]
            out[f"{s}_dragons"] = float(len(mine))
            for el in ELEMENTS:
                out[f"{s}_dragon_{el.lower()}"] = float(sum(1 for x in mine if x == el))
            out[f"{s}_elders"] = float(sum(1 for ts, tm in self.elders[:ne] if tm == team))
            out[f"{s}_has_soul"] = float(self.soul_kill is not None and self.soul_kill[0] <= t and self.soul_kill[1] == team)
            for mt, name in (("BARON_NASHOR", "barons"), ("RIFTHERALD", "heralds"), ("ATAKHAN", "atakhans"), ("HORDE", "voidgrubs")):
                n = self._n(self._ts_mon[mt], t)
                out[f"{s}_{name}"] = float(sum(1 for ts, tm in self.monsters[mt][:n] if tm == team))
        element = self.rift_element(t)
        out["rift_element_known"] = float(element is not None)
        for el in ELEMENTS:
            out[f"rift_element_{el.lower()}"] = float(element == el)
        return out

    def rift_element(self, t: float) -> Optional[str]:
        """Soul element announced by DRAGON_SOUL_GIVEN (any teamId) with timestamp <= t."""
        n = self._n(self._ts_soul, t)
        for ts, _tid, name in reversed(self.soul_events[:n]):
            el = SOUL_NAME_TO_ELEMENT.get(name.strip().lower())
            if el:
                return el
        return None

    def soul_holder(self, t: float) -> Optional[int]:
        return self.soul_kill[1] if (self.soul_kill is not None and self.soul_kill[0] <= t) else None

    def soul_event_diagnostics(self) -> Dict[str, object]:
        """Outcome-side check (reads the whole game): how the DRAGON_SOUL_GIVEN events line up with
        the drake kills.  Not a feature."""
        rows = []
        for ts, tid, name in self.soul_events:
            n_before = self._n(self._ts_drakes, ts)
            last = self.drakes[n_before - 1][0] if n_before else None
            rows.append(dict(ts=ts, teamId=tid, name=name, drakes_before=n_before,
                             lag_ms=None if last is None else ts - last))
        return dict(events=rows, derived_soul=self.soul_kill,
                    later_drake_elements=[el for _ts, _tm, el in self.drakes[2:]])

    # ------------------------------------------------------------------ all
    def state(self, t: float) -> Dict[str, float]:
        out = {"t_s": float(t) / 1000.0}
        out.update(self.timers(t))
        out.update(self.buffs(t))
        out.update(self.structures(t))
        out.update(self.soul(t))
        for grp, names in TIMER_RULES.items():
            out[f"rule_uncertain_{grp}"] = float(any(self.r[n].uncertain for n in names))
        return out

    def feature_names(self) -> List[str]:
        return list(self.state(0).keys())

    def features(self, t: float) -> np.ndarray:
        return np.asarray(list(self.state(t).values()), dtype=float)


def objective_state_at(events: Iterable[Mapping], patch: str, t: float,
                       team_map: Optional[Mapping] = None) -> Dict[str, float]:
    """One-shot convenience: ObjectiveTimeline(events <= t, ...).state(t)."""
    ev = [e for e in events or [] if int(e.get("timestamp", 0)) <= t]
    return ObjectiveTimeline(ev, patch, team_map).state(t)


def feature_names(patch: str = "15.14") -> List[str]:
    return ObjectiveTimeline([], patch).feature_names()
