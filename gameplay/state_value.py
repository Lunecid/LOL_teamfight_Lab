"""Causal, event-aware states for an auxiliary match-outcome value model.

Acquisition ages and subsequent deaths are observable buff-history proxies, not
claims that a buff is currently active. Legacy cached buff flags are deliberately
unused: their builder updates them a frame late and does not clear on death.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass

import numpy as np

STATE_VERSION = "objective_history_v1"
DRAGONS = ("AIR", "EARTH", "FIRE", "WATER", "HEXTECH", "CHEMTECH", "OTHER")
OBJECTIVES = ("baron", "elder", "herald", "horde", "atakhan")
SNAPSHOT_FIELDS = ("totalGold_norm", "curGold_norm", "level_norm", "xp_norm",
                   "hp_pct", "mp_pct", "alive", "laneCS_norm", "jgCS_norm")


@dataclass
class State:
    values: dict[str, float]
    query_ms: int
    snapshot_ms: int


def final_outcome(events: list[dict]) -> tuple[int, int]:
    """Read the terminal target separately; never include it in a state."""
    ends = [e for e in events if e.get("type") == "GAME_END"]
    winners = {int(e.get("winningTeam", 0)) for e in ends}
    if len(winners) != 1 or not winners.issubset({100, 200}):
        raise ValueError("missing or conflicting GAME_END winningTeam")
    return int(next(iter(winners)) == 100), min(int(e["timestamp"]) for e in ends)


class StateBuilder:
    def __init__(self, pack: dict, node_names: list[str]):
        self.pack = pack
        self.ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        self.node = np.asarray(pack["node_minute"])
        self.idx = {n: i for i, n in enumerate(node_names)}
        required = set(SNAPSHOT_FIELDS) | {"champion_id"}
        if not required.issubset(self.idx) or self.node.shape != (len(self.ts), 10, len(node_names)):
            raise ValueError("unsupported node schema")
        if not len(self.ts) or np.any(np.diff(self.ts) <= 0):
            raise ValueError("invalid snapshot timestamps")
        self.tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        self.slots = {int(k): int(v) for k, v in (pack["meta"].get("role_slots") or {}).items()}
        if set(self.tm) != set(range(1, 11)) or any(list(self.tm.values()).count(t) != 5 for t in (100, 200)):
            raise ValueError("invalid participant teams")
        if set(self.slots) != set(self.tm) or set(self.slots.values()) != set(range(10)):
            raise ValueError("missing or non-bijective role slots")
        if any((self.slots[p] < 5) != (self.tm[p] == 100) for p in self.tm):
            raise ValueError("role slots disagree with team map")
        self.events = sorted((e for e in pack["events"] if e.get("type") in {
            "CHAMPION_KILL", "ELITE_MONSTER_KILL", "DRAGON_SOUL_GIVEN",
            "BUILDING_KILL", "TURRET_PLATE_DESTROYED"}), key=lambda e: int(e["timestamp"]))
        self.event_ts = [int(e["timestamp"]) for e in self.events]

    def at(self, query_ms: int) -> State:
        q = int(query_ms)
        i = int(np.searchsorted(self.ts, q, side="right") - 1)
        if i < 0 or q > int(self.ts[-1]):
            raise ValueError("query outside observed match")
        snapshot_ms = int(self.ts[i])
        out = {"time_minutes": q / 60000., "time_minutes_sq": (q / 60000.) ** 2,
               "snapshot_age_s": (q - snapshot_ms) / 1000.}
        counts = {t: Counter() for t in (100, 200)}
        kills, deaths = {p: [] for p in self.tm}, {p: [] for p in self.tm}
        acquisitions = {t: {o: [] for o in OBJECTIVES} for t in (100, 200)}
        souls = {t: set() for t in (100, 200)}
        unknown_team = 0
        for e in self.events[:bisect_right(self.event_ts, q)]:
            typ, ts = e["type"], int(e["timestamp"])
            killer = int(e.get("killerId", 0) or 0)
            if typ == "CHAMPION_KILL":
                victim = int(e.get("victimId", 0) or 0)
                if killer in kills:
                    kills[killer].append(ts)
                    counts[self.tm[killer]]["kills"] += 1
                if victim in deaths:
                    deaths[victim].append(ts)
                continue
            if typ == "ELITE_MONSTER_KILL":
                team = int(e.get("killerTeamId", 0) or self.tm.get(killer, 0))
                if team not in counts:
                    unknown_team += 1
                    continue
                monster = e.get("monsterType", "")
                sub = str(e.get("monsterSubType", "")).upper()
                if monster == "DRAGON" and sub != "ELDER_DRAGON":
                    kind = sub[:-7] if sub.endswith("_DRAGON") else sub
                    kind = kind if kind in DRAGONS else "OTHER"
                    counts[team]["dragon_" + kind] += 1
                    counts[team]["dragons"] += 1
                else:
                    obj = {"BARON_NASHOR": "baron", "RIFTHERALD": "herald", "HORDE": "horde",
                           "ATAKHAN": "atakhan", "DRAGON": "elder"}.get(monster)
                    if obj:
                        acquisitions[team][obj].append(ts)
                        counts[team][obj] += 1
            elif typ == "DRAGON_SOUL_GIVEN":
                team = int(e.get("teamId", 0) or 0)
                if team in counts:
                    soul = str(e.get("dragonSoul", e.get("soulType", "OTHER"))).upper()
                    soul = {"INFERNAL": "FIRE", "CLOUD": "AIR", "OCEAN": "WATER", "MOUNTAIN": "EARTH"}.get(soul, soul)
                    souls[team].add(soul if soul in DRAGONS else "OTHER")
                else:
                    unknown_team += 1
            elif typ in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                lost = int(e.get("teamId", 0) or 0)
                if lost not in counts:
                    unknown_team += 1
                    continue
                team = 300 - lost
                if typ == "TURRET_PLATE_DESTROYED":
                    counts[team]["plates"] += 1
                elif e.get("buildingType") == "INHIBITOR_BUILDING":
                    counts[team]["inhibitor_kills"] += 1
                else:
                    tower = str(e.get("towerType", "OTHER"))
                    tower = tower if tower in ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET") else "OTHER"
                    counts[team]["tower_" + tower] += 1
        out["unknown_objective_team_count"] = float(unknown_team)
        for pid in sorted(self.tm, key=self.slots.get):
            slot = self.slots[pid]
            prefix = f"slot{slot}_"
            for name in (*SNAPSHOT_FIELDS, "champion_id"):
                out[prefix + name] = float(self.node[i, pid - 1, self.idx[name]])
            out[prefix + "kills"] = float(len(kills[pid]))
            out[prefix + "deaths"] = float(len(deaths[pid]))
            out[prefix + "death_since_snapshot"] = float(any(t > snapshot_ms for t in deaths[pid]))
            out[prefix + "death_last_30s"] = float(any(t > q - 30000 for t in deaths[pid]))
            out[prefix + "death_age_minutes"] = min((q - deaths[pid][-1]) / 60000., 10.) if deaths[pid] else 10.
            # This is recorded history, not an assertion of current buff ownership.
            for obj in ("baron", "elder"):
                acq = acquisitions[self.tm[pid]][obj]
                out[prefix + obj + "_death_since_acquisition"] = float(bool(acq) and any(t >= acq[-1] for t in deaths[pid]))
        for team, prefix in ((100, "blue_"), (200, "red_")):
            names = ["kills", "dragons", "plates", "inhibitor_kills", *OBJECTIVES,
                     *("dragon_" + d for d in DRAGONS),
                     *("tower_" + t for t in ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET", "OTHER"))]
            for name in names:
                out[prefix + name] = float(counts[team][name])
            for d in DRAGONS:
                out[prefix + "soul_" + d] = float(d in souls[team])
            out[prefix + "soul_event_recorded"] = float(bool(souls[team]))
            for obj in OBJECTIVES:
                history = acquisitions[team][obj]
                age = (q - history[-1]) / 1000. if history else float("inf")
                out[prefix + obj + "_ever"] = float(bool(history))
                out[prefix + obj + "_age_minutes"] = min(age / 60., 10.)
                if obj in ("baron", "elder"):
                    for seconds in (60, 120, 180, 300):
                        out[f"{prefix}{obj}_acquired_last_{seconds}s"] = float(age < seconds)
        # A few explicit phase interactions; coefficients, not rewards, are learned.
        for key, value in list(out.items()):
            if key.startswith(("blue_", "red_")):
                out[key + "_x_time"] = value * q / 1800000.
        if not all(np.isfinite(v) for v in out.values()):
            raise ValueError("non-finite state")
        return State(out, q, snapshot_ms)


def value_labels(pre: np.ndarray, post: np.ndarray, valid=None):
    pre, post = np.asarray(pre), np.asarray(post)
    if pre.shape != post.shape:
        raise ValueError("before/after row mismatch")
    ok = np.isfinite(pre) & np.isfinite(post) & (pre >= 0) & (pre <= 1) & (post >= 0) & (post <= 1)
    if valid is not None:
        if np.shape(valid) != pre.shape:
            raise ValueError("valid mask row mismatch")
        ok &= np.asarray(valid, dtype=bool)
    delta = post - pre
    y = np.full(pre.shape, -1, dtype=np.int8)
    y[ok & (delta > 0)] = 1
    y[ok & (delta < 0)] = 0
    reason = np.where(~ok, "invalid_state_or_value", np.where(delta == 0, "tie", "labelled"))
    return y, delta, reason
