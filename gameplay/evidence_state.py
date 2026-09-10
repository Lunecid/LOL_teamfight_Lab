"""State at an arbitrary millisecond from EVIDENCE, not from the last minute frame alone.

The timeline gives two kinds of data: minute frames (60 s, every stat) and events (ms, only what
happened).  The feature pipeline holds the last frame at the query time, which is right for a
quantity that only changes at events - as long as no event has happened since the frame - and
wrong for one that accrues continuously.  This module treats every field by its own nature:

  EVIDENCE     follows exactly from events <= tau.  Level (LEVEL_UP), ultimate rank
               (SKILL_LEVEL_UP), inventory (ITEM_*), alive / respawn (CHAMPION_KILL + timer),
               baron / elder buff (ELITE_MONSTER_KILL, lost on death), souls.
  EXTRAPOLATE  continuous accumulators: last frame + a rate from the PREVIOUS frames only
               + the discrete jumps events prove (kill bounties, shop transactions).
  FUSE         position: the most recent of frame, kill (victim exact; killer / assist near),
               objective kill (killer near) and shop / respawn (fountain), with age and kind.
  HOLD         fast-varying, no evidence: hp / mana between events, damage totals, cc time.
               A respawn since the frame resets hp and mana; death zeroes them.
  STATIC       champion, spells, runes.

Nothing after tau is ever read: frames are indexed <= tau and events filtered <= tau.  Forward
interpolation toward the engagement's own first kill would leak the kill and is not done.

Outputs use the SAME names and normalisation as the cached frame (core.config.NODE_FEATURE_NAMES,
pipeline_cache denominators) so a corrected value can be compared with, or substituted for, the
held one directly.  Where a reconstruction is an approximation it is named as such in
FIELD_CLASSES and the module never claims a held field is corrected.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import CS_DENOM, MAP_MAX, NODE_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DD_DIR = ROOT / "config/game_rules/datadragon"

# pipeline_cache.py denominators (deterministic, not fitted)
DEN = dict(level=18.0, xp=20_000.0, cur_g=4_000.0, tot_g=25_000.0, gps=30.0, lane_cs=400.0,
           jg_cs=250.0, cc=600.0, coord=float(MAP_MAX), ult=3.0)
BARON_MS, ELDER_MS = 180_000, 150_000
# Summoner's Rift fountains in raw map units (approximate; used only as shop / respawn evidence)
FOUNTAIN = {100: (550.0, 580.0), 200: (14_300.0, 14_400.0)}
ORNN = 516                                   # can shop anywhere: a purchase is not position evidence
# Base respawn seconds by level (1..18) and the post-25-minute time-increase factor.  Approximate.
RESPAWN_BASE_S = [10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5]
# Cumulative experience needed to reach each level (index = level).  Fixed table.
XP_AT_LEVEL = [0, 0, 280, 660, 1140, 1720, 2400, 3180, 4060, 5040, 6120, 7300, 8580, 9960, 11440,
               13020, 14700, 16480, 18360]
LEVEL_GROWTH_STATS = ("healthMax", "powerMax", "armor", "magicResist", "attackDamage", "healthRegen", "powerRegen")

FIELD_CLASSES = {
    "EVIDENCE": ["level_norm", "ult_level_norm", "alive", "has_baron", "has_elder", "baron_remain_norm",
                 "elder_remain_norm", "soul_*", "items (new)", "item_gold_owned (new)"],
    "EXTRAPOLATE": ["totalGold_norm", "curGold_norm", "xp_norm", "laneCS_norm", "jgCS_norm", "gps_norm"],
    "FUSE": ["x_norm", "y_norm"],
    "EVIDENCE_APPROX": ["cs_attackDamage", "cs_abilityPower", "cs_armor", "cs_magicResist", "cs_healthMax",
                        "cs_health", "cs_powerMax", "cs_power", "cs_movementSpeed", "cs_attackSpeed",
                        "cs_healthRegen", "cs_powerRegen", "cs_lifesteal",
                        "(frame + item stat deltas + level growth; unique passives, haste, penetration, omnivamp not covered)"],
    "HOLD": ["hp_pct / mp_pct between events", "cs_abilityHaste", "cs_armorPen*", "cs_magicPen*",
             "cs_omnivamp", "cs_physicalVamp", "cs_spellVamp", "cs_ccReduction", "cs_cooldownReduction",
             "ds_* (damage totals)", "ccTime_norm"],
    "STATIC": ["champion_id", "champion_name_id", "summoner_spell_*", "*_style_id", "*_rune_*", "stat_perk_*"],
}

POS_KIND = {"frame": 0, "victim": 1, "killer": 2, "assist": 3, "objective": 4, "shop": 5, "respawn": 6, "dead": 7}
# Mean position error (map units) of each evidence kind by its age, in 10 s bins 0-10 .. 50-60+,
# measured leak-free against the next frame on 1,000 held-out matches (scripts/audit_evidence_state.py,
# outputs/evidence_state/audit_1000.json).  The fusion picks the candidate with the smallest expected
# error, so a fresh kill beats a stale frame and a stale shop visit never beats anything.  The frame
# curve is monotonised; its 50-60 s bin (n = 58,978) is the reliable one.
POS_ERROR_CURVE = {
    "frame":     [750, 2200, 3300, 3900, 3950, 3950],
    "victim":    [1157, 2591, 3649, 4354, 4696, 5045],    # same geometry as a killer at the kill
    "killer":    [1157, 2591, 3649, 4354, 4696, 5045],
    "assist":    [1321, 2718, 3859, 4460, 4863, 5201],
    "objective": [1200, 2584, 3719, 4373, 4499, 4955],
    "shop":      [5833, 8232, 9593, 10010, 10126, 10267],
    "respawn":   [2803, 6360, 8931, 9807, 9920, 9999],
}


def expected_position_error(kind: str, age_s: float) -> float:
    curve = POS_ERROR_CURVE[kind]
    return float(curve[min(len(curve) - 1, max(0, int(age_s // 10)))])


def respawn_seconds(level: int, game_ms: int) -> float:
    base = RESPAWN_BASE_S[max(1, min(18, int(level))) - 1]
    t = game_ms / 1000.0
    if t < 1500:
        tif = 0.0
    elif t < 2100:
        tif = 0.00425 * ((t - 1500) / 15.0)
    else:
        tif = min(0.50, 0.17 + 0.003 * ((t - 2100) / 15.0))
    return base * (1.0 + tif)


def level_growth(level: int) -> float:
    n = max(0, int(level) - 1)
    return n * (0.7025 + 0.0175 * n)


def load_tables(patch: str) -> Tuple[dict, dict]:
    items = json.loads((DD_DIR / f"items_{patch}.json").read_text(encoding="utf-8"))["items"]
    champs = json.loads((DD_DIR / f"champions_{patch}.json").read_text(encoding="utf-8"))["champions"]
    return {int(k): v for k, v in items.items()}, {int(k): v for k, v in champs.items()}


@dataclass
class Evidence:
    """Per-participant event index, all timestamps in ms, all lists sorted by time."""
    deaths: List[Tuple[int, float, float, int]] = field(default_factory=list)      # ts, x, y, level_at_death
    kills: List[Tuple[int, float, float, float]] = field(default_factory=list)     # ts, x, y, gold
    assists: List[Tuple[int, float, float]] = field(default_factory=list)
    objectives: List[Tuple[int, float, float]] = field(default_factory=list)      # killer at objective / building / plate
    shop: List[Tuple[int, str, int, float, int, int]] = field(default_factory=list)  # ts, kind, item, gold_delta, before, after
    level_ups: List[Tuple[int, int]] = field(default_factory=list)
    ult_ups: List[int] = field(default_factory=list)


class EvidenceStateBuilder:
    def __init__(self, pack: dict, patch: Optional[str] = None, tables: Optional[Tuple[dict, dict]] = None):
        self.pack = pack
        self.ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        self.node = np.asarray(pack["node_minute"])
        self.xy = np.asarray(pack["xy_raw_minute"], dtype=np.float64)
        self.idx = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
        self.tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
        self.champion = {int(k): int(v) for k, v in (pack["meta"].get("static_meta", {}).get("champion_by_pid") or {}).items()}
        patch = patch or str(pack["meta"].get("patch"))
        self.items, self.champs = tables if tables else load_tables(patch)
        self.ev = {pid: Evidence() for pid in range(1, 11)}
        self.team_buff: Dict[str, List[Tuple[int, int]]] = {"baron": [], "elder": []}   # (ts, team)
        self.souls: Dict[int, List[Tuple[int, str]]] = {100: [], 200: []}
        self._index_events()

    # ------------------------------------------------------------------ indexing
    def _index_events(self):
        level_now = {pid: 1 for pid in range(1, 11)}
        for e in sorted(self.pack["events"], key=lambda e: int(e.get("timestamp", 0))):
            typ, ts = e.get("type"), int(e.get("timestamp", 0))
            if typ == "LEVEL_UP":
                pid = int(e.get("participantId", 0))
                if pid in self.ev:
                    level_now[pid] = int(e.get("level", level_now[pid]))
                    self.ev[pid].level_ups.append((ts, level_now[pid]))
            elif typ == "SKILL_LEVEL_UP":
                pid = int(e.get("participantId", 0))
                if pid in self.ev and int(e.get("skillSlot", 0)) == 4:
                    self.ev[pid].ult_ups.append(ts)
            elif typ == "CHAMPION_KILL":
                pos = e.get("position") or {}
                x, y = float(pos.get("x", np.nan)), float(pos.get("y", np.nan))
                victim, killer = int(e.get("victimId", 0)), int(e.get("killerId", 0))
                if victim in self.ev:
                    self.ev[victim].deaths.append((ts, x, y, level_now[victim]))
                if killer in self.ev:
                    self.ev[killer].kills.append((ts, x, y, float(e.get("bounty", 0) or 0) + float(e.get("shutdownBounty", 0) or 0)))
                for a in e.get("assistingParticipantIds") or []:
                    if int(a) in self.ev:
                        self.ev[int(a)].assists.append((ts, x, y))
            elif typ in ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                pos = e.get("position") or {}
                killer = int(e.get("killerId", 0) or 0)
                if killer in self.ev and pos:
                    self.ev[killer].objectives.append((ts, float(pos["x"]), float(pos["y"])))
                if typ == "ELITE_MONSTER_KILL":
                    team = int(e.get("killerTeamId", 0) or self.tm.get(killer, 0))
                    mt, st = e.get("monsterType"), str(e.get("monsterSubType", "")).upper()
                    if team in (100, 200):
                        if mt == "BARON_NASHOR":
                            self.team_buff["baron"].append((ts, team))
                        elif mt == "DRAGON" and st == "ELDER_DRAGON":
                            self.team_buff["elder"].append((ts, team))
            elif typ == "DRAGON_SOUL_GIVEN":
                team = int(e.get("teamId", 0) or 0)
                if team in self.souls:
                    self.souls[team].append((ts, str(e.get("name", e.get("dragonSoul", ""))).lower()))
            elif typ in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO", "ITEM_DESTROYED"):
                pid = int(e.get("participantId", 0))
                if pid not in self.ev:
                    continue
                iid = int(e.get("itemId", 0) or 0)
                if typ == "ITEM_PURCHASED":
                    self.ev[pid].shop.append((ts, "buy", iid, 0.0, 0, 0))     # gold resolved at replay
                elif typ == "ITEM_SOLD":
                    self.ev[pid].shop.append((ts, "sell", iid, float(self.items.get(iid, {}).get("gold_sell", 0)), 0, 0))
                elif typ == "ITEM_UNDO":
                    self.ev[pid].shop.append((ts, "undo", 0, float(e.get("goldGain", 0) or 0),
                                              int(e.get("beforeId", 0) or 0), int(e.get("afterId", 0) or 0)))
                else:
                    self.ev[pid].shop.append((ts, "destroy", iid, 0.0, 0, 0))

    # ------------------------------------------------------------------ helpers
    def frame_index(self, tau: int) -> int:
        i = int(bisect_right(self.ts, tau)) - 1
        if i < 0:
            raise ValueError("query precedes the first frame")
        return i

    def _replay_inventory(self, pid: int, upto: int, since: Optional[int] = None):
        """Inventory at `upto`; also the gold delta and the stat delta of transactions in (since, upto]."""
        inv: List[int] = []
        gold_delta, stat_delta = 0.0, defaultdict(float)
        destroyed_at: Dict[int, set] = defaultdict(set)
        for ts, kind, iid, gold, before, after in self.ev[pid].shop:
            if ts > upto:
                break
            if kind == "destroy":
                destroyed_at[ts].add(iid)
        for ts, kind, iid, gold, before, after in self.ev[pid].shop:
            if ts > upto:
                break
            added, removed, delta_gold = [], [], 0.0
            if kind == "buy":
                it = self.items.get(iid, {})
                components = set(it.get("from", []))
                paid = it.get("gold_base", 0) if (components & destroyed_at.get(ts, set())) else it.get("gold_total", 0)
                added, delta_gold = [iid], -float(paid)
            elif kind == "sell":
                removed, delta_gold = [iid], gold
            elif kind == "undo":
                removed, added, delta_gold = ([before] if before else []), ([after] if after else []), gold
            else:                                                       # destroy: consumed / combined
                removed = [iid]
            for x in added:
                inv.append(x)
            for x in removed:
                if x in inv:
                    inv.remove(x)
            if since is not None and ts > since:
                gold_delta += delta_gold
                for x in added:
                    for k, v in self.items.get(x, {}).get("stats", {}).items():
                        stat_delta[k] += v
                for x in removed:
                    for k, v in self.items.get(x, {}).get("stats", {}).items():
                        stat_delta[k] -= v
        return inv, gold_delta, stat_delta

    def _level_at(self, pid: int, t: int) -> int:
        lv = 1
        for ts, level in self.ev[pid].level_ups:
            if ts > t:
                break
            lv = level
        return lv

    def _death_state(self, pid: int, tau: int):
        """(dead_at_tau, last_death_ts, respawn_ts) from CHAMPION_KILL events <= tau."""
        last = None
        for ts, x, y, lv in self.ev[pid].deaths:
            if ts > tau:
                break
            last = (ts, lv)
        if last is None:
            return False, None, None
        d, lv = last
        # the last frame at or before the death gives a lower bound when LEVEL_UP events are missing
        fi = int(bisect_right(self.ts, d)) - 1
        if fi >= 0:
            lv = max(lv, int(round(float(self.node[fi, pid - 1, self.idx["level_norm"]]) * DEN["level"])))
        r = d + int(round(respawn_seconds(lv, d) * 1000))
        return tau < r, d, r

    def _buff(self, pid: int, tau: int, kind: str, dur_ms: int) -> Tuple[float, float]:
        team = self.tm[pid]
        acq = [ts for ts, t in self.team_buff[kind] if t == team and ts <= tau]
        if not acq:
            return 0.0, 0.0
        t0 = acq[-1]
        # the buff is lost on death; also not held if the champion was dead when it was taken
        if any(t0 < d <= tau for d, *_ in self.ev[pid].deaths):
            return 0.0, 0.0
        dead0, _, r0 = self._death_state(pid, t0)
        if dead0:
            return 0.0, 0.0
        remain = t0 + dur_ms - tau
        return (1.0, float(np.clip(remain / dur_ms, 0.0, 1.0))) if remain > 0 else (0.0, 0.0)

    def _position(self, pid: int, tau: int, i: int):
        """Most recent position evidence <= tau.  Returns (x, y, kind, age_s)."""
        cands = [(int(self.ts[i]), float(self.xy[i, pid - 1, 0]), float(self.xy[i, pid - 1, 1]), "frame")]
        e = self.ev[pid]
        fx, fy = FOUNTAIN[self.tm[pid]]
        dead, d, r = self._death_state(pid, tau)
        if dead:
            # a dead champion cannot take part; keep the frame position and let alive=0 carry it
            return cands[0][1], cands[0][2], "dead", (tau - cands[0][0]) / 1000.0
        if r is not None and r <= tau:
            cands.append((r, fx, fy, "respawn"))
        # a death position is only meaningful while dead (handled above); after the respawn the
        # champion is at the fountain, so the death location is never a candidate here
        for ts, x, y, g in e.kills:
            if ts <= tau and np.isfinite(x):
                cands.append((ts, x, y, "killer"))
        for ts, x, y in e.assists:
            if ts <= tau and np.isfinite(x):
                cands.append((ts, x, y, "assist"))
        for ts, x, y in e.objectives:
            if ts <= tau:
                cands.append((ts, x, y, "objective"))
        if self.champion.get(pid) != ORNN:
            for ts, kind, *_ in e.shop:
                if 0 < ts <= tau:
                    cands.append((ts, fx, fy, "shop"))
        # choose the candidate whose kind-and-age expected error is smallest; ties go to the fresher one
        ts, x, y, kind = min(cands, key=lambda c: (expected_position_error(c[3], (tau - c[0]) / 1000.0), -c[0]))
        return x, y, kind, (tau - ts) / 1000.0

    # ------------------------------------------------------------------ main
    def at(self, tau: int) -> Dict[int, Dict[str, float]]:
        i = self.frame_index(tau)
        fts = int(self.ts[i])
        age_s = (tau - fts) / 1000.0
        out: Dict[int, Dict[str, float]] = {}
        for pid in range(1, 11):
            fr = self.node[i, pid - 1]
            g = lambda n: float(fr[self.idx[n]])
            team = self.tm[pid]
            v: Dict[str, float] = {"frame_age_s": age_s}
            # --- EVIDENCE: level, ult, alive, buffs, souls
            lv_frame = int(round(g("level_norm") * DEN["level"]))
            lv = max(lv_frame, self._level_at(pid, tau))
            v["level_norm"] = lv / DEN["level"]
            ult = sum(1 for ts in self.ev[pid].ult_ups if ts <= tau)
            v["ult_level_norm"] = min(ult, 3) / DEN["ult"]
            dead, d, r = self._death_state(pid, tau)
            v["alive"] = 0.0 if dead else 1.0
            v["respawn_in_s"] = ((r - tau) / 1000.0) if dead else 0.0
            for kind, dur, name in (("baron", BARON_MS, "baron"), ("elder", ELDER_MS, "elder")):
                has, rem = self._buff(pid, tau, kind, dur)
                v[f"has_{name}"], v[f"{name}_remain_norm"] = has, rem
            soul = next((s for ts, s in reversed(self.souls[team]) if ts <= tau), None)
            for st in ("infernal", "ocean", "mountain", "cloud", "hextech", "chemtech"):
                v[f"soul_{st}"] = 1.0 if soul and st in soul else 0.0
            inv, gold_delta, stat_delta = self._replay_inventory(pid, tau, since=fts)
            v["item_count"] = float(len(inv))
            v["item_gold_owned"] = float(sum(self.items.get(x, {}).get("gold_total", 0) for x in inv)) / DEN["tot_g"]
            # --- EXTRAPOLATE: continuous accumulators from previous frames only
            rate = {}
            for key, den in (("totalGold_norm", DEN["tot_g"]), ("xp_norm", DEN["xp"]),
                             ("laneCS_norm", DEN["lane_cs"]), ("jgCS_norm", DEN["jg_cs"])):
                if i >= 1:
                    dt = (self.ts[i] - self.ts[i - 1]) / 1000.0
                    prev = float(self.node[i - 1, pid - 1, self.idx[key]])
                    rate[key] = max(0.0, (g(key) - prev) * den / dt) if dt > 0 else 0.0
                else:
                    rate[key] = 0.0
            bounty = sum(gold for ts, x, y, gold in self.ev[pid].kills if fts < ts <= tau)
            tot_g = g("totalGold_norm") * DEN["tot_g"] + rate["totalGold_norm"] * age_s + bounty
            cur_g = g("curGold_norm") * DEN["cur_g"] + rate["totalGold_norm"] * age_s + bounty + gold_delta
            v["totalGold_norm"] = tot_g / DEN["tot_g"]
            v["curGold_norm"] = max(0.0, cur_g) / DEN["cur_g"]
            xp = g("xp_norm") * DEN["xp"] + rate["xp_norm"] * age_s
            if lv > lv_frame:
                xp = max(xp, XP_AT_LEVEL[min(lv, 18)])            # a level-up proves at least this much
            xp = min(xp, XP_AT_LEVEL[min(lv + 1, 18)] - 1) if lv < 18 else xp
            v["xp_norm"] = xp / DEN["xp"]
            v["laneCS_norm"] = (g("laneCS_norm") * DEN["lane_cs"] + rate["laneCS_norm"] * age_s) / DEN["lane_cs"]
            v["jgCS_norm"] = (g("jgCS_norm") * DEN["jg_cs"] + rate["jgCS_norm"] * age_s) / DEN["jg_cs"]
            v["gps_norm"] = g("gps_norm")
            # --- EVIDENCE_APPROX: champion stats = frame + item deltas + level growth
            ch = self.champs.get(self.champion.get(pid, -1), {})
            growth = ch.get("growth", {})
            dg = level_growth(lv) - level_growth(lv_frame)
            base_as = float(ch.get("base", {}).get("attackSpeed", 0.0))
            for stat in ("attackDamage", "abilityPower", "armor", "magicResist", "healthMax", "powerMax",
                         "movementSpeed", "healthRegen", "powerRegen", "lifesteal"):
                raw = g(f"cs_{stat}") * CS_DENOM[stat] + stat_delta.get(stat, 0.0)
                if stat in growth and stat in LEVEL_GROWTH_STATS:
                    raw += float(growth[stat]) * dg
                v[f"cs_{stat}"] = raw / CS_DENOM[stat]
            v["cs_movementSpeed"] += (g("cs_movementSpeed") * stat_delta.get("movementSpeed_pct", 0.0)) / CS_DENOM["movementSpeed"]
            as_raw = g("cs_attackSpeed") * CS_DENOM["attackSpeed"] + base_as * stat_delta.get("attackSpeed_pct", 0.0)
            if "attackSpeed_pct" in growth:
                as_raw += base_as * float(growth["attackSpeed_pct"]) / 100.0 * dg
            v["cs_attackSpeed"] = as_raw / CS_DENOM["attackSpeed"]
            # current health / mana: a flat max increase raises the current pool by the same amount
            v["cs_health"] = (g("cs_health") * CS_DENOM["health"] + stat_delta.get("healthMax", 0.0)) / CS_DENOM["health"]
            v["cs_power"] = (g("cs_power") * CS_DENOM["power"] + stat_delta.get("powerMax", 0.0)) / CS_DENOM["power"]
            # --- HOLD with event resets
            respawned_since = (r is not None) and (fts < r <= tau)
            if dead:
                v["hp_pct"], v["mp_pct"], v["cs_health"] = 0.0, 0.0, 0.0
            elif respawned_since:
                v["hp_pct"], v["mp_pct"] = 1.0, 1.0
                v["cs_health"], v["cs_power"] = v["cs_healthMax"] * CS_DENOM["healthMax"] / CS_DENOM["health"], v["cs_powerMax"] * CS_DENOM["powerMax"] / CS_DENOM["power"]
            else:
                v["hp_pct"], v["mp_pct"] = g("hp_pct"), g("mp_pct")
            # --- FUSE: position
            x, y, kind, page = self._position(pid, tau, i)
            v["x_norm"], v["y_norm"] = x / DEN["coord"], y / DEN["coord"]
            v["pos_evidence_kind"], v["pos_evidence_age_s"] = float(POS_KIND[kind]), page
            out[pid] = v
        return out

    def frame_values(self, i: int, pid: int) -> Dict[str, float]:
        """The raw held frame for a participant, same names, for audits."""
        fr = self.node[i, pid - 1]
        return {n: float(fr[j]) for n, j in self.idx.items()}
