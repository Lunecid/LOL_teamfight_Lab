"""StateV3: the exact-data match state at a millisecond t (v4-exact plan, section 3; stage 1 task C).

Contract
--------
``StateBuilderV3(pack, patch).at(t)`` uses only
  * the last minute frame with timestamp <= t (no interpolation, no 5-s grid), and only its
    totalGold / curGold / level / xp / hp / mp / laneCS / jgCS node columns;
  * events with timestamp <= t;
  * pre-game static data (team map, champion ids, summoner spells, runes) and the patch tables
    (Data Dragon v2, objective rules, respawn rules).
It never reads the frame 'alive' column, the node status / cs_* / ds_* columns or xy_raw_minute.
(Positional *features* live in gameplay.setup_features and are only built when ``at(t, setup=True)``;
the optional role-inference metadata (``role_check=True``) reads the node x_norm / y_norm columns of
frames <= min(t, 8:59.999).)

Blocks, in column order (``STATE_V3_COLUMNS``; sha256 of the newline-joined names is
``STATE_V3_NAME_HASH``):
  global   StateV2 time_minutes, time_minutes_sq, snapshot_age_s (V only, see V_ONLY_COLUMNS),
           unknown_objective_team_count.
  player   10 blocks '<side>_<role>_<field>' in role-slot order (blue TOP JUNGLE MIDDLE BOTTOM
           UTILITY, then red), slots from participant order (``participant_slots``: team_map sorted
           by (team, participant id), i.e. blue pids 1-5 = TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY and
           red 6-10 the same; participant order is the champ-select assigned position):
             StateV2 fields (snapshot values, then event history) with three overrides
               alive   kill-event survival (gameplay.event_survival, events <= t)
               hp_pct / mp_pct  0 while dead at t, 1 if the player respawned after the frame
               level_norm       max(frame level, highest LEVEL_UP level <= t) / 18
             (StateV2's raw champion_id is dropped: the champion vector replaces it);
             added: assists, kills_since_snapshot, respawned_since_snapshot, respawn_remaining_s,
             baron_buff_s, elder_buff_s;
             item vector (gameplay.item_state, 22, 'item_' prefix kept / 'itm_' for the rest);
             champion vector (gameplay.champion_attributes, 29, 'ch_' prefix).
  team     StateV2 blue_ / red_ history counts and their _x_time interactions (unchanged);
           objective timers (gameplay.objective_timers.state(t) without t_s, 'obj_' prefix), minus
           OBJ_METADATA_FIELDS (atakhan_in_patch, rule_uncertain_*): patch-rule constants that are
           the same in every 15.14 / 15.15 row; they are kept on StateV3.objective_rule_flags.
           A timer that is NaN there ("will not spawn again") is written as 0 with the flag
           obj_<objective>_spawn_none = 1.

NaN.  All values are finite except the champion-vector fields of a champion id that is unknown to
the patch table (then <side>_<role>_ch_champ_unknown = 1 and the other 28 ch_ fields are NaN).

No end-of-match guard: StateV2's "query <= last frame" check would need a frame after t, so StateV3
does not apply it (StateV2 is fed a sentinel copy of the snapshot row at t + 1 that is never read);
callers must not query after GAME_END.  hp_pct / mp_pct of a player alive since the frame are the
frame values (damage or healing after the frame is not observable).

Roles.  Player slots are participant order (author decision 2026-09-25: participant order is the
champ-select assigned position; it equals teamPosition in 553/553 16.15 matches).  meta
'role_slots', when present, is cross-checked (StateV3.role_slots_meta_agree).  Role inference
(gameplay.role_inference) is not a model input; with ``StateBuilderV3(..., role_check=True)`` it
is run at t and its outcome is kept as metadata only (StateV3.role_fallback, role_ambiguous,
role_inferred_agree); the former role_fallback / role_ambiguous_* / role_final columns are gone.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gameplay.champion_attributes import (CHAMPION_VECTOR_FIELDS, champion_ids_from_meta, champion_vector,
                                          load_champion_table_v2)
from gameplay.event_survival import death_intervals, event_alive, respawn_remaining_s, respawned_since
from gameplay.item_state import ITEM_VECTOR_NAMES, ItemStateIndex, load_item_table_v2
from gameplay.objective_timers import TIMER_RULES, ObjectiveTimeline, feature_names as objective_feature_names
from gameplay.role_inference import HORIZON_MS as ROLE_HORIZON_MS, ROLES, infer_roles
from gameplay.state_value import SNAPSHOT_FIELDS
from gameplay.state_value_v2 import StateBuilder as StateBuilderV2

STATE_VERSION = "exact_v3.1"  # 3.1: 999 -> 996 columns (role_* moved to metadata)
TEAMS = (100, 200)
SIDE = {100: "blue", 200: "red"}
SLOT_PREFIXES: Tuple[str, ...] = tuple(f"{SIDE[t]}_{r.lower()}_" for t in TEAMS for r in ROLES)

# node columns StateV2 reads, minus 'alive' (overridden from events; the legacy builder still wants
# the column, so it is fed a constant) and 'champion_id' (fed 0, the column is dropped)
V2_NODE_NAMES: Tuple[str, ...] = (*SNAPSHOT_FIELDS, "champion_id")
V2_READ_COLUMNS: Tuple[str, ...] = tuple(n for n in SNAPSHOT_FIELDS if n != "alive")
LEVEL_DEN = 18.0
LEVEL_NORM_MAX = 2.0

V2_PLAYER_FIELDS: Tuple[str, ...] = (
    "totalGold_norm", "curGold_norm", "level_norm", "xp_norm", "hp_pct", "mp_pct", "alive",
    "laneCS_norm", "jgCS_norm", "kills", "deaths", "death_since_snapshot", "death_last_30s",
    "death_age_minutes", "baron_death_since_acquisition", "elder_death_since_acquisition",
)
V2_DROPPED_PLAYER_FIELDS: Tuple[str, ...] = ("champion_id",)
EVENT_PLAYER_FIELDS: Tuple[str, ...] = (
    "assists", "kills_since_snapshot", "respawned_since_snapshot", "respawn_remaining_s",
    "baron_buff_s", "elder_buff_s",
)
ITEM_FIELDS: Tuple[str, ...] = tuple(n if n.startswith("item_") else f"itm_{n}" for n in ITEM_VECTOR_NAMES)
CHAMPION_FIELDS: Tuple[str, ...] = tuple(f"ch_{n}" for n in CHAMPION_VECTOR_FIELDS)
PLAYER_FIELDS: Tuple[str, ...] = V2_PLAYER_FIELDS + EVENT_PLAYER_FIELDS + ITEM_FIELDS + CHAMPION_FIELDS

# role-inference outcome: metadata on StateV3 (role_check=True), never a model input
ROLE_METADATA_FIELDS: Tuple[str, ...] = ("role_fallback", "role_ambiguous_blue", "role_ambiguous_red")
# objective_timers.state() fields that are patch-rule constants in every 15.14 / 15.15 row (checked on
# 500 15.14 matches, stage1/state_value_v3/constant_columns_15.14.json): metadata, not columns
OBJ_METADATA_FIELDS: Tuple[str, ...] = ("atakhan_in_patch",) + tuple(f"rule_uncertain_{g}" for g in TIMER_RULES)
OBJ_NAN_TIMERS: Tuple[str, ...] = ("dragon", "baron", "herald", "atakhan", "voidgrub")
V_ONLY_COLUMNS: Tuple[str, ...] = ("snapshot_age_s",)


# ------------------------------------------------------------------------------------ column contract
def _v2_reference_names() -> List[str]:
    """StateV2 value names, from a synthetic one-frame pack (the names do not depend on data)."""
    tm = {p: (100 if p <= 5 else 200) for p in range(1, 11)}
    node = np.zeros((2, 10, len(V2_NODE_NAMES)), dtype=np.float32)
    pack = {"minute_ts": np.array([0, 1], dtype=np.int64), "node_minute": node, "events": [],
            "meta": {"team_map": tm}}
    return list(StateBuilderV2(pack, list(V2_NODE_NAMES)).at(0).values)


def _split_v2_names(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    glob, team = [], []
    for n in names:
        if n.startswith("participant_slot"):
            continue
        (team if n.startswith(("blue_", "red_")) else glob).append(n)
    return glob, team


def _objective_columns() -> List[str]:
    out = []
    for n in objective_feature_names("15.14"):
        if n == "t_s" or n in OBJ_METADATA_FIELDS:
            continue
        out.append(f"obj_{n}")
    out += [f"obj_{o}_spawn_none" for o in OBJ_NAN_TIMERS]
    return out


def _build_columns() -> Tuple[str, ...]:
    v2 = _v2_reference_names()
    glob, team = _split_v2_names(v2)
    player_v2 = sorted({n.split("_", 2)[2] for n in v2 if n.startswith("participant_slot")})
    expect = sorted(set(V2_PLAYER_FIELDS) | set(V2_DROPPED_PLAYER_FIELDS))
    if player_v2 != expect:
        raise RuntimeError(f"StateV2 player fields changed: {player_v2} != {expect}")
    cols = list(glob)
    for pre in SLOT_PREFIXES:
        cols += [pre + f for f in PLAYER_FIELDS]
    cols += team + _objective_columns()
    if len(set(cols)) != len(cols):
        raise RuntimeError("duplicate StateV3 column names")
    return tuple(cols)


STATE_V3_COLUMNS: Tuple[str, ...] = _build_columns()
# the only columns that may be NaN: champion fields of an id unknown to the patch table
NAN_ALLOWED_COLUMNS = frozenset(pre + f for pre in SLOT_PREFIXES for f in CHAMPION_FIELDS if f != "ch_champ_unknown")
STATE_V3_NAME_HASH_COMPUTED = hashlib.sha256("\n".join(STATE_V3_COLUMNS).encode("utf-8")).hexdigest()
# Frozen name-order hash.  A change of any column name or position changes it; tests compare the two.
STATE_V3_NAME_HASH = "c6cbd83f0feebce8c6d939900175c256c8a437d3a31b70b22e7168f916286069"


def name_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def q_columns(names: Sequence[str] = STATE_V3_COLUMNS) -> List[str]:
    """Columns for q (the engagement model): all StateV3 columns except the V-only frame age."""
    return [n for n in names if n not in V_ONLY_COLUMNS]


def column_blocks(names: Sequence[str] = STATE_V3_COLUMNS) -> Dict[str, int]:
    """Count of columns per block (for reports)."""
    glob, team = _split_v2_names(_v2_reference_names())
    obj = _objective_columns()
    return {"global_v2": len(glob),
            "player_v2": 10 * len(V2_PLAYER_FIELDS), "player_event": 10 * len(EVENT_PLAYER_FIELDS),
            "player_item": 10 * len(ITEM_FIELDS), "player_champion": 10 * len(CHAMPION_FIELDS),
            "team_v2": len(team), "team_objective": len(obj), "total": len(names)}


# ------------------------------------------------------------------------------------ state object
@dataclass
class StateV3:
    """values are the model inputs (STATE_V3_COLUMNS); every other field is metadata."""
    values: Dict[str, float]
    query_ms: int
    snapshot_ms: int
    state_version: str
    slot_by_pid: Dict[int, int]                          # participant order
    role_source: str = "participant_order"
    role_slots_meta_agree: Optional[bool] = None         # meta role_slots == participant order (None: absent)
    objective_rule_flags: Dict[str, float] = field(default_factory=dict)   # OBJ_METADATA_FIELDS at t
    # role-inference outcome at t, only with StateBuilderV3(role_check=True); None otherwise
    role_fallback: Optional[bool] = None                 # inference failed
    role_ambiguous: Optional[Dict[int, bool]] = None     # {team: ambiguous}
    role_inferred_agree: Optional[Dict[int, bool]] = None  # {team: inferred slots == participant order}
    inferred_slot_by_pid: Optional[Dict[int, int]] = None
    setup: Optional[Dict[str, float]] = field(default=None)


def participant_slots(team_map: Mapping) -> Dict[int, int]:
    """pid -> slot 0..9 from participant order: team_map sorted by (team, pid); blue first.

    With the standard team map (1-5 blue, 6-10 red) this is pid - 1, i.e. blue 1-5 = TOP, JUNGLE,
    MIDDLE, BOTTOM, UTILITY and red 6-10 the same."""
    tm = {int(k): int(v) for k, v in (team_map or {}).items()}
    if set(tm) != set(range(1, 11)) or any(list(tm.values()).count(t) != 5 for t in TEAMS):
        raise ValueError("invalid participant roster/team map")
    order = sorted(tm, key=lambda p: (TEAMS.index(tm[p]), p))
    return {p: k for k, p in enumerate(order)}


def meta_role_slots_agree(meta: Mapping, slots: Mapping[int, int]) -> Optional[bool]:
    """meta['role_slots'] equals `slots` (None when the meta has no role_slots)."""
    rs = (meta or {}).get("role_slots")
    if not rs:
        return None
    try:
        return {int(k): int(v) for k, v in rs.items()} == {int(k): int(v) for k, v in slots.items()}
    except (TypeError, ValueError, AttributeError):
        return False


def _meta_patch(meta: Mapping) -> Optional[str]:
    import re
    raw = str((meta or {}).get("patch", "") or (meta or {}).get("patch_full", "") or "")
    nums = re.findall(r"\d+", raw)
    return f"{int(nums[0])}.{int(nums[1])}" if len(nums) >= 2 else None


class StateBuilderV3:
    """Exact StateV3 at any ms t of one match (see module docstring)."""

    state_version = STATE_VERSION

    def __init__(self, pack: Mapping, patch: Optional[str] = None, *, node_names: Optional[Sequence[str]] = None,
                 items_table: Optional[Mapping[int, dict]] = None, champion_table=None, role_check: bool = False):
        meta = dict(pack.get("meta") or {})
        mp = _meta_patch(meta)
        if patch is None:
            if mp is None:
                raise ValueError("patch not given and pack meta has none")
            patch = mp
        patch = str(patch)
        if mp is not None and mp != patch:
            raise ValueError(f"patch {patch} disagrees with pack meta patch {mp}")
        self.patch = patch
        self.slot_by_pid = participant_slots(meta.get("team_map"))
        tm = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
        self.tm = tm
        self.meta = meta
        self.participant_order = tuple(sorted(tm, key=lambda p: (TEAMS.index(tm[p]), p)))
        self.role_slots_meta_agree = meta_role_slots_agree(meta, self.slot_by_pid)
        self.role_check = bool(role_check)
        self.pack = pack
        self.ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        if not len(self.ts) or np.any(np.diff(self.ts) <= 0):
            raise ValueError("invalid snapshot timestamps")
        if node_names is None:
            from core.config import NODE_IDX
            node_names = sorted(NODE_IDX, key=NODE_IDX.get)
        idx = {n: i for i, n in enumerate(node_names)}
        missing = [n for n in V2_READ_COLUMNS if n not in idx]
        if missing:
            raise ValueError(f"node_minute lacks columns {missing}")
        node = np.asarray(pack["node_minute"])
        if node.shape[:2] != (len(self.ts), 10):
            raise ValueError(f"node_minute shape {node.shape} does not match minute_ts {self.ts.shape}")
        # only the columns StateV2 needs are copied; alive -> 1 (overridden), champion_id -> 0 (dropped)
        v2node = np.zeros((len(self.ts), 10, len(V2_NODE_NAMES)), dtype=np.float32)
        for j, n in enumerate(V2_NODE_NAMES):
            if n in V2_READ_COLUMNS:
                v2node[:, :, j] = node[:, :, idx[n]]
            elif n == "alive":
                v2node[:, :, j] = 1.0
        self._v2node = v2node
        self._v2meta = {"team_map": dict(tm)}
        self.events = sorted((e for e in pack.get("events") or [] if isinstance(e, dict)),
                             key=lambda e: int(e.get("timestamp", 0)))
        self._ev_ts = [int(e.get("timestamp", 0)) for e in self.events]
        # static per-player data
        table = champion_table if champion_table is not None else load_champion_table_v2(patch)
        cids = champion_ids_from_meta(meta)
        self.champion_ids = cids
        self._champ = {p: champion_vector(cids[p], table) for p in range(1, 11)}
        self.items = items_table if items_table is not None else load_item_table_v2(patch)
        self._item_index = ItemStateIndex.from_pack({"events": self.events, "meta": meta}, self.items, patch=patch)
        self._objectives = ObjectiveTimeline(self.events, patch, tm)
        self._role_cache: Dict[float, Tuple[Dict[int, int], bool, Dict[int, bool]]] = {}

    # -------------------------------------------------------------------------------- helpers
    def _events_le(self, q: int) -> List[dict]:
        return self.events[: bisect_right(self._ev_ts, q)]

    def roles_at(self, q: int) -> Tuple[Dict[int, int], bool, Dict[int, bool]]:
        """Role INFERENCE (validation / metadata only, never the slots of the columns):
        (inferred slot_by_pid, fallback, {team: ambiguous}) from data <= min(q, role horizon)."""
        key = float(min(q, ROLE_HORIZON_MS))
        hit = self._role_cache.get(key)
        if hit is not None:
            return hit
        try:
            ra = infer_roles(self.pack, key)
            out = (dict(ra.slot_by_pid), False, {t: bool(ra.teams[t].ambiguous) for t in TEAMS})
        except (KeyError, ValueError, IndexError, TypeError):
            out = ({p: k for k, p in enumerate(self.participant_order)}, True, {t: True for t in TEAMS})
        self._role_cache[key] = out
        return out

    def _v2_values(self, q: int, i: int) -> Dict[str, float]:
        snap = int(self.ts[i])
        # Two-row pack: the snapshot frame and a sentinel copy at q + 1, so StateV2's end-of-match guard
        # (query <= last frame) never needs a frame after t.  The sentinel row is never read.
        ts = np.array([snap, max(q, snap) + 1], dtype=np.int64)
        node = np.repeat(self._v2node[i:i + 1], 2, axis=0)
        pack = {"minute_ts": ts, "node_minute": node, "meta": self._v2meta, "events": self._events_le(q)}
        st = StateBuilderV2(pack, list(V2_NODE_NAMES)).at(q)
        if st.snapshot_ms != snap:
            raise RuntimeError("StateV2 snapshot mismatch")
        return st.values

    # -------------------------------------------------------------------------------- query
    def at(self, t: int, setup: bool = False) -> StateV3:
        q = int(t)
        i = int(np.searchsorted(self.ts, q, side="right") - 1)
        if i < 0:
            raise ValueError("query before the first frame")
        snap = int(self.ts[i])
        v2 = self._v2_values(q, i)
        slot_by_pid = self.slot_by_pid
        events = self._events_le(q)

        # event-derived per-player quantities (events <= q only)
        iv = death_intervals(events, self.patch, t=q)
        alive = event_alive(iv, q)
        resp = respawned_since(iv, snap, q)
        remain = respawn_remaining_s(iv, q)
        lvl = np.zeros(10)
        assists = np.zeros(10)
        k_since = np.zeros(10)
        for e in events:
            et = e.get("type")
            if et == "LEVEL_UP":
                p = int(e.get("participantId", 0) or 0)
                if 1 <= p <= 10:
                    lvl[p - 1] = max(lvl[p - 1], float(e.get("level", 0) or 0))
            elif et == "CHAMPION_KILL":
                for a in e.get("assistingParticipantIds") or []:
                    a = int(a)
                    if 1 <= a <= 10:
                        assists[a - 1] += 1
                k = int(e.get("killerId", 0) or 0)
                if 1 <= k <= 10 and int(e["timestamp"]) > snap:
                    k_since[k - 1] += 1
        baron = self._objectives.buff_remaining_by_player("baron", q)
        elder = self._objectives.buff_remaining_by_player("elder", q)

        out: Dict[str, float] = {}
        glob, team = _split_v2_names(list(v2))
        for n in glob:
            out[n] = float(v2[n])

        pid_by_slot = {s: p for p, s in slot_by_pid.items()}
        for s, pre in enumerate(SLOT_PREFIXES):
            p = pid_by_slot[s]
            j = p - 1
            k = self.participant_order.index(p)
            src = f"participant_slot{k}_"
            rec = {f: float(v2[src + f]) for f in V2_PLAYER_FIELDS}
            rec["alive"] = float(alive[j])
            if alive[j] < 0.5:
                rec["hp_pct"], rec["mp_pct"] = 0.0, 0.0
            elif resp[j] > 0.5:
                rec["hp_pct"], rec["mp_pct"] = 1.0, 1.0
            rec["level_norm"] = float(min(LEVEL_NORM_MAX, max(rec["level_norm"], lvl[j] / LEVEL_DEN)))
            rec["assists"] = float(assists[j])
            rec["kills_since_snapshot"] = float(k_since[j])
            rec["respawned_since_snapshot"] = float(resp[j])
            rec["respawn_remaining_s"] = float(remain[j])
            rec["baron_buff_s"] = float(baron[j])
            rec["elder_buff_s"] = float(elder[j])
            iv_vec = self._item_index.vector(p, q)
            for name, col in zip(ITEM_VECTOR_NAMES, ITEM_FIELDS):
                rec[col] = float(iv_vec[name])
            cv = self._champ[p]
            for name, col in zip(CHAMPION_VECTOR_FIELDS, CHAMPION_FIELDS):
                rec[col] = float(cv[name])
            for f in PLAYER_FIELDS:
                out[pre + f] = rec[f]

        for n in team:
            out[n] = float(v2[n])
        obj = self._objectives.state(q)
        for n, v in obj.items():
            if n == "t_s" or n in OBJ_METADATA_FIELDS:
                continue
            out[f"obj_{n}"] = 0.0 if (n.endswith("_next_spawn_s") and math.isnan(v)) else float(v)
        for o in OBJ_NAN_TIMERS:
            out[f"obj_{o}_spawn_none"] = float(math.isnan(obj[f"{o}_next_spawn_s"]))

        if tuple(out) != STATE_V3_COLUMNS:
            raise RuntimeError("StateV3 column order drifted from STATE_V3_COLUMNS")
        bad = [n for n, v in out.items() if not math.isfinite(v) and n not in NAN_ALLOWED_COLUMNS]
        if bad:
            raise ValueError(f"non-finite StateV3 values: {bad[:5]}")
        st = StateV3(out, q, snap, STATE_VERSION, dict(slot_by_pid), role_slots_meta_agree=self.role_slots_meta_agree,
                     objective_rule_flags={n: float(obj[n]) for n in OBJ_METADATA_FIELDS})
        if self.role_check:
            inferred, fallback, amb = self.roles_at(q)
            st.role_fallback, st.role_ambiguous = bool(fallback), dict(amb)
            st.inferred_slot_by_pid = dict(inferred)
            st.role_inferred_agree = {t: all(inferred[p] == slot_by_pid[p] for p in slot_by_pid if self.tm[p] == t)
                                      for t in TEAMS}
        if setup:
            from gameplay.setup_features import build_setup
            st.setup = build_setup(self.pack, q, slot_by_pid)
        return st


def state_matrix_v3(states: Sequence[StateV3], names: Sequence[str] = STATE_V3_COLUMNS,
                    expected_version: str = STATE_VERSION) -> np.ndarray:
    """(n, len(names)) float matrix; refuses states of another version or column order."""
    if name_hash(STATE_V3_COLUMNS) != STATE_V3_NAME_HASH:
        raise RuntimeError("STATE_V3_COLUMNS no longer match STATE_V3_NAME_HASH")
    if not all(isinstance(s, StateV3) and s.state_version == expected_version for s in states):
        raise ValueError("state version mismatch; regenerate states and refit model")
    if not all(tuple(s.values) == STATE_V3_COLUMNS for s in states):
        raise ValueError("state feature schema mismatch")
    missing = [n for n in names if n not in STATE_V3_COLUMNS]
    if missing:
        raise ValueError(f"unknown StateV3 columns {missing[:5]}")
    return np.asarray([[s.values[n] for n in names] for s in states], dtype=float).reshape(len(states), len(names))
