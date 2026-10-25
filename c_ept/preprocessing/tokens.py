"""
Layered Multi-Modal Game State Views for Machine Learning / 머신러닝 모델 입력을 위한 시공간 다층 게임 상태 뷰(View) 생성 모듈

[English]
This module builds a comprehensive, multi-modal "View" of a match at a specific reference time (ref_ms).
A View captures everything the model needs to observe at (match, reference time):
  1. players_num (10, 81) f4 : 10 participants (0-4 Blue, 5-9 Red) x 81 features
       - 33 frame fields (gold, xp, cs, hp, pos, etc.)
       - 7 derived dynamic features (prev_x, prev_y, has_prev, level_now, alive, respawn_s, wall_fraction)
       - 19 champion base & growth traits (patch-normalized stats)
       - 22 completed/current item stat bonuses
  2. players_zone (10,) i1   : Tactical zone ID on Summoner's Rift (0-9), -1 if position missing
  3. landmarks (34, 2) f4    : Status (standing/up 1.0 or 0.0) and respawn seconds for 34 landmarks
  4. ev_* (128,) tokens      : Recent events in (ref_ms - 120s, ref_ms], sorted newest-first
       - ev_type, ev_dt (age in s), ev_team, ev_xy, ev_actors (bitmask), ev_gold, ev_n, ev_dropped
  5. ed_* (128,) edges       : Co-participation & interaction edges between champions in recent events
       - ed_i, ed_j (0-9), ed_dt, ed_kind (killer, assist, ally pair, damage, objective pair)
  6. summary (107,) f4       : Global game summary
       - game_time_s, frame_age_s, 16 old event counts (2-10 min window), 89 objective state values
  7. geo_pp (10, 10) f2      : Pairwise geodesic (walking) distances between players
  8. geo_pl (10, 34) f2      : Geodesic distances between players and 34 map landmarks

Missing values and NAN_ALLOWED:
  NAN_ALLOWED is exhaustive: for each of the six arrays that can hold a NaN (players_num, summary,
  landmarks, ev_xy, geo_pp, geo_pl) its entry gives the cells and the condition when.
  A NaN in any other cell or array of a view is a bug.

[한국어 설명]
이 모듈은 특정 기준 시각(`ref_ms`) 시점의 경기 상황을 모델이 읽을 수 있는 다차원 텐서 묶음인 '뷰(View)'로 조합합니다.
- `MatchContext(rec, terr)`: 경기 1회당 1번 생성되어 레벨 시계, 아이템 인덱스, 누적 사건 테이블을 사전 인덱싱합니다.
- `build_view(ctx, ref_ms, frame_k)`: 기준 시각 `ref_ms`와 직전 정규 프레임 `frame_k`를 바탕으로 `VIEW_SPEC` 규격의 넘파이 배열 딕셔너리를 반환합니다.
- 프레임 시계(Frame clock) 행: `frame_ts == ref_ms`일 때 프레임 내 스냅샷 수치를 그대로 사용합니다.
- 사건 시계(Event clock) 행: `frame_ts < ref_ms`일 때, 직전 프레임의 위치/기본 스탯에 더해 `ref_ms`까지 진행된 실시간 레벨 및 아이템 변동을 반영합니다.
- 결측치 관리: `NAN_ALLOWED`에 명시된 6개 배열(선수, 요약, 표지점, 이벤트좌표, 거리행렬)의 지정된 셀 외에는 NaN이 엄격히 금지됩니다.
"""
# Ref: Che et al. 2018, "Recurrent Neural Networks for Multivariate Time Series with Missing Values", Scientific Reports 8:6085;
#      doi:10.1038/s41598-018-24271-9 (mask + elapsed-time inputs: here the masks are has_prev, champ_unknown and the NaN of a missing
#      value, the elapsed times are frame_age_s and the ages ev_dt and ed_dt). Bibliographic data read from the DOI record (Crossref API)
#      on 2026-10-21; the publisher page redirected to a cookie handshake and was not read.
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 4 (two clocks, masking), 5.1 (layered window: recent events 120 s, old
#      events 2-10 min as counts, frames, event-derived state), 5.2 (tokens, relations, co-participation edges), 5.3 (patch-invariant
#      inputs: traits and item statistics, no ids) and section 11 items 7, 8, 13 and 14 (frame time, views and frames, event tokens,
#      edges).
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from c_ept import config as C
from c_ept.domain import state
from c_ept.preprocessing import traits
from c_ept.data.casts import narrow
from c_ept.data.reader import MatchFormatError, MatchRecord
from c_ept.domain.units import BLUE, RED, _assist_mask, _event_id, _is_int, event_xy

N_PLAYERS = 10
N_LANDMARKS = len(state.LANDMARKS)
MAX_EVENTS, MAX_EDGES = C.MAX_EVENTS, C.MAX_EDGES
TEAM_BLUE, TEAM_RED, TEAM_NONE = 0, 1, 2
EDGE_KINDS = ('killer_victim', 'assist_victim', 'killing_side_pair', 'damage_victim', 'objective_pair')      # ed_kind = index
EDGE_KILLER, EDGE_ASSIST, EDGE_SIDE, EDGE_DAMAGE, EDGE_OBJECTIVE = range(len(EDGE_KINDS))

# ---- the columns --------------------------------------------------------------------------------------------------------------------
DERIVED_COLUMNS = ('prev_x', 'prev_y', 'has_prev', 'level_now', 'alive', 'respawn_s', 'wall_fraction')
PLAYER_NUM = tuple(C.FRAME_FIELDS) + DERIVED_COLUMNS + tuple(traits.TRAIT_NAMES) + tuple(traits.item_vector_names())
OLD_COLUMNS = tuple(f'old_{name.lower()}_{side}' for name in C.EVENT_TYPES for side in 'br')           # (type, blue), (type, red), ...
SUMMARY = ('game_time_s', 'frame_age_s') + OLD_COLUMNS + tuple(state.OBJ_STATE_NAMES)    # game_time_s == t_s: plan-mandated twins
if (len(PLAYER_NUM), len(SUMMARY)) != (81, 107) or len(set(PLAYER_NUM)) != 81 or len(set(SUMMARY)) != 107:
    raise RuntimeError(f'the view columns are {len(PLAYER_NUM)} player and {len(SUMMARY)} summary columns, not 81 and 107, '
                       'or have duplicates')
_COL = {name: i for i, name in enumerate(PLAYER_NUM)}
_X, _Y, _LEVEL, _PREV_X, _PREV_Y, _HAS_PREV = (_COL[n] for n in ('x', 'y', 'level', 'prev_x', 'prev_y', 'has_prev'))
_LEVEL_NOW, _ALIVE, _RESPAWN, _WALL = (_COL[n] for n in ('level_now', 'alive', 'respawn_s', 'wall_fraction'))
_N_FRAME = len(C.FRAME_FIELDS)
_TRAITS = slice(_COL[traits.TRAIT_NAMES[0]], _COL[traits.TRAIT_NAMES[-1]] + 1)
_UNKNOWN = _COL['champ_unknown']
_ITEMS = slice(_COL[traits.item_vector_names()[0]], _COL[traits.item_vector_names()[-1]] + 1)
_OLD0, _OBJ0 = SUMMARY.index(OLD_COLUMNS[0]), SUMMARY.index(state.OBJ_STATE_NAMES[0])


def _nan_entry(cells, when: str):
    return MappingProxyType({'cells': cells, 'when': when})


NAN_ALLOWED = MappingProxyType({            # exhaustive: array name -> {'cells': where a NaN may stand, 'when': under which condition}
    'players_num': _nan_entry(              # cells: column names
        ('x', 'y', 'prev_x', 'prev_y', 'wall_fraction') + tuple(n for n in traits.TRAIT_NAMES if n != 'champ_unknown'),
        'x, y: the frame has no position for the player; prev_x, prev_y: has_prev is 0 (frame 0, or no position in the frame before); '
        'wall_fraction: x or y is NaN; the 18 trait columns: champ_unknown is 1'),
    'summary': _nan_entry(tuple(state.NAN_FEATURES),          # cells: column names
                          'LEGACY: that monster will not spawn again in this game (or never in this patch)'),
    'landmarks': _nan_entry(                # cells: (row, column) pairs; column 1 is the seconds cell
        tuple((i, 1) for i, lm in enumerate(state.LANDMARKS) if lm['kind'] in ('herald', 'grubs', 'atakhan')),
        'LEGACY, as for the summary: the seconds cell of the herald, grubs and atakhan rows'),
    'ev_xy': _nan_entry('all', 'the event has no position (items, levels and wards have none), or the row is padding (ev_type -1)'),
    'geo_pp': _nan_entry('all', 'the row and the column of a player whose position is missing (players_zone -1)'),
    'geo_pl': _nan_entry('all', 'the row of a player whose position is missing (players_zone -1)'),
})

_F4, _F2, _I1, _I2 = np.dtype('f4'), np.dtype('f2'), np.dtype('i1'), np.dtype('i2')
VIEW_SPEC = MappingProxyType({                                  # name -> (shape, dtype), in the order of the dict that build_view returns
    'players_num': ((N_PLAYERS, len(PLAYER_NUM)), _F4), 'players_zone': ((N_PLAYERS,), _I1), 'landmarks': ((N_LANDMARKS, 2), _F4),
    'ev_type': ((MAX_EVENTS,), _I1), 'ev_dt': ((MAX_EVENTS,), _F4), 'ev_team': ((MAX_EVENTS,), _I1), 'ev_xy': ((MAX_EVENTS, 2), _F4),
    'ev_actors': ((MAX_EVENTS,), _I2), 'ev_gold': ((MAX_EVENTS,), _F4), 'ev_n': ((), _I2), 'ev_dropped': ((), _I2),
    'ed_i': ((MAX_EDGES,), _I1), 'ed_j': ((MAX_EDGES,), _I1), 'ed_dt': ((MAX_EDGES,), _F4), 'ed_kind': ((MAX_EDGES,), _I1),
    'ed_n': ((), _I2), 'ed_dropped': ((), _I2), 'summary': ((len(SUMMARY),), _F4), 'geo_pp': ((N_PLAYERS, N_PLAYERS), _F2),
    'geo_pl': ((N_PLAYERS, N_LANDMARKS), _F2),
})


def bytes_per_view() -> int:
    """The bytes of one view, from the dtypes and shapes of ``VIEW_SPEC``.

    수식
    ----
    (TOK-7)  bytes = Σ_{name ∈ VIEW_SPEC} (∏ shape_name) · itemsize(dtype_name)      (0-d 배열의 ∏ shape는 1)

      기호          | 뜻                                | 단위   | 범위              | 코드 변수
      shape_name    | 배열 name의 모양                  | 개     | 정수 ≥ 0의 튜플   | VIEW_SPEC[name][0]
      itemsize      | 그 dtype의 원소 하나의 바이트 수  | 바이트 | 1, 2, 4           | dtype.itemsize
      bytes         | 한 보기의 바이트 수               | 바이트 | 양의 정수         | 반환값

    shape와 dtype은 ``VIEW_SPEC``이 정하고 이 함수는 그 합만 계산한다.

    근거: S1–S2 계획 Task 12(보기의 저장 바이트)."""
    return sum(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize for shape, dtype in VIEW_SPEC.values())


LANDMARK_XY = np.array([[lm['x'], lm['y']] for lm in state.LANDMARKS], dtype=np.float64)       # (34, 2) absolute game units; read only
LANDMARK_XY.setflags(write=False)

_KIND = {name: i for i, name in enumerate(C.EVENT_TYPES)}
_KILL, _ELITE, _BUILDING, _PLATE, _ITEM, _LEVEL_UP, _WARD_PLACED, _WARD_KILL = (
    _KIND[n] for n in ('CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL', 'TURRET_PLATE_DESTROYED', 'ITEM_PURCHASED', 'LEVEL_UP',
                       'WARD_PLACED', 'WARD_KILL'))
_N_OLD = 2 * len(C.EVENT_TYPES)


def _ms(value, name: str) -> int:
    """``value`` as a Python int; TypeError (naming ``name``) unless it is a whole number of ms or an index (an int or a numpy integer;
    not a bool or a float)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be a whole number (int), not {type(value).__name__}')
    return int(value)


# ---- the events of one match ----------------------------------------------------------------------------------------------------------
# (TOK-6)의 bit(pid)이다.
def _bit(pid: int) -> int:
    return 1 << (pid - 1) if 1 <= pid <= N_PLAYERS else 0


def _pids(mask: int) -> list:
    """The participant ids (1-10) of a bit mask, ascending."""
    return [pid for pid in range(1, N_PLAYERS + 1) if mask >> (pid - 1) & 1]


# (TOK-6)의 code(id)이다.
def _team_code(team_id) -> int:
    """0 for the team id 100, 1 for 200, 2 for anything else (absent, another number, a float, a bool, a string)."""
    if _is_int(team_id) and team_id in (BLUE, RED):
        return TEAM_RED if team_id == RED else TEAM_BLUE
    return TEAM_NONE


# (TOK-6)의 taker(id)이다.
def _taker(team_id) -> int:
    """The team that took a building or plate whose owner ``team_id`` lost it: the other team; 2 when the owner is not a team."""
    owner = _team_code(team_id)
    return TEAM_NONE if owner == TEAM_NONE else 1 - owner


def _number(rec: MatchRecord, e: dict, name: str, label: str) -> float:
    """``e[name]`` as a float: 0 when it is absent or null, MatchFormatError (no key in the message) when it is not a finite number."""
    v = e.get(name)
    if v is None:
        return 0.0
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise MatchFormatError(rec.key, rec.patch, f'{label} has a {name} of type {type(v).__name__} that is not a finite number')
    return float(v)


def _dealers(rec: MatchRecord, e: dict, victim: int, side: tuple) -> list:
    """The distinct participants 1-10 on the other team than the victim in the victim's ``victimDamageReceived``, ascending [R12-f]. The
    list is absent or null (none), or a list of entries; an entry that is not an object, or has no integer participantId 1-10 (minions,
    towers and monsters have 0 or none), or is on the victim's own team, is dropped. Anything else for the list: MatchFormatError.
    (TOK-6)의 피해 간선(종류 3)의 딜러를 고른다."""
    entries = e.get('victimDamageReceived')
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise MatchFormatError(rec.key, rec.patch,
                               f'a CHAMPION_KILL has a victimDamageReceived of type {type(entries).__name__}, not a list')
    own = side[victim - 1]
    found = set()
    for d in entries:
        pid = d.get('participantId') if isinstance(d, dict) else None
        if _is_int(pid) and 1 <= pid <= N_PLAYERS and side[pid - 1] != own:
            found.add(pid)
    return sorted(found)


def _pairs(members: list) -> list:
    """(TOK-6)의 쌍 간선: 목록에서 앞선 원소와 뒤의 원소로 된 모든 쌍을 (번호 − 1, 번호 − 1)로 낸다."""
    return [(members[a] - 1, members[b] - 1) for a in range(len(members)) for b in range(a + 1, len(members))]


def _order(pids: list, side: tuple, ref) -> list:
    """The participants ``pids`` with those on the team ``ref`` first, then the others; ascending within each of the two groups. This is
    the order that the team swap keeps: the swap adds 5 to a participant (mod 10), which keeps the order inside a team, and it turns
    ``ref`` and every team into the other one together. (A plain ascending order would not survive a swap for a list on both teams.)
    (TOK-6)의 order(pids, lead)이며, 인자 ref가 lead이다."""
    return sorted(pids, key=lambda p: (side[p - 1] != ref, p))


def _read_event(rec: MatchRecord, e: dict, side: tuple) -> tuple:
    """(type index, team, actor mask, gold, edges) of one event of the eight types; ``edges`` is a list of (i, j, kind).
    (TOK-6)의 사건 기록을 만든다."""
    typ = e['type']
    label = f'a {typ}'
    team_of = lambda pid: side[pid - 1] if 1 <= pid <= N_PLAYERS else TEAM_NONE                    # noqa: E731
    if typ == 'CHAMPION_KILL':
        killer = _event_id(rec, e, 'killerId', 0, 0, label)
        victim = _event_id(rec, e, 'victimId', 1, None, label)
        assists = _assist_mask(rec, e, label)
        killing = team_of(killer) if killer else 1 - side[victim - 1]            # the side of the kill: the killer's, else not the victim's
        helpers = _order([a for a in _pids(assists) if a not in (killer, victim)], side, killing)    # neither the killer nor the victim
        edges = [(killer - 1, victim - 1, EDGE_KILLER)] if killer else []
        edges += [(a - 1, victim - 1, EDGE_ASSIST) for a in helpers]
        edges += [(i, j, EDGE_SIDE) for i, j in _pairs(([killer] if killer else []) + helpers)]
        edges += [(d - 1, victim - 1, EDGE_DAMAGE) for d in _dealers(rec, e, victim, side)]
        gold = _number(rec, e, 'bounty', label) + _number(rec, e, 'shutdownBounty', label)
        return _KILL, team_of(killer), _bit(killer) | _bit(victim) | assists, gold, edges
    if typ in ('ELITE_MONSTER_KILL', 'BUILDING_KILL'):
        killer = _event_id(rec, e, 'killerId', 0, 0, label)
        assists = _assist_mask(rec, e, label)
        team = _team_code(e.get('killerTeamId')) if typ == 'ELITE_MONSTER_KILL' else _taker(e.get('teamId'))
        ref = team if team != TEAM_NONE else team_of(killer)                       # the team of the objective (none: ascending ids)
        members = ([killer] if killer else []) + _order([a for a in _pids(assists) if a != killer], side, ref)
        edges = [(i, j, EDGE_OBJECTIVE) for i, j in _pairs(members)]
        return (_ELITE if typ == 'ELITE_MONSTER_KILL' else _BUILDING), team, _bit(killer) | assists, _number(rec, e, 'bounty', label), edges
    if typ == 'TURRET_PLATE_DESTROYED':
        killer = _event_id(rec, e, 'killerId', 0, 0, label)
        return _PLATE, _taker(e.get('teamId')), _bit(killer), 0.0, []
    if typ == 'ITEM_PURCHASED':                                  # participant 0 occurs (system purchases): an event without an actor
        pid = _event_id(rec, e, 'participantId', 0, None, label)
        return _ITEM, team_of(pid), _bit(pid), 0.0, []
    if typ == 'LEVEL_UP':
        pid = _event_id(rec, e, 'participantId', 1, None, label)
        return _LEVEL_UP, team_of(pid), _bit(pid), 0.0, []
    if typ == 'WARD_PLACED':
        pid = _event_id(rec, e, 'creatorId', 0, 0, label)
        return _WARD_PLACED, team_of(pid), _bit(pid), 0.0, []
    pid = _event_id(rec, e, 'killerId', 0, 0, label)                      # WARD_KILL
    return _WARD_KILL, team_of(pid), _bit(pid), 0.0, []


class _Events:
    """The events of the eight token types of one match, in event order, as arrays in the dtypes of the view.

    ``t`` (ms), ``type``, ``team``, ``actors``, ``gold``, ``xy`` (NaN without a position) and ``edges`` (per event None or an (n_edges, 3)
    int8 array of (i, j, kind)); ``edge_events`` the sorted indices of the events that have edges and ``edge_cum`` (n + 1) the edges of
    the first e events; ``cum`` (n + 1, 16): how many events of each (type, team) there are among the first e events, so that a window of
    old events is a difference of two rows.

    (TOK-6)의 사건 기록과 누적표를 담고, (TOK-1)–(TOK-3)의 창은 이 배열의 색인으로 자른다.
    """

    def __init__(self, rec: MatchRecord, side: tuple):
        rows = [e for e in rec.events if e.get('type') in _KIND]
        n = len(rows)
        t, typ, team, gold = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64), np.zeros(n)
        actors, xy = np.zeros(n, np.int64), np.full((n, 2), np.nan)
        self.edges: list = [None] * n
        for j, e in enumerate(rows):
            t[j] = e['timestamp']
            typ[j], team[j], actors[j], gold[j], edges = _read_event(rec, e, side)
            xy[j] = event_xy(e)
            if edges:
                self.edges[j] = narrow('edges', np.array(edges, dtype=np.int64), np.int8)
        if n > 1 and np.any(np.diff(t) < 0):
            raise MatchFormatError(rec.key, rec.patch, 'the events are not in time order')
        self.n = n
        self.t = t
        self.type = narrow('ev_type', typ, np.int8)
        self.team = narrow('ev_team', team, np.int8)
        self.actors = narrow('ev_actors', actors, np.int16)
        self.gold = narrow('ev_gold', gold, np.float32)
        self.xy = narrow('ev_xy', xy, np.float32)
        self.edge_events = np.array([j for j, e in enumerate(self.edges) if e is not None], dtype=np.int64)
        self.edge_cum = np.zeros(n + 1, dtype=np.int64)          # edges of the first e events: a window's edges are a difference of two
        np.cumsum(np.array([0 if e is None else len(e) for e in self.edges], dtype=np.int64), out=self.edge_cum[1:])
        counted = np.flatnonzero(team < TEAM_NONE)
        step = np.zeros((n, _N_OLD), dtype=np.int32)
        step[counted, 2 * typ[counted] + team[counted]] = 1
        self.cum = np.zeros((n + 1, _N_OLD), dtype=np.int32)
        np.cumsum(step, axis=0, out=self.cum[1:])


class MatchContext:
    """
    Match Context Pre-Indexer / 경기 단위 정적 정보 및 사건/아이템 사전 인덱싱 클래스

    [English]
    Indexes and caches match-level structures once per match for fast view extraction:
      - `rec`: The parsed match record (MatchRecord)
      - `terr`: The map terrain model providing geodesic paths, zone mapping, and wall fractions
      - `events`: Preprocessed recent & old event arrays with fast cumulative tables (_Events)
      - `levels`: Fast level-up lookup clock (traits.LevelIndex)
      - `items`: Item replay index tracking participant inventory bonuses (traits.ItemIndex)
      - `state_index`: Objective & landmark tracker (state.StateIndex)

    [한국어 설명]
    뷰(View)를 생성할 때마다 반복 계산되는 경기 데이터를 매치당 단 한 번 인덱싱하여 캐싱합니다:
      - `rec`: 경기 원본 레코드 (`MatchRecord`)
      - `terr`: 소환사의 협곡 지형 모델 (`terrain.Terrain`, 측지선/영역/벽 밀도 조회)
      - `events`: 8개 주요 사건 토큰화 및 구간별 누적 카운트 테이블 (`_Events`)
      - `levels`: 실시간 레벨 조회 시계 (`traits.LevelIndex`)
      - `items`: 선수별 아이템 보유 상태 및 스탯 재생 인덱스 (`traits.ItemIndex`)
      - `state_index`: 오브젝트 및 맵 표지점 상태 추적기 (`state.StateIndex`)
    """

    def __init__(self, rec: MatchRecord, terr):
        if not isinstance(rec, MatchRecord):
            raise TypeError(f'rec must be a MatchRecord, not {type(rec).__name__}')
        missing = [name for name in ('geo', 'zone', 'wall_fraction') if not callable(getattr(terr, name, None))]
        if missing:
            raise TypeError(f'terr must answer {", ".join(missing)} like terrain.Terrain does')
        if rec.team.tolist() != [BLUE] * 5 + [RED] * 5:
            raise ValueError(f'a view needs participants 1-5 on team {BLUE} and 6-10 on team {RED}')
        self.rec, self.terr = rec, terr
        self.side = (TEAM_BLUE,) * 5 + (TEAM_RED,) * 5
        self.events = _Events(rec, self.side)                  # first: it is strict and its errors never carry the key
        self.levels = traits.LevelIndex(rec)
        self.items = traits.ItemIndex(rec)
        self.state_index = state.StateIndex(rec)


# ---- one view ---------------------------------------------------------------------------------------------------------------------------
def _frame_values(rec: MatchRecord, k: int) -> np.ndarray:
    """The (10, 33) frame ``k`` as read; ValueError naming the columns if any value other than a position is missing [S2-4]."""
    frame = rec.frame_num[k]
    missing = np.isnan(frame)
    missing[:, [_X, _Y]] = False
    if missing.any():
        cols = ', '.join(C.FRAME_FIELDS[j] for j in np.flatnonzero(missing.any(axis=0)))
        raise ValueError(f'players_num: NaN in the frame column(s) {cols}: only a position (x, y) may be missing')
    return frame


def _previous_position(rec: MatchRecord, k: int) -> tuple:
    """((10, 2) float64 position in the frame before, (10,) bool has_prev): NaN and False at frame 0 or where that position is missing."""
    if k == 0:
        return np.full((N_PLAYERS, 2), np.nan), np.zeros(N_PLAYERS, dtype=bool)
    prev = rec.frame_num[k - 1][:, [_X, _Y]]
    if np.isinf(prev).any():
        raise ValueError('prev_x, prev_y: an infinite position in the frame before')
    has = np.isfinite(prev).all(axis=1)
    return np.where(has[:, None], prev, np.nan), has


def build_view(ctx: MatchContext, ref_ms: int, frame_k: int) -> dict:
    """
    Build Multi-Modal State View at Reference Time / 기준 시각(ref_ms)에서의 다차원 관측 뷰 생성

    [English]
    Builds the complete multi-modal view dictionary (arrays conforming to VIEW_SPEC)
    at the given reference timestamp `ref_ms`, based on regular frame `frame_k`:
      - Validates temporal consistency: frame_k must be the latest regular frame <= ref_ms.
      - Assembles 10 x 81 participant feature matrix (frame stats, level, traits, items).
      - Computes zone classification, wall fraction, and pairwise/landmark geodesic walking distances.
      - Tokenizes up to 128 recent events in (ref_ms - 120s, ref_ms] (newest first).
      - Extracts co-participation champion edges (kills, assists, damages, objective takedowns).
      - Aggregates old events (2-10 min) and objective control states into a 107-dim summary vector.

    Parameters:
      ctx: Pre-indexed match context (MatchContext)
      ref_ms: Reference query timestamp in milliseconds (integer)
      frame_k: Index of the regular frame providing snapshot base stats

    Returns:
      Dictionary of numpy arrays matching VIEW_SPEC layout.

    [한국어 설명]
    정규 프레임 `frame_k`와 기준 시각 `ref_ms` 시점의 게임 상태를 다차원 텐서 뷰(View)로 조합합니다:
      - 인과성 검증: `frame_k`는 반드시 `ref_ms` 이전의 가장 최신 정규 프레임이어야 합니다.
      - 10명의 소환사 특성 행렬(10 x 81): 기본 프레임 스탯, 레벨, 특성치, 아이템 보너스 결합.
      - 맵 지형 특성: 선수별 전술 구역(zone), 5x5 국소 벽 밀도(wall_fraction), 측지선 보행 거리(geo_pp, geo_pl).
      - 최근 120초 이내 사건 토큰화: 최대 128개 이벤트를 최신순으로 정렬하여 텐서화.
      - 챔피언 간 상호작용 엣지: 킬러-피해자, 어시스트, 팀워크, 딜러 기여, 오브젝트 사냥 관계 추출.
      - 전역 요약 벡터(107차원): 낡은 사건 카운트(2~10분 전) 및 타워/오브젝트 잔존 상태.
    """
    rec = ctx.rec
    ref = _ms(ref_ms, 'ref_ms')
    k = _ms(frame_k, 'frame_k')
    if not 0 <= k < len(rec.frame_ts):
        raise ValueError(f'frame_k must be a regular frame, 0 .. {len(rec.frame_ts) - 1}, got {k}')

    # [Temporal Validity Check / 시점 인과성 검증]
    # Frame must not be in the future, and no other regular frame may exist between frame_k and ref_ms.
    # 프레임이 미래일 수 없으며, frame_k와 ref_ms 사이에 또 다른 정규 프레임이 존재하면 안 됨.
    t_frame = int(rec.frame_ts[k])
    if t_frame > ref:
        raise ValueError(f'frame {k} is at {t_frame} ms, after the reference time {ref} ms: '
                         'a view reads frames up to its reference time only')
    if k + 1 < len(rec.frame_ts) and int(rec.frame_ts[k + 1]) < ref:
        raise ValueError(f'frame_k {k} is stale: the regular frame {k + 1} at {int(rec.frame_ts[k + 1])} ms lies before the reference '
                         f'time {ref} ms; a view takes the last regular frame before its reference time')
    terr, ev = ctx.terr, ctx.events

    # [1. Players Feature Matrix / 선수 특성 행렬 조합 (10, 81)]
    # Assemble frame values, dynamic level, traits, and item stats.
    # 기본 프레임 스탯, 실시간 레벨, 챔피언 고유 특성, 아이템 스탯을 결합합니다.
    frame = _frame_values(rec, k)
    levels = frame[:, _LEVEL] if t_frame == ref else ctx.levels.at(ref)                     # frame row: the frame's level; else the clock
    prev, has_prev = _previous_position(rec, k)
    alive, respawn = ctx.state_index.alive_respawn(ref)
    px, py = frame[:, _X], frame[:, _Y]
    players = np.empty((N_PLAYERS, len(PLAYER_NUM)), dtype=np.float32)
    players[:, :_N_FRAME] = narrow('frame values', frame, np.float32)
    players[:, [_PREV_X, _PREV_Y]] = narrow('prev_x, prev_y', prev, np.float32)
    players[:, _HAS_PREV] = has_prev
    players[:, _LEVEL_NOW] = narrow('level_now', np.asarray(levels, dtype=np.float64), np.float32)    # frame float or clock int
    players[:, _ALIVE], players[:, _RESPAWN] = alive, respawn
    players[:, _WALL] = terr.wall_fraction(px, py)
    players[:, _TRAITS] = traits.champion_traits(rec, levels)
    players[:, _ITEMS] = ctx.items.vectors(ref)
    _check_players(players, has_prev)
    zone = narrow('players_zone', terr.zone(px, py), np.int8)

    # [2. Objectives & Landmarks State / 오브젝트 및 표지점 상태 추적]
    obj = ctx.state_index.objective_state(ref)
    landmarks = ctx.state_index.landmark_state(ref, obj)

    # [3. Geodesic Walking Distances / 측지선 보행 거리 행렬 계산 (float16)]
    # Pairwise player distances (10x10) and player-to-landmark distances (10x34)
    # 선수 간 상호 보행 거리 및 선수-표지점 간 최단 보행 거리를 계산합니다.
    geo_pp = narrow('geo_pp', terr.geo(px[:, None], py[:, None], px[None, :], py[None, :]), np.float16)
    geo_pl = narrow('geo_pl', terr.geo(px[:, None], py[:, None], LANDMARK_XY[None, :, 0], LANDMARK_XY[None, :, 1]), np.float16)

    # [4. Recent Event Tokens / 최근 120초 이벤트 토큰화 (최신순 정렬, 최대 128개)]
    # Window: (ref_ms - 120s, ref_ms]. Older events are dropped and counted in ev_dropped.
    lo = int(np.searchsorted(ev.t, ref - C.RECENT_MS, side='right'))                           # events of the window are lo .. hi - 1
    hi = int(np.searchsorted(ev.t, ref, side='right'))
    n = min(hi - lo, MAX_EVENTS)
    newest = np.arange(hi - 1, hi - 1 - n, -1)
    ev_type, ev_team, ev_actors = np.full(MAX_EVENTS, -1, np.int8), np.full(MAX_EVENTS, TEAM_NONE, np.int8), np.zeros(MAX_EVENTS, np.int16)
    ev_dt, ev_gold, ev_xy = np.zeros(MAX_EVENTS, np.float32), np.zeros(MAX_EVENTS, np.float32), np.full((MAX_EVENTS, 2), np.nan, np.float32)
    ev_type[:n], ev_team[:n], ev_actors[:n] = ev.type[newest], ev.team[newest], ev.actors[newest]
    ev_gold[:n], ev_xy[:n] = ev.gold[newest], ev.xy[newest]
    ev_dt[:n] = narrow('ev_dt', (ref - ev.t[newest]) / 1000.0, np.float32)
    ed_i, ed_j, ed_kind = np.full(MAX_EDGES, -1, np.int8), np.full(MAX_EDGES, -1, np.int8), np.full(MAX_EDGES, -1, np.int8)
    ed_dt = np.zeros(MAX_EDGES, np.float32)

    # [5. Co-participation Edges / 사건 내 챔피언 간 상호작용 엣지 추출 (최대 128개)]
    # Extract killer-victim, assist-victim, killing pairs, damage dealers, and objective helpers.
    first, last = np.searchsorted(ev.edge_events, [lo, hi], side='left')
    m = 0
    for j in ev.edge_events[first:last][::-1]:                                                  # the newest event with edges first
        e = ev.edges[j]
        take = min(len(e), MAX_EDGES - m)
        ed_i[m:m + take], ed_j[m:m + take], ed_kind[m:m + take] = e[:take, 0], e[:take, 1], e[:take, 2]
        ed_dt[m:m + take] = narrow('ed_dt', np.full(take, (ref - ev.t[j]) / 1000.0), np.float32)
        m += take
        if m == MAX_EDGES:
            break
    window_edges = int(ev.edge_cum[hi] - ev.edge_cum[lo])  # all edges of the window, cut or not

    # [6. Global Summary Vector / 전역 요약 벡터 (107차원)]
    # Contains game time, frame age, 16 old event counts (2-10 min window), and 89 objective flags.
    summary = np.empty(len(SUMMARY), dtype=np.float32)
    summary[:2] = narrow('summary', np.array([ref / 1000.0, (ref - t_frame) / 1000.0]), np.float32)
    before_old = int(np.searchsorted(ev.t, ref - C.OLD_MS, side='right'))                       # events up to ref - 600,000 ms are too old
    summary[_OLD0:_OLD0 + _N_OLD] = ev.cum[lo] - ev.cum[before_old]                              # counts: whole numbers, exact in float32
    summary[_OBJ0:] = obj
    return {
        'players_num': players, 'players_zone': zone, 'landmarks': landmarks,
        'ev_type': ev_type, 'ev_dt': ev_dt, 'ev_team': ev_team, 'ev_xy': ev_xy, 'ev_actors': ev_actors, 'ev_gold': ev_gold,
        'ev_n': narrow('ev_n', np.array(n), np.int16), 'ev_dropped': narrow('ev_dropped', np.array(hi - lo - n), np.int16),
        'ed_i': ed_i, 'ed_j': ed_j, 'ed_dt': ed_dt, 'ed_kind': ed_kind, 'ed_n': narrow('ed_n', np.array(m), np.int16),
        'ed_dropped': narrow('ed_dropped', np.array(window_edges - m), np.int16),
        'summary': summary, 'geo_pp': geo_pp, 'geo_pl': geo_pl,
    }


def _check_players(players: np.ndarray, has_prev: np.ndarray) -> None:
    """ValueError naming the columns unless NaN stands only where the module text allows it and nothing is infinite [S2-4, R12-d]."""
    if np.isinf(players).any():
        cols = ', '.join(PLAYER_NUM[j] for j in np.flatnonzero(np.isinf(players).any(axis=0)))
        raise ValueError(f'players_num: an infinite value in {cols}')
    nan = np.isnan(players)
    if not nan.any():
        return
    ok = np.zeros_like(nan)
    ok[:, [_X, _Y]] = True                                                      # the file has no position
    ok[:, _WALL] = nan[:, _X] | nan[:, _Y]                                      # NaN wall fraction only for such a player
    ok[:, [_PREV_X, _PREV_Y]] = (~has_prev)[:, None]
    unknown = players[:, _UNKNOWN] == 1
    ok[unknown, _TRAITS.start:_UNKNOWN] = True                                  # the 18 trait values of an unknown champion
    stray = nan & ~ok
    if stray.any():
        cols = ', '.join(PLAYER_NUM[j] for j in np.flatnonzero(stray.any(axis=0)))
        raise ValueError(f'players_num: NaN in {cols}, where a value must exist')
