"""Layered views of a match at a reference time: the arrays that the model reads for one (match, reference time) pair.

``MatchContext(rec, terr)`` is built once per match (the level clock, the item replay, the event-derived state and the event arrays are
indexed there); ``build_view(ctx, ref_ms, frame_k)`` gives the view of ``ref_ms`` from the regular frame ``frame_k`` as a dict of new
numpy arrays, ``VIEW_SPEC`` (shape, dtype), ``bytes_per_view()`` bytes in all:

    players_num (10, 81) f4   one row per participant (id - 1; rows 0-4 blue, 5-9 red), columns ``PLAYER_NUM``: the 33 frame values
                              (``config.FRAME_FIELDS``), ``prev_x, prev_y`` (the position in the frame before, ``has_prev`` 1 when
                              it is there), ``level_now``, ``alive``, ``respawn_s``, ``wall_fraction`` (5 x 5 window at the position),
                              the 19 trait columns (``traits.TRAIT_NAMES``) and the 22 item columns (``traits.item_vector_names()``)
    players_zone (10,) i1     zone index at the position (``terrain.ZONES``), -1 for a missing position
    landmarks (34, 2) f4      (standing / up, seconds left) of the 34 landmarks of ``state.LANDMARKS``
    ev_*  (128,) ...          the recent event tokens, newest first: ``ev_type`` (index in ``config.EVENT_TYPES``, -1 = padding),
                              ``ev_dt`` (age in s), ``ev_team`` (0 blue, 1 red, 2 none), ``ev_xy`` (NaN without a position),
                              ``ev_actors`` (bit pid - 1 of every participant of the event), ``ev_gold``; ``ev_n`` tokens, ``ev_dropped``
                              events of the window that did not fit
    ed_*  (128,) ...          co-participation edges, newest first: ``ed_i, ed_j`` (pid - 1, -1 = padding), ``ed_dt`` (age in s),
                              ``ed_kind``; ``ed_n`` edges, ``ed_dropped`` edges of the window that did not fit
    summary (107,) f4         ``SUMMARY``: game time, frame age, 16 counts of old events, the 89 values of the objective state
                              (``game_time_s`` and ``t_s``, the first of the 89, are always equal: the plan mandates both columns,
                              and the S4 loader drops one of them)
    geo_pp (10, 10) f2        geodesic distance between the players, ``geo_pl`` (10, 34) between players and landmarks

The rows of ``ev_*`` and ``ed_*`` after the last token or edge are padding: ``ev_type``, ``ed_i``, ``ed_j`` and ``ed_kind`` -1, ``ev_team``
2, ``ev_xy`` NaN, everything else 0. ``ev_n``, ``ev_dropped``, ``ed_n`` and ``ed_dropped`` are 0-d int16 arrays.

Clocks and visibility. A view reads the events with ``timestamp <= ref_ms`` (inclusive, as ``traits`` and ``state`` do) and the frames up
to the reference time: ``frame_k`` is the regular frame the view takes its frame values from; its time must be ``<= ref_ms``, and no
other regular frame may lie strictly between it and ``ref_ms`` (a stale ``frame_k`` is a ValueError; a frame exactly at ``ref_ms`` is
allowed, so an event row on the ms of a frame takes the frame before it, as S1 does). When its time is exactly ``ref_ms`` the view is a
frame-clock row (the frame is the present); otherwise an event-clock view (the frame is ``ref_ms - frame_ts[frame_k]`` old,
``summary['frame_age_s']``). The two differ in one rule [R12-a]: a frame-clock row takes ``level_now`` and the
trait levels from the frame itself (the snapshot stays exact; a LEVEL_UP on the frame's own ms shows only as a token with age 0), an
event-clock view takes them from the level clock at ``ref_ms`` (``traits.LevelIndex``). The view reads nothing else: no S1 target, no
store, no later frame [S2-11]; this module, ``symmetry``, ``state`` and ``traits`` import none of ``targets``, ``event_rows``,
``frame_rows``, ``store_core`` or ``run_s1`` at module level (a test scans them).

Tokens. The events of the eight types of ``config.EVENT_TYPES`` with ``ref_ms - 120,000 < t <= ref_ms`` (``RECENT_MS``), the newest
first (events of one ms: the later one in the file first), at most ``MAX_EVENTS`` = 128 (S2 gate S2G-1, 2026-10-21; it was 80); the
older ones are dropped and counted in ``ev_dropped``. Actors (bit pid - 1; participant 0 and ids outside 1-10 are no champion: system
item purchases have participant 0, and the killer of an execution or of a minion-killed plate is 0): a kill has its killer, victim and
assisters, an elite monster or building its killer and assisters, a turret plate its killer, an item purchase or level-up its
participantId, a ward placement its creatorId, a ward kill its killerId. Team (0 blue, 1 red, 2 none): a kill the killer's team (an
execution, with no champion killer: 2), an elite monster ``killerTeamId``, a building or plate the team that took it (the other team of
``teamId``, the owner that lost it), the others the actor's team; a team id other than 100 or 200 is 2. Gold: bounty + shutdownBounty of
a kill, bounty of an elite monster or building, 0 for the rest.

Edges. From the kills, elite monsters and buildings of the same window (a plate has one actor and no edge), newest event first, at most
``MAX_EDGES`` = 128 (the older ones, and the last ones of an event that does not fit, are dropped and counted in ``ed_dropped``). Per
event, in this order: for a kill 0 killer-victim, 1 assister-victim, 2 the killer and the assisters in pairs (killer first, then the
assisters; the first of a pair is i; an assister is a participant 1-10 of the assist list who is neither the killer nor the victim, so
kind 1 never repeats the kind-0 edge and no pair holds the victim), 3 each distinct dealer in the victim's ``victimDamageReceived``
against the victim (ascending ids; entries on the victim's own team and entries without a participant id 1-10, which are minions,
towers and monsters, are dropped) [R12-f]; for an elite monster or building 4 the killer and the assisters in pairs. An edge is (i, j) =
(pid - 1, pid - 1) of the first and second person. The assisters come with the team of the event first (the killer's team; for a kill
without a champion killer the team that is not the victim's; for an objective its team), each team ascending: an assister list that
holds both teams then has the order that a team swap keeps, so the view of a mirrored match is exactly the swapped view
(``symmetry``). Real objective events do list assisters of both teams (measured 2026-10-21 over S1: 35,243 of 551,390 objective events;
see task-12-report.md), so an objective edge can join opponents; the edge keeps what the event says, and the relation type (ally or
enemy) is the model's to read from the two ids.

Old events. ``summary`` holds, for every type and team (``old_<type>_b``, ``old_<type>_r``), how many events fall in ``(ref_ms - 600,000,
ref_ms - 120,000]``; an event with no team (2) is not counted.

Missing values [S2-4, R12-d, R12-k]. NaN stands only where the data has no value: a missing player position (the frame's ``x``, ``y``,
and then ``wall_fraction``, the geodesics of that player, and zone -1), the previous position (``prev_x``, ``prev_y``, with ``has_prev``
0: at frame 0 or when the frame before has no position), the 18 trait values of a champion that is not in the table of the patch
(``champ_unknown`` 1), an event without a position and the padding rows of ``ev_xy``, and the LEGACY value "will not spawn again": the
three ``NAN_FEATURES`` of the summary and the seconds cell of the herald, grubs and atakhan landmark rows. ``NAN_ALLOWED`` is
exhaustive: for each of the six arrays that can hold a NaN (``players_num``, ``summary``, ``landmarks``, ``ev_xy``, ``geo_pp``,
``geo_pl``) its entry gives the ``cells`` (column names for ``players_num`` and ``summary``, (row, column) cells for ``landmarks``,
``'all'`` for the others) and the condition ``when``; a NaN in any other cell or array of a view is a bug (a test checks every array of
the golden matches' frame-clock rows and event-clock views). The S4 loader adds the mask (a NaN is never replaced by a sentinel here).
A NaN in another frame column, or an infinity anywhere, stops the view with a ValueError that names the column.

Geodesics. ``geo_pp`` and ``geo_pl`` are float16 in game units: the distance between the snapped cells of the two positions
(``Terrain.geo``), NaN where a position is missing. A snapped distance is at most 20,930 on the open 75 x 75 grid (74 diagonal steps; the
diagonal of its 15,000 u square is 21,213), where a float16 has a step of 16 units, so the stored value is within 8 units of the table's
and the table's own step is 1 unit; the cast is range-checked and raises on an overflow or an infinity (``casts.narrow``). Landmark
positions are those of ``state.LANDMARKS``, absolute, read only.

Every cast to a narrower dtype goes through ``casts.narrow``; no match key, player name or ID is in a view.

이 모듈은 한 (경기, 기준 시각) 쌍을 모형이 읽는 층별 배열 묶음, 곧 보기로 만든다. 기준 시각 ref는 정수 ms이고, 정규 프레임은 ``MatchRecord.frame_ts``의 항목이며 frame_k는 그
0 기준 색인, t_frame = frame_ts[frame_k]는 그 프레임의 시각이다. 참가자 번호는 1–10이고 배열의 선수 행은 번호 − 1(p = 0–4 파랑, 5–9 빨강)이다. 프레임 시계 행은 t_frame = ref인 보기,
사건 시계 보기는 t_frame < ref인 보기이다. 토큰은 최근 사건 하나의 한 줄, 간선은 한 사건에 함께 나온 두 참가자의 한 줄이며, 낡은 사건은 개수로만 요약한다. 한 보기는 프레임 값, ref까지의
사건에서 얻은 상태(``state``), 토큰과 간선, 낡은 사건의 개수, 측지 거리를 담는다.

식 목록
  (TOK-1) build_view: 최근 사건 창, 토큰 칸의 배정(새 사건부터), 나이 ev_dt, 잘린 수 ev_dropped
  (TOK-2) build_view: 간선 칸의 채움(새 사건부터), 나이 ed_dt, 잘린 수 ed_dropped
  (TOK-3) build_view: 낡은 사건의 (유형, 팀)별 개수
  (TOK-4) build_view: 허용하는 frame_k, 프레임 나이 a, 프레임 시계 행과 사건 시계 보기의 레벨 선택
  (TOK-5) build_view: 측지 거리 geo_pp, geo_pl
  (TOK-6) MatchContext: 사건 하나의 기록(행위자 마스크, 팀 코드, 골드, 간선 목록)
  (TOK-7) bytes_per_view: 한 보기의 바이트 수

공통 가정과 검사. 검사하는 것: ref와 frame_k는 정수이고 frame_k는 정규 프레임 범위 안이며 t_frame ≤ ref이고 더 최근의 정규 프레임이 사이에 없다; 사건 시각이 시간 순이다; 사건 필드가 정수 범위나 유한한 수이다;
frame 값은 위치 밖에서 NaN이 없고 players_num에는 허용 밖의 NaN과 ±∞가 없다(``_check_players``); 한 경기의 팀 배치가 1–5 파랑, 6–10 빨강이다.
검사하지 않는 것: frame_ts가 오름차순인 것, 지형이 그 경기 패치의 것인 것, NAN_ALLOWED가 players_num 밖의 다섯 배열에서 지켜지는 것(구성상 그렇게 만들고, 시험은 골든 경기의 보기에서 확인한다), 측지 거리의
float16 오차(영어 설명의 한계). 영어 본문의 (i, j)는 열 이름 ed_i, ed_j이며 이 모듈의 수식은 쓰지 않는다.
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
from c_ept import state, traits
from c_ept.casts import narrow
from c_ept.reader import MatchFormatError, MatchRecord
from c_ept.units import BLUE, RED, _assist_mask, _event_id, _is_int, event_xy

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
    """Everything a view needs from one match, indexed once: ``rec`` (the MatchRecord), ``terr`` (the terrain; anything that answers
    ``geo``, ``zone`` and ``wall_fraction`` as ``terrain.Terrain`` does), the level clock (``levels``), the item replay (``items``),
    the event-derived state (``state_index``) and the event arrays (``events``).

    Build one per match and ask it for many views; building one costs a few ms, an item replay alone 1-2 ms per view. A context holds
    no S1 object and reads no target: it is made from a record and a terrain only [S2-11]. ValueError unless the record's teams are
    the usual layout (participants 1-5 team 100, 6-10 team 200), which the swap of ``symmetry`` relies on.

    수식
    ----
    (TOK-6)  bit(pid) = 2^(pid − 1) (1 ≤ pid ≤ 10), 그 밖은 0;   side(pid) = 0 (1–5), 1 (6–10), 2 (그 밖)
             code(id) = 0 (id = 100), 1 (id = 200), 2 (그 밖)
             taker(id) = 2 (code(id) = 2), 그 밖은 1 − code(id);   order(pids, lead) = pids를 (side ≠ lead, 번호) 순으로 정렬;
             사건 유형마다 (행위자 마스크 / 팀 / 골드):
               CHAMPION_KILL              bit(killer) ∨ bit(victim) ∨ assists   /  side(killer)               /  bounty + shutdownBounty
               ELITE_MONSTER_KILL         bit(killer) ∨ assists                  /  code(killerTeamId)         /  bounty
               BUILDING_KILL              bit(killer) ∨ assists                  /  taker(teamId)              /  bounty
               TURRET_PLATE_DESTROYED     bit(killer)                            /  taker(teamId)              /  0
               ITEM_PURCHASED, LEVEL_UP, WARD_PLACED, WARD_KILL
                                          bit(pid), pid = participantId, participantId, creatorId, killerId   /  side(pid)  /  0
             간선 (from, to, 종류), from과 to는 참가자 번호 − 1; 목록의 쌍은 목록에서 앞선 원소를 from, 뒤의 원소를 to로 하는 모든 조합:
               CHAMPION_KILL:  0 (killer, victim) [killer ≥ 1];  1 (helper, victim) 각 helper
                 2 ([killer] (killer ≥ 1일 때) + helpers)의 쌍;  3 (dealer, victim) 각 dealer
                 helpers = order(assists ∖ {killer, victim}, lead), lead = side(killer) (killer = 0이면 1 − side(victim))
                 dealers = victimDamageReceived의 참가자 1–10 가운데 피해자의 반대 팀, 오름차순
               ELITE_MONSTER_KILL, BUILDING_KILL:  4 ([killer] (killer ≥ 1일 때) + order(assists ∖ {killer}, lead))의 쌍,
                 lead = 사건의 팀(2이면 side(killer))
             cum[e, 2 · type + team] = 색인이 e보다 작은 사건 가운데 그 (유형, 팀)인 것의 수(team < 2인 사건만),  e = 0, …, n_events

      기호          | 뜻                                          | 단위 | 범위                     | 코드 변수
      pid           | 참가자 번호                                 | —    | 정수, 1–10만 챔피언      | killerId, victimId, participantId, creatorId
      killer, victim | 킬러, 피해자(0은 챔피언 아님)              | —    | killer 0–10, victim 1–10 | _event_id의 값
      assists       | 도움 참가자의 마스크                        | —    | 0, …, 1,023              | _assist_mask
      helpers, dealers | 도움 참가자 목록, 딜러 목록              | —    | 번호의 오름차순 규칙     | helpers, _dealers
      lead          | 정렬에서 앞에 두는 팀(사건의 팀)            | —    | 0, 1, 2                  | _order의 인자 ref, killing
      type, team    | 사건 유형 색인(0–7), 팀 코드                | —    | 0–7; 0, 1, 2             | _KIND, TEAM_*
      e             | 토큰 사건 배열의 0 기준 색인(누적표의 행)   | —    | 0, …, n_events           | _Events의 행 (파일 번호가 아니다)
      cum           | (유형, 팀)별 누적 개수                      | 개   | 0, 1, …                  | cum (n_events + 1, 16) int32

    해석
    ----
    식 (TOK-6)은 사건 유형마다 누가 행위자이고 어느 팀의 사건이며 골드가 얼마인지와 어떤 간선을 내는지를 정한다. 참가자 0은 챔피언이 아니어서 마스크에 비트를 더하지 않고, 팀 코드 2는 팀 없음이며
    낡은 사건 개수에서 빠진다. 도움 목록이 두 팀을 포함할 수 있어, 사건의 팀을 앞에 두는 순서를 쓴다. 팀 교환이 참가자 번호에 5를 더해(mod 10) 팀 안의 순서를 보존하므로 교환 뒤에도 같은 규칙의 순서가 된다.
    오브젝트 간선(종류 4)도 도움 목록에 두 팀이 있으면 상대끼리 이어지며 그대로 둔다.

    근거
    ----
    설계서 5.2절(토큰, 공동 참여 간선), 11절 항목 13–14(사건 토큰, 간선); S1–S2 판정 R12-f(피해 간선의 딜러), R12-b(경기별 캐시), S2 관문 S2G-5(두 팀을 잇는 오브젝트 간선 유지).
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
    """The view (dict of numpy arrays, see the module text) of ``ctx.rec`` at ``ref_ms`` from the regular frame ``frame_k``.

    ``ref_ms`` must be a whole number of ms and not before the frame (``frame_ts[frame_k]``): ValueError otherwise (a frame in the
    future is never read); a stale ``frame_k`` (another regular frame lies strictly between it and ``ref_ms``) is a ValueError too, so
    a view never takes an older frame than the last one before its reference time. The frame is the present when its time is
    ``ref_ms`` (a frame-clock row, which takes the level from the frame), otherwise the last frame before ``ref_ms`` (an event-clock
    view, which takes the level from the level clock) [R12-a]. Events with ``t <= ref_ms`` are seen. The result holds new arrays only;
    calls do not change the context.

    수식
    ----
    (TOK-1)  win_lo = #{e : t_e ≤ ref − RECENT_MS},  win_hi = #{e : t_e ≤ ref};  창의 사건 = {win_lo, …, win_hi − 1}
             n = min(win_hi − win_lo, MAX_EVENTS)
             토큰 칸 slot = 0, …, n − 1은 사건 e = win_hi − 1 − slot (새 사건부터);  ev_dt[slot] = (ref − t_e) / 1000
             ev_n = n;  ev_dropped = win_hi − win_lo − n;  칸 n 이후는 패딩
    (TOK-2)  창의 간선 사건 = {e ∈ edge_events : win_lo ≤ e < win_hi}를 e 내림차순으로, 사건의 간선을 (TOK-6)의 순서로 칸 edge_slot = 0, 1, …에 넣되 MAX_EDGES에서 멈춤
             ed_dt = (ref − t_e) / 1000;  ed_n = edge_slot;  ed_dropped = edge_cum[win_hi] − edge_cum[win_lo] − ed_n
    (TOK-3)  before_old = #{e : t_e ≤ ref − OLD_MS}  (team 0 파랑, 1 빨강)
             summary[old_<type>_<team>] = cum[win_lo, 2 · type + team] − cum[before_old, 2 · type + team]
    (TOK-4)  t_frame = frame_ts[frame_k];  허용 ⇔ 0 ≤ frame_k < 프레임 수, t_frame ≤ ref, 그리고 (frame_k가 마지막이거나 frame_ts[frame_k + 1] ≥ ref)
             a = (ref − t_frame) / 1000;  summary[game_time_s, frame_age_s] = (ref / 1000, a)
             레벨 = 프레임의 level (t_frame = ref),  레벨 시계 at(ref) (t_frame < ref)
    (TOK-5)  geo_pp[p, p′] = geo(위치(p), 위치(p′)),  geo_pl[p, w] = geo(위치(p), LANDMARK_XY[w]);  위치가 없으면 NaN;  float16으로 줄임

      기호          | 뜻                                          | 단위 | 범위                       | 코드 변수
      ref           | 기준 시각                                   | ms   | 정수, ≥ t_frame            | ref
      e, t_e        | 토큰 사건 배열의 0 기준 색인, 그 사건의 시각 | —, ms | 0, …, n_events − 1; 정수   | ev.t
      win_lo, win_hi | 창의 첫 색인, 창 뒤의 첫 색인              | —    | 0, …, n_events             | lo, hi
      n, slot       | 토큰 수, 토큰 칸(0부터)                     | —    | 0 … MAX_EVENTS            | n, ev_* 배열의 색인
      edge_slot     | 간선 칸(0부터)                              | —    | 0 … MAX_EDGES − 1          | m
      type, team    | 사건 유형 색인, 팀 0 또는 1                 | —    | 0–7; 0, 1                  | cum의 열 2 · type + team
      a             | 프레임 나이                                 | s    | ≥ 0 (float32)              | frame_age_s
      p, p′, w      | 선수 행(0부터), 표지점 행(0부터)            | —    | 0–9; 0–33                  | geo의 축
      ev_dt, ed_dt  | 사건과 간선의 나이                          | s    | ≥ 0 (float32)              | ev_dt, ed_dt

    해석
    ----
    식 (TOK-1)의 최근 사건 창은 오른쪽이 닫혀 있어 ref와 같은 ms의 사건도 보이며, 같은 ms의 사건끼리는 파일에서 나중의 것이 앞 칸에 온다. 창의 사건이 토큰 칸보다
    많으면 오래된 사건을 잘라 그 수를 따로 센다. 식 (TOK-2)는 간선을 새 사건부터 채우며, 칸이 차면 그 사건의 앞부분 간선만 넣고 나머지와 그보다 오래된 사건의 간선은
    잘린 간선 수로 센다. 식 (TOK-3)의 낡은 사건은 최근 사건 창 바로 앞 구간의 사건이고, 팀이 없는 사건(팀 코드 2)은 세지 않는다. 식 (TOK-4)는 프레임이 ref 이하이고
    그 사이에 다른 정규 프레임이 없기를 요구하며, ref와 같은 시각의 프레임은 사이에 낀 것으로 보지 않으므로 그 프레임과 바로 앞 프레임이 모두 허용된다.

    근거
    ----
    설계서 4절(두 시계, 가림), 5.1절(층: 최근 사건, 낡은 사건 개수, 프레임, 사건 유래 상태), 11절 항목 7–8(프레임 시각, 보기와 프레임); S1–S2 판정 R12-a(프레임 시계 행의 레벨), R12-c(ref 포함 가시성),
    R12-d(결측 위치), R12-e(측지 float16), S2-4(좁히는 변환은 검사), S2-11(미래 정보 없음); S2 관문 S2G-1(사건 상한).
    """
    rec = ctx.rec
    ref = _ms(ref_ms, 'ref_ms')
    k = _ms(frame_k, 'frame_k')
    if not 0 <= k < len(rec.frame_ts):
        raise ValueError(f'frame_k must be a regular frame, 0 .. {len(rec.frame_ts) - 1}, got {k}')
    # (TOK-4)의 허용 조건: 아래 두 검사가 t_frame ≤ ref와 사이에 다른 정규 프레임이 없을 것을 가린다.
    t_frame = int(rec.frame_ts[k])
    if t_frame > ref:
        raise ValueError(f'frame {k} is at {t_frame} ms, after the reference time {ref} ms: '
                         'a view reads frames up to its reference time only')
    if k + 1 < len(rec.frame_ts) and int(rec.frame_ts[k + 1]) < ref:
        raise ValueError(f'frame_k {k} is stale: the regular frame {k + 1} at {int(rec.frame_ts[k + 1])} ms lies before the reference '
                         f'time {ref} ms; a view takes the last regular frame before its reference time')
    terr, ev = ctx.terr, ctx.events

    # -- players
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

    # -- the objective state and the landmarks
    obj = ctx.state_index.objective_state(ref)
    landmarks = ctx.state_index.landmark_state(ref, obj)

    # -- geodesics (float16, game units)
    # (TOK-5): 위치가 없는 선수의 행과 열은 geo가 NaN으로 주며, float16으로 줄일 때 범위를 검사한다.
    geo_pp = narrow('geo_pp', terr.geo(px[:, None], py[:, None], px[None, :], py[None, :]), np.float16)
    geo_pl = narrow('geo_pl', terr.geo(px[:, None], py[:, None], LANDMARK_XY[None, :, 0], LANDMARK_XY[None, :, 1]), np.float16)

    # -- tokens, edges, old events
    # (TOK-1)의 win_lo, win_hi: 오른쪽 탐색이므로 창은 (ref − RECENT_MS, ref]이다. 토큰은 새 사건부터 칸에 놓는다.
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
    # (TOK-2): 창의 간선 사건을 새 사건부터 채우고, 칸이 차면 멈춘다. 멈춘 뒤의 간선은 아래 window_edges에서 ed_dropped로 센다.
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
    summary = np.empty(len(SUMMARY), dtype=np.float32)
    summary[:2] = narrow('summary', np.array([ref / 1000.0, (ref - t_frame) / 1000.0]), np.float32)
    # (TOK-3): 낡은 사건의 개수는 누적표의 두 행의 차이이다.
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
