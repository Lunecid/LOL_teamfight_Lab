"""State of the players and of the map that comes from events: who is dead and for how long, the objective timers, and the 34 landmarks.

Everything here reads only the events of the ``MatchRecord`` with ``timestamp <= ref_ms`` (inclusive: an event on the reference ms is
visible, as in ``traits``) and the LEGACY rule tables of the match's patch. No frame value and no S1 target is read [S2-11], and the module
imports none of the S1 target, store or runner modules.

``alive_respawn(rec, ref_ms)``: two (10,) float32 arrays indexed by participant id - 1, LEGACY's ``death_intervals(events, patch, t=ref)``
through ``event_alive`` and ``respawn_remaining_s``. A death interval is [kill ms, kill ms + timer); the timer is the patch's death timer
at the victim's level at the kill (the highest LEVEL_UP level at or before the kill ms) and the game time of the kill. Measured on
2026-10-21 against LEGACY: dead on the kill ms with the whole timer left; alive again, 0 left, on the respawn ms; a LEVEL_UP on the kill ms
counts for that kill's timer (one ms later does not); a second death of the player before the respawn ends the first interval, and it is
not seen before it happens (at a ``ref_ms`` before it the first interval runs its whole length).

``objective_state(rec, ref_ms)``: the (89,) float32 vector of LEGACY's ``ObjectiveTimeline`` at ``ref_ms`` in the order of
``OBJ_STATE_NAMES`` (``feature_names``; the same 89 names for every patch of ``config.PATCHES``): spawn timers (seconds until the next
spawn, 0 while it is up), Baron and Elder buffs, tower, inhibitor and plate counts and timers, dragon, soul and monster counts, and the
flags ``rule_uncertain_*`` of the patch table. ``t_s`` is the reference time in seconds. NaN appears only where LEGACY defines it:
``NAN_FEATURES`` (``herald_next_spawn_s``, ``atakhan_next_spawn_s``, ``voidgrub_next_spawn_s``: that monster will not spawn again in this
game, or never in the patch). Any other value that is not finite raises ValueError.

``LANDMARKS``: a tuple of 34 read-only mappings (``types.MappingProxyType``: ``entry['kind']`` reads, an assignment raises TypeError;
``dict(entry)`` is a copy that can be changed, pickled or written as JSON, for a consumer that wants to mirror positions) with the
keys ``kind, team, lane, building, tower, slot, x, y``, in a fixed order: the 28 buildings of LEGACY's ``map_anchors.json`` in the
file's order (``kind`` is ``'tower'`` or ``'inhibitor'``; ``team`` the owner 100 or 200; ``building`` and ``tower`` the Match-V5 names;
the two nexus turrets of a team are ``slot`` 0 and 1) and then the six objective nests ``dragon``, ``baron``, ``herald``, ``grubs`` and
``atakhan`` twice (two places; ``team``, ``lane``, ``building`` and ``tower`` are None). The positions are medians of event positions
over 2,000 matches, in game units. ``landmark_index`` finds a building.
``landmark_state(rec, ref_ms, obj)``: (34, 2) float32, per landmark (standing / up, seconds left), from ``obj = objective_state(rec,
ref_ms)`` and the BUILDING_KILL events at or before ``ref_ms``:

    lane turrets   (1, 0) until a BUILDING_KILL fits them, then (0, 0): a lane turret never comes back
    nexus turrets  (0, seconds until it stands again) after a kill, (1, 0) after LEGACY's rule ``nexus_turret_respawn_ms``
    inhibitors     (0, ``<side>_inhib_<lane>_respawn_s`` of ``obj``) while that is above 0, else (1, 0)
    nests          (``<monster>_up``, ``<monster>_next_spawn_s`` of ``obj``); NaN seconds for herald, grubs and atakhan after the
                   monster is taken, despawned or removed from the patch (LEGACY's NaN, see above)

A BUILDING_KILL is matched to a landmark by (teamId, laneType, buildingType, towerType) (``building_kills``). A nexus turret kill takes,
of the team's two turrets, the nearest one (to the event position) among those that stand at that ms (never killed, or back up after
the rule delay); without a position, the standing one with the lowest slot; if none stands, the nearest of both (flagged
``already_down``). ``landmark_state`` refuses an ``obj`` whose shape is not that of the objective state or whose ``t_s`` is not the
reference time; it cannot tell whether ``obj`` comes from the same match.

``StateIndex(rec)`` holds what one match needs for many reference times (the LEGACY timeline of all its events, the death and level events,
the matched kills) and has the same three methods; each query reads only what is at or before ``ref_ms``. Its results equal those of the
module functions bit for bit (tested on random synthetic matches and on the golden matches, also against a match cut at ``ref_ms``;
``c_ept/checks/sweep_state.py`` compares them on every S1 match). The functions rebuild LEGACY's timeline on every call; for many
reference times of one match use the index. The cost per call, the sweep counts and the other measurements were taken on 2026-10-21 at
5680022; see the S2 report (numbers kept here would go stale).

The LEGACY modules are verified and imported when this module is imported (``OBJ_STATE_NAMES`` and ``LANDMARKS`` need them) and kept
for the rest of the process. Every cast to a narrower dtype goes through ``casts.narrow`` [S2-4].

이 모듈은 사건에서 얻는 상태 세 가지를 기준 시각 ref(정수 ms)마다 계산한다: 선수의 생존과 남은 부활 시간, 오브젝트 상태 89개 값(타이머·개수·버프·규칙 표시), 표지점 34개의 상태. 세 상태는 ref 이하의 사건과 패치의 규칙표만
읽는다. 참가자 번호는 1–10이고 배열은 번호 − 1(p = 0, …, 9)로 색인하며, 표지점 번호 w는 ``LANDMARKS``의 0 기준 행(건물 28개, 그다음 둥지 6개)이다. 생존과 오브젝트 값은 LEGACY가 계산하고 이 모듈은 값을
옮겨 float32로 줄이며(``casts.narrow``), 표지점의 포탑 칸은 이 모듈이 BUILDING_KILL을 표지점에 짝지어 정한다.

식 목록
  (STATE-1) alive_respawn: 죽음 구간 [kill_ms, end_ms)에서 alive(p)와 respawn_s(p)
  (STATE-2) landmark_state: 표지점 w의 (서 있음 또는 있음, 남은 초) 두 칸
  (STATE-3) building_kills: BUILDING_KILL의 표지점 짝짓기와 상태 이름

공통 가정과 검사. 검사하는 것: ref와 until_ms가 bool이나 float이 아닌 정수이다(TypeError); map_anchors.json이 건물 28개와 둥지 6곳을 담고 슬롯이 파일 순서의
0, 1, …이다(ValueError); 오브젝트 상태가 NAN_FEATURES 세 열 밖에서 유한하다(그 세 열도 ±∞는 거부); landmark_state의 obj가 (89,) 모양이고 그 t_s가 ref / 1000과 같다(float32로
비교하며, 같은 경기의 값인지는 검사하지 않는다).
검사하지 않는 것: 사건이 시각 순으로 놓인 것(building_kills와 StateIndex의 킬 순서가 여기에 기댄다), LEGACY 값의 의미, 표지점 위치가 사건 위치의 중앙값이라는 것.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 5.1 (layers: event-derived state), 5.2 (landmarks as tokens: towers,
#      inhibitors, nexus turrets and objective nests with state and respawn time), 5.3 (rules from the patch table) and 11 item 10 (game
#      rule reuse behind the hash-pinned adapter).
# Ref: LEGACY gameplay/event_survival.py::death_intervals / event_alive / respawn_remaining_s (death timer from CHAMPION_KILL and
#      LEVEL_UP), gameplay/objective_timers.py::ObjectiveTimeline / objective_state_at / feature_names / rule (objective, buff and
#      structure timers; rules in config/game_rules/objective_rules.json) and config/game_rules/map_anchors.json (positions of buildings
#      and nests), all loaded through the hash-pinned c_ept.legacy.
from __future__ import annotations

import json
import math
import types
from bisect import bisect_right
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import legacy
from c_ept.data.casts import narrow
from c_ept.data.reader import MatchRecord
from c_ept.domain.units import BLUE, _is_int, event_xy

N_PLAYERS = 10
NAN_FEATURES = ('herald_next_spawn_s', 'atakhan_next_spawn_s', 'voidgrub_next_spawn_s')     # the features LEGACY may give as NaN
_DEATH_EVENTS = ('CHAMPION_KILL', 'LEVEL_UP')                                               # what LEGACY's death intervals read
_LANE_SHORT = {'TOP_LANE': 'top', 'MID_LANE': 'mid', 'BOT_LANE': 'bot'}
_NESTS = (('dragon', 'DRAGON', 0), ('baron', 'BARON_NASHOR', 0), ('herald', 'RIFTHERALD', 0), ('grubs', 'HORDE', 0),
          ('atakhan', 'ATAKHAN', 0), ('atakhan', 'ATAKHAN', 1))                             # (kind, key in map_anchors.json, place)
_NEST_COLUMNS = {'dragon': ('dragon_up', 'dragon_next_spawn_s'), 'baron': ('baron_up', 'baron_next_spawn_s'),
                 'herald': ('herald_up', 'herald_next_spawn_s'), 'grubs': ('voidgrub_up', 'voidgrub_next_spawn_s'),
                 'atakhan': ('atakhan_up', 'atakhan_next_spawn_s')}                         # (up flag, seconds) of objective_state

_LEGACY = None


def _leg():
    """The pinned LEGACY modules: verified by ``legacy.load`` on the first call of this process, then kept."""
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = legacy.load()
    return _LEGACY


def _ms(value, name: str = 'ref_ms') -> int:
    """``value`` as a Python int; TypeError (naming ``name``) unless it is a whole number of milliseconds (an int or a numpy integer; not
    a bool or a float)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be a whole number of ms (int), not {type(value).__name__}')
    return int(value)


# ---- the names and the landmarks ------------------------------------------------------------------------------------------------------
OBJ_STATE_NAMES = tuple(_leg().objective_timers.feature_names('15.18'))      # the same 89 names for every patch (tested)
_T_S = OBJ_STATE_NAMES.index('t_s')
_NAN_COLUMNS = [OBJ_STATE_NAMES.index(n) for n in NAN_FEATURES]


def _load_landmarks() -> tuple:
    """The 28 buildings of ``map_anchors.json`` in its order, then the six nests, each as a read-only mapping; ValueError if the file does
    not hold exactly those."""
    _leg()                                          # the map file is one of the pinned files: the pins are checked before it is read
    data = json.loads((Path(C.LEGACY_ROOT) / 'config' / 'game_rules' / 'map_anchors.json').read_text(encoding='utf-8'))
    kinds = {'INHIBITOR_BUILDING': 'inhibitor', 'TOWER_BUILDING': 'tower'}
    out = [dict(kind=kinds[a['building']], team=int(a['team']), lane=a['lane'], building=a['building'], tower=a['tower'],
                slot=int(a['slot']), x=float(a['x']), y=float(a['y'])) for a in data['towers'] if a['building'] in kinds]
    out += [dict(kind=kind, team=None, lane=None, building=None, tower=None, slot=k, x=float(data['objectives'][key][k]['x']),
                 y=float(data['objectives'][key][k]['y'])) for kind, key, k in _NESTS]
    per_team = {(team, kind, tower): n for team in (100, 200) for kind, tower, n in (
        ('inhibitor', None, 3), ('tower', 'OUTER_TURRET', 3), ('tower', 'INNER_TURRET', 3), ('tower', 'BASE_TURRET', 3),
        ('tower', 'NEXUS_TURRET', 2))}
    if len(data['towers']) != 28 or Counter((b['team'], b['kind'], b['tower']) for b in out[:28]) != Counter(per_team):
        raise ValueError('map_anchors.json does not hold the 28 buildings (per team 3 inhibitors, 3 outer, 3 inner, 3 base, 2 nexus '
                         'turrets)')
    if [len(data['objectives'][key]) for key in ('DRAGON', 'BARON_NASHOR', 'RIFTHERALD', 'HORDE', 'ATAKHAN')] != [1, 1, 1, 1, 2]:
        raise ValueError('map_anchors.json does not hold the 6 objective places (one each for dragon, baron, herald, grubs; two for '
                         'atakhan)')
    return tuple(types.MappingProxyType(b) for b in out)          # read-only: the nexus matching reads these entries (see the docstring)


LANDMARKS = _load_landmarks()


# (STATE-3)의 후보 표이다.
def _candidates() -> dict:
    """(team, lane, building, tower) -> the indices of the buildings of that kind in slot order (one, or the two nexus turrets)."""
    found: dict = {}
    for i, lm in enumerate(LANDMARKS[:28]):
        found.setdefault((lm['team'], lm['lane'], lm['building'], lm['tower']), []).append(i)
    if any([LANDMARKS[i]['slot'] for i in rows] != list(range(len(rows))) for rows in found.values()):
        raise ValueError('map_anchors.json: the slots of a building are not 0, 1, ... in the order of the file')
    return {key: tuple(rows) for key, rows in found.items()}


_CANDIDATES = _candidates()


# (STATE-2)의 억제기·둥지 열을 정한다.
def _obj_columns(lm: dict):
    """The columns of ``objective_state`` that give a landmark its (flag, seconds), None for a turret (the events give it)."""
    if lm['kind'] == 'inhibitor':
        side = 'blue' if lm['team'] == BLUE else 'red'
        return None, OBJ_STATE_NAMES.index(f'{side}_inhib_{_LANE_SHORT[lm["lane"]]}_respawn_s')
    if lm['kind'] in _NEST_COLUMNS:
        return tuple(OBJ_STATE_NAMES.index(n) for n in _NEST_COLUMNS[lm['kind']])
    return None


_OBJ_COLUMNS = [_obj_columns(lm) for lm in LANDMARKS]


def landmark_index(team, lane, building, tower=None, slot=0) -> int:
    """The index in ``LANDMARKS`` of a building (``tower`` None for an inhibitor; ``slot`` 0 or 1 tells the two nexus turrets apart);
    KeyError if there is no such building.

    (팀, 레인, 건물, 포탑)과 슬롯으로 ``LANDMARKS``의 0 기준 행 번호를 찾는 조회이며, 그런 건물이 없거나 슬롯이 범위 밖이면 KeyError로 거부한다."""
    rows = _CANDIDATES.get((team, lane, building, tower), ())
    if not 0 <= slot < len(rows):
        raise KeyError(f'no landmark for team {team!r}, lane {lane!r}, building {building!r}, tower {tower!r}, slot {slot!r}')
    return rows[slot]


# ---- the BUILDING_KILL events matched to landmarks ------------------------------------------------------------------------------------
BuildingKill = namedtuple('BuildingKill', 't landmark status')    # landmark: index in LANDMARKS, -1 for none; status: see building_kills


def _text(e: dict, name: str) -> str:
    v = e.get(name)
    return v.upper() if isinstance(v, str) else ''


def _rule_ms(patch: str, name: str) -> int:
    return int(_leg().objective_timers.rule(name, patch).value)


def building_kills(rec: MatchRecord, until_ms=None) -> list:
    """The BUILDING_KILL events of ``rec`` with ``timestamp <= until_ms`` (all if None; otherwise a whole number of ms like ``ref_ms``,
    TypeError if not), in event order, as ``BuildingKill(t, landmark, status)``; ``landmark`` is the index of the building in
    ``LANDMARKS``, or -1, which is no index (it would read the last landmark): guard it. ``status``:

        ``ok``            the building stood and falls now
        ``after_respawn`` an inhibitor or nexus turret that was killed before and had come back by now (a respawn: nothing wrong)
        ``already_down``  the building was down when it is killed again: a lane turret killed twice, an inhibitor killed again before
                          its respawn, a nexus turret kill when neither turret of the team stood (the nearest one is taken)
        ``unmatched``     no landmark fits (teamId, laneType, buildingType, towerType): the kill changes no landmark

    A kill depends only on the kills before it, so the list for ``until_ms`` is the start of the list for a later time.

    The matching is strict: ``teamId`` must be an int, the lane must match, and an unknown ``towerType`` is unmatched. LEGACY's own
    turret counters (``structures`` of the objective state) are more lenient: they read ``int(teamId)``, count a tier whatever the lane,
    key a nexus turret by its position whatever the lane, and raise a bare KeyError on an unknown ``towerType``. So on a malformed
    event the landmark rows and LEGACY's counts can disagree (tested). S1 has no such event (measured 2026-10-21 at 5680022: 0 unmatched
    among 311,622 kills; see the S2 report); when S8 opens the holdout patches they must be swept again (``c_ept/checks/sweep_state.py``).

    A second kill of the same nexus turret while its partner stands is reassigned to the partner, with the status ``ok`` (the rule takes
    the nearest turret among those that stand). Such a case shows only through the sweep's ``building_kills_far_from_their_landmark`` and
    ``tower_count_mismatch`` counts, so 'no kill on a destroyed landmark' means something for nexus turrets only together with those two.

    수식
    ----
    (STATE-3)  후보 cand = CANDIDATES[(팀, 레인, 건물, 포탑)]  (팀은 정수일 때만, 억제기의 포탑은 None; 문자열은 대문자); cand가 비면 landmark = −1, status = unmatched
               cand = {w}: standing = [w ∉ last] ∨ (억제기 ∧ bk_ms ≥ last(w) + inhibitor_ms),  i = w
               cand = {w₀, w₁}(넥서스 포탑): up = {w ∈ cand : w ∉ last ∨ bk_ms ≥ last(w) + nexus_ms}
                 pick = up (비지 않으면) 아니면 cand,  standing = [up ≠ ∅]
                 i = pick[0] (pos_x가 NaN),  아니면 argmin_{w ∈ pick} ((anchor_x(w) − pos_x)² + (anchor_y(w) − pos_y)², w)
               status = already_down (¬standing);  after_respawn (standing ∧ i ∈ last);  ok (그 밖);  그 뒤 last(i) ← bk_ms

      기호                     | 뜻                                      | 단위 | 범위                       | 코드 변수
      bk_ms                    | BUILDING_KILL의 시각                    | ms   | 정수, until_ms 이하        | t
      w, i                     | 표지점 번호(LANDMARKS의 0 기준 행)      | —    | 0, …, 27 (건물)            | rows의 원소, i
      cand, up, pick, standing | 키에 맞는 후보 표지점, 서 있는 후보, 고르는 후보, 서 있었는지 | —    | 후보 1~2개; 참/거짓        | rows, up, pool, standing
      last(w)                  | 지금까지 w의 마지막 킬 시각             | ms   | 정수(없으면 w ∉ last)      | last
      inhibitor_ms, nexus_ms   | 억제기·넥서스 포탑의 부활 지연(패치 규칙) | ms   | 정수                       | inhibitor_ms, nexus_ms
      pos_x, pos_y             | 사건 위치(위치가 없으면 NaN)            | u    | 유한 또는 NaN              | x, y (event_xy)
      anchor_x(w), anchor_y(w) | 표지점 w의 위치                         | u    | 유한                       | LANDMARKS[w]['x'], ['y']

    해석
    ----
    식 (STATE-3)은 킬을 (팀, 레인, 건물, 포탑) 키로 표지점에 짝지으며, 키가 맞지 않거나 팀 값이 정수가 아니면 어느 표지점도 바꾸지 않는 unmatched로 둔다.
    레인 포탑과 억제기는 후보가 하나이고, 억제기는 부활 지연이 지나면 다시 서 있는 것으로 보지만 레인 포탑은 지연이 없어 한 번 부서지면 다시 서지 않는다(두 번째 킬은 already_down).
    넥서스 포탑 둘 가운데서는 그 시각에 서 있는 것 중 사건 위치에 가장 가까운 것(거리가 같으면 낮은 번호)을, 위치가 없으면 그 가운데 첫 슬롯을 고르며, 둘 다 서 있지 않으면
    둘 모두에 같은 규칙을 쓰고 already_down으로 적는다. 각 킬의 짝짓기는 앞선 킬에만 기대므로, 사건이 시각 순이면 어느 시각까지의 목록은 더 늦은 시각 목록의 앞부분이다.
    부활 시각과 같은 ms는 서 있는 것으로 센다(≥).

    근거
    ----
    설계서 5.2절(건물 표지점 토큰); S1–S2 계획 Task 10; S1–S2 판정 R10-a(ref 포함), R10-d(짝짓기 전수 점검), R10-g(S1 표적 읽지 않음).
    """
    until = None if until_ms is None else _ms(until_ms, 'until_ms')
    inhibitor_ms, nexus_ms = _rule_ms(rec.patch, 'inhibitor_respawn_ms'), _rule_ms(rec.patch, 'nexus_turret_respawn_ms')
    last: dict = {}                                          # landmark index -> ms of its latest kill so far
    out = []
    for e in rec.events:
        t = e['timestamp']
        if e.get('type') != 'BUILDING_KILL' or (until is not None and t > until):
            continue
        team, building = e.get('teamId'), _text(e, 'buildingType')
        tower = None if building == 'INHIBITOR_BUILDING' else _text(e, 'towerType')
        rows = _CANDIDATES.get((team if _is_int(team) else None, _text(e, 'laneType'), building, tower), ())
        if not rows:
            out.append(BuildingKill(t, -1, 'unmatched'))
            continue
        if len(rows) == 1:                                   # an inhibitor or a lane turret
            i = rows[0]
            delay = inhibitor_ms if building == 'INHIBITOR_BUILDING' else None            # a lane turret does not come back
            standing = i not in last or (delay is not None and t >= last[i] + delay)
        else:                                                # the two nexus turrets of the team
            up = [j for j in rows if j not in last or t >= last[j] + nexus_ms]
            pool, standing = up or rows, bool(up)
            x, y = event_xy(e)
            i = pool[0] if math.isnan(x) else min(pool, key=lambda j: ((LANDMARKS[j]['x'] - x) ** 2 + (LANDMARKS[j]['y'] - y) ** 2, j))
        # (STATE-3)의 status: 서 있지 않으면 already_down, 서 있고 전에 부서진 적이 있으면 부활 뒤의 after_respawn이다.
        out.append(BuildingKill(t, i, ('after_respawn' if i in last else 'ok') if standing else 'already_down'))
        last[i] = t
    return out


# ---- the three states -----------------------------------------------------------------------------------------------------------------
def _team_map(rec: MatchRecord) -> dict:
    return {pid: int(rec.team[pid - 1]) for pid in range(1, N_PLAYERS + 1)}


def _alive_respawn(events, patch: str, ref: int) -> tuple:
    """(STATE-1)의 alive와 respawn_s를 LEGACY에서 받아 float32로 줄인다."""
    survival = _leg().event_survival
    intervals = survival.death_intervals(events, patch, t=ref)                  # the events at or before ref only
    return (narrow('alive', survival.event_alive(intervals, ref), np.float32),
            narrow('respawn_remaining_s', survival.respawn_remaining_s(intervals, ref), np.float32))


def _vector(d: dict) -> np.ndarray:
    """The 89 values of a LEGACY state dict in the order of ``OBJ_STATE_NAMES`` as float32; NaN only in ``NAN_FEATURES``."""
    v = np.array([d[name] for name in OBJ_STATE_NAMES], dtype=np.float64)
    bad = ~np.isfinite(v)
    bad[_NAN_COLUMNS] = np.isinf(v[_NAN_COLUMNS])
    if bad.any():
        names = ', '.join(OBJ_STATE_NAMES[i] for i in np.flatnonzero(bad))
        raise ValueError(f'objective_state: LEGACY gave a value that is not finite in {names}; only {", ".join(NAN_FEATURES)} may be NaN')
    return narrow('objective_state', v, np.float32)


def _landmark_rows(patch: str, kills: list, ref: int, obj) -> np.ndarray:
    """(34, 2) float32 landmark state at ``ref`` from the matched kills at or before it and the objective state ``obj`` of that time.
    (STATE-2)의 표를 채운다."""
    obj = np.asarray(obj)
    if obj.shape != (len(OBJ_STATE_NAMES),) or np.float32(obj[_T_S]) != np.float32(ref / 1000.0):
        raise ValueError(f'landmark_state: obj must be the objective_state of the same match at ref_ms (shape ({len(OBJ_STATE_NAMES)},) '
                         f'with t_s = ref_ms / 1000), got shape {obj.shape}')
    nexus_ms = _rule_ms(patch, 'nexus_turret_respawn_ms')
    # (STATE-2)의 last(w): landmark = −1(짝 없음)은 어느 표지점의 킬도 아니므로 뺀다. 킬 목록은 호출자가 ref 이하로 잘라 넘긴다.
    last = {k.landmark: k.t for k in kills if k.landmark >= 0}                  # the latest kill of each building
    out = np.ones((len(LANDMARKS), 2))
    out[:, 1] = 0.0
    for i, (lm, cols) in enumerate(zip(LANDMARKS, _OBJ_COLUMNS)):
        if cols is None:                                                        # a turret: the kill events
            if i in last and lm['tower'] != 'NEXUS_TURRET':
                out[i] = (0.0, 0.0)                                             # a lane turret does not come back
            elif i in last:
                left = (last[i] + nexus_ms - ref) / 1000.0
                out[i] = (0.0, left) if left > 0 else (1.0, 0.0)                # a nexus turret stands again after the rule delay
        elif cols[0] is None:                                                   # an inhibitor: the countdown of the objective state
            left = float(obj[cols[1]])
            out[i] = (0.0, left) if left > 0 else (1.0, 0.0)
        else:                                                                   # a nest: its (up, seconds) of the objective state
            out[i] = (obj[cols[0]], obj[cols[1]])
    return narrow('landmark_state', out, np.float32)


def alive_respawn(rec: MatchRecord, ref_ms) -> tuple:
    """(alive, respawn_remaining_s): two (10,) float32 arrays (index = participant id - 1) at ``ref_ms``; alive is 1.0 or 0.0, the seconds
    are 0 for a living player. Reads the events at or before ``ref_ms`` only.

    수식
    ----
    (STATE-1)  p의 죽음 구간 = [kill_ms, end_ms),  end_ms = min(kill_ms + timer(킬 때의 레벨, 킬의 게임 시각), 같은 선수의 다음 kill_ms(ref 이하일 때))
               alive(p) = 0 (ref ∈ 어떤 구간), 아니면 1;   respawn_s(p) = (end_ms − ref) / 1000 (죽어 있을 때), 아니면 0

      기호         | 뜻                                          | 단위 | 범위                   | 코드 변수
      p            | 참가자 번호 − 1                             | —    | 0, …, 9                | 반환 배열의 색인
      kill_ms      | p의 CHAMPION_KILL 시각(p가 피해자)          | ms   | ≤ ref                  | LEGACY의 구간 시작
      timer        | 패치의 죽음 타이머(LEGACY가 계산)           | ms   | —                      | LEGACY의 구간 길이
      end_ms       | 구간의 끝(부활 시각, 부활 전에 다시 죽으면 그 시각) | ms   | kill_ms 초과           | LEGACY의 구간 끝
      alive(p)     | 생존 표시                                   | —    | 1.0, 0.0 (float32)     | alive
      respawn_s(p) | 남은 부활 시간                              | s    | ≥ 0 (float32)          | respawn_remaining_s

    해석
    ----
    식 (STATE-1)은 죽음 구간을 [kill_ms, end_ms)의 반열린 구간으로 두므로 킬 ms에는 죽어 있고 남은 시간이 타이머 전체이며, 부활 ms에는 살아 있고 남은 시간이 0이다.
    타이머는 킬 때의 레벨과 게임 시각으로 정해지며 킬과 같은 ms의 LEVEL_UP은 그 킬의 타이머에 들어간다. 두 번째 죽음이 부활 전에 생기면 첫 구간이 거기서 끝나고,
    ref가 두 번째 죽음 앞이면 두 번째 죽음은 보이지 않아 첫 구간이 온전하다. 구간 계산과 ref 뒤 사건의 제외는 LEGACY의 죽음 구간 계산이 하며, 이 모듈은 값을 다시 계산하지 않는다. 두 값은 float32로 줄인다.

    근거
    ----
    설계서 5.1절(사건 유래 상태); S1–S2 계획 Task 10; S1–S2 판정 R10-a(ref 이하의 사건 적용), R10-g(S1 표적 읽지 않음), S2-11(미래 정보 없음).
    """
    return _alive_respawn(rec.events, rec.patch, _ms(ref_ms))


def objective_state(rec: MatchRecord, ref_ms) -> np.ndarray:
    """(89,) float32 objective state at ``ref_ms`` in the order of ``OBJ_STATE_NAMES``; NaN only in ``NAN_FEATURES``. Builds LEGACY's
    timeline from the events at or before ``ref_ms`` on every call (``StateIndex`` for many reference times).

    LEGACY의 오브젝트 타임라인이 ref 이하의 사건으로 계산한 89개 값을 ``OBJ_STATE_NAMES`` 순서로 옮긴다. NaN은 NAN_FEATURES 세 열에서만 받고 그 밖의 비유한 값과 ±∞는
    거부하며, 값은 LEGACY 규칙표가 정하고 이 모듈은 다시 계산하지 않는다."""
    ref = _ms(ref_ms)
    return _vector(_leg().objective_timers.objective_state_at(rec.events, rec.patch, ref, _team_map(rec)))


def landmark_state(rec: MatchRecord, ref_ms, obj: np.ndarray) -> np.ndarray:
    """(34, 2) float32: for each landmark (standing / up, seconds left) at ``ref_ms``; ``obj`` is ``objective_state(rec, ref_ms)``
    (ValueError if it is of another time or shape). NaN seconds only for the nests of herald, grubs and atakhan.

    수식
    ----
    (STATE-2)  레인 포탑:  state(w) = (1, 0) (w ∉ last),  (0, 0) (w ∈ last)
               넥서스 포탑:  left(w) = (last(w) + nexus_ms − ref) / 1000;  state(w) = (0, left) (left > 0),  (1, 0) (left ≤ 0 또는 w ∉ last)
               억제기:     left(w) = obj[<진영>_inhib_<레인>_respawn_s];  state(w) = (0, left) (left > 0),  (1, 0) (그 밖)
               둥지:       state(w) = (obj[<몬스터>_up], obj[<몬스터>_next_spawn_s])

      기호       | 뜻                                          | 단위   | 범위                       | 코드 변수
      w          | 표지점 번호(0 기준 행)                      | —      | 0, …, 33                   | 반환 배열의 행
      state(w)   | (서 있음 또는 있음, 남은 초)                | —, s   | (0 또는 1, ≥ 0 또는 NaN)   | 반환 (34, 2) float32
      last(w)    | ref 이하에서 짝지은 w의 마지막 킬 시각      | ms     | 정수(없으면 w ∉ last)      | last
      nexus_ms   | 넥서스 포탑의 부활 지연(패치 규칙)          | ms     | 정수                       | nexus_ms
      ref        | 기준 시각                                   | ms     | 정수                       | ref
      obj        | ref의 오브젝트 상태(89열)                   | 열마다 | 열마다(NaN은 세 열)        | obj

    해석
    ----
    식 (STATE-2)는 표지점마다 (서 있음 또는 있음, 남은 초)를 채우며, 사건만으로 아는 포탑은 (STATE-3)의 짝지은 킬로, 억제기와 둥지는 오브젝트 상태로 정한다.
    레인 포탑은 한 번 부서지면 되돌아오지 않아 (0, 0)이고, 넥서스 포탑과 억제기는 남은 초가 양수인 동안만 (0, 남은 초)이며 0 이하이면 다시 (1, 0)이다.
    둥지는 오브젝트 상태의 (등장 여부, 다음 등장까지의 초)를 그대로 옮기므로, 전령, 공허 유충, 아타칸의 초는 LEGACY가 주는 NaN일 수 있다. 킬이 없는 포탑의 기본값은 (1, 0)이다.
    오브젝트 상태의 t_s가 ref와 다르면 이 표를 만들지 않는다.

    근거
    ----
    설계서 5.2절(건물과 둥지의 상태와 부활 시간); S1–S2 계획 Task 10; S1–S2 판정 R10-a(ref 포함), R10-g(S1 표적 읽지 않음); 짝짓기는 (STATE-3).
    """
    ref = _ms(ref_ms)
    return _landmark_rows(rec.patch, building_kills(rec, ref), ref, obj)


class StateIndex:
    """The event-derived state of one match for many reference times; build it once per match. The methods are those of the module
    functions without the record (``alive_respawn(ref_ms)``, ``objective_state(ref_ms)``, ``landmark_state(ref_ms, obj)``) and give the
    same arrays; each reads only what is at or before ``ref_ms``.

    한 경기를 여러 ref로 물을 때 LEGACY 타임라인과 짝지은 킬을 한 번만 만들고, ref마다 킬 시각의 오른쪽 이분 탐색으로 ref 이하의 킬만 쓴다. 이 자름은 킬이 시각 순이라는
    가정에 기대며 그 순서는 여기서 검사하지 않는다."""

    def __init__(self, rec: MatchRecord):
        self._patch = rec.patch
        self._death_events = tuple(e for e in rec.events if e.get('type') in _DEATH_EVENTS)
        self._timeline = _leg().objective_timers.ObjectiveTimeline(rec.events, rec.patch, _team_map(rec))
        self._kills = building_kills(rec)
        self._kill_ms = [k.t for k in self._kills]

    def alive_respawn(self, ref_ms) -> tuple:
        return _alive_respawn(self._death_events, self._patch, _ms(ref_ms))

    def objective_state(self, ref_ms) -> np.ndarray:
        return _vector(self._timeline.state(_ms(ref_ms)))

    def landmark_state(self, ref_ms, obj: np.ndarray) -> np.ndarray:
        ref = _ms(ref_ms)
        return _landmark_rows(self._patch, self._kills[:bisect_right(self._kill_ms, ref)], ref, obj)
