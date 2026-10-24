"""The team swap: the 180-degree rotation of the map with the two teams exchanged, for views, marks and targets.

The map rotation is (pos_x, pos_y) -> (map_width - pos_x, map_width - pos_y) with map_width = 14,875 (``config.W``), and the teams change
places: blue becomes red. So a swapped match is the same match seen from the other side, and the model can be trained on both sides with one
set of weights. This module is the arithmetic of that swap and nothing else: it does not know a loader, a model or a target table.

``swap_view(v)`` swaps a view of ``tokens.build_view`` (docstring there for the arrays); ``swap_mark``, ``swap_K`` and ``swap_O`` swap
the first-kill mark (52 marks), the kill gold K (8 cells) and the objective counts O (18 cells); ``swap_actors`` and
``swap_participants`` are the two pieces of ``swap_view`` that a loader needs for other arrays, and ``OBJ_SWAP`` is the permutation of
the 89 objective-state columns. Every function returns new arrays and,
on its documented domain (integer positions), is its own inverse: applying it twice gives back the input.
``LANDMARK_SWAP`` is the permutation of the 34 landmark rows (and of the matching columns of ``geo_pl``) that ``swap_view`` leaves in
place and one permitted S4 loader strategy applies; it is derived from the landmark positions by ``derive_landmark_swap`` and checked at
import to be its own inverse.

The rotation turns the top lane into the bottom lane and the blue base into the red base; the objective nests and the map-level flags
stay what they are (the dragon pit does not become a baron pit: the loader flags the broken symmetry with a side marker, spec 5.2 and
the JSD of the rotated nest positions, 0.55-0.57). So in ``OBJ_SWAP`` the blue and red copies of a column change places, with the lane
crosswise for the lane columns (``blue_inhib_top_respawn_s`` and ``red_inhib_bot_respawn_s`` are partners), and everything else stays.
In ``LANDMARK_SWAP`` a building changes places with the building of the other team that lies at its rotated position (the same kind and
tier, the lane crosswise, the two nexus turrets matched by place, 133.2 u from the rotated place at most), and a nest stays.

방법에서의 역할
---------------
이 모듈은 한 경기를 상대 진영에서 본 모습으로 바꾸는 배열 연산을 모은다. 변환은 지도를 180° 돌리는 (pos_x, pos_y) ↦ (map_width − pos_x, map_width − pos_y)와
두 팀의 자리 교환이며, 모형이 한 벌의 가중치로 양쪽 진영을 학습하게 하려는 것이다 [설계서 5.2절]. 이 모듈은 보기, 표지, 결과 K와 O의 치환 산술만 맡고 로더,
모형, 목표 표를 알지 못한다. 치환은 정수 위치 영역(문서에 적은 영역)에서 두 번 적용하면 입력으로 돌아오는 자기 역이다. 그러나 지도의 둥지와 지도 전체 플래그는 회전해도
제자리이므로 이 변환은 물리적으로 정확한 대칭이 아니며, 그 비대칭은 로더가 진영 표시로 맡는다 [설계서 5.2절].

식 목록
-------
    (SYM-1)  위치의 180° 회전                         `swap_view`
    (SYM-2)  참가자 번호 치환                         `swap_participants`
    (SYM-3)  행위자 비트 마스크 치환                  `swap_actors`
    (SYM-4)  첫 킬 표지 치환                          `swap_mark`
    (SYM-5)  K와 O의 팀 칸 교환                       `swap_K` (`swap_O`, `_pair_swap`)
    (SYM-6)  건물 짝과 최대 어긋남                    `derive_landmark_swap`
    (SYM-7)  보기 전체의 치환                         `swap_view`
    (SYM-8)  객체 상태 열의 짝                        `swap_view` (`OBJ_SWAP`, `_partner`, `_obj_swap`)

가정
----
- 회전 상수 map_width는 코드의 `W`(`config.W`) 하나이고 위치 좌표 (pos_x, pos_y)는 코드의 x, y 열이다. 지형 격자는 [0, 15,000] 범위이고 map_width는 이보다 작으므로,
  보기에 담긴 벽 비율과 측지 거리는 회전한 위치에서 다시
  계산하지 않고 선수와 함께 옮긴다.
- 위치는 map_width에서 빼므로 NaN은 NaN으로 남고, 문서에 적은 정수 위치 영역에서는 정확하다. 분수 위치의 float32 뺄셈은 한 번 반올림될 수 있어 자기 역이
  정확하지 않을 수 있다.
- 참가자 번호 −1(없음)과 구역 −1(위치 없음)은 치환하지 않는다. 팀 코드 2(없음)도 그대로이다.
- 표지점 34행과 `geo_pl`의 34열은 `swap_view`가 제자리에 둔다. 적재기는 `LANDMARK_SWAP`으로 행과 열을 치환하거나, 행 정체성 없이 소유와 물리 위치를
  따로 실어 나른다(둘 중 하나; S1–S2 판정 60, R12-i). 보기의 키와 모양이 `tokens.VIEW_SPEC`과 다르면 ValueError이다(검사된다).
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 5.2 (symmetry: each team in its own coordinate system, rotation by 180
#      degrees (pos_x, pos_y) -> (map_width - pos_x, map_width - pos_y), map_width = 14,875, plus a side marker; outputs swapped with it;
#      JSD of the team-swapped position distribution: no transform 0.245, diagonal reflection 0.353, 180-degree rotation 0.112
#      [measured: symmetry_check.json]) and section 10, law A1.
from __future__ import annotations

import math
import re

import numpy as np

from c_ept import config as C
from c_ept import state, terrain, tokens

W = C.W
PLAYER_ORDER = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])           # row p of a swapped view is row PLAYER_ORDER[p] of the view
PLAYER_ORDER.setflags(write=False)
_XY_COLUMNS = [tokens.PLAYER_NUM.index(name) for name in ('x', 'y', 'prev_x', 'prev_y')]
_N_OLD = len(tokens.OLD_COLUMNS)
_LANE = {'top': 'bot', 'mid': 'mid', 'bot': 'top'}                  # the rotation turns the top lane into the bottom lane
_LANE_COLUMN = re.compile(r'(inhib|plates)_(top|mid|bot)(_respawn_s)?')


def _integers(name: str, a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must hold whole numbers (int), not {a.dtype}')
    return a


# ---- participants, actor bits, marks, K and O ------------------------------------------------------------------------------------
def swap_participants(p) -> np.ndarray:
    """Participant indices (pid - 1, 0-9) after the swap: (p + 5) % 10; -1 (no participant) stays -1. ValueError outside -1 .. 9.

    수식
    ----
    (SYM-2)  p′ = (p + 5) mod 10   (p ≥ 0),     p′ = −1   (p = −1)

      기호 | 뜻                          | 단위 | 범위       | 코드 변수
      p    | 참가자 번호 − 1(0부터)      | —    | −1, 0, …, 9 | p
      p′   | 상대 진영에서 본 참가자 번호 | —    | −1, 0, …, 9 | 반환값

    해석
    ----
    파랑은 0–4, 빨강은 5–9이므로 5를 더하고 10으로 나눈 나머지가 두 팀의 같은 슬롯을 서로 바꾸고, 두 번 적용하면 p로 돌아온다. −1은 참가자가 없음을 뜻하므로
    바꾸지 않는다. 범위 밖 번호는 어느 슬롯에도 대응하지 않아 거부한다.

    근거
    ----
    설계서 5.2절(진영 교환과 출력의 같은 교환), S1–S2 계획 Task 12(ed_i, ed_j의 치환 규칙).
    """
    p = _integers('participants', p)
    if ((p < -1) | (p > 9)).any():
        raise ValueError('participants must be -1 (none) or 0 .. 9')
    return np.where(p >= 0, (p + 5) % 10, p)


def swap_actors(a) -> np.ndarray:
    """Actor bit masks (bit pid - 1, 10 bits) after the swap: the low five bits (blue) and the high five bits (red) change places.
    ValueError outside 0 .. 1,023.

    수식
    ----
    (SYM-3)  mask′ = ((mask ∧ 31) ≪ 5) ∨ ((mask ≫ 5) ∧ 31)

      기호  | 뜻                                | 단위 | 범위        | 코드 변수
      mask  | 행위자 비트 마스크(비트 pid − 1)  | —    | 0, …, 1,023 | a
      mask′ | 상대 진영에서 본 마스크           | —    | 0, …, 1,023 | 반환값

    해석
    ----
    마스크의 아래 5비트는 파랑 참가자, 위 5비트는 빨강 참가자이므로 두 묶음의 자리를 바꾼다. 같은 슬롯의 비트가 반대 팀의 같은 슬롯 비트로 가므로 (SYM-2)와
    같은 치환을 비트에서 하며, 두 번 적용하면 mask로 돌아온다. 10비트 밖의 값은 거부한다.

    근거
    ----
    설계서 5.2절, S1–S2 계획 Task 12(ev_actors의 치환 규칙).
    """
    a = _integers('actors', a)
    if ((a < 0) | (a > 1023)).any():
        raise ValueError('an actor mask holds ten bits: 0 .. 1023')
    return ((a & 31) << 5) | ((a >> 5) & 31)


def swap_mark(m):
    """The first-kill mark (0 .. 51, ``units.first_kill_mark``) after the swap. Marks 0-49 are 25 d + 5 i + j (d = 0 for a blue killer,
    i the killer's slot, j the victim's slot): d flips and the slots stay, so (m + 25) % 50. Marks 50 and 51 are the executions of a blue
    and a red victim: they change places. An int gives an int, an array an array of its dtype; ValueError outside 0 .. 51.

    수식
    ----
    (SYM-4)  m′ = (m + 25) mod 50   (m < 50),     m′ = 101 − m   (m ∈ {50, 51})
             m = 25·d + 5·i + j 이면 m′ = 25·(1 − d) + 5·i + j  (m < 50)

      기호 | 뜻                           | 단위 | 범위     | 코드 변수
      m    | 첫 킬의 표지 (UNIT-4)        | —    | 0, …, 51 | m
      m′   | 상대 진영에서 본 표지        | —    | 0, …, 51 | 반환값
      d    | 킬러의 진영(0 파랑, 1 빨강)  | —    | 0, 1     | —
      i, j | 킬러·피해자의 슬롯(0부터)    | —    | 0, …, 4  | —

    해석
    ----
    쌍 표지에서는 킬러의 진영 d만 뒤집히고 슬롯 i, j는 그대로이므로 25를 더하고 50으로 나눈 나머지가 되며, 처형 표지 50(파랑 피해자)과 51(빨강 피해자)은 서로 자리를
    바꾼다. 두 번 적용하면 m으로 돌아온다. 표지는 슬롯 번호이지 역할이 아니다 [설계서 5.2절]. 표지 번호 밖의 값은 거부한다.

    근거
    ----
    설계서 3.1절(표지 m)과 5.2절(출력의 교환), 11절 항목 2(처형 표지), (UNIT-4)(표지 번호 규칙).
    """
    a = _integers('mark', m)
    if ((a < 0) | (a >= C.N_MARKS)).any():
        raise ValueError(f'a mark is 0 .. {C.N_MARKS - 1}')
    out = np.where(a < 50, (a + 25) % 50, 101 - a)
    return int(out) if out.ndim == 0 else out


def _pair_swap(name: str, x, n: int) -> np.ndarray:
    """``x`` with the cells (2k, 2k + 1) of its last axis exchanged: the layout 2 * field + team (0 blue, 1 red) of K and O.

    (SYM-5)의 칸 교환을 계산한다.
    """
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] != n:
        raise ValueError(f'{name} must have {n} cells on its last axis, got shape {x.shape}')
    return x[..., np.arange(n) ^ 1]


def swap_K(K) -> np.ndarray:
    """The kill gold K (8 cells: kills, bounty, shutdown, assists, each for blue then red; ``targets.K_FIELDS``) after the swap: the two
    teams of every field change places. The last axis has 8 cells; further axes are batches.

    수식
    ----
    (SYM-5)  K′[…, c] = K[…, c ⊕ 1]     (c = 0, …, n − 1;  K: n = 8,  O: n = 2·(오브젝트 종류의 수))

      기호 | 뜻                                 | 단위      | 범위           | 코드 변수
      K, O | 결과의 킬 부분(칸 8개), 오브젝트 부분(칸 2·종류 수개) | 골드·개 | ≥ 0 | K, O
      c    | 마지막 축의 칸 번호(0부터)         | —         | 0, …, n − 1    | `np.arange(n)`
      ⊕    | 비트 배타적 논리합(마지막 비트 뒤집기) | —     | —              | `^ 1`

    해석
    ----
    칸이 2·필드 + 팀(0 파랑, 1 빨강)으로 놓여 있으므로 번호의 마지막 비트를 뒤집으면 필드마다 파랑 칸과 빨강 칸이 서로 바뀌고, 두 번 적용하면 원래로 돌아온다.
    마지막 축의 칸 수가 n이 아니면 거부하고 앞의 축은 묶음으로 본다. 결과는 새 배열이다.

    근거
    ----
    설계서 5.2절(출력의 교환), `targets.K_FIELDS`와 `config.O_KINDS`(칸 배치).
    """
    return _pair_swap('K', K, 8)


def swap_O(O) -> np.ndarray:                                                          # noqa: E741 (the brief's name for the argument)
    """The objective counts O (18 cells: 9 kinds, each for the blue then the red taker; cell = 2 * kind + taker) after the swap: the two
    teams of every kind change places. The last axis has 18 cells; further axes are batches.

    식 (SYM-5)를 n = 2·(오브젝트 종류의 수)로 계산한다.
    """
    return _pair_swap('O', O, 2 * len(C.O_KINDS))


# ---- the objective state ---------------------------------------------------------------------------------------------------------
def _partner(name: str) -> str:
    """The column that a column of the objective state turns into: blue and red exchange, the lane of a lane column crosses.

    (SYM-8)의 짝 규칙을 계산한다.
    """
    if not name.startswith(('blue_', 'red_')):
        return name                                                    # nests, timers and map-level flags belong to no team
    side, rest = name.split('_', 1)
    lane = _LANE_COLUMN.fullmatch(rest)
    if lane:
        rest = f'{lane.group(1)}_{_LANE[lane.group(2)]}{lane.group(3) or ""}'
    return f'{"red" if side == "blue" else "blue"}_{rest}'


# (SYM-8)의 OBJ_SWAP을 계산한다(식은 `swap_view`의 문서).
def _obj_swap() -> np.ndarray:
    names = tuple(state.OBJ_STATE_NAMES)
    index = {name: i for i, name in enumerate(names)}
    partners = [_partner(name) for name in names]
    unknown = sorted(p for p in partners if p not in index)
    if unknown:
        raise ValueError(f'the objective state has no column {unknown} to swap with')
    perm = np.array([index[p] for p in partners], dtype=np.int64)
    if sorted(perm.tolist()) != list(range(len(names))) or not np.array_equal(perm[perm], np.arange(len(names))):
        raise ValueError('the swap of the objective-state columns is not a permutation that is its own inverse')
    perm.setflags(write=False)
    return perm


# (SYM-8)의 식과 기호는 `swap_view`의 문서에 있다. 이 상수는 그 이름 규칙으로 import 때 만든다.
OBJ_SWAP = _obj_swap()          # (89,) int64: column c of a swapped objective state is column OBJ_SWAP[c] of it (its own inverse)


def _summary_swap() -> np.ndarray:
    perm = np.arange(len(tokens.SUMMARY))
    old0, obj0 = tokens.SUMMARY.index(tokens.OLD_COLUMNS[0]), tokens.SUMMARY.index(state.OBJ_STATE_NAMES[0])
    perm[old0:old0 + _N_OLD] = old0 + (np.arange(_N_OLD) ^ 1)          # old_<type>_b and old_<type>_r are neighbours
    perm[obj0:] = obj0 + OBJ_SWAP
    perm.setflags(write=False)
    return perm


# Public summary-column permutation of (SYM-7); the legacy alias shares its identity.
SUMMARY_SWAP = _summary_swap()
# Compatibility alias: both names refer to the same permutation.
_SUMMARY_SWAP = SUMMARY_SWAP
if terrain.ZONES[terrain.ZONE_SWAP[terrain.ZONES.index('top')]] != 'bot':
    raise RuntimeError('the zone rules and the lane rule of OBJ_SWAP disagree: the rotation must turn the top lane into the bottom lane')


# ---- the landmark rows -------------------------------------------------------------------------------------------------------------
_LANDMARK_LANE = {f'{lane.upper()}_LANE': f'{other.upper()}_LANE' for lane, other in _LANE.items()}      # TOP_LANE <-> BOT_LANE
_PARTNER_MAX_OFFSET = 200.0         # map units (one terrain cell): the farthest a rotated building may lie from its partner


def derive_landmark_swap(landmarks) -> tuple[np.ndarray, float]:
    """``(permutation, offset)``: the rows of ``landmarks`` (the entries of ``state.LANDMARKS``, or copies of them) that the team swap
    exchanges, and the largest distance, in map units, between a building rotated by (map_width - pos_x, map_width - pos_y) and its partner.

    A building has an owner (``team`` 100 or 200). Its partner is the building of the other team, of the same ``kind``, ``building`` and
    ``tower``, that lies nearest to its rotated position (a tie goes to the lower row). So the pairing comes from the positions, and the
    rule behind ``OBJ_SWAP``, that the rotation turns the top lane into the bottom lane, is checked against it and does not make it;
    the two nexus turrets of a team, which have the same kind and tier, are told apart by their places, and so they exchange their slots.
    A landmark without an owner (an objective nest) is its own partner: the dragon pit does not become a baron pit.

    ``permutation[w]`` is the partner of row w, an int64 array that is read-only. ValueError if a building has no partner, if the pairing
    is not a permutation that is its own inverse, if a partner does not lie in the lane that the rule gives, or if a rotated building
    lies more than 200 u from its partner. The function reads ``landmarks`` and changes nothing; a test hands it a list of ``dict(entry)``
    copies, ``state.LANDMARKS`` itself is never rebound or edited.

    수식
    ----
    (SYM-6)  owned = { w : team_w ≠ 없음 },   cross(TOP_LANE) = BOT_LANE,   cross(BOT_LANE) = TOP_LANE,   cross(MID_LANE) = MID_LANE
             cand(w) = { w′ ∈ owned : team_{w′} ≠ team_w,  (kind, building, tower)_{w′} = (kind, building, tower)_w }          (w ∈ owned)
             partner(w) = argmin_{w′ ∈ cand(w)} ( (pos_x(w′) − (map_width − pos_x(w)))² + (pos_y(w′) − (map_width − pos_y(w)))², w′ )
                          (w ∈ owned; 사전식 최소: 거리가 같으면 작은 행)
             partner(w) = w                                    (w ∉ owned: 둥지)
             offset = max( {0} ∪ { hypot( pos_x(partner(w)) − (map_width − pos_x(w)),
                                          pos_y(partner(w)) − (map_width − pos_y(w)) ) : w ∈ owned } )
             채택 조건: cand(w)가 비지 않음(w ∈ owned),  partner(partner(w)) = w (모든 w),  lane_{partner(w)} = cross(lane_w) (w ∈ owned),
                        offset ≤ max_offset

      기호          | 뜻                                      | 단위 | 범위                  | 코드 변수
      n             | `landmarks`의 길이(행의 수)             | 개   | ≥ 0                   | `len(landmarks)`
      w, w′         | `landmarks`의 행 번호(0부터)            | —    | 0, …, n − 1           | i, j
      owned         | 소유 팀이 있는 행의 집합(건물)          | —    | {0, …, n − 1}의 부분집합 | `owned`
      team_w        | 행 w의 소유 팀                          | —    | 100, 200, 없음        | `landmarks[w]['team']`
      kind, building, tower | 행의 종류, 건물 종류, 포탑 단계 | —    | 항목 값(같으면 같은 종류와 단계) | `landmarks[w]['kind']`, `['building']`, `['tower']`
      lane_w, cross | 행 w의 레인, 위아래를 맞바꾸는 대응    | —    | TOP_LANE, MID_LANE, BOT_LANE | `['lane']`, `_LANDMARK_LANE`
      pos_x(w), pos_y(w) | 행 w의 위치 좌표                   | u    | 유한                  | `landmarks[w]['x']`, `['y']`
      map_width     | 지도 폭(회전 상수)                      | u    | 고정                  | `W`
      cand(w)       | 행 w의 짝 후보 집합                     | —    | 비어 있으면 거부      | `mates`
      partner(w)    | 행 w의 짝                               | —    | 0, …, n − 1           | perm[w]
      offset        | 회전한 건물과 짝 사이의 가장 큰 거리(소유 행이 없으면 0) | u | ≥ 0           | offset
      max_offset    | offset의 허용 상한                      | u    | 고정                  | `_PARTNER_MAX_OFFSET`

    해석
    ----
    식 (SYM-6)은 소유 팀이 있는 건물마다 회전한 위치에 가장 가까운, 반대 팀의 같은 종류와 단계의 건물을 짝으로 고르므로 짝은 이름이 아니라 위치에서 나오고, 오브젝트
    둥지는 소유 팀이 없어 자기 짝이다. 같은 종류와 단계가 팀마다 둘인 넥서스 포탑은 위치로 가려진다. 짝이 자기 역 치환이고 레인이 `OBJ_SWAP`의 규칙대로 엇갈리는지는 만든 뒤
    확인하며, 이 규칙이 짝을 만들지는 않는다. 짝이 없거나, 짝이 이 조건을 어기거나, 어긋남이 상한을 넘으면 거부한다.

    근거
    ----
    설계서 5.2절(진영 교환, 둥지의 처리), S1–S2 계획 Task 12(표지점 행의 처리), S1–S2 판정 60과 R12-i(적재기의 두 방법).
    """
    def kind_of(j):
        return landmarks[j]['kind'], landmarks[j]['building'], landmarks[j]['tower']

    n = len(landmarks)
    owned = [i for i in range(n) if landmarks[i]['team'] is not None]
    perm, offset = np.arange(n, dtype=np.int64), 0.0
    for i in owned:
        a = landmarks[i]
        rx, ry = W - a['x'], W - a['y']
        mates = [j for j in owned if landmarks[j]['team'] != a['team'] and kind_of(j) == kind_of(i)]
        if not mates:
            raise ValueError(f'landmark {i} has no partner: there is no building of the other team of the same kind and tier')
        best = min(mates, key=lambda j: ((landmarks[j]['x'] - rx) ** 2 + (landmarks[j]['y'] - ry) ** 2, j))
        perm[i] = best
        offset = max(offset, math.hypot(landmarks[best]['x'] - rx, landmarks[best]['y'] - ry))
    if sorted(perm.tolist()) != list(range(n)) or not np.array_equal(perm[perm], np.arange(n)):
        raise ValueError('the swap of the landmark rows is not a permutation that is its own inverse')
    crossed = [i for i in owned if landmarks[perm[i]]['lane'] != _LANDMARK_LANE.get(landmarks[i]['lane'])]
    if crossed:
        raise ValueError(f'the partners of the landmarks {crossed} do not lie in the lane that the rule of OBJ_SWAP gives '
                         '(the rotation turns the top lane into the bottom lane)')
    if offset > _PARTNER_MAX_OFFSET:
        raise ValueError(f'a rotated building lies {offset:,.1f} u from its partner: more than {_PARTNER_MAX_OFFSET:,.0f} u')
    perm.setflags(write=False)
    return perm, offset


LANDMARK_SWAP, LANDMARK_SWAP_OFFSET = derive_landmark_swap(state.LANDMARKS)
# LANDMARK_SWAP: (34,) int64, read-only, its own inverse: row w of the landmarks of the mirrored match is row LANDMARK_SWAP[w] of the
# swapped view (``landmarks[LANDMARK_SWAP]``, ``geo_pl[:, LANDMARK_SWAP]``); the nests stay. LANDMARK_SWAP_OFFSET: the largest distance, in
# map units, between a rotated building and its partner (133.2 on the pinned map file; test_symmetry.py pins it).
if len(LANDMARK_SWAP) != tokens.N_LANDMARKS:
    raise RuntimeError('LANDMARK_SWAP and the views disagree on the number of landmarks')


# ---- the view ----------------------------------------------------------------------------------------------------------------------
def _check_view(v: dict) -> None:
    spec = tokens.VIEW_SPEC
    extra, missing = sorted(set(v) - set(spec)), sorted(set(spec) - set(v))
    if extra:
        raise ValueError(f'swap_view does not know how to swap {extra}: a silent pass-through would leave a team-dependent array unswapped')
    if missing:
        raise ValueError(f'swap_view needs {missing}')
    for key, (shape, _) in spec.items():
        if np.shape(v[key]) != shape:
            raise ValueError(f'{key}: expected shape {shape}, got {np.shape(v[key])}')


def swap_view(v: dict) -> dict:
    """The view ``v`` of ``tokens.build_view`` as seen from the other side: new arrays, ``v`` is not changed, and
    ``swap_view(swap_view(v))`` equals ``v`` (NaN equal to NaN) for the integer positions of the data (see the float32 note below).

    Players: the rows change places, [5-9, 0-4]; x, y, prev_x and prev_y become 14,875 - value (NaN stays NaN). The zone follows the
    rotation through ``terrain.ZONE_SWAP``; a missing position (-1) stays -1 (``ZONE_SWAP`` has ten entries, so -1 is never looked up).
    Tokens: the team codes 0 and 1 change places (2, none, stays), the actor bits of the blue and red players change places, ev_xy
    becomes 14,875 - value; type, age and gold stay. Edges: i and j become (p + 5) % 10, -1 stays; age, kind and the counts stay.
    Summary: the blue and red counts of old events change places and the objective state is permuted by ``OBJ_SWAP``. The distances
    follow their players: ``geo_pp`` rows and columns, ``geo_pl`` rows. Exact for the integer positions of the data (a fractional
    position can round once in float32). ValueError for a view with a missing or an unknown key, or a wrong shape. The swap is checked
    against a mirrored match (players [5-9, 0-4], positions rotated, teams and lanes exchanged in the events): its view equals the
    swapped view in every array except the wall fraction, the geodesics and the landmarks, and its landmarks and the building columns of
    its ``geo_pl`` equal those of the swapped view once ``LANDMARK_SWAP`` is applied (tests, also on the golden matches).

    What this does NOT do, and the S4 loader must (a swapped view is physically consistent only after that step): the 34 landmark rows of
    ``landmarks`` and the 34 landmark columns of ``geo_pl`` stay in place, so row w still names the building w of the original view,
    which now stands at the rotated position (map_width - pos_x, map_width - pos_y) and belongs to the other side of the new viewer. The
    loader takes one of two ways:

    1. It permutes the rows and the columns with ``LANDMARK_SWAP``: ``landmarks[LANDMARK_SWAP]`` and ``geo_pl[:, LANDMARK_SWAP]``. Row w
       then holds the building that stands, in the swapped view, at the place of landmark w (at most ``LANDMARK_SWAP_OFFSET``, 133.2 u,
       away) and belongs to the side of the view that row w belongs to, so the ownership and the position of a row are the row's own
       and need no computing. The six nests stay in their rows: a nest belongs to no team, and the rotated dragon pit lies at the baron
       pit (the broken symmetry that the side marker flags, spec 5.2); the two Atakhan places do rotate onto each other (235 u), but
       their rows carry the same state, so that pairing would not show in ``landmarks``. The result is the view of the mirrored match:
       the landmarks exactly, the 28 building columns of ``geo_pl`` to within three cells (the 75 x 75 terrain grid spans
       [0, 15,000], the swap constant is 14,875). One case is not covered: a nexus-turret kill without a position, which the state
       module assigns by slot, and a slot is not rotation-covariant (real events all have a position).
    2. Or it gives the landmarks no per-row identity (tokens that carry their own ownership and position), computes the ownership
       (which side a building belongs to) from the side marker for the swapped view, and uses the rotated landmark positions,
       (map_width - pos_x, map_width - pos_y), as their positions.

    The learned terrain field (G6c) is attached to physical coordinates, so for a swapped view it must be sampled at the physical
    position, map_width - pos_x and map_width - pos_y of the coordinates that the view carries (players, events), and, for a landmark row
    after way 1, at the position of the building that the row now holds, ``tokens.LANDMARK_XY[LANDMARK_SWAP[w]]``. The wall fraction and
    the geodesic distances move with their players and are not recomputed: the 75 x 75 terrain grid spans [0, 15,000] while the rotation
    constant is 14,875, so they are those of the original positions, the physically same pairs, but not a recomputation on the rotated
    positions.

    Here the physical unrotation is ``W - x``, where W is the map width and x is the carried coordinate.

    수식
    ----
    (SYM-1)  pos_x′ = map_width − pos_x,   pos_y′ = map_width − pos_y        (x, y, prev_x, prev_y, ev_xy의 좌표;  NaN은 NaN)
             이 뺄셈은 float32 배열에서 하므로, 정수 위치에서는 정확하고 분수 위치에서는 한 번 반올림될 수 있다.
    (SYM-7)  players′[p] = players[σ(p)],   σ(p) = (p + 5) mod 10,   그 위에 (SYM-1)을 좌표 열에 적용
             zone′[p] = ZONE_SWAP[zone[σ(p)]]  (zone ≥ 0),   zone′[p] = −1  (zone = −1)
             ev_team′ = ev_team ⊕ 1   (ev_team < 2),   ev_team′ = ev_team   (2: 없음);   ev_actors′ = (SYM-3);   ed_i′, ed_j′ = (SYM-2)
             summary′ = summary[SUMMARY_SWAP]:  old 사건 칸은 파랑·빨강 쌍을 교환, 객체 상태 열은 (SYM-8)
             geo_pp′[p, p₂] = geo_pp[σ(p), σ(p₂)],   geo_pl′[p, ·] = geo_pl[σ(p), ·]
             그 밖의 배열(ev_type, ev_dt, ev_gold, ev_n, ev_dropped, ed_dt, ed_kind, ed_n, ed_dropped, landmarks)은 복사만 한다.
    (SYM-8)  OBJ_SWAP[c] = index(column_partner(name_c)),   c = 0, …, N_obj − 1
             column_partner(name) = name                                   (이름이 blue_ 또는 red_로 시작하지 않을 때)
             column_partner(blue_X) = red_{X′},  column_partner(red_X) = blue_{X′}
                 (X′ = X: 레인 열이 아닐 때;  레인 열이면 X′ = X의 레인을 top ↔ bot, mid는 그대로)
             레인 열: (inhib|plates)_(top|mid|bot)(_respawn_s)? 꼴의 이름. 짝이 없거나 OBJ_SWAP∘OBJ_SWAP이 항등이 아니면 import에서 ValueError.

      기호                | 뜻                                    | 단위 | 범위               | 코드 변수
      pos_x, pos_y        | 선수·사건의 위치 좌표                 | u    | 유한 또는 NaN      | players_num의 x, y 열, ev_xy
      map_width           | 지도 폭(회전 상수)                    | u    | 고정               | W
      p, p₂               | 선수 행 번호(참가자 번호 − 1)         | —    | 0, …, 9            | PLAYER_ORDER의 위치
      σ(p)                | 교환한 행                             | —    | 0, …, 9            | PLAYER_ORDER[p]
      zone                | 구역 번호                             | —    | −1, 0, …, 9        | players_zone
      ev_team             | 사건의 팀 코드(0 파랑, 1 빨강, 2 없음) | —   | 0, 1, 2            | ev_team
      c                   | 객체 상태의 열 번호(0부터)            | —    | 0, …, N_obj − 1    | OBJ_SWAP의 인덱스 i
      N_obj               | 객체 상태 열의 수                     | 개   | ≥ 1                | `len(state.OBJ_STATE_NAMES)`
      name_c, X, X′       | 열 c의 이름, blue_·red_ 뒤의 이름, 레인을 맞바꾼 이름 | — | 문자열        | `state.OBJ_STATE_NAMES`, `_partner`
      OBJ_SWAP[c]         | 열 c의 짝 열 번호                     | —    | 0, …, N_obj − 1    | `OBJ_SWAP`
      column_partner(name) | 열 이름의 짝 열 이름(행 w의 partner(w)와 다르다) | — | 문자열        | `_partner(name)`
      index(name)         | 열 이름의 `OBJ_STATE_NAMES` 안 위치(0부터) | — | 0, …, N_obj − 1    | `index[name]`

    해석
    ----
    식 (SYM-7)은 보기를 상대 진영에서 본 보기로 바꾸되 팀에 따라 달라지는 항목만 치환하고, 팀과 무관한 항목은 그대로 둔다. 선수 행은 σ로 맞바꾸고 좌표는 (SYM-1)로 돌려,
    같은 사건이 다른 진영의 좌표계에서 보이게 하며, 객체 상태의 열은 (SYM-8)의 이름 규칙으로 맞바꾼다(둥지, 타이머, 지도 전체 플래그는 팀이 없어 제자리이고 레인은
    위아래가 엇갈린다). 두 번 적용하면 입력과 같지만 이는 정수 위치 영역에서의 산술이며(NaN은 NaN으로 본다), 물리적으로 정확한 대칭은 아니다 [설계서 5.2절]. 표지점 34행과
    `geo_pl`의 34열, 벽 비율과 측지 거리는 다시 계산하지 않으므로, 적재기가 `LANDMARK_SWAP`으로 행과 열을 치환하거나 행 정체성 없이 소유와 물리 위치를 따로 실어
    나르기 전에는 이 보기가 새 시점과 맞지 않는다 [S1–S2 판정 60, R12-i]. 둥지와 지도 전체 플래그는 회전해도 제자리이므로 비대칭이 남고 진영 표시가 맡는다.

    근거
    ----
    설계서 5.2절(대칭, 진영 표시, 출력의 교환), S1–S2 계획 Task 12(항목별 규칙), S1–S2 판정 60과 R12-i(적재기의 두 방법).
    """
    _check_view(v)
    # (SYM-7)의 선수 행 σ와 (SYM-1)의 좌표 열: 팬시 인덱싱이 복사본을 만들므로 입력 v는 바뀌지 않는다.
    players = np.asarray(v['players_num'])[PLAYER_ORDER]                       # fancy indexing: a copy
    players[:, _XY_COLUMNS] = W - players[:, _XY_COLUMNS]
    # (SYM-7)의 구역: −1(위치 없음)은 ZONE_SWAP을 찾지 않고(항목이 열 개뿐이다) 그대로 두며, 행은 선수와 함께 σ로 옮긴다.
    z = np.asarray(v['players_zone'])
    zone = np.where(z >= 0, terrain.ZONE_SWAP[np.where(z >= 0, z, 0)], z)[PLAYER_ORDER]
    team = np.asarray(v['ev_team'])
    geo_pp = np.asarray(v['geo_pp'])
    return {
        'players_num': players,
        'players_zone': zone,
        'landmarks': np.array(v['landmarks']),
        'ev_type': np.array(v['ev_type']),
        'ev_dt': np.array(v['ev_dt']),
        'ev_team': np.where(team < tokens.TEAM_NONE, team ^ 1, team),
        'ev_xy': W - np.asarray(v['ev_xy']),
        'ev_actors': swap_actors(v['ev_actors']),
        'ev_gold': np.array(v['ev_gold']),
        'ev_n': np.array(v['ev_n']),
        'ev_dropped': np.array(v['ev_dropped']),
        'ed_i': swap_participants(v['ed_i']),
        'ed_j': swap_participants(v['ed_j']),
        'ed_dt': np.array(v['ed_dt']),
        'ed_kind': np.array(v['ed_kind']),
        'ed_n': np.array(v['ed_n']),
        'ed_dropped': np.array(v['ed_dropped']),
        'summary': np.asarray(v['summary'])[SUMMARY_SWAP],
        'geo_pp': geo_pp[PLAYER_ORDER][:, PLAYER_ORDER],
        'geo_pl': np.asarray(v['geo_pl'])[PLAYER_ORDER],
    }
