"""Engagement units of the C model: the kill table of a match, the link rule, the unit clocks, size class and first-kill mark.

A unit is a connected component of the CHAMPION_KILL events of one match under a single rule: two kills are linked iff their times
differ by at most ``G_MS`` and their positions by at most ``L_U`` map units (dx*dx + dy*dy <= L_U*L_U, boundary included); links chain
(single linkage). A kill without a position links to nothing and is a unit of its own. Per unit (``UnitTable``):

    f, last   time of the first / last kill (the first kill is the unit's smallest row; rows are in time order, so a tie in the
              same ms keeps the record order)
    tau       f - TAU_LEAD_MS, the start of the engagement: a rule anchor (kill-credit window), not a measured start
    delta     max(last + 1, f + DELTA_AFTER_F_MS), the end (the thesis label window: f + 20 s = tau + 35 s)
    in_scope  SCOPE_FIRST_MS <= f <= T - END_CUT_MS, both ends included
    nb, nr    people credited on the blue / red team: killers, victims and assisters with ids 1-10, each person once per unit
    size      class of (nb, nr): 0 T, 1 S, 2 ASYM, 3 P
    mark      one of ``N_MARKS``: the (killer slot, victim slot) pair of the first kill, or an execution mark (killer is no champion)
    spread    largest distance between two kills of the unit that have a position (0 for fewer than two)

``kill_people_mask`` (the people a kill credits) and ``event_xy`` (the position rule of an event) are exported so that code built on
units, such as the attribution of objectives, reuses these two rules instead of deriving them again. The same holds for the id rule
(``_event_id`` and ``_assist_mask``, which take a label to name the event in their error messages) and the participant helpers
(``_is_int``, ``_is_pid``, ``_ALL_PIDS``, ``_POPCOUNT``): ``targets.objective_table`` imports them, so that kills and objectives parse
ids alike.

방법에서의 역할
---------------
이 모듈은 한 경기의 챔피언 킬 사건을 교전 단위(이하 단위)로 묶는 규칙을 구현한다. 단위는 그 경기 모든 킬 사이의 연결 성분이고, 두 킬은
시간 간격이 G 이하이며 거리가 L 이하일 때 이어진다(단일 연결). 단위마다 첫 킬 시각 f와 마지막 킬 시각 last에서 시작 τ와 끝 δ를 정하고,
킬 크레딧을 받은 팀별 인원 n_blue, n_red에서 규모 층 s를 정하며, 첫 킬의 (킬러 슬롯, 피해자 슬롯) 쌍 또는 처형에서 표지 m을 정한다.
규모와 표지는 결과 쪽에 속하는 단위의 속성이고 발생 관문이나 t₀ 입력이 아니다 [설계서 2절]. 이 모듈은 로더, 모형, 목표 표를 알지 못한다.

식 목록
-------
    (UNIT-1)  연결 규칙과 연결 성분                `link_components`
    (UNIT-2)  킬 하나가 크레딧하는 인원의 비트 마스크  `kill_people_mask`
    (UNIT-3)  규모 층 s                             `size_class`
    (UNIT-4)  첫 킬의 표지 m                         `first_kill_mark`
    (UNIT-5)  단위의 시계·범위·인원·퍼짐              `build_units` (`_spread`)

가정
----
- 시각은 모두 ms이다(설계서 본문의 s와 같은 양을 ms로 쓴다). 지도 거리는 지도 단위 u이다. 시계 상수 G, L, 시작 간격, 끝 간격, 범위의 두
  끝은 `config`에서 오며 이 모듈이 정하지 않는다(L은 규칙값이고 τ는 규칙 앵커이지 측정한 시작이 아니다 [설계서 2절]).
- `gap_ms`가 정수이고 `link_u`가 0 이상이라는 것은 호출하는 쪽의 가정이며 `link_components`가 검사하지 않는다.
- 킬은 시간순으로 들어와야 한다. `link_components`가 줄어드는 시각을 ValueError로 거부하므로 검사되는 조건이다. 같은 ms의 킬은 기록 순서를
  지킨다.
- 팀은 두 곳에서 정해진다. 표지의 팀(식 (UNIT-4)의 d)은 참가자 번호 1–5와 6–10으로, 인원의 팀(식 (UNIT-5)의 n_blue, n_red)은 `team`
  배열로 정한다. 두 규약이 같다는 것은 입력의 가정이고 검사하지 않으며, `team`이 100도 200도 아닌 참가자는 어느 팀에도 세지 않는다.
- 위치가 없는 킬은 거리 규칙을 적용할 수 없어 혼자 단위가 된다 [설계서 11절 항목 4]. 경기 끝 T는 인자로 받으며 검사하지 않는다.
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 2 (unit: link rule, tau, delta, scope, size classes, people) and
#      section 11, items 2, 4 and 5 (execution marks, kills without a position, game end T).
# Ref: thesis ch3_data.tex:95-190 (link rule: single-linkage components over all kills of a match; L at 139-145) and
#      THESIS_DESIGN_SPEC_KO.md:174 (size classes); LEGACY gameplay/cohorts_exact.py::kill_credit and cohort_label (credit = killer,
#      victim and assisters 1-10; T / S / ASYM / P).
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\link_units.py::link_components (the prototype: same sweep, union-find and order).
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept.reader import MatchFormatError, MatchRecord, MatchRejected

BLUE, RED = 100, 200                      # team ids of the Match-V5 files (participants 1-5 and 6-10)
SIZE_T, SIZE_S, SIZE_ASYM, SIZE_P = range(4)
SIZE_NAMES = ('T', 'S', 'ASYM', 'P')      # the name of each size code
N_SLOTS = 5                               # participants per team: slot = (pid - 1) % N_SLOTS
EXEC_MARK0 = 2 * N_SLOTS * N_SLOTS        # marks 0-49: (killer team, killer slot, victim slot); 50 / 51: execution, victim on blue / red
if EXEC_MARK0 + 2 != C.N_MARKS:           # an explicit raise, not an assert: asserts vanish under python -O
    raise RuntimeError(f'config.N_MARKS is {C.N_MARKS}, but the mark layout of units.py has {EXEC_MARK0 + 2} marks')
_NAN = float('nan')
_ALL_PIDS = (1 << 10) - 1                 # bit mask of the participants 1-10 (bit pid - 1)
_POPCOUNT = np.array([bin(m).count('1') for m in range(1 << 10)], dtype=np.int8)


@dataclass(frozen=True, eq=False)
class KillTable:
    """The CHAMPION_KILL events of one match, one row each, in record (time) order.

    ``killer`` is the raw killerId (``kill_table`` accepts 0-10; 0 is an execution: a minion, tower or monster) and ``victim`` the raw
    victimId (1-10); ``build_units`` treats a killer outside 1-10 as no champion. ``amask`` has bit (pid - 1) set for every assisting
    participant 1-10; a kill without a position has x = y = NaN and ``pos_missing``.
    """
    t: np.ndarray             # int64, ms
    x: np.ndarray             # float64
    y: np.ndarray             # float64
    killer: np.ndarray        # int8
    victim: np.ndarray        # int8
    amask: np.ndarray         # int16
    bounty: np.ndarray        # int32
    shutdown: np.ndarray      # int32
    pos_missing: np.ndarray   # bool


@dataclass(frozen=True, eq=False)
class UnitTable:
    """The units of one match, ordered by (first kill time, first kill row); see the module text for the columns.

    ``comp[unit_idx]`` holds the KillTable rows of unit unit_idx, ascending.

    열의 식은 (UNIT-5)이고 `build_units`가 채운다.
    """
    comp: list[np.ndarray]    # per unit the KillTable rows (int array)
    f: np.ndarray             # int64
    last: np.ndarray          # int64
    tau: np.ndarray           # int64
    delta: np.ndarray         # int64
    nb: np.ndarray            # int8
    nr: np.ndarray            # int8
    size: np.ndarray          # int8
    mark: np.ndarray          # int16
    in_scope: np.ndarray      # bool
    spread: np.ndarray        # float32
    n_kills: np.ndarray       # int16


# ---- kill table ------------------------------------------------------------------------------------------------------------
def event_xy(e: dict) -> tuple[float, float]:
    """(x, y) of the ``position`` of an event; (NaN, NaN) when the event has no position, it is not a dict, or a coordinate is missing,
    not a number or not finite. One rule for every table built from events, so that kills and objectives read positions alike.

    위치 규칙 하나를 모든 사건 표가 공유하며, 읽을 수 없는 위치는 NaN으로 두어 (UNIT-1)의 거리 비교가 거짓이 되게 한다.
    """
    pos = e.get('position')
    if isinstance(pos, dict):
        try:
            x, y = float(pos['x']), float(pos['y'])
        except (KeyError, TypeError, ValueError):
            return _NAN, _NAN
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return _NAN, _NAN


def _is_int(v) -> bool:
    """A JSON integer (not a float, a string or a bool), as in the reader."""
    return isinstance(v, int) and not isinstance(v, bool)


_KILL_LABEL = 'a CHAMPION_KILL'     # how the id errors of kill_table name the event; c_ept.targets passes its own for the objective events


def _event_id(rec: MatchRecord, e: dict, name: str, lowest: int, default: int | None, label: str = _KILL_LABEL) -> int:
    """The integer ``e[name]`` in ``lowest``..10. A key that is absent gives ``default`` (None: the key is required). Anything else
    raises MatchFormatError: null, a bool, a float or a string, an integer outside the range. The message begins with ``label`` (the
    event), then names the field, the type or the number, and the patch; it never carries the match key or a value other than a number."""
    if name not in e:
        if default is None:
            raise MatchFormatError(rec.key, rec.patch, f'{label} has no {name}')
        return default
    v = e[name]
    if not _is_int(v):
        raise MatchFormatError(rec.key, rec.patch, f'{label} has a {name} of type {type(v).__name__}, not an integer')
    if not lowest <= v <= 10:
        raise MatchFormatError(rec.key, rec.patch, f'{label} has {name} {v}, not in {lowest}-10')
    return v


def _assist_mask(rec: MatchRecord, e: dict, label: str = _KILL_LABEL) -> int:
    """Bit mask (bit pid - 1) of the participants 1-10 in ``assistingParticipantIds``. Absent or null means no assists; integers
    outside 1-10 are ignored. MatchFormatError (the message begins with ``label``) when the value is not a list or an entry is not an
    integer."""
    ids = e.get('assistingParticipantIds')
    if ids is None:
        return 0
    if not isinstance(ids, (list, tuple)):
        raise MatchFormatError(rec.key, rec.patch, f'{label} has assistingParticipantIds of type {type(ids).__name__}, not a list')
    mask = 0
    for a in ids:
        if not _is_int(a):
            raise MatchFormatError(rec.key, rec.patch,
                                   f'{label} has an assistingParticipantIds entry of type {type(a).__name__}, not an integer')
        if 1 <= a <= 10:
            mask |= 1 << (a - 1)
    return mask


def kill_table(rec: MatchRecord) -> KillTable:
    """The CHAMPION_KILL events of ``rec`` as a KillTable, in record order (the events are ordered by time).

    killerId is an integer 0-10 (0: an execution) and counts as 0 when the key is absent; victimId is an integer 1-10 and required.
    ``assistingParticipantIds`` absent or null means no assists; its entries are integers, those outside 1-10 are ignored. A missing
    (or null) bounty or shutdownBounty counts as 0, an unusable position gives NaN (``event_xy``). Every other value of an id field
    raises MatchFormatError, never a bare OverflowError, TypeError or KeyError: the reason and the patch are in the message, the key
    is not.

    기록의 킬 사건을 열 배열로 옮기는 단계이며, 킬러·피해자·도움 번호는 사건 표 공통의 읽기 규칙(`_event_id`, `_assist_mask`)으로 읽는다.
    이후 함수도 필요한 번호 범위를 스스로 확인한다. 행 번호는 0부터이고 기록(시간) 순서를 그대로 둔다.
    """
    kills = [e for e in rec.events if e.get('type') == 'CHAMPION_KILL']
    n = len(kills)
    t, x, y = np.empty(n, np.int64), np.full(n, _NAN), np.full(n, _NAN)
    killer, victim = np.empty(n, np.int8), np.empty(n, np.int8)
    amask, bounty, shutdown = np.zeros(n, np.int16), np.zeros(n, np.int32), np.zeros(n, np.int32)
    for r, e in enumerate(kills):
        killer[r] = _event_id(rec, e, 'killerId', 0, 0)
        victim[r] = _event_id(rec, e, 'victimId', 1, None)
        amask[r] = _assist_mask(rec, e)
        t[r] = e['timestamp']
        x[r], y[r] = event_xy(e)
        bounty[r], shutdown[r] = int(e.get('bounty') or 0), int(e.get('shutdownBounty') or 0)
    return KillTable(t, x, y, killer, victim, amask, bounty, shutdown, np.isnan(x))


# ---- link rule ---------------------------------------------------------------------------------------------------------------
def link_components(t, x, y, gap_ms: int = C.G_MS, link_u: float = C.L_U) -> list[np.ndarray]:
    """Connected components of the kills under the link rule: k ~ k' iff |t_k - t_k'| <= gap_ms and dx*dx + dy*dy <= link_u*link_u.

    ``t`` must not decrease (kills in time order). Each component is an ascending int array of indices; the components are ordered
    by (t of the first index, first index), which for sorted times is the order of the first indices. A NaN coordinate makes every
    comparison false, so such a kill stays alone. Sweep and union-find as in the prototype ``link_units.link_components``.

    수식
    ----
    (UNIT-1)  k ~ k′  ⇔  |t_k − t_k′| ≤ G  ∧  (pos_x(k) − pos_x(k′))² + (pos_y(k) − pos_y(k′))² ≤ L²      (k ≠ k′)
              성분 Q = ~의 반사·추이적 닫힘이 만드는 동치류. 각 Q는 행 번호 오름차순, Q들은 (t_{min Q}, min Q) 순이다.
              (UNIT-1) = 설계서 2절 연결 규칙, 학위논문 `ch3_data.tex:95–190`의 연결 성분 정의.

      기호  | 뜻                              | 단위      | 범위                     | 코드 변수
      k, k′ | 킬 표의 행 번호(0부터)          | —         | 0, …, n − 1              | 순환 변수 i, j
      t_k   | 킬 k의 시각                     | ms        | 정수, 비감소             | t
      pos_x(k), pos_y(k) | 킬 k의 위치 좌표    | u         | 유한, 또는 NaN(위치 없음) | x, y
      n     | 킬 표의 행 수                   | 개        | ≥ 0                      | t.size
      G     | 연결 시간 간격                  | ms        | 정수(호출자의 가정)      | gap_ms (기본 `config.G_MS`)
      L     | 연결 거리                       | u         | ≥ 0(호출자의 가정)       | link_u (기본 `config.L_U`)
      Q     | 한 단위의 킬 행 집합            | —         | 비지 않음                | 반환 목록의 원소

    해석
    ----
    식 (UNIT-1)은 시간과 거리가 모두 가까운 두 킬을 같은 교전에 놓는 대칭 규칙이고, 단위는 이 관계의 반사·추이적 닫힘, 곧 사슬로 이어진 킬 묶음(단일 연결 성분)이며 홀로 있는 킬도 성분이다.
    사슬을 허용하므로 한 단위의 첫 킬과 마지막 킬은 G나 L보다 멀어질 수 있다. 두 조건 모두 등호를 포함하므로 시간 차가 정확히 G이거나
    거리가 정확히 L인 쌍도 이어진다 [설계서 2절]. 위치가 NaN인 킬은 거리 비교가 거짓이 되어 어느 킬과도 이어지지 않고 혼자 단위가 된다 [설계서 11절 항목 4].
    경기에 킬이 없으면 단위도 없다.

    근거
    ----
    설계서 2절(연결 규칙, 단일 연결 성분), 11절 항목 4(위치 없는 킬), S1–S2 계획 Task 3(시제품과 같은 성분과 순서).
    """
    t = np.asarray(t)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.ndim != 1 or x.shape != t.shape or y.shape != t.shape:
        raise ValueError('t, x and y must be 1-d arrays of the same length')
    if t.size > 1 and np.any(t[1:] < t[:-1]):
        raise ValueError('kill times must not decrease')
    n = t.size
    tl, xl, yl = t.tolist(), x.tolist(), y.tolist()
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]            # path compression (halving)
            a = parent[a]
        return a

    # (UNIT-1): 시각이 정렬되어 있으므로 i에서 시간 차가 G를 넘는 첫 j 뒤의 j는 모두 넘는다. 그래서 그 j에서 안쪽 반복을 멈추고,
    # 거리 조건은 거리 제곱과 L²의 비교이고 등호를 포함한다.
    reach2 = float(link_u) * float(link_u)
    for i in range(n):
        j = i + 1
        while j < n and tl[j] - tl[i] <= gap_ms:     # sorted times: no later j can be within the gap once this one is not
            dx, dy = xl[i] - xl[j], yl[i] - yl[j]
            if dx * dx + dy * dy <= reach2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            j += 1
    # (UNIT-1)의 성분 Q: 같은 뿌리를 가진 행을 모은다. 행을 오름차순으로 훑으므로 각 Q도 오름차순이고 Q의 순서는 첫 행 순서이다.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)       # a group is created at its smallest index: groups come in first-index order
    return [np.array(g, dtype=np.intp) for g in groups.values()]


# ---- size class and first-kill mark ----------------------------------------------------------------------------------------
def size_class(nb: int, nr: int) -> int:
    """Size code from the credited people per team: 0 T (both teams >= 4), 1 S (both 2-3), 2 ASYM (one side 2-3, the other >= 4),
    3 P (fewer than 2 on a side). The thesis cohorts.

    수식
    ----
    (UNIT-3)  s = P                    (n_min < 2)
              s = T                    (n_min ≥ 4)
              s = S                    (2 ≤ n_min,  n_max ≤ 3)
              s = ASYM                 (2 ≤ n_min ≤ 3,  n_max ≥ 4)             n_min = min(n_blue, n_red),  n_max = max(n_blue, n_red)
              코드: T = 0, S = 1, ASYM = 2, P = 3.  (UNIT-3) = 설계서 2절 규모 표, 학위논문 `THESIS_DESIGN_SPEC_KO.md:174`.

      기호          | 뜻                              | 단위 | 범위          | 코드 변수
      s             | 규모 층                         | —    | T, S, ASYM, P | 반환값(0–3)
      n_blue, n_red | 킬 크레딧을 받은 파랑·빨강 인원 | 명   | 정수 ≥ 0      | nb, nr
      n_min, n_max  | 두 인원 가운데 작은 값, 큰 값   | 명   | 정수 ≥ 0      | lo, hi

    해석
    ----
    식 (UNIT-3)은 단위의 크기를 두 팀의 크레딧 인원으로 네 층에 나눈다. 판정 순서가 P, T, S, ASYM이므로 한 팀이 2명 미만이면 다른 팀이
    몇 명이든 P이고, 두 팀이 모두 4명 이상일 때만 T이다. 경계에서는 n_min = 2가 P가 아니고 n_min = 4가 T이다. 규모는 결과 쪽 표지이며 발생 관문이나 t₀
    입력이 아니다 [설계서 2절].

    근거
    ----
    설계서 2절(규모 표, 규모는 결과 쪽 표지), S1–S2 계획 Task 3(규모 층 코드의 순서).
    """
    lo, hi = min(nb, nr), max(nb, nr)
    if lo < 2:
        return SIZE_P
    if lo >= 4:
        return SIZE_T
    return SIZE_S if hi <= 3 else SIZE_ASYM


def first_kill_mark(killer: int, victim: int) -> int:
    """Mark (0 .. N_MARKS - 1) of a unit's first kill. A champion killer (1-10): 25 d + 5 i + j with d = 0 for the blue killer (1-5)
    and 1 for the red killer (6-10), i the killer's slot and j the victim's slot ((pid - 1) % 5). An execution (killer outside 1-10):
    50 when the victim is on blue (1-5), 51 when on red. The victim must be a participant 1-10.

    수식
    ----
    (UNIT-4)  m = 25·d + 5·i + j      (killer ∈ 1..10),   d = [killer ≥ 6],  i = (killer − 1) mod 5,  j = (victim − 1) mod 5
              m = 50 + [victim ≥ 6]   (killer ∉ 1..10: 처형)                   [·]은 참이면 1, 거짓이면 0
              (UNIT-4) = 설계서 3.1절의 선제 킬 쌍 i→j에 처형 표지 둘을 더한 표지 번호.

      기호           | 뜻                          | 단위 | 범위                      | 코드 변수
      m              | 첫 킬의 표지                | —    | 0, …, 51                  | 반환값
      d              | 킬러의 진영(0 파랑, 1 빨강) | —    | 0, 1                      | d
      i, j           | 킬러·피해자의 슬롯(0부터)   | —    | 0, …, 4                   | i, j
      killer, victim | 킬러·피해자의 참가자 번호   | —    | 피해자 1–10; 킬러는 그 밖도 가능 | killer, victim

    해석
    ----
    식 (UNIT-4)는 챔피언이 낸 첫 킬을 (킬러 진영, 킬러 슬롯, 피해자 슬롯)의 쌍으로 하나의 번호에 담고, 챔피언이 아닌 킬러의 첫 킬은
    피해자 진영에 따라 두 처형 표지에 담는다. 쌍은 슬롯 번호이지 역할이 아니다 [설계서 5.2절]. 진영은 team 배열이 아니라 참가자 번호
    1–5와 6–10으로 정한다. 피해자가 참가자 1–10이 아니면 어느 표지에도 담을 수 없어 거부한다.

    근거
    ----
    설계서 3.1절(표지 m), 11절 항목 2(처형 표지 둘), S1–S2 계획 Task 3(표지 식).
    """
    killer, victim = int(killer), int(victim)
    if not 1 <= victim <= 10:
        raise ValueError(f'victim must be a participant 1-10, got {victim}')
    if 1 <= killer <= 10:
        d = 0 if killer <= N_SLOTS else 1
        i, j = (killer - 1) % N_SLOTS, (victim - 1) % N_SLOTS
        return (d * N_SLOTS + i) * N_SLOTS + j
    return EXEC_MARK0 + (0 if victim <= N_SLOTS else 1)


# ---- units -------------------------------------------------------------------------------------------------------------------
def _spread(x: np.ndarray, y: np.ndarray) -> float:
    """Largest distance between two kills of a unit; 0 for a unit of one kill.

    A kill without a position links to nothing, so it is always a unit of its own: the kills of a unit of two or more all have
    positions, and a unit of one kill has fewer than two positions (spread 0).

    (UNIT-5)의 spread 항을 계산한다.
    """
    # (UNIT-5)의 spread 항: 단위 안 모든 킬 쌍의 거리 제곱의 최댓값에 제곱근을 한 번 씌운다.
    if x.size < 2:
        return 0.0
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    return float(np.sqrt((dx * dx + dy * dy).max()))


def _is_pid(a: np.ndarray) -> np.ndarray:
    """Which entries are participant ids 1-10."""
    return (a >= 1) & (a <= 10)


def _pid_bits(pid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bit (pid - 1) for the entries marked valid, 0 for the others."""
    return np.where(valid, np.left_shift(1, np.clip(pid - 1, 0, 9)), 0)


def kill_people_mask(k: KillTable) -> np.ndarray:
    """(n,) int16: the people each kill credits, as a bit mask (bit pid - 1): the killer if 1-10, the victim if 1-10 and the assisters
    (``amask``, bits 0-9 only). The people of a unit are the OR of the masks of its kills (``build_units`` counts them per team);
    code that needs a unit's people, such as the attribution of objectives, should take them from here and not derive them again.

    수식
    ----
    (UNIT-2)  people_k = (amask_k ∧ mask_all) ∨ bit(killer_k) ∨ bit(victim_k),    bit(pid) = 2^(pid − 1) (1 ≤ pid ≤ 10),  0 (그 밖),
              mask_all = 2^10 − 1

      기호               | 뜻                                   | 단위 | 범위        | 코드 변수
      people_k           | 킬 k가 크레딧하는 인원의 비트 마스크 | —    | 0, …, 1,023 | 반환값의 k번째 원소(int16)
      amask_k            | 킬 k의 도움 마스크                   | —    | 정수        | `amask`
      mask_all           | 참가자 1–10의 비트 마스크            | —    | 1,023       | `_ALL_PIDS`
      killer_k, victim_k | 킬 k의 킬러·피해자 번호              | —    | 정수        | `killer`, `victim`
      pid                | 참가자 번호                          | —    | 정수        | 함수 `_pid_bits`의 pid

    해석
    ----
    식 (UNIT-2)는 킬러, 피해자, 도움 참가자를 한 정수의 비트로 합쳐, 단위의 인원이 비트 합집합으로 구해지게 한다. 같은 사람이 여러 킬이나
    여러 역할로 나와도 비트는 하나이므로 한 단위에서 한 번만 센다. 챔피언이 아닌 킬러(번호가 1–10 밖)는 비트를 만들지 않고, 도움 마스크는
    아래 10비트만 쓴다 [설계서 2절 인원, C-10].

    근거
    ----
    설계서 2절(인원 규칙), S1–S2 계획 Task 3(인원은 참가자 번호 범위만).
    """
    killer, victim = k.killer.astype(np.int64), k.victim.astype(np.int64)
    people = (k.amask.astype(np.int64) & _ALL_PIDS) | _pid_bits(killer, _is_pid(killer)) | _pid_bits(victim, _is_pid(victim))
    return people.astype(np.int16)


def build_units(k: KillTable, team: np.ndarray, T_ms: int) -> UnitTable:
    """The units of a match from its kill table, the (10,) team ids (indexed by pid - 1; blue 100, red 200) and the game end ``T_ms``.

    MatchRejected('same_team_kill') when a kill has a killer and a victim (both 1-10) on the same team; an execution is not checked.
    ``kill_table`` guarantees victims 1-10; a table built by hand with another victim raises ValueError when that kill is a unit's first.

    수식
    ----
    (UNIT-5)  f = t_{min Q},   last = t_{max Q},   τ = f − lead,   δ = max(last + 1, f + after)                  (ms)
              범위 안 ⇔ scope_first ≤ f ≤ T − cut                                                   (양 끝 포함)
              people = ⋁_{k∈Q} people_k   (people_k는 (UNIT-2)),
              n_blue = popcount(people ∧ mask_blue),   n_red = popcount(people ∧ mask_red)
              mask_blue = Σ_{pid: team_pid = 100} 2^(pid − 1),   mask_red = Σ_{pid: team_pid = 200} 2^(pid − 1)
              s = (UNIT-3)(n_blue, n_red),   m = (UNIT-4)(killer_{k₀}, victim_{k₀}),   k₀ = min Q
              spread = max_{k,k′∈Q} √((pos_x(k) − pos_x(k′))² + (pos_y(k) − pos_y(k′))²)  (|Q| ≥ 2),   0 (|Q| = 1),   n_kills = |Q|
              spread의 최댓값은 float64로 계산하고 열에는 float32로 저장한다(반올림).   Q = comp[unit_idx]
              Q는 (UNIT-1)의 성분이고, 행은 시간순이므로 min Q와 max Q는 단위의 첫 킬과 마지막 킬이다.

      기호             | 뜻                                                  | 단위 | 범위                          | 코드 변수
      f, last          | 첫·마지막 킬 시각                                   | ms   | 정수                          | f, last
      τ, δ             | 단위의 시작, 끝(설계서의 τ, δ와 같은 양, 단위만 ms) | ms   | 정수                          | tau, delta
      lead, after      | 시작 간격, 끝 간격                                  | ms   | 정수                  | `TAU_LEAD_MS`, `DELTA_AFTER_F_MS`
      scope_first, cut | 범위의 처음 시각, 끝에서 자르는 길이              | ms   | 정수                          | `SCOPE_FIRST_MS`, `END_CUT_MS`
      T                | 경기 끝 시각                                        | ms   | 정수                          | T_ms
      people           | 단위 인원의 비트 마스크                             | —    | 0, …, 1,023                   | people
      mask_blue, mask_red | 파랑·빨강 참가자의 비트 마스크                   | —    | 0, …, 1,023                   | blue, red
      pid              | 참가자 번호                                         | —    | 1, …, 10                      | 비트 위치 pid − 1
      n_blue, n_red    | 단위에서 크레딧을 받은 팀별 인원                    | 명   | ≥ 0 (팀마다 5명이면 ≤ 5)      | nb, nr
      k₀               | 단위의 첫 킬의 행                                   | —    | 0, …, n − 1                   | first
      spread           | 위치가 있는 킬 사이의 최대 거리                     | u    | ≥ 0 (float64 계산, float32 저장) | spread
      unit_idx         | 단위 번호(0부터)                                    | —    | 0, …, 단위 수 − 1             | comp의 인덱스 u
      n_kills          | 단위의 킬 수                                        | 개   | ≥ 1                           | n_kills

    해석
    ----
    식 (UNIT-5)는 연결 성분 Q마다 첫·마지막 킬의 시각에서 단위의 시작, 끝, 범위를 정하고 인원과 규모, 표지, 퍼짐을 붙이는 계산이다. τ는 첫 킬
    f보다 정해진 간격만큼 앞선 규칙 앵커이지 측정한 시작이 아니고, δ는 학위논문 라벨 창의 끝 f + after가 마지막 킬보다 일찍 닫힐 때 마지막
    킬 뒤 1 ms로 늘린 값이다 [설계서 2절]. 범위는 첫 킬 f로만 정하고 단위를 지우지 않으며, 범위 밖 단위도 표에 남아 귀속에 쓰인다
    [설계서 11절 항목 6]. 인원은 크레딧 마스크의 합집합을 팀 마스크로 갈라 세므로 한 사람은 한 번만 세고, 팀은 team[pid − 1]로 정한다.
    위치가 NaN인 킬은 혼자 단위가 되므로 킬이 둘 이상인 단위의 모든 킬은 위치가 있고, 퍼짐은 그 사이의 최대 거리이다.

    근거
    ----
    설계서 2절(τ, δ, 범위, 인원), 11절 항목 4(위치 없는 킬), 5(경기 끝 T), 6(범위 밖 단위도 귀속), S1–S2 계획 Task 3(팀 출처, 첫 킬의 행,
    같은 팀 킬의 거부).
    """
    team = np.asarray(team)
    if team.shape != (10,):
        raise ValueError(f'team must hold 10 team ids, got shape {team.shape}')
    # 팀킬 거부: 킬러와 피해자가 모두 참가자 1–10인 킬만 팀을 비교한다(처형은 비교하지 않는다).
    killer, victim = k.killer.astype(np.int64), k.victim.astype(np.int64)
    both = _is_pid(killer) & _is_pid(victim)                # a killer outside 1-10 is an execution
    if np.any(team[killer[both] - 1] == team[victim[both] - 1]):
        raise MatchRejected('same_team_kill')
    # (UNIT-2)의 people_k와 (UNIT-5)의 팀 마스크 mask_blue, mask_red: 팀 구성은 team 배열에서 읽는다.
    credit = kill_people_mask(k)
    blue = sum(1 << i for i in range(10) if team[i] == BLUE)
    red = sum(1 << i for i in range(10) if team[i] == RED)

    # (UNIT-5)의 f, last: 성분의 첫·마지막 행(시간순)에서 시각을 읽고, 인원 people은 성분의 크레딧 마스크를 OR로 합친 뒤 팀 마스크로 가른다.
    comp = link_components(k.t, k.x, k.y)
    first = np.array([g[0] for g in comp], dtype=np.intp)
    final = np.array([g[-1] for g in comp], dtype=np.intp)
    f, last = k.t[first].astype(np.int64), k.t[final].astype(np.int64)
    people = np.array([np.bitwise_or.reduce(credit[g]) for g in comp], dtype=np.int64)
    nb, nr = _POPCOUNT[people & blue], _POPCOUNT[people & red]
    # (UNIT-5)의 τ, δ, 범위, s, m, spread: 열마다 dtype을 직접 지정하며 `casts.narrow`를 거치지 않는다. 범위의 두 끝(≤)은 모두 포함이다.
    return UnitTable(
        comp=comp, f=f, last=last, tau=f - C.TAU_LEAD_MS, delta=np.maximum(last + 1, f + C.DELTA_AFTER_F_MS), nb=nb, nr=nr,
        size=np.array([size_class(int(a), int(b)) for a, b in zip(nb, nr)], dtype=np.int8),
        mark=np.array([first_kill_mark(int(killer[i]), int(victim[i])) for i in first], dtype=np.int16),
        in_scope=(f >= C.SCOPE_FIRST_MS) & (f <= T_ms - C.END_CUT_MS),
        spread=np.array([_spread(k.x[g], k.y[g]) for g in comp], dtype=np.float32),
        n_kills=np.array([len(g) for g in comp], dtype=np.int16))
