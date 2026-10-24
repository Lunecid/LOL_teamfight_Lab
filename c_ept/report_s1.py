r"""S1 report: the S1 store against the link-rule prototype, the extras the S1 gate needs, and the figures.

Three groups of functions, all of them reading only the finished S1 store (``run_s1.load_s1``), never the raw files:

1. Prototype statistics (``fit_*`` and ``*_stats``). Each one recomputes a number of the prototype folder
   ``D:\LOL_Data\runs\20261019_link_rule_fit_1`` from the stored tables with the prototype's own formula (the probes there iterate
   over matches; here the same formulas run over the tables). The names of the object kinds of the probes ('towers', 'inhibitors',
   'plates'; the elder dragon is a 'dragon') are matched inside ``attribution_stats`` only. ``compare`` collects the comparisons with the
   prototype values and their pass / fail; the golden tests (tests/test_golden_s1.py) and S1_REPORT.json use the same functions.
2. Extras for the S1 gate (``coverage_extras`` ... ``age_extras``): conversion coverage, the team-blind overlap of the attribution, the
   tower tiers, the objectives with an unknown taker, the lateness of the frames and the frame age of the event rows.
3. ``report``: S1_REPORT.json and the PNG figures under ``<run>\report``.

Selections. A statistic takes ``sel``, a boolean array with one entry per match (``fit_mask`` gives the FIT matches, the prototype's set);
the ``fit_*`` functions are the prototype's set. Rows of a table belong to the matches by the offsets ``matches/*_off``.

Explained deviations from the prototype (rulings 16 and 23, pinned in ``ATTRIBUTION_DEVIATIONS``): the prototype skips the FIT matches
without kills (two plates there are in our objective count) and it counts a herald with killerTeamId 300 as overlapping an active unit,
while our rule leaves an objective with an unknown taker unattributed. Frame-clock rows (t0 is the real frame time) differ from the
prototype's nominal minutes by the rows whose frame falls at or after T - 30 s (ruling 24a).

Privacy. ``assert_no_keys`` reads ``matches/key`` to look for identifiers in the finished report text without printing them.
``cmp_golden`` also reads ``matches/key`` arrays for their lengths. S1_REPORT.json carries aggregate statistics and run/report provenance.


방법
----
이 모듈은 완성된 S1 표로 시제품의 통계를 재구성하고 설명된 편차·회계·관문 진단·그림의 수치를 함께 제공한다.
교전 단위는 연결된 킬들의 집합이며 발생은 질문 행 뒤 첫 킬, 귀속은 사건이 연결된 단위, 전환은 교전 뒤의 고유 사건 귀속이다.
시제품 뒤 전환 진단은 단위별 창의 반복 계수이므로 고유 전환과 구별하고, 진행 중 해석도 다섯 경계·집단으로 나눠 보고한다.
비율의 분모는 함수별 선택 행·사건·기간·인원 출현 횟수이며 경기 가중 평균으로 일괄 해석하지 않는다(S1–S2 Task8b).

식 목록
-------
(REP1-1) 교전 규모 코드
(REP1-2) 킬에서 재구성한 단위 통계
(REP1-3) 오브젝트 귀속·진행 중 비율
(REP1-4) 설명된 시제품 편차의 반영
(REP1-5) 시제품의 승패별 뒤 전환율
(REP1-6) 명목 분 질문 행 수
(REP1-7) 명목·저장 행 수의 차이
(REP1-8) 선택 프레임 행의 발생 분포
(REP1-9) 귀속·전환·나머지의 회계
(REP1-10) 창 안 질문 행의 비율
(REP1-11) 진행 중의 다섯 경계·집단
(REP1-12) 단위가 발생으로 나오는 행 수
(REP1-13) 뒤 전환 재현의 반올림·비율
(REP1-14) 단위 합과 사건 칸의 교차 회계
(REP1-15) 경기별 창 합집합의 길이
(REP1-16) 시간 가중 전환 창 피복
(REP1-17) 팀을 묻지 않는 인원 겹침 진단
(REP1-18) 알려지지 않은 가져간 팀의 비율
(REP1-19) 정규 프레임 위치별 지연
(REP1-20) 사건 행 나이와 공통 표본 요약
(REP1-21) 회계 표의 표시 백분율
(REP1-22) 나이 빈도 그림의 행 분모
(REP1-23) 시제품 비교의 공통 판정

공통 가정
---------
경기·킬·단위·프레임의 배열 행은 0 기반이고 참가자 ID는 1부터 시작한다.
저장 시각·나이·지연은 ms이며 초 단위 판정과 그림 표시에만 변환을 적용한다.
선택 마스크·오프셋·정렬·참조·좌표의 유효성은 완성된 저장소 계약이며 개별 함수가 모두 다시 검증하지 않는다.
복합 키의 경기 간 비충돌과 int64 범위 안전성도 입력 가정이다.
시제품 기준 경로는 대조 입력의 참조 위치이며 저장소 집계와 외부 기준 읽기는 구분된다(S1–S2 Task8b).
"""
# Ref: D:\LOL_Data\runs\20261019_link_rule_fit_1\link_g15_check.py, objective_participation_probe.py, objective_attribution_probe.py,
#      aftermath_probe.py and g2_anchor_check.py (the prototype formulas this module recomputes from the stored tables).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 3-4 and 11 (units, G3 attribution, G4 conversion, frame-clock rows)
#      and docs/superpowers/plans/2026-10-20-c-ept-s1-s2.md task 8 (the S1 report and its figures).
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from c_ept import config as C
from c_ept import event_rows, run_s1, runlog, store_core, targets, units
from c_ept.store_core import Store

PROTO_DIR = Path(r'D:\LOL_Data\runs\20261019_link_rule_fit_1')     # the prototype folder (read only)
BIG = 1 << 31                  # larger than any time in ms: key = match * BIG + time orders (match, time) in one sorted array
KILL_TABLE, UNIT_TABLE, OBJ_TABLE, ROW_TABLE = 'kills', 'units', 'objectives', 'frame_rows'

# The kind names of the prototype's objective probes (objective_participation_probe.py::_kind), keyed by our kind name. The elder dragon
# is a 'dragon' there; a building that is no tower is an 'inhibitor' there (ours: kind -1, which does not occur in FIT).
PROTO_KIND = {'dragon': 'dragon', 'elder': 'dragon', 'grubs': 'grubs', 'herald': 'herald', 'baron': 'baron', 'atakhan': 'atakhan',
              'tower': 'towers', 'inhibitor': 'inhibitors', 'plate': 'plates'}
PROTO_KINDS = ('atakhan', 'baron', 'dragon', 'grubs', 'herald', 'inhibitors', 'plates', 'towers')     # the keys of the probe JSONs
OBJ_FROM_MS = 120_000                                  # the probes count objectives from 120 s on
LINK_CHECK_U = 4264.0                                  # the thesis check distance of link_units.CHECK
AFTERMATH_MS, AFTERMATH_BIN_MS = 240_000, 10_000       # the aftermath probe: 24 bins of 10 s behind delta
SIZE_NAMES = units.SIZE_NAMES                          # size code 0-3 -> 'T', 'S', 'ASYM', 'P'

# Ruling 23: the two explained deviations of the attribution from the prototype, per prototype kind.
#   extra_n:         objectives that the prototype never sees: plates of FIT matches without any kill (it skips such matches)
#   missing_overlap: overlaps that the prototype counts and our rule does not: a herald with killerTeamId 300 (an unknown taker is
#                    never attributed, ruling 16)
ATTRIBUTION_DEVIATIONS = {'extra_n': {'plates': 2}, 'missing_overlap': {'herald': 1}}
# Ruling 24a: rows of the FIT matches with kills, nominal minutes (the prototype) and actual frame times (ours); and of all FIT matches.
ROW_PINS = {'fit_with_kills': {'nominal': 457_441, 'actual': 457_274}, 'fit_all': {'nominal': 457_445, 'actual': 457_278}}
UNITS_WITH_WINNER, FIT_IN_SCOPE_UNITS = 562_069, 574_285      # aftermath_probe.json and g2_anchor_check.json
ONSET_TOLERANCE = 0.001                                       # ruling 25


# ---- the prototype's reference files ----------------------------------------------------------------------------------------------
def load_prototype(folder: Path = PROTO_DIR) -> dict:
    """The prototype references: ``npz`` (fit_kills.npz as a dict of arrays, ``allow_pickle=False``) and the JSON files ``g15``
    (link_g15_check.link_15s), ``part`` (objective_participation_probe), ``attr`` (objective_attribution_probe), ``after``
    (aftermath_probe) and ``anchor`` (g2_anchor_check). FileNotFoundError when one is missing.

    시제품의 읽기 전용 기준값을 모아 같은 계산의 대조 입력으로 제공한다(S1–S2 Task8b).
    """
    folder = Path(folder)

    def j(name):
        return json.loads((folder / name).read_text(encoding='utf-8'))
    with np.load(folder / 'fit_kills.npz', allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    return {'npz': npz, 'g15': j('link_g15_check.json')['link_15s'], 'part': j('objective_participation_probe.json'),
            'attr': j('objective_attribution_probe.json'), 'after': j('aftermath_probe.json'), 'anchor': j('g2_anchor_check.json')}


# ---- table access -------------------------------------------------------------------------------------------------------------------
QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def _ratio(a, b):
    """numerator / denominator as a float, None when denominator is 0 (a selection can be empty).

    식 (REP1-3)의 해당 항을 계산한다.
    """
    return float(a / b) if b else None


def _dist(x: np.ndarray) -> dict:
    """n, mean, standard deviation and the percentiles ``QUANTILES`` of a sample (just n = 0 for an empty one).

    식 (REP1-20)의 해당 항을 계산한다.
    """
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()), **{f'p{q}': float(np.percentile(x, q)) for q in QUANTILES}}


def _col(s: Store, name: str) -> np.ndarray:
    """A stored array as an ndarray (a view of the read-only memory map)."""
    return np.asarray(s.array(name))


def _off(s: Store, table: str) -> np.ndarray:
    """The (n_matches + 1) offsets of a table."""
    return _col(s, 'matches/' + run_s1.OFFSETS[table])


def _row_match(s: Store, table: str) -> np.ndarray:
    """The match index of every row of a table (int64)."""
    off = _off(s, table)
    return np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))


def role_mask(s: Store, role: str) -> np.ndarray:
    """(n_matches,) bool: the matches of the macro role ``role``.

    요청한 역할 코드의 경기만 선택하여 집계 분모를 고정한다(S1–S2 Task8).
    """
    return _col(s, 'matches/role') == C.ROLES.index(role)


def fit_mask(s: Store) -> np.ndarray:
    """(n_matches,) bool: the FIT matches, the prototype's set.

    시제품과 비교할 FIT 경기 집합을 반환한다(S1–S2 Task8b).
    """
    return role_mask(s, 'FIT')


def patch_mask(s: Store, patch: str) -> np.ndarray:
    """(n_matches,) bool: the matches of one patch.

    한 패치의 저장 경기를 선택하여 패치별 통계를 구분한다(S1–S2 Task8b).
    """
    return _col(s, 'matches/patch') == C.PATCHES.index(patch)


def fit_matches(s: Store) -> np.ndarray:
    """Indices of the FIT matches in stored order (the order of the prototype's match index).

    FIT 경기의 저장 행 번호를 순서대로 반환한다(S1–S2 Task8b).
    """
    return np.flatnonzero(fit_mask(s))


def fit_kills(s: Store, name: str) -> np.ndarray:
    """A column of ``kills`` over the FIT matches, in stored order.

    FIT에 속한 킬 열을 저장 순서대로 제공한다(S1–S2 Task8b).
    """
    return _col(s, f'kills/{name}')[fit_mask(s)[_row_match(s, KILL_TABLE)]]


def fit_T(s: Store) -> np.ndarray:
    """T_ms of the FIT matches (int64).

    FIT 경기의 종료 시각을 저장 단위로 제공한다(S1–S2 Task8b).
    """
    return _col(s, 'matches/T_ms')[fit_mask(s)]


def fit_team(s: Store) -> np.ndarray:
    """The (n_fit, 10) team ids of the FIT matches.

    FIT 경기 참가자의 팀 정보를 같은 경기 순서로 제공한다(S1–S2 Task8b).
    """
    return _col(s, 'matches/team')[fit_mask(s)]


def kill_txy(s: Store, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, x, y) of the kills of match ``m`` (a stored match index), in time order.

    지정 경기의 킬 시각과 위치를 저장 순서대로 읽는다(S1–S2 Task8b).
    """
    off = _off(s, KILL_TABLE)
    a, b = int(off[m]), int(off[m + 1])
    return _col(s, 'kills/t')[a:b], _col(s, 'kills/x')[a:b], _col(s, 'kills/y')[a:b]


def unit_lists(s: Store, m: int) -> list[list[int]]:
    """The units of match ``m`` as lists of the match's local kill indices, units in table order and kills ascending: the form of the
    prototype's ``link_components``.

    전역 단위 참조에서 경기 시작 오프셋을 빼고 킬을 안정 정렬하여 시제품의 지역 킬 목록으로 바꾼다(S1–S2 Task8b).
    단위가 없는 경기는 빈 목록을 반환한다(unit_lists).
    """
    k_off, u_off = _off(s, KILL_TABLE), _off(s, UNIT_TABLE)
    n_units = int(u_off[m + 1] - u_off[m])
    if n_units == 0:
        return []
    unit = _col(s, 'kills/unit')[int(k_off[m]):int(k_off[m + 1])].astype(np.int64) - int(u_off[m])
    order = np.argsort(unit, kind='stable')
    cuts = np.cumsum(np.bincount(unit, minlength=n_units))[:-1]
    return [g.tolist() for g in np.split(order, cuts)]


# ---- units: link_g15_check ------------------------------------------------------------------------------------------------------
_POPCOUNT = np.array([bin(m).count('1') for m in range(1 << 10)], dtype=np.int64)


def _bit(pid: np.ndarray) -> np.ndarray:
    """Bit (pid - 1) for pid in 1-10, else 0.

    식 (REP1-2)의 해당 항을 계산한다.
    """
    pid = pid.astype(np.int64)
    return np.where((pid >= 1) & (pid <= 10), np.left_shift(1, np.clip(pid - 1, 0, 9)), 0)


def _unit_people(s: Store) -> np.ndarray:
    """(n_units,) int64 bit mask of the people of every unit, the prototype's ``part``: killer and victim when they are participants
    1-10, and the assisters 1-10 (bit pid - 1), over the unit's kills.

    식 (REP1-2)의 해당 항을 계산한다.
    """
    kill_unit = _col(s, 'kills/unit').astype(np.int64)
    people_kill = _bit(_col(s, 'kills/killer')) | _bit(_col(s, 'kills/victim')) | (_col(s, 'kills/amask').astype(np.int64) & 0x3FF)
    people = np.zeros(len(_col(s, 'units/f')), dtype=np.int64)
    np.bitwise_or.at(people, kill_unit, people_kill)
    return people


def _team_masks(s: Store) -> tuple[np.ndarray, np.ndarray]:
    """(blue, red): (n_matches,) int64 bit masks (bit pid - 1) of the participants on team 100 / 200.

    식 (REP1-2)의 해당 항을 계산한다.
    """
    team = _col(s, 'matches/team')
    bits = np.left_shift(1, np.arange(10, dtype=np.int64))
    return ((team == units.BLUE) * bits).sum(axis=1), ((team == units.RED) * bits).sum(axis=1)


def size_code(nb: np.ndarray, nr: np.ndarray) -> np.ndarray:
    """The thesis size cohort of the people per team, the prototype's ``link_units.cohort``: 0 T (both >= 4), 1 S (both 2-3), 2 ASYM (one
    side 2-3, the other >= 4), 3 P (a side below 2).

    수식
    ----
    (REP1-1)  small = min(blue_count, red_count), large = max(blue_count, red_count),
             size = 3 (small < 2), 0 (small ≥ 4), 1 (2 ≤ small이고 large ≤ 3), 그 밖에는 2.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      blue_count, red_count, small, large | 양 팀 인원과 그 최솟값·최댓값 | 명 | 비음수 정수 입력 계약 | nb, nr, lo, hi
      size | T·S·ASYM·P 규모의 저장 코드 | — | T=0, S=1, ASYM=2, P=3 | 반환값, units.SIZE_*

    해석
    ----
    규모는 양 팀의 참여 인원으로 나눈 결과 표지이며 발생 입력이 아니다(설계서 §2).
    한 팀 인원이 적은 경우를 먼저 분류하고 나머지를 양쪽이 큰 경우·양쪽이 작은 경우·비대칭인 경우로 나눈다.
    인원 배열의 정수성과 비음수성은 이 함수에서 검사하지 않는다.

    근거
    ----
    설계서 §2(규모); S1–S2 계획 Task3; size_code(입력 계약).
    """
    lo, hi = np.minimum(nb, nr), np.maximum(nb, nr)
    return np.where(lo < 2, units.SIZE_P, np.where(lo >= 4, units.SIZE_T, np.where(hi <= 3, units.SIZE_S, units.SIZE_ASYM)))


def _unit_kills(x: np.ndarray, y: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(unit_ids, n_kills, spread, first, last)`` of every unit that has kills in the arrays (``unit`` is the unit of each kill), ids
    ascending. spread is float64, the largest pairwise distance with the prototype's expression (sqrt of the largest of dx**2 + dy**2
    over all pairs), 0 for a unit of one kill; the units are handled in groups of equal size and every element goes through the same
    operations as in the prototype's per-unit expression. ``first`` and ``last`` are the positions in the arrays of the unit's first and
    last kill (the kills are in time order, so these are its earliest and latest).

    식 (REP1-2)의 해당 항을 계산한다.
    """
    # 식 (REP1-2)의 거리 재구성은 저장 float32 값이 아닌 킬 좌표에서 시작한다(S1–S2 Task8).
    order = np.argsort(unit, kind='stable')
    ids, start, cnt = np.unique(unit[order], return_index=True, return_counts=True)
    out = np.zeros(len(ids))
    for size in np.unique(cnt[cnt >= 2]):
        g = np.flatnonzero(cnt == size)
        idx = order[start[g][:, None] + np.arange(size)]
        px, py = x[idx], y[idx]
        d2 = (px[:, :, None] - px[:, None, :]) ** 2 + (py[:, :, None] - py[:, None, :]) ** 2
        out[g] = np.sqrt(d2.max(axis=(1, 2)))
    return ids, cnt, out, order[start], order[start + cnt - 1]


def unit_stats(s: Store, sel: np.ndarray) -> dict:
    """The statistics of ``link_g15_check.link_15s`` over the units of the selected matches, from the stored tables.

    ``units_per_kill`` (units over all kills of the selection), ``share_single_kill``, ``spread_gt_4264_share`` / ``_n`` /
    ``spread_p99`` / ``spread_max`` (over the units of two or more kills, the spread recomputed in float64 from the stored kill
    positions), ``cohort_share_in_scope`` (the size cohort recomputed from the kills' people, units with 120 s <= f <= T - 30 s by the
    prototype's float expression), plus ``n_kills`` / ``n_units`` / ``n_in_scope``. The agreement of the stored columns with the
    recomputed values is reported as ``stored_*`` flags: ``size`` and ``nb`` / ``nr`` (from the people), ``in_scope``, ``spread`` (float32
    of the float64 value) and the times and counts ``f``, ``last``, ``tau`` = f - 15,000, ``delta`` = max(last + 1, f + 20,000),
    ``n_kills`` and ``kill0`` (the prototype's literals, from the kills of each unit).


    수식
    ----
    (REP1-2)  spread_unit = max_{kill,other∈unit} √((pos_x_kill−pos_x_other)^2 + (pos_y_kill−pos_y_other)^2),
             singleton의 spread_unit = 0,
             bit(pid) = 2^(pid−1) (1 ≤ pid ≤ 10), 그 밖에는 0,
             people_unit = ⋁_{kill∈unit}(bit(killer) ∨ bit(victim) ∨ (amask ∧ 1023)),
             blue_count = popcount(people_unit ∧ blue_mask), red_count도 같은 식,
             first_ms = 첫 킬, last_ms = 마지막 킬, start_ms = first_ms − 15000,
             end_ms = max(last_ms + 1, first_ms + 20000),
             scope_unit = 1[120 ≤ first_ms/1000 ≤ T_ms/1000 − 30].
             units_per_kill = unit_count/kill_count, single_share = stored_single_count/unit_count,
             spread_share = #{spread_unit > 4264 : multi}/#multi,
             size_share_class = #{scope_unit ∧ size=class}/#{scope_unit}.
             분모 0의 비율은 None; spread_p99와 spread_max는 multi가 비면 None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      unit, kill, other, pid | 단위·킬 행과 비교 킬·정수 변환한 입력 ID | — | 저장 행은 0 기반; pid는 int64 변환 후 값, 유효 참가자 ID만 1..10 | ids, unit, first, last, killer, victim
      pos_x_kill, pos_y_kill, spread_unit | 킬 좌표와 최대 거리 | 지도 단위 | 유효 좌표·시간순 입력 계약 | kills/x, kills/y, spread
      bit, people_unit, blue_mask, red_mask, amask | 참가자 비트와 인원/팀/도움 마스크 | — | 하위 10비트 | _bit, _unit_people, _team_masks
      blue_count, red_count, size, class | 인원과 규모 코드·범주 | 명; — | (REP1-1) | nb, nr, code, SIZE_NAMES
      first_ms, last_ms, start_ms, end_ms, T_ms | 첫/마지막 킬·창 시작/끝·경기 끝 | ms | int 저장값 | f, last_kill, times, matches/T_ms
      scope_unit, multi | 범위 안 표시와 복수 킬 집합 | — | bool; cnt ≥ 2 | ins, multi
      unit_count, kill_count, stored_single_count | 선택 단위·킬·저장 킬 수가 하나인 단위 수 | 개 | ≥ 0 | n_units, n_kills, n_kills_unit
      units_per_kill, single_share, spread_share, size_share_class | 각 보고 비율 | — | 분모가 양수일 때, 아니면 None | 반환 필드
      spread_p99, spread_max | 복수 킬 단위 거리의 백분위·최대 | 지도 단위 | 표본이 있으면 계산 | sp

    해석
    ----
    선택 경기의 킬로 시간·인원·거리를 다시 계산하여 저장 열과 맞대본다(S1–S2 Task8).
    킬의 단위 행과 선택 단위 행이 일치하는지는 명시적으로 검사하지만, 좌표의 유한성이나 킬의 시간순서는 저장소 계약이다.
    거리 요약은 복수 킬 단위만 사용하며, 저장 거리 검사는 재계산 거리의 float32 변환값과 정확히 비교한다(S1–S2 Task8 시제품 재현).
    범위 판정은 초 단위 부동소수점 식을 사용하고 끝점을 포함하므로 질문 행의 엄격한 끝 경계와 구별한다.
    빈 분모와 빈 거리 집합은 위 식의 누락 규약으로 보고한다(S1–S2 Task8).

    근거
    ----
    설계서 §2; S1–S2 계획 Task8; unit_stats·_unit_kills(시제품 재현); _ratio(빈 분모).
    """
    um, km = _row_match(s, UNIT_TABLE), _row_match(s, KILL_TABLE)
    u_sel, k_sel = sel[um], sel[km]
    n_kills, n_units = int(k_sel.sum()), int(u_sel.sum())
    # spreads and sizes of the units of the selected matches
    ids, cnt, spread, first, last = _unit_kills(_col(s, 'kills/x')[k_sel], _col(s, 'kills/y')[k_sel], _col(s, 'kills/unit')[k_sel])
    if not np.array_equal(ids, np.flatnonzero(u_sel)):
        raise ValueError('the units of the kills are not the units of the selected matches')
    multi = cnt >= 2
    sp = spread[multi]
    n_kills_unit = _col(s, 'units/n_kills')[u_sel]
    # the times of every unit from its kills, with the prototype's literals: tau = f - 15000, delta = max(last + 1, f + 20000)
    kill_t = _col(s, 'kills/t')[k_sel]
    f, last_kill = kill_t[first], kill_t[last]
    times = {'f': f, 'last': last_kill, 'tau': f - 15_000, 'delta': np.maximum(last_kill + 1, f + 20_000), 'n_kills': cnt,
             'kill0': np.flatnonzero(k_sel)[first]}
    times_ok = all(np.array_equal(_col(s, f'units/{c}')[u_sel], v) for c, v in times.items())
    # size cohort from the people of the kills (the prototype's definition), in scope by the prototype's float expression
    blue, red = _team_masks(s)
    people = _unit_people(s)[u_sel]
    nb, nr = _POPCOUNT[people & blue[um[u_sel]]], _POPCOUNT[people & red[um[u_sel]]]
    code = size_code(nb, nr)
    T = _col(s, 'matches/T_ms')[um[u_sel]].astype(np.float64) / 1000.0
    ins = (120 <= f / 1000.0) & (f / 1000.0 <= T - 30)
    stored_spread = _col(s, 'units/spread')[u_sel]
    return {
        'n_kills': n_kills, 'n_units': n_units, 'n_in_scope': int(ins.sum()),
        'units_per_kill': _ratio(n_units, n_kills), 'share_single_kill': _ratio((n_kills_unit == 1).sum(), n_units),
        'multi_kill_units': int(sp.size), 'spread_gt_4264_share': _ratio((sp > LINK_CHECK_U).sum(), sp.size),
        'spread_gt_4264_n': int((sp > LINK_CHECK_U).sum()), 'spread_p99': float(np.percentile(sp, 99)) if sp.size else None,
        'spread_max': float(sp.max()) if sp.size else None,
        'cohort_share_in_scope': {name: _ratio((code[ins] == k).sum(), ins.sum()) for k, name in enumerate(SIZE_NAMES)},
        'stored_in_scope_equals_prototype_rule': bool(np.array_equal(ins, _col(s, 'units/in_scope')[u_sel])),
        'stored_times_equal_prototype_rule': bool(times_ok),
        'stored_nb_nr_equal_people_rule': bool(np.array_equal(nb, _col(s, 'units/nb')[u_sel])
                                               and np.array_equal(nr, _col(s, 'units/nr')[u_sel])),
        'stored_size_equals_people_rule': bool(np.array_equal(code, _col(s, 'units/size')[u_sel])),
        'stored_spread_equals_float32_of_float64': bool(np.array_equal(stored_spread, spread.astype(np.float32))),
    }


def fit_unit_stats(s: Store) -> dict:
    """``unit_stats`` over the FIT matches.

    FIT 선택으로 (REP1-2)의 통계를 계산한다(S1–S2 Task8b).
    """
    return unit_stats(s, fit_mask(s))


def fit_in_scope_units(s: Store) -> int:
    """Units of the FIT matches that are in scope (120 s <= f <= T - 30 s), as stored.

    FIT의 저장 범위 표시를 합하여 시제품 대조 개수로 제공한다(S1–S2 판정 24a).
    """
    return int(_col(s, 'units/in_scope')[fit_mask(s)[_row_match(s, UNIT_TABLE)]].sum())


# ---- objectives: participation and attribution probes ----------------------------------------------------------------------------
def _active(s: Store, match: np.ndarray, t: np.ndarray) -> np.ndarray:
    """For objectives at time ``t`` (ms) of match ``match``: is some unit of the match active, tau <= t < delta (all units, in scope or
    not)? Counted as (#tau <= t) - (#delta <= t) in one sorted array of keys match * BIG + time; units of other matches cancel.

    식 (REP1-3)의 해당 항을 계산한다.
    """
    um = _col(s, 'units/match').astype(np.int64)
    starts = np.sort(um * BIG + _col(s, 'units/tau'))
    ends = np.sort(um * BIG + _col(s, 'units/delta'))
    key = match.astype(np.int64) * BIG + t
    return (np.searchsorted(starts, key, side='right') - np.searchsorted(ends, key, side='right')) > 0


def _proto_kind_index(kind: np.ndarray) -> np.ndarray:
    """Our kind index -> index in ``PROTO_KINDS`` (-1 for kind -1: another monster or building, no entry in the probes)."""
    table = np.full(len(C.O_KINDS) + 1, -1, dtype=np.int64)          # the last slot takes kind -1
    for k, name in enumerate(C.O_KINDS):
        table[k] = PROTO_KINDS.index(PROTO_KIND[name])
    return table[kind.astype(np.int64)]


def attribution_stats(s: Store, sel: np.ndarray) -> dict:
    """The counts and shares of ``objective_participation_probe`` / ``objective_attribution_probe`` over the objectives of the selected
    matches that have a position and lie at 120 s or later, by prototype kind and for ``all``.

    Per kind: ``n``; ``overlap`` = objectives attributed to a unit (our ``objectives/unit >= 0``: the prototype's overlap of a credited
    champion with the people of an active unit); ``no_champion`` (no credited champion); ``active`` (some unit is active at the time);
    and the shares ``participant_overlap_with_active_unit``, ``no_champion_credit``, ``active_unit`` and ``share_no_active_unit``.
    ``other`` counts the objectives of kind -1 (none in FIT); they are part of ``all`` like the prototype's ``other_monster``.


    수식
    ----
    (REP1-3)  active_ev = 1[#{동일 경기의 start_ms ≤ time_ms_ev} − #{동일 경기의 end_ms ≤ time_ms_ev} > 0],
             집계 집합 = 선택 경기 ∧ 위치 있음 ∧ time_ms_ev ≥ 120000,
             share_field = count_field / obj_count,
             no_active_share = (obj_count − active_count) / obj_count.
             count_field는 attributed(unit≥0), no_champion(cred=0), active의 개수이며,
             obj_count = 0이면 각 비율은 None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ev, time_ms_ev, start_ms, end_ms | 오브젝트와 단위 창의 시각 | —; ms | 0 기반 행, 유효 시간 계약 | objectives/t, units/tau, units/delta
      active_ev, active_count | 같은 경기의 활성 단위 유무와 해당 사건 수 | —; 개 | bool; ≥ 0 | _active, active
      obj_count, count_field, share_field, no_active_share | 종류별/전체 사건 수·표시 수·비율 | 개; 개; — | 비율은 None 가능 | entry, shares, out
      attributed, no_champion | 저장 귀속·크레딧 없음 표시 | — | bool | objectives/unit, objectives/cred

    해석
    ----
    진행 중 창은 시작을 포함하고 끝을 제외하며, 귀속 여부는 저장된 단위 참조에서 읽는다(설계서 §3.2).
    종류를 시제품의 분류로 바꾸고 모든 선택 사건을 전체 분모에 포함하므로, 기타 종류가 있어도 전체에서 빠지지 않는다(S1–S2 Task8b).
    기타 종류가 없을 때만 해당 항목을 사건 수만 있는 사전으로 줄이며, 비율의 빈 분모는 None이다.
    활성 검색은 정렬된 복합 키를 사용하므로 시간의 유효 범위와 키의 비충돌은 저장소 입력 계약이다.

    근거
    ----
    설계서 §3.2; S1–S2 Task8b 판정 23; _active(복합 키).
    """
    om = _row_match(s, OBJ_TABLE)
    keep = sel[om] & _col(s, 'objectives/has_pos') & (_col(s, 'objectives/t') >= OBJ_FROM_MS)
    pk = _proto_kind_index(_col(s, 'objectives/kind')[keep])
    attributed = _col(s, 'objectives/unit')[keep] >= 0
    no_champion = _col(s, 'objectives/cred')[keep] == 0
    active = _active(s, om[keep], _col(s, 'objectives/t')[keep])
    out = {}

    def entry(m):
        n = int(m.sum())
        c = {'n': n, 'overlap': int((attributed & m).sum()), 'no_champion': int((no_champion & m).sum()), 'active': int((active & m).sum())}
        return c

    def shares(c):
        n = c['n']
        c.update(participant_overlap_with_active_unit=_ratio(c['overlap'], n), no_champion_credit=_ratio(c['no_champion'], n),
                 active_unit=_ratio(c['active'], n), share_no_active_unit=_ratio(n - c['active'], n))
        return c
    for i, name in enumerate(PROTO_KINDS):
        out[name] = shares(entry(pk == i))
    other = entry(pk < 0)
    out['other'] = {'n': other['n']} if other['n'] == 0 else shares(other)
    total = entry(np.ones(len(pk), dtype=bool))
    out['all'] = shares(total)
    return out


def fit_attribution_stats(s: Store) -> dict:
    """``attribution_stats`` over the FIT matches.

    FIT 선택으로 (REP1-3)의 귀속 통계를 계산한다(S1–S2 Task8b).
    """
    return attribution_stats(s, fit_mask(s))


def expected_attribution(part: dict, attr: dict) -> dict:
    """What the prototype's attribution numbers become under the two explained deviations (``ATTRIBUTION_DEVIATIONS``).

    From the participation probe ``part`` (per kind n and the overlap share) and the attribution probe ``attr`` (share without an active
    unit): the counts are rebuilt (share * n, rounded to an integer: the probes divide integers), the deviations are applied (``extra_n``
    objectives with no active unit and no overlap are added to n; ``missing_overlap`` overlaps are taken off the kind and ``all``) and
    the shares are recomputed from the adjusted counts. Returns {kind: {n, overlap, no_active, participant_overlap_with_active_unit,
    share_no_active_unit}} for the kinds and ``all``.


    수식
    ----
    (REP1-4)  adjusted_count_kind = original_count_kind + extra_kind,
             overlap_kind = round(part_share_kind · original_count_kind) − missing_kind,
             no_active_kind = round(attr_share_kind · attr_count_kind) + extra_kind,
             adjusted_share = adjusted_numerator / adjusted_count_kind.
             전체 개수는 종류별 보정 개수의 합, 전체 비율은 합한 분자/합한 분모이다.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      kind | 시제품 종류 이름 | — | PROTO_KINDS | name
      original_count_kind, attr_count_kind | 참여·귀속 조사 원래 분모 | 개 | 양수 입력 가정 | part[name]['n'], attr['by_kind'][name]['n']
      part_share_kind, attr_share_kind | 겹침 비율·비활성 비율 | — | 유효 비율 가정 | participant_overlap_with_active_unit, share_no_active_unit
      extra_kind, missing_kind | 추가 사건·빼는 겹침 | 개 | plates 추가 2, herald 겹침 1, 그 밖은 0 | ATTRIBUTION_DEVIATIONS
      adjusted_count_kind, overlap_kind, no_active_kind, adjusted_share, adjusted_numerator | 보정 분모·분자·비율 | 개; — | 분모 > 0 가정 | out

    해석
    ----
    원래 비율에서 정수 개수를 round로 복원한 뒤 설명된 편차를 반영하고 비율을 다시 계산한다(S1–S2 판정 23).
    추가 사건은 비활성·비겹침으로 더하고, 누락 겹침은 분모를 바꾸지 않고 뺀다.
    round는 Python의 가장 가까운 정수와 동률 짝수 규칙을 따르며, 보정 뒤 분모가 양수인지는 여기에서 검사하거나 보정하지 않는다.

    근거
    ----
    S1–S2 Task8b 판정 23(편차); expected_attribution(복원·분모).
    """
    extra, missing = ATTRIBUTION_DEVIATIONS['extra_n'], ATTRIBUTION_DEVIATIONS['missing_overlap']
    out = {}
    for name in PROTO_KINDS:
        n0 = part[name]['n']
        overlap = round(part[name]['participant_overlap_with_active_unit'] * n0) - missing.get(name, 0)
        no_active = round(attr['by_kind'][name]['share_no_active_unit'] * attr['by_kind'][name]['n']) + extra.get(name, 0)
        n = n0 + extra.get(name, 0)
        out[name] = {'n': n, 'overlap': overlap, 'no_active': no_active, 'participant_overlap_with_active_unit': overlap / n,
                     'share_no_active_unit': no_active / n}
    n = sum(v['n'] for v in out.values())
    overlap, no_active = sum(v['overlap'] for v in out.values()), sum(v['no_active'] for v in out.values())
    out['all'] = {'n': n, 'overlap': overlap, 'no_active': no_active, 'participant_overlap_with_active_unit': overlap / n,
                  'share_no_active_unit': no_active / n}
    return out


# ---- aftermath: aftermath_probe -------------------------------------------------------------------------------------------------
def aftermath_stats(s: Store, sel: np.ndarray, chunk: int = 100_000) -> dict:
    """The winner / loser conversion rates of ``aftermath_probe`` over the in-scope units of the selected matches that have a winner.

    The prototype's formula on the stored tables: every such unit has its own window (delta, delta + 240 s] cut into 24 bins of 10 s; an
    objective with a position that no unit claims (``objectives/unit == -1``) and a known taker is counted in the bin of its time, for the
    unit's winner side or its loser side (the winner is the sign of the kill gold, bounty + shutdown bounty, by the killer's team). The
    objectives are counted once per unit window, so one objective is in the windows of several units: the unique ``conv_unit`` is not
    used. The exposure of a bin is the part of it before the game end T. Rates are per unit-minute.

    Returns ``units_with_winner``, ``bins_s`` (the lower edges), ``winner_count`` / ``loser_count`` / ``exposure_s`` per bin and the
    rates ``winner_rate`` / ``loser_rate`` / ``excess_rate`` (per unit-minute, not rounded) and ``cumulative_excess_per_unit``.


    수식
    ----
    (REP1-5)  kill_margin = K[:,2] + K[:,4] − K[:,3] − K[:,5],
             선택 단위 = 선택 경기 ∧ in_scope ∧ kill_margin ≠ 0,
             win_side = 1[kill_margin < 0], edge_bin = 10000 · bin (bin = 0..24),
             count_{bin,winner} = Σ_unit #{같은 경기의 선택 오브젝트: δ_unit+edge_bin < time_ms ≤ δ_unit+edge_{bin+1}, taker=win_side_unit},
             count_{bin,loser}는 taker=1−win_side_unit로 같은 식,
             exposure_s_bin = Σ_unit max(0, min(δ_unit+edge_{bin+1}, T_ms_unit) − δ_unit−edge_bin)/1000,
             rate_{bin,side} = 60 · count_{bin,side}/max(exposure_s_bin, 1), side∈{winner,loser},
             excess_bin = rate_{bin,winner} − rate_{bin,loser},
             cumulative_bin = Σ_{slot≤bin}(winner_count_slot−loser_count_slot)/max(unit_count, 1).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      unit, bin, slot, edge_bin | 단위와 구간·누적 위치·구간 끝점 | —; ms | 0 기반, 구간 0..23 | use, edges
      K, kill_margin, win_side | 저장 킬 결과·도움 풀 제외 골드 차·승리 팀 | 개·골드; 골드; — | 차가 0인 단위 제외, 팀 0/1 | units/K, gold, win_side
      δ_unit, T_ms_unit, time_ms | 교전 끝·경기 끝·사건 시각 | ms | 저장소 계약 | delta, T, objectives/t
      count_{bin,side}, winner_count_slot, loser_count_slot | 단위 창별 승자·패자 개수 합 | 개 | 비음수 | win, lose, counts
      exposure_s_bin, rate_{bin,side}, excess_bin | 합한 관측 시간·단위분당 비율·차 | s; 개/단위분 | 노출 ≥ 0, 차는 부호 가능 | exposure_s, rate_w, rate_l
      unit_count, cumulative_bin | 승자가 있는 단위 수·단위당 누적 초과 | 개; 개/단위 | 분모는 최소 1 | n_units, cumulative_excess_per_unit

    해석
    ----
    이 진단은 고유 전환 귀속 대신 각 교전의 뒤 창을 별도로 세므로 한 사건이 여러 창에 들어갈 수 있다(S1–S2 Task8b aftermath).
    선택 오브젝트는 위치가 있고 어느 단위에도 귀속되지 않았으며 가져간 팀이 알려진 사건이다.
    승자는 도움 풀을 넣지 않은 기록 bounty와 shutdown 차이로 정하므로 이후 가치 모듈의 골드 정의와 구별한다(S1–S2 Task8b 시제품 계약).
    구간 노출은 경기 끝에서 자르고 분모 하한으로 빈 노출과 빈 단위의 나눗셈을 처리한다.
    개수 검색은 창 끝점을 포함하지만 경기 끝으로 개수 창을 별도 절단하지 않으므로 유효 사건 시각은 저장소 계약이다.

    근거
    ----
    S1–S2 계획 Task8; S1–S2 Task8b Aftermath; aftermath_stats(시제품 계약).
    """
    um = _row_match(s, UNIT_TABLE)
    K = _col(s, 'units/K').astype(np.int64)
    gold = K[:, 2] + K[:, 4] - K[:, 3] - K[:, 5]              # blue minus red: bounty + shutdown bounty of the kills each team scored
    use = np.flatnonzero(sel[um] & _col(s, 'units/in_scope') & (gold != 0))
    nb = AFTERMATH_MS // AFTERMATH_BIN_MS
    match, delta = um[use], _col(s, 'units/delta')[use]
    win_side = (gold[use] < 0).astype(np.int64)               # 0 blue, 1 red
    T = _col(s, 'matches/T_ms')[match]
    # objectives that no unit claims and that have a position and a known taker, by taker side: sorted keys match * BIG + time
    om = _row_match(s, OBJ_TABLE)
    free = sel[om] & _col(s, 'objectives/has_pos') & (_col(s, 'objectives/unit') < 0) & (_col(s, 'objectives/taker') >= 0)
    key = om * BIG + _col(s, 'objectives/t')
    keys = [np.sort(key[free & (_col(s, 'objectives/taker') == side)]) for side in (0, 1)]
    edges = AFTERMATH_BIN_MS * np.arange(nb + 1, dtype=np.int64)
    win, lose = np.zeros(nb, dtype=np.int64), np.zeros(nb, dtype=np.int64)
    exposure_ms = np.zeros(nb, dtype=np.int64)
    for a in range(0, len(use), chunk):
        b = min(a + chunk, len(use))
        base = (match[a:b] * BIG + delta[a:b])[:, None] + edges[None, :]                         # (u, nb + 1) edges of the bins
        counts = [np.diff(np.searchsorted(k, base, side='right'), axis=1) for k in keys]       # objectives in (edge_k, edge_k+1]
        side = win_side[a:b][:, None]
        win += np.where(side == 0, counts[0], counts[1]).sum(axis=0)
        lose += np.where(side == 0, counts[1], counts[0]).sum(axis=0)
        cut = np.minimum(delta[a:b, None] + edges[None, 1:], T[a:b, None]) - (delta[a:b, None] + edges[None, :-1])
        exposure_ms += np.maximum(cut, 0).sum(axis=0)
    n_units = len(use)
    # 식 (REP1-5)의 노출 합을 초로 바꾼 뒤 단위분당 비율로 환산한다(S1–S2 Task8b).
    exposure_s = exposure_ms / 1000.0
    rate_w, rate_l = win / np.maximum(exposure_s, 1) * 60.0, lose / np.maximum(exposure_s, 1) * 60.0
    return {'units_with_winner': n_units, 'bins_s': (edges[:-1] // 1000).tolist(), 'winner_count': win.tolist(),
            'loser_count': lose.tolist(), 'exposure_s': exposure_s.tolist(), 'winner_rate': rate_w.tolist(), 'loser_rate': rate_l.tolist(),
            'excess_rate': (rate_w - rate_l).tolist(),
            'cumulative_excess_per_unit': (np.cumsum(win - lose) / max(n_units, 1)).tolist()}


def fit_aftermath_stats(s: Store) -> dict:
    """``aftermath_stats`` over the FIT matches.

    FIT 선택으로 (REP1-5)의 시제품 뒤 전환을 계산한다(S1–S2 Task8b).
    """
    return aftermath_stats(s, fit_mask(s))


# ---- rows and onsets: g2_anchor_check -------------------------------------------------------------------------------------------
def nominal_row_count(T_ms: int) -> int:
    """Rows of the prototype for a game of T_ms: t0 = 120, 180, ... s while t0 < T - 30 s (``g2_anchor_check.py``), with its float
    expression for the end.

    수식
    ----
    (REP1-6)  T = float(T_ms)/1000,
             nominal_count = #{minute_idx ≥ 0 : 120 + 60·minute_idx < T − 30}.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      T_ms, T | 경기 끝 시각의 ms와 s 표현 | ms; s | 유효 유한 종료 시각 계약 | T_ms, T
      minute_idx, nominal_count | 명목 분 위치와 행 수 | —; 행 | 0 기반 비음수 정수 | while의 t0, n

    해석
    ----
    시제품의 명목 분 시계를 그대로 반복하여 행 수를 센다(S1–S2 판정 24a).
    경기 끝을 초 단위 float로 바꾼 뒤 엄격한 끝 부등식을 적용하므로 끝과 같은 명목 시각은 제외한다.
    유한 종료 시각은 호출 계약이며 무한 종료값을 별도로 거부하지 않는다.

    근거
    ----
    S1–S2 Task8b 판정 24a(명목·실제 시계); nominal_row_count.
    """
    T, t0, n = float(T_ms) / 1000.0, 120, 0
    while t0 < T - 30:
        n += 1
        t0 += 60
    return n


def row_counts(s: Store) -> dict:
    """The rows of the FIT matches by set: ``fit_with_kills`` (the matches with at least one kill: the prototype's set) and ``fit_all``.

    ``actual`` is the number of stored frame-clock rows (t0 = the real frame time); ``nominal`` counts the prototype's minutes 120, 180, ...
    s from T alone; ``frame_late`` is their difference (rows whose frame falls at or after T - 30 s); ``matches`` the match count.


    수식
    ----
    (REP1-7)  nominal = Σ_{u∈selected} nominal_row_count(T_ms_u),
             actual = Σ_{u∈selected} (row_off_{u+1}−row_off_u), frame_late = nominal − actual.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      u, selected | 저장 경기 행과 집계 경기 집합 | — | 0 기반, FIT 전체 또는 킬 있는 FIT | m, fit, kills
      T_ms_u, row_off_u | 경기 끝과 누적 프레임 행 오프셋 | ms; 행 | 유효 저장소 계약 | T, rows
      nominal, actual, frame_late | 명목 행·저장 행·그 차 | 행 | 정수, 차의 원인을 재분류하지 않음 | out

    해석
    ----
    킬이 있는 FIT와 모든 FIT를 분리하여 시제품의 분모를 맞춘다(S1–S2 판정 24a).
    frame_late는 두 합의 차이이며 누락 이유별 서로 겹치지 않는 개수를 별도로 계산한 값은 아니다.

    근거
    ----
    S1–S2 Task8b 판정 24a(집단별 행 수); (REP1-6).
    """
    fit = fit_mask(s)
    kills = np.diff(_off(s, KILL_TABLE))
    rows = np.diff(_off(s, ROW_TABLE))
    T = _col(s, 'matches/T_ms')
    out = {}
    for name, m in (('fit_with_kills', fit & (kills > 0)), ('fit_all', fit)):
        nominal = sum(nominal_row_count(int(t)) for t in T[m])
        actual = int(rows[m].sum())
        out[name] = {'matches': int(m.sum()), 'nominal': nominal, 'actual': actual, 'frame_late': nominal - actual}
    return out


def fit_frame_rows(s: Store) -> int:
    """Stored frame-clock rows of the FIT matches that have kills (the prototype's set).

    킬이 있는 FIT의 실제 프레임 행 수를 골라 반환한다(S1–S2 판정 24a).
    """
    return row_counts(s)['fit_with_kills']['actual']


def onset_count_share(s: Store, sel: np.ndarray, n_bins: int = 6) -> dict:
    """The share of rows with 0, 1, 2 and 3 or more onsets in the first ``n_bins`` bins (6 bins = the 60 s of the prototype, 12 = 120 s,
    18 = 180 s), over the frame rows of the selected matches. Keys '0', '1', '2', '3+' (the form of ``g2_anchor_check``) and ``rows``.

    수식
    ----
    (REP1-8)  count_row = Σ_ev 1[onset_row_ev=row ∧ onset_bin_ev<n_bins],
             hist_cat = #{선택 row : min(count_row,3)=cat},
             share_cat = hist_cat/selected_rows, mean_count = Σ_{선택 row} count_row/selected_rows.
             selected_rows = 0이면 비율과 평균은 None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, ev, cat | 프레임 행·발생 행·개수 범주 | — | 0 기반, cat=0,1,2,3(3+) | onsets/row, onsets/bin, hist
      count_row, n_bins | 앞 구간 발생 수와 구간 상한 | 개; — | 비음수 구간 계약, 기본 6 | count, first, n_bins
      selected_rows, hist_cat, share_cat, mean_count | 선택 행 수·빈도·비율·평균 | 행; 행; —; 개/행 | 분모 0이면 None | row_sel, out

    해석
    ----
    발생 수는 먼저 전체 저장 행에 세고 선택 경기의 행만 남기며, 비율의 분모는 경기 수가 아닌 선택 행 수이다(S1–S2 판정 25).
    많은 발생은 마지막 범주로 묶되 평균은 자르지 않은 개수로 계산한다.
    구간의 비음수성과 유효한 행 참조는 S1 발생 표의 계약이다.
    FIT 집계의 기본 집단은 킬이 있는 FIT이며 요청하면 모든 FIT도 허용한다(S1–S2 Task8b).

    근거
    ----
    S1–S2 Task8b 판정 25(발생 분포); S1–S2 계획 Task5; fit_onset_count_share.
    """
    row_sel = sel[_col(s, 'frame_rows/match')]
    n_rows = len(row_sel)
    first = _col(s, 'onsets/bin') < n_bins
    count = np.bincount(_col(s, 'onsets/row')[first], minlength=n_rows)[row_sel]
    hist = np.bincount(np.minimum(count, 3), minlength=4)
    out = {('3+' if k == 3 else str(k)): _ratio(hist[k], len(count)) for k in range(4)}
    out['rows'] = int(len(count))
    out['mean'] = float(count.mean()) if len(count) else None
    return out


def fit_onset_count_share(s: Store, n_bins: int = 6, kills_only: bool = True) -> dict:
    """``onset_count_share`` over the FIT rows: of the matches with at least one kill (the prototype's set, default) or of all of them.

    킬 유무 선택을 반영한 FIT 분모로 (REP1-8)을 호출한다(S1–S2 판정 25).
    """
    sel = fit_mask(s)
    if kills_only:
        sel = sel & (np.diff(_off(s, KILL_TABLE)) > 0)
    return onset_count_share(s, sel, n_bins)


# ---- accounting identity --------------------------------------------------------------------------------------------------------
def accounting(s: Store, sel: np.ndarray) -> dict:
    """The accounting identity over the objectives with a known kind and taker (kind >= 0, taker >= 0) of the selected matches.

    ``unit`` (attributed, objectives/unit >= 0), ``conversion`` (conv_unit >= 0), ``rest`` (neither) and ``both`` (attributed and
    converted at once; the identity needs it to be 0); ``all`` is the count. ``by_kind`` has the same entries per kind of
    ``config.O_KINDS``.


    수식
    ----
    (REP1-9)  valid = 선택 경기 ∧ kind≥0 ∧ taker≥0,
             attr_count = #{valid ∧ unit≥0}, conv_count = #{valid ∧ conv_unit≥0},
             rest_count = #{valid ∧ unit<0 ∧ conv_unit<0}, both_count = #{valid ∧ unit≥0 ∧ conv_unit≥0},
             attr_count + conv_count + rest_count = all_count + both_count.
             고유 귀속 계약이면 both_count = 0이고 왼쪽 합은 all_count이다.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      valid, kind, taker, unit, conv_unit | 집계 표시·종류·가져간 팀·단위 참조 | — | kind/taker 음수 제외, 참조는 0 기반·음수 없음 표시 | objectives 열
      attr_count, conv_count, rest_count, both_count, all_count | 귀속·전환·나머지·동시·전체 수 | 개 | 비음수 정수, 전체/종류별 | entry, by_kind

    해석
    ----
    이 함수는 개수를 보고하고 동시 귀속을 바로 거부하지 않으며, 회계 대조 단계가 회계와 저장 단위 합을 검사한다(S1–S2 Task8b).
    귀속과 전환이 겹치면 왼쪽 합이 그만큼 커지는 항등식이므로 동시 개수도 함께 남긴다.
    유효한 사건이 없으면 각 개수는 0이다.

    근거
    ----
    설계서 §3.3 D4; S1–S2 Task8b 회계 항; cmp_accounting(교차 검사).
    """
    om = _row_match(s, OBJ_TABLE)
    kind, taker = _col(s, 'objectives/kind'), _col(s, 'objectives/taker')
    valid = sel[om] & (kind >= 0) & (taker >= 0)
    unit, conv = _col(s, 'objectives/unit') >= 0, _col(s, 'objectives/conv_unit') >= 0

    def entry(m):
        return {'all': int(m.sum()), 'unit': int((m & unit).sum()), 'conversion': int((m & conv).sum()),
                'rest': int((m & ~unit & ~conv).sum()), 'both_unit_and_conversion': int((m & unit & conv).sum())}
    out = entry(valid)
    out['by_kind'] = {name: entry(valid & (kind == k)) for k, name in enumerate(C.O_KINDS)}
    return out


def fit_accounting(s: Store) -> dict:
    """``accounting`` over the FIT matches.

    FIT 선택으로 (REP1-9)의 개수 회계를 제공한다(S1–S2 Task8b).
    """
    return accounting(s, fit_mask(s))


# ---- rows of the frame clock that see an engagement ----------------------------------------------------------------------------
def rows_inside(s: Store, sel: np.ndarray, start: np.ndarray, end: np.ndarray, unit_mask: np.ndarray, end_closed: bool = False) -> dict:
    """The share of the frame rows of the selected matches whose t0 lies in ``[start, end)`` (``[start, end]`` when ``end_closed``) of
    some unit of the same match, over the units marked in ``unit_mask``. ``start`` and ``end`` are per-unit arrays (ms) with
    start <= end. Counted as (#start <= t0) - (#end <= t0) in one sorted array of keys match * BIG + time. Returns ``rows`` and
    ``inside`` (counts) and ``share``.

    수식
    ----
    (REP1-10)  begun_row = #{동일 경기의 선택 단위 : start_ms≤t0_ms},
             ended_row = #{동일 경기의 선택 단위 : end_ms≤t0_ms} (end_closed=False),
             ended_row = #{동일 경기의 선택 단위 : end_ms<t0_ms} (end_closed=True),
             inside_count = #{선택 row : begun_row−ended_row>0}, share = inside_count/selected_rows,
             key = match_row · BIG + time_ms, BIG = 2^31; selected_rows=0이면 share=None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      row, match_row, time_ms, key, BIG | 프레임 행·경기 행·시각·복합 키·간격 | —; ms | 행은 0 기반, int64 키의 비충돌·범위 계약 | rm, um, key, BIG
      start_ms, end_ms, t0_ms | 단위 창의 시작·끝과 질문 시각 | ms | start≤end 계약 | start, end, frame_rows/t0
      begun_row, ended_row, inside_count, selected_rows, share | 누적 시작·끝 개수·포함 행·전체 행·비율 | 개; 행; — | 비음수 개수 계약, 비율 None 가능 | begun, ended, inside, row_sel
      end_closed | 끝점 포함 선택 | — | bool | end_closed

    해석
    ----
    정렬된 시작·끝 키의 검색 개수 차로 어떤 단위 창에 들어 있는 행인지 계산한다(S1–S2 Task8의 진행 중 비교).
    시작은 항상 포함하고 끝은 호출자가 지정한 두 방식을 모두 허용한다.
    같은 시각의 빈 창은 반열림에서는 포함되지 않고 닫힘에서는 해당 시각을 포함하며, 유효 창·정수 키의 안전성은 직접 검사하지 않는다.
    여러 단위가 겹쳐도 행은 한 번만 세고 빈 행 집합의 비율은 None이다.

    근거
    ----
    S1–S2 계획 Task8; rows_inside(끝점 선택·입력 계약).
    """
    um = _col(s, 'units/match').astype(np.int64)
    keep = sel[um] & unit_mask
    rm = _col(s, 'frame_rows/match').astype(np.int64)
    row_sel = sel[rm]
    key = rm[row_sel] * BIG + _col(s, 'frame_rows/t0')[row_sel]
    # 식 (REP1-10)의 시작/끝 검색 방향이 닫힌 끝과 반열린 끝을 구분한다(rows_inside).
    begun = np.searchsorted(np.sort(um[keep] * BIG + start[keep]), key, side='right')
    ended = np.searchsorted(np.sort(um[keep] * BIG + end[keep]), key, side='left' if end_closed else 'right')
    inside = int(((begun - ended) > 0).sum())
    return {'rows': int(row_sel.sum()), 'inside': inside, 'share': _ratio(inside, row_sel.sum())}


def in_progress_stats(s: Store, sel: np.ndarray) -> dict:
    """Shares of the frame rows of the selected matches that see an engagement, under four readings (the prototype's first three are
    ``g2_anchor_check.json`` ``share_rows_t0_inside_ongoing_unit_kills`` / ``share_rows_t0_in_started_no_kill_yet`` and
    ``link_g15_check.json`` ``g2.t0_inside``; they use nominal minutes and in-scope units):

    ``ongoing_kills``    a unit in scope with f <= t0 <= last kill
    ``started_no_kill``  a unit in scope with tau <= t0 < f
    ``inside_window``    a unit in scope with tau <= t0 < delta
    ``in_progress``      a unit with f <= t0 < delta (the design's reading: first kill seen, engagement not over; no occurrence of the row);
                         ``in_progress_in_scope`` counts the units in scope only


    수식
    ----
    (REP1-11)  ongoing_kills: [first_ms,last_ms], 범위 안 단위;
             started_no_kill: [start_ms,first_ms), 범위 안 단위;
             inside_window: [start_ms,end_ms), 범위 안 단위;
             in_progress: [first_ms,end_ms), 모든 단위;
             in_progress_in_scope: [first_ms,end_ms), 범위 안 단위.
             각 비율 = (REP1-10)(해당 창·단위 집합).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      first_ms, last_ms, start_ms, end_ms | 첫·마지막 킬·시작·끝 | ms | 유효 저장 단위 계약 | f, last, tau, delta
      ongoing_kills, started_no_kill, inside_window, in_progress, in_progress_in_scope | 다섯 보고 필드 | — | (REP1-10)의 rows/inside/share | 반환 사전
      범위 안 단위, 모든 단위 | 범위 필터와 그 필터 없는 집합 | — | in_scope; everyone | scope, everyone

    해석
    ----
    시제품의 세 해석과 설계서의 진행 중 해석을 서로 다른 필드로 남긴다(S1–S2 Task8).
    설계서 해석은 첫 킬을 이미 본 뒤 교전이 끝나기 전이며, 모든 단위와 범위 안 단위의 집단을 둘 다 보고한다(설계서 §3.1·§4).
    빈 질문 행의 비율과 끝점 처리는 구간 포함 판정의 각 호출 규약을 따른다.

    근거
    ----
    S1–S2 계획 Task8; 설계서 §3.1·§4; in_progress_stats(다섯 호출).
    """
    f, last, tau, delta = (_col(s, f'units/{c}') for c in ('f', 'last', 'tau', 'delta'))
    scope = _col(s, 'units/in_scope')
    everyone = np.ones_like(scope)
    return {'ongoing_kills': rows_inside(s, sel, f, last, scope, end_closed=True), 'started_no_kill': rows_inside(s, sel, tau, f, scope),
            'inside_window': rows_inside(s, sel, tau, delta, scope), 'in_progress': rows_inside(s, sel, f, delta, everyone),
            'in_progress_in_scope': rows_inside(s, sel, f, delta, scope)}


def rows_per_unit(s: Store, sel: np.ndarray) -> dict:
    """How many frame rows each in-scope unit of the selected matches appears in (as an onset): the counts of units with 0, 1, 2 and 3 or
    more rows ('3+'), their shares, and the largest number of rows of one unit.

    수식
    ----
    (REP1-12)  selected_unit_set = {unit : 선택 경기 ∧ in_scope_unit},
             count_unit = #{ev : onset_unit_ev=unit}, selected_units = #selected_unit_set,
             hist_cat = #{unit∈selected_unit_set : min(count_unit,3)=cat},
             share_cat = hist_cat/selected_units, max_rows = max_{unit∈selected_unit_set}(count_unit).
             selected_units=0이면 share_cat=None, max_rows=0.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      selected_unit_set, in_scope_unit | 선택 경기의 범위 안 단위 집합과 범위 표시 | — | 유효 단위 행 집합; bool | keep, units/in_scope
      unit, ev, cat | 단위·발생 행·출현 범주 | — | 0 기반, cat=0..3(3+) | units/match, onsets/unit, hist
      count_unit, hist_cat, selected_units, share_cat, max_rows | 발생 행 수·빈도·분모·비율·최대 | 행; 개; 개; —; 행 | 비음수, None 가능 | count, keep, 반환 필드

    해석
    ----
    교전 창의 진행 중 행 수가 아니라 발생 표에 그 단위가 등장한 횟수를 센다(S1–S2 Task8).
    범위 안 단위 가운데 한 번도 등장하지 않은 단위도 첫 범주의 분모에 포함하고, 빈 단위 집합은 누락 비율로 남긴다.

    근거
    ----
    S1–S2 계획 Task8(단위별 출현); rows_per_unit.
    """
    um = _col(s, 'units/match')
    keep = sel[um] & _col(s, 'units/in_scope')
    count = np.bincount(_col(s, 'onsets/unit'), minlength=len(um))[keep]
    hist = np.bincount(np.minimum(count, 3), minlength=4)
    return {'units': int(keep.sum()), 'count': {('3+' if k == 3 else str(k)): int(hist[k]) for k in range(4)},
            'share': {('3+' if k == 3 else str(k)): _ratio(hist[k], keep.sum()) for k in range(4)},
            'max_rows': int(count.max()) if count.size else 0}


# ---- comparisons with the prototype (golden tests and the report) ----------------------------------------------------------------
def _py(x):
    """JSON-ready copy: numpy scalars and arrays become Python numbers and lists, tuples lists, dict values and list items recursively."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _check(ours, theirs, tol: float = 0.0) -> dict:
    """One comparison {ours, prototype, ok}: exact equality, or an absolute difference of at most ``tol`` for numbers.

    식 (REP1-23)의 해당 항을 계산한다.
    """
    ours, theirs = _py(ours), _py(theirs)
    ok = ours == theirs if tol == 0 else abs(ours - theirs) <= tol
    return {'ours': ours, 'prototype': theirs, 'ok': bool(ok)}


def _same_array(a: np.ndarray, b: np.ndarray, dtype: bool = False) -> dict:
    """An array comparison that does not put the arrays into the result: their lengths and whether they are equal (NaN equals NaN in a
    float array; with ``dtype`` the dtypes must agree too).

    식 (REP1-23)의 해당 항을 계산한다.
    """
    a, b = np.asarray(a), np.asarray(b)
    ok = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f' and b.dtype.kind == 'f'))
    return {'ours': int(a.shape[0]), 'prototype': int(b.shape[0]), 'ok': bool(ok and (not dtype or a.dtype == b.dtype))}


def _result(checks: dict, **info) -> dict:
    """식 (REP1-23)의 해당 항을 계산한다."""
    return {'ok': all(c['ok'] for c in checks.values()), 'checks': checks, **info}


def cmp_kills(s: Store, ref: dict) -> dict:
    """The kills of the FIT matches against fit_kills.npz: the match order, t, x, y, killer, victim, amask (equal, dtypes too), T_ms,
    teams and patch of the matches.

    FIT 킬 배열의 모양·dtype·값을 시제품과 대조한다(S1–S2 Task8b).
    """
    z = ref['npz']
    n, fit = len(fit_mask(s)), fit_matches(s)
    rank = np.full(n, -1, dtype=np.int64)
    rank[fit] = np.arange(len(fit))
    kill_match = _row_match(s, KILL_TABLE)
    checks = {'match_order': _same_array(rank[kill_match[fit_mask(s)[kill_match]]], z['mi'])}
    for name in ('t', 'x', 'y', 'killer', 'victim', 'amask'):
        checks[name] = _same_array(fit_kills(s, name), z[name], dtype=True)
    checks['T_ms'] = _same_array(fit_T(s), z['T_ms'])
    checks['teams'] = _same_array(fit_team(s), z['teams'])
    checks['patch'] = _same_array(_col(s, 'matches/patch')[fit], z['patch'])
    return _result(checks, matches=int(len(fit)), kills=int(len(z['t'])))


def cmp_units(s: Store, link_units) -> dict:
    """The units of every FIT match against the prototype's ``link_units.link_components(t, x, y, 15000)`` run on the stored kills:
    the same lists of kills, in the same order. ``link_units`` is the prototype module.

    경기별 연결 단위를 시제품 함수로 다시 만들어 순서까지 대조한다(S1–S2 Task8b).
    """
    fit = fit_matches(s)
    bad = 0
    for m in fit:
        t, x, y = kill_txy(s, int(m))
        if unit_lists(s, int(m)) != link_units.link_components(t, x, y, 15000):
            bad += 1
    return _result({'matches_with_other_units': {'ours': bad, 'prototype': 0, 'ok': bad == 0}}, matches=int(len(fit)))


def cmp_unit_checks(s: Store, ref: dict) -> dict:
    """``unit_stats`` of the FIT matches against ``link_g15_check.json`` ``link_15s``: exact (the formulas are the prototype's), and the
    flags that the stored size, scope and spread columns agree with the recomputed definitions.

    단위 통계와 저장 열 재구성 표시를 시제품 기준과 대조한다(S1–S2 Task8b).
    """
    got, g = fit_unit_stats(s), ref['g15']
    checks = {k: _check(got[k], g[k]) for k in ('units_per_kill', 'share_single_kill', 'spread_gt_4264_share', 'spread_gt_4264_n',
                                                'spread_p99', 'spread_max', 'cohort_share_in_scope')}
    for flag in ('stored_in_scope_equals_prototype_rule', 'stored_times_equal_prototype_rule', 'stored_nb_nr_equal_people_rule',
                 'stored_size_equals_people_rule', 'stored_spread_equals_float32_of_float64'):
        checks[flag] = {'ours': got[flag], 'prototype': True, 'ok': got[flag] is True}
    return _result(checks, n_kills=got['n_kills'], n_units=got['n_units'], n_in_scope=got['n_in_scope'])


def cmp_attribution(s: Store, ref: dict) -> dict:
    """``attribution_stats`` of the FIT matches against the participation and attribution probes under the two explained deviations
    (``expected_attribution``): per kind and for all, n, the objectives attributed, the objectives without an active unit and their
    shares, exactly. Also that the probes agree on n and that no objective has kind -1.

    설명된 편차를 반영한 (REP1-4)의 기준과 귀속 개수·비율을 대조한다(S1–S2 판정 23).
    """
    got, exp = fit_attribution_stats(s), expected_attribution(ref['part'], ref['attr'])
    checks = {}
    for name in (*PROTO_KINDS, 'all'):
        g, e = got[name], exp[name]
        checks[f'{name}.n'] = _check(g['n'], e['n'])
        checks[f'{name}.overlap'] = _check(g['overlap'], e['overlap'])
        checks[f'{name}.no_active'] = _check(g['n'] - g['active'], e['no_active'])
        checks[f'{name}.overlap_share'] = _check(g['participant_overlap_with_active_unit'], e['participant_overlap_with_active_unit'])
        checks[f'{name}.no_active_share'] = _check(g['share_no_active_unit'], e['share_no_active_unit'])
    checks['other.n'] = _check(got['other']['n'], 0)
    checks['probes_agree_on_n'] = _check({k: ref['part'][k]['n'] for k in PROTO_KINDS},
                                         {k: ref['attr']['by_kind'][k]['n'] for k in PROTO_KINDS})
    checks['all.n_prototype'] = _check(ref['part']['all']['n'], ref['attr']['all']['n'])
    return _result(checks, deviations=ATTRIBUTION_DEVIATIONS)


def cmp_aftermath(s: Store, ref: dict) -> dict:
    """``aftermath_stats`` of the FIT matches against aftermath_probe.json: units with a winner, and the winner / loser / excess rates and
    the cumulative excess rounded to 4 decimals (the probe's own rounding), the shares of the total excess by 60 / 90 / 120 / 180 s.

    수식
    ----
    (REP1-13)  rounded_field = np.round(field, 4),
             share_horizon = cumulative_{horizon//10−1}/cumulative_last,
             horizon ∈ {60,90,120,180}.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      field, rounded_field | 전환율·초과율·누적 초과와 그 반올림 | 개/단위분 또는 개/단위 | (REP1-5) | got, rounded
      cumulative, cumulative_last | 누적 초과 배열과 마지막 값 | 개/단위 | 분모가 0 아님을 가정 | cum, cum[-1]
      horizon, share_horizon | 비교 지평과 전체 초과 중 비율 | s; — | 지정 지평, 부호 가능 | x, share_of_total_excess

    해석
    ----
    시제품이 저장한 자리수로 반올림하여 맞대며 누적 초과가 지평별로 차지하는 비율도 비교한다(S1–S2 Task8b aftermath).
    지평의 배열 위치는 끝점에 해당하므로 초를 구간 번호로 바꾼 뒤 한 칸 뺀다.
    마지막 누적값이 0인 경우를 별도 처리하지 않으며 비교 결과의 비유한 값도 이 경로에서 대체하지 않는다.

    근거
    ----
    S1–S2 Task8b Aftermath·Floats; (REP1-5); cmp_aftermath.
    """
    got, a = fit_aftermath_stats(s), ref['after']
    rounded = {k: np.round(np.array(got[k]), 4).tolist()
               for k in ('winner_rate', 'loser_rate', 'excess_rate', 'cumulative_excess_per_unit')}
    cum = np.array(got['cumulative_excess_per_unit'])
    checks = {'units_with_winner': _check(got['units_with_winner'], a['units_with_winner']),
              'units_with_winner_pinned': _check(got['units_with_winner'], UNITS_WITH_WINNER),
              'bins_s': _check(got['bins_s'], a['bins_s'][:-1]),
              'winner_rate': _check(rounded['winner_rate'], a['winner_rate_per_unit_minute']),
              'loser_rate': _check(rounded['loser_rate'], a['loser_rate_per_unit_minute']),
              'excess_rate': _check(rounded['excess_rate'], a['excess_rate_per_unit_minute']),
              'cumulative_excess_per_unit': _check(rounded['cumulative_excess_per_unit'], a['cumulative_excess_conversions_per_unit']),
              'share_of_total_excess': _check({str(x): float(cum[x // 10 - 1] / cum[-1]) for x in (60, 90, 120, 180)},
                                              a['share_of_total_excess_by_60_90_120_180s'])}
    return _result(checks)


def cmp_rows(s: Store) -> dict:
    """The row counts (ruling 24a) and the in-scope units against the prototype: FIT matches with kills 457,441 nominal-minute rows and
    457,274 actual-clock rows, all FIT matches 457,445 and 457,278, 574,285 FIT units in scope, and one M3 event row per such unit.

    명목·실제 행 수와 사건 보기 수를 각각 잠긴 기준과 대조한다(S1–S2 판정 24a).
    """
    rows = row_counts(s)
    checks = {f'{name}.{kind}': _check(rows[name][kind], ROW_PINS[name][kind]) for name in ROW_PINS for kind in ('nominal', 'actual')}
    checks['fit_units_in_scope'] = _check(fit_in_scope_units(s), FIT_IN_SCOPE_UNITS)
    m3 = (_col(s, 'event_rows/kind') == event_rows.KIND_M3) & fit_mask(s)[_col(s, 'units/match')[_col(s, 'event_rows/unit')]]
    checks['fit_m3_event_rows'] = _check(int(m3.sum()), FIT_IN_SCOPE_UNITS)
    return _result(checks, rows=rows)


def cmp_onsets(s: Store, ref: dict) -> dict:
    """The 60 s onset-count shares (0 / 1 / 2 / 3+, first 6 bins) of the FIT rows against ``g2_anchor_check.json`` ``first_kill`` (the
    first-kill anchor, nominal-minute rows) within ``ONSET_TOLERANCE`` (ruling 25), and the same shares over all FIT rows.

    앞 구간의 발생 분포를 절대 허용차로 대조하고 전체 FIT 분모 결과도 보관한다(S1–S2 판정 25).
    """
    got, want = fit_onset_count_share(s, 6), ref['anchor']['first_kill']['share_rows_by_count']
    checks = {k: _check(got[k], want[k], tol=ONSET_TOLERANCE) for k in ('0', '1', '2', '3+')}
    return _result(checks, rows=got['rows'], tolerance=ONSET_TOLERANCE, all_fit_rows=fit_onset_count_share(s, 6, kills_only=False))


def cmp_accounting(s: Store) -> dict:
    """The accounting identity: over the objectives with a known kind and taker, no objective is both attributed and converted, and
    attributed + converted + rest = all (in total and per kind) over the FIT matches and over all stored matches. The attributed and the
    converted counts must also equal what the units hold: ``units/O`` and ``units/conv`` summed over the units of the selection (per kind
    two cells each), a second path to the same counts, so that ``rest`` cannot absorb a mistake. And the cell counts of the whole store
    equal the cells of the attributed / converted objectives.

    수식
    ----
    (REP1-14)  cell_ev = 2·kind_ev + taker_ev,
             events_cell = #{valid_ev ∧ ref_ev≥0 ∧ cell_ev=cell},
             held_cell = Σ_unit units_target_{unit,cell},
             비교: events_cell = held_cell, both_count=0,
             attr_count+conv_count+rest_count=all_count.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ev, unit, cell, kind_ev, taker_ev | 사건·단위·종류팀 칸·종류·팀 코드 | — | 0 기반; 유효 kind; taker≥0 입력 계약 | kind, taker, cell
      valid_ev, ref_ev, units_target | 유효 사건·귀속/전환 참조·저장 칸 | —; 개 | ref≥0; target는 units/O 또는 units/conv | counted, column, target
      events_cell, held_cell | 사건으로 센 칸과 단위에서 합한 칸 | 개 | 비음수 int64 합 | np.bincount, held
      both_count, attr_count, conv_count, rest_count, all_count | 회계 개수 | 개 | (REP1-9) | acc

    해석
    ----
    나머지 개수만으로 오차를 숨기지 않도록 단위에 저장된 칸과 사건을 직접 센 칸을 따로 맞댄다(S1–S2 Task8b 회계).
    종류별 합과 전체 칸을 모두 검사하고, 동시 귀속 여부도 검사한다.
    가져간 팀 코드가 알려진 경우의 정상 범위는 저장소 계약이며 이 함수의 valid 선택은 음수 여부만 확인한다.

    근거
    ----
    S1–S2 Task8b 회계 항; 설계서 §3.3 D4; cmp_accounting.
    """
    checks = {}
    for scope, sel in (('fit', fit_mask(s)), ('all', np.ones(len(fit_mask(s)), dtype=bool))):
        acc = accounting(s, sel)
        in_sel = sel[_col(s, 'units/match')]
        held = {'unit': _col(s, 'units/O')[in_sel].sum(axis=0, dtype=np.int64),
                'conversion': _col(s, 'units/conv')[in_sel].sum(axis=0, dtype=np.int64)}
        for name, a in (('total', acc), *acc['by_kind'].items()):
            checks[f'{scope}.{name}.both_zero'] = _check(a['both_unit_and_conversion'], 0)
            checks[f'{scope}.{name}.identity'] = _check(a['unit'] + a['conversion'] + a['rest'], a['all'])
            for field, cells in held.items():
                if name != 'total':
                    k = C.O_KINDS.index(name)
                    cells = cells[2 * k:2 * k + 2]
                checks[f'{scope}.{name}.{field}_equals_units'] = _check(a[field], int(cells.sum()))
    kind, taker = _col(s, 'objectives/kind').astype(np.int64), _col(s, 'objectives/taker').astype(np.int64)
    counted = (kind >= 0) & (kind < len(C.O_KINDS)) & (taker >= 0)
    cell = 2 * kind + taker
    for column, target in (('unit', 'units/O'), ('conv_unit', 'units/conv')):
        m = counted & (_col(s, f'objectives/{column}') >= 0)
        checks[f'{target}_cells'] = _check(np.bincount(cell[m], minlength=targets.N_CELLS), _col(s, target).sum(axis=0, dtype=np.int64))
    return _result(checks)


def cmp_golden(s: Store, golden: Path) -> dict:
    """The golden lock against the S1 run: the golden store (the first 20 FIT matches of 15.18, built by one worker in one block) holds
    exactly the rows of the first 20 matches of the S1 store, array by array. Needs ``<golden>/s1`` and ``GOLDEN.json``.

    기준 저장소 배열과 현재 저장소의 앞부분을 대조하며 결과에 실제 식별자를 넣지 않는다(S1–S2 Task8b Golden lock).
    """
    info = json.loads((Path(golden) / 'GOLDEN.json').read_text(encoding='utf-8'))
    g = Store(Path(golden) / run_s1.STORE_DIR)
    checks = {'matches_in_the_lock': _check(len(g.array('matches/key')), info['n_matches'])}
    for name in g.manifest['arrays']:
        a = _col(g, name)
        b = _col(s, name)[:len(a)]
        same = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'))
        checks[name] = {'ours': int(len(b)), 'prototype': int(len(a)), 'ok': same}
    g.maps.clear()
    return _result(checks, n_matches=info['n_matches'])


def compare(s: Store, ref: dict, link_units=None, golden: Path | None = None) -> dict:
    """Every comparison with the prototype, {name: result}; a result has ``ok``, ``checks`` ({check: {ours, prototype, ok}}) and some
    context. ``units`` needs the prototype module ``link_units`` and ``golden`` the golden folder; without them the entry is left out.

    수식
    ----
    (REP1-23)  number_ok = (ours=prototype) (tol=0),
             number_ok = (abs(ours−prototype)≤tol) (tol≠0),
             array_ok = (shape가 같음) ∧ array_equal(equal_nan=부동소수 배열),
             dtype=True이면 dtype도 같아야 한다; result_ok = all(check.ok).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ours, prototype, tol, number_ok | 보고 값·참조 값·절대 허용차·판정 | 원 값 단위; — | tol=0은 정확 비교 | _check
      array_ok, dtype, result_ok | 배열 비교·dtype 확인 선택·종합 비교 | — | bool | _same_array, _result
      shape, equal_nan | 배열 모양과 NaN 동등 비교 선택 | — | 모양 동일 필요, 부동소수에만 NaN 동등 | a.shape, a.dtype.kind

    해석
    ----
    각 비교는 같은 집단·경계·자리수 규칙으로 만든 결과를 기준값과 맞댄다(S1–S2 Task8b).
    공통 수치 허용차는 절대 차이며, 단위·뒤 전환·발생의 구체 비교 항목은 각 항목의 비교 단계에 따른다.
    외부 연결 함수나 기준 폴더가 없으면 해당 비교는 생략할 수 있으나 최종 보고 단계는 생략 표시를 남기고 전체 통과로 처리하지 않는다(S1–S2 Task8b).

    근거
    ----
    S1–S2 Task8b Golden-test·Floats; _check·_same_array·_result; report.
    """
    out = {'kills': cmp_kills(s, ref)}
    if link_units is not None:
        out['units'] = cmp_units(s, link_units)
    out.update(unit_checks=cmp_unit_checks(s, ref), attribution=cmp_attribution(s, ref), aftermath=cmp_aftermath(s, ref),
               rows=cmp_rows(s), onsets=cmp_onsets(s, ref), accounting=cmp_accounting(s))
    if golden is not None and (Path(golden) / 'GOLDEN.json').is_file():
        out['golden'] = cmp_golden(s, golden)
    return out


# ---- extras for the S1 gate -------------------------------------------------------------------------------------------------------
TIER_NAMES = ('outer', 'inner', 'base', 'nexus')     # objectives/sub 0-3 (targets: OUTER_TURRET ... NEXUS_TURRET); -1 is no tower tier
N_MONSTER_KINDS = 6                                  # config.O_KINDS[:6] are the monsters (dragon ... atakhan), the rest buildings, plates


def union_length(match: np.ndarray, start: np.ndarray, end: np.ndarray, n_matches: int) -> np.ndarray:
    """(n_matches,) float64: the total length in ms of the union of the intervals (start, end] of every match, intervals given in any
    order (empty ones, end <= start, are dropped). Sorted by (match, start); an interval adds the part beyond the largest end before it,
    found with a running maximum over keys match * BIG + end, which restarts by itself where the match changes.

    수식
    ----
    (REP1-15)  end_ms>start_ms인 창을 int64로 바꾸고 (match_row,start_ms)순으로 정렬한다.
             running_idx = max_{prior≤idx}(end_ms_prior + match_row_prior·BIG),
             before_idx = running_{idx−1} − match_row_idx·BIG,
             첫 before는 (int64_min//2) − match_row_0·BIG,
             covered_idx = max(0, end_ms_idx−max(start_ms_idx,before_idx)),
             union_ms_match = Σ_{idx: match_row_idx=match} covered_idx, BIG=2^31.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      idx, prior, match_row, match | 정렬 창 위치·이전 위치·경기 행 | — | 모두 0 기반 유효 행 | order, match
      start_ms_idx, end_ms_idx, running_idx, before_idx | 창 끝점·누적 최대 키·직전 끝점 표현 | ms 또는 키 | int64 연산 안전·비충돌 가정 | start, end, running, before
      BIG, int64_min | 경기 키 간격과 정수 최솟값 | — | 구현 상수 | BIG, np.iinfo(np.int64).min
      covered_idx, union_ms_match | 창이 더하는 길이와 경기별 합 | ms | ≥ 0, 반환 float64 | covered, np.bincount
      int64 | 정수 변환 연산 | — | 비정수 입력은 변환됨 | np.asarray(dtype=np.int64)

    해석
    ----
    정렬 뒤 이미 덮인 부분을 빼서 겹친 전환 창의 길이를 한 번만 센다(S1–S2 Task8b coverage).
    끝이 시작 이하인 창은 변환 전에 제외하고 남은 끝점은 int64로 변환한다.
    유효 경기 행·키 비충돌·정수 연산 범위는 가정이며 여기에서 검사하지 않는다.
    창이 없으면 minlength에 따른 경기별 0 길이 배열을 반환한다.

    근거
    ----
    S1–S2 Task8b Extras coverage; union_length(합집합 구현).
    """
    keep = end > start
    match, start, end = (np.asarray(a, dtype=np.int64)[keep] for a in (match, start, end))
    order = np.lexsort((start, match))
    match, start, end = match[order], start[order], end[order]
    base = match * BIG
    # 식 (REP1-15)의 직전 피복 끝은 정렬된 복합 키의 누적 최대로 표현한다(union_length).
    running = np.maximum.accumulate(end + base)
    before = np.concatenate(([np.iinfo(np.int64).min // 2], running[:-1])) - base     # largest end earlier in the same match
    covered = np.maximum(0, end - np.maximum(start, before))
    return np.bincount(match, weights=covered, minlength=n_matches)


def coverage_extras(s: Store) -> dict:
    """The conversion coverage: the fraction of game time from 120 s to T - 30 s that lies within 60 s after some delta, i.e. in the union
    of the conversion windows (delta, delta + 60 s]. ``all_units`` takes every unit's delta (the conversion rule: every unit counts),
    ``in_scope_units`` only those of units in scope. For all stored matches and for FIT: ``pooled`` (covered time over game time of the
    matches) and the distribution over matches (``per_match``; matches with no game time in the range are left out).

    Also the share of the valid objectives (known kind and taker) that is converted, by kind (``converted_share_of_valid``) and among those
    that no unit claims (``converted_share_of_unattributed``), and for those objectives that lie in 120 s .. T - 30 s the converted share
    of the unattributed ones (``in_range``) beside the pooled coverage, as a baseline. Interpreting coverage as an expected conversion share requires independent, uniform event placement over the
    same time support and compatible pooling weights; the code does not test these assumptions.


    수식
    ----
    (REP1-16)  span_ms_u = max(T_ms_u−30000−120000,0),
             cover_ms_u = union_length(u, max(δ,120000), min(δ+60000,T_ms_u−30000)),
             pooled = Σ_{선택 u: span_ms_u>0} cover_ms_u / Σ_{선택 u: span_ms_u>0} span_ms_u,
             per_match_u = cover_ms_u/span_ms_u (span_ms_u>0),
             converted_share_of_valid = conv_count/all_count,
             converted_share_of_unattributed = conv_count/(all_count−attr_count),
             in_range_share = #{pool ∧ conv_unit≥0}/#pool,
             pool = 선택 경기 ∧ kind≥0 ∧ taker≥0 ∧ unit<0 ∧ 120000≤time_ms≤T_ms_u−30000.
             개수 비율의 분모가 0이면 None; pooled에는 분모 0 보정이 없다.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      u, δ, T_ms_u, time_ms | 경기 행·교전 끝·경기 끝·사건 시각 | —; ms | 0 기반 경기, 유효 시각 계약 | um, delta, T, t
      span_ms_u, cover_ms_u, per_match_u, pooled | 유효 기간·피복 길이·경기 비율·합산 비율 | ms; — | span>0 경기만, pooled는 빈 분모 가능 | span, covered, out
      conv_count, all_count, attr_count | 전환·전체·귀속 개수 | 개 | (REP1-9), 고유 귀속 가정 | acc
      pool, in_range_share | 구간 안 미귀속 사건과 그 전환 비율 | — | 양 끝 포함, 분모 0이면 None | pool, in_range
      converted_share_of_valid, converted_share_of_unattributed | 전체/미귀속 중 전환 비율 | — | 고유 귀속 계약 아래 [0,1] | share, of_rest

    해석
    ----
    모든 단위와 범위 안 단위를 각각 사용하고, 전체 경기와 FIT를 따로 보고한다(S1–S2 Task8b Extras).
    pooled는 기간 길이로 가중된 비율이며 기간이 없는 경기는 경기별 분포에서도 제외하지만 빈 pooled 분모는 따로 처리하지 않는다.
    미귀속 중 전환 비율은 고유 귀속을 가정한 개수 나눗셈이며 교집합을 다시 계산하지 않는다(설계서 §3.3 D4).
    시간 피복을 무작위 기준선으로 읽으려면 같은 기간의 독립·균등 배치와 일관된 합산 가중 같은 추가 가정이 필요하고 코드가 그 가정을 검증하지 않는다.
    사건의 in_range는 두 끝점을 포함하며, 경기별 분포 요약은 (REP1-20)의 요약 규약을 쓴다.

    근거
    ----
    S1–S2 Task8b Extras coverage; 설계서 §3.3 D4; coverage_extras; (REP1-15)·(REP1-20).
    """
    n = len(_col(s, 'matches/T_ms'))
    T = _col(s, 'matches/T_ms').astype(np.int64)
    um, delta, scope = _col(s, 'units/match').astype(np.int64), _col(s, 'units/delta'), _col(s, 'units/in_scope')
    lo, hi = C.SCOPE_FIRST_MS, T - C.END_CUT_MS
    span = np.maximum(hi - lo, 0)
    groups = {'all': np.ones(n, dtype=bool), 'fit': fit_mask(s)}
    out = {'game_time': '120 s .. T - 30 s', 'window': '(delta, delta + 60 s]'}
    for name, units_used in (('all_units', np.ones_like(scope)), ('in_scope_units', scope)):
        m, d = um[units_used], delta[units_used]
        covered = union_length(m, np.maximum(d, lo), np.minimum(d + C.CONV_MS, hi[m]), n)
        out[name] = {}
        for group, sel in groups.items():
            g = sel & (span > 0)
            # 식 (REP1-16)의 pooled는 기간 길이 가중이며 빈 분모 보정은 없다(coverage_extras).
            out[name][group] = {'matches': int(g.sum()), 'pooled': float(covered[g].sum() / span[g].sum()),
                                'per_match': _dist(covered[g] / span[g])}
    om = _row_match(s, OBJ_TABLE)
    kind, taker = _col(s, 'objectives/kind'), _col(s, 'objectives/taker')
    unit, conv, t = _col(s, 'objectives/unit'), _col(s, 'objectives/conv_unit'), _col(s, 'objectives/t')
    for group, sel in groups.items():
        acc = accounting(s, sel)
        share = {name: _ratio(a['conversion'], a['all']) for name, a in (('total', acc), *acc['by_kind'].items())}
        of_rest = {name: _ratio(a['conversion'], a['all'] - a['unit']) for name, a in (('total', acc), *acc['by_kind'].items())}
        pool = sel[om] & (kind >= 0) & (taker >= 0) & (unit < 0) & (t >= lo) & (t <= hi[om])
        out.setdefault('converted_share_of_valid', {})[group] = share
        out.setdefault('converted_share_of_unattributed', {})[group] = of_rest
        out.setdefault('in_range', {})[group] = {
            'unattributed_valid_objectives': int(pool.sum()), 'converted': int((pool & (conv >= 0)).sum()),
            'converted_share': _ratio((pool & (conv >= 0)).sum(), pool.sum()),
            'pooled_coverage_all_units': out['all_units'][group]['pooled']}
    return out


def team_blind_extras(s: Store, sel: np.ndarray) -> dict:
    """How team-blind the attribution is. An objective is attributed to a unit when a credited champion (the killer, and the assisters of a
    monster or building) is among the people of the unit; the rule does not ask for that champion to be on the team that got the objective.
    Among the attributed objectives of the selected matches, by kind and for the monsters (dragon ... atakhan) and for the buildings and
    plates together:

    ``overlap_only_non_taker_team``     the overlap (credited champions that are people of the unit) holds no player of the taker team
    ``overlap_includes_non_taker_team`` the overlap holds at least one player of the other team
    ``attributed_with_empty_overlap``   attributed objectives without any overlap (the rule makes this 0)

    and, for the monsters with a known taker, the share of the credited champions that play for the other team.


    수식
    ----
    (REP1-17)  overlap_ev = cred_ev ∧ people_{max(unit_ev,0)},
             attributed_ev = 선택 경기 ∧ unit_ev≥0,
             only_count = #{attributed_ev ∧ (overlap_ev ∧ taker_mask_ev)=0},
             some_count = #{attributed_ev ∧ (overlap_ev ∧ other_mask_ev)≠0},
             empty_count = #{attributed_ev ∧ overlap_ev=0},
             only_share = only_count/attributed_count, some_share = some_count/attributed_count,
             credit_share = Σ_{credited monster} popcount(cred_ev ∧ other_mask_ev) /
                            Σ_{credited monster} popcount(cred_ev).
             분모 0의 비율은 None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      ev, unit_ev, cred_ev, people | 오브젝트 행·단위 참조·크레딧·단위 인원 마스크 | — | 0 기반 행, 음수 참조는 미귀속 | objectives/unit, cred, _unit_people
      taker_mask_ev, other_mask_ev | 가져간 팀과 반대 팀의 참가자 마스크 | — | taker=0이면 파랑, 그 밖에는 빨강으로 구성 | taker_team, other_team
      attributed_ev, attributed_count, only_count, some_count, empty_count | 선택 귀속 사건·분모·겹침 조건별 개수 | 개 또는 bool | 종류/몬스터/건물별 | entry
      only_share, some_share, credit_share | 비율 | — | 유효 입력에서 [0,1], None 가능 | out
      credited monster | 선택 경기의 알려진 taker 몬스터 | — | kind는 앞 몬스터 종류, taker≥0 | credited

    해석
    ----
    귀속 규칙이 크레딧 선수와 단위 인원의 겹침에 팀 조건을 더하지 않는다는 점을 진단한다(S1–S2 Task8b team-blind).
    가져간 팀과 겹치지 않는 항목은 그 팀과의 겹침이 없는 사건을 세므로 빈 겹침도 포함하고, 빈 겹침 수를 따로 보고한다.
    팀 마스크 구성에서 알려지지 않은 taker도 빨강 쪽으로 매핑되며, 모든 종류별 집계에서 이를 명시적으로 제거하는 것은 아니다.
    몬스터 크레딧 비율의 분모는 사건 수나 고유 선수 수가 아니라 사건마다 크레딧을 받은 참가자 출현 횟수의 합이다.
    미귀속 행도 인원 배열을 먼저 인덱싱하므로 유효한 단위 배열 등 저장소 계약이 필요하다.

    근거
    ----
    설계서 §3.2(참여 겹침); S1–S2 Task8b Extras team-blind; team_blind_extras.
    """
    om = _row_match(s, OBJ_TABLE)
    kind, taker = _col(s, 'objectives/kind').astype(np.int64), _col(s, 'objectives/taker')
    unit, cred = _col(s, 'objectives/unit').astype(np.int64), _col(s, 'objectives/cred').astype(np.int64)
    blue, red = _team_masks(s)
    taker_team = np.where(taker == 0, blue[om], red[om])        # the players of the team that got the objective (taker -1: red, unused)
    other_team = np.where(taker == 0, red[om], blue[om])
    # 식 (REP1-17)은 가져간 팀을 귀속 조건에 넣지 않은 규칙을 진단한다(S1–S2 Task8b).
    overlap = cred & _unit_people(s)[np.maximum(unit, 0)]
    attributed = sel[om] & (unit >= 0)

    def entry(m):
        m = attributed & m
        n = int(m.sum())
        only, some = m & ((overlap & taker_team) == 0), m & ((overlap & other_team) != 0)
        return {'attributed': n, 'attributed_with_empty_overlap': int((m & (overlap == 0)).sum()),
                'overlap_only_non_taker_team': int(only.sum()), 'share_only_non_taker_team': _ratio(only.sum(), n),
                'overlap_includes_non_taker_team': int(some.sum()), 'share_includes_non_taker_team': _ratio(some.sum(), n)}
    monster = (kind >= 0) & (kind < N_MONSTER_KINDS)
    out = {'monsters': entry(monster), 'buildings_and_plates': entry(kind >= N_MONSTER_KINDS),
           'by_kind': {name: entry(kind == k) for k, name in enumerate(C.O_KINDS)}}
    credited = sel[om] & monster & (taker >= 0)
    out['monster_credit'] = {'objectives': int(credited.sum()), 'credited_champions': int(_POPCOUNT[cred[credited]].sum()),
                             'on_non_taker_team': int(_POPCOUNT[cred[credited] & other_team[credited]].sum())}
    credit = out['monster_credit']
    credit['share_on_non_taker_team'] = _ratio(credit['on_non_taker_team'], credit['credited_champions'])
    return out


def count_extras(s: Store, sel: np.ndarray) -> dict:
    """Counts over the objectives of the selected matches: all of them (``objectives``), the tower tiers (``tower_tiers``: towers by
    ``objectives/sub`` outer / inner / base / nexus, and ``tower_without_tier``; ``tier_on_other_kinds`` counts a tier on anything but a
    tower and must be 0), the objectives with an unknown taker (``taker_unknown``, in total and by kind, killerTeamId not 100 or 200) and
    the objectives of kind -1 (``kind_unknown``).

    수식
    ----
    (REP1-18)  unknown_share = #{선택 사건 : taker<0}/selected_objectives,
             tower_tier_count_tier = #{선택 사건 : kind=tower ∧ sub=tier},
             tower_without_tier = #{선택 사건 : kind=tower ∧ sub<0},
             tier_on_other = #{선택 사건 : kind≠tower ∧ sub≥0},
             kind_unknown = #{선택 사건 : kind<0}.
             selected_objectives=0이면 unknown_share=None.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      kind, taker, sub, tier | 종류·가져간 팀·세부 단계·표시 단계 | — | 저장 코드, tier=0..3 | objectives 열, TIER_NAMES
      selected_objectives, unknown_share, tower_tier_count_tier, tower_without_tier, tier_on_other, kind_unknown | 선택 사건 수·비율·진단 개수 | 개; — | 비음수 개수, 비율 None 가능 | out, unknown, m

    해석
    ----
    포탑 단계·다른 종류의 단계 표시·음수 종류와 팀 코드를 보고하여 입력 상태를 드러낸다(S1–S2 Task8b counts).
    다른 종류의 단계 표시가 없어야 한다는 기대는 보고 항목이며 이 함수 자체가 오류를 던져 강제하지 않는다.

    근거
    ----
    S1–S2 Task8b Extras counts; count_extras.
    """
    om = _row_match(s, OBJ_TABLE)
    kind, taker, sub = _col(s, 'objectives/kind'), _col(s, 'objectives/taker'), _col(s, 'objectives/sub')
    m = sel[om]
    tower = C.O_KINDS.index('tower')
    out = {'objectives': int(m.sum()),
           'tower_tiers': {name: int((m & (kind == tower) & (sub == i)).sum()) for i, name in enumerate(TIER_NAMES)},
           'tower_without_tier': int((m & (kind == tower) & (sub < 0)).sum()),
           'tier_on_other_kinds': int((m & (kind != tower) & (sub >= 0)).sum()),
           'kind_unknown': int((m & (kind < 0)).sum())}
    unknown = m & (taker < 0)
    out['taker_unknown'] = {'total': int(unknown.sum()), 'share_of_objectives': _ratio(unknown.sum(), m.sum()),
                            'by_kind': {name: int((unknown & (kind == k)).sum()) for k, name in enumerate(C.O_KINDS)}}
    return out


def lateness_extras(s: Store, min_matches: int = 100) -> dict:
    """The lateness of the regular frames, ts[k] - 60,000 k in ms, by the frame index k: the median, the 99th percentile and the number
    of matches that have frame k, for every k with at least ``min_matches`` matches (frame 0 is at 0 ms by construction).

    수식
    ----
    (REP1-19)  frame_idx = global_frame_row − frame_off_match,
             late_ms = frame_ts − 60000·frame_idx,
             유지 frame_idx ⇔ count_frame_idx≥min_matches,
             median_ms = median(late_ms), p99_ms = Pct₉₉(late_ms), max_ms = max(late_ms)
                 (같은 frame_idx의 사용 가능한 경기 프레임에 대해).

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      frame_idx, global_frame_row, frame_off_match | 경기 안 위치·전체 행·경기 시작 오프셋 | — | 0 기반 | k, np.arange, off
      frame_ts, late_ms, median_ms, p99_ms, max_ms | 실제 시각·지연·요약 | ms | 유효 정규 프레임 계약, 지연 부호 가능 | frames/ts, late, groups
      count_frame_idx, min_matches | 해당 프레임이 있는 경기 수와 포함 하한 | 경기 | 기본 하한 100, 값 검증 없음 | counts, min_matches
      Pct | numpy 백분위 연산 | — | 선형 보간 기본값 | np.percentile

    해석
    ----
    경기 안 프레임 위치별로 명목 분 시각과의 차이를 계산한다(S1–S2 Task8b Figure8).
    긴 경기만 뒤 위치를 제공하므로 각 요약은 그 위치가 실제로 있는 경기들의 분포이며 같은 경기 집단을 고정한 시계열이 아니다.
    포함할 위치는 경기 수 하한으로 고르며 반환 배열이 비어도 대체 위치를 만들지 않는다.

    근거
    ----
    S1–S2 Task8b Figure8; 설계서 §4(프레임 시계); lateness_extras.
    """
    off = _off(s, 'frames')
    k = np.arange(int(off[-1]), dtype=np.int64) - np.repeat(off[:-1], np.diff(off))
    late = _col(s, 'frames/ts') - 60_000 * k
    order = np.argsort(k, kind='stable')
    counts = np.bincount(k)
    groups = np.split(late[order], np.cumsum(counts)[:-1])
    keep = [i for i in range(len(counts)) if counts[i] >= min_matches]
    return {'k': keep, 'matches': [int(counts[i]) for i in keep], 'median_ms': [float(np.median(groups[i])) for i in keep],
            'p99_ms': [float(np.percentile(groups[i], 99)) for i in keep], 'max_ms': [int(groups[i].max()) for i in keep],
            'min_matches': min_matches}


def age_extras(s: Store, sel: np.ndarray) -> dict:
    """The frame age (ms between the last regular frame and the reference time) of the event rows of the selected matches, for the M3
    rows (view at the first kill f) and the M4 rows (view at delta): the number of rows, the distribution, the rows whose frame is no
    frame-clock row (``frame_row`` -1) and the histogram in bins of 1 s up to 61 s (age >= 61 s counts in ``over_61s``).

    수식
    ----
    (REP1-20)  age_hist_slot = #{선택 사건 행 : min(age_ms//1000,61)=slot} (slot=0..60),
             overflow = #{age_ms≥61000}, missing_frame_count = #{frame_row<0}.
             dist(values): count=#values, mean=Σ value/count,
             std=√(Σ(value−mean)^2/count), percentiles=Pct_{0,1,5,25,50,75,95,99,100}(values).
             values는 float64 변환; 비면 dist는 {n:0}만 반환한다.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      age_ms, age_hist_slot, slot, overflow | 프레임 나이·초별 빈도·위치·넘침 수 | ms; 행; —; 행 | 나이 ≥ 0 계약, slot은 0 기반 | age, hist_1s, over_61s
      frame_row, missing_frame_count | 프레임 시계 참조와 음수 참조 수 | —; 행 | 저장 행 0 기반, 모든 음수 참조를 집계 | frame_row_minus_1
      values, value, count, mean, std, percentiles, Pct | 요약 표본과 크기·평균·모표준편차·백분위 | 입력 단위, count는 개 | 유한성 미검사; 빈 표본은 n만 | _dist, x, QUANTILES
      M3, M4 | 첫 킬과 교전 끝의 사건 보기 종류 | — | 별도 집계 | event_rows.KIND_M3, KIND_M4

    해석
    ----
    선택 경기의 두 사건 보기 종류를 따로 집계하여 정규 프레임으로부터 얼마나 오래되었는지 보고한다(S1–S2 Task8b Figure8).
    초별 표에는 마지막 초 미만까지 남기고 그 이상은 별도 넘침 수로 보관하므로 그림의 표시 빈도 합이 전체 행 수보다 작을 수 있다.
    frame_row_minus_1이라는 필드는 실제로 모든 음수 참조를 센다.
    공통 요약은 모집단 분모의 표준편차를 쓰고 비유한 값을 제거하지 않으며, 빈 표본에서 평균과 백분위 필드를 만들지 않는다.
    나이의 비음수성과 유효한 단위 참조는 저장소 계약이며 이 함수에서 검사하지 않는다.

    근거
    ----
    S1–S2 Task8b Figure8; S1–S2 계획 Task6; age_extras·_dist.
    """
    unit_match = _col(s, 'units/match')[_col(s, 'event_rows/unit')]
    keep = sel[unit_match]
    kind, age, frame_row = _col(s, 'event_rows/kind')[keep], _col(s, 'event_rows/age_ms')[keep], _col(s, 'event_rows/frame_row')[keep]
    out = {}
    for name, code in (('M3', event_rows.KIND_M3), ('M4', event_rows.KIND_M4)):
        a = age[kind == code]
        out[name] = {'rows': int(a.size), 'age_ms': _dist(a), 'frame_row_minus_1': int((frame_row[kind == code] < 0).sum()),
                     'hist_1s': np.bincount(np.minimum(a // 1000, 61), minlength=62)[:61].tolist(), 'over_61s': int((a >= 61_000).sum())}
    return out


# ---- the numbers behind the figures --------------------------------------------------------------------------------------------------
def collect(s: Store) -> dict:
    """Every number the figures show, computed once from the store: the unit statistics by patch (all stored matches of a patch) and for
    FIT; the onset-count shares of the FIT frame rows (matches with kills) for H = 60 / 120 / 180 s; attribution, aftermath and accounting
    for FIT and for all stored matches; the in-progress shares and rows per unit; the frame lateness and the frame ages.

    그림들이 공유할 집단별 통계를 한 번 계산하여 같은 수치를 재사용한다(S1–S2 Task8b).
    """
    n = len(fit_mask(s))
    everyone, fit = np.ones(n, dtype=bool), fit_mask(s)
    with_kills = fit & (np.diff(_off(s, KILL_TABLE)) > 0)
    groups = {p: unit_stats(s, patch_mask(s, p)) for p in run_s1.S1_PATCHES}
    groups['FIT'] = unit_stats(s, fit)
    return {
        'groups': groups,
        'onsets': {str(h // 1000): onset_count_share(s, with_kills, h // C.BIN_MS) for h in C.EVAL_H_MS},
        'onsets_all_stored_rows': {str(h // 1000): onset_count_share(s, everyone, h // C.BIN_MS) for h in C.EVAL_H_MS},
        'attribution': {'fit': attribution_stats(s, fit), 'all': attribution_stats(s, everyone)},
        'aftermath': {'fit': aftermath_stats(s, fit), 'all': aftermath_stats(s, everyone)},
        'accounting': {'fit': accounting(s, fit), 'all': accounting(s, everyone)},
        'in_progress': {'fit_rows_with_kills': in_progress_stats(s, with_kills), 'all': in_progress_stats(s, everyone)},
        'rows_per_unit': {'fit': rows_per_unit(s, fit), 'all': rows_per_unit(s, everyone)},
        'lateness': lateness_extras(s), 'ages': {'fit': age_extras(s, fit), 'all': age_extras(s, everyone)},
    }


# ---- figures -------------------------------------------------------------------------------------------------------------------------
FONT = 'Malgun Gothic'
BLUE, ORANGE, GREEN, PURPLE, SKY, GREY = '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#7F7F7F'
PATCH_COLORS = ('#9ECAE1', '#6BAED6', '#3182BD', '#08519C')                     # 15.18 ... 15.21
CLASS_COLORS = (BLUE, ORANGE, GREEN, PURPLE)


def _plt():
    """pyplot on the Agg backend with Malgun Gothic and the plain minus sign. RuntimeError when the font is missing: the Korean text would
    be boxes."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(f'the font {FONT} is not available; the figure text needs its Hangul glyphs') from exc
    plt.rcParams.update({'font.family': FONT, 'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 150, 'figure.dpi': 100})
    return plt


def _save(plt, fig, path: Path) -> None:
    """Save a figure as PNG. A warning about a glyph that the font lacks raises RuntimeError (the file is removed): no figure is kept with
    boxes in it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.savefig(path)
    plt.close(fig)
    bad = sorted({str(w.message) for w in caught if 'missing from font' in str(w.message) or 'Glyph' in str(w.message)})
    if bad:
        Path(path).unlink(missing_ok=True)
        raise RuntimeError(f'{Path(path).name}: {bad[:3]}')


def _annotate(ax, bars, fmt: str = '{:.3f}', size: int = 8) -> None:
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha='center', va='bottom', fontsize=size,
                    xytext=(0, 2), textcoords='offset points')


def _proto_marks(ax, xs, values, half_width: float, label: str) -> None:
    """Dashed marks at the prototype values, one short line across each bar position; the first one carries the legend entry."""
    for i, (x, v) in enumerate(zip(xs, values)):
        ax.hlines(v, x - half_width, x + half_width, colors='black', linestyles='--', linewidth=1.5, zorder=5,
                  label=label if i == 0 else None)


def fig_units(plt, groups: dict, g15: dict, path: Path) -> dict:
    """Figure 1: units per kill and the share of single-kill units, by patch and for FIT; dashed: the prototype's FIT values.

    (REP1-2)의 패치별·FIT 통계를 시제품 기준선과 함께 표시한다(S1–S2 Task8b Figure1).
    """
    names = list(groups)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    labels = [f'{n}\n{groups[n]["n_units"]:,} 단위' for n in names]
    for ax, key, title in ((axes[0], 'units_per_kill', '킬당 교전 단위 수 (단위 수 / 킬 수)'), (axes[1], 'share_single_kill', '킬 하나짜리 단위의 비율')):
        vals = [groups[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=[*PATCH_COLORS, ORANGE][:len(names)], width=0.62)
        _annotate(ax, bars)
        ax.axhline(g15[key], color='black', linestyle='--', linewidth=1.5, label=f'시제품 FIT 15.18-15.20: {g15[key]:.4f}')
        ax.set_xticks(range(len(names)), labels)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8.5, frameon=False)
    fig.suptitle('S1 교전 단위: 패치별 (한 패치의 저장된 모든 경기), FIT = 15.18-15.20의 FIT 경기', fontsize=11)
    _save(plt, fig, path)
    return {'groups': {n: {k: groups[n][k] for k in ('units_per_kill', 'share_single_kill', 'n_units', 'n_kills')} for n in names},
            'prototype': {k: g15[k] for k in ('units_per_kill', 'share_single_kill')}}


def fig_sizes(plt, groups: dict, g15: dict, path: Path) -> dict:
    """Figure 2: the shares of the size classes T / S / ASYM / P among the units in scope, by patch and for FIT; dashed: the prototype.

    (REP1-1)·(REP1-2)의 범위 안 단위 규모 비율을 표시한다(S1–S2 Task8b Figure2).
    """
    names = list(groups)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.8 / len(names)
    x = np.arange(len(SIZE_NAMES))
    for i, n in enumerate(names):
        vals = [groups[n]['cohort_share_in_scope'][c] for c in SIZE_NAMES]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width * 0.92, color=[*PATCH_COLORS, ORANGE][i],
                      label=f'{n} ({groups[n]["n_in_scope"]:,})')
        _annotate(ax, bars, '{:.3f}', 7)
    _proto_marks(ax, x, [g15['cohort_share_in_scope'][c] for c in SIZE_NAMES], 0.42, '시제품 FIT')
    ax.set_xticks(x, SIZE_NAMES)
    ax.set_ylabel('범위 안 단위 중 비율')
    ax.set_title('범위 안 교전 단위(120 s <= f <= T - 30 s)의 규모 분류 비율: T, S, ASYM, P', loc='left')
    ax.legend(title='막대: 패치 (범위 안 단위 수)', fontsize=8.5, frameon=False)
    _save(plt, fig, path)
    return {'groups': {n: {k: groups[n][k] for k in ('cohort_share_in_scope', 'n_in_scope')} for n in names},
            'prototype': g15['cohort_share_in_scope']}


def fig_onsets(plt, onsets: dict, anchor: dict, path: Path) -> dict:
    """Figure 3: the share of FIT rows with 0 / 1 / 2 / 3+ first kills in their window for H = 60, 120 and 180 s; dashed: the prototype's
    60 s shares (first-kill anchor, nominal-minute rows).

    (REP1-8)의 행 분포를 지평별로 표시하며 기준선은 해당 시제품 창에만 둔다(S1–S2 Task8b Figure3).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    cats = ('0', '1', '2', '3+')
    for ax, h in zip(axes, ('60', '120', '180')):
        vals = [onsets[h][c] for c in cats]
        bars = ax.bar(range(4), vals, color=CLASS_COLORS, width=0.62)
        _annotate(ax, bars)
        if h == '60':
            _proto_marks(ax, range(4), [anchor['first_kill']['share_rows_by_count'][c] for c in cats], 0.31, '시제품 g2_anchor_check')
            ax.legend(loc='upper right', fontsize=8.5, frameon=False)
        ax.set_xticks(range(4), cats)
        ax.set_xlabel('창 안의 첫 킬 수')
        ax.set_title(f'H = {h} s (평균 {onsets[h]["mean"]:.2f}개)')
    axes[0].set_ylabel('행 비율')
    fig.suptitle(f'행당 발생(첫 킬) 수 분포: FIT 프레임 행 {onsets["60"]["rows"]:,}개 (킬이 있는 경기)', fontsize=11)
    _save(plt, fig, path)
    return {'ours': onsets, 'prototype_60s': anchor['first_kill']['share_rows_by_count']}


def fig_attribution(plt, att: dict, part: dict, path: Path) -> dict:
    """Figure 4: the share of objectives (FIT, from 120 s, with a position) attributed to an engagement unit, and the share with an active
    unit, by kind; dashed: the prototype's participation probe.

    (REP1-3)의 귀속·활성 비율을 종류별 시제품 값과 함께 표시한다(S1–S2 Task8b Figure4).
    """
    names = (*PROTO_KINDS, 'all')
    fig, ax = plt.subplots(figsize=(11.5, 5), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.36
    own = [att[k]['participant_overlap_with_active_unit'] for k in names]
    act = [att[k]['active_unit'] for k in names]
    _annotate(ax, ax.bar(x - w / 2, own, w * 0.94, color=BLUE, label='교전 단위에 귀속 (참여자 겹침)'), '{:.3f}', 7.5)
    _annotate(ax, ax.bar(x + w / 2, act, w * 0.94, color=SKY, label='창 안에 진행 중인 단위 있음'), '{:.3f}', 7.5)
    ax.hlines([part[k]['participant_overlap_with_active_unit'] for k in names], x - w / 2 - w * 0.47, x - w / 2 + w * 0.47, colors='black',
              linestyles='--', linewidth=1.5, zorder=5, label='시제품 (점선)')
    ax.hlines([part[k]['active_unit'] for k in names], x + w / 2 - w * 0.47, x + w / 2 + w * 0.47, colors='black', linestyles='--',
              linewidth=1.5, zorder=5)
    ax.set_xticks(x, [f'{k}\n{att[k]["n"]:,}' for k in names])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel('오브젝트 중 비율')
    ax.set_title('오브젝트 귀속 비율, 종류별 (FIT, 120 s 이후, 위치 있음; 아래 숫자는 오브젝트 수)', loc='left')
    ax.legend(loc='upper left', fontsize=8.5, frameon=False, ncol=3)
    _save(plt, fig, path)
    fields = ('n', 'overlap', 'active', 'participant_overlap_with_active_unit', 'active_unit')
    return {'ours': {k: {f: att[k][f] for f in fields} for k in names},
            'prototype': {k: {'n': part[k]['n'], 'participant_overlap_with_active_unit': part[k]['participant_overlap_with_active_unit'],
                              'active_unit': part[k]['active_unit']} for k in names}}


def fig_conversion(plt, ours: dict, after: dict, path: Path) -> dict:
    """Figure 5: the conversion rate of the winner and of the loser of an engagement in the 240 s after delta (10 s bins, per unit-minute),
    their difference (the excess) and the cumulative excess per unit; dashed: the prototype's aftermath probe.

    (REP1-5)의 전환율은 구간 중앙, 누적 초과는 구간 오른쪽 끝에 표시한다(S1–S2 Task8b Figure5).
    """
    mid = np.array(ours['bins_s']) + 5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    for key, pkey, color, label in (('winner_rate', 'winner_rate_per_unit_minute', GREEN, '승자'),
                                    ('loser_rate', 'loser_rate_per_unit_minute', ORANGE, '패자'),
                                    ('excess_rate', 'excess_rate_per_unit_minute', BLUE, '초과 (승자 - 패자)')):
        ax.plot(mid, ours[key], color=color, linewidth=2, marker='o', markersize=3.5, label=label)
        ax.plot(mid, after[pkey], color='black', linestyle='--', linewidth=1.2)
    ax.plot([], [], color='black', linestyle='--', linewidth=1.2, label='시제품 (점선)')
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel('delta 이후 시간 (s), 10 s 구간의 가운데')
    ax.set_ylabel('단위당 분당 전환 수')
    ax.set_title(f'delta 이후 240 s의 전환율 (승자가 있는 단위 {ours["units_with_winner"]:,}개)', loc='left')
    ax.legend(fontsize=8.5, frameon=False)
    ax = axes[1]
    ax.plot(mid + 5, ours['cumulative_excess_per_unit'], color=BLUE, linewidth=2, marker='o', markersize=3.5, label='우리 S1 표')
    ax.plot(mid + 5, after['cumulative_excess_conversions_per_unit'], color='black', linestyle='--', linewidth=1.2, label='시제품')
    ax.set_xlabel('delta 이후 시간 (s)')
    ax.set_ylabel('단위당 누적 초과 전환 수')
    ax.set_title('누적 초과 전환 수', loc='left')
    ax.legend(fontsize=8.5, frameon=False)
    _save(plt, fig, path)
    keys = ('units_with_winner', 'bins_s', 'winner_rate', 'loser_rate', 'excess_rate', 'cumulative_excess_per_unit')
    return {'ours': {k: ours[k] for k in keys},
            'prototype': {k: after[k] for k in ('units_with_winner', 'winner_rate_per_unit_minute', 'loser_rate_per_unit_minute',
                                                'excess_rate_per_unit_minute', 'cumulative_excess_conversions_per_unit')}}


def fig_accounting(plt, acc: dict, path: Path, n_matches: dict) -> dict:
    """Figure 6: the accounting identity as a table: every objective with a known kind and taker is attributed to a unit, converted to one
    or in the rest, by kind; for all stored matches and for FIT.

    수식
    ----
    (REP1-21)  display_share_field = format(100·count_field/all_count, '.1f'),
             field ∈ {unit, conversion, rest}.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      count_field, all_count | 종류별 또는 전체의 부분 개수·분모 | 개 | all_count>0 가정, 보정 없음 | e[k], e['all']
      display_share_field, field | 표시 백분율 문자열과 필드 이름 | %; — | 소수 한 자리 | share, k

    해석
    ----
    회계 개수와 백분율을 같은 표에 표시한다(S1–S2 Task8b Figure6).
    표시 비율만 소수 한 자리로 서식화하고 반환 개수는 바꾸지 않으며, 분모가 0인 종류는 여기에서 처리하지 않는다.

    근거
    ----
    S1–S2 Task8b Figure6; (REP1-9); fig_accounting.
    """
    header = ['종류', '전체', '교전 귀속', '전환', '나머지', '귀속 %', '전환 %', '나머지 %', '동시']
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.6), constrained_layout=True)
    for ax, scope, title in ((axes[0], 'all', f'저장된 모든 경기 ({n_matches["all"]:,}개)'), (axes[1], 'fit', f'FIT 경기 ({n_matches["fit"]:,}개)')):
        a = acc[scope]
        rows = []
        for name, e in (*a['by_kind'].items(), ('합계', a)):
            share = [f'{100 * e[k] / e["all"]:.1f}' for k in ('unit', 'conversion', 'rest')]
            rows.append([name, f'{e["all"]:,}', f'{e["unit"]:,}', f'{e["conversion"]:,}', f'{e["rest"]:,}', *share,
                         f'{e["both_unit_and_conversion"]}'])
        ax.axis('off')
        ax.set_title(title, loc='left')
        tab = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='right', colLoc='right')
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.45)
        for (r, c), cell in tab.get_celld().items():
            if c == 0:
                cell.set_text_props(ha='left')
            if r == 0 or r == len(rows):
                cell.set_text_props(fontweight='bold')
    fig.suptitle('회계 항등식: 교전 귀속 + 전환 + 나머지 = 전체 (종류와 taker가 알려진 오브젝트, 동시 = 귀속이면서 전환)', fontsize=11)
    _save(plt, fig, path)
    return {scope: {'total': {k: acc[scope][k] for k in ('all', 'unit', 'conversion', 'rest', 'both_unit_and_conversion')},
                    'by_kind': acc[scope]['by_kind']} for scope in ('all', 'fit')}


def fig_rows(plt, in_progress: dict, rows_unit: dict, ref: dict, path: Path) -> dict:
    """Figure 7: left, the share of FIT frame rows that see an engagement under five readings (dashed: the prototype's three, nominal-minute
    rows); right, the number of rows each unit in scope appears in (0, 1, 2, 3+), for FIT and for all stored matches.

    (REP1-11)의 다섯 경계·집단과 (REP1-12)의 단위 출현 빈도를 함께 표시한다(S1–S2 Task8b Figure7).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={'width_ratios': [3, 2]}, constrained_layout=True)
    ax = axes[0]
    keys = ('ongoing_kills', 'started_no_kill', 'inside_window', 'in_progress', 'in_progress_in_scope')
    labels = ['ongoing\nf<=t0<=last\n(범위 안 단위)', 'started\ntau<=t0<f\n(범위 안 단위)', 'inside\ntau<=t0<delta\n(범위 안 단위)',
              '진행 중\nf<=t0<delta\n(모든 단위)', '진행 중\nf<=t0<delta\n(범위 안 단위)']
    vals = [in_progress[k]['share'] for k in keys]
    bars = ax.bar(range(5), vals, color=[GREY, GREY, GREY, BLUE, SKY], width=0.62)
    _annotate(ax, bars, '{:.4f}')
    proto = [ref['anchor']['share_rows_t0_inside_ongoing_unit_kills'], ref['anchor']['share_rows_t0_in_started_no_kill_yet'],
             ref['g15']['g2']['t0_inside']]
    _proto_marks(ax, range(3), proto, 0.31, '시제품 (g2_anchor_check, link_g15_check)')
    ax.set_xticks(range(5), labels, fontsize=8.5)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel('FIT 프레임 행 중 비율')
    ax.set_title(f'교전이 진행 중인 행의 비율 (FIT 프레임 행 {in_progress["in_progress"]["rows"]:,}개)', loc='left')
    ax.legend(loc='upper left', fontsize=8.5, frameon=False)
    ax = axes[1]
    cats = ('0', '1', '2', '3+')
    for i, (scope, color) in enumerate((('fit', ORANGE), ('all', BLUE))):
        v = [rows_unit[scope]['share'][c] for c in cats]
        name = 'FIT' if scope == 'fit' else '저장된 모든 경기'
        _annotate(ax, ax.bar(np.arange(4) + (i - 0.5) * 0.38, v, 0.36, color=color, label=f'{name} ({rows_unit[scope]["units"]:,})'),
                  '{:.4f}', 7.5)
    ax.set_xticks(range(4), cats)
    ax.set_xlabel('한 교전(단위)이 나오는 행 수')
    ax.set_ylabel('범위 안 단위 중 비율')
    ax.set_title('한 교전이 나오는 행 수', loc='left')
    ax.legend(fontsize=8.5, frameon=False)
    _save(plt, fig, path)
    return {'in_progress_fit_rows': in_progress,
            'prototype': {'ongoing_kills': proto[0], 'started_no_kill': proto[1], 'inside_window': proto[2]}, 'rows_per_unit': rows_unit}


def fig_frames(plt, late: dict, ages: dict, path: Path) -> dict:
    """Figure 8: left, the lateness of the regular frames ts[k] - 60,000 k (median and 99th percentile) by the frame index; right, the
    frame age of the M3 and M4 event rows (all stored matches), per 1 s.

    수식
    ----
    (REP1-22)  age_share_slot = age_hist_slot/event_rows_count,
             center_s_slot = slot+0.5, shown_quantile_s = quantile_ms/1000.

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      slot, age_hist_slot, age_share_slot | 초 구간 위치·빈도·표시 비율 | —; 행; — | slot=0..60 | centers, a['hist_1s'], h
      event_rows_count | 해당 보기 종류의 전체 행 수 | 행 | > 0 가정, 보정 없음 | a['rows']
      center_s_slot, quantile_ms, shown_quantile_s | 그림 중심·나이 백분위·초 표시 | s; ms; s | 나이 요약 필드가 있다고 가정 | centers, p50, p99

    해석
    ----
    나이 빈도는 넘침 행을 포함한 전체 사건 행 수로 나눠 표시하고 넘침을 표시 구간에 재분배하지 않는다(S1–S2 Task8b Figure8).
    중앙값과 백분위는 초 단위로 바꾸어 표시하지만 지연 곡선은 원래 ms 단위를 유지한다.
    비어 있는 종류나 누락된 요약 필드에 대한 별도 그림 경로는 없다.

    근거
    ----
    S1–S2 Task8b Figure8; (REP1-19)·(REP1-20); fig_frames.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(late['k'], late['median_ms'], color=BLUE, linewidth=2, label='중앙값')
    ax.plot(late['k'], late['p99_ms'], color=ORANGE, linewidth=2, label='99 백분위')
    ax.set_xlabel('프레임 번호 k')
    ax.set_ylabel('ts[k] - 60,000 k (ms)')
    ax.set_title(f'정규 프레임의 지연 (경기 {late["min_matches"]}개 이상이 가진 k)', loc='left')
    ax.legend(fontsize=8.5, frameon=False)
    ax = axes[1]
    centers = np.arange(61) + 0.5
    for name, color, label in (('M3', BLUE, 'M3: 첫 킬 f에서의 보기'), ('M4', ORANGE, 'M4: delta에서의 보기')):
        a = ages[name]
        h = np.array(a['hist_1s']) / a['rows']
        ax.step(centers, h, where='mid', color=color, linewidth=1.8,
                label=f'{label}  (중앙값 {a["age_ms"]["p50"] / 1000:.1f} s, 99 백분위 {a["age_ms"]["p99"] / 1000:.1f} s)')
    ax.set_xlabel('프레임 나이 = 기준 시각 - 직전 프레임 시각 (s)')
    ax.set_ylabel('사건 행 비율 (1 s 구간)')
    ax.set_title(f'사건 행의 프레임 나이 (M3 {ages["M3"]["rows"]:,}행, M4 {ages["M4"]["rows"]:,}행)', loc='left')
    ax.set_ylim(0, max(max(ages['M3']['hist_1s']) / ages['M3']['rows'], max(ages['M4']['hist_1s']) / ages['M4']['rows']) * 1.55)
    ax.legend(fontsize=8, frameon=False, loc='upper center')
    _save(plt, fig, path)
    fields = ('rows', 'age_ms', 'frame_row_minus_1', 'hist_1s', 'over_61s')
    return {'lateness': late, 'ages': {k: {f: ages[k][f] for f in fields} for k in ('M3', 'M4')}}


# ---- the report ----------------------------------------------------------------------------------------------------------------------
def load_link_units(folder: Path = PROTO_DIR):
    """The prototype module ``link_units.py`` (it has ``link_components``), loaded with bytecode writing off so that no ``__pycache__``
    appears in the read-only prototype folder. The module puts a second prototype folder on ``sys.path`` and imports ``g2_window_check``
    from it: both are undone. The same loader as ``tests/conftest.py::load_proto``, which keeps its own copy so that this module does not
    import from the tests. Raises whatever the import raises.

    시제품 연결 함수를 비교용으로 불러오며 이 모듈에서 연결 규칙을 재정의하지 않는다(S1–S2 Task8b).
    """
    flag, path = sys.dont_write_bytecode, list(sys.path)
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('proto_link_units', Path(folder) / 'link_units.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = flag
        sys.path[:] = path
        sys.modules.pop('g2_window_check', None)


def assert_no_keys(s: Store, text: str) -> None:
    """RuntimeError if the key of any stored match occurs in ``text`` (the report). Only the number of hits is named, never a key.

    보고 문구에서 저장 경기 식별자를 찾아 출력 전 거부한다(S1–S2 Task8b 개인정보 규칙).
    """
    hits = sum(1 for k in _col(s, 'matches/key').tolist() if k.decode('ascii') in text)
    if hits:
        raise RuntimeError(f'{hits} match keys occur in the report text')


def _golden_info(golden: Path) -> dict | None:
    """What the golden lock says about itself (no array hashes): the matches, plan hash, code state and the number of locked files."""
    path = Path(golden) / 'GOLDEN.json'
    if not path.is_file():
        return None
    info = json.loads(path.read_text(encoding='utf-8'))
    out = {k: info.get(k) for k in ('n_matches', 'plan_sha256', 'code_commit', 'extraction_clean', 'dirty_paths', 'created', 'rows')}
    out['locked_files'] = len(info.get('files', {}))
    out['golden_json_sha256'] = store_core.sha(path)
    return out


def gate_summary(run: dict, comparisons: dict, data: dict, extras: dict) -> dict:
    """The numbers the S1 gate looks at first, collected from the other parts of the report (nothing new is computed here): whether
    every comparison passed, the table sizes, the conversion coverage, the converted share of the objectives, the team-blind overlap, the
    tower tiers, the objectives with an unknown taker, the lateness of the frames, the frame age of the event rows and the rows that see
    an engagement. ``run`` is the ``run`` section, ``data`` the result of ``collect`` and ``extras`` the ``extras`` section.

    이미 계산한 관문 항목을 모으고 사건 나이의 백분위만 ms에서 초로 바꾼다(S1–S2 Task8b Extras).
    지정 프레임 위치와 요약 필드가 있는 것은 입력 계약이며 없을 때 대체값을 만들지 않는다(gate_summary).
    """
    cov, late, ages = extras['conversion_coverage'], extras['frame_lateness'], extras['event_row_ages']['all']
    blind, counts = extras['team_blind_overlap']['all']['monsters'], extras['counts']['all']
    at = {k: late['k'].index(k) for k in (2, 10, 30)}
    return {
        'comparisons_all_ok': comparisons['all_ok'],
        'tables': {'matches': run['matches'], 'units_in_scope': sum(data['groups'][p]['n_in_scope'] for p in run_s1.S1_PATCHES),
                   **{k: run['rows'][k] for k in ('kills', 'units', 'objectives', 'frame_rows', 'onsets', 'event_rows')}},
        'conversion_coverage': {
            'pooled_all_units': {g: cov['all_units'][g]['pooled'] for g in ('all', 'fit')},
            'pooled_in_scope_units': {g: cov['in_scope_units'][g]['pooled'] for g in ('all', 'fit')},
            'per_match_all_units_p5_p50_p95': [cov['all_units']['all']['per_match'][q] for q in ('p5', 'p50', 'p95')],
            'converted_share_of_valid_objectives': {g: cov['converted_share_of_valid'][g]['total'] for g in ('all', 'fit')},
            'converted_share_of_unattributed_objectives_in_range': {g: cov['in_range'][g]['converted_share'] for g in ('all', 'fit')}},
        'team_blind_overlap': {'attributed_monster_objectives': blind['attributed'],
                               'overlap_only_non_taker_team': blind['share_only_non_taker_team'],
                               'overlap_includes_non_taker_team': blind['share_includes_non_taker_team']},
        'tower_tiers': counts['tower_tiers'], 'objectives_with_taker_minus_1': counts['taker_unknown']['total'],
        'frame_lateness_ms': {str(k): {'median': late['median_ms'][i], 'p99': late['p99_ms'][i]} for k, i in at.items()},
        'event_row_age_s': {name: {'median': ages[name]['age_ms']['p50'] / 1000, 'p99': ages[name]['age_ms']['p99'] / 1000,
                                   'max': ages[name]['age_ms']['p100'] / 1000} for name in ('M3', 'M4')},
        'event_rows_without_a_frame_clock_row': {name: ages[name]['frame_row_minus_1'] for name in ('M3', 'M4')},
        'fit_rows_with_an_engagement_in_progress': data['in_progress']['fit_rows_with_kills']['in_progress']['share'],
        'fit_units_by_number_of_rows': data['rows_per_unit']['fit']['share'],
    }


def report(run, report_dir=None, json_path=None, golden=None, verify: bool = True) -> Path:
    r"""Write ``S1_REPORT.json`` and the figures of an S1 run folder and return the path of the JSON.

    ``report_dir`` (default ``<run>\report``) receives the eight PNG figures, ``json_path`` (default ``<run>\S1_REPORT.json``) the
    report and ``golden`` (default ``<run>\golden``) is read for the golden lock. The store is loaded with ``run_s1.load_s1``
    (``verify`` hashes every file). The JSON holds: the run folder name and the code commit of RUN.json and of this report; the
    comparisons with the prototype with pass / fail (``comparisons``, ``all_ok`` is true only when every one passed and none was
    skipped); the first numbers the gate looks at (``gate_summary``); the numbers behind each figure (``figures``); the extras for the
    gate (``extras``); the golden lock. It carries no match key and no player identifier: the finished text is checked against the keys
    of the store before it is written.


    비교 생략을 명시하고 전부 성공한 경우만 전체 통과로 기록한 뒤 표·그림·출처를 묶는다(S1–S2 Task8b).
    최종 문구의 저장 경기 식별자 검사는 쓰기 전에 수행한다.
    """
    run = Path(run)
    report_dir = run / 'report' if report_dir is None else Path(report_dir)
    json_path = run / 'S1_REPORT.json' if json_path is None else Path(json_path)
    golden = run / 'golden' if golden is None else Path(golden)
    s = run_s1.load_s1(run, verify=verify)
    manifest = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
    result = manifest['s1_result']
    ref = load_prototype()
    try:
        link_units, skipped = load_link_units(), None
    except Exception as exc:                  # the units comparison needs it; the report says so instead of leaving it out silently
        link_units, skipped = None, f'{type(exc).__name__}: {exc}'
    n = len(fit_mask(s))
    everyone, fit = np.ones(n, dtype=bool), fit_mask(s)

    comparisons = compare(s, ref, link_units, golden)
    if link_units is None:
        comparisons['units'] = {'ok': None, 'skipped': skipped}
    if 'golden' not in comparisons:
        comparisons['golden'] = {'ok': None, 'skipped': f'no GOLDEN.json under {golden.name}'}
    data = collect(s)
    plt = _plt()
    report_dir.mkdir(parents=True, exist_ok=True)
    g = data['groups']
    figures = {
        'fig1_units_per_kill': fig_units(plt, g, ref['g15'], report_dir / 'fig1_units_per_kill.png'),
        'fig2_size_classes': fig_sizes(plt, g, ref['g15'], report_dir / 'fig2_size_classes.png'),
        'fig3_onset_counts': fig_onsets(plt, data['onsets'], ref['anchor'], report_dir / 'fig3_onset_counts.png'),
        'fig4_objective_attribution': fig_attribution(plt, data['attribution']['fit'], ref['part'],
                                                      report_dir / 'fig4_objective_attribution.png'),
        'fig5_winner_loser_excess': fig_conversion(plt, data['aftermath']['fit'], ref['after'],
                                                   report_dir / 'fig5_winner_loser_excess.png'),
        'fig6_accounting': fig_accounting(plt, data['accounting'], report_dir / 'fig6_accounting.png',
                                          {'all': int(n), 'fit': int(fit.sum())}),
        'fig7_rows_with_engagement': fig_rows(plt, data['in_progress']['fit_rows_with_kills'], data['rows_per_unit'], ref,
                                              report_dir / 'fig7_rows_with_engagement.png'),
        'fig8_frame_lateness_and_ages': fig_frames(plt, data['lateness'], data['ages']['all'],
                                                   report_dir / 'fig8_frame_lateness_and_ages.png'),
    }
    for name in figures:
        figures[name] = {'file': f'report/{name}.png', 'numbers': figures[name]}
    extras = {
        'conversion_coverage': coverage_extras(s),
        'team_blind_overlap': {'all': team_blind_extras(s, everyone), 'fit': team_blind_extras(s, fit)},
        'counts': {'all': count_extras(s, everyone), 'fit': count_extras(s, fit)},
        'frame_lateness': data['lateness'],
        'event_row_ages': data['ages'],
        'attribution_all_stored_matches': data['attribution']['all'], 'aftermath_all_stored_matches': data['aftermath']['all'],
        'onsets_all_stored_rows': data['onsets_all_stored_rows'], 'in_progress_all_stored_rows': data['in_progress']['all'],
    }
    dirty = runlog._git('status', '--porcelain', '--', 'c_ept')
    run_info = {'name': run.name, 'code_commit': manifest['code_commit'], 'code_dirty': manifest['code_dirty'],
                'created': manifest['created'], 'plan_sha256': manifest['args']['plan_sha256'], 'spec_sha256': manifest['spec_sha256'],
                'config_sha256': manifest['config_sha256'], 'complete': result['complete'], 'matches': result['matches_stored'],
                'matches_by_macro': result['matches_by_macro'], 'matches_by_patch': result['matches_by_patch'],
                'rejects_total': result['rejects_total'], 'rows': result['rows'],
                'wall_s': result['wall_s'], 'peak_private_gib': result['peak_private_gib'], 'store_bytes': result['store_bytes']}
    comparisons = {'all_ok': all(c['ok'] is True for c in comparisons.values()), **comparisons}
    doc = {
        'report': 'S1_REPORT', 'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'run': run_info,
        'report_code': {'commit': (runlog._git('rev-parse', 'HEAD') or '').strip() or None,
                        'dirty': None if dirty is None else bool(dirty.strip()), 'store_verified': bool(verify)},
        'prototype_folder': PROTO_DIR.name,
        'gate_summary': gate_summary(run_info, comparisons, data, extras),
        'comparisons': comparisons,
        'golden_lock': _golden_info(golden),
        'figures': figures,
        'extras': extras,
    }
    text = json.dumps(_py(doc), indent=1) + '\n'
    assert_no_keys(s, text)
    store_core.write_atomic(json_path, text)
    s.maps.clear()
    return json_path


def main(argv=None) -> int:
    """호출 인자를 보고 경로에 전달하고 전체 비교 통과 여부를 종료 코드로 반환한다(S1–S2 Task8b)."""
    ap = argparse.ArgumentParser(prog='python -m c_ept.report_s1', description='S1 report: S1_REPORT.json and the figures of an S1 run.')
    ap.add_argument('run', type=Path, help='the S1 run folder')
    ap.add_argument('--report-dir', type=Path, help='folder of the figures (default <run>/report)')
    ap.add_argument('--json', type=Path, dest='json_path', help='path of the report (default <run>/S1_REPORT.json)')
    ap.add_argument('--golden', type=Path, help='golden lock folder (default <run>/golden)')
    ap.add_argument('--no-verify', action='store_true', help='do not hash the store files before reading them')
    a = ap.parse_args(argv)
    path = report(a.run, a.report_dir, a.json_path, a.golden, verify=not a.no_verify)
    doc = json.loads(path.read_text(encoding='utf-8'))
    c = doc['comparisons']
    print(f'S1 report written: {path}')
    print('figures: ' + ', '.join(f['file'] for f in doc['figures'].values()))
    print('comparisons: ' + ', '.join(f'{k}={v["ok"]}' for k, v in c.items() if k != 'all_ok') + f'; all_ok={c["all_ok"]}')
    return 0 if c['all_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
