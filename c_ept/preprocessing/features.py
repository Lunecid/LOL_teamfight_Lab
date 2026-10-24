r"""Data splits and match weights, the columns to drop, the shared baseline features b0 and b1, and the input of the masked M4 head
(stage S3 Task 5).

Splits and weights. ``split_rows(s1, role)`` marks the frame-clock rows of the matches of a role, ``role_mask(s1, table, role)`` the rows of
any S1 table with a match column (``units``, ``frame_rows``; ``onsets`` through the frame row of each onset) and ``split_views(s1, s2,
role)`` the views of an S2 store. A role is FIT, STOP or DEV (several may be given); TEST and EXTERNAL are refused as holdout roles and
CAL, AUDIT and UNUSED as roles stage S3 does not use, before any array is read (``flowprice`` has the one rule). ``match_weights(match)``
gives every row the weight 1/n_u, n_u the number of the rows passed that belong to its match, so that a match counts once in an average.

The zone label (ruling R17). ``unit_zone(s1)`` gives every engagement unit the zone (``terrain.ZONES`` index 0 .. 9) of the position of
its first kill (``units/kill0``; ``kills/x``, ``kills/y`` through ``terrain.zone_of``), and -1 where ``kills/pos_missing`` is set. It is
the label of the zone head rho and the zone input of the M3 and M4 heads; both baselines (B0, B1) use it, so the label exists once. It
reads those four arrays and nothing else; the b0 and b1 rows below read only the three S2 arrays.

Columns to drop (S2 gate S2G-4). ``dropped_columns(s2, fit_views)``: the summary columns and the player columns whose cells have one value
in all the views given (the FIT views of the three sources: frame-clock, M3 and M4 views), and ``t_s``, which always equals ``game_time_s``.
"One value" is the NaN-ignoring minimum equal to the maximum, over all the players of a player column; a column with no value at all counts
as constant. The result is a tuple of plain names, the summary columns in the order of ``tokens.SUMMARY`` and then the player columns in the
order of ``tokens.PLAYER_NUM``; the two lists have no name in common. ``fit_views`` has no default, and None or no view is refused (ruling
R14): an S2 store holds the views of every role (also CAL and AUDIT; store_build never builds the holdout roles) and only the FIT views may
decide which columns go, so the caller passes ``np.flatnonzero(split_views(s1, s2, 'FIT'))`` (or that mask) and reports the list.

b0, the small time-state table of the B0 heads: 39 columns, ``B0_NAMES``, in this order (blue minus red; the teams are the rows 0-4 and 5-9
of ``x/players_num``)

    time (8)       game_time_s / 60 (minutes), phase indicators 1 .. 7 of the 8 phases of ``flowprice.PHASE_EDGES_S`` (phase 0, the first,
                   is the reference and has no column)
    players (8)    team sums, blue minus red, of totalGold, xp, level, minionsKilled, jungleMinionsKilled, alive, respawn_s, item_gold_owned
    alive (1)      alive_total: players alive on both teams
    distance (2)   the minimum and the mean of the geodesic distances between the teams (``x/geo_pp``, the 5 x 5 block blue x red); pairs
                   with a missing position are left out of both, and a view with no pair left has NaN
    summary (15)   blue minus red: the four tower tiers, inhibitors down, plates (the three lanes added), dragons, elders, barons, heralds,
                   atakhans, voidgrubs, has_soul; and the baron and the elder buff, each an indicator, active when ``*_buff_s`` > 0
    appear (5)     dragon_up, baron_up, herald_up, atakhan_up, voidgrub_up, as the summary has them

The time is the summary column ``game_time_s`` (float32 seconds; its milliseconds, rounded, give the phase, as in the frame clock);
``level`` is the frame's level column, not ``level_now``, as the brief names it (on an event view the frame is up to a minute old).

No champion, item or patch id is in it, and nothing but ``x/summary``, ``x/players_num`` and ``x/geo_pp`` is read: a b0 row is a function of
the view alone. A NaN in a b0 column is filled with the mean of that column over the FIT views (ruling R2): ``b0_fit_means(s2,
fit_views)`` is computed once, from the cells that have a value, and passed to ``b0``; ``b0`` counts the cells it fills in
``fill_counts`` (an int array of 39 that it adds to). There are no missing-indicator columns, so the width is 39 whatever the data.
Without the means a b0 row with a NaN is refused, never filled from the views at hand.

b1, the table of the B1 LightGBM heads: ``b1_columns(dropped)`` names the columns, in this order

    the summary columns minus the dropped ones, in the order of ``tokens.SUMMARY``;
    ``sum_blue_<c>``, ``sum_red_<c>``, ``max_blue_<c>``, ``max_red_<c>`` for each player column c that is not dropped (the sum and the
    maximum of the five players of the team);
    ``geo_min``, ``geo_mean`` as in b0.

A NaN stays NaN (LightGBM reads it as missing): the sum or the maximum of a team is NaN when one of its five players has NaN in the column
(for example a missing position), and nothing is filled.

The masked M4 input. ``m4_masked_inputs(row_view, q, m, z, K, O)`` is the only way to build what the masked M4 head sees (the leakage test
(a) and decision P6): the features of the t0 row, then q, m and z of the onset, then the kills and the kill gold of the two teams (K =
kills blue, kills red, gold blue, gold red; the gold in thousands) and the objectives O of the 18 cells, as one float32 matrix.
``m4_columns(row_names)`` names its columns. The baselines turn q, m and z into one-hot columns (B0) or categories (B1) themselves.
Everything is read with a chunk of ``_CHUNK`` views at a time, so the 18.7 GiB S2 store is never in memory.

이 모듈은 B0·B1 기준 모형과 가린 M4 머리가 공유하는 입력을 만든다: 역할별 행 선택과 경기 가중, 첫 킬의 구역 값, 분산이 없는 열의 목록, b0·b1 특징 표, 가린 M4 입력 행렬.
보기(view)는 S2 저장소의 한 행이고 보기 번호는 그 0 기준 행이다. 한 보기의 특징은 그 보기 하나로 정해지며, 팀은 ``x/players_num``의 선수 행 p = 0–4(파랑)와 5–9(빨강)이다.
교전 단위는 한 교전이고 구역 z는 그 단위의 첫 킬 위치에서 얻는다.

식 목록
  (FEAT-1) match_weights: 행 가중 w_i = 1 / n_u
  (FEAT-2) unit_zone: 단위 i의 첫 킬 구역 z_i (위치가 없으면 −1)
  (FEAT-3) dropped_columns: 열 c를 빼는 조건 = (FIT 보기에서 한 값뿐이거나 값이 없음) ∨ t_s
  (FEAT-4) b0: 39열 = 시간 8, 선수 합의 차 8, 생존 1, 거리 2, 요약의 차 15, 등장 5
  (FEAT-5) b0의 거리 열: 파랑 × 빨강 측지 거리의 최솟값과 평균
  (FEAT-6) b0_fit_means: FIT 보기의 열 평균 mean_c와 결측 칸의 채움
  (FEAT-7) b1: 요약 열, 팀별 합과 최댓값, 거리 두 열
  (FEAT-8) m4_masked_inputs: 가린 M4 입력 행 (b0 또는 b1 행, q, m, z, K, O)

공통 가정과 검사. 검사하는 것: 역할 이름(FIT, STOP, DEV 밖은 배열을 읽기 전에 거부); 저장소 배열의 존재와 모양과 보기 수; 경기 번호가 역할 표 안에 있는 것; fit_views가 None이 아니고 비지 않고 범위 안인 것; b0의
시각이 유한하고 값이 ±∞가 아닌 것; q, m, z의 정수와 범위, K와 O의 유한과 비음수. 검사하지 않는 것: fit_views가 실제 FIT 보기인지(호출자가 split_views로 고르며, 판정 R14는 fit_views를 기본값 없이 요구할
뿐이고 FIT 보기를 넘기는 일은 러너가 맡는다); dropped가 FIT 보기에서 나온 것인지; K와 O에 참값이 들어갔는지 표본이 들어갔는지; LightGBM 버전.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (K is the kills and the kill gold of the two teams; the 18 cells of O), P6
#      (B0's masked M4 is conditioned on the t0 row and on q, m, z only, and B0 gets one-hot columns; B1's masked M4 also reads K and O,
#      plan Task 7), Task 5 and ruling R2 (b0 has 39 columns, a NaN takes the FIT mean, no indicator columns); S2 gate S2G-4 (columns
#      without variance in the FIT views, and t_s, are dropped); the global constraint "zone z" and ruling R17 (``unit_zone``: the zone of
#      the first kill, -1 without a position).
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md sections 5.3 (patch-invariant inputs: no champion, item or patch id) and 9 (the
#      baselines B0 and B1 and the masked and open heads).
# Ref: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017), "LightGBM: A Highly Efficient Gradient Boosting
#      Decision Tree", Advances in Neural Information Processing Systems 30 (a tree learner that reads a NaN as missing, so b1 fills
#      nothing).
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import numpy as np

from c_ept import config as C
from c_ept import flowprice
from c_ept.casts import narrow

# ---- the columns of b0 -----------------------------------------------------------------------------------------------------------------
_PLAYER_DIFFS = ('totalGold', 'xp', 'level', 'minionsKilled', 'jungleMinionsKilled', 'alive', 'respawn_s', 'item_gold_owned')
_SUMMARY_DIFFS = (('towers_outer', ('towers_outer',)), ('towers_inner', ('towers_inner',)), ('towers_base', ('towers_base',)),
                  ('towers_nexus', ('towers_nexus',)), ('inhibs_down', ('inhibs_down',)),
                  ('plates', ('plates_top', 'plates_mid', 'plates_bot')), ('dragons', ('dragons',)), ('elders', ('elders',)),
                  ('barons', ('barons',)), ('heralds', ('heralds',)), ('atakhans', ('atakhans',)), ('voidgrubs', ('voidgrubs',)),
                  ('has_soul', ('has_soul',)))
_BUFFS = ('baron', 'elder')
_UPS = ('dragon_up', 'baron_up', 'herald_up', 'atakhan_up', 'voidgrub_up')
N_B0 = 39
B0_NAMES = (('game_time_min',) + tuple(f'phase_{k}' for k in range(1, flowprice.N_PHASES))
            + tuple(f'diff_{name}' for name in _PLAYER_DIFFS) + ('alive_total', 'geo_min', 'geo_mean')
            + tuple(f'diff_{name}' for name, _ in _SUMMARY_DIFFS) + tuple(f'diff_{b}_buff' for b in _BUFFS) + _UPS)
# where each group starts in a b0 row
_O_PLAYERS = flowprice.N_PHASES                                   # after game_time_min and the 7 phase indicators
_O_ALIVE = _O_PLAYERS + len(_PLAYER_DIFFS)
_O_GEO = _O_ALIVE + 1
_O_SUMMARY = _O_GEO + 2
_O_UPS = _O_SUMMARY + len(_SUMMARY_DIFFS) + len(_BUFFS)
if len(B0_NAMES) != N_B0 or len(set(B0_NAMES)) != N_B0 or _O_UPS + len(_UPS) != N_B0:
    raise RuntimeError(f'b0 must have {N_B0} distinct columns, the code makes {len(B0_NAMES)}')

# ---- the input of the masked M4 head -----------------------------------------------------------------------------------------------------
N_ZONES = 10                                                      # categories of the zone rho (``terrain.ZONES``; a test pins this)
N_O = 2 * len(C.O_KINDS)                                          # cells of O: 2 kind + taker (0 blue, 1 red), as in S1
M4_EXTRA_NAMES = ('q', 'm', 'z', 'kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k')
O_NAMES = tuple(f'obj_{kind}_{team}' for kind in C.O_KINDS for team in ('blue', 'red'))
if len(O_NAMES) != N_O:
    raise RuntimeError('the cells of O are 2 per objective kind')

_CHUNK = 1 << 14                                                  # views read at a time: 16,384 x 3,240 bytes = 53 MB of players
_TABLE = re.compile(r'[a-z_]+')


# ---- the S2 columns ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Plan:
    """Which column of ``x/summary`` and of ``x/players_num`` each part of b0 reads."""
    summary_names: tuple
    player_names: tuple
    summary_index: dict
    player_index: dict
    time: int
    player_diffs: list
    alive: int
    diffs: tuple          # per summary difference: (blue columns, red columns)
    buffs: tuple          # per buff: (blue seconds column, red seconds column)
    ups: list


@functools.lru_cache(maxsize=1)
def _plan() -> _Plan:
    """The column numbers of b0. Imports ``tokens`` here, so that a caller of the split functions does not load the S2 modules."""
    from c_ept import tokens
    summary, players = tuple(tokens.SUMMARY), tuple(tokens.PLAYER_NUM)
    both = sorted(set(summary) & set(players))
    if both:
        raise RuntimeError(f'the S2 summary and player columns must have no name in common, but both have {", ".join(both)}')
    if 't_s' not in summary:
        raise RuntimeError('the S2 summary has no column t_s')
    s_index, p_index = {n: i for i, n in enumerate(summary)}, {n: i for i, n in enumerate(players)}

    def need(index, name, where):
        if name not in index:
            raise RuntimeError(f'the S2 {where} has no column {name!r}')
        return index[name]
    return _Plan(
        summary_names=summary, player_names=players, summary_index=s_index, player_index=p_index,
        time=need(s_index, 'game_time_s', 'summary'), player_diffs=[need(p_index, n, 'players') for n in _PLAYER_DIFFS],
        alive=need(p_index, 'alive', 'players'),
        diffs=tuple((tuple(need(s_index, f'blue_{p}', 'summary') for p in parts),
                     tuple(need(s_index, f'red_{p}', 'summary') for p in parts)) for _, parts in _SUMMARY_DIFFS),
        buffs=tuple((need(s_index, f'blue_{b}_buff_s', 'summary'), need(s_index, f'red_{b}_buff_s', 'summary')) for b in _BUFFS),
        ups=[need(s_index, n, 'summary') for n in _UPS])


# ---- splits and weights ------------------------------------------------------------------------------------------------------------------
def _in_roles(roles: np.ndarray, match: np.ndarray, codes: np.ndarray, where: str) -> np.ndarray:
    """(len(match),) bool: the role of each match number is one of ``codes``; ValueError for a number the role table does not have."""
    match = match.astype(np.int64)
    if match.size and (match.min() < 0 or match.max() >= len(roles)):
        raise ValueError(f'{where} holds match numbers outside 0 .. {len(roles) - 1}: the match number(s) do not belong to this S1 store')
    return np.isin(roles[match], codes)


def role_mask(s1, table: str, role) -> np.ndarray:
    """(n rows of ``table``,) bool: the rows of the S1 table whose match has the role ``role`` (a name or names out of FIT, STOP, DEV).

    ``table`` is an S1 table with a match column (``<table>/match``: ``frame_rows``, ``units``), or ``onsets``, whose rows are found through
    the frame row of each onset (``onsets/row`` -> ``frame_rows/match``). ValueError for a holdout role or a role stage S3 does not use
    (before anything is read), for a table without a match column, and for match numbers the store does not have.

    각 행의 경기 번호로 역할 표를 찾아 요청한 역할에 드는 행만 고른다. onsets는 자기 경기 열이 없어 프레임 행을 거쳐 경기에 닿는다."""
    codes = flowprice._role_codes(role)
    if not isinstance(table, str) or not _TABLE.fullmatch(table):
        raise ValueError(f'the S1 table {table!r} has no match column')
    if table == 'onsets':
        row = np.asarray(s1.array('onsets/row')).astype(np.int64)
        frame_match = np.asarray(s1.array('frame_rows/match'))
        if row.size and (row.min() < 0 or row.max() >= len(frame_match)):
            raise ValueError('onsets/row names a frame row that the S1 store does not have')
        match, where = frame_match[row], 'onsets/row through frame_rows/match'
    else:
        where = f'{table}/match'
        try:
            match = np.asarray(s1.array(where))
        except FileNotFoundError:
            raise ValueError(f'the S1 table {table!r} has no match column (the store has no array {where})') from None
    return _in_roles(np.asarray(s1.array('matches/role')), match, codes, where)


def split_rows(s1, role) -> np.ndarray:
    """(n frame rows,) bool: the frame-clock rows (``frame_rows``) of the matches whose role is ``role`` (``matches/role``).

    프레임 시계 행을 역할로 가르는 ``role_mask``의 한 경우이다."""
    return role_mask(s1, 'frame_rows', role)


def split_views(s1, s2, role) -> np.ndarray:
    """(n views,) bool: the views of the S2 store ``s2`` (all three sources) whose match, ``views/match``, has the role ``role`` in the S1
    store ``s1``. Their indices (``numpy.flatnonzero``) are the ``fit_views`` of ``dropped_columns`` and ``b0_fit_means`` for role FIT.

    S2의 보기를 그 경기의 역할로 가른다. FIT 보기를 고르는 일은 이 함수의 결과를 넘기는 호출자의 몫이고, 받는 쪽은 그것이 FIT인지 다시 확인하지 않는다."""
    codes = flowprice._role_codes(role)
    return _in_roles(np.asarray(s1.array('matches/role')), np.asarray(s2.array('views/match')), codes, 'views/match')


def match_weights(match) -> np.ndarray:
    """(n,) float64: 1 / n_u for each row, n_u the number of the rows in ``match`` (the match number of every row, integers) that have the
    same match number. The weights of a match add up to 1. TypeError for match numbers that are not integers, ValueError for a shape that
    is not 1-d.

    수식
    ----
    (FEAT-1)  w_i = 1 / n_u,   u = match_i,   n_u = #{ i′ : match_i′ = u }      (float64)

      기호    | 뜻                                      | 단위 | 범위                  | 코드 변수
      i       | 행 번호(0부터)                          | —    | 0, …, n − 1           | 반환 배열의 색인
      match_i | 행 i의 경기 번호(정수)                  | —    | 정수                  | match
      u       | 경기 번호                               | —    | match_i의 값          | unique의 값
      n_u     | 경기 u에 속한 행의 수                   | 행   | ≥ 1                   | counts
      w_i     | 행 가중                                 | —    | (0, 1]                | 반환값

    해석
    ----
    식 (FEAT-1)은 한 경기의 행들이 가중 합 1을 나누어 가지게 하므로, 가중 평균에서 경기마다 같은 무게를 받고 행이 많은 경기가 평균을 지배하지 않는다. 경기를 서로
    독립인 단위로 보기 때문에 경기 u의 행마다 1 / n_u를 준다. 한 경기의 가중 합은 부동소수점 반올림 안에서 1이고, 행이 없으면 빈 배열을 낸다. 계획 P9에 따라 학습은
    행마다 같은 무게를 쓰고 이 가중은 평가에 쓴다.

    근거
    ----
    S3 계획 P9(학습은 행마다 같은 무게, 평가는 경기 가중); 설계서 8절(비교 공통 규칙, 경기 가중); S3 계획 Task 5.
    """
    m = np.asarray(match)
    if m.size == 0:
        return np.empty(0, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError(f'match_weights: the match numbers must be 1-d, not shape {m.shape}')
    if m.dtype.kind not in 'iu':
        raise TypeError(f'match_weights: the match numbers must be integers, not {m.dtype}')
    _, inverse, counts = np.unique(m, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse.reshape(-1)]


# ---- the zone label of a unit (ruling R17) -----------------------------------------------------------------------------------------------
def unit_zone(s1) -> np.ndarray:
    """(n units,) int8: the zone of the first kill of every unit of the S1 store, an index into ``terrain.ZONES`` (0 .. 9), and -1 where the
    kill has no position (``kills/pos_missing``; a NaN coordinate is no position either). Units are in the order of the S1 ``units`` table,
    all roles; select the rows of a role with ``role_mask(s1, 'units', role)``.

    Reads ``units/kill0`` (the global kill number of the first kill of each unit), ``kills/x``, ``kills/y`` and ``kills/pos_missing``.
    ValueError for a store without one of them (naming it), for arrays of different numbers of kills and for a first kill that is no kill
    of the store. The ``-1`` is no zone: the heads leave it out of the fit of rho and count it, and give it its own category as an input.

    수식
    ----
    (FEAT-2)  z_i = −1 (pos_missing[k₀(i)]),   z_i = zone_of(pos_x(k₀(i)), pos_y(k₀(i))) (그 밖);   zone_of는 (TER-7), 좌표가 NaN이면 −1      (int8)

      기호          | 뜻                                          | 단위 | 범위                  | 코드 변수
      i             | 단위 번호(0부터, S1 units 표의 행)          | —    | 0, …, 단위 수 − 1     | 반환 배열의 색인
      k₀(i)         | 단위 i의 첫 킬의 전역 킬 번호(0부터)        | —    | 0, …, 킬 수 − 1       | units/kill0
      pos_x, pos_y  | 그 킬의 위치 좌표(코드의 kills/x, kills/y)  | u    | 유한 또는 NaN         | kills/x, kills/y
      pos_missing   | 위치가 없다는 표시                          | —    | 참 또는 거짓          | kills/pos_missing
      z_i           | 첫 킬의 구역(``terrain.ZONES``의 0 기준 색인) | —  | −1, 0, …, 9           | 반환값

    해석
    ----
    식 (FEAT-2)는 단위마다 첫 킬의 구역을 구역 머리 ρ의 정답으로 주고, M3·M4 머리의 구역 입력으로도 준다. 위치가 없는 킬(위치 없음 표시가 참이거나 좌표가 NaN)은
    구역이 없어 −1이며, 머리는 −1을 ρ의 적합에서 빼고 세며 입력에서는 별도 범주로 둔다(S3 계획 전역 제약 구역 z). 두 기준 모형이 같은 구역 값을 쓰도록 이 값은 한 곳에서만
    만든다(판정 R17). 결과는 모든 역할의 단위를 담으므로 한 역할의 단위는 역할 마스크로 따로 고른다.

    근거
    ----
    S3 판정 R17(첫 킬의 구역, 위치가 없으면 구역 없음); S3 계획 전역 제약 구역 z; 구역 규칙은 (TER-7).
    """
    from c_ept import terrain       # here, not at module level: the zone rules live in ``terrain``, which imports the run log
    arrays = {}
    for name in ('units/kill0', 'kills/x', 'kills/y', 'kills/pos_missing'):
        try:
            arrays[name] = s1.array(name)
        except FileNotFoundError:
            raise ValueError(f'the S1 store has no array {name}') from None
    n_kills = len(arrays['kills/x'])
    if len(arrays['kills/y']) != n_kills or len(arrays['kills/pos_missing']) != n_kills:
        raise ValueError('kills/x, kills/y and kills/pos_missing hold different numbers of kills')
    kill0 = np.asarray(arrays['units/kill0']).astype(np.int64)
    if kill0.size and (kill0.min() < 0 or kill0.max() >= n_kills):
        raise ValueError(f'units/kill0 names a kill outside 0 .. {n_kills - 1}')
    zone = terrain.zone_of(np.asarray(arrays['kills/x'])[kill0], np.asarray(arrays['kills/y'])[kill0])
    return narrow('unit_zone', np.where(np.asarray(arrays['kills/pos_missing'])[kill0].astype(bool), -1, zone), np.int8)


# ---- reading the S2 arrays ---------------------------------------------------------------------------------------------------------------
_X_ARRAYS = {'summary': 'x/summary', 'players_num': 'x/players_num', 'geo_pp': 'x/geo_pp'}


def _arrays(s2, *keys: str) -> dict:
    """{key: read-only memmap of ``x/<key>``} after checking that each is what ``tokens`` says and that they hold the same views."""
    plan = _plan()
    shapes = {'summary': ('(n, {})', lambda a: a.ndim == 2 and a.shape[1] == len(plan.summary_names), len(plan.summary_names)),
              'players_num': ('(n, 10, {})', lambda a: a.ndim == 3 and a.shape[1:] == (10, len(plan.player_names)), len(plan.player_names)),
              'geo_pp': ('(n, 10, {})', lambda a: a.ndim == 3 and a.shape[1:] == (10, 10), 10)}
    out = {}
    for key in keys:
        name = _X_ARRAYS[key]
        try:
            out[key] = s2.array(name)
        except FileNotFoundError:
            raise ValueError(f'the S2 store has no array {name}') from None
        form, ok, width = shapes[key]
        if not ok(out[key]):
            raise ValueError(f'{name} must have shape {form.format(width)}, not {out[key].shape}'
                             + (f' (tokens.SUMMARY has {width} columns)' if key == 'summary' else ''))
    if len({len(a) for a in out.values()}) > 1:
        raise ValueError('the arrays ' + ', '.join(_X_ARRAYS[k] for k in out) + ' hold different numbers of views: '
                         + ', '.join(str(len(a)) for a in out.values()))
    return out


def view_index(n_views: int, views) -> np.ndarray:
    """``views`` (a 1-d array or list of view numbers, or a bool mask over all the views) as an int64 array of view numbers.

    수식
    ----
    (FEAT-9) I_view = sorted{j : selected[j] = True} (bool mask); I_view = int64(view_request) (integer indices); I_view = [] (empty non-bool input).

      기호   | 뜻                           | 단위    | 범위                     | 코드 변수
      N_view | 전체 보기 수                 | 보기 수 | 호출자가 주는 비음수 정수 | n_views
      view_request | 요청한 보기 번호 배열        | 보기 번호 | 정수, 1차원 (비어 있으면 형태 검사 전 반환) | views, v
      selected[j] | 보기 j의 선택 표시       | —      | False 또는 True; selected의 모양 (N_view,) | views[j], v[j]
      j      | bool 마스크의 위치           | 보기 번호 | 0 .. N_view − 1         | flatnonzero의 위치
      I_view | 반환하는 보기 번호 열        | 보기 번호 | int64, 1차원, 0 .. N_view − 1 | 반환값

    해석
    ----
    bool 마스크는 참인 위치를 오름차순으로 내고, 정수 번호는 요청 순서와 중복을 보존한다.
    bool 마스크의 형태를 먼저 검사하며, 그 밖의 빈 입력은 기존 계약대로 dtype·차원 검사 전에 빈 int64 열을 낸다.
    비어 있지 않은 입력의 dtype이 정수가 아니면 TypeError, 차원이나 번호 범위가 틀리면 ValueError이다.
    전체 보기 수 자체의 별도 검증은 추가하지 않으며, 이전 이름 _view_index는 같은 함수의 호환 별칭이다.

    근거
    ----
    기존 features._view_index의 검증·순서 계약; S4 deferred T1-d6의 공개 API 정리 (계산 변경 없음).
    """
    v = np.asarray(views)
    if v.dtype == np.bool_:
        if v.shape != (n_views,):
            raise ValueError(f'a mask over the views must have {n_views} entries, not shape {v.shape}')
        return np.flatnonzero(v).astype(np.int64)
    if v.size == 0:
        return np.empty(0, dtype=np.int64)
    if v.dtype.kind not in 'iu':
        raise TypeError(f'view numbers must be integers, not {v.dtype}')
    if v.ndim != 1:
        raise ValueError(f'view numbers must be 1-d, not shape {v.shape}')
    if v.min() < 0 or v.max() >= n_views:
        raise ValueError(f'view numbers must lie in 0 .. {n_views - 1}')
    return v.astype(np.int64)


# Compatibility alias for callers using the former private name.
_view_index = view_index


def _chunks(idx: np.ndarray):
    """(first position, view numbers) of consecutive chunks of at most ``_CHUNK`` views."""
    for lo in range(0, len(idx), _CHUNK):
        yield lo, idx[lo:lo + _CHUNK]


def _rows(array, sel: np.ndarray) -> np.ndarray:
    return np.asarray(array[sel])


# ---- columns to drop ---------------------------------------------------------------------------------------------------------------------
def dropped_columns(s2, fit_views) -> tuple:
    """The names of the summary and player columns to leave out of the baseline inputs (see the module text): those with one value in
    all the views of ``fit_views`` and ``'t_s'``. ``fit_views`` is required: the view numbers (or a bool mask over the views) of the FIT
    matches, all three sources, that is ``np.flatnonzero(split_views(s1, s2, 'FIT'))``; the store holds the views of every role it was built
    for, so there is no default that could judge them together. Reads ``x/summary`` and ``x/players_num`` in chunks. ValueError for None or
    no view.

    수식
    ----
    (FEAT-3)  min_c = min_{ i ∈ I_fit, p } x_{i,p,c},  max_c = max_{ i ∈ I_fit, p } x_{i,p,c}  (NaN은 건너뜀; 요약 열은 p 없이)
              열 c를 뺀다 ⇔ min_c가 NaN이거나 min_c = max_c, 또는 c = t_s

      기호    | 뜻                                        | 단위   | 범위                    | 코드 변수
      I_fit   | 호출자가 준 보기 번호의 집합              | —      | 0, …, 보기 수 − 1       | fit_views (idx)
      c       | 요약 열 또는 선수 열                      | —      | tokens.SUMMARY, PLAYER_NUM | plan.summary_names, player_names
      p       | 선수 행(0부터)                            | —      | 0, …, 9                 | 선수 열의 축
      x       | 그 열의 값(float32)                       | 열마다 | 유한 또는 NaN           | summary, players_num
      min_c, max_c | I_fit에서의 최솟값, 최댓값(NaN 제외) | 열마다 | float32 또는 NaN        | s_lo, s_hi, p_lo, p_hi

    해석
    ----
    식 (FEAT-3)은 FIT 보기에서 값이 하나뿐이거나 아예 없는 열과, 늘 게임 시각과 같은 중복 열 t_s를 기준 모형의 입력에서 뺀다. 비교는 NaN을 건너뛴 float32
    최솟값과 최댓값이 같은지이며, 선수 열은 선수 10명의 값을 모두 모아 한 번에 비교한다.

    근거
    ----
    S2 관문 S2G-4(FIT 보기에서 분산 없는 열과 t_s를 뺀다); S3 판정 R14(fit_views는 기본값 없이 요구); S3 계획 Task 5.
    """
    if fit_views is None:
        raise ValueError("dropped_columns: fit_views is required: pass the FIT views, np.flatnonzero(features.split_views(s1, s2, 'FIT')); "
                         'an S2 store holds the views of every role and only the FIT views may decide which columns are dropped')
    plan = _plan()
    a = _arrays(s2, 'summary', 'players_num')
    idx = view_index(len(a['summary']), fit_views)
    if idx.size == 0:
        raise ValueError('dropped_columns: there are no FIT views to judge the columns by (pass the FIT views of split_views)')
    s_lo = s_hi = np.full(len(plan.summary_names), np.nan, dtype=np.float32)
    p_lo = p_hi = np.full(len(plan.player_names), np.nan, dtype=np.float32)
    for _, sel in _chunks(idx):
        s = _rows(a['summary'], sel)
        p = _rows(a['players_num'], sel).reshape(-1, len(plan.player_names))
        # (FEAT-3)의 min_c, max_c: 조각마다 NaN을 건너뛴 최솟값·최댓값을 합쳐 가므로 저장소 전체는 메모리에 오르지 않는다.
        s_lo, s_hi = np.fmin(s_lo, np.fmin.reduce(s, axis=0)), np.fmax(s_hi, np.fmax.reduce(s, axis=0))      # fmin and fmax skip NaN
        p_lo, p_hi = np.fmin(p_lo, np.fmin.reduce(p, axis=0)), np.fmax(p_hi, np.fmax.reduce(p, axis=0))

    def constant(lo, hi):
        return np.isnan(lo) | (lo == hi)                                 # no value at all, or one value
    drop_s, drop_p = constant(s_lo, s_hi), constant(p_lo, p_hi)
    return (tuple(name for name, drop in zip(plan.summary_names, drop_s) if drop or name == 't_s')
            + tuple(name for name, drop in zip(plan.player_names, drop_p) if drop))


# ---- b0 ----------------------------------------------------------------------------------------------------------------------------------
def _geodesic(geo: np.ndarray) -> tuple:
    """(minimum, mean) of the blue x red block of ``geo`` (k, 10, 10), each (k,) float64: NaN pairs are left out, NaN without a pair.
    (FEAT-5)의 두 열을 계산한다."""
    block = geo[:, :5, 5:].astype(np.float64).reshape(len(geo), 25)
    have = ~np.isnan(block)
    count = have.sum(axis=1)
    with np.errstate(all='ignore'):
        lowest = np.fmin.reduce(block, axis=1)
        mean = np.where(count > 0, np.where(have, block, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    return lowest, mean


def _b0_raw(plan: _Plan, summary: np.ndarray, players: np.ndarray, geo: np.ndarray) -> np.ndarray:
    """(k, 39) float64: the b0 columns of k views, NaN where a cell is missing. (FEAT-4)의 열을 계산한다."""
    k = len(summary)
    # (FEAT-4): 열을 float64로 계산하고, float32로 줄이는 변환은 b0에서 한 번만 한다.
    S = summary.astype(np.float64)
    t = S[:, plan.time]
    if not np.isfinite(t).all():
        raise ValueError(f'b0: game_time_s is not finite in {int((~np.isfinite(t)).sum())} of {k} views')
    out = np.empty((k, N_B0))
    out[:, 0] = t / 60.0
    phase = flowprice.phase_of(np.rint(t * 1000.0).astype(np.int64))             # the milliseconds of the time, as in the frame clock
    out[:, 1:_O_PLAYERS] = phase[:, None] == np.arange(1, flowprice.N_PHASES)[None, :]
    P = players[:, :, plan.player_diffs].astype(np.float64)
    out[:, _O_PLAYERS:_O_ALIVE] = P[:, :5].sum(axis=1) - P[:, 5:].sum(axis=1)
    out[:, _O_ALIVE] = players[:, :, plan.alive].astype(np.float64).sum(axis=1)
    # (FEAT-5)의 geo_min, geo_mean: 위치가 없는 쌍은 두 열에서 모두 빠진다.
    out[:, _O_GEO], out[:, _O_GEO + 1] = _geodesic(geo)
    for j, (blue, red) in enumerate(plan.diffs):
        out[:, _O_SUMMARY + j] = S[:, list(blue)].sum(axis=1) - S[:, list(red)].sum(axis=1)
    for j, (blue, red) in enumerate(plan.buffs, start=len(plan.diffs)):
        # (FEAT-4)의 버프 열: 남은 초가 양수인 쪽이 활성이고, 한 쪽이라도 NaN이면 차이는 NaN이다.
        b, r = S[:, blue], S[:, red]
        out[:, _O_SUMMARY + j] = np.where(np.isnan(b) | np.isnan(r), np.nan, (b > 0).astype(np.float64) - (r > 0).astype(np.float64))
    out[:, _O_UPS:] = S[:, plan.ups]
    infinite = np.isinf(out).any(axis=0)
    if infinite.any():
        raise ValueError('b0: infinite value in ' + ', '.join(B0_NAMES[j] for j in np.flatnonzero(infinite)))
    return out


def _b0_blocks(idx: np.ndarray, a: dict):
    """(first position, (k, 39) float64 raw block) for the chunks of the views ``idx``."""
    plan = _plan()
    for lo, sel in _chunks(idx):
        yield lo, _b0_raw(plan, _rows(a['summary'], sel), _rows(a['players_num'], sel), _rows(a['geo_pp'], sel))


def b0_fit_means(s2, fit_views) -> np.ndarray:
    """(39,) float64: the mean of every b0 column over the views ``fit_views`` (the FIT views), taken over the cells that have a value (NaN
    cells are left out). Compute it once and pass it to ``b0`` for every set of views; it is what a missing cell takes (ruling R2).
    ValueError for no views, and for a column without a value in them (the message names it).

    수식
    ----
    (FEAT-6)  mean_c = Σ_{ i ∈ I_fit : raw_{i,c}는 NaN이 아님 } raw_{i,c} / #{ i ∈ I_fit : raw_{i,c}는 NaN이 아님 }      (float64, c = 0, …, 38)

      기호   | 뜻                                          | 단위   | 범위                | 코드 변수
      raw_{i,c} | 보기 i의 b0 열 c의 채우기 전 값          | 열마다 | 유한 또는 NaN       | _b0_blocks의 raw
      I_fit  | 호출자가 준 보기 번호의 집합                | —      | 비지 않음           | fit_views (idx)
      mean_c | 열 c의 평균                                 | 열마다 | 유한                | 반환 (39,)

    해석
    ----
    식 (FEAT-6)의 평균은 값이 있는 칸만으로 구하며 보기마다 같은 무게이고 경기 가중은 없다. 이 평균이 B0 특징의 결측 칸이 받는 값이므로, 한 번 구해 모든 보기
    집합에 넘겨 손에 든 보기로 채우는 일이 없게 한다(판정 R2). 값이 하나도 없는 열은 평균이 정의되지 않으므로 거부한다.

    근거
    ----
    S3 판정 R2(결측 칸은 FIT 평균, 결측 표시 열 없음); S3 판정 R34(열별 채움 수를 보고); S3 계획 Task 5.
    """
    a = _arrays(s2, 'summary', 'players_num', 'geo_pp')
    idx = view_index(len(a['summary']), fit_views)
    if idx.size == 0:
        raise ValueError('b0_fit_means: there are no views to take the means over')
    total, count = np.zeros(N_B0), np.zeros(N_B0, dtype=np.int64)
    for _, raw in _b0_blocks(idx, a):
        have = ~np.isnan(raw)
        total += np.where(have, raw, 0.0).sum(axis=0)
        count += have.sum(axis=0)
    if (count == 0).any():
        raise ValueError('b0_fit_means: no value in these views for ' + ', '.join(B0_NAMES[j] for j in np.flatnonzero(count == 0)))
    return total / count


def _checked_means(fit_means):
    if fit_means is None:
        return None
    means = np.asarray(fit_means, dtype=np.float64)
    if means.shape != (N_B0,) or not np.isfinite(means).all():
        raise ValueError(f'fit_means must be {N_B0} finite numbers (features.b0_fit_means), not shape {means.shape} with non-finite values')
    return means


def _checked_counts(fill_counts):
    if fill_counts is None:
        return None
    if not isinstance(fill_counts, np.ndarray) or fill_counts.dtype.kind not in 'iu':
        raise TypeError(f'fill_counts must be a numpy integer array of {N_B0} counts, not {type(fill_counts).__name__}')
    if fill_counts.shape != (N_B0,):
        raise ValueError(f'fill_counts must have shape ({N_B0},), not {fill_counts.shape}')
    return fill_counts


def b0(s2, views, fit_means=None, *, fill_counts=None) -> np.ndarray:
    """(n, 39) float32: the b0 columns (``B0_NAMES``) of the views ``views`` of the S2 store ``s2`` (view numbers in any order, or a bool
    mask).

    ``fit_means``: the (39,) FIT means of ``b0_fit_means``. A NaN cell takes the mean of its column (ruling R2); without ``fit_means`` a
    view with a NaN cell is refused (ValueError), so that no view is ever filled from the views at hand. ``fill_counts``: an int array
    of 39 that gets the number of cells filled, per column, added to it (nothing is added if the call fails). Reads ``x/summary``,
    ``x/players_num`` and ``x/geo_pp`` only. ValueError for a time that is not finite or an infinite value (naming the column).

    수식
    ----
    (FEAT-4)  열 0: summary[game_time_s] / 60;   열 1–7: [phase(rint(1000 · summary[game_time_s])) = ph], ph = 1, …, 7
              열 8–15: Σ_{p=0}^{4} players[p, c] − Σ_{p=5}^{9} players[p, c]
                (c: totalGold, xp, level, minionsKilled, jungleMinionsKilled, alive, respawn_s, item_gold_owned)
              열 16: Σ_{p=0}^{9} players[p, alive];   열 17, 18: (FEAT-5)
              열 19–31: Σ_{blue 열} summary − Σ_{red 열} summary
                (포탑 4종, inhibs_down, plates(세 레인의 합), dragons, elders, barons, heralds, atakhans, voidgrubs, has_soul)
              열 32, 33: [blue_<buff>_buff_s > 0] − [red_<buff>_buff_s > 0] (buff = baron, elder; 둘 중 하나가 NaN이면 NaN)
              열 34–38: summary[dragon_up, baron_up, herald_up, atakhan_up, voidgrub_up]
    (FEAT-5)  pairs = {(p, p′) : p = 0, …, 4, p′ = 5, …, 9, geo[p, p′]가 NaN이 아님}
              geo_min = min geo[p, p′],  geo_mean = Σ geo[p, p′] / #pairs;  pairs가 비면 둘 다 NaN
    (FEAT-6)  채움: 열 c의 NaN 칸 ← mean_c (``b0_fit_means``);  출력은 float32

      기호          | 뜻                                          | 단위   | 범위                   | 코드 변수
      summary[c]    | x/summary의 열 c(float32를 float64로)       | 열마다 | 유한 또는 NaN          | summary
      players[p, c] | x/players_num의 선수 행 p, 열 c             | 열마다 | 유한 또는 NaN          | players
      geo[p, p′]    | x/geo_pp의 두 선수 사이 측지 거리           | u      | ≥ 0 또는 NaN           | geo
      pairs, geo_min, geo_mean | 파랑 × 빨강 중 거리가 있는 쌍, 최솟값, 평균 | —; u; u | 쌍이 없으면 NaN | have, 열 이름 geo_min, geo_mean
      phase(t)      | 프레임 시계의 시기 번호(ms에서)             | —      | 0, …, 7                | flowprice.phase_of
      ph            | 시기 번호(기준인 0은 열이 없음)             | —      | 1, …, 7                | 열 1–7의 번호
      buff          | 버프 이름                                   | —      | baron, elder           | _BUFFS

    해석
    ----
    식 (FEAT-4)는 한 보기를 39열로 줄이며, 선수 값의 팀 합과 요약 값은 파랑 − 빨강의 차이로 두고 시간과 시기 지시자, 두 팀의 생존 합, 거리, 등장 표시는 차이를 내지
    않는다. 시기 지시자는 시각을 ms로 반올림해 프레임 시계와 같은 규칙으로 정하고, 첫 시기를 기준으로 두어 그 열은 없다. 버프는 남은 초가 양수인 쪽을 활성으로 보고 두
    팀 지시자의 차이를 낸다. 거리 (FEAT-5)는 위치가 없는 쌍을 두 열에서 모두 빼므로 쌍이 하나도 없으면 NaN이고, 그 칸과 다른 결측 칸은 (FEAT-6)의 FIT 평균을 받는다.
    결측 표시 열을 두지 않으므로 열의 수는 자료와 무관하게 39이고(판정 R2), 챔피언·아이템·패치 id는 읽지 않는다(설계서 5.3절).

    근거
    ----
    S3 판정 R2(고정 폭, FIT 평균, 결측 표시 열 없음), R34(열별 채움 수); S3 계획 Task 5; 설계서 5.3절(챔피언·아이템·패치 id 없음)과 9절(기준 모형 B0).
    """
    means, counts = _checked_means(fit_means), _checked_counts(fill_counts)
    a = _arrays(s2, 'summary', 'players_num', 'geo_pp')
    idx = view_index(len(a['summary']), views)
    out = np.empty((len(idx), N_B0), dtype=np.float32)
    filled = np.zeros(N_B0, dtype=np.int64)
    for lo, raw in _b0_blocks(idx, a):
        nan = np.isnan(raw)
        if nan.any():
            if means is None:
                raise ValueError(f'b0: {int(nan.sum())} cell(s) in {int(nan.any(axis=1).sum())} view(s) of this chunk have no value (a '
                                 'missing position leaves no distance): pass fit_means, the FIT mean of every column '
                                 '(features.b0_fit_means)')
            # (FEAT-6)의 채움: NaN 칸만 FIT 평균으로 바꾸고 센다. 평균이 없으면 위에서 이미 거부했다.
            raw = np.where(nan, means[None, :], raw)
            filled += nan.sum(axis=0)
        out[lo:lo + len(raw)] = narrow('b0', raw, np.float32)
    if counts is not None:
        counts += filled.astype(counts.dtype)
    return out


# ---- b1 ----------------------------------------------------------------------------------------------------------------------------------
# (FEAT-7)의 남긴 열을 정한다.
def _kept(plan: _Plan, dropped) -> tuple:
    """(kept summary column numbers, kept player column numbers) once the names ``dropped`` are left out; ValueError for a name that is no
    column of the view."""
    if isinstance(dropped, str):
        raise TypeError('dropped must be a collection of column names, not one string')
    names = list(dropped)
    unknown = sorted({str(n) for n in names if n not in plan.summary_index and n not in plan.player_index})
    if unknown:
        raise ValueError('not columns of the S2 view (tokens.SUMMARY, tokens.PLAYER_NUM): ' + ', '.join(unknown))
    drop = set(names)
    return (tuple(i for i, n in enumerate(plan.summary_names) if n not in drop),
            tuple(i for i, n in enumerate(plan.player_names) if n not in drop))


def b1_columns(dropped=()) -> tuple:
    """The names of the b1 columns when the columns ``dropped`` are left out (see the module text for the order).

    (FEAT-7)의 열 이름을 순서대로 낸다. 모르는 열 이름은 거부한다."""
    plan = _plan()
    ks, kp = _kept(plan, dropped)
    players = tuple(plan.player_names[i] for i in kp)
    return (tuple(plan.summary_names[i] for i in ks)
            + tuple(f'{how}_{team}_{c}' for how in ('sum', 'max') for team in ('blue', 'red') for c in players) + ('geo_min', 'geo_mean'))


def b1(s2, views, dropped) -> np.ndarray:
    """(n, d) float32 with d = ``len(b1_columns(dropped))``: the b1 columns of the views ``views`` (view numbers in any order, or a bool
    mask). A NaN stays NaN; nothing is filled. Reads ``x/summary``, ``x/players_num`` and ``x/geo_pp`` only.

    수식
    ----
    (FEAT-7)  요약 열 c ∈ 남긴 열: summary[c];   남긴 선수 열 c마다: sum_blue_c = Σ_{p=0}^{4} players[p, c],  sum_red_c = Σ_{p=5}^{9} players[p, c],
              max_blue_c = max_{p=0..4} players[p, c],  max_red_c = max_{p=5..9} players[p, c]
              마지막 두 열: geo_min, geo_mean (FEAT-5);   float32

      기호          | 뜻                                          | 단위   | 범위                | 코드 변수
      c             | 빼지 않은 요약 열 또는 선수 열              | —      | tokens의 열         | ks, kp
      summary[c]    | x/summary의 열 c                            | 열마다 | 유한 또는 NaN       | summary
      players[p, c] | x/players_num의 선수 행 p, 열 c             | 열마다 | 유한 또는 NaN       | players
      d             | 열의 수                                     | 열     | 요약 열 수 + 4 × 선수 열 수 + 2 | len(b1_columns(dropped))

    해석
    ----
    식 (FEAT-7)은 선수 열마다 두 팀의 합과 최댓값을 따로 두고(차이로 줄이지 않는다), 요약 열은 그대로 둔다. 선수 다섯 가운데 하나라도 NaN이면 그 팀의 합과 최댓값이
    NaN이며, LightGBM이 NaN을 결측으로 읽으므로 아무것도 채우지 않는다. 마지막 두 거리 열은 B0 특징의 거리 (FEAT-5)와 같다.

    근거
    ----
    S3 계획 Task 5(b1의 열); S2 관문 S2G-4(뺄 열); 설계서 9절(기준 모형 B1); 거리 열은 (FEAT-5).
    """
    plan = _plan()
    ks, kp = _kept(plan, dropped)
    a = _arrays(s2, 'summary', 'players_num', 'geo_pp')
    idx = view_index(len(a['summary']), views)
    ns, nk = len(ks), len(kp)
    out = np.empty((len(idx), ns + 4 * nk + 2), dtype=np.float32)
    for lo, sel in _chunks(idx):
        P = _rows(a['players_num'], sel)[:, :, list(kp)].astype(np.float64)
        block = np.empty((len(sel), out.shape[1]))
        block[:, :ns] = _rows(a['summary'], sel)[:, list(ks)]
        blue, red = P[:, :5], P[:, 5:]
        for j, part in enumerate((blue.sum(axis=1), red.sum(axis=1), blue.max(axis=1), red.max(axis=1))):         # NaN stays NaN
            block[:, ns + j * nk:ns + (j + 1) * nk] = part
        block[:, -2], block[:, -1] = _geodesic(_rows(a['geo_pp'], sel))
        out[lo:lo + len(sel)] = narrow('b1', block, np.float32)
    return out


# ---- the masked M4 input ---------------------------------------------------------------------------------------------------------------
def _index_column(name: str, x, n: int, lo: int, hi: int) -> np.ndarray:
    a = np.asarray(x)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'm4_masked_inputs: {name} must be an integer array, not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'm4_masked_inputs: {name} must be 1-d, not shape {a.shape}')
    if len(a) != n:
        raise ValueError(f'm4_masked_inputs: {name} has {len(a)} entries but row_view has {n} rows')
    if a.size and (a.min() < lo or a.max() > hi):
        raise ValueError(f'm4_masked_inputs: {name} must lie in {lo} .. {hi}')
    return a.astype(np.float64)


def _count_table(name: str, x, n: int, width: int) -> np.ndarray:
    a = np.array(x, dtype=np.float64)                                                # a copy: K is divided in place below
    if a.shape != (n, width):
        raise ValueError(f'm4_masked_inputs: {name} must have shape (n, {width}) = ({n}, {width}), not {a.shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'm4_masked_inputs: {name} must be finite and not negative')
    return a


def m4_masked_inputs(row_view, q, m, z, K, O) -> np.ndarray:
    """(n, d + 25) float32: the input of the masked M4 head and the only way to build it: ``row_view`` (n, d), the features of the t0 row
    (b0 or b1, already computed; a NaN stays), then q (the onset bin, 0 .. 17), m (the mark, 0 .. 51), z (the zone, -1 .. 9; -1 = the first
    kill has no position), K (n, 4) = kills blue, kills red, gold blue, gold red (the raw gold; it is divided by 1,000 here) and O (n, 18)
    the objectives of the cells 2 kind + taker. ``m4_columns(row_names)`` names the columns. q, m and z are integer arrays; K and O are
    finite and not negative. TypeError / ValueError otherwise (naming the argument). The arguments are not changed.

    수식
    ----
    (FEAT-8)  x_i = ( rv_i (d개), q_i, m_i, z_i, k_blue, k_red, g_blue / 1000, g_red / 1000, O_i (18개) )      (n, d + 25), float32
              K_i = (k_blue, k_red, g_blue, g_red);   q ∈ {0, …, N_BINS − 1},  m ∈ {0, …, N_MARKS − 1},  z ∈ {−1, 0, …, 9}

      기호         | 뜻                                          | 단위   | 범위                    | 코드 변수
      rv_i         | t₀ 행의 특징 행(b0 또는 b1, NaN은 그대로)   | 열마다 | 유한 또는 NaN           | row_view
      q_i, m_i, z_i | 발생 구간, 표지, 첫 킬 구역(수로 넣는다)   | —      | 위 범위                 | q, m, z
      k_blue, k_red | 파랑·빨강의 킬 수                          | 개     | 유한, ≥ 0               | K[:, 0], K[:, 1]
      g_blue, g_red | 파랑·빨강의 킬 골드(원값; 1,000으로 나눔)  | 골드   | 유한, ≥ 0               | K[:, 2], K[:, 3]
      O_i          | 오브젝트 18칸(종류 × 가져간 팀)의 개수      | 개     | 유한, ≥ 0               | O
      d            | row_view의 열 수                            | 열     | ≥ 0                     | row_view.shape[1]

    해석
    ----
    식 (FEAT-8)은 가린 M4 머리가 읽는 값을 한 행렬로 만들며, t₀ 행의 특징 뒤에 앞선 사슬의 값 q, m, z와 K, O를 놓는다. 계획의 전역 제약 인과(A3)는 가린 M4 입력을
    이 함수로만 만들게 하고, 누설 시험 (a)가 그 경로를 확인한다. 어느 머리가 K, O 열을 읽는지는 머리가 정하며(계획 P6: B0의 가린 M4는 t₀ 행과 q, m, z만 읽는다),
    이 함수는 호출자가 준 값(학습의 참값이든 합성의 표본이든)을 그대로 넣는다. q, m, z는 원핫이 아니라 수로 넣고 기준 모형이 스스로 원핫이나 범주로 바꾼다. 골드는 천 단위로
    나누어 개수와 크기를 맞추며, float32로 줄일 때 범위를 검사한다.

    근거
    ----
    S3 계획 P4(K는 두 팀의 킬 수와 킬 골드, O는 오브젝트 칸), P6(B0 가린 M4의 조건), Task 5(유일한 경로), Task 7(학습의 K, O는 실제값), 전역 제약 인과(A3)(누설 시험 (a)); 설계서 3.2절(결과 Y의
    두 부분).
    """
    rv = np.asarray(row_view)
    if rv.dtype.kind not in 'fiu':
        raise TypeError(f'm4_masked_inputs: row_view must be a numeric array, not {rv.dtype}')
    if rv.ndim != 2:
        raise ValueError(f'm4_masked_inputs: row_view must be 2-d (rows x features), not shape {rv.shape}')
    n = len(rv)
    cols = [_index_column(name, x, n, lo, hi) for name, x, lo, hi in (('q', q, 0, C.N_BINS - 1), ('m', m, 0, C.N_MARKS - 1),
                                                                      ('z', z, -1, N_ZONES - 1))]
    kills = _count_table('K', K, n, 4)
    # (FEAT-8)의 g / 1000: K의 뒤 두 열(골드)만 천 단위로 나눈다. 앞 두 열은 킬 수이다.
    kills[:, 2:] /= 1000.0                                                           # gold in thousands
    objectives = _count_table('O', O, n, N_O)
    x = np.hstack([rv.astype(np.float64)] + [c[:, None] for c in cols] + [kills, objectives])
    return narrow('m4_masked_inputs', x, np.float32)


def m4_columns(row_names) -> tuple:
    """The names of the columns of ``m4_masked_inputs``: ``row_names`` (the b0 or b1 names), q, m, z, kills_blue, kills_red, gold_blue_k,
    gold_red_k and the 18 cells ``obj_<kind>_<blue|red>``. ValueError for a name that appears twice.

    (FEAT-8)의 열 순서에 맞는 이름이며, 이름이 겹치면 열이 뒤섞이므로 거부한다."""
    names = tuple(row_names) + M4_EXTRA_NAMES + O_NAMES
    twice = sorted({str(n) for n in names if names.count(n) > 1})
    if twice:
        raise ValueError('m4_columns: the column name(s) ' + ', '.join(twice) + ' appear more than once')
    return names
