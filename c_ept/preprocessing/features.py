r"""
Feature Stores and Baseline Feature Tables / S3 단계 기준 모형 및 분석용 통합 피처 생성 모듈

[English]
This module extracts baseline feature representations (B0, B1) and input matrices for evaluating
teamfight outcome models (stage S3):
  1. Data Splits & Match Weights:
       - `split_rows(s1, role)`, `split_views(s1, s2, role)`, `role_mask(s1, table, role)`:
         Partition dataset into FIT, STOP, and DEV splits.
       - `match_weights(match)`: Computes weights w_i = 1 / n_u so that each match has equal unit weight.
  2. First-Kill Tactical Zone (`unit_zone`):
       - Identifies the tactical zone (0..9) of the first kill of each engagement unit (-1 if position missing).
  3. Feature Pruning (`dropped_columns`):
       - Identifies zero-variance (constant) columns and redundant timestamps (`t_s`) on the FIT split.
  4. B0 Baseline Feature Table (`b0`, 39 columns):
       - Compact 39-dim difference vector: game time (min), phase indicators 1..7, player stat diffs (Blue - Red),
         total alive players, team geodesic distance (min/mean), objective diffs, active buffs, and objective presence.
       - Missing values are imputed with the column mean across the FIT split (`b0_fit_means`).
  5. B1 LightGBM Feature Table (`b1`, `b1_swapped`):
       - Tree-model feature table containing summary columns, team sums/maxima for player features, and distances.
       - Preserves NaNs as native missing values for gradient boosting trees.
       - `b1_swapped` makes the same table for the views seen from the other team (S4b preparation M13, decision B22: B1 is averaged over the two frames at test time).
  6. Masked M4 Head Inputs (`m4_masked_inputs`):
       - Constructs the conditioning matrix for post-teamfight conversion evaluation:
         t0 baseline features + onset bin q + first kill mark m + zone z + kill/gold counts K + 18 objective cells O.

[한국어 설명]
이 모듈은 교전 및 경기 상태 평가를 위한 머신러닝 기준 모형(B0 선형/로지스틱 회귀, B1 LightGBM)의 피처 테이블을 생성합니다:
  1. 데이터 분할 및 경기 가중치:
       - 역할(`FIT`, `STOP`, `DEV`)별 행/뷰 마스킹.
       - 경기별 가중치($w_i = 1/n_u$): 경기 내 행 수에 반비례하여 가중치를 부여, 특정 경기가 전체 평균을 왜곡하지 않도록 균등화.
  2. 첫 킬 발생 구역(`unit_zone`):
       - 교전 단위의 첫 번째 킬이 발생한 협곡 전술 구역(0~9) 또는 결측(-1).
  3. 무분산 열 제거(`dropped_columns`):
       - FIT 데이터에서 분산이 없는(상수) 열 및 중복 열(`t_s`)을 자동 식별하여 제외.
  4. B0 선형 회귀 피처 테이블(`b0`, 39차원):
       - 블루팀 - 레드팀 차분 벡터: 경기 시간, 7개 위상(Phase), 주요 8개 스탯 차이, 생존자 수, 팀 간 측지선 거리, 오브젝트/버프 차이.
       - 결측치는 FIT 세트의 평균값(`b0_fit_means`)으로 대치.
  5. B1 트리 피처 테이블(`b1`):
       - 요약 통계량 및 선수 스탯의 팀별 합계(sum)와 최댓값(max) 분리 보존. 결측치(NaN)는 LightGBM이 직접 처리.
  6. 가면 M4 입력 행렬(`m4_masked_inputs`):
       - 교전 직후의 가치 전환(Conversion) 평가를 위한 조건부 결합 행렬 생성.
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
from c_ept.valuation import flowprice
from c_ept.data.casts import narrow

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
    from c_ept.preprocessing import tokens
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


def checked_matches(s1, role, matches) -> np.ndarray:
    """Validate global match indices against authoritative roles before subsetting.

    A role is validated before any store access. No synthetic role codes are used.
    The sorted result is a copy; callers retain canonical S1/S2 row numbering.
    """
    codes = flowprice._role_codes(role)
    selected = np.asarray(matches)
    if selected.ndim != 1 or selected.dtype.kind not in 'iu':
        raise ValueError('matches must be a one-dimensional integer selection')
    roles = np.asarray(s1.array('matches/role'))
    if selected.size == 0 or not _in_roles(roles, selected, codes, 'selected matches').all():
        raise ValueError('selected matches must be nonempty and belong to the requested role')
    if len(np.unique(selected)) != len(selected):
        raise ValueError('selected matches must be distinct')
    return np.sort(selected.astype(np.int64))


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
    """
    Engagement Unit Zone Label / 교전 단위의 첫 번째 킬 발생 전술 구역 식별

    [English]
    Identifies the tactical zone (0..9 in terrain.ZONES) of the first kill of every engagement unit:
      - Reads `units/kill0`, `kills/x`, `kills/y`, and `kills/pos_missing`.
      - Returns zone ID (0..9) via `terrain.zone_of(x, y)`.
      - Returns -1 if the kill has no valid position or coordinates are NaN.

    [한국어 설명]
    S1 스토어의 모든 교전 단위에 대해 첫 번째 킬이 일어난 소환사의 협곡 전술 구역(0~9)을 구합니다:
      - `units/kill0` (단위별 첫 킬 인덱스), `kills/x`, `kills/y`, `kills/pos_missing` 조회.
      - 구역 번호(0~9, `terrain.ZONES`) 반환. 위치 정보가 없거나 결측인 경우 -1 반환.
    """
    return _unit_zone(s1)


def unit_zone_subset(s1, units) -> np.ndarray:
    """Global unit-zone array; only selected units' first-kill positions are read.

    Unselected positions are -1 placeholders and are never used as fitted rows.
    """
    return _unit_zone(s1, units=units)


def _unit_zone(s1, *, units=None) -> np.ndarray:
    from c_ept.preprocessing import terrain       # here, not at module level: the zone rules live in ``terrain``, which imports the run log
    arrays = {}
    for name in ('units/kill0', 'kills/x', 'kills/y', 'kills/pos_missing'):
        try:
            arrays[name] = s1.array(name)
        except FileNotFoundError:
            raise ValueError(f'the S1 store has no array {name}') from None
    n_kills = len(arrays['kills/x'])
    if len(arrays['kills/y']) != n_kills or len(arrays['kills/pos_missing']) != n_kills:
        raise ValueError('kills/x, kills/y and kills/pos_missing hold different numbers of kills')
    selected = None if units is None else view_index(len(arrays['units/kill0']), units)
    kill0 = np.asarray(arrays['units/kill0'] if selected is None else arrays['units/kill0'][selected]).astype(np.int64)
    if kill0.size and (kill0.min() < 0 or kill0.max() >= n_kills):
        raise ValueError(f'units/kill0 names a kill outside 0 .. {n_kills - 1}')
    if selected is None:
        zone = terrain.zone_of(np.asarray(arrays['kills/x'])[kill0], np.asarray(arrays['kills/y'])[kill0])
        return narrow('unit_zone', np.where(np.asarray(arrays['kills/pos_missing'])[kill0].astype(bool), -1, zone), np.int8)
    zone = terrain.zone_of(np.asarray(arrays['kills/x'][kill0]), np.asarray(arrays['kills/y'][kill0]))
    values = narrow('unit_zone', np.where(np.asarray(arrays['kills/pos_missing'][kill0]).astype(bool), -1, zone), np.int8)
    result = np.full(len(arrays['units/kill0']), -1, dtype=np.int8)
    result[selected] = values
    return result


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
def dropped_columns(s2, fit_views, pooled: bool = False) -> tuple:
    """The names of the summary and player columns to leave out of the baseline inputs (see the module text): those with one value in
    all the views of ``fit_views`` and ``'t_s'``. ``fit_views`` is required: the view numbers (or a bool mask over the views) of the FIT
    matches, all three sources, that is ``np.flatnonzero(split_views(s1, s2, 'FIT'))``; the store holds the views of every role it was built
    for, so there is no default that could judge them together. Reads ``x/summary`` and ``x/players_num`` in chunks. ValueError for None or
    no view.

    ``pooled`` (S4b preparation M5, decision D7): ``False`` (the default) is the list of every earlier run, each column judged alone.
    ``True`` judges the columns that the team swap exchanges on their pooled values (equation (FEAT-3p) below) so that a pair goes together or
    stays together and the list is closed under the swap; the EPT loader and the baselines use this list (``ept.data.store_constant_columns``).

    수식
    ----
    (FEAT-3)  min_c = min_{ i ∈ I_fit, p } x_{i,p,c},  max_c = max_{ i ∈ I_fit, p } x_{i,p,c}  (NaN은 건너뜀; 요약 열은 p 없이)
              열 c를 뺀다 ⇔ min_c가 NaN이거나 min_c = max_c, 또는 c = t_s
    (FEAT-3p) pooled=True:  min′_c = min(min_c, min_π(c)),  max′_c = max(max_c, max_π(c))   (요약 열; π = 팀 교환의 열 순열, symmetry.SUMMARY_SWAP)
              위치 열 c ∈ {x, y, prev_x, prev_y}:  min′_c = min(min_c, W − max_c),  max′_c = max(max_c, W − min_c)   (180° 회전 x ↦ W − x)
              열 c를 뺀다 ⇔ min′_c가 NaN이거나 min′_c = max′_c, 또는 c = t_s   (그 밖의 선수 열은 열마다 그대로)

      기호    | 뜻                                        | 단위   | 범위                    | 코드 변수
      I_fit   | 호출자가 준 보기 번호의 집합              | —      | 0, …, 보기 수 − 1       | fit_views (idx)
      c       | 요약 열 또는 선수 열                      | —      | tokens.SUMMARY, PLAYER_NUM | plan.summary_names, player_names
      p       | 선수 행(0부터)                            | —      | 0, …, 9                 | 선수 열의 축
      x       | 그 열의 값(float32)                       | 열마다 | 유한 또는 NaN           | summary, players_num
      min_c, max_c | I_fit에서의 최솟값, 최댓값(NaN 제외) | 열마다 | float32 또는 NaN        | s_lo, s_hi, p_lo, p_hi
      π(c)    | 팀 교환이 열 c와 맞바꾸는 요약 열 (제자리 열은 c 자신)  | —      | 0, …, 106               | symmetry.SUMMARY_SWAP
      W       | 지도 폭 (180° 회전의 상수)                | 맵 단위 | 14,875                  | C.W
      min′_c, max′_c | 짝(또는 회전)과 합친 최솟값, 최댓값  | 열마다 | float64 또는 NaN        | pooled 분기의 lo, hi

    해석
    ----
    식 (FEAT-3)은 FIT 보기에서 값이 하나뿐이거나 아예 없는 열과, 늘 게임 시각과 같은 중복 열 t_s를 기준 모형의 입력에서 뺀다. 비교는 NaN을 건너뛴 float32
    최솟값과 최댓값이 같은지이며, 선수 열은 선수 10명의 값을 모두 모아 한 번에 비교한다.
    식 (FEAT-3p)는 팀 교환이 서로 맞바꾸는 요약 열 한 쌍을 한 열처럼 판정한다. 한 열만 상수여도 팀 교환을 거친 뒤에는 그 자리에 다른 열의 값이 들어오므로, 짝의
    합친 값이 하나뿐일 때만 두 열을 함께 뺀다(DAT-5와 같은 이유). 위치 열은 회전 x ↦ W − x에서 짝이 자기 자신의 거울상이므로 c와 W − c를 합쳐 판정하고, 그래서 위치가
    W/2에서 상수일 때만 빠진다. 합친 판정은 짝 두 열 모두에 같은 결과를 주므로 목록은 팀 교환에 닫혀 있다. 전체 FIT 저장소에서 상수이면 그 부분집합에서도 상수이므로
    부분집합으로 판정한 합친 목록은 전체 저장소 목록을 담는다(D7 목록 ⊆ D12 목록).

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
    if pooled:
        # (FEAT-3p): the swap pairs share one judgement; fmin / fmax skip a NaN, so a pair with one empty column is judged on the other one
        from c_ept.domain import symmetry                                # here, not at module level: like ``tokens``, the S2 modules load when a baseline needs them
        pair = np.asarray(symmetry.SUMMARY_SWAP, dtype=np.int64)
        if pair.shape != (len(plan.summary_names),):
            raise RuntimeError('the summary swap permutation does not match the summary columns')
        s_lo, s_hi = np.fmin(s_lo, s_lo[pair]), np.fmax(s_hi, s_hi[pair])
        at = [plan.player_index[name] for name in ('x', 'y', 'prev_x', 'prev_y')]
        width = np.float64(C.W)
        lo64, hi64 = p_lo.astype(np.float64), p_hi.astype(np.float64)
        p_lo, p_hi = lo64.copy(), hi64.copy()
        p_lo[at], p_hi[at] = np.fmin(lo64[at], width - hi64[at]), np.fmax(hi64[at], width - lo64[at])      # the 180 degree rotation x -> W - x
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
    """
    B0 Baseline Feature Matrix (39 columns) / B0 선형 회귀 모형용 39차원 특성 행렬 생성

    [English]
    Generates the 39-column difference feature matrix (B0_NAMES) for the requested S2 views:
      - Columns 0..7: Game time (minutes) and 7 phase indicators.
      - Columns 8..15: Player stat differences (Blue team sum - Red team sum).
      - Column 16: Total alive players on both teams.
      - Columns 17..18: Min and mean geodesic walking distance between Blue and Red champions.
      - Columns 19..33: Objective score differences (towers, inhibs, plates, dragons, barons, etc.).
      - Columns 34..38: Up/available indicators for neutral objectives.
      - Imputation: NaN cells are filled using `fit_means` (the mean of each column on the FIT split).

    [한국어 설명]
    요청된 S2 뷰들에 대해 39차원 B0 특성 행렬(`B0_NAMES`)을 계산합니다:
      - 0~7열: 게임 시간(분) 및 7개 경기 페이즈 원핫 플래그.
      - 8~15열: 주요 선수 8개 스탯의 팀 간 차분(블루팀 합계 - 레드팀 합계).
      - 16열: 양 팀 전체 생존자 수.
      - 17~18열: 양 팀 선수들 간의 최소/평균 최단 보행(측지선) 거리.
      - 19~33열: 오브젝트 득실 차이(포탑, 억제기, 채굴, 용, 바론 등).
      - 34~38열: 중립 오브젝트 현재 출현 여부.
      - 결측치 처리: 결측 셀(NaN)은 `fit_means`(FIT 세트의 열별 평균값)로 대치됩니다.
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
            # (FEAT-6) imputation: only replace NaNs with FIT split means
            raw = np.where(nan, means[None, :], raw)
            filled += nan.sum(axis=0)
        out[lo:lo + len(raw)] = narrow('b0', raw, np.float32)
    if counts is not None:
        counts += filled.astype(counts.dtype)
    return out


# ---- b1 ----------------------------------------------------------------------------------------------------------------------------------
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
    """The names of the b1 columns when the columns ``dropped`` are left out (see the module text for the order)."""
    plan = _plan()
    ks, kp = _kept(plan, dropped)
    players = tuple(plan.player_names[i] for i in kp)
    return (tuple(plan.summary_names[i] for i in ks)
            + tuple(f'{how}_{team}_{c}' for how in ('sum', 'max') for team in ('blue', 'red') for c in players) + ('geo_min', 'geo_mean'))


def b1(s2, views, dropped) -> np.ndarray:
    """
    B1 LightGBM Baseline Feature Matrix / B1 트리 모델(LightGBM)용 피처 행렬 생성

    [English]
    Generates the feature matrix for LightGBM baseline models (b1_columns):
      - Retained summary columns (after excluding `dropped`).
      - Team sums and maxima (sum_blue, sum_red, max_blue, max_red) for retained player stats.
      - Min and mean geodesic distances between teams.
      - NaNs are kept untouched (LightGBM natively branches on missing values).

    [한국어 설명]
    LightGBM 기준 모형을 위한 피처 행렬을 생성합니다:
      - 분산이 없어 제거된 열(`dropped`)을 제외한 잔존 요약 통계량.
      - 선수 스탯의 블루/레드 팀별 합계(sum) 및 최댓값(max).
      - 양 팀 간 최소 및 평균 측지선 보행 거리.
      - 결측치(NaN)는 LightGBM이 직접 처리하므로 대치하지 않고 그대로 유지합니다.
    """
    return _b1_rows(s2, views, dropped, False)


def b1_swapped(s2, views, dropped) -> np.ndarray:
    """
    B1 Feature Matrix of the Team-Swapped Views / 팀을 맞바꾼 보기의 B1 특징 행렬 (S4b 준비 M13, 결정 B22)

    [English]
    The ``b1`` rows (same columns, same dtype, same order of the views) of the views as they look from the other side: the summary columns that the team swap exchanges are exchanged
    (``symmetry.SUMMARY_SWAP``), the player rows change places (blue 0-4 with red 5-9, ``symmetry.PLAYER_ORDER``), the four position columns turn by 180 degrees (x -> W - x) and the
    geodesic table follows the players. The views are not written anywhere: the answer equals ``b1`` of a store whose views went through ``symmetry.swap_view`` in every bit (a test
    checks it), and ``b1_swapped`` of such a store is ``b1`` of the original. With ``b1`` it makes the two frames over which B1 is averaged at test time (decision B22).

    [한국어 설명]
    각 보기를 반대편에서 본 것으로 바꾼 뒤 만드는 ``b1`` 행이다. 요약 열은 팀 교환이 맞바꾸는 짝끼리 바꾸고, 선수 행은 파랑과 빨강이 자리를 바꾸며, 위치 네 열은 180도 돌리고(x -> W - x),
    측지선 표는 선수 순서를 따른다. 팀별 합과 최댓값은 파랑·빨강 자리가 서로 바뀌어 나오고, 팀 사이 거리(geo_min, geo_mean)는 측지선 표가 대칭이면 그대로이다(표를 다시 만들지 않고 선수 순서만 따른다).

    수식
    ----
    b1_swapped(v) = b1(Π v) ;  Π = (요약 열 SUMMARY_SWAP, 선수 행 PLAYER_ORDER, x ↦ W − x, 측지선 행·열 PLAYER_ORDER) ;  Π(Π v) = v

    해석
    ----
    B1은 표의 특징으로만 예측하므로 EPT의 프레임 평균(``FrameAveraged``)과 같은 평균을 받으려면 맞바꾼 틀의 특징을 따로 만들어야 한다. 변환은 ``symmetry.swap_view``의 것을 그대로 쓴다(요약 열과
    선수 행의 순열, 위치의 회전). 열이 상수라서 뺀 짝 열의 값은 맞바꾼 틀에서 남은 짝 열에 들어온다(``dropped_columns(pooled=True)``의 목록은 짝을 함께 빼므로 닫혀 있다).
    """
    return _b1_rows(s2, views, dropped, True)


def swap_unclosed_columns(dropped) -> tuple:
    """The pairs ``(dropped column, kept partner)`` of the summary columns that the team swap exchanges (``symmetry.SUMMARY_SWAP``) when one member of the pair is in ``dropped`` and the other is not: ``()`` when
    the list is closed under the swap. ValueError for a name that is no column of the view. S4b preparation M13 (decision B22), the check of ``run_inference(b1_swap_average=True)``.

    수식
    ----
    닫히지 않은 쌍 = {(c, π(c)) : c ∈ dropped, π(c) ∉ dropped},   π = SUMMARY_SWAP (요약 열의 팀 교환 순열)

    해석
    ----
    맞바꾼 틀에서는 열 π(c)의 값이 열 c 자리에 들어온다. c가 FIT에서 상수라서 뺐고 짝 π(c)는 남았다면, 남은 열 π(c)가 맞바꾼 틀에서 FIT에서 한 번도 보지 않은 상수 값을 받아 B1의 맞바꾼 틀 예측이
    외삽이 되고, 평균은 여전히 정확히 불변이지만 B1이 말없이 나빠진다. 짝을 함께 판정한 목록(``dropped_columns(pooled=True)``)은 닫혀 있다. 선수 열은 모든 선수에서 같은 이름으로 빠지고 팀 교환은
    그 열의 이름을 바꾸지 않으므로(위치 열은 회전할 뿐) 늘 닫혀 있다.
    """
    from c_ept.domain import symmetry                                    # here, not at module level: like ``tokens``, the S2 modules load when a baseline needs them
    plan = _plan()
    _kept(plan, dropped)                                                 # refuses a name that is no column of the view
    drop = set(dropped)
    pair = symmetry.SUMMARY_SWAP
    return tuple((name, plan.summary_names[int(pair[i])]) for i, name in enumerate(plan.summary_names)
                 if name in drop and plan.summary_names[int(pair[i])] not in drop)


def _b1_rows(s2, views, dropped, swapped: bool) -> np.ndarray:
    """The b1 rows of ``views``; ``swapped`` reads each view from the other side (see ``b1_swapped``). The default path is the arithmetic of every earlier run."""
    plan = _plan()
    ks, kp = _kept(plan, dropped)
    a = _arrays(s2, 'summary', 'players_num', 'geo_pp')
    idx = view_index(len(a['summary']), views)
    ns, nk = len(ks), len(kp)
    out = np.empty((len(idx), ns + 4 * nk + 2), dtype=np.float32)
    if swapped:
        from c_ept.domain import symmetry                                # here, not at module level: like ``tokens``, the S2 modules load when a baseline needs them
        turned = [plan.player_index[name] for name in ('x', 'y', 'prev_x', 'prev_y')]
    for lo, sel in _chunks(idx):
        summary, players, geo = _rows(a['summary'], sel), _rows(a['players_num'], sel), _rows(a['geo_pp'], sel)
        if swapped:
            order = symmetry.PLAYER_ORDER
            summary = summary[:, symmetry.SUMMARY_SWAP]
            players = players[:, order]                                  # a copy (fancy indexing): the stored views are not touched
            players[:, :, turned] = symmetry.W - players[:, :, turned]   # the rotation x -> W - x of ``symmetry.swap_view``
            geo = geo[:, order][:, :, order]
        P = players[:, :, list(kp)].astype(np.float64)
        block = np.empty((len(sel), out.shape[1]))
        block[:, :ns] = summary[:, list(ks)]
        blue, red = P[:, :5], P[:, 5:]
        for j, part in enumerate((blue.sum(axis=1), red.sum(axis=1), blue.max(axis=1), red.max(axis=1))):         # NaN stays NaN
            block[:, ns + j * nk:ns + (j + 1) * nk] = part
        block[:, -2], block[:, -1] = _geodesic(geo)
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
    """
    Masked M4 Conditioning Matrix Builder / 사후 가치 전환(M4) 평가를 위한 조건부 입력 행렬 조합

    [English]
    Builds the conditioning input matrix for the masked M4 model:
      - `row_view` (n, d): Baseline features of the onset timestamp (b0 or b1).
      - `q` (n,): Onset temporal bin (0..17, 10s bins within the 3-minute window).
      - `m` (n,): First-kill mark index (0..51, indicating who died first or execution).
      - `z` (n,): Tactical zone of the first kill (-1..9).
      - `K` (n, 4): Raw combat stats (Blue kills, Red kills, Blue gold / 1000, Red gold / 1000).
      - `O` (n, 18): Post-fight objective outcomes (towers, dragons, barons taken).

    [한국어 설명]
    교전 종료 후의 가치 전환(Post-fight Conversion)을 평가하는 M4 모델의 입력 행렬을 조합합니다:
      - `row_view`: 교전 시작 시점의 베이스라인 피처 행렬 (`b0` 또는 `b1`).
      - `q`: 교전 발생 시점의 10초 단위 구간 인덱스 (0~17).
      - `m`: 첫 킬 사망자 표지 마크 (0~51).
      - `z`: 첫 킬 발생 전술 구역 (-1~9).
      - `K`: 양 팀 킬 수 및 킬 골드 (골드는 1,000으로 나누어 정규화).
      - `O`: 교전 후 획득한 18개 셀의 오브젝트 결과.
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
