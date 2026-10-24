r"""EPT의 자료 층 (S4a 과제 1): S2 보기를 신경망 입력 텐서로 바꾸고, 두 팀을 맞바꾸고, 배치를 만든다.

이 모듈은 설계서 4절(두 시계와 가림)과 5.2절(팀 대칭: 180° 회전과 진영 표시, 출력도 같이 교환), S4a 계획의 결정 Q4(행과 정답은 head_rows에서
그대로)·Q5(FIT 표준화, 허용된 NaN은 0과 가림 채널, 프레임 나이·경과 시간·보기 종류)·Q8(한 교전의 가린 보기와 열린 보기는 같은 배치)과 검토 초점
1(팀 교환의 반쪽 적용)·3(NaN 입력)을 구현한다. "보기"는 S2 저장소의 한 행(기준 시각 하나에서 본 선수 10명, 표지점 34개, 최근 사건과 간선,
요약)이다. "가린 행"은 프레임 시계 보기(기준 시각 t₀, t₀ 이전 정보만)를 읽는 행이고, "열린 행"은 교전의 사건 시계 보기(M3는 첫 킬 f, M4는 교전 끝 δ에서 본
보기)를 읽는 행이다. 이 모듈은 킬이 하나 이상 난 교전을 다룬다 (G-S3c C3).

식 (식 번호는 ``(DAT-n)``)

    (DAT-1)  x̃ = (x − x₀) / σ ;  x₀ = x̄ (FIT 평균), 위치 열은 x₀ = W/2 ;  σ = √(D / n), 값이 하나뿐인 열과 값이 없는 열은 σ = 1
    (DAT-2)  ν = 1[x가 NaN] ;  x가 NaN이면 x̃ = 0   (허용된 NaN 칸마다 가림 채널 ν)
    (DAT-3)  n = n₁ + n₂ ;  x̄ = x̄₁ + (x̄₂ − x̄₁) n₂ / n ;  D = D₁ + D₂ + (x̄₂ − x̄₁)² n₁ n₂ / n   (Chan 병합)
    (DAT-4)  위치 열의 척도: σ² = D / n + (x̄ − W/2)²   (평균이 아니라 회전 중심 W/2에서 잰 제곱평균)
    (DAT-5)  팀 교환이 맞바꾸는 두 요약 열 h, h′는 (n, x̄, D)를 (DAT-3)으로 합쳐 같은 (x₀, σ)를 쓴다.
    (DAT-6)  팀 교환: x ↦ W − x, x̃ ↦ −x̃ (위치 열) ;  ξ ↦ (ξ + 5) mod 10 ;  m ↦ (m + 25) mod 50 (m < 50) 또는 101 − m (m ≥ 50) ;
             z ↦ ZONE_PERM[z] (z ≥ 0) ;  K ↦ (K₂, K₁, K₄, K₃) ;  O는 두 팀 칸을 맞바꿈 ;  side ↦ 1 − side
    (DAT-7)  대리 경기: M′ = max(1, ⌈ζ M⌉), 시드 순열 ω_{seed,role}로 섞은 경기 열의 앞 M′개 (작은 몫이 큰 몫 안에 든다).
    (DAT-8)  배치: J = ⌈N / L⌉ ;  교전 열과 프레임 행 열을 ω_{seed,epoch}로 섞어 J덩어리로 나눈다.

기호 (단위와 범위는 식을 계산하는 함수의 표에 있다)

    x, x̃  표의 한 칸의 값(모형 입력 x의 한 성분), 표준화한 값      x₀, σ  중심, 척도        ν  가림 표시      W = 14,875  지도 폭 (맵 단위)
    n, x̄, D  NaN을 뺀 개수, 평균, 평균에서 잰 제곱합              h, h′  요약 열과 팀 교환이 짝지우는 열      ξ  선수 번호 (0–4 파랑, 5–9 빨강)
    M, M′, ζ  경기 수, 뽑은 경기 수, 몫      N, L, J  교전 수, 배치당 교전 수, 에폭의 배치 수      ω  시드로 만든 임의 순열
    m, z, K, O, f, δ, t₀  식 색인 기호 등록부의 뜻 그대로

가정과 선택

    · 표준화 통계는 FIT 경기의 모든 보기(프레임 시계, M3, M4)에서만 잰다 (R14). 다른 역할의 값은 통계에 들어가지 않는다.
    · 팀 교환은 입력과 정답을 함께 바꾼다. 입력만 바꾸면 쌍 표지, 팀별 K·O, 구역 정답이 뒤바뀐다 (검토 초점 1). 모르는 키는 오류로 멈추므로 팀에
      매인 배열이 바뀌지 않은 채 지나가는 일이 없다.
    · 팀 교환이 입력의 정확한 변환이 되도록 표준화를 고른다 (R-S4a-3): 위치 열은 평균이 아니라 W/2에서 재고 (DAT-4), 팀 교환이 맞바꾸는 두 요약
      열은 통계를 나눠 쓴다 (DAT-5). 그래서 표준화한 보기에 swap을 쓴 결과는 원자료 보기를 symmetry.swap_view로 맞바꾸고(표지점 행과 geo_pl
      열은 LANDMARK_PERM으로 옮긴다) 표준화한 결과와 0의 부호 비트까지 같다. 위치가 정수이고 부호를 0.0 − x̃로 바꿀 때이며, 시험이 확인한다.
    · 요약의 t_s는 game_time_s와 항상 같은 짝 열이라 입력에서 뺀다 (tokens 모듈 머리). 그 밖에 FIT에서 상수인 열은 입력에 남고 척도만 1이다.
    · 허용된 칸 밖의 NaN, 한 좌표만 빈 위치, 모든 무한대는 오류로 멈춘다 (검토 초점 3). 그래서 신경망에 NaN이 흘러들 수 없다.
    · 사건과 간선의 개수는 저장소 목록(STORE.json)의 모양에서 읽는다. 코드에 적어 두지 않는다 (판정 R-S4a-1).

입력과 출력

    fit_stats(s1, s2)             → Stats (JSON과 해시로 저장)
    ViewTensors(s2, views, stats) → dict[str, torch.Tensor] (VIEW_KEYS)
    swap(batch)                   → 팀을 맞바꾼 새 배치 (입력과 정답을 함께; 표는 MARK_PERM 등으로 내보낸다)
    proxy_matches(s1)             → FIT와 STOP의 대리 경기
    ept_head_rows, EngagementRows, Batches  → head_rows의 행과 정답을 교전 단위로 묶어 섞은 배치 (ROW_FIELDS, row_key, block)
"""
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 4 (two clocks and masking) and 5.2 (symmetry: rotation by 180 degrees plus
#      a side marker, outputs swapped with it).
# Ref: docs/superpowers/plans/2026-10-22-c-ept-s4a.md decisions Q4, Q5, Q8, Q10 and review focus 1 and 3.
# Ref: Che, Z., Purushotham, S., Cho, K., Sontag, D., Liu, Y. (2018), "Recurrent Neural Networks for Multivariate Time Series with Missing
#      Values", Scientific Reports 8:6085, doi:10.1038/s41598-018-24271-9 (a mask and an elapsed time next to a zero-filled input).
# Ref: Chan, T. F., Golub, G. H., LeVeque, R. J. (1983), "Algorithms for Computing the Sample Variance: Analysis and Recommendations",
#      The American Statistician 37(3):242-247, doi:10.1080/00031305.1983.10483115 (merging counts, means and sums of squares: DAT-3).
#      Bibliographic data read from the DOI record (Crossref API) on 2026-10-22.
# Ref: ruling R14 (statistics on FIT only), R18 (rows and targets from head_rows), R-S4a-1 (shapes from the manifest), R-S4a-3 (the centre W/2
#      of the position columns and the pooled statistics of the exchanged summary columns, so that standardising commutes with the swap).
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import torch

from c_ept import config as C
from c_ept import features, head_rows, runlog, symmetry, terrain, tokens
from c_ept.casts import narrow
from c_ept.store_core import write_atomic

# ---- the columns -----------------------------------------------------------------------------------------------------------------------
W = float(C.W)                                   # the rotation constant: (x, y) -> (W - x, W - y)
CENTRE = W / 2.0                                 # the fixed point of that rotation
N_PLAYERS, N_LANDMARKS = tokens.N_PLAYERS, tokens.N_LANDMARKS
KIND_FRAME, KIND_M3, KIND_M4 = 0, 1, 2           # the kind of a view: ``views/src`` (frame clock, M3 event clock, M4 event clock)
PLAYER_COLUMNS = tuple(tokens.PLAYER_NUM)
POSITION_COLUMNS = ('x', 'y', 'prev_x', 'prev_y')                      # the player columns that the rotation turns
_POSITION_AT = np.array([PLAYER_COLUMNS.index(name) for name in POSITION_COLUMNS], dtype=np.int64)
_NAN_PLAYER = frozenset(tokens.NAN_ALLOWED['players_num']['cells'])
PLAYER_NAN_COLUMNS = tuple(name for name in PLAYER_COLUMNS if name in _NAN_PLAYER)       # the columns that get a mask channel
_PLAYER_NAN_AT = np.array([PLAYER_COLUMNS.index(name) for name in PLAYER_NAN_COLUMNS], dtype=np.int64)
_PLAYER_NAN_OK = np.zeros(len(PLAYER_COLUMNS), dtype=bool)
_PLAYER_NAN_OK[_PLAYER_NAN_AT] = True
DROPPED_SUMMARY = ('t_s',)                       # the twin of game_time_s (tokens module text): not an input
SUMMARY_COLUMNS = tuple(name for name in tokens.SUMMARY if name not in DROPPED_SUMMARY)
_SUMMARY_AT = np.array([i for i, name in enumerate(tokens.SUMMARY) if name not in DROPPED_SUMMARY], dtype=np.int64)
_NAN_SUMMARY = frozenset(tokens.NAN_ALLOWED['summary']['cells'])
SUMMARY_NAN_COLUMNS = tuple(name for name in SUMMARY_COLUMNS if name in _NAN_SUMMARY)
_SUMMARY_NAN_AT = np.array([SUMMARY_COLUMNS.index(name) for name in SUMMARY_NAN_COLUMNS], dtype=np.int64)
_SUMMARY_NAN_OK = np.array([name in _NAN_SUMMARY for name in tokens.SUMMARY], dtype=bool)      # over the 107 columns of the store
_AGE_AT = SUMMARY_COLUMNS.index('frame_age_s')
_LANDMARK_NAN_OK = np.zeros((N_LANDMARKS, 2), dtype=bool)
for _row, _col in tokens.NAN_ALLOWED['landmarks']['cells']:
    _LANDMARK_NAN_OK[_row, _col] = True
if (len(PLAYER_NAN_COLUMNS), len(SUMMARY_NAN_COLUMNS), len(SUMMARY_COLUMNS), len(PLAYER_COLUMNS)) != (23, 3, 106, 81) or not (
        _LANDMARK_NAN_OK[:, 0].sum() == 0 and _LANDMARK_NAN_OK[:, 1].sum() == 4):
    raise RuntimeError('the NaN cells or the columns of the S2 views are not those that this module was written for')

# ---- what the swap exchanges -----------------------------------------------------------------------------------------------------------
N_CELLS = 2 * len(C.O_KINDS)                     # the cells of O and of the conversions: 2 * kind + taker
PLAYER_PERM = np.asarray(symmetry.PLAYER_ORDER, dtype=np.int64)       # player ξ in a swapped view reads PLAYER_PERM[ξ] (DAT-6)
LANDMARK_PERM = np.asarray(symmetry.LANDMARK_SWAP, dtype=np.int64)     # the landmark rows and the geo_pl columns (swap_view leaves them)
MARK_PERM = np.array([symmetry.swap_mark(m) for m in range(C.N_MARKS)], dtype=np.int64)          # the 52 marks
ZONE_PERM = np.asarray(terrain.ZONE_SWAP, dtype=np.int64)              # the ten zones; -1 (no position) stays -1
K_PERM = np.array([1, 0, 3, 2], dtype=np.int64)  # K = kills blue, kills red, kill gold blue, red (thousands): the teams change places
TEAM_PERM = np.array([1, 0], dtype=np.int64)     # a column for each team (the kills of M3Kills)
CELL_PERM = np.arange(N_CELLS, dtype=np.int64) ^ 1                     # O and the conversions: blue and red taker of every kind


def _summary_perm() -> np.ndarray:
    """The swap of the 106 summary columns: ``symmetry``'s permutation of the 107 columns, cut down to the columns that are kept.

    해석: 팀 교환이 요약의 파랑·빨강 열을 맞바꾸는 순열을 대칭 기준 모듈에서 가져와, 입력에서 뺀 열을 건너뛴 번호로 옮긴다. 뺀 열이 다른 열과
    맞바뀌면 그 열의 값이 입력에 새어 드는 셈이므로 오류로 멈춘다.
    """
    full = np.asarray(symmetry.SUMMARY_SWAP, dtype=np.int64)
    where = {int(old): new for new, old in enumerate(_SUMMARY_AT)}
    dropped = [i for i in range(len(full)) if i not in where]
    if any(full[i] != i for i in dropped):
        raise RuntimeError('a dropped summary column is exchanged by the swap with another column')
    return np.array([where[int(full[old])] for old in _SUMMARY_AT], dtype=np.int64)


SUMMARY_PERM = _summary_perm()                  # summary column h of a swapped view reads SUMMARY_PERM[h] (DAT-5; its own inverse)


def _check_tables() -> None:
    """Every table is a permutation that is its own inverse, and the cells that may be NaN lie where the swap leaves them.

    해석: 표가 순열이 아니거나 두 번 적용해도 제자리로 오지 않으면 팀 교환이 자기 역이 아니게 되어 프레임 평균이 깨지므로 모듈을 가져올 때 한 번
    검사한다. NaN이 설 수 있는 요약 열이 제자리에 있어야 하는 것은 그 가림 채널을 순열 없이 그대로 두기 때문이다. NaN이 설 수 있는 표지점
    행이 제자리에 있어야 하는 것은 NaN 규칙이 행 번호로 칸을 지목하므로 교환한 보기도 같은 칸에서만 NaN을 갖게 하려는 것이다 (그 가림 채널은
    행과 함께 옮긴다).
    """
    for name, perm, size in (('PLAYER_PERM', PLAYER_PERM, N_PLAYERS), ('LANDMARK_PERM', LANDMARK_PERM, N_LANDMARKS),
                             ('SUMMARY_PERM', SUMMARY_PERM, len(SUMMARY_COLUMNS)), ('MARK_PERM', MARK_PERM, C.N_MARKS),
                             ('ZONE_PERM', ZONE_PERM, len(terrain.ZONES)), ('K_PERM', K_PERM, 4), ('TEAM_PERM', TEAM_PERM, 2),
                             ('CELL_PERM', CELL_PERM, N_CELLS)):
        if sorted(perm.tolist()) != list(range(size)) or not np.array_equal(perm[perm], np.arange(size)):
            raise RuntimeError(f'{name} is not a permutation that is its own inverse')
    if not np.array_equal(SUMMARY_PERM[_SUMMARY_NAN_AT], _SUMMARY_NAN_AT):
        raise RuntimeError('a summary column that may be NaN is exchanged by the swap: its mask channel would need the permutation')
    nan_rows = np.flatnonzero(_LANDMARK_NAN_OK.any(axis=1))
    if not np.array_equal(LANDMARK_PERM[nan_rows], nan_rows):
        raise RuntimeError('a landmark row that may be NaN is exchanged by the swap')


_check_tables()


# ---- the layout of an S2 store ---------------------------------------------------------------------------------------------------------
_READ = ('players_num', 'players_zone', 'landmarks', 'ev_type', 'ev_dt', 'ev_team', 'ev_xy', 'ev_actors', 'ev_gold', 'ev_n',
         'ed_i', 'ed_j', 'ed_dt', 'ed_kind', 'ed_n', 'summary', 'geo_pp', 'geo_pl')     # the arrays of a view that this module reads
_FLOATS = ('players_num', 'landmarks', 'ev_dt', 'ev_xy', 'ev_gold', 'ed_dt', 'summary', 'geo_pp', 'geo_pl')
_NO_NAN = ('ev_dt', 'ev_gold', 'ed_dt')          # float arrays that may hold no NaN at all
_TAIL = {'players_num': (N_PLAYERS, len(tokens.PLAYER_NUM)), 'players_zone': (N_PLAYERS,), 'landmarks': (N_LANDMARKS, 2),
         'summary': (len(tokens.SUMMARY),), 'geo_pp': (N_PLAYERS, N_PLAYERS), 'geo_pl': (N_PLAYERS, N_LANDMARKS), 'ev_n': (), 'ed_n': ()}


def _layout(s2) -> SimpleNamespace:
    """``SimpleNamespace(n_views, events, edges)`` of an S2 store: the shapes of its arrays, read from the manifest (STORE.json) when the
    store has one and from the arrays otherwise. ValueError naming the array for a missing array, a shape that is not that of the
    view arrays (``tokens.VIEW_SPEC``), or arrays that hold different numbers of views, events or edges. Nothing but shapes is read.

    해석: 사건 수와 간선 수를 코드에 적지 않고 저장소 목록에서 읽어, 저장소가 바뀌어도 같은 코드가 맞게 한다 (판정 R-S4a-1).
    """
    manifest = getattr(s2, 'manifest', None)
    listed = manifest.get('arrays', {}) if isinstance(manifest, Mapping) else {}
    shapes = {}
    for name in _READ:
        entry = listed.get(f'x/{name}')
        try:
            shapes[name] = tuple(int(v) for v in entry['shape']) if entry is not None else tuple(s2.array(f'x/{name}').shape)
        except FileNotFoundError:
            raise ValueError(f'the S2 store has no array x/{name}') from None
    n = shapes['players_num'][0]
    events, edges = shapes['ev_type'][1] if len(shapes['ev_type']) == 2 else -1, shapes['ed_i'][1] if len(shapes['ed_i']) == 2 else -1
    want = {**{name: (n,) + tail for name, tail in _TAIL.items()},
            **{name: (n, events) for name in ('ev_type', 'ev_dt', 'ev_team', 'ev_actors', 'ev_gold')}, 'ev_xy': (n, events, 2),
            **{name: (n, edges) for name in ('ed_i', 'ed_j', 'ed_dt', 'ed_kind')}}
    for name, shape in want.items():
        if shapes[name] != shape:
            raise ValueError(f'x/{name} must have shape {shape} (views, then the shape of the view array), not {shapes[name]}')
    src = tuple(s2.array('views/src').shape)
    if src != (n,):
        raise ValueError(f'views/src must have one entry for each of the {n} views, not shape {src}')
    return SimpleNamespace(n_views=n, events=events, edges=edges)


def _check_cells(a: dict, layout: SimpleNamespace) -> None:
    """Stop with a ValueError that names the array unless the rows read in ``a`` obey the rules of ``tokens``: NaN only in the allowed
    cells, a position (x and y, prev_x and prev_y, an event position) NaN in both coordinates or in none, no infinity, indices in range,
    and the padding marks (``ev_n``, ``ed_n``) in agreement with the token types.

    해석: 허용된 칸 밖의 NaN이나 무한대는 S2의 오류이며, 그대로 표준화하면 신경망의 손실과 기울기가 NaN이 되므로 읽은 자리에서 막는다 (검토 초점 3).
    위치는 두 좌표가 함께 비거나 함께 차야 하고, 한쪽만 빈 위치는 가림 채널과 통계가 반쪽 위치를 세게 하므로 막는다. 채움 행 표시가 토큰 종류와
    맞는지 보는 것은 채움 행이 진짜 사건으로 들어가거나 그 반대가 되는 것을 막기 위해서이다. 메시지에는 배열과 열의 이름만 있고 경기 키는 없다.
    """
    for name in _FLOATS:
        if np.isinf(a[name]).any():
            raise ValueError(f'{name}: an infinite value')
    for name in _NO_NAN:
        if np.isnan(a[name]).any():
            raise ValueError(f'{name}: NaN, where a value must exist')
    stray = np.isnan(a['players_num']) & ~_PLAYER_NAN_OK
    if stray.any():
        cols = ', '.join(PLAYER_COLUMNS[j] for j in np.flatnonzero(stray.any(axis=(0, 1))))
        raise ValueError(f'players_num: NaN in {cols}, where a value must exist')
    for cx, cy in zip(POSITION_COLUMNS[0::2], POSITION_COLUMNS[1::2]):             # a position is missing in both coordinates or in none
        if (np.isnan(a['players_num'][..., PLAYER_COLUMNS.index(cx)]) != np.isnan(a['players_num'][..., PLAYER_COLUMNS.index(cy)])).any():
            raise ValueError(f'players_num: exactly one of {cx} and {cy} is NaN, but a missing position is NaN in both')
    if (np.isnan(a['ev_xy'][..., 0]) != np.isnan(a['ev_xy'][..., 1])).any():
        raise ValueError('ev_xy: exactly one of the two coordinates is NaN, but a missing position is NaN in both')
    stray = np.isnan(a['summary']) & ~_SUMMARY_NAN_OK
    if stray.any():
        cols = ', '.join(tokens.SUMMARY[j] for j in np.flatnonzero(stray.any(axis=0)))
        raise ValueError(f'summary: NaN in {cols}, where a value must exist')
    if (np.isnan(a['landmarks']) & ~_LANDMARK_NAN_OK).any():
        raise ValueError('landmarks: NaN in a cell where a value must exist')
    for name, low, high in (('players_zone', -1, len(terrain.ZONES) - 1), ('ev_type', -1, len(C.EVENT_TYPES) - 1), ('ev_team', 0, 2),
                            ('ev_actors', 0, 1023), ('ed_i', -1, N_PLAYERS - 1), ('ed_j', -1, N_PLAYERS - 1),
                            ('ed_kind', -1, len(tokens.EDGE_KINDS) - 1), ('ev_n', 0, layout.events), ('ed_n', 0, layout.edges)):
        if a[name].size and (a[name].min() < low or a[name].max() > high):
            raise ValueError(f'{name}: values must lie in {low} .. {high}')
    ev_valid = np.arange(layout.events)[None, :] < a['ev_n'][:, None]
    if not np.array_equal(ev_valid, a['ev_type'] >= 0):
        raise ValueError('ev_n and ev_type disagree about which tokens are padding')
    ed_valid = np.arange(layout.edges)[None, :] < a['ed_n'][:, None]
    if not all(np.array_equal(ed_valid, a[name] >= 0) for name in ('ed_i', 'ed_j', 'ed_kind')):
        raise ValueError('ed_n, ed_i, ed_j and ed_kind disagree about which edges are padding')


def _read_checked(s2, idx: np.ndarray, layout: SimpleNamespace) -> dict:
    """The rows ``idx`` of the arrays in ``_READ`` as new numpy arrays (their stored dtypes), after ``_check_cells``."""
    a = {name: np.asarray(s2.array(f'x/{name}')[idx]) for name in _READ}
    _check_cells(a, layout)
    return a


# ---- the statistics of the FIT views ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ColumnStats:
    """The centre x₀, the scale σ and the count n of every column of one table of the views (float64, float64, int64), so that
    x̃ = (x − x₀) / σ (DAT-1). The centre is the mean of the FIT values, except for the position columns, whose centre is W/2 (DAT-4).
    A column without a value has the scale 1 and the count 0, and the centre 0 (W/2 for a position column)."""
    centre: np.ndarray
    scale: np.ndarray
    count: np.ndarray

    def __post_init__(self):
        centre, scale, count = (np.asarray(a) for a in (self.centre, self.scale, self.count))
        if not (centre.ndim == 1 and centre.shape == scale.shape == count.shape):
            raise ValueError('centre, scale and count must be 1-d arrays of one length')
        if not (np.isfinite(centre).all() and np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError('every centre must be finite and every scale finite and positive')


@dataclass(frozen=True, eq=False)
class Stats:
    """The standardisation statistics of the EPT inputs, computed on the FIT views only: ``arrays`` maps a table (``players_num``,
    ``summary``, ``landmarks``, ``geo_pp``, ``geo_pl``, ``ev_dt``, ``ev_gold``, ``ev_xy``) to its ``ColumnStats``, ``n_views`` is the number
    of FIT views read and ``role`` the role they came from, which can only be FIT (ruling R14): ValueError for another role, when a
    ``Stats`` is made and when a file is loaded. ``sha256`` is the hash of the content (``runlog.sha256_json``); ``save`` writes it into
    the JSON file and ``load`` refuses a file whose content does not give the hash it states."""
    arrays: Mapping[str, ColumnStats]
    n_views: int
    role: str = 'FIT'

    def __post_init__(self):
        if self.role != 'FIT':
            raise ValueError(f'the statistics are those of the FIT views only (ruling R14), not of {self.role!r}')

    def _body(self) -> dict:
        return {'version': 1, 'kind': 'ept_view_stats', 'role': self.role, 'n_views': int(self.n_views), 'centre': CENTRE,
                'columns': {'players_num': list(PLAYER_COLUMNS), 'summary': list(SUMMARY_COLUMNS)},
                'arrays': {name: {'centre': cs.centre.tolist(), 'scale': cs.scale.tolist(), 'count': cs.count.tolist()}
                           for name, cs in sorted(self.arrays.items())}}

    @property
    def sha256(self) -> str:
        return runlog.sha256_json(self._body())

    def to_json(self) -> dict:
        """The content and its hash (key ``sha256``) as a JSON-ready dict."""
        return {**self._body(), 'sha256': self.sha256}

    def save(self, path) -> str:
        """Write the JSON (atomically) to ``path`` and return the hash."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, json.dumps(self.to_json(), indent=1, ensure_ascii=False) + '\n')
        return self.sha256

    @classmethod
    def load(cls, path) -> Stats:
        """Read the JSON of ``save``. ValueError if the content does not give the stated hash, if it is not a statistics file, if its role
        is not FIT, or if its column lists differ from the columns this code reads (statistics of another layout)."""
        js = json.loads(Path(path).read_text(encoding='utf-8'))
        stated = js.get('sha256') if isinstance(js, dict) else None
        body = {k: v for k, v in js.items() if k != 'sha256'} if isinstance(js, dict) else None
        if body is None or js.get('kind') != 'ept_view_stats' or runlog.sha256_json(body) != stated:
            raise ValueError('the file is not an EPT statistics file, or its content does not give the hash that it states')
        if js['columns'] != {'players_num': list(PLAYER_COLUMNS), 'summary': list(SUMMARY_COLUMNS)}:
            raise ValueError('the statistics belong to another set of player or summary columns than this code reads')
        arrays = {name: ColumnStats(np.array(v['centre'], dtype=np.float64), np.array(v['scale'], dtype=np.float64),
                                    np.array(v['count'], dtype=np.int64)) for name, v in js['arrays'].items()}
        return cls(arrays=arrays, n_views=int(js['n_views']), role=js['role'])


def _merge(n_a, mean_a, m2_a, n_b, mean_b, m2_b) -> tuple:
    """Merge the (count, mean, sum of squares about the mean) of two parts, column by column (DAT-3).

    해석: (DAT-3)의 병합을 계산한다. 한 덩어리가 비면 다른 덩어리가 비트 단위로 그대로 남고, 둘 다 비면 평균 0, 제곱합 0이다.
    """
    n_a, n_b = np.asarray(n_a, dtype=np.float64), np.asarray(n_b, dtype=np.float64)
    n = n_a + n_b
    delta = mean_b - mean_a
    safe = np.maximum(n, 1.0)
    # a part without a value leaves the other part as it is, to the last bit (the formula would only be exact up to rounding)
    mean = np.where(n_b == 0, mean_a, np.where(n_a == 0, mean_b, mean_a + delta * n_b / safe))
    m2 = np.where(n_b == 0, m2_a, np.where(n_a == 0, m2_b, m2_a + m2_b + delta * delta * n_a * n_b / safe))
    return n.astype(np.int64), mean, m2


class _Moments:
    """The running count, mean, sum of squares, minimum and maximum of the columns of one table, NaN left out (DAT-3)."""

    def __init__(self, width: int):
        self.n = np.zeros(width, dtype=np.int64)
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.lo = np.full(width, np.inf)
        self.hi = np.full(width, -np.inf)

    def add(self, x: np.ndarray) -> None:
        """Take in the rows ``x`` (rows, width); a NaN is no value and is not counted.

        해석: 이 덩어리의 (n, x̄, D)를 구해 (DAT-3)으로 병합하고, 상수 열을 알아보는 최솟값과 최댓값을 갱신한다.
        """
        x = np.asarray(x, dtype=np.float64)
        have = ~np.isnan(x)
        n = have.sum(axis=0)
        mean = np.divide(np.where(have, x, 0.0).sum(axis=0), n, out=np.zeros(x.shape[1]), where=n > 0)
        dev = np.where(have, x - mean, 0.0)
        self.n, self.mean, self.m2 = _merge(self.n, self.mean, self.m2, n, mean, (dev * dev).sum(axis=0))
        self.lo = np.minimum(self.lo, np.where(have, x, np.inf).min(axis=0))
        self.hi = np.maximum(self.hi, np.where(have, x, -np.inf).max(axis=0))

    def finish(self, centred=(), pool: np.ndarray | None = None) -> ColumnStats:
        """The ``ColumnStats`` of the columns. ``centred``: the position columns (centre W/2, DAT-4). ``pool``: for every column the column
        that the swap exchanges it with (the two share their statistics, DAT-5).

        해석: (DAT-1)의 중심과 척도를 구하고, 위치 열에는 (DAT-4)를, 짝 열에는 (DAT-5)를 쓴다. 값이 하나뿐인 열과 값이 없는 열은 σ = 1이다.
        """
        n, mean, m2, lo, hi = self.n.copy(), self.mean.copy(), self.m2.copy(), self.lo.copy(), self.hi.copy()
        if pool is not None:
            idx = np.arange(len(n))
            moved = pool != idx
            first, second = np.minimum(idx, pool), np.maximum(idx, pool)       # one expression for both columns of a pair: the same bits
            n2, mean2, m22 = _merge(n[first], mean[first], m2[first], n[second], mean[second], m2[second])
            n, mean, m2 = np.where(moved, n2, n), np.where(moved, mean2, mean), np.where(moved, m22, m2)
            lo, hi = np.where(moved, np.minimum(lo, lo[pool]), lo), np.where(moved, np.maximum(hi, hi[pool]), hi)
        spread, centre = m2.copy(), np.where(n > 0, mean, 0.0)
        for j in centred:                                                       # squares about the centre, not the mean (DAT-4)
            spread[j], centre[j] = m2[j] + n[j] * (mean[j] - CENTRE) ** 2, CENTRE
        var = np.divide(spread, n, out=np.zeros_like(spread), where=n > 0)
        none = (n == 0) | (lo == hi) | (var <= 0)
        return ColumnStats(centre, np.where(none, 1.0, np.sqrt(var)), n)


DEFAULT_ROLES = MappingProxyType({'fit': 'FIT'})
_CHUNK = 1 << 12                                 # views read at a time by ``fit_stats`` (4,096 x about 8 KB)


def fit_stats(s1, s2, roles: Mapping = DEFAULT_ROLES) -> Stats:
    """The standardisation statistics of the EPT inputs, computed on the FIT views of the S2 store ``s2`` only.

    ``roles`` must be ``{'fit': 'FIT'}`` (the default): the statistics are those of FIT and of nothing else (ruling R14), and any other
    role, key or extra entry is a ValueError before an array is read. The FIT views are the views of every FIT match, all three sources
    (frame clock, M3 and M4 views, ``features.split_views``). The cells that are NaN are not counted. Reads the arrays of ``_READ`` in
    chunks of ``_CHUNK`` views, so the 18.7 GiB store is never in memory. ValueError for no FIT view, and for the cells that
    ``_check_cells`` refuses.

    수식
    ----
    (DAT-1)  x̃ = (x − x₀) / σ ;  x₀ = x̄, σ = √(D / n) ;  값이 하나뿐인 열(최솟값 = 최댓값)과 값이 없는 열(n = 0)은 σ = 1
    (DAT-3)  n = n₁ + n₂ ;  x̄ = x̄₁ + (x̄₂ − x̄₁) n₂ / n ;  D = D₁ + D₂ + (x̄₂ − x̄₁)² n₁ n₂ / n   (보기 덩어리 1과 2의 병합)
    (DAT-4)  위치 열: x₀ = W/2,  σ² = D / n + (x̄ − W/2)²
    (DAT-5)  팀 교환이 맞바꾸는 요약 열 h와 h′: (n, x̄, D)를 (DAT-3)으로 합친 값을 두 열이 쓴다

      기호          | 뜻                                          | 단위      | 범위           | 코드 변수
      x             | 표의 한 칸의 값(모형 입력 x의 한 성분)      | 칸 단위   | 실수 또는 NaN  | 저장소의 x/*
      x₀, σ         | 중심과 척도                                 | 칸 단위   | σ > 0          | ColumnStats.centre, .scale
      n             | NaN을 뺀 개수                               | 개        | ≥ 0            | ColumnStats.count
      x̄             | 평균                                        | 칸 단위   | 실수           | _Moments.mean
      D             | 평균에서 잰 제곱합                          | 칸 단위²  | ≥ 0            | _Moments.m2
      n₁, x̄₁, D₁ 와 n₂, x̄₂, D₂ | 병합하는 두 덩어리의 같은 양      | 위와 같음 | 위와 같음      | _merge의 n_a, mean_a, m2_a 와 n_b, mean_b, m2_b
      h, h′         | 요약 열 번호와 팀 교환이 짝지우는 열        | —         | 0 .. 105       | SUMMARY_PERM[h]
      W             | 지도 폭 (180° 회전의 상수)                  | 맵 단위   | 14,875         | CENTRE = W/2

    표마다 센 값과 중심은 다음과 같다:

      표             | 센 값                                  | 열 수 | 중심 x₀         | 통계를 나누는 열
      players_num    | 선수 10명 × 보기의 값, 열마다          | 81    | 위치 4열은 W/2  | 선수 행은 열마다 한 통계를 나눔
      summary        | 보기마다, t_s를 뺀 열                  | 106   | x̄               | 팀 교환이 맞바꾸는 열끼리 (DAT-5)
      landmarks      | 표지점 34 × 보기, 열 (서 있음, 남은 초) | 2     | x̄               | 표지점 행은 열마다 한 통계를 나눔
      geo_pp, geo_pl | 모든 쌍                                | 1     | x̄               | 한 표 전체가 한 통계
      ev_dt, ev_gold | 진짜 사건 토큰 (채움 행 뺌)            | 1     | x̄               | —
      ev_xy          | 위치가 있는 진짜 사건                  | 2     | W/2             | —

    해석
    ----
    통계는 훈련 자료인 FIT 보기의 분포만으로 정하고 다른 역할의 값은 읽지도 않으므로, 표준화가 평가 자료를 미리 보는 일이 없다.
    보기를 덩어리로 읽어도 덩어리마다 자기 평균에서 잰 제곱합을 병합하므로 큰 평균에서도 자릿수를 잃지 않고, NaN은 개수에서 빠져 결측이 많은 열도
    있는 값으로만 평균을 낸다. 값이 하나뿐인 열은 (x − x₀) / σ가 0/0이 되므로 σ = 1로 두어 값이 (x − x₀)로 남게 한다.
    위치 열은 평균이 아니라 회전 중심 W/2에서 재므로 팀 교환 x ↦ W − x가 표준화한 좌표에서 정확히 부호 바꿈이 된다.
    선수 10명(표지점 34개)이 한 열의 통계를, 맞바꾸는 요약 열 두 개가 한 쌍의 통계를 나눠 쓰는 것도 교환 뒤에 표준화가 변하지 않게 하려는 조건이다.

    근거
    ----
    계획 Q5 (통계는 FIT로만), 판정 R14 (같은 원칙), Task 1 인터페이스 (NaN은 빼고 센다, 상수 열은 척도 1), 판정 R-S4a-3 (위치 열 중심 W/2, 짝 열
    통계 합침), Chan 외 1983 (덩어리 병합), tokens 모듈 머리 (사건 채움 행).
    """
    if not isinstance(roles, Mapping) or dict(roles) != dict(DEFAULT_ROLES):
        raise ValueError(f"the normalisation statistics are computed on FIT only: roles must be {{'fit': 'FIT'}}, not {roles!r}")
    layout = _layout(s2)
    fit = np.flatnonzero(features.split_views(s1, s2, 'FIT'))
    if fit.size == 0:
        raise ValueError('there is no FIT view to compute the statistics on')
    acc = {'players_num': _Moments(len(PLAYER_COLUMNS)), 'summary': _Moments(len(SUMMARY_COLUMNS)), 'landmarks': _Moments(2),
           'geo_pp': _Moments(1), 'geo_pl': _Moments(1), 'ev_dt': _Moments(1), 'ev_gold': _Moments(1), 'ev_xy': _Moments(2)}
    for lo in range(0, len(fit), _CHUNK):
        a = _read_checked(s2, fit[lo:lo + _CHUNK], layout)
        valid = np.arange(layout.events)[None, :] < a['ev_n'][:, None]
        acc['players_num'].add(a['players_num'].reshape(-1, len(PLAYER_COLUMNS)))
        acc['summary'].add(a['summary'][:, _SUMMARY_AT])
        acc['landmarks'].add(a['landmarks'].reshape(-1, 2))
        acc['geo_pp'].add(a['geo_pp'].reshape(-1, 1))
        acc['geo_pl'].add(a['geo_pl'].reshape(-1, 1))
        acc['ev_dt'].add(np.where(valid, a['ev_dt'], np.nan).reshape(-1, 1))
        acc['ev_gold'].add(np.where(valid, a['ev_gold'], np.nan).reshape(-1, 1))
        acc['ev_xy'].add(np.where(valid[..., None], a['ev_xy'], np.nan).reshape(-1, 2))          # NaN: padding or no position
    arrays = {name: m.finish() for name, m in acc.items() if name not in ('players_num', 'summary', 'ev_xy')}
    arrays['players_num'] = acc['players_num'].finish(centred=_POSITION_AT.tolist())
    arrays['summary'] = acc['summary'].finish(pool=SUMMARY_PERM)
    arrays['ev_xy'] = acc['ev_xy'].finish(centred=(0, 1))
    return Stats(arrays=arrays, n_views=int(fit.size))


# ---- the view tensors ------------------------------------------------------------------------------------------------------------------
VIEW_KEYS = ('view', 'kind', 'side', 'age', 'players', 'players_nan', 'players_zone', 'players_xy', 'landmarks', 'landmarks_nan',
             'summary', 'summary_nan', 'ev_valid', 'ev_type', 'ev_team', 'ev_num', 'ev_xy', 'ev_nan', 'ev_actors', 'ed_valid', 'ed_i',
             'ed_j', 'ed_kind', 'ed_dt', 'geo_pp', 'geo_pp_nan', 'geo_pl', 'geo_pl_nan')


def _standardise(x, cs: ColumnStats) -> np.ndarray:
    """The float64 array x̃ = (x − x₀) / σ of the table cells ``x`` (the columns on the last axis), exactly 0.0 where ``x`` is NaN.

    해석: (DAT-1)의 x̃를 계산하고 NaN 칸은 (DAT-2)대로 0으로 채운다. 채웠다는 사실은 부르는 쪽이 가림 채널로 알린다.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(x), 0.0, (x - cs.centre) / cs.scale)


def _f32(name: str, a) -> torch.Tensor:
    """A float32 tensor of a float array (``casts.narrow``: a value that does not fit stops the call)."""
    return torch.from_numpy(narrow(name, a, np.float32))


def _mask(a) -> torch.Tensor:
    """A float32 tensor of 0.0 and 1.0 of a bool or whole-number array (a mask channel, a multi-hot)."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _i64(a) -> torch.Tensor:
    """An int64 tensor of a whole-number array (a copy, so the tensor never shares memory with a memory map)."""
    return torch.from_numpy(np.array(a, dtype=np.int64))


def _check_stats(stats: Stats) -> None:
    """ValueError naming the table unless ``stats`` has every table with the width of this code's columns."""
    for name, width in (('players_num', len(PLAYER_COLUMNS)), ('summary', len(SUMMARY_COLUMNS)), ('landmarks', 2), ('geo_pp', 1),
                        ('geo_pl', 1), ('ev_dt', 1), ('ev_gold', 1), ('ev_xy', 2)):
        cs = stats.arrays.get(name)
        if cs is None or cs.centre.shape != (width,):
            raise ValueError(f'the statistics have no {name} table of {width} column(s): they belong to another layout')


def _view_tensors(s2, views, stats: Stats) -> dict:
    """The tensors of ``ViewTensors``; see there. Steps: read and check the tables, standardise every table and make its mask channel,
    zero the padding rows of the events and edges and make the valid flags, move to tensors and check that every float is finite."""
    _check_stats(stats)
    layout = _layout(s2)
    idx = features.view_index(layout.n_views, views)       # shared check: int64, 1-d, in range
    a = _read_checked(s2, idx, layout)
    kind = np.asarray(s2.array('views/src'))[idx].astype(np.int64)
    if kind.size and (kind.min() < KIND_FRAME or kind.max() > KIND_M4):
        raise ValueError('views/src must be 0 (frame clock), 1 (M3 event clock) or 2 (M4 event clock)')
    n, st = len(idx), stats.arrays

    # players: the standardised table, the mask of the cells that may be NaN, the zone, the raw position (the centre where it is missing)
    players_raw = a['players_num'].astype(np.float64)
    xy = players_raw[:, :, _POSITION_AT[:2]]
    # summary: the same, without the twin column; the frame age is also given as a column of its own
    summary_raw = a['summary'][:, _SUMMARY_AT].astype(np.float64)
    summary = _standardise(summary_raw, st['summary'])
    # events: a real token has an age, a gold value and maybe a position; the rows of padding are zero in every standardised channel
    valid = np.arange(layout.events)[None, :] < a['ev_n'][:, None]
    ev_xy_raw = a['ev_xy'].astype(np.float64)
    missing = valid & np.isnan(ev_xy_raw).any(axis=-1)
    has_xy = (valid & ~missing)[..., None]
    ev_num = np.concatenate([
        np.where(valid[..., None], _standardise(a['ev_dt'][..., None], st['ev_dt']), 0.0),
        np.where(valid[..., None], _standardise(a['ev_gold'][..., None], st['ev_gold']), 0.0),
        np.where(has_xy, _standardise(ev_xy_raw, st['ev_xy']), 0.0)], axis=-1)
    actors = (a['ev_actors'].astype(np.int64)[..., None] >> np.arange(N_PLAYERS)) & 1
    ed_valid = np.arange(layout.edges)[None, :] < a['ed_n'][:, None]
    out = {
        'view': _i64(idx), 'kind': _i64(kind), 'side': torch.zeros(n, dtype=torch.float32),
        'age': _f32('age', summary[:, _AGE_AT]),
        'players': _f32('players', _standardise(players_raw, st['players_num'])),
        'players_nan': _mask(np.isnan(players_raw)[:, :, _PLAYER_NAN_AT]),
        'players_zone': _i64(a['players_zone']),
        'players_xy': _f32('players_xy', np.where(np.isnan(xy), CENTRE, xy)),
        'landmarks': _f32('landmarks', _standardise(a['landmarks'], st['landmarks'])),
        'landmarks_nan': _mask(np.isnan(a['landmarks'][..., 1:])),
        'summary': _f32('summary', summary),
        'summary_nan': _mask(np.isnan(summary_raw[:, _SUMMARY_NAN_AT])),
        'ev_valid': torch.from_numpy(np.ascontiguousarray(valid)), 'ev_type': _i64(a['ev_type']), 'ev_team': _i64(a['ev_team']),
        'ev_num': _f32('ev_num', ev_num), 'ev_xy': _f32('ev_xy', np.where(has_xy, ev_xy_raw, CENTRE)),
        'ev_nan': _mask(missing[..., None]), 'ev_actors': _mask(actors),
        'ed_valid': torch.from_numpy(np.ascontiguousarray(ed_valid)), 'ed_i': _i64(a['ed_i']), 'ed_j': _i64(a['ed_j']),
        'ed_kind': _i64(a['ed_kind']), 'ed_dt': _f32('ed_dt', a['ed_dt']),
        'geo_pp': _f32('geo_pp', _standardise(a['geo_pp'], st['geo_pp'])), 'geo_pp_nan': _mask(np.isnan(a['geo_pp'])),
        'geo_pl': _f32('geo_pl', _standardise(a['geo_pl'], st['geo_pl'])), 'geo_pl_nan': _mask(np.isnan(a['geo_pl'])),
    }
    for key, t in out.items():
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise ValueError(f'{key}: a value is not finite after the standardisation (statistics of another store?)')
    return out


class ViewTensors(dict):
    """The tensors of the S2 views ``views`` (any order, a view may appear twice), standardised with ``stats``.

    A ``dict[str, torch.Tensor]`` with the keys of ``VIEW_KEYS``. In the shapes below ``views`` stands for ``len(views)``, ``events`` and
    ``edges`` for the numbers of event tokens and edges per view (read from the manifest of ``s2``):

        view (views,) int64, kind (views,) int64   the S2 view number; 0 frame clock, 1 M3 event clock, 2 M4 event clock (``views/src``)
        side (views,) float32                      0 as recorded; ``swap`` makes it 1 (the side marker of design section 5.2)
        age (views,) float32                       the standardised frame age a (``frame_age_s``; 0 s before standardising for a frame-clock view)
        players (views,10,81), players_nan (views,10,23), players_zone (views,10), players_xy (views,10,2)
                                                   standardised columns (positions centred on W/2), the mask of the NaN-able columns
                                                   (``PLAYER_NAN_COLUMNS``), the zone (-1 = no position), the raw position in map units
        landmarks (views,34,2), landmarks_nan (views,34,1)   (standing, seconds left); the mask is for the seconds cell
        summary (views,106), summary_nan (views,3) the summary without ``t_s``; the mask is for ``SUMMARY_NAN_COLUMNS``
        ev_valid (views,events) bool, ev_type, ev_team (views,events) int64, ev_num (views,events,4), ev_xy (views,events,2),
        ev_nan (views,events,1), ev_actors (views,events,10)
                                                   the recent events: ev_num holds the standardised age (the elapsed time to the reference
                                                   time), gold and the two position coordinates; ev_xy the raw position (W/2 where there is
                                                   none); ev_nan marks a real event without a position; ev_actors is the multi-hot of the participants
        ed_valid (views,edges) bool, ed_i, ed_j, ed_kind (views,edges) int64, ed_dt (views,edges) float32
                                                   the co-participation edges; ``ed_dt`` stays in seconds (the decay is exp(-dt / kappa))
        geo_pp (views,10,10), geo_pl (views,10,34) and their masks   the standardised geodesic distances

    A cell that may be NaN is 0.0 in the tensor and 1.0 in the mask channel. The rows of padding are zero in every standardised and mask
    channel (``ev_num``, ``ev_nan``, ``ev_actors``, ``ed_dt``) and hold ``W/2`` in the raw position ``ev_xy``, as a missing position does.
    The tensors hold no NaN and no infinity. ValueError for a NaN where the data may not have one (``_check_cells``), for stats of
    another layout and for view numbers outside the store; the messages name arrays and columns, never a match.

    The positions (``players_xy``, ``ev_xy``, in map units) are those of the frame of the batch: after ``swap`` (``side`` = 1) they are
    the rotated ones, so the physical position is ``W - xy``. A landmark row j stands at ``tokens.LANDMARK_XY[j]`` in the frame of the
    batch and, when ``side`` is 1, at ``tokens.LANDMARK_XY[LANDMARK_PERM[j]]`` on the physical map (``symmetry.swap_view``, way 1).

    수식
    ----
    (DAT-1)  x̃ = (x − x₀) / σ  (x₀, σ는 stats의 값)
    (DAT-2)  ν = 1[x가 NaN] ;  x가 NaN이면 x̃ = 0

      기호  | 뜻                                          | 단위     | 범위           | 코드 변수
      x     | 표의 한 칸의 값(모형 입력 x의 한 성분)      | 칸 단위  | 실수 또는 NaN  | 저장소의 x/*
      x₀, σ | FIT 보기의 중심과 척도 (fit_stats가 잰다)   | 칸 단위  | σ > 0          | stats.arrays[...]
      x̃     | 표준화한 값                                 | 무차원   | 실수           | 텐서의 칸
      ν     | 허용된 NaN 칸의 가림 표시                   | —        | 0 또는 1       | *_nan 텐서

    해석
    ----
    각 칸은 FIT 분포를 기준으로 한 표준화 점수가 되고, 값이 없는 칸은 0(기준 분포의 중심)으로 채운 뒤 가림 채널로 채웠다는 사실을 알린다.
    이렇게 하면 신경망은 NaN을 보지 않으면서도 채운 0과 실제로 0인 값을 가림 채널로 구별한다.
    프레임 시계 보기는 프레임 나이 a가 0이고 사건 시계 보기는 a가 양수이므로, 프레임 나이와 보기 종류가 두 시계를 알려 준다.
    사건 토큰의 나이는 그 사건에서 기준 시각까지의 경과 시간이다.

    근거
    ----
    계획 Q5 (FIT 표준화, NaN → 0 + 가림 채널, 프레임 나이·경과 시간·보기 종류), 검토 초점 3 (NaN 입력), 판정 R-S4a-1 (모양은 저장소 목록에서),
    Che 외 2018 (0 채움 + 가림 + 경과 시간), tokens 모듈 머리 (NaN이 서는 자리).
    """

    def __init__(self, s2, views, stats: Stats):
        super().__init__(_view_tensors(s2, views, stats))


# ---- the rows of a head in a batch -----------------------------------------------------------------------------------------------------
# The fields of the rows of every head, as a batch holds them under the key ``<head>/<kind>/<field>``. The rows, targets, exposures and
# categories are those of ``head_rows.HeadRows.parts`` (ruling R18), unchanged:
#   pos      index of the view of the row in the ``view`` tensor of the batch
#   match    the match of the row                     row    the id of the evaluation row (``HeadParts.row``: a frame row, an onset, a unit)
#   unit     the engagement the row belongs to (an onset's unit for a masked head; the unit itself for an open head)
#   q m z s  the categories: occurrence bin, mark, zone (-1 = none), size class     layer, team   kill layer and team of a gold entry
#   K        kills blue, kills red, kill gold blue and red in thousands (the true values of the engagement)
#   O        the 18 objectives of the engagement (the true values)
#   y        the target: M2Rate the counts of the 18 bins (dense: one row per frame row), M2Mark the mark, M2Zone the zone, M3Size the
#            size class, M3Kills the kills of both teams, M3Gold the gold bin of a team entry, M3Obj and M4Conv the 18 counts of the cells
#   e        the exposure in seconds: M2Rate per bin (0 = not observed), M4Conv per row; the other Poisson heads have exposure 1
ROW_FIELDS = MappingProxyType({
    ('M2Rate', 'masked'): ('pos', 'match', 'row', 'y', 'e'),
    ('M2Mark', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'y'),
    ('M2Zone', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'y'),
    ('M3Size', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'z', 'y'),
    ('M3Kills', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'z', 's', 'y'),
    ('M3Gold', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'z', 's', 'layer', 'team', 'y'),
    ('M3Obj', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'z', 's', 'K', 'y'),
    ('M4Conv', 'masked'): ('pos', 'match', 'row', 'unit', 'q', 'm', 'z', 'K', 'O', 'y', 'e'),
    ('M3Size', 'open'): ('pos', 'match', 'row', 'unit', 'y'),
    ('M3Kills', 'open'): ('pos', 'match', 'row', 'unit', 's', 'y'),
    ('M3Gold', 'open'): ('pos', 'match', 'row', 'unit', 's', 'layer', 'team', 'y'),
    ('M3Obj', 'open'): ('pos', 'match', 'row', 'unit', 's', 'K', 'y'),
    ('M4Conv', 'open'): ('pos', 'match', 'row', 'unit', 'K', 'O', 'y', 'e'),
})
if set(ROW_FIELDS) != set(head_rows.VALID_HEADS):
    raise RuntimeError('ROW_FIELDS must have a block for every head and kind of head_rows')
_FLOAT_FIELDS = frozenset({'K', 'O', 'e'})       # float32 in a batch; every other field is int64
FRAME_BLOCK = ('M2Rate', 'masked')               # the one block whose rows belong to a frame row and not to an engagement


def row_key(head: str, kind: str, field: str) -> str:
    """The key of a field of a row block in a batch: ``'<head>/<kind>/<field>'``."""
    return f'{head}/{kind}/{field}'


def block(batch: Mapping, head: str, kind: str) -> dict:
    """The fields (without the prefix) of the row block of ``(head, kind)`` in ``batch``: ``{'pos': ..., 'y': ...}``."""
    return {field: batch[row_key(head, kind, field)] for field in ROW_FIELDS[(head, kind)]}


# ---- the swap --------------------------------------------------------------------------------------------------------------------------
_PLAYER_T, _LANDMARK_T, _SUMMARY_T = _i64(PLAYER_PERM), _i64(LANDMARK_PERM), _i64(SUMMARY_PERM)       # the tables as torch tensors (copies)
_MARK_T, _ZONE_T, _K_T = _i64(MARK_PERM), _i64(ZONE_PERM), _i64(K_PERM)
_TEAM_T, _CELL_T, _POSITION_T = _i64(TEAM_PERM), _i64(CELL_PERM), _i64(_POSITION_AT)


def _on(perm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The table ``perm`` (kept on the CPU) on the device of ``t``: ``Tensor.to`` returns the table itself when ``t`` is on the CPU, and a
    batch that was moved to a GPU is swapped there (the tables have at most 106 entries)."""
    return perm.to(t.device)


def _keep(t):
    return t.clone()


def _rows(perm):
    """Rule: the rows (dimension 1) change places by ``perm``."""
    return lambda t: t.index_select(1, _on(perm, t))


def _last(perm):
    """Rule: the cells of the last dimension change places by ``perm``."""
    return lambda t: t.index_select(-1, _on(perm, t))


def _both(first, second):
    """Rule: dimension 1 changes places by ``first`` and dimension 2 by ``second`` (the geodesic tables)."""
    return lambda t: t.index_select(1, _on(first, t)).index_select(2, _on(second, t))


_player_rows = _rows(_PLAYER_T)


def _zone(t):
    """Rule: a zone 0 .. 9 becomes its partner under the rotation; -1 (no position) stays.

    해석: (DAT-6)의 구역 규칙을 계산한다.
    """
    return torch.where(t >= 0, _on(_ZONE_T, t)[t.clamp(min=0)], t)


def _mark(t):
    """Rule: a mark 0 .. 51 becomes the mark of the swapped kill; ValueError for a value outside 0 .. 51.

    해석: (DAT-6)의 표지 규칙을 계산한다.
    """
    if t.numel() and (t.min() < 0 or t.max() >= C.N_MARKS):
        raise ValueError(f'a mark is 0 .. {C.N_MARKS - 1}')
    return _on(_MARK_T, t)[t]


def _players(t):
    """Rule for ``players`` (views, 10, 81): the rows change places and the four centred position columns change sign.

    해석: (DAT-6)의 선수 규칙을 계산한다.
    """
    out = _player_rows(t)
    at = _on(_POSITION_T, t)
    out[..., at] = 0.0 - out[..., at]                                    # not -out: 0.0 - (+0.0) is +0.0, but -(+0.0) is -0.0
    return out


def _ev_num(t):
    """Rule for ``ev_num`` (views, events, 4) = (age, gold, x̃, ỹ): the centred position changes sign, age and gold stay.

    해석: (DAT-6)의 사건 규칙을 계산한다.
    """
    out = t.clone()
    out[..., 2:] = 0.0 - out[..., 2:]                                   # age and gold stay; the centred x and y change sign (0.0 - t: see _players)
    return out


def _participant(t):
    """Rule: a participant index 0 .. 9 becomes (index + 5) mod 10; -1 (no participant, a padding edge) stays.

    해석: (DAT-6)의 선수 번호 규칙을 간선의 두 끝에 계산한다. 간선의 끝은 선수 행 번호이다.
    """
    if t.numel() and (t.min() < -1 or t.max() > N_PLAYERS - 1):
        raise ValueError('a participant index is -1 (none) or 0 .. 9')
    return torch.where(t >= 0, (t + N_PLAYERS // 2) % N_PLAYERS, t)


_VIEW_RULES = {
    'view': _keep, 'kind': _keep, 'age': _keep, 'side': lambda t: 1.0 - t,
    'players': _players, 'players_nan': _player_rows, 'players_zone': lambda t: _player_rows(_zone(t)),
    'players_xy': lambda t: W - _player_rows(t),
    'landmarks': _rows(_LANDMARK_T), 'landmarks_nan': _rows(_LANDMARK_T), 'summary': _last(_SUMMARY_T), 'summary_nan': _keep,
    'ev_valid': _keep, 'ev_type': _keep, 'ev_team': lambda t: torch.where(t < 2, 1 - t, t), 'ev_num': _ev_num, 'ev_xy': lambda t: W - t,
    'ev_nan': _keep, 'ev_actors': _last(_PLAYER_T),
    'ed_valid': _keep, 'ed_i': _participant, 'ed_j': _participant, 'ed_kind': _keep, 'ed_dt': _keep,
    'geo_pp': _both(_PLAYER_T, _PLAYER_T), 'geo_pp_nan': _both(_PLAYER_T, _PLAYER_T),
    'geo_pl': _both(_PLAYER_T, _LANDMARK_T), 'geo_pl_nan': _both(_PLAYER_T, _LANDMARK_T),
}
if set(_VIEW_RULES) != set(VIEW_KEYS):
    raise RuntimeError('the swap needs a rule for every key of the view tensors, and for no other')


_BY_FIELD = {'m': 'mark', 'z': 'zone', 'team': 'team', 'K': 'k', 'O': 'cells'}      # the fields whose meaning is the same in every head
_BY_TARGET = {('M2Mark', 'masked'): 'mark', ('M2Zone', 'masked'): 'zone', ('M3Kills', 'masked'): 'teams', ('M3Kills', 'open'): 'teams',
              ('M3Obj', 'masked'): 'cells', ('M3Obj', 'open'): 'cells', ('M4Conv', 'masked'): 'cells', ('M4Conv', 'open'): 'cells'}


def _field_swap(head: str, kind: str, field: str) -> str | None:
    """What the team swap does to a field of a row block (see ``FIELD_SWAP``): the name of the change, or None for a field that does not
    depend on the team. The target ``y`` means a mark, a zone, a column for each team or the 18 cells, depending on the head."""
    return _BY_TARGET.get((head, kind)) if field == 'y' else _BY_FIELD.get(field)


# The one table of the team-dependent fields (pre-flight finding T1 -> T3): for every field of every row block, how the swap changes it.
#   'mark'   a mark of the first kill, 0 .. 51 (MARK_PERM)                'zone'   a zone, -1 .. 9 (ZONE_PERM, -1 stays)
#   'team'   the team of a gold entry, 0 blue / 1 red (1 - team)     'teams'  a column for each team (TEAM_PERM)
#   'k'      the K block: kills blue, red, kill gold blue, red (K_PERM)     'cells'  the 18 cells of O and of the conversions (CELL_PERM)
#   None     the field does not depend on the team (pos, match, row, unit, q, s, layer, the gold bin, the size, the exposures)
# A model that is run on both frames maps its outputs back with the same tables: the mark axis of M2's mark_logp and zone_logp
# (MARK_PERM), the zone axis of zone_logp (ZONE_PERM), the team axis of kill_rate and gold_logp (TEAM_PERM), the cells of obj_rate and
# conv_rate (CELL_PERM); the rate, the size law and the gold bin are the same in both frames.
FIELD_SWAP = MappingProxyType({(h, k, f): _field_swap(h, k, f) for (h, k), fields in ROW_FIELDS.items() for f in fields})
_APPLY = {None: _keep, 'mark': _mark, 'zone': _zone, 'team': lambda t: 1 - t, 'teams': _last(_TEAM_T), 'k': _last(_K_T),
          'cells': _last(_CELL_T)}
_RULES = {**_VIEW_RULES, **{row_key(h, k, f): _APPLY[change] for (h, k, f), change in FIELD_SWAP.items()}}


def swap(batch: Mapping) -> dict:
    """The batch seen from the other side: every input and every team-dependent target and category changed together, as new tensors.

    ``batch`` is a mapping from keys to tensors: the keys of ``ViewTensors`` (``VIEW_KEYS``) and the row blocks of ``ROW_FIELDS``
    (``<head>/<kind>/<field>``), all of them or any part (a caller that holds only views may swap those). The input is not
    changed and the output shares no memory with it. ``swap(swap(batch))`` equals ``batch`` bit for bit when the input contains no negative zero (−0.0). ValueError, naming the keys, for a key that
    is neither a view key nor a field of a row block (a silent pass-through would leave a team-dependent array unswapped, the half
    swap of review focus 1); TypeError for a value that is not a tensor.

    The views: the players change places, [5-9, 0-4]; the positions turn by 180° (raw: x ↦ W − x; centred and standardised: x̃ ↦ −x̃);
    the zones follow ``terrain.ZONE_SWAP`` (-1 stays); the team codes of the events and the actor columns, the participants of the edges
    (-1 stays) and the rows and columns of the geodesics follow the players; the summary columns, the landmark rows and the ``geo_pl``
    columns are exchanged by ``SUMMARY_PERM`` and ``LANDMARK_PERM``; the side marker ``side`` becomes 1 − side. The rows of a block follow
    ``FIELD_SWAP``: the mark, the zone, the two teams of ``K``, of ``O`` and of the kills, and the team of a gold entry change, and
    everything else stays. The tables are exported (``MARK_PERM``, ``ZONE_PERM``, ``K_PERM``, ``TEAM_PERM``, ``CELL_PERM``,
    ``PLAYER_PERM``): a model that is run on both frames maps its outputs back with the same tables.

    수식
    ----
    (DAT-6)  x ↦ W − x ;  x̃ ↦ −x̃ (위치 열) ;  ξ ↦ (ξ + 5) mod 10 ;  m ↦ (m + 25) mod 50 (m < 50) 또는 101 − m (m ≥ 50) ;  z ↦ ZONE_PERM[z] (z ≥ 0) ;
             K ↦ (K₂, K₁, K₄, K₃) ;  O ↦ O[CELL_PERM] ;  팀 칸 (파랑, 빨강) ↦ (빨강, 파랑), 골드 항목의 팀 ↦ 1 − 팀 ;  side ↦ 1 − side

      기호  | 뜻                                              | 단위            | 범위        | 코드 변수
      x, x̃  | 위치 열의 칸의 값, 표준화한 값                  | 맵 단위, 무차원 | 0 .. W, 실수 | players, players_xy, ev_num, ev_xy
      W     | 지도 폭 (180° 회전의 상수)                      | 맵 단위         | 14,875      | W
      ξ     | 선수 번호 (0–4 파랑, 5–9 빨강; 간선 끝의 −1은 없음) | —           | −1, 0 .. 9  | players의 둘째 축, ed_i, ed_j
      m     | 선제 킬 표지 (쌍 50개와 처형 50, 51)            | —               | 0 .. 51     | m, y(M2Mark)
      z     | 첫 킬 구역 (−1은 위치 없음)                     | —               | −1 .. 9     | z, y(M2Zone), players_zone
      K     | 킬 부분 (K₁, K₂ 파랑·빨강 킬 수, K₃, K₄ 파랑·빨강 킬 골드) | 개, 천 골드 | ≥ 0     | K
      O     | 오브젝트 개수 18칸 (칸 = 2 · 종류 + 취한 팀)    | 개              | 0, 1, …     | O, y(M3Obj, M4Conv)
      side  | 진영 표시 (기록대로 0, 교환하면 1)              | —               | 0 또는 1    | side

    해석
    ----
    팀 교환은 같은 경기를 반대편에서 본 것이므로 입력과 정답을 한꺼번에 바꾼다.
    표지는 킬러 팀만 뒤집히고 (선수 행이 팀 안에서 같은 순서를 지키므로 칸 번호는 그대로이며 처형 두 표지는 서로 바뀐다), 구역은 지도 회전의 짝 구역으로 가고
    (중앙 길과 강은 제자리), K와 O는 두 팀 칸이 맞바뀐다.
    이 변환은 제곱하면 항등이므로 원래 입력과 맞바꾼 입력의 출력을 원래 틀로 되돌려 평균한 값 (프레임 평균)이 팀 교환에 정확히 불변이다.
    위치의 부호는 0.0 − x̃로 바꾸어 0이 −0.0이 되지 않으므로, 교환한 입력은 원자료를 맞바꾸어 표준화한 결과와 부호 비트까지 같다.
    모르는 키를 오류로 막는 것은 팀에 매인 배열이 바뀌지 않은 채 지나가 정답이 뒤바뀌는 것을 막기 위해서이다.

    근거
    ----
    설계서 5.2절 (회전 + 진영 표시, 출력도 같이 교환), 검토 초점 1 (반쪽 적용), 계획 Q7 (프레임 평균), 판정 R-S4a-3 (표준화가 교환과 맞바뀐다),
    S4a 사전 점검 T1→T3 (팀 의존 표는 한 곳), symmetry 모듈 (기준 구현), terrain.ZONE_SWAP (구역 규칙에서 유도).
    """
    unknown = sorted(k for k in batch if k not in _RULES)
    if unknown:
        raise ValueError(f'swap does not know how to swap {unknown}: a silent pass-through would leave a team-dependent array unswapped')
    for k, t in batch.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f'{k}: swap takes tensors, not {type(t).__name__}')
    return {k: _RULES[k](t) for k, t in batch.items()}


# ---- the proxy task --------------------------------------------------------------------------------------------------------------------
PROXY_SEED = 20261022
PROXY_ROLES = ('FIT', 'STOP')


def proxy_matches(s1, share: float = 0.25, seed: int = PROXY_SEED) -> dict:
    """The matches of the proxy task: ``{'FIT': matches, 'STOP': matches}``, each a sorted int64 array of match numbers (S1 rows) of that
    role. Training reads the FIT matches, early stopping the STOP matches (plan Q10); DEV is not in it.

    ``share`` is the fraction of the matches of each role, ``seed`` the seed of the choice. For one seed a smaller share is contained in
    every larger one. Reads only ``matches/role`` of ``s1``. ValueError for a share outside (0, 1] and for a role without a match.

    수식
    ----
    (DAT-7)  M′ = max(1, ⌈ζ M⌉) ;  선택 = 순열 ω_{seed,role}로 섞은 그 역할의 경기 열의 앞 M′개 ;  ω는 ζ에 의존하지 않는다

      기호  | 뜻                                | 단위 | 범위    | 코드 변수
      M     | 그 역할의 경기 수                 | 경기 | ≥ 1     | len(matches)
      ζ     | 뽑는 몫                           | —    | (0, 1]  | share
      M′    | 뽑은 경기 수                      | 경기 | 1 .. M  | k
      ω     | 시드와 역할 번호로 만든 경기 순열 | —    | —       | order

    해석
    ----
    대리 과제는 훈련 비용을 재기 위해 경기의 일부(몫 ζ)만 쓴다.
    같은 순열의 앞쪽을 자르므로 몫을 키우면 앞서 뽑은 경기가 그대로 남고 경기가 더해질 뿐이어서, 작은 몫에서 잰 시간과 곡선이 큰 몫과 같은 경기 위에서 이어진다.
    순열은 시드와 역할 번호로만 정해지므로 다른 역할의 경기 수나 몫이 바뀌어도 이 역할의 선택은 변하지 않는다.
    곱 ζ M은 소수 아홉 자리로 반올림한 뒤 올려서 부동소수 잡음이 경기 하나를 더하지 못하게 한다.
    경기 번호만 보고 경기 키는 읽지 않는다.

    근거
    ----
    계획 Q10 (대리 과제: FIT 경기 일부를 시드로 고름, 조기 종료는 STOP), Task 1 인터페이스 (같은 시드에서 작은 몫은 큰 몫 안에 든다).
    """
    if isinstance(share, bool) or not isinstance(share, (int, float, np.floating)) or not 0.0 < share <= 1.0:
        raise ValueError(f'share must be a number in (0, 1], not {share!r}')
    roles = np.asarray(s1.array('matches/role'))
    out = {}
    for name in PROXY_ROLES:
        code = C.ROLES.index(name)
        matches = np.flatnonzero(roles == code).astype(np.int64)
        if matches.size == 0:
            raise ValueError(f'the S1 store has no {name} match')
        order = np.random.default_rng(np.random.SeedSequence([int(seed), code])).permutation(matches)
        k = max(1, math.ceil(round(float(share) * matches.size, 9)))
        out[name] = np.sort(order[:k])
    return out


# ---- the rows of an engagement ---------------------------------------------------------------------------------------------------------
_M4_K_NAMES = ('kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k')      # K among the extras of the masked M4 matrix, in the order of K_PERM


def _no_features(s2, views) -> np.ndarray:
    """The feature builder of the EPT rows: no feature, one zero column. The EPT reads the S2 views itself (``view_numbers``)."""
    return np.zeros((len(views), 1), dtype=np.float32)


def ept_head_rows(s1, s2, labels, role: str) -> head_rows.HeadRows:
    """``head_rows.HeadRows`` of the matches of ``role`` (FIT, STOP or DEV; ValueError for any other role, before an array is read) whose
    rows and targets are those of the baselines B0 and B1 (ruling R18) and whose feature rows are empty: the EPT takes the view numbers
    of ``HeadRows.view_numbers`` and reads the views with ``ViewTensors``."""
    return head_rows.HeadRows(s1, s2, labels, role, _no_features)


def _typed(fields: dict) -> dict:
    """The fields of a block in the dtypes of a batch: ``K``, ``O`` and ``e`` float32 (``casts.narrow``), every other field int64."""
    return {f: (narrow(f, np.asarray(a, dtype=np.float64), np.float32) if f in _FLOAT_FIELDS else np.ascontiguousarray(a, dtype=np.int64))
            for f, a in fields.items()}


def _block_rows(head: str, kind: str, p: head_rows.HeadParts, views: np.ndarray, onset_unit: np.ndarray) -> tuple:
    """``(fields, views, group)`` of one head: the fields of ``ROW_FIELDS`` (numpy, int64 or float32, without ``pos``), the S2 view of every
    row and the group of every row (the engagement for a unit block, the frame row for the M2Rate block).

    해석: 이 단계는 새 수치를 만들지 않고 행과 정답을 배치가 읽는 필드 이름으로 옮긴다. 프레임 행의 구간 칸은 프레임 행마다 한 줄의 개수와
    노출로 모으며, 노출 0 칸은 개수 0이다. 가린 M4 머리의 K와 O는 그 입력을 만드는 한 함수의 열에서 이름으로 찾아 읽고, 열 배치가 다르면 오류로
    멈춘다 (누설 시험 (a)).
    """
    if head == 'M2Rate':
        frame, first, inverse = np.unique(p.row, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        if not np.array_equal(views, views[first][inverse]):
            raise ValueError('the cells of one frame row read different views')
        y, e = np.zeros((len(frame), C.N_BINS), dtype=np.int64), np.zeros((len(frame), C.N_BINS))
        y[inverse, p.codes['q']], e[inverse, p.codes['q']] = p.y, p.e
        return _typed({'match': p.match[first], 'row': frame, 'y': y, 'e': e}), views[first], frame
    fields = {'match': p.match, 'row': p.row, 'unit': p.row if kind == 'open' else onset_unit[p.row], 'y': p.y}
    fields.update(p.codes)
    if p.team is not None:
        fields['team'] = p.team
    if head == 'M4Conv':
        fields['e'] = p.e
    if head == 'M4Conv' and kind == 'masked':
        d, extra = p.feat.shape[1], features.M4_EXTRA_NAMES                 # the matrix is [row features, extras (q, m, z, K), the cells of O]
        k0 = d + extra.index(_M4_K_NAMES[0]) if _M4_K_NAMES[0] in extra else -1
        if k0 < 0 or extra[k0 - d:k0 - d + len(_M4_K_NAMES)] != _M4_K_NAMES or p.m4.shape[1] != d + len(extra) + N_CELLS:
            raise ValueError(f'the masked M4 matrix has {p.m4.shape[1]} columns, not {d} row features + the extras {extra} + {N_CELLS} cells '
                             'of O in the layout of features.m4_masked_inputs')
        fields['K'], fields['O'] = p.m4[:, k0:k0 + len(_M4_K_NAMES)], p.m4[:, d + len(extra):]
    else:
        if p.K is not None:
            fields['K'] = p.K
        if p.O is not None:
            fields['O'] = p.O
    if set(fields) != set(ROW_FIELDS[(head, kind)]) - {'pos'}:
        raise RuntimeError(f'the rows of {head} ({kind}) have the fields {sorted(fields)}, not those of ROW_FIELDS')
    return _typed(fields), views, fields['unit'].astype(np.int64)


class _Block:
    """The rows of one head, grouped by engagement (or by frame row): ``fields``, the S2 ``views`` of the rows and the ``group`` of each."""

    def __init__(self, fields: dict, views: np.ndarray, group: np.ndarray):
        self.fields, self.views, self.group = fields, views, group
        self._order = np.argsort(group, kind='stable')
        self._sorted = group[self._order]

    def select(self, ids) -> np.ndarray:
        """The rows whose group is one of ``ids``: first the rows of ``ids[0]`` (in the order of the block), then those of ``ids[1]``...

        해석: 정렬해 둔 묶음 번호에서 이진 탐색으로 범위를 찾아 이어 붙이므로, 한 묶음 안의 행은 블록의 원래 순서이고 같은 입력은 늘 같은 순서의 행을 낸다.
        """
        ids = np.asarray(ids, dtype=np.int64)
        lo, hi = np.searchsorted(self._sorted, ids, side='left'), np.searchsorted(self._sorted, ids, side='right')
        found = [self._order[a:b] for a, b in zip(lo.tolist(), hi.tolist()) if b > a]
        return np.concatenate(found) if found else np.empty(0, dtype=np.int64)


class EngagementRows:
    """The rows of the 13 heads of one role, grouped by engagement, for the matches ``matches`` (all of the role's matches by default).

    Built from ``head_rows.HeadRows`` (``ept_head_rows``): the rows, targets, exposures and categories of every head are those of
    ``HeadRows.parts`` (the masked M3 object head with its K block), their views those of ``HeadRows.view_numbers``. ``units`` are the
    engagements that have a row in any block but the M2Rate block, ``frames`` the frame rows of the M2Rate block (both sorted, int64);
    ``blocks[(head, kind)]`` holds the rows. ``matches`` (an array of match numbers, for example ``proxy_matches(s1)['FIT']``) keeps only
    the rows of those matches. ``role`` is the role of the rows.

    해석: 한 교전의 가린 행과 열린 행은 같은 묶음으로 모이고 프레임 행은 한 줄씩 따로 모이며, 배치는 이 묶음 단위로 만든다.
    """

    def __init__(self, rows: head_rows.HeadRows, matches=None):
        if not isinstance(rows, head_rows.HeadRows):
            raise TypeError(f'rows must be a head_rows.HeadRows (ept_head_rows), not {type(rows).__name__}')
        self.role, self.head_rows = rows.role, rows
        keep = None if matches is None else np.unique(np.asarray(matches, dtype=np.int64))
        onset_unit = np.asarray(rows.s1.array('onsets/unit')).astype(np.int64)
        self.blocks = {}
        for (head, kind) in ROW_FIELDS:
            parts = rows.parts(head, kind, masked_obj_k=True)               # masked_obj_k: the masked M3Obj head gets its K block
            fields, views, group = _block_rows(head, kind, parts, rows.view_numbers(head, kind), onset_unit)
            if keep is not None:
                sel = np.isin(fields['match'], keep)
                fields, views, group = {f: a[sel] for f, a in fields.items()}, views[sel], group[sel]
            self.blocks[(head, kind)] = _Block(fields, views, group)
        unit_groups = [blk.group for key, blk in self.blocks.items() if key != FRAME_BLOCK]
        self.units = np.unique(np.concatenate(unit_groups)) if unit_groups else np.empty(0, dtype=np.int64)
        self.frames = np.unique(self.blocks[FRAME_BLOCK].group)

    def counts(self) -> dict:
        """``{'units': n, 'frames': n, 'rows': {'<head>/<kind>': n}}``: the engagements, the frame rows and the rows of every head (for the
        run record: the size of the task)."""
        return {'units': int(len(self.units)), 'frames': int(len(self.frames)),
                'rows': {f'{head}/{kind}': int(len(blk.group)) for (head, kind), blk in self.blocks.items()}}


class Batches:
    """The batches of an epoch over the rows of an ``EngagementRows``: ``Batches(rows, batch_size, seed, stats=None, training=True)``.

    ``rows`` may also be a ``HeadRows`` (``ept_head_rows``): all its matches are used. ``batch_size`` is the number of engagements of a
    batch (the rows of an engagement, of all its heads, come with it: its masked rows, which read frame-clock views, and its open rows,
    which read the M3 and M4 views of the same engagement, plan Q8). The frame rows of the M2Rate head are spread over the batches of
    an epoch in the same way, so every row of every head is in exactly one batch of an epoch. A batch is a flat ``dict`` of tensors: the
    ``ViewTensors`` of the distinct views of the batch (``view`` lists them), and for every head ``<head>/<kind>/<field>`` of
    ``ROW_FIELDS`` with ``pos``, the index into those views. Without ``stats`` a batch holds only ``view`` and the row blocks: the caller
    completes it with ``ViewTensors(s2, batch['view'], stats)``. With ``training=True`` (the default) the engagements and frame rows are
    shuffled across matches, in an order that depends on ``seed`` and the epoch number only, and the rows must be those of FIT: a
    training batch never holds a STOP or DEV row (ValueError otherwise). With ``training=False`` the order is fixed (ascending) and the
    role may be FIT, STOP or DEV: the batches of early stopping and of evaluation. ``len(batches)`` is the number of batches of an epoch.
    ``epoch(epoch, start=0)`` yields the batches ``start ..`` of that epoch without building the first ones (to resume a cut run).
    There is no default epoch: a ``Batches`` is not iterable, so that a loop cannot repeat the order of epoch 0.

    수식
    ----
    (DAT-8)  J = ⌈N / L⌉ ;  교전 열과 프레임 행 열을 순열 ω_{seed,epoch}로 섞어 J덩어리로 나눈다 (덩어리 크기는 서로 많아야 하나 다르다)

      기호   | 뜻                                                   | 단위 | 범위 | 코드 변수
      N      | 나누는 단위의 수: 교전 수, 교전이 없으면 프레임 행 수 | 개   | ≥ 1  | len(rows.units), len(rows.frames)
      L      | 배치당 교전 수                                       | 개   | ≥ 1  | batch_size
      J      | 한 에폭의 배치 수                                    | 개   | ≥ 1  | len(self)
      ω      | 시드와 에폭 번호로 만든 임의 순열 (training=False이면 쓰지 않는다) | — | — | rng.permutation

    해석
    ----
    한 에폭은 모든 교전과 모든 프레임 행을 정확히 한 번 훑는다.
    교전을 경기와 상관없이 섞으므로 한 배치에 여러 경기의 교전이 들어가 배치 안 표본이 서로 덜 닮는다.
    한 교전의 모든 행을 한 배치에 두므로 같은 교전의 가린 머리와 열린 머리의 손실이 같은 갱신에서 같은 가중으로 합쳐진다.
    덩어리 크기는 서로 많아야 하나 다르므로 마지막 배치만 작아지는 일이 없다.
    순서는 시드와 에폭 번호로만 정해지므로 같은 입력은 같은 순서를 내고, 중간에서 다시 시작해도 같은 순서가 나온다.

    근거
    ----
    계획 Q8 (한 교전의 두 보기는 같은 배치), Q4 (행과 정답은 head_rows), Q3 (같은 입력은 같은 비트), 계획 P9 (학습 행 가중은 같다).
    """

    def __init__(self, rows, batch_size: int, seed: int, *, stats: Stats | None = None, training: bool = True):
        if isinstance(rows, head_rows.HeadRows):
            rows = EngagementRows(rows)
        if not isinstance(rows, EngagementRows):
            raise TypeError(f'rows must be an EngagementRows or a HeadRows, not {type(rows).__name__}')
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError(f'batch_size must be an int, not {type(batch_size).__name__}')
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, not {batch_size}')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f'seed must be a non-negative int, not {seed!r}')
        if stats is not None and not isinstance(stats, Stats):
            raise TypeError(f'stats must be the Stats of fit_stats or None, not {type(stats).__name__}')
        if training and rows.role != 'FIT':
            raise ValueError(f'training batches read FIT rows only (ruling R14), not the rows of {rows.role}; '
                             'pass training=False for the batches of early stopping or evaluation')
        if len(rows.units) == 0 and len(rows.frames) == 0:
            raise ValueError('the rows hold no engagement and no frame row')
        self.rows, self.batch_size, self.seed, self.stats, self.training = rows, int(batch_size), int(seed), stats, bool(training)
        self._n = -(-len(rows.units) // self.batch_size) if len(rows.units) else -(-len(rows.frames) // self.batch_size)

    def __len__(self) -> int:
        return self._n

    def plan(self, epoch: int) -> list:
        """``[(engagements, frame rows)]`` of each batch of the epoch ``epoch`` (int arrays), without reading a view.

        해석: (DAT-8)의 순열과 분할을 계산한다. 순서가 시드와 에폭 번호만의 함수이므로 앞의 배치를 만들지 않고도 중간 배치부터 이어 갈 수 있다.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError(f'epoch must be a non-negative int, not {epoch!r}')
        units, frames = self.rows.units, self.rows.frames
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
            units, frames = units[rng.permutation(len(units))], frames[rng.permutation(len(frames))]
        return list(zip(np.array_split(units, self._n), np.array_split(frames, self._n)))

    def assemble(self, units, frames) -> dict:
        """The batch of the engagements ``units`` and the frame rows ``frames``: the rows of every block, their distinct views as
        ``ViewTensors`` and the position of each row's view in them.

        해석: 한 교전의 모든 머리의 행과 고른 프레임 행이 읽는 보기를 중복 없이 한 번씩만 텐서로 만들고, 각 행에는 그 보기가 놓인 자리를 적는다.
        """
        chosen = {key: blk.select(frames if key == FRAME_BLOCK else units) for key, blk in self.rows.blocks.items()}
        wanted = np.concatenate([self.rows.blocks[key].views[idx] for key, idx in chosen.items()])
        distinct, inverse = np.unique(wanted, return_inverse=True)
        inverse, start = inverse.reshape(-1).astype(np.int64), 0
        out = ({'view': torch.from_numpy(distinct.astype(np.int64))} if self.stats is None
               else dict(ViewTensors(self.rows.head_rows.s2, distinct, self.stats)))
        for key, idx in chosen.items():
            head, kind = key
            out[row_key(head, kind, 'pos')] = torch.from_numpy(inverse[start:start + len(idx)].copy())
            start += len(idx)
            for field, a in self.rows.blocks[key].fields.items():
                out[row_key(head, kind, field)] = torch.from_numpy(a[idx])
        return out

    def epoch(self, epoch: int, start: int = 0):
        """The batches ``start ..`` of the epoch ``epoch`` (an iterator of dicts). ValueError for a ``start`` outside 0 .. len(self)."""
        plan = self.plan(epoch)
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start <= len(plan):
            raise ValueError(f'start must be an int in 0 .. {len(plan)}, not {start!r}')
        return (self.assemble(units, frames) for units, frames in plan[start:])
