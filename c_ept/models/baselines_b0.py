r"""B0 베이스라인 모델: M2(발생), M3(교전 결과), M4(전환) 예측을 위한 GLM 헤드 번들 (Stage S3).

[1. 모듈 개요 및 하네스 상의 역할]
- 본 모듈은 EPT(교전 과정 변환기) 및 B1(LightGBM) 모델이 반드시 능가해야 하는 기준선(Floor) 모델 B0를 정의한다.
- 39차원 시간-상태(time-state) 특징 표(`features.b0`: 시간, 경기 국면, 골드/레벨 차이, 오브젝트 상태, 진영 간 거리)와
  각 머리가 조건으로 삼는 범주형 변수를 입력으로 하는 로그-선형 일반화선형모형(GLM) 헤드들의 집합이다.
- 가린 헤드(Masked Heads)는 선형 예측자 표 조회를 통해 닫힌 꼴(closed-form) 기댓값 계산을 지원하여,
  합성기(Composer)가 수백만 개의 발생(Onset) 행에 대해 빠른 가치 평가를 수행할 수 있게 한다.
- 모든 모델은 scikit-learn 기반으로 FIT 분할 데이터에 대해 난수 없이 결정론적으로 적합된다.

[2. 8대 예측 헤드 명세]
  * M2Rate  (포아송 회귀) : 시간 구간(q)별 초당 첫 킬 발생률 λ (3개 게이트 윈도우 상호작용 반영) [Masked 전용]
  * M2Mark  (다항 로지스틱) : 첫 킬을 낸 선제 킬 표지 m (52개 클래스: 50개 킬러-피해자 슬롯 쌍 + 2개 처형) [Masked 전용]
  * M2Zone  (다항 로지스틱) : 첫 킬 발생 위치 z (10개 맵 구역 + 1개 미식별 구역 -1) [Masked 전용]
  * M3Size  (다항 로지스틱) : 교전 규모 s (4개 범주 0: T, 1: S, 2: ASYM, 3: P = units.SIZE_NAMES; 킬 크레딧 참여 인원으로 정한다:
                              T 양 팀 4명 이상, S 양 팀 2~3명, ASYM 한 팀 2~3명 대 상대 4명 이상, P 한 팀이 2명 미만)
  * M3Kills (포아송 회귀) : 교전 결과 양 팀 킬 수 k_blue, k_red (블루/레드 독립 2개 모델)
  * M3Gold  (다항 로지스틱) : 교전 결과 획득 골드 구간 (팀별 31개 골드 구간, 6개 킬 수 층별 조건부)
  * M3Obj   (포아송 회귀) : 교전 중 획득 오브젝트 수 (18개 오브젝트 종류/팀별 독립 포아송 셀)
  * M4Conv  (포아송 회귀) : 교전 종료 후 60초 전환창 내 오브젝트 전환 획득 수 (18개 셀, 노출시간 보정)

[3. 데이터 뷰 및 행 구조]
- Masked (가린 헤드): 합성기가 호출하는 시점(Onset, t0 분 프레임)에서 알 수 있는 사전 정보만 조건으로 사용.
  - M3Obj는 실제 킬(K)을 읽지 않고, M4Conv는 K와 O를 읽지 않아 결합 확률이 닫힌 꼴로 인수분해됨.
- Open (열린 헤드): 실제 S2 교전 사건(Event) 관측치를 읽는 사후 평가용 헤드. 실제 크기 s, 실제 킬 K, 실제 오브젝트 O를 조건에 추가.

[4. 핵심 수식 및 알고리즘 해석]
- (B0-1) 수치 열 표준화: x̃_h = (x_h - x̄_h) / σ_h (훈련 데이터 기준 평균과 모표준편차로 정규화)
- (B0-2) 포아송 헤드: λ_c = exp(η_c), 평균 μ_c = e * λ_c (노출시간 e를 오프셋으로 하는 가중 deviance 최소화)
- (B0-3) 미발생 셀 처리: 훈련 데이터에서 사건이 0건인 셀은 λ_c = 0.5 / Σ e_i (Jeffreys 사전분포 사후 평균) 부여
- (B0-4) M2Rate 구간별 기울기: log λ_q(x) = γ_q + β_w(q) · x̃ (60s, 120s, 180s 게이트 윈도우 w(q)=⌊q/6⌋별 독립 기울기)
- (B0-5) 다항 로지스틱 헤드: p_c = softmax(η_c) (L2 능형 규제 다항 분류)
- (B0-6) 균등 분포 스무딩: p = (1 - 1e-6) * p_model + 1e-6 / C (미발생 클래스에 최소 확률 부여로 NLL 발산 방지)
- (B0-7) 선형 예측자 표 조회: η_c = γ_c + x̃ · β_c + Σ ω_c[level] (설계 행렬 재생성 없이 사전 계산된 표 조회)
- (B0-8) 원래 척도 복원: β^raw = β / σ, γ^raw = γ - Σ (β * x̄ / σ)

[5. 하네스 수렴 및 검증 계약]
- 엄격한 기울기 허용오차(tol=1e-8) 적용으로 정확한 최대우도 수렴 보장.
- scikit-learn 수렴 경고를 포착하여 converged_ 플래그 및 커스텀 경고 발생.
"""
from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from c_ept.preprocessing import features
from c_ept.data import head_rows, store_core
from c_ept.models import heads, killlaw, nll as nll_scores
from c_ept.data.casts import narrow
# the names of the heads, their shapes, and the rows and scores that B0 and B1 share (ruling R18); ``baselines_b0.HeadData`` etc. stay
from c_ept.data.head_rows import (HEAD_NAMES, KINDS, MIX, N_CELLS, N_GOLD, N_K, N_LAYER, N_M, N_Q, N_SIZE, N_ZONE, OPEN_HEAD_NAMES,  # noqa: F401
                             PREDICT_ALL_WIDTHS, TEAMS, HeadBatch, HeadData, HeadScores)

# ==========================================================================================================================
# 1. 모델 적합 하이퍼파라미터 및 하네스 상수 정의
# ==========================================================================================================================
# PoissonRegressor: L2 규제 강도 (alpha). 거의 순수 최대우도추정(MLE)에 가깝도록 매우 작은 값(1e-6) 사용
ALPHA = 1e-6
# PoissonRegressor: 수렴 판정 기울기 허용오차 (tol). scikit-learn 기본값(1e-4)은 최적점에서 조기 종료되므로 1e-8로 엄격화
TOL_POISSON = 1e-8
MAX_ITER_POISSON = 1000

# LogisticRegression: 역정규화 강도 C (1.0) 및 수렴 허용오차 (tol)
C_LOGISTIC = 1.0
MAX_ITER_LOGISTIC = 2000
TOL_LOGISTIC = 1e-8

# B0 저장 스키마 버전: v3 (M2Rate 게이트 윈도우별 상태 계수 벡터 분리 및 M3Kills/M3Gold 팀별 독립 모델)
SCHEMA = 'b0_heads_v3'
NPZ_NAME, JSON_NAME = 'b0_heads.npz', 'b0_heads.json'

N_B0 = features.N_B0            # 39: 기본 시간-상태 특징 수
_CHUNK = 1 << 16                # 65,536: 통계 계산 및 표준화 시 메모리 절약을 위한 청크 크기

# M2Rate 게이트 윈도우 정의: q 구간(0~17)을 3개 구간([0,6), [6,12), [12,18))으로 분할.
# 60초, 120초, 180초 관측 지평에 대응하며 각 구간 w(q)마다 독립적인 상태 기울기 계수를 가짐.
M2_WINDOWS = ((0, 6), (6, 12), (12, 18))
if [q for lo, hi in M2_WINDOWS for q in range(lo, hi)] != list(range(N_Q)):
    raise RuntimeError('M2_WINDOWS must cover the bins 0 .. N_Q - 1 once, in order')
_WINDOW_OF_Q = np.repeat(np.arange(len(M2_WINDOWS)), [hi - lo for lo, hi in M2_WINDOWS])        # (18,) 각 구간 q의 윈도우 인덱스
_WINDOW_OF_Q.setflags(write=False)


# ==========================================================================================================================
# 2. 설계 행렬(Design Matrix) 구조: One-Hot 블록 및 레이아웃 정의
# ==========================================================================================================================
class Block(NamedTuple):
    """범주형 변수의 One-Hot 인코딩 블록 정의.

    - name: 범주 변수명 ('q', 'm', 'z', 's', 'layer')
    - lo: 최소 정수 코드값 (예: z는 -1, q는 0)
    - n: 범주 수준(수준의 총 개수)
    - drop_first: 첫 번째(최소) 수준을 제외할지 여부.
      - True (포아송): 절편(intercept)이 존재하므로 다중공선성(collinearity) 방지를 위해 기준 수준 계수를 0으로 제외 (n-1개 열).
      - False (로지스틱): L2 능형 규제가 모든 수준을 식별하므로 전체 수준 열 유지 (n개 열).
    """
    name: str
    lo: int
    n: int
    drop_first: bool

    @property
    def width(self) -> int:
        """이 블록이 차지하는 열의 개수."""
        return self.n - (1 if self.drop_first else 0)


class Layout(NamedTuple):
    """헤드별 전체 입력 설계 행렬의 열 구성 레이아웃.

    - n_numeric: 표준화되는 연속형 수치 열의 개수 (기본 39차원 b0 특징, open 헤드는 추가 K/O 포함).
    - blocks: 뒤따르는 One-Hot Block들의 튜플.
    - windows: M2Rate 전용 게이트 윈도우 튜플. (윈도우가 있으면 수치 열이 윈도우 개수만큼 반복됨).
    """
    n_numeric: int
    blocks: tuple
    windows: tuple = ()

    @property
    def width(self) -> int:
        """호출자가 헤드에 전달하는 원본 설계 행렬의 열 수 [수치열 + 원핫열]."""
        return self.n_numeric + sum(b.width for b in self.blocks)

    @property
    def model_width(self) -> int:
        """실제 내부 모델이 사용하는 계수 벡터의 총 열 수: D * max(1, 윈도우수) + sum(블록 열)."""
        return self.n_numeric * max(1, len(self.windows)) + sum(b.width for b in self.blocks)


# 각 범주 변수별 유효 정수 범위: (최소값, 수준 개수)
# 'd'는 이동 킬 법칙(결정 D5)의 열린 M3Kills 머리에만 있는 첫 킬 팀이다: −1 처형(팀 없음), 0 파랑, 1 빨강. LAYOUTS에는 없고 layout_for가 이동 법칙에서만 더한다.
_RANGE = {'q': (0, N_Q), 'm': (0, N_M), 'z': (-1, N_ZONE + 1), 's': (0, N_SIZE), 'layer': (1, N_LAYER), 'd': (-1, 3)}


def _blocks(drop: bool, *names: str) -> tuple:
    """지정된 변수 이름 순서대로 Block 튜플 생성. drop은 기준 수준 제외 여부."""
    return tuple(Block(name, *_RANGE[name], drop) for name in names)


# 헤드별 설계 행렬 스키마 정의 (식 B0-7 기준)
# - masked: 합성기 전용 (Onset 시점 정보만 사용, K/O 누출 방지)
# - open: 실제 관측 사후 평가용 (실제 K, O 조건 추가)
LAYOUTS = {
    ('M2Rate', 'masked'): Layout(N_B0, _blocks(True, 'q'), M2_WINDOWS),           # 윈도우별 상태 벡터 상호작용
    ('M2Mark', 'masked'): Layout(N_B0, _blocks(False, 'q')),
    ('M2Zone', 'masked'): Layout(N_B0, _blocks(False, 'q', 'm')),
    ('M3Size', 'masked'): Layout(N_B0, _blocks(False, 'q', 'm', 'z')),
    ('M3Kills', 'masked'): Layout(N_B0, _blocks(True, 'q', 'm', 'z', 's')),
    ('M3Gold', 'masked'): Layout(N_B0, _blocks(False, 'q', 'm', 'z', 's', 'layer')),
    ('M3Obj', 'masked'): Layout(N_B0, _blocks(True, 'q', 'm', 'z', 's')),
    ('M4Conv', 'masked'): Layout(N_B0, _blocks(True, 'q', 'm', 'z')),
    ('M3Size', 'open'): Layout(N_B0, ()),
    ('M3Kills', 'open'): Layout(N_B0, _blocks(True, 's')),
    ('M3Gold', 'open'): Layout(N_B0, _blocks(False, 's', 'layer')),
    ('M3Obj', 'open'): Layout(N_B0 + N_K, _blocks(True, 's')),
    ('M4Conv', 'open'): Layout(N_B0 + N_K + N_CELLS, ()),
}


def layout_for(head: str, kind: str, kill_law: str = 'poisson') -> Layout:
    """The layout of the head ``head`` of ``kind`` for the kill law ``kill_law``: ``LAYOUTS[(head, kind)]`` itself for the plain law and for every head but the open
    M3Kills head. Under the shifted law (decision D5) the open M3Kills head gets the one-hot block of the first-kill team ``d`` after ``s``, because it is the only
    kill head that does not see the mark m, and each team's cell must be able to say whether its rate is the rate λ_d of the excess N_d or the rate μ of the plain count
    (the masked head has the mark in its blocks already).

    수식
    ----
    (B0-9)  η_{t}(x, s, d) = γ_t + x̃·β_t + ω^s_t[s] + ω^d_t[d]          (열린 M3Kills, 이동 법칙; 팀 t의 칸마다 따로)
            d = t이면 exp(η_t) = λ_t (초과분 N_t의 평균), d ≠ t이면 exp(η_t) = μ_t (킬 수의 평균)

    해석: d의 세 수준(처형, 파랑, 빨강) 가운데 첫 수준(처형)이 기준이다. 칸 t는 d = t인 행과 d ≠ t인 행을 다른 절편으로 맞추므로 한 칸의 모형이 λ와 μ를 모두 낼 수 있다.
    """
    killlaw.check_law(kill_law)
    if kill_law == 'shifted' and (head, kind) == ('M3Kills', 'open'):
        return Layout(N_B0, _blocks(True, 's', 'd'))
    return LAYOUTS[(head, kind)]


# ==========================================================================================================================
# 3. 입력 검증, 원핫 인코딩 및 열 이름 생성 유틸리티
# ==========================================================================================================================
def _int_codes(name: str, values) -> np.ndarray:
    """입력 배열을 1차원 int64 배열로 변환 및 정수형 타입 검증."""
    a = np.asarray(values)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must be an integer array, not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'{name} must be 1-d, not shape {a.shape}')
    return a.astype(np.int64)


def _level(block: Block, values) -> np.ndarray:
    """범주형 값을 0부터 시작하는 수준 인덱스(value - lo)로 변환하고 유효 범위를 검사."""
    a = _int_codes(block.name, values)
    if a.size and (a.min() < block.lo or a.max() > block.lo + block.n - 1):
        raise ValueError(f'{block.name} must lie in {block.lo} .. {block.lo + block.n - 1}')
    return a - block.lo


def _design(numeric: np.ndarray, blocks, codes: Mapping) -> np.ndarray:
    """수치형 행렬과 One-Hot 범주 블록들을 결합하여 식 (B0-7)의 입력 설계 행렬 [x | One-Hot] 생성.

    - numeric: (n, a) 표준화된 연속형 수치 배열
    - blocks: One-Hot 블록들의 튜플
    - codes: 변수명 -> 범주값 1차원 배열 매핑
    - 반환값: (n, a + sum(block.width)) float64 2차원 설계 행렬
    """
    n, a = numeric.shape
    X = np.zeros((n, a + sum(b.width for b in blocks)))
    X[:, :a] = numeric
    at = a
    for b in blocks:
        col = _level(b, codes[b.name]) - (1 if b.drop_first else 0)
        if len(col) != n:
            raise ValueError(f'{b.name} has {len(col)} entries but the numeric part has {n} rows')
        hit = np.flatnonzero(col >= 0)
        X[hit, at + col[hit]] = 1.0
        at += b.width
    return X


def column_names(layout: Layout, numeric_names=()) -> list:
    """모델 계수 행렬의 각 열에 대응하는 설명적 열 이름 목록 생성 (JSON 직렬화 및 디버깅용)."""
    names = list(numeric_names) + [f'x{j}' for j in range(len(numeric_names), layout.n_numeric)]
    if layout.windows:
        names = [f'w{w}:{name}' for w in range(len(layout.windows)) for name in names]
    for b in layout.blocks:
        names += [f'{b.name}={b.lo + level}' for level in range(1 if b.drop_first else 0, b.n)]
    return names


# ==========================================================================================================================
# 4. 수치 열 표준화 및 메모리 효율적 2-패스 통계 연산 (식 B0-1)
# ==========================================================================================================================
def _column_stats(X: np.ndarray, n: int) -> tuple:
    """연속형 수치 열(앞선 n개 열)의 표본 평균과 모표준편차를 청크 단위 2-패스로 계산 (식 B0-1).

    - 분산이 0(단일 상수값)인 열은 scale을 1.0으로 두어 표준화 시 0으로 변환되도록 처리.
    - 반환: (mean, scale) 각각 형상 (n,)
    """
    total, low, high = np.zeros(n), np.full(n, np.inf), np.full(n, -np.inf)
    for a in range(0, len(X), _CHUNK):
        block = X[a:a + _CHUNK, :n]
        total += block.sum(axis=0)
        low, high = np.minimum(low, block.min(axis=0)), np.maximum(high, block.max(axis=0))
    constant = low == high
    mean = np.where(constant, low, total / len(X))
    squares = np.zeros(n)
    for a in range(0, len(X), _CHUNK):
        d = X[a:a + _CHUNK, :n] - mean
        squares += (d * d).sum(axis=0)
    return mean, np.where(constant, 1.0, np.sqrt(squares / len(X)))


def _standardise_(X: np.ndarray, n: int, mean: np.ndarray, scale: np.ndarray) -> None:
    """X 행렬의 앞선 n개 수치 열을 제자리(in-place)에서 표준화: (X - mean) / scale (식 B0-1)."""
    for a in range(0, len(X), _CHUNK):
        block = X[a:a + _CHUNK, :n]
        block -= mean
        block /= scale


def _check_finite(X: np.ndarray, what: str = 'X') -> None:
    """행렬 내에 NaN 또는 무한대(Inf)가 포함되어 있는지 청크 단위로 검증."""
    for a in range(0, len(X), _CHUNK):
        if not np.isfinite(X[a:a + _CHUNK]).all():
            raise ValueError(f'{what} holds a NaN or an infinite value')


def _design_matrix(X) -> np.ndarray:
    """입력 배열이 2차원 실수형 설계 행렬인지 검증하고 float64 ndarray로 변환."""
    A = np.asarray(X)
    if A.dtype.kind not in 'fiub':
        raise TypeError(f'the design must be numeric, not {A.dtype}')
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError(f'the design must be a 2-d array with rows and columns, not shape {A.shape}')
    return A


# ==========================================================================================================================
# 5. 모델 최적화 수렴 모니터링 및 경고 처리 (하네스 엄격 수렴 규약)
# ==========================================================================================================================
def _fit_recording(model, *args, **kwargs) -> bool:
    """scikit-learn 모델의 fit()을 실행하고 ConvergenceWarning 발생 여부를 감지하여 반환."""
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(*args, **kwargs)
    fired = False
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            fired = True
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    return fired


def _converged(fired: bool, n_iter: int, limit: int) -> bool:
    """경고가 발생하지 않았고 반복 횟수가 상한선 미만일 때만 수렴(True)으로 판정."""
    return (not fired) and n_iter < limit


def _warn_not_converged(label: str, cell, n_iter: int, limit: int, fired: bool) -> None:
    """수렴하지 않은 헤드 및 셀에 대해 상세한 원인과 함께 경고 발생."""
    from sklearn.exceptions import ConvergenceWarning
    where = label if cell is None else f'{label} cell {cell}'
    why = 'scikit-learn raised a ConvergenceWarning' if fired else 'the iteration limit was reached'
    warnings.warn(f'{where}: the optimiser did not converge ({why}; {n_iter} iterations, the limit is {limit})', ConvergenceWarning,
                  stacklevel=3)


# ==========================================================================================================================
# 6. 기저 헤드(Base Head) 추상 클래스군: _Head, _PoissonHead, _CategoricalHead
# ==========================================================================================================================
class _Head:
    """포아송 헤드와 다항 로지스틱 헤드가 공유하는 공통 기저 추상 클래스.

    [역할]
    - 연속형 수치 열의 표본 표준화 파라미터(mean_, scale_) 관리 (식 B0-1).
    - 표준화된 계수(coef_std_, intercept_std_)와 원본 척도 복원 계수(coef_, intercept_) 계산 (식 B0-8).
    - 사전 계산된 One-Hot 계수 테이블(_tables)을 구축하여 합성기(Composer)의 초고속 선형 예측 지원 (식 B0-7).
    - 두 가지 추론 방식 제공:
      1) decision(X): (n, d) 2차원 설계 행렬 전체를 곱하는 기본 방식.
      2) decision_blocks(numeric, **codes): 설계 행렬을 빌드하지 않고 수치 항과 표 조회 합으로 계산하는 닫힌 꼴 방식.
    수식
    ----
    (B0-1)  x̃_{ih} = (x_{ih} − x̄_h) / σ_h,    x̄_h = (1/N)·Σ_i x_{ih},    σ_h = √[(1/N)·Σ_i (x_{ih} − x̄_h)²]        (h = 1, …, D)
            열 h에 값이 하나뿐이면 x̄_h는 그 값이고 σ_h = 1이다(x̃_{ih} = 0). 원핫 열은 표준화하지 않는다.

      기호   | 뜻                                          | 단위    | 범위                   | 코드 변수
      x_{ih} | 설계 행 i의 숫자 열 h의 원래 값             | 열마다  | 유한                   | X[i, h]
      x̃_{ih} | 표준화한 값                                 | 무차원  | 유한                   | 표준화한 X[i, h]
      x̄_h    | 열 h의 평균(가중 없음)                      | 열 단위 | 유한                   | mean_
      σ_h    | 열 h의 모표준편차(분모 N)                   | 열 단위 | > 0 (값이 하나뿐이면 1) | scale_
      N      | 이 머리의 적합에 쓰는 행의 수               | 행      | ≥ 1                    | n_fit_
      D      | 숫자 열의 수                                | 열      | 0, …, d                | n_numeric_
      d      | 설계의 열 수(숫자 열과 원핫 열)             | 열      | ≥ 1                    | X.shape[1]

    해석
    ----
    식 (B0-1)은 숫자 열마다 평균을 빼고 표준편차로 나누어 열의 단위를 없앤다. 능선 벌점은 열의 단위에 따라 달라지므로 표준화한 열에 건다 [도출].
    평균과 표준편차는 그 머리의 적합에 실제로 쓴 FIT 행(노출이 0이거나 범주가 없는 행은 뺀 행)의 값이며 가중하지 않는다. 가린 머리에서 이 행은 FIT 프레임 보기가 아니라 그 머리의
    가린 행이다(발생 하나, M3Gold는 그 발생의 팀 항목, M2Rate는 (t₀ 행, 구간) 쌍; 판정 R65). 값이 하나뿐인 열은 σ_h = 1로 두어 0으로 나누지 않고 0 열이 된다(모듈 글의
    설계 문단). 원핫 열은 이미 0과 1이어서 표준화하지 않는다.

    근거
    ----
    계획 Task 6(표준화), P9(행마다 같은 무게, 가중 없음), 판정 R65(머리 자신의 FIT 학습 행), 모듈 글의 설계 문단."""
    name = ''
    WINDOWS = ()                    # the bin windows for which the numeric part is repeated in the model (M2Rate only: G-S3b B2-a)

    @property
    def parts(self) -> tuple:
        """The single models that this head is made of (itself; the gold head has one for each team)."""
        return (self,)

    def __init__(self, kind: str = 'masked'):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        self.kind = kind
        self._label = self.name                                  # what a warning calls this head (a member of M3Gold: with its team)
        self.n_numeric_ = 0
        self.blocks_ = ()
        self.mean_ = self.scale_ = None
        self.coef_std_ = self.intercept_std_ = None
        self.n_fit_ = 0
        self.n_iter_ = self.converged_ = None
        self._tables = {}

    # -- the layout and the data -------------------------------------------------------------------------------------------------------
    def _set_layout(self, d: int, n_numeric, blocks) -> None:
        n = d if n_numeric is None else n_numeric
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 0 <= n <= d:
            raise ValueError(f'n_numeric must be a whole number in 0 .. {d} (the columns of the design), not {n_numeric!r}')
        blocks = tuple(Block(*b) for b in blocks)
        if blocks and int(n) + sum(b.width for b in blocks) != d:
            need = int(n) + sum(b.width for b in blocks)
            raise ValueError(f'the design has {d} columns but n_numeric {int(n)} and the blocks need {need} columns')
        self.n_numeric_, self.blocks_ = int(n), blocks

    def _training_design(self, X, keep, overwrite_X: bool) -> np.ndarray:
        """The rows ``keep`` of ``X`` as a standardised float64 design; ``mean_`` and ``scale_`` are those of these rows. ``X`` is changed
        only when ``overwrite_X`` and no row is left out.

        (B0-1)의 표준화를 적합에 쓸 행에서만 한다."""
        if keep is not None and not keep.all():
            X = X[keep]                                                                     # a copy: ours to standardise in place
        elif not overwrite_X or not X.flags.writeable:
            X = X.copy()
        n = self.n_numeric_
        self.mean_, self.scale_ = _column_stats(X, n)
        _standardise_(X, n, self.mean_, self.scale_)
        return X

    # -- the parameters ------------------------------------------------------------------------------------------------------------------
    def _store(self, coef_std: np.ndarray, intercept_std: np.ndarray) -> None:
        self.coef_std_, self.intercept_std_ = np.asarray(coef_std, dtype=np.float64), np.asarray(intercept_std, dtype=np.float64)
        self._build_tables()

    def _block_start(self) -> int:
        """The column of the coefficients where the first one-hot block starts: after the numeric columns (once for each window)."""
        return self.n_numeric_ * max(1, len(self.WINDOWS))

    def _model_width(self, d: int) -> int:
        """The columns of the model for a caller design of ``d`` columns: the same for every head but M2Rate."""
        return d

    def _model_design(self, Xs: np.ndarray) -> np.ndarray:
        """The design that the model is fitted on, made of the standardised design that the caller gave: the same (not a copy) for every
        head but M2Rate."""
        return Xs

    def _build_tables(self) -> None:
        """The coefficient of every level of every block, (n levels, J), the reference level being a row of zeros.

        (B0-7)의 표 ω를 만든다."""
        at = self._block_start()
        self._tables = {}
        for b in self.blocks_:
            table = np.zeros((b.n, self.coef_std_.shape[0]))
            table[(1 if b.drop_first else 0):] = self.coef_std_[:, at:at + b.width].T
            self._tables[b.name] = table
            at += b.width

    def _raw(self) -> tuple:
        """(B0-8)의 원래 척도의 계수 β^raw와 절편 γ^raw를 낸다."""
        n = self.n_numeric_
        W = self.coef_std_.copy()
        W[:, :n] = W[:, :n] / self.scale_
        return W, self.intercept_std_ - (self.coef_std_[:, :n] * (self.mean_ / self.scale_)).sum(axis=1)

    @property
    def coef_(self) -> np.ndarray:
        """The coefficients on the raw (unstandardised) design: (d,) for a head with one cell fitted on a 1-d target, else (J, d).

        수식
        ----
        (B0-8)  β^raw_{c,h} = β_{c,h} / σ_h,    γ^raw_c = γ_c − Σ_h β_{c,h}·x̄_h / σ_h,    원핫 열의 계수 ω는 그대로이다
                M2Rate: 창 w의 기울기는 β_{w,h} / σ_h이고, ζ_w = Σ_h β_{w,h}·x̄_h / σ_h일 때 절편은 γ_0 − ζ_0이며
                        구간 q ≥ 1의 열은 (γ_q − γ_0) − ζ_{w(q)} + ζ_0이다

          기호         | 뜻                                              | 단위                | 범위    | 코드 변수
          β_{c,h}      | 표준화한 입력에서 출력 c의 숫자 열 h의 계수     | 로그 척도           | 유한    | coef_std_[c, h]
          β^raw_{c,h}  | 원래 입력에서의 같은 계수                       | 로그 척도 / 열 단위 | 유한    | coef_[c, h]
          γ_c, γ^raw_c | 표준화한 입력과 원래 입력에서의 절편            | 로그 척도           | 유한    | intercept_std_[c], intercept_[c]
          x̄_h, σ_h     | 열 h의 평균과 모표준편차                        | 열 단위             | σ_h > 0 | mean_, scale_
          ζ_w          | 창 w의 이동량 Σ_h β_{w,h}·x̄_h / σ_h            | 로그 척도           | 유한    | shift

        해석
        ----
        식 (B0-8)은 표준화한 입력에서 맞춘 η = γ_c + x̃·β_c를 같은 값을 내는 원래 척도의 모형으로 옮긴다. x̃_{ih} = (x_{ih} − x̄_h) / σ_h를 대입하면 기울기는 σ_h로
        나뉘고 평균을 뺀 만큼은 절편으로 옮겨 간다 [도출]. 원핫 열은 표준화하지 않았으므로 계수가 그대로이다. M2Rate는 창마다 기울기가 달라 이동량 ζ_w도 창마다
        다르며, 창 사이의 차이를 구간 열이 맡아서 모든 구간의 발생률이 표준화한 형과 같다.

        근거
        ----
        (B0-1), 판정 R36(M2Rate의 창별 기울기), 계획 Task 6(표준화)."""
        W = self._raw()[0]
        return W[0] if getattr(self, 'squeeze_', False) else W

    @property
    def intercept_(self):
        """(B0-8)의 γ^raw_c이다."""
        b = self._raw()[1]
        return float(b[0]) if getattr(self, 'squeeze_', False) else b

    def _need_fit(self) -> None:
        if self.coef_std_ is None:
            raise RuntimeError(f'{self.name} is not fitted')

    # -- the linear predictor ------------------------------------------------------------------------------------------------------------
    def decision(self, X) -> np.ndarray:
        """(n, J) the linear predictor of the full raw design ``X``.

        (B0-7)의 η를 설계 행렬 X = [x | 원핫 열]로 계산한다."""
        self._need_fit()
        X = np.asarray(X, dtype=np.float64)
        n = self.n_numeric_
        if X.ndim != 2 or X.shape[1] != self.coef_std_.shape[1]:
            raise ValueError(f'{self.name}: the design must have {self.coef_std_.shape[1]} columns, not shape {X.shape}')
        Z = ((X[:, :n] - self.mean_) / self.scale_) @ self.coef_std_[:, :n].T
        if X.shape[1] > n:
            Z += X[:, n:] @ self.coef_std_[:, n:].T
        return Z + self.intercept_std_

    def _numeric_part(self, numeric) -> np.ndarray:
        """(n, J): the numeric part of the linear predictor, intercept included.

        (B0-7)의 γ_c + x̃_i·β_c를 계산한다."""
        self._need_fit()
        A = np.asarray(numeric, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != self.n_numeric_:
            raise ValueError(f'{self.name}: the numeric part must have {self.n_numeric_} columns, not shape {A.shape}')
        return ((A - self.mean_) / self.scale_) @ self.coef_std_[:, :self.n_numeric_].T + self.intercept_std_

    def decision_blocks(self, numeric, **codes) -> np.ndarray:
        """(n, J) the linear predictor from the raw numeric part and the integer codes of every one-hot block (keyword = block name).

        수식
        ----
        (B0-7)  η_{ic} = γ_c + x̃_i·β_c + ω^q_c[q_i] + ω^m_c[m_i] + ω^z_c[z_i] + ω^s_c[s_i] + ω^ξ_c[ξ_i]            (머리가 가진 블록의 항만)
                ω^ψ_c[수준]은 블록 ψ(q, m, z, s, ξ 중 하나)의 그 수준의 계수이고, 기준 수준을 둔 블록(포아송 머리)에서는 기준 수준의 ω가 0이다.

          기호                    | 뜻                                              | 단위      | 범위                        | 코드 변수
          η_{ic}                  | 행 i의 출력 c의 선형 예측량                     | 로그 척도 | 유한                        | 반환값 (n, J)
          γ_c, β_c                | 출력 c의 절편과 숫자 열 계수(표준화한 입력에서) | 로그 척도 | 유한                        | intercept_std_, coef_std_
          x̃_i                    | 행 i의 표준화한 숫자 열(D개)                    | 무차원    | 유한                        | numeric에서 (B0-1)로
          q_i, m_i, z_i, s_i, ξ_i | 행 i의 범주 번호(블록의 수준)                   | —         | 0–17, 0–51, −1–9, 0–3, 1–6  | codes
          ω^ψ_c[수준]             | 블록 ψ의 그 수준의 계수                         | 로그 척도 | 유한; 기준 수준은 0         | _tables[ψ][수준, c]

        해석
        ----
        식 (B0-7)은 선형 예측량이 숫자 열의 선형 항과 범주마다의 표 항의 합이라는 로그 선형 모형 그 자체이며, 원핫 열과 계수의 곱은 표에서 그 수준의 계수를 읽는
        것과 같다 [도출]. 그래서 합성기는 설계 행렬을 만들지 않고 숫자 항 한 번과 표 찾기 몇 번으로 값을 내어 수백만 행을 물을 수 있다(모듈 글의 합성기 문단, 계획 Task 6). 포아송 머리에서
        exp(η)는 발생률이어서 0보다 크고(설계서 10절 A2), 분류 머리에서 η는 범주 점수이며 (B0-6)을 거쳐 확률이 된다. 머리가 가진 블록의 코드를 모두
        주어야 하고 수준이 블록에 정해 둔 범위 밖이면 거부한다.

        근거
        ----
        계획 Task 6(합성기 머리), 판정 R3(합성기 규약), 모듈 글의 설계 문단(블록과 기준 수준)과 합성기 문단(표 찾기), 설계서 10절 A2(λ는 로그 링크).
        Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
             doi:10.1017/CBO9781139013567 (로그 링크). Hastie, T., Tibshirani, R. and Friedman, J. (2009), "The Elements of
             Statistical Learning", 2nd ed., Springer; doi:10.1007/978-0-387-84858-7 (다항 로지스틱의 선형 예측량, 4.4절)."""
        if self.n_numeric_ + sum(b.width for b in self.blocks_) != self.coef_std_.shape[1]:
            raise ValueError(f'{self.name}: the design has columns that are not described by blocks_, so the block form cannot be used')
        Z = self._numeric_part(numeric)
        names = {b.name for b in self.blocks_}
        if set(codes) != names:
            raise ValueError(f'{self.name}: the codes must be given for the blocks {sorted(names)}, not {sorted(codes)}')
        for b in self.blocks_:
            level = _level(b, codes[b.name])
            if len(level) != len(Z):
                raise ValueError(f'{b.name} has {len(level)} entries but the numeric part has {len(Z)} rows')
            Z += self._tables[b.name][level]
        return Z

    def predict_blocks(self, numeric, **codes) -> np.ndarray:
        """선형 예측량 (B0-7)을 머리의 출력으로 바꾼다: 포아송 머리는 발생률 exp(η)((B0-2)), 분류 머리는 섞은 확률 p((B0-6))이다."""
        return self._output(self.decision_blocks(numeric, **codes))

    def predict(self, X) -> np.ndarray:
        """같은 출력을 설계 행렬 X로 낸다."""
        return self._output(self.decision(X))

    # -- storage -------------------------------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        """(arrays, meta): the arrays are the numeric fields, the meta is a dict of plain JSON values."""
        self._need_fit()
        arrays = {'coef_std': self.coef_std_, 'intercept_std': self.intercept_std_, 'mean': self.mean_, 'scale': self.scale_,
                  'n_iter': np.asarray(self.n_iter_, dtype=np.int64), 'converged': np.asarray(self.converged_, dtype=np.bool_)}
        meta = {'class': self.name, 'kind': self.kind, 'n_numeric': self.n_numeric_, 'n_fit': int(self.n_fit_),
                'blocks': [[b.name, b.lo, b.n, bool(b.drop_first)] for b in self.blocks_], 'width': int(self.coef_std_.shape[1])}
        return arrays, meta

    def _restore(self, arrays: Mapping, meta: Mapping) -> None:
        self.kind = meta['kind']
        self.n_numeric_ = int(meta['n_numeric'])
        self.blocks_ = tuple(Block(*b) for b in meta['blocks'])
        self.n_fit_ = int(meta['n_fit'])
        self.mean_, self.scale_ = np.array(arrays['mean']), np.array(arrays['scale'])
        self.n_iter_, self.converged_ = np.array(arrays['n_iter']), np.array(arrays['converged'])
        self._store(np.array(arrays['coef_std']), np.array(arrays['intercept_std']))

    @classmethod
    def from_state(cls, arrays: Mapping, meta: Mapping) -> '_Head':
        head = cls(meta['kind'])
        head._restore(arrays, meta)
        return head

    def summary(self) -> dict:
        """The counts of the fit as plain JSON values."""
        self._need_fit()
        return {'kind': self.kind, 'n_fit': int(self.n_fit_), 'width': int(self.coef_std_.shape[1]), 'n_numeric': self.n_numeric_,
                'converged': [bool(c) for c in self.converged_], 'n_iter': [int(i) for i in self.n_iter_]}


class _PoissonHead(_Head):
    """로그 링크 함수와 노출 시간(exposure) 오프셋을 갖는 포아송 회귀 기저 헤드.

    [역할 및 특징]
    - 단위 노출시간당 사건 발생률(초당 킬 수, 오브젝트 획득률 등)을 추정.
    - 각 셀(종속변수 열)마다 독립적인 포아송 회귀 모형을 적합(식 B0-2).
    - 노출시간 e > 0인 유효 행만 학습에 사용하며, e=0인 행은 자동으로 제외 및 기록.
    - 사건이 전혀 발생하지 않은 셀(degenerate_)은 최대우도추정 불가이므로 Jeffreys 사전분포의 사후 평균 상수 모형(식 B0-3) 적용.
    - 주요 출력:
      - rate(X): (n, J) 단위 노출당 추정 발생률 λ = exp(η)
      - expected(X, exposure_s): (n, J) 기대 발생 횟수 μ = e * λ
      - nll(X, Y, e): (n,) 또는 (n, J) 포아송 음의 로그우도(NLL) 점수

    수식
    ----
    (B0-2)  λ_{ic} = exp(η_{ic}),   μ_{ic} = e_i·λ_{ic},   ℓ_{ic} = μ_{ic} − y_{ic}·log μ_{ic} + log(y_{ic}!)            ((NLL-1)의 칸 점수)
            (γ̂_c, β̂_c, ω̂_c) = argmin [ (1 / Σ_i e_i)·Σ_i ℓ_{ic} + (α/2)·(‖β_c‖² + ‖ω_c‖²) ]     (합은 e_i > 0인 행에 대해서; 칸 c마다 따로)
            맞춤은 목표 y/e와 표본 가중 e의 포아송 회귀이며, 노출 오프셋 log e를 둔 이탈도와 같은 해를 준다 [도출].
    (B0-3)  칸 c에 사건이 없으면(Σ_i y_{ic} = 0) 상수 모형 λ_c = 0.5 / Σ_i e_i를 쓴다(β_c = 0, ω_c = 0).

      기호          | 뜻                            | 단위              | 범위             | 코드 변수
      y_{ic}        | 행 i의 칸 c에서 센 개수       | 개                | 0, 1, …          | Y[i, c]
      e_i           | 행 i의 노출                   | s (M3는 1)        | > 0인 행만 적합  | e
      λ_{ic}        | 칸 c의 예측 발생률            | 1/s (M3는 개)     | > 0              | predict의 반환값
      μ_{ic}        | 기대 개수                     | 개                | > 0              | expected
      ℓ_{ic}        | 칸 점수(음의 로그 우도)       | nat               | ≥ 0              | nll(..., cells=True)
      γ_c, β_c, ω_c | 절편, 숫자 열 계수, 원핫 계수 | 로그 척도         | 유한             | intercept_std_, coef_std_
      α             | 능선 벌점 계수                | 1/s (M3는 무차원) | > 0              | ALPHA

    해석
    ----
    식 (B0-2)는 칸마다 개수를 평균 μ_{ic} = e_i·λ_{ic}의 포아송으로 보고 로그 발생률 η_{ic}를 선형으로 두어 노출 합으로 나눈 (NLL-1)의 합을 낮추는 계수를 찾는 것이며,
    벌점 α는 최대우도 해에 가깝도록 작게 둔다. 노출 e_i는 기대 개수에 곱해지고, 목표 y/e와 가중 e로 맞추는 것이 오프셋 log e를 둔 포아송 이탈도와 같으므로 이 형을 쓴다
    (계획 Task 6) [도출]. 노출이 0인 행은 기대 개수가 0이어서 우도에 아무것도 더하지 않으므로 적합에서 빼고(계획 전역 제약의 확률 족, Task 6) 그 수를 기록하며(판정 R22),
    그 행에 개수가 있으면 자료 오류로 거부한다(검산 노트 03 C절). 사건이 하나도 없는 칸은 발생률이 0으로 내려갈수록 우도가 올라 최대우도 해가 없으므로 제프리스 사전분포의
    사후 평균인 상수 발생률을 쓰고 퇴화한 칸으로 표시한다(판정 R22) [도출]. 칸마다 따로 맞추므로 한 칸의 계수가 다른 칸에 영향을 주지 않는다(계획 P4, Task 6; 두 팀의 킬 수는 판정 R19).

    근거
    ----
    계획 P4(포아송 칸), P9(행마다 같은 무게), P10(노출), Task 6(y/e와 가중 e, 칸마다 모형), 판정 R19(팀마다 킬 수 모형), R20(허용오차와 수렴 판정), R22(노출 0 행의 기록과 사건 없는 칸),
    설계서 3.1절(계수 과정의 우도), 검산 노트 math-control-03 C절(칸 점수).
    Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
         doi:10.1017/CBO9781139013567 (노출이 있는 포아송, 로그 노출 오프셋). Jeffreys, H. (1946), Proc. R. Soc. Lond. A 186(1007): 453-461;
         doi:10.1098/rspa.1946.0056 (제프리스 사전분포)."""

    def __init__(self, kind: str = 'masked'):
        super().__init__(kind)
        self.squeeze_ = False
        self.n_zero_exposure_ = 0
        self.n_events_ = self.degenerate_ = None

    def fit(self, X, Y, e, n_numeric=None, blocks=(), *, overwrite_X: bool = False):
        """(B0-2)와 (B0-3)을 칸마다 맞추고 머리 자신을 돌려준다."""
        from sklearn.linear_model import PoissonRegressor
        X = _design_matrix(X)
        n, d = X.shape
        Y = np.asarray(Y)
        if Y.ndim not in (1, 2) or Y.shape[0] != n:
            raise ValueError(f'{self.name}: the counts must have {n} rows (one per design row), not shape {Y.shape}')
        if Y.dtype.kind not in 'fiu':
            raise TypeError(f'{self.name}: the counts must be numbers, not {Y.dtype}')
        self.squeeze_ = Y.ndim == 1
        Y = Y.reshape(n, -1).astype(np.float64)
        e = np.asarray(e, dtype=np.float64)
        if e.shape != (n,):
            raise ValueError(f'{self.name}: the exposure must have shape ({n},), not {e.shape}')
        if not np.isfinite(e).all() or (e < 0).any():
            raise ValueError(f'{self.name}: an exposure is negative or not finite')
        if not np.isfinite(Y).all() or (Y < 0).any() or (Y != np.floor(Y)).any():
            raise ValueError(f'{self.name}: a count is not a non-negative whole number')
        # 식 (B0-2)의 합은 e_i > 0인 행에 대해서이다.
        keep = e > 0
        if (Y[~keep] > 0).any():
            raise ValueError(f'{self.name}: {int((Y[~keep] > 0).any(axis=1).sum())} row(s) have a count but no exposure')
        _check_finite(X, 'the design')
        self._set_layout(d, n_numeric, blocks)
        self.n_zero_exposure_ = int((~keep).sum())
        if not keep.any():
            raise ValueError(f'{self.name}: no row has any exposure')
        Y, e = Y[keep], e[keep]
        Xs = self._training_design(X, keep, overwrite_X)
        cells = Y.shape[1]
        coef, intercept = np.zeros((cells, self._model_width(d))), np.zeros(cells)
        self.n_events_ = Y.sum(axis=0).astype(np.int64)
        self.degenerate_ = self.n_events_ == 0
        self.n_iter_, self.converged_ = np.zeros(cells, dtype=np.int64), np.ones(cells, dtype=np.bool_)
        Xm = None
        # 칸마다 따로 맞춘다(식 (B0-2), 계획 Task 6). 사건이 없는 칸은 식 (B0-3)의 상수 모형이고, 나머지는 목표 y/e와 표본 가중 e의 회귀이다.
        for c in range(cells):
            if self.degenerate_[c]:
                intercept[c] = np.log(0.5 / e.sum())               # no event: the Jeffreys posterior mean of the rate, a constant model
                continue
            if Xm is None:
                Xm = self._model_design(Xs)                        # the design of the model (M2Rate: the window interaction)
            model = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER_POISSON, tol=TOL_POISSON)
            fired = _fit_recording(model, Xm, Y[:, c] / e, sample_weight=e)
            coef[c], intercept[c] = model.coef_, model.intercept_
            self.n_iter_[c] = int(model.n_iter_)
            self.converged_[c] = _converged(fired, int(model.n_iter_), MAX_ITER_POISSON)
            if not self.converged_[c]:
                _warn_not_converged(self._label, c, int(model.n_iter_), MAX_ITER_POISSON, fired)
        self.n_fit_ = int(len(e))
        self._store(coef, intercept)
        return self

    def _output(self, Z: np.ndarray) -> np.ndarray:
        """(B0-2)의 λ_{ic} = exp(η_{ic})를 낸다. exp가 넘쳐 유한하지 않은 값은 (NLL-1)의 정의역 밖이므로 거부한다."""
        with np.errstate(over='ignore'):
            rate = np.exp(Z)
        if not np.isfinite(rate).all():
            raise ValueError(f'{self.name}: the rate of {int((~np.isfinite(rate)).any(axis=1).sum())} row(s) is not finite (the linear '
                             'predictor is out of the range of exp)')
        return rate[:, 0] if self.squeeze_ else rate

    def rate(self, X) -> np.ndarray:
        """The rate per unit of exposure of every cell: (n,) for a 1-cell head fitted on a 1-d target, else (n, J). Same as ``predict``."""
        return self.predict(X)

    def expected(self, X, exposure_s) -> np.ndarray:
        """The expected count of every cell: ``rate(X) * exposure_s``. ``exposure_s``: a number, one value per row (n,) or one per cell and
        row (n, J), in the unit of the exposure of the fit (seconds). ValueError for a negative or non-finite exposure.

        (B0-2)의 μ_{ic} = e·λ_{ic}를 계산한다."""
        rate = self.predict(X)
        ex = np.asarray(exposure_s, dtype=np.float64)
        if not np.isfinite(ex).all() or (ex < 0).any():
            raise ValueError(f'{self.name}: the exposure must be finite and not negative')
        if ex.ndim == 1 and rate.ndim == 2:
            ex = ex[:, None]
        return rate * ex

    def nll(self, X, Y, e, cells: bool = False) -> np.ndarray:
        """The Poisson NLL of the counts ``Y`` with the exposure ``e`` (``nll.poisson``, ``log y!`` included): one value per row, the
        cells summed, or (n, J) with ``cells``. A row with no exposure scores 0 (its rate is not read).

        (NLL-1)의 칸 점수 ℓ_{ic}를 내고, 행의 점수는 칸 점수의 합이다((HR-5)의 ℓ_i)."""
        rate = self.predict(X)
        Y = np.asarray(Y, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        per_cell = nll_scores.poisson(Y, rate, e[:, None] if rate.ndim == 2 and e.ndim == 1 else e)
        return per_cell if cells or per_cell.ndim == 1 else per_cell.sum(axis=1)

    def state(self) -> tuple:
        arrays, meta = super().state()
        arrays['n_events'] = np.asarray(self.n_events_, dtype=np.int64)
        arrays['degenerate'] = np.asarray(self.degenerate_, dtype=np.bool_)
        meta.update(squeeze=bool(self.squeeze_), n_zero_exposure=int(self.n_zero_exposure_))
        return arrays, meta

    def _restore(self, arrays: Mapping, meta: Mapping) -> None:
        super()._restore(arrays, meta)
        self.squeeze_, self.n_zero_exposure_ = bool(meta['squeeze']), int(meta['n_zero_exposure'])
        self.n_events_, self.degenerate_ = np.array(arrays['n_events']), np.array(arrays['degenerate'])

    def summary(self) -> dict:
        out = super().summary()
        out.update(n_zero_exposure=int(self.n_zero_exposure_), n_events=[int(v) for v in self.n_events_],
                   degenerate=[bool(v) for v in self.degenerate_])
        return out


class _CategoricalHead(_Head):
    """L2 규제와 균등 분포 스무딩 혼합(Uniform Mixing)을 적용한 다항 로지스틱 회귀 기저 헤드.

    [역할 및 특징]
    - 선제 킬 표지(M2Mark), 맵 구역(M2Zone), 참가 규모(M3Size), 킬 골드 구간(M3Gold) 등의 범주 확률 추정.
    - 미관측(-1) 범주 행은 학습에서 안전하게 제외하고 결측 건수(n_missing_)로 집계.
    - 훈련 세트에 미출현한 클래스에 대해서도 NLL이 무한대로 발산하지 않도록 미세한 균등 분포 확률을 혼합(식 B0-6):
      p = (1 - 1e-6) * p_model + (1e-6 / C)
    - 주요 출력:
      - predict(X) / predict_blocks(...): (n, C) 각 클래스별 조건부 사후 확률 (합계 1.0)
      - nll(X, y): (n,) 실제 관측 클래스 y에 대한 교차 엔트로피 손실 (식 NLL-2)

    수식
    ----
    (B0-5)  p^model_{ic} = exp(η_{ic}) / Σ_{c'∈A} exp(η_{ic'})   (c ∈ A),      p^model_{ic} = 0   (c ∉ A)
            (γ̂, β̂, ω̂) = argmin [ χ·Σ_{i: c_i ≥ 0} (−log p^model_{i,c_i}) + ½·Σ_{c∈A} (‖β_c‖² + ‖ω_c‖²) ]       (절편 γ는 벌점이 없다)
            A가 한 범주뿐이면 p^model = 1(상수 모형), 두 범주면 로짓 (0, η)의 이항 로지스틱이다(η는 번호가 큰 범주의 점수).
    (B0-6)  p_{ic} = (1 − ν)·p^model_{ic} + ν / C                                                             (c = 0, …, C−1)

      기호                  | 뜻                                              | 단위      | 범위                        | 코드 변수
      c_i                   | 행 i의 실제 범주 번호(−1은 관측되지 않은 범주)  | —         | −1, 0, …, C−1               | y[i]
      A                     | 훈련 행에 나온 범주의 집합                      | —         | 1개 이상                    | classes_
      C                     | 범주의 수                                       | 개        | 표지 52, 구역 10, 규모 4, 골드 31 | N_CLASSES
      η_{ic}                | 행 i의 범주 c의 선형 예측량((B0-7))             | 로그 척도 | 유한                        | decision
      χ                     | 벌점의 역수(scikit-learn의 C)                   | —         | > 0                         | C_LOGISTIC
      ν                     | 균등 분포를 섞는 비율                           | —         | 0 < ν < 1 (1e-6)            | MIX
      p^model_{ic}, p_{ic}  | 섞기 전과 섞은 뒤의 확률                        | —         | [0, 1]; p_{ic} > 0          | softmax의 결과, predict의 반환값

    해석
    ----
    식 (B0-5)는 범주 확률을 선형 예측량의 소프트맥스로 두고 로그 손실에 능선 벌점을 더해 맞추며, 벌점이 있어 모든 수준의 열을 두어도 계수가 하나로 정해진다(모듈 글, Hastie 외의
    18.3.2절) [도출]. 범주가 −1인 훈련 행(위치를 모르는 첫 킬의 구역이 이런 행이다)은 관측되지 않은 범주이므로 적합에서 빼고 그 수를 기록한다(계획 전역 제약의 구역 z, Task 6;
    판정 R22). 훈련에 나오지 않은 범주는 p^model = 0이지만 (B0-6)이 균등 분포를 섞어 p ≥ ν/C > 0이므로 점수 (NLL-2)가 유한하다(판정 R13).

    근거
    ----
    계획 P4(모든 모형에 같은 범주 집합), P9(행마다 같은 무게), Task 6(로지스틱 규정, 구역 −1 행 제외), 전역 제약의 구역 z(−1은 적합에서 제외), 판정 R13(균등 분포 섞기),
    R19(팀마다 골드 모형), R20(허용오차와 수렴 판정), R22(제외한 행을 센다), 설계서 3.1절(표지·구역의 범주 확률).
    Ref: Hastie, T., Tibshirani, R. and Friedman, J. (2009), "The Elements of Statistical Learning", 2nd ed., Springer;
         doi:10.1007/978-0-387-84858-7 (다항 로지스틱과 소프트맥스, 4.4절; L2 벌점을 더한 로지스틱 회귀, 18.3.2절 "Logistic Regression with Quadratic
         Regularization"). Chen, S. F. and Goodman, J. (1999), Computer Speech and Language 13(4); doi:10.1006/csla.1999.0128 (균등 분포와의 보간).
         Liu, D. C. and Nocedal, J. (1989), Mathematical Programming 45(1-3): 503-528; doi:10.1007/BF01589116 (lbfgs)."""
    N_CLASSES = 0
    _SCORE = staticmethod(nll_scores.categorical)

    def __init__(self, kind: str = 'masked'):
        super().__init__(kind)
        self.classes_ = None
        self.n_missing_ = 0

    def fit(self, X, y, n_numeric=None, blocks=(), *, overwrite_X: bool = False):
        """(B0-5)로 맞추고 머리 자신을 돌려준다. 범주가 −1인 행은 뺀다."""
        from sklearn.linear_model import LogisticRegression
        X = _design_matrix(X)
        n, d = X.shape
        y = _int_codes('y', y)
        if y.shape != (n,):
            raise ValueError(f'{self.name}: the classes must have shape ({n},), not {y.shape}')
        if y.min(initial=0) < -1 or y.max(initial=0) >= self.N_CLASSES:
            raise ValueError(f'{self.name}: the classes must lie in -1 .. {self.N_CLASSES - 1}')
        _check_finite(X, 'the design')
        self._set_layout(d, n_numeric, blocks)
        # 식 (B0-5)의 합은 범주가 있는 행(c_i ≥ 0)에 대해서이다.
        keep = y >= 0
        self.n_missing_ = int((~keep).sum())
        if not keep.any():
            raise ValueError(f'{self.name}: no row has a class')
        y = y[keep]
        Xs = self._training_design(X, keep, overwrite_X)
        # A = 훈련에 나온 범주. 모형은 A에 대해서만 맞추고, 나머지 범주는 (B0-6)의 균등 분포 몫만 갖는다.
        classes = np.unique(y)
        if len(classes) == 1:                                          # one class: the constant model (LogisticRegression needs two)
            W, b, n_iter, converged = np.zeros((1, d)), np.zeros(1), 0, True
        else:
            model = LogisticRegression(C=C_LOGISTIC, max_iter=MAX_ITER_LOGISTIC, tol=TOL_LOGISTIC)
            fired = _fit_recording(model, Xs, y)
            n_iter = int(np.max(model.n_iter_))
            converged = _converged(fired, n_iter, MAX_ITER_LOGISTIC)
            if not converged:
                _warn_not_converged(self._label, None, n_iter, MAX_ITER_LOGISTIC, fired)
            if len(classes) == 2:                                      # the binary model has one coefficient row: logits [0, z]
                W, b = np.vstack([np.zeros(d), model.coef_[0]]), np.array([0.0, model.intercept_[0]])
            else:
                W, b = model.coef_, model.intercept_
        self.classes_ = classes
        self.n_iter_, self.converged_ = np.array([n_iter]), np.array([converged])
        self.n_fit_ = int(len(y))
        self._store(W, b)
        return self

    def _output(self, Z: np.ndarray) -> np.ndarray:
        """The probabilities of the C classes: softmax over the classes that occurred, put on the full class set and mixed (ruling R13).

        (B0-5)의 소프트맥스를 A에서 계산하고 (B0-6)으로 섞는다."""
        # 식 (B0-5)의 소프트맥스: 행마다 최댓값을 빼서 exp가 넘치지 않게 한다. 값은 같다.
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        # 식 (B0-6): 훈련에 나온 범주 A에만 모형 확률을 놓고 나머지는 0으로 둔 뒤 균등 분포를 섞는다.
        full = np.zeros((len(Z), self.N_CLASSES))
        full[:, self.classes_] = P
        return (1.0 - MIX) * full + MIX / self.N_CLASSES

    def nll(self, X, y) -> np.ndarray:
        """-log p of the class of every row (``nll.categorical`` / ``nll.gold``); a row with the class -1 scores 0.

        (NLL-2)(골드 머리는 (NLL-3))의 칸 점수 ℓ_i를 낸다."""
        return type(self)._SCORE(y, self.predict(X))

    def state(self) -> tuple:
        arrays, meta = super().state()
        arrays['classes'] = np.asarray(self.classes_, dtype=np.int64)
        meta.update(n_missing=int(self.n_missing_))
        return arrays, meta

    def _restore(self, arrays: Mapping, meta: Mapping) -> None:
        super()._restore(arrays, meta)
        self.classes_, self.n_missing_ = np.array(arrays['classes']), int(meta['n_missing'])

    def summary(self) -> dict:
        out = super().summary()
        out.update(n_missing=int(self.n_missing_), classes=[int(c) for c in self.classes_])
        return out


# ==========================================================================================================================
# 7. 개별 구체 예측 헤드 구현 클래스군: M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv
# ==========================================================================================================================
class M2Rate(_PoissonHead):
    """첫 킬 발생률 포아송 헤드: 18개 10초 시간 구간(q=0..17, 0~180초)별 초당 첫 킬 발생률 λ_q 추정.

    [역할 및 핵심 알고리즘]
    - 경기 시작/경과 상태에 따라 교전 시작(첫 킬)이 발생할 위험도(Hazard rate)를 모델링.
    - 시간 지평에 따른 반응성 차이를 반영하기 위해 3개 게이트 윈도우 w(q)=⌊q/6⌋별로 독립된 상태 계수 벡터 β_w를 갖는 상호작용 설계 (식 B0-4):
      - 윈도우 0 (q=0..5  : 0~60초)
      - 윈도우 1 (q=6..11 : 60~120초)
      - 윈도우 2 (q=12..17: 120~180초)
    - 주요 메서드:
      - rate_bins(numeric): (n, 18) 합성기 호출용 - 18개 구간 전체의 발생률 행렬을 고속 일괄 계산.

    수식
    ----
    (B0-4)  log λ_q(x) = γ_q + β_{w(q)}·x̃,    w(q) = ⌊q / 6⌋ ∈ {0, 1, 2},    γ_q = γ_0 + ω^q[q]    (ω^q[0] = 0: 구간 0이 기준 구간)
            모형의 열은 [β_0 | β_1 | β_2 | q 블록 17열], 곧 3D + 17개이다. 표준화한 숫자 열을 창 지시자와 곱해 창마다 따로 둔다. 맞춤은 (B0-2)이다
            (칸은 하나; 행 i = (t₀ 행, 구간 q) 쌍, y_i = 그 구간의 발생 수, e_i = Δ_q).

      기호     | 뜻                                               | 단위      | 범위       | 코드 변수
      λ_q(x)   | 입력 x인 t₀ 행의 구간 q의 첫 킬 발생률           | 1/s       | > 0        | rate_bins의 q열
      x̃       | t₀ 행의 표준화한 b0 특징(D = 39개)               | 무차원    | 유한       | numeric에서 (B0-1)로
      γ_q      | 구간 q의 절편                                    | 로그 척도 | 유한       | intercept_std_[0] + _tables['q'][q, 0]
      ω^q[q]   | q 블록의 계수(구간 0은 0)                        | 로그 척도 | 유한       | _tables['q'][q, 0]
      β_w      | 창 w의 상태 계수 벡터(D개)                       | 로그 척도 | 유한       | coef_std_[0, w·D:(w+1)·D]
      w(q)     | 구간 q가 속한 창 번호                            | —         | 0, 1, 2    | _WINDOW_OF_Q[q]
      Δ_q      | t₀ 행의 구간 q의 노출(관측한 시간)               | s         | (0, 10]    | e

    해석
    ----
    식 (B0-4)는 구간마다 절편 γ_q를 따로 두고 상태 x̃에 대한 반응은 세 창(0–5, 6–11, 12–17구간)마다 기울기 벡터 β_w 하나씩 두는 로그 선형 발생률이며, 세 창의 끝은
    평가 지평 H = 60·120·180초와 맞는다(설계서 3.1절). 기울기 하나로 18구간을 모두 설명하던 처음 형은 연기 실행의 기울기가 FIT 표본 안과 DEV에서 같게 첫 1분에서 1보다 크고
    구간별 기울기도 첫 구간에서 가장 커서(모형이 상태에 덜 반응한다) 적합이 아니라 형이 틀렸다고 보아 바꾸었다(수치는 모듈 글의 창 문단; 저자 승인 G-S3b B2-a, 판정 R36).
    구간마다 절편이 있어 우도는 창별로 나뉘지만 한 번의 공동 적합으로 맞추며, 그 대가로 모형 설계의 상태 칸은 행마다 3분의 2가 0이다. 이 변경은 연기 실행의 DEV 결과를 본 뒤에
    정한 것이어서 계획의 사전 고정에서 벗어나며, G-S3c는 관문을 연기 DEV 경기를 넣은 것과 뺀 것 둘로 보고한다(S3 as-built 기록 B2-a 항).

    근거
    ----
    G-S3b B2-a(저자 2026-10-22), 판정 R36(창별 상태 계수), S3 as-built 기록(B2-a 항), 계획 P10(노출), 설계서 3.1절(구간별 발생률, 지평과 구간의 대응)과 8절(평가 지평)."""
    name = 'M2Rate'
    WINDOWS = M2_WINDOWS

    # -- the design ----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, Y, e, n_numeric=None, blocks=(), *, overwrite_X: bool = False):
        """(B0-4)를 (B0-2)로 맞춘다. 칸은 구간의 첫 킬 수 하나뿐이어서 목표의 열이 둘 이상이면 거부한다."""
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] != 1:
            raise ValueError(f'{self.name} has one cell (the first-kill count of a bin), not {Y.shape[1]}')
        return super().fit(X, Y, e, n_numeric, blocks, overwrite_X=overwrite_X)

    def _set_layout(self, d: int, n_numeric, blocks) -> None:
        super()._set_layout(d, n_numeric, blocks)
        q = Block('q', 0, N_Q, True)
        if self.blocks_ != (q,):
            raise ValueError(f'{self.name} needs the design [numeric columns, the q block] with the single block {tuple(q)} (the bin 0 is '
                             f'its reference level); got the blocks {[tuple(b) for b in self.blocks_]}')

    def _model_width(self, d: int) -> int:
        """(B0-4)의 열 수 3D + 17이다."""
        return self.n_numeric_ * len(self.WINDOWS) + self.blocks_[0].width

    def _bins(self, hot: np.ndarray) -> np.ndarray:
        """The bin of every row from its 17 q columns (the bin 0 has none): ValueError for a row that is not all 0 or one single 1."""
        if not ((hot == 0.0) | (hot == 1.0)).all() or (hot.sum(axis=1) > 1.0).any():
            raise ValueError(f'{self.name}: the q columns of a design row must be all 0 (the bin 0) or hold exactly one 1')
        return np.where(hot.any(axis=1), hot.argmax(axis=1) + 1, 0)

    def _model_design(self, Xs: np.ndarray) -> np.ndarray:
        """[the standardised numeric columns of the window of the row, in the columns of that window, 0 elsewhere | the q columns], made in
        chunks from the standardised caller design ``Xs`` (the interacted columns are those of ``Xs``, not standardised again).

        (B0-4)의 열 [x̃·1[w = 0] | x̃·1[w = 1] | x̃·1[w = 2] | q 열]을 만든다."""
        k, n_windows = self.n_numeric_, len(self.WINDOWS)
        out = np.zeros((len(Xs), self._model_width(Xs.shape[1])))
        offset = np.arange(k)[None, :]
        # 식 (B0-4)의 창 상호작용: 행의 구간이 속한 창의 열 묶음에만 x̃를 놓고 다른 창의 열 묶음은 0으로 둔다. x̃는 다시 표준화하지 않는다.
        for a in range(0, len(Xs), _CHUNK):
            part = Xs[a:a + _CHUNK]
            window = _WINDOW_OF_Q[self._bins(part[:, k:])]
            out[a + np.arange(len(part))[:, None], (window * k)[:, None] + offset] = part[:, :k]
            out[a:a + len(part), k * n_windows:] = part[:, k:]
        return out

    # -- the linear predictor ------------------------------------------------------------------------------------------------------------
    def _block_start(self) -> int:
        return self.n_numeric_ * len(self.WINDOWS)

    def _state_terms(self, numeric) -> np.ndarray:
        """(n, windows): the state part x~ . beta_w of the linear predictor in each window, from the raw numeric part ``numeric`` (n, D).

        (B0-4)의 x̃·β_w를 창마다 계산한다."""
        self._need_fit()
        A = np.asarray(numeric, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != self.n_numeric_:
            raise ValueError(f'{self.name}: the numeric part must have {self.n_numeric_} columns, not shape {A.shape}')
        beta = self.coef_std_[0, :self._block_start()].reshape(len(self.WINDOWS), self.n_numeric_)
        return ((A - self.mean_) / self.scale_) @ beta.T

    def _linear(self, numeric, q: np.ndarray) -> np.ndarray:
        """(n, 1): the linear predictor of the rows ``numeric`` in the bins ``q`` (0 .. 17).

        (B0-4)의 γ_q + x̃·β_{w(q)}를 낸다."""
        terms = self._state_terms(numeric)
        if len(q) != len(terms):
            raise ValueError(f'q has {len(q)} entries but the numeric part has {len(terms)} rows')
        return (terms[np.arange(len(q)), _WINDOW_OF_Q[q]] + self.intercept_std_[0] + self._tables['q'][q, 0])[:, None]

    def decision(self, X) -> np.ndarray:
        """(n, 1) the linear predictor of the design [numeric columns | q block] (the bin of a row is what its q columns say).

        (B0-7)의 M2Rate 판 η = γ_q + x̃·β_{w(q)}를 설계 행렬로 계산한다."""
        self._need_fit()
        X = np.asarray(X, dtype=np.float64)
        width = self.n_numeric_ + self.blocks_[0].width
        if X.ndim != 2 or X.shape[1] != width:
            raise ValueError(f'{self.name}: the design must have {width} columns (the numeric ones and the q block), not shape {X.shape}')
        return self._linear(X[:, :self.n_numeric_], self._bins(X[:, self.n_numeric_:]))

    def decision_blocks(self, numeric, **codes) -> np.ndarray:
        """(n, 1) the same from the raw numeric part and the bin of every row (the keyword ``q``).

        같은 η를 구간 번호로 계산한다."""
        self._need_fit()
        if set(codes) != {'q'}:
            raise ValueError(f"{self.name}: the codes must be given for the blocks ['q'], not {sorted(codes)}")
        return self._linear(numeric, _level(self.blocks_[0], codes['q']))

    def rate_bins(self, numeric) -> np.ndarray:
        """(n, 18) the rate of every bin q = 0 .. 17 for the rows with the raw b0 features ``numeric`` (n, 39), per second: the bin q uses
        the state vector of the window q // 6.

        (B0-4)의 λ_q(x) = exp(γ_q + x̃·β_{w(q)})를 18구간 모두에 낸다. 합성기가 (HEAD-1)의 ``rate``로 쓴다."""
        base = self._state_terms(numeric)[:, _WINDOW_OF_Q] + self.intercept_std_[0] + self._tables['q'][:, 0][None, :]
        with np.errstate(over='ignore'):
            rate = np.exp(base)
        if not np.isfinite(rate).all():
            raise ValueError(f'{self.name}: a rate is not finite (the linear predictor is out of the range of exp)')
        return rate

    def _raw(self) -> tuple:
        """The coefficients of the raw design [state * 1[w = 0], state * 1[w = 1], state * 1[w = 2], q block] and its intercept. A window
        has its own shift (mean / scale) . beta_w of the intercept, which the bin columns take over: the bin q (q >= 1) gets the term
        gamma_q - gamma_0 - shift_w(q) + shift_0, the intercept gamma_0 - shift_0, so every bin has the same rate as in the standardised
        form.

        (B0-8)의 M2Rate 판이다."""
        k, n_windows = self.n_numeric_, len(self.WINDOWS)
        W = self.coef_std_.copy()
        # 식 (B0-8)의 M2Rate 판: 창 w의 기울기 β_{w,h} / σ_h(slopes)와 이동량 ζ_w(shift), 구간 열의 보정 ζ_0 − ζ_{w(q)}, 절편 γ_0 − ζ_0.
        slopes = W[:, :k * n_windows].reshape(-1, n_windows, k) / self.scale_
        shift = (slopes * self.mean_).sum(axis=2)                                       # (J, windows): (mean / scale) . beta_w
        W[:, :k * n_windows] = slopes.reshape(-1, k * n_windows)
        W[:, k * n_windows:] += shift[:, [0]] - shift[:, _WINDOW_OF_Q[1:]]
        return W, self.intercept_std_ - shift[:, 0]

    # -- storage -------------------------------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        arrays, meta = super().state()
        meta['windows'] = [list(w) for w in self.WINDOWS]
        return arrays, meta

    def _restore(self, arrays: Mapping, meta: Mapping) -> None:
        if 'windows' not in meta:
            raise ValueError(f'{self.name} was saved before G-S3b B2-a (it has no windows: one coefficient vector for all the bins); '
                             'it is not loaded, fit it again')
        if [tuple(w) for w in meta['windows']] != list(self.WINDOWS):
            raise ValueError(f'{self.name}: the saved windows {meta["windows"]} are not the windows {[list(w) for w in self.WINDOWS]} of '
                             'this module')
        blocks = tuple(Block(*b) for b in meta['blocks'])
        need = int(meta['n_numeric']) * len(self.WINDOWS) + sum(b.width for b in blocks)
        have = np.asarray(arrays['coef_std']).shape[1]
        if blocks != (Block('q', 0, N_Q, True),) or have != need:
            raise ValueError(f'{self.name}: the saved coefficients have {have} columns, but the layout of the model (n_numeric '
                             f'{meta["n_numeric"]}, {len(self.WINDOWS)} windows, the blocks {[tuple(b) for b in blocks]}) needs {need}')
        super()._restore(arrays, meta)

    def summary(self) -> dict:
        out = super().summary()
        out['windows'] = [list(w) for w in self.WINDOWS]
        return out


class M2Mark(_CategoricalHead):
    """첫 킬의 표지 52개를 t₀ 행의 특징과 구간 q로 예측한다. (HEAD-2)의 표지 분포를 (B0-5)로 맞춘 것이다."""
    name, N_CLASSES = 'M2Mark', N_M


class M2Zone(_CategoricalHead):
    """첫 킬의 구역 10개를 특징과 q, m으로 예측한다. (HEAD-2)의 구역 분포를 (B0-5)로 맞춘 것이다."""
    name, N_CLASSES = 'M2Zone', N_ZONE


class M3Size(_CategoricalHead):
    """규모 4개를 예측한다. 가린 머리는 특징과 q, m, z로, 열린 머리는 교전의 사건 보기의 b0 열만으로 예측하며, (HEAD-2)의 규모 분포를 (B0-5)로 맞춘 것이다.
    입력에서 관측되지 않은 구역(z = −1)은 한 수준으로 둔다(계획 전역 제약의 구역 z)."""
    name, N_CLASSES = 'M3Size', N_SIZE


class M3Kills(_PoissonHead):
    """The kills of the two teams (ruling R19): a Poisson head with two cells, 0 the blue and 1 the red team, on one row per onset; each
    cell is a model of its own with its own coefficient for every column. ``predict`` gives the mean kills (n, 2), blue then red.

    (HEAD-3)의 μ_blue, μ_red를 (B0-2)로 낸다(노출 1). 두 칸의 계수를 따로 맞추므로 한 팀을 올리는 입력이 다른 팀을 낮출 수 있다(판정 R19).

    ``kill_law`` (S4b 준비 M1, 결정 D5): ``'poisson'``(기본값)은 위의 머리 그대로(옛 적합과 비트까지 같다). ``'shifted'``는 첫 킬 팀 d의 킬 수를 K_d = 1 + N_d로 두는 이동 법칙이다.

    수식
    ----
    (B0-2k)  y*_{i,t} = y_{i,t} − 1[t = d_i] ;   (γ̂_t, β̂_t, ω̂_t) = (B0-2)로 y*를 맞춘 해 ;   exp(η_{i,t}) = λ_{i,t}  (d_i = t)  또는  μ_{i,t}  (d_i ≠ t)
             점수:  ℓ_{i,t} = r_{i,t} − y*_{i,t} log r_{i,t} + log(y*_{i,t}!),   r_{i,t} = predict(X)[i, t]  (``killlaw.logpmf``; y*_{i,d_i} < 0이면 +∞)

      기호      | 뜻                                                | 단위 | 범위            | 코드 변수
      d_i       | 행 i의 첫 킬 팀 (0 파랑, 1 빨강, −1 처형)          | —    | −1, 0, 1        | first_team
      y*_{i,t}  | 칸 t의 유효 개수: 첫 킬 팀이면 초과분 y − 1, 아니면 y | 개   | 0, 1, …        | killlaw.excess
      r_{i,t}   | 팀 t 칸의 율 (d 칸은 초과분의 평균 λ_d, 다른 칸은 킬 수의 평균 μ) | 개 | > 0            | predict

    해석: 두 팀의 칸은 따로 맞추고 입력이 같은 (B0-7)의 선형 예측량이다. 이동 법칙에서는 한 칸이 d = t인 행과 d ≠ t인 행을 함께 보므로 칸이 두 율을 낼 수 있어야 한다: 가린 머리는 표지 m의
    원핫 블록이(d = m // 25), 열린 머리는 ``layout_for``가 더하는 d 블록이 그 구분을 맡는다. 적합은 y*를 목표로 한 (B0-2) 그대로이고(``first_team``을 함께 준다), 점수와 예측 함수는 d를 몰라도
    율을 낸다(합성기가 d를 안다)."""
    name = 'M3Kills'

    def __init__(self, kind: str = 'masked', kill_law: str = 'poisson'):
        super().__init__(kind)
        self.kill_law = killlaw.check_law(kill_law)

    def fit(self, X, Y, e, n_numeric=None, blocks=(), *, overwrite_X: bool = False, first_team=None):
        """(B0-2) with the target of the kill law: the counts ``Y`` for the plain law, their excess ``killlaw.excess(Y, first_team)`` for the shifted law, whose first teams
        ``first_team`` (n,) are required (ValueError otherwise, and for a first team with no kill)."""
        if self.kill_law == 'shifted':
            if first_team is None:
                raise ValueError(f'{self.name}: the shifted kill law needs the first-kill team of every row (first_team)')
            Y = killlaw.excess(Y, first_team, self.kill_law)
        return super().fit(X, Y, e, n_numeric, blocks, overwrite_X=overwrite_X)

    def nll(self, X, Y, e, cells: bool = False, first_team=None) -> np.ndarray:
        """The NLL of the kill counts ``Y`` (n, 2): ``_PoissonHead.nll`` for the plain law; for the shifted law (KL-3) with the first teams ``first_team`` (n,) of the rows
        (required; the exposure is 1). The result is one value per row, or (n, 2) with ``cells``."""
        if self.kill_law == 'poisson':
            return super().nll(X, Y, e, cells)
        if first_team is None:
            raise ValueError(f'{self.name}: the shifted kill law needs the first-kill team of every row (first_team)')
        if not np.array_equal(np.asarray(e, dtype=np.float64), np.ones(len(np.asarray(Y)))):
            raise ValueError(f'{self.name}: the kill counts have the exposure 1')
        per_cell = -killlaw.logpmf(Y, self.predict(X), first_team, self.kill_law)
        return per_cell if cells else per_cell.sum(axis=1)

    def state(self) -> tuple:
        arrays, meta = super().state()
        if self.kill_law != 'poisson':
            meta['kill_law'] = self.kill_law                       # absent means the plain law: a plain head is saved as every earlier head was
        return arrays, meta

    def _restore(self, arrays: Mapping, meta: Mapping) -> None:
        super()._restore(arrays, meta)
        self.kill_law = killlaw.check_law(meta.get('kill_law', 'poisson'))

    def summary(self) -> dict:
        out = super().summary()
        if self.kill_law != 'poisson':
            out['kill_law'] = self.kill_law
        return out


class _GoldModel(_CategoricalHead):
    """The gold model of one team: a member of ``M3Gold``.

    (HEAD-3)의 골드 구간 분포를 (B0-5)로 맞춘 한 팀의 모형이다(범주 31개; 점수는 (NLL-3))."""
    name, N_CLASSES = 'M3Gold', N_GOLD
    _SCORE = staticmethod(nll_scores.gold)


class M3Gold:
    """팀별 교전 획득 킬 골드 다항 로지스틱 헤드 번들: 31개 골드 구간, 블루/레드 팀별 독립 모델.

    [역할 및 설계 이유]
    - 교전 결과 각 팀이 획득하는 킬 골드 구간(0~30, 31개 클래스)의 조건부 사후 확률 분포를 예측.
    - 블루 팀(member 0)과 레드 팀(member 1) 각각 독립된 다항 로지스틱 모델(_GoldModel)을 보유:
      진영 비대칭성 및 제로섬 효과를 반영하기 위해 두 팀의 계수를 완전히 분리하여 학습.
    - 킬 수 층 ξ = min(max(⌊k⌉, 1), 6) (1~6층)을 추가 조건부 특징으로 입력받음.
    - 킬이 1 이상인 유효 케이스만 학습에 참여 (킬 0인 팀의 골드는 항상 0으로 확정).
    ``M3Kills``와 같다(판정 R19)."""
    name = 'M3Gold'
    N_CLASSES = N_GOLD

    def __init__(self, kind: str = 'masked'):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        self.kind = kind
        self.members = (_GoldModel(kind), _GoldModel(kind))
        for team, member in zip(TEAMS, self.members):
            member._label = f'{self.name} ({team} team)'

    # -- the members -------------------------------------------------------------------------------------------------------------------
    @property
    def parts(self) -> tuple:
        return self.members

    def member(self, team) -> _GoldModel:
        """The model of team ``team``: 0 blue, 1 red."""
        if isinstance(team, bool) or not isinstance(team, (int, np.integer)) or team not in (0, 1):
            raise ValueError(f'team must be 0 (blue) or 1 (red), not {team!r}')
        return self.members[int(team)]

    @staticmethod
    def _teams(team, n: int) -> np.ndarray:
        t = _int_codes('team', team)
        if len(t) != n:
            raise ValueError(f'team has {len(t)} entries but there are {n} rows')
        if t.size and (t.min() < 0 or t.max() > 1):
            raise ValueError('team must lie in 0 (blue) .. 1 (red)')
        return t

    def _rows(self, X) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f'{self.name}: the design must be a 2-d array, not shape {A.shape}')
        return A

    # -- fit and predict -----------------------------------------------------------------------------------------------------------------
    def fit(self, X, y, team, n_numeric=None, blocks=(), *, overwrite_X: bool = False):
        """Fit the member of each team on the rows of that team. ``y`` is the gold bin (0 .. 30, -1 = none: left out and counted), ``team``
        the team of every row (0 blue, 1 red). ``overwrite_X`` is accepted for the call of ``B0Heads.fit`` and has no effect: each member
        is fitted on a copy of the rows of its team, and the design given is left as it is.

        팀의 항목만으로 팀마다 (B0-5)를 맞춘다. 한 팀의 항목이 하나도 없으면 그 팀의 모형을 만들 수 없어 거부한다(판정 R19)."""
        X = _design_matrix(X)
        team = self._teams(team, len(X))
        y = _int_codes('y', y)
        if y.shape != (len(X),):
            raise ValueError(f'{self.name}: the classes must have shape ({len(X)},), not {y.shape}')
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size == 0:
                raise ValueError(f'{self.name}: no row of the {TEAMS[t]} team')
            member.fit(X[rows], y[rows], n_numeric=n_numeric, blocks=blocks, overwrite_X=True)
        return self

    def predict(self, X, team) -> np.ndarray:
        """(n, 31) the gold law of every row, from the model of its team."""
        X = self._rows(X)
        team = self._teams(team, len(X))
        out = np.empty((len(X), N_GOLD))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.predict(X[rows])
        return out

    def predict_blocks(self, numeric, team, **codes) -> np.ndarray:
        """(n, 31) the same from the numeric part and the integer codes of the one-hot blocks (the closed form of the composer)."""
        numeric = self._rows(numeric)
        n = len(numeric)
        team = self._teams(team, n)
        arrays = {}
        for name, values in codes.items():
            a = _int_codes(name, values)
            if len(a) != n:
                raise ValueError(f'{name} has {len(a)} entries but the numeric part has {n} rows')
            arrays[name] = a
        out = np.empty((n, N_GOLD))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.predict_blocks(numeric[rows], **{k: v[rows] for k, v in arrays.items()})
        return out

    def nll(self, X, y, team) -> np.ndarray:
        """-log p of the gold bin of every row under the model of its team; a row with the bin -1 (no kill) scores 0.

        (NLL-3)의 칸 점수 ℓ_i를 팀의 모형으로 낸다."""
        X = self._rows(X)
        team = self._teams(team, len(X))
        y = _int_codes('y', y)
        if y.shape != (len(X),):
            raise ValueError(f'{self.name}: the classes must have shape ({len(X)},), not {y.shape}')
        out = np.zeros(len(X))
        for t, member in enumerate(self.members):
            rows = np.flatnonzero(team == t)
            if rows.size:
                out[rows] = member.nll(X[rows], y[rows])
        return out

    # -- what the members have in common ---------------------------------------------------------------------------------------------------
    def _need_fit(self) -> None:
        for member in self.members:
            member._need_fit()

    @property
    def n_numeric_(self) -> int:
        return self.members[0].n_numeric_

    @property
    def blocks_(self) -> tuple:
        return self.members[0].blocks_

    @property
    def n_fit_(self) -> int:
        return sum(m.n_fit_ for m in self.members)

    @property
    def n_missing_(self) -> int:
        return sum(m.n_missing_ for m in self.members)

    @property
    def converged_(self) -> np.ndarray:
        """(2,) whether the model of the blue and of the red team converged."""
        self._need_fit()
        return np.concatenate([m.converged_ for m in self.members])

    @property
    def n_iter_(self) -> np.ndarray:
        """(2,) the iterations of the model of the blue and of the red team."""
        self._need_fit()
        return np.concatenate([m.n_iter_ for m in self.members])

    # -- storage ---------------------------------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        """(arrays, meta): the arrays of both members (prefixed ``blue.`` and ``red.``) and the meta of both under ``members``."""
        self._need_fit()
        arrays, members = {}, {}
        for team, member in zip(TEAMS, self.members):
            member_arrays, members[team] = member.state()
            arrays.update({f'{team}.{field}': a for field, a in member_arrays.items()})
        return arrays, {'class': type(self).__name__, 'kind': self.kind, 'n_fit': int(self.n_fit_), 'members': members}

    @classmethod
    def from_state(cls, arrays: Mapping, meta: Mapping) -> 'M3Gold':
        head = cls(meta['kind'])
        for i, team in enumerate(TEAMS):
            prefix = f'{team}.'
            head.members[i]._restore({k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}, meta['members'][team])
        return head

    def summary(self) -> dict:
        """The counts of the fit as plain JSON values: ``converged`` and ``n_iter`` have one entry for each team (blue, red), ``teams``
        the summary of each member."""
        self._need_fit()
        teams = {team: member.summary() for team, member in zip(TEAMS, self.members)}
        return {'kind': self.kind, 'n_fit': int(self.n_fit_), 'width': teams['blue']['width'], 'n_numeric': int(self.n_numeric_),
                'converged': [bool(c) for c in self.converged_], 'n_iter': [int(i) for i in self.n_iter_],
                'n_missing': int(self.n_missing_), 'teams': teams}


class M3Obj(_PoissonHead):
    """오브젝트 18칸의 개수를 칸마다 독립인 (B0-2)로 낸다. (HEAD-4)의 포아송 평균 μ_c이고 노출은 1이다. 가린 머리는 K를 숫자 열에 두지 않는다(계획 P6)."""
    name = 'M3Obj'


class M4Conv(_PoissonHead):
    """오브젝트 18칸의 전환 수를 칸마다 독립인 (B0-2)로 낸다. (HEAD-4)의 전환율 λ_c이고 노출은 전환 창의 길이(초)이다. 가린 머리는 K와 O를 두지 않는다(계획 P6, P10)."""
    name = 'M4Conv'


HEAD_CLASSES = {cls.name: cls for cls in (M2Rate, M2Mark, M2Zone, M3Size, M3Kills, M3Gold, M3Obj, M4Conv)}
if tuple(HEAD_CLASSES) != HEAD_NAMES or not set(OPEN_HEAD_NAMES) <= set(HEAD_NAMES):
    raise RuntimeError('the head classes and HEAD_NAMES differ')


# ---- the training and scoring rows -------------------------------------------------------------------------------------------------------
# ``HeadData``, ``HeadScores`` and ``HeadBatch`` are those of ``head_rows`` (imported above). The rows themselves are chosen there, from the
# S1 tables and the S2 views, once for B0 and for B1 (ruling R18). B0's own are the feature builder (b0, a NaN filled with the FIT mean)
# and the design (the standardised numeric columns and the one-hot blocks of ``LAYOUTS``).
def _checked_means(fit_means) -> np.ndarray:
    if fit_means is None:
        raise ValueError('fit_means is required: the b0 FIT means (features.b0_fit_means) fill the NaN cells of b0')
    means = np.asarray(fit_means, dtype=np.float64)
    if means.shape != (N_B0,) or not np.isfinite(means).all():
        raise ValueError(f'fit_means must be {N_B0} finite numbers (features.b0_fit_means), not shape {means.shape} with non-finite values')
    return means


class HeadRows(head_rows.HeadRows):
    """The rows that the B0 heads are trained or scored on, for the matches of one role of an (S1, S2, labels) triple: the rows of
    ``head_rows.HeadRows`` with the b0 features of the FIT means ``fit_means`` (``features.b0_fit_means``).

    ``data(head, kind)`` gives the ``HeadData`` of one head, the design being that of ``LAYOUTS``. ValueError for an S2 store or label store
    that does not belong to the S1 store (counts only, never a key), for a role that is a holdout role or not used by S3, for fit means
    that are not 39 finite numbers, and for a unit that lacks a view.

    ``k_condition`` (S4b 준비 M3b, 결정 D24): 열린 오브젝트 머리(M3Obj)와 열린 전환 머리(M4Conv)의 설계에 들어가는 K 블록의 골드. ``'raw'``(기본값)는 라벨의 실제 킬 골드(이전 실행의 행)이고,
    ``'binned'``는 B1·EPT와 같은 값, 곧 킬이 있는 팀은 관측한 골드 구간의 대표값 rep[b]이고 킬이 없는 팀은 0이다((HR-9); 31개의 rep는 ``head_rows.HeadRows.k_reps``의 범위 안 FIT 한 출처). 가린 머리는 K를 설계에 두지 않으므로(계획 P6)
    어느 조건에서도 같은 행이다.

    ``matches`` (S4b 준비 M11, 결정 D16): 역할의 경기 가운데 고른 경기 번호(``head_rows.HeadRows``의 ``matches=``, B1과 같다). 기본값 None은 역할의 모든 경기이고 이전 행과 같다.
    B0의 상대 NLL R_h = Σ w·NLL_모형 / Σ w·NLL_B0 는 두 모형이 같은 경기의 같은 행에서 채점돼야 하므로, B0도 다른 모형과 같은 ``matches=``를 받을 수 있어야 한다.

    행의 정의는 (HR-1)–(HR-4)이고, 이 클래스는 거기에 b0 특징(결측 칸은 FIT 평균; 판정 R2)과 모듈 글의 설계 표대로 만든 설계를 얹는다(판정 R18)."""

    def __init__(self, s1, s2, labels, fit_means, role, *, matches=None, kill_law: str = 'poisson', k_condition: str = 'raw'):
        self.means = _checked_means(fit_means)
        super().__init__(s1, s2, labels, role, lambda store, views: features.b0(store, views, self.means), matches=matches, kill_law=kill_law, k_condition=k_condition)

    def data(self, head: str, kind: str) -> HeadData:
        """The rows of the head ``head`` ('M2Rate', ... 'M4Conv') of ``kind`` ('masked' or 'open'), see ``HeadData``. ValueError for a head
        and kind that do not exist (M2 has no open heads).

        (B0-7)의 입력 행렬 X = [숫자 열 | 원핫 열]과 (HR-1)–(HR-5)의 y, e, 경기 번호, 평가 행 번호를 묶어 ``HeadData``로 낸다."""
        if (head, kind) not in LAYOUTS:
            raise ValueError(f'there is no {kind} head {head!r}; the heads are {[k for k in LAYOUTS]}')
        layout = layout_for(head, kind, self.kill_law)
        p = self.parts(head, kind)
        if kind == 'masked':
            # the masked M4 design keeps [b0, q, m, z] of the input of features.m4_masked_inputs (neither K nor O: decision P6)
            numeric = p.feat[:, :N_B0] if head == 'M4Conv' else p.gather()
        else:
            numeric = p.gather().astype(np.float64)
            for block in (p.K, p.O):                                      # the open heads add the true K, and the open M4 head the true O
                if block is not None:
                    numeric = np.hstack([numeric, block.astype(np.float64)])
        X = _design(numeric, layout.blocks, {b.name: p.codes[b.name] for b in layout.blocks}) if layout.blocks else numeric
        return HeadData(X, p.y, p.e, p.match, p.row, layout, p.n_zero_exposure, p.team, p.first_team)


# ==========================================================================================================================
# 8. 합성기(Composer) 전용 닫힌 꼴(Closed-Form) 초고속 추론 체인
# ==========================================================================================================================
class _MaskedChain:
    """합성기가 호출하는 가린 헤드(Masked Heads)들의 닫힌 꼴(closed-form) 연결 체인 (heads.ComposerHeads 프로토콜 구현).

    [역할 및 최적화 메커니즘]
    - EPT 평가 파이프라인에서 수백만 건의 발생(Onset) 행에 대해 교전 가치를 합성할 때 사용.
    - 무거운 2차원 설계 행렬을 매번 생성하지 않고, 사전 구축된 One-Hot 계수 테이블(_tables)에서 단순 슬라이싱/인덱싱 합으로 선형 예측자를 계산:
      η = γ + x̃·β + ω^q[q] + ω^m[m] + ω^z[z] + ω^s[s]
    - 이를 통해 초당 수십만 행 이상의 압도적인 평가 속도를 달성.
    - 정보 누출 방지: M3Obj는 K(킬)를 읽지 않고, M4Conv는 K와 O(오브젝트)를 읽지 않음.
    """
    feature_kind = 'b0'
    analytic_ok = True
    obj_uses_K = False
    conv_uses_KO = False

    def __init__(self, masked: Mapping, fit_means: np.ndarray):
        self._h = masked
        self._means = fit_means
        self._b0 = None
        # 킬 수 법칙(결정 D5, 검토 I1): ``kill_rate``의 d 칸이 초과분의 평균 λ_d('shifted')인지 킬 수의 평균 μ('poisson')인지는 M3Kills 머리가 정한다. 합성기가 이 선언을 자기 법칙과
        # 대조하므로(``Composer``) 다른 법칙으로 맞춘 머리를 말없이 합성하지 못한다. 킬 머리가 없으면(부분 묶음) 옛 법칙이다.
        self.kill_law = getattr(masked.get('M3Kills'), 'kill_law', 'poisson')

    def bind(self, s2, views) -> None:
        """Compute and keep the b0 rows of the frame-clock views ``views`` (view numbers in any order, or a bool mask). ValueError for a
        view that is not a frame-clock view (``views/src`` 0): an event view holds the events up to the first kill or the end of the
        engagement, which a masked head must never see (A3).

        결측 칸은 FIT 평균으로 채운다(판정 R2)."""
        b0 = features.b0(s2, views, self._means)
        v = np.asarray(views)
        chosen = np.flatnonzero(v) if v.dtype == np.bool_ else v.astype(np.int64)
        try:
            src = np.asarray(s2.array('views/src'))[chosen]
        except FileNotFoundError:
            raise ValueError('the S2 store has no array views/src') from None
        if (src != 0).any():
            raise ValueError(f'bind takes frame-clock views (views/src 0) only, but {int((src != 0).sum())} of the {len(chosen)} views are '
                             'event views')
        self._b0 = b0

    def _set_rows(self, row_view) -> None:
        """Use the b0 features ``row_view`` (n, 39) as the bound rows (no S2 store): the rows of a ``HeadBatch``. They are float32, as the
        rows that ``bind`` computes are."""
        A = np.asarray(row_view)
        if A.dtype.kind not in 'fiu' or A.ndim != 2 or A.shape[1] != N_B0:
            raise ValueError(f'row_view must be a numeric array of shape (n, {N_B0}), not {A.dtype} {A.shape}')
        self._b0 = narrow('row_view', A.astype(np.float64), np.float32)

    def _rows(self, rows) -> np.ndarray:
        if self._b0 is None:
            raise RuntimeError('call bind(s2, views) before asking the heads for rows')
        r = _int_codes('rows', rows)
        if r.size and (r.min() < 0 or r.max() >= len(self._b0)):
            raise ValueError(f'rows must lie in 0 .. {len(self._b0) - 1} (the bound views)')
        return r

    def row_view(self, rows) -> np.ndarray:
        return self._b0[self._rows(rows)].copy()

    def _numeric(self, rows) -> np.ndarray:
        return self._b0[self._rows(rows)].astype(np.float64)

    @staticmethod
    def _same_length(n: int, **named) -> None:
        for name, a in named.items():
            if len(a) != n:
                raise ValueError(f'{name} has {len(a)} entries but rows has {n}')

    def rate(self, rows) -> np.ndarray:
        return self._h['M2Rate'].rate_bins(self._numeric(rows))

    def mark_prob(self, rows, q) -> np.ndarray:
        num, q = self._numeric(rows), _int_codes('q', q)
        self._same_length(len(num), q=q)
        return self._h['M2Mark'].predict_blocks(num, q=q)

    def zone_prob(self, rows, q, m) -> np.ndarray:
        num, q, m = self._numeric(rows), _int_codes('q', q), _int_codes('m', m)
        self._same_length(len(num), q=q, m=m)
        return self._h['M2Zone'].predict_blocks(num, q=q, m=m)

    def size_prob(self, rows, q, m, z) -> np.ndarray:
        num, q, m, z = self._numeric(rows), _int_codes('q', q), _int_codes('m', m), _int_codes('z', z)
        self._same_length(len(num), q=q, m=m, z=z)
        return self._h['M3Size'].predict_blocks(num, q=q, m=m, z=z)

    def _categories(self, rows, q, m, z, s) -> tuple:
        num, q, m, z, s = self._numeric(rows), _int_codes('q', q), _int_codes('m', m), _int_codes('z', z), _int_codes('s', s)
        self._same_length(len(num), q=q, m=m, z=z, s=s)
        return num, q, m, z, s

    def kill_rate(self, rows, q, m, z, s) -> np.ndarray:
        num, q, m, z, s = self._categories(rows, q, m, z, s)
        return self._h['M3Kills'].predict_blocks(num, q=q, m=m, z=z, s=s)                  # (n, 2): the blue and the red cell

    def gold_prob(self, rows, q, m, z, s, layer) -> np.ndarray:
        num, q, m, z, s = self._categories(rows, q, m, z, s)
        layer = np.asarray(layer)
        if layer.shape != (len(num), 2):
            raise ValueError(f'layer must have shape ({len(num)}, 2), not {layer.shape}')
        out = np.empty((len(num), 2, N_GOLD))
        for team in (0, 1):                                                   # the model of each team, on the kill layer of that team
            out[:, team] = self._h['M3Gold'].member(team).predict_blocks(num, q=q, m=m, z=z, s=s, layer=layer[:, team])
        return out

    def obj_rate(self, rows, q, m, z, s, K) -> np.ndarray:
        num, q, m, z, s = self._categories(rows, q, m, z, s)
        if np.shape(K) != (len(num), N_K):
            raise ValueError(f'K must have shape ({len(num)}, {N_K}), not {np.shape(K)}')
        return self._h['M3Obj'].predict_blocks(num, q=q, m=m, z=z, s=s)                    # K is not read: decision P6

    def conv_rate(self, m4_inputs) -> np.ndarray:
        width = N_B0 + len(features.M4_EXTRA_NAMES) + N_CELLS
        A = np.asarray(m4_inputs)
        if A.ndim != 2 or A.shape[1] != width:
            raise ValueError(f'm4_inputs must have shape (n, {width}) (features.m4_masked_inputs of a b0 row), not {A.shape}')
        codes = A[:, N_B0:N_B0 + 3].astype(np.float64)
        if not np.array_equal(codes, np.rint(codes)):
            raise ValueError('the q, m and z columns of m4_inputs must be whole numbers')
        codes = codes.astype(np.int64)
        # the K and O columns are not read (decision P6)
        return self._h['M4Conv'].predict_blocks(A[:, :N_B0].astype(np.float64), q=codes[:, 0], m=codes[:, 1], z=codes[:, 2])


# ---- the fitted set ----------------------------------------------------------------------------------------------------------------------
def _fit_role(roles):
    """The role(s) to fit on from ``roles`` = {'fit': name or names}; the default is FIT."""
    if roles is None:
        roles = {'fit': 'FIT'}
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be a mapping {'fit': role name(s)}")
    if set(roles) != {'fit'}:
        raise ValueError("roles must have exactly the key 'fit'")
    return roles['fit']


def _write_npz_bytes(arrays: Mapping) -> bytes:
    """A stored (uncompressed) .npz that ``np.load`` reads, without time stamps: the same arrays give the same bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            part = io.BytesIO()
            np.lib.format.write_array(part, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            zf.writestr(info, part.getvalue())
    return buffer.getvalue()


# ==========================================================================================================================
# 9. B0 전체 헤드 번들 관리자 (B0Heads)
# ==========================================================================================================================
class B0Heads:
    """B0 베이스라인의 전체 예측 헤드(가린 헤드 8종 + 열린 헤드 5종 = 총 13종) 번들 통합 관리자.

    [주요 역할 및 라이프사이클 메서드]
    - fit(data): FIT 역할 데이터셋에 대해 전체 헤드를 결정론적으로 일괄 학습 (식 B0-1 ~ B0-5).
    - nll(data): DEV/TEST 데이터셋에 대해 헤드별 손실(NLL) 및 가중 평균 평가 (HeadScores 반환).
    - composer_heads("masked"): 합성기용 고속 닫힌 꼴 체인(_MaskedChain) 반환.
    - predict_all(batch): 배치에 대한 전체 184차원 예측치 행렬 일괄 생성.
    - save(dir) / load(dir): b0_heads.npz 및 b0_heads.json을 통한 원자적 저장 및 무결성 검증 복원.
    """

    def __init__(self, masked: Mapping, open_: Mapping, fit_means, fit_roles=('FIT',), kill_law: str = 'poisson', k_condition: str = 'raw', k_reps_sha256=None):
        unknown = (set(masked) - set(HEAD_NAMES)) | (set(open_) - set(OPEN_HEAD_NAMES))
        if unknown:
            raise ValueError(f'unknown head(s): {", ".join(sorted(unknown))}')
        self.kill_law = killlaw.check_law(kill_law)
        # the gold of the K block that the open object and conversion heads were fitted on (decision D24, as the B1 set of decision D21 B16); the set says it, the heads carry no declaration of it
        self.k_condition = head_rows.check_k_condition(k_condition)
        # the sha256 of the 31 representatives of the binned K block (``value.gold_reps_sha256``): None for the raw gold and for a binned set whose representatives are not recorded.
        # A lowercase 64-digit hex string or None.
        if k_reps_sha256 is not None and (self.k_condition != 'binned' or not isinstance(k_reps_sha256, str) or len(k_reps_sha256) != 64
                                          or any(c not in '0123456789abcdef' for c in k_reps_sha256)):
            raise ValueError("k_reps_sha256 must be None or a lowercase 64-digit hex sha256, and only for k_condition 'binned'")
        self.k_reps_sha256 = k_reps_sha256
        for group in (masked, open_):
            if 'M3Kills' in group and group['M3Kills'].kill_law != self.kill_law:
                raise ValueError(f'the kill head is fitted for kill_law {group["M3Kills"].kill_law!r} but the set says {self.kill_law!r}')
        # always in the order of HEAD_NAMES and OPEN_HEAD_NAMES: summary(), nll() and the files do not depend on how the set was made
        self.masked = {name: masked[name] for name in HEAD_NAMES if name in masked}
        self.open = {name: open_[name] for name in OPEN_HEAD_NAMES if name in open_}
        self.fit_means = _checked_means(fit_means)
        self.fit_roles = tuple(fit_roles)

    # -- fit -------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, s1, s2, labels, fit_means, roles=None, *, check=None, kill_law: str = 'poisson', k_condition: str = 'raw') -> 'B0Heads':
        """Fit the eight masked and the five open heads on the matches of the role ``roles['fit']`` (default FIT), one head at a time (each
        design is built, fitted and dropped, so that the memory is that of the largest head). ``fit_means``: the b0 FIT means
        (``features.b0_fit_means``). ``check``: a function without arguments (the E6 guard), called before anything is read and before
        each head. ValueError for a holdout role or a role that S3 does not use, before any array is read.

        적합 역할 ``roles['fit']``(기본 FIT)의 경기로 맞춘다. 홀드아웃 역할과 S3가 쓰지 않는 역할은 배열을 읽기 전에 거부한다(계획 전역 제약의 분할).

        ``kill_law`` (S4b 준비 M1, 결정 D5): 킬 머리 M3Kills의 킬 수 법칙. 기본값 ``'poisson'``은 이전 적합과 같은 행·설계·계수(저장 파일까지 비트 같음)이고, ``'shifted'``는 첫 킬 팀의
        킬 수를 1 + N_d로 둔 적합이다(행이 첫 킬 팀을 담고, 두 M3Kills 머리가 초과분을 목표로 맞추며, 열린 머리의 설계에 d 블록이 더해진다: ``layout_for``). 다른 머리는 그대로이다.
        새 법칙으로 맞춘 B0는 옛 `B0_SCALES.json`의 척도(옛 법칙으로 잰 것)를 쓸 수 없고 FIT 전체에서 새로 재야 한다(계획 Q9, S4b 실행의 첫 단계).

        ``k_condition`` (S4b 준비 M3b, 결정 D24): K 블록을 설계에 읽는 두 열린 머리(오브젝트 머리 M3Obj와 전환 머리 M4Conv)가 학습에서 보는 골드. 기본값 ``'raw'``는 이전 적합과 같다(행, 설계, 계수, 저장 파일).
        ``'binned'``는 라벨의 실제 연속 골드 대신 관측한 골드 구간의 대표값(``head_rows``의 (HR-9); 킬이 없는 팀은 0)을 넣는 적합으로, B1·EPT가 보는 K와 같고 합성기가 넣는 값과 같은 지지에 있다.
        가린 머리 여덟과 열린 머리의 나머지 셋은 그대로이다. 이름은 ``head_rows.K_CONDITIONS``에 있어야 한다(배열을 읽기 전에 거부). 'binned'이면 묶음이 대표값 31개의 ``k_reps_sha256``(float64
        little-endian 바이트의 sha256, ``value.gold_reps_sha256``)를 기록하고 ``save``가 JSON에 적는다. B0의 적합 행은 FIT 한 묶음이라 B1의 'FIT와 STOP 행이 같은 조건' 점검은 필요 없고, 맞춘 묶음을 채점할 때의
        행이 같은 조건·같은 대표값인지는 ``nll``이 점검한다. 같은 조건으로 `B0_SCALES.json`을 다시 재야 한다(계획 Q9; ``scales.b0_scales``가 조건을 파일에 적는다)."""
        role = _fit_role(roles)
        killlaw.check_law(kill_law)
        head_rows.check_k_condition(k_condition)
        tick = check if check is not None else (lambda: None)
        tick()
        law = {} if kill_law == 'poisson' else {'kill_law': kill_law}
        condition = {} if k_condition == 'raw' else {'k_condition': k_condition}       # the raw gold is not passed: the calls are those of every earlier run
        rows = HeadRows(s1, s2, labels, fit_means, role, **law, **condition)
        fitted = {kind: {} for kind in KINDS}
        for kind, names in (('masked', HEAD_NAMES), ('open', OPEN_HEAD_NAMES)):
            for name in names:
                tick()
                d = rows.data(name, kind)
                head = M3Kills(kind, kill_law) if name == 'M3Kills' and kill_law != 'poisson' else HEAD_CLASSES[name](kind)
                args = (d.X, d.y) if d.e is None else (d.X, d.y, d.e)
                extra = {} if d.team is None else {'team': d.team}                      # the gold head: the team of every row (ruling R19)
                if name == 'M3Kills' and kill_law != 'poisson':
                    extra = {'first_team': d.first_team}                                # the kill head of the shifted law fits the excess of the first team
                head.fit(*args, **extra, n_numeric=d.layout.n_numeric, blocks=d.layout.blocks, overwrite_X=True)
                # 행을 고를 때 노출이 0이라 빠진 칸의 수를 머리의 기록에 합친다(판정 R22).
                if d.e is not None:
                    head.n_zero_exposure_ += d.n_zero_exposure
                fitted[kind][name] = head
                del d
        recorded = {} if k_condition == 'raw' else {'k_reps_sha256': rows.k_reps_sha256()}       # the 31 numbers of the binned K block
        return cls(fitted['masked'], fitted['open'], fit_means, (role,) if isinstance(role, str) else tuple(role), **law, **condition, **recorded)

    # -- scoring ---------------------------------------------------------------------------------------------------------------------------
    def head_rows(self, s1, s2, labels, role, *, matches=None) -> 'HeadRows':
        """The ``HeadRows`` of ``role`` (and of the chosen ``matches``, default all) with the means, the kill law and the K condition of this set, checked to be the rows the set was fitted for
        (``_check_rows``: the condition in both directions and, for a set that records them, the 31 representatives). This is the one way the set builds the rows it scores, for ``nll`` and for
        ``adapters.run_inference`` (S4b 준비 M11, 결정 D16), so that B0 is scored on the same rows as the model it is the denominator of.

        세트가 맞춘 조건·평균·대표값으로 행을 만들고 확인한다. ``nll``과 평가 어댑터가 같은 경로로 B0의 채점 행을 만든다."""
        rows = HeadRows(s1, s2, labels, self.fit_means, role, matches=matches, **({} if self.kill_law == 'poisson' else {'kill_law': self.kill_law}),
                        **({} if self.k_condition == 'raw' else {'k_condition': self.k_condition}))          # the K heads are scored on the K they were fitted on
        self._check_rows(rows)
        return rows

    def entry_nll(self, head: str, kind: str, d: HeadData) -> np.ndarray:
        """(n,) the NLL of every design entry of ``d`` (the ``HeadData`` of ``head`` and ``kind`` from ``head_rows(...).data``), none left out: an M2Zone entry without a zone and an M3Gold entry
        without a bin score 0 (the head says so), an entry without exposure scores 0. ``nll`` is built from these scores; ``adapters.run_inference`` aggregates them over the scored entries of
        the model it compares (the entries that ``loss.scored_rows`` keeps), which is the same rule as for B1.

        설계 행마다 NLL 하나(식 (NLL-1)–(NLL-3))를 빠짐없이 낸다. 점수가 없는 행(구역 없음, 노출 0)은 0이고, 어느 행을 평균에 넣을지는 호출자가 정한다."""
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        model = (self.masked if kind == 'masked' else self.open).get(head)
        if model is None:
            raise ValueError(f'this B0 set has no {kind} head {head!r}')
        if head == 'M3Gold':
            return model.nll(d.X, d.y, d.team)
        if head == 'M3Kills' and self.kill_law != 'poisson':
            return model.nll(d.X, d.y, d.e, first_team=d.first_team)           # (KL-3): the shifted score
        if d.e is None:
            return model.nll(d.X, d.y)
        return model.nll(d.X, d.y, d.e)

    def nll(self, s1, s2, labels, role, kind: str = 'masked', *, check=None, matches=None) -> dict:
        """The row-wise NLL of every head of ``kind`` on the matches of ``role`` (DEV, say): {head name: HeadScores}. The rows without a
        zone (M2Zone) and the rows without exposure (cells beyond the game end, units without a conversion window) are not rows of the
        head's mean; they are counted in ``n_unscored`` (a cell without exposure adds 0 to the NLL). ``matches``: score only the rows of
        these matches of the role (S4b 준비 M11, 결정 D16: the rows of the model B0 is compared with); the default is every match of the role.

        머리별 설계 행 NLL (NLL-1)–(NLL-3)을 ``HeadScores``에 담아 평가 행의 점수 (HR-5)와 평균 (NLL-4)로 이어 준다."""
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        tick = check if check is not None else (lambda: None)
        tick()
        rows = self.head_rows(s1, s2, labels, role, matches=matches)
        group = self.masked if kind == 'masked' else self.open
        out = {}
        for name, head in group.items():
            tick()
            d = rows.data(name, kind)
            if name == 'M2Zone':
                scored = d.y >= 0
                out[name] = HeadScores(head.nll(d.X[scored], d.y[scored]), d.match[scored], d.row[scored], int((~scored).sum()))
            else:                                                                  # one rule for every other head (``entry_nll``), also the one of ``adapters.run_inference``
                out[name] = HeadScores(self.entry_nll(name, kind, d), d.match, d.row, 0 if d.e is None or name == 'M3Gold' else d.n_zero_exposure)
        return out

    def _check_rows(self, rows) -> None:
        """ValueError unless the rows ``rows`` (a ``HeadRows``) hold the K block of the condition of this set: the same ``k_condition`` and, for a binned set that records the sha256 of its
        representatives, the same 31 representatives. The rows of the other condition are refused in both directions (a raw set on binned rows, a binned set on raw rows).

        K 조건 'binned'의 대표값은 행을 만든 라벨 저장소의 범위 안 FIT 구간 평균이다. 같은 조건이어도 다른 저장소에서 온 31개는 다른 K 변수이므로, 맞출 때 적은 sha256과 다르면 채점하지 않는다
        (B1의 FIT/STOP 행 점검과 같은 뜻이다: 한 묶음의 K 변수는 하나이다)."""
        if rows.k_condition != self.k_condition:
            raise ValueError(f"the rows have the k_condition {rows.k_condition!r} and the B0 set was fitted on {self.k_condition!r}: the open object and conversion heads would be scored "
                             'on another input')
        if self.k_reps_sha256 is not None and rows.k_reps_sha256() != self.k_reps_sha256:
            raise ValueError('the rows hold other representatives of the gold bins than the B0 set was fitted on (k_reps_sha256 differs): the open object and conversion heads would be '
                             'scored on another K')

    # -- the composer ----------------------------------------------------------------------------------------------------------------------
    def composer_heads(self, mode: str = 'masked') -> 'heads.ComposerHeads':
        """The masked heads as a ``heads.ComposerHeads`` (call ``bind`` before asking for rows). The open heads are scored, not composed.

        합성은 t₀ 정보만 쓰는 가린 머리로 하므로 열린 머리는 거부한다(설계서 3.4절, 8절의 평가 층)."""
        if mode == 'open':
            raise ValueError("composer_heads('open'): the open B0 heads are scored (B0Heads.nll), not composed")
        if mode != 'masked':
            raise ValueError(f"composer_heads: mode must be 'masked', not {mode!r}")
        return _MaskedChain(self.masked, self.fit_means)

    def predict_all(self, batch: HeadBatch) -> np.ndarray:
        """(n, 184) every masked head on the rows of ``batch``, side by side in the order of ``PREDICT_ALL_WIDTHS``: the rate of the 18
        bins (per second), the laws of the mark, the zone and the size class, the mean kills of the two teams, the 62 gold probabilities
        (blue 31, then red 31, for the kill layer min(max(kills, 1), 6) of each team), the expected objectives and the conversion rates
        (per second). TypeError for something that is not a ``HeadBatch``, ValueError for rows out of range.

        (HEAD-1)–(HEAD-4)의 머리 전부를 한 배치에서 계산한다. 골드 머리에 주는 킬 수 층은 팀의 킬 수 k에서 ξ = min(max(⌊k⌉, 1), 6)이다(⌊k⌉은 가장 가까운 정수, 동점은
        짝수 쪽; 층은 1–6, 계획 P4). 킬이 0인 팀에도 층 1의 분포를 낸다."""
        if not isinstance(batch, HeadBatch):
            raise TypeError(f'predict_all takes a HeadBatch, not {type(batch).__name__}')
        chain = self.composer_heads('masked')
        chain._set_rows(batch.row_view)
        n = len(batch.q)
        rows = np.arange(n)
        K = np.asarray(batch.K, dtype=np.float64)
        if K.shape != (n, N_K) or not np.isfinite(K).all() or (K < 0).any():
            raise ValueError(f'K must be a finite, non-negative array of shape ({n}, {N_K})')
        # 식 (HR-3)의 층 ξ를 K의 킬 수 열에서 만들어 골드 머리의 조건으로 준다.
        layer = np.clip(np.rint(K[:, :2]), 1, N_LAYER).astype(np.int64)
        m4 = features.m4_masked_inputs(chain.row_view(rows), batch.q, batch.m, batch.z, K, batch.O)
        pieces = [chain.rate(rows), chain.mark_prob(rows, batch.q), chain.zone_prob(rows, batch.q, batch.m),
                  chain.size_prob(rows, batch.q, batch.m, batch.z), chain.kill_rate(rows, batch.q, batch.m, batch.z, batch.s),
                  chain.gold_prob(rows, batch.q, batch.m, batch.z, batch.s, layer).reshape(n, -1),
                  chain.obj_rate(rows, batch.q, batch.m, batch.z, batch.s, K), chain.conv_rate(m4)]
        return np.hstack(pieces)

    # -- reports and files -----------------------------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        """{'masked': {head: counts}, 'open': {head: counts}} as plain JSON values (rows, left-out rows, cells without events, whether the
        optimiser converged)."""
        return {kind: {name: head.summary() for name, head in getattr(self, kind).items()} for kind in KINDS}

    def save(self, directory) -> Path:
        """Write ``b0_heads.npz`` and ``b0_heads.json`` into ``directory`` (created). FileExistsError if either file is there: a saved
        set is never written over.

        적합한 머리를 합성과 보고가 다시 읽도록 계수 배열(npz)과 설계 설명(JSON) 한 쌍으로 남긴다."""
        out = Path(directory)
        if (out / NPZ_NAME).exists() or (out / JSON_NAME).exists():
            raise FileExistsError(f'{out} already holds saved B0 heads: they are not written over')
        blob, text = self._files()
        out.mkdir(parents=True, exist_ok=True)
        store_core.write_atomic(out / NPZ_NAME, blob)
        store_core.write_atomic(out / JSON_NAME, text)
        return out

    def saved_sha256(self) -> dict:
        """``{'npz_sha256', 'json_sha256'}``: the sha256 of the two files that ``save`` writes, computed without writing anything. This is the identity of the set as a file artefact (the JSON also holds the
        array file's sha256), which ``adapters.run_inference`` records in its provenance when it is given a B0 (S4b preparation M11 review, m4).

        세트를 파일로 저장했을 때의 두 파일의 SHA-256. 저장 없이 계산하며, 평가 기록의 B0 정체성으로 쓴다."""
        blob, text = self._files()
        return {'npz_sha256': hashlib.sha256(blob).hexdigest(), 'json_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}

    def _files(self) -> tuple:
        """``(npz bytes, JSON text)`` exactly as ``save`` writes them."""
        arrays = {'fit_means': self.fit_means}
        described = {kind: {} for kind in KINDS}
        for kind in KINDS:
            for name, head in getattr(self, kind).items():
                head_arrays, meta = head.state()
                for field, a in head_arrays.items():
                    arrays[f'{kind}.{name}.{field}'] = a
                layout = layout_for(name, kind, self.kill_law)
                numeric_names = list(features.B0_NAMES)
                if layout.n_numeric > N_B0:
                    numeric_names += ['kills_blue', 'kills_red', 'gold_blue_k', 'gold_red_k']
                if layout.n_numeric > N_B0 + N_K:
                    numeric_names += list(features.O_NAMES)
                meta['columns'] = column_names(layout, numeric_names)
                described[kind][name] = {**meta, **head.summary()}
        blob = _write_npz_bytes(arrays)
        import sklearn
        doc = {'schema': SCHEMA, 'npz_sha256': hashlib.sha256(blob).hexdigest(), 'fit_roles': list(self.fit_roles),
               'b0_names': list(features.B0_NAMES),
               'fits': {'poisson': {'alpha': ALPHA, 'max_iter': MAX_ITER_POISSON, 'tol': TOL_POISSON},
                        'logistic': {'C': C_LOGISTIC, 'max_iter': MAX_ITER_LOGISTIC}, 'mix': MIX, 'sklearn': sklearn.__version__},
               'heads': described}
        if self.kill_law != 'poisson':
            doc['kill_law'] = self.kill_law                         # absent means the plain law: a plain set is saved as every earlier set was
        if self.k_condition != 'raw':
            doc['k_condition'] = self.k_condition                   # absent means the raw gold: a raw set is saved as every earlier set was
            if self.k_reps_sha256 is not None:
                doc['k_reps_sha256'] = self.k_reps_sha256           # the 31 representatives of the binned K block (sha256 of the float64 little-endian bytes)
        return blob, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n'

    @classmethod
    def load(cls, directory) -> 'B0Heads':
        """Read heads saved by ``save``. ValueError for another schema, for an array file whose SHA-256 differs from the JSON's, and for
        arrays that do not fit the described heads; FileNotFoundError for a missing file.

        스키마, 배열 파일의 SHA-256, 설계 배치가 저장 때와 같을 때만 머리를 돌려준다. B2-a 이전의 저장본(스키마 v2)은 M2Rate에 창이 없어 거부한다(판정 R36)."""
        folder = Path(directory)
        doc = json.loads((folder / JSON_NAME).read_text(encoding='utf-8'))
        if doc.get('schema') != SCHEMA:
            raise ValueError(f'{folder / JSON_NAME}: the schema is {doc.get("schema")!r}, not {SCHEMA!r}')
        if store_core.sha(folder / NPZ_NAME) != doc.get('npz_sha256'):
            raise ValueError(f'{folder / NPZ_NAME}: the array file has changed (its sha256 differs from the one in the JSON)')
        with np.load(folder / NPZ_NAME, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        wanted = {'masked': HEAD_NAMES, 'open': OPEN_HEAD_NAMES}
        if any(set(doc['heads'].get(kind, {})) != set(names) for kind, names in wanted.items()):
            raise ValueError('the saved set does not hold the eight masked and the five open heads')
        kill_law = killlaw.check_law(doc.get('kill_law', 'poisson'))
        k_condition = head_rows.check_k_condition(doc.get('k_condition', 'raw'))
        groups = {}
        for kind, names in wanted.items():                                              # the order of the head names, not that of the JSON
            groups[kind] = {}
            for name in names:
                meta = doc['heads'][kind][name]
                cls_ = HEAD_CLASSES.get(meta['class'])
                if cls_ is None or cls_.name != name or meta['kind'] != kind:
                    raise ValueError(f'{kind} head {name!r}: the JSON names the class {meta["class"]!r} of kind {meta["kind"]!r}')
                prefix = f'{kind}.{name}.'
                part = {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
                head = cls_.from_state(part, meta)
                layout = layout_for(name, kind, kill_law)
                for single in head.parts:
                    same_design = (single.n_numeric_, single.blocks_, single.WINDOWS) == (layout.n_numeric, layout.blocks, layout.windows)
                    if not same_design or single.coef_std_.shape[1] != layout.model_width:
                        raise ValueError(f'{kind} head {name!r}: the saved design is not the design of the layout of this module')
                groups[kind][name] = head
        condition = {} if k_condition == 'raw' else {'k_condition': k_condition}
        recorded = {'k_reps_sha256': doc['k_reps_sha256']} if 'k_reps_sha256' in doc else {}         # a sha256 of a raw set, or a malformed one, is refused by the constructor
        return cls(groups['masked'], groups['open'], arrays['fit_means'], doc['fit_roles'], **({} if kill_law == 'poisson' else {'kill_law': kill_law}), **condition, **recorded)
