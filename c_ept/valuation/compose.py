r"""[모듈 개요 및 하네스 위치]
This module evaluates value forecasts using the t0 information plus the censoring time T of the match (decision P10: exposures and conversion windows are cut at T - 30 s and at T; censoring is observation design, not a model input).

이 모듈은 평가 시점(t₀ 분 프레임)에서 각 머리(heads)가 예측한 조건부 확률법칙들을 연결하여
향후 예측 시계(H = 60초, 120초, 180초) 동안 발생하는 교전들의 미래 누적 가치 V_start(t₀, H)를
합성(Composition)하는 핵심 가치 합성 엔진이다 (stage S3 Task 9; 결정 P4, P6, P10; 관문 D6).

A. 하네스 및 파이프라인 내 위치:
   - 교전 발생(M2), 결과(M3), 사후 전환(M4) 머리들을 체이닝하여 교전 순 골드 가치(f₀)를 산출한다.
   - 두 가지 합성 경로 지원:
     1) 닫힌 꼴(Closed-Form) 기댓값 계산 (Composer.expect): B0 GLM처럼 M3 및 M4 머리가 K와 O를 조건으로
        읽지 않는 독립 구조일 때 켐벨 정리(Campbell's theorem)를 통해 해석적 기댓값을 완전 정밀 계산 (식 COMP-6).
     2) 몬테카를로 모의 시뮬레이션 (Composer.simulate): B1 LightGBM 및 EPT 신경망 모형처럼 사슬의
        선행 결과(K, O)를 조건부로 반영하는 모든 모형에 대해 S회 복원 표집을 수행하여 평균, 표준편차, 19개 분위수 산출 (식 COMP-1 ~ COMP-4).
   - 관문 검증(Gate D6): 동일한 머리와 가격 하에서 B0의 몬테카를로 모의 평균과 닫힌 꼴 기댓값의 차이가
     모의 표준오차 이내로 일치하는지 검증한다 (결정 P7 (iv)).

B. 모듈 아키텍처 및 7대 블록 구성:
   - 블록 1: 전역 상수 및 설정 (H_SECONDS: 60, 120, 180s, 19개 분위수, 워커 환경 변수)
   - 블록 2: 전환 노출 시간 계산 (conv_exposure_s; 결정 P10, 식 COMP-5)
   - 블록 3: 프레임 행 테이블 및 뷰 바인딩 (RowTable, row_table; t₀ 프레임 뷰만 보관)
   - 블록 4: 결과 데이터 구조 (SimResult, Draws; 모의 표본 및 요약 통계량)
   - 블록 5: 수치 역변환 및 헬퍼 (_inverse_cdf, _poisson_head, _checked)
   - 블록 6: 합성기 코어 (Composer; expect, simulate, expected_onsets)
   - 블록 7: 멀티프로세스 병렬 워커 풀 (ProcessPoolExecutor, E6 스트라타 가드 check 연동)

C. 합성 사슬 및 가치 회계 규칙 (식 COMP-3, COMP-4):
   - 발생 표집: 각 구간 q별 발생 수 n_{jq} ~ Poisson(λ_q · Δ_q)
   - 단일 발생 사슬:
     1) 첫 킬 표지 m ~ π(· | q)
     2) 첫 킬 구역 z ~ ρ(· | q, m) (관측 구역 0 .. 9 중 역변환 표집)
     3) 교전 규모 s ~ P(· | q, m, z)
     4) 팀별 킬 수 k_blue, k_red ~ Poisson(kill_rate)
     5) 킬 골드: k ≥ 1인 팀은 gold_prob(layer = min(k, 6))에서 구간 표집 후 대표 골드 환산, 0킬 팀은 0골드 확정
     6) K = (k_blue, k_red, gold_blue, gold_red)
     7) 오브젝트 O ~ Poisson(obj_rate(rows, q, m, z, s, K))
     8) 전환 수 ~ Poisson(conv_rate(m4_inputs) · conv_exposure_s)
   - 가치 계산 (식 COMP-4):
     v = (gold_blue − gold_red) + Σ_c sign_c · price_{qc} · (O_c + conversions_c)
     (포탑 칸은 towertier.expected_tower_price를 통해 t₀ 시점 상대 기파괴 상태 기준 기대 가격 적용)

D. 핵심 수식 (COMP-1 ~ COMP-8):
   (COMP-1)  모의 추정량     V_j(H) = Σ_{q<H/10s} Σ_{ι=1}^{n_{jq}} v_{jqι},  V̂(H) = (1/S)·Σ_j V_j(H),  sd(H),  Q_α(H)
   (COMP-2)  발생 수         n_{jq} ~ Poisson(λ_q · Δ_q)
   (COMP-3)  한 발생의 사슬  m → z → s → (k_blue, k_red) → 골드 구간 → O → 전환 수 y_c
   (COMP-4)  발생의 가치     v = g_blue − g_red + Σ_c sign_c · price_{qc} · (O_c + y_c)
   (COMP-5)  전환 노출       e_q = max(0, min(60s, T − δ̂)),   δ̂ = t₀ + (q + 0.5)·10s + 20s
   (COMP-6)  닫힌 꼴         E[V(H)] = Σ_{q<H/10s} λ_q · Δ_q · Σ_m π(m | q) · Σ_z ρ(z | q, m) · Σ_s P(s | q, m, z) · v̄(q, m, z, s)
   (COMP-7)  기대 킬 골드    E[g_t] = Σ_{k=1}^{5} P(k; μ_t) · G_t(k) + P(≥ 6; μ_t) · G_t(6)
   (COMP-8)  기대 발생 수    E[N(H)] = Σ_{q<H/10s} λ_q · Δ_q

E. 하네스 제약 및 결정성 보장:
   - 결정론적 난수열: 각 행의 난수 생성기는 default_rng([seed, global_frame_row])로 초기화되어 워커 수와 블록 크기 무관 동일 재현 보장.
   - 엄격한 정보 누설 방지: 사건 뷰는 절대 바인딩하지 않으며 오직 t₀ 프레임 뷰만을 바인딩 (규약 A3).
   - E6 스트라타 가드: 각 연산 블록 디스패치 전 check 콜백을 호출하여 백그라운드 모델 서버 실행 방지.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P4 (the factorisation and the probability families), P6 (the masked heads: M3
#      objects without K, M4 without K and O in the closed form), P10 (the exposures), the global constraints A3 (t0 information only) and
#      "determinism" (the generator of a row from the seed and the row number), Review Focus 3 (no onset and no conversion without
#      exposure) and Task 9; controller rulings R3 (the protocol of ``heads``), R13 (strictly positive categorical outputs), R22.
# Ref: docs/superpowers/notes/2026-10-21-math-control-03.md section E (one composed row: 32.16, and 22.12 when the game ends at 1,075 s).
# Ref: Kingman, J. F. C. (1993), "Poisson Processes", Oxford University Press, chapter 3; doi:10.1093/oso/9780198536932.001.0001 (Campbell's
#      theorem: the expected sum of a function over the points of a marked Poisson process is the integral of its mean over the intensity).
# Ref: Devroye, L. (1986), "Non-Uniform Random Variate Generation", Springer, section III.2; doi:10.1007/978-1-4613-8643-8 (inverse
#      transform sampling of a discrete law).
# Ref: Abramowitz, M. and Stegun, I. A. (1964), "Handbook of Mathematical Functions", section 6.5.13 (the Poisson tail P(k >= a) is the
#      regularised lower incomplete gamma function P(a, mu)).
# Ref: NumPy documentation, "numpy.random.SeedSequence" (a list of integers as entropy gives independent streams: one for each row).
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import math
import multiprocessing as mp
import numbers
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc

from c_ept import config as C
from c_ept.preprocessing import features
from c_ept.valuation import flowprice, towertier
from c_ept.models.heads import OUTPUT_WIDTHS, ComposerHeads

# =============================================================================
# 블록 1: 전역 상수 및 설정 (H_SECONDS: 60/120/180s, 분위수, 워커 환경)
# =============================================================================
DEFAULT_S = 1000                                     # simulations of a row (the gate of D6 uses 1,000 for B0 and 200 for B1)
DEFAULT_SEED = 20261021                              # the base seed of stage S3
DEFAULT_BLOCK = 16                                   # rows in the unit of work that goes to a worker
MAX_WORKERS = 61                                     # the limit of a process pool on Windows
MAX_ONSETS_PER_ROW = 1_000_000                       # a row whose rate implies more onsets in its S simulations is refused (memory)
CALL_ROWS = 1024                                     # inputs in one call of a head of the closed form: its outputs then stay in the cache
# the environment of a worker process: MKL results depend on the thread count and (in the default mode) on the alignment of the operands,
# so each worker uses one thread and MKL's compatible branch; with it the workers agree with each other to the last bit
WORKER_ENVIRONMENT = {'MKL_CBWR': 'COMPATIBLE', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
N_Q, N_M = C.N_BINS, C.N_MARKS
N_Z, N_S = OUTPUT_WIDTHS['zone_prob'][0], OUTPUT_WIDTHS['size_prob'][0]
N_GOLD = OUTPUT_WIDTHS['gold_prob'][1]
N_CELLS = 2 * len(C.O_KINDS)                         # the cell of an object: 2 kind + taker (0 blue, 1 red), as in S1
N_LAYER = 6                                          # the kill layers of the gold head: 1 .. 5 kills, 6 = six or more
N_POISSON = N_LAYER                                  # P(k) is written out for k = 0 .. 5; the rest is the tail
HORIZON_BINS = tuple(h // C.BIN_MS for h in C.EVAL_H_MS)          # (6, 12, 18): the bins of V_start(t0, 60 / 120 / 180 s)
Q_LEVELS = np.arange(1, 20) / 20.0                   # the 19 quantile levels 0.05 .. 0.95
Q_LEVELS.setflags(write=False)
BLUE, RED = 0, 1
_TOWER = C.O_KINDS.index('tower')

if (N_Z, N_S, N_GOLD) != (features.N_ZONES, 4, 31) or HORIZON_BINS[-1] != N_Q or N_CELLS != features.N_O:
    raise RuntimeError('the widths of the heads (zones, sizes, gold bins), the horizons and the cells of O must be those of features and '
                       'config')
if any(name not in flowprice.GROUPS for name in C.O_KINDS if name != 'tower'):
    raise RuntimeError('every kind of object but the tower must be a group of the price table')


# =============================================================================
# 블록 2: 전환 노출 시간 계산 (decision P10, 식 COMP-5)
# =============================================================================
def conv_exposure_s(t0_ms, q, T_ms):
    """Seconds of conversion window that an onset of the bin ``q`` of a row at ``t0_ms`` can have in a game that ends at ``T_ms``:
    max(0, min(60, (T - delta_hat) / 1000)), delta_hat = t0 + (q + 0.5) x 10 s + 20 s (decision P10). Arguments are numbers or arrays that
    broadcast; the answer is a float for numbers and an array otherwise. ValueError for a q that is no bin (0 .. 17) and for a value that is
    not finite.

    수식
    ----
    (COMP-5)  e_q = max(0, min(60 s, T − δ̂)),      δ̂ = t₀ + (q + 0.5)·10 s + 20 s                       (코드는 시각을 ms로 받아 e_q를 초로 낸다)

      기호 | 뜻                                          | 단위             | 범위     | 코드 변수
      e_q  | 구간 q의 발생이 가질 수 있는 전환 창의 길이   | s                | 0–60     | 반환값
      t₀   | 분 프레임의 기준 시각                         | s (코드는 ms)    | ≥ 0      | t0_ms
      T    | 경기 끝 시각                                  | s (코드는 ms)    | > 0      | T_ms
      δ̂    | 교전 끝의 추정                                | s (코드는 ms)    | —        | delta_hat
      q    | 발생 구간 번호                                | —                | 0, …, 17 | q

    해석
    ----
    식 (COMP-5)는 발생이 속한 교전이 끝난 뒤 60초의 전환 창 가운데 경기 안에 드는 길이이며, 교전의 끝은 실제 δ가 아니라 t₀와 q만으로 정한 추정 δ̂를 쓴다. δ̂는 첫 킬 시각 f를 구간의 가운데로
    두고 교전의 끝 δ = max(마지막 킬 + 1 ms, f + 20 s)의 둘째 항으로 정한 값이다(계획 P10, 설계서 2절). 실제 δ를 쓰지 않는 것은 가린 합성이 교전의 끝을 알아서는 안 되기 때문이다(설계서 3.3절
    누설 시험 (c)). 경기가 T에서 끝나므로 T − δ̂가 60초보다 작으면 창이 줄고 δ̂ ≥ T이면 창이 없어 전환 노출은 0이다(계획 P10, 검토 초점 3). q는 구간 번호이므로 0–17의 정수가 아니면 거부한다(계획 P10).

    근거
    ----
    계획 P10(전환 노출의 정의), 검토 초점 3(노출이 없으면 전환이 없다), 설계서 2절(교전의 끝 δ)과 3.3절(전환 창, 누설 시험 (c)), 검산 노트 math-control-03 E절(손 계산;
    시험이 같은 수를 확인한다).
    """
    arrays = []
    for name, x in (('t0_ms', t0_ms), ('q', q), ('T_ms', T_ms)):
        a = np.asarray(x)
        if a.dtype.kind not in 'iuf':
            raise TypeError(f'{name} must be numbers, not {a.dtype}')
        a = a.astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'{name} must be finite')
        arrays.append(a)
    t0, bin_, T = arrays
    if ((bin_ < 0) | (bin_ >= N_Q) | (bin_ != np.floor(bin_))).any():
        raise ValueError(f'q must be a bin number, a whole number in 0 .. {N_Q - 1}')
    delta_hat = t0 + (bin_ + 0.5) * C.BIN_MS + C.DELTA_AFTER_F_MS
    out = np.clip((T - delta_hat) / 1000.0, 0.0, C.CONV_MS / 1000.0)
    return float(out) if out.ndim == 0 else out


# =============================================================================
# 블록 3: 프레임 행 테이블 및 뷰 바인딩 (RowTable, row_table)
# =============================================================================
_ROW_FIELDS = (('frame_row', ()), ('match', ()), ('t0_ms', ()), ('exposure_ms', (N_Q,)), ('T_ms', ()), ('views', ()), ('destroyed', (2, 4)))


def _whole_array(name: str, values, trail: tuple) -> np.ndarray:
    """A read-only int64 copy of ``values``. TypeError unless every entry is a whole number (a float or a bool is refused), ValueError for
    the wrong shape: (n,) + ``trail``."""
    a = np.array(values)
    if a.size == 0 and a.dtype.kind == 'f':
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must hold whole numbers, not {a.dtype}')
    a = a.astype(np.int64)
    if a.ndim != 1 + len(trail) or a.shape[1:] != trail:
        raise ValueError(f'{name} must have shape (n{"".join(f", {t}" for t in trail)}), not {a.shape}')
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class RowTable:
    """The t0 information of some frame-clock rows, in the order of S1; the number of a row in the table (0 .. n - 1) is what the heads, the
    generator of a row and ``Composer.expect`` / ``simulate`` call it.

    frame_row    (n,)      the row in ``frame_rows/*`` of the S1 store: its global number, which seeds the generator of the row
    match        (n,)      the match of the row
    t0_ms        (n,)      the reference time of the row (the frame time), ms
    exposure_ms  (n, 18)   the part of each occurrence bin inside (t0, min(t0 + 180 s, T - 30 s)], 0 .. 10,000 ms
    T_ms         (n,)      the end of the game, ms
    views        (n,)      the number of the frame-clock S2 view of the row (the heads are bound to these views, in this order)
    destroyed    (n, 2, 4) the towers of each tier (outer, inner, base, nexus) that team 0 (blue) and team 1 (red) have lost before t0
    roles        the roles of the rows, a tuple of FIT / STOP / DEV (empty for a table made by hand)

    The arrays are read-only int64 copies.

    식 (COMP-2)–(COMP-6)이 읽는 t₀ 정보의 표이다. 합성은 이 표와 머리가 읽는 t₀ 행의 특징을 쓰며(설계서 10절 A3), 경기 끝 T는 노출과 전환 창을 자르는 관측 설계로만 쓴다(계획 P10)."""
    frame_row: np.ndarray
    match: np.ndarray
    t0_ms: np.ndarray
    exposure_ms: np.ndarray
    T_ms: np.ndarray
    views: np.ndarray
    destroyed: np.ndarray
    roles: tuple = ()

    def __post_init__(self):
        n = None
        for name, trail in _ROW_FIELDS:
            a = _whole_array(name, getattr(self, name), trail)
            if n is not None and len(a) != n:
                raise ValueError(f'the arrays of a row table must have the same number of rows, but {name} has {len(a)} and the first '
                                 f'has {n}')
            n = len(a) if n is None else n
            object.__setattr__(self, name, a)
        for name in ('frame_row', 'match', 't0_ms', 'views'):
            if (getattr(self, name) < 0).any():
                raise ValueError(f'{name} must not be negative')
        if (self.T_ms <= 0).any():
            raise ValueError('T_ms must be positive')
        if ((self.exposure_ms < 0) | (self.exposure_ms > C.BIN_MS)).any():
            raise ValueError(f'exposure_ms must lie in 0 .. {C.BIN_MS}')
        if (self.destroyed < 0).any():
            raise ValueError('destroyed must not be negative')
        if isinstance(self.roles, str) or not all(isinstance(r, str) for r in self.roles):
            raise TypeError('roles must be a tuple of role names')
        roles = tuple(self.roles)
        if not set(roles) <= set(flowprice.S3_ROLES):
            raise ValueError(f'roles must be among {", ".join(flowprice.S3_ROLES)}: a row table never holds a holdout role or one '
                             'that S3 does not use')
        object.__setattr__(self, 'roles', roles)

    def __len__(self) -> int:
        return len(self.t0_ms)


def _read(store, name: str) -> np.ndarray:
    """``store.array(name)`` as an array; a missing array is a ValueError that names the array (never a path)."""
    try:
        return np.asarray(store.array(name))
    except FileNotFoundError:
        raise ValueError(f'the store has no array {name}') from None


def _frame_views(s2, frame_row: np.ndarray, match: np.ndarray, t0: np.ndarray, n_frame_rows: int) -> np.ndarray:
    """The number of the frame-clock S2 view of each frame row: the view with ``views/src`` 0 and ``views/src_index`` = the row. The event
    views (src 1 and 2) have colliding ``src_index`` numbers and are never taken. The views are checked against S1: one view for each
    row, its match number and its reference time (the frame time). ValueError (counts only, no key) otherwise."""
    src, index = _read(s2, 'views/src'), _read(s2, 'views/src_index').astype(np.int64)
    if src.ndim != 1 or src.shape != index.shape:
        raise ValueError('views/src and views/src_index must be 1-d arrays of one length')
    frame_views = np.flatnonzero(src == 0)
    found = index[frame_views]
    if found.size and (found.min() < 0 or found.max() >= n_frame_rows):
        raise ValueError('views/src_index names a frame row that the S1 store does not have')
    if np.bincount(found, minlength=n_frame_rows).max(initial=0) > 1:
        raise ValueError('two frame-clock views of the S2 store have the same src_index')
    view_of_row = np.full(n_frame_rows, -1, dtype=np.int64)
    view_of_row[found] = frame_views
    views = view_of_row[frame_row]
    if (views < 0).any():
        raise ValueError(f'{int((views < 0).sum())} frame row(s) of the roles have no frame-clock view in the S2 store')
    if (_read(s2, 'views/match')[views] != match).any() or (_read(s2, 'views/ref')[views] != t0).any():
        raise ValueError('the frame-clock views of the S2 store have other match numbers or reference times than the S1 rows: '
                         'the S2 store does not belong to this S1 store')
    return views


def row_table(s1, s2, roles) -> RowTable:
    """The ``RowTable`` of the frame-clock rows of the matches whose role is in ``roles`` (a name or names out of FIT, STOP, DEV), in the
    order of S1. ValueError for a holdout role, a role that stage S3 does not use, an unknown name or no role (before any array is read),
    for a store without an array that is needed, and for an S2 store that does not belong to the S1 store. Reads, in S1, ``matches/role``,
    ``matches/T_ms``,
    ``frame_rows/{match, t0, exposure_ms}`` and, through ``towertier.destroyed_before``, the objective table; in S2 the arrays ``views/{src,
    src_index, match, ref}``.

    프레임 시계 행마다 t₀ 정보와 경기 끝 T만 모은다. 사건 보기가 아니라 프레임 시계 보기를 고르는 것은 가린 머리가 t₀ 뒤의 사건을 읽지 못하게 하기 위해서이다(설계서 10절 A3). 홀드아웃 역할과
    S3가 쓰지 않는 역할은 배열을 읽기 전에 거부한다(계획 전역 제약의 분할)."""
    mask = features.split_rows(s1, roles)               # the one rule for the roles: it refuses a bad role before the first array is read
    named = {roles} if isinstance(roles, str) else set(roles)
    frame_row = np.flatnonzero(mask).astype(np.int64)
    frame_match, frame_t0, frame_exposure = (_read(s1, f'frame_rows/{name}') for name in ('match', 't0', 'exposure_ms'))
    if frame_t0.shape != mask.shape or frame_exposure.shape != mask.shape + (N_Q,):
        raise ValueError('frame_rows/t0 and frame_rows/exposure_ms must hold one row (18 bins) for each frame row of the S1 store')
    match = frame_match[frame_row].astype(np.int64)
    t0 = frame_t0[frame_row].astype(np.int64)
    T = _read(s1, 'matches/T_ms')[match].astype(np.int64)
    views = _frame_views(s2, frame_row, match, t0, len(mask))
    destroyed = towertier.destroyed_before(s1, match, t0)
    return RowTable(frame_row=frame_row, match=match, t0_ms=t0, exposure_ms=frame_exposure[frame_row], T_ms=T, views=views,
                    destroyed=destroyed, roles=tuple(r for r in C.ROLES if r in named))


# =============================================================================
# 블록 4: 모의 시뮬레이션 결과 구조체 (SimResult, Draws)
# =============================================================================
@dataclass(frozen=True, eq=False)
class SimResult:
    """The simulated V_start of some rows for H = 60, 120 and 180 s: ``mean`` (n, 3), ``sd`` (n, 3; standard deviation with n - 1 of the S
    simulations) and ``q19`` (n, 3, 19; the quantiles 0.05 .. 0.95, ``Q_LEVELS``); ``S`` is the number of simulations.

    (COMP-1)의 V̂(H), sd(H), Q_α(H)를 H = 60, 120, 180 s마다 담는다."""
    mean: np.ndarray
    sd: np.ndarray
    q19: np.ndarray
    S: int


@dataclass(frozen=True, eq=False)
class Draws:
    """The onsets of the S simulations of one row, ordered by simulation and bin: ``sim`` (the simulation 0 .. S - 1), ``q`` (the bin), the
    mark ``m``, the zone ``z``, the size ``s``, the kills ``k`` (n, 2) of blue and red, the gold bin ``gold_bin`` (n, 2; -1 without a kill)
    and its value ``gold`` (n, 2), the objects ``O`` (n, 18), the conversions ``conv`` (n, 18) and the value of the onset ``value`` (blue
    minus red).

    (COMP-2)–(COMP-4)의 표본이다: 발생마다 뽑은 값과 그 가치 v를 모의 번호와 구간 순서로 담는다.

    수식
    ----
    (COMP-4)  v = f₀(K, O) + f₀(전환) = g_blue − g_red + Σ_c sign_c·price_{qc}·(O_c + y_c)      (sign_c = +1 파랑 칸 c 짝수, −1 빨강 칸 c 홀수)
              탑 칸: price_{qc} = Σ_tier P(tier | 당한 팀의 단계별 파괴 수, 가져간 팀, t₀ + (q + 0.5)·10 s)·price_tier

      기호               | 뜻                                                     | 단위    | 범위           | 코드 변수
      v                  | 발생의 가치 f₀(K, O) + f₀(전환)(파랑 − 빨강)           | 골드    | 부호 있음      | value
      g_t                | 팀 t의 킬 골드(킬 0이면 0)                             | 골드    | ≥ 0            | gold
      O_c, y_c           | 칸 c의 오브젝트 수, 전환 수                            | 개      | 0, 1, …        | O, conv
      sign_c, price_{qc} | 칸 c의 부호, 구간 q의 오브젝트 하나의 가격             | —, 골드 | ±1; 유한       | Composer._cell_values
      tier, price_tier   | 포탑 단계(바깥, 안쪽, 억제기, 넥서스)와 그 단계의 가격 | —, 골드 | 단계 4개; 유한 | towertier, prices

    해석
    ----
    식 (COMP-4)의 가치는 킬 골드와 오브젝트와 전환의 가격 합이며, O와 전환은 서로 다른 오브젝트이고 도움 풀은 g에 이미 들어 있어 같은 몫을 두 번 세지 않는다(공통 가정 (가); 판정 R11, R48). 탑 칸은 예측한 탑의 단계를
    모르므로 단계 분포의 기대 가격을 쓴다(설계서 8.1절의 포탑 단계, 계획 Task 8).

    근거
    ----
    판정 R11, R48(회계 항등식), 계획 P3·P4(킬 골드), Task 8(포탑 단계 분포), 설계서 3.4절(합성; 중복 없음 D4)과 8.1절(포탑 단계 가격).
    """
    sim: np.ndarray
    q: np.ndarray
    m: np.ndarray
    z: np.ndarray
    s: np.ndarray
    k: np.ndarray
    gold_bin: np.ndarray
    gold: np.ndarray
    O: np.ndarray
    conv: np.ndarray
    value: np.ndarray


# =============================================================================
# 블록 5: 수치 역변환 및 표집 헬퍼 (_inverse_cdf, _poisson_head)
# =============================================================================
def _positive(name: str, value, lo: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer, not {type(value).__name__}')
    if value < lo:
        raise ValueError(f'{name} must be at least {lo}, not {value}')
    return int(value)


def _prices(prices) -> dict:
    """The {group: price} dict as plain floats; TypeError for no mapping, ValueError for a group that is missing or a price that is not a
    finite number."""
    if not isinstance(prices, Mapping):
        raise TypeError('prices must be the {group: price} dict that value.load_prices returns')
    out = {}
    for group in flowprice.GROUPS:
        v = prices.get(group)
        if v is None:
            raise ValueError(f'the price table has no price for the group {group}')
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise ValueError(f'the price of {group} is not a finite number')
        out[group] = float(v)
    return out


def _gold_reps(values) -> np.ndarray:
    a = np.array(values, dtype=np.float64)
    if a.shape != (N_GOLD,):
        raise ValueError(f'gold_reps must have shape ({N_GOLD},), one representative for each gold bin, not {a.shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('gold_reps must be finite and not negative')
    a.setflags(write=False)
    return a


def _checked(name: str, out, shape: tuple, unit_sum: bool = False) -> np.ndarray:
    """The answer of the head method ``name`` as a float64 array of ``shape``; ValueError (naming the head) for another shape, a value that
    is not finite and non-negative and, when ``unit_sum``, rows of probabilities that do not add up to 1."""
    a = np.asarray(out, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f'the head {name} returned an array of shape {a.shape}, but the composer needs {shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'the head {name} returned values that are not finite and non-negative')
    if unit_sum and np.abs(a.sum(axis=-1) - 1.0).max(initial=0.0) > 1e-6:
        raise ValueError(f'the head {name} returned probabilities that do not add up to 1')
    return a


def _inverse_cdf(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The category of each uniform ``u`` (n,) in [0, 1) for the laws ``p`` (n, C): the number of cumulative probabilities that are <= u, so
    that a category with probability 0 is never drawn; capped at C - 1 against rounding.

    (COMP-3)의 역변환 표집 inv(p, U) = min(#{c : p_0 + … + p_c ≤ U}, C − 1)이다(C는 범주의 수)."""
    cdf = np.cumsum(p, axis=1)
    return np.minimum((cdf <= u[:, None]).sum(axis=1), p.shape[1] - 1).astype(np.int64)


def _spans(n: int):
    """(lo, hi) of the consecutive spans of ``range(n)`` that have at most ``CALL_ROWS`` entries."""
    for lo in range(0, n, CALL_ROWS):
        yield lo, min(lo + CALL_ROWS, n)


def _poisson_head(mu: np.ndarray) -> tuple:
    """(P(k) for k = 0 .. 5 as (..., 6), P(k >= 6) as (...)) of Poisson laws with the means ``mu`` (...).

    (COMP-7)의 P(k; μ) (k = 0 … 5)와 꼬리 P(≥ 6; μ)를 계산한다."""
    p = np.empty(mu.shape + (N_POISSON,))
    p[..., 0] = np.exp(-mu)
    for k in range(1, N_POISSON):
        p[..., k] = p[..., k - 1] * mu / k
    return p, gammainc(float(N_POISSON), mu)


# =============================================================================
# 블록 6: 가치 합성 엔진 코어 (Composer: expect, simulate, expected_onsets)
# =============================================================================
class Composer:
    """The value of the engagements after t0 from the masked heads: ``expect`` (closed form), ``simulate`` and ``simulate_onsets``.

    ``heads`` implements ``heads.ComposerHeads`` and is bound once, here, to the frame-clock views of ``rows``
    (``heads.bind(s2, rows.views)``); ``s2`` is not kept. ``prices`` is the {group: value} dict of ``value.load_prices``, ``tier_model``
    the ``towertier.TierModel`` that prices a predicted tower, ``gold_reps`` (31,) the FIT mean of each gold bin, ``rows`` a ``RowTable``.
    ``S`` (at least 2) is the number of simulations of a row and ``seed`` the base seed. The attributes ``heads``, ``rows``, ``prices``,
    ``tier_model``, ``gold_reps``, ``S`` and ``seed`` are public and not to be changed.

    ``expect``, ``simulate`` and ``expected_onsets`` take ``idx``, the numbers of rows of the table. ``block`` rows form a unit of work and
    ``workers`` processes share the units; neither changes a bit of the answer. ``check`` is the E6 hook (see ``_run``). The value of a row
    depends on the inputs of that row only: the heads, the prices, the tier model, the seed and the row of the table (its tower prices are
    made from its own tower state and times).

    ``simulate``는 (COMP-1)–(COMP-4), ``expect``는 (COMP-6)과 (COMP-7), ``expected_onsets``는 (COMP-8)을 행 하나씩 계산한다. 머리는 만들 때 한 번 t₀ 행의 보기에
    묶고(판정 R32), 가격과 포탑 단계 분포와 골드 구간 대표값은 모든 행에 같은 것을 쓴다.

    ``dead_time`` is keyword-only and must be None in S4a (Q12); refusal happens before binding."""

    def __init__(self, heads, prices, tier_model, gold_reps, rows, s2, S: int = DEFAULT_S, seed: int = DEFAULT_SEED, *, dead_time=None):
        if dead_time is not None:
            raise ValueError('dead_time must be None in S4a')
        if not isinstance(heads, ComposerHeads):
            raise TypeError('heads must implement the ComposerHeads protocol of c_ept.heads')
        if not isinstance(rows, RowTable):
            raise TypeError('rows must be a RowTable (compose.row_table)')
        if not callable(getattr(tier_model, 'prob', None)):
            raise TypeError('tier_model must have a prob(destroyed_victim, taker, t_s) method (towertier.TierModel)')
        self.S = _positive('S', S, 2)
        self.seed = _positive('seed', seed, 0)
        self.gold_reps = _gold_reps(gold_reps)
        self.prices = _prices(prices)
        self.rows = rows
        self.tier_model = tier_model
        n = len(rows)
        # 식 (COMP-4)의 sign_c·price: 파랑 칸은 +, 빨강 칸은 −로 채우고 탑 칸은 비워 둔다(``_cell_values``가 구간마다 채운다).
        self._flat_value = np.zeros(N_CELLS)                                          # sign x price of the cells but the tower cells
        for kind, name in enumerate(C.O_KINDS):
            if name != 'tower':
                self._flat_value[2 * kind], self._flat_value[2 * kind + 1] = self.prices[name], -self.prices[name]
        heads.bind(s2, rows.views)                                                              # last: the one expensive step
        self.heads = heads
        self._probe_rows = np.unique([0, n - 1]).astype(np.int64) if n else np.empty(0, dtype=np.int64)
        self._probe = np.array(heads.row_view(self._probe_rows)) if n else None                # to notice a later bind of the same heads

    # -- the inputs of one row ------------------------------------------------------------------------------------------------------------
    # Everything that a row needs besides the heads is made from the inputs of that row alone, inside the block function that computes it,
    # never for the whole table: a row has the same value in any table, block and process (R30: the calling process computes no value).
    def _seconds(self, i: int) -> np.ndarray:
        """(18,) the seconds of observation of each occurrence bin of the row i."""
        return self.rows.exposure_ms[i] / 1000.0

    def _conv_exposure(self, i: int) -> np.ndarray:
        """(18,) the seconds of conversion window of an onset in each bin of the row i (decision P10).

        (COMP-5)의 e_q를 18구간 모두에 낸다."""
        return conv_exposure_s(self.rows.t0_ms[i], np.arange(N_Q), self.rows.T_ms[i])

    def _tower_prices(self, i: int) -> np.ndarray:
        """(18 bins, 2 teams) the expected price of a tower taken by blue / red in each bin of the row i: the tier law at the tower state of
        this row (the victim is the other team) and at the game time t0 + (q + 0.5) 10 s. The law is asked twice, once for each team, with
        the 18 bins of the row as its inputs: the answer depends on nothing but this row.

        (COMP-4)의 탑 칸 가격 price_{qc}를 18구간 모두에 낸다."""
        rows = self.rows
        t_s = (rows.t0_ms[i] + (np.arange(N_Q) + 0.5) * C.BIN_MS) / 1000.0
        out = np.empty((N_Q, 2))
        for taker in (BLUE, RED):
            victim = towertier.victim_counts(rows.destroyed[i][None], np.array([taker]))              # (1, 4): the team that lost the tower
            out[:, taker] = towertier.expected_tower_price(
                self.tier_model.prob(np.repeat(victim, N_Q, axis=0), np.full(N_Q, taker, dtype=np.int64), t_s), self.prices)
        return out

    def _cell_values(self, i: int) -> np.ndarray:
        """(18 bins, 18 cells): sign x price of one object of each cell in each bin of the row i; the tower cells have the expected price of
        a predicted tower.

        (COMP-4)의 sign_c·price_{qc}를 18구간 × 18칸에 낸다."""
        tower = self._tower_prices(i)
        v = np.tile(self._flat_value, (N_Q, 1))
        v[:, 2 * _TOWER] = tower[:, BLUE]
        v[:, 2 * _TOWER + 1] = -tower[:, RED]
        return v

    def _check_bound(self) -> None:
        """RuntimeError when the heads of this composer were bound again, to other views, after it was made (the rows of a head set are one
        table at a time, so one chain of heads serves one composer): the first and the last row are asked for again."""
        if self._probe is None:
            return
        try:
            same = np.array_equal(self.heads.row_view(self._probe_rows), self._probe, equal_nan=True)
        except (ValueError, IndexError):
            same = False
        if not same:
            raise RuntimeError('the heads of this composer were bound again, to other views, after it was made: use one chain of heads '
                               'for each composer')

    def _index(self, idx) -> np.ndarray:
        a = np.asarray(idx)
        if a.size == 0 and a.ndim <= 1:
            return np.empty(0, dtype=np.int64)
        if a.dtype.kind not in 'iu':
            raise TypeError(f'row numbers must be integers, not {a.dtype}')
        if a.ndim != 1:
            raise ValueError(f'row numbers must be a flat list, not shape {a.shape}')
        if a.min() < 0 or a.max() >= len(self.rows):
            raise ValueError(f'row numbers must lie in 0 .. {len(self.rows) - 1} (the rows of the table)')
        return a.astype(np.int64)

    def _onset_mass(self, i: int) -> np.ndarray:
        """(18,) expected number of onsets in each bin of the row i: rate (per second) x seconds of observation.

        (COMP-2)의 평균 λ_q·Δ_q이다."""
        return _checked('rate', self.heads.rate(np.array([i], dtype=np.int64)), (1, N_Q))[0] * self._seconds(i)

    # -- the closed form -------------------------------------------------------------------------------------------------------------------
    def _require_closed_form(self) -> None:
        h = self.heads
        if not h.analytic_ok or h.obj_uses_K or h.conv_uses_KO:
            raise ValueError(f'these heads have no closed form (analytic_ok={h.analytic_ok}, obj_uses_K={h.obj_uses_K}, '
                             f'conv_uses_KO={h.conv_uses_KO}): the expectation is simulated, not computed')

    def _expect_row(self, i: int) -> np.ndarray:
        """(3,) E[V_start] of the row i for H = 60, 120, 180 s: the sum over every (q, m, z, s) of the bins with exposure (see the module
        text). The heads are asked about the whole grid of the row, ``CALL_ROWS`` inputs at a time (the gold head once for each kill
        layer in each call).

        (COMP-6)과 (COMP-7)을 한 행에 대해 계산한다."""
        h, reps = self.heads, self.gold_reps
        a = self._onset_mass(i)
        active = np.flatnonzero(a > 0)
        if active.size == 0:
            return np.zeros(len(HORIZON_BINS))
        nA = active.size
        # 식 (COMP-6)의 합이 도는 격자: 기대 발생 수가 0보다 큰 구간 × 표지 × 구역 × 규모 전부이며 가지치기가 없다(나머지 구간은 가중이 0이다).
        q2, m2 = (g.ravel() for g in np.meshgrid(active, np.arange(N_M), indexing='ij'))
        q3, m3, z3 = (g.ravel() for g in np.meshgrid(active, np.arange(N_M), np.arange(N_Z), indexing='ij'))
        q4, m4, z4, s4 = (g.ravel() for g in np.meshgrid(active, np.arange(N_M), np.arange(N_Z), np.arange(N_S), indexing='ij'))
        full = lambda size: np.full(size, i, dtype=np.int64)                                      # noqa: E731
        # 머리의 법칙 π(m | q), ρ(z | q, m), P(s | q, m, z)를 격자의 각 점에서 묻는다. 머리 호출은 CALL_ROWS개씩 한다(판정 R29).
        pi = _checked('mark_prob', h.mark_prob(full(nA), active), (nA, N_M), unit_sum=True)
        rho = np.empty((len(q2), N_Z))
        for lo, hi in _spans(len(q2)):
            rho[lo:hi] = _checked('zone_prob', h.zone_prob(full(hi - lo), q2[lo:hi], m2[lo:hi]), (hi - lo, N_Z), unit_sum=True)
        size = np.empty((len(q3), N_S))
        for lo, hi in _spans(len(q3)):
            size[lo:hi] = _checked('size_prob', h.size_prob(full(hi - lo), q3[lo:hi], m3[lo:hi], z3[lo:hi]), (hi - lo, N_S), unit_sum=True)
        N = len(q4)
        mu, obj = np.empty((N, 2)), np.empty((N, N_CELLS))
        # 식 (COMP-7): 층 ξ마다 G_t(ξ)를 구해 둔다. 두 팀을 같은 층으로 묻는데, 한 팀의 골드 분포가 그 팀의 층만 읽는 머리여야 팀별 값이 맞는다(공통 가정 (라)).
        G = np.empty((N, 2, N_LAYER))                                                              # E[gold of a team | its layer]
        for lo, hi in _spans(N):
            k, rows_k, cell = hi - lo, full(hi - lo), (q4[lo:hi], m4[lo:hi], z4[lo:hi], s4[lo:hi])
            mu[lo:hi] = _checked('kill_rate', h.kill_rate(rows_k, *cell), (k, 2))
            for layer in range(1, N_LAYER + 1):
                layers = np.full((k, 2), layer, dtype=np.int64)
                pg = _checked('gold_prob', h.gold_prob(rows_k, *cell, layers), (k, 2, N_GOLD), unit_sum=True)
                G[lo:hi, :, layer - 1] = np.einsum('ntb,b->nt', pg, reps)
            obj[lo:hi] = _checked('obj_rate', h.obj_rate(rows_k, *cell, np.zeros((k, 4))), (k, N_CELLS))
        pk, tail = _poisson_head(mu)
        # 식 (COMP-7): k = 0은 골드 0이어서 합에 없고(pk[..., 1:]), k ≥ 6은 꼬리 확률 하나에 층 6의 G를 곱한다. 이어서 E[g_blue] − E[g_red]를 만든다.
        gold_mean = (pk[..., 1:] * G[..., :N_POISSON - 1]).sum(axis=-1) + tail * G[..., N_POISSON - 1]       # (N, 2): 0 kills pay nothing
        gold = (gold_mean[:, 0] - gold_mean[:, 1]).reshape(nA, N_M, N_Z, N_S)
        # 전환 머리의 입력은 K = O = 0으로 만든 행렬이다: 닫힌 꼴 머리는 K와 O를 읽지 않는다(계획 P6).
        conv = np.empty((len(q3), N_CELLS))
        for lo, hi in _spans(len(q3)):
            k = hi - lo
            x = features.m4_masked_inputs(h.row_view(full(k)), q3[lo:hi], m3[lo:hi], z3[lo:hi], np.zeros((k, 4)), np.zeros((k, N_CELLS)))
            conv[lo:hi] = _checked('conv_rate', h.conv_rate(x), (k, N_CELLS))
        rho, size = rho.reshape(nA, N_M, N_Z), size.reshape(nA, N_M, N_Z, N_S)
        obj, conv = obj.reshape(nA, N_M, N_Z, N_S, N_CELLS), conv.reshape(nA, N_M, N_Z, N_CELLS)
        # 식 (COMP-6)의 v̄(코드의 g): 킬 골드 + 오브젝트 가치 + 전환 가치. 전환 항은 s에 의존하지 않으므로 (q, m, z)마다 한 번 구해 s 축으로 퍼뜨린다. 바깥 합은 s, z, m 순으로 접는다.
        value = self._cell_values(i)[active]                                                       # (nA, 18)
        g = gold + np.einsum('qmzsc,qc->qmzs', obj, value) \
            + np.einsum('qmzc,qc->qmz', conv, value * self._conv_exposure(i)[active][:, None])[..., None]
        per_zone_size = np.einsum('qmzs,qmzs->qmz', size, g)
        per_mark = np.einsum('qmz,qmz->qm', rho, per_zone_size)
        per_bin = a[active] * np.einsum('qm,qm->q', pi, per_mark)
        return np.array([per_bin[active < n_bins].sum() for n_bins in HORIZON_BINS])

    def _expect_block(self, idx: np.ndarray) -> np.ndarray:
        return np.array([self._expect_row(int(i)) for i in idx]).reshape(len(idx), len(HORIZON_BINS))

    def expect(self, idx, block: int = DEFAULT_BLOCK, workers: int = 1, check=None) -> np.ndarray:
        """(n, 3) the expected V_start of the rows ``idx`` for H = 60, 120, 180 s by the closed form; ValueError unless the heads have one
        (``analytic_ok`` and neither ``obj_uses_K`` nor ``conv_uses_KO``). ``check``: see ``_run`` (the E6 guard, before each block).

        수식
        ----
        (COMP-6)  E[V(H)] = Σ_{q<H/10 s} λ_q·Δ_q · Σ_m π(m | q) · Σ_z ρ(z | q, m) · Σ_s P(s | q, m, z) · v̄(q, m, z, s)
                  v̄(q, m, z, s) = E[g_blue] − E[g_red] + Σ_c sign_c·price_{qc}·(μ_c + λ_c·e_q)       (μ_c는 K = 0, λ_c는 K = O = 0으로 물은 값)
        (COMP-7)  E[g_t] = Σ_{k=1}^{5} P(k; μ_t)·G_t(k) + P(≥ 6; μ_t)·G_t(6),      G_t(ξ) = Σ_b gold_prob(ξ)[t, b]·rep_b
                  P(k; μ) = e^{−μ}·μ^k / k!,      P(≥ 6; μ) = 1 − Σ_{k=0}^{5} P(k; μ) = gammainc(6, μ)

          기호               | 뜻                                                         | 단위    | 범위      | 코드 변수
          λ_q·Δ_q            | 구간 q의 기대 발생 수                                      | 개      | ≥ 0       | a (_onset_mass)
          v̄(q, m, z, s)      | 발생 하나의 기대 가치                                      | 골드    | 부호 있음 | g
          E[g_t]             | 팀 t의 기대 킬 골드                                        | 골드    | ≥ 0       | gold_mean[:, t]
          μ_t                | 팀 t의 킬 수의 포아송 평균                                 | 개      | ≥ 0       | mu
          P(k; μ), P(≥ 6; μ) | 평균 μ인 포아송의 k 확률, 킬 여섯 개 이상의 확률(꼬리)     | —, —    | [0, 1]    | _poisson_head의 p, gammainc(6, μ)
          G_t(ξ), rep_b      | 킬 수 층 ξ인 팀 t의 골드 분포의 평균 골드, 구간 b의 대표값 | 골드    | ≥ 0       | G[..., t, ξ − 1], gold_reps
          μ_c, λ_c           | 칸 c의 오브젝트 평균(K = 0으로 물음), 전환율(초당)         | 개, 1/s | ≥ 0       | obj, conv
          e_q                | 구간 q의 전환 노출 (COMP-5)                                | s       | 0–60      | _conv_exposure

        해석
        ----
        식 (COMP-6)은 구간 q의 발생 수가 평균 λ_q·Δ_q의 포아송이고 발생마다 표지, 구역, 규모와 결과가 구간 q의 법칙으로 다른 발생과 독립으로 뽑히면 가치 합의 기대값이 기대 발생 수와 발생 하나의 기대 가치의 곱이라는 캠벨
        정리의 꼴이다 [Kingman 1993]. 가치 f₀가 K와 O에 선형이고 닫힌 꼴 머리가 K와 O를 읽지 않으므로 발생 하나의 기대 가치를 킬 골드의 기대값과 오브젝트와 전환 평균의 가격 합으로 닫을 수 있다(계획 P6). 식
        (COMP-7)은 킬이 있는 팀만 골드를 받고 골드 분포가 킬 수를 층 ξ = min(k, 6)으로만 읽으므로 킬 여섯 개 이상의 꼬리 확률을 한 층으로 묶고, 킬이 0인 팀은 골드 0이다(계획 P4). 기대 발생 수가 0보다 큰 구간만
        더하며 그런 구간이 없으면 0이다(계획 P10). 머리가 K 또는 O를 읽으면 발생 하나의 기대 가치가 K와 O의 분포에 대한 적분이 되어 이 합으로 닫히지 않으므로 거부하고, 닫힌 꼴을 표시하지 않은 머리도 거부한다(계획 P6).

        근거
        ----
        계획 P4(킬 수 층 조건부 골드), P6(닫힌 꼴 머리), P10(노출), 판정 R29(머리 호출은 CALL_ROWS 입력씩), 설계서 3.4절(전체 기댓값 법칙), 검산 노트 math-control-03 E절(손 계산).
        Ref: Kingman, J. F. C. (1993), "Poisson Processes", Oxford University Press; doi:10.1093/oso/9780198536932.001.0001 (캠벨 정리).
        """
        self._require_closed_form()
        parts = self._run('_expect_block', idx, block, workers, check)
        return np.concatenate(parts) if parts else np.empty((0, len(HORIZON_BINS)))

    def _onsets_block(self, idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), len(HORIZON_BINS)))
        for r, i in enumerate(idx):
            a = self._onset_mass(int(i))
            out[r] = [a[:n_bins].sum() for n_bins in HORIZON_BINS]
        return out

    def expected_onsets(self, idx, block: int = DEFAULT_BLOCK, workers: int = 1, check=None) -> np.ndarray:
        """(n, 3) the expected number of onsets of the rows ``idx`` in the first 60, 120 and 180 s: the sum of rate x exposure over the
        bins, one row at a time like the other two (``block``, ``workers`` and ``check`` as in ``expect``).

        수식
        ----
        (COMP-8)  E[N(H)] = Σ_{q<H/10 s} λ_q·Δ_q                                              (H = 60, 120, 180 s)

          기호 | 뜻                                    | 단위 | 범위    | 코드 변수
          N(H) | t₀ 뒤 H초 안에 첫 킬이 있는 교전의 수 | 개   | 0, 1, … | 반환값의 열
          λ_q  | 구간 q의 발생률(초당)                 | 1/s  | ≥ 0     | heads.rate
          Δ_q  | 구간 q의 관측 노출                    | s    | 0–10    | _seconds

        해석
        ----
        식 (COMP-8)은 (COMP-2)의 포아송 평균을 앞 H/10 s개 구간에서 더한 기대 발생 수이고, 관문 (i) 발생 보정의 기대 합이다(계획 P7 (i)). 실행기는 합성 전에 모든 행의 이 값을 점검하여, 터무니없는 행이
        하나라도 있으면 그 행을 빼지 않고 실행을 멈춘다(판정 R31: 행을 빼면 관문이 치우친다). 노출이 0인 구간은 평균이 0이어서 기여하지 않는다(계획 P7, P10).

        근거
        ----
        설계서 3.1절(구간 q의 발생 수는 포아송), 계획 P7 (i)(발생 보정), P10(노출), 판정 R31(발생률 선별).
        """
        parts = self._run('_onsets_block', idx, block, workers, check)
        return np.concatenate(parts) if parts else np.empty((0, len(HORIZON_BINS)))

    # -- the simulation --------------------------------------------------------------------------------------------------------------------
    def _draw_row(self, i: int) -> Draws:
        """The onsets of the S simulations of the row i (see the module text for the chain and the order of the draws).

        (COMP-2)–(COMP-4)를 한 행에 대해 계산한다."""
        h, S = self.heads, self.S
        mass = self._onset_mass(i)
        expected = float(mass.sum()) * S
        if expected > MAX_ONSETS_PER_ROW:
            raise ValueError(f'the head rate gives the row {i} about {expected:.3g} onsets in {S} simulations, more than the limit '
                             f'{MAX_ONSETS_PER_ROW}: its onset rate is out of range')
        rng = np.random.default_rng([self.seed, int(self.rows.frame_row[i])])
        # 식 (COMP-2): 모의 j, 구간 q의 발생 수 n_{jq}(노출이 0인 구간의 평균은 0이다).
        counts = rng.poisson(mass, size=(S, N_Q))                                                # onsets of each simulation and bin
        n = int(counts.sum())
        sim = np.repeat(np.arange(S, dtype=np.int64), counts.sum(axis=1))
        q = np.repeat(np.tile(np.arange(N_Q, dtype=np.int64), S), counts.ravel())
        if n == 0:
            e, e2 = np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
            return Draws(sim=sim, q=q, m=e, z=e, s=e, k=e2, gold_bin=e2, gold=np.empty((0, 2)), O=np.empty((0, N_CELLS), dtype=np.int64),
                         conv=np.empty((0, N_CELLS), dtype=np.int64), value=np.empty(0))
        rows = np.full(n, i, dtype=np.int64)
        # 식 (COMP-3): 표지, 구역, 규모를 뽑는 균등 난수 U₁, U₂, U₃를 한 번에 뽑는다. 이 호출들의 순서가 모듈 글이 정한 난수 사용 순서이다.
        u = rng.random((n, 3))
        bins, back = np.unique(q, return_inverse=True)                                           # the mark law depends on the bin only
        pm = _checked('mark_prob', h.mark_prob(np.full(len(bins), i, dtype=np.int64), bins), (len(bins), N_M), unit_sum=True)
        m = _inverse_cdf(pm[back], u[:, 0])
        z = _inverse_cdf(_checked('zone_prob', h.zone_prob(rows, q, m), (n, N_Z), unit_sum=True), u[:, 1])
        s = _inverse_cdf(_checked('size_prob', h.size_prob(rows, q, m, z), (n, N_S), unit_sum=True), u[:, 2])
        # 식 (COMP-3)의 킬 수와 킬 골드: 팀별 킬 수를 뽑고, 킬이 있는 팀만 층 ξ = min(k, 6)의 분포에서 골드 구간을 뽑는다. 킬이 0인 팀의 난수는 뽑아 두되
        # 쓰지 않고 그 팀의 골드는 0이다(계획 P4).
        mu = _checked('kill_rate', h.kill_rate(rows, q, m, z, s), (n, 2))
        k = rng.poisson(mu).astype(np.int64)
        layer = np.clip(k, 1, N_LAYER)
        pg = _checked('gold_prob', h.gold_prob(rows, q, m, z, s, layer), (n, 2, N_GOLD), unit_sum=True)
        ug = rng.random((n, 2))
        drawn = np.stack([_inverse_cdf(pg[:, t], ug[:, t]) for t in (BLUE, RED)], axis=1)
        gold_bin = np.where(k >= 1, drawn, -1)
        gold = np.where(k >= 1, self.gold_reps[drawn], 0.0)
        # 식 (COMP-3)의 K: 머리에는 뽑은 값만 넘긴다. 실제 K와 O는 들어가지 않는다(설계서 3.3절 누설 시험 (b)).
        K = np.column_stack([k, gold]).astype(np.float64)                                        # kills blue, red, kill gold blue, red
        O = rng.poisson(_checked('obj_rate', h.obj_rate(rows, q, m, z, s, K), (n, N_CELLS))).astype(np.int64)
        x = features.m4_masked_inputs(h.row_view(rows), q, m, z, K, O)
        rate = _checked('conv_rate', h.conv_rate(x), (n, N_CELLS))
        # 식 (COMP-3)의 전환 수: 평균은 전환율 × 전환 노출 e_q이다.
        conv = rng.poisson(rate * self._conv_exposure(i)[q][:, None]).astype(np.int64)            # no window, no conversion
        # 식 (COMP-4): 가치 = 킬 골드 + (오브젝트 + 전환) × 부호 × 가격. 도움 풀은 킬 골드 g에 이미 들어 있어 따로 더하지 않는다(계획 P3, P4).
        value = gold[:, 0] - gold[:, 1] + ((O + conv) * self._cell_values(i)[q]).sum(axis=1)
        return Draws(sim=sim, q=q, m=m, z=z, s=s, k=k, gold_bin=gold_bin, gold=gold, O=O, conv=conv, value=value)

    def _horizon_values(self, d: Draws) -> np.ndarray:
        """(S, 3) V_start of each simulation: the sum of the values of its onsets in the bins of the horizon.

        (COMP-1)의 V_j(H)를 계산한다. 식의 j는 V의 행(모의)이고 코드의 j는 지평의 열 번호이다."""
        V = np.zeros((self.S, len(HORIZON_BINS)))
        for j, n_bins in enumerate(HORIZON_BINS):
            sel = d.q < n_bins
            V[:, j] = np.bincount(d.sim[sel], weights=d.value[sel], minlength=self.S)
        return V

    def _simulate_block(self, idx: np.ndarray) -> tuple:
        n, H = len(idx), len(HORIZON_BINS)
        mean, sd, q19 = np.empty((n, H)), np.empty((n, H)), np.empty((n, H, len(Q_LEVELS)))
        for r, i in enumerate(idx):
            V = self._horizon_values(self._draw_row(int(i)))
            # 식 (COMP-1)의 V̂, sd, Q_α: 표준편차는 S − 1로 나누고 분위수는 선형 보간이다.
            mean[r], sd[r] = V.mean(axis=0), V.std(axis=0, ddof=1)
            q19[r] = np.quantile(V, Q_LEVELS, axis=0).T
        return mean, sd, q19

    def simulate(self, idx, block: int = DEFAULT_BLOCK, workers: int = 1, check=None) -> SimResult:
        """The simulated V_start of the rows ``idx``: ``SimResult`` with the mean, the standard deviation and the 19 quantiles over S
        simulations, for H = 60, 120, 180 s. ``check``: see ``_run`` (the E6 guard, before each block).

        수식
        ----
        (COMP-1)  V_j(H) = Σ_{q<H/10 s} Σ_{ι=1}^{n_{jq}} v_{jqι}                   (모의 j의 V_start(t₀, H); j = 1, …, S; H = 60, 120, 180 s)
                  V̂(H) = (1/S)·Σ_j V_j(H),    sd(H) = √[ Σ_j (V_j(H) − V̂(H))² / (S − 1) ]
                  Q_α(H) = {V_j(H)}의 α분위수                                  (α = 0.05, 0.10, …, 0.95; 선형 보간)

          기호       | 뜻                                                      | 단위    | 범위                    | 코드 변수
          V_j(H)     | 모의 j의 V_start(t₀, H): 구간 q < H/10 s인 발생의 가치 합 | 골드    | 부호 있음(파랑 − 빨강)  | V (``_horizon_values``; 행이 모의 j, 열이 지평)
          n_{jq}     | 모의 j의 구간 q의 발생 수 (COMP-2)                        | 개      | 0, 1, …                 | counts
          ι, v_{jqι} | 모의 j, 구간 q의 ι번째 발생과 그 가치 (COMP-4)             | —, 골드 | 1, …, n_{jq}; 부호 있음 | Draws.value
          S          | 모의 횟수                                                | 회      | ≥ 2                     | S
          V̂, sd, Q_α | 행 t₀의 V_start에 대한 모의 평균, 표준편차, 분위수          | 골드    | 유한                    | SimResult.mean, sd, q19

        해석
        ----
        식 (COMP-1)은 t₀ 행에서 본 첫 킬이 (t₀, t₀ + H]에 드는 교전들의 가치 합 V_start를 S번 모의해 그 평균, 표준편차, 19개 분위수로 요약한다(설계서 3.4절). V_start는 발생 수와
        발생마다의 가치가 모두 확률적인 합이므로 평균만이 아니라 분포를 내며, EPT는 이 경험 분위로 분위 머리를 둔 A1과 같은 19분위 핀볼 점수로 비교된다(설계서 3.4절 D3, 8절). 세 지평은 같은 표본의
        앞 6, 12, 18구간 부분합이다. 발생이 하나도 없는 모의의 V_j는 빈 합이어서 0이다. 표본 표준편차는 S − 1로 나누므로 S는 2 이상이어야 하고, 평균의 표준오차 sd/√S는 닫힌 꼴과의 일치 기준(계획
        P7 (iv): S = 1,000에서 차이가 표준오차의 3배 안에 드는 행이 99% 이상)의 척도이다.

        근거
        ----
        설계서 3.4절(합성; D3), 8절(합성 가치의 V_start CRPS와 분위 점수), 계획 P6(합성은 모의와 닫힌 꼴), P7 (iv)(모의 평균과 닫힌 꼴의 일치), 계획 전역 제약의 결정성,
        검산 노트 math-control-03 E절.
        """
        parts = self._run('_simulate_block', idx, block, workers, check)
        if not parts:
            H = len(HORIZON_BINS)
            return SimResult(mean=np.empty((0, H)), sd=np.empty((0, H)), q19=np.empty((0, H, len(Q_LEVELS))), S=self.S)
        return SimResult(mean=np.concatenate([p[0] for p in parts]), sd=np.concatenate([p[1] for p in parts]),
                         q19=np.concatenate([p[2] for p in parts]), S=self.S)

    def simulate_onsets(self, row: int) -> Draws:
        """The raw onsets of the S simulations of one row (``Draws``); ``simulate`` is a summary of these.

        수식
        ----
        (COMP-2)  n_{jq} ~ Poisson(λ_q·Δ_q)                  (j = 1, …, S; q = 0, …, 17; 행의 입력이 주어지면 모의와 구간마다 독립)
        (COMP-3)  m = inv(π(· | q), U₁),   z = inv(ρ(· | q, m), U₂),   s = inv(P(· | q, m, z), U₃)              (U₁, U₂, U₃ ~ 균등 [0, 1))
                  k_blue ~ Poisson(μ_blue),   k_red ~ Poisson(μ_red)                                            (μ = kill_rate(q, m, z, s))
                  k_t ≥ 1이면 ξ_t = min(k_t, 6),   b_t = inv(gold_prob(ξ_t)[t, ·], U_t),   g_t = rep_{b_t};   k_t = 0이면 g_t = 0
                  K = (k_blue, k_red, g_blue, g_red),   O_c ~ Poisson(μ_c(K)),   y_c ~ Poisson(λ_c(K, O)·e_q)        (c = 0, …, 17)

          기호              | 뜻                                                | 단위          | 범위               | 코드 변수
          n_{jq}, λ_q, Δ_q  | 모의 j, 구간 q의 발생 수, 발생률, 관측 노출       | 개, 1/s, s    | 0, 1, …; ≥ 0; 0–10 | counts, heads.rate, _seconds
          π, ρ, P(s)        | 표지, 구역, 규모의 조건부 분포                    | —             | [0, 1], 합 1       | mark_prob, zone_prob, size_prob
          U₁, U₂, U₃, U_t   | 표지, 구역, 규모, 팀 t의 골드 구간을 뽑는 난수    | —             | [0, 1)             | u, ug
          inv(p, U)         | 확률 행 p에서 U로 범주를 뽑는 역변환 표집         | —             | 0, …, C − 1        | _inverse_cdf
          μ_t, k_t, ξ_t     | 팀 t의 킬 수의 포아송 평균, 킬 수, 킬 수 층       | 개, 개, —     | ≥ 0; 0, 1, …; 1–6  | mu, k, layer
          b_t, g_t, rep_b   | 팀 t의 골드 구간, 킬 골드(킬 0이면 −1, 0), 대표값 | —, 골드, 골드 | −1, 0–30; ≥ 0      | gold_bin, gold, gold_reps
          O_c, y_c, e_q     | 칸 c의 오브젝트 수, 전환 수, 전환 노출 (COMP-5)   | 개, 개, s     | 0, 1, …; 0–60      | O, conv, _conv_exposure
          μ_c(K), λ_c(K, O) | 칸 c의 오브젝트 평균, 전환율(뽑은 값을 읽는다)    | 개, 1/s       | ≥ 0                | obj_rate, conv_rate

        해석
        ----
        식 (COMP-2)와 (COMP-3)은 구간마다 발생 수를 포아송으로 뽑고, 발생마다 EPT의 분해 순서(표지, 구역, 규모, 팀별 킬과 킬 골드, 오브젝트, 전환)를 앞 단계의 값에 조건부로 한 단계씩 뽑는다(계획 P4; 범주는 역변환
        표집 [Devroye 1986]). 뽑은 값의 가치 v는 ``Draws``의 (COMP-4)가 정한다. 킬이 있는 팀의 골드는 킬 수 층의 분포에서 구간을 뽑아 그 구간의 FIT 평균으로 쓰고 킬이 없는 팀의 골드는 0이다(계획 P4).
        머리에 들어가는 K와 O는 뽑은 값이고 실제 값은 들어가지 않는다(설계서 3.3절 누설 시험 (b)). S번 모의의 기대 발생 수가 한도를 넘는 행은 표본이 메모리를 넘으므로 거부한다.

        근거
        ----
        계획 P4(분해와 확률 족, 킬 골드), P6(가린 머리), P10(노출), Task 9(모의 순서), 설계서 3.2절(K, O), 3.3절(누설 시험), 3.4절(합성), 검산 노트 math-control-03 E절(손 계산).
        Ref: Devroye, L. (1986), "Non-Uniform Random Variate Generation", Springer; doi:10.1007/978-1-4613-8643-8 (역변환 표집).
        """
        self._check_bound()
        return self._draw_row(int(self._index([row])[0]))

    # -- blocks and workers ----------------------------------------------------------------------------------------------------------------
    def _run(self, method: str, idx, block, workers, check=None) -> list:
        """The result of ``method`` (a block function of this class) for every block of ``idx``, in order. With one worker the blocks are
        computed in this process; with two or more every block, also a single one, is computed in a spawned worker, and the calling process
        computes no value (R30). A row is computed alone in either case, so the split changes nothing.

        ``check``, a function without arguments (the E6 guard: the runner passes its Strata guard), is called before each block is
        computed or dispatched: before the first block and between blocks. Whatever it raises stops the run: no further block is
        dispatched (a block that is in flight in a worker is finished, one that has not started is cancelled). A worker process gets at
        most one block at a time, so a refusal takes effect after at most ``workers`` blocks."""
        self._check_bound()
        idx = self._index(idx)
        block = _positive('block', block)
        workers = _positive('workers', workers)
        if workers > MAX_WORKERS:
            raise ValueError(f'workers must be at most {MAX_WORKERS}, not {workers}')
        if check is not None and not callable(check):
            raise TypeError('check must be a function without arguments (the E6 guard) or None')
        chunks = [idx[a:a + block] for a in range(0, len(idx), block)]
        parts = [None] * len(chunks)
        if workers == 1 or not chunks:
            for k, chunk in enumerate(chunks):
                if check is not None:
                    check()
                parts[k] = getattr(self, method)(chunk)
            return parts
        n_processes = min(workers, len(chunks))
        pending = {}                                                                         # future -> the number of its block
        # a pool of concurrent.futures, not multiprocessing.Pool: a worker that dies (memory) is a BrokenProcessPool, not a hang
        with _worker_environment(), cf.ProcessPoolExecutor(n_processes, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                                                           initargs=(self,)) as pool:
            try:
                for k, chunk in enumerate(chunks):
                    while len(pending) >= n_processes:                                      # one block in flight for each process
                        _collect(pending, parts)
                    if check is not None:
                        check()
                    pending[pool.submit(_run_task, (method, chunk))] = k
                while pending:
                    _collect(pending, parts)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        return parts


def _collect(pending: dict, parts: list) -> None:
    """Wait until one of the futures of ``pending`` (future -> number of its block) is done, and put its result into ``parts``."""
    done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
    for future in done:
        parts[pending.pop(future)] = future.result()


@contextlib.contextmanager
# =============================================================================
# 블록 7: 멀티프로세스 병렬 워커 풀 및 작업 분배기
# =============================================================================
def _worker_environment():
    """``WORKER_ENVIRONMENT`` in ``os.environ`` for the length of a ``with`` block (the processes that are spawned inside it inherit it),
    the variables put back as they were afterwards."""
    saved = {k: os.environ.get(k) for k in WORKER_ENVIRONMENT}
    os.environ.update(WORKER_ENVIRONMENT)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_WORKER = None
_LIMITS = None


def _init_worker(composer: Composer) -> None:
    """Start of a worker process: keep its copy of the composer; one BLAS thread for each process (the workers share the cores)."""
    global _WORKER, _LIMITS
    _WORKER = composer
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(limits=1)
    except ImportError:
        pass


def _run_task(task):
    method, chunk = task
    return getattr(_WORKER, method)(chunk)
