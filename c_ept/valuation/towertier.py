r"""The tower state before t0, the tier of a taken tower and the expected price of a predicted tower (stage S3 Task 8, design S1-5).

A tower is priced by its tier (``c_ept.valuation.flowprice.GROUPS``: tower_outer, tower_inner, tower_base, tower_nexus; the S1 column
``objectives/sub``, 0-3). M3 and M4 predict how many towers a team takes, not which tier, and the composer may know only the state at t0
(rule A3). Three pieces turn a predicted tower into a price:

    state   ``destroyed_before(s1, match, t_ms)``: (n, 2, 4) int64, ``[i, v, j]`` = the towers of tier j that team v (0 blue, 1 red)
            has lost in match ``match[i]`` strictly before ``t_ms[i]``. A tower counts when it has a tier (0-3) and a taker (blue or
            red); the victim is the other team, 1 - taker (``victim_counts`` picks that team's row).
            The test is t < t_ms: an event on the millisecond of t0 is not counted. The S2 views see events at t <= ref, so an objective
            exactly at t0 is in the head inputs but not in this state (a convention recorded in the Task 8 review). An objective that is no
            tower, a tower without a tier or without a taker is not counted (the rule of ``c_ept.domain.targets._counted`` plus a known
            tier, as in ``c_ept.valuation.flowprice.event_table``). Only events before ``t_ms`` enter the answer: two stores that differ
            only at or after ``t_ms`` give the same state (a test pins this). It reads ``matches/obj_off`` and
            ``objectives/{t, kind, sub, taker}`` and nothing else, and it is the only route by which the composer may learn the tower
            state (global constraint A3).

    law     ``TierModel``: a state-conditional distribution over the four tiers of a taken tower, a multinomial logistic regression
            (``LogisticRegression(C=1.0, max_iter=2000, tol=1e-8)``, lbfgs; ruling R20) fitted on the tower events of the FIT matches that
            have a known tier and taker (``tier_rows``: one row per such event). Features (``FEATURE_NAMES``): the destroyed counts by
            tier of the victim team before the event (4 columns of ``destroyed_before``), the taker side (0 blue, 1 red) and the game
            time in minutes. Every column is standardised with the mean and the population standard deviation of the training rows (a
            column with a single value keeps the scale 1). The target is the tier. The output is mixed with the uniform law over the
            four tiers,

                p = (1 - 1e-6) p_model + 1e-6 / 4        (ruling R13)

            so that a tier that never occurred in training still has the probability 1e-6 / 4 and every score is finite. ``n_iter_`` and
            ``converged_`` are recorded (ruling R20): the ConvergenceWarning of scikit-learn is captured, the fit has converged when none
            was raised and the iteration limit was not reached, and a fit that has not raises one warning that names TierModel. One
            training tier is a constant model (0 iterations, converged); two tiers give the logits [0, z] of scikit-learn's single row.

    price   ``expected_tower_price(prob, prices)`` = sum over the four tiers of p x the price of the tier's group (a linear map: rows of
            probabilities give the expected price of a tower). Only the four tower groups of the price table are read; one that is missing
            or not a finite number is refused. The price of a tier is the ``value`` that ``c_ept.valuation.value.load_prices`` reads from
            the price table of the S3a run: the flow price where the run used it (``used == 'flow'``: the pre-trend test adopted it and the
            flow fit is estimable), otherwise the immediate value; ``tower_base`` and ``tower_nexus`` are always the immediate value
            (``c_ept.valuation.flowprice.FIXED_IMMEDIATE``, design 8.1). This module only reads that value and does not choose between
            the two.

The state the law is fitted on is the state at the event (towers destroyed before the tower), the state it is asked about is whatever the
caller passes: the composer passes the state at t0, since nothing later is known (S1-5). A tower predicted for the minutes after t0 is
therefore conditioned on a state that can be older than the tower itself by up to the evaluation horizon; that is the design of S1-5, not
something this module corrects.

E6. ``TierModel.fit`` calls ``check()`` (the E6 guard; the runner passes ``c_ept.workflows.run_s1.require_no_strata``) once, after the roles
have been checked and before any array is read, exactly as ``B0Heads.fit`` does. ``run_s1`` is not imported here at all. Holdout roles
(TEST, EXTERNAL) and roles that stage S3 does not use are refused with a ValueError before any read, also by ``tier_rows``.

Files. ``TierModel.save(dir)`` writes ``tower_tier.npz`` (classes, coefficients, intercepts, mean and scale; a stored zip without time
stamps, so the same model gives the same bytes) and ``tower_tier.json`` (schema, SHA-256 of the npz, features, groups, classes, roles,
counts and the facts of the fit). No pickle. ``load`` checks the schema, the features, the groups, the SHA-256 and the shapes. Messages
carry counts, role names and array names, never a match key; ``matches/key`` is not read.

이 모듈은 질문 시각 전의 포탑 상태, 가져간 포탑의 단계 분포, 예측한 포탑의 기대 가격을 계산한다. 단계 j는 0 기준 번호(0 외곽, 1 내부, 2 억제기 포탑,
3 넥서스 포탑; 설계서 8.1절)이고 오브젝트 표의 ``sub`` 열에서 온다. 가져간 팀 taker는 0 파랑, 1 빨강이며 포탑을 잃은 팀은 1 − taker이다. 계획의 전역 제약 인과(A3)에
따라 합성은 질문 시각 t₀ 앞의 포탑 상태를 ``destroyed_before``로만 얻고, 단계 분포는 사건 시점의 상태로 맞춘 뒤 호출자가 주는 상태로 묻는다(합성은 t₀의 상태를 준다).

식 목록
  (TIER-1) destroyed_before: 팀 v가 시각 t_i 앞에 잃은 j단계 포탑의 수(엄격한 t_e < t_i)
  (TIER-2) victim_counts: 행마다 잃은 팀 1 − taker의 단계별 파괴 수
  (TIER-3) TierModel.fit: 특징 x = (파괴 수 4열, taker, 분)와 표준화 x̃ = (x − mean) / scale
  (TIER-4) TierModel.prob: 단계 확률 p_{i,j} = (1 − ν) · p_model(i, j) + ν / 4
  (TIER-5) expected_tower_price: 기대 가격 exp_price_i = Σ_j prob_{i,j} · price_j

공통 가정과 검사. 검사하는 것: 역할은 FIT, STOP, DEV 가운데 하나 이상이고 그 밖은 배열을 읽기 전에 거부한다; taker는 0 또는 1이다; 개수와 시각은 유한한 비음수이다; 네 포탑 묶음의
가격은 유한하다; 저장 파일의 SHA-256, 스키마, 모양이 맞다. 검사하지 않는 것: ``expected_tower_price``에 주는 확률의 합이 1인 것(유한한 비음수만 검사); 포탑 사건 표가 시각 순인
것(경기마다 정렬해 쓴다); 가격표 값의 의미. 경기별 가중은 없고 포탑 사건 하나가 학습 행 하나이다.

가격의 종류. 단계 가격은 가격표(PRICES.json)의 ``value``이며, 앞 추세 시험을 통과하고 흐름 쪽을 추정할 수 있는 묶음은 흐름 가격, 그 밖에는 즉시값이다. 억제기 포탑(tower_base)과 넥서스 포탑(tower_nexus)은
늘 즉시값이다(설계서 8.1절). 이 모듈은 둘 가운데 어느 쪽인지 고르지 않고 가격표의 값을 그대로 읽는다.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md Task 8 and rulings R13 (every categorical output is mixed with the uniform law) and
#      R20 (tolerance 1e-8, an honest convergence flag) of the S3 ledger.
# Ref: docs/superpowers/specs/2026-10-20-c-ept-design.md section 8.1 (S1-5: the tier column `sub`, the state-conditional tier law that
#      prices the towers M3 and M4 predict) and section 11, item 16 (`objectives/sub`: 0 outer, 1 inner, 2 inhibitor tower, 3 nexus tower).
# Ref: Hastie, T., Tibshirani, R. and Friedman, J. (2009), "The Elements of Statistical Learning", 2nd ed., Springer, section 4.4
#      (multinomial logistic regression, softmax, L2 penalty).
# Ref: Liu, D. C. and Nocedal, J. (1989), "On the limited memory BFGS method for large scale optimization", Mathematical Programming 45:
#      503-528 (the lbfgs solver).
# Ref: Pedregosa, F. et al. (2011), "Scikit-learn: Machine learning in Python", JMLR 12: 2825-2830 (LogisticRegression).
# Ref: Chen, S. F. and Goodman, J. (1999), "An empirical study of smoothing techniques for language modeling", Computer Speech and Language
#      13(4): 359-394 (interpolation with the uniform distribution, so that no outcome has probability zero).
from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import warnings
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from c_ept import config as C
from c_ept.valuation import flowprice
from c_ept.data import store_core

# -----------------------------------------------------------------------------
# 기본 상수 및 타워 티어(Tier) 정의
# -----------------------------------------------------------------------------
# 진영 정의: 0 = 블루팀(Blue), 1 = 레드팀(Red)
BLUE, RED = 0, 1

# 포탑 티어 개수 (0~3 총 4종)
N_TIERS = 4

# 타워 티어 그룹명 (0: 외곽 Outer, 1: 내부 Inner, 2: 억제기 Base, 3: 넥서스 Nexus)
TIER_GROUPS = ('tower_outer', 'tower_inner', 'tower_base', 'tower_nexus')

# 티어 예측 모델에 사용되는 6개 피처:
# [피해 팀의 외곽 파괴수, 내부 파괴수, 억제기 파괴수, 넥서스 파괴수, 타워 획득 진영(taker), 경과 시간(분)]
FEATURE_NAMES = ('destroyed_outer', 'destroyed_inner', 'destroyed_base', 'destroyed_nexus', 'taker', 'minute')
N_FEATURES = len(FEATURE_NAMES)

# 소프트맥스 확률 스무딩 가중치 (학습 시 등장하지 않은 클래스의 확률이 0이 되는 것을 방지하기 위해 1e-6 균등 분포 혼합)
MIX = 1e-6

# 로지스틱 회귀 최적화 하이퍼파라미터
C_LOGISTIC = 1.0  # L2 규제 강도의 역수
MAX_ITER = 2000   # 최대 최적화 반복 횟수
TOL = 1e-8        # L-BFGS 수렴 판정 공차(Tolerance)

SCHEMA = 'tower_tier_v1'
NPZ_NAME, JSON_NAME = 'tower_tier.npz', 'tower_tier.json'

_TOWER = C.O_KINDS.index('tower')
_FIRST_TOWER = flowprice.GROUPS.index('tower_outer')
if flowprice.GROUPS[_FIRST_TOWER:_FIRST_TOWER + N_TIERS] != TIER_GROUPS:
    raise RuntimeError('the four tower groups of flowprice.GROUPS must be tower_outer, tower_inner, tower_base, tower_nexus, in this order')
_ARRAY_KEYS = ('classes', 'coef', 'intercept', 'mean', 'scale')


# -----------------------------------------------------------------------------
# 1. S1 저장소 오브젝트 테이블 및 티어 파싱
# -----------------------------------------------------------------------------
def _objectives(s1) -> tuple:
    """``(obj_off, t, kind, sub, taker)`` of the S1 store: the offsets (n_matches + 1,) int64 and the four objective columns as arrays.
    ValueError (array names only) when the offsets do not cut the table."""
    off = np.asarray(s1.array('matches/obj_off'))
    if off.ndim != 1 or off.size < 1 or off.dtype.kind not in 'iu':
        raise ValueError('matches/obj_off must be a 1-d integer array with one entry more than there are matches')
    off = off.astype(np.int64)
    columns = tuple(np.asarray(s1.array(name)) for name in ('objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker'))
    if off[0] != 0 or (np.diff(off) < 0).any() or any(c.shape != (int(off[-1]),) for c in columns):
        raise ValueError('matches/obj_off does not cut the objective table: it must start at 0, never go down and end at the number of '
                         'rows of objectives/t, kind, sub and taker')
    return (off, *columns)


# (TIER-1)의 tier(e)를 정한다.
def _tier_of(kind, sub) -> np.ndarray:
    """(n,) int64: the tier 0-3 of a tower that has one, -1 for everything else (another kind, a tower without a tier)."""
    tier = flowprice.group_of(kind, sub).astype(np.int64) - _FIRST_TOWER
    return np.where((tier >= 0) & (tier < N_TIERS), tier, -1)


def _integers(name: str, values) -> np.ndarray:
    """``values`` as an int64 1-d array. TypeError for a dtype that is not an integer (a bool, a float), ValueError for another shape."""
    a = np.asarray(values)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must be an integer array, not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'{name} must be 1-d, not shape {a.shape}')
    return a.astype(np.int64)


def _taker_array(taker) -> np.ndarray:
    """(n,) int64 takers: every entry must be 0 (blue) or 1 (red). TypeError for text, ValueError for another shape or value."""
    a = np.asarray(taker)
    if a.dtype.kind not in 'biuf':
        raise TypeError(f'taker must be numbers (0 blue, 1 red), not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'taker must be 1-d, not shape {a.shape}')
    if not np.isin(a, (BLUE, RED)).all():
        raise ValueError('taker must be 0 (blue) or 1 (red)')
    return a.astype(np.int64)


# -----------------------------------------------------------------------------
# 2. 질문 시각 t0 이전의 기파괴 타워 상태 집계
# -----------------------------------------------------------------------------
def destroyed_before(s1, match, t_ms) -> np.ndarray:
    """(n, 2, 4) int64: ``[i, v, j]`` is the number of towers of tier j (0 outer, 1 inner, 2 base, 3 nexus) lost by team v (0 blue, 1 red)
    in match row ``match[i]`` with an objective time t < ``t_ms[i]`` (strictly: an event on the millisecond ``t_ms`` is not counted).

    ``match`` (integer match rows, as in ``frame_rows/match``) and ``t_ms`` (integer ms, as in ``frame_rows/t0``) have the same length n.
    A tower is counted when it has a tier and a taker; the victim is the team that did not take it. The towers of a match are sorted by time
    and the queries of that match are answered with one ``searchsorted`` over the running counts, so the cost is one pass over the
    objectives of the matches asked for. TypeError for non-integer inputs, ValueError for different lengths, a match row outside the
    store or a negative time.

    수식
    ----
    (TIER-1)  destroyed[i, v, j] = #{ e ∈ 경기 match(i) : tier(e) = j, taker(e) ∈ {0, 1}, 1 − taker(e) = v, t_e < t_i }      (int64)

      기호     | 뜻                                          | 단위 | 범위                | 코드 변수
      i        | 질문 행(0부터)                              | —    | 0, …, n − 1         | 반환 배열의 첫 축
      match(i) | 질문 행의 경기(S1 저장소의 0 기준 행)       | —    | 0, …, 경기 수 − 1   | match[i]
      t_i      | 질문 시각                                   | ms   | 정수, ≥ 0           | t_ms[i]
      e        | 그 경기의 포탑 사건                         | —    | 단계와 taker가 있는 것 | 정렬된 objectives 행
      t_e      | 사건의 시각                                 | ms   | 정수                | objectives/t
      v        | 포탑을 잃은 팀                              | —    | 0 파랑, 1 빨강      | 반환 배열의 둘째 축
      j        | 단계                                        | —    | 0, …, 3             | 반환 배열의 셋째 축
      tier(e), taker(e) | 사건의 단계, 가져간 팀              | —    | 0–3; 0 또는 1       | tier, who

    해석
    ----
    식 (TIER-1)은 질문 시각 t_i 앞에 팀 v가 잃은 j단계 포탑의 개수이다. 부등호를 엄격하게 둔 것은 계획 Task 8의 규칙(사건 전의 파괴 수)이며, 학습 행에서
    사건 자신이 그 사건의 특징에 들어가지 않게 한다. 기준 시각 이하의 사건을 보는 S2 보기(S1–S2 판정 R12-c)와 달리 t_i와 같은 ms의 포탑은 이 상태에 들지 않으며,
    S3 장부의 Task 8 검토가 이를 알려진 경계 규약으로 기록하였다. 단계나 가져간 팀을 모르는 포탑과 포탑이 아닌 오브젝트는 세지 않는다. 질문 행이 없으면 모양만 맞는
    0 배열이고, 포탑 사건이 없는 경기의 행은 0이다.

    근거
    ----
    S3 계획 Task 8(사건 전의 파괴 수), S3 계획 전역 제약 인과(A3)(포탑 상태의 유일한 경로); 설계서 8.1절(S1-5); 개수 규칙은 ``targets._counted``와 ``flowprice.event_table``.
    """
    match, t_ms = _integers('match', match), _integers('t_ms', t_ms)
    if match.shape != t_ms.shape:
        raise ValueError(f'match and t_ms must have the same length, not {match.shape} and {t_ms.shape}')
    off, t, kind, sub, taker = _objectives(s1)
    n_matches = len(off) - 1
    if match.size and (match.min() < 0 or match.max() >= n_matches):
        raise ValueError(f'match numbers must lie in 0 .. {n_matches - 1} (the matches of the store)')
    if t_ms.size and t_ms.min() < 0:
        raise ValueError('t_ms must not be negative')

    out = np.zeros((len(match), 2, N_TIERS), dtype=np.int64)
    if not match.size:
        return out

    # 매치별로 쿼리 행들을 그룹화하여 배치 처리
    order = np.argsort(match, kind='stable')
    by_match = match[order]
    starts = np.flatnonzero(np.r_[True, by_match[1:] != by_match[:-1]])
    stops = np.r_[starts[1:], len(match)]
    cells = np.eye(2 * N_TIERS, dtype=np.int64)

    for a, b in zip(starts.tolist(), stops.tolist()):
        m = int(by_match[a])
        lo, hi = int(off[m]), int(off[m + 1])
        if hi == lo:
            continue
        tier, who = _tier_of(kind[lo:hi], sub[lo:hi]), taker[lo:hi].astype(np.int64)
        counted = (tier >= 0) & ((who == BLUE) | (who == RED))
        if not counted.any():
            continue

        when = t[lo:hi].astype(np.int64)[counted]
        # (TIER-1)의 (v, j): 칸 번호 4v + j를 단위 행렬 cells로 펼쳐 누적합하면 t_e < t_i인 사건의 (v, j)별 개수가 된다.
        # 피해 팀 v = 1 - who (타워를 잃은 진영)와 티어 번호(0~3)를 결합하여 고유 셀 인덱스(4*v + tier) 계산
        cell = (1 - who[counted]) * N_TIERS + tier[counted]
        first = np.argsort(when, kind='stable')
        when, cell = when[first], cell[first]

        # 시간순 누적합(running cumulative counts) 행렬 구축
        running = np.zeros((len(when) + 1, 2 * N_TIERS), dtype=np.int64)
        running[1:] = np.cumsum(cells[cell], axis=0)
        rows = order[a:b]
        # searchsorted('left')를 적용하여 t < t_ms (엄격히 이전 시점)의 누적 파괴 상태를 빠르게 슬라이싱
        out[rows] = running[np.searchsorted(when, t_ms[rows], side='left')].reshape(-1, 2, N_TIERS)

    return out


def victim_counts(destroyed, taker) -> np.ndarray:
    """(n, 4) int64: for each row the destroyed counts of the team that lost the tower, 1 - taker, out of ``destroyed`` (n, 2, 4), the
    state of both teams that ``destroyed_before`` gives. ``taker`` is 0 (blue took the tower) or 1 (red took it), one per row.

    수식
    ----
    (TIER-2)  lost_{i,j} = destroyed[i, 1 − taker_i, j]      (int64)

      기호    | 뜻                                      | 단위 | 범위            | 코드 변수
      destroyed | 두 팀의 단계별 파괴 수 (TIER-1)        | 개   | 0, 1, …         | destroyed
      taker_i | 행 i의 가져간 팀                        | —    | 0 또는 1        | taker
      lost_{i,j} | 잃은 팀의 단계별 파괴 수             | 개   | 0, 1, …         | 반환 (n, 4)

    해석
    ----
    식 (TIER-2)는 (TIER-1)의 두 팀 가운데 포탑을 잃은 팀 1 − taker_i의 행만 고르며, 이 값이 계획 Task 8의 특징인 당한 팀의 단계별 파괴 수이다.

    근거
    ----
    S3 계획 Task 8(특징은 당한 팀의 단계별 파괴 수); 설계서 8.1절(S1-5).
    """
    D = np.asarray(destroyed)
    if D.ndim != 3 or D.shape[1:] != (2, N_TIERS):
        raise ValueError(f'destroyed must have shape (n, 2, {N_TIERS}), not {D.shape}')
    who = _taker_array(taker)
    if who.shape != (len(D),):
        raise ValueError(f'taker must have one entry for each of the {len(D)} rows, not shape {who.shape}')
    return D[np.arange(len(D)), 1 - who].astype(np.int64)


# -----------------------------------------------------------------------------
# 3. 학습용 타워 사건 데이터셋 구성
# -----------------------------------------------------------------------------
class TierRows(NamedTuple):
    """The tower events of the matches of some roles that have a known tier and taker, in the order of the store, with their state."""
    match: np.ndarray        # 매치 행 인덱스
    t_ms: np.ndarray         # 타워 파괴 시각 (밀리초, ms)
    taker: np.ndarray        # 타워 파괴 팀 (0: Blue, 1: Red)
    tier: np.ndarray         # 파괴된 타워 티어 (0~3: Outer, Inner, Base, Nexus - 모델 타깃)
    destroyed: np.ndarray    # 사건 직전 피해 팀의 티어별 기파괴 수 (n, 4)
    n_matches: int           # 학습 대상 매치 수
    n_tower_events: int      # 전체 타워 파괴 사건 수
    n_no_tier: int           # 티어 미상으로 제외된 사건 수
    n_no_taker: int          # 획득팀 미상으로 제외된 사건 수


def tier_rows(s1, roles=('FIT',)) -> TierRows:
    """The rows the tier law is fitted on (and scored on): the tower events of the matches whose role is in ``roles`` that have a tier and
    a taker, with the destroyed counts of the victim team before each event. ``roles`` is a name or names out of FIT, STOP, DEV; ValueError
    for a holdout role (TEST, EXTERNAL), a role that stage S3 does not use, an unknown name or no role, before any array is read.

    사건마다 사건 시점의 상태를 (TIER-1)과 (TIER-2)로 붙이므로 그 상태에는 사건 자신과 같은 ms의 사건이 들지 않는다. 단계나 가져간 팀을 모르는 사건은 학습 행에서 빼고 그 수를 센다."""
    codes = flowprice._role_codes(roles)
    role = np.asarray(s1.array('matches/role'))
    off, t, kind, sub, taker = _objectives(s1)
    if len(role) != len(off) - 1:
        raise ValueError(f'matches/role has {len(role)} entries but matches/obj_off cuts {len(off) - 1} matches')

    in_role = np.isin(role, codes)
    match_of = np.repeat(np.arange(len(role), dtype=np.int64), np.diff(off))
    chosen = in_role[match_of]
    tier, who = _tier_of(kind, sub), taker.astype(np.int64)
    known_taker = (who == BLUE) | (who == RED)
    is_tower = chosen & (kind == _TOWER)
    rows = np.flatnonzero(chosen & (tier >= 0) & known_taker)

    match_u, t_u, who_u = match_of[rows], t[rows].astype(np.int64), who[rows]
    destroyed = victim_counts(destroyed_before(s1, match_u, t_u), who_u)

    return TierRows(
        match=match_u, t_ms=t_u, taker=who_u, tier=tier[rows], destroyed=destroyed,
        n_matches=int(in_role.sum()), n_tower_events=int(is_tower.sum()),
        n_no_tier=int((is_tower & (tier < 0)).sum()),
        n_no_taker=int((is_tower & (tier >= 0) & ~known_taker).sum())
    )


# -----------------------------------------------------------------------------
# 4. 피처 매트릭스 구성 및 표준화 (Design Matrix)
# -----------------------------------------------------------------------------
def _design(destroyed_victim, taker, t_s) -> np.ndarray:
    """(n, 6) float64, the raw features in the order of ``FEATURE_NAMES``: the destroyed counts of the victim by tier, the taker side and
    the game time in minutes (``t_s`` is seconds). The one place that makes the columns, for the fit and for ``prob``. ValueError for the
    wrong shape, a taker that is not 0 or 1, a count or a time that is negative or not finite; TypeError for text.
    (TIER-3)의 특징 행 x를 만든다."""
    D = np.asarray(destroyed_victim)
    if D.dtype.kind not in 'iuf':
        raise TypeError(f'destroyed_victim must be numbers, not {D.dtype}')
    D = D.astype(np.float64)
    if D.ndim != 2 or D.shape[1] != N_TIERS:
        raise ValueError(f'destroyed_victim must have shape (n, {N_TIERS}), not {D.shape}')
    who = _taker_array(taker)
    t = np.asarray(t_s)
    if t.dtype.kind not in 'iuf':
        raise TypeError(f't_s must be numbers (seconds), not {t.dtype}')
    t = t.astype(np.float64)
    if who.shape != (len(D),) or t.shape != (len(D),):
        raise ValueError(f'taker and t_s must have one entry for each of the {len(D)} rows, not shapes {who.shape} and {t.shape}')
    if not np.isfinite(D).all() or (D < 0).any():
        raise ValueError('destroyed_victim must hold finite counts that are not negative')
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('t_s must hold finite times that are not negative')
    return np.column_stack([D, who.astype(np.float64), t / 60.0])


def _column_stats(X: np.ndarray) -> tuple:
    """(mean, scale) of the columns of ``X``: the mean and the population standard deviation; a column with one value gets that value as
    its mean and the scale 1, so that it standardises to zeros and a different value later cannot blow up.
    (TIER-3)의 mean과 scale을 계산한다."""
    constant = X.min(axis=0) == X.max(axis=0)
    mean = np.where(constant, X.min(axis=0), X.mean(axis=0))
    scale = np.where(constant, 1.0, X.std(axis=0))
    return mean, scale


# -----------------------------------------------------------------------------
# 5. 모델 가중치 직렬화 및 수렴 보조 헬퍼
# -----------------------------------------------------------------------------
def _fit_recording(estimator, X, y) -> bool:
    """Fit the scikit-learn ``estimator`` and say whether it raised a ConvergenceWarning. That warning is captured here (``fit`` raises one
    of its own that names TierModel); any other warning goes on as it was."""
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(X, y)
    fired = False
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            fired = True
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    return fired


def _warn_not_converged(n_iter: int, fired: bool) -> None:
    from sklearn.exceptions import ConvergenceWarning
    why = 'scikit-learn raised a ConvergenceWarning' if fired else 'the iteration limit was reached'
    warnings.warn(f'TierModel: the optimiser did not converge ({why}; {n_iter} iterations, the limit is {MAX_ITER})', ConvergenceWarning,
                  stacklevel=3)


def _blank_info() -> dict:
    return {
        'fit_roles': [], 'n_matches': 0, 'n_rows': 0, 'n_by_tier': [0] * N_TIERS,
        'n_tower_events': 0, 'n_no_tier': 0, 'n_no_taker': 0, 'n_iter': 0,
        'converged': True, 'sklearn': None
    }


def _checked_info(info) -> dict:
    """A plain-JSON copy of the ``info`` of a model; ValueError unless it has exactly the keys of ``_blank_info``."""
    if not isinstance(info, Mapping) or set(info) != set(_blank_info()):
        raise ValueError(f'info must be a mapping with exactly the keys {", ".join(_blank_info())}')
    return json.loads(json.dumps(info))


def _npz_bytes(arrays: Mapping) -> bytes:
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


_COUNT_KEYS = ('n_matches', 'n_rows', 'n_tower_events', 'n_no_tier', 'n_no_taker')


def _is_whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _info_of(doc: Mapping) -> dict:
    """The ``info`` of a model out of the counts, the fit facts and the roles of a saved JSON; ValueError for anything malformed."""
    counts, fit, roles = doc.get('counts'), doc.get('fit'), doc.get('fit_roles')
    if not (isinstance(counts, dict) and isinstance(fit, dict) and isinstance(roles, list) and all(r in C.ROLES for r in roles)):
        raise ValueError(f'{JSON_NAME}: counts and fit must be objects and fit_roles a list of role names')
    by_tier = counts.get('n_by_tier')
    whole = [counts.get(k) for k in _COUNT_KEYS] + [fit.get('n_iter')]
    if not (isinstance(by_tier, list) and len(by_tier) == N_TIERS and all(map(_is_whole, by_tier)) and all(map(_is_whole, whole))):
        raise ValueError(f'{JSON_NAME}: the counts and n_iter must be whole numbers ({N_TIERS} of them for n_by_tier)')
    if not isinstance(fit.get('converged'), bool):
        raise ValueError(f'{JSON_NAME}: fit.converged must be true or false')
    # 저장 파일의 설정이 이 모듈의 상수와 같아야 한다: prob는 이 모듈의 MIX를 읽으므로 다른 설정으로 만든 모형은 거부한다.
    for key, want in (('C', C_LOGISTIC), ('max_iter', MAX_ITER), ('tol', TOL), ('mix', MIX)):
        if fit.get(key) != want or isinstance(fit.get(key), bool):
            raise ValueError(f'{JSON_NAME}: fit.{key} is {fit.get(key)!r} but this module uses {want!r}: '
                             'the model was made with other settings')
    return {
        'fit_roles': roles, **{k: counts[k] for k in _COUNT_KEYS}, 'n_by_tier': by_tier,
        'n_iter': fit['n_iter'], 'converged': fit['converged'], 'sklearn': fit.get('sklearn')
    }


# -----------------------------------------------------------------------------
# 6. 타워 티어 조건부 확률 모델 클래스 (TierModel)
# -----------------------------------------------------------------------------
class TierModel:
    """The distribution of the tier of a taken tower given the state: ``fit`` it on the FIT matches, ``prob`` to ask it, ``save`` and
    ``load`` to keep it, ``summary`` for the facts of the fit.

    The parameters are on the standardised features: ``classes_`` (K,) the tiers that occurred in training (sorted), ``coef_std_``
    (K, 6) and ``intercept_std_`` (K,) the logits of those tiers (softmax over them), ``mean_`` and ``scale_`` (6,) the standardisation;
    all of them are copies and read-only. ``n_iter_`` and ``converged_`` are the facts of ruling R20. Build a model by hand with the
    same arguments (``info`` is the dict of ``summary`` without the classes)."""

    def __init__(self, classes, coef, intercept, mean, scale, info=None):
        classes = np.asarray(classes)
        if classes.size and classes.dtype.kind not in 'iu':
            raise TypeError(f'classes must be integers, not {classes.dtype}')
        classes = classes.astype(np.int64)
        if classes.ndim != 1 or classes.size == 0 or classes.min() < 0 or classes.max() >= N_TIERS or (np.diff(classes) <= 0).any():
            raise ValueError(f'classes must be 1-d, not empty and strictly increasing within 0 .. {N_TIERS - 1}, not {classes.tolist()}')
        k = len(classes)
        arrays = {
            'coef': (coef, (k, N_FEATURES)), 'intercept': (intercept, (k,)),
            'mean': (mean, (N_FEATURES,)), 'scale': (scale, (N_FEATURES,))
        }
        checked = {}
        for name, (values, shape) in arrays.items():
            a = np.array(values, dtype=np.float64)
            if a.shape != shape:
                raise ValueError(f'{name} must have shape {shape} ({k} classes, {N_FEATURES} features), not {a.shape}')
            if not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite')
            checked[name] = a
        if (checked['scale'] <= 0).any():
            raise ValueError('scale must be positive')
        for a in (classes, *checked.values()):
            a.setflags(write=False)
        self.classes_, self.coef_std_, self.intercept_std_ = classes, checked['coef'], checked['intercept']
        self.mean_, self.scale_ = checked['mean'], checked['scale']
        self._info = _blank_info() if info is None else _checked_info(info)

    @classmethod
    def fit(cls, s1, roles=('FIT',), check=None) -> 'TierModel':
        """Fit the tier law on the tower events (known tier and taker) of the matches whose role is in ``roles`` (default FIT).

        ``check``: a function without arguments (the E6 guard), called once before anything is read. ValueError for a holdout role or a
        role that stage S3 does not use (before ``check`` and before any read), and for roles without a single usable tower event.

        수식
        ----
        (TIER-3)  x_i = (lost_{i,0}, lost_{i,1}, lost_{i,2}, lost_{i,3}, taker_i, t_i / 60000)
                  mean_k = (1/n) Σ_i x_{ik};   scale_k = √((1/n) Σ_i (x_{ik} − mean_k)²);
                  x̃_{ik} = (x_{ik} − mean_k) / scale_k       (한 값뿐인 열은 mean_k = 그 값, scale_k = 1)
                  모형 계수 = scikit-learn LogisticRegression(C = χ, max_iter, tol)이 (x̃_i, tier_i)에 대해 낸 값(학습 단계가 둘 이상일 때)

          기호        | 뜻                                          | 단위 | 범위                     | 코드 변수
          i, n        | 학습 행(포탑 사건)의 번호(0부터), 행의 수   | —    | 0, …, n − 1; n ≥ 1       | tier_rows의 행
          lost_{i,j}  | 사건 앞에 잃은 팀이 잃은 j단계 포탑 수      | 개   | 0, 1, …                  | rows.destroyed
          taker_i     | 가져간 팀                                   | —    | 0 또는 1                 | rows.taker
          t_i         | 사건의 시각(분으로 바꾸어 넣는다)           | ms   | 정수, ≥ 0                | rows.t_ms
          x_{ik}      | 특징 k(0–5, FEATURE_NAMES 순서)             | 개, —, 분 | ≥ 0                 | _design의 열
          mean_k, scale_k | 열 k의 평균, 모표준편차                 | 열마다 | 유한, scale_k > 0       | mean, scale
          x̃_{ik}      | 표준화한 특징                               | —    | 유한                     | Z
          χ           | 로지스틱 벌점의 역수(C_LOGISTIC)            | —    | 상수                     | C_LOGISTIC

        해석
        ----
        식 (TIER-3)은 사건 시점의 상태를 특징으로 삼아 단계를 맞추는 다항 로지스틱 회귀의 입력을 만들며, 표준화의 평균과 모표준편차는 학습 행에서만 구해 (TIER-4)가
        그대로 쓴다. 값이 하나뿐인 열은 scale을 1로 두어, 나중에 다른 값이 들어와도 표준화한 값이 폭발하지 않게 한다. 벌점의 형태(L2), 풀이기(lbfgs), 다항 처리는
        scikit-learn의 기본값이고, 이 모듈이 정하는 것은 χ, max_iter, tol뿐이다. 학습 단계가 하나뿐이면 계수가 모두 0인 상수 모형이다. 둘이면 scikit-learn이 내는
        계수 한 줄 앞에 0인 줄을 더해, 두 단계 위의 소프트맥스가 그 로지스틱 회귀와 같게 한다.

        근거
        ----
        S3 계획 Task 8(사건 전 상태, 가져간 팀, 경기 시각); S3 판정 R20(tol, max_iter, 수렴 표시); 설계서 8.1절(S1-5).
        """
        codes = flowprice._role_codes(roles)
        if check is not None:
            check()
        names = [C.ROLES[c] for c in codes.tolist()]
        rows = tier_rows(s1, names)
        n = len(rows.tier)
        if n == 0:
            raise ValueError(f'no tower event with a known tier and taker in the matches of {", ".join(names)} ({rows.n_matches} matches, '
                             f'{rows.n_tower_events} tower events): the tier law cannot be fitted')

        import sklearn
        from sklearn.linear_model import LogisticRegression

        # 피처 매트릭스 구성 및 z-score 표준화 (학습 세트의 평균과 표준편차 산출)
        X = _design(rows.destroyed, rows.taker, rows.t_ms / 1000.0)
        mean, scale = _column_stats(X)
        # (TIER-3)의 x̃: 평균과 모표준편차는 학습 행에서만 구해 prob가 같은 값으로 표준화한다.
        Z = (X - mean) / scale
        classes = np.unique(rows.tier)

        # 학습 단계가 하나뿐이면 상수 모형이다(LogisticRegression은 두 단계 이상을 요구한다).
        if len(classes) == 1:
            coef, intercept, n_iter, converged = np.zeros((1, N_FEATURES)), np.zeros(1), 0, True
        else:
            estimator = LogisticRegression(C=C_LOGISTIC, max_iter=MAX_ITER, tol=TOL)
            fired = _fit_recording(estimator, Z, rows.tier)
            n_iter = int(np.max(estimator.n_iter_))
            converged = (not fired) and n_iter < MAX_ITER
            if not converged:
                _warn_not_converged(n_iter, fired)
            coef, intercept = np.asarray(estimator.coef_, dtype=np.float64), np.asarray(estimator.intercept_, dtype=np.float64)
            # 두 단계이면 scikit-learn은 계수 한 줄(첫 단계 대비 둘째 단계의 로짓)만 내므로 0인 줄을 앞에 더한다.
            if coef.shape[0] == 1:
                coef, intercept = np.vstack([np.zeros(N_FEATURES), coef[0]]), np.array([0.0, intercept[0]])

        info = {
            'fit_roles': names, 'n_matches': rows.n_matches, 'n_rows': n,
            'n_by_tier': np.bincount(rows.tier, minlength=N_TIERS).tolist(),
            'n_tower_events': rows.n_tower_events, 'n_no_tier': rows.n_no_tier,
            'n_no_taker': rows.n_no_taker, 'n_iter': n_iter, 'converged': bool(converged),
            'sklearn': sklearn.__version__
        }
        return cls(classes, coef, intercept, mean, scale, info)

    def prob(self, destroyed_victim, taker, t_s) -> np.ndarray:
        """(n, 4) float64: the probability of each tier (outer, inner, base, nexus) for a tower taken by ``taker`` (0 blue, 1 red) at game
        time ``t_s`` (seconds) when the victim team has lost ``destroyed_victim`` (n, 4) towers of each tier (``victim_counts`` of
        ``destroyed_before``). Rows add up to 1; no tier has less than 1e-6 / 4 (ruling R13).

        수식
        ----
        (TIER-4)  η_{i,j} = Σ_k coef_{jk} · x̃_{ik} + intercept_j  (j ∈ A)
                  p_model(i, j) = exp(η_{i,j}) / Σ_{j′ ∈ A} exp(η_{i,j′})  (j ∈ A), 0 (j ∉ A)
                  p_{i,j} = (1 − ν) · p_model(i, j) + ν / C,   C = 4
                  x̃는 (TIER-3)의 mean_, scale_로 표준화한 특징

          기호       | 뜻                                      | 단위 | 범위                 | 코드 변수
          A          | 학습에 나온 단계의 집합(오름차순)       | —    | {0, …, 3}의 부분집합 | classes_
          coef_{jk}  | 단계 j의 표준화 특징 k 계수             | —    | 유한                 | coef_std_
          intercept_j| 단계 j의 절편                           | —    | 유한                 | intercept_std_
          η_{i,j}    | 행 i, 단계 j의 선형 예측량              | —    | 유한                 | logits
          p_model    | 학습 단계 위의 소프트맥스 확률          | —    | [0, 1]               | full
          ν          | 균등 분포의 비율                        | —    | 상수                 | MIX
          C          | 단계의 수                               | —    | 4                    | N_TIERS
          p_{i,j}    | 단계 확률                               | —    | ν / C 이상           | 반환 (n, 4)

        해석
        ----
        식 (TIER-4)는 학습에 나온 단계들 위에서 소프트맥스를 취한 뒤 균등 분포와 섞어, 학습에 없던 단계도 ν / C의 확률을 갖게 한다. 섞는 것은 모든 범주형 출력에 같은 규칙을
        쓰는 판정 R13이고, 점수가 유한해진다. 소프트맥스는 최댓값을 빼고 계산하며, 행의 합은 부동소수점 반올림 안에서 1이다. 입력은 (TIER-3)과 같은 검사를 거치고 계수와 표준화는 읽기
        전용 사본이다.

        근거
        ----
        S3 판정 R13(모든 범주형 출력은 균등 분포와 섞는다); S3 계획 Task 8(prob의 모양); 설계서 8.1절(S1-5).
        """
        X = _design(destroyed_victim, taker, t_s)
        logits = ((X - self.mean_) / self.scale_) @ self.coef_std_.T + self.intercept_std_

        # (TIER-4)의 소프트맥스: 학습 단계 A에서만 정규화하고 나머지 단계는 0으로 둔 뒤 균등 분포와 섞는다.
        # 수치 안정성을 위해 로짓의 최댓값을 뺀 후 소프트맥스 계산
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        full = np.zeros((len(X), N_TIERS))
        full[:, self.classes_] = e / e.sum(axis=1, keepdims=True)

        return (1.0 - MIX) * full + MIX / N_TIERS

    @property
    def n_iter_(self) -> int:
        """The iterations of the optimiser (0 for a constant model, which is not iterated)."""
        return self._info['n_iter']

    @property
    def converged_(self) -> bool:
        """Ruling R20: no ConvergenceWarning was raised and the iteration limit was not reached (True for a constant model)."""
        return self._info['converged']

    def summary(self) -> dict:
        """The facts of the fit as plain JSON values (a copy): ``classes`` (the tiers that occurred), ``fit_roles``, the counts of matches,
        training rows (``n_rows``, ``n_by_tier``) and tower events (``n_tower_events``, left out for want of a tier ``n_no_tier`` or a taker
        ``n_no_taker``), ``n_iter``, ``converged`` and the scikit-learn version."""
        return {'classes': self.classes_.tolist(), **json.loads(json.dumps(self._info))}

    def save(self, directory) -> Path:
        """Write ``tower_tier.npz`` and ``tower_tier.json`` into ``directory`` (created). FileExistsError if either file is there: a saved
        model is never written over."""
        out = Path(directory)
        if (out / NPZ_NAME).exists() or (out / JSON_NAME).exists():
            raise FileExistsError(f'{out} already holds a saved tower tier model: it is not written over')
        blob = _npz_bytes({
            'classes': self.classes_, 'coef': self.coef_std_, 'intercept': self.intercept_std_,
            'mean': self.mean_, 'scale': self.scale_
        })
        info = self._info
        doc = {
            'schema': SCHEMA, 'npz_sha256': hashlib.sha256(blob).hexdigest(), 'features': list(FEATURE_NAMES),
            'groups': list(TIER_GROUPS), 'classes': self.classes_.tolist(), 'fit_roles': list(info['fit_roles']),
            'counts': {k: info[k] for k in (*_COUNT_KEYS, 'n_by_tier')},
            'fit': {
                'C': C_LOGISTIC, 'max_iter': MAX_ITER, 'tol': TOL, 'mix': MIX,
                'n_iter': info['n_iter'], 'converged': info['converged'], 'sklearn': info['sklearn']
            }
        }
        out.mkdir(parents=True, exist_ok=True)
        store_core.write_atomic(out / NPZ_NAME, blob)
        store_core.write_atomic(out / JSON_NAME, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return out

    @classmethod
    def load(cls, directory) -> 'TierModel':
        """Read a model saved by ``save``. ValueError for another schema, other features or groups, an array file whose SHA-256 differs
        from the JSON's and arrays that do not fit; FileNotFoundError for a missing file."""
        folder = Path(directory)
        doc = json.loads((folder / JSON_NAME).read_text(encoding='utf-8'))
        if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
            raise ValueError(f'{folder / JSON_NAME}: the schema is {doc.get("schema") if isinstance(doc, dict) else doc!r}, not {SCHEMA!r}')
        if doc.get('features') != list(FEATURE_NAMES):
            raise ValueError(f'{JSON_NAME}: the features are not {list(FEATURE_NAMES)}')
        if doc.get('groups') != list(TIER_GROUPS):
            raise ValueError(f'{JSON_NAME}: the groups are not {list(TIER_GROUPS)}')
        if store_core.sha(folder / NPZ_NAME) != doc.get('npz_sha256'):
            raise ValueError(f'{folder / NPZ_NAME}: the array file has changed (its sha256 differs from the one in the JSON)')

        with np.load(folder / NPZ_NAME, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        if set(arrays) != set(_ARRAY_KEYS):
            raise ValueError(f'{NPZ_NAME} must hold exactly the arrays {", ".join(_ARRAY_KEYS)}')

        model = cls(arrays['classes'], arrays['coef'], arrays['intercept'], arrays['mean'], arrays['scale'], _info_of(doc))
        if doc.get('classes') != model.classes_.tolist():
            raise ValueError(f'{JSON_NAME}: the classes differ from those in {NPZ_NAME}')
        return model


# -----------------------------------------------------------------------------
# 7. 타워 기대 골드 가치 계산 (Expected Tower Price)
# -----------------------------------------------------------------------------
def _price(group: str, v) -> float:
    if v is None:
        raise ValueError(f'the price table has no price for the group {group}')
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ValueError(f'the price of {group} is not a finite number')
    return float(v)


def expected_tower_price(prob, prices) -> np.ndarray:
    """(n,) float64: sum over the tiers of ``prob[:, tier]`` x the price of the tier's group (``TIER_GROUPS``), for ``prob`` (n, 4) the
    probabilities of the tiers (``TierModel.prob``) and ``prices`` the {group: price} dict of ``c_ept.valuation.value.load_prices``
    (each price is the flow price the S3a run used for the group, or else its immediate value; ``tower_base`` and ``tower_nexus`` are
    always immediate). The weights need not add up to 1 (the result is linear in them). Only the four tower groups are read. ValueError
    for a group that is missing or whose price is not a finite number, and for weights that are not an (n, 4) array of finite,
    non-negative numbers; TypeError if ``prices`` is no mapping.

    수식
    ----
    (TIER-5)  exp_price_i = Σ_{j=0}^{3} prob_{i,j} · price_j      (float64)

      기호      | 뜻                                          | 단위 | 범위                    | 코드 변수
      prob_{i,j}| 행 i의 단계 j 확률(또는 가중)               | —    | 유한, ≥ 0               | prob
      price_j   | 단계 j의 묶음 가격(TIER_GROUPS[j])          | 골드 | 유한(부호는 검사 안 함) | prices[group]
      exp_price_i | 행 i의 기대 포탑 가격                     | 골드 | —                       | 반환 (n,)

    해석
    ----
    식 (TIER-5)는 단계 확률과 단계 가격의 내적으로 예측한 포탑 한 개의 기대 가격이다. 확률의 합이 1인지는 검사하지 않으므로 가중이 합 1이 아니어도 선형으로 계산하고,
    네 포탑 묶음 밖의 가격은 읽지 않는다.

    근거
    ----
    S3 계획 Task 8(단계 확률로 가중한 단계 가격); 설계서 8.1절(S1-5, 단계 가격이 포탑을 값매김); 가격표는 ``c_ept.valuation.value.load_prices``.
    """
    if not isinstance(prices, Mapping):
        raise TypeError('prices must be the {group: price} dict that value.load_prices returns')
    price = np.array([_price(group, prices.get(group)) for group in TIER_GROUPS], dtype=np.float64)
    P = np.asarray(prob)
    if P.dtype.kind not in 'iuf':
        raise TypeError(f'prob must be numbers, not {P.dtype}')
    P = P.astype(np.float64)
    if P.ndim != 2 or P.shape[1] != N_TIERS:
        raise ValueError(f'prob must have shape (n, {N_TIERS}), not {P.shape}')
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('prob must hold finite numbers that are not negative')
    return P @ price
