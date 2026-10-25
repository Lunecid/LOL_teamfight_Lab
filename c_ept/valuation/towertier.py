"""C-EPT 포탑 티어(Tower Tier) 확률 모델 및 기대 가치(Expected Price) 계산 모듈 (c_ept.valuation.towertier).

교전 평가 파이프라인(C-EPT)에서 교전 결과로 특정 팀이 획득하는 포탑(Tower/Turret)의 개수는 예측할 수 있지만,
구체적으로 어떤 티어(외곽 Outer, 내부 Inner, 억제기 Base, 넥서스 Nexus)의 포탑인지 직접 알 수 없습니다.
본 모듈은 교전 기준 시각(t0) 이전까지 상대 팀이 이미 잃은 포탑 상황과 게임 경과 시간을 바탕으로,
다음에 파괴될 포탑의 티어별 조건부 확률 분포를 예측하고 각 티어의 경제적 가치(Flow Price)를 곱해
타워 1개당 기대 골드 가치를 산출합니다.

주요 구성 요소:
  1. destroyed_before: 특정 시각(t_ms) 이전에 양 팀이 잃은 티어별 누적 파괴 상태 집계 (cumsum + 이진 탐색).
  2. victim_counts: 타워를 파괴당한 피해 팀(1 - taker)의 티어별 누적 파괴 수 슬라이싱.
  3. TierRows & tier_rows: 모델 학습용 타워 파괴 사건 데이터셋 구성 (사건 직전 피해 팀의 기파괴 상태 매핑).
  4. TierModel: 6개 피처(피해 팀 4개 티어 기파괴 수, 획득팀, 게임 시간) 기반 다항 로지스틱 회귀 모델.
     - L-BFGS 최적화 및 1e-6 균등 혼합 스무딩(Smoothing) 적용.
     - 가중치(npz)와 메타데이터(json)의 원자적 저장/로드 및 무결성 검증.
  5. expected_tower_price: 예측된 티어별 확률과 티어별 Flow Price의 선형 내적을 통한 기대 골드 가치 계산.
"""
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
    """S1 저장소에서 매치 경계 오프셋(matches/obj_off)과 오브젝트 타임라인 열(t, kind, sub, taker)을 로드하고 검증합니다."""
    off = np.asarray(s1.array('matches/obj_off'))
    if off.ndim != 1 or off.size < 1 or off.dtype.kind not in 'iu':
        raise ValueError('matches/obj_off must be a 1-d integer array with one entry more than there are matches')
    off = off.astype(np.int64)
    columns = tuple(np.asarray(s1.array(name)) for name in ('objectives/t', 'objectives/kind', 'objectives/sub', 'objectives/taker'))
    if off[0] != 0 or (np.diff(off) < 0).any() or any(c.shape != (int(off[-1]),) for c in columns):
        raise ValueError('matches/obj_off does not cut the objective table: it must start at 0, never go down and end at the number of '
                         'rows of objectives/t, kind, sub and taker')
    return (off, *columns)


def _tier_of(kind, sub) -> np.ndarray:
    """오브젝트 종류가 타워인 경우 0~3 티어 인덱스로 매핑하고, 타워가 아닌 오브젝트는 -1을 반환합니다."""
    tier = flowprice.group_of(kind, sub).astype(np.int64) - _FIRST_TOWER
    return np.where((tier >= 0) & (tier < N_TIERS), tier, -1)


def _integers(name: str, values) -> np.ndarray:
    """입력 배열이 1차원 정수형(int)인지 검증하고 int64 배열로 반환합니다."""
    a = np.asarray(values)
    if a.size == 0 and a.ndim <= 1:
        a = a.astype(np.int64)
    if a.dtype.kind not in 'iu':
        raise TypeError(f'{name} must be an integer array, not {a.dtype}')
    if a.ndim != 1:
        raise ValueError(f'{name} must be 1-d, not shape {a.shape}')
    return a.astype(np.int64)


def _taker_array(taker) -> np.ndarray:
    """타워 획득 팀(taker) 배열이 0(Blue) 또는 1(Red) 값만 갖는지 검증합니다."""
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
    """각 질문 행(match[i], t_ms[i])에 대해, 시각 t_ms[i]보다 엄격히 이전(t < t_ms[i])에
    양 팀(0: Blue, 1: Red)이 이미 잃은 티어별(0~3) 타워 누적 수를 (n, 2, 4) 정수 배열로 집계합니다.

    - 인과성 보장(Causality): t < t_ms 시점의 사건만 포함하여 미래 정보 누수를 방지합니다.
    - 고속 조회: 누적합(cumsum)과 이진 탐색(searchsorted)을 결합하여 매치당 한 번의 정렬로 O(N log K) 처리합니다.

    Returns:
        (n, 2, 4) int64 배열: [행 인덱스, 피해 팀(0: Blue, 1: Red), 티어(0~3)]
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
    """양 팀의 기파괴 타워 배열(n, 2, 4)에서 타워를 파괴당한 피해 팀(1 - taker)의
    4개 티어별 기파괴 수만 슬라이싱하여 (n, 4) 정수 배열로 반환합니다.
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
    """티어 예측 모델 학습에 사용되는 타워 파괴 사건 및 사건 직전 상태 튜플"""
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
    """지정된 데이터 분할 역할군(기본 FIT) 매치들에서 티어와 획득팀이 명확한 타워 사건들을 추출하고,
    각 사건 직전 시점의 피해 팀 타워 파괴 상태를 결합하여 TierRows 데이터셋을 생성합니다.
    """
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
    """티어 예측 모델의 입력 피처 6열을 생성합니다:
    [외곽파괴수, 내부파괴수, 억제기파괴수, 넥서스파괴수, taker(0: Blue, 1: Red), 게임경과시간(분)]
    """
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
    """피처 열별 평균(mean)과 표준편차(scale)를 계산합니다 (값이 일정한 열은 분모 0 방지를 위해 scale=1.0으로 고정)."""
    constant = X.min(axis=0) == X.max(axis=0)
    mean = np.where(constant, X.min(axis=0), X.mean(axis=0))
    scale = np.where(constant, 1.0, X.std(axis=0))
    return mean, scale


# -----------------------------------------------------------------------------
# 5. 모델 가중치 직렬화 및 수렴 보조 헬퍼
# -----------------------------------------------------------------------------
def _fit_recording(estimator, X, y) -> bool:
    """scikit-learn 분류기를 학습시키며 ConvergenceWarning 발생 여부를 감지합니다."""
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
    if not isinstance(info, Mapping) or set(info) != set(_blank_info()):
        raise ValueError(f'info must be a mapping with exactly the keys {", ".join(_blank_info())}')
    return json.loads(json.dumps(info))


def _npz_bytes(arrays: Mapping) -> bytes:
    """동일한 배열 데이터에 대해 항상 일정한 SHA-256 해시 바이트를 갖도록 타임스탬프를 제거한 결정론적 zip/npz 바이트 스트림을 생성합니다."""
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
    counts, fit, roles = doc.get('counts'), doc.get('fit'), doc.get('fit_roles')
    if not (isinstance(counts, dict) and isinstance(fit, dict) and isinstance(roles, list) and all(r in C.ROLES for r in roles)):
        raise ValueError(f'{JSON_NAME}: counts and fit must be objects and fit_roles a list of role names')
    by_tier = counts.get('n_by_tier')
    whole = [counts.get(k) for k in _COUNT_KEYS] + [fit.get('n_iter')]
    if not (isinstance(by_tier, list) and len(by_tier) == N_TIERS and all(map(_is_whole, by_tier)) and all(map(_is_whole, whole))):
        raise ValueError(f'{JSON_NAME}: the counts and n_iter must be whole numbers ({N_TIERS} of them for n_by_tier)')
    if not isinstance(fit.get('converged'), bool):
        raise ValueError(f'{JSON_NAME}: fit.converged must be true or false')
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
    """게임 상태(피해 팀의 기파괴 타워 수, 파괴 팀, 게임 시간)가 주어졌을 때,
    다음에 파괴될 타워의 티어(0: 외곽, 1: 내부, 2: 억제기, 3: 넥서스) 확률 분포를 예측하는 다항 로지스틱 회귀 모델입니다.

    - fit: FIT 매치들의 타워 파괴 사건으로 모델을 학습합니다.
    - prob: 상태에 따른 4개 티어의 소프트맥스 확률에 1e-6 균등 혼합 스무딩을 적용해 반환합니다.
    - save / load: 모델 가중치(npz)와 메타데이터(json)를 원자적으로 파일에 저장하고 무결성을 검증하며 로드합니다.
    """

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
        """지정된 매치(roles, 기본 FIT)들의 타워 파괴 사건 데이터를 바탕으로 다항 로지스틱 회귀 모델을 학습합니다."""
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
        Z = (X - mean) / scale
        classes = np.unique(rows.tier)

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
        """주어진 게임 상태(피해 팀 기파괴 수 destroyed_victim, 타워 획득 팀 taker, 게임 시간 t_s)에 대해
        4개 타워 티어(0: 외곽, 1: 내부, 2: 억제기, 3: 넥서스)별 파괴 확률을 (n, 4) float64 배열로 계산합니다.

        - 수치 안정성: 로짓 최댓값을 차감한 후 소프트맥스를 계산합니다.
        - 스무딩(Smoothing): 미등장 클래스 확률이 0이 되지 않도록 1e-6 균등 혼합 분포((1-MIX)*p + MIX/4)를 적용합니다.
        """
        X = _design(destroyed_victim, taker, t_s)
        logits = ((X - self.mean_) / self.scale_) @ self.coef_std_.T + self.intercept_std_

        # 수치 안정성을 위해 로짓의 최댓값을 뺀 후 소프트맥스 계산
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        full = np.zeros((len(X), N_TIERS))
        full[:, self.classes_] = e / e.sum(axis=1, keepdims=True)

        return (1.0 - MIX) * full + MIX / N_TIERS

    @property
    def n_iter_(self) -> int:
        return self._info['n_iter']

    @property
    def converged_(self) -> bool:
        return self._info['converged']

    def summary(self) -> dict:
        return {'classes': self.classes_.tolist(), **json.loads(json.dumps(self._info))}

    def save(self, directory) -> Path:
        """학습된 모델 파라미터(tower_tier.npz)와 메타데이터(tower_tier.json)를 디렉토리에 원자적으로 저장합니다."""
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
        """저장된 디렉토리에서 모델을 로드하고 스키마 및 SHA-256 해시 무결성을 검증합니다."""
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
    """예측된 4개 티어 확률(prob)과 각 티어별 Flow Price(골드 가치)를 내적하여
    예측된 타워 1개당 기대 가치(Gold)를 (n,) float64 배열로 산출합니다:
        exp_price = prob @ [price_outer, price_inner, price_base, price_nexus]

    Args:
        prob: (n, 4) float64 배열, TierModel.prob에서 반환된 4개 티어별 예측 확률 분포
        prices: value.load_prices 등에서 로드된 {group_name: price} 매핑 딕셔너리

    Returns:
        (n,) float64 배열: 각 행별 파괴 타워 1개의 기대 골드 가치
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
