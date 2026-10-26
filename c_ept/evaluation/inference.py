"""The inference of the pre-registered comparisons in one module (decision D15 as amended by D20; D11 gives it one seed policy): the one
common match-resample matrix, the ratio bootstrap, the paired-seed statistic, the two-sided p, the Holm family of seven rows, the
non-inferiority rule, the claim / withdrawn / inconclusive verdict and the C3-patch intersection-union rule. Pure numpy.

How a caller uses one matrix for every comparison and all three seeds (D15 "one common match-resample matrix"):
  1. Fix ONE sorted, unique id list per judging split (the matches of the split, not of any head or comparison): ``common`` .
     Every head, arm and seed is aggregated onto it with ``match_aggregate(match, values, ids=common)``. A head or a population
     (D16 per-head row sets, the D14 primary population) that lacks a match puts 0 there; an id outside ``common`` raises.
  2. ``matrix = resample_matrix(len(common))`` once per judging split (B = 10,000, seed 20261022; D13's C4 uses B = 2,000, which is the
     first 2,000 rows of the same matrix). The matrix depends on nothing else: not on the model, the metric, the head or the
     training seed. The same ``n_matches`` and seed give the same bits. The matrix carries no ids: the shared order is the
     ``common`` list, so one list per split must be passed to every aggregation (``bootstrap_stat`` can only check the length).
  3. The weights 1/n_m enter through the per-match aggregation: a per-match mean of a row quantity v is Σ_{r in m} (1/n_m)·v_r
     (``match_aggregate``). For a ratio, the numerator and the denominator are aggregated this way; an absent match has
     a_m = d_m = 0 and so drops out of both sums in every resample. For a mean, the numerator is the per-match mean (0 if absent) and
     the denominator is ``match_presence`` (1 present, 0 absent). The bootstrap then resamples whole matches by the matrix counts.
  4. Intervals of ``metrics.summarize_endpoints`` come from its own per-endpoint stream and are not the common-matrix intervals: the
     confirmatory path (S8, K6, M11) must use ``resample_matrix``. Record the numpy version (and ideally a sha256 of the matrix) in the
     run identity, since numpy does not promise a stable ``Generator.integers`` stream across versions.

수식 기호 (이 모듈 전체)
------------------------
  기호 | 뜻 | 단위 | 범위 | 코드 변수
  m, n | 경기와 경기 수 | —; 개 | 0 기반 | n_matches
  c_{b,m} | 재표본 b에서 경기 m이 뽑힌 횟수 | 개 | 0..n, 행 합 n | matrix
  a_m, d_m | 경기 m의 분자·분모 (경기 안 행 평균, 가중치 1/n_m) | 지표 단위 | d_m ≥ 0 | per_match_num, per_match_den
  Δ | metric(EPT) − metric(comparator); 손실 지표에서 Δ<0이 EPT에 유리 | 지표 단위 | 유한 | per_seed_diffs, draws
  B | 재표본 수 | 개 | 10,000 | matrix.shape[0]
  α | 가족 오류율 | — | 0.05 | alpha

# Ref: Efron, B. (1979), Annals of Statistics 7(1): 1-26 (the bootstrap; percentile intervals).
# Ref: Phipson, B. and Smyth, G. K. (2010), Statistical Applications in Genetics and Molecular Biology 9(1): Art. 39 (the permutation /
#      resampling p-value (1 + #extreme) / (B + 1) never equals 0).
# Ref: Holm, S. (1979), Scandinavian Journal of Statistics 6(2): 65-70 (the step-down adjustment).
# Ref: Berger, R. L. and Hsu, J. C. (1996), Statistical Science 11(4): 283-319 (intersection-union tests).
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np

SEED = 20261022                                       # D15 / D11: the one seed policy; the training seeds are SEED, SEED + 1, SEED + 2
SEEDS = (SEED, SEED + 1, SEED + 2)
B_DEFAULT = 10_000                                    # D15
ALPHA = 0.05                                          # D20: one Holm family of seven rows at alpha = 0.05
HOLM_ROWS = ('structure', 'C1', 'C2a', 'C2b', 'C3-edges', 'C3-terrain', 'C3-patch')    # the seven primary rows of D20
VERDICTS = ('claim', 'withdrawn', 'inconclusive')
_CHUNK = 256                                          # replicates drawn per block (memory only; the stream, hence the matrix, is the same)


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _draws(draws, name: str = 'draws') -> np.ndarray:
    a = np.asarray(draws, dtype=np.float64)
    if a.ndim != 1 or not len(a):
        raise ValueError(f'{name} must be a non-empty vector')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} holds a NaN or an infinite value')
    return a


def _probability(value, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    lo_ok = value > 0 if open_interval else value >= 0
    hi_ok = value < 1 if open_interval else value <= 1
    if not (lo_ok and hi_ok):
        raise ValueError(f'{name} is outside the allowed range')
    return float(value)


# ---- the common resample matrix and the per-match aggregation --------------------------------------------------------------------------
def resample_matrix(n_matches: int, B: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    """The (B, n_matches) matrix of match counts of B match resamples, from ``np.random.default_rng(seed)``.

    수식
    ----
    (INF-1)  c_{b,m} = #{ j in 0..n−1 : i_{b,j} = m },  i_{b,j} ~ Uniform{0..n−1} iid,  Σ_m c_{b,m} = n

    해석
    ----
    한 판의 재표본은 경기 n개를 복원 추출한 횟수 벡터이다. 행 b는 ``rng.integers(n, size=n)``의 b번째 호출 결과의 횟수로, 같은
    (n_matches, B, seed)는 같은 행렬이다. 모든 비교와 세 시드가 이 행렬 하나를 공유한다 (D15): 경기 순서를 한 번 고정하고
    (``match_aggregate``의 정렬된 경기 번호) 행렬도 한 번만 만든다. 한 경기의 모든 행은 함께 뽑혀 경기 안의 1/n_m 가중이 유지된다.
    """
    n = _count(n_matches, 'n_matches')
    reps = _count(B, 'B')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    out = np.empty((reps, n), dtype=np.int32)
    offset = np.arange(_CHUNK, dtype=np.int64)[:, None] * n
    for start in range(0, reps, _CHUNK):
        rows = min(_CHUNK, reps - start)
        idx = rng.integers(n, size=(rows, n)) + offset[:rows]                      # the draws of consecutive replicates, row by row
        out[start:start + rows] = np.bincount(idx.ravel(), minlength=rows * n).reshape(rows, n)       # (INF-1)
    return out


def _index_matches(match, ids):
    """``(ids, position of every row's match in ids)``; ``ids=None`` takes the sorted unique matches present in ``match``."""
    m = np.asarray(match)
    if m.ndim != 1 or not len(m):
        raise ValueError('match must be a non-empty vector')
    if ids is None:
        ids, inverse = np.unique(m, return_inverse=True)
        return ids, inverse.ravel()
    ids = np.asarray(ids)
    if ids.ndim != 1 or not len(ids) or (len(ids) > 1 and not (ids[1:] > ids[:-1]).all()):
        raise ValueError('ids must be a non-empty, sorted, unique vector')
    pos = np.searchsorted(ids, m)
    pos_c = np.minimum(pos, len(ids) - 1)
    if (ids[pos_c] != m).any():
        raise ValueError('a row belongs to a match that is not in ids')
    return ids, pos_c


def match_aggregate(match, values, ids=None) -> tuple:
    """``(ids, per_match_mean)``: for each match of ``ids`` the mean of ``values`` over its rows. With ``ids=None`` these are the sorted
    unique matches present in ``match``. With ``ids`` (the sorted, unique id list of the judging split, shared by every head and
    comparison) the result is on exactly that list: a match with no row gets 0, and a row whose match is not in ``ids`` raises.

    수식
    ----
    (INF-2)  v̄_m = Σ_{r in m} (1/n_m)·v_r  (m present);  v̄_m = 0  (m absent from the rows)

    해석
    ----
    D15의 가중치 1/n_m(경기마다 합이 1)가 들어가는 곳이다. 분자와 분모를 이 함수로 각각 모아 ``bootstrap_stat``에 넘긴다. 머리마다
    행 집합이 다르거나(D16) 주 모집단이 일부 경기를 빼는(D14) 경우에도 같은 판정 분할의 id 목록을 넘기면 모든 벡터가 같은 열
    순서를 갖는다. 빠진 경기는 분자·분모가 모두 0이라 비에서 빠진다. 평균이면 분모로 ``match_presence``(있으면 1, 없으면 0)를 쓴다.
    """
    v = np.asarray(values, dtype=np.float64)
    ids, pos = _index_matches(match, ids)
    if v.shape != pos.shape:
        raise ValueError('match and values must be vectors of the same length')
    if not np.isfinite(v).all():
        raise ValueError('values holds a NaN or an infinite value')
    count = np.bincount(pos, minlength=len(ids))
    total = np.bincount(pos, weights=v, minlength=len(ids))
    return ids, np.where(count > 0, total / np.maximum(count, 1), 0.0)           # (INF-2)


def match_presence(match, ids) -> np.ndarray:
    """1.0 for each match of ``ids`` that has at least one row in ``match``, else 0.0 (the denominator of a mean over present matches).

    수식
    ----
    (INF-2b)  π_m = [n_m ≥ 1]
    """
    ids, pos = _index_matches(match, ids)
    return (np.bincount(pos, minlength=len(ids)) > 0).astype(np.float64)

def bootstrap_stat(per_match_num, per_match_den, matrix) -> np.ndarray:
    """The ratio estimator on every resample, the numerator and the denominator recomputed together on the same resample (D16).

    수식
    ----
    (INF-3)  T*_b = Σ_m c_{b,m}·a_m / Σ_m c_{b,m}·d_m,   b = 0..B−1
             (the point estimate is Σ_m a_m / Σ_m d_m: the matrix of ones)

    해석
    ----
    상대 NLL R_h = Σw·NLL_model / Σw·NLL_B0 처럼 분자와 분모가 모두 경기에 따라 변하는 비의 추정량이다. 재표본마다 둘을 같이 다시
    합하므로 분모를 점 추정에 고정하거나 경기별 비의 평균을 내는 것과 다르다. 평균(분모 1)은 d_m = 1로 얻는다. 재표본의 분모가
    0이면 비가 정의되지 않으므로 ValueError로 멈춘다.
    """
    a = np.asarray(per_match_num, dtype=np.float64)
    d = np.asarray(per_match_den, dtype=np.float64)
    c = np.asarray(matrix)
    if a.ndim != 1 or d.shape != a.shape or not len(a):
        raise ValueError('per_match_num and per_match_den must be non-empty vectors of the same length')
    if c.ndim != 2 or c.shape[1] != len(a):
        raise ValueError(f'matrix must have the shape (B, {len(a)}), not {c.shape}')
    if not (np.isfinite(a).all() and np.isfinite(d).all()):
        raise ValueError('per-match numerator or denominator holds a NaN or an infinite value')
    num = c @ a
    den = c @ d
    if not (den > 0).all():
        raise ValueError('a resample has a denominator of 0 or less: the ratio is undefined')
    return num / den                                                          # (INF-3)


# ---- the paired seeds ------------------------------------------------------------------------------------------------------------------
def paired_seed_diffs(ept_by_seed, comparator_by_seed) -> dict:
    """``{seed: metric(EPT) − metric(comparator)}`` for the seed values 20261022, +1, +2, paired by equal seed value.

    수식
    ----
    (INF-4)  Δ_s = metric_EPT(s) − metric_comparator(s),  s in {20261022, 20261023, 20261024}

    해석
    ----
    두 팔은 같은 시드 값끼리만 짝지는다 (D15). 시드 값이 세 개가 아니거나 두 팔의 시드 값이 다르면 ValueError이다.
    """
    for name, arm in (('EPT', ept_by_seed), ('comparator', comparator_by_seed)):
        if not isinstance(arm, Mapping) or set(arm) != set(SEEDS):
            raise ValueError(f'the {name} arm must give exactly the seed values {SEEDS}')
    return {s: ept_by_seed[s] - comparator_by_seed[s] for s in SEEDS}


def paired_seed_mean(per_seed_diffs):
    """The primary statistic: the mean of the three paired differences. A mapping must hold exactly the seed values of ``SEEDS``; a
    sequence must hold three values (in seed order). Scalars give a float, equal-length vectors (the bootstrap draws of each seed, from
    the one shared matrix) give the vector of means.

    수식
    ----
    (INF-5)  Δ̄ = (Δ_1 + Δ_2 + Δ_3) / 3;  Δ̄*_b = (Δ*_{1,b} + Δ*_{2,b} + Δ*_{3,b}) / 3
    """
    if isinstance(per_seed_diffs, Mapping):
        if set(per_seed_diffs) != set(SEEDS):
            raise ValueError(f'the differences must be keyed by exactly the seed values {SEEDS}')
        items = [per_seed_diffs[s] for s in SEEDS]
    else:
        items = list(per_seed_diffs)
        if len(items) != len(SEEDS):
            raise ValueError('three paired differences are needed')
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in items])
    if not np.isfinite(stack).all():
        raise ValueError('a paired difference is NaN or infinite')
    mean = stack.mean(axis=0)                                                  # (INF-5)
    return float(mean) if mean.ndim == 0 else mean


# ---- p, Holm, non-inferiority, verdict -------------------------------------------------------------------------------------------------
def two_sided_p(draws, observed) -> float:
    """p = min(1, 2·(1 + #{Δ* on the far side of 0}) / (B + 1)), the far side being the side of 0 opposite to ``observed`` (draws
    exactly at 0 count as far side, which is conservative). An observed difference of exactly 0 has p = 1.

    수식
    ----
    (INF-6)  observed < 0:  far = #{b : Δ*_b ≥ 0};   observed > 0:  far = #{b : Δ*_b ≤ 0}
             p = min(1, 2·(1 + far) / (B + 1))

    해석
    ----
    백분위 부트스트랩 분포가 점 추정을 중심으로 놓이므로, 0이 분포의 한쪽 꼬리에 얼마나 깊이 들어가는지가 양측 p이다. 분자의 1은
    p가 0이 되지 않게 하는 보정이다 (Phipson-Smyth). 최솟값은 2/(B+1)이다.
    """
    d = _draws(draws)
    if isinstance(observed, bool) or not isinstance(observed, numbers.Real) or not np.isfinite(observed):
        raise ValueError('observed must be a finite number')
    if observed == 0:
        return 1.0
    far = int((d >= 0).sum()) if observed < 0 else int((d <= 0).sum())
    return float(min(1.0, 2.0 * (1 + far) / (len(d) + 1)))                     # (INF-6)


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    """The Holm step-down adjustment of a family: ``{name: {'p', 'adjusted_p', 'reject'}}`` (input order), ``reject`` meaning
    adjusted p < alpha. D20 gives one family of the seven rows ``HOLM_ROWS`` at alpha = 0.05; the function takes any non-empty family
    (the caller passes the seven).

    수식
    ----
    (INF-7)  p_(1) ≤ ... ≤ p_(m) (ties by name);  adj_(i) = min(1, max_{j ≤ i} (m − j + 1)·p_(j));  reject_(i) = [adj_(i) < α]

    해석
    ----
    가장 작은 p부터 (m, m−1, …, 1)을 곱하고 앞 단계의 최댓값을 이어받아 조정 p가 단조가 되게 한다. 같은 수준 α에서 판정하므로
    D15의 철회(비교군이 유의하게 더 낫다)도 이 조정 p를 쓴다 (D20: Holm 하나, α = 0.05).
    """
    if not isinstance(pvalues, Mapping) or not pvalues:
        raise ValueError('pvalues must be a non-empty mapping')
    level = _probability(alpha, 'alpha', open_interval=True)
    p = {k: _probability(v, f'p of {k}') for k, v in pvalues.items()}
    order = sorted(p, key=lambda k: (p[k], str(k)))
    m = len(order)
    adjusted, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))                         # (INF-7)
        adjusted[k] = running
    return {k: {'p': p[k], 'adjusted_p': adjusted[k], 'reject': adjusted[k] < level} for k in pvalues}


def noninferior(draws, eps, level: float = 0.95) -> bool:
    """True if the upper end of the two-sided ``level`` percentile interval of Δ is strictly below ``eps`` (> 0). Lower is better (a loss
    difference such as the C3-patch NLL ratio); there is no direction switch.

    수식
    ----
    (INF-8)  upper = quantile(Δ*, (1 + level) / 2);  noninferior = [upper < ε]

    해석
    ----
    비열등은 평균이 아니라 구간의 위쪽 끝이 한계 ε 아래에 있을 때 성립한다: 손실 차 Δ의 가장 불리한 그럴듯한 값도 ε보다 작아야
    한다 (D15, C3-패치의 DEV 부분). ε는 호출자가 미리 잠근 절대 단위 값이다.
    """
    d = _draws(draws)
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be a positive finite number')
    lvl = _probability(level, 'level', open_interval=True)
    return bool(np.quantile(d, (1 + lvl) / 2) < eps)                         # (INF-8)


def verdict(per_seed_diffs, adjusted_p: float, *, alpha: float = ALPHA, lower_is_better: bool = True) -> str:
    """'claim', 'withdrawn' or 'inconclusive' for one row of the family, with ``adjusted_p`` from ``holm`` (the same alpha).

    수식
    ----
    (INF-9)  s = +1 if lower is better else −1;  Δ̄ = paired_seed_mean
             claim:       adj < α  and  s·Δ̄ < 0  and  s·Δ_s < 0 for all three seeds
             withdrawn:   adj < α  and  s·Δ̄ > 0
             inconclusive: otherwise

    해석
    ----
    주장은 조정 p가 α보다 작고, 점 추정과 세 시드의 짝 차이가 모두 EPT에 유리해야 선다 (0은 유리하지 않다). 철회는 같은 조정
    수준에서 비교군이 유의하게 더 나을 때이며 세 시드 부호는 요구하지 않는다. 그 밖은 결론 없음이다. 시드별 차이는 호출자가 항상
    함께 보고한다 (D15). 가족은 D20의 일곱 줄 하나이다.
    """
    items = [per_seed_diffs[s] for s in SEEDS] if isinstance(per_seed_diffs, Mapping) and set(per_seed_diffs) == set(SEEDS) \
        else (list(per_seed_diffs) if not isinstance(per_seed_diffs, Mapping) else None)
    if items is None or len(items) != len(SEEDS):
        raise ValueError('three paired differences are needed (a sequence of three, or a mapping keyed by the three seed values)')
    d = np.asarray(items, dtype=np.float64)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError('the paired differences must be three finite numbers')
    adj = _probability(adjusted_p, 'adjusted_p')
    level = _probability(alpha, 'alpha', open_interval=True)
    s = 1.0 if lower_is_better else -1.0
    point = s * float(d.mean())
    if adj < level:
        if point < 0 and bool((s * d < 0).all()):
            return 'claim'
        if point > 0:
            return 'withdrawn'
    return 'inconclusive'                                                      # (INF-9)


def c3_patch_passes(*, noninferior_dev: bool, superiority_verdict: str) -> bool:
    """The C3-patch rule as an intersection-union: the DEV non-inferiority (``noninferior``) and the 15.22 superiority (a 'claim' of
    ``verdict`` on the TEST row, adjusted inside the one Holm family of D20) must both hold.

    수식
    ----
    (INF-10)  pass = [noninferior_DEV] ∧ [verdict_TEST = 'claim']

    해석
    ----
    두 부분이 모두 통과해야 통과한다 (교집합-합집합 시험). D15의 별도 수준 0.05/7은 D20이 일곱 줄 Holm 하나로 바꿨으므로 여기서는
    그 가족 안의 판정 문자열만 받는다.
    """
    if superiority_verdict not in VERDICTS:
        raise ValueError(f'superiority_verdict must be one of {VERDICTS}')
    if not isinstance(noninferior_dev, (bool, np.bool_)):
        raise ValueError('noninferior_dev must be a bool')
    return bool(noninferior_dev) and superiority_verdict == 'claim'            # (INF-10)
