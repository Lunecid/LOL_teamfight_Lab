"""The proper scores and weighted means of the evaluation, in one module (decision D11): ``crps19``, ``pit_histogram``, ``quantile_hits``,
``brier`` and ``weighted_mean``. Pure numpy; no data, model or device.

The three quantile scores are the arithmetic of the frozen S3 report (``c_ept.reporting.report_s3``), operation for operation, so that
they equal it bit for bit on the same inputs (a test pins this); the frozen file is not changed and keeps its own copy.

수식 기호 (이 모듈 전체)
------------------------
  기호 | 뜻 | 단위 | 범위 | 코드 변수
  q_{r,k} | 행 r의 k번째 분위수 (수준 τ_k = k/20) | 골드 | 유한, 행 안에서 비감소 | q19
  y_r | 행 r의 실현값 | 골드 | 유한 | realized
  w_r | 행 가중치 | — | 유한·비음수, 합 양수 | w
  τ_k | 분위 수준 | — | 0.05 .. 0.95 (19개) | LEVELS (= compose.Q_LEVELS)
  p_r, o_r | 예측 확률과 이진 결과 | —; 0/1 | [0,1]; {0,1} | p, y

# Ref: Gneiting, T. and Raftery, A. E. (2007), JASA 102(477): 359-378 (CRPS as twice the integral of the pinball loss over the levels).
# Ref: Brier, G. W. (1950), Monthly Weather Review 78(1): 1-3.
# Ref: Czado, C., Gneiting, T. and Held, L. (2009), Biometrics 65(4): 1254-1261 (PIT with point masses; background for the coarse bin rule).
"""
from __future__ import annotations

import math

import numpy as np

from c_ept.valuation import compose

LEVELS = compose.Q_LEVELS                              # tau_k = k / 20, the 19 levels of the composer (read-only array, the same object)
N_QUANTILES = len(LEVELS)                              # 19
N_PIT_BINS = N_QUANTILES + 1                           # the 19 quantiles cut the unit interval of the PIT into 20 bins of 0.05


class ProperScoreError(ValueError):
    """The inputs of a score cannot be judged (shape, NaN, order, weights). The message names the function, never a key.

    점수를 매길 수 없는 입력(모양, NaN, 분위 순서, 가중치)일 때 쓰는 예외이다.
    """


def weighted_mean(values, w) -> float:
    """Σ w·values / Σ w as ``float(np.dot(values, w / w.sum()))``: the normalised weights first, so that a representable mean does not
    overflow in the sum. ValueError for no row, a shape mismatch, a negative or non-finite weight, weights adding up to 0, or a
    non-finite result.

    수식
    ----
    (PRP-1)  mean = Σ_r (w_r / W)·v_r,  W = Σ_r w_r

      기호 | 뜻 | 단위 | 범위 | 코드 변수
      v_r | 행 값 | 값의 단위 | 유한 | values
      w_r, W | 가중치와 합 | — | 비음수, W>0 | w, w.sum()

    해석
    ----
    합이 1이 되도록 가중치를 먼저 나눈 뒤 내적한다. 합 W를 먼저 곱하지 않으므로 표현 가능한 평균이 중간 합에서 넘치지 않는다.
    기존 evaluation.metrics의 가중 평균과 같은 연산 순서여서 그 출력이 비트까지 같다.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if v.ndim != 1 or w.shape != v.shape:
        raise ProperScoreError(f'weighted_mean: values {v.shape} and w {w.shape} must be vectors of the same length')
    if not len(w):
        raise ProperScoreError('weighted_mean: there is no row')
    if not np.isfinite(w).all() or (w < 0).any():
        raise ProperScoreError('weighted_mean: a weight is negative, NaN or infinite')
    total = w.sum()
    if total == 0:
        raise ProperScoreError('weighted_mean: the weights add up to 0')
    result = float(np.dot(v, w / total))               # (PRP-1)
    if not np.isfinite(result):
        raise ProperScoreError('weighted_mean: nonfinite metric arithmetic')
    return result


def brier(p, y, w) -> float:
    """The weighted Brier score Σ w·(p − y)² / Σ w for probabilities ``p`` in [0, 1] and binary outcomes ``y`` (bool accepted).

    수식
    ----
    (PRP-2)  Brier = Σ_r w_r·(p_r − o_r)² / Σ_r w_r

    해석
    ----
    이진 사건 확률의 적정 점수이다. 값이 작을수록 좋고, 비교는 같은 행·같은 가중치에서 해야 한다.
    """
    p = np.asarray(p, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    if p.shape != o.shape or p.ndim != 1:
        raise ProperScoreError(f'brier: p {p.shape} and y {o.shape} must be vectors of the same length')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ProperScoreError('brier: p holds a value that is not a probability')
    if not np.isin(o, (0.0, 1.0)).all():
        raise ProperScoreError('brier: y holds a value that is not a binary label')
    return weighted_mean((p - o) ** 2, w)              # (PRP-2)


# ---- the three quantile-grid scores (the arithmetic of report_s3) ---------------------------------------------------------------------
def _quantile_rows(fn: str, q19, realized, w) -> tuple:
    """``(q19, realized, w, total weight)`` as float64 arrays, or ProperScoreError for shapes, values and weights that cannot be judged."""
    q = np.asarray(q19, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != N_QUANTILES:
        raise ProperScoreError(f'{fn}: q19 must have the shape (n, {N_QUANTILES}), not {q.shape}')
    n = len(q)
    if n == 0:
        raise ProperScoreError(f'{fn}: there is no row')
    if y.shape != (n,) or w.shape != (n,):
        raise ProperScoreError(f'{fn}: realized {y.shape} and w {w.shape} must be vectors of the {n} rows of q19')
    if not (np.isfinite(q).all() and np.isfinite(y).all()):
        raise ProperScoreError(f'{fn}: q19 or realized holds a NaN or an infinite value')
    if (np.diff(q, axis=1) < 0).any():
        raise ProperScoreError(f'{fn}: a row of q19 decreases: the quantiles must not decrease with the level')
    if not np.isfinite(w).all() or (w < 0).any():
        raise ProperScoreError(f'{fn}: a weight is negative, NaN or infinite')
    total = math.fsum(w)
    if not total > 0:
        raise ProperScoreError(f'{fn}: the weights add up to 0')
    return q, y, w, total


def pit_histogram(q19, realized, w) -> dict:
    """The coarse probability-integral-transform histogram of ``realized`` against predictive laws given by 19 quantiles (levels k / 20).

    ``q19`` is (n, 19), non-decreasing along each row; ``realized`` and ``w`` (sum > 0) are (n,). Bin j (j = 0 .. 19) is the interval
    [j / 20, (j + 1) / 20]. A row with no tie lies in bin ``n_lt``; a row that ties with quantiles (a point mass of the law, for example
    V_start = 0 when no engagement starts) has its weight spread evenly over the bins ``n_lt .. n_le``. Returns ``{'bins', 'mass' (the
    weighted share of each bin, adding up to 1), 'expected_mass' (0.05), 'n_rows', 'n_tied_rows', 'tied_weight_share',
    'max_abs_deviation'}``.

    수식
    ----
    (PRP-3)  less_r = Σ_k [q_{r,k} < y_r]; leq_r = Σ_k [q_{r,k} ≤ y_r]; span_r = leq_r − less_r + 1
             mass_j = max(0, Σ_r w_r·[less_r ≤ j ≤ leq_r] / span_r / W);  max_dev = max_j |mass_j − 1/20|

    해석
    ----
    동점 행의 가중치를 포함된 빈들에 균등 배분하는 거친 분위 빈 진단이다. 알려진 전체 이산 CDF의 정확한 randomized PIT와 같지
    않고, 기준 0.05는 보정된 이산 법에서도 보장되지 않는다. 기준에서 벗어난 것만으로 보정 실패를 증명하지 못한다 (S3 R81).
    """
    q, y, w, total = _quantile_rows('pit_histogram', q19, realized, w)
    n_lt = (q < y[:, None]).sum(axis=1)
    n_le = (q <= y[:, None]).sum(axis=1)
    span = n_le - n_lt + 1
    bins = np.arange(N_PIT_BINS)[None, :]
    inside = (bins >= n_lt[:, None]) & (bins <= n_le[:, None])
    mass = np.maximum((inside * (w / span)[:, None]).sum(axis=0) / total, 0.0)      # (PRP-3)
    tied = span > 1
    return {'bins': N_PIT_BINS, 'mass': mass, 'expected_mass': 1.0 / N_PIT_BINS, 'n_rows': int(len(y)), 'n_tied_rows': int(tied.sum()),
            'tied_weight_share': math.fsum(w[tied]) / total, 'max_abs_deviation': float(np.max(np.abs(mass - 1.0 / N_PIT_BINS)))}


def quantile_hits(q19, realized, w) -> dict:
    """For each level k / 20 the weighted share of rows whose realized value lies below the quantile (``below``) and at or below it
    (``at_or_below``). ``{'levels', 'below', 'at_or_below'}``.

    수식
    ----
    (PRP-4)  below_k = Σ_r w_r·[y_r < q_{r,k}] / W;  at_or_below_k = Σ_r w_r·[y_r ≤ q_{r,k}] / W

    해석
    ----
    엄격 비교와 포함 비교를 함께 보고해 동점 질량을 분리한다. 모집단의 두 확률 사이에 명목 수준이 놓이는 것이 보정이며, 유한 표본의
    가중 적중률에는 그 괄호 관계가 반드시 성립하지 않는다 (S3 R81).
    """
    q, y, w, total = _quantile_rows('quantile_hits', q19, realized, w)
    below = (w[:, None] * (y[:, None] < q)).sum(axis=0) / total
    at_or_below = (w[:, None] * (y[:, None] <= q)).sum(axis=0) / total
    return {'levels': LEVELS, 'below': below, 'at_or_below': at_or_below}                # (PRP-4)


def crps19(q19, realized, w) -> float:
    """The finite-grid pinball proxy for the CRPS on the 19 levels k / 20 (design section S1-4): twice the mean over the levels of the
    pinball loss, averaged over the rows with the weights ``w``. For a row whose 19 quantiles all equal one value the row loss is the
    absolute error.

    수식
    ----
    (PRP-5)  e_{r,k} = y_r − q_{r,k};  L_{r,k} = τ_k·e (e ≥ 0), (τ_k − 1)·e (e < 0)
             crps19 = Σ_r w_r·(2/19)·Σ_k L_{r,k} / W

    해석
    ----
    연속 분위 적분 CRPS의 유한 격자 대리값이다 (Gneiting-Raftery 2007: CRPS = 핀볼 손실 적분의 2배). 한 행의 분위수가 모두 같으면
    그 행의 손실은 절대오차이다. 점수의 비교는 같은 대상·가중치에서 이루어져야 한다.
    """
    q, y, w, total = _quantile_rows('crps19', q19, realized, w)
    error = y[:, None] - q
    pinball = np.where(error >= 0, LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return math.fsum(w * (2.0 * pinball.mean(axis=1))) / total                          # (PRP-5)
