r"""The S3 composition gate: calibration metrics, the agreement of the simulation with the closed form, the over-dispersion test and the
verdict (stage S3 Task 10; decisions P7 and P8).

The gate decides whether the composition of the small M2-M4 heads is good enough to go on to S4. It is made of four criteria and one report
item. Everything here is a pure function of arrays: it reads no store, no file and no match key.

    occurrence_calibration  (i)   observed / expected 60 s occurrences, and the slope of observed on expected
    value_calibration       (ii)  the slope of realized V_start on E[V_start | t0], and the mean error in gold
    (accounting identity)   (iii) ``value.accounting()['max_abs_error']`` must be exactly 0; it is judged in ``verdict``
    mc_agreement            (iv)  the share of rows whose simulated mean lies within 3 standard errors of the closed form
    overdispersion          P8    the Pearson ratio of the 60 s counts: reported, not a criterion
    verdict                 the pass or fail of every criterion, with the numbers it was judged on and the bands

The thresholds are module constants, written before any DEV result exists (guideline F3). The plan states why: with 3,967 DEV matches the
rule "the interval contains 1" fails for any small drift between patches, so the question asked is an equivalence one, "the interval lies
inside a band that was fixed in advance". A band is a judgement, not a measurement; to change one after a DEV result has been seen is a
change of the protocol, and it has to be reported as such.

Intervals. Every interval is a percentile 95% interval (the 2.5th and 97.5th percentiles) of ``B`` match-bootstrap replicates, and ``B`` is
2,000 by default (``nll.BOOT_B``) with the project seed (``nll.BOOT_SEED``). Replicate j has the match counts of the j-th draw of one
stream, ``numpy.random.default_rng(seed).multinomial(M, 1/M)`` with M the number of distinct matches (in increasing order of their
number), exactly the draw of ``nll.match_bootstrap`` (ruling R26): the same seed resamples the same matches in every S3 statistic, so
the intervals of different heads and different statistics can be put side by side. The matches carry the weights the caller gives (the
gate gives 1/n_u, ``features.match_weights``). A replicate is computed from per-match sufficient statistics,

    W = sum w,  Sx = sum w x,  Sy = sum w y,  Sxx = sum w x^2,  Sxy = sum w x y,  (and sum w o, sum w e, sum w (y - x), sum w z)

each added up over the rows of a match with ``math.fsum``. A sum that is exactly rounded does not depend on the order of the rows, so the
results of this module are bit for bit the same for any order of the rows (a resumed run may write its rows in another order). The second
moments are taken about the weighted means of the whole sample, so that a slope is not computed as the difference of two large, almost
equal numbers when the values sit far from 0. The estimate itself is the same statistic of the whole sample, not the mean of the
replicates.

A replicate that has no value, for example a slope when the resampled expected values do not vary, is NaN and makes both ends of that
interval NaN: an interval that could not be computed is not inside any band. (With thousands of matches it does not happen.) When the whole
sample has no such value, the function raises, because there is nothing to judge.

The deciles are descriptive, not judged. Ten bins of the expected value by the 10th, 20th, ... 90th percentiles of the rows (half open: a
row on an edge is in the upper bin); equal expected values are never split, so a bin can be empty (count 0, no means). Each bin has the
weighted mean of the expected and of the observed (realized) value, the number of rows and the total weight.

Rows without information (ruling R22). A row that carries no information for a statistic is left out and counted, as in the head NLLs:
for the over-dispersion ratio these are the rows with an expected count of 0 (the cell covers no time), which must also have a count
of 0. Their weights are dropped, not given to the other rows of the match, and a match with no remaining row is not resampled. The two
calibration functions leave nothing out: they judge every row they are given, also a row with an expected value of 0 (it adds nothing to
a sum of expected values, but it is a point of the line), so the caller chooses the rows of the gate.

Output. Every result is a dict of plain JSON values (``float``, ``int``, ``bool``, ``str``, ``None``, ``list``, ``dict``): an interval is
a list of two floats. No message names a match number.

방법에서의 위치. 이 모듈은 합성기가 낸 기대값을 DEV 경기의 실현값과 맞대어 S4로 갈지 정하는 합성 관문의 네 기준(계획 P7)과 과산포 보고(계획 P8)를 계산한다. 행은 분 프레임(경기 상태를
1분마다 찍은 기록)의 기준 시각 t₀ 하나이고, 기준은 t₀ 뒤 60초 안 발생 수(첫 킬이 그 안에 있는 교전의 수)와 V_start(t₀, H)(첫 킬이 t₀ 뒤 H초 안에 있는 교전들의 f₀(K, O) + f₀(전환)의 합)를 모형의
기대값과 맞댄다. 공통 조건은 DEV 경기, H = 60 s, 경기 가중 1/n_u, 경기 부트스트랩 B = 2,000이고 네 기준이 모두 통과해야 S4로 가자고 권한다(계획 P7). 관문은 B0 머리로 합성한 값으로
판정하고 분산비는 B1 머리의 M2 기대 개수로 계산한다(계획 P6, P8). 보정 기준 (i)·(ii)의 판정은 점추정이 아니라 경기를 통째로 다시 뽑은 백분위 구간을 미리 잠근 띠와 맞대어 한다. 기대값 E_i와 실현값 R_i는
행 i에 모형이 준 값과 그 행에서 실제로 나온 값이다(발생 보정은 기대·관측 발생 수, 가치 보정은 E[V_start | t₀]와 실현 V_start).

식 목록. 식 번호는 (모듈 약어-번호)이고 식과 기호는 docs/superpowers/notes/equation-index.md에 있다. i는 행, u는 경기, j는 복제의 번호이다.
    (GATE-1)  경기 재표본 구간   θ*_j = stat(Σ_u d_ju·⟨f⟩_u),  구간 = [Pct₂.₅, Pct₉₇.₅](θ*)      (d_j ~ Multinomial(M; 1/M, …, 1/M), j = 1, …, B)
    (GATE-2)  가중 직선          R_i = γ + β·E_i + 오차,   β = 가중 공분산(E, R) / 가중 분산(E)     (중심화한 합 ⟨f⟩로 계산)
    (GATE-3)  발생 보정 (i)      oe = ⟨R⟩ / ⟨E⟩,   기울기 β
    (GATE-4)  가치 보정 (ii)     기울기 β,   bias = ⟨R − E⟩ / ⟨1⟩
    (GATE-5)  모의 일치 (iv)     |V̂_i − E_i| < 3·sd_i / √S 인 행의 비율
    (GATE-6)  피어슨 분산비      φ = Σ_i w_i·(y_i − μ_i)² / μ_i  /  Σ_i w_i      (μ_i > 0인 행)
    (GATE-7)  판정               구간이 띠 안인가, 회계 오차 (iii)가 0인가, 일치 비율이 한계 이상인가

공통 가정.
    (가) 경기 단위. 같은 경기의 행은 같은 흐름을 공유해 서로 의존하므로 다른 경기끼리만 독립이라고 가정하고, 관문은 행마다 1/n_u를 주며 구간은 경기를
         통째로 다시 뽑아 만든다(계획 P7, 판정 R26).
    (나) 사전 고정. 띠와 한계는 DEV 결과를 보기 전에 모듈 상수로 잠근 판단값이고 결과를 본 뒤의 변경은 프로토콜 변경이다(계획 P7, 지침 F3). 점추정은 보이기만 하고 판정하지 않으며,
         값이 없는 구간(NaN)은 어떤 띠 안에도 들지 않는다.
    (다) 정보 없는 행. 분산비는 기대 개수가 0인 행을 빼고 그 수를 세며(판정 R22), 두 보정 함수는 받은 행을 모두 판정하므로 어느 행을 넘길지는 호출한 쪽이 정한다.
"""
# Ref: docs/superpowers/plans/2026-10-21-c-ept-s3.md decisions P7 (the four criteria, the bands locked before the DEV results, the match
#      weights 1/n_u, B = 2,000) and P8 (the Pearson ratio, the limit 1.1) and Task 10; the ledger rulings R22 (rows without information
#      are left out and counted) and R26 (one weighting and one bootstrap draw for every S3 interval); math-control note 03 sections F
#      (calibration) and G (over-dispersion), whose numbers the tests reproduce.
# Ref: Efron (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1): 1-26; doi:10.1214/aos/1176344552,
#      and Cameron, Gelbach, Miller (2008), "Bootstrap-based improvements for inference with clustered errors", Review of Economics
#      and Statistics 90(3): 414-427; doi:10.1162/rest.90.3.414 (resample whole clusters, here matches; percentile intervals); the draw is
#      the one of ``nll.match_bootstrap``.
# Ref: Steyerberg et al. (2010), "Assessing the performance of prediction models: a framework for traditional and novel measures",
#      Epidemiology 21(1): 128-138; doi:10.1097/EDE.0b013e3181c30fb2, and Van Calster et al. (2016), "A calibration hierarchy for risk
#      models was defined: from utopia to empirical data", Journal of Clinical Epidemiology 74: 167-176; doi:10.1016/j.jclinepi.2015.12.005
#      (the calibration slope, the mean calibration, observed / expected).
# Ref: Hosmer and Lemeshow (1980), "Goodness of fit tests for the multiple logistic regression model", Communications in Statistics -
#      Theory and Methods 9(10): 1043-1069; doi:10.1080/03610928008827941 (the table of groups by deciles of the predicted value).
# Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
#      doi:10.1017/CBO9781139013567 (the Pearson statistic as a measure of dispersion of counts in a Poisson model; its mean is 1 when
#      the variance equals the mean).
# Ref: Schuirmann (1987), Journal of Pharmacokinetics and Biopharmaceutics 15(6): 657-680; doi:10.1007/BF01068419, and Lakens (2017),
#      "Equivalence tests: a practical primer for t tests, correlations, and meta-analyses", Social Psychological and Personality
#      Science 8(4): 355-362; doi:10.1177/1948550617697177 (an interval inside a band fixed in advance is an equivalence test; the
#      reason for the bands of P7).
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from c_ept import nll

# ---- the pre-registered thresholds (P7 and P8); locked before any DEV result exists ------------------------------------------------------
OE_BAND = (0.95, 1.05)           # (i)  the 95% interval of observed / expected must lie inside
OCC_SLOPE_BAND = (0.9, 1.1)      # (i)  the 95% interval of the slope of the observed on the expected counts must lie inside
VALUE_SLOPE_BAND = (0.8, 1.2)    # (ii) the 95% interval of the slope of realized V_start on E[V_start | t0] must lie inside
BIAS_BAND = (-50.0, 50.0)        # (ii) the 95% interval of the mean error (realized - expected), in gold, must lie inside
MC_SHARE = 0.99                  # (iv) at least this share of the rows must agree with the closed form
DISPERSION_LIMIT = 1.1           # P8   the lower end of the interval of the Pearson ratio must be above this: over-dispersed

GATE_CRITERIA = ('occurrence', 'value', 'accounting', 'mc')       # the four criteria that all have to pass (P7), in the order of the plan
REPORTED = ('overdispersion',)                                    # P8: reported with the verdict, not a pass or fail item

_PERCENTILES = (2.5, 97.5)       # a 95% interval
_N_DECILES = 10
_MC_Z = 3.0                      # (iv): within 3 standard errors, strictly
_DEGENERATE = 1e-12              # a resample whose x vary less than this (relative) has no slope, not a quotient of rounding errors


# ---- checks of the arguments -------------------------------------------------------------------------------------------------------------
def _count(mask) -> int:
    return int(np.count_nonzero(mask))


def _whole(fn: str, what: str, x, minimum: int) -> int:
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f'{fn}: {what} must be an integer, not {type(x).__name__}')
    if x < minimum:
        raise ValueError(f'{fn}: {what} must be at least {minimum}, not {x}')
    return int(x)


def _vector(fn: str, what: str, x) -> np.ndarray:
    """``x`` as a non-empty 1-d float64 array of finite numbers."""
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f'{fn}: {what} must be a non-empty 1-d array, not shape {v.shape}')
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f'{fn}: {_count(bad)} value(s) of {what} are NaN or infinite')
    return v


def _same_length(fn: str, what: str, v: np.ndarray, n: int) -> None:
    if v.shape != (n,):
        raise ValueError(f'{fn}: {what} has shape {v.shape} but there are {n} rows')


def _counts(fn: str, what: str, v: np.ndarray) -> None:
    bad = (v < 0) | (v != np.floor(v))
    if bad.any():
        raise ValueError(f'{fn}: {_count(bad)} value(s) of {what} are not non-negative whole numbers')


def _non_negative(fn: str, what: str, v: np.ndarray) -> None:
    if (v < 0).any():
        raise ValueError(f'{fn}: {_count(v < 0)} value(s) of {what} are negative')


def _match_and_weights(fn: str, match, w, n: int) -> tuple:
    """The match numbers (integers) and the weights (``None``: equal weights) of ``n`` rows. The match numbers are never printed."""
    match = np.asarray(match)
    _same_length(fn, 'match', match, n)
    if match.dtype.kind not in 'iu':
        raise TypeError(f'{fn}: the match numbers must be integers, not {match.dtype}')
    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=np.float64)
        _same_length(fn, 'w', w, n)
        bad = ~(np.isfinite(w) & (w >= 0))
        if bad.any():
            raise ValueError(f'{fn}: {_count(bad)} weight(s) are negative, NaN or infinite')
    if math.fsum(w) <= 0:
        raise ValueError(f'{fn}: the weights add up to 0')
    return match, w


# ---- the match bootstrap -----------------------------------------------------------------------------------------------------------------
def _matches(match) -> tuple:
    """(index of the match of every row in 0 .. M-1, with the matches in increasing order of their number; M)."""
    ids, inverse = np.unique(match, return_inverse=True)
    return inverse.reshape(-1), len(ids)


def _match_sums(inverse: np.ndarray, n_matches: int, columns) -> np.ndarray:
    """(len(columns), M) float64: the exactly rounded sum (``math.fsum``) of every column over the rows of every match. An exactly rounded
    sum does not depend on the order of the rows, so neither does anything computed from these sums.

    (GATE-1)의 경기별 합 ⟨f⟩_u = Σ_{i∈u} w_i·f_i를 열마다 계산한다(식은 occurrence_calibration에 있다).
    """
    order = np.argsort(inverse, kind='stable')
    bounds = np.searchsorted(inverse[order], np.arange(n_matches + 1))
    sums = np.empty((len(columns), n_matches))
    for j, column in enumerate(columns):
        ordered = column[order]
        for m in range(n_matches):
            sums[j, m] = math.fsum(ordered[bounds[m]:bounds[m + 1]])
    return sums


def _replicates(sums: np.ndarray, stat, n_out: int, B: int, seed: int) -> np.ndarray:
    """(B, n_out): ``stat`` of the sums of the j-th resample of the matches, j = 1 .. B (row j - 1 of the result). The counts of the resample are the j-th draw of
    ``default_rng(seed).multinomial(M, 1/M)``, one replicate at a time as in ``nll.match_bootstrap``, so that replicate j does not depend
    on B. ``sums`` rows are contiguous, and each total is a numpy sum of one row (no BLAS call).

    (GATE-1)의 복제 θ*_j = stat(Σ_u d_ju·⟨f⟩_u)를 j = 1, …, B에 대해 하나씩 계산한다(식은 occurrence_calibration에 있다).
    """
    n_matches = sums.shape[1]
    rng = np.random.default_rng(seed)
    pvals = np.full(n_matches, 1.0 / n_matches)
    out = np.empty((B, n_out))
    for b in range(B):
        counts = rng.multinomial(n_matches, pvals)
        out[b] = stat([float(np.sum(counts * row)) for row in sums])
    return out


def _interval(reps: np.ndarray) -> list:
    """The 2.5th and 97.5th percentiles of the replicates of one statistic: a list of two floats; NaN both if any replicate has no value.

    (GATE-1)의 구간 [Pct₂.₅, Pct₉₇.₅](θ*)이다(식은 occurrence_calibration에 있다).
    """
    if np.isnan(reps).any():
        return [math.nan, math.nan]
    lo, hi = np.percentile(reps, _PERCENTILES)
    return [float(lo), float(hi)]


def _line(W: float, Sx: float, Sy: float, Sxx: float, Sxy: float, x0: float, y0: float) -> tuple:
    """(slope, intercept) of the weighted least-squares line y = intercept + slope x, from the sums of the weights and of the weighted
    centred values x - x0, y - y0 (x0 and y0: the weighted means of the whole sample). NaN, NaN when the weights add up to 0 or the x do not
    vary. Centring leaves the slope unchanged; the intercept is brought back to the original x and y.

    수식
    ----
    (GATE-2)  R_i = γ + β·E_i + 오차의 가중 최소제곱 직선. 입력은 중심화한 값 Ẽ_i = E_i − E₀, R̃_i = R_i − R₀의 가중 합 ⟨f⟩이다.
              Ē = ⟨Ẽ⟩/⟨1⟩,   R̄ = ⟨R̃⟩/⟨1⟩,   Var_E = ⟨Ẽ²⟩/⟨1⟩ − Ē²
              β = (⟨ẼR̃⟩/⟨1⟩ − Ē·R̄) / Var_E,      γ = (R̄ + R₀) − β·(Ē + E₀)
              ⟨1⟩ > 0이고 Var_E > 10⁻¹²·⟨Ẽ²⟩/⟨1⟩일 때만 값이 있고, 아니면 (β, γ) = (NaN, NaN)이다.

      기호                      | 뜻                                                       | 단위          | 범위          | 코드 변수
      ⟨1⟩, ⟨Ẽ⟩, ⟨R̃⟩, ⟨Ẽ²⟩, ⟨ẼR̃⟩ | 가중치, 중심화한 값, 제곱, 곱의 가중 합 ((GATE-1)의 ⟨f⟩) | 열의 단위     | 유한          | W, Sx, Sy, Sxx, Sxy
      E₀, R₀                    | 전 표본의 가중 평균(중심)                                | E, R의 단위   | 유한          | x0, y0
      Ē, R̄                      | 중심화한 값의 가중 평균                                  | E, R의 단위   | 유한          | xm, ym
      Var_E                     | E의 가중 분산                                            | E 단위의 제곱 | Var_E > 0     | variance
      β, γ                      | 기울기와 절편                                            | R/E, R의 단위 | 유한 또는 NaN | slope, intercept

    해석
    ----
    식 (GATE-2)의 기울기 β는 E와 R의 가중 공분산을 E의 가중 분산으로 나눈 값이고 γ는 그 직선이 가중 평균 점을 지나게 하는 절편이다. 합을 전 표본의 가중 평균 E₀, R₀에서
    잰 편차로 모으므로 값이 0에서 멀 때 큰 두 수의 차로 분산을 구하지 않으며, 중심을 옮겨도 기울기는 그대로이고 절편만 원래 값으로 되돌린다. 가중치의 합이 0이면 직선이 없고, 분산이
    ⟨Ẽ²⟩/⟨1⟩의 10⁻¹²배 이하이면 E가 반올림 오차 안에서 변하지 않는 것이어서 그 몫을 기울기로 내지 않으므로 둘 다 NaN을 돌려준다.

    근거
    ----
    계획 P7 (i)·(ii)(관문의 보정 기울기), 검산 노트 math-control-03 F절(손 계산; 시험이 같은 수를 확인한다).
    """
    if not W > 0:
        return math.nan, math.nan
    # 식 (GATE-2)의 Ē, R̄와 Var_E. 분산이 ⟨Ẽ²⟩/⟨1⟩의 상대 문턱 이하이면 기울기를 만들지 않는다(NaN도 이 검사에서 걸린다).
    xm, ym = Sx / W, Sy / W
    second = Sxx / W
    variance = second - xm * xm
    if not variance > _DEGENERATE * second:                    # also true for a NaN
        return math.nan, math.nan
    # 식 (GATE-2)의 β와 γ. 중심화는 기울기를 바꾸지 않으므로 절편만 원래 E, R로 되돌린다.
    slope = (Sxy / W - xm * ym) / variance
    return slope, (ym + y0) - slope * (xm + x0)


def _decile_edges(x: np.ndarray) -> np.ndarray:
    """The edges of the ten bins of ``x``: the 10th, 20th, ... 90th percentiles, each as an order statistic of ``x``.

    The k-th percentile of ``numpy.quantile`` (linear interpolation) lies at the position k (n - 1) / 10 of the sorted values. When that is
    not a whole number the edge falls strictly between two order statistics (or on two equal ones), and the rows above it are the same
    wherever it falls there; so the edge is the upper one, at position ceil(k (n - 1) / 10), and no interpolation is needed. The ceiling is
    taken with integers: with float levels (0.1 x 3 is 0.30000000000000004) a position that is a whole number moves by one unit in the
    last place, and the row on that edge changes bins."""
    s = np.sort(x)
    return np.array([s[-(-k * (len(s) - 1) // _N_DECILES)] for k in range(1, _N_DECILES)])


def _deciles(x: np.ndarray, y: np.ndarray, w: np.ndarray, y_name: str) -> list:
    """The table of ten bins of the expected value ``x``: count, total weight and the weighted means of ``x`` and of ``y`` (under the key
    ``y_name``); the means are None for a bin without weight. A bin is half open, [edge, next edge): a row on an edge is in the upper.

    (GATE-3), (GATE-4)가 돌려주는 서술용 십분위 표이다. 구간마다 E와 R의 가중 평균을 내며 판정에 쓰지 않는다.
    """
    which = np.searchsorted(_decile_edges(x), x, side='right')      # equal values go to the same bin: they are never split
    table = []
    for k in range(_N_DECILES):
        rows = which == k
        weight = math.fsum(w[rows])
        seen = weight > 0
        table.append({'count': _count(rows), 'weight': weight,
                      'mean_expected': nll.weighted_mean(x[rows], w[rows]) if seen else None,
                      y_name: nll.weighted_mean(y[rows], w[rows]) if seen else None})
    return table


# ---- (i) occurrence calibration ----------------------------------------------------------------------------------------------------------
def occurrence_calibration(expected, observed, match, w, B=nll.BOOT_B, seed=nll.BOOT_SEED) -> dict:
    """Criterion (i): how well the expected 60 s occurrence counts of the rows agree with the observed counts.

    ``expected`` (finite, >= 0) and ``observed`` (non-negative whole numbers) are one value per row; ``match`` is the match number of each
    row (integers); ``w`` the weights (the gate passes 1/n_u). Returns a dict:

        oe                     sum(w observed) / sum(w expected), with ``oe_ci``
        slope, intercept       the weighted least-squares line observed = intercept + slope x expected, with ``slope_ci``, ``intercept_ci``
        deciles                ten bins of the expected value: count, weight, mean_expected, mean_observed
        n_rows, n_matches, B, seed

    The intervals are the percentile 95% intervals of ``B`` match-bootstrap replicates (see the module text). ValueError for values that
    are NaN or infinite, an expected value below 0, an observed value that is not a non-negative whole number, arrays of different lengths,
    weights that are negative or add up to 0, no rows, expected counts that add up to 0, and expected counts that do not vary (no slope);
    TypeError for match numbers that are not integers and for a ``B`` or ``seed`` that is not an integer.

    수식
    ----
    (GATE-1)  ⟨f⟩_u = Σ_{i∈u} w_i·f_i,   ⟨f⟩ = Σ_u ⟨f⟩_u,   ⟨f⟩*_j = Σ_u d_ju·⟨f⟩_u,   d_j ~ Multinomial(M; 1/M, …, 1/M)   (j = 1, …, B)
              θ̂ = stat(⟨f⟩),   θ*_j = stat(⟨f⟩*_j),   구간 = [Pct₂.₅, Pct₉₇.₅](θ*)      (θ*_j가 하나라도 NaN이면 양끝이 NaN)
              d_j는 default_rng(seed)의 한 난수열에서 복제마다 차례로 뽑으며 (NLL-5)와 같은 뽑기이고, B는 기본값 nll.BOOT_B이다.
              열 f는 1, Ẽ, R̃, Ẽ², ẼR̃(직선 (GATE-2))와 E, R(비 (GATE-3))의 일곱 개이며 코드의 sums 행 W, Sx, Sy, Sxx, Sxy, sum_e, sum_o이다.
    (GATE-3)  oe = ⟨R⟩ / ⟨E⟩   (⟨E⟩ > 0이 아니면 NaN),   기울기 β와 절편 γ는 (GATE-2)이다.
              기준 (i)(계획 P7): oe의 95% 구간이 [0.95, 1.05] 안에 들고 β의 95% 구간이 [0.9, 1.1] 안에 든다. 판정은 (GATE-7)이 한다.

      기호               | 뜻                                                 | 단위      | 범위                      | 코드 변수
      w_i, u, M          | 행 i의 가중치, 경기 번호, 경기 수 ((NLL-5)와 같다) | —         | w_i ≥ 0 유한; u = 1, …, M | w, match, n_matches
      E_i, R_i           | 행 i의 기대 발생 수와 관측 발생 수                 | 개        | 유한 ≥ 0; 0, 1, 2, …      | expected, observed
      ⟨f⟩_u, ⟨f⟩, ⟨f⟩*_j | 열 f의 경기 u 가중 합, 전 표본 합, 복제 j의 합     | 열의 단위 | 유한                      | sums의 행
      d_ju, j, B         | 복제 j에서 경기 u가 뽑힌 횟수, 복제 번호, 복제 수  | 회, —, —  | 0, …, M; 1, …, B; ≥ 1     | counts, b = j − 1, B
      stat, θ̂, θ*_j      | 합에서 통계량을 만드는 함수, 그 전 표본 값과 복제 j의 값 | 지표 단위 | 유한 또는 NaN             | stat, point, reps
      Pct₂.₅, Pct₉₇.₅    | 복제의 2.5, 97.5 백분위수(선형 보간)               | 지표 단위 | —                         | _interval
      oe                 | 관측 합 대 기대 합의 비                            | —         | 유한 또는 NaN             | oe

    해석
    ----
    식 (GATE-3)의 oe는 관측 발생 수의 가중 합을 기대 발생 수의 가중 합으로 나눈 비이고 β는 관측을 기대에 회귀한 보정 기울기이다. 둘 다 1이면 회귀 직선이 항등선이 되어 기대가 평균에서 맞고
    기대의 크기에 따른 선형 어긋남도 없지만(약한 보정), 기대의 크기별 비선형 어긋남까지 없다는 뜻은 아니며 그것은 서술용 십분위 표가 보인다. (GATE-1)은 경기를 균등하게 M번 복원 추출한 횟수 d_ju로 경기별 합을 다시 더해 같은 통계량을 다시 계산하므로 같은 경기 안 행들의 상관이 구간에 들어가고, 점추정은
    복제의 평균이 아니라 전 표본에서 같은 통계량으로 얻는다. 한 복제에서 기대값이 변하지 않으면 그 복제의 기울기는 NaN이고 그 구간의 양끝이 NaN이 되어 띠 안에 들지 못하며(공통 가정
    (나)), 전 표본에서 기대 합이 0이거나 기대값이 변하지 않으면 판정할 것이 없어 거부한다. 기대 발생 수가 0인 행도 직선의 한 점이므로 빼지 않는다(공통 가정 (다)).

    근거
    ----
    계획 P7 (i)(발생 보정과 동등성 띠), 판정 R26(모든 관문 구간이 같은 가중과 재표본), 설계서 3.1절(발생)과 8절(경기 부트스트랩 구간), 검산 노트 math-control-03 F절(손 계산; 시험이 같은
    수를 확인한다).
    Ref: Efron, B. (1979), Annals of Statistics 7(1): 1-26; doi:10.1214/aos/1176344552 (표본을 다시 뽑아 통계량의 분포를 얻는다).
         Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008), Review of Economics and Statistics 90(3): 414-427;
         doi:10.1162/rest.90.3.414 (군집을 통째로 다시 뽑는다).
    Ref: Steyerberg, E. W. et al. (2010), Epidemiology 21(1): 128-138; doi:10.1097/EDE.0b013e3181c30fb2 (보정 기울기, 관측 대 기대).
         Van Calster, B. et al. (2016), Journal of Clinical Epidemiology 74: 167-176; doi:10.1016/j.jclinepi.2015.12.005 (보정의 위계).
    """
    fn = 'occurrence_calibration'
    B, seed = _whole(fn, 'B', B, 1), _whole(fn, 'seed', seed, 0)
    e, o = _vector(fn, 'expected', expected), _vector(fn, 'observed', observed)
    _same_length(fn, 'observed', o, len(e))
    _non_negative(fn, 'expected', e)
    _counts(fn, 'observed', o)
    match, w = _match_and_weights(fn, match, w, len(e))
    # 식 (GATE-2)의 중심 E₀, R₀: 전 표본의 가중 평균이며 모든 복제에서 같은 값을 쓴다.
    x0, y0 = nll.weighted_mean(e, w), nll.weighted_mean(o, w)             # the centres of the second moments
    xc, yc = e - x0, o - y0
    inverse, n_matches = _matches(match)
    sums = _match_sums(inverse, n_matches, [w, w * xc, w * yc, w * xc * xc, w * xc * yc, w * e, w * o])

    # stat는 합에서 식 (GATE-3)의 oe와 식 (GATE-2)의 β, γ를 만든다: 전 표본에는 정확한 합 ⟨f⟩를, 복제에는 재표본 합 ⟨f⟩*_j를 넣는다.
    def stat(t):
        W, Sx, Sy, Sxx, Sxy, sum_e, sum_o = t
        slope, intercept = _line(W, Sx, Sy, Sxx, Sxy, x0, y0)
        return (sum_o / sum_e if sum_e > 0 else math.nan, slope, intercept)

    # 점추정은 전 표본의 합에서 같은 stat로 얻는다. 판정할 값이 없으면 구간을 만들지 않고 거부한다.
    point = stat([math.fsum(row) for row in sums])
    if math.isnan(point[0]):
        raise ValueError(f'{fn}: the expected counts add up to 0, so observed / expected has no value')
    if math.isnan(point[1]):
        raise ValueError(f'{fn}: the expected counts do not vary, so the line has no slope')
    # 식 (GATE-1)의 복제 θ*_j, j = 1, …, B.
    reps = _replicates(sums, stat, 3, B, seed)
    return {'oe': float(point[0]), 'oe_ci': _interval(reps[:, 0]),
            'slope': float(point[1]), 'slope_ci': _interval(reps[:, 1]),
            'intercept': float(point[2]), 'intercept_ci': _interval(reps[:, 2]),
            'deciles': _deciles(e, o, w, 'mean_observed'),
            'n_rows': int(len(e)), 'n_matches': int(n_matches), 'B': B, 'seed': seed}


# ---- (ii) value calibration --------------------------------------------------------------------------------------------------------------
def value_calibration(expected_v, realized_v, match, w, B=nll.BOOT_B, seed=nll.BOOT_SEED) -> dict:
    """Criterion (ii): how well E[V_start | t0] (``expected_v``) agrees with the realized V_start (``realized_v``), in gold, per row.

    ``match`` and ``w`` as in ``occurrence_calibration``. Returns a dict:

        slope                  the slope of the weighted least-squares line realized = intercept + slope x expected, with ``slope_ci``
        bias                   the weighted mean error sum(w (realized - expected)) / sum(w), with ``bias_ci``
        deciles                ten bins of the expected value: count, weight, mean_expected, mean_realized
        n_rows, n_matches, B, seed

    ValueError and TypeError as in ``occurrence_calibration`` (the values may have any sign and need not be whole numbers).

    수식
    ----
    (GATE-4)  β: (GATE-2)의 기울기 (E_i = 행 i의 t₀ 조건부 기대 V_start, R_i = 실현 V_start)
              bias = ⟨R − E⟩ / ⟨1⟩ = Σ_i w_i·(R_i − E_i) / Σ_i w_i      (⟨1⟩ > 0이 아니면 NaN)
              기준 (ii)(계획 P7): β의 95% 구간이 [0.8, 1.2] 안에 들고 bias의 95% 구간이 [−50, 50] 골드 안에 든다. 판정은 (GATE-7)이 한다.
              구간은 (GATE-1)이고 w_i, u, M, ⟨f⟩, d_ju, j, B는 occurrence_calibration과 같다.

      기호     | 뜻                                           | 단위 | 범위            | 코드 변수
      E_i, R_i | 행 i의 t₀ 조건부 기대 V_start와 실현 V_start | 골드 | 유한, 부호 있음 | expected_v, realized_v
      bias     | 실현에서 기대를 뺀 가중 평균 오차            | 골드 | 유한 또는 NaN   | bias

    해석
    ----
    식 (GATE-4)의 β는 실현 가치를 합성 기대 가치에 회귀한 기울기이고 1이면 기대가 1골드 클 때 실현도 평균 1골드 크며, bias는 실현에서 기대를 뺀 가중 평균 오차이고 0이면 평균 높이가 맞는다.
    기울기는 기대의 크기에 따른 선형 어긋남을, bias는 전체의 높이를 재므로 둘 다 띠 안이어야 통과한다(계획 P7 (ii)). 값은 골드이고 부호가 있으며 정수일 필요가 없다. 기대값이 변하지 않으면
    기울기가 없어 전 표본에서는 거부하고 한 복제에서는 그 구간의 양끝이 NaN이 되며(공통 가정 (나)), 복제에서 가중치의 합이 0이면 bias도 NaN이다.

    근거
    ----
    계획 P7 (ii)(가치 보정과 동등성 띠), 판정 R26(모든 관문 구간이 같은 가중과 재표본), 설계서 3.4절(V_start)과 8절(합성 가치), 검산 노트 math-control-03 F절(손 계산; 시험이 같은
    수를 확인한다). 논문은 occurrence_calibration의 Ref와 같다.
    """
    fn = 'value_calibration'
    B, seed = _whole(fn, 'B', B, 1), _whole(fn, 'seed', seed, 0)
    e, v = _vector(fn, 'expected_v', expected_v), _vector(fn, 'realized_v', realized_v)
    _same_length(fn, 'realized_v', v, len(e))
    match, w = _match_and_weights(fn, match, w, len(e))
    # 식 (GATE-2)의 중심 E₀, R₀와 식 (GATE-1)의 경기별 합 여섯 열 ⟨1⟩, ⟨Ẽ⟩, ⟨R̃⟩, ⟨Ẽ²⟩, ⟨ẼR̃⟩와 ⟨R − E⟩(식 (GATE-4)의 bias의 분자).
    x0, y0 = nll.weighted_mean(e, w), nll.weighted_mean(v, w)
    xc, yc = e - x0, v - y0
    inverse, n_matches = _matches(match)
    sums = _match_sums(inverse, n_matches, [w, w * xc, w * yc, w * xc * xc, w * xc * yc, w * (v - e)])

    # stat는 합에서 식 (GATE-2)의 β와 식 (GATE-4)의 bias = ⟨R − E⟩/⟨1⟩를 만든다.
    def stat(t):
        W, Sx, Sy, Sxx, Sxy, Sd = t
        return (_line(W, Sx, Sy, Sxx, Sxy, x0, y0)[0], Sd / W if W > 0 else math.nan)

    point = stat([math.fsum(row) for row in sums])
    if math.isnan(point[0]):
        raise ValueError(f'{fn}: the expected values do not vary, so the line has no slope')
    reps = _replicates(sums, stat, 2, B, seed)
    return {'slope': float(point[0]), 'slope_ci': _interval(reps[:, 0]),
            'bias': float(point[1]), 'bias_ci': _interval(reps[:, 1]),
            'deciles': _deciles(e, v, w, 'mean_realized'),
            'n_rows': int(len(e)), 'n_matches': int(n_matches), 'B': B, 'seed': seed}


# ---- (iv) the simulation against the closed form -----------------------------------------------------------------------------------------
def mc_agreement(mc_mean, mc_sd, S, analytic) -> dict:
    """Criterion (iv): the share of rows whose simulated mean lies within 3 standard errors of the closed-form value.

    ``mc_mean`` and ``mc_sd`` are the mean and the standard deviation of the ``S`` simulated values of each row (``S``: a positive integer),
    ``analytic`` the closed-form expectation. A row agrees when |mc_mean - analytic| < 3 mc_sd / sqrt(S), strictly. A row whose standard
    deviation is 0 has no simulation error, so it agrees only when the two numbers are equal. Returns a dict:

        share                  agreeing rows / rows
        n_rows, n_agree
        worst_z                the largest |mc_mean - analytic| / (mc_sd / sqrt(S)) over the rows with a standard deviation above 0
                               (None when there is none; it is only for reading, the share is judged by the strict rule above)
        n_sd_zero              the number of rows whose standard deviation is 0
        n_sd_zero_disagree     how many of them differ from the closed form
        S

    ValueError for values that are NaN or infinite, a negative standard deviation, arrays of different lengths or no rows; TypeError for an
    ``S`` that is not an integer (a bool is not), ValueError for an ``S`` below 1.

    수식
    ----
    (GATE-5)  SE_i = sd_i / √S,     일치_i = [ |V̂_i − E_i| < 3·SE_i ]   (sd_i > 0),      일치_i = [ V̂_i = E_i ]   (sd_i = 0)
              share = (1/n)·Σ_i 일치_i,      worst_z = max_{i: sd_i > 0} |V̂_i − E_i| / SE_i      (sd_i > 0인 행이 없으면 없음)
              기준 (iv)(계획 P7): S = 1,000에서 행의 99% 이상이 |V̂_i − E_i| < 3·SE_i를 만족한다, 곧 share ≥ 0.99. 판정은 (GATE-7)이 한다.

      기호     | 뜻                                         | 단위  | 범위        | 코드 변수
      V̂_i      | 행 i의 모의 평균 ((COMP-1)의 V̂(H))         | 골드  | 유한        | mc_mean
      sd_i     | 행 i의 모의 표준편차 ((COMP-1)의 sd(H))    | 골드  | ≥ 0 유한    | mc_sd
      E_i      | 행 i의 닫힌 꼴 기대값 ((COMP-6)의 E[V(H)]) | 골드  | 유한        | analytic
      S        | 행마다의 모의 횟수 ((COMP-1)과 같다)       | 회    | 정수 ≥ 1    | S
      SE_i     | 모의 평균의 표준오차                       | 골드  | ≥ 0         | sd / root (root = √S)
      share, n | 일치한 행의 비율, 행의 수                  | —, 행 | [0, 1]; ≥ 1 | share, n_rows

    해석
    ----
    식 (GATE-5)는 모의 평균이 닫힌 꼴 기대값에서 모의 표준오차의 3배보다 가까운 행의 비율이며, 두 길이 같은 머리와 같은 가격을 쓰므로 둘의 차는 모의 오차뿐이어야 한다(계획 P7 (iv)).
    차가 3 SE_i보다 엄격하게 작아야 일치이고, 모의 표준편차가 0인 행은 모의 오차가 없으므로 두 수가 같을 때만 일치하여 0으로 나누지 않는다. worst_z는 읽기 위한 값이고 판정은
    share로만 한다.

    근거
    ----
    계획 P7 (iv)(모의와 닫힌 꼴의 일치), 계획 P6(닫힌 꼴로 모의를 검산한다), 검산 노트 math-control-03 E절(닫힌 꼴; 시험이 같은 수를 확인한다).
    """
    fn = 'mc_agreement'
    mean, sd, ana = _vector(fn, 'mc_mean', mc_mean), _vector(fn, 'mc_sd', mc_sd), _vector(fn, 'analytic', analytic)
    _same_length(fn, 'mc_sd', sd, len(mean))
    _same_length(fn, 'analytic', ana, len(mean))
    _non_negative(fn, 'mc_sd', sd)
    S = _whole(fn, 'S', S, 1)
    root = math.sqrt(S)
    diff = np.abs(mean - ana)
    spread = sd > 0
    # 식 (GATE-5)의 일치_i: sd_i > 0이면 차가 3·SE_i보다 엄격하게 작을 때, sd_i = 0이면 두 수가 같을 때(모의 오차가 없어 0으로 나누지 않는다).
    agree = np.where(spread, diff < _MC_Z * sd / root, diff == 0.0)
    n_rows, n_agree = int(len(mean)), _count(agree)
    # 식 (GATE-5)의 worst_z: 읽기용 값이다(판정은 share).
    worst = float(np.max(diff[spread] / (sd[spread] / root))) if spread.any() else None
    return {'share': n_agree / n_rows, 'n_rows': n_rows, 'n_agree': n_agree, 'worst_z': worst,
            'n_sd_zero': _count(~spread), 'n_sd_zero_disagree': _count(~spread & ~agree), 'S': S}


# ---- P8 the over-dispersion test ---------------------------------------------------------------------------------------------------------
def overdispersion(y, mu, match, w=None, B=nll.BOOT_B, seed=nll.BOOT_SEED) -> dict:
    """P8: the Pearson ratio phi = sum(w (y - mu)^2 / mu) / sum(w) of the counts ``y`` (non-negative whole numbers) against their expected
    counts ``mu`` (>= 0), one value per row. phi is 1 on average when the counts are Poisson with mean mu. ``match`` and ``w`` as in
    ``occurrence_calibration``; ``w`` defaults to equal weights (the gate passes 1/n_u).

    Rows with ``mu`` = 0 cover no time; they must have ``y`` = 0 and are left out and counted (ruling R22; their weights are dropped), and
    a match without any other row is not resampled. A count where ``mu`` is 0 raises. Returns a dict:

        phi, phi_ci            the ratio and its percentile 95% interval over ``B`` match-bootstrap replicates
        overdispersed          True when the lower end of ``phi_ci`` is above ``DISPERSION_LIMIT`` (P8: S4 then changes M2 to a negative
                               binomial or an interval-by-interval hazard)
        n_rows                 the rows in the ratio (those with mu > 0)
        n_zero_mu              the rows left out because mu = 0
        n_matches              the matches of those rows
        B, seed

    ValueError for values that are NaN or infinite, ``y`` that is not a non-negative whole number, a negative ``mu``, a count where
    ``mu`` = 0, arrays of different lengths, weights that are negative or add up to 0, no row with ``mu`` > 0, and a ratio term that
    overflows; TypeError for match numbers that are not integers and for a ``B`` or ``seed`` that is not an integer.

    수식
    ----
    (GATE-6)  χ²_i = (y_i − μ_i)² / μ_i,      φ = Σ_i w_i·χ²_i / Σ_i w_i      (μ_i > 0인 행 i만)
              구간은 (GATE-1)에서 stat = ⟨χ²⟩ / ⟨1⟩ (⟨1⟩ > 0이 아니면 NaN)로 만든다. μ_i = 0인 행은 y_i = 0이어야 하고 빼서 센다.
              P8: φ의 95% 구간의 아래끝이 1.1을 넘으면 과산포로 본다. y_i ~ Poisson(μ_i)이면 E[χ²_i] = 1이다 [도출: 분산 = 평균].

      기호 | 뜻                                       | 단위 | 범위       | 코드 변수
      y_i  | 행 i에서 센 개수(관문은 60초 안 발생 수) | 개   | 0, 1, 2, … | y
      μ_i  | 행 i의 기대 개수                         | 개   | 유한 ≥ 0   | mu
      χ²_i | 행 i의 피어슨 카이제곱 항 (y_i − μ_i)²/μ_i(피어슨 잔차의 제곱) | — | 유한 ≥ 0 | z
      φ    | 피어슨 분산비의 가중 평균                | —    | 유한 ≥ 0   | phi

    해석
    ----
    식 (GATE-6)의 φ는 피어슨 통계량의 가중 평균이며, μ_i가 참 조건부 평균이고 분산이 평균과 같은 포아송이면 기대값이 1이고 과산포이면 1보다 커진다 [도출]. μ_i가 참 조건부 평균에서 어긋나면
    E[χ²_i] = (Var(y_i) + (E[y_i] − μ_i)²)/μ_i이어서 φ에 평균의 어긋남이 함께 실리므로, 한계를 넘는 φ를 과산포로 읽는 것은 평균이 맞을 때이다 [도출].
    판정은 φ의 점추정이 아니라 경기 부트스트랩 구간의 아래끝이 1.1을 넘는지이며, 0.1의 여유는 큰 표본에서 사소한 과산포까지 잡지 않으려는 것이고, 과산포로 보면 S4의 M2를 음이항이나 구간 순차 hazard로
    바꾸는 안을 낸다(계획 P8). 기대 개수가 0인 행은 시간을 덮지 않아 정보가 없으므로 개수가 0일 때만 받아 빼고 세며, 그 가중치는 남은 행에 주지 않고 남은 행이 없는 경기는 다시 뽑지 않는다(판정 R22).
    μ_i가 아주 작으면 χ²_i가 넘칠 수 있으므로 유한하지 않은 항이 있으면 거부한다.

    근거
    ----
    계획 P8(피어슨 분산비와 한계), 판정 R22(정보 없는 행은 빼고 센다), 판정 R26(같은 가중과 재표본), 검산 노트 math-control-03 G절(손 계산; 시험이 같은 수를 확인한다).
    Ref: Cameron, A. C. and Trivedi, P. K. (2013), "Regression Analysis of Count Data", 2nd ed., Cambridge University Press;
         doi:10.1017/CBO9781139013567 (피어슨 통계량으로 개수의 분산도를 잰다).
    """
    fn = 'overdispersion'
    B, seed = _whole(fn, 'B', B, 1), _whole(fn, 'seed', seed, 0)
    y, mu = _vector(fn, 'y', y), _vector(fn, 'mu', mu)
    _same_length(fn, 'mu', mu, len(y))
    _counts(fn, 'y', y)
    _non_negative(fn, 'mu', mu)
    match, w = _match_and_weights(fn, match, w, len(y))
    # 식 (GATE-6)의 정의역: μ_i = 0인 행은 시간을 덮지 않으므로 y_i = 0이어야 하고(아니면 자료 오류) 빼서 센다.
    zero = mu == 0.0
    if (zero & (y > 0)).any():
        raise ValueError(f'{fn}: {_count(zero & (y > 0))} row(s) have a count but an expected count of 0')
    keep = ~zero
    if not keep.any():
        raise ValueError(f'{fn}: no row has a positive expected count')
    y, mu, w, match = y[keep], mu[keep], w[keep], match[keep]
    if math.fsum(w) <= 0:
        raise ValueError(f'{fn}: the weights of the rows with a positive expected count add up to 0')
    # 식 (GATE-6)의 χ²_i (코드의 z).
    z = (y - mu) ** 2 / mu
    if not np.isfinite(z).all():
        raise ValueError(f'{fn}: {_count(~np.isfinite(z))} term(s) of the ratio are not finite')
    inverse, n_matches = _matches(match)
    sums = _match_sums(inverse, n_matches, [w, w * z])

    # stat는 합 ⟨1⟩, ⟨χ²⟩에서 식 (GATE-6)의 φ = ⟨χ²⟩/⟨1⟩를 만든다.
    def stat(t):
        W, Sz = t
        return (Sz / W if W > 0 else math.nan,)

    phi = stat([math.fsum(row) for row in sums])[0]
    phi_ci = _interval(_replicates(sums, stat, 1, B, seed)[:, 0])
    return {'phi': float(phi), 'phi_ci': phi_ci, 'overdispersed': bool(phi_ci[0] > DISPERSION_LIMIT),
            'n_rows': int(len(y)), 'n_zero_mu': int(_count(zero)), 'n_matches': int(n_matches), 'B': B, 'seed': seed}


# ---- the verdict -------------------------------------------------------------------------------------------------------------------------
def _entry(name: str, results) -> Mapping:
    entry = results[name]
    if not isinstance(entry, Mapping):
        raise ValueError(f'verdict: the {name!r} result must be a dict, not {type(entry).__name__}')
    return entry


def _number(name: str, entry: Mapping, key: str, required: bool = True):
    """The float ``entry[key]``; ValueError if it is missing (None when not required) or not a number."""
    if key not in entry:
        if required:
            raise ValueError(f'verdict: the {name!r} result has no {key!r}')
        return None
    try:
        return float(entry[key])
    except (TypeError, ValueError):
        raise ValueError(f'verdict: {key!r} of the {name!r} result is not a number') from None


def _interval_of(name: str, entry: Mapping, key: str) -> list:
    """The interval ``entry[key]`` as a list of two floats (low, high); ValueError for anything else or a low end above the high end."""
    if key not in entry:
        raise ValueError(f'verdict: the {name!r} result has no {key!r}')
    try:
        lo, hi = (float(v) for v in entry[key])
    except (TypeError, ValueError):
        raise ValueError(f'verdict: {key!r} of the {name!r} result must be two numbers (low, high)') from None
    if lo > hi:                                  # False for a NaN: an interval that could not be computed is judged, not refused
        raise ValueError(f'verdict: {key!r} of the {name!r} result has its low end above its high end')
    return [lo, hi]


def _inside(interval: list, band) -> bool:
    """True when the whole interval lies inside the closed band; False for a NaN.

    (GATE-7)의 내부([lo, hi], 띠)이다.
    """
    return bool(band[0] <= interval[0] and interval[1] <= band[1])


def _occurrence_verdict(r: Mapping) -> dict:
    """(GATE-7)의 기준 (i): oe와 β의 구간이 각자의 띠 안에 드는가."""
    oe_ci, slope_ci = _interval_of('occurrence', r, 'oe_ci'), _interval_of('occurrence', r, 'slope_ci')
    oe_in, slope_in = _inside(oe_ci, OE_BAND), _inside(slope_ci, OCC_SLOPE_BAND)
    return {'pass': oe_in and slope_in,
            'oe': _number('occurrence', r, 'oe', False), 'oe_ci': oe_ci, 'oe_band': list(OE_BAND), 'oe_in_band': oe_in,
            'slope': _number('occurrence', r, 'slope', False), 'slope_ci': slope_ci, 'slope_band': list(OCC_SLOPE_BAND),
            'slope_in_band': slope_in}


def _value_verdict(r: Mapping) -> dict:
    """(GATE-7)의 기준 (ii): β와 bias의 구간이 각자의 띠 안에 드는가."""
    slope_ci, bias_ci = _interval_of('value', r, 'slope_ci'), _interval_of('value', r, 'bias_ci')
    slope_in, bias_in = _inside(slope_ci, VALUE_SLOPE_BAND), _inside(bias_ci, BIAS_BAND)
    return {'pass': slope_in and bias_in,
            'slope': _number('value', r, 'slope', False), 'slope_ci': slope_ci, 'slope_band': list(VALUE_SLOPE_BAND),
            'slope_in_band': slope_in,
            'bias': _number('value', r, 'bias', False), 'bias_ci': bias_ci, 'bias_band': list(BIAS_BAND), 'bias_in_band': bias_in}


def _accounting_verdict(r: Mapping) -> dict:
    """(GATE-7)의 기준 (iii): 회계 항등식의 개수 수준 오차 max_abs_error가 정확히 0인가."""
    error = _number('accounting', r, 'max_abs_error')
    return {'pass': bool(error == 0.0), 'max_abs_error': error, 'must_equal': 0.0}      # exactly 0 (a NaN is not)


def _mc_verdict(r: Mapping) -> dict:
    """(GATE-7)의 기준 (iv): 일치한 행의 비율 share가 MC_SHARE 이상인가."""
    share = _number('mc', r, 'share')
    if not 0.0 <= share <= 1.0:
        raise ValueError("verdict: 'share' of the 'mc' result must lie in [0, 1]")
    return {'pass': bool(share >= MC_SHARE), 'share': share, 'min_share': float(MC_SHARE)}


def _overdispersion_verdict(r: Mapping) -> dict:
    """(GATE-6)의 보고 항목: φ의 구간 아래끝이 DISPERSION_LIMIT을 넘는가. 판정 항목이 아니다((GATE-7))."""
    phi_ci = _interval_of('overdispersion', r, 'phi_ci')
    return {'overdispersed': bool(phi_ci[0] > DISPERSION_LIMIT), 'phi': _number('overdispersion', r, 'phi', False), 'phi_ci': phi_ci,
            'limit': float(DISPERSION_LIMIT)}


_VERDICTS = {'occurrence': _occurrence_verdict, 'value': _value_verdict, 'accounting': _accounting_verdict, 'mc': _mc_verdict,
             'overdispersion': _overdispersion_verdict}


def verdict(results) -> dict:
    """The pass or fail of every gate criterion, with the numbers it was judged on and the bands (P7, P8).

    ``results`` is a dict that may hold any of the keys

        'occurrence'      the dict of ``occurrence_calibration``: passes when ``oe_ci`` lies inside ``OE_BAND`` and ``slope_ci`` inside
                          ``OCC_SLOPE_BAND``
        'value'           the dict of ``value_calibration``: passes when ``slope_ci`` lies inside ``VALUE_SLOPE_BAND`` and ``bias_ci``
                          inside ``BIAS_BAND``
        'accounting'      a dict with ``max_abs_error`` (``value.accounting()``): passes when that is exactly 0 (the entry says
                          ``must_equal``: 0.0)
        'mc'              the dict of ``mc_agreement``: passes when ``share`` >= ``MC_SHARE``
        'overdispersion'  the dict of ``overdispersion``: not a pass or fail item; the entry reports ``overdispersed``, True when the lower
                          end of ``phi_ci`` is above ``DISPERSION_LIMIT``

    A band is closed (an interval equal to the band is inside) and the whole interval has to be inside; an interval with a NaN end is not
    inside. The point estimates are shown but not judged. What is judged comes from the numbers and the module constants, never from a
    ``pass`` flag in the input. Returns a dict with one entry per key given (each with ``pass`` except the over-dispersion entry, and with
    the numbers, intervals and bands as plain lists and floats) and

        all_pass    True when every gate criterion that is present passes and at least one is present (the over-dispersion entry does
                    not count; nothing judged is no pass)
        missing     the gate criteria that ``results`` does not have, in the order above; ``all_pass`` says nothing about them, so a gate
                    is complete only when this list is empty

    TypeError for a ``results`` that is not a dict; ValueError for a key that is not one of the five (a typo must not turn into a
    "missing" criterion without a word), an entry that is not a dict, or an entry without the numbers it is judged on.

    수식
    ----
    (GATE-7)  내부([lo, hi], 띠) ⇔ 띠의 아래끝 ≤ lo 그리고 hi ≤ 띠의 위끝                    (닫힌 띠; lo나 hi가 NaN이면 거짓)
              통과 (i)   = 내부(oe_ci, OE_BAND) ∧ 내부(slope_ci, OCC_SLOPE_BAND)            (띠는 (GATE-3)에 쓴 값이다)
              통과 (ii)  = 내부(slope_ci, VALUE_SLOPE_BAND) ∧ 내부(bias_ci, BIAS_BAND)      (띠는 (GATE-4)에 쓴 값이다)
              통과 (iii) = [max_abs_error = 0]                  (계획 P7 (iii)의 오차 0을 판정 R11이 개수 수준으로 읽는다; 오차의 식은 (VAL-9))
              통과 (iv)  = [share ≥ MC_SHARE]                   (한계는 (GATE-5)에 쓴 값이다)
              all_pass = (판정한 기준이 하나 이상) ∧ (판정한 기준이 모두 통과)       (분산비 (GATE-6)은 판정하지 않고 overdispersed만 보고한다)

      기호                        | 뜻                           | 단위      | 범위             | 코드 변수
      [lo, hi]                    | 95% 구간의 아래끝과 위끝     | 지표 단위 | lo ≤ hi 또는 NaN | 'oe_ci', 'slope_ci', 'bias_ci'
      OE_BAND, OCC_SLOPE_BAND     | 발생 보정의 띠               | —, —      | 아래끝 < 위끝    | 모듈 상수
      VALUE_SLOPE_BAND, BIAS_BAND | 가치 보정의 띠               | —, 골드   | 아래끝 < 위끝    | 모듈 상수
      MC_SHARE                    | 모의 일치 비율의 하한        | —         | [0, 1]           | MC_SHARE
      max_abs_error               | 회계 항등식의 개수 수준 오차 | 개        | ≥ 0 또는 NaN     | results['accounting']

    해석
    ----
    식 (GATE-7)의 기준 (i)·(ii)는 점추정이 아니라 95% 구간이 미리 잠근 닫힌 띠 안에 통째로 드는지를 보는 동등성 판정이고, (iii)과 (iv)는 수치를 정해진 값과 맞댄다. 띠는 닫혀 있어 구간이 띠와
    같아도 안에 들고 끝이 NaN인 구간은 들지 못한다.
    구간이 1을 포함하는지를 묻지 않는 것은 DEV 경기가 많으면 패치 사이의 작은 표류에도 그 기준이 깨지기 때문이다(계획 P7; 지침 F3).
    회계 항등식 (iii)은 계획 P7 (iii)의 오차 0을 개수 수준에서 읽어 max_abs_error가 정확히 0일 때만 통과하고(판정 R11),
    골드 수준의 오차 max_abs_value_error는 이 판정에 넣지 않고 보고서가 판정 R48의 허용오차로 판정 곁에 보인다(저자가 G-S3c 결정 C2로 이 읽기를 승인하였다).

    근거
    ----
    계획 P7(관문 기준과 사전 고정 띠), P8(분산비는 보고 항목), 판정 R11(회계 항등식은 개수에서 정확히 맞는다), R48(골드 수준은 보고서가 보인다), 지침 F3(결과 전에 잠근다).
    Ref: Schuirmann, D. J. (1987), J. Pharmacokinet. Biopharm. 15(6): 657-680; doi:10.1007/BF01068419 (구간이 사전 고정 띠 안이면 동등하다고 본다).
         Lakens, D. (2017), Soc. Psychol. Personal. Sci. 8(4): 355-362; doi:10.1177/1948550617697177 (동등성 검정 입문).
    """
    if not isinstance(results, Mapping):
        raise TypeError(f'verdict: results must be a dict, not {type(results).__name__}')
    unknown = sorted(str(k) for k in results if k not in _VERDICTS)
    if unknown:
        raise ValueError(f'verdict: unknown result key(s) {unknown}; the keys are {list(_VERDICTS)}')
    # 식 (GATE-7): 기준마다 입력의 수치를 띠와 맞댄다(입력의 pass 표시는 읽지 않는다).
    out = {name: _VERDICTS[name](_entry(name, results)) for name in GATE_CRITERIA + REPORTED if name in results}
    # 식 (GATE-7)의 all_pass: 판정한 기준이 하나 이상이고 모두 통과해야 참이다. 분산비는 out에 있어도 이 목록에 넣지 않는다.
    present = [name for name in GATE_CRITERIA if name in out]
    out['all_pass'] = bool(present) and all(out[name]['pass'] for name in present)
    out['missing'] = [name for name in GATE_CRITERIA if name not in out]
    return out
