"""The composed V_start pass on a sample of STOP matches: the sample, the Monte Carlo losses and their error, the CRPS on both populations, the D14 excluded-row share, the occurrence
calibration and the A7 paired contrast (S4b-prep K6e; decisions D9, D14, D30; K6b review I3 item 4).

What it is for. D30 restricts the composed (Monte Carlo) V_start to two S4b checks, both made on a sample of whole STOP matches with the Monte Carlo standard error reported: A7, the V_start CRPS
gain of the zone mark (part of D9), and D14, the share of the CRPS that the rows left out of the primary population carry. The size of the sample and the number of draws are parameters of the
evaluate profile (``composed``), set by a later, separately approved cost probe; nothing here is a constant of that kind. ``run.evaluate`` builds the Composer chains (it holds the model and
the heavy objects) and hands them to the functions here; this module holds the arithmetic, so that it can be tested without a model.

Symbols
-------
    P          the pool of the STOP matches whose every frame row has a finite realised V_start; N the sample size; seed_N the seed of the sample
    A          the sample, A = sort(P[ choice(|P|, N, without replacement) ]), drawn by ``np.random.default_rng(seed_N)``; h_A its id hash
    r, m(r)    a row of the sample and its match;  n_m the rows of match m (in the population that is scored)
    H, j       the horizon in seconds (60, 120, 180) and its column (0, 1, 2)
    S, K, s    the draws of every row, the batches, s = S / K the draws of one batch
    V_r        (S, 3) the simulated V_start of the row r, one row per draw;  y_r (3,) the realised V_start
    Q_r        (3, 19) the quantiles of V_r at the levels tau_k = k/20 (``compose.Q_LEVELS``); Q_r^(b) the same from the batch b only (draws b s .. (b+1) s - 1)
    l_r(H)     the row loss, 2 x the mean over the 19 levels of the pinball loss of y_r(H) against Q_r(H)   (the arithmetic of ``proper.crps19`` for one row)
    L          the row loss of the pass that scores the row: the nominal composer for the primary rows, the observed composer for the others (D14)
    P_H, E_H   the primary rows of horizon H (t0 + H <= T - 30 s) and the excluded ones, E_H = rows \\ P_H

Equations
---------
    (VS-1)  l_r(H) = (2/19) sum_k rho_tau_k( y_r(H) - Q_r(H)_k ),   rho_tau(e) = tau e (e >= 0), (tau - 1) e (e < 0)
    (VS-2)  CRPS(P_H) = sum_{r in P_H} w_r l_r(H) / sum_{r in P_H} w_r,    w_r = 1 / n_m(r) counted inside P_H           (the weights of ``proper.crps19``, D15)
    (VS-3)  share(H) = sum_{r in E_H} u_r L_r(H) / sum_r u_r L_r(H),   u_r = 1 / n_m(r) counted over all rows of the sample             (D14)
    (VS-4)  MC error of a statistic T:   T^(b) from the losses of the batch b,  se_MC(T) = sd_b( T^(b) ) / sqrt(K)                      (batch means over the draws)
    (VS-5)  bootstrap of T (a ratio sum_m a_m / sum_m d_m over matches):  T*_b = sum_m c_{b,m} a_m / sum_m c_{b,m} d_m,  c = the first B rows of the common resample matrix of the STOP split
    (VS-6)  A7 contrast:   Delta(H) = mean_s [ CRPS_on,s(P_H) - CRPS_off,s(P_H) ]   on the same rows, the same weights and the same streams;  below 0 favours the zone mark
    (VS-7)  occurrence CITL:   CITL(H) = sum_m (1/n_m) sum_{r in m} (p_r - o_r) / (number of present matches),  p the composed 1 - exp(-E N(H)),  o the observed occurrence

해석
----
합성 V_start는 비싸서 STOP 전체가 아니라 표본에서만 한다(D30). 표본은 값이 매겨진 STOP 경기 가운데에서 시드로 뽑고 번호 목록의 해시를 남긴다. 한 행의 V_start는 S번 모의한 값의 분위수(19개)이고, 점수는 실현값에 대한
핀볼 손실의 2배 평균(CRPS 대리값)이다. 몬테카를로 오차는 S번의 모의를 K개의 같은 크기 배치로 나누어 배치마다 같은 통계를 다시 구하고, 그 표준편차를 √K로 나눈 값이다(배치 평균). 한 배치의 모의 수는 S/K이므로 배치
통계는 S회 통계보다 분위수 잡음이 크고 조금 치우치지만, 잡음의 분산은 모의 수에 반비례하므로 √K로 나눈 값이 S회 통계의 몬테카를로 표준오차에 대한 추정이다(S/K와 K는 둘 다 작지 않아야 믿을 만하다). 행의 난수 흐름은 [시드,
프레임 행]이어서 구역 표지를 켠 모형과 끈 모형에 같은 흐름이 쓰이고(공통 난수), 두 점수의 차이에서 몬테카를로 잡음이 상당 부분 상쇄된다. 경기 표본의 불확실성(어느 경기가 뽑혔는가)은 이와 다른 것이어서 경기 부트스트랩(판정 분할
전체의 공통 재표본 행렬의 앞 B행, 표본에 없는 경기는 분모에서 빠진다)으로 따로 보고한다. A7의 대조는 두 기록의 같은 행에서 구역 켬 − 끔의 차이이며, 판정(한계, 이득의 크기)은 이 모듈이 하지 않는다.

# Ref: Gneiting, T. and Raftery, A. E. (2007), JASA 102(477): 359-378 (CRPS as twice the integral of the pinball loss over the levels; the 19-level grid is ``proper.crps19``).
# Ref: Schmeiser, B. (1982), "Batch size effects in the analysis of simulation output", Operations Research 30(3): 556-568; doi:10.1287/opre.30.3.556 (batch means).
# Ref: Efron, B. (1979), Annals of Statistics 7(1): 1-26 (the bootstrap; percentile intervals).
# Ref: Law, A. M. (2015), "Simulation Modeling and Analysis", 5th ed., McGraw-Hill, section 11.2 (common random numbers for the comparison of two systems).
# Ref: Steyerberg, E. W. et al. (2010), Epidemiology 21(1): 128-138; doi:10.1097/EDE.0b013e3181c30fb2 (calibration-in-the-large).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from c_ept.evaluation import inference
from c_ept.preprocessing import features
from c_ept.valuation import compose

SCHEMA = 'ept-run-vstart-v1'
ROWS_SCHEMA = 'ept-run-vstart-rows-v1'
ROWS_FILE = 'evaluate/VSTART_ROWS.npz'
HORIZONS_S = (60, 120, 180)                                  # the horizons of V_start in seconds, the order of the columns (compose.HORIZON_BINS = 6, 12, 18 bins of 10 s)
LEVELS = compose.Q_LEVELS                                    # tau_k = k / 20
ROW_FIELDS = ('frame_row', 'match', 'primary', 'loss', 'loss_batches', 'loss_observed', 'loss_observed_batches')       # the arrays of one model in the rows file
PAIRED_FIELDS = ('frame_row', 'match', 'primary')            # the arrays that two records must share to be paired (the same rows)


class VStartRefused(ValueError):
    """The composed pass cannot be made as asked (a sample larger than the pool, draws that do not split into batches, records that are not on the same rows). Counts only in the message, never a match number."""


# ---- the sample ---------------------------------------------------------------------------------------------------------------------------
def ids_sha256(ids) -> str:
    """The sha256 of the match numbers as little-endian int64 bytes, in the order given (the sample is sorted): the hash that the record names the sample by.

    해석: 표본의 경기 번호 목록은 기록에 싣지 않고 이 해시만 남긴다."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype='<i8')).tobytes()).hexdigest()


def valued_matches(match, realized) -> np.ndarray:
    """The sorted match numbers whose every row has a finite realised V_start at every horizon (``value.v_start`` gives NaN to the rows of a match that is not valued): the pool of the sample.

    ``match`` (n,) integers, ``realized`` (n, 3). A match with one non-finite entry is out of the pool whole, so a sampled match is scored on all of its rows.

    해석: V_start는 가치가 매겨지지 않은 경기의 행에 NaN이다. 표본은 모든 행이 유한한 경기에서만 뽑아 경기를 통째로 점수 매긴다."""
    m, y = np.asarray(match), np.asarray(realized, dtype=np.float64)
    if m.ndim != 1 or y.shape != (len(m), len(HORIZONS_S)):
        raise ValueError(f'match must be a vector and realized of the shape (n, {len(HORIZONS_S)}), not {y.shape}')
    bad = np.unique(m[~np.isfinite(y).all(axis=1)])
    return np.setdiff1d(np.unique(m), bad).astype(np.int64)


def sample_matches(pool, size, seed) -> np.ndarray:
    """``size`` match numbers drawn without replacement from ``pool`` (a sorted, unique int vector), as a sorted int64 vector: ``sort(pool[default_rng(seed).choice(len(pool), size, replace=False)])``.

    (SA-1)  A = sort( P[ i ] ),  i = default_rng(seed_N).choice(|P|, N, replace=False)

    ``VStartRefused`` (counts only) for a size that is not a whole number of at least 1, a size above the pool, a pool that is not sorted and unique or a negative seed. The rule is the contract: the record
    names the sample by ``ids_sha256``, and the numpy version is recorded next to it.

    해석: 표본은 시드와 크기만으로 정해진다. 풀이 N보다 작으면 거부한다(표본을 줄여 쓰지 않는다)."""
    ids = np.asarray(pool)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (len(ids) > 1 and not (ids[1:] > ids[:-1]).all()):
        raise VStartRefused('the pool of the sample must be a sorted, unique vector of whole match numbers')
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise VStartRefused('the sample size must be a whole number of at least 1')
    if size > len(ids):
        raise VStartRefused(f'the sample asks for {int(size)} matches but the pool holds {len(ids)} valued matches')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise VStartRefused('the sample seed must be a non-negative whole number')
    rng = np.random.default_rng(int(seed))
    return np.sort(ids[rng.choice(len(ids), size=int(size), replace=False)]).astype(np.int64)


def sample_record(pool, sample, size, seed) -> dict:
    """What the record says about the sample: the requested and selected sizes, the pool size, the seed, the hash of the sorted ids and the numpy version (the stream of ``choice`` is a function of both)."""
    return {'matches_requested': int(size), 'matches_selected': int(len(sample)), 'pool_matches': int(len(pool)), 'seed': int(seed), 'ids_sha256': ids_sha256(sample),
            'rule': 'sort(pool[default_rng(seed).choice(len(pool), size, replace=False)])', 'numpy': np.__version__}


# ---- the row losses and their Monte Carlo batches ----------------------------------------------------------------------------------------------
def row_loss(q19, realized) -> np.ndarray:
    """The row loss (VS-1): ``q19`` (..., 19) quantiles, ``realized`` (...) values, the result (...).

    The arithmetic is that of ``proper.crps19``: the error y - q, the pinball with ``compose.Q_LEVELS``, the mean over the levels, times 2; for one row it is bit for bit the value
    that ``crps19`` averages (a test pins this).

    해석: 한 행의 CRPS 대리값(핀볼 손실의 2배 평균)이다. 행의 분위수가 모두 같으면 절대오차이다."""
    q = np.ascontiguousarray(q19, dtype=np.float64)                                          # a C-contiguous array: the mean over the levels sums in the order of ``proper.crps19``, whatever the layout of the argument
    y = np.asarray(realized, dtype=np.float64)
    if q.shape[-1] != len(LEVELS) or q.shape[:-1] != y.shape:
        raise ValueError(f'q19 must be (..., {len(LEVELS)}) and realized the shape of its leading axes, not {q.shape} and {y.shape}')
    error = y[..., None] - q
    pinball = np.where(error >= 0, LEVELS * error, (LEVELS - 1.0) * error)
    return 2.0 * pinball.mean(axis=-1)                                                       # (VS-1)


def horizon_values(draws, n_sims: int) -> np.ndarray:
    """(S, 3) V_start of every simulation from the onsets of one row (``compose.Draws``): the sum of the values of the onsets in the bins before 6, 12 and 18 (60, 120, 180 s); a simulation with no onset
    is a 0. The rule of ``Composer`` (COMP-1), written out from the public ``simulate_onsets`` so that the draws of the row stay available for the batches.

    해석: Composer.simulate가 분위수를 내기 전에 쓰는 모의별 V_start이다. 같은 행의 모의를 배치로 나누려면 분위수 이전의 값이 필요하다."""
    values = np.zeros((int(n_sims), len(compose.HORIZON_BINS)))
    for j, n_bins in enumerate(compose.HORIZON_BINS):
        keep = draws.q < n_bins
        values[:, j] = np.bincount(draws.sim[keep], weights=draws.value[keep], minlength=int(n_sims))
    return values


def simulate_losses(composer, realized, *, batches: int, block: int, progress=None, check=None, label: str = 'nominal') -> tuple:
    """``(loss (n, 3), loss_batches (K, n, 3))``: for every row of the composer's table the row loss (VS-1) of the quantiles of all S draws and of each of the K batches of s = S / K draws.

    ``composer`` is a bound ``compose.Composer`` with ``S`` simulations; ``realized`` (n, 3) the realised V_start of its n rows, in the order of its table. Rows are drawn through the public
    ``simulate_onsets`` (the draws of one row are one stream ``[seed, frame row]``, the same in every composer and every block), in blocks of ``block`` rows: ``check()`` is called before each block
    (the E6 guard) and ``progress('begin', f'{label}/{first row}', {})`` / ``progress('end', ..., {'rows': k, 'draws': S})`` around it; none of them reads a result. The pooled quantiles are
    ``np.quantile(V, Q_LEVELS, axis=0).T`` exactly as ``Composer.simulate`` makes them, so ``loss`` is the loss of that method's ``q19``. ``VStartRefused`` when S is not K x s with s >= 2.

    해석: 모의 S번의 분위수로 낸 손실과, 모의를 K등분한 배치마다의 분위수로 낸 손실을 행마다 함께 구한다. 진행 줄은 블록마다 한 번이며 개수만 담는다."""
    y = np.asarray(realized, dtype=np.float64)
    n, S, K = len(y), int(composer.S), int(batches)
    if y.shape != (n, len(HORIZONS_S)) or n != len(composer.rows):
        raise ValueError(f'realized must be (n, {len(HORIZONS_S)}) for the {len(composer.rows)} rows of the composer table, not {y.shape}')
    if K < 2 or S % K != 0 or S // K < 2:
        raise VStartRefused(f'{S} draws do not split into {K} batches of at least two draws')
    s = S // K
    loss, loss_b = np.empty((n, len(HORIZONS_S))), np.empty((K, n, len(HORIZONS_S)))
    for first in range(0, n, max(1, int(block))):
        last = min(n, first + max(1, int(block)))
        if check is not None:
            check()
        if progress is not None:
            progress('begin', f'{label}/{first}', {})
        for i in range(first, last):
            values = horizon_values(composer.simulate_onsets(i), S)                                            # (S, 3)
            loss[i] = row_loss(np.quantile(values, LEVELS, axis=0).T, y[i])                                    # all S draws
            for b in range(K):
                loss_b[b, i] = row_loss(np.quantile(values[b * s:(b + 1) * s], LEVELS, axis=0).T, y[i])        # the draws of batch b
        if progress is not None:
            progress('end', f'{label}/{first}', {'rows': last - first, 'draws': S})
    return loss, loss_b


# ---- the statistics ---------------------------------------------------------------------------------------------------------------------
def weighted_crps(loss, match, mask) -> float | None:
    """(VS-2): the weighted mean of ``loss[mask]`` with the weights 1 / n_m counted among the masked rows, in the arithmetic of ``proper.crps19`` (``fsum`` of weight x loss over ``fsum`` of the weights);
    None when no row is masked.

    해석: 모집단 안의 행만 세어 경기마다 합이 1인 가중 평균이다. 모집단에 행이 없으면 점수가 없다."""
    keep = np.asarray(mask, dtype=bool)
    if not keep.any():
        return None
    w = features.match_weights(np.asarray(match)[keep])
    return math.fsum(w * np.asarray(loss, dtype=np.float64)[keep]) / math.fsum(w)


def mc_error(batch_values) -> dict:
    """The Monte Carlo error from the K batch values of one statistic (VS-4): ``{'batches': K, 'se': sd_b / sqrt(K) with ddof 1, 'batch_mean', 'batch_values'}``; the ``se`` is None when a batch value is None.

    해석: 같은 통계를 배치마다 구한 K개 값의 표준편차를 √K로 나눈다. 이것이 S회 통계의 몬테카를로 표준오차 추정이다."""
    values = list(batch_values)
    if len(values) < 2 or any(v is None for v in values):                                    # one batch has no spread; a missing batch value (an empty population) has no error
        return {'batches': len(values), 'se': None, 'batch_mean': None, 'batch_values': None}
    a = np.asarray(values, dtype=np.float64)
    return {'batches': int(len(a)), 'se': float(a.std(ddof=1) / math.sqrt(len(a))), 'batch_mean': float(a.mean()), 'batch_values': [float(v) for v in a]}


def bootstrap_summary(per_match_num, per_match_den, matrix) -> dict:
    """The match bootstrap of the ratio sum a / sum d (VS-5) on the rows of ``matrix`` (the prefix of the common resample matrix of the judging split): ``{'B', 'ci95': [2.5 %, 97.5 %], 'se'}``, or
    ``{'unavailable': text}`` when a resample has no present match (the ratio is undefined there; as C4).

    해석: 판정 분할 전체의 공통 재표본 행렬의 앞 B행으로 표본의 경기를 다시 뽑아 같은 비를 구한다. 표본에 없는 경기는 분모 0이어서 빠진다."""
    try:
        draws = inference.bootstrap_stat(per_match_num, per_match_den, matrix)
    except ValueError as error:
        return {'unavailable': str(error)}
    return {'B': int(matrix.shape[0]), 'ci95': [float(v) for v in np.quantile(draws, [0.025, 0.975])], 'se': float(draws.std(ddof=1)) if len(draws) > 1 else 0.0}


def _ratio_parts(match, values, mask, ids) -> tuple:
    """(a, d) per match of ``ids`` for the mean of ``values`` over the masked rows: a_m = the mean over the masked rows of m, d_m = 1 if m has a masked row."""
    keep = np.asarray(mask, dtype=bool)
    return inference.match_aggregate(np.asarray(match)[keep], np.asarray(values)[keep], ids)[1], inference.match_presence(np.asarray(match)[keep], ids)


def population_stat(loss, loss_batches, match, mask, ids, matrix) -> dict | None:
    """``{'crps19', 'n_rows', 'n_matches', 'bootstrap', 'mc'}`` of the masked rows, or None when the mask holds no row: the weighted CRPS (VS-2) of ``loss``, its match bootstrap (VS-5) and its Monte Carlo error
    (VS-4) from the same statistic of every batch of ``loss_batches`` (K, n).

    해석: 한 모집단(주 또는 보조)의 점수와 그 불확실성 두 가지(경기 표본, 몬테카를로)를 함께 낸다."""
    keep = np.asarray(mask, dtype=bool)
    point = weighted_crps(loss, match, keep)
    if point is None:
        return None
    a, d = _ratio_parts(match, loss, keep, ids)
    return {'crps19': point, 'n_rows': int(keep.sum()), 'n_matches': int(len(np.unique(np.asarray(match)[keep]))), 'bootstrap': bootstrap_summary(a, d, matrix),
            'mc': mc_error([weighted_crps(lb, match, keep) for lb in loss_batches])}


def excluded_share(loss, loss_batches, excluded, match, ids, matrix) -> dict:
    """The D14 check (VS-3) for one horizon: the share of the weighted total loss ``L`` that the excluded rows carry, with the weights 1 / n_m over all rows of the sample.

    ``loss`` (n,) is L (the loss of the pass that scores each row: nominal for the primary rows, observed for the excluded ones), ``excluded`` (n,) bool the excluded rows. Returns ``{'crps_share',
    'row_share', 'weight_share', 'bootstrap', 'mc'}``; ``crps_share`` is None when the total loss is 0. The bootstrap treats the share as the ratio sum_m a_m / sum_m d_m with a_m the match mean of the
    loss of its excluded rows (0 for an included row) and d_m the match mean of the loss of all of its rows.

    해석: 제외된 행(그 지평의 창이 경기 끝에 잘리는 행)이 전체 가중 손실에서 차지하는 몫이다. D14는 이 몫이 큰지 보려고 한다. 이 모듈은 수만 내고 기준을 정하지 않는다."""
    ex = np.asarray(excluded, dtype=bool)
    u = features.match_weights(np.asarray(match))

    def share(values):
        total = math.fsum(u * values)
        return math.fsum(u[ex] * values[ex]) / total if total > 0 else None
    point = share(np.asarray(loss, dtype=np.float64))
    num = inference.match_aggregate(match, np.where(ex, loss, 0.0), ids)[1]
    den = inference.match_aggregate(match, loss, ids)[1]
    return {'crps_share': point, 'row_share': float(ex.sum() / len(ex)), 'weight_share': math.fsum(u[ex]) / math.fsum(u),
            'bootstrap': bootstrap_summary(num, den, matrix) if point is not None else {'unavailable': 'the total loss is 0'},
            'mc': mc_error([share(np.asarray(lb, dtype=np.float64)) for lb in loss_batches])}


def summarize(parts: Mapping, ids, matrix) -> dict:
    """The V_start CRPS of one model from the per-row losses of the pass, per horizon, on both populations, with the excluded-row share (D14).

    ``parts`` holds ``match`` (n,), ``primary`` (n, 3) bool (True: the row is in the primary population of the horizon), ``loss`` (n, 3) and ``loss_batches`` (K, n, 3) of the nominal pass for every row, and
    ``loss_observed`` (n, 3), ``loss_observed_batches`` (K, n, 3) of the observed pass, finite at least for every row that some horizon leaves out (NaN elsewhere). ``ids`` is the sorted match list of the
    judging split and ``matrix`` the prefix of the common resample matrix over it. Returns ``{'by_horizon': {'60': {'primary', 'secondary', 'excluded'}, ...}, 'n_rows', 'n_matches'}``: ``primary`` is the
    nominal CRPS of the rows of P_H, ``secondary`` the observed-exposure CRPS of the other rows (None where there are none), ``excluded`` the D14 share.

    해석: 주 모집단은 T를 모르는 명목 예측으로, 보조 모집단은 모든 비교 대상에 같은 T 유래 노출을 준 관측 예측으로 점수를 낸다. 제외된 행의 몫은 두 점수를 같은 가중치로 합한 전체에 대한 비율이다."""
    match = np.asarray(parts['match'])
    primary = np.asarray(parts['primary'], dtype=bool)
    nominal, nominal_b = np.asarray(parts['loss'], dtype=np.float64), np.asarray(parts['loss_batches'], dtype=np.float64)
    observed, observed_b = np.asarray(parts['loss_observed'], dtype=np.float64), np.asarray(parts['loss_observed_batches'], dtype=np.float64)
    n = len(match)
    if primary.shape != (n, len(HORIZONS_S)) or nominal.shape != primary.shape or observed.shape != primary.shape or nominal_b.shape[1:] != primary.shape or observed_b.shape != nominal_b.shape:
        raise ValueError('the parts of the composed V_start do not agree in their number of rows, horizons and batches')
    by_horizon = {}
    for j, horizon in enumerate(HORIZONS_S):
        first, second = primary[:, j], ~primary[:, j]
        if second.any() and not (np.isfinite(observed[second, j]).all() and np.isfinite(observed_b[:, second, j]).all()):
            raise ValueError(f'a row that the horizon {horizon} s leaves out has no loss of the observed pass')
        both = np.where(first, nominal[:, j], np.where(second, observed[:, j], np.nan))
        both_b = np.where(first[None], nominal_b[:, :, j], np.where(second[None], observed_b[:, :, j], np.nan))
        by_horizon[str(horizon)] = {'primary': population_stat(nominal[:, j], nominal_b[:, :, j], match, first, ids, matrix),
                                    'secondary': population_stat(observed[:, j], observed_b[:, :, j], match, second, ids, matrix),
                                    'excluded': excluded_share(both, both_b, second, match, ids, matrix)}
    return {'by_horizon': by_horizon, 'n_rows': int(n), 'n_matches': int(len(np.unique(match)))}


# ---- the occurrence calibration (the bias that the A7 reading must be checked against, H120 diagnosis) ----------------------------------------
def _citl(match, p, y, ids, matrix) -> dict | None:
    m = np.asarray(match)
    if not len(m):
        return None
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    parts = {name: inference.match_aggregate(m, values, ids)[1] for name, values in (('p', p), ('y', y), ('diff', p - y))}
    d = inference.match_presence(m, ids)
    present = float(d.sum())
    return {'n_rows': int(len(m)), 'n_matches': int(present), 'mean_p': float(parts['p'].sum() / present), 'observed_rate': float(parts['y'].sum() / present),
            'citl': float(parts['diff'].sum() / present), 'bootstrap': bootstrap_summary(parts['diff'], d, matrix)}


def occurrence_calibration(endpoints, sample_ids, ids, matrix) -> dict:
    """The calibration-in-the-large of the occurrence probability (VS-7) per horizon, from the endpoints ``M2/t0/H{H}/occurrence`` of the primary pass of the inference: for the arms ``ept`` (``p_ept``) and ``b1``
    (``p_b1``), on the rows of the sampled matches (``sample``) and on every row of the endpoint (``stop``, the whole primary population of the split). Each entry holds the weighted mean probability, the
    observed rate, ``citl`` = mean p - rate (negative: the model under-predicts occurrence) and its match bootstrap. ``None`` for a horizon without an endpoint.

    The probability is the composed hazard 1 - exp(-sum_q lambda_q Delta_q) of the first-kill rate head at the nominal exposure, which is what the Monte Carlo of the pass draws its onsets from: a model whose
    occurrence is biased low draws too few onsets, and the V_start CRPS then inherits the bias. Nothing is changed or recalibrated here.

    해석: 모든 모형이 발생을 25% 가량 낮게 예측한다는 H120 진단 때문에, A7의 V_start 이득을 읽을 때 이 평균 편향을 곁에 둔다. 합성의 발생 위험률은 같은 머리에서 나오므로 닫힌 꼴 확률의 평균 편차가 그 편향의 척도이다."""
    found = {str(ep['name']): ep for ep in endpoints}
    sample = np.asarray(sample_ids)
    out = {}
    for horizon in HORIZONS_S:
        ep = found.get(f'M2/t0/H{horizon}/occurrence')
        if ep is None:
            out[str(horizon)] = None
            continue
        in_sample = np.isin(np.asarray(ep['match']), sample)
        out[str(horizon)] = {scope: {arm: _citl(np.asarray(ep['match'])[sel], np.asarray(ep[key])[sel], np.asarray(ep['y'])[sel], ids, matrix) for arm, key in (('ept', 'p_ept'), ('b1', 'p_b1'))}
                             for scope, sel in (('sample', in_sample), ('stop', np.ones(len(in_sample), dtype=bool)))}
    return out


# ---- the A7 paired contrast ------------------------------------------------------------------------------------------------------------
def paired_contrast(on: Mapping, off: Mapping, ids, matrix) -> dict:
    """The A7 contrast (VS-6) of two records: the V_start CRPS of the models with the zone mark (``on``) minus that of the models without it (``off``), on the same sampled rows, paired by training seed.

    ``on`` and ``off`` map the training seed of a model to its per-row payload (``read_rows``: ``frame_row``, ``match``, ``primary``, ``loss``, ``loss_batches``). Both must hold the same seeds and the same rows
    (``frame_row``, ``match`` and ``primary`` equal, equal batch counts): otherwise ``VStartRefused``. For each horizon the difference of the weighted CRPS of the primary rows is made per seed and averaged over the seeds; the
    match bootstrap (VS-5) takes the same matrix rows for every seed and averages the seed draws, and the Monte Carlo error (VS-4) is the batch-mean error of the seed-averaged difference per batch (the
    streams of the two records are the same, so the Monte Carlo noise of a row is common to both and largely cancels). A negative ``delta`` favours the zone mark. No margin and no verdict: A7's margins are the author's (B4).

    해석: 같은 표본 행에서 구역 켬 − 끔의 V_start CRPS 차이이다. 시드 값끼리 짝지어 평균하고(D15), 같은 재표본 행렬로 경기 표본의 불확실성을, 배치로 몬테카를로 오차를 잰다."""
    if not isinstance(on, Mapping) or not isinstance(off, Mapping) or not on or set(on) != set(off):
        raise VStartRefused('the contrast needs the same, non-empty set of training seeds on both sides')
    seeds = sorted(on)
    reference = on[seeds[0]]
    for side in (on, off):
        for seed in seeds:
            payload = side[seed]
            if any(not np.array_equal(np.asarray(payload[field]), np.asarray(reference[field])) for field in PAIRED_FIELDS):
                raise VStartRefused('the records are not on the same rows (frame row, match and primary population must be equal for every model)')
            if np.shape(payload['loss_batches']) != np.shape(reference['loss_batches']):
                raise VStartRefused('the records have other numbers of Monte Carlo batches or rows')
    match, primary = np.asarray(reference['match']), np.asarray(reference['primary'], dtype=bool)
    by_horizon = {}
    for j, horizon in enumerate(HORIZONS_S):
        mask = primary[:, j]
        if not mask.any():
            by_horizon[str(horizon)] = None
            continue
        per_seed, draws, batch_values = {}, [], []
        for seed in seeds:
            a_on, a_off = on[seed], off[seed]
            diff = np.asarray(a_on['loss'])[:, j] - np.asarray(a_off['loss'])[:, j]
            a, d = _ratio_parts(match, diff, mask, ids)
            per_seed[str(seed)] = float(weighted_crps(a_on['loss'][:, j], match, mask) - weighted_crps(a_off['loss'][:, j], match, mask))
            try:
                draws.append(inference.bootstrap_stat(a, d, matrix))
            except ValueError as error:
                draws.append(str(error))
            batch_values.append([weighted_crps(lb_on[:, j], match, mask) - weighted_crps(lb_off[:, j], match, mask)
                                 for lb_on, lb_off in zip(np.asarray(a_on['loss_batches']), np.asarray(a_off['loss_batches']))])
        delta = float(np.mean(list(per_seed.values())))
        if any(isinstance(d, str) for d in draws):
            boot = {'unavailable': next(d for d in draws if isinstance(d, str))}
        else:
            mean_draws = np.mean(np.stack(draws), axis=0)
            boot = {'B': int(matrix.shape[0]), 'ci95': [float(v) for v in np.quantile(mean_draws, [0.025, 0.975])], 'se': float(mean_draws.std(ddof=1)) if len(mean_draws) > 1 else 0.0,
                    'p': float(inference.two_sided_p(mean_draws, delta))}
        by_horizon[str(horizon)] = {'delta': delta, 'per_seed': per_seed, 'n_rows': int(mask.sum()), 'n_matches': int(len(np.unique(match[mask]))), 'bootstrap': boot,
                                    'mc': mc_error(np.mean(np.asarray(batch_values, dtype=np.float64), axis=0))}
    return {'name': 'A7_vstart_contrast', 'direction': 'zone mark on minus off, CRPS of the primary rows; below 0 favours the zone mark', 'seeds': [int(s) for s in seeds], 'by_horizon': by_horizon,
            'sample_rows': int(len(match)), 'note': 'no margin and no verdict: the A7 margins are the author\'s (B4)'}


# ---- the per-row file of a run ---------------------------------------------------------------------------------------------------------------
def rows_content_sha256(arrays: Mapping) -> str:
    """The sha256 of the arrays of a rows file by content (name, dtype, shape, bytes, in sorted name order): the zip container of an ``.npz`` carries times, so the file bytes are not a stable identity.

    해석: npz 파일은 압축 틀에 시각이 들어가 파일 해시가 매번 다르다. 그래서 배열의 내용으로 해시한다."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode('utf-8'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_rows(path, payloads: Mapping, meta: Mapping) -> str:
    """Write the per-row arrays of the models (``payloads``: model name -> {field of ``ROW_FIELDS``: array}) and ``meta`` (strict JSON; the training seed of every model, the draws, the batches, the seed and
    the sample hash) to the ``.npz`` ``path`` atomically; return ``rows_content_sha256`` of everything that was written.

    해석: A7의 대조는 두 실행의 같은 행을 짝지어야 하므로 행별 손실을 실행 폴더에 남긴다. 경기 번호가 들어 있어 보고서에는 싣지 않고 파일로만 둔다."""
    arrays = {}
    for name, payload in payloads.items():
        if set(payload) != set(ROW_FIELDS):
            raise ValueError(f'the payload of {name!r} must hold exactly the fields {ROW_FIELDS}')
        arrays.update({f'{name}/{field}': np.asarray(payload[field]) for field in ROW_FIELDS})
    arrays['__meta__'] = np.array(json.dumps({'schema': ROWS_SCHEMA, **dict(meta)}, sort_keys=True, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez(temporary, **arrays)
    os.replace(temporary, path)
    return rows_content_sha256(arrays)


def read_rows(path, sha256: str | None = None) -> tuple:
    """``(meta, payloads)`` of a rows file; ``VStartRefused`` when ``sha256`` is given and the content hash differs (a rows file changed after its report named it) or the file is not a rows file.

    ``payloads`` maps the model name to its arrays; ``meta['models'][name]['seed']`` is the training seed (``by_seed`` makes the mapping that ``paired_contrast`` takes).

    해석: 보고서가 적은 내용 해시와 다르면 파일이 바뀐 것이므로 거부한다."""
    with np.load(Path(path), allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name]) for name in stored.files}
    if sha256 is not None and rows_content_sha256(arrays) != sha256:
        raise VStartRefused('the rows file does not have the content hash that the report names')
    if '__meta__' not in arrays:
        raise VStartRefused('this is not a rows file of the composed V_start pass')
    meta = json.loads(str(arrays.pop('__meta__')))
    if meta.get('schema') != ROWS_SCHEMA:
        raise VStartRefused('this is not a rows file of the composed V_start pass')
    payloads = {}
    for key, array in arrays.items():
        name, _, field = key.rpartition('/')
        payloads.setdefault(name, {})[field] = array
    return meta, payloads


def contrast_rows_files(zone_on_path, zone_off_path, ids, matrix, *, zone_on_sha256: str | None = None, zone_off_sha256: str | None = None) -> dict:
    """The A7 contrast (``paired_contrast``) of two rows files, after checking that they are the two sides of the zone-mark comparison and nothing else: the same sample (``ids_sha256``), the same draws,
    batches and draw seed, every model of the first with the zone mark and every model of the second without it, and equal kill law, K condition and frame average on both sides. ``VStartRefused`` otherwise.
    The content hashes of the reports (``zone_*_sha256``) are checked when given. The result also names the sample hash, the draws, the batches and the draw seed.

    해석: 구역 표지를 켠 실행과 끈 실행의 행 파일을 짝지을 때, 표본과 모의 설정이 같고 다른 것이 구역 표지뿐일 때만 대조를 낸다. 공통 난수(같은 draw_seed)가 이 짝짓기의 전제이다."""
    meta_on, rows_on = read_rows(zone_on_path, zone_on_sha256)
    meta_off, rows_off = read_rows(zone_off_path, zone_off_sha256)
    same = {'sample': lambda m: (m.get('sample') or {}).get('ids_sha256'), 'draws': lambda m: m.get('draws'), 'mc_batches': lambda m: m.get('mc_batches'), 'draw_seed': lambda m: m.get('draw_seed')}
    for key, getter in same.items():
        if getter(meta_on) != getter(meta_off) or getter(meta_on) is None:
            raise VStartRefused(f'the two rows files do not share the {key} of the pass: the contrast is on the same sample, the same draws and the same streams')

    def values(meta, key):
        return {item.get(key) for item in meta['models'].values()}
    if values(meta_on, 'zone_mark') != {True} or values(meta_off, 'zone_mark') != {False}:
        raise VStartRefused('the first rows file must hold models with the zone mark and the second models without it')
    for key in ('kill_law', 'k_condition', 'eval_average'):
        if len(values(meta_on, key) | values(meta_off, key)) != 1:
            raise VStartRefused(f'the two sides differ in {key}: the zone mark is the one setting in which they may differ')
    out = paired_contrast(by_seed(meta_on, rows_on), by_seed(meta_off, rows_off), ids, matrix)
    out.update(sample_ids_sha256=meta_on['sample']['ids_sha256'], draws=meta_on['draws'], mc_batches=meta_on['mc_batches'], draw_seed=meta_on['draw_seed'], bootstrap_B=int(matrix.shape[0]))
    return out


def by_seed(meta: Mapping, payloads: Mapping) -> dict:
    """``{training seed: payload}`` of a rows file, for ``paired_contrast``; ``VStartRefused`` for two models of one seed."""
    out = {}
    for name, payload in payloads.items():
        seed = int(meta['models'][name]['seed'])
        if seed in out:
            raise VStartRefused('two models of the rows file have the same training seed')
        out[seed] = payload
    return out

