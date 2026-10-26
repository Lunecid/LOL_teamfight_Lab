"""Tests of the composed V_start pass helpers (``c_ept.run.vstart``; S4b-prep K6e, decisions D9, D14, D30).

The helpers hold the arithmetic of the pass on a sample of STOP matches: the sample, the row losses of the Monte Carlo draws and their batches, the weighted CRPS on both populations, the D14 excluded-row
share, the Monte Carlo and bootstrap errors, the occurrence calibration, the A7 paired contrast and the rows file. They need no model: a fake composer stands in for the draws, and hand-made loss arrays for
the scores, with the numbers worked out by hand in the docstrings. The pass on the toy EPT world, through ``run.evaluate``, is tested in ``test_run_evaluate.py``.

해석: 합성 V_start 표본 계산의 산술(표본 추출, 행 손실과 배치, 모집단별 CRPS, 제외 행의 몫, 몬테카를로·부트스트랩 오차, 발생 보정, A7 짝 대조, 행 파일)을 모형 없이 손으로 푼 숫자로 확인한다.
"""
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from c_ept.evaluation import inference, proper
from c_ept.run import vstart
from c_ept.valuation import compose

POOL = np.arange(100, 160, dtype=np.int64)


# ---- the sample ---------------------------------------------------------------------------------------------------------------------------
def test_the_sample_is_the_sorted_choice_of_the_seed_and_nothing_else():
    """(SA-1): ``sort(pool[default_rng(seed).choice(|pool|, N, replace=False)])`` is the contract (the record names the sample by its hash): same seed and size, same ids; another seed, another ids; the
    whole pool when N = |pool|."""
    first = vstart.sample_matches(POOL, 12, 7)
    assert first.dtype == np.int64 and len(first) == 12 and (first[1:] > first[:-1]).all() and np.isin(first, POOL).all()
    assert np.array_equal(first, np.sort(POOL[np.random.default_rng(7).choice(len(POOL), size=12, replace=False)]))
    assert np.array_equal(first, vstart.sample_matches(POOL, 12, 7)) and not np.array_equal(first, vstart.sample_matches(POOL, 12, 8))
    assert np.array_equal(vstart.sample_matches(POOL, len(POOL), 3), POOL)
    assert np.array_equal(vstart.sample_matches(POOL, 1, 0), np.sort(POOL[np.random.default_rng(0).choice(len(POOL), size=1, replace=False)]))


@pytest.mark.parametrize('size, seed, pool', [(0, 1, POOL), (-1, 1, POOL), (61, 1, POOL), (2.0, 1, POOL), (True, 1, POOL), (3, -1, POOL), (3, 1.5, POOL), (3, True, POOL),
                                              (3, 1, POOL[::-1]), (3, 1, np.array([5, 5, 6])), (3, 1, np.array([0.5, 1.5, 2.5, 3.5])), (3, 1, np.arange(6).reshape(2, 3))])
def test_a_sample_that_cannot_be_drawn_is_refused_with_counts_only(size, seed, pool):
    with pytest.raises(vstart.VStartRefused) as raised:
        vstart.sample_matches(pool, size, seed)
    assert len(str(raised.value)) < 120 and '100' not in str(raised.value) and '159' not in str(raised.value)               # counts only: no match number of the pool in the message


def test_the_hash_of_the_ids_is_the_sha256_of_their_little_endian_int64_bytes_and_the_record_names_the_sample():
    ids = np.array([3, 5, 2 ** 40], dtype=np.int64)
    assert vstart.ids_sha256(ids) == hashlib.sha256(np.array([3, 5, 2 ** 40], dtype='<i8').tobytes()).hexdigest()
    assert vstart.ids_sha256(ids[::-1]) != vstart.ids_sha256(ids)                                      # the order is part of the hash (the sample is sorted)
    sample = vstart.sample_matches(POOL, 5, 9)
    record = vstart.sample_record(POOL, sample, 5, 9)
    assert record['matches_requested'] == 5 and record['matches_selected'] == 5 and record['pool_matches'] == 60 and record['seed'] == 9
    assert record['ids_sha256'] == vstart.ids_sha256(sample) and record['numpy'] == np.__version__ and not any(str(int(m)) in str(record) for m in sample)


def test_the_pool_holds_the_matches_whose_every_row_is_finite_at_every_horizon():
    """``value.v_start`` gives NaN to the rows of a match that is not valued: one non-finite entry takes the whole match out of the pool."""
    match = np.array([1, 1, 2, 2, 3, 3, 4])
    realized = np.ones((7, 3))
    realized[2, 1] = np.nan                                  # match 2
    realized[5, 0] = np.inf                                  # match 3
    assert np.array_equal(vstart.valued_matches(match, realized), [1, 4])
    with pytest.raises(ValueError):
        vstart.valued_matches(match, realized[:, :2])


# ---- the row loss and the draws ---------------------------------------------------------------------------------------------------------
def test_the_row_loss_is_the_row_of_crps19_bit_for_bit_and_the_weighted_crps_is_crps19():
    rng = np.random.default_rng(3)
    q = np.sort(rng.normal(0, 400, (9, 3, 19)), axis=-1)
    y = rng.normal(0, 500, (9, 3))
    loss = vstart.row_loss(q, y)
    assert loss.shape == (9, 3)
    for r in range(9):
        for j in range(3):
            assert loss[r, j] == proper.crps19(q[r, j][None], y[r, j:j + 1], np.ones(1))                 # equal, not close
    match = np.array([5, 5, 5, 6, 6, 7, 7, 7, 7])
    mask = np.array([True, True, False, True, True, False, True, True, True])
    for j in range(3):
        expected = proper.crps19(q[mask, j], y[mask, j], 1.0 / np.array([2, 2, 2, 2, 3, 3, 3]))             # the masked rows: matches 5, 5, 6, 6, 7, 7, 7 count 2, 2, 2, 2, 3, 3, 3 rows
        assert vstart.weighted_crps(loss[:, j], match, mask) == expected
    assert vstart.weighted_crps(loss[:, 0], match, np.zeros(9, dtype=bool)) is None
    assert float(vstart.row_loss(np.full((19,), 2.0), np.float64(-1.0))) == pytest.approx(3.0, rel=1e-15)         # a row of equal quantiles: the absolute error
    with pytest.raises(ValueError):
        vstart.row_loss(q[:, :, :18], y)
    transposed = np.asfortranarray(q)                                                                  # the layout of the argument does not change a bit of the loss (the sums keep the order of crps19)
    assert transposed.flags['C_CONTIGUOUS'] is False and np.array_equal(vstart.row_loss(transposed, y), loss) and np.array_equal(vstart.row_loss(np.moveaxis(np.moveaxis(q, -1, 0), 0, -1), y), loss)


class FakeComposer:
    """The part of a ``Composer`` that ``simulate_losses`` uses: ``S``, ``rows`` and ``simulate_onsets(i)``, with onsets drawn from the stream ``[seed, i]`` (a row has its own stream, as in the real one)."""

    def __init__(self, n_rows, S, seed=5, scale=100.0):
        self.rows, self.S, self.seed, self.scale, self.called = np.arange(n_rows), S, seed, scale, []

    def simulate_onsets(self, i):
        self.called.append(int(i))
        rng = np.random.default_rng([self.seed, int(i)])
        counts = rng.poisson(1.5, self.S)
        sim = np.repeat(np.arange(self.S), counts)
        return SimpleNamespace(sim=sim, q=rng.integers(0, 18, len(sim)), value=rng.normal(0.0, self.scale, len(sim)))


def test_the_draws_of_a_row_are_summed_by_the_bins_of_each_horizon():
    """A hand-made row: S = 4 draws; onsets (sim, bin, value): (0, 0, 10), (0, 7, 5), (1, 11, 2), (1, 12, -4), (3, 17, 8); V for H = 60 (bins < 6), 120 (< 12), 180 (< 18): sim 0 = (10, 15, 15),
    sim 1 = (0, 2, -2), sim 2 = (0, 0, 0) (no onset: a 0), sim 3 = (0, 0, 8)."""
    draws = SimpleNamespace(sim=np.array([0, 0, 1, 1, 3]), q=np.array([0, 7, 11, 12, 17]), value=np.array([10.0, 5.0, 2.0, -4.0, 8.0]))
    expected = np.array([[10.0, 15.0, 15.0], [0.0, 2.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 8.0]])
    assert np.array_equal(vstart.horizon_values(draws, 4), expected)


def test_the_losses_of_the_pool_and_of_every_batch_are_the_row_losses_of_their_own_quantiles():
    """S = 12 draws in K = 3 batches of s = 4: the pooled loss is the loss of the quantiles of all 12 draws (as ``Composer.simulate`` makes them), the batch b the loss of the quantiles of its 4 draws."""
    composer, y = FakeComposer(5, 12), np.random.default_rng(1).normal(0, 100, (5, 3))
    loss, loss_b = vstart.simulate_losses(composer, y, batches=3, block=2)
    assert loss.shape == (5, 3) and loss_b.shape == (3, 5, 3) and composer.called == [0, 1, 2, 3, 4]
    for i in range(5):
        values = vstart.horizon_values(FakeComposer(5, 12).simulate_onsets(i), 12)
        assert np.array_equal(loss[i], vstart.row_loss(np.quantile(values, compose.Q_LEVELS, axis=0).T, y[i]))
        for b in range(3):
            assert np.array_equal(loss_b[b, i], vstart.row_loss(np.quantile(values[4 * b:4 * b + 4], compose.Q_LEVELS, axis=0).T, y[i]))
    assert not np.array_equal(loss_b[0], loss_b[1])


def test_the_progress_and_the_guard_come_once_per_block_with_counts_only():
    """Five rows in blocks of two: three blocks (2, 2, 1 rows), each with a begin and an end call, the guard before each block, nothing but the row and draw counts."""
    events, checks = [], []
    vstart.simulate_losses(FakeComposer(5, 8), np.zeros((5, 3)), batches=2, block=2, progress=lambda event, block, counts: events.append((event, block, dict(counts))), check=lambda: checks.append(1),
                           label='nominal')
    assert events == [('begin', 'nominal/0', {}), ('end', 'nominal/0', {'rows': 2, 'draws': 8}), ('begin', 'nominal/2', {}), ('end', 'nominal/2', {'rows': 2, 'draws': 8}),
                      ('begin', 'nominal/4', {}), ('end', 'nominal/4', {'rows': 1, 'draws': 8})]
    assert len(checks) == 3


@pytest.mark.parametrize('S, K', [(10, 3), (3, 3), (8, 1), (8, 0), (4, 4)])
def test_draws_that_do_not_split_into_batches_of_two_are_refused(S, K):
    with pytest.raises(vstart.VStartRefused):
        vstart.simulate_losses(FakeComposer(2, S), np.zeros((2, 3)), batches=K, block=1)


def test_the_monte_carlo_error_falls_with_the_draws_like_one_over_the_root_of_s():
    """The batch-means error of the sample CRPS: the same rows with S = 80 and S = 720 draws in K = 8 batches. The error of the larger run is about a third of the smaller one's (the ratio of the roots,
    sqrt(80 / 720) = 1/3); the test asks for less than 0.6, with the fixed seed of the fake composer."""
    n, ids = 40, np.arange(10, 20)
    match, y = np.repeat(ids, 4), np.random.default_rng(2).normal(0, 100, (n, 3))
    errors = {}
    for S in (80, 720):
        loss, loss_b = vstart.simulate_losses(FakeComposer(n, S), y, batches=8, block=10)
        stat = vstart.population_stat(loss[:, 0], loss_b[:, :, 0], match, np.ones(n, dtype=bool), ids, np.ones((1, len(ids)), dtype=np.int32))
        errors[S] = stat['mc']['se']
    assert errors[80] > 0 and errors[720] > 0 and errors[720] / errors[80] < 0.6


def test_the_mc_error_is_the_sd_of_the_batch_values_over_root_k():
    values = [1.0, 3.0, 1.0, 3.0]
    error = vstart.mc_error(values)
    assert error['batches'] == 4 and error['batch_mean'] == 2.0 and error['se'] == pytest.approx(np.std(values, ddof=1) / 2) and error['batch_values'] == values
    assert vstart.mc_error([1.0, None]) == {'batches': 2, 'se': None, 'batch_mean': None, 'batch_values': None}


# ---- the CRPS on both populations and the excluded share, by hand ------------------------------------------------------------------------
IDS = np.array([10, 11, 12])
MATCH = np.array([10, 10, 11, 11, 11])
NOMINAL = np.array([4.0, 2.0, 0.0, 3.0, 10.0])                  # |y - c| of the nominal composer (constant quantiles c = 0, 0, 10, 0, 5 against y = 4, -2, 10, 3, -5)
OBSERVED = np.array([np.nan, np.nan, np.nan, 3.0, 10.0])        # the observed composer ran on rows 3 and 4 only: |3 - 0|, |-5 - 5|


def parts(batch_shift=(0.0,)):
    """Three horizons: H = 60 leaves rows 3, 4 out, H = 120 leaves row 4 out, H = 180 leaves none. ``batch_shift`` is added to the loss of the batches (one batch for each entry)."""
    primary = np.zeros((5, 3), dtype=bool)
    primary[:3, 0], primary[:4, 1], primary[:, 2] = True, True, True
    loss, observed = np.repeat(NOMINAL[:, None], 3, axis=1), np.repeat(OBSERVED[:, None], 3, axis=1)
    return {'match': MATCH, 'primary': primary, 'loss': loss, 'loss_batches': np.stack([loss + shift for shift in batch_shift]),
            'loss_observed': observed, 'loss_observed_batches': np.stack([observed + shift for shift in batch_shift])}


def test_the_crps_of_both_populations_and_the_d14_share_are_the_hand_values():
    """By hand (``summarize``): primary H = 60 is rows 0-2, weights 1/n_m inside the population (match 10: 0.5, 0.5; match 11: 1): (0.5 x 4 + 0.5 x 2 + 1 x 0) / 2 = 1.5; secondary rows 3, 4 of match 11 (0.5 each):
    (0.5 x 3 + 0.5 x 10) / 1 = 6.5. The excluded share has the weights 1/n_m over ALL rows (match 10: 1/2 per row, match 11: 1/3 per row): primary sum 0.5 x 4 + 0.5 x 2 + 0 = 3, excluded
    sum (3 + 10) / 3 = 13/3, share (13/3) / (22/3) = 13/22; row share 2/5; weight share (2/3) / 2 = 1/3. H = 120: row 4 is left out: primary rows 0-3, secondary (-5 against 5) 10. H = 180: no row is left out."""
    matrix = np.ones((2, 3), dtype=np.int32)
    out = vstart.summarize(parts((0.0, 0.0)), IDS, matrix)
    h60, h120, h180 = (out['by_horizon'][h] for h in ('60', '120', '180'))
    assert h60['primary']['crps19'] == pytest.approx(1.5) and (h60['primary']['n_rows'], h60['primary']['n_matches']) == (3, 2)
    assert h60['secondary']['crps19'] == pytest.approx(6.5) and (h60['secondary']['n_rows'], h60['secondary']['n_matches']) == (2, 1)
    assert h60['excluded']['crps_share'] == pytest.approx(13 / 22) and h60['excluded']['row_share'] == pytest.approx(0.4) and h60['excluded']['weight_share'] == pytest.approx(1 / 3)
    assert h120['primary']['n_rows'] == 4 and h120['secondary']['n_rows'] == 1 and h120['secondary']['crps19'] == pytest.approx(10.0)
    assert h180['secondary'] is None and h180['primary']['n_rows'] == 5 and h180['excluded']['crps_share'] == 0.0 and h180['excluded']['row_share'] == 0.0
    assert out['n_rows'] == 5 and out['n_matches'] == 2
    # the share is the bootstrap ratio at the point: with the matrix of ones the interval is the point
    assert h60['excluded']['bootstrap']['ci95'] == [pytest.approx(13 / 22)] * 2 and h60['primary']['bootstrap']['ci95'] == [pytest.approx(1.5)] * 2
    assert h60['primary']['crps19'] == pytest.approx(proper.crps19(np.zeros((3, 19)), NOMINAL[:3], np.array([0.5, 0.5, 1.0])), rel=1e-12)       # constant quantiles 0: the loss is the absolute error


def test_the_monte_carlo_error_of_a_statistic_is_the_sd_of_its_batch_values_over_root_k():
    """The loss of the batches is the pooled loss plus a shift of -1, +1, -1, +1: every weighted mean shifts by the same amount, the batch values of the primary CRPS (H = 60) are 1.5 -/+ 1, their sd with
    ddof 1 is 2/sqrt(3) and the error is that over sqrt(4)."""
    out = vstart.summarize(parts((-1.0, 1.0, -1.0, 1.0)), IDS, np.ones((1, 3), dtype=np.int32))
    mc = out['by_horizon']['60']['primary']['mc']
    assert mc['batches'] == 4 and mc['batch_values'] == pytest.approx([0.5, 2.5, 0.5, 2.5]) and mc['se'] == pytest.approx(2 / math.sqrt(3) / 2) and mc['batch_mean'] == pytest.approx(1.5)
    assert out['by_horizon']['60']['secondary']['mc']['se'] == pytest.approx(2 / math.sqrt(3) / 2)


def test_the_match_bootstrap_is_the_ratio_estimator_on_the_prefix_of_the_matrix_over_the_split():
    """The ratio of (VS-5) recomputed by hand on a resample matrix over the split (3 matches, the sample has 2): the interval is the 2.5/97.5 percentiles of the resampled primary CRPS of H = 60 and the
    match that is not in the sample (12) has a zero denominator."""
    matrix = inference.resample_matrix(len(IDS), B=300)
    matrix = matrix[matrix[:, :2].sum(axis=1) > 0]                                     # a resample that draws match 12 only has no present match (the next test)
    out = vstart.summarize(parts(), IDS, matrix)['by_horizon']['60']['primary']['bootstrap']
    a, d = np.array([3.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])                      # match 10: mean (4 + 2)/2 = 3; match 11: its primary row has the loss 0; match 12: absent
    draws = np.array([(row @ a) / (row @ d) for row in matrix.astype(float)])
    assert out['B'] == len(matrix) and out['ci95'] == pytest.approx(list(np.nanquantile(draws, [0.025, 0.975]))) and out['se'] == pytest.approx(np.nanstd(draws, ddof=1))


def test_a_resample_without_a_present_match_says_so_instead_of_a_number():
    matrix = np.array([[0, 0, 5], [1, 1, 1]], dtype=np.int32)                         # the first resample draws only the match that the sample does not hold
    out = vstart.summarize(parts(), IDS, matrix)['by_horizon']['60']
    assert 'unavailable' in out['primary']['bootstrap'] and 'unavailable' in out['excluded']['bootstrap'] and out['primary']['crps19'] == pytest.approx(1.5)


def test_a_row_that_a_horizon_leaves_out_needs_a_loss_of_the_observed_pass():
    bad = parts()
    bad['loss_observed'][3, 0] = np.nan
    with pytest.raises(ValueError, match='observed pass'):
        vstart.summarize(bad, IDS, np.ones((1, 3), dtype=np.int32))
    short = parts()
    short['loss_batches'] = short['loss_batches'][:, :4]
    with pytest.raises(ValueError, match='agree'):
        vstart.summarize(short, IDS, np.ones((1, 3), dtype=np.int32))


# ---- the occurrence calibration ---------------------------------------------------------------------------------------------------------------
def occurrence_endpoint(horizon, match, p, y, b1=None):
    return {'name': f'M2/t0/H{horizon}/occurrence', 'match': np.asarray(match), 'row': np.arange(len(match)), 'y': np.asarray(y), 'p_ept': np.asarray(p, dtype=float),
            'p_b1': np.asarray(p if b1 is None else b1, dtype=float)}


def test_the_calibration_in_the_large_is_the_match_balanced_mean_of_p_minus_the_observed_rate():
    """By hand: matches 10 (two rows) and 11 (one row) are the sample, match 12 is not. p = (0.2, 0.4 | 0.6 | 0.9), o = (0, 1 | 0 | 1). Match means: p 0.3, 0.6; o 0.5, 0; so mean p = 0.45, rate = 0.25 and
    CITL = 0.20 on the sample; over every row of the endpoint (three matches) mean p = (0.3 + 0.6 + 0.9) / 3 = 0.6, rate = (0.5 + 0 + 1) / 3 = 0.5, CITL 0.1."""
    ep = occurrence_endpoint(60, [10, 10, 11, 12], [0.2, 0.4, 0.6, 0.9], [0, 1, 0, 1], b1=[0.1, 0.1, 0.1, 0.1])
    out = vstart.occurrence_calibration([ep], np.array([10, 11]), IDS, np.ones((1, 3), dtype=np.int32))
    sample, stop = out['60']['sample']['ept'], out['60']['stop']['ept']
    assert (sample['n_rows'], sample['n_matches']) == (3, 2) and sample['mean_p'] == pytest.approx(0.45) and sample['observed_rate'] == pytest.approx(0.25) and sample['citl'] == pytest.approx(0.2)
    assert (stop['n_rows'], stop['n_matches']) == (4, 3) and stop['mean_p'] == pytest.approx(0.6) and stop['observed_rate'] == pytest.approx(0.5) and stop['citl'] == pytest.approx(0.1)
    assert out['60']['sample']['b1']['mean_p'] == pytest.approx(0.1) and out['120'] is None and out['180'] is None
    assert sample['bootstrap']['ci95'] == [pytest.approx(0.2)] * 2


def test_a_sample_without_a_row_of_the_endpoint_has_no_calibration_entry():
    ep = occurrence_endpoint(60, [12], [0.5], [1])
    out = vstart.occurrence_calibration([ep], np.array([10]), IDS, np.ones((1, 3), dtype=np.int32))
    assert out['60']['sample'] == {'ept': None, 'b1': None} and out['60']['stop']['ept']['n_matches'] == 1


# ---- the A7 paired contrast ------------------------------------------------------------------------------------------------------------
def payload(offset, seeds_shift=0.0, batches=4):
    """The arrays of one model on the 5 rows of ``parts``: the nominal loss plus ``offset``, batches around it by +/- 0.5."""
    base = parts()
    loss = base['loss'] + offset + seeds_shift
    batch = np.stack([loss + (0.5 if b % 2 else -0.5) * (b + 1) / 2 for b in range(batches)])
    return {'frame_row': np.arange(5, dtype=np.int64), 'match': MATCH, 'primary': base['primary'], 'loss': loss, 'loss_batches': batch,
            'loss_observed': base['loss_observed'], 'loss_observed_batches': np.stack([base['loss_observed']] * batches)}


def test_the_contrast_is_on_minus_off_on_the_same_rows_averaged_over_the_paired_seeds():
    """Zone-on losses are the zone-off losses minus 0.5 (seed 1) and minus 1.5 (seed 2) on every row: the CRPS difference of any population is exactly -0.5 and -1.5, so the seed mean is -1 at every
    horizon, in every batch (the batch shifts are common to both sides and cancel) and in every resample."""
    off = {1: payload(0.0), 2: payload(0.0, 0.25)}
    on = {1: payload(-0.5), 2: payload(-1.5, 0.25)}
    out = vstart.paired_contrast(on, off, IDS, np.ones((5, 3), dtype=np.int32))
    assert out['seeds'] == [1, 2] and out['sample_rows'] == 5 and 'verdict' not in out
    for horizon in ('60', '120', '180'):
        entry = out['by_horizon'][horizon]
        assert entry['delta'] == pytest.approx(-1.0) and entry['per_seed'] == {'1': pytest.approx(-0.5), '2': pytest.approx(-1.5)}
        assert entry['bootstrap']['ci95'] == [pytest.approx(-1.0)] * 2 and entry['bootstrap']['se'] == pytest.approx(0.0, abs=1e-12) and entry['mc']['se'] == pytest.approx(0.0, abs=1e-12)
    assert out['by_horizon']['60']['n_rows'] == 3 and out['by_horizon']['120']['n_rows'] == 4 and out['by_horizon']['180']['n_rows'] == 5


def test_the_contrast_reports_the_monte_carlo_error_of_the_difference_and_the_bootstrap_of_the_sample():
    """The on side is the off side minus 0.5 on the rows of match 10 and minus 1.5 on those of match 11, and its batches are shifted by +1, -1, +1, -1 against the off side's. The pooled primary difference of
    H = 60 (rows 0, 1 of match 10 and row 2 of match 11, a weight of 1 for each match) is (-0.5 - 1.5) / 2 = -1; the batch values are -1 + shift = 0, -2, 0, -2, so the error is
    sd(0, -2, 0, -2; ddof 1) / sqrt(4) = sqrt(4/3) / 2. The match bootstrap is the ratio (-0.5 c10 - 1.5 c11) / (c10 + c11) on each row of the matrix, recomputed here by a loop."""
    drop = np.array([0.5, 0.5, 1.5, 1.5, 1.5])[:, None]
    off = {7: payload(0.0)}
    on = {7: {**off[7], 'loss': off[7]['loss'] - drop, 'loss_batches': off[7]['loss_batches'] - drop[None] + np.array([1.0, -1.0, 1.0, -1.0])[:, None, None]}}
    matrix = inference.resample_matrix(len(IDS), B=300)
    matrix = matrix[matrix[:, :2].sum(axis=1) > 0]
    out = vstart.paired_contrast(on, off, IDS, matrix)['by_horizon']['60']
    assert out['delta'] == pytest.approx(-1.0) and out['n_rows'] == 3 and out['n_matches'] == 2
    assert out['mc']['batch_values'] == pytest.approx([0.0, -2.0, 0.0, -2.0]) and out['mc']['se'] == pytest.approx(math.sqrt(4 / 3) / 2) and out['mc']['batches'] == 4
    draws = np.array([(-0.5 * c[0] - 1.5 * c[1]) / (c[0] + c[1]) if c[0] + c[1] > 0 else np.nan for c in matrix[:, :2].astype(float)])        # match 12 (column 2) holds no sampled row
    assert out['bootstrap']['B'] == len(matrix) and out['bootstrap']['ci95'] == pytest.approx(list(np.nanquantile(draws, [0.025, 0.975])))
    assert out['bootstrap']['se'] == pytest.approx(np.nanstd(draws, ddof=1)) and 0.0 < out['bootstrap']['p'] <= 1.0


def test_the_contrast_is_refused_for_records_that_are_not_on_the_same_rows_or_seeds():
    off, on = {1: payload(0.0)}, {1: payload(-1.0)}
    matrix = np.ones((1, 3), dtype=np.int32)
    other_rows = {1: {**payload(-1.0), 'frame_row': np.arange(5, dtype=np.int64) + 1}}
    other_primary = {1: {**payload(-1.0), 'primary': ~payload(-1.0)['primary']}}
    other_match = {1: {**payload(-1.0), 'match': np.array([10, 10, 11, 11, 12])}}
    other_batches = {1: payload(-1.0, batches=6)}
    for bad in (other_rows, other_primary, other_match, other_batches, {2: payload(-1.0)}, {}, {1: payload(-1.0), 2: payload(-1.0)}):
        with pytest.raises(vstart.VStartRefused):
            vstart.paired_contrast(bad, off, matrix=matrix, ids=IDS)
    with pytest.raises(vstart.VStartRefused):
        vstart.paired_contrast(on, {1: other_rows[1]}, IDS, matrix)
    assert vstart.paired_contrast(on, off, IDS, matrix)['seeds'] == [1]


def test_a_horizon_without_a_primary_row_has_no_contrast():
    base = payload(0.0)
    none60 = {1: {**base, 'primary': np.zeros_like(base['primary'])}}
    assert vstart.paired_contrast(none60, none60, IDS, np.ones((1, 3), dtype=np.int32))['by_horizon']['60'] is None


# ---- the rows file --------------------------------------------------------------------------------------------------------------------
def test_the_rows_file_round_trips_by_content_hash_and_refuses_a_changed_file(tmp_path):
    payloads = {'seed1': payload(0.0), 'seed2': payload(-0.25)}
    meta = {'models': {'seed1': {'seed': 20261022}, 'seed2': {'seed': 20261023}}, 'draws': 8}
    path = tmp_path / 'evaluate' / 'VSTART_ROWS.npz'
    digest = vstart.write_rows(path, payloads, meta)
    got_meta, got = vstart.read_rows(path, digest)
    assert got_meta['draws'] == 8 and got_meta['schema'] == vstart.ROWS_SCHEMA and set(got) == {'seed1', 'seed2'}
    for name in payloads:
        assert set(got[name]) == set(vstart.ROW_FIELDS) and all(np.array_equal(got[name][f], payloads[name][f], equal_nan=got[name][f].dtype.kind == 'f') and got[name][f].dtype == payloads[name][f].dtype for f in vstart.ROW_FIELDS)
    assert set(vstart.by_seed(got_meta, got)) == {20261022, 20261023} and not list(tmp_path.rglob('*.tmp.npz'))
    again = vstart.write_rows(tmp_path / 'again.npz', payloads, meta)
    assert again == digest                                    # the content hash does not depend on the container (an npz carries times)
    with pytest.raises(vstart.VStartRefused, match='content hash'):
        vstart.read_rows(path, '0' * 64)
    changed = {**payloads, 'seed2': {**payloads['seed2'], 'loss': payloads['seed2']['loss'] + 1e-9}}
    assert vstart.write_rows(tmp_path / 'changed.npz', changed, meta) != digest
    with pytest.raises(ValueError, match='exactly the fields'):
        vstart.write_rows(tmp_path / 'bad.npz', {'seed1': {'loss': np.zeros(1)}}, meta)
    np.savez(tmp_path / 'foreign.npz', x=np.zeros(2))
    with pytest.raises(vstart.VStartRefused, match='not a rows file'):
        vstart.read_rows(tmp_path / 'foreign.npz')
    twin = {'seed1': payloads['seed1'], 'seed2': payloads['seed2']}
    with pytest.raises(vstart.VStartRefused, match='same training seed'):
        vstart.by_seed({'models': {'seed1': {'seed': 5}, 'seed2': {'seed': 5}}}, twin)


def zone_meta(zone, **over):
    model = {'seed': 1, 'zone_mark': zone, 'kill_law': 'poisson', 'k_condition': 'raw', 'eval_average': True}
    return {'sample': {'ids_sha256': 'a' * 64}, 'draws': 8, 'mc_batches': 4, 'draw_seed': 1, 'models': {'seed1': {**model, **over.pop('model', {})}}, **over}


def test_the_contrast_of_two_rows_files_needs_the_two_sides_of_the_zone_comparison_and_nothing_else(tmp_path):
    """The files are paired only when they hold the same sample, draws, batches and draw seed, the first the models with the zone mark and the second those without, and the zone mark is the only setting in which
    they differ; the report hashes are checked when given."""
    matrix = np.ones((2, 3), dtype=np.int32)
    on = tmp_path / 'on.npz'
    off = tmp_path / 'off.npz'
    on_sha = vstart.write_rows(on, {'seed1': payload(-1.0)}, zone_meta(True))
    off_sha = vstart.write_rows(off, {'seed1': payload(0.0)}, zone_meta(False))
    out = vstart.contrast_rows_files(on, off, IDS, matrix, zone_on_sha256=on_sha, zone_off_sha256=off_sha)
    assert out['by_horizon']['60']['delta'] == pytest.approx(-1.0) and out['sample_ids_sha256'] == 'a' * 64 and (out['draws'], out['mc_batches'], out['draw_seed'], out['bootstrap_B']) == (8, 4, 1, 2)
    cases = {'draws': zone_meta(False, draws=16), 'mc_batches': zone_meta(False, mc_batches=2), 'draw_seed': zone_meta(False, draw_seed=2), 'sample': zone_meta(False, sample={'ids_sha256': 'b' * 64}),
             'zone_mark': zone_meta(True), 'kill_law': zone_meta(False, model={'kill_law': 'shifted'}), 'k_condition': zone_meta(False, model={'k_condition': 'binned'}),
             'eval_average': zone_meta(False, model={'eval_average': False})}
    for word, meta in cases.items():
        other = tmp_path / f'{word}.npz'
        vstart.write_rows(other, {'seed1': payload(0.0)}, meta)
        with pytest.raises(vstart.VStartRefused, match=word if word != 'zone_mark' else 'zone mark'):
            vstart.contrast_rows_files(on, other, IDS, matrix)
    with pytest.raises(vstart.VStartRefused, match='zone mark'):
        vstart.contrast_rows_files(off, on, IDS, matrix)                                              # the sides are not interchangeable: the first is the zone mark on
    with pytest.raises(vstart.VStartRefused, match='content hash'):
        vstart.contrast_rows_files(on, off, IDS, matrix, zone_on_sha256='0' * 64)
    gone = tmp_path / 'nosample.npz'
    vstart.write_rows(gone, {'seed1': payload(0.0)}, {**zone_meta(False), 'sample': None})
    with pytest.raises(vstart.VStartRefused, match='sample'):
        vstart.contrast_rows_files(on, gone, IDS, matrix)
