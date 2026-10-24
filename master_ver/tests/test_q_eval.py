"""q evaluation and calibration (studies/q_production_20261005/qeval.py, qcalibrate.py) on synthetic data only."""
from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from studies.q_production_20261005 import qcalibrate, qeval


def _synthetic(n_matches: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    match_ids = np.repeat(np.arange(n_matches) * 7 + 3, rng.integers(1, 9, size=n_matches))
    rng.shuffle(match_ids)  # rows of a match are not contiguous
    p = rng.uniform(0.01, 0.99, size=match_ids.size)
    y = (rng.random(match_ids.size) < p).astype(np.int8)
    return y, p, match_ids


def _newton(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Independent Newton solution of min mean(w * BCE(sigmoid(a*x + b), y))."""

    theta = np.array([1.0, 0.0])
    design = np.column_stack([x, np.ones_like(x)])
    for _ in range(100):
        q = expit(design @ theta)
        gradient = design.T @ (w * (q - y)) / x.size
        hessian = (design * (w * q * (1.0 - q))[:, None]).T @ design / x.size
        step = np.linalg.solve(hessian, gradient)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return float(theta[0]), float(theta[1])


# ---------------------------------------------------------------- metrics

def test_weighted_metrics_equal_sklearn_with_sample_weight():
    y, p, match_ids = _synthetic()
    random_weights = np.random.default_rng(1).uniform(0.1, 3.0, size=y.size)
    for weights in (random_weights, qeval.match_weights(match_ids), np.ones(y.size)):
        assert qeval.brier(y, p, weights) == pytest.approx(brier_score_loss(y, p, sample_weight=weights), rel=1e-12)
        assert qeval.logloss(y, p, weights) == pytest.approx(log_loss(y, p, sample_weight=weights), rel=1e-12)
        assert qeval.auc(y, p, weights) == pytest.approx(roc_auc_score(y, p, sample_weight=weights), rel=1e-12)
        assert qeval.accuracy(y, p, weights) == pytest.approx(accuracy_score(y, p >= 0.5, sample_weight=weights), rel=1e-12)
    tied = np.round(p, 1)  # many exact ties
    assert qeval.auc(y, tied, random_weights) == pytest.approx(roc_auc_score(y, tied, sample_weight=random_weights), rel=1e-12)
    # weights=None with match_ids is the match weighting; one of the two is required
    assert qeval.brier(y, p, match_ids=match_ids) == qeval.brier(y, p, qeval.match_weights(match_ids))
    with pytest.raises(ValueError):
        qeval.brier(y, p)
    with pytest.raises(ValueError):
        qeval.brier(y, p, np.ones(y.size), match_ids=match_ids)


def test_match_weight_identity_and_normalized_weights():
    y, p, match_ids = _synthetic()
    matches = np.unique(match_ids)
    explicit = np.mean([np.mean((p[match_ids == m] - y[match_ids == m]) ** 2) for m in matches])
    assert qeval.brier(y, p, match_ids=match_ids) == pytest.approx(explicit, rel=1e-12)
    ids, losses = qeval.per_match_loss(y, p, match_ids)
    assert np.array_equal(ids, matches)
    assert losses.mean() == pytest.approx(explicit, rel=1e-12)
    raw = qeval.match_weights(match_ids)
    normalized = qeval.normalized_match_weights(match_ids)
    assert raw.sum() == pytest.approx(matches.size, rel=1e-12)
    assert normalized.mean() == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(normalized, raw * y.size / matches.size, rtol=1e-14, atol=0)


def test_auc_ties_count_half_and_accuracy_threshold_inclusive():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.7, 0.2])
    assert qeval.auc(y, p, np.ones(4)) == pytest.approx(3.5 / 4.0, abs=1e-15)
    assert qeval.auc(y, p, np.array([2.0, 1.0, 1.0, 3.0])) == pytest.approx(11.0 / 12.0, abs=1e-15)
    assert qeval.accuracy(y, p, np.ones(4)) == pytest.approx(0.75)  # p = 0.5 predicts 1
    with pytest.raises(ValueError):
        qeval.auc(np.zeros(4), p, np.ones(4))


def test_logloss_clip_is_documented_value():
    assert qeval.LOGLOSS_EPS == 1e-15
    assert qeval.logloss([1, 0], [0.0, 0.0], np.ones(2)) == pytest.approx(-np.log(1e-15) / 2.0, rel=1e-12)


def test_reliability_bin_edges():
    p = np.array([0.0, 0.0999999, 0.1, 0.3, 0.95, 1.0])
    y = np.array([0, 0, 1, 0, 1, 1])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    table = qeval.reliability_table(y, p, w)
    assert [b["rows"] for b in table] == [2, 1, 0, 1, 0, 0, 0, 0, 0, 2]
    assert table[1]["weight_sum"] == 3.0 and table[1]["mean_y"] == 1.0  # p = 0.1 opens bin [0.1, 0.2)
    assert table[9]["interval"] == "[0.9,1.0]" and table[0]["interval"] == "[0.0,0.1)"
    assert table[9]["mean_p"] == pytest.approx((5 * 0.95 + 6 * 1.0) / 11.0)  # p = 1.0 joins the closed last bin
    assert table[2]["mean_p"] is None and table[2]["mean_y"] is None
    rng = np.random.default_rng(2)
    p_random, w_random = rng.random(5000), rng.uniform(0.1, 2.0, 5000)
    weights = [b["weight_sum"] for b in qeval.reliability_table(rng.integers(0, 2, 5000), p_random, w_random)]
    assert np.allclose(weights, np.histogram(p_random, bins=qeval.RELIABILITY_EDGES, weights=w_random)[0], rtol=1e-12)


# ---------------------------------------------------------------- strata

def test_stratum_labels():
    minutes = qeval.minute_strata([0.0, 1.99, 2.0, 2.5, 39.99, 40.0, 75.0])
    assert minutes.tolist() == ["[0,2)", "[0,2)", "[2,3)", "[2,3)", "[39,40)", "[40,inf)", "[40,inf)"]
    assert len(qeval.MINUTE_ORDER) == 40 and qeval.MINUTE_ORDER[1] == "[2,3)" and qeval.MINUTE_ORDER[-2] == "[39,40)"
    assert set(qeval.minute_strata(np.arange(0, 60, 0.25))) == set(qeval.MINUTE_ORDER)
    assert qeval.game_time_strata([0.0, 14.999, 15.0, 24.999, 25.0]).tolist() == ["[0,15)", "[0,15)", "[15,25)", "[15,25)", "[25,inf)"]
    assert qeval.balanced_p_strata([0.39999, 0.4, 0.5, 0.6, 0.60001]).tolist() == ["outside", "[0.4,0.6]", "[0.4,0.6]", "[0.4,0.6]", "outside"]
    assert qeval.frame_age_strata([0.0, 5.0, 5.001, 10.0]).tolist() == ["[0,5]", "[0,5]", "(5,10]", "(5,10]"]
    assert qeval.endpoint_strata(np.array([False, True])).tolist() == ["full90", "truncated"]
    for bad in ([10.5], [-0.1], [np.nan]):
        with pytest.raises(ValueError):
            qeval.frame_age_strata(bad)
    with pytest.raises(ValueError):
        qeval.minute_strata([-1.0])


def test_metrics_by_stratum_dv1_thresholds_and_counts():
    labels, ys = [], []
    for name, rows, positives in (("A", 99, 30), ("B", 100, 9), ("C", 120, 10)):
        labels += [name] * rows
        ys += [1] * positives + [0] * (rows - positives)
    y = np.array(ys)
    n = y.size
    p = np.random.default_rng(3).uniform(0.05, 0.95, n)
    match_ids = np.arange(n) // 3  # matches straddle strata
    table = qeval.metrics_by_stratum(y, p, None, labels, match_ids=match_ids, order=["A", "B", "C", "D"])   # D-V1 defaults (D-Q14)
    assert list(table) == ["A", "B", "C", "D"]
    assert table["A"]["suppressed"] == ["rows<100"] and table["A"]["metrics"]["brier"] is None
    assert table["B"]["suppressed"] == ["class<10"] and table["B"]["metrics"]["auc"] is None
    assert table["C"]["suppressed"] == [] and table["C"]["metrics"]["auc"] is not None
    assert table["D"]["rows"] == 0 and table["D"]["matches"] == 0 and table["D"]["metrics"]["brier"] is None
    assert (table["A"]["rows"], table["A"]["positives"], table["A"]["negatives"]) == (99, 30, 69)
    rows_a = np.array(labels) == "A"
    assert table["A"]["matches"] == np.unique(match_ids[rows_a]).size
    # match weights are recomputed inside the stratum (D-Q13): each match present weighs equally there
    rows_c = np.array(labels) == "C"
    w = qeval.match_weights(match_ids[rows_c])
    expected = np.sum(w * (p[rows_c] - y[rows_c]) ** 2) / np.sum(w)
    assert table["C"]["metrics"]["brier"] == pytest.approx(expected, rel=1e-12)
    assert table["C"]["weight_sum"] == pytest.approx(np.unique(match_ids[rows_c]).size, rel=1e-12)
    with pytest.raises(ValueError):
        qeval.metrics_by_stratum(y, p, None, labels, match_ids=match_ids, order=["A", "B"])


def test_metrics_by_stratum_relaxed_thresholds_blank_one_class_strata_only():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.3, 0.4, 0.6, 0.3, 0.7, 0.8, 0.5])
    labels = ["neg"] * 3 + ["mix"] * 5
    match_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    assert qeval.metrics_by_stratum(y, p, np.ones(8), labels, match_ids=match_ids)["mix"]["suppressed"] == ["rows<100", "class<10"]
    table = qeval.metrics_by_stratum(y, p, np.ones(8), labels, match_ids=match_ids, min_rows=0, min_class=1)
    assert table["neg"]["metrics"]["brier"] is None and table["neg"]["suppressed"] == ["class<1"]
    assert table["neg"]["rows"] == 3 and table["neg"]["matches"] == 2
    assert table["mix"]["metrics"]["brier"] is not None
    relaxed = qeval.metrics_by_stratum(y, p, np.ones(8), labels, match_ids=match_ids, min_rows=0, min_class=0)
    assert relaxed["neg"]["metrics"]["brier"] == pytest.approx(np.mean(p[:3] ** 2))
    assert relaxed["neg"]["metrics"]["auc"] is None and relaxed["neg"]["metrics"]["calibration_status"] == "one_class"


def test_calibration_slope_intercept_matches_newton_and_flags_degenerate_input():
    y, p, match_ids = _synthetic(n_matches=800, seed=4)
    result = qeval.calibration_slope_intercept(y, p, match_ids=match_ids)
    w = qeval.match_weights(match_ids)
    a, b = _newton(qeval.logit_clipped(p), y.astype(float), w * (w.size / w.sum()))
    assert result["status"] == "fitted"
    assert result["slope"] == pytest.approx(a, abs=1e-7) and result["intercept"] == pytest.approx(b, abs=1e-7)
    assert qeval.calibration_slope_intercept(y, np.full(y.size, 0.3), np.ones(y.size))["slope"] is None
    separated = qeval.calibration_slope_intercept([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], np.ones(4))
    assert separated["status"] == "complete_separation" and separated["slope"] is None


# ---------------------------------------------------------------- bootstrap

def _cells(m: int, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(0.0, 0.4, m) for name in ("a", "b", "c")}


def test_bootstrap_reproducible_and_shared_across_cells():
    m = 400
    ids = np.arange(m) * 3
    cells = _cells(m)
    two_cells = {key: cells[key] for key in ("a", "b")}
    contrast = {"a-b": qeval.difference("a", "b")}
    first = qeval.paired_match_bootstrap(ids, two_cells, contrast, return_draws=True)
    second = qeval.paired_match_bootstrap(ids, two_cells, contrast, return_draws=True)
    assert np.array_equal(first["cell_draws"], second["cell_draws"])
    assert first["contrasts"]["a-b"]["interval"] == second["contrasts"]["a-b"]["interval"]
    wider = qeval.paired_match_bootstrap(ids, cells, {"a-b": qeval.difference("a", "b"), "c-a": qeval.difference("c", "a")},
                                         return_draws=True)
    assert wider["cell_names"] == ["a", "b", "c"]
    assert np.array_equal(wider["cell_draws"][:, :2], first["cell_draws"])  # same draws whatever the cell set
    assert np.array_equal(wider["contrasts"]["a-b"]["draws"], first["contrasts"]["a-b"]["draws"])
    draws = first["cell_draws"]
    assert np.array_equal(first["contrasts"]["a-b"]["draws"], draws[:, 0] - draws[:, 1])  # contrast on identical draws
    assert first["resamples"] == 2000 and first["seed"] == 20261011 and draws.shape == (2000, 2)


def test_bootstrap_interval_quantiles_and_point():
    y, p_sel, match_ids = _synthetic(seed=6)
    p_pt = np.clip(p_sel + np.random.default_rng(7).normal(0.0, 0.1, p_sel.size), 0.0, 1.0)
    ids, sel = qeval.per_match_loss(y, p_sel, match_ids)
    ids_pt, pt = qeval.per_match_loss(y, p_pt, match_ids)
    common, cells = qeval.align_cells({"sel": (ids, sel), "PT_flex": (ids_pt, pt)})
    result = qeval.paired_match_bootstrap(common, cells, {"P1": qeval.p1_contrast("sel"), "rev": qeval.difference("PT_flex", "sel")},
                                          levels={"P1": qeval.PRIMARY_LEVEL, "rev": qeval.EXPLORATORY_LEVEL}, return_draws=True)
    p1, rev = result["contrasts"]["P1"], result["contrasts"]["rev"]
    assert p1["quantiles"] == [0.0125, 0.9875] and rev["quantiles"] == [0.025, 0.975]
    assert p1["interval"] == np.quantile(p1["draws"], [0.0125, 0.9875]).tolist()
    assert rev["interval"] == np.quantile(rev["draws"], [0.025, 0.975]).tolist()
    expected = qeval.brier(y, p_sel, match_ids=match_ids) - qeval.brier(y, p_pt, match_ids=match_ids)
    assert p1["point"] == pytest.approx(expected, abs=1e-15)
    assert qeval.bootstrap_quantiles(0.975) == (0.0125, 0.9875) and qeval.bootstrap_quantiles(0.95) == (0.025, 0.975)


@pytest.mark.parametrize("m", [1, 7, 50])
def test_sequential_draws_equal_small_matrix_reference(m):
    resamples = 300
    cells = _cells(m, seed=m)
    result = qeval.paired_match_bootstrap(np.arange(m), cells, {"a-b": qeval.difference("a", "b")},
                                          resamples=resamples, return_draws=True)
    matrix = np.random.Generator(np.random.PCG64(20261011)).integers(0, m, size=(resamples, m))
    reference = np.stack([cells[name][matrix].mean(axis=1) for name in ("a", "b", "c")], axis=1)
    assert np.array_equal(np.stack(list(qeval.bootstrap_indices(m, resamples))), matrix)
    assert np.array_equal(result["cell_draws"], reference)


def test_bootstrap_never_allocates_resamples_by_matches():
    m, resamples = 60_000, 200
    cells = {key: value for key, value in _cells(m).items() if key != "c"}
    tracemalloc.start()
    try:
        qeval.paired_match_bootstrap(np.arange(m), cells, {"a-b": qeval.difference("a", "b")}, resamples=resamples)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < resamples * m * 8 / 4  # a resamples x M float matrix alone would be 96 MB


def test_bootstrap_rejects_mismatched_or_unsorted_matches():
    with pytest.raises(ValueError):
        qeval.align_cells({"a": (np.arange(4), np.zeros(4)), "b": (np.arange(1, 5), np.zeros(4))})
    with pytest.raises(ValueError):
        qeval.paired_match_bootstrap(np.array([2, 1, 3]), {"a": np.zeros(3), "b": np.zeros(3)}, {"d": qeval.difference("a", "b")})
    with pytest.raises(ValueError):
        qeval.paired_match_bootstrap(np.arange(3), {"a": np.zeros(3)}, {"d": qeval.difference("a", "b")})


# ---------------------------------------------------------------- contrasts

def test_p1_and_p2_arithmetic_on_hand_made_losses():
    sel = np.array([0.1, 0.2, 0.3, 0.2])
    pt = np.array([0.2, 0.2, 0.2, 0.3])
    assert qeval.contrast_point({"sel": sel, "PT_flex": pt}, qeval.p1_contrast("sel")) == pytest.approx(-0.025, abs=1e-15)
    values = {0: {"G1_S1": 0.20, "G0_S1": 0.22, "G1_S0": 0.21, "G0_S0": 0.215},
              1: {"G1_S1": 0.18, "G0_S1": 0.20, "G1_S0": 0.19, "G0_S0": 0.19}}
    m = 5
    cells = {qeval.P2_QUADS[d][key]: np.full(m, value) for d in values for key, value in values[d].items()}
    coefficients = qeval.p2_contrast()
    assert coefficients == {"N_G1_S1_A0_M0_E1_C1_R1_D0": 0.5, "N_G0_S1_A0_M0_E1_C1_R1_D0": -0.5,
                            "N_G1_S0_A0_M0_E1_C1_R1_D0": -0.5, "N_G0_S0_A0_M0_E1_C1_R1_D0": 0.5,
                            "N_G1_S1_A0_M0_E1_C1_R1_D1": 0.5, "N_G0_S1_A0_M0_E1_C1_R1_D1": -0.5,
                            "N_G1_S0_A0_M0_E1_C1_R1_D1": -0.5, "N_G0_S0_A0_M0_E1_C1_R1_D1": 0.5}
    expected = 0.5 * ((0.20 - 0.22) - (0.21 - 0.215)) + 0.5 * ((0.18 - 0.20) - (0.19 - 0.19))  # -0.0175
    assert qeval.contrast_point(cells, coefficients) == pytest.approx(expected, abs=1e-15)
    result = qeval.paired_match_bootstrap(np.arange(m), cells, {"P2": coefficients}, resamples=50)
    assert result["contrasts"]["P2"]["interval"] == pytest.approx([expected, expected], abs=1e-15)
    rng = np.random.default_rng(8)
    a1, b1, a0, b0 = (rng.random(6) for _ in range(4))
    dod = qeval.contrast_per_match({"a1": a1, "b1": b1, "a0": a0, "b0": b0}, qeval.difference_of_differences("a1", "b1", "a0", "b0"))
    assert np.allclose(dod, (a1 - b1) - (a0 - b0), rtol=0, atol=1e-15)


# ---------------------------------------------------------------- calibrator

def _calibration_data(seed: int = 11, n_matches: int = 5000, a_true: float = 0.7, b_true: float = -0.3):
    rng = np.random.default_rng(seed)
    match_ids = np.repeat(np.arange(n_matches), rng.integers(1, 10, size=n_matches))
    raw_logit = rng.normal(0.0, 1.2, size=match_ids.size)
    y = (rng.random(match_ids.size) < expit(a_true * raw_logit + b_true)).astype(np.int8)
    return expit(raw_logit), y, match_ids


def test_calibrator_recovers_known_parameters_and_is_the_weighted_mle():
    p_raw, y, match_ids = _calibration_data()
    calibrator = qcalibrate.fit_q_calibrator(p_raw, y, match_ids)
    assert calibrator.status == "fitted" and calibrator.success and calibrator.reasons == ()
    assert calibrator.a == pytest.approx(0.7, abs=0.06) and calibrator.b == pytest.approx(-0.3, abs=0.06)
    weights = qeval.normalized_match_weights(match_ids)
    a, b = _newton(qeval.logit_clipped(p_raw), y.astype(float), weights)
    assert calibrator.a == pytest.approx(a, abs=1e-7) and calibrator.b == pytest.approx(b, abs=1e-7)
    assert calibrator.class_counts == {"0": int((y == 0).sum()), "1": int((y == 1).sum())}
    assert calibrator.weight_sum == pytest.approx(y.size, rel=1e-12) and calibrator.matches == 5000
    # the objective is the Q_CAL match-weighted log loss of the calibrated predictions
    assert qeval.logloss(y, calibrator.predict(p_raw), match_ids=match_ids) == pytest.approx(calibrator.fun, rel=1e-10)
    # recalibrating the calibrated predictions on the same rows gives slope 1, intercept 0
    again = qeval.calibration_slope_intercept(y, calibrator.predict(p_raw), match_ids=match_ids)
    assert again["slope"] == pytest.approx(1.0, abs=1e-6) and again["intercept"] == pytest.approx(0.0, abs=1e-6)
    # raw weights 1/m_g give the same minimizer; only the objective scale (stopping tests) differs
    raw = qeval.fit_logit_regression(qeval.logit_clipped(p_raw), y, qeval.match_weights(match_ids))
    assert raw["a"] == pytest.approx(calibrator.a, abs=1e-6) and raw["b"] == pytest.approx(calibrator.b, abs=1e-6)
    record = calibrator.to_dict()
    json.dumps(record)
    assert qcalibrate.QCalibrator.from_dict(record) == calibrator


def test_calibrator_accepts_a_stationary_fit_scipy_did_not_flag_successful(monkeypatch):
    y, p, match_ids = _synthetic(n_matches=400, seed=11)
    real = qeval.fit_logit_regression

    def abnormal(x, labels, w):
        fit = real(x, labels, w)
        return {**fit, "success": False, "message": "ABNORMAL_TERMINATION_IN_LNSRCH"}
    monkeypatch.setattr(qeval, "fit_logit_regression", abnormal)
    calibrator = qcalibrate.fit_q_calibrator(p, y, match_ids)
    assert calibrator.status == "fitted" and calibrator.accepted_by_kkt and calibrator.grad_max_abs <= qeval.KKT_TOL

    def not_stationary(x, labels, w):
        return {**real(x, labels, w), "success": False, "grad_max_abs": 1e-3}
    monkeypatch.setattr(qeval, "fit_logit_regression", not_stationary)
    failed = qcalibrate.fit_q_calibrator(p, y, match_ids)
    assert failed.status == "failed" and failed.reasons == ("optimizer_not_successful",) and not failed.accepted_by_kkt


def test_calibrator_failure_records_and_no_replacement():
    p_raw, y, match_ids = _calibration_data(n_matches=200)
    one_class = qcalibrate.fit_q_calibrator(p_raw, np.zeros_like(y), match_ids)
    assert one_class.status == "failed" and one_class.reasons == ("one_class",) and one_class.success is None
    assert one_class.class_counts == {"0": y.size, "1": 0}
    with pytest.raises(RuntimeError, match="no replacement"):
        one_class.predict(p_raw)
    separated = qcalibrate.fit_q_calibrator([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], [1, 1, 2, 2])
    assert separated.status == "failed" and separated.reasons == ("complete_separation",)
    quasi = qcalibrate.fit_q_calibrator([0.1, 0.3, 0.3, 0.4], [0, 0, 1, 1], [1, 1, 2, 2])
    assert quasi.status == "failed" and quasi.reasons == ("quasi_complete_separation",)
    for bad in (dict(p_raw=[0.2, np.nan], y=[0, 1], match_ids=[1, 2]), dict(p_raw=[0.2, 0.3], y=[0, 2], match_ids=[1, 2]),
                dict(p_raw=[0.2, 1.3], y=[0, 1], match_ids=[1, 2]), dict(p_raw=[0.2, 0.3], y=[0, 1], match_ids=[1])):
        with pytest.raises(ValueError):
            qcalibrate.fit_q_calibrator(**bad)


def test_constant_model_and_constant_input():
    p_raw, y, match_ids = _calibration_data(n_matches=300)
    constant = qcalibrate.fit_q_calibrator(np.full(y.size, y.mean()), y, match_ids, constant_model=True)
    assert constant.status == "not_calibrated" and constant.reasons == ("constant_model",) and constant.a is None
    with pytest.raises(RuntimeError):
        constant.predict([0.5])
    flat = qcalibrate.fit_q_calibrator(np.full(y.size, 0.3), y, match_ids)
    assert flat.status == "fitted" and flat.input_constant
    weights = qeval.normalized_match_weights(match_ids)
    prevalence = float(np.sum(weights * y) / np.sum(weights))
    assert flat.predict([0.3])[0] == pytest.approx(prevalence, abs=1e-8)
    # slope not identifiable: L-BFGS-B moves from (1, 0) along (x0, 1) to the optimal line
    x0 = float(qeval.logit_clipped(0.3))
    t = (np.log(prevalence / (1.0 - prevalence)) - x0) / (x0 ** 2 + 1.0)
    assert flat.a == pytest.approx(1.0 + t * x0, abs=1e-6) and flat.b == pytest.approx(t, abs=1e-6)
    assert not qcalibrate.fit_q_calibrator(p_raw, y, match_ids).input_constant


def test_scale_strata_for_d_q17():
    k = np.asarray([1, 2, 3, 4, 9])
    assert list(qeval.kill_count_strata(k)) == ["1", "2-3", "2-3", "4+", "4+"]
    assert list(qeval.credit_strata(np.asarray([1, 3, 1]), np.asarray([1, 1, 5]))) == ["1", "2-3", "4+"]
    with pytest.raises(ValueError):
        qeval.credit_strata(np.asarray([0]), np.asarray([2]))
    with pytest.raises(ValueError):
        qeval.kill_count_strata(np.asarray([0]))
