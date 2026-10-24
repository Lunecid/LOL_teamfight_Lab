"""Task 11 (T4.2): E4 direction labels and their agreement with the main labels
(studies/tog_experiments_20261012/e4_run.py), synthetic data only.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (sections 3 and 6); SDD Ruling 44 (TRAIN,
Q_DEV and Q_CAL rows only; TEST labels are made in Task 12, straight into the sealed file).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from studies.label_production_20261005 import assemble
from studies.tog_experiments_20261012 import e4_run as er
from studies.tog_experiments_20261012.e4_run import direction_labels


# ----------------------------------------------------------------------------- the plan's two tests (verbatim)
def test_positive_slope_calibration_keeps_direction():
    rng = np.random.default_rng(0)
    pre, post = rng.uniform(0.01, 0.99, 1000), rng.uniform(0.01, 0.99, 1000)
    cal = lambda p: 1 / (1 + np.exp(-(1.3 * np.log(p / (1 - p)) - 0.2)))
    assert np.array_equal(direction_labels(pre, post), direction_labels(cal(pre), cal(post)))

def test_ties_are_negative():
    assert direction_labels(np.array([0.4, 0.4]), np.array([0.4, 0.41])).tolist() == [0, 1]


# ----------------------------------------------------------------------------- labels and zero flags
# p_pre = 0 makes the difference exactly p_post, so the threshold cases below are exact (no rounding in p_post - p_pre).
HAND_PRE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.012, 0.02, 0.3])
HAND_POST = np.array([0.0, 0.004, 0.005, 0.0099, 0.01, 0.015, 0.0199, 0.02, 0.5, 0.0, 0.0, 0.0, 0.3])
# bit 0: |d| < 0.005, bit 1: |d| < 0.01, bit 2: |d| < 0.02 (assemble.py:40 and :122-124; strict "<")
HAND_NEAR = [7, 7, 6, 6, 4, 4, 4, 0, 0, 7, 4, 0, 7]
HAND_EXACT = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
HAND_Y = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_direction_labels_hand_values_and_dtype():
    y = direction_labels(HAND_PRE, HAND_POST)
    assert y.dtype == np.int8 and y.tolist() == HAND_Y


def test_zero_flags_near_zero_bits_hand_values():
    flags = er.zero_flags(HAND_PRE, HAND_POST)
    assert set(flags) == {"exact_zero", "near_zero_bits"}
    assert flags["near_zero_bits"].dtype == np.uint8 and flags["near_zero_bits"].tolist() == HAND_NEAR
    assert flags["exact_zero"].dtype == np.int8 and flags["exact_zero"].tolist() == HAND_EXACT
    assert er.NEAR_ZERO == (0.005, 0.01, 0.02)


def _frozen_label_table(p_pre, p_post):
    """The frozen main-label code (assemble.label_table) on Q_DEV rows whose final-context probabilities are given."""

    n = len(p_pre)
    zeros = np.zeros(n, dtype=np.int64)
    eng = {"role": np.full(n, 3, dtype=np.int8), "fold": np.full(n, -1, dtype=np.int8), "valid_h90": np.ones(n, dtype=bool),
           "query_pre": 2 * np.arange(n), "query_h90": 2 * np.arange(n) + 1, "match_index": np.arange(n),
           "canonical_match_key": np.array([b"m%d" % i for i in range(n)]), "component_key": np.full(n, b"c0"),
           "global_isolation_verified": np.ones(n, dtype=bool)}
    for name in ("first_kill_ms", "last_kill_ms", "pre_ms", "post_ms_h90", "F_pre_ms", "F_post_ms_h90", "pre_frame_age_ms",
                 "post_frame_age_ms_h90", "first_kill_frame_lag_ms", "actual_followup_ms_h90", "comparison_elapsed_ms_h90",
                 "truncation_bits_h90"):
        eng[name] = zeros
    v = np.empty(2 * n)
    v[0::2], v[1::2] = p_pre, p_post
    preds = {"final": {"query_index": np.arange(2 * n), "logit": np.zeros(2 * n), "raw_p": v, "v": v}}
    return assemble.label_table(eng, preds, np.zeros((6, n), dtype=np.uint8), 90)


def test_labels_and_flags_mirror_the_frozen_main_label_code():
    rng = np.random.default_rng(11)
    pre = np.concatenate([HAND_PRE, rng.uniform(0, 1, 400), np.full(6, 0.5)])
    post = np.concatenate([HAND_POST, pre[len(HAND_PRE):len(HAND_PRE) + 400] + rng.normal(0, 0.012, 400),
                           0.5 + np.array([0.005, -0.005, 0.01, -0.01, 0.02, -0.02])])
    post = np.clip(post, 0, 1)
    table = _frozen_label_table(pre, post)
    assert table["valid"].all()
    flags = er.zero_flags(pre, post)
    assert np.array_equal(direction_labels(pre, post), table["direction_positive"])
    assert np.array_equal(flags["exact_zero"], table["exact_zero"])
    assert np.array_equal(flags["near_zero_bits"], table["near_zero_bits"])
    assert np.array_equal(post - pre, table["svi_delta"])


@pytest.mark.parametrize("pre, post", [
    (np.array([0.5, np.nan]), np.array([0.5, 0.5])),
    (np.array([0.5, 0.5]), np.array([0.5, np.inf])),
    (np.array([0.5, 1.2]), np.array([0.5, 0.5])),
    (np.array([0.5, 0.5]), np.array([-0.1, 0.5])),
    (np.array([0.5]), np.array([0.5, 0.5])),
    (np.array([[0.5]]), np.array([[0.5]])),
])
def test_label_functions_refuse_invalid_probabilities(pre, post):
    with pytest.raises(ValueError):
        direction_labels(pre, post)
    with pytest.raises(ValueError):
        er.zero_flags(pre, post)
    with pytest.raises(ValueError):
        er.tiny_deltas(pre, post)


def test_tiny_deltas_count_nonzero_differences_below_1e_12():
    pre = np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.3])
    post = np.array([0.5, 0.5 + 2.0 ** -53, 0.5 + 1e-11, 0.5 - 2.0 ** -50, 0.25, 0.3 + 2.0 ** -54])
    out = er.tiny_deltas(pre, post)
    assert out["count"] == 3                       # rows 1, 3, 5: 0 < |d| < 1e-12; ties (rows 0, 4) do not count
    assert out["bound"] == 1e-12
    assert out["min_nonzero_abs_delta"] == 2.0 ** -54
    assert er.tiny_deltas(np.array([0.5]), np.array([0.5]))["min_nonzero_abs_delta"] is None


# ----------------------------------------------------------------------------- agreement
def test_svi_bins_edges():
    svi = np.array([0.0, -0.0099, 0.0099, 0.01, -0.01, 0.0499, -0.05, 0.05, 0.3])
    assert er.svi_bins(svi).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert er.BINS == ("abs_svi_lt_0.01", "abs_svi_0.01_to_0.05", "abs_svi_ge_0.05")


def test_agreement_bins_follow_abs_main_svi():
    svi = np.array([0.0, -0.0099, 0.0099, 0.01, -0.01, 0.0499, -0.05, 0.05, 0.3])
    y_main = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1])
    y_new = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0])        # bin 0: 3/3 equal, bin 1: 2/3, bin 2: 0/3
    out = er.agreement(y_main, y_new, svi, np.arange(9))
    bins = out["by_abs_main_svi"]
    assert list(bins) == list(er.BINS)
    assert [bins[name]["rows"] for name in er.BINS] == [3, 3, 3]
    for name, share in zip(er.BINS, (1.0, 2 / 3, 0.0)):
        assert bins[name]["share_equal"]["row_weighted"] == pytest.approx(share)
        assert bins[name]["share_equal"]["match_weighted"] == pytest.approx(share)   # one row per match
    assert out["overall"]["rows"] == 9 and out["overall"]["share_equal"]["row_weighted"] == pytest.approx(5 / 9)


def test_agreement_match_weighting_hand_values():
    # match 7: three rows, two equal; match 9: one row, not equal
    y_main, y_new = np.array([1, 0, 1, 1]), np.array([1, 0, 0, 0])
    match = np.array([7, 7, 7, 9])
    out = er.agreement(y_main, y_new, np.array([0.2, -0.2, 0.2, 0.2]), match)
    overall = out["overall"]
    assert overall["rows"] == 4 and overall["matches"] == 2
    assert overall["share_equal"]["row_weighted"] == pytest.approx(0.5)
    assert overall["share_equal"]["match_weighted"] == pytest.approx((2 / 3 + 0) / 2)
    assert list(overall["share_equal"]) == ["match_weighted", "row_weighted"]       # match weighting is primary
    # within a bin each match present weighs 1 (weights recomputed inside the bin, qeval.metrics_by_stratum, D-Q13):
    # bin >= 0.05 holds rows 0, 1 (match 7, both equal) and row 3 (match 9, not equal) -> (1 + 0) / 2, not 0.4
    out = er.agreement(y_main, y_new, np.array([0.2, -0.2, 0.001, 0.2]), match)
    top, low = out["by_abs_main_svi"]["abs_svi_ge_0.05"], out["by_abs_main_svi"]["abs_svi_lt_0.01"]
    assert top["matches"] == 2 and top["share_equal"]["match_weighted"] == pytest.approx(0.5)
    assert top["share_equal"]["row_weighted"] == pytest.approx(2 / 3)
    assert low["rows"] == 1 and low["share_equal"]["match_weighted"] == 0.0


def test_agreement_kappa_hand_values_and_sklearn():
    y_main = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_new = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    out = er.agreement(y_main, y_new, np.full(8, 0.2), np.arange(8))
    assert out["overall"]["kappa"]["row_weighted"] == pytest.approx(0.5)         # p_o 6/8, p_e 1/2
    assert out["overall"]["kappa"]["row_weighted"] == pytest.approx(cohen_kappa_score(y_main, y_new))
    match = np.array([0, 0, 0, 1, 2, 2, 3, 3])
    out = er.agreement(y_main, y_new, np.full(8, 0.2), match)
    # weights 1/3,1/3,1/3,1,1/2,1/2,1/2,1/2: p_o 19/24, p_main 5/12, p_new 11/24, p_e 73/144 -> kappa 41/71
    assert out["overall"]["kappa"]["match_weighted"] == pytest.approx(41 / 71)
    weights = 1.0 / np.bincount(match)[match]
    assert out["overall"]["kappa"]["match_weighted"] == pytest.approx(cohen_kappa_score(y_main, y_new, sample_weight=weights))
    assert out["overall"]["kappa"]["row_weighted"] == pytest.approx(0.5)
    assert out["overall"]["confusion_rows"] == {"main1_new1": 3, "main1_new0": 1, "main0_new1": 1, "main0_new0": 3}


def test_agreement_kappa_undefined_when_both_labels_constant():
    out = er.agreement(np.ones(3, dtype=np.int8), np.ones(3, dtype=np.int8), np.full(3, 0.2), np.arange(3))
    assert out["overall"]["share_equal"]["row_weighted"] == 1.0
    assert out["overall"]["kappa"] == {"match_weighted": None, "row_weighted": None}


def test_agreement_keeps_exact_zero_rows_and_reports_their_share():
    # rows 1 and 2 are E4 ties (negative, exact zero) and stay in every denominator; row 4 is a main exact zero
    y_main, y_new = np.array([1, 1, 0, 1, 0]), np.array([1, 0, 0, 1, 1])
    exact = np.array([0, 1, 1, 0, 0])
    match = np.array([1, 1, 2, 3, 3])
    svi = np.array([0.2, 0.001, -0.001, 0.03, 0.0])
    out = er.agreement(y_main, y_new, svi, match, exact_zero_new=exact)
    overall = out["overall"]
    assert overall["rows"] == 5
    assert overall["share_equal"]["row_weighted"] == pytest.approx(3 / 5)
    assert overall["exact_zero_share_new"]["row_weighted"] == pytest.approx(2 / 5)
    assert overall["exact_zero_share_new"]["match_weighted"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert overall["exact_zero_share_main"]["row_weighted"] == pytest.approx(1 / 5)
    assert overall["exact_zero_share_main"]["match_weighted"] == pytest.approx((0 + 0 + 1 / 2) / 3)
    low = out["by_abs_main_svi"]["abs_svi_lt_0.01"]                                 # rows 1, 2, 4
    assert low["rows"] == 3 and low["exact_zero_share_new"]["row_weighted"] == pytest.approx(2 / 3)
    assert er.agreement(y_main, y_new, svi, match)["overall"]["exact_zero_share_new"] is None
    no_match = er.agreement(y_main, y_new, svi)["overall"]
    assert no_match["matches"] is None and no_match["share_equal"]["match_weighted"] is None
    assert no_match["share_equal"]["row_weighted"] == pytest.approx(3 / 5)


def test_agreement_empty_bin_reports_counts_only():
    out = er.agreement(np.array([1, 0]), np.array([1, 1]), np.array([0.2, 0.3]), np.array([0, 1]))
    empty = out["by_abs_main_svi"]["abs_svi_lt_0.01"]
    assert empty["rows"] == 0 and empty["matches"] == 0
    assert empty["share_equal"] == {"match_weighted": None, "row_weighted": None}


@pytest.mark.parametrize("y_main, y_new, svi, match, exact", [
    (np.array([1, -1]), np.array([1, 0]), np.array([0.1, 0.1]), np.array([0, 1]), None),         # sealed TEST y = -1
    (np.array([1, 0]), np.array([1, 0]), np.array([0.1, np.nan]), np.array([0, 1]), None),       # sealed TEST SVI NaN
    (np.array([1, 0]), np.array([1, 0, 1]), np.array([0.1, 0.1]), np.array([0, 1]), None),
    (np.array([1, 0]), np.array([1, 0]), np.array([0.1, 0.1]), np.array([0, 1, 2]), None),
    (np.array([1, 0]), np.array([1, 0]), np.array([0.1, 0.1]), np.array([0, 1]), np.array([0, 2])),
    (np.array([1, 0]), np.array([1, 0]), np.array([0.1, 0.1]), np.array([0, 1]), np.array([1, 0])),  # a tie must be 0
])
def test_agreement_refuses_placeholder_or_misaligned_inputs(y_main, y_new, svi, match, exact):
    with pytest.raises(ValueError):
        er.agreement(y_main, y_new, svi, match, exact_zero_new=exact)


# ----------------------------------------------------------------------------- rows: E4 labels of non-TEST rows
def _q_rows():
    # TRAIN fold 0 (oof0), Q_DEV (final), Q_CAL (final), TEST (final), TRAIN fold 1 (oof1)
    return {"engagement_key": np.array([b"a:c0", b"b:c0", b"c:c0", b"d:c0", b"e:c0"]),
            "match_index": np.array([0, 1, 2, 3, 4]), "role": np.array([0, 3, 4, 5, 0], dtype=np.int8),
            "fold": np.array([0, -1, -1, -1, 1], dtype=np.int8), "query_pre": np.array([10, 20, 30, 40, 50])}


def _preds():
    nan = np.nan   # TEST rows' probabilities: any computation over them would fail the validity checks
    return {"final": {"query_index": np.array([20, 21, 30, 31, 40, 41]), "raw_p": np.array([0.5, 0.6, 0.4, 0.4, nan, nan])},
            "oof0": {"query_index": np.array([9, 10, 11]), "raw_p": np.array([0.1, 0.7, 0.65])},
            "oof1": {"query_index": np.array([50, 51]), "raw_p": np.array([0.3, 0.3 + 2.0 ** -54])}}


def test_nontest_labels_use_each_rows_context_and_skip_test_rows():
    rows, preds = _q_rows(), _preds()
    out = er.nontest_labels(rows, np.array([11, 21, 31, 41, 51]), preds)
    assert out["q_row"].tolist() == [0, 1, 2, 4]                                    # the TEST row is never labelled
    assert out["engagement_key"].tolist() == [b"a:c0", b"b:c0", b"c:c0", b"e:c0"]
    assert out["role"].tolist() == [0, 3, 4, 0] and out["context"].tolist() == [1, 0, 0, 2]
    assert out["p_pre"].tolist() == [0.7, 0.5, 0.4, 0.3]
    assert out["p_post"].tolist() == [0.65, 0.6, 0.4, 0.3 + 2.0 ** -54]
    assert out["y4"].tolist() == [0, 1, 0, 1]
    assert out["exact_zero"].tolist() == [0, 0, 1, 0]
    assert out["near_zero_bits"].tolist() == [0, 0, 7, 7]
    assert np.array_equal(out["label_svi_delta4"], out["p_post"] - out["p_pre"])
    assert out["query_post"].tolist() == [11, 21, 31, 51]


def test_nontest_labels_refuse_a_missing_query_or_context():
    rows, preds = _q_rows(), _preds()
    with pytest.raises(ValueError):                                                 # Q_DEV endpoint 22 is not scored
        er.nontest_labels(rows, np.array([11, 22, 31, 41, 51]), preds)
    del preds["oof1"]
    with pytest.raises(ValueError):
        er.nontest_labels(rows, np.array([11, 21, 31, 41, 51]), preds)


def test_probabilities_at_reads_only_the_requested_positions():
    pred = {"query_index": np.array([3, 5, 8]), "raw_p": np.array([0.2, np.nan, 0.9])}
    assert er.probabilities_at(pred, np.array([8, 3, 8]), "x").tolist() == [0.9, 0.2, 0.9]
    with pytest.raises(ValueError):
        er.probabilities_at(pred, np.array([5]), "x")                               # a non-finite stored value
    with pytest.raises(ValueError):
        er.probabilities_at(pred, np.array([4]), "x")                               # not in the table
    with pytest.raises(ValueError):
        er.probabilities_at({"query_index": np.zeros(0, dtype=np.int64), "raw_p": np.zeros(0)}, np.array([1]), "x")


# ----------------------------------------------------------------------------- rows: main labels (labels_1) for the same rows
def _labels_table():
    # label rows in engagement order: a V_DEV row (role 1) and an invalid TRAIN row are not row-table rows
    return {"canonical_match_key": np.array([b"a", b"v", b"b", b"x", b"c", b"d", b"e"]),
            "component_key": np.full(7, b"c0"), "match_index": np.array([0, 9, 1, 8, 2, 3, 4], dtype=np.int32),
            "role": np.array([0, 1, 3, 0, 4, 5, 0], dtype=np.int8), "fold": np.array([0, -1, -1, 2, -1, -1, 1], dtype=np.int8),
            "query_pre": np.array([10, 90, 20, 80, 30, 40, 50]), "valid": np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool),
            "direction_positive": np.array([1, 0, 0, 0, 1, 1, 0], dtype=np.int8),
            "svi_delta": np.array([0.03, 0.1, -0.2, 0.0, 0.004, 0.5, -0.01])}


def test_q_row_positions_match_the_row_table_by_key_and_order():
    labels, rows = _labels_table(), {**_q_rows(), "canonical_match_key": np.array([b"a", b"b", b"c", b"d", b"e"]),
                                     "component_key": np.full(5, b"c0")}
    assert er.q_row_positions(labels, rows).tolist() == [0, 2, 4, 5, 6]
    rows["engagement_key"] = rows["engagement_key"][[1, 0, 2, 3, 4]]
    with pytest.raises(ValueError):
        er.q_row_positions(labels, rows)


def test_main_labels_of_nontest_rows_and_their_row_table_copy():
    labels = _labels_table()
    rows = {**_q_rows(), "canonical_match_key": np.array([b"a", b"b", b"c", b"d", b"e"]), "component_key": np.full(5, b"c0"),
            "y": np.array([1, 0, 1, -1, 0], dtype=np.int8), "label_svi_delta": np.array([0.03, -0.2, 0.004, np.nan, -0.01])}
    main = er.main_labels(labels, rows, np.array([0, 1, 2, 4]))
    assert main["y"].tolist() == [1, 0, 1, 0] and main["y"].dtype == np.int8
    assert main["svi"].tolist() == [0.03, -0.2, 0.004, -0.01]
    without_copies = {name: value for name, value in rows.items() if name not in ("y", "label_svi_delta")}
    with pytest.raises(ValueError):                                                 # TEST main labels are sealed
        er.main_labels(labels, without_copies, np.array([0, 3]))
    assert er.main_labels(labels, without_copies, np.array([4]))["y"].tolist() == [0]
    rows["y"] = np.array([1, 1, 1, -1, 0], dtype=np.int8)                           # the row table's copy disagrees
    with pytest.raises(ValueError):
        er.main_labels(labels, rows, np.array([0, 1, 2, 4]))


# ----------------------------------------------------------------------------- driver (synthetic inputs under tmp_path)
CONTEXT_NAMES = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
ROLE_ROWS = {"TRAIN": 2, "Q_DEV": 3, "Q_CAL": 1, "TEST": 1}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)


def _write_inputs(root, *, leak=False, y_copy_differs=False):
    """Synthetic Task 10 outputs (E4_VSCORE.json, PRED_<ctx>.npz/.json), labels_1, the q row table and the membership.

    Engagement rows: TRAIN a (fold 0), V_DEV v, Q_DEV b:c0 and b:c1 (match 1), invalid TRAIN x, Q_CAL c, TEST d, TRAIN e
    (fold 1), Q_DEV f (match 7).  TEST d's probabilities are NaN, so a driver that read them would stop."""

    labels_dir, pred_dir, projection = root / "labels", root / "pred", root / "projection"
    for folder in (labels_dir, pred_dir, projection):
        folder.mkdir()
    cmk = np.array([b"a", b"v", b"b", b"b", b"x", b"c", b"d", b"e", b"f"])
    ck = np.array([b"c0", b"c0", b"c0", b"c1", b"c0", b"c0", b"c0", b"c0", b"c0"])
    match = np.array([0, 6, 1, 1, 5, 2, 3, 4, 7])
    role = np.array([0, 1, 3, 3, 0, 4, 5, 0, 3], dtype=np.int8)
    fold = np.array([0, -1, -1, -1, 2, -1, -1, 1, -1], dtype=np.int8)
    q_pre = np.array([10, 90, 20, 22, 80, 30, 40, 50, 60])
    q_post = q_pre + 1
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    direction = np.array([1, 0, 1, 0, -1, 0, 1, 0, 1], dtype=np.int8)
    svi = np.array([0.03, 0.1, 0.2, -0.004, np.nan, -0.02, 0.5, -0.01, 0.06])
    _save_npz(labels_dir / "ENGAGEMENTS.npz", canonical_match_key=cmk, component_key=ck, match_index=match, role=role,
              fold=fold, query_pre=q_pre, query_h90=q_post, valid_h90=valid)
    _save_npz(labels_dir / "LABELS_h90.npz", canonical_match_key=cmk, component_key=ck, match_index=match.astype(np.int32),
              role=role, fold=fold, query_pre=q_pre, query_post=q_post, valid=valid, direction_positive=direction, svi_delta=svi)
    np.save(labels_dir / "Q_DEV_H90_KEYS.npy", np.array([b"b:c0", b"b:c1", b"f:c0"]))
    (labels_dir / "LABELS_MANIFEST.json").write_text("{}", encoding="utf-8")
    lock = {"q_may_start": True, "locked_utc": "2026-10-06T00:00:00+00:00",
            "files": {name: _sha(labels_dir / name) for name in ("LABELS_h90.npz", "ENGAGEMENTS.npz", "Q_DEV_H90_KEYS.npy")}}
    (labels_dir / "LABEL_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    keep = np.flatnonzero(np.isin(role, (0, 3, 4, 5)) & valid)
    y, label_svi = direction[keep].copy(), svi[keep].copy()
    test = role[keep] == 5
    y[test], label_svi[test] = -1, np.nan                  # the row table's sealed TEST placeholders
    if y_copy_differs:
        y[1] = 1 - y[1]
    q_rows_path = root / "Q_ROWS_h90.npz"
    _save_npz(q_rows_path, canonical_match_key=cmk[keep], component_key=ck[keep],
              engagement_key=np.char.add(np.char.add(cmk[keep], b":"), ck[keep]), match_index=match[keep], role=role[keep],
              fold=fold[keep], query_pre=q_pre[keep], y=y, label_svi_delta=label_svi)
    membership = np.zeros((6, 8), dtype=np.uint8)
    membership[1, 0] = membership[2, 4] = 32               # TRAIN a: the held-out target of oof0; TRAIN e: of oof1
    if leak:
        membership[0, 7] = 8                               # final saw Q_DEV f's match in V_DEV
    np.save(projection / "CONTEXT_MEMBERSHIP.npy", membership)
    nan = np.nan
    tables = {"final": ([20, 21, 22, 23, 30, 31, 40, 41, 60, 61], [0.5, 0.6, 0.4, 0.4, 0.7, 0.65, nan, nan, 0.3, 0.2]),
              "oof0": ([10, 11], [0.45, 0.55]), "oof1": ([50, 51], [0.6, 0.6 + 2.0 ** -53])}
    by_context = {"final": {"Q_CAL": 1, "Q_DEV": 3, "TEST": 1}, "oof0": {"TRAIN": 1}, "oof1": {"TRAIN": 1}}
    row_inputs = {Path(path).as_posix(): _sha(path) for path in (q_rows_path, labels_dir / "ENGAGEMENTS.npz",
                                                                 labels_dir / "LABEL_LOCK.json", projection / "CONTEXT_MEMBERSHIP.npy")}
    contexts = {}
    for ctx in CONTEXT_NAMES:
        index, raw = tables.get(ctx, ([], []))
        _save_npz(pred_dir / f"PRED_{ctx}.npz", query_index=np.array(index, dtype=np.int64), raw_p=np.array(raw, dtype=np.float64))
        record = {"context": ctx, "columns": ["query_index", "raw_p"], "pred_sha256": _sha(pred_dir / f"PRED_{ctx}.npz"),
                  "inputs": row_inputs, "rows": {"q_rows_by_role": by_context.get(ctx, {})}}
        (pred_dir / f"PRED_{ctx}.json").write_text(json.dumps(record), encoding="utf-8")
        contexts[ctx] = {"pred_sha256": record["pred_sha256"], "record_sha256": _sha(pred_dir / f"PRED_{ctx}.json")}
    report = {"v_dev_passed": True, "oof_check": {"passed": True}, "smoke": None, "q_rows_by_role": ROLE_ROWS, "contexts": contexts}
    (pred_dir / "E4_VSCORE.json").write_text(json.dumps(report), encoding="utf-8")
    return {"pred_dir": pred_dir, "labels_dir": labels_dir, "q_rows_path": q_rows_path, "projection": projection,
            "role_rows": ROLE_ROWS}


def test_run_labels_nontest_rows_and_reports_the_q_dev_agreement(tmp_path):
    sources = _write_inputs(tmp_path)
    out = tmp_path / "out"
    assert er.run(out, **sources) == 0
    with np.load(out / er.LABELS_NAME, allow_pickle=False) as stored:
        table = {name: stored[name] for name in stored.files}
    assert table["q_row"].tolist() == [0, 1, 2, 3, 5, 6]                          # row-table row 4 (TEST d) is never labelled
    assert table["engagement_key"].tolist() == [b"a:c0", b"b:c0", b"b:c1", b"c:c0", b"e:c0", b"f:c0"]
    assert table["role"].tolist() == [0, 3, 3, 4, 0, 3] and table["context"].tolist() == [1, 0, 0, 0, 2, 0]
    assert table["p_pre"].tolist() == [0.45, 0.5, 0.4, 0.7, 0.6, 0.3]
    assert table["y4"].tolist() == [1, 1, 0, 0, 1, 0] and table["exact_zero"].tolist() == [0, 0, 1, 0, 0, 0]
    record = json.loads((out / er.AGREEMENT_NAME).read_text(encoding="utf-8"))
    overall = record["q_dev_agreement"]["overall"]                              # Q_DEV b:c0, b:c1 (match 1) and f (match 7)
    assert overall["rows"] == 3 and overall["matches"] == 2
    assert overall["share_equal"]["match_weighted"] == pytest.approx(0.5)
    assert overall["share_equal"]["row_weighted"] == pytest.approx(2 / 3)
    assert overall["kappa"]["row_weighted"] == pytest.approx(0.4)
    assert overall["exact_zero_share_new"] == {"match_weighted": pytest.approx(0.25), "row_weighted": pytest.approx(1 / 3)}
    assert {name: value["count"] for name, value in record["tiny_delta_by_role"].items()} == {"TRAIN": 1, "Q_DEV": 0, "Q_CAL": 0}
    assert set(record["labels"]["by_role"]) == {"TRAIN", "Q_DEV", "Q_CAL"}
    assert record["labels"]["sha256"] == _sha(out / er.LABELS_NAME)
    assert record["checks"]["oof_check"] == {**record["checks"]["oof_check"], "rows_checked": 6, "leaked": 0, "passed": True}
    before = {name: _sha(out / name) for name in (er.LABELS_NAME, er.AGREEMENT_NAME)}
    assert er.run(out, **sources) == 0                                          # written once: a rerun changes nothing
    assert {name: _sha(out / name) for name in before} == before


def test_run_blocks_before_writing_labels_when_a_task10_table_changed(tmp_path):
    sources = _write_inputs(tmp_path)
    _save_npz(sources["pred_dir"] / "PRED_oof0.npz", query_index=np.array([10, 11]), raw_p=np.array([0.45, 0.35]))
    out = tmp_path / "out"
    assert er.run(out, **sources) == 2
    assert sorted(path.name for path in out.iterdir()) == ["E4_LABELS_BLOCKED_inputs_1.json"]
    blocked = json.loads((out / "E4_LABELS_BLOCKED_inputs_1.json").read_text(encoding="utf-8"))
    assert any("PRED_oof0.npz" in reason for reason in blocked["reasons"])


@pytest.mark.parametrize("change, reason", [("leak", "OOF exclusion failed on 1 rows"),
                                            ("y_copy", "the row table's y differs")])
def test_run_blocks_at_the_rows_stage(tmp_path, change, reason):
    sources = _write_inputs(tmp_path, leak=change == "leak", y_copy_differs=change == "y_copy")
    out = tmp_path / "out"
    assert er.run(out, **sources) == 2
    assert sorted(path.name for path in out.iterdir()) == ["E4_LABELS_BLOCKED_rows_1.json"]
    blocked = json.loads((out / "E4_LABELS_BLOCKED_rows_1.json").read_text(encoding="utf-8"))
    assert any(reason in item for item in blocked["reasons"])
