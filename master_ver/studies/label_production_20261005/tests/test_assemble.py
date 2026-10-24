"""Label rows follow the contract: same-context SVI, no sign tolerance, exact zero, NA, whole-game exclusion."""
import numpy as np

from studies.label_production_20261005 import assemble, engagements, queries


def _engagements():
    # rows: TRAIN fold0 (+), TRAIN fold0 (exact zero), Q_DEV (-), V_DEV (not a q row), TEST (missing post prediction)
    n = 5
    eng = {
        "canonical_match_key": np.asarray([b"a", b"a", b"b", b"c", b"d"], dtype="S64"),
        "component_key": np.asarray([b"1", b"2", b"1", b"1", b"1"], dtype="S64"),
        "match_index": np.asarray([0, 0, 1, 2, 3], dtype=np.int32),
        "role": np.asarray([0, 0, 3, 1, 5], dtype=np.int8), "fold": np.asarray([0, 0, -1, -1, -1], dtype=np.int8),
        "query_pre": np.asarray([10, 12, 20, 30, 40]), "global_isolation_verified": np.ones(n, dtype=bool),
    }
    for name in ("first_kill_ms", "last_kill_ms", "pre_ms", "F_pre_ms", "pre_frame_age_ms", "first_kill_frame_lag_ms"):
        eng[name] = np.arange(n, dtype=np.int64)
    for h in engagements.HORIZONS:
        eng[f"query_h{h}"] = np.asarray([11, 13, 21, 31, 41])
        eng[f"valid_h{h}"] = np.ones(n, dtype=bool)
        for name in ("post_ms", "F_post_ms", "post_frame_age_ms", "actual_followup_ms", "comparison_elapsed_ms", "truncation_bits"):
            eng[f"{name}_h{h}"] = np.arange(n, dtype=np.int64)
    return eng


def _pred(query_index, v):
    v = np.asarray(v, dtype=np.float64)
    return {"query_index": np.asarray(query_index, dtype=np.int64), "logit": np.log(v / (1 - v)), "raw_p": v, "v": v}


def _membership():
    bits = np.zeros((6, 4), dtype=np.uint8)
    bits[1, 0] = 32           # match 0 is oof0's held-out target
    bits[0, 2] = 8            # match 2 (V_DEV) is a dev game of final
    return bits


def test_label_rows_follow_the_contract():
    preds = {"oof0": _pred([10, 11, 12, 13], [0.40, 0.55, 0.30, 0.30]),
             "final": _pred([20, 21, 40], [0.60, 0.45, 0.50])}
    table = assemble.label_table(_engagements(), preds, _membership(), 90)
    assert table["context"].tolist() == [1, 1, 0, -1, 0]
    assert table["valid"].tolist() == [True, True, True, False, False]
    assert np.isclose(table["svi_delta"][0], 0.15) and table["direction_positive"][0] == 1
    assert table["svi_delta"][1] == 0.0 and table["direction_positive"][1] == 0 and table["exact_zero"][1] == 1
    assert table["direction_positive"][2] == 0 and table["exact_zero"][2] == 0
    assert np.isnan(table["svi_delta"][3]) and table["direction_positive"][3] == -1
    assert table["invalid_bits"][3] & assemble.INVALID["v_role_game_not_a_q_row"]
    assert table["invalid_bits"][4] & assemble.INVALID["missing_prediction"]
    assert table["near_zero_bits"][1] == 7 and table["near_zero_bits"][0] == 0
    assert table["clip_saturation_tie_bits"][1] & assemble.FLAGS["calibrated_tie"]


def test_a_game_the_scoring_v_saw_is_never_labelled():
    membership = _membership()
    membership[1, 0] = 32 | 1  # oof0 also fitted on match 0: whole-game exclusion broken
    preds = {"oof0": _pred([10, 11, 12, 13], [0.40, 0.55, 0.30, 0.30]), "final": _pred([20, 21, 40, 41], [0.6, 0.45, 0.5, 0.5])}
    table = assemble.label_table(_engagements(), preds, membership, 90)
    assert not table["valid"][0] and table["invalid_bits"][0] & assemble.INVALID["oof_exclusion_failed"]
    assert table["valid"][4]


def test_context_query_sets_cover_own_rows_and_the_stability_keys():
    eng = _engagements()
    assert queries.context_queries(eng, "final").tolist() == [20, 21, 40, 41]
    assert queries.context_queries(eng, "oof0").tolist() == [10, 11, 12, 13, 20, 21]   # own fold + Q_DEV pre/h90
    assert queries.context_queries(eng, "oof1").tolist() == [20, 21]


def test_endpoint_rule_and_truncation_bits():
    assert engagements.endpoint(100_000, 90, -1, -1, -1) == (190_000, 1)
    assert engagements.endpoint(100_000, 90, 150_000, 150_000, -1) == (149_999, 2 | 4)
    assert engagements.endpoint(100_000, 90, -1, -1, 120_000) == (119_999, 8)
