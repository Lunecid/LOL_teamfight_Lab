"""Task 15 (T1.4): E1 placebo labels, the placebo and size-matched engagement row tables, the placebo raw store and XY,
the balance tables and the q_prepare plumbing (studies/tog_experiments_20261012/e1_rows.py), synthetic data only.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md (amended 2026-10-13, re-approved; section 3
균형, 라벨, 입력과 전처리, 실행 세부; section 6 step 3; section 7); SDD Rulings 28, 48, 54.  No test opens a real sealed
file, a real TEST label or a real tog_e1_1 output.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from studies.label_production_20261005 import assemble, controls_states as cs
from studies.label_production_20261005.controls_states import EndState
from studies.observable_state_20261002.kernel import STATIC_NAMES
from studies.q_production_20261005 import features as qf, rows as q_rows_module
from studies.tog_experiments_20261012 import e1_rows as er1, e1_states as es
from studies.v_input_projection_20261002.bank import QUERY_SPECS, STATIC_SPECS
from studies.v_model_fit_input_20261003.common import ITEM_GROUPS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest().encode("ascii")


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _keys_of(value):
    """Every dictionary key of a JSON-like value, recursively."""

    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys_of(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys_of(item)}
    return set()


# ----------------------------------------------------------------------------- a small q_run_1 table, windows and V
ROLES = (0, 0, 3, 3, 4, 5, 5, 0)      # rows 0-1 TRAIN (match 0, fold 0), 2-3 Q_DEV (match 1), 4 Q_CAL (match 2),
MATCH = (0, 0, 1, 1, 2, 3, 3, 4)      # 5-6 TEST (match 3), 7 TRAIN (match 4, fold 2)
FOLD = (0, 0, -1, -1, -1, -1, -1, 2)
S_MS = {0: 250_000, 2: 640_000, 3: 700_000, 5: 720_000, 7: 1_450_000}   # rows 1, 4 and 6 have no placebo window
MATCHED = tuple(S_MS)
P_START = (0.40, 0.55, 0.61, 0.47, 0.33)
P_END = (0.43, 0.554, 0.61, 0.455, 0.30)   # the Q_DEV row 3 ties; the fourth value is the TEST row's (sealed)


# match 0 (TRAIN) is a q inner STOP match, so a wrong q_stop shows
STOP_TEXT = next(f"stop{i}" for i in range(1000) if q_rows_module.q_inner_stop(_hex(f"stop{i}").decode("ascii")))


def _match_key(m):
    return _hex(STOP_TEXT) if m == 0 else _hex(f"match{m}")


def _q_rows():
    n = len(ROLES)
    cmk = np.array([_match_key(m) for m in MATCH], dtype="S64")
    ck = np.array([_hex(f"component{i}") for i in range(n)], dtype="S64")
    role, match = np.array(ROLES, dtype=np.int8), np.array(MATCH, dtype=np.int64)
    test = role == 5
    pre_ms = np.array([300_000, 420_000, 600_000, 660_000, 900_000, 700_000, 760_000, 1_500_000], dtype=np.int64)
    delta = np.array([0.03, -0.01, 0.002, 0.0, 0.05, np.nan, np.nan, -0.2])
    near = np.zeros(n, dtype=np.uint8)
    for bit, threshold in enumerate(assemble.NEAR_ZERO):
        near[~test & (np.abs(np.nan_to_num(delta)) < threshold)] |= 1 << bit
    return {"canonical_match_key": cmk, "component_key": ck, "engagement_key": np.char.add(np.char.add(cmk, b":"), ck),
            "match_index": match, "role": role, "fold": np.array(FOLD, dtype=np.int8),
            "q_stop": np.array([bool(r == 0 and q_rows_module.q_inner_stop(k.decode("ascii"))) for k, r in zip(cmk, role)]),
            "y": np.where(test, -1, np.nan_to_num(delta) > 0).astype(np.int8), "weight_raw": q_rows_module.match_weights(match),
            "query_pre": 100 + 7 * np.arange(n, dtype=np.int64), "pre_ms": pre_ms, "elapsed_min": pre_ms / 60000.0,
            "frame_age_s": np.array([5.7, 4.2, 2.5, 0.0, 9.9, 3.3, 7.7, 1.0]),
            "truncation_bits": np.array([1, 4, 1, 2, 1, 1, 4, 1], dtype=np.int64), "near_zero_bits": near,
            "exact_zero": np.where(test, -1, np.nan_to_num(delta, nan=1.0) == 0).astype(np.int8), "label_svi_delta": delta,
            "pt_p_pre": np.array([0.52, 0.47, 0.05, 0.61, 0.5, 0.45, 0.44, 0.9]),
            "eval_n_kills": np.array([2, 1, 1, 3, 1, 2, 1, 4], dtype=np.int16),
            "eval_firstkill_credit_blue": np.array([2, 1, 1, 2, 1, 1, 1, 3], dtype=np.int16),
            "eval_firstkill_credit_red": np.ones(n, dtype=np.int16)}


def _windows(q):
    n = len(ROLES)
    s = np.full(n, -1, dtype=np.int64)
    for row, value in S_MS.items():
        s[row] = value
    matched = s >= 0

    def plc(values):                      # a placebo-side column: -1 without a window
        return np.where(matched, np.asarray(values), -1).astype(np.int64)

    return {"engagement_key": q["engagement_key"], "match_index": q["match_index"].astype(np.int32), "role": q["role"],
            "tau_ms": q["pre_ms"].copy(), "D_ms": np.array([20_000, 30_000, 25_000, 40_000, 35_000, 28_000, 33_000, 45_000]),
            "s_ms": s, "start_age_bin": np.array([1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int8),
            "n_bound": np.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=np.int16),
            "end_age_bin": np.array([0, 1, 1, 2, 0, 1, 0, 2], dtype=np.int8),
            "time_bin": np.array([0, 0, 0, 0, 1, 0, 0, 2], dtype=np.int8),
            "kill_gap_bin": np.array([0, 4, 0, 1, 2, 0, 3, 4], dtype=np.int8),
            "gold_bin": np.array([2, 1, 0, 4, 3, 2, 2, 1], dtype=np.int8),
            "start_age_eng_ms": np.array([5_700, 4_200, 2_500, 0, 9_900, 3_300, 7_700, 1_000], dtype=np.int64),
            "start_age_plc_ms": plc([3_000, 0, 1_000, 9_500, 0, 4_000, 0, 5_000]),
            "kill_gap_eng_ms": np.array([12_000, -1, 25_000, 40_000, 70_000, 9_000, 100_000, -1], dtype=np.int64),
            "kill_gap_plc_ms": plc([10_000, 0, 28_000, 35_000, 0, 7_000, 0, -1]),
            "gold_eng": np.array([150.0, -20.0, 300.0, -1_200.0, 900.0, 40.0, 0.0, 2_500.0]),
            "gold_plc": np.where(matched, np.array([90.0, 0.0, 250.0, -1_000.0, 0.0, 60.0, 0.0, 2_400.0]), np.nan),
            "n_candidates": np.full(n, 1_201, dtype=np.int32), "n_survivors": np.where(matched, 3, 0).astype(np.int32),
            "reject_counts": np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int32), (n, 1)),
            "xy_finite_plc": np.where(matched, 1, -1).astype(np.int8),
            "meets_component_window_plc": np.where(matched, 1, -1).astype(np.int8)}


def _pred(q, w, p_start=P_START, p_end=P_END):
    m = np.flatnonzero(w["s_ms"] >= 0)
    role, fold = q["role"][m], q["fold"][m]
    test = role == 5
    p_end = np.array(p_end)
    public = {"window_id": m.astype(np.int64), "engagement_key": q["engagement_key"][m], "match_index": q["match_index"][m],
              "role": role, "fold": fold, "ctx": es.context_codes(role, fold), "s_ms": w["s_ms"][m],
              "end_ms": w["s_ms"][m] + w["D_ms"][m], "p_start": np.array(p_start), "p_end": np.where(test, np.nan, p_end)}
    sealed = {"window_id": m[test].astype(np.int64), "engagement_key": q["engagement_key"][m][test], "p_end": p_end[test]}
    return public, sealed


def _placebo():
    q = _q_rows()
    w = _windows(q)
    pred, sealed = _pred(q, w)
    full, matched = er1.placebo_table(q, w, pred, sealed)
    public, sealed_labels = q_rows_module.seal_test(full)
    return q, w, pred, full, matched, public, sealed_labels


# ----------------------------------------------------------------------------- labels (assemble.py:121-127)
def _frozen_label_table(p_pre, p_post):
    """The frozen main-label code (assemble.label_table) on Q_DEV rows whose final-context V are given."""

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


def test_placebo_labels_mirror_the_frozen_main_label_code():
    rng = np.random.default_rng(15)
    start = np.concatenate([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2], rng.uniform(0, 1, 300)])
    end = np.concatenate([[0.5, 0.504, 0.496, 0.509, 0.519, 0.53, 0.47, 0.2],
                          np.clip(start[8:] + rng.normal(0, 0.012, 300), 0, 1)])
    labels = er1.placebo_labels(start, end)
    table = _frozen_label_table(start, end)
    assert table["valid"].all()
    for ours, frozen in (("y", "direction_positive"), ("exact_zero", "exact_zero"), ("near_zero_bits", "near_zero_bits"),
                         ("label_svi_delta", "svi_delta")):
        assert labels[ours].dtype == table[frozen].dtype
        np.testing.assert_array_equal(labels[ours], table[frozen])
    assert labels["y"][:8].tolist() == [0, 1, 0, 1, 1, 1, 0, 0]              # an exact zero is negative
    assert labels["exact_zero"][:8].tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    assert labels["near_zero_bits"][:8].tolist() == [7, 7, 7, 6, 4, 0, 0, 7]   # bit k when |W| < (0.005, 0.01, 0.02)[k]


def test_placebo_labels_refuse_an_invalid_probability():
    with pytest.raises(ValueError):
        er1.placebo_labels(np.array([0.5, np.nan]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        er1.placebo_labels(np.array([0.5, 0.5]), np.array([0.5, 1.2]))


# ----------------------------------------------------------------------------- the placebo row table
def test_placebo_table_has_the_q_run_1_columns_and_follows_the_window_order():
    q, w, pred, full, matched, public, _ = _placebo()
    assert matched.tolist() == list(MATCHED)
    assert list(full) == list(q) == list(er1.COLUMN_MAP)
    assert all(full[name].dtype == q[name].dtype for name in q)
    n = len(MATCHED)
    for name in ("canonical_match_key", "match_index", "role", "fold"):          # the matched engagement's identity
        np.testing.assert_array_equal(full[name], q[name][matched])
    np.testing.assert_array_equal(full["query_pre"], np.arange(n))               # the synthetic raw-store key
    np.testing.assert_array_equal(full["pre_ms"], w["s_ms"][matched])            # pre_ms = s
    np.testing.assert_array_equal(full["elapsed_min"], w["s_ms"][matched] / 60000.0)
    np.testing.assert_array_equal(full["frame_age_s"], w["start_age_plc_ms"][matched] / 1000.0)
    np.testing.assert_array_equal(full["pt_p_pre"], np.array(P_START))            # pt_p_pre = V(s)
    np.testing.assert_array_equal(full["weight_raw"], q_rows_module.match_weights(q["match_index"][matched]))
    assert full["weight_raw"].tolist() == [1.0, 0.5, 0.5, 1.0, 1.0]              # match 0 keeps one row of two
    expected_stop = [bool(r == 0 and q_rows_module.q_inner_stop(k.decode("ascii")))
                     for k, r in zip(q["canonical_match_key"][matched], q["role"][matched])]
    assert full["q_stop"].tolist() == expected_stop and expected_stop[0] is True
    for name in ("truncation_bits", "eval_n_kills", "eval_firstkill_credit_blue", "eval_firstkill_credit_red"):
        assert not full[name].any()
    w_label = np.array(P_END) - np.array(P_START)
    np.testing.assert_array_equal(full["label_svi_delta"], w_label)              # labels for every row before the seal
    assert full["y"].tolist() == [1, 1, 0, 0, 0] and full["exact_zero"].tolist() == [0, 0, 1, 0, 0]
    assert full["near_zero_bits"].tolist() == [0, 7, 7, 4, 0]


def test_test_labels_go_only_to_the_sealed_part():
    q, _, pred, full, matched, public, sealed = _placebo()
    test = public["role"] == 5
    assert test.tolist() == [False, False, False, True, False]
    assert public["y"][test].tolist() == [-1] and np.isnan(public["label_svi_delta"][test]).all()
    assert public["exact_zero"][test].tolist() == [-1] and public["near_zero_bits"][test].tolist() == [0]
    assert list(sealed) == ["engagement_key", *q_rows_module.SEALED_COLUMNS]
    np.testing.assert_array_equal(sealed["engagement_key"], public["engagement_key"][test])
    assert sealed["label_svi_delta"].tolist() == [P_END[3] - P_START[3]] and sealed["y"].tolist() == [0]
    assert sealed["near_zero_bits"].tolist() == [4]
    np.testing.assert_array_equal(public["pt_p_pre"], np.array(P_START))          # V(s) stays public for TEST too
    assert er1.placebo_problems(q, public, sealed) == []
    nontest = ~test
    for name in ("y", "label_svi_delta", "exact_zero", "near_zero_bits"):
        np.testing.assert_array_equal(public[name][nontest], full[name][nontest])


def test_placebo_problems_catch_an_unsealed_or_inconsistent_table():
    q, _, _, full, _, public, sealed = _placebo()
    unsealed = {name: value.copy() for name, value in public.items()}
    unsealed["y"][3] = 0
    assert any("placeholder" in p for p in er1.placebo_problems(q, unsealed, sealed))
    flipped = {name: value.copy() for name, value in public.items()}
    flipped["y"][0] = 0
    assert any("y" in p for p in er1.placebo_problems(q, flipped, sealed))
    reweighted = {name: value.copy() for name, value in public.items()}
    reweighted["weight_raw"][0] = 0.5
    assert any("weight_raw" in p for p in er1.placebo_problems(q, reweighted, sealed))
    renamed = {("Y" if name == "y" else name): value for name, value in public.items()}
    assert er1.placebo_problems(q, renamed, sealed)


@pytest.mark.parametrize("change, reason", [
    (lambda q, w, p, s: p.update(window_id=p["window_id"] + 1), "matched windows"),
    (lambda q, w, p, s: p.update(engagement_key=p["engagement_key"][::-1].copy()), "key mismatch"),
    (lambda q, w, p, s: p.update(end_ms=p["end_ms"] + 1), "window times"),
    (lambda q, w, p, s: p.update(s_ms=p["s_ms"] + 1), "window times"),
    (lambda q, w, p, s: p.update(ctx=np.zeros_like(p["ctx"])), "V context"),
    (lambda q, w, p, s: p.update(p_end=np.where(np.isnan(p["p_end"]), 0.5, p["p_end"])), "TEST end value"),
    (lambda q, w, p, s: s.update(window_id=s["window_id"] + 1), "sealed TEST end values"),
    (lambda q, w, p, s: s.update(p_end=np.array([1.5])), "finite probability"),
    (lambda q, w, p, s: w.update(match_index=w["match_index"] + 1), "WINDOWS_h90"),
])
def test_placebo_table_blocks_on_misaligned_or_unsealed_inputs(change, reason):
    q = _q_rows()
    w = _windows(q)
    pred, sealed = _pred(q, w)
    change(q, w, pred, sealed)
    with pytest.raises(er1.Blocked, match=reason):
        er1.placebo_table(q, w, pred, sealed)


def test_conform_refuses_a_missing_extra_or_lossy_column():
    q = _q_rows()
    values = {name: value.copy() for name, value in q.items()}
    with pytest.raises(er1.Blocked, match="missing"):
        er1.conform({k: v for k, v in values.items() if k != "fold"}, q)
    with pytest.raises(er1.Blocked, match="extra"):
        er1.conform({**values, "objective": np.zeros(len(ROLES))}, q)
    lossy = dict(values, fold=np.full(len(ROLES), 300))
    with pytest.raises(er1.Blocked, match="fold"):
        er1.conform(lossy, q)
    assert list(er1.conform(values, q)) == list(q)


# ----------------------------------------------------------------------------- keys
def test_placebo_keys_fit_the_columns_and_never_equal_an_engagement_key():
    q = _q_rows()
    component, engagement = er1.placebo_keys(q["canonical_match_key"], q["component_key"])
    assert component.dtype == np.dtype("S64") and engagement.dtype == np.dtype("S129")
    assert all(bytes(key).startswith(b"plc-") and len(bytes(key)) == 64 for key in component)
    assert all(bytes(key)[4:] == bytes(source)[:60] for key, source in zip(component, q["component_key"]))
    np.testing.assert_array_equal(engagement, np.char.add(np.char.add(q["canonical_match_key"], b":"), component))
    assert not np.isin(engagement, q["engagement_key"]).any() and not np.isin(component, q["component_key"]).any()
    assert max(np.frombuffer(engagement.tobytes(), dtype=np.uint8)) < 128            # ASCII
    again, _ = er1.placebo_keys(q["canonical_match_key"], q["component_key"])
    np.testing.assert_array_equal(component, again)                                  # deterministic
    assert er1.key_problems(engagement, component, q["engagement_key"], q["component_key"]) == []


def test_key_problems_name_duplicates_and_collisions():
    q = _q_rows()
    component, engagement = er1.placebo_keys(q["canonical_match_key"], q["component_key"])
    duplicate = engagement.copy()
    duplicate[1] = duplicate[0]
    assert any("not unique" in p for p in er1.key_problems(duplicate, component, q["engagement_key"], q["component_key"]))
    clash = engagement.copy()
    clash[2] = q["engagement_key"][5]
    assert any("engagement key" in p for p in er1.key_problems(clash, component, q["engagement_key"], q["component_key"]))
    component_clash = component.copy()
    component_clash[0] = q["component_key"][4]
    assert any("component key" in p for p in er1.key_problems(engagement, component_clash, q["engagement_key"],
                                                               q["component_key"]))


def test_key_map_pairs_each_placebo_row_with_its_engagement():
    q, _, _, _, matched, public, _ = _placebo()
    eng, _ = er1.engagement_table(q, matched)
    key_map = er1.key_map(matched, public, eng)
    n = len(MATCHED)
    assert key_map["plc_row"].tolist() == key_map["eng_row"].tolist() == list(range(n))
    assert key_map["window_id"].tolist() == list(MATCHED)
    np.testing.assert_array_equal(key_map["engagement_key_eng"], q["engagement_key"][matched])
    np.testing.assert_array_equal(key_map["engagement_key_plc"], public["engagement_key"])
    assert er1.pair_problems(public, eng) == []
    moved = {name: value.copy() for name, value in eng.items()}
    moved["match_index"][0] = 9
    assert er1.pair_problems(public, moved)


# ----------------------------------------------------------------------------- the size-matched engagement table
def test_engagement_table_is_the_matched_q_run_1_rows_with_recomputed_weights():
    q = _q_rows()
    matched = np.array(MATCHED)
    eng, info = er1.engagement_table(q, matched)
    assert list(eng) == list(q) and info["q_stop_recomputed_equals_stored"] is True
    for name in q:
        if name == "weight_raw":
            continue
        np.testing.assert_array_equal(eng[name], q[name][matched])
        assert eng[name].dtype == q[name].dtype
    np.testing.assert_array_equal(eng["weight_raw"], q_rows_module.match_weights(q["match_index"][matched]))
    assert eng["weight_raw"][0] == 1.0 and q["weight_raw"][0] == 0.5                  # recomputed on the subset
    assert eng["y"][3] == -1 and np.isnan(eng["label_svi_delta"][3])                  # TEST stays sealed as in q_run_1
    assert er1.engagement_problems(q, matched, eng) == []
    changed = {name: value.copy() for name, value in eng.items()}
    changed["pt_p_pre"][0] = 0.9
    assert any("pt_p_pre" in p for p in er1.engagement_problems(q, matched, changed))


def test_engagement_table_blocks_when_the_stored_q_stop_differs():
    q = _q_rows()
    q["q_stop"] = ~q["q_stop"] & (q["role"] == 0)
    q["q_stop"][0] = not bool(q_rows_module.q_inner_stop(q["canonical_match_key"][0].decode("ascii")))
    with pytest.raises(er1.Blocked, match="q_stop"):
        er1.engagement_table(q, np.array(MATCHED))


# ----------------------------------------------------------------------------- the engagement XY subset
def test_subset_xy_takes_rows_by_engagement_key():
    q = _q_rows()
    source = np.random.default_rng(3).normal(size=(len(ROLES), 10, 2))
    matched = np.array(MATCHED)
    xy = er1.subset_xy(source, q["engagement_key"], q["engagement_key"][matched], matched)
    np.testing.assert_array_equal(xy, source[matched])
    assert xy.dtype == np.float64 and xy.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("change", ["swap", "missing", "window", "short", "altered"])
def test_subset_xy_blocks_on_a_row_or_key_mismatch(change):
    q = _q_rows()
    source = np.zeros((len(ROLES), 10, 2))
    matched = np.array(MATCHED)
    keys, window_id, source_keys = q["engagement_key"][matched].copy(), matched.copy(), q["engagement_key"]
    if change == "swap":
        keys[[0, 1]] = keys[[1, 0]]
    elif change == "missing":
        keys[2] = b"x" * 129
    elif change == "window":
        window_id[0] = 1
    elif change == "altered":                     # a key just below the original: found at the same row, yet not equal
        keys[0] = bytes(keys[0])[:-1] + b"/"
    else:
        source = source[:-1]
    with pytest.raises(er1.Blocked):
        er1.subset_xy(source, source_keys, keys, window_id)


# ----------------------------------------------------------------------------- balance
def test_smd_of_hand_checked_cases():
    out = er1.smd_summary(np.array([0.0, 2.0]), np.array([1.0, 3.0]))
    assert out["pairs"] == 2 and out["pooled_sd"] == 1.0 and out["smd"] == 1.0
    assert out["mean_engagement"] == 1.0 and out["mean_placebo"] == 2.0
    out = er1.smd_summary(np.array([0.0, 0.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert math.isclose(out["smd"], (1.0 - 1.5) / math.sqrt((2.25 + 0.0) / 2), rel_tol=0, abs_tol=1e-15)
    assert out["sd_placebo"] == 0.0
    assert er1.smd_summary(np.array([]), np.array([]))["smd"] is None


def test_objective_gain_is_the_diagnostics_definition():
    start = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    end = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.int32)
    gain = er1.objective_gain(start, end)
    assert gain.tolist() == [1, 0, 1] and gain.dtype == np.int64                     # diagnostics.py:155-158
    assert (gain > 0).tolist() == [True, False, True]


def _pairs(q, w, pred, objective_eng=None, objective_plc=None, delta_plc=None):
    matched = np.flatnonzero(w["s_ms"] >= 0)
    return {"window_id": matched.astype(np.int64), "role": q["role"][matched].astype(np.int64),
            "match_index": q["match_index"][matched], "v_eng": q["pt_p_pre"][matched], "v_plc": pred["p_start"],
            "obj_eng": np.array([True, False, True, True, False]) if objective_eng is None else objective_eng,
            "obj_plc": np.array([False, False, True, False, True]) if objective_plc is None else objective_plc,
            "delta_plc": (np.where(pred["role"] == 5, np.nan, np.array(P_END) - np.array(P_START)) if delta_plc is None
                          else delta_plc),
            "delta_eng": q["label_svi_delta"][matched]}


def test_balance_reports_the_prespec_quantities_per_role():
    q = _q_rows()
    w = _windows(q)
    pred, _ = _pred(q, w)
    public, sealed, summaries = er1.balance_tables(w, _pairs(q, w, pred))
    assert sorted(public) == ["Q_CAL", "Q_DEV", "TRAIN"] and sorted(sealed) == ["TEST"]
    train = public["TRAIN"]                                        # pairs: rows 0 and 7
    assert train["pairs"] == 2 and train["engagements"] == 3 and train["matched"] == 2
    smd = train["smd"]
    assert smd["v_start"]["smd"] == er1.smd_summary(q["pt_p_pre"][[0, 7]], np.array([0.40, 0.33]))["smd"]
    assert smd["game_time_min"]["smd"] == er1.smd_summary(q["pre_ms"][[0, 7]] / 60000.0,
                                                          np.array([250_000, 1_450_000]) / 60000.0)["smd"]
    assert smd["time_since_last_kill_s"]["pairs"] == 1                                # row 7 has no earlier kill
    assert smd["time_since_last_kill_s"]["pairs_dropped_without_earlier_kill"] == 1
    assert train["objective_share"]["engagement"] == {"windows": 2, "with_objective_gain": 1, "share": 0.5}
    assert train["objective_share"]["placebo"]["with_objective_gain"] == 1
    assert set(train["bins"]) == {"start_age_bin", "n_bound", "end_age_bin", "time_bin", "kill_gap_bin", "gold_bin"}
    assert train["bins"]["kill_gap_bin"]["4"] == {"engagements": 2, "matched": 1, "match_rate": 0.5}
    assert train["reuse"]["share_rows_in_reused_windows"] == 0.0
    assert set(train["triggers"]) == {"stratified_r", "deduplicated_r"}
    assert train["triggers"]["deduplicated_r"]["flag"] is False
    assert "extras" in train and "precision" in train
    assert summaries["TRAIN"]["kill_gap_balance"]["smd"] == smd["time_since_last_kill_s"]["smd"]
    q_dev = public["Q_DEV"]
    assert q_dev["precision"]["placebo"]["exact_zero"] == 1                           # the Q_DEV row 3 ties
    assert "extras" not in sealed["TEST"] and "precision" not in sealed["TEST"]


def test_stratified_r_trigger_uses_abs_smd_above_0_1():
    smd = {"v_start": {"smd": -0.11}, "game_time_min": {"smd": 0.05}}
    assert er1.triggers(smd, {"share_rows_in_reused_windows": 0.051})["stratified_r"]["flag"] is True
    assert er1.triggers(smd, {"share_rows_in_reused_windows": 0.051})["deduplicated_r"]["flag"] is True
    calm = {"v_start": {"smd": 0.1}, "game_time_min": {"smd": -0.1}}
    out = er1.triggers(calm, {"share_rows_in_reused_windows": 0.05})
    assert out["stratified_r"]["flag"] is False and out["deduplicated_r"]["flag"] is False


def test_public_balance_and_counts_hold_no_test_figure():
    q = _q_rows()
    w = _windows(q)
    pred, _ = _pred(q, w)
    public, _, _ = er1.balance_tables(w, _pairs(q, w, pred))
    assert "TEST" not in _keys_of(json.loads(json.dumps(es._json_safe(public))))
    # every TEST input changed (one more TEST window, other V, gold, bins, objectives): the public part does not move
    q2 = _q_rows()
    w2 = _windows(q2)
    test_rows = np.flatnonzero(q2["role"] == 5)
    q2["pt_p_pre"][test_rows] = 0.99
    w2["gold_eng"][test_rows] = 9e9
    w2["s_ms"][6] = 700_000                                                           # the unmatched TEST row gets a window
    w2["start_age_plc_ms"][6], w2["kill_gap_plc_ms"][6], w2["gold_plc"][6] = 1_000, 5_000, 1.0
    w2["kill_gap_bin"][test_rows] = 1
    start2, end2 = (0.40, 0.55, 0.61, 0.20, 0.80, 0.33), (0.43, 0.554, 0.61, 0.90, 0.10, 0.30)
    pred2, _ = _pred(q2, w2, start2, end2)                    # placebo rows: 0, 2, 3 (public), 5, 6 (TEST), 7 (public)
    pairs2 = _pairs(q2, w2, pred2, objective_eng=np.array([True, False, True, False, True, False]),
                    objective_plc=np.array([False, False, True, True, True, True]),
                    delta_plc=np.where(pred2["role"] == 5, np.nan, np.array(end2) - np.array(start2)))
    public2, sealed2, _ = er1.balance_tables(w2, pairs2)
    assert json.dumps(es._json_safe(public2), sort_keys=True) == json.dumps(es._json_safe(public), sort_keys=True)
    assert sealed2["TEST"]["pairs"] == 2
    # the row-table record counts: TRAIN, Q_DEV and Q_CAL only, no total row count
    _, _, _, _, _, table, _ = _placebo()
    counts = er1.nontest_counts(table)
    assert sorted(counts) == ["Q_CAL", "Q_DEV", "TRAIN"] and "rows" not in counts
    other = {name: value.copy() for name, value in table.items()}
    other["match_index"][3] = 77
    other["role"] = np.r_[other["role"], np.int8(5)]
    other["match_index"] = np.r_[other["match_index"], 78]
    other["y"] = np.r_[other["y"], np.int8(-1)]
    other["weight_raw"] = np.r_[other["weight_raw"], 1.0]
    other["q_stop"] = np.r_[other["q_stop"], False]
    other["exact_zero"] = np.r_[other["exact_zero"], np.int8(-1)]
    assert er1.nontest_counts(other) == counts


def test_balance_cross_check_compares_with_the_windows_record():
    q = _q_rows()
    w = _windows(q)
    pred, _ = _pred(q, w)
    _, _, summaries = er1.balance_tables(w, _pairs(q, w, pred))
    record = json.loads(json.dumps(es._json_safe(summaries)))
    assert er1.balance_cross_checks(summaries, record) == []
    record["Q_DEV"]["bins"]["gold_bin"]["0"]["matched"] = 0
    assert er1.balance_cross_checks(summaries, record) == ["Q_DEV: bins differs from WINDOWS_h90.json"]


def test_d_quartiles_and_medians():
    d = np.array([10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000])
    matched = np.array([True, True, True, False, True, False, False, False])
    out = er1.d_quartiles(d, matched)
    assert out["edges_s"] == [27.5, 45.0, 62.5]
    assert [out["quartiles"][str(k)]["engagements"] for k in range(4)] == [2, 2, 2, 2]
    assert [out["quartiles"][str(k)]["match_rate"] for k in range(4)] == [1.0, 0.5, 0.5, 0.0]
    gap = np.array([5_000, -1, 15_000, 40_000, 20_000, -1, 60_000, 90_000])
    med = er1.matched_medians(d, gap, matched)
    assert med["D_s"] == {"matched": 25.0, "unmatched": 65.0}
    assert med["time_since_last_kill_s"]["matched"] == 15.0 and med["time_since_last_kill_s"]["unmatched"] == 60.0
    assert med["time_since_last_kill_s"]["matched_without_earlier_kill"] == 1
    assert med["time_since_last_kill_s"]["unmatched_without_earlier_kill"] == 1


def test_precision_counts_exact_zeros_and_tiny_w():
    out = er1.precision_counts(np.array([0.0, 2e-7, -5e-7, 2e-6, 0.3, -0.0]))
    assert out == {"rows": 6, "exact_zero": 2, "abs_lt_1e-6": 4, "nonzero_abs_lt_1e-6": 2, "min_nonzero_abs": 2e-7}
    with pytest.raises(ValueError):
        er1.precision_counts(np.array([np.nan]))


# ----------------------------------------------------------------------------- state parts, the store chunks
def _end_state(time_ms):
    features = {name: np.zeros(tail, dtype=dtype) for name, (dtype, tail) in {**QUERY_SPECS, **STATIC_SPECS}.items()}
    features["elapsed_minutes"] = np.float64(time_ms / 60000.0)
    features["objective_counts"] = (np.arange(9) + time_ms // 60_000).astype(np.int32)
    features["items"] = {"item_ids": np.array([1001], dtype=np.int64), "counts": np.array([1], dtype=np.int64),
                         "offsets": np.array([0] + [1] * 10, dtype=np.int64), "recorded_ledger_supported": np.ones(10, dtype=bool),
                         "reason_mask": np.zeros(10, dtype=np.uint64), "unassigned_observed": False}
    return EndState(int(time_ms), features, "[]", "{}")


def _semantic_row(time_ms):
    token = json.dumps(["LEVEL_UP", "level", "value", ["int", int(time_ms // 1000) % 18 + 1]], separators=(",", ":"))
    empty = {group: [{} for _ in range(10)] for group in ITEM_GROUPS}
    return {"numeric": {"node": np.full((10, 102), time_ms / 1000.0), "team": np.ones((2, 8)), "global": np.arange(75) + time_ms},
            "categories": {"node": {name: np.arange(10) for name in STATIC_NAMES},
                           "team": {f"ban{k}": np.array([k, -1]) for k in range(5)}, "global": {}},
            "bags": {"node": {"current_item": [{1001: 1}] + [{} for _ in range(9)], **empty,
                              "nonitem": [{token: 2}] + [{} for _ in range(9)]},
                     "team": {"nonitem": [{token: 1}, {}]}, "global": {**{group: [{}] for group in ITEM_GROUPS}, "nonitem": [{}]}},
            "team_ids": np.array([100, 200] * 5), "focal_is_blue": True, "current_item_supported": np.ones(10, dtype=bool)}


def _fake_semantic_rows(states):
    return [_semantic_row(int(state.time_ms)) for state in states]


def _write_part(path, times, matches, windows, first_id=0, frames=None):
    """A state part as e1_states.build_part writes it (encode_states payload plus the plan and window columns)."""

    states = [_end_state(t) for t in times]
    payload = cs.encode_states(states)
    n = len(times)
    payload.update({"frame_ms": np.asarray(times, dtype=np.int64) - 1_000 if frames is None else np.asarray(frames, dtype=np.int64),
                    "xy": np.arange(n * 20, dtype=np.float64).reshape(n, 10, 2) + 1.0, "xy_support": np.ones(n, dtype=bool),
                    "state_id": np.arange(first_id, first_id + n, dtype=np.int64), "state_match": np.asarray(matches, dtype=np.int64),
                    "state_ms": np.asarray(times, dtype=np.int64),
                    "window_id": np.asarray([w for w, _, _ in windows], dtype=np.int64),
                    "window_start_state": np.asarray([first_id + a for _, a, _ in windows], dtype=np.int64),
                    "window_end_state": np.asarray([first_id + b for _, _, b in windows], dtype=np.int64),
                    "identity_json": es.identity_array({"synthetic": True})})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return es.write_bytes_once(Path(path), es.npz_bytes(payload, compress=True))


def test_store_chunk_writes_the_requested_states_in_key_order(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    part = tmp_path / "part_0000.npz"
    digest = _write_part(part, [130_000, 150_000, 170_000, 190_000], [7, 7, 7, 7], [(0, 0, 1), (1, 2, 3)])
    task = {"chunk_dir": str(tmp_path / "chunks" / "chunk_0000"), "identity": {"chunk": 0},
            "items": [{"path": str(part), "sha256": digest, "local": np.array([0, 2]), "keys": np.array([0, 1]),
                       "match": np.array([7, 7]), "ms": np.array([130_000, 170_000])}]}
    result = er1.store_chunk(task)
    assert result["reused"] is False
    store = qf.RawStore(tmp_path / "chunks" / "chunk_0000")
    assert store.queries.tolist() == [0, 1]
    np.testing.assert_array_equal(store.numeric["global"][1], _semantic_row(170_000)["numeric"]["global"])
    np.testing.assert_array_equal(store.numeric["node"][0], _semantic_row(130_000)["numeric"]["node"])
    assert store.record["identity"] == {"chunk": 0}
    assert er1.store_chunk(task)["reused"] is True                                   # a finished chunk is kept
    with pytest.raises(er1.Blocked, match="another chunk"):
        er1.store_chunk({**task, "identity": {"chunk": 1}})


@pytest.mark.parametrize("change, reason", [("sha256", "differs from its record"), ("ms", "differs from its window"),
                                            ("match", "differs from its window")])
def test_store_chunk_blocks_on_a_changed_part_or_a_wrong_state(tmp_path, monkeypatch, change, reason):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    part = tmp_path / "part_0000.npz"
    digest = _write_part(part, [130_000, 150_000], [7, 7], [(0, 0, 1)])
    item = {"path": str(part), "sha256": digest, "local": np.array([0]), "keys": np.array([0]), "match": np.array([7]),
            "ms": np.array([130_000])}
    item[change] = "0" * 64 if change == "sha256" else item[change] + 1
    with pytest.raises(er1.Blocked, match=reason):
        er1.store_chunk({"chunk_dir": str(tmp_path / "chunk"), "identity": {}, "items": [item]})


def test_an_unfinished_chunk_is_moved_aside_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    part = tmp_path / "part_0000.npz"
    digest = _write_part(part, [130_000, 150_000], [7, 7], [(0, 0, 1)])
    chunk = tmp_path / "chunks" / "chunk_0000"
    chunk.mkdir(parents=True)
    (chunk / "numeric_node.f64.npy").write_bytes(b"partial")
    er1.store_chunk({"chunk_dir": str(chunk), "identity": {}, "items": [
        {"path": str(part), "sha256": digest, "local": np.array([0]), "keys": np.array([0]), "match": np.array([7]),
         "ms": np.array([130_000])}]})
    aside = [p for p in chunk.parent.iterdir() if p.name.startswith("chunk_0000.partial_")]
    assert len(aside) == 1 and (aside[0] / "numeric_node.f64.npy").read_bytes() == b"partial"
    assert (chunk / "CHUNK.json").is_file()


def test_chunk_plan_groups_consecutive_parts_and_covers_each_row_once():
    parts = [{"keys": np.array([0, 1, 2])}, {"keys": np.array([3, 4])}, {"keys": np.array([5, 6, 7, 8])}]
    chunks = er1.chunk_plan(parts, parts_per_chunk=2)
    assert [[len(p["keys"]) for p in chunk] for chunk in chunks] == [[3, 2], [4]]
    for bad in ([{"keys": np.array([0, 2])}, {"keys": np.array([1, 3])}],           # interleaved
                [{"keys": np.array([0, 1])}, {"keys": np.array([3])}],              # a gap
                [{"keys": np.array([0, 1])}, {"keys": np.array([1, 2])}],           # a row twice
                [{"keys": np.array([0, 1])}, {"keys": np.array([], dtype=np.int64)}]):
        with pytest.raises(er1.Blocked):
            er1.chunk_plan(bad, parts_per_chunk=2)


def _states_record(directory, parts):
    return {"parts": {f"shard_{k:04d}": {"file": Path(path).name, "sha256": digest, "shard": k}
                      for k, (path, digest) in enumerate(parts)}}


def test_part_index_maps_each_placebo_row_to_its_start_and_end_states(tmp_path):
    # part 0 holds windows 2 and 5 (match 1), part 1 holds window 9 (match 4)
    a = _write_part(tmp_path / "part_0000.npz", [130_000, 160_000, 200_000, 230_000], [1, 1, 1, 1],
                    [(2, 0, 1), (5, 2, 3)], first_id=0)
    b = _write_part(tmp_path / "part_0001.npz", [300_000, 330_000], [4, 4], [(9, 0, 1)], first_id=4)
    record = _states_record(tmp_path, [(tmp_path / "part_0000.npz", a), (tmp_path / "part_0001.npz", b)])
    window_id, match = np.array([2, 5, 9]), np.array([1, 1, 4])
    s, end, age = np.array([130_000, 200_000, 300_000]), np.array([160_000, 230_000, 330_000]), np.array([1_000, 1_000, 1_000])
    index = er1.part_index(tmp_path, record, window_id, match, s, end, age)
    assert index["part"].tolist() == [0, 0, 1] and index["local_start"].tolist() == [0, 2, 0]
    assert index["local_end"].tolist() == [1, 3, 1]
    assert index["objective_start"][1].tolist() == (np.arange(9) + 3).tolist()       # 200 s -> minute 3
    assert index["objective_end"][2].tolist() == (np.arange(9) + 5).tolist()
    np.testing.assert_array_equal(index["xy"][2], np.arange(20, dtype=np.float64).reshape(10, 2) + 1.0)
    assert [f["file"] for f in index["files"]] == ["part_0000.npz", "part_0001.npz"]
    with pytest.raises(er1.Blocked, match="differs from its window"):
        er1.part_index(tmp_path, record, window_id, match, s + np.array([0, 1, 0]), end, age)
    with pytest.raises(er1.Blocked, match="start frame"):
        er1.part_index(tmp_path, record, window_id, match, s, end, age + 1)
    with pytest.raises(er1.Blocked, match="no state part"):
        er1.part_index(tmp_path, record, np.array([2, 5, 9, 11]), np.r_[match, 4], np.r_[s, 1], np.r_[end, 2], np.r_[age, 1])


# ----------------------------------------------------------------------------- q_run_1 untouched, q_prepare
def test_tree_hashes_detect_a_changed_added_or_removed_file(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"2")
    before = er1.tree_hashes(tmp_path)
    assert sorted(before) == ["a.bin", "sub/b.bin"] and er1.tree_changes(before, er1.tree_hashes(tmp_path)) == []
    (tmp_path / "a.bin").write_bytes(b"3")
    (tmp_path / "c.bin").write_bytes(b"4")
    (tmp_path / "sub" / "b.bin").unlink()
    assert er1.tree_changes(before, er1.tree_hashes(tmp_path)) == ["changed a.bin", "added c.bin", "removed sub/b.bin"]
    assert er1.tree_digest(before) == er1.tree_digest(dict(reversed(list(before.items()))))


def test_q_prepare_is_the_frozen_cli_with_cuda_hidden(tmp_path):
    command = er1.q_prepare_command(tmp_path / "plc")
    assert command == [sys.executable, "-B", "studies/q_production_20261005/q_prepare.py", "--run", (tmp_path / "plc").as_posix(),
                       "--variant", "main_h90", "--release", "reports/q_production_20261005/SCIENTIFIC_Q_15638e1f9d2fc7ce.json"]
    env = er1.prepare_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["OMP_NUM_THREADS"] == "2"


def test_prepared_problems_check_hashes_scopes_and_release():
    prepared = {"rows_file": "Q_ROWS_h90.npz", "rows_sha256": "r", "xy_file": "inputs/main_h90/XY.npy", "xy_sha256": "x",
                "store_build_sha256": "b", "release_digest": er1.RELEASE_DIGEST, "variant": "main_h90",
                "contexts": {"FIT": {"scope_rows": 9}, "REFIT_TRAIN": {"scope_rows": 10}}}
    expect = {"rows_sha256": "r", "xy_sha256": "x", "store_build_sha256": "b", "fit_rows": 9, "train_rows": 10}
    assert er1.prepared_problems(prepared, expect) == []
    assert er1.prepared_problems({**prepared, "xy_sha256": "y"}, expect) == ["PREPARED.json xy_sha256 is not the XY written before"]
    bad = {**prepared, "contexts": {"FIT": {"scope_rows": 10}, "REFIT_TRAIN": {"scope_rows": 10}}}
    assert er1.prepared_problems(bad, expect) == ["FIT scope rows are not the non-STOP TRAIN rows"]


# ----------------------------------------------------------------------------- the rows stage end to end
def _write_npz(path, arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return es.write_bytes_once(Path(path), es.npz_bytes(arrays, compress=False))


def _stage_inputs(tmp_path):
    q = _q_rows()
    w = _windows(q)
    pred, sealed = _pred(q, w)
    source, q_run = tmp_path / "tog_e1_1", tmp_path / "q_run_1"
    q_sha = _write_npz(q_run / "Q_ROWS_h90.npz", q)
    w_sha = _write_npz(source / "WINDOWS_h90.npz", w)
    (source / "WINDOWS_h90.json").write_text(json.dumps({"windows_npz": {"sha256": w_sha}, "stage": "final_windows", "smoke": False,
                                                         "xy_check": {"passed": True}, "gate_record": {"passed": True}}))
    p_sha = _write_npz(source / "PRED_PLACEBO_h90.npz", pred)
    sealed_sha = _write_npz(source / "PRED_PLACEBO_h90_TEST_SEALED.npz", sealed)
    (source / "STATES_h90").mkdir()
    (source / "STATES_h90" / "STATES_h90.json").write_text(json.dumps({"stage": "states", "smoke": False, "parts": {},
                                                                       "windows": {"sha256": w_sha}, "q_rows": {"sha256": q_sha}}))
    (source / "PRED_PLACEBO_h90.json").write_text(json.dumps({
        "stage": "placebo_v", "smoke": False, "outputs": {"public": {"sha256": p_sha}, "sealed": {"sha256": sealed_sha}},
        "inputs": {"windows_npz": w_sha, "q_rows": q_sha},
        "states": {"record_sha256": _sha(source / "STATES_h90" / "STATES_h90.json")}}))
    return er1.Paths(source=source, out=source, q_run=q_run, labels=tmp_path / "labels_1", projection=tmp_path / "projection")


def test_rows_stage_writes_both_tables_the_sealed_file_and_the_records(tmp_path):
    paths = _stage_inputs(tmp_path)
    assert er1.rows_stage(paths, expected={}) == 0
    plc, eng = _load(paths.plc / "Q_ROWS_h90.npz"), _load(paths.eng / "Q_ROWS_h90.npz")
    sealed = _load(paths.plc / "Q_TEST_SEALED_h90.npz")
    assert list(plc) == list(_q_rows()) == list(eng)
    assert plc["y"][3] == -1 and sealed["y"].tolist() == [0] and np.isnan(eng["label_svi_delta"][3])
    key_map = _load(paths.plc / "PLC_KEY_MAP_h90.npz")
    assert key_map["window_id"].tolist() == list(MATCHED)
    stage = json.loads((paths.out / "E1_ROWS.json").read_text(encoding="utf-8"))
    assert stage["passed"] is True
    assert {entry["sha256"] for entry in stage["files"].values()} >= {_sha(paths.plc / "Q_ROWS_h90.npz"),
                                                                      _sha(paths.plc / "Q_TEST_SEALED_h90.npz")}
    text = (paths.out / "E1_ROWS.json").read_text(encoding="utf-8")
    for record in (stage, json.loads((paths.plc / "Q_ROWS_h90.json").read_text(encoding="utf-8")),
                   json.loads((paths.eng / "Q_ROWS_h90.json").read_text(encoding="utf-8"))):
        assert "TEST" not in _keys_of(record)
    assert stage["counts"]["plc"]["TRAIN"] == {"rows": 2, "matches": 2}
    assert '"TEST"' not in text
    assert er1.rows_stage(paths, expected={}) == 0                                    # a finished stage is verified, kept


def test_rows_stage_is_blocked_by_a_changed_input_and_writes_the_record(tmp_path, capsys):
    paths = _stage_inputs(tmp_path)
    (paths.source / "PRED_PLACEBO_h90.npz").write_bytes(b"changed")
    with pytest.raises(er1.Blocked, match="PRED_PLACEBO_h90.npz"):
        er1.rows_stage(paths, expected={})
    assert er1.run_stage("rows", paths, expected={}) == 2
    blocked = list(paths.out.glob("E1_ROWS_BLOCKED_rows_*.json"))
    assert len(blocked) == 1 and json.loads(blocked[0].read_text(encoding="utf-8"))["blocked"] is True
    assert not (paths.plc / "Q_ROWS_h90.npz").exists()


def test_rows_stage_checks_the_dispatch_hashes_when_given(tmp_path):
    paths = _stage_inputs(tmp_path)
    with pytest.raises(er1.Blocked, match="dispatch"):
        er1.rows_stage(paths, expected={"windows": "0" * 64})


# ----------------------------------------------------------------------------- the store check sample
def test_check_sample_takes_the_smallest_sha256_per_public_role_and_never_test():
    rng = np.random.default_rng(4)
    n = 400
    keys = np.array([_hex(f"m{i}") + b":" + _hex(f"c{i}") for i in range(n)], dtype="S129")
    role = rng.choice(np.array([0, 3, 4, 5], dtype=np.int8), size=n)
    matched = rng.random(n) < 0.6
    sample = er1.check_sample(keys, role, matched, per_role=7)
    assert len(sample) == 21 and np.all(np.diff(sample) > 0)
    assert not np.isin(role[sample], [5]).any() and matched[sample].all()
    for code in (0, 3, 4):
        rows = np.flatnonzero((role == code) & matched)
        digest = {int(r): hashlib.sha256(f"{er1.CHECK_SALT}:{keys[r].decode()}".encode()).hexdigest() for r in rows}
        expected = sorted(sorted(rows.tolist(), key=digest.get)[:7])
        assert sample[role[sample] == code].tolist() == expected
    with pytest.raises(ValueError, match="fewer than"):
        er1.check_sample(keys, role, matched, per_role=1_000)


# ----------------------------------------------------------------------------- the stages end to end (synthetic)
# placebo rows (= matched windows 0, 2, 3, 5, 7): part 0 holds windows 0, 2, 3 (matches 0, 1); part 1 windows 5, 7 (3, 4)
PART_TIMES = ([250_000, 270_000, 640_000, 665_000, 700_000, 740_000], [720_000, 748_000, 1_450_000, 1_495_000])
PART_MATCHES = ([0, 0, 1, 1, 1, 1], [3, 3, 4, 4])
PART_WINDOWS = ([(0, 0, 1), (2, 2, 3), (3, 4, 5)], [(5, 0, 1), (7, 2, 3)])
PART_FRAMES = ([247_000, 268_000, 639_000, 660_000, 690_500, 739_000], [716_000, 740_000, 1_445_000, 1_490_000])


def _pipeline_inputs(tmp_path):
    from studies.tog_experiments_20261012 import e1_candidates as ec

    paths = _stage_inputs(tmp_path)
    q, w = _q_rows(), _windows(_q_rows())
    states = paths.states_dir
    parts = {}
    first = 0
    for k in range(2):
        digest = _write_part(states / f"part_{k:04d}.npz", PART_TIMES[k], PART_MATCHES[k], PART_WINDOWS[k], first_id=first,
                             frames=PART_FRAMES[k])
        parts[f"shard_{k:04d}"] = {"file": f"part_{k:04d}.npz", "sha256": digest, "shard": k}
        first += len(PART_TIMES[k])
    w_sha, q_sha = _sha(paths.windows), _sha(paths.q_rows)
    (states / "STATES_h90.json").write_text(json.dumps({"stage": "states", "smoke": False, "parts": parts,
                                                        "windows": {"sha256": w_sha}, "q_rows": {"sha256": q_sha},
                                                        "parts_list_sha256": "synthetic"}))
    pred_record = json.loads(paths.pred_record.read_text())
    pred_record["states"]["record_sha256"] = _sha(states / "STATES_h90.json")
    paths.pred_record.write_text(json.dumps(pred_record))
    record = json.loads(paths.windows_record.read_text())
    record["roles"] = json.loads(json.dumps(es._json_safe({name: ec.role_summary(w, w["role"] == code)
                                                           for name, code in er1.PUBLIC_ROLES.items()})))
    paths.windows_record.write_text(json.dumps(record))
    # q_run_1 inputs, ENGAGEMENTS (with one V_DEV row first), DIAGNOSTICS and OBJECTIVE_COUNTS
    store = paths.q_store
    store.mkdir(parents=True)
    (store / "BUILD.json").write_text(json.dumps({"rows": len(ROLES)}))
    paths.q_inputs.mkdir(parents=True)
    np.save(paths.q_inputs / "XY.npy", np.random.default_rng(9).normal(size=(len(ROLES), 10, 2)))
    (paths.q_inputs / "PREPARED.json").write_text(json.dumps({"rows_sha256": q_sha, "xy_sha256": _sha(paths.q_inputs / "XY.npy"),
                                                              "store_build_sha256": _sha(store / "BUILD.json")}))
    engagements = {"canonical_match_key": np.r_[np.array([_hex("vdev")], dtype="S64"), q["canonical_match_key"]],
                   "component_key": np.r_[np.array([_hex("vdev_c")], dtype="S64"), q["component_key"]],
                   "role": np.r_[np.int8(1), q["role"]].astype(np.int8), "valid_h90": np.ones(len(ROLES) + 1, dtype=bool),
                   "query_pre": np.r_[0, q["query_pre"]], "pre_ms": np.r_[100_000, q["pre_ms"]],
                   "query_h90": np.r_[1, q["query_pre"] + 3], "post_ms_h90": np.r_[130_000, q["pre_ms"] + w["D_ms"]],
                   "F_pre_ms": np.r_[95_000, q["pre_ms"] - 1_000]}
    _write_npz(paths.engagements, engagements)
    counts = np.zeros((160, 9), dtype=np.int32)
    counts[:, 0] = np.arange(160) // 4
    counts[:, 3] = np.arange(160) // 9
    paths.projection.mkdir(parents=True)
    np.save(paths.objective_counts, counts)
    gain = (counts[q["query_pre"] + 3].sum(axis=1) - counts[q["query_pre"]].sum(axis=1))[q["role"] == 3]
    (paths.labels / "DIAGNOSTICS_LAYER2.json").write_text(json.dumps({"objective_events_in_window": {
        "some": {"rows": int((gain > 0).sum())}, "none": {"rows": int((gain <= 0).sum())}}}))
    return paths


def _fake_q_prepare(change=None):
    def runner(folder):
        base = Path(folder) / "inputs" / "main_h90"
        table = _load(Path(folder) / "Q_ROWS_h90.npz")
        rows_sha, xy_sha = _sha(Path(folder) / "Q_ROWS_h90.npz"), _sha(base / "XY.npy")
        build_sha = _sha(Path(folder) / "raw" / "store" / "BUILD.json")
        train = table["role"] == 0
        contexts = {}
        for scope, scope_rows in (("FIT", int((train & ~table["q_stop"]).sum())), ("REFIT_TRAIN", int(train.sum()))):
            directory = base / scope
            (directory / "bundle").mkdir(parents=True)
            np.savez(directory / "CONTEXT.npz", sigma=np.array([1.0]))
            for name in ("MANIFEST.json", "evidence.json", "numeric.json", "vocab.json"):
                (directory / "bundle" / name).write_text(name)
            (directory / "CONTEXT.json").write_text(json.dumps({
                "context_npz_sha256": _sha(directory / "CONTEXT.npz"),
                "bundle_manifest_sha256": _sha(directory / "bundle" / "MANIFEST.json"),
                "rows_sha256": rows_sha, "xy_sha256": xy_sha, "store_build_sha256": build_sha, "scope_rows": scope_rows,
                "sigma": 1.0}))
            contexts[scope] = {"context_json_sha256": _sha(directory / "CONTEXT.json"), "scope_rows": scope_rows, "sigma": 1.0}
        (base / "PREPARED.json").write_text(json.dumps({
            "rows_file": "Q_ROWS_h90.npz", "rows_sha256": rows_sha, "xy_file": "inputs/main_h90/XY.npy", "xy_sha256": xy_sha,
            "store_build_sha256": build_sha, "release_digest": er1.RELEASE_DIGEST, "variant": "main_h90",
            "rows": len(table["role"]), "prepare_version": "q_prepare_v1_20261005", "contexts": contexts}))
        if change is not None:
            change(Path(folder))
        return {"returncode": 0, "stderr_log": "none", "peak_memory_bytes_sampled": 1}
    return runner


def _fake_check(paths):
    (paths.out / "E1_STORE_CHECK.json").write_text(json.dumps({"passed": True, "files": {}, "stage": "store_path_check",
                                                               "started_utc": "x", "utc": "y", "stages": [],
                                                               "peak_memory_bytes": {}}))


def test_the_stages_run_end_to_end_and_hold_no_test_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    paths = _pipeline_inputs(tmp_path)
    assert er1.rows_stage(paths, expected={}) == 0
    _fake_check(paths)                                    # the check stage needs the bank; its pieces are tested above
    assert er1.store_stage(paths, workers=1, expected={}) == 0
    store = qf.RawStore(paths.plc / "raw" / "store")
    assert store.queries.tolist() == [0, 1, 2, 3, 4]
    starts = [250_000, 640_000, 700_000, 720_000, 1_450_000]
    for row, start in enumerate(starts):                  # each placebo row holds its start state's semantic row
        np.testing.assert_array_equal(store.numeric["global"][row], _semantic_row(start)["numeric"]["global"])
    xy = np.load(paths.plc / "inputs" / "main_h90" / "XY.npy")
    part0, part1 = _load(paths.states_dir / "part_0000.npz"), _load(paths.states_dir / "part_0001.npz")
    np.testing.assert_array_equal(xy, np.concatenate([part0["xy"][[0, 2, 4]], part1["xy"][[0, 2]]]))
    build = json.loads((paths.plc / "raw" / "store" / "BUILD.json").read_text())
    assert build["keys"].startswith("placebo row index") and "code_sha256" in build
    assert er1.eng_inputs_stage(paths) == 0
    assert os.path.isjunction(paths.eng / "raw" / "store")
    np.testing.assert_array_equal(np.load(paths.eng / "inputs" / "main_h90" / "XY.npy"),
                                  np.load(paths.q_inputs / "XY.npy")[list(MATCHED)])
    assert er1.prepare_stage(paths, runner=_fake_q_prepare()) == 0
    assert er1.balance_stage(paths, expected={}) == 0
    assert er1.record_stage(paths) == 0
    balance = json.loads((paths.out / "BALANCE.json").read_text(encoding="utf-8"))
    assert sorted(balance["roles"]) == ["Q_CAL", "Q_DEV", "TRAIN"] and balance["cross_checks"]["passed"] is True
    assert balance["roles"]["Q_DEV"]["objective_share"]["engagement"]["share"] == 1.0
    assert balance["files"]["BALANCE_TEST_SEALED.json"]["sha256"] == _sha(paths.out / "BALANCE_TEST_SEALED.json")
    for name in ("E1_ROWS.json", "E1_STORE.json", "E1_ENG_INPUTS.json", "E1_PREPARE.json", "BALANCE.json", "E1_ROWS_RUN.json"):
        record = json.loads((paths.out / name).read_text(encoding="utf-8"))
        assert record["passed"] is True and "TEST" not in _keys_of(record), name
    prepare = json.loads((paths.out / "E1_PREPARE.json").read_text(encoding="utf-8"))
    assert prepare["q_run_1_untouched"]["changed"] is False and "rows" not in prepare["prepared"]["plc"]
    run = json.loads((paths.out / "E1_ROWS_RUN.json").read_text(encoding="utf-8"))
    assert run["counts"]["plc"]["Q_DEV"] == {"rows": 2, "matches": 1}
    for stage in (er1.rows_stage, er1.balance_stage):     # finished stages are verified and kept
        assert stage(paths, expected={}) == 0


def test_prepare_is_blocked_when_q_run_1_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    paths = _pipeline_inputs(tmp_path)
    assert er1.rows_stage(paths, expected={}) == 0
    _fake_check(paths)
    assert er1.store_stage(paths, workers=1, expected={}) == 0
    assert er1.eng_inputs_stage(paths) == 0

    def touch(folder):
        if folder.name == "eng":
            (paths.q_store / "BUILD.json").write_text("changed")
    with pytest.raises(er1.Blocked, match="q_run_1 changed"):
        er1.prepare_stage(paths, runner=_fake_q_prepare(touch))
    assert not (paths.out / "E1_PREPARE.json").exists()


def test_store_stage_is_blocked_by_a_nonfinite_xy(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    paths = _pipeline_inputs(tmp_path)
    assert er1.rows_stage(paths, expected={}) == 0
    _fake_check(paths)
    arrays = _load(paths.states_dir / "part_0001.npz")
    arrays["xy"][2, 3, 0] = np.nan                         # the start state of window 7 (TRAIN)
    (paths.states_dir / "part_0001.npz").unlink()
    digest = es.write_bytes_once(paths.states_dir / "part_0001.npz", es.npz_bytes(arrays, compress=True))
    states = json.loads(paths.states_record.read_text())
    states["parts"]["shard_0001"]["sha256"] = digest
    paths.states_record.write_text(json.dumps(states))
    pred_record = json.loads(paths.pred_record.read_text())
    pred_record["states"]["record_sha256"] = _sha(paths.states_record)
    paths.pred_record.write_text(json.dumps(pred_record))
    with pytest.raises(er1.Blocked, match="XY is not finite"):
        er1.store_stage(paths, workers=1, expected={})
    assert not (paths.plc / "inputs" / "main_h90" / "XY.npy").exists()


def _edit_json(path, change):
    record = json.loads(Path(path).read_text())
    change(record)
    Path(path).write_text(json.dumps(record))


def test_rows_stage_is_blocked_when_the_placebo_rows_differ_from_the_window_record(tmp_path):
    paths = _pipeline_inputs(tmp_path)
    _edit_json(paths.windows_record, lambda r: r["roles"]["TRAIN"].update(matched=3))
    with pytest.raises(er1.Blocked, match="WINDOWS_h90.json matched windows"):
        er1.rows_stage(paths, expected={})


@pytest.mark.parametrize("which, reason", [("windows", "bins differs from WINDOWS_h90.json"),
                                           ("diagnostics", "DIAGNOSTICS_LAYER2.json")])
def test_balance_stage_is_blocked_by_a_failed_cross_check(tmp_path, monkeypatch, which, reason):
    paths = _pipeline_inputs(tmp_path)
    assert er1.rows_stage(paths, expected={}) == 0
    if which == "windows":
        _edit_json(paths.windows_record, lambda r: r["roles"]["Q_CAL"]["bins"]["time_bin"]["1"].update(matched=1))
    else:
        _edit_json(paths.labels / "DIAGNOSTICS_LAYER2.json",
                   lambda r: r["objective_events_in_window"]["some"].update(rows=r["objective_events_in_window"]["some"]["rows"] + 1))
    with pytest.raises(er1.Blocked, match=reason):
        er1.balance_stage(paths, expected={})
    assert not (paths.out / "BALANCE.json").exists() and not (paths.out / "BALANCE_TEST_SEALED.json").exists()


def test_merge_store_keeps_its_own_build_record_and_refuses_another(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "semantic_rows", _fake_semantic_rows)
    part = tmp_path / "part_0000.npz"
    digest = _write_part(part, [130_000, 150_000, 170_000], [7, 7, 7], [(0, 0, 1)])
    tasks = []
    for k, (local, keys) in enumerate(((np.array([0, 1]), np.array([0, 1])), (np.array([2]), np.array([2])))):
        tasks.append({"chunk_dir": str(tmp_path / "chunks" / f"chunk_{k:04d}"), "identity": {"k": k},
                      "items": [{"path": str(part), "sha256": digest, "local": local, "keys": keys,
                                 "match": np.full(len(keys), 7), "ms": np.array([130_000, 150_000, 170_000])[local]}]})
        er1.store_chunk(tasks[-1])
    dirs = [task["chunk_dir"] for task in tasks]
    first = er1.merge_store(dirs, tmp_path / "store", {"keys": "test"})
    assert first["status"] == "written" and qf.RawStore(tmp_path / "store").queries.tolist() == [0, 1, 2]
    again = er1.merge_store(dirs, tmp_path / "store", {"keys": "test"})
    assert again["status"] == "kept" and again["entry"]["sha256"] == first["entry"]["sha256"]
    with pytest.raises(er1.Blocked, match="other chunks"):
        er1.merge_store(dirs[:1], tmp_path / "store", {"keys": "test"})
