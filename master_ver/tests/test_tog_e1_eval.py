"""Task 16 (T1.5) part A: E1's gains and ratio, the joint bootstrap, decide(), the sensitivities, the evaluation path that
the Q_CAL rehearsal and the TEST step share, the train driver, the lock's sealed-summary list and the TEST-step order
(studies/tog_experiments_20261012/e1_run.py); synthetic data only.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E1_KO.md sections 3-7 and 10 (amended 2026-10-13,
re-approved 7a841db); SDD Rulings 16 (decide on L1, L2 and the net gain), 27/28, 31, 52, 55, 56 and 58.  No test opens a
real sealed file or a real TEST label; the frozen job plan and release are read from the repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from studies.q_production_20261005 import jobs as qjobs, qeval, qtest, rows as q_rows_module
from studies.tog_experiments_20261012 import e1_candidates as ec, e1_rows as er1, e1_run as e1, e1_states as es, e4_q as eq
from studies.tog_experiments_20261012.e1_run import decide, gains_and_ratio


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_npz(path, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)
    return _sha(path)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return _sha(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# ======================================================================= gains, ratio (the brief's test, kept as given)
def test_ratio_and_net_gain():
    out = gains_and_ratio(np.array([0.20, 0.22]), np.array([0.23, 0.25]),
                          np.array([0.24, 0.24]), np.array([0.25, 0.25]))
    assert np.isclose(out["G_eng"], 0.03) and np.isclose(out["G_plc"], 0.01)
    assert np.isclose(out["r"], 1 / 3) and np.isclose(out["net"], 0.02)


def test_gains_and_ratio_give_the_linear_contrasts_and_refuse_bad_arrays():
    out = gains_and_ratio(np.array([0.20, 0.22]), np.array([0.23, 0.25]), np.array([0.24, 0.24]), np.array([0.25, 0.25]))
    assert np.isclose(out["L1"], 0.01 - 0.25 * 0.03) and np.isclose(out["L2"], 0.01 - 0.75 * 0.03)
    flat = gains_and_ratio(np.array([0.2]), np.array([0.2]), np.array([0.2]), np.array([0.3]))
    assert flat["G_eng"] == 0.0 and flat["r"] is None and np.isclose(flat["L1"], 0.1)        # r undefined, L defined
    with pytest.raises(ValueError):
        gains_and_ratio(np.array([0.2]), np.array([0.2, 0.3]), np.array([0.2]), np.array([0.3]))
    with pytest.raises(ValueError):
        gains_and_ratio(np.array([np.nan]), np.array([0.2]), np.array([0.2]), np.array([0.3]))
    with pytest.raises(ValueError):
        gains_and_ratio(np.array([]), np.array([]), np.array([]), np.array([]))


# ======================================================================= decide() (Ruling 16, PRESPEC_E1 section 5)
KEEP_L = {"L1_ci": (-0.004, -0.001), "L2_ci": (-0.010, -0.006), "net_ci": (0.008, 0.012)}


def test_decide_returns_every_section_5_label():
    assert decide({"G_eng_ci": (-0.004, -0.001), **KEEP_L}) == "no_gain"
    assert decide({"G_eng_ci": (-0.001, 0.004), **KEEP_L}) == "inconclusive"
    assert decide({"G_eng_ci": (0.0, 0.004), **KEEP_L}) == "inconclusive"           # lower bound 0 is not > 0
    assert decide({"G_eng_ci": (-0.002, 0.0), **KEEP_L}) == "inconclusive"          # upper bound 0 is not < 0
    assert decide({"G_eng_ci": (0.010, 0.014), **KEEP_L}) == "keep"
    assert decide({"G_eng_ci": (0.010, 0.014), "L1_ci": (0.001, 0.004), "L2_ci": (0.0005, 0.002),
                   "net_ci": (-0.010, -0.002)}) == "rewrite"
    assert decide({"G_eng_ci": (0.010, 0.014), "L1_ci": (-0.002, 0.001), "L2_ci": (-0.006, -0.001),
                   "net_ci": (0.001, 0.006)}) == "net_gain"
    assert decide({"G_eng_ci": (0.010, 0.014), "L1_ci": (-0.002, 0.001), "L2_ci": (-0.006, -0.001),
                   "net_ci": (-0.001, 0.006)}) == "indistinguishable"
    assert decide({"G_eng_ci": (0.010, 0.014), "L1_ci": (-0.002, 0.0), "L2_ci": (-0.006, 0.0),
                   "net_ci": (0.0, 0.006)}) == "indistinguishable"                   # every bound exactly on 0
    assert set(e1.OUTCOMES) == {"no_gain", "inconclusive", "keep", "rewrite", "net_gain", "indistinguishable"}


def test_decide_follows_the_section_5_order():
    # contrived intervals where several conditions hold: the earlier rule wins
    assert decide({"G_eng_ci": (0.01, 0.02), "L1_ci": (-0.02, -0.01), "L2_ci": (0.001, 0.002), "net_ci": (0.001, 0.002)}) == "keep"
    assert decide({"G_eng_ci": (0.01, 0.02), "L1_ci": (0.0, 0.01), "L2_ci": (0.001, 0.002), "net_ci": (0.001, 0.002)}) == "rewrite"
    assert decide({"G_eng_ci": (-0.02, -0.01), "L1_ci": (-0.02, -0.01), "L2_ci": (0.001, 0.002), "net_ci": (0.001, 0.002)}) == "no_gain"


def test_decide_refuses_a_non_interval_and_decision_gives_the_text_and_net_note():
    with pytest.raises(ValueError):
        decide({"G_eng_ci": (0.02, 0.01), **KEEP_L})
    with pytest.raises(ValueError):
        decide({"G_eng_ci": (np.nan, 0.01), **KEEP_L})
    with pytest.raises(KeyError):
        decide({"G_eng_ci": (0.01, 0.02)})
    out = {"G_eng_ci": (0.010, 0.014), "L1_ci": (-0.002, 0.001), "L2_ci": (-0.006, -0.001), "net_ci": (-0.004, -0.001)}
    d = e1.decision(out)
    assert d["outcome"] == "indistinguishable" and d["placebo_gain_exceeds_engagement_gain"] is True
    assert "측정기 표류와 구별되지 않는다" in d["text"] and d["default_handling"] is not None
    keep = e1.decision({"G_eng_ci": (0.010, 0.014), **KEEP_L})
    assert keep["outcome"] == "keep" and keep["placebo_gain_exceeds_engagement_gain"] is False and keep["default_handling"] is None


def _intervals(draws):
    return {f"{name}_ci": e1.percentile_interval(draws[name]) for name in ("G_eng", "G_plc", "L1", "L2", "net", "r")}


def test_decide_follows_L1_when_exactly_50_draws_have_L1_at_or_above_0():
    # 2,000 draws, G_eng > 0 in every one, exactly 50 with L1 >= 0 (r >= 0.25).  numpy's linear interpolation puts the
    # 97.5th percentile of L1 below 0 while that of r is above 0.25: PRESPEC_E1 section 4 decides by L ("keep").
    l1 = np.r_[np.linspace(-0.02, -1e-6, 1950), np.linspace(1e-5, 2e-5, 50)]
    g_eng = np.r_[np.ones(1950), np.linspace(1e-5, 2e-5, 50) / 0.75]
    draws = e1.derived_draws(g_eng, l1 + 0.25 * g_eng)
    ci = _intervals(draws)
    assert int((draws["L1"] >= 0).sum()) == 50 and bool((draws["G_eng"] > 0).all())
    assert ci["L1_ci"][1] < 0.0 and ci["r_ci"][1] >= 0.25                     # the two readings part
    assert decide(ci) == "keep"


def test_decide_follows_L2_when_exactly_50_draws_have_L2_at_or_below_0():
    l2 = np.r_[np.linspace(-2e-5, -1e-5, 50), np.linspace(1e-6, 0.02, 1950)]
    g_eng = np.r_[np.linspace(2e-5, 1e-5, 50), np.ones(1950)]                   # r = -0.25 on the 50 draws
    draws = e1.derived_draws(g_eng, l2 + 0.75 * g_eng)
    ci = _intervals(draws)
    assert int((draws["L2"] <= 0).sum()) == 50 and bool((draws["G_eng"] > 0).all())
    assert ci["L2_ci"][0] > 0.0 and ci["r_ci"][0] <= 0.75                      # the two readings part
    assert ci["L1_ci"][1] >= 0.0 and decide(ci) == "rewrite"


# ======================================================================= the joint bootstrap (PRESPEC_E1 section 4)
def _losses(seed=3, m=60, shift=0.01):
    rng = np.random.default_rng(seed)
    eng_q = rng.uniform(0.15, 0.25, m)
    eng_pt = eng_q + rng.normal(shift, 0.01, m)
    plc_q = rng.uniform(0.20, 0.26, m)
    plc_pt = plc_q + rng.normal(0.003, 0.01, m)
    return eng_q, eng_pt, plc_q, plc_pt


def test_joint_bootstrap_is_reproducible_and_derives_every_quantity_from_one_set_of_draws():
    a = e1.joint_bootstrap(*_losses(), return_draws=True)
    b = e1.joint_bootstrap(*_losses(), return_draws=True)
    assert all(np.array_equal(a["draws"][k], b["draws"][k], equal_nan=True) for k in a["draws"])
    eng_q, eng_pt, plc_q, plc_pt = _losses()
    g_eng, g_plc = [], []
    for idx in qeval.bootstrap_indices(60, 2000, 20261011):                    # match-level draws, PCG64 seed 20261011
        g_eng.append(eng_pt[idx].mean() - eng_q[idx].mean())
        g_plc.append(plc_pt[idx].mean() - plc_q[idx].mean())
    np.testing.assert_allclose(a["draws"]["G_eng"], g_eng, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["draws"]["G_plc"], g_plc, rtol=0, atol=1e-12)
    d = a["draws"]
    assert np.array_equal(d["L1"], d["G_plc"] - 0.25 * d["G_eng"]) and np.array_equal(d["L2"], d["G_plc"] - 0.75 * d["G_eng"])
    assert np.array_equal(d["net"], d["G_eng"] - d["G_plc"]) and np.array_equal(d["r"], d["G_plc"] / d["G_eng"])
    for name in ("G_eng", "G_plc", "L1", "L2", "net", "r"):
        assert a[f"{name}_ci"] == [float(v) for v in np.quantile(d[name], (0.025, 0.975), method="linear")]
    point = gains_and_ratio(eng_q, eng_pt, plc_q, plc_pt)
    assert all(np.isclose(a["point"][name], point[name], rtol=0, atol=1e-15) for name in ("G_eng", "G_plc", "L1", "L2", "net", "r"))
    assert (a["resamples"], a["seed"], a["level"], a["matches"]) == (2000, 20261011, 0.95, 60)
    assert a["draws_G_eng_le_0"] == int((d["G_eng"] <= 0).sum())
    assert "draws" not in e1.joint_bootstrap(*_losses())


def test_joint_bootstrap_counts_nonpositive_engagement_draws_and_never_drops_them():
    out = e1.joint_bootstrap(*_losses(seed=8, m=40, shift=0.0005), return_draws=True)
    g = out["draws"]["G_eng"]
    k = int((g <= 0).sum())
    assert 0 < k < 2000 and out["draws_G_eng_le_0"] == k and len(out["draws"]["r"]) == 2000
    assert out["equivalence"]["every_draw_G_eng_positive"] is False
    assert out["r_ci"] == [float(v) for v in np.quantile(out["draws"]["r"], (0.025, 0.975))]   # every draw, none dropped


# ======================================================================= the four arrays (one match order)
def _arms(n=10, seed=4):
    rng = np.random.default_rng(seed)
    match = rng.integers(0, 4, n)
    y_eng, y_plc = rng.integers(0, 2, n), rng.integers(0, 2, n)
    p_eng = {"lightgbm": rng.uniform(0.05, 0.95, n), "PT_flex": rng.uniform(0.05, 0.95, n)}
    p_plc = {"lightgbm": rng.uniform(0.05, 0.95, n), "PT_flex": rng.uniform(0.05, 0.95, n)}
    return match, y_eng, p_eng, y_plc, p_plc


def test_four_arrays_are_per_match_mean_losses_in_ascending_match_order():
    match, y_eng, p_eng, y_plc, p_plc = _arms()
    ids, arrays, info = e1.four_arrays(match, y_eng, p_eng, y_plc, p_plc)
    assert ids.tolist() == sorted(set(match.tolist())) and list(arrays) == ["eng_q", "eng_pt", "plc_q", "plc_pt"]
    for position, m in enumerate(ids):
        rows = match == m
        assert arrays["eng_q"][position] == pytest.approx(np.mean((p_eng["lightgbm"][rows] - y_eng[rows]) ** 2))
        assert arrays["plc_pt"][position] == pytest.approx(np.mean((p_plc["PT_flex"][rows] - y_plc[rows]) ** 2))
    assert np.isclose(arrays["eng_q"].mean(), qeval.brier(y_eng, p_eng["lightgbm"], match_ids=match))   # match-weighted Brier
    perm = np.random.default_rng(1).permutation(match.size)                       # the pair order does not matter
    ids2, arrays2, _ = e1.four_arrays(match[perm], y_eng[perm], {k: v[perm] for k, v in p_eng.items()}, y_plc[perm],
                                      {k: v[perm] for k, v in p_plc.items()})
    assert np.array_equal(ids, ids2) and all(np.allclose(arrays[k], arrays2[k]) for k in arrays)
    assert info["rows_eng"] == info["rows_plc"] == match.size and info["matches"] == ids.size


def test_four_arrays_refuse_different_match_sets_unless_asked_to_intersect():
    match = np.array([1, 1, 2, 3, 3])
    y = np.array([0, 1, 1, 0, 1])
    p = {"lightgbm": np.full(5, 0.4), "PT_flex": np.full(5, 0.6)}
    eng_mask = np.array([True, True, False, True, True])                          # match 2 loses its engagement row
    with pytest.raises(ValueError):
        e1.four_arrays(match, y, p, y, p, eng_mask=eng_mask)
    ids, arrays, info = e1.four_arrays(match, y, p, y, p, eng_mask=eng_mask, intersect=True)
    assert ids.tolist() == [1, 3] and info["matches_dropped"] == 1 and info["rows_plc"] == 4
    assert all(arrays[k].shape == (2,) for k in arrays)


# ======================================================================= sensitivities (PRESPEC_E1 sections 3 균형 and 4)
def test_r001_masks_keep_rows_at_or_above_0_01_in_each_arm():
    svi = np.array([0.02, -0.0099, 0.01, -0.03, 0.005, 0.2])
    w = np.array([-0.01, 0.02, 0.0, 0.015, -0.5, 0.001])
    eng_mask, plc_mask = e1.r001_masks(svi, w)
    assert eng_mask.tolist() == [True, False, True, True, False, True]
    assert plc_mask.tolist() == [True, True, False, True, True, False]
    assert e1.R001 == er1.NEAR_ZERO[1] == 0.01                                    # the middle near-zero threshold


def test_objective_free_mask_needs_no_objective_in_either_window():
    obj_eng = np.array([True, False, False, True])
    obj_plc = np.array([False, False, True, True])
    assert e1.objective_free_mask(obj_eng, obj_plc).tolist() == [False, True, False, False]


def test_quintile_strata_put_a_value_on_an_edge_in_the_upper_stratum():
    edges, strata = e1.quintile_strata(np.array([0.6, 0.1, 0.5, 0.2, 0.4, 0.3]))   # edges fall on data values here
    assert edges.tolist() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert strata.tolist() == [4, 0, 4, 1, 3, 2]                                    # searchsorted side='right'


def test_stratified_r_uses_engagement_v_quintiles_their_shares_and_match_weights_within_strata():
    rng = np.random.default_rng(9)
    n = 200
    match = rng.integers(0, 40, n)
    v = np.r_[np.full(80, 0.2), rng.uniform(0.25, 0.9, n - 80)]                   # ties: unequal strata, stratum 0 empty
    y_eng, y_plc = rng.integers(0, 2, n), rng.integers(0, 2, n)
    p_eng = {"lightgbm": rng.uniform(0.1, 0.9, n), "PT_flex": rng.uniform(0.1, 0.9, n)}
    p_plc = {"lightgbm": rng.uniform(0.1, 0.9, n), "PT_flex": rng.uniform(0.1, 0.9, n)}
    out = e1.stratified_r(match, v, y_eng, p_eng, y_plc, p_plc)
    edges = np.quantile(v, (0.2, 0.4, 0.6, 0.8), method="linear")
    strata = np.searchsorted(edges, v, side="right")
    num = den = 0.0
    assert out["strata"]["0"]["rows"] == 0 and out["strata"]["0"]["G_eng"] is None
    for s in range(5):
        rows = strata == s
        if not rows.any():
            continue

        def brier(y, p):                                   # mean over the stratum's matches of their mean loss there
            return np.mean([np.mean((p[rows & (match == m)] - y[rows & (match == m)]) ** 2) for m in np.unique(match[rows])])

        g_eng = brier(y_eng, p_eng["PT_flex"]) - brier(y_eng, p_eng["lightgbm"])
        g_plc = brier(y_plc, p_plc["PT_flex"]) - brier(y_plc, p_plc["lightgbm"])
        share = rows.sum() / n
        num, den = num + share * g_plc, den + share * g_eng
        assert out["strata"][str(s)]["rows"] == int(rows.sum()) and np.isclose(out["strata"][str(s)]["G_eng"], g_eng)
        assert np.isclose(out["strata"][str(s)]["weight"], share)
    assert out["edges"] == [float(e) for e in edges] and np.isclose(out["r"], num / den, rtol=1e-12, atol=0)


def test_deduplication_keeps_the_pair_with_the_smallest_selection_hash_per_window():
    match = np.array([1, 1, 1, 2, 2, 2])
    s = np.array([100_000, 100_000, 130_000, 100_000, 100_000, 100_000])
    keys = np.array([b"m1:a", b"m1:b", b"m1:c", b"m2:a", b"m2:b", b"m2:c"], dtype="S129")
    keep = e1.dedup_keep(match, s, keys)
    digest = [ec.selection_digest(k, int(t)) for k, t in zip(keys, s)]
    first = 0 if digest[0] < digest[1] else 1
    second = 3 + int(np.argmin(digest[3:]))
    expected = np.zeros(6, dtype=bool)
    expected[[first, 2, second]] = True
    assert keep.tolist() == expected.tolist()


def test_gain_is_brier_pt_minus_brier_lightgbm_on_shared_draws():
    rng = np.random.default_rng(2)
    n = 120
    match = rng.integers(0, 30, n)
    y = rng.integers(0, 2, n)
    p_q = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.1, n), 0.02, 0.98)
    p_pt = np.clip(0.5 + 0.1 * (2 * y - 1) + rng.normal(0, 0.1, n), 0.02, 0.98)
    g = e1.gain(y, p_q, p_pt, match)
    assert np.isclose(g["point"], qeval.brier(y, p_pt, match_ids=match) - qeval.brier(y, p_q, match_ids=match))
    from studies.tog_experiments_20261012 import e2_run
    e2 = e2_run.evaluate_pair(y, p_q, p_pt, match)["brier_difference"]            # LightGBM - PT_flex (E2's sign)
    assert np.isclose(g["point"], -e2["point"]) and np.allclose(g["interval"], [-e2["interval"][1], -e2["interval"][0]])
    assert e1.gain(y[:0], p_q[:0], p_pt[:0], match[:0])["point"] is None          # an empty subset is reported, not raised


def test_evaluate_pairs_reports_the_primary_decision_sensitivities_and_auxiliaries():
    pairs = _pairs()
    out = e1.evaluate_pairs(pairs)
    ids, arrays, _ = e1.four_arrays(pairs["match"], pairs["eng_y"], pairs["eng_p"], pairs["plc_y"], pairs["plc_p"])
    assert out["primary"] == e1.joint_bootstrap(arrays["eng_q"], arrays["eng_pt"], arrays["plc_q"], arrays["plc_pt"])
    assert out["decision"]["outcome"] == decide(out["primary"])
    assert set(out["sensitivities"]) == {"r_0.01", "objective_free", "stratified_r", "deduplicated_r"}
    assert out["sensitivities"]["stratified_r"]["computed_regardless_of_trigger"] is True
    assert out["sensitivities"]["deduplicated_r"]["trigger"] == pairs["triggers"]["deduplicated_r"]
    assert out["sensitivities"]["objective_free"]["pairs"] == int((~pairs["obj_eng"] & ~pairs["obj_plc"]).sum())
    aux = out["auxiliary"]
    assert set(aux) == {"auc", "G_eng_full", "G_plc_over_main_test_gain", "main_matched_vs_unmatched"}
    assert np.isclose(aux["G_plc_over_main_test_gain"]["point"], out["primary"]["point"]["G_plc"] / 0.01386)
    main = pairs["main"]
    full = e1.gain(main["y"][main["matched"]], main["p"]["lightgbm"][main["matched"]], main["p"]["PT_flex"][main["matched"]],
                   main["match"][main["matched"]])
    assert aux["G_eng_full"]["point"] == full["point"]
    assert aux["auc"]["eng"]["auc"]["lightgbm"] is not None and "P1" not in aux["auc"]["eng"]["auc"]
    assert np.isclose(out["cells"]["eng"]["lightgbm"], arrays["eng_q"].mean())


def _pairs(n_match=30, seed=12):
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(n_match), 2)
    n = match.size
    y_eng, y_plc = rng.integers(0, 2, n), rng.integers(0, 2, n)

    def p(y, strength):
        return np.clip(0.5 + strength * (2 * y - 1) + rng.normal(0, 0.1, y.size), 0.02, 0.98)

    s = 100_000 + 1_000 * np.arange(n)
    s[1] = s[0]                                                                    # one reused window
    main_match = np.repeat(np.arange(n_match), 3)
    main_y = rng.integers(0, 2, main_match.size)
    matched = np.tile([True, True, False], n_match)
    return {"match": match, "eng_y": y_eng, "eng_svi": rng.normal(0, 0.05, n), "eng_p": {"lightgbm": p(y_eng, 0.3), "PT_flex": p(y_eng, 0.1)},
            "plc_y": y_plc, "plc_w": rng.normal(0, 0.05, n), "plc_p": {"lightgbm": p(y_plc, 0.15), "PT_flex": p(y_plc, 0.1)},
            "v_tau": rng.uniform(0.2, 0.8, n), "eng_key": np.array([f"m{m:03d}:c{i % 2}".encode() for i, m in enumerate(match)], dtype="S129"),
            "s_ms": s, "obj_eng": rng.random(n) < 0.5, "obj_plc": rng.random(n) < 0.4,
            "triggers": {"stratified_r": False, "deduplicated_r": True},
            "main": {"y": main_y, "p": {"lightgbm": p(main_y, 0.3), "PT_flex": p(main_y, 0.1)}, "match": main_match, "matched": matched}}


# ======================================================================= the lock's sealed summaries (Ruling 58)
SEALED_SUMMARY_FILES = ("plc/raw/chunks/chunk_0000/CHUNK.json", "plc/raw/chunks/chunk_0013/CHUNK.json", "plc/raw/store/CHUNK.json",
                        "plc/raw/store/BUILD.json", "plc/inputs/main_h90/PREPARED.json", "eng/inputs/main_h90/PREPARED.json",
                        "plc/logs/q_prepare_20261013T070325Z.stdout.log", "eng/logs/q_prepare_20261013T070340Z.stdout.log",
                        "plc/Q_ROWS_h90.json", "eng/Q_ROWS_h90.json")
DECOYS = ("STORE_PATH_CHECK_h90/store/BUILD.json", "STORE_PATH_CHECK_h90/chunks/chunk_0000/CHUNK.json", "gate_tog/raw_store/CHUNK.json",
          "plc/logs/q_prepare_20261013T070325Z.stderr.log", "plc/raw/store/queries.i64.npy", "plc/jobs/x/RESULT.json",
          "plc/inputs/main_h90/FIT/CONTEXT.json", "WINDOWS_h90.json", "e1_rows_prepare.stdout.log", "plc/raw/chunks/chunk_0000/queries.i64.npy")


def test_sealed_summaries_are_found_by_path_pattern_and_hashed_without_being_read(tmp_path):
    for k, name in enumerate(SEALED_SUMMARY_FILES + DECOYS):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00not json " + bytes([k]))                         # unparsable: a reader would fail
    found = e1.sealed_summaries(tmp_path)
    assert sorted(found) == sorted(SEALED_SUMMARY_FILES)
    for name, entry in found.items():
        assert entry["sha256"] == _sha(tmp_path / name) and entry["pattern"] and entry["why"]
    missing = e1.sealed_summary_gaps(tmp_path)
    assert missing == []
    (tmp_path / "eng/inputs/main_h90/PREPARED.json").unlink()
    assert e1.sealed_summary_gaps(tmp_path) == ["eng/inputs/main_h90/PREPARED.json"]


# ======================================================================= cards and the train driver
def test_cards_are_prespec_e1_verbatim_and_the_frozen_plan_jobs():
    text = e1.PRESPEC.read_text(encoding="utf-8")
    model_line = next(line for line in text.splitlines() if line.startswith("- (가) 위약: LightGBM 카드"))
    for family, cards in e1.CARDS.items():
        assert all(card in model_line for card in cards)
    assert e1.CARDS == dict(eq.CARDS)
    jobs = eq.plan_jobs(qjobs.plan())
    assert [job.card_id for job in jobs] == [*e1.CARDS["lightgbm"], *e1.CARDS["PT_flex"]]


def test_job_order_queues_lightgbm_of_both_runs_first_and_launches_one_at_a_time():
    jobs = eq.plan_jobs(qjobs.plan())
    order = e1.job_order(jobs)
    assert len(order) == 32 and [family for _, family, _ in order[:8]] == ["lightgbm"] * 8
    assert {arm for arm, _, _ in order} == set(e1.ARMS)
    pending = [(e1.entry_name(arm, qtest.key_of(job.job_id)), family) for arm, family, job in order]
    chosen = eq.next_launches(pending, [], e1.MAX_WORKERS)
    families = [dict(pending)[name] for name in chosen]
    assert len(chosen) == 4 and families.count("lightgbm") == 1
    assert eq.next_launches(pending, ["lightgbm", "PT_flex"], 4) == [name for name, family in pending if family == "PT_flex"][:2]


FAKE_WORKER = r"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text(encoding="utf-8"))
out = Path(spec["out_dir"])
mode = spec.get("fake_mode", "ok")
attempts = len(list(out.glob("RESULT_attempt*.json")))
time.sleep(float(spec.get("fake_sleep", 0.2)))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if mode == "fail_once" and attempts == 0:
    result = {"status": "failed", "reason": "fake", "terminal": False}
elif mode == "terminal":
    result = {"status": "failed", "reason": "fake", "terminal": True}
else:
    with np.load(spec["inputs"]["rows"], allow_pickle=False) as data:
        role, keys = data["role"], data["engagement_key"]
    dev, cal = np.flatnonzero(role == 3), np.flatnonzero(role == 4)
    for name, idx in (("pred_q_dev.npz", dev), ("pred_q_cal.npz", cal)):
        with (out / name).open("wb") as stream:
            np.savez(stream, row=idx, logit=np.zeros(idx.size), p=np.full(idx.size, 0.5))
    (out / "model_refit.pkl").write_bytes(b"model")
    result = {"status": "completed", "model_sha256": sha(out / "model_refit.pkl"),
              "q_dev_keys_sha256": hashlib.sha256(b"\n".join(bytes(k) for k in keys[dev])).hexdigest(),
              "scores": {"q_dev": {"rows": int(dev.size), "raw_match_brier": 0.25, "pred_sha256": sha(out / "pred_q_dev.npz")},
                         "q_cal": {"rows": int(cal.size), "raw_match_brier": 0.25, "pred_sha256": sha(out / "pred_q_cal.npz")}}}
result.update(job_id=spec["job"]["job_id"], identity=spec["job"]["identity"], variant=spec["variant"],
              release_digest=spec["release_digest"], spec_sha256=sha(spec_path), seconds=0.2, finished_utc="t", threads={})
(out / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
"""


def _fake_launcher(calls):
    def launch(spec_path, directory, launch_number, env):
        calls.append(Path(directory).name + "@" + Path(directory).parent.parent.name)
        with (directory / f"worker_{launch_number}.stdout.log").open("ab") as stdout, \
                (directory / f"worker_{launch_number}.stderr.log").open("ab") as stderr:
            return subprocess.Popen([sys.executable, "-c", FAKE_WORKER, str(spec_path)], stdout=stdout, stderr=stderr)
    return launch


def _driver_runs(root, modes=None, sleeps=None):
    """Two run folders with a 6-row table (Q_DEV rows 1-2, Q_CAL rows 3-4) and 2 LightGBM + 3 PT_flex jobs each."""

    jobs = eq.plan_jobs(qjobs.plan())
    chosen = [job for job in jobs if job.family == "lightgbm"][:2] + [job for job in jobs if job.family == "PT_flex"][:3]
    entries = []
    for arm in e1.ARMS:
        run = root / arm
        rows = run / "Q_ROWS_h90.npz"
        _save_npz(rows, role=np.array([0, 3, 3, 4, 4, 5], dtype=np.int8),
                  engagement_key=np.array([f"{arm}{i}".encode() for i in range(6)], dtype="S129"))
        for job in chosen:
            key = qtest.key_of(job.job_id)
            directory = run / "jobs" / key
            directory.mkdir(parents=True)
            spec = {"job": job.to_dict(), "variant": "main_h90", "inputs": {"rows": str(rows)}, "out_dir": str(directory),
                    "release": "r", "release_digest": "d" * 64, "fake_sleep": (sleeps or {}).get((arm, job.card_id), 0.3),
                    "fake_mode": (modes or {}).get((arm, job.card_id), "ok")}
            _write_json(directory / "SPEC.json", spec)
            entries.append(e1.Entry(arm=arm, key=key, family=job.family, jobs_dir=run / "jobs", spec=directory / "SPEC.json", job=job))
    return entries


def test_drive_runs_every_job_with_at_most_four_workers_and_one_lightgbm_and_retries_once(tmp_path):
    entries = _driver_runs(tmp_path, modes={("plc", "ptflex_k4_C0.01"): "fail_once"})
    calls = []
    outcome = e1.drive(entries, workers=4, poll_s=0.05, launcher=_fake_launcher(calls), env={})
    assert outcome["blocked"] == {}
    for event in outcome["events"]:
        if event["event"] == "started":
            running = event["running"]
            assert len(running) <= 4 and sum(family == "lightgbm" for family in running.values()) <= 1
    assert all(eq.job_state(entry.jobs_dir / entry.key) == "completed" for entry in entries)
    retried = tmp_path / "plc" / "jobs" / qtest.key_of(next(e.job.job_id for e in entries if e.job.card_id == "ptflex_k4_C0.01"))
    assert (retried / "RESULT_attempt1.json").exists() and len(calls) == len(entries) + 1
    keys = {arm: hashlib.sha256(b"\n".join(f"{arm}{i}".encode() for i in (1, 2))).hexdigest() for arm in e1.ARMS}
    records, problems = e1.completed_records(entries, "d" * 64, {"Q_DEV": 2, "Q_CAL": 2}, keys)
    assert problems == [] and len(records) == len(entries)
    _, wrong = e1.completed_records(entries, "d" * 64, {"Q_DEV": 2, "Q_CAL": 2}, {**keys, "eng": "0" * 64})
    assert any("Q_DEV keys" in problem for problem in wrong)


def test_drive_blocks_a_terminal_failure_and_adopts_a_live_worker(tmp_path):
    entries = _driver_runs(tmp_path, modes={("eng", "ptflex_k4_C0.001"): "terminal"}, sleeps={("plc", "lgbm_l15_m100"): 2.0})
    live = next(e for e in entries if e.arm == "plc" and e.job.card_id == "lgbm_l15_m100")
    process = subprocess.Popen([sys.executable, "-c", FAKE_WORKER, str(live.spec)])
    _write_json(live.jobs_dir / eq.STATE_NAME, {"running": {live.key: {"pid": process.pid, "launch": 1, "started_utc": "t",
                                                                       "spec": str(live.spec)}}})
    calls = []
    outcome = e1.drive(entries, workers=4, poll_s=0.05, launcher=_fake_launcher(calls), env={})
    process.wait(timeout=30)
    assert list(outcome["blocked"]) == [e1.entry_name("eng", next(e.key for e in entries if e.arm == "eng" and e.job.card_id == "ptflex_k4_C0.001"))]
    assert live.key + "@plc" not in calls                                           # adopted, not launched again
    ledger = [json.loads(line) for line in (live.jobs_dir / eq.LEDGER_NAME).read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "adopted" and event["key"] == live.key for event in ledger)


# ======================================================================= a synthetic E1 world (tog_e1_1, q_run_1, labels_1, projection)
ROLE_OF_MATCH = np.repeat(np.array([0, 3, 4, 5]), 6)       # 24 matches: TRAIN 0-5, Q_DEV 6-11, Q_CAL 12-17, TEST 18-23
PER_MATCH = 3                                              # engagements c0, c1 (matched) and c2 (unmatched)
REUSED = (12, 18)                                          # a Q_CAL and a TEST match where c1 reuses c0's placebo window
SELECTED = {"lightgbm": "lgbm_l15_m200", "PT_flex": "ptflex_k8_C0.001"}
GROUPS = (("train", (0,)), ("validation", (3, 4)), ("test", (5,)))


def _rng(seed, role, what):
    return np.random.default_rng([int(seed), int(role), int(what)])


def _logit(p):
    return np.log(p / (1.0 - p))


def _follow(y, rng, strength):
    return np.clip(0.5 + strength * (2 * np.asarray(y) - 1) + rng.normal(0.0, 0.08, np.asarray(y).size), 0.02, 0.98)


def _near(delta):
    bits = np.zeros(delta.size, dtype=np.uint8)
    for bit, threshold in enumerate((0.005, 0.01, 0.02)):
        bits[np.abs(delta) < threshold] |= 1 << bit
    return bits


def _world(root, *, seed=17, test_shift=0.0, tamper=None):
    root = Path(root)
    tog, q_run, labels, projection = root / "tog", root / "q_run", root / "labels", root / "projection"
    n_match = ROLE_OF_MATCH.size
    n_q = n_match * PER_MATCH
    match = np.repeat(np.arange(n_match), PER_MATCH).astype(np.int64)
    role = ROLE_OF_MATCH[match].astype(np.int8)
    slot = np.tile(np.arange(PER_MATCH), n_match)
    cmk = np.array([f"m{m:03d}".encode() for m in match], dtype="S64")
    ck = np.array([f"c{s}".encode() for s in slot], dtype="S64")
    keys = np.char.add(np.char.add(cmk, b":"), ck).astype("S129")
    pre_ms = (300_000 + 200_000 * slot + 1_000 * match).astype(np.int64)
    d_ms = (20_000 + 1_000 * (np.arange(n_q) % 7)).astype(np.int64)
    query_pre = (10 * np.arange(n_q) + 5).astype(np.int64)
    query_h90 = query_pre + 3
    v_tau, svi = np.zeros(n_q), np.zeros(n_q)
    for code in (0, 3, 4, 5):
        rows = role == code
        rng = _rng(seed, code, 1)
        v_tau[rows] = rng.uniform(0.2, 0.8, int(rows.sum()))
        svi[rows] = rng.normal(0.0, 0.05, int(rows.sum())) + (test_shift if code == 5 else 0.0)
    y_q = (svi > 0).astype(np.int8)
    q_full = {"canonical_match_key": cmk, "component_key": ck, "engagement_key": keys, "match_index": match, "role": role,
              "fold": np.where(role == 0, match % 5, -1).astype(np.int8), "q_stop": np.zeros(n_q, dtype=bool), "y": y_q,
              "weight_raw": np.full(n_q, 1.0 / PER_MATCH), "query_pre": query_pre, "pre_ms": pre_ms,
              "elapsed_min": pre_ms / 60000.0, "frame_age_s": np.full(n_q, 2.0), "truncation_bits": np.zeros(n_q, dtype=np.int64),
              "near_zero_bits": _near(svi), "exact_zero": np.zeros(n_q, dtype=np.int8), "label_svi_delta": svi, "pt_p_pre": v_tau,
              "eval_n_kills": np.ones(n_q, dtype=np.int16), "eval_firstkill_credit_blue": np.ones(n_q, dtype=np.int16),
              "eval_firstkill_credit_red": np.ones(n_q, dtype=np.int16)}
    q_public, q_sealed = q_rows_module.seal_test({name: value.copy() for name, value in q_full.items()})
    if tamper == "missing_main_key":
        q_sealed = {name: value[1:] for name, value in q_sealed.items()}
    _save_npz(q_run / "Q_ROWS_h90.npz", **q_public)
    _save_npz(q_run / "Q_TEST_SEALED_h90.npz", **q_sealed)
    _save_npz(labels / "ENGAGEMENTS.npz", canonical_match_key=cmk, component_key=ck, role=role, valid_h90=np.ones(n_q, dtype=bool),
              query_pre=query_pre, pre_ms=pre_ms, query_h90=query_h90, post_ms_h90=pre_ms + d_ms, F_pre_ms=pre_ms - 2_000)
    objective = np.zeros((int(query_h90.max()) + 1, 9), dtype=np.int64)
    for code in (0, 3, 4, 5):
        rows = np.flatnonzero(role == code)
        rng = _rng(seed, code, 4)
        base = rng.integers(0, 3, (rows.size, 9))
        objective[query_pre[rows]] = base
        objective[query_h90[rows]] = base + (rng.random(rows.size) < 0.5)[:, None] * (np.arange(9) == 0)
    projection.mkdir(parents=True, exist_ok=True)
    np.save(projection / "OBJECTIVE_COUNTS.npy", objective)

    # windows, the matched pairs, V and the placebo labels ---------------------------------------------------------------
    matched = slot < 2
    s_ms = np.where(matched, pre_ms - 40_000, -1).astype(np.int64)
    for m in REUSED:
        s_ms[m * PER_MATCH + 1] = s_ms[m * PER_MATCH]
    start_age = np.where(matched, 3_000, -1).astype(np.int64)
    _save_npz(tog / "WINDOWS_h90.npz", engagement_key=keys, match_index=match, role=role, tau_ms=pre_ms, D_ms=d_ms, s_ms=s_ms,
              start_age_plc_ms=start_age)
    pairs = np.flatnonzero(matched)
    n_p = pairs.size
    p_role, p_match = role[pairs], match[pairs]
    p_s, p_end_ms = s_ms[pairs], s_ms[pairs] + d_ms[pairs]
    p_start, p_end = np.zeros(n_p), np.zeros(n_p)
    for code in (0, 3, 4, 5):
        rows = p_role == code
        rng = _rng(seed, code, 2)
        p_start[rows] = rng.uniform(0.2, 0.8, int(rows.sum()))
        p_end[rows] = np.clip(p_start[rows] + rng.normal(0.0, 0.04, int(rows.sum())) + (test_shift if code == 5 else 0.0), 0.01, 0.99)
    for m in REUSED:                                       # a reused window has one start state, so one V(s)
        a, b = np.searchsorted(pairs, [m * PER_MATCH, m * PER_MATCH + 1])
        p_start[b] = p_start[a]
    for code in (4, 5):                                    # an exact tie in Q_CAL and in TEST
        first = np.flatnonzero(p_role == code)[2]
        p_end[first] = p_start[first]
    start_state, end_state = np.full(n_p, -1, dtype=np.int64), np.full(n_p, -1, dtype=np.int64)
    parts, offset, state_counts = {}, 0, {}
    for k, (prefix, codes) in enumerate(GROUPS):
        sel = np.flatnonzero(np.isin(p_role, codes))
        times = sorted({(int(p_match[i]), int(t)) for i in sel for t in (p_s[i], p_end_ms[i])})
        index = {state: offset + j for j, state in enumerate(times)}
        starts = {(int(p_match[i]), int(p_s[i])) for i in sel}
        start_state[sel] = [index[(int(p_match[i]), int(p_s[i]))] for i in sel]
        end_state[sel] = [index[(int(p_match[i]), int(p_end_ms[i]))] for i in sel]
        rng = _rng(seed, codes[0], 3)
        n_s = len(times)
        part = {"state_id": np.arange(offset, offset + n_s, dtype=np.int64), "state_match": np.array([m for m, _ in times]),
                "state_ms": np.array([t for _, t in times]),
                "frame_ms": np.array([t - 3_000 if (m, t) in starts else t - 1_000 for m, t in times]),
                "xy": rng.normal(size=(n_s, 10, 2)), "xy_support": np.ones(n_s, dtype=bool), "window_id": pairs[sel],
                "window_start_state": start_state[sel], "window_end_state": end_state[sel],
                "feature_objective_counts": rng.integers(0, 3, (n_s, 9)).astype(np.int32)}
        name = f"part_{k:04d}.npz"
        parts[f"{prefix}_0000"] = {"file": name, "sha256": _save_npz(tog / "STATES_h90" / name, **part), "shard": k}
        state_counts.update({offset + j: int(part["feature_objective_counts"][j].sum()) for j in range(n_s)})
        offset += n_s
    listing = "".join(sorted(f"{entry['file']}:{entry['sha256']}\n" for entry in parts.values()))
    _write_json(tog / "STATES_h90" / "STATES_h90.json", {"stage": "states", "smoke": False, "parts": parts,
                                                         "parts_list_sha256": hashlib.sha256(listing.encode()).hexdigest()})
    test_p = p_role == 5
    public_end = np.where(test_p, np.nan, p_end)
    fold = q_full["fold"][pairs]
    _save_npz(tog / "PRED_PLACEBO_h90.npz", window_id=pairs, engagement_key=keys[pairs], match_index=p_match, role=p_role, fold=fold,
              ctx=np.zeros(n_p, dtype=np.int8), s_ms=p_s, end_ms=p_end_ms, start_state=start_state, end_state=end_state,
              logit_start=_logit(p_start), logit_end=_logit(public_end), raw_start=p_start, raw_end=public_end, p_start=p_start,
              p_end=public_end)
    _save_npz(tog / "PRED_PLACEBO_h90_TEST_SEALED.npz", window_id=pairs[test_p], engagement_key=keys[pairs][test_p],
              ctx=np.zeros(int(test_p.sum()), dtype=np.int8), end_state=end_state[test_p], logit_end=_logit(p_end[test_p]),
              raw_end=p_end[test_p], p_end=p_end[test_p], end_equals_a_start=es.end_is_a_start(start_state, end_state)[test_p])
    _write_json(tog / "PRED_PLACEBO_h90.json", {"stage": "placebo_v", "smoke": False})
    lab = er1.placebo_labels(p_start, p_end)
    plc_ck = np.char.add(b"plc-", ck[pairs].astype("S60")).astype("S64")
    plc_keys = np.char.add(np.char.add(cmk[pairs], b":"), plc_ck).astype("S129")
    plc_full = {"canonical_match_key": cmk[pairs], "component_key": plc_ck, "engagement_key": plc_keys, "match_index": p_match,
                "role": p_role, "fold": fold, "q_stop": np.zeros(n_p, dtype=bool), "y": lab["y"], "weight_raw": np.full(n_p, 0.5),
                "query_pre": np.arange(n_p, dtype=np.int64), "pre_ms": p_s, "elapsed_min": p_s / 60000.0,
                "frame_age_s": np.full(n_p, 3.0), "truncation_bits": np.zeros(n_p, dtype=np.int64),
                "near_zero_bits": lab["near_zero_bits"], "exact_zero": lab["exact_zero"], "label_svi_delta": lab["label_svi_delta"],
                "pt_p_pre": p_start.copy(), "eval_n_kills": np.zeros(n_p, dtype=np.int16),
                "eval_firstkill_credit_blue": np.zeros(n_p, dtype=np.int16), "eval_firstkill_credit_red": np.zeros(n_p, dtype=np.int16)}
    plc_public, plc_sealed = q_rows_module.seal_test({name: value.copy() for name, value in plc_full.items()})
    if tamper == "plc_label":                              # a valid label that p_end does not give
        plc_sealed["y"][0] = 1 - plc_sealed["y"][0]
    if tamper == "unsealed":
        plc_public["y"][np.flatnonzero(p_role == 5)[0]] = 1
    eng_full = {name: value[pairs].copy() for name, value in q_full.items()}
    eng_full["weight_raw"] = np.full(n_p, 0.5)
    eng_public, _ = q_rows_module.seal_test(eng_full)
    _save_npz(tog / "plc" / "Q_ROWS_h90.npz", **plc_public)
    _save_npz(tog / "plc" / "Q_TEST_SEALED_h90.npz", **plc_sealed)
    _save_npz(tog / "eng" / "Q_ROWS_h90.npz", **eng_public)
    _save_npz(tog / "plc" / "PLC_KEY_MAP_h90.npz", plc_row=np.arange(n_p), eng_row=np.arange(n_p), window_id=pairs,
              engagement_key_plc=plc_keys, engagement_key_eng=keys[pairs], query_pre_plc=np.arange(n_p, dtype=np.int64))
    for arm in e1.ARMS:
        _write_json(tog / arm / "Q_ROWS_h90.json", {"counts": "sealed summary"})

    # summaries (the TEST ones sealed), the stage records this task reads --------------------------------------------
    names = {0: "TRAIN", 3: "Q_DEV", 4: "Q_CAL", 5: "TEST"}

    def window_block(code):
        engagements, chosen = int((role == code).sum()), int((p_role == code).sum())
        return {"engagements": engagements, "matched": chosen, "match_rate": chosen / engagements,
                "note": f"shift {test_shift}" if code == 5 else "public"}

    obj_eng = objective[query_h90[pairs]].sum(axis=1) > objective[query_pre[pairs]].sum(axis=1)     # diagnostics.py:152-158
    obj_plc = np.array([state_counts[int(b)] > state_counts[int(a)] for a, b in zip(start_state, end_state)])

    def balance_block(code):
        own = p_role == code

        def share(flags):
            return {"windows": int(own.sum()), "with_objective_gain": int(flags[own].sum()), "share": float(flags[own].mean())}

        return {"pairs": int(own.sum()), "triggers": {"stratified_r": {"flag": False}, "deduplicated_r": {"flag": True}},
                "objective_share": {"engagement": share(obj_eng), "placebo": share(obj_plc),
                                    "pairs_without_objective_in_either": int((~obj_eng & ~obj_plc)[own].sum())},
                "reuse": {"rows": int(own.sum()), "distinct_windows": len({(int(m), int(t)) for m, t in zip(p_match[own], p_s[own])})},
                "note": f"shift {test_shift}" if code == 5 else "public"}

    sealed_windows = _write_json(tog / "WINDOWS_h90_TEST_SEALED.json", {"stage": "windows_TEST_sealed", "TEST": window_block(5)})
    _write_json(tog / "WINDOWS_h90.json", {"stage": "final_windows", "roles": {names[c]: window_block(c) for c in (0, 3, 4)},
                                           "windows_npz": {"sha256": _sha(tog / "WINDOWS_h90.npz")},
                                           "test_sealed_summary": {"sha256": sealed_windows}})
    test_balance = balance_block(5)
    if tamper == "balance_engagement":                     # sealed summaries that do not describe the TEST pairs
        test_balance["objective_share"]["engagement"]["with_objective_gain"] += 1
    if tamper == "balance_placebo":
        test_balance["objective_share"]["placebo"]["with_objective_gain"] += 1
    if tamper == "balance_neither":
        test_balance["objective_share"]["pairs_without_objective_in_either"] += 1
    if tamper == "balance_reuse":
        test_balance["reuse"]["distinct_windows"] += 1
    sealed_balance = _write_json(tog / "BALANCE_TEST_SEALED.json", {"stage": "balance_test_sealed", "roles": {"TEST": test_balance}})
    _write_json(tog / "BALANCE.json", {"stage": "balance", "passed": True, "roles": {names[c]: balance_block(c) for c in (0, 3, 4)},
                                       "files": {"BALANCE_TEST_SEALED.json": {"sha256": sealed_balance}}})
    rows_files = {f"{arm}/Q_ROWS_h90.npz": tog / arm / "Q_ROWS_h90.npz" for arm in e1.ARMS}
    rows_files.update({"plc/PLC_KEY_MAP_h90.npz": tog / "plc" / "PLC_KEY_MAP_h90.npz", "plc/Q_TEST_SEALED_h90.npz": tog / "plc" / "Q_TEST_SEALED_h90.npz"})
    _write_json(tog / "E1_ROWS.json", {"stage": "rows", "passed": True,
                                       "files": {label: {"path": str(path), "sha256": _sha(path)} for label, path in rows_files.items()}})

    # q_run_1's main models: TEST_SCORES (TEST step) and the reused search jobs' Q_CAL predictions (rehearsal) -----------
    for k, (family, (final, search)) in enumerate(e1.MAIN_CELLS.items()):
        strength = 0.3 if family == "lightgbm" else 0.1
        test_rows = np.flatnonzero(role == 5)
        p = _follow(y_q[test_rows], _rng(seed, 5, 10 + k), strength)
        digest = _save_npz(q_run / "TEST_SCORES" / f"{final}.npz", row=test_rows, engagement_key=keys[test_rows], logit=_logit(p), p=p)
        _write_json(q_run / "TEST_SCORES" / f"{final}.json", {"key": final, "pred_sha256": digest})
        _write_json(q_run / "jobs" / final / "REUSE.json", {"reuse_of_key": search})
        cal_rows = np.flatnonzero(role == 4)
        p = _follow(y_q[cal_rows], _rng(seed, 4, 10 + k), strength)
        _save_npz(q_run / "jobs" / search / "pred_q_cal.npz", row=cal_rows, logit=_logit(p), p=p)

    # the selected cells of both runs (their Q_DEV/Q_CAL files; TEST scores come from the scorer stubs) ---------------
    plan = {job.card_id: job for job in eq.plan_jobs(qjobs.plan())}
    tables = {"eng": eng_public, "plc": plc_public}
    truth = {"eng": y_q[pairs], "plc": lab["y"]}
    cells, selection, test_p_of, qcal_p_of = {}, {}, {}, {}
    for a, arm in enumerate(e1.ARMS):
        table = tables[arm]
        cells[arm], selection[arm] = {}, {}
        for k, family in enumerate(e1.FAMILIES):
            job = plan[SELECTED[family]]
            key = qtest.key_of(job.job_id)
            directory = tog / arm / "jobs" / key
            p_all = _follow(truth[arm], _rng(seed, 7 + a, 20 + k), 0.3 if family == "lightgbm" else 0.1)
            dev, cal = np.flatnonzero(table["role"] == 3), np.flatnonzero(table["role"] == 4)
            dev_sha = _save_npz(directory / "pred_q_dev.npz", row=dev, logit=_logit(p_all[dev]), p=p_all[dev])
            cal_sha = _save_npz(directory / "pred_q_cal.npz", row=cal, logit=_logit(p_all[cal]), p=p_all[cal])
            spec_sha = _write_json(directory / "SPEC.json", {"job": job.to_dict(), "inputs": {"rows": str(tog / arm / "Q_ROWS_h90.npz")}})
            (directory / "model_refit.pkl").write_bytes(f"{arm}:{key}".encode())
            brier_cal = qeval.brier(table["y"][cal], p_all[cal], match_ids=table["match_index"][cal])
            _write_json(directory / "RESULT.json", {"status": "completed", "spec_sha256": spec_sha,
                                                    "scores": {"q_dev": {"raw_match_brier": 0.2, "rows": int(dev.size), "pred_sha256": dev_sha},
                                                               "q_cal": {"raw_match_brier": brier_cal, "rows": int(cal.size), "pred_sha256": cal_sha}}})
            cells[arm][family] = qtest.cell_record(tog / arm, job.to_dict())
            selection[arm][family] = {"selected": job.card_id, "selected_key": key, "brier": 0.2}
            test_p_of[key + arm] = (np.flatnonzero(table["role"] == 5), p_all)
            qcal_p_of[key + arm] = (cal, p_all)
    _write_json(tog / e1.SELECTION_NAME, {"stage": "select", "passed": True, "files": {}, "selection": selection})

    def scorer_for(lookup, extra):
        calls = []

        def score(cell):
            arm = Path(cell["source_dir"]).parent.parent.name
            rows, p_all = lookup[cell["key"] + arm]
            calls.append(f"{arm}:{cell['key']}")
            table = tables[arm]
            return {"row": rows, "engagement_key": table["engagement_key"][rows], "logit": _logit(p_all[rows]), "p": p_all[rows],
                    "reload_max_abs_logit_diff": 0.0, "reload_rows": 0, "kind": "reference", **extra}
        score.calls = calls
        return score

    paths = e1.Paths(source=tog, out=tog, q_run=q_run, labels=labels, projection=projection, prespec=root / "PRESPEC_E1_KO.md",
                     amendment=root / "AMENDMENT_E1_KO.md", exposure=root / "EXPOSURE_KO.md", openings=root / "OPENINGS.json")
    paths.prespec.write_text("prespec e1", encoding="utf-8")
    paths.amendment.write_text("amendment e1", encoding="utf-8")
    return SimpleNamespace(root=root, paths=paths, cells=cells, selection=selection, pairs=pairs, p_role=p_role, keys=keys,
                           plc_keys=plc_keys, test_scorer=scorer_for(test_p_of, {}),
                           qcal_scorer=scorer_for(qcal_p_of, {"stored_q_cal_max_abs_logit_diff": 0.0}))


def _lock(world):
    """A lock over the synthetic world in the shape lock_stage writes (core files, code, the four cells, rules)."""

    paths = world.paths
    verify = {label: {"path": str(path), "sha256": _sha(path)} for label, path in e1.core_files(paths).items()}
    verify.update({f"code:{name}": {"path": str(e1.ROOT / name), "sha256": _sha(e1.ROOT / name)} for name in e1.CODE_FILES})
    for arm in e1.ARMS:
        for family in e1.FAMILIES:
            cell = world.cells[arm][family]
            for name in ("SPEC.json", "RESULT.json", cell["model_file"], "pred_q_dev.npz"):
                path = Path(cell["source_dir"]) / name
                verify[e1.cell_label(arm, cell["key"], name)] = {"path": str(path), "sha256": _sha(path)}
    lock = {"version": "test", "experiment": "E1", "locked_utc": "2026-10-13T00:00:00+00:00",
            "prespec": {"sha256": _sha(paths.prespec), "commit": e1.PRESPEC_COMMIT},
            "amendment": {"sha256": _sha(paths.amendment), "commit": e1.AMENDMENT_COMMIT},
            "verify": verify, "rules": eq._clean(e1.rules()), "environment": eq.environment(), "cells": world.cells,
            "selection": world.selection}
    path = paths.out / e1.LOCK_NAME
    _write_json(path, lock)
    return _sha(path)


# ======================================================================= the Q_CAL rehearsal (Ruling 55) and its record
def test_qcal_stage_runs_the_test_evaluation_path_on_q_cal_with_stand_ins(tmp_path):
    world = _world(tmp_path)
    assert e1.qcal_stage(world.paths, scorer=world.qcal_scorer) == 0
    record = _read(world.paths.out / e1.QCAL_NAME)
    assert record["split"] == "Q_CAL" and record["rehearsal"]["passed"] is True
    assert record["result"]["decision"]["outcome"] in e1.OUTCOMES
    assert sorted(world.qcal_scorer.calls) == sorted(f"{arm}:{world.cells[arm][f]['key']}" for arm in e1.ARMS for f in e1.FAMILIES)
    checks = record["rehearsal"]["checks"]
    assert checks["joined_engagement_labels_equal_the_eng_table"] and checks["joined_placebo_labels_equal_the_plc_table"]
    assert checks["state_parts_follow_their_shard_groups"] is True
    assert all(diff <= 1e-10 for diff in checks["brier_abs_diff_to_worker"].values())
    standins = record["rehearsal"]["stand_ins"]
    plc = _load(standins["plc_labels"]["path"])
    sealed = _load(world.paths.source / "plc" / "Q_TEST_SEALED_h90.npz")                # the synthetic sealed file (test only)
    assert list(plc) == list(sealed) and all(plc[k].dtype == sealed[k].dtype for k in plc)
    pred = _load(standins["pred_end"]["path"])
    assert tuple(pred) == es.SEALED_COLUMNS
    q_sealed = _load(world.paths.q_run / "Q_TEST_SEALED_h90.npz")
    eng = _load(standins["eng_labels"]["path"])
    assert list(eng) == list(q_sealed) and all(eng[k].dtype == q_sealed[k].dtype for k in eng)
    for family in e1.FAMILIES:
        main = _load(standins[f"main:{family}"]["path"])
        assert {"row", "engagement_key", "logit", "p"} <= set(main)
    q_cal_keys = set(world.keys[world.pairs][world.p_role == 4].tolist())
    assert set(eng["engagement_key"].tolist()) >= q_cal_keys and not set(plc["engagement_key"].tolist()) & set(
        world.plc_keys[world.p_role == 5].tolist())
    assert record["result"]["rows"] == int((world.p_role == 4).sum())
    with pytest.raises(FileExistsError):
        e1.qcal_stage(world.paths, scorer=world.qcal_scorer)


def test_parts_follow_shard_groups_refuses_a_part_of_another_group_or_a_window_in_no_part(tmp_path):
    world = _world(tmp_path)
    paths = world.paths
    assert e1.parts_follow_shard_groups(paths) is True
    record = _read(paths.states_record)
    swapped = dict(record, parts={("test_0000" if name == "validation_0000" else "validation_0000" if name == "test_0000" else name): entry
                                  for name, entry in record["parts"].items()})
    paths.states_record.write_text(json.dumps(swapped), encoding="utf-8")
    assert e1.parts_follow_shard_groups(paths) is False
    dropped = dict(record, parts={name: entry for name, entry in record["parts"].items() if name != "test_0000"})
    paths.states_record.write_text(json.dumps(dropped), encoding="utf-8")
    assert e1.parts_follow_shard_groups(paths) is False


def test_qcal_stage_stops_when_the_test_windows_would_not_be_found_at_the_test_step(tmp_path):
    world = _world(tmp_path)
    record = _read(world.paths.states_record)
    record["parts"] = {name: entry for name, entry in record["parts"].items() if name != "test_0000"}
    world.paths.states_record.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(e1.Blocked, match="state_parts_follow_their_shard_groups"):
        e1.qcal_stage(world.paths, scorer=world.qcal_scorer)
    assert not (world.paths.out / e1.QCAL_NAME).exists()


def _keys_anywhere(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            _keys_anywhere(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys_anywhere(item, found)
    return found


def test_qcal_record_holds_no_test_figure_and_ignores_every_test_input(tmp_path):
    a, b = _world(tmp_path / "a"), _world(tmp_path / "b", test_shift=0.3)
    for world in (a, b):
        assert e1.qcal_stage(world.paths, scorer=world.qcal_scorer) == 0
    ra, rb = _read(a.paths.out / e1.QCAL_NAME), _read(b.paths.out / e1.QCAL_NAME)
    assert ra["result"] == rb["result"]                                                  # every TEST input differs
    assert "TEST" not in _keys_anywhere(ra) and "TEST" not in _keys_anywhere(ra["result"])
    assert _sha(a.paths.source / "plc" / "Q_TEST_SEALED_h90.npz") != _sha(b.paths.source / "plc" / "Q_TEST_SEALED_h90.npz")


# ======================================================================= part B: the TEST step (order and failure handling)
def test_run_test_step_checks_then_scores_then_appends_the_opening_before_any_sealed_read(tmp_path, monkeypatch):
    world = _world(tmp_path)
    lock_sha = _lock(world)
    events = []
    real_read, real_append = e1.read_sealed, eq.append_opening

    def read(label, path, *args, **kwargs):
        events.append(("sealed", label))
        return real_read(label, path, *args, **kwargs)

    def append(path, entry):
        events.append(("openings", entry["experiment"]))
        return real_append(path, entry)

    def scorer(cell):
        events.append(("score", cell["key"]))
        return world.test_scorer(cell)

    monkeypatch.setattr(e1, "read_sealed", read)
    monkeypatch.setattr(eq, "append_opening", append)
    assert e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=scorer) == 0
    kinds = [kind for kind, _ in events]
    opening = kinds.index("openings")
    assert kinds.count("score") == 4 and max(i for i, kind in enumerate(kinds) if kind == "score") < opening
    assert min(i for i, kind in enumerate(kinds) if kind == "sealed") > opening
    assert {label for kind, label in events if kind == "sealed"} == set(e1.SEALED_LABELS)
    report = _read(world.paths.out / e1.REPORT_NAME)
    assert report["split"] == "TEST" and report["result"]["decision"]["outcome"] in e1.OUTCOMES
    assert report["result"]["rows"] == int((world.p_role == 5).sum())
    assert [o["experiment"] for o in _read(world.paths.openings)["openings"]] == ["E1"]
    assert _read(world.paths.out / e1.OPENING_NAME)["lock_sha256"] == lock_sha
    with pytest.raises(FileExistsError):                                                  # opened once
        e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=world.test_scorer)


def test_run_test_step_evaluates_with_the_function_the_rehearsal_ran(tmp_path):
    world = _world(tmp_path)
    lock_sha = _lock(world)
    assert e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=world.test_scorer) == 0
    report = _read(world.paths.out / e1.REPORT_NAME)
    scores = {arm: {family: e1.load_scores(world.paths.out / arm / e1.SCORES_DIR, world.cells[arm][family]) for family in e1.FAMILIES}
              for arm in e1.ARMS}
    again = e1.evaluate_opened(world.paths, "TEST", e1.sealed_paths(world.paths), scores, e1.main_paths(world.paths))
    assert report["result"] == eq._clean(again)


def test_run_test_step_blocks_before_scoring_and_opening(tmp_path):
    world = _world(tmp_path)
    lock_sha = _lock(world)
    assert e1.run_test_step(world.paths, expected_lock_sha256="0" * 64, scorer=world.test_scorer) == 2
    assert world.test_scorer.calls == [] and not world.paths.openings.exists()
    with (world.paths.source / "plc" / "Q_ROWS_h90.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")                                                                 # a bound file changed after the lock
    assert e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=world.test_scorer) == 2
    blocked = _read(world.paths.out / "E1_BLOCKED_test_2.json")
    assert "plc:Q_ROWS_h90.json" in blocked["reasons"]["changed_since_lock"]
    assert world.test_scorer.calls == [] and not world.paths.openings.exists() and not (world.paths.out / e1.OPENING_NAME).exists()


def test_run_test_step_refuses_a_row_table_that_does_not_seal_test_before_scoring(tmp_path):
    world = _world(tmp_path, tamper="unsealed")
    lock_sha = _lock(world)
    assert e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=world.test_scorer) == 2
    assert world.test_scorer.calls == [] and not world.paths.openings.exists()


def test_run_test_step_blocks_when_scoring_is_refused(tmp_path):
    world = _world(tmp_path)
    lock_sha = _lock(world)

    def refusing(cell):
        raise eq.ScoringRefused(f"{cell['key']}: re-loaded model differs")

    assert e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=refusing) == 2
    assert not world.paths.openings.exists() and not (world.paths.out / e1.OPENING_NAME).exists()


@pytest.mark.parametrize("tamper, message", [("plc_label", "placebo labels"), ("missing_main_key", "no sealed label"),
                                             ("balance_engagement", "objective_flags_equal_the_balance_table"),
                                             ("balance_placebo", "objective_flags_equal_the_balance_table"),
                                             ("balance_neither", "objective_flags_equal_the_balance_table"),
                                             ("balance_reuse", "reused_pairs_equal_the_balance_table")])
def test_run_test_step_writes_a_failure_record_after_the_opening_and_never_reports(tmp_path, tamper, message):
    world = _world(tmp_path, tamper=tamper)
    lock_sha = _lock(world)
    with pytest.raises(Exception) as raised:
        e1.run_test_step(world.paths, expected_lock_sha256=lock_sha, scorer=world.test_scorer)
    assert message in str(raised.value)
    failed = _read(world.paths.out / "E1_FAILED_test_1.json")
    assert failed["failed"] is True and failed["opening"]["experiment"] == "E1" and failed["rule"].startswith("do not re-run")
    assert [o["experiment"] for o in _read(world.paths.openings)["openings"]] == ["E1"]
    assert not (world.paths.out / e1.REPORT_NAME).exists()


def test_verify_lock_names_a_missing_or_changed_entry_and_another_commit(tmp_path):
    world = _world(tmp_path)
    _lock(world)
    lock = _read(world.paths.out / e1.LOCK_NAME)
    assert e1.verify_lock(lock, world.paths) == {}
    assert "prespec_hash_missing" in e1.verify_lock({**lock, "prespec": {}}, world.paths)
    assert "prespec_not_the_approved_commit" in e1.verify_lock({**lock, "prespec": {**lock["prespec"], "commit": "bd300bf"}}, world.paths)
    assert "amendment_not_the_approved_commit" in e1.verify_lock({**lock, "amendment": {**lock["amendment"], "commit": "x"}}, world.paths)
    verify = {label: entry for label, entry in lock["verify"].items() if label != "sealed:q_run_1/Q_TEST_SEALED_h90.npz"}
    assert e1.verify_lock({**lock, "verify": verify}, world.paths)["entries_missing"] == ["sealed:q_run_1/Q_TEST_SEALED_h90.npz"]
    assert "rules_differ" in e1.verify_lock({**lock, "rules": {}}, world.paths)
    assert "environment_differs" in e1.verify_lock({**lock, "environment": {**lock["environment"], "numpy": "0"}}, world.paths)
    moved = {**lock, "verify": {**lock["verify"], "plc:Q_ROWS_h90.npz": {**lock["verify"]["plc:Q_ROWS_h90.npz"], "path": str(tmp_path / "x.npz")}}}
    assert "plc:Q_ROWS_h90.npz" in e1.verify_lock(moved, world.paths)["paths_differ_from_this_run"]


def test_lock_counts_and_records_hold_no_test_entry():
    counts = {"eng": {"TRAIN": {"rows": 3}, "Q_DEV": {"rows": 2}, "Q_CAL": {"rows": 1}},
              "plc": {"TRAIN": {"rows": 3}, "Q_DEV": {"rows": 2}, "Q_CAL": {"rows": 1}}}
    assert e1.public_counts(counts) == counts
    with pytest.raises(e1.Blocked):
        e1.public_counts({**counts, "eng": {**counts["eng"], "TEST": {"rows": 9}}})
