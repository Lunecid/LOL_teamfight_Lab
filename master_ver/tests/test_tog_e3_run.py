"""Tests for the E3 run (PRESPEC_E3_KO.md; PLAN_E1_E5_KO.md Task 9; SDD Rulings 18, 38, 39).

Pure checks first: the contrast weights read from the external report, the e3_evaluate.py mirror (score order, sealed
labels, per-match losses), the point and bootstrap-reproduction rules, the decision, the influence bootstrap, the kill
strata, the patch dispersion, the player distribution and the JSON cleaning.  One test pins the bootstrap-reproduction
path at the real size (38,424 matches, 146,845 rows, 2,000 draws): the six-cell call of e3_run gives the meanD draws of
e3_evaluate's 23-cell, five-contrast call bit for bit.  The pipeline tests build a small synthetic external set under
tmp_path (row table, player matrix, six score files, sealed labels, external report, score lock and opening record)
and run the lock and the run stage there; the real sealed labels are never touched.
"""
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from studies.q_production_20261005 import qeval
from studies.tog_experiments_20261012 import e3_run as E
from studies.tog_experiments_20261012.boot import residual_interval, stratum_difference, twoway_interval
from studies.tog_experiments_20261012.common import sha256_file
from studies.tog_experiments_20261012.e3_players import players_by_match_count

PATCHES = ("15.18", "15.19", "15.20", "15.21", "15.22")
SHORT = ("LGBM_S1_A1_M1_E1_C1_R1_D1", "PT_FLEX", "LGBM_S1_A0_M0_E1_C1_R1_D0", "LGBM_S0_A0_M0_E1_C1_R1_D0",
         "LGBM_S1_A0_M0_E1_C1_R1_D1", "LGBM_S0_A0_M0_E1_C1_R1_D1")


def full(short):
    """The external score key of a cell (EXTERNAL_REPORT.json contrasts)."""
    if short == "PT_FLEX":
        return "final__reference__PT_FLEX__PT_flex__selected~PT_flex"
    return f"final__module__{short}__lightgbm__selected~lightgbm"


def neural(g, s, a, c, d):
    model = "relational_mpnn" if g else "team_set_pair_readout"
    return f"final__module__N_G{g}_S{s}_A{a}_M0_E1_C{c}_R1_D{d}__{model}__selected~common"


CONS = {"P1": {full("LGBM_S1_A1_M1_E1_C1_R1_D1"): 1.0, full("PT_FLEX"): -1.0},
        "S_effect_LGBM_meanD": {full("LGBM_S1_A0_M0_E1_C1_R1_D0"): 0.5, full("LGBM_S0_A0_M0_E1_C1_R1_D0"): -0.5,
                                full("LGBM_S1_A0_M0_E1_C1_R1_D1"): 0.5, full("LGBM_S0_A0_M0_E1_C1_R1_D1"): -0.5}}
LEVELS = {"P1": 0.975, "S_effect_LGBM_meanD": 0.95}
D1 = {full("LGBM_S1_A0_M0_E1_C1_R1_D1"): 1.0, full("LGBM_S0_A0_M0_E1_C1_R1_D1"): -1.0}


def external_contrasts():
    """The five contrasts of EXTERNAL_REPORT.json: the same 23 cells, weights and order as e3_evaluate.py:216."""
    p2 = {c: {neural(g, s, 0, c, d): w for d in (0, 1) for g, s, w in ((1, 1, 0.5), (0, 1, -0.5), (1, 0, -0.5), (0, 0, 0.5))}
          for c in (1, 0)}
    return {"P1": CONS["P1"], "P2_C1": p2[1], "P2_C0": p2[0], "P1_best": {neural(1, 1, 1, 0, 0): 1.0, full("PT_FLEX"): -1.0},
            "S_effect_LGBM_meanD": CONS["S_effect_LGBM_meanD"]}


EXTERNAL_LEVELS = {"P1": 0.975, "P2_C1": 0.975, "P2_C0": 0.95, "P1_best": 0.95, "S_effect_LGBM_meanD": 0.95}


def external_boot(y, preds, match, cons, levels, return_draws=False):
    """A literal transcription of e3_evaluate.py q_block lines 106-108 (preds in sorted key order, line 174)."""
    losses = {k: qeval.per_match_loss(y, p, match) for k, p in preds.items()}
    shared, aligned = qeval.align_cells(losses)
    return qeval.paired_match_bootstrap(shared, aligned, cons, levels=levels, return_draws=return_draws)


def q_contrasts(boot):
    return {n: {"point": c["point"], "interval": c["interval"], "level": c["level"]} for n, c in boot["contrasts"].items()}


def hand_losses(y, p, match):
    ids = np.unique(match)
    return ids, np.array([np.mean((p[match == i] - y[match == i]) ** 2) for i in ids])


def hand_contrast(y, preds, match, weights):
    total = None
    for key, w in weights.items():
        ids, losses = hand_losses(y, preds[key], match)
        total = w * losses if total is None else total + w * losses
    return ids, total


# ---------------------------------------------------------------- pure checks

def test_contrast_weights_follow_the_report_and_the_prespec():
    weights = E.contrast_weights({"contrasts": external_contrasts()})
    assert list(weights) == ["P1", "S_effect_LGBM_meanD", "D1"]
    assert list(weights["P1"].items()) == list(CONS["P1"].items())
    assert list(weights["S_effect_LGBM_meanD"].items()) == list(CONS["S_effect_LGBM_meanD"].items())   # report order
    assert list(weights["D1"].items()) == list(D1.items())
    swapped = external_contrasts()
    swapped["S_effect_LGBM_meanD"] = {k: -w for k, w in CONS["S_effect_LGBM_meanD"].items()}
    with pytest.raises(ValueError):
        E.contrast_weights({"contrasts": swapped})
    missing = external_contrasts()
    del missing["S_effect_LGBM_meanD"]
    with pytest.raises(ValueError):
        E.contrast_weights({"contrasts": missing})
    renamed = external_contrasts()
    renamed["P1"] = {full("LGBM_S1_A1_M1_E1_C1_R1_D0"): 1.0, full("PT_FLEX"): -1.0}
    with pytest.raises(ValueError):
        E.contrast_weights({"contrasts": renamed})


def test_ordered_scores_and_sealed_labels_mirror_e3_evaluate(tmp_path):
    keys = np.asarray([b"a:1", b"b:1", b"c:1"], dtype="S129")
    np.savez(tmp_path / "other.npz", engagement_key=keys[[2, 0, 1]], p=np.array([0.3, 0.1, 0.2]))
    np.savez(tmp_path / "same.npz", engagement_key=keys, p=np.array([0.1, 0.2, 0.3]))
    p, identical = E.ordered_scores(tmp_path / "other.npz", keys)
    assert p.tolist() == [0.1, 0.2, 0.3] and identical is False and p.dtype == np.float64
    p, identical = E.ordered_scores(tmp_path / "same.npz", keys)
    assert p.tolist() == [0.1, 0.2, 0.3] and identical is True
    np.savez(tmp_path / "short.npz", engagement_key=keys[:2], p=np.array([0.1, 0.2]))
    with pytest.raises(RuntimeError):
        E.ordered_scores(tmp_path / "short.npz", keys)

    def sealed(name, **fields):
        base = {"h90__engagement_key": keys[[1, 2, 0]], "h90__valid": np.ones(3, bool), "h90__y": np.array([True, False, True]),
                "h60__engagement_key": keys, "h60__valid": np.ones(3, bool), "h60__y": np.zeros(3, bool)}
        base.update(fields)
        np.savez(tmp_path / name, **base)
        return tmp_path / name

    y = E.sealed_labels(sealed("s.npz"), keys)
    assert y.tolist() == [1.0, 1.0, 0.0] and y.dtype == np.float64      # h90 only; h60 is ignored
    with pytest.raises(RuntimeError):
        E.sealed_labels(sealed("invalid.npz", h90__valid=np.array([True, False, True])), keys)
    with pytest.raises(RuntimeError):
        E.sealed_labels(sealed("missing.npz", h90__engagement_key=keys[[1, 2, 2]]), keys)
    with pytest.raises(RuntimeError):
        E.sealed_labels(sealed("nonbinary.npz", h90__y=np.array([1, 2, 0], dtype=np.int8)), keys)


def test_per_match_contrast_is_the_weighted_sum_of_match_mean_briers():
    match = np.array([30, 10, 30, 20, 10, 10], dtype=np.int64)          # rows not ordered by match
    y = np.array([1, 0, 0, 1, 1, 0], dtype=float)
    preds = {"b": np.array([0.2, 0.4, 0.6, 0.9, 0.5, 0.1]), "a": np.array([0.7, 0.3, 0.2, 0.8, 0.6, 0.2])}
    shared, aligned = E.cell_losses(y, {k: preds[k] for k in sorted(preds)}, match)
    assert shared.tolist() == [10, 20, 30] and list(aligned) == ["a", "b"]
    weights = {"a": 1.0, "b": -1.0}
    D = E.per_match(aligned, weights)
    ids, expected = hand_contrast(y, preds, match, weights)
    assert ids.tolist() == shared.tolist()
    assert D == pytest.approx(expected, abs=1e-15)
    assert float(np.mean(D)) == pytest.approx(qeval.contrast_point(aligned, weights), abs=1e-15)


def test_point_check_and_bootstrap_reproduction_rules():
    report_point = -0.01350838677477667
    ok = E.point_check(report_point + 5e-10, report_point)
    assert ok["passed"] is True and ok["abs_diff"] == pytest.approx(5e-10, rel=1e-6) and ok["tolerance"] == 1e-9
    assert E.point_check(report_point + 2e-9, report_point)["passed"] is False
    assert E.point_check(float("nan"), report_point)["passed"] is False
    report = [-0.004699071263718071, -0.0039845740852971715]
    exact = E.reproduction(list(report), report)
    assert exact["reproduced"] is True and exact["bit_identical"] is True and exact["decimals"] == 6
    assert exact["rounded"] == [-0.004699, -0.003985] and exact["report_rounded"] == [-0.004699, -0.003985]
    close = E.reproduction([-0.0046990712, -0.0039845741], report)
    assert close["reproduced"] is True and close["bit_identical"] is False
    assert E.reproduction([-0.0046980712, -0.0039845741], report)["reproduced"] is False
    assert E.reproduction([float("nan"), -0.0039845741], report)["reproduced"] is False


def test_decide_lowers_claim_2_only_when_the_meanD_interval_contains_zero():
    assert E.decide([-0.0050, -0.0040]) == "kept"
    assert E.decide([0.0010, 0.0040]) == "kept"
    assert E.decide([-0.0010, 0.0010]) == "lowered"
    assert E.decide([-0.0020, 0.0]) == "lowered"            # an end at 0 contains 0
    assert E.decide([0.0, 0.0020]) == "lowered"
    with pytest.raises(ValueError):
        E.decide([float("nan"), 0.1])
    with pytest.raises(ValueError):
        E.decide([0.2, 0.1])


def test_bootstrap_reproduction_path_is_the_external_call_at_the_real_size():
    """e3_run's six-cell call reproduces the meanD draws of e3_evaluate's 23-cell call bit for bit (38,424 matches)."""
    rng = np.random.default_rng(20261012)
    m, rows = 38_424, 146_845
    match = np.sort(np.concatenate([np.arange(m), rng.integers(0, m, rows - m)])) * 3 + 7
    order = rng.permutation(rows)
    match = match[order]
    y = rng.integers(0, 2, rows).astype(float)
    cons = external_contrasts()
    cells = sorted({k for w in cons.values() for k in w})
    assert len(cells) == 23
    preds = {k: rng.uniform(0.02, 0.98, rows) for k in cells}
    external = external_boot(y, preds, match, cons, EXTERNAL_LEVELS, return_draws=True)
    weights = E.contrast_weights({"contrasts": cons})
    six = sorted({k for name in ("P1", "S_effect_LGBM_meanD") for k in weights[name]})
    shared, aligned = E.cell_losses(y, {k: preds[k] for k in six}, match)
    ours = E.pooled_bootstrap(shared, aligned, weights, return_draws=True)
    a, b = external["contrasts"]["S_effect_LGBM_meanD"], ours["contrasts"]["S_effect_LGBM_meanD"]
    assert np.array_equal(a["draws"], b["draws"])
    assert a["interval"] == b["interval"] and a["point"] == b["point"] and b["level"] == 0.95
    assert ours["resamples"] == 2000 and ours["seed"] == 20261011 and ours["matches"] == m
    assert np.array_equal(external["contrasts"]["P1"]["draws"], ours["contrasts"]["P1"]["draws"])   # same draws, P1 at 95 %
    assert E.reproduction(b["interval"], a["interval"])["bit_identical"] is True


def test_influence_bootstrap_is_the_mean_bootstrap_for_a_mean_and_point_plus_mean_e_otherwise():
    rng = np.random.default_rng(4)
    D = rng.normal(0.0, 1.0, 50)
    out = E.influence_bootstrap(float(D.mean()), D - D.mean(), resamples=200, seed=7, return_draws=True)
    plain = np.array([np.mean(D[pos]) for pos in qeval.bootstrap_indices(50, 200, 7)])
    assert out["draws"] == pytest.approx(plain, abs=1e-14)
    e = rng.normal(0.0, 1.0, 30)
    out = E.influence_bootstrap(0.25, e, resamples=100, seed=3, return_draws=True)
    expected = np.array([0.25 + float(np.mean(e[pos])) for pos in qeval.bootstrap_indices(30, 100, 3)])
    assert np.array_equal(out["draws"], expected)
    assert out["interval"] == np.quantile(expected, [0.025, 0.975]).tolist()
    assert out["level"] == 0.95 and out["quantiles"] == [0.025, 0.975] and out["matches"] == 30


def test_kill_strata_average_rows_within_match_and_stratum():
    match = np.array([10, 10, 10, 20, 20, 30, 40, 40, 40, 50], dtype=np.int64)
    kills = np.array([1, 1, 4, 2, 5, 1, 3, 4, 6, 2], dtype=np.int16)
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    a, b = full("LGBM_S1_A1_M1_E1_C1_R1_D1"), full("PT_FLEX")
    preds = {a: np.array([0.7, 0.2, 0.6, 0.3, 0.8, 0.9, 0.1, 0.4, 0.7, 0.5]),
             b: np.array([0.5, 0.5, 0.5, 0.4, 0.6, 0.7, 0.3, 0.3, 0.6, 0.6])}
    weights = {a: 1.0, b: -1.0}
    match_index = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    players = np.array([[0, 1, 2], [2, 3, 4], [0, 5, 6], [1, 4, 7], [8, 9, -1]], dtype=np.int64)
    out = E.kill_strata(y, preds, match, kills, weights, match_index, players)
    expect = {}
    for name, rows in (("1", kills == 1), ("2-3", (kills >= 2) & (kills <= 3)), ("4+", kills >= 4)):
        ids, D = hand_contrast(y[rows], {k: v[rows] for k, v in preds.items()}, match[rows], weights)
        entry = out["strata"][name]
        assert entry["rows"] == int(rows.sum()) and entry["matches"] == ids.size
        assert entry["twoway"]["point"] == pytest.approx(float(D.mean()), abs=1e-15)
        direct = twoway_interval(D, players[np.searchsorted(match_index, ids)])
        assert entry["twoway"]["interval"] == pytest.approx(list(direct["interval"]), abs=1e-15)
        assert entry["twoway"]["VU"] == pytest.approx(direct["VU"], abs=1e-15)
        assert entry["bootstrap"]["level"] == 0.95
        expect[name] = (ids, D)
    assert [out["strata"][s]["matches"] for s in ("1", "2-3", "4+")] == [2, 3, 3]
    full_d = {}
    for name in ("4+", "1"):
        ids, D = expect[name]
        values = np.full(match_index.size, np.nan)
        values[np.searchsorted(match_index, ids)] = D
        full_d[name] = (values, ~np.isnan(values))
    delta, e = stratum_difference(full_d["4+"][0], full_d["4+"][1], full_d["1"][0], full_d["1"][1])
    diff = out["difference"]
    assert diff["strata"] == ["4+", "1"]
    assert diff["point"] == delta == out["strata"]["4+"]["twoway"]["point"] - out["strata"]["1"]["twoway"]["point"]
    assert diff["equals_stratum_points"] is True
    direct = residual_interval(delta, e, players)
    assert diff["twoway"]["interval"] == pytest.approx(list(direct["interval"]), abs=1e-15)
    assert diff["twoway"]["V1"] == pytest.approx(direct["V1"], abs=1e-15) and diff["twoway"]["matches"] == 5
    assert diff["bootstrap"]["matches"] == 5


def test_dispersion_statistic_by_hand():
    out = E.dispersion({"a": 1.0, "b": 2.0, "c": 4.0}, {"a": 1.0, "b": 0.5, "c": 2.0}, 2.0)
    assert out["terms"] == {"a": 1.0, "b": 0.0, "c": 2.0} and out["statistic"] == 3.0 and out["df"] == 2
    assert out["p_value"] == pytest.approx(math.exp(-1.5), rel=1e-12)          # chi-square(2) upper tail
    five = E.dispersion(dict(zip(PATCHES, (1.0, 2.0, 3.0, 4.0, 5.0))), dict.fromkeys(PATCHES, 2.0), 3.0)
    assert five["statistic"] == 5.0 and five["df"] == 4
    assert five["p_value"] == pytest.approx(float(chi2.sf(5.0, 4)), rel=1e-12)
    assert five["chi2_upper_95"] == pytest.approx(float(chi2.ppf(0.95, 4)), rel=1e-12)
    assert "not independent" in five["note"]


def test_player_distribution_by_hand():
    players = np.array([[0, 1, 2], [0, 3, -1], [4, 1, 0]], dtype=np.int64)
    patch = np.array(["15.18", "15.19", "15.19"])
    out = E.player_distribution(players, patch)
    assert out["players"] == 5 and out["identified_slots"] == 8 and out["unidentified_slots"] == 1
    spread = out["matches_per_player"]
    assert (spread["min"], spread["median"], spread["max"]) == (1, 1.0, 3)
    assert spread["mean"] == pytest.approx(1.6) and spread["share_in_one_match"] == pytest.approx(0.6)
    assert spread["p90"] == pytest.approx(2.6) and spread["p99"] == pytest.approx(2.96)
    assert spread["histogram"] == [[1, 3], [2, 1], [3, 1]]
    assert out["players_in_more_than_one_patch"] == 2 and out["share_in_more_than_one_patch"] == pytest.approx(0.4)
    assert out["players_by_patch"] == {"15.18": 3, "15.19": 4}


def test_clean_maps_nan_and_inf_to_null():
    cleaned = E._clean({"a": float("nan"), "b": [np.float64("inf"), 1.0], "c": (1, 2), "d": np.int64(3), "e": np.bool_(True),
                        "f": Path("x/y"), "g": np.array([1.5, np.nan])})
    assert cleaned == {"a": None, "b": [None, 1.0], "c": [1, 2], "d": 3, "e": True, "f": str(Path("x/y")), "g": [1.5, None]}
    json.dumps(cleaned, allow_nan=False)


def test_append_opening_creates_appends_and_refuses_a_second_e3_opening(tmp_path):
    path = tmp_path / "OPENINGS.json"
    assert E.append_opening(path, {"experiment": "E2", "utc": "t0", "lock_sha256": "a"}) == []
    before = E.append_opening(path, {"experiment": "E3", "utc": "t1", "lock_sha256": "b"})
    assert [o["experiment"] for o in before] == ["E2"]
    with pytest.raises(RuntimeError):
        E.append_opening(path, {"experiment": "E3", "utc": "t2", "lock_sha256": "c"})
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert [o["lock_sha256"] for o in doc["openings"]] == ["a", "b"] and "rule" in doc


# ---------------------------------------------------------------- synthetic external set

def make_world(tmp_path, *, kills=True, failed_matches=0, drop_sealed_rows=0, edit_report=None):
    """A row table, player matrix, six score files, sealed labels, external report, score lock and opening record."""
    rng = np.random.default_rng(20261012)
    m = 60
    match_ids = 100 + 3 * np.arange(m)
    canon = [hashlib.sha256(f"fake-match-{i}".encode()).hexdigest() for i in range(m)]
    match_of_row = np.repeat(np.arange(m), rng.integers(1, 6, size=m))
    match_of_row = match_of_row[rng.permutation(match_of_row.size)]          # the row table is not sorted by match
    r = match_of_row.size
    comp = [hashlib.sha256(f"fake-component-{i}".encode()).hexdigest() for i in range(r)]
    keys = np.asarray([f"{canon[g]}:{comp[i]}" for i, g in enumerate(match_of_row)], dtype="S129")
    rows = {"engagement_key": keys, "canonical_match_key": np.asarray([canon[g] for g in match_of_row], dtype="S64"),
            "component_key": np.asarray(comp, dtype="S64"), "match_index": match_ids[match_of_row].astype(np.int64),
            "role": np.full(r, 6, dtype=np.int8), "y": np.full(r, -1, dtype=np.int8),
            "patch": np.asarray([PATCHES[g % 5] for g in match_of_row], dtype="S8")}
    if kills:
        rows["eval_n_kills"] = rng.integers(1, 7, size=r).astype(np.int16)
    q_inputs = tmp_path / "q_inputs"
    q_inputs.mkdir()
    rows_path = q_inputs / "Q_ROWS_h90.npz"
    np.savez(rows_path, **rows)
    rows_sha = sha256_file(rows_path)
    (q_inputs / "Q_ROWS_h90.json").write_text(json.dumps({"rows_sha256": rows_sha}), encoding="utf-8")

    out = tmp_path / "tog_e3_1"
    out.mkdir()
    # 150 players for 60 matches: about half the match pairs share a player, so VU differs from V1 (with 25 players
    # every pair is linked and VU = (sum e)^2 / M^2 = 0).  A player effect makes the S1 cells worse in some matches.
    players = np.stack([rng.choice(150, size=10, replace=False) for _ in range(m)]).astype(np.int64)
    effect = 0.3 * rng.normal(0.0, 1.0, size=150)[players].mean(axis=1)
    players[5, 3] = -1
    status = np.zeros(m, dtype=np.int8)
    status[:failed_matches] = 1
    players[:failed_matches] = -1
    players_path = out / "PLAYERS_kr_15_18_22.npz"
    np.savez(players_path, match_index=match_ids.astype(np.int64), canonical_match_key=np.asarray(canon, dtype="S64"),
             player_idx=players, map_status=status, players_by_match_count=players_by_match_count(players))
    (out / "PLAYERS_kr_15_18_22.json").write_text(json.dumps({"npz_sha256": sha256_file(players_path),
                                                              "run": {"input_sha256": {rows_path.as_posix(): rows_sha}}}),
                                                  encoding="utf-8")
    y = rng.integers(0, 2, size=r).astype(np.float64)
    scores = tmp_path / "scores"
    scores.mkdir()
    preds = {}
    for i, key in enumerate(sorted(full(s) for s in SHORT)):
        p = rng.uniform(0.02, 0.98, size=r)
        if "_S1_" in key:                                                  # away from y by the match's player effect
            p = np.clip(p + effect[match_of_row] * (1.0 - 2.0 * y), 0.02, 0.98)
        preds[key] = p
        perm = rng.permutation(r) if i % 2 else np.arange(r)               # half the files store another row order
        npz_path = scores / f"{key}.npz"
        np.savez(npz_path, row=perm, engagement_key=keys[perm], logit=np.log(p[perm] / (1 - p[perm])), p=p[perm], p_cal=p[perm])
        (scores / f"{key}.json").write_text(json.dumps({"key": key, "rows": r, "pred_sha256": sha256_file(npz_path),
                                                        "rows_table_sha256": rows_sha}), encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    sealed_path = labels / "EXTERNAL_LABELS_SEALED_kr_15_18_22.npz"
    perm = rng.permutation(r)[drop_sealed_rows:]
    np.savez(sealed_path, h90__engagement_key=keys[perm], h90__valid=np.ones(perm.size, bool), h90__y=y[perm].astype(bool),
             h90__svi_delta=rng.normal(size=perm.size), h60__engagement_key=keys[perm], h60__valid=np.ones(perm.size, bool),
             h60__y=(1.0 - y[perm]).astype(bool))
    sealed_sha = sha256_file(sealed_path)

    match, patch = rows["match_index"], rows["patch"].astype(str)
    ordered = {k: preds[k] for k in sorted(preds)}
    report = {"contrasts": CONS, "levels": LEVELS, "datasets": {"kr_15_18_22": {
        "q_h90": {"rows": r, "matches": m, "contrasts": q_contrasts(external_boot(y, ordered, match, CONS, LEVELS)),
                  "metrics": {k: {"raw": {"brier": qeval.brier(y, p, match_ids=match)}} for k, p in ordered.items()}},
        "q_h90_by_patch": {pp: {"contrasts": q_contrasts(external_boot(
            y[patch == pp], {k: v[patch == pp] for k, v in ordered.items()}, match[patch == pp], CONS, LEVELS))}
            for pp in PATCHES}}}}
    if edit_report:
        edit_report(report)
    external = tmp_path / "external"
    external.mkdir()
    (external / "EXTERNAL_REPORT.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    cells = {k: {"pred_sha256": sha256_file(scores / f"{k}.npz"), "meta_sha256": sha256_file(scores / f"{k}.json")}
             for k in preds}
    (external / "EXTERNAL_SCORE_LOCK.json").write_text(json.dumps({"datasets": {"kr_15_18_22": {
        "cells": cells, "sealed": {"sha256": sealed_sha},
        "files": {"q_inputs:Q_ROWS_h90.npz": {"sha256": rows_sha}, "labels:SEALED": {"sha256": sealed_sha}}}}}),
        encoding="utf-8")
    (external / "EXTERNAL_LABELS_OPENED.json").write_text(json.dumps({"openings": [
        {"utc": "2026-10-09T11:57:17+00:00", "sealed_sha256": {"kr_15_18_22": sealed_sha}}]}), encoding="utf-8")
    paths = E.Paths(out=out, players=players_path, rows=rows_path, scores=scores, sealed=sealed_path, external=external,
                    openings=tmp_path / "OPENINGS.json")
    truth = {"y": y, "preds": preds, "match": match, "patch": patch, "kills": rows.get("eval_n_kills"),
             "match_ids": match_ids, "players": players, "canon": canon, "keys": keys}
    return paths, truth


def _sealed_guard(monkeypatch, paths, calls=None):
    """np.load raises on the sealed file (calls is None) or records each load with the opening state (a list)."""
    real_load = np.load

    def load(file, *args, **kwargs):
        if Path(file).resolve() == paths.sealed.resolve():
            if calls is None:
                raise AssertionError("the sealed label file was loaded")
            opened = json.loads(paths.openings.read_text(encoding="utf-8"))["openings"] if paths.openings.exists() else []
            calls.append(([o["experiment"] for o in opened], (paths.out / E.OPENING_NAME).is_file()))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_lock_then_run_on_a_synthetic_world(tmp_path, monkeypatch):
    paths, truth = make_world(tmp_path)
    _sealed_guard(monkeypatch, paths)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    monkeypatch.undo()
    lock = _json(paths.out / E.LOCK_NAME)
    assert lock["prespec"]["commit"] == E.PRESPEC_COMMIT and lock["prespec"]["sha256"] == sha256_file(paths.prespec)
    assert lock["verify"]["sealed"]["sha256"] == sha256_file(paths.sealed)
    assert lock["verify"]["input:players"]["sha256"] == sha256_file(paths.players)
    assert sorted(lock["selected_cells"]["all"]) == sorted(full(s) for s in SHORT)
    assert {"python", "numpy", "scipy"} <= set(lock["environment"])
    assert lock["counts"]["rows"] == truth["keys"].size and lock["counts"]["matches"] == 60
    assert sum(lock["counts"]["kill_strata"][s]["rows"] for s in ("1", "2-3", "4+")) == truth["keys"].size
    assert lock["counts"]["matches_by_patch"] == dict.fromkeys(PATCHES, 12)
    assert all(lock["alignment"].values()) and all(lock["cross_checks"].values())
    assert not paths.openings.exists() and not (paths.out / E.REPORT_NAME).exists()
    with pytest.raises(FileExistsError):
        E.lock_stage(paths, require_committed_code=False)

    calls = []
    _sealed_guard(monkeypatch, paths, calls)
    assert E.run_stage(paths, expected_lock_sha256=sha256_file(paths.out / E.LOCK_NAME)) == 0
    monkeypatch.undo()
    assert calls == [(["E3"], True)]                       # one read, after both opening records exist
    report = _json(paths.out / E.REPORT_NAME)
    lock_sha = sha256_file(paths.out / E.LOCK_NAME)
    openings = _json(paths.openings)["openings"]
    assert [o["experiment"] for o in openings] == ["E3"] and openings[0]["lock_sha256"] == lock_sha
    assert report["lock"]["sha256"] == lock_sha and report["opening"]["lock_sha256"] == lock_sha
    assert _json(paths.out / E.OPENING_NAME)["lock_sha256"] == lock_sha
    checks = report["checks"]
    assert checks["points"]["P1"]["passed"] and checks["points"]["S_effect_LGBM_meanD"]["passed"]
    assert checks["bootstrap_reproduction"]["reproduced"] and checks["bootstrap_reproduction"]["bit_identical"]
    assert checks["d1_point_vs_report_metrics"]["abs_diff"] < 1e-12

    y, preds, match, players = truth["y"], truth["preds"], truth["match"], truth["players"]
    weights = {"P1": CONS["P1"], "S_effect_LGBM_meanD": CONS["S_effect_LGBM_meanD"], "D1": D1}
    patch_of_match = np.asarray([PATCHES[g % 5] for g in range(60)])
    for name, w in weights.items():
        ids, D = hand_contrast(y, preds, match, w)
        assert ids.tolist() == truth["match_ids"].tolist()
        entry = report["pooled"][name]["twoway"]
        direct = twoway_interval(D, players)
        assert entry["point"] == pytest.approx(float(D.mean()), abs=1e-15)
        assert entry["interval"] == pytest.approx(list(direct["interval"]), rel=1e-10)
        assert entry["V2"] == pytest.approx(direct["V2"], rel=1e-10) and entry["z"] == 1.96
        assert entry["theta_over_se_twoway"] == pytest.approx(entry["point"] / entry["se_twoway"], rel=1e-12)
        assert report["pooled"][name]["bootstrap"]["level"] == 0.95
        points, variances = {}, {}
        for pp in PATCHES:
            mask = patch_of_match == pp
            got = report["by_patch"][pp]["contrasts"][name]
            assert got["point"] == pytest.approx(float(D[mask].mean()), abs=1e-15)
            assert got["V"] == pytest.approx(twoway_interval(D[mask], players[mask])["V"], rel=1e-10)
            points[pp], variances[pp] = got["point"], got["V"]
        statistic = sum((points[pp] - entry["point"]) ** 2 / variances[pp] for pp in PATCHES)
        assert report["dispersion"][name]["statistic"] == pytest.approx(statistic, rel=1e-10)
        assert report["dispersion"][name]["df"] == 4
    assert any(report["by_patch"][pp]["contrasts"][name]["V"] != report["by_patch"][pp]["contrasts"][name]["V1"]
               for pp in PATCHES for name in weights)                    # the world tells V from V1
    meand = report["pooled"]["S_effect_LGBM_meanD"]["twoway"]["interval"]
    assert report["decision"]["outcome"] == E.decide(meand) and report["decision"]["text"] == E.DECISIONS[E.decide(meand)]
    strata = report["kill_strata"]
    kills = truth["kills"]
    ids, D4 = hand_contrast(y[kills >= 4], {k: v[kills >= 4] for k, v in preds.items()}, match[kills >= 4], CONS["P1"])
    assert strata["strata"]["4+"]["matches"] == ids.size
    assert strata["strata"]["4+"]["twoway"]["point"] == pytest.approx(float(D4.mean()), abs=1e-15)
    assert strata["difference"]["point"] == pytest.approx(
        strata["strata"]["4+"]["twoway"]["point"] - strata["strata"]["1"]["twoway"]["point"], abs=0)
    assert report["counts"]["rows"] == truth["keys"].size
    assert "15.16" in report["descriptive"]["test_15_16_sentence"]
    assert report["descriptive"]["players"]["players"] == int(np.unique(players[players >= 0]).size)
    texts = [(paths.out / name).read_text(encoding="utf-8") for name in (E.LOCK_NAME, E.REPORT_NAME, E.OPENING_NAME)]
    for key in truth["canon"][:10]:
        assert all(key not in text for text in texts)          # no match key leaves the player matrix
    assert sorted(p.name for p in paths.out.iterdir()) == sorted(
        ["PLAYERS_kr_15_18_22.npz", "PLAYERS_kr_15_18_22.json", E.LOCK_NAME, E.OPENING_NAME, E.REPORT_NAME])
    with pytest.raises(FileExistsError):
        E.run_stage(paths)
    assert [o["experiment"] for o in _json(paths.openings)["openings"]] == ["E3"]


def test_a_file_changed_after_the_lock_blocks_before_the_opening(tmp_path):
    paths, _ = make_world(tmp_path)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    key = full("PT_FLEX")
    data = dict(np.load(paths.scores / f"{key}.npz"))
    data["p"] = data["p"][::-1].copy()
    np.savez(paths.scores / f"{key}.npz", **data)
    assert E.run_stage(paths) == 2
    blocked = _json(paths.out / "E3_BLOCKED_run_1.json")
    assert f"score:{key}:npz" in blocked["reasons"]["changed_since_lock"]
    assert not paths.openings.exists() and not (paths.out / E.OPENING_NAME).exists()
    assert not (paths.out / E.REPORT_NAME).exists()


def test_a_lock_other_than_the_reviewed_one_blocks_before_the_opening(tmp_path):
    paths, _ = make_world(tmp_path)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    assert E.run_stage(paths, expected_lock_sha256="0" * 64) == 2
    blocked = _json(paths.out / "E3_BLOCKED_run_1.json")
    assert blocked["reasons"]["lock_is_not_the_reviewed_lock"]["now"] == sha256_file(paths.out / E.LOCK_NAME)
    assert not paths.openings.exists() and not (paths.out / E.OPENING_NAME).exists()


def test_the_command_line_run_needs_both_confirmations():
    for argv in (["run"], ["run", "--confirm-second-opening"], ["run", "--lock-sha256", "0" * 64]):
        with pytest.raises(SystemExit) as stop:
            E.main(argv)
        assert stop.value.code == 2                     # argparse refusal before any file is read


def test_another_interpreter_version_blocks_before_the_opening(tmp_path, monkeypatch):
    paths, _ = make_world(tmp_path)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    real = E.environment()
    monkeypatch.setattr(E, "environment", lambda: {**real, "numpy": "0.0.0"})
    assert E.run_stage(paths) == 2
    blocked = _json(paths.out / "E3_BLOCKED_run_1.json")
    assert blocked["reasons"]["environment_differs"] == {"numpy": {"locked": real["numpy"], "now": "0.0.0"}}
    assert not paths.openings.exists() and not (paths.out / E.OPENING_NAME).exists()


def test_a_point_that_differs_from_the_report_is_blocked_after_the_opening(tmp_path):
    def shift(report):
        report["datasets"]["kr_15_18_22"]["q_h90"]["contrasts"]["P1"]["point"] += 2e-9

    paths, _ = make_world(tmp_path, edit_report=shift)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    assert E.run_stage(paths) == 2
    blocked = _json(paths.out / "E3_BLOCKED_run_1.json")
    assert blocked["reasons"]["point_check_failed"]["P1"]["passed"] is False
    assert blocked["reasons"]["point_check_failed"]["S_effect_LGBM_meanD"]["passed"] is True
    assert blocked["opening"]["experiment"] == "E3"
    assert [o["experiment"] for o in _json(paths.openings)["openings"]] == ["E3"]
    assert not (paths.out / E.REPORT_NAME).exists()


def test_a_bootstrap_interval_that_is_not_reproduced_is_blocked_after_the_opening(tmp_path):
    def shift(report):
        report["datasets"]["kr_15_18_22"]["q_h90"]["contrasts"]["S_effect_LGBM_meanD"]["interval"][0] += 1e-6

    paths, _ = make_world(tmp_path, edit_report=shift)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    assert E.run_stage(paths) == 2
    blocked = _json(paths.out / "E3_BLOCKED_run_1.json")
    assert blocked["reasons"]["bootstrap_reproduction_failed"]["reproduced"] is False
    assert not (paths.out / E.REPORT_NAME).exists()


def test_missing_eval_n_kills_blocks_only_the_kill_strata(tmp_path):
    paths, _ = make_world(tmp_path, kills=False)
    assert E.lock_stage(paths, require_committed_code=False) == 0
    assert _json(paths.out / E.LOCK_NAME)["counts"]["kill_strata"]["blocked"] is True
    assert E.run_stage(paths) == 0
    report = _json(paths.out / E.REPORT_NAME)
    assert report["kill_strata"]["blocked"] is True and "eval_n_kills" in report["kill_strata"]["reason"]
    assert report["checks"]["bootstrap_reproduction"]["reproduced"] and report["decision"]["outcome"] in E.DECISIONS


def test_an_uncommitted_prespec_blocks_the_lock(tmp_path):
    paths, _ = make_world(tmp_path)
    loose = tmp_path / "PRESPEC_E3_copy.md"
    loose.write_bytes(paths.prespec.read_bytes())
    assert E.lock_stage(dataclasses.replace(paths, prespec=loose), require_committed_code=False) == 2
    blocked = _json(paths.out / "E3_BLOCKED_lock_1.json")
    assert "prespec_hash_or_commit_missing" in blocked["reasons"]
    assert not (paths.out / E.LOCK_NAME).exists()


def test_more_than_one_percent_unmapped_matches_blocks_the_lock(tmp_path):
    paths, _ = make_world(tmp_path, failed_matches=2)                    # 2 of 60 matches
    assert E.lock_stage(paths, require_committed_code=False) == 2
    blocked = _json(paths.out / "E3_BLOCKED_lock_1.json")
    assert blocked["reasons"]["mapping_failed_share_over_max"]["failed"] == 2
    assert not (paths.out / E.LOCK_NAME).exists()


def test_a_failure_after_the_opening_leaves_a_failure_record(tmp_path):
    paths, _ = make_world(tmp_path, drop_sealed_rows=1)                  # one q row has no sealed label
    assert E.lock_stage(paths, require_committed_code=False) == 0
    with pytest.raises(RuntimeError):
        E.run_stage(paths)
    failed = _json(paths.out / "E3_FAILED_run_1.json")
    assert failed["error"]["type"] == "RuntimeError" and "sealed" in failed["error"]["message"]
    assert failed["opening"]["experiment"] == "E3" and failed["stages"]
    assert (paths.out / E.OPENING_NAME).is_file() and not (paths.out / E.REPORT_NAME).exists()
    with pytest.raises(FileExistsError):                                 # no second opening under the same lock
        E.run_stage(paths)
