"""Tests for the KA post-hoc analyses K1-K6 (reports/tog_experiments_20261012/PLAN_KA_POSTHOC_KO.md section 6).

Synthetic data only.  The analysis functions run on small arrays; the stages (qdev, lock, test, launch) run on a
synthetic q_run/q_ka/tog_ka tree under tmp_path with a throw-away git repository holding a stand-in plan.  The real
KA labels, the q_run_1 files and the repository's OPENINGS/EXPOSURE ledgers are never touched.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from reports.q_production_20261005 import frame_age_seconds as FA
from reports.q_production_20261005 import kill_label as KL
from studies.q_production_20261005 import qeval, qtest, qtrain
from studies.tog_experiments_20261012 import ka_posthoc as K

NEW_PRED_CELLS = (K.NEST, *K.SEED_CELLS.values())


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def synthetic(seed: int = 0, matches: int = 60, low: int = 1, high: int = 6):
    """Rows of one split: match ids (not contiguous), KA labels, and predictions of every cell the analyses read."""
    rng = np.random.default_rng(seed)
    per = rng.integers(low, high, size=matches)
    match = np.repeat(np.arange(matches) * 7 + 3, per)
    y = rng.integers(0, 2, size=match.size).astype(np.float64)
    preds = {}
    for i, key in enumerate([*K.KA_CELLS, *NEW_PRED_CELLS]):
        preds[key] = _sigmoid((0.4 + 0.1 * (i % 4)) * (2 * y - 1) + rng.normal(0.0, 1.0, match.size))
    return match, y, preds, rng


def gate_p1(y, preds, match):
    return KL._contrast_block(y, {k: preds[k] for k in K.KA_CELLS}, match)["contrasts"]["contrasts"]["P1"]


# ----------------------------------------------------------------------------- K1 one-kill share (plan 2.2)

def test_share_one_kill_is_the_positive_gain_share_of_the_inventory():
    # kill-label-inventory.md:47: 2613.1 / (2613.1 + 550.1 - 79.1) = 84.73 %; g(s) = -P1(s)
    out = K.share_one_kill({"1": -2.6131, "2-3": -0.5501, "4+": 0.0791}, {"1": 1000, "2-3": 1000, "4+": 1000})
    assert out["defined"] is True
    assert out["gain"] == pytest.approx({"1": 2.6131, "2-3": 0.5501, "4+": -0.0791})
    assert out["value"] == pytest.approx(2613.1 / (2613.1 + 550.1 - 79.1), rel=1e-12)
    assert out["denominator"] == pytest.approx(2613.1 + 550.1 - 79.1)


def test_share_one_kill_overall_gain_gives_a_positive_denominator():
    out = K.share_one_kill({"1": -0.02, "2-3": -0.01, "4+": -0.001}, {"1": 600, "2-3": 300, "4+": 100})
    assert out["defined"] is True and out["denominator"] == pytest.approx(0.02 * 600 + 0.01 * 300 + 0.001 * 100)
    assert out["value"] == pytest.approx(12.0 / 15.1)


@pytest.mark.parametrize("points", [{"1": 0.01, "2-3": 0.02, "4+": -0.001},      # every stratum but 4+ loses
                                    {"1": 0.0, "2-3": 0.0, "4+": 0.0},           # no gain anywhere
                                    {"1": -0.01, "2-3": 0.0, "4+": 0.1},         # gains cancel: denominator exactly 0
                                    {"1": -0.01, "2-3": 0.0, "4+": 0.2}])        # net loss
def test_share_one_kill_is_undefined_when_the_gain_denominator_is_not_positive(points):
    out = K.share_one_kill(points, {"1": 100, "2-3": 50, "4+": 10})
    assert out["defined"] is False and out["value"] is None
    assert out["denominator"] <= 0


# ----------------------------------------------------------------------------- K1 strata on eligible rows (plan 2.2)

def test_k1_strata_split_the_eligible_rows_and_count_ties_apart():
    match, y, preds, rng = synthetic(1, matches=90)
    n_kills = rng.integers(1, 7, size=y.size).astype(np.int16)
    ties = np.array([2, 2, 3, 4, 5, 6, 2], dtype=np.int16)      # kill counts of the split's tie rows
    out = K.k1_kill_count(y, preds, match, n_kills, np.concatenate([n_kills, ties]))
    strata = out["strata"]
    assert sum(v["rows"] for v in strata.values()) == y.size
    assert {s: v["ties"] for s, v in strata.items()} == {"1": 0, "2-3": 4, "4+": 3}
    mask = qeval.kill_count_strata(n_kills) == "4+"                 # scale_strata.py:71-80 inside the stratum
    losses = {k: qeval.per_match_loss(y[mask], preds[k][mask], match[mask]) for k in (K.P1, K.PT)}
    ids, aligned = qeval.align_cells(losses)
    boot = qeval.paired_match_bootstrap(ids, aligned, {"P1": qeval.difference(K.P1, K.PT)}, levels=0.95)
    assert strata["4+"]["contrasts"]["P1"]["point"] == boot["contrasts"]["P1"]["point"]
    assert strata["4+"]["contrasts"]["P1"]["interval"] == boot["contrasts"]["P1"]["interval"]
    assert strata["4+"]["matches"] == ids.size
    ev = qeval.evaluate(y[mask], preds[K.PT][mask], match[mask])["match_weighted"]
    assert strata["4+"]["brier"][K.PT] == ev["brier"] and strata["4+"]["auc"][K.PT] == ev["auc"]
    share = out["share_one_kill"]
    assert share["numerator"] == pytest.approx(-strata["1"]["contrasts"]["P1"]["point"] * strata["1"]["rows"])


def test_k1_refuses_kill_counts_below_one():
    match, y, preds, _ = synthetic(2)
    kills = np.ones(y.size, dtype=np.int16)
    kills[3] = 0
    with pytest.raises(K.Blocked):
        K.k1_kill_count(y, preds, match, kills, kills)


# ----------------------------------------------------------------------------- K2 cohorts (plan 2.3)

def test_k2_cohorts_split_the_rows_and_the_all_cohort_p1_equals_the_gate():
    match, y, preds, rng = synthetic(3, matches=120)
    n = y.size
    cohort = rng.integers(0, 4, size=n).astype(np.uint8)
    t5 = (cohort == 0) & (rng.random(n) < 0.5)
    n_blue, n_red = rng.integers(0, 6, n).astype(np.uint8), rng.integers(0, 6, n).astype(np.uint8)
    kills = rng.integers(1, 7, n).astype(np.int16)
    out = K.k2_cohorts(y, preds, match, cohort, t5, n_blue, n_red, kills)
    c = out["cohorts"]
    assert c["all"]["rows"] == n and c["all"]["row_share"] == 1.0
    assert sum(c[name]["rows"] for name in ("T", "S", "ASYM", "P")) == n
    assert c["T5"]["rows"] == int(t5.sum())
    gate = gate_p1(y, preds, match)
    assert c["all"]["contrasts"]["P1"]["point"] == pytest.approx(gate["point"], abs=1e-12)
    assert c["all"]["contrasts"]["P1"]["interval"] == pytest.approx(gate["interval"], abs=1e-12)
    mask = cohort == 1                                                  # S: AUC and slope inside the cohort
    assert c["S"]["auc"][K.P1] == qeval.auc(y[mask], preds[K.P1][mask], match_ids=match[mask])
    cal = qeval.calibration_slope_intercept(y[mask], preds[K.PT][mask], match_ids=match[mask])
    assert c["S"]["calibration_slope"][K.PT] == cal.get("slope")
    for name in ("T", "S", "ASYM", "P", "T5"):
        assert sum(v["rows"] for v in out["crosstab"]["kill_count"][name].values()) == c[name]["rows"]
    assert sum(out["distributions"]["min_side"].values()) == n
    assert "headcount" not in out["crosstab"]                          # plan 2.3: the pre-engagement cross-tab is dropped


# ----------------------------------------------------------------------------- K3 frame age (plan 2.4)

def _frame_age_case(seed: int = 5):
    rng = np.random.default_rng(seed)
    per = np.where(np.arange(60) % 2 == 0, 1, 8)                       # very unequal matches, one 1-row and one 8-row
    match = np.repeat(np.arange(60) * 5 + 1, per)                       # match per bin pair, whole matches in one bin
    order = np.repeat(np.arange(60), per)
    n = match.size
    age = (order // 2) % 10 + 0.5
    age[0], age[1] = 0.0, 10.0                                          # 10 s falls in [9,10]
    y = np.tile([0.0, 1.0], n // 2 + 1)[:n]
    rng.shuffle(y)
    minutes = rng.uniform(2, 40, n) + 10 * (per[order] > 1)
    p_pre = rng.uniform(0.05, 0.95, n)
    kills = rng.integers(1, 7, n).astype(np.int16)
    preds = {key: _sigmoid((0.3 + 0.1 * i) * (2 * y - 1) + rng.normal(0, 1, n)) for i, key in enumerate(K.KA_CELLS)}
    return match, y, preds, age, minutes, p_pre, kills


def test_k3_balance_means_are_unweighted_row_means_and_ten_seconds_close_the_last_bin():
    match, y, preds, age, minutes, p_pre, kills = _frame_age_case()
    bins = np.minimum(np.floor(age).astype(int), 9)
    for b in range(10):
        assert len(set(y[bins == b].tolist())) == 2                     # precondition: both classes in each bin
    out = K.k3_frame_age(y, preds, match, age, minutes, p_pre, kills, resamples=40)
    assert [e["bin"] for e in out["bins"]] == [f"[{b},{b + 1})" for b in range(9)] + ["[9,10]"]
    discriminating = False
    for b, entry in enumerate(out["bins"]):
        mask = bins == b
        assert entry["rows"] == int(mask.sum()) and entry["matches"] == np.unique(match[mask]).size
        assert entry["mean_game_minute"] == pytest.approx(minutes[mask].mean(), abs=1e-12)
        assert entry["mean_abs_p_pre_dev"] == pytest.approx(np.abs(p_pre[mask] - 0.5).mean(), abs=1e-12)
        assert entry["mean_kills"] == pytest.approx(kills[mask].mean(), abs=1e-12)
        weighted = np.average(minutes[mask], weights=qeval.match_weights(match[mask]))
        discriminating |= abs(weighted - minutes[mask].mean()) > 1e-6
    assert discriminating                                               # match weighting would change the means
    assert sum(e["rows"] for e in out["bins"]) == y.size
    assert set(out["trends"]) == {"auc:lightgbm", "auc:pt_flex", "contrast:P1", "contrast:S_effect_LGBM_meanD"}
    mid = np.arange(10) + 0.5
    weights = np.array([e["matches"] for e in out["bins"]], dtype=np.float64)
    points = np.array([e["metrics"]["lightgbm"]["auc"] for e in out["bins"]])
    a, b = FA.wls_line(mid, points, weights)
    assert out["trends"]["auc:lightgbm"]["slope_per_s"] == pytest.approx(b) and out["trends"]["auc:lightgbm"]["intercept_0s"] == pytest.approx(a)
    s_points = np.array([e["contrasts"]["S_effect_LGBM_meanD"]["point"] for e in out["bins"]])
    assert out["trends"]["contrast:S_effect_LGBM_meanD"]["slope_per_s"] == pytest.approx(FA.wls_line(mid, s_points, weights)[1])
    assert out["overall_auc"]["lightgbm"] == qeval.evaluate(y, preds[K.P1], match)["match_weighted"]["auc"]
    assert "p1_best" not in out["overall_auc"]                          # no P1-best cell on KA


def test_k3_refuses_frame_age_outside_zero_to_ten():
    match, y, preds, age, minutes, p_pre, kills = _frame_age_case()
    age = age.copy()
    age[7] = 10.25
    with pytest.raises(K.Blocked):
        K.k3_frame_age(y, preds, match, age, minutes, p_pre, kills, resamples=5)


# ----------------------------------------------------------------------------- K4 constant cell (plan 2.5)

def test_constant_cell_rate_brier_and_skill():
    rng = np.random.default_rng(6)
    y_train, w_train = rng.integers(0, 2, 40).astype(np.float64), rng.uniform(0.1, 1.0, 40)
    rate = K.constant_rate(y_train, w_train)
    assert rate == pytest.approx((w_train * y_train).sum() / w_train.sum(), abs=1e-15)
    match, y, preds, _ = synthetic(6)
    out = K.k4_brier_skill(y, preds, match, rate)
    p_const = qtest.sigmoid(np.full(y.size, np.log(rate / (1.0 - rate))))       # qtest.py:230-232, :264-265
    assert out["brier"]["CONSTANT"] == pytest.approx(qeval.brier(y, p_const, match_ids=match), abs=1e-15)
    b1, bpt, bc = out["brier"][K.P1], out["brier"][K.PT], out["brier"]["CONSTANT"]
    assert b1 == qeval.evaluate(y, preds[K.P1], match)["match_weighted"]["brier"]
    assert out["brier_reduction"] == pytest.approx(1 - b1 / bpt)
    assert out["skill_lgbm"] == pytest.approx(1 - b1 / bc) and out["skill_pt"] == pytest.approx(1 - bpt / bc)
    assert out["constant_logit"] == pytest.approx(np.log(rate / (1.0 - rate)))


# ----------------------------------------------------------------------------- K5 and K6 (plan 3.2, 3.3)

def test_k5_p1_equals_the_gate_and_the_kept_share_is_a_point():
    match, y, preds, rng = synthetic(4, matches=120)
    kills = rng.integers(1, 7, y.size).astype(np.int16)
    out = K.k5_nest(y, preds, match, kills)
    gate = gate_p1(y, preds, match)
    assert out["contrasts"]["P1"]["point"] == pytest.approx(gate["point"], abs=1e-12)
    assert out["contrasts"]["P1"]["interval"] == pytest.approx(gate["interval"], abs=1e-12)
    keep = out["kept_share"]
    assert keep["value"] == pytest.approx(out["contrasts"]["NEST_minus_PT"]["point"] / out["contrasts"]["P1"]["point"])
    assert "interval" not in keep
    assert set(out["strata"]["kill_count"]) == set(qeval.kill_count_strata(kills).tolist())
    assert set(out["contrasts"]) == {"P1", "NEST_minus_PT", "NEST_minus_P1"}


def test_k6_seeds_summary_and_the_loss_average():
    match, y, preds, _ = synthetic(8, matches=90)
    by_seed = {20261001: preds[K.P1], 20261002: preds["P1_s20261002"], 20261003: preds["P1_s20261003"]}
    out = K.k6_seeds(y, by_seed, preds[K.PT], match)
    points = [out["per_seed"][str(s)]["contrasts"]["P1"]["point"] for s in by_seed]
    assert out["summary"]["points"] == points
    assert out["summary"]["sd"] == pytest.approx(np.std(points, ddof=1))
    upper = [out["per_seed"][str(s)]["contrasts"]["P1"]["interval"][1] for s in by_seed]
    assert out["summary"]["all_intervals_below_zero"] == all(u < 0 for u in upper)
    ids, pt_loss = qeval.per_match_loss(y, preds[K.PT], match)
    mean_loss = np.mean([qeval.per_match_loss(y, p, match)[1] for p in by_seed.values()], axis=0)
    assert out["loss_average"]["point"] == pytest.approx(mean_loss.mean() - pt_loss.mean(), abs=1e-15)
    assert out["per_seed"]["20261001"]["contrasts"]["P1"]["point"] == pytest.approx(gate_p1(y, preds, match)["point"], abs=1e-15)


# ----------------------------------------------------------------------------- gate (plan 2.1)

def test_gate_tolerance_is_one_e_minus_nine():
    match, y, preds, _ = synthetic(7, matches=80)
    reference = json.loads(json.dumps(K.jsonable(KL._contrast_block(y, {k: preds[k] for k in K.KA_CELLS}, match))))
    assert K.run_gate(y, preds, match, reference)["passed"] is True
    close = copy.deepcopy(reference)
    close["metrics"][K.P1]["auc"] += 5e-10
    assert K.run_gate(y, preds, match, close)["passed"] is True
    far = copy.deepcopy(reference)
    far["contrasts"]["contrasts"]["S_effect_D1"]["interval"][1] += 2e-9
    gate = K.run_gate(y, preds, match, far)
    assert gate["passed"] is False and gate["max_abs_diff"] == pytest.approx(2e-9, rel=1e-3)
    assert gate["tolerance"] == 1e-9


# ----------------------------------------------------------------------------- guards: SEALED, write once

def test_paths_with_the_seal_word_are_refused_before_any_open(tmp_path):
    assert "SEALED" not in str(tmp_path).upper()                         # only the file name below carries the word
    sealed = tmp_path / "Q_TEST_SEALED_h90.npz"
    sealed.write_bytes(b"not a label file")
    for function in (K.sha256_file, K.read_json, K.load_npz):
        with pytest.raises(K.SealedPathError):
            function(sealed)
    with pytest.raises(K.SealedPathError):
        K.write_json_once(tmp_path / "x_sealed_dir" / "a.json", {})
    with pytest.raises(K.SealedPathError):
        K.files_record({"x": sealed})


def test_outputs_are_written_once_with_lf_only(tmp_path):
    target = tmp_path / "a.json"
    K.write_json_once(target, {"x": 1, "path": "D:\\x\\y", "nan": float("nan")})
    with pytest.raises(FileExistsError):
        K.write_json_once(target, {"x": 2})
    raw = target.read_bytes()
    assert b"\r" not in raw and b"\t" not in raw
    assert json.loads(raw) == {"x": 1, "path": "D:\\x\\y", "nan": None}
    sha = K.write_npz_once(tmp_path / "b.npz", {"a": np.arange(3)})
    assert sha == hashlib.sha256((tmp_path / "b.npz").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        K.write_npz_once(tmp_path / "b.npz", {"a": np.arange(4)})
    K.write_pickle_once(tmp_path / "m.pkl", {"m": 1})
    with pytest.raises(FileExistsError):
        K.write_pickle_once(tmp_path / "m.pkl", {"m": 2})


def test_interrupted_outputs_are_moved_aside_not_overwritten(tmp_path):
    out = tmp_path / "cell"
    out.mkdir()
    for name in ("model_refit.pkl", "pred_q_dev.npz", "FIT_DONE.json", "worker.log", ".pred_q_cal.npz.12.pending"):
        (out / name).write_bytes(name.encode())
    moved = K.move_aside_interrupted(out)
    assert sorted(moved["files"]) == [".pred_q_cal.npz.12.pending", "model_refit.pkl", "pred_q_dev.npz"]
    assert (out / "FIT_DONE.json").is_file() and (out / "worker.log").is_file()
    assert (out / "interrupted_1" / "model_refit.pkl").read_bytes() == b"model_refit.pkl"
    assert K.move_aside_interrupted(out) == {"directory": None, "files": []}


# ----------------------------------------------------------------------------- REFIT_INPUTS recheck (plan 4 step 3)

def test_refit_inputs_recheck_finds_changed_missing_and_extra_files(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "a.bin").write_bytes(b"1")
    (store / "b.bin").write_bytes(b"2")
    other = tmp_path / "x.json"
    other.write_text("{}", encoding="utf-8")
    files = {"x": other, "store:a.bin": store / "a.bin", "store:b.bin": store / "b.bin"}
    doc = {"files": K.files_record(files), "store": {"dir": str(store), "names": ["a.bin", "b.bin"]}}
    assert K.verify_refit_inputs(doc) == {}
    other.write_text("{ }", encoding="utf-8")
    assert set(K.verify_refit_inputs(doc)["changed"]) == {"x"}
    other.write_text("{}", encoding="utf-8")
    (store / "c.bin").write_bytes(b"3")
    assert "store_files_differ" in K.verify_refit_inputs(doc)
    (store / "c.bin").unlink()
    (store / "b.bin").unlink()
    problems = K.verify_refit_inputs(doc)
    assert problems["changed"]["store:b.bin"]["now"] is None and "store_files_differ" in problems


# ----------------------------------------------------------------------------- fit cores (plan 3)

def _table(seed: int, n: int = 2400, d: int = 12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logit = X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] * X[:, 3]
    y = (rng.random(n) < _sigmoid(logit)).astype(np.float64)
    return sparse.csr_matrix(X.astype(np.float32)), y, X


def _source(x, y, rows, size=500):
    # weights 1/m as in the q sources; an all-ones vector would be dropped by LightGBM (Dataset.set_weight) and the
    # frozen STOP metric would see no weights
    w = 1.0 / (1 + (np.asarray(rows) % 4))

    def make():
        for start in range(0, len(rows), size):
            part = rows[start:start + size]
            yield qtrain.Batch(x[part], y[part], w[start:start + len(part)], w[start:start + len(part)], part)
    return make


def test_seeded_fit_changes_only_the_lightgbm_random_state(tmp_path):
    x, y, _ = _table(12)
    rows = np.arange(y.size)
    sources = {"fit": _source(x, y, rows[:1400]), "stop": _source(x, y, rows[1400:1800]), "refit": _source(x, y, rows[:1800]),
               "q_dev": _source(x, y, rows[1800:2200]), "q_cal": _source(x, y, rows[2200:])}
    card = {"num_leaves": 7, "min_data_in_leaf": 40}
    before = qtrain.SEED
    results = {name: K.seeded_fit_lightgbm(card, sources, tmp_path / name, {"cell": name}, seed=seed)
               for name, seed in (("a", 101), ("b", 101), ("c", 202))}
    assert qtrain.SEED == before                                        # the worker's global is restored
    logits = {name: np.load(tmp_path / name / "pred_q_dev.npz")["logit"] for name in results}
    assert np.array_equal(logits["a"], logits["b"]) and not np.array_equal(logits["a"], logits["c"])
    with (tmp_path / "a" / "model_refit.pkl").open("rb") as stream:
        assert pickle.load(stream).random_state == 101
    assert json.loads((tmp_path / "a" / "FIT_DONE.json").read_text(encoding="utf-8"))["identity"] == {"cell": "a"}


def test_nest_fit_includes_the_offsets_and_checks_the_stop_metric():
    x, y, X = _table(13)
    offset = 0.9 * X[:, 0]
    fit, stop, refit = np.arange(0, 1400), np.arange(1400, 1800), np.arange(0, 1800)
    w = 1.0 / (1 + (np.arange(y.size) % 4))                             # 1/m-like, not all ones (see _source)
    card = {"num_leaves": 7, "min_data_in_leaf": 40}
    parts = [(x[r], y[r], w[r], offset[r]) for r in (fit, stop, refit)]
    model, info = K.nest_fit_and_refit(card, *parts, seed=7)
    assert info["stop_metric_abs_diff"] <= 1e-9
    assert model.estimator.n_estimators == info["best_iteration"] >= 1
    again, info2 = K.nest_fit_and_refit(card, *parts, seed=7)
    q = np.arange(1800, 2400)
    assert np.array_equal(model.estimator.predict(x[q], raw_score=True), again.estimator.predict(x[q], raw_score=True))
    zero = [(a, b, c, np.zeros_like(d)) for a, b, c, d in parts]
    plain, _ = K.nest_fit_and_refit(card, *zero, seed=7)
    assert not np.allclose(plain.estimator.predict(x[q], raw_score=True), model.estimator.predict(x[q], raw_score=True))


# ----------------------------------------------------------------------------- plan hash (plan 4 step 5)

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout.decode("ascii", "replace").strip()


def make_repo(tmp_path: Path, text: bytes = b"plan v1\nline two\n") -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "PLAN.md").write_bytes(text)
    _git(repo, "add", "PLAN.md")
    _git(repo, "commit", "-q", "-m", "plan")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_plan_hash_uses_the_committed_blob_bytes_not_the_working_copy(tmp_path):
    repo, commit = make_repo(tmp_path, b"line one\nline two\n")
    (repo / "PLAN.md").write_bytes(b"line one\r\nline two\r\n")
    sha, size = K.plan_blob_sha256(repo, commit, "PLAN.md")
    assert sha == hashlib.sha256(b"line one\nline two\n").hexdigest() and size == 18


# ----------------------------------------------------------------------------- the synthetic tree for the stages

ROLE_MATCHES = ((0, 40), (3, 70), (4, 8), (5, 70))      # TRAIN, Q_DEV, Q_CAL, TEST


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return _sha(path)


def _write(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(K.jsonable(payload), indent=1), encoding="utf-8")
    return _sha(path)


def build_fixture(tmp_path: Path, seed: int = 21, broken: str | None = None) -> K.Paths:
    """``broken`` plants one label-free TEST input fault that the lock binds as it is: 'pred_rows' (P1_NEST TEST rows in
    another order), 'frame_age' (an eligible TEST row 10.5 s old), 'kill_count' (a TEST tie row with 0 kills),
    'pred_shape' (P1_NEST TEST: one row id more than probabilities) or 'ref_interval' (a KA_REPORT TEST interval
    ['bad', None])."""
    rng = np.random.default_rng(seed)
    q_run, q_ka, out, review = tmp_path / "q_run_1", tmp_path / "q_ka_1", tmp_path / "tog_ka_1", tmp_path / "review"
    roles, matches, mid = [], [], 0
    for role, count in ROLE_MATCHES:
        for _ in range(count):
            m = int(rng.integers(1, 7))
            roles += [role] * m
            matches += [mid] * m
            mid += 1
    role, match = np.array(roles, dtype=np.int8), np.array(matches, dtype=np.int64)
    n = role.size
    keys = np.array([f"eng{i:05d}".encode() for i in range(n)], dtype="S16")
    ka = rng.integers(-3, 4, size=n)
    ka[rng.random(n) < 0.12] = 0
    ka[(ka == 0) & (rng.random(n) < 0.5)] = 1
    eligible = ka != 0
    y_ka = (ka > 0).astype(np.float64)
    per = np.bincount(match[eligible], minlength=mid).astype(np.float64)
    weight = np.where(eligible, 1.0 / np.maximum(per[match], 1.0), 0.0)
    kills = (np.maximum(np.abs(ka), 1) + rng.integers(0, 3, n)).astype(np.int16)
    kills[~eligible] = np.maximum(kills[~eligible], 2)
    age = (np.arange(n) % 10) + rng.uniform(0.0, 0.999, n)
    age[:2] = (0.0, 10.0)
    test_eligible, test_ties = np.flatnonzero((role == 5) & eligible), np.flatnonzero((role == 5) & ~eligible)
    if broken == "frame_age":
        age[test_eligible[0]] = 10.5
    if broken == "kill_count":
        kills[test_ties[0]] = 0
    rows_sha = _save_npz(q_run / "Q_ROWS_h90.npz", engagement_key=keys, match_index=match, role=role, eval_n_kills=kills,
                         frame_age_s=age, elapsed_min=rng.uniform(2, 40, n), pt_p_pre=rng.uniform(0.05, 0.95, n),
                         y=np.where(role == 5, -1, rng.integers(0, 2, n)).astype(np.int8))
    _write(q_run / "Q_ROWS_h90.json", {"rows_sha256": rows_sha})
    labels_sha = _save_npz(q_ka / "KA_LABELS.npz", engagement_key=keys, ka=ka.astype(np.int32), eligible=eligible, y_ka=y_ka,
                           weight_raw_ka=weight)
    per_role = {name: {"rows": int((role == code).sum()), "ties": int(((role == code) & ~eligible).sum())}
                for name, code in (("TRAIN", 0), ("Q_DEV", 3), ("Q_CAL", 4), ("TEST", 5))}
    _write(q_ka / "KA_PREPARE.json", {"ka_labels_sha256": labels_sha, "rows_sha256": rows_sha, "per_role": per_role})
    cohort = rng.integers(0, 4, n).astype(np.uint8)
    part_sha = _save_npz(review / "PARTICIPANTS_KILLCREDIT.npz", engagement_key=keys, n_blue=rng.integers(0, 6, n).astype(np.uint8),
                         n_red=rng.integers(0, 6, n).astype(np.uint8), cohort=cohort, t5=(cohort == 0) & (rng.random(n) < 0.6))
    _write(review / "PARTICIPANTS_KILLCREDIT.json", {"npz_sha256": part_sha, "definition": "synthetic"})
    report = {"splits": {}}
    split_preds = {}
    cells = [*K.KA_CELLS, *NEW_PRED_CELLS, K.CHECK]
    for split, code in (("q_dev", 3), ("test", 5), ("q_cal", 4)):
        idx = np.flatnonzero((role == code) & eligible)
        yy = y_ka[idx]
        split_preds[split] = {}
        for i, key in enumerate(cells):
            logit = (0.5 + 0.12 * (i % 5)) * (2 * yy - 1) + rng.normal(0.0, 1.0, idx.size)
            if key == K.CHECK:
                logit = split_preds[split][K.P1][0]
            split_preds[split][key] = (logit, _sigmoid(logit))
        if split != "q_cal":
            block = KL._contrast_block(yy, {k: split_preds[split][k][1] for k in K.KA_CELLS}, match[idx])
            report["splits"][split] = {"ka_trained_on_ka": block}
    if broken == "ref_interval":                                      # Astra's counterexample: ['bad', None]
        report["splits"]["test"]["ka_trained_on_ka"]["contrasts"]["contrasts"]["S_effect_D1"]["interval"] = ["bad", None]
    _write(q_ka / "KA_REPORT.json", report)
    code_sha = _sha(K.ROOT / K.CODE_REL)
    for key in cells:
        new = key not in K.KA_CELLS
        directory = (out / "cells" / key) if new else (q_ka / "cells" / key)
        shas = {}
        for split, name in (("q_dev", "pred_q_dev.npz"), ("q_cal", "pred_q_cal.npz"), ("test", "pred_test.npz")):
            idx = np.flatnonzero((role == {"q_dev": 3, "q_cal": 4, "test": 5}[split]) & eligible)
            logit, p = split_preds[split][key]
            if broken == "pred_rows" and key == K.NEST and split == "test":
                swap = np.r_[1, 0, np.arange(2, idx.size)]
                idx, logit, p = idx[swap], logit[swap], p[swap]
            if broken == "pred_shape" and key == K.NEST and split == "test":
                logit, p = logit[:-1], p[:-1]                          # Astra's counterexample: one id more than probabilities
            shas[split] = _save_npz(directory / name, row=idx, logit=logit, p=p)
        with (directory / "model_refit.pkl").open("wb") as stream:
            pickle.dump({"synthetic": key}, stream)
        result = {"status": "completed", "scores": {"q_dev": {"pred_sha256": shas["q_dev"]}, "q_cal": {"pred_sha256": shas["q_cal"]}},
                  "pred_test_sha256": shas["test"]}
        if new:
            result.update(cell=key, model_sha256=_sha(directory / "model_refit.pkl"), loaded_modules={K.CODE_REL: code_sha},
                          best_iteration=10, seconds=1.0)
            if key != K.NEST:
                _write(directory / "FIT_DONE.json", {"identity": {"cell": key}, "best_iteration": 10})
            if key == K.CHECK:
                result["reproduction"] = {"passed": True, "max_abs_qdev_logit_diff": 0.0, "tolerance": 1e-7}
        _write(directory / "RESULT.json", result)
    store = tmp_path / "store"
    store.mkdir()
    (store / "BUILD.json").write_text("{}", encoding="utf-8")
    (store / "numeric_global.f64.npy").write_bytes(b"\x00" * 64)
    refit_files = {"q_run_1:Q_ROWS_h90.npz": q_run / "Q_ROWS_h90.npz", "q_ka_1:KA_LABELS.npz": q_ka / "KA_LABELS.npz",
                   "q_run_1:raw/store/BUILD.json": store / "BUILD.json",
                   "q_run_1:raw/store/numeric_global.f64.npy": store / "numeric_global.f64.npy"}
    _write(out / "REFIT_INPUTS.json", {"files": K.files_record(refit_files), "store": {"dir": str(store), "names": sorted(p.name for p in store.iterdir())},
                                       "loaded_modules": {K.CODE_REL: code_sha}})
    repo, commit = make_repo(tmp_path)
    exposure = tmp_path / "EXPOSURE_KO.md"
    exposure.write_text("# exposure\n", encoding="utf-8")
    openings = tmp_path / "OPENINGS.json"
    openings.write_text(json.dumps({"rule": "synthetic", "openings": [{"experiment": "E2", "utc": "2026-10-12T00:00:00+00:00"}]}),
                        encoding="utf-8")
    return K.Paths(out=out, q_run=q_run, q_ka=q_ka, participants=review / "PARTICIPANTS_KILLCREDIT.npz", openings=openings,
                   exposure=exposure, repo=repo, plan_rel="PLAN.md", plan_commit=commit)


def expose(paths: K.Paths, lock_path: Path) -> None:
    """The planned repeated-look row with the lock SHA-256 (plan 4 step 6)."""
    with paths.exposure.open("a", encoding="utf-8") as stream:
        stream.write(f"| planned repeated look | `{K.sha256_file(lock_path)}` |\n")


def ka_openings(paths: K.Paths) -> list:
    return [o for o in json.loads(paths.openings.read_text(encoding="utf-8"))["openings"] if o["experiment"] == K.EXPERIMENT]


@pytest.fixture()
def tree(tmp_path):
    return build_fixture(tmp_path)


# ----------------------------------------------------------------------------- stage: qdev

def test_qdev_stage_writes_once_on_eligible_rows_only(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    with pytest.raises(FileExistsError):
        K.qdev_stage(tree, time_limit_s=None)
    doc = json.loads(tree.qdev.read_text(encoding="utf-8"))
    labels = np.load(tree.ka_labels)
    rows = np.load(tree.rows)
    split = rows["role"] == 3
    eligible = split & labels["eligible"]
    assert doc["rows"] == int(eligible.sum()) and doc["ties"] == int((split & ~labels["eligible"]).sum())
    assert doc["gate"]["passed"] is True and doc["gate"]["max_abs_diff"] <= 1e-9
    assert sum(v["rows"] for v in doc["K1"]["strata"].values()) == doc["rows"]
    assert sum(v["ties"] for v in doc["K1"]["strata"].values()) == doc["ties"]
    assert doc["K2"]["cohorts"]["all"]["rows"] == doc["rows"]
    assert sum(b["rows"] for b in doc["K3"]["bins"]) == doc["rows"]
    assert all(check["passed"] for check in doc["consistency"]["checks"].values())
    train = (rows["role"] == 0) & labels["eligible"]
    rate = (labels["weight_raw_ka"][train] * labels["y_ka"][train]).sum() / labels["weight_raw_ka"][train].sum()
    assert doc["K4"]["rate"] == pytest.approx(rate, abs=1e-15) and doc["K4_train"]["rows"] == int(train.sum())
    assert doc["run"]["seed"] == 20261011 and doc["run"]["rows"] == doc["rows"] and doc["run"]["input_sha256"]
    assert doc["stage"]["peak_memory_bytes"] is None or doc["stage"]["peak_memory_bytes"] > 0
    assert K.CODE_REL in doc["loaded_modules"]
    raw = tree.qdev.read_bytes()
    assert b"\r" not in raw and b"\t" not in raw


def test_qdev_stage_stops_before_new_numbers_when_the_gate_fails(tree):
    report = json.loads(tree.ka_report.read_text(encoding="utf-8"))
    report["splits"]["q_dev"]["ka_trained_on_ka"]["metrics"][K.PT]["brier"] += 2e-9
    tree.ka_report.write_text(json.dumps(report), encoding="utf-8")
    assert K.qdev_stage(tree, time_limit_s=None) == 2
    assert not tree.qdev.exists()
    blocked = list(tree.out.glob("QDEV_BLOCKED_*.json"))
    assert len(blocked) == 1
    record = json.loads(blocked[0].read_text(encoding="utf-8"))
    assert record["gate"]["passed"] is False and "K1" not in record


def test_qdev_stage_needs_the_reproduction_check(tree):
    path = tree.cell(K.CHECK) / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["reproduction"]["passed"] = False
    path.write_text(json.dumps(result), encoding="utf-8")
    assert K.qdev_stage(tree, time_limit_s=None) == 2
    assert not tree.qdev.exists()


# ----------------------------------------------------------------------------- stage: lock

def test_lock_contents(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    assert K.lock_stage(tree, require_committed_code=False) == 0
    lock_path = tree.out / "KA_POSTHOC_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    plan = lock["plan"]
    assert plan["commit"] == tree.plan_commit and plan["path"] == "PLAN.md"
    assert plan["blob_sha256"] == hashlib.sha256(b"plan v1\nline two\n").hexdigest()
    code = lock["code"]["files"]
    for rel in (K.CODE_REL, K.TEST_REL, "reports/q_production_20261005/kill_label.py", "reports/review_followup_20261011/r2_retrain.py",
                "reports/review_followup_20261011/rcommon.py", "reports/thesis_followup_20261010/common.py",
                "reports/v_report_20261005/v_report.py", "studies/tog_experiments_20261012/common.py",
                "studies/q_production_20261005/qeval.py"):
        assert code[rel] == _sha(K.ROOT / rel), rel
    assert set(lock["code"]["stages"]) == {"inputs", "qdev", "lock", *(f"fit:{c}" for c in K.FIT_ORDER)}
    assert "head" in lock["code"]["git"] and "status_porcelain" in lock["code"]["git"]
    binding = lock["code"]["binding_check"]                              # Ruling 67: behaviour, not the AST record, binds
    assert "Q_DEV" in binding and "exact" in binding and "secondary" in lock["code"]["revisions_rule"]
    assert lock["refit_inputs"]["sha256"] == _sha(tree.refit_inputs) and lock["refit_inputs"]["recheck"] == {}
    verify = lock["verify"]
    for label in ("input:KA_LABELS.npz", "input:KA_PREPARE.json", "input:Q_ROWS_h90.npz", "input:Q_ROWS_h90.json", "input:KA_REPORT.json",
                  "input:PARTICIPANTS_KILLCREDIT.npz", "input:PARTICIPANTS_KILLCREDIT.json", "refit_inputs", "qdev"):
        assert label in verify, label
    for key in K.KA_CELLS:
        for name in ("pred_q_dev.npz", "pred_test.npz", "RESULT.json", "model_refit.pkl"):
            assert f"q_ka_1:{key}:{name}" in verify
    for cell in K.FIT_ORDER:
        names = {label.split(":")[-1] for label in verify if label.startswith(f"fit:{cell}:")}
        expected = {"RESULT.json", "model_refit.pkl", "pred_q_dev.npz", "pred_q_cal.npz", "pred_test.npz"}
        assert names == (expected | {"FIT_DONE.json"} if cell != K.NEST else expected), cell
    for entry in verify.values():
        assert K.sha256_file(entry["path"]) == entry["sha256"]
    assert lock["qdev"]["sha256"] == _sha(tree.qdev) and lock["locked_utc"]
    with pytest.raises(FileExistsError):
        K.lock_stage(tree, require_committed_code=False)
    assert K.lock_stage(tree, new_version=True, require_committed_code=False) == 0
    assert (tree.out / "KA_POSTHOC_LOCK_r1.json").is_file() and K.latest_lock(tree.out) == tree.out / "KA_POSTHOC_LOCK_r1.json"


def test_lock_refuses_when_a_refit_input_or_stage_code_changed(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    doc = json.loads(tree.refit_inputs.read_text(encoding="utf-8"))
    store = Path(doc["store"]["dir"])
    (store / "numeric_global.f64.npy").write_bytes(b"\x01" * 64)
    assert K.lock_stage(tree, require_committed_code=False) == 2
    assert not (tree.out / "KA_POSTHOC_LOCK.json").exists()
    (store / "numeric_global.f64.npy").write_bytes(b"\x00" * 64)
    path = tree.cell(K.NEST) / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["loaded_modules"] = {K.CODE_REL: "0" * 64}                    # the fit ran another version of the code
    path.write_text(json.dumps(result), encoding="utf-8")
    assert K.lock_stage(tree, require_committed_code=False) == 2
    assert not (tree.out / "KA_POSTHOC_LOCK.json").exists()


def test_lock_needs_committed_own_code_but_allows_untracked_frozen_modules():
    clean = {"status_returncode": 0, "modified": [], "untracked": ["reports/thesis_followup_20261010/common.py"],
             "last_commit_of_ka_posthoc": "abc"}
    assert K.code_commit_problems(clean) == []
    assert K.code_commit_problems({**clean, "untracked": [K.CODE_REL]}) != []
    assert K.code_commit_problems({**clean, "untracked": [K.TEST_REL]}) != []
    assert K.code_commit_problems({**clean, "modified": [" M studies/q_production_20261005/qeval.py"]}) != []
    assert K.code_commit_problems({**clean, "last_commit_of_ka_posthoc": ""}) != []


def test_commit_charge_is_read_from_the_system():
    assert K.commit_bytes() > 2**30


# ----------------------------------------------------------------------------- stage: test (exposure rules, plan 4 step 7)

def test_lock_check_refuses_modules_outside_the_lock(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    assert K.lock_stage(tree, require_committed_code=False) == 0
    lock_path = tree.out / "KA_POSTHOC_LOCK.json"
    expose(tree, lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sha = K.sha256_file(lock_path)
    assert K.verify_lock(lock, sha, tree) == {}
    del lock["code"]["files"]["reports/review_followup_20261011/rcommon.py"]
    assert "reports/review_followup_20261011/rcommon.py" in K.verify_lock(lock, sha, tree)["loaded_modules_not_in_lock"]

def test_ka_test_labels_need_a_saved_opening(tree):
    rows, _ = K.load_rows_table(tree)
    with pytest.raises(K.LabelAccessError):
        K.load_ka_split(tree, rows, "test")
    entry = {"experiment": K.EXPERIMENT, "attempt": 1, "utc": "2026-10-14T00:00:00+00:00", "lock_sha256": "a" * 64}
    with pytest.raises(K.LabelAccessError):
        K.load_ka_split(tree, rows, "test", opening=entry)              # not on disk yet
    K.append_opening(tree.openings, entry)
    split = K.load_ka_split(tree, rows, "test", opening=entry)
    assert split["y"].size == split["idx"].size > 0
    assert K.load_ka_split(tree, rows, "q_dev")["y"].size > 0         # development rows need no opening


def test_test_stage_saves_the_opening_before_reading_ka_test_labels(tree, monkeypatch):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    assert K.lock_stage(tree, require_committed_code=False) == 0
    lock_path = tree.out / "KA_POSTHOC_LOCK.json"
    expose(tree, lock_path)
    seen = []
    real = K.load_ka_split

    def spy(paths, rows, split, *, opening=None):
        if split == "test":
            seen.append(ka_openings(paths))
        return real(paths, rows, split, opening=opening)

    monkeypatch.setattr(K, "load_ka_split", spy)
    assert K.test_stage(tree, time_limit_s=None) == 0
    assert len(seen) == 1 and len(seen[0]) == 1
    entry = seen[0][0]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert entry["attempt"] == 1 and entry["lock"] == str(lock_path) and entry["lock_sha256"] == _sha(lock_path)
    assert entry["prespec_sha256"] == lock["plan"]["blob_sha256"]
    assert entry["labels"] == [str(tree.ka_labels)] and entry["labels_sha256"] == _sha(tree.ka_labels)
    assert entry["note"].startswith("repeated look at the unsealed KA TEST labels (first evaluated 2026-10-09T11:30:47Z")
    report = json.loads((tree.out / "KA_POSTHOC_TEST.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True and report["opening"] == entry and report["split"] == "TEST"
    with pytest.raises(FileExistsError):
        K.test_stage(tree, time_limit_s=None)


def test_test_stage_failures_before_exposure_leave_no_opening_and_can_rerun(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    assert K.lock_stage(tree, require_committed_code=False) == 0
    lock_path = tree.out / "KA_POSTHOC_LOCK.json"
    assert K.test_stage(tree, time_limit_s=None) == 2                  # no EXPOSURE row with the lock SHA-256 yet
    assert ka_openings(tree) == []
    expose(tree, lock_path)
    original = tree.ka_prepare.read_bytes()
    tree.ka_prepare.write_bytes(original + b" ")                          # a locked file changed after the lock
    assert K.test_stage(tree, time_limit_s=None) == 2
    assert ka_openings(tree) == [] and not (tree.out / "KA_POSTHOC_TEST.json").exists()
    assert len(list(tree.out.glob("TEST_BLOCKED_*.json"))) == 2
    tree.ka_prepare.write_bytes(original)
    assert K.test_stage(tree, time_limit_s=None) == 0
    assert [o["attempt"] for o in ka_openings(tree)] == [1]


def test_after_exposure_a_rerun_needs_a_plan_revision_and_a_new_lock(tree, monkeypatch):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    assert K.lock_stage(tree, require_committed_code=False) == 0
    expose(tree, tree.out / "KA_POSTHOC_LOCK.json")

    real_reference = K.gate_reference

    def broken(paths, split):
        if split == "test":                                             # the TEST reference is read only after the opening
            raise RuntimeError("synthetic failure after the opening")
        return real_reference(paths, split)

    monkeypatch.setattr(K, "gate_reference", broken)
    assert K.test_stage(tree, time_limit_s=None) == 3
    assert [o["attempt"] for o in ka_openings(tree)] == [1]             # the exposed attempt stays recorded
    assert (tree.out / "KA_POSTHOC_TEST_FAILED.json").is_file() and not (tree.out / "KA_POSTHOC_TEST.json").exists()
    monkeypatch.undo()
    assert K.test_stage(tree, time_limit_s=None) == 2                  # same lock and plan: refused before a new opening
    assert [o["attempt"] for o in ka_openings(tree)] == [1]
    (tree.repo / "PLAN.md").write_bytes(b"plan v2 (revision 1, dated, reason, changed sections)\n")
    _git(tree.repo, "commit", "-q", "-am", "plan revision 1")
    revised = dataclasses.replace(tree, plan_commit=_git(tree.repo, "rev-parse", "HEAD"))
    assert K.lock_stage(revised, new_version=True, require_committed_code=False) == 0
    expose(revised, revised.out / "KA_POSTHOC_LOCK_r1.json")
    assert K.test_stage(revised, time_limit_s=None) == 0
    assert [o["attempt"] for o in ka_openings(revised)] == [1, 2]
    assert (revised.out / "KA_POSTHOC_TEST_a2.json").is_file()


# ----------------------------------------------------------------------------- fix round 1 (Astra review of part A)

def _locked_and_exposed(paths: K.Paths) -> Path:
    assert K.qdev_stage(paths, time_limit_s=None) == 0
    assert K.lock_stage(paths, require_committed_code=False) == 0
    lock_path = paths.out / "KA_POSTHOC_LOCK.json"
    expose(paths, lock_path)
    return lock_path


@pytest.mark.parametrize("broken", ["pred_rows", "frame_age", "kill_count", "pred_shape", "ref_interval"])
def test_label_free_test_inputs_are_checked_before_the_opening(tmp_path, broken):
    paths = build_fixture(tmp_path, broken=broken)
    _locked_and_exposed(paths)                                          # the lock binds the faulty input as it is
    assert K.test_stage(paths, time_limit_s=None) == 2
    assert ka_openings(paths) == []                                     # no look was spent on an input fault
    assert not (paths.out / "KA_POSTHOC_TEST_FAILED.json").exists() and not (paths.out / "KA_POSTHOC_TEST.json").exists()
    blocked = [json.loads(p.read_text(encoding="utf-8")) for p in paths.out.glob("TEST_BLOCKED_*.json")]
    assert len(blocked) == 1 and "before exposure" in blocked[0]["reason"]


def test_modules_and_inputs_are_checked_before_the_opening_and_labels_after_it(tree, monkeypatch):
    _locked_and_exposed(tree)
    order = []

    def spy(name):
        real = getattr(K, name)

        def wrapped(*args, **kwargs):
            order.append(name)
            return real(*args, **kwargs)
        return wrapped

    for name in ("warm_up_analysis", "verify_lock", "test_preflight", "reproduce_qdev", "append_opening"):
        monkeypatch.setattr(K, name, spy(name))
    real_load = K.load_ka_split

    def load(paths, rows, split, *, opening=None):
        order.append(f"labels:{split}")
        return real_load(paths, rows, split, opening=opening)

    monkeypatch.setattr(K, "load_ka_split", load)
    assert K.test_stage(tree, time_limit_s=None) == 0
    opening = order.index("append_opening")
    assert (order.index("warm_up_analysis") < order.index("verify_lock") < order.index("test_preflight") < order.index("reproduce_qdev")
            < opening)
    assert "labels:test" not in order[:opening]                         # no KA TEST label before the opening
    assert set(order[:opening]) & {"labels:q_dev", "labels:train"}      # the Q_DEV reproduction reads Q_DEV / TRAIN labels only
    assert order.index("labels:test") > opening


def test_a_module_outside_the_lock_blocks_before_exposure(tree, monkeypatch):
    _locked_and_exposed(tree)
    real_modules = K.repo_modules
    monkeypatch.setattr(K, "repo_modules", lambda: {**real_modules(), "studies/late_module.py": "0" * 64})
    assert K.test_stage(tree, time_limit_s=None) == 2
    assert ka_openings(tree) == []


def test_a_module_loaded_after_exposure_stops_the_result(tree, monkeypatch):
    _locked_and_exposed(tree)
    real_reference, real_modules = K.gate_reference, K.repo_modules

    def reference_then_a_late_module(paths, split):
        if split == "test":                                             # the TEST reference is read only after the opening
            monkeypatch.setattr(K, "repo_modules", lambda: {**real_modules(), "studies/late_module.py": "0" * 64})
        return real_reference(paths, split)

    monkeypatch.setattr(K, "gate_reference", reference_then_a_late_module)
    assert K.test_stage(tree, time_limit_s=None) == 3                  # after exposure: recorded failure, no result
    assert not (tree.out / "KA_POSTHOC_TEST.json").exists()
    failure = json.loads((tree.out / "KA_POSTHOC_TEST_FAILED.json").read_text(encoding="utf-8"))
    assert "studies/late_module.py" in failure["reason"] and failure["status"] == "failed_after_exposure"
    assert [o["attempt"] for o in ka_openings(tree)] == [1]


def test_python_definitions_diff_names_changed_added_and_removed():
    old = "import os\nA = 1\n\n\ndef f(x):\n    return x + 1\n\n\ndef g():\n    return 2\n\n\ndef h():\n    pass\n"
    new = "import os\nA = 1\n\n\ndef f(x):\n    # a comment only\n    return x + 1\n\n\ndef g():\n    return 3\n\n\ndef k():\n    pass\n"
    assert K.python_definitions_diff(old, new) == {"module_level_changed": False, "changed": ["g"], "added": ["k"], "removed": ["h"]}
    assert K.python_definitions_diff(old, old.replace("A = 1", "A = 2"))["module_level_changed"] is True


def test_code_revision_accepts_changes_to_the_test_step_only(tmp_path):
    repo = tmp_path / "rev"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "test@example.invalid"), ("config", "user.name", "test"),
                 ("config", "core.autocrlf", "false")):
        _git(repo, *args)
    v1 = b"X = 1\n\n\ndef fit_seed():\n    return 1\n\n\ndef test_stage():\n    return 1\n"
    (repo / "m.py").write_bytes(v1)
    _git(repo, "add", "m.py")
    _git(repo, "commit", "-q", "-m", "v1")
    stage_sha = hashlib.sha256(v1).hexdigest()
    (repo / "m.py").write_bytes(v1.replace(b"def test_stage():\n    return 1", b"def test_stage():\n    return 2") + b"\n\ndef test_preflight():\n    return 0\n")
    ok = K.code_revision(repo, "m.py", stage_sha)
    assert ok["problems"] == [] and ok["confinement_findings"] == [] and ok["stage_commit"] == _git(repo, "rev-parse", "HEAD")
    assert ok["changed"] == ["test_stage"] and ok["added"] == ["test_preflight"]
    # Ruling 67: the AST confinement is a secondary record (findings), not a gate; git provenance still is (problems)
    (repo / "m.py").write_bytes(v1.replace(b"def fit_seed():\n    return 1", b"def fit_seed():\n    return 2"))
    fit_change = K.code_revision(repo, "m.py", stage_sha)
    assert fit_change["problems"] == [] and any("fit_seed" in f for f in fit_change["confinement_findings"])
    (repo / "m.py").write_bytes(v1.replace(b"X = 1", b"X = 2"))
    assert any("module-level" in f for f in K.code_revision(repo, "m.py", stage_sha)["confinement_findings"])
    assert K.code_revision(repo, "m.py", "f" * 64)["problems"]          # the stage's version is not in the history


# ----------------------------------------------------------------------------- fix round 2 (Astra re-review; Ruling 67)

def test_prediction_arrays_need_one_dimension_equal_lengths_dtypes_and_finite_probabilities():
    good = {"row": np.arange(10, dtype=np.int64), "logit": np.zeros(10), "p": np.full(10, 0.5)}
    K.check_prediction_arrays("cell", good)
    bad = {"ten ids, nine probabilities": {**good, "logit": np.zeros(9), "p": np.full(9, 0.5)},   # Astra's counterexample
           "two-dimensional p": {**good, "p": np.full((10, 1), 0.5)},
           "float row ids": {**good, "row": np.arange(10, dtype=np.float64)},
           "integer probabilities": {**good, "p": np.ones(10, dtype=np.int64)},
           "probability above 1": {**good, "p": np.r_[np.full(9, 0.5), 1.5]},
           "NaN logit": {**good, "logit": np.r_[np.zeros(9), np.nan]},
           "missing array": {"row": good["row"], "p": good["p"]}}
    for name, arrays in bad.items():
        with pytest.raises(K.Blocked):
            K.check_prediction_arrays(name, arrays)


def test_reference_numbers_must_be_finite_with_ordered_intervals():
    match, y, preds, _ = synthetic(31, matches=60)
    block = json.loads(json.dumps(K.jsonable(KL._contrast_block(y, {k: preds[k] for k in K.KA_CELLS}, match))))
    assert K.check_reference_block(block) == []
    cases = [(("contrasts", "contrasts", "S_effect_D1", "interval"), ["bad", None]),        # Astra's counterexample
             (("contrasts", "contrasts", "P1", "interval"), [float("nan"), 0.1]),
             (("contrasts", "contrasts", "P1", "interval"), [0.2, 0.1]),
             (("contrasts", "contrasts", "P1", "interval"), [0.1]),
             (("contrasts", "contrasts", "P1", "point"), float("inf")),
             (("contrasts", "contrasts", "S_effect_meanD", "point"), True),
             (("metrics", K.PT, "auc"), "0.7")]
    for path, value in cases:
        broken = copy.deepcopy(block)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert K.check_reference_block(broken), (path, value)


@pytest.mark.parametrize("revision", ["default_argument_global", "helper_body"])
def test_a_revision_that_changes_analysis_output_is_caught_before_exposure(tree, monkeypatch, revision):
    """Astra's counterexamples: an allowed definition whose default argument rebinds a global the analysis reads, and an
    allowed helper body that replaces an analysis routine.  Neither breaks the analysis' internal consistency checks (so
    the synthetic warm-up passes) and the AST record reports only the allowed definition; the Q_DEV reproduction
    (Ruling 67) stops the TEST step before exposure."""
    _locked_and_exposed(tree)
    if revision == "default_argument_global":
        source = (K.ROOT / K.CODE_REL).read_text(encoding="utf-8")
        revised = source.replace("def warm_up_analysis() -> None:", "def warm_up_analysis(_m=globals().__setitem__('S10', S00)) -> None:", 1)
        assert revised != source
        assert K.python_definitions_diff(source, revised) == {"module_level_changed": False, "changed": ["warm_up_analysis"],
                                                               "added": [], "removed": []}
        monkeypatch.setattr(K, "S10", K.S10)                            # undone after the test
        exec("def _revised_default(_m=globals().__setitem__('S10', S00)):\n    pass\n", vars(K))   # what importing it does
        del K._revised_default
        assert K.S10 == K.S00
    else:
        real = FA.wls_line
        monkeypatch.setattr(FA, "wls_line", lambda x, y, w: (lambda a, b: (a, b + 1e-9))(*real(x, y, w)))
    assert K.test_stage(tree, time_limit_s=None) == 2
    assert ka_openings(tree) == []
    blocked = [json.loads(p.read_text(encoding="utf-8")) for p in tree.out.glob("TEST_BLOCKED_*.json")]
    assert len(blocked) == 1 and "Q_DEV" in blocked[0]["reason"]
    assert blocked[0]["reproduction"]["equal"] is False and "K3" in blocked[0]["reproduction"]["differ"]


def test_the_qdev_reproduction_equals_the_stored_results_exactly_and_names_any_difference(tree):
    assert K.qdev_stage(tree, time_limit_s=None) == 0
    rows, _ = K.load_rows_table(tree)
    check = K.reproduce_qdev(tree, rows)
    assert check["equal"] is True and check["differ"] == [] and check["error"] is None
    assert {"gate", "K1", "K2", "K3", "K4", "K5", "K6", "consistency", "K4_train", "rows", "matches", "ties"} <= set(check["compared"])
    assert check["seconds"] >= 0 and check["reference_sha256"] == _sha(tree.qdev)
    stored = json.loads(tree.qdev.read_text(encoding="utf-8"))
    stored["K3"]["trends"]["auc:lightgbm"]["slope_per_s"] += 1e-15             # one stored output off by one ulp-sized step
    other = dataclasses.replace(tree, out=tree.out.parent / "other_out")
    shutil.copytree(tree.out, other.out)
    (other.qdev).unlink()
    (other.qdev).write_text(json.dumps(stored), encoding="utf-8")
    changed = K.reproduce_qdev(other, rows)
    assert changed["equal"] is False and changed["differ"] == ["K3"]


# ----------------------------------------------------------------------------- stage: launch (plan 3.4)

class FakeProc:
    def __init__(self, paths, cell, outcome, log):
        self.paths, self.cell, self.outcome, self.log, self.pid, self.polls, self.killed = paths, cell, outcome, log, 4242, 0, False
        self.code = None

    def poll(self):
        self.polls += 1
        if self.code is not None or self.outcome == "hang" or self.polls < 2:
            return self.code
        if self.outcome == "vanish":
            self.code = 9
            return self.code
        result = {"status": "completed" if self.outcome != "fail" else "failed"}
        if self.cell == K.CHECK:
            result["reproduction"] = {"passed": self.outcome != "bad_repro"}
        K.write_json_once(self.paths.cell(self.cell) / "RESULT.json", result)
        self.code = 0 if self.outcome != "fail" else 1
        return self.code

    def kill(self):
        self.killed = True


def _launch(tree, outcomes, commit=0, clock_step=1.0):
    started, procs = [], {}
    now = [0.0]

    def starter(paths, cell):
        started.append(cell)
        paths.cell(cell).mkdir(parents=True, exist_ok=True)
        procs[cell] = FakeProc(paths, cell, outcomes.get(cell, "ok"), started)
        return procs[cell]

    def clock():
        now[0] += clock_step
        return now[0]

    code = K.launch_stage(tree, poll_s=0.0, starter=starter, commit=lambda: commit, other_launcher=lambda: None, adopt=lambda cell: None,
                          clock=clock, sleep=lambda s: None)
    return code, started, procs


def _fresh(tree):
    for cell in K.FIT_ORDER:
        for path in tree.cell(cell).glob("*"):
            path.unlink()


def test_launch_runs_the_four_fits_one_at_a_time_in_order(tree):
    _fresh(tree)
    code, started, procs = _launch(tree, {})
    assert code == 0 and started == list(K.FIT_ORDER) == [K.CHECK, "P1_s20261002", "P1_s20261003", K.NEST]
    log = tree.launch_log.read_text(encoding="utf-8")
    assert "launch done" in log
    code, started, _ = _launch(tree, {})
    assert code == 0 and started == []                                  # completed cells are not refit


@pytest.mark.parametrize("outcomes, blocked_at", [({K.CHECK: "bad_repro"}, K.CHECK), ({"P1_s20261002": "fail"}, "P1_s20261002"),
                                                  ({K.NEST: "vanish"}, K.NEST)])
def test_launch_stops_at_a_failure(tree, outcomes, blocked_at):
    _fresh(tree)
    code, started, _ = _launch(tree, outcomes)
    assert code == 2 and started[-1] == blocked_at
    assert len(list(tree.out.glob("LAUNCH_BLOCKED_*.json"))) == 1


def test_launch_does_not_start_a_fit_above_the_commit_threshold(tree):
    _fresh(tree)
    code, started, _ = _launch(tree, {}, commit=K.COMMIT_LIMIT_BYTES + 1)
    assert code == 2 and started == []


def test_launch_kills_a_fit_over_the_time_limit(tree):
    _fresh(tree)
    code, started, procs = _launch(tree, {K.CHECK: "hang"}, clock_step=600.0)
    assert code == 2 and started == [K.CHECK] and procs[K.CHECK].killed


def test_worker_environment_caps_threads_and_hides_the_gpu():
    env = K.worker_env({"PATH": "x"})
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PATH"] == "x"
    assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    assert K.LGBM_THREADS + int(env["OMP_NUM_THREADS"]) == 4
