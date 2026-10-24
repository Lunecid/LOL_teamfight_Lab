"""Task 12 (T4.3): the E4 row table, the q preprocessing and training plumbing, the selection, the lock and the part-B
TEST step (studies/tog_experiments_20261012/e4_q.py), synthetic data only.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (sections 3-7, 10); SDD Rulings 19, 28, 46, 47.
No test opens a real sealed file or a real TEST label; the frozen job plan and release are read from the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from studies.q_production_20261005 import jobs as qjobs, qbaselines, qtest, rows as q_rows_module
from studies.tog_experiments_20261012 import e2_run, e4_q as eq, e4_run as er


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# ----------------------------------------------------------------------------- a small row table (q_run_1 layout)
# rows: TRAIN a (fold 0, oof0), Q_DEV b:c0 and b:c1 (match 1), Q_CAL c, TEST d:c0 and d:c1 (match 3), TRAIN e (fold 1, q STOP)
QUERY_POST = np.array([11, 21, 23, 31, 41, 43, 51])
ROLE_ROWS = {"TRAIN": 2, "Q_DEV": 2, "Q_CAL": 1, "TEST": 2}


def _source_rows():
    cmk = np.array([b"a", b"b", b"b", b"c", b"d", b"d", b"e"], dtype="S64")
    ck = np.array([b"c0", b"c0", b"c1", b"c0", b"c0", b"c1", b"c0"], dtype="S64")
    n = len(cmk)
    pre_ms = np.array([600_000, 900_000, 910_000, 1_200_000, 700_000, 710_000, 1_500_000], dtype=np.int64)
    return {
        "canonical_match_key": cmk, "component_key": ck, "engagement_key": np.char.add(np.char.add(cmk, b":"), ck),
        "match_index": np.array([0, 1, 1, 2, 3, 3, 4], dtype=np.int64), "role": np.array([0, 3, 3, 4, 5, 5, 0], dtype=np.int8),
        "fold": np.array([0, -1, -1, -1, -1, -1, 1], dtype=np.int8),
        "q_stop": np.array([False, False, False, False, False, False, True]),
        "y": np.array([1, 0, 1, 0, -1, -1, 1], dtype=np.int8), "weight_raw": np.array([1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]),
        "query_pre": np.array([10, 20, 22, 30, 40, 42, 50], dtype=np.int64), "pre_ms": pre_ms, "elapsed_min": pre_ms / 60000.0,
        "frame_age_s": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), "truncation_bits": np.zeros(n, dtype=np.int64),
        "near_zero_bits": np.array([0, 4, 0, 7, 0, 0, 0], dtype=np.uint8),
        "exact_zero": np.array([0, 0, 0, 1, -1, -1, 0], dtype=np.int8),
        "label_svi_delta": np.array([0.03, -0.012, 0.1, 0.0, np.nan, np.nan, 0.2]),
        "pt_p_pre": np.array([0.41, 0.52, 0.63, 0.55, 0.45, 0.47, 0.31]),
        "eval_n_kills": np.array([1, 2, 1, 3, 1, 4, 2], dtype=np.int16),
        "eval_firstkill_credit_blue": np.ones(n, dtype=np.int16), "eval_firstkill_credit_red": np.ones(n, dtype=np.int16),
    }


def _preds():
    return {"final": {"query_index": np.array([20, 21, 22, 23, 30, 31, 40, 41, 42, 43]),
                      "raw_p": np.array([0.5, 0.6, 0.4, 0.4, 0.7, 0.65, 0.3, 0.35, 0.8, 0.8])},
            "oof0": {"query_index": np.array([10, 11]), "raw_p": np.array([0.45, 0.55])},
            "oof1": {"query_index": np.array([50, 51]), "raw_p": np.array([0.6, 0.6])}}


def _table():
    source, preds = _source_rows(), _preds()
    nontest = er.nontest_labels(source, QUERY_POST, preds)
    test = eq.labels_of_test_rows(source, QUERY_POST, preds["final"])
    return source, nontest, test


# ----------------------------------------------------------------------------- TEST labels and the row table
def test_labels_of_test_rows_use_the_final_context_at_both_ends():
    out = eq.labels_of_test_rows(_source_rows(), QUERY_POST, _preds()["final"])
    assert out["q_row"].tolist() == [4, 5]
    assert out["engagement_key"].tolist() == [b"d:c0", b"d:c1"]
    assert out["p_pre"].tolist() == [0.3, 0.8] and out["p_post"].tolist() == [0.35, 0.8]
    assert out["y4"].dtype == np.int8 and out["y4"].tolist() == [1, 0]
    assert out["exact_zero"].tolist() == [0, 1] and out["near_zero_bits"].tolist() == [0, 7]
    assert np.array_equal(out["label_svi_delta4"], out["p_post"] - out["p_pre"])


def test_labels_of_test_rows_refuse_a_missing_query_or_an_invalid_probability():
    rows, final = _source_rows(), _preds()["final"]
    with pytest.raises(ValueError):                                       # TEST endpoint 53 is not in PRED_final
        eq.labels_of_test_rows(rows, QUERY_POST + np.array([0, 0, 0, 0, 0, 10, 0]), final)
    broken = {"query_index": final["query_index"], "raw_p": final["raw_p"].copy()}
    broken["raw_p"][7] = np.nan
    with pytest.raises(ValueError):
        eq.labels_of_test_rows(rows, QUERY_POST, broken)
    with pytest.raises(ValueError):                                       # one endpoint per row
        eq.labels_of_test_rows(rows, QUERY_POST[:-1], final)


def test_e4_row_table_replaces_five_columns_and_seals_the_test_labels():
    source, nontest, test = _table()
    public, sealed = eq.e4_row_table(source, nontest, test)
    assert list(public) == list(source)
    assert all(public[name].dtype == source[name].dtype and public[name].shape == source[name].shape for name in source)
    assert public["y"].tolist() == [1, 1, 0, 0, -1, -1, 0]
    assert public["exact_zero"].tolist() == [0, 0, 1, 0, -1, -1, 1]
    assert public["near_zero_bits"].tolist() == [0, 0, 7, 0, 0, 0, 7]
    assert np.isnan(public["label_svi_delta"][[4, 5]]).all()
    assert public["label_svi_delta"][[0, 1, 2, 3, 6]].tolist() == nontest["label_svi_delta4"].tolist()
    assert public["pt_p_pre"].tolist() == [0.45, 0.5, 0.4, 0.7, 0.3, 0.8, 0.6]     # TEST rows: the new measurer's p_pre too
    for name in set(source) - set(eq.REPLACED):
        assert np.array_equal(public[name], source[name]), name
    assert list(sealed) == ["engagement_key", *q_rows_module.SEALED_COLUMNS]
    assert sealed["engagement_key"].tolist() == [b"d:c0", b"d:c1"]
    assert sealed["y"].tolist() == [1, 0] and sealed["exact_zero"].tolist() == [0, 1]
    assert sealed["near_zero_bits"].tolist() == [0, 7]
    assert sealed["label_svi_delta"].tolist() == (test["p_post"] - test["p_pre"]).tolist()
    assert [sealed[name].dtype for name in q_rows_module.SEALED_COLUMNS] == [np.int8, np.float64, np.int8, np.uint8]
    assert eq.public_problems(source, public) == [] and eq.sealed_problems(source, sealed) == []


def test_e4_row_table_refuses_gaps_overlaps_foreign_keys_and_lossy_values():
    source, nontest, test = _table()
    gap = {name: value[1:] for name, value in nontest.items()}
    with pytest.raises(ValueError):
        eq.e4_row_table(source, gap, test)
    overlap = {name: np.concatenate([value, value[:1]]) for name, value in test.items()}
    overlap["q_row"][-1] = 0
    overlap["engagement_key"][-1] = b"a:c0"
    with pytest.raises(ValueError):
        eq.e4_row_table(source, nontest, overlap)
    foreign = {**test, "engagement_key": test["engagement_key"][::-1]}
    with pytest.raises(ValueError):
        eq.e4_row_table(source, nontest, foreign)
    lossy = {**nontest, "near_zero_bits": nontest["near_zero_bits"].astype(np.int64) + 300}
    with pytest.raises(ValueError):
        eq.e4_row_table(source, lossy, test)


def test_public_and_sealed_problems_catch_changes():
    source, nontest, test = _table()
    public, sealed = eq.e4_row_table(source, nontest, test)
    changed = {**public, "pre_ms": public["pre_ms"] + 1}
    assert any("pre_ms" in problem for problem in eq.public_problems(source, changed))
    narrowed = {**public, "pre_ms": public["pre_ms"].astype(np.int32)}               # equal values, another dtype
    assert eq.public_problems(source, narrowed) == ["pre_ms: dtype or shape differs from the source row table"]
    unsealed = {**public, "y": public["y"].copy()}
    unsealed["y"][4] = 1
    assert eq.public_problems(source, unsealed)
    reordered = {**sealed, "engagement_key": sealed["engagement_key"][::-1]}
    assert eq.sealed_problems(source, reordered)
    invalid = {**sealed, "y": np.array([1, 2], dtype=np.int8)}
    assert eq.sealed_problems(source, invalid)


def test_row_counts_hold_no_test_label_statistic():
    source, nontest, test = _table()
    public, _ = eq.e4_row_table(source, nontest, test)
    counts = eq.row_counts(public)
    assert set(counts["TEST"]) == {"rows", "matches", "labels"}
    assert counts["TEST"]["rows"] == 2 and counts["TEST"]["matches"] == 1
    assert counts["Q_DEV"]["positive"] == 1 and counts["Q_DEV"]["positive_rate_unit"] == 0.5
    assert counts["Q_DEV"]["positive_rate_match_weighted"] == 0.5 and counts["Q_DEV"]["exact_zero"] == 1
    assert counts["TRAIN"]["positive"] == 1 and counts["TRAIN"]["stop_rows"] == 1 and counts["TRAIN"]["fit_rows"] == 1
    assert counts["Q_CAL"]["positive"] == 0 and counts["rows"] == 7


def test_nontest_differences_name_every_differing_column():
    _, nontest, _ = _table()
    same = {name: value.copy() for name, value in nontest.items()}
    assert eq.nontest_differences(nontest, same) == []
    same["y4"] = same["y4"].copy()
    same["y4"][0] = 1 - same["y4"][0]
    same["p_pre"] = same["p_pre"].astype(np.float32)
    same["near_zero_bits"] = same["near_zero_bits"].astype(np.uint16)              # equal values, another dtype
    assert eq.nontest_differences(nontest, same) == ["near_zero_bits", "p_pre", "y4"]
    del same["context"]
    assert any("context" in item for item in eq.nontest_differences(nontest, same))


def test_npz_bytes_equal_the_written_file_and_write_bytes_once_keeps_or_refuses(tmp_path):
    arrays = {"k": np.array([b"a", b"b"]), "x": np.array([1.5, np.nan])}
    data = eq.npz_bytes(arrays)
    path = tmp_path / "t.npz"
    digest = eq.write_bytes_once(path, data)
    assert digest == _sha(path) == hashlib.sha256(data).hexdigest()
    reference = tmp_path / "r.npz"
    _save_npz(reference, **arrays)
    assert _sha(reference) == digest
    assert eq.write_bytes_once(path, eq.npz_bytes(arrays)) == digest           # identical bytes: kept
    with pytest.raises(FileExistsError):
        eq.write_bytes_once(path, eq.npz_bytes({"k": arrays["k"]}))
    assert _sha(path) == digest


# ----------------------------------------------------------------------------- jobs, specs, scheduling, states
def test_plan_jobs_are_the_frozen_search_jobs_of_the_prespec_cards():
    jobs = eq.plan_jobs(qjobs.plan())
    assert [job.card_id for job in jobs] == [*eq.CARDS["lightgbm"], *eq.CARDS["PT_flex"]]
    assert eq.CARDS["lightgbm"] == ("lgbm_l15_m100", "lgbm_l15_m200", "lgbm_l31_m100", "lgbm_l31_m200")
    assert eq.CARDS["PT_flex"] == tuple(f"ptflex_k{k}_C{c}" for k in (4, 6, 8) for c in ("0.001", "0.01", "0.1", "1"))
    assert all(job.stage == "search" and job.variant == "main_h90" for job in jobs)
    assert {job.condition_id for job in jobs[:4]} == {"LGBM_S1_A1_M1_E1_C1_R1_D1"}
    assert {job.family for job in jobs[:4]} == {"lightgbm"} and {job.family for job in jobs[4:]} == {"PT_flex"}
    # the same job identities as the main q run (q_run_1 jobs/<key>/SPEC.json job.identity)
    assert jobs[0].identity == "e54a43fe575a8934c486022539d9d9f100b9b81dace8c6d0219060585b72390b"
    assert jobs[4].identity == "0dbfde6b1ae494e15ff9dc59133d38529b301a90b7c66373cb85a090c442e21f"


def test_job_spec_has_the_supervisor_shape():
    prepared = {"rows_file": "Q_ROWS_h90.npz", "rows_sha256": "r", "xy_file": "inputs/main_h90/XY.npy", "xy_sha256": "x",
                "contexts": {"FIT": {"dir": "inputs/main_h90/FIT"}, "REFIT_TRAIN": {"dir": "inputs/main_h90/REFIT_TRAIN"}}}
    job = eq.plan_jobs(qjobs.plan())[0]
    run = Path("D:/run")
    spec = eq.job_spec(job, run, prepared, Path("C:/release.json"), "digest", "2026-10-13T00:00:00+00:00")
    assert tuple(spec) == eq.SPEC_KEYS
    assert spec["job"] == job.to_dict() and spec["variant"] == "main_h90" and spec["device"] == "cpu"
    assert spec["out_dir"] == str(run / "jobs" / qtest.key_of(job.job_id))
    assert spec["release"] == str(Path("C:/release.json")) and spec["release_digest"] == "digest"
    assert spec["inputs"] == {"rows": str(run / "Q_ROWS_h90.npz"), "rows_sha256": "r", "xy": str(run / "inputs/main_h90/XY.npy"),
                              "xy_sha256": "x", "store": str(run / "raw" / "store"),
                              "contexts": {"FIT": str(run / "inputs/main_h90/FIT"),
                                           "REFIT_TRAIN": str(run / "inputs/main_h90/REFIT_TRAIN")}}


def test_ensure_spec_writes_once_and_refuses_another_spec(tmp_path):
    spec = {"job": {"job_id": "x"}, "variant": "main_h90", "written_utc": "t1"}
    path = eq.ensure_spec(tmp_path / "job", spec)
    digest = _sha(path)
    assert eq.ensure_spec(tmp_path / "job", {**spec, "written_utc": "t2"}) == path    # same spec: the first file stays
    assert _sha(path) == digest
    with pytest.raises(ValueError):
        eq.ensure_spec(tmp_path / "job", {**spec, "variant": "other"})


def test_next_launches_run_one_lightgbm_at_a_time_and_at_most_four_workers():
    pending = [("l1", "lightgbm"), ("l2", "lightgbm"), ("p1", "PT_flex"), ("p2", "PT_flex"), ("p3", "PT_flex"), ("p4", "PT_flex")]
    assert eq.next_launches(pending, []) == ["l1", "p1", "p2", "p3"]
    assert eq.next_launches(pending, ["lightgbm"]) == ["p1", "p2", "p3"]
    assert eq.next_launches(pending, ["PT_flex", "PT_flex"]) == ["l1", "p1"]
    assert eq.next_launches(pending, ["PT_flex"] * 4) == []
    assert eq.next_launches(pending[:2], []) == ["l1"]
    assert eq.next_launches(pending, [], max_workers=1) == ["l1"]


def test_job_state_reads_the_worker_result(tmp_path):
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "SPEC.json").write_text("{}", encoding="utf-8")
    assert eq.job_state(directory) == "pending"
    spec_sha = _sha(directory / "SPEC.json")
    (directory / "RESULT.json").write_text(json.dumps({"status": "failed", "terminal": False}), encoding="utf-8")
    assert eq.job_state(directory) == "failed"
    (directory / "RESULT.json").write_text(json.dumps({"status": "failed", "terminal": True}), encoding="utf-8")
    assert eq.job_state(directory) == "terminal"
    (directory / "RESULT.json").write_text(json.dumps({"status": "completed", "spec_sha256": spec_sha}), encoding="utf-8")
    assert eq.job_state(directory) == "completed"
    (directory / "RESULT.json").write_text(json.dumps({"status": "completed", "spec_sha256": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        eq.job_state(directory)


def test_select_cards_uses_qselect_and_needs_every_card():
    def result(brier):
        return {"status": "completed", "scores": {"q_dev": {"raw_match_brier": brier}}, "q_dev_keys_sha256": "k"}

    orders = {"lightgbm": ["l1", "l2"], "PT_flex": ["p1", "p2", "p3"]}
    results = {"lightgbm": {"l1": result(0.23), "l2": result(0.22)},
               "PT_flex": {"p1": result(0.24), "p2": result(0.24), "p3": result(0.25)}}
    records, problems = eq.select_cards(results, orders)
    assert problems == [] and records["lightgbm"]["selected"] == "l2" and records["PT_flex"]["selected"] == "p1"   # tie: card order
    assert records["lightgbm"]["brier"] == 0.22
    results["PT_flex"]["p3"] = {"status": "failed"}
    records, problems = eq.select_cards(results, orders)
    assert records["PT_flex"]["selected"] == "p1" and problems and "PT_flex" in problems[0]


# ----------------------------------------------------------------------------- preprocessing helpers
def _context_dir(root, npz=b"npz", manifest=b"manifest", sigma=1.5):
    root.mkdir(parents=True)
    (root / "bundle").mkdir()
    (root / "CONTEXT.npz").write_bytes(npz)
    (root / "bundle" / "MANIFEST.json").write_bytes(manifest)
    for name in ("evidence.json", "numeric.json", "vocab.json"):
        (root / "bundle" / name).write_bytes(name.encode())
    record = {"context_npz_sha256": hashlib.sha256(npz).hexdigest(), "bundle_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
              "scope": "FIT", "scope_rows": 3, "scope_positions_sha256": "s", "sigma": sigma, "m_median": [0.5], "m_stats": ["level"],
              "version": "v", "release_digest": "d", "store_build_sha256": "b", "xy_sha256": "x", "rows_sha256": str(root)}
    (root / "CONTEXT.json").write_text(json.dumps(record), encoding="utf-8")


def test_context_comparison_requires_equal_context_npz_and_manifest(tmp_path):
    _context_dir(tmp_path / "base")
    _context_dir(tmp_path / "same")
    same = eq.context_comparison(tmp_path / "same", tmp_path / "base")
    assert same["context_npz_equal"] and same["bundle_manifest_equal"] and same["bundle_files_equal"]
    assert eq.context_problems("FIT", same) == []
    _context_dir(tmp_path / "other", npz=b"other")
    other = eq.context_comparison(tmp_path / "other", tmp_path / "base")
    assert not other["context_npz_equal"] and other["bundle_manifest_equal"]
    assert any("CONTEXT.npz" in problem for problem in eq.context_problems("FIT", other))
    _context_dir(tmp_path / "sigma", sigma=2.5)
    assert any("sigma" in problem for problem in eq.context_problems("FIT", eq.context_comparison(tmp_path / "sigma", tmp_path / "base")))


def test_copy_verified_checks_both_ends(tmp_path):
    source = tmp_path / "XY.npy"
    source.write_bytes(b"xy-bytes")
    digest = _sha(source)
    record = eq.copy_verified(source, tmp_path / "out" / "XY.npy", digest)
    assert record["created"] and _sha(tmp_path / "out" / "XY.npy") == digest
    assert not eq.copy_verified(source, tmp_path / "out" / "XY.npy", digest)["created"]
    with pytest.raises(ValueError):
        eq.copy_verified(source, tmp_path / "out2" / "XY.npy", "0" * 64)
    (tmp_path / "out3").mkdir()
    (tmp_path / "out3" / "XY.npy").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        eq.copy_verified(source, tmp_path / "out3" / "XY.npy", digest)
    source.write_bytes(b"changed")                                     # the source no longer has the recorded bytes
    with pytest.raises(ValueError, match="recorded SHA-256"):
        eq.copy_verified(source, tmp_path / "out" / "XY.npy", digest)


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are a Windows feature")
def test_ensure_store_link_makes_and_checks_a_junction(tmp_path):
    target = tmp_path / "q_run" / "raw" / "store"
    target.mkdir(parents=True)
    (target / "BUILD.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "out" / "raw" / "store"
    record = eq.ensure_store_link(link, target)
    assert record["created"] and os.path.isjunction(link) and os.path.samefile(link, target)
    assert _sha(link / "BUILD.json") == _sha(target / "BUILD.json")
    assert not eq.ensure_store_link(link, target)["created"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError):
        eq.ensure_store_link(link, elsewhere)
    plain = tmp_path / "plain" / "store"
    plain.mkdir(parents=True)
    with pytest.raises(ValueError):
        eq.ensure_store_link(plain, target)
    with pytest.raises(ValueError, match="not a directory junction"):        # the target itself is no junction
        eq.ensure_store_link(target, target)


# ----------------------------------------------------------------------------- part B: decision, alignment, evaluation
def test_decide_three_way_on_the_95_interval():
    assert eq.decide([-0.02, -0.001]) == "same_direction"
    assert eq.decide([0.001, 0.02]) == "opposite_direction"
    assert eq.decide([-0.01, 0.01]) == "undetermined"
    assert eq.decide([-0.01, 0.0]) == "undetermined" and eq.decide([0.0, 0.01]) == "undetermined"   # 0 on a bound
    for bad in ([0.01, -0.01], [float("nan"), 0.1]):
        with pytest.raises(ValueError):
            eq.decide(bad)
    assert set(eq.DECISIONS) == {"same_direction", "opposite_direction", "undetermined"}


def test_align_keys_by_engagement_key():
    table = np.array([b"x", b"y", b"z"])
    assert eq.align_keys(np.array([b"z", b"x"]), table).tolist() == [2, 0]
    with pytest.raises(ValueError):
        eq.align_keys(np.array([b"w"]), table)
    with pytest.raises(ValueError):
        eq.align_keys(np.array([b"x"]), np.array([b"x", b"x"]))


def _evaluation_arrays(seed=7, matches=40):
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(matches), 2)
    n = match.size
    y4 = rng.integers(0, 2, n).astype(np.int8)
    exact = np.zeros(n, dtype=np.int8)
    exact[::7] = 1
    y4[exact == 1] = 0
    flip = rng.random(n) < 0.2
    y_main = np.where(flip, 1 - y4, y4).astype(np.int8)
    svi = (2.0 * y_main - 1.0) * rng.uniform(0.001, 0.1, n)
    p_lgbm = np.clip(0.75 * y4 + 0.125 + rng.normal(0, 0.05, n), 0.01, 0.99)
    p_pt = np.clip(0.5 + 0.05 * (y4 - 0.5) + rng.normal(0, 0.05, n), 0.01, 0.99)
    return y4, exact, y_main, svi, p_lgbm, p_pt, match


def test_evaluate_reports_comparison_decision_agreement_and_the_tie_free_comparison():
    y4, exact, y_main, svi, p_lgbm, p_pt, match = _evaluation_arrays()
    out = eq.evaluate(y4, exact, y_main, svi, p_lgbm, p_pt, match)
    direct = e2_run.evaluate_pair(y4, p_lgbm, p_pt, match)
    assert out["comparison"]["brier_difference"] == direct["brier_difference"]
    assert out["comparison"]["brier"] == {"lightgbm": direct["brier"]["P1"], "PT_flex": direct["brier"]["PT_flex"]}
    assert out["comparison"]["auc"] == {"lightgbm": direct["auc"]["P1"], "PT_flex": direct["auc"]["PT_flex"]}
    interval = out["comparison"]["brier_difference"]["interval"]
    assert out["decision"]["outcome"] == eq.decide(interval) == "same_direction"
    assert out["decision"]["text"] == eq.DECISIONS["same_direction"] and out["decision"]["level"] == 0.95
    assert out["agreement"] == er.agreement(y_main, y4, svi, match, exact_zero_new=exact)
    keep = exact == 0
    tie_free = out["supplementary_without_exact_zero"]
    assert tie_free["rows_excluded"] == int((~keep).sum()) and tie_free["rows"] == int(keep.sum())
    assert tie_free["comparison"]["brier_difference"] == e2_run.evaluate_pair(y4[keep], p_lgbm[keep], p_pt[keep], match[keep])["brier_difference"]
    share = out["agreement"]["overall"]["exact_zero_share_new"]
    assert tie_free["prespec_condition_share_over_0.01"] == {name: share[name] > 0.01 for name in share}
    assert out["rows"] == y4.size and out["matches"] == 40


def test_evaluate_refuses_inputs_that_would_drop_a_primary_figure(monkeypatch):
    y4, exact, y_main, svi, p_lgbm, p_pt, match = _evaluation_arrays()
    with pytest.raises(ValueError):                                              # a TEST placeholder is not a label
        eq.evaluate(np.where(np.arange(y4.size) == 0, -1, y4), exact, y_main, svi, p_lgbm, p_pt, match)
    original = er.agreement

    def without_match_weights(y_main_, y_new, svi_main, match_index=None, *, exact_zero_new=None):
        return original(y_main_, y_new, svi_main, None, exact_zero_new=exact_zero_new)

    monkeypatch.setattr(er, "agreement", without_match_weights)
    with pytest.raises(ValueError, match="match_weighted"):
        eq.evaluate(y4, exact, y_main, svi, p_lgbm, p_pt, match)


# ----------------------------------------------------------------------------- part B: the frozen qtest scoring path
def _reference_cell(root, *, tamper=False, unseal=False):
    rng = np.random.default_rng(3)
    role = np.array([0] * 160 + [3] * 40 + [5] * 30, dtype=np.int8)
    n = role.size
    p = rng.uniform(0.05, 0.95, n)
    t = rng.uniform(1.0, 40.0, n)
    y = (rng.random(n) < p).astype(np.int8)
    y[role == 5] = -1
    if unseal:
        y[-1] = 1
    keys = np.array([f"m{i:03d}:c0".encode() for i in range(n)])
    rows_path = root / "Q_ROWS_h90.npz"
    root.mkdir(parents=True)
    _save_npz(rows_path, engagement_key=keys, role=role, y=y, pt_p_pre=p, elapsed_min=t, match_index=np.arange(n))
    card = {"p_knots": 4, "time_knots": 4, "degree": 3, "C": 1.0}
    train = role == 0
    model, design = qbaselines.fit_reference("PT_flex", card, {"p": p[train], "t": t[train], "y": y[train].astype(np.float64),
                                                                "w_norm": np.ones(int(train.sum()))})
    source = root / "job"
    source.mkdir()
    with (source / "model_refit.pkl").open("wb") as stream:
        pickle.dump({"model": model, "design": design}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    dev = np.flatnonzero(role == 3)
    logits = np.asarray(model.predict_logits(design(p[dev], t[dev])), dtype=np.float64)
    if tamper:
        logits = logits + 1e-6
    _save_npz(source / "pred_q_dev.npz", row=dev, logit=logits, p=1.0 / (1.0 + np.exp(-logits)))
    spec = {"job": {"family": "PT_flex", "card": card, "factors": {}, "inputs": "pt"},
            "inputs": {"rows": str(rows_path), "rows_sha256": _sha(rows_path)}}
    (source / "SPEC.json").write_text(json.dumps(spec), encoding="utf-8")
    test = np.flatnonzero(role == 5)
    expected = np.asarray(model.predict_logits(design(p[test], t[test])), dtype=np.float64)
    return {"key": "job_pt", "source_dir": str(source), "model_file": "model_refit.pkl"}, test, keys[test], expected


def test_score_cell_follows_the_frozen_qtest_path_on_a_reference_cell(tmp_path):
    cell, test, keys, expected = _reference_cell(tmp_path / "ok")
    out = eq.score_cell(cell)
    assert out["kind"] == "reference" and out["reload_max_abs_logit_diff"] == 0.0 and out["reload_rows"] == 40
    assert out["row"].tolist() == test.tolist() and out["engagement_key"].tolist() == keys.tolist()
    assert np.array_equal(out["logit"], expected) and np.array_equal(out["p"], qtest.sigmoid(expected))
    tampered, *_ = _reference_cell(tmp_path / "tampered", tamper=True)
    with pytest.raises(eq.ScoringRefused, match="re-loaded"):
        eq.score_cell(tampered)
    unsealed, *_ = _reference_cell(tmp_path / "unsealed", unseal=True)
    with pytest.raises(eq.ScoringRefused, match="not sealed"):
        eq.score_cell(unsealed)


# ----------------------------------------------------------------------------- rows stage (synthetic run under tmp_path)
CONTEXT_NAMES = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")


def _write_rows_inputs(root):
    """Synthetic labels_1, q_run_1 row table, projection and Task 10 outputs; Task 11's driver then writes
    E4_LABELS_nonTEST.npz and E4_QDEV_AGREEMENT.json.  Engagement rows: TRAIN a (fold 0), V_DEV v, Q_DEV b:c0 and b:c1,
    invalid TRAIN x, Q_CAL c, TEST d:c0 and d:c1, TRAIN e (fold 1)."""

    labels_dir, q_run, projection, out = root / "labels", root / "q_run", root / "projection", root / "out"
    for folder in (labels_dir, q_run, projection, out):
        folder.mkdir()
    cmk = np.array([b"a", b"v", b"b", b"b", b"x", b"c", b"d", b"d", b"e"], dtype="S64")
    ck = np.array([b"c0", b"c0", b"c0", b"c1", b"c0", b"c0", b"c0", b"c1", b"c0"], dtype="S64")
    match = np.array([0, 6, 1, 1, 5, 2, 3, 3, 4])
    role = np.array([0, 1, 3, 3, 0, 4, 5, 5, 0], dtype=np.int8)
    fold = np.array([0, -1, -1, -1, 2, -1, -1, -1, 1], dtype=np.int8)
    q_pre = np.array([10, 90, 20, 22, 80, 30, 40, 42, 50])
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    direction = np.array([1, 0, 0, 1, -1, 0, 1, 0, 1], dtype=np.int8)
    svi = np.array([0.03, 0.1, -0.012, 0.1, np.nan, 0.0, 0.5, -0.2, 0.2])
    _save_npz(labels_dir / "ENGAGEMENTS.npz", canonical_match_key=cmk, component_key=ck, match_index=match, role=role,
              fold=fold, query_pre=q_pre, query_h90=q_pre + 1, valid_h90=valid)
    _save_npz(labels_dir / "LABELS_h90.npz", canonical_match_key=cmk, component_key=ck, match_index=match.astype(np.int32),
              role=role, fold=fold, query_pre=q_pre, query_post=q_pre + 1, valid=valid, direction_positive=direction, svi_delta=svi)
    np.save(labels_dir / "Q_DEV_H90_KEYS.npy", np.array([b"b:c0", b"b:c1"]))
    (labels_dir / "LABELS_MANIFEST.json").write_text("{}", encoding="utf-8")
    lock = {"q_may_start": True, "locked_utc": "2026-10-06T00:00:00+00:00",
            "files": {name: _sha(labels_dir / name) for name in ("LABELS_h90.npz", "ENGAGEMENTS.npz", "Q_DEV_H90_KEYS.npy")}}
    (labels_dir / "LABEL_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    source = _source_rows()
    source_path = q_run / "Q_ROWS_h90.npz"
    _save_npz(source_path, **source)
    membership = np.zeros((6, 7), dtype=np.uint8)
    membership[1, 0] = membership[2, 4] = 32                        # TRAIN a: held out by oof0; TRAIN e: by oof1
    np.save(projection / "CONTEXT_MEMBERSHIP.npy", membership)
    tables = {name: (value["query_index"], value["raw_p"]) for name, value in _preds().items()}
    by_context = {"final": {"Q_CAL": 1, "Q_DEV": 2, "TEST": 2}, "oof0": {"TRAIN": 1}, "oof1": {"TRAIN": 1}}
    row_inputs = {Path(path).as_posix(): _sha(path) for path in (source_path, labels_dir / "ENGAGEMENTS.npz",
                                                                 labels_dir / "LABEL_LOCK.json", projection / "CONTEXT_MEMBERSHIP.npy")}
    contexts = {}
    for ctx in CONTEXT_NAMES:
        index, raw = tables.get(ctx, ([], []))
        _save_npz(out / f"PRED_{ctx}.npz", query_index=np.array(index, dtype=np.int64), raw_p=np.array(raw, dtype=np.float64))
        record = {"context": ctx, "columns": ["query_index", "raw_p"], "pred_sha256": _sha(out / f"PRED_{ctx}.npz"),
                  "inputs": row_inputs, "rows": {"q_rows_by_role": by_context.get(ctx, {})}}
        (out / f"PRED_{ctx}.json").write_text(json.dumps(record), encoding="utf-8")
        contexts[ctx] = {"pred_sha256": record["pred_sha256"], "record_sha256": _sha(out / f"PRED_{ctx}.json")}
    report = {"v_dev_passed": True, "oof_check": {"passed": True}, "smoke": None, "q_rows_by_role": ROLE_ROWS, "contexts": contexts}
    (out / "E4_VSCORE.json").write_text(json.dumps(report), encoding="utf-8")
    assert er.run(out, pred_dir=out, labels_dir=labels_dir, q_rows_path=source_path, projection=projection, role_rows=ROLE_ROWS) == 0
    paths = eq.Paths(out=out, q_run=q_run, labels=labels_dir, projection=projection)
    expected = {"labels": _sha(out / er.LABELS_NAME), "agreement": _sha(out / er.AGREEMENT_NAME)}
    return paths, expected


def test_rows_stage_writes_the_table_the_sealed_file_and_the_records(tmp_path):
    paths, expected = _write_rows_inputs(tmp_path)
    assert eq.rows_stage(paths, role_rows=ROLE_ROWS, expected=expected) == 0
    public, sealed = _load(paths.rows), _load(paths.sealed)
    source = _source_rows()
    assert list(public) == list(source) and public["y"].tolist() == [1, 1, 0, 0, -1, -1, 0]
    assert public["pt_p_pre"].tolist() == [0.45, 0.5, 0.4, 0.7, 0.3, 0.8, 0.6]
    assert sealed["engagement_key"].tolist() == [b"d:c0", b"d:c1"] and sealed["y"].tolist() == [1, 0]
    record = json.loads(paths.rows_record.read_text(encoding="utf-8"))
    assert record["rows_sha256"] == _sha(paths.rows) and record["test_sealed_sha256"] == _sha(paths.sealed)
    assert set(record["counts"]["TEST"]) == {"rows", "matches", "labels"}
    assert record["sealed_test_columns"] == list(q_rows_module.SEALED_COLUMNS)
    stage = json.loads((paths.out / eq.ROWS_STAGE).read_text(encoding="utf-8"))
    assert stage["passed"] is True
    assert {label: entry["sha256"] for label, entry in stage["files"].items()} == {
        "rows": _sha(paths.rows), "sealed": _sha(paths.sealed), "record": _sha(paths.rows_record)}
    assert all(stage["checks"][name] is True for name in stage["checks"])
    text = (paths.out / eq.ROWS_STAGE).read_text(encoding="utf-8")
    assert "positive" not in text                                        # the stage record holds no label statistic
    before = {path.name: _sha(path) for path in (paths.rows, paths.sealed, paths.rows_record, paths.out / eq.ROWS_STAGE)}
    assert eq.rows_stage(paths, role_rows=ROLE_ROWS, expected=expected) == 0       # written once: a rerun changes nothing
    assert {name: _sha(paths.out / name) for name in before} == before


@pytest.mark.parametrize("which, name", [("labels", er.LABELS_NAME), ("agreement", er.AGREEMENT_NAME)])
def test_rows_stage_blocks_on_a_task11_file_other_than_ruling_46(tmp_path, which, name):
    paths, expected = _write_rows_inputs(tmp_path)
    assert eq.rows_stage(paths, role_rows=ROLE_ROWS, expected={**expected, which: "0" * 64}) == 2
    assert not paths.rows.exists() and not paths.sealed.exists() and not paths.rows_record.exists()
    blocked = json.loads((paths.out / "E4_BLOCKED_rows_1.json").read_text(encoding="utf-8"))
    assert any(reason.startswith(f"{name} has SHA-256") and "Ruling 46" in reason for reason in blocked["reasons"])


def test_rows_stage_blocks_when_the_stored_nontest_labels_differ(tmp_path):
    paths, expected = _write_rows_inputs(tmp_path)
    labels_path = paths.out / er.LABELS_NAME
    stored = _load(labels_path)
    stored["y4"] = stored["y4"].copy()
    stored["y4"][0] = 1 - stored["y4"][0]
    os.replace(labels_path, paths.out / "E4_LABELS_nonTEST.original.npz")
    _save_npz(labels_path, **stored)
    assert eq.rows_stage(paths, role_rows=ROLE_ROWS, expected={**expected, "labels": _sha(labels_path)}) == 2
    first = json.loads((paths.out / "E4_BLOCKED_rows_1.json").read_text(encoding="utf-8"))
    assert first["reasons"] == [f"{er.AGREEMENT_NAME} describes another {er.LABELS_NAME}"]      # the record names the file
    agreement_path = paths.out / er.AGREEMENT_NAME
    record = json.loads(agreement_path.read_text(encoding="utf-8"))
    record["labels"]["sha256"] = _sha(labels_path)                                               # a consistent tampered pair
    os.replace(agreement_path, paths.out / "E4_QDEV_AGREEMENT.original.json")
    agreement_path.write_text(json.dumps(record), encoding="utf-8")
    assert eq.rows_stage(paths, role_rows=ROLE_ROWS, expected={"labels": _sha(labels_path), "agreement": _sha(agreement_path)}) == 2
    assert not paths.rows.exists() and not paths.sealed.exists()
    blocked = json.loads((paths.out / "E4_BLOCKED_rows_2.json").read_text(encoding="utf-8"))
    assert any("y4" in reason for reason in blocked["reasons"])


# ----------------------------------------------------------------------------- part B: the TEST step (synthetic run)
def _write_test_inputs(root, *, drop_main_key=False, tamper_e4=False, bad_main=False):
    """A locked synthetic E4 run: row table (4 TRAIN rows, 60 TEST rows in 30 matches), PRED_final, ENGAGEMENTS, both
    sealed files, TEST_REPORT.json, two cells and E4_LOCK.json binding them.  Every 6th TEST row is an exact tie."""

    rng = np.random.default_rng(11)
    out, q_run, labels_dir = root / "out", root / "q_run", root / "labels"
    for folder in (out, q_run, labels_dir):
        folder.mkdir()
    n_train, n_test = 4, 60
    n = n_train + n_test
    match = np.concatenate([np.repeat([0, 1], 2), np.repeat(np.arange(2, 32), 2)]).astype(np.int64)
    cmk = np.array([f"m{m:03d}".encode() for m in match], dtype="S64")
    ck = np.array([b"c0", b"c1"] * (n // 2), dtype="S64")
    role = np.array([0] * n_train + [5] * n_test, dtype=np.int8)
    fold = np.array([0, 0, 1, 1] + [-1] * n_test, dtype=np.int8)
    query_pre = (100 + 10 * np.arange(n)).astype(np.int64)
    query_post = query_pre + 1
    test = np.flatnonzero(role == 5)
    p_pre = rng.uniform(0.1, 0.9, n_test)
    p_post = np.clip(p_pre + rng.normal(0, 0.05, n_test), 0.01, 0.99)
    p_post[::6] = p_pre[::6]
    order = np.argsort(np.concatenate([query_pre[test], query_post[test]]))
    pred_final = {"query_index": np.concatenate([query_pre[test], query_post[test]])[order],
                  "raw_p": np.concatenate([p_pre, p_post])[order]}
    _save_npz(out / "PRED_final.npz", **pred_final)
    y = np.array([1, 0, 1, 0] + [-1] * n_test, dtype=np.int8)
    public = {"canonical_match_key": cmk, "component_key": ck, "engagement_key": np.char.add(np.char.add(cmk, b":"), ck),
              "match_index": match, "role": role, "fold": fold, "query_pre": query_pre, "y": y,
              "label_svi_delta": np.where(role == 5, np.nan, 0.05), "exact_zero": np.where(role == 5, -1, 0).astype(np.int8),
              "near_zero_bits": np.zeros(n, dtype=np.uint8),
              "pt_p_pre": np.concatenate([np.full(n_train, 0.5), p_pre])}
    _save_npz(out / "Q_ROWS_h90.npz", **public)
    labels = eq.labels_of_test_rows(public, query_post, pred_final)
    sealed = {"engagement_key": labels["engagement_key"], "y": labels["y4"].copy(), "label_svi_delta": labels["label_svi_delta4"],
              "exact_zero": labels["exact_zero"], "near_zero_bits": labels["near_zero_bits"]}
    if tamper_e4:                                                          # a valid label that PRED_final does not give
        sealed["y"][1] = 1 - sealed["y"][1]
    _save_npz(out / "Q_TEST_SEALED_h90.npz", **sealed)
    (out / "Q_ROWS_h90.json").write_text("{}", encoding="utf-8")
    _save_npz(labels_dir / "ENGAGEMENTS.npz", canonical_match_key=cmk, component_key=ck, match_index=match, role=role,
              fold=fold, query_pre=query_pre, query_h90=query_post, valid_h90=np.ones(n, dtype=bool))
    (labels_dir / "LABEL_LOCK.json").write_text("{}", encoding="utf-8")
    y4 = labels["y4"]
    y_main = np.where(rng.random(n_test) < 0.2, 1 - y4, y4).astype(np.int8)
    main_keys = public["engagement_key"][test][::-1]                       # another order: alignment is by key
    main_y = y_main[::-1]
    main_svi = (2.0 * main_y - 1.0) * rng.uniform(0.001, 0.1, n_test)
    if bad_main:                                                           # a main label against its own SVI sign
        main_svi[0] = -main_svi[0]
    if drop_main_key:
        main_keys, main_y, main_svi = main_keys[1:], main_y[1:], main_svi[1:]
    _save_npz(q_run / "Q_TEST_SEALED_h90.npz", engagement_key=main_keys, y=main_y, label_svi_delta=main_svi,
              exact_zero=np.zeros(main_y.size, dtype=np.int8), near_zero_bits=np.zeros(main_y.size, dtype=np.uint8))
    _save_npz(q_run / "Q_ROWS_h90.npz", engagement_key=public["engagement_key"])
    report = {"bootstrap": {"contrasts": {"P1": {"point": -0.013862, "interval": [-0.014451, -0.013281], "level": 0.975}}}}
    (q_run / "TEST_REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    _save_npz(out / er.LABELS_NAME, q_row=np.arange(n_train))
    (out / er.AGREEMENT_NAME).write_text("{}", encoding="utf-8")
    (out / eq.SELECTION_NAME).write_text("{}", encoding="utf-8")
    prespec = root / "PRESPEC_E4_KO.md"
    prespec.write_text("prespec", encoding="utf-8")
    verify = {"prespec": prespec, "release": eq.RELEASE, "rows:Q_ROWS_h90.npz": out / "Q_ROWS_h90.npz",
              "rows:Q_ROWS_h90.json": out / "Q_ROWS_h90.json", "rows:Q_TEST_SEALED_h90.npz": out / "Q_TEST_SEALED_h90.npz",
              "q_run_1:Q_TEST_SEALED_h90.npz": q_run / "Q_TEST_SEALED_h90.npz", "q_run_1:Q_ROWS_h90.npz": q_run / "Q_ROWS_h90.npz",
              "q_run_1:TEST_REPORT.json": q_run / "TEST_REPORT.json", "task10:PRED_final.npz": out / "PRED_final.npz",
              "labels_1:ENGAGEMENTS.npz": labels_dir / "ENGAGEMENTS.npz", "labels_1:LABEL_LOCK.json": labels_dir / "LABEL_LOCK.json",
              "task11:E4_LABELS_nonTEST.npz": out / er.LABELS_NAME, "task11:E4_QDEV_AGREEMENT.json": out / er.AGREEMENT_NAME,
              "select:E4_SELECTION.json": out / eq.SELECTION_NAME}
    verify.update({f"code:{name}": eq.ROOT / name for name in eq.CODE_FILES})
    cells = {}
    for family, key, model in (("lightgbm", "job_lgbm", "model_refit.pkl"), ("PT_flex", "job_pt", "model_refit.pkl")):
        source = out / "jobs" / key
        source.mkdir(parents=True)
        for name in ("SPEC.json", "RESULT.json", model, "pred_q_dev.npz"):
            (source / name).write_bytes(f"{key}:{name}".encode())
            verify[f"job:{key}:{name}"] = source / name
        cells[family] = {"key": key, "source_dir": str(source), "model_file": model, "status": "completed",
                         "spec_sha256": _sha(source / "SPEC.json"), "result_sha256": _sha(source / "RESULT.json"),
                         "model_sha256": _sha(source / model), "pred_q_dev_sha256": _sha(source / "pred_q_dev.npz")}
    lock = {"version": "test", "experiment": "E4", "locked_utc": "2026-10-13T00:00:00+00:00",
            "prespec": {"sha256": _sha(prespec), "commit": eq.PRESPEC_COMMIT},
            "verify": {label: {"path": str(path), "sha256": _sha(path)} for label, path in verify.items()},
            "rules": eq._clean(eq.rules()), "environment": eq.environment(), "cells": cells,
            "selection": {"lightgbm": {"selected": "lgbm_l15_m100", "brier": 0.22}, "PT_flex": {"selected": "ptflex_k8_C0.001", "brier": 0.24}},
            "row_counts": {"TEST": {"rows": n_test, "matches": 30}}, "stages": {"lock": {"wall_seconds": 1.0}}}
    (out / eq.LOCK_NAME).write_text(json.dumps(lock), encoding="utf-8")
    p_lgbm = np.clip(0.75 * y4 + 0.125 + rng.normal(0, 0.05, n_test), 0.01, 0.99)
    p_pt = np.clip(0.5 + 0.05 * (y4 - 0.5) + rng.normal(0, 0.05, n_test), 0.01, 0.99)
    scored = {"job_lgbm": p_lgbm, "job_pt": p_pt}
    calls = []

    def scorer(cell):
        calls.append(cell["key"])
        p = scored[cell["key"]]
        return {"row": test, "engagement_key": public["engagement_key"][test], "logit": np.log(p / (1 - p)), "p": p,
                "reload_max_abs_logit_diff": 0.0, "reload_rows": 0, "kind": "table"}

    paths = eq.Paths(out=out, q_run=q_run, labels=labels_dir, prespec=prespec, openings=root / "OPENINGS.json")
    arrays = {"y4": y4, "exact": labels["exact_zero"], "y_main": y_main, "match": match[test], "p_lgbm": p_lgbm, "p_pt": p_pt,
              "svi_main": main_svi[::-1] if not drop_main_key else None}
    return paths, _sha(out / eq.LOCK_NAME), scorer, calls, arrays


def test_run_test_step_scores_then_opens_once_and_reports(tmp_path):
    paths, lock_sha, scorer, calls, arrays = _write_test_inputs(tmp_path)
    assert eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=scorer) == 0
    assert calls == ["job_lgbm", "job_pt"]
    openings = json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]
    assert [entry["experiment"] for entry in openings] == ["E4"]
    assert openings[0]["lock_sha256"] == lock_sha and len(openings[0]["labels"]) == 2
    opening = json.loads((paths.out / eq.OPENING_NAME).read_text(encoding="utf-8"))
    assert opening["openings_before"] == [] and opening["experiment"] == "E4"
    report = json.loads((paths.out / eq.REPORT_NAME).read_text(encoding="utf-8"))
    expected = eq._clean(eq.evaluate(arrays["y4"], arrays["exact"], arrays["y_main"], arrays["svi_main"], arrays["p_lgbm"],
                                     arrays["p_pt"], arrays["match"]))
    assert report["comparison"] == expected["comparison"] and report["agreement"] == expected["agreement"]
    assert report["decision"]["outcome"] == "same_direction"
    assert report["supplementary_without_exact_zero"]["rows_excluded"] == 10
    assert report["main_comparison"]["test_report"] == {"point": -0.013862, "interval": [-0.014451, -0.013281], "level": 0.975}
    assert report["checks"] == {"sealed_e4_labels_equal_pred_final": True, "main_labels_follow_the_direction_rule": True,
                                "scores_are_the_test_rows": True, "agreement_primary_figures_present": True}
    assert report["openings_before"] == [] and report["rows"] == 60 and report["matches"] == 30
    for key in ("job_lgbm", "job_pt"):
        meta = json.loads((paths.out / eq.SCORES_DIR / f"{key}.json").read_text(encoding="utf-8"))
        assert meta["pred_sha256"] == _sha(paths.out / eq.SCORES_DIR / f"{key}.npz")
    with pytest.raises(FileExistsError):                                  # the sealed labels are opened once
        eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=scorer)


def test_run_test_step_blocks_before_any_opening(tmp_path):
    paths, lock_sha, scorer, calls, _ = _write_test_inputs(tmp_path)
    assert eq.run_test_step(paths, expected_lock_sha256="0" * 64, scorer=scorer) == 2
    assert (paths.out / "E4_BLOCKED_test_1.json").exists() and calls == []
    assert not paths.openings.exists() and not (paths.out / eq.OPENING_NAME).exists()
    with (paths.out / "Q_ROWS_h90.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")                                                  # a bound file changed after the lock
    assert eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=scorer) == 2
    blocked = json.loads((paths.out / "E4_BLOCKED_test_2.json").read_text(encoding="utf-8"))
    assert "rows:Q_ROWS_h90.json" in blocked["reasons"]["changed_since_lock"]
    assert not paths.openings.exists() and calls == []


def test_run_test_step_blocks_when_scoring_is_refused(tmp_path):
    paths, lock_sha, _, _, _ = _write_test_inputs(tmp_path)

    def refusing(cell):
        raise eq.ScoringRefused(f"{cell['key']}: re-loaded model differs")

    assert eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=refusing) == 2
    assert not paths.openings.exists() and not (paths.out / eq.OPENING_NAME).exists()


def test_run_test_step_writes_a_failure_record_after_the_opening(tmp_path):
    paths, lock_sha, scorer, _, _ = _write_test_inputs(tmp_path, drop_main_key=True)
    with pytest.raises(ValueError):
        eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=scorer)
    failed = json.loads((paths.out / "E4_FAILED_test_1.json").read_text(encoding="utf-8"))
    assert failed["failed"] is True and failed["opening"]["experiment"] == "E4"
    assert [entry["experiment"] for entry in json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]] == ["E4"]
    assert not (paths.out / eq.REPORT_NAME).exists()


@pytest.mark.parametrize("change, message", [("tamper_e4", "differ from the labels of the locked PRED_final.npz"),
                                             ("bad_main", "do not follow y = 1[SVI > 0]")])
def test_run_test_step_checks_both_sealed_files_after_the_opening(tmp_path, change, message):
    paths, lock_sha, scorer, _, _ = _write_test_inputs(tmp_path, **{change: True})
    with pytest.raises(RuntimeError) as raised:
        eq.run_test_step(paths, expected_lock_sha256=lock_sha, scorer=scorer)
    assert message in str(raised.value)
    failed = json.loads((paths.out / "E4_FAILED_test_1.json").read_text(encoding="utf-8"))
    assert message in failed["error"]["message"] and failed["rule"].startswith("do not re-run")
    assert not (paths.out / eq.REPORT_NAME).exists()


def test_verify_lock_names_a_missing_prespec_and_missing_entries(tmp_path):
    paths, _, _, _, _ = _write_test_inputs(tmp_path)
    lock = json.loads((paths.out / eq.LOCK_NAME).read_text(encoding="utf-8"))
    assert eq.verify_lock(lock) == {}
    assert "prespec_hash_missing" in eq.verify_lock({**lock, "prespec": {}})
    assert "prespec_not_the_approved_commit" in eq.verify_lock({**lock, "prespec": {**lock["prespec"], "commit": "d78912e"}})
    verify = {label: entry for label, entry in lock["verify"].items() if label != "q_run_1:Q_TEST_SEALED_h90.npz"}
    assert eq.verify_lock({**lock, "verify": verify})["entries_missing"] == ["q_run_1:Q_TEST_SEALED_h90.npz"]
    assert "rules_differ" in eq.verify_lock({**lock, "rules": {}})
    assert "environment_differs" in eq.verify_lock({**lock, "environment": {**lock["environment"], "numpy": "0"}})


def test_append_opening_refuses_a_second_e4_opening(tmp_path):
    path = tmp_path / "OPENINGS.json"
    assert eq.append_opening(path, {"experiment": "E2", "utc": "t"}) == []
    assert eq.append_opening(path, {"experiment": "E4", "utc": "t"}) == [{"experiment": "E2", "utc": "t"}]
    with pytest.raises(RuntimeError):
        eq.append_opening(path, {"experiment": "E4", "utc": "t2"})
    assert [o["experiment"] for o in json.loads(path.read_text(encoding="utf-8"))["openings"]] == ["E2", "E4"]
