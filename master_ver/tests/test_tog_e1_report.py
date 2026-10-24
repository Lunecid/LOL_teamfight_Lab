"""Task 16 (T1.5) part B documents: the Korean E1 report and the E1 ledger rows (studies/tog_experiments_20261012/e1_report.py),
on synthetic records with the production structure (the TEST and Q_CAL results come from e1_run.evaluate_pairs).

Binding: PRESPEC_E1_KO.md section 10 (ledger schema, sign note, no 'abs' outside magnitude rows); SDD Ruling 60.  No test
reads a real record or a sealed file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from reports.thesis_manuscript_20261010.ledger import verify_ledger
from studies.tog_experiments_20261012 import e1_report as rep, e1_run as e1
from studies.tog_experiments_20261012.common import sha256_file

HEADER = ["key", "printed", "kind", "source", "locator", "scale", "dataset", "note"]


def _pairs(seed, n_match=40):
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(n_match), 2)
    n = match.size
    y_eng, y_plc = rng.integers(0, 2, n), rng.integers(0, 2, n)

    def p(y, strength):
        return np.clip(0.5 + strength * (2 * y - 1) + rng.normal(0, 0.1, y.size), 0.02, 0.98)

    s = 100_000 + 1_000 * np.arange(n)
    s[1] = s[0]
    main_match = np.repeat(np.arange(n_match), 3)
    main_y = rng.integers(0, 2, main_match.size)
    return {"match": match, "eng_y": y_eng, "eng_svi": rng.normal(0, 0.05, n), "eng_p": {"lightgbm": p(y_eng, 0.3), "PT_flex": p(y_eng, 0.1)},
            "plc_y": y_plc, "plc_w": rng.normal(0, 0.05, n), "plc_p": {"lightgbm": p(y_plc, 0.35), "PT_flex": p(y_plc, 0.05)},
            "v_tau": rng.uniform(0.2, 0.8, n), "eng_key": np.array([f"m{m:03d}:c{i % 2}".encode() for i, m in enumerate(match)], dtype="S129"),
            "s_ms": s, "obj_eng": rng.random(n) < 0.5, "obj_plc": rng.random(n) < 0.4,
            "triggers": {"stratified_r": False, "deduplicated_r": False},
            "main": {"y": main_y, "p": {"lightgbm": p(main_y, 0.3), "PT_flex": p(main_y, 0.1)}, "match": main_match,
                     "matched": np.tile([True, True, False], n_match)}}


def _smd(value, pairs=None, dropped=None):
    out = {"smd": value}
    if pairs is not None:
        out.update(pairs=pairs, pairs_dropped_without_earlier_kill=dropped)
    return out


def _balance_block(public):
    block = {"engagements": 300, "matched": 106, "match_rate": 106 / 300, "pairs": 106, "matches": 60,
             "smd": {"v_start": _smd(0.0048), "game_time_min": _smd(-0.0566), "gold_difference": _smd(-0.0004),
                     "start_frame_age_s": _smd(-0.0087), "time_since_last_kill_s": _smd(-0.0564, 100, 6)},
             "objective_share": {"engagement": {"windows": 106, "with_objective_gain": 51, "share": 51 / 106},
                                 "placebo": {"windows": 106, "with_objective_gain": 45, "share": 45 / 106},
                                 "pairs_without_objective_in_either": 30},
             "reuse": {"rows": 106, "distinct_windows": 105, "rows_in_reused_windows": 2, "share_rows_in_reused_windows": 2 / 106},
             "triggers": {"stratified_r": {"flag": False, "abs_smd_v_start": 0.0048, "abs_smd_game_time": 0.0566},
                          "deduplicated_r": {"flag": False, "share_rows_in_reused_windows": 2 / 106}}}
    if public:
        block["precision"] = {"placebo": {"rows": 106, "exact_zero": 0, "abs_lt_1e-6": 3, "min_nonzero_abs": 7.3e-8},
                              "engagement": {"rows": 106, "exact_zero": 0, "abs_lt_1e-6": 1, "min_nonzero_abs": 1.7e-7}}
        block["extras"] = {"d_quartiles": {"quartiles": {str(k): {"match_rate": rate} for k, rate in enumerate((0.537, 0.451, 0.291, 0.118))}},
                           "medians": {"D_s": {"matched": 29.0, "unmatched": 52.7},
                                       "time_since_last_kill_s": {"matched": 17.5, "unmatched": 39.8}}}
    return block


def _windows_block():
    return {"engagements": 300, "matched": 106, "unmatched": 194, "match_rate": 106 / 300, "matches": 70, "matched_matches": 60,
            "overlap": {"rows": 106, "rows_overlapping_another_start": 15, "share": 15 / 106},
            "component_windows": {"matched_windows_meeting_a_component_outcome_window": 99, "share": 99 / 106},
            "reuse": {"rows": 106, "distinct_windows": 105, "rows_in_reused_windows": 2},
            "kill_gap_balance": {"pairs": 100, "pairs_dropped_without_earlier_kill": 6, "smd": -0.0564},
            "bins": {"kill_gap_bin": {str(b): {"engagements": 60, "matched": 20, "match_rate": 1 / 3} for b in range(5)}},
            "matched_vs_unmatched_by_kill_gap_bin": {str(b): {"share_of_matched": 0.2, "share_of_unmatched": 0.2} for b in range(5)},
            "xy": {"matched_windows": 106, "nonfinite_f_s_xy": 0}}


def _records(root):
    """Synthetic tog_e1_1 records with the fields e1_report reads, written as JSON files; returns the paths."""

    tog = Path(root) / "tog"
    tog.mkdir(parents=True)
    test = e1.evaluate_pairs(_pairs(1))
    qcal = e1.evaluate_pairs(_pairs(2))
    test.update(split="TEST", summaries={"windows": _windows_block(), "balance": _balance_block(False)}, checks={"a": True, "b": True},
                objective={"shard_group": "test", "parts": ["part_0400.npz"]})
    qcal.update(split="Q_CAL", summaries={"windows": _windows_block(), "balance": _balance_block(True)}, checks={"a": True})
    cells = {arm: {family: f"search__{arm}__{family}" for family in e1.FAMILIES} for arm in e1.ARMS}
    lock_path = tog / e1.LOCK_NAME
    opening = {"experiment": "E1", "utc": "2026-10-13T09:55:19.661036+00:00", "lock": str(lock_path), "lock_sha256": "3c" * 32,
               "prespec_sha256": "a2" * 32, "labels": [str(tog / "plc" / "Q_TEST_SEALED_h90.npz")], "note": "n"}
    before = [{"experiment": name, "utc": f"2026-10-12T0{k}:00:00+00:00", "lock_sha256": "ab" * 32,
               "labels": [str(Path(root) / f"tog_{name.lower()}_1" / "Q_TEST_SEALED_h90.npz")]} for k, name in enumerate(("E2", "E3", "E4"))]
    environment = {"python": "3.13.5", "numpy": "2.1.3", "scipy": "1.15.3", "lightgbm": "4.6.0", "sklearn": "1.6.1"}
    report = {"split": "TEST", "result": e1._clean(test), "opening": opening, "openings_before": before,
              "lock": {"path": str(lock_path), "sha256": "3c" * 32, "locked_utc": "2026-10-13T09:10:13+00:00"},
              "prespec": {"path": "p", "sha256": "a2" * 32, "commit": e1.PRESPEC_COMMIT}, "cells": cells,
              "scores": {arm: {family: {"rows": 106, "reload_max_abs_logit_diff": 0.0, "reload_rows": 4096, "kind": "table",
                                        "pred_sha256": "cd" * 32} for family in e1.FAMILIES} for arm in e1.ARMS},
              "stages": {"test": {"wall_seconds": 53.2, "peak_memory_bytes_main_process": 6.0 * 2**30,
                                  "parts": {"verify_lock": {"wall_seconds": 16.1}, "score": {"wall_seconds": 22.2},
                                            "evaluate": {"wall_seconds": 14.8}}}},
              "environment": environment, "run": {"code_sha256": {"studies/tog_experiments_20261012/e1_run.py": "8b" * 32}}}
    selection = {arm: {family: {"selected": f"card_{arm}_{family}", "brier": 0.2 + 0.01 * k,
                                "cards": {f"card_{arm}_{family}": {"brier": 0.2 + 0.01 * k}, "other": {"brier": 0.25}}}
                       for k, family in enumerate(e1.FAMILIES)} for arm in e1.ARMS}
    files = {
        "report": (tog / e1.REPORT_NAME, report),
        "qcal": (tog / e1.QCAL_NAME, {"result": e1._clean(qcal), "split": "Q_CAL",
                                      "rehearsal": {"checks": {"scores_reproduce_the_stored_q_cal_predictions": {"eng:lightgbm": 0.0}}}}),
        "selection": (tog / e1.SELECTION_NAME, {"selection": selection}),
        "lock": (lock_path, {"locked_utc": "2026-10-13T09:10:13+00:00", "hashing": {"files": 992},
                             "cross_checks": {"a": True, "b": True}, "environment": environment,
                             "prespec": {"sha256": "a2" * 32, "commit": e1.PRESPEC_COMMIT},
                             "amendment": {"sha256": "b4" * 32, "committed_sha256": "80" * 32, "commit": e1.AMENDMENT_COMMIT},
                             "code": {"studies/tog_experiments_20261012/e1_run.py": {"commit": "db0a19b" + "0" * 33, "sha256": "8b" * 32}},
                             "sealed_files": {label: {"path": str(tog / f"{label}.bin"), "sha256": "ef" * 32} for label in e1.SEALED_LABELS},
                             "sealed_summaries": {"files": {f"s{k}": {} for k in range(34)}}}),
        "train": (tog / e1.TRAIN_NAME, {"wall": {arm: {"first_start_utc": "2026-10-13T08:40:39+00:00", "last_end_utc": "2026-10-13T09:07:22+00:00",
                                                     "seconds": 1567.9, "launches": 16} for arm in e1.ARMS},
                                       "timing": {"wall_seconds": 1606.6}, "jobs": [{"state": "completed"}] * 32}),
        "opening": (tog / e1.OPENING_NAME, {**opening, "openings_before": before}),
        "balance": (tog / "BALANCE.json", {"roles": {name: _balance_block(True) for name in ("TRAIN", "Q_DEV", "Q_CAL")}}),
        "windows": (tog / "WINDOWS_h90.json", {"roles": {name: _windows_block() for name in ("TRAIN", "Q_DEV", "Q_CAL")},
                                               "gate_record": {"q_dev_match_rate": 0.349}}),
    }
    for path, payload in files.values():
        path.write_text(json.dumps(payload), encoding="utf-8")
    (tog / rep.STDOUT_LOG).write_text(json.dumps({"done": str(tog / e1.REPORT_NAME), "outcome": test["decision"]["outcome"]}) + "\n",
                                      encoding="utf-8")
    (tog / rep.STDERR_LOG).write_text(f"x.py:1: {rep.FEATURE_NAME_WARNING}\n  warnings.warn(\n", encoding="utf-8")
    openings = Path(root) / "OPENINGS.json"
    openings.write_text(json.dumps({"openings": [*before, opening]}), encoding="utf-8")
    return rep.Paths(tog=tog, report_md=Path(root) / "E1_REPORT_KO.md", ledger=Path(root) / "TOG_NUMBERS.csv", openings=openings)


def _ledger(path, rows=()):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


# ----------------------------------------------------------------------------- formats
def test_numbers_use_the_minus_sign_fixed_digits_and_thousands_separators():
    assert rep.num(-0.009640069476889224, 5) == "−0.00964" and rep.num(0.013160573205790865, 5) == "0.01316"
    assert rep.num(1.7324961706567186, 3) == "1.732" and rep.count(79089) == "79,089" and rep.pct(0.35600979500706714) == "35.60"
    assert rep.ci([0.012284752577, 0.014038074276], 5) == "[0.01228, 0.01404]"
    assert rep.ci([-0.0109252427, -0.0083588304], 5) == "[−0.01093, −0.00836]"
    assert rep.win("tog_e1_1", "plc", "Q_TEST_SEALED_h90.npz") == "tog_e1_1" + chr(92) + "plc" + chr(92) + "Q_TEST_SEALED_h90.npz"
    assert rep.sci(7.302469416374416e-08) == "7.3e−08" and rep.sci(1.5e-12) == "1.5e−12" and rep.sci(2.0e5) == "2.0e+05"
    assert "opening" not in rep.RECORDS                                  # E1_OPENING.json is hashed, never loaded (Minor 3)


# ----------------------------------------------------------------------------- the ledger rows
def test_ledger_rows_round_to_their_sources_under_the_frozen_verifier(tmp_path):
    paths = _records(tmp_path)
    rows = rep.ledger_rows(paths)
    keys = [row["key"] for row in rows]
    assert len(keys) == len(set(keys)) and all(key.startswith("e1_") for key in keys)
    assert {row["dataset"] for row in rows} <= {"test", "q_cal", "q_dev"} and {row["kind"] for row in rows} == {"json"}
    assert all(Path(row["source"]).is_absolute() and "abs" not in row["note"] for row in rows)       # no magnitude-only row
    signed = [row for row in rows if row["key"].split("_", 2)[-1].split("_lo")[0].split("_hi")[0] in rep.SIGNED_KEYS]
    assert signed and all(rep.SIGN_NOTE in row["note"] for row in signed)
    assert all("-" not in row["printed"] for row in rows)                                            # U+2212, not a hyphen
    ledger = tmp_path / "ledger.csv"
    _ledger(ledger, [[row[name] for name in HEADER] for row in rows])
    assert verify_ledger.main(ledger) == 0
    for key in ("e1_test_g_eng", "e1_test_g_eng_lo", "e1_test_l2_lo", "e1_test_net_hi", "e1_qcal_g_plc", "e1_test_draws_g_eng_le_0",
                "e1_test_r001_r", "e1_test_rstrat", "e1_test_main_all", "e1_qdev_brier_plc_lightgbm"):
        assert key in keys
    assert keys[-3:] == ["e1_qcal_objfree_r", "e1_qcal_objfree_r_lo", "e1_qcal_objfree_r_hi"]     # appended in fix round 1
    qcal = json.loads((paths.tog / e1.QCAL_NAME).read_text(encoding="utf-8"))["result"]["sensitivities"]["objective_free"]
    assert [row["printed"] for row in rows[-3:]] == [rep.num(qcal["point"]["r"], 3), *(rep.num(v, 3) for v in qcal["r_ci"])]
    assert all(row["dataset"] == "q_cal" and row["source"].endswith(e1.QCAL_NAME) for row in rows[-3:])


def test_append_rows_adds_lf_rows_once_and_refuses_an_existing_e1_key(tmp_path):
    paths = _records(tmp_path)
    rows = rep.ledger_rows(paths)
    _ledger(paths.ledger, [["e4_x", "1", "json", "s", "l", "1", "test", "n"]])
    rep.append_rows(paths.ledger, rows)
    data = paths.ledger.read_bytes()
    assert b"\r" not in data and data.count(b"\n") == 2 + len(rows)
    with pytest.raises(FileExistsError):
        rep.append_rows(paths.ledger, rows[:1])
    with paths.ledger.open(encoding="utf-8", newline="") as stream:
        assert next(csv.reader(stream)) == HEADER


def test_append_missing_rows_adds_only_absent_keys_and_never_edits_a_row(tmp_path):
    paths = _records(tmp_path)
    rows = rep.ledger_rows(paths)
    _ledger(paths.ledger, [["e4_x", "1", "json", "s", "l", "1", "test", "n"]])
    rep.append_rows(paths.ledger, rows[:-3])                                   # the ledger as committed in 51b760b
    before = paths.ledger.read_bytes()
    assert rep.append_missing_rows(paths.ledger, rows) == [row["key"] for row in rows[-3:]]
    data = paths.ledger.read_bytes()
    assert data.startswith(before) and b"\r" not in data and data.count(b"\n") == 2 + len(rows)
    assert rep.append_missing_rows(paths.ledger, rows) == [] and paths.ledger.read_bytes() == data
    edited = data.replace(rows[0]["printed"].encode("utf-8") + b",json", b"9.99999,json", 1)
    assert edited != data
    paths.ledger.write_bytes(edited)
    with pytest.raises(ValueError):                                            # an existing row differs: never edited, refused
        rep.append_missing_rows(paths.ledger, rows)
    paths.ledger.write_bytes(data + b"e1_unknown,1,json,s,l,1,test,n\n")
    with pytest.raises(ValueError):                                            # an e1_ row the code does not make
        rep.append_missing_rows(paths.ledger, rows)


# ----------------------------------------------------------------------------- the Korean report
def test_report_has_every_required_part_and_no_tab_or_carriage_return(tmp_path):
    paths = _records(tmp_path)
    text = rep.render(paths)
    assert chr(9) not in text and chr(13) not in text
    for heading in ("## 1. 질문과 범위", "## 2. 짝 맞추기 설계", "## 3. 개발 자료 Q_CAL", "## 4. 판정", "## 5. 민감도", "## 6. 보조",
                    "## 7. 짝 맞은 집합의 대표성", "## 8. 정밀도 공개", "## 9. 한계", "## 10. 절차와 점검", "## 11. 산출물과 해시"):
        assert heading in text
    result = json.loads((paths.tog / e1.REPORT_NAME).read_text(encoding="utf-8"))["result"]
    primary = result["primary"]
    assert rep.num(primary["point"]["G_eng"], 5) in text and rep.ci(primary["L2_ci"], 5) in text
    assert f'"{result["decision"]["outcome"]}"' in text and e1.DECISIONS[result["decision"]["outcome"]] in text
    assert "교차 뒤" in text or "교집합" in text                                              # r_0.01 counts are post-intersection
    assert "E2" in text and "E3" in text and "E4" in text and rep.SIGN_NOTE in text
    assert str(paths.tog / e1.REPORT_NAME) in text                                               # Windows paths kept whole
    assert "80" * 32 in text and "b4" * 32 in text                         # working-copy and committed AMENDMENT hashes
    assert "95% 구간 기준" in text                                                              # section 4 qualifier (Minor 2)


def _section(text, number):
    return text.split(f"## {number}. ")[1].split("\n## ")[0]


def test_limitations_set_the_objective_free_values_of_q_cal_and_test_side_by_side(tmp_path):
    paths = _records(tmp_path)
    text = rep.render(paths, revisions=())
    limits = _section(text, 9)
    qcal = json.loads((paths.tog / e1.QCAL_NAME).read_text(encoding="utf-8"))["result"]["sensitivities"]["objective_free"]
    test = json.loads((paths.tog / e1.REPORT_NAME).read_text(encoding="utf-8"))["result"]["sensitivities"]["objective_free"]
    for block in (qcal, test):
        assert f"{rep.num(block['point']['r'], 3)} {rep.ci(block['r_ci'], 3)}" in limits
        assert f"{rep.num(block['point']['L2'], 5)} {rep.ci(block['L2_ci'], 5)}" in limits
        assert f"{rep.count(block['pairs'])}" in limits
    assert "DECISION_LOG" in limits and f"r({rep.num(qcal['point']['r'], 2)})" in limits     # the author's pre-TEST row cited it
    assert "Ruling 61" in limits                                                  # TEST exact-zero and small-W counts stay uncomputed
    assert "+0.14" in limits and "불리" in limits and "자료를 보기 전" in limits     # both directions of the residual imbalance (Opus I1)
    item = int(text.split("(반복 열람, §9 한계 ")[1].split(")")[0])                  # the header's pointer follows the numbering
    assert rep._l2_reading({"L2_ci": [-0.00347, 0.00449]}).endswith("(미결).") and rep._l2_reading({"L2_ci": [0.003, 0.007]}) == "L₂ 구간이 0보다 크다."
    assert rep._r_reading({"r_ci": [0.502, 1.180]}) == "r 구간이 0.75와 1을 모두 포함한다."
    assert rep._r_reading({"r_ci": [1.061, 1.555]}) == "r 구간의 하한(1.061)이 1보다 크다."
    assert rep._r_reading({"r_ci": [0.2, 0.6]}) == "r 구간의 상한(0.600)이 0.75보다 작다."
    assert f"\n{item}. 주 TEST 라벨의 반복 열람" in limits


def test_section_11_explains_the_3c_flag_and_how_to_read_the_ledger_sign_notes(tmp_path):
    paths = _records(tmp_path)
    report = paths.tog / e1.REPORT_NAME
    _rewrite(report, lambda d: d["result"]["decision"].update(outcome="rewrite", placebo_gain_exceeds_engagement_gain=False))
    (paths.tog / rep.STDOUT_LOG).write_text(json.dumps({"done": str(report), "outcome": "rewrite"}) + "\n", encoding="utf-8")
    files = _section(rep.render(paths, revisions=()), 11)
    result = json.loads(report.read_text(encoding="utf-8"))["result"]
    assert "placebo_gain_exceeds_engagement_gain" in files and "거짓" in files and "indistinguishable" in files
    assert rep.ci(result["primary"]["net_ci"], 5) in files                       # the net interval is read beside the flag (Opus M2)
    for quantity in ("L₁", "L₂", "순이득", "r = G_plc / G_eng_size"):            # how to read the sign note on these rows (Opus M3)
        assert quantity in files


def test_a_recorded_rule_3c_note_is_printed(tmp_path):
    paths = _records(tmp_path)
    report = paths.tog / e1.REPORT_NAME

    def indistinguishable(d):
        d["result"]["decision"].update(outcome="indistinguishable", placebo_gain_exceeds_engagement_gain=True, note=e1.NET_NOTE,
                                       default_handling=e1.DEFAULT_HANDLING)

    _rewrite(report, indistinguishable)
    (paths.tog / rep.STDOUT_LOG).write_text(json.dumps({"done": str(report), "outcome": "indistinguishable"}) + "\n", encoding="utf-8")
    text = rep.render(paths, revisions=())
    assert "§5 규칙 3c" in text and e1.DEFAULT_HANDLING in text


def test_write_report_writes_once(tmp_path):
    paths = _records(tmp_path)
    rep.write_report(paths)
    data = paths.report_md.read_bytes()
    assert data.decode("utf-8").startswith("# E1 결과") and b"\r" not in data and data.endswith(b"\n")
    with pytest.raises(FileExistsError):
        rep.write_report(paths)


def test_a_replacement_needs_the_flag_and_a_revision_record_naming_the_current_file(tmp_path):
    paths = _records(tmp_path)
    rep.write_report(paths, revisions=())
    first = sha256_file(paths.report_md)
    record = {"date": "2026-10-13", "label": "수정 1회차", "reason": "검토 반영 이유", "previous_sha256": first,
              "previous_commit": "abc1234"}
    with pytest.raises(FileExistsError):                                       # the guard holds without the flag
        rep.write_report(paths, revisions=(record,))
    with pytest.raises(ValueError):                                            # the flag alone is not enough
        rep.write_report(paths, replace=True, revisions=())
    rep.write_report(paths, replace=True, revisions=(record,))
    text = paths.report_md.read_text(encoding="utf-8")
    assert "검토 반영 이유" in text and first in text and "abc1234" in text and sha256_file(paths.report_md) != first
    with pytest.raises(ValueError):                                            # the record names the replaced version only
        rep.write_report(paths, replace=True, revisions=(record,))
    paths.report_md.unlink()
    with pytest.raises(FileNotFoundError):
        rep.write_report(paths, replace=True, revisions=(record,))


def test_the_revision_records_are_complete():
    assert rep.REVISIONS
    for record in rep.REVISIONS:
        assert set(record) == {"date", "label", "reason", "previous_sha256", "previous_commit"}
        assert len(record["previous_sha256"]) == 64 and all(ch in "0123456789abcdef" for ch in record["previous_sha256"])
        assert record["reason"].strip() and chr(9) not in record["reason"] and chr(13) not in record["reason"]


def _rewrite(path, change):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    change(payload)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("tamper", ["check_after_opening", "lock_cross_check", "environment", "code", "openings_before", "stdout"])
def test_render_stops_when_the_records_disagree(tmp_path, tamper):
    paths = _records(tmp_path)
    report, lock = paths.tog / e1.REPORT_NAME, paths.tog / e1.LOCK_NAME
    if tamper == "check_after_opening":
        _rewrite(report, lambda d: d["result"]["checks"].update(a=False))
    elif tamper == "lock_cross_check":
        _rewrite(lock, lambda d: d["cross_checks"].update(b=False))
    elif tamper == "environment":
        _rewrite(report, lambda d: d["environment"].update(numpy="2.2.0"))
    elif tamper == "code":
        _rewrite(report, lambda d: d["run"]["code_sha256"].update({"studies/tog_experiments_20261012/e1_run.py": "00" * 32}))
    elif tamper == "openings_before":
        _rewrite(paths.openings, lambda d: d["openings"][0].update(utc="2026-10-12T05:00:00+00:00"))
    else:
        (paths.tog / rep.STDOUT_LOG).write_text(json.dumps({"done": str(report), "outcome": "keep_or_other"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rep.render(paths)
    assert not paths.report_md.exists()
