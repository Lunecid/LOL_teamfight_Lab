"""E6 part B documents: the Korean E6 report and the e6_ ledger rows (studies/tog_experiments_20261012/e6_report.py), on
synthetic records whose result block comes from e6_run.evaluate (the production structure).

Binding: PRESPEC_E6_KO.md sections 6, 7 and 12 (ledger keys and schema, sign, interval method and status notes, no 'abs'
outside magnitude rows, the decision wording); SDD Ruling 60 (e6_run.py unchanged).  No test reads a real record or a
sealed file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from reports.thesis_manuscript_20261010.ledger import verify_ledger
from studies.tog_experiments_20261012 import e6_report as rep, e6_run as E6

HEADER = ["key", "printed", "kind", "source", "locator", "scale", "dataset", "note"]
LOCK_SHA = "c" * 64


def _result(seed=0, players=True):
    rng = np.random.default_rng(seed)
    n_match = 60
    match = np.repeat(np.arange(n_match), 4)
    y = rng.integers(0, 2, match.size).astype(np.float64)

    def p(strength):
        return np.clip(0.5 + strength * (2 * y - 1) + rng.normal(0, 0.1, y.size), 0.02, 0.98)

    preds = {E6.P1: p(0.3), E6.PT: p(0.1), E6.S10: p(0.28), E6.S00: p(0.25), E6.S11: p(0.29), E6.S01: p(0.26)}
    n_kills = rng.integers(1, 6, y.size)
    if players:
        idx = np.stack([rng.choice(150, 10, replace=False) for _ in range(n_match)])     # ten distinct players a match
        pin = E6.PlayerInput(True, None, np.arange(n_match), idx, None)
    else:
        pin = E6.PlayerInput(False, "incomplete_identification_over_1pct", None, None, None)
    res = E6.evaluate(y, preds, match, n_kills, np.concatenate([n_kills, [2, 2, 5]]), pin, s_scored=True)
    res["scale"] = {"q_rows": int(y.size) + 3, "ties": 3, "non_tie_rows": int(y.size), "matches": res["matches"]}
    res["branch"] = {"branch": "가" if players else "나", "primary_interval": res["primary_interval"],
                     "twoway_unavailable_reason": None if players else "incomplete_identification_over_1pct",
                     "incomplete_identification": {"available": True, "incomplete_matches": 0 if players else 2,
                                                   "q_row_matches": n_match, "share": 0.0 if players else 2 / n_match}}
    res["selected"] = 64
    res["stage_counts"] = {"build": {"built": 64, "failed": 0}, "audit": {"audited": 64, "audit_fail": 0},
                           "bank": {"bank_matches": 64, "excluded": 0}, "engagements": {"engagements": int(y.size) + 3},
                           "e2c": {"q_rows": int(y.size) + 3, "q_row_matches": n_match}}
    return res


def _tree(tmp_path, result):
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    lock_path = out / "E6_LOCK_r2.json"
    lock = {"locked_utc": "2026-10-14T07:41:49.000000+00:00",
            "verify": {f"code:{E6.CODE_REL}": {"sha256": "d" * 64}},
            "previous_locks": [{"path": str(out / "E6_LOCK.json"), "sha256": "a" * 64, "opened": False,
                                "superseded_before_any_look": True},
                               {"path": str(out / "E6_LOCK_r1.json"), "sha256": "b" * 64, "opened": False,
                                "superseded_before_any_look": True}]}
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha = E6.sha256_file(lock_path)
    before = [{"experiment": "KA_POSTHOC", "attempt": 1, "utc": "2026-10-13T17:46:22.217134+00:00", "lock_sha256": "e" * 64}]
    entry = {"experiment": "E6", "attempt": 1, "utc": "2026-10-14T07:49:25.287150+00:00", "lock": str(lock_path),
             "lock_sha256": lock_sha, "prespec_sha256": "f" * 64, "labels": [str(out / "KA_LABELS_E6_SEALED.npz")],
             "sealed_sha256": "9" * 64, "note": E6.OPENING_NOTE}
    openings = tmp_path / "OPENINGS.json"
    openings.write_text(json.dumps({"openings": before + [entry]}), encoding="utf-8")
    (out / "E6_OPENING.json").write_text(json.dumps({**entry, "openings_before": before}), encoding="utf-8")
    report = {"version": E6.VERSION, "experiment": "E6", "status": "completed", "attempt": 1, "opening": entry,
              "openings_before": before, "lock": {"path": str(lock_path), "sha256": lock_sha, "locked_utc": lock["locked_utc"]},
              "prespec": {"commit": "86e1541", "blob_sha256": "f" * 64}, "result": result,
              "pre_exposure_checks": {"lock": "held", "inputs": {"rows": result["scale"]["q_rows"]},
                                      "reproduction": {"equal": True, "error": None}},
              "environment": {"python": "3.13.5", "numpy": "2.1.3"}, "started_utc": "2026-10-14T07:49:11.000000+00:00",
              "finished_utc": "2026-10-14T07:49:29.000000+00:00", "wall_seconds": 17.7, "peak_memory_bytes": 984219648,
              "captured_output_after_opening": "", "ledger_writer": E6.SINGLE_WRITER_NOTE,
              "run": {"seed": 20261011, "rows": result["rows"], "matches": result["matches"],
                      "code_sha256": {E6.CODE_REL: "d" * 64}}}
    (out / "E6_REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    (out / "logs" / "E6_TEST.stdout.log").write_text("E6 TEST written (attempt 1)\n", encoding="utf-8")
    (out / "logs" / "E6_TEST.stderr.log").write_bytes(b"")
    (out / "logs" / "E6_TEST.exit.txt").write_text("exit=0 utc=2026-10-14T07:49:29Z\n", encoding="utf-8")
    ledger = tmp_path / "TOG_NUMBERS.csv"
    ledger.write_bytes((",".join(HEADER) + "\n").encode("utf-8"))                       # LF, as the real ledger
    return rep.Paths(out=out, openings=openings, ledger=ledger, report_md=tmp_path / "E6_REPORT_KO.md")


def _expected_keys(players):
    keys = {"e6_p1", "e6_p1_lo", "e6_p1_hi", "e6_brier_p1", "e6_brier_pt", "e6_selected", "e6_q_rows", "e6_rows", "e6_ties",
            "e6_matches", "e6_s_meand", "e6_s_meand_lo", "e6_s_meand_hi"}
    keys |= {f"e6_auc_{c}" for c in ("p1", "pt", "s10", "s00", "s11", "s01")}
    keys |= {f"e6_{k}_p1{s}" for k in ("k1", "k23", "k4") for s in ("", "_lo", "_hi")}
    keys |= {f"e6_{k}_{s}" for k in ("k1", "k23", "k4") for s in ("rows", "ties")}
    if players:
        keys |= {"e6_p1_boot_lo", "e6_p1_boot_hi", "e6_p1_de", "e6_players", "e6_s_meand_boot_lo", "e6_s_meand_boot_hi"}
    return keys


@pytest.mark.parametrize("players", [True, False])
def test_ledger_rows_follow_the_prespec_keys_and_round_trip_the_thesis_verifier(tmp_path, players):
    paths = _tree(tmp_path, _result(players=players))
    records = rep.load(paths)
    rep.check(records)
    rows = rep.ledger_rows(records)
    assert {r["key"] for r in rows} == _expected_keys(players) and len(rows) == len(_expected_keys(players))
    rep.append_rows(paths.ledger, rows)
    assert verify_ledger.main(paths.ledger) == 0
    with paths.ledger.open(encoding="utf-8", newline="") as stream:
        written = list(csv.DictReader(stream))
    assert [r["key"] for r in written] == [r["key"] for r in rows]
    for r in written:
        assert r["kind"] == "json" and r["dataset"] == "external_e6" and r["source"] == str(paths.report_json)
        assert Path(r["source"]).is_absolute() and "abs" not in r["note"]
        assert "PRESPEC_E6_KO.md" in r["note"]
    signed = {r["key"]: r for r in written if r["key"] in ("e6_p1", "e6_p1_lo", "e6_p1_hi", "e6_k4_p1")}
    assert all("negative = LightGBM better" in r["note"] for r in signed.values())
    method = "two-way" if players else "match bootstrap"
    assert method in signed["e6_p1_lo"]["note"] and "kill-record strata" in signed["e6_k4_p1"]["note"]
    lo = next(r for r in written if r["key"] == "e6_s_meand_lo")
    assert lo["locator"].endswith(("twoway/interval/0" if players else "bootstrap/interval/0"))


@pytest.mark.parametrize("decision,interval,word", [("confirmed", (-0.03, -0.01), "확증"),
                                                     ("unresolved", (-0.01, 0.01), "미결"),
                                                     ("opposite", (0.01, 0.03), "반대 방향")])
def test_the_report_states_the_pre_registered_decision_and_never_no_effect(tmp_path, decision, interval, word):
    result = _result()
    result["primary"]["interval"] = list(interval)
    result["twoway"]["interval"] = list(interval)
    result["bootstrap"]["interval"] = list(interval)
    result["decision"], result["decision_text"] = decision, E6.DECISION_TEXT[decision]
    result["decisions_by_interval"] = {"match_bootstrap": decision, "twoway": decision}
    paths = _tree(tmp_path, result)
    records = rep.load(paths)
    rep.check(records)
    text = rep.render(records)
    assert f"판정: **{word}**" in text
    for phrase in ("효과 없음", "효과가 없", "no effect", "사라", "재현했"):
        assert phrase not in text
    assert chr(9) not in text and chr(13) not in text
    assert "킬 기록으로 고른" in text and "Ruling 69" in text and "열람 전" in text
    assert "E6 TEST written (attempt 1)" in text and "0바이트" in text


def test_check_refuses_records_that_disagree(tmp_path):
    paths = _tree(tmp_path, _result())
    records = rep.load(paths)
    rep.check(records)
    bad = json.loads(json.dumps(records["report"]))
    bad["result"]["decision"] = "unresolved"
    with pytest.raises(ValueError):
        rep.check({**records, "report": bad})
    bad = json.loads(json.dumps(records["report"]))
    bad["opening"]["utc"] = "2026-10-14T00:00:00+00:00"
    with pytest.raises(ValueError):
        rep.check({**records, "report": bad})
    with pytest.raises(ValueError):
        rep.check({**records, "lock_sha256": "0" * 64})
    bad = json.loads(json.dumps(records["report"]))
    bad["pre_exposure_checks"]["reproduction"]["equal"] = False
    with pytest.raises(ValueError):
        rep.check({**records, "report": bad})


def test_documents_are_written_once(tmp_path):
    paths = _tree(tmp_path, _result())
    records = rep.load(paths)
    rows = rep.ledger_rows(records)
    rep.append_rows(paths.ledger, rows)
    with pytest.raises(FileExistsError):
        rep.append_rows(paths.ledger, rows)
    rep.write_report(paths.report_md, rep.render(records))
    with pytest.raises(FileExistsError):
        rep.write_report(paths.report_md, "again\n")
    with pytest.raises(ValueError):
        rep.write_report(tmp_path / "other.md", "a" + chr(9) + "b\n")
    assert b"\r" not in paths.report_md.read_bytes() and b"\r" not in paths.ledger.read_bytes()
