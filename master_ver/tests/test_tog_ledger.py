"""ToG ledger: the thesis ledger schema and rules, plus an exact header and unique keys (Ruling 10)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reports.tog_paper_20261011.ledger.verify_tog_numbers import main

HEADER = "key,printed,kind,source,locator,scale,dataset,note"
SCRIPT = Path(__file__).resolve().parents[1] / "reports" / "tog_paper_20261011" / "ledger" / "verify_tog_numbers.py"


def _row(key, printed, kind, source, locator, scale="1", note=""):
    return [key, printed, kind, str(source), locator, scale, "test", note]


def _ledger(tmp_path, rows, header=HEADER):
    path = tmp_path / "ledger.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(header + "\n")
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return path


def _json(tmp_path, data):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("header", [
    "name,value,source,field,note",                        # the retired five-column schema
    "key,printed,kind,source,locator,scale,dataset",       # a column missing
    "key,printed,kind,locator,source,scale,dataset,note",  # columns out of order
    "\ufeff" + HEADER,                                     # a BOM: the thesis verifier reads plain UTF-8
])
def test_header_must_be_exact(tmp_path, capsys, header):
    ledger = _ledger(tmp_path, [_row("x", "0.5", "json", _json(tmp_path, {"x": 0.5}), "x")], header)
    assert main(ledger) == 1
    assert capsys.readouterr().out.startswith("FAIL header:")


def test_duplicate_keys_are_rejected_before_any_row_is_checked(tmp_path, capsys):
    src = _json(tmp_path, {"a": 0.5, "b": 0.25})
    ledger = _ledger(tmp_path, [_row("k", "0.5", "json", src, "a"), _row("k", "0.25", "json", src, "b")])
    assert main(ledger) == 1
    assert capsys.readouterr().out.splitlines() == ["FAIL duplicate key 'k' on lines 2, 3", "0 rows checked, 1 failed"]


def test_json_locator_reaches_a_key_that_contains_a_dot(tmp_path):
    src = _json(tmp_path, {"patches": {"15.14": {"n": 74673}}})
    assert main(_ledger(tmp_path, [_row("n_1514", "74673", "json", src, "patches/15.14/n")])) == 0
    assert main(_ledger(tmp_path, [_row("n_1514", "74674", "json", src, "patches/15.14/n")])) == 1


@pytest.mark.parametrize("printed, expected", [("67.0", 0), ("67.0%", 0), ("67.1", 1), ("0.670", 1)])
def test_scale_100_checks_a_stored_fraction_against_a_printed_percentage(tmp_path, printed, expected):
    src = _json(tmp_path, {"auc": 0.67041})
    assert main(_ledger(tmp_path, [_row("auc_pct", printed, "json", src, "auc", scale="100")])) == expected


@pytest.mark.parametrize("value, printed, expected", [
    (74673, "74,673", 0),
    (74673, "74{,}673", 0),        # the LaTeX form of the separator
    (74673, "74,674", 1),
    (-0.0139, "\u22120.0139", 0),  # U+2212 minus sign
    (-0.0139, "0.0139", 1),
    (-0.0139, "\u22120.0140", 1),
])
def test_printed_value_with_thousands_separator_or_unicode_minus(tmp_path, value, printed, expected):
    src = _json(tmp_path, {"x": value})
    assert main(_ledger(tmp_path, [_row("x", printed, "json", src, "x")])) == expected


@pytest.mark.parametrize("value, printed, expected", [
    (0.285, "0.28", 0),        # a decimal tie: both neighbours pass
    (0.285, "0.29", 0),
    (0.125, "0.12", 0),        # a tie that is exact in binary: both neighbours pass
    (0.125, "0.13", 0),
    (0.285000001, "0.28", 1),  # just beyond half a unit
    (0.284999999, "0.29", 1),
])
def test_tolerance_is_half_a_unit_and_accepts_both_neighbours_of_a_tie(tmp_path, value, printed, expected):
    src = _json(tmp_path, {"x": value})
    assert main(_ledger(tmp_path, [_row("x", printed, "json", src, "x")])) == expected


@pytest.mark.parametrize("kind", ["memo", "spec"])
def test_memo_and_spec_rows_need_their_locator_in_the_source_text(tmp_path, capsys, kind):
    notes = tmp_path / "notes.md"
    notes.write_text("15.15 수집 74,748 (분석 74,747)\n", encoding="utf-8")
    assert main(_ledger(tmp_path, [_row("n", "74,747", kind, notes, "74,748 (분석 74,747)")])) == 0
    assert main(_ledger(tmp_path, [_row("n", "74,747", kind, notes, "74,748 (분석 74,746)")])) == 1
    assert "not found" in capsys.readouterr().out


def test_derived_row_combines_other_rows_by_key(tmp_path):
    src = _json(tmp_path, {"p1": 0.6704, "p2": 0.6512})
    inputs = [_row("auc_p1", "0.6704", "json", src, "p1"), _row("auc_p2", "0.6512", "json", src, "p2")]
    gap = "v('auc_p1') - v('auc_p2')"
    assert main(_ledger(tmp_path, [*inputs, _row("gap", "0.019", "derived", "", gap)])) == 0
    assert main(_ledger(tmp_path, [*inputs, _row("gap", "0.020", "derived", "", gap)])) == 1


def test_abs_note_compares_magnitudes_only(tmp_path):
    src = _json(tmp_path, {"contrast": -0.0213})
    assert main(_ledger(tmp_path, [_row("size", "0.0213", "json", src, "contrast", note="abs")])) == 0
    assert main(_ledger(tmp_path, [_row("size", "0.0213", "json", src, "contrast")])) == 1


def test_cli_exit_codes(tmp_path):
    def run(ledger):
        return subprocess.run([sys.executable, str(SCRIPT), str(ledger)], cwd=tmp_path, capture_output=True, text=True)

    passed = run(_ledger(tmp_path, []))
    assert (passed.returncode, passed.stdout, passed.stderr) == (0, "0 rows checked, 0 failed\n", "")
    src = _json(tmp_path, {"x": -0.0139})
    failed = run(_ledger(tmp_path, [_row("x", "\u22120.0140", "json", src, "x")]))
    # the U+2212 in the failure line must not crash a console encoding that lacks it
    assert (failed.returncode, failed.stderr) == (1, "")
    assert failed.stdout.splitlines()[-1] == "1 rows checked, 1 failed"
