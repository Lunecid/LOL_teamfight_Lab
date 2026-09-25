"""Tests for scripts/exact_v4/ev4_record_lock.py (assert_record1_locked and the guard-patch AST whitelist).

Synthetic records directories live in tmp_path.  The real records/ directory is only read (the pre-decision record
and record 1A are hashed to show they are refused); nothing is written there.  No match data is read.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
WT = Path(__file__).resolve().parents[1]
SCRIPTS = WT / "scripts" / "exact_v4"
for _p in (str(WT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_record_lock as RL  # noqa: E402

from gameplay.split_guard import SplitViolation  # noqa: E402
from tests._ev4_lock_helpers import (PRE_NAME, R1A_NAME, V_SHA, make_records_dir, sha,  # noqa: E402
                                     write_record)


def _ok(p, s, rd, pre):
    return RL.assert_record1_locked(p, s, records_dir=rd, predecisions_sha256=pre)


def test_accepts_a_synthetic_locked_record(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd)
    rec = _ok(p, s, rd, pre)
    assert rec["V_frozen_bundle_sha256"] == V_SHA and rec["status"] == "LOCKED"
    assert _ok(str(p), s.upper(), rd, pre)["record"] == "1"          # str path, upper-case hex accepted


def test_refusals_are_split_violations(tmp_path):
    assert issubclass(RL.RecordNotLocked, SplitViolation)


@pytest.mark.parametrize("name", ["record1_DRAFT_20260927T000000Z.json", "record1_draft_SMOKE_x.json",
                                  "record1a_boundaries_x.json", "RECORD1A_x.json"])
def test_refuses_draft_and_record1a_names(tmp_path, name):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd, name=name)                                  # content is a perfect locked record
    with pytest.raises(RL.RecordNotLocked, match="draft or record 1A"):
        _ok(p, s, rd, pre)


def test_refuses_draft_content_under_a_final_name(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd, edit=lambda b: b.update(record="1 DRAFT - not locked", status="DRAFT",
                                                   is_final_lock=False, usable_as_record1=False))
    with pytest.raises(RL.RecordNotLocked, match="record field"):
        _ok(p, s, rd, pre)


@pytest.mark.parametrize("edit, match", [
    (lambda b: b.update(record=1), "record field"),
    (lambda b: b.update(record="1 "), "record field"),
    (lambda b: b.update(record="1A"), "record field"),
    (lambda b: b.update(status="DRAFT"), "status"),
    (lambda b: b.update(is_final_lock=False), "is_final_lock"),
    (lambda b: b.update(usable_as_record1="true"), "usable_as_record1"),
    (lambda b: b["author_signoff"].update(signed=False), "author_signoff"),
    (lambda b: b["author_signoff"].update(signed_by=" "), "author_signoff"),
    (lambda b: b.update(smoke=True), "smoke"),
    (lambda b: b.pop("smoke"), "smoke"),
    (lambda b: b.pop("V_frozen_bundle_sha256"), "V_frozen_bundle_sha256"),
    (lambda b: b.update(V_frozen_bundle_sha256="C" * 64), "V_frozen_bundle_sha256"),
    (lambda b: b["V"]["V_frozen"].update(bundle_sha256="f" * 64), "disagrees"),
    (lambda b: b["record1a"].update(sha256="0" * 64), "record 1A"),
    (lambda b: b.pop("record1a"), "record 1A"),
    (lambda b: b["records"]["predecisions_pinned"].update(sha256="0" * 64), "pre-decision"),
    (lambda b: b.pop("records"), "pre-decision"),
])
def test_refuses_each_missing_lock_field(tmp_path, edit, match):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd, edit=edit)
    with pytest.raises(RL.RecordNotLocked, match=match):
        _ok(p, s, rd, pre)


def test_refuses_smoke_record(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd, name="record1_SMOKE_20260927T000000Z.json", edit=lambda b: b.update(smoke=True))
    with pytest.raises(RL.RecordNotLocked, match="smoke"):
        _ok(p, s, rd, pre)


def test_refuses_wrong_or_missing_sha(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd)
    with pytest.raises(SplitViolation, match="hash mismatch"):
        _ok(p, "0" * 64, rd, pre)
    for bad in (None, "", "abc"):
        with pytest.raises(RL.RecordNotLocked, match="sha256"):
            _ok(p, bad, rd, pre)


def test_refuses_embedded_files_changed_in_records(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd)
    (rd / R1A_NAME).write_text("{}", encoding="utf-8")                 # record 1A changed after the lock
    with pytest.raises(RL.RecordNotLocked, match="record 1A"):
        _ok(p, s, rd, pre)
    rd2, pre2 = make_records_dir(tmp_path / "b")
    p2, s2 = write_record(rd2)
    with pytest.raises(RL.RecordNotLocked, match="pinned"):             # not the ev4_common pin
        _ok(p2, s2, rd2, "1" * 64)
    (rd2 / PRE_NAME).unlink()
    with pytest.raises(RL.RecordNotLocked, match="pre-decision"):
        _ok(p2, s2, rd2, pre2)


def test_refuses_outside_records_dir_and_subdirs(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p, s = write_record(rd, where=tmp_path / "elsewhere")
    with pytest.raises(RL.RecordNotLocked, match="directly inside"):
        _ok(p, s, rd, pre)
    p, s = write_record(rd, where=rd / "sub")
    with pytest.raises(RL.RecordNotLocked, match="directly inside"):
        _ok(p, s, rd, pre)
    with pytest.raises(RL.RecordNotLocked, match="not found"):
        _ok(rd / "record1_missing.json", "0" * 64, rd, pre)
    with pytest.raises(RL.RecordNotLocked, match="no record 1"):
        _ok(None, "0" * 64, rd, pre)
    # the default records dir is the real one: a perfect record in tmp is refused by location alone
    p, s = write_record(rd)
    with pytest.raises(RL.RecordNotLocked, match="directly inside"):
        RL.assert_record1_locked(p, s)


def test_refuses_a_markdown_summary(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    p = rd / "record1_20260927T000000Z_ko.md"
    p.write_text("# 기록 1\nrecord: 1\nstatus: LOCKED\n", encoding="utf-8")
    with pytest.raises(RL.RecordNotLocked, match=".json record"):
        _ok(p, sha(p), rd, pre)


def test_refuses_a_fit_v_frozen_manifest(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    fm = {"smoke": False, "frozen": True, "V_frozen": {"bundle_sha256": V_SHA}, "V_frozen_bundle_sha256": V_SHA}
    out = tmp_path / "fit_v" / "frozen_manifest.json"
    out.parent.mkdir()
    out.write_text(json.dumps(fm), encoding="utf-8")
    with pytest.raises(RL.RecordNotLocked, match="directly inside"):
        _ok(out, sha(out), rd, pre)
    inside = rd / "frozen_manifest.json"                                 # even copied into records/
    inside.write_text(json.dumps(fm), encoding="utf-8")
    with pytest.raises(RL.RecordNotLocked, match="not a record 1 file"):
        _ok(inside, sha(inside), rd, pre)
    named = rd / "record1_frozen_manifest.json"                          # and renamed: content refused
    named.write_text(json.dumps(fm), encoding="utf-8")
    with pytest.raises(RL.RecordNotLocked, match="record field"):
        _ok(named, sha(named), rd, pre)


def test_refuses_the_pre_decision_record(tmp_path):
    rd, pre = make_records_dir(tmp_path)
    with pytest.raises(RL.RecordNotLocked, match="not a record 1 file"):
        _ok(rd / PRE_NAME, pre, rd, pre)
    copy_ = rd / "record1_predecisions.json"
    copy_.write_bytes((rd / PRE_NAME).read_bytes())
    with pytest.raises(RL.RecordNotLocked, match="record field"):
        _ok(copy_, sha(copy_), rd, pre)


@pytest.mark.skipif(not RL.EC.PREDECISIONS_RECORD.is_file(), reason="real pre-decision record absent")
def test_refuses_the_real_pre_decision_record_and_record1a():
    p = RL.EC.PREDECISIONS_RECORD
    with pytest.raises(RL.RecordNotLocked):
        RL.assert_record1_locked(p, sha(p))
    r1a = sorted(RL.RECORDS_DIR.glob("record1a_*.json"))
    for f in r1a:
        with pytest.raises(RL.RecordNotLocked, match="record 1A"):
            RL.assert_record1_locked(f, sha(f))


# ------------------------------------------------------------------ guard-patch AST whitelist
OLD = '''import json


def check_patch_access(patch, record1=None, record1_sha256=None):
    """old"""
    if patch == "15.14":
        return {"patch": patch}
    return {"patch": patch, "record1": record1}


def other(x):
    return x + 1
'''


def test_only_functions_differ():
    new_body = OLD.replace('    return {"patch": patch, "record1": record1}',
                           '    from ev4_record_lock import assert_record1_locked\n'
                           '    assert_record1_locked(record1, record1_sha256)\n'
                           '    return {"patch": patch, "record1": record1, "locked": True}')
    assert RL.only_functions_differ(OLD, new_body, ["check_patch_access"])
    assert not RL.only_functions_differ(OLD, new_body, ["other"])                      # not whitelisted
    assert RL.only_functions_differ(OLD, OLD + "\n# comment\n", ["check_patch_access"])  # comments are not code
    changed_other = new_body.replace("return x + 1", "return x + 2")
    assert not RL.only_functions_differ(OLD, changed_other, ["check_patch_access"])
    new_sig = new_body.replace("record1_sha256=None):", "record1_sha256=None, force=False):")
    assert not RL.only_functions_differ(OLD, new_sig, ["check_patch_access"])
    new_import = "import os\n" + new_body
    assert not RL.only_functions_differ(OLD, new_import, ["check_patch_access"])
