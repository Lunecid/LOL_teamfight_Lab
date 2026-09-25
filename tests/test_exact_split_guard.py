"""Tests for gameplay/split_guard.py (v4-exact plan test 8 and the record-1 rule)."""
from __future__ import annotations

import hashlib
import json

import pytest

from gameplay import split_guard as SG
from gameplay.split_guard import (SELECTION_PATCHES, SplitViolation, assert_patch_access, assert_record_exists,
                                  assert_selection_patches, normalize_patch)


def test_selection_patches_allow_only_1514_1515():
    assert SELECTION_PATCHES == frozenset({"15.14", "15.15"})
    assert assert_selection_patches(["15.14", "15.15"]) == {"15.14", "15.15"}
    assert assert_selection_patches("15.14") == {"15.14"}
    assert assert_selection_patches(("15.15.701.1234",)) == {"15.15"}
    assert assert_selection_patches({"15.14", "15.14.1"}) == {"15.14"}


@pytest.mark.parametrize("bad", [["15.16"], ["15.14", "15.16"], ["16.15"], ["15.13"], ["15.18"], "16.13",
                                 ["15.141"]])
def test_selection_patches_refuse_everything_else(bad):
    with pytest.raises(SplitViolation):
        assert_selection_patches(bad)


def test_selection_patches_refuse_empty_float_and_garbage():
    for bad in ([], set(), ["x"], ["15"], 15.14, [15.14]):
        with pytest.raises(SplitViolation):
            assert_selection_patches(bad)
    assert issubclass(SplitViolation, RuntimeError)


def test_normalize_patch():
    assert normalize_patch("15.14") == "15.14"
    assert normalize_patch("Version 15.14.701.4353") == "15.14"
    assert normalize_patch("16.05") == "16.5"
    with pytest.raises(SplitViolation):
        normalize_patch(15.1)


def test_record_exists_ok_and_hash(tmp_path):
    rec = tmp_path / "record1_20260925.json"
    rec.write_text(json.dumps({"record": 1, "v_choice": "V2"}), encoding="utf-8")
    digest = hashlib.sha256(rec.read_bytes()).hexdigest()
    assert assert_record_exists(rec) == digest
    assert assert_record_exists(str(rec), expected_sha256=digest.upper()) == digest
    with pytest.raises(SplitViolation, match="hash mismatch"):
        assert_record_exists(rec, expected_sha256="0" * 64)
    txt = tmp_path / "record1.txt"
    txt.write_text("sha256 abc\n", encoding="utf-8")
    assert assert_record_exists(txt) == hashlib.sha256(txt.read_bytes()).hexdigest()


def test_record_missing_empty_or_invalid(tmp_path):
    for bad in (None, "", tmp_path / "absent.json", tmp_path):
        with pytest.raises(SplitViolation):
            assert_record_exists(bad)
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(SplitViolation, match="empty"):
        assert_record_exists(empty)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SplitViolation, match="JSON"):
        assert_record_exists(broken)
    for blob in ([], {}, "x"):
        p = tmp_path / "obj.json"
        p.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(SplitViolation):
            assert_record_exists(p)


def test_patch_access_needs_record_for_1516(tmp_path):
    assert assert_patch_access(["15.14", "15.15"]) == {"15.14", "15.15"}      # no record needed
    with pytest.raises(SplitViolation):
        assert_patch_access(["15.16"])
    with pytest.raises(SplitViolation):
        assert_patch_access(["15.14", "15.16"], tmp_path / "nope.json")
    rec = tmp_path / "r1.json"
    rec.write_text(json.dumps({"record": 1}), encoding="utf-8")
    assert assert_patch_access(["15.16", "16.15"], rec) == {"15.16", "16.15"}
    with pytest.raises(SplitViolation):
        assert_patch_access("15.16", rec, expected_sha256="f" * 64)


def test_module_reads_no_engagement_data():
    src = open(SG.__file__, encoding="utf-8").read()
    for bad in ("np.load", "events.json", "match_cache", "D:/"):
        assert bad not in src
