"""Tests for data/index_split.py -- split functions using FightRef objects."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from collections import Counter
from unittest.mock import patch

from core.fight_types import FightRef, ref_key
from data.index_split import (
    split_refs_random,
    split_refs_group_match,
    split_refs_match_patch_stratified,
    validate_split_label_balance,
)


def _make_refs(n_matches=10, fights_per_match=3, n_patches=3):
    """Helper to create a list of FightRef objects for testing."""
    refs = []
    for i in range(n_matches):
        match_id = f"KR_{1000 + i}"
        patch = f"14.{(i % n_patches) + 1}"
        for j in range(fights_per_match):
            ts = (i * fights_per_match + j) * 60000
            refs.append(FightRef(
                match_id=match_id,
                patch=patch,
                t_start=j,
                t_start_ts=ts,
            ))
    return refs


# =========================================================
# split_refs_random (note: delegates to split_refs_group_match
# when SPLIT_GROUP_BY_MATCH_ID is True)
# =========================================================
class TestSplitRefsRandom:
    def test_all_refs_present_no_duplicates(self):
        refs = _make_refs(n_matches=20, fights_per_match=3)
        # Patch cfg.SPLIT_GROUP_BY_MATCH_ID to False for true random split
        with patch("data.index_split.cfg") as mock_cfg:
            mock_cfg.SPLIT_GROUP_BY_MATCH_ID = False
            tr, va, te, info = split_refs_random(refs, seed=42, ratios=(0.6, 0.2, 0.2))

        all_keys = [ref_key(r) for r in tr + va + te]
        original_keys = [ref_key(r) for r in refs]

        # All refs accounted for
        assert sorted(all_keys) == sorted(original_keys)

        # No duplicates
        assert len(all_keys) == len(set(all_keys))

    def test_empty_input(self):
        tr, va, te, info = split_refs_random([], seed=42)
        assert tr == []
        assert va == []
        assert te == []
        assert info["mode"] == "empty"

    def test_approximate_ratios(self):
        refs = _make_refs(n_matches=50, fights_per_match=2)
        with patch("data.index_split.cfg") as mock_cfg:
            mock_cfg.SPLIT_GROUP_BY_MATCH_ID = False
            tr, va, te, info = split_refs_random(refs, seed=42, ratios=(0.6, 0.2, 0.2))

        total = len(refs)
        # Allow some tolerance (ratios are approximate due to rounding)
        assert abs(len(tr) / total - 0.6) < 0.1
        assert abs(len(va) / total - 0.2) < 0.1
        assert abs(len(te) / total - 0.2) < 0.1


# =========================================================
# split_refs_group_match
# =========================================================
class TestSplitRefsGroupMatch:
    def test_same_match_same_split(self):
        """Refs from the same match must all be in the same split (no leakage)."""
        refs = _make_refs(n_matches=20, fights_per_match=5)
        tr, va, te, info = split_refs_group_match(refs, seed=42, ratios=(0.6, 0.2, 0.2))

        # Build match -> split mapping
        match_splits = {}
        for split_name, split_refs in [("train", tr), ("val", va), ("test", te)]:
            for r in split_refs:
                if r.match_id in match_splits:
                    assert match_splits[r.match_id] == split_name, (
                        f"Match {r.match_id} appears in both "
                        f"{match_splits[r.match_id]} and {split_name}"
                    )
                else:
                    match_splits[r.match_id] = split_name

    def test_all_refs_present(self):
        refs = _make_refs(n_matches=15, fights_per_match=3)
        tr, va, te, info = split_refs_group_match(refs, seed=7, ratios=(0.6, 0.2, 0.2))

        all_keys = set(ref_key(r) for r in tr + va + te)
        original_keys = set(ref_key(r) for r in refs)
        assert all_keys == original_keys

    def test_empty_input(self):
        tr, va, te, info = split_refs_group_match([], seed=42)
        assert info["mode"] == "empty"


# =========================================================
# split_refs_match_patch_stratified
# =========================================================
class TestSplitRefsMatchPatchStratified:
    def test_each_patch_in_at_least_one_split(self):
        """Each patch should appear in at least one split."""
        refs = _make_refs(n_matches=30, fights_per_match=2, n_patches=5)
        tr, va, te, info = split_refs_match_patch_stratified(
            refs, seed=42, ratios=(0.6, 0.2, 0.2)
        )

        patches_in_output = set()
        for r in tr + va + te:
            patches_in_output.add(r.patch)

        patches_in_input = set(r.patch for r in refs)
        assert patches_in_output == patches_in_input

    def test_all_refs_present_no_duplicates(self):
        refs = _make_refs(n_matches=20, fights_per_match=3, n_patches=4)
        tr, va, te, info = split_refs_match_patch_stratified(
            refs, seed=42, ratios=(0.6, 0.2, 0.2)
        )

        all_keys = [ref_key(r) for r in tr + va + te]
        original_keys = [ref_key(r) for r in refs]
        assert sorted(all_keys) == sorted(original_keys)
        assert len(all_keys) == len(set(all_keys))

    def test_no_match_leakage(self):
        """Refs from the same match should be in the same split."""
        refs = _make_refs(n_matches=20, fights_per_match=4, n_patches=3)
        tr, va, te, info = split_refs_match_patch_stratified(
            refs, seed=42, ratios=(0.6, 0.2, 0.2)
        )

        match_splits = {}
        for split_name, split_refs in [("train", tr), ("val", va), ("test", te)]:
            for r in split_refs:
                if r.match_id in match_splits:
                    assert match_splits[r.match_id] == split_name
                else:
                    match_splits[r.match_id] = split_name

    def test_empty_input(self):
        tr, va, te, info = split_refs_match_patch_stratified([], seed=42)
        assert info["mode"] == "empty"


# =========================================================
# validate_split_label_balance
# =========================================================
class TestValidateSplitLabelBalance:
    def test_returns_expected_structure(self):
        refs = _make_refs(n_matches=10, fights_per_match=2)
        mid = len(refs) // 3
        tr = refs[:mid]
        va = refs[mid:2*mid]
        te = refs[2*mid:]

        result = validate_split_label_balance(tr, va, te)
        assert "train_n" in result
        assert "val_n" in result
        assert "test_n" in result
        assert "balanced" in result

    def test_with_label_fn(self):
        refs = _make_refs(n_matches=10, fights_per_match=2)
        mid = len(refs) // 3
        tr = refs[:mid]
        va = refs[mid:2*mid]
        te = refs[2*mid:]

        # Simple label function: alternate 0/1
        label_fn = lambda r: 1 if r.t_start % 2 == 0 else 0

        result = validate_split_label_balance(tr, va, te, label_fn=label_fn)
        assert "train_pos_rate" in result
        assert "test_pos_rate" in result
        assert "label_drift" in result
        assert isinstance(result["balanced"], bool)

    def test_without_label_fn(self):
        refs = _make_refs(n_matches=6, fights_per_match=1)
        tr = refs[:2]
        va = refs[2:4]
        te = refs[4:]

        result = validate_split_label_balance(tr, va, te, label_fn=None)
        # Without label_fn, rates are nan, balanced defaults to True
        assert result["balanced"] is True
