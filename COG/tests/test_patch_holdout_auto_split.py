"""Tests for auto patch-holdout behavior."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fight_types import FightRef
from data.indexing import split_refs_patch_holdout


def _make_refs_by_patches(patches):
    refs = []
    ts = 0
    for i, patch in enumerate(patches):
        # Two matches per patch to keep split-by-match fallback meaningful.
        for j in range(2):
            refs.append(
                FightRef(
                    match_id=f"KR_{i}_{j}",
                    patch=str(patch),
                    t_start=0,
                    t_start_ts=ts,
                )
            )
            ts += 60000
    return refs


def test_patch_holdout_auto_uses_prev_patch_for_val():
    refs = _make_refs_by_patches(["15.10", "15.11", "15.12", "15.13"])

    tr, va, te, meta = split_refs_patch_holdout(
        refs=refs,
        seed=7,
        train_patches=None,
        test_patches=None,
        val_patches=None,
        val_ratio_from_train=0.15,
    )

    assert meta["mode_detail"] == "auto_prev_patch_val"
    assert meta["test_patches"] == ["15.13"]
    assert meta["val_patches"] == ["15.12"]
    assert meta["train_patches"] == ["15.10", "15.11"]

    assert {r.patch for r in te} == {"15.13"}
    assert {r.patch for r in va} == {"15.12"}
    assert {r.patch for r in tr} == {"15.10", "15.11"}


def test_patch_holdout_two_patches_falls_back_to_match_split():
    refs = _make_refs_by_patches(["15.12", "15.13"])

    tr, va, te, meta = split_refs_patch_holdout(
        refs=refs,
        seed=7,
        train_patches=None,
        test_patches=None,
        val_patches=None,
        val_ratio_from_train=0.5,
    )

    assert meta["mode_detail"] == "split_by_match_id"
    assert meta["test_patches"] == ["15.13"]
    assert meta["val_patches"] == []
    assert {r.patch for r in te} == {"15.13"}
    assert all(r.patch != "15.13" for r in tr + va)
