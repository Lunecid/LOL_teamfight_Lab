"""Tests for core/fight_types.py -- FightRef construction and ref_key()."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.fight_types import FightRef, PruneSpec, ref_key


class TestFightRefConstruction:
    """Test FightRef dataclass construction with all fields."""

    def test_basic_construction(self):
        ref = FightRef(
            match_id="KR_7123456789",
            patch="14.10",
            t_start=5,
            t_start_ts=300000,
            label_end_ts=360000,
        )
        assert ref.match_id == "KR_7123456789"
        assert ref.patch == "14.10"
        assert ref.t_start == 5
        assert ref.t_start_ts == 300000
        assert ref.label_end_ts == 360000

    def test_defaults(self):
        ref = FightRef(match_id="KR_1", patch="14.1", t_start=3)
        assert ref.t_start_ts == -1
        assert ref.label_end_ts == -1

    def test_post_init_resets_invalid_label_end(self):
        """label_end_ts <= t_start_ts should be reset to -1."""
        ref = FightRef(
            match_id="KR_1",
            patch="14.1",
            t_start=3,
            t_start_ts=300000,
            label_end_ts=200000,
        )
        assert ref.label_end_ts == -1

    def test_post_init_equal_timestamps_reset(self):
        """label_end_ts == t_start_ts should be reset to -1."""
        ref = FightRef(
            match_id="KR_1",
            patch="14.1",
            t_start=3,
            t_start_ts=300000,
            label_end_ts=300000,
        )
        assert ref.label_end_ts == -1

    def test_post_init_valid_label_end_preserved(self):
        """Valid label_end_ts > t_start_ts should be preserved."""
        ref = FightRef(
            match_id="KR_1",
            patch="14.1",
            t_start=3,
            t_start_ts=300000,
            label_end_ts=400000,
        )
        assert ref.label_end_ts == 400000


class TestRefKey:
    """Test ref_key() produces unique, stable keys."""

    def test_key_with_t_start_ts(self):
        ref = FightRef(match_id="KR_123", patch="14.10", t_start=5, t_start_ts=300000)
        key = ref_key(ref)
        assert key == "KR_123|t_start_ts=300000"

    def test_key_without_t_start_ts_uses_t_start(self):
        ref = FightRef(match_id="KR_123", patch="14.10", t_start=5, t_start_ts=-1)
        key = ref_key(ref)
        assert key == "KR_123|t_start=5"

    def test_key_stability(self):
        """Same inputs should always produce the same key."""
        ref = FightRef(match_id="KR_ABC", patch="15.1", t_start=10, t_start_ts=600000)
        assert ref_key(ref) == ref_key(ref)

    def test_key_uniqueness_different_matches(self):
        ref_a = FightRef(match_id="KR_1", patch="14.10", t_start=5, t_start_ts=300000)
        ref_b = FightRef(match_id="KR_2", patch="14.10", t_start=5, t_start_ts=300000)
        assert ref_key(ref_a) != ref_key(ref_b)

    def test_key_uniqueness_different_timestamps(self):
        ref_a = FightRef(match_id="KR_1", patch="14.10", t_start=5, t_start_ts=300000)
        ref_b = FightRef(match_id="KR_1", patch="14.10", t_start=5, t_start_ts=300001)
        assert ref_key(ref_a) != ref_key(ref_b)

    def test_key_negative_t_start_ts_falls_back(self):
        """Negative t_start_ts should fall back to t_start."""
        ref = FightRef(match_id="KR_1", patch="14.10", t_start=7, t_start_ts=-1)
        key = ref_key(ref)
        assert "t_start=7" in key
        assert "t_start_ts" not in key

    def test_key_zero_t_start_ts(self):
        """t_start_ts == 0 is valid (game start) and should be used."""
        ref = FightRef(match_id="KR_1", patch="14.10", t_start=0, t_start_ts=0)
        key = ref_key(ref)
        assert key == "KR_1|t_start_ts=0"

    def test_key_negative_t_start(self):
        """Edge case: negative t_start with no t_start_ts."""
        ref = FightRef(match_id="KR_1", patch="14.10", t_start=-1, t_start_ts=-1)
        key = ref_key(ref)
        assert key == "KR_1|t_start=-1"


class TestPruneSpec:
    """Basic PruneSpec tests."""

    def test_default_none(self):
        spec = PruneSpec()
        assert spec.x_keep is None
        assert spec.extra_keep is None
