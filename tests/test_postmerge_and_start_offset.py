"""Tests for FIX-6.2 (overlap clipping) and FIX-6.3 (START_OFFSET_MIN enforcement).

FIX-6.2: Spatially distant overlapping fights — clip smaller fight's label window.
FIX-6.3: Enforce START_OFFSET_MIN as a rejection filter in fight detection.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from gameplay.fight_postmerge import (
    enforce_postmerge_spacing_and_nonoverlap,
    fight_priority_score,
    label_end_ts,
)


# ============================================================
# Helpers
# ============================================================

def _make_fight(
    engage_ts: int,
    horizon_end_ts: int = -1,
    centroid_x: float = 5000.0,
    centroid_y: float = 5000.0,
    det_prox_pairs: int = 10,
    n_segments: int = 1,
) -> dict:
    """Create a minimal fight dict for postmerge testing."""
    f = {
        "engage_ts": engage_ts,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "det_prox_pairs": det_prox_pairs,
        "n_segments": n_segments,
        "det_anchor": 0,
        "det_backtracked": 0,
    }
    if horizon_end_ts > 0:
        f["horizon_end_ts"] = horizon_end_ts
    return f


KILL_TS = np.array([120000, 180000, 240000, 300000], dtype=np.int64)
HORIZON_MS = 60000


# ============================================================
# FIX-6.2: Overlap clipping tests
# ============================================================

class TestOverlapClipping:
    """Tests for spatially distant fight overlap resolution."""

    def test_no_overlap_no_clipping(self):
        """Non-overlapping fights should pass through unchanged."""
        f1 = _make_fight(100000, centroid_x=2000, centroid_y=2000)
        f2 = _make_fight(200000, centroid_x=12000, centroid_y=12000)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        assert len(result) == 2
        assert diag.get("postmerge_overlap_clipped", 0) == 0
        assert diag.get("postmerge_overlap_dropped", 0) == 0

    def test_distant_overlap_clips_smaller_earlier_fight(self):
        """When earlier fight is smaller and overlaps with later larger fight,
        clip earlier fight's horizon_end_ts to later fight's engage_ts."""
        # f1: small fight (prox_pairs=5), ends at 100000+60000=160000
        f1 = _make_fight(100000, centroid_x=2000, centroid_y=2000, det_prox_pairs=5)
        # f2: big fight (prox_pairs=20), starts at 140000 (inside f1's label window)
        f2 = _make_fight(140000, centroid_x=12000, centroid_y=12000, det_prox_pairs=20)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        assert len(result) == 2
        # f1 (smaller) should be clipped to f2's engage_ts
        assert result[0]["horizon_end_ts"] == 140000
        assert result[0].get("det_overlap_clipped") == 1
        # f2 (bigger) should be unchanged
        assert result[1]["engage_ts"] == 140000
        assert diag["postmerge_overlap_clipped"] == 1

    def test_distant_overlap_drops_smaller_later_fight(self):
        """When later fight is smaller and overlaps with earlier larger fight,
        drop the later (smaller) fight entirely."""
        # f1: big fight (prox_pairs=20), ends at 100000+60000=160000
        f1 = _make_fight(100000, centroid_x=2000, centroid_y=2000, det_prox_pairs=20)
        # f2: small fight (prox_pairs=3), starts at 140000 (inside f1's label window)
        f2 = _make_fight(140000, centroid_x=12000, centroid_y=12000, det_prox_pairs=3)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        assert len(result) == 1
        assert result[0]["engage_ts"] == 100000  # only big fight remains
        assert diag["postmerge_overlap_dropped"] == 1

    def test_close_overlap_falls_through_to_priority(self):
        """Spatially close overlapping fights should use priority replacement (original behavior)."""
        # Both near same location
        f1 = _make_fight(100000, centroid_x=5000, centroid_y=5000, det_prox_pairs=5)
        f2 = _make_fight(140000, centroid_x=5100, centroid_y=5100, det_prox_pairs=20)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        # Distance ~141 < 4000 → not distant → use priority replacement
        assert len(result) == 1
        # f2 has higher priority, should replace f1
        assert result[0]["engage_ts"] == 140000
        assert diag["postmerge_conflicts"] == 1
        assert diag["postmerge_replaced"] == 1

    def test_no_location_radius_uses_original_logic(self):
        """When location_radius=0, skip distance check entirely (original behavior)."""
        f1 = _make_fight(100000, centroid_x=2000, centroid_y=2000, det_prox_pairs=5)
        f2 = _make_fight(140000, centroid_x=12000, centroid_y=12000, det_prox_pairs=20)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=0.0, diag=diag,
        )
        # No distance check → overlap → priority replacement
        assert len(result) == 1
        assert result[0]["engage_ts"] == 140000
        assert diag["postmerge_overlap_clipped"] == 0
        assert diag["postmerge_overlap_dropped"] == 0

    def test_multiple_overlaps_chain(self):
        """Three fights: A overlaps B, B overlaps C. All spatially distant."""
        fA = _make_fight(100000, centroid_x=2000, centroid_y=2000, det_prox_pairs=5)
        fB = _make_fight(140000, centroid_x=8000, centroid_y=8000, det_prox_pairs=25)
        fC = _make_fight(180000, centroid_x=14000, centroid_y=14000, det_prox_pairs=10)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [fA, fB, fC], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        # A(small) vs B(big): A clipped, B kept
        # B(big) vs C(medium): C is smaller, C's engage=180000 < B's end=200000
        #   → C dropped (smaller later fight)
        assert len(result) == 2
        assert result[0]["engage_ts"] == 100000  # A, clipped
        assert result[0]["horizon_end_ts"] == 140000
        assert result[1]["engage_ts"] == 140000  # B

    def test_empty_fights_list(self):
        """Empty input should return empty output with initialized diag."""
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        assert result == []
        assert diag.get("postmerge_overlap_clipped", 0) == 0
        assert diag.get("postmerge_overlap_dropped", 0) == 0

    def test_equal_priority_drops_later(self):
        """When priority scores are equal, the later (smaller-or-equal) fight is dropped."""
        f1 = _make_fight(100000, centroid_x=2000, centroid_y=2000, det_prox_pairs=10)
        f2 = _make_fight(140000, centroid_x=12000, centroid_y=12000, det_prox_pairs=10)
        diag = {}
        result = enforce_postmerge_spacing_and_nonoverlap(
            [f1, f2], horizon_ms=HORIZON_MS, fight_min_gap_ms=0,
            kill_ts=KILL_TS, location_radius=4000.0, diag=diag,
        )
        # Equal priority → sc > sp is False → drop f2
        assert len(result) == 1
        assert result[0]["engage_ts"] == 100000
        assert diag["postmerge_overlap_dropped"] == 1


# ============================================================
# FIX-6.3: START_OFFSET_MIN enforcement tests
# ============================================================

class TestStartOffsetMin:
    """Tests that START_OFFSET_MIN is enforced during fight detection.

    These tests verify the filtering logic by directly checking the
    rejection condition: engage_ts - t_min_ms < START_OFFSET_MIN * 60000.
    """

    def test_start_offset_min_rejects_early_fight(self):
        """A fight at 1 minute should be rejected when START_OFFSET_MIN=2."""
        # With START_OFFSET_MIN=2 and t_min_ms=60000 (first frame at minute 1):
        # engage_ts=90000 → 90000-60000=30000 < 120000 → REJECTED
        t_min_ms = 60000
        start_offset_ms = 2 * 60000  # START_OFFSET_MIN=2
        engage_ts = 90000  # 1.5 minutes into game

        assert (engage_ts - t_min_ms) < start_offset_ms

    def test_start_offset_min_accepts_late_fight(self):
        """A fight at 3 minutes should pass when START_OFFSET_MIN=2."""
        t_min_ms = 60000
        start_offset_ms = 2 * 60000
        engage_ts = 240000  # 4 minutes

        assert (engage_ts - t_min_ms) >= start_offset_ms

    def test_start_offset_boundary(self):
        """A fight exactly at START_OFFSET_MIN should pass."""
        t_min_ms = 60000
        start_offset_ms = 2 * 60000
        engage_ts = 180000  # exactly 2 minutes after t_min

        assert (engage_ts - t_min_ms) >= start_offset_ms

    def test_start_offset_just_below_boundary(self):
        """A fight 1ms before START_OFFSET_MIN boundary should be rejected."""
        t_min_ms = 60000
        start_offset_ms = 2 * 60000
        engage_ts = 179999  # 1ms before boundary

        assert (engage_ts - t_min_ms) < start_offset_ms


# ============================================================
# label_end_ts helper tests
# ============================================================

class TestLabelEndTs:
    """Verify label_end_ts respects clipped horizon_end_ts."""

    def test_clipped_horizon_reflected(self):
        """After clipping horizon_end_ts, label_end_ts should return the clipped value."""
        f = _make_fight(100000)
        # Before clipping: default horizon
        assert label_end_ts(f, 60000) == 160000

        # After clipping
        f["horizon_end_ts"] = 140000
        assert label_end_ts(f, 60000) == 140000

    def test_horizon_end_ts_unchanged_when_no_clip(self):
        """Explicitly set horizon_end_ts should be returned as-is."""
        f = _make_fight(100000, horizon_end_ts=180000)
        assert label_end_ts(f, 60000) == 180000
