"""Tests for the teamfight_v2 detection pipeline.

Covers:
  - 5-second position grid (baseline + pre-kill override)
  - Temporal kill clustering
  - Teamfight validity (radius 1800, 2+ per team)
  - detect_fights_teamfight_v2 integration
  - Pick/skirmish rejection (only 1 per team)
  - Post-fight outcome aggregation
  - XY excluded from model input
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from gameplay.fights import (
    _build_5s_position_grid,
    _cluster_kills_temporal,
    _validate_teamfight_at_engage,
    _collect_interactions_in_radius,
    _finalize_kill_cluster,
    detect_fights_teamfight_v2,
    FightDetectorConfig,
    safe_int,
    _event_xy,
)


# ============================================================
# Helpers: build synthetic data
# ============================================================

def _make_minute_ts(n_minutes=10):
    """Create 60s-spaced timestamps starting at 60000 (minute 1)."""
    return np.arange(n_minutes, dtype=np.int64) * 60000 + 60000


def _make_xy_minute(n_minutes=10, n_players=10, base_positions=None):
    """Create (T, 10, 2) xy data. Default: blue near (5000,5000), red near (8000,8000)."""
    xy = np.zeros((n_minutes, n_players, 2), dtype=np.float32)
    if base_positions is None:
        # Blue team (0-4) clustered around (5000, 5000)
        for i in range(5):
            xy[:, i, 0] = 5000.0 + i * 100
            xy[:, i, 1] = 5000.0 + i * 100
        # Red team (5-9) clustered around (8000, 8000)
        for i in range(5):
            xy[:, 5 + i, 0] = 8000.0 + i * 100
            xy[:, 5 + i, 1] = 8000.0 + i * 100
    else:
        for pid, (px, py) in base_positions.items():
            xy[:, pid, 0] = px
            xy[:, pid, 1] = py
    return xy


def _make_kill_event(ts, killer_id, victim_id, x, y, assisting_ids=None):
    return {
        "timestamp": int(ts),
        "killer_id": int(killer_id),
        "victim_id": int(victim_id),
        "assisting_ids": assisting_ids or [],
        "position": (float(x), float(y)),
    }


def _make_event(etype, ts, **kwargs):
    ev = {"type": etype, "timestamp": int(ts)}
    ev.update(kwargs)
    return ev


def _make_tm():
    """Standard team mapping: pid 1-5 = team 100, pid 6-10 = team 200."""
    return {i: 100 for i in range(1, 6)} | {i: 200 for i in range(6, 11)}


def _make_cache(n_minutes=10, xy_minute=None, events=None, node_minute=None):
    """Build a minimal cache dict for testing."""
    minute_ts = _make_minute_ts(n_minutes)
    if xy_minute is None:
        xy_minute = _make_xy_minute(n_minutes)

    if node_minute is None:
        # Create minimal node_minute with x_norm, y_norm, alive
        from core.config import NODE_IDX, F_NODE
        nm = np.zeros((n_minutes, 10, F_NODE), dtype=np.float32)
        xi = NODE_IDX.get("x_norm", 0)
        yi = NODE_IDX.get("y_norm", 1)
        ai = NODE_IDX.get("alive", None)
        nm[:, :, xi] = xy_minute[:, :, 0] / 16000.0
        nm[:, :, yi] = xy_minute[:, :, 1] / 16000.0
        if ai is not None:
            nm[:, :, ai] = 1.0
        node_minute = nm

    return {
        "minute_ts": minute_ts,
        "xy_raw_minute": xy_minute,
        "node_minute": node_minute,
        "events": events or [],
        "meta": {"anchor_is_norm": False},
    }


# ============================================================
# Tests: 5-second position grid
# ============================================================

class TestBuild5sPositionGrid:
    def test_baseline_grid_shape(self):
        """Baseline grid should have 5s spacing."""
        minute_ts = _make_minute_ts(3)
        xy = _make_xy_minute(3)
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [], {})
        # 3 minutes = 120s range, 5s step → ~25 points
        assert xy_dense.ndim == 3
        assert xy_dense.shape[1] == 10
        assert xy_dense.shape[2] == 2
        assert len(dense_ts) == xy_dense.shape[0]
        assert int(dense_ts[1] - dense_ts[0]) == 5000

    def test_baseline_interpolates_xy(self):
        """Midpoint between two frames should be their average."""
        minute_ts = np.array([0, 60000], dtype=np.int64)
        xy = np.zeros((2, 10, 2), dtype=np.float32)
        xy[0, 0, 0] = 0.0
        xy[1, 0, 0] = 12000.0
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [], {})
        mid_idx = len(dense_ts) // 2
        assert xy_dense[mid_idx, 0, 0] == pytest.approx(6000.0, abs=500)

    def test_pre_kill_override(self):
        """Kill participants should be moved toward kill position."""
        minute_ts = np.array([0, 60000, 120000], dtype=np.int64)
        xy = np.zeros((3, 10, 2), dtype=np.float32)
        # Player 0 at (1000, 1000) in all frames
        xy[:, 0, 0] = 1000.0
        xy[:, 0, 1] = 1000.0

        kill = _make_kill_event(
            ts=90000, killer_id=1, victim_id=6,
            x=5000.0, y=5000.0,
        )
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [kill], _make_tm())

        # At kill time (90000), player 0 should be at kill position (5000, 5000)
        kill_idx = int(np.searchsorted(dense_ts, 90000, side="right")) - 1
        assert xy_dense[kill_idx, 0, 0] == pytest.approx(5000.0, abs=200)

    def test_non_participants_unaffected(self):
        """Players not in kill should keep baseline positions."""
        minute_ts = np.array([0, 60000, 120000], dtype=np.int64)
        xy = np.zeros((3, 10, 2), dtype=np.float32)
        xy[:, 2, 0] = 3000.0  # Player 2 not in kill
        xy[:, 2, 1] = 3000.0

        kill = _make_kill_event(
            ts=90000, killer_id=1, victim_id=6,
            x=9000.0, y=9000.0,
        )
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [kill], _make_tm())

        # Player 2 should remain at baseline position
        kill_idx = int(np.searchsorted(dense_ts, 90000, side="right")) - 1
        assert xy_dense[kill_idx, 2, 0] == pytest.approx(3000.0, abs=200)


# ============================================================
# Tests: Kill clustering
# ============================================================

class TestClusterKillsTemporal:
    def test_single_kill(self):
        kills = [_make_kill_event(100000, 1, 6, 5000, 5000)]
        clusters = _cluster_kills_temporal(kills, 18000)
        assert len(clusters) == 1
        assert clusters[0]["n_kills"] == 1

    def test_close_kills_same_cluster(self):
        """Kills within 18s should be in same cluster."""
        kills = [
            _make_kill_event(100000, 1, 6, 5000, 5000),
            _make_kill_event(110000, 2, 7, 5100, 5100),
            _make_kill_event(115000, 3, 8, 5200, 5200),
        ]
        clusters = _cluster_kills_temporal(kills, 18000)
        assert len(clusters) == 1
        assert clusters[0]["n_kills"] == 3

    def test_distant_kills_separate_clusters(self):
        """Kills separated by >18s should form separate clusters."""
        kills = [
            _make_kill_event(100000, 1, 6, 5000, 5000),
            _make_kill_event(200000, 2, 7, 8000, 8000),  # 100s later
        ]
        clusters = _cluster_kills_temporal(kills, 18000)
        assert len(clusters) == 2

    def test_fight_center_is_first_kill(self):
        """Fight center should be the first kill's position."""
        kills = [
            _make_kill_event(100000, 1, 6, 3000, 4000),
            _make_kill_event(110000, 2, 7, 5000, 6000),
        ]
        clusters = _cluster_kills_temporal(kills, 18000)
        assert clusters[0]["fight_center"] == pytest.approx((3000.0, 4000.0))

    def test_empty_kills(self):
        clusters = _cluster_kills_temporal([], 18000)
        assert len(clusters) == 0


# ============================================================
# Tests: Teamfight validity
# ============================================================

class TestValidateTeamfightAtEngage:
    def test_teamfight_passes(self):
        """5v5 cluster near fight center should pass."""
        minute_ts = _make_minute_ts(5)
        # All players near (6000, 6000) within radius 1800
        positions = {}
        for i in range(5):
            positions[i] = (6000 + i * 200, 6000 + i * 200)
        for i in range(5, 10):
            positions[i] = (6000 + (i - 5) * 200 + 500, 6000 + (i - 5) * 200 + 500)
        xy = _make_xy_minute(5, base_positions=positions)
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [], {})

        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

        result = _validate_teamfight_at_engage(
            xy_dense=xy_dense,
            dense_ts=dense_ts,
            engage_ts=int(minute_ts[2]),
            fight_center=(6500.0, 6500.0),
            b=b, r=r,
            validity_radius=1800.0,
            min_per_team=2,
            is_norm=False,
            scale_factor=16000.0,
        )
        assert result is True

    def test_pick_fails(self):
        """1v1 pick should fail the 2-per-team check."""
        minute_ts = _make_minute_ts(5)
        # Only 1 blue + 1 red near fight center; rest far away
        positions = {
            0: (6000, 6000),
            1: (1000, 1000), 2: (1000, 1000), 3: (1000, 1000), 4: (1000, 1000),
            5: (6200, 6200),
            6: (14000, 14000), 7: (14000, 14000), 8: (14000, 14000), 9: (14000, 14000),
        }
        xy = _make_xy_minute(5, base_positions=positions)
        dense_ts, xy_dense = _build_5s_position_grid(xy, minute_ts, [], {})

        b = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        r = np.array([5, 6, 7, 8, 9], dtype=np.int32)

        result = _validate_teamfight_at_engage(
            xy_dense=xy_dense,
            dense_ts=dense_ts,
            engage_ts=int(minute_ts[2]),
            fight_center=(6100.0, 6100.0),
            b=b, r=r,
            validity_radius=1800.0,
            min_per_team=2,
            is_norm=False,
            scale_factor=16000.0,
        )
        assert result is False


# ============================================================
# Tests: Finalize kill cluster
# ============================================================

class TestFinalizeKillCluster:
    def test_participants(self):
        kills = [
            _make_kill_event(100000, 1, 6, 5000, 5000, assisting_ids=[2, 3]),
            _make_kill_event(110000, 7, 4, 5100, 5100, assisting_ids=[8]),
        ]
        cluster = _finalize_kill_cluster(kills)
        # Should include: 1, 6, 2, 3, 7, 4, 8
        assert {1, 2, 3, 4, 6, 7, 8}.issubset(cluster["participants"])
        assert cluster["first_kill_ts"] == 100000
        assert cluster["last_kill_ts"] == 110000
        assert cluster["n_kills"] == 2


# ============================================================
# Tests: Integration — detect_fights_teamfight_v2
# ============================================================

class TestDetectFightsTeamfightV2:
    def _make_teamfight_scenario(self):
        """Create a scenario where a valid 5v5 teamfight occurs."""
        n_minutes = 10
        minute_ts = _make_minute_ts(n_minutes)
        # All players near (7000, 7000) for a proper teamfight
        positions = {}
        for i in range(5):
            positions[i] = (7000 + i * 100, 7000 + i * 100)
        for i in range(5, 10):
            positions[i] = (7500 + (i - 5) * 100, 7500 + (i - 5) * 100)
        xy = _make_xy_minute(n_minutes, base_positions=positions)

        kills = [
            _make_kill_event(300000, 1, 6, 7200, 7200, [2, 3]),
            _make_kill_event(310000, 2, 7, 7300, 7300, [1, 4]),
            _make_kill_event(315000, 8, 3, 7400, 7400, [9]),
        ]
        events = [
            {"type": "CHAMPION_KILL", "timestamp": k["timestamp"],
             "killerId": k["killer_id"], "victimId": k["victim_id"],
             "assistingParticipantIds": k["assisting_ids"],
             "position": {"x": k["position"][0], "y": k["position"][1]}}
            for k in kills
        ]

        cache = _make_cache(n_minutes, xy, events)
        return cache

    def test_detects_teamfight(self):
        """A 5v5 fight cluster should be detected."""
        cache = self._make_teamfight_scenario()
        tm = _make_tm()
        fights = detect_fights_teamfight_v2(cache, tm)
        assert len(fights) >= 1
        fight = fights[0]
        assert "engage_ts" in fight
        assert "first_kill_ts" in fight
        assert "last_kill_ts" in fight
        assert fight["first_kill_ts"] == 300000

    def test_rejects_1v1_pick(self):
        """A 1v1 pick should NOT be detected as a teamfight."""
        n_minutes = 10
        minute_ts = _make_minute_ts(n_minutes)
        # Only player 0 and player 5 are near each other; rest far away
        positions = {
            0: (7000, 7000),
            1: (1000, 1000), 2: (1000, 1000), 3: (1000, 1000), 4: (1000, 1000),
            5: (7100, 7100),
            6: (14000, 14000), 7: (14000, 14000), 8: (14000, 14000), 9: (14000, 14000),
        }
        xy = _make_xy_minute(n_minutes, base_positions=positions)
        kills = [_make_kill_event(300000, 1, 6, 7050, 7050)]
        events = [
            {"type": "CHAMPION_KILL", "timestamp": 300000,
             "killerId": 1, "victimId": 6,
             "assistingParticipantIds": [],
             "position": {"x": 7050, "y": 7050}}
        ]
        cache = _make_cache(n_minutes, xy, events)
        tm = _make_tm()
        fights = detect_fights_teamfight_v2(cache, tm)
        assert len(fights) == 0

    def test_no_kills_returns_empty(self):
        """No kills → no fights."""
        cache = _make_cache(10, events=[])
        tm = _make_tm()
        fights = detect_fights_teamfight_v2(cache, tm)
        assert len(fights) == 0

    def test_post_fight_outcome_present(self):
        """Detected fights should have post_fight_outcome field."""
        cache = self._make_teamfight_scenario()
        tm = _make_tm()
        fights = detect_fights_teamfight_v2(cache, tm)
        if len(fights) > 0:
            assert "post_fight_outcome" in fights[0]

    def test_diag_stored_in_cache(self):
        """Diagnostics should be stored in cache."""
        cache = self._make_teamfight_scenario()
        tm = _make_tm()
        detect_fights_teamfight_v2(cache, tm)
        assert "fight_detect_diag" in cache
        diag = cache["fight_detect_diag"]
        assert diag["detector"] == "teamfight_v2"
        assert diag["clusters_total"] >= 1


# ============================================================
# Tests: XY exclusion from model input
# ============================================================

class TestXYExclusion:
    def test_zero_xy_default_false(self):
        """[P0-1] Default config should have ZERO_XY_NODE_FEATURES=False (dual-path design)."""
        from core.config import cfg
        assert bool(getattr(cfg, "ZERO_XY_NODE_FEATURES", False)) is False

    def test_dual_path_xy_flags(self):
        """[P0-1] Dual-path XY config flags exist and have correct defaults."""
        from core.config import cfg
        assert bool(getattr(cfg, "ZERO_XY_IN_EXTRA_SEQ", True)) is True
        assert bool(getattr(cfg, "USE_RELATIVE_XY", True)) is True
        assert isinstance(getattr(cfg, "ADJ_SIGMA_FACTOR", 1.0), float)
        assert cfg.ADJ_SIGMA_FACTOR > 1.0


# ============================================================
# Tests: Config defaults
# ============================================================

class TestTeamfightV2Config:
    def test_default_detector(self):
        from core.config import cfg
        assert str(getattr(cfg, "FIGHT_DETECTOR", "")).lower() in ("teamfight_v2",)

    def test_tf2_parameters_exist(self):
        from core.config import cfg
        assert hasattr(cfg, "TF2_KILL_CLUSTER_GAP_MS")
        assert hasattr(cfg, "TF2_ENGAGE_PRE_KILL_MS")
        assert hasattr(cfg, "TF2_VALIDITY_RADIUS")
        assert hasattr(cfg, "TF2_INTERACTION_RADIUS")
        assert hasattr(cfg, "TF2_POST_FIGHT_WINDOW_MS")
        assert hasattr(cfg, "TF2_MIN_PER_TEAM")

    def test_tf2_parameter_values(self):
        from core.config import cfg
        assert cfg.TF2_KILL_CLUSTER_GAP_MS == 18000
        assert cfg.TF2_ENGAGE_PRE_KILL_MS == 10000
        assert cfg.TF2_VALIDITY_RADIUS == 1800.0
        assert cfg.TF2_INTERACTION_RADIUS == 3000.0
        assert cfg.TF2_POST_FIGHT_WINDOW_MS == 20000
        assert cfg.TF2_MIN_PER_TEAM == 2
