"""The label window is half-open; its end must include the fight's last kill."""
from gameplay.fights import label_window_end_ts


def test_last_kill_inside_window_when_fight_outlasts_horizon():
    engage, horizon, last_kill = 5_000, 35_000, 41_000
    end = label_window_end_ts(last_kill, engage, horizon)
    assert end == 41_001 and last_kill < end


def test_minimum_horizon_kept_for_short_fights():
    engage, horizon, last_kill = 5_000, 35_000, 20_000
    assert label_window_end_ts(last_kill, engage, horizon) == 40_000
