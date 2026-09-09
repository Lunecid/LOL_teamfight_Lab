"""Spatial anchors must not know about buildings destroyed after the cutoff."""
from gameplay.anchors import causal_anchors, load_static_anchors, standing_towers


def _kill(ts, team, lane, tower, x, y):
    return {"type": "BUILDING_KILL", "timestamp": ts, "teamId": team, "laneType": lane,
            "buildingType": "TOWER_BUILDING", "towerType": tower, "position": {"x": x, "y": y}}


def test_static_map_has_all_towers_and_pits():
    s = load_static_anchors()
    assert len(s["towers"]) == 22
    assert set(s["objectives"]) >= {"DRAGON", "BARON", "RIFTHERALD", "HORDE", "ATAKHAN"}


def test_tower_destroyed_after_cutoff_is_still_standing():
    events = [_kill(100_000, 100, "TOP_LANE", "OUTER_TURRET", 981, 10441),      # before cutoff
              _kill(300_000, 200, "BOT_LANE", "OUTER_TURRET", 13866, 4505)]     # after cutoff
    t = standing_towers(events, cutoff_ms=200_000)
    assert [981.0, 10441.0] not in t["TOWER_T100"]
    assert [13866.0, 4505.0] in t["TOWER_T200"]
    assert len(t["TOWER_T100"]) == 10 and len(t["TOWER_T200"]) == 11


def test_nexus_turrets_are_removed_one_at_a_time():
    events = [_kill(50_000, 100, "MID_LANE", "NEXUS_TURRET", 2177, 1807)]
    t = standing_towers(events, cutoff_ms=60_000)
    blue = t["TOWER_T100"]
    assert [2177.0, 1807.0] not in blue and [1748.0, 2270.0] in blue


def test_causal_anchors_shape():
    a = causal_anchors({"events": []}, cutoff_ms=0)
    assert set(a) == {"obj", "tower"} and len(a["tower"]["TOWER_T100"]) == 11
