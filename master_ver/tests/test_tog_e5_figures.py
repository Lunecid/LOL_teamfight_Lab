"""E5 case figures (PRESPEC_E5_KO.md sections 2, 4, 6 and 9; PLAN_E1_E5_KO.md Task 6; SDD rulings 26, 26b, 32 and 33).

Only the parts that need no bank, model or run data: the minimap transform and the map check on synthetic images,
the time grid, the Q_DEV table assembly on a synthetic row table, the Brier and gate checks, the kill records, the
write-once helpers and a drawing smoke test.  The real run is studies/tog_experiments_20261012/e5_figures.py itself.
"""
import json

import numpy as np
import pytest

from studies.tog_experiments_20261012 import e5_figures as F

LANE = np.array([145, 152, 112], dtype=np.uint8)


# ---------------------------------------------------------------- minimap transform and map check

def test_minimap_formula_extent_matches_prespec():
    # PRESPEC_E5 line 36: x = -120 -> 25.34, x = 14,870 -> 490.16, y = 14,980 -> 25.34, y = -120 -> 490.16 pixels.
    x, y = F.minimap_xy(np.array([-120.0, 14_870.0]), np.array([14_980.0, -120.0]))
    np.testing.assert_allclose(x, [29 * 512 / 586, 561 * 512 / 586])
    np.testing.assert_allclose(y, [29 * 512 / 586, 561 * 512 / 586])
    assert [round(v, 2) for v in x.tolist()] == [25.34, 490.16]


def test_minimap_turrets_match_the_prespec_table():
    # PRESPEC_E5 line 38: the twelve lane turrets and their transformed pixels (two decimals).
    for turret in F.PRESPEC_TURRETS:
        x, y = F.minimap_xy(turret["game_xy"][0], turret["game_xy"][1])
        assert (round(float(x), 2), round(float(y), 2)) == turret["pixel_xy"]
    assert sum(t["checked"] for t in F.PRESPEC_TURRETS) == 8           # six outer, two mid inner (Ruling 26b)
    assert {(t["lane"], t["tower"]) for t in F.PRESPEC_TURRETS if not t["checked"]} == {
        ("TOP_LANE", "INNER_TURRET"), ("BOT_LANE", "INNER_TURRET")}


def _anchors(override=None):
    towers = [{"team": t["team"], "lane": t["lane"], "building": "TOWER_BUILDING", "tower": t["tower"], "slot": 0,
               "x": t["game_xy"][0], "y": t["game_xy"][1], "n": 1} for t in F.PRESPEC_TURRETS]
    towers.append({"team": 100, "lane": "MID_LANE", "building": "TOWER_BUILDING", "tower": "NEXUS_TURRET", "slot": 0,
                   "x": 2177, "y": 1807, "n": 1})
    for (team, lane, tower), xy in (override or {}).items():
        for row in towers:
            if (row["team"], row["lane"], row["tower"]) == (team, lane, tower):
                row["x"], row["y"] = xy
    return {"towers": towers}


def test_lane_turrets_read_the_anchor_file_and_refuse_a_differing_one():
    turrets = F.lane_turrets(_anchors())
    assert [(t["team"], t["lane"], t["tower"]) for t in turrets] == [
        (t["team"], t["lane"], t["tower"]) for t in F.PRESPEC_TURRETS]
    with pytest.raises(F.Blocked):
        F.lane_turrets(_anchors({(100, "BOT_LANE", "OUTER_TURRET"): (10_600, 1_029)}))


def test_lane_colour_is_the_median_of_the_centre_9x9_block():
    image = np.full((512, 512, 3), 255, dtype=np.uint8)            # outside the block: not counted
    block = np.zeros((9, 9), dtype=np.uint8)
    ring = np.ones((9, 9), dtype=bool)
    ring[1:8, 1:8] = False
    block[~ring] = np.arange(49)                                   # the inner 7 x 7 holds 0..48 (its median is 24)
    block[ring] = np.arange(49, 81)                                # the outer ring holds 49..80
    image[252:261, 252:261, 0] = block
    image[252:261, 252:261, 1] = 2 * block
    image[252:261, 252:261, 2] = 100
    assert F.lane_colour(image) == (40, 80, 100)


def test_lane_mask_is_an_rgb_distance_of_at_most_40():
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    image[0, 0] = LANE
    image[0, 1] = LANE + np.array([40, 0, 0], dtype=np.uint8)       # distance 40: lane
    image[0, 2] = LANE + np.array([24, 24, 24], dtype=np.uint8)     # distance 41.6: not lane
    assert F.lane_mask(image, (145, 152, 112)).tolist() == [[True, True, False, False]]


def test_border_scans_the_centre_row_and_column_inward():
    mask = np.zeros((512, 512), dtype=bool)
    mask[256, [30, 300, 480]] = True
    mask[[40, 200, 470], 256] = True
    assert F.measure_border(mask) == {"left": 30, "right": 480, "top": 40, "bottom": 470}
    mask[:, 256] = False
    assert F.measure_border(mask)["top"] is None and F.measure_border(mask)["bottom"] is None


def test_nearest_lane_pixel_uses_integer_pixel_indices():
    mask = np.zeros((128, 128), dtype=bool)
    mask[20, 10] = True                          # row 20, column 10
    mask[100, 100] = True
    distance, pixel = F.nearest_lane_pixel(mask, 13.0, 24.0)
    assert distance == 5.0 and pixel == (10, 20)
    assert F.nearest_lane_pixel(np.zeros((4, 4), dtype=bool), 1.0, 1.0) == (float("inf"), None)


def _map_image(shift=None, border_left=25):
    """Black image, lane colour at the centre block, the border points on the centre row and column, and one lane
    pixel at each turret's transformed point (``shift`` moves the pixel of the named turrets)."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[252:261, 252:261] = LANE
    image[256, border_left] = LANE
    image[256, 490] = LANE
    image[25, 256] = LANE
    image[490, 256] = LANE
    for turret in F.PRESPEC_TURRETS:
        x, y = F.minimap_xy(*turret["game_xy"])
        dx = (shift or {}).get((turret["team"], turret["lane"], turret["tower"]), 0)
        image[int(round(float(y))), int(round(float(x))) + dx] = LANE
    return image


def test_map_check_passes_on_a_drawn_lane_and_border():
    check = F.map_check(_map_image(), F.lane_turrets(_anchors()))
    assert check["passed"] and check["failures"] == []
    assert check["lane_colour"] == [145, 152, 112]
    assert {side: entry["measured"] for side, entry in check["border"].items()} == {
        "left": 25, "right": 490, "top": 25, "bottom": 490}
    assert all(t["distance_px"] <= 0.71 for t in check["turrets"])
    assert check["tolerance_px"] == pytest.approx(10.24)


def test_map_check_fails_on_a_checked_turret_or_the_border_but_not_on_an_excluded_turret():
    turrets = F.lane_turrets(_anchors())
    far = F.map_check(_map_image(shift={(100, "BOT_LANE", "OUTER_TURRET"): 11}), turrets)
    assert not far["passed"] and far["failures"] == ["turret:100:BOT_LANE:OUTER_TURRET"]
    excluded = F.map_check(_map_image(shift={(100, "BOT_LANE", "INNER_TURRET"): 30}), turrets)
    assert excluded["passed"]
    entry = next(t for t in excluded["turrets"] if (t["team"], t["lane"], t["tower"]) == (100, "BOT_LANE", "INNER_TURRET"))
    assert entry["checked"] is False and entry["passed"] is None and entry["distance_px"] > 10.24
    border = F.map_check(_map_image(border_left=36), turrets)      # |36 - 25.34| = 10.66 > 10.24
    assert not border["passed"] and border["failures"] == ["border:left"]


# ---------------------------------------------------------------- time grid

def test_time_grid_runs_every_5_s_from_tau_minus_60_s_to_the_endpoint():
    assert F.time_grid(100_000, 150_000) == list(range(40_000, 150_001, 5_000))
    grid = F.time_grid(100_000, 152_500)
    assert grid[-2:] == [150_000, 152_500] and 100_000 in grid    # the endpoint is added; tau is on the grid
    assert F.time_grid(32_000, 50_000)[:2] == [2_000, 7_000]       # times before 0 are left out, the grid stays
    with pytest.raises(ValueError):
        F.time_grid(100_000, 100_000)


# ---------------------------------------------------------------- Q_DEV table

ROLES = np.array([0, 3, 3, 4, 3, 5, 3, 0], dtype=np.int8)
DEV = np.flatnonzero(ROLES == 3)                                   # q rows 1, 2, 4, 6


def _sources():
    n = len(ROLES)
    match = np.array([b"m%d" % (i // 2) for i in range(n)], dtype="S64")
    comp = np.array([b"c%d" % i for i in range(n)], dtype="S64")
    key = np.char.add(np.char.add(match, b":"), comp)
    y = np.array([-1, 1, 0, 1, 0, -1, 1, -1], dtype=np.int8)
    svi = np.array([np.nan, 0.2, -0.1, 0.3, 0.0, np.nan, 0.05, np.nan])
    q = {"engagement_key": key, "canonical_match_key": match, "component_key": comp, "match_index": np.arange(n) // 2,
         "role": ROLES, "query_pre": np.arange(n) * 10, "pre_ms": 100_000 + np.arange(n), "y": y,
         "label_svi_delta": svi, "eval_n_kills": np.array([1, 2, 1, 3, 1, 2, 2, 1], dtype=np.int16),
         "weight_raw": np.full(n, 0.5), "truncation_bits": np.array([1, 8, 4, 1, 2, 1, 1, 6], dtype=np.int64)}
    order = np.array([5, 2, 7, 0, 6, 4, 1, 3])                     # label rows in another order
    test_svi = svi.copy()
    test_svi[5] = 0.9                                              # a TEST label value: must not matter
    labels = {"role": ROLES[order], "canonical_match_key": match[order], "component_key": comp[order],
              "valid": np.ones(n, dtype=bool)[order], "direction_positive": np.where(y < 0, 1, y)[order],
              "svi_delta": test_svi[order], "F_pre_ms": np.array([0, 5, 5, 0, 7, 0, 9, 0])[order],
              "F_post_ms": np.array([0, 5, 6, 0, 7, 0, 9, 0])[order], "query_pre": q["query_pre"][order],
              "query_post": (q["query_pre"] + 1)[order], "pre_ms": q["pre_ms"][order],
              "post_ms": (q["pre_ms"] + 50_000)[order], "first_kill_ms": (q["pre_ms"] + 10_000)[order],
              "last_kill_ms": (q["pre_ms"] + 12_000)[order], "p_pre": np.linspace(0.3, 0.7, n)[order],
              "p_post": np.linspace(0.3, 0.7, n)[order] + svi[order], "match_index": q["match_index"][order],
              "truncation_bits": q["truncation_bits"][order]}
    ka = {"engagement_key": key.copy(), "blue_kills": np.array([1, 1, 0, 2, 1, 1, 0, 0], dtype=np.int32),
          "red_kills": np.array([0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int32)}
    objective = np.zeros((n * 10 + 1, 9), dtype=np.int32)
    objective[q["query_pre"][4] + 1, 5] = 1                        # q row 4: a tower falls between pre and post
    preds = {"P1": {"row": DEV.copy(), "p": np.array([0.9, 0.2, 0.6, 0.4])},
             "PT_flex": {"row": DEV.copy(), "p": np.array([0.6, 0.5, 0.5, 0.5])}}
    return q, labels, ka, objective, preds


def test_table_has_the_ruling_32_columns_on_the_q_dev_rows():
    built = F.assemble_table(*_sources())
    table, rows = built["table"], built["rows"]
    q = _sources()[0]
    np.testing.assert_array_equal(table["engagement_key"], q["engagement_key"][DEV])
    np.testing.assert_array_equal(table["p_q"], [0.9, 0.2, 0.6, 0.4])
    np.testing.assert_array_equal(table["p_pt"], [0.6, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(table["y"], [1, 0, 0, 1])
    np.testing.assert_array_equal(table["svi"], [0.2, -0.1, 0.0, 0.05])
    np.testing.assert_array_equal(table["n_kills"], [2, 1, 1, 2])
    np.testing.assert_array_equal(table["ka_kills"], [2, 1, 1, 1])
    np.testing.assert_array_equal(table["kill_side_blue"], [1, 0, 1, 0])
    np.testing.assert_array_equal(table["same_frame"], [1, 0, 1, 1])
    np.testing.assert_array_equal(table["objective"], [0, 0, 1, 0])
    np.testing.assert_array_equal(rows["q_row"], DEV)
    np.testing.assert_array_equal(rows["post_ms"], q["pre_ms"][DEV] + 50_000)
    np.testing.assert_array_equal(rows["truncation_bits"], [8, 4, 2, 1])
    assert sorted(table) == sorted(["engagement_key", *F.e5_cases.NUMERIC])
    assert built["checks"]["rows"] == 4 and built["checks"]["matches"] == 4


def test_table_refuses_ka_labels_in_another_order_as_blocked():
    q, labels, ka, objective, preds = _sources()
    ka["engagement_key"] = ka["engagement_key"][::-1].copy()
    with pytest.raises(F.Blocked) as caught:
        F.assemble_table(q, labels, ka, objective, preds)
    assert caught.value.stage == "table" and "ka_key_order" in caught.value.reasons


@pytest.mark.parametrize("breakage", ["pred_rows", "label_y", "label_missing", "label_invalid", "label_query", "label_svi",
                                      "label_pre_ms", "label_match", "label_truncation", "label_repeated",
                                      "query_pre_negative"])
def test_table_refuses_misaligned_sources(breakage):
    q, labels, ka, objective, preds = _sources()
    at, other = (int(i) for i in np.flatnonzero(labels["role"] == 3)[:2])
    if breakage == "pred_rows":
        preds["PT_flex"]["row"] = DEV[::-1].copy()
    elif breakage == "label_y":
        labels["direction_positive"][at] = 1 - labels["direction_positive"][at]
    elif breakage == "label_missing":
        labels["role"][at] = 4
    elif breakage == "label_invalid":
        labels["valid"][at] = False
    elif breakage == "label_query":
        labels["query_pre"][at] += 1
    elif breakage == "label_svi":
        labels["svi_delta"][at] += 1e-12
    elif breakage == "label_pre_ms":
        labels["pre_ms"][at] += 1
    elif breakage == "label_match":
        labels["match_index"][at] += 100
    elif breakage == "label_truncation":
        labels["truncation_bits"][at] ^= 1
    elif breakage == "label_repeated":
        labels["canonical_match_key"][at] = labels["canonical_match_key"][other]
        labels["component_key"][at] = labels["component_key"][other]
    else:                                          # a negative query index would wrap around OBJECTIVE_COUNTS
        row = int(np.flatnonzero(labels["component_key"] == q["component_key"][DEV[0]])[0])
        q["query_pre"][DEV[0]] = -1
        labels["query_pre"][row] = -1
    with pytest.raises(ValueError):
        F.assemble_table(q, labels, ka, objective, preds)


def test_brier_check_reproduces_the_match_weighted_brier():
    p, y, w = np.array([0.9, 0.2, 0.6]), np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5, 1.0])
    value = ((0.5 * 0.01) + (0.5 * 0.04) + 0.36) / 2.0
    assert F.brier_check(p, y, w, value)["passed"]
    assert not F.brier_check(p, y, w, value + 1e-9)["passed"]


def test_closer_prediction():
    assert F.closer_prediction(0.7, 0.4, 1) == "p_q"
    assert F.closer_prediction(0.3, 0.6, 1) == "p_pt"
    assert F.closer_prediction(0.4, 0.6, 1) == "p_pt" and F.closer_prediction(0.5, 0.5, 0) == "tie"


# ---------------------------------------------------------------- gate, kills, writing

def test_gate_check_needs_a_passed_gate_on_unchanged_code():
    gate = {"passed": True, "run": {"code_sha256": {"a.py": "1", "b.py": "2", "common.py": "3"}}}
    ok = F.gate_check(gate, {"a.py": "1", "b.py": "2", "common.py": "9"}, ignore=("common.py",))
    assert ok["passed"] and ok["changed"] == [] and ok["ignored_changed"] == ["common.py"]
    changed = F.gate_check(gate, {"a.py": "1", "b.py": "7", "common.py": "3"}, ignore=("common.py",))
    assert not changed["passed"] and changed["changed"] == ["b.py"]
    assert not F.gate_check({**gate, "passed": False}, {"a.py": "1", "b.py": "2"}, ignore=("common.py",))["passed"]


def _kill(ordinal, t, killer, victim, x=1000.0, y=2000.0):
    return {"source_ordinal": ordinal, "timestamp_ms": t, "killer_id": killer, "victim_id": victim, "x": x, "y": y,
            "xy_valid": True}


def test_kill_records_name_teams_and_champions_and_split_members_from_other_kills():
    kills = [_kill(4, 105_000, 2, 7), _kill(9, 104_000, 6, 1), _kill(11, 104_000, 0, 3), _kill(2, 50_000, 1, 6),
             _kill(13, 130_000, 7, 2)]
    team_ids = np.array([100] * 5 + [200] * 5)
    champions = np.array([11, 12, 13, 14, 15, 21, 22, 23, 24, 25])
    names = {11: "Ahri", 12: "Lee Sin", 13: "Kai'Sa", 21: "Garen", 22: "Jinx"}
    members, others = F.kill_records(kills, [4, 9, 11], team_ids, champions, names, (60_000, 120_000))
    assert [k["source_ordinal"] for k in members] == [9, 11, 4]    # time, then ordinal
    assert [k["order"] for k in members] == [1, 2, 3]
    assert members[0]["killer_team"] == "red" and members[0]["killer_champion"] == "Garen"
    assert members[0]["victim_team"] == "blue" and members[0]["victim_champion"] == "Ahri"
    assert members[1]["killer_team"] == "environment" and members[1]["killer_champion"] is None
    assert members[2]["victim_champion"] == "Jinx" and members[2]["victim_team"] == "red"
    assert [k["source_ordinal"] for k in others] == []             # 50,000 and 130,000 lie outside the window
    members, others = F.kill_records(kills, [4], team_ids, champions, names, (0, 200_000))
    assert [k["source_ordinal"] for k in others] == [2, 9, 11, 13] and all(k["order"] is None for k in others)
    assert all("killer_id" not in k and "victim_id" not in k for k in members + others)   # no participant slots


def test_objective_records_keep_the_window_and_drop_duplicates():
    history = [{"timestamp_ms": 90_000, "event_type": "BUILDING_KILL", "objective_category": "tower", "team_id": 200,
                "building_type": "TOWER_BUILDING", "tower_type": "INNER_TURRET", "lane_type": "MID_LANE",
                "monster_type": None, "duplicate_identity": False},
               {"timestamp_ms": 70_000, "event_type": "ELITE_MONSTER_KILL", "objective_category": "dragon", "team_id": 100,
                "monster_type": "DRAGON", "duplicate_identity": False},
               {"timestamp_ms": 75_000, "event_type": "ELITE_MONSTER_KILL", "objective_category": "dragon", "team_id": 100,
                "monster_type": "DRAGON", "duplicate_identity": True},
               {"timestamp_ms": 20_000, "event_type": "TURRET_PLATE_DESTROYED", "objective_category": "plate", "team_id": 100,
                "duplicate_identity": False},
               {"timestamp_ms": None, "event_type": "BUILDING_KILL", "objective_category": None, "team_id": None}]
    records = F.objective_records(history, (60_000, 100_000), tau=80_000)
    assert [(r["time_ms"], r["category"], r["team"]) for r in records] == [(70_000, "dragon", "blue"), (90_000, "tower", "red")]
    assert records[0]["time_from_tau_s"] == -10.0 and records[1]["detail"] == "INNER_TURRET MID_LANE"


def test_zoom_window_is_a_square_around_the_points_inside_the_map():
    assert F.zoom_window([(1000.0, 2000.0), (3000.0, 2000.0)]) == (2000.0, 2000.0, 1900.0)
    assert F.zoom_window([(5000.0, 5000.0)]) == (5000.0, 5000.0, 1800.0)
    # near a corner the square is moved inside the formula extent (x -120..14,870, y -120..14,980), not shrunk
    assert F.zoom_window([(14_500.0, 300.0)]) == (14_870.0 - 1800.0, -120.0 + 1800.0, 1800.0)


def test_blocked_files_are_numbered_and_never_overwritten(tmp_path):
    first = F.write_blocked(tmp_path, "map", {"border:left": 11.0})
    second = F.write_blocked(tmp_path, "map", {"border:left": 11.0})
    assert first.name == "E5_BLOCKED_map_1.json" and second.name == "E5_BLOCKED_map_2.json"
    assert json.loads(first.read_text(encoding="utf-8"))["blocked"] is True


def test_figures_are_written_once(tmp_path):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 1))
    digest = F.save_figure_once(figure, tmp_path / "x.pdf")
    assert (tmp_path / "x.pdf").read_bytes().startswith(b"%PDF") and len(digest) == 64
    with pytest.raises(FileExistsError):
        F.save_figure_once(figure, tmp_path / "x.pdf")


def test_case_figure_draws_from_case_facts(tmp_path):
    image = _map_image()
    times = list(range(40_000, 150_001, 5_000)) + [152_500]
    kill = {"order": 1, "time_ms": 110_000, "killer_team": "blue", "killer_champion": "Ahri", "victim_team": "red",
            "victim_champion": "Jinx", "x": 7000.0, "y": 7000.0, "xy_valid": True, "member": True}
    case = {"rule": "a", "title": "(a) confidently wrong", "engagement_key": "m:c", "p_q": 0.9, "p_pt": 0.6, "y": 0,
            "svi": -0.08, "tau_ms": 100_000, "endpoint_ms": 152_500, "F_pre_ms": 95_000, "F_post_ms": 150_000,
            "time_axis": {"times_ms": times, "v": np.linspace(0.6, 0.5, len(times)).tolist(),
                          "frame_ms": [60_000] * 8 + [120_000] * (len(times) - 8)},
            "kills": [kill, {**kill, "order": 2, "time_ms": 110_500, "killer_team": "red"}],
            "other_kills_in_window": [{**kill, "order": None, "member": False, "time_ms": 60_000}],
            "objective_events_in_window": [{"time_ms": 120_000, "time_from_tau_s": 20.0, "category": "baron", "team": "blue"},
                                           {"time_ms": 70_000, "time_from_tau_s": -30.0, "category": "plate", "team": None}],
            "positions_at_F_tau": [{"team": "blue" if i < 5 else "red", "champion": f"C{i}", "x": 1000.0 * (i + 1),
                                    "y": 1000.0 * (i + 1), "known": i != 3, "likely_dead": i == 1} for i in range(10)],
            "endpoint_reasons": ["game_end_minus_1"], "summary_lines": ["p_q 0.900", "p_pt 0.600"]}
    digest = F.draw_case(tmp_path / "case_a.pdf", case, image)
    assert (tmp_path / "case_a.pdf").read_bytes().startswith(b"%PDF") and len(digest) == 64
    case["kills"] = []                             # no engagement kill position: the zoom falls back to the whole map
    assert len(F.draw_case(tmp_path / "case_b.pdf", case, image)) == 64


# ---------------------------------------------------------------- fix round 1 (Ruling 34)

def test_likely_dead_is_a_position_equal_to_the_own_last_death_position():
    xy = np.array([[100, 200], [300, 400], [700, 800], [900, 1000], [50, 60], [0, 5], [70, 80], [11, 12], [13, 14],
                   [15, 16]], dtype=float)
    known = np.ones(10, dtype=bool)
    known[5] = False                               # a zero coordinate: unknown, never marked
    kills = [_kill(1, 50_000, 6, 1, 100.0, 200.0),  # slot 1 died where it stands at the frame: likely dead
             _kill(3, 55_000, 6, 2, 500.0, 600.0),  # slot 2 died here at 40 s and elsewhere at 55 s: the last death
             _kill(2, 40_000, 6, 2, 300.0, 400.0),  #   decides (records out of time order on purpose)
             _kill(4, 61_000, 6, 3, 700.0, 800.0),  # slot 3 dies here only after the frame
             _kill(5, 60_000, 0, 5, 50.0, 60.0),    # slot 5: environment kill at the frame time, same position
             _kill(6, 30_000, 7, 6, 0.0, 5.0),      # slot 6: same position, but the position is unknown
             _kill(7, 20_000, 1, 7, 70.0, 80.5)]    # slot 7: half a unit off (exact equality only)
    out = F.likely_dead(xy, known, kills, 60_000)
    assert [o["likely_dead"] for o in out] == [True, False, False, False, True, False, False, False, False, False]
    assert [o["last_death_ms"] for o in out] == [50_000, 55_000, None, None, 60_000, 30_000, 20_000, None, None, None]


def test_endpoint_reasons_decode_the_label_truncation_bits():
    assert F.endpoint_reasons(1) == ["horizon"]
    assert F.endpoint_reasons(8) == ["game_end_minus_1"]
    assert F.endpoint_reasons(6) == ["next_kill_minus_1", "next_anchor_minus_1"]
    for bad in (0, 16, -1):                        # a valid h90 label has at least one of the four reasons
        with pytest.raises(ValueError):
            F.endpoint_reasons(bad)
    assert F.endpoint_text(["game_end_minus_1"]) == "game end - 1 ms"


def test_selection_audit_counts_candidates_and_the_runner_up():
    from tests.test_tog_e5_cases import T
    from studies.tog_experiments_20261012.e5_cases import select_cases

    audit = F.selection_audit(T, select_cases(T))
    a, b, c = audit["a"], audit["b"], audit["c"]
    assert a["candidates_by_threshold"] == {"0.8": 2, "0.7": 2, "0.6": 2} and a["threshold"] == 0.8
    assert (a["candidates"], a["chosen"], a["runner_up"], a["tied_at_top"]) == (2, "k2", "k1", 1)
    assert a["margin"] == pytest.approx(0.03)
    assert (b["candidates"], b["chosen"], b["runner_up"]) == (2, "k3", "k6") and b["margin"] == pytest.approx(0.35)
    assert (c["population_rows"], c["violations_before_exclusion"], c["candidates"]) == (3, 3, 2)
    assert (c["chosen"], c["runner_up"]) == ("k6", "k4") and c["margin"] == pytest.approx(0.01)
    tie = {k: v.copy() for k, v in T.items()}
    tie["svi"][1] = -0.05
    tie["n_kills"][0] = 3                          # the plan's (a) tie: k2 wins by hash, the margin is 0
    a = F.selection_audit(tie, select_cases(tie))["a"]
    assert (a["chosen"], a["runner_up"], a["margin"], a["tied_at_top"]) == ("k2", "k1", 0.0, 2)
    wrong = select_cases(T) | {"b": "k6"}
    with pytest.raises(RuntimeError):              # the audit must agree with select_cases
        F.selection_audit(T, wrong)


def test_time_axis_limits_zoom_to_the_data_with_a_minimum_span():
    assert F.time_axis_limits(np.array([0.5, 0.6])) == pytest.approx((0.484, 0.61))
    assert F.time_axis_limits(np.array([0.99, 0.999])) == pytest.approx((0.99 - 0.16 * 0.02, 1.0))


def test_labels_avoid_each_other_the_markers_and_the_axes_edge():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.05, 0.05, 0.9, 0.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    points = [(50, 50), (50, 50), (51, 50), (50, 51), (97, 97)]
    markers = [(x, y, 7.0) for x, y in points]
    placed = F.place_labels(ax, [(x, y, name) for (x, y), name in zip(points, ["Alpha", "Beta", "Gamma", "Delta", "Edge"])],
                            markers, fontsize=7)
    renderer = figure.canvas.get_renderer()
    boxes = [box for _artist, box in placed]
    scale = figure.dpi / 72.0
    marker_boxes = [(*ax.transData.transform((x, y)), size * scale / 2) for x, y, size in markers]
    axbox = ax.get_window_extent(renderer)
    for i, box in enumerate(boxes):
        assert axbox.x0 <= box.x0 and box.x1 <= axbox.x1 and axbox.y0 <= box.y0 and box.y1 <= axbox.y1
        assert all(not F.boxes_overlap(box, other) for other in boxes[i + 1:])
        assert all(not (box.x0 < mx + h and mx - h < box.x1 and box.y0 < my + h and my - h < box.y1)
                   for mx, my, h in marker_boxes)


def _anchor_file(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(_anchors()), encoding="utf-8")
    return path


def test_run_stops_at_a_failed_map_check_before_any_case_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "load_map_image", lambda: _map_image(border_left=36))
    monkeypatch.setattr(F, "MAP_ANCHORS", _anchor_file(tmp_path))
    out = tmp_path / "out"
    assert F.run(out) == 2
    names = sorted(path.name for path in out.iterdir())
    assert names == ["E5_BLOCKED_map_1.json", "MAP_CHECK.json", "MAP_CHECK.png"]
    assert json.loads((out / "MAP_CHECK.json").read_text(encoding="utf-8"))["failures"] == ["border:left"]


def test_run_refuses_a_folder_with_e5_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "load_map_image", lambda: _map_image())
    (tmp_path / "E5_CASES.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        F.run(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["E5_CASES.json"]


def test_an_error_after_the_map_check_is_recorded_and_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "load_map_image", lambda: _map_image())
    monkeypatch.setattr(F, "MAP_ANCHORS", _anchor_file(tmp_path))

    def broken(*_args, **_kwargs):
        raise RuntimeError("gate check broke")

    monkeypatch.setattr(F, "gate_check", broken)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        F.run(out)
    assert sorted(path.name for path in out.iterdir()) == ["E5_FAILED_gate_1.json", "MAP_CHECK.json", "MAP_CHECK.png"]
    record = json.loads((out / "E5_FAILED_gate_1.json").read_text(encoding="utf-8"))
    assert record["failed"] is True and record["error"] == "RuntimeError" and record["message"] == "gate check broke"


def test_importing_the_module_leaves_cuda_visible():
    import os
    import subprocess
    import sys

    code = ("import os; os.environ['CUDA_VISIBLE_DEVICES'] = 'probe'; "
            "import studies.tog_experiments_20261012.e5_figures; print(os.environ['CUDA_VISIBLE_DEVICES'])")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
                            cwd=str(F.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.stdout.strip() == "probe"
