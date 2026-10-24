"""Focused scientific tests for objective_timing_20261002."""
from __future__ import annotations

import numpy as np
import pytest

from studies.objective_timing_20261002 import config as C
from studies.objective_timing_20261002 import aggregate as AGG
from studies.objective_timing_20261002.parse import (
    classify_monster,
    parse_objective_events,
    parse_timestamp_ms,
    resolve_building_acquisition,
    resolve_monster_acquisition,
    rows_to_arrays,
)
from studies.objective_timing_20261002.query import (
    classify_uncapped_vs_next_anchor,
    first_objective_after_L,
    query_component_intervals_scalar,
    query_match_components_vectorized,
    real_time_or_none,
)


BLUE = [1, 2, 3, 4, 5]


def test_timestamp_rejects_bool_and_fractional():
    assert parse_timestamp_ms(True)[0] is None
    assert parse_timestamp_ms(1.5)[0] is None
    assert parse_timestamp_ms(-1)[0] is None
    assert parse_timestamp_ms(12000) == (12000, "")


def test_building_owner_inversion():
    r = resolve_building_acquisition({"teamId": 100})
    assert r["acq_status"] == C.ACQ_CONFIRMED
    assert r["team"] == 200
    r2 = resolve_building_acquisition({"teamId": 200})
    assert r2["team"] == 100


def test_neutral_300_not_roster_promoted():
    # killerId 1 is blue, but explicit 300 must stay neutral
    r = resolve_monster_acquisition({"killerTeamId": 300, "killerId": 1}, BLUE)
    assert r["acq_status"] == C.ACQ_NEUTRAL
    assert r["team"] == C.TEAM_UNKNOWN
    assert r["team_source"] == "explicit_neutral_300"


def test_roster_fallback_when_team_missing():
    r = resolve_monster_acquisition({"killerTeamId": 0, "killerId": 7}, BLUE)
    assert r["acq_status"] == C.ACQ_CONFIRMED
    assert r["team"] == C.TEAM_RED
    assert r["team_source"] == "killerId_roster"


def test_contradictory_team_evidence():
    # killerTeamId blue, killerId on red roster
    r = resolve_monster_acquisition({"killerTeamId": 100, "killerId": 8}, BLUE)
    assert r["acq_status"] == C.ACQ_CONTRADICTORY
    assert r["team"] == C.TEAM_UNKNOWN


def test_unknown_monster_team_without_roster():
    r = resolve_monster_acquisition({"killerTeamId": None, "killerId": 0}, BLUE)
    assert r["acq_status"] == C.ACQ_UNKNOWN


def test_dragon_unknown_subtype_not_elemental():
    leaf, _, _ = classify_monster({"monsterType": "DRAGON", "monsterSubType": ""})
    assert leaf == "dragon_unknown_subtype"
    leaf2, _, _ = classify_monster({"monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"})
    assert leaf2 == "dragon_elemental"
    leaf3, _, _ = classify_monster({"monsterType": "DRAGON", "monsterSubType": "ELDER_DRAGON"})
    assert leaf3 == "dragon_elder"


def test_objective_strictly_after_L_and_at_L():
    rows, _ = parse_objective_events(
        [
            {"type": "ELITE_MONSTER_KILL", "timestamp": 1000, "monsterType": "BARON_NASHOR", "killerTeamId": 100, "killerId": 1},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 2000, "monsterType": "BARON_NASHOR", "killerTeamId": 200, "killerId": 6},
        ],
        BLUE,
        last_frame_ms=5000,
    )
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "baron", 1000, 5000, None,
    )
    assert hit["status"] == C.ST_OBSERVED
    assert hit["t_ms"] == 2000
    assert hit["same_ms_at_L"] is True
    assert hit["delay_ms"] == 1000


def test_neutral_only_is_no_acquisition_not_unknown():
    rows, census = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 3000,
                "monsterType": "RIFTHERALD",
                "killerTeamId": 300,
                "killerId": 1,
            }
        ],
        BLUE,
        last_frame_ms=9000,
    )
    assert census["n_acq_neutral"] == 1
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "herald", 1000, 9000, None,
    )
    assert hit["status"] == C.ST_NO_ACQ
    assert hit["uncertain_in_interval"] is False
    assert hit["neutral_in_interval"] is True


def test_neutral_before_confirmed_is_observed_without_uncertainty():
    rows, _ = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 2000,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 300,
                "killerId": 1,
            },
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 4000,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 100,
                "killerId": 1,
            },
        ],
        BLUE,
        last_frame_ms=9000,
    )
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "baron", 1000, 9000, None,
    )
    assert hit["status"] == C.ST_OBSERVED
    assert hit["t_ms"] == 4000
    assert hit["uncertain_precedes_confirmed"] is False
    assert hit["uncertain_in_interval"] is False
    assert hit["neutral_precedes_confirmed"] is True
    assert hit["neutral_in_interval"] is True


def test_true_unknown_interval_is_unknown_obs():
    rows, census = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 3000,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 0,
                "killerId": 0,
            }
        ],
        BLUE,
        last_frame_ms=9000,
    )
    assert census["n_acq_unknown"] == 1
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "baron", 1000, 9000, None,
    )
    assert hit["status"] == C.ST_UNKNOWN_OBS
    assert hit["uncertain_in_interval"] is True
    assert hit["neutral_in_interval"] is False


def test_same_ms_multi_objective_ties():
    rows, _ = parse_objective_events(
        [
            {"type": "BUILDING_KILL", "timestamp": 5000, "buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", "teamId": 100},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 5000, "monsterType": "HORDE", "killerTeamId": 200, "killerId": 6},
        ],
        BLUE,
        last_frame_ms=9000,
    )
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "any_objective", 1000, 9000, None,
    )
    assert hit["status"] == C.ST_OBSERVED
    assert hit["t_ms"] == 5000
    assert hit["n_tie_events"] >= 1
    assert hit["tie_category_mask"] != 0


def test_no_future_event_and_game_ending():
    rows, _ = parse_objective_events([], BLUE, last_frame_ms=1000)
    arr = rows_to_arrays(rows)
    hit = first_objective_after_L(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"], "any_objective", 1000, 2000, None,
    )
    assert hit["status"] == C.ST_NO_ACQ


def test_after_last_frame_before_game_end():
    rows, census = parse_objective_events(
        [
            {
                "type": "BUILDING_KILL",
                "timestamp": 8000,
                "buildingType": "TOWER_BUILDING",
                "towerType": "OUTER_TURRET",
                "teamId": 200,
            }
        ],
        BLUE,
        last_frame_ms=7000,
    )
    assert census["n_after_last_frame"] == 1
    assert rows[0]["after_last_frame"] is True


def test_next_anchor_le_L_no_interval_and_classify():
    rows, _ = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 5000,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 100,
                "killerId": 1,
            }
        ],
        BLUE,
        last_frame_ms=9000,
    )
    arr = rows_to_arrays(rows)
    q = query_component_intervals_scalar(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"],
        L=4000, game_end_ms=9000, game_end_known=True,
        next_kill_after_L=6000, next_component_tau=3000,  # tau <= L
    )
    pre = q[f"{C.INTERVAL_PRE_NEXT}:baron"]
    assert pre["status"] == C.ST_NO_INTERVAL
    label = classify_uncapped_vs_next_anchor(
        5000, C.ST_OBSERVED, 4000, 6000, 3000, 9000, True
    )
    assert label == "already_active_overlapping_next_component"


def test_intervening_kill_equality_not_strictly_earlier():
    rows, _ = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 6000,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 100,
                "killerId": 1,
            }
        ],
        BLUE,
        last_frame_ms=9000,
    )
    arr = rows_to_arrays(rows)
    q = query_component_intervals_scalar(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"],
        L=4000, game_end_ms=9000, game_end_known=True,
        next_kill_after_L=6000, next_component_tau=-1,
    )
    # restricted requires t < 6000, so objective at 6000 excluded
    assert q[f"{C.INTERVAL_PRE_NEXT}:baron"]["status"] == C.ST_NO_ACQ
    assert q[f"{C.INTERVAL_UNCAPED}:baron"]["status"] == C.ST_OBSERVED
    label = classify_uncapped_vs_next_anchor(
        6000, C.ST_OBSERVED, 4000, 6000, -1, 9000, True
    )
    assert label == "same_ms_as_next_kill_or_anchor"


def test_sentinel_conversion():
    assert real_time_or_none(-1) is None
    assert real_time_or_none(-2) is None
    assert real_time_or_none(0) == 0
    assert real_time_or_none(12) == 12


def test_equal_match_mean_and_weighted_median():
    # Two matches: match0 has 2 obs delays 1000,3000; match1 has 1 obs delay 2000
    delay = np.asarray([1000, 3000, 2000], dtype=np.int64)
    ml = np.asarray([0, 0, 1], dtype=np.int64)
    s = AGG.delay_summary(delay, ml)
    # event-weighted mean = (1+3+2)/3 = 2s
    assert s["event_weighted"]["mean_s"] == pytest.approx(2.0)
    # equal-match: weights 0.5,0.5,1 -> normalize /2 → 0.25,0.25,0.5; mean = 0.25*1+0.25*3+0.5*2 = 2.0
    assert s["equal_match_weighted"]["mean_s"] == pytest.approx(2.0)
    # weighted median at 0.5 of total weight
    assert s["equal_match_weighted"]["median_s"] == pytest.approx(2.0)


def test_vectorized_equals_scalar_synthetic():
    events = [
        {"type": "ELITE_MONSTER_KILL", "timestamp": 1500, "monsterType": "RIFTHERALD", "killerTeamId": 300, "killerId": 1},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 2500, "monsterType": "BARON_NASHOR", "killerTeamId": 100, "killerId": 1},
        {"type": "BUILDING_KILL", "timestamp": 2500, "buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", "teamId": 200},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 8000, "monsterType": "DRAGON", "monsterSubType": "", "killerTeamId": 200, "killerId": 6},
        {"type": "BUILDING_KILL", "timestamp": 12000, "buildingType": "NEXUS_BUILDING", "teamId": 100},
    ]
    rows, _ = parse_objective_events(events, BLUE, last_frame_ms=10000)
    arr = rows_to_arrays(rows)
    L_arr = np.asarray([1000, 2000, 7000, 11000], dtype=np.int64)
    nk = np.asarray([4000, 9000, -1, 15000], dtype=np.int64)
    nt = np.asarray([500, 3000, 20000, -1], dtype=np.int64)  # first has overlap
    ge = 20000
    vec = query_match_components_vectorized(
        arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
        arr["after_last_frame"], arr["acq_status"],
        L_arr, ge, True, nk, nt,
    )
    for i, L in enumerate(L_arr.tolist()):
        sc = query_component_intervals_scalar(
            arr["t_ms"], arr["event_ord"], arr["category_mask"], arr["team"],
            arr["after_last_frame"], arr["acq_status"],
            L=L, game_end_ms=ge, game_end_known=True,
            next_kill_after_L=int(nk[i]), next_component_tau=int(nt[i]),
        )
        for key, block in vec.items():
            srow = sc[key]
            assert int(block["status"][i]) == int(srow["status"]), (key, i, block["status"][i], srow["status"])
            assert int(block["t_ms"][i]) == int(srow["t_ms"])
            assert int(block["delay_ms"][i]) == int(srow["delay_ms"])
            assert int(block["event_ord"][i]) == int(srow["event_ord"])
            assert int(block["uncertain_precedes_confirmed"][i]) == int(bool(srow["uncertain_precedes_confirmed"]))
            assert int(block["neutral_in_interval"][i]) == int(bool(srow["neutral_in_interval"]))
            assert int(block["neutral_precedes_confirmed"][i]) == int(bool(srow["neutral_precedes_confirmed"]))


def test_load_components_no_pickle():
    from studies.objective_timing_20261002.io_util import load_components, verify_anchor_manifest

    checks = verify_anchor_manifest()
    assert checks["sha256_ok"]
    assert checks["components_sha256_ok"]
    assert checks["terminals_sha256_ok"]
    assert checks["match_ids_sha256_ok"]
    assert checks["match_meta_sha256_ok"]
    assert checks["all_ok"]
    comps = load_components()
    assert comps["L"].shape[0] == C.EXPECTED_MIN1_COMPONENTS
    band = (comps["delta"] >= 10000) & (comps["delta"] <= 20000)
    isol = band & (comps["foreign_kills_in_tau_L"] == 0) & (comps["other_anchors_in_tau_L"] == 0)
    assert int(band.sum()) == C.EXPECTED_FULL_WINDOW
    assert int(isol.sum()) == C.EXPECTED_FULL_ISOL


def test_threshold_coverage_cohort_denominator_hand():
    """One <=30s acquisition + one NO_ACQ => cohort cov 50%, conditional prop_le 100%."""
    status = np.asarray([C.ST_OBSERVED, C.ST_NO_ACQ], dtype=np.int64)
    delay = np.asarray([20_000, C.SENTINEL_MISSING], dtype=np.int64)
    ml = np.asarray([0, 0], dtype=np.int64)
    eord = np.asarray([1, -1], dtype=np.int64)
    mask = np.asarray([True, True])
    out = AGG.summarize_query_slice(status, delay, ml, eord, mask)
    assert out["n_candidates"] == 2
    assert out["n_observed_acquisition"] == 1
    assert out["delays"]["event_weighted"]["prop_le"]["30"] == pytest.approx(1.0)
    assert out["threshold_coverage_cohort"]["event_weighted"]["30"] == pytest.approx(0.5)
    assert out["threshold_coverage_cohort"]["equal_match_weighted"]["30"] == pytest.approx(0.5)


def test_threshold_coverage_unequal_candidates_and_unknown():
    # Match0: 2 candidates (obs 10s, no-acq); Match1: 1 candidate UNKNOWN
    status = np.asarray([C.ST_OBSERVED, C.ST_NO_ACQ, C.ST_UNKNOWN_OBS], dtype=np.int64)
    delay = np.asarray([10_000, -1, -1], dtype=np.int64)
    ml = np.asarray([0, 0, 1], dtype=np.int64)
    eord = np.asarray([1, -1, -1], dtype=np.int64)
    out = AGG.summarize_query_slice(status, delay, ml, eord, np.ones(3, dtype=bool))
    # Event-weighted cohort cov@30 = 1/3
    assert out["threshold_coverage_cohort"]["event_weighted"]["30"] == pytest.approx(1.0 / 3.0)
    # Equal-match: weights 1/2,1/2,1 → normalize /2 → 0.25,0.25,0.5; hit only first → 0.25
    assert out["threshold_coverage_cohort"]["equal_match_weighted"]["30"] == pytest.approx(0.25)
    # NO_INTERVAL row excluded from numerator
    status2 = np.asarray([C.ST_OBSERVED, C.ST_NO_INTERVAL], dtype=np.int64)
    delay2 = np.asarray([90_000, -1], dtype=np.int64)
    ml2 = np.asarray([0, 0], dtype=np.int64)
    out2 = AGG.summarize_query_slice(status2, delay2, ml2, np.asarray([1, -1]), np.ones(2, dtype=bool))
    assert out2["threshold_coverage_cohort"]["event_weighted"]["30"] == pytest.approx(0.0)
    assert out2["threshold_coverage_cohort"]["event_weighted"]["120"] == pytest.approx(0.5)


def test_zero_observed_delay_stats_na_coverage_defined():
    status = np.asarray([C.ST_NO_ACQ, C.ST_NO_ACQ], dtype=np.int64)
    delay = np.asarray([-1, -1], dtype=np.int64)
    ml = np.asarray([0, 1], dtype=np.int64)
    out = AGG.summarize_query_slice(status, delay, ml, np.asarray([-1, -1]), np.ones(2, dtype=bool))
    assert out["delays"]["n_observed"] == 0
    assert out["delays"]["event_weighted"]["mean_s"] != out["delays"]["event_weighted"]["mean_s"]  # nan
    assert out["threshold_coverage_cohort"]["event_weighted"]["30"] == pytest.approx(0.0)


def test_events_array_unsupported_and_malformed():
    from studies.objective_timing_20261002.io_util import events_array_from_json

    ev, err = events_array_from_json({"not": "events"})
    assert ev == [] and err == "unsupported_events_payload"
    ev, err = events_array_from_json({"events": {"bad": 1}})
    assert err == "malformed_events_field_not_list"
    ev, err = events_array_from_json({"frames": [1, 2]})
    assert err.startswith("malformed_frame_not_dict")
    ev, err = events_array_from_json({"frames": [{"events": "x"}]})
    assert err.startswith("malformed_frame_events_not_list")
    ev, err = events_array_from_json({"info": {"frames": [{"events": [{"type": "X"}]}]}})
    assert err == "" and len(ev) == 1


def test_invalid_objective_timestamp_marks_support_unknown():
    rows, census = parse_objective_events(
        [
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 1.5,
                "monsterType": "BARON_NASHOR",
                "killerTeamId": 100,
                "killerId": 1,
            }
        ],
        BLUE,
        last_frame_ms=9000,
    )
    assert census["n_invalid_objective_timestamp"] == 1
    assert census["observation_support_unknown"] is True
    assert rows == []  # dropped, but must not be treated as valid empty NO_ACQ by callers


def test_hash_identity_rejects_reordered_fixture(tmp_path, monkeypatch):
    from studies.objective_timing_20261002 import io_util as IO
    from studies.objective_timing_20261002 import config as cfg

    # Isolate loaders to a tiny fake match_ids file with wrong hash
    fake_ids = tmp_path / "train_match_ids.json"
    fake_ids.write_text('["a"]\n', encoding="utf-8")
    monkeypatch.setattr(cfg, "ST_SOURCE", tmp_path)
    # Also need a fake anchor manifest with matching expected source hash field
    man = {
        "complete": True,
        "components_sha256": cfg.EXPECTED_COMPONENTS_SHA256,
        "terminals_sha256": cfg.EXPECTED_TERMINALS_SHA256,
        "source_hashes": {
            "match_ids_sha256": "0" * 64,
            "match_meta_sha256": cfg.EXPECTED_MATCH_META_SHA256,
        },
        "identity": {"all_ok": True},
        "aggregates": {"n_components": cfg.EXPECTED_MIN1_COMPONENTS},
    }
    man_path = tmp_path / "RUN_MANIFEST.json"
    man_path.write_text(__import__("json").dumps(man), encoding="utf-8")
    monkeypatch.setattr(cfg, "ANCHOR_MANIFEST", man_path)
    monkeypatch.setattr(cfg, "EXPECTED_ANCHOR_MANIFEST_SHA256", IO.file_sha256(man_path))
    monkeypatch.setattr(cfg, "COMPONENTS_NPZ", tmp_path / "missing_components.npz")
    monkeypatch.setattr(cfg, "TERMINALS_JSON", tmp_path / "missing_terminals.json")
    with pytest.raises(RuntimeError, match="match_ids sha256"):
        IO.load_match_ids()
