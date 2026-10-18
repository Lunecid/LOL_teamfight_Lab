"""Tests for app/detection_quality_report.py helper logic."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app.detection_quality_report import (
    FightDigest,
    compare_variants_against_reference,
    match_fight_sets,
    summarize_fight_set,
    temporal_iou,
)


def _fight(
    *,
    match_id: str = "KR_TEST",
    engage_ts: int,
    end_ts: int,
    first_kill_ts: int,
    centroid_x: float,
    centroid_y: float,
    total_kills: int = 2,
    kill_diff: int = 1,
    post_obj_diff: int = 0,
    post_tower_diff: int = 0,
    post_gold_diff: float = 0.0,
) -> FightDigest:
    return FightDigest(
        match_id=match_id,
        engage_ts=engage_ts,
        end_ts=end_ts,
        first_kill_ts=first_kill_ts,
        last_kill_ts=first_kill_ts + 5000,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        total_kills=total_kills,
        kill_diff=kill_diff,
        post_obj_diff=post_obj_diff,
        post_tower_diff=post_tower_diff,
        post_gold_diff=post_gold_diff,
    )


def test_temporal_iou_basic():
    assert temporal_iou(0, 100, 50, 150) == pytest.approx(1.0 / 3.0)
    assert temporal_iou(0, 100, 100, 200) == 0.0
    assert temporal_iou(0, 0, 10, 20) == 0.0


def test_match_fight_sets_one_to_one_greedy():
    refs = [
        _fight(engage_ts=100000, end_ts=160000, first_kill_ts=120000, centroid_x=4000, centroid_y=4000),
        _fight(engage_ts=300000, end_ts=360000, first_kill_ts=320000, centroid_x=9000, centroid_y=9000),
    ]
    cands = [
        _fight(engage_ts=102000, end_ts=162000, first_kill_ts=121000, centroid_x=4100, centroid_y=4050),
        _fight(engage_ts=302000, end_ts=359000, first_kill_ts=321000, centroid_x=9050, centroid_y=9100),
    ]

    pairs = match_fight_sets(refs, cands, engage_tol_ms=20000, center_tol=3000.0, iou_min=0.1)
    assert len(pairs) == 2
    assert sorted([p.ref_idx for p in pairs]) == [0, 1]
    assert sorted([p.cand_idx for p in pairs]) == [0, 1]


def test_summarize_fight_set_impact_rates():
    fights = [
        _fight(engage_ts=0, end_ts=60000, first_kill_ts=10000, centroid_x=0, centroid_y=0, post_obj_diff=1),
        _fight(engage_ts=70000, end_ts=130000, first_kill_ts=80000, centroid_x=0, centroid_y=0, post_tower_diff=1),
        _fight(engage_ts=140000, end_ts=200000, first_kill_ts=150000, centroid_x=0, centroid_y=0, post_gold_diff=2000.0),
        _fight(engage_ts=210000, end_ts=270000, first_kill_ts=220000, centroid_x=0, centroid_y=0, post_gold_diff=100.0),
    ]
    s = summarize_fight_set(fights, gold_impact_threshold=1500.0)
    assert s["n_fights"] == 4
    assert s["objective_impact_rate"] == pytest.approx(0.25)
    assert s["tower_impact_rate"] == pytest.approx(0.25)
    assert s["gold_impact_rate"] == pytest.approx(0.25)
    assert s["impact_rate"] == pytest.approx(0.75)


def test_compare_variants_against_reference_counts_and_delta():
    ref_map = {
        "KR_A": [
            _fight(
                match_id="KR_A",
                engage_ts=100000,
                end_ts=160000,
                first_kill_ts=120000,
                centroid_x=4000,
                centroid_y=4000,
                post_gold_diff=1800.0,
            ),
            _fight(
                match_id="KR_A",
                engage_ts=300000,
                end_ts=360000,
                first_kill_ts=320000,
                centroid_x=9000,
                centroid_y=9000,
                post_gold_diff=200.0,
            ),
        ]
    }
    cand_map = {
        "KR_A": [
            _fight(
                match_id="KR_A",
                engage_ts=103000,
                end_ts=162000,
                first_kill_ts=121000,
                centroid_x=4100,
                centroid_y=4050,
                post_gold_diff=1700.0,
            ),
            _fight(
                match_id="KR_A",
                engage_ts=500000,
                end_ts=560000,
                first_kill_ts=520000,
                centroid_x=10000,
                centroid_y=10000,
                post_tower_diff=1,
            ),
        ]
    }

    out = compare_variants_against_reference(
        reference_fights=ref_map,
        candidate_fights=cand_map,
        engage_tol_ms=20000,
        center_tol=3000.0,
        iou_min=0.1,
        gold_impact_threshold=1500.0,
    )
    cnt = out["counts"]
    dq = out["detection_quality"]
    mf = out["meaningfulness"]

    assert cnt["n_ref"] == 2
    assert cnt["n_candidate"] == 2
    assert cnt["n_matched"] == 1
    assert dq["precision"] == pytest.approx(0.5)
    assert dq["recall"] == pytest.approx(0.5)
    assert dq["f1"] == pytest.approx(0.5)
    assert mf["reference_only"]["n_fights"] == 1
    assert mf["candidate_only"]["n_fights"] == 1
    assert mf["delta_candidate_only_minus_reference_only_impact_rate"] == pytest.approx(1.0)
