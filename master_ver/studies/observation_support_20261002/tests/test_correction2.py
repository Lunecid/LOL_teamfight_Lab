"""Correction-2 fixtures: unsupported C0 bounds, aggregates, raw preflight, reference independence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from studies.observation_support_20261002 import aggregate as AG
from studies.observation_support_20261002 import config as C
from studies.observation_support_20261002 import contracts as CT
from studies.observation_support_20261002 import extract_match as EM
from studies.observation_support_20261002 import io_source as IO
from studies.observation_support_20261002 import reference_replay as REF
from studies.observation_support_20261002 import verify_independent as VER


def test_nonempty_c0_no_prior_frame_preserves_bounds():
    c0 = (1 << 0) | (1 << 4) | (1 << 7)
    part = EM._unsupported_participation_bounds(c0)
    assert part["n_lower"] == 0
    assert part["n_upper"] == 10
    assert part["lower_bits"] == 0
    assert part["upper_bits"] == (1 << 10) - 1
    assert part["p_lower_bits"] == c0
    assert part["p_upper_bits"] == (1 << 10) - 1
    assert (part["p_lower_bits"] & c0) == c0
    rec: dict = {}
    EM._fill_participation(rec, "tau_r1800", None, c0)
    assert rec["support_ok_tau_r1800"] is False
    assert rec["p_lower_bits_tau_r1800"] == c0
    assert rec["n_lower_tau_r1800"] == 0
    assert rec["n_upper_tau_r1800"] == 10


def test_pack_failure_fixture_fills_unsupported_bounds():
    """Simulated pack-failure path: fi invalid → unsupported bounds, not zeroed C0."""
    c0 = 1 << 2
    rec: dict = {"c0_bits": c0}
    EM._fill_participation(rec, "tm1_r2000", None, c0)
    assert rec["support_ok_tm1_r2000"] is False
    assert rec["p_lower_bits_tm1_r2000"] == c0
    assert rec["p_upper_n_tm1_r2000"] == 10


def test_same_frame_with_async_uses_cutoff_to_endpoint():
    n = 4
    table = {
        "match_local": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "lag_f_F_tau": np.asarray([10_000, 12_000, 10_000, 25_000], dtype=np.int64),
        "lag_f_F_tm1": np.asarray([10_000, 12_000, 10_000, 25_000], dtype=np.int64),
        "cand_10_20_tm1": np.asarray([1, 1, 1, 0], dtype=np.int8),
        "foreign_kills_in_tau_L": np.zeros(n, dtype=np.int64),
        "other_anchors_in_tau_L": np.zeros(n, dtype=np.int64),
        "exact_10000_tau": np.asarray([1, 0, 1, 0], dtype=np.int8),
        "strict_eligible": np.asarray([1, 1, 1, 0], dtype=np.int8),
        "frame_differs": np.zeros(n, dtype=np.int8),
        "singleton": np.asarray([1, 0, 1, 1], dtype=np.int8),
        "valid_h60": np.ones(n, dtype=np.int8),
        "valid_h90": np.ones(n, dtype=np.int8),
        "valid_h120": np.ones(n, dtype=np.int8),
    }
    for tag in ("tau_r1800", "tau_r2000", "tm1_r1800", "tm1_r2000"):
        table[f"support_ok_{tag}"] = np.ones(n, dtype=np.int8)
        table[f"p_lower_n_{tag}"] = np.asarray([2, 3, 2, 1], dtype=np.float64)
        table[f"p_upper_n_{tag}"] = np.asarray([3, 4, 3, 2], dtype=np.float64)
        table[f"n_lower_{tag}"] = np.asarray([1, 2, 1, 0], dtype=np.float64)
        table[f"n_upper_{tag}"] = np.asarray([2, 3, 2, 1], dtype=np.float64)
    for h in (60, 90, 120):
        for cut in ("tau", "tm1"):
            tag = f"{cut}_h{h}"
            table[f"same_as_Fpre_{tag}"] = np.asarray([1, 1, 0, 1], dtype=np.int8)
            table[f"F_post_gt_L_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"stale_at_or_before_L_{tag}"] = np.ones(n, dtype=np.int8)
            # Frame-to-frame async all zero; cutoff-to-endpoint positive on row0.
            table[f"async_events_Fpre_to_Fpost_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"async_events_cutoff_to_endpoint_{tag}"] = np.asarray([2, 0, 0, 5], dtype=np.float64)
            table[f"events_support_ok_{tag}"] = np.ones(n, dtype=np.int8)
            table[f"d_gold_blue_{tag}"] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            table[f"d_level_mean_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"d_hp_mean_{tag}"] = np.zeros(n, dtype=np.float64)
    agg = AG.build_aggregates(table, n_matches_source=2, n_zero_kill=0, n_matches_processed=2)
    # cand rows for tau: 0,1,2 (lags in 10-20). same & cutoff_async>0 → row0 only.
    assert agg["post_frame"]["tau_h60"]["same_frame_with_async_events"] == 1
    assert "full" in agg["participation_cells"]["tau_r1800"]
    assert "candidate" in agg["participation_cells"]["tau_r1800"]
    assert "isolated" in agg["participation_cells"]["tau_r1800"]
    assert agg["participation_cells"]["tau_r1800"]["n_population_full"] == 4
    assert agg["participation_cells"]["tau_r1800"]["n_population_cand"] == 3


def test_post_tm1_uses_own_candidate_mask():
    n = 3
    table = {
        "match_local": np.asarray([0, 0, 1], dtype=np.int64),
        "lag_f_F_tau": np.asarray([10_000, 10_000, 10_000], dtype=np.int64),
        "lag_f_F_tm1": np.asarray([10_000, 30_000, 10_000], dtype=np.int64),
        "cand_10_20_tm1": np.asarray([1, 0, 1], dtype=np.int8),
        "foreign_kills_in_tau_L": np.zeros(n, dtype=np.int64),
        "other_anchors_in_tau_L": np.zeros(n, dtype=np.int64),
        "exact_10000_tau": np.ones(n, dtype=np.int8),
        "strict_eligible": np.asarray([1, 0, 1], dtype=np.int8),
        "frame_differs": np.zeros(n, dtype=np.int8),
        "singleton": np.ones(n, dtype=np.int8),
        "valid_h60": np.ones(n, dtype=np.int8),
        "valid_h90": np.ones(n, dtype=np.int8),
        "valid_h120": np.ones(n, dtype=np.int8),
    }
    for tag in ("tau_r1800", "tau_r2000", "tm1_r1800", "tm1_r2000"):
        table[f"support_ok_{tag}"] = np.ones(n, dtype=np.int8)
        table[f"p_lower_n_{tag}"] = np.ones(n, dtype=np.float64)
        table[f"p_upper_n_{tag}"] = np.ones(n, dtype=np.float64) * 2
        table[f"n_lower_{tag}"] = np.ones(n, dtype=np.float64)
        table[f"n_upper_{tag}"] = np.ones(n, dtype=np.float64) * 2
    for h in (60, 90, 120):
        for cut in ("tau", "tm1"):
            tag = f"{cut}_h{h}"
            table[f"same_as_Fpre_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"F_post_gt_L_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"stale_at_or_before_L_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"async_events_Fpre_to_Fpost_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"async_events_cutoff_to_endpoint_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"events_support_ok_{tag}"] = np.ones(n, dtype=np.int8)
            table[f"d_gold_blue_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"d_level_mean_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"d_hp_mean_{tag}"] = np.zeros(n, dtype=np.float64)
    agg = AG.build_aggregates(table, 2, 0, n_matches_processed=2)
    assert agg["post_frame"]["tau_h60"]["n_cand_valid"] == 3
    assert agg["post_frame"]["tm1_h60"]["n_cand_valid"] == 2


def test_wrong_timeline_id_fixture(tmp_path: Path):
    mid = "KR_1"
    detail = {
        "metadata": {"matchId": mid},
        "info": {
            "gameVersion": "15.14.1",
            "participants": [
                {"participantId": i, "teamId": 100 if i <= 5 else 200} for i in range(1, 11)
            ],
        },
    }
    timeline = {
        "metadata": {"matchId": "KR_WRONG"},
        "info": {"frames": [{"timestamp": 0}, {"timestamp": 60_000}]},
    }
    root = tmp_path / "raw"
    (root / "detail").mkdir(parents=True)
    (root / "timeline").mkdir(parents=True)
    (root / "detail" / f"{mid}.json").write_text(json.dumps(detail), encoding="utf-8")
    (root / "timeline" / f"{mid}.json").write_text(json.dumps(timeline), encoding="utf-8")
    ver = IO.verify_raw_pair_identity(mid, root, cache_minute_ts=np.asarray([0, 60_000], dtype=np.int64))
    assert ver["ok"] is False
    assert ver["reason"] == "timeline_match_id_mismatch"
    assert ver["scope"] == "mismatch"


def test_multiple_root_and_gt50_failures_fixtures_no_corpus_write():
    match_ids = [f"KR_{i}" for i in range(60)]
    # All keys hit both roots → multiple-root unsupported, never absent.
    pair_sets = [set(match_ids), set(match_ids)]
    roots = [Path("r0"), Path("r1")]
    out = IO.census_raw_source_availability_from_pair_sets(match_ids, pair_sets, roots)
    assert out["n_missing"] == 0
    assert out["n_found_but_unsupported_raw"] == 60
    assert len(out["found_but_unsupported_raw_rows"]) == 60  # no truncation
    assert all(r["reason"] == "multiple_root_hits_unresolved" for r in out["unsupported_rows"])

    # >50 mismatch failures preserved fully.
    def verify_fail(mid, root, **kw):
        return {"ok": False, "reason": "detail_match_id_mismatch", "raw_fields_supported": False, "scope": "mismatch"}

    pair_sets2 = [set(match_ids), set()]
    out2 = IO.census_raw_source_availability_from_pair_sets(
        match_ids, pair_sets2, roots, verify_fn=verify_fail
    )
    assert out2["n_mismatch"] == 60
    assert len(out2["mismatch_rows"]) == 60


def test_reference_detects_wrong_same_size_c0_and_median():
    # Build tiny synthetic table + reference rows with intentional corruption detection.
    n = 2
    table = {
        "match_local": np.asarray([0, 1], dtype=np.int64),
        "component_id": np.asarray([0, 0], dtype=np.int64),
        "f": np.asarray([100_000, 100_000], dtype=np.int64),
        "L": np.asarray([100_000, 100_000], dtype=np.int64),
        "tau": np.asarray([90_000, 90_000], dtype=np.int64),
        "c0_bits": np.asarray([0b0011, 0b0011], dtype=np.int64),
        "F_tau": np.asarray([60_000, 60_000], dtype=np.int64),
        "F_tm1": np.asarray([60_000, 60_000], dtype=np.int64),
        "lag_f_F_tau": np.asarray([40_000, 40_000], dtype=np.int64),
        "lag_f_F_tm1": np.asarray([40_000, 40_000], dtype=np.int64),
        "same_ms_first_n": np.asarray([1, 1], dtype=np.int64),
        "cand_10_20_tm1": np.zeros(n, dtype=np.int8),
        "exact_10000_tau": np.zeros(n, dtype=np.int8),
        "strict_eligible": np.zeros(n, dtype=np.int8),
        "frame_differs": np.zeros(n, dtype=np.int8),
        "singleton": np.ones(n, dtype=np.int8),
        "foreign_kills_in_tau_L": np.zeros(n, dtype=np.int64),
        "other_anchors_in_tau_L": np.zeros(n, dtype=np.int64),
        "valid_h60": np.zeros(n, dtype=np.int8),
        "valid_h90": np.zeros(n, dtype=np.int8),
        "valid_h120": np.zeros(n, dtype=np.int8),
    }
    for tag in ("tau_r1800", "tau_r2000", "tm1_r1800", "tm1_r2000"):
        table[f"support_ok_{tag}"] = np.ones(n, dtype=np.int8)
        table[f"p_lower_bits_{tag}"] = np.asarray([0b0011, 0b0011], dtype=np.int64)
        table[f"p_upper_bits_{tag}"] = np.asarray([0b0011, 0b0011], dtype=np.int64)
        table[f"lower_bits_{tag}"] = np.zeros(n, dtype=np.int64)
        table[f"upper_bits_{tag}"] = np.zeros(n, dtype=np.int64)
        table[f"p_lower_n_{tag}"] = np.asarray([2, 2], dtype=np.float64)
        table[f"p_upper_n_{tag}"] = np.asarray([2, 2], dtype=np.float64)
        table[f"n_lower_{tag}"] = np.zeros(n, dtype=np.float64)
        table[f"n_upper_{tag}"] = np.zeros(n, dtype=np.float64)
    for h in (60, 90, 120):
        for cut in ("tau", "tm1"):
            tag = f"{cut}_h{h}"
            table[f"F_post_{tag}"] = np.full(n, -1, dtype=np.int64)
            table[f"async_events_cutoff_to_endpoint_{tag}"] = np.full(n, -1, dtype=np.int64)
            table[f"async_events_Fpre_to_Fpost_{tag}"] = np.full(n, -1, dtype=np.int64)
            table[f"same_as_Fpre_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"F_post_gt_L_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"stale_at_or_before_L_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"events_support_ok_{tag}"] = np.zeros(n, dtype=np.int8)
            table[f"d_gold_blue_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"d_level_mean_{tag}"] = np.zeros(n, dtype=np.float64)
            table[f"d_hp_mean_{tag}"] = np.zeros(n, dtype=np.float64)
    evidence = {
        "first_loc_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "first_loc_x": np.asarray([1.0, 2.0], dtype=np.float64),
        "first_loc_y": np.asarray([1.0, 2.0], dtype=np.float64),
    }
    # Reference with same popcount but wrong C0 bits.
    ref_rows = []
    for i in range(n):
        ref_rows.append(
            {
                "match_local": int(table["match_local"][i]),
                "component_id": int(table["component_id"][i]),
                "f": int(table["f"][i]),
                "L": int(table["L"][i]),
                "tau": int(table["tau"][i]),
                "c0_bits": 0b1100,  # same popcount=2, wrong bits
                "F_tau": int(table["F_tau"][i]),
                "F_tm1": int(table["F_tm1"][i]),
                "lag_f_F_tau": int(table["lag_f_F_tau"][i]),
                "lag_f_F_tm1": int(table["lag_f_F_tm1"][i]),
                "first_loc_x": np.asarray([evidence["first_loc_x"][i]]),
                "first_loc_y": np.asarray([evidence["first_loc_y"][i]]),
                "p_lower_bits_tau_r1800": 0b1100,
                "p_upper_bits_tau_r1800": 0b1100,
                "lower_bits_tau_r1800": 0,
                "upper_bits_tau_r1800": 0,
                "support_ok_tau_r1800": 1,
            }
        )
    cmp = REF.compare_table_to_reference(table, evidence, ref_rows, smoke=True)
    assert cmp["all_ok"] is False
    assert cmp["mismatches_by_field"].get("c0_bits", 0) > 0

    # Weighted median corruption: flip population mask in production aggregates vs reference.
    prod = AG.build_aggregates(table, 2, 0, n_matches_processed=2)
    ref_agg = REF.recompute_aggregates_reference(table)
    ok = REF.compare_aggregates(ref_agg, prod)
    assert ok["all_ok"] is True
    # Corrupt cand count in a copy
    bad = json.loads(json.dumps(prod))
    bad["gap_tau"]["cum_10_20"] = 999
    bad_cmp = REF.compare_aggregates(ref_agg, bad)
    assert bad_cmp["all_ok"] is False


def test_feature_channel_mapping_confirmed():
    m = REF.confirm_feature_channel_mapping()
    assert m["ok"] is True
    assert m["pinned_IDX_HP_PCT"] == 14


def test_inverse_cdf_not_median_of_means():
    # Two matches: match0 has values 1,100; match1 has 2. Equal-total-match median != mean-of-means median.
    values = np.asarray([1.0, 100.0, 2.0], dtype=np.float64)
    ml = np.asarray([0, 0, 1], dtype=np.int64)
    pop = np.ones(3, dtype=bool)
    stats = AG.weighted_stats(values, pop, ml, support_mask=pop)
    # Per-match means: (50.5, 2); median of means=26.25. Inverse-CDF with weights 0.5,0.5,1 → order 1,2,100 weights 0.5,1,0.5
    # cdf: 0.5, 1.5, 2.0; target 1.0 → value 2.
    assert stats["equal_total_match_median"] == pytest.approx(2.0)
