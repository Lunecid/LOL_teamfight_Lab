"""Bounded source-replay fixtures and corruption coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from studies.observation_support_20261002 import config as C
from studies.observation_support_20261002 import reference_replay as REF


def _fixture(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(C, "CACHE_DIR", cache)
    match_id = "SYNTH-1"
    minute_ts = np.asarray([0, 50_000, 60_000, 120_000, 180_000], dtype=np.int64)
    xy = np.full((5, 10, 2), 10_000.0, dtype=np.float32)
    xy[2, 0] = [100.0, 100.0]
    xy[2, 1] = [110.0, 100.0]
    xy[2, 2] = [5_000.0, 5_000.0]
    xy[2, 3] = [0.0, 0.0]
    node = np.zeros((5, 10, 76), dtype=np.float32)
    node[:, :, C.IDX_HP_PCT] = 1.0
    node[2, 1, C.IDX_HP_PCT] = 0.0
    node[2, 3, C.IDX_HP_PCT] = 0.0
    node[2, :, C.IDX_LEVEL_NORM] = 1.0
    node[3, :, C.IDX_LEVEL_NORM] = 2.0
    node[2, :, C.IDX_XP_NORM] = 0.25
    node[3, :, C.IDX_XP_NORM] = 0.5
    gold = np.asarray([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0], [400.0, 400.0]], dtype=np.float32)
    np.savez(
        cache / f"{match_id}.npz",
        minute_ts=minute_ts,
        xy_raw_minute=xy,
        node_minute=node,
        gold_team_minute=gold,
    )
    team_map = {str(i): (C.TEAM_BLUE if i <= 5 else C.TEAM_RED) for i in range(1, 11)}
    (cache / f"{match_id}.meta.json").write_text(json.dumps({"team_map": team_map}), encoding="utf-8")
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 70_000},
        {"type": "CHAMPION_KILL", "timestamp": 90_000},
        {"type": "CHAMPION_KILL", "timestamp": 130_000},
        {"type": "GAME_END", "timestamp": 180_000},
    ]
    (cache / f"{match_id}.events.json").write_text(json.dumps(events), encoding="utf-8")
    arrays = {
        "t": np.asarray([70_000, 90_000, 130_000], dtype=np.int64),
        "x": np.asarray([100.0, 110.0, 200.0], dtype=np.float64),
        "y": np.asarray([100.0, 100.0, 200.0], dtype=np.float64),
        "killer": np.asarray([1, 3, 4], dtype=np.int64),
        "victim": np.asarray([6, 7, 8], dtype=np.int64),
        "assist": np.asarray([1 << 1, 0, 0], dtype=np.int64),
        "labels": np.asarray([0, 0, 1], dtype=np.int64),
    }
    blue_pids = np.asarray([1, 2, 3, 4, 5], dtype=np.int16)
    return match_id, arrays, blue_pids


def _replay(tmp_path: Path, monkeypatch):
    match_id, arrays, blue_pids = _fixture(tmp_path, monkeypatch)
    baseline = {
        "component_id": np.asarray([0, 1], dtype=np.int64),
        "e_h60": np.asarray([999_999, 999_999], dtype=np.int64),
        "valid_h60": np.asarray([0, 0], dtype=np.int8),
    }
    result = REF.replay_match_components(
        match_local=4,
        match_id=match_id,
        **arrays,
        blue_pids=blue_pids,
        baseline_rows=baseline,
    )
    return result


def _table_and_evidence(rows):
    n = len(rows)
    table = {}
    for field in REF.REQUIRED_REFERENCE_TABLE_FIELDS:
        values = [row[field] for row in rows]
        kind = REF.REFERENCE_FIELD_METADATA.get(field, {}).get("kind", "float")
        if kind == "string":
            table[field] = np.asarray(values, dtype="U96")
        elif kind == "bool":
            table[field] = np.asarray(values, dtype=np.int8)
        elif kind == "int":
            table[field] = np.asarray(values, dtype=np.int64)
        else:
            table[field] = np.asarray(values, dtype=np.float64)
    xs = [np.asarray(row["first_kill_x"], dtype=np.float64) for row in rows]
    ys = [np.asarray(row["first_kill_y"], dtype=np.float64) for row in rows]
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, arr in enumerate(xs):
        offsets[i + 1] = offsets[i] + arr.size
    evidence = {
        "first_loc_offsets": offsets,
        "first_loc_x": np.concatenate(xs),
        "first_loc_y": np.concatenate(ys),
    }
    for name in REF.REQUIRED_REFERENCE_EVIDENCE_FIELDS:
        if name in evidence:
            continue
        values = []
        for row in rows:
            if name.startswith("dist_"):
                values.append(row[f"_dists_{name[len('dist_'):]}"])
            elif name.startswith("pos_codes_"):
                values.append(row[f"_pos_codes_{name[len('pos_codes_'):]}"])
            elif name.startswith("hp_codes_"):
                values.append(row[f"_hp_codes_{name[len('hp_codes_'):]}"])
            else:
                values.append(row[f"_{name}"])
        evidence[name] = np.asarray(values)
    return table, evidence


def test_replay_source_endpoints_and_complete_flat_evidence(tmp_path, monkeypatch):
    result = _replay(tmp_path, monkeypatch)
    assert result["source_status"]["zero_kill"] is False
    assert result["source_status"]["source_hashes_complete"] is True
    rows = result["components"]
    assert len(rows) == 2
    assert rows[0]["e_h60"] == 119_999  # next component start is source-derived
    assert rows[0]["valid_h60"] == 1  # corrupted baseline endpoint/validity is ignored
    assert rows[0]["c0_bits"] != rows[0]["call_bits"]
    assert rows[0]["call_n_unknown_team"] == 0
    for tag in REF.REFERENCE_PARTICIPATION_TAGS:
        assert f"_dists_{tag}" in rows[0]
        assert f"support_ok_{tag}" in rows[0]
    for tag in REF.REFERENCE_POST_TAGS:
        assert f"F_post_{tag}" in rows[0]
        assert f"d_xp_mean_{tag}" in rows[0]
    table, evidence = _table_and_evidence(rows)
    compared = REF.compare_table_to_reference(table, evidence, rows)
    assert compared["all_ok"] is True
    assert compared["coverage_ok"] is True
    assert compared["unsupported_reference_rows"] == 0
    assert compared["expected_comparisons_by_field"]["c0_bits"] == len(rows)
    assert compared["compared_by_field"]["c0_bits"] == len(rows)


def test_early_cutoff_preserves_complete_unknown_post_evidence(tmp_path, monkeypatch):
    match_id, arrays, blue_pids = _fixture(tmp_path, monkeypatch)
    arrays["t"][0] = 5_000
    event_path = C.CACHE_DIR / f"{match_id}.events.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    events[0]["timestamp"] = 5_000
    event_path.write_text(json.dumps(events), encoding="utf-8")
    result = REF.replay_match_components(match_local=4, match_id=match_id,
                **arrays, blue_pids=blue_pids, baseline_rows=None)
    early = result["components"][0]
    assert early["a_strict"] == -1 and early["valid_h60"] == 0
    assert early["endpoint_invalid_reason_h60"]
    for tag in REF.REFERENCE_POST_TAGS:
        assert early[f"post_support_reason_{tag}"] == "invalid_endpoint"
        assert np.isnan(early[f"_pre_gold_{tag}"]).all()
        assert np.isnan(early[f"_post_xp_mean_{tag}"])
    table, evidence = _table_and_evidence(result["components"])
    assert REF.compare_table_to_reference(table, evidence, result["components"])["all_ok"]


def test_compare_catches_scientific_field_corruptions(tmp_path, monkeypatch):
    result = _replay(tmp_path, monkeypatch)
    rows = result["components"]
    table, evidence = _table_and_evidence(rows)
    mutations = {
        "c0_bits": lambda t: t["c0_bits"].__setitem__(0, int(t["c0_bits"][0]) ^ (1 << 8)),
        "call_bits": lambda t: t["call_bits"].__setitem__(0, int(t["call_bits"][0]) ^ (1 << 9)),
        "call_n": lambda t: t["call_n"].__setitem__(0, int(t["call_n"][0]) + 1),
        "c0_n_blue": lambda t: t["c0_n_blue"].__setitem__(0, int(t["c0_n_blue"][0]) + 1),
        "F_tau": lambda t: t["F_tau"].__setitem__(0, int(t["F_tau"][0]) + 1),
        "F_tm1": lambda t: t["F_tm1"].__setitem__(0, int(t["F_tm1"][0]) + 1),
        "foreign_kills_in_tau_L": lambda t: t["foreign_kills_in_tau_L"].__setitem__(0, 99),
        "support_ok_tau_r1800": lambda t: t["support_ok_tau_r1800"].__setitem__(0, 0),
        "p_lower_bits_tau_r1800": lambda t: t["p_lower_bits_tau_r1800"].__setitem__(0, int(t["p_lower_bits_tau_r1800"][0]) ^ (1 << 7)),
        "e_h60": lambda t: t["e_h60"].__setitem__(0, int(t["e_h60"][0]) + 1),
        "valid_h60": lambda t: t["valid_h60"].__setitem__(0, 0),
        "F_post_tau_h60": lambda t: t["F_post_tau_h60"].__setitem__(0, int(t["F_post_tau_h60"][0]) + 1),
        "async_events_cutoff_to_endpoint_tau_h60": lambda t: t["async_events_cutoff_to_endpoint_tau_h60"].__setitem__(0, int(t["async_events_cutoff_to_endpoint_tau_h60"][0]) + 1),
        "d_gold_blue_tau_h60": lambda t: t["d_gold_blue_tau_h60"].__setitem__(0, float(t["d_gold_blue_tau_h60"][0]) + 1),
    }
    for field, mutate in mutations.items():
        mutated = {key: value.copy() for key, value in table.items()}
        mutate(mutated)
        cmp = REF.compare_table_to_reference(mutated, evidence, rows)
        assert cmp["all_ok"] is False, field
        assert cmp["mismatches_by_field"].get(field, 0) > 0, field
    dist_bad = {key: value.copy() for key, value in evidence.items()}
    dist_bad["dist_tau_r1800"][0, 0] += 1.0
    cmp = REF.compare_table_to_reference(table, dist_bad, rows)
    assert cmp["all_ok"] is False
    assert cmp["mismatches_by_field"].get("dist_tau_r1800", 0) > 0

    for field in ("first_loc_x", "dist_tm1_r2000", "pre_gold_tau_h60"):
        evidence_bad = {key: value.copy() for key, value in evidence.items()}
        evidence_bad[field].flat[0] = 12345.0
        cmp = REF.compare_table_to_reference(table, evidence_bad, rows)
        assert cmp["all_ok"] is False, field
        assert cmp["mismatches_by_field"].get(field, 0) > 0, field


def test_compare_rejects_schema_key_and_none_coverage_errors(tmp_path, monkeypatch):
    result = _replay(tmp_path, monkeypatch)
    rows = result["components"]
    table, evidence = _table_and_evidence(rows)
    missing = dict(table)
    missing.pop("call_bits")
    cmp = REF.compare_table_to_reference(missing, evidence, rows)
    assert cmp["all_ok"] is False
    assert "call_bits" in cmp["missing_table_fields"]

    duplicate = rows + [dict(rows[0])]
    cmp = REF.compare_table_to_reference(table, evidence, duplicate)
    assert cmp["all_ok"] is False
    assert cmp["duplicate_reference_keys"] == 1

    short_evidence = dict(evidence)
    short_evidence["dist_tau_r1800"] = short_evidence["dist_tau_r1800"][:-1]
    cmp = REF.compare_table_to_reference(table, short_evidence, rows)
    assert cmp["all_ok"] is False


def test_feature_mapping_failure_does_not_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    result = REF.confirm_feature_channel_mapping()
    assert result["ok"] is False
    assert "fallback" not in result


def test_feature_mapping_preserves_import_path_and_spawn_workers():
    import sys
    from concurrent.futures import ProcessPoolExecutor
    from studies.observation_support_20261002 import run_diagnostic as RUN

    before = sys.path[:]
    assert REF.confirm_feature_channel_mapping()["ok"]
    assert sys.path == before
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = [future.result(timeout=30) for future in
                   [pool.submit(RUN._utc) for _ in range(3)]]
    assert len(results) == 3 and all(value.endswith("+00:00") for value in results)


def test_feature_mapping_restores_import_path_on_source_error(monkeypatch, tmp_path):
    import sys

    source = tmp_path / "worktrees" / "engagement-state-value" / "core" / "config.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('fixture config failure')", encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    before = sys.path[:]
    result = REF.confirm_feature_channel_mapping()
    assert not result["ok"] and "fixture config failure" in result["error"]
    assert sys.path == before


def test_direct_cache_loader_rejects_float_minute_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    np.savez(
        tmp_path / "bad.npz",
        minute_ts=np.asarray([0.0, 60_000.0], dtype=np.float64),
        xy_raw_minute=np.zeros((2, 10, 2), dtype=np.float32),
        node_minute=np.zeros((2, 10, 76), dtype=np.float32),
        gold_team_minute=np.zeros((2, 2), dtype=np.float32),
    )
    pack, reason, digest = REF._load_cache_direct("bad")
    assert pack is None
    assert reason == "bad_dtype:minute_ts_not_integer"
    assert digest


def test_zero_kill_returns_source_diagnostics(tmp_path, monkeypatch):
    match_id, arrays, blue_pids = _fixture(tmp_path, monkeypatch)
    empty = {key: value[:0] for key, value in arrays.items()}
    result = REF.replay_match_components(
        match_local=4,
        match_id=match_id,
        **empty,
        blue_pids=blue_pids,
        baseline_rows={"component_id": np.asarray([], dtype=np.int64)},
    )
    assert result["components"] == []
    status = result["source_status"]
    assert status["zero_kill"] is True
    assert status["n_kills"] == 0
    assert status["n_components"] == 0
    assert status["source_hashes_complete"] is True
