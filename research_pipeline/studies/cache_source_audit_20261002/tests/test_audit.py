from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np

from studies.cache_source_audit_20261002.audit import aggregate_rows, audit_match_bytes
from studies.cache_source_audit_20261002.champion_map import load_champion_map
from studies.cache_source_audit_20261002.run_audit import run_audit


def _team_map() -> dict[str, int]:
    return {str(pid): (100 if pid <= 5 else 200) for pid in range(1, 11)}


def _meta(match_id: str, patch: str) -> dict:
    return {
        "match_id": match_id,
        "patch": patch,
        "patch_full": f"{patch}.999.9999",
        "team_map": _team_map(),
        "static_meta": {
            "champion_by_pid": {str(pid): pid for pid in range(1, 11)},
            "champion_name_by_pid": {str(pid): f"champion-{pid}" for pid in range(1, 11)},
            "summoner_spells_by_pid": {
                str(pid): {"summoner_spell_1_id": 1, "summoner_spell_2_id": 2}
                for pid in range(1, 11)
            },
            "runes_by_pid": {
                str(pid): {"primary_style_id": 8000, "sub_style_id": 8100}
                for pid in range(1, 11)
            },
            "bans": {"blue": [1, 2, 3, 4, 5], "red": [6, 7, 8, 9, 10]},
        },
        "feature_version": "fixture-v1",
        "interp_policy": {"frame_ms": 60_000, "interp_method": "zoh"},
    }


def _npz_bytes(minute_ts=(0, 60_000, 120_000), *, omit=(), node_nan=False, node_width=76) -> bytes:
    timestamps = np.asarray(minute_ts, dtype=np.int64)
    n = len(timestamps)
    arrays = {
        "minute_ts": timestamps,
        "node_minute": np.zeros((n, 10, node_width), dtype=np.float32),
        "global_minute": np.zeros((n, 26), dtype=np.float32),
        "gold_team_minute": np.zeros((n, 2), dtype=np.float32),
        "xy_raw_minute": np.zeros((n, 10, 2), dtype=np.float32),
    }
    if node_nan:
        arrays["node_minute"][0, 0, 0] = np.nan
    for key in omit:
        arrays.pop(key, None)
    stream = BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _events_bytes(*, timestamp=120_000, winner=100, wrapper=False) -> bytes:
    events = [
        {
            "type": "CHAMPION_KILL",
            "timestamp": 60_000,
            "realTimestamp": 60_000,
            "killerId": 1,
            "victimId": 6,
            "assistingParticipantIds": [2],
            "position": {"x": 100.0, "y": 200.0},
        },
        {"type": "GAME_END", "timestamp": timestamp, "winningTeam": winner, "realTimestamp": timestamp},
    ]
    payload = {"events": events} if wrapper else events
    return json.dumps(payload).encode("utf-8")


def _bytes(match_id="M1", patch="15.14", **kwargs):
    meta = _meta(match_id, patch)
    if "meta_update" in kwargs:
        meta.update(kwargs.pop("meta_update"))
    return {
        "npz_bytes": _npz_bytes(**kwargs.pop("npz_options", {})),
        "meta_bytes": json.dumps(meta).encode("utf-8"),
        "events_bytes": _events_bytes(**kwargs),
    }


def test_valid_schema_retains_unknown_clock_provenance_and_zero_counts():
    row = audit_match_bytes("M1", "15.14", **_bytes())

    assert row["status"] == "unknown"
    assert row["training_ready"] is False and row["final_ready"] is False
    assert row["source_clock"]["status"] == "unknown"
    assert row["npz"]["main_state_ok"] is True
    assert row["npz"]["frame_clock"]["n_index_clock_mismatches"] == 0
    assert row["npz"]["stored_positions"]["n_pairs_zero"] == 30
    assert row["npz"]["gold_values"]["n_frames_zero"] == 3
    assert row["events"]["game_end"]["status"] == "pass"
    assert row["events"]["terminal_frame_order"]["status"] == "pass"


def test_duplicate_or_missing_clock_is_critical_and_never_normalized():
    duplicate = audit_match_bytes("M1", "15.14", **_bytes(npz_options={"minute_ts": (0, 60_000, 60_000)}))
    missing = audit_match_bytes("M2", "15.14", **_bytes(npz_options={"omit": ("minute_ts",)}))

    assert duplicate["status"] == "fail"
    assert "minute_ts_duplicate" in duplicate["reasons"]
    assert "minute_ts_not_strictly_increasing" in duplicate["reasons"]
    assert missing["status"] == "fail"
    assert "missing_minute_ts" in missing["reasons"]
    assert missing["npz"]["frame_clock"]["n_frames"] == 0


def test_metadata_identity_and_team_partition_are_checked():
    bad_team = _team_map()
    bad_team["5"] = 200
    row = audit_match_bytes(
        "M1",
        "15.14",
        **_bytes(meta_update={"match_id": "OTHER", "team_map": bad_team}),
    )

    assert row["status"] == "fail"
    assert "metadata_match_id_mismatch" in row["reasons"]
    assert "team_map_not_exact_5v5_pid_partition" in row["reasons"]
    assert row["team_map"]["team_counts"] == {"100": 4, "200": 6}


def test_malformed_terminal_retains_unknown_status_and_does_not_fallback_to_last_frame():
    row = audit_match_bytes("M1", "15.14", **_bytes(timestamp="bad", winner=300))

    assert row["status"] == "fail"
    assert row["source_clock"]["status"] == "unknown"
    assert row["events"]["game_end"]["status"] == "unknown"
    assert row["events"]["game_end"]["timestamp_ms"] is None
    assert "game_end_missing_malformed_or_winner_unknown" in row["reasons"]
    assert row["events"]["terminal_frame_order"]["status"] == "unknown"


def test_missing_game_end_or_invalid_winner_is_critical():
    payload = _bytes()
    events = json.loads(payload["events_bytes"])
    payload["events_bytes"] = json.dumps([events[0]]).encode("utf-8")
    missing = audit_match_bytes("M1", "15.14", **payload)
    assert missing["status"] == "fail"
    assert "game_end_missing" in missing["reasons"]
    assert missing["events"]["game_end"]["status"] == "fail"

    invalid = audit_match_bytes("M1", "15.14", **_bytes(winner=300))
    assert invalid["status"] == "fail"
    assert "game_end_winner_invalid" in invalid["reasons"]
    assert invalid["events"]["game_end"]["status"] == "fail"
    assert invalid["events"]["game_end"]["timestamp_ms"] == 120_000


def test_static_maps_and_bans_require_mapping_schema_and_exact_pid_keys():
    meta = _meta("M1", "15.14")
    meta["static_meta"]["champion_by_pid"] = []
    meta["static_meta"]["bans"] = None
    payload = _bytes()
    payload["meta_bytes"] = json.dumps(meta).encode("utf-8")
    row = audit_match_bytes("M1", "15.14", **payload)

    static = row["static_metadata"]
    assert static["status"] == "unknown"
    assert static["fields"]["champion_by_pid"]["pid_key_set_ok"] is False
    assert static["fields"]["bans"]["structure_ok"] is False


def test_data_dragon_id_name_consistency_diagnostics_are_numeric():
    mapping = load_champion_map("15.14")
    cache_before = load_champion_map.cache_info()
    assert load_champion_map("15.14") is mapping
    assert load_champion_map.cache_info().hits >= cache_before.hits + 1
    known_id, other_id = sorted(mapping["id_to_names"])[:2]
    known_name = mapping["id_to_names"][known_id][0]
    other_name = mapping["id_to_names"][other_id][0]

    def row_for(ids, names):
        payload = _bytes()
        meta = _meta("M1", "15.14")
        meta["static_meta"]["champion_by_pid"] = {str(pid): ids[pid - 1] for pid in range(1, 11)}
        meta["static_meta"]["champion_name_by_pid"] = {str(pid): names[pid - 1] for pid in range(1, 11)}
        payload["meta_bytes"] = json.dumps(meta).encode("utf-8")
        return audit_match_bytes("M1", "15.14", **payload)

    exact = row_for([known_id] * 10, [known_name] * 10)
    assert exact["static_champion_consistency"]["status"] == "pass"
    assert exact["static_champion_consistency"]["n_exact"] == 10

    mismatch = row_for([known_id] * 10, [other_name] + [known_name] * 9)
    assert mismatch["static_champion_consistency"]["status"] == "unknown"
    assert mismatch["static_champion_consistency"]["n_mismatch"] == 1
    assert mismatch["static_champion_consistency"]["n_unknown_id"] == 0

    positive_unknown = row_for([999_999] + [known_id] * 9, ["hash_999999"] + [known_name] * 9)
    diagnostic = positive_unknown["static_champion_consistency"]
    assert diagnostic["status"] == "unknown"
    assert diagnostic["n_unknown_id"] == 1
    assert diagnostic["n_unknown_name"] == 1
    assert all(type(value) is int for key, value in diagnostic.items() if key.startswith("n_"))


def test_channel_full_stats_reconcile_to_field_totals_and_keep_nonfinite_diagnostic():
    row = audit_match_bytes("M1", "15.14", **_bytes(npz_options={"node_nan": True}))
    aggregate = aggregate_rows([row])

    for field in ("minute_ts", "node_minute", "global_minute", "gold_team_minute", "xy_raw_minute"):
        totals = aggregate["npz_field_counts"][field]
        channels = aggregate["npz_channel_counts"][field]
        assert sum(channels["n_values"]) == totals["n_values"]
        assert sum(channels["n_finite"]) == totals["n_finite"]
        assert sum(channels["n_nonfinite"]) == totals["n_nonfinite"]
        assert sum(channels["n_zero"]) == totals["n_zero"]
        assert len(channels["channel_names"]) == len(channels["n_values"])
    assert aggregate["npz_channel_counts"]["node_minute"]["n_nonfinite"][0] == 1
    assert aggregate["static_metadata_id_counts"]["champion_by_pid"]["nonzero"] == 10


def test_run_writes_bounded_hash_only_ledger_and_row_coverage(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    index = {}
    for index_no, (match_id, patch) in enumerate((("M1", "15.14"), ("M2", "15.15"), ("M3", "15.16"))):
        index[match_id] = patch
        payload = _bytes(match_id, patch, npz_options={"node_width": 75} if match_id == "M2" else {})
        (cache / f"{match_id}.npz").write_bytes(payload["npz_bytes"])
        (cache / f"{match_id}.meta.json").write_bytes(payload["meta_bytes"])
        (cache / f"{match_id}.events.json").write_bytes(payload["events_bytes"])
    index_path = tmp_path / "patch_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    output = tmp_path / "run"

    manifest = run_audit(index_path=index_path, cache_dir=cache, output_dir=output, workers=1, strict_index=False)

    assert manifest["run_status"] == "completed"
    assert manifest["full_corpus_scanned"] is True
    feature_binding = manifest["code_binding"]["feature_contract"]
    assert feature_binding["ok"] is True
    assert len(feature_binding["index_to_name"]["node_minute"]) == 76
    assert len(feature_binding["name_to_index"]["global_minute"]) == 26
    assert manifest["source_map"]["production_identity_ok"] is True
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["n_rows"] == 3
    assert aggregate["input_support"]["rows_processed"] == 3
    assert aggregate["npz_channel_width_mismatch_counts"]["node_minute"] == 1
    assert aggregate["terminal_and_kill_diagnostics"]["kill_support_numeric_counts"]["n_kills"] == 3
    champion_aggregate = aggregate["static_champion_consistency"]
    assert champion_aggregate["overall"]["numeric_counts"]["n_participants"] == 30
    assert set(champion_aggregate["by_patch"]) == {"15.14", "15.15", "15.16"}
    lines = (output / "row_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    ledger_record = json.loads(lines[0])
    assert ledger_record["numbers"]["first_frame_ms"] == 0
    assert ledger_record["numbers"]["last_frame_ms"] == 120_000
    assert ledger_record["numbers"]["game_end_winning_team"] == 100
    assert ledger_record["numbers"]["terminal_gap_ms"] == 0
    assert ledger_record["numbers"]["kill_support"]["killerId_valid_player_id"] == 1
    ledger_text = "\n".join(lines)
    assert all(match_id not in ledger_text for match_id in index)
    assert json.loads((output / "STATUS.json").read_text(encoding="utf-8"))["run_status"] == "completed"
