from __future__ import annotations

import numpy as np

from studies.population_adoption_20261002.projection import _derive_direct_opposition, _rows_from_arrays
from studies.population_adoption_20261002.tests.test_contracts import _row


def test_direct_join_requires_opposing_killer_or_assist(tmp_path):
    ledger = tmp_path / "source"
    batches = tmp_path / "batches"
    ledger.mkdir()
    batches.mkdir()
    np.savez(
        ledger / "event_ledger_train.npz",
        t_ms=np.asarray([100, 100], dtype=np.int64),
        killer=np.asarray([0, 1], dtype=np.int64),
        victim=np.asarray([6, 7], dtype=np.int64),
        assist_mask=np.asarray([0, 0], dtype=np.int64),
    )
    np.savez(ledger / "ledger_match_ranges.npz", starts=np.asarray([0]), ends=np.asarray([2]))
    np.savez(ledger / "train_match_meta.npz", blue_pids=np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64))
    np.savez(
        batches / "batch_0000.npz",
        match_locals=np.asarray([0], dtype=np.int32),
        labels_t14000_s2000_m1=np.asarray([0, 0], dtype=np.int64),
    )
    components = {
        "match_local": np.asarray([0], dtype=np.int64),
        "component_id": np.asarray([0], dtype=np.int64),
        "f": np.asarray([100], dtype=np.int64),
    }
    status, meta = _derive_direct_opposition(components, ledger_dir=ledger, batch_dir=batches)
    assert status.tolist() == ["pass"]
    assert meta["status"] == "pass"


def test_direct_join_unknown_when_frozen_sources_are_missing(tmp_path):
    components = {
        "match_local": np.asarray([0], dtype=np.int64),
        "component_id": np.asarray([0], dtype=np.int64),
        "f": np.asarray([100], dtype=np.int64),
    }
    status, meta = _derive_direct_opposition(components, ledger_dir=tmp_path, batch_dir=tmp_path)
    assert status.tolist() == ["unknown"]
    assert meta["status"] == "unknown"


def test_vector_projection_keeps_strict_time_and_credit_unknowns():
    rows = [
        _row(anon_match_key="m0", member_key="m0:c0"),
        _row(anon_match_key="m1", member_key="m1:c0", f=-1, L=-1, tau=-11, F_tau=-10011),
        _row(anon_match_key="m2", member_key="m2:c0", L=129999),
        _row(anon_match_key="m3", member_key="m3:c0", F_tau=119999),
        _row(anon_match_key="m4", member_key="m4:c0", c0_n_blue=-1),
    ]
    for index, row in enumerate(rows):
        row.update(match_local=index, component_id=0)
    arrays = {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }
    arrays["firstkill_direct_opposition"] = np.asarray(["pass"] * len(rows))
    projected = _rows_from_arrays(arrays, "pass")
    assert projected["candidate_eligible"].dtype.kind == "b"
    assert projected["supported_candidate_eligible"].dtype.kind == "b"
    assert bool(projected["candidate_eligible"][0]) is True
    assert bool(projected["candidate_eligible"][1]) is False
    assert projected["timing_status"][1] == "unknown"
    assert projected["timing_status"][2] == "fail"
    assert projected["timing_status"][3] == "unknown"
    assert projected["critical_source_status"][4] == "unknown"


def test_direct_join_parses_entire_first_block_before_relation_or(tmp_path):
    ledger = tmp_path / "source"
    batches = tmp_path / "batches"
    ledger.mkdir()
    batches.mkdir()
    np.savez(
        ledger / "event_ledger_train.npz",
        t_ms=np.asarray([100, 100], dtype=np.int64),
        killer=np.asarray([1, 2], dtype=np.int64),
        victim=np.asarray([6, 7], dtype=np.int64),
        assist_mask=np.asarray([0, -1], dtype=np.int64),
    )
    np.savez(ledger / "ledger_match_ranges.npz", starts=np.asarray([0]), ends=np.asarray([2]))
    np.savez(ledger / "train_match_meta.npz", blue_pids=np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64))
    np.savez(
        batches / "batch_0000.npz",
        match_locals=np.asarray([0], dtype=np.int32),
        labels_t14000_s2000_m1=np.asarray([0, 0], dtype=np.int64),
    )
    status, meta = _derive_direct_opposition(
        {"match_local": np.asarray([0]), "component_id": np.asarray([0]), "f": np.asarray([100])},
        ledger_dir=ledger,
        batch_dir=batches,
    )
    assert status.tolist() == ["unknown"]
    assert meta["credit_parse_status"].tolist() == ["unknown"]


def test_direct_join_rejects_noninteger_ledger_fields(tmp_path):
    ledger = tmp_path / "source"
    batches = tmp_path / "batches"
    ledger.mkdir()
    batches.mkdir()
    np.savez(
        ledger / "event_ledger_train.npz",
        t_ms=np.asarray([100.5]),
        killer=np.asarray([1.0]),
        victim=np.asarray([6.0]),
        assist_mask=np.asarray([0.0]),
    )
    np.savez(ledger / "ledger_match_ranges.npz", starts=np.asarray([0]), ends=np.asarray([1]))
    np.savez(ledger / "train_match_meta.npz", blue_pids=np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64))
    np.savez(
        batches / "batch_0000.npz",
        match_locals=np.asarray([0], dtype=np.int32),
        labels_t14000_s2000_m1=np.asarray([0], dtype=np.int64),
    )
    status, meta = _derive_direct_opposition(
        {"match_local": np.asarray([0]), "component_id": np.asarray([0]), "f": np.asarray([100])},
        ledger_dir=ledger,
        batch_dir=batches,
    )
    assert status.tolist() == ["unknown"]
    assert meta["reason"] == "direct_firstkill_source_noninteger_ledger"
