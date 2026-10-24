"""Acceptance-module and reusable-source identity tests (tiny fixtures only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from studies.objective_timing_20261002 import config as C
from studies.objective_timing_20261002 import io_util as IO
from studies.objective_timing_20261002.acceptance import arrays as ARR
from studies.objective_timing_20261002.acceptance import pipeline as PIPE
from studies.objective_timing_20261002.acceptance import source as SRC
from studies.objective_timing_20261002.acceptance import __main__ as ACC_MAIN


def _comp(**kwargs):
    n = int(kwargs.pop("n", 2))
    base = {
        "match_local": np.asarray([0, 0], dtype=np.int64)[:n],
        "component_id": np.arange(n, dtype=np.int64),
        "L": np.asarray([1000, 2000], dtype=np.int64)[:n],
        "game_end_ms": np.asarray([10_000, 10_000], dtype=np.int64)[:n],
        "game_end_known": np.ones(n, dtype=np.int8),
        "next_kill_after_L": np.asarray([5000, 5000], dtype=np.int64)[:n],
        "next_component_tau": np.asarray([6000, 6000], dtype=np.int64)[:n],
    }
    base.update(kwargs)
    return base


def _ev(t_ms, match_local, event_ord, team, mask, acq=None):
    t = np.asarray(t_ms, dtype=np.int64)
    n = t.size
    return {
        "t_ms": t,
        "match_local": np.asarray(match_local, dtype=np.int64),
        "event_ord": np.asarray(event_ord, dtype=np.int64),
        "team": np.asarray(team, dtype=np.int16),
        "category_mask": np.asarray(mask, dtype=np.int64),
        "acq_status": np.asarray(
            acq if acq is not None else [C.ACQ_CONFIRMED] * n, dtype=np.int8
        ),
    }


def test_query_detects_changed_component_keys():
    comps = _comp()
    # Baron bit 7 + any_objective + any_epic
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev = _ev([1500], [0], [1], [100], [mask])
    expected, _ = ARR.expected_first_event_queries(comps, ev)
    q = dict(expected)
    q["L"] = q["L"].copy()
    q["L"][0] = 999  # tamper component key
    out = ARR.verify_query_identity(comps, ev, q)
    assert out["all_ok"] is False
    assert any(f[0] == "L" for f in out["failures_head"])


def test_query_detects_changed_event_time_team_category():
    comps = _comp(n=1, match_local=np.asarray([0]), L=np.asarray([1000]),
                  game_end_ms=np.asarray([9000]), next_kill_after_L=np.asarray([8000]),
                  next_component_tau=np.asarray([8000]), component_id=np.asarray([0]),
                  game_end_known=np.ones(1, dtype=np.int8))
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev = _ev([1500], [0], [3], [100], [mask])
    expected, _ = ARR.expected_first_event_queries(comps, ev)
    q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in expected.items()}
    key = f"t_ms__{C.INTERVAL_UNCAPED}:baron"
    q[key] = q[key].copy()
    q[key][0] = 1600
    assert ARR.compare_query_arrays(q, expected)["all_ok"] is False


def test_missing_earlier_first_event_changes_expectation():
    comps = _comp(n=1, match_local=np.asarray([0]), L=np.asarray([1000]),
                  game_end_ms=np.asarray([9000]), next_kill_after_L=np.asarray([8000]),
                  next_component_tau=np.asarray([8000]), component_id=np.asarray([0]),
                  game_end_known=np.ones(1, dtype=np.int8))
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev_late = _ev([3000], [0], [2], [100], [mask])
    ev_both = _ev([1500, 3000], [0, 0], [1, 2], [100, 100], [mask, mask])
    exp_late, _ = ARR.expected_first_event_queries(comps, ev_late)
    exp_both, _ = ARR.expected_first_event_queries(comps, ev_both)
    key = f"t_ms__{C.INTERVAL_UNCAPED}:baron"
    assert int(exp_late[key][0]) == 3000
    assert int(exp_both[key][0]) == 1500
    assert ARR.compare_query_arrays(exp_late, exp_both)["all_ok"] is False


def test_cross_match_spillover_excluded():
    comps = _comp(
        n=1,
        match_local=np.asarray([0], dtype=np.int64),
        component_id=np.asarray([0], dtype=np.int64),
        L=np.asarray([1000], dtype=np.int64),
        game_end_ms=np.asarray([9000], dtype=np.int64),
        game_end_known=np.ones(1, dtype=np.int8),
        next_kill_after_L=np.asarray([8000], dtype=np.int64),
        next_component_tau=np.asarray([8000], dtype=np.int64),
    )
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    # Event belongs to match 1 only — must not satisfy match 0 query
    ev = _ev([1500], [1], [1], [100], [mask])
    expected, _ = ARR.expected_first_event_queries(comps, ev)
    st = expected[f"status__{C.INTERVAL_UNCAPED}:baron"]
    assert int(st[0]) == C.ST_NO_ACQ


def test_equal_L_excluded_and_pre_next_boundary():
    comps = _comp(
        n=1,
        match_local=np.asarray([0], dtype=np.int64),
        component_id=np.asarray([0], dtype=np.int64),
        L=np.asarray([1000], dtype=np.int64),
        game_end_ms=np.asarray([9000], dtype=np.int64),
        game_end_known=np.ones(1, dtype=np.int8),
        next_kill_after_L=np.asarray([2000], dtype=np.int64),
        next_component_tau=np.asarray([2000], dtype=np.int64),
    )
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev_at_L = _ev([1000], [0], [1], [100], [mask])
    exp, _ = ARR.expected_first_event_queries(comps, ev_at_L)
    assert int(exp[f"status__{C.INTERVAL_UNCAPED}:baron"][0]) == C.ST_NO_ACQ

    ev_at_hi = _ev([2000], [0], [1], [100], [mask])
    exp2, _ = ARR.expected_first_event_queries(comps, ev_at_hi)
    # uncapped allows t<=ge; pre_next requires t<hi
    assert int(exp2[f"status__{C.INTERVAL_UNCAPED}:baron"][0]) == C.ST_OBSERVED
    assert int(exp2[f"status__{C.INTERVAL_PRE_NEXT}:baron"][0]) == C.ST_NO_ACQ


def test_same_ms_ties_count_and_team_bits():
    comps = _comp(
        n=1,
        match_local=np.asarray([0], dtype=np.int64),
        component_id=np.asarray([0], dtype=np.int64),
        L=np.asarray([1000], dtype=np.int64),
        game_end_ms=np.asarray([9000], dtype=np.int64),
        game_end_known=np.ones(1, dtype=np.int8),
        next_kill_after_L=np.asarray([8000], dtype=np.int64),
        next_component_tau=np.asarray([8000], dtype=np.int64),
    )
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev = _ev([1500, 1500], [0, 0], [1, 2], [100, 200], [mask, mask])
    exp, _ = ARR.expected_first_event_queries(comps, ev)
    assert int(exp[f"n_tie_events__{C.INTERVAL_UNCAPED}:baron"][0]) == 2
    # team bits 2|4 = 6
    assert int(exp[f"tie_teams__{C.INTERVAL_UNCAPED}:baron"][0]) == 6
    assert int(exp[f"event_ord__{C.INTERVAL_UNCAPED}:baron"][0]) == 1  # first ord


def test_empty_category_all_zero_observations():
    comps = _comp(
        n=2,
        match_local=np.asarray([0, 1], dtype=np.int64),
        component_id=np.asarray([0, 1], dtype=np.int64),
        L=np.asarray([1000, 1000], dtype=np.int64),
        game_end_ms=np.asarray([9000, 9000], dtype=np.int64),
        game_end_known=np.ones(2, dtype=np.int8),
        next_kill_after_L=np.asarray([8000, 8000], dtype=np.int64),
        next_component_tau=np.asarray([8000, 8000], dtype=np.int64),
    )
    # only baron events; atakhan slice must be all NO_ACQ
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev = _ev([1500], [0], [1], [100], [mask])
    exp, _ = ARR.expected_first_event_queries(comps, ev)
    st = exp[f"status__{C.INTERVAL_UNCAPED}:atakhan"]
    assert np.all(st == C.ST_NO_ACQ)


def test_classify_neutral_vs_unknown():
    team, status, _ = SRC.classify_raw_objective(
        "ELITE_MONSTER_KILL",
        {"killerTeamId": 300, "killerId": 0, "monsterType": "BARON_NASHOR"},
    )
    assert team == 0 and status == C.ACQ_NEUTRAL
    team2, status2, _ = SRC.classify_raw_objective(
        "ELITE_MONSTER_KILL",
        {"killerTeamId": 0, "killerId": 0, "monsterType": "BARON_NASHOR"},
    )
    assert status2 == C.ACQ_UNKNOWN and team2 == 0


def test_raw_audit_malformed_non_object_and_bad_timestamp(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "m0.events.json").write_text(
        json.dumps(
            [
                "not-an-object",
                {"type": "CHAMPION_KILL", "timestamp": 1.5},
                {"type": "ELITE_MONSTER_KILL", "timestamp": True, "monsterType": "BARON_NASHOR", "killerTeamId": 100},
            ]
        ),
        encoding="utf-8",
    )
    terminals = [{"n_kills": 0, "game_end_all": [], "game_end_ms": -1}]
    ev = _ev([], [], [], [], [])
    out = SRC.audit_raw_sources(
        match_ids=["m0"],
        terminals=terminals,
        kill_match_local=np.zeros(0, dtype=np.int64),
        kill_t_ms=np.zeros(0, dtype=np.int64),
        events={
            "match_local": np.zeros(0, dtype=np.int64),
            "event_ord": np.zeros(0, dtype=np.int64),
            "t_ms": np.zeros(0, dtype=np.int64),
            "team": np.zeros(0, dtype=np.int16),
            "acq_status": np.zeros(0, dtype=np.int8),
            "category_mask": np.zeros(0, dtype=np.int64),
        },
        cache_dir=cache,
    )
    assert out["all_ok"] is False
    kinds = {tuple(f[:2]) for f in out["failures_head"]}
    assert (0, "non_object") in kinds
    assert any(f[1] == "kill_timestamp" for f in out["failures_head"])


def test_out_dir_collision_rejected(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    (out / "marker.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        PIPE._reject_out_dir(out)


def _write_source_manifest(path: Path, **overrides):
    man = {
        "study": "objective_timing_20261002",
        "complete": True,
        "max_matches": None,
        "n_matches": C.EXPECTED_TRAIN_MATCHES,
        "n_components": C.EXPECTED_MIN1_COMPONENTS,
        "gates": {"full_corpus": True},
        "source": {
            "anchor_manifest_sha256": C.EXPECTED_ANCHOR_MANIFEST_SHA256,
            "components_sha256": C.EXPECTED_COMPONENTS_SHA256,
            "terminals_sha256": C.EXPECTED_TERMINALS_SHA256,
        },
        "artifacts": {},
    }
    man.update(overrides)
    path.write_text(json.dumps(man), encoding="utf-8")


def test_reuse_source_missing_manifest(tmp_path):
    checks = IO.verify_reusable_source_run(tmp_path)
    assert checks["all_ok"] is False
    assert "missing_source_manifest" in checks["failures"]


def test_reuse_source_hash_mismatch(tmp_path):
    ev = tmp_path / "objective_events.npz"
    np.savez_compressed(ev, t_ms=np.asarray([1], dtype=np.int64))
    _write_source_manifest(
        tmp_path / "RUN_MANIFEST.json",
        artifacts={"objective_events_sha256": "0" * 64},
    )
    checks = IO.verify_reusable_source_run(tmp_path)
    assert checks["all_ok"] is False
    assert "objective_events_sha_mismatch" in checks["failures"]


def test_reuse_source_incomplete_and_smoke(tmp_path):
    ev = tmp_path / "objective_events.npz"
    np.savez_compressed(ev, t_ms=np.asarray([1], dtype=np.int64))
    sha = IO.file_sha256(ev)
    _write_source_manifest(
        tmp_path / "RUN_MANIFEST.json",
        complete=False,
        artifacts={"objective_events_sha256": sha},
    )
    checks = IO.verify_reusable_source_run(tmp_path)
    assert checks["all_ok"] is False
    assert "source_manifest_incomplete" in checks["failures"]

    _write_source_manifest(
        tmp_path / "RUN_MANIFEST.json",
        complete=True,
        max_matches=10,
        artifacts={"objective_events_sha256": sha},
    )
    checks2 = IO.verify_reusable_source_run(tmp_path)
    assert checks2["all_ok"] is False
    assert "smoke_or_partial_source" in checks2["failures"]


def test_reuse_source_accepts_matching_full_manifest(tmp_path):
    ev = tmp_path / "objective_events.npz"
    np.savez_compressed(ev, t_ms=np.asarray([1], dtype=np.int64))
    sha = IO.file_sha256(ev)
    _write_source_manifest(
        tmp_path / "RUN_MANIFEST.json",
        artifacts={"objective_events_sha256": sha},
    )
    checks = IO.verify_reusable_source_run(tmp_path)
    assert checks["all_ok"] is True
    assert checks["events_sha_matches_manifest"] is True


def test_no_interval_when_hi_le_L():
    comps = _comp(
        n=1,
        match_local=np.asarray([0], dtype=np.int64),
        component_id=np.asarray([0], dtype=np.int64),
        L=np.asarray([5000], dtype=np.int64),
        game_end_ms=np.asarray([9000], dtype=np.int64),
        game_end_known=np.ones(1, dtype=np.int8),
        next_kill_after_L=np.asarray([4000], dtype=np.int64),
        next_component_tau=np.asarray([4000], dtype=np.int64),
    )
    mask = (1 << 0) | (1 << 1) | (1 << 7)
    ev = _ev([6000], [0], [1], [100], [mask])
    exp, _ = ARR.expected_first_event_queries(comps, ev)
    assert int(exp[f"status__{C.INTERVAL_PRE_NEXT}:baron"][0]) == C.ST_NO_INTERVAL


def _make_scoped_run(path: Path, **manifest_overrides) -> Path:
    """Tiny target run with events + RUN_MANIFEST for scope-gate tests."""
    path.mkdir(parents=True, exist_ok=True)
    ev = path / "objective_events.npz"
    np.savez_compressed(ev, t_ms=np.asarray([1], dtype=np.int64))
    sha = IO.file_sha256(ev)
    _write_source_manifest(
        path / "RUN_MANIFEST.json",
        artifacts={"objective_events_sha256": sha},
        **manifest_overrides,
    )
    return path


def test_acceptance_rejects_incomplete_target_before_raw(tmp_path, monkeypatch):
    run = _make_scoped_run(tmp_path / "run", complete=False)
    out = tmp_path / "out"
    called: list[str] = []

    def _raw_should_not_run(**_kwargs):
        called.append("raw")
        raise AssertionError("raw audit must not run after scope rejection")

    monkeypatch.setattr(PIPE.SRC, "audit_raw_sources", _raw_should_not_run)
    result = PIPE.run_acceptance(run, out, argv=["test"], skip_raw_source=False)
    assert result["accepted"] is False
    assert result["complete"] is False
    assert result["stage"] == "target_scope"
    assert result["error"] == "target_scope_rejected"
    assert "source_manifest_incomplete" in result["failure_evidence"]["failures"]
    assert called == []
    assert not (out / "full_raw_source_review.json").exists()
    man = json.loads((out / "ACCEPTANCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["accepted"] is False
    assert (out / "acceptance.log").is_file()
    assert "target_scope" in (out / "acceptance.log").read_text(encoding="utf-8")


def test_acceptance_rejects_smoke_and_count_and_source_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        PIPE.SRC,
        "audit_raw_sources",
        lambda **_k: (_ for _ in ()).throw(AssertionError("raw must not run")),
    )

    # smoke / limited max_matches
    run_smoke = _make_scoped_run(tmp_path / "smoke", max_matches=10)
    out_smoke = tmp_path / "out_smoke"
    r1 = PIPE.run_acceptance(run_smoke, out_smoke, argv=["test"])
    assert r1["accepted"] is False and r1["stage"] == "target_scope"
    assert "smoke_or_partial_source" in r1["failure_evidence"]["failures"]

    # count mismatch
    run_cnt = _make_scoped_run(tmp_path / "counts", n_matches=1)
    out_cnt = tmp_path / "out_counts"
    r2 = PIPE.run_acceptance(run_cnt, out_cnt, argv=["test"])
    assert r2["accepted"] is False and r2["stage"] == "target_scope"
    assert "n_matches_mismatch" in r2["failure_evidence"]["failures"]

    # source baseline hash mismatch (same events file, wrong recorded source refs)
    run_src = _make_scoped_run(
        tmp_path / "src",
        source={
            "anchor_manifest_sha256": "0" * 64,
            "components_sha256": C.EXPECTED_COMPONENTS_SHA256,
            "terminals_sha256": C.EXPECTED_TERMINALS_SHA256,
        },
    )
    out_src = tmp_path / "out_src"
    r3 = PIPE.run_acceptance(run_src, out_src, argv=["test"])
    assert r3["accepted"] is False and r3["stage"] == "target_scope"
    assert "baseline_refs_mismatch" in r3["failure_evidence"]["failures"]


def test_acceptance_persists_missing_manifest_and_corrupt_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(
        PIPE.SRC,
        "audit_raw_sources",
        lambda **_k: (_ for _ in ()).throw(AssertionError("raw must not run")),
    )

    # missing manifest
    run_miss = tmp_path / "missing"
    run_miss.mkdir()
    np.savez_compressed(run_miss / "objective_events.npz", t_ms=np.asarray([1], dtype=np.int64))
    out_miss = tmp_path / "out_missing"
    r1 = PIPE.run_acceptance(run_miss, out_miss, argv=["test"])
    assert r1["accepted"] is False and r1["complete"] is False
    assert r1["error"] == "missing_manifest"
    assert (out_miss / "ACCEPTANCE_MANIFEST.json").is_file()
    assert (out_miss / "acceptance.log").is_file()

    # corrupt manifest JSON
    run_bad = tmp_path / "bad_json"
    run_bad.mkdir()
    (run_bad / "RUN_MANIFEST.json").write_text("{not-json", encoding="utf-8")
    np.savez_compressed(run_bad / "objective_events.npz", t_ms=np.asarray([1], dtype=np.int64))
    out_bad = tmp_path / "out_bad_json"
    r2 = PIPE.run_acceptance(run_bad, out_bad, argv=["test"])
    assert r2["accepted"] is False
    assert r2["error"] == "corrupt_manifest"
    assert (out_bad / "ACCEPTANCE_MANIFEST.json").is_file()

    # scope-ok run but corrupt/missing component_queries after scope gate
    run_ok = _make_scoped_run(tmp_path / "scoped_ok")
    # also need matching artifact hashes for events only — scope passes; artifact stage fails
    # or pass artifact stage by writing hashes for missing files → fail at load
    man = json.loads((run_ok / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    # Provide fake hashes so scope (events only) passes; artifact_hashes will fail on missing files
    man["artifacts"] = {
        "objective_events_sha256": IO.file_sha256(run_ok / "objective_events.npz"),
        "component_queries_sha256": "a" * 64,
        "aggregates_sha256": "b" * 64,
        "schema_census_sha256": "c" * 64,
    }
    (run_ok / "RUN_MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    out_art = tmp_path / "out_art"
    r3 = PIPE.run_acceptance(run_ok, out_art, argv=["test"])
    assert r3["accepted"] is False
    assert r3["stage"] == "artifact_hashes"
    assert (out_art / "ACCEPTANCE_MANIFEST.json").is_file()
    assert "artifact_hash" in (out_art / "acceptance.log").read_text(encoding="utf-8")


def test_acceptance_schema_key_exception_persists(tmp_path, monkeypatch):
    """Validation exception after safe out_dir is created must persist failure evidence."""
    run = _make_scoped_run(tmp_path / "run")
    out = tmp_path / "out"

    def _boom(*_a, **_k):
        raise KeyError("missing_schema_key")

    monkeypatch.setattr(PIPE.IO, "verify_reusable_source_run", _boom)
    result = PIPE.run_acceptance(run, out, argv=["test"])
    assert result["accepted"] is False
    assert result["complete"] is False
    assert result["error"] == "validation_exception"
    assert result["stage"] == "target_scope"
    assert "KeyError" in result["detail"]
    assert "KeyError" in result["failure_evidence"]["exception_type"]
    man = json.loads((out / "ACCEPTANCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["accepted"] is False
    log_text = (out / "acceptance.log").read_text(encoding="utf-8")
    assert "exception" in log_text.lower() or "KeyError" in log_text


def test_out_dir_descendant_of_target_rejected_no_write(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    nested = run / "acceptance_nested"
    with pytest.raises(ValueError):
        PIPE.run_acceptance(run, nested, argv=["test"])
    assert not nested.exists()


def test_out_dir_collision_cli_no_write(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "exists"
    out.mkdir()
    marker = out / "prior_evidence.json"
    marker.write_text('{"keep": true}', encoding="utf-8")
    rc = ACC_MAIN.main(["--run-dir", str(run), "--out-dir", str(out)])
    assert rc == 2
    assert marker.read_text(encoding="utf-8") == '{"keep": true}'
    assert not (out / "ACCEPTANCE_MANIFEST.json").exists()


def test_is_same_or_descendant_case_normalized(tmp_path):
    base = tmp_path.resolve()
    sub = (tmp_path / "child").resolve()
    assert PIPE.is_same_or_descendant(base, base) is True
    assert PIPE.is_same_or_descendant(sub, base) is True
    assert PIPE.is_same_or_descendant(base.parent / (base.name + "_other"), base) is False
    # Windows: different case of the same path must still count as same/descendant
    if PIPE.os.name == "nt":
        mixed = Path(str(base).swapcase()) / "child"
        assert PIPE.is_same_or_descendant(mixed, base) is True
        assert PIPE.is_same_or_descendant(Path(str(sub).swapcase()), base) is True
