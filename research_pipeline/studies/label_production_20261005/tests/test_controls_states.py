"""Control-end rows (phase 2): plan and parts, the time-extension rule, prefix invariance, round trip, equality gate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from studies.label_production_20261005 import controls_states as cs
from studies.observable_state_20261002 import producer
from studies.observable_state_20261002.bank import ShardBuilder, match_view
from studies.observable_state_20261002.sampling import sample_match_queries
from studies.observable_state_20261002.storage import close_shard, open_shard, write_shard
from studies.observable_state_20261002.tests.test_kernel import _meta
from studies.v_input_projection_20261002.projection import SemanticHistory
from studies.v_model_input_20261003.history import HistoryAccumulator

END_MS = 300000
FRAMES = np.array([0, 120000, 240000, END_MS], dtype=np.int64)
LATE_MS = 290000


# ---------------------------------------------------------------- plan and resumable parts


def _tables():
    matches = np.array([(b"a", 0, 0, 3), (b"b", 0, 1, 3), (b"c", 1, 0, 3)],
                       dtype=[("key", "S64"), ("shard", "<u2"), ("local_match", "<u2"), ("role", "u1")])
    queries = np.array([(0, 100), (0, 200), (0, 300), (1, 150), (1, 250), (2, 400), (2, 500)],
                       dtype=[("match_index", "<u4"), ("query_ms", "<i8")])
    match_queries = np.array([(0, 3), (3, 2), (5, 2)], dtype=[("first_query", "<i8"), ("n_queries", "<i4")])
    controls = {"canonical_match_key": np.array([b"a", b"a", b"b", b"c", b"a", b"c", b"b"], dtype="S64"),
                "component_key": np.array([b"1", b"2", b"1", b"1", b"3", b"2", b"2"], dtype="S64"),
                "control_start_query": np.array([0, 1, 3, 5, 0, 6, -1]),
                "control_start_ms": np.array([100, 200, 150, 400, 100, 500, -1]),
                "control_end_ms": np.array([250, 260, 250, 600, 250, 450, -1])}
    return controls, matches, queries, match_queries


def test_plan_has_distinct_sorted_ends_with_shards_and_engagement_mapping():
    plan = cs.plan_ends(*_tables())
    assert plan["match_index"].tolist() == [0, 0, 1, 2, 2]
    assert plan["end_ms"].tolist() == [250, 260, 250, 450, 600]
    assert plan["shard"].tolist() == [0, 0, 0, 1, 1]
    assert plan["local_match"].tolist() == [0, 0, 1, 0, 0]
    assert plan["engagement_end_index"].tolist() == [0, 1, 2, 4, 0, 3, -1]   # shared end, unmatched engagement
    assert plan["n_engagements"].tolist() == [2, 1, 1, 1, 1]
    assert plan["last_stored_query_ms"].tolist() == [300, 300, 250, 500, 500]
    assert plan["after_last_stored_query"].tolist() == [False, False, False, False, True]
    assert plan["end_is_stored_query"].tolist() == [False, False, True, False, False]


def test_plan_rejects_a_start_query_of_another_match_and_a_non_shard_order():
    controls, matches, queries, match_queries = _tables()
    wrong = dict(controls, control_start_query=np.array([3, 1, 3, 5, 0, 6, -1]))
    with pytest.raises(ValueError, match="another match"):
        cs.plan_ends(wrong, matches, queries, match_queries)
    unordered = matches.copy()
    unordered["shard"] = [1, 1, 0]
    with pytest.raises(ValueError, match="shard order"):
        cs.plan_ends(controls, unordered, queries, match_queries)


def test_part_counts_as_complete_only_with_its_receipt_plan_hash_and_ends(tmp_path):
    plan_sha = "a" * 64
    path = cs.part_path(tmp_path, 7)
    ends = np.array([3, 4], dtype=np.int64)
    sha = cs.write_npz_atomic(path, {"end_index": ends, "time_ms": np.array([10, 20])})
    assert not cs.part_complete(path, ends, plan_sha)                                  # no receipt yet (interrupted)
    cs.write_json_atomic(path.with_suffix(".json"), {"version": cs.VERSION, "plan_sha256": plan_sha, "sha256": sha})
    assert cs.part_complete(path, ends, plan_sha)
    assert not cs.part_complete(path, np.array([3, 4, 5]), plan_sha)                   # a partial (--limit-matches) part
    assert not cs.part_complete(path, ends, "b" * 64)                                  # another plan
    path.write_bytes(path.read_bytes() + b"\0")
    assert not cs.part_complete(path, ends, plan_sha)                                  # changed bytes


def test_output_under_outputs_or_the_training_run_is_refused(tmp_path):
    for forbidden in (cs.ROOT / "outputs" / "x", cs.ROOT / "studies", Path(os.environ["LOL_TRAINING_ROOT"]) / 'run_3' / 'labels'):
        with pytest.raises(ValueError, match="refusing"):
            cs.checked_out(forbidden)
    assert cs.checked_out(tmp_path / "new") == (tmp_path / "new").resolve()


def test_comparison_is_exact_about_types_dtypes_key_order_and_float_bits():
    def differs(left, right):
        return not cs.Comparison().check(left, right)

    assert differs(np.zeros(2, np.float32), np.zeros(2, np.float64))
    assert differs(np.array([0.0]), np.array([-0.0]))
    assert differs({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert differs({1: 1}, {"1": 1})
    assert differs([True], [1])
    assert differs(0.1 + 0.2, 0.3)
    assert not differs({"x": [np.array([np.nan]), 1.5, "s"]}, {"x": [np.array([np.nan]), 1.5, "s"]})
    lenient = cs.Comparison()
    assert lenient.check(np.zeros(2).view(np.memmap), np.zeros(2), subclass_ok=True)
    assert differs(np.zeros(2).view(np.memmap), np.zeros(2))


# ---------------------------------------------------------------- synthetic bank shard


def _late_free_id():
    """A match ID whose stored queries all lie before LATE_MS (the kill component queries end by 250 s)."""

    for number in range(1000):
        match_id = f"fixture-{number}"
        stored = [row["actual_query_ms"] for row in sample_match_queries(match_id, END_MS, FRAMES) if row["query_support_valid"]]
        if max(stored) < LATE_MS - 10000:
            return match_id
    raise AssertionError("no fixture id")


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    """One produced match: an early purchase and kill, then a level-up and a purchase after the last stored query."""

    events = [{"type": "ITEM_PURCHASED", "timestamp": 125000, "participantId": 1, "itemId": 1001},
              {"type": "CHAMPION_KILL", "timestamp": 130000, "killerId": 1, "victimId": 6, "position": {"x": 100, "y": 200}},
              {"type": "LEVEL_UP", "timestamp": LATE_MS, "participantId": 3, "level": 2},
              {"type": "ITEM_PURCHASED", "timestamp": LATE_MS, "participantId": 2, "itemId": 1055},
              {"type": "GAME_END", "timestamp": END_MS, "winningTeam": 100}]

    def sources(_entry):
        arrays = {"minute_ts": FRAMES, "node_minute": np.zeros((len(FRAMES), 10, 76), dtype=np.float32),
                  "xy_raw_minute": np.ones((len(FRAMES), 10, 2), dtype=np.float32),
                  "gold_team_minute": np.zeros((len(FRAMES), 2), dtype=np.float32)}
        return arrays, _meta(), [dict(event) for event in events], {k: {"stable": True} for k in ("npz", "meta", "events")}

    monkeypatch.setattr(producer, "read_cached_triplet", sources)
    entry = {"match_id": _late_free_id(), "canonical_match_key": "f" * 64, "patch": "15.15", "role": "Q_DEV",
             "original_index": 0, "source_hashes": {}, "audited_numbers": {
                 "n_frames": len(FRAMES), "n_events": len(events), "n_kills": 1, "game_end_ms": END_MS, "game_end_winning_team": 100}}
    bank, ledger = producer.produce_match(entry)
    builder = ShardBuilder()
    builder.append(bank)
    arrays, layout = builder.finish()
    write_shard(tmp_path / "shard", arrays, compressed=True)
    loaded = open_shard(tmp_path / "shard")
    try:
        source = cs.MatchSource(0, 0, 0, match_view(loaded, layout, 0), layout, ledger)
        stored = [int(value) for value in source.view["query_ms"]]
        assert stored[-1] < LATE_MS
        yield source, stored
    finally:
        close_shard(loaded)


def test_end_after_the_last_stored_query_includes_the_events_in_between(synthetic):
    source, stored = synthetic
    last = stored[-1]
    at_last, at_end = cs.match_states(source, [last, END_MS])
    assert at_end.nonitem != at_last.nonitem and at_end.item != at_last.item
    items = at_end.features["items"]
    assert items["item_ids"][items["offsets"][1]:items["offsets"][2]].tolist() == [1055]     # participant 2
    assert 1055 not in at_last.features["items"]["item_ids"].tolist()
    # iter_match_views' cutoff (records up to the last stored query) would give the stale history
    accumulator = HistoryAccumulator()
    for record in SemanticHistory(cs.projection_bank(source)).iter_prefix(last):
        accumulator.add(record)
    stale = cs.record_text(accumulator.snapshot_at(END_MS))
    assert stale == at_last.nonitem != at_end.nonitem


def test_stored_time_rows_reproduce_the_bank_arrays_and_ignore_extra_times(synthetic):
    source, stored = synthetic
    built = cs.build_match(source, stored)
    for name, value in built.query.items():
        comparison = cs.Comparison()
        assert comparison.check(value, np.asarray(source.view[name]), name), comparison.mismatches
    rows = cs.semantic_rows(built.states)
    extra = sorted(set(stored) | {stored[-1] + 1, LATE_MS - 1, LATE_MS, END_MS})
    both = cs.semantic_rows(cs.match_states(source, extra))
    comparison = cs.Comparison()
    for index, cutoff in enumerate(stored):
        assert comparison.check(both[extra.index(cutoff)], rows[index], f"t{cutoff}"), comparison.mismatches
    alone = cs.rows_at_times(source, [END_MS])
    assert comparison.check(both[-1], alone[0], "end"), comparison.mismatches


def test_sidecar_round_trip_rebuilds_identical_rows(synthetic, tmp_path):
    source, stored = synthetic
    times = [*stored, LATE_MS, END_MS]
    states = cs.match_states(source, times)
    rows = cs.semantic_rows(states)
    merged = cs.concat_parts([cs.encode_states(states[:2]), cs.encode_states(states[2:])])
    n = len(times)
    merged.update(end_index=np.arange(n, dtype=np.int64), match_index=np.zeros(n, dtype=np.int64), end_ms=np.asarray(times, dtype=np.int64),
                  after_last_stored_query=np.asarray(times) > stored[-1], outside_training_frame_age_support=np.zeros(n, dtype=bool))
    path = tmp_path / "rows.npz"
    cs.write_npz_atomic(path, merged)
    rebuilt = list(cs.iter_rows(path))
    assert [index for index, _row, _meta in rebuilt] == list(range(n))
    comparison = cs.Comparison()
    assert comparison.check([row for _index, row, _meta in rebuilt], rows), comparison.mismatches
    with np.load(path, allow_pickle=False) as stored_arrays:
        decoded = cs.decode_states({name: stored_arrays[name] for name in stored_arrays.files})
    assert [(state.time_ms, state.nonitem, state.item) for state in decoded] == [(s.time_ms, s.nonitem, s.item) for s in states]
    assert comparison.check([state.features for state in decoded], [state.features for state in states], subclass_ok=True), comparison.mismatches


def test_finish_consolidates_complete_parts_once_and_flags_frame_age(synthetic, tmp_path):
    source, stored = synthetic
    times = [*stored, LATE_MS, END_MS]
    build = cs.build_match(source, times)
    n, plan_sha, max_age = len(times), "c" * 64, 45.0      # LATE_MS has frame age 50 s, END_MS 0 s
    plan = {"end_index": np.arange(n, dtype=np.int64), "match_index": np.zeros(n, dtype=np.int64),
            "end_ms": np.asarray(times, dtype=np.int64), "shard": np.asarray([0, 0] + [1] * (n - 2), dtype=np.int64),
            "after_last_stored_query": np.asarray(times) > stored[-1], "engagement_end_index": np.arange(n, dtype=np.int64)}
    out = tmp_path / "out"

    def finish():
        return cs.finish(out, plan, plan_sha, {}, max_age, {}, None, [], [], 0.0)

    for shard in (0, 1):
        rows = np.flatnonzero(plan["shard"] == shard)
        payload = cs.part_payload([build.states[i] for i in rows], {name: plan[name][rows] for name in cs.PART_PLAN_FIELDS},
                                  {name: build.query[name][rows] for name in cs.AUDIT_FIELDS}, max_age)
        cs.write_part(out, shard, payload, plan_sha, {})
        if shard == 0:
            partial = finish()
            assert not partial["complete"] and partial["parts_complete"] == 1
            assert not (out / f"{cs.ROWS_NAME}.npz").exists()
    manifest = finish()
    rows_path = out / f"{cs.ROWS_NAME}.npz"
    assert manifest["complete"] and manifest["rows"] == n
    assert finish()["rows_sha256"] == manifest["rows_sha256"] == cs.sha256_file(rows_path)   # not rewritten
    comparison = cs.Comparison()
    assert comparison.check([row for _index, row, _meta in cs.iter_rows(rows_path)], cs.semantic_rows(build.states)), comparison.mismatches
    with np.load(rows_path, allow_pickle=False) as table:
        ages = table["feature_frame_age_seconds"]
        assert np.array_equal(table["outside_training_frame_age_support"], ages > max_age) and ages.max() > max_age > ages.min()
        assert manifest["outside_training_frame_age_support"] == int((ages > max_age).sum())


# ---------------------------------------------------------------- real data (skipped without the banks)


REAL_INPUTS = (cs.PROJECTION / "RUN_MANIFEST.json", cs.STATIC_PRESERVED_BANK / "RUN_MANIFEST.json",
               cs.ROOT / "outputs/shared_history_eight_cpu_20261003/full_1/ACCEPTANCE.json",
               cs.LABELS / f"{cs.MATCHING}.npz", cs.LABELS / "ENGAGEMENTS.npz")
real = pytest.mark.skipif(not all(path.is_file() for path in REAL_INPUTS), reason="label banks are not available")


@real
def test_equality_gate_on_real_matches(tmp_path):
    projection = cs.Projection()
    plan, _inputs = cs.load_plan(cs.LABELS, projection)
    bins = np.digitize(plan["end_ms"] / 60000.0, (15, 25))
    after = plan["after_last_stored_query"]
    chosen = {int(plan["match_index"][after][0]), int(plan["match_index"][(bins == 0) & ~after][0]),
              int(plan["match_index"][(bins == 2) & ~after][0])}
    result = cs.equality_gate(cs.LABELS, tmp_path / "gate", matches=sorted(chosen))
    assert result["passed"], [report["checks"] for report in result["reports"] if not report["passed"]]
    assert result["after_last_stored_query_checked"] >= 1
    assert all(total["mismatches"] == 0 and total["max_abs_diff"] == 0.0 for total in result["totals"].values())


@real
def test_run_resumes_and_its_part_rebuilds_the_direct_rows(tmp_path):
    out = tmp_path / "rows"
    first = cs.run(cs.LABELS, out, limit_matches=2)
    assert first["this_run"]["parts_written"] >= 1 and not first["complete"]
    second = cs.run(cs.LABELS, out, limit_matches=2)
    assert second["this_run"]["parts_written"] == 0 and second["this_run"]["parts_skipped"] == first["this_run"]["parts_written"]
    projection = cs.Projection()
    part = sorted((out / "parts").glob("part_*.npz"))[0]
    rebuilt = list(cs.iter_rows(part))
    match = rebuilt[0][2]["match_index"]
    times = [meta["end_ms"] for _index, _row, meta in rebuilt if meta["match_index"] == match]
    shard = int(projection.matches["shard"][match])
    bank = cs.BankShard(shard, projection.binding(shard), projection.source_root)
    direct = cs.rows_at_times(bank.source(int(projection.matches["local_match"][match]), match, projection.key(match)), times)
    comparison = cs.Comparison()
    assert comparison.check([row for _index, row, meta in rebuilt if meta["match_index"] == match], direct), comparison.mismatches
