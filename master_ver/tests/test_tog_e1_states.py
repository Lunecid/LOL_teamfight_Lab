"""T1.1 equality gate (studies/tog_experiments_20261012/e1_gate.py): the parts that need no bank or run data."""
import copy
import hashlib
import json

import numpy as np
import pytest

from studies.observable_state_20261002.kernel import STATIC_NAMES
from studies.q_production_20261005 import features as qf
from studies.tog_experiments_20261012 import e1_gate
from studies.v_model_fit_input_20261003.common import ITEM_GROUPS


def _queries(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    queries = np.sort(rng.choice(5_000_000, size=n, replace=False)).astype(np.int64)
    return queries, rng.choice(np.array([0, 3, 4, 5], dtype=np.int8), size=n)


def test_sample_takes_500_per_role_with_the_smallest_sha256():
    queries, roles = _queries()
    sample = e1_gate.sample_queries(queries, roles)
    assert sorted(sample) == [0, 3, 4, 5]
    for role, chosen in sample.items():
        def digest(query):
            return hashlib.sha256(f"{e1_gate.SALT}:{query}".encode()).hexdigest()
        expected = np.sort(np.asarray(sorted(queries[roles == role].tolist(), key=digest)[:500], dtype=np.int64))
        assert chosen.dtype == np.int64 and len(chosen) == 500
        np.testing.assert_array_equal(chosen, expected)


def test_sample_is_deterministic_and_ignores_row_order():
    queries, roles = _queries()
    order = np.random.default_rng(1).permutation(len(queries))
    first = e1_gate.sample_queries(queries, roles)
    again = e1_gate.sample_queries(queries.copy(), roles.copy())
    shuffled = e1_gate.sample_queries(queries[order], roles[order])
    for role in first:
        np.testing.assert_array_equal(first[role], again[role])
        np.testing.assert_array_equal(first[role], shuffled[role])


def test_sample_refuses_a_short_role_and_a_query_in_two_roles():
    queries, roles = _queries(n=1000)                      # about 250 queries per role
    with pytest.raises(ValueError, match="fewer than 500"):
        e1_gate.sample_queries(queries, roles)
    with pytest.raises(ValueError, match="more than one role"):
        e1_gate.sample_queries(np.array([7, 7, 9]), np.array([0, 3, 3]), per_role=1)


def test_mismatch_report_names_the_differing_fields():
    expected = {"numeric:node": np.zeros((4, 10, 3)),
                "team_ids": np.tile(np.array([100, 200] * 5, dtype=np.int16), (4, 1)),
                "bag:team:nonitem": np.array(["[[],[]]"] * 4, dtype=object)}
    actual = {name: value.copy() for name, value in expected.items()}
    actual["numeric:node"][1, 2, 0] = -0.0                  # equal as a number, not bitwise
    actual["bag:team:nonitem"][3] = '[[["tok",1]],[]]'
    report = e1_gate.mismatch_report(e1_gate.field_mismatches(actual, expected), np.array([0, 3, 3, 5]),
                                     keys=np.array([11, 12, 13, 14]))
    assert report["mismatched_fields"] == ["numeric:node", "bag:team:nonitem"]
    assert report["mismatched_rows"] == 2
    assert report["roles"]["Q_DEV"] == {"rows": 2, "mismatched_rows": 1, "fields": {"numeric:node": 1}}
    assert report["roles"]["TEST"]["fields"] == {"bag:team:nonitem": 1}
    assert report["roles"]["TRAIN"]["mismatched_rows"] == 0 and report["roles"]["Q_CAL"]["rows"] == 0
    assert report["examples"] == {"numeric:node": [12], "bag:team:nonitem": [14]}


def test_a_missing_field_or_another_dtype_differs_on_every_row():
    expected = {"xy": np.zeros((3, 10, 2)), "frame_index": np.arange(3, dtype=np.int32)}
    actual = {"xy": np.zeros((3, 10, 2), dtype=np.float32), "extra": np.zeros(3)}
    report = e1_gate.mismatch_report(e1_gate.field_mismatches(actual, expected), np.array([0, 0, 4]))
    assert report["mismatched_fields"] == ["xy", "frame_index", "extra"]
    assert report["roles"]["TRAIN"]["fields"] == {"xy": 2, "frame_index": 2, "extra": 2}


def test_the_same_nan_is_equal():
    values = {"numeric:global": np.array([[np.nan, 1.0]])}
    report = e1_gate.mismatch_report(e1_gate.field_mismatches(values, {"numeric:global": values["numeric:global"].copy()}),
                                     np.array([3]))
    assert report["mismatched_rows"] == 0 and report["mismatched_fields"] == []


def _token(event, value):
    return json.dumps([event, "level", "value", ["int", value]], separators=(",", ":"))


def _row(tokens, item_count=1):
    empty_items = {group: [{} for _ in range(10)] for group in ITEM_GROUPS}
    return {"numeric": {"node": np.full((10, 102), 0.5), "team": np.ones((2, 8)), "global": np.arange(75, dtype=np.float64)},
            "categories": {"node": {name: np.arange(10) for name in STATIC_NAMES}, "team": {f"ban{k}": np.array([k, -1]) for k in range(5)},
                           "global": {}},
            "bags": {"node": {"current_item": [{1001: item_count}] + [{} for _ in range(9)], **empty_items,
                              "nonitem": [{token: 2 for token in tokens}] + [{} for _ in range(9)]},
                     "team": {"nonitem": [{tokens[-1]: 1}, {}]},
                     "global": {**{group: [{}] for group in ITEM_GROUPS}, "nonitem": [{}]}},
            "team_ids": np.array([100, 200] * 5), "focal_is_blue": True, "current_item_supported": np.ones(10, dtype=bool)}


def _store(path, rows):
    writer = qf.RawStoreWriter(path, np.arange(len(rows), dtype=np.int64))
    for index, row in enumerate(rows):
        writer.add(index, row)
    writer.close()
    return qf.RawStore(path)


def test_stores_with_other_token_dictionaries_compare_by_token_text(tmp_path):
    a, b = _token("LEVEL_UP", 2), _token("LEVEL_UP", 3)
    alone = _store(tmp_path / "alone", [_row([b, a])])
    merged = _store(tmp_path / "merged", [_row([a]), _row([b, a])])           # other first-seen order: other codes
    assert alone.tokens != merged.tokens
    same = e1_gate.field_mismatches(e1_gate.store_fields(alone, np.array([0])), e1_gate.store_fields(merged, np.array([1])))
    assert not any(flags.any() for flags in same.values())
    changed = _store(tmp_path / "changed", [_row([b, a], item_count=2)])
    differs = e1_gate.field_mismatches(e1_gate.store_fields(changed, np.array([0])), e1_gate.store_fields(merged, np.array([1])))
    assert [name for name, flags in differs.items() if flags.any()] == ["bag:node:current_item"]


def test_semantic_rows_compare_exactly_and_name_the_fields():
    row = _row([_token("LEVEL_UP", 2)])
    fields, leaves, _ = e1_gate.compare_rows(row, copy.deepcopy(row))
    assert fields == [] and leaves > 0
    other = copy.deepcopy(row)
    other["numeric"]["global"][0] = -0.0                     # bitwise, not numeric, equality
    other["bags"]["node"]["nonitem"][0] = {_token("LEVEL_UP", 2): 3}
    other["team_ids"] = other["team_ids"].astype(np.int32)    # same values, another dtype
    fields, _, _ = e1_gate.compare_rows(other, row)
    assert fields == ["bags/node/nonitem", "numeric/global", "team_ids"]


def test_new_times_split_the_widest_stored_gap_and_follow_the_last_stored_query_to_the_limit():
    assert e1_gate.new_times([0, 10, 100], 104) == [55, 102, 104]
    assert e1_gate.new_times([100], 102) == [101, 102]                           # no gap between stored queries
    assert e1_gate.new_times([0, 10, 100], 101) == [] and e1_gate.new_times([0, 10, 100], 90) == []   # no room after
    frames = np.array([0, 60_000, 120_000])
    assert e1_gate.time_limit(frames, np.array([150_000])) == 120_000            # the last frame
    assert e1_gate.time_limit(frames, np.array([110_000])) == 109_999            # before the game end
    assert e1_gate.time_limit(frames, np.array([], dtype=np.int64)) == 120_000    # game end not recorded


def test_xy_follows_the_spatial_frame_rules():
    clock = np.array([0, 60_000, 120_000], dtype=np.int64)
    frames = np.arange(3 * 10 * 2, dtype=np.float32).reshape(3, 10, 2) + np.float32(0.25)
    view = {"minute_ts": clock, "xy": frames}
    xy, support = e1_gate.xy_at_times(view, [59_999, 60_000, 130_000], [0, 1, 2])
    assert xy.dtype == np.float64 and support.tolist() == [True, True, False]      # 130 s lies after the last frame
    np.testing.assert_array_equal(xy[:2], frames[[0, 1]].astype(np.float64))
    assert np.isnan(xy[2]).all()
    with pytest.raises(ValueError, match="future frame"):
        e1_gate.xy_at_times(view, [59_999], [1])
    with pytest.raises(ValueError, match="latest frame"):
        e1_gate.xy_at_times(view, [60_000], [0])


def test_frozen_checks_split_the_four_checks():
    totals = {name: {"leaves_compared": 5, "mismatches": 0} for name in ("query_arrays", "features", "history_text", "rows", "roundtrip")}
    totals["prefix"] = {"leaves_compared": 5, "mismatches": 1}
    result = {"passed": False, "totals": totals,
              "reports": [{"after_last_stored_query": [{"consistent": False}, {"consistent": True}]}]}
    checks = e1_gate.frozen_checks(result)
    assert list(checks) == ["stored_rows_equal_query_surface", "unchanged_with_new_times", "sidecar_roundtrip",
                            "after_last_stored_query_events"]
    assert checks["unchanged_with_new_times"]["mismatches"] == 0 and checks["unchanged_with_new_times"]["passed"]
    assert checks["after_last_stored_query_events"] == {"ends_checked": 2, "mismatches": 1, "passed": False}
    result["reports"] = [{"after_last_stored_query": []}]
    result["totals"]["prefix"]["mismatches"] = 0
    assert not e1_gate.frozen_checks(result)["after_last_stored_query_events"]["passed"]       # nothing checked


# ----------------------------------------------------------------------------- T1.3 E1 states and V (e1_states.py)
import io  # noqa: E402
import zipfile  # noqa: E402

from studies.tog_experiments_20261012 import e1_states  # noqa: E402
from studies.tog_experiments_20261012.e1_states import context_for  # noqa: E402


def test_context_by_role():
    assert context_for(0, 3) == "oof3"
    assert all(context_for(r, 0) == "final" for r in (3, 4, 5))


def test_a_role_or_fold_outside_the_rule_raises():
    for role in (1, 2, 6, -1):                     # V_DEV, V_CAL and unknown codes are not q roles
        with pytest.raises(ValueError, match="not a q role"):
            context_for(role, 0)
    for fold in (-1, 5):
        with pytest.raises(ValueError, match="OOF fold"):
            context_for(0, fold)
    assert (context_for(5, -1), context_for(0, 0), context_for(0, 4)) == ("final", "oof0", "oof4")
    codes = e1_states.context_codes(np.array([0, 0, 3, 4, 5], dtype=np.int8), np.array([2, 4, -1, -1, -1], dtype=np.int8))
    assert codes.dtype == np.int8 and codes.tolist() == [3, 5, 0, 0, 0]
    with pytest.raises(ValueError, match="not a q role"):
        e1_states.context_codes(np.array([0, 1]), np.array([0, 0]))
    with pytest.raises(ValueError, match="OOF fold"):
        e1_states.context_codes(np.array([3, 0]), np.array([-1, -1]))


def test_one_match_has_one_context_and_the_context_never_saw_the_match():
    e1_states.check_match_contexts(np.array([4, 4, 9]), np.array([2, 2, 0]))
    with pytest.raises(e1_states.Blocked, match="one V context"):
        e1_states.check_match_contexts(np.array([4, 4, 9]), np.array([2, 3, 0]))
    membership = np.zeros((6, 4), dtype=np.uint8)
    membership[2, 0] = 32                          # oof1: match 0 is its held-out target fold
    membership[2, 1] = 1                           # oof1 was fitted on match 1 (and match 1 is not its target)
    membership[0, 3] = 8                           # final saw match 3 in V_DEV
    role = np.array([0, 0, 3, 5])
    codes = e1_states.context_codes(role, np.array([1, 1, -1, -1]))
    assert e1_states.oof_leaks(codes, role, np.array([0, 1, 2, 3]), membership).tolist() == [False, True, False, True]


def test_windows_must_follow_the_row_table():
    keys = np.array([b"a:1", b"a:2", b"b:1"])
    windows = {"engagement_key": keys, "match_index": np.array([0, 0, 1], dtype=np.int32), "role": np.array([0, 0, 5], dtype=np.int8)}
    rows = {"engagement_key": keys.copy(), "match_index": np.array([0, 0, 1]), "role": np.array([0, 0, 5], dtype=np.int8)}
    e1_states.check_alignment(windows, rows)
    with pytest.raises(e1_states.Blocked, match="engagement_key"):
        e1_states.check_alignment({**windows, "engagement_key": keys[[1, 0, 2]]}, rows)
    with pytest.raises(e1_states.Blocked, match="role"):
        e1_states.check_alignment({**windows, "role": np.array([0, 3, 5], dtype=np.int8)}, rows)


def test_states_are_unique_per_match_and_time_in_match_time_order():
    match = np.array([7, 7, 3, 7, 7])
    s = np.array([500_000, 200_000, 150_000, 200_000, 260_000])          # windows 1 and 3 share a start (reuse)
    end = np.array([530_000, 260_000, 190_000, 260_000, 300_000])        # window 4 starts where windows 1 and 3 end
    plan = e1_states.state_plan(match, s, end)
    assert plan["state_match"].tolist() == [3, 3, 7, 7, 7, 7, 7]
    assert plan["state_ms"].tolist() == [150_000, 190_000, 200_000, 260_000, 300_000, 500_000, 530_000]
    assert plan["start_state"].tolist() == [5, 2, 0, 2, 3] and plan["end_state"].tolist() == [6, 3, 1, 3, 4]
    assert e1_states.end_is_a_start(plan["start_state"], plan["end_state"]).tolist() == [False, True, False, True, False]
    with pytest.raises(ValueError, match="ends at or before"):
        e1_states.state_plan(np.array([1]), np.array([5_000]), np.array([5_000]))


def test_scoring_batches_follow_the_context_then_the_stored_order():
    ctx = np.array([2, 0, 2, 2, 0, 2, 2], dtype=np.int8)
    batches = e1_states.canonical_batches(ctx, batch_rows=2)
    assert [(code, rows.tolist()) for code, rows in batches] == [(0, [1, 4]), (2, [0, 2]), (2, [3, 5]), (2, [6])]
    assert [(code, rows.tolist()) for code, rows in e1_states.canonical_batches(ctx)] == [(0, [1, 4]), (2, [0, 2, 3, 5, 6])]


def _windows_and_scores():
    windows = {"window_id": np.array([4, 9, 11, 20], dtype=np.int64),
               "engagement_key": np.array([b"m1:a", b"m1:b", b"m2:a", b"m3:a"]),
               "match_index": np.array([1, 1, 2, 3], dtype=np.int64), "role": np.array([5, 5, 0, 3], dtype=np.int8),
               "fold": np.array([-1, -1, 2, -1], dtype=np.int8), "ctx": np.array([0, 0, 3, 0], dtype=np.int8),
               "s_ms": np.array([200_000, 230_000, 300_000, 400_000], dtype=np.int64),
               "end_ms": np.array([230_000, 260_000, 330_000, 450_000], dtype=np.int64),
               "start_state": np.array([0, 1, 3, 5], dtype=np.int64), "end_state": np.array([1, 2, 4, 6], dtype=np.int64)}
    scores = {"logit": np.arange(7, dtype=np.float64) - 3.0, "raw": np.linspace(0.1, 0.7, 7), "v": np.linspace(0.2, 0.8, 7)}
    return windows, scores


def test_a_test_end_value_reaches_only_the_sealed_file():
    windows, scores = _windows_and_scores()
    public, sealed = e1_states.split_predictions(windows, scores)
    test = windows["role"] == 5
    for kind, source in (("logit", "logit"), ("raw", "raw"), ("p", "v")):
        np.testing.assert_array_equal(public[f"{kind}_start"], scores[source][windows["start_state"]])   # V(s): every role
        assert np.isnan(public[f"{kind}_end"][test]).all()
        np.testing.assert_array_equal(public[f"{kind}_end"][~test], scores[source][windows["end_state"]][~test])
        np.testing.assert_array_equal(sealed[f"{kind}_end"], scores[source][windows["end_state"]][test])
    for name in ("window_id", "engagement_key", "role", "ctx", "s_ms", "end_ms", "start_state", "end_state"):
        np.testing.assert_array_equal(public[name], windows[name])
    assert sealed["window_id"].tolist() == [4, 9] and sealed["engagement_key"].tolist() == [b"m1:a", b"m1:b"]
    assert sealed["end_equals_a_start"].tolist() == [True, False]          # window 4 ends where window 9 starts
    assert "end_equals_a_start" not in public and set(sealed) == set(e1_states.SEALED_COLUMNS)


def test_npz_bytes_are_fixed_and_files_are_written_once(tmp_path):
    arrays = {"a": np.arange(5, dtype=np.int64), "k": np.array([b"x:1", b"y:22"]), "v": np.array([0.5, np.nan])}
    for compress in (False, True):
        first = e1_states.npz_bytes(arrays, compress=compress)
        assert first == e1_states.npz_bytes(dict(arrays), compress=compress)
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        path = tmp_path / f"t_{compress}.npz"
        assert e1_states.write_bytes_once(path, first) == hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as stored:
            assert stored.files == ["a", "k", "v"]
            for name in arrays:
                np.testing.assert_array_equal(stored[name], arrays[name])
        with pytest.raises(FileExistsError):
            e1_states.write_bytes_once(path, first)
    with pytest.raises(ValueError, match="object"):
        e1_states.npz_bytes({"o": np.array([{}], dtype=object)}, compress=False)


def test_an_existing_output_is_checked_by_hash_without_opening_it(tmp_path, monkeypatch):
    data = e1_states.npz_bytes({"p_end": np.array([0.25])}, compress=False)
    path = tmp_path / "PRED_PLACEBO_h90_TEST_SEALED.npz"
    assert e1_states.write_or_keep(path, data) == ("written", hashlib.sha256(data).hexdigest())

    def refuse(*_args, **_kwargs):
        raise AssertionError("the sealed file was opened")

    monkeypatch.setattr(np, "load", refuse)
    assert e1_states.write_or_keep(path, data) == ("kept", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="never overwritten"):
        e1_states.write_or_keep(path, e1_states.npz_bytes({"p_end": np.array([0.75])}, compress=False))
    assert path.read_bytes() == data


def test_a_part_made_by_other_code_or_inputs_is_never_reused(tmp_path):
    identity = {"version": e1_states.PART_VERSION, "shard": 3, "code_sha256": {"e1_states.py": "aa"}, "plan_sha256": "p1"}
    path = e1_states.part_path(tmp_path, 3)
    assert path.name == "part_0003.npz" and e1_states.part_status(path, identity) == "missing"
    e1_states.write_bytes_once(path, e1_states.npz_bytes({"state_id": np.arange(2), "identity_json": e1_states.identity_array(identity)},
                                                         compress=True))
    assert e1_states.part_status(path, identity) == "done"
    for change in ({"code_sha256": {"e1_states.py": "bb"}}, {"version": "tog_e1_states_part_v0"}, {"plan_sha256": "p2"}):
        with pytest.raises(ValueError, match="never reused"):
            e1_states.part_status(path, {**identity, **change})
    broken = e1_states.part_path(tmp_path, 4)
    broken.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="never reused"):
        e1_states.part_status(broken, {**identity, "shard": 4})
    tasks = [{"shard": 3, "path": str(path), "identity": identity},
             {"shard": 5, "path": str(e1_states.part_path(tmp_path, 5)), "identity": {**identity, "shard": 5}}]
    todo, done = e1_states.split_done(tasks)
    assert [task["shard"] for task in todo] == [5] and list(done) == [3]
    assert done[3] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_reproduction_sample_and_gate():
    role = np.repeat(np.array([0, 3, 4, 5], dtype=np.int8), [5000, 1000, 1000, 1000])
    fold = np.where(role == 0, np.arange(len(role)) % 5, -1).astype(np.int8)
    sample = e1_states.repro_sample(role, fold)
    assert list(sample) == [f"TRAIN_fold{f}" for f in range(5)] + ["Q_DEV", "Q_CAL"]
    rows = np.concatenate(list(sample.values()))
    assert len(rows) == len(np.unique(rows)) == 1400 and not np.any(role[rows] == 5)
    assert all(np.all((role[sample[f"TRAIN_fold{f}"]] == 0) & (fold[sample[f"TRAIN_fold{f}"]] == f)) for f in range(5))
    assert np.all(role[sample["Q_DEV"]] == 3) and np.all(role[sample["Q_CAL"]] == 4)
    again = e1_states.repro_sample(role.copy(), fold.copy())
    assert all(np.array_equal(sample[name], again[name]) for name in sample) and e1_states.REPRO_SEED == 20261013
    reference = np.linspace(0.1, 0.9, 6)
    v = reference + np.array([0.0, 2e-7, -3e-7, 1.5e-6, 0.0, 9e-6])
    summary = e1_states.repro_summary(v, reference)
    assert summary["passed"] and summary["rows"] == 6 and summary["exact_equal"] == 2 and summary["over_1e-6"] == 2
    assert summary["over_tolerance"] == 0 and summary["tolerance"] == 1e-5 and summary["float32_equal"] >= 2
    assert summary["max_abs_diff"] == pytest.approx(9e-6)
    failed = e1_states.repro_summary(v + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2e-6]), reference)        # 1.1e-5 > 1e-5
    assert not failed["passed"] and failed["over_tolerance"] == 1
    assert not e1_states.repro_summary(np.array([np.nan]), np.array([0.5]))["passed"]
    strength = e1_states.strength_summary(np.array([0.5, 0.52, 0.5000005]), np.array([0.5, 0.5, 0.5]))
    assert strength["within_tolerance"] == 2 and strength["within_1e-6"] == 2 and strength["min_abs_diff"] == 0.0


def test_public_counts_leave_test_out():
    role = np.array([0, 0, 3, 4, 5, 5], dtype=np.int8)
    counts = e1_states.public_counts(role, np.array([1, 2, 3, 4, 5, 6]))
    assert list(counts) == ["TRAIN", "Q_DEV", "Q_CAL"] and "TEST" not in json.dumps(counts)
    assert counts["TRAIN"] == {"rows": 2, "matches": 2} and counts["Q_CAL"] == {"rows": 1, "matches": 1}
    windows, _scores = _windows_and_scores()
    plan = {**windows, "state_role": np.array([5, 5, 5, 0, 0, 3, 3], dtype=np.int8)}
    record = e1_states.role_record(plan, np.ones(4, dtype=bool), np.ones(7, dtype=bool))
    assert list(record) == ["TRAIN", "Q_DEV", "Q_CAL"] and "TEST" not in json.dumps(record)
    assert record["TRAIN"] == {"matched_windows": 1, "matches": 1, "states": 2, "windows_by_context": {"oof2": 1},
                               "windows_ending_at_a_start_state": 0}
    assert record["Q_CAL"]["matched_windows"] == 0 and record["Q_DEV"]["states"] == 2


def test_runs_write_only_where_allowed(tmp_path):
    main = e1_states.TOG_E1 / e1_states.STATES_DIR
    e1_states._check_destination(main, smoke=False)                                    # the full run: tog_e1_1 only
    e1_states._check_destination(tmp_path / "STATES_h90", smoke=True, only=["train_0000", "validation_0000"])
    with pytest.raises(ValueError, match="full run"):
        e1_states._check_destination(tmp_path / "STATES_h90", smoke=False)
    with pytest.raises(ValueError, match="full run"):
        e1_states._check_destination(main, smoke=False, only=["train_0000"])
    with pytest.raises(ValueError, match="never writes into tog_e1_1"):
        e1_states._check_destination(main, smoke=True, only=["train_0000"])
    with pytest.raises(ValueError, match="TEST shards"):
        e1_states._check_destination(tmp_path / "STATES_h90", smoke=True, only=["train_0000", "test_0003"])
    assert e1_states.HIDE_CUDA == "-1"          # never "": on Windows an empty value deletes the variable for workers
