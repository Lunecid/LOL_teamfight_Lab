"""Tests for the E2 run (PRESPEC_E2_KO.md, PLAN_E1_E5_KO.md Task 4).

Pure checks first (team map, frame checks, decision, AUC on shared draws, within-stratum weights).  The pipeline tests
build a small synthetic bank, row table, predictions and sealed labels under tmp_path and run every stage there,
the TEST step included; the real TEST files are never touched.
"""
import dataclasses
import json

import numpy as np
import pytest

from studies.label_production_20261005.engagements import COMPONENT_COLUMNS
from studies.q_production_20261005 import qeval
from studies.tog_experiments_20261012 import e2_run as E
from studies.tog_experiments_20261012.common import sha256_file, write_npz_once
from studies.tog_experiments_20261012.e2_groups import pre_tau_group, stratum

COL = {name: i for i, name in enumerate(COMPONENT_COLUMNS)}
TEAM = np.array([100] * 5 + [200] * 5, dtype=np.int16)
MATCH_ROLES = (0, 0, 3, 3, 3, 4, 5, 5)          # role code of each synthetic match: TRAIN, Q_DEV, Q_CAL, TEST
COMPONENTS = 6                                 # per match; the last one is not primary-supported (no q row)
# Players parked on a grid 3,000 apart (no link at any radius); the listed participants sit in a 900-unit box at
# least 4,100 from the grid, so the stratum is the same at radius 1,800, 2,000 and 2,500.
FAR = np.array([[500 + 3_000 * (i % 5), 16_000 + 3_000 * (i // 5)] for i in range(10)], dtype=float)
CLOSE = {0: tuple(range(10)), 1: (0, 1, 5, 6), 2: (), 3: (0, 1, 2, 3, 5, 6), 4: (0, 5, 6, 7), 5: ()}
STRATUM_OF_COMPONENT = {0: "teamfight", 1: "small", 2: "separated", 3: "asymmetric", 4: "pick"}


# ---------------------------------------------------------------- pure checks

def test_team_ok_needs_five_players_of_each_team():
    assert E.team_ok(TEAM)
    assert not E.team_ok(np.array([100] * 6 + [200] * 4))
    assert not E.team_ok(np.array([100] * 5 + [300] * 5))
    assert not E.team_ok(np.array([100] * 5 + [200] * 4))


def test_frame_failures_mirror_headcount_tau():
    clock = np.array([0, 60_000, 120_000], dtype=np.int64)
    assert E.frame_failures(1, 3, clock, 60_000, 65_000, 5_000) == (0, 5_000)
    assert E.frame_failures(1, 3, clock, 61_000, 65_000, 5_000) == (2, 5_000)    # frame time is not F_pre_ms
    assert E.frame_failures(1, 3, clock, 60_000, 65_000, 4_000) == (4, 5_000)    # age differs from the stored age
    assert E.frame_failures(1, 3, clock, 60_000, 71_000, 11_000) == (4, 11_000)  # age above 10 s
    assert E.frame_failures(3, 3, clock, 60_000, 65_000, 5_000) == (7, 65_001)   # index outside the frames
    assert E.frame_failures(-1, 3, clock, 60_000, 65_000, 5_000) == (7, 65_001)
    assert E.frame_failures(1, 4, clock, 60_000, 65_000, 5_000) == (7, 65_001)   # clock and frames differ in length


def test_decide_has_three_outcomes_on_the_interval():
    assert E.decide([0.001, 0.004]) == "reversal"
    assert E.decide([-0.004, -0.001]) == "gain_remains"
    assert E.decide([-0.001, 0.001]) == "undetermined"
    assert E.decide([0.0, 0.002]) == "undetermined"      # the lower bound must be above 0
    assert E.decide([-0.002, 0.0]) == "undetermined"     # the upper bound must be below 0
    with pytest.raises(ValueError):
        E.decide([float("nan"), 0.1])
    with pytest.raises(ValueError):
        E.decide([0.2, 0.1])


def test_auc_difference_equals_qeval_on_each_expanded_resample():
    rng = np.random.default_rng(3)
    match = np.repeat(np.arange(14) * 7 + 2, rng.integers(1, 5, size=14))
    y = rng.integers(0, 2, size=match.size).astype(float)
    p_a, p_b = np.round(rng.random(match.size), 1), rng.random(match.size)     # p_a has ties
    out = E.auc_difference(y, p_a, p_b, match, resamples=8, seed=11, level=0.95, return_draws=True)
    assert out["point"] == pytest.approx(qeval.auc(y, p_a, match_ids=match) - qeval.auc(y, p_b, match_ids=match), abs=1e-12)
    unique, code = np.unique(match, return_inverse=True)
    for draw, positions in zip(out["draws"], qeval.bootstrap_indices(unique.size, 8, 11)):
        parts = [np.flatnonzero(code == g) for g in positions]
        rows = np.concatenate(parts)
        copies = np.repeat(np.arange(len(parts)), [len(part) for part in parts])   # a match drawn twice counts twice
        if np.unique(y[rows]).size < 2:
            assert np.isnan(draw)
            continue
        expected = qeval.auc(y[rows], p_a[rows], match_ids=copies) - qeval.auc(y[rows], p_b[rows], match_ids=copies)
        assert draw == pytest.approx(expected, abs=1e-12)
    if out["undefined_draws"] == 0:
        assert out["interval"] == pytest.approx(np.quantile(out["draws"], [0.025, 0.975]).tolist(), abs=0)


def test_evaluate_pair_is_the_qeval_paired_match_bootstrap():
    rng = np.random.default_rng(5)
    match = np.repeat(np.arange(40), rng.integers(1, 6, size=40))
    y = rng.integers(0, 2, size=match.size).astype(float)
    p1, pt = rng.random(match.size), rng.random(match.size)
    out = E.evaluate_pair(y, p1, pt, match)
    ids, l1 = qeval.per_match_loss(y, p1, match)
    _, l2 = qeval.per_match_loss(y, pt, match)
    boot = qeval.paired_match_bootstrap(ids, {"P1": l1, "PT_flex": l2}, {"d": qeval.difference("P1", "PT_flex")}, levels=0.95)
    assert out["rows"] == y.size and out["matches"] == ids.size == 40
    assert out["brier_difference"]["point"] == pytest.approx(l1.mean() - l2.mean(), abs=1e-15)
    assert out["brier_difference"]["interval"] == boot["contrasts"]["d"]["interval"]
    assert out["brier_difference"]["level"] == 0.95
    assert out["auc"]["P1"] == pytest.approx(qeval.auc(y, p1, match_ids=match), abs=1e-15)


def test_evaluate_strata_weights_each_match_equally_within_a_stratum():
    y = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    p1 = np.array([0.9, 0.2, 0.4, 0.7, 0.5, 0.1])
    pt = np.full(6, 0.5)
    match = np.array([0, 0, 0, 1, 1, 2])
    labels = np.array(["teamfight", "teamfight", "pick", "teamfight", "pick", "pick"])
    out = E.evaluate_strata(y, p1, pt, match, labels)
    # teamfight: match 0 rows (0.01 + 0.04) / 2, match 1 row 0.09 -> (0.025 + 0.09) / 2
    assert out["teamfight"]["brier"]["P1"] == pytest.approx(0.0575)
    assert out["teamfight"]["brier_difference"]["point"] == pytest.approx(0.0575 - 0.25)
    assert out["pick"]["brier"]["P1"] == pytest.approx((0.36 + 0.25 + 0.01) / 3)
    assert out["teamfight"]["matches"] == 2 and out["pick"]["matches"] == 3
    assert out["separated"] == {"rows": 0, "matches": 0}
    with pytest.raises(ValueError):
        E.evaluate_strata(y, p1, pt, match, np.array(["fight"] * 6))


def test_build_groups_keeps_the_q_row_order_of_one_role():
    groups = np.full((5, len(E.RADII), 3), -1, dtype=np.int8)
    groups[:, :, :] = [[0, 0, 0], [1, 4, 0], [4, 5, 0]]                    # per radius: separated, pick, teamfight
    groups[2, 1] = (2, 3, 0)
    aligned = {"engagement_key": np.array([b"a", b"b", b"c", b"d", b"e"], dtype="S129"),
               "role": np.array([3, 5, 3, 3, 0], dtype=np.int8), "groups": groups}
    table = E.build_groups("Q_DEV", 2000.0, aligned)
    assert table["engagement_key"].tolist() == [b"a", b"c", b"d"]
    assert table["stratum"].tolist() == ["pick", "small", "pick"]
    assert table["n_small"].tolist() == [1, 2, 1] and table["n_large"].tolist() == [4, 3, 4]
    assert E.build_groups("TEST", 2500.0, aligned)["stratum"].tolist() == ["teamfight"]


def test_append_opening_creates_appends_and_refuses_a_second_opening(tmp_path):
    path = tmp_path / "OPENINGS.json"
    assert E.append_opening(path, {"experiment": "E3", "utc": "t0", "lock_sha256": "a"}) == []
    before = E.append_opening(path, {"experiment": "E2", "utc": "t1", "lock_sha256": "b"})
    assert [o["experiment"] for o in before] == ["E3"]
    with pytest.raises(RuntimeError):
        E.append_opening(path, {"experiment": "E2", "utc": "t2", "lock_sha256": "c"})
    assert [o["lock_sha256"] for o in json.loads(path.read_text(encoding="utf-8"))["openings"]] == ["a", "b"]


def test_write_npz_once_refuses_a_second_write(tmp_path):
    target = tmp_path / "t.npz"
    digest = write_npz_once(target, {"a": np.arange(3)})
    assert digest == sha256_file(target)
    with pytest.raises(FileExistsError):
        write_npz_once(target, {"a": np.arange(3)})


# ---------------------------------------------------------------- synthetic world

def _offsets(counts):
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def make_world(tmp_path, *, team_fault=None, age_fault=None, zero_position=None):
    """A one-shard bank, Q_ROWS_h90, XY.npy, predictions, TEST scores and sealed labels; returns E.Paths."""

    rng = np.random.default_rng(20261012)
    shard = tmp_path / "bank" / "test_0000"
    shard.mkdir(parents=True)
    n_frames = 40
    values, keys, xys, clocks, teams, ledger, xy_rows = [], [], [], [], [], [], []
    q = {name: [] for name in ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "pre_ms",
                               "frame_age_s", "y")}
    for m, role in enumerate(MATCH_ROLES):
        match_key = f"m{m:05d}"
        ledger.append({"canonical_match_key": match_key, "role": role})
        clock = np.arange(n_frames, dtype=np.int64) * 60_000
        xy = (rng.random((n_frames, 10, 2)) * 14_000 + 500).astype(np.float32)
        team = TEAM.copy()
        if team_fault == m:
            team[5] = 100                                   # six of team 100
        rows = np.zeros((COMPONENTS, len(COMPONENT_COLUMNS)), dtype=np.int32)
        for c in range(COMPONENTS):
            fidx = 2 + 5 * c
            frame = FAR + rng.random((10, 2)) * 50
            close = list(CLOSE[c])
            frame[close] = rng.random(2) * 8_000 + 3_000 + rng.random((len(close), 2)) * 900
            xy[fidx] = frame.astype(np.float32)
            if zero_position == (m, c):
                xy[fidx, 0] = (0.0, 500.0)
            age = int(rng.integers(0, 10_001))
            pre = int(clock[fidx]) + age
            row = rows[c]
            row[COL["F_pre_index"]], row[COL["F_pre_ms"]], row[COL["pre"]] = fidx, clock[fidx], pre
            row[COL["f"]], row[COL["pre_frame_age_ms"]] = pre + 10_000, age + (1 if age_fault == (m, c) else 0)
            eligible = c < COMPONENTS - 1
            row[COL["primary_candidate_eligible"]] = row[COL["supported_candidate_eligible"]] = int(eligible)
            key = f"c{c:04d}"
            keys.append(key.encode("ascii"))
            if eligible:
                q["engagement_key"].append(f"{match_key}:{key}".encode("ascii"))
                q["canonical_match_key"].append(match_key.encode("ascii"))
                q["component_key"].append(key.encode("ascii"))
                q["match_index"].append(m)
                q["role"].append(role)
                q["pre_ms"].append(pre)
                q["frame_age_s"].append(age / 1000.0)
                q["y"].append(-1 if role == 5 else int(rng.integers(0, 2)))
                xy_rows.append(xy[fidx].astype(np.float64))
        values.append(rows)
        xys.append(xy)
        clocks.append(clock)
        teams.append(team)
    arrays = {"component_values": np.concatenate(values), "component_key": np.asarray(keys, dtype="S64"),
              "xy": np.concatenate(xys), "minute_ts": np.concatenate(clocks), "team_ids": np.concatenate(teams)}
    n = len(MATCH_ROLES)
    arrays.update(offset_component_values=_offsets([COMPONENTS] * n), offset_component_key=_offsets([COMPONENTS] * n),
                  offset_xy=_offsets([n_frames] * n), offset_minute_ts=_offsets([n_frames] * n),
                  offset_team_ids=_offsets([10] * n), winning_team=np.full(n, 100, dtype=np.int16),
                  offset_winning_team=_offsets([1] * n))
    np.savez(shard / "BANK.npz", **arrays)
    (shard / "MATCH_LEDGER.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger), encoding="utf-8")
    bank = shard.parent
    (bank / "FIELD_MANIFEST.json").write_text(json.dumps({"component_columns": list(COMPONENT_COLUMNS),
                                                          "participant_order": list(range(1, 11)), "team_order": [100, 200]}))
    (bank / "RUN_MANIFEST.json").write_text("{}")
    labels = tmp_path / "labels"
    labels.mkdir()
    manifest = labels / "ENGAGEMENTS_MANIFEST.json"
    manifest.write_text(json.dumps({"inputs": {"bank_shards_sha256": {"test_0000": sha256_file(shard / "BANK.npz")}}}))

    q_run = tmp_path / "q_run"
    (q_run / "inputs" / "main_h90").mkdir(parents=True)
    table = {"engagement_key": np.asarray(q["engagement_key"], dtype="S129"),
             "canonical_match_key": np.asarray(q["canonical_match_key"], dtype="S64"),
             "component_key": np.asarray(q["component_key"], dtype="S64"),
             "match_index": np.asarray(q["match_index"], dtype=np.int64), "role": np.asarray(q["role"], dtype=np.int8),
             "pre_ms": np.asarray(q["pre_ms"], dtype=np.int64), "frame_age_s": np.asarray(q["frame_age_s"]),
             "y": np.asarray(q["y"], dtype=np.int8)}
    np.savez(q_run / "Q_ROWS_h90.npz", **table)
    np.save(q_run / "inputs" / "main_h90" / "XY.npy", np.asarray(xy_rows))
    test = np.flatnonzero(table["role"] == 5)
    dev = np.flatnonzero(table["role"] == 3)
    np.savez(q_run / "Q_TEST_SEALED_h90.npz", engagement_key=table["engagement_key"][test][::-1],
             y=rng.integers(0, 2, size=test.size).astype(np.int8))
    sealed_sha = sha256_file(q_run / "Q_TEST_SEALED_h90.npz")
    (q_run / "Q_ROWS_h90.json").write_text(json.dumps({"rows_sha256": sha256_file(q_run / "Q_ROWS_h90.npz"),
                                                       "test_sealed_sha256": sealed_sha}))
    (q_run / "SCORE_LOCK.json").write_text(json.dumps({"files": {"Q_TEST_SEALED_h90.npz": sealed_sha}}))
    (q_run / "TEST_SCORES").mkdir()
    for final_key, search_key in E.CELLS.values():
        job = q_run / "jobs" / search_key
        job.mkdir(parents=True)
        p = rng.uniform(0.05, 0.95, size=dev.size)
        np.savez(job / "pred_q_dev.npz", row=dev, logit=np.log(p / (1 - p)), p=p)
        for name in ("SPEC.json", "RESULT.json"):
            (job / name).write_text("{}")
        (job / "model_refit.pkl").write_bytes(b"model")
        (q_run / "jobs" / final_key).mkdir()
        (q_run / "jobs" / final_key / "REUSE.json").write_text(json.dumps({"reuse_of_key": search_key}))
        p = rng.uniform(0.05, 0.95, size=test.size)
        scores = q_run / "TEST_SCORES" / f"{final_key}.npz"
        np.savez(scores, row=test, engagement_key=table["engagement_key"][test], logit=np.log(p / (1 - p)), p=p, p_cal=p)
        (q_run / "TEST_SCORES" / f"{final_key}.json").write_text(json.dumps({"pred_sha256": sha256_file(scores),
                                                                              "rows": int(test.size)}))
    return E.Paths(out=tmp_path / "out", bank=bank, q_run=q_run, engagements_manifest=manifest,
                   openings=tmp_path / "OPENINGS.json")


def _match_weighted_brier(y, p, match):
    ids, inverse = np.unique(match, return_inverse=True)
    return float(np.mean(np.bincount(inverse, weights=(p - y) ** 2) / np.bincount(inverse)))


def test_pipeline_runs_every_stage_on_synthetic_data(tmp_path):
    paths = make_world(tmp_path)
    assert E.groups_stage(paths, workers=0) == 0
    q = np.load(paths.q_rows)
    xy = np.load(paths.xy)
    for radius in E.RADII:
        for role, code in E.ROLE_CODES.items():
            table = np.load(paths.table(radius, role, "npz"))
            rows = np.flatnonzero(q["role"] == code)
            assert table["engagement_key"].tolist() == q["engagement_key"][rows].tolist()
            expected = [stratum(*pre_tau_group(xy[i], TEAM, radius)[:2]) for i in rows]
            assert table["stratum"].tolist() == expected
            component = [int(k.split(b":c")[1]) for k in table["engagement_key"].tolist()]
            assert expected == [STRATUM_OF_COMPONENT[c] for c in component]
            doc = json.loads(paths.table(radius, role, "json").read_text(encoding="utf-8"))
            assert doc["rows"] == rows.size == sum(s["rows"] for s in doc["strata"].values())
            assert doc["npz"]["sha256"] == sha256_file(paths.table(radius, role, "npz"))

    assert E.qdev_stage(paths) == 0
    qdev = json.loads((paths.out / E.QDEV_NAME).read_text(encoding="utf-8"))
    assert qdev["rows"] == int((q["role"] == 3).sum()) and set(qdev["results"]) == {"ALL", "r1800", "r2000", "r2500"}

    assert E.lock_stage(paths, require_committed_code=False) == 0
    lock = json.loads((paths.out / E.LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["prespec"]["commit"] and lock["prespec"]["sha256"] == sha256_file(paths.prespec)
    assert lock["verify"]["test:sealed"]["sha256"] == sha256_file(paths.sealed)
    assert set(lock["strata_tables"]) == {"r1800", "r2000", "r2500"}
    assert not paths.openings.exists() and not (paths.out / E.REPORT_NAME).exists()

    assert E.test_stage(paths) == 0
    report = json.loads((paths.out / E.REPORT_NAME).read_text(encoding="utf-8"))
    openings = json.loads(paths.openings.read_text(encoding="utf-8"))["openings"]
    assert [o["experiment"] for o in openings] == ["E2"]
    assert openings[0]["lock_sha256"] == sha256_file(paths.out / E.LOCK_NAME) == report["lock"]["sha256"]
    assert report["openings_before"] == []
    # the decision stratum, recomputed by hand from the sealed labels, the scores and the TEST group table
    sealed = np.load(paths.sealed)
    by_key = dict(zip(sealed["engagement_key"].tolist(), sealed["y"].tolist()))
    test = np.flatnonzero(q["role"] == 5)
    y = np.array([by_key[k] for k in q["engagement_key"][test].tolist()], dtype=float)
    p1 = np.load(paths.scores / f"{E.CELLS['P1'][0]}.npz")["p"]
    pt = np.load(paths.scores / f"{E.CELLS['PT_flex'][0]}.npz")["p"]
    mask = np.load(paths.table(2000.0, "TEST", "npz"))["stratum"] == "teamfight"
    match = q["match_index"][test][mask]
    point = _match_weighted_brier(y[mask], p1[mask], match) - _match_weighted_brier(y[mask], pt[mask], match)
    deciding = report["results"]["r2000"]["teamfight"]["brier_difference"]
    assert deciding["point"] == pytest.approx(point, abs=1e-12)
    assert report["decision"]["outcome"] == E.decide(deciding["interval"])
    assert report["counts"]["r2000"]["TRAIN"]["rows"] == int((q["role"] == 0).sum())
    assert set(report["stages"]) == {"groups", "qdev", "lock", "test"}
    with pytest.raises(FileExistsError):
        E.test_stage(paths)


def test_changed_file_after_the_lock_blocks_before_any_opening(tmp_path):
    paths = make_world(tmp_path)
    assert E.groups_stage(paths, workers=0) == 0 and E.qdev_stage(paths) == 0
    assert E.lock_stage(paths, require_committed_code=False) == 0
    sealed = dict(np.load(paths.sealed))
    sealed["y"] = 1 - sealed["y"]
    np.savez(paths.sealed, **sealed)
    assert E.test_stage(paths) == 2
    blocked = json.loads((paths.out / "E2_BLOCKED_test_1.json").read_text(encoding="utf-8"))
    assert "test:sealed" in blocked["reasons"]["changed_since_lock"]
    assert not paths.openings.exists() and not (paths.out / E.REPORT_NAME).exists()


def test_bad_team_map_is_counted_and_blocks_the_tables(tmp_path):
    paths = make_world(tmp_path, team_fault=1)
    assert E.groups_stage(paths, workers=0) == 2
    blocked = json.loads((paths.out / "E2_BLOCKED_groups_1.json").read_text(encoding="utf-8"))
    assert blocked["reasons"]["team_map_five_of_100_and_five_of_200"] == {"TRAIN": COMPONENTS - 1}
    assert not any(paths.out.glob("E2_GROUPS_r*"))


def test_frame_age_and_unknown_position_block_the_tables(tmp_path):
    paths = make_world(tmp_path, age_fault=(2, 0), zero_position=(3, 1))
    assert E.groups_stage(paths, workers=0) == 2
    reasons = json.loads((paths.out / "E2_BLOCKED_groups_1.json").read_text(encoding="utf-8"))["reasons"]
    assert reasons["frame_age_equals_stored_and_within_0_10000"] == {"Q_DEV": 1}
    assert reasons["rows_with_unknown_position"] == {"Q_DEV": 1}
    assert not any(paths.out.glob("E2_GROUPS_r*"))


def test_lock_is_blocked_without_a_committed_prespec(tmp_path):
    paths = make_world(tmp_path)
    assert E.groups_stage(paths, workers=0) == 0 and E.qdev_stage(paths) == 0
    loose = tmp_path / "PRESPEC_E2_copy.md"
    loose.write_text("not committed", encoding="utf-8")
    assert E.lock_stage(dataclasses.replace(paths, prespec=loose), require_committed_code=False) == 2
    assert (paths.out / "E2_BLOCKED_lock_1.json").is_file() and not (paths.out / E.LOCK_NAME).exists()
