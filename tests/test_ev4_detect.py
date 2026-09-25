"""scripts/exact_v4/ev4_01_detect.py (v4-exact stage 2, R3 detection)."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scripts" / "exact_v4"))

import ev4_01_detect as D  # noqa: E402

from gameplay import fight_clustering as FC  # noqa: E402
from gameplay.exact_population import ExactParams, detect_engagements_exact, load_exact_pack  # noqa: E402
from gameplay.grid_guard import GridForbiddenError, forbid_grid  # noqa: E402
from gameplay.split_guard import SplitViolation  # noqa: E402

TM = {i: 100 if i <= 5 else 200 for i in range(1, 11)}
P_AT, Q_AT = (5000.0, 5000.0), (12000.0, 12000.0)


def _k(ts, killer, victim, at, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": int(ts), "killerId": killer, "victimId": victim,
            "assistingParticipantIds": list(assists), "position": {"x": at[0], "y": at[1]}}


def _pack(events, n_frames=40):
    return {"meta": {"patch": "15.14", "match_id": "SYN", "team_map": dict(TM)},
            "events": sorted(events, key=lambda e: e["timestamp"]),
            "minute_ts": np.arange(n_frames, dtype=np.int64) * 60000}


def _syn_events():
    return [
        # 5 v 5 by credit, tau = 600,000 (frame age 0 -> clean), isolated
        _k(615_000, 6, 1, P_AT, assists=[7, 8, 9, 10]), _k(617_000, 2, 7, P_AT, assists=[3, 4, 5]),
        # 1 v 2 pick far away, tau = 885,000 (frame age 45 s -> not clean)
        _k(900_000, 6, 2, Q_AT, assists=[7]),
        # 2 v 2 skirmish
        _k(1_215_000, 3, 8, P_AT, assists=[4]), _k(1_216_000, 9, 3, P_AT),
    ]


@pytest.fixture(scope="module")
def params():
    c, info = D.build_v4_cfg()
    return D.v4_params(c), D.r2_params(), info


# ------------------------------------------------------------------ parameters and preset
def test_params_from_preset_and_r2(params):
    p4, pr2, info = params
    assert (p4.gap_ms, p4.diameter, p4.clean_max_age_ms) == (14000, 4300.0, 10000)
    assert (pr2.gap_ms, pr2.diameter) == (13700, 4264.0)
    ref = asdict(ExactParams(13700, 4264.0))
    got = asdict(pr2)
    ref.pop("clean_max_age_ms"), got.pop("clean_max_age_ms")    # not used by r2_repro
    assert got == ref
    assert info["locked_checked"] == {"TF2_KILL_CLUSTER_GAP_MS": 14000, "CLUSTER_MAX_DIAMETER": 4300.0,
                                      "ENG_BOUNDARIES_LOCKED": True}
    assert len(info["preset_values_sha256"]) == 64


def test_record1a_mismatch_stops(tmp_path):
    rec = json.loads(D.RECORD1A.read_text(encoding="utf-8"))
    rec["locked"]["TF2_KILL_CLUSTER_GAP_MS"] = 13700
    bad = tmp_path / "record1a_bad.json"
    bad.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="record 1A locked"):
        D.build_v4_cfg(bad)
    rec = json.loads(D.RECORD1A.read_text(encoding="utf-8"))
    rec["hashes"] = {"core/presets.py": "0" * 64}
    drift = tmp_path / "record1a_drift.json"
    drift.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="presets.py"):
        D.build_v4_cfg(drift)
    _, info = D.build_v4_cfg(drift, allow_presets_drift=True)
    assert info["presets_py_drift"] is True


# ------------------------------------------------------------------ split protection
def test_selection_patches_pass_without_record():
    for p in ("15.14", "15.15", "15.14.701.4353"):
        assert D.check_patch_access(p, None, None)["held_out"] is False


@pytest.mark.parametrize("patch", ["15.16", "16.14", "15.18", "15.22"])
def test_held_out_refused_without_record(patch):
    with pytest.raises(SplitViolation):
        D.check_patch_access(patch, None, None)


def test_main_refuses_1516_before_reading_any_list(tmp_path):
    missing = tmp_path / "no_such_matches.json"          # would raise FileNotFoundError if it were read
    with pytest.raises(SplitViolation):
        D.main(["--patch", "15.14", "--patch", "15.16", "--matches-json", str(missing), "--out-dir", str(tmp_path)])
    assert not any(tmp_path.glob("engagements_*"))        # 15.14 is not run either: all checks come first


@pytest.mark.skipif(not D.EC.PREDECISIONS_RECORD.is_file(), reason="pre-decision record not present")
def test_main_refuses_edited_or_missing_predecision_record_first(tmp_path, monkeypatch):
    edited = tmp_path / "predecisions_edited.json"
    edited.write_bytes(D.EC.PREDECISIONS_RECORD.read_bytes() + b"\n")    # a copy; the real record is untouched

    def boom(*a, **k):
        raise AssertionError("work started before the pre-decision record was checked")

    for name in ("check_patch_access", "build_v4_cfg", "load_match_list", "run_patch"):
        monkeypatch.setattr(D, name, boom)
    for rec, exc in ((edited, RuntimeError), (tmp_path / "missing.json", FileNotFoundError)):
        monkeypatch.setattr(D.EC, "PREDECISIONS_RECORD", rec)
        with pytest.raises(exc):
            D.main(["--patch", "15.14", "--limit", "1", "--out-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_held_out_needs_full_record_with_matching_sha(tmp_path):
    sha1a = hashlib.sha256(D.RECORD1A.read_bytes()).hexdigest()
    with pytest.raises(SplitViolation, match="record 1A"):
        D.check_patch_access("15.16", str(D.RECORD1A), sha1a)            # partial record does not unlock
    with pytest.raises(SplitViolation):
        D.check_patch_access("15.16", str(D.RECORD1A), None)             # sha is mandatory
    full = tmp_path / "record1_full_test.json"
    full.write_text(json.dumps({"record": "1 (complete)", "locked": {"x": 1}}), encoding="utf-8")
    sha = hashlib.sha256(full.read_bytes()).hexdigest()
    with pytest.raises(SplitViolation, match="hash mismatch"):
        D.check_patch_access("15.16", str(full), "f" * 64)
    with pytest.raises(SplitViolation, match="not found"):
        D.check_patch_access("15.16", str(tmp_path / "nope.json"), sha)
    acc = D.check_patch_access("15.16", str(full), sha)
    assert acc["held_out"] is True and acc["record1_sha256"] == sha


def test_match_list_filters_patch_first(tmp_path):
    rows = [{"match_id": "B", "patch": "15.14"}, {"match_id": "X", "patch": "15.16"},
            {"match_id": "A", "patch": "15.14.701"}, {"match_id": "Y", "patch": "16.14"}, {"match_id": "Z"}]
    f = tmp_path / "m.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    assert [r["match_id"] for r in D.load_match_list("15.14", f)] == ["A", "B"]
    sel = D.select_matches([{"match_id": str(i)} for i in range(10)], None, 4, seed=1)
    assert sel == D.select_matches([{"match_id": str(i)} for i in range(10)], None, 4, seed=1)
    assert [r["match_id"] for r in sel] == sorted(r["match_id"] for r in sel)
    assert [r["match_id"] for r in D.select_matches([{"match_id": str(i)} for i in range(10)], 3, None, 0)] == ["0", "1", "2"]


def test_workers_capped():
    with pytest.raises(SystemExit):
        D.parse_args(["--patch", "15.14", "--workers", "5"])


# ------------------------------------------------------------------ synthetic detection
def test_detect_match_requires_grid_guard(params):
    p4, pr2, _ = params
    with pytest.raises(RuntimeError, match="forbid_grid"):
        D.detect_match(_pack(_syn_events()), "15.14", {}, p4, pr2)


def test_synthetic_rows_equal_direct_detector(params):
    p4, pr2, _ = params
    pk = _pack(_syn_events())
    mrow = {"match_id": "SYN", "duration_ms": 2_400_000, "n_frames": 40}
    with forbid_grid():
        v4, r2, d4, d2 = D.detect_match(pk, "15.14", mrow, p4, pr2)
        with pytest.raises(GridForbiddenError):
            FC.build_5s_position_grid(None)
    ref4, _ = detect_engagements_exact(pk, TM, p4, "v4")
    ref2, _ = detect_engagements_exact(pk, TM, pr2, "r2_repro")
    assert len(v4) == len(ref4) == 3 and len(r2) == len(ref2)
    for i, (row, ref) in enumerate(zip(v4, ref4)):
        assert row["eng_idx"] == i and row["tau"] == ref["tau"]
        for k in ("first_kill_ts", "last_kill_ts", "kill_idx", "n_blue", "n_red", "cohort", "t5", "alive_blue",
                  "alive_red", "frame_age_ms", "clean", "n_foreign_kills", "n_other_tau", "isolated", "n_kills"):
            assert row[k] == ref[k], k
        assert row["dur_ms"] == ref["last_kill_ts"] - ref["tau"]
        assert row["game_min"] == ref["tau"] // 60000 and row["game_minute"] == ref["tau"] / 60000
        assert row["match_duration_ms"] == 2_400_000 and row["match_last_frame_ts"] == 39 * 60000
    a, b, c = v4
    assert (a["cohort"], a["t5"], a["clean"], a["isolated"], a["frame_age_ms"]) == ("T", 1, 1, 1, 0)
    assert (b["cohort"], b["clean"], b["frame_age_ms"]) == ("P", 0, 45_000)
    assert (c["cohort"], c["n_blue"], c["n_red"]) == ("S", 2, 2)
    t = D.tally_v4(v4)
    assert t["by_cohort"] == {"T": 1, "S": 1, "ASYM": 0, "P": 1}
    assert t["cohort_x_clean_x_isolated"]["T"]["clean1"]["isolated1"] == 1
    assert sum(x["n"] for x in t["cohort_x_clean_x_isolated_flat"]) == 3
    assert t["clean_isolated_by_cohort"]["T"] == 1 and t["t5_clean_isolated"] == 1
    assert D.tally_r2(r2)["n"] == len(ref2)


# ------------------------------------------------------------------ 20 real 15.14 matches
def _real_ok():
    return D.MATCHES_JSON.is_file() and D.DEFAULT_CACHE_DIR.is_dir()


@pytest.mark.slow
@pytest.mark.skipif(not _real_ok(), reason="matches_full.json or match cache not available")
def test_real_20_matches_reproduce_detector(tmp_path, params):
    import pandas as pd
    p4, pr2, _ = params
    outs = {}
    for w in (1, 2):
        od = tmp_path / f"w{w}"
        res = D.main(["--patch", "15.14", "--limit", "20", "--workers", str(w), "--out-dir", str(od)])
        outs[w] = (od, res["15.14"])
    od, diag = outs[2]
    e = pd.read_parquet(od / "engagements_15.14.parquet")
    r = pd.read_parquet(od / "r2repro_15.14.parquet")
    m = pd.read_parquet(od / "matches_15.14.parquet")
    # deterministic: 1 worker and 2 workers give the same tables
    pd.testing.assert_frame_equal(e, pd.read_parquet(outs[1][0] / "engagements_15.14.parquet"))
    pd.testing.assert_frame_equal(r, pd.read_parquet(outs[1][0] / "r2repro_15.14.parquet"))
    ids = [x["match_id"] for x in D.load_match_list("15.14")[:20]]
    assert list(m["match_id"]) == ids and (m["status"] == "ok").all()
    assert diag["selection"]["n_selected"] == 20 and diag["status"] == {"ok": 20}
    assert list(dict.fromkeys(e["match_id"])) == [i for i in ids if (e["match_id"] == i).any()]
    n4 = n2 = 0
    for mid in ids:
        pk = load_exact_pack(mid, allowed_patches=["15.14"])
        tm = pk["meta"]["team_map"]
        ref4, _ = detect_engagements_exact(pk, tm, p4, "v4")
        ref2, _ = detect_engagements_exact(pk, tm, pr2, "r2_repro")
        got4 = e[e["match_id"] == mid].reset_index(drop=True)
        got2 = r[r["match_id"] == mid].reset_index(drop=True)
        assert len(got4) == len(ref4) and len(got2) == len(ref2), mid
        for i, ref in enumerate(ref4):
            g = got4.iloc[i]
            assert int(g["eng_idx"]) == i
            for k in ("tau", "first_kill_ts", "last_kill_ts", "n_blue", "n_red", "alive_blue", "alive_red",
                      "clean", "n_foreign_kills", "n_other_tau", "isolated", "n_kills", "t5"):
                assert int(g[k]) == int(ref[k]), (mid, i, k)
            assert g["cohort"] == ref["cohort"]
            assert list(g["kill_idx"]) == list(ref["kill_idx"])
            assert (pd.isna(g["frame_age_ms"]) and ref["frame_age_ms"] is None) or int(g["frame_age_ms"]) == ref["frame_age_ms"]
        for i, ref in enumerate(ref2):
            g = got2.iloc[i]
            for k in ("engage_ts", "first_kill_ts", "last_kill_ts", "killer", "victim", "n_blue", "n_red", "prox"):
                assert int(g[k]) == int(ref[k]), (mid, i, k)
            assert g["cohort"] == ref["cohort"] and g["context"] == ref["context"]
        n4 += len(ref4)
        n2 += len(ref2)
    assert diag["v4"]["n"] == n4 == len(e) and diag["r2_repro"]["n"] == n2 == len(r)
    assert sum(x["n"] for x in diag["v4"]["cohort_x_clean_x_isolated_flat"]) == n4
    assert diag["v4"]["detector_diag_sums"]["finals"] == n4
    assert diag["r2_repro"]["detector_diag_sums"]["finals"] == n2
    stored = json.loads((od / "diag_15.14.json").read_text(encoding="utf-8"))
    for name, rec in stored["outputs"].items():
        assert hashlib.sha256(Path(rec["path"]).read_bytes()).hexdigest() == rec["sha256"], name


def test_buffer_chunks_concat_in_order():
    rows = [{"a": i, "b": [i]} for i in range(7)]
    b = D._Buffer(("a", "b"), flush_rows=3)
    for i in range(0, 7, 2):
        b.extend(rows[i:i + 2])
    df = b.frame()
    assert b.n == 7 and list(df["a"]) == list(range(7)) and [list(x) for x in df["b"]] == [[i] for i in range(7)]
    assert len(D._Buffer(("a",)).frame()) == 0


# ------------------------------------------------------------------ remake rule (GAME_END < 300,000 ms)
def _ge(ts):
    return {"type": "GAME_END", "timestamp": int(ts), "winningTeam": 100}


def test_remake_threshold_shared_and_split_remakes():
    import ev4_common as EC
    assert EC.REMAKE_MAX_GAME_END_MS == 300_000
    assert D.EC is EC
    assert EC.is_remake(299_999) and not EC.is_remake(300_000) and not EC.is_remake(None)
    rows = [{"match_id": "A", "duration_ms": 200_000}, {"match_id": "B", "duration_ms": None},
            {"match_id": "C", "duration_ms": 300_000}, {"match_id": "D", "duration_ms": 299_999}, {"match_id": "E"}]
    kept, out = D.split_remakes(rows)
    assert [r["match_id"] for r in kept] == ["B", "C", "E"]
    assert out == [{"match_id": "A", "duration_ms": 200_000}, {"match_id": "D", "duration_ms": 299_999}]
    assert D.game_end_of([_ge(5), _ge(3), {"type": "X", "timestamp": 1}]) == 3 and D.game_end_of([]) is None
    info = EC.predecisions_info()
    assert info["sha256"] == EC.PREDECISIONS_SHA256 and "remakes" in info["content"]


def test_run_patch_excludes_remakes_at_list_and_event_stage(tmp_path, params, monkeypatch):
    import argparse
    import pandas as pd
    p4, pr2, info = params
    rows = [{"match_id": "A", "patch": "15.14", "duration_ms": 200_000},      # list stage
            {"match_id": "B", "patch": "15.14", "duration_ms": None},         # event stage (GAME_END 250 s)
            {"match_id": "C", "patch": "15.14", "duration_ms": 1_800_000},
            {"match_id": "D", "patch": "15.14", "duration_ms": 2_000_000},    # list duration != GAME_END
            {"match_id": "E", "patch": "15.14", "duration_ms": 300_000}]      # exactly 300 s: kept
    mj = tmp_path / "m.json"
    mj.write_text(json.dumps(rows), encoding="utf-8")
    packs = {"B": _pack([_k(200_000, 6, 1, P_AT), _ge(250_000)]),
             "C": _pack(_syn_events() + [_ge(1_800_000)]),
             "D": _pack(_syn_events() + [_ge(1_900_000)]),
             "E": _pack([_ge(300_000)])}
    loaded = []

    def fake_load(mid, cache_dir, allowed_patches=None):
        loaded.append(mid)
        pk = packs[mid]
        return {"meta": dict(pk["meta"], match_id=mid), "events": list(pk["events"]), "minute_ts": pk["minute_ts"]}

    monkeypatch.setattr(D, "load_exact_pack", fake_load)
    args = argparse.Namespace(matches_json=mj, limit=None, sample=None, seed=1, workers=1, cache_dir=tmp_path)
    d = D.run_patch("15.14", args, info, p4, pr2, {"patch": "15.14", "held_out": False}, tmp_path / "out")
    assert "A" not in loaded                                           # never loaded
    r = d["remakes"]
    assert r["threshold_ms"] == 300_000 and r["n_excluded"] == 2 and r["n_selected_before_exclusion"] == 5
    assert [x["match_id"] for x in r["excluded_list_stage"]] == ["A"]
    assert [x["match_id"] for x in r["excluded_event_stage"]] == ["B"] and r["excluded_event_stage"][0]["game_end"] == 250_000
    assert r["n_duration_vs_game_end_mismatch"] == 1 and r["duration_vs_game_end_mismatch"][0]["match_id"] == "D"
    assert d["status"] == {"remake_excluded": 2, "ok": 3}
    assert d["predecisions"]["sha256"] == D.EC.PREDECISIONS_SHA256
    m = pd.read_parquet(tmp_path / "out" / "matches_15.14.parquet")
    assert m["match_id"].tolist() == ["A", "B", "C", "D", "E"]
    assert m.set_index("match_id")["status"].to_dict() == {"A": "remake_excluded", "B": "remake_excluded", "C": "ok",
                                                           "D": "ok", "E": "ok"}
    e = pd.read_parquet(tmp_path / "out" / "engagements_15.14.parquet")
    assert set(e["match_id"]) == {"C", "D"} and len(e) == 6
    assert d["v4"]["per_match"] == 6 / 3                               # per 'ok' match
    r2 = pd.read_parquet(tmp_path / "out" / "r2repro_15.14.parquet")
    assert not set(r2["match_id"]) & {"A", "B"}
