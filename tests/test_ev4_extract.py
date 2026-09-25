"""Tests for scripts/exact_v4/ev4_02_extract.py (v4-exact stage 2, R4 extraction).

Synthetic tests cover query-time rules, strata, guards and the detection-table contract; the tests marked
'slow' run the real extractor on a handful of cached 15.14 / 15.15 matches (skipped when the cache is absent).
No other patch's data is read: held-out patches are only exercised through the guard, which refuses them
before any data access.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

WT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("ev4_02_extract", WT / "scripts" / "exact_v4" / "ev4_02_extract.py")
EX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EX)

from gameplay.split_guard import SplitViolation  # noqa: E402

HAVE_CACHE = EX.CACHE.is_dir() and EX.PATCH_INDEX.is_file()


# ------------------------------------------------------------------ V query times (decision D1)
@pytest.mark.parametrize("ge", [120_001, 239_999, 240_000, 240_001, 1_234_567, 2_400_000])
def test_v_query_times_buckets(ge):
    qs = EX.v_query_times("KR_1", ge)
    n_expected = math.ceil((ge - 1 - EX.V_START_MS + 1) / EX.V_BUCKET_MS)
    assert len(qs) == n_expected
    for j, (k, lo, hi, t) in enumerate(qs):
        assert k == j
        assert lo == EX.V_START_MS + EX.V_BUCKET_MS * k
        assert hi == min(lo + EX.V_BUCKET_MS - 1, ge - 1)
        assert lo <= t <= hi <= ge - 1
    assert qs == EX.v_query_times("KR_1", ge)                       # deterministic
    assert qs[-1][2] == ge - 1


def test_v_query_times_short_game_and_seed():
    assert EX.v_query_times("KR_1", 120_000) == []
    assert EX.v_query_times("KR_1", 60_000) == []
    a = [q[3] for q in EX.v_query_times("KR_1", 3_000_000)]
    b = [q[3] for q in EX.v_query_times("KR_2", 3_000_000)]
    assert a != b
    k = 3
    lo = EX.V_START_MS + EX.V_BUCKET_MS * k
    want = lo + int(hashlib.sha256(f"KR_1:{k}".encode()).hexdigest()[:16], 16) % EX.V_BUCKET_MS
    assert a[k] == want


def test_v_offsets_roughly_uniform():
    offs = [t - lo for m in range(400) for (_, lo, hi, t) in EX.v_query_times(f"M{m}", 1_800_000) if hi - lo == 119_999]
    assert 0.45 < np.mean(offs) / 120_000 < 0.55


# ------------------------------------------------------------------ martingale (decision D5)
def test_mart_interval_range_and_determinism():
    h_sorted = np.array([10_000, 30_000, 60_000, 90_000], dtype=np.int64)
    for m in range(200):
        mid = f"KR_{m}"
        iv = EX.mart_interval(mid, 1_500_000, h_sorted)
        assert iv is not None
        t, h, i = iv
        assert h == h_sorted[i]
        assert EX.MART_START_MS <= t <= 1_500_000 - h - 1
        assert iv == EX.mart_interval(mid, 1_500_000, h_sorted)
    assert EX.mart_interval("KR_x", 100_000, h_sorted) is None      # range empty
    with pytest.raises(ValueError):
        EX.draw_h("KR_x", np.zeros(0, dtype=np.int64))


def test_mart_interval_edge_exact_fit():
    h_sorted = np.array([50_000], dtype=np.int64)
    ge = EX.MART_START_MS + 50_000 + 1                              # range [120000, 120000]
    assert EX.mart_interval("A", ge, h_sorted) == (EX.MART_START_MS, 50_000, 0)
    assert EX.mart_interval("A", ge - 1, h_sorted) is None


def test_mart_strata_boundaries():
    ts = np.array([0, 60_000, 120_000, 180_000], dtype=np.int64)
    kills = sorted([40_000, 70_000, 99_000, 100_000, 100_000, 130_000, 150_000])
    s = EX.mart_strata(ts, kills, 100_000, 50_000)                  # interval (100000, 150000]
    assert s["frame_update"] == 1 and s["n_frames_in"] == 1           # 120000 inside
    assert s["deaths_pre60"] == 4                                    # (40000, 100000]: 70000, 99000, 100000 x2
    assert s["recent_deaths_ge3"] == 1
    assert s["deaths_in"] == 2                                       # 130000, 150000
    assert s["deaths_end60"] == 5                                    # (90000, 150000]
    s2 = EX.mart_strata(ts, kills, 60_000, 50_000)                  # (60000, 110000]: no frame
    assert s2["frame_update"] == 0 and s2["n_frames_in"] == 0
    assert EX.mart_strata(ts, kills, 120_000, 1_000)["n_frames_in"] == 0   # frame at t itself is not inside
    assert [EX.phase_of(t) for t in (0, 899_999, 900_000, 1_499_999, 1_500_000)] == [0, 0, 1, 1, 2]


def test_mart_strata_end60_counts():
    kills = [90_000, 91_000, 149_000, 150_000]
    s = EX.mart_strata(np.array([0]), kills, 100_000, 50_000)        # end60 window (90000, 150000]
    assert s["deaths_end60"] == 3
    assert s["deaths_pre60"] == 2 and s["recent_deaths_ge3"] == 0


# ------------------------------------------------------------------ next start (decision D6)
def test_next_starts_use_all_engagements():
    taus = [100, 300, 200, 300, 900]
    assert EX.next_starts(taus) == [200, 900, 300, 900, None]


def test_frame_age():
    ts = np.array([0, 60_000, 120_000], dtype=np.int64)
    assert EX.frame_age(ts, 119_999) == (60_000, 59_999)
    assert EX.frame_age(ts, 120_000) == (120_000, 0)
    with pytest.raises(ValueError):
        EX.frame_age(np.array([10], dtype=np.int64), 5)


# ------------------------------------------------------------------ item observations (decision D8)
def test_item_observations():
    table = {1001: {}, 3006: {}}
    ev = [{"type": "ITEM_PURCHASED", "itemId": 1001}, {"type": "ITEM_PURCHASED", "itemId": 9999},
          {"type": "ITEM_UNDO", "beforeId": 3006, "afterId": 0}, {"type": "ITEM_SOLD", "itemId": 9999},
          {"type": "CHAMPION_KILL", "itemId": 777}, {"type": "ITEM_DESTROYED", "itemId": 8888}]
    n, unk = EX.item_observations(ev, table)
    assert n == 5
    assert unk == Counter({9999: 2, 8888: 1})


# ------------------------------------------------------------------ detection-table contract
def _row(**kw):
    r = dict(match_id="KR_1", tau=200_000, first_kill_ts=215_000, last_kill_ts=220_000, cohort="T", clean=True,
             isolated=1, frame_age_ms=20_000)
    r.update(kw)
    return r


def test_normalise_detect_rows_by_name():
    rows = EX.normalise_detect_rows([_row(tau=500_000, extra="x", n_kills=3), _row(clean=0, isolated=False)])
    assert [r["tau"] for r in rows] == [200_000, 500_000]
    assert rows[0]["clean"] == 0 and rows[0]["isolated"] == 0
    assert rows[1]["n_kills"] == 3 and "extra" not in rows[1]
    assert EX.normalise_detect_rows([_row(frame_age_ms=None)])[0]["frame_age_ms"] == -1
    bad = _row()
    del bad["frame_age_ms"]
    with pytest.raises(KeyError):
        EX.normalise_detect_rows([bad])
    with pytest.raises(ValueError):
        EX.normalise_detect_rows([_row(cohort="X")])


def test_read_detect_table_by_column_name(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = [_row(), _row(tau=300_000, isolated=0), _row(match_id="KR_2", tau=250_000), _row(match_id="KR_3")]
    df = pd.DataFrame(rows)
    df["patch"] = "15.14"
    df["unrelated"] = 1.5
    df = df[list(reversed(df.columns))]                              # order must not matter
    p = tmp_path / "det.parquet"
    df.to_parquet(p, index=False)
    by, info = EX.read_detect_table(p, "15.14", ["KR_1", "KR_2"])
    assert sorted(by) == ["KR_1", "KR_2"]
    assert [r["tau"] for r in by["KR_1"]] == [200_000, 300_000]
    assert "unrelated" not in info["columns_read"] and info["rows"] == 3
    with pytest.raises(RuntimeError):
        EX.read_detect_table(p, "15.15", ["KR_1"])                  # rows of another patch
    df.drop(columns=["cohort"]).to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(KeyError):
        EX.read_detect_table(tmp_path / "bad.parquet", "15.14", ["KR_1"])


# ------------------------------------------------------------------ guards
@pytest.mark.parametrize("patch", ["15.14", "15.15", "15.14.701.1"])
def test_selection_patches_pass_without_record(patch):
    assert EX.check_patch_access(patch)["selection_patch"] is True


@pytest.mark.parametrize("patch", ["15.16", "16.13", "15.18"])
def test_heldout_patches_need_record_and_hash(patch, tmp_path):
    with pytest.raises(SplitViolation):
        EX.check_patch_access(patch)
    rec = tmp_path / "record1.json"
    rec.write_text(json.dumps({"record": "1"}), encoding="utf-8")
    digest = hashlib.sha256(rec.read_bytes()).hexdigest()
    with pytest.raises(SplitViolation):
        EX.check_patch_access(patch, str(rec), None)               # hash is mandatory
    with pytest.raises(SplitViolation):
        EX.check_patch_access(patch, str(rec), "0" * 64)
    with pytest.raises(SplitViolation):
        EX.check_patch_access(patch, str(tmp_path / "missing.json"), digest)
    out = EX.check_patch_access(patch, str(rec), digest)
    assert out["record1_sha256"] == digest and out["selection_patch"] is False


def test_heldout_refuses_record_1a(tmp_path):
    rec = tmp_path / "some_record.json"
    rec.write_text(json.dumps({"record": "1A (boundaries)", "locked": {}}), encoding="utf-8")
    digest = hashlib.sha256(rec.read_bytes()).hexdigest()
    with pytest.raises(SplitViolation):
        EX.check_patch_access("15.16", str(rec), digest)


def test_match_universe_from_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    p = tmp_path / "matches_15.14.parquet"
    pd.DataFrame({"match_id": ["B", "A", "C", "D"], "status": ["ok", "ok", "missing_files", "ok"]}).to_parquet(p)
    ids = EX.match_universe("15.14", None, p)
    assert sorted(ids) == ["A", "B", "D"]
    assert ids == sorted(ids, key=lambda m: hashlib.sha256(m.encode()).hexdigest())
    assert EX.match_universe("15.14", 2, p) == ids[:2]
    pd.DataFrame({"match_id": ["A"], "patch": ["15.15"]}).to_parquet(p)
    with pytest.raises(RuntimeError):
        EX.match_universe("15.14", None, p)


def test_main_refuses_heldout_before_any_data(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("data touched before the split guard")
    monkeypatch.setattr(EX, "match_universe", boom)
    monkeypatch.setattr(EX, "load_pack", boom)
    monkeypatch.setattr(EX, "read_detect_table", boom)
    monkeypatch.setattr(EX, "load_params", boom)
    with pytest.raises(SplitViolation):
        EX.main(["--patch", "15.16", "--inline-detect", "--limit", "1"])


@pytest.mark.skipif(not EX.EC.PREDECISIONS_RECORD.is_file(), reason="pre-decision record not present")
def test_main_refuses_edited_or_missing_predecision_record_first(tmp_path, monkeypatch):
    edited = tmp_path / "predecisions_edited.json"
    edited.write_bytes(EX.EC.PREDECISIONS_RECORD.read_bytes() + b"\n")   # a copy; the real record is untouched

    def boom(*a, **k):
        raise AssertionError("work started before the pre-decision record was checked")

    for name in ("check_patch_access", "load_params", "match_universe", "load_pack", "read_detect_table",
                 "load_h_dist"):
        monkeypatch.setattr(EX, name, boom)
    for rec, exc in ((edited, RuntimeError), (tmp_path / "missing.json", FileNotFoundError)):
        monkeypatch.setattr(EX.EC, "PREDECISIONS_RECORD", rec)
        with pytest.raises(exc):
            EX.main(["--patch", "15.14", "--inline-detect", "--limit", "1", "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_load_params_preset_and_record():
    c, params, info = EX.load_params()
    assert (params.gap_ms, params.diameter) == (14_000, 4300.0)
    assert info["record1a_locked"]["TF2_KILL_CLUSTER_GAP_MS"] == 14_000
    assert c.ENG_BOUNDARIES_LOCKED is True
    assert EX.verify_params(info["params"]).gap_ms == 14_000


def test_load_params_rejects_record_mismatch(tmp_path):
    rec = tmp_path / "r1a.json"
    rec.write_text(json.dumps({"locked": {"TF2_KILL_CLUSTER_GAP_MS": 13_700, "CLUSTER_MAX_DIAMETER": 4300.0,
                                          "ENG_BOUNDARIES_LOCKED": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        EX.load_params(rec)
    rec.write_text(json.dumps({"locked": {"TF2_KILL_CLUSTER_GAP_MS": 14_000, "CLUSTER_MAX_DIAMETER": 4300.0}}),
                   encoding="utf-8")
    with pytest.raises(RuntimeError):
        EX.load_params(rec)


def test_run_chunk_runs_inside_forbid_grid(monkeypatch):
    from gameplay.grid_guard import grid_forbidden
    monkeypatch.setattr(EX, "_run_chunk_inner", lambda task: grid_forbidden())
    assert grid_forbidden() is False
    assert EX.run_chunk({}) is True
    assert grid_forbidden() is False


def test_load_pack_never_needs_xy_raw(tmp_path):
    mid = "KR_9"
    tm = {str(p): (100 if p <= 5 else 200) for p in range(1, 11)}
    (tmp_path / f"{mid}.meta.json").write_text(json.dumps({"patch": "15.14", "team_map": tm}), encoding="utf-8")
    (tmp_path / f"{mid}.events.json").write_text(json.dumps([{"type": "GAME_END", "timestamp": 5}]), encoding="utf-8")
    np.savez_compressed(tmp_path / f"{mid}.npz", minute_ts=np.array([0, 60_000]), node_minute=np.zeros((2, 10, 3)))
    pack = EX.load_pack(mid, "15.14", tmp_path)                     # no xy_raw_minute member at all
    assert set(pack) == {"minute_ts", "node_minute", "events", "meta"}
    assert pack["meta"]["team_map"][1] == 100
    with pytest.raises(RuntimeError):
        EX.load_pack(mid, "15.15", tmp_path)


def test_truncate_pack():
    pack = {"minute_ts": np.array([0, 60_000, 120_000]), "node_minute": np.zeros((3, 10, 2)),
            "events": [{"timestamp": 5}, {"timestamp": 60_001}, {"timestamp": 70_000}], "meta": {}}
    tp = EX.truncate_pack(pack, 60_001)
    assert list(tp["minute_ts"]) == [0, 60_000] and tp["node_minute"].shape[0] == 2
    assert [e["timestamp"] for e in tp["events"]] == [5, 60_001]


def test_load_h_dist_provenance(tmp_path):
    h = np.array([5, 7, 9], dtype=np.int64)
    np.save(tmp_path / "h_distribution.npy", h)
    digest = hashlib.sha256((tmp_path / "h_distribution.npy").read_bytes()).hexdigest()
    man = {"patch": "15.14", "h_distribution": {"sha256": digest}, "sample": False, "sample_limit": None, "n_matches": 3}
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    got = EX.load_h_dist(tmp_path / "h_distribution.npy")
    assert got["n"] == 3 and got["source_sample"] is False
    (tmp_path / "manifest.json").write_text(json.dumps({k: v for k, v in man.items() if k != "sample"}), encoding="utf-8")
    assert EX.load_h_dist(tmp_path / "h_distribution.npy")["source_sample"] is True     # unknown = sample
    (tmp_path / "manifest.json").write_text(json.dumps(dict(man, patch="15.15")), encoding="utf-8")
    with pytest.raises(RuntimeError):
        EX.load_h_dist(tmp_path / "h_distribution.npy")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(man, h_distribution={"sha256": "0"})), encoding="utf-8")
    with pytest.raises(RuntimeError):
        EX.load_h_dist(tmp_path / "h_distribution.npy")


# ------------------------------------------------------------------ real matches (15.14 / 15.15 only)
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CACHE, reason="match cache not available")
def test_extract_end_to_end_small(tmp_path):
    pd = pytest.importorskip("pandas")
    from gameplay.setup_features import SETUP_NAME_HASH
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH
    m = EX.main(["--patch", "15.14", "--inline-detect", "--limit", "4", "--chunk-size", "2", "--workers", "1",
                 "--out", str(tmp_path)])
    d = tmp_path / "15.14"
    assert m["STATE_V3_NAME_HASH"] == STATE_V3_NAME_HASH and m["SETUP_NAME_HASH"] == SETUP_NAME_HASH
    assert m["n_chunks"] == 2 and m["rows"]["mart"] == 0 and m["sample"] is True
    assert m["spot_check"]["ran"] and m["spot_check"]["failures"] == 0 and m["spot_check"]["n"] >= 1
    for cid, ch in m["chunks"].items():
        for name, f in ch["files"].items():
            assert hashlib.sha256((d / name).read_bytes()).hexdigest() == f["sha256"]
        z = np.load(d / f"chunk_{cid}.npz")
        eng = pd.read_parquet(d / f"chunk_{cid}_eng.parquet")
        v = pd.read_parquet(d / f"chunk_{cid}_v.parquet")
        assert z["eng_X"].dtype == np.float32 and z["eng_X"].shape == (len(eng), 996)
        assert z["eng_setup"].shape == (len(eng), 157) and z["v_X"].shape == (len(v), 996)
        assert tuple(z["state_columns"]) == STATE_V3_COLUMNS
        assert (eng["isolated"] == 1).all()
        clean = eng["clean"].to_numpy() == 1
        assert np.isnan(z["eng_setup"][~clean]).all() and np.isfinite(z["eng_setup"][clean]).all()
        assert (eng["setup_built"].to_numpy() == clean).all()
        assert (eng["e90"] > eng["last_kill_ts"] - 1).all() and (eng["e90_minus_tau"] > 0).all()
        age_col = STATE_V3_COLUMNS.index("snapshot_age_s")
        assert np.allclose(z["eng_X"][:, age_col], eng["frame_age_pre_ms"].to_numpy() / 1000.0, atol=1e-3)
        assert "y_blue_win" not in eng.columns and set(v["y_blue_win"].unique()) <= {0, 1}
        assert ((v["t"] >= v["bucket_start"]) & (v["t"] <= v["bucket_end"]) & (v["t"] < v["game_end"])).all()
    assert (d / "h_distribution.npy").is_file() and m["h_distribution"]["n"] >= 1
    # resume: identical plan -> nothing re-run, same manifest rows
    m2 = EX.main(["--patch", "15.14", "--inline-detect", "--limit", "4", "--chunk-size", "2", "--workers", "1",
                  "--out", str(tmp_path)])
    assert m2["rows"] == m["rows"] and m2["plan_hash"] == m["plan_hash"]
    # 15.15 with the 15.14 h distribution: one martingale interval per match
    with pytest.raises(SystemExit):
        EX.main(["--patch", "15.15", "--inline-detect", "--limit", "2", "--out", str(tmp_path),
                 "--h-dist", str(d / "h_distribution.npy")])       # sample h without --allow-sample-h
    m3 = EX.main(["--patch", "15.15", "--inline-detect", "--limit", "2", "--workers", "1", "--out", str(tmp_path),
                  "--h-dist", str(d / "h_distribution.npy"), "--allow-sample-h"])
    mart = pd.read_parquet(tmp_path / "15.15" / "chunk_00000_mart.parquet")
    z = np.load(tmp_path / "15.15" / "chunk_00000.npz")
    assert m3["rows"]["mart"] == len(mart) == z["m_X0"].shape[0] == z["m_X1"].shape[0]
    assert len(mart) <= 2 and (mart["t"] >= 120_000).all() and (mart["t1"] <= mart["game_end"] - 1).all()
    h = np.load(d / "h_distribution.npy")
    assert all(int(x) in set(h.tolist()) for x in mart["h_ms"])


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CACHE, reason="match cache not available")
def test_extract_halts_on_unknown_items(tmp_path, monkeypatch):
    import gameplay.item_state as IS
    real = IS.load_item_table_v2("15.14")
    keep = set().union(*IS.load_effect_flags(real, patch="15.14").values())
    small = {k: v for k, v in real.items() if k in keep}             # every bought item but the flag ids is unknown
    monkeypatch.setattr(IS, "load_item_table_v2", lambda patch, **kw: small)
    with pytest.raises(IS.UnknownItemRateError):
        EX.main(["--patch", "15.14", "--inline-detect", "--limit", "1", "--workers", "1", "--out", str(tmp_path)])
    assert not (tmp_path / "15.14" / "chunk_00000.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CACHE, reason="match cache not available")
def test_parquet_input_equals_inline_detection(tmp_path):
    pd = pytest.importorskip("pandas")
    ids = EX.match_universe("15.14", 2)
    (tmp_path / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
    _, params, _ = EX.load_params()
    rows = []
    for mid in ids:
        for r in EX.inline_detect(EX.load_pack(mid, "15.14"), params):
            rows.append(dict(r, patch="15.14", eng_idx=len(rows)))
    det = tmp_path / "det"
    det.mkdir()
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_parquet(det / "engagements_15.14.parquet", index=False)
    a = EX.main(["--patch", "15.14", "--inline-detect", "--matches", str(tmp_path / "ids.txt"), "--workers", "1",
                 "--out", str(tmp_path / "a")])
    b = EX.main(["--patch", "15.14", "--detect", str(det / "engagements_15.14.parquet"), "--matches",
                 str(tmp_path / "ids.txt"), "--workers", "1", "--out", str(tmp_path / "b")])
    assert a["rows"] == b["rows"] and b["detect_input"]["mode"] == "parquet"
    za, zb = np.load(tmp_path / "a" / "15.14" / "chunk_00000.npz"), np.load(tmp_path / "b" / "15.14" / "chunk_00000.npz")
    for k in ("eng_X", "eng_setup", "v_X"):
        assert np.array_equal(za[k], zb[k], equal_nan=True)
    ea = pd.read_parquet(tmp_path / "a" / "15.14" / "chunk_00000_eng.parquet")
    eb = pd.read_parquet(tmp_path / "b" / "15.14" / "chunk_00000_eng.parquet")
    pd.testing.assert_frame_equal(ea, eb)


# ------------------------------------------------------------------ remake rule (decision D11)
def test_remake_threshold_is_shared():
    import ev4_common as EC
    assert EX.EC is EC and EC.REMAKE_MAX_GAME_END_MS == 300_000
    assert "D11_remakes" in EX.DECISIONS and "scripts/exact_v4/ev4_common.py" in EX.code_hashes()


def test_worker_skips_remakes_and_records_them(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    game_end = {"R1": 106_255, "R2": 299_999}
    touched = []

    def fake_pack(mid, patch, cache=None):
        ev = [{"type": "ITEM_PURCHASED", "timestamp": 1000, "itemId": 999999},     # would count as unknown item
              {"type": "GAME_END", "timestamp": game_end[mid], "winningTeam": 100}]
        return {"minute_ts": np.array([0, 60_000]), "node_minute": np.zeros((2, 10, 2)), "events": ev,
                "meta": {"patch": patch, "match_id": mid, "team_map": {}}}

    def boom(*a, **k):
        touched.append(a)
        raise AssertionError("a remake match must not be processed")

    monkeypatch.setattr(EX, "load_pack", fake_pack)
    monkeypatch.setattr(EX, "process_match", boom)
    monkeypatch.setattr(EX, "inline_detect", boom)
    det = tmp_path / "det"
    det.mkdir()
    df = pd.DataFrame([_row(match_id="R1", tau=60_000, first_kill_ts=70_000, last_kill_ts=75_000),
                       _row(match_id="R1", tau=90_000, first_kill_ts=95_000, last_kill_ts=96_000, isolated=0)])
    df["patch"] = "15.14"
    df.to_parquet(det / "engagements_15.14.parquet", index=False)
    (tmp_path / "ids.txt").write_text("R1\nR2\n", encoding="utf-8")
    m = EX.main(["--patch", "15.14", "--detect", str(det / "engagements_15.14.parquet"), "--matches",
                 str(tmp_path / "ids.txt"), "--workers", "1", "--out", str(tmp_path / "o")])
    assert not touched
    assert m["remakes"]["threshold_ms"] == 300_000 and m["remakes"]["n_excluded"] == 2
    assert m["remakes"]["match_ids"] == ["R1", "R2"]
    assert {r["match_id"]: r["n_eng_input"] for r in m["remakes"]["excluded"]} == {"R1": 2, "R2": 0}
    assert m["match_status"] == {"remake_excluded": 2}
    assert m["rows"] == {"eng": 0, "eng_clean_setup": 0, "v": 0, "mart": 0}
    assert m["item_check"]["observations"] == 0                       # nothing of a remake is observed
    assert m["predecisions"]["sha256"] == EX.EC.PREDECISIONS_SHA256
    mt = pd.read_parquet(tmp_path / "o" / "15.14" / "chunk_00000_matches.parquet")
    assert set(mt["status"]) == {"remake_excluded"} and set(mt["game_end"]) == {106_255, 299_999}
    side = json.loads((tmp_path / "o" / "15.14" / "chunk_00000.json").read_text(encoding="utf-8"))
    assert sorted(r["match_id"] for r in side["remakes"]) == ["R1", "R2"]
    assert len(np.load(tmp_path / "o" / "15.14" / "h_distribution.npy")) == 0
