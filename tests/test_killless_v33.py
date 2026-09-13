"""Kill-less encounters (R4): preset gate, grid reproduction, scanner equivalence, participants, matched baseline,
event counts, market attribution, node columns and the match-clustered bootstrap.  Synthetic data only; no cache access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KV = _load("run_killless_v33")
CK = _load("characterize_killless_v33")
SC = _load("run_killless_encounters")

from core.config import DS_DENOM, F_GLOBAL, NODE_FEATURE_NAMES, NODE_IDX, cfg  # noqa: E402
from gameplay import labels as L  # noqa: E402

TM = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
BLUE, RED = np.arange(5), np.arange(5, 10)
TABLE = {"kills": 20.0, "assists": 45.0, "plates": 120.0, "tower_outer_turret": 540.0, "dragon": 0.0, "ward_kills": 25.0}


# ------------------------------------------------------------------ gate and denominators
def test_gate_reads_v33_preset_and_records_sources():
    g = KV.resolve_gate("v3.3")
    assert (g["radius_u"], g["grace_ms"], g["min_per_team"]) == (1600.0, 15000, 4)
    assert g["min_duration_s"] == pytest.approx(13.7)
    assert g["min_active_frames"] == 3 and g["min_span_s"] == 10.0
    assert g["sources"]["radius_u"].startswith("preset v3.3") and g["sources"]["grace_ms"].startswith("preset v3.3")
    o = KV.resolve_gate("v3.3", radius=1800.0, grace_ms=10000, min_per_team=3, min_duration_s=20.0)
    assert o["sources"]["radius_u"] == "flag --radius" and o["min_active_frames"] == 4 and o["name"] == "r1800_t3_d20_g10"
    assert KV.resolve_gate("v3.3", min_duration_s=13.72)["min_active_frames"] == 3
    with pytest.raises(KeyError):
        KV.resolve_gate("v9")


def test_grid_setting_names_match_the_grid_rows():
    assert KV.resolve_gate("v3.3")["grid_setting"] == "r1600_t4_dG_g15"
    assert KV.resolve_gate("v3.3", min_per_team=2)["grid_setting"] == "r1600_t2_dG_g15"
    assert KV.resolve_gate("v3.3", min_duration_s=20.0)["grid_setting"] == "r1600_t4_d20_g15"
    assert KV.resolve_gate("v3.3", grace_ms=10000)["grid_setting"] == "r1600_t4_dG_g10"
    assert KV.resolve_gate("v3.3", radius=1200.0)["grid_setting"] == "r1200_t4_dG_g15"
    assert KV.resolve_gate("v3.3", min_duration_s=13.7)["grid_setting"] == "r1600_t4_dG_g15"


def test_gate_records_the_track_and_suffixes_the_frame_only_track():
    on = KV.resolve_gate("v3.3")
    assert on["kill_trajectory_interp"] is True and on["track"] == "kill-adjusted"
    assert on["grid_row_same_constants"] == on["grid_setting"] == "r1600_t4_dG_g15" and "ktoff" not in on["name"]
    off = KV.resolve_gate("v3.3", min_per_team=2, kill_trajectory_interp=False)
    assert off["kill_trajectory_interp"] is False and off["track"] == "frame-only"
    assert off["name"] == "r1600_t2_d13.7_g15_ktoff" and off["grid_setting"] == "r1600_t2_dG_g15_ktoff"
    assert off["grid_row_same_constants"] == "r1600_t2_dG_g15"
    assert "TF2_USE_KILL_TRAJECTORY_INTERP=False" in off["track_rule"]
    assert "kill-adjusted" in KV.TRACK_DEVIATION and KV.TRACK_DEVIATION in KV.WRAPPER_DEVIATIONS
    assert KV.TRACK_DEVIATION in CK.DEVIATIONS and KV.ALIVE_DEVIATION in CK.DEVIATIONS
    ns = type("C", (), {})()
    assert KV.set_track(ns, False) == {"TF2_USE_KILL_TRAJECTORY_INTERP_before": None, "TF2_USE_KILL_TRAJECTORY_INTERP": False}
    assert KV.track_uses_kill_trajectory(ns) is False and KV.track_uses_kill_trajectory(None) is True
    proxy = KV.TrackCfg(cfg, False)
    assert proxy.TF2_USE_KILL_TRAJECTORY_INTERP is False and proxy.TF2_GRID_STEP_MS == cfg.TF2_GRID_STEP_MS
    assert getattr(KV.TrackCfg(None, True), "TF2_GRID_STEP_MS", 5000) == 5000


def test_scan_slice_parsing_and_code_provenance():
    assert KV.parse_scan_slice("10000:10150", 20000) == (10000, 10150)
    assert KV.parse_scan_slice("-200:", 20000) == (19800, 20000)
    assert KV.parse_scan_slice(":5", 20000) == (0, 5)
    with pytest.raises(SystemExit):
        KV.parse_scan_slice("5:5", 20000)
    with pytest.raises(SystemExit):
        KV.parse_scan_slice("5", 20000)
    prov = KV.code_provenance(("scripts/run_killless_encounters.py", "no/such/file.py"))
    assert len(prov["scripts/run_killless_encounters.py"]["sha1"]) == 40 and prov["no/such/file.py"]["sha1"] is None
    assert "scripts/run_killless_v33.py" in KV.CODE_FILES and "gameplay/labels.py" in KV.CODE_FILES


def test_n_valid_reps_counts_what_the_ci_keeps():
    reps = np.array([0.1, np.nan, 0.3, np.inf, 0.2])
    assert KV.n_valid_reps(reps) == 3 and KV._ci(reps)[0] == pytest.approx(np.percentile([0.1, 0.3, 0.2], 2.5))
    d = KV.denominator_block(np.array([2, 0, 1, 3]), np.array([1, 0, 0, 1]), [10.0], n_boot=40, seed=3,
                             enc_bands=np.array([[0, 2, 0, 0, 0], [0] * 5, [0, 0, 1, 0, 0], [0, 0, 0, 3, 0]]),
                             kl_bands=np.array([[0, 1, 0, 0, 0], [0] * 5, [0] * 5, [0, 0, 0, 1, 0]]))
    assert d["killless_share_ci_n_valid_reps"] == 40
    # band 2-10 has encounters in one match only: resamples without that match give no replicate
    assert 0 < d["by_start_band"]["2-10"]["killless_share_ci_n_valid_reps"] < 40
    assert d["by_start_band"]["lt2"]["killless_share_ci_n_valid_reps"] == 0


def test_denominator_block_and_quote_carry_the_gate(tmp_path):
    scale = tmp_path / "scale.json"
    scale.write_text(json.dumps({"n": 532547, "n_matches": 191940, "teamfight_min": 4, "y_key": "y_market_event",
                                 "teamfight_min4": {"n": 109829, "auc": 0.68}}), encoding="utf-8")
    pub = KV.published_corpus_rates(scale)
    assert pub["engagements_per_match"] == pytest.approx(2.7746, abs=1e-4)
    assert pub["teamfight_class_per_match"] == pytest.approx(0.5722, abs=1e-4)
    assert pub["teamfight_class_key"] == "teamfight_min4.n" and pub["teamfight_keys_agree"] is None
    d = KV.denominator_block(np.array([2, 0, 1, 3]), np.array([1, 0, 0, 1]), [10.0, 20.0], n_boot=50, seed=1)
    assert (d["encounters"], d["killless"], d["with_kill"]) == (6, 2, 4)
    assert d["killless_share_of_encounters"] == pytest.approx(1 / 3) and d["killless_per_match"] == 0.5
    assert d["killless_share_ci"][0] <= 1 / 3 <= d["killless_share_ci"][1]
    gate = KV.resolve_gate("v3.3")
    cmp4 = KV.corpus_comparison(d, gate, pub, None)
    # no ratio of encounter rates to engagement rates (it invites a 'share of fights missed' misquote)
    assert not any("per_comparable" in k or "ratio" in k for k in cmp4)
    side = cmp4["side_by_side_per_match"]
    assert side["killless_encounters_per_scanned_match"] == 0.5
    assert side["published_corpus.teamfight_class_per_match"] == pytest.approx(109829 / 191940)
    assert any("kill-adjusted track" in c and "conditional on the track" in c for c in cmp4["caveats"])
    q = KV.quote(gate, d, cmp4)
    assert gate["name"] in q and gate["grid_setting"] in q and "proximity encounters" in q and "teamfight-class" in q
    assert "0.5722 corpus teamfight-class engagements per corpus match (109,829 / 191,940)" in q
    assert "per scanned match" not in q and "of the kill-adjusted track" in q
    cmp2 = KV.corpus_comparison(d, KV.resolve_gate("v3.3", min_per_team=2), pub, None)
    assert "frame-only" in KV.corpus_comparison(d, KV.resolve_gate("v3.3", kill_trajectory_interp=False), pub, None)[
        "side_by_side_per_match"]["track"]
    q2 = KV.quote(KV.resolve_gate("v3.3", min_per_team=2), d, cmp2)
    assert "2.7745 corpus engagements per corpus match (532,547 / 191,940)" in q2 and "teamfight-class" not in q2
    same = {"teamfight_min": 4, "engagements_per_match": 2.408, "teamfight_class_per_match": 0.494}
    q3 = KV.quote(gate, d, KV.corpus_comparison(d, gate, pub, same))
    assert "0.4940 corpus teamfight-class engagements per scanned match; 0.5722 corpus teamfight-class" in q3


def test_published_rates_read_by_participation_scale_and_cross_check(tmp_path):
    scale = tmp_path / "scale.json"
    base = {"n": 532547, "n_matches": 191940, "teamfight_min": 4, "y_key": "y_market_event",
            "by_participation_scale": {"teamfight": {"n": 109829}, "pick": {"n": 101798}}}
    scale.write_text(json.dumps({**base, "teamfight_min4": {"n": 109829}}), encoding="utf-8")
    pub = KV.published_corpus_rates(scale)
    assert pub["teamfight_class_key"] == "by_participation_scale.teamfight.n"
    assert pub["n_teamfight_class"] == 109829 and pub["teamfight_keys_agree"] is True
    scale.write_text(json.dumps({**base, "teamfight_min4": {"n": 1}}), encoding="utf-8")
    pub = KV.published_corpus_rates(scale)
    assert pub["n_teamfight_class"] == 109829 and pub["teamfight_keys_agree"] is False
    scale.write_text(json.dumps({"n": 1, "n_matches": 1}), encoding="utf-8")
    with pytest.raises(KeyError):
        KV.published_corpus_rates(scale)
    assert KV.published_corpus_rates(tmp_path / "missing.json") is None


def test_corpus_counts_by_match_drop_draws_and_count_teamfights(tmp_path):
    np.savez_compressed(tmp_path / "shard_000.npz", groups=np.array(["KR_1", "KR_1", "KR_2", "KR_2"]),
                        y_market_event=np.array([1, -1, 0, 1], dtype=np.int8),
                        cluster_blue=np.array([4, 5, 2, 4], dtype=np.int16),
                        cluster_red=np.array([4, 5, 3, 5], dtype=np.int16),
                        patch=np.array(["15.14", "15.14", "15.15", "15.15"]))
    counts = KV.corpus_counts_by_match(tmp_path)
    assert counts == {"KR_1": (1, 1, "15.14"), "KR_2": (2, 1, "15.15")}
    assert KV.corpus_totals(counts) == {"n_matches": 2, "n_engagements": 3, "n_teamfight_class": 2}
    pub = {"n_engagements": 3, "n_teamfight_class": 2, "n_matches": 2}
    same = KV.same_sample_corpus(counts, ["KR_2", "EUW_9"], tmp_path, pub)
    assert same["matches_in_corpus"] == 1 and same["n_engagements"] == 2 and same["engagements_per_match"] == 1.0
    assert same["corpus_totals_agree_with_published"] is True


def _enc(start_s, has_kill, dur_s=10):
    return {"start_ms": int(start_s * 1000), "end_ms": int((start_s + dur_s) * 1000), "has_kill": has_kill}


def test_scan_tally_bands_checkpoints_and_band_block():
    tally = KV.ScanTally()
    tally.add(199, "A", "15.14", [_enc(90, False), _enc(300, True), _enc(700, False, 20)])
    tally.add(200, "B", "15.15", [_enc(1300, False), _enc(2000, True)])
    tally.add(201, "C", "15.15", [])
    assert tally.checkpoints == {200: {"encounters": 5, "killless": 3}}
    assert tally.patch_counts() == {"15.14": 1, "15.15": 2}
    assert tally.durations_killless == [10.0, 20.0, 10.0]
    eb, kb = tally.band_arrays()
    assert eb.shape == (3, 5) and eb[0].tolist() == [1, 1, 1, 0, 0] and kb[1].tolist() == [0, 0, 0, 1, 0]
    d = KV.denominator_block(np.array(tally.n_enc), np.array(tally.n_kl), tally.durations_killless, n_boot=30, seed=2,
                             enc_bands=eb, kl_bands=kb)
    bands = d["by_start_band"]
    assert bands["lt2"]["killless"] == 1 and bands["lt2"]["encounters"] == 1
    assert bands["from_minute_2"]["encounters"] == 4 and bands["from_minute_2"]["killless"] == 2
    assert bands["from_minute_2"]["killless_share_of_encounters"] == pytest.approx(0.5)
    assert bands["10-20"]["killless"] == 1 and bands["20-30"]["killless"] == 1 and bands["30+"]["with_kill"] == 1
    assert bands["2-10"]["with_kill"] == 1 and bands["20-30"]["with_kill"] == 0
    assert sum(bands[s]["encounters"] for s in KV.BAND_SLOTS) == d["encounters"]
    assert "killless_share_ci" in bands["from_minute_2"]
    q = KV.quote(KV.resolve_gate("v3.3"), d, {"published_corpus": None, "same_sample_corpus": None})
    assert "from minute 2 on: 50.00 %" in q


# ------------------------------------------------------------------ grid reproduction
def _grid(tmp_path, row, log_lines=None, what="v3.3 grid, seed 7"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary.json").write_text(json.dumps({"what": what, "git_commit": "abc", "rows": [row]}), encoding="utf-8")
    if log_lines is not None:
        (tmp_path / f"{row['setting']}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return tmp_path


def test_parse_grid_log_and_progress_pct(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("200/20000 matches | encounters=164 killless=2.4%\nnoise\n{\n400/20000 matches | encounters=373 killless=3.2%\n",
                 encoding="utf-8")
    log = KV.parse_grid_log(p)
    assert log == {200: {"n_sampled": 20000, "encounters": 164, "killless_pct": "2.4"},
                   400: {"n_sampled": 20000, "encounters": 373, "killless_pct": "3.2"}}
    assert KV.progress_pct(4, 164) == "2.4" and KV.progress_pct(0, 0) == "0.0"
    # the scanner's own formatting
    assert KV.progress_pct(12, 373) == f"{12 / max(1, 373) * 100:.1f}"


def test_grid_reproduction_full_sample(tmp_path):
    gate = KV.resolve_gate("v3.3")
    d = KV.denominator_block(np.array([2, 0, 1, 3]), np.array([1, 0, 0, 1]), [10.0, 20.0], n_boot=0)
    row = {"setting": "r1600_t4_dG_g15", "radius_u": 1600.0, "min_per_team": 4, "min_duration_s": 13.7, "grace_ms": 15000,
           "matches": 4, "encounters": 6, "with_kill": 4, "killless": 2, "killless_share_of_encounters": 2 / 6,
           "encounters_per_match": 1.5, "killless_per_match": 0.5, "killless_median_duration_s": 15.0}
    kw = dict(n_sampled=4, seed=7, match_source="cache", patches_filter=False, full_sample_scanned=True)
    ok = KV.grid_reproduction(gate, d, {}, grid_dir=_grid(tmp_path / "a", row), **kw)
    assert ok["status"] == "reproduced" and ok["mode"] == "full_sample" and not ok["mismatches"]
    bad = KV.grid_reproduction(gate, d, {}, grid_dir=_grid(tmp_path / "b", {**row, "killless": 3}), **kw)
    assert bad["status"] == "MISMATCH" and set(bad["mismatches"]) == {"killless"}
    other = KV.grid_reproduction(gate, d, {}, grid_dir=_grid(tmp_path / "c", row), **{**kw, "seed": 8})
    assert other["status"] == "not_comparable" and any("seed" in r for r in other["reasons"])
    t2 = KV.grid_reproduction(KV.resolve_gate("v3.3", min_per_team=2), d, {}, grid_dir=_grid(tmp_path / "d", row), **kw)
    assert t2["status"] == "not_comparable"
    assert KV.grid_reproduction(gate, d, {}, grid_dir=tmp_path / "none", **kw)["status"] == "no_grid"


def test_grid_reproduction_prefix_against_the_log(tmp_path):
    gate = KV.resolve_gate("v3.3")
    d = KV.denominator_block(np.array([1]), np.array([0]), [], n_boot=0)
    row = {"setting": "r1600_t4_dG_g15", "radius_u": 1600.0, "min_per_team": 4, "min_duration_s": 13.7, "grace_ms": 15000,
           "matches": 400, "encounters": 373, "with_kill": 361, "killless": 12}
    grid = _grid(tmp_path, row, ["200/400 matches | encounters=164 killless=2.4%", "400/400 matches | encounters=373 killless=3.2%"])
    kw = dict(n_sampled=400, seed=7, match_source="cache", patches_filter=False, full_sample_scanned=False, grid_dir=grid)
    ok = KV.grid_reproduction(gate, d, {200: {"encounters": 164, "killless": 4}}, **kw)
    assert ok["status"] == "prefix_reproduced" and ok["checkpoints_compared"] == [200]
    bad = KV.grid_reproduction(gate, d, {200: {"encounters": 164, "killless": 5}}, **kw)
    assert bad["status"] == "MISMATCH" and "200" in bad["mismatches"]
    gap = KV.grid_reproduction(gate, d, {200: {"encounters": 164, "killless": 4}, 600: {"encounters": 500, "killless": 9}}, **kw)
    assert gap["status"] == "MISMATCH" and gap["mismatches"]["checkpoints_missing"]["only_in_this_run"] == [600]
    wrong_n = KV.grid_reproduction(gate, d, {200: {"encounters": 164, "killless": 4}}, **{**kw, "n_sampled": 200})
    assert wrong_n["status"] == "not_comparable"
    sliced = KV.grid_reproduction(gate, d, {200: {"encounters": 164, "killless": 4}}, **kw, scan_offset=10)
    assert sliced["status"] == "not_comparable" and any("sample position 11" in r for r in sliced["reasons"])
    assert "track_contrast" not in sliced


def test_grid_reproduction_frame_only_track_is_not_comparable_but_contrasted(tmp_path):
    off = KV.resolve_gate("v3.3", kill_trajectory_interp=False)
    row = {"setting": "r1600_t4_dG_g15", "radius_u": 1600.0, "min_per_team": 4, "min_duration_s": 13.7, "grace_ms": 15000,
           "matches": 400, "encounters": 373, "with_kill": 361, "killless": 12, "killless_share_of_encounters": 12 / 373,
           "encounters_per_match": 373 / 400, "killless_per_match": 12 / 400, "killless_median_duration_s": 15.0}
    grid = _grid(tmp_path, row, ["200/400 matches | encounters=164 killless=2.4%", "400/400 matches | encounters=373 killless=3.2%"])
    kw = dict(n_sampled=400, seed=7, match_source="cache", patches_filter=False, grid_dir=grid)
    d = KV.denominator_block(np.array([1] * 400), np.array([0] * 399 + [1]), [10.0], n_boot=0)
    # prefix: identical numbers still do not count as reproduced on the other track
    pre = KV.grid_reproduction(off, d, {200: {"encounters": 120, "killless": 5}}, full_sample_scanned=False, **kw)
    assert pre["status"] == "not_comparable" and pre["reasons"] == [KV.KT_OFF_REASON]
    cp = pre["track_contrast"]["checkpoints"]["200"]
    assert cp["kill_adjusted_grid_log"]["encounters"] == 164 and cp["frame_only_this_run"]["encounters"] == 120
    full = KV.grid_reproduction(off, d, {}, full_sample_scanned=True, **kw)
    tc = full["track_contrast"]
    assert full["status"] == "not_comparable" and tc["mode"] == "full_sample"
    assert tc["kill_adjusted_grid_row"]["encounters"] == 373 and tc["frame_only_this_run"]["encounters"] == 400
    assert tc["frame_only_this_run"]["matches"] == 400
    both = KV.grid_reproduction(off, d, {}, full_sample_scanned=True, **{**kw, "seed": 8})
    assert both["status"] == "not_comparable" and "track_contrast" not in both


# ------------------------------------------------------------------ scanner equivalence and participants
def _tracks(seed, n_frames=160):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(500, 14500, size=(n_frames, 10, 2))
    for start in (20, 70, 120):
        centre = rng.uniform(3000, 12000, size=2)
        length = int(rng.integers(2, 9))
        xy[start:start + length] = centre + rng.normal(0, 900, size=(length, 10, 2))
    alive = rng.random((n_frames, 10)) > 0.08
    xy[rng.random((n_frames, 10)) < 0.01] = np.nan
    ts = 90_000 + 5000 * np.arange(n_frames, dtype=np.int64)
    kills = np.sort(rng.choice(ts, size=4, replace=False))
    return xy.astype(np.float32), ts, alive, kills


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("m", [2, 3, 4])
def test_frame_membership_reproduces_the_scanner(seed, m):
    xy, ts, alive, kills = _tracks(seed)
    for alive_arg in (alive, None):
        mem = CK.frame_membership(xy, alive_arg, BLUE, RED, 1600.0, m)
        for dur, frames in ((0.1, 1), (13.7, 3)):
            ref = SC.encounters_for_match(xy, ts, alive_arg, BLUE, RED, kills, 1600.0, m, dur, 15000)
            mine = CK.runs_from_active(mem["active"], frames)
            assert [(int(ts[s]), int(ts[e])) for s, e in mine] == [(r["start_ms"], r["end_ms"]) for r in ref]
            if m == 2 and frames == 1:
                assert ref, "synthetic tracks should hold at least one encounter"


@pytest.mark.parametrize("seed", [0, 3])
def test_active_at_equals_membership_at_that_threshold(seed):
    xy, ts, alive, kills = _tracks(seed)
    mem4 = CK.frame_membership(xy, alive, BLUE, RED, 1600.0, 4)
    for m in (1, 2, 3, 4):
        assert np.array_equal(CK.active_at(mem4, m), CK.frame_membership(xy, alive, BLUE, RED, 1600.0, m)["active"])
    assert np.all(CK.active_at(mem4, 2) >= mem4["active"])


def test_encounter_members_exclude_bystanders_and_centre_the_fight():
    xy = np.full((4, 10, 2), 1000.0)
    for p in (0, 1, 2, 3, 5, 6, 7, 8):
        xy[:, p] = (6000.0 + (p % 4) * 100.0, 6000.0)
    xy[:, 4] = (14000.0, 14000.0)
    xy[:, 9] = (9500.0, 6000.0)
    mem = CK.frame_membership(xy.astype(np.float32), None, BLUE, RED, 1600.0, 4)
    assert mem["active"].all()
    info = CK.encounter_members(mem, 0, 3)
    members = [0, 1, 2, 3, 5, 6, 7, 8]
    assert np.flatnonzero(info["members"]).tolist() == members
    assert (info["n_blue"], info["n_red"], info["peak_blue"], info["peak_red"]) == (4, 4, 4, 4)
    assert np.allclose(info["anchor"], xy[0, members].mean(axis=0))


def test_find_baseline_keeps_grace_band_and_kill_clear():
    ts = 5000 * np.arange(400, dtype=np.int64)
    active = np.zeros(400, dtype=bool)
    active[150:156] = True
    kills = np.array([5000 * 100], dtype=np.int64)
    band = CK.band_index(ts[150])
    assert band == 1
    b = CK.find_baseline(active, ts, kills, 6, band, int(ts[150]), 15000, 5000, 15000, int(ts[-1]))
    assert b == 141  # 141 and 159 are both 9 frames away; the earlier wins
    assert not active[b - 3:b + 6 + 3].any() and CK.band_index(ts[b]) == band
    b2 = CK.find_baseline(active, ts, np.array([int(ts[141])], dtype=np.int64), 6, band, int(ts[150]), 15000, 5000, 15000, int(ts[-1]))
    assert b2 == 159  # a kill within grace of every earlier near candidate
    late = active.copy()
    late[360:] = True
    assert CK.find_baseline(late, ts, kills, 6, 3, int(ts[380]), 15000, 5000, 15000, int(ts[-1])) is None
    assert CK.find_baseline(active, ts, kills, 6, -1, int(ts[10]), 15000, 5000, 15000, int(ts[-1])) is None
    # a looser exclusion (2-per-side proximity just before the encounter) pushes the baseline away
    looser = active.copy()
    looser[137:141] = True
    b3 = CK.find_baseline(looser, ts, kills, 6, band, int(ts[150]), 15000, 5000, 15000, int(ts[-1]))
    assert b3 == 159 and not looser[b3 - 3:b3 + 6 + 3].any()


# ------------------------------------------------------------------ events, node columns
def test_event_index_counts_participants_and_map():
    events = [
        {"type": "WARD_PLACED", "timestamp": 1000, "creatorId": 3},
        {"type": "ITEM_PURCHASED", "timestamp": 1500, "participantId": 3},
        {"type": "WARD_KILL", "timestamp": 2000, "killerId": 7},
        {"type": "BUILDING_KILL", "timestamp": 3000, "killerId": 0, "assistingParticipantIds": [2], "teamId": 200},
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 4000, "killerId": 0, "teamId": 200},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 5000, "killerId": 9, "killerTeamId": 200},
    ]
    idx = CK.EventIndex(events)
    members = np.zeros(10, dtype=bool)
    members[[1, 2]] = True
    bits = CK.member_bits(members)
    assert idx.count("WARD_PLACED", 0, 4500, bits) == (1, 1)
    assert idx.count("WARD_KILL", 0, 4500, bits) == (0, 1)
    assert idx.count("BUILDING_KILL", 0, 4500, bits) == (1, 1)
    assert idx.count("TURRET_PLATE_DESTROYED", 0, 4000, bits) == (0, 0)
    ev = CK.window_events(idx, 0, 3999, 1000, bits)
    assert ev["ev_in_plate_map"] == 0 and ev["ev_after_plate_map"] == 1 and ev["ev_after_elite_map"] == 0
    assert ev["ev_in_ward_placed_part"] == 1 and ev["ev_in_building_part"] == 1
    assert set(CK.EVENT_COLS) <= set(ev) and len(ev) == len(CK.EVENT_COLS)
    # without a team map only structure / monster sides resolve
    assert ev["ev_in_ward_placed_bmr"] == 0 and ev["ev_in_building_bmr"] == 1 and ev["ev_after_plate_bmr"] == 1


def test_event_side_balance_follows_the_label_team_sign():
    events = [
        {"type": "WARD_PLACED", "timestamp": 1000, "creatorId": 3},
        {"type": "WARD_PLACED", "timestamp": 1100, "creatorId": 8},
        {"type": "WARD_PLACED", "timestamp": 1200, "creatorId": 9},
        {"type": "WARD_KILL", "timestamp": 2000, "killerId": 7},
        {"type": "BUILDING_KILL", "timestamp": 3000, "killerId": 0, "teamId": 200},   # red tower down -> blue
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 4000, "killerId": 0, "teamId": 100},  # blue plate -> red
        {"type": "ELITE_MONSTER_KILL", "timestamp": 5000, "killerId": 9, "killerTeamId": 200},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 5500, "killerId": 2, "killerTeamId": 100},
        {"type": "WARD_PLACED", "timestamp": 6000, "creatorId": 0},
    ]
    for e in events:
        if e["type"] != "WARD_PLACED":
            assert CK.event_side(e, TM) == L._label_event_team_sign(e, TM)
    idx = CK.EventIndex(events, TM)
    assert idx.side_balance("WARD_PLACED", 0, 7000) == 1 - 2
    assert idx.side_balance("WARD_KILL", 0, 7000) == -1
    assert idx.side_balance("BUILDING_KILL", 0, 7000) == 1 and idx.side_balance("TURRET_PLATE_DESTROYED", 0, 7000) == -1
    assert idx.side_balance("ELITE_MONSTER_KILL", 0, 5500) == -1 and idx.side_balance("ELITE_MONSTER_KILL", 0, 5501) == 0
    assert idx.side_balance("WARD_PLACED", 1100, 1200) == -1 and idx.side_balance("WARD_PLACED", 7000, 9000) == 0
    ev = CK.window_events(idx, 1000, 4000, 2000, 0)
    assert ev["ev_in_ward_placed_bmr"] == -1 and ev["ev_in_ward_placed_part"] == 0 and ev["ev_in_ward_placed_map"] == 3
    assert ev["ev_after_elite_bmr"] == 0 and ev["ev_after_elite_map"] == 2


def _consistent_damage(node, minute_rate):
    """Cumulative damage columns that satisfy the data checks: all physical, total >= to-champions."""
    ix = NODE_IDX
    frames = np.arange(node.shape[0], dtype=float)[:, None]
    rate = np.broadcast_to(np.asarray(minute_rate, dtype=float), (node.shape[0], node.shape[1]))
    node[:, :, ix["ds_totalDamageDoneToChampions"]] = frames * rate / DS_DENOM["totalDamageDoneToChampions"]
    node[:, :, ix["ds_physicalDamageDoneToChampions"]] = frames * rate / DS_DENOM["physicalDamageDoneToChampions"]
    node[:, :, ix["ds_totalDamageDone"]] = frames * (rate + 1000.0) / DS_DENOM["totalDamageDone"]


def test_node_columns_resolved_by_name_and_physio_rates():
    cols = CK.verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    assert cols["damage"]["index"] == list(NODE_FEATURE_NAMES).index("ds_totalDamageDoneToChampions")
    assert cols["damage"]["denominator"] == DS_DENOM["totalDamageDoneToChampions"]
    assert set(cols["damage_check_columns"]) == set(CK.DAMAGE_COMPONENTS) | {CK.DAMAGE_ALL}
    bad = dict(NODE_IDX)
    bad["hp_pct"] = bad["alive"]
    with pytest.raises(ValueError):
        CK.verify_node_columns(bad, NODE_FEATURE_NAMES, DS_DENOM)

    minute_ts = np.array([0, 60000, 120000, 180000], dtype=np.int64)
    node = np.zeros((4, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    d, h, a = cols["damage"]["index"], cols["hp"]["index"], cols["alive"]["index"]
    node[:, :, a] = 1.0
    node[:, 0, d] = np.array([0, 1000, 4000, 4000]) / cols["damage"]["denominator"]
    node[:, 1, d] = np.array([0, 0, 2000, 2000]) / cols["damage"]["denominator"]
    node[:, 0, h] = [1.0, 1.0, 0.4, 0.4]
    node[:, 1, h] = [1.0, 0.9, 0.5, 0.5]
    node[2, 1, a] = 0.0
    members = np.zeros(10, dtype=bool)
    members[[0, 1]] = True
    out = CK.window_physio(node, minute_ts, 70000, 110000, members, cols)
    assert out["bracket_min"] == 1.0
    assert out["dmg_pm"] == pytest.approx(2500.0, rel=1e-5) and out["dmg_pm_sum"] == pytest.approx(5000.0, rel=1e-5)
    assert out["hp_pm"] == pytest.approx(-60.0, rel=1e-5) and out["n_hp"] == 1
    assert np.isnan(out["dmg_pm_bmr"]) and np.isnan(out["hp_pm_bmr"])


def test_window_physio_side_resolved():
    cols = CK.verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    d, h, a = cols["damage"]["index"], cols["hp"]["index"], cols["alive"]["index"]
    minute_ts = np.array([0, 60000, 120000], dtype=np.int64)
    node = np.zeros((3, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node[:, :, a] = 1.0
    node[:, :, h] = 1.0
    node[1, 0, d] = 900 / cols["damage"]["denominator"]   # blue 1: 900 / min
    node[1, 1, d] = 300 / cols["damage"]["denominator"]   # blue 2: 300 / min
    node[1, 5, d] = 200 / cols["damage"]["denominator"]   # red 6: 200 / min
    node[1, 5, h] = 0.5                                   # red 6 lost 50 hp%
    node[1, 1, h] = 0.9                                   # blue 2 lost 10 hp%
    members = np.zeros(10, dtype=bool)
    members[[0, 1, 5]] = True
    is_blue = np.zeros(10, dtype=bool)
    is_blue[:5] = True
    out = CK.window_physio(node, minute_ts, 10000, 40000, members, cols, is_blue=is_blue)
    assert out["dmg_pm_bmr"] == pytest.approx(600.0 - 200.0, rel=1e-5)
    assert out["hp_pm_bmr"] == pytest.approx(-5.0 - (-50.0), rel=1e-4)
    only_blue = np.zeros(10, dtype=bool)
    only_blue[[0, 1]] = True
    assert np.isnan(CK.window_physio(node, minute_ts, 10000, 40000, only_blue, cols, is_blue=is_blue)["hp_pm_bmr"])


def test_node_value_checks_pass_on_consistent_columns_and_catch_swaps():
    cols = CK.verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    node = np.zeros((6, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    node[:, :, NODE_IDX["hp_pct"]] = 0.8
    node[3, 2, NODE_IDX["alive"]] = 0.0
    node[3, 2, NODE_IDX["hp_pct"]] = 0.0
    _consistent_damage(node, 350.0)
    stats = CK.new_stats()
    CK.node_value_checks(node, cols, stats)
    assert stats["node_cells_checked"] == 60 and stats["damage_component_cells"] == 60
    for k in ("hp_pct_out_of_range_cells", "alive_nonbinary_cells", "alive_hp_inconsistent_cells",
              "damage_frame_deltas_negative", "damage_component_sum_mismatch_cells", "damage_to_champions_exceeds_total_cells"):
        assert stats[k] == 0, k
    swapped = json.loads(json.dumps(cols))
    swapped["hp"]["index"], swapped["alive"]["index"] = cols["alive"]["index"], cols["hp"]["index"]
    swapped["damage"]["index"] = NODE_IDX["ds_totalDamageDone"]
    stats = CK.new_stats()
    CK.node_value_checks(node, swapped, stats)
    assert stats["alive_nonbinary_cells"] > 0
    assert stats["damage_component_sum_mismatch_cells"] > 0
    short_total = node.copy()
    short_total[:, :, NODE_IDX["ds_totalDamageDone"]] = 0.0
    stats = CK.new_stats()
    CK.node_value_checks(short_total, cols, stats)
    assert stats["damage_to_champions_exceeds_total_cells"] == 50 and stats["damage_component_sum_mismatch_cells"] == 0
    # Riot rounds each component separately: a total 2 units above the parts is rounding, 6 units is a mismatch
    rounded = node.copy()
    rounded[:, :, NODE_IDX["ds_totalDamageDoneToChampions"]] += np.float32(2.0 / DS_DENOM["totalDamageDoneToChampions"])
    stats = CK.new_stats()
    CK.node_value_checks(rounded, cols, stats)
    assert stats["damage_component_rounding_gap_cells"] == 60 and stats["damage_component_sum_mismatch_cells"] == 0
    off = node.copy()
    off[:, :, NODE_IDX["ds_totalDamageDoneToChampions"]] += np.float32(6.0 / DS_DENOM["totalDamageDoneToChampions"])
    stats = CK.new_stats()
    CK.node_value_checks(off, cols, stats)
    assert stats["damage_component_sum_mismatch_cells"] == 60


# ------------------------------------------------------------------ market attribution
def _pack(events):
    n_frames = 3
    node = np.zeros((n_frames, 10, len(NODE_IDX)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    return {"minute_ts": np.array([0, 60000, 120000], dtype=np.int64), "node_minute": node,
            "global_minute": np.zeros((n_frames, F_GLOBAL), np.float32), "gold_team_minute": np.zeros((n_frames, 2), np.float32),
            "events": events, "events_ts": np.array([e["timestamp"] for e in events], dtype=np.int64)}


def _interp(cache, ms):
    return cache["node_minute"][0], cache["global_minute"][0]


@pytest.fixture
def market_cfg(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    monkeypatch.setattr(cfg, "LABEL_GOLD_DEADZONE", 300.0)
    monkeypatch.setattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 0.0, raising=False)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 1.0, raising=False)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", {}, raising=False)
    monkeypatch.setattr(L, "_event_price_table", lambda: dict(TABLE))


def _market(pack, anchor, attribution, numbers=1, leader=-1, hp_leader=0, dmg_leader=0):
    return CK.market_outcome(pack, TM, 10000, 40000, anchor, attribution, interp=_interp,
                             numbers_side=numbers, gold_leader=leader, node_idx=NODE_IDX, hp_leader=hp_leader,
                             dmg_leader=dmg_leader)


STRUCTURES = [{"type": "TURRET_PLATE_DESTROYED", "timestamp": 15000, "killerId": 0, "teamId": 200, "position": {"x": 5200, "y": 5000}},
              {"type": "BUILDING_KILL", "timestamp": 20000, "killerId": 2, "teamId": 200, "buildingType": "TOWER_BUILDING",
               "towerType": "OUTER_TURRET", "position": {"x": 5000, "y": 5000}}]


def test_market_outcome_priced_structures_name_the_winner(market_cfg):
    ev = STRUCTURES
    pack = _pack(ev)
    f, ok = _market(pack, (5000.0, 5000.0), "engagement", hp_leader=-1, dmg_leader=1)
    assert ok
    assert f["priced_any"] == 1 and f["priced_nonward_any"] == 1 and f["n_priced_nonkill"] == 2 and f["gd"] == pytest.approx(660.0)
    assert f["tier_market_gold"] == 1 and f["decided"] == 1 and f["verdict_blue"] == 1 and f["verdict_unpriced"] == 0
    assert (f["fav_blue"], f["fav_numbers"], f["fav_gold_leader"], f["fav_hp_leader"]) == (1.0, 1.0, 0.0, 0.0)
    assert f["fav_dmg_leader"] == 1.0 and set(f) == set(CK.MARKET_FIELDS)
    assert np.isnan(_market(pack, (5000.0, 5000.0), "engagement")[0]["fav_dmg_leader"])  # no damage leader: undefined
    direct = L._compute_label_market_event(L.attribute_events(ev, (5000.0, 5000.0), "engagement"), TM, pack, 10000, 40000,
                                           interp_node_global=_interp, first_kill_ts=None, last_kill_ts=None)
    assert direct == 1

    far, ok = _market(pack, (15000.0, 15000.0), "engagement")
    assert ok and far["n_events"] == 0 and far["priced_any"] == 0 and far["tier_draw"] == 1 and far["verdict_defined"] == 0
    assert np.isnan(far["fav_blue"]) and np.isnan(far["verdict_blue"]) and np.isnan(far["fav_hp_leader"])
    win, ok = _market(pack, (15000.0, 15000.0), "window")
    assert ok and win["priced_any"] == 1 and win["verdict_blue"] == 1
    none, ok = _market(pack, None, "engagement")
    assert ok and np.isnan(none["priced_any"])


def test_market_outcome_follows_the_label_price_variant(market_cfg, monkeypatch):
    if not hasattr(L, "_apply_price_table_variant"):
        pytest.skip("labels.py has no price-table variant")
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 0.25, raising=False)
    pack = _pack(STRUCTURES)
    f, ok = _market(pack, (5000.0, 5000.0), "engagement")
    assert ok  # the reproduced tier agrees with the label under the scaled table
    assert f["gd"] == pytest.approx(165.0) and f["decided"] == 0 and f["tier_refine_structures"] == 1 and f["verdict_blue"] == 1
    assert CK.label_price_table(L)["plates"] == pytest.approx(30.0) and CK.label_price_table(L)["kills"] == 20.0


def test_market_outcome_unpriced_events_fall_to_refinement(market_cfg):
    dragon = [{"type": "ELITE_MONSTER_KILL", "timestamp": 20000, "killerId": 7, "killerTeamId": 200, "monsterType": "DRAGON",
               "monsterSubType": "FIRE_DRAGON", "position": {"x": 9800, "y": 4400}}]
    f, ok = _market(_pack(dragon), (9800.0, 4400.0), "engagement")
    assert ok and f["table_type_any"] == 1 and f["priced_any"] == 0 and f["tier_refine_structures"] == 1 and f["verdict_blue"] == 0
    assert f["verdict_defined"] == 1 and f["verdict_unpriced"] == 1  # no priced event, yet the label names a winner

    ward = [{"type": "WARD_KILL", "timestamp": 20000, "killerId": 1, "wardType": "YELLOW_TRINKET"}]
    f, ok = _market(_pack(ward), (5000.0, 5000.0), "engagement")
    assert ok and f["n_events"] == 0 and f["tier_draw"] == 1 and f["verdict_unpriced"] == 0
    f, ok = _market(_pack(ward), (5000.0, 5000.0), "window")
    assert ok and f["priced_any"] == 1 and f["priced_nonward_any"] == 0 and f["decided"] == 0
    assert f["tier_draw"] == 1 and f["verdict_defined"] == 0 and f["verdict_unpriced"] == 0

    pack = _pack([])
    pack["node_minute"][:, 9, NODE_IDX["alive"]] = 0.0
    f, ok = _market(pack, (5000.0, 5000.0), "engagement")
    assert ok and f["tier_refine_alive"] == 1 and f["verdict_blue"] == 1 and f["priced_any"] == 0
    assert f["verdict_unpriced"] == 1


def test_price_table_provenance_compares_pooled_values(tmp_path):
    table = tmp_path / "prices.json"
    table.write_text(json.dumps({"pooled": {"plates": 120.0, "kills": 20.0}, "note": "x"}), encoding="utf-8")
    out = CK.price_table_provenance(str(table), {"label": {"price_table_sha1": "nope"}, "git_commit": "abc"},
                                    {"plates": 120.0, "kills": 20.0})
    assert out["label_table_equals_file_pooled"] is True and out["bytes_sha1_matches_manifest"] is False
    assert out["pooled_matches_corpus_build_commit"] is None  # absolute path: no git lookup
    assert CK.price_table_provenance(str(table), {}, {"plates": 1.0})["label_table_equals_file_pooled"] is False


# ------------------------------------------------------------------ bootstrap
def test_cell_stats_pooled_mean_and_joint_resampling():
    rng = np.random.default_rng(3)
    n_matches = 300
    mi = np.repeat(np.arange(n_matches), 4)
    effect = rng.normal(0, 1, n_matches)[mi]
    enc = effect + 1.0 + rng.normal(0, 0.1, len(mi))
    base = effect + rng.normal(0, 0.1, len(mi))
    w = KV.cluster_weights(n_matches, 400, 7)
    assert np.all(w.sum(axis=1) == n_matches)
    vals = np.column_stack([enc, base, enc - base])
    vals[5, 0] = np.nan
    point, reps, n, nm = CK.cell_stats(w, mi, vals, np.ones(len(mi), dtype=bool), n_matches)
    assert point[1] == pytest.approx(base.mean())
    assert n[0] == len(mi) - 1 and nm[0] == n_matches
    lo, hi = KV._ci(reps[:, 2])
    assert lo < 1.0 < hi
    lo0, hi0 = KV._ci(reps[:, 0])
    assert (hi - lo) < 0.25 * (hi0 - lo0)


# ------------------------------------------------------------------ one synthetic match end to end
def _synthetic_match():
    minutes = 21
    minute_ts = 60000 * np.arange(minutes, dtype=np.int64)
    angle = 2 * np.pi * np.arange(10) / 10
    ring = np.stack([7500 + 6000 * np.cos(angle), 7500 + 6000 * np.sin(angle)], axis=1)
    xy = np.repeat(ring[None], minutes, axis=0)
    for m in (12, 16):
        xy[m] = 7500 + 150 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    node = np.zeros((minutes, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    node[:, :, NODE_IDX["hp_pct"]] = 1.0
    _consistent_damage(node, 300.0)
    kill = {"type": "CHAMPION_KILL", "timestamp": 16 * 60000 + 500, "killerId": 1, "victimId": 6, "bounty": 300,
            "shutdownBounty": 0, "assistingParticipantIds": [2], "position": {"x": 7500, "y": 7500}}
    gold = np.stack([1000.0 + 100 * np.arange(minutes), 900.0 + 100 * np.arange(minutes)], axis=1)
    return {"minute_ts": minute_ts, "node_minute": node, "global_minute": np.zeros((minutes, F_GLOBAL), np.float32),
            "gold_team_minute": gold.astype(np.float32), "events": [kill], "events_ts": np.array([kill["timestamp"]]),
            "xy_raw_minute": xy.astype(np.float32),
            "meta": {"team_map": dict(TM), "anchor_is_norm": False, "patch": "15.14",
                     "feature_version": str(getattr(cfg, "FEATURE_VERSION", ""))}}


@pytest.mark.parametrize("exclude", [None, 2])
def test_characterize_match_end_to_end(monkeypatch, exclude):
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    from gameplay.pipeline_interp import interpolate_node_global

    pack = _synthetic_match()
    gate = KV.resolve_gate("v3.3", min_per_team=4)
    prep = KV.prepare_match(pack, gate["radius_u"], cfg, NODE_IDX, track_diff=True)
    assert prep["xy_source"] == "xy_raw_minute" and not prep["is_norm"]
    # a kill 0.5 s after a minute frame: the adjustment starts at that frame's own position, so no cell moves
    assert prep["kt_cells"].shape == prep["xy_dense"].shape[:2] and not prep["kt_cells"].any()
    found = KV.scan_prepared(prep, gate)
    assert [e["has_kill"] for e in found] == [False, True]
    cols = CK.verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    stats = CK.new_stats()
    rows = CK.characterize_match(pack, prep, found, gate, gate["grace_ms"], cols, interpolate_node_global, NODE_IDX, 0,
                                 stats, baseline_exclude_min_per_team=exclude,
                                 feature_version=str(getattr(cfg, "FEATURE_VERSION", "")))
    assert rows.shape == (2, len(CK.COLUMNS))
    nonzero = {k: v for k, v in stats.items() if v and k not in ("node_cells_checked", "damage_frame_deltas", "damage_component_cells")}
    assert nonzero == {}
    kl, wk = rows[0], rows[1]
    col = CK.COL
    assert kl[col["has_kill"]] == 0 and wk[col["has_kill"]] == 1
    assert kl[col["band"]] == 1 and kl[col["n_members"]] == 10 and kl[col["window_kills_map"]] == 0
    assert kl[col["base_found"]] == 1 and CK.band_index(kl[col["base_start_ms"]]) == 1
    assert kl[col["dmg_pm"]] == pytest.approx(300.0, rel=1e-4) and kl[col["base_dmg_pm"]] == pytest.approx(300.0, rel=1e-4)
    assert kl[col["dmg_pm_bmr"]] == pytest.approx(0.0, abs=1e-3) and kl[col["hp_pm_bmr"]] == pytest.approx(0.0, abs=1e-6)
    assert kl[col["mk_eng_priced_any"]] == 0 and kl[col["mk_eng_tier_draw"]] == 1 and kl[col["no_anchor"]] == 0
    assert wk[col["mk_eng_kills"]] == 1 and wk[col["mk_eng_priced_any"]] == 1 and wk[col["mk_eng_verdict_blue"]] == 1
    assert kl[col["gold_side"]] == 1
    assert kl[col["kt_touch"]] == 0 and wk[col["kt_member_share"]] == 0
    assert kl[col["mk_eng_verdict_unpriced"]] == 0

    summary = CK.summarize(rows, 1, 20, 7)
    assert summary["summary"]["killless"]["10-20"]["metrics"]["dmg_pm"]["n_valid_reps"] == 20
    assert summary["summary"]["killless"]["10-20"]["n_windows"] == 1
    assert summary["band_distribution"]["with_kill"]["counts"]["10-20"] == 1
    assert summary["band_distribution"]["killless"]["baseline_offset_s"]["10-20"]["n"] == 1
    head = CK.headline(summary)
    assert head["eng"]["10-20"]["n_windows"] == 1 and head["eng"]["10-20"]["unmeasurable_share"] == 1.0
    assert head["eng"]["10-20"]["no_verdict_share"] == 1.0 and head["eng"]["10-20"]["no_verdict_share_n_valid_reps"] == 20
    assert "fav_dmg_leader" in head["eng"]["10-20"] and "verdict_unpriced" in head["eng"]["10-20"]
    assert "no_verdict_share" in head["note"] and "always name the attribution" in head["note"]
    phys = CK.headline_physio_events(summary)
    cell = phys["killless"]["10-20"]["dmg_pm"]
    assert phys["killless"]["10-20"]["n_pairs"] == 1
    assert cell["encounter"]["mean"] == pytest.approx(300.0, rel=1e-4) and cell["baseline"]["mean"] == pytest.approx(300.0, rel=1e-4)
    assert cell["paired_diff"]["mean"] == pytest.approx(0.0, abs=1e-2)
    assert cell["killless_minus_with_kill"]["diff"] == pytest.approx(0.0, abs=1e-2)
    assert "killless_minus_with_kill" not in phys["with_kill"]["10-20"]["dmg_pm"]
    assert set(phys["columns"]) <= set(CK.WINDOW_COLS)
    assert phys["killless"]["10-20"]["kt_touch"]["encounter"]["mean"] == 0.0
    assert "killless_minus_with_kill" in phys["killless"]["10-20"]["kt_member_share"]
    # the with-kill window holds a blue kill; no ward / structure events in this synthetic match
    assert wk[col["ev_in_ward_placed_bmr"]] == 0 and kl[col["ev_after_building_map"]] == 0

    wrong = dict(pack)
    wrong["node_minute"] = pack["node_minute"][:, :, :-1]
    stats = CK.new_stats()
    assert CK.characterize_match(wrong, prep, found, gate, gate["grace_ms"], cols, interpolate_node_global, NODE_IDX, 0,
                                 stats).shape == (0, len(CK.COLUMNS))
    assert stats["node_layout_mismatch_matches"] == 1


# ------------------------------------------------------------------ the 5 s track: kill-adjusted vs frame-only
def _late_kill_match(offset_s):
    pack = _synthetic_match()
    pack["events"][0]["timestamp"] = 16 * 60000 + int(offset_s * 1000)
    pack["events_ts"] = np.array([pack["events"][0]["timestamp"]])
    return pack


def test_kill_trajectory_track_moves_kill_participants_and_can_flip_the_kill_test():
    gate = KV.resolve_gate("v3.3")
    pack = _late_kill_match(25)  # kill at 16:25, participants 1 (killer), 6 (victim), 2 (assist)
    on = KV.prepare_match(pack, gate["radius_u"], cfg, NODE_IDX, track_diff=True)
    assert on["kill_trajectory_interp"] is True
    frames, slots = np.nonzero(on["kt_cells"])
    assert sorted(set(slots.tolist())) == [0, 1, 5]
    assert set(on["dense_ts"][frames].tolist()) == {965000, 970000, 975000, 980000, 985000}
    from gameplay.fight_clustering import build_5s_position_grid
    from gameplay.fights import _extract_kill_events
    xy = pack["xy_raw_minute"]
    kills = _extract_kill_events(pack["events"])
    _, frame_only = build_5s_position_grid(xy, pack["minute_ts"], kills, pack["meta"]["team_map"], cfg_obj=KV.TrackCfg(cfg, False))
    _, adjusted = build_5s_position_grid(xy, pack["minute_ts"], kills, pack["meta"]["team_map"], cfg_obj=cfg)
    assert np.array_equal(on["xy_dense"], adjusted) and np.array_equal(on["xy_dense_alt"], frame_only)
    off = KV.prepare_match(pack, gate["radius_u"], KV.TrackCfg(cfg, False), NODE_IDX, track_diff=True)
    assert off["kill_trajectory_interp"] is False and np.array_equal(off["xy_dense"], frame_only)
    assert np.array_equal(off["kt_cells"], on["kt_cells"])

    enc_on = KV.scan_prepared(on, gate)
    enc_off = KV.scan_prepared(on, gate, alt=True)
    # the kill-adjusted track lengthens the run by one frame, which brings the kill inside +-15 s
    assert [(e["start_ms"], e["end_ms"], e["has_kill"]) for e in enc_on][1:] == [(955000, 970000, True)]
    assert [(e["start_ms"], e["end_ms"], e["has_kill"]) for e in enc_off][1:] == [(955000, 965000, False)]
    assert KV.scan_prepared(off, KV.resolve_gate("v3.3", kill_trajectory_interp=False)) == enc_off

    kt_frames = on["kt_cells"].any(axis=1)
    primary, alt = KV.ScanTally(), KV.ScanTally()
    primary.add(1, "M", "15.14", enc_on, kt_frames, on["dense_ts"])
    alt.add(1, "M", "15.14", enc_off, kt_frames, on["dense_ts"])
    primary.add(2, "N", "15.14", [], None, None)
    alt.add(2, "N", "15.14", [], None, None)
    assert primary.kt_enc == [1, 0] and primary.kt_kl == [0, 0] and primary.kt_missing == 1
    assert alt.kt_enc == [1, 0] and alt.kt_kl == [1, 0]
    d = KV.denominator_block(np.array(primary.n_enc), np.array(primary.n_kl), primary.durations_killless, n_boot=20,
                             seed=1, kt_enc=np.array(primary.kt_enc), kt_kl=np.array(primary.kt_kl),
                             kt_missing=primary.kt_missing)
    kt = d["kill_adjusted_track"]
    assert kt["with_kill_touching"] == 1 and kt["with_kill_touching_share"] == 1.0 and kt["killless_touching"] == 0
    assert kt["matches_without_track_diff"] == 1 and "with_kill_touching_share_ci" in kt

    w = KV.cluster_weights(2, 30, 5)
    sens = KV.track_sensitivity_block(primary, alt, True, w)
    assert sens["kill-adjusted"]["killless"] == 1 and sens["frame-only"]["killless"] == 2
    diff = sens["frame_only_minus_kill_adjusted"]
    assert diff["killless_share_of_encounters"]["diff"] == pytest.approx(1.0 - 0.5)
    assert diff["killless_per_match"]["diff"] == pytest.approx(0.5) and diff["with_kill_per_match"]["diff"] == pytest.approx(-0.5)
    assert diff["encounters_per_match"]["diff"] == 0.0 and "ci95" in diff["killless_share_of_encounters"]
    assert KV.track_sensitivity_block(alt, primary, False, None)["primary_track"] == "frame-only"
    bad = KV.ScanTally()
    bad.add(1, "X", "15.14", [])
    with pytest.raises(ValueError):
        KV.track_sensitivity_block(primary, bad, True, None)
    q = KV.quote(gate, KV.denominator_block(np.array(primary.n_enc), np.array(primary.n_kl), [], n_boot=0),
                 {"published_corpus": None, "same_sample_corpus": None}, sens)
    assert "On the frame-only track of the same matches: 100.00 %" in q and "frame-only minus kill-adjusted share" in q


def test_iter_scanned_refuses_a_gate_on_another_track():
    gate_off = KV.resolve_gate("v3.3", kill_trajectory_interp=False)
    with pytest.raises(ValueError):
        next(KV.iter_scanned([], gate_off, KV.TrackCfg(cfg, True), NODE_IDX, KV.ScanTally()))
    assert list(KV.iter_scanned([], gate_off, KV.TrackCfg(cfg, False), NODE_IDX, KV.ScanTally())) == []


@pytest.mark.parametrize("kt_on", [True, False])
def test_characterize_on_both_tracks_reports_the_adjustment(monkeypatch, kt_on):
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    from gameplay.pipeline_interp import interpolate_node_global

    pack = _late_kill_match(25)
    gate = KV.resolve_gate("v3.3", kill_trajectory_interp=kt_on)
    track_cfg = KV.TrackCfg(cfg, kt_on)
    prep = KV.prepare_match(pack, gate["radius_u"], track_cfg, NODE_IDX, track_diff=True)
    found = KV.scan_prepared(prep, gate)
    cols = CK.verify_node_columns(NODE_IDX, NODE_FEATURE_NAMES, DS_DENOM)
    stats = CK.new_stats()
    rows = CK.characterize_match(pack, prep, found, gate, gate["grace_ms"], cols, interpolate_node_global, NODE_IDX, 0,
                                 stats, baseline_exclude_min_per_team=2)
    assert stats["scanner_disagreement_matches"] == 0 and stats["track_diff_missing_matches"] == 0
    last = rows[-1]
    col = CK.COL
    assert last[col["has_kill"]] == float(kt_on)
    assert last[col["kt_touch"]] == 1.0
    # kill-adjusted run 955-970 s: frames 965 and 970 move a member (2 of 4); frame-only run 955-965 s: frame 965 (1 of 3)
    assert last[col["kt_member_share"]] == pytest.approx(2 / 4 if kt_on else 1 / 3)
    assert rows[0][col["kt_touch"]] == 0.0
