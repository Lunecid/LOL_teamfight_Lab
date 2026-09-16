"""Evidence-based state: causality, event replay, and the formulas the reconstruction rests on."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import NODE_FEATURE_NAMES  # noqa: E402
from gameplay.evidence_state import (  # noqa: E402
    DEN, EvidenceStateBuilder, FOUNTAIN, LEGACY_POS_ERROR_CURVE, POS_ERROR_CURVE, XP_AT_LEVEL,
    default_pos_error_calibration_path, level_growth, load_pos_error_curve, match_ids_sha256, respawn_seconds,
)
from scripts.calibrate_position_error_v3 import (  # noqa: E402
    corpus_patch_matches, curve_disjointness_record, fit_monotone, ids_sha256, select_calibration_matches,
)
from scripts.evidence_block_patch_holdout_v33 import (  # noqa: E402
    RankedAUC, joint_paired_bootstrap, overlap_with_patches,
)

IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
ITEMS = {1001: {"gold_total": 300, "gold_base": 300, "gold_sell": 210, "from": [], "into": [3006],
                "stats": {"movementSpeed": 25.0}},
         1036: {"gold_total": 350, "gold_base": 350, "gold_sell": 245, "from": [], "into": [3134],
                "stats": {"attackDamage": 10.0}},
         3134: {"gold_total": 1100, "gold_base": 400, "gold_sell": 770, "from": [1036, 1036], "into": [],
                "stats": {"attackDamage": 30.0}},
         2003: {"gold_total": 50, "gold_base": 50, "gold_sell": 20, "from": [], "into": [], "stats": {}}}
CHAMPS = {266: {"name": "Aatrox", "base": {"attackDamage": 60.0, "attackSpeed": 0.651, "healthMax": 650.0},
                "growth": {"attackDamage": 5.0, "attackSpeed_pct": 2.5, "healthMax": 114.0}}}


def _pack(events, n_frames=4):
    ts = np.array([120_000 + 60_000 * k for k in range(n_frames)], dtype=np.int64)
    node = np.zeros((n_frames, 10, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    xy = np.zeros((n_frames, 10, 2))
    for k in range(n_frames):
        for p in range(10):
            node[k, p, IDX["level_norm"]] = 3 / DEN["level"]
            node[k, p, IDX["xp_norm"]] = (700 + 200 * k) / DEN["xp"]
            node[k, p, IDX["totalGold_norm"]] = (1000 + 300 * k) / DEN["tot_g"]
            node[k, p, IDX["curGold_norm"]] = (500 + 300 * k) / DEN["cur_g"]
            node[k, p, IDX["cs_attackDamage"]] = 70.0 / 600.0
            node[k, p, IDX["cs_attackSpeed"]] = 0.7 / 3.0
            node[k, p, IDX["cs_healthMax"]] = 900.0 / 8500.0
            node[k, p, IDX["hp_pct"]] = 0.8
            node[k, p, IDX["alive"]] = 1.0
            xy[k, p] = (4000.0 + 100 * k, 7000.0)
    return {"minute_ts": ts, "node_minute": node, "xy_raw_minute": xy, "events": events,
            "meta": {"patch": "15.15", "team_map": {str(i): (100 if i <= 5 else 200) for i in range(1, 11)},
                     "static_meta": {"champion_by_pid": {"1": 266}}}}


def builder(events, curve="legacy", **kw):
    # the expectations below encode the legacy (pilot v1) curve's numbers, so pin it explicitly
    return EvidenceStateBuilder(_pack(events, **kw), tables=(ITEMS, CHAMPS), pos_error_curve=curve)


def test_never_reads_a_frame_after_tau():
    b = builder([])
    assert b.frame_index(179_999) == 0 and b.frame_index(180_000) == 1
    with pytest.raises(ValueError):
        b.frame_index(100)


def test_level_up_and_ultimate_come_from_events_not_the_frame():
    b = builder([{"type": "LEVEL_UP", "timestamp": 190_000, "participantId": 1, "level": 4},
                 {"type": "LEVEL_UP", "timestamp": 195_000, "participantId": 1, "level": 5},
                 {"type": "SKILL_LEVEL_UP", "timestamp": 196_000, "participantId": 1, "skillSlot": 4, "levelUpType": "NORMAL"}])
    s = b.at(200_000)[1]                                   # frame at 180s says level 3
    assert s["level_norm"] == pytest.approx(5 / 18)
    assert s["ult_level_norm"] == pytest.approx(1 / 3)
    assert s["xp_norm"] * DEN["xp"] >= XP_AT_LEVEL[5] - 1e-6   # a level-up proves at least that much xp
    assert b.at(189_000)[1]["level_norm"] == pytest.approx(3 / 18)


def test_inventory_replay_handles_buy_combine_sell_and_undo():
    ev = [{"type": "ITEM_PURCHASED", "timestamp": 185_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_PURCHASED", "timestamp": 186_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_DESTROYED", "timestamp": 190_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_DESTROYED", "timestamp": 190_000, "participantId": 1, "itemId": 1036},
          {"type": "ITEM_PURCHASED", "timestamp": 190_000, "participantId": 1, "itemId": 3134},
          {"type": "ITEM_PURCHASED", "timestamp": 192_000, "participantId": 1, "itemId": 2003},
          {"type": "ITEM_UNDO", "timestamp": 193_000, "participantId": 1, "beforeId": 2003, "afterId": 0, "goldGain": 50}]
    b = builder(ev)
    inv, gold_delta, stat_delta = b._replay_inventory(1, 200_000, since=180_000)
    assert inv == [3134]
    # paid 350 + 350 + 400 (recipe, components destroyed at the same timestamp) + 50, undo refunded 50
    assert gold_delta == pytest.approx(-(350 + 350 + 400 + 50) + 50)
    assert stat_delta["attackDamage"] == pytest.approx(30.0)   # 10 + 10 - 10 - 10 + 30
    s = b.at(200_000)[1]
    assert s["cs_attackDamage"] * 600 == pytest.approx(70.0 + 30.0)
    assert s["item_count"] == 1.0


def test_level_growth_formula_and_attack_speed_growth():
    assert level_growth(1) == 0.0
    assert level_growth(2) == pytest.approx(0.72)            # (L-1) * (0.7025 + 0.0175 * (L-1))
    assert level_growth(18) == pytest.approx(17.0)           # the growth multiplier reaches exactly 17 at 18
    b = builder([{"type": "LEVEL_UP", "timestamp": 190_000, "participantId": 1, "level": 4}])
    s = b.at(200_000)[1]
    dg = level_growth(4) - level_growth(3)
    assert s["cs_attackDamage"] * 600 == pytest.approx(70.0 + 5.0 * dg)
    assert s["cs_attackSpeed"] * 3 == pytest.approx(0.7 + 0.651 * 0.025 * dg)
    assert s["cs_healthMax"] * 8500 == pytest.approx(900.0 + 114.0 * dg)


def test_death_respawn_and_hp_reset():
    ev = [{"type": "CHAMPION_KILL", "timestamp": 185_000, "victimId": 1, "killerId": 6,
           "position": {"x": 9000, "y": 9000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)                                          # no LEVEL_UP events: the frame's level 3 is the bound
    timer = respawn_seconds(3, 185_000)
    assert timer == 12                                      # level 3, before minute 25: base only
    during = b.at(185_000 + int(timer * 1000) - 500)[1]
    after = b.at(185_000 + int(timer * 1000) + 500)[1]
    assert during["alive"] == 0.0 and during["hp_pct"] == 0.0 and during["respawn_in_s"] > 0
    assert after["alive"] == 1.0 and after["hp_pct"] == 1.0 and after["mp_pct"] == 1.0
    killer = b.at(190_000)[6]
    assert killer["curGold_norm"] * DEN["cur_g"] >= 500 + 300      # bounty landed at 185s
    assert respawn_seconds(18, 40 * 60_000) > respawn_seconds(18, 10 * 60_000)


def test_position_uses_the_latest_evidence_and_never_the_engagement_kill():
    ev = [{"type": "CHAMPION_KILL", "timestamp": 185_000, "victimId": 2, "killerId": 1,
           "assistingParticipantIds": [3], "position": {"x": 12000, "y": 3000}, "bounty": 300, "shutdownBounty": 0},
          {"type": "ITEM_PURCHASED", "timestamp": 195_000, "participantId": 4, "itemId": 1001},
          {"type": "CHAMPION_KILL", "timestamp": 230_000, "victimId": 5, "killerId": 7,
           "position": {"x": 1000, "y": 1000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)
    s = b.at(200_000)
    assert (s[1]["x_norm"] * 16000, s[1]["y_norm"] * 16000) == (12000, 3000)   # killer near the kill
    assert s[1]["pos_evidence_kind"] == 2.0 and s[1]["pos_evidence_age_s"] == pytest.approx(15.0)
    assert s[3]["pos_evidence_kind"] == 3.0                                     # assist
    # a shop visit is weak evidence (players leave the fountain at once): the calibrated error curve
    # keeps the 20 s-old frame (3,300 u expected) over a 5 s-old purchase (5,833 u)
    assert s[4]["pos_evidence_kind"] == 0.0
    assert s[4]["x_norm"] * 16000 == pytest.approx(4100.0)
    # victim 2 died at 185s at level 3 (12 s timer): dead at 190s (frame position kept, flagged),
    # respawned at 197s -> a 3 s-old respawn (2,803 u) beats the 20 s-old frame (3,300 u)
    d = b.at(190_000)[2]
    assert d["alive"] == 0.0 and d["pos_evidence_kind"] == 7.0
    assert s[2]["alive"] == 1.0 and s[2]["pos_evidence_kind"] == 6.0
    assert (s[2]["x_norm"] * 16000, s[2]["y_norm"] * 16000) == FOUNTAIN[100]
    # but once the respawn is 15 s old (6,360 u) a 30 s-old frame (3,900 u) wins again
    late = b.at(212_000)[2]
    assert late["pos_evidence_kind"] == 0.0
    assert s[5]["pos_evidence_kind"] == 0.0                                     # kill at 230s is after tau
    assert s[5]["x_norm"] * 16000 == pytest.approx(4100.0)                      # frame at 180s, held
    assert s[5]["pos_evidence_age_s"] == pytest.approx(20.0)


def test_continuous_accumulators_extrapolate_from_previous_frames_only():
    b = builder([])
    s = b.at(210_000)[1]                       # frame at 180s, previous at 120s: 300 gold / 60s = 5 g/s
    assert s["totalGold_norm"] * DEN["tot_g"] == pytest.approx(1300 + 5.0 * 30)
    assert s["xp_norm"] * DEN["xp"] == pytest.approx(900 + (200 / 60) * 30)
    first = b.at(150_000)[1]                   # only one frame available: no rate, plain hold
    assert first["totalGold_norm"] * DEN["tot_g"] == pytest.approx(1000)


def test_baron_buff_is_lost_on_death():
    ev = [{"type": "ELITE_MONSTER_KILL", "timestamp": 185_000, "killerId": 1, "killerTeamId": 100,
           "monsterType": "BARON_NASHOR", "monsterSubType": "", "position": {"x": 5000, "y": 10000}},
          {"type": "CHAMPION_KILL", "timestamp": 200_000, "victimId": 2, "killerId": 6,
           "position": {"x": 6000, "y": 6000}, "bounty": 300, "shutdownBounty": 0}]
    b = builder(ev)
    s = b.at(210_000)
    assert s[1]["has_baron"] == 1.0 and 0 < s[1]["baron_remain_norm"] < 1
    assert s[2]["has_baron"] == 0.0                                            # died after taking it
    assert s[6]["has_baron"] == 0.0                                            # other team
    assert b.at(185_000 + 180_000 + 1)[1]["has_baron"] == 0.0                  # expired


# ---------------------------------------------------------------- position-error calibration (leak fix)

PATCHES = {f"M{i:02d}": ("15.14", "15.15", "15.16")[i % 3] for i in range(60)}
POOL = [f"M{i:02d}" for i in range(30)]                  # training-side split
TEST_SPLIT = [f"M{i:02d}" for i in range(30, 45)]        # an evaluation split (predict_test)
CORPUS_HOLDOUT = {f"corpus_patch_{p}": sorted(m for m, q in PATCHES.items() if q == p) for p in ("15.15", "15.16")}


def _select(pool=POOL, patch_of=PATCHES.get, n=100):
    return select_calibration_matches(pool, n, patch_of=patch_of, holdout_patches=("15.15", "15.16"),
                                      split_evaluation_sets={"predict_test": TEST_SPLIT},
                                      patch_holdout_sets=CORPUS_HOLDOUT)


def test_calibration_matches_never_intersect_evaluation_matches():
    ids, stats, checks = _select()
    assert ids and all(PATCHES[m] == "15.14" for m in ids)
    assert stats["excluded_holdout_patch"] == 20 and len(ids) == 10
    for name, eval_ids in {"predict_test": TEST_SPLIT, **CORPUS_HOLDOUT}.items():
        assert not set(ids) & set(eval_ids)
        assert checks[name]["intersection"] == 0 and checks[name]["n_eval"] == len(set(eval_ids))
    # a pool that overlaps an evaluation split is a split bug: it raises instead of being filtered
    with pytest.raises(ValueError, match="predict_test"):
        _select(pool=POOL + TEST_SPLIT[:1])
    # a hold-out-patch match whose patch lookup lies still cannot get through: the corpus sets are re-checked
    lying = dict(PATCHES, M01="15.14")
    assert PATCHES["M01"] == "15.15"
    with pytest.raises(ValueError, match="corpus_patch_15"):
        _select(patch_of=lying.get)
    # an unverifiable patch is excluded, never assumed to be a training patch
    ids_unknown, stats_unknown, _ = _select(patch_of=lambda m: None if m == "M00" else PATCHES[m])
    assert "M00" not in ids_unknown and stats_unknown["excluded_unknown_patch"] == 1
    # the downstream guards: corpus hold-out patches and a run's own evaluation matches
    groups = np.array(sorted(PATCHES))
    patch = np.array([PATCHES[m] for m in groups])
    assert all(v["intersection"] == 0 for v in overlap_with_patches(set(ids), groups, patch, ["15.15", "15.16"]).values())
    assert overlap_with_patches({"M01"}, groups, patch, ["15.15"])["corpus_patch_15.15"]["intersection"] == 1


def _calibration(tmp_path, name, **over):
    ids = ["M00", "M03"]
    doc = {"version": "position_error_calibration_v2",
           "fitted": {k: [float(v) for v in vals] for k, vals in LEGACY_POS_ERROR_CURVE.items()},
           "match_ids": ids, "match_ids_sha256": ids_sha256(ids),
           "disjointness": {"predict_test": {"intersection": 0}, "value_validation": {"intersection": 0},
                            "corpus_patch_15.16": {"intersection": 0}}}
    doc.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_curve_loader_refuses_a_calibration_that_touched_evaluation_matches(tmp_path):
    assert ids_sha256(["b", "a", "a"]) == match_ids_sha256(["a", "b"])        # one hash formula everywhere
    curve, prov = load_pos_error_curve(_calibration(tmp_path, "good.json"))
    assert curve["frame"] == [750.0, 2200.0, 3300.0, 3900.0, 3950.0, 3950.0] and prov["match_ids_sha256"] == ids_sha256(["M00", "M03"])
    bad_sets = {"predict_test": {"intersection": 1}, "value_validation": {"intersection": 0},
                "corpus_patch_15.16": {"intersection": 0}}
    cases = {"touched.json": {"disjointness": bad_sets},
             "unchecked.json": {"disjointness": {}},
             "no_value_validation.json": {"disjointness": {"predict_test": {"intersection": 0},
                                                           "corpus_patch_15.16": {"intersection": 0}}},
             "no_corpus_patch.json": {"disjointness": {"predict_test": {"intersection": 0},
                                                       "value_validation": {"intersection": 0}}},
             "edited_ids.json": {"match_ids": ["M00", "M03", "M31"]},
             "not_monotone.json": {"fitted": dict(LEGACY_POS_ERROR_CURVE, frame=[750, 2200, 3300, 3900, 3950, 3000])}}
    for name, over in cases.items():
        with pytest.raises(ValueError):
            load_pos_error_curve(_calibration(tmp_path, name, **over))


def test_new_runs_default_to_the_disjoint_calibration_and_never_fall_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("LOL_POS_ERROR_CALIBRATION", str(tmp_path / "absent.json"))
    assert default_pos_error_calibration_path() == tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        EvidenceStateBuilder(_pack([]), tables=(ITEMS, CHAMPS))
    # a calibration under which a fresh shop visit is cheap: the builder must follow the loaded curve
    cheap_shop = dict(LEGACY_POS_ERROR_CURVE, shop=[100, 200, 300, 400, 500, 600])
    path = _calibration(tmp_path, "cheap_shop.json", fitted=cheap_shop)
    monkeypatch.setenv("LOL_POS_ERROR_CALIBRATION", str(path))
    ev = [{"type": "ITEM_PURCHASED", "timestamp": 195_000, "participantId": 4, "itemId": 1001}]
    b = EvidenceStateBuilder(_pack(ev), tables=(ITEMS, CHAMPS))
    assert b.pos_error_provenance["source"] == str(path) and b.pos_error_curve["shop"][0] == 100.0
    assert b.at(200_000)[4]["pos_evidence_kind"] == 5.0                       # shop, not the frame
    assert builder(ev).at(200_000)[4]["pos_evidence_kind"] == 0.0             # legacy: frame
    # the legacy table stays selectable and unchanged, so the v1 pilot remains reproducible
    assert POS_ERROR_CURVE["frame"] == [750, 2200, 3300, 3900, 3950, 3950]
    assert builder([]).pos_error_provenance["source"] == "legacy"


def test_run_guard_checks_the_curve_behind_the_blocks_against_the_runs_evaluation_matches(tmp_path):
    path = _calibration(tmp_path, "run_guard.json")
    _, prov = load_pos_error_curve(path)
    rec = curve_disjointness_record({"train": prov, "test": prov}, {"test_patch": ["M30", "M31"]})
    assert rec["evaluation_calibrated_curve"] is False and rec["disjointness_from_this_run"]["test_patch"]["intersection"] == 0
    with pytest.raises(ValueError, match="intersect"):
        curve_disjointness_record({"train": prov, "test": prov}, {"test_patch": ["M03"]})
    with pytest.raises(ValueError, match="different"):
        curve_disjointness_record({"train": prov, "test": {"source": "legacy"}}, {"test_patch": ["M30"]})
    with pytest.raises(ValueError, match="evaluation matches"):
        curve_disjointness_record({"train": {"source": "legacy"}, "test": {"source": "legacy"}}, {"t": ["M30"]})
    legacy = curve_disjointness_record({"train": None, "test": None}, {"t": ["M30"]}, allow_legacy=True)
    assert legacy["evaluation_calibrated_curve"] is True
    path.write_text(path.read_text(encoding="utf-8").replace("M03", "M04"), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):                          # calibration edited after the build
        curve_disjointness_record({"train": prov, "test": prov}, {"test_patch": ["M30"]})


def test_monotone_fit_is_a_count_weighted_isotonic_regression_with_no_hand_edits():
    means, counts = [749.5, 2226.7, 3332.1, 4478.4, 5007.3, 3944.4], [1000, 800, 600, 300, 150, 58978]
    iso = fit_monotone(means, counts, "isotonic_weighted")
    assert all(b >= a for a, b in zip(iso, iso[1:]))
    assert iso[:3] == [749.5, 2226.7, 3332.1]
    pooled = (4478.4 * 300 + 5007.3 * 150 + 3944.4 * 58978) / (300 + 150 + 58978)
    assert iso[3] == iso[4] == iso[5] == pytest.approx(pooled, abs=0.06)       # the 58,978-row bin dominates
    assert fit_monotone(means, counts, "cummax") == [749.5, 2226.7, 3332.1, 4478.4, 5007.3, 5007.3]
    assert fit_monotone([np.nan, 2000, np.nan, 3000, np.nan, np.nan], [0, 5, 0, 5, 0, 0]) == [2000.0, 2000.0, 2000.0, 3000.0, 3000.0, 3000.0]


def test_weighted_auc_matches_sklearn_and_the_bootstrap_is_paired():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400), 1)                                           # heavy ties
    w = rng.integers(0, 3, 400).astype(float)
    r = RankedAUC(y, p)
    assert r.auc(np.ones(400)) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert r.auc(w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    groups = np.array([f"M{i // 4}" for i in range(400)])
    minute = rng.uniform(2, 40, 400)
    other = np.clip(p + rng.normal(0, .2, 400), 0, 1)
    out, meta = joint_paired_bootstrap(y, {"X": p, "X_plus_evidence": p.copy(), "X_plus_hold": other},
                                       groups, minute, n_boot=40)
    same = out["evidence_minus_X"]["overall"]
    assert same["delta"] == same["lo"] == same["hi"] == 0.0                   # identical models: zero in every draw
    assert out["evidence_minus_hold"]["overall"]["delta"] == pytest.approx(-out["hold_minus_X"]["overall"]["delta"])
    assert meta["auc_implementation_max_abs_diff_vs_sklearn"] < 1e-9 and meta["clusters"] == 100


def test_bootstrap_bands_share_one_match_resample_and_models_get_their_own_cis():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    n = 600
    groups = np.array([f"M{i // 3:03d}" for i in range(n)])
    y = rng.integers(0, 2, n)
    base = np.clip(0.5 * y + rng.normal(0, .6, n), -2, 2)
    preds = {"X": base, "X_plus_hold": base + rng.normal(0, .3, n), "X_plus_evidence": base + rng.normal(0, .1, n)}
    minute = rng.uniform(0, 40, n)
    pick = rng.random(n) < 0.4
    extra = {"participation_pick": pick, "participation_skirmish": ~pick, "tiny": np.arange(n) < 10}
    # one draw: every band's AUCs must use the same match multiplicities (default_rng(seed).integers(M, size=M))
    out, meta, models = joint_paired_bootstrap(y, preds, groups, minute, n_boot=1, seed=11, extra_masks=extra, model_cis=True)
    uniq, inv = np.unique(groups, return_inverse=True)
    w = np.bincount(np.random.default_rng(11).integers(len(uniq), size=len(uniq)), minlength=len(uniq))[inv].astype(float)
    for band, mask in (("overall", np.ones(n, bool)), ("participation_pick", pick), ("10-20", (minute >= 10) & (minute < 20))):
        auc = {k: roc_auc_score(y[mask], p[mask], sample_weight=w[mask]) for k, p in preds.items()}
        assert out["evidence_minus_X"][band]["mean"] == pytest.approx(auc["X_plus_evidence"] - auc["X"], abs=1e-12)
        assert models["X_plus_hold"][band]["mean"] == pytest.approx(auc["X_plus_hold"], abs=1e-12)
        assert models["X"][band]["auc"] == pytest.approx(roc_auc_score(y[mask], preds["X"][mask]), abs=1e-12)
        assert models["X"][band]["n_rows"] == int(mask.sum())
    assert "tiny" not in out["evidence_minus_X"] and "tiny" not in meta["bands"]        # < 50 rows: skipped
    assert "0-2" not in meta["bands"] and meta["bands"][:5] == ["overall", "2-10", "10-20", "20-30", "30-1000"]
    # many draws: percentile interval brackets the draws' mean, and the same seed reproduces it exactly
    out2, _, models2 = joint_paired_bootstrap(y, preds, groups, minute, n_boot=60, extra_masks=extra, model_cis=True)
    again, _, _ = joint_paired_bootstrap(y, preds, groups, minute, n_boot=60, extra_masks=extra, model_cis=True)
    assert out2 == again
    for band, cell in models2["X_plus_evidence"].items():
        assert cell["lo"] <= cell["mean"] <= cell["hi"] and cell["n_boot"] == 60
    with pytest.raises(ValueError, match="taken"):
        joint_paired_bootstrap(y, preds, groups, minute, n_boot=1, extra_masks={"overall": pick})


def test_calibration_on_disk_never_intersects_evaluation_matches():
    """Data check of the calibration new runs load (skipped until scripts/calibrate_position_error_v3.py has run)."""
    path = default_pos_error_calibration_path()
    splits_path = ROOT / "outputs/state_value_main_50k_eval/match_splits.json"
    if not path.exists() or not splits_path.exists():
        pytest.skip(f"no calibration at {path}")
    cal = json.loads(path.read_text(encoding="utf-8"))
    ids = set(cal["match_ids"])
    assert ids and match_ids_sha256(ids) == cal["match_ids_sha256"]
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    for name in ("predict_test", "value_validation"):
        assert not ids & set(splits[name]), name
    # the v1 pilot's own evaluation matches (protocol 'engagement', the set the legacy curve was read from) and the
    # value model's test matches (protocol 'test'), read from the protocol itself rather than through match_splits
    protocol_path = ROOT / "outputs/temporal_winprob_v3_buckets/protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))["splits"]
        for name in ("engagement", "test"):
            assert protocol.get(name) and not ids & set(protocol[name]), name
    shards = Path(cal["split"]["corpus_shards"])
    if not shards.exists():
        pytest.skip(f"corpus shards {shards} not available")
    corpus = corpus_patch_matches(shards)
    assert cal["split"]["holdout_patches"]
    for patch in cal["split"]["holdout_patches"]:
        assert corpus.get(patch) and not ids & corpus[patch], patch
