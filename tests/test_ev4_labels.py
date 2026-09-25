"""Tests for scripts/exact_v4/ev4_04_labels.py (v4-exact R6 labels: SVI h 60/90/120, E5 outcome labels, prices).

Synthetic tests cover the endpoint / validity / SVI / E5 rules, the price OLS, the OOF contract and the guards (held-out
patches are refused before any data is touched; the fake paths do not exist).  Tests marked 'slow' run the labeller on
the 15.14 / 15.15 smoke outputs in outputs/reest_exact_v4_20260925/stage2/smoke_prerun (read only; outputs go to
tmp_path) and are skipped when those are absent.  No other patch's data is read.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
WT = Path(__file__).resolve().parents[1]
SCRIPTS = WT / "scripts" / "exact_v4"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ev4_v_models as VM  # noqa: E402
from tests import _ev4_lock_helpers as H  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("ev4_04_labels", SCRIPTS / "ev4_04_labels.py")
LB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LB)
_SPEC3 = importlib.util.spec_from_file_location("ev4_03_fit_v", SCRIPTS / "ev4_03_fit_v.py")
FV = importlib.util.module_from_spec(_SPEC3)
_SPEC3.loader.exec_module(FV)

from gameplay.split_guard import SplitViolation  # noqa: E402
from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH  # noqa: E402

NAMES = list(STATE_V3_COLUMNS)
SMOKE = LB.OUT_BASE / "stage2" / "smoke_prerun"
HAVE_SMOKE = (SMOKE / "extract" / "15.15" / "manifest.json").is_file() and \
    (SMOKE / "fit_v" / "frozen_manifest.json").is_file() and LB.CACHE.is_dir()
R = LB.dr_rules()


# ------------------------------------------------------------------ synthetic match
def _kill(t, killer, victim, x=5000, y=5000, bounty=300, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": t, "killerId": killer, "victimId": victim, "bounty": bounty,
            "shutdownBounty": 0, "assistingParticipantIds": list(assists), "position": {"x": x, "y": y}}


def _events():
    return [
        {"type": "LEVEL_UP", "timestamp": 100_000, "participantId": 7, "level": 6},
        {"type": "ITEM_PURCHASED", "timestamp": 101_000, "participantId": 1, "itemId": 1055},
        _kill(300_000, 1, 6, assists=(2,)),
        _kill(305_000, 7, 2, x=5100),
        _kill(310_000, 0, 3, y=5100),                                  # execution of a blue player
        {"type": "WARD_PLACED", "timestamp": 311_000, "creatorId": 4},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 420_000, "killerTeamId": 100, "killerId": 2,
         "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON", "position": {"x": 9800, "y": 4400}},
        _kill(450_000, 1, 8, x=1000, y=1000),
        {"type": "GAME_END", "timestamp": 900_000, "winningTeam": 100},
    ]


class _State:
    def __init__(self, t, snap):
        self.values = {"a": float(t), "b": 1.0}
        self.snapshot_ms = snap


class _Builder:
    """StateBuilderV3 stand-in: snapshot = last multiple of 60 s; raises for the times in `fail`."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def at(self, t):
        self.calls.append(int(t))
        if int(t) in self.fail:
            raise ValueError("boom")
        return _State(t, (int(t) // 60_000) * 60_000)


TM = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
PRICES = {"kills": 20.0, "assists": 45.0, "plates": 120.0, "dragon": 0.0}


def _row(**kw):
    r = {"match_id": "KR_1", "tau": 285_000, "first_kill_ts": 300_000, "last_kill_ts": 310_000, "game_end": 900_000,
         "next_start": -1, "e90": 400_000}
    r.update(kw)
    return r


def _det(**kw):
    d = {"eng_idx": 0, "kill_idx": [0, 1, 2], "kill_ts": [300_000, 305_000, 310_000], "centroid_x": 5000.0,
         "centroid_y": 5000.0}
    d.update(kw)
    return d


def _label(row=None, det=None, ts=None, builder=None, views=None):
    ev = _events()
    ts = np.arange(0, 900_001, 60_000) if ts is None else ts
    return LB.label_engagement(row or _row(), det or _det(), views or LB.event_views(ev), ts, builder or _Builder(), TM,
                               "15.14", PRICES, 4300.0, R, {})


def test_horizons_and_primary_match_dr_rules():
    assert LB.HORIZONS_S == tuple(R.HORIZONS_S) == (60, 90, 120) and LB.PRIMARY_H == R.PRIMARY_HORIZON_S == 90
    cols = LB.LABEL_COLUMNS
    assert len(cols) == len(set(cols))
    for c in ("match_id", "eng_idx", "tau", "cohort", "clean", "isolated", "p_pre", "svi", "valid", "kill_diff_label"):
        assert c in cols
    for h in LB.HORIZONS_S:
        for c in ("p_post_h", "svi_h", "valid_h", "e_h", "exchange_label_h", "next_obj_label_h"):
            assert f"{c}{h}" in cols


def test_endpoints_validity_and_e5_labels():
    out, posts = _label()
    for h in (60, 90, 120):                                              # min(L + h, next_kill - 1, game_end - 1)
        assert out[f"e_h{h}"] == 310_000 + 1000 * h and out[f"e_h{h}_reasons"] == "horizon"
        assert out[f"valid_h{h}"] == 1 and out[f"invalid_h{h}"] == ""
        assert out[f"post_snapshot_ms_h{h}"] == (out[f"e_h{h}"] // 60_000) * 60_000
        assert out[f"post_frame_age_ms_h{h}"] == out[f"e_h{h}"] - out[f"post_snapshot_ms_h{h}"]
    assert sorted(posts) == [60, 90, 120] and posts[90].dtype == np.float32 and posts[90][0] == 400_000
    # executions count against the victim team: +1 (blue kill) -1 (red kill) -1 (blue executed) = -1 -> red
    assert out["kill_diff"] == -1 and out["kill_diff_label"] == 0 and out["n_executions"] == 1 and out["n_own_kills"] == 3
    # exchange: kill gold blue 300+20+45, red 300+20 -> +45 inside the dead zone -> kill tier -> red
    assert out["exchange_label_h90"] == 0 and out["exchange_decided_by_h90"] == "kills"
    assert out["exchange_gold_h90"] == pytest.approx(45.0)
    # next objective: blue dragon at 420 s is in (e_h, e_h + 180 s] for h = 60 / 90, not for h = 120
    assert out["next_obj_label_h60"] == 1 == out["next_obj_label_h90"] and out["next_obj_outcome_h90"] == "Blue"
    assert out["next_obj_label_h120"] is None and out["next_obj_outcome_h120"] == "none"


def test_next_kill_cuts_the_endpoint_and_post_states_are_cached():
    ev = _events()
    ev[-2] = _kill(360_000, 1, 8, x=1000, y=1000)                        # next kill at 360 s
    b = _Builder()
    out, posts = LB.label_engagement(_row(e90=359_999), _det(), LB.event_views(ev), np.arange(0, 900_001, 60_000), b,
                                     TM, "15.14", PRICES, 4300.0, R, {})
    assert out["e_h60"] == out["e_h90"] == out["e_h120"] == 359_999 and out["e_h90_reasons"] == "next_kill"
    assert b.calls == [359_999]                                          # one post state for three equal endpoints


def test_invalid_endpoints_are_excluded_not_imputed():
    # last frame before e_h -> endpoint_le_last_frame fails for every h
    out, posts = _label(ts=np.arange(0, 360_001, 60_000))
    for h in (60, 90, 120):
        assert out[f"valid_h{h}"] == 0 and "endpoint_le_last_frame" in out[f"invalid_h{h}"]
        assert out[f"post_snapshot_ms_h{h}"] == -1
    assert posts == {}
    assert out["exchange_label_h90"] == 0                                # E5 labels do not depend on SVI validity
    # next engagement starts before L -> overlap flag, e_h < L -> no exchange label
    out, posts = _label(row=_row(next_start=308_000, e90=307_999))
    for h in (60, 90, 120):
        assert out[f"e_h{h}"] == 307_999 and out[f"valid_h{h}"] == 0
        assert out[f"invalid_h{h}"].split("|")[0] == "same_match_overlap_next_start_le_L"
        assert "endpoint_ge_L" in out[f"invalid_h{h}"]
        assert out[f"exchange_label_h{h}"] is None and out[f"exchange_decided_by_h{h}"] == "endpoint_before_L"
    assert posts == {}
    # a failing post state blocks only that horizon
    out, posts = _label(builder=_Builder(fail={400_000}))
    assert out["valid_h90"] == 0 and out["invalid_h90"].startswith("post_state_error:ValueError")
    assert out["valid_h60"] == 1 == out["valid_h120"] and sorted(posts) == [60, 120]


def test_input_consistency_is_enforced():
    with pytest.raises(RuntimeError, match="e90"):
        _label(row=_row(e90=399_999))
    with pytest.raises(RuntimeError, match="kill_ts"):
        _label(det=_det(kill_ts=[300_000, 305_000, 310_001]))
    with pytest.raises(RuntimeError, match="outside"):
        _label(det=_det(kill_idx=[0, 1, 9]))
    with pytest.raises(RuntimeError, match="first / last"):
        _label(row=_row(first_kill_ts=301_000))


def test_reduced_event_views_equal_full_events():
    ev = _events()
    a, pa = _label(views=LB.event_views(ev))
    b, pb = _label(views=LB.full_event_views(ev))
    assert LB._same_labels(a, b) and sorted(pa) == sorted(pb)
    assert not LB._same_labels(a, dict(b, exchange_label_h90=1))
    v = LB.event_views(ev)
    assert [e["timestamp"] for e in v["kills"]] == [300_000, 305_000, 310_000, 450_000]
    assert {e["type"] for e in v["exchange"]} <= set(LB.EXCHANGE_EVENT_TYPES)
    assert all(e["type"] == "ELITE_MONSTER_KILL" for e in v["elite"])


class _FakeV:
    """predict(X, mids) -> (X[:, 0] as probability, fold -1)."""

    def predict(self, X, mids):
        return np.asarray(X, dtype=np.float64)[:, 0], np.full(len(mids), -1, dtype=np.int8)


def test_attach_v_svi_rule_and_exact_zero():
    eng_X = np.array([[0.40, 0.0], [0.50, 0.0], [0.60, 0.0]], dtype=np.float32)
    rows = []
    for i in range(3):
        r = {"match_id": f"M{i}", "extract_row": i}
        for h in LB.HORIZONS_S:
            r[f"valid_h{h}"] = 1
        rows.append(r)
    rows[2]["valid_h120"] = 0
    post_X, post_ref = [], []
    for i, post in enumerate([(0.45, 0.40, 0.30), (0.50, 0.55, 0.49), (0.70, 0.60, None)]):
        for h, p in zip(LB.HORIZONS_S, post):
            if p is not None:
                post_X.append(np.array([p, 0.0], dtype=np.float32))
                post_ref.append((i, h))
    LB.attach_v(rows, eng_X, post_X, post_ref, _FakeV(), "frozen", R)
    r0, r1, r2 = rows
    assert r0["p_pre"] == pytest.approx(0.40) and r0["svi_h60"] == 1 and r0["svi_h90"] == 0 and r0["svi_h120"] == 0
    assert r0["svi_zero_h90"] == 1 and r0["dv_h90"] == 0.0                # exact zero -> 0, flagged
    assert r1["svi_h60"] == 0 and r1["svi_zero_h60"] == 1 and r1["svi_h90"] == 1 and r1["svi_h120"] == 0
    assert r2["svi_h120"] is None and math.isnan(r2["p_post_h120"]) and math.isnan(r2["dv_h120"])
    assert all(r["svi"] == r["svi_h90"] and r["valid"] == r["valid_h90"] for r in rows)
    assert all(r["v_source"] == "frozen" and r["v_fold"] == -1 for r in rows)
    with pytest.raises(RuntimeError, match="valid endpoints"):
        LB.attach_v(rows, eng_X, post_X[:-1], post_ref[:-1], _FakeV(), "frozen", R)


# ------------------------------------------------------------------ prices (15.14 only)
def test_price_ols_from_sufficient_stats_equals_pooled_ols_and_cluster_se():
    rng = np.random.default_rng(3)
    names = ["a", "b", "c"]
    blocks = []
    for g in range(60):
        X = rng.poisson(1.0, (20, 3)).astype(float)
        y = 10 + X @ np.array([100.0, 50.0, -5.0]) + rng.normal(0, 20, 20) + rng.normal(0, 30)   # match effect
        blocks.append((X, y))
    stats = [LB.match_price_stats(X, y) for X, y in blocks]
    fit = LB.fit_prices_from_stats(stats, names)
    Xa = np.vstack([b[0] for b in blocks])
    ya = np.concatenate([b[1] for b in blocks])
    A = np.hstack([np.ones((len(ya), 1)), Xa])
    beta = np.linalg.lstsq(A, ya, rcond=None)[0]
    assert [fit["coef"][k] for k in names] == pytest.approx(beta[1:], rel=1e-9) and fit["intercept"] == pytest.approx(beta[0])
    from analysis.event_prices import fit_prices
    import analysis.event_prices as EP
    old = EP.REGRESSORS
    try:
        EP.REGRESSORS = names
        ref = fit_prices(Xa, ya)
    finally:
        EP.REGRESSORS = old
    assert fit["r2"] == pytest.approx(ref["r2"], rel=1e-9)
    assert [fit["se"][k] for k in names] == pytest.approx([ref["se"][k] for k in names], rel=1e-6)
    assert [fit["count"][k] for k in names] == pytest.approx(Xa.sum(axis=0))
    # CR1 cluster sandwich computed directly on the rows
    e = ya - A @ beta
    Binv = np.linalg.inv(A.T @ A)
    meat = np.zeros((4, 4))
    a = 0
    for X, y in blocks:
        s = A[a:a + len(y)].T @ e[a:a + len(y)]
        meat += np.outer(s, s)
        a += len(y)
    n, p, G = len(ya), 4, len(blocks)
    V = (G / (G - 1)) * ((n - 1) / (n - p)) * Binv @ meat @ Binv
    assert [fit["se_cluster"][k] for k in names] == pytest.approx(np.sqrt(np.diag(V))[1:], rel=1e-8)
    bs = LB.bootstrap_prices(stats, names, n_boot=50, seed=1)
    assert bs == LB.bootstrap_prices(stats, names, n_boot=50, seed=1)
    assert bs["a"]["p2.5"] < fit["coef"]["a"] < bs["a"]["p97.5"]
    assert 0.5 < bs["a"]["sd"] / fit["se_cluster"]["a"] < 2.0


def test_price_table_keys_are_the_keys_the_exchange_label_reads():
    from analysis.event_prices import REGRESSORS, price_table
    from gameplay.labels import _priced_event_gold
    fit = {"coef": {k: 100.0 for k in REGRESSORS}, "count": {k: 1e6 for k in REGRESSORS}}
    fit["coef"]["d_jg_cs"] = 30.0
    table = price_table(fit)
    assert "kills" in table and "assists" in table and "first_tower" not in table
    assert _priced_event_gold({"type": "BUILDING_KILL", "towerType": "OUTER_TURRET"}, table, None) == 100.0
    assert _priced_event_gold({"type": "ELITE_MONSTER_KILL", "monsterType": "DRAGON",
                               "monsterSubType": "ELDER_DRAGON"}, table, None) == 130.0
    assert _priced_event_gold({"type": "TURRET_PLATE_DESTROYED"}, table, None) == 100.0


def _price_blob(**kw):
    b = {"format": LB.PRICES_FORMAT, "patch": "15.14", "source_patches": ["15.14"], "sample": False,
         "table": {"kills": 20.0, "assists": 45.0}, "fit": {"n_matches": 10}}
    b.update(kw)
    return b


@pytest.mark.parametrize("kw, exc", [({"patch": "15.15"}, RuntimeError), ({"source_patches": ["15.14", "15.15"]}, RuntimeError),
                                     ({"format": "x"}, RuntimeError), ({"sample": True}, SystemExit),
                                     ({"table": {"kills": 1.0}}, RuntimeError)])
def test_load_prices_refusals(tmp_path, kw, exc):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(_price_blob(**kw)), encoding="utf-8")
    with pytest.raises(exc):
        LB.load_prices(p, smoke=False)


def test_load_prices_ok_and_sample_under_smoke(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(_price_blob(sample=True)), encoding="utf-8")
    table, info = LB.load_prices(p, smoke=True)
    assert table == {"kills": 20.0, "assists": 45.0} and info["sha256"] == VM.sha256_file(p) and info["sample"]
    with pytest.raises(FileNotFoundError):
        LB.load_prices(tmp_path / "missing.json", smoke=True)


def test_price_pack_refuses_other_patches(tmp_path):
    (tmp_path / "KR_9.meta.json").write_text(json.dumps({"patch": "15.15", "team_map": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="15.14 only"):
        LB.load_price_pack("KR_9", tmp_path)


@pytest.mark.skipif(not LB.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_prices_cli_smoke_rules(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="smoke"):
        LB.main(["prices", "--matches", str(tmp_path / "none.parquet"), "--limit", "5"])
    monkeypatch.setattr(LB, "DEFAULT_PRICES", tmp_path / "prices_1514.json")
    with pytest.raises(SystemExit, match="canonical"):
        LB.main(["prices", "--matches", str(tmp_path / "none.parquet"), "--limit", "5", "--smoke"])
    assert not (tmp_path / "prices_1514.json").exists()
    import pandas as pd
    ml = tmp_path / "matches_15.14.parquet"
    pd.DataFrame({"match_id": ["KR_1", "KR_2"], "status": ["ok", "ok"]}).to_parquet(ml, index=False)
    with pytest.raises(SystemExit, match="is a sample"):                 # a short list without --limit is a sample too
        LB.main(["prices", "--matches", str(ml), "--out", str(tmp_path / "p.json")])
    with pytest.raises(SystemExit, match="canonical"):
        LB.main(["prices", "--matches", str(ml), "--smoke"])


# ------------------------------------------------------------------ OOF contract
def _tiny_bundle(d: Path, seed: int, kind_extra=None):
    rng = np.random.default_rng(seed)
    m = VM.LogisticModel(0.01)
    m.w, m.b = rng.normal(0, 0.05, len(NAMES)), 0.1 * seed
    m.mu, m.sd = np.zeros(len(NAMES)), np.ones(len(NAMES))
    m.fit_info = {}
    cal = VM.PositiveSlopeSigmoid.from_dict({"intercept_a": 0.0, "slope_b": 1.0})
    return VM.save_bundle(d, m, cal, NAMES, np.zeros(len(NAMES)), kind_extra)


def _mids(n=200):
    return [f"KR_{i}" for i in range(n)]


def _oof_run(tmp_path, smoke=True, ids=None):
    """A fake ev4_03 output dir: V_frozen + oof_v (5 tiny logistic folds) + frozen_manifest.json."""
    out = tmp_path / "fit"
    vf = _tiny_bundle(out / "V_frozen", 9)
    ids = ids or _mids()
    fb, ft = {}, {}
    for k in range(5):
        fb[k] = _tiny_bundle(out / "oof_v" / f"fold_{k}", k, {"oof_heldout_fold": k})
        ft[k] = [m for m in ids if VM.cv_fold(m) != k]
    blk = LB.write_oof_manifest(out / "oof_v", "logistic", {"C": 0.01}, fb, ft, vf["bundle_sha256"], smoke=smoke)
    man = {"smoke": smoke, "pilot": False, "frozen": True, "chosen": "logistic",
           "stop_rule": {"stop_before_record1": False}, "state_v3_name_hash": STATE_V3_NAME_HASH,
           "V_frozen": {"dir": str(out / "V_frozen"), "bundle_sha256": vf["bundle_sha256"], "kind": "logistic"},
           "inputs": {"train": {"manifest_sha256": "T" * 64}, "select": {"manifest_sha256": "S" * 64}},
           "oof_v": blk}
    (out / "frozen_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return out, man


def test_1514_refused_without_oof_export(tmp_path):
    out, man = _oof_run(tmp_path)
    del man["oof_v"]
    (out / "frozen_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable) as ei:
        LB.resolve_v("15.14", out, "T" * 64, smoke=True, record1_v=None)
    msg = str(ei.value)
    assert "oof_v" in msg and LB.OOF_FORMAT in msg and "train_match_ids.txt" in msg      # the spec is in the error
    # 15.15 does not need OOF
    assert LB.resolve_v("15.15", out, "S" * 64, smoke=True, record1_v=None)["source"] == "frozen"


def test_oof_routes_each_match_to_its_held_out_fold(tmp_path):
    out, man = _oof_run(tmp_path)
    spec = LB.resolve_v("15.14", out, "T" * 64, smoke=True, record1_v=None)
    assert spec["source"] == "oof" and sorted(spec["oof"]["folds"]) == [0, 1, 2, 3, 4]
    V = LB.VPredictor(spec)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, len(NAMES))).astype(np.float32)
    mids = [f"NEW_{i}" for i in range(40)]                                # not in any fold's training list
    p, folds = V.predict(X, mids)
    assert folds.tolist() == [VM.cv_fold(m) for m in mids]
    for k in range(5):
        sel = folds == k
        if sel.any():
            ref = VM.load_v(out / "oof_v" / f"fold_{k}").predict(X[sel])
            assert p[sel] == pytest.approx(ref)
    assert not np.allclose(p, VM.load_v(out / "V_frozen").predict(X))    # the frozen V is not used for 15.14
    assert V.coverage(mids) == {"n": 40, "full": 0}                      # NEW_* ids are in no training list
    assert V.coverage(["KR_1", "KR_2"]) == {"n": 2, "full": 2}
    # a labelled match inside the fold model's training list is refused
    V.train_ids[VM.cv_fold("NEW_0")].add("NEW_0")
    with pytest.raises(LB.OOFNotAvailable, match="trained on labelled"):
        V.predict(X[:1], ["NEW_0"])


@pytest.mark.parametrize("edit, match", [
    (lambda om: om.update(format="x"), "format"),
    (lambda om: om.update(kind="mlp"), "kind"),
    (lambda om: om.update(fold_rule="other"), "fold_rule"),
    (lambda om: om.update(frozen_bundle_sha256="0" * 64), "frozen_bundle"),
    (lambda om: om.update(pilot=True), "pilot"),
    (lambda om: om["folds"].pop("4"), "folds"),
])
def test_oof_manifest_violations_are_refused(tmp_path, edit, match):
    out, man = _oof_run(tmp_path)
    mp = out / "oof_v" / "oof_manifest.json"
    om = json.loads(mp.read_text(encoding="utf-8"))
    edit(om)
    mp.write_text(json.dumps(om), encoding="utf-8")
    man["oof_v"]["manifest_sha256"] = VM.sha256_file(mp)
    (out / "frozen_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match=match):
        LB.resolve_v("15.14", out, "T" * 64, smoke=True, record1_v=None)


def test_oof_tamper_smoke_and_leak_checks(tmp_path):
    out, man = _oof_run(tmp_path, smoke=True)
    with pytest.raises(LB.OOFNotAvailable, match="smoke"):              # smoke OOF set, labels run without --smoke
        LB.check_oof(dict(man, smoke=False), man["V_frozen"], smoke=False)
    tf = out / "oof_v" / "fold_2" / "train_match_ids.txt"
    tf.write_text(tf.read_text(encoding="utf-8") + "KR_extra\n", encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match="sha256"):
        LB.check_oof(man, man["V_frozen"], smoke=True)
    # the writer refuses a fold trained on its own held-out matches
    ids = _mids()
    own = next(m for m in ids if VM.cv_fold(m) == 0)
    with pytest.raises(RuntimeError, match="held-out fold"):
        LB.write_oof_manifest(out / "oof_v", "logistic", {}, {k: {"dir": str(out / "oof_v" / f"fold_{k}"),
                                                                   "bundle_sha256": "x"} for k in range(5)},
                              {k: ([own] if k == 0 else []) for k in range(5)}, "y", smoke=True)
    # a tampered fold bundle is refused when loaded
    out2, _ = _oof_run(tmp_path / "b")
    spec = LB.resolve_v("15.14", out2, "T" * 64, smoke=True, record1_v=None)
    npz = out2 / "oof_v" / "fold_1" / "logistic.npz"
    npz.write_bytes(npz.read_bytes() + b"0")
    with pytest.raises(RuntimeError, match="sha256"):
        LB.VPredictor(spec)


def test_v_manifest_guards(tmp_path):
    out, man = _oof_run(tmp_path)
    with pytest.raises(RuntimeError, match="extract V was built on"):
        LB.resolve_v("15.15", out, "X" * 64, smoke=True, record1_v=None)
    with pytest.raises(VM.VNotUsable, match="smoke"):
        LB.resolve_v("15.15", out, "S" * 64, smoke=False, record1_v=None)
    (out / "frozen_manifest.json").write_text(json.dumps(dict(man, stop_rule={"stop_before_record1": True})),
                                              encoding="utf-8")
    with pytest.raises(VM.VNotUsable, match="stop rule"):
        LB.resolve_v("15.15", out, "S" * 64, smoke=True, record1_v=None)
    (out / "frozen_manifest.json").write_text(json.dumps(dict(man, state_v3_name_hash="0")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="StateV3"):
        LB.resolve_v("15.15", out, "S" * 64, smoke=True, record1_v=None)


def _locked_v_record(tmp_path, bundle_sha, real_pin=False, **kw):
    """A synthetic records/ dir + locked record 1 naming `bundle_sha` (helpers in tests/_ev4_lock_helpers.py)."""
    rd, pre = H.make_records_dir(tmp_path)
    if real_pin:                                        # pre-decision file = the real one, so the ev4_common pin holds
        (rd / H.PRE_NAME).write_bytes(LB.EC.PREDECISIONS_RECORD.read_bytes())
        pre = H.sha(rd / H.PRE_NAME)

    def edit(b):
        b["V_frozen_bundle_sha256"] = bundle_sha
        b["V"]["V_frozen"]["bundle_sha256"] = bundle_sha
        b["records"]["predecisions_pinned"]["sha256"] = pre
    p, s = H.write_record(rd, edit=edit, **kw)
    return rd, pre, p, s


def test_check_record1_v_reads_only_the_top_level_key_of_a_locked_record(tmp_path):
    rd, pre, p, s = _locked_v_record(tmp_path, "a" * 64)
    rec = LB.RL.assert_record1_locked(p, s, records_dir=rd, predecisions_sha256=pre)
    assert LB.check_record1_v(rec, "A" * 64)["ok"]
    with pytest.raises(SplitViolation, match="names V bundle"):
        LB.check_record1_v(rec, "b" * 64)
    with pytest.raises(SplitViolation, match="not a path"):          # a path is never accepted
        LB.check_record1_v(str(p), "a" * 64)
    nested_only = dict(rec)
    nested_only.pop("V_frozen_bundle_sha256")                        # a nested V block alone is not enough
    with pytest.raises(SplitViolation, match="V_frozen_bundle_sha256"):
        LB.check_record1_v(nested_only, "a" * 64)
    draft = dict(rec, record="1 DRAFT - not locked", status="DRAFT", is_final_lock=False)
    with pytest.raises(SplitViolation, match="record field"):
        LB.check_record1_v(draft, "a" * 64)


def test_held_out_guard_and_resolve_v(tmp_path):
    out, man = _oof_run(tmp_path / "v")
    vb = man["V_frozen"]["bundle_sha256"]
    rd, pre, p, s = _locked_v_record(tmp_path, vb)
    g = LB.held_out_guard("15.16", str(p), s, out, smoke=True, records_dir=rd, predecisions_sha256=pre)
    assert g["ok"] and g["locked"] and g["record1_v_bundle_sha256"] == vb
    spec = LB.resolve_v("15.16", out, "Z" * 64, smoke=True, record1_v=g)   # held-out: frozen V, no extract-sha tie
    assert spec["source"] == "frozen" and spec["record1_v"]["record1_v_bundle_sha256"] == vb
    with pytest.raises(SplitViolation, match="locked record 1"):
        LB.resolve_v("15.16", out, "Z" * 64, smoke=True, record1_v=None)
    with pytest.raises(SplitViolation, match="locked record 1"):
        LB.resolve_v("15.16", out, "Z" * 64, smoke=True, record1_v=dict(g, record1_v_bundle_sha256="0" * 64))
    # another V bundle than the record names
    rd2, pre2, p2, s2 = _locked_v_record(tmp_path / "b", "0" * 64)
    with pytest.raises(SplitViolation, match="names V bundle"):
        LB.held_out_guard("15.16", str(p2), s2, out, smoke=True, records_dir=rd2, predecisions_sha256=pre2)
    # a draft and a smoke record are refused before V is read (the V path does not exist)
    for i, kw in enumerate(({"name": "record1_DRAFT_x.json"}, {"edit": lambda b: b.update(smoke=True)})):
        rd3, pre3 = H.make_records_dir(tmp_path / f"c{i}")
        p3, s3 = H.write_record(rd3, **kw)
        with pytest.raises(SplitViolation):
            LB.held_out_guard("15.16", str(p3), s3, tmp_path / "no_v", smoke=True, records_dir=rd3,
                              predecisions_sha256=pre3)
    with pytest.raises(SplitViolation, match="directly inside"):
        LB.held_out_guard("15.16", str(p), s, out, smoke=True)          # default: the real records/ only


def _heldout_argv(tmp_path, record, sha, vman):
    return ["labels", "--patch", "15.16", "--extract", str(tmp_path / "no_extract"), "--v-manifest", str(vman),
            "--prices", str(tmp_path / "no_prices.json"), "--out", str(tmp_path / "o"), "--smoke",
            "--record1", str(record), "--record1-sha256", sha]


class _Touched(RuntimeError):
    pass


@pytest.mark.skipif(not (LB.EC.PREDECISIONS_RECORD.is_file() and LB.EX.RECORD1A.is_file()),
                    reason="real pre-decision record / record 1A absent")
def test_labels_locked_record_check_runs_before_any_held_out_manifest(tmp_path, monkeypatch):
    def touched(*a, **k):
        raise _Touched("held-out extract manifest read")
    monkeypatch.setattr(LB, "read_extract_manifest", touched)
    out, man = _oof_run(tmp_path / "v")
    vb = man["V_frozen"]["bundle_sha256"]
    # a DRAFT (right sha, perfect content) in the synthetic records dir: refused, the extract is never read
    rd, pre, p, s = _locked_v_record(tmp_path / "a", vb, real_pin=True, name="record1_DRAFT_20260927T000000Z.json")
    monkeypatch.setattr(LB.RL, "RECORDS_DIR", rd)
    with pytest.raises(SplitViolation, match="draft"):
        LB.main(_heldout_argv(tmp_path, p, s, out))
    # a locked record naming another V: refused, the extract is never read
    rd, pre, p, s = _locked_v_record(tmp_path / "b", "0" * 64, real_pin=True)
    monkeypatch.setattr(LB.RL, "RECORDS_DIR", rd)
    with pytest.raises(SplitViolation, match="names V bundle"):
        LB.main(_heldout_argv(tmp_path, p, s, out))
    # the locked record naming this V: the guard passes and only then is the (held-out) extract manifest read
    rd, pre, p, s = _locked_v_record(tmp_path / "c", vb, real_pin=True)
    monkeypatch.setattr(LB.RL, "RECORDS_DIR", rd)
    with pytest.raises(_Touched):
        LB.main(_heldout_argv(tmp_path, p, s, out))
    assert not (tmp_path / "o").exists()


# ------------------------------------------------------------------ split guard / preset lock / extract checks
@pytest.mark.parametrize("extra, exc", [
    ([], SplitViolation),                                                # no record 1
    (["--record1", "C:/nonexistent/record1.json", "--record1-sha256", "0" * 64], Exception),
])
def test_labels_refuses_heldout_patch_before_any_data(tmp_path, extra, exc):
    argv = ["labels", "--patch", "15.16", "--extract", str(tmp_path / "no_extract"), "--v-manifest",
            str(tmp_path / "no_v"), "--prices", str(tmp_path / "no_prices.json"), "--out", str(tmp_path / "o")] + extra
    with pytest.raises(exc):
        LB.main(argv)
    assert not (tmp_path / "o").exists()


@pytest.mark.skipif(not LB.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_labels_refuses_record1a_for_heldout(tmp_path):
    argv = ["labels", "--patch", "16.14", "--extract", str(tmp_path / "x"), "--v-manifest", str(tmp_path / "v"),
            "--record1", str(LB.EX.RECORD1A), "--record1-sha256", VM.sha256_file(LB.EX.RECORD1A),
            "--out", str(tmp_path / "o")]
    with pytest.raises(SplitViolation, match="1A"):
        LB.main(argv)
    assert not (tmp_path / "o").exists()


def test_limit_chunks_is_smoke_only(tmp_path):
    with pytest.raises(SystemExit, match="smoke"):
        LB.main(["labels", "--patch", "15.15", "--extract", str(tmp_path / "x"), "--v-manifest", str(tmp_path / "v"),
                 "--limit-chunks", "1"])


def _xman(guard, **kw):
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH, STATE_VERSION
    m = {"patch": "15.15", "STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "state_version": STATE_VERSION, "sample": False,
         "remakes": LB.EC.remake_rule(), "code_sha256": LB.EX.code_hashes(),
         "guards": {"params": json.loads(json.dumps(guard["params"])), "record1a_sha256": guard["record1a_sha256"]},
         "detect_input": {"mode": "parquet", "path": "x", "sha256": {"x": "y"}}}
    m.update(kw)
    return m


@pytest.mark.skipif(not LB.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_check_extract_rules():
    _, _, guard = LB.EX.load_params()
    assert LB.check_extract(_xman(guard), "15.15", guard, smoke=False, allow_code_drift=False)["drift_found"] is False
    with pytest.raises(RuntimeError, match="patch"):
        LB.check_extract(_xman(guard, patch="15.14"), "15.15", guard, False, False)
    with pytest.raises(SystemExit, match="sample"):
        LB.check_extract(_xman(guard, sample=True), "15.15", guard, False, False)
    assert LB.check_extract(_xman(guard, sample=True), "15.15", guard, True, False)
    with pytest.raises(RuntimeError, match="remake"):
        LB.check_extract(_xman(guard, remakes={}), "15.15", guard, False, False)
    bad = json.loads(json.dumps(guard["params"]))
    bad["gap_ms"] = 13_700
    with pytest.raises(RuntimeError, match="params"):
        LB.check_extract(_xman(guard, guards={"params": bad, "record1a_sha256": guard["record1a_sha256"]}), "15.15",
                         guard, False, False)
    drift = dict(LB.EX.code_hashes(), **{"gameplay/state_value_v3.py": "0" * 64})
    with pytest.raises(RuntimeError, match="code differs"):
        LB.check_extract(_xman(guard, code_sha256=drift), "15.15", guard, False, False)
    got = LB.check_extract(_xman(guard, code_sha256=drift), "15.15", guard, False, True)
    assert got["drift_found"] and "gameplay/state_value_v3.py" in got["drift"]
    with pytest.raises(RuntimeError, match="inline-detect"):
        LB.check_extract(_xman(guard, detect_input={"mode": "inline"}), "15.15", guard, False, False)


def test_resolve_detect_requires_the_extract_detect_table(tmp_path):
    import pandas as pd
    p = tmp_path / "engagements_15.15.parquet"
    pd.DataFrame({"match_id": ["a"], "tau": [1]}).to_parquet(p, index=False)
    man = {"detect_input": {"path": str(p), "sha256": {str(p): VM.sha256_file(p)}}}
    assert LB.resolve_detect(man, None)[0] == p
    man["detect_input"]["sha256"] = {str(p): "0" * 64}
    with pytest.raises(RuntimeError, match="does not hash"):
        LB.resolve_detect(man, None)


# ------------------------------------------------------------------ smoke (15.14 / 15.15 only; read only)
def _smoke_prices(tmp_path) -> Path:
    p = tmp_path / "prices_smoke.json"
    LB.main(["prices", "--matches", str(SMOKE / "detect" / "matches_15.14.parquet"), "--smoke", "--limit", "40",
             "--out", str(p), "--n-boot", "20"])
    return p


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SMOKE, reason="smoke outputs / cache absent")
def test_smoke_prices_1514(tmp_path):
    p = _smoke_prices(tmp_path)
    b = json.loads(p.read_text(encoding="utf-8"))
    assert b["patch"] == "15.14" and b["source_patches"] == ["15.14"] and b["sample"] is True
    assert b["matches"]["n_listed"] == 40 and sum(b["matches"]["status"].values()) == 40
    assert {"kills", "assists", "plates"} <= set(b["table"]) and b["fit"]["r2"] > 0.8
    # equals the pooled OLS of analysis.event_prices on the same rows
    from analysis.event_prices import fit_prices, match_rows
    ids = LB.price_match_ids(SMOKE / "detect" / "matches_15.14.parquet", 40)
    Xs, Ys = [], []
    for m in ids:
        pk = LB.load_price_pack(m)
        if LB.EC.is_remake(LB.EX.game_end_of(pk["events"])):
            continue
        X, Y = match_rows(pk)
        Xs.append(X)
        Ys.append(Y)
    ref = fit_prices(np.vstack(Xs), np.concatenate(Ys))
    for k, v in ref["coef"].items():
        assert b["fit"]["coef"][k] == pytest.approx(v, rel=1e-6, abs=1e-6)
    # a sample fit is not accepted by a full labels run
    with pytest.raises(SystemExit):
        LB.load_prices(p, smoke=False)


def _smoke_labels(tmp_path, patch, vman, prices, extra=()):
    out = tmp_path / "labels"
    argv = ["labels", "--patch", patch, "--extract", str(SMOKE / "extract" / patch), "--v-manifest", str(vman),
            "--prices", str(prices), "--out", str(out), "--workers", "1", "--smoke", "--allow-code-drift",
            "--limit-chunks", "1"] + list(extra)
    return LB.main(argv), out


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SMOKE, reason="smoke outputs / cache absent")
def test_smoke_labels_1515_end_to_end_and_resume(tmp_path):
    import pandas as pd
    prices = _smoke_prices(tmp_path)
    man, out = _smoke_labels(tmp_path, "15.15", SMOKE / "fit_v" / "frozen_manifest.json", prices)
    df = pd.read_parquet(out / "labels_15.15.parquet")
    xman = json.loads((SMOKE / "extract" / "15.15" / "manifest.json").read_text(encoding="utf-8"))
    assert len(df) == xman["chunks"]["00000"]["rows"]["eng"] == man["summary"]["rows"]
    assert list(df.columns) == list(LB.LABEL_COLUMNS)
    assert not df.duplicated(["match_id", "tau"]).any() and (df["isolated"] == 1).all()
    assert (df["game_end"] >= 300_000).all() and (df["v_source"] == "frozen").all() and (df["v_fold"] == -1).all()
    assert man["spot_check_pre_state"]["n"] == 20 and man["spot_check_pre_state"]["failures"] == 0
    assert man["use"].startswith("select") and man["smoke"] and man["inputs"]["prices"]["sha256"] == VM.sha256_file(prices)
    # p_pre is V(eng_X) of the stored rows
    vf = VM.assert_v_usable(SMOKE / "fit_v" / "frozen_manifest.json", allow_smoke=True)
    V = VM.load_v(vf["dir"], expected_sha256=vf["bundle_sha256"])
    with np.load(SMOKE / "extract" / "15.15" / "chunk_00000.npz") as z:
        X = z["eng_X"]
    assert df["p_pre"].to_numpy() == pytest.approx(V.predict(X[df["extract_row"].to_numpy()]))
    for h in LB.HORIZONS_S:
        v = df[f"valid_h{h}"] == 1
        assert ((df.loc[v, f"svi_h{h}"].astype(int)) == (df.loc[v, f"dv_h{h}"] > 0).astype(int)).all()
        assert df.loc[~v, f"svi_h{h}"].isna().all() and df.loc[v, f"svi_h{h}"].notna().all()
        assert (df.loc[v, f"p_post_h{h}"].between(0, 1)).all()
        assert (df.loc[v, f"e_h{h}"] >= df.loc[v, "last_kill_ts"]).all()
        assert (df[f"e_h{h}"] <= df["game_end"] - 1).all()
        assert (df.loc[v, f"post_snapshot_ms_h{h}"] <= df.loc[v, f"e_h{h}"]).all()
        g = df[f"exchange_decided_by_h{h}"] == "gold"
        assert ((df.loc[g, f"exchange_gold_h{h}"] > 0).astype(int) == df.loc[g, f"exchange_label_h{h}"].astype(int)).all()
        assert set(df[f"next_obj_label_h{h}"].dropna().astype(int)) <= {0, 1}
    assert (df["e_h60"] <= df["e_h90"]).all() and (df["e_h90"] <= df["e_h120"]).all()
    assert (df["e_h90"] == df["e90"]).all()
    assert (df["svi"].fillna(-1) == df["svi_h90"].fillna(-1)).all()
    kd = df["kill_diff_label"]
    assert ((kd.dropna().astype(int)) == (df.loc[kd.notna(), "kill_diff"] > 0).astype(int)).all()
    assert (df.loc[kd.isna(), "kill_diff"] == 0).all()
    # resume: the part is reused and the output is identical
    before = VM.sha256_file(out / "labels_15.15.parquet")
    man2, _ = _smoke_labels(tmp_path, "15.15", SMOKE / "fit_v" / "frozen_manifest.json", prices)
    assert VM.sha256_file(out / "labels_15.15.parquet") == before and man2["parts"] == man["parts"]
    assert man2["seconds_wall_this_invocation"] < man["seconds_wall_this_invocation"]


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SMOKE, reason="smoke outputs / cache absent")
def test_smoke_labels_1514_refused_with_the_current_ev4_03_output(tmp_path):
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps(_price_blob(sample=True)), encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match="oof_v"):
        _smoke_labels(tmp_path, "15.14", SMOKE / "fit_v" / "frozen_manifest.json", prices)
    assert not (tmp_path / "labels" / "labels_15.14.parquet").exists()


def build_smoke_oof(fit_dir: Path, out_dir: Path, maxiter: int = 60) -> Path:
    """DEV ONLY (smoke): an OOF_SPEC export built from the smoke 15.14 / 15.15 extracts for the smoke V (logistic):
    five fold logistic models at the chosen C, each calibrated on V_CAL, plus a copy of frozen_manifest.json with an
    'oof_v' block in out_dir.  Stand-in for the ev4_03 export until it exists; marked smoke."""
    vman = json.loads((fit_dir / "frozen_manifest.json").read_text(encoding="utf-8"))
    vf = vman["V_frozen"]
    bundle = json.loads((Path(vf["dir"]) / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["kind"] == "logistic"
    C = float(bundle["model"]["C"])
    tr_dir, se_dir = SMOKE / "extract" / "15.14", SMOKE / "extract" / "15.15"
    Xtr, mtr = FV.load_v_rows(tr_dir, FV.read_manifest(tr_dir))
    Xse, mse = FV.load_v_rows(se_dir, FV.read_manifest(se_dir))
    fill, _ = VM.nan_fill_values(Xtr)
    VM.fill_nan_inplace(Xtr, fill)
    VM.fill_nan_inplace(Xse, fill)
    y = mtr["y_blue_win"].to_numpy().astype(np.float64)
    perm = VM.swap_permutation(NAMES)
    VM.apply_swap_inplace(Xtr, y, VM.swap_mask(len(y)), perm)
    mids = mtr["match_id"].astype(str).to_numpy()
    folds = VM.per_match(mids, VM.cv_fold).astype(int)
    cal = VM.per_match(mse["match_id"].astype(str).to_numpy(), VM.v_split) == "cal"
    yse = mse["y_blue_win"].to_numpy().astype(np.float64)
    oof = out_dir / "oof_v"
    fb, ft = {}, {}
    for k in range(5):
        tr = np.flatnonzero(folds != k)
        mu, sd = VM.pooled_standardizer(Xtr, tr, VM.swap_pairs(perm))
        m = VM.LogisticModel(C).fit(Xtr, y, tr, mu, sd, maxiter=maxiter)
        c = VM.PositiveSlopeSigmoid().fit(m.predict_proba(Xse[cal]), yse[cal], np.ones(int(cal.sum())))
        fb[k] = VM.save_bundle(oof / f"fold_{k}", m, c, NAMES, fill, {"oof_heldout_fold": k, "smoke_dev": True})
        ft[k] = sorted(set(mids[tr]))
    blk = LB.write_oof_manifest(oof, "logistic", {"C": C}, fb, ft, vf["bundle_sha256"], smoke=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_manifest.json").write_text(json.dumps(dict(vman, oof_v=blk)), encoding="utf-8")
    return out_dir / "frozen_manifest.json"


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SMOKE, reason="smoke outputs / cache absent")
def test_smoke_labels_1514_with_a_spec_conforming_oof_export(tmp_path):
    import pandas as pd
    vman = build_smoke_oof(SMOKE / "fit_v", tmp_path / "fit_oof", maxiter=30)
    prices = _smoke_prices(tmp_path)
    man, out = _smoke_labels(tmp_path, "15.14", vman, prices)
    df = pd.read_parquet(out / "labels_15.14.parquet")
    assert man["use"].startswith("train") and man["inputs"]["v"]["source"] == "oof"
    assert (df["v_source"] == "oof").all()
    assert df["v_fold"].tolist() == [VM.cv_fold(m) for m in df["match_id"]]
    assert set(df["v_fold"]) <= set(range(5)) and len(set(df["v_fold"])) > 1
    # each row's p_pre comes from its own fold model, which never trained on that match
    with np.load(SMOKE / "extract" / "15.14" / "chunk_00000.npz") as z:
        X = z["eng_X"]
    oof = tmp_path / "fit_oof" / "oof_v"
    for k in sorted(set(df["v_fold"])):
        sel = (df["v_fold"] == k).to_numpy()
        train = set((oof / f"fold_{k}" / "train_match_ids.txt").read_text(encoding="utf-8").split())
        assert not set(df.loc[sel, "match_id"]) & train
        ref = VM.load_v(oof / f"fold_{k}").predict(X[df.loc[sel, "extract_row"].to_numpy()])
        assert df.loc[sel, "p_pre"].to_numpy() == pytest.approx(ref)
    assert man["spot_check_pre_state"]["failures"] == 0
    cov = man["oof_coverage"]
    assert cov["n"] == df["match_id"].nunique() and cov["full"] >= 0.99 * cov["n"]
