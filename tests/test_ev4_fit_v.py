"""Tests for scripts/exact_v4/ev4_03_fit_v.py and scripts/exact_v4/ev4_v_models.py (v4-exact R5, E4).

All data here is synthetic.  The split guard is exercised with fake extract manifests (patch 15.16 etc.) and is
required to refuse them before any array file is opened (the fake directories contain no array files at all).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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

_SPEC = importlib.util.spec_from_file_location("ev4_03_fit_v", SCRIPTS / "ev4_03_fit_v.py")
FV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FV)

from gameplay.split_guard import SplitViolation  # noqa: E402
from gameplay.state_value_v3 import SLOT_PREFIXES, STATE_V3_COLUMNS, STATE_V3_NAME_HASH, STATE_VERSION  # noqa: E402

NAMES = list(STATE_V3_COLUMNS)
PERM = VM.swap_permutation(NAMES)


# ------------------------------------------------------------------ selection rule
@pytest.mark.parametrize("losses, chosen", [
    ({"logistic": 0.6000, "lgbm": 0.5996, "mlp": 0.5990}, "mlp"),          # nobody within 0.0005 of the best
    ({"logistic": 0.5995, "lgbm": 0.5990, "mlp": 0.5990}, "logistic"),     # exactly 0.0005 -> simpler wins
    ({"logistic": 0.61, "lgbm": 0.5993, "mlp": 0.5990}, "lgbm"),           # lgbm within, logistic not
    ({"logistic": 0.5, "lgbm": 0.5, "mlp": 0.5}, "logistic"),              # ties -> simplest
    ({"logistic": 0.5, "lgbm": 0.4, "mlp": 0.45}, "lgbm"),                 # plain best
    ({"lgbm": 0.5003, "mlp": 0.5}, "lgbm"),                                # subset of candidates
])
def test_select_candidate(losses, chosen):
    out = VM.select_candidate(losses)
    assert out["chosen"] == chosen
    assert out["best_loss"] == min(losses.values())
    assert out["simpler_taken"] == (chosen != min(losses, key=lambda c: (losses[c], VM.SIMPLICITY_ORDER.index(c))))


def test_select_candidate_refuses_bad_input():
    with pytest.raises(ValueError):
        VM.select_candidate({"xgb": 0.5})
    with pytest.raises(ValueError):
        VM.select_candidate({"logistic": float("nan"), "lgbm": 0.5})
    with pytest.raises(ValueError):
        VM.select_candidate({})
    assert VM.select_candidate({"logistic": 0.5006, "mlp": 0.5}, tol=0.0005)["chosen"] == "mlp"


# ------------------------------------------------------------------ team swap
def test_swap_permutation_statev3():
    assert np.array_equal(PERM[PERM], np.arange(len(NAMES)))
    idx = {n: i for i, n in enumerate(NAMES)}
    assert PERM[idx["blue_top_totalGold_norm"]] == idx["red_top_totalGold_norm"]
    assert PERM[idx["red_utility_ch_tag_Tank"]] == idx["blue_utility_ch_tag_Tank"]
    assert PERM[idx["obj_blue_barons"]] == idx["obj_red_barons"]
    assert PERM[idx["blue_dragon_FIRE_x_time"]] == idx["red_dragon_FIRE_x_time"]
    for g in ("time_minutes", "snapshot_age_s", "obj_dragon_up", "obj_rift_element_fire", "unknown_objective_team_count"):
        assert PERM[idx[g]] == idx[g]
    fixed = [NAMES[j] for j in range(len(NAMES)) if PERM[j] == j]
    assert all("blue" not in n.split("_") and "red" not in n.split("_") for n in fixed)
    moved = len(NAMES) - len(fixed)
    assert moved % 2 == 0 and moved >= 700            # all 10 player blocks at least


def test_swap_permutation_refuses_unmapped_or_missing():
    with pytest.raises(ValueError):
        VM.swap_permutation(["blue_a", "red_a", "x_blue"])     # team token that is not mapped
    with pytest.raises(ValueError):
        VM.swap_permutation(["blue_a", "t"])                   # partner missing


def test_apply_swap_inplace_is_involution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, len(NAMES))).astype(np.float32)
    y = (rng.random(500) < 0.5).astype(np.float64)
    X0, y0 = X.copy(), y.copy()
    m = VM.swap_mask(500, seed=3)
    assert 200 < m.sum() < 300
    VM.apply_swap_inplace(X, y, m, PERM, chunk=37)
    r = np.flatnonzero(m)[0]
    assert np.array_equal(X[r], X0[r][PERM]) and y[r] == 1 - y0[r]
    k = np.flatnonzero(~m)[0]
    assert np.array_equal(X[k], X0[k]) and y[k] == y0[k]
    VM.apply_swap_inplace(X, y, m, PERM)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    assert np.array_equal(VM.swap_mask(500, seed=3), m)          # seeded


def test_pooled_standardizer_is_swap_equivariant():
    rng = np.random.default_rng(1)
    X = (rng.normal(size=(400, len(NAMES))) * rng.uniform(0.1, 3, len(NAMES)) + rng.normal(size=len(NAMES))).astype(np.float32)
    for groups in (VM.swap_pairs(PERM), VM.mlp_groups(NAMES, PERM)):
        mu, sd = VM.pooled_standardizer(X, np.arange(0, 400, 2), groups)
        assert np.allclose(mu[PERM], mu) and np.allclose(sd[PERM], sd)
    mu, sd = VM.pooled_standardizer(X, None, VM.mlp_groups(NAMES, PERM))
    slot_idx, other, _ = VM.player_layout(NAMES)
    assert slot_idx.shape == (10, 72) and len(other) == len(NAMES) - 720 == 276
    assert np.allclose(mu[slot_idx[:, 0]], mu[slot_idx[0, 0]])      # pooled over the 10 slots


def _antisymmetric_data(n, seed, dense=True, bias=0.0, gold_coef=0.8, kill_coef=0.5):
    """Synthetic V rows whose true P(blue) depends on blue - red gold and blue - red kills only.

    dense=False: only the gold / kill columns and 20 noise columns vary (the rest are 0), so a model's noise
    weights stay small and swap-antisymmetry can be checked tightly.  kill_coef=0: the gold-difference baseline
    is the true model (used to make the F12 stop rule trigger)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    idx = {c: i for i, c in enumerate(NAMES)}
    if not dense:
        keep = [idx[p + "totalGold_norm"] for p in SLOT_PREFIXES] + [idx["blue_kills"], idx["red_kills"]]
        keep += list(np.random.default_rng(99).choice(len(NAMES), 20, replace=False))
        mask = np.zeros(len(NAMES), bool)
        mask[keep] = True
        X[:, ~mask] = 0.0
    gb = sum(X[:, idx[p + "totalGold_norm"]] for p in SLOT_PREFIXES[:5])
    gr = sum(X[:, idx[p + "totalGold_norm"]] for p in SLOT_PREFIXES[5:])
    eta = bias + gold_coef * (gb - gr) + kill_coef * (X[:, idx["blue_kills"]] - X[:, idx["red_kills"]])
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    return X, y


def test_logistic_trained_with_swaps_is_nearly_antisymmetric():
    # the data have a blue-side bias (+0.6 logit): without swaps V learns it, with swaps it is averaged out
    X, y = _antisymmetric_data(20000, 2, dense=False, bias=0.6)
    X0, y0 = X.copy(), y.copy()
    m = VM.swap_mask(len(y), seed=5)
    VM.apply_swap_inplace(X, y, m, PERM)
    mu, sd = VM.pooled_standardizer(X, None, VM.swap_pairs(PERM))
    lm = VM.LogisticModel(0.01).fit(X, y, np.arange(len(y)), mu, sd)
    l0 = VM.LogisticModel(0.01).fit(X0, y0, np.arange(len(y0)), mu, sd)
    Xt, _ = _antisymmetric_data(1000, 9, dense=False)
    asym = np.mean(np.abs(lm.predict_proba(Xt) + lm.predict_proba(Xt[:, PERM]) - 1))
    asym0 = np.mean(np.abs(l0.predict_proba(Xt) + l0.predict_proba(Xt[:, PERM]) - 1))
    assert asym < 0.03 and asym0 > asym + 0.1
    assert abs(lm.b) < 0.1 and l0.b > 0.4


def test_mlp_minibatch_swaps_give_nearly_antisymmetric_model():
    X, y = _antisymmetric_data(4000, 4, dense=False, bias=0.6)
    tr, st = np.arange(3600), np.arange(3600, 4000)
    mu, sd = VM.pooled_standardizer(X, tr, VM.mlp_groups(NAMES, PERM))
    mm = VM.MLPModel(1e-4, NAMES, seeds=(0,), max_epochs=6, patience=3, threads=2).fit(X, y, tr, st, mu, sd)
    Xt, yt = _antisymmetric_data(800, 11, dense=False)
    p, ps = mm.predict_proba(Xt), mm.predict_proba(Xt[:, PERM])
    assert VM.log_loss(yt, p) < 0.66                                   # it learnt something
    assert np.mean(np.abs(p + ps - 1)) < 0.1                          # and it is close to swap-antisymmetric


def test_logistic_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n, p = 2000, 12
    X = (rng.normal(size=(n, p)) * rng.uniform(0.5, 5, p) + rng.normal(size=p)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * X[:, 0] - 0.2 * X[:, 1])))).astype(float)
    mu, sd = VM.pooled_standardizer(X, None, [])
    for C in VM.GRIDS["logistic"]["C"]:
        m = VM.LogisticModel(C).fit(X, y, np.arange(n), mu, sd)
        sk = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit((X - mu) / sd, y)
        assert np.max(np.abs(m.predict_proba(X) - sk.predict_proba((X - mu) / sd)[:, 1])) < 1e-4
    # rows argument: fitting on a subset == fitting on the subset copy
    sub = np.arange(0, n, 3)
    a = VM.LogisticModel(0.01).fit(X, y, sub, mu, sd).predict_proba(X)
    b = VM.LogisticModel(0.01).fit(X[sub], y[sub], np.arange(len(sub)), mu, sd).predict_proba(X)
    assert np.max(np.abs(a - b)) < 1e-5


# ------------------------------------------------------------------ calibration
def test_calibration_monotone_even_when_anticorrelated():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.98, 3000)
    y_good = (rng.random(3000) < p).astype(float)
    y_bad = (rng.random(3000) < 1 - p).astype(float)
    grid = np.linspace(0.001, 0.999, 500)
    for y in (y_good, y_bad):
        c = VM.PositiveSlopeSigmoid().fit(p, y, np.ones(len(y)))
        assert c.b >= VM.PositiveSlopeSigmoid.SLOPE_MIN
        assert np.all(np.diff(c.predict(grid)) >= 0)
    c = VM.PositiveSlopeSigmoid().fit(p, y_bad, np.ones(len(p)))
    assert c.fit_info["slope_bound_active"]
    c2 = VM.PositiveSlopeSigmoid.from_dict(VM.PositiveSlopeSigmoid().fit(p, y_good, np.ones(3000)).to_dict())
    assert 0.8 < c2.b < 1.2 and np.all(np.diff(c2.predict(grid)) > 0)


def test_binary_metrics_on_calibrated_data():
    rng = np.random.default_rng(6)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.random(40000) < p).astype(float)
    m = VM.binary_metrics(y, p)
    assert abs(m["cal_slope"] - 1) < 0.06 and abs(m["cal_intercept"]) < 0.05 and abs(m["cal_in_the_large"]) < 0.05
    assert 0.7 < m["auc"] < 0.8 and m["brier"] < 0.25
    const = VM.binary_metrics(y[:500], np.full(500, 0.4))                       # slope not identified -> NaN
    assert np.isnan(const["cal_slope"]) and abs(const["cal_in_the_large"] - (np.log(y[:500].mean() / (1 - y[:500].mean())) - np.log(0.4 / 0.6))) < 1e-6
    sep = VM.binary_metrics(np.r_[np.zeros(50), np.ones(50)], np.r_[np.full(50, 0.3), np.full(50, 0.31)])
    assert np.isnan(sep["cal_slope"])                                                # quasi-separation -> NaN, not 1e10
    bins = VM.metrics_by_bins(y, p, rng.integers(0, 2_400_000, 40000))
    assert sum(bins[b]["n"] for b in ("lt15", "15to25", "ge25")) == 40000


# ------------------------------------------------------------------ martingale checks
def test_martingale_means_bonferroni_and_bands():
    assert abs(VM.bonferroni_z(4) - 2.2414) < 1e-3
    rng = np.random.default_rng(7)
    n = 40000
    cl = np.array([f"m{i}" for i in range(n)])
    strata = {"all": np.ones(n, bool), "frame_update_yes": np.arange(n) % 2 == 0,
              "frame_update_no": np.arange(n) % 2 == 1, "recent_deaths_ge3": np.arange(n) % 3 == 0}
    ok = VM.martingale_means(rng.normal(0, 0.03, n), cl, strata, FV.MART_BANDS)
    assert ok["all_pass"] and ok["k"] == 4
    drift = VM.martingale_means(rng.normal(0.004, 0.03, n), cl, strata, FV.MART_BANDS)
    assert not drift["strata"]["all"]["pass"] and drift["strata"]["recent_deaths_ge3"]["band"] == 0.005


def test_efficiency_regression_detects_signal_and_flags_reduction():
    rng = np.random.default_rng(8)
    n = 4000
    X = rng.normal(size=(n, 30))
    X = np.column_stack([X, X[:, :3] @ rng.normal(size=(3, 4)), np.ones(n)])     # collinear + constant columns
    cl = np.array([f"m{i}" for i in range(n)])
    half = rng.integers(0, 2, n)
    null = VM.efficiency_regression(rng.normal(0, 0.05, n), X, cl, half, n_boot=49)
    assert null["rank"] == 30 and null["n_constant_dropped"] == 1 and not null["reduced"]
    assert null["p_value_F"] > 1e-3 and null["wild_bootstrap"]["p_value"] > 0.02
    assert null["p_value_primary"] == null["wild_bootstrap"]["p_value"]
    assert null["r2_oos_mean"] < 0.01
    sig = VM.efficiency_regression(0.02 * X[:, 0] + rng.normal(0, 0.05, n), X, cl, half)
    assert sig["p_value_F"] < 1e-10 and sig["r2_oos_mean"] > 0.05
    assert sig["p_value_primary"] is None and "wild_bootstrap" not in sig          # n_boot = 0
    small = VM.efficiency_regression(rng.normal(size=100), X[:100], cl[:100], half[:100])
    assert small["reduced"] and small["k_used"] == 10


# ------------------------------------------------------------------ frozen bundles and predict API
@pytest.mark.parametrize("kind", ["logistic", "lgbm", "mlp"])
def test_bundle_roundtrip_and_tamper(tmp_path, kind):
    X, y = _antisymmetric_data(1500, 12)
    fill = np.zeros(len(NAMES))
    if kind == "logistic":
        mu, sd = VM.pooled_standardizer(X, None, VM.swap_pairs(PERM))
        m = VM.LogisticModel(0.01).fit(X, y, np.arange(len(y)), mu, sd)
    elif kind == "lgbm":
        ds = VM.LGBMModel.dataset(X, y)
        m = VM.LGBMModel(31, max_rounds=30, patience=5, threads=2).fit(ds, np.arange(1300), np.arange(1300, 1500))
    else:
        mu, sd = VM.pooled_standardizer(X, None, VM.mlp_groups(NAMES, PERM))
        m = VM.MLPModel(1e-3, NAMES, seeds=(0, 1), max_epochs=2, threads=2).fit(X, y, np.arange(1300),
                                                                              np.arange(1300, 1500), mu, sd)
    cal = VM.PositiveSlopeSigmoid().fit(m.predict_proba(X), y, np.ones(len(y)))
    info = VM.save_bundle(tmp_path / kind, m, cal, NAMES, fill)
    assert VM.save_bundle(tmp_path / f"{kind}_again", m, cal, NAMES, fill)["bundle_sha256"] == info["bundle_sha256"]
    V = VM.load_v(tmp_path / kind, expected_sha256=info["bundle_sha256"], expected_columns_hash=STATE_V3_NAME_HASH)
    want = cal.predict(m.predict_proba(X[:200]))
    assert np.max(np.abs(V.predict(X[:200]) - want)) < 1e-7
    rev = NAMES[::-1]
    assert np.max(np.abs(V.predict(X[:50, ::-1], columns=rev) - want[:50])) < 1e-5   # float32 summation order
    Xn = X[:5].copy()
    Xn[0, 3] = np.nan
    assert np.all(np.isfinite(V.predict(Xn)))
    with pytest.raises(ValueError):
        V.predict(X[:5, :100])
    with pytest.raises(RuntimeError):
        VM.load_v(tmp_path / kind, expected_sha256="0" * 64)
    victim = tmp_path / kind / "prep.npz"
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(RuntimeError):
        VM.load_v(tmp_path / kind)


# ------------------------------------------------------------------ split guard and CLI rules
def _fake_manifest(d: Path, patch: str, sample: bool = True, **kw) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    m = {"patch": patch, "STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "state_version": STATE_VERSION, "sample": sample,
         "chunks": {}}
    m.update(kw)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d


@pytest.mark.parametrize("train, select", [("15.14", "15.16"), ("15.16", "15.15"), ("15.14", "16.13"),
                                           ("15.14", "15.18"), ("15.15", "15.14"), ("15.14", "15.14")])
def test_check_inputs_refuses_non_selection_patches(tmp_path, train, select):
    a = _fake_manifest(tmp_path / "a", train)
    b = _fake_manifest(tmp_path / "b", select)
    with pytest.raises(SplitViolation):
        FV.check_inputs(a, b, smoke=True)


def test_main_refuses_1516_before_any_array(tmp_path):
    a = _fake_manifest(tmp_path / "a", "15.14")
    b = _fake_manifest(tmp_path / "b", "15.16", chunks={"00000": {"rows": {"v": 1, "mart": 0}, "files": {}}})
    with pytest.raises(SplitViolation):                    # not FileNotFoundError: no chunk file is ever opened
        FV.main(["--train", str(a), "--select", str(b), "--smoke", "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_main_cli_rules(tmp_path):
    a = _fake_manifest(tmp_path / "a", "15.14", sample=True)
    b = _fake_manifest(tmp_path / "b", "15.15", sample=True)
    with pytest.raises(SystemExit):                        # sample extracts need --smoke
        FV.main(["--train", str(a), "--select", str(b), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):                        # budget overrides need --smoke
        FV.main(["--train", str(a), "--select", str(b), "--mlp-epochs", "2", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):                        # candidate subsets need --smoke
        FV.main(["--train", str(a), "--select", str(b), "--candidates", "logistic", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        FV.main(["--train", str(a), "--select", str(b), "--smoke", "--threads", "8", "--out", str(tmp_path / "o")])
    with pytest.raises(RuntimeError):                      # StateV3 hash mismatch
        FV.check_inputs(_fake_manifest(tmp_path / "c", "15.14", STATE_V3_NAME_HASH="x"), b, smoke=True)


def test_split_functions_deterministic_and_balanced():
    mids = [f"KR_{i}" for i in range(4000)]
    s = VM.per_match(mids, VM.v_split)
    assert 0.46 < np.mean(s == "select") < 0.54
    assert np.array_equal(s, VM.per_match(mids, VM.v_split))
    f = VM.per_match(mids, VM.cv_fold)
    assert set(f.tolist()) == set(range(5)) and all(700 < c < 900 for c in np.bincount(f))
    inner = VM.per_match(mids, VM.inner_holdout)
    assert 0.08 < inner.mean() < 0.12


# ------------------------------------------------------------------ synthetic end to end
def _write_fake_extract(d: Path, patch: str, n_matches: int, seed: int, guard, h_from: Path = None,
                        game_end_of=None, sample: bool = True, data_kw=None) -> None:
    """Fake ev4_02 extract.  15.14 writes h_distribution.npy; 15.15 links its h source to h_from (a 15.14 dir).
    game_end_of(match_id) -> GAME_END (default 2,000,000) goes into the V and martingale index tables.
    sample=False writes a 'full' (non-sample) manifest, which ev4_03 accepts without --smoke; data_kw goes to
    _antisymmetric_data."""
    import pandas as pd
    ge = game_end_of or (lambda m: 2_000_000)
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    chunks = {}
    for c in range(2):
        mids = [f"SYN{patch}_{c}_{i}" for i in range(n_matches)]
        rows = []
        for m in mids:
            for k in range(6):
                rows.append({"match_id": m, "bucket": k, "t": 120_000 * (k + 1) + int(rng.integers(0, 120_000))})
        X, y = _antisymmetric_data(len(rows), seed + c, **(data_kw or {}))
        X[:, NAMES.index("time_minutes")] = [r["t"] / 60000 for r in rows]
        X[:, NAMES.index("snapshot_age_s")] = rng.uniform(0, 60, len(rows))       # frame ages over the 4 age bins
        mv = pd.DataFrame(rows)
        mv.insert(0, "row", np.arange(len(rows)))
        mv["game_end"] = [ge(r["match_id"]) for r in rows]
        mv["y_blue_win"] = y.astype(np.int64)
        n_m = n_matches if patch == "15.15" else 0
        m0 = X[:n_m].copy()
        m1 = m0 + rng.normal(0, 0.05, m0.shape).astype(np.float32)
        m1[:, NAMES.index("snapshot_age_s")] = rng.uniform(0, 60, n_m)
        mm = pd.DataFrame({"row": np.arange(n_m), "match_id": mids[:n_m], "t": 300_000, "h_ms": 50_000,
                           "game_end": [ge(m) for m in mids[:n_m]],
                           "frame_update": np.arange(n_m) % 2, "recent_deaths_ge3": (np.arange(n_m) % 4 == 0).astype(int)})
        stem = f"chunk_{c:05d}"
        np.savez_compressed(d / f"{stem}.npz", v_X=X, m_X0=m0, m_X1=m1, state_columns=np.array(NAMES))
        mv.to_parquet(d / f"{stem}_v.parquet", index=False)
        mm.to_parquet(d / f"{stem}_mart.parquet", index=False)
        files = {fn: {"sha256": VM.sha256_file(d / fn), "bytes": (d / fn).stat().st_size}
                 for fn in (f"{stem}.npz", f"{stem}_v.parquet", f"{stem}_mart.parquet")}
        (d / f"{stem}.json").write_text(json.dumps({"chunk": c, "files": files}), encoding="utf-8")
        chunks[f"{c:05d}"] = {"rows": {"v": len(rows), "mart": n_m}, "files": files}
    man = {"patch": patch, "STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "state_version": STATE_VERSION, "sample": sample,
           "n_matches": 2 * n_matches, "chunks": chunks,
           "guards": {"params": json.loads(json.dumps(guard["params"])), "record1a_sha256": guard["record1a_sha256"]},
           "remakes": {**FV.EC.remake_rule(), "n_excluded": 0, "match_ids": []}, "code_sha256": FV.EX.code_hashes()}
    if patch == "15.14":
        np.save(d / "h_distribution.npy", np.array([20_000, 50_000, 90_000], dtype=np.int64))
        man["h_distribution"] = {"file": "h_distribution.npy", "sha256": VM.sha256_file(d / "h_distribution.npy"), "n": 3}
    elif h_from is not None:
        hm = json.loads((h_from / "manifest.json").read_text(encoding="utf-8"))
        man["h_source"] = {"path": str(h_from / "h_distribution.npy"), "sha256": hm["h_distribution"]["sha256"],
                           "n": 3, "source_sample": hm["sample"], "source_n_matches": hm["n_matches"]}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


@pytest.mark.slow
@pytest.mark.skipif(not FV.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_end_to_end_synthetic(tmp_path):
    _, _, guard = FV.EX.load_params()
    _write_fake_extract(tmp_path / "e14", "15.14", 150, 20, guard)
    _write_fake_extract(tmp_path / "e15", "15.15", 150, 40, guard, h_from=tmp_path / "e14")
    out = tmp_path / "fit"
    rep = FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--smoke", "--out", str(out),
                   "--lgbm-rounds", "40", "--lgbm-patience", "5", "--mlp-epochs", "2", "--mart-boot", "19",
                   "--threads", "2"])
    man = json.loads((out / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert man["chosen"] == rep["selection"]["chosen"] in VM.SIMPLICITY_ORDER
    V = VM.load_v(out / "V_frozen", expected_sha256=man["V_frozen"]["bundle_sha256"])
    assert V.kind == man["chosen"]
    # V revision: side marker in every candidate; conditional recalibration by the pre-specified rules
    assert V.side_marker and V.bundle_format == VM.BUNDLE_FORMAT == man["V_frozen"]["bundle_format"]
    for k in man["candidates"]:
        assert VM.read_bundle_structure(out / "candidates" / k)["side_marker"]
    assert rep["census"]["side_minus_rows"] == rep["census"]["swapped_rows"]
    vr = rep["v_revision"]
    assert vr["record"]["sha256"] == FV.V_REVISION_SHA256 == man["v_revision"]["record"]["sha256"]
    assert vr["side_marker"]["added"] and vr["side_marker"]["n_model_columns"] == len(NAMES) + 1
    rc = vr["recalibration"]
    ch = man["chosen"]
    want_trig = VM.recal_trigger(rep["martingale"][ch]["a_means"], rep["martingale"][ch]["b_p_primary_wild_bootstrap"])
    assert rc["triggered"] == want_trig["triggered"] == man["v_revision"]["recalibration"]["triggered"]
    assert V.recalibrated == rc["adopted"] == man["V_frozen"]["recalibrated"]
    if rc["fitted"]:
        ad = rc["adoption"]
        assert ad["adopted"] == (ad["select_logloss_recalibrated"] - ad["select_logloss_side_marker"] <= 0.0005 + 1e-12)
        assert ad["select_logloss_side_marker"] == pytest.approx(rep["e4"][ch]["cal"]["select"]["all"]["log_loss"])
        assert rc["recalibrated_v"]["martingale"]["b_p_primary_wild_bootstrap"] is not None
    assert man["V_frozen"]["bundle_sha256"] == (rc["bundle"]["bundle_sha256"] if rc["adopted"] else man["candidates"][ch])
    assert set(man["candidates"]) == set(VM.SIMPLICITY_ORDER)
    assert rep["census"]["select_rows"] + rep["census"]["cal_rows"] == 2 * 150 * 6
    assert rep["e4"]["gold"]["raw"]["select"]["all"]["auc"] > 0.6          # gold difference drives the synthetic target
    assert rep["martingale"][man["chosen"]]["primary"]
    eff = rep["martingale"][man["chosen"]]["b_efficiency"]
    assert eff["wild_bootstrap"]["n_boot"] == 19 == rep["martingale"]["b_boot_replicates"]
    assert rep["martingale"][man["chosen"]]["b_p_primary_wild_bootstrap"] == eff["wild_bootstrap"]["p_value"] == eff["p_value_primary"]
    assert rep["martingale"][man["chosen"]]["b_p_secondary_F"] == eff["p_value_F"]
    other = [k for k in VM.SIMPLICITY_ORDER if k != man["chosen"]][0]
    assert rep["martingale"][other]["b_p_primary_wild_bootstrap"] is None and rep["martingale"][other]["b_p_secondary_F"] is not None
    # decisions: script decisions + the whole author record with path and sha256
    for block in (rep["decisions"], man["decisions"]):
        pr = block["predecisions_record"]
        assert pr["sha256"] == FV.EC.PREDECISIONS_SHA256 and pr["path"] == str(FV.EC.PREDECISIONS_RECORD)
        assert pr["technical_defaults"]["recent_deaths_window"] == "kills in (t - 60 s, t]"
        assert "deviation_side_marker" in block and "F12_stop_rule" in block
    assert rep["input_checks"]["h_source"]["ok"] and rep["input_checks"]["code"]["drift_found"] is False
    assert rep["census"]["remake_rows_found"] == {"train_v": 0, "select_v": 0, "martingale": 0}
    st = rep["stop_rule"]
    assert st["smoke"] and st["stop_before_record1"] is False and st["beats_gold"] == (
        st["chosen_raw_select_logloss"] < st["gold_raw_select_logloss"])
    assert rep["frozen"] is man["frozen"] is True and rep["pilot"] is man["pilot"] is False
    assert not (out / "V_NOT_FROZEN.md").exists()
    assert VM.assert_v_usable(out, allow_smoke=True)["bundle_sha256"] == man["V_frozen"]["bundle_sha256"]
    with pytest.raises(VM.VNotUsable, match="smoke"):
        VM.assert_v_usable(out / "frozen_manifest.json")                   # a smoke V is not for full stages
    assert (out / "predictions_v_15.15.parquet").is_file() and (out / "predictions_mart_15.15.parquet").is_file()
    for rel, h in man["files_sha256"].items():
        assert VM.sha256_file(out / rel) == h
    # a tampered chunk file is refused on the next run
    npz = tmp_path / "e14" / "chunk_00000.npz"
    npz.write_bytes(npz.read_bytes() + b"0")
    with pytest.raises(RuntimeError):
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--smoke", "--out", str(out)])


# ------------------------------------------------------------------ F10 / F11: remakes, h source, code drift
def _linked_manifests(tmp_path, sample=False, n=100):
    """Minimal 15.14 / 15.15 manifests with a real h_distribution.npy and matching linkage (no V arrays)."""
    a = tmp_path / "a"
    a.mkdir(parents=True, exist_ok=True)
    np.save(a / "h_distribution.npy", np.array([1, 2, 3], dtype=np.int64))
    hsha = VM.sha256_file(a / "h_distribution.npy")
    common = {"remakes": FV.EC.remake_rule(), "code_sha256": FV.EX.code_hashes()}
    _fake_manifest(a, "15.14", sample=sample, n_matches=n, h_distribution={"file": "h_distribution.npy", "sha256": hsha},
                   **common)
    b = _fake_manifest(tmp_path / "b", "15.15", sample=sample, n_matches=n,
                       h_source={"path": str(a / "h_distribution.npy"), "sha256": hsha, "source_sample": sample,
                                 "source_n_matches": n}, **common)
    return a, b


def _edit(d: Path, **kw):
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k] = {**m[k], **v}
        else:
            m[k] = v
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_mart_boot_full_run_is_999_and_bootstrap_is_primary():
    assert FV.MART_BOOT == 999
    assert "PRIMARY p = wild cluster" in FV.DECISIONS["F8_martingale"]
    assert FV.build_arg_parser().parse_args(["--train", "a", "--select", "b"]).mart_boot is None   # -> MART_BOOT


def test_technical_defaults_match_code():
    rec = FV.EC.predecisions_info()["content"]
    assert rec["technical_defaults"]["V_selection"].startswith("raw (uncalibrated) V_SELECT log loss, tie within 0.0005")
    assert VM.SELECT_TOL == 0.0005 and VM.SIMPLICITY_ORDER == ("logistic", "lgbm", "mlp")
    assert VM.MLP_SEEDS_CV == (0,) and len(VM.MLP_SEEDS_FINAL) == 3
    assert FV.EX.RECENT_DEATH_WINDOW_MS == 60_000 and FV.EX.V_BUCKET_MS == 120_000
    assert "remakes" in rec and "300,000" in rec["remakes"]


def test_check_inputs_accepts_linked_full_extracts(tmp_path):
    a, b = _linked_manifests(tmp_path, sample=False)
    mt, ms, checks = FV.check_inputs(a, b, smoke=False)
    assert checks["h_source"]["ok"] and checks["code"]["drift_found"] is False
    assert checks["remake_rule"]["train"]["threshold_ms"] == 300_000


def test_check_inputs_refuses_sample_h_source_without_smoke(tmp_path):
    a, b = _linked_manifests(tmp_path, sample=False)
    _edit(b, h_source={"source_sample": True})
    with pytest.raises(RuntimeError, match="sample h distribution"):
        FV.check_inputs(a, b, smoke=False)
    FV.check_inputs(a, b, smoke=True)                                       # a smoke may link sample to sample
    _edit(b, h_source={"source_sample": False, "source_n_matches": 99})
    with pytest.raises(RuntimeError, match="matches"):
        FV.check_inputs(a, b, smoke=False)


def test_check_inputs_refuses_h_source_sha_mismatch(tmp_path):
    a, b = _linked_manifests(tmp_path)
    _edit(b, h_source={"sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="h source"):
        FV.check_inputs(a, b, smoke=True)                                   # enforced in smoke too
    a, b = _linked_manifests(tmp_path / "2")
    np.save(a / "h_distribution.npy", np.array([1, 2, 4], dtype=np.int64))  # file no longer the one in the manifest
    with pytest.raises(RuntimeError, match="does not hash"):
        FV.check_inputs(a, b, smoke=False)
    a, b = _linked_manifests(tmp_path / "3")
    _edit(b, h_source={"path": str(a / "other.npy")})
    with pytest.raises(RuntimeError, match="not an h_distribution.npy"):
        FV.check_inputs(a, b, smoke=False)
    a, b = _linked_manifests(tmp_path / "4")
    _edit(b, h_source=None)
    with pytest.raises(RuntimeError):
        FV.check_inputs(a, b, smoke=True)


def test_check_inputs_code_drift_refused_unless_allowed(tmp_path):
    a, b = _linked_manifests(tmp_path)
    _edit(b, code_sha256={"scripts/exact_v4/ev4_02_extract.py": "f" * 64})
    with pytest.raises(RuntimeError, match="extract code differs"):
        FV.check_inputs(a, b, smoke=False)
    _, _, checks = FV.check_inputs(a, b, smoke=False, allow_code_drift=True)
    assert checks["code"]["drift_found"] and checks["code"]["allow_code_drift"]
    assert list(checks["code"]["drift"]) == ["select"]
    assert checks["code"]["drift"]["select"]["scripts/exact_v4/ev4_02_extract.py"]["extract"] == "f" * 64
    a, b = _linked_manifests(tmp_path / "2")
    m = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    del m["code_sha256"]["scripts/exact_v4/ev4_common.py"]                  # extract built before ev4_common existed
    (a / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(RuntimeError, match="extract code differs"):
        FV.check_inputs(a, b, smoke=True)
    assert "--allow-code-drift" in FV.build_arg_parser().format_help()


def test_check_inputs_refuses_extract_without_remake_rule(tmp_path):
    a, b = _linked_manifests(tmp_path)
    _edit(a, remakes=None)
    with pytest.raises(RuntimeError, match="remake rule"):
        FV.check_inputs(a, b, smoke=True)
    a, b = _linked_manifests(tmp_path / "2")
    _edit(b, remakes={"threshold_ms": 420_000})
    with pytest.raises(RuntimeError, match="remake rule"):
        FV.check_inputs(a, b, smoke=True)


def test_assert_no_remakes():
    import pandas as pd
    ok = pd.DataFrame({"match_id": ["a", "b"], "game_end": [300_000, 2_000_000]})
    assert FV.assert_no_remakes({"v": ok, "empty": pd.DataFrame()}) == {"v": 0, "empty": 0}
    bad = pd.DataFrame({"match_id": ["a", "r", "r"], "game_end": [2_000_000, 299_999, 299_999]})
    with pytest.raises(RuntimeError, match="1 remake matches"):
        FV.assert_no_remakes({"v": bad})
    with pytest.raises(RuntimeError, match="game_end"):
        FV.assert_no_remakes({"v": pd.DataFrame({"match_id": ["a"]})})


@pytest.mark.skipif(not FV.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_main_refuses_remake_rows_in_extract(tmp_path):
    _, _, guard = FV.EX.load_params()
    ge = lambda m: 250_000 if m.endswith("_0_3") else 2_000_000     # noqa: E731  one remake match in chunk 0
    _write_fake_extract(tmp_path / "e14", "15.14", 20, 20, guard, game_end_of=ge)
    _write_fake_extract(tmp_path / "e15", "15.15", 20, 40, guard, h_from=tmp_path / "e14")
    with pytest.raises(RuntimeError, match="remake"):
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--smoke",
                 "--out", str(tmp_path / "o"), "--candidates", "logistic"])
    assert not (tmp_path / "o" / "report_e4.json").exists()


# ------------------------------------------------------------------ F12 stop rule enforcement, F13 pilot, pre-decisions
def test_stop_rule_full_size_and_informational_modes():
    lose = FV.stop_rule({"logistic": 0.60, "mlp": 0.62}, 0.50, "logistic", smoke=False)
    assert lose["beats_gold"] is False and lose["would_stop_at_full_size"] is True
    assert lose["stop_before_record1"] is True and lose["smoke"] is False and lose["pilot"] is False
    win = FV.stop_rule({"logistic": 0.45}, 0.50, "logistic", smoke=False)
    assert win["beats_gold"] is True and win["stop_before_record1"] is False and win["would_stop_at_full_size"] is False
    tie = FV.stop_rule({"lgbm": 0.5}, 0.5, "lgbm", smoke=False)                # must BEAT gold: a tie stops
    assert tie["stop_before_record1"] is True
    for kw in ({"smoke": True}, {"smoke": False, "pilot": True}):              # informational only
        st = FV.stop_rule({"logistic": 0.60}, 0.50, "logistic", **kw)
        assert st["stop_before_record1"] is False and st["would_stop_at_full_size"] is True
    assert FV.stop_rule({"logistic": 0.6}, 0.5, "logistic", smoke=False, pilot=True)["pilot"] is True
    assert FV.EXIT_STOP_RULE == 3 and "exit code 3" in FV.DECISIONS["F12_stop_rule"]


def _good_manifest(**kw):
    m = {"smoke": False, "pilot": False, "frozen": True, "not_frozen_reason": None,
         "stop_rule": {"chosen": "logistic", "stop_before_record1": False, "beats_gold": True},
         "V_frozen": {"dir": "x/V_frozen", "bundle_sha256": "a" * 64, "kind": "logistic"}}
    m.update(kw)
    return m


def test_assert_v_usable(tmp_path):
    assert VM.assert_v_usable(_good_manifest())["bundle_sha256"] == "a" * 64
    bad = [_good_manifest(stop_rule={"chosen": "logistic", "stop_before_record1": True, "beats_gold": False}),
           _good_manifest(stop_rule={"chosen": "logistic", "stop_before_record1": None}),
           _good_manifest(stop_rule=None),
           _good_manifest(pilot=True),
           _good_manifest(frozen=False, not_frozen_reason="stop_rule"),
           _good_manifest(V_frozen=None),
           _good_manifest(V_frozen={"dir": "x"})]
    for m in bad:
        with pytest.raises(VM.VNotUsable):
            VM.assert_v_usable(m)
    with pytest.raises(VM.VNotUsable, match="stop rule"):
        VM.assert_v_usable(bad[0], allow_smoke=True)
    with pytest.raises(VM.VNotUsable, match="smoke"):
        VM.assert_v_usable(_good_manifest(smoke=True))
    assert VM.assert_v_usable(_good_manifest(smoke=True), allow_smoke=True)["kind"] == "logistic"
    with pytest.raises(VM.VNotUsable, match="pilot"):
        VM.assert_v_usable(_good_manifest(smoke=True, pilot=True), allow_smoke=True)
    (tmp_path / "frozen_manifest.json").write_text(json.dumps(_good_manifest()), encoding="utf-8")
    assert VM.assert_v_usable(tmp_path)["kind"] == "logistic"                     # directory or file path
    assert VM.assert_v_usable(tmp_path / "frozen_manifest.json")["kind"] == "logistic"
    with pytest.raises(VM.VNotUsable, match="missing"):
        VM.assert_v_usable(tmp_path / "nowhere")


def _record_copies(tmp_path):
    """An edited copy of the author pre-decision record and a path that does not exist (the real record is untouched)."""
    edited = tmp_path / "stage2_predecisions_edited.json"
    edited.write_bytes(FV.EC.PREDECISIONS_RECORD.read_bytes() + b" ")
    return edited, tmp_path / "stage2_predecisions_missing.json"


@pytest.mark.skipif(not FV.EC.PREDECISIONS_RECORD.is_file(), reason="pre-decision record not present")
def test_predecisions_refuses_edited_or_missing_record(tmp_path, monkeypatch):
    info = FV.EC.predecisions_info()
    assert info["sha256"] == FV.EC.PREDECISIONS_SHA256 and info["path"] == str(FV.EC.PREDECISIONS_RECORD)
    edited, missing = _record_copies(tmp_path)
    with pytest.raises(RuntimeError, match="pre-decision record"):
        FV.EC.predecisions_info(edited)
    with pytest.raises(FileNotFoundError):
        FV.EC.predecisions_info(missing)
    # ev4_03 main refuses before any input check (no manifest is read, no output directory is made)
    a = _fake_manifest(tmp_path / "a", "15.14")
    b = _fake_manifest(tmp_path / "b", "15.15")

    def boom(*_a, **_k):
        raise AssertionError("inputs touched before the pre-decision record was checked")

    monkeypatch.setattr(FV, "check_inputs", boom)
    for rec, exc in ((edited, RuntimeError), (missing, FileNotFoundError)):
        monkeypatch.setattr(FV.EC, "PREDECISIONS_RECORD", rec)
        with pytest.raises(exc):
            FV.main(["--train", str(a), "--select", str(b), "--smoke", "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def _small_budget(monkeypatch):
    """Shrink the FULL-run budget constants (not the --smoke overrides) so a non-smoke synthetic run is quick."""
    monkeypatch.setattr(FV.VM, "LGBM_MAX_ROUNDS", 30)
    monkeypatch.setattr(FV.VM, "LGBM_PATIENCE", 5)
    monkeypatch.setattr(FV.VM, "MLP_MAX_EPOCHS", 2)
    monkeypatch.setitem(FV.VM.GRIDS["logistic"]["fixed"], "maxiter", 60)
    monkeypatch.setattr(FV, "MART_BOOT", 19)


GOLD_ONLY = {"gold_coef": 1.5, "kill_coef": 0.0}          # the gold-difference baseline is the true model


@pytest.mark.slow
@pytest.mark.skipif(not FV.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_stop_rule_full_run_writes_no_v_frozen_and_exits_nonzero(tmp_path, monkeypatch):
    _small_budget(monkeypatch)
    _, _, guard = FV.EX.load_params()
    # >= 100 matches per chunk: with min_data_in_leaf 200 a smaller LightGBM cannot split, is constant, and its
    # martingale dV is identically 0 (the efficiency regression's R^2 then divides by zero)
    _write_fake_extract(tmp_path / "e14", "15.14", 100, 20, guard, sample=False, data_kw=GOLD_ONLY)
    _write_fake_extract(tmp_path / "e15", "15.15", 100, 40, guard, h_from=tmp_path / "e14", sample=False,
                        data_kw=GOLD_ONLY)
    out = tmp_path / "fit"
    (out / "V_frozen").mkdir(parents=True)                                   # stale V of an earlier run
    (out / "V_frozen" / "bundle.json").write_text("{}", encoding="utf-8")
    (out / "frozen_manifest.json").write_text(json.dumps(_good_manifest()), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:                                    # no --smoke: a full-size run
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--out", str(out),
                 "--threads", "2"])
    assert ei.value.code == FV.EXIT_STOP_RULE != 0
    rep = json.loads((out / "report_e4.json").read_text(encoding="utf-8"))
    man = json.loads((out / "frozen_manifest.json").read_text(encoding="utf-8"))
    st = rep["stop_rule"]
    assert rep["smoke"] is False and rep["pilot"] is False
    assert st["beats_gold"] is False and st["stop_before_record1"] is True
    assert rep["frozen"] is man["frozen"] is False and man["not_frozen_reason"] == "stop_rule"
    assert man["V_frozen"] is None and man["stop_rule"] == st
    assert not (out / "V_frozen").exists()                                  # the stale V is gone, no new one
    note = (out / "V_NOT_FROZEN.md").read_text(encoding="utf-8")
    assert "stop rule" in note and "Do NOT write record 1" in note and st["chosen"] in note
    chosen = st["chosen"]                                                    # candidates kept for diagnosis
    VM.load_v(out / "candidates" / chosen, expected_sha256=man["candidates"][chosen])
    assert rep["budget"]["lgbm_rounds"] == 30 and rep["budget"]["mart_boot"] == 19   # full budget constants
    assert (out / "predictions_v_15.15.parquet").is_file() and rep["martingale"][chosen]["primary"]
    assert "V_frozen/bundle.json" not in man["files_sha256"] and "V_NOT_FROZEN.md" in man["files_sha256"]
    for src in (out, out / "frozen_manifest.json", man):
        with pytest.raises(VM.VNotUsable, match="stop rule"):
            VM.assert_v_usable(src)
    assert "STOP RULE" in (out / "run.log").read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.skipif(not FV.EX.RECORD1A.is_file(), reason="record 1A not present")
def test_pilot_subsets_by_sha_order_and_never_freezes(tmp_path, monkeypatch):
    import pandas as pd
    _small_budget(monkeypatch)
    _, _, guard = FV.EX.load_params()
    # >= 100 matches per chunk: with min_data_in_leaf 200 a smaller LightGBM cannot split, is constant, and its
    # martingale dV is identically 0 (the efficiency regression's R^2 then divides by zero)
    _write_fake_extract(tmp_path / "e14", "15.14", 100, 20, guard, sample=False, data_kw=GOLD_ONLY)
    _write_fake_extract(tmp_path / "e15", "15.15", 100, 40, guard, h_from=tmp_path / "e14", sample=False,
                        data_kw=GOLD_ONLY)
    n = 120
    want = {}
    for p in ("15.14", "15.15"):
        ids = [f"SYN{p}_{c}_{i}" for c in range(2) for i in range(100)]
        want[p] = sorted(ids, key=lambda m: (hashlib.sha256(m.encode()).hexdigest(), m))[:n]
    assert FV.pilot_match_ids(tmp_path / "e15", FV.read_manifest(tmp_path / "e15"), n) == (want["15.15"], 200)
    out = tmp_path / "pilot"
    rep = FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--out", str(out),
                   "--threads", "2", "--pilot-matches", str(n)])          # returns: a pilot never exits non-zero
    man = json.loads((out / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert rep["pilot"] is man["pilot"] is True and rep["smoke"] is False
    assert rep["frozen"] is man["frozen"] is False and man["not_frozen_reason"] == "pilot" and man["V_frozen"] is None
    assert not (out / "V_frozen").exists() and "pilot run" in (out / "V_NOT_FROZEN.md").read_text(encoding="utf-8")
    ps = rep["pilot_subset"]
    assert ps["n_matches_requested"] == n and ps["train"]["n_matches"] == ps["select"]["n_matches"] == n
    assert ps["train"]["match_ids_sha256"] == hashlib.sha256("\n".join(want["15.14"]).encode()).hexdigest()
    assert ps["select"]["match_ids_sha256"] == hashlib.sha256("\n".join(want["15.15"]).encode()).hexdigest()
    assert rep["census"]["train_matches"] == n and rep["census"]["train_rows"] == 6 * n
    assert rep["census"]["select_rows"] + rep["census"]["cal_rows"] == 6 * n and rep["census"]["mart_rows"] == n
    pv = pd.read_parquet(out / "predictions_v_15.15.parquet")
    pm = pd.read_parquet(out / "predictions_mart_15.15.parquet")
    assert set(pv["match_id"]) == set(pm["match_id"]) == set(want["15.15"])
    # same rules as a full run: full budget, fixed grids, all candidates, same selection; stop rule informational
    assert rep["budget"]["lgbm_rounds"] == 30 and rep["budget"]["mart_boot"] == 19 and rep["grids"] == VM.GRIDS
    assert rep["candidates"] == list(VM.SIMPLICITY_ORDER)
    assert rep["selection"] == VM.select_candidate({k: rep["e4"][k]["raw"]["select"]["all"]["log_loss"]
                                                    for k in VM.SIMPLICITY_ORDER})
    st = rep["stop_rule"]
    assert st["pilot"] and st["stop_before_record1"] is False and st["would_stop_at_full_size"] == (not st["beats_gold"])
    with pytest.raises(VM.VNotUsable, match="pilot"):
        VM.assert_v_usable(out)
    # a pilot refuses an --out that holds a frozen V; budget overrides still need --smoke; N must be >= 1
    (tmp_path / "frozen_run" / "V_frozen").mkdir(parents=True)
    with pytest.raises(SystemExit):
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--out",
                 str(tmp_path / "frozen_run"), "--pilot-matches", "5"])
    assert (tmp_path / "frozen_run" / "V_frozen").is_dir() and not (tmp_path / "frozen_run" / "report_e4.json").exists()
    with pytest.raises(SystemExit):
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--out", str(tmp_path / "x"),
                 "--pilot-matches", "5", "--mlp-epochs", "1"])
    with pytest.raises(SystemExit):
        FV.main(["--train", str(tmp_path / "e14"), "--select", str(tmp_path / "e15"), "--out", str(tmp_path / "x"),
                 "--pilot-matches", "0"])
    assert FV.DEFAULT_OUT_PILOT.name == "fit_v_pilot" and FV.DEFAULT_OUT_PILOT != FV.DEFAULT_OUT
