"""Tests for the V revision (records/v_revision_prespec_20260925T233953Z.json) in scripts/exact_v4/ev4_v_models.py,
ev4_03_fit_v.py, ev4_03b_oof_v.py and the consumers ev4_04_labels.py / ev4_r1_record.py:

  * side marker: the 'side' column (+1 original, -1 swapped) flips with the team swap; an antisymmetric model gives
    exactly 1 - p for (swapped blue / red, flipped side); trained models are nearly so and learn the blue advantage;
  * conditional recalibration: age bins, natural spline, recovery of a known model, trigger and adoption rules;
  * frozen bundles: format 2 with side marker / recalibration applied inside predict(); format-1 bundles still load;
  * synthetic end to end (slow): forced adoption / no trigger, OOF folds reproduce the frozen structure, labels
    record the structure.

All data are synthetic except the optional read-only load of the first full fit-V bundle (format 1).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
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
from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH  # noqa: E402
from tests.test_ev4_fit_v import _antisymmetric_data, _write_fake_extract  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FV = _load("ev4_03_fit_v")
OB = _load("ev4_03b_oof_v")
LB = OB.LB

NAMES = list(STATE_V3_COLUMNS)                    # StateV3 (996)
NS = len(NAMES)
MNAMES = VM.with_side(NAMES)                      # model input (997)
PERM = VM.swap_permutation(MNAMES)
NEG = VM.swap_negate_idx(MNAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
AGE, MIN = IDX["snapshot_age_s"], IDX["time_minutes"]
FIRST_FIT = FV.EX.OUT_BASE / "stage2" / "fit_v" / "V_frozen"          # first full fit (format 1), read only
HAVE_R1A = FV.EX.RECORD1A.is_file()


def _with_side(X: np.ndarray, side=1.0) -> np.ndarray:
    Xm = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float32)
    Xm[:, :-1] = X
    Xm[:, -1] = side
    return Xm


def _swap_state(X: np.ndarray) -> np.ndarray:
    """Blue <-> red of a StateV3 matrix (the side column is not part of it)."""
    return X[:, VM.swap_permutation(NAMES)]


def _rows(n, seed, bias=0.6, dense=False):
    X, y = _antisymmetric_data(n, seed, dense=dense, bias=bias)
    rng = np.random.default_rng(seed + 1000)
    X[:, AGE] = rng.uniform(0, 60, n)
    X[:, MIN] = rng.uniform(2, 40, n)
    return X, y


# ------------------------------------------------------------------ side marker: swap plumbing
def test_side_column_names_and_negation():
    assert MNAMES[-1] == VM.SIDE_COLUMN and len(MNAMES) == NS + 1
    assert NEG.tolist() == [NS] and VM.swap_negate_idx(NAMES).size == 0
    assert PERM[NS] == NS                                         # side maps to itself, sign flips separately
    with pytest.raises(ValueError):
        VM.with_side(MNAMES)


def test_apply_swap_flips_side_and_is_involution():
    rng = np.random.default_rng(0)
    X = _with_side(rng.normal(size=(300, NS)).astype(np.float32))
    y = (rng.random(300) < 0.5).astype(np.float64)
    X0, y0 = X.copy(), y.copy()
    m = VM.swap_mask(300, seed=4)
    VM.apply_swap_inplace(X, y, m, PERM, chunk=41, negate=NEG)
    assert np.all(X[m, NS] == -1) and np.all(X[~m, NS] == 1)
    r = np.flatnonzero(m)[0]
    assert np.array_equal(X[r, :NS], X0[r, :NS][VM.swap_permutation(NAMES)]) and y[r] == 1 - y0[r]
    VM.apply_swap_inplace(X, y, m, PERM, negate=NEG)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


def test_standardizer_pools_side_over_the_swap():
    rng = np.random.default_rng(1)
    X = _with_side(rng.normal(size=(400, NS)).astype(np.float32))
    X[:150, NS] = -1                                              # unbalanced sides: still mean 0, sd 1
    for groups in (VM.swap_pairs(PERM), VM.mlp_groups(MNAMES, PERM)):
        mu, sd = VM.pooled_standardizer(X, None, groups, negate=NEG)
        assert mu[NS] == 0.0 and sd[NS] == pytest.approx(1.0)
        assert np.allclose(mu[PERM[:NS]], mu[:NS]) and np.allclose(sd[PERM], sd)
    mu0, sd0 = VM.pooled_standardizer(X, None, VM.swap_pairs(PERM))              # without negate: raw moments
    assert mu0[NS] == pytest.approx((250 - 150) / 400)
    slot_idx, other, _ = VM.player_layout(MNAMES)
    assert NS in other.tolist() and len(other) == 277


# ------------------------------------------------------------------ side marker: symmetry of V
def _antisym_logistic(seed=3) -> VM.LogisticModel:
    """Logistic V with exactly antisymmetric weights (w[perm] = -w on the StateV3 part, any side weight, b = 0)."""
    rng = np.random.default_rng(seed)
    perm_s = VM.swap_permutation(NAMES)
    w = rng.normal(0, 0.05, NS)
    w = (w - w[perm_s]) / 2
    m = VM.LogisticModel(0.01)
    m.w = np.concatenate([w, [0.4]])                              # blue-side advantage on 'side'
    m.b = 0.0
    m.mu, m.sd = np.zeros(NS + 1), np.ones(NS + 1)
    m.fit_info = {}
    return m


def test_swapping_teams_and_flipping_side_gives_one_minus_p(tmp_path):
    m = _antisym_logistic()
    cal = VM.PositiveSlopeSigmoid.from_dict({"intercept_a": 0.0, "slope_b": 1.0})
    info = VM.save_bundle(tmp_path / "v", m, cal, NAMES, np.zeros(NS), side_marker=True)
    V = VM.load_v(tmp_path / "v", expected_sha256=info["bundle_sha256"], expected_columns_hash=STATE_V3_NAME_HASH)
    X, _ = _rows(500, 5)
    p = V.predict(X)                                              # side = +1 by default (blue perspective)
    p_swap = V.predict(_swap_state(X), side=-1)
    assert np.max(np.abs(p + p_swap - 1)) < 1e-6
    assert np.allclose(V.predict(X, side=1.0), p) and np.allclose(V.predict(X, side=np.ones(len(X))), p)
    # the side weight is the blue advantage: the same state seen as the swapped (red) side is worth less
    assert np.all(V.predict(X, side=-1) < p)
    with pytest.raises(ValueError):
        V.predict(X[:3], side=0.5)


def test_logistic_with_side_marker_learns_blue_advantage_and_is_nearly_symmetric():
    X, y = _rows(20000, 2, bias=0.6)                              # true blue-side advantage: +0.6 logit
    Xm = _with_side(X)
    VM.apply_swap_inplace(Xm, y, VM.swap_mask(len(y), seed=5), PERM, negate=NEG)
    mu, sd = VM.pooled_standardizer(Xm, None, VM.swap_pairs(PERM), negate=NEG)
    lm = VM.LogisticModel(0.01).fit(Xm, y, np.arange(len(y)), mu, sd)
    assert 0.4 < lm.w[NS] < 0.8 and abs(lm.b) < 0.1               # the side column carries the advantage
    Xt, yt = _rows(2000, 9, bias=0.6)
    p = lm.predict_proba(_with_side(Xt, 1.0))
    ps = lm.predict_proba(_with_side(_swap_state(Xt), -1.0))
    assert np.mean(np.abs(p + ps - 1)) < 0.03                     # nearly antisymmetric (50 % swap, not copies)
    # without the side marker the blue advantage is averaged out: worse on natural (blue-perspective) rows
    X0, y0 = X.copy(), _rows(20000, 2, bias=0.6)[1]
    VM.apply_swap_inplace(X0, y0, VM.swap_mask(len(y0), seed=5), VM.swap_permutation(NAMES))
    mu0, sd0 = VM.pooled_standardizer(X0, None, VM.swap_pairs(VM.swap_permutation(NAMES)))
    l0 = VM.LogisticModel(0.01).fit(X0, y0, np.arange(len(y0)), mu0, sd0)
    assert VM.log_loss(yt, p) < VM.log_loss(yt, l0.predict_proba(Xt)) - 0.01


def test_mlp_minibatch_swaps_flip_side():
    X, y = _rows(4000, 4, bias=0.6)
    Xm = _with_side(X)
    tr, st = np.arange(3600), np.arange(3600, 4000)
    mu, sd = VM.pooled_standardizer(Xm, tr, VM.mlp_groups(MNAMES, PERM), negate=NEG)
    assert mu[NS] == 0 and sd[NS] == 1                            # natural rows only (+1), pooled over the swap
    mm = VM.MLPModel(1e-4, MNAMES, seeds=(0,), max_epochs=6, patience=3, threads=2).fit(Xm, y, tr, st, mu, sd)
    assert mm.negate.tolist() == [NS]
    Xt, yt = _rows(800, 11, bias=0.6)
    p = mm.predict_proba(_with_side(Xt, 1.0))
    ps = mm.predict_proba(_with_side(_swap_state(Xt), -1.0))
    assert VM.log_loss(yt, p) < 0.66 and np.mean(np.abs(p + ps - 1)) < 0.1
    assert np.mean(p) > np.mean(mm.predict_proba(_with_side(Xt, -1.0)))   # learnt the blue advantage


# ------------------------------------------------------------------ recalibration model
def test_age_bins_and_natural_spline():
    assert VM.age_bin([-1, 0, 14.99, 15, 29.9, 30, 44.9, 45, 59.9, 60.007]).tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
    with pytest.raises(ValueError):
        VM.age_bin([np.nan])
    knots = [5.0, 12.0, 20.0, 32.0]
    x = np.linspace(-10, 60, 2801)
    B = VM.rcs_basis(x, knots)
    assert B.shape == (len(x), 3) and np.array_equal(B[:, 0], x)
    for lo, hi in ((-10, 5), (32, 60)):                           # linear beyond the boundary knots
        sel = (x >= lo) & (x <= hi)
        d2 = np.diff(B[sel, 1:], n=2, axis=0)
        assert np.max(np.abs(d2)) < 1e-9
    inner = (x > 5.5) & (x < 31.5)
    assert np.max(np.abs(np.diff(B[inner, 1:], n=2, axis=0))) > 1e-8   # cubic inside
    with pytest.raises(ValueError):
        VM.rcs_basis(x, [1, 1, 2, 3])


def _recal_data(n, seed, truth):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.03, 0.97, n)
    age = rng.uniform(0, 60, n)
    minute = rng.uniform(3, 40, n)
    r = VM.LogitRecalibrator(knots=np.quantile(minute, VM.RECAL_KNOT_QUANTILES))
    eta = r.design(p, age, minute) @ truth
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return p, age, minute, y


def test_recalibrator_recovers_known_model_and_roundtrips():
    truth = np.array([0.1, 0.0, -0.1, -0.15, 1.1, 1.0, 0.9, 0.8, -0.004, 0.01, -0.02])
    p, age, minute, y = _recal_data(120_000, 1, truth)
    rc = VM.LogitRecalibrator().fit(p, age, minute, y)
    assert rc.fit_info["converged"] and rc.fit_info["rank"] == 11 and len(rc.coef) == 11
    assert np.max(np.abs(rc.coef[:8] - truth[:8])) < 0.06
    assert rc.knots == pytest.approx(np.quantile(minute, VM.RECAL_KNOT_QUANTILES))
    pt, at, mt, yt = _recal_data(40_000, 2, truth)
    assert VM.log_loss(yt, rc.predict(pt, at, mt)) < VM.log_loss(yt, pt) - 0.002
    rc2 = VM.LogitRecalibrator.from_dict(json.loads(json.dumps(rc.to_dict())))
    assert np.array_equal(rc2.predict(pt, at, mt), rc.predict(pt, at, mt))
    assert rc.to_dict()["terms"] == VM.LogitRecalibrator.term_names() and len(rc.to_dict()["terms"]) == 11
    # identity truth: the recalibration is (close to) the identity map
    ident = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    p, age, minute, y = _recal_data(60_000, 3, ident)
    rc = VM.LogitRecalibrator().fit(p, age, minute, y)
    assert np.max(np.abs(rc.predict(p, age, minute) - p)) < 0.03
    with pytest.raises(ValueError, match="age bins"):                 # thin age bin: not identified
        VM.LogitRecalibrator().fit(p[:500], np.full(500, 3.0), minute[:500], y[:500])


def _a_means(passes):
    return {"strata": {s: {"pass": v} for s, v in passes.items()}}


@pytest.mark.parametrize("passes, b_p, triggered", [
    ({"all": True, "frame_update_yes": True, "frame_update_no": True, "recent_deaths_ge3": True}, 0.5, False),
    ({"all": True, "frame_update_yes": False, "frame_update_no": True, "recent_deaths_ge3": True}, 0.5, True),
    ({"all": True, "frame_update_yes": True, "frame_update_no": True, "recent_deaths_ge3": False}, 0.9, True),
    ({"all": True, "frame_update_yes": True, "frame_update_no": True, "recent_deaths_ge3": True}, 0.05, True),
    ({"all": True, "frame_update_yes": True, "frame_update_no": True, "recent_deaths_ge3": True}, 0.0501, False),
    ({"all": True, "frame_update_yes": None, "frame_update_no": True, "recent_deaths_ge3": True}, None, False),
    ({"all": True, "frame_update_yes": True, "frame_update_no": True, "recent_deaths_ge3": True}, 0.049, True),
])
def test_recal_trigger_rule(passes, b_p, triggered):
    t = VM.recal_trigger(_a_means(passes), b_p)
    assert t["triggered"] is triggered
    assert t["a_failed_strata"] == [s for s, v in passes.items() if v is False]
    assert t["a_untested_strata"] == [s for s, v in passes.items() if v is None]


def test_recal_adoption_rule():
    assert VM.recal_adopt(0.4700, 0.4700)["adopted"]
    assert VM.recal_adopt(0.4690, 0.4700)["adopted"]
    assert VM.recal_adopt(0.4705, 0.4700)["adopted"]                  # exactly 0.0005 worse: adopted
    assert not VM.recal_adopt(0.47051, 0.4700)["adopted"]
    a = VM.recal_adopt(0.5, 0.49)
    assert a["difference"] == pytest.approx(0.01) and a["tol"] == 0.0005 and not a["adopted"]


# ------------------------------------------------------------------ bundles: predict with recalibration, old format
def _fit_small(kind, Xm, y):
    if kind == "logistic":
        mu, sd = VM.pooled_standardizer(Xm, None, VM.swap_pairs(PERM), negate=NEG)
        return VM.LogisticModel(0.01).fit(Xm, y, np.arange(len(y)), mu, sd)
    if kind == "lgbm":
        ds = VM.LGBMModel.dataset(Xm, y)
        return VM.LGBMModel(31, max_rounds=30, patience=5, threads=2).fit(ds, np.arange(1300), np.arange(1300, 1500))
    mu, sd = VM.pooled_standardizer(Xm, None, VM.mlp_groups(MNAMES, PERM), negate=NEG)
    return VM.MLPModel(1e-3, MNAMES, seeds=(0, 1), max_epochs=2, threads=2).fit(Xm, y, np.arange(1300),
                                                                              np.arange(1300, 1500), mu, sd)


@pytest.mark.parametrize("kind", ["logistic", "lgbm", "mlp"])
def test_bundle_with_side_marker_and_recalibration_predicts_through_both(tmp_path, kind):
    X, y = _rows(1500, 12, dense=True)
    Xm = _with_side(X)
    VM.apply_swap_inplace(Xm, y, VM.swap_mask(len(y)), PERM, negate=NEG)
    m = _fit_small(kind, Xm, y)
    VM.apply_swap_inplace(Xm, y, VM.swap_mask(len(y)), PERM, negate=NEG)     # back to natural (side +1)
    cal = VM.PositiveSlopeSigmoid().fit(m.predict_proba(Xm), y, np.ones(len(y)))
    q = cal.predict(m.predict_proba(Xm))
    rc = VM.LogitRecalibrator().fit(q, X[:, AGE], X[:, MIN], y)
    fill = np.zeros(NS)
    plain = VM.save_bundle(tmp_path / "plain", m, cal, NAMES, fill, side_marker=True)
    info = VM.save_bundle(tmp_path / "rc", m, cal, NAMES, fill, {"note": "x"}, side_marker=True, recalibrator=rc)
    assert info["side_marker"] and info["recalibrated"] and plain["side_marker"] and not plain["recalibrated"]
    assert VM.save_bundle(tmp_path / "rc2", m, cal, NAMES, fill, {"note": "x"}, side_marker=True,
                          recalibrator=rc)["bundle_sha256"] == info["bundle_sha256"]          # deterministic
    V = VM.load_v(tmp_path / "rc", expected_sha256=info["bundle_sha256"], expected_columns_hash=STATE_V3_NAME_HASH)
    V0 = VM.load_v(tmp_path / "plain", expected_sha256=plain["bundle_sha256"])
    assert V.structure() == {"bundle_format": VM.BUNDLE_FORMAT, "side_marker": True, "recalibrated": True,
                             "kind": kind}
    assert V.names == NAMES and V.model_names == MNAMES
    want = rc.predict(q[:200], X[:200, AGE], X[:200, MIN])
    assert np.max(np.abs(V.predict(X[:200]) - want)) < 1e-7
    assert np.max(np.abs(V0.predict(X[:200]) - q[:200])) < 1e-7            # side-marker V: no recalibration
    assert np.max(np.abs(V.predict_calibrated(X[:200]) - q[:200])) < 1e-7
    rev = NAMES[::-1]
    assert np.max(np.abs(V.predict(X[:50, ::-1], columns=rev) - want[:50])) < 1e-5
    # the recalibration reads the state's own frame age and game minute
    X2 = X[:200].copy()
    X2[:, AGE] = np.where(X2[:, AGE] < 30, X2[:, AGE] + 30, X2[:, AGE] - 30)
    q2 = V0.predict(X2)
    assert np.max(np.abs(V.predict(X2) - rc.predict(q2, X2[:, AGE], X2[:, MIN]))) < 1e-7
    assert VM.read_bundle_structure(tmp_path / "rc", info["bundle_sha256"])["recalibration_knots_minute"] == \
        pytest.approx(list(rc.knots))
    with pytest.raises(RuntimeError):
        VM.read_bundle_structure(tmp_path / "rc", "0" * 64)
    with pytest.raises(ValueError):
        V.predict(X[:5, :100])


def _as_format1(d: Path) -> str:
    """Rewrite a no-side, no-recalibration format-2 bundle.json as the format-1 writer produced it."""
    b = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    b["format"] = "ev4_v_bundle_1"
    b.pop("side_marker")
    b.pop("recalibration")
    blob = json.dumps(b, indent=2, sort_keys=True).encode("utf-8")
    (d / "bundle.json").write_bytes(blob)
    return VM.sha256_bytes(blob)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_old_format_bundle_still_loads(tmp_path, kind):
    X, y = _rows(1500, 13, dense=True)
    perm_s = VM.swap_permutation(NAMES)
    if kind == "logistic":
        mu, sd = VM.pooled_standardizer(X, None, VM.swap_pairs(perm_s))
        m = VM.LogisticModel(0.01).fit(X, y, np.arange(len(y)), mu, sd)
    else:
        mu, sd = VM.pooled_standardizer(X, None, VM.mlp_groups(NAMES, perm_s))
        m = VM.MLPModel(1e-3, NAMES, seeds=(0,), max_epochs=1, threads=2).fit(X, y, np.arange(1300),
                                                                            np.arange(1300, 1500), mu, sd)
    cal = VM.PositiveSlopeSigmoid().fit(m.predict_proba(X), y, np.ones(len(y)))
    VM.save_bundle(tmp_path / "old", m, cal, NAMES, np.zeros(NS))
    sha = _as_format1(tmp_path / "old")
    V = VM.load_v(tmp_path / "old", expected_sha256=sha, expected_columns_hash=STATE_V3_NAME_HASH)
    assert V.structure() == {"bundle_format": "ev4_v_bundle_1", "side_marker": False, "recalibrated": False,
                             "kind": kind}
    assert V.model_names == NAMES
    assert np.max(np.abs(V.predict(X[:100]) - cal.predict(m.predict_proba(X[:100])))) < 1e-7
    with pytest.raises(ValueError, match="no side marker"):
        V.predict(X[:5], side=-1)
    b = json.loads((tmp_path / "old" / "bundle.json").read_text(encoding="utf-8"))
    b["format"] = "ev4_v_bundle_0"
    (tmp_path / "old" / "bundle.json").write_text(json.dumps(b), encoding="utf-8")
    with pytest.raises(RuntimeError, match="format"):
        VM.load_v(tmp_path / "old")


@pytest.mark.skipif(not (FIRST_FIT / "bundle.json").is_file(), reason="first full fit-V bundle absent")
def test_first_full_fit_bundle_format1_loads_for_comparison():
    st = VM.read_bundle_structure(FIRST_FIT)
    assert st["bundle_format"] == "ev4_v_bundle_1" and not st["side_marker"] and not st["recalibrated"]
    V = VM.load_v(FIRST_FIT, expected_columns_hash=STATE_V3_NAME_HASH)
    X, _ = _rows(64, 21, dense=True)
    p = V.predict(X)
    assert p.shape == (64,) and np.all((p > 0) & (p < 1))


# ------------------------------------------------------------------ synthetic end to end (slow)
@pytest.fixture(scope="module")
def extracts(tmp_path_factory):
    if not HAVE_R1A:
        pytest.skip("record 1A not present")
    d = tmp_path_factory.mktemp("vrev_ext")
    _, _, guard = FV.EX.load_params()
    _write_fake_extract(d / "e14", "15.14", 120, 20, guard)
    _write_fake_extract(d / "e15", "15.15", 120, 40, guard, h_from=d / "e14")
    return d / "e14", d / "e15"


def _fit(extracts, out: Path, kind: str = "logistic"):
    e14, e15 = extracts
    return FV.main(["--train", str(e14), "--select", str(e15), "--smoke", "--out", str(out), "--candidates", kind,
                    "--lgbm-rounds", "40", "--lgbm-patience", "5", "--mlp-epochs", "2", "--mart-boot", "9",
                    "--threads", "2"])


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_forced_adoption_freezes_recalibrated_v_and_oof_reproduces_it(extracts, tmp_path, monkeypatch, kind):
    import pandas as pd
    monkeypatch.setattr(VM, "recal_trigger", lambda a, b, level=0.05: {"triggered": True, "forced": "test"})
    monkeypatch.setattr(VM, "recal_adopt", lambda r, s, tol=0.0005: {"adopted": True, "difference": r - s,
                                                                      "forced": "test"})
    out = tmp_path / "fit_v"
    rep = _fit(extracts, out, kind)
    man = json.loads((out / "frozen_manifest.json").read_text(encoding="utf-8"))
    rc = rep["v_revision"]["recalibration"]
    assert rc["triggered"] and rc["fitted"] and rc["adopted"]
    assert man["V_frozen"]["recalibrated"] is True and man["V_frozen"]["side_marker"] is True
    assert man["V_frozen"]["variant"] == "side_marker_recalibrated" == rep["frozen_v"]["variant"]
    assert man["V_frozen"]["bundle_sha256"] == rc["bundle"]["bundle_sha256"] != man["candidates"][kind]
    assert man["v_revision"]["record"] == {"path": str(FV.V_REVISION_RECORD), "sha256": FV.V_REVISION_SHA256}
    assert rep["v_revision"]["record"]["content"]["revision_2_conditional_recalibration"].startswith("Only if")
    assert man["decisions"]["v_revision_record"]["sha256"] == FV.V_REVISION_SHA256
    assert VM.sha256_file(out / "V_frozen" / "bundle.json") == VM.sha256_file(out / "recalibrated" / kind / "bundle.json")
    # both variants reported: E4 metrics and martingale (a) / (b) with the bootstrap for both
    assert rc["e4"]["recalibrated"]["select"]["all"]["n"] == rc["e4"]["side_marker"]["select"]["all"]["n"] > 0
    assert rc["recalibrated_v"]["martingale"]["b_efficiency"]["wild_bootstrap"]["n_boot"] == 9
    assert rc["side_marker_v"]["martingale"]["b_efficiency"]["wild_bootstrap"]["n_boot"] == 9
    pv = pd.read_parquet(out / "predictions_v_15.15.parquet")
    assert np.allclose(pv["p_frozen_variant"], pv[f"p_recal_{kind}"])
    assert not np.allclose(pv[f"p_recal_{kind}"], pv[f"p_cal_{kind}"])
    pm = pd.read_parquet(out / "predictions_mart_15.15.parquet")
    assert {f"v0_{kind}_recal", f"v1_{kind}_recal"} <= set(pm.columns)
    V = VM.load_v(out / "V_frozen", expected_sha256=man["V_frozen"]["bundle_sha256"])
    assert V.recalibrated and V.side_marker
    e14, e15 = extracts
    # frozen predict on the stored 15.15 rows (StateV3 only) == the recalibrated predictions of the run
    Xse, mse = FV.load_v_rows(e15, FV.read_manifest(e15))
    assert np.max(np.abs(V.predict(Xse) - pv["p_frozen_variant"].to_numpy())) < 1e-6

    # OOF reproduces the structure: side marker + a per-fold recalibration with the same terms and knots
    res = OB.main(["--fit-v", str(out), "--train", str(e14), "--select", str(e15), "--smoke"])
    assert res["verification"]["v_structure"]["folds_recalibrated"] is True
    assert res["verification"]["v_structure"]["folds_side_marker"] is True
    knots = json.loads((out / "V_frozen" / "bundle.json").read_text(encoding="utf-8"))["recalibration"]["knots_minute"]
    coefs = []
    for k in range(5):
        st = VM.read_bundle_structure(out / "oof_v" / f"fold_{k}")
        assert st["side_marker"] and st["recalibrated"] and st["bundle_format"] == VM.BUNDLE_FORMAT
        assert st["recalibration_knots_minute"] == pytest.approx(knots)
        coefs.append(json.loads((out / "oof_v" / f"fold_{k}" / "bundle.json").read_text(encoding="utf-8"))
                     ["recalibration"]["coef"])
    assert len({tuple(np.round(c, 8)) for c in coefs}) == 5                 # refitted per fold
    om = json.loads((out / "oof_v" / "oof_manifest.json").read_text(encoding="utf-8"))
    assert om["v_structure"]["frozen"]["recalibrated"] is True
    # labels: the V source records the structure, and a fold with another structure is refused
    spec = LB.resolve_v("15.14", out / "frozen_manifest.json", VM.sha256_file(e14 / "manifest.json"), smoke=True)
    assert spec["v_structure"]["recalibrated"] is True and spec["v_structure"]["side_marker"] is True
    spec15 = LB.resolve_v("15.15", out / "frozen_manifest.json", VM.sha256_file(e15 / "manifest.json"), smoke=True)
    assert spec15["source"] == "frozen" and spec15["v_structure"]["bundle_format"] == VM.BUNDLE_FORMAT
    vp = LB.VPredictor(spec15, threads=1)
    assert np.allclose(vp.predict(Xse[:50], list(mse["match_id"].astype(str)[:50]))[0], V.predict(Xse[:50]))


@pytest.mark.slow
def test_not_triggered_keeps_side_marker_v_as_byte_copy(extracts, tmp_path, monkeypatch):
    monkeypatch.setattr(VM, "recal_trigger", lambda a, b, level=0.05: {"triggered": False, "forced": "test"})
    out = tmp_path / "fit_v"
    (out / "recalibrated" / "stale").mkdir(parents=True)                      # a previous run's leftovers go
    rep = _fit(extracts, out, "logistic")
    man = json.loads((out / "frozen_manifest.json").read_text(encoding="utf-8"))
    rc = rep["v_revision"]["recalibration"]
    assert not rc["triggered"] and not rc["fitted"] and not rc["adopted"] and "bundle" not in rc
    assert not (out / "recalibrated").exists()
    assert man["V_frozen"]["bundle_sha256"] == man["candidates"]["logistic"]
    assert man["V_frozen"]["recalibrated"] is False and man["V_frozen"]["variant"] == "side_marker"
    for f in (out / "candidates" / "logistic").iterdir():
        assert (out / "V_frozen" / f.name).read_bytes() == f.read_bytes()
    V = VM.load_v(out / "V_frozen")
    assert V.side_marker and not V.recalibrated and V.bundle_format == "ev4_v_bundle_2"
    res = OB.main(["--fit-v", str(out), "--train", str(extracts[0]), "--select", str(extracts[1]), "--smoke"])
    assert res["verification"]["v_structure"] == {"frozen": {"bundle_format": VM.BUNDLE_FORMAT, "side_marker": True,
                                                             "recalibrated": False, "kind": "logistic"},
                                                  "folds_side_marker": True, "folds_recalibrated": False,
                                                  "matches_frozen": True, "recalibration_knots_minute": None}
    # OOF fold models with the side marker reproduce fit-V's logistic CV fits exactly (same swapped rows)
    assert res["verification"]["cv_reproduction_max_abs_diff"] <= 1e-9
    # a fold bundle whose structure differs from the frozen V is refused by the labels
    fold0 = out / "oof_v" / "fold_0"
    b = json.loads((fold0 / "bundle.json").read_text(encoding="utf-8"))
    b["side_marker"] = None
    blob = json.dumps(b, indent=2, sort_keys=True).encode("utf-8")
    (fold0 / "bundle.json").write_bytes(blob)
    om_path = out / "oof_v" / "oof_manifest.json"
    om = json.loads(om_path.read_text(encoding="utf-8"))
    om["folds"]["0"]["bundle_sha256"] = VM.sha256_bytes(blob)
    om_path.write_text(json.dumps(om), encoding="utf-8")
    side = json.loads((out / LB.OOF_BLOCK_FILE).read_text(encoding="utf-8"))
    side["manifest_sha256"] = VM.sha256_file(om_path)
    (out / LB.OOF_BLOCK_FILE).write_text(json.dumps(side), encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match="side_marker"):
        LB.resolve_v("15.14", out / "frozen_manifest.json", VM.sha256_file(extracts[0] / "manifest.json"), smoke=True)


@pytest.mark.slow
@pytest.mark.skipif(not (FIRST_FIT / "bundle.json").is_file() or not HAVE_R1A, reason="smoke_prerun absent")
def test_oof_of_an_old_format_fit_keeps_its_structure(tmp_path):
    """A format-1 frozen V (the smoke_prerun fit, copied) gives OOF folds without side marker / recalibration."""
    smoke = FV.EX.OUT_BASE / "stage2" / "smoke_prerun"
    if not (smoke / "fit_v" / "frozen_manifest.json").is_file():
        pytest.skip("smoke_prerun fit-V absent")
    fit_dir = tmp_path / "fit_v"
    shutil.copytree(smoke / "fit_v", fit_dir)
    assert VM.read_bundle_structure(fit_dir / "V_frozen")["bundle_format"] == "ev4_v_bundle_1"
    res = OB.main(["--fit-v", str(fit_dir), "--train", str(smoke / "extract" / "15.14"),
                   "--select", str(smoke / "extract" / "15.15"), "--smoke", "--allow-code-drift"])
    vs = res["verification"]["v_structure"]
    assert vs["folds_side_marker"] is False and vs["folds_recalibrated"] is False
    for k in range(5):
        st = VM.read_bundle_structure(fit_dir / "oof_v" / f"fold_{k}")
        assert not st["side_marker"] and not st["recalibrated"]
