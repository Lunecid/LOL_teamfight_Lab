"""Tests for scripts/exact_v4/ev4_03c_prev_v_compare.py (previous evaluator A_MLP_expanded on 15.15 V_SELECT rows).

Synthetic: fake fit-V directories / prev parquets for compare_with_fit_v (identical-row rule, sign, CR1 SEs, subset and
y guards, --v-column); fake 15.15 extract V indexes for load_select_rows (split by sha256, sha256 check, remakes,
patch refusal); the write guard for the old repository.  Real data (marked slow, skipped when the bundle, the cache or
the 15.15 extract is absent): a 5-match score run into tmp_path, single process, with the old-loader parity gate.
No patch other than 15.15 is read.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
WT = Path(__file__).resolve().parents[1]
SCRIPTS = WT / "scripts" / "exact_v4"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ev4_v_models as VM  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PV = _load("ev4_03c_prev_v_compare")

from gameplay.split_guard import SplitViolation  # noqa: E402

pd = pytest.importorskip("pandas")


# ------------------------------------------------------------------ synthetic helpers
def _mids(n):
    return [f"KR_{1000 + i}" for i in range(n)]


def _fake_fit_v(d: Path, n_matches=60, rows=6, chosen="mlp", seed=0, frozen=True):
    rng = np.random.default_rng(seed)
    recs = []
    for mid in _mids(n_matches):
        y = int(rng.integers(0, 2))
        for k in range(rows):
            t = 120_000 + 120_000 * k + int(rng.integers(0, 120_000))
            z = rng.normal(0.8 * (2 * y - 1) * (k + 1) / rows, 1.0)
            p = 1 / (1 + np.exp(-z))
            recs.append(dict(match_id=mid, t=t, bucket=k, split=VM.v_split(mid), y_blue_win=y,
                             p_raw_mlp=p, p_cal_mlp=np.clip(p * 0.98 + 0.01, 0, 1), p_cal_alt=0.5))
    df = pd.DataFrame(recs)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "predictions_v_15.15.parquet", index=False)
    (d / "frozen_manifest.json").write_text(json.dumps({"chosen": chosen, "frozen": frozen, "smoke": False,
                                                        "pilot": False}), encoding="utf-8")
    return df


def _fake_prev(p: Path, fitv: "pd.DataFrame", seed=1):
    rng = np.random.default_rng(seed)
    s = fitv.loc[fitv["split"] == "select", ["match_id", "t", "y_blue_win", "p_cal_mlp"]].copy()
    raw = np.clip(s["p_cal_mlp"].to_numpy() + rng.normal(0, 0.08, len(s)), 0.01, 0.99)
    s["p_prev_raw"] = raw
    s["p_prev"] = np.clip(raw * 0.95 + 0.02, 0, 1)
    s["status"] = "ok"
    s = s.drop(columns=["p_cal_mlp"]).sample(frac=1.0, random_state=0)   # order must not matter
    s.to_parquet(p, index=False)
    return s


# ------------------------------------------------------------------ compare_with_fit_v
def test_compare_identical_rows_and_cluster_se(tmp_path):
    fv = _fake_fit_v(tmp_path / "fitv")
    prev = _fake_prev(tmp_path / "prev.parquet", fv)
    res = PV.compare_with_fit_v(tmp_path / "prev.parquet", tmp_path / "fitv", auc_boot=20)
    assert res["chosen"] == "mlp" and res["v_column"] == "p_cal_mlp"
    sel = fv.loc[fv["split"] == "select"]
    assert res["rows"]["compared"] == len(sel) == len(prev) and res["rows"]["subset"] is False
    m = sel.merge(prev, on=["match_id", "t"])
    y = m["y_blue_win_x"].to_numpy()
    d_ll = VM.row_log_loss(y, m["p_prev"]) - VM.row_log_loss(y, m["p_cal_mlp"])
    mean, se, G = VM.cluster_mean_se(d_ll, m["match_id"].to_numpy())
    got = res["delta_prev_cal_vs_v"]["all"]["log_loss_delta"]
    assert got["delta"] == pytest.approx(mean, abs=1e-12) and got["se_cluster"] == pytest.approx(se, abs=1e-12)
    assert got["n_clusters"] == G == sel["match_id"].nunique()
    bd = (m["p_prev"] - y) ** 2 - (m["p_cal_mlp"] - y) ** 2
    assert res["delta_prev_cal_vs_v"]["all"]["brier_delta"]["delta"] == pytest.approx(bd.mean(), abs=1e-12)
    # delta = prev - V
    all_ = res["delta_prev_cal_vs_v"]["all"]
    assert all_["log_loss_delta"]["delta"] == pytest.approx(all_["log_loss_prev"] - all_["log_loss_v"], abs=1e-12)
    assert res["metrics"]["prev_cal"]["all"]["n"] == len(sel)
    assert set(res["delta_prev_cal_vs_v"]) == {"all", "lt15", "15to25", "ge25"}
    assert all_["auc_delta"]["reps"] == 20
    assert "v_raw" in res["metrics"] and "delta_prev_cal_vs_v_raw" in res


def test_compare_guards(tmp_path):
    fv = _fake_fit_v(tmp_path / "fitv")
    prev = _fake_prev(tmp_path / "prev.parquet", fv)
    # subset: refused unless allow_subset
    sub = prev.iloc[: len(prev) // 2]
    sub.to_parquet(tmp_path / "sub.parquet", index=False)
    with pytest.raises(RuntimeError, match="no prev score"):
        PV.compare_with_fit_v(tmp_path / "sub.parquet", tmp_path / "fitv", auc_boot=0)
    r = PV.compare_with_fit_v(tmp_path / "sub.parquet", tmp_path / "fitv", allow_subset=True, auc_boot=0)
    assert r["rows"]["subset"] and r["rows"]["compared"] == len(sub)
    # a prev row that is not a V_SELECT row (a V_CAL row) is always refused
    cal = fv.loc[fv["split"] == "cal"].iloc[:1]
    extra = pd.concat([prev, pd.DataFrame(dict(match_id=cal["match_id"], t=cal["t"], y_blue_win=cal["y_blue_win"],
                                                p_prev=0.5, p_prev_raw=0.5, status="ok"))])
    extra.to_parquet(tmp_path / "extra.parquet", index=False)
    with pytest.raises(RuntimeError, match="not fit-V V_SELECT"):
        PV.compare_with_fit_v(tmp_path / "extra.parquet", tmp_path / "fitv", allow_subset=True, auc_boot=0)
    # y disagreement
    bad = prev.copy()
    bad.iloc[0, bad.columns.get_loc("y_blue_win")] = 1 - bad.iloc[0]["y_blue_win"]
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(RuntimeError, match="y_blue_win differs"):
        PV.compare_with_fit_v(tmp_path / "bad.parquet", tmp_path / "fitv", auc_boot=0)
    # unknown column
    with pytest.raises(KeyError):
        PV.compare_with_fit_v(tmp_path / "prev.parquet", tmp_path / "fitv", v_column="p_cal_nope", auc_boot=0)


def test_compare_nonfinite_dropped_from_both_and_v_column(tmp_path):
    fv = _fake_fit_v(tmp_path / "fitv")
    prev = _fake_prev(tmp_path / "prev.parquet", fv)
    prev = prev.copy()
    prev.iloc[:3, prev.columns.get_loc("p_prev")] = np.nan
    prev.iloc[:3, prev.columns.get_loc("status")] = "state:ValueError:x"
    prev.to_parquet(tmp_path / "nan.parquet", index=False)
    r = PV.compare_with_fit_v(tmp_path / "nan.parquet", tmp_path / "fitv", auc_boot=0)
    assert r["rows"]["dropped_nonfinite"] == 3 and r["rows"]["compared"] == len(prev) - 3
    assert r["metrics"]["v"]["all"]["n"] == r["metrics"]["prev_cal"]["all"]["n"] == len(prev) - 3
    r2 = PV.compare_with_fit_v(tmp_path / "prev.parquet", tmp_path / "fitv", v_column="p_cal_alt", auc_boot=0)
    assert r2["v_column"] == "p_cal_alt"
    assert r2["metrics"]["v"]["all"]["log_loss"] == pytest.approx(np.log(2), abs=1e-12)


def test_auc_bootstrap_deterministic():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400).astype(float)
    a = np.clip(y * 0.3 + rng.uniform(0, 0.7, 400), 0, 1)
    b = rng.uniform(0, 1, 400)
    cl = np.repeat(np.arange(40), 10).astype(str)
    r1 = PV.auc_delta_bootstrap(y, a, b, cl, reps=30)
    r2 = PV.auc_delta_bootstrap(y, a, b, cl, reps=30)
    assert r1 == r2 and r1["delta"] > 0 and r1["se_cluster_boot"] > 0 and r1["reps"] == 30


# ------------------------------------------------------------------ rows / guards
def _fake_extract(d: Path, patch="15.15", n_matches=40, remake=False):
    d.mkdir(parents=True, exist_ok=True)
    chunks = {}
    mids = _mids(n_matches)
    for c, part in enumerate((mids[: n_matches // 2], mids[n_matches // 2:])):
        recs = []
        for mid in part:
            for k in range(3):
                recs.append(dict(match_id=mid, bucket=k, bucket_start=0, bucket_end=0, t=120_000 * (k + 1) + 7,
                                 snapshot_ms=0, frame_age_ms=7, game_end=200_000 if remake else 1_500_000,
                                 y_blue_win=1))
        v = pd.DataFrame(recs)
        v.insert(0, "row", np.arange(len(v)))
        fn = f"chunk_{c:05d}_v.parquet"
        v.to_parquet(d / fn, index=False)
        chunks[f"{c:05d}"] = {"files": {fn: {"sha256": VM.sha256_file(d / fn)}}}
    (d / "manifest.json").write_text(json.dumps({"patch": patch, "chunks": chunks}), encoding="utf-8")
    return mids


def test_load_select_rows(tmp_path):
    mids = _fake_extract(tmp_path / "e")
    man = PV.read_extract_manifest(tmp_path / "e")
    v = PV.load_select_rows(tmp_path / "e", man)
    want = {m for m in mids if VM.v_split(m) == "select"}
    assert set(v["match_id"]) == want and len(v) == 3 * len(want)
    assert list(v["chunk"]) == sorted(v["chunk"])          # extract order kept
    v2 = PV.load_select_rows(tmp_path / "e", man, limit_matches=3)
    first3 = sorted(want, key=lambda m: (PV.EX.sha256_text(m), m))[:3]
    assert set(v2["match_id"]) == set(first3)
    # tampered file
    p = tmp_path / "e" / "chunk_00001_v.parquet"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="sha256"):
        PV.load_select_rows(tmp_path / "e", man)


def test_remake_rows_refused(tmp_path):
    _fake_extract(tmp_path / "e", remake=True)
    man = PV.read_extract_manifest(tmp_path / "e")
    with pytest.raises(RuntimeError, match="remake"):
        PV.load_select_rows(tmp_path / "e", man)


def test_patch_refusals(tmp_path):
    _fake_extract(tmp_path / "a", patch="15.14")
    with pytest.raises(RuntimeError, match="15.15 only"):
        PV.read_extract_manifest(tmp_path / "a")
    _fake_extract(tmp_path / "b", patch="15.16")
    with pytest.raises(SplitViolation):
        PV.read_extract_manifest(tmp_path / "b")


def test_no_writes_under_old_repo(tmp_path):
    with pytest.raises(PermissionError):
        PV.assert_writable_out(PV.OLD_REPO / "outputs" / "x")
    with pytest.raises(PermissionError):
        PV.assert_writable_out(PV.EVAL_DIR / "copy.joblib")
    PV.assert_writable_out(tmp_path / "ok")


# ------------------------------------------------------------------ real data (read only), single process
HAVE_REAL = ((PV.EVAL_DIR / PV.BUNDLE_NAME).is_file() and (PV.DEFAULT_EXTRACT / "manifest.json").is_file()
             and Path(PV.EX.CACHE).is_dir() and PV.V_REVISION_RECORD.is_file())


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_REAL, reason="bundle / 15.15 extract / cache not present")
def test_real_score_5_matches(tmp_path):
    out = tmp_path / "pv"
    s = PV.main(["score", "--limit-matches", "5", "--out", str(out), "--parity-matches", "5", "--workers", "1",
                 "--threads", "1"])
    assert s["checks"]["old_loader_parity"]["pass"] and s["checks"]["old_loader_parity"]["n_matches"] == 5
    assert s["checks"]["frame_age_linkage"]["pass"]
    assert s["evaluator"]["bundle_sha256"] == s["evaluator"]["files"][PV.BUNDLE_NAME]["sha256"]
    df = pd.read_parquet(out / PV.PREV_PARQUET)
    assert list(df.columns[:3]) == ["match_id", "t", "y_blue_win"] and df["match_id"].nunique() == 5
    assert np.isfinite(df["p_prev"]).all() and (df["status"] == "ok").all()
    assert all(VM.v_split(m) == "select" for m in df["match_id"].unique())
    # scores equal the old module's predict_calibrated on the same StateV2 rows
    ev, EB, _ = PV.load_prev_evaluator()
    names = list(ev["preproc"]["schema"]["raw_names"])
    mid = df["match_id"].iloc[0]
    g = df.loc[df["match_id"] == mid]
    X, st = PV.states_for_match(PV.load_pack_v2(mid), g["t"].tolist(), names)
    assert st == ["ok"] * len(g)
    np.testing.assert_allclose(EB.predict_calibrated(ev, X), g["p_prev"].to_numpy(), atol=1e-7, rtol=0)
    # the prev rows are exactly the extract's V_SELECT rows of these matches
    man = PV.read_extract_manifest(PV.DEFAULT_EXTRACT)
    v = PV.load_select_rows(PV.DEFAULT_EXTRACT, man, limit_matches=5)
    assert list(zip(v["match_id"], v["t"])) == list(zip(df["match_id"], df["t"]))
