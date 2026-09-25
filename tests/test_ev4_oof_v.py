"""Tests for scripts/exact_v4/ev4_03b_oof_v.py (out-of-fold V for the 15.14 training labels, ev4_04_labels.OOF_SPEC).

Synthetic: fake 15.14 / 15.15 extracts (tests.test_ev4_fit_v._write_fake_extract) -> ev4_03_fit_v --smoke with one
candidate (so each kind is the chosen one) -> ev4_03b; the OOF set must reproduce fit-V's CV fits (logistic / lgbm),
exclude every fold's own matches, cover 100 % of the 15.14 matches, and be accepted by ev4_04_labels.check_oof /
resolve_v / VPredictor and ev4_r1_record.oof_block through the oof_v_block.json sidecar (frozen_manifest.json is
never modified).  Smoke (marked slow): the stage-2 smoke_prerun fit-V run is COPIED to tmp_path (the original is
read only) and ev4_03b + the 15.14 labeller run on the smoke 15.14 / 15.15 extracts.  No other patch's data is read.
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OB = _load("ev4_03b_oof_v")
FV = OB.FV
LB = OB.LB
R1 = _load("ev4_r1_record")

from gameplay.split_guard import SplitViolation  # noqa: E402
from tests.test_ev4_fit_v import _write_fake_extract  # noqa: E402

SMOKE = LB.OUT_BASE / "stage2" / "smoke_prerun"
HAVE_SMOKE = (SMOKE / "extract" / "15.14" / "manifest.json").is_file() and \
    (SMOKE / "fit_v" / "frozen_manifest.json").is_file()
HAVE_R1A = FV.EX.RECORD1A.is_file()


# ------------------------------------------------------------------ synthetic fit-V runs (one per chosen kind)
@pytest.fixture(scope="module")
def extracts(tmp_path_factory):
    if not HAVE_R1A:
        pytest.skip("record 1A not present")
    d = tmp_path_factory.mktemp("ext")
    _, _, guard = FV.EX.load_params()
    _write_fake_extract(d / "e14", "15.14", 120, 20, guard)
    _write_fake_extract(d / "e15", "15.15", 120, 40, guard, h_from=d / "e14")
    return d / "e14", d / "e15"


def _fit(extracts, out: Path, kind: str) -> Path:
    e14, e15 = extracts
    FV.main(["--train", str(e14), "--select", str(e15), "--smoke", "--out", str(out), "--candidates", kind,
             "--lgbm-rounds", "40", "--lgbm-patience", "5", "--mlp-epochs", "2", "--mart-boot", "9",
             "--threads", "2"])
    return out


def _oof(extracts, fit_dir: Path, *extra) -> dict:
    e14, e15 = extracts
    return OB.main(["--fit-v", str(fit_dir), "--train", str(e14), "--select", str(e15), "--smoke", *extra])


@pytest.fixture(scope="module")
def runs(extracts, tmp_path_factory):
    """runs(kind): a synthetic fit-V run with `kind` as the only (so chosen) candidate + its OOF set (cached)."""
    cache = {}

    def get(kind):
        if kind not in cache:
            fit_dir = _fit(extracts, tmp_path_factory.mktemp(f"fit_{kind}") / "fit_v", kind)
            before = VM.sha256_file(fit_dir / "frozen_manifest.json")
            res = _oof(extracts, fit_dir)
            cache[kind] = {"kind": kind, "fit": fit_dir, "res": res, "vman_sha_before": before, "extracts": extracts}
        return cache[kind]
    return get


@pytest.fixture(params=["logistic", "lgbm", "mlp"])
def oof_run(request, runs):
    return runs(request.param)


@pytest.fixture
def logistic_run(runs):
    return runs("logistic")


@pytest.mark.slow
def test_oof_export_follows_spec_and_is_accepted_by_labels(oof_run):
    kind, fit_dir, res = oof_run["kind"], oof_run["fit"], oof_run["res"]
    vpath = fit_dir / "frozen_manifest.json"
    assert VM.sha256_file(vpath) == oof_run["vman_sha_before"]           # frozen manifest untouched
    vman = json.loads(vpath.read_text(encoding="utf-8"))
    assert "oof_v" not in vman and vman["chosen"] == kind
    vf = VM.assert_v_usable(vman, allow_smoke=True)
    side = json.loads((fit_dir / LB.OOF_BLOCK_FILE).read_text(encoding="utf-8"))
    assert side["format"] == LB.OOF_BLOCK_FORMAT and side["dir_rel"] == "oof_v"
    assert side["frozen_manifest_sha256"] == VM.sha256_file(vpath) and side["frozen_bundle_sha256"] == vf["bundle_sha256"]
    om = json.loads((fit_dir / "oof_v" / "oof_manifest.json").read_text(encoding="utf-8"))
    assert side["manifest_sha256"] == VM.sha256_file(fit_dir / "oof_v" / "oof_manifest.json")
    # spec keys (ev4_04_labels.OOF_SPEC)
    assert om["format"] == LB.OOF_FORMAT and om["train_patch"] == "15.14" and om["kind"] == kind
    assert om["n_folds"] == 5 and om["fold_rule"] == LB.OOF_FOLD_RULE and om["smoke"] is True and om["pilot"] is False
    assert om["frozen_bundle_sha256"] == vf["bundle_sha256"]
    key = OB.HP_KEY[kind]
    rep = json.loads((fit_dir / "report_e4.json").read_text(encoding="utf-8"))
    assert float(om["hyperparameter"][key]) == float(rep["cv"][kind]["chosen"])           # never re-selected
    assert om["hyperparameter_selection"]["reselected_here"] is False
    assert any("ALL 15.14 folds" in d for d in om["disclosures"])
    assert om["report"]["sha256"] == VM.sha256_file(fit_dir / "oof_v" / "oof_report.json")
    # fold bundles: chosen kind / value, held-out fold, calibrated on V_CAL, ids exclude the own fold
    import pandas as pd
    mids_all = set()
    for c in ("00000", "00001"):
        mids_all |= set(pd.read_parquet(oof_run["extracts"][0] / f"chunk_{c}_v.parquet")["match_id"].astype(str))
    ids = {}
    for k in range(5):
        f = om["folds"][str(k)]
        b = json.loads((fit_dir / "oof_v" / f"fold_{k}" / "bundle.json").read_text(encoding="utf-8"))
        assert b["kind"] == kind and b["extra"]["oof_heldout_fold"] == k and b["extra"]["calibration_rows"] == "V_CAL"
        assert OB._hp_from_bundle(kind, b) == om["hyperparameter"]
        if kind == "mlp":
            assert b["model"]["seeds"] == [0, 1, 2]
        V = VM.load_v(fit_dir / "oof_v" / f"fold_{k}", expected_sha256=f["bundle_sha256"])
        assert V.kind == kind
        ids[k] = set((fit_dir / "oof_v" / f["train_match_ids_file"]).read_text(encoding="utf-8").split())
        assert ids[k] == {m for m in mids_all if VM.cv_fold(m) != k}                     # fit + stop matches
    cov = res["verification"]["coverage_v_rows"]
    assert cov["n"] == len(mids_all) == cov["full"] and cov["in_own_fold"] == 0 and cov["rate"] == 1.0
    # CV reproduction: logistic / lgbm refits == fit-V's CV fits of the chosen value
    for k in range(5):
        r = res["folds"][str(k)]["cv_reproduction"]
        if kind in ("logistic", "lgbm"):
            assert r["abs_diff"] <= 1e-9, r
            if kind == "lgbm":
                assert r["refit_best_iteration"] == r["fit_v_best_iteration"]
            else:
                assert r["refit_nit"] == r["fit_v_nit"]
        a = res["folds"][str(k)]["agreement_with_frozen_V_SELECT"]
        assert a["n"] > 0 and np.isfinite(a["mean_abs_diff"])
    # the consumers: ev4_04_labels (sidecar path) ...
    spec = LB.check_oof(vman, vf, smoke=True, vpath=vpath)
    assert spec["block_source"] == LB.OOF_BLOCK_FILE and spec["manifest_sha256"] == side["manifest_sha256"]
    tr_sha = vman["inputs"]["train"]["manifest_sha256"]
    vs = LB.resolve_v("15.14", fit_dir, tr_sha, smoke=True, record1_v=None)
    assert vs["source"] == "oof"
    P = LB.VPredictor(vs)
    Xtr, mtr = FV.load_v_rows(oof_run["extracts"][0], FV.read_manifest(oof_run["extracts"][0]))
    mids = mtr["match_id"].astype(str).tolist()
    p, folds = P.predict(Xtr, mids)                       # every 15.14 match routed to a model that never saw it
    assert folds.tolist() == [VM.cv_fold(m) for m in mids] and np.isfinite(p).all()
    assert P.coverage(mids) == {"n": len(set(mids)), "full": len(set(mids))}
    with pytest.raises(LB.OOFNotAvailable, match="smoke"):
        LB.check_oof(dict(vman, smoke=False), vf, smoke=False, vpath=vpath)
    # ... and the record-1 builder
    ob = R1.oof_block(fit_dir, {"usable": True, "V_frozen": vf}, allow_smoke=True)
    assert ob["ok"] and ob["block_source"] == LB.OOF_BLOCK_FILE and ob["manifest_sha256"] == side["manifest_sha256"]
    assert ob["fold_bundle_sha256"] == side["fold_bundle_sha256"]


@pytest.mark.slow
def test_sidecar_binding_and_tampering(logistic_run, tmp_path):
    oof_run = logistic_run
    src = oof_run["fit"]
    fit_dir = tmp_path / "fit_v"
    shutil.copytree(src, fit_dir)
    vpath = fit_dir / "frozen_manifest.json"
    vman = json.loads(vpath.read_text(encoding="utf-8"))
    vf = VM.assert_v_usable(vman, allow_smoke=True)
    assert LB.check_oof(vman, vf, smoke=True, vpath=vpath)["dir"] == str(fit_dir / "oof_v")    # resolved by location
    # an explicit oof_v block in the frozen manifest wins over the sidecar
    blk = {"dir": str(fit_dir / "oof_v"), "manifest_sha256": VM.sha256_file(fit_dir / "oof_v" / "oof_manifest.json"),
           "kind": vf["kind"]}
    assert LB.check_oof(dict(vman, oof_v=blk), vf, smoke=True, vpath=vpath)["block_source"] == "frozen_manifest"
    # a tampered training-id list is refused
    tf = fit_dir / "oof_v" / "fold_3" / "train_match_ids.txt"
    orig = tf.read_bytes()
    tf.write_text(tf.read_text(encoding="utf-8") + "SYN_extra\n", encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match="sha256"):
        LB.check_oof(vman, vf, smoke=True, vpath=vpath)
    tf.write_bytes(orig)
    # the sidecar is bound to this frozen manifest: a changed manifest refuses it (labels and record 1)
    side = fit_dir / LB.OOF_BLOCK_FILE
    sb = json.loads(side.read_text(encoding="utf-8"))
    vpath.write_text(json.dumps(dict(vman, note="edited")), encoding="utf-8")
    with pytest.raises(LB.OOFNotAvailable, match="another frozen_manifest"):
        LB.check_oof(vman, vf, smoke=True, vpath=vpath)
    with pytest.raises(R1.DraftRefused, match="sidecar"):
        R1.oof_block(fit_dir, {"usable": True, "V_frozen": vf}, allow_smoke=True)
    vpath.write_text(json.dumps(vman), encoding="utf-8")
    for bad in ({"dir_rel": str(fit_dir / "oof_v")}, {"dir_rel": "../oof_v"}, {"format": "x"}):
        side.write_text(json.dumps(dict(sb, frozen_manifest_sha256=VM.sha256_file(vpath), **bad)), encoding="utf-8")
        with pytest.raises(LB.OOFNotAvailable):
            LB.check_oof(vman, vf, smoke=True, vpath=vpath)
    side.unlink()                                          # no sidecar and no block: 15.14 refused with the spec
    with pytest.raises(LB.OOFNotAvailable, match="oof_v_block.json"):
        LB.check_oof(vman, vf, smoke=True, vpath=vpath)
    assert R1.oof_block(fit_dir, {"usable": True, "V_frozen": vf}, allow_smoke=True)["ok"] is False


@pytest.mark.slow
def test_guards(logistic_run, tmp_path):
    oof_run = logistic_run
    e14, e15 = oof_run["extracts"]
    fit_dir = tmp_path / "fit_v"
    shutil.copytree(oof_run["fit"], fit_dir)
    with pytest.raises(SystemExit, match="--force"):                     # existing output
        _oof(oof_run["extracts"], fit_dir)
    with pytest.raises(SystemExit, match="smoke"):                       # smoke fit-V without --smoke
        OB.main(["--fit-v", str(fit_dir), "--train", str(e14), "--select", str(e15), "--force"])
    vpath = fit_dir / "frozen_manifest.json"
    vbytes = vpath.read_bytes()
    vman = json.loads(vpath.read_text(encoding="utf-8"))
    vpath.write_text(json.dumps(dict(vman, pilot=True)), encoding="utf-8")
    with pytest.raises(SystemExit, match="pilot"):
        _oof(oof_run["extracts"], fit_dir, "--force")
    vpath.write_text(json.dumps(dict(vman, stop_rule=dict(vman["stop_rule"], stop_before_record1=True))),
                     encoding="utf-8")
    with pytest.raises(VM.VNotUsable, match="stop rule"):
        _oof(oof_run["extracts"], fit_dir, "--force")
    vpath.write_bytes(vbytes)
    # a report whose chosen value differs from the frozen bundle is refused (nothing is re-selected)
    rp = fit_dir / "report_e4.json"
    rbytes = rp.read_bytes()
    rep = json.loads(rp.read_text(encoding="utf-8"))
    other = [c for c in VM.GRIDS["logistic"]["C"] if float(c) != float(rep["cv"]["logistic"]["chosen"])][0]
    rp.write_text(json.dumps(dict(rep, cv=dict(rep["cv"], logistic=dict(rep["cv"]["logistic"], chosen=other)))),
                  encoding="utf-8")
    with pytest.raises(RuntimeError, match="chosen C"):
        _oof(oof_run["extracts"], fit_dir, "--force")
    rp.write_bytes(rbytes)
    # another extract than the fit's is refused before any array is read
    _, _, guard = FV.EX.load_params()
    _write_fake_extract(tmp_path / "e14b", "15.14", 60, 99, guard)
    _write_fake_extract(tmp_path / "e15b", "15.15", 60, 98, guard, h_from=tmp_path / "e14b")
    with pytest.raises(RuntimeError, match="not the one fit-V used"):
        OB.main(["--fit-v", str(fit_dir), "--train", str(tmp_path / "e14b"), "--select", str(tmp_path / "e15b"),
                 "--smoke", "--force"])
    # --plan reads only the report
    plan = OB.main(["--fit-v", str(fit_dir), "--plan", "--smoke"])["plan"]
    assert plan["kind"] == "logistic" and plan["projected_s"] >= 0 and len(plan["fit_v_cv_fit_s_chosen_value"]) == 5
    # a --force rerun on the unchanged fit-V run rebuilds identical bundles (deterministic)
    before = json.loads((fit_dir / LB.OOF_BLOCK_FILE).read_text(encoding="utf-8"))["fold_bundle_sha256"]
    _oof(oof_run["extracts"], fit_dir, "--force")
    after = json.loads((fit_dir / LB.OOF_BLOCK_FILE).read_text(encoding="utf-8"))["fold_bundle_sha256"]
    assert after == before


def _fake_manifest(d: Path, patch: str) -> Path:
    from gameplay.state_value_v3 import STATE_V3_NAME_HASH, STATE_VERSION
    d.mkdir(parents=True, exist_ok=True)
    m = {"patch": patch, "STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "state_version": STATE_VERSION, "sample": True,
         "chunks": {"00000": {"rows": {"v": 1, "mart": 0}, "files": {}}}}
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d


@pytest.mark.parametrize("train, select", [("15.14", "15.16"), ("16.13", "15.15"), ("15.15", "15.14")])
def test_held_out_patch_refused_before_any_array(tmp_path, train, select):
    fit_dir = tmp_path / "fit_v"
    fit_dir.mkdir()
    with pytest.raises(SplitViolation):                    # no chunk file exists: refused on the manifests alone
        OB.main(["--fit-v", str(fit_dir), "--train", str(_fake_manifest(tmp_path / "a", train)),
                 "--select", str(_fake_manifest(tmp_path / "b", select)), "--smoke"])
    assert not (fit_dir / "oof_v").exists() and not (fit_dir / LB.OOF_BLOCK_FILE).exists()


def test_write_oof_manifest_extra_cannot_replace_spec_keys(tmp_path):
    with pytest.raises(RuntimeError, match="spec keys"):
        LB.write_oof_manifest(tmp_path, "logistic", {"C": 0.01},
                              {k: {"dir": str(tmp_path / f"fold_{k}"), "bundle_sha256": "x"} for k in range(5)},
                              {k: [] for k in range(5)}, "y", smoke=True, extra={"kind": "mlp"})


def test_coverage_check_and_agreement():
    mids = [f"M{i}" for i in range(50)]
    ids = {k: [m for m in mids if VM.cv_fold(m) != k] for k in range(5)}
    c = OB.coverage_check(ids, mids)
    assert c["full"] == c["n"] == 50 and c["in_own_fold"] == 0
    ids[0] = ids[0][1:]
    ids[1] = ids[1] + [m for m in mids if VM.cv_fold(m) == 1][:1]
    c = OB.coverage_check(ids, mids)
    assert c["full"] == 49 and c["in_own_fold"] == 1
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200)
    a = OB.agreement(p, p, (rng.random(200) < p).astype(float))
    assert a["pearson_r"] == pytest.approx(1.0) and a["max_abs_diff"] == 0.0


# ------------------------------------------------------------------ smoke (15.14 / 15.15 smoke_prerun, copied)
@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_SMOKE and HAVE_R1A), reason="smoke outputs / record 1A absent")
def test_smoke_prerun_oof_and_labels_1514(tmp_path):
    import pandas as pd
    fit_dir = tmp_path / "fit_v"
    shutil.copytree(SMOKE / "fit_v", fit_dir)                            # the original stays read only
    before = VM.sha256_file(fit_dir / "frozen_manifest.json")
    res = OB.main(["--fit-v", str(fit_dir), "--train", str(SMOKE / "extract" / "15.14"),
                   "--select", str(SMOKE / "extract" / "15.15"), "--smoke", "--allow-code-drift"])
    assert VM.sha256_file(fit_dir / "frozen_manifest.json") == before
    ver = res["verification"]
    assert ver["coverage_v_rows"]["rate"] == 1.0 and ver["coverage_v_rows"]["in_own_fold"] == 0
    assert ver["coverage_engagement_rows"]["rate"] == 1.0
    if res["kind"] in ("logistic", "lgbm"):
        assert ver["cv_reproduction_max_abs_diff"] <= 1e-9
    assert all(r > 0.5 for r in ver["agreement_with_frozen_V_SELECT"]["per_fold_pearson_r"])
    vman = json.loads((fit_dir / "frozen_manifest.json").read_text(encoding="utf-8"))
    vf = VM.assert_v_usable(vman, allow_smoke=True)
    assert R1.oof_block(fit_dir, {"usable": True, "V_frozen": vf}, allow_smoke=True)["ok"]
    if not LB.CACHE.is_dir():
        pytest.skip("match cache absent: labeller not run")
    prices = tmp_path / "prices_smoke.json"
    LB.main(["prices", "--matches", str(SMOKE / "detect" / "matches_15.14.parquet"), "--smoke", "--limit", "40",
             "--out", str(prices), "--n-boot", "20"])
    out = tmp_path / "labels"
    man = LB.main(["labels", "--patch", "15.14", "--extract", str(SMOKE / "extract" / "15.14"), "--v-manifest",
                   str(fit_dir / "frozen_manifest.json"), "--prices", str(prices), "--out", str(out), "--workers", "1",
                   "--smoke", "--allow-code-drift", "--limit-chunks", "1"])
    df = pd.read_parquet(out / "labels_15.14.parquet")
    assert man["inputs"]["v"]["source"] == "oof" and man["inputs"]["v"]["oof"]["block_source"] == LB.OOF_BLOCK_FILE
    assert (df["v_source"] == "oof").all() and df["v_fold"].tolist() == [VM.cv_fold(m) for m in df["match_id"]]
    with np.load(SMOKE / "extract" / "15.14" / "chunk_00000.npz") as z:
        X = z["eng_X"]
    for k in sorted(set(df["v_fold"])):
        sel = (df["v_fold"] == k).to_numpy()
        train = set((fit_dir / "oof_v" / f"fold_{k}" / "train_match_ids.txt").read_text(encoding="utf-8").split())
        assert not set(df.loc[sel, "match_id"]) & train
        ref = VM.load_v(fit_dir / "oof_v" / f"fold_{k}").predict(X[df.loc[sel, "extract_row"].to_numpy()])
        assert df.loc[sel, "p_pre"].to_numpy() == pytest.approx(ref)
    cov = man["oof_coverage"]
    assert cov["n"] == df["match_id"].nunique() == cov["full"]
