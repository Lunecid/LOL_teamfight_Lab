"""Tests for scripts/exact_v4/ev4_r1_record.py (record 1 DRAFT builder).

Unit tests use synthetic inputs only.  The integration tests read the stage-2 smoke outputs
(outputs/reest_exact_v4_20260925/stage2/smoke_prerun, 200 + 200 sampled matches) and the p3 v3.3 / evr summary, and
write only into pytest's tmp_path; they are skipped when the smoke outputs are absent.  No full-run output directory
is read, and nothing is written into records/.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
WT = Path(__file__).resolve().parents[1]
SCRIPTS = WT / "scripts" / "exact_v4"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_SPEC = importlib.util.spec_from_file_location("ev4_r1_record", SCRIPTS / "ev4_r1_record.py")
R1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R1)

from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH  # noqa: E402

SMOKE = R1.OUT_BASE / "stage2" / "smoke_prerun"
HAVE_SMOKE = all((SMOKE / x).exists() for x in ("detect/diag_15.14.json", "detect/diag_15.15.json",
                                                "extract/15.14/manifest.json", "extract/15.15/manifest.json",
                                                "fit_v/frozen_manifest.json"))
HAVE_P3 = all(p.is_file() for p in R1.P3_DEFAULT)
needs_smoke = pytest.mark.skipif(not (HAVE_SMOKE and HAVE_P3), reason="stage-2 smoke outputs or p3 summary absent")

SENTINELS = (987_654_321, 876_543, 765_432)       # fake held-out values; must never appear in any output


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ SMD
def test_smd_values():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert R1.smd(a, b) == pytest.approx(1.0 / s)
    assert R1.smd(b, a) == pytest.approx(-1.0 / s)
    assert R1.smd(a, a) == 0.0
    assert R1.smd(np.array([1.0]), b) is None                     # < 2 values
    assert R1.smd(np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0])) == 0.0   # NaN dropped
    assert R1.smd(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.0
    assert R1.smd(np.array([3.0, 3.0]), np.array([2.0, 2.0])) == np.inf


def _synthetic_state(n: int, rng) -> np.ndarray:
    X = np.zeros((n, len(STATE_V3_COLUMNS)), dtype=np.float32)
    ix = {c: i for i, c in enumerate(STATE_V3_COLUMNS)}
    for side, gold in (("blue", 0.4), ("red", 0.3)):
        for role in ("top", "jungle", "middle", "bottom", "utility"):
            X[:, ix[f"{side}_{role}_totalGold_norm"]] = gold + rng.normal(0, 0.01, n)
            X[:, ix[f"{side}_{role}_level_norm"]] = (10 if side == "blue" else 9) / 18.0
    X[:, ix["blue_kills"]] = 3
    X[:, ix["red_kills"]] = 2
    X[:, ix["blue_dragons"]] = 1
    X[:, ix["red_tower_OUTER_TURRET"]] = 2
    X[:, ix["blue_tower_INNER_TURRET"]] = 1
    return X


def test_covariates_from_rows():
    rng = np.random.default_rng(0)
    X = _synthetic_state(5, rng)
    eng = pd.DataFrame({"tau": [60_000, 120_000, 180_000, 240_000, 300_000], "alive_blue": [5, 4, 5, 3, 5],
                        "alive_red": [5, 5, 2, 5, 4]})
    cov = R1.covariates_from_rows(eng, X, STATE_V3_COLUMNS)
    ix = {c: i for i, c in enumerate(STATE_V3_COLUMNS)}
    gb = sum(X[0, ix[f"blue_{r}_totalGold_norm"]] for r in ("top", "jungle", "middle", "bottom", "utility"))
    gr = sum(X[0, ix[f"red_{r}_totalGold_norm"]] for r in ("top", "jungle", "middle", "bottom", "utility"))
    assert cov["gold_diff"][0] == pytest.approx((gb - gr) * 25000.0, rel=1e-5)
    assert cov["level_diff"] == pytest.approx(np.full(5, 5.0), abs=1e-4)
    assert list(cov["game_minute"]) == [1, 2, 3, 4, 5]
    assert list(cov["alive_total"]) == [10, 9, 7, 8, 9]
    assert list(cov["kills_so_far"]) == [5] * 5
    assert list(cov["dragons"]) == [1] * 5
    assert list(cov["towers"]) == [3] * 5
    assert cov["wave_phase_s"] == pytest.approx([25.0, 25.0, 25.0, 25.0, 25.0])     # (tau - 65 s) mod 30 s
    assert list(cov["early_game"]) == [1] * 5
    assert set(cov) == {n for n, _ in R1.BALANCE_COVARIATES}


def _balance_frame(shift: float, n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    cohorts = np.array(list(R1.COHORTS_V4))[rng.integers(0, 4, n)]
    clean = rng.integers(0, 2, n)
    eng = pd.DataFrame({"cohort": cohorts, "clean": clean, "isolated": 1})
    cov = {name: rng.normal(0, 1, n) for name, _ in R1.BALANCE_COVARIATES}
    cov["game_minute"] = cov["game_minute"] + shift * clean
    return eng, cov


def test_balance_rule_triggers_above_threshold():
    eng, cov = _balance_frame(shift=1.0)
    b = R1.balance_table(eng, cov)
    assert b["rule_triggered"] and b["max_abs_smd"] > 0.1
    assert b["max_at"]["covariate"] == "game_minute"
    assert b["disclosure_ko"] and b["disclosure_en"] and f"{b['max_abs_smd']:.3f}" in b["disclosure_ko"]
    assert set(b["by_cohort"]) == set(R1.COHORTS_V4) | {"all"}
    assert b["by_cohort"]["all"]["n_clean"] + b["by_cohort"]["all"]["n_nonclean"] == len(eng)


def test_balance_rule_not_triggered_when_balanced():
    eng = pd.DataFrame({"cohort": ["T", "S", "ASYM", "P"] * 4, "clean": [0] * 8 + [1] * 8, "isolated": 1})
    v = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), 2)     # same values in both groups
    cov = {name: v.copy() for name, _ in R1.BALANCE_COVARIATES}
    b = R1.balance_table(eng, cov)
    assert b["max_abs_smd"] == pytest.approx(0.0) and not b["rule_triggered"]
    assert b["disclosure_ko"] is None and b["disclosure_en"] is None


def test_balance_threshold_is_strict():
    eng, cov = _balance_frame(shift=0.0, n=2000, seed=3)
    b_max = R1.balance_table(eng, cov)["max_abs_smd"]
    assert not R1.balance_table(eng, cov, threshold=b_max)["rule_triggered"]    # == threshold is not "> threshold"
    assert R1.balance_table(eng, cov, threshold=b_max - 1e-9)["rule_triggered"]


# ------------------------------------------------------------------ p3 (held-out values never written)
def _fake_p3(tmp_path: Path) -> Path:
    blob = {"generated_utc": "x", "n_listed": SENTINELS[0], "n_loaded": SENTINELS[0], "workers": 4,
            "counts": {"ref": {"T": SENTINELS[1], "S": 5, "P": 6, "total": SENTINELS[2]}},
            "per_patch": {
                "ref": {"15.14|T": 11, "15.14|S": 12, "15.14|P": 13, "15.15|T": 21, "15.15|S": 22, "15.15|P": 23,
                        "15.16|T": SENTINELS[1], "15.16|S": 7, "15.16|P": 8},
                "evr": {"15.14|T": 31, "15.14|S": 32, "15.14|P": 33, "15.15|T": 41, "15.15|S": 42, "15.15|P": 43,
                        "15.16|T": 29856, "15.16|S": SENTINELS[2], "15.16|P": 9}}}
    p = tmp_path / "summary_fake.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


def test_read_p3_keeps_only_selection_rows(tmp_path):
    p3, held = R1.read_p3([_fake_p3(tmp_path)])
    assert p3["rows"]["ref"] == {"15.14": {"T": 11, "S": 12, "P": 13}, "15.15": {"T": 21, "S": 22, "P": 23}}
    assert p3["rows"]["evr"]["15.15"] == {"T": 41, "S": 42, "P": 43}
    text = json.dumps(p3["rows"])
    assert "15.16" not in text and not any(str(s) in json.dumps(p3) for s in SENTINELS)
    paths = {k for k, _ in held}
    assert {"per_patch/ref/15.16|T", "per_patch/evr/15.16|S", "n_listed", "counts/ref/total"} <= paths
    assert "987654321" not in repr(held)


def test_read_p3_refuses_missing_rows(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"per_patch": {"ref": {"15.14|T": 1}, "evr": {}}}), encoding="utf-8")
    with pytest.raises(R1.DraftRefused):
        R1.read_p3([p])


def test_leak_check(tmp_path):
    _p3, held = R1.read_p3([_fake_p3(tmp_path)])
    ok = R1.leak_check(["a 15.14 count 11 and the disclosed 29,856 / 29856", "sha 987654321abc"], held)
    assert ok["hits"] == 0 and ok["n_held_out_values_checked"] >= 3
    for leaked in (f"T = {SENTINELS[1]:,}", f"| {SENTINELS[2]} |", f"{SENTINELS[0]}"):
        with pytest.raises(R1.DraftRefused) as e:
            R1.leak_check(["ok text", leaked], held)
        assert e.value.code == 3
        assert not any(str(s) in str(e.value) or f"{s:,}" in str(e.value) for s in SENTINELS)


# ------------------------------------------------------------------ compliance scan
def _pq(path: Path, patch: str) -> None:
    pd.DataFrame({"patch": [patch, patch], "x": [1, 2]}).to_parquet(path)


def test_compliance_scan(tmp_path):
    root = tmp_path / "stage2"
    (root / "detect").mkdir(parents=True)
    _pq(root / "detect" / "engagements_15.14.parquet", "15.14")
    (root / "detect" / "diag_15.15.json").write_text(json.dumps({"patch": "15.15", "access": {"held_out": False}}),
                                                     encoding="utf-8")
    (root / "RUNBOOK.md").write_text("15.16 comes only after record 1\n", encoding="utf-8")
    ok = R1.compliance_scan([root])
    assert ok["ok"] and ok["n_parquet"] == 1 and ok["n_json"] == 1
    assert ok["text_mentions (informational)"] == [{"file": "stage2/RUNBOOK.md", "lines": 1}]
    assert ok["statement_ko"]

    bad = tmp_path / "bad"
    (bad / "x").mkdir(parents=True)
    _pq(bad / "x" / "rows.parquet", "15.16.702.1234")
    (bad / "x" / "m.json").write_text(json.dumps({"inputs": {"a": {"patch": "16.14"}}, "access": {"held_out": True},
                                                 "record1": "r.json"}), encoding="utf-8")
    (bad / "extract" / "15.16").mkdir(parents=True)
    (bad / "diag_15.21.json").write_text("{}", encoding="utf-8")
    res = R1.compliance_scan([bad])
    v = "\n".join(res["violations"])
    assert not res["ok"] and res["statement_ko"] is None
    for needle in ("rows.parquet: patch column", "/inputs/a/patch", "/access/held_out", "/record1 set",
                   "extract/15.16: held-out patch in the path", "diag_15.21.json: held-out patch in the path"):
        assert needle in v
    assert not R1.compliance_scan([tmp_path / "missing"])["ok"]


@pytest.mark.parametrize("name, held", [("diag_15.16.json", True), ("x_16.14_y", True), ("15.22", True),
                                        ("diag_15.14.json", False), ("chunk_00016.npz", False),
                                        ("v15.160", False), ("20260925T151437Z", False), ("115.16", False)])
def test_held_out_pattern(name, held):
    assert bool(R1.HELD_OUT_PATCH_RE.search(name)) == held


# ------------------------------------------------------------------ fixed grids and statements
def test_q_and_pt_grids_fixed():
    q = R1.Q_CANDIDATE_GRIDS
    assert q["learners"]["logistic"]["C"] == [0.001, 0.01, 0.1, 1.0]
    assert "StandardScaler" in q["learners"]["logistic"]["standardise"]
    lg = q["learners"]["lgbm"]
    assert lg["num_leaves"] == [15, 31] and lg["learning_rate"] == 0.05 and lg["min_data_in_leaf"] == 200
    assert lg["feature_fraction"] == 0.8 and lg["max_rounds"] == 2000 and lg["early_stopping"] is True
    sel = q["selection"]
    assert sel["patch"] == "15.15" and sel["tie_tol"] == 0.0005 and sel["tie_winner"] == "logistic"
    assert "Brier" in sel["metric"] and q["per_cohort"] == ["T", "S", "ASYM", "P"]
    pf = R1.PT_CANDIDATE_GRIDS["PT_flex"]
    assert "with_std=False" in pf["A8_fix"] and min(pf["C"]) < 0.01 < max(pf["C"])      # widened past the old edge
    assert R1.PT_CANDIDATE_GRIDS["PT_linear"]["features"] == ["p_pre", "time_minutes"]
    assert "29,856" in R1.PRIOR_EXPOSURE["statement_ko"] and "29,856" in R1.PRIOR_EXPOSURE["statement_en"]


def test_detect_yields_refuses_inconsistent_cells():
    cells = {c: {"clean0": {"isolated0": 1, "isolated1": 2}, "clean1": {"isolated0": 3, "isolated1": 4}}
             for c in R1.COHORTS_V4}
    d = {"patch": "15.14", "status": {"ok": 2}, "selection": {"n_list_patch": 2, "n_selected": 2},
         "v4": {"n": 40, "by_cohort": {c: 10 for c in R1.COHORTS_V4}, "cohort_x_clean_x_isolated": cells,
                "clean_isolated_by_cohort": {c: 4 for c in R1.COHORTS_V4}},
         "r2_repro": {"n": 39, "by_cohort": {}, "per_match": 19.5}, "remakes": {}}
    y = R1.detect_yields(d)
    assert y["full_run"] and y["v4"]["isolated_by_cohort"]["T"] == 6 and y["v4"]["per_ok_match"] == 20
    assert y["r2_repro"]["within_3pct"] is True
    bad = copy.deepcopy(d)
    bad["v4"]["n"] = 41
    with pytest.raises(R1.DraftRefused):
        R1.detect_yields(bad)
    bad = copy.deepcopy(d)
    bad["v4"]["clean_isolated_by_cohort"]["S"] = 5
    with pytest.raises(R1.DraftRefused):
        R1.detect_yields(bad)
    samp = copy.deepcopy(d)
    samp["selection"]["sample"] = 200
    assert R1.detect_yields(samp)["full_run"] is False


# ------------------------------------------------------------------ V gate (fake frozen manifests)
def _fake_fit_v(tmp_path: Path, bundle=None, **overrides) -> Path:
    """A self-consistent fake fit-V directory (tiny files) whose manifest links to fake extract shas."""
    d = tmp_path / "fit_v"
    (d / "V_frozen").mkdir(parents=True)
    (d / "V_frozen" / "bundle.json").write_text(json.dumps(bundle or {"format": "ev4_v_bundle_1", "kind": "logistic"}),
                                                encoding="utf-8")
    (d / "report_e4.json").write_text(json.dumps({"census": {"train_rows": 10}, "martingale": {}}), encoding="utf-8")
    bsha = _sha(d / "V_frozen" / "bundle.json")
    m = {"smoke": False, "pilot": False, "frozen": True, "chosen": "logistic",
         "selection": {"chosen": "logistic", "best_loss": 0.5, "losses": {"logistic": 0.5}, "simpler_taken": False},
         "V_frozen": {"dir": str(d / "V_frozen"), "bundle_sha256": bsha, "kind": "logistic"},
         "stop_rule": {"chosen": "logistic", "beats_gold": True, "stop_before_record1": False},
         "state_v3_name_hash": STATE_V3_NAME_HASH,
         "decisions": {"predecisions_record": {"sha256": R1.EC.PREDECISIONS_SHA256}},
         "inputs": {"train": {"manifest_sha256": "a" * 64}, "select": {"manifest_sha256": "b" * 64}},
         "files_sha256": {"V_frozen/bundle.json": bsha, "report_e4.json": _sha(d / "report_e4.json")},
         "git": {"dirty_paths": []}}
    for k, v in overrides.items():
        if v is KeyError:
            m.pop(k, None)
        else:
            m[k] = v
    (d / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d


EX_SHAS = {"15.14": "a" * 64, "15.15": "b" * 64}


def test_v_block_usable_fake(tmp_path):
    d = _fake_fit_v(tmp_path)
    v = R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    assert v["usable"] and v["chosen"] == "logistic" and v["V_frozen"]["bundle_sha256"] == _sha(d / "V_frozen/bundle.json")
    assert v["frozen_manifest"]["sha256"] == _sha(d / "frozen_manifest.json") and v["warnings"] == []
    assert R1.v_block(None, allow_smoke=False, extract_shas=EX_SHAS)["usable"] is None     # --no-fit-v: pending
    assert v["V_structure"]["bundle_format"] == "ev4_v_bundle_1" and v["recalibrated"] is False
    assert v["V_structure"]["side_marker"] is False                                        # old bundle: no side


def test_v_block_records_side_marker_and_recalibration_flag(tmp_path):
    """V revision: a format-2 bundle with side marker and recalibration is accepted and its flags are recorded; the
    manifest's V_frozen flags must agree with the bundle."""
    b2 = {"format": "ev4_v_bundle_2", "kind": "logistic", "side_marker": {"column": "side"},
          "recalibration": {"kind": "logit_recal_agebin_rcs4", "knots_minute": [5, 12, 19, 30], "terms": []}}
    d = _fake_fit_v(tmp_path / "a", bundle=b2)
    m = json.loads((d / "frozen_manifest.json").read_text(encoding="utf-8"))
    m["V_frozen"].update(side_marker=True, recalibrated=True, variant="side_marker_recalibrated")
    m["v_revision"] = {"recalibration": {"triggered": True, "fitted": True, "adopted": True}}
    (d / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    v = R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    assert v["recalibrated"] is True and v["V_structure"]["side_marker"] is True
    assert v["V_structure"]["variant"] == "side_marker_recalibrated" and v["v_revision"]["recalibration"]["adopted"]
    m["V_frozen"]["recalibrated"] = False                                                  # manifest disagrees
    (d / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(R1.DraftRefused, match="recalibrated"):
        R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    d3 = _fake_fit_v(tmp_path / "b", bundle={"format": "ev4_v_bundle_9", "kind": "logistic"})
    with pytest.raises(R1.DraftRefused, match="format"):
        R1.v_block(d3, allow_smoke=False, extract_shas=EX_SHAS)


@pytest.mark.parametrize("overrides", [
    {"stop_rule": {"chosen": "logistic", "beats_gold": False, "stop_before_record1": True}},
    {"stop_rule": KeyError},
    {"pilot": True},
    {"smoke": True},
    {"frozen": False, "V_frozen": None},
    {"V_frozen": {"dir": "x", "kind": "logistic"}},                 # no bundle sha
])
def test_v_block_refuses_not_usable(tmp_path, overrides):
    d = _fake_fit_v(tmp_path, **overrides)
    with pytest.raises(R1.DraftRefused) as e:
        R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    assert e.value.code == 2 and "not usable" in str(e.value)


def test_v_block_refuses_missing_manifest(tmp_path):
    with pytest.raises(R1.DraftRefused) as e:
        R1.v_block(tmp_path / "nothing", allow_smoke=False, extract_shas=EX_SHAS)
    assert e.value.code == 2


def test_v_block_smoke_needs_allow(tmp_path):
    d = _fake_fit_v(tmp_path, smoke=True)
    assert R1.v_block(d, allow_smoke=True, extract_shas=EX_SHAS)["smoke"] is True


def test_v_block_consistency_refusals(tmp_path):
    d = _fake_fit_v(tmp_path / "a")
    with pytest.raises(R1.DraftRefused, match="not the 15.14 extract"):
        R1.v_block(d, allow_smoke=False, extract_shas={"15.14": "c" * 64, "15.15": "b" * 64})
    (d / "report_e4.json").write_text("{}", encoding="utf-8")                  # file changed after the manifest
    with pytest.raises(R1.DraftRefused, match="changed since"):
        R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    d2 = _fake_fit_v(tmp_path / "b", state_v3_name_hash="0" * 64)
    with pytest.raises(R1.DraftRefused, match="other StateV3"):
        R1.v_block(d2, allow_smoke=False, extract_shas=EX_SHAS)
    d3 = _fake_fit_v(tmp_path / "c", decisions={"predecisions_record": {"sha256": "0" * 64}})
    with pytest.raises(R1.DraftRefused, match="pre-decision"):
        R1.v_block(d3, allow_smoke=False, extract_shas=EX_SHAS)


# ------------------------------------------------------------------ integration on the smoke outputs
@pytest.fixture(scope="module")
def smoke_build(tmp_path_factory):
    if not (HAVE_SMOKE and HAVE_P3):
        pytest.skip("smoke outputs absent")
    rec, held = R1.build_record(SMOKE / "detect", SMOKE / "extract", SMOKE / "fit_v", list(R1.P3_DEFAULT),
                                [SMOKE], smoke=True, argv=["test"])
    out = tmp_path_factory.mktemp("r1")
    jp, mp, leak = R1.write_draft(rec, held, out)
    return rec, held, jp, mp, leak


@needs_smoke
def test_smoke_draft_contents(smoke_build):
    rec, held, jp, mp, leak = smoke_build
    assert jp.name.startswith("record1_DRAFT_SMOKE_") and mp.name.endswith("_ko.md")
    blob = json.loads(jp.read_text(encoding="utf-8"))
    assert blob["status"] == "DRAFT" and blob["is_final_lock"] is False and blob["usable_as_record1"] is False
    assert not str(blob["record"]).upper().startswith("1A") and blob["smoke"] is True
    assert blob["author_signoff"]["signed"] is False
    assert blob["preset"]["values"]["TF2_KILL_CLUSTER_GAP_MS"] == 14000
    assert blob["preset"]["values_sha256"] == json.loads((SMOKE / "detect/diag_15.14.json").read_text(
        encoding="utf-8"))["preset"]["preset_values_sha256"]
    assert blob["state_columns"]["STATE_V3_NAME_HASH"] == STATE_V3_NAME_HASH
    recs = blob["records"]
    assert recs["predecisions_pinned"]["sha256"] == R1.EC.PREDECISIONS_SHA256
    assert "stage2_predecisions_20260925T151437Z.json" in recs["decision_records"]
    assert all(not k.startswith("record1_") for k in recs["files"])
    # yields: detect diag numbers reproduced
    y = blob["detect"]["15.14"]
    assert y["v4"]["n"] == sum(c["n"] for c in y["v4"]["cells"]) and y["n_ok"] == y["status"]["ok"]
    assert y["remakes"]["n_excluded"] == len(y["remakes"]["match_ids"])
    # E2 rows: six stages, legacy rows equal the p3 15.14 values
    p3 = json.loads(R1.P3_DEFAULT[0].read_text(encoding="utf-8"))
    t = blob["e2_cumulative"]["tables"]["15.14"]
    assert [r["stage"] for r in t["rows"]] == ["v3.3_legacy", "evr", "r2_repro", "v4_all", "v4_isolated",
                                                "v4_isolated_clean"]
    assert t["rows"][0]["by_cohort"]["T"] == p3["per_patch"]["ref"]["15.14|T"]
    assert t["rows"][1]["by_cohort"]["S"] == p3["per_patch"]["evr"]["15.14|S"]
    assert t["rows"][5]["by_cohort"] == blob["detect"]["15.14"]["v4"]["clean_isolated_by_cohort"]
    assert t["v4_is_sample"] is True and t["legacy_denominator_same_cache"] is True
    # balance: 15.14 extract, every isolated row used
    b = blob["balance_15_14"]
    n = b["by_cohort"]["all"]["n_clean"] + b["by_cohort"]["all"]["n_nonclean"]
    assert n == blob["extract"]["15.14"]["rows"]["eng"]
    assert b["by_cohort"]["all"]["n_clean"] == blob["extract"]["15.14"]["rows"]["eng_clean_setup"]
    assert b["rule_triggered"] == (b["max_abs_smd"] > 0.1)
    assert (b["disclosure_ko"] is not None) == b["rule_triggered"]
    # V, grids, statements
    assert blob["V"]["usable"] is True and blob["V"]["smoke"] is True
    assert blob["q_candidate_grids"]["learners"]["lgbm"]["num_leaves"] == [15, 31]
    assert blob["compliance_no_heldout_before_record1"]["ok"] is True
    ids = [d["id"] for d in blob["deviations"]]
    for need in ("DEV-side-marker", "DEV-E1-sequential", "DEV-items-strict-exception", "DEV-R15-5-removed"):
        assert need in ids
    assert all(d.get("sha256") for d in blob["deviations"])
    assert len(blob["e1_disclosures"]) >= 5


@needs_smoke
def test_smoke_draft_writes_no_held_out_value(smoke_build):
    rec, held, jp, mp, leak = smoke_build
    text = jp.read_text(encoding="utf-8") + mp.read_text(encoding="utf-8")
    assert leak["hits"] == 0 and leak["n_held_out_values_checked"] > 0
    R1.leak_check([text], held)                                # re-check the files as written
    assert "15.16|" not in text                               # no p3 15.16 key copied
    md = mp.read_text(encoding="utf-8")
    assert "29,856" in md and "기록 1 초안" in md


@needs_smoke
def test_smoke_refusals(smoke_build, tmp_path):
    rec, held, *_ = smoke_build
    with pytest.raises(R1.DraftRefused, match="records/"):
        R1.write_draft(rec, held, R1.RECORDS_DIR)             # a smoke draft never goes into records/
    with pytest.raises(R1.DraftRefused, match="--smoke"):
        R1.build_record(SMOKE / "detect", SMOKE / "extract", SMOKE / "fit_v", list(R1.P3_DEFAULT), [SMOKE],
                        smoke=False)
    # a not-usable V (stop rule says stop) refuses the draft even with --smoke
    fv = tmp_path / "fit_v"
    shutil.copytree(SMOKE / "fit_v", fv)
    m = json.loads((fv / "frozen_manifest.json").read_text(encoding="utf-8"))
    m["stop_rule"]["stop_before_record1"] = True
    (fv / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(R1.DraftRefused) as e:
        R1.build_record(SMOKE / "detect", SMOKE / "extract", fv, list(R1.P3_DEFAULT), [SMOKE], smoke=True)
    assert e.value.code == 2


@needs_smoke
def test_smoke_v_block_non_smoke_fake(tmp_path):
    """The real smoke fit-V copied and marked non-smoke: usable without allow_smoke, all file hashes verified."""
    fv = tmp_path / "fit_v"
    shutil.copytree(SMOKE / "fit_v", fv)
    m = json.loads((fv / "frozen_manifest.json").read_text(encoding="utf-8"))
    m["smoke"] = False
    m["V_frozen"]["dir"] = str(fv / "V_frozen")
    (fv / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    shas = {p: _sha(SMOKE / "extract" / p / "manifest.json") for p in ("15.14", "15.15")}
    v = R1.v_block(fv, allow_smoke=False, extract_shas=shas)
    assert v["usable"] and v["chosen"] == m["chosen"] and v["smoke"] is False
    assert v["warnings"]                                      # the smoke fit-V ran with dirty code


@needs_smoke
def test_cli_exit_codes(tmp_path, capsys):
    base = ["--detect", str(SMOKE / "detect"), "--extract", str(SMOKE / "extract"), "--scan-root", str(SMOKE)]
    assert R1.main(base + ["--fit-v", str(SMOKE / "fit_v"), "--smoke", "--out-dir", str(R1.RECORDS_DIR)]) == 1
    assert R1.main(base + ["--fit-v", str(tmp_path / "none"), "--smoke", "--out-dir", str(tmp_path)]) == 2
    assert R1.main(base + ["--no-fit-v", "--out-dir", str(tmp_path)]) == 1        # sample inputs without --smoke
    assert R1.main(base + ["--no-fit-v", "--smoke", "--out-dir", str(tmp_path), "--no-verify-chunks"]) == 0
    out = capsys.readouterr().out
    assert "draft:" in out and "PENDING" in out
    js = list(tmp_path.glob("record1_DRAFT_SMOKE_*.json"))
    assert len(js) == 1 and json.loads(js[0].read_text(encoding="utf-8"))["V"]["usable"] is None


# ================================================================== Stage 2c review fixes
from tests import _ev4_lock_helpers as H  # noqa: E402


def test_wave_phase_and_early_game_covariates():
    rng = np.random.default_rng(0)
    taus = [65_000, 80_000, 94_999, 839_999, 840_000, 905_000]
    X = _synthetic_state(len(taus), rng)
    eng = pd.DataFrame({"tau": taus, "alive_blue": [5] * len(taus), "alive_red": [5] * len(taus)})
    cov = R1.covariates_from_rows(eng, X, STATE_V3_COLUMNS)
    assert cov["wave_phase_s"] == pytest.approx([0.0, 15.0, 29.999, 24.999, 25.0, 0.0])
    assert list(cov["early_game"]) == [1, 1, 1, 1, 0, 0]
    assert cov["wave_phase_s_early"][:4] == pytest.approx([0.0, 15.0, 29.999, 24.999])
    assert np.isnan(cov["wave_phase_s_early"][4:]).all()
    names = [n for n, _ in R1.BALANCE_COVARIATES]
    assert {"wave_phase_s", "early_game", "wave_phase_s_early"} <= set(names)


def test_balance_table_wave_rows_and_nan_means():
    eng, cov = _balance_frame(shift=0.0)
    cov["wave_phase_s_early"] = np.full(len(eng), np.nan)          # e.g. no early-game row at all
    b = R1.balance_table(eng, cov, gold_info=R1.gold_field(STATE_V3_COLUMNS))
    row = b["by_cohort"]["all"]["covariates"]["wave_phase_s_early"]
    assert row == {"smd": None, "mean_clean": None, "mean_nonclean": None}
    assert "wave_phase_s" in b["by_cohort"]["T"]["covariates"] and b["wave_phase"]["period_ms"] == 30_000
    assert b["gold_field"]["field"] == "totalGold_norm" and b["gold_field"]["event_updated"] is False


def test_gold_field_prefers_an_event_updated_column():
    gf = R1.gold_field(STATE_V3_COLUMNS)
    assert gf["field"] == R1.FRAME_GOLD_FIELD and "no event-updated gold column" in gf["source"]
    names = list(STATE_V3_COLUMNS) + [p + "totalGold_event_norm" for p in R1.SLOT_BLUE + R1.SLOT_RED]
    ev = R1.gold_field(names)
    assert ev["field"] == "totalGold_event_norm" and ev["event_updated"] is True
    # the covariate then reads that column
    rng = np.random.default_rng(3)
    X = np.hstack([_synthetic_state(2, rng), np.zeros((2, 10), np.float32)])
    X[:, len(STATE_V3_COLUMNS)] = 1.0                                   # blue top event gold = 25,000
    eng = pd.DataFrame({"tau": [60_000, 60_000], "alive_blue": [5, 5], "alive_red": [5, 5]})
    assert R1.covariates_from_rows(eng, X, names)["gold_diff"] == pytest.approx([25_000.0, 25_000.0])
    with pytest.raises(R1.DraftRefused, match="gold"):
        R1.gold_field(["x"])


def test_leak_check_exemption_is_exact_path_and_value():
    held = R1.HeldOutValues()
    held.add("per_patch/evr/15.16|T", 29856)
    assert R1.leak_check(["T = 29,856 (disclosed)"], held)["n_exempt"] == 1
    other_path = R1.HeldOutValues()
    other_path.add("counts/evr/T", 29856)                               # same value, another key path
    with pytest.raises(R1.DraftRefused) as e:
        R1.leak_check(["T = 29,856 (disclosed)"], other_path)
    assert e.value.code == 3 and "counts/evr/T" in str(e.value)
    other_value = R1.HeldOutValues()
    other_value.add("per_patch/evr/15.16|T", 31234)                     # the disclosed path with another value
    with pytest.raises(R1.DraftRefused):
        R1.leak_check(["T = 31,234"], other_value)
    assert R1.leak_check(["nothing here"], other_value)["n_held_out_values_checked"] == 1


def test_no_verify_chunks_and_drift_need_smoke(tmp_path, capsys):
    with pytest.raises(R1.DraftRefused, match="smoke"):
        R1.build_record(tmp_path / "d", tmp_path / "e", None, [], [], smoke=False, verify_chunks=False)
    assert R1.main(["--no-fit-v", "--no-verify-chunks", "--detect", str(tmp_path / "d"), "--extract",
                    str(tmp_path / "e"), "--out-dir", str(tmp_path)]) == 1
    assert "--no-verify-chunks" in capsys.readouterr().err
    drift = {"extract_manifest_15.14": {"changed": {"gameplay/a.py": {}, "gameplay/b.py": {}}, "not_hashed_now": []},
             "detect_diag_15.14": {"changed": {}, "not_hashed_now": []}}
    with pytest.raises(R1.DraftRefused, match="gameplay/b.py"):
        R1.drift_policy(drift, ["gameplay/a.py"], smoke=False)
    ok = R1.drift_policy(drift, ["gameplay/a.py", "gameplay\\b.py", "core/c.py"], smoke=False)
    assert ok["allowed_by"] == "--allow-drift" and ok["listed_but_unchanged"] == ["core/c.py"]
    sm = R1.drift_policy(drift, [], smoke=True)
    assert sm["allowed_by"] == "smoke" and sm["unlisted_allowed_by_smoke"] == ["gameplay/a.py", "gameplay/b.py"]
    assert R1.drift_policy({"x": {"changed": {}}}, [], smoke=False)["allowed_by"] is None


def test_main_does_not_crash_when_max_smd_is_none(tmp_path, monkeypatch, capsys):
    rec = {"balance_15_14": {"max_abs_smd": None, "max_at": {"cohort": None, "covariate": None, "smd": None},
                             "rule_triggered": False},
           "V": {"status": "PENDING"}, "lockable": False,
           "pending_blocking": [{"id": "B01", "text": "balance: no SMD could be computed"}]}
    jp = tmp_path / "record1_DRAFT_SMOKE_x.json"
    jp.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(R1, "build_record", lambda *a, **k: (rec, R1.HeldOutValues()))
    monkeypatch.setattr(R1, "write_draft", lambda *a, **k: (jp, tmp_path / "x.md", {"n_held_out_values_checked": 0}))
    assert R1.main(["--no-fit-v", "--smoke", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "not computable" in out and "lockable: False" in out
    # a group with < 2 rows gives max |SMD| None and a blocking item
    eng = pd.DataFrame({"cohort": ["T", "T"], "clean": [1, 0], "isolated": [1, 1]})
    b = R1.balance_table(eng, {n: np.array([1.0, 2.0]) for n, _ in R1.BALANCE_COVARIATES})
    assert b["max_abs_smd"] is None and b["rule_triggered"] is False


def _fake_oof(d: Path, vb: str, **over):
    """A fake oof_v export (the files ev4_04_labels.OOF_SPEC names) linked from the fake fit-V manifest."""
    oof = d / "oof_v"
    folds = {}
    for k in range(5):
        fd = oof / f"fold_{k}"
        fd.mkdir(parents=True)
        (fd / "bundle.json").write_text(json.dumps({"fold": k}), encoding="utf-8")
        (fd / "train_match_ids.txt").write_text(f"KR_{k}\n", encoding="utf-8")
        folds[str(k)] = {"dir": f"fold_{k}", "bundle_sha256": _sha(fd / "bundle.json"), "heldout_fold": k,
                         "train_match_ids_file": f"fold_{k}/train_match_ids.txt",
                         "train_match_ids_sha256": _sha(fd / "train_match_ids.txt"), "n_train_matches": 1}
    om = {"format": R1.OOF_FORMAT, "train_patch": "15.14", "kind": "logistic", "hyperparameter": {"C": 0.01},
          "n_folds": 5, "fold_rule": "x", "frozen_bundle_sha256": vb, "smoke": False, "pilot": False, "folds": folds}
    om.update(over)
    (oof / "oof_manifest.json").write_text(json.dumps(om), encoding="utf-8")
    m = json.loads((d / "frozen_manifest.json").read_text(encoding="utf-8"))
    m["oof_v"] = {"dir": str(oof), "manifest_sha256": _sha(oof / "oof_manifest.json"), "kind": "logistic"}
    (d / "frozen_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return oof


def test_oof_block(tmp_path):
    d = _fake_fit_v(tmp_path / "a")
    v = R1.v_block(d, allow_smoke=False, extract_shas=EX_SHAS)
    pend = R1.oof_block(d, v, allow_smoke=False)                       # no oof_v block yet
    assert pend["ok"] is False and pend["status"].startswith("PENDING") and "oof_v" in pend["status"]
    assert R1.oof_block(None, {"usable": None}, allow_smoke=False)["ok"] is False
    _fake_oof(d, v["V_frozen"]["bundle_sha256"])
    ok = R1.oof_block(d, v, allow_smoke=False)
    assert ok["ok"] and sorted(ok["fold_bundle_sha256"]) == ["0", "1", "2", "3", "4"]
    assert ok["fold_bundle_sha256"]["3"] == _sha(d / "oof_v" / "fold_3" / "bundle.json")
    assert ok["manifest_sha256"] == _sha(d / "oof_v" / "oof_manifest.json")
    (d / "oof_v" / "fold_2" / "bundle.json").write_text("{}", encoding="utf-8")      # tampered fold bundle
    with pytest.raises(R1.DraftRefused, match="fold 2"):
        R1.oof_block(d, v, allow_smoke=False)
    d2 = _fake_fit_v(tmp_path / "b")
    _fake_oof(d2, "0" * 64)                                             # another V run's OOF set
    with pytest.raises(R1.DraftRefused, match="frozen_bundle"):
        R1.oof_block(d2, v, allow_smoke=False)
    d3 = _fake_fit_v(tmp_path / "c")
    _fake_oof(d3, v["V_frozen"]["bundle_sha256"], smoke=True)
    with pytest.raises(R1.DraftRefused, match="smoke"):
        R1.oof_block(d3, v, allow_smoke=False)
    assert R1.oof_block(d3, v, allow_smoke=True)["ok"]


def _prices(tmp_path: Path, **over) -> Path:
    b = {"format": R1.PRICES_FORMAT, "patch": "15.14", "source_patches": ["15.14"], "sample": False,
         "table": {"kills": 300.0, "assists": 150.0}, "fit": {"n_matches": 12000},
         "matches": {"source_sha256": "f" * 64}}
    b.update(over)
    p = tmp_path / "prices_1514.json"
    p.write_text(json.dumps(b), encoding="utf-8")
    return p


def test_prices_block(tmp_path):
    assert R1.prices_block(None, allow_smoke=False)["ok"] is False
    assert R1.prices_block(tmp_path / "none.json", allow_smoke=False)["status"].startswith("PENDING")
    p = _prices(tmp_path)
    ok = R1.prices_block(p, allow_smoke=False)
    assert ok["ok"] and ok["sha256"] == _sha(p) and ok["n_matches"] == 12000
    for over in ({"patch": "15.15"}, {"source_patches": ["15.14", "15.15"]}, {"format": "x"}):
        with pytest.raises(R1.DraftRefused, match="only"):
            R1.prices_block(_prices(tmp_path, **over), allow_smoke=False)
    with pytest.raises(R1.DraftRefused, match="sample"):
        R1.prices_block(_prices(tmp_path, sample=True), allow_smoke=False)
    assert R1.prices_block(_prices(tmp_path, sample=True), allow_smoke=True)["ok"]


def _yields_e2():
    y = {p: {"r2_repro": {"within_3pct": True}} for p in R1.PATCHES}
    e2 = {p: {"counts_comparable": True, "legacy_denominator_same_cache": True} for p in R1.PATCHES}
    return y, e2


def test_pending_items_split_blocking_and_author():
    y, e2 = _yields_e2()
    bal = {"rule_triggered": False, "max_abs_smd": 0.05, "gold_field": R1.gold_field(STATE_V3_COLUMNS)}
    ok = {"ok": True}
    blocking, author = R1.pending_items({"usable": True}, bal, e2, y, {}, ok, ok)
    assert blocking == [] and [a["id"] for a in author] == ["A01", "A02", "A03", "A04"]
    assert "totalGold_norm" in author[3]["text"]
    blocking, _ = R1.pending_items({"usable": None}, bal, e2, y, {}, {"ok": False, "status": "PENDING: oof"},
                                   {"ok": False, "status": "PENDING: prices"})
    assert [b["id"] for b in blocking] == ["B01", "B02", "B03"]
    assert "oof_v" in blocking[1]["text"] and "prices_1514" in blocking[2]["text"]
    y["15.15"]["r2_repro"]["within_3pct"] = False
    e2["15.14"]["counts_comparable"] = False
    blocking, _ = R1.pending_items({"usable": True}, dict(bal, max_abs_smd=None), e2, y, {}, ok, ok)
    assert len(blocking) == 3


@needs_smoke
def test_smoke_draft_top_level_v_and_pending(smoke_build):
    rec, *_ = smoke_build
    assert rec["V_frozen_bundle_sha256"] == rec["V"]["V_frozen"]["bundle_sha256"]
    assert R1.RL.HEX64.match(rec["V_frozen_bundle_sha256"])
    assert rec["oof_v"]["ok"] is False and rec["prices_1514"]["ok"] is False       # not given to the smoke build
    assert rec["lockable"] is False
    texts = " ".join(b["text"] for b in rec["pending_blocking"])
    assert "oof_v" in texts and "prices_1514" in texts and "sample" in texts
    assert rec["balance_15_14"]["gold_field"]["field"] == "totalGold_norm"
    assert "wave_phase_s" in rec["balance_15_14"]["by_cohort"]["all"]["covariates"]


# ------------------------------------------------------------------ lock (smoke only, temp records dir)
def _draft(tmp_path: Path, smoke: bool = True, **over):
    rd, pre = H.make_records_dir(tmp_path)
    b = H.draft_body(rd, smoke=smoke)
    b["records"]["predecisions_pinned"]["sha256"] = pre
    b.update(over)
    dd = tmp_path / "drafts"
    dd.mkdir(exist_ok=True)
    dp = dd / f"{R1.DRAFT_PREFIX}{'SMOKE_' if smoke else ''}20260927T000000Z.json"
    dp.write_text(json.dumps(b), encoding="utf-8")
    res = tmp_path / "resolutions.json"
    res.write_text(json.dumps({"A01": "grid confirmed", "A02": "wording confirmed"}), encoding="utf-8")
    return rd, pre, dp, _sha(dp), res


def _lock_argv(dp, sha, res, rd, signoff="Author Name / 2026-09-27", smoke=True):
    return (["lock", "--draft", str(dp), "--draft-sha256", sha, "--author-signoff", signoff, "--resolutions",
             str(res), "--records-dir", str(rd)] + (["--smoke"] if smoke else []))


def test_lock_smoke_draft_into_temp_records(tmp_path, monkeypatch, capsys):
    rd, pre, dp, sha, res = _draft(tmp_path)
    monkeypatch.setattr(R1.RL.EC, "PREDECISIONS_SHA256", pre)          # the fake records dir's pin (test only)
    assert R1.main(_lock_argv(dp, sha, res, rd)) == 0
    outs = sorted(rd.glob("record1_SMOKE_*.json"))
    assert len(outs) == 1 and "locked record 1" in capsys.readouterr().out
    rec = json.loads(outs[0].read_text(encoding="utf-8"))
    assert rec["record"] == "1" and rec["status"] == "LOCKED" and rec["is_final_lock"] is True
    assert rec["usable_as_record1"] is True and rec["smoke"] is True and rec["pending_blocking"] == []
    assert rec["author_signoff"]["signed"] is True and rec["author_signoff"]["signed_by"] == "Author Name"
    assert rec["author_signoff"]["signed_date"] == "2026-09-27"
    assert rec["author_resolutions"] == {"A01": "grid confirmed", "A02": "wording confirmed"}
    assert rec["locked_from_draft"]["sha256"] == sha and rec["V_frozen_bundle_sha256"] == H.V_SHA
    # the smoke lock is never usable as record 1
    with pytest.raises(R1.RL.RecordNotLocked, match="smoke"):
        R1.RL.assert_record1_locked(outs[0], _sha(outs[0]), records_dir=rd, predecisions_sha256=pre)
    # one lock only
    assert R1.main(_lock_argv(dp, sha, res, rd)) == 1
    assert "already exists" in capsys.readouterr().err


def _refused(tmp_path, argv, match, capsys):
    assert R1.main(argv) == 1
    err = capsys.readouterr().err
    assert match in err, err
    assert not list(Path(argv[argv.index("--records-dir") + 1]).glob("record1_*.json"))


def test_lock_refusals(tmp_path, monkeypatch, capsys):
    rd, pre, dp, sha, res = _draft(tmp_path)
    monkeypatch.setattr(R1.RL.EC, "PREDECISIONS_SHA256", pre)
    _refused(tmp_path, _lock_argv(dp, sha, res, rd, smoke=False), "smoke draft locks only", capsys)
    _refused(tmp_path, _lock_argv(dp, "0" * 64, res, rd), "not the approved", capsys)
    for bad in ("Author", "2026-09-27", " / 2026-09-27", "Author / 27.09.2026"):
        _refused(tmp_path, _lock_argv(dp, sha, res, rd, signoff=bad), "author-signoff", capsys)
    res.write_text(json.dumps({"A01": "ok"}), encoding="utf-8")
    _refused(tmp_path, _lock_argv(dp, sha, res, rd), "without a resolution ['A02']", capsys)
    res.write_text(json.dumps({"A01": "ok", "A02": "ok", "A99": "?"}), encoding="utf-8")
    _refused(tmp_path, _lock_argv(dp, sha, res, rd), "unknown ids ['A99']", capsys)
    # a smoke lock into the real records/ is refused before anything is read
    assert R1.main(_lock_argv(dp, sha, res, R1.RECORDS_DIR)) == 1
    assert "records/" in capsys.readouterr().err
    # a file that is not a draft
    other = tmp_path / "drafts" / "record1_20260927T000000Z.json"
    other.write_bytes(dp.read_bytes())
    _refused(tmp_path, _lock_argv(other, _sha(other), res, rd), "not a record1_DRAFT", capsys)


@pytest.mark.parametrize("over, match", [
    ({"lockable": False}, "not lockable"),
    ({"pending_blocking": [{"id": "B01", "text": "prices"}]}, "not lockable"),
    ({"V_frozen_bundle_sha256": None}, "V_frozen_bundle_sha256"),
    ({"oof_v": {"ok": False, "status": "PENDING: oof"}}, "oof_v"),
    ({"prices_1514": {"ok": False, "status": "PENDING: prices"}}, "prices_1514"),
    ({"status": "LOCKED"}, "unlocked record 1 DRAFT"),
    ({"record1a": {"path": "x/record1a_other.json", "sha256": "0" * 64}}, "record 1A"),
])
def test_lock_refuses_unlockable_drafts(tmp_path, monkeypatch, capsys, over, match):
    rd, pre, dp, sha, res = _draft(tmp_path, **over)
    monkeypatch.setattr(R1.RL.EC, "PREDECISIONS_SHA256", pre)
    _refused(tmp_path, _lock_argv(dp, sha, res, rd), match, capsys)


def test_real_lock_needs_the_author_at_a_terminal(tmp_path, monkeypatch):
    """A non-smoke draft is refused without a TTY (agents and scripts); nothing is written.  The successful real
    lock is not exercised here: it is the author's step."""
    rd, pre, dp, sha, res = _draft(tmp_path, smoke=False)
    monkeypatch.setattr(R1.RL.EC, "PREDECISIONS_SHA256", pre)

    class _NoTTY:
        def isatty(self):
            return False
    monkeypatch.setattr(R1.sys, "stdin", _NoTTY())
    with pytest.raises(R1.DraftRefused, match="interactive terminal"):
        R1.lock_record(dp, sha, "Author / 2026-09-27", res, records_dir=rd, smoke=False)
    assert not list(rd.glob("record1_*.json"))

    class _TTY:
        def isatty(self):
            return True
    with pytest.raises(R1.DraftRefused, match="not typed"):
        R1._interactive_confirm(sha, stdin=_TTY(), prompt=lambda _m: "LOCK 000000000000")
    R1._interactive_confirm(sha, stdin=_TTY(), prompt=lambda _m: f"LOCK {sha[:12]}")      # typed correctly: passes
