"""External validation, step 3b: the one evaluation pass (EXTERNAL_PROTOCOL_20261009.json "metrics" and "reporting").

Preconditions, checked here: EXTERNAL_LOCK_CHECK.json passed for the current EXTERNAL_SCORE_LOCK.json, and every file
bound by EXTERNAL_SCORE_LOCK_ADDENDUM.json (nominal V predictions) re-hashes equal.  Then the sealed labels are opened
once (EXTERNAL_LABELS_OPENED.json records when) and, per set (KR 15.18-15.22 also per patch):
  - q, h90 labels: per cell match-weighted AUC, Brier, logloss, accuracy and calibration slope on raw probabilities
    (calibrated beside); P1, P2 (C=1), P2 (C=0), P1-best and the LightGBM position effect (mean over D) from
    qeval.paired_match_bootstrap (2000 draws, seed 20261011; P1 and P2 (C=1) at 97.5 %, the rest at 95 %);
  - descriptive: rows, matches, positive rate (match weighted and by row), exact zeros, |SVI| < 0.01 share;
  - V: calibrated final V for the match winner at tau (engagement pre states, match weight 1/m over engagements, as
    v_test_tau.py tau) and on the nominal V rows (v_report.report, as v_test_tau.py report).
The TEST (15.16) values from TEST_REPORT.json, V_TEST_REPORT_final.json and V_TAU_REPORT_final.json are copied beside
for comparison.  Nothing is selected or re-run.  Writes reports/external_validation_20261009/EXTERNAL_REPORT.json once.

Usage: python -B studies/external_validation_20261009/e3_evaluate.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

DATA = Path(r"D:\LOL_External_20261009")
REPORTS = ROOT / "reports/external_validation_20261009"
LOCK = REPORTS / "EXTERNAL_SCORE_LOCK.json"
ADDENDUM = REPORTS / "EXTERNAL_SCORE_LOCK_ADDENDUM.json"
CHECK = REPORTS / "EXTERNAL_LOCK_CHECK.json"
OPENED = REPORTS / "EXTERNAL_LABELS_OPENED.json"
OUT = REPORTS / "EXTERNAL_REPORT.json"
DATASETS = ("kr_15_18_22", "kr_16_13", "na1_16_13")
PATCHES_2025 = ("15.18", "15.19", "15.20", "15.21", "15.22")
CONTRAST_NAMES = ("P1", "P2_C1", "P2_C0", "P1_best", "S_effect_LGBM_meanD")
LEVELS = {"P1": 0.975, "P2_C1": 0.975, "P2_C0": 0.95, "P1_best": 0.95, "S_effect_LGBM_meanD": 0.95}
METRICS = ("auc", "brier", "logloss", "accuracy", "calibration_slope", "calibration_intercept")
TEST_REPORT = ROOT / "reports/q_production_20261005/test_results_20261008/TEST_REPORT.json"
V_TEST = Path(r"D:\LOL_Training_20261004\v_report_1\V_TEST_REPORT_final.json")
V_TAU = Path(r"D:\LOL_Training_20261004\v_report_1\V_TAU_REPORT_final.json")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def preconditions() -> dict:
    check = json.loads(CHECK.read_text(encoding="utf-8"))
    if not check.get("passed") or check.get("lock_sha256") != sha_file(LOCK):
        raise SystemExit("EXTERNAL_LOCK_CHECK.json does not cover the current lock")
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if addendum["lock"]["sha256"] != check["lock_sha256"] or not addendum["lock_check"]["passed"]:
        raise SystemExit("the addendum names another score lock or a failed lock check")
    for flag in ("sealed_files_opened", "label_values_read", "match_winner_read", "metrics_computed"):
        if addendum.get(flag) is not False:
            raise SystemExit(f"addendum flag {flag} is not False")
    bad = []
    for item in _addendum_files(addendum):
        if sha_file(Path(item["path"])) != item["sha256"]:
            bad.append(item["path"])
    if bad:
        raise SystemExit(f"addendum files changed: {bad[:3]}")
    return {"lock_sha256": check["lock_sha256"], "addendum_sha256": sha_file(ADDENDUM), "addendum_files_checked": len(_addendum_files(addendum))}


def _addendum_files(addendum: dict) -> list[dict]:
    out = []

    def walk(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node and isinstance(node["path"], str):
                out.append({"path": node["path"], "sha256": node["sha256"]})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(addendum)
    return out


def q_block(y, preds, preds_cal, match, cons) -> dict:
    from studies.q_production_20261005 import qeval

    out = {"rows": int(len(y)), "matches": int(len(np.unique(match))),
           "positive_rate_match_weighted": float(np.average(y, weights=qeval.match_weights(match))), "positive_rate_rows": float(y.mean()),
           "metrics": {}}
    for key in preds:
        raw = qeval.evaluate(y, preds[key], match)["match_weighted"]
        cal = qeval.evaluate(y, preds_cal[key], match)["match_weighted"]
        out["metrics"][key] = {"raw": {m: raw[m] for m in METRICS}, "calibrated": {m: cal[m] for m in METRICS}}
    losses = {k: qeval.per_match_loss(y, p, match) for k, p in preds.items()}
    shared, aligned = qeval.align_cells(losses)
    boot = qeval.paired_match_bootstrap(shared, aligned, cons, levels=LEVELS)
    out["contrasts"] = {n: {"point": c["point"], "interval": c["interval"], "level": c["level"]} for n, c in boot["contrasts"].items()}
    return out


def v_tau_block(y_win, p_pre, match) -> dict:
    from studies.q_production_20261005 import qeval

    ev = qeval.evaluate(y_win, p_pre, match)["match_weighted"]
    return {m: ev[m] for m in METRICS}


def v_nominal_block(dataset: str, addendum: dict) -> dict:
    """Final V on the nominal V rows, y = match winner (projection MATCH_TABLE), as v_test_tau.py report: overall raw and
    calibrated match-weighted metrics, game-time strata, frame-age strata [0,5] / (5,10] / (10,inf) s, and per patch."""
    from studies.q_production_20261005 import qeval

    info = addendum["datasets"][dataset]
    pred = npz(Path(info["pred_nominal_final"]["path"]))
    table_path = (DATA / "parity_15_16" / "projection" if dataset == "parity_15_16" else DATA / "projection" / dataset) / "MATCH_TABLE.npy"
    if sha_file(table_path) != info["projection_tables_sha256"]["MATCH_TABLE.npy"]:
        raise SystemExit(f"{dataset}: MATCH_TABLE differs from the addendum")
    matches = np.load(table_path, mmap_mode="r", allow_pickle=False)
    m = np.asarray(pred["match_index"], dtype=np.int64)
    if not np.array_equal(np.asarray(matches["key"])[m], pred["canonical_match_key"]):
        raise RuntimeError(f"{dataset}: nominal rows and MATCH_TABLE keys disagree")
    y = np.asarray(matches["target_Blue_win"], dtype=np.float64)[m]
    p_raw, p_cal = np.asarray(pred["raw_p"], dtype=np.float64), np.asarray(pred["v"], dtype=np.float64)
    minutes = np.asarray(pred["query_ms"], dtype=np.float64) / 60000.0
    age = np.asarray(pred["frame_age_seconds"], dtype=np.float64)
    patch = np.asarray(pred["patch"]).astype(str)

    def block(mask: np.ndarray) -> dict:
        out = {"rows": int(mask.sum()), "matches": int(len(np.unique(m[mask]))), "positive_rate_rows": float(y[mask].mean())}
        for kind, p in (("raw", p_raw), ("calibrated", p_cal)):
            ev = qeval.evaluate(y[mask], p[mask], m[mask])["match_weighted"]
            out[kind] = {k: ev[k] for k in METRICS}
        game = np.asarray(qeval.game_time_strata(minutes[mask]))
        out["game_time_raw_auc"] = {s: float(qeval.evaluate(y[mask][game == s], p_raw[mask][game == s], m[mask][game == s])["match_weighted"]["auc"])
                                    for s in qeval.GAME_TIME_ORDER if (game == s).sum() >= 100}
        ages = np.where(age[mask] <= 5, "[0,5]", np.where(age[mask] <= 10, "(5,10]", "(10,inf)"))
        out["frame_age_raw_auc"] = {s: float(qeval.evaluate(y[mask][ages == s], p_raw[mask][ages == s], m[mask][ages == s])["match_weighted"]["auc"])
                                    for s in ("[0,5]", "(5,10]", "(10,inf)") if (ages == s).sum() >= 100}
        return out

    result = {"all": block(np.ones(len(y), dtype=bool))}
    if dataset == "kr_15_18_22":
        result["by_patch"] = {p: block(patch == p) for p in PATCHES_2025}
    return result


def evaluate_dataset(dataset: str, cons: dict, addendum: dict) -> dict:
    from studies.q_production_20261005 import qeval

    rows = npz(DATA / "q_inputs" / dataset / "Q_ROWS_h90.npz")
    keys = rows["engagement_key"]
    sealed = npz(DATA / "labels" / dataset / f"EXTERNAL_LABELS_SEALED_{dataset}.npz")
    index = {k: i for i, k in enumerate(sealed["h90__engagement_key"].tolist())}
    pick = np.asarray([index[k] for k in keys.tolist()], dtype=np.int64)
    if not np.asarray(sealed["h90__valid"], dtype=bool)[pick].all():
        raise RuntimeError(f"{dataset}: a q row has an invalid h90 label")
    y = np.asarray(sealed["h90__y"], dtype=np.float64)[pick]
    svi = np.asarray(sealed["h90__svi_delta"], dtype=np.float64)[pick]
    exact_zero = np.asarray(sealed["h90__exact_zero"], dtype=bool)[pick]
    match = np.asarray(rows["match_index"])
    preds, preds_cal = {}, {}
    for key in sorted({k for w in cons.values() for k in w}):
        s = npz(DATA / "scores" / dataset / f"{key}.npz")
        if not np.array_equal(s["engagement_key"], keys):
            position = {k: i for i, k in enumerate(s["engagement_key"].tolist())}
            order = np.asarray([position[k] for k in keys.tolist()], dtype=np.int64)
        else:
            order = np.arange(len(keys))
        preds[key] = np.asarray(s["p"], dtype=np.float64)[order]
        preds_cal[key] = np.asarray(s["p_cal"], dtype=np.float64)[order]
    out = {"descriptive": {"rows": int(len(y)), "matches": int(len(np.unique(match))), "exact_zero": int(exact_zero.sum()),
                           "abs_svi_lt_0.01_share": float((np.abs(svi) < 0.01).mean())},
           "q_h90": q_block(y, preds, preds_cal, match, cons)}
    if dataset == "kr_15_18_22":
        patch = np.asarray(rows["patch"]).astype(str)
        out["q_h90_by_patch"] = {p: q_block(y[patch == p], {k: v[patch == p] for k, v in preds.items()},
                                            {k: v[patch == p] for k, v in preds_cal.items()}, match[patch == p], cons) for p in PATCHES_2025}
    # V at tau (calibrated V at the engagement pre state; y = match winner)
    public = npz(DATA / "labels" / dataset / "LABELS_PUBLIC_h90.npz")
    eng = npz(DATA / "labels" / dataset / "ENGAGEMENTS.npz")
    pub_index = {k: i for i, k in enumerate(public["engagement_key"].tolist())}
    eng_keys = [f"{m.decode() if isinstance(m, bytes) else m}:{c.decode() if isinstance(c, bytes) else c}"
                for m, c in zip(eng["canonical_match_key"].tolist(), eng["component_key"].tolist())]
    eng_index = {k.encode(): i for i, k in enumerate(eng_keys)}
    p_pre = np.asarray(public["p_pre"], dtype=np.float64)[[pub_index[k] for k in keys.tolist()]]
    y_win = np.asarray(eng["target_Blue_win"], dtype=np.float64)[[eng_index[k if isinstance(k, bytes) else k.encode()] for k in keys.tolist()]]
    if not np.allclose(p_pre, np.asarray(rows["pt_p_pre"], dtype=np.float64), atol=0.0, rtol=0.0):
        raise RuntimeError(f"{dataset}: public p_pre differs from the q rows' pt_p_pre")
    out["v_tau"] = v_tau_block(y_win, p_pre, match)
    if dataset == "kr_15_18_22":
        out["v_tau_by_patch"] = {p: v_tau_block(y_win[patch == p], p_pre[patch == p], match[patch == p]) for p in PATCHES_2025}
    out["v_nominal"] = v_nominal_block(dataset, addendum)
    return out


def main() -> int:
    if OUT.exists():
        raise FileExistsError("the external evaluation already ran")
    pre = preconditions()
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    from reports.q_production_20261005 import followup_tier1 as ft

    best = json.loads(ft.RULE.read_text(encoding="utf-8"))["selected"]["key"]
    cons = {name: ft.contrasts(best)[name] for name in CONTRAST_NAMES}
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    opened = {"utc": datetime.now(timezone.utc).isoformat(), "by": "e3_evaluate.py (author-approved protocol step 5)",
              "sealed_sha256": {}}
    for dataset in DATASETS:
        path = DATA / "labels" / dataset / f"EXTERNAL_LABELS_SEALED_{dataset}.npz"
        digest = sha_file(path)
        if digest != lock["datasets"][dataset]["sealed"]["sha256"]:
            raise SystemExit(f"{dataset}: sealed labels differ from the lock")
        opened["sealed_sha256"][dataset] = digest
    history = json.loads(OPENED.read_text(encoding="utf-8")) if OPENED.exists() else {"openings": []}
    history["openings"].append(opened)        # a crashed pass is kept on record; the report is written once
    OPENED.write_text(json.dumps(history, indent=1), encoding="utf-8")
    report = {"kind": "EXTERNAL_REPORT", "utc": datetime.now(timezone.utc).isoformat(), "preconditions": pre,
              "protocol_sha256": sha_file(REPORTS / "EXTERNAL_PROTOCOL_20261009.json"), "contrasts": cons, "levels": LEVELS,
              "datasets": {ds: evaluate_dataset(ds, cons, addendum) for ds in DATASETS}}
    test = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    report["test_15_16_reference"] = {"P1": {k: test["bootstrap"]["contrasts"]["P1"][k] for k in ("point", "interval", "level")},
                                      "P2": {k: test["bootstrap"]["contrasts"]["P2"][k] for k in ("point", "interval", "level")}}
    v_test = json.loads(V_TEST.read_text(encoding="utf-8"))["metrics"]["overall"]
    v_tau = json.loads(V_TAU.read_text(encoding="utf-8"))["roles"]["TEST"]["p_pre"]
    report["v_15_16_reference"] = {"nominal_raw_auc": v_test["raw"]["match_weighted"]["auc"],
                                   "nominal_calibrated_auc": v_test["calibrated_optimistic"]["match_weighted"]["auc"],
                                   "tau_auc": v_tau["auc"], "files_sha256": {"V_TEST_REPORT_final.json": sha_file(V_TEST), "V_TAU_REPORT_final.json": sha_file(V_TAU)}}
    report["code_sha256"] = sha_file(Path(__file__))
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for ds, r in report["datasets"].items():
        q = r["q_h90"]
        print(f"== {ds}: rows {q['rows']} matches {q['matches']} pos {q['positive_rate_match_weighted']:.4f}")
        print("   AUC", {k.split('__')[2][:22]: round(v["raw"]["auc"], 4) for k, v in q["metrics"].items() if k in (ft.P1, ft.PT, best)})
        print("   ", {n: (round(c["point"], 5), [round(x, 5) for x in c["interval"]]) for n, c in q["contrasts"].items()})
        print("   V tau AUC", round(r["v_tau"]["auc"], 4), "V nominal AUC raw", round(r["v_nominal"]["all"]["raw"]["auc"], 4),
              "cal", round(r["v_nominal"]["all"]["calibrated"]["auc"], 4), "Brier cal", round(r["v_nominal"]["all"]["calibrated"]["brier"], 4))
        for p, b in r.get("q_h90_by_patch", {}).items():
            print(f"   {p}: P1 {b['contrasts']['P1']['point']:+.5f} {[round(x, 5) for x in b['contrasts']['P1']['interval']]} "
                  f"P2C1 {b['contrasts']['P2_C1']['point']:+.5f} AUC {b['metrics'][ft.P1]['raw']['auc']:.4f} PT {b['metrics'][ft.PT]['raw']['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
