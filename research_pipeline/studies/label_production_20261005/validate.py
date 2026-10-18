"""L4 of label production: computation-correctness checks (they block the label lock).

Checks (reports/research_execution_20261003/PROTOCOL_KO.md §2 first layer and
post_state_label contract §5 items 2-4):
  1. coverage and key uniqueness of every q-role engagement in every horizon file;
  2. same pipeline at both ends (context per row from role/fold; one V identity per context);
  3. finite probabilities in [0, 1]; SVI, direction, exact zero and near-zero flags recomputed
     independently from the prediction tables; calibrated V recomputed from the raw probability
     with the V_LOCK slope/intercept by a separate implementation;
  4. whole-game exclusion proven from the V caches themselves: no labelled TRAIN game of fold k
     appears in a fit/stop/refit/dev/cal row of oof_k's input caches;
  5. reload and batch-size invariance on a random sample: a freshly loaded V scores the sample
     with batch sizes 1 and 257; the largest differences and any label flips are reported;
  6. hashes: label files, prediction files, V_LOCK and selected checkpoints are unchanged.
Nothing here tunes or relabels; failures are reported and block the lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import assemble, queries, score  # noqa: E402

VERSION = "label_validation_layer1_v1_20261005"
SAMPLE_FRACTION = 0.02
SAMPLE_SEED = 20261005
SPLIT_BITS = 1 | 2 | 4 | 8 | 16


def _independent_v(raw: np.ndarray, a: float, b: float) -> np.ndarray:
    out = np.empty(len(raw), dtype=np.float64)
    for i, p in enumerate(np.asarray(raw, dtype=np.float64)):
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        z = a * (math.log(p) - math.log1p(-p)) + b
        out[i] = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return out


def check_rows(labels: Path) -> dict[str, Any]:
    manifest = json.loads((labels / "LABELS_MANIFEST.json").read_text(encoding="utf-8"))
    eng = dict(np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False))
    preds = {name: assemble.load_predictions(labels, name) for name in assemble.CONTEXTS}
    results: dict[str, Any] = {}
    for horizon in assemble.HORIZONS:
        name = f"LABELS_h{horizon}.npz"
        path = labels / name
        problems: list[str] = []
        if score.sha256_file(path) != manifest["files"][name]:
            problems.append("label file differs from its manifest")
        table = dict(np.load(path, allow_pickle=False))
        if not (np.array_equal(table["canonical_match_key"], eng["canonical_match_key"]) and np.array_equal(table["component_key"], eng["component_key"])):
            problems.append("label rows do not follow the engagement table one to one")
        keys = np.char.add(np.char.add(table["canonical_match_key"], b":"), table["component_key"])
        if len(np.unique(keys)) != len(keys):
            problems.append("duplicate engagement keys")
        role, fold, context = table["role"].astype(int), table["fold"].astype(int), table["context"].astype(int)
        expected_context = np.where(np.isin(role, assemble.Q_FINAL_ROLES), 0, np.where(role == 0, 1 + fold, -1))
        if not np.array_equal(context, expected_context):
            problems.append("a row is scored by a context other than its role/fold requires")
        q_rows = context >= 0
        valid = table["valid"].astype(bool)
        if (q_rows & ~valid).any():
            problems.append(f"{int((q_rows & ~valid).sum())} q rows are invalid")
        # independent recomputation from the prediction tables (dict lookup, not searchsorted)
        mismatch = 0
        for code, cname in enumerate(assemble.CONTEXTS):
            pred, _identity = preds[cname]
            index = {int(q): i for i, q in enumerate(pred["query_index"])}
            rows = np.flatnonzero(valid & (context == code))
            pre = np.asarray([pred["v"][index[int(q)]] for q in table["query_pre"][rows]])
            post = np.asarray([pred["v"][index[int(q)]] for q in table["query_post"][rows]])
            svi = post - pre
            mismatch += int((svi != table["svi_delta"][rows]).sum())
            mismatch += int(((svi > 0).astype(np.int8) != table["direction_positive"][rows]).sum())
            mismatch += int(((svi == 0).astype(np.int8) != table["exact_zero"][rows]).sum())
            near = np.zeros(len(rows), dtype=np.uint8)
            for bit, threshold in enumerate(assemble.NEAR_ZERO):
                near[np.abs(svi) < threshold] |= 1 << bit
            mismatch += int((near != table["near_zero_bits"][rows]).sum())
        if mismatch:
            problems.append(f"{mismatch} recomputed SVI/direction/zero/near-zero values differ")
        p = np.stack([table["p_pre"][valid], table["p_post"][valid]])
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            problems.append("a valid row has a non-finite or out-of-range probability")
        results[f"h{horizon}"] = {"rows": int(len(keys)), "q_rows": int(q_rows.sum()), "valid": int(valid.sum()),
                                  "positive_rate_valid": float((table["direction_positive"][valid] == 1).mean()) if valid.any() else None,
                                  "exact_zero": int((table["exact_zero"] == 1).sum()), "problems": problems}
    # calibrator recomputed by a separate implementation on a sample of each context
    calibration: dict[str, Any] = {}
    rng = np.random.default_rng(SAMPLE_SEED)
    for cname, (pred, identity) in preds.items():
        pick = rng.choice(len(pred["raw_p"]), size=min(20_000, len(pred["raw_p"])), replace=False)
        a, b = float(identity["v"]["calibrator"]["a"]), float(identity["v"]["calibrator"]["b"])
        diff = np.abs(_independent_v(pred["raw_p"][pick], a, b) - pred["v"][pick])
        calibration[cname] = {"rows": int(len(pick)), "max_abs_diff": float(diff.max())}
    return {"rows": results, "calibrator_recomputed": calibration}


def check_exclusion(labels: Path, run_root: Path) -> dict[str, Any]:
    """No labelled TRAIN game of fold k is a fit/stop/refit/dev/cal row in oof_k's own input caches."""

    from studies.v_production_20261004.cache import InputCache

    eng = dict(np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False))
    out: dict[str, Any] = {}
    for k in range(5):
        context = f"oof{k}"
        labelled = set(eng["canonical_match_key"][(eng["role"] == 0) & (eng["fold"] == k)].tolist())
        plan = json.loads((Path(run_root) / context / "CONTEXT_PLAN.json").read_text(encoding="utf-8"))
        seen: set[bytes] = set()
        rows = 0
        for cache_name in (plan["fit_cache_name"], plan["refit_cache_name"]):
            cache = InputCache(Path(run_root) / "cache" / cache_name, verify=False)
            try:
                metadata = cache.metadata
                used = (np.asarray(metadata["context_membership"], dtype=np.uint8) & SPLIT_BITS) != 0
                seen |= set(np.asarray(metadata["game_key"])[used].tolist())
                rows += int(used.sum())
            finally:
                cache.close()
        overlap = labelled & seen
        out[context] = {"labelled_train_games": len(labelled), "v_rows_checked": rows, "overlap": len(overlap),
                        "examples": [value.decode() for value in list(overlap)[:3]]}
    return out


def check_reload(labels: Path, run_root: Path) -> dict[str, Any]:
    """Fresh model load + batch sizes 1 and 257 on a random sample of each context's queries."""

    out: dict[str, Any] = {}
    rng = np.random.default_rng(SAMPLE_SEED)
    for cname in assemble.CONTEXTS:
        pred, _identity = assemble.load_predictions(labels, cname)
        n = len(pred["query_index"])
        pick = np.sort(rng.choice(n, size=max(1, int(n * SAMPLE_FRACTION)), replace=False))
        sample = pred["query_index"][pick]
        v = score.LockedV(run_root, cname)
        surface, bundle = queries.open_inputs(cname)
        diffs: dict[str, float] = {}
        try:
            rows = [row for _, row, _ in surface.iter_query_rows(sample)]
            for size in (1, 257):
                parts = [v.score(queries.transform(rows[i:i + size], bundle)) for i in range(0, len(rows), size)]
                logit, raw, cal = (np.concatenate([part[j] for part in parts]) for j in range(3))
                diffs[f"batch{size}_max_abs_logit"] = float(np.abs(logit - pred["logit"][pick]).max())
                diffs[f"batch{size}_max_abs_v"] = float(np.abs(cal - pred["v"][pick]).max())
                diffs[f"batch{size}_rows_differing"] = int((cal != pred["v"][pick]).sum())
        finally:
            surface.close()
        out[cname] = {"sample_rows": int(len(sample)), **diffs}
    return out


def check_hashes(labels: Path, run_root: Path) -> dict[str, Any]:
    problems = []
    for cname in assemble.CONTEXTS:
        manifest = json.loads((labels / f"PRED_{cname}_MANIFEST.json").read_text(encoding="utf-8"))
        lock = Path(run_root) / cname / "V_LOCK.json"
        if score.sha256_file(lock) != manifest["v"]["v_lock_sha256"]:
            problems.append(f"{cname}: V_LOCK changed after scoring")
        if score.sha256_file(Path(manifest["v"]["checkpoint"])) != manifest["v"]["checkpoint_sha256"]:
            problems.append(f"{cname}: selected checkpoint changed after scoring")
        if score.sha256_file(labels / f"PRED_{cname}.npz") != manifest["pred_sha256"]:
            problems.append(f"{cname}: PRED file changed")
    return {"problems": problems}


def run(labels: Path, run_root: Path) -> dict[str, Any]:
    report = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "rows": None}
    report.update(check_rows(labels))
    report["exclusion"] = check_exclusion(labels, run_root)
    report["reload_and_batch"] = check_reload(labels, run_root)
    report["hashes"] = check_hashes(labels, run_root)
    blocking = [f"{h}: {p}" for h, value in report["rows"].items() for p in value["problems"]]
    blocking += [f"{c}: {v['overlap']} labelled games seen by the V" for c, v in report["exclusion"].items() if v["overlap"]]
    blocking += report["hashes"]["problems"]
    blocking += [f"{c}: calibrator recomputation differs by {v['max_abs_diff']}" for c, v in report["calibrator_recomputed"].items()
                 if v["max_abs_diff"] > 1e-12]
    report["blocking_problems"] = blocking
    report["passed"] = not blocking  # batch-size differences are reported, not hidden; see reload_and_batch
    pending = labels / f".VALIDATION_LAYER1.json.{os.getpid()}.pending"
    pending.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, labels / "VALIDATION_LAYER1.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    report = run(Path(args.labels), Path(args.run_root))
    print(json.dumps({"passed": report["passed"], "blocking_problems": report["blocking_problems"]}, indent=1))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
