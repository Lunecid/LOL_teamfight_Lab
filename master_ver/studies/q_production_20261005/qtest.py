"""q TEST stage: SCORE_LOCK, one TEST scoring pass, then contrasts (ARCHITECTURE_KO.md 5 "TEST", 6.3).

Three subcommands, run in this order and each only once:

  lock      SCORE_LOCK.json binds the protocol and matrix files, the release, the job plan, every row
            table and sealed TEST label file, the label lock, the three selections, and for every final
            cell its result, model file, stored Q_DEV predictions and calibrator, plus the exposure record.
            It refuses while any final or pure-set job is still pending or running.
  score     Re-loads every final model exactly as its worker built it and writes TEST predictions (raw and
            calibrated) to TEST_SCORES/.  No label is read: the row table carries y = -1 on TEST rows and
            the sealed file is not opened.  Before scoring TEST, each model re-scores a fixed slice of its
            own Q_DEV rows and must match the stored predictions (the loader is the trained model).
  evaluate  Opens the sealed TEST labels once, aligns them by engagement key, and writes TEST_REPORT.json:
            per-cell metrics (raw and calibrated), P1 and P2 (paired match bootstrap, 2000 draws, seed
            20261011, 97.5 % intervals), the 424 exploratory axis pairs and the graph-algorithm references
            (95 %), strata for the P1 cells, and an independent recomputation of every cell's Brier, logloss
            and AUC (atol 1e-10).

Nothing here changes a selection, card or calibrator; results are reported as they come out.
CLI: python -B -m studies.q_production_20261005.qtest {lock,score,evaluate} --run D:/LOL_Training_20261004/q_run_1
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Iterable, Mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from studies.q_production_20261005 import jobs as qjobs, qeval  # noqa: E402
from studies.q_production_20261005.rows import VARIANT_HORIZON  # noqa: E402

VERSION = "q_test_v1_20261008"
TEST_ROLE, Q_DEV_ROLE = 5, 3
LOCK_NAME, SCORES_DIR, REPORT_NAME = "SCORE_LOCK.json", "TEST_SCORES", "TEST_REPORT.json"
RELOAD_ROWS = 4096           # Q_DEV rows re-scored per model before its TEST pass
RELOAD_ATOL = {"neural": 1e-7, "table": 1e-9, "reference": 1e-9}  # neural: same bound as the worker reload check
RECOMPUTE_ATOL = 1e-10
TABLE_BATCH = 8192           # neural cells use their training batch size (qtrain.neural_rules), as their worker scored Q_DEV
REFERENCES = ("constant", "time_only", "p_only", "PT_linear", "PT_flex")
READS_P = ("p_only", "PT_linear", "PT_flex")
GRAPH_FAMILY = {"N_G0": "team_set_pair_readout", "N_G1": "relational_mpnn"}
PROTOCOL_FILES = ("reports/feature_control_study_20261001/FULL_STUDY_PROTOCOL.md",
                  "reports/feature_control_study_20261001/EXECUTION_CARDS.json",
                  "reports/feature_control_study_20261001/CARTESIAN_COMPARISON_MATRIX.json",
                  "reports/architecture_20261005/ARCHITECTURE_KO.md",
                  "reports/architecture_20261005/DECISIONS_20261005.json")
EXPOSURES = (
    "TEST (patch 15.16) was analysed in the superseded lineage before this study; scoring it here is a "
    "fixed-procedure re-analysis, not an unseen test (FULL_STUDY_PROTOCOL.md 126).",
    "rows.py printed TEST row counts and positive rates (base rate only) in Q_ROWS_h*.json on 2026-10-07 "
    "06:17 KST; no model output on TEST existed (DECISION_LOG_KO.md).",
)


# ----------------------------------------------------------------------------- small helpers
def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending, path)


def key_of(job_id: str) -> str:
    return job_id.replace("/", "__").replace(":", "~")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# ----------------------------------------------------------------------------- cells
def planned_cells(run: Path) -> list[dict[str, Any]]:
    """Final jobs plus the selected pure-set card per S (the cells TEST is scored for)."""

    plan = read_json(run / "jobs" / "JOBS.json")
    pure = read_json(run / "Q_PURE_SET_SELECTION.json") or {}
    chosen = {v.get("job") for k, v in pure.items() if k.startswith("S")}
    return [*plan["final"], *[j for j in plan["pure_set"] if j["job_id"] in chosen]]


def cell_record(run: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    """Where a cell's trained model lives (own directory or the reused source) and its state."""

    key = key_of(job["job_id"])
    directory = run / "jobs" / key
    reuse = read_json(directory / "REUSE.json")
    source = run / "jobs" / reuse["reuse_of_key"] if reuse else directory
    result = read_json(source / "RESULT.json") or {}
    record = {"key": key, "job_id": job["job_id"], "block": job["block"], "family": job["family"],
              "condition_id": job["condition_id"], "variant": job["variant"], "source_dir": str(source),
              "reused": bool(reuse), "status": result.get("status", "missing"), "reason": result.get("reason")}
    if record["status"] == "completed":
        model = next((source / name for name in ("model_refit.pt", "model_refit.pkl", "model_refit.json")
                      if (source / name).is_file()), None)
        calibrator = run / "calibrators" / f"{key}.json"
        record.update(result_sha256=sha256_file(source / "RESULT.json"), spec_sha256=sha256_file(source / "SPEC.json"),
                      model_file=model.name if model else None, model_sha256=sha256_file(model) if model else None,
                      pred_q_dev_sha256=sha256_file(source / "pred_q_dev.npz"),
                      calibrator_sha256=sha256_file(calibrator) if calibrator.is_file() else None)
    return record


def coverage_open(run: Path) -> dict[str, Any]:
    cov = read_json(run / "COVERAGE.json") or {}
    open_blocks = {block: {k: v for k, v in states.items() if k in ("pending", "running")}
                   for block, states in (cov.get("by_block") or {}).items()
                   if not block.startswith("search/") and any(k in ("pending", "running") for k in states)}
    return open_blocks


# ----------------------------------------------------------------------------- lock
def lock(run: Path) -> dict[str, Any]:
    run = Path(run)
    target = run / LOCK_NAME
    if target.exists():
        raise FileExistsError("SCORE_LOCK.json exists; TEST is scored once")
    still_open = coverage_open(run)
    if still_open:
        raise RuntimeError(f"final or pure-set jobs are still open: {still_open}")
    selection = read_json(run / "Q_SELECTION.json")
    if not selection or not (selection.get("selection") or {}).get("complete"):
        raise RuntimeError("Q_SELECTION.json is missing or incomplete")
    if not (run / "Q_PURE_SET_SELECTION.json").is_file():
        raise RuntimeError("Q_PURE_SET_SELECTION.json is missing")
    release = read_json(run / "RUN_RELEASE.json")
    files: dict[str, str] = {}
    for rel in PROTOCOL_FILES:
        files[rel] = sha256_file(ROOT / rel)
    for name in ("RUN_RELEASE.json", "jobs/JOBS.json", "Q_SELECTION.json", "Q_SEARCH_SELECTION.json", "Q_PURE_SET_SELECTION.json",
                 *[p.name for p in sorted(run.glob("Q_ROWS_h*.npz"))], *[p.name for p in sorted(run.glob("Q_TEST_SEALED_h*.npz"))]):
        files[name] = sha256_file(run / name)
    for path in sorted((run / "inputs").glob("*/Q_ROWS.npz")):
        files[str(path.relative_to(run)).replace("\\", "/")] = sha256_file(path)
    files["release_json"] = sha256_file(Path(release["release"]))
    label_lock = Path(r"D:\LOL_Training_20261004\labels_1\LABEL_LOCK.json")
    files["LABEL_LOCK.json"] = sha256_file(label_lock)
    files["code:qtest.py"] = sha256_file(Path(__file__))
    files["code:qeval.py"] = sha256_file(Path(qeval.__file__))
    cells = [cell_record(run, job) for job in planned_cells(run)]
    payload = {"version": VERSION, "locked_utc": utc(), "release_digest": release["digest"], "files": files,
               "cells": cells, "cells_completed": sum(c["status"] == "completed" for c in cells),
               "selected_family": selection["selection"]["selected"], "exposures": list(EXPOSURES),
               "rules": {"bootstrap_resamples": qeval.BOOTSTRAP_RESAMPLES, "bootstrap_seed": qeval.BOOTSTRAP_SEED,
                         "primary_level": qeval.PRIMARY_LEVEL, "exploratory_level": qeval.EXPLORATORY_LEVEL,
                         "reload_rows": RELOAD_ROWS, "reload_atol": RELOAD_ATOL, "recompute_atol": RECOMPUTE_ATOL}}
    write_once(target, payload)
    return payload


def verify_lock(run: Path) -> dict[str, Any]:
    payload = read_json(run / LOCK_NAME)
    if payload is None:
        raise RuntimeError("no SCORE_LOCK.json: run `lock` first")
    release = read_json(run / "RUN_RELEASE.json")
    for name, expected in payload["files"].items():
        if name.startswith("code:"):
            path = Path(__file__) if name == "code:qtest.py" else Path(qeval.__file__)
        elif name == "release_json":
            path = Path(release["release"])
        elif name == "LABEL_LOCK.json":
            path = Path(r"D:\LOL_Training_20261004\labels_1\LABEL_LOCK.json")
        elif name in PROTOCOL_FILES:
            path = ROOT / name
        else:
            path = run / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"{name} changed after SCORE_LOCK")
    for cell in payload["cells"]:
        if cell["status"] != "completed":
            continue
        source = Path(cell["source_dir"])
        if sha256_file(source / cell["model_file"]) != cell["model_sha256"] or sha256_file(source / "RESULT.json") != cell["result_sha256"]:
            raise RuntimeError(f"{cell['key']}: model or result changed after SCORE_LOCK")
    return payload


# ----------------------------------------------------------------------------- model loading
class LoadedCell:
    """A final cell's trained model, re-built the way its worker built it, scoring arbitrary row indices."""

    def __init__(self, cell: Mapping[str, Any], device: str = "cpu") -> None:
        from studies.q_production_20261005 import features as qf, qbatch, q_worker

        self.cell = dict(cell)
        source = Path(cell["source_dir"])
        self.spec = read_json(source / "SPEC.json")
        job, inputs = self.spec["job"], self.spec["inputs"]
        self.family, self.card = job["family"], dict(job["card"])
        self.rows = load_npz(Path(inputs["rows"]))
        if sha256_file(Path(inputs["rows"])) != inputs["rows_sha256"]:
            raise RuntimeError(f"{cell['key']}: row table changed since training")
        self.kind = "reference" if self.family in REFERENCES else ("table" if self.family in ("lightgbm", "logistic") else "neural")
        self.device = device
        if self.kind == "reference":
            if self.family == "constant":
                rate = float(read_json(source / "model_refit.json")["rate"])
                self._constant = float(np.log(rate / (1.0 - rate)))
            else:
                with (source / "model_refit.pkl").open("rb") as stream:
                    loaded = pickle.load(stream)
                self._model, self._design = loaded["model"], loaded["design"]
            return
        store = qf.RawStore(Path(inputs["store"]))
        xy = np.load(Path(inputs["xy"]), allow_pickle=False)
        self._cond = qbatch.Condition.from_job(SimpleNamespace(factors=job["factors"], inputs=job["inputs"]))
        self._ctx = qbatch.ScopeContext(store, self.rows, xy, "REFIT_TRAIN", saved=Path(inputs["contexts"]["REFIT_TRAIN"]))
        self._assemble = qbatch.assemble
        if self.kind == "table":
            with (source / "model_refit.pkl").open("rb") as stream:
                self._model = pickle.load(stream)
            return
        import torch

        from studies.q_production_20261005 import qtrain

        train = np.flatnonzero(np.asarray(self.rows["role"]) == 0)
        shape = q_worker.structure(self.family, self.card, self._ctx)
        build = q_worker.builder(self.family, self.card, self._ctx, self._cond, shape, train)
        self._model = qtrain._new_model(build, torch.device(device))
        state = torch.load(source / "model_refit.pt", map_location=torch.device(device), weights_only=False)["model"]
        self._model.load_state_dict(state)
        self._model.eval()
        self._forward = qbatch.forward_for(self.family)
        self._batch = qtrain.neural_rules(self.family, self.card.get("weight_decay")).batch_size

    def logits(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if self.kind == "reference":
            if self.family == "constant":
                return np.full(len(rows), self._constant, dtype=np.float64)
            t = np.asarray(self.rows["elapsed_min"], dtype=np.float64)[rows]
            p = np.asarray(self.rows["pt_p_pre"], dtype=np.float64)[rows] if self.family in READS_P else np.full(len(rows), 0.5)
            return np.asarray(self._model.predict_logits(self._design(p, t)), dtype=np.float64)
        if self.kind == "table":
            from scipy import sparse

            parts = [self._assemble(self._ctx, rows[i:i + TABLE_BATCH], self._cond, self.family) for i in range(0, len(rows), TABLE_BATCH)]
            return np.asarray(self._model.predict_logits(sparse.vstack(parts, format="csr")), dtype=np.float64)
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(rows), self._batch):
                inputs = self._assemble(self._ctx, rows[i:i + self._batch], self._cond, self.family)
                out.append(self._forward(self._model, inputs, self.device).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ----------------------------------------------------------------------------- score
def score(run: Path, device: str = "cuda:0", only: Iterable[str] | None = None) -> dict[str, Any]:
    from studies.label_production_20261005.assemble import write_npz
    from studies.q_production_20261005 import qcalibrate

    run = Path(run)
    payload = verify_lock(run)
    out_dir = run / SCORES_DIR
    out_dir.mkdir(exist_ok=True)
    wanted = set(only) if only else None
    done = {}
    for cell in payload["cells"]:
        if cell["status"] != "completed" or (wanted and cell["key"] not in wanted):
            continue
        target = out_dir / f"{cell['key']}.npz"
        meta_path = out_dir / f"{cell['key']}.json"
        if meta_path.is_file():
            done[cell["key"]] = read_json(meta_path)
            continue
        loaded = LoadedCell(cell, device=device)
        stored = load_npz(Path(cell["source_dir"]) / "pred_q_dev.npz")
        check_rows = np.asarray(stored["row"], dtype=np.int64)[:RELOAD_ROWS]
        again = loaded.logits(check_rows)
        reload_diff = float(np.abs(again - np.asarray(stored["logit"])[:RELOAD_ROWS]).max()) if len(check_rows) else 0.0
        if reload_diff > RELOAD_ATOL[loaded.kind]:
            raise RuntimeError(f"{cell['key']}: re-loaded model differs from its stored Q_DEV predictions by {reload_diff}")
        rows = np.flatnonzero(np.asarray(loaded.rows["role"]) == TEST_ROLE)
        if (np.asarray(loaded.rows["y"])[rows] != -1).any():
            raise RuntimeError(f"{cell['key']}: TEST labels are not sealed in the row table")
        logits = loaded.logits(rows)
        p = sigmoid(logits)
        calibrator_path = run / "calibrators" / f"{cell['key']}.json"
        p_cal, calibration = None, "missing"
        if calibrator_path.is_file():
            calibrator = qcalibrate.QCalibrator.from_dict(read_json(calibrator_path)["calibrator"])
            calibration = calibrator.status
            if calibrator.fitted:
                p_cal = calibrator.predict(p)
        arrays = {"row": rows, "engagement_key": np.asarray(loaded.rows["engagement_key"])[rows], "logit": logits, "p": p}
        if p_cal is not None:
            arrays["p_cal"] = p_cal
        sha = write_npz(target, arrays)
        meta = {"key": cell["key"], "rows": int(len(rows)), "pred_sha256": sha, "reload_max_abs_logit_diff": reload_diff,
                "reload_rows": int(len(check_rows)), "calibration": calibration, "kind": loaded.kind, "utc": utc(),
                "rows_table": loaded.spec["inputs"]["rows"]}
        write_once(meta_path, meta)
        done[cell["key"]] = meta
    return done


# ----------------------------------------------------------------------------- evaluate
def _independent(y: np.ndarray, p: np.ndarray, match: np.ndarray) -> dict[str, float]:
    """Match-weighted Brier / logloss / AUC by a separate route (per-match sums; rank-sum AUC)."""

    _, inv, counts = np.unique(match, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv] / counts.size
    q = np.clip(p, 1e-15, 1 - 1e-15)
    brier = float(np.sum(w * (p - y) ** 2))
    logloss = float(-np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))))
    order = np.argsort(p, kind="mergesort")
    ps, ys, ws = p[order], y[order], w[order]
    _, first = np.unique(ps, return_index=True)
    bounds = np.append(first, len(ps))
    below, num = 0.0, 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        pos = ws[a:b][ys[a:b] == 1].sum(); neg = ws[a:b][ys[a:b] == 0].sum()
        num += pos * (below + 0.5 * neg)
        below += neg
    auc = float(num / (ws[ys == 1].sum() * ws[ys == 0].sum()))
    return {"brier": brier, "logloss": logloss, "auc": auc}


def _family_of(condition: str) -> str | None:
    if condition.startswith("LGBM_"):
        return "lightgbm"
    if condition.startswith("SAGE_"):
        return "graphsage"
    return GRAPH_FAMILY.get(condition[:4])


def _cell_entry(cell: Mapping[str, Any], y: np.ndarray, p: np.ndarray, p_cal: np.ndarray | None, rows: np.ndarray,
                rows_table: str, calibration: str) -> dict[str, Any]:
    """Metrics of one cell on one split, the independent check, and its per-match losses."""

    table = load_npz(Path(rows_table))
    match = np.asarray(table["match_index"])[rows]
    if not np.isin(y, (0.0, 1.0)).all():
        raise RuntimeError(f"{cell['key']}: labels are not binary")
    raw = qeval.evaluate(y, p, match)
    independent = _independent(y, p, match)
    for name, value in independent.items():
        reference = raw["match_weighted"][name]
        if reference is None or abs(float(reference) - value) > RECOMPUTE_ATOL:
            raise RuntimeError(f"{cell['key']}: independent {name} {value} differs from qeval {reference}")
    record = {"raw": raw, "independent": independent, "rows": int(len(y)), "calibration": calibration}
    if p_cal is not None:
        record["calibrated"] = qeval.evaluate(y, p_cal, match)
    ids, losses = qeval.per_match_loss(y, p, match)
    return {**dict(cell), "metrics": record, "_ids": ids, "_losses": losses, "_y": y, "_p": p, "_match": match,
            "_rows": rows, "_rows_table": rows_table}


def analyze(cells: Mapping[str, Mapping[str, Any]], selected: str) -> dict[str, Any]:
    """P1, P2, exploratory axis pairs and graph references (shared paired match bootstrap) and P1-cell strata."""

    by_condition = {(c["block"], c["condition_id"], c["family"], c["variant"]): key for key, c in cells.items()}

    def main_cell(condition: str, family: str, block: str = "module") -> str | None:
        return by_condition.get((block, condition, family, "main_h90"))

    p1_cell = main_cell(qjobs.FULL_CONDITIONS[selected], selected)
    pt_flex = next((k for k, c in cells.items() if c["block"] == "reference" and c["family"] == "PT_flex" and c["variant"] == "main_h90"), None)
    quads = {d: {name: main_cell(cond, GRAPH_FAMILY[cond[:4]]) for name, cond in qeval.P2_QUADS[d].items()} for d in (0, 1)}
    contrasts: dict[str, dict[str, float]] = {}
    levels: dict[str, float] = {}
    missing: list[str] = []
    if p1_cell and pt_flex:
        contrasts["P1"] = qeval.p1_contrast(p1_cell, pt_flex)
        levels["P1"] = qeval.PRIMARY_LEVEL
    else:
        missing.append("P1")
    if all(all(v for v in quads[d].values()) for d in (0, 1)):
        contrasts["P2"] = qeval.p2_contrast(quads)
        levels["P2"] = qeval.PRIMARY_LEVEL
    else:
        missing.append("P2")
    matrix = json.loads((ROOT / PROTOCOL_FILES[2]).read_text(encoding="utf-8"))
    for axis, pairs in matrix["paired_axis_comparisons"].items():
        for pair in pairs:
            off = main_cell(pair["off"], _family_of(pair["off"]) or "")
            on = main_cell(pair["on"], _family_of(pair["on"]) or "")
            name = f"axis:{axis}:{pair['on']}-{pair['off']}"
            if off and on:
                contrasts[name] = qeval.difference(on, off)
                levels[name] = qeval.EXPLORATORY_LEVEL
            else:
                missing.append(name)
    for pair in matrix["graph_algorithm_reference_comparisons"]:
        sage, mpnn = main_cell(pair["graphsage"], "graphsage"), main_cell(pair["relational_mpnn"], "relational_mpnn")
        name = f"graph:{pair['graphsage']}-{pair['relational_mpnn']}"
        if sage and mpnn:
            contrasts[name] = qeval.difference(sage, mpnn)
            levels[name] = qeval.EXPLORATORY_LEVEL
        else:
            missing.append(name)
    used = sorted({cell for contrast in contrasts.values() for cell in contrast})
    bootstrap = None
    if used:
        shared, aligned = qeval.align_cells({k: (cells[k]["_ids"], cells[k]["_losses"]) for k in used})
        bootstrap = qeval.paired_match_bootstrap(shared, aligned, contrasts, levels=levels)
    strata = {}
    for name in (p1_cell, pt_flex):
        if not name:
            continue
        c = cells[name]
        table = load_npz(Path(c["_rows_table"]))
        rows = c["_rows"]
        truncated = (np.asarray(table["truncation_bits"])[rows] & 1) == 0   # bit 0 = horizon reached (labels TRUNCATION order)
        layers = {"game_time": qeval.game_time_strata(np.asarray(table["elapsed_min"])[rows]),
                  "balanced_p": qeval.balanced_p_strata(np.asarray(table["pt_p_pre"])[rows]),
                  "frame_age": qeval.frame_age_strata(np.asarray(table["frame_age_s"])[rows]),
                  "endpoint": qeval.endpoint_strata(truncated),
                  "kill_count": qeval.kill_count_strata(np.asarray(table["eval_n_kills"])[rows]),
                  "credit": qeval.credit_strata(np.asarray(table["eval_firstkill_credit_blue"])[rows],
                                                np.asarray(table["eval_firstkill_credit_red"])[rows])}
        strata[name] = {layer: qeval.metrics_by_stratum(c["_y"], c["_p"], None, labels, match_ids=c["_match"])
                        for layer, labels in layers.items()}
    return {"selected_family": selected, "p1_cell": p1_cell, "pt_flex_cell": pt_flex, "p2_quads": quads, "missing_contrasts": missing,
            "cells": {k: {kk: vv for kk, vv in c.items() if not kk.startswith("_")} for k, c in cells.items()},
            "bootstrap": bootstrap, "strata": strata,
            "notes": ["1-seed: intervals carry match resampling only, not training-seed variation (OC 9).",
                      "Exploratory contrasts are descriptive; no selection or card is changed by them."]}


def evaluate(run: Path) -> dict[str, Any]:
    """Open the sealed TEST labels once and analyze the TEST scores."""

    run = Path(run)
    if (run / REPORT_NAME).exists():
        raise FileExistsError("TEST_REPORT.json exists; TEST is evaluated once")
    payload = verify_lock(run)
    sealed: dict[int, tuple[dict[bytes, int], dict[str, np.ndarray]]] = {}
    cells: dict[str, dict[str, Any]] = {}
    for cell in payload["cells"]:
        if cell["status"] != "completed":
            continue
        meta = read_json(run / SCORES_DIR / f"{cell['key']}.json")
        if meta is None:
            raise RuntimeError(f"{cell['key']} has no TEST scores: run `score` first")
        if sha256_file(run / SCORES_DIR / f"{cell['key']}.npz") != meta["pred_sha256"]:
            raise RuntimeError(f"{cell['key']}: TEST scores changed")
        scores = load_npz(run / SCORES_DIR / f"{cell['key']}.npz")
        horizon = VARIANT_HORIZON[cell["variant"]]
        if horizon not in sealed:
            table = load_npz(run / f"Q_TEST_SEALED_h{horizon}.npz")
            sealed[horizon] = ({k: i for i, k in enumerate(table["engagement_key"].tolist())}, table)
        index, table = sealed[horizon]
        pos = np.asarray([index[k] for k in scores["engagement_key"].tolist()], dtype=np.int64)
        y = np.asarray(table["y"], dtype=np.float64)[pos]
        p_cal = np.asarray(scores["p_cal"], dtype=np.float64) if "p_cal" in scores else None
        cells[cell["key"]] = _cell_entry(cell, y, np.asarray(scores["p"], dtype=np.float64), p_cal, scores["row"],
                                         meta["rows_table"], meta["calibration"])
    report = {"version": VERSION, "utc": utc(), "split": "TEST", "lock_sha256": sha256_file(run / LOCK_NAME),
              "labels_opened": sorted(str(h) for h in sealed), **analyze(cells, payload["selected_family"])}
    write_once(run / REPORT_NAME, json.loads(json.dumps(report, default=_json_default)))
    return report


def rehearse(run: Path, out: Path) -> dict[str, Any]:
    """Run the whole analysis on Q_DEV (stored predictions, unsealed Q_DEV labels) to check the code before TEST.

    Needs no lock and reads no TEST data.  The output is a code check and a development-split summary,
    not a TEST result."""

    from studies.q_production_20261005 import qcalibrate

    run = Path(run)
    selection = read_json(run / "Q_SELECTION.json")
    cells: dict[str, dict[str, Any]] = {}
    for job in planned_cells(run):
        cell = cell_record(run, job)
        if cell["status"] != "completed":
            continue
        source = Path(cell["source_dir"])
        spec = read_json(source / "SPEC.json")
        rows_table = spec["inputs"]["rows"]
        table = load_npz(Path(rows_table))
        pred = load_npz(source / "pred_q_dev.npz")
        rows = np.asarray(pred["row"], dtype=np.int64)
        if not (np.asarray(table["role"])[rows] == Q_DEV_ROLE).all():
            raise RuntimeError(f"{cell['key']}: stored Q_DEV predictions are not Q_DEV rows")
        p = np.asarray(pred["p"], dtype=np.float64)
        calibrator_path = run / "calibrators" / f"{cell['key']}.json"
        p_cal, calibration = None, "missing"
        if calibrator_path.is_file():
            calibrator = qcalibrate.QCalibrator.from_dict(read_json(calibrator_path)["calibrator"])
            calibration = calibrator.status
            p_cal = calibrator.predict(p) if calibrator.fitted else None
        cells[cell["key"]] = _cell_entry(cell, np.asarray(table["y"], dtype=np.float64)[rows], p, p_cal, rows, rows_table, calibration)
    report = {"version": VERSION, "utc": utc(), "split": "Q_DEV (rehearsal; not a TEST result)",
              **analyze(cells, selection["selection"]["selected"])}
    Path(out).write_text(json.dumps(json.loads(json.dumps(report, default=_json_default)), indent=1), encoding="utf-8")
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("lock", "score", "evaluate", "rehearse"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "lock":
        out = lock(args.run)
        print(json.dumps({"cells": len(out["cells"]), "completed": out["cells_completed"]}))
    elif args.command == "score":
        out = score(args.run, device=args.device, only=args.only)
        print(json.dumps({"scored": len(out)}))
    elif args.command == "rehearse":
        out = rehearse(args.run, args.out)
        print(json.dumps({"contrasts": len((out["bootstrap"] or {}).get("contrasts", {})), "missing": len(out["missing_contrasts"])}))
    else:
        out = evaluate(args.run)
        print(json.dumps({"contrasts": len((out["bootstrap"] or {}).get("contrasts", {})), "missing": len(out["missing_contrasts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
