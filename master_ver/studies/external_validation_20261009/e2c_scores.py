"""E2c: external q inputs and the protocol cells' raw and Q_CAL-calibrated predictions (no label is read).

README
  Rows   Q_ROWS_h90.npz per dataset from the public h90 label table: every valid EXTERNAL engagement, columns as
         studies/q_production_20261005/rows.build_rows writes them (role 6, fold -1, q_stop False, weight_raw 1/m,
         pt_p_pre = V_cal at tau, eval_* strata from ENGAGEMENTS.npz) with the label columns sealed as rows.seal_test
         does for TEST: y = -1, label_svi_delta NaN, exact_zero -1, near_zero_bits 0.  Positive counts are not
         computed.  An added column 'patch' (per match, from the E2a bank entries) serves the per-patch report.
  Store  raw input store of the rows' pre queries: features.build_chunk (frozen) with the dataset's
         ExternalQuerySurface, 20,000-row chunks, features.merge_chunks (frozen).
  XY     spatial.read_query_xy(query_pre, projection=<dataset projection>) (frozen); every row must have finite XY
         (q_prepare.query_xy rule; a row without would change the population).
  Context the frozen main_h90 REFIT_TRAIN context of q_run_1 (bundle, sigma, M medians, block statistics; never
         refitted).  ExternalScopeContext = qbatch.ScopeContext whose scope digest, which proves that a context was
         fitted on the TRAIN rows of its row table, is taken from the context's own CONTEXT.json after that file's
         sha256 is checked against q_run_1 PREPARED.json (an EXTERNAL row table has no TRAIN rows).  Unseen
         categories map to UNK through the frozen vocabularies (features._map_ints / vocab.mapping), numeric lanes use
         the frozen scaling.
  Cells  the protocol cells (EXTERNAL_PROTOCOL 'cells'): P1, PT_flex, P1-best (P1_BEST_RULE_20261008.json), P2 quads
         C=1 and C=0, LightGBM S{0,1} D{0,1} at A=M=0, E=C=R=1.  Each is re-loaded with qtest.LoadedCell (frozen) and
         must re-score its first 4,096 stored Q_DEV rows within qtest.RELOAD_ATOL, as qtest.score requires.  The
         external logits use the frozen LoadedCell.logits on a copy whose row table and context are the external
         ones; for neural cells the final partial batch is padded to the cell's batch size with copies of its last row
         (padded outputs dropped), the shape rule found in the E2b parity (GPU kernels are per-row identical only for
         equal batch shapes).  p = sigmoid(logit), p_cal = the cell's Q_CAL calibrator (qcalibrate.QCalibrator).
  Parity parity_15_16 logits versus q_run_1 TEST_SCORES for the same engagement keys: LightGBM and references <= 1e-7,
         neural <= qtest.RELOAD_ATOL['neural'].
Outputs: q_inputs/<dataset>/{Q_ROWS_h90.npz, Q_ROWS_h90.json, raw/store, XY.npy, XY.json},
         scores/<dataset>/<cell key>.npz (row, engagement_key, logit, p, p_cal) + .json.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402

CHUNK_ROWS = 20_000
PARITY_TOL = {"table": 1e-7, "reference": 1e-7}  # neural: qtest.RELOAD_ATOL["neural"]


# ----------------------------------------------------------------------------- protocol cells
def protocol_cells() -> dict[str, list[str]]:
    rule = C.read_json(B.P1_BEST_RULE)
    proto = B.protocol()["cells"]
    if B.sha256_file(B.P1_BEST_RULE) != B.protocol()["frozen"]["p1_best_rule_sha256"]:
        raise ValueError("P1_BEST_RULE_20261008.json differs from the protocol hash")
    graph = ((0, "team_set_pair_readout"), (1, "relational_mpnn"))
    cells = {
        "P1": [proto["P1"]],
        "PT_flex": [proto["PT_flex"]],
        "P1_best": [rule["selected"]["key"]],
        "P2_quads_C1": [f"final__module__N_G{g}_S{s}_A0_M0_E1_C1_R1_D{d}__{fam}__selected~common" for g, fam in graph for s in (0, 1) for d in (0, 1)],
        "P2_quads_C0": [f"final__module__N_G{g}_S{s}_A0_M0_E1_C0_R1_D{d}__{fam}__selected~common" for g, fam in graph for s in (0, 1) for d in (0, 1)],
        "position_effect": [f"final__module__LGBM_S{s}_A0_M0_E1_C1_R1_D{d}__lightgbm__selected~lightgbm" for s in (0, 1) for d in (0, 1)],
    }
    return cells


def cell_keys() -> list[str]:
    seen, out = set(), []
    for keys in protocol_cells().values():
        for key in keys:
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


# ----------------------------------------------------------------------------- rows
def build_rows(dataset: str, log) -> dict:
    import numpy as np
    from studies.q_production_20261005 import rows as qrows

    out_dir = B.q_inputs_dir(dataset)
    target = out_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.npz"
    meta_path = out_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.json"
    if meta_path.is_file():
        meta = C.read_json(meta_path)
        if meta["rows_sha256"] == B.sha256_file(target):
            log(f"[e2c] {dataset}: q rows exist, reused")
            return meta
    labels_root = B.labels_dir(dataset)
    table = dict(np.load(labels_root / f"LABELS_PUBLIC_h{B.MAIN_HORIZON}.npz", allow_pickle=False))
    with np.load(labels_root / "ENGAGEMENTS.npz", allow_pickle=False) as eng:
        engagements = {name: eng[name] for name in ("canonical_match_key", "component_key", *qrows.EVAL_COLUMNS.values())}
    role = table["role"].astype(np.int64)
    if np.any(role != C.EXTERNAL_ROLE_CODE):
        raise ValueError("a non-EXTERNAL row in an external label table")
    keep = table["valid"].astype(bool)
    rows = np.flatnonzero(keep)
    match_key = table["canonical_match_key"][rows]
    match_index = table["match_index"][rows].astype(np.int64)
    pre_ms = table["pre_ms"][rows].astype(np.int64)
    patch_of = {}
    with (C.stage_dir(dataset, "bank") / "BANK_ENTRIES.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            patch_of[entry["canonical_match_key"].encode("ascii")] = entry["patch"].encode("ascii")
    out = {
        "canonical_match_key": match_key, "component_key": table["component_key"][rows],
        "engagement_key": np.char.add(np.char.add(match_key, b":"), table["component_key"][rows]),
        "match_index": match_index, "role": role[rows].astype(np.int8), "fold": table["fold"][rows].astype(np.int8),
        "q_stop": np.zeros(len(rows), dtype=bool), "y": np.full(len(rows), -1, dtype=np.int8),
        "weight_raw": qrows.match_weights(match_index),
        "query_pre": table["query_pre"][rows].astype(np.int64), "pre_ms": pre_ms, "elapsed_min": pre_ms / 60000.0,
        "frame_age_s": (pre_ms - table["F_pre_ms"][rows].astype(np.int64)) / 1000.0,
        "truncation_bits": table["truncation_bits"][rows], "near_zero_bits": np.zeros(len(rows), dtype=np.uint8),
        "exact_zero": np.full(len(rows), -1, dtype=np.int8), "label_svi_delta": np.full(len(rows), np.nan),
        "pt_p_pre": table["p_pre"][rows],
        "patch": np.asarray([patch_of[bytes(k)] for k in match_key], dtype="S8"),
    }
    out.update(qrows.evaluation_columns(rows, {"canonical_match_key": table["canonical_match_key"], "component_key": table["component_key"]}, engagements))
    if len(np.unique(out["engagement_key"])) != len(rows):
        raise ValueError("engagement keys are not unique")
    if not np.isfinite(out["pt_p_pre"]).all():
        raise ValueError("a valid q row lacks its pre probability")
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = B.write_npz(target, out)
    patches = {p.decode(): int((out["patch"] == p).sum()) for p in sorted(set(out["patch"].tolist()))}
    meta = {"version": B.VERSION, "kind": "external_q_rows", "dataset": dataset, "horizon_s": B.MAIN_HORIZON,
            "rows": int(len(rows)), "matches": int(len(np.unique(match_index))), "rows_by_patch": patches,
            "excluded_invalid_rows": int((~keep).sum()), "rows_sha256": sha,
            "labels_public_sha256": B.sha256_file(labels_root / f"LABELS_PUBLIC_h{B.MAIN_HORIZON}.npz"),
            "sealed_columns": ["y = -1", "label_svi_delta = NaN", "exact_zero = -1", "near_zero_bits = 0"],
            "rule": "rows.build_rows on the valid EXTERNAL rows (q role of every external engagement) + rows.seal_test semantics",
            "positive_counts_computed": False, "utc": C.utc_now()}
    C.write_json(meta_path, meta)
    log(f"[e2c] {dataset}: {meta['rows']} q rows in {meta['matches']} matches {patches}")
    return meta


# ----------------------------------------------------------------------------- raw store
def _chunk_job(job: tuple) -> dict:
    B.env_runtime()
    import numpy as np
    from studies.q_production_20261005 import features as qf

    dataset, queries, chunk_dir = job
    surface = B.external_surface(dataset)
    try:
        return qf.build_chunk(np.asarray(queries, dtype=np.int64), Path(chunk_dir), surface=surface)
    finally:
        surface.close()


def build_store(dataset: str, workers: int, log) -> dict:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import numpy as np
    from studies.q_production_20261005 import features as qf

    out_dir = B.q_inputs_dir(dataset) / "raw"
    build_path = out_dir / "store" / "BUILD.json"
    if build_path.is_file():
        log(f"[e2c] {dataset}: raw store exists, reused")
        return C.read_json(build_path)
    rows = np.load(B.q_inputs_dir(dataset) / f"Q_ROWS_h{B.MAIN_HORIZON}.npz", allow_pickle=False)
    queries = np.unique(np.asarray(rows["query_pre"], dtype=np.int64))
    jobs = [(dataset, queries[start:start + CHUNK_ROWS], str(out_dir / "chunks" / f"chunk_{k:04d}"))
            for k, start in enumerate(range(0, len(queries), CHUNK_ROWS))]
    n_workers = max(1, min(B.cpu_workers_allowed(workers), len(jobs)))
    log(f"[e2c] {dataset}: raw store of {len(queries)} pre queries in {len(jobs)} chunks with {n_workers} worker(s)")
    t0 = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
        for future in as_completed([pool.submit(_chunk_job, job) for job in jobs]):
            record = future.result()
            log(f"[e2c] {dataset}: chunk of {record['rows']} rows done")
    record = qf.merge_chunks([Path(job[2]) for job in jobs], out_dir / "store")
    record["code_sha256"] = B.sha256_file(Path(qf.__file__))
    record["surface"] = "e2bc_common.ExternalQuerySurface"
    record["dataset"] = dataset
    (out_dir / "store" / "BUILD.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    log(f"[e2c] {dataset}: raw store merged, {record['rows']} rows, {record['nonitem_tokens']} non-item tokens, {time.perf_counter() - t0:.0f} s")
    return record


# ----------------------------------------------------------------------------- XY
def build_xy(dataset: str, log) -> dict:
    import numpy as np
    from studies.q_production_20261005 import spatial

    out_dir = B.q_inputs_dir(dataset)
    meta_path = out_dir / "XY.json"
    if meta_path.is_file():
        log(f"[e2c] {dataset}: XY exists, reused")
        return C.read_json(meta_path)
    rows = np.load(out_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.npz", allow_pickle=False)
    record = spatial.read_query_xy(np.asarray(rows["query_pre"], dtype=np.int64), projection=C.stage_dir(dataset, "projection"))
    counts = spatial.support_counts(record)
    if not record.support.all() or not np.isfinite(record.xy).all():
        raise ValueError(f"{dataset}: {int((~record.support).sum())} q rows lack finite pre-frame XY (population would change)")
    xy = np.asarray(record.xy, dtype=np.float64)
    path = out_dir / "XY.npy"
    pending = path.with_name(path.name + ".pending.npy")
    np.save(pending, xy, allow_pickle=False)
    os.replace(pending, path)
    meta = {"version": B.VERSION, "dataset": dataset, "rows": int(len(xy)), "xy_sha256": B.sha256_file(path),
            "support_counts": counts, "reader": "spatial.read_query_xy(projection=dataset projection)", "utc": C.utc_now()}
    C.write_json(meta_path, meta)
    log(f"[e2c] {dataset}: XY for {len(xy)} rows ({counts['rows_with_zero_coordinate_player']} rows with a (0,0) player)")
    return meta


# ----------------------------------------------------------------------------- external context
_CONTEXT_CLASS = None


def external_context_class():
    global _CONTEXT_CLASS
    if _CONTEXT_CLASS is not None:
        return _CONTEXT_CLASS
    from studies.q_production_20261005 import qbatch

    class ExternalScopeContext(qbatch.ScopeContext):
        """The frozen REFIT_TRAIN context over an EXTERNAL row table (no TRAIN rows, so no position digest to recompute)."""

        def __init__(self, store, rows, xy, scope, *, saved, expected_context_json_sha256):
            saved = Path(saved)
            if B.sha256_file(saved / "CONTEXT.json") != expected_context_json_sha256:
                raise ValueError(f"{saved}/CONTEXT.json differs from q_run_1 PREPARED.json")
            self._frozen_scope_digest = json.loads((saved / "CONTEXT.json").read_text(encoding="utf-8"))["scope_positions_sha256"]
            super().__init__(store, rows, xy, scope, saved=saved)

        def scope_digest(self) -> str:
            if len(self.scope_rows):
                raise ValueError("an EXTERNAL row table holds no TRAIN rows")
            return self._frozen_scope_digest

    _CONTEXT_CLASS = ExternalScopeContext
    return ExternalScopeContext


def _prepared_context_sha(saved: Path) -> str:
    saved = Path(saved)
    prepared = C.read_json(saved.parent / "PREPARED.json")
    scope = saved.name
    return prepared["contexts"][scope]["context_json_sha256"]


# ----------------------------------------------------------------------------- scoring
def _neural_logits_padded(cell, rows):
    """LoadedCell.logits (neural branch) with the final partial batch padded to the cell's batch size."""
    import numpy as np
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), cell._batch):
            part = rows[i:i + cell._batch]
            n_real = len(part)
            if n_real < cell._batch:
                part = np.concatenate((part, np.full(cell._batch - n_real, part[-1], dtype=np.int64)))
            inputs = cell._assemble(cell._ctx, part, cell._cond, cell.family)
            out.append(cell._forward(cell._model, inputs, cell.device).double().cpu().numpy()[:n_real])
    return np.concatenate(out) if out else np.zeros(0)


def score_cells(dataset: str, device: str, log, keys=None) -> dict:
    import numpy as np
    from studies.q_production_20261005 import features as qf, qcalibrate, qtest

    lock = qtest.verify_lock(B.Q_RUN_1)
    by_key = {cell["key"]: cell for cell in lock["cells"]}
    q_dir = B.q_inputs_dir(dataset)
    rows_path = q_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.npz"
    ext_rows = qtest.load_npz(rows_path)
    store = qf.RawStore(q_dir / "raw" / "store")
    xy = np.load(q_dir / "XY.npy", allow_pickle=False)
    out_dir = B.scores_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = {}
    results = {}
    all_rows = np.arange(len(ext_rows["query_pre"]), dtype=np.int64)
    for key in (keys or cell_keys()):
        meta_path = out_dir / f"{key}.json"
        if meta_path.is_file():
            meta = C.read_json(meta_path)
            if meta["pred_sha256"] == B.sha256_file(out_dir / f"{key}.npz"):
                results[key] = meta
                continue
        cell = by_key[key]
        if cell["status"] != "completed":
            raise ValueError(f"{key} is not a completed SCORE_LOCK cell")
        t0 = time.perf_counter()
        loaded = qtest.LoadedCell(cell, device=device)
        stored = qtest.load_npz(Path(cell["source_dir"]) / "pred_q_dev.npz")
        check_rows = np.asarray(stored["row"], dtype=np.int64)[:qtest.RELOAD_ROWS]
        again = loaded.logits(check_rows)
        reload_diff = float(np.abs(again - np.asarray(stored["logit"])[:qtest.RELOAD_ROWS]).max()) if len(check_rows) else 0.0
        if reload_diff > qtest.RELOAD_ATOL[loaded.kind]:
            raise RuntimeError(f"{key}: re-loaded model differs from its stored Q_DEV predictions by {reload_diff}")
        ext = copy.copy(loaded)
        ext.rows = ext_rows
        if loaded.kind != "reference":
            saved = Path(loaded.spec["inputs"]["contexts"]["REFIT_TRAIN"])
            if str(saved) not in contexts:
                contexts[str(saved)] = external_context_class()(store, ext_rows, xy, "REFIT_TRAIN", saved=saved,
                                                                 expected_context_json_sha256=_prepared_context_sha(saved))
            ext._ctx = contexts[str(saved)]
        logits = _neural_logits_padded(ext, all_rows) if loaded.kind == "neural" else qtest.LoadedCell.logits(ext, all_rows)
        if not np.isfinite(logits).all() or len(logits) != len(all_rows):
            raise ValueError(f"{key}: non-finite or missing external logits")
        p = qtest.sigmoid(logits)
        calibrator_path = B.Q_RUN_1 / "calibrators" / f"{key}.json"
        p_cal, calibration = None, "missing"
        if calibrator_path.is_file():
            calibrator = qcalibrate.QCalibrator.from_dict(qtest.read_json(calibrator_path)["calibrator"])
            calibration = calibrator.status
            if calibrator.fitted:
                p_cal = calibrator.predict(p)
        arrays = {"row": all_rows, "engagement_key": np.asarray(ext_rows["engagement_key"]), "logit": logits, "p": p}
        if p_cal is not None:
            arrays["p_cal"] = p_cal
        sha = B.write_npz(out_dir / f"{key}.npz", arrays)
        meta = {"key": key, "dataset": dataset, "rows": int(len(all_rows)), "pred_sha256": sha, "kind": loaded.kind,
                "reload_max_abs_logit_diff": reload_diff, "reload_rows": int(len(check_rows)), "reload_atol": qtest.RELOAD_ATOL[loaded.kind],
                "calibration": calibration, "calibrator_sha256": B.sha256_file(calibrator_path) if calibrator_path.is_file() else None,
                "model_sha256": cell.get("model_sha256"), "result_sha256": cell.get("result_sha256"),
                "rows_table": str(rows_path), "rows_table_sha256": B.sha256_file(rows_path),
                "context": None if loaded.kind == "reference" else str(saved), "device": device if loaded.kind == "neural" else "cpu",
                "neural_last_batch_padded": loaded.kind == "neural", "seconds": round(time.perf_counter() - t0, 1), "utc": C.utc_now()}
        C.write_json(meta_path, meta)
        results[key] = meta
        log(f"[e2c] {dataset}: {key} {loaded.kind} rows={len(all_rows)} reload_diff={reload_diff:.2e} {meta['seconds']} s")
        del loaded, ext
    return results


# ----------------------------------------------------------------------------- parity
def _shape_matched_logits(dataset: str, key: str, rows: list, shapes: list, device: str):
    """External-path logits of single parity rows, each in a batch of the reference's batch shape (padded copies)."""
    import numpy as np
    import torch
    from studies.q_production_20261005 import features as qf, qtest

    lock = qtest.verify_lock(B.Q_RUN_1)
    cell = {c["key"]: c for c in lock["cells"]}[key]
    q_dir = B.q_inputs_dir(dataset)
    ext_rows = qtest.load_npz(q_dir / f"Q_ROWS_h{B.MAIN_HORIZON}.npz")
    store = qf.RawStore(q_dir / "raw" / "store")
    xy = np.load(q_dir / "XY.npy", allow_pickle=False)
    loaded = qtest.LoadedCell(cell, device=device)
    ext = copy.copy(loaded)
    ext.rows = ext_rows
    saved = Path(loaded.spec["inputs"]["contexts"]["REFIT_TRAIN"])
    ext._ctx = external_context_class()(store, ext_rows, xy, "REFIT_TRAIN", saved=saved,
                                        expected_context_json_sha256=_prepared_context_sha(saved))
    out = []
    with torch.no_grad():
        for row, shape in zip(rows, shapes):
            part = np.full(int(shape), int(row), dtype=np.int64)
            inputs = ext._assemble(ext._ctx, part, ext._cond, ext.family)
            out.append(float(ext._forward(ext._model, inputs, ext.device).double().cpu().numpy()[0]))
    return np.asarray(out)


def parity_scores(log, device: str = "cuda") -> dict:
    import numpy as np
    from studies.q_production_20261005 import qtest

    dataset = C.DATASET_PARITY
    report = {"version": B.VERSION, "kind": "E2c_PARITY_REPORT", "created_utc": C.utc_now(), "dataset": dataset,
              "reference": str(B.Q_RUN_1 / qtest.SCORES_DIR), "tolerance": {**PARITY_TOL, "neural": qtest.RELOAD_ATOL["neural"]},
              "rule": ("logits of the external path versus TEST_SCORES for the same engagement keys. Neural cells are scored "
                       "in full 512-row batches (final partial batch padded), so a parity row that TEST_SCORES computed in "
                       "its own final partial batch (the last 222,154 mod 512 = 458 TEST rows) was computed under another "
                       "batch shape; such rows are also re-scored by the external path in a batch of the reference shape "
                       "('shape_matched'), and the gate uses that value for them."),
              "cells": {}}
    ok = True
    test_rows = qtest.load_npz(B.Q_RUN_1 / "Q_ROWS_h90.npz")
    n_test = int((np.asarray(test_rows["role"]) == qtest.TEST_ROLE).sum())
    del test_rows
    for key in cell_keys():
        meta = C.read_json(B.scores_dir(dataset) / f"{key}.json")
        mine = qtest.load_npz(B.scores_dir(dataset) / f"{key}.npz")
        ref = qtest.load_npz(B.Q_RUN_1 / qtest.SCORES_DIR / f"{key}.npz")
        index = {k: i for i, k in enumerate(ref["engagement_key"].tolist())}
        missing = [k for k in mine["engagement_key"].tolist() if k not in index]
        entry = {"kind": meta["kind"], "rows": int(len(mine["logit"])), "missing_in_reference": len(missing)}
        if missing:
            ok = False
            entry["pass"] = False
            report["cells"][key] = entry
            continue
        pos = np.asarray([index[k] for k in mine["engagement_key"].tolist()], dtype=np.int64)
        tol = qtest.RELOAD_ATOL["neural"] if meta["kind"] == "neural" else PARITY_TOL[meta["kind"]]
        diff = np.abs(np.asarray(mine["logit"]) - np.asarray(ref["logit"])[pos])
        entry.update(logit_max_abs_diff=float(diff.max()), logit_bitwise_equal_rows=int((diff == 0).sum()), tolerance=tol,
                     p_max_abs_diff=float(np.abs(mine["p"] - ref["p"][pos]).max()))
        if "p_cal" in mine and "p_cal" in ref:
            entry["p_cal_max_abs_diff"] = float(np.abs(mine["p_cal"] - ref["p_cal"][pos]).max())
        gate = diff.copy()
        if meta["kind"] == "neural":
            batch = 512
            ref_test_pos = pos  # TEST_SCORES arrays are in TEST order (qtest.score)
            last_start = n_test - n_test % batch
            partial = np.flatnonzero(ref_test_pos >= last_start)
            entry["rows_in_reference_partial_batch"] = int(len(partial))
            entry["full_batch_rows_bitwise_equal"] = bool(np.all(diff[np.setdiff1d(np.arange(len(diff)), partial)] == 0))
            if len(partial):
                shape = n_test - last_start
                matched = _shape_matched_logits(dataset, key, partial.tolist(), [shape] * len(partial), device)
                matched_diff = np.abs(matched - np.asarray(ref["logit"])[pos][partial])
                entry["shape_matched"] = {"rows": int(len(partial)), "batch_shape": int(shape),
                                          "as_scored_max_abs_diff": float(diff[partial].max()),
                                          "shape_matched_max_abs_diff": float(matched_diff.max())}
                gate[partial] = matched_diff
        entry["gate_max_abs_diff"] = float(gate.max())
        entry["pass"] = bool(gate.max() <= tol)
        ok &= entry["pass"]
        report["cells"][key] = entry
    report["n_cells"] = len(report["cells"])
    report["status"] = "PASS" if ok else "FAIL"
    C.write_json(B.E2C_PARITY, report)
    log(f"[e2c] parity status={report['status']} max diffs " + json.dumps({k[:60]: v.get("logit_max_abs_diff") for k, v in report["cells"].items()}))
    return report


def _score_cells_job(job: tuple) -> dict:
    B.env_runtime()
    dataset, keys, device = job
    return score_cells(dataset, device, C.Log(C.LOGS / f"E2c_{dataset}.log"), keys=keys)


def run_score_cells(dataset: str, workers: int, device: str, log) -> dict:
    """score_cells over the protocol cells, split round-robin across <= the allowed worker processes."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    keys = cell_keys()
    out_dir = B.scores_dir(dataset)
    pending = [k for k in keys if not ((out_dir / f"{k}.json").is_file() and
                                       C.read_json(out_dir / f"{k}.json")["pred_sha256"] == B.sha256_file(out_dir / f"{k}.npz"))]
    n_workers = max(1, min(B.cpu_workers_allowed(workers), len(pending) or 1))
    log(f"[e2c] {dataset}: {len(pending)} of {len(keys)} cells to score with {n_workers} worker(s)")
    if pending:
        groups = [pending[i::n_workers] for i in range(n_workers)]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
            for result in pool.map(_score_cells_job, [(dataset, group, device) for group in groups]):
                log(f"[e2c] {dataset}: worker finished {len(result)} cells")
    return {k: C.read_json(out_dir / f"{k}.json") for k in keys}


def run_e2c(dataset: str, workers: int, device: str, log) -> dict:
    with B.Timer(dataset, "e2c_rows", log):
        rows = build_rows(dataset, log)
    with B.Timer(dataset, "e2c_raw_store", log):
        store = build_store(dataset, workers, log)
    with B.Timer(dataset, "e2c_xy", log):
        xy = build_xy(dataset, log)
    with B.Timer(dataset, "e2c_score_cells", log, device=device):
        scores = run_score_cells(dataset, workers, device, log)
    return {"rows": rows["rows"], "store_rows": store["rows"], "xy_rows": xy["rows"], "cells": len(scores)}
