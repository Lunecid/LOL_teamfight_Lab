"""E2b: external engagement table, final-context V scores and sealed labels (labels_1 rules and code paths).

README
  L0  studies/label_production_20261005/engagements.build(output, bank, projection) - frozen, called with the dataset's
      E2a bank and projection (EXTERNAL rows keep role 6 / patch_role 3; the frozen role_counts only count codes 0..5).
  L1  query list of the final context: sorted unique pre/h60/h90/h120 queries of the EXTERNAL rows - the result of
      queries.context_queries(.., 'final') for rows the final V scores (transcribed: that function selects the final
      rows by the 2025 role names Q_DEV/Q_CAL/TEST).  Written as QUERIES_final.npy, then read by score.query_list.
  L2  scoring: transcription of score.score_parts with the dataset's ExternalQuerySurface instead of
      queries.open_inputs (2025 paths); unchanged frozen pieces: score.LockedV(run_3, 'final') with its V_CAL
      calibrator, queries.transform with the final_REFIT_TRAIN bundle (queries.open_inputs' bundle), canonical
      1,024-row batches, score.part_ranges / part_path / part_complete, part receipts; merged by
      reports/label_production_20261005/run_labels.merge into PRED_final.npz + manifest.  One addition: the final
      partial batch is padded to 1,024 rows with copies of its last row (padded outputs dropped).  The V's GPU matmuls
      are per-row identical for every full batch but not across batch shapes (labels_1 VALIDATION_LAYER1: up to
      2.4e-6 in logits), and without the padding the parity gate showed 229 of 308 last-batch rows 1-2 float32 ulps
      away from labels_1 while all 4,096 full-batch rows were bitwise equal.
  L3  assemble.label_table(eng, {'final': pred}, membership, h) - frozen, with assemble.Q_FINAL_ROLES extended by the
      EXTERNAL code 6 for the call (the final V labels every external engagement, as it labels Q_DEV/Q_CAL/TEST); the
      membership matrix is the dataset projection's CONTEXT_MEMBERSHIP (all 0).
  Sealing: the label-bearing columns (e2bc_common.SEALED_LABEL_COLUMNS) of every horizon go only to
  EXTERNAL_LABELS_SEALED_<dataset>.npz (h{h}__engagement_key, h{h}__y = direction_positive, h{h}__svi_delta,
  h{h}__exact_zero, h{h}__near_zero_bits, ... ); LABELS_PUBLIC_h{h}.npz keep the rest (keys, times, validity,
  invalid bits, query refs, p_pre / raw_pre / logit_pre: p_pre is the PT_flex input).  LABELS_MANIFEST_EXTERNAL.json
  records hashes and row / valid counts only.  No label value, positive rate or V outcome metric is computed for an
  external set.  PRED_final.npz (V at every pre and post query) implies the labels; it is bound by hash and read only
  by the assembly.
  Parity (parity_15_16): the 500 accepted 15.16 TEST matches; labels_1 TEST rows were opened on 10-08, so the parity
  labels are compared directly with labels_1 (keys and direction exact; SVI, logits and p within 1e-6).
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402

LABEL_TOL = 1e-6


# ----------------------------------------------------------------------------- L0 engagements
def run_engagements(dataset: str, log) -> dict:
    import numpy as np
    from studies.label_production_20261005 import engagements

    out = B.labels_dir(dataset)
    manifest = out / "ENGAGEMENTS_MANIFEST.json"
    if manifest.is_file() and (out / "ENGAGEMENTS.npz").is_file():
        record = C.read_json(manifest)
        if record.get("engagements_sha256") == B.sha256_file(out / "ENGAGEMENTS.npz"):
            log(f"[e2b] {dataset}: engagement table exists, reused")
            return record
    out.mkdir(parents=True, exist_ok=True)
    record = engagements.build(out, bank=C.stage_dir(dataset, "bank"), projection=C.stage_dir(dataset, "projection"))
    with np.load(out / "ENGAGEMENTS.npz", allow_pickle=False) as eng:
        role, patch_role, fold = eng["role"], eng["patch_role"], eng["fold"]
        if not (np.all(role == C.EXTERNAL_ROLE_CODE) and np.all(patch_role == C.EXTERNAL_PATCH_ROLE_CODE) and np.all(fold == -1)):
            raise ValueError(f"{dataset}: an engagement row is not EXTERNAL")
    log(f"[e2b] {dataset}: {record['rows']} engagement rows, valid {record['valid_counts']}")
    return record


# ----------------------------------------------------------------------------- L1 query list
def write_query_list(dataset: str) -> tuple:
    import numpy as np

    out = B.labels_dir(dataset)
    path = out / f"QUERIES_{B.V_CONTEXT}.npy"
    with np.load(out / "ENGAGEMENTS.npz", allow_pickle=False) as eng:
        own = eng["role"].astype(np.int64) == C.EXTERNAL_ROLE_CODE
        parts = [np.asarray(eng[name])[own] for name in ("query_pre", "query_h60", "query_h90", "query_h120")]
    values = np.concatenate(parts).astype(np.int64)
    indices = np.unique(values[values >= 0])
    if path.is_file():
        stored = np.load(path, allow_pickle=False)
        if not np.array_equal(stored, indices):
            raise ValueError(f"{dataset}: stored query list differs from the engagement table")
        return indices, path
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.save(stream, indices, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return indices, path


# ----------------------------------------------------------------------------- L2 scoring
_WORKER: dict = {}


def _init_scorer(dataset: str, device: str) -> None:
    B.env_runtime()
    from studies.label_production_20261005 import queries, score
    from studies.v_full_preprocess_20261003.artifacts import load_bundle

    _WORKER["v"] = score.LockedV(B.RUN_3, B.V_CONTEXT, device=device)
    _WORKER["surface"] = B.external_surface(dataset)
    _WORKER["bundle"] = load_bundle(queries.PREPROCESS / f"{B.V_CONTEXT}_REFIT_TRAIN")
    _WORKER["dataset"] = dataset


def _score_part(part: int) -> dict:
    """score.score_parts body for one part (canonical batches, receipt last)."""
    import numpy as np
    from studies.label_production_20261005 import queries, score

    dataset, v, surface, bundle = _WORKER["dataset"], _WORKER["v"], _WORKER["surface"], _WORKER["bundle"]
    labels_root = B.labels_dir(dataset)
    indices, list_path = score.query_list(labels_root, B.V_CONTEXT)
    ranges = score.part_ranges(len(indices))
    start, stop = ranges[part]
    chunk = indices[start:stop]
    target = score.part_path(labels_root, B.V_CONTEXT, part)
    if score.part_complete(target, chunk):
        return {"part": part, "skipped": True}
    t0 = time.perf_counter()
    logits, raw, calibrated = [], [], []
    for b_start in range(0, len(chunk), queries.BATCH_ROWS):
        batch = chunk[b_start:b_start + queries.BATCH_ROWS]
        rows, seen = [], []
        for query_index, row, _meta in surface.iter_query_rows(batch):
            rows.append(row)
            seen.append(query_index)
        if not np.array_equal(np.asarray(seen, dtype=np.int64), batch):
            raise ValueError(f"{dataset}: query rows came back out of order in part {part}")
        n_real = len(rows)
        if n_real < queries.BATCH_ROWS:
            # Shape padding (E2b parity diagnosis): the V's GPU matmuls give the same per-row result for every
            # 1,024-row batch but not for a shorter one (labels_1 VALIDATION_LAYER1 batch-size check, up to 2.4e-6
            # in logits).  The final partial batch is filled to 1,024 rows with copies of its last row and only the
            # real rows are kept, so every query is scored under the shape of the canonical full batches.
            rows = rows + [rows[-1]] * (queries.BATCH_ROWS - n_real)
        values = v.score(queries.transform(rows, bundle))
        for store, value in zip((logits, raw, calibrated), values):
            store.append(np.asarray(value)[:n_real])
    payload = {"query_index": chunk, "logit": np.concatenate(logits), "raw_p": np.concatenate(raw), "v": np.concatenate(calibrated)}
    for name in ("raw_p", "v"):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"{dataset}: non-finite {name} in part {part}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, target)
    code = score.code_hashes()
    code.update(B.code_hashes([Path(__file__), C.PACKAGE / "e2bc_common.py"]))
    receipt = {"version": score.VERSION, "context": B.V_CONTEXT, "part": part, "rows": int(len(chunk)), "range": [int(start), int(stop)],
               "sha256": score.sha256_file(target), "queries_sha256": score.sha256_file(list_path), "v": v.identity,
               "code_sha256": code, "batch_rows": queries.BATCH_ROWS, "utc": C.utc_now(), "dataset": dataset,
               "surface": "e2bc_common.ExternalQuerySurface"}
    pending_receipt = target.with_suffix(".json.pending")
    pending_receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending_receipt, target.with_suffix(".json"))
    return {"part": part, "skipped": False, "rows": int(len(chunk)), "seconds": round(time.perf_counter() - t0, 1)}


def run_scoring(dataset: str, workers: int, device: str, log) -> dict:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import numpy as np
    from studies.label_production_20261005 import score

    sys.path.insert(0, str(C.ROOT / "reports" / "label_production_20261005"))
    import run_labels  # noqa: E402  (frozen merge)

    labels_root = B.labels_dir(dataset)
    if (labels_root / f"PRED_{B.V_CONTEXT}_MANIFEST.json").is_file():
        log(f"[e2b] {dataset}: PRED_{B.V_CONTEXT} exists, reused")
        return C.read_json(labels_root / f"PRED_{B.V_CONTEXT}_MANIFEST.json")
    indices, _ = write_query_list(dataset)
    ranges = score.part_ranges(len(indices))
    todo = [part for part, (a, b) in enumerate(ranges) if not score.part_complete(score.part_path(labels_root, B.V_CONTEXT, part), indices[a:b])]
    n_workers = max(1, min(B.cpu_workers_allowed(workers), len(todo) or 1))
    log(f"[e2b] {dataset}: {len(indices)} queries in {len(ranges)} parts ({len(todo)} to score) with {n_workers} worker(s) on {device}")
    t0 = time.perf_counter()
    if todo:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context, initializer=_init_scorer,
                                 initargs=(dataset, device)) as pool:
            done = 0
            for future in as_completed([pool.submit(_score_part, part) for part in todo]):
                result = future.result()
                done += 1
                log(f"[e2b] {dataset}: part {result['part']} done ({done}/{len(todo)}) {result.get('seconds')} s")
    record = run_labels.merge(labels_root, B.V_CONTEXT)
    log(f"[e2b] {dataset}: merged PRED_{B.V_CONTEXT}: {record['rows']} rows in {time.perf_counter() - t0:.0f} s")
    return record


# ----------------------------------------------------------------------------- L3 labels + sealing
@contextmanager
def _final_roles_with_external():
    from studies.label_production_20261005 import assemble

    saved = assemble.Q_FINAL_ROLES
    assemble.Q_FINAL_ROLES = tuple(saved) + (C.EXTERNAL_ROLE_CODE,)
    try:
        yield assemble
    finally:
        assemble.Q_FINAL_ROLES = saved


def assemble_labels(dataset: str, log) -> dict:
    import numpy as np

    labels_root = B.labels_dir(dataset)
    manifest_path = labels_root / "LABELS_MANIFEST_EXTERNAL.json"
    if manifest_path.is_file():
        log(f"[e2b] {dataset}: sealed labels exist, reused")
        return C.read_json(manifest_path)
    eng_path = labels_root / "ENGAGEMENTS.npz"
    eng = dict(np.load(eng_path, allow_pickle=False))
    membership_path = C.stage_dir(dataset, "projection") / "CONTEXT_MEMBERSHIP.npy"
    membership = np.load(membership_path, allow_pickle=False)
    if np.any(membership != 0):
        raise ValueError(f"{dataset}: EXTERNAL context membership is not all 0")
    keys = np.char.add(np.char.add(eng["canonical_match_key"], b":"), eng["component_key"])
    sealed, files, counts = {}, {}, {}
    with _final_roles_with_external() as assemble:
        pred, pred_manifest = assemble.load_predictions(labels_root, B.V_CONTEXT)
        for horizon in B.HORIZONS:
            table = assemble.label_table(eng, {B.V_CONTEXT: pred}, membership, horizon)
            if not np.array_equal(table["canonical_match_key"], eng["canonical_match_key"]):
                raise ValueError("label rows do not follow the engagement table")
            if np.any(table["context"] != 0):
                raise ValueError(f"{dataset}: an EXTERNAL row is not scored by the final V")
            public = {name: value for name, value in table.items() if name not in B.SEALED_LABEL_COLUMNS}
            public["engagement_key"] = keys
            files[f"LABELS_PUBLIC_h{horizon}.npz"] = B.write_npz(labels_root / f"LABELS_PUBLIC_h{horizon}.npz", public)
            sealed[f"h{horizon}__engagement_key"] = keys
            sealed[f"h{horizon}__valid"] = table["valid"]
            sealed[f"h{horizon}__y"] = table["direction_positive"]
            for name in B.SEALED_LABEL_COLUMNS:
                if name != "direction_positive":
                    sealed[f"h{horizon}__{name}"] = table[name]
            valid = table["valid"].astype(bool)
            counts[f"h{horizon}"] = {"rows": int(len(valid)), "valid": int(valid.sum()), "invalid": int((~valid).sum()),
                                     "invalid_reasons": {reason: int(((table["invalid_bits"] & bit) != 0).sum())
                                                         for reason, bit in assemble.INVALID.items()}}
        invalid_codes = dict(assemble.INVALID)
    sealed_path = B.sealed_labels_path(dataset)
    files[sealed_path.name] = B.write_npz(sealed_path, sealed)
    record = {"version": B.VERSION, "kind": "external_labels", "dataset": dataset, "created_utc": C.utc_now(),
              "rule": "labels_1 (LABEL_LOCK.json): SVI = V_cal(post) - V_cal(pre), final-context V; y = 1[SVI > 0]",
              "engagements_sha256": B.sha256_file(eng_path),
              "engagements_manifest_sha256": B.sha256_file(labels_root / "ENGAGEMENTS_MANIFEST.json"),
              "pred_final_sha256": pred_manifest["pred_sha256"], "pred_final_manifest_sha256": B.sha256_file(labels_root / f"PRED_{B.V_CONTEXT}_MANIFEST.json"),
              "v": pred_manifest["v"], "context_membership_sha256": B.sha256_file(membership_path),
              "files": files, "sealed_file": str(sealed_path), "sealed_sha256": files[sealed_path.name],
              "sealed_columns": ["engagement_key", "valid", "y (= direction_positive)", *[c for c in B.SEALED_LABEL_COLUMNS if c != "direction_positive"]],
              "public_columns_note": "LABELS_PUBLIC_h*.npz: frozen label_table columns minus the sealed ones, plus engagement_key",
              "counts": counts, "invalid_bits": invalid_codes,
              "hook": "assemble.Q_FINAL_ROLES extended by EXTERNAL code 6 during label_table only",
              "label_values_read": False}
    C.write_json(manifest_path, record)
    log(f"[e2b] {dataset}: sealed labels written ({ {h: c['valid'] for h, c in counts.items()} } valid rows); sealed sha {files[sealed_path.name][:16]}")
    return record


# ----------------------------------------------------------------------------- parity against labels_1
def parity_labels(log) -> dict:
    """parity_15_16 versus labels_1 TEST rows (labels opened 10-08): keys/direction exact, SVI/logits/p within 1e-6."""
    import numpy as np

    dataset = C.DATASET_PARITY
    labels_root = B.labels_dir(dataset)
    sealed = dict(np.load(B.sealed_labels_path(dataset), allow_pickle=False))
    subset = {row["canonical_match_key"] for row in C.read_json(C.PARITY_SUBSET)["matches"]}
    report = {"version": B.VERSION, "kind": "E2b_PARITY_REPORT", "created_utc": C.utc_now(), "dataset": dataset,
              "reference": str(B.LABELS_1), "tolerance": LABEL_TOL, "horizons": {}}
    ok = True
    for horizon in B.HORIZONS:
        ref = dict(np.load(B.LABELS_1 / f"LABELS_h{horizon}.npz", allow_pickle=False))
        pub = dict(np.load(labels_root / f"LABELS_PUBLIC_h{horizon}.npz", allow_pickle=False))
        in_subset = np.isin(np.char.decode(ref["canonical_match_key"], "ascii"), list(subset))
        ref_keys = np.char.add(np.char.add(ref["canonical_match_key"], b":"), ref["component_key"])[in_subset]
        new_keys = sealed[f"h{horizon}__engagement_key"]
        result = {"reference_rows": int(in_subset.sum()), "rows": int(len(new_keys)),
                  "reference_roles": sorted({int(v) for v in ref["role"][in_subset]})}
        same_set = set(ref_keys.tolist()) == set(new_keys.tolist()) and len(ref_keys) == len(new_keys)
        result["keys_exact"] = bool(same_set)
        if not same_set:
            ok = False
            report["horizons"][f"h{horizon}"] = result
            continue
        pos_ref = {k: i for i, k in enumerate(ref_keys.tolist())}
        order = np.asarray([pos_ref[k] for k in new_keys.tolist()], dtype=np.int64)
        ref_rows = np.flatnonzero(in_subset)[order]

        def col(table, name, rows=None):
            value = np.asarray(table[name])
            return value if rows is None else value[rows]

        checks = {}
        checks["valid"] = bool(np.array_equal(col(ref, "valid", ref_rows), sealed[f"h{horizon}__valid"]))
        checks["invalid_bits"] = bool(np.array_equal(col(ref, "invalid_bits", ref_rows), pub["invalid_bits"]))
        checks["direction_exact"] = bool(np.array_equal(col(ref, "direction_positive", ref_rows), sealed[f"h{horizon}__y"]))
        checks["exact_zero_exact"] = bool(np.array_equal(col(ref, "exact_zero", ref_rows), sealed[f"h{horizon}__exact_zero"]))
        checks["near_zero_bits_exact"] = bool(np.array_equal(col(ref, "near_zero_bits", ref_rows), sealed[f"h{horizon}__near_zero_bits"]))
        for name, new in (("svi_delta", sealed[f"h{horizon}__svi_delta"]), ("p_post", sealed[f"h{horizon}__p_post"]),
                          ("logit_post", sealed[f"h{horizon}__logit_post"]), ("p_pre", pub["p_pre"]), ("logit_pre", pub["logit_pre"])):
            a, b = col(ref, name, ref_rows).astype(np.float64), np.asarray(new, dtype=np.float64)
            both = np.isfinite(a) & np.isfinite(b)
            same_nan = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
            diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            checks[f"{name}_max_abs_diff"] = diff
            checks[f"{name}_within_tol"] = bool(same_nan and diff <= LABEL_TOL)
            checks[f"{name}_bitwise_equal_rows"] = int(np.sum(a[both] == b[both]))
        # global query indices / match_index are subset positions; the times and endpoint reasons must be equal
        for name in ("first_kill_ms", "last_kill_ms", "pre_ms", "post_ms", "F_pre_ms", "F_post_ms", "truncation_bits"):
            checks[f"{name}_exact"] = bool(np.array_equal(col(ref, name, ref_rows), pub[name]))
        result["checks"] = checks
        result["pass"] = all(v for k, v in checks.items() if k.endswith(("_exact", "_within_tol")) or k in ("valid", "invalid_bits"))
        ok &= result["pass"]
        report["horizons"][f"h{horizon}"] = result
    report["status"] = "PASS" if ok else "FAIL"
    C.write_json(B.E2B_PARITY, report)
    log(f"[e2b] parity status={report['status']} " + json.dumps({h: {k: v for k, v in r.get('checks', {}).items() if 'max_abs' in k} for h, r in report['horizons'].items()}))
    return report


def run_e2b(dataset: str, workers: int, device: str, log) -> dict:
    with B.Timer(dataset, "e2b_engagements", log):
        engagements_record = run_engagements(dataset, log)
    with B.Timer(dataset, "e2b_v_scoring", log, device=device):
        pred_record = run_scoring(dataset, workers, device, log)
    with B.Timer(dataset, "e2b_labels_sealed", log):
        labels_record = assemble_labels(dataset, log)
    return {"engagements": {k: engagements_record[k] for k in ("rows", "valid_counts", "engagements_sha256")},
            "pred_final": {k: pred_record[k] for k in ("rows", "pred_sha256", "queries_sha256")}, "labels": labels_record["counts"]}
