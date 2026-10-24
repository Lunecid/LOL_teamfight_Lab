"""E2v: final-context V on the nominal V rows of the external sets, and the EXTERNAL_SCORE_LOCK addendum.

README (coordinator 10-09, after EXTERNAL_LOCK_CHECK.json passed; protocol metrics.v needs the final V on the nominal V
rows as well as at tau, and E2b scored only the engagement pre/post states)
  N1  Query list: transcription of reports/v_report_20261005/v_test_tau.nominal_queries with the EXTERNAL role code 6 on
      the dataset projection's table copies (V_ROWS, MATCH_TABLE, MATCH_QUERIES): supported nominal rows, query index =
      MATCH_QUERIES.first_query[match_index] + V_ROWS.query_ref, sorted and unique.  Written once as
      labels/<dataset>/QUERIES_nominal_final.npy.
  N2  Scoring: the body of v_test_tau.score_queries / score.score_parts with the dataset's ExternalQuerySurface
      (e2bc_common) in place of queries.open_inputs.  Unchanged frozen pieces: score.LockedV(run_3, 'final') with its
      V_CAL calibrator, queries.transform with the final_REFIT_TRAIN bundle, 1,024-row batches in 32-batch parts
      (score.part_ranges; v_test_tau.PART_ROWS is the same 32 x 1,024).  The workers start through
      e2b_labels._init_scorer (the E2b scorer).  The E2b padding rule holds: the final partial batch is padded to
      1,024 rows with copies of its last row, and the padded outputs are dropped.  Part files go to
      labels/<dataset>/PRED_nominal_final/part_*.npz, each with a receipt.
  N3  Merge: the part checks of run_labels.merge (receipt sha256, query order, one V identity and code set).  Each query
      then gets its public nominal identity: canonical match key, patch (from BANK_ENTRIES.jsonl), dataset match index,
      V_ROWS position, nominal_index, bucket_id, bucket_lo_ms, bucket_hi_ms, nominal_ms, actual_ms, query_ms and
      frame_age_seconds.  The result is written once as labels/<dataset>/PRED_nominal_final.npz plus
      PRED_nominal_final_MANIFEST.json.  MATCH_TABLE.target_Blue_win (the match winner) is never read or written, and
      no metric is computed.
  Parity (parity_15_16): each parity nominal row is found in the main-lineage TEST nominal list
      (v_test_tau.nominal_queries(5) on the main projection), joined by canonical match key and nominal position.  The
      nominal identity fields must agree exactly, and the reference parts must cover the list as v_test_tau.cmd_report
      requires.  Its stored V_TEST_PRED_final value must then be bitwise equal or within 1e-6 in logit, raw p and
      calibrated p.  The reference did not pad its batches, so the rows that it scored in a partial batch are counted
      separately.
  Stage times: reports/external_validation_20261009/E2v_NOMINAL_STAGE_TIMES.jsonl, a new file.  E2bc_STAGE_TIMES.jsonl
      is bound by EXTERNAL_SCORE_LOCK.json and stays unchanged.
  Addendum: EXTERNAL_SCORE_LOCK_ADDENDUM.json binds the lock sha256, the nominal parity report, the four
      PRED_nominal_final files (with manifests and query lists), the new code and the stage times.  The addendum's own
      stage line is appended before the stage-time file is hashed.

Usage
  python -B studies/external_validation_20261009/e2v_nominal.py parity --workers 1
  python -B studies/external_validation_20261009/e2v_nominal.py run --dataset kr_15_18_22 --workers 4
  python -B studies/external_validation_20261009/e2v_nominal.py addendum
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402
from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402

VERSION = "external_validation_e2v_nominal_20261009"
TAG = "nominal_final"                 # QUERIES_nominal_final.npy, PRED_nominal_final/ parts, PRED_nominal_final.npz
MAX_WORKERS = 4                       # coordinator 10-09: a q LightGBM job runs beside this stage
TOL = 1e-6
BATCH_ROWS = 1024                     # queries.BATCH_ROWS (checked at run time)
PART_ROWS = 32 * BATCH_ROWS           # score.part_ranges = v_test_tau.PART_ROWS
REFERENCE_ROLE = 5                    # TEST in the main-lineage MATCH_TABLE
LOCK_SHA256 = "0ba8c255b465587b19813d3925a8acdc603877dd9790de27ccb117804dfe0160"

STAGE_TIMES = C.REPORTS / "E2v_NOMINAL_STAGE_TIMES.jsonl"
PARITY_REPORT = C.REPORTS / "E2v_NOMINAL_PARITY_REPORT.json"
ADDENDUM = C.REPORTS / "EXTERNAL_SCORE_LOCK_ADDENDUM.json"
LOCK_CHECK = C.REPORTS / "EXTERNAL_LOCK_CHECK.json"
MAIN_PROJECTION = C.ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
REFERENCE_DIR = B.TRAINING / "v_report_1" / "V_TEST_PRED_final"
REFERENCE_SCRIPT = C.ROOT / "reports/v_report_20261005/v_test_tau.py"
IDENTITY_FIELDS = ("nominal_index", "bucket_id", "bucket_lo_ms", "bucket_hi_ms", "nominal_ms", "actual_ms")
PARITY_FIELDS = (*IDENTITY_FIELDS, "query_ref", "frame_index", "event_limit", "supported", "support_reason_mask")
VALUE_FIELDS = ("logit", "raw_p", "v")
WINNER_FIELD = "target_Blue_win"
V_IDENTITY_FIELDS = ("context", "family", "card_id", "checkpoint_sha256", "v_lock_sha256", "scientific_sha256",
                     "training_acceptance_sha256", "calibrator")
PROJECTION_TABLES = ("V_ROWS.npy", "MATCH_TABLE.npy", "MATCH_QUERIES.npy", "QUERY_TABLE.npy", "FRAME_AGE_SECONDS.npy")


# ----------------------------------------------------------------------------- small rules (unit-tested)
def workers_allowed(requested: int) -> int:
    """The E2bc CPU gate, capped at 4 for this stage."""
    return max(1, min(int(requested), MAX_WORKERS, B.cpu_workers_allowed(requested)))


def pad_batch(rows: list, batch_rows: int = BATCH_ROWS) -> tuple[list, int]:
    """(rows padded to batch_rows with copies of the last row, number of real rows)."""
    n_real = len(rows)
    if n_real == 0 or n_real > batch_rows:
        raise ValueError(f"batch of {n_real} rows")
    return rows + [rows[-1]] * (batch_rows - n_real), n_real


def batch_size_at(position: int, n_total: int, part_rows: int = PART_ROWS, batch_rows: int = BATCH_ROWS) -> int:
    """Rows in the unpadded batch that held list position `position` (score_queries batching of score.part_ranges parts)."""
    if not 0 <= position < n_total:
        raise IndexError(position)
    part_lo = (position // part_rows) * part_rows
    part_len = min(part_rows, n_total - part_lo)
    batch_lo = ((position - part_lo) // batch_rows) * batch_rows
    return min(batch_rows, part_len - batch_lo)


def nominal_rows(projection: Path, role: int) -> dict:
    """v_test_tau.nominal_queries(role) on a projection, plus the V_ROWS position of every kept row."""
    import numpy as np

    rows = np.load(Path(projection) / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    matches = np.load(Path(projection) / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    mq = np.load(Path(projection) / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
    mi = np.array(rows["match_index"], dtype=np.int64)
    supported = np.array(rows["supported"], dtype=bool)
    keep = (np.array(matches["role"])[mi] == role) & supported
    q = np.array(mq["first_query"], dtype=np.int64)[mi[keep]] + np.array(rows["query_ref"], dtype=np.int64)[keep]
    order = np.argsort(q, kind="stable")
    q, m, v_row = q[order], mi[keep][order], np.flatnonzero(keep)[order].astype(np.int64)
    if np.any(np.diff(q) <= 0):
        raise ValueError("nominal query indices are not unique")
    out = {"query_index": q, "match_index": m, "v_row": v_row, "n_v_rows": int(len(mi)), "n_supported": int(supported.sum()),
           "n_role_rows": int((np.array(matches["role"])[mi] == role).sum())}
    del rows, matches, mq
    return out


# ----------------------------------------------------------------------------- stage times (new file)
def record_time(row: dict) -> None:
    STAGE_TIMES.parent.mkdir(parents=True, exist_ok=True)
    with STAGE_TIMES.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Timer:
    """e2bc_common.Timer semantics, writing to E2v_NOMINAL_STAGE_TIMES.jsonl."""

    def __init__(self, dataset: str, stage: str, log, **extra):
        self.dataset, self.stage, self.log, self.extra = dataset, stage, log, extra

    def __enter__(self):
        self.started, self.t0 = C.utc_now(), time.perf_counter()
        return self

    def __exit__(self, kind, value, tb):
        seconds = round(time.perf_counter() - self.t0, 1)
        status = "ok" if kind is None else "failed"
        record_time({"dataset": self.dataset, "stage": self.stage, "started_utc": self.started, "completed_utc": C.utc_now(),
                     "seconds": seconds, "status": status, "error": None if kind is None else f"{kind.__name__}: {value}"[:500],
                     "argv": sys.argv, **self.extra})
        self.log(f"[time] {self.dataset}/{self.stage}: {status} {seconds} s")
        return False


# ----------------------------------------------------------------------------- N1 query list
def query_list_path(dataset: str) -> Path:
    return B.labels_dir(dataset) / f"QUERIES_{TAG}.npy"


def write_query_list(dataset: str):
    import numpy as np

    nominal = nominal_rows(C.stage_dir(dataset, "projection"), C.EXTERNAL_ROLE_CODE)
    path = query_list_path(dataset)
    if path.is_file():
        stored = np.load(path, allow_pickle=False)
        if not np.array_equal(stored, nominal["query_index"]):
            raise ValueError(f"{dataset}: stored nominal query list differs from the projection")
        return nominal, path
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.save(stream, nominal["query_index"], allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return nominal, path


def read_query_list(dataset: str):
    import numpy as np

    path = query_list_path(dataset)
    return np.load(path, allow_pickle=False), path


# ----------------------------------------------------------------------------- N2 scoring
def _init_worker(dataset: str, device: str) -> None:
    from studies.external_validation_20261009 import e2b_labels

    e2b_labels._init_scorer(dataset, device)  # E2b scorer: LockedV(run_3, 'final'), ExternalQuerySurface, final_REFIT_TRAIN bundle


def _score_part(part: int) -> dict:
    """score.score_parts / v_test_tau.score_queries body for one part, with the E2b final-batch padding."""
    import numpy as np
    from studies.external_validation_20261009 import e2b_labels
    from studies.label_production_20261005 import queries, score

    worker = e2b_labels._WORKER
    dataset, v, surface, bundle = worker["dataset"], worker["v"], worker["surface"], worker["bundle"]
    if queries.BATCH_ROWS != BATCH_ROWS or score.BATCHES_PER_PART * queries.BATCH_ROWS != PART_ROWS:
        raise ValueError("frozen batch constants differ from this stage's")
    root = B.labels_dir(dataset)
    indices, list_path = read_query_list(dataset)
    start, stop = score.part_ranges(len(indices))[part]
    chunk = indices[start:stop]
    target = score.part_path(root, TAG, part)
    if score.part_complete(target, chunk):
        return {"part": part, "skipped": True}
    t0 = time.perf_counter()
    logits, raw, calibrated, padded_rows = [], [], [], 0
    for b_start in range(0, len(chunk), queries.BATCH_ROWS):
        batch = chunk[b_start:b_start + queries.BATCH_ROWS]
        rows, seen = [], []
        for query_index, row, _meta in surface.iter_query_rows(batch):
            rows.append(row)
            seen.append(query_index)
        if not np.array_equal(np.asarray(seen, dtype=np.int64), batch):
            raise ValueError(f"{dataset}: query rows came back out of order in part {part}")
        rows, n_real = pad_batch(rows, queries.BATCH_ROWS)
        padded_rows += queries.BATCH_ROWS - n_real
        values = v.score(queries.transform(rows, bundle))
        for store, value in zip((logits, raw, calibrated), values):
            store.append(np.asarray(value)[:n_real])
    payload = {"query_index": chunk, "logit": np.concatenate(logits), "raw_p": np.concatenate(raw), "v": np.concatenate(calibrated)}
    for name in ("logit", "raw_p", "v"):
        if len(payload[name]) != len(chunk) or not np.isfinite(payload[name]).all():
            raise ValueError(f"{dataset}: bad {name} in part {part}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, target)
    receipt = {"version": VERSION, "frozen_scoring_version": score.VERSION, "context": B.V_CONTEXT, "tag": TAG, "part": part,
               "rows": int(len(chunk)), "range": [int(start), int(stop)], "sha256": score.sha256_file(target),
               "queries_sha256": score.sha256_file(list_path), "v": v.identity, "code_sha256": scoring_code_hashes(),
               "batch_rows": queries.BATCH_ROWS, "padded_rows": int(padded_rows), "utc": C.utc_now(), "dataset": dataset,
               "surface": "e2bc_common.ExternalQuerySurface"}
    pending_receipt = target.with_suffix(".json.pending")
    pending_receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(pending_receipt, target.with_suffix(".json"))
    return {"part": part, "skipped": False, "rows": int(len(chunk)), "seconds": round(time.perf_counter() - t0, 1)}


def scoring_code_hashes() -> dict:
    from studies.label_production_20261005 import score

    code = score.code_hashes()
    code.update(B.code_hashes([Path(__file__), C.PACKAGE / "e2b_labels.py", C.PACKAGE / "e2bc_common.py", C.PACKAGE / "e2a_common.py"]))
    return code


def run_scoring(dataset: str, workers: int, device: str, log) -> dict:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from studies.label_production_20261005 import score

    nominal, _ = write_query_list(dataset)
    indices = nominal["query_index"]
    root = B.labels_dir(dataset)
    ranges = score.part_ranges(len(indices))
    todo = [p for p, (a, b) in enumerate(ranges) if not score.part_complete(score.part_path(root, TAG, p), indices[a:b])]
    n_workers = max(1, min(workers_allowed(workers), len(todo) or 1))
    log(f"[e2v] {dataset}: {len(indices)} nominal queries ({nominal['n_v_rows']} V rows, {nominal['n_supported']} supported) "
        f"in {len(ranges)} parts ({len(todo)} to score) with {n_workers} worker(s) on {device}")
    if todo:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context, initializer=_init_worker, initargs=(dataset, device)) as pool:
            done = 0
            for future in as_completed([pool.submit(_score_part, p) for p in todo]):
                result = future.result()
                done += 1
                log(f"[e2v] {dataset}: part {result['part']} done ({done}/{len(todo)}) {result.get('seconds')} s")
    return {"rows": int(len(indices)), "parts": len(ranges), "workers": n_workers}


# ----------------------------------------------------------------------------- N3 merge
def pred_path(dataset: str) -> Path:
    return B.labels_dir(dataset) / f"PRED_{TAG}.npz"


def manifest_path(dataset: str) -> Path:
    return B.labels_dir(dataset) / f"PRED_{TAG}_MANIFEST.json"


def _patch_of(dataset: str) -> dict:
    patch_of = {}
    with (C.stage_dir(dataset, "bank") / "BANK_ENTRIES.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            patch_of[entry["canonical_match_key"].encode("ascii")] = entry["patch"].encode("ascii")
    return patch_of


def identity_columns(dataset: str, nominal: dict) -> dict:
    """Public nominal identity of each query; MATCH_TABLE is read for key and role only (never the winner field)."""
    import numpy as np

    proj = C.stage_dir(dataset, "projection")
    rows = np.load(proj / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    matches = np.load(proj / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    qt = np.load(proj / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
    age = np.load(proj / "FRAME_AGE_SECONDS.npy", mmap_mode="r", allow_pickle=False)
    q, m, r = nominal["query_index"], nominal["match_index"], nominal["v_row"]
    selected = np.array(rows[r])
    if not np.array_equal(selected["match_index"].astype(np.int64), m):
        raise ValueError(f"{dataset}: V_ROWS match index differs from the query list")
    if not np.array_equal(np.array(qt["match_index"], dtype=np.int64)[q], m):
        raise ValueError(f"{dataset}: QUERY_TABLE match index differs from the nominal row")
    role = np.array(matches["role"])[m]
    if np.any(role != C.EXTERNAL_ROLE_CODE):
        raise ValueError(f"{dataset}: a nominal row is not EXTERNAL")
    keys = np.array(matches["key"])[m]
    patch_of = _patch_of(dataset)
    out = {"canonical_match_key": keys, "patch": np.asarray([patch_of[bytes(k)] for k in keys], dtype="S8"),
           "match_index": m.astype(np.int64), "v_row": r.astype(np.int64)}
    for name in IDENTITY_FIELDS:
        out[name] = np.array(selected[name])
    out["query_ms"] = np.array(qt["query_ms"])[q]
    out["frame_age_seconds"] = np.asarray(np.array(age)[q], dtype=np.float64)
    del rows, matches, qt, age
    return out


def merge(dataset: str, log) -> dict:
    import numpy as np
    from studies.label_production_20261005 import queries, score

    target, mpath = pred_path(dataset), manifest_path(dataset)
    if target.is_file() and mpath.is_file():
        record = C.read_json(mpath)
        if record.get("pred_sha256") == B.sha256_file(target):
            log(f"[e2v] {dataset}: PRED_{TAG} exists, reused")
            return record
    nominal, list_path = write_query_list(dataset)
    indices = nominal["query_index"]
    root = B.labels_dir(dataset)
    ranges = score.part_ranges(len(indices))
    pieces, identities, parts, padded = [], set(), {}, 0
    for part, (start, stop) in enumerate(ranges):
        path = score.part_path(root, TAG, part)
        if not score.part_complete(path, indices[start:stop]):
            raise ValueError(f"{dataset}: part {part} is missing or does not verify")
        receipt = score.read_json(path.with_suffix(".json"))
        identities.add(json.dumps({"v": receipt["v"], "code": receipt["code_sha256"]}, sort_keys=True))
        padded += int(receipt["padded_rows"])
        parts[path.name] = {"sha256": receipt["sha256"], "rows": receipt["rows"], "receipt_sha256": B.sha256_file(path.with_suffix(".json"))}
        with np.load(path, allow_pickle=False) as stored:
            pieces.append({name: stored[name] for name in ("query_index", *VALUE_FIELDS)})
    if len(identities) != 1:
        raise ValueError(f"{dataset}: parts were scored by different V identities or code")
    payload = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}
    if not np.array_equal(payload["query_index"], indices):
        raise ValueError(f"{dataset}: merged query order differs from the query list")
    payload.update(identity_columns(dataset, nominal))
    if WINNER_FIELD in payload:
        raise ValueError("the match winner must not be written")
    sha = B.write_npz(target, payload)
    identity = json.loads(identities.pop())
    proj = C.stage_dir(dataset, "projection")
    patches = {p.decode(): int((payload["patch"] == p).sum()) for p in sorted(set(payload["patch"].tolist()))}
    record = {"version": VERSION, "kind": "external_nominal_v_predictions", "dataset": dataset, "context": B.V_CONTEXT,
              "rows": int(len(indices)), "matches": int(len(np.unique(payload["match_index"]))), "rows_by_patch": patches,
              "v_rows_total": nominal["n_v_rows"], "v_rows_supported": nominal["n_supported"], "v_rows_external_role": nominal["n_role_rows"],
              "parts": parts, "padded_rows_total": padded, "batch_rows": queries.BATCH_ROWS,
              "queries_path": str(list_path), "queries_sha256": B.sha256_file(list_path),
              "pred_path": str(target), "pred_sha256": sha, "columns": sorted(payload),
              "v": identity["v"], "code_sha256": identity["code"],
              "projection_tables_sha256": {name: B.sha256_file(proj / name) for name in PROJECTION_TABLES},
              "bank_entries_sha256": B.sha256_file(C.stage_dir(dataset, "bank") / "BANK_ENTRIES.jsonl"),
              "rule": "v_test_tau.nominal_queries (role = EXTERNAL code 6) scored as v_test_tau.score_queries / score.score_parts with "
                      "ExternalQuerySurface; final partial batch padded to 1,024 rows with copies of its last row (padded outputs dropped)",
              "match_winner_written": False, "match_winner_read": False, "metrics_computed": False, "utc": C.utc_now()}
    C.write_json(mpath, record)
    log(f"[e2v] {dataset}: PRED_{TAG}.npz {record['rows']} rows / {record['matches']} matches {patches} sha {sha[:16]}")
    return record


def run_dataset(dataset: str, workers: int, device: str, log) -> dict:
    with Timer(dataset, "e2v_nominal_scoring", log, device=device):
        scoring = run_scoring(dataset, workers, device, log)
    with Timer(dataset, "e2v_nominal_merge", log):
        record = merge(dataset, log)
    return {"scoring": scoring, "rows": record["rows"], "matches": record["matches"], "pred_sha256": record["pred_sha256"]}


# ----------------------------------------------------------------------------- parity against V_TEST_PRED_final
def _reference_parts(n_rows: int) -> tuple[dict, dict]:
    import numpy as np

    n_parts = (n_rows + PART_ROWS - 1) // PART_ROWS
    arrays = {name: [] for name in ("query_index", *VALUE_FIELDS)}
    receipts = {}
    for part in range(n_parts):
        path = REFERENCE_DIR / f"part_{part:05d}.npz"
        receipt = C.read_json(path.with_suffix(".json"))
        sha = B.sha256_file(path)
        if receipt.get("sha256") != sha or receipt.get("part") != part:
            raise ValueError(f"reference part {part} does not match its receipt")
        receipts[path.name] = {"sha256": sha, "range": receipt["range"], "v": receipt["v"]}
        with np.load(path, allow_pickle=False) as stored:
            for name in arrays:
                arrays[name].append(np.asarray(stored[name]))
    return {name: np.concatenate(values) for name, values in arrays.items()}, receipts


def parity_nominal(log) -> dict:
    import numpy as np

    dataset = C.DATASET_PARITY
    mine = dict(np.load(pred_path(dataset), allow_pickle=False))
    own = C.read_json(manifest_path(dataset))
    ref_list = nominal_rows(MAIN_PROJECTION, REFERENCE_ROLE)
    ref, receipts = _reference_parts(len(ref_list["query_index"]))
    report = {"version": VERSION, "kind": "E2v_NOMINAL_PARITY_REPORT", "created_utc": C.utc_now(), "dataset": dataset,
              "reference": {"dir": str(REFERENCE_DIR), "script": str(REFERENCE_SCRIPT), "script_sha256": B.sha256_file(REFERENCE_SCRIPT),
                            "projection": str(MAIN_PROJECTION), "parts": receipts, "rows": int(len(ref_list["query_index"]))},
              "tolerance": TOL, "rule": "bitwise equal or within 1e-6 in logit, raw p and calibrated p", "checks": {}}
    checks = report["checks"]
    checks["reference_covers_test_nominal_list"] = bool(np.array_equal(ref["query_index"], ref_list["query_index"]))
    ref_v = {json.dumps({k: r["v"][k] for k in V_IDENTITY_FIELDS}, sort_keys=True) for r in receipts.values()}
    own_v = json.dumps({k: own["v"][k] for k in V_IDENTITY_FIELDS}, sort_keys=True)
    checks["v_identity_equal"] = len(ref_v) == 1 and own_v in ref_v
    # join: parity row -> main match (canonical key) -> same nominal position -> main query -> reference list position
    main_rows = np.load(MAIN_PROJECTION / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    main_matches = np.load(MAIN_PROJECTION / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    main_mq = np.load(MAIN_PROJECTION / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)
    par_rows = np.load(C.stage_dir(dataset, "projection") / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    par_matches = np.load(C.stage_dir(dataset, "projection") / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    main_index = {bytes(k): i for i, k in enumerate(np.array(main_matches["key"]))}
    keys = mine["canonical_match_key"]
    missing = [bytes(k).decode("ascii") for k in np.unique(keys) if bytes(k) not in main_index]
    checks["every_match_in_main_projection"] = not missing
    if missing:
        report["status"] = "FAIL"
        report["missing_matches"] = missing[:10]
        C.write_json(PARITY_REPORT, report)
        return report
    m_main = np.asarray([main_index[bytes(k)] for k in keys], dtype=np.int64)
    first_par = np.array(par_matches["first_nominal_row"], dtype=np.int64)[mine["match_index"]]
    local = mine["v_row"] - first_par
    first_main = np.array(main_matches["first_nominal_row"], dtype=np.int64)[m_main]
    n_main = np.array(main_matches["n_nominal_rows"], dtype=np.int64)[m_main]
    checks["nominal_position_in_range"] = bool(np.all((local >= 0) & (local < n_main)))
    main_v_row = first_main + local
    a, b = np.array(par_rows[mine["v_row"]]), np.array(main_rows[main_v_row])
    field_equal = {name: bool(np.array_equal(a[name], b[name])) for name in PARITY_FIELDS}
    checks["nominal_identity_fields_exact"] = field_equal
    checks["main_rows_are_test"] = bool(np.all(np.array(main_matches["role"])[m_main] == REFERENCE_ROLE))
    q_main = np.array(main_mq["first_query"], dtype=np.int64)[m_main] + b["query_ref"].astype(np.int64)
    pos = np.searchsorted(ref_list["query_index"], q_main)
    found = (pos < len(ref_list["query_index"])) & (ref_list["query_index"][np.minimum(pos, len(ref_list["query_index"]) - 1)] == q_main)
    checks["every_row_in_reference"] = bool(found.all())
    subset_main = np.unique(m_main)
    ref_rows_for_subset = int(np.isin(ref_list["match_index"], subset_main).sum())
    checks["reference_rows_for_subset_equal_mine"] = ref_rows_for_subset == len(keys)
    report["rows"] = {"mine": int(len(keys)), "reference_for_the_500_matches": ref_rows_for_subset, "matches": int(len(subset_main))}
    del main_rows, main_matches, main_mq, par_rows, par_matches
    ok = all(v for k, v in checks.items() if k != "nominal_identity_fields_exact") and all(field_equal.values())
    if found.all():
        sizes = np.asarray([batch_size_at(int(p), len(ref_list["query_index"])) for p in pos], dtype=np.int64)
        groups = {"all": np.ones(len(pos), dtype=bool), "reference_full_batch": sizes == BATCH_ROWS, "reference_partial_batch": sizes < BATCH_ROWS}
        own_partial = np.zeros(len(pos), dtype=bool)
        own_partial[(len(pos) // BATCH_ROWS) * BATCH_ROWS:] = len(pos) % BATCH_ROWS != 0
        groups["own_padded_final_batch"] = own_partial
        values = {}
        for group, mask in groups.items():
            entry = {"rows": int(mask.sum())}
            for name in VALUE_FIELDS:
                x, y = np.asarray(mine[name], dtype=np.float64)[mask], np.asarray(ref[name], dtype=np.float64)[pos][mask]
                entry[name] = {"bitwise_equal_rows": int(np.sum(x.view(np.uint64) == y.view(np.uint64))),
                               "max_abs_diff": float(np.abs(x - y).max()) if mask.any() else 0.0,
                               "finite": bool(np.isfinite(x).all() and np.isfinite(y).all())}
            values[group] = entry
        report["values"] = values
        report["reference_partial_batch_sizes"] = sorted({int(s) for s in sizes[sizes < BATCH_ROWS]})
        within = {name: values["all"][name]["max_abs_diff"] <= TOL and values["all"][name]["finite"] for name in VALUE_FIELDS}
        checks["values_within_tolerance"] = within
        ok = ok and all(within.values())
    else:
        ok = False
    report["own_manifest_sha256"] = B.sha256_file(manifest_path(dataset))
    report["own_pred_sha256"] = B.sha256_file(pred_path(dataset))
    report["status"] = "PASS" if ok else "FAIL"
    C.write_json(PARITY_REPORT, report)
    log(f"[e2v] nominal parity {report['status']}: " + json.dumps({g: {n: (e[n]['bitwise_equal_rows'], e[n]['max_abs_diff']) for n in VALUE_FIELDS} | {'rows': e['rows']}
                                                                    for g, e in report.get("values", {}).items()}))
    return report


# ----------------------------------------------------------------------------- addendum
def code_files() -> list[Path]:
    return [Path(__file__).resolve(), (C.PACKAGE / "tests" / "test_e2v_nominal.py").resolve()]


def frozen_imported_files() -> list[Path]:
    from studies.label_production_20261005 import score

    files = [C.ROOT / rel for rel in score.code_hashes()]
    files += [C.PACKAGE / name for name in ("e2b_labels.py", "e2bc_common.py", "e2a_common.py", "e2a_na1.py")]
    files += [REFERENCE_SCRIPT, C.ROOT / "reports/label_production_20261005/run_labels.py"]
    return files


def write_addendum(log) -> dict:
    t0, started = time.perf_counter(), C.utc_now()
    if ADDENDUM.exists():
        raise FileExistsError(f"{ADDENDUM} exists; it is written once")
    lock_sha = B.sha256_file(B.SCORE_LOCK_OUT)
    if lock_sha != LOCK_SHA256:
        raise ValueError(f"EXTERNAL_SCORE_LOCK.json sha256 {lock_sha} differs from the coordinator's {LOCK_SHA256}")
    lock = C.read_json(B.SCORE_LOCK_OUT)
    parity = C.read_json(PARITY_REPORT)
    if parity.get("status") != "PASS":
        raise SystemExit("BLOCKED: the nominal parity gate has not passed")
    unchanged = {rel: B.sha256_file(C.ROOT / rel) == digest for group in ("external", "frozen_imported") for rel, digest in lock["code"][group].items()}
    if not all(unchanged.values()):
        raise ValueError(f"lock-bound code changed: {[rel for rel, ok in unchanged.items() if not ok]}")
    datasets = {}
    for dataset in B.EXTERNAL_DATASETS:
        record = C.read_json(manifest_path(dataset))
        if record["pred_sha256"] != B.sha256_file(pred_path(dataset)):
            raise ValueError(f"{dataset}: PRED_{TAG}.npz differs from its manifest")
        datasets[dataset] = {"pred_nominal_final": {"path": str(pred_path(dataset)), "sha256": record["pred_sha256"]},
                             "manifest": {"path": str(manifest_path(dataset)), "sha256": B.sha256_file(manifest_path(dataset))},
                             "queries": {"path": record["queries_path"], "sha256": B.sha256_file(Path(record["queries_path"]))},
                             "rows": record["rows"], "matches": record["matches"], "rows_by_patch": record["rows_by_patch"],
                             "parts": len(record["parts"]), "padded_rows_total": record["padded_rows_total"],
                             "projection_tables_sha256": record["projection_tables_sha256"]}
    payload = {"kind": "EXTERNAL_SCORE_LOCK_ADDENDUM", "version": VERSION, "created_utc": C.utc_now(),
               "purpose": "protocol metrics.v: final V on the nominal V rows (E2b scored only the engagement pre/post states)",
               "lock": {"path": str(B.SCORE_LOCK_OUT), "sha256": lock_sha},
               "lock_check": {"path": str(LOCK_CHECK), "sha256": B.sha256_file(LOCK_CHECK), "passed": C.read_json(LOCK_CHECK).get("passed")},
               "lock_bound_code_unchanged": True,
               "parity": {"path": str(PARITY_REPORT), "sha256": B.sha256_file(PARITY_REPORT), "status": parity["status"],
                          "values_all": parity.get("values", {}).get("all")},
               "reference": {"dir": str(REFERENCE_DIR), "parts_sha256": {k: v["sha256"] for k, v in parity["reference"]["parts"].items()}},
               "v": C.read_json(manifest_path(C.DATASET_PARITY))["v"],
               "datasets": datasets,
               "code": {"new": B.code_hashes(code_files()), "changed": {},
                        "frozen_imported": B.code_hashes(frozen_imported_files())},
               "sealed_files_opened": False, "label_values_read": False, "match_winner_read": False, "metrics_computed": False}
    # Stage-time line first, then the hash of the stage-time file (no line is appended after the addendum).
    record_time({"dataset": "all", "stage": "external_score_lock_addendum", "started_utc": started, "completed_utc": C.utc_now(),
                 "seconds": round(time.perf_counter() - t0, 1), "status": "ok", "error": None, "argv": sys.argv})
    lines = STAGE_TIMES.read_text(encoding="utf-8").splitlines()
    payload["stage_times"] = {"path": str(STAGE_TIMES), "sha256": B.sha256_file(STAGE_TIMES), "lines": len(lines),
                              "last_line_is_this_addendum": json.loads(lines[-1])["stage"] == "external_score_lock_addendum",
                              "rows": [json.loads(line) for line in lines]}
    B.write_json_once(ADDENDUM, payload)
    log(f"[e2v] addendum written sha {B.sha256_file(ADDENDUM)}")
    return payload


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, choices=[d for d in B.EXTERNAL_DATASETS if d != C.DATASET_PARITY])
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--device", default="cuda")
    sub.add_parser("addendum")
    args = parser.parse_args(argv)
    B.env_runtime()
    if args.command == "parity":
        log = C.Log(C.LOGS / "E2v_nominal_parity.log")
        run_dataset(C.DATASET_PARITY, args.workers, args.device, log)
        with Timer(C.DATASET_PARITY, "e2v_nominal_parity_compare", log):
            report = parity_nominal(log)
        return 0 if report["status"] == "PASS" else 2
    if args.command == "run":
        if not PARITY_REPORT.exists() or C.read_json(PARITY_REPORT).get("status") != "PASS":
            raise SystemExit("BLOCKED: the nominal parity gate has not passed; external sets are not run")
        log = C.Log(C.LOGS / f"E2v_nominal_{args.dataset}.log")
        log(f"[run] e2v nominal {args.dataset} workers={args.workers} device={args.device}")
        run_dataset(args.dataset, args.workers, args.device, log)
        return 0
    log = C.Log(C.LOGS / "E2v_addendum.log")
    write_addendum(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
