"""E2a common definitions: datasets, the EXTERNAL role, the on-disk layout and runtime hygiene.

README (E2a, 2026-10-09)
  E2a turns per-match caches (2025 cache format) into the three upstream artifacts that the label and q stages
  consume, using the study's own frozen code:
    1. observable-state bank with the static-preservation repair  (e2a_bank: audit -> bank_raw -> bank)
    2. V input projection with its inputs: V reference preparation (MATCH_TABLE / V_ROWS / CONTEXT_MEMBERSHIP),
       recipient-confirmed item-node bank, raw projection (QUERY_TABLE / feature arrays)
       (e2a_preparation, e2a_items, e2a_projection)
    3. shared history store (e2a_history)
  for an explicit EXTERNAL role, never the TEST code 5.  The parity gate (e2a_parity) re-runs the adapted pipeline on
  500 accepted 15.16 TEST matches and compares every artifact with the accepted 2025 releases.

  No file of the workspace outside this package and reports/external_validation_20261009/E2a_* is written; frozen
  modules are imported, never edited.  Where a frozen function hard-codes the 2025 cache path, the 15.14-15.16 lookup
  tables or the 2025 acceptance certificates, the adapters below pass the dataset's path or replace only that gate;
  each such hook is listed in FROZEN_HOOKS and in E2a_REPORT.json.

Layout (DATA_ROOT = D:/LOL_External_20261009):
  parity_15_16:   DATA_ROOT/parity_15_16/<stage>
  kr_15_18_22 / kr_16_13:  DATA_ROOT/<stage>/<dataset>
  stages: audit, bank_raw, bank, preparation, items, projection, history

Role coding for EXTERNAL rows (codes beyond the frozen tuples, so no frozen validator can read them as a 2025 role):
  observable ledger / item MATCH_TABLE role string  "EXTERNAL"
  preparation MATCH_TABLE.patch_role                3   (frozen PATCH_ROLES = TRAIN 0, VALIDATION 1, TEST 2)
  preparation MATCH_TABLE.role                      6   (frozen ROLES = TRAIN 0, V_DEV 1, V_CAL 2, Q_DEV 3, Q_CAL 4, TEST 5)
  CONTEXT_MEMBERSHIP                                0 in all six contexts (never FIT/STOP/REFIT/DEV/CAL/TARGET)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "external_validation_20261009"
LOGS = REPORTS / "logs"
DATA_ROOT = Path(r"D:\LOL_External_20261009")

VERSION = "external_validation_e2a_20261009"
PARITY_TAG = "parity_20261009:"
PARITY_N = 500
PARITY_PATCH = "15.16"

DATASET_PARITY = "parity_15_16"
DATASET_2025 = "kr_15_18_22"
DATASET_2026 = "kr_16_13"
DATASETS = (DATASET_PARITY, DATASET_2025, DATASET_2026)

REF_CACHE_2025 = Path(r"D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13")
REF_PATCH_INDEX_2025 = Path(r"D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13_patch_index.json")
Q_ROWS_H90 = Path(r"D:\LOL_Training_20261004\q_run_1\Q_ROWS_h90.npz")
Q_TEST_ROLE_CODE = 5

CACHE = {
    DATASET_PARITY: REF_CACHE_2025,
    DATASET_2025: DATA_ROOT / "cache_kr_15_18_22",
    DATASET_2026: ROOT / "outputs" / "full_corpus_training_20260915" / "external" / "KR_16.13" / "cache",
}
PATCH_INDEX = {
    DATASET_PARITY: REF_PATCH_INDEX_2025,  # restricted to E2a_PARITY_SUBSET.json, original 2025 positions kept
    DATASET_2025: DATA_ROOT / "match_cache_kr_15_18_22_patch_index.json",
    DATASET_2026: DATA_ROOT / "match_cache_kr_16_13_patch_index.json",
}
EXPECTED_PATCHES = {
    DATASET_PARITY: ("15.16",),
    DATASET_2025: ("15.18", "15.19", "15.20", "15.21", "15.22"),
    DATASET_2026: ("16.13",),
}

PARITY_SUBSET = REPORTS / "E2a_PARITY_SUBSET.json"
PARITY_REPORT = REPORTS / "E2a_PARITY_REPORT.json"
E2A_REPORT = REPORTS / "E2a_REPORT.json"

EXTERNAL_ROLE = "EXTERNAL"
EXTERNAL_PATCH_ROLE_CODE = 3
EXTERNAL_ROLE_CODE = 6

STAGES = ("audit", "bank_raw", "bank", "preparation", "items", "projection", "history")
MAX_WORKERS = 6
SHARD_MATCHES = 512           # observable bank shard size of the accepted run (run.py default)
HISTORY_SHARD_MATCHES = 256   # shared-history shard size of the accepted run

# Accepted 2025 artifacts used only as read-only parity references.
ACCEPTED = {
    "audit_ledger": ROOT / "outputs/cache_source_audit_20261002/run_full_20261002T055000Z/row_ledger.jsonl",
    "bank": ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1",
    "bank_parent": ROOT / "outputs/observable_state_20261002/full_20261002_1",
    "preparation": ROOT / "outputs/v_preparation_20261002/full_nominal_refs_20261002_1",
    "items": ROOT / "outputs/item_node_observed_20261002/full_bank_v3_20261002_1",
    "projection": ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1",
    "history": ROOT / "outputs/shared_history_eight_cpu_20261003/full_1",
}

# Every place where a frozen function is adapted instead of called with its own defaults.
FROZEN_HOOKS = {
    "cache_source_audit": "studies.cache_source_audit_20261002.audit.audit_match_files(match_id, patch, dataset cache) "
                          "unchanged; its row role (config.PATCH_ROLES.get(patch,'UNKNOWN')) and canonical_key are "
                          "relabelled to EXTERNAL before run_audit._ledger_record",
    "bank_source_read": "observable_state_20261002.producer.read_cached_triplet rebound in the worker to "
                        "source.read_cached_triplet with cache_dir=<dataset cache> (frozen default = 2025 cache)",
    "bank_static_lookup": "producer.static_lookup_counts / repair.static_lookup_counts rebound to "
                          "e2a_bank.external_static_lookup_counts: frozen categorical.static_lookup_counts for "
                          "15.14-15.16; for later patches the same status vocabulary with local Data Dragon champion "
                          "and item tables and 'recorded_ID_map_unavailable' for summoner/rune tables (none offline). "
                          "Ledger diagnostic only, never a bank array",
    "bank_shard_loop": "e2a_bank.produce_external_shard = producer.produce_shard with per-match exclusion "
                       "(frozen produce_match per entry; a raising match is excluded and logged instead of failing "
                       "the shard) and an added ledger key 'dataset'",
    "static_repair_cache": "observable_static_preservation_20261002.repair.repair_shard called unchanged with "
                           "cache_source_audit_20261002.config.CACHE_DIR set to the dataset cache in the worker",
    "preparation": "v_preparation_20261002.run.run transcribed with the EXTERNAL codes (patch_role 3, role 6, fold -1, "
                   "stop False, context masks 0); frozen nominal_reference_rows and verify_nominal_sampler unchanged",
    "items": "item_node_observed_20261002.audit.run_bank transcribed without the 2025 acceptance gates, CPU pinning "
             "and vocabulary fitting; frozen AcceptedSourceReader, PreparationIndex, observe_match, "
             "_compact_for_writer, _BankWriter unchanged",
    "projection": "v_input_projection_20261002.run.build transcribed without the 2025 acceptance gates and with "
                  "role names extended by EXTERNAL; frozen project_match, load_source_shard, local_source unchanged",
    "history": "reports/shared_history_eight_cpu_20261003/run.py scheduling replaced by a <=6-process pool without CPU "
               "pinning; per shard the frozen iter_match_views, semantic_row/query_view reference equality, "
               "store.write_shard, ShardReader reload and semantic re-check are unchanged",
}


def stage_dir(dataset: str, stage: str) -> Path:
    if dataset not in DATASETS or stage not in STAGES:
        raise ValueError(f"unknown dataset/stage {dataset}/{stage}")
    if dataset == DATASET_PARITY:
        return DATA_ROOT / DATASET_PARITY / stage
    return DATA_ROOT / stage / dataset


def canonical_key(match_id: str) -> str:
    """Same as observable_state_20261002.source.canonical_key."""
    return hashlib.sha256(str(match_id).strip().encode("utf-8")).hexdigest()


def parity_hash(match_id: str) -> str:
    return hashlib.sha256((PARITY_TAG + str(match_id)).encode("utf-8")).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_bytes(path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    else:
        raise PermissionError(f"could not replace {path}")
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj) -> str:
    return atomic_write_bytes(path, (json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def runtime_hygiene() -> dict:
    """One numeric thread, no GPU, no bytecode, below-normal priority (call before importing numpy-heavy code)."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    below = False
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        below = True
    except Exception:
        below = False
    return {"numeric_threads": 1, "gpu_visible": False, "below_normal_priority": below}


class Log:
    """Append-only UTC log, flushed per line, mirrored to stdout."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{utc_now()} {msg}"
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def dir_bytes(path) -> tuple[int, int]:
    """(files, bytes) under path, recursive, via os.scandir."""
    files = total = 0
    stack = [str(path)]
    while stack:
        top = stack.pop()
        try:
            it = os.scandir(top)
        except OSError:
            continue
        with it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    total += entry.stat().st_size
    return files, total


def code_hashes(paths) -> dict:
    out = {}
    for path in sorted({Path(p).resolve() for p in paths if Path(p).is_file()}):
        try:
            name = str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            name = str(path)
        out[name] = sha256_file(path)
    return out


def package_code_hashes() -> dict:
    return code_hashes(list(PACKAGE.glob("*.py")) + list((PACKAGE / "tests").glob("*.py")))


def frozen_code_hashes() -> dict:
    """Hashes of every frozen module folder the E2a adapters import (read-only)."""
    folders = ("observable_state_20261002", "observable_static_preservation_20261002", "cache_source_audit_20261002",
               "anchor_frame_audit_20261002", "v_preparation_20261002", "item_node_observed_20261002",
               "item_node_encoding_20261002", "v_input_projection_20261002", "v_shared_history_20261003",
               "v_shared_history_production_20261003", "v_model_input_20261003", "v_model_fit_input_20261003")
    paths = []
    for folder in folders:
        paths.extend((ROOT / "studies" / folder).glob("*.py"))
    paths += [ROOT / "reports/shared_history_eight_cpu_20261003/run.py",
              ROOT / "reports/shared_history_production_20261003/run.py",
              ROOT / "reports/v_stream_preparation_20261003/history_probe.py",
              ROOT / "worktrees/engagement-state-value/studies/st_dbscan_20261001/kernel.py",
              ROOT / "reports/input_population_specification_20261002/INPUT_POLICY.json"]
    return code_hashes(paths)
