"""Post-hoc kill-advantage (KA) reinforcement analyses K1-K6 for the ToG paper (PLAN_KA_POSTHOC_KO.md, commit 219605b).

Status: post-hoc analysis on previously opened labels.  The KA TEST labels (q_ka_1/KA_LABELS.npz, not sealed) were first
evaluated on 2026-10-09T11:30:47Z (EXPOSURE_KO.md row "15.16 킬 우위 라벨"); no KA version of K1-K6 was fixed before
that.  The definitions are the thesis scripts' (plan section 2 table); the only new choices are the plan's
"unavoidable differences" (no P1-best cell on KA, ties dropped, KA-retrained P1 and PT_flex, KA PT_flex offsets).

  K1 kill-count strata 1 / 2-3 / 4+ and the positive-gain one-kill share   scale_strata.py (per-stratum bootstrap)
  K2 participant cohorts T / S / ASYM / P / T5 / all                       r8_evaluate.py, rcommon.py (CellBoot)
  K3 frame-age trends over 1-s bins                                         frame_age_seconds.py analyse (shared draws)
  K4 Brier reduction and skill against the TRAIN constant cell              ledger derived rows (points only)
  K5 nested LightGBM on the KA PT_flex logit (refit, cells/P1_NEST)         r2_retrain.py P1_NEST (CellBoot)
  K6 two more P1 seeds 20261002, 20261003 (refits)                          seed_replicates.py, n2_seed_aggregation.py

Sign: Brier(LightGBM cell) - Brier(PT_flex); negative = LightGBM better.  Intervals: paired match bootstrap, 2,000
draws, PCG64(20261011), 95 % percentile (numpy.quantile linear), per stratum or shared over the split as the script.
Rows: the non-tie (eligible) rows of each split; match weights 1/m are recomputed inside every subset (D-Q13).

Subcommands (outputs below D:/LOL_Training_20261004/tog_ka_1, every file written once):
  launch [--inputs-only]  REFIT_INPUTS.json (once, before any fit), then the four LightGBM refits one at a time
                          (one slot; LightGBM 3 threads + BLAS 1; no GPU): P1_s20261001_check (reproduction gate),
                          P1_s20261002, P1_s20261003, P1_NEST.  Re-running it skips completed fits, adopts a live
                          worker and resumes an interrupted one (outputs of the interrupted attempt are moved aside).
  fit --cell NAME         one refit worker (rechecks REFIT_INPUTS.json first)
  qdev                    gate (KA_REPORT D.kk within 1e-9), then K1-K6 on Q_DEV -> KA_POSTHOC_QDEV.json
  lock [--new-version]    light lock KA_POSTHOC_LOCK.json (next versions KA_POSTHOC_LOCK_r1.json, ...)
  test                    the single TEST look (part B): lock recheck -> OPENINGS.json entry saved -> KA TEST labels
                          -> gate (T.kk) -> K1-K6 -> KA_POSTHOC_TEST.json
No path whose name contains SEALED is ever opened (``guard``).  KA TEST labels are read only by ``load_ka_split`` after
the OPENINGS.json entry of the TEST step is on disk.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_name, "4")              # evaluation: one process, BLAS 4 (plan 3.4); fit workers get 1
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only (plan 3.4)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Iterable, Mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REVIEW_DIR = ROOT / "reports" / "review_followup_20261011"
if str(REVIEW_DIR) not in sys.path:
    sys.path.append(str(REVIEW_DIR))               # r2_retrain.py and r8_evaluate.py import ``rcommon`` by plain name

import numpy as np  # noqa: E402

from reports.q_production_20261005 import frame_age_seconds as FA  # noqa: E402
from reports.q_production_20261005 import kill_label as KL  # noqa: E402
from reports.q_production_20261005 import scale_strata as SS  # noqa: E402
from reports.q_production_20261005 import seed_replicates as SR  # noqa: E402
from reports.thesis_followup_20261010 import common as TC  # noqa: E402
import r2_retrain as R2  # noqa: E402
import r8_evaluate as R8  # noqa: E402
import rcommon as R  # noqa: E402
from studies.q_production_20261005 import qbaselines, qeval, qtest  # noqa: E402
from studies.q_production_20261005 import release as qrelease  # noqa: E402
from studies.tog_experiments_20261012.common import peak_memory_bytes, run_record  # noqa: E402

VERSION = "tog_ka_posthoc_v1_20261014"
EXPERIMENT = "KA_POSTHOC"
PLAN_REL = "reports/tog_experiments_20261012/PLAN_KA_POSTHOC_KO.md"
PLAN_COMMIT = "219605be163c5293779ce8f171550d6e9bc1c540"     # the plan the author confirmed on 10-14 (DECISION_LOG)
CODE_REL = "studies/tog_experiments_20261012/ka_posthoc.py"
TEST_REL = "tests/test_tog_ka_posthoc.py"
MIRRORED = ("reports/q_production_20261005/scale_strata.py", "reports/review_followup_20261011/r8_evaluate.py",
            "reports/review_followup_20261011/rcommon.py", "reports/thesis_followup_20261010/common.py",
            "reports/q_production_20261005/frame_age_seconds.py", "reports/q_production_20261005/followup_tier1.py",
            "reports/review_followup_20261011/r2_retrain.py", "reports/q_production_20261005/seed_replicates.py",
            "reports/thesis_followup_20261010/n2_seed_aggregation.py", "reports/q_production_20261005/kill_label.py",
            "studies/q_production_20261005/qeval.py", "studies/q_production_20261005/qbaselines.py",
            "studies/q_production_20261005/qtest.py", "studies/q_production_20261005/qtrain.py")
TRAINING = Path(r"D:\LOL_Training_20261004")
LABEL_TAG = KL.LABEL_TAG
ROLE = {"train": KL.TRAIN, "q_dev": KL.Q_DEV, "q_cal": KL.Q_CAL, "test": KL.TEST}
SPLIT_NAME = {"q_dev": "Q_DEV", "test": "TEST"}
P1, PT = KL.P1, KL.PT
S10, S00, S11, S01 = KL.lgbm(1, 0), KL.lgbm(0, 0), KL.lgbm(1, 1), KL.lgbm(0, 1)
KA_CELLS = tuple(KL.CELLS)
NEST = "P1_NEST"
SEED_MAIN = int(SR.CHECK_SEED)                      # 20261001: the q_ka_1 P1 cell
EXTRA_SEEDS = tuple(int(s) for s in SR.SEEDS)       # 20261002, 20261003
CHECK = f"P1_s{SEED_MAIN}_check"
SEED_CELLS = {seed: f"P1_s{seed}" for seed in EXTRA_SEEDS}
FIT_SEEDS = {CHECK: SEED_MAIN, **{name: seed for seed, name in SEED_CELLS.items()}}
FIT_ORDER = (CHECK, *SEED_CELLS.values(), NEST)    # the cheap reproduction gate first (seed_replicates.py cmd_plan)
LEVEL, RESAMPLES, BOOT_SEED = qeval.EXPLORATORY_LEVEL, qeval.BOOTSTRAP_RESAMPLES, qeval.BOOTSTRAP_SEED
GATE_TOL, PT_REFIT_TOL, SEED_REPRO_TOL, STOP_TOL, RELOAD_TOL = 1e-9, 1e-8, 1e-7, 1e-9, 1e-7
FIT_LIMIT_S, EVAL_LIMIT_S, KILL_GRACE_S = 3600.0, 7200.0, 300.0
COMMIT_LIMIT_BYTES = int(95.5 * 2**30)             # review_followup_20261011/memwatch.ps1: Committed Bytes / 1GB > 95.5
LGBM_THREADS = 3                                    # frozen: qtrain.py:359, :367; r2_retrain.py:209, :220
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
KILL_ORDER = qeval.KILL_COUNT_ORDER
SIGN = "Brier(LightGBM cell) - Brier(PT_flex); negative = LightGBM better"
POSTHOC = "post-hoc analysis on previously opened labels (PLAN_KA_POSTHOC_KO.md)"
OPENING_NOTE = ("repeated look at the unsealed KA TEST labels (first evaluated 2026-10-09T11:30:47Z by kill_label.py evaluate); "
                "post-hoc reinforcement, PLAN_KA_POSTHOC_KO.md")
LOCK_BASE = "KA_POSTHOC_LOCK"

if (FA.RESAMPLES, FA.SEED, TC.RESAMPLES, TC.SEED, TC.LEVEL) != (RESAMPLES, BOOT_SEED, RESAMPLES, BOOT_SEED, LEVEL):
    raise RuntimeError("the mirrored scripts' bootstrap settings differ from the plan's 2,000 / 20261011 / 95 %")
if (SS.P1_CELL, SS.PT_FLEX, SR.P1_CELL, SR.PT_FLEX, R.P1, R.PT) != (P1, PT) * 3 or R2.VARIANTS[NEST] != (P1, "NEST"):
    raise RuntimeError("the mirrored scripts name other P1 / PT_flex cells")
if tuple(R8.COHORTS) != ("T", "S", "ASYM", "P") or EXTRA_SEEDS != (20261002, 20261003) or SEED_MAIN != 20261001:
    raise RuntimeError("the mirrored cohorts or seeds differ from the plan")

jsonable = TC.jsonable


# ----------------------------------------------------------------------------- errors
class Blocked(RuntimeError):
    """A plan section 7 condition: stop at this step and report BLOCKED."""


class GateFailed(Blocked):
    def __init__(self, gate: Mapping[str, Any]) -> None:
        super().__init__(f"KA_REPORT reproduction gate failed: max |diff| {gate['max_abs_diff']} > {gate['tolerance']}")
        self.gate = dict(gate)


class SealedPathError(PermissionError):
    """A path whose name contains SEALED: never opened (plan 2.1)."""


class LabelAccessError(RuntimeError):
    """KA TEST labels requested before the TEST step's OPENINGS.json entry was saved (plan 4 step 7)."""


# ----------------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    out: Path = TRAINING / "tog_ka_1"
    q_run: Path = TRAINING / "q_run_1"
    q_ka: Path = TRAINING / "q_ka_1"
    participants: Path = TRAINING / "review_followup_20261011" / "PARTICIPANTS_KILLCREDIT.npz"
    openings: Path = ROOT / "reports" / "tog_experiments_20261012" / "OPENINGS.json"
    exposure: Path = ROOT / "reports" / "tog_experiments_20261012" / "EXPOSURE_KO.md"
    repo: Path = ROOT
    plan_rel: str = PLAN_REL
    plan_commit: str = PLAN_COMMIT

    @property
    def rows(self) -> Path:
        return self.q_run / "Q_ROWS_h90.npz"

    @property
    def rows_manifest(self) -> Path:
        return self.q_run / "Q_ROWS_h90.json"

    @property
    def ka_labels(self) -> Path:
        return self.q_ka / "KA_LABELS.npz"

    @property
    def ka_prepare(self) -> Path:
        return self.q_ka / "KA_PREPARE.json"

    @property
    def ka_report(self) -> Path:
        return self.q_ka / "KA_REPORT.json"

    @property
    def participants_manifest(self) -> Path:
        return self.participants.with_suffix(".json")

    @property
    def refit_inputs(self) -> Path:
        return self.out / "REFIT_INPUTS.json"

    @property
    def qdev(self) -> Path:
        return self.out / "KA_POSTHOC_QDEV.json"

    @property
    def launch_log(self) -> Path:
        return self.out / "LAUNCH.log"

    def cell(self, name: str) -> Path:
        return self.out / "cells" / name

    def ka_cell(self, key: str) -> Path:
        return self.q_ka / "cells" / key


# ----------------------------------------------------------------------------- guarded io (SEALED refused, write once)
def guard(path: Path | str) -> Path:
    """Refuse any path with SEALED in a component name before it is opened, listed or hashed (plan 2.1)."""

    path = Path(path)
    if any("SEALED" in part.upper() for part in path.parts):
        raise SealedPathError(f"refusing a path whose name contains SEALED: {path}")
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with guard(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(guard(path).read_text(encoding="utf-8"))


def load_npz(path: Path | str, names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(guard(path), allow_pickle=False) as data:
        return {name: data[name] for name in (list(names) if names is not None else data.files)}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(jsonable(payload), indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _place_once(path: Path, writer: Callable[[Any], None]) -> Path:
    """Write through a pending file and move it in place; an existing target is refused (os.rename does not replace
    on Windows, so a concurrent writer cannot overwrite either)."""

    path = guard(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        pending.unlink()
        raise FileExistsError(f"{path} exists; it is written once")
    os.rename(pending, path)
    return path


def write_json_once(path: Path | str, payload: Any) -> str:
    """JSON (UTF-8, LF only, non-finite floats as null) written once; returns its SHA-256."""

    data = _json_bytes(payload)
    return sha256_file(_place_once(Path(path), lambda stream: stream.write(data)))


def write_npz_once(path: Path | str, arrays: Mapping[str, np.ndarray]) -> str:
    return sha256_file(_place_once(Path(path), lambda stream: np.savez(stream, **arrays)))


def write_pickle_once(path: Path | str, obj: Any) -> str:
    return sha256_file(_place_once(Path(path), lambda stream: pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)))


def files_record(files: Mapping[str, Path | str]) -> dict[str, dict[str, Any]]:
    """{label: {path, sha256, bytes}} of the files, hashed now (SEALED refused)."""

    out = {}
    for label, path in files.items():
        path = guard(path)
        out[label] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def repo_modules() -> dict[str, str]:
    """Repository files of every module this process has loaded (sys.modules ``__file__`` under the root), hashed now
    as disk bytes (plan 4 step 5)."""

    return {name: sha256_file(ROOT / name) for name in sorted(qrelease.workspace_files(list(sys.modules.values())))}


def mirrored_scripts() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in MIRRORED}


def environment() -> dict[str, Any]:
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "threads_env": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()}


def stage_record(name: str, started: str, tick: float, cpu0: float) -> dict[str, Any]:
    return {"stage": name, "started_utc": started, "finished_utc": utc(), "wall_seconds": round(time.perf_counter() - tick, 3),
            "cpu_seconds": round(time.process_time() - cpu0, 3), "peak_memory_bytes": peak_memory_bytes()}


def watchdog(limit_s: float | None, on_timeout: Callable[[], None]) -> threading.Event:
    """Stop this process after ``limit_s`` seconds (plan 3.4 time limits): ``on_timeout`` records it, then exit 3."""

    done = threading.Event()
    if limit_s is None:
        return done

    def run() -> None:
        if done.wait(float(limit_s)):
            return
        try:
            on_timeout()
        finally:
            os._exit(3)

    threading.Thread(target=run, name="ka_posthoc_watchdog", daemon=True).start()
    return done


def _blocked(paths: Paths, stage: str, details: Mapping[str, Any]) -> int:
    """A BLOCKED record (stage, UTC time, reason) next to the outputs; the caller stops."""

    path = paths.out / f"{stage.upper()}_BLOCKED_{_stamp()}.json"
    while path.exists():
        path = path.with_name(path.stem + "_1.json")
    write_json_once(path, {"version": VERSION, "experiment": EXPERIMENT, "stage": stage, "status": "BLOCKED", "utc": utc(), **dict(details)})
    _say(paths, f"BLOCKED {stage}: {path.name}")
    return 2


def _say(paths: Paths, message: str) -> None:
    try:
        print(message, flush=True)
    except Exception:  # noqa: BLE001  (a detached process may have no console)
        pass


# ----------------------------------------------------------------------------- git (plan 4 step 5)
def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)


def plan_blob_sha256(repo: Path, commit: str, rel: str) -> tuple[str, int]:
    """SHA-256 and size of ``git cat-file blob <commit>:<rel>`` stdout as bytes (no shell, no text conversion)."""

    blob = _git(Path(repo), "cat-file", "blob", f"{commit}:{rel}")
    if blob.returncode:
        raise Blocked(f"git cat-file blob {commit}:{rel} failed: {blob.stderr.decode('utf-8', 'replace').strip()}")
    return hashlib.sha256(blob.stdout).hexdigest(), len(blob.stdout)


def plan_record(paths: Paths) -> dict[str, Any]:
    sha, size = plan_blob_sha256(paths.repo, paths.plan_commit, paths.plan_rel)
    last = _git(paths.repo, "log", "-1", "--format=%H", "--", paths.plan_rel).stdout.decode("ascii", "replace").strip()
    unchanged = _git(paths.repo, "diff", "--quiet", paths.plan_commit, "--", paths.plan_rel).returncode == 0
    return {"path": paths.plan_rel, "commit": paths.plan_commit, "blob_sha256": sha, "blob_bytes": size, "last_commit_of_path": last,
            "working_copy_unchanged_since_commit": unchanged,
            "rule": "SHA-256 of `git cat-file blob <commit>:<path>` stdout bytes (plan 4 step 5); OPENINGS prespec_sha256 is this value"}


def code_git(files: Iterable[str]) -> dict[str, Any]:
    """HEAD, the last commit of this module and ``git status --porcelain`` of the code paths (plan 4 step 5)."""

    files = sorted(set(files))
    head = _git(ROOT, "rev-parse", "HEAD").stdout.decode("ascii", "replace").strip()
    last = _git(ROOT, "log", "-1", "--format=%H", "--", CODE_REL).stdout.decode("ascii", "replace").strip()
    status = _git(ROOT, "status", "--porcelain", "--untracked-files=all", "--", *files)
    text = status.stdout.decode("utf-8", "replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return {"head": head, "last_commit_of_ka_posthoc": last, "status_porcelain": text, "status_returncode": status.returncode,
            "modified": sorted(line for line in lines if not line.startswith("??")),
            "untracked": sorted(line[3:].strip() for line in lines if line.startswith("??")), "paths_checked": len(files)}


def code_commit_problems(git: Mapping[str, Any]) -> list[str]:
    """The lock is written from committed code: this module and its test committed and unmodified, and no tracked code
    path with an uncommitted change.  Untracked frozen modules of other work (not this task's to commit) are allowed;
    the lock lists them and binds their disk SHA-256."""

    problems = []
    if git.get("status_returncode") != 0:
        problems.append("git status failed")
    if git.get("modified"):
        problems.append(f"uncommitted changes: {git['modified']}")
    own = sorted({CODE_REL, TEST_REL} & set(git.get("untracked") or []))
    if own:
        problems.append(f"not committed: {own}")
    if not git.get("last_commit_of_ka_posthoc"):
        problems.append(f"{CODE_REL} has no commit")
    return problems


# ----------------------------------------------------------------------------- inputs of the re-analyses (plan 2.1)
ROW_COLUMNS = ("engagement_key", "match_index", "role", "eval_n_kills", "frame_age_s", "elapsed_min", "pt_p_pre")


def load_rows_table(paths: Paths) -> tuple[dict[str, np.ndarray], str]:
    """The q row table (columns y and label_svi_delta are not loaded: plan 2.1) after its SHA-256 check."""

    manifest = read_json(paths.rows_manifest)
    sha = sha256_file(paths.rows)
    if sha != manifest.get("rows_sha256"):
        raise Blocked(f"{paths.rows}: SHA-256 {sha} differs from Q_ROWS_h90.json rows_sha256")
    return load_npz(paths.rows, ROW_COLUMNS), sha


def opening_saved(paths: Paths, opening: Mapping[str, Any] | None) -> bool:
    if not opening or not guard(paths.openings).is_file():
        return False
    fields = ("experiment", "attempt", "utc", "lock_sha256")
    return any(all(o.get(k) == opening.get(k) for k in fields) for o in read_json(paths.openings).get("openings", []))


def load_ka_split(paths: Paths, rows_table: Mapping[str, np.ndarray], split: str, *, opening: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """KA labels of one split's eligible rows.  TEST is served only after the TEST step's OPENINGS.json entry is on
    disk (plan 4 step 7); other splits never need it.  Only the requested split's rows are returned."""

    if split not in ("train", "q_dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not opening_saved(paths, opening):
        raise LabelAccessError("KA TEST labels are read only after the OPENINGS.json entry of the TEST step is saved")
    prepare = read_json(paths.ka_prepare)
    sha = sha256_file(paths.ka_labels)
    if sha != prepare.get("ka_labels_sha256"):
        raise Blocked(f"{paths.ka_labels}: SHA-256 {sha} differs from KA_PREPARE.json ka_labels_sha256")
    ka = load_npz(paths.ka_labels, ["engagement_key", "eligible", "y_ka", "weight_raw_ka"])
    if not np.array_equal(ka["engagement_key"], rows_table["engagement_key"]):
        raise Blocked("KA labels are not aligned with the q row table (kill_label.py:160-161)")
    split_rows = np.flatnonzero(np.asarray(rows_table["role"]) == ROLE[split])
    eligible = np.asarray(ka["eligible"], dtype=bool)[split_rows]
    idx = split_rows[eligible]
    out = {"split": split, "split_rows": split_rows, "eligible_of_split": eligible, "idx": idx,
           "y": np.asarray(ka["y_ka"], dtype=np.float64)[idx], "weight_raw": np.asarray(ka["weight_raw_ka"], dtype=np.float64)[idx],
           "sha256": sha}
    del ka
    if not np.isin(out["y"], (0.0, 1.0)).all():
        raise Blocked(f"{split}: KA labels are not binary on eligible rows")
    return out


def load_participants(paths: Paths, rows_table: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    manifest = read_json(paths.participants_manifest)
    sha = sha256_file(paths.participants)
    if sha != manifest.get("npz_sha256"):
        raise Blocked(f"{paths.participants}: SHA-256 differs from its npz_sha256 (r8_evaluate.py:28-29)")
    part = load_npz(paths.participants, ["engagement_key", "n_blue", "n_red", "cohort", "t5"])
    if not np.array_equal(part["engagement_key"], rows_table["engagement_key"]):
        raise Blocked("participant file is not aligned with the q rows (r8_evaluate.py:30-31)")
    return part


def stored_predictions(result_path: Path, pred_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(rows, logit, p, sha256) of a saved prediction file whose SHA-256 equals its cell's RESULT.json value."""

    result = read_json(result_path)
    if result.get("status") != "completed":
        raise Blocked(f"{result_path}: the cell is not completed")
    expected = (result.get("scores") or {}).get("q_dev", {}).get("pred_sha256") if split == "q_dev" else result.get("pred_test_sha256")
    sha = sha256_file(pred_path)
    if sha != expected:
        raise Blocked(f"{pred_path}: SHA-256 {sha} differs from {result_path.name} ({expected})")
    data = load_npz(pred_path, ["row", "logit", "p"])
    return np.asarray(data["row"], dtype=np.int64), np.asarray(data["logit"], dtype=np.float64), np.asarray(data["p"], dtype=np.float64), sha


def analysis_cells(paths: Paths) -> dict[str, Path]:
    """Cell directories whose predictions the analyses read: the six q_ka_1 cells, P1_NEST and the two new seeds (the
    seed-20261001 check fit is not used: plan 3.3)."""

    return {**{key: paths.ka_cell(key) for key in KA_CELLS}, **{name: paths.cell(name) for name in (NEST, *SEED_CELLS.values())}}


def split_data(paths: Paths, split: str, rows_table: Mapping[str, np.ndarray], *, opening: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything K1-K6 need on one split, in the prediction row order of the cells (kill_label.py:316-328)."""

    ka = load_ka_split(paths, rows_table, split, opening=opening)
    pred_name = "pred_q_dev.npz" if split == "q_dev" else "pred_test.npz"
    preds, rows_ref, files = {}, None, []
    for key, directory in analysis_cells(paths).items():
        rows, _, p, _ = stored_predictions(directory / "RESULT.json", directory / pred_name, split)
        if rows_ref is None:
            rows_ref = rows
        elif not np.array_equal(rows, rows_ref):
            raise Blocked(f"{key}: KA prediction rows differ (kill_label.py:322-323)")
        preds[key] = p
        files += [directory / "RESULT.json", directory / pred_name]
    if not np.array_equal(np.sort(rows_ref), ka["idx"]):
        raise Blocked(f"{split}: KA prediction rows are not the eligible rows (kill_label.py:325-327)")
    position = np.searchsorted(ka["idx"], rows_ref)
    part = load_participants(paths, rows_table)
    data = {"split": split, "rows": rows_ref, "y": ka["y"][position], "match": np.asarray(rows_table["match_index"])[rows_ref],
            "n_kills": np.asarray(rows_table["eval_n_kills"])[rows_ref], "split_n_kills": np.asarray(rows_table["eval_n_kills"])[ka["split_rows"]],
            "age": np.asarray(rows_table["frame_age_s"], dtype=np.float64)[rows_ref],
            "minutes": np.asarray(rows_table["elapsed_min"], dtype=np.float64)[rows_ref],
            "p_pre": np.asarray(rows_table["pt_p_pre"], dtype=np.float64)[rows_ref],
            "cohort": np.asarray(part["cohort"])[rows_ref], "t5": np.asarray(part["t5"], dtype=bool)[rows_ref],
            "n_blue": np.asarray(part["n_blue"], dtype=np.int64)[rows_ref], "n_red": np.asarray(part["n_red"], dtype=np.int64)[rows_ref],
            "preds": preds, "split_rows": int(ka["split_rows"].size), "ties": int((~ka["eligible_of_split"]).sum()),
            "files": [paths.rows, paths.rows_manifest, paths.ka_labels, paths.ka_prepare, paths.participants, paths.participants_manifest, *files]}
    return data


def train_rate(paths: Paths, rows_table: Mapping[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    """K4 constant cell: TRAIN eligible rows' match-weighted KA positive rate (qbaselines.constant_rate, weight_raw_ka)."""

    train = load_ka_split(paths, rows_table, "train")
    rate = constant_rate(train["y"], train["weight_raw"])
    match = np.asarray(rows_table["match_index"])[train["idx"]]
    return rate, {"rows": int(train["idx"].size), "matches": int(np.unique(match).size), "rate": rate,
                  "rule": "sum w*y_ka / sum w over the eligible TRAIN rows, w = weight_raw_ka (qbaselines.py:77-79, q_worker.py:144-146)"}


def gate_reference(paths: Paths, split: str) -> dict[str, Any]:
    report = read_json(paths.ka_report)
    try:
        return report["splits"][split]["ka_trained_on_ka"]
    except (KeyError, TypeError) as error:
        raise Blocked(f"KA_REPORT.json has no splits.{split}.ka_trained_on_ka") from error


# ----------------------------------------------------------------------------- statistics helpers
def constant_rate(y: np.ndarray, w: np.ndarray) -> float:
    return float(qbaselines.constant_rate(y, w))


def _kill_strata(n_kills: np.ndarray) -> np.ndarray:
    kills = np.asarray(n_kills)
    if kills.size and int(kills.min()) < 1:
        raise Blocked("a kill count is below 1 (plan section 7)")
    return qeval.kill_count_strata(kills)


def _auc_or_none(y: np.ndarray, p: np.ndarray, match: np.ndarray) -> float | None:
    """qeval.auc with match weights (= CellBoot.auc point, common.py:329); None when a class is absent."""

    if not (y == 1.0).any() or not (y == 0.0).any():
        return None
    return float(qeval.auc(y, p, match_ids=match))


def _slope(y: np.ndarray, p: np.ndarray, match: np.ndarray) -> float | None:
    cal = qeval.calibration_slope_intercept(y, p, match_ids=match)
    return cal.get("slope") if isinstance(cal, dict) else None


def _point_interval(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"point": float(entry["point"]), "interval": [float(v) for v in entry["interval"]], "level": float(entry.get("level", LEVEL))}


# ----------------------------------------------------------------------------- gate (plan 2.1)
def run_gate(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, reference: Mapping[str, Any],
             tol: float = GATE_TOL) -> dict[str, Any]:
    """Recompute KA_REPORT ``ka_trained_on_ka`` (kill_label._contrast_block: per-cell AUC and Brier, P1 and the S
    contrasts with intervals) and compare every value with the report."""

    block = KL._contrast_block(y, {k: preds[k] for k in KA_CELLS}, match)
    mine = block["contrasts"]["contrasts"]
    items: list[tuple[str, Any, Any]] = []
    try:
        for cell in KA_CELLS:
            for metric in ("auc", "brier"):
                items.append((f"metrics/{cell}/{metric}", block["metrics"][cell][metric], reference["metrics"][cell][metric]))
        for name in KL.CONTRASTS:
            ref = reference["contrasts"]["contrasts"][name]
            items.append((f"contrasts/{name}/point", mine[name]["point"], ref["point"]))
            for i in (0, 1):
                items.append((f"contrasts/{name}/interval/{i}", mine[name]["interval"][i], ref["interval"][i]))
    except (KeyError, TypeError) as error:
        raise Blocked(f"KA_REPORT reference lacks {error}") from error
    diffs = {}
    for label, a, b in items:
        value = abs(float(a) - float(b)) if a is not None and b is not None else math.inf
        diffs[label] = value if math.isfinite(value) else math.inf
    worst = max(diffs, key=diffs.get)
    return {"reference": "KA_REPORT.json splits.<split>.ka_trained_on_ka (kill_label.py:291-298)", "tolerance": tol, "items": len(items),
            "max_abs_diff": diffs[worst], "worst": worst, "passed": bool(diffs[worst] <= tol),
            "recomputed": {"metrics": block["metrics"], "contrasts": {name: _point_interval(mine[name]) for name in KL.CONTRASTS}}}


# ----------------------------------------------------------------------------- K1 kill-count strata (plan 2.2)
def share_one_kill(points: Mapping[str, float | None], rows: Mapping[str, int]) -> dict[str, Any]:
    """share1 = g(1) n(1) / sum_s g(s) n(s) with the gain g(s) = -P1(s); undefined when the denominator is <= 0."""

    gain = {s: (-float(points[s]) if points.get(s) is not None else 0.0) for s in KILL_ORDER}
    terms = {s: gain[s] * float(rows.get(s, 0)) for s in KILL_ORDER}
    numerator, denominator = terms["1"], float(sum(terms.values()))
    defined = bool(denominator > 0)
    return {"defined": defined, "value": numerator / denominator if defined else None, "gain": gain, "gain_times_rows": terms,
            "numerator": numerator, "denominator": denominator,
            "reason": None if defined else "the gain denominator sum_s g(s) n(s) is <= 0: undefined (plan 2.2)",
            "rule": "g(s) = -P1(s); share1 = g(1) n(1) / sum_s g(s) n(s), n(s) = non-tie rows; point only"}


def k1_kill_count(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, n_kills: np.ndarray,
                  split_n_kills: np.ndarray) -> dict[str, Any]:
    """scale_strata.py kill_count layer on the eligible rows: per stratum rows, matches, P1 / PT_flex Brier and AUC,
    P1 with a paired match bootstrap inside the stratum (95 %); ties = split rows - eligible rows of the stratum."""

    strata = _kill_strata(n_kills)
    split_strata = _kill_strata(split_n_kills)
    out: dict[str, Any] = {}
    for s in KILL_ORDER:
        mask = strata == s
        ties = int((split_strata == s).sum() - mask.sum())
        if not mask.any():
            out[s] = {"rows": 0, "matches": 0, "ties": ties, "brier": None, "auc": None, "contrasts": None}
            continue
        losses = {key: qeval.per_match_loss(y[mask], preds[key][mask], match[mask]) for key in (P1, PT)}
        shared, aligned = qeval.align_cells(losses)
        boot = qeval.paired_match_bootstrap(shared, aligned, {"P1": qeval.difference(P1, PT)}, levels=LEVEL)
        metrics = {key: qeval.evaluate(y[mask], preds[key][mask], match[mask])["match_weighted"] for key in (P1, PT)}
        out[s] = {"rows": int(mask.sum()), "matches": int(shared.size), "ties": ties,
                  "brier": {k: m["brier"] for k, m in metrics.items()}, "auc": {k: m["auc"] for k, m in metrics.items()},
                  "contrasts": {"P1": _point_interval(boot["contrasts"]["P1"])}}
    points = {s: (v["contrasts"] or {}).get("P1", {}).get("point") for s, v in out.items()}
    return {"strata": out, "share_one_kill": share_one_kill(points, {s: v["rows"] for s, v in out.items()}),
            "interval": "paired match bootstrap inside each stratum (scale_strata.py:73-76; qeval.py:554-642), 2,000, PCG64(20261011), 95 %"}


# ----------------------------------------------------------------------------- K2 participant cohorts (plan 2.3)
def k2_cohorts(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, cohort: np.ndarray, t5: np.ndarray,
               n_blue: np.ndarray, n_red: np.ndarray, n_kills: np.ndarray, counts: np.ndarray | None = None) -> dict[str, Any]:
    """r8_evaluate.py on the eligible rows with the KA P1 and PT_flex cells; CellBoot draws shared by every cohort."""

    keys = [P1, PT]
    n = int(np.asarray(y).size)
    cb = TC.CellBoot(match, y, {k: preds[k] for k in keys}, R.draws_for(match) if counts is None else counts)
    cons = {"P1": {P1: 1.0, PT: -1.0}}
    masks = {name: np.asarray(cohort) == i for i, name in enumerate(R8.COHORTS)}
    masks["T5"] = np.asarray(t5, dtype=bool)
    masks["all"] = np.ones(n, dtype=bool)
    cohorts: dict[str, Any] = {}
    for name, mask in masks.items():
        entry = {"rows": int(mask.sum()), "row_share": float(mask.mean()), "matches": int(np.unique(match[mask]).size),
                 "suppressed": cb.suppressed(mask)}
        if mask.any():
            point, _ = cb.briers(mask, keys)
            con, _ = R.contrast_summary(cb, mask, cons)
            entry.update(brier=dict(zip(keys, point.tolist())), auc={k: _auc_or_none(y[mask], preds[k][mask], match[mask]) for k in keys},
                         calibration_slope={k: _slope(y[mask], preds[k][mask], match[mask]) for k in keys}, contrasts=con)
        cohorts[name] = entry
    kill_s = _kill_strata(n_kills)
    table = {}
    for name in (*R8.COHORTS, "T5"):
        m = masks[name]
        table[name] = {s: {"rows": int((m & (kill_s == s)).sum()), "share_of_cohort": float((m & (kill_s == s)).sum() / max(m.sum(), 1))}
                       for s in KILL_ORDER}
    n_blue, n_red = np.asarray(n_blue), np.asarray(n_red)
    lo, hi = np.minimum(n_blue, n_red), np.maximum(n_blue, n_red)
    distributions = {name: {str(k): int((values == k).sum()) for k in range(6)}
                     for name, values in (("n_blue", n_blue), ("n_red", n_red), ("min_side", lo), ("max_side", hi))}
    return {"cohort_codes": {str(i): c for i, c in enumerate(R8.COHORTS)}, "cohorts": cohorts, "crosstab": {"kill_count": table},
            "distributions": distributions,
            "interval": "CellBoot (common.py:290-367): one PCG64(20261011) draw set over the split's matches shared by every cohort, 95 %",
            "denominators": "row_share and cross-tab shares over non-tie rows; metrics re-weight 1/m inside the cohort"}


# ----------------------------------------------------------------------------- K3 frame age (plan 2.4)
def k3_frame_age(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, age: np.ndarray, minutes: np.ndarray,
                 p_pre: np.ndarray, kills: np.ndarray, *, resamples: int = RESAMPLES, seed: int = BOOT_SEED) -> dict[str, Any]:
    """frame_age_seconds.analyse on the eligible rows with the KA cells (no P1-best): 1-s bins, per-bin match-weighted
    AUC / Brier / contrasts with shared draws, WLS trends on the bin midpoints weighted by bin matches."""

    age = np.asarray(age, dtype=np.float64)
    if age.min() < 0 or age.max() > 10:
        raise Blocked("frame age outside [0, 10] s (frame_age_seconds.py:55-56)")
    y, match = np.asarray(y, dtype=np.float64), np.asarray(match)
    minutes, p_pre, kills = np.asarray(minutes, dtype=np.float64), np.asarray(p_pre, dtype=np.float64), np.asarray(kills, dtype=np.float64)
    bins = np.minimum(np.floor(age).astype(int), 9)          # [9,10] closed at 10 s
    match_ids, pos = np.unique(match, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(match_ids), len(match_ids)), minlength=len(match_ids)).astype(np.float64)
                       for _ in range(resamples)])
    cons = {"P1": {P1: 1.0, PT: -1.0}, "S_effect_LGBM_meanD": {S10: 0.5, S00: -0.5, S11: 0.5, S01: -0.5}}
    auc_cells = {"lightgbm": P1, "pt_flex": PT}
    per_bin = []
    draws_auc = {k: np.zeros((resamples, 10)) for k in auc_cells}
    draws_con = {k: np.zeros((resamples, 10)) for k in cons}
    point_auc = {k: np.zeros(10) for k in auc_cells}
    point_con = {k: np.zeros(10) for k in cons}
    bin_matches = np.zeros(10)
    for b in range(10):
        mask = bins == b
        if not mask.any():
            raise Blocked(f"frame-age bin {b} has no rows")
        present, inv = np.unique(pos[mask], return_inverse=True)
        m_per = np.bincount(inv).astype(np.float64)
        base_w = 1.0 / m_per[inv]
        bin_matches[b] = len(present)
        entry = {"bin": f"[{b},{b + 1})" if b < 9 else "[9,10]", "rows": int(mask.sum()), "matches": int(len(present)),
                 "mean_game_minute": float(minutes[mask].mean()), "mean_abs_p_pre_dev": float(np.abs(p_pre[mask] - 0.5).mean()),
                 "mean_kills": float(kills[mask].mean()), "metrics": {}, "contrasts": {}}
        yb = y[mask]
        for name, key in auc_cells.items():
            p = preds[key][mask]
            ev = qeval.evaluate(yb, p, match[mask])["match_weighted"]
            entry["metrics"][name] = {"auc": ev["auc"], "brier": ev["brier"]}
            point_auc[name][b] = ev["auc"] if ev["auc"] is not None else np.nan
            order = np.argsort(p, kind="stable")
            ps = p[order]
            groups = np.flatnonzero(np.concatenate(([True], ps[1:] != ps[:-1])))
            ys, inv_o, bw = yb[order], inv[order], base_w[order]
            for r in range(resamples):
                draws_auc[name][r, b] = FA._auc_sorted(ys, counts[r, present][inv_o] * bw, groups)
        names = sorted({k for weights in cons.values() for k in weights})
        losses = np.stack([np.bincount(inv, weights=(preds[k][mask] - yb) ** 2) / m_per for k in names], axis=1)
        col = {k: i for i, k in enumerate(names)}
        pt = losses.mean(axis=0)
        dr = np.empty((resamples, len(names)))
        for start in range(0, resamples, 250):
            w = counts[start:start + 250][:, present]
            dr[start:start + 250] = (w @ losses) / w.sum(axis=1, keepdims=True)
        for cname, weights in cons.items():
            vec = np.zeros(len(names))
            for k, v in weights.items():
                vec[col[k]] = v
            point_con[cname][b] = float(pt @ vec)
            draws_con[cname][:, b] = dr @ vec
            d = dr @ vec
            entry["contrasts"][cname] = {"point": float(pt @ vec), "interval": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
        per_bin.append(entry)
    mid = np.arange(10) + 0.5
    trends = {}
    for family, points, draws in (("auc", point_auc, draws_auc), ("contrast", point_con, draws_con)):
        for name in points:
            a, b = FA.wls_line(mid, points[name], bin_matches)
            boot = np.array([FA.wls_line(mid, draws[name][r], bin_matches) for r in range(resamples)])
            trends[f"{family}:{name}"] = {"intercept_0s": a, "slope_per_s": b, "value_at_5s": a + 5 * b,
                                          "intercept_0s_interval": [float(np.quantile(boot[:, 0], 0.025)), float(np.quantile(boot[:, 0], 0.975))],
                                          "slope_interval": [float(np.quantile(boot[:, 1], 0.025)), float(np.quantile(boot[:, 1], 0.975))]}
    overall = {name: qeval.evaluate(y, preds[key], match)["match_weighted"]["auc"] for name, key in auc_cells.items()}
    return {"rows": int(y.size), "matches": int(len(match_ids)), "overall_auc": overall, "bins": per_bin, "trends": trends,
            "interval": "one np.random.default_rng(20261011) match draw set (2,000) shared by every bin; WLS refit per draw; 2.5/97.5 percentiles",
            "reporting": "report and manuscript tables carry the bin table and the four slopes only; intercepts and 5-s values stay in this JSON (plan 2.4)",
            "balance": "mean game minute, mean |p_pre - 0.5| and mean kill count are unweighted row means (frame_age_seconds.py:74-76)"}


# ----------------------------------------------------------------------------- K4 Brier reduction and skill (plan 2.5)
def k4_brier_skill(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, rate: float) -> dict[str, Any]:
    """Match-weighted Brier of P1, PT_flex and the closed-form constant cell; points only."""

    logit = float(np.log(rate / (1.0 - rate)))                     # qtest.py:230-232
    p_const = qtest.sigmoid(np.full(np.asarray(y).size, logit))   # qtest.py:264-265, no calibration
    brier = {P1: qeval.evaluate(y, preds[P1], match)["match_weighted"]["brier"], PT: qeval.evaluate(y, preds[PT], match)["match_weighted"]["brier"],
             "CONSTANT": qeval.evaluate(y, p_const, match)["match_weighted"]["brier"]}
    reduction = 1.0 - brier[P1] / brier[PT]
    skill_lgbm, skill_pt = 1.0 - brier[P1] / brier["CONSTANT"], 1.0 - brier[PT] / brier["CONSTANT"]
    return {"rate": float(rate), "constant_logit": logit, "brier": brier, "brier_reduction": reduction, "skill_lgbm": skill_lgbm,
            "skill_pt": skill_pt, "brier_reduction_pct": 100 * reduction, "skill_lgbm_pct": 100 * skill_lgbm, "skill_pt_pct": 100 * skill_pt,
            "rules": {"brier_reduction": "1 - Brier(P1) / Brier(PT_flex) (numbers.csv:20)",
                      "skill": "1 - Brier(cell) / Brier(constant) (numbers.csv:627-628)", "interval": "none (points only)"}}


# ----------------------------------------------------------------------------- K5 nested model evaluation (plan 3.2)
def k5_nest(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, n_kills: np.ndarray,
            counts: np.ndarray | None = None) -> dict[str, Any]:
    """r2_retrain.cmd_evaluate restricted to the KA cells: P1, NEST - PT, NEST - P1 with CellBoot (shared draws),
    Brier and AUC of the three cells, the kept share (NEST - PT) / P1 as a point, and the kill-count strata."""

    keys = [P1, PT, NEST]
    cb = TC.CellBoot(match, y, {k: preds[k] for k in keys}, R.draws_for(match) if counts is None else counts)
    allm = np.ones(np.asarray(y).size, dtype=bool)
    cons = {"P1": {P1: 1, PT: -1}, "NEST_minus_PT": {NEST: 1, PT: -1}, "NEST_minus_P1": {NEST: 1, P1: -1}}
    whole, _ = R.contrast_summary(cb, allm, cons)
    point, _ = cb.briers(allm, keys)
    kept = whole["NEST_minus_PT"]["point"] / whole["P1"]["point"] if whole["P1"]["point"] != 0 else None
    out = {"rows": int(allm.size), "matches": int(np.unique(match).size), "brier": dict(zip(keys, point.tolist())),
           "auc": {k: _auc_or_none(y, preds[k], match) for k in keys}, "contrasts": whole,
           "kept_share": {"value": kept, "pct": None if kept is None else 100 * kept,
                          "rule": "(NEST - PT) / P1, point only (numbers.csv:632); P1 is this computation's KA P1 (= the gate's)"},
           "strata": {"kill_count": {}},
           "interval": "CellBoot (common.py:290-367), one PCG64(20261011) draw set shared by the whole split and its strata, 95 %",
           "strata_note": "the kill-count P1 intervals used in the manuscript are K1's; this block's P1 intervals are not (plan 3.2)"}
    labels = _kill_strata(n_kills)
    scons = {"P1": {P1: 1, PT: -1}, "NEST_minus_PT": {NEST: 1, PT: -1}, "NEST_minus_P1": {NEST: 1, P1: -1}}
    for s in sorted(set(labels.tolist())):
        mask = labels == s
        con, _ = R.contrast_summary(cb, mask, scons)
        out["strata"]["kill_count"][s] = {"rows": int(mask.sum()), "contrasts": con,
                                          "auc": {k: _auc_or_none(y[mask], preds[k][mask], match[mask]) for k in keys},
                                          "calibration_slope": {k: _slope(y[mask], preds[k][mask], match[mask]) for k in keys}}
    return out


# ----------------------------------------------------------------------------- K6 seeds (plan 3.3)
def k6_seeds(y: np.ndarray, p1_by_seed: Mapping[int, np.ndarray], pt: np.ndarray, match: np.ndarray) -> dict[str, Any]:
    """seed_replicates.cmd_evaluate for the P1 cell (paired match bootstrap over the split, 95 %) and the
    n2_seed_aggregation loss average (per-match losses averaged over the seeds, then the contrast)."""

    con = {"P1": qeval.difference(P1, PT)}
    pt_ids, pt_loss = qeval.per_match_loss(y, pt, match)
    per_seed, seed_losses = {}, {}
    for seed, p in p1_by_seed.items():
        ids, loss = qeval.per_match_loss(y, p, match)
        shared, aligned = qeval.align_cells({P1: (ids, loss), PT: (pt_ids, pt_loss)})
        seed_losses[int(seed)] = aligned[P1]
        boot = qeval.paired_match_bootstrap(shared, aligned, con, levels=LEVEL)
        per_seed[str(int(seed))] = {"contrasts": {"P1": _point_interval(boot["contrasts"]["P1"])},
                                    "brier": {P1: qeval.evaluate(y, p, match)["match_weighted"]["brier"],
                                              PT: qeval.evaluate(y, pt, match)["match_weighted"]["brier"]}}
    seeds = [str(int(s)) for s in p1_by_seed]
    points = [per_seed[s]["contrasts"]["P1"]["point"] for s in seeds]
    below = [per_seed[s]["contrasts"]["P1"]["interval"][1] < 0 for s in seeds]
    summary = {"seeds": seeds, "points": points, "mean": float(np.mean(points)), "sd": float(np.std(points, ddof=1)) if len(points) > 1 else None,
               "min": float(np.min(points)), "max": float(np.max(points)), "all_intervals_below_zero": all(below)}
    mean_losses = {P1: np.mean(np.stack([seed_losses[int(s)] for s in seeds]), axis=0), PT: pt_loss}
    boot = qeval.paired_match_bootstrap(pt_ids, mean_losses, con, levels=LEVEL, return_draws=True)
    entry = boot["contrasts"]["P1"]
    loss_average = {"point": float(entry["point"]), "interval95": TC.percentile_interval(entry["draws"], 0.95),
                    "bootstrap_sd": float(np.std(entry["draws"], ddof=1)), "qeval_interval95": [float(v) for v in entry["interval"]],
                    "rule": "per-match Brier averaged over the three seeds, then the contrast (n2_seed_aggregation.py:145-154); PT_flex has no seed"}
    return {"per_seed": per_seed, "summary": summary, "loss_average": loss_average,
            "interval": "paired match bootstrap over the split (seed_replicates.py:303-309), 2,000, PCG64(20261011), 95 %"}


# ----------------------------------------------------------------------------- the analysis of one split
def consistency_checks(gate: Mapping[str, Any], k1: Mapping[str, Any], k2: Mapping[str, Any], k3: Mapping[str, Any], k4: Mapping[str, Any],
                       k5: Mapping[str, Any], k6: Mapping[str, Any], *, rows: int, ties: int, tol: float = GATE_TOL) -> dict[str, Any]:
    ref = gate["recomputed"]["contrasts"]["P1"]
    metrics = gate["recomputed"]["metrics"]

    def p1_diff(entry: Mapping[str, Any]) -> float:
        return max(abs(entry["point"] - ref["point"]), *(abs(a - b) for a, b in zip(entry["interval"], ref["interval"])))

    checks: dict[str, dict[str, Any]] = {
        "K2_all_cohort_P1_equals_gate_P1": {"max_abs_diff": p1_diff(k2["cohorts"]["all"]["contrasts"]["P1"])},
        "K5_P1_equals_gate_P1": {"max_abs_diff": p1_diff(k5["contrasts"]["P1"])},
        f"K6_seed_{SEED_MAIN}_P1_equals_gate_P1": {"max_abs_diff": p1_diff(k6["per_seed"][str(SEED_MAIN)]["contrasts"]["P1"])},
        "K4_P1_PT_brier_equal_gate": {"max_abs_diff": max(abs(k4["brier"][P1] - metrics[P1]["brier"]), abs(k4["brier"][PT] - metrics[PT]["brier"]))},
        "K3_overall_auc_equals_gate": {"max_abs_diff": max(abs(k3["overall_auc"]["lightgbm"] - metrics[P1]["auc"]),
                                                           abs(k3["overall_auc"]["pt_flex"] - metrics[PT]["auc"]))},
    }
    for check in checks.values():
        check.update(tolerance=tol, passed=bool(check["max_abs_diff"] <= tol))
    strata = k1["strata"].values()
    checks["K1_rows_and_ties_add_up"] = {"passed": sum(v["rows"] for v in strata) == rows and sum(v["ties"] for v in strata) == ties}
    cohorts = k2["cohorts"]
    checks["K2_cohorts_partition_the_rows"] = {"passed": sum(cohorts[c]["rows"] for c in R8.COHORTS) == rows and cohorts["all"]["rows"] == rows}
    checks["K3_bins_partition_the_rows"] = {"passed": sum(b["rows"] for b in k3["bins"]) == rows}
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


def analyse(split: str, data: Mapping[str, Any], reference: Mapping[str, Any], *, rate: float) -> dict[str, Any]:
    """Gate first; K1-K6 only when it holds (plan 2.1); internal consistency must hold before anything is written."""

    y, match, preds = data["y"], data["match"], data["preds"]
    timings: dict[str, float] = {}
    tick = time.perf_counter()
    gate = run_gate(y, preds, match, reference)
    timings["gate"] = time.perf_counter() - tick
    if not gate["passed"]:
        raise GateFailed(gate)
    results: dict[str, Any] = {"gate": gate}
    counts = R.draws_for(match)                        # one CellBoot draw set for K2 and K5 (r8_evaluate.py:37, r2_retrain.py:357)
    steps = (("K1", lambda: k1_kill_count(y, preds, match, data["n_kills"], data["split_n_kills"])),
             ("K2", lambda: k2_cohorts(y, preds, match, data["cohort"], data["t5"], data["n_blue"], data["n_red"], data["n_kills"], counts)),
             ("K3", lambda: k3_frame_age(y, preds, match, data["age"], data["minutes"], data["p_pre"], data["n_kills"])),
             ("K4", lambda: k4_brier_skill(y, preds, match, rate)),
             ("K5", lambda: k5_nest(y, preds, match, data["n_kills"], counts)),
             ("K6", lambda: k6_seeds(y, {SEED_MAIN: preds[P1], **{s: preds[SEED_CELLS[s]] for s in EXTRA_SEEDS}}, preds[PT], match)))
    for name, step in steps:
        tick = time.perf_counter()
        results[name] = step()
        timings[name] = time.perf_counter() - tick
    checks = consistency_checks(gate, results["K1"], results["K2"], results["K3"], results["K4"], results["K5"], results["K6"],
                                rows=int(np.asarray(y).size), ties=int(data["ties"]))
    if not checks["passed"]:
        error = Blocked(f"{split}: internal consistency checks failed: {sorted(k for k, v in checks['checks'].items() if not v['passed'])}")
        error.checks = checks  # type: ignore[attr-defined]
        raise error
    results["consistency"] = checks
    results["timings_seconds"] = {k: round(v, 3) for k, v in timings.items()}
    return results


def completed_fits(paths: Paths) -> dict[str, dict[str, Any]]:
    out = {}
    for cell in FIT_ORDER:
        path = paths.cell(cell) / "RESULT.json"
        if not guard(path).is_file():
            raise Blocked(f"{cell}: no RESULT.json (run launch first)")
        result = read_json(path)
        if result.get("status") != "completed":
            raise Blocked(f"{cell}: the fit is {result.get('status')!r}")
        out[cell] = result
    reproduction = out[CHECK].get("reproduction") or {}
    if reproduction.get("passed") is not True:
        raise Blocked(f"the seed-{SEED_MAIN} reproduction check did not pass: {reproduction}")
    return out


def fits_summary(fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keep = ("best_iteration", "best_stop_brier", "seed", "seconds", "stage", "pred_test_sha256", "model_sha256", "reproduction",
            "offsets", "stop_metric_check", "reload", "started_utc", "finished_utc")
    return {cell: {k: result.get(k) for k in keep if k in result} for cell, result in fits.items()}


def _print_split(name: str, doc: Mapping[str, Any]) -> None:
    k1, k2, k3, k4, k5, k6 = (doc[k] for k in ("K1", "K2", "K3", "K4", "K5", "K6"))
    print(f"[{name}] rows {doc['rows']} matches {doc['matches']} ties {doc['ties']} gate max|diff| {doc['gate']['max_abs_diff']:.2e}")
    for s, v in k1["strata"].items():
        c = (v["contrasts"] or {}).get("P1") or {}
        print(f"  K1 {s:4s} rows {v['rows']:7d} ties {v['ties']:6d} P1 {c.get('point', float('nan')):+.5f} {c.get('interval')}")
    print(f"  K1 share1 {k1['share_one_kill']['value']}")
    for c in ("T", "S", "ASYM", "P", "T5", "all"):
        e = k2["cohorts"][c]
        p1 = (e.get("contrasts") or {}).get("P1") or {}
        print(f"  K2 {c:4s} rows {e['rows']:7d} P1 {p1.get('point', float('nan')):+.5f} {p1.get('interval')}")
    for key, t in k3["trends"].items():
        print(f"  K3 {key:30s} slope/s {t['slope_per_s']:+.5f} {[round(x, 5) for x in t['slope_interval']]}")
    print(f"  K4 Brier {k4['brier']} reduction {k4['brier_reduction_pct']:.2f}% skill {k4['skill_lgbm_pct']:.2f}% / {k4['skill_pt_pct']:.2f}%")
    for cname, c in k5["contrasts"].items():
        print(f"  K5 {cname:14s} {c['point']:+.5f} {[round(x, 5) for x in c['interval']]}")
    print(f"  K5 kept {k5['kept_share']['pct']}")
    print(f"  K6 points {k6['summary']['points']} sd {k6['summary']['sd']} all<0 {k6['summary']['all_intervals_below_zero']} "
          f"loss-avg {k6['loss_average']['point']:+.5f} {k6['loss_average']['interval95']}")


# ----------------------------------------------------------------------------- stage: qdev (plan 4 step 4)
def qdev_stage(paths: Paths = Paths(), *, time_limit_s: float | None = EVAL_LIMIT_S) -> int:
    target = guard(paths.qdev)
    if target.exists():
        raise FileExistsError(f"{target} exists; the development results are written once")
    started, tick, cpu0 = utc(), time.perf_counter(), time.process_time()
    done = watchdog(time_limit_s, lambda: _blocked(paths, "qdev", {"reason": f"evaluation time limit {time_limit_s} s (plan 3.4)"}))
    try:
        fits = completed_fits(paths)
        rows_table, rows_sha = load_rows_table(paths)
        data = split_data(paths, "q_dev", rows_table)
        rate, train_info = train_rate(paths, rows_table)
        results = analyse("q_dev", data, gate_reference(paths, "q_dev"), rate=rate)
    except GateFailed as error:
        done.set()
        return _blocked(paths, "qdev", {"reason": str(error), "gate": error.gate})
    except Blocked as error:
        done.set()
        return _blocked(paths, "qdev", {"reason": str(error), "checks": getattr(error, "checks", None)})
    rows, matches = int(data["y"].size), int(np.unique(data["match"]).size)
    inputs = sorted({str(Path(p)) for p in [*data["files"], paths.ka_report]} | {str(paths.cell(c) / "RESULT.json") for c in FIT_ORDER})
    payload = {"version": VERSION, "experiment": EXPERIMENT, "split": "Q_DEV", "status": "completed", "label": f"{POSTHOC}; development data",
               "reading": "Q_DEV is development evidence: 'the sign is the same (differs) on development data' only (plan 5 rule 6)",
               "plan": {"path": paths.plan_rel, "commit": paths.plan_commit}, "sign": SIGN,
               "rows": rows, "matches": matches, "split_rows": data["split_rows"], "ties": data["ties"], "rows_sha256": rows_sha,
               **results, "K4_train": train_info, "fits": fits_summary(fits),
               "cells": {"P1": P1, "PT_flex": PT, "S": [S10, S00, S11, S01], "NEST": str(paths.cell(NEST)),
                         "seeds": {str(SEED_MAIN): P1, **{str(s): str(paths.cell(n)) for s, n in SEED_CELLS.items()}}},
               "loaded_modules": repo_modules(), "mirrored_scripts": mirrored_scripts(), "environment": environment(),
               "stage": stage_record("qdev", started, tick, cpu0),
               "run": run_record(rows=rows, matches=matches, inputs=[Path(p) for p in inputs], code=[ROOT / CODE_REL, *(ROOT / m for m in MIRRORED)],
                                 seed=BOOT_SEED, started_utc=started)}
    write_json_once(target, payload)
    done.set()
    _print_split("Q_DEV", payload)
    print(f"wrote {target} sha256 {sha256_file(target)}")
    return 0


# ----------------------------------------------------------------------------- REFIT_INPUTS.json (plan 4 step 3)
def _real_paths_only(paths: Paths) -> None:
    if Path(paths.q_run).resolve() != Path(KL.RUN).resolve() or Path(paths.q_ka).resolve() != Path(KL.OUT).resolve():
        raise Blocked("the refits read kill_label.py's own q_run_1 and q_ka_1 paths (kill_label.py:58, :60)")


def refit_input_files(paths: Paths) -> tuple[dict[str, Path], dict[str, Any]]:
    """Every dependency kill_label.py:181-202 reads for the P1 cell and r2_retrain.py:118-143 (KA version) reads for K5."""

    _real_paths_only(paths)
    q, ka = paths.q_run, paths.q_ka
    specs = {}
    for name, key in (("P1", P1), ("PT", PT)):
        spec, source = KL.source_spec(key)
        specs[name] = {"key": key, "spec": spec, "source": Path(source)}
    files: dict[str, Path] = {}

    def add(label: str, path: Path | str) -> None:
        files[label] = guard(Path(path))

    add("q_run_1:jobs/JOBS.json", q / "jobs" / "JOBS.json")                         # qtest.planned_cells (qtest.py:107-113)
    add("q_run_1:Q_PURE_SET_SELECTION.json", q / "Q_PURE_SET_SELECTION.json")
    for name, entry in specs.items():
        key, source = entry["key"], entry["source"]
        add(f"q_run_1:jobs/{key}/REUSE.json", q / "jobs" / key / "REUSE.json")        # qtest.cell_record (qtest.py:116-135)
        for fname in ("SPEC.json", "RESULT.json", "model_refit.pkl", "pred_q_dev.npz"):
            if (source / fname).is_file():
                add(f"q_run_1:jobs/{source.name}/{fname}", source / fname)
        calibrator = q / "calibrators" / f"{key}.json"
        if calibrator.is_file():
            add(f"q_run_1:calibrators/{key}.json", calibrator)
    releases = {str(Path(entry["spec"]["release"])) for entry in specs.values()}
    if len(releases) != 1:
        raise Blocked(f"the two SPECs name different releases: {sorted(releases)}")
    release_path = Path(releases.pop())
    add(f"release:{release_path.name}", release_path)
    inputs = specs["P1"]["spec"]["inputs"]
    add("q_run_1:Q_ROWS_h90.npz", inputs["rows"])
    if str(Path(specs["PT"]["spec"]["inputs"]["rows"])) != str(Path(inputs["rows"])):
        add("q_run_1:PT_rows", specs["PT"]["spec"]["inputs"]["rows"])
    add("q_run_1:inputs/main_h90/XY.npy", inputs["xy"])
    add("q_run_1:inputs/main_h90/PREPARED.json", Path(inputs["xy"]).parent / "PREPARED.json")
    manifests = {}
    for scope in ("FIT", "REFIT_TRAIN"):
        directory = Path(inputs["contexts"][scope])
        for fname in ("CONTEXT.json", "CONTEXT.npz", "bundle/MANIFEST.json"):
            add(f"q_run_1:inputs/main_h90/{scope}/{fname}", directory / fname)
        manifests[scope] = read_json(directory / "bundle" / "MANIFEST.json")
        for fname in sorted(manifests[scope]["files"]):
            add(f"q_run_1:inputs/main_h90/{scope}/bundle/{fname}", directory / "bundle" / fname)
    store = Path(inputs["store"])
    names = sorted(p.name for p in store.iterdir() if p.is_file())
    for fname in names:
        add(f"q_run_1:raw/store/{fname}", store / fname)
    add("q_ka_1:KA_LABELS.npz", ka / "KA_LABELS.npz")
    add("q_ka_1:KA_PREPARE.json", ka / "KA_PREPARE.json")
    for fname in ("model_refit.pkl", "pred_q_dev.npz", "RESULT.json"):
        add(f"q_ka_1:cells/{PT}/{fname}", paths.ka_cell(PT) / fname)                # K5 offsets and the 1e-8 check
    for fname in ("pred_q_dev.npz", "RESULT.json"):
        add(f"q_ka_1:cells/{P1}/{fname}", paths.ka_cell(P1) / fname)                # K6 reproduction gate
    return files, {"specs": specs, "release_path": release_path, "inputs": inputs, "manifests": manifests, "store": store, "store_names": names}


def refit_cross_checks(record: Mapping[str, Mapping[str, Any]], ctx: Mapping[str, Any], release_digest: str) -> dict[str, bool]:
    sha = {label: entry["sha256"] for label, entry in record.items()}
    p1, pt = ctx["specs"]["P1"]["spec"], ctx["specs"]["PT"]["spec"]
    rows, xy = sha["q_run_1:Q_ROWS_h90.npz"], sha["q_run_1:inputs/main_h90/XY.npy"]
    prepared = read_json(record["q_run_1:inputs/main_h90/PREPARED.json"]["path"])
    contexts = {scope: read_json(record[f"q_run_1:inputs/main_h90/{scope}/CONTEXT.json"]["path"]) for scope in ("FIT", "REFIT_TRAIN")}
    build = sha["q_run_1:raw/store/BUILD.json"]
    ka_prepare = read_json(record["q_ka_1:KA_PREPARE.json"]["path"])
    pt_result, p1_result = read_json(record[f"q_ka_1:cells/{PT}/RESULT.json"]["path"]), read_json(record[f"q_ka_1:cells/{P1}/RESULT.json"]["path"])
    checks = {
        "rows_equal_both_spec_rows_sha256": rows == p1["inputs"]["rows_sha256"] == pt["inputs"]["rows_sha256"],
        "xy_equals_spec_and_prepared": xy == p1["inputs"]["xy_sha256"] == prepared.get("xy_sha256"),
        "prepared_rows_equal_rows": prepared.get("rows_sha256") == rows,
        "store_build_equals_prepared_and_contexts": build == prepared.get("store_build_sha256") and all(
            c.get("store_build_sha256") == build for c in contexts.values()),
        "contexts_rows_and_xy_equal": all(c.get("rows_sha256") == rows and c.get("xy_sha256") == xy for c in contexts.values()),
        "context_npz_and_manifest_equal_context_json": all(
            sha[f"q_run_1:inputs/main_h90/{s}/CONTEXT.npz"] == contexts[s].get("context_npz_sha256")
            and sha[f"q_run_1:inputs/main_h90/{s}/bundle/MANIFEST.json"] == contexts[s].get("bundle_manifest_sha256") for s in contexts),
        "bundle_files_equal_manifest": all(sha[f"q_run_1:inputs/main_h90/{s}/bundle/{f}"] == digest
                                           for s, m in ctx["manifests"].items() for f, digest in m["files"].items()),
        "release_digest_equals_both_specs": release_digest == p1["release_digest"] == pt["release_digest"],
        "ka_labels_equal_ka_prepare": sha["q_ka_1:KA_LABELS.npz"] == ka_prepare.get("ka_labels_sha256"),
        "ka_prepare_rows_equal_rows": ka_prepare.get("rows_sha256") == rows,
        "ka_pt_pred_q_dev_equals_its_result": sha[f"q_ka_1:cells/{PT}/pred_q_dev.npz"] == ((pt_result.get("scores") or {}).get("q_dev") or {}).get("pred_sha256"),
        "ka_p1_pred_q_dev_equals_its_result": sha[f"q_ka_1:cells/{P1}/pred_q_dev.npz"] == ((p1_result.get("scores") or {}).get("q_dev") or {}).get("pred_sha256"),
        "ka_cells_completed": pt_result.get("status") == "completed" == p1_result.get("status"),
        "reuse_points_to_the_source": all(read_json(record[f"q_run_1:jobs/{e['key']}/REUSE.json"]["path"]).get("reuse_of_key") == e["source"].name
                                          for e in ctx["specs"].values()),
        "sources_completed": all(read_json(record[f"q_run_1:jobs/{e['source'].name}/RESULT.json"]["path"]).get("status") == "completed"
                                 for e in ctx["specs"].values()),
    }
    return {k: bool(v) for k, v in checks.items()}


def write_refit_inputs(paths: Paths = Paths()) -> dict[str, Any]:
    """REFIT_INPUTS.json once, before any fit: every refit dependency hashed file by file and cross-checked."""

    target = guard(paths.refit_inputs)
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    started, tick, cpu0 = utc(), time.perf_counter(), time.process_time()
    files, ctx = refit_input_files(paths)
    record = files_record(files)
    payload = qrelease.verify(ctx["release_path"])
    checks = refit_cross_checks(record, ctx, payload["digest"])
    if not all(checks.values()):
        raise Blocked(f"REFIT_INPUTS cross-checks failed: {sorted(k for k, ok in checks.items() if not ok)}")
    store_sha = {name: record[f"q_run_1:raw/store/{name}"]["sha256"] for name in ctx["store_names"]}
    doc = {"version": VERSION, "experiment": EXPERIMENT, "utc": utc(), "plan": {"path": paths.plan_rel, "commit": paths.plan_commit},
           "rule": ("every dependency read by kill_label.py:181-202 (P1 cell) and by r2_retrain.py:118-143 in its KA version (K5), "
                    "hashed file by file; rechecked when every fit starts and at the lock (PLAN_KA_POSTHOC_KO.md section 4 step 3)"),
           "files": record, "cross_checks": checks,
           "release": {"path": str(ctx["release_path"]), "file_sha256": record[f"release:{ctx['release_path'].name}"]["sha256"],
                       "digest": payload["digest"], "evidence_files": len(payload["evidence"]),
                       "spec_release_digests": {name: e["spec"]["release_digest"] for name, e in ctx["specs"].items()},
                       "rule": "release.verify(path)['digest'] == spec['release_digest'] for both SPECs (kill_label.py:183-185)"},
           "store": {"dir": str(ctx["store"]), "names": ctx["store_names"], "files": len(ctx["store_names"]),
                     "bytes": int(sum(record[f"q_run_1:raw/store/{n}"]["bytes"] for n in ctx["store_names"])),
                     "tree_sha256": hashlib.sha256("".join(f"{n}:{store_sha[n]}\n" for n in ctx["store_names"]).encode("utf-8")).hexdigest(),
                     "why": "BUILD.json binds the chunk records only, not the merged store arrays (plan 4 step 3)"},
           "sources": {name: {"final_key": e["key"], "source_dir": str(e["source"]), "spec_sha256": record[f"q_run_1:jobs/{e['source'].name}/SPEC.json"]["sha256"],
                              "card": e["spec"]["job"]["card"], "card_id": e["spec"]["job"]["card_id"], "family": e["spec"]["job"]["family"],
                              "release_digest": e["spec"]["release_digest"]} for name, e in ctx["specs"].items()},
           "loaded_modules": repo_modules(), "environment": environment(), "stage": stage_record("inputs", started, tick, cpu0)}
    write_json_once(target, doc)
    return doc


def verify_refit_inputs(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Problems with a REFIT_INPUTS document now ({} when every hash, the store listing and the release still hold)."""

    problems: dict[str, Any] = {}
    changed = {}
    for label, entry in (doc.get("files") or {}).items():
        path = guard(entry["path"])
        now = sha256_file(path) if path.is_file() else None
        if now != entry["sha256"]:
            changed[label] = {"path": entry["path"], "recorded": entry["sha256"], "now": now}
    if changed:
        problems["changed"] = changed
    store = doc.get("store")
    if store:
        directory = guard(store["dir"])
        names = sorted(p.name for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []
        if names != sorted(store["names"]):
            problems["store_files_differ"] = {"missing": sorted(set(store["names"]) - set(names)), "extra": sorted(set(names) - set(store["names"]))}
    release = doc.get("release")
    if release:
        try:
            payload = qrelease.verify(guard(release["path"]))
        except Exception as error:  # noqa: BLE001  (recorded as a problem)
            problems["release"] = f"{type(error).__name__}: {error}"
        else:
            digests = {payload["digest"], release["digest"], *release.get("spec_release_digests", {}).values()}
            if len(digests) != 1:
                problems["release_digest"] = {"now": payload["digest"], "recorded": release["digest"], "specs": release.get("spec_release_digests")}
    return problems


# ----------------------------------------------------------------------------- fits (plan section 3)
def move_aside_interrupted(out: Path) -> dict[str, Any]:
    """Outputs of an interrupted attempt (no RESULT.json) go to interrupted_<k>/ instead of being overwritten;
    FIT_DONE.json (the frozen fit path's resume record) and worker.log stay."""

    out = guard(out)
    if (out / "RESULT.json").exists():
        raise FileExistsError(f"{out} has a RESULT.json: the cell is done")
    leftovers = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in ("FIT_DONE.json", "worker.log"))
    if not leftovers:
        return {"directory": None, "files": []}
    k = 1
    while (out / f"interrupted_{k}").exists():
        k += 1
    target = out / f"interrupted_{k}"
    target.mkdir()
    for name in leftovers:
        os.rename(out / name, target / name)
    return {"directory": str(target), "files": leftovers}


def seeded_fit_lightgbm(card: Mapping[str, Any], sources: Mapping[str, Any], out: Path, identity: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """qtrain.fit_lightgbm with qtrain.SEED (LightGBM random_state) set to ``seed``; nothing else changes
    (seed_replicates.py:147; plan 3.3)."""

    from studies.q_production_20261005 import qtrain

    previous = qtrain.SEED
    qtrain.SEED = int(seed)
    try:
        return qtrain.fit_lightgbm(card, sources, Path(out), identity)
    finally:
        qtrain.SEED = previous


def nest_fit(card: Mapping[str, Any], fit: tuple, stop: tuple, *, seed: int, threads: int = LGBM_THREADS) -> tuple[int, dict[str, Any]]:
    """r2_retrain.py:207-218 (P1_NEST): FIT with init_score = offset, early stopping on the STOP raw match Brier whose
    predictions include the STOP offset; the recomputed STOP Brier must equal LightGBM's within 1e-9."""

    from lightgbm import early_stopping

    from studies.q_production_20261005 import qtrain
    from studies.v_model_fit_input_20261003.models import LightGBMAdapter

    x, yy, wn, off = fit
    xs, ys, wrs, off_s = stop
    adapter = LightGBMAdapter(card=dict(card), random_state=int(seed), num_threads=int(threads))
    est = adapter.estimator
    est.set_params(metric="None")
    est.fit(x, yy, sample_weight=wn, init_score=off, eval_set=[(xs, ys)], eval_sample_weight=[wrs], eval_init_score=[off_s],
            eval_metric=qtrain.brier_eval, callbacks=[early_stopping(100, first_metric_only=True, verbose=False)])
    best = int(est.best_iteration_ or est.n_estimators)
    stop_p = qtrain.sigmoid(est.predict(xs, raw_score=True, num_iteration=best) + off_s)
    stop_brier = qtrain.match_brier(stop_p, ys, wrs)
    recorded = float(est.best_score_["valid_0"]["match_brier"])
    diff = abs(stop_brier - recorded)
    if diff > STOP_TOL:
        raise Blocked(f"STOP metric does not include the offset: {stop_brier} vs {recorded} (r2_retrain.py:217-218)")
    return best, {"best_iteration": best, "best_stop_brier": stop_brier, "best_stop_brier_lightgbm": recorded, "stop_metric_abs_diff": diff,
                  "stop_tolerance": STOP_TOL}


def nest_refit(card: Mapping[str, Any], refit: tuple, best: int, *, seed: int, threads: int = LGBM_THREADS) -> Any:
    """r2_retrain.py:219-221: REFIT on all TRAIN with the best round count and init_score = the REFIT offset."""

    from studies.v_model_fit_input_20261003.models import LightGBMAdapter

    xr, yr, wnr, off_r = refit
    model = LightGBMAdapter(card=dict(card), random_state=int(seed), n_estimators=int(best), num_threads=int(threads))
    model.estimator.fit(xr, yr, sample_weight=wnr, init_score=off_r)
    return model


def nest_fit_and_refit(card: Mapping[str, Any], fit: tuple, stop: tuple, refit: tuple, *, seed: int, threads: int = LGBM_THREADS) -> tuple[Any, dict[str, Any]]:
    best, info = nest_fit(card, fit, stop, seed=seed, threads=threads)
    return nest_refit(card, refit, best, seed=seed, threads=threads), info


def _ka_fit_inputs(spec: Mapping[str, Any]) -> SimpleNamespace:
    """kill_label.py:186-211: the KA row table (y = y_ka, weight_raw = weight_raw_ka), eligible rows of every split,
    the saved FIT / REFIT_TRAIN preprocessing contexts of the cell's SPEC and the q batch source."""

    from studies.q_production_20261005 import features as qf
    from studies.q_production_20261005 import q_worker as W
    from studies.q_production_20261005 import qbatch

    job, inputs = spec["job"], spec["inputs"]
    base = W.load_npz(W._checked(Path(inputs["rows"]), inputs["rows_sha256"], "rows"))
    rows, eligible = KL.ka_rows(base)
    part = {name: idx[eligible[idx]] for name, idx in W.splits(rows).items()}
    part["test"] = np.flatnonzero((np.asarray(rows["role"]) == KL.TEST) & eligible)
    mi, w = np.asarray(rows["match_index"]), np.asarray(rows["weight_raw"], dtype=np.float64)
    store = qf.RawStore(Path(inputs["store"]))
    xy = np.load(W._checked(Path(inputs["xy"]), inputs["xy_sha256"], "XY"), allow_pickle=False)
    cond = qbatch.Condition.from_job(SimpleNamespace(factors=job["factors"], inputs=job["inputs"]))
    ctx = {scope: qbatch.ScopeContext(store, rows, xy, scope, saved=Path(inputs["contexts"][scope])) for scope in ("FIT", "REFIT_TRAIN")}
    for scope, context in ctx.items():
        if context.record.get("rows_sha256") != inputs["rows_sha256"] or context.record.get("xy_sha256") != inputs["xy_sha256"]:
            raise ValueError(f"{scope} context was prepared from other rows or XY")

    def src(scope: str, idx: np.ndarray, weights: np.ndarray):
        return qbatch.source(ctx[scope], idx, cond, "lightgbm", W.TABLE_BATCH, weights)

    return SimpleNamespace(rows=rows, eligible=eligible, part=part, mi=mi, w=w, src=src)


def _sources(ki: SimpleNamespace) -> dict[str, Any]:
    from studies.q_production_20261005 import rows as qrows

    part, mi, w, src = ki.part, ki.mi, ki.w, ki.src
    return {"fit": src("FIT", part["fit"], qrows.normalized_weights(mi[part["fit"]])),          # kill_label.py:207-211
            "stop": src("FIT", part["stop"], w[part["stop"]]),
            "refit": src("REFIT_TRAIN", part["train"], qrows.normalized_weights(mi[part["train"]])),
            "q_dev": src("REFIT_TRAIN", part["q_dev"], w[part["q_dev"]]),
            "q_cal": src("REFIT_TRAIN", part["q_cal"], w[part["q_cal"]])}


def _part_counts(ki: SimpleNamespace) -> tuple[dict[str, int], dict[str, int]]:
    return ({k: int(len(v)) for k, v in ki.part.items()}, {k: int(np.unique(ki.mi[v]).size) for k, v in ki.part.items()})


def reproduction_check(paths: Paths, out: Path) -> dict[str, Any]:
    """Seed-20261001 refit vs the q_ka_1 P1 cell: Q_DEV logits within 1e-7 (seed_replicates.py:223-227; qtrain.py:381-383)."""

    reference = paths.ka_cell(P1)
    ref_rows, ref_logit, _, ref_sha = stored_predictions(reference / "RESULT.json", reference / "pred_q_dev.npz", "q_dev")
    mine = load_npz(Path(out) / "pred_q_dev.npz", ["row", "logit"])
    same = bool(np.array_equal(np.asarray(mine["row"], dtype=np.int64), ref_rows))
    diff = float(np.max(np.abs(np.asarray(mine["logit"], dtype=np.float64) - ref_logit))) if same and ref_rows.size else math.inf
    return {"reference": str(reference / "pred_q_dev.npz"), "reference_sha256": ref_sha, "rows_equal": same, "max_abs_qdev_logit_diff": diff,
            "tolerance": SEED_REPRO_TOL, "passed": bool(same and diff <= SEED_REPRO_TOL),
            "rule": "max |logit(refit) - logit(q_ka_1 P1)| over the Q_DEV rows (seed_replicates.py:223-227); 1e-7 (qtrain.py:381-383)",
            "used_in_analysis": False}


def fit_seed(paths: Paths, out: Path, seed: int, *, check: bool) -> dict[str, Any]:
    """K6 and the reproduction gate: kill_label.cmd_fit's LightGBM path for the P1 cell with qtrain.SEED = seed."""

    from studies.q_production_20261005 import qtrain

    _real_paths_only(paths)
    spec, source = KL.source_spec(P1)
    job = spec["job"]
    if job["family"] != "lightgbm":
        raise Blocked(f"{P1}: not a LightGBM cell")
    payload = qrelease.verify(Path(spec["release"]))
    if payload["digest"] != spec["release_digest"]:
        raise Blocked("spec names another release digest (kill_label.py:183-185)")
    ki = _ka_fit_inputs(spec)
    identity = {"job_identity": hashlib.sha256(f"{job['identity']}|label={LABEL_TAG}|seed={int(seed)}".encode()).hexdigest(),
                "variant": spec["variant"], "rows_sha256": spec["inputs"]["rows_sha256"], "release": spec["release_digest"],
                "label": LABEL_TAG, "seed": int(seed)}
    result = seeded_fit_lightgbm(job["card"], _sources(ki), out, identity, seed=seed)
    fit_done = read_json(Path(out) / "FIT_DONE.json")
    if fit_done.get("identity") != identity:
        raise Blocked("FIT_DONE.json belongs to another identity")
    result["best_stop_brier"] = fit_done.get("best_stop_brier")
    with guard(Path(out) / "model_refit.pkl").open("rb") as stream:
        adapter = pickle.load(stream)
    if int(adapter.random_state) != int(seed):
        raise Blocked(f"the refit model carries random_state {adapter.random_state}, not {seed}")
    x, _, _, _, test_rows = qtrain._stack(ki.src("REFIT_TRAIN", ki.part["test"], ki.w[ki.part["test"]]))
    logits = adapter.predict_logits(x)
    result["pred_test_sha256"] = write_npz_once(Path(out) / "pred_test.npz", {"row": np.asarray(test_rows, dtype=np.int64), "logit": logits,
                                                                              "p": qtrain.sigmoid(logits)})
    if check:
        result["reproduction"] = reproduction_check(paths, out)
    rows, matches = _part_counts(ki)
    result.update(kind="K6 seed refit" if not check else "seed-20261001 reproduction check", seed=int(seed), identity=identity,
                  seed_mechanism="qtrain.SEED (LightGBM random_state) set before the unchanged qtrain.fit_lightgbm (seed_replicates.py:147)",
                  source_dir=str(source), source_spec_sha256=sha256_file(Path(source) / "SPEC.json"), release_digest=payload["digest"],
                  card=job["card"], card_id=job["card_id"], rows=rows, matches=matches,
                  test_scoring="TEST eligible rows scored without labels after the fit (kill_label.py:215-216, :235-236); not evaluated here")
    return result


def ka_pt_models(paths: Paths, rows: Mapping[str, np.ndarray], part: Mapping[str, np.ndarray]) -> tuple[Callable, Callable, dict[str, Any]]:
    """KA version of r2_retrain.pt_models (:118-143): FIT/STOP offsets from a KA PT_flex (same card) refitted on the
    eligible FIT rows; REFIT/scoring offsets from the q_ka_1 PT_flex REFIT model, which must reproduce its stored KA
    Q_DEV logits within 1e-8."""

    from studies.q_production_20261005 import rows as qrows

    spec, source = KL.source_spec(PT)
    card = dict(spec["job"]["card"])
    y, t, p = (np.asarray(rows[k], dtype=np.float64) for k in ("y", "elapsed_min", "pt_p_pre"))
    mi = np.asarray(rows["match_index"])
    fit = part["fit"]
    model_fit, design_fit = qbaselines.fit_reference("PT_flex", card, {"p": p[fit], "t": t[fit], "y": y[fit], "w_norm": qrows.normalized_weights(mi[fit])})
    pt_dir = paths.ka_cell(PT)
    with guard(pt_dir / "model_refit.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    model_ref, design_ref = saved["model"], saved["design"]
    stored = load_npz(pt_dir / "pred_q_dev.npz", ["row", "logit"])
    check = np.asarray(model_ref.predict_logits(design_ref(p[stored["row"]], t[stored["row"]])), dtype=np.float64)
    diff = float(np.max(np.abs(check - np.asarray(stored["logit"], dtype=np.float64))))
    if diff > PT_REFIT_TOL:
        raise Blocked(f"KA PT_flex REFIT model does not reproduce its stored Q_DEV logits: {diff} > {PT_REFIT_TOL}")

    def lf(idx: np.ndarray) -> np.ndarray:
        return np.asarray(model_fit.predict_logits(design_fit(p[idx], t[idx])), dtype=np.float64)

    def lr(idx: np.ndarray) -> np.ndarray:
        return np.asarray(model_ref.predict_logits(design_ref(p[idx], t[idx])), dtype=np.float64)

    info = {"fit_stop_offset": "KA PT_flex refitted on the eligible FIT rows with the q_run_1 PT_flex card (normalized FIT match weights)",
            "fit_offset_rows": int(fit.size), "card": card, "pt_source_spec": str(source),
            "refit_offset_model": str(pt_dir / "model_refit.pkl"), "refit_offset_model_sha256": sha256_file(pt_dir / "model_refit.pkl"),
            "refit_offset_check": {"max_abs_qdev_logit_diff": diff, "tolerance": PT_REFIT_TOL, "rows": int(np.asarray(stored["row"]).size)}}
    return lf, lr, info


def fit_nest(paths: Paths, out: Path) -> dict[str, Any]:
    """K5: r2_retrain.cmd_fit NEST branch (:205-235) on the KA rows with KA PT_flex offsets; TEST scored in this command."""

    from studies.q_production_20261005 import qtrain

    _real_paths_only(paths)
    key, mode = R2.VARIANTS[NEST]
    spec, source = KL.source_spec(key)
    job = spec["job"]
    if job["family"] != "lightgbm" or mode != "NEST":
        raise Blocked(f"{key}: not the LightGBM P1 cell of the NEST variant")
    payload = qrelease.verify(Path(spec["release"]))
    if payload["digest"] != spec["release_digest"]:
        raise Blocked("spec names another release digest (r2_retrain.py:166-168)")
    ki = _ka_fit_inputs(spec)
    sources = _sources(ki)
    lf, lr, offsets = ka_pt_models(paths, ki.rows, ki.part)
    x, yy, wn, _, r_fit = qtrain._stack(sources["fit"])
    xs, ys, _, wrs, r_stop = qtrain._stack(sources["stop"])
    best, info = nest_fit(job["card"], (x, yy, wn, lf(r_fit)), (xs, ys, wrs, lf(r_stop)), seed=qtrain.SEED)
    del x, yy, wn, xs, ys, wrs
    xr, yr, wnr, _, r_ref = qtrain._stack(sources["refit"])
    model = nest_refit(job["card"], (xr, yr, wnr, lr(r_ref)), best, seed=qtrain.SEED)
    del xr, yr, wnr
    out = Path(out)
    model_sha = write_pickle_once(out / "model_refit.pkl", model)
    scores, reload_diff = {}, None
    for name in ("q_dev", "q_cal"):
        xq, yq, _, wq, rq = qtrain._stack(sources[name])
        lg = model.estimator.predict(xq, raw_score=True) + lr(rq)
        sha = write_npz_once(out / f"pred_{name}.npz", {"row": rq, "logit": lg, "p": qtrain.sigmoid(lg)})
        scores[name] = {"rows": int(len(rq)), "raw_match_brier": qtrain.match_brier(qtrain.sigmoid(lg), yq, wq), "pred_sha256": sha}
        if name == "q_dev":                                  # the saved model must give the stored Q_DEV logits again
            with guard(out / "model_refit.pkl").open("rb") as stream:
                reloaded = pickle.load(stream)
            reload_diff = float(np.max(np.abs(reloaded.estimator.predict(xq, raw_score=True) + lr(rq) - lg)))
            if reload_diff > RELOAD_TOL:
                raise Blocked(f"NEST reload differs by {reload_diff}")
        del xq
    x, _, _, _, test_rows = qtrain._stack(ki.src("REFIT_TRAIN", ki.part["test"], ki.w[ki.part["test"]]))
    logits = model.estimator.predict(x, raw_score=True) + lr(test_rows)
    pred_test = write_npz_once(out / "pred_test.npz", {"row": np.asarray(test_rows, dtype=np.int64), "logit": logits, "p": qtrain.sigmoid(logits)})
    rows, matches = _part_counts(ki)
    identity = {"job_identity": hashlib.sha256(f"{job['identity']}|review={NEST}|label={LABEL_TAG}".encode()).hexdigest(), "variant": NEST,
                "rows_sha256": spec["inputs"]["rows_sha256"], "release": spec["release_digest"], "label": LABEL_TAG}
    return {"status": "completed", "kind": "K5 nested refit", "variant": NEST, "mode": mode, "seed": int(qtrain.SEED), "identity": identity,
            "best_iteration": info["best_iteration"], "best_stop_brier": info["best_stop_brier"], "stop_metric_check": info,
            "scores": scores, "model_sha256": model_sha, "pred_test_sha256": pred_test,
            "reload": {"max_abs_logit_diff": reload_diff, "tolerance": RELOAD_TOL, "note": "extra integrity check of model_refit.pkl"},
            "offsets": offsets, "source_dir": str(source), "source_spec_sha256": sha256_file(Path(source) / "SPEC.json"),
            "release_digest": payload["digest"], "card": job["card"], "card_id": job["card_id"], "rows": rows, "matches": matches,
            "mirrors": "reports/review_followup_20261011/r2_retrain.py P1_NEST (:205-233), KA rows and KA PT_flex offsets",
            "test_scoring": "TEST eligible rows scored in this command without labels (plan 3.2); not evaluated here"}


def worker_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """One training process: LightGBM 3 threads (frozen) + BLAS 1 = 4; GPU hidden (plan 3.4)."""

    env = dict(os.environ if base is None else base)
    env.update({"PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE", "CUDA_VISIBLE_DEVICES": ""})
    env.update({name: "1" for name in THREAD_VARIABLES})
    return env


def _fit_timeout(path: Path, cell: str, started: str, tick: float) -> None:
    try:
        write_json_once(path, {"status": "failed", "reason": "time_limit", "cell": cell, "limit_s": FIT_LIMIT_S, "started_utc": started,
                               "finished_utc": utc(), "seconds": round(time.perf_counter() - tick, 1), "pid": os.getpid()})
    except FileExistsError:
        pass


def fit_stage(cell: str, paths: Paths = Paths(), *, time_limit_s: float | None = FIT_LIMIT_S) -> int:
    """One refit (plan section 3): REFIT_INPUTS.json recheck, the fit, RESULT.json once."""

    if cell not in FIT_ORDER:
        raise ValueError(f"unknown cell {cell!r}")
    out = guard(paths.cell(cell))
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "RESULT.json"
    if result_path.exists():
        return 0 if read_json(result_path).get("status") == "completed" else 1
    started, tick, cpu0 = utc(), time.perf_counter(), time.process_time()
    done = watchdog(time_limit_s, lambda: _fit_timeout(result_path, cell, started, tick))
    moved = move_aside_interrupted(out)
    refit = {"path": str(paths.refit_inputs)}
    try:
        doc = read_json(paths.refit_inputs)
        refit.update(sha256=sha256_file(paths.refit_inputs))
        check_tick = time.perf_counter()
        problems = verify_refit_inputs(doc)
        refit.update(recheck_problems=problems, recheck_seconds=round(time.perf_counter() - check_tick, 1), files=len(doc.get("files") or {}))
        if problems:
            raise Blocked(f"REFIT_INPUTS.json no longer holds: {sorted(problems)}")
        result = fit_nest(paths, out) if cell == NEST else fit_seed(paths, out, FIT_SEEDS[cell], check=cell == CHECK)
    except Exception as error:  # noqa: BLE001  (recorded; the launcher stops on a failed RESULT.json)
        result = {"status": "failed", "reason": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    import lightgbm

    result.update(version=VERSION, experiment=EXPERIMENT, cell=cell, label=LABEL_TAG, refit_inputs=refit, interrupted_outputs=moved,
                  started_utc=started, finished_utc=utc(), seconds=round(time.perf_counter() - tick, 1), pid=os.getpid(),
                  threads={"lightgbm_num_threads": LGBM_THREADS, "env": {name: os.environ.get(name) for name in THREAD_VARIABLES}},
                  environment={**environment(), "lightgbm": lightgbm.__version__}, loaded_modules=repo_modules(),
                  mirrored_scripts=mirrored_scripts(), stage=stage_record(f"fit:{cell}", started, tick, cpu0))
    done.set()
    write_json_once(result_path, result)
    print(json.dumps({"cell": cell, "status": result["status"], "seconds": result["seconds"]}), flush=True)
    return 0 if result["status"] == "completed" else 1


# ----------------------------------------------------------------------------- stage: launch (plan 3.4)
def commit_bytes() -> int:
    """System commit charge (\\Memory\\Committed Bytes) via GetPerformanceInfo."""

    import ctypes
    from ctypes import wintypes

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise OSError("GetPerformanceInfo failed")
    return int(info.CommitTotal) * int(info.PageSize)


def _ka_processes(*markers: str) -> list[int]:
    import psutil

    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = [str(a) for a in (proc.info.get("cmdline") or [])]
        except Exception:  # noqa: BLE001
            continue
        if any(a.replace("\\", "/").endswith("tog_experiments_20261012/ka_posthoc.py") for a in cmd) and all(m in cmd for m in markers):
            pids.append(int(proc.info["pid"]))
    return pids


def find_other_launcher() -> int | None:
    others = [pid for pid in _ka_processes("launch") if pid != os.getpid()]
    return others[0] if others else None


class _Adopted:
    """A live fit worker left by an earlier launcher, known by its PID."""

    def __init__(self, pid: int) -> None:
        import psutil

        self.pid = int(pid)
        self._proc = psutil.Process(self.pid)

    def poll(self) -> int | None:
        import psutil

        try:
            if self._proc.is_running() and self._proc.status() != psutil.STATUS_ZOMBIE:
                return None
        except Exception:  # noqa: BLE001  (a vanished process has ended)
            pass
        return -1

    def kill(self) -> None:
        try:
            self._proc.kill()
        except Exception:  # noqa: BLE001
            pass


def adopt_worker(cell: str) -> _Adopted | None:
    pids = _ka_processes("fit", "--cell", cell)
    return _Adopted(pids[0]) if pids else None


def start_worker(paths: Paths, cell: str) -> subprocess.Popen:
    directory = guard(paths.cell(cell))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.log").open("ab") as stream:
        return subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "fit", "--cell", cell], cwd=str(ROOT), env=worker_env(),
                                stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _log(paths: Paths, message: str) -> None:
    paths.out.mkdir(parents=True, exist_ok=True)
    with guard(paths.launch_log).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{utc()} {message}\n")


def cell_state(paths: Paths, cell: str) -> str:
    path = guard(paths.cell(cell) / "RESULT.json")
    if not path.is_file():
        return "pending"
    return "completed" if read_json(path).get("status") == "completed" else "failed"


def _launch_blocked(paths: Paths, reason: str, **details: Any) -> int:
    _log(paths, f"BLOCKED {reason}")
    return _blocked(paths, "launch", {"reason": reason, **details})


def launch_stage(paths: Paths = Paths(), *, inputs_only: bool = False, poll_s: float = 15.0, starter: Callable | None = None,
                 commit: Callable[[], int] | None = None, other_launcher: Callable[[], int | None] | None = None,
                 adopt: Callable[[str], Any] | None = None, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None) -> int:
    """REFIT_INPUTS.json (once), then the four fits one at a time.  The supervisor only watches: it starts a worker,
    polls it, stops it at the time limit, and never starts a fit above the commit threshold (plan 3.4)."""

    starter, commit = starter or start_worker, commit or commit_bytes
    other_launcher, adopt = other_launcher or find_other_launcher, adopt or adopt_worker
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    paths.out.mkdir(parents=True, exist_ok=True)
    other = other_launcher()
    if other:
        _log(paths, f"launch pid={os.getpid()} refused: launcher pid {other} is running")
        return 2
    _log(paths, f"launch pid={os.getpid()} inputs_only={inputs_only} order={list(FIT_ORDER)} slots=1 lightgbm_threads={LGBM_THREADS} blas_threads=1")
    if not guard(paths.refit_inputs).exists():
        _log(paths, "writing REFIT_INPUTS.json")
        try:
            doc = write_refit_inputs(paths)
        except Blocked as error:
            return _launch_blocked(paths, f"REFIT_INPUTS: {error}")
        _log(paths, f"REFIT_INPUTS.json written: {len(doc['files'])} files, store {doc['store']['files']} files {doc['store']['bytes']} bytes, "
                    f"sha256 {sha256_file(paths.refit_inputs)}")
    else:
        _log(paths, f"REFIT_INPUTS.json exists (sha256 {sha256_file(paths.refit_inputs)})")
    if inputs_only:
        _log(paths, "inputs only: done")
        return 0
    for cell in FIT_ORDER:
        state = cell_state(paths, cell)
        if state == "pending":
            handle = adopt(cell)
            if handle is not None:
                _log(paths, f"adopted {cell} pid={handle.pid}")
            else:
                charge = commit()
                if charge > COMMIT_LIMIT_BYTES:
                    return _launch_blocked(paths, f"system commit {charge / 2**30:.1f} GB above 95.5 GB: {cell} not started")
                handle = starter(paths, cell)
                _log(paths, f"started {cell} pid={handle.pid} commit={charge / 2**30:.1f}GB")
            begun, warned = clock(), False
            code = handle.poll()
            while code is None:
                if clock() - begun > FIT_LIMIT_S + KILL_GRACE_S:
                    handle.kill()
                    return _launch_blocked(paths, f"{cell} exceeded the {FIT_LIMIT_S:.0f} s fit limit; killed pid={handle.pid}")
                charge = commit()
                if charge > COMMIT_LIMIT_BYTES and not warned:
                    warned = True
                    _log(paths, f"system commit {charge / 2**30:.1f} GB above 95.5 GB while {cell} runs: no new fit will start")
                sleep(poll_s)
                code = handle.poll()
            _log(paths, f"ended {cell} code={code}")
            state = cell_state(paths, cell)
            if state == "pending":
                return _launch_blocked(paths, f"{cell} ended without RESULT.json")
        if state == "failed":
            return _launch_blocked(paths, f"{cell} failed (RESULT.json)", result=str(paths.cell(cell) / "RESULT.json"))
        if cell == CHECK:
            reproduction = read_json(paths.cell(cell) / "RESULT.json").get("reproduction") or {}
            if reproduction.get("passed") is not True:
                return _launch_blocked(paths, f"seed-{SEED_MAIN} reproduction check failed", reproduction=reproduction)
            _log(paths, f"reproduction check passed: max |diff| {reproduction.get('max_abs_qdev_logit_diff')}")
        _log(paths, f"{cell} completed")
    _log(paths, "launch done")
    return 0


# ----------------------------------------------------------------------------- stage: lock (plan 4 step 5)
def lock_versions(out: Path) -> list[Path]:
    found, base = [], Path(out) / f"{LOCK_BASE}.json"
    if guard(base).exists():
        found.append(base)
        k = 1
        while (Path(out) / f"{LOCK_BASE}_r{k}.json").exists():
            found.append(Path(out) / f"{LOCK_BASE}_r{k}.json")
            k += 1
    return found


def latest_lock(out: Path) -> Path | None:
    versions = lock_versions(out)
    return versions[-1] if versions else None


def reanalysis_files(paths: Paths) -> dict[str, Path]:
    files = {"input:KA_LABELS.npz": paths.ka_labels, "input:KA_PREPARE.json": paths.ka_prepare, "input:Q_ROWS_h90.npz": paths.rows,
             "input:Q_ROWS_h90.json": paths.rows_manifest, "input:KA_REPORT.json": paths.ka_report,
             "input:PARTICIPANTS_KILLCREDIT.npz": paths.participants, "input:PARTICIPANTS_KILLCREDIT.json": paths.participants_manifest}
    for key in KA_CELLS:
        for name in ("pred_q_dev.npz", "pred_test.npz", "RESULT.json", "model_refit.pkl"):
            files[f"q_ka_1:{key}:{name}"] = paths.ka_cell(key) / name
    return files


def fit_files(paths: Paths) -> dict[str, Path]:
    files = {}
    for cell in FIT_ORDER:
        names = ["RESULT.json", "model_refit.pkl", "pred_q_dev.npz", "pred_q_cal.npz", "pred_test.npz"]
        if cell != NEST:
            names.append("FIT_DONE.json")                       # only the three qtrain.fit_lightgbm fits (plan 6)
        files.update({f"fit:{cell}:{name}": paths.cell(cell) / name for name in names})
    return files


def python_definitions_diff(old_source: str, new_source: str) -> dict[str, Any]:
    """Top-level definitions (functions, classes) whose AST differs between two versions of a module, plus whether
    any module-level statement differs (comments and formatting are ignored; docstrings count)."""

    import ast

    def parts(source: str) -> tuple[dict[str, str], list[str]]:
        defs, module = {}, []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[node.name] = ast.dump(node)
            else:
                module.append(ast.dump(node))
        return defs, module

    old_defs, old_module = parts(old_source)
    new_defs, new_module = parts(new_source)
    return {"module_level_changed": old_module != new_module,
            "changed": sorted(name for name in old_defs.keys() & new_defs.keys() if old_defs[name] != new_defs[name]),
            "added": sorted(new_defs.keys() - old_defs.keys()), "removed": sorted(old_defs.keys() - new_defs.keys())}


def code_revision(repo: Path, rel: str, stage_sha256: str, *,
                  allowed: tuple[str, ...] = ("test_stage", "lock_stage", "warm_up_analysis", "test_preflight", "test_split_data",
                                               "check_gate_reference_structure", "check_reference_block", "check_prediction_arrays",
                                               "load_test_predictions", "reproduce_qdev", "modules_outside_lock",
                                               "python_definitions_diff", "code_revision")) -> dict[str, Any]:
    """A module revised after a stage ran.  ``problems`` (binding): the stage's version must be a commit in git (the
    commit whose blob has the stage's SHA-256).  ``confinement_findings`` (a secondary record, Ruling 67): the
    definition-level comparison with the file now - module-level code changed, or changed / added / removed definitions
    outside the TEST and lock steps.  An AST comparison cannot prove that analysis behaviour is unchanged (an allowed
    definition can still rebind a global or replace a routine); the binding behavioural check is the TEST step's exact
    Q_DEV reproduction before the opening (reproduce_qdev)."""

    repo = Path(repo)
    record: dict[str, Any] = {"path": rel, "stage_sha256": stage_sha256, "stage_commit": None, "now_sha256": sha256_file(repo / rel),
                              "allowed": list(allowed), "problems": [], "confinement_findings": []}
    old = None
    for commit in _git(repo, "log", "--format=%H", "--", rel).stdout.decode("ascii", "replace").split():
        blob = _git(repo, "cat-file", "blob", f"{commit}:{rel}")
        if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == stage_sha256:
            record["stage_commit"], old = commit, blob.stdout
            break
    if old is None:
        record["problems"].append(f"{rel}: no commit holds the version a stage ran (SHA-256 {stage_sha256})")
        return record
    diff = python_definitions_diff(old.decode("utf-8"), guard(repo / rel).read_bytes().decode("utf-8"))
    record.update(diff)
    if diff["module_level_changed"]:
        record["confinement_findings"].append(f"{rel}: module-level code changed after a stage ran")
    outside = sorted((set(diff["changed"]) | set(diff["added"]) | set(diff["removed"])) - set(allowed))
    if outside:
        record["confinement_findings"].append(f"{rel}: definitions outside the TEST and lock steps changed after a stage ran: {outside}")
    return record


def lock_stage(paths: Paths = Paths(), *, new_version: bool = False, require_committed_code: bool = True) -> int:
    """KA_POSTHOC_LOCK.json (next versions _r1, ...): plan blob, code, refit and re-analysis inputs, the four fits and
    the Q_DEV results, hashed; the refit inputs rechecked; code committed; every module unchanged since its stage ran,
    except a ka_posthoc.py revision whose stage version is a git commit (code_revision; its definition-level comparison
    is a secondary record).  The binding check of a revision is the TEST step's exact Q_DEV reproduction (Ruling 67)."""

    versions = lock_versions(paths.out)
    if versions and not new_version:
        raise FileExistsError(f"{versions[-1]} exists; pass --new-version to write the next lock version")
    target = paths.out / (f"{LOCK_BASE}.json" if not versions else f"{LOCK_BASE}_r{len(versions)}.json")
    started, tick, cpu0 = utc(), time.perf_counter(), time.process_time()
    try:
        fits = completed_fits(paths)
        if not guard(paths.qdev).is_file():
            raise Blocked("no KA_POSTHOC_QDEV.json: run qdev first")
        qdev = read_json(paths.qdev)
        plan = plan_record(paths)
        if plan["last_commit_of_path"] != paths.plan_commit or not plan["working_copy_unchanged_since_commit"]:
            raise Blocked(f"the plan changed after commit {paths.plan_commit}: {plan}")
        refit_doc = read_json(paths.refit_inputs)
        refit_tick = time.perf_counter()
        recheck = verify_refit_inputs(refit_doc)
        if recheck:
            raise Blocked(f"REFIT_INPUTS.json no longer holds: {recheck}")
        refit_seconds = time.perf_counter() - refit_tick
        stages = {"inputs": refit_doc.get("loaded_modules") or {}, "qdev": qdev.get("loaded_modules") or {},
                  **{f"fit:{cell}": fits[cell].get("loaded_modules") or {} for cell in FIT_ORDER}}
        now = repo_modules()
        stages["lock"] = now
        names = sorted({CODE_REL, TEST_REL, *MIRRORED, *now, *(rel for mods in stages.values() for rel in mods)})
        code = {rel: sha256_file(ROOT / rel) for rel in names}
        changed = {stage: sorted(rel for rel, sha in mods.items() if code.get(rel) != sha) for stage, mods in stages.items()}
        changed = {stage: rels for stage, rels in changed.items() if rels}
        revisions: dict[str, dict[str, Any]] = {}
        revision_problems: list[str] = []
        for stage, rels in changed.items():
            for rel in rels:
                if rel != CODE_REL:
                    revision_problems.append(f"{stage}: {rel} changed after the stage ran")
                    continue
                stage_sha = stages[stage][rel]
                if stage_sha not in revisions:
                    revisions[stage_sha] = code_revision(ROOT, rel, stage_sha)
                    revision_problems += revisions[stage_sha]["problems"]
                revisions[stage_sha].setdefault("stages", []).append(stage)
        if revision_problems:
            raise Blocked(f"code changed after a stage ran beyond the TEST and lock steps: {revision_problems}")
        git = code_git(names)
        commit_problems = code_commit_problems(git)
        if require_committed_code and commit_problems:
            raise Blocked(f"commit the code before the lock: {commit_problems}")
        files = {**reanalysis_files(paths), **fit_files(paths), "refit_inputs": paths.refit_inputs, "qdev": paths.qdev,
                 **{f"code:{rel}": ROOT / rel for rel in names}}
        verify = {label: {"path": str(guard(path)), "sha256": sha256_file(path)} for label, path in files.items()}
        mismatch = []
        for cell, result in fits.items():
            for label, recorded in (("pred_q_dev.npz", ((result.get("scores") or {}).get("q_dev") or {}).get("pred_sha256")),
                                    ("pred_q_cal.npz", ((result.get("scores") or {}).get("q_cal") or {}).get("pred_sha256")),
                                    ("pred_test.npz", result.get("pred_test_sha256")), ("model_refit.pkl", result.get("model_sha256"))):
                if verify[f"fit:{cell}:{label}"]["sha256"] != recorded:
                    mismatch.append(f"{cell}:{label}")
        if mismatch:
            raise Blocked(f"fit files differ from their RESULT.json: {mismatch}")
    except Blocked as error:
        return _blocked(paths, "lock", {"reason": str(error)})
    openings = read_json(paths.openings).get("openings", []) if guard(paths.openings).is_file() else []
    payload = {"version": VERSION, "experiment": EXPERIMENT, "locked_utc": utc(), "label": POSTHOC,
               "plan": plan, "previous_locks": [{"path": str(p), "sha256": sha256_file(p)} for p in versions],
               "code": {"files": code, "stages": stages, "git": git, "commit_problems": commit_problems,
                        "revisions": list(revisions.values()),
                        "revisions_rule": ("a module may differ from the version a stage ran only if it is ka_posthoc.py and the stage's "
                                           "version is a commit in git (binding); the definition-level comparison (confinement_findings) "
                                           "is a secondary record (Ruling 67)"),
                        "binding_check": ("before the OPENINGS.json entry the TEST step recomputes the gate and K1-K6 on Q_DEV with the "
                                          "locked code from the saved Q_DEV predictions (q_ka_1 cells and the four refits; nothing is "
                                          "retrained) and requires every output to equal the lock-bound KA_POSTHOC_QDEV.json exactly "
                                          "(canonical JSON of each output); any difference stops the run before exposure (reproduce_qdev)"),
                        "untracked_modules": git["untracked"],
                        "untracked_note": ("modules under folders this repository does not track (frozen work of other sessions, not "
                                           "this task's to commit); they are bound by their disk SHA-256 in 'files' and 'verify'"),
                        "rule": ("disk bytes of ka_posthoc.py, its test, the mirrored scripts and every repository module loaded by the "
                                 "inputs, fit, qdev and lock processes; each equals the hash its stage recorded")},
               "refit_inputs": {"path": str(paths.refit_inputs), "sha256": verify["refit_inputs"]["sha256"], "recheck": recheck,
                                "recheck_seconds": round(refit_seconds, 1), "files": len(refit_doc.get("files") or {})},
               "qdev": {"path": str(paths.qdev), "sha256": verify["qdev"]["sha256"], "utc": (qdev.get("stage") or {}).get("finished_utc")},
               "fits": fits_summary(fits),
               "exposure_ledger": {"path": str(paths.exposure), "sha256": sha256_file(paths.exposure),
                                   "note": "recorded, not bound: the planned repeated-look row with this lock's SHA-256 is appended after the lock"},
               "openings_before_lock": [o for o in openings if o.get("experiment") == EXPERIMENT],
               "test_step": {"labels": str(paths.ka_labels), "openings": str(paths.openings), "note": OPENING_NOTE,
                             "order": ["lock recheck (files, plan blob, REFIT_INPUTS, loaded modules, EXPOSURE row)", "OPENINGS.json entry saved",
                                       "KA TEST labels read", "gate T.kk (1e-9)", "K1-K6 TEST", "KA_POSTHOC_TEST.json once"]},
               "verify": verify, "environment": environment(), "stage": stage_record("lock", started, tick, cpu0)}
    write_json_once(target, payload)
    print(json.dumps({"lock": str(target), "sha256": sha256_file(target), "files": len(verify), "plan_blob_sha256": plan["blob_sha256"]}), flush=True)
    return 0


# ----------------------------------------------------------------------------- stage: test (plan 4 step 7; part B)
def test_result_path(out: Path, attempt: int) -> Path:
    return Path(out) / ("KA_POSTHOC_TEST.json" if int(attempt) == 1 else f"KA_POSTHOC_TEST_a{int(attempt)}.json")


def test_failure_path(out: Path, attempt: int) -> Path:
    return Path(out) / ("KA_POSTHOC_TEST_FAILED.json" if int(attempt) == 1 else f"KA_POSTHOC_TEST_a{int(attempt)}_FAILED.json")


def ka_openings(paths: Paths) -> list[dict[str, Any]]:
    if not guard(paths.openings).is_file():
        return []
    return [o for o in read_json(paths.openings).get("openings", []) if o.get("experiment") == EXPERIMENT]


def append_opening(path: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append the TEST step's entry to OPENINGS.json and make sure it is on disk; returns the earlier openings."""

    path = guard(path)
    doc = read_json(path) if path.is_file() else {"rule": ("EXPOSURE_KO.md rule 1: every ToG experiment that opens sealed labels appends "
                                                         "experiment, UTC time and lock SHA-256"), "openings": []}
    before = list(doc.get("openings", []))
    attempts = [int(o.get("attempt", 0)) for o in before if o.get("experiment") == entry["experiment"]]
    if int(entry["attempt"]) != len(attempts) + 1:
        raise RuntimeError(f"attempt {entry['attempt']} does not follow the recorded attempts {attempts}")
    doc["openings"] = before + [dict(entry)]
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        stream.write((json.dumps(jsonable(doc), indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    if not opening_saved(SimpleNamespace(openings=path), entry):
        raise RuntimeError("the OPENINGS.json entry is not on disk")
    return before


def verify_lock(lock: Mapping[str, Any], lock_sha: str, paths: Paths) -> dict[str, Any]:
    """Why the TEST step may not start (empty when the lock holds)."""

    problems: dict[str, Any] = {}
    plan = lock.get("plan") or {}
    if not plan.get("blob_sha256"):
        problems["plan_hash_missing"] = True
    else:
        try:
            now = plan_record(dataclasses.replace(paths, plan_commit=plan["commit"], plan_rel=plan["path"]))
        except Blocked as error:
            problems["plan"] = str(error)
        else:
            if now["blob_sha256"] != plan["blob_sha256"] or now["last_commit_of_path"] != plan["commit"] or not now["working_copy_unchanged_since_commit"]:
                problems["plan_changed_since_lock"] = {"locked": plan.get("blob_sha256"), "now": now}
    changed = {}
    for label, entry in (lock.get("verify") or {}).items():
        path = guard(entry["path"])
        now_sha = sha256_file(path) if path.is_file() else None
        if now_sha != entry["sha256"]:
            changed[label] = {"path": entry["path"], "locked": entry["sha256"], "now": now_sha}
    if changed:
        problems["changed_since_lock"] = changed
    refit_path = (lock.get("refit_inputs") or {}).get("path")
    if refit_path:
        recheck = verify_refit_inputs(read_json(refit_path))
        if recheck:
            problems["refit_inputs"] = recheck
    code = (lock.get("code") or {}).get("files") or {}
    loaded = repo_modules()
    outside = {rel: sha for rel, sha in loaded.items() if code.get(rel) != sha}
    if outside:
        problems["loaded_modules_not_in_lock"] = outside
    text = guard(paths.exposure).read_text(encoding="utf-8") if guard(paths.exposure).is_file() else ""
    if lock_sha not in text:
        problems["exposure_row_missing"] = f"EXPOSURE_KO.md has no planned repeated-look row with {lock_sha} (plan 4 step 6)"
    return problems


def warm_up_analysis() -> None:
    """Run the K1-K6 code once on small synthetic rows, so every module the TEST analysis needs is loaded before the
    lock check and none first loads after the opening (fix round 1)."""

    rng = np.random.default_rng(20261014)
    per = np.tile([1, 2, 3, 4], 30)
    match = np.repeat(np.arange(per.size) * 3 + 1, per)
    n = match.size
    y = np.tile([0.0, 1.0], n // 2 + 1)[:n]
    rng.shuffle(y)
    preds = {key: qtest.sigmoid((0.3 + 0.1 * (i % 4)) * (2 * y - 1) + rng.normal(0.0, 1.0, n))
             for i, key in enumerate([*KA_CELLS, NEST, *SEED_CELLS.values()])}
    kills = np.tile(np.arange(1, 7, dtype=np.int16), n // 6 + 1)[:n]
    data = {"y": y, "match": match, "preds": preds, "n_kills": kills, "split_n_kills": kills, "ties": 0,
            "age": (np.arange(n) % 10) + 0.5, "minutes": rng.uniform(2.0, 40.0, n), "p_pre": rng.uniform(0.05, 0.95, n),
            "cohort": (np.arange(n) % 4).astype(np.uint8), "t5": np.arange(n) % 8 == 0,
            "n_blue": np.arange(n) % 6, "n_red": (np.arange(n) + 3) % 6}
    analyse("warm-up", data, jsonable(KL._contrast_block(y, {k: preds[k] for k in KA_CELLS}, match)), rate=0.5)


def check_reference_block(block: Any) -> list[str]:
    """Entries of a KA_REPORT ``ka_trained_on_ka`` block the gate could not compare (fix round 2): every cell's AUC and
    Brier and every contrast's point must be a finite number (not a boolean), and every interval two finite numbers
    with lo <= hi.  Returns the bad entries (empty when the block is usable)."""

    def get(mapping: Any, key: str) -> Any:
        return mapping.get(key) if isinstance(mapping, Mapping) else None

    def finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    bad = []
    for cell in KA_CELLS:
        for metric in ("auc", "brier"):
            if not finite(get(get(get(block, "metrics"), cell), metric)):
                bad.append(f"metrics/{cell}/{metric}")
    for name in KL.CONTRASTS:
        entry = get(get(get(block, "contrasts"), "contrasts"), name)
        if not finite(get(entry, "point")):
            bad.append(f"contrasts/{name}/point")
        interval = get(entry, "interval")
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2 and all(finite(v) for v in interval) and interval[0] <= interval[1]):
            bad.append(f"contrasts/{name}/interval")
    return bad


def check_gate_reference_structure(paths: Paths, split: str) -> dict[str, Any]:
    """The KA_REPORT block the gate will compare with has every entry it needs as a finite number, intervals ordered
    (check_reference_block); the values are compared with the recomputation only after the opening."""

    block = ((read_json(paths.ka_report).get("splits") or {}).get(split) or {}).get("ka_trained_on_ka") or {}
    bad = check_reference_block(block)
    if bad:
        raise Blocked(f"KA_REPORT splits.{split}.ka_trained_on_ka has unusable entries: {bad}")
    return {"entries_checked": 2 * len(KA_CELLS) + 3 * len(KL.CONTRASTS),
            "rule": "finite numbers (not booleans); intervals two finite numbers with lo <= hi"}


def check_prediction_arrays(key: str, arrays: Mapping[str, Any]) -> int:
    """A prediction file's raw arrays before any conversion (fix round 2): row, logit and p present, one-dimensional and
    of one length; row an integer dtype, logit and p floating dtypes; logit finite and p finite in [0, 1].  Returns the
    length."""

    missing = [name for name in ("row", "logit", "p") if name not in arrays]
    if missing:
        raise Blocked(f"{key}: prediction arrays missing: {missing}")
    row, logit, p = (np.asarray(arrays[name]) for name in ("row", "logit", "p"))
    if row.ndim != 1 or logit.ndim != 1 or p.ndim != 1:
        raise Blocked(f"{key}: prediction arrays are not one-dimensional: row {row.shape}, logit {logit.shape}, p {p.shape}")
    if not row.size == logit.size == p.size:
        raise Blocked(f"{key}: prediction arrays differ in length: row {row.size}, logit {logit.size}, p {p.size}")
    if row.dtype.kind not in "iu" or logit.dtype.kind != "f" or p.dtype.kind != "f":
        raise Blocked(f"{key}: prediction dtypes are not integer rows with floating logit and p: {row.dtype}, {logit.dtype}, {p.dtype}")
    if not (np.isfinite(logit).all() and np.isfinite(p).all() and ((p >= 0.0) & (p <= 1.0)).all()):
        raise Blocked(f"{key}: logit or p is not finite, or p is outside [0, 1]")
    return int(row.size)


def load_test_predictions(key: str, directory: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """A cell's TEST predictions before the opening: RESULT.json completed, the file's SHA-256 equal to its
    pred_test_sha256, and the raw arrays checked (check_prediction_arrays) before any conversion."""

    result = read_json(directory / "RESULT.json")
    if result.get("status") != "completed":
        raise Blocked(f"{directory / 'RESULT.json'}: the cell is not completed")
    sha = sha256_file(directory / "pred_test.npz")
    if sha != result.get("pred_test_sha256"):
        raise Blocked(f"{directory / 'pred_test.npz'}: SHA-256 {sha} differs from RESULT.json ({result.get('pred_test_sha256')})")
    with np.load(guard(directory / "pred_test.npz"), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    check_prediction_arrays(key, arrays)
    return np.asarray(arrays["row"], dtype=np.int64), np.asarray(arrays["p"], dtype=np.float64), sha


def reproduce_qdev(paths: Paths, rows_table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Ruling 67, the binding behavioural check before the opening: recompute the gate and K1-K6 on Q_DEV with the code
    that runs now (the same analyse the TEST step calls) from the saved Q_DEV predictions of the q_ka_1 cells and the
    four refits (nothing is retrained), and compare every output with the lock-bound KA_POSTHOC_QDEV.json exactly
    (canonical JSON: sorted keys, shortest round-trip float text).  It reads Q_DEV and TRAIN KA labels only
    (load_ka_split slices them out of the label arrays); no TEST label is used."""

    tick = time.perf_counter()
    stored = read_json(paths.qdev)
    record: dict[str, Any] = {"reference": str(paths.qdev), "reference_sha256": sha256_file(paths.qdev), "compared": [], "differ": [],
                              "equal": False, "error": None,
                              "rule": ("gate, K1-K6, consistency, K4_train and the row / match / tie counts recomputed on Q_DEV must equal "
                                       "KA_POSTHOC_QDEV.json exactly (canonical JSON); any difference or error stops the TEST step before "
                                       "exposure (Ruling 67)")}

    def canonical(value: Any) -> str:
        return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False)

    try:
        completed_fits(paths)
        data = split_data(paths, "q_dev", rows_table)
        rate, train_info = train_rate(paths, rows_table)
        results = analyse("q_dev", data, gate_reference(paths, "q_dev"), rate=rate)
        now = {**{key: results[key] for key in ("gate", "K1", "K2", "K3", "K4", "K5", "K6", "consistency")}, "K4_train": train_info,
               "rows": int(data["y"].size), "matches": int(np.unique(data["match"]).size), "split_rows": data["split_rows"], "ties": data["ties"]}
    except Exception as error:  # noqa: BLE001  (a reproduction that cannot run is not equal)
        record.update(error=f"{type(error).__name__}: {error}", seconds=round(time.perf_counter() - tick, 3))
        return record
    record["compared"] = sorted(now)
    record["differ"] = sorted(key for key, value in now.items() if key not in stored or canonical(value) != canonical(stored[key]))
    record.update(equal=not record["differ"], seconds=round(time.perf_counter() - tick, 3))
    return record


def test_preflight(paths: Paths, rows_table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Every TEST check that needs no KA label, run before the OPENINGS.json entry (fix rounds 1-2): the nine analysis
    cells' TEST predictions (hashes; raw arrays one-dimensional, of one length, integer rows and floating logit / p,
    finite, p in [0, 1]; one row order; unique TEST-role rows; row count = KA_PREPARE TEST rows - ties = each RESULT
    rows.test), frame age within [0, 10] s with every 1-s bin filled, kill counts >= 1 on every TEST row, the
    participant table, the KA_LABELS hash and engagement keys (no label array is loaded here) and the KA_REPORT TEST
    block's numbers (finite, intervals ordered)."""

    role = np.asarray(rows_table["role"])
    split_rows = np.flatnonzero(role == ROLE["test"])
    preds, rows_ref, files, counts = {}, None, [], {}
    for key, directory in analysis_cells(paths).items():
        rows, p, _ = load_test_predictions(key, directory)
        if rows_ref is None:
            rows_ref = rows
        elif not np.array_equal(rows, rows_ref):
            raise Blocked(f"{key}: KA TEST prediction rows differ (kill_label.py:322-323)")
        preds[key] = p
        recorded = (read_json(directory / "RESULT.json").get("rows") or {}).get("test")
        if recorded is not None:
            counts[f"{key} RESULT rows.test"] = int(recorded)
        files += [directory / "RESULT.json", directory / "pred_test.npz"]
    if np.unique(rows_ref).size != rows_ref.size or not np.isin(rows_ref, split_rows).all():
        raise Blocked("KA TEST prediction rows repeat or fall outside the TEST role")
    prepare = read_json(paths.ka_prepare)
    test_counts = (prepare.get("per_role") or {}).get("TEST") or {}
    if "rows" not in test_counts or "ties" not in test_counts or int(test_counts["rows"]) != split_rows.size:
        raise Blocked("KA_PREPARE.json per_role.TEST does not give the TEST rows of the row table")
    counts["KA_PREPARE TEST rows - ties"] = int(test_counts["rows"]) - int(test_counts["ties"])
    if set(counts.values()) != {int(rows_ref.size)}:
        raise Blocked(f"the TEST prediction rows ({rows_ref.size}) are not the recorded eligible TEST count: {counts}")
    age = np.asarray(rows_table["frame_age_s"], dtype=np.float64)[rows_ref]
    if age.min() < 0 or age.max() > 10:
        raise Blocked("TEST frame age outside [0, 10] s (frame_age_seconds.py:55-56)")
    filled = np.bincount(np.minimum(np.floor(age).astype(int), 9), minlength=10)
    if filled.min() == 0:
        raise Blocked("an empty 1-s TEST frame-age bin")
    if int(np.asarray(rows_table["eval_n_kills"])[split_rows].min()) < 1:
        raise Blocked("a TEST kill count is below 1 (plan section 7)")
    participants = load_participants(paths, rows_table)
    sha = sha256_file(paths.ka_labels)
    if sha != prepare.get("ka_labels_sha256"):
        raise Blocked(f"{paths.ka_labels}: SHA-256 {sha} differs from KA_PREPARE.json ka_labels_sha256")
    if not np.array_equal(load_npz(paths.ka_labels, ["engagement_key"])["engagement_key"], rows_table["engagement_key"]):
        raise Blocked("KA labels are not aligned with the q row table (kill_label.py:160-161)")
    structure = check_gate_reference_structure(paths, "test")
    return {"rows": rows_ref, "preds": preds, "split_rows": split_rows, "participants": participants, "files": files,
            "checks": {"prediction_rows": int(rows_ref.size), "counts": counts, "frame_age_bin_rows": filled.tolist(),
                       "test_rows": int(split_rows.size), "ka_labels_sha256": sha, "ka_report_test_block": structure,
                       "label_arrays_loaded": "none here (engagement_key only); the Q_DEV reproduction then reads Q_DEV / TRAIN rows"}}


def test_split_data(paths: Paths, rows_table: Mapping[str, np.ndarray], inputs: Mapping[str, Any], opening: Mapping[str, Any]) -> dict[str, Any]:
    """After the opening: the KA TEST labels, whose eligible rows must be the prediction rows checked before it; the
    fields are the ones split_data gives (prediction row order, kill_label.py:316-328)."""

    ka = load_ka_split(paths, rows_table, "test", opening=opening)
    rows_ref = inputs["rows"]
    if not np.array_equal(np.sort(rows_ref), ka["idx"]):
        raise Blocked("test: KA prediction rows are not the eligible rows (kill_label.py:325-327)")
    position = np.searchsorted(ka["idx"], rows_ref)
    part = inputs["participants"]
    return {"split": "test", "rows": rows_ref, "y": ka["y"][position], "match": np.asarray(rows_table["match_index"])[rows_ref],
            "n_kills": np.asarray(rows_table["eval_n_kills"])[rows_ref], "split_n_kills": np.asarray(rows_table["eval_n_kills"])[ka["split_rows"]],
            "age": np.asarray(rows_table["frame_age_s"], dtype=np.float64)[rows_ref],
            "minutes": np.asarray(rows_table["elapsed_min"], dtype=np.float64)[rows_ref],
            "p_pre": np.asarray(rows_table["pt_p_pre"], dtype=np.float64)[rows_ref],
            "cohort": np.asarray(part["cohort"])[rows_ref], "t5": np.asarray(part["t5"], dtype=bool)[rows_ref],
            "n_blue": np.asarray(part["n_blue"], dtype=np.int64)[rows_ref], "n_red": np.asarray(part["n_red"], dtype=np.int64)[rows_ref],
            "preds": inputs["preds"], "split_rows": int(ka["split_rows"].size), "ties": int((~ka["eligible_of_split"]).sum()),
            "files": [paths.rows, paths.rows_manifest, paths.ka_labels, paths.ka_prepare, paths.participants, paths.participants_manifest,
                      *inputs["files"]]}


def modules_outside_lock(lock: Mapping[str, Any]) -> dict[str, str]:
    """Repository modules loaded now whose file is not in the lock with the same SHA-256."""

    code = (lock.get("code") or {}).get("files") or {}
    return {rel: sha for rel, sha in repo_modules().items() if code.get(rel) != sha}


def test_stage(paths: Paths = Paths(), *, time_limit_s: float | None = EVAL_LIMIT_S) -> int:
    """The single TEST look (plan 4 step 7; fix rounds 1-2).  Before the OPENINGS.json entry: the analysis code runs
    once on synthetic rows (every module it needs is loaded), the lock is rechecked (bound files, plan blob,
    REFIT_INPUTS, loaded modules, EXPOSURE row), every label-free TEST input is checked (test_preflight) and the gate
    and K1-K6 are recomputed on Q_DEV with this code and must equal the lock-bound KA_POSTHOC_QDEV.json exactly
    (reproduce_qdev, the binding check of Ruling 67); a failure there writes TEST_BLOCKED_<utc>.json and can be fixed and
    rerun.  Then the entry is saved, the KA TEST labels are read, the gate and K1-K6 run, and a module outside the lock
    stops the result before it is written.  After the entry, a failure keeps the attempt recorded and a rerun needs a
    plan revision and a new lock version."""

    started, tick, cpu0 = utc(), time.perf_counter(), time.process_time()
    lock_path = latest_lock(paths.out)
    if lock_path is None:
        return _blocked(paths, "test", {"reason": "no KA_POSTHOC_LOCK.json"})
    lock, lock_sha = read_json(lock_path), sha256_file(lock_path)
    previous = ka_openings(paths)
    for opening in previous:
        if opening.get("lock_sha256") == lock_sha:
            if test_result_path(paths.out, opening["attempt"]).exists():
                raise FileExistsError(f"{test_result_path(paths.out, opening['attempt'])} exists: TEST was evaluated with this lock")
            return _blocked(paths, "test", {"reason": "this lock was already used by an exposed attempt; a rerun needs a plan revision and a new lock version",
                                            "attempt": opening["attempt"]})
    prespec = (lock.get("plan") or {}).get("blob_sha256")
    if any(o.get("prespec_sha256") == prespec for o in previous):
        return _blocked(paths, "test", {"reason": "a rerun after exposure needs a revised plan (another plan blob) under the new lock"})
    try:
        warm_up_analysis()
    except Exception as error:  # noqa: BLE001  (before exposure: recorded, fixable)
        return _blocked(paths, "test", {"reason": f"the analysis warm-up failed (before exposure): {type(error).__name__}: {error}",
                                        "traceback": traceback.format_exc()})
    problems = verify_lock(lock, lock_sha, paths)
    if problems:
        return _blocked(paths, "test", {"reason": "the lock does not hold (before exposure)", "lock": str(lock_path), "problems": problems})
    try:
        rows_table, _ = load_rows_table(paths)
        inputs = test_preflight(paths, rows_table)
    except Exception as error:  # noqa: BLE001  (before exposure: recorded, fixable)
        return _blocked(paths, "test", {"reason": f"a label-free TEST input check failed (before exposure): {type(error).__name__}: {error}",
                                        "lock": str(lock_path)})
    reproduction = reproduce_qdev(paths, rows_table)
    if not reproduction["equal"]:
        return _blocked(paths, "test", {"reason": ("the gate and K1-K6 recomputed on Q_DEV with the locked code do not equal the lock-bound "
                                                   "KA_POSTHOC_QDEV.json (Ruling 67; before exposure)"),
                                        "lock": str(lock_path), "reproduction": reproduction})
    inputs["checks"]["qdev_reproduction"] = reproduction
    late = modules_outside_lock(lock)
    if late:
        return _blocked(paths, "test", {"reason": "modules outside the lock were loaded by the TEST input checks (before exposure)",
                                        "modules": late})
    attempt = len(previous) + 1
    target = test_result_path(paths.out, attempt)
    if target.exists():
        raise FileExistsError(f"{target} exists")
    entry = {"experiment": EXPERIMENT, "attempt": attempt, "utc": utc(), "lock": str(lock_path), "lock_sha256": lock_sha,
             "prespec_sha256": prespec, "labels": [str(paths.ka_labels)], "labels_sha256": sha256_file(paths.ka_labels), "note": OPENING_NOTE}

    def record_failure(reason: str, **details: Any) -> None:
        write_json_once(test_failure_path(paths.out, attempt), {"version": VERSION, "experiment": EXPERIMENT, "status": "failed_after_exposure",
                                                                "reason": reason, "opening": entry, "utc": utc(), **details})

    before = append_opening(paths.openings, entry)          # exposure starts once this entry is on disk
    done = watchdog(time_limit_s, lambda: record_failure(f"evaluation time limit {time_limit_s} s (plan 3.4)"))
    try:
        data = test_split_data(paths, rows_table, inputs, entry)
        rate, train_info = train_rate(paths, rows_table)
        results = analyse("test", data, gate_reference(paths, "test"), rate=rate)
        rows, matches = int(data["y"].size), int(np.unique(data["match"]).size)
        fits = completed_fits(paths)
        files = sorted({str(Path(p)) for p in [*data["files"], paths.ka_report]} | {str(paths.cell(c) / "RESULT.json") for c in FIT_ORDER})
        payload = {"version": VERSION, "experiment": EXPERIMENT, "split": "TEST", "status": "completed", "label": POSTHOC,
                   "attempt": attempt, "opening": entry, "openings_before": before,
                   "lock": {"path": str(lock_path), "sha256": lock_sha, "locked_utc": lock.get("locked_utc")},
                   "plan": lock.get("plan"), "sign": SIGN, "rows": rows, "matches": matches, "split_rows": data["split_rows"], "ties": data["ties"],
                   **results, "K4_train": train_info, "fits": fits_summary(fits), "pre_exposure_checks": inputs["checks"],
                   "environment": environment(), "stage": stage_record("test", started, tick, cpu0),
                   "run": run_record(rows=rows, matches=matches, inputs=[Path(p) for p in files], code=[ROOT / CODE_REL, *(ROOT / m for m in MIRRORED)],
                                     seed=BOOT_SEED, started_utc=started)}
        late = modules_outside_lock(lock)                 # no result is written when a module outside the lock loaded
        if late:
            raise Blocked(f"modules outside the lock were loaded after exposure: {sorted(late)}")
        payload["loaded_modules"] = repo_modules()
        write_json_once(target, payload)
    except Exception as error:  # noqa: BLE001  (recorded; the opening stays)
        done.set()
        record_failure(f"{type(error).__name__}: {error}", gate=getattr(error, "gate", None), checks=getattr(error, "checks", None),
                       traceback=traceback.format_exc())
        _say(paths, f"TEST step failed after exposure: {error}")
        return 3
    done.set()
    _print_split("TEST", payload)
    print(f"wrote {target} sha256 {sha256_file(target)}")
    return 0


# ----------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--inputs-only", action="store_true")
    fit = sub.add_parser("fit")
    fit.add_argument("--cell", choices=FIT_ORDER, required=True)
    sub.add_parser("qdev")
    lock = sub.add_parser("lock")
    lock.add_argument("--new-version", action="store_true")
    sub.add_parser("test")
    args = parser.parse_args(argv)
    paths = Paths()
    if args.command == "launch":
        try:
            return launch_stage(paths, inputs_only=args.inputs_only)
        except Exception:  # noqa: BLE001  (a detached launcher has no console: the traceback goes to LAUNCH.log)
            _log(paths, "launch crashed:\n" + traceback.format_exc())
            raise
    if args.command == "fit":
        return fit_stage(args.cell, paths)
    if args.command == "qdev":
        return qdev_stage(paths)
    if args.command == "lock":
        return lock_stage(paths, new_version=args.new_version)
    return test_stage(paths)


if __name__ == "__main__":
    raise SystemExit(main())
