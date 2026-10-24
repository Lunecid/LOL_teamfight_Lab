"""Paths and run records shared by the ToG pre-submission experiments (PLAN_E1_E5_KO.md sections 0 and 4).

Existing outputs under the training root are read only.  New outputs go to their own tog_e*_1 folder, and every
result file is written once (``write_once`` of the frozen q TEST stage).  Each result JSON carries a run record:
seed (if any), row and match counts, SHA-256 of its input files and of its code, and UTC times.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.q_production_20261005.qtest import sha256_file, utc, write_once  # noqa: E402,F401  (write_once is re-exported)

TRAINING = Path("D:/LOL_Training_20261004")
LABELS = TRAINING / "labels_1"        # locked labels (LABEL_LOCK.json), read only
Q_RUN = TRAINING / "q_run_1"          # q row tables, raw input store and prepared inputs, read only
TOG_E1 = TRAINING / "tog_e1_1"        # E1 outputs
TOG_E2 = TRAINING / "tog_e2_1"        # E2 outputs


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write an uncompressed .npz once (no pickled objects) and return its SHA-256; an existing file is refused."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; it is written once")
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return sha256_file(path)


def peak_memory_bytes() -> int | None:
    """Peak working set of this process (Windows; psutil), else the peak resident size; None if unavailable."""

    try:
        import psutil

        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", None):
            return int(info.peak_wset)
    except Exception:  # noqa: BLE001  (a missing psutil only drops the number)
        pass
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # noqa: BLE001
        return None


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)


def git_head() -> str | None:
    """The commit checked out in this repo (None outside git)."""

    head = _git("rev-parse", "HEAD")
    return head.stdout.decode("ascii", "replace").strip() or None if head.returncode == 0 else None


def document_record(path: Path) -> dict[str, Any]:
    """A file's SHA-256 on disk, and for a file in this repo the last commit that touched it, the SHA-256 and blob id
    of the committed bytes and whether the file is unchanged since that commit (``git diff``, so the checkout's
    line-ending conversion is not a change).  Outside the repo or never committed: ``commit`` is None."""

    path = Path(path).resolve()
    record: dict[str, Any] = {"path": str(path), "repo_path": None, "sha256": sha256_file(path), "commit": None,
                              "git_blob": None, "committed_sha256": None, "unchanged_since_commit": False}
    if not path.is_relative_to(ROOT):
        return record
    rel = path.relative_to(ROOT).as_posix()
    record["repo_path"] = rel
    log = _git("log", "-1", "--format=%H", "--", rel)
    commit = log.stdout.decode("ascii", "replace").strip()
    blob = _git("cat-file", "blob", f"{commit}:{rel}") if commit else None
    if log.returncode or not commit or blob is None or blob.returncode:
        return record
    record.update(commit=commit, git_blob=_git("rev-parse", f"{commit}:{rel}").stdout.decode("ascii", "replace").strip(),
                  committed_sha256=hashlib.sha256(blob.stdout).hexdigest(),
                  unchanged_since_commit=_git("diff", "--quiet", commit, "--", rel).returncode == 0)
    return record


def run_record(*, rows: Any, matches: Any, inputs: Iterable[Path], code: Iterable[Path], seed: int | None = None,
               started_utc: str | None = None) -> dict[str, Any]:
    """The provenance block of a result JSON; ``rows`` and ``matches`` are counts (a number or one per role)."""

    return {"seed": seed, "rows": rows, "matches": matches,
            "input_sha256": {Path(path).as_posix(): sha256_file(path) for path in inputs},
            "code_sha256": {_code_name(path): sha256_file(path) for path in code},
            "started_utc": started_utc, "utc": utc(), "python": platform.python_version(), "numpy": np.__version__}


def _code_name(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
