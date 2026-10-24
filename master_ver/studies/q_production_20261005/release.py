"""q scientific release: the content hash of the q worker's import closure (ARCHITECTURE_KO.md section 6.1).

``SCIENTIFIC_Q_<digest16>.json`` binds, by SHA256 and workspace-relative path, every workspace
module that ``q_worker`` imports (found by importing it in a fresh interpreter, as the V release
does in reports/v_production_20261004/make_release.py) plus the hash-locked cards and matrix.
Its bytes depend only on that evidence, so re-issuing unchanged code yields the same file.  A
release file is never edited.  The worker verifies the release before it runs and checks, after
its imports, that every workspace module it loaded is bound by the release.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
VERSION = "q_scientific_release_v1_20261005"
WORKER_MODULES = ("studies.q_production_20261005.q_worker", "studies.q_production_20261005.q_prepare",
                  "studies.q_production_20261005.jobs", "studies.q_production_20261005.qselect")
EXTRA = ("reports/feature_control_study_20261001/EXECUTION_CARDS.json", "reports/feature_control_study_20261001/CARTESIAN_COMPARISON_MATRIX.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_files(modules: Iterable[Any]) -> set[str]:
    """Workspace-relative paths of the loaded modules that live under the workspace root."""

    root = ROOT.resolve()
    out = set()
    for module in modules:
        name = getattr(module, "__file__", None)
        if name and Path(name).is_absolute() and Path(name).resolve().is_relative_to(root):
            out.add(Path(name).resolve().relative_to(root).as_posix())
    return out


def closure(modules: Iterable[str] = WORKER_MODULES) -> list[str]:
    code = ("import sys; sys.path.insert(0, %r); import importlib; [importlib.import_module(m) for m in %r];"
            "from studies.q_production_20261005.release import workspace_files;"
            "print('\\n'.join(sorted(workspace_files(list(sys.modules.values())))))") % (str(ROOT), list(modules))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    output = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, encoding="utf-8", env=env, check=True).stdout
    return [line for line in output.splitlines() if line.strip() and (ROOT / line).is_file()]


def evidence(modules: Iterable[str] = WORKER_MODULES) -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in sorted({*closure(modules), *EXTRA})}


def digest(files: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(files.items())), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def issue(out_dir: Path) -> Path:
    """Write (or confirm) the release file for the current code; returns its path."""

    files = evidence()
    value = digest(files)
    path = Path(out_dir) / f"SCIENTIFIC_Q_{value[:16]}.json"
    payload = {"version": VERSION, "digest": value, "evidence": files}
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"{path.name} exists with different content")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def verify(path: Path) -> dict[str, Any]:
    """Every bound file still has its recorded hash, and the digest matches the evidence."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("digest") != digest(payload["evidence"]):
        raise RuntimeError(f"{Path(path).name} is not a valid q release")
    changed = [name for name, value in payload["evidence"].items() if not (ROOT / name).is_file() or sha256_file(ROOT / name) != value]
    if changed:
        raise RuntimeError(f"code differs from release {payload['digest'][:16]}: {changed[:5]}")
    return payload


def check_loaded(payload: Mapping[str, Any]) -> None:
    """Refuse to run when a loaded workspace module is not bound by the release."""

    loaded = workspace_files(list(sys.modules.values())) - set(payload["evidence"])
    unbound = sorted(name for name in loaded if name.startswith(("studies/", "reports/")))   # tests/ only when run under pytest
    if unbound:
        raise RuntimeError(f"loaded modules outside release {payload['digest'][:16]}: {unbound[:5]}")
