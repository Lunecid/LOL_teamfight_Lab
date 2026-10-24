"""External validation, step 3a: re-hash everything EXTERNAL_SCORE_LOCK.json binds, before any sealed label is opened.

Checks the protocol, the frozen identities, the parity reports, every external and frozen-imported code file, every
per-dataset input/label/row/store file, every prediction and its metadata file, the sealed label files, and the
stage-time log.  The stage-time log had one line (the lock's own timing) appended after the lock was written; its
recorded hash is checked against the file without that last line, and the appended line is reported.
Writes reports/external_validation_20261009/EXTERNAL_LOCK_CHECK.json.  Reads no label content.

Usage: python -B studies/external_validation_20261009/e3_verify_lock.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/external_validation_20261009"
LOCK = REPORTS / "EXTERNAL_SCORE_LOCK.json"
DATA = Path(r"D:\LOL_External_20261009")
OUT = REPORTS / "EXTERNAL_LOCK_CHECK.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    lock_bytes = LOCK.read_bytes()
    lock = json.loads(lock_bytes)
    checked, bad, notes = 0, [], []

    def check(label: str, path: Path, expected: str) -> None:
        nonlocal checked
        checked += 1
        path = Path(path)
        if not path.is_file():
            bad.append({"item": label, "path": str(path), "problem": "missing"})
        elif sha_file(path) != expected:
            bad.append({"item": label, "path": str(path), "problem": "sha256 differs"})

    check("protocol", Path(lock["protocol"]["path"]), lock["protocol"]["sha256"])
    for name, entry in lock["frozen"].items():
        check(f"frozen:{name}", Path(entry["path"]), entry["sha256"])
        if "protocol" in entry and entry["protocol"] != entry["sha256"]:
            bad.append({"item": f"frozen:{name}", "problem": "lock value differs from protocol value"})
    for name, entry in lock["parity"].items():
        check(f"parity:{name}", Path(entry["path"]), entry["sha256"])
        if entry.get("status") != "PASS":
            bad.append({"item": f"parity:{name}", "problem": f"status {entry.get('status')}"})
    for group in ("external", "frozen_imported"):
        for rel, digest in lock["code"][group].items():
            check(f"code:{rel}", ROOT / rel, digest)
    for dataset, entry in lock["datasets"].items():
        for name, f in entry["files"].items():
            check(f"{dataset}:{name}", Path(f["path"]), f["sha256"])
        for cell, c in entry["cells"].items():
            check(f"{dataset}:pred:{cell}", DATA / "scores" / dataset / f"{cell}.npz", c["pred_sha256"])
            check(f"{dataset}:meta:{cell}", DATA / "scores" / dataset / f"{cell}.json", c["meta_sha256"])
        check(f"{dataset}:sealed", DATA / "labels" / dataset / f"EXTERNAL_LABELS_SEALED_{dataset}.npz", entry["sealed"]["sha256"])
        if entry["sealed"].get("opened") is not False:
            bad.append({"item": f"{dataset}:sealed", "problem": "lock does not record the sealed file as unopened"})
    times = Path(lock["stage_times"]["path"])
    raw = times.read_bytes()
    stage_times = {"full_file_equal": hashlib.sha256(raw).hexdigest() == lock["stage_times"]["sha256"]}
    if not stage_times["full_file_equal"]:
        lines = raw.splitlines(keepends=True)
        head = b"".join(lines[:-1])
        stage_times.update(without_last_line_equal=hashlib.sha256(head).hexdigest() == lock["stage_times"]["sha256"],
                           appended_last_line=lines[-1].decode("utf-8", "replace").strip())
        if stage_times["without_last_line_equal"]:
            notes.append("E2bc_STAGE_TIMES.jsonl: one line (the lock's own timing) was appended after the lock; the rest equals the locked bytes")
        else:
            bad.append({"item": "stage_times", "problem": "differs even without the last line"})
    checked += 1
    report = {"kind": "EXTERNAL_LOCK_CHECK", "utc": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "files_checked": checked, "mismatches": bad,
              "stage_times": stage_times, "notes": notes, "passed": not bad,
              "code_sha256": sha_file(Path(__file__))}
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("lock_sha256", "files_checked", "passed", "stage_times", "notes")}, ensure_ascii=False, indent=1))
    if bad:
        print(json.dumps(bad[:20], ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
