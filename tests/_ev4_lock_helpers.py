"""Synthetic records/ directories and locked record 1 files for the ev4 record-lock tests (tmp_path only)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

R1A_NAME = "record1a_boundaries_20260925T000000Z.json"
PRE_NAME = "stage2_predecisions_20260925T000000Z.json"
V_SHA = "c" * 64


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_records_dir(tmp: Path) -> Tuple[Path, str]:
    """tmp/records with a fake record 1A and a fake pre-decision record; returns (dir, pre-decision sha256)."""
    rd = Path(tmp) / "records"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / R1A_NAME).write_text(json.dumps({"record": "1A", "locked": {"G": 14000}}), encoding="utf-8")
    (rd / PRE_NAME).write_text(json.dumps({"record": "stage2 pre-decisions", "remakes": "excluded"}),
                               encoding="utf-8")
    return rd, sha(rd / PRE_NAME)


def record_body(rd: Path) -> Dict[str, Any]:
    return {"record": "1", "status": "LOCKED", "is_final_lock": True, "usable_as_record1": True, "smoke": False,
            "author_signoff": {"signed": True, "signed_by": "Author"},
            "V_frozen_bundle_sha256": V_SHA, "V": {"V_frozen": {"bundle_sha256": V_SHA}},
            "record1a": {"path": f"C:/elsewhere/{R1A_NAME}", "sha256": sha(rd / R1A_NAME)},
            "records": {"predecisions_pinned": {"path": f"C:/elsewhere/{PRE_NAME}", "sha256": sha(rd / PRE_NAME)}}}


def write_record(rd: Path, name: str = "record1_20260927T000000Z.json",
                 edit: Optional[Callable[[Dict[str, Any]], Any]] = None, where: Optional[Path] = None) -> Tuple[Path, str]:
    body = copy.deepcopy(record_body(rd))
    if edit is not None:
        edit(body)
    p = Path(where or rd) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return p, sha(p)


def draft_body(rd: Path, smoke: bool = True) -> Dict[str, Any]:
    """A lockable DRAFT (as ev4_r1_record writes it, reduced to the fields the lock reads)."""
    b = record_body(rd)
    b.update({"record": "1 DRAFT - not locked", "status": "DRAFT", "is_final_lock": False,
              "usable_as_record1": False, "smoke": smoke, "lockable": True, "created_utc": "20260927T000000Z",
              "author_signoff": {"signed": False, "signed_by": None, "signed_utc": None},
              "oof_v": {"status": "ok", "ok": True, "manifest_sha256": "d" * 64,
                        "fold_bundle_sha256": {str(k): f"{k}" * 64 for k in range(5)}},
              "prices_1514": {"status": "ok", "ok": True, "sha256": "e" * 64},
              "pending_blocking": [],
              "pending_author_confirmation": [{"id": "A01", "text": "q grid"}, {"id": "A02", "text": "wording"}]})
    return b
