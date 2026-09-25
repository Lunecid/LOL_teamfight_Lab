"""ev4_record_lock: the one test of "record 1 is locked" that every held-out consumer runs.

assert_record1_locked(path, sha256) accepts a file only when ALL of these hold (SplitViolation otherwise):
  * it is a regular .json file directly inside outputs/reest_exact_v4_20260925/records/ (symlinks resolved; not a
    sub-directory), its name starts with 'record1_' and NOT with 'record1_DRAFT' or 'record1a' (case-insensitive);
  * its sha256 equals the given sha256 (gameplay.split_guard.assert_record_exists: non-empty JSON object);
  * record == "1" exactly (the string), status == "LOCKED", is_final_lock is true, usable_as_record1 is true,
    author_signoff.signed is true with a non-empty signed_by, smoke is false (present and false);
  * a top-level V_frozen_bundle_sha256 (64 lower-case hex) is present; if the V block also names a bundle
    (V.V_frozen.bundle_sha256) it must be the same value;
  * the embedded record 1A sha256 (record1a.sha256) equals the sha256 of the record 1A file of the same name in
    records/ (name starting with 'record1a');
  * the embedded pre-decision sha256 (records.predecisions_pinned.sha256) equals the sha256 of the pre-decision file
    of the same name in records/ AND the sha256 pinned in ev4_common (PREDECISIONS_SHA256).
It returns the parsed record.  It reads no engagement or match data.

So a DRAFT (any name or content), the pre-decision record, record 1A, an ev4_03 frozen_manifest.json, a .md summary,
a smoke lock, or a record copied anywhere else is refused.

only_functions_differ(old_src, new_src, names) is the AST test the code-drift whitelist of the guard-only patch uses
(stage2/GUARD_PATCH_PLAN.md): True when two Python sources differ only inside the bodies of the named top-level
functions (same signatures, same everything else).

This module is not in any ev4_01 / ev4_02 / ev4_03 code-hash list.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
for _p in (str(WT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402

from gameplay.split_guard import SplitViolation, assert_record_exists  # noqa: E402

RECORDS_DIR = EC.OUT_BASE / "records"
FINAL_PREFIX = "record1_"
REFUSED_PREFIXES = ("record1_draft", "record1a")
RECORD1A_PREFIX = "record1a"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

PathLike = Union[str, Path]


class RecordNotLocked(SplitViolation):
    """The file is not a locked record 1 (a SplitViolation: held-out access is refused)."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _hex64(x: Any) -> Optional[str]:
    return x if isinstance(x, str) and HEX64.match(x) else None


def check_location(path: PathLike, records_dir: Optional[PathLike] = None) -> Path:
    """The resolved path when it is a .json file directly inside records_dir with an acceptable name."""
    if path is None or str(path).strip() == "":
        raise RecordNotLocked("no record 1 given; held-out patches need the locked record 1")
    rd = Path(RECORDS_DIR if records_dir is None else records_dir).resolve()
    try:
        rp = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        raise RecordNotLocked(f"record 1 not found: {path}") from e
    if not rp.is_file():
        raise RecordNotLocked(f"record 1 is not a regular file: {rp}")
    if rp.parent != rd:
        raise RecordNotLocked(f"record 1 must be a file directly inside {rd}; got {rp}")
    if rp.suffix.lower() != ".json":
        raise RecordNotLocked(f"record 1 must be the .json record, not {rp.name}")
    low = rp.name.lower()
    if any(low.startswith(x) for x in REFUSED_PREFIXES):
        raise RecordNotLocked(f"{rp.name} is a draft or record 1A, not the locked record 1")
    if not low.startswith(FINAL_PREFIX):
        raise RecordNotLocked(f"{rp.name} is not a record 1 file (name must start with '{FINAL_PREFIX}')")
    return rp


def check_embedded_hashes(rec: Mapping[str, Any], records_dir: Optional[PathLike] = None,
                          predecisions_sha256: Optional[str] = None) -> Dict[str, str]:
    """record1a.sha256 and records.predecisions_pinned.sha256 against the files of the same name in records_dir
    (and the pre-decision sha256 against the pin in ev4_common).  Returns {'record1a': sha, 'predecisions': sha}."""
    rd = Path(RECORDS_DIR if records_dir is None else records_dir).resolve()
    pin = EC.PREDECISIONS_SHA256 if predecisions_sha256 is None else predecisions_sha256
    r1a = rec.get("record1a") if isinstance(rec.get("record1a"), Mapping) else {}
    want_1a = _hex64(r1a.get("sha256"))
    name_1a = Path(str(r1a.get("path") or "")).name
    if not want_1a or not name_1a.lower().startswith(RECORD1A_PREFIX):
        raise RecordNotLocked("record 1 does not embed the record 1A path and sha256 (key 'record1a')")
    f1a = rd / name_1a
    if not f1a.is_file() or _sha256_file(f1a) != want_1a:
        raise RecordNotLocked(f"embedded record 1A sha256 does not match {f1a}")
    pre = ((rec.get("records") or {}).get("predecisions_pinned") or {}) if isinstance(rec.get("records"), Mapping) \
        else {}
    want_pre = _hex64(pre.get("sha256"))
    name_pre = Path(str(pre.get("path") or "")).name
    if not want_pre or not name_pre:
        raise RecordNotLocked("record 1 does not embed the pre-decision record (records.predecisions_pinned)")
    if want_pre != str(pin).lower():
        raise RecordNotLocked("embedded pre-decision sha256 is not the one pinned in ev4_common")
    fpre = rd / name_pre
    if not fpre.is_file() or _sha256_file(fpre) != want_pre:
        raise RecordNotLocked(f"embedded pre-decision sha256 does not match {fpre}")
    return {"record1a": want_1a, "predecisions": want_pre}


def check_locked_fields(rec: Mapping[str, Any], allow_smoke: bool = False) -> str:
    """The lock fields of a parsed record; returns V_frozen_bundle_sha256.  allow_smoke is for the smoke test of
    the lock command only (assert_record1_locked never passes it)."""
    if not isinstance(rec, Mapping):
        raise RecordNotLocked("record 1 is not a JSON object")
    if rec.get("record") != "1":
        raise RecordNotLocked(f"record field is {rec.get('record')!r}, not exactly '1'")
    if rec.get("status") != "LOCKED":
        raise RecordNotLocked(f"status is {rec.get('status')!r}, not 'LOCKED'")
    for k in ("is_final_lock", "usable_as_record1"):
        if rec.get(k) is not True:
            raise RecordNotLocked(f"{k} is not true")
    so = rec.get("author_signoff")
    if not isinstance(so, Mapping) or so.get("signed") is not True or not str(so.get("signed_by") or "").strip():
        raise RecordNotLocked("author_signoff.signed is not true (or signed_by is empty)")
    if "smoke" not in rec or not isinstance(rec.get("smoke"), bool):
        raise RecordNotLocked("record 1 has no boolean 'smoke' field")
    if rec["smoke"] and not allow_smoke:
        raise RecordNotLocked("a smoke record can never be record 1")
    vb = _hex64(rec.get("V_frozen_bundle_sha256"))
    if not vb:
        raise RecordNotLocked("record 1 has no top-level V_frozen_bundle_sha256 (64 lower-case hex)")
    nested = (((rec.get("V") or {}).get("V_frozen") or {}).get("bundle_sha256")
              if isinstance(rec.get("V"), Mapping) else None)
    if nested is not None and nested != vb:
        raise RecordNotLocked("V.V_frozen.bundle_sha256 disagrees with the top-level V_frozen_bundle_sha256")
    return vb


def assert_record1_locked(path: PathLike, sha256: Optional[str], records_dir: Optional[PathLike] = None,
                          predecisions_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Refuse (RecordNotLocked, a SplitViolation) unless `path` is the author-locked record 1 with this sha256.
    Returns the parsed record.  records_dir / predecisions_sha256 exist for tests; production uses the defaults."""
    rp = check_location(path, records_dir)
    if not sha256 or not _hex64(str(sha256).strip().lower()):
        raise RecordNotLocked("the record 1 sha256 (64 hex) must be given")
    assert_record_exists(rp, str(sha256).strip().lower())
    rec = json.loads(rp.read_text(encoding="utf-8"))
    check_locked_fields(rec, allow_smoke=False)
    check_embedded_hashes(rec, records_dir, predecisions_sha256)
    return rec


# ------------------------------------------------------------------ guard-patch drift whitelist (AST)
def _top_level(tree: ast.Module):
    return list(tree.body)


def _head(fn: ast.AST) -> tuple:
    """Name, arguments, decorators and return annotation of a function definition (everything but the body)."""
    return (fn.name, ast.dump(fn.args), tuple(ast.dump(d) for d in fn.decorator_list),
            ast.dump(fn.returns) if fn.returns is not None else None)


def only_functions_differ(old_src: str, new_src: str, names: Iterable[str]) -> bool:
    """True when old_src and new_src have the same top-level statements in the same order, and every difference is
    inside the body (docstring included) of a top-level function named in `names` whose name, decorators, arguments
    and return annotation are unchanged."""
    allowed = set(names)
    a, b = _top_level(ast.parse(old_src)), _top_level(ast.parse(new_src))
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if type(x) is not type(y):
            return False
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name in allowed:
            if _head(x) != _head(y):
                return False
            continue
        if ast.dump(x) != ast.dump(y):
            return False
    return True
