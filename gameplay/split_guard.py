"""Split protection for the v4-exact pipeline (plan test 8 and the record-1 rule).

assert_selection_patches(patches)
    Selection / model-choice inputs (V candidates, q learners, thresholds) may come only from the
    training and selection patches {'15.14', '15.15'}.  Raises SplitViolation for any other patch
    (15.16 test, 16.x external, anything unparseable) or for an empty patch set.
assert_record_exists(path, expected_sha256=None)
    A script that touches 15.16 (or any held-out patch) must first find the pre-registration record
    (record 1): an existing, non-empty regular file; a .json record must parse as a JSON object.
    Optionally its SHA-256 must equal expected_sha256.  Returns the file's SHA-256 hex digest.
assert_patch_access(patches, record_path=None, expected_sha256=None)
    Convenience for scripts that read several patches: the selection patches are always allowed;
    any other patch requires assert_record_exists(record_path).

Patches are compared as Match-V5 'major.minor' strings ('15.14'); '15.14.701.4353' and 15.14 are
normalised to '15.14'.  Neither function reads any engagement data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Set, Union

SELECTION_PATCHES = frozenset({"15.14", "15.15"})

PatchSpec = Union[str, float, Iterable[Union[str, float]]]


class SplitViolation(RuntimeError):
    """A split-protection rule was violated (wrong patch in a selection input, or missing record)."""


def normalize_patch(p: Union[str, float]) -> str:
    """'15.14', '15.14.701.4353', 'Version 15.14' -> '15.14'.  Floats are refused (15.1 vs 15.10)."""
    if isinstance(p, float):
        raise SplitViolation(f"patch {p!r} given as float is ambiguous; pass a string")
    nums = re.findall(r"\d+", str(p))
    if len(nums) < 2:
        raise SplitViolation(f"unparseable patch {p!r}")
    return f"{int(nums[0])}.{int(nums[1])}"


def _patch_set(patches: PatchSpec) -> Set[str]:
    if isinstance(patches, (str, float)):
        patches = [patches]
    out = {normalize_patch(p) for p in patches}
    if not out:
        raise SplitViolation("empty patch set")
    return out


def assert_selection_patches(patches: PatchSpec) -> Set[str]:
    """Raise SplitViolation unless every patch is 15.14 or 15.15; returns the normalised set."""
    ps = _patch_set(patches)
    bad = sorted(ps - SELECTION_PATCHES)
    if bad:
        raise SplitViolation(f"selection inputs may use only {sorted(SELECTION_PATCHES)}; got {bad}")
    return ps


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_record_exists(path: Union[str, Path, None], expected_sha256: Optional[str] = None) -> str:
    """Raise SplitViolation unless the record file exists (non-empty; JSON object if .json) and, when
    expected_sha256 is given, has that SHA-256.  Returns the SHA-256 hex digest."""
    if path is None or str(path) == "":
        raise SplitViolation("no pre-registration record given; held-out patches are refused")
    p = Path(path)
    if not p.is_file():
        raise SplitViolation(f"pre-registration record not found: {p}")
    if p.stat().st_size == 0:
        raise SplitViolation(f"pre-registration record is empty: {p}")
    if p.suffix.lower() == ".json":
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SplitViolation(f"pre-registration record is not valid JSON: {p} ({e})") from e
        if not isinstance(blob, dict) or not blob:
            raise SplitViolation(f"pre-registration record must be a non-empty JSON object: {p}")
    digest = _sha256(p)
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).strip().lower():
        raise SplitViolation(f"pre-registration record hash mismatch: {p} has {digest}, expected {expected_sha256}")
    return digest


def assert_patch_access(patches: PatchSpec, record_path: Union[str, Path, None] = None,
                        expected_sha256: Optional[str] = None) -> Set[str]:
    """Selection patches always pass; any other patch requires the record (assert_record_exists)."""
    ps = _patch_set(patches)
    if ps - SELECTION_PATCHES:
        assert_record_exists(record_path, expected_sha256)
    return ps
