"""Constants shared by the v4-exact stage-2 scripts (ev4_01_detect, ev4_02_extract, ev4_03_fit_v).

Author pre-decisions (made in chat before any full-run result):
  outputs/reest_exact_v4_20260925/records/stage2_predecisions_20260925T151437Z.json
  sha256 b724ed7973bab4ab5481092b73b9bd890103a8d32fd92d96be0eb1759b7be101 (pinned below; any edit of the record
  is refused by predecisions_info()).

Remake rule (record key 'remakes'): every match whose GAME_END timestamp is < 300,000 ms is excluded from V
training, from the engagement detection outputs used in analysis, from labels and from all evaluations, on every
patch.  The cache meta has no remake flag, so GAME_END (the earliest GAME_END event timestamp; matches_full.json
'duration_ms' equals it for every match checked) is the only criterion.  A match with GAME_END exactly 300,000 ms
is kept.
  ev4_01_detect   excludes at the match-list stage (duration_ms < 300,000) and again in the worker from the events
                  (GAME_END < 300,000, catches a missing / wrong duration_ms); status 'remake_excluded', no rows;
                  count and ids in diag_<patch>.json 'remakes'.
  ev4_02_extract  skips such a match in the worker (status 'remake_excluded', no rows of any kind); count and ids
                  in every chunk sidecar and in manifest.json 'remakes'.
  ev4_03_fit_v    refuses an extract whose manifest does not carry the same rule, and asserts that no V row and no
                  martingale row has game_end < 300,000.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

OUT_BASE = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
PREDECISIONS_RECORD = OUT_BASE / "records" / "stage2_predecisions_20260925T151437Z.json"
PREDECISIONS_SHA256 = "b724ed7973bab4ab5481092b73b9bd890103a8d32fd92d96be0eb1759b7be101"

REMAKE_MAX_GAME_END_MS = 300_000          # GAME_END < this -> remake, excluded everywhere
REMAKE_STATUS = "remake_excluded"


def is_remake(game_end_ms: Optional[int]) -> bool:
    """True when GAME_END is known and < 300,000 ms.  Unknown GAME_END (None) is not a remake by this rule."""
    return game_end_ms is not None and int(game_end_ms) < REMAKE_MAX_GAME_END_MS


def remake_rule() -> Dict[str, Any]:
    """The rule as written into every diag / manifest / report (ev4_03 compares it)."""
    return {"threshold_ms": REMAKE_MAX_GAME_END_MS, "rule": "exclude if GAME_END < threshold_ms",
            "status": REMAKE_STATUS, "record": str(PREDECISIONS_RECORD), "record_sha256": PREDECISIONS_SHA256}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def predecisions_info(path: Optional[Path] = None, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """{path, sha256, content} of the author pre-decision record; refuses a missing or edited record.

    Defaults are read at call time (PREDECISIONS_RECORD / PREDECISIONS_SHA256).  ev4_01 / ev4_02 / ev4_03 call this
    at the top of main(), before any data is read, and again when writing the diag / manifest / report."""
    p = Path(PREDECISIONS_RECORD if path is None else path)
    expected_sha256 = PREDECISIONS_SHA256 if expected_sha256 is None else expected_sha256
    if not p.is_file():
        raise FileNotFoundError(f"author pre-decision record missing: {p}")
    digest = _sha256_file(p)
    if digest != expected_sha256:
        raise RuntimeError(f"pre-decision record {p} has sha256 {digest}, expected {expected_sha256}")
    return {"path": str(p), "sha256": digest, "content": json.loads(p.read_text(encoding="utf-8"))}
