"""E2a step 1: freeze the parity subset (written before any E2a computation).

README
  Rule: the 500 matches of patch 15.16 (2025 patch index) that have at least one row with role 5 (TEST) in
  D:/LOL_Training_20261004/q_run_1/Q_ROWS_h90.npz (field canonical_match_key) and the smallest
  h = SHA256("parity_20261009:" + match_id).hexdigest(), match_id = 2025 cache file-name stem (= patch-index key),
  ascending h (ties by match_id).  Only the two fields canonical_match_key and role are read from Q_ROWS_h90.npz;
  no label, score or outcome field is opened.  original_index = position of the match in the 2025 patch index (the
  same field the accepted bank and preparation store).

Usage: python -B studies/external_validation_20261009/e2a_parity_subset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402


def select(log=print) -> dict:
    import numpy as np

    if C.PARITY_SUBSET.exists():
        raise FileExistsError(f"{C.PARITY_SUBSET} exists; the parity subset is frozen")
    index = C.read_json(C.REF_PATCH_INDEX_2025)
    pairs = list(index.items())
    with np.load(C.Q_ROWS_H90, allow_pickle=False) as archive:
        keys = archive["canonical_match_key"]
        roles = archive["role"]
    test_keys = {bytes(k).decode("ascii") for k in np.unique(keys[roles == C.Q_TEST_ROLE_CODE])}
    candidates = []
    n_patch = 0
    for original_index, (match_id, patch) in enumerate(pairs):
        if patch != C.PARITY_PATCH:
            continue
        n_patch += 1
        key = C.canonical_key(match_id)
        if key in test_keys:
            candidates.append((C.parity_hash(match_id), match_id, key, original_index))
    candidates.sort()
    chosen = sorted(candidates[: C.PARITY_N], key=lambda row: row[3])
    if len(chosen) != C.PARITY_N:
        raise ValueError(f"only {len(chosen)} parity candidates")
    log(f"15.16 matches in index {n_patch}; with a TEST q row {len(candidates)}; selected {len(chosen)}")
    out = {
        "version": C.VERSION,
        "kind": "E2a_PARITY_SUBSET",
        "created_utc": C.utc_now(),
        "rule": ("The 500 matches of patch 15.16 in the 2025 patch index that have at least one row with role 5 in "
                 "Q_ROWS_h90.npz (field canonical_match_key) and the smallest SHA256('parity_20261009:' + match_id)"
                 ".hexdigest() (UTF-8, lowercase hex, ascending, ties by match_id); match_id = 2025 cache file stem. "
                 "Only canonical_match_key and role are read from Q_ROWS_h90.npz."),
        "inputs": {
            "patch_index": str(C.REF_PATCH_INDEX_2025),
            "patch_index_sha256": C.sha256_file(C.REF_PATCH_INDEX_2025),
            "q_rows": str(C.Q_ROWS_H90),
            "q_rows_sha256": C.sha256_file(C.Q_ROWS_H90),
            "q_rows_fields_read": ["canonical_match_key", "role"],
            "q_test_role_code": C.Q_TEST_ROLE_CODE,
        },
        "counts": {"patch_15_16_in_index": n_patch, "patch_15_16_with_test_q_row": len(candidates),
                   "q_test_distinct_keys": len(test_keys), "selected": len(chosen)},
        "h_max_selected": max(row[0] for row in chosen),
        "identifiers": "match ids and canonical match keys only; no player identifiers",
        "matches": [{"match_id": mid, "canonical_match_key": key, "original_index": oi, "h": h}
                    for h, mid, key, oi in chosen],
    }
    C.write_json(C.PARITY_SUBSET, out)
    return out


def main() -> int:
    C.runtime_hygiene()
    log = C.Log(C.LOGS / "E2a_parity_subset.log")
    out = select(log)
    log(f"wrote {C.PARITY_SUBSET} sha256={C.sha256_file(C.PARITY_SUBSET)} n={out['counts']['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
