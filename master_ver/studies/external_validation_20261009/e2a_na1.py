"""E2a dataset registration: na1_16_13, the third external set (author decision, DECISION_LOG_KO.md row 10-09 17:04).

README
  Registers na1_16_13 additively in the e2a_common registry at import time.  It is kept out of e2a_common.py on
  purpose: the kr_15_18_22 / kr_16_13 shared-history identities (PLAN.json 'code') bind the sha256 of e2a_common.py and
  e2a_history.py, so neither file is edited after those runs.  NA1 runs through exactly the stage modules used for
  kr_16_13 (same patch 16.13, build 16.13.791.5903, Data Dragon 16.13.1); there is no separate parity gate (the 15.16
  parity PASS covers this code path).
  Inputs: reports/external_validation_20261009/SAMPLE_MANIFEST_NA1.json (+ .sha256), cache
  D:/LOL_External_20261009/cache_na1_16_13 (built fresh from raw by na1.py with the unchanged E1 builder), patch index
  D:/LOL_External_20261009/match_cache_na1_16_13_patch_index.json, BUILD_MANIFEST_na1_16_13.json and
  POSTBUILD_CHECK_na1_16_13.json.  preflight() refuses to run unless the post-build check says passed: true and the
  manifest, build manifest, patch index and cache agree; the result is written to E2a_NA1_PREFLIGHT.json.
  Outputs follow the external layout DATA_ROOT/<stage>/na1_16_13.  Match ids carry the prefix NA1_.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

DATASET_NA1 = "na1_16_13"
SAMPLE_MANIFEST_NA1 = C.REPORTS / "SAMPLE_MANIFEST_NA1.json"
SAMPLE_MANIFEST_NA1_SHA = C.REPORTS / "SAMPLE_MANIFEST_NA1.sha256"
BUILD_MANIFEST_NA1 = C.REPORTS / "BUILD_MANIFEST_na1_16_13.json"
POSTBUILD_CHECK_NA1 = C.REPORTS / "POSTBUILD_CHECK_na1_16_13.json"
PREFLIGHT = C.REPORTS / "E2a_NA1_PREFLIGHT.json"
DECISION = "reports/thesis_notes_20261005/DECISION_LOG_KO.md row '10-09 17:04 (외부 검증 추가, 저자 지시)'"


def register() -> None:
    if DATASET_NA1 in C.DATASETS:
        return
    C.DATASETS = C.DATASETS + (DATASET_NA1,)
    C.CACHE[DATASET_NA1] = C.DATA_ROOT / "cache_na1_16_13"
    C.PATCH_INDEX[DATASET_NA1] = C.DATA_ROOT / "match_cache_na1_16_13_patch_index.json"
    C.EXPECTED_PATCHES[DATASET_NA1] = ("16.13",)


def preflight(log=print) -> dict:
    register()
    problems = []
    check = C.read_json(POSTBUILD_CHECK_NA1) if POSTBUILD_CHECK_NA1.is_file() else {}
    if check.get("passed") is not True:
        problems.append("POSTBUILD_CHECK_na1_16_13.json missing or not passed")
    manifest_sha = C.sha256_file(SAMPLE_MANIFEST_NA1)
    recorded = SAMPLE_MANIFEST_NA1_SHA.read_text(encoding="utf-8").split()[0] if SAMPLE_MANIFEST_NA1_SHA.is_file() else None
    if recorded != manifest_sha or check.get("sample_manifest_sha256") != manifest_sha:
        problems.append("sample manifest sha256 differs from its .sha256 file or the post-build check")
    if check.get("build_manifest_sha256") != C.sha256_file(BUILD_MANIFEST_NA1):
        problems.append("build manifest sha256 differs from the post-build check")
    manifest = C.read_json(SAMPLE_MANIFEST_NA1)
    build = C.read_json(BUILD_MANIFEST_NA1)
    index = C.read_json(C.PATCH_INDEX[DATASET_NA1])
    ids = {row["match_id"] for row in manifest["matches"]}
    if set(index) != ids or set(index.values()) != {"16.13"} or len(index) != manifest["total_selected"]:
        problems.append("patch index differs from the sample manifest or holds a patch other than 16.13")
    if build.get("patch_index_sha256") != C.sha256_file(C.PATCH_INDEX[DATASET_NA1]):
        problems.append("patch index sha256 differs from the build manifest")
    if Path(build.get("cache_dir", "")).resolve() != C.CACHE[DATASET_NA1].resolve():
        problems.append("build manifest cache_dir differs from the registered cache")
    n_files = sum(1 for _ in os.scandir(C.CACHE[DATASET_NA1]))
    if n_files != 3 * len(index):
        problems.append(f"cache holds {n_files} files, expected {3 * len(index)}")
    result = {"version": C.VERSION, "kind": "E2a_NA1_PREFLIGHT", "created_utc": C.utc_now(), "dataset": DATASET_NA1,
              "decision": DECISION, "passed": not problems, "problems": problems,
              "sample_manifest": str(SAMPLE_MANIFEST_NA1), "sample_manifest_sha256": manifest_sha,
              "build_manifest_sha256": C.sha256_file(BUILD_MANIFEST_NA1),
              "postbuild_check_sha256": C.sha256_file(POSTBUILD_CHECK_NA1) if POSTBUILD_CHECK_NA1.is_file() else None,
              "patch_index": str(C.PATCH_INDEX[DATASET_NA1]), "patch_index_sha256": C.sha256_file(C.PATCH_INDEX[DATASET_NA1]),
              "cache_dir": str(C.CACHE[DATASET_NA1]), "cache_files": n_files, "n_matches": len(index),
              "builds": sorted({row.get("build") for row in manifest["matches"]}),
              "registration": "e2a_na1.register() (additive; e2a_common.py unchanged)"}
    C.write_json(PREFLIGHT, result)
    log(f"[na1] preflight passed={result['passed']} problems={problems}")
    return result


register()
