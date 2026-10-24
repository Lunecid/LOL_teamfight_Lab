"""External validation: NA1 16.13 as a third set (author instruction, DECISION_LOG_KO.md 10-09 17:04).

The author added NA1 16.13 (every match of the NA1 raw root with queueId 420, mapId 11, gameMode CLASSIC, a timeline
file and patch 16.13) without a separate pre-specification; the procedure is that of the 10-09 12:10 row (frozen V,
calibrators and q cells, same cohort rules, same reported metrics).  KR 16.13 reused its old-lineage cache after
builder parity; the NA1 cache is instead built fresh from raw with the E1 builder (build_cache.build_one), so no
old-lineage NA1 file enters the new lineage.  The old-lineage NA1 cache is read only by ``compare-old``, for
information.

This module registers the dataset at import (E.RAW / E.CACHE_DIR / E.PATCH_INDEX; spawn workers re-import it as
``__mp_main__`` and so see the same registration) and reuses the E1 functions unchanged.  It never edits the E1
files, SAMPLE_MANIFEST.json or the E1 reports.  Only membership rule: every passing 16.13 match; ranks follow
h = SHA256("external_20261009:" + match_id) as in the KR sets.

Modes:
  sample       scan the NA1 details, keep every 16.13 candidate, hash the raw files and freeze
               SAMPLE_MANIFEST_NA1.json (+ .sha256) before any NA1 cache file exists.
  build        build_cache.mode_build for na1_16_13 (resumable; BUILD_MANIFEST_na1_16_13.json and the patch index).
  check        build_cache.mode_check for na1_16_13 (POSTBUILD_CHECK_na1_16_13.json).
  compare-old  file-by-file comparison of the fresh cache with the old-lineage NA1 16.13 cache (NA1_OLD_CACHE_COMPARE.json).

Usage: python -B studies/external_validation_20261009/na1.py sample --workers 3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import studies.external_validation_20261009 as E  # noqa: E402
from studies.external_validation_20261009 import build_cache as B  # noqa: E402  (sets the builder environment)
from studies.external_validation_20261009 import parity_compare as P  # noqa: E402
from studies.external_validation_20261009 import sample as S  # noqa: E402

VERSION = "external_validation_na1_20261009"
DATASET_NA1 = "na1_16_13"
PATCH_NA1 = "16.13"
RAW_NA1 = Path(r"D:\LOL_Project\data\raw\2026_current\na1")
OLD_CACHE_NA1 = ROOT / "outputs" / "full_corpus_training_20260915" / "external" / "NA1_16.13" / "cache"
SAMPLE_MANIFEST_NA1 = E.REPORTS / "SAMPLE_MANIFEST_NA1.json"
SAMPLE_MANIFEST_NA1_SHA = E.REPORTS / "SAMPLE_MANIFEST_NA1.sha256"
COMPARE_REPORT = E.REPORTS / "NA1_OLD_CACHE_COMPARE.json"

E.RAW.setdefault(DATASET_NA1, RAW_NA1)
E.CACHE_DIR.setdefault(DATASET_NA1, E.DATA_ROOT / "cache_na1_16_13")
E.PATCH_INDEX.setdefault(DATASET_NA1, E.DATA_ROOT / "match_cache_na1_16_13_patch_index.json")

RULE_TEXT = (
    "Author instruction (DECISION_LOG_KO.md 10-09 17:04), no separate pre-specification; procedure of the 10-09 12:10 "
    "row. Candidates: every NA1 match-detail JSON under D:/LOL_Project/data/raw/2026_current/na1/detail whose "
    "info.queueId == 420, info.mapId == 11, info.gameMode == 'CLASSIC' and whose timeline/<match_id>.json file exists; "
    "patch = major.minor of info.gameVersion; match_id = file-name stem. Set na1_16_13: every candidate with patch 16.13, "
    "ranked by h = SHA256('external_20261009:' + match_id).hexdigest(). Frozen here before any NA1 cache file is built; "
    "the cache is built fresh from raw (the old-lineage NA1 cache is not reused)."
)


def _use_na1_manifest() -> None:
    """build_cache.load_manifest reads E.SAMPLE_MANIFEST at call time; point it at the NA1 manifest (this process only)."""
    E.SAMPLE_MANIFEST = SAMPLE_MANIFEST_NA1
    E.SAMPLE_MANIFEST_SHA = SAMPLE_MANIFEST_NA1_SHA


def mode_sample(args) -> int:
    if SAMPLE_MANIFEST_NA1.exists() and not args.force:
        raise SystemExit(f"{SAMPLE_MANIFEST_NA1} exists: the NA1 sample is frozen")
    cache_dir = E.CACHE_DIR[DATASET_NA1]
    if cache_dir.exists() and any(os.scandir(cache_dir)):
        raise SystemExit(f"{cache_dir} already holds files: the sample must be frozen before any cache is built")
    E.set_below_normal_priority()
    log = E.Log(E.LOGS / "na1_sample.log")
    t_start = time.time()
    raw_root = RAW_NA1
    d_stems = S._stems(raw_root / "detail")
    t_stems = set(S._stems(raw_root / "timeline"))
    log(f"{DATASET_NA1}: listed detail={len(d_stems)} timeline={len(t_stems)}")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        tasks = [dict(folder=str(raw_root / "detail"), stems=d_stems[i:i + args.chunk]) for i in range(0, len(d_stems), args.chunk)]
        rows = [row for res in pool.map(S._scan_details, tasks) for row in res]
        scan_seconds = round(time.time() - t_start, 1)
        reasons, by_patch, builds, matchid_ne_stem = Counter(), {}, {}, 0
        for row in rows:
            reason, patch = S._classify(row, t_stems)
            reasons[reason] += 1
            if reason == "candidate":
                by_patch.setdefault(patch, []).append(row[0])
                builds[row[0]] = str(row[5])
                matchid_ne_stem += int(row[6] is False and patch == PATCH_NA1)
        cands = sorted(by_patch.get(PATCH_NA1, []))
        chosen = E.select_smallest(cands, len(cands))
        log(f"{DATASET_NA1}: reasons={dict(reasons)} other_patches={ {p: len(v) for p, v in by_patch.items() if p != PATCH_NA1} } "
            f"selected={len(chosen)}")
        t1 = time.time()
        items = [(str(raw_root), stem) for stem in chosen]
        raw_info = {stem: rec for res in pool.map(S._hash_raw, [dict(items=items[i:i + 200]) for i in range(0, len(items), 200)])
                    for stem, rec in res}
        hash_seconds = round(time.time() - t1, 1)
    kr = {m["match_id"] for m in E.read_json(E.SAMPLE_MANIFEST)["matches"]} if E.SAMPLE_MANIFEST.exists() else set()
    if kr & set(chosen):
        raise SystemExit("NA1 match ids overlap the KR sample manifest")
    matches = [dict(match_id=stem, patch=PATCH_NA1, build=builds[stem], dataset=DATASET_NA1, h=E.match_hash(stem), rank_in_patch=rank,
                    detail_bytes=raw_info[stem]["detail_bytes"], detail_sha256=raw_info[stem]["detail_sha256"],
                    timeline_bytes=raw_info[stem]["timeline_bytes"], timeline_sha256=raw_info[stem]["timeline_sha256"])
               for rank, stem in enumerate(chosen)]
    summary = {DATASET_NA1: dict(
        raw_root=str(raw_root), detail_files=len(d_stems), timeline_files=len(t_stems),
        detail_without_timeline=len(set(d_stems) - t_stems), timeline_without_detail=len(t_stems - set(d_stems)),
        scan_reasons=dict(sorted(reasons.items())),
        candidates_other_patches={p: len(v) for p, v in sorted(by_patch.items()) if p != PATCH_NA1},
        per_patch={PATCH_NA1: dict(candidates=len(cands), selected=len(chosen),
                                   candidate_ids_sha256=E.sha256_bytes("\n".join(cands).encode("utf-8")),
                                   selected_ids_sha256=E.sha256_bytes("\n".join(sorted(chosen)).encode("utf-8")),
                                   builds=dict(Counter(builds[s] for s in chosen).most_common()),
                                   metadata_matchId_ne_stem_candidates=matchid_ne_stem)},
        scan_seconds=scan_seconds)}
    manifest = dict(
        version=VERSION, kind="external_sample_manifest", frozen_utc=E.utc_now(), rule=RULE_TEXT, hash_tag=E.HASH_TAG,
        filters=E.FILTERS, datasets=summary, counts={f"{DATASET_NA1}|{PATCH_NA1}": len(matches)}, total_selected=len(matches),
        kr_manifest_sha256=E.sha256_file(E.SAMPLE_MANIFEST) if E.SAMPLE_MANIFEST.exists() else None,
        environment=dict(python=sys.version.split()[0], platform=platform.platform(), json_parser="stdlib json",
                         workers=args.workers, script=str(Path(__file__).resolve()), script_sha256=E.sha256_file(Path(__file__).resolve()),
                         sample_module_sha256=E.sha256_file(E.PACKAGE / "sample.py"),
                         package_init_sha256=E.sha256_file(E.PACKAGE / "__init__.py")),
        timing=dict(scan_seconds=scan_seconds, raw_hash_seconds=hash_seconds, total_seconds=round(time.time() - t_start, 1)),
        identifiers="no player identifiers are read into this file (match ids, versions, counts and file hashes only)",
        matches=matches)
    digest = E.write_json(SAMPLE_MANIFEST_NA1, manifest, list_key="matches")
    E.atomic_write_bytes(SAMPLE_MANIFEST_NA1_SHA, f"{digest}  {SAMPLE_MANIFEST_NA1.name}\n".encode("ascii"))
    log(f"frozen {SAMPLE_MANIFEST_NA1} sha256={digest} selected={len(matches)} total {time.time() - t_start:.0f}s")
    return 0


def mode_build(args) -> int:
    _use_na1_manifest()
    log = E.Log(E.LOGS / "na1_build.log")
    return B.mode_build(SimpleNamespace(dataset=DATASET_NA1, workers=args.workers, chunk=args.chunk,
                                        retry_failed=args.retry_failed), log)


def mode_check(args) -> int:
    _use_na1_manifest()
    log = E.Log(E.LOGS / "na1_check.log")
    return B.mode_check(SimpleNamespace(dataset=DATASET_NA1, workers=args.workers, determinism_n=args.determinism_n), log)


def _compare_chunk(task):
    E.set_below_normal_priority()
    out = []
    for mid in task["ids"]:
        c = P.compare_match(mid, task["fresh"], task["old"])
        out.append(dict(match_id=mid, equal=bool(c.get("equal")), missing=c.get("missing"),
                        npz_file_bytes_equal=(c.get("npz") or {}).get("file_bytes_equal"),
                        meta_equal=(c.get("meta") or {}).get("equal"), events_equal=(c.get("events") or {}).get("equal"),
                        npz_equal=(c.get("npz") or {}).get("equal")))
    return out


def mode_compare_old(args) -> int:
    log = E.Log(E.LOGS / "na1_compare_old.log")
    t0 = time.time()
    _use_na1_manifest()
    manifest, man_sha = B.load_manifest()
    bm = E.read_json(E.REPORTS / f"BUILD_MANIFEST_{DATASET_NA1}.json")
    ok_ids = sorted(r["match_id"] for r in bm["records"] if r["status"] != "failed")
    fresh = E.CACHE_DIR[DATASET_NA1]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        tasks = [dict(ids=ok_ids[i:i + 250], fresh=str(fresh), old=str(OLD_CACHE_NA1)) for i in range(0, len(ok_ids), 250)]
        recs = [r for res in pool.map(_compare_chunk, tasks) for r in res]
    old_ids = {e.name.partition(".")[0] for e in os.scandir(OLD_CACHE_NA1) if e.name.endswith(".meta.json")}
    doc = dict(version=VERSION, kind="na1_old_cache_compare", created_utc=E.utc_now(), purpose="information only; the fresh cache is the one used",
               sample_manifest_sha256=man_sha, fresh_cache=str(fresh), old_cache=str(OLD_CACHE_NA1),
               fresh_matches=len(ok_ids), old_cache_matches=len(old_ids), fresh_not_in_old=len(set(ok_ids) - old_ids),
               old_not_in_fresh=len(old_ids - set(ok_ids)), compared=len(recs), equal=sum(r["equal"] for r in recs),
               npz_file_bytes_equal=sum(bool(r["npz_file_bytes_equal"]) for r in recs),
               not_equal=[r for r in recs if not r["equal"]][:50], seconds=round(time.time() - t0, 1))
    E.write_json(COMPARE_REPORT, doc)
    log(f"compare-old: equal {doc['equal']}/{doc['compared']} fresh_not_in_old={doc['fresh_not_in_old']} "
        f"old_not_in_fresh={doc['old_not_in_fresh']} in {doc['seconds']}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sample", "build", "check", "compare-old"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--determinism-n", type=int, default=5)
    args = ap.parse_args()
    if not 1 <= args.workers <= E.MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{E.MAX_WORKERS}")
    if args.mode == "sample" and args.chunk == 100:
        args.chunk = 500
    return {"sample": mode_sample, "build": mode_build, "check": mode_check, "compare-old": mode_compare_old}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
