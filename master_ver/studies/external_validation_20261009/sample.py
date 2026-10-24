"""E1 task 1: freeze the external sample (written before any cache is built).

Scans every KR detail JSON of both raw roots (stdlib json, the parser the cache builder uses), keeps
queueId 420 / mapId 11 / gameMode CLASSIC with an existing timeline file, sets patch = major.minor of
info.gameVersion, and selects per 2025 patch the 8,000 smallest h = SHA256("external_20261009:" + match_id)
(match_id = file-name stem); KR 16.13 keeps every passing match.  Only the hash decides membership; no outcome,
timeline content or other field is read for selection.  Raw detail/timeline sizes and SHA-256 are then recorded for
the selected matches, and SAMPLE_MANIFEST.json + SAMPLE_MANIFEST.sha256 are written atomically.  Player identifiers
are never read into the outputs (counts only).

Usage: python -B studies/external_validation_20261009/sample.py --workers 3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import studies.external_validation_20261009 as E  # noqa: E402

RULE_TEXT = (
    "Pre-registered (DECISION_LOG_KO.md 10-09 12:10; METHODS_MEMO_KO.md 6.12). Candidates: every KR match-detail JSON "
    "under the raw root's detail/ folder whose info.queueId == 420, info.mapId == 11, info.gameMode == 'CLASSIC' and "
    "whose timeline/<match_id>.json file exists; patch = major.minor of info.gameVersion; match_id = file-name stem. "
    "Primary set kr_15_18_22 (C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr): for each patch in "
    "15.18, 15.19, 15.20, 15.21, 15.22 the 8,000 candidates with the smallest h = SHA256('external_20261009:' + "
    "match_id).hexdigest() (UTF-8, lowercase hex, ascending). Secondary set kr_16_13 "
    "(D:/LOL_Project/data/raw/2026_current/kr): every candidate with patch 16.13. Builder-parity subset: the 200 "
    "kr_16_13 matches with the smallest h. Selection uses nothing but the hash; it is frozen here before any cache is "
    "built."
)


def _scan_details(task):
    """Parse detail JSONs; returns (stem, status, queueId, mapId, gameMode, gameVersion, matchId_eq_stem)."""
    E.set_below_normal_priority()
    folder = Path(task["folder"])
    out = []
    for stem in task["stems"]:
        try:
            raw = (folder / f"{stem}.json").read_bytes()
        except OSError as exc:
            out.append((stem, f"detail_read_error:{type(exc).__name__}", None, None, None, None, None))
            continue
        try:
            d = json.loads(raw)
        except ValueError as exc:
            out.append((stem, f"detail_json_error:{type(exc).__name__}", None, None, None, None, None))
            continue
        if not isinstance(d, dict):
            out.append((stem, "detail_not_object", None, None, None, None, None))
            continue
        info = d.get("info") or {}
        meta = d.get("metadata") or {}
        if not isinstance(info, dict):
            info = {}
        out.append((stem, "ok", info.get("queueId"), info.get("mapId"), info.get("gameMode"),
                    info.get("gameVersion"), (meta.get("matchId") == stem) if isinstance(meta, dict) else None))
    return out


def _hash_raw(task):
    E.set_below_normal_priority()
    out = []
    for raw_root, stem in task["items"]:
        rec = {}
        for kind in ("detail", "timeline"):
            p = Path(raw_root) / kind / f"{stem}.json"
            rec[f"{kind}_bytes"] = p.stat().st_size
            rec[f"{kind}_sha256"] = E.sha256_file(p)
        out.append((stem, rec))
    return out


def _stems(folder: Path) -> list[str]:
    return sorted(e.name[:-5] for e in os.scandir(folder) if e.is_file() and e.name.endswith(".json"))


def _classify(row, timeline_stems):
    stem, status, q, m, mode, gv, _eq = row
    if status != "ok":
        return status, ""
    if q is None and m is None and mode is None and gv is None:
        return "detail_info_empty_or_missing", ""
    try:
        q_ok = int(q) == E.FILTERS["queueId"]
    except (TypeError, ValueError):
        q_ok = False
    try:
        m_ok = int(m) == E.FILTERS["mapId"]
    except (TypeError, ValueError):
        m_ok = False
    patch = E.major_minor(gv)
    if not q_ok:
        return "queueId_ne_420", patch
    if not m_ok:
        return "mapId_ne_11", patch
    if mode != E.FILTERS["gameMode"]:
        return "gameMode_ne_CLASSIC", patch
    if not patch:
        return "gameVersion_missing_or_unparsable", patch
    if stem not in timeline_stems:
        return "timeline_missing", patch
    return "candidate", patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen manifest (refused by default)")
    args = ap.parse_args()
    if not 1 <= args.workers <= E.MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{E.MAX_WORKERS}")
    if E.SAMPLE_MANIFEST.exists() and not args.force:
        raise SystemExit(f"{E.SAMPLE_MANIFEST} exists: the sample is frozen (use --force only to rebuild before any cache)")
    for d in E.CACHE_DIR.values():
        if d.exists() and any(os.scandir(d)):
            raise SystemExit(f"{d} already holds files: the sample must be frozen before any cache is built")
    E.set_below_normal_priority()
    log = E.Log(E.LOGS / "sample.log")
    t_start = time.time()
    log(f"sample start; workers={args.workers} chunk={args.chunk} version={E.VERSION}")

    datasets = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for ds, raw_root in E.RAW.items():
            t0 = time.time()
            d_stems = _stems(raw_root / "detail")
            t_stems = set(_stems(raw_root / "timeline"))
            log(f"{ds}: listed detail={len(d_stems)} timeline={len(t_stems)} in {time.time() - t0:.1f}s")
            tasks = [dict(folder=str(raw_root / "detail"), stems=d_stems[i:i + args.chunk])
                     for i in range(0, len(d_stems), args.chunk)]
            rows = []
            done = 0
            for res in pool.map(_scan_details, tasks):
                rows.extend(res)
                done += len(res)
                if done % 50000 < args.chunk:
                    log(f"{ds}: scanned {done}/{len(d_stems)} ({time.time() - t0:.0f}s)")
            datasets[ds] = dict(raw_root=raw_root, d_stems=d_stems, t_stems=t_stems, rows=rows,
                                scan_seconds=round(time.time() - t0, 1))
            log(f"{ds}: scan done {len(rows)} details in {time.time() - t0:.1f}s")

        selected = []          # (dataset, patch, stem, gameVersion, h, rank)
        summary = {}
        parity = []
        for ds, info in datasets.items():
            reasons = Counter()
            by_patch = defaultdict(list)
            builds = {}
            matchid_ne_stem = Counter()
            for row in info["rows"]:
                reason, patch = _classify(row, info["t_stems"])
                reasons[reason] += 1
                if reason == "candidate":
                    by_patch[patch].append(row[0])
                    builds[row[0]] = str(row[5])
                    if row[6] is False:
                        matchid_ne_stem[patch] += 1
            want = E.PATCHES_2025 if ds == E.DATASET_2025 else (E.PATCH_2026,)
            per_patch = {}
            for patch in want:
                cands = sorted(by_patch.get(patch, []))
                n_take = E.PER_PATCH if ds == E.DATASET_2025 else len(cands)
                chosen = E.select_smallest(cands, n_take)
                for rank, stem in enumerate(chosen):
                    selected.append((ds, patch, stem, builds[stem], E.match_hash(stem), rank))
                per_patch[patch] = dict(
                    candidates=len(cands), selected=len(chosen),
                    candidate_ids_sha256=E.sha256_bytes("\n".join(cands).encode("utf-8")),
                    selected_ids_sha256=E.sha256_bytes("\n".join(sorted(chosen)).encode("utf-8")),
                    h_min=E.match_hash(chosen[0]) if chosen else None,
                    h_max_selected=E.match_hash(chosen[-1]) if chosen else None,
                    builds=dict(Counter(builds[s] for s in chosen).most_common()),
                    candidate_builds=dict(Counter(builds[s] for s in cands).most_common()),
                    metadata_matchId_ne_stem_candidates=matchid_ne_stem.get(patch, 0))
                if ds == E.DATASET_2026:
                    parity = chosen[: E.PARITY_N]
            other = {p: len(v) for p, v in sorted(by_patch.items()) if p not in want}
            summary[ds] = dict(
                raw_root=str(info["raw_root"]), detail_files=len(info["d_stems"]), timeline_files=len(info["t_stems"]),
                detail_without_timeline=len(set(info["d_stems"]) - info["t_stems"]),
                timeline_without_detail=len(info["t_stems"] - set(info["d_stems"])),
                scan_reasons=dict(sorted(reasons.items())), candidates_other_patches=other, per_patch=per_patch,
                scan_seconds=info["scan_seconds"])
            log(f"{ds}: reasons={dict(reasons)} other_patches={other} "
                f"selected={ {p: v['selected'] for p, v in per_patch.items()} }")

        t1 = time.time()
        items = [(str(E.RAW[ds]), stem) for ds, _p, stem, _b, _h, _r in selected]
        tasks = [dict(items=items[i:i + 200]) for i in range(0, len(items), 200)]
        raw_info = {}
        done = 0
        for res in pool.map(_hash_raw, tasks):
            for stem, rec in res:
                raw_info[stem] = rec
            done += len(res)
            if done % 10000 < 200:
                log(f"raw sha256: {done}/{len(items)} ({time.time() - t1:.0f}s)")
        hash_seconds = round(time.time() - t1, 1)

    matches = []
    for ds, patch, stem, build, h, rank in selected:
        r = raw_info[stem]
        matches.append(dict(match_id=stem, patch=patch, build=build, dataset=ds, h=h, rank_in_patch=rank,
                            detail_bytes=r["detail_bytes"], detail_sha256=r["detail_sha256"],
                            timeline_bytes=r["timeline_bytes"], timeline_sha256=r["timeline_sha256"]))
    if len({m["match_id"] for m in matches}) != len(matches):
        raise SystemExit("duplicate match_id across datasets")
    manifest = dict(
        version=E.VERSION, kind="external_sample_manifest", frozen_utc=E.utc_now(), rule=RULE_TEXT,
        hash_tag=E.HASH_TAG, per_patch_n=E.PER_PATCH, filters=E.FILTERS,
        datasets=summary,
        counts=dict(Counter(f"{m['dataset']}|{m['patch']}" for m in matches)),
        total_selected=len(matches),
        parity_subset=dict(dataset=E.DATASET_2026, n=len(parity), rule="the 200 kr_16_13 matches with the smallest h",
                           match_ids=parity),
        environment=dict(python=sys.version.split()[0], platform=platform.platform(), json_parser="stdlib json",
                         workers=args.workers, script=str(Path(__file__).resolve()),
                         script_sha256=E.sha256_file(Path(__file__).resolve()),
                         package_init_sha256=E.sha256_file(E.PACKAGE / "__init__.py")),
        timing=dict(scan_seconds={ds: v["scan_seconds"] for ds, v in summary.items()}, raw_hash_seconds=hash_seconds,
                    total_seconds=round(time.time() - t_start, 1)),
        identifiers="no player identifiers are read into this file (match ids, versions, counts and file hashes only)",
        matches=matches,
    )
    digest = E.write_json(E.SAMPLE_MANIFEST, manifest, list_key="matches")
    E.atomic_write_bytes(E.SAMPLE_MANIFEST_SHA, f"{digest}  SAMPLE_MANIFEST.json\n".encode("ascii"))
    log(f"frozen {E.SAMPLE_MANIFEST} sha256={digest} total_selected={len(matches)} "
        f"counts={manifest['counts']} total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
