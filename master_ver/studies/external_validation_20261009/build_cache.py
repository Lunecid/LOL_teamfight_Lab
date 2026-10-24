"""E1 tasks 2-4: builder parity on KR 16.13, resumable cache build of the frozen 15.18-15.22 sample, E1 report.

Builder: the old repository's code, imported read-only (never edited, no bytecode written into it):
  data.cache_io._parse_timeline_to_minute_cache_compat -> gameplay.pipeline_cache.parse_timeline_to_minute_cache,
  gameplay.fights.build_anchors_from_events / normalize_patch, core.roles.get_role_slots_from_detail,
  data.cache_io._extract_static_meta_from_detail / _interp_policy_snapshot / load_match_cache.
The per-match writer is the one of scripts/fc20260915_external_prepare.py (which produced the old-lineage KR 16.13
cache and mirrors data.cache_io.prebuild_cache field for field): npz = savez_compressed(minute_ts, node_minute,
global_minute, gold_team_minute, xy_raw_minute); events.json = json.dumps(events, ensure_ascii=False);
meta.json = {match_id, patch_full, patch, team_map, role_slots, anchors, anchor_is_norm, static_meta, interp_policy,
feature_version}.  Environment as that driver: LOL_CFG_PRESET=v3.3 (detector-only constants, none read by the
builder), LOL_CFG_OVERRIDES {CACHE_DIRNAME, FIGHT_INDEX_CACHE_ENABLED, FIGHT_INDEX_NUM_WORKERS, DUMP_FIGHTS,
CACHE_IN_RAM}, LOL_OUTPUT_ROOT redirected to D:/LOL_External_20261009/runtime_repo (import-time mkdirs land there).
Raw bytes are checked against the frozen SAMPLE_MANIFEST.json (size + SHA-256) before parsing.

Each match is written to a per-process staging folder, verified there (load_match_cache, 2025 reference format:
meta keys, interp_policy, feature_version, npz names/dtypes/trailing dims, patch/build, static channels vs detail,
no player identifier in the written bytes) and only then moved into the cache folder, so the cache folder holds
verified matches only.  A re-run skips matches whose three files exist and verify.

Modes:
  parity  build the 200 parity matches into D:/LOL_External_20261009/parity_16_13/cache, compare file by file with
          the old-lineage KR 16.13 caches, audit the inventory-named cache for all selected 16.13 matches.
  build   --dataset kr_15_18_22 (kr_16_13 only if parity failed): resumable build + patch index + BUILD_MANIFEST.
  check   independent post-build checks (census, patch index, re-hash of every file, determinism re-build); added
          after the build run, see CODE_AT_RUNS.json.
  report  E1_REPORT.json from the artifacts.

Usage: python -B studies/external_validation_20261009/build_cache.py --mode parity --workers 3
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import studies.external_validation_20261009 as E  # noqa: E402

BUILDER_ENV = {
    "LOL_OUTPUT_ROOT": str(E.RUNTIME_REPO),
    "LOL_CFG_PRESET": "v3.3",
    "LOL_CFG_OVERRIDES": json.dumps({
        "CACHE_DIRNAME": str(E.RUNTIME_REPO / "_unused_cache_dir"), "FIGHT_INDEX_CACHE_ENABLED": False,
        "FIGHT_INDEX_NUM_WORKERS": 1, "DUMP_FIGHTS": False, "CACHE_IN_RAM": False}),
}
os.environ.update(BUILDER_ENV)

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from studies.external_validation_20261009 import parity_compare as P  # noqa: E402

BUILDER_VERSION = "e1_builder_v1_20261009"
NPZ_ORDER = ("minute_ts", "node_minute", "global_minute", "gold_team_minute", "xy_raw_minute")
CFG_FIELDS = ("FEATURE_VERSION", "MAX_T", "FRAME_MS", "BIN_MS", "COORD_NORM_DIV", "USE_STATUS_FEATURES", "USE_BUFF_STATUS",
              "USE_DRAGON_SOUL_STATUS", "USE_ULT_LEVEL", "BUFF_DUR_SEC", "CHAMPION_NAME_VOCAB", "PATCH_LEVEL",
              "CACHE_IN_RAM", "INTERP_METHOD", "INTERP_XY", "INTERP_SCALARS", "INTERP_XY_METHOD", "INTERP_XY_CURVE",
              "INTERP_EXP_K", "INTERP_SCALARS_METHOD", "XY_DISCONT_DIST_RAW", "XY_DISCONT_USE_ALIVE", "XY_GUARD_MODE")
IDENT_KEYS = {"puuid", "summonerName", "summonerId", "riotIdGameName", "riotIdTagline", "riotIdName", "accountId",
              "gameName", "tagLine", "profileIcon", "summonerLevel"}
_CTX: dict = {}


# --------------------------------------------------------------------------------------------- old-repo context
def _ctx() -> dict:
    if not _CTX:
        if str(E.OLD_REPO) not in sys.path:
            sys.path.insert(0, str(E.OLD_REPO))
        import core.config as cc
        from core.config import cfg
        import data.cache_io as cio
        import gameplay.pipeline_cache as pc
        from gameplay.fights import build_anchors_from_events, normalize_patch
        from core.roles import get_role_slots_from_detail
        import gameplay.fights as gf
        import core.roles as cr
        for mod in (cc, cio, pc, gf, cr):
            if not str(Path(mod.__file__).resolve()).startswith(str(E.OLD_REPO)):
                raise RuntimeError(f"{mod.__name__} imported from {mod.__file__}, not the old repository")
        cfg.CACHE_IN_RAM = False
        _CTX.update(cfg=cfg, cc=cc, cio=cio, pc=pc, anchors=build_anchors_from_events, norm=normalize_patch,
                    roles=get_role_slots_from_detail,
                    champion_col=int(cc.NODE_IDX["champion_id"]))
    return _CTX


def builder_snapshot() -> dict:
    ctx = _ctx()
    cfg, cc = ctx["cfg"], ctx["cc"]
    return dict(
        cfg={k: (str(getattr(cfg, k)) if isinstance(getattr(cfg, k, None), Path) else getattr(cfg, k, None)) for k in CFG_FIELDS},
        interp_policy=ctx["cio"]._interp_policy_snapshot(), F_NODE=int(cc.F_NODE), F_GLOBAL=int(cc.F_GLOBAL),
        MAP_MAX=float(cc.MAP_MAX), node_feature_names_sha256=E.sha256_bytes(json.dumps(list(cc.NODE_FEATURE_NAMES)).encode()),
        global_feature_names_sha256=E.sha256_bytes(json.dumps(list(cc.GLOBAL_FEATURE_NAMES)).encode()),
        preset_values=__import__("core.presets", fromlist=["PRESETS"]).PRESETS["v3.3"],
        cache_dir_at_import=str(ctx["cio"].CACHE_DIR),
        old_repo_modules=sorted(str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                                if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(E.OLD_REPO))),
    )


def _init_worker():
    E.set_below_normal_priority()
    _ctx()


# --------------------------------------------------------------------------------------------- reference format
def reference_format(per_patch: int = 10) -> dict:
    """Format of the 2025 corpus cache from the per_patch smallest-h matches of each patch (all must agree)."""
    index = E.read_json(E.REF_PATCH_INDEX_2025)
    by_patch = defaultdict(list)
    for mid, patch in index.items():
        by_patch[patch].append(mid)
    sample = [m for p in sorted(by_patch) for m in E.select_smallest(by_patch[p], per_patch)]
    seen = None
    for mid in sample:
        meta = E.read_json(E.REF_CACHE_2025 / f"{mid}.meta.json")
        with np.load(E.REF_CACHE_2025 / f"{mid}.npz", allow_pickle=False) as z:
            names = list(z.files)
            arrs = {k: z[k] for k in names}
        fmt = dict(meta_keys=list(meta.keys()), static_meta_keys=list(meta["static_meta"].keys()),
                   interp_policy=meta["interp_policy"], feature_version=meta["feature_version"], npz_names=names,
                   npz_dtypes={k: str(a.dtype) for k, a in arrs.items()},
                   npz_trailing_dims={k: list(a.shape[1:]) for k, a in arrs.items()},
                   npz_ndim={k: int(a.ndim) for k, a in arrs.items()})
        if seen is None:
            seen = fmt
        elif fmt != seen:
            raise SystemExit(f"2025 reference cache format is not uniform (match {mid})")
    seen.update(sample_match_ids=sample, sample_rule=f"{per_patch} smallest-h matches per patch of the 2025 patch index",
                cache=str(E.REF_CACHE_2025))
    return seen


# --------------------------------------------------------------------------------------------- verification
def _cache_names(mid: str) -> dict:
    return dict(npz=f"{mid}.npz", events=f"{mid}.events.json", meta=f"{mid}.meta.json")


def verify_cache(mid: str, folder: Path, row: dict, ref: dict, detail: dict | None = None) -> tuple[bool, dict, dict]:
    """Checks of a written match against the 2025 reference format; returns (ok, checks, facts)."""
    ctx = _ctx()
    cio = ctx["cio"]
    cio.CACHE_DIR = Path(folder)
    names = _cache_names(mid)
    checks, facts = {}, {}
    pack = cio.load_match_cache(mid)
    checks["load_match_cache"] = pack is not None
    if pack is None:
        return False, checks, facts
    meta = E.read_json(Path(folder) / names["meta"])
    checks["meta_match_id"] = meta.get("match_id") == mid
    checks["meta_patch"] = meta.get("patch") == row["patch"] and pack["meta"]["patch"] == row["patch"]
    checks["meta_patch_full"] = meta.get("patch_full") == row["build"]
    checks["meta_keys"] = list(meta.keys()) == ref["meta_keys"]
    checks["static_meta_keys"] = isinstance(meta.get("static_meta"), dict) and list(meta["static_meta"].keys()) == ref["static_meta_keys"]
    checks["interp_policy"] = meta.get("interp_policy") == ref["interp_policy"]
    checks["feature_version"] = meta.get("feature_version") == ref["feature_version"]
    with np.load(Path(folder) / names["npz"], allow_pickle=False) as z:
        znames = list(z.files)
        arrs = {k: z[k] for k in znames}
    checks["npz_names"] = znames == ref["npz_names"]
    checks["npz_dtypes"] = {k: str(a.dtype) for k, a in arrs.items()} == ref["npz_dtypes"]
    checks["npz_dims"] = ({k: list(a.shape[1:]) for k, a in arrs.items()} == ref["npz_trailing_dims"]
                          and {k: int(a.ndim) for k, a in arrs.items()} == ref["npz_ndim"])
    t_len = {int(a.shape[0]) for a in arrs.values()}
    checks["npz_T_consistent"] = len(t_len) == 1 and min(t_len) >= 1
    facts.update(T=int(arrs["minute_ts"].shape[0]) if "minute_ts" in arrs else None, n_events=len(pack["events"]))
    if detail is not None and "node_minute" in arrs:
        parts = (detail.get("info") or {}).get("participants") or []
        champ = {int(p.get("participantId", 0) or 0): int(p.get("championId", 0) or 0) for p in parts if isinstance(p, dict)}
        col = ctx["champion_col"]
        checks["static_channels_match_detail"] = all(
            champ.get(pid, 0) <= 0 or float(arrs["node_minute"][0, pid - 1, col]) == float(champ[pid]) for pid in range(1, 11))
    ok = all(bool(v) for v in checks.values())
    return ok, checks, facts


def identifier_hits(detail: dict, blobs: list[bytes], events) -> dict:
    """Counts only: puuid/summonerId strings of this match found in the written bytes; identifier-like keys in events."""
    toks = set()
    meta = detail.get("metadata") or {}
    for p in meta.get("participants") or []:
        if isinstance(p, str) and len(p) >= 30:
            toks.add(p)
    for p in (detail.get("info") or {}).get("participants") or []:
        if isinstance(p, dict):
            for k in ("puuid", "summonerId"):
                v = p.get(k)
                if isinstance(v, str) and len(v) >= 30:
                    toks.add(v)
    hits = sum(1 for t in toks for b in blobs if t.encode("utf-8") in b)
    keys = set()
    stack = list(events) if isinstance(events, list) else []
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            keys.update(k for k in o.keys() if k in IDENT_KEYS)
            stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
        elif isinstance(o, list):
            stack.extend(v for v in o if isinstance(v, (dict, list)))
    return dict(tokens_checked=len(toks), token_hits=hits, ident_keys=sorted(keys))


# --------------------------------------------------------------------------------------------- per-match build
def _move_aside(paths: dict, failed_dir: Path):
    failed_dir.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        if Path(p).exists():
            os.replace(p, failed_dir / Path(p).name)


def build_one(row: dict, final_dir: Path, staging: Path, failed_dir: Path, ref: dict) -> dict:
    ctx = _ctx()
    cio = ctx["cio"]
    t0 = time.time()
    mid = row["match_id"]
    names = _cache_names(mid)
    final = {k: final_dir / n for k, n in names.items()}
    rec = dict(match_id=mid, dataset=row["dataset"], patch=row["patch"], build=row["build"], status="failed", reason="")
    if all(p.exists() for p in final.values()):
        ok, checks, facts = verify_cache(mid, final_dir, row, ref)
        if ok:
            rec.update(status="skipped_existing_verified", **facts, **_file_facts(final), seconds=round(time.time() - t0, 3))
            return rec
        rec["existing_failed_checks"] = [k for k, v in checks.items() if not v]
    staged = {k: staging / n for k, n in names.items()}
    try:
        raw = E.RAW[row["dataset"]]
        paths = dict(detail=raw / "detail" / f"{mid}.json", timeline=raw / "timeline" / f"{mid}.json")
        blobs = {}
        for kind, p in paths.items():
            if not p.exists():
                rec["reason"] = f"raw_missing:{kind}"
                return rec
            blobs[kind] = p.read_bytes()
            if len(blobs[kind]) != int(row[f"{kind}_bytes"]) or E.sha256_bytes(blobs[kind]) != row[f"{kind}_sha256"]:
                rec["reason"] = f"raw_size_or_sha256_mismatch_vs_sample_manifest:{kind}"
                return rec
        try:
            detail = json.loads(blobs["detail"])
        except ValueError as exc:
            rec["reason"] = f"detail_json_error:{type(exc).__name__}"
            return rec
        try:
            tl = json.loads(blobs["timeline"])
        except ValueError as exc:
            rec["reason"] = f"timeline_json_error:{type(exc).__name__}:{str(exc)[:120]}"
            return rec
        del blobs
        if not isinstance(detail, dict) or not isinstance(tl, dict):
            rec["reason"] = "raw_json_not_object"
            return rec
        info = detail.get("info", {}) or {}
        gv = str(info.get("gameVersion", ""))
        cond = dict(queue=int(info.get("queueId", -1) or -1) == E.FILTERS["queueId"],
                    map=int(info.get("mapId", -1) or -1) == E.FILTERS["mapId"],
                    mode=info.get("gameMode") == E.FILTERS["gameMode"],
                    version=E.major_minor(gv) == row["patch"], build=gv == row["build"])
        if not all(cond.values()):
            rec["reason"] = "game_condition_failed:" + ",".join(k for k, v in cond.items() if not v)
            return rec
        parts = info.get("participants", None)
        tm = {}
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict):
                    pid, tid = int(p.get("participantId", 0) or 0), int(p.get("teamId", 0) or 0)
                    if pid > 0 and tid in (100, 200):
                        tm[pid] = tid
        if not isinstance(parts, list) or len(parts) < 10 or len(tm) < 10:
            rec["reason"] = "participants_or_team_map_incomplete"
            return rec
        frames = (tl.get("info", {}) or {}).get("frames", []) or []
        rec["frames_raw"] = len(frames) if isinstance(frames, list) else None
        cache = cio._parse_timeline_to_minute_cache_compat(tl, tm, detail=detail)
        if not cache or any(k not in cache for k in ("minute_ts", "node_minute", "global_minute")):
            diag = ""
            try:
                ctx["pc"].parse_timeline_to_minute_cache(tl, tm, detail=detail, cfg_in=ctx["cfg"])
            except Exception as exc:  # diagnostic only; the builder's own result above decides
                diag = f":{type(exc).__name__}:{str(exc)[:120]}"
            rec["reason"] = "timeline_parse_empty_or_missing_arrays" + diag
            return rec
        anchors = ctx["anchors"](cache["events"])
        arrays = dict(minute_ts=cache["minute_ts"], node_minute=cache["node_minute"], global_minute=cache["global_minute"],
                      gold_team_minute=cache["gold_team_minute"])
        if cache.get("xy_raw_minute", None) is not None:
            arrays["xy_raw_minute"] = cache["xy_raw_minute"]
        staging.mkdir(parents=True, exist_ok=True)
        with open(staged["npz"], "wb") as f:
            np.savez_compressed(f, **arrays)
        ev_bytes = json.dumps(cache["events"], ensure_ascii=False).encode("utf-8")
        staged["events"].write_bytes(ev_bytes)
        rec["n_events_written"] = len(cache["events"])
        meta = {"match_id": mid, "patch_full": gv, "patch": ctx["norm"](gv), "team_map": tm,
                "role_slots": ctx["roles"](detail), "anchors": anchors, "anchor_is_norm": False,
                "static_meta": cio._extract_static_meta_from_detail(detail),
                "interp_policy": cio._interp_policy_snapshot(),
                "feature_version": str(getattr(ctx["cfg"], "FEATURE_VERSION", ""))}
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        staged["meta"].write_bytes(meta_bytes)
        ident = identifier_hits(detail, [ev_bytes, meta_bytes], cache["events"])
        rec["identifier_check"] = dict(tokens_checked=ident["tokens_checked"], token_hits=ident["token_hits"],
                                       ident_keys=ident["ident_keys"])
        if ident["token_hits"] or ident["ident_keys"]:
            for p in staged.values():  # own temporary files; never kept when they would store identifiers
                if p.exists():
                    p.unlink()
            rec["reason"] = "identifier_found_in_cache_output"
            return rec
        ok, checks, facts = verify_cache(mid, staging, row, ref, detail=detail)
        rec.update(facts)
        if not ok:
            rec["reason"] = "verification_failed:" + ",".join(k for k, v in checks.items() if not v)
            _move_aside(staged, failed_dir)
            return rec
        final_dir.mkdir(parents=True, exist_ok=True)
        for k in ("npz", "events", "meta"):
            os.replace(staged[k], final[k])
        rec.update(status="built", **_file_facts(final))
    except Exception as exc:
        rec["reason"] = f"exception:{type(exc).__name__}:{str(exc)[:200]}"
        rec["traceback_tail"] = traceback.format_exc()[-600:]
        try:
            _move_aside(staged, failed_dir)
        except Exception:
            pass
    finally:
        rec["seconds"] = round(time.time() - t0, 3)
    return rec


def _file_facts(paths: dict) -> dict:
    out = {}
    for k in ("npz", "events", "meta"):
        p = Path(paths[k])
        out[f"{k}_bytes"] = p.stat().st_size
        out[f"{k}_sha256"] = E.sha256_file(p)
    return out


def build_chunk(task: dict) -> dict:
    staging = Path(task["staging_root"]) / f"p{os.getpid()}"
    recs = [build_one(r, Path(task["final_dir"]), staging, Path(task["failed_dir"]), task["ref"]) for r in task["rows"]]
    return dict(chunk_id=task["chunk_id"], plan_sha256=task["plan_sha256"], records=recs)


# --------------------------------------------------------------------------------------------- manifest helpers
def load_manifest() -> tuple[dict, str]:
    digest = E.sha256_file(E.SAMPLE_MANIFEST)
    frozen = E.SAMPLE_MANIFEST_SHA.read_text(encoding="ascii").split()[0]
    if digest != frozen:
        raise SystemExit(f"SAMPLE_MANIFEST.json sha256 {digest} != frozen {frozen}")
    return E.read_json(E.SAMPLE_MANIFEST), digest


def run_chunks(rows: list[dict], final_dir: Path, work: Path, ref: dict, manifest_sha: str, workers: int, chunk: int,
               log: E.Log, retry_failed: bool = False) -> list[dict]:
    chunks_dir = work / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    tasks, files = [], []
    for c, s0 in enumerate(range(0, len(rows), chunk)):
        part = rows[s0:s0 + chunk]
        plan = E.sha256_bytes(json.dumps(dict(manifest=manifest_sha, builder=BUILDER_VERSION, final_dir=str(final_dir),
                                              ids=[r["match_id"] for r in part]), sort_keys=True).encode("utf-8"))
        cf = chunks_dir / f"chunk_{c:04d}.json"
        files.append((cf, plan))
        if cf.exists():
            prev = E.read_json(cf)
            if prev.get("plan_sha256") == plan and (not retry_failed or all(r["status"] != "failed" for r in prev["records"])):
                continue
        tasks.append(dict(chunk_id=c, plan_sha256=plan, rows=part, final_dir=str(final_dir), failed_dir=str(work / "failed"),
                          staging_root=str(work / "staging"), ref=ref, path=str(cf)))
    total = len(rows)
    done = total - sum(len(t["rows"]) for t in tasks)
    log(f"chunks: {len(files)} total, {len(tasks)} to run, {done} matches already recorded; workers={workers}")
    status = Counter()
    t0 = time.time()
    n_run = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
            futs = {ex.submit(build_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                res = fut.result()
                E.write_json(Path(t["path"]), res, list_key="records")
                for r in res["records"]:
                    status[r["status"]] += 1
                done += len(t["rows"])
                n_run += len(t["rows"])
                rate = n_run / max(1e-9, time.time() - t0)
                eta = (total - done) / max(1e-9, rate)
                log(f"progress {done}/{total} this_run={n_run} {dict(status)} rate={rate:.2f}/s eta={eta / 60:.1f} min")
    records = []
    for cf, plan in files:
        d = E.read_json(cf)
        if d.get("plan_sha256") != plan:
            raise SystemExit(f"stale chunk {cf}")
        records.extend(d["records"])
    return records


def _summarise(records: list[dict]) -> dict:
    by = defaultdict(Counter)
    for r in records:
        by[r["patch"]][r["status"]] += 1
    fail = [dict(match_id=r["match_id"], patch=r["patch"], reason=r["reason"]) for r in records if r["status"] == "failed"]
    ok = [r for r in records if r["status"] != "failed"]
    return dict(per_patch={p: dict(v) for p, v in sorted(by.items())},
                ok=len(ok), failed=len(fail), failure_reasons=dict(Counter(f["reason"].split(":")[0] for f in fail)),
                failures=fail,
                identifier_token_hits=sum((r.get("identifier_check") or {}).get("token_hits", 0) for r in records),
                identifier_keys_found=sorted({k for r in records for k in (r.get("identifier_check") or {}).get("ident_keys", [])}),
                frames=dict(min=min((r["T"] for r in ok), default=None), max=max((r["T"] for r in ok), default=None),
                            mean=round(float(np.mean([r["T"] for r in ok])), 2) if ok else None))


def _final_folder_check(final_dir: Path, records: list[dict]) -> dict:
    ok = {r["match_id"]: r for r in records if r["status"] != "failed"}
    present = Counter()
    extra = []
    size_mismatch = 0
    for e in os.scandir(final_dir):
        mid, _, suffix = e.name.partition(".")
        if mid in ok and suffix in ("npz", "events.json", "meta.json"):
            present[mid] += 1
            k = {"npz": "npz", "events.json": "events", "meta.json": "meta"}[suffix]
            size_mismatch += int(e.stat().st_size != ok[mid][f"{k}_bytes"])
        else:
            extra.append(e.name)
    missing = [m for m in ok if present[m] != 3]
    return dict(files=sum(present.values()), expected_files=3 * len(ok), matches_complete=len(ok) - len(missing),
                missing=missing[:50], n_missing=len(missing), extra_files=extra[:50], n_extra=len(extra),
                size_mismatch=size_mismatch, ok=not missing and not extra and size_mismatch == 0)


# --------------------------------------------------------------------------------------------- parity mode
def _audit_ref_chunk(task):
    """Integrity + format audit of reference-cache matches (no writes)."""
    E.set_below_normal_priority()
    out = []
    prepared = task["prepared"]
    for row in task["rows"]:
        mid = row["match_id"]
        folder = Path(task["folder"])
        r = dict(match_id=mid)
        names = _cache_names(mid)
        r["exists"] = all((folder / n).exists() for n in names.values())
        if r["exists"]:
            if prepared is not None:
                rec = prepared.get(mid)
                r["cache_sha256_equal_prepared_manifest"] = bool(rec) and all(
                    E.sha256_file(folder / names[k]) == rec[f"{k}_sha256"] for k in ("npz", "events", "meta"))
            ok, checks, facts = verify_cache(mid, folder, row, task["ref"])
            r.update(format_ok=ok, failed_checks=[k for k, v in checks.items() if not v], **facts)
        out.append(r)
    return out


def mode_parity(args, log: E.Log) -> int:
    t_start = time.time()
    manifest, man_sha = load_manifest()
    ref = reference_format()
    snap = builder_snapshot()
    if snap["interp_policy"] != ref["interp_policy"] or snap["cfg"]["FEATURE_VERSION"] != ref["feature_version"]:
        raise SystemExit("builder interp_policy/feature_version differ from the 2025 reference cache")
    rows16 = [m for m in manifest["matches"] if m["dataset"] == E.DATASET_2026]
    by_id = {m["match_id"]: m for m in rows16}
    parity_ids = manifest["parity_subset"]["match_ids"]
    if parity_ids != [m for m in E.select_smallest(by_id, E.PARITY_N)]:
        raise SystemExit("parity subset in the manifest is not the 200 smallest-h kr_16_13 matches")
    out_dir = E.PARITY_DIR
    if out_dir.exists() and any(os.scandir(out_dir)) and not args.resume:
        raise SystemExit(f"{out_dir} is not empty (pass --resume to verify/skip existing files)")
    work = E.WORK / "parity_16_13"
    log(f"parity: building {len(parity_ids)} kr_16_13 matches into {out_dir}")
    t0 = time.time()
    records = run_chunks([by_id[m] for m in parity_ids], out_dir, work, ref, man_sha, args.workers, 20, log)
    build_seconds = round(time.time() - t0, 1)
    built_ok = [r for r in records if r["status"] != "failed"]
    log(f"parity build: {Counter(r['status'] for r in records)} in {build_seconds}s")

    refs = {"primary": E.OLD_CACHE_16_13, "alternate": E.OLD_CACHE_16_13_ALT}
    comparisons = {}
    for name, folder in refs.items():
        res = [P.compare_match(r["match_id"], out_dir, folder) for r in built_ok]
        agg = dict(
            reference=str(folder), compared=len(res), equal_all=sum(r["equal"] for r in res),
            missing=[r["match_id"] for r in res if "missing" in r],
            meta_bytes_equal=sum(r.get("meta", {}).get("bytes_equal", False) for r in res),
            meta_differing_top_level_keys=dict(Counter(k for r in res for k in r.get("meta", {}).get("differing_top_level_keys", []))),
            events_bytes_equal=sum(r.get("events", {}).get("bytes_equal", False) for r in res),
            npz_names_equal=sum(r.get("npz", {}).get("names_equal", False) for r in res),
            npz_order_equal=sum(r.get("npz", {}).get("order_equal", False) for r in res),
            npz_equal=sum(r.get("npz", {}).get("equal", False) for r in res),
            npz_file_bytes_equal_info=sum(bool(r.get("npz", {}).get("file_bytes_equal", False)) for r in res),
            npz_arrays={k: dict(dtype_equal=sum(r["npz"]["arrays"].get(k, {}).get("dtype_equal", False) for r in res if "npz" in r),
                                shape_equal=sum(r["npz"]["arrays"].get(k, {}).get("shape_equal", False) for r in res if "npz" in r),
                                values_equal=sum(r["npz"]["arrays"].get(k, {}).get("values_equal", False) for r in res if "npz" in r),
                                bytes_equal=sum(r["npz"]["arrays"].get(k, {}).get("bytes_equal", False) for r in res if "npz" in r),
                                max_abs_diff=max([r["npz"]["arrays"].get(k, {}).get("max_abs_diff", 0.0) for r in res if "npz" in r] or [0.0]))
                        for k in NPZ_ORDER},
            not_equal=[dict(match_id=r["match_id"], meta_paths=r.get("meta", {}).get("differing_paths", [])[:10],
                            events_bytes_equal=r.get("events", {}).get("bytes_equal"),
                            events_parsed_equal=r.get("events", {}).get("parsed_equal"),
                            npz={k: {kk: vv for kk, vv in v.items() if kk in ("dtype", "shape", "values_equal", "max_abs_diff", "n_diff")}
                                 for k, v in r.get("npz", {}).get("arrays", {}).items() if not v.get("equal")})
                       for r in res if not r["equal"]][:20],
        )
        agg["exact"] = bool(agg["compared"] == len(parity_ids) and agg["equal_all"] == agg["compared"])
        comparisons[name] = agg
        log(f"parity vs {name} ({folder}): equal_all={agg['equal_all']}/{agg['compared']} meta_bytes={agg['meta_bytes_equal']} "
            f"events_bytes={agg['events_bytes_equal']} npz={agg['npz_equal']} meta_diff_keys={agg['meta_differing_top_level_keys']}")

    # audit of the inventory-named reference cache for every selected kr_16_13 match
    t1 = time.time()
    ids16 = sorted(by_id)
    primary_ids = sorted(e.name[: -len(".meta.json")] for e in os.scandir(E.OLD_CACHE_16_13) if e.name.endswith(".meta.json"))
    alt_ids = {e.name[: -len(".meta.json")] for e in os.scandir(E.OLD_CACHE_16_13_ALT) if e.name.endswith(".meta.json")}
    pre = {r["match_id"]: r for r in csv.DictReader(open(E.OLD_PREFLIGHT_16_13, encoding="utf-8", newline=""))}
    raw_equal = sum(1 for m in ids16 if m in pre and pre[m]["detail_sha256"] == by_id[m]["detail_sha256"]
                    and pre[m]["timeline_sha256"] == by_id[m]["timeline_sha256"]
                    and int(pre[m]["detail_bytes"]) == by_id[m]["detail_bytes"] and int(pre[m]["timeline_bytes"]) == by_id[m]["timeline_bytes"])
    prepared_doc = E.read_json(E.OLD_CACHE_16_13_PREPARED)
    prepared = {r["match_id"]: {k: r.get(k) for k in ("npz_sha256", "events_sha256", "meta_sha256")}
                for r in prepared_doc["matches"] if r.get("cache_status") == "written"}
    tasks = [dict(rows=[by_id[m] for m in ids16[i:i + 200]], folder=str(E.OLD_CACHE_16_13), ref=ref,
                  prepared={m: prepared.get(m) for m in ids16[i:i + 200]}) for i in range(0, len(ids16), 200)]
    audit = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        for res in ex.map(_audit_ref_chunk, tasks):
            audit.extend(res)
    audit_seconds = round(time.time() - t1, 1)
    ref_audit = dict(
        cache=str(E.OLD_CACHE_16_13), selected=len(ids16), cache_matches=len(primary_ids),
        selected_equals_cache_set=set(ids16) == set(primary_ids),
        selected_not_in_cache=len(set(ids16) - set(primary_ids)), cache_not_selected=len(set(primary_ids) - set(ids16)),
        alternate_cache=str(E.OLD_CACHE_16_13_ALT), alternate_cache_matches=len(alt_ids),
        alternate_contains_all_selected=set(ids16) <= alt_ids,
        raw_sha256_and_size_equal_09_15_preflight=raw_equal, preflight_csv=str(E.OLD_PREFLIGHT_16_13),
        all_three_files_exist=sum(r["exists"] for r in audit),
        cache_sha256_equal_prepared_manifest=sum(bool(r.get("cache_sha256_equal_prepared_manifest")) for r in audit),
        prepared_manifest=str(E.OLD_CACHE_16_13_PREPARED),
        prepared_manifest_written=prepared_doc.get("written"), prepared_manifest_feature_version=prepared_doc.get("detector_config", {}).get("FEATURE_VERSION"),
        full_format_ok=sum(bool(r.get("format_ok")) for r in audit),
        format_failed_checks=dict(Counter(c for r in audit for c in r.get("failed_checks", []))),
        frames=dict(min=min(r.get("T") or 0 for r in audit), max=max(r.get("T") or 0 for r in audit)),
        seconds=audit_seconds)
    n = len(ids16)
    ref_audit["ok"] = bool(ref_audit["selected_equals_cache_set"] and raw_equal == n and ref_audit["all_three_files_exist"] == n
                           and ref_audit["cache_sha256_equal_prepared_manifest"] == n and ref_audit["full_format_ok"] == n)
    log(f"reference audit: {json.dumps({k: v for k, v in ref_audit.items() if k not in ('format_failed_checks',)})}")

    passed = bool(comparisons["primary"]["exact"] and ref_audit["ok"] and len(built_ok) == len(parity_ids))
    doc = dict(
        version=E.VERSION, builder_version=BUILDER_VERSION, created_utc=E.utc_now(), sample_manifest_sha256=man_sha,
        criterion=("PASS when all 200 rebuilt matches equal the inventory-named old-lineage cache exactly (meta.json bytes "
                   "and keys, events.json bytes, every npz array dtype/shape/values/bytes) and the reference cache audit holds "
                   "for every selected kr_16_13 match (same id set, raw SHA-256 equal to the 09-15 preflight, cache files equal "
                   "to the 09-15 prepared manifest, 2025 full format)"),
        passed=passed, decision=("reuse the existing old-lineage KR 16.13 cache; no rebuild" if passed else
                                 "rebuild all KR 16.13 matches into D:/LOL_External_20261009/cache_kr_16_13"),
        parity_dir=str(out_dir), parity_build=dict(records=len(records), status=dict(Counter(r["status"] for r in records)),
                                                   failures=[dict(match_id=r["match_id"], reason=r["reason"]) for r in records if r["status"] == "failed"],
                                                   seconds=build_seconds),
        comparisons=comparisons, reference_audit=ref_audit, reference_format_2025=ref, builder=snap,
        builder_env=BUILDER_ENV, seconds_total=round(time.time() - t_start, 1))
    E.write_json(E.PARITY_REPORT, doc)
    if passed:
        index = {m: by_id[m]["patch"] for m in sorted(ids16)}
        E.atomic_write_bytes(E.PATCH_INDEX[E.DATASET_2026], json.dumps(index).encode("utf-8"))
        log(f"wrote {E.PATCH_INDEX[E.DATASET_2026]} ({len(index)} matches -> cache {E.OLD_CACHE_16_13})")
    log(f"parity {'PASSED' if passed else 'FAILED'}; report {E.PARITY_REPORT}")
    return 0


# --------------------------------------------------------------------------------------------- build mode
def mode_build(args, log: E.Log) -> int:
    t_start = time.time()
    ds = args.dataset
    manifest, man_sha = load_manifest()
    if ds == E.DATASET_2026:
        par = E.read_json(E.PARITY_REPORT) if E.PARITY_REPORT.exists() else None
        if par is None or par.get("passed"):
            raise SystemExit("kr_16_13 is rebuilt only after a FAILED parity report (parity passed or not run)")
    ref = reference_format()
    snap = builder_snapshot()
    if snap["interp_policy"] != ref["interp_policy"] or snap["cfg"]["FEATURE_VERSION"] != ref["feature_version"]:
        raise SystemExit("builder interp_policy/feature_version differ from the 2025 reference cache")
    rows = [m for m in manifest["matches"] if m["dataset"] == ds]
    final_dir = E.CACHE_DIR[ds]
    work = E.WORK / ds
    disk_before = shutil.disk_usage(str(E.DATA_ROOT.anchor))
    log(f"build {ds}: {len(rows)} matches -> {final_dir}; D: free {disk_before.free / 2**30:.1f} GiB; workers={args.workers}")
    records = run_chunks(rows, final_dir, work, ref, man_sha, args.workers, args.chunk, log, retry_failed=args.retry_failed)
    summ = _summarise(records)
    folder = _final_folder_check(final_dir, records)
    ok_ids = sorted(r["match_id"] for r in records if r["status"] != "failed")
    patch_of = {r["match_id"]: r["patch"] for r in records}
    index = {m: patch_of[m] for m in ok_ids}
    E.atomic_write_bytes(E.PATCH_INDEX[ds], json.dumps(index).encode("utf-8"))
    files, nbytes = E.dir_bytes(final_dir)
    disk_after = shutil.disk_usage(str(E.DATA_ROOT.anchor))
    doc = dict(version=E.VERSION, builder_version=BUILDER_VERSION, kind="build_manifest", dataset=ds, created_utc=E.utc_now(),
               sample_manifest_sha256=man_sha, cache_dir=str(final_dir), patch_index=str(E.PATCH_INDEX[ds]),
               patch_index_sha256=E.sha256_file(E.PATCH_INDEX[ds]), summary=summ, final_folder_check=folder,
               cache_files=files, cache_bytes=nbytes, disk_free_before=disk_before.free, disk_free_after=disk_after.free,
               workers=args.workers, chunk=args.chunk, seconds_this_run=round(time.time() - t_start, 1),
               build_seconds_sum=round(sum(r.get("seconds", 0.0) for r in records), 1),
               records=[{k: r.get(k) for k in ("match_id", "patch", "build", "status", "reason", "T", "n_events", "n_events_written", "frames_raw",
                                                "npz_bytes", "npz_sha256", "events_bytes", "events_sha256", "meta_bytes",
                                                "meta_sha256", "seconds")} for r in records])
    E.write_json(E.REPORTS / f"BUILD_MANIFEST_{ds}.json", doc, list_key="records")
    log(f"build {ds} done: ok={summ['ok']} failed={summ['failed']} reasons={summ['failure_reasons']} folder_ok={folder['ok']} "
        f"cache {nbytes / 2**30:.2f} GiB in {files} files; this run {time.time() - t_start:.0f}s")
    return 0


# --------------------------------------------------------------------------------------------- check mode
def _rehash_chunk(task):
    """Re-hash written cache files and compare with the build-manifest record (read-only)."""
    E.set_below_normal_priority()
    folder = Path(task["folder"])
    bad = []
    for rec in task["records"]:
        mid = rec["match_id"]
        for k, n in _cache_names(mid).items():
            p = folder / n
            if not p.exists() or p.stat().st_size != rec[f"{k}_bytes"] or E.sha256_file(p) != rec[f"{k}_sha256"]:
                bad.append(f"{mid}:{k}")
    return dict(n=len(task["records"]), bad=bad)


def mode_check(args, log: E.Log) -> int:
    """Independent post-build checks (added after the build run; see CODE_AT_RUNS.json): folder census, patch index vs
    sample manifest, every file re-hashed against BUILD_MANIFEST, T vs raw frame count, and a determinism re-build of the
    5 smallest-h matches per patch into work/determinism_check compared byte for byte with the cache."""
    t_start = time.time()
    ds = args.dataset
    manifest, man_sha = load_manifest()
    rows = {m["match_id"]: m for m in manifest["matches"] if m["dataset"] == ds}
    final_dir = E.CACHE_DIR[ds]
    bm_path = E.REPORTS / f"BUILD_MANIFEST_{ds}.json"
    bm = E.read_json(bm_path)
    recs = {r["match_id"]: r for r in bm["records"]}
    ok_recs = [r for r in bm["records"] if r["status"] != "failed"]
    suffix, ids, extra = Counter(), set(), []
    for e in os.scandir(final_dir):
        mid, _, suf = e.name.partition(".")
        if suf in ("npz", "events.json", "meta.json"):
            suffix[suf] += 1
            ids.add(mid)
        else:
            extra.append(e.name)
    index = E.read_json(E.PATCH_INDEX[ds])
    census = dict(files_by_suffix=dict(suffix), match_ids=len(ids), extra_files=extra[:20],
                  ids_equal_sample_manifest=ids == set(rows), ids_equal_ok_records=ids == {r["match_id"] for r in ok_recs},
                  patch_index_n=len(index), patch_index_per_patch=dict(Counter(index.values())),
                  patch_index_keys_equal_cache_ids=set(index) == ids, patch_index_sorted=list(index) == sorted(index),
                  patch_index_patch_equal_manifest=all(rows[k]["patch"] == v for k, v in index.items()),
                  patch_index_sha256_equal_build_manifest=E.sha256_file(E.PATCH_INDEX[ds]) == bm["patch_index_sha256"],
                  T_equals_raw_frames=sum(1 for r in ok_recs if r.get("frames_raw") == r.get("T")),
                  T_vs_raw_frames_unknown=sum(1 for r in ok_recs if r.get("frames_raw") is None))
    log(f"check {ds}: census {json.dumps(census)}")
    t1 = time.time()
    tasks = [dict(folder=str(final_dir), records=ok_recs[i:i + 500]) for i in range(0, len(ok_recs), 500)]
    bad, n = [], 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        for res in ex.map(_rehash_chunk, tasks):
            bad.extend(res["bad"])
            n += res["n"]
    rehash = dict(matches=n, files=3 * n, mismatching_files=len(bad), mismatches=bad[:20], seconds=round(time.time() - t1, 1))
    log(f"check {ds}: rehash {json.dumps(rehash)}")
    ref = reference_format()
    by_patch = defaultdict(list)
    for m in rows.values():
        by_patch[m["patch"]].append(m)
    pick = [m for p in sorted(by_patch) for m in sorted(by_patch[p], key=lambda r: r["h"])[: args.determinism_n]]
    det_dir = E.WORK / "determinism_check" / ds
    det = []
    for m in pick:
        rec = build_one(m, det_dir / "cache", det_dir / "staging", det_dir / "failed", ref)
        c = P.compare_match(m["match_id"], det_dir / "cache", final_dir)
        det.append(dict(match_id=m["match_id"], patch=m["patch"], status=rec["status"], equal=c["equal"],
                        npz_file_bytes_equal=c.get("npz", {}).get("file_bytes_equal"),
                        sha256_equal_build_manifest=all(rec.get(f"{k}_sha256") == recs[m["match_id"]].get(f"{k}_sha256")
                                                        for k in ("npz", "events", "meta"))))
    determinism = dict(rule=f"{args.determinism_n} smallest-h matches per patch rebuilt from raw into {det_dir / 'cache'}",
                       n=len(det), rebuilt_fresh=sum(d["status"] == "built" for d in det),
                       equal=sum(bool(d["equal"]) for d in det),
                       file_bytes_equal=sum(bool(d["npz_file_bytes_equal"]) for d in det),
                       sha256_equal_build_manifest=sum(d["sha256_equal_build_manifest"] for d in det), matches=det)
    log(f"check {ds}: determinism equal {determinism['equal']}/{determinism['n']}")
    passed = bool(census["ids_equal_sample_manifest"] and census["ids_equal_ok_records"] and not extra
                  and census["patch_index_keys_equal_cache_ids"] and census["patch_index_patch_equal_manifest"]
                  and census["patch_index_sha256_equal_build_manifest"] and rehash["mismatching_files"] == 0
                  and determinism["equal"] == determinism["n"] == determinism["sha256_equal_build_manifest"])
    doc = dict(version=E.VERSION, kind="postbuild_check", dataset=ds, created_utc=E.utc_now(), sample_manifest_sha256=man_sha,
               build_manifest=str(bm_path), build_manifest_sha256=E.sha256_file(bm_path), cache_dir=str(final_dir),
               census=census, rehash=rehash, determinism=determinism, passed=passed,
               seconds=round(time.time() - t_start, 1))
    E.write_json(E.REPORTS / f"POSTBUILD_CHECK_{ds}.json", doc)
    log(f"check {ds}: {'PASSED' if passed else 'FAILED'} in {time.time() - t_start:.0f}s")
    return 0


# --------------------------------------------------------------------------------------------- report mode
def _inventory_unknown_bucket() -> dict | None:
    """The inventory's empty-gameVersion detail bucket of the 15.18-15.22 raw root (explains queueId_ne_420)."""
    p = E.ROOT / "reports" / "journal_tog_20261005" / "evidence" / "EXTERNAL_DATA_INVENTORY_20261009.json"
    try:
        u = E.read_json(p)["inventory"]["candidates"]["Lol_project/matches/kr"]["by_patch"]["UNKNOWN"]
    except (OSError, KeyError, ValueError):
        return None
    keep = ("detail_files_parsed", "queueId", "mapId", "gameMode", "game_version_builds", "n_participants")
    return dict(source=str(p), **{k: u.get(k) for k in keep})


def mode_report(args, log: E.Log) -> int:
    manifest, man_sha = load_manifest()
    snap = builder_snapshot()
    parent = E.read_json(E.PARENT_SOURCE_HASHES).get("source_hashes", {}) if E.PARENT_SOURCE_HASHES.exists() else {}
    old_mods = {}
    for f in snap["old_repo_modules"]:
        cur = E.sha256_file(f)
        old_mods[f] = dict(sha256=cur, equal_to_09_15_parent_run=(parent.get(f) == cur) if f in parent else None)
    own = {str(p): E.sha256_file(p) for p in sorted(E.PACKAGE.rglob("*.py"))}
    car_path = E.REPORTS / "CODE_AT_RUNS.json"
    at_runs = E.read_json(car_path) if car_path.exists() else None
    env = manifest.get("environment", {})
    provenance = dict(
        code_at_runs=str(car_path) if at_runs else None,
        code_at_runs_sha256=E.sha256_file(car_path) if at_runs else None,
        sample_py_equal_at_sample_run=own.get(env.get("script")) == env.get("script_sha256"),
        init_py_equal_at_sample_run=own.get(str(E.PACKAGE / "__init__.py")) == env.get("package_init_sha256"),
        changed_since_runs=sorted(f for f, v in (at_runs or {}).get("files", {}).items() if own.get(f) != v["sha256"]),
        note=("build_cache.py gained the check mode and these report fields after the parity and build runs; the runs used "
              "the sha256 recorded in CODE_AT_RUNS.json (build/parity functions unchanged)"))
    checks = {}
    for ds in (E.DATASET_2025, E.DATASET_2026):
        p = E.REPORTS / f"POSTBUILD_CHECK_{ds}.json"
        if p.exists():
            c = E.read_json(p)
            checks[ds] = dict(report=str(p), sha256=E.sha256_file(p), passed=c["passed"], census=c["census"],
                              rehash={k: v for k, v in c["rehash"].items() if k != "mismatches"},
                              determinism={k: v for k, v in c["determinism"].items() if k != "matches"},
                              seconds=c["seconds"])
    parity = E.read_json(E.PARITY_REPORT) if E.PARITY_REPORT.exists() else None
    builds = {}
    for ds in (E.DATASET_2025, E.DATASET_2026):
        p = E.REPORTS / f"BUILD_MANIFEST_{ds}.json"
        if p.exists():
            b = E.read_json(p)
            builds[ds] = {k: b[k] for k in ("cache_dir", "patch_index", "patch_index_sha256", "summary", "final_folder_check",
                                            "cache_files", "cache_bytes", "disk_free_before", "disk_free_after",
                                            "seconds_this_run", "build_seconds_sum", "workers", "chunk", "created_utc")}
            builds[ds]["build_manifest"] = str(p)
            builds[ds]["build_manifest_sha256"] = E.sha256_file(p)
    sel16 = sorted(m["match_id"] for m in manifest["matches"] if m["dataset"] == E.DATASET_2026)
    kr16 = dict(selected=len(sel16), selected_ids_sha256=E.sha256_bytes("\n".join(sel16).encode("utf-8")),
                selected_list="SAMPLE_MANIFEST.json matches with dataset == 'kr_16_13'")
    if parity and parity.get("passed"):
        kr16.update(status="reused_existing_old_lineage_cache (parity passed; not rebuilt)", cache_dir=str(E.OLD_CACHE_16_13),
                    patch_index=str(E.PATCH_INDEX[E.DATASET_2026]),
                    patch_index_sha256=E.sha256_file(E.PATCH_INDEX[E.DATASET_2026]) if E.PATCH_INDEX[E.DATASET_2026].exists() else None)
    elif E.DATASET_2026 in builds:
        kr16.update(status="rebuilt (parity failed)", **{k: builds[E.DATASET_2026][k] for k in ("cache_dir", "patch_index")})
    else:
        kr16.update(status="pending")
    disk = {}
    for name, p in (("cache_kr_15_18_22", E.CACHE_DIR[E.DATASET_2025]), ("cache_kr_16_13", E.CACHE_DIR[E.DATASET_2026]),
                    ("parity_16_13", E.PARITY_DIR.parent), ("work", E.WORK), ("runtime_repo", E.RUNTIME_REPO),
                    ("reports", E.REPORTS)):
        f, b = E.dir_bytes(p)
        disk[name] = dict(path=str(p), files=f, bytes=b, gib=round(b / 2**30, 3))
    for ds, p in E.PATCH_INDEX.items():
        if p.exists():
            disk[f"patch_index_{ds}"] = dict(path=str(p), files=1, bytes=p.stat().st_size)
    data_total = sum(v["bytes"] for k, v in disk.items() if k != "reports")
    disk["data_root_total_bytes"] = data_total
    disk["data_root_total_gib"] = round(data_total / 2**30, 3)
    disk["d_free_gib_now"] = round(shutil.disk_usage("D:\\").free / 2**30, 1)
    tim = dict(sample_scan_and_hash_s=manifest.get("timing", {}).get("total_seconds"),
               parity_total_s=parity.get("seconds_total") if parity else None,
               parity_build_s=(parity or {}).get("parity_build", {}).get("seconds"),
               build_kr_15_18_22_s=builds.get(E.DATASET_2025, {}).get("seconds_this_run"),
               build_kr_15_18_22_worker_seconds_sum=builds.get(E.DATASET_2025, {}).get("build_seconds_sum"),
               postbuild_check_kr_15_18_22_s=checks.get(E.DATASET_2025, {}).get("seconds"))
    tim["wall_total_s"] = round(sum(v for k, v in tim.items() if k in ("sample_scan_and_hash_s", "parity_total_s", "build_kr_15_18_22_s",
                                                                    "postbuild_check_kr_15_18_22_s") and v), 1)
    per_patch = {}
    b25 = builds.get(E.DATASET_2025, {}).get("summary", {}).get("per_patch", {})
    for ds, info in manifest["datasets"].items():
        for patch, v in info["per_patch"].items():
            pp = dict(candidates=v["candidates"], selected=v["selected"])
            if ds == E.DATASET_2025:
                st = b25.get(patch, {})
                pp.update(built=st.get("built", 0), skipped_existing_verified=st.get("skipped_existing_verified", 0),
                          failed=st.get("failed", 0))
            per_patch[f"{ds}|{patch}"] = pp
    doc = dict(
        version=E.VERSION, kind="E1_REPORT", created_utc=E.utc_now(),
        rule=manifest["rule"],
        sample=dict(manifest=str(E.SAMPLE_MANIFEST), sha256=man_sha, frozen_utc=manifest["frozen_utc"],
                    total_selected=manifest["total_selected"], counts=manifest["counts"],
                    scan_reasons={ds: v["scan_reasons"] for ds, v in manifest["datasets"].items()},
                    candidates_other_patches={ds: v["candidates_other_patches"] for ds, v in manifest["datasets"].items()}),
        per_patch=per_patch,
        code=dict(package=own, provenance=provenance, old_repo_builder_modules=old_mods,
                  old_repo_modules_all_equal_09_15_parent_run=all(v["equal_to_09_15_parent_run"] is not False for v in old_mods.values()),
                  old_repo_modules_not_in_parent_record=[f for f, v in old_mods.items() if v["equal_to_09_15_parent_run"] is None],
                  earlier_driver=str(E.DRIVER_2026), earlier_driver_sha256=E.sha256_file(E.DRIVER_2026)),
        builder=dict(version=BUILDER_VERSION, env=BUILDER_ENV, call=(
            "per match: json.loads(raw bytes) -> team_map{participantId: teamId} -> "
            "data.cache_io._parse_timeline_to_minute_cache_compat(timeline, team_map, detail=detail) -> "
            "gameplay.fights.build_anchors_from_events(events); npz savez_compressed(minute_ts, node_minute, global_minute, "
            "gold_team_minute, xy_raw_minute); events.json json.dumps(events, ensure_ascii=False); meta.json {match_id, "
            "patch_full, patch=normalize_patch, team_map, role_slots=get_role_slots_from_detail, anchors, anchor_is_norm=False, "
            "static_meta=_extract_static_meta_from_detail, interp_policy=_interp_policy_snapshot, feature_version}"),
            cfg=snap["cfg"], interp_policy=snap["interp_policy"], F_NODE=snap["F_NODE"], F_GLOBAL=snap["F_GLOBAL"],
            python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform()),
        parity=(dict(passed=parity["passed"], decision=parity["decision"], criterion=parity["criterion"],
                     parity_dir=parity["parity_dir"], parity_build=parity["parity_build"],
                     primary={k: parity["comparisons"]["primary"][k] for k in parity["comparisons"]["primary"] if k != "not_equal"},
                     alternate={k: parity["comparisons"]["alternate"][k] for k in parity["comparisons"]["alternate"] if k != "not_equal"},
                     reference_audit=parity["reference_audit"], report=str(E.PARITY_REPORT),
                     report_sha256=E.sha256_file(E.PARITY_REPORT)) if parity else None),
        builds=builds, kr_16_13=kr16,
        postbuild_checks=checks,
        failures={ds: b["summary"]["failures"] for ds, b in builds.items()},
        sample_exclusions=dict(
            note="candidate-filter exclusions at the sample stage (not build failures); counts from SAMPLE_MANIFEST.json",
            scan_reasons={ds: v["scan_reasons"] for ds, v in manifest["datasets"].items()},
            candidates_outside_set={ds: v["candidates_other_patches"] for ds, v in manifest["datasets"].items()},
            inventory_empty_detail_bucket=_inventory_unknown_bucket()),
        timing_seconds=tim, disk=disk,
        identifiers=("no player identifiers are written by this package; per built match the written events/meta bytes are "
                     "checked for that match's puuid/summonerId strings and identifier-like keys (counts in builds.*.summary)"),
    )
    E.write_json(E.E1_REPORT, doc)
    log(f"wrote {E.E1_REPORT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parity", "build", "check", "report"), required=True)
    ap.add_argument("--determinism-n", type=int, default=5, help="check mode: rebuilt matches per patch")
    ap.add_argument("--dataset", choices=(E.DATASET_2025, E.DATASET_2026), default=E.DATASET_2025)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.workers <= E.MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{E.MAX_WORKERS}")
    E.set_below_normal_priority()
    name = {"parity": "parity_16_13", "build": f"build_{args.dataset}", "check": f"check_{args.dataset}",
            "report": "report"}[args.mode]
    log = E.Log(E.LOGS / f"{name}.log")
    log(f"start mode={args.mode} dataset={args.dataset} workers={args.workers} chunk={args.chunk} pid={os.getpid()}")
    try:
        return dict(parity=mode_parity, build=mode_build, check=mode_check, report=mode_report)[args.mode](args, log)
    except SystemExit as exc:
        log(f"STOPPED: {exc}")
        raise
    except Exception:
        log("FAILED:\n" + traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())
