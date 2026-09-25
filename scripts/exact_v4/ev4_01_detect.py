"""Stage 2 / R3: engagement detection per patch (v4-exact plan, stage 2 'ev4_01_detect.py').

What it does, per requested patch
  1. Split protection BEFORE any data is read.  Selection patches {15.14, 15.15} always pass.  Any other
     patch (15.16 test, 16.x external, KR replication 15.18-15.22) is refused unless --record1 names an
     existing full record-1 file AND --record1-sha256 gives its SHA-256 (gameplay.split_guard
     .assert_record_exists).  The boundaries-only record 1A (records/record1a_boundaries_*.json, whose
     'record' field starts with '1A') does not unlock held-out patches: record 1 is complete only after
     R5 (V selection, yields).
  2. Preset: apply_preset(<copy of core.config.cfg>, 'v4-exact') (the global cfg object is not mutated,
     as in tests/test_exact_params_mode.py), then its locked values (G = 14,000 ms, D = 4,300,
     ENG_BOUNDARIES_LOCKED) are compared with records/record1a_boundaries_20260925T114606Z.json 'locked',
     and the SHA-256 of core/presets.py with the record's 'hashes' entry (mismatch = stop, unless
     --allow-presets-drift, which is written to the diag file).  v4 parameters come only from
     ExactParams.from_cfg(c, 'v4', require_locked=True).
  3. Match list: outputs/diag_survival_dbscan_20260925/p1/matches_full.json (list of {match_id, patch,
     duration_ms, n_frames, ...}).  The list is filtered to the requested patch in the same statement that
     reads it; nothing about other patches is counted, kept or printed.  Order = sorted by match_id.
     --limit N keeps the first N; --sample N --seed S keeps random.Random(S).sample(ids, N) re-sorted.
  3b. Remakes (author pre-decision, records/stage2_predecisions_20260925T151437Z.json 'remakes'; threshold in
     ev4_common.REMAKE_MAX_GAME_END_MS = 300,000): AFTER --limit / --sample, every selected match whose list
     duration_ms (= GAME_END) is < 300,000 ms is excluded before detection; the worker excludes again when the
     events' GAME_END is < 300,000 (missing / wrong duration_ms).  Excluded matches get status 'remake_excluded'
     in matches_<patch>.parquet, contribute no engagement or r2 row, and are listed (count + ids, by stage) in
     diag_<patch>.json 'remakes'.  Per-match rates use the 'ok' matches only.
  4. Per match (multiprocessing 'spawn' pool, <= 4 workers; results kept in input order, rows by tau):
     inside gameplay.grid_guard.forbid_grid() (the 5-s position grid raises), load_exact_pack(mid,
     allowed_patches=[patch]) (meta, events, minute_ts only; no xy_raw_minute, no node features), then
       * detect_engagements_exact(pack, team_map, v4_params, mode='v4')      -> engagement rows
       * detect_engagements_exact(pack, team_map, r2_params, mode='r2_repro') -> E2 support rows
     r2_params = ExactParams.from_cfg(<copy of cfg with preset 'v3.3' and the R2 switches
     ENG_ALIVE_SOURCE='event', ENG_PARTICIPATION='kill_credit', ENG_OVERLAP_RULE='legacy_priority',
     ENG_MERGE_A6=False, ENG_ISOLATION=False>, 'r2_repro'): G = 13,700 ms, D = 4,264, i.e. the R2 prototype
     (identical to the ExactParams(13700, 4264.0) that reproduces finals_r2_kill_full.tsv row for row in
     tests/test_exact_population.py; clean_max_age_ms is not used by r2_repro).
     A match whose cache files are missing or whose meta patch differs from the list is counted by status
     and skipped; any exception in detection stops the run (no silent drops).

Outputs (default <OUT>/stage2/detect/, or <OUT>/stage2/detect_smoke/ when --limit / --sample is given)
  engagements_<patch>.parquet   one row per v4 engagement (csv.gz with the same columns if pyarrow is
                                missing; list columns are then comma-joined strings)
      patch, match_id, eng_idx (0.. per match, tau order), tau, first_kill_ts, last_kill_ts,
      dur_ms (= last_kill_ts - tau), kill_span_ms (= last_kill_ts - first_kill_ts), game_minute (tau / 60000,
      float), game_min (floor), n_kills, kill_idx (indices into fights._extract_kill_events(events), i.e.
      CHAMPION_KILL events stably sorted by timestamp), kill_ts, fks, killer, victim (first kill),
      n_blue, n_red, n_min, n_max, cohort (T / S / ASYM / P), t5, blue_parts, red_parts,
      alive_blue, alive_red (event-alive at tau), frame_age_ms (nullable), clean, n_foreign_kills,
      n_other_tau, isolated, n_segments, horizon_end_ts, centroid_x, centroid_y,
      match_duration_ms, match_n_frames (from matches_full.json), match_last_frame_ts
  r2repro_<patch>.parquet       one row per r2_repro engagement: patch, match_id, eng_idx, engage_ts,
      first_kill_ts, last_kill_ts, killer, victim, n_blue, n_red, n_min, n_max, cohort, t5, n_segments,
      prox, context, horizon_end_ts, n_kills, kill_idx
  matches_<patch>.parquet       one row per selected match: match_id, status, n_v4, n_r2, sec
  diag_<patch>.json             counts by cohort x clean x isolated (nested and flat), cohort totals,
      T5, r2_repro counts by cohort, summed detector diagnostics of both modes, status counts, parameters,
      runtime (wall, per-match mean / median / p95), code SHA (git HEAD, dirty flag, SHA-256 of this
      script and of every module on the detection path), preset SHA (canonical JSON as in
      ev4_00_manifest.preset_hash, core/presets.py file SHA, record 1A SHA), output file SHA-256s.

Pipeline decisions fixed before any stage-2 result (recorded here and in the later stage-2 scripts):
  * V training rows = one uniformly random ms per 2-minute bucket per match (not per minute, for memory);
    plan section 4 said 'per minute'.
  * Post-engagement states (V(e90) etc.) are NOT stored at extraction; they are computed at the label
    stage (ev4_04_labels.py) after V is frozen.
  * This script uses the event-based detector only; no frame alive, no position grid, no xy_raw_minute.

CLI
  python scripts/exact_v4/ev4_01_detect.py --patch 15.14 [--patch 15.15] [--limit N | --sample N --seed S]
         [--workers 4] [--out-dir DIR] [--cache-dir DIR] [--matches-json FILE]
         [--record1 FILE --record1-sha256 HEX] [--allow-presets-drift]
  Environment: PYTHONIOENCODING=utf-8, PYTHONDONTWRITEBYTECODE=1 (set by the caller).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402

from core.config import cfg  # noqa: E402
from core.presets import PRESETS, apply_preset  # noqa: E402
from gameplay.cohorts_exact import COHORTS  # noqa: E402
from gameplay.exact_population import (DEFAULT_CACHE_DIR, R2_DIAG_KEYS, ExactParams,  # noqa: E402
                                       detect_engagements_exact, load_exact_pack)
from gameplay.grid_guard import forbid_grid, grid_forbidden  # noqa: E402
from gameplay.split_guard import (SELECTION_PATCHES, SplitViolation, assert_record_exists,  # noqa: E402
                                  normalize_patch)

PROJECT = ROOT.parents[1]                                     # C:/Users/todtj/문서/LOL_Teamfight
OUT_ROOT = PROJECT / "outputs" / "reest_exact_v4_20260925"
RECORD1A = OUT_ROOT / "records" / "record1a_boundaries_20260925T114606Z.json"
MATCHES_JSON = PROJECT / "outputs" / "diag_survival_dbscan_20260925" / "p1" / "matches_full.json"
PRESET = "v4-exact"
MAX_WORKERS = 4
R2_G_MS, R2_D = 13_700, 4264.0
R2_FLAGS = {"ENG_ALIVE_SOURCE": "event", "ENG_PARTICIPATION": "kill_credit",
            "ENG_OVERLAP_RULE": "legacy_priority", "ENG_MERGE_A6": False, "ENG_ISOLATION": False}
CODE_FILES = ("scripts/exact_v4/ev4_01_detect.py", "gameplay/exact_population.py", "gameplay/cohorts_exact.py",
              "gameplay/event_survival.py", "gameplay/respawn_rules.py", "gameplay/fights.py",
              "gameplay/fight_clustering.py", "gameplay/fight_postmerge.py", "gameplay/grid_guard.py",
              "gameplay/split_guard.py", "core/presets.py", "core/config.py", "scripts/exact_v4/ev4_common.py")

V4_COLUMNS = ("patch", "match_id", "eng_idx", "tau", "first_kill_ts", "last_kill_ts", "dur_ms", "kill_span_ms",
              "game_minute", "game_min", "n_kills", "kill_idx", "kill_ts", "fks", "killer", "victim",
              "n_blue", "n_red", "n_min", "n_max", "cohort", "t5", "blue_parts", "red_parts",
              "alive_blue", "alive_red", "frame_age_ms", "clean", "n_foreign_kills", "n_other_tau", "isolated",
              "n_segments", "horizon_end_ts", "centroid_x", "centroid_y",
              "match_duration_ms", "match_n_frames", "match_last_frame_ts")
R2_COLUMNS = ("patch", "match_id", "eng_idx", "engage_ts", "first_kill_ts", "last_kill_ts", "killer", "victim",
              "n_blue", "n_red", "n_min", "n_max", "cohort", "t5", "n_segments", "prox", "context",
              "horizon_end_ts", "n_kills", "kill_idx")
LIST_COLUMNS = ("kill_idx", "kill_ts", "fks", "blue_parts", "red_parts")


# ------------------------------------------------------------------ hashing
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preset_sha256(name: str = PRESET) -> str:
    """Same canonical form as scripts/exact_v4/ev4_00_manifest.preset_hash."""
    canon = json.dumps(PRESETS[name], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def git_info() -> Dict[str, Any]:
    try:
        def run(*a):
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, check=True).stdout
        status = run("status", "--porcelain", "--untracked-files=no")
        return {"head": run("rev-parse", "HEAD").decode().strip(), "dirty_tracked": bool(status.strip()),
                "diff_head_sha256": hashlib.sha256(run("diff", "HEAD", "--binary")).hexdigest()}
    except Exception as e:  # git missing
        return {"error": str(e)}


def code_info() -> Dict[str, Any]:
    files = {f: (sha256_file(ROOT / f) if (ROOT / f).is_file() else None) for f in CODE_FILES}
    return {"git": git_info(), "files_sha256": files}


# ------------------------------------------------------------------ guards
def check_patch_access(patch: str, record1: Optional[str], record1_sha256: Optional[str]) -> Dict[str, Any]:
    """Split protection for one patch.  Selection patches pass; any other patch needs the full record 1
    (existing file + matching SHA-256 given on the command line; the boundaries-only record 1A is refused).
    Raises SplitViolation.  Reads no engagement or match data."""
    p = normalize_patch(patch)
    if p in SELECTION_PATCHES:
        return {"patch": p, "held_out": False}
    if not record1 or not record1_sha256:
        raise SplitViolation(f"patch {p} is held out: --record1 FILE and --record1-sha256 HEX are required")
    digest = assert_record_exists(record1, record1_sha256)
    rp = Path(record1)
    if rp.suffix.lower() == ".json":
        blob = json.loads(rp.read_text(encoding="utf-8"))
        tag = str(blob.get("record", "")).strip().upper()
        if tag.startswith("1A") or rp.name.lower().startswith("record1a"):
            raise SplitViolation(f"{rp.name} is the boundaries-only record 1A; held-out patch {p} needs the full record 1")
    return {"patch": p, "held_out": True, "record1": str(rp), "record1_sha256": digest}


def build_v4_cfg(record1a: Path = RECORD1A, allow_presets_drift: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Copy of cfg with apply_preset(.., 'v4-exact'), checked against record 1A.  Returns (cfg, info)."""
    c = copy.copy(cfg)
    apply_preset(c, PRESET)
    rec = json.loads(Path(record1a).read_text(encoding="utf-8"))
    bad = {k: (getattr(c, k, None), v) for k, v in (rec.get("locked") or {}).items() if getattr(c, k, None) != v}
    if bad or not rec.get("locked"):
        raise RuntimeError(f"preset {PRESET!r} disagrees with record 1A locked values: {bad or 'no locked block'}")
    presets_sha = sha256_file(ROOT / "core" / "presets.py")
    want = next((v for k, v in (rec.get("hashes") or {}).items() if k.replace("\\", "/").endswith("core/presets.py")), None)
    drift = want is not None and want != presets_sha
    if drift and not allow_presets_drift:
        raise RuntimeError(f"core/presets.py sha256 {presets_sha} != record 1A {want}; pass --allow-presets-drift "
                           "only after checking that the v4-exact preset values are unchanged")
    info = {"preset": PRESET, "preset_values_sha256": preset_sha256(PRESET), "presets_py_sha256": presets_sha,
            "record1a": str(record1a), "record1a_sha256": sha256_file(Path(record1a)),
            "record1a_presets_py_sha256": want, "presets_py_drift": bool(drift),
            "allow_presets_drift": bool(allow_presets_drift), "locked_checked": rec.get("locked")}
    return c, info


def v4_params(c: Any) -> ExactParams:
    return ExactParams.from_cfg(c, "v4", require_locked=True)


def r2_params() -> ExactParams:
    """R2 prototype parameters (G = 13,700 ms, D = 4,264) through from_cfg(mode='r2_repro')."""
    c = copy.copy(cfg)
    apply_preset(c, "v3.3")
    for k, v in R2_FLAGS.items():
        setattr(c, k, v)
    p = ExactParams.from_cfg(c, "r2_repro")
    if (p.gap_ms, p.diameter) != (R2_G_MS, R2_D):
        raise RuntimeError(f"r2_repro parameters drifted: G={p.gap_ms}, D={p.diameter}")
    ref = ExactParams(R2_G_MS, R2_D)  # the parameters that reproduce finals_r2_kill_full.tsv
    for f in ("pre_kill_ms", "min_alive_per_team", "context_ms", "start_offset_ms", "horizon_ms",
              "max_duration_ms", "merge", "merge_max_gap_ms", "merge_radius", "min_gap_ms", "tail_buffer_ms",
              "postmerge_location_radius"):
        if getattr(p, f) != getattr(ref, f):
            raise RuntimeError(f"r2_repro parameter {f} = {getattr(p, f)!r} differs from the R2 reproduction {getattr(ref, f)!r}")
    return p


# ------------------------------------------------------------------ match list
def _norm_or_none(v: Any) -> Optional[str]:
    nums = re.findall(r"\d+", str(v if v is not None else ""))
    return f"{int(nums[0])}.{int(nums[1])}" if len(nums) >= 2 else None


def load_match_list(patch: str, path: Path = MATCHES_JSON) -> List[Dict[str, Any]]:
    """Rows of matches_full.json for this patch only (filtered in the reading statement), sorted by match_id."""
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in json.load(fh) if isinstance(r, dict) and _norm_or_none(r.get("patch")) == patch]
    rows.sort(key=lambda r: str(r["match_id"]))
    return rows


def select_matches(rows: Sequence[Mapping], limit: Optional[int], sample: Optional[int], seed: int) -> List[Mapping]:
    rows = list(rows)
    if sample is not None:
        idx = sorted(random.Random(seed).sample(range(len(rows)), min(int(sample), len(rows))))
        rows = [rows[i] for i in idx]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def split_remakes(rows: Sequence[Mapping]) -> Tuple[List[Mapping], List[Dict[str, Any]]]:
    """(kept rows, excluded [{match_id, duration_ms}]) by the list duration_ms (= GAME_END) < 300,000 ms rule.
    A row without duration_ms is kept here; the worker checks the events' GAME_END."""
    kept, out = [], []
    for r in rows:
        d = r.get("duration_ms")
        if d is not None and EC.is_remake(int(d)):
            out.append({"match_id": str(r["match_id"]), "duration_ms": int(d)})
        else:
            kept.append(r)
    return kept, out


def game_end_of(events: Iterable[Mapping]) -> Optional[int]:
    """Earliest GAME_END timestamp of the events (None when absent)."""
    ends = [int(e["timestamp"]) for e in events if isinstance(e, Mapping) and e.get("type") == "GAME_END"
            and "timestamp" in e]
    return min(ends) if ends else None


# ------------------------------------------------------------------ per match
def v4_rows(recs: Sequence[Mapping], patch: str, mrow: Mapping, last_frame_ts: Optional[int]) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(recs):
        tau = int(r["tau"])
        out.append({
            "patch": patch, "match_id": str(r["match_id"]), "eng_idx": i, "tau": tau,
            "first_kill_ts": int(r["first_kill_ts"]), "last_kill_ts": int(r["last_kill_ts"]),
            "dur_ms": int(r["last_kill_ts"]) - tau, "kill_span_ms": int(r["last_kill_ts"]) - int(r["first_kill_ts"]),
            "game_minute": tau / 60000.0, "game_min": int(tau // 60000), "n_kills": int(r["n_kills"]),
            "kill_idx": [int(x) for x in r["kill_idx"]], "kill_ts": [int(x) for x in r["kill_ts"]],
            "fks": [int(x) for x in r["fks"]], "killer": int(r["killer"]), "victim": int(r["victim"]),
            "n_blue": int(r["n_blue"]), "n_red": int(r["n_red"]), "n_min": int(r["n_min"]), "n_max": int(r["n_max"]),
            "cohort": str(r["cohort"]), "t5": int(r["t5"]),
            "blue_parts": [int(x) for x in r["blue_parts"]], "red_parts": [int(x) for x in r["red_parts"]],
            "alive_blue": int(r["alive_blue"]), "alive_red": int(r["alive_red"]),
            "frame_age_ms": None if r["frame_age_ms"] is None else int(r["frame_age_ms"]),
            "clean": int(r["clean"]), "n_foreign_kills": int(r["n_foreign_kills"]),
            "n_other_tau": int(r["n_other_tau"]), "isolated": int(r["isolated"]),
            "n_segments": int(r["n_segments"]), "horizon_end_ts": int(r["horizon_end_ts"]),
            "centroid_x": float(r["centroid_x"]), "centroid_y": float(r["centroid_y"]),
            "match_duration_ms": mrow.get("duration_ms"), "match_n_frames": mrow.get("n_frames"),
            "match_last_frame_ts": last_frame_ts,
        })
    return out


def r2_rows(recs: Sequence[Mapping], patch: str) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(recs):
        out.append({
            "patch": patch, "match_id": str(r["match_id"]), "eng_idx": i, "engage_ts": int(r["engage_ts"]),
            "first_kill_ts": int(r["first_kill_ts"]), "last_kill_ts": int(r["last_kill_ts"]),
            "killer": int(r["killer"]), "victim": int(r["victim"]), "n_blue": int(r["n_blue"]),
            "n_red": int(r["n_red"]), "n_min": int(r["n_min"]), "n_max": int(r["n_max"]), "cohort": str(r["cohort"]),
            "t5": int(r["t5"]), "n_segments": int(r["n_segments"]), "prox": int(r["prox"]),
            "context": str(r["context"]), "horizon_end_ts": int(r["horizon_end_ts"]),
            "n_kills": len(r["kill_idx"]), "kill_idx": [int(x) for x in r["kill_idx"]],
        })
    return out


def detect_match(pack: Mapping, patch: str, mrow: Mapping, p4: ExactParams, pr2: ExactParams):
    """(v4 rows, r2 rows, v4 diag, r2 diag) of one loaded pack.  Must run inside forbid_grid()."""
    if not grid_forbidden():
        raise RuntimeError("detect_match must run inside gameplay.grid_guard.forbid_grid()")
    tm = {int(k): int(v) for k, v in (pack["meta"].get("team_map") or {}).items()}
    recs4, d4 = detect_engagements_exact(pack, tm, p4, mode="v4")
    recs2, d2 = detect_engagements_exact(pack, tm, pr2, mode="r2_repro")
    ts = pack.get("minute_ts")
    last = int(ts[-1]) if ts is not None and len(ts) else None
    return v4_rows(recs4, patch, mrow, last), r2_rows(recs2, patch), d4, d2


def _job(args) -> Dict[str, Any]:
    mrow, patch, cache_dir, p4d, pr2d = args
    mid = str(mrow["match_id"])
    t0 = time.perf_counter()
    with forbid_grid():
        pack = load_exact_pack(mid, Path(cache_dir), allowed_patches=[patch])
        if pack is None:
            cd = Path(cache_dir)
            have = all((cd / f"{mid}{s}").exists() for s in (".meta.json", ".events.json", ".npz"))
            return {"match_id": mid, "status": "patch_mismatch" if have else "missing_files",
                    "v4": [], "r2": [], "d4": {}, "d2": {}, "sec": time.perf_counter() - t0}
        if str(pack["meta"].get("match_id", mid)) != mid:
            raise RuntimeError(f"{mid}: cache meta match_id {pack['meta'].get('match_id')!r}")
        pack["meta"]["match_id"] = mid
        ge = game_end_of(pack.get("events") or [])
        if EC.is_remake(ge):
            return {"match_id": mid, "status": EC.REMAKE_STATUS, "v4": [], "r2": [], "d4": {}, "d2": {},
                    "sec": time.perf_counter() - t0, "game_end": ge}
        try:
            v4, r2, d4, d2 = detect_match(pack, patch, mrow, ExactParams(**p4d), ExactParams(**pr2d))
        except Exception as e:
            raise RuntimeError(f"detection failed on {mid}: {type(e).__name__}: {e}") from e
    return {"match_id": mid, "status": "ok", "v4": v4, "r2": r2, "d4": d4, "d2": d2,
            "sec": time.perf_counter() - t0, "game_end": ge}


def run_jobs(jobs: List[tuple], workers: int) -> Iterator[Dict[str, Any]]:
    """Per-match results in input order (streamed; imap keeps order)."""
    if workers <= 1:
        for j in jobs:
            yield _job(j)
        return
    with get_context("spawn").Pool(workers) as pool:
        yield from pool.imap(_job, jobs, chunksize=max(1, min(16, len(jobs) // (workers * 8) or 1)))


# ------------------------------------------------------------------ tallies
def v4_key(r: Mapping) -> Tuple[str, int, int, int]:
    return (str(r["cohort"]), int(r["clean"]), int(r["isolated"]), int(r["t5"]))


def tally_v4_counts(keys: Mapping[Tuple[str, int, int, int], int]) -> Dict[str, Any]:
    """Tally from a Counter of v4_key(row) -> n."""
    cube = {c: {f"clean{cl}": {f"isolated{iso}": 0 for iso in (0, 1)} for cl in (0, 1)} for c in COHORTS}
    by, ci = Counter(), Counter()
    t5 = t5ci = n_clean = n_iso = n = 0
    for (c, cl, iso, t), k in keys.items():
        cube[c][f"clean{cl}"][f"isolated{iso}"] += k
        by[c] += k
        n += k
        n_clean += k * cl
        n_iso += k * iso
        t5 += k * t
        if cl and iso:
            ci[c] += k
            t5ci += k * t
    flat = [{"cohort": c, "clean": cl, "isolated": iso, "n": cube[c][f"clean{cl}"][f"isolated{iso}"]}
            for c in COHORTS for cl in (0, 1) for iso in (0, 1)]
    return {"n": n, "by_cohort": {c: by.get(c, 0) for c in COHORTS},
            "clean_isolated_by_cohort": {c: ci.get(c, 0) for c in COHORTS},
            "t5": t5, "t5_clean_isolated": t5ci, "clean": n_clean, "isolated": n_iso,
            "cohort_x_clean_x_isolated": cube, "cohort_x_clean_x_isolated_flat": flat}


def tally_v4(rows: Iterable[Mapping]) -> Dict[str, Any]:
    return tally_v4_counts(Counter(v4_key(r) for r in rows))


def tally_r2(rows: Iterable[Mapping]) -> Dict[str, Any]:
    rows = list(rows)
    by = Counter(r["cohort"] for r in rows)
    return {"n": len(rows), "by_cohort": {c: by.get(c, 0) for c in COHORTS}, "t5": sum(int(r["t5"]) for r in rows)}


def _q(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return float(s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))])


# ------------------------------------------------------------------ writing
def to_frame(rows: List[Mapping], columns: Sequence[str]):
    import pandas as pd
    df = pd.DataFrame(list(rows), columns=list(columns))
    for c in ("frame_age_ms", "match_duration_ms", "match_n_frames", "match_last_frame_ts"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return df


def write_frame(df, stem: Path) -> Path:
    try:
        import pyarrow  # noqa: F401
        out = stem.parent / (stem.name + ".parquet")
        df.to_parquet(out, index=False)
    except ImportError:
        df = df.copy()
        for c in LIST_COLUMNS:
            if c in df.columns:
                df[c] = df[c].map(lambda v: ",".join(map(str, v)))
        out = stem.parent / (stem.name + ".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
    return out


def write_table(rows: List[Mapping], columns: Sequence[str], stem: Path) -> Path:
    return write_frame(to_frame(rows, columns), stem)


class _Buffer:
    """Row buffer flushed to DataFrame chunks (bounded memory for the full ~75k-match runs)."""

    def __init__(self, columns: Sequence[str], flush_rows: int = 100_000):
        self.columns, self.flush_rows, self.rows, self.frames, self.n = columns, flush_rows, [], [], 0

    def extend(self, rows: List[Mapping]) -> None:
        self.rows.extend(rows)
        self.n += len(rows)
        if len(self.rows) >= self.flush_rows:
            self.frames.append(to_frame(self.rows, self.columns))
            self.rows = []

    def frame(self):
        import pandas as pd
        if self.rows or not self.frames:
            self.frames.append(to_frame(self.rows, self.columns))
            self.rows = []
        return pd.concat(self.frames, ignore_index=True) if len(self.frames) > 1 else self.frames[0]


def run_patch(patch: str, args, v4c_info: Dict[str, Any], p4: ExactParams, pr2: ExactParams,
              access: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    t_wall = time.perf_counter()
    rows = load_match_list(patch, Path(args.matches_json))
    n_list = len(rows)
    sel = select_matches(rows, args.limit, args.sample, args.seed)
    del rows
    ids = [str(r["match_id"]) for r in sel]
    run_sel, remake_list = split_remakes(sel)          # remake rule, match-list stage (after --limit / --sample)
    run_ids = [str(r["match_id"]) for r in run_sel]
    jobs = [(dict(r), patch, str(args.cache_dir), asdict(p4), asdict(pr2)) for r in run_sel]
    t_det = time.perf_counter()
    b4, b2 = _Buffer(V4_COLUMNS), _Buffer(R2_COLUMNS)
    k4, k2 = Counter(), Counter()
    status, d4, d2 = Counter(), Counter(), Counter()
    mrows, secs, got_ids = [], [], []
    remake_events, dur_mismatch = [], []
    dur_by_id = {str(r["match_id"]): r.get("duration_ms") for r in run_sel}
    for x in remake_list:
        status[EC.REMAKE_STATUS] += 1
        mrows.append({"match_id": x["match_id"], "status": EC.REMAKE_STATUS, "n_v4": 0, "n_r2": 0, "sec": 0.0})
    for r in run_jobs(jobs, int(args.workers)):
        got_ids.append(r["match_id"])
        ge, dur = r.get("game_end"), dur_by_id.get(r["match_id"])
        if ge is not None and dur is not None and int(dur) != int(ge):
            dur_mismatch.append({"match_id": r["match_id"], "duration_ms": int(dur), "game_end": int(ge)})
        if r["status"] == EC.REMAKE_STATUS:
            remake_events.append({"match_id": r["match_id"], "game_end": ge, "duration_ms": dur})
        status[r["status"]] += 1
        d4.update({k: int(v) for k, v in r["d4"].items()})
        d2.update({k: int(v) for k, v in r["d2"].items()})
        k4.update(v4_key(x) for x in r["v4"])
        k2.update((str(x["cohort"]), int(x["t5"])) for x in r["r2"])
        b4.extend(r["v4"])
        b2.extend(r["r2"])
        mrows.append({"match_id": r["match_id"], "status": r["status"], "n_v4": len(r["v4"]),
                      "n_r2": len(r["r2"]), "sec": float(r["sec"])})
        if r["status"] == "ok":
            secs.append(float(r["sec"]))
    det_sec = time.perf_counter() - t_det
    if got_ids != run_ids:
        raise RuntimeError("result order differs from input order")
    order = {m: i for i, m in enumerate(ids)}
    mrows.sort(key=lambda m: order[m["match_id"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    f_v4 = write_frame(b4.frame(), out_dir / f"engagements_{patch}")
    f_r2 = write_frame(b2.frame(), out_dir / f"r2repro_{patch}")
    f_m = write_table(mrows, ("match_id", "status", "n_v4", "n_r2", "sec"), out_dir / f"matches_{patch}")
    r2_by = Counter()
    for (c, _t), k in k2.items():
        r2_by[c] += k
    r2_tally = {"n": sum(k2.values()), "by_cohort": {c: r2_by.get(c, 0) for c in COHORTS},
                "t5": sum(k for (_c, t), k in k2.items() if t)}
    n_ok = status.get("ok", 0)
    diag = {
        "script": "scripts/exact_v4/ev4_01_detect.py", "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "argv": sys.argv[1:], "patch": patch, "access": access,
        "selection": {"matches_json": str(args.matches_json), "matches_json_sha256": sha256_file(Path(args.matches_json)),
                      "n_list_patch": n_list, "limit": args.limit, "sample": args.sample, "seed": args.seed,
                      "n_selected": len(ids), "match_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                      "order": "sorted by match_id"},
        "cache_dir": str(args.cache_dir), "status": dict(status),
        "remakes": {**EC.remake_rule(), "n_selected_before_exclusion": len(ids),
                    "n_excluded": len(remake_list) + len(remake_events),
                    "n_excluded_list_stage": len(remake_list), "n_excluded_event_stage": len(remake_events),
                    "excluded_list_stage": remake_list, "excluded_event_stage": remake_events,
                    "n_duration_vs_game_end_mismatch": len(dur_mismatch),
                    "duration_vs_game_end_mismatch": dur_mismatch[:50]},
        "v4": {**tally_v4_counts(k4), "per_match": (b4.n / n_ok) if n_ok else None,
               "detector_diag_sums": dict(d4)},
        "r2_repro": {**r2_tally, "per_match": (b2.n / n_ok) if n_ok else None,
                     "detector_diag_sums": {k: int(d2.get(k, 0)) for k in (*R2_DIAG_KEYS, "finals", "kills",
                                                                             "clusters_total", "clusters_after_spatial")}},
        "params": {"v4": asdict(p4), "r2_repro": asdict(pr2)},
        "runtime": {"wall_sec": time.perf_counter() - t_wall, "detect_sec": det_sec, "workers": int(args.workers),
                    "per_match_sec_mean": (sum(secs) / len(secs)) if secs else None,
                    "per_match_sec_median": _q(secs, 0.5), "per_match_sec_p95": _q(secs, 0.95),
                    "wall_per_match_sec": (det_sec / len(ids)) if ids else None},
        "code": code_info(), "preset": v4c_info,
        "decisions": ["V training rows: one random ms per 2-minute bucket per match (memory)",
                      "post-engagement states computed at the label stage after V is frozen, not at extraction",
                      f"remakes: matches with GAME_END < {EC.REMAKE_MAX_GAME_END_MS} ms excluded (status "
                      f"{EC.REMAKE_STATUS!r})"],
        "predecisions": {k: v for k, v in EC.predecisions_info().items() if k != "content"},
        "outputs": {p.name: {"path": str(p), "sha256": sha256_file(p)} for p in (f_v4, f_r2, f_m)},
    }
    f_d = out_dir / f"diag_{patch}.json"
    f_d.write_text(json.dumps(diag, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return diag


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="v4-exact engagement detection (stage 2, ev4_01)")
    ap.add_argument("--patch", action="append", required=True, help="patch 'major.minor'; repeatable")
    g = ap.add_argument_group("smoke subset")
    g.add_argument("--limit", type=int, default=None, help="first N matches (sorted by match_id)")
    g.add_argument("--sample", type=int, default=None, help="random N matches (random.Random(seed))")
    g.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--matches-json", type=Path, default=MATCHES_JSON)
    ap.add_argument("--record1", default=None, help="full record-1 file (required for any held-out patch)")
    ap.add_argument("--record1-sha256", default=None, help="SHA-256 of --record1")
    ap.add_argument("--record1a", type=Path, default=RECORD1A, help=argparse.SUPPRESS)
    ap.add_argument("--allow-presets-drift", action="store_true")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS}")
    for k in ("limit", "sample"):
        if getattr(a, k) is not None and getattr(a, k) < 1:
            ap.error(f"--{k} must be >= 1")
    if a.out_dir is None:
        a.out_dir = OUT_ROOT / "stage2" / ("detect_smoke" if (a.limit or a.sample) else "detect")
    return a


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    a = parse_args(argv)
    # 0. author pre-decision record: refuse a missing or edited record before anything else
    EC.predecisions_info()
    # 1. split protection for every requested patch before any data is read
    access = {}
    for raw in a.patch:
        acc = check_patch_access(raw, a.record1, a.record1_sha256)
        access[acc["patch"]] = acc
    # 2. preset and parameters
    c, info = build_v4_cfg(Path(a.record1a), a.allow_presets_drift)
    p4, pr2 = v4_params(c), r2_params()
    out = {}
    for patch, acc in access.items():
        d = run_patch(patch, a, info, p4, pr2, acc, Path(a.out_dir))
        rt = d["runtime"]
        print(f"[ev4_01] {patch}: matches={d['selection']['n_selected']} status={d['status']} "
              f"remakes_excluded={d['remakes']['n_excluded']} "
              f"v4={d['v4']['n']} {d['v4']['by_cohort']} clean&iso={d['v4']['clean_isolated_by_cohort']} "
              f"r2={d['r2_repro']['n']} {d['r2_repro']['by_cohort']} "
              f"wall={rt['wall_sec']:.1f}s per_match={rt['per_match_sec_mean']}", flush=True)
        out[patch] = d
    return out


if __name__ == "__main__":
    main()
