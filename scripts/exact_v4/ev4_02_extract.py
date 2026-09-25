"""v4-exact stage 2, step R4: extract StateV3 engagement pre-states, V training rows and martingale intervals.

Plan: outputs/diag_survival_dbscan_20260925/docs/REESTIMATION_PLAN_V4_EXACT_20260925.md, sections 2-5, "단계 2"
(ev4_02_extract.py), "테스트", "실행 순서" R4.  Preset 'v4-exact' (G = 14,000 ms, D = 4,300), locked in record 1A
(outputs/reest_exact_v4_20260925/records/record1a_boundaries_20260925T114606Z.json).

Implementation decisions made BEFORE any result (2026-09-25, recorded here and in every manifest):
  D1  V training rows = ONE uniform random ms per 2-minute bucket per match (not one per minute, as the plan's
      section 4 says; memory).  Buckets k = 0, 1, ...: [120,000 + 120,000 k, min(120,000 + 120,000 (k + 1),
      GAME_END) - 1]; the last bucket is partial.  The query time is
      bucket_start + (int(sha256("<match_id>:<k>").hexdigest()[:16], 16) mod bucket_width).
  D2  Post-engagement states (V at e90 etc.) are NOT stored here.  They are computed at the label stage
      (ev4_04) after V is frozen.  Only the label end e90 (a timestamp) is kept as engagement metadata.
  D3  Team-flipped copies of the V rows (plan section 4) are not materialised; ev4_03 builds them from these rows.
  D4  The V target (blue team won, GAME_END winningTeam via gameplay.state_value.final_outcome) is read ONLY in
      the V-row code and written ONLY to the V index table; engagement and martingale rows never carry it.
  D5  Martingale intervals (15.15 only): one per match; h = the (int(sha256("<match_id>:mart_h")[:16],16) mod n)-th
      element of the SORTED empirical 15.14 distribution of e90 - tau over isolated engagements of all cohorts
      (h_distribution.npy written by the 15.14 run); t = 120,000 + (int(sha256("<match_id>:mart_t")[:16],16) mod
      (GAME_END - h - 1 - 120,000 + 1)), i.e. uniform on [120 s, GAME_END - h - 1].  No interval when that range is
      empty.  States at t and t + h.  Strata: frame_update = a minute frame in (t, t + h]; deaths_pre60 = kills in
      (t - 60 s, t] (the primary 'recent deaths' flag recent_deaths_ge3 = deaths_pre60 >= 3; deaths_in (t, t+h]
      and deaths_end60 (t + h - 60 s, t + h] are stored too); phase 0 / 1 / 2 = t < 15 min / < 25 min / >= 25 min.
  D6  e90 = labels_exact.label_endpoint(L, events, h_s=90, next_start, game_end) (the DR rule
      scripts/engagement_labels_v3_rules.py, unchanged); next_start = the smallest tau of another engagement of the
      same match with a larger tau (ALL detected engagements, isolated or not), else absent.
  D7  Engagement pre-states: every ISOLATED engagement (all cohorts): StateV3 at tau - 1 (996 columns incl.
      snapshot_age_s; q drops it via state_value_v3.q_columns) and, for clean engagements only (input 'clean'
      flag = frame age at tau < 10 s), setup_features.build_setup(pack, tau - 1, participant slots) (157 columns);
      a NaN block for non-clean rows.  frame_age_pre_ms (at tau - 1) is stored next to the input frame_age_ms.
  D8  Item check: every item id in ITEM_PURCHASED / ITEM_SOLD / ITEM_DESTROYED / ITEM_UNDO events (itemId, beforeId,
      afterId; 0 ignored) of all matches of a chunk is an observation; item_state.check_unknown_rate (> 0.5 %
      halts) runs per chunk.
  D9  Exactness spot check (chunk 0 only): for the first 20 matches of the chunk that have an isolated engagement,
      the middle isolated engagement (index n // 2 by tau): StateV3 (and setup when clean) at tau - 1 from the full
      pack == from the pack truncated to frames and events <= tau - 1 (NaN == NaN).  Any mismatch halts.
  D10 Match universe: --matches, else the ev4_01 match list matches_<patch>.parquet next to --detect (status 'ok'),
      else the cache patch index; ordered by sha256(match_id); --limit keeps the first N (smoke).  A run with
      --limit or < 50,000 matches is marked sample=true; 15.15 refuses a sample h source without --allow-sample-h.
  D11 Remakes (author pre-decision 2026-09-25, outputs/reest_exact_v4_20260925/records/
      stage2_predecisions_20260925T151437Z.json, key 'remakes'; threshold ev4_common.REMAKE_MAX_GAME_END_MS): a match
      whose GAME_END (earliest GAME_END event) is < 300,000 ms is skipped in the worker before anything is built:
      no engagement, V or martingale row, no item observation, status 'remake_excluded' in chunk_*_matches.parquet;
      the ids are listed in the chunk sidecar and in manifest.json 'remakes'.  The ev4_01 match list already marks
      them 'remake_excluded' (not 'ok'), so with --detect this is a second line of defence.  The 15.14 h
      distribution therefore contains no remake engagement.

Guards: core.presets.apply_preset(cfg, 'v4-exact'); ExactParams.from_cfg(cfg, 'v4', require_locked=True) and the
record-1A locked G / D must equal the preset; every worker runs inside grid_guard.forbid_grid(); split_guard: patches
15.14 / 15.15 pass, any other patch needs --record1 PATH --record1-sha256 HEX (assert_record_exists; the
boundaries-only record 1A is refused, as in ev4_01_detect) BEFORE any data is touched; core/presets.py must match
the sha256 listed in record 1A; the pack loader reads only meta.json, events.json and the npz members minute_ts and node_minute (never
xy_raw_minute).

Input: detection table (ev4_01_detect.py output), read BY COLUMN NAME:
  required  match_id, tau, first_kill_ts, last_kill_ts, cohort, clean, isolated, frame_age_ms
  optional  patch (must equal --patch), n_kills, n_blue, n_red, n_min, t5, alive_blue, alive_red (passed through)
  The table must hold ALL detected engagements of each match (next_start uses the non-isolated ones too).
  --inline-detect runs gameplay.exact_population.detect_engagements_exact (mode 'v4') in the worker instead and
  produces the same columns (smoke / before ev4_01 exists).

Outputs: <out>/<patch>/ (default out = outputs/reest_exact_v4_20260925/stage2/extract)
  chunk_NNNNN.npz            eng_X (n_eng, 996) f4, eng_setup (n_eng, 157) f4 (NaN rows = not clean),
                             v_X (n_v, 996) f4, m_X0 / m_X1 (n_m, 996) f4 (15.15 only, else 0 rows),
                             state_columns, setup_columns (unicode)
  chunk_NNNNN_eng.parquet    engagement index (row = row of eng_X / eng_setup) and metadata
  chunk_NNNNN_v.parquet      V index: row, match_id, bucket, bucket_start, bucket_end, t, snapshot_ms,
                             frame_age_ms, game_end, y_blue_win
  chunk_NNNNN_mart.parquet   martingale index (15.15)
  chunk_NNNNN_matches.parquet per-match status / reason / counts
  chunk_NNNNN.json           chunk sidecar: plan hash, row counts, sha256 + bytes per file, item check, spot check
  h_distribution.npy         (15.14) sorted int64 e90 - tau of isolated engagements, the martingale h source
  manifest.json              STATE_V3_NAME_HASH, SETUP_NAME_HASH, row counts, sha256 per file, guards, runtimes

CLI
  python scripts/exact_v4/ev4_02_extract.py --patch 15.14 [--detect PARQUET | --inline-detect] [--limit N]
         [--chunk-size 1000] [--workers 4] [--out DIR]
  python scripts/exact_v4/ev4_02_extract.py --patch 15.15 ... --h-dist <out>/15.14/h_distribution.npy
  held-out patches additionally: --record1 PATH --record1-sha256 HEX
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import sys  # noqa: E402

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from bisect import bisect_right  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
OUT_BASE = Path(r"C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925")
os.environ.setdefault("LOL_OUTPUT_ROOT", str(OUT_BASE / "stage2" / "runtime"))
for _p in (str(WT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ev4_common as EC  # noqa: E402

DEFAULT_OUT = OUT_BASE / "stage2" / "extract"
CACHE = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
PATCH_INDEX = Path("D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json")
RECORD1A = OUT_BASE / "records" / "record1a_boundaries_20260925T114606Z.json"
PRESET = "v4-exact"

V_START_MS = 120_000
V_BUCKET_MS = 120_000
MART_START_MS = 120_000
H_S = 90
PHASE_EDGES_MS = (900_000, 1_500_000)
RECENT_DEATH_WINDOW_MS = 60_000
RECENT_DEATH_MIN = 3
SPOT_CHECK_N = 20
MAX_CHUNK = 1000
FULL_RUN_MIN_MATCHES = 50_000   # a run with --limit or fewer matches is a SAMPLE (15.14 / 15.15 have ~74,700 each)
MAX_WORKERS = 4
ITEM_EVENT_TYPES = ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO")
ITEM_KEYS = ("itemId", "beforeId", "afterId")

DETECT_REQUIRED = ("match_id", "tau", "first_kill_ts", "last_kill_ts", "cohort", "clean", "isolated",
                   "frame_age_ms")
DETECT_OPTIONAL = ("n_kills", "n_blue", "n_red", "n_min", "t5", "alive_blue", "alive_red")
COHORT_CODES = {"T": 0, "S": 1, "ASYM": 2, "P": 3}

DECISIONS = {
    "D1_v_rows": "one uniform ms per 2-minute bucket [120000+120000k, min(.., GAME_END)-1], seed sha256('<mid>:<k>')",
    "D2_post_states": "not stored; computed at the label stage after V is frozen",
    "D3_team_flip": "not materialised; built by ev4_03 from the stored rows",
    "D4_target": "GAME_END winningTeam == 100, read only for V rows, stored only in the V index",
    "D5_martingale": "15.15 only; h from sorted 15.14 e90-tau (isolated, all cohorts), seed sha256('<mid>:mart_h'); "
                     "t uniform [120000, GAME_END-h-1] seed sha256('<mid>:mart_t'); recent_deaths_ge3 = kills in "
                     "(t-60s, t] >= 3",
    "D6_e90": "labels_exact.label_endpoint(L, events, h_s=90, next_start=next larger tau of any engagement)",
    "D7_eng_rows": "isolated engagements, all cohorts; StateV3(tau-1); setup(tau-1) for clean only, NaN otherwise",
    "D8_items": "ITEM_* event ids (itemId/beforeId/afterId) per chunk, check_unknown_rate max 0.5%",
    "D9_spot_check": "chunk 0: middle isolated engagement of the first 20 matches with one; full == truncated pack",
    "D10_sample": "match ids ordered by sha256(match_id), first --limit",
    "D11_remakes": f"GAME_END < {EC.REMAKE_MAX_GAME_END_MS} ms -> skipped in the worker, status "
                   f"{EC.REMAKE_STATUS!r}, no rows; author record {EC.PREDECISIONS_RECORD.name}",
}


# ============================================================================ small helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_int(key: str) -> int:
    """Deterministic 64-bit integer from sha256(key)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ============================================================================ guards
def check_patch_access(patch: str, record1: Optional[str] = None, record1_sha256: Optional[str] = None) -> Dict[str, Any]:
    """split_guard: 15.14 / 15.15 always; any other patch needs record 1 with the given sha256.

    Runs before any data (patch index, cache, detection table) is touched."""
    from gameplay.split_guard import SELECTION_PATCHES, SplitViolation, assert_patch_access, assert_record_exists, \
        normalize_patch
    p = normalize_patch(patch)
    out: Dict[str, Any] = {"patch": p, "selection_patch": p in SELECTION_PATCHES}
    if p not in SELECTION_PATCHES:
        if not record1 or not record1_sha256:
            raise SplitViolation(f"patch {p} is held out: pass --record1 PATH and --record1-sha256 HEX")
        out["record1"] = str(record1)
        out["record1_sha256"] = assert_record_exists(record1, record1_sha256)
        rp = Path(record1)
        tag = ""
        if rp.suffix.lower() == ".json":
            tag = str(json.loads(rp.read_text(encoding="utf-8")).get("record", "")).strip().upper()
        if tag.startswith("1A") or rp.name.lower().startswith("record1a"):
            raise SplitViolation(f"{rp.name} is the boundaries-only record 1A; held-out patch {p} needs the full record 1")
    assert_patch_access([p], record1, record1_sha256)
    return out


def load_params(record1a: Path = RECORD1A) -> Tuple[Any, Any, Dict[str, Any]]:
    """(cfg copy after apply_preset('v4-exact'), ExactParams (require_locked), guard info)."""
    from core.config import cfg
    from core.presets import apply_preset
    from gameplay.exact_population import ExactParams
    c = copy.copy(cfg)
    applied = apply_preset(c, PRESET)
    params = ExactParams.from_cfg(c, "v4", require_locked=True)
    rec = json.loads(Path(record1a).read_text(encoding="utf-8"))
    locked = rec.get("locked") or {}
    bad = {}
    if locked.get("ENG_BOUNDARIES_LOCKED") is not True:
        bad["ENG_BOUNDARIES_LOCKED"] = locked.get("ENG_BOUNDARIES_LOCKED")
    for k, v in (("TF2_KILL_CLUSTER_GAP_MS", params.gap_ms), ("CLUSTER_MAX_DIAMETER", params.diameter)):
        if k not in locked or float(locked[k]) != float(v):
            bad[k] = (locked.get(k), v)
    if bad:
        raise RuntimeError(f"preset {PRESET} disagrees with record 1A {record1a}: {bad}")
    presets_sha = sha256_file(WT / "core" / "presets.py")
    want_sha = next((v for k, v in (rec.get("hashes") or {}).items()
                     if str(k).replace(chr(92), "/").endswith("core/presets.py")), None)
    if want_sha is not None and want_sha != presets_sha:
        raise RuntimeError(f"core/presets.py sha256 {presets_sha} != record 1A {want_sha}")
    info = {"preset": PRESET, "preset_values": {k: applied[k] for k in sorted(applied) if k.startswith(("ENG_", "TF2_", "CLUSTER_"))},
            "params": dict(params.__dict__), "record1a": str(record1a), "record1a_sha256": sha256_file(Path(record1a)),
            "record1a_locked": locked, "presets_py_sha256": presets_sha, "record1a_presets_py_sha256": want_sha}
    return c, params, info


def verify_params(params_dict: Mapping[str, Any]) -> Any:
    """Worker side: recompute the params from the preset and require equality with the main process."""
    _, params, _ = load_params()
    if dict(params.__dict__) != dict(params_dict):
        raise RuntimeError("worker ExactParams differ from the main process")
    return params


# ============================================================================ match universe and loading
def match_universe(patch: str, limit: Optional[int] = None, matches_file: Optional[Path] = None) -> List[str]:
    """Match ids of `patch`, ordered by sha256(match_id); first `limit`.

    Source: --matches (.txt, or .parquet with match_id and optional status / patch: only status 'ok' rows),
    else the patch index (patch-filtered in the reading statement)."""
    if matches_file is not None:
        p = Path(matches_file)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            names = pq.ParquetFile(p).schema.names
            df = pq.read_table(p, columns=[c for c in ("match_id", "status", "patch") if c in names]).to_pandas()
            if "patch" in df.columns and set(df["patch"].astype(str)) - {patch}:
                raise RuntimeError(f"match list {p} has rows of other patches")
            if "status" in df.columns:
                df = df[df["status"].astype(str) == "ok"]
            ids = [str(x) for x in df["match_id"].unique()]
        else:
            ids = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        idx = json.loads(PATCH_INDEX.read_text(encoding="utf-8"))
        ids = [k for k, v in idx.items() if str(v) == patch]
        del idx
    if not ids:
        raise RuntimeError(f"no matches for patch {patch}")
    ids = sorted(set(ids), key=lambda m: (sha256_text(m), m))
    return ids[:limit] if limit else ids


def load_pack(mid: str, patch: str, cache: Path = CACHE) -> Dict[str, Any]:
    """meta, events, minute_ts, node_minute of one match.  Never decompresses xy_raw_minute."""
    meta = json.loads((cache / f"{mid}.meta.json").read_text(encoding="utf-8"))
    if str(meta.get("patch")) != patch:
        raise RuntimeError(f"{mid}: meta patch {meta.get('patch')!r} != {patch}")
    meta = dict(meta)
    meta.setdefault("match_id", mid)
    meta["team_map"] = {int(k): int(v) for k, v in (meta.get("team_map") or {}).items()}
    ev = json.loads((cache / f"{mid}.events.json").read_text(encoding="utf-8"))
    ev = [e for e in ev if isinstance(e, dict)] if isinstance(ev, list) else []
    with np.load(cache / f"{mid}.npz", allow_pickle=False) as z:
        ts = np.asarray(z["minute_ts"]).astype(np.int64)
        node = np.asarray(z["node_minute"])
    return {"minute_ts": ts, "node_minute": node, "events": ev, "meta": meta}


def truncate_pack(pack: Mapping[str, Any], t: int) -> Dict[str, Any]:
    keep = np.asarray(pack["minute_ts"]) <= t
    return {"minute_ts": pack["minute_ts"][keep], "node_minute": pack["node_minute"][keep],
            "events": [e for e in pack["events"] if int(e.get("timestamp", 0)) <= t], "meta": pack["meta"]}


# ============================================================================ query times
def v_query_times(match_id: str, game_end: int) -> List[Tuple[int, int, int, int]]:
    """[(bucket k, bucket_start, bucket_end (inclusive), t)] for t in [120,000, GAME_END - 1] (decision D1)."""
    out = []
    last = int(game_end) - 1
    k = 0
    while True:
        lo = V_START_MS + V_BUCKET_MS * k
        if lo > last:
            break
        hi = min(lo + V_BUCKET_MS - 1, last)
        t = lo + hash_int(f"{match_id}:{k}") % (hi - lo + 1)
        out.append((k, lo, hi, t))
        k += 1
    return out


def draw_h(match_id: str, h_sorted: np.ndarray) -> Tuple[int, int]:
    """(index, h ms) from the sorted empirical distribution (decision D5)."""
    n = len(h_sorted)
    if n == 0:
        raise ValueError("empty h distribution")
    i = hash_int(f"{match_id}:mart_h") % n
    return int(i), int(h_sorted[i])


def mart_interval(match_id: str, game_end: int, h_sorted: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """(t, h, h_index) with t uniform on [120,000, GAME_END - h - 1]; None when that range is empty."""
    i, h = draw_h(match_id, h_sorted)
    hi = int(game_end) - h - 1
    if hi < MART_START_MS or h <= 0:
        return None
    t = MART_START_MS + hash_int(f"{match_id}:mart_t") % (hi - MART_START_MS + 1)
    return int(t), h, i


def phase_of(t: int) -> int:
    return 0 if t < PHASE_EDGES_MS[0] else 1 if t < PHASE_EDGES_MS[1] else 2


def mart_strata(minute_ts: np.ndarray, kill_ts: Sequence[int], t: int, h: int) -> Dict[str, int]:
    """Strata flags of the interval (t, t + h] (decision D5).  kill_ts sorted ascending."""
    ts = np.asarray(minute_ts, dtype=np.int64)
    t1 = t + h
    n_frames = int(((ts > t) & (ts <= t1)).sum())

    def n_in(lo_excl: int, hi_incl: int) -> int:
        return bisect_right(kill_ts, hi_incl) - bisect_right(kill_ts, lo_excl)

    pre = n_in(t - RECENT_DEATH_WINDOW_MS, t)
    return {"frame_update": int(n_frames > 0), "n_frames_in": n_frames,
            "deaths_pre60": pre, "deaths_in": n_in(t, t1), "deaths_end60": n_in(t1 - RECENT_DEATH_WINDOW_MS, t1),
            "recent_deaths_ge3": int(pre >= RECENT_DEATH_MIN), "phase": phase_of(t)}


def next_starts(taus: Sequence[int]) -> List[Optional[int]]:
    """For each tau, the smallest strictly larger tau among all engagements (None if none)."""
    srt = sorted(set(int(t) for t in taus))
    out = []
    for t in taus:
        j = bisect_right(srt, int(t))
        out.append(srt[j] if j < len(srt) else None)
    return out


def frame_age(minute_ts: np.ndarray, t: int) -> Tuple[int, int]:
    """(snapshot ms, t - snapshot) of the last frame <= t."""
    i = int(np.searchsorted(minute_ts, int(t), side="right")) - 1
    if i < 0:
        raise ValueError("query before the first frame")
    return int(minute_ts[i]), int(t) - int(minute_ts[i])


def kill_times(events: Iterable[Mapping]) -> List[int]:
    return sorted(int(e.get("timestamp", 0)) for e in events if e.get("type") == "CHAMPION_KILL")


def game_end_of(events: Sequence[Mapping]) -> Optional[int]:
    """Earliest GAME_END timestamp (no winner read)."""
    ends = [int(e["timestamp"]) for e in events if e.get("type") == "GAME_END" and "timestamp" in e]
    return min(ends) if ends else None


def item_observations(events: Iterable[Mapping], table: Mapping[int, Any]) -> Tuple[int, Counter]:
    """(number of item-id observations, Counter of ids absent from the patch table) (decision D8)."""
    n, unk = 0, Counter()
    for e in events:
        if e.get("type") not in ITEM_EVENT_TYPES:
            continue
        for key in ITEM_KEYS:
            iid = int(e.get(key, 0) or 0)
            if iid:
                n += 1
                if iid not in table:
                    unk[iid] += 1
    return n, unk


# ============================================================================ detection input
def _missing(v: Any) -> bool:
    return v is None or type(v).__name__ == "NAType" or (isinstance(v, float) and math.isnan(v))


def normalise_detect_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep DETECT_REQUIRED (must exist) and the present DETECT_OPTIONAL fields, typed, sorted by tau."""
    out = []
    for r in rows:
        miss = [c for c in DETECT_REQUIRED if c not in r]
        if miss:
            raise KeyError(f"detection row lacks columns {miss}")
        coh = str(r["cohort"])
        if coh not in COHORT_CODES:
            raise ValueError(f"unknown cohort {coh!r}")
        fa = r["frame_age_ms"]
        d = {"match_id": str(r["match_id"]), "tau": int(r["tau"]), "first_kill_ts": int(r["first_kill_ts"]),
             "last_kill_ts": int(r["last_kill_ts"]), "cohort": coh, "clean": int(bool(r["clean"])),
             "isolated": int(bool(r["isolated"])),
             "frame_age_ms": -1 if _missing(fa) else int(fa)}
        for c in DETECT_OPTIONAL:
            if c in r and not _missing(r[c]):
                d[c] = int(r[c])
        out.append(d)
    out.sort(key=lambda d: d["tau"])
    return out


def _detect_columns(p: Path) -> List[str]:
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetDataset(p).schema.names)
    miss = [c for c in DETECT_REQUIRED if c not in schema_names]
    if miss:
        raise KeyError(f"detection table {p} lacks required columns {miss}")
    cols = list(DETECT_REQUIRED) + [c for c in DETECT_OPTIONAL if c in schema_names]
    if "patch" in schema_names:
        cols.append("patch")
    return cols


def detect_files_sha256(path: Path) -> Dict[str, str]:
    p = Path(path)
    files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    return {str(f): sha256_file(f) for f in files}


def read_detect_rows(path: Path, patch: str, match_ids: Sequence[str]) -> Dict[str, List[dict]]:
    """{match_id: normalised rows} for `match_ids` (a chunk) from a parquet file / directory, by column name."""
    import pyarrow.parquet as pq
    p = Path(path)
    cols = _detect_columns(p)
    want = set(match_ids)
    tab = pq.read_table(p, columns=cols, filters=[("match_id", "in", list(want))]).to_pandas()
    if "patch" in tab.columns:
        bad = sorted(set(str(x) for x in tab["patch"].unique()) - {patch})
        if bad:
            raise RuntimeError(f"detection table has rows of other patches {bad}")
    by: Dict[str, List[dict]] = {}
    for mid, g in tab.groupby("match_id", sort=False):
        by[str(mid)] = normalise_detect_rows(g.to_dict("records"))
    extra = sorted(set(by) - want)
    if extra:
        raise RuntimeError(f"detection table returned unrequested matches: {extra[:5]}")
    return by


def detect_table_info(path: Path, patch: str, match_ids: Sequence[str]) -> Dict[str, Any]:
    """Main-process check of the detection table (light columns only): schema, patch, row counts, sha256."""
    import pyarrow.parquet as pq
    p = Path(path)
    cols = _detect_columns(p)
    light = ["match_id", "isolated"] + (["patch"] if "patch" in cols else [])
    tab = pq.read_table(p, columns=light, filters=[("match_id", "in", list(set(match_ids)))]).to_pandas()
    if "patch" in tab.columns and set(str(x) for x in tab["patch"].unique()) - {patch}:
        raise RuntimeError("detection table has rows of other patches")
    n_rows = int(len(tab))
    n_noniso = int((tab["isolated"].astype(int) == 0).sum()) if n_rows else 0
    if n_rows > 100 and n_noniso == 0:
        raise RuntimeError("detection table has no non-isolated rows; it must hold ALL engagements (next_start)")
    return {"path": str(p), "columns_read": cols, "rows": n_rows, "rows_nonisolated": n_noniso,
            "matches_with_rows": int(tab["match_id"].nunique()) if n_rows else 0, "sha256": detect_files_sha256(p)}


def read_detect_table(path: Path, patch: str, match_ids: Sequence[str]) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    """(rows by match, info) - convenience wrapper of detect_table_info + read_detect_rows."""
    info = detect_table_info(path, patch, match_ids)
    return read_detect_rows(path, patch, match_ids), info


def inline_detect(pack: Mapping[str, Any], params: Any) -> List[Dict[str, Any]]:
    from gameplay.exact_population import detect_engagements_exact
    meta = pack["meta"]
    epack = {"meta": meta, "events": pack["events"], "minute_ts": pack["minute_ts"]}
    recs, _ = detect_engagements_exact(epack, meta["team_map"], params, mode="v4")
    return normalise_detect_rows(recs)


# ============================================================================ per match
def _vec(st) -> np.ndarray:
    return np.fromiter(st.values.values(), dtype=np.float64, count=len(st.values))


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def process_match(mid: str, patch: str, pack: Dict[str, Any], det_rows: List[Dict[str, Any]], *,
                  items_table, champion_table, h_sorted: Optional[np.ndarray], do_mart: bool) -> Dict[str, Any]:
    """All rows of one match.  Returns dict with eng / v / mart row lists (vectors float64) and status."""
    from gameplay.labels_exact import label_endpoint
    from gameplay.setup_features import SETUP_COLUMNS, build_setup
    from gameplay.state_value import final_outcome
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, StateBuilderV3

    ts = pack["minute_ts"]
    events = pack["events"]
    b = StateBuilderV3(pack, patch, items_table=items_table, champion_table=champion_table)
    slots = b.slot_by_pid
    ge = game_end_of(events)
    res: Dict[str, Any] = {"match_id": mid, "eng": [], "v": [], "mart": [], "status": "ok", "reason": "",
                           "n_events": len(events), "game_end": -1 if ge is None else ge,
                           "n_eng_input": len(det_rows), "n_eng_isolated": 0}
    nan_setup = np.full(len(SETUP_COLUMNS), np.nan)

    # (1) engagement pre-states (isolated, all cohorts)
    nxt = next_starts([r["tau"] for r in det_rows])
    for r, ns in zip(det_rows, nxt):
        if not r["isolated"]:
            continue
        q = r["tau"] - 1
        st = b.at(q)
        snap, age_pre = frame_age(ts, q)
        if st.snapshot_ms != snap:
            raise RuntimeError(f"{mid}: snapshot mismatch at {q}")
        snap_tau, age_tau = frame_age(ts, r["tau"])
        if r["frame_age_ms"] >= 0 and age_tau != r["frame_age_ms"]:
            raise RuntimeError(f"{mid}@{r['tau']}: input frame_age_ms {r['frame_age_ms']} != recomputed {age_tau}")
        setup = setup_vec = None
        if r["clean"]:
            setup = build_setup(pack, q, slots)
            setup_vec = np.fromiter(setup.values(), dtype=np.float64, count=len(setup))
        e90, reasons = label_endpoint(r["last_kill_ts"], events, h_s=H_S, next_start=ns, game_end=ge)
        meta_row = dict(r)
        meta_row.update(snapshot_ms=snap, frame_age_pre_ms=age_pre, setup_built=int(setup is not None),
                        next_start=-1 if ns is None else int(ns), game_end=-1 if ge is None else ge,
                        e90=int(e90), e90_reasons="|".join(reasons), e90_minus_tau=int(e90) - r["tau"])
        res["eng"].append((meta_row, _vec(st), nan_setup if setup_vec is None else setup_vec))
        res["n_eng_isolated"] += 1

    # (2) V rows: the winner is read here and only here (decision D4)
    try:
        y_blue, terminal = final_outcome(events)
    except ValueError as exc:
        res["status"], res["reason"] = "no_v_rows", f"final_outcome:{exc}"
        return res
    if ge is None or int(terminal) != int(ge):
        raise RuntimeError(f"{mid}: GAME_END disagreement {terminal} vs {ge}")
    for k, lo, hi, t in v_query_times(mid, terminal):
        st = b.at(t)
        snap, age = frame_age(ts, t)
        res["v"].append(({"match_id": mid, "bucket": k, "bucket_start": lo, "bucket_end": hi, "t": t,
                          "snapshot_ms": snap, "frame_age_ms": age, "game_end": int(terminal),
                          "y_blue_win": int(y_blue)}, _vec(st)))

    # (3) martingale interval (15.15 only)
    if do_mart:
        iv = mart_interval(mid, terminal, h_sorted)
        if iv is None:
            res["reason"] = "mart_game_too_short"
        else:
            t, h, hi_ = iv
            s0, s1 = b.at(t), b.at(t + h)
            snap0, age0 = frame_age(ts, t)
            snap1, age1 = frame_age(ts, t + h)
            row = {"match_id": mid, "t": t, "h_ms": h, "t1": t + h, "h_index": hi_, "snapshot0_ms": snap0,
                   "frame_age0_ms": age0, "snapshot1_ms": snap1, "frame_age1_ms": age1, "game_end": int(terminal)}
            row.update(mart_strata(ts, kill_times(events), t, h))
            res["mart"].append((row, _vec(s0), _vec(s1)))
    if len(STATE_V3_COLUMNS) != 996:
        raise RuntimeError("unexpected StateV3 width")
    return res


def spot_check(mid: str, patch: str, pack: Dict[str, Any], row: Mapping[str, Any], items_table, champion_table,
               full_vec: np.ndarray, full_setup: np.ndarray) -> Dict[str, Any]:
    """StateV3 / setup at tau - 1 from the pack truncated to <= tau - 1 (decision D9)."""
    from gameplay.setup_features import build_setup
    from gameplay.state_value_v3 import StateBuilderV3
    q = int(row["tau"]) - 1
    tp = truncate_pack(pack, q)
    bt = StateBuilderV3(tp, patch, items_table=items_table, champion_table=champion_table)
    ok_state = _same(full_vec, _vec(bt.at(q)))
    ok_setup = None
    if row["clean"]:
        tv = build_setup(tp, q, bt.slot_by_pid)
        ok_setup = _same(full_setup, np.fromiter(tv.values(), dtype=np.float64, count=len(tv)))
    return {"match_id": mid, "tau": int(row["tau"]), "clean": int(row["clean"]), "state_equal": bool(ok_state),
            "setup_equal": ok_setup}


# ============================================================================ chunk worker
def _parquet(rows: List[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows, columns=list(columns)) if rows else pd.DataFrame({c: [] for c in columns})
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


ENG_INDEX_COLUMNS = ("row", "match_id", "tau", "first_kill_ts", "last_kill_ts", "cohort", "cohort_code", "clean",
                     "isolated", "frame_age_ms", "frame_age_pre_ms", "snapshot_ms", "setup_built", "next_start",
                     "game_end", "e90", "e90_reasons", "e90_minus_tau") + DETECT_OPTIONAL
V_INDEX_COLUMNS = ("row", "match_id", "bucket", "bucket_start", "bucket_end", "t", "snapshot_ms", "frame_age_ms",
                   "game_end", "y_blue_win")
MART_INDEX_COLUMNS = ("row", "match_id", "t", "h_ms", "t1", "h_index", "snapshot0_ms", "frame_age0_ms",
                      "snapshot1_ms", "frame_age1_ms", "game_end", "frame_update", "n_frames_in", "deaths_pre60",
                      "deaths_in", "deaths_end60", "recent_deaths_ge3", "phase")
MATCH_COLUMNS = ("match_id", "status", "reason", "n_events", "game_end", "n_eng_input", "n_eng_isolated", "n_v",
                 "n_mart", "item_obs", "item_unknown", "seconds")


def run_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry: everything inside forbid_grid()."""
    from gameplay.grid_guard import forbid_grid
    with forbid_grid():
        return _run_chunk_inner(task)


def _run_chunk_inner(task: Dict[str, Any]) -> Dict[str, Any]:
    from gameplay.champion_attributes import load_champion_table_v2
    from gameplay.item_state import check_unknown_rate, load_item_table_v2
    from gameplay.setup_features import SETUP_COLUMNS, SETUP_NAME_HASH, name_hash as setup_name_hash
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH, name_hash

    t0 = time.time()
    if name_hash(STATE_V3_COLUMNS) != STATE_V3_NAME_HASH or setup_name_hash(SETUP_COLUMNS) != SETUP_NAME_HASH:
        raise RuntimeError("column name hash drift")
    patch, cid, out_dir = task["patch"], int(task["chunk"]), Path(task["out_dir"])
    params = verify_params(task["params"])        # preset + require_locked, re-checked in the worker
    cache = Path(task.get("cache", str(CACHE)))
    items_table = load_item_table_v2(patch)
    champion_table = load_champion_table_v2(patch)
    h_sorted = np.load(task["h_dist"]) if task.get("h_dist") else None
    do_mart = bool(task.get("do_mart"))
    det: Dict[str, List[dict]] = {}
    if task.get("detect_path"):
        # the worker reads only its own chunk's rows; the input must be the file the main process hashed
        if detect_files_sha256(Path(task["detect_path"])) != task["detect_sha256"]:
            raise RuntimeError("detection table changed during the run")
        det = read_detect_rows(Path(task["detect_path"]), patch, task["match_ids"])

    eng_meta, eng_x, eng_s, v_meta, v_x, m_meta, m_x0, m_x1, matches = [], [], [], [], [], [], [], [], []
    remakes: List[Dict[str, Any]] = []
    n_obs, unk = 0, Counter()
    spots: List[Dict[str, Any]] = []
    for mid in task["match_ids"]:
        tm0 = time.time()
        try:
            pack = load_pack(mid, patch, cache)
        except (OSError, ValueError, KeyError) as exc:
            matches.append({"match_id": mid, "status": "unreadable", "reason": f"{type(exc).__name__}:{exc}"[:200],
                            "n_events": 0, "game_end": -1, "n_eng_input": 0, "n_eng_isolated": 0, "n_v": 0,
                            "n_mart": 0, "item_obs": 0, "item_unknown": 0, "seconds": time.time() - tm0})
            continue
        ge = game_end_of(pack["events"])
        if EC.is_remake(ge):                        # decision D11: excluded before anything is built
            n_in = len(det.get(mid, [])) if not task.get("inline_detect") else 0
            remakes.append({"match_id": mid, "game_end": int(ge), "n_eng_input": n_in})
            matches.append({"match_id": mid, "status": EC.REMAKE_STATUS,
                            "reason": f"GAME_END {int(ge)} < {EC.REMAKE_MAX_GAME_END_MS}", "n_events": len(pack["events"]),
                            "game_end": int(ge), "n_eng_input": n_in, "n_eng_isolated": 0, "n_v": 0, "n_mart": 0,
                            "item_obs": 0, "item_unknown": 0, "seconds": time.time() - tm0})
            del pack
            continue
        o, u = item_observations(pack["events"], items_table)
        n_obs += o
        unk.update(u)
        rows = inline_detect(pack, params) if task.get("inline_detect") else det.get(mid, [])
        res = process_match(mid, patch, pack, rows, items_table=items_table, champion_table=champion_table,
                            h_sorted=h_sorted, do_mart=do_mart)
        if task.get("spot_check") and len(spots) < SPOT_CHECK_N and res["eng"]:
            meta_row, vec, svec = res["eng"][len(res["eng"]) // 2]
            spots.append(spot_check(mid, patch, pack, meta_row, items_table, champion_table, vec, svec))
        for meta_row, vec, svec in res["eng"]:
            meta_row = dict(meta_row, row=len(eng_meta), cohort_code=COHORT_CODES[meta_row["cohort"]])
            eng_meta.append(meta_row)
            eng_x.append(vec.astype(np.float32))
            eng_s.append(svec.astype(np.float32))
        for meta_row, vec in res["v"]:
            v_meta.append(dict(meta_row, row=len(v_meta)))
            v_x.append(vec.astype(np.float32))
        for meta_row, a, b in res["mart"]:
            m_meta.append(dict(meta_row, row=len(m_meta)))
            m_x0.append(a.astype(np.float32))
            m_x1.append(b.astype(np.float32))
        matches.append({"match_id": mid, "status": res["status"], "reason": res["reason"], "n_events": res["n_events"],
                        "game_end": res["game_end"], "n_eng_input": res["n_eng_input"],
                        "n_eng_isolated": res["n_eng_isolated"], "n_v": len(res["v"]), "n_mart": len(res["mart"]),
                        "item_obs": o, "item_unknown": int(sum(u.values())), "seconds": time.time() - tm0})
        del pack, res

    rate = check_unknown_rate(unk, n_obs)          # raises UnknownItemRateError (> 0.5 %): the chunk halts
    bad_spots = [s for s in spots if not s["state_equal"] or s["setup_equal"] is False]
    if task.get("spot_check"):
        if len(spots) < min(SPOT_CHECK_N, sum(1 for m in matches if m["n_eng_isolated"])):
            raise RuntimeError("spot check drew too few engagements")
        if bad_spots:
            raise RuntimeError(f"exactness spot check failed: {bad_spots[:3]}")

    W, S = len(STATE_V3_COLUMNS), len(SETUP_COLUMNS)

    def stack(rows, w):
        return np.stack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, w), dtype=np.float32)

    base = f"chunk_{cid:05d}"
    npz_path = out_dir / f"{base}.npz"
    tmp = out_dir / f"{base}.tmp.npz"
    np.savez_compressed(tmp, eng_X=stack(eng_x, W), eng_setup=stack(eng_s, S), v_X=stack(v_x, W),
                        m_X0=stack(m_x0, W), m_X1=stack(m_x1, W),
                        state_columns=np.asarray(STATE_V3_COLUMNS), setup_columns=np.asarray(SETUP_COLUMNS))
    os.replace(tmp, npz_path)
    _parquet(eng_meta, out_dir / f"{base}_eng.parquet", ENG_INDEX_COLUMNS)
    _parquet(v_meta, out_dir / f"{base}_v.parquet", V_INDEX_COLUMNS)
    _parquet(m_meta, out_dir / f"{base}_mart.parquet", MART_INDEX_COLUMNS)
    _parquet(matches, out_dir / f"{base}_matches.parquet", MATCH_COLUMNS)
    files = {}
    for suf in (".npz", "_eng.parquet", "_v.parquet", "_mart.parquet", "_matches.parquet"):
        f = out_dir / f"{base}{suf}"
        files[f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    status = Counter(m["status"] for m in matches)
    side = {"chunk": cid, "plan_hash": task["plan_hash"], "n_matches": len(task["match_ids"]),
            "match_ids_sha256": sha256_text("\n".join(task["match_ids"])),
            "rows": {"eng": len(eng_meta), "eng_clean_setup": int(sum(m["setup_built"] for m in eng_meta)),
                     "v": len(v_meta), "mart": len(m_meta)},
            "match_status": dict(status),
            "remakes": remakes,
            "item_check": {"observations": n_obs, "unknown": int(sum(unk.values())), "rate": rate,
                           "unknown_ids": {str(k): v for k, v in unk.most_common(20)}, "max_rate": 0.005},
            "spot_check": {"ran": bool(task.get("spot_check")), "n": len(spots), "failures": len(bad_spots),
                           "n_clean_checked": sum(1 for s in spots if s["setup_equal"] is not None), "cases": spots},
            "files": files, "seconds": round(time.time() - t0, 2),
            "seconds_per_match": round((time.time() - t0) / max(1, len(task["match_ids"])), 4)}
    write_json_atomic(out_dir / f"{base}.json", side)
    return side


def chunk_is_done(out_dir: Path, cid: int, plan_hash: str) -> Optional[Dict[str, Any]]:
    """The chunk sidecar, if it exists, carries plan_hash and every file verifies."""
    side_p = out_dir / f"chunk_{cid:05d}.json"
    if not side_p.exists():
        return None
    side = json.loads(side_p.read_text(encoding="utf-8"))
    if side.get("plan_hash") != plan_hash:
        return None
    for name, f in side.get("files", {}).items():
        p = out_dir / name
        if not p.exists() or sha256_file(p) != f["sha256"]:
            return None
    return side


# ============================================================================ main
def code_hashes() -> Dict[str, str]:
    mods = ("gameplay/exact_population.py", "gameplay/event_survival.py", "gameplay/cohorts_exact.py",
            "gameplay/item_state.py", "gameplay/champion_attributes.py", "gameplay/objective_timers.py",
            "gameplay/state_value_v3.py", "gameplay/state_value_v2.py", "gameplay/state_value.py",
            "gameplay/setup_features.py", "gameplay/labels_exact.py", "gameplay/grid_guard.py",
            "gameplay/split_guard.py", "core/presets.py", "scripts/exact_v4/ev4_common.py")
    out = {"scripts/exact_v4/ev4_02_extract.py": sha256_file(HERE)}
    for m in mods:
        p = WT / m
        if p.exists():
            out[m] = sha256_file(p)
    from gameplay.labels_exact import DEFAULT_DR_RULES
    out["DR/engagement_labels_v3_rules.py"] = sha256_file(Path(DEFAULT_DR_RULES))
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_h_dist(path: Path) -> Dict[str, Any]:
    """The 15.14 h distribution and its provenance (its sibling manifest must be patch 15.14 and list its sha)."""
    p = Path(path)
    man_p = p.parent / "manifest.json"
    if not p.is_file() or not man_p.is_file():
        raise FileNotFoundError(f"h distribution or its manifest missing: {p}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    digest = sha256_file(p)
    if man.get("patch") != "15.14":
        raise RuntimeError(f"h distribution must come from the 15.14 extract, manifest says {man.get('patch')}")
    if (man.get("h_distribution") or {}).get("sha256") != digest:
        raise RuntimeError("h distribution sha256 does not match its manifest")
    h = np.load(p)
    if h.ndim != 1 or not len(h) or np.any(np.diff(h) < 0):
        raise RuntimeError("h distribution must be a non-empty sorted 1-D array")
    return {"path": str(p), "sha256": digest, "n": int(len(h)), "source_sample": bool(man.get("sample", True)),
            "source_sample_limit": man.get("sample_limit"), "source_n_matches": man.get("n_matches"), "quantiles_ms": {q: float(np.quantile(h, q)) for q in (0.05, 0.5, 0.95)}}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patch", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--detect", type=Path, help="ev4_01_detect parquet file or directory")
    src.add_argument("--inline-detect", action="store_true", help="run detect_engagements_exact in the workers")
    ap.add_argument("--limit", type=int, default=None, help="first N matches by sha256(match_id) (smoke)")
    ap.add_argument("--matches", type=Path, default=None, help="match id list (.txt or .parquet with match_id)")
    ap.add_argument("--chunk-size", type=int, default=MAX_CHUNK)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--h-dist", type=Path, default=None, help="15.14 h_distribution.npy (needed for 15.15)")
    ap.add_argument("--allow-sample-h", action="store_true", help="accept an h distribution from a sample run")
    ap.add_argument("--record1", default=None)
    ap.add_argument("--record1-sha256", default=None)
    ap.add_argument("--cache", type=Path, default=CACHE)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    t_start = time.time()
    # --- guards before any data access
    EC.predecisions_info()                      # refuse a missing or edited author pre-decision record first
    access = check_patch_access(args.patch, args.record1, args.record1_sha256)
    patch = access["patch"]
    if not 1 <= args.chunk_size <= MAX_CHUNK:
        raise SystemExit(f"--chunk-size must be in [1, {MAX_CHUNK}]")
    workers = max(1, min(MAX_WORKERS, int(args.workers)))
    _, params, guard = load_params()
    from gameplay.setup_features import SETUP_COLUMNS, SETUP_NAME_HASH
    from gameplay.state_value_v3 import STATE_V3_COLUMNS, STATE_V3_NAME_HASH, STATE_VERSION, V_ONLY_COLUMNS
    do_mart = patch == "15.15"
    h_info = None
    if do_mart:
        if args.h_dist is None:
            raise SystemExit("15.15 needs --h-dist (the 15.14 h_distribution.npy)")
        h_info = load_h_dist(args.h_dist)
        if h_info["source_sample"] and not args.allow_sample_h:
            raise SystemExit("the h distribution comes from a sample run; pass --allow-sample-h to accept it")

    matches_file = args.matches
    if matches_file is None and args.detect is not None:
        sib = (args.detect if args.detect.is_dir() else args.detect.parent) / f"matches_{patch}.parquet"
        if sib.is_file():
            matches_file = sib                   # the ev4_01 match list (status 'ok') is the universe
    ids = match_universe(patch, args.limit, matches_file)
    out_dir = Path(args.out) / patch
    out_dir.mkdir(parents=True, exist_ok=True)
    det_info = {"mode": "inline", "detector": "gameplay.exact_population.detect_engagements_exact v4"}
    if args.detect is not None:
        det_info = detect_table_info(args.detect, patch, ids)
        det_info["mode"] = "parquet"
    code = code_hashes()
    plan = {"patch": patch, "state_v3_name_hash": STATE_V3_NAME_HASH, "setup_name_hash": SETUP_NAME_HASH,
            "state_version": STATE_VERSION, "params": guard["params"], "record1a_sha256": guard["record1a_sha256"],
            "match_ids_sha256": sha256_text("\n".join(ids)), "chunk_size": args.chunk_size, "code": code,
            "detect": {k: v for k, v in det_info.items() if k in ("mode", "sha256", "columns_read")},
            "h_dist_sha256": None if h_info is None else h_info["sha256"], "decisions": DECISIONS,
            "remake_threshold_ms": EC.REMAKE_MAX_GAME_END_MS}
    plan_hash = sha256_text(json.dumps(plan, sort_keys=True, default=json_default))
    chunks = [ids[i:i + args.chunk_size] for i in range(0, len(ids), args.chunk_size)]
    tasks = []
    sides: Dict[int, Dict[str, Any]] = {}
    for cid, mids in enumerate(chunks):
        done = chunk_is_done(out_dir, cid, plan_hash)
        if done is not None:
            sides[cid] = done
            continue
        tasks.append({"patch": patch, "chunk": cid, "match_ids": mids, "out_dir": str(out_dir),
                      "plan_hash": plan_hash, "params": guard["params"], "inline_detect": bool(args.inline_detect),
                      "detect_path": None if args.detect is None else str(args.detect),
                      "detect_sha256": det_info.get("sha256"),
                      "h_dist": None if h_info is None else h_info["path"], "do_mart": do_mart,
                      "spot_check": cid == 0, "cache": str(args.cache)})
    print(f"[ev4_02] patch {patch}: {len(ids)} matches, {len(chunks)} chunks ({len(tasks)} to run), "
          f"{workers} workers, out {out_dir}", flush=True)
    t_run = time.time()
    if tasks:
        if workers == 1 or len(tasks) == 1:
            for tk in tasks:
                s = run_chunk(tk)
                sides[s["chunk"]] = s
                print(f"[ev4_02] chunk {s['chunk']} done: {s['rows']} {s['seconds']} s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_chunk, tk): tk["chunk"] for tk in tasks}
                for f in as_completed(futs):
                    try:
                        s = f.result()
                    except Exception:
                        for g in futs:
                            g.cancel()
                        print(f"[ev4_02] chunk {futs[f]} FAILED", flush=True)
                        raise
                    sides[s["chunk"]] = s
                    print(f"[ev4_02] chunk {s['chunk']} done: {s['rows']} {s['seconds']} s", flush=True)
    wall = time.time() - t_run

    # --- 15.14: h distribution (e90 - tau of isolated engagements, all cohorts) for the 15.15 martingale check
    import pandas as pd
    h_out = None
    if patch == "15.14":
        hs = []
        for cid in sorted(sides):
            df = pd.read_parquet(out_dir / f"chunk_{cid:05d}_eng.parquet", columns=["isolated", "e90_minus_tau"])
            hs.append(df.loc[df["isolated"] == 1, "e90_minus_tau"].to_numpy(np.int64))
        h = np.sort(np.concatenate(hs)) if hs else np.zeros(0, np.int64)
        h = h[h > 0]
        np.save(out_dir / "h_distribution.npy", h)
        h_out = {"file": "h_distribution.npy", "sha256": sha256_file(out_dir / "h_distribution.npy"), "n": int(len(h)),
                 "quantiles_ms": {str(q): float(np.quantile(h, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(h) else {}}

    tot = Counter()
    status = Counter()
    n_obs = n_unk = 0
    remakes = sorted((r for s in sides.values() for r in s.get("remakes", [])), key=lambda r: r["match_id"])
    for s in sides.values():
        tot.update(s["rows"])
        status.update(s["match_status"])
        n_obs += s["item_check"]["observations"]
        n_unk += s["item_check"]["unknown"]
    total_bytes = sum(f["bytes"] for s in sides.values() for f in s["files"].values())
    manifest = {
        "script": "scripts/exact_v4/ev4_02_extract.py", "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "git_head": git_head(), "plan_hash": plan_hash, "patch": patch, "access": access,
        "sample": bool(args.limit is not None or len(ids) < FULL_RUN_MIN_MATCHES),
        "sample_limit": args.limit, "n_matches": len(ids),
        "match_list": {"source": str(matches_file) if matches_file else str(PATCH_INDEX),
                       "sha256": sha256_file(Path(matches_file)) if matches_file else None,
                       "order": "sha256(match_id)"}, "match_ids_sha256": plan["match_ids_sha256"],
        "n_chunks": len(chunks), "chunk_size": args.chunk_size, "workers": workers,
        "guards": {"apply_preset": PRESET, "require_locked": True, "forbid_grid": "every worker (run_chunk)",
                   "split_guard": "check_patch_access before data", "check_unknown_rate": "per chunk, max 0.5%",
                   "exactness_spot_check": "chunk 0, 20 engagements", **guard},
        "decisions": DECISIONS, "code_sha256": code, "detect_input": det_info,
        "remakes": {**EC.remake_rule(), "n_excluded": len(remakes), "match_ids": [r["match_id"] for r in remakes],
                    "excluded": remakes},
        "predecisions": {k: v for k, v in EC.predecisions_info().items() if k != "content"},
        "STATE_V3_NAME_HASH": STATE_V3_NAME_HASH, "SETUP_NAME_HASH": SETUP_NAME_HASH, "state_version": STATE_VERSION,
        "n_state_columns": len(STATE_V3_COLUMNS), "n_setup_columns": len(SETUP_COLUMNS),
        "v_only_columns": list(V_ONLY_COLUMNS),
        "arrays": {"eng_X": "float32 (n_eng, 996) StateV3 at tau-1", "eng_setup": "float32 (n_eng, 157), NaN = not clean",
                   "v_X": "float32 (n_v, 996) StateV3 at t", "m_X0": "float32 (n_m, 996) StateV3 at t",
                   "m_X1": "float32 (n_m, 996) StateV3 at t+h"},
        "rows": dict(tot), "match_status": dict(status),
        "rows_per_match": {k: v / max(1, len(ids)) for k, v in tot.items()},
        "item_check": {"observations": n_obs, "unknown": n_unk, "rate": n_unk / max(1, n_obs)},
        "spot_check": sides.get(0, {}).get("spot_check"),
        "h_distribution": h_out, "h_source": h_info,
        "bytes_total": total_bytes, "bytes_per_1000_matches": total_bytes / max(1, len(ids)) * 1000,
        "seconds_wall_this_invocation": round(wall, 1), "seconds_total_this_invocation": round(time.time() - t_start, 1),
        "worker_seconds_per_match": sum(s["seconds"] for s in sides.values()) / max(1, len(ids)),
        "chunks": {f"{cid:05d}": {"n_matches": s["n_matches"], "rows": s["rows"], "files": s["files"],
                                  "item_rate": s["item_check"]["rate"], "seconds": s["seconds"]}
                   for cid, s in sorted(sides.items())},
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(f"[ev4_02] remakes excluded (GAME_END < {EC.REMAKE_MAX_GAME_END_MS} ms): {len(remakes)} "
          f"{[r['match_id'] for r in remakes][:20]}", flush=True)
    print(json.dumps({k: manifest[k] for k in ("patch", "n_matches", "rows", "rows_per_match", "match_status",
                                               "item_check", "bytes_per_1000_matches", "seconds_wall_this_invocation",
                                               "worker_seconds_per_match")}, indent=1, default=json_default), flush=True)
    return manifest


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
