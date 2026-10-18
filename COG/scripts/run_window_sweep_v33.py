"""Observation-window and bin-width sweep on the v3.3 corpus (CoG 2026 review R2).

R2: "The observation window spans only 6 timesteps across 30 seconds ... experiments with
alternative temporal resolutions or extended observation windows are recommended."  The only
earlier window sweep (features/causal_stratified_w{10,20,30}.json) never changed the telemetry
model -- telemetry_auc is 0.5463352952517039 in all three files -- so it answers nothing.  This
script changes the telemetry representation itself, and proves that it did:

    ctx  15 / 30 / 60 / 120 s at BIN_MS 5,000     L = 3 / 6 / 12 / 24 bins
    BIN_MS 2,500 / 10,000 at ctx 30 s              L = 12 / 3 bins

(gameplay/pipeline.build_ms_sequence: L = ctx_ms // bin_ms bins ending at the engage cutoff;
gameplay/features.seq_to_tabular: 7 statistics per base feature.)

Held fixed
  * the corpus definition: core.presets "v3.3" (G 13.7 s, D 4,264 u, R 1,600 u, B 15 s, horizon
    35 s, market_event label with engagement attribution, clean feature path: TIME_NORM_ABSOLUTE,
    ANCHORS_CAUSAL, TAB_FRAME_AGE_FEATURE), draws dropped (LABEL_TIE_POLICY drop);
  * the engagements: ONE fight index, built under the reference setting (ctx 30 s, BIN_MS 5,000),
    feeds every setting.  The detector is rebuilt under every other setting and compared field
    by field (--verify-index); any difference aborts unless --allow-index-drift.  None is
    expected: the detector reads FIGHT_CONTEXT_SEC only in its guard engage_ts - ctx < t_min
    (gameplay/fights.py, "Context / horizon guards"), which START_OFFSET_MIN = 2 min already
    implies for ctx <= 120 s, and BIN_MS reaches it only as the fallback of DETECT_STEP_MS, which
    the CFG sets explicitly.  A row is identified by (match_id, engage_ts, position in the shared
    index); models are scored on the rows every setting produced (the COMMON rows), and their
    labels must agree;
  * the learner: LightGBM (Ke et al., NeurIPS 2017) with the paper's parameters --
      patch  lgbm_paper of scripts/run_model_comparison_v33.py::run_model (commit 3fb00c3):
             n_estimators 400, learning_rate 0.05, num_leaves 31, subsample 0.9,
             colsample_bytree 0.9, random_state 7; fit 15.14, early_stopping(100) on 15.15,
             score 15.16
      oof    scripts/run_scale_decomposition.py::oof_predictions: same parameters, sklearn
             GroupKFold(5) by match, no early stopping.

Proven to change
  per setting the JSON records L, the feature-matrix shape and the SHA-1 of the telemetry block
  (every column except frame_age_s) on the common rows; the run aborts before any fit if two
  settings produce identical telemetry matrices.

Uncertainty
  match-clustered percentile bootstrap (Efron & Tibshirani 1993, ch. 13; whole clusters resampled
  with replacement, Field & Welsh 2007, JRSS-B 69(3), sec. 2.1): a CI per setting and the paired
  CI of every setting minus the reference, with the SAME match draw applied to both prediction
  vectors.  A drawn match enters with its multiplicity and AUC is the weighted Mann-Whitney
  statistic (Hanley & McNeil 1982, eq. 1; ties count one half) -- identical to concatenating the
  drawn matches' rows.

Leak discipline
  Features come from train.baseline.build_tabular_Xy under the preset: frames strictly before the
  cutoff (pipeline_interp max_snapshot_ms = cutoff - 1), events in [b0, b1) with b1 <= cutoff,
  absolute time_norm, causal anchors, frame age measured at the cutoff.  In the patch protocol
  15.15 is used only for early stopping and 15.16 only for scoring.  The column filter (merge_shards'
  max > min rule) reads each fit's training rows only: 15.14 for the patch holdout, the training
  folds for OOF.  A leak probe (--leak-probe-rows) rebuilds sampled engagements from a copy of the
  match pack whose frames and events at or after the cutoff are rewritten (plus a large-bounty
  kill, a Baron and a tower exactly at the cutoff, and jittered match-wide anchors): the per-bin
  sequence must stay bit-identical, while noise on the frames before the cutoff must change it.

Caching: every setting's matrix (and, for --save-seq settings, the per-bin sequence) is written
atomically to --out-dir as .npz with a JSON signature, so a crash never rebuilds a finished
setting; LightGBM predictions and index verifications are cached the same way.  A signature names
the preset, match list, setting, the SHA-1 of the feature-defining sources and of this script --
not the git commit, which moves with unrelated commits to the shared checkout; the commit each
cache was built at is kept in its sidecar and reported per setting (built_git_commit).

    LOL_OUTPUT_ROOT=D:/LOL_Project LOL_CFG_PRESET=v3.3 python scripts/run_window_sweep_v33.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ITEM = "A3-temporal-windows"
PRESET = "v3.3"
SEED = 7
LABEL_KEY = "market_event"
TIE_POLICY = "drop"
DEFAULT_OUT_DIR = Path("D:/LOL_Project/fusion_2615/features/tog_revision") / ITEM
DEFAULT_SETTINGS = "15:5000,30:5000,60:5000,120:5000,30:2500,30:10000"
REFERENCE = "30:5000"
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
                   colsample_bytree=0.9, random_state=SEED, verbose=-1)
PATCH_EARLY_STOPPING = 100
FRAME_AGE = "frame_age_s"
REF_FIELDS = ("match_id", "patch", "t_start", "t_start_ts", "label_end_ts", "first_kill_ts", "last_kill_ts",
              "det_cluster_blue", "det_cluster_red", "det_present_blue", "det_present_red", "anchor_x", "anchor_y")
_STR_FIELDS = ("match_id", "patch")
_FLOAT_FIELDS = ("anchor_x", "anchor_y")
OWNED_OVERRIDES = ("FIGHT_CONTEXT_SEC", "BIN_MS", "FIGHT_INDEX_CACHE_ENABLED", "DUMP_FIGHTS")
SNAPSHOT_KEYS = ("FIGHT_CONTEXT_SEC", "BIN_MS", "FRAME_MS", "DETECT_STEP_MS", "START_OFFSET_MIN",
                 "PREDICTION_GAP_MS", "LABEL_TIE_POLICY", "TEMPORAL_SEQ_PRIORITY", "USE_PER_PLAYER_ITEMS",
                 "USE_GAME_PHASE", "USE_MOMENTUM_FEATURES", "ZERO_XY_IN_EXTRA_SEQ", "INTERP_SCALARS_METHOD",
                 "FIGHT_INDEX_NUM_WORKERS")
# code whose behaviour defines the engagements, labels and features; hashed into every cache signature
FEATURE_SOURCES = ("gameplay", "data/index_split.py", "data/cache_io.py", "data/events_index.py", "train/baseline.py",
                   "core/config.py", "core/presets.py", "core/feature_contract.py", "core/timeutils.py",
                   "core/interpolation.py", "core/fight_types.py", "core/contract.py", "core/time_contract.py")
PACK_KEYS = ("minute_ts", "node_minute", "global_minute", "gold_team_minute", "xy_raw_minute", "events", "meta")
FRAME_KEYS = ("node_minute", "global_minute", "gold_team_minute", "xy_raw_minute")
LEAK_PROBE_SCOPE = (
    "Per setting, sampled engagements are rebuilt from a copy of the match pack in which every frame stamped at "
    "or after the cutoff carries noise, every event at or after the cutoff has scrambled actors, team and "
    "position, a large-bounty kill, a Baron and a tower kill are added exactly at the cutoff, and the match-wide "
    "meta anchors are jittered; x_seq and extra_seq must be bit-identical.  Positive control: noise on the "
    "frames before the cutoff must change x_seq.  Scope: the feature read path.  The engagement's own kill "
    "timestamps and anchor are passed only to the label, and frame fields derived when the cache was built are "
    "not re-derived.")
DEVIATIONS = [
    "LightGBM n_jobs is set explicitly (--n-jobs); the paper used n_jobs=-1.  Histogram sums are "
    "thread-count dependent at floating-point level, so trees can differ in the last digits.",
    "Sample of --n-matches matches from the cache (paper: all 191,940 matches with rows).",
    "The constant-column filter (max > min, merge_shards' rule) is computed on the TRAINING rows of each "
    "fit -- the 15.14 rows for the patch holdout, each fold's training matches for OOF -- so no value of a "
    "validation, test or held-out-fold row reaches column selection.  The headline applied the same "
    "label-free rule once over all corpus rows; LightGBM already discards columns that are constant in its "
    "training data, so the fitted trees are expected to be unaffected.",
    "Rows are ordered by (match_id, engage_ts) before fitting (paper: shard order), so GroupKFold "
    "fold membership differs from the headline run.",
    "Bootstrap is a match-level percentile bootstrap with --n-boot draws (headline scale "
    "decomposition used 1,000; model comparison used 300), computed as a multiplicity-weighted "
    "Mann-Whitney AUC instead of concatenating resampled rows (numerically identical).",
    "Settings other than ctx 30 s / BIN_MS 5,000 are not published corpus configurations; the "
    "frames-in-window statistic assumes frames every FRAME_MS and ignores bin-midpoint sampling.",
]


def log(*parts) -> None:
    print(*parts, flush=True)


# ------------------------------------------------------------------ settings and configuration

def parse_settings(spec: str) -> list[tuple[int, int]]:
    """'ctx_sec:bin_ms,...' -> [(ctx_sec, bin_ms)]; bin defaults to 5,000 and must divide ctx exactly."""
    out: list[tuple[int, int]] = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        ctx, _, bin_ms = item.partition(":")
        ctx_sec, bin_val = int(ctx), int(bin_ms or 5000)
        if ctx_sec <= 0 or bin_val <= 0 or (ctx_sec * 1000) % bin_val:
            raise ValueError(f"setting {item!r}: ctx must be a positive whole multiple of BIN_MS")
        if (ctx_sec, bin_val) not in out:
            out.append((ctx_sec, bin_val))
    return out


def n_bins(ctx_sec: int, bin_ms: int) -> int:
    """L as gameplay/pipeline.build_ms_sequence computes it."""
    return int(ctx_sec * 1000) // int(bin_ms)


def setting_name(ctx_sec: int, bin_ms: int) -> str:
    return f"ctx{int(ctx_sec)}_bin{int(bin_ms)}"


def check_env_overrides() -> None:
    """The corpus must be exactly v3.3: refuse foreign LOL_CFG_OVERRIDES (they would beat the preset)."""
    raw = str(os.environ.get("LOL_CFG_OVERRIDES", "")).strip()
    if not raw:
        return
    foreign = sorted(set(json.loads(raw)) - set(OWNED_OVERRIDES))
    if foreign:
        raise SystemExit(f"LOL_CFG_OVERRIDES sets {foreign}; this script runs preset {PRESET} only")


def configure(ctx_sec: int, bin_ms: int, index_workers: int = 1) -> dict:
    """Apply preset v3.3 and one setting in-process, and in the environment that index workers inherit.

    build_fight_index worker processes re-import core.config, which applies LOL_CFG_PRESET and then
    LOL_CFG_OVERRIDES, so the setting must travel through the environment as well as through cfg.
    """
    check_env_overrides()
    from core.config import cfg
    from core.presets import apply_preset
    applied = apply_preset(cfg, PRESET)
    cfg.FIGHT_CONTEXT_SEC = int(ctx_sec)
    cfg.BIN_MS = int(bin_ms)
    cfg.LABEL_TIE_POLICY = TIE_POLICY          # read by gameplay/labels.py via getattr
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.DUMP_FIGHTS = False
    cfg.FIGHT_INDEX_NUM_WORKERS = int(index_workers)
    os.environ["LOL_CFG_PRESET"] = PRESET
    os.environ["LOL_CFG_OVERRIDES"] = json.dumps({"FIGHT_CONTEXT_SEC": int(ctx_sec), "BIN_MS": int(bin_ms),
                                                  "FIGHT_INDEX_CACHE_ENABLED": False, "DUMP_FIGHTS": False})
    clean = {k: bool(getattr(cfg, k)) for k in ("TIME_NORM_ABSOLUTE", "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE")}
    if not all(clean.values()) or str(cfg.LABEL_TYPE) != LABEL_KEY or int(getattr(cfg, "PREDICTION_GAP_MS", 0)) != 0:
        raise SystemExit(f"not the clean v3.3 path: {clean} LABEL_TYPE={cfg.LABEL_TYPE} "
                         f"PREDICTION_GAP_MS={getattr(cfg, 'PREDICTION_GAP_MS', None)}")
    return applied


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def cfg_snapshot() -> dict:
    from core.config import CACHE_DIR, cfg
    from core.presets import PRESETS
    out = {k: _jsonable(getattr(cfg, k, None)) for k in list(PRESETS[PRESET]) + list(SNAPSHOT_KEYS)}
    out["CACHE_DIR"] = str(CACHE_DIR)
    return out


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return ""
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked_files": [line[3:] for line in status.splitlines() if line.strip()]}


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def source_digest(root: Path = PROJECT_ROOT) -> str:
    """SHA-1 over the feature-defining sources as they are on disk when the process starts."""
    h = hashlib.sha1()
    for rel in FEATURE_SOURCES:
        p = root / rel
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            if f.exists():
                h.update(str(f.relative_to(root)).replace("\\", "/").encode("utf-8"))
                h.update(f.read_bytes())
    return h.hexdigest()


def sha1_lines(items) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(str(it).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def common_rows_sha1(groups, engage_ts, y) -> str:
    return sha1_lines(f"{g}|{int(t)}|{int(v)}" for g, t, v in zip(
        np.asarray(groups).astype(str).tolist(), np.asarray(engage_ts).astype(np.int64).tolist(),
        np.asarray(y).astype(np.int64).tolist()))


def sample_match_ids(cache_dir: Path, n_matches: int | None, seed: int) -> list[str]:
    """Sorted cache ids, seeded sample, sorted again (deterministic regardless of glob order)."""
    suffix = ".meta.json"
    mids = sorted(p.name[: -len(suffix)] for p in Path(cache_dir).glob("*" + suffix))
    if n_matches and int(n_matches) < len(mids):
        mids = sorted(random.Random(int(seed)).sample(mids, int(n_matches)))
    return mids


# ------------------------------------------------------------------ fight index

def refs_to_arrays(refs) -> dict:
    out = {}
    for f in REF_FIELDS:
        vals = [getattr(r, f) for r in refs]
        if f in _STR_FIELDS:
            out[f] = np.array([str(v) for v in vals], dtype=str)
        elif f in _FLOAT_FIELDS:
            out[f] = np.array(vals, dtype=np.float64)
        else:
            out[f] = np.array(vals, dtype=np.int64)
    return out


def refs_from_arrays(arrays) -> list:
    from core.fight_types import FightRef
    cols = {f: np.asarray(arrays[f]).tolist() for f in REF_FIELDS}
    out = []
    for i in range(len(cols["match_id"])):
        kw = {}
        for f in REF_FIELDS:
            v = cols[f][i]
            kw[f] = str(v) if f in _STR_FIELDS else float(v) if f in _FLOAT_FIELDS else int(v)
        out.append(FightRef(**kw))
    return out


def ref_tuples(refs) -> list[tuple]:
    return sorted(tuple(getattr(r, f) for f in REF_FIELDS) for r in refs)


def index_sha1(refs) -> str:
    return sha1_lines(repr(t) for t in ref_tuples(refs))


def compare_indexes(reference, other) -> dict:
    """Field-by-field multiset comparison of two fight indexes."""
    ca, cb = Counter(ref_tuples(reference)), Counter(ref_tuples(other))
    only_a, only_b = ca - cb, cb - ca
    return {"n_reference": int(sum(ca.values())), "n_setting": int(sum(cb.values())),
            "only_reference": int(sum(only_a.values())), "only_setting": int(sum(only_b.values())),
            "identical": not only_a and not only_b,
            "examples_only_reference": [_jsonable(list(t)) for t in list(only_a)[:3]],
            "examples_only_setting": [_jsonable(list(t)) for t in list(only_b)[:3]]}


def build_index(mids: list[str], ctx_sec: int, bin_ms: int, index_workers: int) -> list:
    configure(ctx_sec, bin_ms, index_workers)
    from data.index_split import build_fight_index
    return build_fight_index(cache_match_ids=list(mids))


# ------------------------------------------------------------------ features

@contextmanager
def capture_sequences(n_max: int, keep: bool):
    """Record the per-bin sequence that build_tabular_Xy summarises, from inside the canonical builder.

    train.baseline.build_tabular_Xy calls seq_to_tabular exactly once for every row it keeps, just
    before appending that row, so the i-th call belongs to the i-th row of X.  The wrapper stores
    the array (or only its shape) and returns the original summary unchanged.
    """
    import train.baseline as tb
    original = tb.seq_to_tabular
    state = {"n": 0, "shapes": set(), "buf": None}

    def wrapped(x_seq):
        x = np.asarray(x_seq, dtype=np.float32)
        state["shapes"].add(tuple(int(s) for s in x.shape))
        if keep:
            if state["buf"] is None:
                state["buf"] = np.empty((max(1, int(n_max)),) + x.shape, dtype=np.float32)
            state["buf"][state["n"]] = x
        state["n"] += 1
        return original(x_seq)

    tb.seq_to_tabular = wrapped
    try:
        yield state
    finally:
        tb.seq_to_tabular = original


def build_setting(refs, feature_set: str = "full", keep_seq: bool = False) -> dict:
    """Tabular matrix (and optionally the per-bin sequence) for the current cfg, via build_tabular_Xy."""
    import train.baseline as tb
    started = time.time()
    with capture_sequences(len(refs), keep_seq) as cap:
        X, y, names, used = tb.build_tabular_Xy(list(refs), feature_set=feature_set)
    if cap["n"] != len(used):
        raise RuntimeError(f"captured {cap['n']} sequences for {len(used)} rows: build_tabular_Xy's loop changed")
    if len(cap["shapes"]) > 1:
        raise RuntimeError(f"sequence shape varies across rows: {sorted(cap['shapes'])}")
    shape = next(iter(cap["shapes"])) if cap["shapes"] else (0, 0)
    y = np.asarray(y)
    if y.size and not np.isin(y, (0, 1)).all():
        raise RuntimeError("labels outside {0, 1}: draws were not dropped")
    rows = refs_to_arrays(used)
    position = {id(r): i for i, r in enumerate(refs)}       # build_tabular_Xy keeps the ref objects it was given
    out = {"X": np.asarray(X, dtype=np.float32), "y": y.astype(np.int8), "names": list(names),
           "L": int(shape[0]), "D_seq": int(shape[1]) if len(shape) > 1 else 0,
           "build_seconds": round(time.time() - started, 1)}
    out.update({f"row_{f}": rows[f] for f in REF_FIELDS})
    out["row_ref_ordinal"] = np.fromiter((position[id(r)] for r in used), dtype=np.int64, count=len(used))
    if keep_seq:
        buf = cap["buf"]
        out["seq"] = buf[: cap["n"]] if buf is not None else np.zeros((0,) + tuple(shape), dtype=np.float32)
    return out


def row_keys(data: dict) -> list[tuple[str, int, int]]:
    """(match_id, engage_ts, position in the shared fight index): unique even when two engagements share a timestamp."""
    return list(zip(np.asarray(data["row_match_id"]).astype(str).tolist(),
                    np.asarray(data["row_t_start_ts"]).astype(np.int64).tolist(),
                    np.asarray(data["row_ref_ordinal"]).astype(np.int64).tolist()))


# ------------------------------------------------------------------ leak probe

def _jitter(obj, rng: np.random.Generator):
    if isinstance(obj, dict):
        return {k: _jitter(v, rng) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jitter(v, rng) for v in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj) + float(rng.uniform(500.0, 2000.0))
    return obj


def perturb_pack(pack: dict, cutoff_ms: int, seed: int, side: str = "future") -> dict:
    """Copy of a match pack with its data rewritten on one side of the cutoff.

    side="future": frames stamped >= cutoff_ms get noise, events stamped >= cutoff_ms get scrambled
    actors, team and position, decisive synthetic events are added exactly at the cutoff, and the
    match-wide meta anchors are jittered.  A causal feature path must not change.
    side="past": frames stamped < cutoff_ms get noise -- the positive control.
    Only the raw pack keys are copied, so no event index or derived cache built from the original
    pack survives into the copy.
    """
    from data.events_index import _attach_event_index_inplace
    if side not in ("future", "past"):
        raise ValueError(side)
    rng = np.random.default_rng(int(seed))
    cutoff_ms = int(cutoff_ms)
    out = {k: pack[k] for k in PACK_KEYS if k in pack}
    ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    rows = ts >= cutoff_ms if side == "future" else ts < cutoff_ms
    for key in FRAME_KEYS:
        if key in pack:
            arr = np.array(pack[key], dtype=np.float32, copy=True)
            if rows.any():
                arr[rows] += rng.uniform(0.5, 5.0, size=arr[rows].shape).astype(np.float32)
            out[key] = arr
    events = []
    for e in pack.get("events") or []:
        e2 = dict(e)
        t = int(e2.get("timestamp", -1) or -1)
        if side == "future" and t >= cutoff_ms:
            if isinstance(e2.get("position"), dict):
                e2["position"] = {"x": float(rng.uniform(0.0, 14800.0)), "y": float(rng.uniform(0.0, 14800.0))}
            for k in ("killerId", "victimId", "participantId", "creatorId"):
                if k in e2:
                    e2[k] = int(rng.integers(1, 11))
            if e2.get("teamId") in (100, 200):
                e2["teamId"] = 300 - int(e2["teamId"])
        events.append(e2)
    if side == "future":
        centre = {"x": 7400.0, "y": 7400.0}
        events += [
            {"type": "CHAMPION_KILL", "timestamp": cutoff_ms, "killerId": 1, "victimId": 6, "bounty": 3000,
             "shutdownBounty": 1000, "killStreakLength": 9, "position": dict(centre),
             "assistingParticipantIds": [2, 3, 4, 5]},
            {"type": "ELITE_MONSTER_KILL", "timestamp": cutoff_ms, "killerId": 1, "killerTeamId": 100,
             "monsterType": "BARON_NASHOR", "position": dict(centre)},
            {"type": "BUILDING_KILL", "timestamp": cutoff_ms, "killerId": 1, "teamId": 200,
             "buildingType": "TOWER_BUILDING", "laneType": "MID_LANE", "towerType": "OUTER_TURRET",
             "position": dict(centre)},
            {"type": "ITEM_PURCHASED", "timestamp": cutoff_ms, "participantId": 1, "itemId": 3031},
            {"type": "WARD_PLACED", "timestamp": cutoff_ms, "creatorId": 1, "wardType": "CONTROL_WARD"},
        ]
        events.sort(key=lambda ev: int(ev.get("timestamp", -1) or -1))
        if isinstance(pack.get("meta"), dict) and pack["meta"].get("anchors") is not None:
            out["meta"] = {**pack["meta"], "anchors": _jitter(pack["meta"]["anchors"], rng)}
    out["events"] = events
    _attach_event_index_inplace(out)
    return out


def leak_probe(refs, n_rows: int, seed: int, feature_set: str = "full") -> dict:
    """Is the per-bin sequence invariant to everything at or after the cutoff?  (current cfg)"""
    import train.baseline as tb
    order = list(range(len(refs)))
    random.Random(int(seed)).shuffle(order)
    stats = {"probed": 0, "sequence_identical_after_future_perturbation": 0,
             "label_changed_by_future_perturbation": 0, "skipped_label_unavailable": 0,
             "positive_control_evaluated": 0, "positive_control_sequence_changed": 0, "failures": []}

    def features(pack, r, cutoff):
        raw = tb.build_ms_sequence(pack, pack["meta"]["team_map"], -1, engage_ts=cutoff,
                                   label_end_ts=tb._ref_label_end_ts(r), first_kill_ts=tb._ref_first_kill_ts(r),
                                   last_kill_ts=tb._ref_last_kill_ts(r), anchor_xy=tb._ref_anchor_xy(r))
        if not raw:
            return None
        return tb.build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None), feature_set)

    for i in order:
        if stats["probed"] >= int(n_rows):
            break
        r = refs[i]
        cutoff = tb._ref_engage_ts(r)
        pack = tb.load_match_cache(r.match_id) if cutoff is not None else None
        if not pack:
            continue
        base = features(pack, r, cutoff)
        if base is None:
            continue
        future = features(perturb_pack(pack, cutoff, seed + i, "future"), r, cutoff)
        if future is None:
            stats["skipped_label_unavailable"] += 1
            continue
        stats["probed"] += 1
        keys = [k for k in ("x_seq", "extra_seq") if k in base]
        if all(np.array_equal(base[k], future[k], equal_nan=True) for k in keys):
            stats["sequence_identical_after_future_perturbation"] += 1
        else:
            stats["failures"].append([r.match_id, int(cutoff)])
        stats["label_changed_by_future_perturbation"] += int(int(base["y"]) != int(future["y"]))
        past = features(perturb_pack(pack, cutoff, seed + i, "past"), r, cutoff)
        if past is not None:
            stats["positive_control_evaluated"] += 1
            stats["positive_control_sequence_changed"] += int(not np.array_equal(base["x_seq"], past["x_seq"], equal_nan=True))
    stats["passed"] = (stats["probed"] > 0
                       and stats["sequence_identical_after_future_perturbation"] == stats["probed"]
                       and stats["positive_control_sequence_changed"] == stats["positive_control_evaluated"])
    return stats


# ------------------------------------------------------------------ caches

def sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name[: -len(".npz")] + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def canonical(sig: dict) -> dict:
    return json.loads(json.dumps(_jsonable(sig), sort_keys=True))


def cache_valid(path: Path, signature: dict) -> bool:
    side = sidecar(path)
    if not (path.exists() and side.exists()):
        return False
    try:
        return json.loads(side.read_text(encoding="utf-8")).get("signature") == canonical(signature)
    except Exception:
        return False


def save_setting(path: Path, data: dict, signature: dict, provenance: dict | None = None) -> None:
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and k != "seq"}
    arrays["names"] = np.array(data["names"], dtype=str)
    atomic_savez(path, **arrays)
    write_json(sidecar(path), {"signature": canonical(signature), "L": data["L"], "D_seq": data["D_seq"],
                               "shape": list(data["X"].shape), "build_seconds": data["build_seconds"],
                               "provenance": provenance or {}})


def load_setting(path: Path, signature: dict, with_X: bool = True) -> dict | None:
    if not cache_valid(path, signature):
        return None
    meta = json.loads(sidecar(path).read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if with_X or k != "X"}
    data["names"] = data["names"].tolist()
    data.update(L=meta["L"], D_seq=meta["D_seq"], build_seconds=meta["build_seconds"], shape=meta["shape"],
                provenance=meta.get("provenance", {}), from_cache=True)
    return data


def save_sequences(path: Path, data: dict, signature: dict, seq_key: str, seq_names: list[str],
                   provenance: dict | None = None) -> None:
    names = data["names"]
    frame_age = data["X"][:, names.index(FRAME_AGE)] if FRAME_AGE in names else np.zeros(len(data["y"]), np.float32)
    atomic_savez(path, seq=data["seq"], frame_age=np.asarray(frame_age, dtype=np.float32), y=data["y"],
                 match_id=data["row_match_id"], engage_ts=data["row_t_start_ts"], ref_ordinal=data["row_ref_ordinal"],
                 patch=data["row_patch"], seq_names=np.array(seq_names, dtype=str))
    write_json(sidecar(path), {"signature": canonical(signature), "L": data["L"], "D_seq": data["D_seq"],
                               "seq_key": seq_key, "shape": list(data["seq"].shape), "provenance": provenance or {}})


def save_index(path: Path, refs, signature: dict, seconds: float, provenance: dict | None = None) -> None:
    atomic_savez(path, **refs_to_arrays(refs))
    write_json(sidecar(path), {"signature": canonical(signature), "n_refs": len(refs), "seconds": seconds,
                               "provenance": provenance or {}})


def load_index(path: Path, signature: dict):
    if not cache_valid(path, signature):
        return None
    with np.load(path, allow_pickle=False) as z:
        return refs_from_arrays({k: z[k] for k in z.files})


# ------------------------------------------------------------------ proofs

def matrix_sha1(X: np.ndarray, cols=None, chunk: int = 4096) -> str:
    """SHA-1 of the C-order bytes of X[:, cols], streamed in row chunks so no full copy is made."""
    n_cols = X.shape[1] if cols is None else len(cols)
    h = hashlib.sha1(f"{np.dtype(X.dtype).str}({X.shape[0]}, {n_cols})".encode("utf-8"))
    for i in range(0, X.shape[0], chunk):
        part = X[i:i + chunk] if cols is None else X[i:i + chunk][:, cols]
        h.update(memoryview(np.ascontiguousarray(part)).cast("B"))
    return h.hexdigest()


def identical_pairs(hashes: dict[str, str]) -> list[list[str]]:
    names = sorted(hashes)
    return [[a, b] for i, a in enumerate(names) for b in names[i + 1:] if hashes[a] == hashes[b]]


def column_differences(X: np.ndarray, X_ref: np.ndarray, cols=None, chunk: int = 8192) -> dict:
    """Columns / rows where X[:, cols] differs from X_ref[:, cols] (NaN equals NaN), chunked."""
    if X.shape != X_ref.shape:
        return {"same_shape": False}
    changed = np.zeros(X.shape[1] if cols is None else len(cols), dtype=bool)
    rows = 0
    max_abs = 0.0
    for i in range(0, X.shape[0], chunk):
        a, b = X[i:i + chunk], X_ref[i:i + chunk]
        if cols is not None:
            a, b = a[:, cols], b[:, cols]
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        changed |= diff.any(axis=0)
        rows += int(diff.any(axis=1).sum())
        if diff.any():
            gap = np.abs(a[diff].astype(np.float64) - b[diff].astype(np.float64))
            gap = gap[np.isfinite(gap)]
            if gap.size:
                max_abs = max(max_abs, float(gap.max()))
    return {"same_shape": True, "n_columns_differing": int(changed.sum()), "n_rows_differing": rows,
            "share_rows_differing": rows / max(1, X.shape[0]), "max_abs_difference": max_abs}


class InputsUnchangedError(RuntimeError):
    """Two settings produced the same telemetry matrix: the setting never reached the features."""


class RowIdentityError(RuntimeError):
    """The settings disagree about which engagements exist or what their labels are."""


def telemetry_columns(names: list[str]) -> np.ndarray:
    """Every column except frame_age_s, which is measured at the cutoff and cannot depend on the window."""
    return np.array([j for j, n in enumerate(names) if n != FRAME_AGE], dtype=np.int64)


def assert_inputs_changed(hashes: dict[str, str]) -> list:
    """Raise InputsUnchangedError unless every setting's telemetry SHA-1 is distinct; returns [] when they are."""
    if len(hashes) < 2:
        raise InputsUnchangedError(f"need at least two settings to show that the inputs changed, got {sorted(hashes)}")
    same = identical_pairs(hashes)
    if same:
        raise InputsUnchangedError(f"identical telemetry matrices for {same}: the setting did not reach the features")
    return same


def common_rows(key_lists: dict[str, list]) -> tuple[list, dict[str, np.ndarray]]:
    """Sorted keys present in every setting, and each setting's row index for them."""
    names = list(key_lists)
    common = sorted(set.intersection(*(set(k) for k in key_lists.values()))) if names else []
    take = {}
    for name in names:
        pos = {k: i for i, k in enumerate(key_lists[name])}
        if len(pos) != len(key_lists[name]):
            raise RowIdentityError(f"{name}: duplicate row keys ({len(key_lists[name]) - len(pos)})")
        take[name] = np.fromiter((pos[k] for k in common), dtype=np.int64, count=len(common))
    return common, take


def align_settings(key_lists: dict[str, list], labels: dict[str, np.ndarray], reference: str):
    """Common rows, each setting's row index for them, the reference labels and per-setting label mismatches.

    Raises RowIdentityError when no engagement is common to every setting or any label differs on a common row."""
    common, take = common_rows(key_lists)
    if not common:
        raise RowIdentityError("no engagement is common to every setting")
    y = np.asarray(labels[reference])[take[reference]].astype(np.int8)
    mismatch = {n: int((np.asarray(labels[n])[take[n]].astype(np.int8) != y).sum()) for n in key_lists}
    if any(mismatch.values()):
        raise RowIdentityError(f"labels differ across settings on common rows: {mismatch}")
    return common, take, y, mismatch


def frames_in_window(frame_age_s: np.ndarray, ctx_sec: int, frame_ms: int = 60000) -> dict:
    """Distinct timeline frames the window [cutoff - ctx, cutoff) can show: the frame held at the
    window start plus every later frame, assuming a frame every frame_ms (approximation)."""
    age = np.asarray(frame_age_s, dtype=np.float64) * 1000.0
    ctx = float(ctx_sec) * 1000.0
    n = np.where(age < ctx, np.ceil((ctx - age) / float(frame_ms)), 0.0) + 1.0
    return {"mean": float(n.mean()) if n.size else float("nan"),
            "share_ge_2": float((n >= 2).mean()) if n.size else float("nan"),
            "share_ge_3": float((n >= 3).mean()) if n.size else float("nan")}


# ------------------------------------------------------------------ learners

def nonconstant_columns(X: np.ndarray, rows=None, chunk: int = 8192) -> np.ndarray:
    """merge_shards' rule, keep a column iff max > min, evaluated over `rows` only (all rows if None).

    `rows` is a boolean mask or an index array; the scan is chunked so no copy of X[rows] is made.
    NaN propagates as in X.max(axis=0) > X.min(axis=0): a column holding NaN on those rows is dropped."""
    if rows is None:
        idx = np.arange(X.shape[0])
    else:
        rows = np.asarray(rows)
        idx = np.flatnonzero(rows) if rows.dtype == bool else rows.astype(np.int64)
    lo = np.full(X.shape[1], np.inf, dtype=np.float64)
    hi = np.full(X.shape[1], -np.inf, dtype=np.float64)
    for s in range(0, len(idx), chunk):
        part = X[idx[s:s + chunk]]
        lo = np.minimum(lo, part.min(axis=0))
        hi = np.maximum(hi, part.max(axis=0))
    return np.flatnonzero(hi > lo)


def fit_patch_lgbm(X, y, tr, va, te, n_jobs: int) -> dict:
    """lgbm_paper (scripts/run_model_comparison_v33.py::run_model, commit 3fb00c3), columns filtered on 15.14 rows."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score
    started = time.time()
    tr_i, va_i, te_i = (np.flatnonzero(m) for m in (tr, va, te))
    keep = nonconstant_columns(X, tr_i)
    model = LGBMClassifier(**LGBM_PARAMS, n_jobs=int(n_jobs))
    model.fit(X[np.ix_(tr_i, keep)], y[tr_i], eval_set=[(X[np.ix_(va_i, keep)], y[va_i])], eval_metric="auc",
              callbacks=[early_stopping(PATCH_EARLY_STOPPING, verbose=False), log_evaluation(0)])
    pred_val = model.predict_proba(X[np.ix_(va_i, keep)])[:, 1]
    pred_test = model.predict_proba(X[np.ix_(te_i, keep)])[:, 1]
    return {"pred_test": pred_test.astype(np.float64), "val_auc": float(roc_auc_score(y[va_i], pred_val)),
            "test_auc": float(roc_auc_score(y[te_i], pred_test)),
            "best_iteration": int(model.best_iteration_ or LGBM_PARAMS["n_estimators"]),
            "n_features_nonconstant": int(len(keep)), "column_filter_rows": "train patch",
            "seconds": round(time.time() - started, 1)}


def oof_lgbm(X, y, groups, n_jobs: int, n_splits: int = 5) -> dict:
    """scripts/run_scale_decomposition.py::oof_predictions with an explicit thread count, columns filtered per fold on its training rows."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    started = time.time()
    oof = np.full(len(y), np.nan, dtype=np.float64)
    folds = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros((len(y), 1)), y, groups), 1):
        keep = nonconstant_columns(X, train)
        model = LGBMClassifier(**LGBM_PARAMS, n_jobs=int(n_jobs))
        X_train = X[np.ix_(train, keep)]
        model.fit(X_train, y[train])
        del X_train
        oof[test] = model.predict_proba(X[np.ix_(test, keep)])[:, 1]
        folds.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "n_features_nonconstant": int(len(keep)),
                      "auc": float(roc_auc_score(y[test], oof[test]))})
        log(f"    fold {fold}/{n_splits}: train={len(train)} test={len(test)} cols={len(keep)} auc={folds[-1]['auc']:.4f}")
    assert not np.isnan(oof).any()
    return {"pred": oof, "auc": float(roc_auc_score(y, oof)), "folds": folds, "column_filter_rows": "training folds",
            "n_features_nonconstant": [f["n_features_nonconstant"] for f in folds], "seconds": round(time.time() - started, 1)}


# ------------------------------------------------------------------ match-clustered bootstrap

def _auc_plan(y: np.ndarray, score: np.ndarray):
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] != s[:-1]
    tie = np.cumsum(new) - 1
    return order, tie, int(tie[-1]) + 1 if len(s) else 0, (np.asarray(y)[order] == 1)


def _auc_from_plan(plan, weight: np.ndarray | None) -> float:
    order, tie, n_tie, pos_sorted = plan
    w = np.ones(len(order)) if weight is None else np.asarray(weight, dtype=np.float64)[order]
    pos = np.bincount(tie, weights=w * pos_sorted, minlength=n_tie)
    neg = np.bincount(tie, weights=w * ~pos_sorted, minlength=n_tie)
    w_pos, w_neg = float(pos.sum()), float(neg.sum())
    if w_pos <= 0.0 or w_neg <= 0.0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float(np.dot(pos, below + 0.5 * neg) / (w_pos * w_neg))


def weighted_auc(y, score, weight=None) -> float:
    """Mann-Whitney AUC with non-negative row weights, ties counted one half."""
    return _auc_from_plan(_auc_plan(np.asarray(y), np.asarray(score)), weight)


def match_weights(rng: np.random.Generator, inverse: np.ndarray, n_groups: int) -> np.ndarray:
    """One cluster-bootstrap draw as row weights: each row gets its match's multiplicity."""
    counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
    return counts[inverse].astype(np.float64)


def match_bootstrap(y, preds: dict, groups, *, n_boot: int = 1000, seed: int = SEED, pairs=(), alpha: float = 0.05) -> dict:
    """Per-model AUC CIs and paired AUC differences (a - b) under one shared match draw per replicate."""
    y = np.asarray(y).astype(np.int8)
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    inverse = inverse.reshape(-1)
    n_groups = int(inverse.max()) + 1 if len(inverse) else 0
    plans = {k: _auc_plan(y, np.asarray(p, dtype=np.float64)) for k, p in preds.items()}
    point = {k: _auc_from_plan(plan, None) for k, plan in plans.items()}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in preds}
    deltas = {(a, b): [] for a, b in pairs}
    skipped = 0
    for _ in range(int(n_boot)):
        w = match_weights(rng, inverse, n_groups)
        aucs = {k: _auc_from_plan(plan, w) for k, plan in plans.items()}
        if any(np.isnan(v) for v in aucs.values()):
            skipped += 1
            continue
        for k, v in aucs.items():
            draws[k].append(v)
        for a, b in pairs:
            deltas[(a, b)].append(aucs[a] - aucs[b])
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    def interval(values):
        arr = np.asarray(values, dtype=np.float64)
        if not arr.size:
            return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
        return {"ci_lo": float(np.percentile(arr, lo)), "ci_hi": float(np.percentile(arr, hi)),
                "boot_mean": float(arr.mean()), "n_boot": int(arr.size)}

    out = {"method": "match-clustered percentile bootstrap, shared draw per replicate",
           "n_rows": int(len(y)), "n_matches": n_groups, "n_boot_requested": int(n_boot),
           "n_boot_skipped_single_class": skipped, "alpha": alpha, "seed": int(seed),
           "auc": {k: {"auc": point[k], **interval(draws[k])} for k in preds}, "paired": {}}
    for a, b in pairs:
        arr = np.asarray(deltas[(a, b)], dtype=np.float64)
        out["paired"][f"{a} - {b}"] = {"delta": point[a] - point[b], **interval(arr),
                                       "p_delta_leq_0": float((arr <= 0).mean()) if arr.size else float("nan")}
    return out


# ------------------------------------------------------------------ published-corpus cross-check

def crosscheck_published(shard_dir: Path, names: list[str], data: dict, max_rows: int) -> dict:
    """Does the reference arm reproduce corpus_shards_v33 on the engagements both contain?"""
    out: dict = {"shards": str(shard_dir)}
    names_path = shard_dir / "feature_names.json"
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not names_path.exists() or not paths:
        out["status"] = "shards absent"
        return out
    out["feature_names_equal"] = json.loads(names_path.read_text(encoding="utf-8"))["names"] == list(names)
    ours = {k[:2]: i for i, k in enumerate(row_keys(data))}   # the shards identify a row by (match, engage_ts)
    our_mids = np.array(sorted({k[0] for k in ours}), dtype=str)
    found, label_n, label_agree, best = 0, 0, 0, (0, None, None)
    pub_rows_for_our_matches = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            groups = z["groups"].astype(str)
            mask = np.isin(groups, our_mids)
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            ts = z["engage_ts"][idx].astype(np.int64)
            pub_rows_for_our_matches += int(len(idx))
            hits = [(int(j), ours[(g, int(t))]) for j, g, t in zip(idx.tolist(), groups[idx].tolist(), ts.tolist())
                    if (g, int(t)) in ours]
            if not hits:
                continue
            found += len(hits)
            if "y_market_event" in z.files:
                pub_y = z["y_market_event"][[j for j, _ in hits]]
                ok = pub_y >= 0
                label_n += int(ok.sum())
                label_agree += int((pub_y[ok] == data["y"][[i for _, i in hits]][ok]).sum())
            if len(hits) > best[0]:
                best = (len(hits), path, hits)
    out.update(published_rows_found=found, our_rows=len(ours), published_rows_for_our_matches=pub_rows_for_our_matches,
               label_rows_compared=label_n, label_agreement=(label_agree / label_n) if label_n else float("nan"))
    if best[1] is not None and out["feature_names_equal"]:
        hits = best[2][: int(max_rows)]
        with np.load(best[1], allow_pickle=False) as z:
            pub_X = z["X"][[j for j, _ in hits]]
        ours_X = data["X"][[i for _, i in hits]]
        out["feature_check"] = {"shard": best[1].name, "rows": len(hits), **column_differences(ours_X, pub_X),
                                "share_rows_allclose_1e-5": float(np.mean([
                                    np.allclose(a, b, atol=1e-5, equal_nan=True) for a, b in zip(ours_X, pub_X)]))}
    return out


# ------------------------------------------------------------------ main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=20000, help="matches sampled from the cache (0 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--settings", default=DEFAULT_SETTINGS, help="comma list of ctx_sec:bin_ms")
    ap.add_argument("--reference", default=REFERENCE,
                    help="setting that builds the fight index and that every other setting is paired against")
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--protocols", default="patch,oof", help="subset of patch,oof")
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=8, help="LightGBM threads (the paper used -1)")
    ap.add_argument("--index-workers", type=int, default=1,
                    help="build_fight_index processes; the setting reaches them via LOL_CFG_PRESET / LOL_CFG_OVERRIDES")
    ap.add_argument("--verify-index", choices=("all", "none"), default="all",
                    help="rebuild the detector under every non-reference setting and compare with the reference index")
    ap.add_argument("--verify-index-matches", type=int, default=0,
                    help="verify on the first N sampled matches only (0 = every sampled match)")
    ap.add_argument("--allow-index-drift", action="store_true",
                    help="continue on the common rows even if the detector output depends on the setting")
    ap.add_argument("--leak-probe-rows", type=int, default=40,
                    help="engagements per setting rebuilt with post-cutoff data rewritten (0 = skip)")
    ap.add_argument("--save-seq", default="30:5000,60:5000,120:5000",
                    help="settings whose per-bin sequences are cached for run_sequence_baseline_v33.py ('' = none)")
    ap.add_argument("--crosscheck-shards", default="D:/LOL_Project/fusion_2615/corpus_shards_v33",
                    help="published v3.3 shards the reference arm is compared with ('' = skip)")
    ap.add_argument("--crosscheck-max-rows", type=int, default=5000)
    ap.add_argument("--no-cache", action="store_true", help="rebuild features and refit even when a valid cache exists")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--output", type=Path, default=None, help="summary JSON (default <out-dir>/window_sweep_v33.json)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    settings = parse_settings(args.settings)
    reference = parse_settings(args.reference)[0]
    if reference not in settings:
        settings.insert(0, reference)
    if len(settings) < 2:
        raise SystemExit("need at least two settings: the proof that inputs changed compares settings")
    save_seq = [s for s in parse_settings(args.save_seq) if s in settings]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    if not set(protocols) <= {"patch", "oof"}:
        raise SystemExit(f"unknown protocol in {protocols}")
    names = {s: setting_name(*s) for s in settings}
    ref_name = names[reference]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or out_dir / "window_sweep_v33.json"

    configure(*reference, index_workers=args.index_workers)
    source = source_digest()
    script_sha1 = file_sha1(Path(__file__))
    git = git_info()
    provenance = {"git_commit": git["commit"], "dirty_tracked_files": git["dirty_tracked_files"]}
    from core.config import CACHE_DIR
    from core.presets import PRESETS
    mids = sample_match_ids(CACHE_DIR, args.n_matches, args.seed)
    match_sha1 = sha1_lines(mids)
    log(f"[{ITEM}] preset {PRESET} cache {CACHE_DIR} matches {len(mids)} (sha1 {match_sha1[:12]}) "
        f"commit {git['commit'][:10]} sources {source[:12]}")
    # Caches are keyed by the code that defines them (feature sources + this script), not by the git commit:
    # the shared checkout receives unrelated commits during a multi-hour run, and a resume must not rebuild
    # for those.  The commit each cache was built at is kept in its sidecar and reported per setting.
    base_sig = {"item": ITEM, "preset": PRESET, "preset_values": PRESETS[PRESET], "label": LABEL_KEY,
                "tie_policy": TIE_POLICY, "match_list_sha1": match_sha1, "n_matches": len(mids),
                "source_sha1": source, "script_sha1": script_sha1}

    # 1. the fight index, built once under the reference setting
    index_path = out_dir / "fight_index.npz"
    index_sig = {**base_sig, "kind": "fight_index", "setting": ref_name}
    refs = None if args.no_cache else load_index(index_path, index_sig)
    index_seconds = None
    if refs is None:
        t = time.time()
        refs = build_index(mids, *reference, args.index_workers)
        index_seconds = round(time.time() - t, 1)
        save_index(index_path, refs, index_sig, index_seconds, provenance)
    idx_sha1 = index_sha1(refs)
    log(f"fight index: {len(refs)} refs (sha1 {idx_sha1[:12]}){'' if index_seconds is None else f' in {index_seconds}s'}")

    # 2. engagement identity: rebuild the detector under every other setting
    verify: dict = {}
    if args.verify_index == "all":
        subset = mids[: args.verify_index_matches] if args.verify_index_matches else mids
        subset_set = set(subset)
        ref_subset = refs if subset is mids else [r for r in refs if r.match_id in subset_set]
        verify_path = out_dir / "index_verify.json"
        stored = json.loads(verify_path.read_text(encoding="utf-8")) if verify_path.exists() else {}
        for s in settings:
            if s == reference:
                continue
            sig = canonical({**base_sig, "kind": "index_verify", "setting": names[s], "reference_index_sha1": idx_sha1,
                             "verify_matches_sha1": sha1_lines(subset)})
            entry = stored.get(names[s])
            if entry and entry.get("signature") == sig and not args.no_cache:
                result = entry["result"]
            else:
                t = time.time()
                result = compare_indexes(ref_subset, build_index(subset, *s, args.index_workers))
                result.update(n_matches=len(subset), seconds=round(time.time() - t, 1))
                stored[names[s]] = {"signature": sig, "result": result}
                write_json(verify_path, stored)
            verify[names[s]] = result
            log(f"index under {names[s]}: identical={result['identical']} "
                f"(only reference {result['only_reference']}, only setting {result['only_setting']})")
            if not result["identical"] and not args.allow_index_drift:
                raise SystemExit(f"engagement identity depends on {names[s]}; see {verify_path} "
                                 "(--allow-index-drift evaluates on the common rows anyway)")

    # 3. features per setting (cached), each followed by the leak probe under that setting
    from train.baseline import _choose_tab_seq_key_and_names
    seq_key, seq_names = _choose_tab_seq_key_and_names(args.feature_set, {"x_seq": np.zeros(1), "extra_seq": np.zeros(1)})
    meta: dict = {}
    probes: dict = {}
    for s in settings:
        name = names[s]
        path = out_dir / f"setting_{name}.npz"
        sig = {**base_sig, "kind": "setting", "setting": name, "ctx_sec": s[0], "bin_ms": s[1],
               "feature_set": args.feature_set, "index_sha1": idx_sha1, "row_key": "match_id|t_start_ts|ref_ordinal"}
        seq_path = out_dir / f"seq_{name}.npz"
        seq_sig = {**sig, "kind": "sequence", "seq_key": seq_key}
        need_seq = s in save_seq and (args.no_cache or not cache_valid(seq_path, seq_sig))
        data = None if (args.no_cache or need_seq) else load_setting(path, sig, with_X=False)
        if data is None:
            configure(*s, index_workers=args.index_workers)
            log(f"[{name}] building L={n_bins(*s)} for {len(refs)} refs ...")
            built = build_setting(refs, args.feature_set, keep_seq=s in save_seq)
            if built["L"] != n_bins(*s):
                raise SystemExit(f"[{name}] builder produced L={built['L']}, expected {n_bins(*s)}: "
                                 "the setting did not reach the pipeline")
            save_setting(path, built, sig, provenance)
            if "seq" in built:
                save_sequences(seq_path, built, seq_sig, seq_key, seq_names, provenance)
            log(f"[{name}] X {built['X'].shape} L={built['L']} D_seq={built['D_seq']} in {built['build_seconds']}s")
            built.pop("seq", None)
            built.pop("X")
            data = {**built, "shape": [len(built["y"]), len(built["names"])], "provenance": provenance, "from_cache": False}
        else:
            log(f"[{name}] cached: shape {data['shape']} L={data['L']}")
        meta[name] = {"setting": s, "path": path, "sig": sig, "keys": row_keys(data), "y": data["y"],
                      "patch": data["row_patch"], "L": int(data["L"]), "D_seq": int(data["D_seq"]),
                      "shape": list(data["shape"]), "names": data["names"], "build_seconds": data["build_seconds"],
                      "from_cache": bool(data.get("from_cache", False)), "provenance": data.get("provenance", {}),
                      "cluster": (data["row_det_cluster_blue"], data["row_det_cluster_red"])}
        if args.leak_probe_rows > 0:
            configure(*s, index_workers=args.index_workers)
            probes[name] = leak_probe(refs, args.leak_probe_rows, args.seed, args.feature_set)
            log(f"[{name}] leak probe: {probes[name]['sequence_identical_after_future_perturbation']}/"
                f"{probes[name]['probed']} identical after rewriting post-cutoff data, label changed in "
                f"{probes[name]['label_changed_by_future_perturbation']}, positive control "
                f"{probes[name]['positive_control_sequence_changed']}/{probes[name]['positive_control_evaluated']}")
            if not probes[name]["passed"]:
                write_json(out_dir / "leak_probe_failure.json", probes)
                raise SystemExit(f"[{name}] leak probe failed: {probes[name]}")
    configure(*reference, index_workers=args.index_workers)

    # 4. common rows; labels must not depend on the setting
    common, take, y, label_mismatch = align_settings({n: m["keys"] for n, m in meta.items()},
                                                     {n: m["y"] for n, m in meta.items()}, ref_name)
    groups = np.array([k[0] for k in common], dtype=str)
    engage_ts = np.array([k[1] for k in common], dtype=np.int64)
    ref_ordinal = np.array([k[2] for k in common], dtype=np.int64)
    patch = meta[ref_name]["patch"][take[ref_name]].astype(str)
    blue, red = (c[take[ref_name]] for c in meta[ref_name]["cluster"])
    atomic_savez(out_dir / "common_rows.npz", match_id=groups, engage_ts=engage_ts, ref_ordinal=ref_ordinal, y=y,
                 patch=patch, cluster_blue=blue, cluster_red=red)
    common_sha1 = common_rows_sha1(groups, engage_ts, y)
    log(f"common rows {len(common)} of {[len(m['keys']) for m in meta.values()]}, "
        f"matches {len(set(groups.tolist()))}, positive {y.mean():.4f}")

    # 5. prove the inputs changed, before any fit
    def load_common_X(name):
        with np.load(meta[name]["path"], allow_pickle=False) as z:
            return z["X"][take[name]]

    ref_names_list = meta[ref_name]["names"]
    telemetry = telemetry_columns(ref_names_list)
    fa_col = ref_names_list.index(FRAME_AGE) if FRAME_AGE in ref_names_list else None
    X_ref = load_common_X(ref_name)
    per_setting: dict = {}
    hashes: dict = {}
    for name, m in meta.items():
        X = X_ref if name == ref_name else load_common_X(name)
        if m["names"] != ref_names_list:
            raise SystemExit(f"[{name}] feature names differ from the reference setting")
        hashes[name] = matrix_sha1(X, telemetry)
        per_setting[name] = {
            "ctx_sec": m["setting"][0], "bin_ms": m["setting"][1], "L": m["L"], "D_seq": m["D_seq"],
            "feature_matrix_shape_built": m["shape"], "feature_matrix_shape_common": list(X.shape),
            "n_rows_built": len(m["keys"]), "telemetry_sha1_common": hashes[name],
            "full_matrix_sha1_common": matrix_sha1(X), "build_seconds": m["build_seconds"],
            "from_cache": m["from_cache"], "built_git_commit": m["provenance"].get("git_commit"), "cache": str(m["path"]),
            "vs_reference_telemetry": column_differences(X, X_ref, telemetry) if name != ref_name else None,
            "frame_age_identical_to_reference": bool(np.array_equal(X[:, fa_col], X_ref[:, fa_col])) if fa_col is not None else None,
            "frames_in_window": frames_in_window(X[:, fa_col], m["setting"][0]) if fa_col is not None else None,
            "sequence_cache": str(out_dir / f"seq_{name}.npz") if m["setting"] in save_seq else None,
            "leak_probe": probes.get(name),
        }
        del X
    del X_ref
    try:
        same = assert_inputs_changed(hashes)
    except InputsUnchangedError:
        write_json(out_dir / "inputs_unchanged_failure.json", {"settings": per_setting, "telemetry_sha1": hashes})
        raise
    log("inputs changed: telemetry SHA-1 distinct across " + ", ".join(f"{n}={h[:10]}" for n, h in hashes.items()))

    # 6. learners
    tr, va, te = patch == args.train_patch, patch == args.val_patch, patch == args.test_patch
    split = {"kind": "patch holdout", "train": args.train_patch, "val": args.val_patch, "test": args.test_patch,
             "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
             "matches": {k: int(len(set(groups[m].tolist()))) for k, m in (("train", tr), ("val", va), ("test", te))},
             "rows_outside_split": int((~(tr | va | te)).sum())}
    if "patch" in protocols and not (tr.any() and va.any() and te.any()):
        raise SystemExit(f"empty patch split; patches present: {sorted(set(patch.tolist()))}")
    pred_store = {"match_id": groups, "engage_ts": engage_ts, "y": y, "patch": patch,
                  "patch_test_index": np.flatnonzero(te)}
    results: dict = {}
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    for protocol in protocols:
        preds, fits = {}, {}
        for name, m in meta.items():
            cache_path = out_dir / f"pred_{protocol}_{name}.npz"
            sig = {**m["sig"], "kind": f"pred_{protocol}", "common_rows_sha1": common_sha1, "lgbm": LGBM_PARAMS,
                   "early_stopping": PATCH_EARLY_STOPPING if protocol == "patch" else None,
                   "split": split if protocol == "patch" else "GroupKFold(5)",
                   "column_filter": "nonconstant on training rows only"}
            if cache_valid(cache_path, sig) and not args.no_cache:
                with np.load(cache_path, allow_pickle=False) as z:
                    preds[name] = z["pred"]
                fits[name] = json.loads(sidecar(cache_path).read_text(encoding="utf-8"))["fit"]
                log(f"[{protocol}] {name}: cached")
                continue
            X = load_common_X(name)
            log(f"[{protocol}] {name}: {X.shape[0]} rows x {X.shape[1]} columns (constant columns dropped on training rows)")
            if protocol == "patch":
                fit = fit_patch_lgbm(X, y, tr, va, te, args.n_jobs)
                preds[name] = fit.pop("pred_test")
                log(f"    val {fit['val_auc']:.4f} test {fit['test_auc']:.4f} best_iter {fit['best_iteration']} "
                    f"cols {fit['n_features_nonconstant']} ({fit['seconds']}s)")
            else:
                fit = oof_lgbm(X, y, groups, args.n_jobs)
                preds[name] = fit.pop("pred")
                log(f"    oof auc {fit['auc']:.4f} ({fit['seconds']}s)")
            del X
            fits[name] = fit
            atomic_savez(cache_path, pred=preds[name])
            write_json(sidecar(cache_path), {"signature": canonical(sig), "fit": fit})
        eval_mask = te if protocol == "patch" else np.ones(len(y), dtype=bool)
        pairs = [(n, ref_name) for n in meta if n != ref_name]
        boot = match_bootstrap(y[eval_mask], preds, groups[eval_mask], n_boot=args.n_boot, seed=args.seed, pairs=pairs)
        results[protocol] = {"evaluated_rows": int(eval_mask.sum()),
                             "evaluated_matches": int(len(set(groups[eval_mask].tolist()))),
                             "fits": fits, "bootstrap": boot}
        for name, p in preds.items():
            pred_store[f"{protocol}__pred__{name}"] = p
        for name, v in boot["auc"].items():
            log(f"[{protocol}] {name}: AUC {v['auc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
        for key, v in boot["paired"].items():
            log(f"[{protocol}] {key}: {v['delta']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    # 7. does the reference arm reproduce the published corpus?
    crosscheck = None
    if args.crosscheck_shards:
        ref_data = load_setting(meta[ref_name]["path"], meta[ref_name]["sig"])
        crosscheck = crosscheck_published(Path(args.crosscheck_shards), ref_names_list, ref_data, args.crosscheck_max_rows)
        del ref_data
        log(f"published-corpus cross-check: {json.dumps(_jsonable(crosscheck))[:400]}")

    summary = {
        "item": ITEM, "script": "scripts/run_window_sweep_v33.py", "script_sha1": script_sha1,
        "git": git, "source_sha1": source, "preset": PRESET, "preset_values": PRESETS[PRESET],
        "label_key": LABEL_KEY, "tie_policy": TIE_POLICY, "feature_set": args.feature_set, "seed": int(args.seed),
        "cfg": cfg_snapshot(), "n_matches_requested": int(args.n_matches), "n_matches_sampled": len(mids),
        "match_list_sha1": match_sha1, "reference": ref_name, "settings_order": [names[s] for s in settings],
        "fight_index": {"n_refs": len(refs), "sha1": idx_sha1, "cache": str(index_path), "seconds": index_seconds,
                        "verification": verify or None},
        "common_rows": {"n_rows": int(len(y)), "n_matches": int(len(set(groups.tolist()))), "positive_rate": float(y.mean()),
                        "sha1": common_sha1, "label_mismatch_vs_reference": label_mismatch,
                        "n_rows_built": {n: len(m["keys"]) for n, m in meta.items()},
                        "patch_counts": {str(p): int(c) for p, c in zip(*np.unique(patch, return_counts=True))}},
        "split": split, "settings": per_setting,
        "inputs_changed": {"telemetry_columns": int(len(telemetry)), "identical_pairs": same, "passed": not same},
        "leak_probe": {"rows_per_setting": int(args.leak_probe_rows), "scope": LEAK_PROBE_SCOPE,
                       "passed": all(p["passed"] for p in probes.values()) if probes else None},
        "lgbm_params": LGBM_PARAMS, "patch_early_stopping_rounds": PATCH_EARLY_STOPPING, "n_jobs": int(args.n_jobs),
        "n_boot": int(args.n_boot), "protocols": results, "published_crosscheck": crosscheck,
        "sequence_cache": {"seq_key": seq_key, "n_seq_features": len(seq_names), "settings": [names[s] for s in save_seq]},
        "references": ["Ke et al. 2017, LightGBM, NeurIPS", "Efron & Tibshirani 1993, An Introduction to the Bootstrap, ch. 13",
                       "Field & Welsh 2007, Bootstrapping clustered data, JRSS-B 69(3) sec. 2.1",
                       "Hanley & McNeil 1982, Radiology 143(1) eq. 1"],
        "deviations": DEVIATIONS, "outputs": {"preds": str(output.with_suffix(".preds.npz")),
                                              "common_rows": str(out_dir / "common_rows.npz")},
        "wall_clock_s": round(time.time() - started, 1),
    }
    np.savez_compressed(output.with_suffix(".preds.npz"), **pred_store)
    write_json(output, summary)
    log(f"wrote {output} in {summary['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
