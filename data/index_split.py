from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import random
import re
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from core.common import Any, Dict, List, Optional, Tuple, np, dataclass, field
from core.config import CACHE_DIR, META_DIR, cfg

from core.fight_types import FightRef, PruneSpec
from data.cache_io import load_match_cache
from gameplay.pipeline import build_ms_sequence
from gameplay.fights import detect_fights, normalize_patch
from core.diagnostics import _maybe_dump_fights_for_match
from gameplay.features import build_sequence_features, get_xseq_feature_names, get_extra_feature_names, prune_correlated_columns
from core.contract import TIME_CONTRACT


def _stable_int_hash(s: str) -> int:
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


_FIGHT_INDEX_CACHE_CFG_KEYS: Tuple[str, ...] = (
    # Core detector choices
    "FIGHT_DETECTOR",
    "FIGHT_DETECT_ALGO",
    "STANDOFF_RADIUS",
    "STANDOFF_MIN_PAIRS",
    "ENGAGE_MIN_DIST_DROP",
    "ENGAGE_MIN_PAIR_GAIN",
    "VERIFY_KILL_IN_HORIZON",
    "USE_KILL_ANCHOR",
    "KILL_ANCHOR_PRE_SEC",
    "KILL_ANCHOR_COOLDOWN_SEC",
    # Temporal windowing
    "START_OFFSET_MIN",
    "FIGHT_CONTEXT_MIN",
    "FIGHT_MIN_GAP_MIN",
    "FIGHT_MIN_GAP_MS",
    "FIGHT_HORIZON_SEC",
    # Backtrack / merge behavior
    "USE_BACKTRACK",
    "BACKTRACK_MAX_MS",
    "BACKTRACK_MIN_MS",
    "BACKTRACK_MIN_PAIRS",
    "CONTINUOUS_FIGHT_MERGE",
    "CONTINUOUS_FIGHT_MAX_GAP_MS",
    "CONTINUOUS_FIGHT_MERGE_RADIUS",
    # Spatial/cluster guards
    "DETECT_STEP_MS",
    "REQUIRE_ALIVE_PER_TEAM",
    "REQUIRE_ENGAGED_PER_TEAM",
    "REQUIRE_LCC_TOTAL",
    "REQUIRE_LCC_PER_TEAM",
    "CLUSTER_MAX_DIAMETER",
    # Per-match sampling controls
    "MAX_FIGHTS_PER_MATCH",
    "FIGHT_SUBSAMPLE_STRATEGY",
    "FIGHT_SUBSAMPLE_SEED_OFFSET",
    # Versioning / split guards
    "FEATURE_VERSION",
    "PATCH_ALLOWLIST",
    "PATCH_LEVEL",
)


def _jsonable_cfg_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple, set)):
        return [_jsonable_cfg_value(x) for x in list(v)]
    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for k in sorted(v.keys(), key=lambda x: str(x)):
            out[str(k)] = _jsonable_cfg_value(v[k])
        return out
    return str(v)


def _fight_index_cfg_signature() -> Dict[str, Any]:
    sig: Dict[str, Any] = {}
    for k in _FIGHT_INDEX_CACHE_CFG_KEYS:
        sig[k] = _jsonable_cfg_value(getattr(cfg, k, None))
    return sig


def _fight_index_cache_plan(mids: List[str]) -> Tuple[Optional[Path], Dict[str, Any], str]:
    if not bool(getattr(cfg, "FIGHT_INDEX_CACHE_ENABLED", True)):
        return None, {}, ""
    # When explicit dumps are enabled, keep detector execution path as-is.
    if bool(getattr(cfg, "DUMP_FIGHTS", False)):
        return None, {}, ""

    root = META_DIR / str(getattr(cfg, "FIGHT_INDEX_CACHE_DIRNAME", "fight_index_cache"))
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None, {}, ""

    h_mids = hashlib.blake2b(digest_size=12)
    for mid in mids:
        h_mids.update(str(mid).encode("utf-8"))
        h_mids.update(b"\0")
    mids_digest = h_mids.hexdigest()

    sig = _fight_index_cfg_signature()
    sig["n_mids"] = int(len(mids))
    sig["mids_digest"] = mids_digest

    sig_blob = json.dumps(sig, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    cache_key = hashlib.blake2b(sig_blob.encode("utf-8"), digest_size=16).hexdigest()
    return root / f"{cache_key}.json.gz", sig, cache_key


def _load_cached_fight_index(path: Path, cache_key: str) -> Optional[List[FightRef]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        if str(obj.get("cache_key", "")) != str(cache_key):
            return None
        rows = obj.get("refs", None)
        if not isinstance(rows, list):
            return None

        refs: List[FightRef] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                refs.append(
                    FightRef(
                        match_id=str(r.get("match_id", "")),
                        patch=str(r.get("patch", "0.0")),
                        t_start=int(r.get("t_start", 0)),
                        t_start_ts=int(r.get("t_start_ts", -1)),
                        label_end_ts=int(r.get("label_end_ts", -1)),
                    )
                )
            except Exception:
                continue
        if not refs:
            return None
        return refs
    except Exception:
        return None


def _save_cached_fight_index(path: Path, cache_key: str, cfg_sig: Dict[str, Any], refs: List[FightRef]) -> None:
    try:
        rows = [
            {
                "match_id": str(r.match_id),
                "patch": str(r.patch),
                "t_start": int(r.t_start),
                "t_start_ts": int(getattr(r, "t_start_ts", -1)),
                "label_end_ts": int(getattr(r, "label_end_ts", -1)),
            }
            for r in refs
        ]
        payload = {
            "cache_key": str(cache_key),
            "created_at_unix": int(time.time()),
            "n_refs": int(len(rows)),
            "cfg_signature": cfg_sig,
            "refs": rows,
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        # Best-effort cache write only; never break index build.
        return


def _subsample_fights_in_match(fights, *args, **kwargs):
    """
    fights list -> subsample at most K per match, deterministic by (seed + hash(match_id))
    """
    if fights is None:
        return []
    fights = list(fights)
    if len(fights) <= 1:
        return fights

    match_id = kwargs.get("match_id", None)
    max_keep = kwargs.get("max_keep", None) or kwargs.get("max_per_match", None) or kwargs.get("k", None)
    seed = int(kwargs.get("seed", 0) or 0)
    method = kwargs.get("method", kwargs.get("strategy", "uniform"))
    time_key = kwargs.get("time_key", None)

    ints = []
    for a in args:
        if isinstance(a, str) and match_id is None:
            match_id = a
        elif isinstance(a, (int, np.integer)):
            ints.append(int(a))
        elif isinstance(a, float) and float(a).is_integer():
            ints.append(int(a))
    if max_keep is None and len(ints) >= 1:
        max_keep = ints[0]
    if len(ints) >= 2 and ("seed" not in kwargs):
        seed = ints[1]

    if max_keep is None:
        _raw = getattr(cfg, "MAX_FIGHTS_PER_MATCH", None)
        max_keep = int(_raw) if _raw is not None else None
    if max_keep is None or int(max_keep) <= 0 or len(fights) <= int(max_keep):
        return fights

    def _get_t(x):
        if isinstance(x, dict):
            # ✅ 수정: engage_ts 우선
            if "engage_ts" in x:
                return x.get("engage_ts", 0)
            if time_key and time_key in x:
                return x.get(time_key, 0)
            for k in ("t_engage_ts", "t_engage", "t_start", "t_ms", "start", "t", "time"):
                if k in x:
                    return x.get(k, 0)
            return 0
        if time_key and hasattr(x, time_key):
            return getattr(x, time_key)
        for k in ("engage_ts", "t_engage_ts", "t_engage", "t_start", "t_ms", "start", "t", "time"):
            if hasattr(x, k):
                return getattr(x, k)
        return 0

    fights = sorted(fights, key=_get_t)

    base = int(seed)
    mix = base + (_stable_int_hash(str(match_id)) if match_id is not None else 0)
    rng = np.random.RandomState(mix)

    method = str(method).lower().strip()
    max_keep = int(max_keep)

    if method in ("random", "rand"):
        idx = rng.choice(len(fights), size=max_keep, replace=False)
        idx = sorted([int(i) for i in idx])
        return [fights[i] for i in idx]

    n = len(fights)
    if max_keep == 1:
        return [fights[n // 2]]
    lin = np.linspace(0, n - 1, num=max_keep)
    idx = sorted(set(int(round(x)) for x in lin.tolist()))
    while len(idx) < max_keep:
        idx = sorted(set(idx + [int(rng.randint(0, n))]))
    if len(idx) > max_keep:
        idx = idx[:max_keep]
    return [fights[i] for i in idx]


def _fight_to_ref_row(
    fight: Dict[str, Any],
    minute_ts: np.ndarray,
    match_id: str,
    patch: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(fight, dict):
        return None

    engage_ts = fight.get("engage_ts", None)
    if engage_ts is None:
        engage_ts = fight.get("t_engage_ts", None)
    label_end_raw = fight.get("horizon_end_ts", None)

    t_raw = fight.get("t_engage", None)
    if t_raw is None:
        t_raw = fight.get("t_start", None)
    if t_raw is None:
        t_raw = fight.get("t_ms", None)

    if engage_ts is None and t_raw is None:
        return None

    try:
        if engage_ts is not None and len(minute_ts) > 0:
            t_idx = int(np.searchsorted(minute_ts, engage_ts, side="right") - 1)
            t_idx = int(np.clip(t_idx, 0, len(minute_ts) - 1))
            t_start_ts = int(engage_ts)
        else:
            t_idx = int(TIME_CONTRACT.coerce_t_start_minute_idx(minute_ts, t_raw))
            if len(minute_ts) > 0:
                t_idx = int(np.clip(t_idx, 0, len(minute_ts) - 1))
                t_start_ts = int(minute_ts[t_idx])
            else:
                t_start_ts = -1
    except Exception:
        try:
            t_idx = int(t_raw) if t_raw is not None else 0
            t_start_ts = int(engage_ts) if engage_ts is not None else -1
        except Exception:
            return None

    try:
        label_end_ts = int(label_end_raw) if label_end_raw is not None else -1
    except Exception:
        label_end_ts = -1

    if t_start_ts >= 0 and label_end_ts >= 0 and label_end_ts <= t_start_ts:
        label_end_ts = -1
    if t_idx < 0:
        return None

    return {
        "match_id": str(match_id),
        "patch": str(patch),
        "t_start": int(t_idx),
        "t_start_ts": int(t_start_ts),
        "label_end_ts": int(label_end_ts),
    }


def _build_fight_rows_for_match(
    mid: str,
    allow_patches: Optional[Tuple[str, ...]],
    max_fights_per_match: Optional[int],
    subsample_strategy: str,
    subsample_seed: int,
    tag: Optional[str],
    do_dump: bool = False,
) -> List[Dict[str, Any]]:
    """Build FightRef rows for one match id (worker-safe, picklable output)."""
    try:
        pack = load_match_cache(mid)
        if not pack:
            return []

        meta = pack.get("meta", {})
        if not isinstance(meta, dict):
            return []
        patch = str(meta.get("patch", meta.get("patch_full", "0.0")))
        allow_set = set(allow_patches) if allow_patches is not None else None
        if allow_set is not None and patch not in allow_set:
            return []

        team_map = meta.get("team_map", {})
        fights = detect_fights(pack, team_map)
        fights_raw = list(fights)
        fights = _subsample_fights_in_match(
            fights=fights,
            k=max_fights_per_match,
            strategy=subsample_strategy,
            seed=subsample_seed,
            match_id=mid,
        )
        fights_kept = list(fights)

        if do_dump:
            _maybe_dump_fights_for_match(
                match_id=mid,
                patch=patch,
                pack=pack,
                fights_raw=fights_raw,
                fights_kept=fights_kept,
                tag=tag,
            )

        minute_ts = np.asarray(pack.get("minute_ts", np.array([])), dtype=np.int64)
        match_id = str(meta.get("match_id", mid))

        rows: List[Dict[str, Any]] = []
        for fight in fights:
            row = _fight_to_ref_row(fight, minute_ts, match_id=match_id, patch=patch)
            if row is not None:
                rows.append(row)
        return rows
    except Exception:
        return []


def _build_fight_rows_for_match_task(
    task: Tuple[str, Optional[Tuple[str, ...]], Optional[int], str, int, Optional[str], bool]
) -> List[Dict[str, Any]]:
    return _build_fight_rows_for_match(*task)


def _resolve_fight_index_workers(n_matches: int) -> int:
    if n_matches <= 1:
        return 1

    user_n = int(getattr(cfg, "FIGHT_INDEX_NUM_WORKERS", 0) or 0)
    if user_n > 0:
        return max(1, min(user_n, n_matches))

    cpu_n = int(os.cpu_count() or 1)
    auto_cap = int(getattr(cfg, "FIGHT_INDEX_MAX_AUTO_WORKERS", 8) or 8)
    # Too-small workloads are usually slower with process startup overhead.
    if n_matches < 8:
        return 1
    return max(1, min(cpu_n, auto_cap, n_matches))


def build_fight_index(
        cache_match_ids: Optional[Iterable[str]] = None,
        max_matches: Optional[int] = None,
        tag: Optional[str] = None,
) -> List[FightRef]:
    """
    ✅ 수정: FightRef에 t_start_ts(ms) 필드 설정
    """
    refs: List[FightRef] = []

    if cache_match_ids is None:
        files = sorted(CACHE_DIR.glob("*.meta.json"))
        mids = [p.stem.replace(".meta", "") for p in files]
    else:
        mids = [str(x) for x in cache_match_ids]

    if max_matches:
        mids = mids[:int(max_matches)]

    cache_path, cfg_sig, cache_key = _fight_index_cache_plan(mids)
    if cache_path is not None and cache_path.exists():
        cached_refs = _load_cached_fight_index(cache_path, cache_key)
        if cached_refs is not None:
            return cached_refs

    seed0 = int(getattr(cfg, "SEEDS", (7,))[0])
    k = getattr(cfg, "MAX_FIGHTS_PER_MATCH", None)
    strat = str(getattr(cfg, "FIGHT_SUBSAMPLE_STRATEGY", "uniform"))
    seed_off = int(getattr(cfg, "FIGHT_SUBSAMPLE_SEED_OFFSET", 0))

    allow = getattr(cfg, "PATCH_ALLOWLIST", None)
    allow_tuple = tuple(str(x) for x in allow) if isinstance(allow, (tuple, list)) and len(allow) > 0 else None

    dump_enabled = bool(getattr(cfg, "DUMP_FIGHTS", False))
    dump_allow_set: set[str] = set()
    if dump_enabled:
        dump_candidates = list(mids)
        dump_allowlist = getattr(cfg, "DUMP_FIGHTS_MATCH_ALLOWLIST", None)
        if isinstance(dump_allowlist, (list, tuple, set)) and len(dump_allowlist) > 0:
            allow_mids = {str(x) for x in dump_allowlist}
            dump_candidates = [mid for mid in dump_candidates if mid in allow_mids]

        max_dump = int(getattr(cfg, "DUMP_FIGHTS_MAX_MATCHES", 5000))
        if max_dump > 0:
            dump_allow_set = set(dump_candidates[:max_dump])

    tasks: List[Tuple[str, Optional[Tuple[str, ...]], Optional[int], str, int, Optional[str], bool]] = [
        (
            str(mid),
            allow_tuple,
            k,
            strat,
            seed0 + seed_off,
            tag,
            bool(dump_enabled and str(mid) in dump_allow_set),
        )
        for mid in mids
    ]

    worker_n = _resolve_fight_index_workers(len(tasks))
    chunk_size = max(1, int(getattr(cfg, "FIGHT_INDEX_MP_CHUNK_SIZE", 8) or 8))

    def _append_rows(rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            try:
                refs.append(
                    FightRef(
                        match_id=str(row["match_id"]),
                        patch=str(row["patch"]),
                        t_start=int(row["t_start"]),
                        t_start_ts=int(row.get("t_start_ts", -1)),
                        label_end_ts=int(row.get("label_end_ts", -1)),
                    )
                )
            except Exception:
                continue

    if worker_n <= 1:
        for task in tasks:
            _append_rows(_build_fight_rows_for_match_task(task))
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_n) as ex:
                for rows in ex.map(_build_fight_rows_for_match_task, tasks, chunksize=chunk_size):
                    _append_rows(rows)
        except Exception:
            for task in tasks:
                _append_rows(_build_fight_rows_for_match_task(task))

    if cache_path is not None:
        _save_cached_fight_index(cache_path, cache_key, cfg_sig, refs)

    return refs


def _patch_sort_key(p: str) -> Tuple[int, ...]:
    """
    Robust numeric patch sorter.
    """
    s = str(p or "")
    nums = re.findall(r"\d+", s)
    if not nums:
        return (0, 0)
    level = str(getattr(cfg, "PATCH_LEVEL", "major_minor")).lower()
    if level == "full":
        t = tuple(int(x) for x in nums[:4])
        return t + (0,) * (4 - len(t))
    t = tuple(int(x) for x in nums[:2])
    return t + (0,) * (2 - len(t))


def split_refs_patch_forward(
        refs: List[FightRef],
        seed: int = 7,
):
    """
    Patch-forward split.
    """
    if not refs:
        return [], [], [], {"mode": "empty"}

    min_patches = int(getattr(cfg, "PATCH_FORWARD_MIN_PATCHES", 3))
    test_last_n = int(getattr(cfg, "PATCH_FORWARD_TEST_LAST_N", 1))
    val_last_n = int(getattr(cfg, "PATCH_FORWARD_VAL_LAST_N", 1))
    group_by_match = bool(getattr(cfg, "PATCH_FORWARD_GROUP_BY_MATCH", True))

    if group_by_match:
        by_match: Dict[str, List[FightRef]] = defaultdict(list)
        for r in refs:
            by_match[r.match_id].append(r)

        match_ids = list(by_match.keys())
        patch_by_match = {mid: str(by_match[mid][0].patch) for mid in match_ids}

        by_patch: Dict[str, List[str]] = defaultdict(list)
        for mid in match_ids:
            by_patch[patch_by_match[mid]].append(mid)

        patches_sorted = sorted(by_patch.keys(), key=_patch_sort_key)

        if len(patches_sorted) < min_patches:
            return split_refs_match_patch_stratified(refs, seed=seed, ratios=(0.8, 0.1, 0.1))

        test_last_n = max(1, min(test_last_n, len(patches_sorted) - 1))
        test_patches = patches_sorted[-test_last_n:]

        remain = patches_sorted[:-test_last_n]
        val_last_n = max(1, min(val_last_n, max(1, len(remain) - 1)))
        val_patches = remain[-val_last_n:]
        train_patches = remain[:-val_last_n]

        if len(train_patches) == 0:
            train_patches = remain[:]
            val_patches = []

        tr_mids = [mid for p in train_patches for mid in by_patch[p]]
        va_mids = [mid for p in val_patches for mid in by_patch[p]]
        te_mids = [mid for p in test_patches for mid in by_patch[p]]

        tr = [r for mid in tr_mids for r in by_match[mid]]
        va = [r for mid in va_mids for r in by_match[mid]]
        te = [r for mid in te_mids for r in by_match[mid]]

        info = {
            "mode": "patch_forward_match",
            "seed": int(seed),
            "patches_sorted": patches_sorted,
            "train_patches": train_patches,
            "val_patches": val_patches,
            "test_patches": test_patches,
            "n_matches_total": len(match_ids),
            "n_matches_train": len(tr_mids),
            "n_matches_val": len(va_mids),
            "n_matches_test": len(te_mids),
            "patch_match_counts": {p: len(by_patch[p]) for p in patches_sorted},
        }
        return tr, va, te, info

    by_patch_ref: Dict[str, List[FightRef]] = defaultdict(list)
    for r in refs:
        by_patch_ref[str(r.patch)].append(r)

    patches_sorted = sorted(by_patch_ref.keys(), key=_patch_sort_key)
    if len(patches_sorted) < min_patches:
        return split_refs_random(refs, seed=seed, ratios=(0.8, 0.1, 0.1))

    test_last_n = max(1, min(test_last_n, len(patches_sorted) - 1))
    test_patches = patches_sorted[-test_last_n:]
    remain = patches_sorted[:-test_last_n]
    val_last_n = max(1, min(val_last_n, max(1, len(remain) - 1)))
    val_patches = remain[-val_last_n:]
    train_patches = remain[:-val_last_n]

    tr = [r for p in train_patches for r in by_patch_ref[p]]
    va = [r for p in val_patches for r in by_patch_ref[p]]
    te = [r for p in test_patches for r in by_patch_ref[p]]

    info = {
        "mode": "patch_forward_ref",
        "seed": int(seed),
        "patches_sorted": patches_sorted,
        "train_patches": train_patches,
        "val_patches": val_patches,
        "test_patches": test_patches,
        "patch_ref_counts": {p: len(by_patch_ref[p]) for p in patches_sorted},
        "group_by_match": False,
    }
    return tr, va, te, info


def split_refs_match_patch_stratified(
        refs: List[FightRef],
        seed: int = 7,
        ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
):
    if not refs:
        return [], [], [], {"mode": "empty"}

    if abs(sum(ratios) - 1.0) > 1e-6:
        ratios = (0.6, 0.2, 0.2)

    by_match: Dict[str, List[FightRef]] = defaultdict(list)
    for r in refs:
        by_match[r.match_id].append(r)

    match_ids = list(by_match.keys())
    patch_by_match = {mid: str(by_match[mid][0].patch) for mid in match_ids}

    by_patch: Dict[str, List[str]] = defaultdict(list)
    for mid in match_ids:
        by_patch[patch_by_match[mid]].append(mid)

    rng = np.random.RandomState(int(seed))
    tr_mids, va_mids, te_mids = [], [], []

    for patch, mids in by_patch.items():
        mids = list(mids)
        rng.shuffle(mids)
        n = len(mids)

        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        n_te = max(0, n - n_tr - n_va)

        if n >= 3:
            n_tr = max(1, n_tr)
            n_va = max(1, n_va)
            n_te = max(1, n - n_tr - n_va)
            while n_tr + n_va + n_te > n:
                n_tr = max(1, n_tr - 1)

        tr_mids += mids[:n_tr]
        va_mids += mids[n_tr:n_tr + n_va]
        te_mids += mids[n_tr + n_va:n_tr + n_va + n_te]

    tr = [r for mid in tr_mids for r in by_match[mid]]
    va = [r for mid in va_mids for r in by_match[mid]]
    te = [r for mid in te_mids for r in by_match[mid]]

    info = {
        "mode": "match_patch_stratified",
        "seed": int(seed),
        "ratios": ratios,
        "n_matches_total": len(match_ids),
        "n_matches_train": len(tr_mids),
        "n_matches_val": len(va_mids),
        "n_matches_test": len(te_mids),
        "patch_matches": {p: len(m) for p, m in by_patch.items()},
    }
    return tr, va, te, info


def split_refs_group_match(
        refs: List[FightRef],
        seed: int = 7,
        ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
):
    if not refs:
        return [], [], [], {"mode": "empty"}

    by_match: Dict[str, List[FightRef]] = defaultdict(list)
    for r in refs:
        by_match[r.match_id].append(r)

    match_ids = sorted(by_match.keys())
    rng = np.random.RandomState(int(seed))
    rng.shuffle(match_ids)

    n = len(match_ids)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    n_te = max(0, n - n_tr - n_va)

    if n >= 3:
        n_tr = max(1, n_tr)
        n_va = max(1, n_va)
        n_te = max(1, n - n_tr - n_va)
        while n_tr + n_va + n_te > n:
            n_tr = max(1, n_tr - 1)

    tr_mids = match_ids[:n_tr]
    va_mids = match_ids[n_tr:n_tr + n_va]
    te_mids = match_ids[n_tr + n_va:n_tr + n_va + n_te]

    tr = [r for mid in tr_mids for r in by_match[mid]]
    va = [r for mid in va_mids for r in by_match[mid]]
    te = [r for mid in te_mids for r in by_match[mid]]

    info = {
        "mode": "group_match",
        "seed": int(seed),
        "ratios": ratios,
        "n_matches_total": len(match_ids),
        "n_matches_train": len(tr_mids),
        "n_matches_val": len(va_mids),
        "n_matches_test": len(te_mids),
    }
    return tr, va, te, info


def split_refs_random(
        refs: List[FightRef],
        seed: int = 7,
        ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
):
    if not refs:
        return [], [], [], {"mode": "empty"}

    group_by_match = bool(getattr(cfg, "SPLIT_GROUP_BY_MATCH_ID", True))
    if group_by_match:
        return split_refs_group_match(refs, seed=seed, ratios=ratios)

    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(refs))
    rng.shuffle(idx)

    n = len(refs)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    n_te = max(0, n - n_tr - n_va)

    if n >= 10:
        n_tr = max(1, n_tr)
        n_va = max(1, n_va)
        n_te = max(1, n - n_tr - n_va)
        while n_tr + n_va + n_te > n:
            n_tr = max(1, n_tr - 1)

    tr = [refs[i] for i in idx[:n_tr]]
    va = [refs[i] for i in idx[n_tr:n_tr + n_va]]
    te = [refs[i] for i in idx[n_tr + n_va:n_tr + n_va + n_te]]

    info = {
        "mode": "random_ref",
        "seed": int(seed),
        "ratios": ratios,
        "n_refs_total": int(n),
        "n_refs_train": len(tr),
        "n_refs_val": len(va),
        "n_refs_test": len(te),
        "group_by_match": False,
    }
    return tr, va, te, info


def validate_split_label_balance(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    label_fn=None,
    max_drift: float = 0.10,
) -> Dict[str, Any]:
    """Validate that label distribution does not drift excessively across splits.

    Checks that |pos_rate_train - pos_rate_test| < max_drift to catch
    cases where random splitting creates unbalanced label distributions.

    Returns a dict with per-split stats and a 'balanced' boolean.
    """
    import logging
    logger = logging.getLogger(__name__)

    def _pos_rate(refs):
        if not refs:
            return float("nan")
        # If label_fn provided, use it; otherwise return nan
        if label_fn is not None:
            labels = [label_fn(r) for r in refs]
            labels = [l for l in labels if l is not None and l >= 0]
            if not labels:
                return float("nan")
            return float(np.mean(labels))
        return float("nan")

    rates = {
        "train_pos_rate": _pos_rate(tr_refs),
        "val_pos_rate": _pos_rate(va_refs),
        "test_pos_rate": _pos_rate(te_refs),
        "train_n": len(tr_refs),
        "val_n": len(va_refs),
        "test_n": len(te_refs),
    }

    tr_rate = rates["train_pos_rate"]
    te_rate = rates["test_pos_rate"]

    if not (np.isnan(tr_rate) or np.isnan(te_rate)):
        drift = abs(tr_rate - te_rate)
        rates["label_drift"] = float(drift)
        rates["balanced"] = drift < max_drift
        if drift >= max_drift:
            logger.warning(
                "Label drift between train (%.3f) and test (%.3f) exceeds threshold %.2f",
                tr_rate, te_rate, max_drift,
            )
    else:
        rates["label_drift"] = float("nan")
        rates["balanced"] = True  # can't check without labels

    return rates


def split_refs(
        refs: List[FightRef],
        mode: Optional[str] = None,
        seed: Optional[int] = None,
):
    if not refs:
        return [], [], [], {"mode": "empty"}

    seed0 = int(seed) if seed is not None else int(getattr(cfg, "SEEDS", (7,))[0])
    mode0 = str(mode if mode is not None else getattr(cfg, "SPLIT_MODE", "multi_patch")).lower().strip()
    if mode0 in ("", "auto"):
        mode0 = str(getattr(cfg, "SPLIT_MODE", "multi_patch")).lower().strip()
    if mode0 in ("match_id", "match", "group", "group_match"):
        mode0 = "group_match"
    elif mode0 in ("rand",):
        mode0 = "random"
    elif mode0 in ("forward_patch", "patch_time"):
        mode0 = "patch_forward"
    elif mode0 in ("stratified",):
        mode0 = "multi_patch"

    if mode0 in ("patch_forward",):
        return split_refs_patch_forward(refs, seed=seed0)

    ratios = getattr(cfg, "SPLIT_RATIOS", None)
    if ratios is None:
        va = float(getattr(cfg, "VAL_FRAC", 0.2))
        te = float(getattr(cfg, "TEST_FRAC", 0.2))
        tr = float(max(0.0, 1.0 - va - te))
        ratios = (tr, va, te)

    if mode0 in ("group_match",):
        return split_refs_group_match(refs, seed=seed0, ratios=ratios)
    if mode0 in ("random",):
        return split_refs_random(refs, seed=seed0, ratios=ratios)
    return split_refs_match_patch_stratified(refs, seed=seed0, ratios=ratios)


def estimate_seq_keep_indices(
        refs: List[FightRef],
        feature_set: str,
        seed: int,
        max_samples: int,
        thresh: float,
        kind: str = "x_seq",
):
    if not getattr(cfg, "DROP_CORR_FEATURES", False) or not refs:
        return None, []

    if feature_set == "tri_modal" and kind == "x_seq":
        return None, []

    rs = np.random.RandomState(seed)
    pick = [refs[i] for i in rs.choice(len(refs), min(len(refs), max_samples), replace=False)]
    vecs = []

    for r in pick:
        pack = load_match_cache(r.match_id)
        if not pack:
            continue

        try:
            label_end_ts = int(getattr(r, "label_end_ts", -1))
        except Exception:
            label_end_ts = -1

        # ✅ ms 기반 호출
        if r.t_start_ts >= 0:
            raw = build_ms_sequence(
                pack,
                pack["meta"]["team_map"],
                -1,
                engage_ts=r.t_start_ts,
                label_end_ts=(label_end_ts if label_end_ts >= 0 else None),
            )
        else:
            raw = build_ms_sequence(pack, pack["meta"]["team_map"], r.t_start)

        if not raw:
            continue

        feats = build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None),
                                        feature_set)

        if kind == "x_seq":
            arr = feats.get("x_seq", None)
            names = get_xseq_feature_names(feature_set)
        else:
            arr = feats.get("extra_seq", None) if "extra_seq" in feats else feats.get("macro_seq", None)
            names = get_extra_feature_names(feature_set)

        if arr is None:
            continue

        vecs.append(arr.mean(axis=0))

    if len(vecs) < 50:
        return None, []

    X = np.stack(vecs, axis=0)
    keep_idx, dropped = prune_correlated_columns(X, names, thresh, max_rows=2000, seed=seed)
    return keep_idx, dropped
