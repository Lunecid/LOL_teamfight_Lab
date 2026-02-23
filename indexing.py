from __future__ import annotations

import random
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import CACHE_DIR, cfg
from fight_types import FightRef, ref_key
from pipeline import build_ms_sequence
from cache_io import load_match_cache
from features import build_sequence_features
from utils import read_json, write_log




def patch_sort_key(p: str) -> Tuple[int, ...]:
    """Robust numeric patch sort key.

    - "15.14" -> (15, 14)
    - "15.14.695.3589" -> (15,14,695,3589)
    """
    s = str(p or "0.0")
    nums = []
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(int(cur))
                cur = ""
    if cur:
        nums.append(int(cur))
    if not nums:
        return (0, 0)

    level = str(getattr(cfg, "PATCH_LEVEL", "major_minor")).lower()
    if level == "full":
        nums = nums[:4]
        return tuple(nums + [0] * (4 - len(nums)))
    nums = nums[:2]
    return tuple(nums + [0] * (2 - len(nums)))


def count_patches_from_refs(refs: List[FightRef]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in refs:
        p = str(getattr(r, "patch", "") or "0.0")
        d[p] = d.get(p, 0) + 1
    return dict(sorted(d.items(), key=lambda x: patch_sort_key(x[0])))


def format_patch_counts(d: Dict[str, int], max_items: int = 30) -> str:
    items = sorted(list(d.items()), key=lambda x: patch_sort_key(x[0]))
    total = sum(v for _, v in items)
    show = items[:max_items]
    s = ", ".join([f"{k}:{v}" for k, v in show])
    if len(items) > max_items:
        s += f" (+{len(items)-max_items} more)"
    s += f" | total={total}"
    return s


def log_patch_block(title: str, d: Dict[str, int], log_fp: Path) -> None:
    write_log(f"[PATCH] {title}: {format_patch_counts(d)}", log_fp)


def scan_cache_match_patch_counts(max_matches: Optional[int] = None) -> Dict[str, int]:
    d: Dict[str, int] = {}
    files = sorted(CACHE_DIR.glob("*.meta.json"))

    if max_matches is not None and int(max_matches) <= 0:
        max_matches = None
    if max_matches:
        files = files[: int(max_matches)]

    for fp in files:
        try:
            m = read_json(fp)
            if not isinstance(m, dict):
                continue
            p = str(m.get("patch", "") or m.get("patch_full", "") or "0.0")
            d[p] = d.get(p, 0) + 1
        except Exception:
            continue

    return dict(sorted(d.items(), key=lambda x: patch_sort_key(x[0])))


def scan_cache_match_ids(max_matches: Optional[int] = None) -> List[str]:
    """Return match_ids present in CACHE_DIR based on *.meta.json.

    Note: `max_matches=0` should mean "no limit" (None), not "empty".
    """
    if max_matches is not None and int(max_matches) <= 0:
        max_matches = None

    meta_files = sorted(CACHE_DIR.glob("*.meta.json"))
    ids: List[str] = []
    for fp in meta_files:
        mid = fp.name.replace(".meta.json", "")
        if mid:
            ids.append(mid)
        if max_matches is not None and len(ids) >= int(max_matches):
            break
    return ids


def _match_ids(refs: List[FightRef]) -> set:
    return set(r.match_id for r in refs)


def check_split_leakage(
    tr: List[FightRef],
    va: List[FightRef],
    te: List[FightRef],
    log_fp: Path,
    *,
    fail_on_leakage: bool = True,
) -> bool:
    a, b, c = _match_ids(tr), _match_ids(va), _match_ids(te)
    ab = a & b
    ac = a & c
    bc = b & c
    if ab or ac or bc:
        msg = (
            f"match_id leakage detected: train∩val={len(ab)} "
            f"train∩test={len(ac)} val∩test={len(bc)}"
        )
        if fail_on_leakage:
            write_log(f"[FATAL] {msg}", log_fp)
            raise RuntimeError(msg)
        write_log(f"[WARN] {msg}", log_fp)
        return True

    write_log(f"[SPLIT] match_id disjoint ✅ (n_match tr/va/te = {len(a)}/{len(b)}/{len(c)})", log_fp)
    return False


def split_by_match_id(refs: List[FightRef], val_ratio: float, seed: int) -> Tuple[List[FightRef], List[FightRef]]:
    """Split refs by match_id disjoint."""
    if not refs:
        return [], []
    val_ratio = float(np.clip(val_ratio, 0.0, 0.9))
    mids = sorted(list({r.match_id for r in refs}))
    rng = np.random.RandomState(seed)
    rng.shuffle(mids)
    n_val = int(round(len(mids) * val_ratio))
    val_set = set(mids[:n_val])
    tr = [r for r in refs if r.match_id not in val_set]
    va = [r for r in refs if r.match_id in val_set]
    return tr, va


def split_refs_patch_holdout(
    refs: List[FightRef],
    seed: int,
    train_patches: Optional[List[str]] = None,
    test_patches: Optional[List[str]] = None,
    val_patches: Optional[List[str]] = None,
    val_ratio_from_train: float = 0.15,
    log_fp: Optional[Path] = None,
) -> Tuple[List[FightRef], List[FightRef], List[FightRef], Dict[str, Any]]:
    """Patch holdout split.

    Priority:
      - test refs: patch in test_patches
      - val refs: patch in val_patches (if provided)
      - train refs: remaining patches (or train_patches if provided), excluding test/val

    If test_patches not given:
      - latest patch -> test
    If train_patches not given:
      - all except test/val -> train
    If val_patches not given:
      - split_by_match_id from train
    """

    if not refs:
        return [], [], [], {"mode": "patch_holdout", "reason": "empty_refs"}

    # --- normalize patch strings
    def _p(r: FightRef) -> str:
        return str(getattr(r, "patch", "") or "0.0")

    all_patches = sorted(list({_p(r) for r in refs}), key=patch_sort_key)

    train_patches = list(train_patches or [])
    test_patches  = list(test_patches or [])
    val_patches   = list(val_patches or [])

    # auto choose test if missing
    if not test_patches:
        test_patches = [all_patches[-1]]

    # --- build TEST first (highest priority)
    test_set = set(test_patches)
    te = [r for r in refs if _p(r) in test_set]
    te_keys = {ref_key(r) for r in te}

    # --- build VAL next if explicit
    mode_detail = ""
    if val_patches:
        val_set = set(val_patches)
        # val should be from refs (not only from tr0). allow val patches independent of train_patches.
        va = [r for r in refs if (_p(r) in val_set) and (ref_key(r) not in te_keys)]
        va_keys = {ref_key(r) for r in va}

        # --- build TRAIN
        if train_patches:
            tr_set = set(train_patches)
            tr0 = [r for r in refs if _p(r) in tr_set]
        else:
            # default: everything except test/val patches
            blocked = set(test_patches) | set(val_patches)
            tr0 = [r for r in refs if _p(r) not in blocked]

        # remove any overlap by ref_key (robust even if FightRef not hashable)
        tr = [r for r in tr0 if (ref_key(r) not in va_keys) and (ref_key(r) not in te_keys)]
        mode_detail = "explicit_val_patches"

    else:
        # --- no explicit val patches: make TRAIN base first, excluding test
        if train_patches:
            tr_set = set(train_patches)
            tr0 = [r for r in refs if (_p(r) in tr_set) and (ref_key(r) not in te_keys)]
        else:
            tr0 = [r for r in refs if _p(r) not in test_set]

        # split val from train by match_id
        tr, va = split_by_match_id(tr0, val_ratio=val_ratio_from_train, seed=seed)
        # enforce disjoint by ref_key (paranoia)
        va_keys = {ref_key(r) for r in va}
        tr = [r for r in tr if ref_key(r) not in va_keys]
        mode_detail = "split_by_match_id"

    # --- build meta (patch lists for logging; infer train_patches if missing)
    if not train_patches:
        # infer from produced tr
        train_patches = sorted(list({_p(r) for r in tr}), key=patch_sort_key)

    meta = {
        "mode": "patch_holdout",
        "mode_detail": mode_detail,
        "all_patches": all_patches,
        "train_patches": train_patches,
        "val_patches": val_patches,
        "test_patches": test_patches,
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "patch_counts": {
            "train": count_patches_from_refs(tr),
            "val": count_patches_from_refs(va),
            "test": count_patches_from_refs(te),
        },
    }

    if log_fp:
        write_log(
            f"[SPLIT] patch_holdout mode={mode_detail} "
            f"train={train_patches} val={val_patches} test={test_patches} "
            f"counts: tr={len(tr)} va={len(va)} te={len(te)}",
            log_fp,
        )
        log_patch_block("train", meta["patch_counts"]["train"], log_fp)
        log_patch_block("val", meta["patch_counts"]["val"], log_fp)
        log_patch_block("test", meta["patch_counts"]["test"], log_fp)

    return tr, va, te, meta


def filter_loadable_refs(
    refs: List[FightRef],
    feature_set: str,
    tag: str,
    log_fp: Path,
    log_every: int = 5000,
) -> Tuple[List[FightRef], Dict[str, int]]:
    """Filter refs by whether `build_ms_sequence` succeeds."""
    used: List[FightRef] = []
    pc: Dict[str, int] = {}

    t0 = time.time()
    for i, r in enumerate(refs):
        pack = load_match_cache(r.match_id)
        if not pack:
            continue
        ts = getattr(r, "t_start_ts", -1)
        raw = build_ms_sequence(pack, pack["meta"]["team_map"], r.t_start, engage_ts=(ts if int(ts) >= 0 else None))
        if not raw:
            continue

        # Also validate we can build features (catches missing keys early)
        try:
            _ = build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None), feature_set)
        except Exception:
            continue

        used.append(r)
        p = str(getattr(r, "patch", "") or "0.0")
        pc[p] = pc.get(p, 0) + 1

        if (i + 1) % log_every == 0:
            write_log(f"[LOADABLE] {tag}: checked={i+1} used={len(used)}", log_fp)

    pc = dict(sorted(pc.items(), key=lambda x: patch_sort_key(x[0])))
    write_log(f"[LOADABLE] {tag}: used={len(used)}/{len(refs)} time={time.time()-t0:.1f}s", log_fp)
    log_patch_block(f"LOADABLE({tag})", pc, log_fp)
    return used, pc


def split_by_match_id_kfold(
    refs,
    n_splits: int | None = None,
    seed: int = 0,
    k: int | None = None,
    folds: int | None = None,
    **kwargs,
):
    """
    Backward/forward compatible match_id K-fold splitter.

    Accepts:
      - n_splits (new)
      - k / folds (legacy aliases)
      - ignores unknown kwargs safely
    Returns:
      List[(train_refs, holdout_refs)]
    """
    refs = list(refs or [])
    if not refs:
        return []

    # --- resolve split count ---
    if n_splits is None:
        n_splits = k if k is not None else folds
    if n_splits is None:
        # last resort: try common legacy names in kwargs
        n_splits = kwargs.get("n_fold") or kwargs.get("n_folds") or kwargs.get("num_folds")

    n_splits = int(n_splits) if n_splits is not None else 5  # default 5
    n_splits = max(2, n_splits)

    # --- build match-id list ---
    mids = sorted({r.match_id for r in refs})
    if len(mids) <= 1:
        # cannot really K-fold; return single fold (train=all, hold=empty)
        return [(refs, [])]

    import numpy as np
    rng = np.random.RandomState(int(seed))
    mids = list(mids)
    rng.shuffle(mids)

    n_splits = min(n_splits, len(mids))  # cannot exceed unique match_ids

    folds_mid = [[] for _ in range(n_splits)]
    for i, mid in enumerate(mids):
        folds_mid[i % n_splits].append(mid)

    out = []
    for fi in range(n_splits):
        hold_set = set(folds_mid[fi])
        hold = [r for r in refs if r.match_id in hold_set]
        tr = [r for r in refs if r.match_id not in hold_set]
        out.append((tr, hold))
    return out
