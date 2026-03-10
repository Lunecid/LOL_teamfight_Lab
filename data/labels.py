from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.fight_types import FightRef, ref_key
from gameplay.pipeline import build_ms_sequence
from data.cache_io import load_match_cache
from gameplay.features import build_sequence_features

# [P2-STRUCT-2] Unified write_log from utils.py (single source of truth).
# Previously: local stub using logging.info() — behaviour mismatch with
# utils.write_log() which uses print(). All modules now share identical
# write_log semantics: stdout + optional file append.
from core.utils import write_log


def get_label_map_from_dataset(dataset) -> Dict[str, int]:
    """Extract label map directly from a preloaded InMemoryFightDataset.

    O(N) with zero disk I/O — avoids the full pipeline rebuild that
    get_label_map() performs.  Each sample already contains 'y' and
    'ref_key' from the preload stage.
    """
    out: Dict[str, int] = {}
    for r, s in zip(dataset.refs, dataset.samples):
        # prefer the pre-injected ref_key, fall back to computing it
        k = s.get("ref_key") or ref_key(r)
        y_val = s.get("y")
        if y_val is None:
            continue
        # y may be a scalar tensor (shape [1,1]) or plain int/float
        if hasattr(y_val, "item"):
            out[k] = int(y_val.item())
        else:
            out[k] = int(y_val)
    return out


def get_label_map(
    refs: List[FightRef],
    feature_set: str,
    log_fp: Optional[Path] = None,
    log_every: int = 5000,
) -> Dict[str, int]:
    """Compute ground-truth labels by replaying build_ms_sequence + build_sequence_features.

    This is intentionally defined in one place so baseline/deep/fusion can share it.
    """

    out: Dict[str, int] = {}
    t0 = time.time()
    for i, r in enumerate(refs):
        pack = load_match_cache(r.match_id)
        if not pack:
            continue
        ts = getattr(r, "t_start_ts", -1)
        try:
            ts = int(ts)
        except Exception:
            ts = -1
        le = getattr(r, "label_end_ts", -1)
        try:
            le = int(le)
        except Exception:
            le = -1

        # Cluster kill timestamps for cluster-scoped labels
        fk = int(getattr(r, "first_kill_ts", -1))
        lk = int(getattr(r, "last_kill_ts", -1))

        # except 블록 바깥으로 이동
        raw = build_ms_sequence(
            pack, pack["meta"]["team_map"], r.t_start,
            engage_ts=(ts if ts >= 0 else None),
            label_end_ts=(le if le >= 0 else None),
            first_kill_ts=(fk if fk >= 0 else None),
            last_kill_ts=(lk if lk >= 0 else None),
        )
        if not raw:
            continue
        feats = build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None), feature_set)
        if "y" not in feats:
            continue
        out[ref_key(r)] = int(feats["y"])
        if log_fp and (i + 1) % log_every == 0:
            write_log(f"[LABEL] built={len(out)}/{i+1}", log_fp)
    if log_fp:
        write_log(f"[LABEL] done n={len(out)}/{len(refs)} time={time.time()-t0:.1f}s", log_fp)
    return out


def aligned_xy_from_maps(
    refs: List[FightRef],
    y_map: Dict[str, int],
    comp_maps: List[Dict[str, float]],
    default_logit: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X,y aligned to keys that exist in y_map (order by refs)."""

    keys: List[str] = []
    ys: List[int] = []
    Xs: List[List[float]] = []

    for r in refs:
        k = ref_key(r)
        if k not in y_map:
            continue
        keys.append(k)
        ys.append(int(y_map[k]))
        Xs.append([float(m.get(k, default_logit)) for m in comp_maps])

    if not Xs:
        return np.zeros((0, len(comp_maps)), np.float32), np.zeros((0,), np.int64), []

    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, keys
