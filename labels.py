from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fight_types import FightRef, ref_key
from pipeline import build_ms_sequence
from cache_io import load_match_cache
from features import build_sequence_features

# [P2-STRUCT-2] Unified write_log from utils.py (single source of truth).
# Previously: local stub using logging.info() — behaviour mismatch with
# utils.write_log() which uses print(). All modules now share identical
# write_log semantics: stdout + optional file append.
from utils import write_log



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

        # except 블록 바깥으로 이동
        raw = build_ms_sequence(
            pack, pack["meta"]["team_map"], r.t_start,
            engage_ts=(ts if ts >= 0 else None)
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