"""Kill-event extraction from the match cache for boundary estimation.

Produces two things per corpus slice:

* ``records``: one entry per match with the time-sorted kill timestamps (s),
  the patch, and the kill count -- the input of the temporal boundary;
* ``pairs``: every consecutive kill pair with time gap, distance, positions,
  whether the two kills share a champion (killer / victim / assister), the
  map region of each kill and the lane tangent at the first -- the input of
  the spatial boundary.

Only CHAMPION_KILL events are read.  Kills without a position are kept for
the temporal side and skipped on the spatial side (they are rare: 0 of
509,010 in the 10,000-match sample).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.map_regions import classify_points

PAIR_KEYS = ("match_idx", "dt", "dd", "shared", "x1", "y1", "x2", "y2", "minute", "same_victim_team")


@dataclass
class MatchRecord:
    match_id: str
    patch: str
    ts: np.ndarray          # kill times, seconds, sorted
    n_kills: int

    @property
    def log_dt(self) -> np.ndarray:
        d = np.diff(self.ts)
        d = d[d > 0]
        return np.log10(d)


def list_cache_match_ids() -> List[str]:
    from core.config import CACHE_DIR
    return sorted(p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json"))


def read_patch(match_id: str) -> Optional[str]:
    """Patch from the .meta.json only (no events load)."""
    from data.cache_io import cache_paths
    meta_path = cache_paths(match_id)[2]   # (npz, events.json, meta.json)
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        p = meta.get("patch")
        return str(p) if p is not None else None
    except Exception:
        return None


def group_by_patch(match_ids: Sequence[str], progress: Optional[Callable[[str], None]] = None,
                   index_path=None) -> Dict[str, List[str]]:
    """Patch -> match ids.  Scanning 200k .meta.json files takes minutes on Windows, so the
    result is cached at ``index_path`` (default: next to the cache directory) and reused for
    every id it already covers."""
    from pathlib import Path
    if index_path is None:
        from core.config import CACHE_DIR
        index_path = Path(CACHE_DIR).parent / (Path(CACHE_DIR).name + "_patch_index.json")
    index_path = Path(index_path)
    cached: Dict[str, str] = {}
    if index_path.exists():
        try:
            cached = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
    missing = [m for m in match_ids if m not in cached]
    for i, mid in enumerate(missing, 1):
        p = read_patch(mid)
        if p:
            cached[mid] = p
        if progress and i % 20000 == 0:
            progress(f"  patch scan {i}/{len(missing)}")
    if missing:
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(cached), encoding="utf-8")
        except Exception:
            pass
    out: Dict[str, List[str]] = {}
    for mid in match_ids:
        p = cached.get(mid)
        if p:
            out.setdefault(p, []).append(mid)
    return out


def sample_ids(match_ids: Sequence[str], n: int, seed: int) -> List[str]:
    ids = sorted(match_ids)
    if n and n < len(ids):
        return sorted(random.Random(seed).sample(ids, n))
    return ids


def _participants(kill: dict) -> set:
    s = set()
    for pid in [kill.get("killer_id", 0), kill.get("victim_id", 0)] + list(kill.get("assisting_ids", []) or []):
        try:
            pid = int(pid)
        except Exception:
            continue
        if 1 <= pid <= 10:
            s.add(pid)
    return s


def extract_kill_pairs(match_ids: Sequence[str], progress: Optional[Callable[[str], None]] = None,
                       ) -> Tuple[List[MatchRecord], Dict[str, np.ndarray]]:
    """Read kills for ``match_ids`` from the cache; return records and pairs.

    Per-match arrays are accumulated and concatenated once, so a 75k-match slice
    (about 3.5M pairs) stays within a few hundred MB.
    """
    from data.cache_io import load_match_cache
    from gameplay.fights import _extract_kill_events

    records: List[MatchRecord] = []
    chunks: Dict[str, list] = {k: [] for k in PAIR_KEYS}
    t0 = time.time()
    for done, mid in enumerate(match_ids, 1):
        pack = load_match_cache(mid)
        if not pack:
            continue
        kills = _extract_kill_events(pack.get("events") or [])
        if len(kills) < 2:
            continue
        tm = pack["meta"].get("team_map", {}) or {}
        patch = str(pack["meta"].get("patch", "?"))
        kills = sorted(kills, key=lambda k: int(k["timestamp"]))
        ts = np.array([k["timestamp"] for k in kills], dtype=np.float64) / 1000.0
        mi = len(records)
        records.append(MatchRecord(mid, patch, ts, len(kills)))
        pos = np.array([k["position"] if k.get("position") else (np.nan, np.nan) for k in kills], dtype=np.float64)
        parts = [_participants(k) for k in kills]
        vteam = np.array([tm.get(str(k.get("victim_id", 0)), tm.get(k.get("victim_id", 0), -1)) or -1 for k in kills])
        n = len(kills)
        d = np.diff(ts)
        ok = (d > 0) & np.isfinite(pos[:-1, 0]) & np.isfinite(pos[1:, 0])
        if not ok.any():
            continue
        i = np.where(ok)[0]
        chunks["match_idx"].append(np.full(i.size, mi, dtype=np.int32))
        chunks["dt"].append(d[i])
        chunks["dd"].append(np.hypot(pos[i + 1, 0] - pos[i, 0], pos[i + 1, 1] - pos[i, 1]))
        chunks["shared"].append(np.array([len(parts[j] & parts[j + 1]) > 0 for j in i], dtype=bool))
        chunks["x1"].append(pos[i, 0].astype(np.float32)); chunks["y1"].append(pos[i, 1].astype(np.float32))
        chunks["x2"].append(pos[i + 1, 0].astype(np.float32)); chunks["y2"].append(pos[i + 1, 1].astype(np.float32))
        chunks["minute"].append((ts[i] / 60.0).astype(np.float32))
        chunks["same_victim_team"].append(vteam[i] == vteam[i + 1])
        if progress and done % 5000 == 0:
            progress(f"  {done}/{len(match_ids)} matches, {sum(a.size for a in chunks['dt']):,} pairs, {time.time() - t0:.0f}s")
    P = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in chunks.items()}
    if P["dt"].size:
        r1, t1 = classify_points(np.stack([P["x1"], P["y1"]], axis=1).astype(np.float64))
        r2, _ = classify_points(np.stack([P["x2"], P["y2"]], axis=1).astype(np.float64))
        P["region1"], P["region2"], P["tangent1"] = r1.astype("<U10"), r2.astype("<U10"), t1.astype(np.float32)
    else:
        P["region1"] = np.empty(0, dtype="<U10"); P["region2"] = np.empty(0, dtype="<U10"); P["tangent1"] = np.empty((0, 2), dtype=np.float32)
    P["patch_per_match"] = np.asarray([r.patch for r in records])
    return records, P


def save_pairs(path, records: Sequence[MatchRecord], pairs: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **pairs,
                        rec_match_id=np.asarray([r.match_id for r in records]),
                        rec_patch=np.asarray([r.patch for r in records]),
                        rec_ts=np.asarray([r.ts for r in records], dtype=object),
                        rec_n_kills=np.asarray([r.n_kills for r in records]))


def load_pairs(path) -> Tuple[List[MatchRecord], Dict[str, np.ndarray]]:
    Z = np.load(path, allow_pickle=True)
    records = [MatchRecord(str(m), str(p), np.asarray(t, dtype=np.float64), int(n))
               for m, p, t, n in zip(Z["rec_match_id"], Z["rec_patch"], Z["rec_ts"], Z["rec_n_kills"])]
    pairs = {k: Z[k] for k in Z.files if not k.startswith("rec_")}
    return records, pairs
