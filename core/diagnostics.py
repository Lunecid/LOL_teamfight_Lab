from __future__ import annotations

import csv
import json
from pathlib import Path
from core.common import Any, Dict, List, Optional, Tuple, np, dataclass
from core.config import CACHE_DIR, RUN_DIR, cfg

try:
    import fcntl  # POSIX file lock (Linux/macOS)
except Exception:  # pragma: no cover - unavailable on Windows
    fcntl = None

# module-level counter
_DUMP_MATCH_COUNT = 0


def _dump_enabled() -> bool:
    return bool(getattr(cfg, "DUMP_FIGHTS", False))


def _dump_max_matches() -> int:
    return int(getattr(cfg, "DUMP_FIGHTS_MAX_MATCHES", 5000))


def _dump_dir(tag: Optional[str] = None) -> Path:
    """
    Prefer cfg.RUN_DIR or cfg.OUTPUT_ROOT if present; fallback to CACHE_DIR.
    Output: <base>/<DUMP_FIGHTS_DIRNAME>/<tag>/...  (tag optional)
    """
    base = getattr(cfg, "RUN_DIR", None)
    if base is None:
        base = getattr(cfg, "OUTPUT_ROOT", None)
    if base is None:
        base = CACHE_DIR
    base = Path(base)

    root = base / str(getattr(cfg, "DUMP_FIGHTS_DIRNAME", "fight_dumps"))
    if tag:
        root = root / str(tag)
    return root


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_sanitize(x: Any) -> Any:
    # numpy types -> python scalars / lists
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    return x


def _lock_file_exclusive(fp) -> None:
    if fcntl is None:
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def _unlock_file(fp) -> None:
    if fcntl is None:
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        return


def _get_continuous_fight_stats(fights: List[dict]) -> Dict[str, Any]:
    """
    ✅ NEW: 연속 교전 merge 관련 통계 계산

    Returns:
        {
            "total_fights": 총 fight 수,
            "merged_fights": sub_segments가 있는 fight 수,
            "total_segments": 전체 세그먼트 수 (원본 + sub),
            "avg_segments_per_fight": fight당 평균 세그먼트,
            "max_segments": 가장 많은 세그먼트를 가진 fight,
        }
    """
    if not fights:
        return {
            "total_fights": 0,
            "merged_fights": 0,
            "total_segments": 0,
            "avg_segments_per_fight": 0.0,
            "max_segments": 0,
        }

    total_fights = len(fights)
    merged_fights = 0
    total_segments = 0
    max_segments = 0

    for f in fights:
        if not isinstance(f, dict):
            continue
        n_seg = f.get("n_segments", 1)
        total_segments += n_seg
        if n_seg > max_segments:
            max_segments = n_seg
        if n_seg > 1 or "sub_segments" in f:
            merged_fights += 1

    avg_segments = float(total_segments) / float(max(1, total_fights))

    return {
        "total_fights": total_fights,
        "merged_fights": merged_fights,
        "total_segments": total_segments,
        "avg_segments_per_fight": round(avg_segments, 2),
        "max_segments": max_segments,
    }


def _maybe_dump_fights_for_match(
        match_id: str,
        patch: str,
        pack: Dict[str, Any],
        fights_raw: List[dict],
        fights_kept: List[dict],
        tag: Optional[str] = None,
) -> None:
    """
    Write:
      - <dump_dir>/<match_id>.fights.json   (raw/kept/diag/meta)
      - <dump_dir>/fight_summary.csv       (append)
    Controlled by cfg.DUMP_FIGHTS and cfg.DUMP_FIGHTS_MAX_MATCHES.
    """
    global _DUMP_MATCH_COUNT
    if not _dump_enabled():
        return

    # Optional allowlist for debugging a few matches only
    allow_mids = getattr(cfg, "DUMP_FIGHTS_MATCH_ALLOWLIST", None)
    if isinstance(allow_mids, (list, tuple, set)) and len(allow_mids) > 0:
        if str(match_id) not in set(map(str, allow_mids)):
            return

    if _DUMP_MATCH_COUNT >= _dump_max_matches():
        return

    out_dir = _dump_dir(tag=tag)
    _safe_mkdir(out_dir)

    diag = pack.get("fight_detect_diag", None)
    meta = pack.get("meta", {}) if isinstance(pack.get("meta", {}), dict) else {}

    # ✅ 모든 config knobs 기록 (재현성)
    knobs = {
        # 기본 탐지 설정
        "FIGHT_DETECTOR": str(getattr(cfg, "FIGHT_DETECTOR", getattr(cfg, "FIGHT_DETECT_ALGO", "engage_v2"))),
        "START_OFFSET_MIN": int(getattr(cfg, "START_OFFSET_MIN", 1)),
        "FIGHT_CONTEXT_MIN": int(getattr(cfg, "FIGHT_CONTEXT_MIN", 1)),
        "FIGHT_MIN_GAP_MIN": int(getattr(cfg, "FIGHT_MIN_GAP_MIN", 3)),
        "FIGHT_MIN_GAP_MS": int(getattr(cfg, "FIGHT_MIN_GAP_MS", int(getattr(cfg, "FIGHT_MIN_GAP_MIN", 3)) * 60000)),
        "FIGHT_HORIZON_SEC": int(getattr(cfg, "FIGHT_HORIZON_SEC", getattr(cfg, "FIGHT_HORIZON_MIN", 1) * 60)),

        # 근접 판정
        "STANDOFF_RADIUS": float(getattr(cfg, "STANDOFF_RADIUS", 1800.0)),
        "STANDOFF_MIN_PAIRS": int(getattr(cfg, "STANDOFF_MIN_PAIRS", 8)),
        "ENGAGE_MIN_DIST_DROP": float(getattr(cfg, "ENGAGE_MIN_DIST_DROP", 250.0)),
        "ENGAGE_MIN_PAIR_GAIN": int(getattr(cfg, "ENGAGE_MIN_PAIR_GAIN", 3)),

        # Kill anchor
        "VERIFY_KILL_IN_HORIZON": bool(getattr(cfg, "VERIFY_KILL_IN_HORIZON", True)),
        "USE_KILL_ANCHOR": bool(getattr(cfg, "USE_KILL_ANCHOR", False)),
        "KILL_ANCHOR_PRE_SEC": int(getattr(cfg, "KILL_ANCHOR_PRE_SEC", 15)),
        "KILL_ANCHOR_COOLDOWN_SEC": int(getattr(cfg, "KILL_ANCHOR_COOLDOWN_SEC", 30)),

        # ✅ NEW: Backtrack 설정
        "USE_BACKTRACK": bool(getattr(cfg, "USE_BACKTRACK", True)),
        "BACKTRACK_MAX_MS": int(getattr(cfg, "BACKTRACK_MAX_MS", 60000)),
        "BACKTRACK_MIN_MS": int(getattr(cfg, "BACKTRACK_MIN_MS", 10000)),
        "BACKTRACK_MIN_PAIRS": int(getattr(cfg, "BACKTRACK_MIN_PAIRS", 3)),

        # ✅ NEW: 연속 교전 merge 설정
        "CONTINUOUS_FIGHT_MERGE": bool(getattr(cfg, "CONTINUOUS_FIGHT_MERGE", True)),
        "CONTINUOUS_FIGHT_MAX_GAP_MS": int(getattr(cfg, "CONTINUOUS_FIGHT_MAX_GAP_MS", 60000)),
        "CONTINUOUS_FIGHT_MERGE_RADIUS": float(getattr(cfg, "CONTINUOUS_FIGHT_MERGE_RADIUS", 2000.0)),

        # Dense detection
        "DETECT_STEP_MS": int(getattr(cfg, "DETECT_STEP_MS", int(getattr(cfg, "BIN_MS", 10000)))),
        "INTERP_METHOD": str(getattr(cfg, "INTERP_METHOD", "none")),

        # Cluster guards
        "REQUIRE_ALIVE_PER_TEAM": int(getattr(cfg, "REQUIRE_ALIVE_PER_TEAM", 0) or 0),
        "REQUIRE_ENGAGED_PER_TEAM": int(getattr(cfg, "REQUIRE_ENGAGED_PER_TEAM", 0) or 0),
        "REQUIRE_LCC_TOTAL": int(getattr(cfg, "REQUIRE_LCC_TOTAL", 0) or 0),
        "REQUIRE_LCC_PER_TEAM": int(getattr(cfg, "REQUIRE_LCC_PER_TEAM", 0) or 0),
        "CLUSTER_MAX_DIAMETER": float(getattr(cfg, "CLUSTER_MAX_DIAMETER", 0.0) or 0.0),

        # Subsample
        "MAX_FIGHTS_PER_MATCH": int(getattr(cfg, "MAX_FIGHTS_PER_MATCH", 0) or 0),
        "FIGHT_SUBSAMPLE_STRATEGY": str(getattr(cfg, "FIGHT_SUBSAMPLE_STRATEGY", "uniform")),

        # ✅ NEW: Label 설정
        "LABEL_TYPE": str(getattr(cfg, "LABEL_TYPE", "micro_win")),
    }

    # ✅ NEW: 연속 교전 통계
    continuous_stats_raw = _get_continuous_fight_stats(fights_raw)
    continuous_stats_kept = _get_continuous_fight_stats(fights_kept)

    payload = {
        "match_id": str(match_id),
        "patch": str(patch),
        "patch_full": str(meta.get("patch_full", "")),
        "meta_keys": sorted(list(meta.keys())),
        "team_map": _json_sanitize(meta.get("team_map", {})),
        "role_slots": _json_sanitize(meta.get("role_slots", None)),
        "anchor_is_norm": bool(meta.get("anchor_is_norm", False)),
        "run_tag": str(tag or ""),
        "knobs": knobs,
        "diag": _json_sanitize(diag),

        # Fight counts
        "n_fights_raw": int(len(fights_raw)),
        "n_fights_kept": int(len(fights_kept)),

        # ✅ NEW: 연속 교전 통계
        "continuous_stats_raw": continuous_stats_raw,
        "continuous_stats_kept": continuous_stats_kept,

        # Fight data
        "fights_raw": _json_sanitize(fights_raw),
        "fights_kept": _json_sanitize(fights_kept),
    }

    # per-match json
    out_json = out_dir / f"{match_id}.fights.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # append summary csv
    out_csv = out_dir / "fight_summary.csv"

    # compute a few extra quick stats
    def _get_ts_list(fs: List[dict]) -> List[int]:
        out = []
        for x in fs:
            if not isinstance(x, dict):
                continue
            # prefer engage_ts (new) then t_engage_ts (legacy)
            v = x.get("engage_ts", x.get("t_engage_ts", x.get("t_ts", x.get("ts", x.get("t_engage", None)))))
            try:
                if v is not None:
                    out.append(int(v))
            except Exception:
                pass
        return out

    ts_raw = _get_ts_list(fights_raw)
    ts_kept = _get_ts_list(fights_kept)
    ts_raw.sort()
    ts_kept.sort()

    def _avg_gap(ts: List[int]) -> str:
        if len(ts) < 2:
            return ""
        gaps = [max(0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]
        return f"{(sum(gaps) / len(gaps)) / 1000.0:.3f}"

    def _safe_diag_int(key: str, default: int = 0) -> int:
        if not isinstance(diag, dict):
            return default
        try:
            return int(diag.get(key, default))
        except Exception:
            return default

    row = {
        # 기본 정보
        "run_tag": str(tag or ""),
        "match_id": str(match_id),
        "patch": str(patch),

        # Fight counts
        "n_raw": int(len(fights_raw)),
        "n_kept": int(len(fights_kept)),

        # Time gaps
        "avg_gap_raw_s": _avg_gap(ts_raw),
        "avg_gap_kept_s": _avg_gap(ts_kept),

        # Detection stats
        "det_candidates": _safe_diag_int("candidates"),
        "det_accepted": _safe_diag_int("accepted"),
        "det_accepted_by_anchor": _safe_diag_int("accepted_by_anchor"),
        "det_backtracked": _safe_diag_int("backtracked"),

        # Rejection reasons
        "rej_gap": _safe_diag_int("rejected_gap"),
        "rej_horizon": _safe_diag_int("rejected_horizon"),
        "rej_startctx": _safe_diag_int("rejected_startctx"),
        "rej_nokill": _safe_diag_int("rejected_nokill"),
        "rej_alive": _safe_diag_int("rejected_alive"),
        "rej_engaged": _safe_diag_int("rejected_engaged"),
        "rej_lcc": _safe_diag_int("rejected_lcc"),
        "rej_compact": _safe_diag_int("rejected_compact"),

        # ✅ NEW: 연속 교전 stats
        "continuous_merged": _safe_diag_int("continuous_merged"),
        "continuous_diff_loc": _safe_diag_int("continuous_different_location"),

        # ✅ NEW: 연속 교전 통계 (kept)
        "merged_fights": continuous_stats_kept.get("merged_fights", 0),
        "total_segments": continuous_stats_kept.get("total_segments", 0),
        "avg_segments": continuous_stats_kept.get("avg_segments_per_fight", 0.0),
        "max_segments": continuous_stats_kept.get("max_segments", 0),
    }

    with out_csv.open("a+", newline="", encoding="utf-8") as f:
        _lock_file_exclusive(f)
        try:
            f.seek(0, 2)
            write_header = (f.tell() == 0)
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
            f.flush()
        finally:
            _unlock_file(f)

    _DUMP_MATCH_COUNT += 1


def reset_dump_counter() -> None:
    """Reset the dump counter (useful for multi-run scenarios)."""
    global _DUMP_MATCH_COUNT
    _DUMP_MATCH_COUNT = 0


def get_dump_count() -> int:
    """Get current dump count."""
    return _DUMP_MATCH_COUNT
