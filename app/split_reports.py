from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import cfg
from core.fight_types import FightRef, ref_key
from core.utils import metrics_from_probs, save_json, write_log
from data.cache_io import load_match_cache
from data.labels import get_label_map


def _sigmoid_logit_arr(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z.astype(np.float64), -30.0, 30.0)))


def _engage_minute(ref: FightRef) -> int:
    ts = int(getattr(ref, "t_start_ts", -1) or -1)
    if ts >= 0:
        return max(0, int(ts // 60000))
    return max(0, int(getattr(ref, "t_start", 0) or 0))


def _subset_metrics(
    refs: List[FightRef],
    logit_map: Dict[str, float],
    y_map: Dict[str, int],
) -> Dict[str, Any]:
    ys: List[int] = []
    zs: List[float] = []
    for r in refs:
        k = ref_key(r)
        if k not in y_map or k not in logit_map:
            continue
        ys.append(int(y_map[k]))
        zs.append(float(logit_map[k]))
    if not ys:
        return {"n": 0}
    y = np.asarray(ys, dtype=np.int64)
    p = _sigmoid_logit_arr(np.asarray(zs, dtype=np.float64))
    met = metrics_from_probs(y, p, threshold=float(getattr(cfg, "CLS_THRESHOLD", 0.5)))
    met["n"] = int(len(y))
    met["mean_prob"] = float(np.mean(p))
    met["pos_rate"] = float(np.mean(y))
    return met


def _build_minutewise_report(
    refs: List[FightRef],
    logit_map: Dict[str, float],
    y_map: Dict[str, int],
) -> Dict[str, Any]:
    max_minute = int(getattr(cfg, "MINUTE_REPORT_MAX_MINUTE", 60))
    minute_to_refs: Dict[int, List[FightRef]] = {}
    for r in refs:
        m = _engage_minute(r)
        if m < 0 or m > max_minute:
            continue
        minute_to_refs.setdefault(int(m), []).append(r)

    rows: List[Dict[str, Any]] = []
    prev_mean_prob: Optional[float] = None
    for m in sorted(minute_to_refs.keys()):
        met = _subset_metrics(minute_to_refs[m], logit_map, y_map)
        row = {"minute": int(m), **met}
        mp = met.get("mean_prob", None)
        if mp is not None and prev_mean_prob is not None:
            row["delta_mean_prob"] = float(mp - prev_mean_prob)
        else:
            row["delta_mean_prob"] = None
        if mp is not None:
            prev_mean_prob = float(mp)
        rows.append(row)

    return {
        "prediction_gap_ms": int(getattr(cfg, "PREDICTION_GAP_MS", 0)),
        "overall": _subset_metrics(refs, logit_map, y_map),
        "by_minute": rows,
    }


def _prefight_gold_state_by_key(refs: List[FightRef]) -> Dict[str, str]:
    from core.timeutils import gold_at_ms

    close_th = float(getattr(cfg, "SITUATION_CLOSE_GOLD_TH", 2000.0))
    stomp_th = float(getattr(cfg, "SITUATION_STOMP_GOLD_TH", 5000.0))

    out: Dict[str, str] = {}
    pack_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for r in refs:
        mid = str(r.match_id)
        if mid not in pack_cache:
            pack_cache[mid] = load_match_cache(mid)
        pack = pack_cache[mid]
        if not pack:
            continue

        ts = int(getattr(r, "t_start_ts", -1) or -1)
        if ts < 0:
            t_idx = int(getattr(r, "t_start", -1) or -1)
            mts = pack.get("minute_ts", None)
            if isinstance(mts, np.ndarray) and 0 <= t_idx < len(mts):
                ts = int(mts[t_idx])
        if ts < 0:
            continue

        try:
            g = gold_at_ms(pack, ts, method=str(getattr(cfg, "LABEL_GOLD_METHOD", "linear")).lower())
            gd = float(g[0] - g[1])
        except Exception:
            continue

        a = abs(gd)
        if a < close_th:
            bucket = "close"
        elif a < stomp_th:
            bucket = "moderate"
        else:
            bucket = "stomp"
        out[ref_key(r)] = bucket

    return out


def _build_situation_report(
    refs: List[FightRef],
    logit_map: Dict[str, float],
    y_map: Dict[str, int],
) -> Dict[str, Any]:
    phase_groups: Dict[str, List[FightRef]] = {"early": [], "mid": [], "late": []}
    patch_groups: Dict[str, List[FightRef]] = {}
    gold_state_groups: Dict[str, List[FightRef]] = {"close": [], "moderate": [], "stomp": [], "unknown": []}

    gold_state_by_key = _prefight_gold_state_by_key(refs)

    for r in refs:
        m = _engage_minute(r)
        if m < 14:
            phase_groups["early"].append(r)
        elif m < 28:
            phase_groups["mid"].append(r)
        else:
            phase_groups["late"].append(r)

        patch = str(getattr(r, "patch", "unknown"))
        patch_groups.setdefault(patch, []).append(r)

        gk = gold_state_by_key.get(ref_key(r), "unknown")
        gold_state_groups.setdefault(gk, []).append(r)

    return {
        "overall": _subset_metrics(refs, logit_map, y_map),
        "by_phase": {k: _subset_metrics(v, logit_map, y_map) for k, v in phase_groups.items()},
        "by_gold_state": {k: _subset_metrics(v, logit_map, y_map) for k, v in gold_state_groups.items()},
        "by_patch": {k: _subset_metrics(v, logit_map, y_map) for k, v in sorted(patch_groups.items())},
    }


def emit_split_reports(
    model_dir: Path,
    model_name: str,
    variant_tag: str,
    feature_set: str,
    refs_by_split: Dict[str, List[FightRef]],
    rep: Dict[str, Any],
    run_log: Path,
) -> None:
    pred_maps = rep.get("_pred_maps_in_memory", {}) if isinstance(rep, dict) else {}
    label_maps = rep.get("_label_maps_in_memory", {}) if isinstance(rep, dict) else {}
    if not isinstance(pred_maps, dict) or not pred_maps:
        return

    for split in ("val", "test"):
        refs = refs_by_split.get(split, [])
        logit_map = pred_maps.get(split, {})
        if not refs or not isinstance(logit_map, dict) or not logit_map:
            continue

        y_map = label_maps.get(split, {}) if isinstance(label_maps, dict) else {}
        if not isinstance(y_map, dict) or not y_map:
            y_map = get_label_map(refs, feature_set=feature_set, log_fp=run_log, log_every=50000)

        if bool(getattr(cfg, "ENABLE_MINUTEWISE_REPORT", True)):
            minute_rep = _build_minutewise_report(refs, logit_map, y_map)
            save_json(model_dir / f"minute_report_{split}.json", minute_rep)

        if bool(getattr(cfg, "ENABLE_SITUATION_REPORT", True)):
            situation_rep = _build_situation_report(refs, logit_map, y_map)
            save_json(model_dir / f"situation_report_{split}.json", situation_rep)

    write_log(f"[REPORT] split reports emitted for {model_name}/{variant_tag}", run_log)
