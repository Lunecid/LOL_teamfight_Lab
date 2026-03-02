from __future__ import annotations

from core.config import cfg
from core.common import Any, Dict, List, Optional, Tuple, np
from core.timeutils import _get_bin_ms, _get_context_ms, _get_horizon_ms
from gameplay.event_aggregation import aggregate_events as _aggregate_events
from gameplay.event_tokens import build_event_tokens_for_xattn as _build_event_tokens_for_xattn
from gameplay.labels import compute_label as _compute_label, compute_label_targets as _compute_label_targets
from gameplay.pipeline_cache import (
    _dragon_subtype_to_soul,
    _extract_champ_runes_bans_from_detail,
    _is_elder_dragon,
    _mean_feat,
    _normalize_cs_raw,
    _pf_xy,
    _stable_name_id,
    _sum_feat,
    build_global_minute_vector,
    parse_timeline_to_minute_cache,
)
from gameplay.pipeline_interp import (
    _interp_xy_guarded,
    _prev_snapshot_idx,
    global_from_prev_snapshot,
    interpolate_node_global,
)


def aggregate_events(events_or_pack: Any, tm: Dict[int, int], s_ms: int, e_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    return _aggregate_events(events_or_pack, tm, s_ms, e_ms)


def build_event_tokens_for_xattn(
    pack: Dict[str, Any],
    tm: Dict[int, int],
    s_ms: int,
    e_ms: int,
    *,
    max_tokens: int = 64,
) -> Dict[str, np.ndarray]:
    return _build_event_tokens_for_xattn(pack, tm, s_ms, e_ms, max_tokens=max_tokens)


def compute_label(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    t_start: int,
    *,
    engage_ts: Optional[int] = None,
    label_end_ts: Optional[int] = None,
    horizon_ms: Optional[int] = None,
) -> Optional[int]:
    return _compute_label(
        cache,
        tm,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=label_end_ts,
        horizon_ms=horizon_ms,
        interp_node_global=interpolate_node_global,
    )


def compute_label_targets(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    t_start: int,
    *,
    engage_ts: Optional[int] = None,
    label_end_ts: Optional[int] = None,
    horizon_ms: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    return _compute_label_targets(
        cache,
        tm,
        t_start,
        engage_ts=engage_ts,
        label_end_ts=label_end_ts,
        horizon_ms=horizon_ms,
        interp_node_global=interpolate_node_global,
    )


def build_ms_sequence(
    cache: Dict[str, Any],
    tm: Dict[int, int],
    t_start: int,
    *,
    engage_ts: Optional[int] = None,
    label_end_ts: Optional[int] = None,
    ctx_ms: Optional[int] = None,
    bin_ms: Optional[int] = None,
    horizon_ms: Optional[int] = None,
    prediction_gap_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if ctx_ms is None:
        ctx_ms = _get_context_ms()
    if bin_ms is None:
        bin_ms = _get_bin_ms()
    if horizon_ms is None:
        horizon_ms = _get_horizon_ms()
    if prediction_gap_ms is None:
        prediction_gap_ms = int(getattr(cfg, "PREDICTION_GAP_MS", 0))
    prediction_gap_ms = max(0, int(prediction_gap_ms))

    t_min = int(cache["minute_ts"][0])
    t_max = int(cache["minute_ts"][-1])

    label_start_ms: int
    label_end_ms: int
    if engage_ts is not None and engage_ts >= 0:
        label_start_ms = int(engage_ts)
        label_end_ms = label_start_ms + int(horizon_ms)
        if label_end_ts is not None:
            try:
                cand_end = int(label_end_ts)
            except Exception:
                cand_end = -1
            if cand_end > label_start_ms:
                label_end_ms = cand_end
        end_ms = int(label_start_ms - prediction_gap_ms)
        start_ms = end_ms - int(ctx_ms)
    else:
        if t_start < 0 or t_start >= len(cache["minute_ts"]):
            return None
        label_start_ms = int(cache["minute_ts"][t_start])
        label_end_ms = label_start_ms + int(horizon_ms)
        end_ms = int(label_start_ms - prediction_gap_ms)
        start_ms = end_ms - int(ctx_ms)

    if end_ms < t_min:
        return None
    if start_ms < t_min:
        return None
    if label_end_ms > t_max:
        return None

    L = int(ctx_ms // bin_ms)
    if L <= 0:
        return None

    glob_seq, node_seq, ev_seq, item_seq = [], [], [], []
    glob_snap_ts_seq: List[int] = []
    node_max_snapshot_ms: Optional[int] = None
    if engage_ts is not None and engage_ts >= 0:
        node_max_snapshot_ms = int(label_start_ms) - 1

    for i in range(L):
        b0 = start_ms + i * bin_ms
        b1 = start_ms + (i + 1) * bin_ms
        q = b0 + bin_ms // 2

        node_i, glob_i = interpolate_node_global(
            cache,
            q,
            max_snapshot_ms=node_max_snapshot_ms,
        )
        g_ref_ms = int(q)
        if engage_ts is not None and engage_ts >= 0:
            g_ref_ms = min(int(g_ref_ms), int(label_start_ms) - 1)
        glob_i, g_ts = global_from_prev_snapshot(cache, g_ref_ms, strict_before=True)
        glob_snap_ts_seq.append(int(g_ts))
        ev_i, it_i = aggregate_events(cache, tm, b0, b1)

        node_seq.append(node_i)
        glob_seq.append(glob_i)
        ev_seq.append(ev_i)
        item_seq.append(it_i)

    if engage_ts is not None and engage_ts >= 0:
        y_pack = compute_label_targets(
            cache,
            tm,
            -1,
            engage_ts=label_start_ms,
            label_end_ts=label_end_ms,
            horizon_ms=horizon_ms,
        )
    else:
        y_pack = compute_label_targets(
            cache,
            tm,
            t_start,
            engage_ts=label_start_ms,
            label_end_ts=label_end_ms,
            horizon_ms=horizon_ms,
        )

    if y_pack is None:
        return None

    y_s = int(y_pack.get("label_start_ms", label_start_ms))
    y_e = int(y_pack.get("label_end_ms", label_end_ms))
    y_dur = max(0, y_e - y_s)

    sample = {
        "node_seq": np.stack(node_seq, axis=0).astype(np.float32),
        "glob_seq": np.stack(glob_seq, axis=0).astype(np.float32),
        "ev_seq": np.stack(ev_seq, axis=0).astype(np.float32),
        "item_seq": np.stack(item_seq, axis=0).astype(np.float32),
        "y": int(y_pack["y"]),
        "y_kill_diff": float(y_pack.get("kill_diff_norm", 0.0)),
        "y_gold_diff": float(y_pack.get("gold_diff_norm", 0.0)),
        "y_obj_diff": float(y_pack.get("obj_diff_norm", 0.0)),
        "label_kill_diff_raw": float(y_pack.get("kill_diff", 0.0)),
        "label_gold_diff_raw": float(y_pack.get("gold_diff", 0.0)),
        "label_obj_diff_raw": float(y_pack.get("obj_diff", 0.0)),
        "label_alive_diff_raw": float(y_pack.get("alive_diff", 0.0)),
        "label_summoner_spells_raw": float(y_pack.get("summoner_spells", 0.0)),
        "engage_ts": int(label_start_ms),
        "obs_end_ts": int(end_ms),
        "ctx_ms": int(ctx_ms),
        "bin_ms": int(bin_ms),
        "horizon_ms": int(y_dur),
        "label_duration_ms": int(y_dur),
        "prediction_gap_ms": int(prediction_gap_ms),
        "label_start_ts": int(y_s),
        "label_end_ts": int(y_e),
        "global_snap_last_ts": int(glob_snap_ts_seq[-1]) if glob_snap_ts_seq else -1,
        "game_duration_min": float(max(1.0, (int(cache["minute_ts"][-1]) - int(cache["minute_ts"][0])) / 60000.0)),
    }

    if bool(getattr(cfg, "USE_EVENT_TOKENS", True)):
        tok = build_event_tokens_for_xattn(
            cache,
            tm,
            start_ms,
            end_ms,
            max_tokens=int(getattr(cfg, "MAX_EVENT_TOKENS", 64)),
        )
        sample.update(tok)

    anchors = cache.get("meta", {}).get("anchors", None)
    if isinstance(anchors, dict):
        sample["anchors"] = anchors
    return sample
