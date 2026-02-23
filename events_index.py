from __future__ import annotations

from common import Any, Dict, List, Optional, np

# -----------------------------------------------------------------------------
# Event time indexing (FAST)
#
# ✅ 요구사항: "함수 이름을 동일하게" 유지
#   - _safe_int_dict
#   - _event_ts_safe
#   - _attach_event_index_inplace
#   - _events_in_window
#   - _ensure_event_time_index
#
# 개선점:
#   - timestamp 키를 ("timestamp", "ts") 모두 지원
#   - _ensure_event_time_index(): non-destructive 인덱스(정렬된 ts + 원본 idx) 구축
#   - _events_in_window(): 인덱스 있으면 binary search로 O(logN)+슬라이스, 없으면 안전 스캔
#   - legacy 호환: pack["events_ts"] 방식(_attach_event_index_inplace)도 그대로 지원
# -----------------------------------------------------------------------------

_EVENT_TS_KEYS = ("timestamp", "ts")


def _safe_int_dict(x: Any) -> Optional[Dict[int, int]]:
    if not isinstance(x, dict):
        return None
    out: Dict[int, int] = {}
    for k, v in x.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _event_ts_safe(e: dict) -> Optional[int]:
    """
    Robust timestamp extractor.
    Supports both:
      - e["timestamp"]
      - e["ts"]
    Returns int milliseconds or None.
    """
    if not isinstance(e, dict):
        return None

    for key in _EVENT_TS_KEYS:
        ts = e.get(key, None)
        if ts is None:
            continue
        try:
            t = int(ts)
            if t < 0:
                return None
            return t
        except Exception:
            continue
    return None


def _attach_event_index_inplace(pack: Dict[str, Any]) -> None:
    """
    Legacy / in-place index builder.
    - Filters events without timestamp
    - Sorts events if needed
    - Adds:
        pack["events_ts"] : np.ndarray int64 aligned with pack["events"]

    ⚠️ This mutates pack["events"] (may drop/ reorder).
    Prefer using _ensure_event_time_index + _events_in_window for non-destructive indexing.
    """
    events = pack.get("events", [])
    if not isinstance(events, list) or len(events) == 0:
        pack["events"] = []
        pack["events_ts"] = np.empty((0,), dtype=np.int64)
        return

    ts_list: List[int] = []
    ev_list: List[dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = _event_ts_safe(e)
        if ts is None:
            continue
        ev_list.append(e)
        ts_list.append(ts)

    if not ev_list:
        pack["events"] = []
        pack["events_ts"] = np.empty((0,), dtype=np.int64)
        return

    # check monotonic -> only sort if needed
    need_sort = False
    prev = ts_list[0]
    for t in ts_list[1:]:
        if t < prev:
            need_sort = True
            break
        prev = t

    if need_sort:
        order = np.argsort(np.asarray(ts_list, dtype=np.int64), kind="mergesort")
        ev_list = [ev_list[i] for i in order.tolist()]
        ts_arr = np.asarray([ts_list[i] for i in order.tolist()], dtype=np.int64)
    else:
        ts_arr = np.asarray(ts_list, dtype=np.int64)

    pack["events"] = ev_list
    pack["events_ts"] = ts_arr


def _ensure_event_time_index(cache: Dict[str, Any]) -> None:
    """
    Non-destructive fast index (built once per cache-pack and reused).

    Stores:
      - cache["_events_ts_sorted"]: int64 timestamps sorted (nondecreasing)
      - cache["_events_idx_sorted"]: int32 indices into cache["events"]
      - cache["_event_index_built"]: bool

    Notes:
      - Does NOT reorder cache["events"].
      - If original order is already nondecreasing, uses it directly (no sort).
      - If out-of-order timestamps exist, builds a stable-sorted index.
    """
    if not isinstance(cache, dict):
        return
    if bool(cache.get("_event_index_built", False)):
        return

    events = cache.get("events", [])
    if not isinstance(events, list) or len(events) == 0:
        cache["_events_ts_sorted"] = np.empty((0,), dtype=np.int64)
        cache["_events_idx_sorted"] = np.empty((0,), dtype=np.int32)
        cache["_event_index_built"] = True
        return

    ts_list: List[int] = []
    idx_list: List[int] = []

    last_ts = -1
    nondecreasing = True

    for i, e in enumerate(events):
        if not isinstance(e, dict):
            continue
        t = _event_ts_safe(e)
        if t is None:
            continue
        ts_list.append(t)
        idx_list.append(i)
        if last_ts > t:
            nondecreasing = False
        last_ts = t

    if len(ts_list) == 0:
        cache["_events_ts_sorted"] = np.empty((0,), dtype=np.int64)
        cache["_events_idx_sorted"] = np.empty((0,), dtype=np.int32)
        cache["_event_index_built"] = True
        return

    if nondecreasing:
        cache["_events_ts_sorted"] = np.asarray(ts_list, dtype=np.int64)
        cache["_events_idx_sorted"] = np.asarray(idx_list, dtype=np.int32)
        cache["_event_index_built"] = True
        return

    # stable sort by timestamp if out-of-order exists
    order = np.argsort(np.asarray(ts_list, dtype=np.int64), kind="mergesort")
    cache["_events_ts_sorted"] = np.asarray(ts_list, dtype=np.int64)[order]
    cache["_events_idx_sorted"] = np.asarray(idx_list, dtype=np.int32)[order]
    cache["_event_index_built"] = True


def _events_in_window(pack: Dict[str, Any], s_ms: int, e_ms: int) -> List[dict]:
    """
    Return events with s_ms <= timestamp < e_ms using the fastest available method.

    Priority:
      1) Non-destructive cache index: pack["_events_ts_sorted"] + pack["_events_idx_sorted"]
         (built by _ensure_event_time_index)
      2) Legacy in-place index: pack["events_ts"] aligned with pack["events"]
         (built by _attach_event_index_inplace)
      3) Fallback scan (safe, slower)

    Output order:
      - If method (1) is used: chronological by timestamp (sorted index)
      - If method (2) is used: chronological by pack["events_ts"]
      - If fallback scan: original order filtered
    """
    events = pack.get("events", [])
    if not isinstance(events, list) or len(events) == 0:
        return []

    s_ms = int(s_ms)
    e_ms = int(e_ms)
    if e_ms <= s_ms:
        return []

    # (1) Preferred: non-destructive index
    _ensure_event_time_index(pack)
    ts_sorted = pack.get("_events_ts_sorted", None)
    idx_sorted = pack.get("_events_idx_sorted", None)

    if (
        isinstance(ts_sorted, np.ndarray)
        and isinstance(idx_sorted, np.ndarray)
        and ts_sorted.dtype == np.int64
        and idx_sorted.dtype == np.int32
        and ts_sorted.size == idx_sorted.size
        and ts_sorted.size > 0
    ):
        i0 = int(np.searchsorted(ts_sorted, s_ms, side="left"))
        i1 = int(np.searchsorted(ts_sorted, e_ms, side="left"))
        if i1 <= i0:
            return []
        out: List[dict] = []
        for j in idx_sorted[i0:i1].tolist():
            try:
                ev = events[int(j)]
            except Exception:
                continue
            if isinstance(ev, dict):
                out.append(ev)
        return out

    # (2) Legacy: in-place events_ts aligned with events
    ts = pack.get("events_ts", None)
    if isinstance(ts, np.ndarray) and ts.dtype == np.int64 and ts.size == len(events) and ts.size > 0:
        i0 = int(np.searchsorted(ts, s_ms, side="left"))
        i1 = int(np.searchsorted(ts, e_ms, side="left"))
        if i1 <= i0:
            return []
        # already aligned with events
        return [e for e in events[i0:i1] if isinstance(e, dict)]

    # (3) Fallback scan (safe)
    out: List[dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        t = _event_ts_safe(e)
        if t is None:
            continue
        if s_ms <= t < e_ms:
            out.append(e)
    return out
