"""The adopted nominal two-minute sampler with explicit support exclusions."""
from __future__ import annotations

import hashlib

import numpy as np

SAMPLER_FORMULA = (
    "lo=120000+120000*k;hi=min(lo+119999,GAME_END_ms-1);"
    "t=lo+uint64_big_endian(SHA256(original_match_id+':'+str(k))[0:8])%(hi-lo+1)"
)
SAMPLER_FORMULA_SHA256 = hashlib.sha256(SAMPLER_FORMULA.encode()).hexdigest()


def query_support(minute_ts: np.ndarray, query_ms: int) -> tuple[int | None, list[str]]:
    ts = np.asarray(minute_ts)
    reasons = []
    if ts.ndim != 1 or ts.dtype.kind not in "iu":
        return None, ["frame_clock_not_integer_vector"]
    if not len(ts):
        return None, ["empty_frame_clock"]
    if np.any(ts < 0) or np.any(ts[1:] <= ts[:-1]):
        return None, ["frame_clock_not_nonnegative_strictly_increasing"]
    if query_ms < 0:
        reasons.append("negative_query")
    if query_ms < int(ts[0]):
        reasons.append("no_preceding_frame")
    if query_ms > int(ts[-1]):
        reasons.append("query_after_last_supported_frame")
    if reasons:
        return None, reasons
    return int(np.searchsorted(ts, query_ms, side="right") - 1), []


def sample_match_queries(original_match_id: str, game_end_ms: int, minute_ts: np.ndarray) -> list[dict]:
    if not isinstance(original_match_id, str) or not original_match_id or original_match_id != original_match_id.strip():
        raise ValueError("sampler requires the unchanged original local match ID")
    if type(game_end_ms) is not int or game_end_ms <= 0:
        raise ValueError("sampler requires a verified positive integer GAME_END timestamp")
    rows = []
    for k, lo in enumerate(range(120000, game_end_ms, 120000)):
        hi = min(lo + 119999, game_end_ms - 1)
        value = int.from_bytes(hashlib.sha256(f"{original_match_id}:{k}".encode()).digest()[:8], "big")
        nominal = lo + value % (hi - lo + 1)
        frame_index, reasons = query_support(minute_ts, nominal)
        rows.append({"bucket_id": k, "bucket_lo_ms": lo, "bucket_hi_ms": hi,
                     "nominal_query_ms": nominal,
                     "actual_query_ms": nominal if not reasons else None,
                     "query_support_valid": not reasons, "query_support_reasons": reasons,
                     "frame_index": frame_index})
    return rows
