"""Deterministic V-study partitions and match-local nominal query references.

The functions in this module only validate metadata and select rows.  They do
not fit preprocessing, models, labels, or any other state shared by matches.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

import numpy as np


ROLE_SALT = "full_study_20261001_roles_v2:"
INNER_STOP_SALT = "full_study_20261001_v_inner_v2:"
OOF_SALT = "full_study_20261001_v_oof_v2:"
N_FOLDS = 5
HASH_MODULUS = 1 << 64

_PATCHES = frozenset(("15.14", "15.15", "15.16"))
_KEY_RE = re.compile(r"[0-9a-f]{64}")
_NOMINAL_WIDTH = 7
_NOMINAL_NAMES = (
    "bucket_id",
    "bucket_lo_ms",
    "bucket_hi_ms",
    "nominal_ms",
    "actual_ms",
    "frame_index",
    "query_ref",
)


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
        raise ValueError("canonical_match_key must be a lowercase SHA256 hex64 string")
    return key


def _validate_patch(patch: str) -> str:
    if not isinstance(patch, str) or patch not in _PATCHES:
        raise ValueError("patch must be one of 15.14, 15.15, or 15.16")
    return patch


def _validate_excluded_fold(excluded_fold: int | None) -> int | None:
    if excluded_fold is None:
        return None
    if isinstance(excluded_fold, (bool, np.bool_)) or not isinstance(excluded_fold, (int, np.integer)):
        raise ValueError("excluded_fold must be None or an integer from 0 through 4")
    value = int(excluded_fold)
    if value < 0 or value >= N_FOLDS:
        raise ValueError("excluded_fold must be None or an integer from 0 through 4")
    return value


def _hash_u64(salt: str, key: str) -> int:
    """Return the first eight SHA256 digest bytes as unsigned big-endian."""
    _validate_key(key)
    return int.from_bytes(hashlib.sha256((salt + key).encode("utf-8")).digest()[:8], "big")


def _role_from_u64(value: int) -> str:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("role hash must be an unsigned 64-bit integer")
    value = int(value)
    if value < 0 or value >= HASH_MODULUS:
        raise ValueError("role hash must be an unsigned 64-bit integer")
    scaled = 5 * value
    if scaled < HASH_MODULUS:
        return "V_DEV"
    if scaled < 2 * HASH_MODULUS:
        return "V_CAL"
    if scaled < 4 * HASH_MODULUS:
        return "Q_DEV"
    return "Q_CAL"


def assigned_role(patch: str, key: str) -> str:
    """Assign the frozen patch/role partition for one canonical match key."""
    _validate_key(key)
    patch = _validate_patch(patch)
    if patch == "15.14":
        return "TRAIN"
    if patch == "15.16":
        return "TEST"
    return _role_from_u64(_hash_u64(ROLE_SALT, key))


def train_fold(key: str) -> int:
    """Return the frozen V OOF fold in ``[0, 4]``."""
    return _hash_u64(OOF_SALT, key) % N_FOLDS


def inner_stop(key: str) -> bool:
    """Return whether the match belongs to the deterministic V inner STOP set."""
    return _hash_u64(INNER_STOP_SALT, key) % 10 == 0


def context_membership(
    patch: str,
    key: str,
    excluded_fold: int | None = None,
) -> dict[str, bool]:
    """Return fit/STOP/refit/DEV/CAL/OOF-target membership for one match.

    For TRAIN, ``excluded_fold`` identifies the outer OOF fold being held out.
    The held-out fold is absent from FIT, STOP, and REFIT and is the OOF target.
    With no excluded fold, REFIT covers all TRAIN matches and TARGET is false.
    V development/calibration roles are assigned only from the 15.15 role
    partition.  Q roles and TEST are scoring populations and never enter the
    V fit, STOP, DEV, or CAL contexts.
    """
    patch = _validate_patch(patch)
    _validate_key(key)
    excluded_fold = _validate_excluded_fold(excluded_fold)

    role = assigned_role(patch, key)
    own_fold = patch == "15.14" and excluded_fold is not None and train_fold(key) == excluded_fold
    is_train = role == "TRAIN"
    available_train = is_train and not own_fold
    stop = available_train and inner_stop(key)

    return {
        "fit": bool(available_train and not stop),
        "stop": bool(stop),
        "refit": bool(available_train),
        "dev": role == "V_DEV",
        "cal": role == "V_CAL",
        "target": bool(own_fold),
    }


def _integer_array(bank: Mapping[str, Any], name: str, *, ndim: int = 1) -> np.ndarray:
    if name not in bank:
        raise ValueError(f"bank is missing {name}")
    value = np.asarray(bank[name])
    if value.ndim != ndim or value.dtype.kind not in "iu" or value.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer array with ndim={ndim}")
    return value.astype(np.int64, copy=False)


def _scalar_integer(bank: Mapping[str, Any], name: str) -> int:
    if name not in bank:
        raise ValueError(f"bank is missing {name}")
    value = np.asarray(bank[name])
    if value.ndim != 0 or value.dtype.kind not in "iu" or value.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer scalar")
    return int(value)


def nominal_reference_rows(bank: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and expose match-local nominal V query references.

    ``v_nominal`` is the producer's ``N x 7`` block:
    ``bucket_id, bucket_lo_ms, bucket_hi_ms, nominal_ms, actual_ms,
    frame_index, query_ref``.  Unsupported nominal rows retain their identity
    and have ``actual_ms``, ``frame_index``, ``query_ref``, and ``event_limit``
    equal to ``-1``.  Engagement/component queries in the bank are retained in
    the query arrays but are intentionally not added to this returned index.
    """
    if not isinstance(bank, Mapping):
        raise TypeError("bank must be a mapping")
    nominal = np.asarray(bank.get("v_nominal"))
    if nominal.ndim != 2 or nominal.shape[1] != _NOMINAL_WIDTH:
        raise ValueError("v_nominal must have shape (N, 7)")
    if nominal.dtype.kind not in "iu" or nominal.dtype.kind == "b":
        raise ValueError("v_nominal must be an integer array")
    nominal = nominal.astype(np.int64, copy=False)

    minute_ts = _integer_array(bank, "minute_ts")
    query_ms = _integer_array(bank, "query_ms")
    query_frame_index = _integer_array(bank, "query_frame_index")
    query_event_limit = _integer_array(bank, "query_event_limit")
    event_timestamp = _integer_array(bank, "event_timestamp")
    end_ms = _scalar_integer(bank, "end_ms")

    if minute_ts.size == 0 or np.any(minute_ts < 0) or np.any(minute_ts[1:] <= minute_ts[:-1]):
        raise ValueError("minute_ts must be non-empty, nonnegative, and strictly increasing")
    if end_ms <= 0:
        raise ValueError("end_ms must be positive")
    if np.any(event_timestamp < 0) or np.any(event_timestamp[1:] < event_timestamp[:-1]):
        raise ValueError("event_timestamp must be nonnegative and nondecreasing")
    for name, value in (("query_frame_index", query_frame_index), ("query_event_limit", query_event_limit)):
        if value.shape != query_ms.shape:
            raise ValueError(f"{name} must have the same length as query_ms")
    if query_ms.size and (np.any(query_ms[1:] <= query_ms[:-1]) or np.any(query_ms < minute_ts[0])
                          or np.any(query_ms > minute_ts[-1]) or np.any(query_ms >= end_ms)):
        raise ValueError("query_ms must be unique, supported by frames, and before end_ms")
    if np.any(query_frame_index < 0) or np.any(query_frame_index >= minute_ts.size):
        raise ValueError("query_frame_index is outside minute_ts")
    expected_frame = np.searchsorted(minute_ts, query_ms, side="right") - 1
    if not np.array_equal(query_frame_index, expected_frame.astype(np.int64)):
        raise ValueError("query_frame_index is not the last frame at query_ms")
    expected_event_limit = np.searchsorted(event_timestamp, query_ms, side="right").astype(np.int64)
    if not np.array_equal(query_event_limit, expected_event_limit):
        raise ValueError("query_event_limit is not the inclusive event prefix at query_ms")

    if nominal.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return {name: empty.copy() for name in _NOMINAL_NAMES[:-1]} | {
            "query_ref": empty.copy(), "supported": np.empty(0, dtype=bool), "event_limit": empty.copy()
        }

    bucket_id, bucket_lo, bucket_hi, nominal_ms, actual_ms, frame_index, query_ref = nominal.T
    if np.any(bucket_id < 0) or len(np.unique(bucket_id)) != len(bucket_id):
        raise ValueError("nominal bucket IDs must be nonnegative and unique")
    expected_lo = 120_000 * (bucket_id + 1)
    expected_hi = np.minimum(expected_lo + 119_999, end_ms - 1)
    if not np.array_equal(bucket_lo, expected_lo) or not np.array_equal(bucket_hi, expected_hi):
        raise ValueError("nominal bucket bounds do not match end_ms")
    if np.any(bucket_lo > bucket_hi) or np.any(nominal_ms < bucket_lo) or np.any(nominal_ms > bucket_hi):
        raise ValueError("nominal query is outside its bucket")

    supported = actual_ms >= 0
    if np.any(actual_ms < -1) or np.any(frame_index < -1) or np.any(query_ref < -1):
        raise ValueError("nominal unsupported sentinels must be -1")
    unsupported = ~supported
    if np.any(actual_ms[unsupported] != -1) or np.any(frame_index[unsupported] != -1) or np.any(query_ref[unsupported] != -1):
        raise ValueError("unsupported nominal rows must retain -1 references")
    if np.any(actual_ms[supported] != nominal_ms[supported]):
        raise ValueError("supported nominal actual_ms must equal nominal_ms")
    refs = query_ref[supported]
    if np.any(refs < 0) or np.any(refs >= query_ms.size) or len(np.unique(refs)) != len(refs):
        raise ValueError("supported nominal query_ref values are out of range or duplicated")
    if np.any(frame_index[supported] < 0) or np.any(frame_index[supported] >= minute_ts.size):
        raise ValueError("supported nominal frame_index is outside minute_ts")
    if refs.size:
        if not np.array_equal(query_ms[refs], actual_ms[supported]):
            raise ValueError("supported nominal query_ref does not identify actual_ms")
        if not np.array_equal(query_frame_index[refs], frame_index[supported]):
            raise ValueError("supported nominal frame reference does not identify query frame")

    event_limit = np.full(len(nominal), -1, dtype=np.int64)
    event_limit[supported] = query_event_limit[refs]
    result = {
        name: nominal[:, index].copy()
        for index, name in enumerate(_NOMINAL_NAMES)
    }
    result["supported"] = supported.copy()
    result["event_limit"] = event_limit
    return result


__all__ = [
    "assigned_role",
    "context_membership",
    "inner_stop",
    "nominal_reference_rows",
    "train_fold",
]
