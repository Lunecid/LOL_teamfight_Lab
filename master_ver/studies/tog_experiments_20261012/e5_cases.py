"""T5.1 E5 case selection rules (PRESPEC_E5_KO.md section 3, PLAN_E1_E5_KO.md Task 5; SDD rulings 20, 25 and 32).

Three Q_DEV engagements are chosen by fixed rules.  "Close" means a small |p - y|.  svi = V(end) - V(start) with V
the blue win probability, so a positive svi is a move toward blue.

(a) Confidently wrong: |p_q - y| >= 0.8 and n_kills >= 2 (eval_n_kills, every kill of the component), the largest
    |svi|.  With no such engagement the threshold drops to 0.7, then 0.6; the first non-empty set is used and its
    threshold is reported.  With none at 0.6 there is no case (a).
(b) Apart from the state baseline: |p_q - y| < |p_pt - y| (strict), the largest |p_q - p_pt|.
(c) Monotonicity violation (the R10 primary definition): same frame (F_pre_ms == F_post_ms), exactly one
    team-credited kill (ka_kills = KA blue_kills + red_kills == 1), no objective change (OBJECTIVE_COUNTS equal at
    query_pre and query_post) and svi against the team that made the kill: s * svi < 0 with s = +1 for a blue kill
    and -1 for a red kill (so svi != 0); the largest |svi|.

Ties go to the ascending SHA-256 hex digest of the key text, a byte-string key decoded as ASCII (the E1 encoding).
The three cases are distinct engagements (Ruling 25): (a), (b) and (c) are chosen in that order, and a later rule
drops the engagements already chosen before it applies its own criterion and tie rule.  A rule without a remaining
candidate gives None.  The choice is deterministic (no seed).
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

A_THRESHOLDS = (0.8, 0.7, 0.6)   # rule (a): |p_q - y| thresholds, tried in this order
A_MIN_KILLS = 2                  # rule (a): eval_n_kills >= 2
NUMERIC = ("p_q", "p_pt", "y", "svi", "n_kills", "ka_kills", "kill_side_blue", "same_frame", "objective")
FLAGS = ("y", "kill_side_blue", "same_frame", "objective")   # 0 or 1


def select_cases(table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Choose the E5 cases.  Returns {"a": key, "b": key, "c": key, "a_threshold": t}.

    ``table`` holds one value per Q_DEV engagement in each column (Ruling 32): ``engagement_key``; ``p_q`` (P1) and
    ``p_pt`` (PT_flex) raw probabilities; ``y`` (direction_positive, 0/1); ``svi`` (svi_delta); ``n_kills``
    (eval_n_kills, rule a); ``ka_kills`` (KA blue_kills + red_kills, rule c); ``kill_side_blue`` (1 when KA
    blue_kills == 1); ``same_frame`` (1 when F_pre_ms == F_post_ms); ``objective`` (1 when OBJECTIVE_COUNTS differ
    between query_pre and query_post).

    "a", "b" and "c" map to the chosen engagement key exactly as it appears in the table (bytes for a byte-string key
    column, str for a text column, so it can be matched against the table without conversion), or to None when the
    rule has no remaining candidate.  "a_threshold" is the threshold whose set gave case (a): 0.8, 0.7 or 0.6, or None
    when there is no case (a).  Distances are computed in float64 from the given values and ties are exact equality.
    A table that lacks a column raises KeyError; columns of unequal length, non-finite values, flags other than 0/1,
    probabilities outside [0, 1] or repeated keys raise ValueError; a key that is neither str nor bytes, TypeError.
    """

    keys, texts, col = _read(table)
    p_q, p_pt, y, svi = col["p_q"], col["p_pt"], col["y"], col["svi"]
    magnitude = np.abs(svi)
    cases: dict[str, Any] = {"a": None, "b": None, "c": None, "a_threshold": None}
    chosen: list[int] = []

    wrong = np.abs(p_q - y)
    for threshold in A_THRESHOLDS:                       # (a) is chosen first, so nothing is excluded yet
        row = _pick((wrong >= threshold) & (col["n_kills"] >= A_MIN_KILLS), magnitude, texts, chosen)
        if row is not None:
            cases["a"], cases["a_threshold"] = keys[row], threshold
            chosen.append(row)
            break

    sign = np.where(col["kill_side_blue"] == 1, 1.0, -1.0)
    rules = (("b", wrong < np.abs(p_pt - y), np.abs(p_q - p_pt)),
             ("c", (col["same_frame"] == 1) & (col["ka_kills"] == 1) & (col["objective"] == 0) & (sign * svi < 0),
              magnitude))
    for name, candidate, score in rules:
        row = _pick(candidate, score, texts, chosen)
        if row is not None:
            cases[name] = keys[row]
            chosen.append(row)
    return cases


def _pick(candidate: np.ndarray, score: np.ndarray, texts: list[str], chosen: list[int]) -> int | None:
    """Row of the candidate with the largest score, ties by ascending key hash; rows in ``chosen`` are not candidates."""

    candidate = candidate.copy()
    candidate[np.asarray(chosen, dtype=np.intp)] = False
    rows = np.flatnonzero(candidate)
    if rows.size == 0:
        return None
    tied = rows[score[rows] == score[rows].max()]
    return min(tied.tolist(), key=lambda row: hashlib.sha256(texts[row].encode("utf-8")).hexdigest())


def _key_text(key: Any) -> str:
    """The text a key is hashed by: a byte-string key decoded as ASCII, a str key as it is."""

    if isinstance(key, bytes):
        return key.decode("ascii")
    if isinstance(key, str):
        return key
    raise TypeError(f"engagement keys must be str or bytes, got {type(key).__name__}")


def _read(table: Mapping[str, np.ndarray]) -> tuple[list[Any], list[str], dict[str, np.ndarray]]:
    """The keys (plain Python objects), their hash texts and the numeric columns as float64, after the input checks."""

    keys = np.asarray(table["engagement_key"])
    col = {name: np.asarray(table[name], dtype=np.float64) for name in NUMERIC}
    if keys.ndim != 1 or any(values.shape != keys.shape for values in col.values()):
        shapes = {"engagement_key": keys.shape, **{name: values.shape for name, values in col.items()}}
        raise ValueError(f"every column needs one value per engagement, got shapes {shapes}")
    bad = [name for name, values in col.items() if not np.isfinite(values).all()]
    if bad:
        raise ValueError(f"non-finite values in {bad}")
    bad = [name for name in FLAGS if not np.isin(col[name], (0.0, 1.0)).all()]
    if bad:
        raise ValueError(f"{bad} must hold 0 or 1 only")
    bad = [name for name in ("p_q", "p_pt") if ((col[name] < 0.0) | (col[name] > 1.0)).any()]
    if bad:
        raise ValueError(f"{bad} must lie in [0, 1]")
    key_list = keys.tolist()
    texts = [_key_text(key) for key in key_list]
    if len(set(texts)) != len(texts):
        raise ValueError("engagement keys are not unique")
    return key_list, texts, col
