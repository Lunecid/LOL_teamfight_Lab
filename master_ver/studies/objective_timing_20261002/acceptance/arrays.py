"""Pure-array independent query verification (no production query/aggregate helpers).

Adapted from reports/.../objective_timing_review_20261002/verify_source_and_queries.py.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .. import config as C

QUERY_FIELDS = (
    "status",
    "t_ms",
    "delay_ms",
    "event_ord",
    "team",
    "n_tie_events",
    "tie_category_mask",
    "tie_teams",
)

COMPONENT_IDENTITY_KEYS = ("match_local", "component_id", "L")


def expected_first_event_queries(
    components: Mapping[str, np.ndarray],
    events: Mapping[str, np.ndarray],
    *,
    categories: Sequence[str] = C.CATEGORY_ORDER,
    intervals: Sequence[str] = (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT),
) -> tuple[dict[str, np.ndarray], list[Any]]:
    """Independent global composite-time first-event calculation for all category×interval slices.

    Returns (expected flat query arrays keyed like production component_queries.npz,
    and a list of structural failures found while building expectations).
    """
    failures: list[Any] = []
    if not np.all(components["game_end_known"]):
        failures.append(["unknown_game_end_requires_separate_audit"])
    if np.any(
        (events["acq_status"] == C.ACQ_UNKNOWN)
        | (events["acq_status"] == C.ACQ_CONTRADICTORY)
    ):
        failures.append(["unknown_acquisition_requires_separate_audit"])

    span = (
        int(
            max(
                int(events["t_ms"].max()) if events["t_ms"].size else 0,
                int(components["game_end_ms"].max()) if components["game_end_ms"].size else 0,
                int(components["L"].max()) if components["L"].size else 0,
            )
        )
        + 1
    )
    ml = np.asarray(components["match_local"], dtype=np.int64)
    L = np.asarray(components["L"], dtype=np.int64)
    ge = np.asarray(components["game_end_ms"], dtype=np.int64)
    target = ml * span + L
    hi = np.minimum(
        ge,
        np.minimum(
            np.where(components["next_kill_after_L"] >= 0, components["next_kill_after_L"], ge),
            np.where(components["next_component_tau"] >= 0, components["next_component_tau"], ge),
        ),
    )

    expected: dict[str, np.ndarray] = {
        "match_local": ml.copy(),
        "component_id": np.asarray(components["component_id"]).copy(),
        "L": L.copy(),
    }

    for bit, cat in enumerate(categories):
        idx = np.flatnonzero(
            (events["acq_status"] == C.ACQ_CONFIRMED)
            & ((events["category_mask"] & (1 << bit)) != 0)
        )
        key = events["match_local"][idx].astype(np.int64) * span + events["t_ms"][idx]
        if key.size and np.any(np.diff(key) < 0):
            failures.append([cat, "event_order"])
        unique, starts, counts = np.unique(key, return_index=True, return_counts=True)
        rep = idx[starts] if len(starts) else idx
        masks = (
            np.bitwise_or.reduceat(events["category_mask"][idx], starts)
            if len(starts)
            else np.array([], dtype=np.int64)
        )
        teams = np.where(
            events["team"][idx] == 100,
            2,
            np.where(events["team"][idx] == 200, 4, 1),
        )
        team_bits = (
            np.bitwise_or.reduceat(teams, starts)
            if len(starts)
            else np.array([], dtype=np.int64)
        )
        pos = np.searchsorted(unique, target, side="right")
        safe = np.minimum(pos, max(0, len(rep) - 1))
        exists = pos < len(rep)
        if len(rep):
            exists = exists & (events["match_local"][rep[safe]] == ml)

        for interval in intervals:
            valid = exists.copy()
            no_interval = (
                (hi <= L)
                if interval == C.INTERVAL_PRE_NEXT
                else np.zeros(L.size, dtype=bool)
            )
            if len(rep):
                if interval == C.INTERVAL_PRE_NEXT:
                    valid = valid & (events["t_ms"][rep[safe]] < hi)
                else:
                    valid = valid & (events["t_ms"][rep[safe]] <= ge)
            valid = valid & ~no_interval
            status = np.where(
                no_interval,
                C.ST_NO_INTERVAL,
                np.where(valid, C.ST_OBSERVED, C.ST_NO_ACQ),
            )
            block: dict[str, np.ndarray] = {"status": status.astype(np.int64)}
            for field in ("t_ms", "event_ord", "team"):
                arr = np.full(
                    L.size,
                    0 if field == "team" else C.SENTINEL_MISSING,
                    dtype=np.int64,
                )
                if len(rep):
                    arr[valid] = events[field][rep[safe[valid]]]
                block[field] = arr
            block["delay_ms"] = np.where(valid, block["t_ms"] - L, C.SENTINEL_MISSING)
            for field, data in (
                ("n_tie_events", counts),
                ("tie_category_mask", masks),
                ("tie_teams", team_bits),
            ):
                arr = np.zeros(L.size, dtype=np.int64)
                if len(rep):
                    arr[valid] = data[safe[valid]]
                block[field] = arr
            for field in QUERY_FIELDS:
                expected[f"{field}__{interval}:{cat}"] = block[field]
    return expected, failures


def compare_query_arrays(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    categories: Sequence[str] = C.CATEGORY_ORDER,
    intervals: Sequence[str] = (C.INTERVAL_UNCAPED, C.INTERVAL_PRE_NEXT),
    fields: Sequence[str] = QUERY_FIELDS,
) -> dict[str, Any]:
    """Compare persisted query arrays to independently computed expectations."""
    failures: list[Any] = []
    slice_checks: list[dict[str, Any]] = []
    for cat in categories:
        for interval in intervals:
            mismatch = {
                field: int(
                    np.count_nonzero(
                        np.asarray(actual[f"{field}__{interval}:{cat}"])
                        != np.asarray(expected[f"{field}__{interval}:{cat}"])
                    )
                )
                for field in fields
            }
            slice_checks.append(
                {"category": cat, "interval": interval, "mismatches": mismatch}
            )
            if any(mismatch.values()):
                failures.append([cat, interval, mismatch])
    n = int(np.asarray(actual["L"]).size) if "L" in actual else 0
    return {
        "n_queries": n * len(categories) * len(intervals),
        "n_failures": len(failures),
        "failures_head": failures[:30],
        "slices": slice_checks,
        "all_ok": len(failures) == 0,
        "method": (
            "Independent global composite-time search; "
            "no production parse/query/aggregate helpers"
        ),
    }


def verify_component_identity(
    query: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    *,
    keys: Sequence[str] = COMPONENT_IDENTITY_KEYS,
) -> list[Any]:
    failures: list[Any] = []
    for k in keys:
        if k not in query or k not in components:
            failures.append([k, "missing_key"])
            continue
        if not np.array_equal(np.asarray(query[k]), np.asarray(components[k])):
            failures.append([k, "source_component_identity"])
    return failures


def verify_query_identity(
    components: Mapping[str, np.ndarray],
    events: Mapping[str, np.ndarray],
    query: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Full independent query identity check against persisted component_queries."""
    id_failures = verify_component_identity(query, components)
    expected, build_failures = expected_first_event_queries(components, events)
    cmp = compare_query_arrays(query, expected)
    n_failures = len(id_failures) + len(build_failures) + int(cmp["n_failures"])
    failures_head = (id_failures + build_failures + list(cmp["failures_head"]))[:30]
    return {
        "n_queries": cmp["n_queries"],
        "n_failures": n_failures,
        "failures_head": failures_head,
        "slices": cmp["slices"],
        "all_ok": n_failures == 0,
        "method": cmp["method"],
    }
