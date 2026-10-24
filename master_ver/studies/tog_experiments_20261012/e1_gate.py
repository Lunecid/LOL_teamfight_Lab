"""T1.1 equality gate, extra checks: rows built at new times reproduce the stored q inputs (E1 and E5 rest on it).

E1 (kill-free placebo windows) and E5 (case time axes) build model inputs at times that are not stored queries,
through the control-state builder (``controls_states.build_match`` -> ``semantic_rows``).  Before those rows are
trusted:

1. The frozen gate (``controls_states.py --equality-gate``, run separately into ``<out>/gate``) passes its four
   checks (controls_states.py lines 51-54).  It samples at most about a dozen Q_DEV control matches.
2. Per q role (TRAIN, Q_DEV, Q_CAL, TEST), the 500 stored engagement pre queries with the smallest
   SHA-256(f"{SALT}:{query}") among the rows of q_run_1/Q_ROWS_h90.npz.  At each, the semantic row is rebuilt at
   the stored time tau through the new-time path, and
   - it equals QuerySurface's row for that query (the frozen ``controls_states.Comparison``: types, key order,
     dtypes, bytes);
   - written with ``features.RawStoreWriter``, it equals the same row of q_run_1/raw/store (numbers bitwise; bag
     entries in stored order, item ids as integers and non-item tokens by their text, since the two token
     dictionaries number tokens differently);
   - its F(tau) XY, read at the rebuilt held frame by spatial.py's rules, equals the same row of
     q_run_1/inputs/main_h90/XY.npy.
3. Per q role, the first 3 q-row matches in SHA-256(f"{SALT}:match:{match}") order whose stretch after the last
   stored query holds at least one non-item history token and one item event (the frozen ``_gap_counts``) go
   through ``controls_states.gate_match`` (all four frozen checks) with three new times: the midpoint of the
   widest gap between stored queries, the midpoint of (last stored query, limit] and the limit, the latest
   admissible time (at most the last frame and before the game end, controls.py:104).  The frozen gate samples
   Q_DEV control matches only; E1 builds TRAIN and TEST states too.

Writes ``<out>/EQUALITY_GATE_TOG.json`` once (mismatch counts per check and per role, 0 expected; any mismatch
blocks E1 and E5) and the evidence under ``<out>/gate_tog``: raw_store/ (the rebuilt rows), XY_REBUILT.npz,
gate_roundtrip/.  Of the row table only query_pre, pre_ms, match_index and role are read (no label column).
One process at below-normal priority.

Usage, after the frozen gate:
  python -B studies/tog_experiments_20261012/e1_gate.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import controls_states as cs  # noqa: E402
from studies.label_production_20261005.queries import PROJECTION as LABEL_PROJECTION, SHARED_BANK, QuerySurface  # noqa: E402
from studies.q_production_20261005.features import LANE_SLOTS, RawStore, RawStoreWriter  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES  # noqa: E402
from studies.q_production_20261005.spatial import N_PLAYERS  # noqa: E402
from studies.tog_experiments_20261012.common import LABELS, Q_RUN, TOG_E1, run_record, utc, write_once  # noqa: E402
from studies.v_model_fit_input_20261003.common import BAG_NAMESPACES  # noqa: E402

VERSION = "tog_e1_gate_v1_20261012"
SALT = "tog_e1_gate_20261012"
PER_ROLE = 500                 # sampled stored engagement pre queries per q role
AFTER_LAST_PER_ROLE = 3        # matches per q role checked with new times after their last stored query
EXAMPLES = 10                  # example keys kept per differing field
RESULT_NAME, WORK_NAME, GATE_DIR = "EQUALITY_GATE_TOG.json", "gate_tog", "gate"
ROLE_NAMES = {code: name for name, code in Q_ROLES.items()}          # 0 TRAIN, 3 Q_DEV, 4 Q_CAL, 5 TEST
GATE_COMPARISONS = ("query_arrays", "features", "history_text", "rows", "prefix", "roundtrip")   # gate_match's
FROZEN_CHECKS = (("stored_rows_equal_query_surface", ("query_arrays", "features", "history_text", "rows")),
                 ("unchanged_with_new_times", ("prefix",)),
                 ("sidecar_roundtrip", ("roundtrip",)))
Q_FILES = ("studies/q_production_20261005/features.py", "studies/q_production_20261005/spatial.py",
           "studies/q_production_20261005/qtest.py", "studies/q_production_20261005/rows.py")


def _digest(text: str) -> str:
    return hashlib.sha256(f"{SALT}:{text}".encode()).hexdigest()


def sample_queries(query_pre: np.ndarray, role: np.ndarray, *, per_role: int = PER_ROLE) -> dict[int, np.ndarray]:
    """Per role, the ``per_role`` queries with the smallest SHA-256(f"{SALT}:{query}"), in ascending query order.

    The choice depends only on the set of (query, role) pairs, not on their order.  A role with fewer queries,
    or a query listed under two roles, is refused.
    """

    queries, roles = np.asarray(query_pre, dtype=np.int64), np.asarray(role, dtype=np.int64)
    if queries.ndim != 1 or queries.shape != roles.shape:
        raise ValueError("one role per query")
    pairs = np.unique(np.stack((queries, roles), axis=1), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs):
        raise ValueError("a query is listed under more than one role")
    sample = {}
    for code in np.unique(pairs[:, 1]).tolist():
        own = pairs[pairs[:, 1] == code, 0]
        if len(own) < per_role:
            raise ValueError(f"role {code} has fewer than {per_role} queries")
        digests = np.asarray([_digest(str(query)) for query in own.tolist()])
        sample[int(code)] = np.sort(own[np.argsort(digests, kind="stable")[:per_role]])
    return sample


def field_mismatches(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per field (``expected``'s order, then fields only ``actual`` has): bool [rows], True where the row differs.

    Numbers compare as bytes, so -0.0 differs from 0.0 and a NaN equals only the same NaN; object arrays compare
    by value.  A field missing on one side, or with another dtype or row shape, differs on every row.
    """

    lengths = {len(values) for values in (*actual.values(), *expected.values())}
    if len(lengths) != 1:
        raise ValueError("the fields need one common row count")
    n = lengths.pop()
    out = {}
    for name in [*expected, *(name for name in actual if name not in expected)]:
        a, b = actual.get(name), expected.get(name)
        if a is None or b is None or a.dtype != b.dtype or a.shape != b.shape:
            out[name] = np.ones(n, dtype=bool)
        elif a.dtype == object:
            out[name] = np.fromiter((left != right for left, right in zip(a, b)), dtype=bool, count=n)
        else:
            width = int(np.prod(a.shape[1:], dtype=np.int64)) * a.dtype.itemsize
            out[name] = (np.ascontiguousarray(a).view(np.uint8).reshape(n, width)
                         != np.ascontiguousarray(b).view(np.uint8).reshape(n, width)).any(axis=1)
    return out


def mismatch_report(differs: Mapping[str, np.ndarray], roles: np.ndarray, keys: np.ndarray | None = None) -> dict[str, Any]:
    """Differing rows per q role and field, the fields that differ anywhere, and up to 10 example keys per field."""

    roles = np.asarray(roles, dtype=np.int64)
    if not np.isin(roles, list(ROLE_NAMES)).all():
        raise ValueError("a row lies outside the q roles")
    keys = np.arange(len(roles)) if keys is None else np.asarray(keys)
    any_field = np.zeros(len(roles), dtype=bool)
    for flags in differs.values():
        any_field |= flags
    per_role = {}
    for code, name in ROLE_NAMES.items():
        own = roles == code
        per_role[name] = {"rows": int(own.sum()), "mismatched_rows": int((any_field & own).sum()),
                          "fields": {field: int((flags & own).sum()) for field, flags in differs.items() if (flags & own).any()}}
    return {"fields_compared": list(differs), "rows": int(len(roles)), "mismatched_rows": int(any_field.sum()),
            "mismatched_fields": [field for field, flags in differs.items() if flags.any()], "roles": per_role,
            "examples": {field: keys[flags][:EXAMPLES].tolist() for field, flags in differs.items() if flags.any()}}


def store_fields(store: RawStore, rows: np.ndarray) -> dict[str, np.ndarray]:
    """The q input fields of raw-store rows, one entry per row on axis 0.

    Each bag becomes one JSON text per row: slots in order, entries in stored order as [value, count], item ids as
    integers and non-item codes as their dictionary text, so stores with different dictionaries compare exactly.
    """

    rows = np.asarray(rows, dtype=np.int64)
    fields = {f"numeric:{lane}": np.asarray(store.numeric[lane][rows]) for lane in ("node", "team", "global")}
    fields.update({"static:node": np.asarray(store.static[rows]), "bans:team": np.asarray(store.bans[rows]),
                   "team_ids": np.asarray(store.team_ids[rows]), "current_item_supported": np.asarray(store.supported[rows]),
                   "focal_is_blue": np.asarray(store.focal_blue[rows])})
    for lane, group in store.offsets:
        codes, counts, lengths = store.gather((lane, group), rows)
        values = [store.tokens[code] for code in codes.tolist()] if BAG_NAMESPACES[group] == "nonitem" else codes.tolist()
        entries = [[value, count] for value, count in zip(values, counts.tolist())]
        bounds = np.concatenate(([0], np.cumsum(lengths))).tolist()
        slots = LANE_SLOTS[lane]
        texts = np.empty(len(rows), dtype=object)
        for row in range(len(rows)):
            texts[row] = json.dumps([entries[bounds[row * slots + slot]:bounds[row * slots + slot + 1]] for slot in range(slots)],
                                    ensure_ascii=False, separators=(",", ":"))
        fields[f"bag:{lane}:{group}"] = texts
    return fields


def compare_rows(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[list[str], int, int]:
    """The frozen gate's exact comparison of two semantic rows: (fields that differ, leaves and elements compared).

    A field is the first three levels of a differing path without list positions (``numeric/node``,
    ``bags/node/nonitem``, ``team_ids``); a difference at the top level is ``row``.
    """

    comparison = cs.Comparison()
    comparison.check(actual, expected)
    fields = set()
    for message in comparison.mismatches:
        parts = [re.sub(r"(\[\d+\])+$", "", part) for part in message.split(": ", 1)[0].split("/")[1:4]]
        fields.add("/".join(parts) or "row")
    return sorted(fields), comparison.leaves, comparison.elements


def xy_at_times(view: Mapping[str, np.ndarray], times: Sequence[int], frame_index: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """F(t) XY of one match at any times, by ``spatial.BankXYReader.read``'s rules (spatial.py:220-249).

    ``frame_index`` is the held frame of each time as the observable state reports it (``query_frame_index`` of
    ``controls_states.build_match``); it must be the latest frame <= t (no future frame).  A time without a held
    frame, or after the last frame, is unsupported and keeps NaN; coordinates are the recorded values as float64,
    nothing is imputed.  Returns (xy [n, 10, 2] float64, support [n]: held frame, inside the frames, all finite).
    """

    t, fi = np.asarray(times, dtype=np.int64), np.asarray(frame_index, dtype=np.int64)
    clock, frames = np.asarray(view["minute_ts"], dtype=np.int64), np.asarray(view["xy"])
    n_ts = len(clock)
    if t.ndim != 1 or t.shape != fi.shape:
        raise ValueError("one frame index per time")
    if len(frames) != n_ts or n_ts < 1 or frames.shape[1:] != (N_PLAYERS, 2):
        raise ValueError("bank xy frames and frame clock differ")
    if (fi >= n_ts).any() or (fi < -1).any():
        raise ValueError("frame index outside the match frames")
    held = fi >= 0
    safe = np.where(held, fi, 0)
    later = safe + 1 < n_ts
    if (held & (clock[safe] > t)).any():
        raise ValueError("held frame lies after the time (future frame)")
    if (held & later & (clock[np.minimum(safe + 1, n_ts - 1)] <= t)).any():
        raise ValueError("held frame is not the latest frame <= time")
    if (~held & (clock[0] <= t)).any():
        raise ValueError("a frame precedes a time recorded without a held frame")
    supported = held & (t <= clock[-1])
    xy = np.full((len(t), N_PLAYERS, 2), np.nan, dtype=np.float64)
    xy[supported] = frames[fi[supported]].astype(np.float64)
    return xy, supported & np.isfinite(xy).all(axis=(1, 2))


def time_limit(minute_ts: np.ndarray, end_ms: np.ndarray) -> int:
    """Latest admissible new time of a match: at most its last frame and before the game end (controls.py:104)."""

    frames, end = np.asarray(minute_ts, dtype=np.int64), np.asarray(end_ms, dtype=np.int64).reshape(-1)
    if not len(frames):
        return -1
    return min(int(frames[-1]), int(end[0]) - 1) if len(end) and int(end[0]) >= 0 else int(frames[-1])


def new_times(stored_ms: Sequence[int], limit_ms: int) -> list[int]:
    """New times of one match: the midpoint of its widest gap between stored queries (if any), the midpoint of
    (last stored query, limit] and the limit; none without room after the last stored query."""

    stored = np.asarray(stored_ms, dtype=np.int64)
    last, limit = int(stored[-1]), int(limit_ms)
    if limit - last < 2:
        return []
    gaps = np.diff(stored)
    inner = [int(stored[np.argmax(gaps)] + gaps.max() // 2)] if len(gaps) and gaps.max() >= 2 else []
    return inner + [last + (limit - last) // 2, limit]


def frozen_checks(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The frozen gate's four checks (controls_states.py lines 51-54) from its comparison totals and match reports.

    The frozen gate records a failed check 4 (an end after the last stored query whose history change does not
    follow the events in between) inside ``prefix``; it is moved to its own check here.  A check that compared
    nothing does not pass.
    """

    totals = result["totals"]
    ends = [row for report in result["reports"] for row in report["after_last_stored_query"]]
    inconsistent = sum(not row["consistent"] for row in ends)
    checks = {}
    for name, parts in FROZEN_CHECKS:
        leaves = sum(int(totals[part]["leaves_compared"]) for part in parts)
        mismatches = sum(int(totals[part]["mismatches"]) for part in parts) - (inconsistent if name == "unchanged_with_new_times" else 0)
        checks[name] = {"comparisons": list(parts), "leaves_compared": leaves, "mismatches": mismatches,
                        "passed": leaves > 0 and mismatches == 0}
    checks["after_last_stored_query_events"] = {"ends_checked": len(ends), "mismatches": inconsistent,
                                                "passed": bool(ends) and inconsistent == 0}
    return checks


def after_last_checks(surface: QuerySurface, projection: cs.Projection, match_index: np.ndarray, role: np.ndarray,
                      work: Path, categories: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``gate_match`` reports for the first ``AFTER_LAST_PER_ROLE`` q-row matches per role in
    SHA-256(f"{SALT}:match:{match}") order whose stretch after the last stored query, up to the limit, holds a
    non-item history token and an item event; and per role how many matches were examined and skipped."""

    first, count = projection.match_queries["first_query"], projection.match_queries["n_queries"]
    reports: list[dict[str, Any]] = []
    record: dict[str, Any] = {}
    bank = None
    for code, name in ROLE_NAMES.items():
        tally = {"examined": 0, "without_room": 0, "without_gap_events": 0, "matches": []}
        for match in sorted(np.unique(match_index[role == code]).tolist(), key=lambda value: _digest(f"match:{value}")):
            if len(tally["matches"]) == AFTER_LAST_PER_ROLE:
                break
            tally["examined"] += 1
            shard = int(projection.matches["shard"][match])
            if bank is None or bank.shard_index != shard:
                bank = None
                bank = cs.BankShard(shard, projection.binding(shard), projection.source_root)
            source = bank.source(int(projection.matches["local_match"][match]), match, projection.key(match))
            stored = np.asarray(projection.queries["query_ms"][int(first[match]):int(first[match]) + int(count[match])], dtype=np.int64)
            limit = time_limit(source.view["minute_ts"], source.view["end_ms"])
            times = new_times(stored, limit)
            if not times:
                tally["without_room"] += 1
                continue
            _records, tokens, items = cs._gap_counts(source, int(stored[-1]), limit)
            if not tokens or not items:
                tally["without_gap_events"] += 1
                continue
            reports.append({"role": name, "time_limit_ms": limit, "times_ms": times}
                           | cs.gate_match(surface, projection, source, times, work, categories))
            tally["matches"].append(match)
        record[name] = tally
    return reports, record


def _input_files(gate: Path, store_dir: Path, projection_root: Path) -> list[Path]:
    files = [Q_RUN / "Q_ROWS_h90.npz", Q_RUN / "inputs" / "main_h90" / "XY.npy"]
    files += sorted(path for path in store_dir.iterdir() if path.is_file())
    files += sorted(path for path in gate.rglob("*") if path.is_file())
    files += [projection_root / "RUN_MANIFEST.json", projection_root / "BANK_METADATA.json",
              SHARED_BANK / "PLAN.json", SHARED_BANK / "ACCEPTANCE.json"]
    return files


def run(out: Path = TOG_E1) -> dict[str, Any]:
    """The extra checks, combined with the frozen gate's result into ``EQUALITY_GATE_TOG.json`` (written once)."""

    started, tick = utc(), time.perf_counter()
    out = Path(out)
    target, work, gate = out / RESULT_NAME, out / WORK_NAME, out / GATE_DIR
    if target.exists() or work.exists():
        raise FileExistsError(f"{target} or {work} exists; the gate is written once")
    frozen = json.loads((gate / f"{cs.GATE_NAME}.json").read_text(encoding="utf-8"))
    if Path(frozen["inputs"]["matching_npz"]) != LABELS / f"{cs.MATCHING}.npz":
        raise ValueError("the frozen gate result was not built from labels_1")
    rows_path, xy_path, store_dir = Q_RUN / "Q_ROWS_h90.npz", Q_RUN / "inputs" / "main_h90" / "XY.npy", Q_RUN / "raw" / "store"
    with np.load(rows_path, allow_pickle=False) as table:            # identity columns only: no label column is read
        q_rows = {name: table[name].astype(np.int64) for name in ("query_pre", "pre_ms", "match_index", "role")}
    if np.any(np.diff(q_rows["query_pre"]) <= 0):
        raise ValueError("Q_ROWS_h90 rows are not in strictly increasing query order")
    projection = cs.Projection()
    if projection.root.resolve() != Path(LABEL_PROJECTION).resolve():
        raise ValueError("the gate projection differs from the label projection")
    projection_sha256 = projection.file_hashes()

    sample = sample_queries(q_rows["query_pre"], q_rows["role"])
    queries = np.sort(np.concatenate(list(sample.values())))
    rows = np.searchsorted(q_rows["query_pre"], queries)
    roles = q_rows["role"][rows]
    match_index = np.asarray(projection.queries["match_index"][queries], dtype=np.int64)
    query_ms = np.asarray(projection.queries["query_ms"][queries], dtype=np.int64)
    if not np.array_equal(match_index, q_rows["match_index"][rows]) or not np.array_equal(query_ms, q_rows["pre_ms"][rows]):
        raise ValueError("the sampled queries are not their rows' pre queries")
    n = len(queries)
    starts = np.flatnonzero(np.r_[True, match_index[1:] != match_index[:-1]])
    segments = {int(match_index[a]): (int(a), int(b)) for a, b in zip(starts, np.r_[starts[1:], n])}
    if len(segments) != len(starts):
        raise ValueError("a match's sampled queries are not contiguous")

    frame = np.full(n, -1, dtype=np.int32)
    xy = np.full((n, N_PLAYERS, 2), np.nan)
    support = np.zeros(n, dtype=bool)
    row_fields: list[list[str]] = [[] for _ in range(n)]
    leaves = elements = 0
    writer = RawStoreWriter(work / "raw_store", queries)
    surface = QuerySurface(SHARED_BANK, LABEL_PROJECTION, seed=cs.REFERENCE_SEED, verify_projection_hashes=False)
    bank, shards = None, []
    try:
        for visited, (match, (begin, end)) in enumerate(segments.items(), start=1):     # ascending query order
            shard = int(projection.matches["shard"][match])
            if bank is None or bank.shard_index != shard:
                bank = None
                bank = cs.BankShard(shard, projection.binding(shard), projection.source_root)
                shards.append(shard)
            source = bank.source(int(projection.matches["local_match"][match]), match, projection.key(match))
            times = query_ms[begin:end].tolist()
            build = cs.build_match(source, times)
            rebuilt = cs.semantic_rows(build.states)
            for offset, (query, row, meta) in enumerate(surface.iter_query_rows(queries[begin:end])):
                if (query, meta["query_ms"], meta["role"]) != (int(queries[begin + offset]), times[offset], ROLE_NAMES[int(roles[begin + offset])]):
                    raise ValueError(f"QuerySurface row identity differs for query {query}")
                fields, leaf_count, element_count = compare_rows(rebuilt[offset], row)
                row_fields[begin + offset], leaves, elements = fields, leaves + leaf_count, elements + element_count
                writer.add(query, rebuilt[offset])
            frame[begin:end] = build.query["query_frame_index"]
            xy[begin:end], support[begin:end] = xy_at_times(source.view, times, build.query["query_frame_index"])
            if visited % 100 == 0:
                print(json.dumps({"matches_visited": visited, "of": len(segments), "shards_loaded": len(shards),
                                  "seconds": round(time.perf_counter() - tick, 1)}), flush=True)
        bank = None
        gate_reports, after_record = after_last_checks(surface, projection, q_rows["match_index"], q_rows["role"], work,
                                                       {"engagement_pre": q_rows["query_pre"]})
    finally:
        surface.close()
    writer.close()
    del writer
    np.savez(work / "XY_REBUILT.npz", query_index=queries, role=roles, frame_index=frame, xy=xy, support=support)

    names = sorted({name for fields in row_fields for name in fields})
    differs = {"whole_row": np.asarray([bool(fields) for fields in row_fields])}
    differs.update({name: np.asarray([name in fields for fields in row_fields]) for name in names})
    semantic = mismatch_report(differs, roles, queries) | {"leaves_compared": leaves, "elements_compared": elements}
    stored = RawStore(store_dir)
    raw = mismatch_report(field_mismatches(store_fields(RawStore(work / "raw_store"), np.arange(n)),
                                           store_fields(stored, stored.positions(queries))), roles, queries)
    stored_xy = np.load(xy_path, mmap_mode="r", allow_pickle=False)
    xy_report = mismatch_report(field_mismatches(
        {"frame_index": frame, "xy": xy, "xy_support": support},
        {"frame_index": np.asarray(projection.queries["frame_index"][queries]), "xy": np.asarray(stored_xy[rows]),
         "xy_support": np.ones(n, dtype=bool)}), roles, queries)

    after: dict[str, Any] = {}
    for name in ROLE_NAMES.values():
        own = [report for report in gate_reports if report["role"] == name]
        totals = {part: {"leaves_compared": sum(report["checks"][part]["leaves_compared"] for report in own),
                         "mismatches": sum(report["checks"][part]["mismatches"] for report in own)} for part in GATE_COMPARISONS}
        checks = frozen_checks({"totals": totals, "reports": own})
        ends = [row for report in own for row in report["after_last_stored_query"]]
        after[name] = {**after_record[name], "checks": checks, "ends_with_nonitem_events": sum(row["gap_nonitem_tokens"] > 0 for row in ends),
                       "ends_with_item_events": sum(row["gap_item_events"] > 0 for row in ends),
                       "passed": len(own) == AFTER_LAST_PER_ROLE and all(report["passed"] for report in own)
                       and all(check["passed"] for check in checks.values())}

    frozen_roles = np.asarray(projection.matches["role"])[np.asarray(frozen["matches"], dtype=np.int64)]
    frozen_summary = {"file": str(gate / f"{cs.GATE_NAME}.json"), "passed": bool(frozen["passed"]), "checks": frozen_checks(frozen),
                      "matches": frozen["matches"], "stored_rows_compared": frozen["stored_rows_compared"],
                      "control_end_rows": frozen["control_end_rows"], "seed": frozen["seed"],
                      "match_roles": {ROLE_NAMES.get(int(code), str(int(code))): int(k) for code, k in zip(*np.unique(frozen_roles, return_counts=True))},
                      "elapsed_seconds": frozen["runtime"]["elapsed_seconds"], "code_unchanged": frozen["code_sha256"] == cs.code_hashes()}
    frozen_passed = frozen_summary["passed"] and frozen_summary["code_unchanged"] and all(c["passed"] for c in frozen_summary["checks"].values())
    sample_passed = (all(semantic["roles"][name]["rows"] == PER_ROLE for name in ROLE_NAMES.values())
                     and semantic["mismatched_rows"] == raw["mismatched_rows"] == xy_report["mismatched_rows"] == 0)
    sample_matches = {name: int(len(np.unique(match_index[roles == code]))) for code, name in ROLE_NAMES.items()}
    code = dict.fromkeys([Path(__file__), ROOT / "studies/tog_experiments_20261012/common.py", *(ROOT / name for name in cs.CODE_FILES),
                          *(ROOT / name for name in Q_FILES)])
    record = run_record(rows={name: report["rows"] for name, report in semantic["roles"].items()},
                        matches={"sample": sample_matches, "after_last": {name: len(after[name]["matches"]) for name in after}},
                        inputs=_input_files(gate, store_dir, projection.root), code=list(code), started_utc=started)
    result = {
        "version": VERSION,
        "passed": bool(frozen_passed and sample_passed and all(after[name]["passed"] for name in after)),
        "summary": {"frozen_gate": {name: check["mismatches"] for name, check in frozen_summary["checks"].items()},
                    "semantic_row": {name: report["mismatched_rows"] for name, report in semantic["roles"].items()},
                    "raw_store": {name: report["mismatched_rows"] for name, report in raw["roles"].items()},
                    "xy": {name: report["mismatched_rows"] for name, report in xy_report["roles"].items()},
                    "after_last": {name: {check: value["mismatches"] for check, value in after[name]["checks"].items()} for name in after}},
        "frozen_gate": frozen_summary,
        "sample": {"rule": f"per q role the {PER_ROLE} Q_ROWS_h90 pre queries with the smallest SHA-256('{SALT}:<query>')",
                   "salt": SALT, "per_role": PER_ROLE, "queries": {ROLE_NAMES[code]: values.tolist() for code, values in sample.items()}},
        "semantic_row": {"reference": "QuerySurface.iter_query_rows", "comparison": "controls_states.Comparison", **semantic},
        "raw_store": {"rebuilt": str(work / "raw_store"), "stored": str(store_dir), **raw},
        "xy": {"rebuilt": str(work / "XY_REBUILT.npz"), "stored": str(xy_path), **xy_report},
        "after_last": {"rule": f"per q role the first {AFTER_LAST_PER_ROLE} q-row matches in SHA-256('{SALT}:match:<match>') order "
                               "whose stretch (last stored query, limit] holds a non-item history token and an item event "
                               "(controls_states._gap_counts); controls_states.gate_match with the new times: midpoint of the widest "
                               "gap between stored queries, midpoint of (last stored query, limit], limit; "
                               "limit = min(last frame, game end - 1)", "roles": after, "reports": gate_reports},
        "sample_bank_shards_loaded": len(shards),
        "projection_files_sha256": projection_sha256,
        "elapsed_seconds": time.perf_counter() - tick,
        "run": record,
    }
    write_once(target, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(TOG_E1))
    args = parser.parse_args(argv)
    priority = cs.below_normal_priority()
    result = run(Path(args.out))
    print(json.dumps({"passed": result["passed"], "summary": result["summary"], "priority": priority,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
