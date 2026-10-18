"""Non-event control windows, phase 2: V input rows at the control ends (Q_DEV h90).

Phase 1 (``controls.py``) matched each Q_DEV h90 engagement to a control window.
Its start is a stored nominal query; its end (start + engagement length) is not a
stored query time (CONTROLS_MATCHING_h90.json: 11,321 distinct (match, end_ms),
none in the bank).  This module builds the V input row (make_view -> semantic_row)
at such new times the way every stored query row was built, so that the final V
can score them later.  Scoring is not done here.

A row at (match, t) is built stage by stage with the production code; only the
producer's array assembly (``query_arrays``) is restated, because it is inline:

1. Observable state.  The match is read from the static-preserved bank shard with
   every stored field (``xy`` is needed by ``from_bank``; the projection's
   ``load_source_shard`` drops it), reopened with
   ``ObservableMatch.from_bank(unpack_bank(...), patch)`` as ``producer.produce_match``
   reopens it, and ``query_numeric(t)`` is called per time.  ``query_arrays``
   assembles the per-query arrays as ``producer._queries`` does
   (producer.py:96-106 and 116-118; same dtypes: query_ms int32, objective counts
   (Q, 9, 3), of which the projection keeps column 0).
2. Projection blocks.  ``project_match`` on the projection's source fields
   (``source_fields``) with those arrays substituted.  The projection run copies its
   output unchanged (v_input_projection run.py:193-209).
3. Current items.  ``observe_match`` on an ``AcceptedMatch`` that carries the new
   times (empty ``v_nominal`` passes its validation).  This is the item bank's
   production replay (item_node_observed audit.py:848).  ``current_items`` turns
   the state into the dict that ``ObservedBankLoader.query`` and
   ``RawProjectionLoader.query_input`` hand on.
4. Histories.  The logic of ``iter_match_views`` (v_shared_history bank.py:55-85):
   ``HistoryAccumulator`` over ``SemanticHistory.iter_prefix``, and the
   recipient-confirmed item ledger rebuilt from ``history_prefix`` events
   (``iter_compact_observed_states`` + ``_snapshot``).  Both prefixes run to the
   latest requested time.  ``iter_match_views`` stops both at the match's last
   stored query; for a control end after it (188 ends in 170 matches) that would
   silently drop every event in between.
5. Row.  The two histories are encoded as the shared-history shard writer encodes
   them (store ``_plain`` + ``json.dumps``) and decoded again; the row is
   ``semantic_row(make_view(features, nonitem, item_history=item))``, the call
   ``QuerySurface.iter_query_rows`` makes (Blue view, no replacement swap).

Sidecar (resumable per source shard) under ``--out``: PLAN.json, END_TABLE.npz (one
row per distinct end in (match_index, end_ms) order, and ``engagement_end_index``
aligned with CONTROLS_MATCHING_h90.npz), parts/part_<shard>.npz with a receipt
(sha256, per-match seconds and RSS), TIMING.jsonl, and after the last part
CONTROL_END_ROWS_h90.npz with CONTROL_END_ROWS_h90.json.  A part stores the
features and the two history texts; ``iter_rows`` rebuilds the semantic rows with
the same make_view -> semantic_row call, in end order.  A later scorer feeds them
in canonical 1,024-row batches to ``queries.transform`` and ``score.LockedV.score``,
as ``score.score_parts`` does with ``QuerySurface.iter_query_rows``.

Equality gate (``--equality-gate``, ``equality_gate``): for sampled matches the rows
at every stored query, built by this path, must equal QuerySurface's rows bitwise;
adding the control ends must not change them; the sidecar must round-trip; and an
end after the last stored query must include the events in between.

Private library APIs used (no public equivalent):
  item_node_observed bank._event_from_value  (as iter_match_views and _SourceReferenceResolver use it)
  item_node_observed ledger RecipientConfirmedItemLedger._snapshot  (as iter_match_views uses it)
  v_shared_history_production store._plain  (the shard writer's JSON conversion)
  v_full_preprocess surface BankSurface._records_for_match  (equality gate only: the stored records)

Usage (one process, BELOW_NORMAL priority is set at start):
  python -B studies/label_production_20261005/controls_states.py --labels D:/LOL_Training_20261004/labels_1
  python -B studies/label_production_20261005/controls_states.py --labels ... --out DIR --limit-matches 20
  python -B studies/label_production_20261005/controls_states.py --labels ... --out DIR --equality-gate
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Iterator, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.item_node_encoding_20261002.source import CORE_MATCH_FIELDS, ITEM_TYPE_CODES, NODE_AUDIT_FIELDS, AcceptedMatch  # noqa: E402
from studies.item_node_observed_20261002.bank import _event_from_value, observe_match  # noqa: E402
from studies.item_node_observed_20261002.ledger import CompactObservedState, RecipientConfirmedItemLedger  # noqa: E402
from studies.observable_state_20261002.bank import match_view, unpack_bank  # noqa: E402
from studies.observable_state_20261002.kernel import OBJECTIVE_TYPES, ObservableMatch  # noqa: E402
from studies.observable_state_20261002.storage import open_shard  # noqa: E402
from studies.v_input_projection_20261002.bank import QUERY_SPECS, STATIC_SPECS, source_fields  # noqa: E402
from studies.v_input_projection_20261002.projection import SemanticHistory, project_match  # noqa: E402
from studies.v_model_fit_input_20261003.common import semantic_row  # noqa: E402
from studies.v_model_input_20261003.history import HistoryAccumulator, event_delta  # noqa: E402
from studies.v_model_input_20261003.view import make_view  # noqa: E402
from studies.v_shared_history_20261003.bank import COMPACT_FIELDS  # noqa: E402
from studies.v_shared_history_production_20261003.store import _plain as store_plain  # noqa: E402

VERSION = "label_control_end_rows_v1_20261005"
LABELS = Path(os.environ["LOL_TRAINING_ROOT"]) / 'labels_1'
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
STATIC_PRESERVED_BANK = ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1"
MATCHING = "CONTROLS_MATCHING_h90"
OUT_NAME = "CONTROL_END_ROWS_h90_1"   # default: a new directory under --labels
ROWS_NAME = "CONTROL_END_ROWS_h90"
GATE_NAME = "EQUALITY_GATE_h90"
Q_DEV = 3                            # MATCH_TABLE.role code
GATE_SEED = 20261005
REFERENCE_SEED = 20261001            # queries.open_inputs; the swap seed is unused for Blue-view rows
BANK_ADDED_PREFIXES = ("query_", "component_", "v_nominal")   # added by produce_match after export_bank
PROJECTION_QUERY_FIELDS = ("query_ms", "query_frame_index", "query_event_limit", "query_item_limit",
                           "query_team_kills", "query_kda", "query_objective_counts", "query_objective_last_ms")
ITEM_QUERY_FIELDS = ("query_ms", "query_event_limit", "query_item_ledger_known")
BOUND_FILES = (("archive_sha256", "BANK.npz"), ("layout_sha256", "LAYOUT.json"), ("shard_manifest_sha256", "SHARD_MANIFEST.json"),
               ("match_ledger_sha256", "MATCH_LEDGER.jsonl"), ("static_repair_sha256", "STATIC_REPAIR.json"))
PROJECTION_FILES = ("MATCH_TABLE.npy", "QUERY_TABLE.npy", "MATCH_QUERIES.npy", "V_ROWS.npy", "FRAME_AGE_SECONDS.npy")
RAGGED = {"item_ids": "item_row_offsets", "item_counts": "item_row_offsets",
          "nonitem_utf8": "nonitem_offsets", "item_utf8": "item_offsets"}
PART_PLAN_FIELDS = ("end_index", "match_index", "end_ms", "after_last_stored_query")
AUDIT_FIELDS = ("query_frame_index", "query_event_limit", "query_item_limit", "query_item_ledger_known")
FORBIDDEN_OUT = (ROOT / "outputs", ROOT / "studies", ROOT / "reports/v_production_20261004", Path(os.environ["LOL_TRAINING_ROOT"]) / 'run_3')
CODE_FILES = (
    "studies/label_production_20261005/controls_states.py", "studies/label_production_20261005/controls.py",
    "studies/label_production_20261005/queries.py", "studies/observable_state_20261002/producer.py",
    "studies/observable_state_20261002/kernel.py", "studies/observable_state_20261002/bank.py",
    "studies/observable_state_20261002/storage.py", "studies/v_input_projection_20261002/projection.py",
    "studies/v_input_projection_20261002/bank.py", "studies/item_node_encoding_20261002/source.py",
    "studies/item_node_encoding_20261002/ledger.py", "studies/item_node_observed_20261002/bank.py",
    "studies/item_node_observed_20261002/ledger.py", "studies/v_model_input_20261003/history.py",
    "studies/v_model_input_20261003/view.py", "studies/v_model_fit_input_20261003/common.py",
    "studies/v_shared_history_20261003/bank.py", "studies/v_shared_history_production_20261003/store.py",
    "studies/v_full_preprocess_20261003/surface.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def _rss() -> tuple[int, int]:
    """(current, peak) resident bytes of this process; -1 when psutil is unavailable."""

    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(info.rss), int(getattr(info, "peak_wset", info.rss))
    except Exception:  # pragma: no cover - diagnostics only
        return -1, -1


def below_normal_priority() -> int:
    import psutil

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    return int(process.nice())


# ---------------------------------------------------------------- one match, new times


@dataclass(frozen=True)
class MatchSource:
    """One match of a static-preserved bank shard with every stored field."""

    match_index: int
    shard_index: int
    local_index: int
    view: Mapping[str, np.ndarray]
    layout: Mapping[str, Any]
    entry: Mapping[str, Any]


@dataclass(frozen=True)
class EndState:
    """The stored form of one row: the query_input features and the two shared-history record texts."""

    time_ms: int
    features: dict[str, Any]
    nonitem: str
    item: str


@dataclass(frozen=True)
class MatchBuild:
    states: list[EndState]
    query: dict[str, np.ndarray]       # producer._queries arrays at the requested times


def _times(times: Sequence[int]) -> list[int]:
    values = [int(value) for value in times]
    if any(value < 0 or value >= 1 << 31 for value in values) or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("times must be strictly increasing nonnegative int32 milliseconds")
    return values


def query_arrays(match: ObservableMatch, times: Sequence[int]) -> dict[str, np.ndarray]:
    """``producer._queries``' per-query arrays (producer.py:96-106, 116-118) at arbitrary sorted times."""

    times = _times(times)
    snapshots = [match.query_numeric(t) for t in times]
    result = {
        "query_ms": np.asarray(times, dtype=np.int32),
        "query_frame_index": np.asarray([s["frame_index"] for s in snapshots], dtype=np.int32),
        "query_event_limit": np.asarray([s["event_cutoff_index"] for s in snapshots], dtype=np.int32),
        "query_item_limit": np.asarray([s["item_event_cutoff_index"] for s in snapshots], dtype=np.int32),
        "query_team_kills": np.asarray([s["team_kills"] for s in snapshots], dtype=np.int32).reshape(-1, 2),
        "query_kda": np.asarray([s["kda"] for s in snapshots], dtype=np.int32).reshape(-1, 3, 10),
        "query_objective_counts": np.asarray([s["objective_counts"] for s in snapshots], dtype=np.int32),
        "query_objective_last_ms": np.asarray([s["objective_last_timestamp_ms"] for s in snapshots], dtype=np.float64),
        "query_item_ledger_known": np.asarray([s["item_ledger_known"] for s in snapshots], dtype=bool),
    }
    if not snapshots:
        result["query_objective_counts"] = np.empty((0, len(OBJECTIVE_TYPES), 3), dtype=np.int32)
        result["query_objective_last_ms"] = np.empty((0, len(OBJECTIVE_TYPES)), dtype=np.float64)
    return result


def current_items(state: Any) -> dict[str, Any]:
    """One ObservedQueryState as the item bank writer stores it and ``query_input`` reads it back."""

    global_offsets = np.asarray(state.offsets, dtype=np.int64)           # NODE_OFFSETS row (writer adds ids_start)
    start, stop = int(global_offsets[0]), int(global_offsets[-1])
    items = {
        "item_ids": np.asarray(state.item_ids, dtype="<i8")[start:stop].copy(),
        "counts": np.asarray(state.counts, dtype="<i8")[start:stop].copy(),
        "offsets": (global_offsets - start).copy(),
        "recorded_ledger_supported": np.asarray(state.supported, dtype=bool).copy(),
        "reason_mask": np.asarray(state.reason_mask, dtype=np.uint64).copy(),
        "unassigned_observed": bool(state.unassigned_observed),
    }
    items["whole_inventory_verified"] = False
    return items


def record_text(history: Any) -> str:
    """One history as the shared-history shard writer serialises it (store._record_payload, _write_match_member)."""

    return json.dumps(store_plain(history), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _item_transactions(accepted: AcceptedMatch, observed: Any) -> list[dict[str, Any]]:
    """``history_prefix`` at the latest requested time, then iter_match_views' transaction mapping.

    The persisted history order is observe_match's assigned/unassigned partition
    order; ``_SourceReferenceResolver.events_for`` maps each event index to
    ``_event_from_value`` of the accepted item event; the prefix length is the
    state's history end at the last requested time (not at the last stored query).
    """

    resolved = {int(event.event_index): _event_from_value(event, int(event.event_index)) for event in accepted.item_events()}
    last = observed.states[-1]
    prefix = {
        "assigned": [resolved[int(event.event_index)].to_dict() for event in observed.assigned_events[:last.assigned_history_end]],
        "unassigned": [resolved[int(event.event_index)].to_dict() for event in observed.unassigned_events[:last.unassigned_history_end]],
    }
    return [_event_from_value(value, index).transaction_mapping()
            for partition in ("assigned", "unassigned") for index, value in enumerate(prefix[partition])]


def projection_bank(source: MatchSource, query: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """The projection's per-match source view (load_source_shard + local_source), query arrays optionally replaced."""

    bank = {name: source.view[name] for name in source_fields(source.layout)}
    if query is not None:
        bank.update({name: query[name] for name in PROJECTION_QUERY_FIELDS})
        bank["v_nominal"] = np.empty((0, 7), dtype=np.int32)
    return bank


def build_match(source: MatchSource, times: Sequence[int]) -> MatchBuild:
    """Stored-form rows of one match at strictly increasing times (stages 1-4 of the module docstring)."""

    times = _times(times)
    if not times:
        return MatchBuild([], {})
    view = source.view
    model_bank = {name: value for name, value in view.items() if not name.startswith(BANK_ADDED_PREFIXES)}
    match = ObservableMatch.from_bank(unpack_bank(model_bank), str(source.entry["patch"]))
    query = query_arrays(match, times)
    raw = projection_bank(source, query)
    blocks = project_match(raw)
    item_arrays = {name: view[name] for name in (*CORE_MATCH_FIELDS, *NODE_AUDIT_FIELDS) if name in view}
    item_arrays.update({name: query[name] for name in ITEM_QUERY_FIELDS})
    item_arrays["v_nominal"] = np.empty((0, 7), dtype=np.int32)
    accepted = AcceptedMatch(item_arrays, source.layout, dict(source.entry), source.shard_index, source.local_index)
    observed = observe_match(accepted)
    accumulator = HistoryAccumulator()
    for record in SemanticHistory(raw).iter_prefix(times[-1]):     # extended to the latest requested time
        accumulator.add(record)
    ledger = RecipientConfirmedItemLedger(_item_transactions(accepted, observed))
    states: list[EndState] = []
    compact = ledger.iter_compact_observed_states(list(times))
    for index, (cutoff, item_state, compact_state) in enumerate(zip(times, observed.states, compact, strict=True)):
        if int(item_state.query_timestamp_ms) != cutoff or int(compact_state["query_timestamp_ms"]) != cutoff:
            raise ValueError("item replay times differ from the requested times")
        features = {name: blocks[name][index] for name in QUERY_SPECS}
        features.update({name: blocks[name] for name in STATIC_SPECS})
        features["items"] = current_items(item_state)
        snapshot = ledger._snapshot(CompactObservedState(**{key: compact_state[key] for key in COMPACT_FIELDS}))
        for key in ("item_ids", "counts", "offsets", "recorded_ledger_supported"):  # the check iter_match_views makes
            if not np.array_equal(np.asarray(features["items"][key]), np.asarray(snapshot[key])):
                raise ValueError(f"current item reference differs: {key}")
        item = {"node": snapshot["assigned_history_by_participant"], "unassigned": snapshot["unassigned_history_context"]}
        built = make_view(features, accumulator.snapshot_at(cutoff), item_history=item)
        states.append(EndState(cutoff, features, record_text(built["nonitem_history"]), record_text(built["item_history"])))
    return MatchBuild(states, query)


def match_states(source: MatchSource, times: Sequence[int]) -> list[EndState]:
    return build_match(source, times).states


def semantic_rows(states: Sequence[EndState]) -> list[dict[str, Any]]:
    """The rows ``QuerySurface.iter_query_rows`` yields: make_view of the stored record -> semantic_row."""

    return [semantic_row(make_view(state.features, json.loads(state.nonitem), item_history=json.loads(state.item)))
            for state in states]


def rows_at_times(source: MatchSource, times: Sequence[int]) -> list[dict[str, Any]]:
    """Semantic rows of one match at strictly increasing new (or stored) times."""

    return semantic_rows(match_states(source, times))


# ---------------------------------------------------------------- sources


class BankShard:
    """A static-preserved bank shard read once and checked against the projection's source binding."""

    def __init__(self, shard_index: int, binding: Mapping[str, Any], source_root: Path) -> None:
        self.shard_index = int(shard_index)
        self.directory = Path(source_root) / str(binding["directory"])
        self.hashes: dict[str, str] = {}
        for key, filename in BOUND_FILES:            # the RawProjectionLoader._history binding check
            digest = sha256_file(self.directory / filename)
            if digest != binding.get(key):
                raise ValueError(f"bank shard differs from the projection binding: {self.directory.name}/{filename}")
            self.hashes[filename] = digest
        self.layout = read_json(self.directory / "LAYOUT.json")
        with (self.directory / "MATCH_LEDGER.jsonl").open(encoding="utf-8") as stream:
            self.ledger = [json.loads(line) for line in stream if line.strip()]
        self.arrays = open_shard(self.directory, verify_hashes=False)   # BANK.npz hash checked above
        if len(self.arrays["offset_event_timestamp"]) - 1 != len(self.ledger):
            raise ValueError(f"bank shard and ledger cardinality differ: {self.directory.name}")

    def source(self, local_index: int, match_index: int, canonical_key: str | None = None) -> MatchSource:
        entry = self.ledger[int(local_index)]
        if canonical_key is not None and str(entry["canonical_match_key"]) != canonical_key:
            raise ValueError(f"match {match_index}: bank ledger key differs from the projection")
        return MatchSource(int(match_index), self.shard_index, int(local_index),
                           match_view(self.arrays, self.layout, int(local_index)), self.layout, entry)


class Projection:
    """The projection tables used to plan and to bind sources (read-only, memory-mapped)."""

    def __init__(self, root: Path = PROJECTION) -> None:
        self.root = Path(root)
        self.manifest = read_json(self.root / "RUN_MANIFEST.json")
        self.metadata = read_json(self.root / "BANK_METADATA.json")
        if self.manifest.get("complete") is not True:
            raise ValueError("projection run is incomplete")
        self.source_root = Path(self.metadata["source_root"])
        if self.source_root.resolve() != STATIC_PRESERVED_BANK.resolve():
            raise ValueError("projection source is not the static-preserved bank")
        self.matches = np.load(self.root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.queries = np.load(self.root / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.match_queries = np.load(self.root / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)

    def binding(self, shard: int) -> Mapping[str, Any]:
        return self.metadata["source_shards"][int(shard)]

    def key(self, match_index: int) -> str:
        return bytes(self.matches["key"][int(match_index)]).decode("ascii")

    def file_hashes(self) -> dict[str, str]:
        saved = self.manifest.get("saved_files", {})
        result = {}
        for name in PROJECTION_FILES:
            digest = sha256_file(self.root / name)
            if digest != saved.get(name, {}).get("sha256"):
                raise ValueError(f"projection file differs from its run manifest: {name}")
            result[name] = digest
        return result


def max_train_frame_age(root: Path = PROJECTION) -> float:
    """The largest frame age of the supported TRAIN nominal V rows (clock_only.py:79-84)."""

    ages = np.load(root / "FRAME_AGE_SECONDS.npy", mmap_mode="r", allow_pickle=False)
    nominal = np.load(root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    matches = np.load(root / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
    first = np.load(root / "MATCH_QUERIES.npy", mmap_mode="r", allow_pickle=False)["first_query"]
    train = nominal[(np.asarray(matches["role"])[np.asarray(nominal["match_index"])] == 0) & np.asarray(nominal["supported"])]
    return float(np.max(ages[np.asarray(first)[train["match_index"]] + train["query_ref"]]))


# ---------------------------------------------------------------- plan, sidecar parts


def plan_ends(controls: Mapping[str, np.ndarray], matches: np.ndarray, queries: np.ndarray, match_queries: np.ndarray) -> dict[str, np.ndarray]:
    """Distinct (match_index, end_ms) of the matched controls, sorted, with their shard and stored-query context."""

    start_q = np.asarray(controls["control_start_query"], dtype=np.int64)
    rows = np.flatnonzero(start_q >= 0)
    end_ms = np.asarray(controls["control_end_ms"], dtype=np.int64)[rows]
    if np.any(end_ms < 0):
        raise ValueError("matched control without an end")
    match_index = np.asarray(queries["match_index"][start_q[rows]], dtype=np.int64)
    if not np.array_equal(np.asarray(matches["key"])[match_index], np.asarray(controls["canonical_match_key"])[rows]):
        raise ValueError("control start query belongs to another match")
    if not np.array_equal(np.asarray(queries["query_ms"])[start_q[rows]], np.asarray(controls["control_start_ms"])[rows]):
        raise ValueError("control start time differs from its query")
    if len(rows):
        pairs, inverse = np.unique(np.stack((match_index, end_ms), axis=1), axis=0, return_inverse=True)
    else:
        pairs, inverse = np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    m, e = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    first = np.asarray(match_queries["first_query"])[m].astype(np.int64)
    count = np.asarray(match_queries["n_queries"])[m].astype(np.int64)
    if np.any(count <= 0):
        raise ValueError("control match without stored queries")
    query_ms = np.asarray(queries["query_ms"])
    stored = np.asarray([bool(np.isin(end, query_ms[a:a + n])) for end, a, n in zip(e, first, count)], dtype=bool)
    shard = np.asarray(matches["shard"])[m].astype(np.int64)
    if np.any(np.diff(shard) < 0):
        raise ValueError("projection match order does not follow the shard order")
    engagement_end = np.full(len(start_q), -1, dtype=np.int64)
    engagement_end[rows] = inverse
    return {"end_index": np.arange(len(pairs), dtype=np.int64), "match_index": m, "end_ms": e, "shard": shard,
            "local_match": np.asarray(matches["local_match"])[m].astype(np.int64),
            "canonical_match_key": np.asarray(matches["key"])[m], "role": np.asarray(matches["role"])[m].astype(np.int8),
            "last_stored_query_ms": query_ms[first + count - 1].astype(np.int64),
            "after_last_stored_query": e > query_ms[first + count - 1], "end_is_stored_query": stored,
            "n_engagements": np.bincount(inverse, minlength=len(pairs)).astype(np.int64),
            "engagement_end_index": engagement_end}


def load_plan(labels: Path, projection: Projection) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = Path(labels)
    path = labels / f"{MATCHING}.npz"
    with np.load(path, allow_pickle=False) as stored:
        controls = {name: stored[name] for name in stored.files}
    plan = plan_ends(controls, projection.matches, projection.queries, projection.match_queries)
    if np.any(plan["role"] != Q_DEV):
        raise ValueError("control ends outside Q_DEV")
    inputs = {"matching_npz": str(path), "matching_npz_sha256": sha256_file(path),
              "matching_json_sha256": sha256_file(labels / f"{MATCHING}.json"), "projection_root": str(projection.root),
              "projection_run_manifest_sha256": sha256_file(projection.root / "RUN_MANIFEST.json"),
              "projection_bank_metadata_sha256": sha256_file(projection.root / "BANK_METADATA.json"),
              "projection_files_sha256": projection.file_hashes(), "bank_root": str(projection.source_root)}
    return plan, inputs


def _plan_digest(plan: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(plan):
        array = np.ascontiguousarray(plan[name])
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def checked_out(out: Path) -> Path:
    out = Path(out).resolve()
    for forbidden in FORBIDDEN_OUT:
        forbidden = forbidden.resolve()
        if out == forbidden or forbidden in out.parents:
            raise ValueError(f"refusing to write under {forbidden}")
    return out


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return sha256_file(path)


def write_json_atomic(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def part_path(out: Path, shard: int) -> Path:
    return Path(out) / "parts" / f"part_{int(shard):04d}.npz"


def part_complete(path: Path, expected_end_index: np.ndarray, plan_sha256: str) -> bool:
    """A part counts only with its receipt, the same plan, its file hash and exactly the expected ends."""

    receipt = path.with_suffix(".json")
    if not path.is_file() or not receipt.is_file():
        return False
    try:
        record = read_json(receipt)
        if record.get("version") != VERSION or record.get("plan_sha256") != plan_sha256 or record.get("sha256") != sha256_file(path):
            return False
        with np.load(path, allow_pickle=False) as stored:
            return np.array_equal(stored["end_index"], np.asarray(expected_end_index, dtype=np.int64))
    except (OSError, ValueError, KeyError):
        return False


def encode_states(states: Sequence[EndState]) -> dict[str, np.ndarray]:
    """Fixed-shape features in their projection dtypes, ragged items and the two UTF-8 history texts."""

    n = len(states)
    out: dict[str, np.ndarray] = {"time_ms": np.asarray([state.time_ms for state in states], dtype=np.int64)}
    for name, (dtype, tail) in {**QUERY_SPECS, **STATIC_SPECS}.items():
        out[f"feature_{name}"] = np.asarray([state.features[name] for state in states], dtype=dtype).reshape((n, *tail))
    items = [state.features["items"] for state in states]
    sizes = [len(item["item_ids"]) for item in items]
    out["item_row_offsets"] = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64))).astype(np.int64)
    for name, key in (("item_ids", "item_ids"), ("item_counts", "counts")):
        out[name] = np.concatenate([np.asarray(item[key], dtype=np.int64) for item in items]) if n else np.empty(0, dtype=np.int64)
    out["item_node_offsets"] = np.asarray([item["offsets"] for item in items], dtype=np.int64).reshape(n, 11)
    out["item_recorded_ledger_supported"] = np.asarray([item["recorded_ledger_supported"] for item in items], dtype=bool).reshape(n, 10)
    out["item_reason_mask"] = np.asarray([item["reason_mask"] for item in items], dtype=np.uint64).reshape(n, 10)
    out["item_unassigned_observed"] = np.asarray([item["unassigned_observed"] for item in items], dtype=bool)
    for name in ("nonitem", "item"):
        encoded = [getattr(state, name).encode("utf-8") for state in states]
        out[f"{name}_utf8"] = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        out[f"{name}_offsets"] = np.concatenate(([0], np.cumsum([len(value) for value in encoded], dtype=np.int64))).astype(np.int64)
    return out


def decode_states(arrays: Mapping[str, np.ndarray]) -> list[EndState]:
    """Inverse of ``encode_states``; the features dict is built as ``RawProjectionLoader.query_input`` builds it."""

    states = []
    for index in range(len(arrays["time_ms"])):
        features = {name: arrays[f"feature_{name}"][index] for name in QUERY_SPECS}
        features.update({name: arrays[f"feature_{name}"][index] for name in STATIC_SPECS})
        start, stop = int(arrays["item_row_offsets"][index]), int(arrays["item_row_offsets"][index + 1])
        items = {"item_ids": arrays["item_ids"][start:stop].copy(), "counts": arrays["item_counts"][start:stop].copy(),
                 "offsets": arrays["item_node_offsets"][index].copy(),
                 "recorded_ledger_supported": arrays["item_recorded_ledger_supported"][index].copy(),
                 "reason_mask": arrays["item_reason_mask"][index].copy(),
                 "unassigned_observed": bool(arrays["item_unassigned_observed"][index])}
        items["whole_inventory_verified"] = False
        features["items"] = items
        texts = []
        for name in ("nonitem", "item"):
            begin, end = int(arrays[f"{name}_offsets"][index]), int(arrays[f"{name}_offsets"][index + 1])
            texts.append(bytes(arrays[f"{name}_utf8"][begin:end]).decode("utf-8"))
        states.append(EndState(int(arrays["time_ms"][index]), features, texts[0], texts[1]))
    return states


def concat_parts(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate encoded parts in order; ragged offsets are rebased."""

    out: dict[str, np.ndarray] = {}
    offsets = set(RAGGED.values())
    for name in parts[0]:
        if name in offsets:
            pieces, base = [np.zeros(1, dtype=np.int64)], 0
            for part in parts:
                values = np.asarray(part[name], dtype=np.int64)
                pieces.append(values[1:] + base)
                base += int(values[-1])
            out[name] = np.concatenate(pieces)
        else:
            out[name] = np.concatenate([np.asarray(part[name]) for part in parts], axis=0)
    return out


def iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    """(end_index, semantic row, meta) of a part or of CONTROL_END_ROWS_h90.npz, in stored (end) order."""

    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    for index, state in enumerate(decode_states(arrays)):
        meta = {"end_index": int(arrays["end_index"][index]), "match_index": int(arrays["match_index"][index]),
                "end_ms": int(arrays["end_ms"][index]),
                "after_last_stored_query": bool(arrays["after_last_stored_query"][index]),
                "outside_training_frame_age_support": bool(arrays["outside_training_frame_age_support"][index])}
        if meta["end_ms"] != state.time_ms:
            raise ValueError("stored end time differs from its state")
        yield meta["end_index"], semantic_rows([state])[0], meta


# ---------------------------------------------------------------- production driver


def _write_or_check_plan(out: Path, plan: Mapping[str, np.ndarray], inputs: Mapping[str, Any]) -> str:
    digest = _plan_digest(plan)
    plan_json = out / "PLAN.json"
    if plan_json.is_file():
        stored = read_json(plan_json)
        if stored.get("version") != VERSION or stored.get("plan_sha256") != digest or stored.get("inputs") != dict(inputs):
            raise ValueError(f"{out} holds another plan; use a new --out")
        return digest
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no PLAN.json; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    table_sha = write_npz_atomic(out / "END_TABLE.npz", plan)
    write_json_atomic(plan_json, {"version": VERSION, "plan_sha256": digest, "end_table_sha256": table_sha, "inputs": dict(inputs),
                                  "n_engagements_matched": int((plan["engagement_end_index"] >= 0).sum()),
                                  "n_ends": int(len(plan["end_index"])), "n_matches": int(len(np.unique(plan["match_index"]))),
                                  "n_shards": int(len(np.unique(plan["shard"]))),
                                  "n_after_last_stored_query": int(plan["after_last_stored_query"].sum()),
                                  "n_end_is_stored_query": int(plan["end_is_stored_query"].sum()),
                                  "utc": datetime.now(timezone.utc).isoformat()})
    return digest


def run(labels: Path, out: Path, *, limit_matches: int | None = None, projection_root: Path = PROJECTION) -> dict[str, Any]:
    """Build every planned end row once, one source shard per part; finished parts are skipped."""

    started = time.perf_counter()
    labels, out = Path(labels), checked_out(out)
    if limit_matches is not None and limit_matches < 1:
        raise ValueError("--limit-matches must be positive")
    projection = Projection(projection_root)
    plan, inputs = load_plan(labels, projection)
    plan_sha = _write_or_check_plan(out, plan, inputs)
    max_age = max_train_frame_age(projection.root)
    selected = np.ones(len(plan["end_index"]), dtype=bool)
    if limit_matches is not None:
        selected = np.isin(plan["match_index"], np.unique(plan["match_index"])[:limit_matches])
    counts = {"parts_written": 0, "parts_skipped": 0, "matches_built": 0, "rows_built": 0}
    shard_seconds: list[float] = []
    write_seconds: list[float] = []
    match_seconds: list[float] = []
    for shard in np.unique(plan["shard"][selected]).tolist():
        rows = np.flatnonzero(selected & (plan["shard"] == shard))
        target = part_path(out, shard)
        if part_complete(target, plan["end_index"][plan["shard"] == shard], plan_sha) or part_complete(target, plan["end_index"][rows], plan_sha):
            counts["parts_skipped"] += 1   # a complete part is never replaced by a --limit-matches part
            continue
        tick = time.perf_counter()
        bank = BankShard(shard, projection.binding(shard), projection.source_root)
        shard_seconds.append(time.perf_counter() - tick)
        states: list[EndState] = []
        audit: dict[str, list[np.ndarray]] = {name: [] for name in AUDIT_FIELDS}
        timings = []
        for match in np.unique(plan["match_index"][rows]).tolist():
            picked = rows[plan["match_index"][rows] == match]
            tick = time.perf_counter()
            source = bank.source(int(plan["local_match"][picked[0]]), match, projection.key(match))
            build = build_match(source, plan["end_ms"][picked].tolist())
            seconds = time.perf_counter() - tick
            rss, peak = _rss()
            states.extend(build.states)
            for name in audit:
                audit[name].append(build.query[name])
            timings.append({"match_index": int(match), "ends": int(len(picked)), "seconds": seconds, "rss_bytes": rss, "peak_rss_bytes": peak,
                            "n_events": int(len(source.view["event_timestamp"])), "n_frames": int(len(source.view["minute_ts"]))})
            match_seconds.append(seconds)
            with (out / "TIMING.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"shard": int(shard), **timings[-1]}) + "\n")
        tick = time.perf_counter()
        payload = part_payload(states, {name: plan[name][rows] for name in PART_PLAN_FIELDS},
                               {name: np.concatenate(values) for name, values in audit.items()}, max_age)
        write_part(out, shard, payload, plan_sha, {
            "complete_shard": bool(len(rows) == int((plan["shard"] == shard).sum())),
            "bank_directory": bank.directory.name, "bank_hashes": bank.hashes, "shard_load_seconds": shard_seconds[-1],
            "match_timings": timings, "max_train_frame_age_s": max_age})
        write_seconds.append(time.perf_counter() - tick)
        counts["parts_written"] += 1
        counts["matches_built"] += len(timings)
        counts["rows_built"] += int(len(rows))
        del bank, states
    return finish(out, plan, plan_sha, inputs, max_age, counts, limit_matches,
                  [a + b for a, b in zip(shard_seconds, write_seconds)], match_seconds, time.perf_counter() - started)


def part_payload(states: Sequence[EndState], plan_rows: Mapping[str, np.ndarray], audit: Mapping[str, np.ndarray],
                 max_age: float) -> dict[str, np.ndarray]:
    """One part: the encoded states, their plan columns, the query references and the frame-age support flag."""

    payload = encode_states(states)
    payload.update({name: np.asarray(plan_rows[name]) for name in PART_PLAN_FIELDS})
    payload.update({name: np.asarray(audit[name]) for name in AUDIT_FIELDS})
    if not np.array_equal(payload["end_ms"], payload["time_ms"]):
        raise ValueError("part states and plan rows differ")
    payload["outside_training_frame_age_support"] = payload["feature_frame_age_seconds"] > max_age   # clock_only.py:86
    return payload


def write_part(out: Path, shard: int, payload: Mapping[str, np.ndarray], plan_sha: str, extra: Mapping[str, Any]) -> str:
    """Part file first (fsync, atomic rename), then the receipt that makes it count."""

    target = part_path(out, shard)
    sha = write_npz_atomic(target, payload)
    write_json_atomic(target.with_suffix(".json"), {
        **dict(extra), "version": VERSION, "plan_sha256": plan_sha, "shard": int(shard), "rows": int(len(payload["end_index"])),
        "sha256": sha, "part_bytes": target.stat().st_size,
        "end_index_first": int(payload["end_index"][0]) if len(payload["end_index"]) else None,
        "end_index_last": int(payload["end_index"][-1]) if len(payload["end_index"]) else None,
        "outside_training_frame_age_support": int(np.sum(payload["outside_training_frame_age_support"])),
        "code_sha256": code_hashes(), "utc": datetime.now(timezone.utc).isoformat()})
    return sha


def finish(out: Path, plan: Mapping[str, np.ndarray], plan_sha: str, inputs: Mapping[str, Any], max_age: float,
           counts: Mapping[str, int], limit_matches: int | None, shard_seconds: list[float], match_seconds: list[float],
           elapsed: float) -> dict[str, Any]:
    """Progress manifest; once every shard part is complete, also the consolidated rows file."""

    shards = np.unique(plan["shard"]).tolist()
    done = [shard for shard in shards if part_complete(part_path(out, shard), plan["end_index"][plan["shard"] == shard], plan_sha)]
    complete = len(done) == len(shards)
    rss, peak = _rss()
    n_matches = int(len(np.unique(plan["match_index"])))
    manifest: dict[str, Any] = {
        "version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "complete": bool(complete), "plan_sha256": plan_sha,
        "inputs": dict(inputs), "parts_complete": len(done), "parts_total": len(shards), "this_run": dict(counts),
        "limit_matches": limit_matches, "n_ends": int(len(plan["end_index"])), "n_matches": n_matches,
        "n_engagements_matched": int((plan["engagement_end_index"] >= 0).sum()),
        "n_after_last_stored_query": int(plan["after_last_stored_query"].sum()),
        "n_matches_with_end_after_last_stored_query": int(len(np.unique(plan["match_index"][plan["after_last_stored_query"]]))),
        "max_train_frame_age_s": max_age, "code_sha256": code_hashes(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "elapsed_seconds": elapsed, "rss_bytes": rss,
                    "peak_rss_bytes": peak, "numeric_threads": {name: os.environ.get(name) for name in
                                                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "timing_this_run": {"shard_load_and_write_seconds": shard_seconds,
                            "match_seconds_mean": float(np.mean(match_seconds)) if match_seconds else None,
                            "match_seconds_median": float(np.median(match_seconds)) if match_seconds else None,
                            "match_seconds_max": float(np.max(match_seconds)) if match_seconds else None},
    }
    if match_seconds:
        per_shard = float(np.mean(shard_seconds)) if shard_seconds else 0.0
        manifest["timing_this_run"]["full_run_estimate_seconds"] = float(np.mean(match_seconds)) * n_matches + per_shard * len(shards)
    if complete:
        part_sha = {str(shard): read_json(part_path(out, shard).with_suffix(".json"))["sha256"] for shard in shards}
        rows_path, manifest_path = out / f"{ROWS_NAME}.npz", out / f"{ROWS_NAME}.json"
        previous = read_json(manifest_path) if manifest_path.is_file() else {}
        if (previous.get("complete") is True and previous.get("plan_sha256") == plan_sha and previous.get("part_sha256") == part_sha
                and rows_path.is_file() and previous.get("rows_sha256") == sha256_file(rows_path)):
            # Already consolidated from these parts; rewriting would only change the zip times inside the npz.
            previous.update(this_run=dict(counts), last_check_utc=manifest["utc"])
            write_json_atomic(manifest_path, previous)
            return previous
        parts = []
        for shard in shards:
            with np.load(part_path(out, shard), allow_pickle=False) as stored:
                parts.append({name: stored[name] for name in stored.files})
        table = concat_parts(parts)
        del parts
        if not np.array_equal(table["end_index"], plan["end_index"]) or not np.array_equal(table["end_ms"], plan["end_ms"]):
            raise ValueError("concatenated parts differ from the plan")
        outside = np.flatnonzero(table["outside_training_frame_age_support"])
        manifest.update({
            "rows_sha256": write_npz_atomic(rows_path, table), "rows": int(len(table["end_index"])), "part_sha256": part_sha,
            "outside_training_frame_age_support": int(len(outside)),
            "outside_training_frame_age_support_rows": [{"end_index": int(table["end_index"][i]), "match_index": int(table["match_index"][i]),
                                                         "end_ms": int(table["end_ms"][i]),
                                                         "frame_age_s": float(table["feature_frame_age_seconds"][i])} for i in outside],
            "scoring": "iter_rows(CONTROL_END_ROWS_h90.npz) -> 1,024-row batches -> queries.transform -> score.LockedV(final).score",
        })
    write_json_atomic(out / f"{ROWS_NAME}.json", manifest)
    return manifest


# ---------------------------------------------------------------- equality gate


class Comparison:
    """Exact comparison: same types, dict key order, dtypes, shapes and bytes (floats bitwise)."""

    def __init__(self) -> None:
        self.leaves = 0
        self.elements = 0
        self.max_abs_diff = 0.0
        self.mismatches: list[str] = []

    def _fail(self, path: str, message: str) -> None:
        self.mismatches.append(f"{path}: {message}")

    def _diff(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.shape == right.shape and left.dtype.kind in "fiu" and right.dtype.kind in "fiu":
            with np.errstate(invalid="ignore", over="ignore"):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            finite = np.isfinite(delta)
            if finite.any():
                self.max_abs_diff = max(self.max_abs_diff, float(delta[finite].max()))
            if (~finite).any():
                self.max_abs_diff = float("inf")

    def check(self, actual: Any, expected: Any, path: str = "", *, subclass_ok: bool = False) -> bool:
        before = len(self.mismatches)
        self._walk(actual, expected, path, subclass_ok)
        return len(self.mismatches) == before

    def _walk(self, a: Any, b: Any, path: str, subclass_ok: bool) -> None:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or (not subclass_ok and type(a) is not type(b)):
                return self._fail(path, f"type {type(a).__name__} != {type(b).__name__}")
            self.leaves += 1
            self.elements += int(b.size)
            if a.dtype != b.dtype or a.shape != b.shape:
                return self._fail(path, f"dtype/shape {a.dtype}{a.shape} != {b.dtype}{b.shape}")
            if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
                self._diff(a, b)
                return self._fail(path, "values differ")
            return None
        if type(a) is not type(b):
            return self._fail(path, f"type {type(a).__name__} != {type(b).__name__}")
        if isinstance(b, dict):
            if [type(key) for key in a] != [type(key) for key in b] or list(a) != list(b):
                return self._fail(path, "dict keys or key order differ")
            for key in b:
                self._walk(a[key], b[key], f"{path}/{key}", subclass_ok)
            return None
        if isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return self._fail(path, f"length {len(a)} != {len(b)}")
            for index, (left, right) in enumerate(zip(a, b)):
                self._walk(left, right, f"{path}[{index}]", subclass_ok)
            return None
        self.leaves += 1
        self.elements += 1
        if isinstance(b, float):
            if struct.pack("<d", a) != struct.pack("<d", b):
                self.max_abs_diff = max(self.max_abs_diff, abs(a - b)) if np.isfinite(a - b) else float("inf")
                return self._fail(path, f"{a!r} != {b!r}")
            return None
        if isinstance(b, np.generic):
            if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                self._diff(np.asarray(a), np.asarray(b))
                return self._fail(path, f"{a!r} != {b!r}")
            return None
        if a != b:
            return self._fail(path, f"{a!r} != {b!r}")
        return None

    def summary(self) -> dict[str, Any]:
        return {"leaves_compared": self.leaves, "elements_compared": self.elements, "max_abs_diff": self.max_abs_diff,
                "mismatches": len(self.mismatches), "first_mismatches": self.mismatches[:20]}


def select_gate_matches(plan: Mapping[str, np.ndarray], *, per_bin: int = 3, after_last: int = 3, seed: int = GATE_SEED) -> list[int]:
    """Matches with an end after their last stored query, then ``per_bin`` per end game-time bin (<15, 15-25, >=25 min)."""

    rng = np.random.default_rng(seed)
    after = np.unique(plan["match_index"][plan["after_last_stored_query"]])
    chosen = [int(value) for value in rng.choice(after, size=min(after_last, len(after)), replace=False)] if len(after) else []
    bins = np.digitize(plan["end_ms"] / 60000.0, (15, 25))
    for game_bin in range(3):
        candidates = np.setdiff1d(np.unique(plan["match_index"][(bins == game_bin) & ~plan["after_last_stored_query"]]), chosen)
        if len(candidates):
            chosen += [int(value) for value in rng.choice(candidates, size=min(per_bin, len(candidates)), replace=False)]
    return sorted(set(chosen))


def _stored_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _gap_counts(source: MatchSource, last_ms: int, end_ms: int) -> tuple[int, int, int]:
    """(source history records, their non-item token increments, item events) with last_ms < timestamp <= end_ms."""

    records = [record for record in SemanticHistory(projection_bank(source)).iter_prefix(end_ms) if int(record["timestamp_ms"]) > last_ms]
    tokens = sum(int(count) for record in records for _lane, _token, count in event_delta(record))
    stamps = np.asarray(source.view["event_timestamp"], dtype=np.int64)
    items = np.isin(np.asarray(source.view["event_type_code"]), tuple(ITEM_TYPE_CODES)) & (stamps > last_ms) & (stamps <= end_ms)
    return len(records), tokens, int(items.sum())


def gate_match(surface: Any, projection: Projection, source: MatchSource, control_times: Sequence[int], out: Path,
               categories: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """All checks for one match; every comparison is exact."""

    match = source.match_index
    tick = time.perf_counter()
    first = int(projection.match_queries["first_query"][match])
    count = int(projection.match_queries["n_queries"][match])
    stored_q = np.arange(first, first + count, dtype=np.int64)
    stored_t = [int(value) for value in projection.queries["query_ms"][stored_q]]
    control_t = _times(control_times)
    checks = {name: Comparison() for name in ("query_arrays", "features", "history_text", "rows", "prefix", "roundtrip")}
    # reference: the existing label path for every stored query of the match
    reference = list(surface.iter_query_rows(stored_q))
    if [int(q) for q, _row, _meta in reference] != stored_q.tolist():
        raise ValueError("reference rows came back out of order")
    records = {int(record["query_id"]): record for record in surface._records_for_match(match)[5]}
    built = build_match(source, stored_t)
    for name, value in built.query.items():                    # producer arrays == the stored bank arrays
        checks["query_arrays"].check(value, np.asarray(source.view[name]), name)
    rows = semantic_rows(built.states)
    for index, query in enumerate(stored_q.tolist()):
        supplied = surface.loader.query_input(query, include_histories=False)
        checks["features"].check(built.states[index].features, supplied["features"], f"q{query}", subclass_ok=True)
        checks["history_text"].check(built.states[index].nonitem, _stored_text(records[query]["nonitem"]), f"q{query}/nonitem")
        checks["history_text"].check(built.states[index].item, _stored_text(records[query]["item"]), f"q{query}/item")
        checks["rows"].check(rows[index], reference[index][1], f"q{query}")
        meta = reference[index][2]
        if meta["match_index"] != match or meta["query_ms"] != stored_t[index]:
            checks["rows"]._fail(f"q{query}", "reference meta differs")
    # prefix/causality: adding the control ends changes neither the stored-time rows nor the end rows
    union = sorted(set(stored_t) | set(control_t))
    position = {value: index for index, value in enumerate(union)}
    both = build_match(source, union)
    both_rows = semantic_rows(both.states)
    alone = build_match(source, control_t)
    alone_rows = semantic_rows(alone.states)
    for index, cutoff in enumerate(stored_t):
        state = both.states[position[cutoff]]
        checks["prefix"].check(both_rows[position[cutoff]], rows[index], f"stored@{cutoff}")
        checks["prefix"].check((state.nonitem, state.item), (built.states[index].nonitem, built.states[index].item), f"stored@{cutoff}/text")
    for index, cutoff in enumerate(control_t):
        state = both.states[position[cutoff]]
        checks["prefix"].check(both_rows[position[cutoff]], alone_rows[index], f"end@{cutoff}")
        checks["prefix"].check((state.nonitem, state.item), (alone.states[index].nonitem, alone.states[index].item), f"end@{cutoff}/text")
        checks["prefix"].check(state.features, alone.states[index].features, f"end@{cutoff}/features")
    # the sidecar encoding round-trips through a file
    path = out / "gate_roundtrip" / f"match_{match:06d}.npz"
    write_npz_atomic(path, encode_states(alone.states))
    with np.load(path, allow_pickle=False) as stored:
        decoded = decode_states({name: stored[name] for name in stored.files})
    checks["roundtrip"].check(semantic_rows(decoded), alone_rows, "rows")
    for index, state in enumerate(decoded):
        checks["roundtrip"].check((state.time_ms, state.nonitem, state.item), (alone.states[index].time_ms, alone.states[index].nonitem,
                                                                              alone.states[index].item), f"end{index}/text")
        checks["roundtrip"].check(state.features, alone.states[index].features, f"end{index}/features", subclass_ok=True)
    # ends after the last stored query: the events in between are in the end row
    last = stored_t[-1]
    after = []
    for cutoff in (value for value in control_t if value > last):
        records_gap, tokens_gap, items_gap = _gap_counts(source, last, cutoff)
        at_end, at_last = both.states[position[cutoff]], both.states[position[last]]
        nonitem_changed, item_changed = at_end.nonitem != at_last.nonitem, at_end.item != at_last.item
        consistent = nonitem_changed == (tokens_gap > 0) and item_changed == (items_gap > 0)
        if not consistent:
            checks["prefix"]._fail(f"after_last@{cutoff}", "history change does not match the events after the last stored query")
        after.append({"end_ms": cutoff, "last_stored_query_ms": last, "gap_history_records": records_gap, "gap_nonitem_tokens": tokens_gap,
                      "gap_item_events": items_gap, "nonitem_changed": bool(nonitem_changed), "item_changed": bool(item_changed),
                      "stale_under_last_query_cutoff": bool(nonitem_changed or item_changed), "consistent": bool(consistent)})
    in_stored = {name: int(np.isin(np.asarray(values), stored_q).sum()) for name, values in categories.items()}
    return {"match_index": int(match), "shard": source.shard_index, "stored_queries": count, "control_ends": len(control_t),
            "control_end_minutes": [round(value / 60000.0, 3) for value in control_t],
            "stored_query_categories": in_stored, "after_last_stored_query": after,
            "checks": {name: comparison.summary() for name, comparison in checks.items()},
            "passed": all(not comparison.mismatches for comparison in checks.values()),
            "seconds": time.perf_counter() - tick}


def equality_gate(labels: Path, out: Path, *, matches: Sequence[int] | None = None, per_bin: int = 3, after_last: int = 3,
                  seed: int = GATE_SEED, projection_root: Path = PROJECTION) -> dict[str, Any]:
    """Exact equality of this path with the existing label path at stored queries, for sampled control matches."""

    from studies.label_production_20261005.queries import PROJECTION as LABEL_PROJECTION, SHARED_BANK, QuerySurface

    started = time.perf_counter()
    labels, out = Path(labels), checked_out(out)
    out.mkdir(parents=True, exist_ok=True)
    projection = Projection(projection_root)
    if projection.root.resolve() != Path(LABEL_PROJECTION).resolve():
        raise ValueError("gate projection differs from the label projection")
    plan, inputs = load_plan(labels, projection)
    chosen = sorted(int(value) for value in matches) if matches is not None else select_gate_matches(
        plan, per_bin=per_bin, after_last=after_last, seed=seed)
    with np.load(labels / "ENGAGEMENTS.npz", allow_pickle=False) as stored:
        engagement = {name: stored[name] for name in ("match_index", "query_pre", "query_h90")}
    with np.load(labels / f"{MATCHING}.npz", allow_pickle=False) as stored:
        starts = stored["control_start_query"]
    nominal = np.load(projection.root / "V_ROWS.npy", mmap_mode="r", allow_pickle=False)
    reports = []
    surface = QuerySurface(SHARED_BANK, LABEL_PROJECTION, seed=REFERENCE_SEED, verify_projection_hashes=False)
    try:
        bank: BankShard | None = None
        for match in chosen:
            shard = int(projection.matches["shard"][match])
            load_seconds = 0.0
            if bank is None or bank.shard_index != shard:
                bank = None
                tick = time.perf_counter()
                bank = BankShard(shard, projection.binding(shard), projection.source_root)
                load_seconds = time.perf_counter() - tick
            source = bank.source(int(projection.matches["local_match"][match]), match, projection.key(match))
            own = engagement["match_index"] == match
            rows = nominal[int(projection.matches["first_nominal_row"][match]):][:int(projection.matches["n_nominal_rows"][match])]
            first = int(projection.match_queries["first_query"][match])
            categories = {"control_start": starts[starts >= 0], "engagement_pre": engagement["query_pre"][own],
                          "engagement_post_h90": engagement["query_h90"][own],
                          "nominal_supported": first + np.asarray(rows["query_ref"])[np.asarray(rows["supported"], dtype=bool)]}
            reports.append(gate_match(surface, projection, source, plan["end_ms"][plan["match_index"] == match].tolist(), out, categories)
                           | {"shard_load_seconds": load_seconds})
    finally:
        surface.close()
    totals: dict[str, Any] = {}
    for name in ("query_arrays", "features", "history_text", "rows", "prefix", "roundtrip"):
        parts = [report["checks"][name] for report in reports]
        totals[name] = {"leaves_compared": sum(p["leaves_compared"] for p in parts), "elements_compared": sum(p["elements_compared"] for p in parts),
                        "mismatches": sum(p["mismatches"] for p in parts), "max_abs_diff": max((p["max_abs_diff"] for p in parts), default=0.0)}
    after = [row for report in reports for row in report["after_last_stored_query"]]
    bins = np.digitize(np.asarray([m for report in reports for m in report["control_end_minutes"]]), (15, 25))
    rss, peak = _rss()
    result = {"version": VERSION, "utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs, "seed": seed, "matches": chosen,
              "passed": bool(reports) and all(report["passed"] for report in reports),
              "stored_rows_compared": sum(report["stored_queries"] for report in reports),
              "control_end_rows": sum(report["control_ends"] for report in reports), "totals": totals,
              "control_end_game_bins": {"lt15": int((bins == 0).sum()), "15to25": int((bins == 1).sum()), "ge25": int((bins == 2).sum())},
              "plan_after_last_stored_query": {"ends": int(plan["after_last_stored_query"].sum()),
                                               "matches": int(len(np.unique(plan["match_index"][plan["after_last_stored_query"]])))},
              "after_last_stored_query_checked": len(after),
              "after_last_stale_under_last_query_cutoff": int(sum(row["stale_under_last_query_cutoff"] for row in after)),
              "reports": reports, "code_sha256": code_hashes(),
              "runtime": {"elapsed_seconds": time.perf_counter() - started, "rss_bytes": rss, "peak_rss_bytes": peak}}
    write_json_atomic(out / f"{GATE_NAME}.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--out", default=None, help=f"default: <labels>/{OUT_NAME} (a new directory; outputs/ is refused)")
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--equality-gate", action="store_true")
    parser.add_argument("--gate-matches", type=int, nargs="+", default=None)
    parser.add_argument("--gate-per-bin", type=int, default=3)
    parser.add_argument("--gate-after-last", type=int, default=3)
    args = parser.parse_args(argv)
    priority = below_normal_priority()
    labels = Path(args.labels)
    out = Path(args.out) if args.out else labels / OUT_NAME
    if args.equality_gate:
        result = equality_gate(labels, out, matches=args.gate_matches, per_bin=args.gate_per_bin, after_last=args.gate_after_last)
        print(json.dumps({key: result[key] for key in ("passed", "matches", "stored_rows_compared", "control_end_rows", "totals",
                                                         "after_last_stored_query_checked", "after_last_stale_under_last_query_cutoff")}
                         | {"priority": priority, "elapsed_seconds": result["runtime"]["elapsed_seconds"]}, indent=1))
        return 0 if result["passed"] else 1
    manifest = run(labels, out, limit_matches=args.limit_matches)
    print(json.dumps({key: manifest.get(key) for key in ("complete", "parts_complete", "parts_total", "this_run", "n_ends",
                                                         "timing_this_run", "runtime")} | {"priority": priority}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
