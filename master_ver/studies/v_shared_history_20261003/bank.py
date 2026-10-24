"""Match-local history cache pilot. Does not authorize full-bank training."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from studies.item_node_observed_20261002.bank import _event_from_value
from studies.item_node_observed_20261002.ledger import RecipientConfirmedItemLedger, CompactObservedState
from studies.v_model_input_20261003.history import HistoryAccumulator
from studies.v_model_input_20261003.view import make_view

VERSION = "shared_history_match_pilot_20261003_v1"
COMPACT_FIELDS = ("query_timestamp_ms", "known", "recorded_ledger_supported", "unknown", "reasons",
                  "item_ids", "counts", "offsets", "unassigned_history_observed", "unassigned_count",
                  "assigned_event_counts", "unassigned_event_counts")


def _integer(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("nonnegative integer required")
    return int(value)


def _source_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("source SHA256 required")
    return value


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loader_fingerprint(loader):
    return hashlib.sha256(json.dumps(loader.manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def iter_match_views(loader, match_index):
    """Read source histories once, emit every existing query in order."""
    match_index = _integer(match_index)
    if match_index >= len(loader.matches):
        raise IndexError("match outside bank")
    start = int(loader.match_queries[match_index]["first_query"])
    count = int(loader.match_queries[match_index]["n_queries"])
    query_ids = range(start, start + count)
    if count <= 0:
        raise ValueError("match has no queries")
    refs = loader.queries[start:start+count]
    if not np.all(refs["match_index"] == match_index) or np.any(np.diff(refs["query_ms"]) <= 0):
        raise ValueError("query ownership/order mismatch")
    accumulator = HistoryAccumulator()
    for record in loader._history(match_index).iter_prefix(int(refs[-1]["query_ms"])):
        accumulator.add(record)
    prefix = loader._item_loader().history_prefix(match_index, int(refs[-1]["local_query"]))
    transactions = [_event_from_value(value, index).transaction_mapping()
                    for partition in ("assigned", "unassigned") for index, value in enumerate(prefix[partition])]
    ledger = RecipientConfirmedItemLedger(transactions)
    for query_id, state in zip(query_ids, ledger.iter_compact_observed_states(refs["query_ms"].tolist()), strict=True):
        snapshot = ledger._snapshot(CompactObservedState(**{key: state[key] for key in COMPACT_FIELDS}))
        supplied = loader.query_input(query_id, include_histories=False)
        current = supplied["features"]["items"]
        for key in ("item_ids", "counts", "offsets", "recorded_ledger_supported"):
            if not np.array_equal(np.asarray(current[key]), np.asarray(snapshot[key])):
                raise ValueError(f"current item reference differs: {key}")
        cutoff = int(refs[query_id-start]["query_ms"])
        item = {"node": snapshot["assigned_history_by_participant"], "unassigned": snapshot["unassigned_history_context"]}
        view = make_view(supplied["features"], accumulator.snapshot_at(cutoff), item_history=item)
        yield {"query_id": query_id, "query_ms": cutoff, "view": view}


def save_match_cache(directory, records, *, match_index, source_hash, loader_manifest_sha256):
    """Exclusive directory, manifest last; incomplete writes never load."""
    source_hash = _source_hash(source_hash)
    loader_manifest_sha256 = _source_hash(loader_manifest_sha256)
    match_index = _integer(match_index)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    query_ids, query_ms = [], []
    path = directory / "history.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        for record in records:
            query_id, cutoff = _integer(record["query_id"]), _integer(record["query_ms"])
            if query_ids and (query_id != query_ids[-1]+1 or cutoff <= query_ms[-1]):
                raise ValueError("cache queries must be contiguous and time ordered")
            view = record["view"]
            payload = {"query_id": query_id, "query_ms": cutoff,
                       "nonitem": _plain(view["nonitem_history"]), "item": _plain(view["item_history"])}
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
            query_ids.append(query_id)
            query_ms.append(cutoff)
    if not query_ids:
        raise ValueError("empty cache prohibited")
    manifest = dict(version=VERSION, complete=True, training_ready=False, pilot=True,
                    match_index=match_index, source_hash=source_hash, loader_manifest_sha256=loader_manifest_sha256,
                    query_ids=query_ids, query_ms=query_ms,
                    bytes=path.stat().st_size, sha256=_digest(path))
    temp = directory / "MANIFEST.pending.json"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.rename(directory / "MANIFEST.json")
    return manifest


class MatchHistoryCache:
    def __init__(self, directory, *, source_hash):
        directory = Path(directory)
        self.manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
        meta = self.manifest
        if meta.get("version") != VERSION or meta.get("complete") is not True or meta.get("source_hash") != _source_hash(source_hash):
            raise ValueError("cache version/completion/source differs")
        _source_hash(meta.get("loader_manifest_sha256"))
        path = directory / "history.jsonl.gz"
        if _digest(path) != meta["sha256"]:
            raise ValueError("cache hash differs")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        ids = [_integer(row["query_id"]) for row in records]
        times = [_integer(row["query_ms"]) for row in records]
        if not ids or ids != meta["query_ids"] or times != meta["query_ms"] or any(b != a+1 for a, b in zip(ids, ids[1:])) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("cache query identity/order differs")
        self.records = dict(zip(ids, records))

    def query_view(self, loader, query_id):
        query_id = _integer(query_id)
        if loader_fingerprint(loader) != self.manifest["loader_manifest_sha256"]:
            raise ValueError("cache/source loader manifest differs")
        row = self.records[query_id]
        supplied = loader.query_input(query_id, include_histories=False)
        audit = supplied["audit"]
        if audit["match_index"] != self.manifest["match_index"] or audit["query_ms"] != row["query_ms"]:
            raise ValueError("cache/source query identity differs")
        return make_view(supplied["features"], row["nonitem"], item_history=row["item"])
