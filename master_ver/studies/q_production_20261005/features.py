"""q inputs at the pre time tau: raw input store, q-scoped preprocessing, vectorized transform, group masks.

The V input surface is reused unchanged (``studies/v_model_fit_input_20261003/common.py``
``semantic_row``: Blue view, actual team routing), but q fits its own preprocessing on its own
rows (EXECUTION_CARDS.json ``inner_stop.preprocessing``; NEXT_PHASE_DESIGN_KO.md line 17):
FIT = q TRAIN rows outside the q STOP matches, REFIT_TRAIN = all q TRAIN rows.

1. ``RawStoreWriter`` / ``build_raw_store``: one pass over the bank for the sorted pre queries ->
   unscaled numeric lanes (float32), static and ban categories, bag tokens per lane, group and
   slot (item ids as integers, non-item tokens through a dictionary).  Built in chunks that
   survive a restart; merged once.
2. ``fit_scope``: ``artifacts.fit_bundle`` on the scope rows (unit rows: FIT median imputation,
   mean/sd, zero sd -> 1; FIT vocabularies, PAD 0, UNK 1), DECISIONS_20261005.json D-I3.
3. ``transform``: vectorized equivalent of ``common.transform_rows`` (StructuredBatch + CSR table),
   tested for exact equality.
4. ``GroupMask``: E, C, R, D on/off with B always on (INPUT_POLICY.json ``main_q_common_groups``,
   ``q_crossed_factors``); inactive numeric channels become 0, categories PAD, bag entries are
   removed and observed counts recomputed, and the CSR table is rebuilt from the masked parts
   (one superset vocabulary, disabled values never reach a model: D-I2).  ``drop_map`` removes
   the map-progress block O (objective counts, ages, tokens) for the B_minus_map reference (D-Q2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.observable_state_20261002.kernel import STATIC_NAMES  # noqa: E402
from studies.v_full_preprocess_20261003 import artifacts  # noqa: E402
from studies.v_model_fit_input_20261003.common import (BAG_LAYOUT, BAG_NAMESPACES, CATEGORY_LAYOUT,  # noqa: E402
                                                       CATEGORY_NAMESPACES, ITEM_GROUPS, NUMERIC_DIMS)
from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch  # noqa: E402
from studies.v_model_fit_input_20261003.preprocess import UNK_INDEX, make_fit_evidence  # noqa: E402

VERSION = "q_features_v1_20261005"
LANE_SLOTS = {"node": 10, "team": 2, "global": 1}
GROUPS = ("B", "E", "C", "R", "D")
OBJECTIVE_EVENTS = frozenset({"ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "DRAGON_SOUL_GIVEN"})


def _channel_groups() -> dict[str, np.ndarray]:
    """Group of every numeric channel (semantic_row order; INPUT_FIELDS_KO.md, v_model_input CONTRACT parent_groups)."""

    node = np.empty(102, dtype=object)
    node[0:6] = "E"      # node_frame growth: level, xp, curGold, totalGold, laneCS, jgCS
    node[6:33] = "R"     # hp/mp and championStats
    node[33:46] = "D"    # ccTime and damageStats
    node[46:49] = "D"    # player KDA
    node[49:53] = "C"    # current-item count, ledger support, reserved 0, unassigned observed
    node[53:102] = "C"   # assigned item-history numerics
    team = np.full(8, "B", dtype=object)           # team growth, cached gold, team kills
    glob = np.empty(75, dtype=object)
    glob[0:26] = "B"     # elapsed minutes, frame age, growth delta, objective counts and ages
    glob[26:75] = "C"    # unassigned item-history numerics
    return {"node": node, "team": team, "global": glob}


CHANNEL_GROUPS = _channel_groups()
MAP_CHANNELS = {"global": np.arange(8, 26)}     # objective counts (9) and ages (9): block O


def nonitem_group(token: str, lane: str) -> str:
    """Group of one non-item counter token (v_model_input CONTRACT ``parent_groups``)."""

    event = json.loads(token)[0]
    if event == "CHAMPION_KILL":
        return "D" if lane == "node" else "B"
    if event in OBJECTIVE_EVENTS:
        return "B"
    if event == "LEVEL_UP":
        return "E"
    if event == "SKILL_LEVEL_UP":
        return "R"
    raise ValueError(f"non-item token of an unrouted event type: {event!r}")


def _slug(group: str) -> str:
    return group.replace(":", "_")


def _bag_keys() -> list[tuple[str, str]]:
    return [(lane, group) for lane in ("node", "team", "global") for group in BAG_LAYOUT[lane]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- raw store
class RawStoreWriter:
    """Write semantic rows for strictly increasing query indices into one chunk directory."""

    def __init__(self, directory: Path, query_indices: np.ndarray) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.queries = np.asarray(query_indices, dtype=np.int64)
        n = len(self.queries)
        if n == 0 or np.any(np.diff(self.queries) <= 0):
            raise ValueError("raw store queries must be non-empty and strictly increasing")
        self.n, self.position = n, 0
        mm = lambda name, dtype, shape: np.lib.format.open_memmap(self.dir / name, mode="w+", dtype=dtype, shape=shape)  # noqa: E731
        # float64 keeps semantic_row values exactly, so fitting and transforming match common.transform_rows bit for bit.
        self.numeric = {lane: mm(f"numeric_{lane}.f64.npy", np.float64, (n, *shape)) for lane, shape in
                        (("node", (10, 102)), ("team", (2, 8)), ("global", (75,)))}
        self.static = mm("static_node.i64.npy", np.int64, (n, 10, len(STATIC_NAMES)))
        self.bans = mm("bans_team.i64.npy", np.int64, (n, 2, 5))
        self.team_ids = mm("team_ids.i16.npy", np.int16, (n, 10))
        self.supported = mm("current_item_supported.u1.npy", np.uint8, (n, 10))
        self.focal_blue = mm("focal_is_blue.u1.npy", np.uint8, (n,))
        self.offsets = {key: mm(f"bag_{key[0]}_{_slug(key[1])}_offsets.i64.npy", np.int64, (n * LANE_SLOTS[key[0]] + 1,)) for key in _bag_keys()}
        self.streams = {key: (open(self.dir / f"bag_{key[0]}_{_slug(key[1])}_codes.i64.bin", "wb"),
                              open(self.dir / f"bag_{key[0]}_{_slug(key[1])}_counts.i32.bin", "wb")) for key in _bag_keys()}
        self.written = {key: 0 for key in _bag_keys()}
        self.tokens: dict[str, int] = {}

    def _code(self, namespace: str, value: Any) -> int:
        if namespace == "item":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("item bag tokens must be integer item ids")
            return int(value)
        if not isinstance(value, str) or not value:
            raise ValueError("non-item bag tokens must be non-empty strings")
        code = self.tokens.get(value)
        if code is None:
            code = self.tokens[value] = len(self.tokens)
        return code

    def add(self, query_index: int, row: Mapping[str, Any]) -> None:
        i = self.position
        if i >= self.n or int(query_index) != int(self.queries[i]):
            raise ValueError("rows must arrive in the declared query order")
        for lane in ("node", "team", "global"):
            self.numeric[lane][i] = row["numeric"][lane]
        self.static[i] = np.stack([row["categories"]["node"][name] for name in STATIC_NAMES], axis=1)
        self.bans[i] = np.stack([row["categories"]["team"][f"ban{k}"] for k in range(5)], axis=1)
        self.team_ids[i] = row["team_ids"]
        self.supported[i] = row["current_item_supported"]
        self.focal_blue[i] = bool(row["focal_is_blue"])
        for lane, group in _bag_keys():
            namespace = BAG_NAMESPACES[group]
            bags = row["bags"][lane][group]
            slots = LANE_SLOTS[lane]
            if len(bags) != slots:
                raise ValueError(f"{lane}:{group} needs {slots} bags")
            codes_stream, counts_stream = self.streams[(lane, group)]
            for slot, bag in enumerate(bags):
                keys = list(bag)
                codes = np.asarray([self._code(namespace, key) for key in keys], dtype=np.int64)
                counts = np.asarray([bag[key] for key in keys], dtype=np.int32)
                codes_stream.write(codes.tobytes())
                counts_stream.write(counts.tobytes())
                self.written[(lane, group)] += len(keys)
                self.offsets[(lane, group)][i * slots + slot + 1] = self.written[(lane, group)]
        self.position += 1

    def close(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.position != self.n:
            raise ValueError(f"raw chunk incomplete: {self.position}/{self.n} rows")
        for codes, counts in self.streams.values():
            codes.close()
            counts.close()
        for array in (*self.numeric.values(), self.static, self.bans, self.team_ids, self.supported, self.focal_blue, *self.offsets.values()):
            array.flush()
        tokens = sorted(self.tokens, key=self.tokens.get)
        (self.dir / "tokens_nonitem.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        np.save(self.dir / "queries.i64.npy", self.queries)
        record = {"version": VERSION, "rows": self.n, "queries_sha256": hashlib.sha256(self.queries.tobytes()).hexdigest(),
                  "bag_entries": {f"{lane}:{group}": n for (lane, group), n in self.written.items()}, "nonitem_tokens": len(tokens),
                  **(extra or {})}
        (self.dir / "CHUNK.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
        return record


class RawStore:
    """Read-only view of a (merged or single-chunk) raw store."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        record_path = self.dir / "CHUNK.json"
        self.record = json.loads(record_path.read_text(encoding="utf-8"))
        load = lambda name: np.load(self.dir / name, mmap_mode="r", allow_pickle=False)  # noqa: E731
        self.queries = np.load(self.dir / "queries.i64.npy", allow_pickle=False)
        self.n = len(self.queries)
        self.numeric = {lane: load(f"numeric_{lane}.f64.npy") for lane in ("node", "team", "global")}
        self.static, self.bans = load("static_node.i64.npy"), load("bans_team.i64.npy")
        self.team_ids, self.supported, self.focal_blue = load("team_ids.i16.npy"), load("current_item_supported.u1.npy"), load("focal_is_blue.u1.npy")
        self.offsets = {key: load(f"bag_{key[0]}_{_slug(key[1])}_offsets.i64.npy") for key in _bag_keys()}
        self.codes = {key: np.memmap(self.dir / f"bag_{key[0]}_{_slug(key[1])}_codes.i64.bin", dtype=np.int64, mode="r")
                      if (self.dir / f"bag_{key[0]}_{_slug(key[1])}_codes.i64.bin").stat().st_size else np.zeros(0, np.int64) for key in _bag_keys()}
        self.counts = {key: np.memmap(self.dir / f"bag_{key[0]}_{_slug(key[1])}_counts.i32.bin", dtype=np.int32, mode="r")
                       if (self.dir / f"bag_{key[0]}_{_slug(key[1])}_counts.i32.bin").stat().st_size else np.zeros(0, np.int32) for key in _bag_keys()}
        self.tokens = json.loads((self.dir / "tokens_nonitem.json").read_text(encoding="utf-8"))
        self._token_groups: dict[str, np.ndarray] = {}
        self._token_active: dict[tuple[str, "GroupMask"], np.ndarray] = {}
        self._objective: np.ndarray | None = None

    def positions(self, query_indices: np.ndarray) -> np.ndarray:
        query_indices = np.asarray(query_indices, dtype=np.int64)
        pos = np.searchsorted(self.queries, query_indices)
        if np.any(pos >= self.n) or np.any(self.queries[np.minimum(pos, self.n - 1)] != query_indices):
            raise KeyError("query index missing from the raw store")
        return pos

    def slot_ranges(self, key: tuple[str, str], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat [start, stop) of every (row, slot) for the given rows, row-major."""

        slots = LANE_SLOTS[key[0]]
        idx = (np.asarray(rows, dtype=np.int64)[:, None] * slots + np.arange(slots)[None, :]).reshape(-1)
        offsets = self.offsets[key]
        return offsets[idx].astype(np.int64), offsets[idx + 1].astype(np.int64)

    def gather(self, key: tuple[str, str], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Codes, counts and per-(row, slot) lengths for the rows."""

        start, stop = self.slot_ranges(key, rows)
        lengths = stop - start
        if lengths.sum() == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int32), lengths
        flat = np.repeat(start - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths) + np.arange(int(lengths.sum()))
        return np.asarray(self.codes[key][flat]), np.asarray(self.counts[key][flat]), lengths

    def token_groups(self, lane: str) -> np.ndarray:
        if lane not in self._token_groups:
            self._token_groups[lane] = np.asarray([nonitem_group(token, lane) for token in self.tokens], dtype=object)
        return self._token_groups[lane]

    def token_active(self, lane: str, mask: "GroupMask") -> np.ndarray:
        """Boolean per dictionary token: does it reach the model under the mask (computed once per lane and mask)."""

        key = (lane, mask)
        if key not in self._token_active:
            active = np.asarray([mask.active(group) for group in self.token_groups(lane)], dtype=bool)
            if mask.drop_map:
                if self._objective is None:
                    self._objective = np.asarray([json.loads(token)[0] in OBJECTIVE_EVENTS for token in self.tokens], dtype=bool)
                active &= ~self._objective
            self._token_active[key] = active
        return self._token_active[key]


def merge_chunks(chunk_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    """Concatenate chunks (in query order) into one store with one non-item token dictionary."""

    chunks = [RawStore(Path(d)) for d in chunk_dirs]
    chunks.sort(key=lambda c: int(c.queries[0]))
    queries = np.concatenate([c.queries for c in chunks])
    if np.any(np.diff(queries) <= 0):
        raise ValueError("chunks overlap or are out of order")
    tokens = sorted({token for c in chunks for token in c.tokens})
    index = {token: i for i, token in enumerate(tokens)}
    writer_dir = Path(out_dir)
    writer_dir.mkdir(parents=True, exist_ok=True)
    n = len(queries)
    mm = lambda name, dtype, shape: np.lib.format.open_memmap(writer_dir / name, mode="w+", dtype=dtype, shape=shape)  # noqa: E731
    outs = {f"numeric_{lane}.f64.npy": mm(f"numeric_{lane}.f64.npy", np.float64, (n, *shape)) for lane, shape in
            (("node", (10, 102)), ("team", (2, 8)), ("global", (75,)))}
    for name, dtype, shape in (("static_node.i64.npy", np.int64, (n, 10, len(STATIC_NAMES))), ("bans_team.i64.npy", np.int64, (n, 2, 5)),
                               ("team_ids.i16.npy", np.int16, (n, 10)), ("current_item_supported.u1.npy", np.uint8, (n, 10)),
                               ("focal_is_blue.u1.npy", np.uint8, (n,))):
        outs[name] = mm(name, dtype, shape)
    cursor = 0
    for c in chunks:
        sl = slice(cursor, cursor + c.n)
        for lane in ("node", "team", "global"):
            outs[f"numeric_{lane}.f64.npy"][sl] = c.numeric[lane]
        outs["static_node.i64.npy"][sl], outs["bans_team.i64.npy"][sl] = c.static, c.bans
        outs["team_ids.i16.npy"][sl], outs["current_item_supported.u1.npy"][sl] = c.team_ids, c.supported
        outs["focal_is_blue.u1.npy"][sl] = c.focal_blue
        cursor += c.n
    entries = {}
    for key in _bag_keys():
        lane, group = key
        slots = LANE_SLOTS[lane]
        offsets = mm(f"bag_{lane}_{_slug(group)}_offsets.i64.npy", np.int64, (n * slots + 1,))
        base, row_base = 0, 0
        with open(writer_dir / f"bag_{lane}_{_slug(group)}_codes.i64.bin", "wb") as codes_out, \
                open(writer_dir / f"bag_{lane}_{_slug(group)}_counts.i32.bin", "wb") as counts_out:
            for c in chunks:
                local = np.asarray(c.offsets[key], dtype=np.int64)
                offsets[row_base * slots:(row_base + c.n) * slots + 1] = local + base
                codes = np.asarray(c.codes[key], dtype=np.int64)
                if BAG_NAMESPACES[group] == "nonitem" and len(codes):
                    remap = np.asarray([index[token] for token in c.tokens], dtype=np.int64)
                    codes = remap[codes]
                codes_out.write(codes.tobytes())
                counts_out.write(np.asarray(c.counts[key], dtype=np.int32).tobytes())
                base += len(codes)
                row_base += c.n
        offsets.flush()
        entries[f"{lane}:{group}"] = int(base)
    for array in outs.values():
        array.flush()
    (writer_dir / "tokens_nonitem.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    np.save(writer_dir / "queries.i64.npy", queries)
    record = {"version": VERSION, "rows": int(n), "queries_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
              "bag_entries": entries, "nonitem_tokens": len(tokens), "chunks": [str(c.dir) for c in chunks],
              "chunk_records_sha256": [_sha256_file(c.dir / "CHUNK.json") for c in chunks]}
    (writer_dir / "CHUNK.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


def build_chunk(query_indices: np.ndarray, chunk_dir: Path, surface: Any = None) -> dict[str, Any]:
    """Read the bank for one chunk of sorted pre queries (resumable: a finished chunk is skipped)."""

    chunk_dir = Path(chunk_dir)
    queries = np.asarray(query_indices, dtype=np.int64)
    done = chunk_dir / "CHUNK.json"
    if done.is_file():
        record = json.loads(done.read_text(encoding="utf-8"))
        if record.get("queries_sha256") == hashlib.sha256(queries.tobytes()).hexdigest() and record.get("version") == VERSION:
            return record
        raise ValueError(f"{chunk_dir} holds a different chunk; move it aside before rebuilding")
    if surface is None:
        from studies.label_production_20261005.queries import PROJECTION, SHARED_BANK, QuerySurface
        surface = QuerySurface(SHARED_BANK, PROJECTION, seed=20261001, verify_projection_hashes=False)
    writer = RawStoreWriter(chunk_dir, queries)
    for query_index, row, _meta in surface.iter_query_rows(queries):
        writer.add(query_index, row)
    return writer.close()


def q_candidate_queries(engagements: Path) -> np.ndarray:
    """Sorted unique pre queries of every engagement in a q role (rows are fixed later by the label lock)."""

    with np.load(engagements, allow_pickle=False) as eng:
        role = eng["role"].astype(np.int64)
        queries = eng["query_pre"][np.isin(role, (0, 3, 4, 5))].astype(np.int64)
    return np.unique(queries)


def _chunk_worker(job: tuple[np.ndarray, str]) -> dict[str, Any]:
    queries, chunk_dir = job
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 - priority is a courtesy, not a requirement
        pass
    return build_chunk(queries, Path(chunk_dir))


def build_raw_store(queries: np.ndarray, out_dir: Path, *, workers: int = 4, chunk_rows: int = 20_000) -> dict[str, Any]:
    """Build the raw store in resumable chunks (parallel processes), then merge into ``out_dir/store``."""

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    out_dir = Path(out_dir)
    queries = np.asarray(queries, dtype=np.int64)
    if np.any(np.diff(queries) <= 0):
        raise ValueError("queries must be sorted and unique")
    jobs = [(queries[start:start + chunk_rows], str(out_dir / "chunks" / f"chunk_{k:04d}"))
            for k, start in enumerate(range(0, len(queries), chunk_rows))]
    progress_path = out_dir / "PROGRESS.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), []

    def publish(state: str) -> None:
        rows_done = sum(int(r["rows"]) for r in done)
        elapsed = time.time() - started
        payload = {"version": VERSION, "state": state, "chunks_done": len(done), "chunks_total": len(jobs), "rows_done": rows_done,
                   "rows_total": int(len(queries)), "elapsed_s": elapsed, "rows_per_s": rows_done / elapsed if elapsed > 0 else None}
        pending = progress_path.with_suffix(".json.pending")
        pending.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(pending, progress_path)

    publish("building")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_chunk_worker, job) for job in jobs]):
            done.append(future.result())
            publish("building")
    publish("merging")
    record = merge_chunks([Path(job[1]) for job in jobs], out_dir / "store")
    record["code_sha256"] = _sha256_file(Path(__file__))
    (out_dir / "store" / "BUILD.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    publish("done")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the q raw input store from the bank (resumable).")
    parser.add_argument("--engagements", default=r"D:\LOL_Training_20261004\labels_1\ENGAGEMENTS.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=20_000)
    parser.add_argument("--limit", type=int, default=0, help="first N queries only (smoke runs)")
    args = parser.parse_args(argv)
    queries = q_candidate_queries(Path(args.engagements))
    if args.limit:
        queries = queries[:args.limit]
    record = build_raw_store(queries, Path(args.out), workers=args.workers, chunk_rows=args.chunk_rows)
    print(json.dumps({key: record[key] for key in ("rows", "nonitem_tokens", "bag_entries")}, indent=1))
    return 0


# ----------------------------------------------------------------------------- q preprocessing scopes
def scope_categories(store: RawStore, rows: np.ndarray, chunk_rows: int = 4096) -> dict[str, set]:
    """Raw categories present in the scope rows, per vocabulary namespace (namespace_rows semantics)."""

    rows = np.sort(np.asarray(rows, dtype=np.int64))
    result: dict[str, set] = {namespace: set() for namespace in (*set(CATEGORY_NAMESPACES.values()), *set(BAG_NAMESPACES.values()))}
    for start in range(0, len(rows), chunk_rows):
        part = rows[start:start + chunk_rows]
        static, bans = np.asarray(store.static[part]), np.asarray(store.bans[part])
        for i, name in enumerate(STATIC_NAMES):
            result[CATEGORY_NAMESPACES[name]].update(int(v) for v in np.unique(static[:, :, i]))
        result["static:champion_id"].update(int(v) for v in np.unique(bans))
        for key in _bag_keys():
            namespace = BAG_NAMESPACES[key[1]]
            codes, _, _ = store.gather(key, part)
            values = np.unique(codes)
            if namespace == "item":
                result[namespace].update(int(v) for v in values)
            else:
                result[namespace].update(store.tokens[int(v)] for v in values)
    return result


def fit_scope(store: RawStore, rows: np.ndarray, scope: str, game_keys: Sequence[str], source_hash: str | None = None):
    """Fit one q preprocessing bundle (scope FIT or REFIT_TRAIN) on the given raw-store rows."""

    rows = np.asarray(rows, dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("scope rows must be sorted and unique")
    if len(game_keys) != len(rows):
        raise ValueError("one game key per scope row")
    flag = "fit" if scope == "FIT" else "refit"
    membership = {flag: [True] * len(rows), "dev": [False] * len(rows), "cal": [False] * len(rows), "target": [False] * len(rows)}
    if scope == "FIT":
        membership["stop"] = [False] * len(rows)
    evidence = make_fit_evidence(rows.tolist(), n_rows=store.n, context="final", scope=scope,
                                 game_keys=_full(store.n, rows, list(game_keys)), roles=_full(store.n, rows, ["TRAIN"] * len(rows)),
                                 membership={name: _full(store.n, rows, flags, False) for name, flags in membership.items()},
                                 source_hash=source_hash)
    return artifacts.fit_bundle(store.numeric, rows.tolist(), evidence, scope_categories(store, rows))


def _full(n: int, rows: np.ndarray, values: Sequence[Any], fill: Any = None) -> list[Any]:
    out = [fill] * n
    for row, value in zip(rows.tolist(), values):
        out[row] = value
    return out


# ----------------------------------------------------------------------------- vectorized transform
@dataclass(frozen=True)
class GroupMask:
    """Which input groups reach the model (B is always on)."""

    E: bool = True
    C: bool = True
    R: bool = True
    D: bool = True
    drop_map: bool = False

    def active(self, group: str) -> bool:
        return group == "B" or bool(getattr(self, group))

    @property
    def full(self) -> bool:
        return self.E and self.C and self.R and self.D and not self.drop_map


FULL = GroupMask()


def _vocab_lookup(vocab: Any, namespace: str, store: RawStore) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Vectorized mapping raw value -> vocabulary index (UNK for unseen/invalid)."""

    if namespace == "nonitem":
        mapping = vocab.mapping
        return np.asarray([mapping.get(token, UNK_INDEX) for token in store.tokens], dtype=np.int64)
    categories = np.asarray(vocab.categories, dtype=np.int64) if len(vocab.categories) else np.zeros(0, np.int64)
    return categories


def _map_ints(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    out = np.full(values.shape, UNK_INDEX, dtype=np.int64)
    if len(categories):
        pos = np.clip(np.searchsorted(categories, values), 0, len(categories) - 1)
        hit = (values >= 0) & (categories[pos] == values)
        out[hit] = pos[hit] + 2
    return out


class Transformer:
    """Apply one fitted q bundle (and a group mask) to raw-store rows."""

    def __init__(self, store: RawStore, preprocessor: Any) -> None:
        self.store, self.pre = store, preprocessor
        self.lookups = {}
        for namespace, vocab in preprocessor.categorical_map.items():
            self.lookups[namespace] = _vocab_lookup(vocab, namespace, store)
        self.ncat = {namespace: len(vocab.categories) + 2 for namespace, vocab in preprocessor.categorical_map.items()}

    def __call__(self, rows: np.ndarray, mask: GroupMask = FULL) -> StructuredBatch:
        rows = np.asarray(rows, dtype=np.int64)
        store, n = self.store, len(rows)
        numeric = {lane: self.pre.numeric_map[lane].transform(np.asarray(store.numeric[lane][rows], dtype=np.float64)) for lane in ("node", "team", "global")}
        for lane, array in numeric.items():
            groups = CHANNEL_GROUPS[lane]
            off = np.asarray([not mask.active(g) for g in groups])
            if mask.drop_map and lane in MAP_CHANNELS:
                off[MAP_CHANNELS[lane]] = True
            if off.any():
                array[..., off] = 0.0
        static = np.asarray(store.static[rows])
        bans = np.asarray(store.bans[rows])
        categories = {"node": {name: _map_ints(static[:, :, i], self.lookups[CATEGORY_NAMESPACES[name]]) for i, name in enumerate(STATIC_NAMES)},
                      "team": {f"ban{k}": _map_ints(bans[:, :, k], self.lookups["static:champion_id"]) for k in range(5)}, "global": {}}
        if not mask.active("C"):
            categories = {lane: {g: np.zeros_like(v) for g, v in groups.items()} for lane, groups in categories.items()}
        bags: dict[str, dict[str, BagInput]] = {"node": {}, "team": {}, "global": {}}
        for key in _bag_keys():
            lane, group = key
            slots = LANE_SLOTS[lane]
            namespace = BAG_NAMESPACES[group]
            codes, counts, lengths = store.gather(key, rows)
            if namespace == "item":
                keep = np.full(len(codes), mask.active("C"), dtype=bool)
                indices = _map_ints(codes, self.lookups["item"])
            elif mask.full or not len(codes):
                keep = np.ones(len(codes), dtype=bool)
                indices = self.lookups["nonitem"][codes] if len(codes) else np.zeros(0, np.int64)
            else:
                keep = store.token_active(lane, mask)[codes]
                indices = self.lookups["nonitem"][codes]
            slot_id = np.repeat(np.arange(n * slots), lengths)
            kept_lengths = np.bincount(slot_id[keep], minlength=n * slots) if len(codes) else np.zeros(n * slots, np.int64)
            flat_offsets = np.concatenate(([0], np.cumsum(kept_lengths)))
            offsets = flat_offsets[np.arange(n)[:, None] * slots + np.arange(slots + 1)[None, :]].astype(np.int64)
            weights = counts[keep].astype(np.float32)
            observed = np.bincount(slot_id[keep], weights=counts[keep].astype(np.float64), minlength=n * slots).astype(np.float32).reshape(n, slots) \
                if len(codes) else np.zeros((n, slots), np.float32)
            bags[lane][group] = BagInput(indices[keep].astype(np.int64), weights, offsets, observed)
        teams = np.asarray(store.team_ids[rows]).astype(np.int64)
        sides = np.asarray(store.focal_blue[rows], dtype=np.float32)
        supported = np.asarray(store.supported[rows], dtype=bool)
        if not mask.active("C"):
            supported = np.zeros_like(supported)
        csr = self.table(numeric, categories, bags, teams, sides)
        return StructuredBatch(node_numeric=numeric["node"], team_numeric=numeric["team"],
                               global_numeric=np.column_stack((numeric["global"], sides)), player_team_ids=teams,
                               current_item_supported=supported,
                               node_categories=categories["node"], team_categories=categories["team"], global_categories=categories["global"],
                               node_bags=bags["node"], team_bags=bags["team"], global_bags=bags["global"],
                               category_namespaces=CATEGORY_NAMESPACES, bag_namespaces=BAG_NAMESPACES, table_csr=csr)

    def layout(self) -> tuple[list[tuple[tuple, int, int]], int]:
        """Same block order and widths as ``common.table_layout``."""

        blocks, cursor = [], 0
        def add(key, width):
            nonlocal cursor
            blocks.append((key, cursor, int(width)))
            cursor += int(width)
        for lane, size in (("node", 10), ("team", 2), ("global", 1)):
            add(("numeric", lane), NUMERIC_DIMS[lane] * size)
        add(("routing", "node"), 10)
        add(("side", "global"), 1)
        for lane, size in (("node", 10), ("team", 2), ("global", 1)):
            for kind, layout, namespaces in (("category", CATEGORY_LAYOUT, CATEGORY_NAMESPACES), ("bag", BAG_LAYOUT, BAG_NAMESPACES)):
                for group in layout[lane]:
                    add((kind, lane, group), size * self.ncat[namespaces[group]])
                    if kind == "bag":
                        add(("bag_count", lane, group), size)
        return blocks, cursor

    def table(self, numeric, categories, bags, teams, sides) -> sparse.csr_matrix:
        n = len(teams)
        blocks, width = self.layout()
        rr, cc, vv = [], [], []
        def dense(array, offset):
            array = np.asarray(array).reshape(n, -1)
            x, y = np.nonzero(array)
            rr.append(x); cc.append(y + offset); vv.append(array[x, y].astype(np.float32))
        for key, start, _ in blocks:
            kind, lane, *tail = key
            if kind == "numeric":
                dense(numeric[lane], start)
            elif kind == "routing":
                dense((teams == 200).astype(np.float32), start)
            elif kind == "side":
                dense(sides, start)
            elif kind == "bag_count":
                dense(bags[lane][tail[0]].observed_count, start)
            elif kind == "category":
                group, size = tail[0], LANE_SLOTS[lane]
                ncat = self.ncat[CATEGORY_NAMESPACES[group]]
                values = np.asarray(categories[lane][group]).reshape(n, size)
                r, s = np.nonzero(values != 0)        # PAD (a masked category) contributes nothing
                rr.append(r); cc.append(start + s * ncat + values[r, s]); vv.append(np.ones(len(r), np.float32))
            else:
                group, size = tail[0], LANE_SLOTS[lane]
                ncat = self.ncat[BAG_NAMESPACES[group]]
                bag = bags[lane][group]
                offsets = np.asarray(bag.offsets)
                lengths = (offsets[:, 1:] - offsets[:, :-1]).reshape(-1)
                if lengths.sum():
                    slot_index = np.repeat(np.arange(n * size), lengths)
                    r, s = slot_index // size, slot_index % size
                    rr.append(r); cc.append(start + s * ncat + np.asarray(bag.indices)); vv.append(np.asarray(bag.counts, dtype=np.float32))
        rows = np.concatenate(rr) if rr else np.zeros(0, np.int64)
        cols = np.concatenate(cc) if cc else np.zeros(0, np.int64)
        vals = np.concatenate(vv) if vv else np.zeros(0, np.float32)
        csr = sparse.csr_matrix((vals, (rows, cols)), shape=(n, width), dtype=np.float32)
        csr.sum_duplicates()
        csr.sort_indices()
        return csr


if __name__ == "__main__":
    raise SystemExit(main())
