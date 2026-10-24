"""q spatial inputs S, A, M (reports/architecture_20261005/ARCHITECTURE_KO.md section 4.2).

Reads the 10 players' XY at the supported pre frame F_pre of each query from the observable-state
bank and computes, in float64 and vectorized over rows:
  sigma  median of every positive unordered-pair distance (45 per row) over the given unit rows
  S200   20 raw coordinates, then 45 pairs i<j x [dx/sigma, dy/sigma, d/sigma, w]
  edges  90 directed non-self edges j->i (40 ally + 50 enemy per row, from actual team ids):
         e_ji = ((x_j-x_i)/sigma, (y_j-y_i)/sigma, d_ij/sigma), w_ij = exp(-(d_ij/sigma)^2);
         S0: zero vectors and unit weights, relations kept, no XY path
  A      Phi_S, 64 = 50 node + 10 team + 4 cross values from coordinates and team membership only
  M      Phi_PS, for each node i and stat z: (sum_ally w_ij z_j / 4, sum_enemy w_ij z_j / 5, difference)
Sources: reports/feature_control_study_20261001/EXECUTION_CARDS.json (schema.tabular_spatial,
graph), HETEROGENEOUS_SPATIAL_GRAPH_DESIGN.md sections 2-4, CARTESIAN_COMPARISON_DESIGN.md
sections 3-4 and 8, author decisions D-I1, D-I3, D-Q2 (reports/architecture_20261005/
DECISIONS_20261005.json).  Column orders, ambiguities and the readings chosen:
reports/q_production_20261005/MEMO_spatial_KO.md.

No imputation, interpolation, alive masks, radius pruning or first-kill centering.  An exact (0,0)
is an ordinary observed coordinate (D-I1).  Standardization of these blocks belongs to the common
q preprocessing (D-Q2), not to this module.  Feature functions refuse non-finite input instead of
dropping rows; select rows with the support mask of ``read_query_xy`` first.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERSION = "q_spatial_v1_20261005"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
BANK = ROOT / "outputs/observable_state_20261002/full_static_preserved_20261002_1"

N_PLAYERS = 10
TEAM_SIZE = 5
BLUE, RED = 100, 200
TEAM_ORDER = (BLUE, RED)                     # team blocks of A: Blue first, then Red
TEAM_NAMES = ("blue", "red")
ALLY, ENEMY = 0, 1                           # relation codes of the directed edges
# Slot s of every XY / team / stat block is participantId s + 1: the bank stores
# xy_raw_minute[F, participantId - 1] (FIELD_MANIFEST.json participant_order 1..10, S.x index 0,
# S.y index 1).  This is the non-spatial canonical participant order of the pairs and edges.
SLOT_PARTICIPANT_IDS = tuple(range(1, N_PLAYERS + 1))
CHUNK_ROWS = 32768                           # bounds the [rows, 10, 10] work arrays only


def _frozen(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    array.setflags(write=False)
    return array


# Unordered pairs i<j in lexicographic participant order: (0,1), (0,2), ..., (0,9), (1,2), ..., (8,9).
PAIR_I, PAIR_J = (_frozen(v) for v in np.triu_indices(N_PLAYERS, k=1))
# Directed edges, receiver-major: for receiver i = 0..9, senders j = 0..9 without j = i.
# Edge index e = 9*i + j - (j > i); edge e carries sender EDGE_SRC[e] = j -> receiver EDGE_DST[e] = i.
EDGE_DST = _frozen(np.repeat(np.arange(N_PLAYERS), N_PLAYERS - 1))
EDGE_SRC = _frozen([j for i in range(N_PLAYERS) for j in range(N_PLAYERS) if j != i])
_TEAM_A, _TEAM_B = np.triu_indices(TEAM_SIZE, k=1)

S200_PAIR_FIELDS = ("dx_over_sigma", "dy_over_sigma", "d_over_sigma", "w")
A_NODE_FIELDS = ("ally_w_sum_div4", "enemy_w_sum_div5", "ally_minus_enemy_w",
                 "nearest_ally_d_over_sigma", "nearest_enemy_d_over_sigma")
A_TEAM_FIELDS = ("within_mean_d_over_sigma", "within_max_d_over_sigma", "mst_total_d_over_sigma",
                 "mst_max_edge_d_over_sigma", "laplacian_lambda2")
A_CROSS_FIELDS = ("cross_mean_d_over_sigma", "cross_min_d_over_sigma", "cross_mean_w", "centroid_d_over_sigma")
M_E_STATS = ("totalGold", "level")
M_R_STATS = ("health", "health_fraction", "attackDamage", "abilityPower", "armor", "magicResist")
M_SUMMARIES = ("ally_w_sum_div4", "enemy_w_sum_div5", "ally_minus_enemy")
S200_XY, S200_PAIRS = slice(0, 20), slice(20, 200)
A_NODE, A_TEAM, A_CROSS = slice(0, 50), slice(50, 60), slice(60, 64)

# Bank fields read per shard; every match field has an offset_<name> companion (bank.ShardBuilder).
SHARD_FIELDS = ("xy", "minute_ts", "query_ms", "query_frame_index", "team_ids")


def slot_participant_order() -> tuple[int, ...]:
    """participantId held by slots 0..9 of every array this module reads or returns."""

    return SLOT_PARTICIPANT_IDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _pid(slot: int) -> str:
    return f"p{SLOT_PARTICIPANT_IDS[slot]:02d}"


# --------------------------------------------------------------------------- XY at F_pre

@dataclass(frozen=True)
class QueryXY:
    """XY of the held frame of each query, in the order of the requested query indices.

    ``has_frame`` is False when no supported frame precedes the query (frame_index -1, or the query
    lies after the last frame); those rows keep NaN XY and frame_ms -1.  Rows with a frame keep the
    recorded values unchanged (float32 -> float64), so a non-finite coordinate stays as read and
    makes ``finite`` False.  ``support`` = has_frame and all 20 coordinates finite.  Nothing is imputed.
    """

    query_index: np.ndarray   # int64 [n] as requested (QUERY_TABLE rows)
    match_index: np.ndarray   # int64 [n] MATCH_TABLE rows
    query_ms: np.ndarray      # int64 [n]
    frame_index: np.ndarray   # int64 [n] match-local held frame, -1 without a frame
    frame_ms: np.ndarray      # int64 [n] held frame timestamp, -1 without a frame
    xy: np.ndarray            # float64 [n, 10, 2], slot s = participantId s + 1
    team_ids: np.ndarray      # int64 [n, 10] actual team id of every slot
    has_frame: np.ndarray     # bool [n]
    finite: np.ndarray        # bool [n]

    @property
    def support(self) -> np.ndarray:
        return self.has_frame & self.finite


def _verify_bank_layout(bank: Path) -> None:
    manifest = _read_json(Path(bank) / "FIELD_MANIFEST.json")
    if manifest.get("participant_order") != list(SLOT_PARTICIPANT_IDS):
        raise ValueError("bank participant order differs from slots 0..9 = participantId 1..10")
    if manifest.get("team_order") != list(TEAM_ORDER):
        raise ValueError("bank team order differs from [100, 200]")
    fields = {field.get("id"): field for field in manifest.get("ordered_fields", []) if isinstance(field, dict)}
    for name, index in (("S.x", 0), ("S.y", 1)):
        field = fields.get(name)
        if field is None or field.get("block") != "xy" or field.get("index") != index:
            raise ValueError(f"bank field {name} is not xy[..., {index}]")


def _segments(arrays: dict, field: str, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(arrays["offset_" + field], dtype=np.int64)
    if len(local) and (local.min() < 0 or local.max() >= len(offsets) - 1):
        raise ValueError(f"local match index outside the shard ({field})")
    return offsets[local], offsets[local + 1] - offsets[local]


class BankXYReader:
    """Read F_pre XY through the raw projection (QUERY_TABLE -> MATCH_TABLE shard/local_match).

    XY at a query = bank xy[offset_xy[local_match] + QUERY_TABLE.frame_index] (as in
    studies/v_input_projection_20261002/bank.py `_history` and run.py `build`).  Rows are grouped by
    shard and each BANK.npz is opened once per ``read`` call (``shards_opened`` records the opens).
    Bank query time and frame index are checked against the projection for every row, and the held
    frame must be the latest frame <= query (no future frame).  ``verify_hashes`` compares each opened
    BANK.npz with BANK_METADATA.json.
    """

    def __init__(self, projection: Path = PROJECTION, bank: Path | None = None, *, verify_hashes: bool = False):
        self.projection = Path(projection)
        self.metadata = _read_json(self.projection / "BANK_METADATA.json")
        self.bank = Path(self.metadata["source_root"]) if bank is None else Path(bank)
        _verify_bank_layout(self.bank)
        self.verify_hashes = bool(verify_hashes)
        self.queries = np.load(self.projection / "QUERY_TABLE.npy", mmap_mode="r", allow_pickle=False)
        self.matches = np.load(self.projection / "MATCH_TABLE.npy", mmap_mode="r", allow_pickle=False)
        for table, names in ((self.queries, ("match_index", "local_query", "query_ms", "frame_index")),
                             (self.matches, ("shard", "local_match"))):
            if not set(names) <= set(table.dtype.names or ()):
                raise ValueError(f"projection table lacks fields {sorted(set(names) - set(table.dtype.names or ()))}")
        self.shards_opened: list[int] = []

    def _load_shard(self, shard: int) -> dict[str, np.ndarray]:
        records = self.metadata["source_shards"]
        if not 0 <= shard < len(records):
            raise ValueError(f"shard {shard} is not bound in BANK_METADATA.json")
        binding = records[shard]
        path = self.bank / binding["directory"] / "BANK.npz"
        if self.verify_hashes and sha256_file(path) != binding["archive_sha256"]:
            raise ValueError(f"bound bank shard changed: {binding['directory']}/BANK.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for field in SHARD_FIELDS for name in (field, "offset_" + field)}
        self.shards_opened.append(int(shard))
        return arrays

    def read(self, query_indices: Sequence[int] | np.ndarray) -> QueryXY:
        index = np.asarray(query_indices)
        if index.ndim != 1 or index.dtype.kind not in "iu":
            raise TypeError("query indices must be a one-dimensional integer array")
        index = index.astype(np.int64)
        if len(index) and (index.min() < 0 or index.max() >= len(self.queries)):
            raise IndexError("query index outside QUERY_TABLE")
        refs = np.asarray(self.queries[index])
        match_index = refs["match_index"].astype(np.int64)
        local_query = refs["local_query"].astype(np.int64)
        query_ms = refs["query_ms"].astype(np.int64)
        frame_index = refs["frame_index"].astype(np.int64)
        records = np.asarray(self.matches[match_index])
        shard, local = records["shard"].astype(np.int64), records["local_match"].astype(np.int64)
        n = len(index)
        xy = np.full((n, N_PLAYERS, 2), np.nan, dtype=np.float64)
        team = np.zeros((n, N_PLAYERS), dtype=np.int64)
        frame_ms = np.full(n, -1, dtype=np.int64)
        has_frame = np.zeros(n, dtype=bool)
        for value in np.unique(shard):
            rows = np.flatnonzero(shard == value)
            arrays = self._load_shard(int(value))
            loc, lq, qms, fi = local[rows], local_query[rows], query_ms[rows], frame_index[rows]
            start, count = _segments(arrays, "query_ms", loc)
            if ((lq < 0) | (lq >= count)).any():
                raise ValueError("local query outside its match in the bank")
            if not np.array_equal(arrays["query_ms"][start + lq].astype(np.int64), qms):
                raise ValueError("bank and projection query times differ")
            start, count = _segments(arrays, "query_frame_index", loc)
            if ((lq >= count).any()
                    or not np.array_equal(arrays["query_frame_index"][start + lq].astype(np.int64), fi)):
                raise ValueError("bank and projection frame indices differ")
            xy_start, n_frames = _segments(arrays, "xy", loc)
            ts_start, n_ts = _segments(arrays, "minute_ts", loc)
            if not np.array_equal(n_frames, n_ts) or (n_ts < 1).any():
                raise ValueError("bank xy frames and frame clock differ")
            if (fi >= n_frames).any() or (fi < -1).any():
                raise ValueError("frame index outside the match frames")
            team_start, team_count = _segments(arrays, "team_ids", loc)
            if (team_count != N_PLAYERS).any():
                raise ValueError("bank team ids are not 10 per match")
            team[rows] = arrays["team_ids"][team_start[:, None] + np.arange(N_PLAYERS)]
            clock = np.asarray(arrays["minute_ts"], dtype=np.int64)
            held = fi >= 0
            safe = np.where(held, fi, 0)
            held_ms = clock[ts_start + safe]
            later = safe + 1 < n_ts
            next_ms = clock[ts_start + np.minimum(safe + 1, n_ts - 1)]
            if (held & (held_ms > qms)).any():
                raise ValueError("held frame lies after the query (future frame)")
            if (held & later & (next_ms <= qms)).any():
                raise ValueError("held frame is not the latest frame <= query")
            if (~held & (clock[ts_start] <= qms)).any():
                raise ValueError("a frame precedes a query recorded without a held frame")
            supported = held & (qms <= clock[ts_start + n_ts - 1])
            take = rows[supported]
            xy[take] = arrays["xy"][xy_start[supported] + fi[supported]].astype(np.float64)
            frame_ms[take] = held_ms[supported]
            has_frame[rows] = supported
        finite = has_frame & np.isfinite(xy).all(axis=(1, 2))
        return QueryXY(index, match_index, query_ms, np.where(has_frame, frame_index, -1), frame_ms,
                       xy, team, has_frame, finite)

    def close(self) -> None:
        for array in (self.queries, self.matches):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "BankXYReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def read_query_xy(query_indices, *, projection: Path = PROJECTION, bank: Path | None = None,
                  verify_hashes: bool = False) -> QueryXY:
    with BankXYReader(projection, bank, verify_hashes=verify_hashes) as reader:
        return reader.read(query_indices)


def xy_at_queries(query_indices, *, projection: Path = PROJECTION, bank: Path | None = None,
                  verify_hashes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(xy float64 [n,10,2], support bool [n]) at the held F_pre frame of each QUERY_TABLE row."""

    record = read_query_xy(query_indices, projection=projection, bank=bank, verify_hashes=verify_hashes)
    return record.xy, record.support


def zero_coordinate_players(xy) -> np.ndarray:
    """bool [n,10]: player XY exactly (0,0) (input of exclude_zero_coordinate_h90; no other use)."""

    array = np.asarray(xy, dtype=np.float64)
    return (array[..., 0] == 0.0) & (array[..., 1] == 0.0)


def support_counts(record: QueryXY) -> dict:
    """Census counts for a read: rows, support, and exact (0,0) players among supported rows."""

    supported = record.support
    zero = zero_coordinate_players(record.xy) & supported[:, None]
    return {"rows": int(len(supported)), "supported": int(supported.sum()),
            "unsupported_no_frame": int((~record.has_frame).sum()),
            "unsupported_nonfinite": int((record.has_frame & ~record.finite).sum()),
            "zero_coordinate_players": int(zero.sum()),
            "rows_with_zero_coordinate_player": int(zero.any(axis=1).sum()),
            "zero_coordinate_players_by_slot": [int(v) for v in zero.sum(axis=0)],
            "slot_participant_ids": list(SLOT_PARTICIPANT_IDS)}


def query_support_counts(query_indices, *, projection: Path = PROJECTION, bank: Path | None = None,
                         verify_hashes: bool = False) -> dict:
    """``support_counts`` of ``read_query_xy`` for a set of query indices (reads only their shards)."""

    return support_counts(read_query_xy(query_indices, projection=projection, bank=bank,
                                        verify_hashes=verify_hashes))


# --------------------------------------------------------------------------- geometry

def _check_xy(xy) -> np.ndarray:
    array = np.asarray(xy, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (N_PLAYERS, 2):
        raise ValueError(f"xy must have shape (n, {N_PLAYERS}, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("xy has non-finite values; pass supported rows only (QueryXY.support)")
    return array


def _check_sigma(sigma) -> float:
    value = np.asarray(sigma, dtype=np.float64)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0.0:
        raise ValueError("sigma must be one finite positive scale fitted on the scope rows")
    return float(value)


def _check_team_ids(team_ids, n: int | None = None) -> np.ndarray:
    array = np.asarray(team_ids)
    if array.ndim != 2 or array.shape[1] != N_PLAYERS or (n is not None and len(array) != n):
        raise ValueError(f"team_ids must have shape (n, {N_PLAYERS}) matching xy")
    if array.dtype.kind not in "iu":
        raise ValueError("team_ids must be integer team ids 100/200")
    array = array.astype(np.int64)
    if not np.isin(array, TEAM_ORDER).all() or ((array == BLUE).sum(axis=1) != TEAM_SIZE).any():
        raise ValueError("every row must be an exact 5v5 partition of team ids 100/200")
    return array


def _chunks(n: int) -> Iterator[slice]:
    for start in range(0, n, CHUNK_ROWS):
        yield slice(start, min(start + CHUNK_ROWS, n))


def _geometry(xy: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(d/sigma, w) as [rows, i, j] from delta[:, i, j] = xy_j - xy_i (the vector of edge j -> i)."""

    delta = xy[:, None, :, :] - xy[:, :, None, :]
    ratio = np.sqrt(delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]) / sigma
    return ratio, np.exp(-(ratio * ratio))


def _relations(team: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    same = team[:, :, None] == team[:, None, :]
    return same & ~np.eye(N_PLAYERS, dtype=bool), ~same


def _team_slots(team: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slots of Blue and of Red in each row, ascending within team (stable sort)."""

    order = np.argsort(team != BLUE, axis=1, kind="stable")
    return order[:, :TEAM_SIZE], order[:, TEAM_SIZE:]


def pair_distances(xy) -> np.ndarray:
    """float64 [n,45] Euclidean distances of the canonical pairs i<j (same arithmetic as all blocks)."""

    xy = _check_xy(xy)
    dx = xy[:, PAIR_J, 0] - xy[:, PAIR_I, 0]
    dy = xy[:, PAIR_J, 1] - xy[:, PAIR_I, 1]
    return np.sqrt(dx * dx + dy * dy)


def fit_sigma(xy) -> float:
    """Median of all positive unordered-pair distances of the given rows, each row weighted equally.

    EXECUTION_CARDS.json graph.sigma; D-I3 (unit rows).  Fit on the scope rows (q FIT or REFIT
    TRAIN) and apply the value unchanged to every other row of that scope.  Zero distances are left
    out only because they are not positive; a player at (0,0) contributes like any other (D-I1).
    """

    distances = pair_distances(xy)
    positive = distances[distances > 0.0]
    if not positive.size:
        raise ValueError("no positive pair distance: input validation failure")
    return float(np.median(positive))


def tabular_s200_names() -> list[str]:
    names = [f"S.{_pid(s)}.{axis}" for s in range(N_PLAYERS) for axis in ("x", "y")]
    names += [f"S.{_pid(i)}_{_pid(j)}.{field}" for i, j in zip(PAIR_I, PAIR_J) for field in S200_PAIR_FIELDS]
    return names


def tabular_s200(xy, sigma) -> tuple[np.ndarray, list[str]]:
    """[n,200]: x0,y0,...,x9,y9 (recorded map units) then, per pair i<j, [dx/s, dy/s, d/s, w].

    dx = x_j - x_i, dy = y_j - y_i: the pair (i, j) carries the vector of edge j -> i, so the reverse
    order negates dx and dy only (EXECUTION_CARDS.json schema.tabular_spatial.reverse).
    """

    xy, sigma = _check_xy(xy), _check_sigma(sigma)
    n = len(xy)
    dx = xy[:, PAIR_J, 0] - xy[:, PAIR_I, 0]
    dy = xy[:, PAIR_J, 1] - xy[:, PAIR_I, 1]
    ratio = np.sqrt(dx * dx + dy * dy) / sigma
    out = np.empty((n, 200), dtype=np.float64)
    out[:, S200_XY] = xy.reshape(n, 2 * N_PLAYERS)
    out[:, S200_PAIRS] = np.stack([dx / sigma, dy / sigma, ratio, np.exp(-(ratio * ratio))], axis=2).reshape(n, 180)
    return out, tabular_s200_names()


# --------------------------------------------------------------------------- edges

@dataclass(frozen=True)
class EdgeSet:
    """90 directed non-self edges in the fixed order EDGE_SRC -> EDGE_DST (receiver-major).

    relation: ALLY when sender and receiver have the same actual team id in that row, else ENEMY.
    vector:   ((x_j-x_i)/sigma, (y_j-y_i)/sigma, d_ij/sigma) for edge j -> i; zeros in S0.
    weight:   exp(-(d_ij/sigma)^2), never renormalized by its sum; ones in S0.
    log_weight: -(d_ij/sigma)^2, the logit of the GraphSAGE S1 softmax (EXECUTION_CARDS.json
              q.auxiliary_families graphsage S1); zeros in S0 (uniform mean over 9).
    """

    src: np.ndarray          # int64 [90]
    dst: np.ndarray          # int64 [90]
    relation: np.ndarray     # int8 [n, 90]
    vector: np.ndarray       # float64 [n, 90, 3]
    weight: np.ndarray       # float64 [n, 90]
    log_weight: np.ndarray   # float64 [n, 90]


def edge_names() -> list[str]:
    return [f"{_pid(j)}->{_pid(i)}" for j, i in zip(EDGE_SRC, EDGE_DST)]


def _edge_relation(team: np.ndarray) -> np.ndarray:
    return np.where(team[:, EDGE_SRC] == team[:, EDGE_DST], ALLY, ENEMY).astype(np.int8)


def edges(xy, team_ids, sigma) -> EdgeSet:
    xy, sigma = _check_xy(xy), _check_sigma(sigma)
    team = _check_team_ids(team_ids, len(xy))
    dx = xy[:, EDGE_SRC, 0] - xy[:, EDGE_DST, 0]
    dy = xy[:, EDGE_SRC, 1] - xy[:, EDGE_DST, 1]
    ratio = np.sqrt(dx * dx + dy * dy) / sigma
    log_weight = -(ratio * ratio)
    return EdgeSet(EDGE_SRC, EDGE_DST, _edge_relation(team), np.stack([dx / sigma, dy / sigma, ratio], axis=2),
                   np.exp(log_weight), log_weight)


def edges_s0(team_ids) -> EdgeSet:
    """S = 0: same 90 edges and relations from team membership only; no coordinate enters."""

    team = _check_team_ids(team_ids)
    n = len(team)
    return EdgeSet(EDGE_SRC, EDGE_DST, _edge_relation(team), np.zeros((n, len(EDGE_SRC), 3)),
                   np.ones((n, len(EDGE_SRC))), np.zeros((n, len(EDGE_SRC))))


# --------------------------------------------------------------------------- A = Phi_S

def phi_s_names() -> list[str]:
    names = [f"A.node.{_pid(s)}.{field}" for s in range(N_PLAYERS) for field in A_NODE_FIELDS]
    names += [f"A.team.{team}.{field}" for team in TEAM_NAMES for field in A_TEAM_FIELDS]
    names += [f"A.cross.{field}" for field in A_CROSS_FIELDS]
    return names


def _mst_total_max(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prim on dense [rows, m, m] Euclidean costs: (total length, longest edge) of the MST.

    Both values are the same for every minimum spanning tree, so ties do not matter; zero-length
    edges (coincident players) are ordinary edges.
    """

    n, m, _ = distance.shape
    rows = np.arange(n)
    in_tree = np.zeros((n, m), dtype=bool)
    in_tree[:, 0] = True
    best = distance[:, 0, :].copy()
    total, longest = np.zeros(n), np.zeros(n)
    for _ in range(m - 1):
        candidate = np.where(in_tree, np.inf, best)
        node = candidate.argmin(axis=1)
        edge = candidate[rows, node]
        total += edge
        longest = np.maximum(longest, edge)
        in_tree[rows, node] = True
        best = np.minimum(best, distance[rows, node, :])
    return total, longest


def _lambda2(weight: np.ndarray) -> np.ndarray:
    """Second-smallest eigenvalue of L = diag(W 1) - W with W_ii = 0 (symmetric, float64)."""

    diagonal = np.arange(weight.shape[1])
    matrix = weight.copy()
    matrix[:, diagonal, diagonal] = 0.0
    laplacian = -matrix
    laplacian[:, diagonal, diagonal] = matrix.sum(axis=2)
    return np.linalg.eigvalsh(laplacian)[:, 1]


def _phi_s_rows(xy: np.ndarray, team: np.ndarray, sigma: float) -> np.ndarray:
    n = len(xy)
    ratio, weight = _geometry(xy, sigma)
    ally, enemy = _relations(team)
    ally_sum = np.where(ally, weight, 0.0).sum(axis=2) / 4.0
    enemy_sum = np.where(enemy, weight, 0.0).sum(axis=2) / 5.0
    node = np.stack([ally_sum, enemy_sum, ally_sum - enemy_sum,
                     np.where(ally, ratio, np.inf).min(axis=2), np.where(enemy, ratio, np.inf).min(axis=2)], axis=2)
    blue, red = _team_slots(team)
    at = np.arange(n)[:, None, None]
    blocks = [node.reshape(n, N_PLAYERS * len(A_NODE_FIELDS))]
    for slots in (blue, red):
        team_ratio = ratio[at, slots[:, :, None], slots[:, None, :]]
        within = team_ratio[:, _TEAM_A, _TEAM_B]
        total, longest = _mst_total_max(team_ratio)
        lam = _lambda2(weight[at, slots[:, :, None], slots[:, None, :]])
        blocks.append(np.stack([within.mean(axis=1), within.max(axis=1), total, longest, lam], axis=1))
    cross_ratio = ratio[at, blue[:, :, None], red[:, None, :]].reshape(n, TEAM_SIZE * TEAM_SIZE)
    cross_weight = weight[at, blue[:, :, None], red[:, None, :]].reshape(n, TEAM_SIZE * TEAM_SIZE)
    rows = np.arange(n)[:, None]
    gap = xy[rows, red].mean(axis=1) - xy[rows, blue].mean(axis=1)
    centroid = np.sqrt(gap[:, 0] * gap[:, 0] + gap[:, 1] * gap[:, 1]) / sigma
    blocks.append(np.stack([cross_ratio.mean(axis=1), cross_ratio.min(axis=1), cross_weight.mean(axis=1), centroid], axis=1))
    return np.concatenate(blocks, axis=1)


def phi_s_A(xy, team_ids, sigma) -> tuple[np.ndarray, list[str]]:
    """[n,64] Phi_S (HETEROGENEOUS_SPATIAL_GRAPH_DESIGN.md section 4 table, in its row order).

    Node block (slot-major, 5 per slot): sum_ally w/4, sum_enemy w/5, their difference, nearest ally
    d/sigma, nearest enemy d/sigma.  Team block (Blue then Red, 5 each): mean and max of the 10
    within-team d/sigma, MST total and longest edge (Euclidean cost) /sigma, lambda2 of the weighted
    Laplacian.  Cross block: mean and min d/sigma and mean w over the 25 Blue-Red pairs, distance of
    the two team centroids (coordinate means) /sigma.
    """

    xy, sigma = _check_xy(xy), _check_sigma(sigma)
    team = _check_team_ids(team_ids, len(xy))
    out = np.empty((len(xy), 64), dtype=np.float64)
    for rows in _chunks(len(xy)):
        out[rows] = _phi_s_rows(xy[rows], team[rows], sigma)
    return out, phi_s_names()


def split_phi_s(a) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node [n,10,5], team [n,2,5] Blue/Red, cross [n,4]) views of a [n,64] Phi_S block (D-Q7 routing)."""

    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 64:
        raise ValueError("Phi_S block must have 64 columns")
    n = len(a)
    return (a[:, A_NODE].reshape(n, N_PLAYERS, len(A_NODE_FIELDS)),
            a[:, A_TEAM].reshape(n, len(TEAM_ORDER), len(A_TEAM_FIELDS)), a[:, A_CROSS])


# --------------------------------------------------------------------------- M = Phi_PS

def m_stat_names(e: bool = True, r: bool = True) -> tuple[str, ...]:
    """Card stat order of the active parents: E (totalGold, level) then R (six stats)."""

    return (M_E_STATS if e else ()) + (M_R_STATS if r else ())


def _check_stat_names(stat_names) -> tuple[str, ...]:
    names = tuple(stat_names)
    if not names or len(set(names)) != len(names) or not all(isinstance(v, str) and v for v in names):
        raise ValueError("stat_names must be unique non-empty strings")
    return names


def phi_ps_names(stat_names) -> list[str]:
    names = _check_stat_names(stat_names)
    return [f"M.{_pid(s)}.{z}.{summary}" for s in range(N_PLAYERS) for z in names for summary in M_SUMMARIES]


def phi_ps_M(xy, team_ids, sigma, stats, stat_names) -> tuple[np.ndarray, list[str]]:
    """[n, 10*3*k]: for slot i, stat z (given order), (sum_ally w_ij z_j/4, sum_enemy w_ij z_j/5, ally-enemy).

    Column index = (i*k + z)*3 + summary.  Stats are used in their recorded field units (D-Q2); no
    renormalization by sum(w) (CARTESIAN_COMPARISON_DESIGN.md section 4).  Non-finite stats are
    refused: missingness must be resolved by the caller, never imputed here.
    """

    xy, sigma = _check_xy(xy), _check_sigma(sigma)
    team = _check_team_ids(team_ids, len(xy))
    names = _check_stat_names(stat_names)
    values = np.asarray(stats, dtype=np.float64)
    if values.shape != (len(xy), N_PLAYERS, len(names)):
        raise ValueError(f"stats must have shape (n, {N_PLAYERS}, {len(names)})")
    if not np.isfinite(values).all():
        raise ValueError("stats have non-finite values; resolve missingness before Phi_PS")
    out = np.empty((len(xy), N_PLAYERS * len(names) * len(M_SUMMARIES)), dtype=np.float64)
    for rows in _chunks(len(xy)):
        _, weight = _geometry(xy[rows], sigma)
        ally, enemy = _relations(team[rows])
        z = values[rows]
        ally_sum = np.einsum("nij,njk->nik", np.where(ally, weight, 0.0), z) / 4.0
        enemy_sum = np.einsum("nij,njk->nik", np.where(enemy, weight, 0.0), z) / 5.0
        out[rows] = np.stack([ally_sum, enemy_sum, ally_sum - enemy_sum], axis=3).reshape(len(z), -1)
    return out, phi_ps_names(names)
