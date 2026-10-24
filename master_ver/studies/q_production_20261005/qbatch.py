"""q job inputs: one context per preprocessing scope, condition masks, family batches and sources.

A ``ScopeContext`` holds what every job of one scope shares (ARCHITECTURE_KO.md sections 4.2, 6.2):
the raw store and its q bundle fitted on the scope's TRAIN rows (FIT = TRAIN minus q STOP,
REFIT_TRAIN = all TRAIN), sigma (median positive pair distance over the scope rows, D-I3), the
medians that impute the M stats in recorded units (D-Q2), and the unit-row standardization of the
spatial blocks (median impute, mean, sd with sd 0 -> 1, the common numeric rule):
  - node blocks pooled over players: XY (2), node A (5), node M (8 stats x 3);
  - team A pooled over the two teams (5); cross A per column (4); S200 per column (200).
``assemble`` builds one family's inputs for a condition (factors G S A M E C R D, B always on):
  - structured families (G0, G1, tied, GraphSAGE, shared slot): the common superset lanes
    node 102 + 31, team 8 + 5, global 76 + 4 (+ 200 S200 columns for the shared-slot head, D-Q7);
    inactive blocks are constant zeros (EXECUTION_CARDS.json q.schema "inactive blocks 0/PAD");
  - pure-set auxiliary: node 102 + XY 2 only ("existing allowed node XY only");
  - tabular families (logistic, LightGBM, plain MLP): the base CSR table with dense S200, A (64)
    and M (240) columns appended;
  - FT-Transformer: numeric fields (structured superset + S200), categorical and bag fields;
  - graph tensors from raw XY and sigma, S0 constants without coordinates.
Sources yield ``qtrain.Batch`` in fixed q-row order; a background thread prepares the next batch
while the current one trains (the transform costs about 50 ms per 512 rows).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy import sparse

from studies.q_production_20261005 import features as qf
from studies.q_production_20261005 import qmodels, spatial
from studies.q_production_20261005.qtrain import Batch
from studies.v_model_fit_input_20261003.common import BAG_LAYOUT, BAG_NAMESPACES, CATEGORY_LAYOUT, CATEGORY_NAMESPACES
from studies.v_model_fit_input_20261003.models import StructuredBatch

M_STATS = spatial.M_E_STATS + spatial.M_R_STATS                    # E stats, then R stats (card order)
M_CHANNELS = {"totalGold": 3, "level": 0, "health": 19, "health_fraction": 6, "attackDamage": 13, "abilityPower": 9, "armor": 10, "magicResist": 25}
M_PARENT = {**{s: "E" for s in spatial.M_E_STATS}, **{s: "R" for s in spatial.M_R_STATS}}
NODE_EXTRA, TEAM_EXTRA, GLOBAL_EXTRA = 2 + 5 + 3 * len(M_STATS), 5, 4
DIMS = {"node": 102 + NODE_EXTRA, "team": 8 + TEAM_EXTRA, "global": 76 + GLOBAL_EXTRA}
PURE_DIMS = {"node": 104, "team": 8, "global": 76}
STRUCTURED = {"team_set_pair_readout", "relational_mpnn", "tied_message", "graphsage", "shared_slot_mlp", "pure_set"}
TABLE = {"logistic", "lightgbm", "plain_mlp"}
TRAIN = 0


@dataclass(frozen=True)
class Condition:
    G: int | None = None
    S: int = 1
    A: int = 1
    M: int = 1
    E: int = 1
    C: int = 1
    R: int = 1
    D: int = 1
    inputs: str = "condition"          # condition | B | B_minus_map

    @classmethod
    def from_job(cls, job: Any) -> "Condition":
        factors = dict(job.factors)
        if job.inputs in ("B", "B_minus_map"):
            return cls(G=None, S=0, A=0, M=0, E=0, C=0, R=0, D=0, inputs=job.inputs)
        return cls(**{k: (None if factors.get(k) is None else int(factors[k])) if k == "G" else int(factors.get(k, 1))
                      for k in ("G", "S", "A", "M", "E", "C", "R", "D")})

    @property
    def mask(self) -> qf.GroupMask:
        return qf.GroupMask(E=bool(self.E), C=bool(self.C), R=bool(self.R), D=bool(self.D), drop_map=self.inputs == "B_minus_map")

    def valid(self) -> bool:
        """The matrix constraints E or C or R, A <= S, M <= S, M <= max(E, R) (B-only references are exempt)."""

        if self.inputs != "condition":
            return True
        return bool((self.E or self.C or self.R) and self.A <= self.S and self.M <= self.S and self.M <= max(self.E, self.R))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per last-axis channel: median of finite values, then mean and sd after median imputation (sd 0 -> 1)."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    median = np.zeros(flat.shape[1])
    for c in range(flat.shape[1]):
        finite = np.isfinite(flat[:, c])
        median[c] = np.median(flat[finite, c]) if finite.any() else 0.0
    imputed = np.where(np.isfinite(flat), flat, median)
    mean, sd = imputed.mean(axis=0), imputed.std(axis=0)
    sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    return median, mean, sd


def _apply(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    median, mean, sd = stats
    clean = np.where(np.isfinite(values), values, median)
    return ((clean - mean) / sd).astype(np.float32)


CONTEXT_VERSION = "q_scope_context_v1_20261005"
STAT_BLOCKS = ("xy", "a_node", "a_team", "a_cross", "m_node", "s200")


class ScopeContext:
    """Shared state of one preprocessing scope (FIT or REFIT_TRAIN) over one q row table.

    Fitted once per (variant, scope) by ``q_prepare`` and saved (``save``); workers load it
    (``saved=directory``), which checks that the scope rows are the ones it was fitted on."""

    def __init__(self, store: qf.RawStore, rows: Mapping[str, np.ndarray], xy: np.ndarray, scope: str, chunk: int = 32768,
                 *, saved: Path | str | None = None) -> None:
        if scope not in ("FIT", "REFIT_TRAIN"):
            raise ValueError("scope must be FIT or REFIT_TRAIN")
        self.store, self.rows, self.scope = store, rows, scope
        self.positions = store.positions(np.asarray(rows["query_pre"], dtype=np.int64))
        self.xy = np.asarray(xy, dtype=np.float64)
        if self.xy.shape != (len(self.positions), 10, 2) or not np.isfinite(self.xy).all():
            raise ValueError("every q row needs finite pre-frame XY (a missing one would change the population, CCD 106)")
        self.team_ids = np.asarray(store.team_ids[self.positions], dtype=np.int64)
        train = np.asarray(rows["role"]) == TRAIN
        self.scope_rows = np.flatnonzero(train & (~np.asarray(rows["q_stop"], dtype=bool) if scope == "FIT" else train))
        self.record: dict[str, Any] = {}
        if saved is not None:
            self._load(Path(saved))
            return
        order = np.argsort(self.positions[self.scope_rows], kind="stable")
        keys = [bytes(k).decode("ascii") for k in np.asarray(rows["canonical_match_key"])[self.scope_rows][order]]
        self.pre = qf.fit_scope(store, self.positions[self.scope_rows][order], scope, keys)
        self.transform = qf.Transformer(store, self.pre)
        self.sigma = spatial.fit_sigma(self.xy[self.scope_rows])
        raw = np.stack([np.asarray(store.numeric["node"][self.positions[self.scope_rows], :, M_CHANNELS[s]], dtype=np.float64) for s in M_STATS], axis=-1)
        self.m_median = np.asarray([np.median(raw[..., k][np.isfinite(raw[..., k])]) for k in range(len(M_STATS))])
        blocks = [self._raw_blocks(self.scope_rows[i:i + chunk]) for i in range(0, len(self.scope_rows), chunk)]
        joined = {name: np.concatenate([b[name] for b in blocks]) for name in blocks[0]}
        self.stats = {name: _channel_stats(joined[name]) for name in STAT_BLOCKS}

    def scope_digest(self) -> str:
        """sha256 of the sorted raw-store positions of the scope rows (what the bundle was fitted on)."""

        return hashlib.sha256(np.ascontiguousarray(np.sort(self.positions[self.scope_rows]), dtype=np.int64).tobytes()).hexdigest()

    def save(self, directory: Path | str, provenance: Mapping[str, Any]) -> dict[str, Any]:
        """Write bundle/ (V artifact format), CONTEXT.npz (sigma, M medians, block stats) and CONTEXT.json (last)."""

        from studies.v_full_preprocess_20261003 import artifacts

        directory = Path(directory)
        if (directory / "CONTEXT.json").exists():
            raise FileExistsError(f"{directory} already holds a saved context")
        directory.mkdir(parents=True, exist_ok=True)
        artifacts.save_bundle(directory / "bundle", self.pre)
        arrays: dict[str, np.ndarray] = {"sigma": np.asarray([self.sigma], dtype=np.float64), "m_median": self.m_median}
        for name in STAT_BLOCKS:
            for part, value in zip(("median", "mean", "sd"), self.stats[name]):
                arrays[f"{name}__{part}"] = np.asarray(value, dtype=np.float64)
        np.savez(directory / "CONTEXT.npz", **arrays)
        record = {"version": CONTEXT_VERSION, "scope": self.scope, "scope_rows": int(len(self.scope_rows)), "scope_positions_sha256": self.scope_digest(),
                  "sigma": float(self.sigma), "m_stats": list(M_STATS), "m_median": [float(v) for v in self.m_median],
                  "context_npz_sha256": _sha256(directory / "CONTEXT.npz"), "bundle_manifest_sha256": _sha256(directory / "bundle" / "MANIFEST.json"),
                  **dict(provenance)}
        pending = directory / "CONTEXT.json.pending"
        pending.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(pending, directory / "CONTEXT.json")
        self.record = record
        return record

    def _load(self, directory: Path) -> None:
        from studies.v_full_preprocess_20261003 import artifacts

        record = json.loads((directory / "CONTEXT.json").read_text(encoding="utf-8"))
        if record.get("version") != CONTEXT_VERSION or record.get("scope") != self.scope:
            raise ValueError(f"{directory} is not a {self.scope} context of version {CONTEXT_VERSION}")
        if record["scope_positions_sha256"] != self.scope_digest():
            raise ValueError(f"{directory} was fitted on other scope rows than this row table gives")
        if _sha256(directory / "CONTEXT.npz") != record["context_npz_sha256"] or _sha256(directory / "bundle" / "MANIFEST.json") != record["bundle_manifest_sha256"]:
            raise ValueError(f"{directory} files differ from CONTEXT.json")
        self.pre = artifacts.load_bundle(directory / "bundle")
        if self.pre.scope != self.scope:
            raise ValueError("bundle scope differs from the context scope")
        with np.load(directory / "CONTEXT.npz", allow_pickle=False) as arrays:
            self.sigma = float(arrays["sigma"][0])
            self.m_median = np.asarray(arrays["m_median"], dtype=np.float64)
            self.stats = {name: tuple(np.asarray(arrays[f"{name}__{part}"], dtype=np.float64) for part in ("median", "mean", "sd")) for name in STAT_BLOCKS}
        self.transform = qf.Transformer(self.store, self.pre)
        self.record = record

    def _raw_blocks(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        xy, team = self.xy[rows], self.team_ids[rows]
        a = spatial.phi_s_A(xy, team, self.sigma)[0]
        a_node, a_team, a_cross = spatial.split_phi_s(a)
        stats = np.stack([np.asarray(self.store.numeric["node"][self.positions[rows], :, M_CHANNELS[s]], dtype=np.float64) for s in M_STATS], axis=-1)
        stats = np.where(np.isfinite(stats), stats, self.m_median)
        m = spatial.phi_ps_M(xy, team, self.sigma, stats, M_STATS)[0].reshape(len(rows), 10, len(M_STATS) * 3)
        return {"xy": xy, "a_node": a_node, "a_team": a_team, "a_cross": a_cross, "m_node": m, "s200": spatial.tabular_s200(xy, self.sigma)[0]}

    def blocks(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        raw = self._raw_blocks(np.asarray(rows))
        return {name: _apply(value, self.stats[name]) for name, value in raw.items()}

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {name: vocab.num_embeddings for name, vocab in self.pre.categorical_map.items()}

    def encoder_kwargs(self, family: str) -> dict[str, Any]:
        dims = PURE_DIMS if family == "pure_set" else DIMS
        global_dim = dims["global"] + (200 if family == "shared_slot_mlp" else 0)
        return {"node_numeric_dim": dims["node"], "team_numeric_dim": dims["team"], "global_numeric_dim": global_dim,
                "vocab_sizes": self.vocab_sizes, "category_namespaces": dict(CATEGORY_NAMESPACES), "bag_namespaces": dict(BAG_NAMESPACES),
                "category_layout": dict(CATEGORY_LAYOUT), "bag_layout": dict(BAG_LAYOUT)}


def _m_mask(cond: Condition) -> np.ndarray:
    """[3 * 8] bool: M columns whose stat parent (E or R) is active under the condition."""

    return np.repeat(np.asarray([bool(cond.M) and bool(getattr(cond, M_PARENT[s])) for s in M_STATS]), 3)


def assemble(context: ScopeContext, rows: np.ndarray, cond: Condition, family: str) -> Any:
    """Model inputs of one family for q rows ``rows`` under condition ``cond``."""

    rows = np.asarray(rows, dtype=np.int64)
    if not cond.valid():
        raise ValueError(f"invalid condition {cond}")
    base = context.transform(context.positions[rows], cond.mask)
    n = len(rows)
    blocks = context.blocks(rows) if cond.S else None
    if family in TABLE:
        dense = np.zeros((n, 200 + 64 + 240), dtype=np.float32)
        if cond.S:
            dense[:, :200] = blocks["s200"]
        if cond.A:
            dense[:, 200:250] = blocks["a_node"].reshape(n, -1)
            dense[:, 250:260] = blocks["a_team"].reshape(n, -1)
            dense[:, 260:264] = blocks["a_cross"]
        if cond.M:
            dense[:, 264:] = (blocks["m_node"] * _m_mask(cond)).reshape(n, -1)
        table = sparse.hstack((base.table_csr, sparse.csr_matrix(dense)), format="csr", dtype=np.float32)
        if family == "plain_mlp":
            return np.asarray(table.toarray(), dtype=np.float32)
        return table
    if family == "pure_set":
        xy = blocks["xy"] if cond.S else np.zeros((n, 10, 2), np.float32)
        return replace(base, node_numeric=np.concatenate((base.node_numeric, xy), axis=-1), table_csr=None)
    node_extra = np.zeros((n, 10, NODE_EXTRA), np.float32)
    team_extra = np.zeros((n, 2, TEAM_EXTRA), np.float32)
    global_extra = np.zeros((n, GLOBAL_EXTRA), np.float32)
    s200 = np.zeros((n, 200), np.float32)
    if cond.S:
        node_extra[..., 0:2] = blocks["xy"]
        s200 = blocks["s200"]
    if cond.A:
        node_extra[..., 2:7] = blocks["a_node"]
        team_extra[:] = blocks["a_team"]
        global_extra[:] = blocks["a_cross"]
    if cond.M:
        node_extra[..., 7:] = blocks["m_node"] * _m_mask(cond)
    global_numeric = np.concatenate((base.global_numeric, global_extra), axis=-1)
    if family == "shared_slot_mlp":
        global_numeric = np.concatenate((global_numeric, s200), axis=-1)
    structured = replace(base, node_numeric=np.concatenate((base.node_numeric, node_extra), axis=-1),
                         team_numeric=np.concatenate((base.team_numeric, team_extra), axis=-1), global_numeric=global_numeric, table_csr=None)
    if family == "ft_transformer":
        return ft_inputs(structured, s200)
    if family == "shared_slot_mlp":
        return structured
    graph = qmodels.graph_inputs(context.xy[rows] if cond.S else None, context.team_ids[rows], context.sigma if cond.S else None, spatial=bool(cond.S))
    return structured, graph


def ft_inputs(structured: StructuredBatch, s200: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[str, Any, int]]]:
    """(numeric fields [B, Fn], categorical fields [B, Fc], bag fields as (namespace, BagInput, slots))."""

    n = len(structured.node_numeric)
    numeric = np.concatenate((structured.node_numeric.reshape(n, -1), structured.team_numeric.reshape(n, -1), structured.global_numeric, s200), axis=1)
    categories = [structured.node_categories[g][:, s] for g in CATEGORY_LAYOUT["node"] for s in range(10)]
    categories += [structured.team_categories[g][:, s] for g in CATEGORY_LAYOUT["team"] for s in range(2)]
    bags = [(lane, group, bag) for lane, groups, in (("node", structured.node_bags), ("team", structured.team_bags), ("global", structured.global_bags))
            for group, bag in groups.items()]
    return numeric.astype(np.float32), np.stack(categories, axis=1), bags


def ft_fields(context: ScopeContext) -> dict[str, Any]:
    """Field declaration of the FT tokenizer for this scope (one token per field plus CLS, D-Q3)."""

    sizes = context.vocab_sizes
    numeric = 10 * DIMS["node"] + 2 * DIMS["team"] + DIMS["global"] + 200
    categories = [(CATEGORY_NAMESPACES[g], sizes[CATEGORY_NAMESPACES[g]]) for g in CATEGORY_LAYOUT["node"] for _ in range(10)]
    categories += [(CATEGORY_NAMESPACES[g], sizes[CATEGORY_NAMESPACES[g]]) for g in CATEGORY_LAYOUT["team"] for _ in range(2)]
    bags = []
    for lane, slots in (("node", 10), ("team", 2), ("global", 1)):
        for group in BAG_LAYOUT[lane]:
            bags += [(BAG_NAMESPACES[group], sizes[BAG_NAMESPACES[group]])] * slots
    return {"numeric_fields": numeric, "category_fields": categories, "bag_fields": bags}


def forward_for(family: str):
    """forward(model, inputs, device) -> logits [B] for one family."""

    import torch

    def structured(model, inputs, device):
        return model(inputs).reshape(-1)

    def graph(model, inputs, device):
        batch, g = inputs
        return model(batch, g).reshape(-1)

    def dense(model, inputs, device):
        return model(torch.as_tensor(inputs, dtype=torch.float32, device=device)).reshape(-1)

    def ft(model, inputs, device):
        numeric, categories, bags = inputs
        bag_values = []
        slots = {"node": 10, "team": 2, "global": 1}
        for lane, _group, bag in bags:
            bag_values += qmodels.ft_bag_inputs(bag, slots[lane], device)
        return model(torch.as_tensor(numeric, device=device), torch.as_tensor(categories, dtype=torch.long, device=device), bag_values)

    if family in ("team_set_pair_readout", "relational_mpnn", "tied_message", "graphsage"):
        return graph
    if family in ("shared_slot_mlp", "pure_set"):
        return structured
    if family == "plain_mlp":
        return dense
    if family == "ft_transformer":
        return ft
    raise ValueError(f"{family} is not a neural family")


def _prefetch(generator: Iterator[Batch], depth: int = 2) -> Iterator[Batch]:
    """Prepare the next batches on a background thread (numpy releases the GIL for most of the work)."""

    box: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def worker() -> None:
        try:
            for item in generator:
                box.put(item)
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            box.put(error)
        box.put(done)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        item = box.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def source(context: ScopeContext, rows: np.ndarray, cond: Condition, family: str, batch_size: int, weights: np.ndarray,
           *, prefetch: bool = True):
    """A ``qtrain.Source`` over q rows ``rows`` (fixed order) with training weights ``weights`` aligned to ``rows``."""

    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    y = np.asarray(context.rows["y"], dtype=np.float32)
    w_raw = np.asarray(context.rows["weight_raw"], dtype=np.float64)
    if (y[rows] < 0).any():
        raise ValueError("a source would read sealed (TEST) labels")

    def make() -> Iterator[Batch]:
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            yield Batch(assemble(context, part, cond, family), y[part], weights[start:start + len(part)], w_raw[part], part)

    def run() -> Iterator[Batch]:
        return _prefetch(make()) if prefetch else make()
    return run
