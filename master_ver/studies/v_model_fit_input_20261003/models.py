"""P2b model adapters for the common, already-transformed V input.

The parent pipeline owns raw reading, FIT/REFIT preprocessing, vocabulary
construction, and sparse matrix assembly.  This module only consumes those
artifacts.  In particular, no adapter learns a vocabulary or interprets an
integer identifier as a continuous feature during ``forward``.

The structured neural input is intentionally small and explicit:

``node_numeric``
    ``[P, 10, F_node]`` float32 values.
``team_numeric``
    ``[P, 2, F_team]`` float32 values.
``global_numeric``
    ``[P, F_global]`` float32 values.
``*_categories`` and ``*_bags``
    FIT-vocabulary indices/counts supplied by the common semantic bank.

Table models receive the parent's sparse ``table_csr`` directly.  Plain MLP
receives the parent's batch-densified ``dense_features`` directly.  The two
paths therefore share a semantic source without duplicating preprocessing in
this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn


MODEL_VERSION = "v_model_fit_input_p2b_20261003_v1"
PAD_INDEX = 0
UNK_INDEX = 1
EMBEDDING_DIM = 16
PARTICIPANTS = 10
TEAM_COUNT = 2
TEAM_IDS = (100, 200)
DEFAULT_NODE_NUMERIC_DIM = 102
DEFAULT_TEAM_NUMERIC_DIM = 8
DEFAULT_GLOBAL_NUMERIC_DIM = 75
FAMILIES = ("logistic", "lightgbm", "plain_mlp", "shared_slot_mlp", "team_set_mlp")


FAMILY_CARDS: dict[str, tuple[dict[str, Any], ...]] = {
    "logistic": tuple(
        {"id": f"lr_C{value:g}", "C": value}
        for value in (0.001, 0.01, 0.1, 1.0)
    ),
    "lightgbm": (
        {"id": "lgbm_l31_m200", "num_leaves": 31, "min_data_in_leaf": 200},
        {"id": "lgbm_l31_m500", "num_leaves": 31, "min_data_in_leaf": 500},
        {"id": "lgbm_l63_m200", "num_leaves": 63, "min_data_in_leaf": 200},
        {"id": "lgbm_l63_m500", "num_leaves": 63, "min_data_in_leaf": 500},
    ),
    "plain_mlp": (
        {"id": "plain_128_64_wd0.001", "hidden": (128, 64), "weight_decay": 0.001},
        {"id": "plain_128_64_wd0.0001", "hidden": (128, 64), "weight_decay": 0.0001},
        {"id": "plain_256_128_wd0.001", "hidden": (256, 128), "weight_decay": 0.001},
        {"id": "plain_256_128_wd0.0001", "hidden": (256, 128), "weight_decay": 0.0001},
    ),
    "shared_slot_mlp": (
        {"id": "v_slot_h64_wd0.001", "hidden_width": 64, "weight_decay": 0.001},
        {"id": "v_slot_h64_wd0.0001", "hidden_width": 64, "weight_decay": 0.0001},
        {"id": "v_slot_h128_wd0.001", "hidden_width": 128, "weight_decay": 0.001},
        {"id": "v_slot_h128_wd0.0001", "hidden_width": 128, "weight_decay": 0.0001},
    ),
    "team_set_mlp": (
        {"id": "v_set_h64_wd0.001", "hidden_width": 64, "weight_decay": 0.001},
        {"id": "v_set_h64_wd0.0001", "hidden_width": 64, "weight_decay": 0.0001},
        {"id": "v_set_h128_wd0.001", "hidden_width": 128, "weight_decay": 0.001},
        {"id": "v_set_h128_wd0.0001", "hidden_width": 128, "weight_decay": 0.0001},
    ),
}


def family_cards(family: str) -> tuple[dict[str, Any], ...]:
    """Return the immutable-by-convention four-card grid for one family."""

    key = str(family).lower()
    if key not in FAMILY_CARDS:
        raise ValueError(f"unknown V model family: {family!r}")
    return FAMILY_CARDS[key]


@dataclass(frozen=True)
class BagInput:
    """Pre-indexed ragged or dense count bag supplied by the common bank.

    ``indices`` and ``counts`` may be dense ``[P, N, L]`` arrays, or flattened
    arrays paired with ``offsets`` in ``[P, N+1]`` form.  The adapter never
    maps raw IDs here; values must already be PAD/UNK/FIT-vocabulary indices.
    ``observed_count`` is optional and, when present, is used verbatim as the
    count channel rather than recomputed from the bag.
    """

    indices: Any
    counts: Any
    offsets: Any | None = None
    observed_count: Any | None = None

    @classmethod
    def from_value(cls, value: Any) -> "BagInput":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "indices" not in value or "counts" not in value:
                raise ValueError("bag mapping requires indices and counts")
            return cls(
                value["indices"],
                value["counts"],
                value.get("offsets"),
                value.get("observed_count", value.get("count")),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) not in (2, 3, 4):
                raise ValueError("bag sequence must contain indices, counts, and optional offsets/count")
            return cls(value[0], value[1], value[2] if len(value) >= 3 else None, value[3] if len(value) == 4 else None)
        raise TypeError("bag must be BagInput or a mapping/sequence with indices and counts")


@dataclass
class StructuredBatch:
    """Common pretransformed batch consumed by structured neural adapters."""

    node_numeric: Any
    team_numeric: Any
    global_numeric: Any
    player_team_ids: Any | None = None
    side_flag: Any | None = None
    current_item_supported: Any | None = None
    node_categories: Mapping[str, Any] = field(default_factory=dict)
    team_categories: Mapping[str, Any] = field(default_factory=dict)
    global_categories: Mapping[str, Any] = field(default_factory=dict)
    node_bags: Mapping[str, Any] = field(default_factory=dict)
    team_bags: Mapping[str, Any] = field(default_factory=dict)
    global_bags: Mapping[str, Any] = field(default_factory=dict)
    category_namespaces: Mapping[str, str] = field(default_factory=dict)
    bag_namespaces: Mapping[str, str] = field(default_factory=dict)
    dense_features: Any | None = None
    table_csr: Any | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StructuredBatch":
        """Build a batch from the root reader's mapping without fitting anything."""

        def _first(*names: str) -> Any:
            for name in names:
                if name in value:
                    return value[name]
            return None

        def _groups(raw: Any, key: str) -> dict[str, Any]:
            if raw is None:
                return {}
            if not isinstance(raw, Mapping):
                raise TypeError(f"{key} must be a mapping of groups")
            nested = raw.get(key)
            if isinstance(nested, Mapping):
                return dict(nested)
            return dict(raw)

        categories = _first("categories", "categorical") or {}
        bags = _first("bags", "count_bags", "countbags") or {}
        return cls(
            node_numeric=_first("node_numeric", "node"),
            team_numeric=_first("team_numeric", "team"),
            global_numeric=_first("global_numeric", "global"),
            player_team_ids=_first("player_team_ids", "team_ids"),
            side_flag=_first("side_flag", "focal_side", "blue_side"),
            current_item_supported=_first("current_item_supported", "item_supported", "recorded_ledger_supported"),
            node_categories=_groups(categories, "node") or _groups(_first("node_categories"), "node"),
            team_categories=_groups(categories, "team") or _groups(_first("team_categories"), "team"),
            global_categories=_groups(categories, "global") or _groups(_first("global_categories"), "global"),
            node_bags=_groups(bags, "node") or _groups(_first("node_bags"), "node"),
            team_bags=_groups(bags, "team") or _groups(_first("team_bags"), "team"),
            global_bags=_groups(bags, "global") or _groups(_first("global_bags"), "global"),
            category_namespaces=dict(_first("category_namespaces") or {}),
            bag_namespaces=dict(_first("bag_namespaces") or {}),
            dense_features=_first("dense_features", "dense", "dense_input"),
            table_csr=_first("table_csr", "table", "csr"),
        )

    @classmethod
    def coerce(cls, value: "StructuredBatch | Mapping[str, Any]") -> "StructuredBatch":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise TypeError("model input must be StructuredBatch or a mapping")

    @property
    def batch_size(self) -> int:
        for value in (self.node_numeric, self.team_numeric, self.global_numeric, self.dense_features):
            if value is not None:
                return int(torch.as_tensor(value).shape[0])
        raise ValueError("cannot infer batch size from an empty structured batch")


# Short names used by callers that prefer the generic term.
ModelBatch = StructuredBatch
StructuredInput = StructuredBatch


@dataclass(frozen=True)
class NamespaceSpec:
    num_embeddings: int
    embedding_dim: int = EMBEDDING_DIM
    pad_index: int = PAD_INDEX
    unk_index: int = UNK_INDEX


def _namespace_spec(value: Any, *, embedding_dim: int) -> NamespaceSpec:
    if isinstance(value, NamespaceSpec):
        spec = value
    elif isinstance(value, Mapping):
        size = value.get("num_embeddings", value.get("vocab_size", value.get("size")))
        if size is None:
            raise ValueError("namespace spec requires num_embeddings/vocab_size")
        spec = NamespaceSpec(
            int(size),
            int(value.get("embedding_dim", embedding_dim)),
            int(value.get("pad_index", PAD_INDEX)),
            int(value.get("unk_index", UNK_INDEX)),
        )
    else:
        spec = NamespaceSpec(int(value), int(embedding_dim))
    if spec.num_embeddings < 2:
        raise ValueError("every namespace must reserve PAD=0 and UNK=1")
    if spec.pad_index != PAD_INDEX or spec.unk_index != UNK_INDEX:
        raise ValueError("namespace indices must reserve PAD=0 and UNK=1")
    if spec.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    return spec


def _merge_namespace_specs(
    vocab_sizes: Mapping[str, Any] | None,
    namespace_specs: Mapping[str, Any] | None,
    *,
    embedding_dim: int,
) -> dict[str, NamespaceSpec]:
    merged: dict[str, NamespaceSpec] = {}
    for source in (vocab_sizes or {}, namespace_specs or {}):
        for name, value in source.items():
            spec = _namespace_spec(value, embedding_dim=embedding_dim)
            if name in merged and merged[name] != spec:
                raise ValueError(f"conflicting namespace specification for {name!r}")
            merged[str(name)] = spec
    return merged


def _module_key(namespace: str, used: set[str]) -> str:
    base = "ns_" + re.sub(r"[^A-Za-z0-9_]", "_", str(namespace))
    if not base or base == "ns_":
        base = "ns_namespace"
    key = base
    index = 2
    while key in used:
        key = f"{base}_{index}"
        index += 1
    used.add(key)
    return key


class NamespaceEmbeddingBank(nn.Module):
    """One trainable embedding table per FIT-vocabulary namespace."""

    def __init__(self, specs: Mapping[str, NamespaceSpec | Mapping[str, Any] | int], *, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        normalized = {str(name): _namespace_spec(value, embedding_dim=embedding_dim) for name, value in specs.items()}
        self.specs = normalized
        self._keys: dict[str, str] = {}
        modules: dict[str, nn.Embedding] = {}
        used: set[str] = set()
        for namespace in sorted(normalized):
            key = _module_key(namespace, used)
            self._keys[namespace] = key
            spec = normalized[namespace]
            modules[key] = nn.Embedding(spec.num_embeddings, spec.embedding_dim, padding_idx=spec.pad_index)
        self.embeddings = nn.ModuleDict(modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for namespace, key in self._keys.items():
            embedding = self.embeddings[key]
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[self.specs[namespace].pad_index].zero_()

    def __contains__(self, namespace: str) -> bool:
        return str(namespace) in self._keys

    def table(self, namespace: str) -> nn.Embedding:
        try:
            return self.embeddings[self._keys[str(namespace)]]
        except KeyError as exc:
            raise KeyError(f"unknown FIT vocabulary namespace: {namespace!r}") from exc

    def forward(self, namespace: str, indices: Any) -> Tensor:
        values = torch.as_tensor(indices, dtype=torch.long, device=self.table(namespace).weight.device)
        table = self.table(namespace)
        if values.numel() and bool(((values < 0) | (values >= table.num_embeddings)).any()):
            raise ValueError(f"indices for namespace {namespace!r} are outside the fitted vocabulary")
        return table(values)


class _BagEncoder:
    def __init__(self, bank: NamespaceEmbeddingBank) -> None:
        self.bank = bank

    @staticmethod
    def _integer_offsets(offsets: Any) -> Tensor:
        raw = torch.as_tensor(offsets, device="cpu")
        if raw.dtype == torch.bool or raw.dtype.is_floating_point or raw.dtype.is_complex:
            raise ValueError("ragged bag offsets must use an integer dtype")
        return raw.to(dtype=torch.long)

    @classmethod
    def _offset_pairs(cls, offsets: Any, batch: int, nodes: int) -> list[tuple[int, int]]:
        value = cls._integer_offsets(offsets)
        if value.ndim == 1:
            if batch != 1 or value.numel() != nodes + 1:
                raise ValueError("one-dimensional offsets require batch=1 and length nodes+1")
            value = value.reshape(1, nodes + 1)
        if value.ndim == 2 and tuple(value.shape) == (batch, nodes + 1):
            return [(int(value[b, n]), int(value[b, n + 1])) for b in range(batch) for n in range(nodes)]
        if value.ndim == 2 and tuple(value.shape) == (batch * nodes, 2):
            return [(int(row[0]), int(row[1])) for row in value]
        if value.ndim == 3 and tuple(value.shape) == (batch, nodes, 2):
            return [(int(value[b, n, 0]), int(value[b, n, 1])) for b in range(batch) for n in range(nodes)]
        raise ValueError("offsets must have shape [batch,nodes+1], [batch*nodes,2], or [batch,nodes,2]")

    @classmethod
    def _canonical_offsets(cls, offsets: Any, batch: int, nodes: int, length: int) -> Tensor:
        value = cls._integer_offsets(offsets)
        if value.ndim != 2 or tuple(value.shape) != (batch, nodes + 1):
            raise ValueError("canonical ragged offsets must have shape [batch,nodes+1]")
        if value.numel() and int(value[0, 0]) != 0:
            raise ValueError("canonical ragged offsets must start at zero")
        if value.shape[1] > 1 and bool((torch.diff(value, dim=1) < 0).any()):
            raise ValueError("canonical ragged offsets must be nondecreasing")
        if batch > 1 and bool((value[1:, 0] != value[:-1, -1]).any()):
            raise ValueError("canonical ragged offsets must be globally contiguous")
        if int(value[-1, -1]) != int(length):
            raise ValueError("canonical ragged offsets must end at the flattened bag length")
        return torch.cat((value[:, :-1].reshape(-1), value[-1, -1:].reshape(1)))

    def encode(self, namespace: str, value: Any, *, batch: int, nodes: int) -> Tensor:
        bag = BagInput.from_value(value)
        table = self.bank.table(namespace)
        indices = torch.as_tensor(bag.indices, dtype=torch.long, device=table.weight.device)
        counts = torch.as_tensor(bag.counts, dtype=table.weight.dtype, device=table.weight.device)
        if indices.shape != counts.shape:
            raise ValueError("bag indices and counts must have equal shape")
        if counts.numel() and (not bool(torch.isfinite(counts).all()) or bool((counts < 0).any())):
            raise ValueError("bag counts must be finite and nonnegative")
        if indices.numel() and bool(((indices < 0) | (indices >= table.num_embeddings)).any()):
            raise ValueError(f"bag indices for namespace {namespace!r} are outside the fitted vocabulary")

        if bag.offsets is None:
            if indices.ndim == 2 and tuple(indices.shape) == (batch, nodes):
                indices = indices.unsqueeze(-1)
                counts = counts.unsqueeze(-1)
            if indices.ndim != 3 or tuple(indices.shape[:2]) != (batch, nodes):
                raise ValueError("dense bags must have shape [batch,nodes,length]")
            pooled = (table(indices) * counts.unsqueeze(-1)).sum(dim=-2)
            observed = counts.sum(dim=-1)
        else:
            flat_indices = indices.reshape(-1)
            flat_counts = counts.reshape(-1)
            offsets_cpu = self._integer_offsets(bag.offsets)
            if offsets_cpu.ndim == 2 and tuple(offsets_cpu.shape) == (batch, nodes + 1):
                offsets = self._canonical_offsets(bag.offsets, batch, nodes, flat_indices.numel()).to(table.weight.device)
                pooled = nn.functional.embedding_bag(
                    flat_indices,
                    table.weight,
                    offsets,
                    mode="sum",
                    per_sample_weights=flat_counts,
                    include_last_offset=True,
                    padding_idx=PAD_INDEX,
                ).reshape(batch, nodes, -1)
                prefix = torch.cat((torch.zeros(1, dtype=flat_counts.dtype, device=flat_counts.device), flat_counts.cumsum(dim=0)))
                observed = (prefix[offsets[1:]] - prefix[offsets[:-1]]).reshape(batch, nodes)
            else:
                pairs = self._offset_pairs(bag.offsets, batch, nodes)
                pooled_rows: list[Tensor] = []
                observed_rows: list[Tensor] = []
                for start, end in pairs:
                    if start < 0 or end < start or end > flat_indices.numel():
                        raise ValueError("ragged bag offsets are outside flattened arrays")
                    local_indices = flat_indices[start:end]
                    local_counts = flat_counts[start:end]
                    pooled_rows.append((table(local_indices) * local_counts.unsqueeze(-1)).sum(dim=0))
                    observed_rows.append(local_counts.sum())
                pooled = torch.stack(pooled_rows, dim=0).reshape(batch, nodes, -1)
                observed = torch.stack(observed_rows, dim=0).reshape(batch, nodes)
        if bag.observed_count is not None:
            explicit = torch.as_tensor(bag.observed_count, dtype=table.weight.dtype, device=table.weight.device)
            if explicit.numel() != batch * nodes:
                raise ValueError("observed_count must have shape [batch,nodes]")
            if explicit.numel() and (not bool(torch.isfinite(explicit).all()) or bool((explicit < 0).any())):
                raise ValueError("observed_count must be finite and nonnegative")
            observed = explicit.reshape(batch, nodes)
        return torch.cat((pooled, observed.unsqueeze(-1)), dim=-1)


def _group_namespace(group: str, mapping: Mapping[str, str], bank: NamespaceEmbeddingBank) -> str:
    namespace = mapping.get(group)
    if namespace is None and group in bank:
        namespace = group
    if namespace is None and len(bank._keys) == 1:
        namespace = next(iter(bank._keys))
    if namespace is None:
        raise KeyError(f"no FIT namespace declared for input group {group!r}")
    if namespace not in bank:
        raise KeyError(f"input group {group!r} references unknown FIT namespace {namespace!r}")
    return namespace


class StructuredFeatureEncoder(nn.Module):
    """Encode the typed batch for player-level structured models."""

    def __init__(
        self,
        *,
        node_numeric_dim: int = DEFAULT_NODE_NUMERIC_DIM,
        team_numeric_dim: int = DEFAULT_TEAM_NUMERIC_DIM,
        global_numeric_dim: int = DEFAULT_GLOBAL_NUMERIC_DIM,
        vocab_sizes: Mapping[str, Any] | None = None,
        namespace_specs: Mapping[str, Any] | None = None,
        category_namespaces: Mapping[str, str] | None = None,
        bag_namespaces: Mapping[str, str] | None = None,
        category_layout: Mapping[str, Sequence[str]] | None = None,
        bag_layout: Mapping[str, Sequence[str]] | None = None,
        include_side_flag: bool = True,
        current_item_group: str | None = "current_item",
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.node_numeric_dim = int(node_numeric_dim)
        self.team_numeric_dim = int(team_numeric_dim)
        self.global_numeric_dim = int(global_numeric_dim)
        if min(self.node_numeric_dim, self.team_numeric_dim, self.global_numeric_dim) < 0:
            raise ValueError("numeric dimensions must be nonnegative")
        specs = _merge_namespace_specs(vocab_sizes, namespace_specs, embedding_dim=embedding_dim)
        self.bank = NamespaceEmbeddingBank(specs, embedding_dim=embedding_dim)
        self.category_namespaces = {str(k): str(v) for k, v in (category_namespaces or {}).items()}
        self.bag_namespaces = {str(k): str(v) for k, v in (bag_namespaces or {}).items()}
        self._category_layout_explicit = category_layout is not None
        self._bag_layout_explicit = bag_layout is not None
        self.category_layout = self._layout(category_layout, self.category_namespaces)
        self.bag_layout = self._layout(bag_layout, self.bag_namespaces)
        self.include_side_flag = bool(include_side_flag)
        if current_item_group is not None and current_item_group not in self.bag_layout.get("node", ()):
            candidates = [
                group
                for group in self.bag_layout.get("node", ())
                if "current" in group.lower() and "item" in group.lower()
            ]
            current_item_group = candidates[0] if candidates else current_item_group
        self.current_item_group = current_item_group
        self.bag_encoder = _BagEncoder(self.bank)
        self.embedding_dim = int(embedding_dim)
        self.unsupported_current_item_embedding = nn.Parameter(torch.zeros(self.embedding_dim))
        nn.init.normal_(self.unsupported_current_item_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _layout(
        layout: Mapping[str, Sequence[str]] | None,
        namespaces: Mapping[str, str],
    ) -> dict[str, tuple[str, ...]]:
        if layout is None:
            # Root adapters should pass an explicit layout.  The fallback is
            # deterministic and keeps small synthetic fixtures convenient.
            return {"node": tuple(sorted(namespaces)), "team": tuple(), "global": tuple()}
        result: dict[str, tuple[str, ...]] = {}
        for lane in ("node", "team", "global"):
            values = layout.get(lane, ())
            if isinstance(values, (str, bytes, bytearray)):
                raise ValueError(f"{lane} layout must be an ordered sequence of group names")
            names = tuple(str(value) for value in values)
            if len(set(names)) != len(names):
                raise ValueError(f"{lane} layout contains duplicate group names")
            result[lane] = names
        return result

    def _numeric(self, value: Any, shape: tuple[int, ...], name: str, device: torch.device) -> Tensor:
        if value is None:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains nonfinite values; preprocessing belongs to the root pipeline")
        return tensor

    def _categorical_group(self, namespace: str, value: Any, *, batch: int, nodes: int, device: torch.device) -> Tensor:
        ids = torch.as_tensor(value, dtype=torch.long, device=device)
        if ids.ndim == 1 and ids.numel() == batch * nodes:
            ids = ids.reshape(batch, nodes)
        if ids.ndim not in (2, 3) or tuple(ids.shape[:2]) != (batch, nodes):
            raise ValueError("categorical group must have shape [batch,nodes] or [batch,nodes,k]")
        embedded = self.bank(namespace, ids)
        return embedded.reshape(batch, nodes, -1)

    def _encode_side(
        self,
        categories: Mapping[str, Any],
        bags: Mapping[str, Any],
        *,
        batch: int,
        nodes: int,
        device: torch.device,
        current_item_supported: Any | None = None,
    ) -> Tensor:
        pieces: list[Tensor] = []
        declared = self.category_layout.get("node" if nodes == PARTICIPANTS else "team", ())
        observed_categories = set(categories)
        if self._category_layout_explicit:
            if observed_categories != set(declared):
                missing = sorted(set(declared) - observed_categories)
                extra = sorted(observed_categories - set(declared))
                raise ValueError(f"category groups must exactly match the declared layout (missing={missing}, extra={extra})")
            groups = declared
        else:
            groups = declared or tuple(sorted(categories))
        for group in groups:
            namespace = _group_namespace(group, self.category_namespaces, self.bank)
            if group in categories:
                pieces.append(self._categorical_group(namespace, categories[group], batch=batch, nodes=nodes, device=device))
            else:
                pieces.append(torch.zeros((batch, nodes, self.bank.specs[namespace].embedding_dim), dtype=torch.float32, device=device))
        declared_bags = self.bag_layout.get("node" if nodes == PARTICIPANTS else "team", ())
        observed_bags = set(bags)
        if self._bag_layout_explicit:
            if observed_bags != set(declared_bags):
                missing = sorted(set(declared_bags) - observed_bags)
                extra = sorted(observed_bags - set(declared_bags))
                raise ValueError(f"bag groups must exactly match the declared layout (missing={missing}, extra={extra})")
            bag_groups = declared_bags
        else:
            bag_groups = declared_bags or tuple(sorted(bags))
        if self.current_item_group is not None and self.current_item_group in bag_groups and current_item_supported is None:
            raise ValueError("current_item_supported is required when current_item is declared")
        for group in bag_groups:
            namespace = _group_namespace(group, self.bag_namespaces, self.bank)
            if group in bags:
                encoded = self.bag_encoder.encode(namespace, bags[group], batch=batch, nodes=nodes)
                if current_item_supported is not None and group == self.current_item_group:
                    support = torch.as_tensor(current_item_supported, device=device)
                    if support.dtype != torch.bool:
                        raise ValueError("current_item_supported must use a boolean dtype")
                    if tuple(support.shape) != (batch, nodes):
                        raise ValueError("current_item_supported must have shape [batch,10]")
                    learned = self.unsupported_current_item_embedding.reshape(1, 1, -1).expand(batch, nodes, -1)
                    pooled = torch.where(support.unsqueeze(-1), encoded[..., :-1], learned)
                    count = torch.where(support, encoded[..., -1], torch.zeros_like(encoded[..., -1]))
                    encoded = torch.cat((pooled, count.unsqueeze(-1)), dim=-1)
                pieces.append(encoded)
            else:
                pieces.append(torch.zeros((batch, nodes, self.bank.specs[namespace].embedding_dim + 1), dtype=torch.float32, device=device))
        if pieces:
            return torch.cat(pieces, dim=-1)
        return torch.zeros((batch, nodes, 0), dtype=torch.float32, device=device)

    def _encode_global_side(
        self,
        categories: Mapping[str, Any],
        bags: Mapping[str, Any],
        *,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        pieces: list[Tensor] = []
        declared = self.category_layout.get("global", ())
        observed_categories = set(categories)
        if self._category_layout_explicit:
            if observed_categories != set(declared):
                missing = sorted(set(declared) - observed_categories)
                extra = sorted(observed_categories - set(declared))
                raise ValueError(f"global category groups must exactly match the declared layout (missing={missing}, extra={extra})")
            groups = declared
        else:
            groups = declared or tuple(sorted(categories))
        for group in groups:
            namespace = _group_namespace(group, self.category_namespaces, self.bank)
            if group not in categories:
                pieces.append(torch.zeros((batch, self.bank.specs[namespace].embedding_dim), dtype=torch.float32, device=device))
                continue
            ids = torch.as_tensor(categories[group], dtype=torch.long, device=device)
            if ids.ndim == 1:
                ids = ids.unsqueeze(-1)
            if ids.ndim != 2 or ids.shape[0] != batch:
                raise ValueError("global categorical group must have shape [batch] or [batch,k]")
            pieces.append(self.bank(namespace, ids).reshape(batch, -1))
        declared_bags = self.bag_layout.get("global", ())
        observed_bags = set(bags)
        if self._bag_layout_explicit:
            if observed_bags != set(declared_bags):
                missing = sorted(set(declared_bags) - observed_bags)
                extra = sorted(observed_bags - set(declared_bags))
                raise ValueError(f"global bag groups must exactly match the declared layout (missing={missing}, extra={extra})")
            bag_groups = declared_bags
        else:
            bag_groups = declared_bags or tuple(sorted(bags))
        for group in bag_groups:
            namespace = _group_namespace(group, self.bag_namespaces, self.bank)
            if group in bags:
                encoded = self.bag_encoder.encode(namespace, bags[group], batch=batch, nodes=1)
                pieces.append(encoded.reshape(batch, -1))
            else:
                pieces.append(torch.zeros((batch, self.bank.specs[namespace].embedding_dim + 1), dtype=torch.float32, device=device))
        if pieces:
            return torch.cat(pieces, dim=-1)
        return torch.zeros((batch, 0), dtype=torch.float32, device=device)

    @staticmethod
    def _team_indices(team_ids: Any | None, batch: int, device: torch.device) -> Tensor:
        if team_ids is None:
            raise ValueError("player_team_ids are required for structured batches")
        raw = torch.as_tensor(team_ids, device=device)
        if raw.dtype == torch.bool or raw.dtype.is_floating_point or raw.dtype.is_complex:
            raise ValueError("player_team_ids must use integer 100/200 labels")
        values = raw.to(dtype=torch.long)
        if tuple(values.shape) != (batch, PARTICIPANTS):
            raise ValueError("player_team_ids must have shape [batch,10]")
        if not bool(torch.all((values == 100) | (values == 200))):
            raise ValueError("player_team_ids must contain only 100/200")
        counts = (values == 100).sum(dim=1)
        if not bool(torch.all(counts == 5)):
            raise ValueError("player_team_ids must contain exactly five players from each team")
        return values

    def encode_components(self, batch_value: StructuredBatch | Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        batch = StructuredBatch.coerce(batch_value)
        node = torch.as_tensor(batch.node_numeric, dtype=torch.float32)
        if node.ndim != 3 or tuple(node.shape[1:]) != (PARTICIPANTS, self.node_numeric_dim):
            raise ValueError(f"node_numeric must have shape [batch,10,{self.node_numeric_dim}]")
        if node.numel() and not bool(torch.isfinite(node).all()):
            raise ValueError("node_numeric contains nonfinite values; preprocessing belongs to the root pipeline")
        device = next(self.parameters(), torch.empty(0)).device
        node = node.to(device)
        size = int(node.shape[0])
        team = self._numeric(batch.team_numeric, (size, TEAM_COUNT, self.team_numeric_dim), "team_numeric", device)
        global_values = self._numeric(batch.global_numeric, (size, self.global_numeric_dim), "global_numeric", device)
        if self.include_side_flag:
            if batch.side_flag is None:
                raise ValueError("side_flag is required for the common V head context")
            side = torch.as_tensor(batch.side_flag, dtype=torch.float32, device=device)
            if side.numel() != size:
                raise ValueError("side_flag must have shape [batch]")
            if side.numel() and not bool(torch.isfinite(side).all()):
                raise ValueError("side_flag contains nonfinite values")
            global_values = torch.cat((global_values, side.reshape(size, 1)), dim=-1)
        node_semantic = self._encode_side(
            batch.node_categories,
            batch.node_bags,
            batch=size,
            nodes=PARTICIPANTS,
            device=device,
            current_item_supported=batch.current_item_supported,
        )
        team_semantic = self._encode_side(batch.team_categories, batch.team_bags, batch=size, nodes=TEAM_COUNT, device=device)
        global_semantic = self._encode_global_side(batch.global_categories, batch.global_bags, batch=size, device=device)
        ids = self._team_indices(batch.player_team_ids, size, device)
        team_indicator = (ids == 200).to(dtype=node.dtype).unsqueeze(-1)
        players = torch.cat(
            (node, node_semantic, team_indicator),
            dim=-1,
        )
        context = torch.cat((team.reshape(size, -1), team_semantic.reshape(size, -1), global_values, global_semantic.reshape(size, -1)), dim=-1)
        return players, context, ids

    def encode_players(self, batch_value: StructuredBatch | Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        players, _, ids = self.encode_components(batch_value)
        return players, ids

    @property
    def player_feature_dim(self) -> int:
        # A caller that needs this before a batch is available can use the
        # declared group mappings; the exact count is computed from specs.
        category_dim = sum(self.bank.specs[_group_namespace(group, self.category_namespaces, self.bank)].embedding_dim for group in self.category_layout.get("node", ()))
        bag_dim = sum(self.bank.specs[_group_namespace(group, self.bag_namespaces, self.bank)].embedding_dim + 1 for group in self.bag_layout.get("node", ()))
        return self.node_numeric_dim + category_dim + bag_dim + 1

    @property
    def context_feature_dim(self) -> int:
        team_category_dim = sum(self.bank.specs[_group_namespace(group, self.category_namespaces, self.bank)].embedding_dim for group in self.category_layout.get("team", ()))
        global_category_dim = sum(self.bank.specs[_group_namespace(group, self.category_namespaces, self.bank)].embedding_dim for group in self.category_layout.get("global", ()))
        team_bag_dim = sum(self.bank.specs[_group_namespace(group, self.bag_namespaces, self.bank)].embedding_dim + 1 for group in self.bag_layout.get("team", ()))
        global_bag_dim = sum(self.bank.specs[_group_namespace(group, self.bag_namespaces, self.bank)].embedding_dim + 1 for group in self.bag_layout.get("global", ()))
        return (
            TEAM_COUNT * (self.team_numeric_dim + team_category_dim + team_bag_dim)
            + self.global_numeric_dim
            + (1 if self.include_side_flag else 0)
            + global_category_dim
            + global_bag_dim
        )


class _GeluMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], output_dim: int = 1, *, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(value) for value in hidden), int(output_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.extend((nn.GELU(), nn.Dropout(float(dropout))))
        self.net = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class PlainMLP(nn.Module):
    """Plain MLP over root-provided batch-dense semantic features."""

    family = "plain_mlp"

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        d_in: int | None = None,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        card: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if input_dim is None:
            input_dim = d_in
        selected = dict(card or {})
        hidden = tuple(selected.get("hidden", hidden))
        if input_dim is None:
            self.head: nn.Module = nn.Sequential(
                nn.LazyLinear(int(hidden[0])),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden[0]), int(hidden[1])),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden[1]), 1),
            )
        else:
            self.head = _GeluMLP(int(input_dim), hidden, 1, dropout=dropout)
        self.hidden = tuple(int(value) for value in hidden)
        self.dropout = float(dropout)
        self.weight_decay = float(selected.get("weight_decay", 0.0))

    @staticmethod
    def _dense(value: Tensor | StructuredBatch | Mapping[str, Any]) -> Tensor:
        if isinstance(value, Tensor):
            dense = value
        else:
            batch = StructuredBatch.coerce(value)
            if batch.dense_features is None:
                raise ValueError("PlainMLP requires root-provided dense_features; it does not densify metadata")
            dense = torch.as_tensor(batch.dense_features, dtype=torch.float32)
        if dense.ndim != 2:
            raise ValueError("dense_features must have shape [batch,features]")
        if dense.numel() and not bool(torch.isfinite(dense).all()):
            raise ValueError("dense_features contains nonfinite values")
        return dense

    def forward(self, value: Tensor | StructuredBatch | Mapping[str, Any]) -> Tensor:
        return self.head(self._dense(value))

    def predict_proba(self, value: Tensor | StructuredBatch | Mapping[str, Any]) -> Tensor:
        return torch.sigmoid(self.forward(value))


class SharedSlotMLP(nn.Module):
    """Shared player encoder followed by an order-sensitive ten-slot head."""

    family = "shared_slot_mlp"

    def __init__(
        self,
        *,
        node_numeric_dim: int = DEFAULT_NODE_NUMERIC_DIM,
        team_numeric_dim: int = DEFAULT_TEAM_NUMERIC_DIM,
        global_numeric_dim: int = DEFAULT_GLOBAL_NUMERIC_DIM,
        hidden_width: int = 64,
        head: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        vocab_sizes: Mapping[str, Any] | None = None,
        namespace_specs: Mapping[str, Any] | None = None,
        category_namespaces: Mapping[str, str] | None = None,
        bag_namespaces: Mapping[str, str] | None = None,
        category_layout: Mapping[str, Sequence[str]] | None = None,
        bag_layout: Mapping[str, Sequence[str]] | None = None,
        include_side_flag: bool = True,
        current_item_group: str | None = "current_item",
        card: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        selected = dict(card or {})
        hidden_width = int(selected.get("hidden_width", hidden_width))
        self.encoder = StructuredFeatureEncoder(
            node_numeric_dim=node_numeric_dim,
            team_numeric_dim=team_numeric_dim,
            global_numeric_dim=global_numeric_dim,
            vocab_sizes=vocab_sizes,
            namespace_specs=namespace_specs,
            category_namespaces=category_namespaces,
            bag_namespaces=bag_namespaces,
            category_layout=category_layout,
            bag_layout=bag_layout,
            include_side_flag=include_side_flag,
            current_item_group=current_item_group,
        )
        self.hidden_width = hidden_width
        self.player = _GeluMLP(self.encoder.player_feature_dim, (hidden_width,), hidden_width, dropout=dropout)
        self.head = _GeluMLP(hidden_width * PARTICIPANTS + self.encoder.context_feature_dim, tuple(head), 1, dropout=dropout)
        self.dropout = float(dropout)
        self.weight_decay = float(selected.get("weight_decay", 0.0))

    def encode_players(self, value: StructuredBatch | Mapping[str, Any] | Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(value, Tensor):
            if value.ndim != 3 or value.shape[1] != PARTICIPANTS:
                raise ValueError("player tensor must have shape [batch,10,features]")
            players = value
            context = torch.zeros((value.shape[0], self.encoder.context_feature_dim), dtype=value.dtype, device=value.device)
        else:
            players, context, _ = self.encoder.encode_components(value)
            context = context.to(players.device)
        return self.player(players), context

    def forward(self, value: StructuredBatch | Mapping[str, Any] | Tensor) -> Tensor:
        encoded, context = self.encode_players(value)
        features = torch.cat((encoded.reshape(encoded.shape[0], PARTICIPANTS * self.hidden_width), context), dim=-1)
        return self.head(features)

    def predict_proba(self, value: StructuredBatch | Mapping[str, Any] | Tensor) -> Tensor:
        return torch.sigmoid(self.forward(value))


class TeamSetMLP(SharedSlotMLP):
    """Shared player encoder with mean/max pooling within teams 100 and 200."""

    family = "team_set_mlp"

    def __init__(self, *args: Any, head: Sequence[int] = (256, 128), **kwargs: Any) -> None:
        super().__init__(*args, head=head, **kwargs)
        self.head = _GeluMLP(self.hidden_width * 4 + self.encoder.context_feature_dim, tuple(head), 1, dropout=self.dropout)

    @staticmethod
    def pool_players(encoded: Tensor, team_ids: Tensor) -> Tensor:
        if encoded.ndim != 3 or tuple(encoded.shape[1:2]) != (PARTICIPANTS,):
            raise ValueError("encoded players must have shape [batch,10,hidden]")
        ids = torch.as_tensor(team_ids, dtype=torch.long, device=encoded.device)
        if tuple(ids.shape) != tuple(encoded.shape[:2]):
            raise ValueError("team IDs must have shape [batch,10]")
        pieces: list[Tensor] = []
        for team in TEAM_IDS:
            mask = ids == team
            if not bool(mask.any(dim=1).all()):
                raise ValueError("each batch row must contain both team IDs")
            expanded = mask.unsqueeze(-1)
            count = expanded.sum(dim=1).clamp_min(1)
            mean = (encoded * expanded).sum(dim=1) / count
            maximum = encoded.masked_fill(~expanded, -torch.inf).amax(dim=1)
            pieces.extend((mean, maximum))
        return torch.cat(pieces, dim=-1)

    def encode_players(self, value: StructuredBatch | Mapping[str, Any] | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(value, Tensor):
            if value.ndim != 3 or value.shape[1] != PARTICIPANTS:
                raise ValueError("player tensor must have shape [batch,10,features]")
            players = value
            ids = torch.tensor([100] * 5 + [200] * 5, device=value.device, dtype=torch.long).reshape(1, 10).expand(value.shape[0], -1)
            context = torch.zeros((value.shape[0], self.encoder.context_feature_dim), dtype=value.dtype, device=value.device)
        else:
            players, context, ids = self.encoder.encode_components(value)
        return self.player(players), context, ids

    def forward(self, value: StructuredBatch | Mapping[str, Any] | Tensor) -> Tensor:
        encoded, context, ids = self.encode_players(value)
        return self.head(torch.cat((self.pool_players(encoded, ids), context), dim=-1))


def _matrix_from(value: Any) -> Any:
    if isinstance(value, StructuredBatch):
        value = value.table_csr
    elif isinstance(value, Mapping):
        value = value.get("table_csr", value.get("table", value.get("csr")))
    if value is None:
        raise ValueError("table adapter requires root-provided table_csr")
    if getattr(value, "ndim", 2) != 2:
        raise ValueError("table_csr must be a two-dimensional matrix")
    return value


class _TableAdapter:
    family = "table"

    def __init__(self) -> None:
        self.estimator: Any | None = None

    def fit(self, table_csr: Any, target: Any, *, sample_weight: Any | None = None) -> "_TableAdapter":
        matrix = _matrix_from(table_csr)
        labels = np.asarray(target)
        if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
            raise ValueError("target must be one-dimensional and match table rows")
        kwargs = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}
        self.estimator.fit(matrix, labels, **kwargs)
        return self

    def predict_logits(self, table_csr: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("table adapter is not fitted")
        matrix = _matrix_from(table_csr)
        if hasattr(self.estimator, "decision_function"):
            result = np.asarray(self.estimator.decision_function(matrix), dtype=np.float64)
            if result.ndim == 2 and result.shape[1] == 2:
                result = result[:, 1] - result[:, 0]
            return result.reshape(-1)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)[:, 1]
        clipped = np.clip(probabilities, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
        return np.log(clipped) - np.log1p(-clipped)

    def predict_proba(self, table_csr: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("table adapter is not fitted")
        return np.asarray(self.estimator.predict_proba(_matrix_from(table_csr)))[:, 1]

    def forward(self, table_csr: Any) -> np.ndarray:
        return self.predict_logits(table_csr)


class LogisticNonconvergenceError(RuntimeError):
    """Raised when the L-BFGS-B optimizer reaches max iterations without converging."""

    def __init__(self, convergence: dict[str, Any]) -> None:
        self.convergence = convergence
        super().__init__(f"logistic candidate failed to converge: {convergence}")


class LogisticAdapter(_TableAdapter):
    """L2 logistic reference over the parent's sparse table matrix.

    The objective is the execution-card objective, rather than a delegated
    sklearn default: weighted mean BCE plus ``||beta||^2/(2*C*n_fit)`` with an
    unpenalized intercept and scipy L-BFGS-B tolerances fixed by the card.
    """

    family = "logistic"

    def __init__(self, *, C: float = 0.001, card: Mapping[str, Any] | None = None, max_iter: int = 4000, random_state: int | None = None, ftol: float = 1e-12, gtol: float = 1e-7) -> None:
        super().__init__()
        selected = dict(card or {})
        self.C = float(selected.get("C", C))
        self.max_iter = int(selected.get("max_iter", max_iter))
        self.random_state = random_state
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.convergence_: dict[str, Any] | None = None

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        output = np.empty_like(value, dtype=np.float64)
        positive = value >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exp_value = np.exp(value[~positive])
        output[~positive] = exp_value / (1.0 + exp_value)
        return output

    def fit(self, table_csr: Any, target: Any, *, sample_weight: Any | None = None) -> "LogisticAdapter":
        matrix = _matrix_from(table_csr)
        rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
        labels = np.asarray(target, dtype=np.float64).reshape(-1)
        if labels.shape[0] != rows or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("target must be finite binary values matching table rows")
        weights = np.ones(rows, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != rows or not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and nonzero")
        try:
            from scipy import sparse
            x = sparse.csr_matrix(matrix, dtype=np.float64)
            sparse_input = True
        except Exception:
            x = np.asarray(matrix, dtype=np.float64)
            sparse_input = False
        total_weight = float(weights.sum())
        n_fit = max(rows, 1)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            beta = parameters[:columns]
            intercept = float(parameters[columns])
            logits = np.asarray(x @ beta, dtype=np.float64).reshape(-1) + intercept
            probabilities = self._sigmoid(logits)
            loss = float(np.dot(weights, np.logaddexp(0.0, logits) - labels * logits) / total_weight)
            loss += float(np.dot(beta, beta) / (2.0 * self.C * n_fit))
            residual = weights * (probabilities - labels) / total_weight
            if sparse_input:
                gradient_beta = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1)
            else:
                gradient_beta = np.asarray(x.T.dot(residual), dtype=np.float64).reshape(-1)
            gradient_beta += beta / (self.C * n_fit)
            gradient = np.concatenate((gradient_beta, np.asarray([residual.sum()], dtype=np.float64)))
            return loss, gradient

        from scipy.optimize import minimize

        initial = np.zeros(columns + 1, dtype=np.float64)
        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.ftol, "gtol": self.gtol},
        )
        self.convergence_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "fun": float(result.fun),
        }
        if not bool(result.success) or not np.isfinite(result.x).all():
            raise LogisticNonconvergenceError(self.convergence_)
        self.coef_ = np.asarray(result.x[:columns], dtype=np.float64)
        self.intercept_ = float(result.x[columns])
        self.n_features_in_ = columns
        return self

    def predict_logits(self, table_csr: Any) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("table adapter is not fitted")
        matrix = _matrix_from(table_csr)
        return np.asarray(matrix @ self.coef_, dtype=np.float64).reshape(-1) + self.intercept_

    decision_function = predict_logits

    def predict_proba(self, table_csr: Any) -> np.ndarray:
        probabilities = self._sigmoid(self.predict_logits(table_csr))
        return probabilities


class LightGBMAdapter(_TableAdapter):
    """CPU LightGBM reference over the parent's sparse table matrix."""

    family = "lightgbm"

    def __init__(self, *, card: Mapping[str, Any] | None = None, random_state: int = 0, n_estimators: int = 2000, learning_rate: float = 0.05, num_threads: int = 1) -> None:
        super().__init__()
        selected = dict(card or {})
        self.num_leaves = int(selected.get("num_leaves", 31))
        self.min_data_in_leaf = int(selected.get("min_data_in_leaf", 200))
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_threads = int(num_threads)
        from lightgbm import LGBMClassifier

        self.estimator = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            num_threads=self.num_threads,
            feature_fraction=0.8,
            max_bin=255,
            device_type="cpu",
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )


def _select_card(family: str, card: Mapping[str, Any] | str | int | None) -> dict[str, Any]:
    cards = family_cards(family)
    if card is None:
        return dict(cards[0])
    if isinstance(card, Mapping):
        return dict(card)
    if isinstance(card, int) and not isinstance(card, bool):
        return dict(cards[int(card)])
    for candidate in cards:
        if candidate["id"] == str(card):
            return dict(candidate)
    raise ValueError(f"unknown {family} card: {card!r}")


def build_model(family: str, *, card: Mapping[str, Any] | str | int | None = None, **kwargs: Any) -> Any:
    """Construct one declared card without fitting it or reading raw data."""

    key = str(family).lower()
    selected = _select_card(key, card)
    if key == "logistic":
        return LogisticAdapter(card=selected, **kwargs)
    if key == "lightgbm":
        return LightGBMAdapter(card=selected, **kwargs)
    if key == "plain_mlp":
        return PlainMLP(card=selected, **kwargs)
    if key == "shared_slot_mlp":
        return SharedSlotMLP(card=selected, **kwargs)
    if key == "team_set_mlp":
        return TeamSetMLP(card=selected, **kwargs)
    raise ValueError(f"unknown V model family: {family!r}")


make_model = build_model
LogisticModel = LogisticAdapter
LightGBMModel = LightGBMAdapter
PlainMLPAdapter = PlainMLP
SharedSlotAdapter = SharedSlotMLP
TeamSetAdapter = TeamSetMLP


__all__ = [
    "BagInput",
    "DEFAULT_GLOBAL_NUMERIC_DIM",
    "DEFAULT_NODE_NUMERIC_DIM",
    "DEFAULT_TEAM_NUMERIC_DIM",
    "EMBEDDING_DIM",
    "FAMILIES",
    "FAMILY_CARDS",
    "LightGBMAdapter",
    "LightGBMModel",
    "LogisticAdapter",
    "LogisticModel",
    "MODEL_VERSION",
    "ModelBatch",
    "NamespaceEmbeddingBank",
    "NamespaceSpec",
    "PAD_INDEX",
    "PlainMLP",
    "PlainMLPAdapter",
    "SharedSlotAdapter",
    "SharedSlotMLP",
    "StructuredBatch",
    "StructuredFeatureEncoder",
    "StructuredInput",
    "TEAM_IDS",
    "TeamSetAdapter",
    "TeamSetMLP",
    "UNK_INDEX",
    "build_model",
    "family_cards",
    "make_model",
]
