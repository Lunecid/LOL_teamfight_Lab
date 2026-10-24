"""Fit-scoped item vocabulary and count-aware node embedding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn


PAD_INDEX = 0
UNK_INDEX = 1
VOCABULARY_VERSION = "item_vocabulary_fit_v1_20261002"


def _item_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("item IDs must be positive integer source IDs")
    value = int(value)
    if value <= 0:
        raise ValueError("item IDs must be positive integer source IDs")
    return value


def _scope_tuple(scope_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(scope_keys, (str, bytes)):
        raise ValueError("scope_keys must be a sequence of exact scope keys")
    values = tuple(scope_keys)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("scope_keys must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("scope_keys must be unique")
    return tuple(sorted(values))


def _validate_fit_provenance(
    scope_keys: tuple[str, ...],
    provenance: Optional[str | Mapping[str, str]],
    excluded_scope_keys: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    excluded = set(excluded_scope_keys)
    if excluded & set(scope_keys):
        raise ValueError("excluded scope keys cannot be admitted to vocabulary FIT")
    if provenance is None:
        raise ValueError("explicit FIT provenance is required")
    if isinstance(provenance, str):
        labels = {scope: provenance for scope in scope_keys}
    elif isinstance(provenance, Mapping):
        labels = {}
        for scope in scope_keys:
            label = provenance.get(scope)
            if not isinstance(label, str):
                raise ValueError(f"missing FIT provenance for scope {scope!r}")
            labels[scope] = label
    else:
        raise ValueError("provenance must be a label or scope-to-label mapping")
    allowed_labels = {"FIT", "REFIT_TRAIN"}
    for scope, label in labels.items():
        upper = label.strip().upper()
        if upper not in allowed_labels:
            raise ValueError("TEST/excluded/unsupported provenance: vocabulary FIT must be FIT or REFIT_TRAIN")
        if scope in excluded:
            raise ValueError("TEST/excluded provenance cannot be used for vocabulary FIT")
        if scope.strip().upper() in {"TEST", "EXCLUDED"} or scope.strip().upper().endswith("_TEST"):
            raise ValueError("TEST scope cannot be used for vocabulary FIT")
    return tuple((scope, labels[scope]) for scope in scope_keys)


@dataclass(frozen=True)
class ItemVocabulary:
    """Deterministic raw-item-ID to embedding-index mapping."""

    item_ids: tuple[int, ...]
    scope_keys: tuple[str, ...]
    fit_provenance: tuple[tuple[str, str], ...]
    version: str = VOCABULARY_VERSION

    @classmethod
    def fit(
        cls,
        item_ids: Iterable[int] | Mapping[str, Iterable[int]],
        *,
        scope_keys: Sequence[str] | None = None,
        provenance: Optional[str | Mapping[str, str]] = None,
        excluded_scope_keys: Sequence[str] = (),
    ) -> "ItemVocabulary":
        if isinstance(item_ids, Mapping):
            by_scope = item_ids
            if scope_keys is None:
                scope_keys = tuple(by_scope.keys())
            raw_scopes = _scope_tuple(scope_keys)
            if set(raw_scopes) != set(by_scope):
                raise ValueError("mapping scopes must equal exact scope_keys")
            tokens: list[Any] = []
            for scope in raw_scopes:
                tokens.extend(list(by_scope[scope]))
        else:
            if scope_keys is None:
                raise ValueError("scope_keys are required for FIT")
            raw_scopes = _scope_tuple(scope_keys)
            tokens = list(item_ids)
        fit_provenance = _validate_fit_provenance(
            raw_scopes, provenance, excluded_scope_keys
        )
        unique = sorted({_item_id(value) for value in tokens})
        return cls(tuple(unique), raw_scopes, fit_provenance)

    @property
    def allowed_scope_keys(self) -> tuple[str, ...]:
        return self.scope_keys

    @property
    def raw_item_ids(self) -> tuple[int, ...]:
        return self.item_ids

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def num_embeddings(self) -> int:
        return len(self.item_ids) + 2

    @property
    def item_id_to_index(self) -> dict[int, int]:
        return {item_id: index + 2 for index, item_id in enumerate(self.item_ids)}

    @property
    def index_to_item_id(self) -> dict[int, Optional[int]]:
        return {PAD_INDEX: None, UNK_INDEX: None, **{index + 2: item_id for index, item_id in enumerate(self.item_ids)}}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pad_index": PAD_INDEX,
            "unk_index": UNK_INDEX,
            "item_ids": list(self.item_ids),
            "scope_keys": list(self.scope_keys),
            "fit_provenance": {scope: label for scope, label in self.fit_provenance},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ItemVocabulary":
        if payload.get("version") != VOCABULARY_VERSION:
            raise ValueError("unsupported item vocabulary version")
        if payload.get("pad_index") != PAD_INDEX or payload.get("unk_index") != UNK_INDEX:
            raise ValueError("item vocabulary must reserve PAD=0 and UNK=1")
        item_ids = tuple(_item_id(value) for value in payload.get("item_ids", ()))
        if tuple(sorted(set(item_ids))) != item_ids:
            raise ValueError("item vocabulary item_ids must be sorted and unique")
        scopes = _scope_tuple(payload.get("scope_keys", ()))
        raw_provenance = payload.get("fit_provenance")
        if not isinstance(raw_provenance, Mapping):
            raise ValueError("item vocabulary is missing fit provenance")
        provenance = _validate_fit_provenance(scopes, raw_provenance, ())
        return cls(item_ids, scopes, provenance, version=VOCABULARY_VERSION)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ItemVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, raw_item_ids: Iterable[int]) -> np.ndarray:
        mapping = self.item_id_to_index
        return np.asarray([mapping.get(_item_id(value), UNK_INDEX) for value in raw_item_ids], dtype=np.int64)

    def transform_with_audit(self, raw_item_ids: Iterable[int]) -> dict[str, np.ndarray]:
        raw = np.asarray([_item_id(value) for value in raw_item_ids], dtype=np.int64)
        mapping = self.item_id_to_index
        indices = np.asarray([mapping.get(int(value), UNK_INDEX) for value in raw], dtype=np.int64)
        return {"raw_item_ids": raw, "indices": indices}


def fit_item_vocabulary(
    item_ids: Iterable[int] | Mapping[str, Iterable[int]],
    *,
    scope_keys: Sequence[str] | None = None,
    provenance: Optional[str | Mapping[str, str]] = None,
    excluded_scope_keys: Sequence[str] = (),
) -> ItemVocabulary:
    """Fit/refit a deterministic vocabulary using explicitly allowed scopes."""
    return ItemVocabulary.fit(
        item_ids,
        scope_keys=scope_keys,
        provenance=provenance,
        excluded_scope_keys=excluded_scope_keys,
    )


class ItemMultisetEmbedding(nn.Module):
    """Embed ten ragged per-player item bags and attach them to node features.

    The returned last dimension is ``embedding_dim + 2``: the count-weighted
    item sum, total recorded item count, and a support flag (1 known, 0
    unknown).  An unknown ledger uses a separate learned vector.  Its count
    channel is an explicit zero placeholder with support=0, so it is never
    interpreted as an observed empty bag; a supported empty ledger returns a
    zero sum and count with support=1.
    """

    def __init__(
        self,
        vocabulary: ItemVocabulary | Mapping[int, int] | int | None = None,
        embedding_dim: int | None = None,
        *,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim is None or isinstance(embedding_dim, (bool, np.bool_)) or int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be an explicit positive integer")
        if vocabulary is None:
            vocabulary = vocab_size
        if vocabulary is None:
            raise ValueError("an ItemVocabulary or explicit vocab_size is required")
        self.vocabulary = vocabulary if isinstance(vocabulary, ItemVocabulary) else None
        if isinstance(vocabulary, ItemVocabulary):
            num_embeddings = vocabulary.num_embeddings
        elif isinstance(vocabulary, Mapping):
            values = [int(value) for value in vocabulary.values()]
            if sorted(values) != list(range(2, len(values) + 2)):
                raise ValueError("mapping vocabulary indices must be contiguous and reserve PAD=0, UNK=1")
            num_embeddings = len(values) + 2
        elif isinstance(vocabulary, (int, np.integer)) and not isinstance(vocabulary, (bool, np.bool_)):
            if int(vocabulary) < 2:
                raise ValueError("vocab_size must include PAD and UNK")
            num_embeddings = int(vocabulary)
        else:
            raise TypeError("vocabulary must be ItemVocabulary, mapping, or explicit vocab_size")
        self.embedding_dim = int(embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings, self.embedding_dim, padding_idx=PAD_INDEX)
        self.unknown_ledger_embedding = nn.Parameter(torch.empty(self.embedding_dim))
        self.reset_parameters()

    @property
    def unknown_embedding(self) -> nn.Parameter:
        return self.unknown_ledger_embedding

    @property
    def output_dim(self) -> int:
        return self.embedding_dim + 2

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embeddings.weight[PAD_INDEX].zero_()
        nn.init.normal_(self.unknown_ledger_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _offset_pairs(offsets: Tensor, batch: int, nodes: int = 10) -> list[tuple[int, int]]:
        offsets = torch.as_tensor(offsets, dtype=torch.long).detach().cpu()
        if offsets.ndim == 1:
            if batch != 1 or offsets.numel() != nodes + 1:
                raise ValueError("one-dimensional offsets require batch=1 and length 11")
            offsets = offsets.reshape(1, nodes + 1)
        if offsets.ndim == 2:
            if offsets.shape == (batch, nodes + 1):
                return [
                    (int(offsets[b, n]), int(offsets[b, n + 1]))
                    for b in range(batch)
                    for n in range(nodes)
                ]
            if offsets.shape == (batch * nodes, 2):
                return [(int(row[0]), int(row[1])) for row in offsets]
        if offsets.ndim == 3 and tuple(offsets.shape) == (batch, nodes, 2):
            return [(int(offsets[b, n, 0]), int(offsets[b, n, 1])) for b in range(batch) for n in range(nodes)]
        raise ValueError("offsets must have shape [batch,11], [batch,10,2], or [batch*10,2]")

    @staticmethod
    def _status_tensor(
        value: Optional[Tensor | Sequence[bool]],
        batch: int,
        nodes: int,
        *,
        default: bool,
        name: str,
    ) -> Tensor:
        if value is None:
            return torch.full((batch, nodes), default, dtype=torch.bool)
        result = torch.as_tensor(value, dtype=torch.bool)
        if result.numel() != batch * nodes:
            raise ValueError(f"{name} must have shape [batch,10]")
        return result.reshape(batch, nodes)

    def forward(
        self,
        item_indices: Tensor | Sequence[int],
        counts: Tensor | Sequence[int],
        offsets: Tensor | Sequence[int],
        recorded_ledger_supported: Optional[Tensor | Sequence[bool]] = None,
        *,
        unknown: Optional[Tensor | Sequence[bool]] = None,
        ledger_supported: Optional[Tensor | Sequence[bool]] = None,
    ) -> Tensor:
        if recorded_ledger_supported is not None and ledger_supported is not None:
            raise ValueError("provide only one support flag")
        if ledger_supported is not None:
            recorded_ledger_supported = ledger_supported
        indices = torch.as_tensor(item_indices, dtype=torch.long, device=self.item_embeddings.weight.device).reshape(-1)
        item_counts = torch.as_tensor(counts, device=indices.device).reshape(-1)
        if indices.numel() != item_counts.numel():
            raise ValueError("item_indices and counts must have equal flattened length")
        if item_counts.is_floating_point() and not bool(torch.isfinite(item_counts).all()):
            raise ValueError("item counts must be finite")
        if bool((item_counts < 0).any()):
            raise ValueError("item counts must be nonnegative")
        if bool((item_counts != torch.floor(item_counts)).any()):
            raise ValueError("item counts must be integer-valued")
        if bool(((indices < 0) | (indices >= self.item_embeddings.num_embeddings)).any()):
            raise ValueError("item embedding indices are outside the fitted vocabulary")
        # Infer batch from support/status when present; otherwise offsets carry
        # the batch dimension and are [batch,11].
        status_value = recorded_ledger_supported if recorded_ledger_supported is not None else unknown
        if status_value is None:
            raw_offsets = torch.as_tensor(offsets)
            if raw_offsets.ndim == 1:
                batch = 1
            elif raw_offsets.ndim == 2 and raw_offsets.shape[1] == 11:
                batch = int(raw_offsets.shape[0])
            elif raw_offsets.ndim == 2 and raw_offsets.shape[1] == 2:
                if raw_offsets.shape[0] % 10:
                    raise ValueError("offsets [batch*10,2] must contain ten nodes per batch")
                batch = int(raw_offsets.shape[0] // 10)
            elif raw_offsets.ndim == 3:
                batch = int(raw_offsets.shape[0])
            else:
                raise ValueError("batch cannot be inferred from offsets")
        else:
            status_shape = torch.as_tensor(status_value).shape
            if len(status_shape) == 1:
                if status_shape[0] % 10:
                    raise ValueError("status must have shape [batch,10]")
                batch = int(status_shape[0] // 10)
            elif len(status_shape) == 2 and status_shape[1] == 10:
                batch = int(status_shape[0])
            else:
                raise ValueError("status must have shape [batch,10]")
        pairs = self._offset_pairs(torch.as_tensor(offsets), batch)
        support = self._status_tensor(
            recorded_ledger_supported,
            batch,
            10,
            default=True,
            name="recorded_ledger_supported",
        )
        if unknown is not None:
            unknown_tensor = self._status_tensor(unknown, batch, 10, default=False, name="unknown")
            expected_support = torch.logical_not(unknown_tensor)
            if recorded_ledger_supported is not None and not bool(torch.equal(support.cpu(), expected_support.cpu())):
                raise ValueError("unknown and recorded_ledger_supported disagree")
            support = expected_support
        outputs: list[Tensor] = []
        item_counts_float = item_counts.to(dtype=self.item_embeddings.weight.dtype)
        for flat_node, (start, end) in enumerate(pairs):
            if start < 0 or end < start or end > indices.numel():
                raise ValueError("ragged offsets are outside flattened item arrays")
            b, node = divmod(flat_node, 10)
            if not bool(support[b, node]):
                pooled = self.unknown_ledger_embedding
                total = torch.zeros((), dtype=pooled.dtype, device=pooled.device)
            else:
                local_indices = indices[start:end]
                local_counts = item_counts_float[start:end]
                if local_indices.numel():
                    pooled = (self.item_embeddings(local_indices) * local_counts.unsqueeze(-1)).sum(dim=0)
                    total = local_counts.sum()
                else:
                    pooled = torch.zeros(self.embedding_dim, dtype=self.item_embeddings.weight.dtype, device=self.item_embeddings.weight.device)
                    total = torch.zeros((), dtype=pooled.dtype, device=pooled.device)
            status = torch.as_tensor(float(bool(support[b, node])), dtype=pooled.dtype, device=pooled.device)
            outputs.append(torch.cat((pooled, total.reshape(1), status.reshape(1)), dim=0))
        return torch.stack(outputs, dim=0).reshape(batch, 10, self.output_dim)

    def encode_components(
        self,
        item_indices: Tensor | Sequence[int],
        counts: Tensor | Sequence[int],
        offsets: Tensor | Sequence[int],
        recorded_ledger_supported: Optional[Tensor | Sequence[bool]] = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        encoded = self(item_indices, counts, offsets, recorded_ledger_supported, **kwargs)
        return {
            "pooled_item_embedding": encoded[..., : self.embedding_dim],
            "total_count": encoded[..., self.embedding_dim : self.embedding_dim + 1],
            "recorded_ledger_supported": encoded[..., self.embedding_dim + 1 :],
            "unknown": 1.0 - encoded[..., self.embedding_dim + 1 :],
            "count_known": encoded[..., self.embedding_dim + 1 :] > 0.5,
            "total_count_known": encoded[..., self.embedding_dim + 1 :] > 0.5,
            "encoded": encoded,
        }

    def attach_to_nodes(
        self,
        node_features: Tensor,
        item_indices: Tensor | Sequence[int],
        counts: Tensor | Sequence[int],
        offsets: Tensor | Sequence[int],
        recorded_ledger_supported: Optional[Tensor | Sequence[bool]] = None,
        **kwargs: Any,
    ) -> Tensor:
        if node_features.ndim != 3 or node_features.shape[1] != 10:
            raise ValueError("node_features must have shape [batch,10,node_feature_dim]")
        item_features = self(
            item_indices,
            counts,
            offsets,
            recorded_ledger_supported,
            **kwargs,
        )
        if item_features.shape[:2] != node_features.shape[:2]:
            raise ValueError("node and item batches must have the same [batch,10] shape")
        return torch.cat((node_features, item_features.to(node_features.device)), dim=-1)


def attach_item_embedding_to_nodes(
    node_features: Tensor,
    item_encoder: ItemMultisetEmbedding,
    item_indices: Tensor | Sequence[int],
    counts: Tensor | Sequence[int],
    offsets: Tensor | Sequence[int],
    recorded_ledger_supported: Optional[Tensor | Sequence[bool]] = None,
    **kwargs: Any,
) -> Tensor:
    """Attach each participant's encoded bag to its matching node row."""
    return item_encoder.attach_to_nodes(
        node_features,
        item_indices,
        counts,
        offsets,
        recorded_ledger_supported,
        **kwargs,
    )


__all__ = [
    "ItemMultisetEmbedding",
    "ItemVocabulary",
    "PAD_INDEX",
    "UNK_INDEX",
    "VOCABULARY_VERSION",
    "attach_item_embedding_to_nodes",
    "fit_item_vocabulary",
]
