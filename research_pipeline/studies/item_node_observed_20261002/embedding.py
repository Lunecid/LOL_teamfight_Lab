"""Count-aware node and history encodings for observed item ledgers."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from studies.item_node_encoding_20261002.embedding import (
    ItemMultisetEmbedding,
    ItemVocabulary,
    PAD_INDEX,
    UNK_INDEX,
    VOCABULARY_VERSION,
    attach_item_embedding_to_nodes as _attach_legacy,
    fit_item_vocabulary as _fit_legacy_vocabulary,
)
from .ledger import FULL_INVENTORY_COMPLETE, ITEM_EVENT_TYPES, ITEM_FIELD_ROLES


OBSERVED_EMBEDDING_VERSION = "recipient_confirmed_item_observation_embedding_v3_20261002"


def fit_item_vocabulary(
    item_ids: Iterable[int] | Mapping[str, Iterable[int]],
    *,
    scope_keys: Sequence[str] | None = None,
    provenance: Optional[str | Mapping[str, str]] = None,
    excluded_scope_keys: Sequence[str] = (),
) -> ItemVocabulary:
    """Reuse v2's FIT/REFIT_TRAIN-only vocabulary checks unchanged."""
    return _fit_legacy_vocabulary(
        item_ids,
        scope_keys=scope_keys,
        provenance=provenance,
        excluded_scope_keys=excluded_scope_keys,
    )


ObservedItemVocabulary = ItemVocabulary


def _context_vectors(context: Mapping[str, Any]) -> tuple[tuple[Tensor, ...], ...]:
    values = context.get("item_id_vectors", context.get("item_ids"))
    if values is None:
        return tuple(tuple(torch.empty(0, dtype=torch.long) for _ in ITEM_FIELD_ROLES) for _ in ITEM_EVENT_TYPES)
    return tuple(
        tuple(torch.as_tensor(ids, dtype=torch.long).reshape(-1) for ids in event_values)
        for event_values in values
    )


class ObservedItemMultisetEmbedding(ItemMultisetEmbedding):
    """Embed supported observed bags and retain uncertainty as separate data.

    The node tensor keeps the v2 ``sum, count, support`` channels.  The v3
    uncertainty channels are returned by :meth:`encode_snapshot` at graph
    level; unknown-owner transactions are never inserted into a node bag.
    """

    version = OBSERVED_EMBEDDING_VERSION

    @property
    def uncertainty_channels(self) -> tuple[str, ...]:
        return (
            "full_inventory_complete",
            "observed_ledger_supported",
            "unassigned_history_observed",
        )

    @property
    def context_output_dim(self) -> int:
        # Twelve typed event/field slots, event counts/presence, field
        # presence/validity/malformed counts, event malformed counts, and one
        # history-presence bit.
        return len(ITEM_EVENT_TYPES) * len(ITEM_FIELD_ROLES) * self.embedding_dim + 4 + 4 + 12 + 12 + 12 + 4 + 1

    @property
    def current_output_dim(self) -> int:
        """Dimension of the current observed item bag alone (v2 shape)."""
        return super().output_dim

    @property
    def combined_output_dim(self) -> int:
        """Dimension of a per-node current-bag/history/uncertainty feature."""
        return self.current_output_dim + 2 + self.context_output_dim

    def _transform_raw_ids(self, values: Iterable[int]) -> Tensor:
        raw = list(values)
        if self.vocabulary is not None:
            return torch.as_tensor(self.vocabulary.transform(raw), dtype=torch.long, device=self.item_embeddings.weight.device)
        return torch.as_tensor([int(value) if 0 <= int(value) < self.item_embeddings.num_embeddings else UNK_INDEX for value in raw], dtype=torch.long, device=self.item_embeddings.weight.device)

    def encode_history_context(self, context: Mapping[str, Any]) -> Tensor:
        """Encode typed item history as one graph-level context vector."""
        vectors = _context_vectors(context)
        pieces: list[Tensor] = []
        for event_values in vectors:
            for ids in event_values:
                if ids.numel():
                    indices = self._transform_raw_ids(ids.tolist())
                    pieces.append(self.item_embeddings(indices).sum(dim=0))
                else:
                    pieces.append(torch.zeros(self.embedding_dim, device=self.item_embeddings.weight.device, dtype=self.item_embeddings.weight.dtype))
        def _array(name: str, shape: tuple[int, ...]) -> Tensor:
            value = np.asarray(context.get(name, np.zeros(shape, dtype=np.float32)))
            if value.size != int(np.prod(shape)):
                raise ValueError(f"history context {name} must have shape {shape}")
            return torch.as_tensor(value.reshape(shape), device=self.item_embeddings.weight.device, dtype=self.item_embeddings.weight.dtype).reshape(-1)
        pieces.extend(
            (
                _array("event_counts", (4,)),
                _array("event_presence", (4,)),
                _array("field_presence_counts", (4, 3)),
                _array("field_validity_counts", (4, 3)),
                _array("field_malformed_counts", (4, 3)),
                _array("event_malformed_counts", (4,)),
                torch.as_tensor(
                    [float(bool(context.get("history_observed", context.get("source_event_count", 0))))],
                    device=self.item_embeddings.weight.device,
                    dtype=self.item_embeddings.weight.dtype,
                ),
            )
        )
        return torch.cat(pieces, dim=0)

    # Short aliases make graph adapters independent of the report wording.
    encode_context = encode_history_context
    encode_history = encode_history_context

    def encode_unassigned_context(self, context: Mapping[str, Any]) -> Tensor:
        return self.encode_history_context(context)

    def encode_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Encode current observed bags plus separate history/uncertainty data."""
        raw_ids = np.asarray(snapshot.get("item_ids", ()), dtype=np.int64)
        item_indices = self._transform_raw_ids(raw_ids.tolist())
        counts = torch.as_tensor(snapshot.get("counts", ()), dtype=torch.long, device=item_indices.device)
        offsets = torch.as_tensor(snapshot.get("offsets", np.zeros(11, dtype=np.int64)), dtype=torch.long, device=item_indices.device)
        if offsets.ndim == 1:
            offsets = offsets.reshape(1, -1)
        support = torch.as_tensor(snapshot.get("recorded_ledger_supported", snapshot.get("observed_ledger_supported")), dtype=torch.bool, device=item_indices.device)
        if support.numel() != 10:
            raise ValueError("snapshot support must contain ten participant flags")
        support = support.reshape(1, 10)
        current_node_features = self(item_indices, counts, offsets, support)
        assigned = snapshot.get("assigned_history_context", snapshot.get("assigned_history", {}))
        unassigned = snapshot.get("unassigned_history_context", snapshot.get("unassigned_context"))
        if unassigned is None or not isinstance(unassigned, Mapping):
            raise ValueError("snapshot must include unassigned_history_context")
        assigned_context = self.encode_history_context(assigned)
        unassigned_context = self.encode_unassigned_context(unassigned)
        by_player = snapshot.get("assigned_history_by_participant")
        if by_player is None or not isinstance(by_player, Sequence) or isinstance(by_player, (str, bytes, bytearray)):
            raise ValueError("snapshot must include assigned_history_by_participant")
        if len(by_player) != 10:
            raise ValueError("assigned_history_by_participant must contain ten contexts")
        node_history_features = torch.stack(
            [self.encode_history_context(context) for context in by_player], dim=0
        ).reshape(1, 10, self.context_output_dim)
        unassigned_observed = float(bool(snapshot.get("unassigned_history_observed", snapshot.get("has_unassigned_history", False))))
        uncertainty = torch.zeros((1, 10, 2), dtype=current_node_features.dtype, device=current_node_features.device)
        uncertainty[..., 1] = unassigned_observed
        node_features = torch.cat((current_node_features, uncertainty, node_history_features), dim=-1)
        return {
            "node_features": node_features,
            "current_node_features": current_node_features,
            "item_node_features": current_node_features,
            "node_history_features": node_history_features,
            "assigned_history_context": assigned_context,
            "assigned_context": assigned_context,
            "unassigned_history_context": unassigned_context,
            "unassigned_context": unassigned_context,
            "uncertainty_channels": uncertainty,
            "full_inventory_complete": 0.0,
            "full_inventory_verification": 0.0,
            "full_inventory_unknown": 1.0,
            "full_inventory_verified": False,
            "observed_ledger_supported": support,
            "unassigned_history_observed": bool(unassigned_observed),
            "version": self.version,
        }

    def attach_snapshot_to_nodes(self, node_features: Tensor, snapshot: Mapping[str, Any]) -> Tensor:
        encoded = self.encode_snapshot(snapshot)["node_features"]
        if node_features.ndim != 3 or node_features.shape[1] != 10:
            raise ValueError("node_features must have shape [batch,10,node_feature_dim]")
        if encoded.shape[:2] != node_features.shape[:2]:
            raise ValueError("node and item batches must have the same [batch,10] shape")
        return torch.cat((node_features, encoded.to(node_features.device)), dim=-1)

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
        item_features = self(item_indices, counts, offsets, recorded_ledger_supported, **kwargs)
        if item_features.shape[:2] != node_features.shape[:2]:
            raise ValueError("node and item batches must have the same [batch,10] shape")
        return torch.cat((node_features, item_features.to(node_features.device)), dim=-1)


ObservedItemEmbedding = ObservedItemMultisetEmbedding
ItemObservationEmbedding = ObservedItemMultisetEmbedding
RecipientConfirmedItemEmbedding = ObservedItemMultisetEmbedding


def attach_observed_item_embedding_to_nodes(
    node_features: Tensor,
    item_encoder: ObservedItemMultisetEmbedding,
    item_indices: Tensor | Sequence[int],
    counts: Tensor | Sequence[int],
    offsets: Tensor | Sequence[int],
    recorded_ledger_supported: Optional[Tensor | Sequence[bool]] = None,
    **kwargs: Any,
) -> Tensor:
    return item_encoder.attach_to_nodes(
        node_features,
        item_indices,
        counts,
        offsets,
        recorded_ledger_supported,
        **kwargs,
    )


def attach_item_embedding_to_nodes(*args: Any, **kwargs: Any) -> Tensor:
    """Alias matching the legacy adapter for the new encoder."""
    return attach_observed_item_embedding_to_nodes(*args, **kwargs)


__all__ = [
    "ItemObservationEmbedding",
    "ItemVocabulary",
    "ObservedItemEmbedding",
    "ObservedItemMultisetEmbedding",
    "ObservedItemVocabulary",
    "PAD_INDEX",
    "RecipientConfirmedItemEmbedding",
    "UNK_INDEX",
    "VOCABULARY_VERSION",
    "OBSERVED_EMBEDDING_VERSION",
    "attach_item_embedding_to_nodes",
    "attach_observed_item_embedding_to_nodes",
    "fit_item_vocabulary",
]
