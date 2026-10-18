"""RIFT-EV snapshot model (Phase 1): relation- and distance-biased attention over the latest pre frame.

Adapted from the external RIFT-EV proposal (RIFT_EV_Implementation_Ideas_KO.md) to the q inputs; history and
event branches are Phase 2.  Inputs are exactly the q graph-family inputs of one condition
(``qbatch.assemble(..., "relational_mpnn")``) plus the raw pre-frame age in seconds:
  - tokens: 10 player tokens (V ``StructuredFeatureEncoder`` player lane -> H -> H) and one context token
    (team and global lanes -> H -> H);
  - L pre-norm blocks of multi-head attention over the 11 tokens with an additive per-head bias
        b_h(i, j) = r_h[type(i, j)] + e_h(edge_ij) - softplus(g_h) * (d_ij / sigma)^2 * exp(-softplus(l_h) * age / 60)
    type in {self, ally, enemy, context}; edge terms only between players; then a GELU feed-forward block;
  - readout: Blue/Red mean and max pools of the player tokens, K latent queries cross-attending to every
    token, the final context token and the raw context vector;
  - head: a mixture of E experts (256-128-1 MLPs) with a softmax gate on the raw context vector.
Every reduction is dense (no scatter kernels), so the model runs under the q CUDA determinism flags.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from studies.q_production_20261005.qmodels import SENDERS, GraphInputs, _mlp, _team_pool
from studies.v_model_fit_input_20261003.models import StructuredBatch, StructuredFeatureEncoder

PLAYERS = 10
TOKENS = PLAYERS + 1
SELF, ALLY, ENEMY, CONTEXT = range(4)
DROPOUT = 0.1
HEAD = (256, 128)


def _full_index() -> np.ndarray:
    """[10, 10] position of sender j in SENDERS[i] (9 = the appended zero slot for j == i)."""

    index = np.full((PLAYERS, PLAYERS), PLAYERS - 1, dtype=np.int64)
    for i in range(PLAYERS):
        for k, j in enumerate(SENDERS[i]):
            index[i, j] = k
    return index


FULL_INDEX = _full_index()


def dense_pairs(value: Tensor) -> Tensor:
    """[B, 10, 9, ...] receiver-sender tensor -> [B, 10, 10, ...] with zeros on the diagonal."""

    pad = torch.zeros_like(value[:, :, :1])
    padded = torch.cat((value, pad), dim=2)
    index = torch.as_tensor(FULL_INDEX, device=value.device)
    if value.ndim == 3:
        return torch.gather(padded, 2, index.unsqueeze(0).expand(value.shape[0], -1, -1))
    expanded = index.view(1, PLAYERS, PLAYERS, *([1] * (value.ndim - 3))).expand(value.shape[0], -1, -1, *value.shape[3:])
    return torch.gather(padded, 2, expanded)


class BiasedBlock(nn.Module):
    """Pre-norm attention with an additive [B, heads, 11, 11] bias, then a GELU feed-forward block."""

    def __init__(self, width: int, heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim = int(heads), width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_mult * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        b, n, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + bias
        attended = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, n, width)
        x = x + self.drop(self.out(attended))
        return x + self.drop(self.ffn(self.norm2(x)))


class LatentQueries(nn.Module):
    """K learned queries cross-attending to every token -> [B, K * width]."""

    def __init__(self, width: int, heads: int, queries: int) -> None:
        super().__init__()
        self.heads, self.head_dim, self.k = int(heads), width // heads, int(queries)
        self.query = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        b, n, width = x.shape
        k, v = self.kv(self.norm(x)).view(b, n, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.query.view(1, self.k, self.heads, self.head_dim).transpose(1, 2)
        attended = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1) @ v
        return self.out(attended.transpose(1, 2).reshape(b, self.k, width)).reshape(b, -1)


class RiftEVSnapshot(nn.Module):
    def __init__(self, *, encoder_kwargs: Mapping[str, Any], width: int, layers: int = 3, heads: int = 4, queries: int = 4,
                 experts: int = 3, ffn_mult: int = 2, dropout: float = DROPOUT, edge_hidden: int = 16) -> None:
        super().__init__()
        self.encoder = StructuredFeatureEncoder(include_side_flag=False, **dict(encoder_kwargs))
        width = int(width)
        context_dim = self.encoder.context_feature_dim
        self.player = _mlp(self.encoder.player_feature_dim, (width,), width)
        self.context = _mlp(context_dim, (width,), width)
        self.blocks = nn.ModuleList([BiasedBlock(width, heads, ffn_mult, dropout) for _ in range(layers)])
        self.relation = nn.Parameter(torch.zeros(layers, heads, 4))
        self.edge = nn.ModuleList([_mlp(3, (edge_hidden,), heads, dropout=0.0) for _ in range(layers)])
        self.distance_scale = nn.Parameter(torch.zeros(layers, heads))
        self.age_rate = nn.Parameter(torch.zeros(layers, heads))
        self.final_norm = nn.LayerNorm(width)
        self.latent = LatentQueries(width, heads, queries)
        readout = 4 * width + queries * width + width + context_dim
        self.gate = nn.Linear(context_dim, experts)
        self.experts = nn.ModuleList([_mlp(readout, HEAD, 1) for _ in range(experts)])
        self.register_buffer("relation_type", self._relation_template(), persistent=False)

    @staticmethod
    def _relation_template() -> Tensor:
        """[11, 11] relation of the context token (index 10) to every token; player pairs are filled per batch."""

        template = torch.full((TOKENS, TOKENS), CONTEXT, dtype=torch.long)
        template[:PLAYERS, :PLAYERS] = SELF
        return template

    def _bias(self, layer: int, ally: Tensor, edge: Tensor, dist2: Tensor, age: Tensor) -> Tensor:
        """[B, heads, 11, 11] additive attention bias of one layer."""

        b = ally.shape[0]
        eye = torch.eye(PLAYERS, dtype=torch.bool, device=ally.device)
        kind = torch.where(eye, torch.full_like(ally, SELF, dtype=torch.long),
                           torch.where(ally, torch.full_like(ally, ALLY, dtype=torch.long), torch.full_like(ally, ENEMY, dtype=torch.long)))
        relation = self.relation[layer]                               # [heads, 4]
        player = (nn.functional.one_hot(kind, 4).to(edge.dtype) @ relation.T).permute(0, 3, 1, 2)   # [B, heads, 10, 10]
        decay = torch.exp(-nn.functional.softplus(self.age_rate[layer]).view(1, -1, 1, 1) * (age / 60.0).view(-1, 1, 1, 1))
        player = player + self.edge[layer](edge).permute(0, 3, 1, 2) * (~eye).to(edge.dtype)
        player = player - nn.functional.softplus(self.distance_scale[layer]).view(1, -1, 1, 1) * dist2.unsqueeze(1) * decay
        full = relation[:, CONTEXT].view(1, -1, 1, 1).expand(b, -1, TOKENS, TOKENS).clone()
        full[:, :, :PLAYERS, :PLAYERS] = player
        return full

    def forward(self, batch: StructuredBatch, graph: GraphInputs, age: Tensor) -> Tensor:
        players, context, ids = self.encoder.encode_components(batch)
        device = players.device
        tokens = torch.cat((self.player(players), self.context(context).unsqueeze(1)), dim=1)
        edge = dense_pairs(torch.as_tensor(np.asarray(graph.edge), dtype=torch.float32, device=device))   # [B, 10, 10, 3]
        ally = dense_pairs(torch.as_tensor(np.asarray(graph.ally), dtype=torch.bool, device=device))
        dist2 = edge[..., 2] ** 2
        age = torch.as_tensor(np.asarray(age), dtype=torch.float32, device=device)
        x = tokens
        for layer, block in enumerate(self.blocks):
            x = block(x, self._bias(layer, ally, edge, dist2, age))
        x = self.final_norm(x)
        features = torch.cat((_team_pool(x[:, :PLAYERS], ids), self.latent(x), x[:, PLAYERS], context), dim=-1)
        gate = torch.softmax(self.gate(context), dim=-1)                                   # [B, experts]
        logits = torch.cat([expert(features) for expert in self.experts], dim=-1)          # [B, experts]
        return (gate * logits).sum(dim=-1)


def forward(model: RiftEVSnapshot, inputs: Any, device) -> Tensor:
    batch, graph, age = inputs
    return model(batch, graph, age).reshape(-1)
