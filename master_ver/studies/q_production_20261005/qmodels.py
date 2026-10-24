"""q model families beyond V's adapters (ARCHITECTURE_KO.md section 4.3; EXECUTION_CARDS.json ``graph``,
``q.auxiliary_families``; HETEROGENEOUS_SPATIAL_GRAPH_DESIGN.md sections 2-5; DECISIONS D-Q3, D-Q6).

Graph models share one typed player encoder (V ``StructuredFeatureEncoder`` + ``->H->H``) over the
q superset lanes, one common pair readout on h0, team mean/max node pools and a 256-128-1 head:
  - ``team_set_pair_readout`` (G0): two self-only refinements h <- LN(h + MLP_{H->K->H}(h));
  - ``relational_mpnn`` (G1): two layers of ally/enemy messages phi(2H+3)->H->H weighted by w and
    divided by 4 / 5, update h <- LN(h + U_{3H->H->H}(h, m_ally, m_enemy));
  - ``tied_message``: G1 with one shared phi (separate ally/enemy aggregates), width matched;
  - ``graphsage``: two layers Linear(2H,H)(self, sum_j alpha_ij h_j) -> ReLU -> L2 normalize, alpha
    = softmax_j(-(d/sigma)^2) at S1 and 1/9 at S0; edge vectors only in the common pair readout.
The 90 directed edges are dense [10 receivers x 9 senders] (sender order ascending, self skipped),
so every reduction is a deterministic sum (no scatter kernels under CUDA determinism).
``FTTransformer``: one token per numeric field, categorical field and bag field, plus CLS; pre-norm
blocks with ReGLU (x2) feed-forward; head CLS -> LayerNorm -> ReLU -> Linear (D-Q3).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch, StructuredFeatureEncoder

PLAYERS = 10
SENDERS = np.asarray([[j for j in range(PLAYERS) if j != i] for i in range(PLAYERS)], dtype=np.int64)   # [10, 9]
GRAPH_FAMILIES = ("team_set_pair_readout", "relational_mpnn", "tied_message", "graphsage")
DROPOUT = 0.1
HEAD = (256, 128)


# ----------------------------------------------------------------------------- graph inputs
@dataclass
class GraphInputs:
    """Dense directed-edge tensors for one batch; edge (i, k) is sender SENDERS[i, k] -> receiver i."""

    edge: np.ndarray        # [B, 10, 9, 3] ((x_j-x_i)/sigma, (y_j-y_i)/sigma, d/sigma); zeros at S0
    weight: np.ndarray      # [B, 10, 9] exp(-(d/sigma)^2); ones at S0
    ally: np.ndarray        # [B, 10, 9] bool, same team
    team_ids: np.ndarray    # [B, 10] 100 / 200
    sage_log_weight: np.ndarray  # [B, 10, 9] -(d/sigma)^2 at S1, zeros at S0 (softmax gives 1/9)


def graph_inputs(xy: np.ndarray | None, team_ids: np.ndarray, sigma: float | None, spatial: bool) -> GraphInputs:
    """Edge features from raw pre-frame XY and the scope's sigma (EXECUTION_CARDS.json graph; S0 constants)."""

    team_ids = np.asarray(team_ids, dtype=np.int64)
    n = len(team_ids)
    ally = team_ids[:, SENDERS] == team_ids[:, :, None]          # [B, 10, 9]: sender on the receiver's team
    if not spatial:
        return GraphInputs(np.zeros((n, PLAYERS, PLAYERS - 1, 3), np.float32), np.ones((n, PLAYERS, PLAYERS - 1), np.float32),
                           ally, team_ids, np.zeros((n, PLAYERS, PLAYERS - 1), np.float32))
    if xy is None or sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("spatial graph inputs need XY and a positive sigma")
    xy = np.asarray(xy, dtype=np.float64)
    sender_xy = xy[:, SENDERS, :]                       # [B, 10, 9, 2]
    delta = (sender_xy - xy[:, :, None, :]) / sigma     # x_j - x_i
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    edge = np.concatenate((delta, dist[..., None]), axis=-1)
    log_w = -(dist ** 2)
    return GraphInputs(edge.astype(np.float32), np.exp(log_w).astype(np.float32), ally, team_ids, log_w.astype(np.float32))


def _t(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _mlp(inp: int, hidden: Sequence[int], out: int, dropout: float = DROPOUT) -> nn.Sequential:
    dims = [int(inp), *(int(h) for h in hidden), int(out)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.extend((nn.GELU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


def _gather_senders(h: Tensor) -> Tensor:
    """[B, 10, H] -> [B, 10, 9, H] sender states in SENDERS order."""

    index = torch.as_tensor(SENDERS, device=h.device)
    return h[:, index, :]


def _team_pool(h: Tensor, team_ids: Tensor) -> Tensor:
    """Blue/Red mean and max of node states -> [B, 4H] (order: Blue mean, Blue max, Red mean, Red max)."""

    pieces = []
    for team in (100, 200):
        mask = (team_ids == team).unsqueeze(-1)
        mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        maximum = h.masked_fill(~mask, -torch.inf).amax(dim=1)
        pieces.extend((mean, maximum))
    return torch.cat(pieces, dim=-1)


class PairReadout(nn.Module):
    """phi_pair(2H+5)->H->H on (h0_i, h0_j, relation one-hot 2, edge 3); w-weighted /4 and /5, then team mean."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.phi = _mlp(2 * hidden + 5, (hidden,), hidden)

    def forward(self, h0: Tensor, graph: GraphInputs) -> Tensor:
        device = h0.device
        ally = _t(graph.ally, device, torch.bool)
        relation = torch.stack((ally, ~ally), dim=-1).to(h0.dtype)
        receiver = h0.unsqueeze(2).expand(-1, -1, PLAYERS - 1, -1)
        message = self.phi(torch.cat((receiver, _gather_senders(h0), relation, _t(graph.edge, device)), dim=-1))
        weighted = message * _t(graph.weight, device).unsqueeze(-1)
        ally_f = ally.unsqueeze(-1).to(h0.dtype)
        per_ally = (weighted * ally_f).sum(dim=2) / 4.0
        per_enemy = (weighted * (1 - ally_f)).sum(dim=2) / 5.0
        team_ids = _t(graph.team_ids, device, torch.long)
        pieces = []
        for team in (100, 200):
            mask = (team_ids == team).unsqueeze(-1).to(h0.dtype)
            pieces.append((per_ally * mask).sum(dim=1) / mask.sum(dim=1))
            pieces.append((per_enemy * mask).sum(dim=1) / mask.sum(dim=1))
        return torch.cat(pieces, dim=-1)


class SelfRefine(nn.Module):
    """G0: h <- LN(h + MLP_{H->K->H}(h))."""

    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()
        self.mlp = _mlp(hidden, (width,), hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, graph: GraphInputs) -> Tensor:
        return self.norm(h + self.mlp(h))


class RelationalLayer(nn.Module):
    """G1 (or tied): ally/enemy messages phi(2H+3)->W->H, w-weighted sums /4 and /5, update U(3H)->H->H residual LN."""

    def __init__(self, hidden: int, tied: bool = False, phi_width: int | None = None) -> None:
        super().__init__()
        width = int(phi_width or hidden)
        self.tied = bool(tied)
        self.phi_ally = _mlp(2 * hidden + 3, (width,), hidden)
        self.phi_enemy = None if tied else _mlp(2 * hidden + 3, (width,), hidden)
        self.update = _mlp(3 * hidden, (hidden,), hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, graph: GraphInputs) -> Tensor:
        device = h.device
        receiver = h.unsqueeze(2).expand(-1, -1, PLAYERS - 1, -1)
        pair = torch.cat((receiver, _gather_senders(h), _t(graph.edge, device)), dim=-1)
        w = _t(graph.weight, device).unsqueeze(-1)
        ally = _t(graph.ally, device, torch.bool).unsqueeze(-1).to(h.dtype)
        ally_msg = self.phi_ally(pair)
        enemy_msg = ally_msg if self.tied else self.phi_enemy(pair)
        m_ally = (ally_msg * w * ally).sum(dim=2) / 4.0
        m_enemy = (enemy_msg * w * (1 - ally)).sum(dim=2) / 5.0
        return self.norm(h + self.update(torch.cat((h, m_ally, m_enemy), dim=-1)))


class SageLayer(nn.Module):
    """GraphSAGE reference: Linear(2H,H)(self, sum_j alpha_ij h_j) -> ReLU -> L2 normalize (eps 1e-12)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * hidden, hidden)

    def forward(self, h: Tensor, graph: GraphInputs) -> Tensor:
        alpha = torch.softmax(_t(graph.sage_log_weight, h.device), dim=-1).unsqueeze(-1)   # stable log-weight normalization
        neighbour = (alpha * _gather_senders(h)).sum(dim=2)
        out = torch.relu(self.linear(torch.cat((h, neighbour), dim=-1)))
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class QGraphModel(nn.Module):
    """Typed player encoder -> h0 -> two refinement layers -> node pools + common pair pools + head."""

    def __init__(self, family: str, *, hidden_width: int, encoder_kwargs: Mapping[str, Any], g0_width: int | None = None,
                 tied_width: int | None = None, weight_decay: float = 0.0) -> None:
        super().__init__()
        if family not in GRAPH_FAMILIES:
            raise ValueError(f"unknown q graph family {family!r}")
        self.family, self.hidden_width, self.weight_decay = family, int(hidden_width), float(weight_decay)
        self.encoder = StructuredFeatureEncoder(include_side_flag=False, **dict(encoder_kwargs))
        hidden = self.hidden_width
        self.player = _mlp(self.encoder.player_feature_dim, (hidden,), hidden)
        self.input_dropout = nn.Dropout(DROPOUT) if family == "graphsage" else nn.Identity()
        if family == "team_set_pair_readout":
            if g0_width is None:
                raise ValueError("G0 needs its structural width K")
            layers = [SelfRefine(hidden, g0_width) for _ in range(2)]
        elif family == "relational_mpnn":
            layers = [RelationalLayer(hidden) for _ in range(2)]
        elif family == "tied_message":
            layers = [RelationalLayer(hidden, tied=True, phi_width=tied_width) for _ in range(2)]
        else:
            layers = [SageLayer(hidden) for _ in range(2)]
        self.layers = nn.ModuleList(layers)
        self.pair = PairReadout(hidden)
        self.head = _mlp(self.encoder.context_feature_dim + 8 * hidden, HEAD, 1)

    def forward(self, batch: StructuredBatch, graph: GraphInputs) -> Tensor:
        players, context, ids = self.encoder.encode_components(batch)
        h0 = self.player(players)
        h = self.input_dropout(h0)
        for layer in self.layers:
            h = layer(h, graph)
        features = torch.cat((context, _team_pool(h, ids), self.pair(h0, graph)), dim=-1)
        return self.head(features).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def nearest8(value: float) -> int:
    return max(8, int(8 * round(value / 8.0)))


def matched_width(hidden: int, build: Any, reference_params: int, start: int, max_gap: float = 0.05) -> tuple[int, float]:
    """Smallest-move multiple of 8 from ``start`` whose built model is within ``max_gap`` of the reference."""

    for step in range(0, 400):
        for width in ((start,) if step == 0 else (start + 8 * step, start - 8 * step)):
            if width < 8:
                continue
            gap = abs(parameter_count(build(width)) - reference_params) / reference_params
            if gap <= max_gap:
                return width, gap
    raise ValueError("no width within the parameter gap")


def g0_start_width(hidden: int) -> int:
    """nearest8((10H^2 + 11H) / (2H + 1)) (EXECUTION_CARDS.json graph.G0_width)."""

    return nearest8((10 * hidden ** 2 + 11 * hidden) / (2 * hidden + 1))


# ----------------------------------------------------------------------------- FT-Transformer
class _Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d_token must be divisible by heads")
        self.d, self.heads = d, heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.attn_drop, self.res_drop, self.ffn_drop = nn.Dropout(DROPOUT), nn.Dropout(DROPOUT), nn.Dropout(DROPOUT)
        hidden = 2 * d                                   # ffn_multiplier 2
        self.ffn_in, self.ffn_out = nn.Linear(d, 2 * hidden), nn.Linear(hidden, d)

    def forward(self, x: Tensor) -> Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        x = x + self.res_drop(self.out(torch.matmul(attn, v).transpose(1, 2).reshape(b, length, d)))
        a, gate = self.ffn_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.res_drop(self.ffn_out(self.ffn_drop(torch.relu(a) * gate)))   # ReGLU


class FTTransformer(nn.Module):
    """Field tokens (numeric w_j*x+b_j, categorical embedding+bias, bag count-weighted sum + count + bias) + CLS."""

    family = "ft_transformer"

    def __init__(self, *, numeric_fields: int, category_fields: Sequence[tuple[str, int]], bag_fields: Sequence[tuple[str, int]],
                 d_token: int, layers: int, heads: int) -> None:
        super().__init__()
        d = int(d_token)
        self.d = d
        self.numeric_weight = nn.Parameter(torch.empty(numeric_fields, d))
        self.numeric_bias = nn.Parameter(torch.zeros(numeric_fields, d))
        nn.init.normal_(self.numeric_weight, std=1 / math.sqrt(d))
        self.category_fields, self.bag_fields = list(category_fields), list(bag_fields)
        namespaces = sorted({ns for ns, _ in self.category_fields} | {ns for ns, _ in self.bag_fields})
        sizes = dict(self.category_fields) | dict(self.bag_fields)
        self.tables = nn.ModuleDict({ns.replace(":", "__"): nn.Embedding(sizes[ns], d, padding_idx=0) for ns in namespaces})
        self.category_bias = nn.Parameter(torch.zeros(len(self.category_fields), d))
        self.bag_bias = nn.Parameter(torch.zeros(len(self.bag_fields), d))
        self.bag_count = nn.Parameter(torch.zeros(len(self.bag_fields), d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=1 / math.sqrt(d))
        self.blocks = nn.ModuleList(_Block(d, int(heads)) for _ in range(int(layers)))
        self.head_norm, self.head = nn.LayerNorm(d), nn.Linear(d, 1)

    @property
    def token_count(self) -> int:
        return 1 + int(self.numeric_weight.shape[0]) + len(self.category_fields) + len(self.bag_fields)

    def forward(self, numeric: Tensor, categories: Tensor, bags: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tensor:
        """numeric [B, Fn]; categories [B, Fc] (indices); bags: per bag field (indices, counts, row_of_entry, observed[B])."""

        b = numeric.shape[0]
        tokens = [self.cls.expand(b, -1, -1), (numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias)]
        if self.category_fields:
            cat = torch.stack([self.tables[ns.replace(":", "__")](categories[:, f]) for f, (ns, _) in enumerate(self.category_fields)], dim=1)
            tokens.append(cat + self.category_bias)
        if self.bag_fields:
            pooled = []
            for f, ((ns, _), (indices, counts, row_of, observed)) in enumerate(zip(self.bag_fields, bags)):
                emb = self.tables[ns.replace(":", "__")](indices) * counts.unsqueeze(-1)
                summed = torch.zeros(b, self.d, device=numeric.device, dtype=numeric.dtype)
                if indices.numel():
                    # Row sums by a dense one-hot matmul keep the reduction deterministic on CUDA.
                    onehot = torch.nn.functional.one_hot(row_of, b).to(emb.dtype).transpose(0, 1)
                    summed = onehot @ emb
                pooled.append(summed + observed.unsqueeze(-1) * self.bag_count[f] + self.bag_bias[f])
            tokens.append(torch.stack(pooled, dim=1))
        x = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(torch.relu(self.head_norm(x[:, 0]))).squeeze(-1)


def ft_bag_inputs(bag: BagInput, slots: int, device: torch.device) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Split one BagInput into per-slot bag fields (indices, counts, row of each entry, observed count)."""

    offsets = np.asarray(bag.offsets)
    indices, counts = np.asarray(bag.indices), np.asarray(bag.counts, dtype=np.float32)
    observed = np.asarray(bag.observed_count, dtype=np.float32).reshape(len(offsets), slots)
    out = []
    for slot in range(slots):
        start, stop = offsets[:, slot], offsets[:, slot + 1]
        lengths = stop - start
        flat = np.concatenate([np.arange(a, z) for a, z in zip(start, stop)]) if lengths.sum() else np.zeros(0, np.int64)
        rows = np.repeat(np.arange(len(offsets)), lengths)
        out.append((_t(indices[flat], device, torch.long), _t(counts[flat], device), _t(rows, device, torch.long), _t(observed[:, slot], device)))
    return out
