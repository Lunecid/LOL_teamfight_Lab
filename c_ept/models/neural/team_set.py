"""N2 TeamSet: shared entity MLPs, team sets and pooled raw relations.

This is an independent encoder, with the token layout and prediction heads of
EPT. Relations are endpoint-conditioned set elements, never attention scores or
messages updating neighbouring players. All endpoint representations are formed
once, before any relation or global pooling. The final global vector conditions
every entity, retaining distinct player outputs for the marked heads.

Input contract: the existing ViewTensors, excluding the non-predictive ``view``
number. Missing essential fields fail closed. Padded events/edges and masked
geometry are zeroed before any learned function. Geometry, field interpolation,
static-channel policy and category embeddings follow the current EPT formulas;
no EPT encoder, attention block or co-participation parameter is constructed.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from c_ept import config as C
from c_ept.ept import data
from c_ept.ept.encoder import Encoded, EncoderConfig, Positions, TerrainField, token_layout
from c_ept.ept.heads import EPTHeads
from c_ept.preprocessing import terrain, tokens

_P, _L = data.N_PLAYERS, data.N_LANDMARKS
_REQUIRED = tuple(k for k in data.VIEW_KEYS if k != 'view')
_X = data.PLAYER_NAN_COLUMNS.index('x')
_EDGE_KINDS = len(tokens.EDGE_KINDS)
# ally, enemy, player-landmark, player-event, event actor, then each edge kind.
_N_RELATIONS = 5 + _EDGE_KINDS
_RELATION_SCALARS = 5  # log Euclidean distance, known, geodesic, missing, log age


def _mlp(n_in: int, hidden: int, d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))


def _pool(h: torch.Tensor, mask: torch.Tensor, *, include_sum: bool = False) -> torch.Tensor:
    """Masked mean and explicit cardinality (also sum for each player team)."""
    total = torch.where(mask[..., None], h, 0.0).sum(dim=1)
    count = mask.sum(dim=1, keepdim=True).to(h.dtype)
    mean = total / count.clamp_min(1.0)
    return torch.cat(([total] if include_sum else []) + [mean, count], dim=-1)


class TeamSetEncoder(nn.Module):
    """N2 encoder, output width ``cfg.d``, shared MLP hidden width ``hidden``.

    ``cfg.dropout`` and ``cfg.field_channels`` retain their usual meaning.
    ``cfg.layers/heads/kappas`` remain compatibility metadata; N2 has no attention
    layers or learned decay scales. Its three MLPs each have two linear layers.
    ``field`` exposes the existing TerrainField (including its penalty), or None.
    """

    def __init__(self, cfg: EncoderConfig, static: np.ndarray | bool, *, hidden: int = 512):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive whole number')
        if static is None or static is True:
            raise ValueError('static channels are required; use static=False to omit them explicitly')
        self.cfg, self.hidden = cfg, hidden
        d = cfg.d
        # Same entity input equations as EPTEncoder._embed, with independent weights.
        self.players_in = nn.Linear(len(data.PLAYER_COLUMNS) + len(data.PLAYER_NAN_COLUMNS), d)
        self.landmarks_in, self.events_in = nn.Linear(3, d), nn.Linear(5, d)
        self.summary_in = nn.Linear(len(data.SUMMARY_COLUMNS) + len(data.SUMMARY_NAN_COLUMNS), d)
        self.context_in = nn.Linear(2, d)
        self.zone_emb, self.group_emb = nn.Embedding(len(terrain.ZONES) + 1, d), nn.Embedding(2, d)
        self.landmark_emb, self.kind_emb = nn.Embedding(_L, d), nn.Embedding(3, d)
        self.type_emb, self.team_emb = nn.Embedding(len(C.EVENT_TYPES), d), nn.Embedding(3, d)
        for emb in (self.zone_emb, self.group_emb, self.landmark_emb, self.kind_emb, self.type_emb, self.team_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.register_buffer('group', torch.arange(_P) // (_P // 2), persistent=False)
        xy = np.asarray(tokens.LANDMARK_XY, dtype=np.float64)
        self.register_buffer('landmark_xy', torch.as_tensor(np.stack([xy, xy[data.LANDMARK_PERM]])), persistent=False)
        undirected = torch.zeros(_EDGE_KINDS, dtype=torch.bool)
        undirected[[tokens.EDGE_SIDE, tokens.EDGE_OBJECTIVE]] = True
        self.register_buffer('undirected', undirected, persistent=False)
        fixed = None if static is False else static
        self.field = TerrainField(cfg.field_channels, fixed) if cfg.field_channels or fixed is not None else None
        if self.field is not None:
            self.field_proj = nn.Linear(self.field.n_channels, d, bias=False)
        self.entity_mlp = _mlp(d, hidden, d, cfg.dropout)
        self.relation_mlp = _mlp(2 * d + _N_RELATIONS + _RELATION_SCALARS, hidden, d, cfg.dropout)
        # 2 teams*(sum+mean+count), 4 sets*(mean+count), and summary.
        self.pool_mlp = _mlp(9 * d + 6, hidden, d, cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def physical_positions(self, batch: Mapping) -> Positions:
        """Same physical-coordinate/unknown-position rule as EPT, no attention."""
        flipped = batch['side'] > 0.5
        def turn(xy):
            return torch.where(flipped[:, None, None], data.W - xy, xy)
        known_p = batch['players_nan'][..., _X] < 0.5
        known_e = batch['ev_valid'] & (batch['ev_nan'][..., 0] < 0.5)
        lm = self.landmark_xy[flipped.long()].to(batch['players_xy'].dtype)
        xy = torch.cat([turn(batch['players_xy']), lm, turn(batch['ev_xy'])], dim=1)
        known = torch.cat([known_p, torch.ones_like(known_p[:, :1]).expand(-1, _L), known_e], dim=1)
        return Positions(torch.where(known[..., None], xy, data.W / 2.0), known)

    def _embed(self, b: Mapping, positions: Positions) -> torch.Tensor:
        valid = b['ev_valid']
        hp = self.players_in(torch.cat([b['players'], b['players_nan']], -1)) + self.zone_emb(b['players_zone'] + 1) + self.group_emb(self.group)
        hl = self.landmarks_in(torch.cat([b['landmarks'], b['landmarks_nan']], -1)) + self.landmark_emb.weight
        event = torch.where(valid[..., None], torch.cat([b['ev_num'], b['ev_nan']], -1), 0.0)
        he = self.events_in(event) + self.type_emb(torch.where(valid, b['ev_type'], 0)) + self.team_emb(torch.where(valid, b['ev_team'], 2))
        hs = self.summary_in(torch.cat([b['summary'], b['summary_nan']], -1))[:, None]
        h = torch.cat([hp, hl, he], dim=1)
        if self.field is not None:
            h = h + torch.where(positions.known[..., None], self.field_proj(self.field.sample(positions.xy)), 0.0)
        context = self.kind_emb(b['kind']) + self.context_in(torch.stack([b['age'], b['side']], -1))
        return torch.cat([h, hs], dim=1) + context[:, None]

    def _relations(self, b: Mapping, h: torch.Tensor, positions: Positions) -> torch.Tensor:
        """One shared map of raw player-pair/landmark/event relations, then pool.

        Each map reads its two *local* endpoint embeddings, relation type, masked
        geodesic, physical distance and masks. No adjacency-normalized weights,
        neighbour aggregation into players, or iterative endpoint updates.
        """
        n, _, d = h.shape
        xy, known = positions
        n_loc = xy.shape[1]
        present = torch.cat([b['ev_valid'].new_ones(n, _P + _L), b['ev_valid']], dim=1)
        pair_valid = present[:, None].expand(-1, _P, -1)
        located = known[:, :_P, None] & known[:, None]
        delta = torch.where(located[..., None], xy[:, :_P, None] - xy[:, None], 0.0)
        distance = torch.log1p(torch.linalg.vector_norm(delta, dim=-1) / C.L_U)
        geo_missing = torch.cat([b['geo_pp_nan'] > 0.5, b['geo_pl_nan'] > 0.5,
                                 torch.ones_like(b['ev_valid'])[:, None].expand(-1, _P, -1)], dim=2)
        geo = torch.cat([b['geo_pp'], b['geo_pl'], distance.new_zeros(n, _P, b['ev_valid'].shape[1])], dim=2)
        geo = torch.where(~geo_missing, geo, 0.0)
        pp_type = (self.group[:, None] != self.group[None]).long()[None].expand(n, -1, -1)
        actor = (torch.where(b['ev_valid'][..., None], b['ev_actors'], 0.0).transpose(1, 2) > 0.5).long()
        kind = torch.cat([pp_type, pp_type.new_full((n, _P, _L), 2), actor + 3], dim=2)
        raw = torch.cat([F.one_hot(kind, _N_RELATIONS).to(h.dtype),
                         torch.stack([distance, located.to(h.dtype), geo, geo_missing.to(h.dtype), torch.zeros_like(distance)], -1)], -1)
        first = h[:, :_P, None].expand(-1, -1, n_loc, -1)
        second = h[:, None, :n_loc].expand(-1, _P, -1, -1)
        inputs = torch.where(pair_valid[..., None], torch.cat([first, second, raw], -1), 0.0)
        rel = self.relation_mlp(inputs).reshape(n, _P * n_loc, d)
        return _pool(rel, pair_valid.reshape(n, _P * n_loc))

    def _edges(self, b: Mapping, h: torch.Tensor) -> torch.Tensor:
        valid = b['ed_valid']
        first = torch.where(valid, b['ed_i'], 0)
        second = torch.where(valid, b['ed_j'], 0)
        kind = torch.where(valid, b['ed_kind'], 0)
        if ((first < 0) | (first >= _P) | (second < 0) | (second >= _P)).any():
            raise ValueError('a valid edge has an invalid player endpoint')
        if ((kind < 0) | (kind >= _EDGE_KINDS)).any():
            raise ValueError('a valid edge has an invalid kind')
        age = torch.where(valid, b['ed_dt'], 0.0)
        if ((age < 0) | ~torch.isfinite(age)).any():
            raise ValueError('valid edge ages must be finite and nonnegative')
        hp = h[:, :_P]
        a = hp.gather(1, first[..., None].expand(-1, -1, h.shape[-1]))
        z = hp.gather(1, second[..., None].expand(-1, -1, h.shape[-1]))
        tied = self.undirected[kind][..., None]
        # Unordered endpoint pairs have no slot-order feature; directed roles do.
        left, right = torch.where(tied, (a + z) / 2.0, a), torch.where(tied, (a - z).abs(), z)
        zero = torch.zeros_like(age)
        raw = torch.cat([F.one_hot(kind + 5, _N_RELATIONS).to(h.dtype),
                         torch.stack([zero, zero, zero, torch.ones_like(age), torch.log1p(age)], -1)], -1)
        inputs = torch.where(valid[..., None], torch.cat([left, right, raw], -1), 0.0)
        return _pool(self.relation_mlp(inputs), valid)

    def forward(self, batch: Mapping) -> Encoded:
        missing = [k for k in _REQUIRED if k not in batch]
        if missing:
            raise ValueError(f'the batch lacks the view tensors {missing}')
        valid = batch['ev_valid']
        layout = token_layout(valid.shape[1])
        mask = torch.cat([valid.new_ones(valid.shape[0], _P + _L), valid, valid.new_ones(valid.shape[0], 1)], dim=1)
        positions = self.physical_positions(batch)
        h = torch.where(mask[..., None], self._embed(batch, positions), 0.0)
        h = torch.where(mask[..., None], self.entity_mlp(h), 0.0)
        p, lm, ev = h[:, layout.players], h[:, layout.landmarks], h[:, layout.events]
        pieces = [_pool(p[:, :_P // 2], mask[:, :_P // 2], include_sum=True),
                  _pool(p[:, _P // 2:], mask[:, _P // 2:_P], include_sum=True),
                  _pool(lm, mask[:, layout.landmarks]), _pool(ev, valid),
                  self._relations(batch, h, positions), self._edges(batch, h), h[:, layout.summary]]
        context = self.pool_mlp(torch.cat(pieces, -1))
        output = torch.where(mask[..., None], self.norm(h + context[:, None]), 0.0)
        return Encoded(output, output[:, layout.summary], mask)


class TeamSetModel(nn.Module):
    """Explicit N2 model; ``encode`` and ``heads`` satisfy FrameAveraged's API.

    ``static`` has the existing EPT field policy (required array, or explicit
    False). ``hidden`` changes real MLP capacity and is identity metadata that
    the integrating checkpoint/registry must bind along with architecture_id.
    """
    architecture_id = 'n2_teamset_v1'

    def __init__(self, cfg: EncoderConfig, static: np.ndarray | bool, *, hidden: int = 512):
        super().__init__()
        self.encoder = TeamSetEncoder(cfg, static, hidden=hidden)
        self.heads = EPTHeads(cfg)

    def encode(self, batch: Mapping) -> Encoded:
        return self.encoder(batch)
