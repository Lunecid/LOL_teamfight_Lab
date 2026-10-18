from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from core.common_torch import autocast_disabled as _autocast_disabled_ctx
from core.common_torch import nan_to_num as _nan_to_num_


@contextmanager
def _autocast_disabled():
    with _autocast_disabled_ctx():
        yield


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float()
    denom = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=dim) / denom


def normalize_adj_gcn(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    squeeze_back = False
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze_back = True

    with _autocast_disabled():
        A32 = _nan_to_num_(A.float())

        if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
            A32 = A32.clamp_min(0.0)

        if bool(getattr(cfg, "ADJ_SYMMETRIZE", False)):
            A32 = 0.5 * (A32 + A32.transpose(-1, -2))

        if bool(getattr(cfg, "ADJ_NORM_ENSURE_SELFLOOP", False)):
            _, n_nodes, _ = A32.shape
            eye = torch.eye(n_nodes, device=A32.device, dtype=A32.dtype).unsqueeze(0)
            A32 = torch.maximum(A32, eye)

        eps32 = float(getattr(cfg, "ADJ_NORM_EPS", eps))
        if eps32 <= 0:
            eps32 = 1e-6

        deg = A32.sum(dim=-1)
        deg = _nan_to_num_(deg).clamp_min(eps32)

        d_inv_sqrt = torch.rsqrt(deg)
        d_inv_sqrt = _nan_to_num_(d_inv_sqrt)

        A_norm = d_inv_sqrt.unsqueeze(-1) * A32 * d_inv_sqrt.unsqueeze(-2)
        A_norm = _nan_to_num_(A_norm)

        if not bool(getattr(cfg, "ADJ_NORM_OUT_FP32", True)):
            A_norm = A_norm.to(dtype=A.dtype)

    if squeeze_back:
        A_norm = A_norm.squeeze(0)
    return A_norm


class ResGCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.msg = nn.Linear(d_in, d_out, bias=False)
        self.self_proj = nn.Linear(d_in, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None
        self.norm2 = nn.LayerNorm(d_out) if use_norm else None

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        m = torch.bmm(A_norm, x)
        h = self.msg(m) + self.self_proj(x)
        h = F.gelu(h)
        h = self.dropout(h)
        if self.norm is not None:
            h = self.norm(h)
        return h


class GraphSAGELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(d_in * 2, d_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        with _autocast_disabled():
            A32 = _nan_to_num_(A.float())
            if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
                A32 = A32.clamp_min(0.0)
            x32 = _nan_to_num_(x.float())

            deg_eps = float(getattr(cfg, "SAGE_DEG_EPS", 1e-6))
            deg = A32.sum(dim=-1).clamp_min(deg_eps)
            neigh = torch.bmm(A32, x32) / deg.unsqueeze(-1)

            h32 = torch.cat([x32, neigh], dim=-1)
            h32 = self.lin(h32)
            h32 = F.relu(h32)

        h = h32.to(dtype=x.dtype)
        h = self.dropout(h)
        if self.norm is not None:
            h = self.norm(h)
        return h


class GATv2Layer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        nhead: int = 4,
        dropout: float = 0.0,
        use_norm: bool = True,
        leaky_alpha: float = 0.2,
    ):
        super().__init__()
        if nhead <= 0:
            nhead = 1
        if d_out % nhead != 0:
            nhead = 1
        self.nhead = nhead
        self.d_out = d_out
        self.dh = d_out // nhead

        self.Wq = nn.Linear(d_in, d_out, bias=False)
        self.Wk = nn.Linear(d_in, d_out, bias=False)
        self.Wv = nn.Linear(d_in, d_out, bias=False)

        self.att = nn.Parameter(torch.empty(nhead, self.dh))
        nn.init.xavier_uniform_(self.att.unsqueeze(-1))

        self.leaky_alpha = float(leaky_alpha)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None
        self.norm2 = nn.LayerNorm(d_out) if use_norm else None

        self.ff = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())
            A32 = _nan_to_num_(A.float())
            if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
                A32 = A32.clamp_min(0.0)

            q = self.Wq(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)
            k = self.Wk(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)
            v = self.Wv(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)

            qi = q.unsqueeze(-2)
            kj = k.unsqueeze(-3)
            z = qi + kj
            z = F.leaky_relu(z, negative_slope=self.leaky_alpha)

            att = self.att.view(1, self.nhead, 1, 1, self.dh)
            e = (z * att).sum(dim=-1)

            bias = torch.log(A32.clamp_min(1e-8)).unsqueeze(1)
            e = e + bias

            hard_th = float(getattr(cfg, "GAT_HARD_MASK_TH", 0.0))
            if hard_th > 0:
                m = (A32 >= hard_th).unsqueeze(1)
            else:
                m = (A32 > 0).unsqueeze(1)
            e = e.masked_fill(~m, -1e9)

            alpha = torch.softmax(e, dim=-1)
            alpha = _nan_to_num_(alpha)

            out = torch.matmul(alpha, v)
            out = out.transpose(1, 2).contiguous().view(B, N, self.d_out)
            out = _nan_to_num_(out)

        out = self.drop(out.to(dtype=x.dtype))
        h = x + out

        if self.norm is not None:
            h = self.norm(h)

        ff = self.drop(self.ff(h))
        h2 = h + ff
        if self.norm2 is not None:
            h2 = self.norm2(h2)
        return h2


class EdgeMPNNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, edge_dim: int = 4, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.edge_dim = int(edge_dim)
        hidden = int(getattr(cfg, "MPNN_HIDDEN", max(64, d_out)))

        self.phi = nn.Sequential(
            nn.Linear(d_in * 2 + self.edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )
        self.self_proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None

    @staticmethod
    def _edge_A_feat(A32: torch.Tensor) -> torch.Tensor:
        A32 = A32.clamp_min(0.0)
        amax = float(getattr(cfg, "EDGE_A_MAX", max(1.0, float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)))))
        amax = max(1e-8, amax)

        use_log1p = bool(getattr(cfg, "EDGE_A_LOG1P", True))
        if use_log1p:
            denom = math.log1p(amax)
            denom = max(1e-8, denom)
            a = torch.log1p(A32) / denom
        else:
            a = A32 / amax

        a = a.clamp(0.0, 1.0)
        return a

    @staticmethod
    def build_edge_attr(xy: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        with _autocast_disabled():
            xy32 = _nan_to_num_(xy.float())
            A32 = _nan_to_num_(A.float())

            dx = (xy32.unsqueeze(2) - xy32.unsqueeze(1))[..., 0]
            dy = (xy32.unsqueeze(2) - xy32.unsqueeze(1))[..., 1]
            dist = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-12))

            a_feat = EdgeMPNNLayer._edge_A_feat(A32)

            e = torch.stack([dx, dy, dist, a_feat], dim=-1)
            e = _nan_to_num_(e)
        return e

    def forward(self, x: torch.Tensor, A: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())
            A32 = _nan_to_num_(A.float())
            if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
                A32 = A32.clamp_min(0.0)

            e = self.build_edge_attr(xy, A32)

            hi = x32.unsqueeze(2).expand(B, N, N, D)
            hj = x32.unsqueeze(1).expand(B, N, N, D)
            inp = torch.cat([hi, hj, e], dim=-1)

            m = self.phi(inp)
            m = _nan_to_num_(m)

            m = m * A32.unsqueeze(-1)

            deg_eps = float(getattr(cfg, "MPNN_DEG_EPS", 1e-6))
            deg = A32.sum(dim=-1).clamp_min(deg_eps)
            agg = m.sum(dim=2) / deg.unsqueeze(-1)
            agg = _nan_to_num_(agg)

            h = F.relu(self.self_proj(x32) + agg)
            h = _nan_to_num_(h)

        h = self.drop(h.to(dtype=x.dtype))
        out = x + h if x.shape[-1] == h.shape[-1] else h

        if self.norm is not None:
            out = self.norm(out)
        return out


def _attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    Dh = q.shape[-1]
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(1.0, Dh))
    if attn_bias is not None:
        score = score + attn_bias
    w = torch.softmax(score, dim=-1)
    return torch.matmul(w, v)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model) if use_norm else None
        self.norm2 = nn.LayerNorm(d_model) if use_norm else None

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())
            A32 = _nan_to_num_(A.float())
            if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
                A32 = A32.clamp_min(0.0)

            hard_th = float(getattr(cfg, "GAT_HARD_MASK_TH", 0.0))
            if hard_th > 0:
                m = A32 >= hard_th
            else:
                m = A32 > 0

            neg_inf = torch.full_like(A32, -1e9)
            bias2 = torch.where(m, torch.log(A32.clamp_min(1e-8)), neg_inf)
            bias = bias2.unsqueeze(1)

            qkv = self.qkv(x32)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            def reshape(z):
                return z.view(B, N, self.nhead, self.dh).transpose(1, 2)

            q = reshape(q)
            k = reshape(k)
            v = reshape(v)

            attn_out = _attn(q, k, v, attn_bias=bias)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
            attn_out = self.proj(attn_out)
            attn_out = _nan_to_num_(attn_out)

        attn_out = self.drop(attn_out.to(dtype=x.dtype))

        h = x + attn_out
        if self.norm1 is not None:
            h = self.norm1(h)

        ff = self.drop(self.ff(h))
        h2 = h + ff
        if self.norm2 is not None:
            h2 = self.norm2(h2)
        return h2


def pool_team_repr(node_emb: torch.Tensor, alive: Optional[torch.Tensor] = None) -> torch.Tensor:
    blue = node_emb[:, :5, :]
    red = node_emb[:, 5:, :]

    if alive is None:
        b = blue.mean(dim=1)
        r = red.mean(dim=1)
    else:
        if alive.ndim == 1:
            alive = alive.unsqueeze(0)
        bmask = alive[:, :5]
        rmask = alive[:, 5:]
        b = masked_mean(blue, bmask, dim=1)
        r = masked_mean(red, rmask, dim=1)

    return torch.cat([b, r, b - r], dim=-1)


class GraphEncoder(nn.Module):
    def __init__(self, kind: str, d_in: int, d_h: int, n_layers: int = 2, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.kind = kind

        layers: List[nn.Module] = []
        din = d_in
        for i in range(n_layers):
            if kind == "gcn":
                layers.append(ResGCNLayer(din, d_h, dropout=dropout, use_norm=use_norm))
                din = d_h
            elif kind == "graphsage":
                layers.append(GraphSAGELayer(din, d_h, dropout=dropout, use_norm=use_norm))
                din = d_h
            elif kind == "graphtransformer":
                if i == 0 and din != d_h:
                    layers.append(nn.Linear(din, d_h))
                    din = d_h
                layers.append(
                    GraphTransformerLayer(
                        d_h,
                        nhead=max(1, int(getattr(cfg, "TRANS_NHEAD", 4))),
                        dropout=dropout,
                        use_norm=use_norm,
                    )
                )
                din = d_h
            elif kind == "gatv2":
                if i == 0 and din != d_h:
                    layers.append(nn.Linear(din, d_h))
                    din = d_h
                layers.append(
                    GATv2Layer(
                        d_in=din,
                        d_out=d_h,
                        nhead=int(getattr(cfg, "GAT_HEADS", max(1, int(getattr(cfg, "TRANS_NHEAD", 4))))),
                        dropout=dropout,
                        use_norm=use_norm,
                        leaky_alpha=float(getattr(cfg, "GAT_LEAKY_ALPHA", 0.2)),
                    )
                )
                din = d_h
            elif kind == "mpnn":
                if i == 0 and din != d_h:
                    layers.append(nn.Linear(din, d_h))
                    din = d_h
                layers.append(
                    EdgeMPNNLayer(
                        d_in=din,
                        d_out=d_h,
                        edge_dim=int(getattr(cfg, "MPNN_EDGE_DIM", 4)),
                        dropout=dropout,
                        use_norm=use_norm,
                    )
                )
                din = d_h
            else:
                raise ValueError(f"Unknown GraphEncoder kind={kind}")

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, A: torch.Tensor, xy: Optional[torch.Tensor] = None) -> torch.Tensor:
        force_fp32 = bool(getattr(cfg, "GNN_FORCE_FP32", True))

        if self.kind == "gcn":
            if force_fp32:
                with _autocast_disabled():
                    A_norm = normalize_adj_gcn(A)
                    h = _nan_to_num_(x.float())
                    for layer in self.layers:
                        h = layer(h, A_norm)
                return h.to(dtype=x.dtype)
            else:
                A_norm = normalize_adj_gcn(A)
                h = x
                for layer in self.layers:
                    h = layer(h, A_norm)
                return h

        if self.kind == "graphsage":
            h = _nan_to_num_(x) if force_fp32 else x
            for layer in self.layers:
                h = layer(h, A)
            return h

        if self.kind == "graphtransformer":
            h = x
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    h = layer(h)
                else:
                    h = layer(h, A)
            return h

        if self.kind == "gatv2":
            h = x
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    h = layer(h)
                else:
                    h = layer(h, A)
            return h

        if self.kind == "mpnn":
            if xy is None:
                raise ValueError("GraphEncoder(kind='mpnn') requires xy (B,N,2)")
            h = x
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    h = layer(h)
                else:
                    h = layer(h, A, xy)
            return h

        raise ValueError(self.kind)
