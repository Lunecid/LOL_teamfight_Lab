# lol_teamfight/deep/models_spatiotemporal_v2.py
from __future__ import annotations

import logging
import math
import time
import contextlib
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# =========================================================
# Robust imports (project-safe)
# =========================================================
# ─────────────────────────────────────────────────────────────
# [P4-IMPORT] Simplified imports — removed duplicate try/except
# where try and except blocks contained identical code (Issue #3).
# ─────────────────────────────────────────────────────────────
from core.config import cfg  # type: ignore
from core.utils import set_seed, write_log, save_json  # type: ignore
from data.file_io import ensure_dir  # type: ignore

# ref_key는 core.fight_types에서 단일 경로로 로드한다.
from core.fight_types import ref_key  # type: ignore

# Ã¢Å“â€¦ build_model(factory) import Ã¬â€¢Ë†Ã¬Â â€žÃ­â„¢â€
from train.models import build_model  # type: ignore  # [P4-IMPORT]

# Dataset/collate import path: fallback only when legacy package is missing.
try:
    from data_loader.dataset import InMemoryFightDataset, collate_batch  # type: ignore
except ModuleNotFoundError as e:
    # Do not swallow unrelated import errors raised inside data_loader.dataset.
    if getattr(e, "name", "") not in {"data_loader", "data_loader.dataset"}:
        raise
    from data.dataset import InMemoryFightDataset, LogitInjectorDataset, collate_batch  # type: ignore


# =========================================================
# Small helpers (self-contained)
# =========================================================
from train.deep_eval import (
    _LogitAuxHead,
    _autocast_ctx,
    _eval_loop,
    _extract_main_logit,
    _predict_logit_and_label_maps_for_loader,
    _predict_logit_map_for_refs,
    _resolve_amp_dtype,
    _to_device,
)
from core.common_torch import autocast_disabled as _autocast_disabled_impl
from core.common_torch import nan_to_num as _nan_to_num_impl


@contextmanager
def _autocast_disabled():
    """Thin wrapper — delegates to common_torch.autocast_disabled()."""
    with _autocast_disabled_impl():
        yield


def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    """Thin wrapper — delegates to common_torch.nan_to_num()."""
    return _nan_to_num_impl(x)


# =========================================================
# Robust NODE_IDX / NODE_FEATURE_NAMES import
# =========================================================
# [P4-DEDUP] NODE_IDX now resolved via common_torch (single source of truth)
from core.common_torch import (
    resolve_node_idx as _resolve_node_idx_impl,
    idx_required as _idx_required,
    idx_optional as _idx_optional,
    idx_first as _idx_first,
)

NODE_IDX: Dict[str, int] = _resolve_node_idx_impl()

# [P4-DEDUP] _idx_required / _idx_optional / _idx_first → imported from common_torch


# ---- position (required) ----
X_IDX = _idx_required(NODE_IDX, "x_norm")
Y_IDX = _idx_required(NODE_IDX, "y_norm")

# ---- alive (required) ----
ALIVE_IDX = _idx_required(NODE_IDX, "alive")

# ---- core snapshot/status (required) ----
HP_IDX = _idx_required(NODE_IDX, "hp_pct")
LVL_IDX = _idx_required(NODE_IDX, "level_norm")

# gold: prefer totalGold_norm, fallback to curGold_norm
GOLD_IDX = _idx_first(NODE_IDX, ["totalGold_norm", "curGold_norm"])
if GOLD_IDX is None:
    raise KeyError("[CONFIG MISMATCH] Need one of: totalGold_norm / curGold_norm")

CC_IDX = _idx_required(NODE_IDX, "ccTime_norm")
VISION_IDX = _idx_optional(NODE_IDX, "vision_nearby_score_norm")
FLASH_IDX = _idx_optional(NODE_IDX, "flash_ready")
ULT_IDX = _idx_required(NODE_IDX, "ult_level_norm")

# optional: recent damage proxy (ds schema)
DMG_TOC_IDX = _idx_optional(NODE_IDX, "ds_totalDamageDoneToChampions")


# =========================================================
# Ã¢Å“â€¦ NEW: NodeFeatureAdapter (categorical id -> embedding)
# =========================================================
class NodeFeatureAdapter(nn.Module):
    """
    If you added categorical ID-like node features (champion_id / rune_id / etc.),
    feeding raw integers into Linear can destabilize learning.
    This adapter optionally replaces those columns with embeddings.

    Configure via cfg:
      - cfg.NODE_CAT_SPECS or cfg.NODE_CATEGORICAL_SPECS:
          dict feature_name -> dict(num_embeddings=..., emb_dim=...)
    Example:
      NODE_CAT_SPECS = {
        "champion_id": {"num_embeddings": 1000, "emb_dim": 16},
        "perk_primary": {"num_embeddings": 5000, "emb_dim": 8},
      }
    If not provided, adapter becomes a no-op (passes x through).
    """

    def __init__(self, f_node: int, node_idx: Dict[str, int]):
        super().__init__()
        self.f_node = int(f_node)
        self.node_idx = dict(node_idx)

        specs = getattr(cfg, "NODE_CAT_SPECS", None)
        if specs is None:
            specs = getattr(cfg, "NODE_CATEGORICAL_SPECS", None)
        if specs is None:
            specs = {}

        # only keep specs that exist in NODE_IDX
        self.cat_items: List[Tuple[str, int, int, int]] = []  # (name, idx, num_emb, emb_dim)
        emb_layers = {}

        for name, s in dict(specs).items():
            if name not in self.node_idx:
                continue
            idx = int(self.node_idx[name])
            num_emb = int(s.get("num_embeddings", s.get("num_emb", 0)))
            emb_dim = int(s.get("emb_dim", 0))
            if num_emb <= 0 or emb_dim <= 0:
                continue
            emb_layers[name] = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
            self.cat_items.append((name, idx, num_emb, emb_dim))

        self.embeddings = nn.ModuleDict(emb_layers)

        # build continuous indices (exclude categorical columns)
        cat_idx_set = {idx for (_, idx, _, _) in self.cat_items}
        self.cont_indices = [i for i in range(self.f_node) if i not in cat_idx_set]

        self.out_dim = len(self.cont_indices) + sum(emb_dim for (_, _, _, emb_dim) in self.cat_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., F)
        if not self.cat_items:
            return x

        orig_shape = x.shape
        F = orig_shape[-1]
        if F != self.f_node:
            # fall back (shouldn't happen if batch contract is consistent)
            return x

        flat = x.reshape(-1, F)  # (M,F)
        cont = flat[:, self.cont_indices].float()

        embs = []
        for (name, idx, num_emb, emb_dim) in self.cat_items:
            ids = flat[:, idx]
            # safe cast to long
            if ids.dtype.is_floating_point:
                ids = ids.round()
            ids = ids.long()
            ids = torch.clamp(ids, 0, num_emb - 1)
            embs.append(self.embeddings[name](ids))

        out = torch.cat([cont] + embs, dim=-1)
        return out.reshape(*orig_shape[:-1], out.shape[-1])


# =========================================================
# Utilities / pooling
# =========================================================
def pick_temporal_seq(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, str]:
    """Sequence selection priority controlled by cfg.TEMPORAL_SEQ_PRIORITY."""
    priority = getattr(cfg, "TEMPORAL_SEQ_PRIORITY",
                       ("macro_seq", "x_seq", "extra_seq"))
    for k in priority:
        x = batch.get(k, None)
        if x is not None:
            return x, k
    raise KeyError("Temporal model requires one of: macro_seq / extra_seq / x_seq")


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float()
    denom = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=dim) / denom


def _safe_alive(alive: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if alive is None:
        return None
    with torch.inference_mode():
        s = float(alive.sum().item())
    if s <= 0.0:
        return None
    return alive


def pool_team_repr(node_emb: torch.Tensor, alive: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    node_emb: (B,N,D), N assumed 10 with [0..4]=blue, [5..9]=red
    return: (B, 3D) = [mean_blue, mean_red, mean_blue-mean_red]
    """
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


# =========================================================
# Adjacency builder (distance-based, soft/hard) for N=10
# =========================================================
def build_adjacency_from_xy(
        xy: torch.Tensor,  # (B, N, 2) normalized coords
        soft: bool = True,
        sigma: float = 0.125,
        team_edge_weight: float = 1.0,
        add_self_loops: bool = True,
        alive: Optional[torch.Tensor] = None,  # (B,N) 0/1 mask
        clamp_min: float = 1e-4,
) -> torch.Tensor:
    B, N, _ = xy.shape
    with _autocast_disabled():
        xy_f = _nan_to_num_(xy.float())
        diff = xy_f.unsqueeze(2) - xy_f.unsqueeze(1)  # (B,N,N,2)
        d2 = (diff * diff).sum(dim=-1)  # (B,N,N)

        if soft:
            s2 = max(1e-8, float(sigma) ** 2)
            A = torch.exp(-0.5 * d2 / s2)
        else:
            A = (d2.sqrt() <= float(sigma)).float()

        # same-team boost
        team = torch.zeros(N, device=A.device, dtype=torch.long)
        team[5:] = 1
        same_team = (team.unsqueeze(0).unsqueeze(2) == team.unsqueeze(0).unsqueeze(1)).to(A.dtype)  # (1,N,N)
        A = A * (1.0 + (float(team_edge_weight) - 1.0) * same_team)

        # alive mask
        if alive is not None:
            if alive.ndim == 1:
                alive = alive.unsqueeze(0)
            am = _nan_to_num_(alive.float()).clamp(0.0, 1.0).to(dtype=A.dtype)
            A = A * (am.unsqueeze(2) * am.unsqueeze(1))

        # self loops
        if add_self_loops:
            eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = torch.maximum(A, eye)

        if clamp_min is not None and float(clamp_min) > 0:
            A = A.clamp_min(float(clamp_min))

        out = A.to(dtype=xy.dtype)
    return out


def normalize_adj_gcn(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GCN normalization: D^-1/2 A D^-1/2"""
    squeeze_back = False
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze_back = True

    with _autocast_disabled():
        A32 = _nan_to_num_(A.float())
        if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
            A32 = A32.clamp_min(0.0)

        eps32 = float(getattr(cfg, "ADJ_NORM_EPS", eps))
        if eps32 <= 0:
            eps32 = 1e-6

        deg = A32.sum(dim=-1)
        deg = _nan_to_num_(deg).clamp_min(eps32)
        d_inv_sqrt = torch.rsqrt(deg)
        d_inv_sqrt = _nan_to_num_(d_inv_sqrt)

        A_norm = d_inv_sqrt.unsqueeze(-1) * A32 * d_inv_sqrt.unsqueeze(-2)
        A_norm = _nan_to_num_(A_norm)

    if squeeze_back:
        A_norm = A_norm.squeeze(0)
    return A_norm


# =========================================================
# MLP / Temporal encoders
# =========================================================
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0, layers: int = 2):
        super().__init__()
        assert layers >= 1
        if layers == 1:
            self.net = nn.Linear(d_in, d_out)
        else:
            mods = [nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(layers - 2):
                mods += [nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            mods += [nn.Linear(d_hidden, d_out)]
            self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type: str, d_in: int, d_h: int, n_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        rnn_dropout = float(dropout) if n_layers > 1 else 0.0
        if rnn_type == "gru":
            self.rnn = nn.GRU(d_in, d_h, num_layers=n_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=bidirectional)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(d_in, d_h, num_layers=n_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            raise ValueError(rnn_type)
        self.out_dim = d_h * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out[:, -1, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        return x + self.pe[:, :L, :].to(x.dtype)


class TransformerTemporalEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int, n_layers: int,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * int(getattr(cfg, "TRANS_FF_MULT", 4)),
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.enc(h)
        return h[:, -1, :]


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.chomp1 = _Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.chomp2 = _Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = self.down(x)
        return self.relu(out + res)


class TCNTemporalEncoder(nn.Module):
    """Input (B,L,D) -> causal conv over time -> return last hidden (B,C)."""

    def __init__(self, d_in: int, channels: int, n_levels: int = 3, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        chans = [channels] * max(1, int(n_levels))
        layers: List[nn.Module] = []
        in_ch = d_in
        for i, out_ch in enumerate(chans):
            dilation = 2 ** i
            layers.append(_TemporalBlock(in_ch, out_ch, kernel=kernel, dilation=dilation, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_dim = chans[-1] if chans else channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.transpose(1, 2)  # (B,D,L)
        h = self.net(x1)  # (B,C,L)
        return h[:, :, -1]


# =========================================================
# Edge features (fixed schema)
# =========================================================
def _make_same_team_mask(N: int, device, dtype) -> torch.Tensor:
    team = torch.zeros(N, device=device, dtype=torch.long)
    team[5:] = 1
    same = (team.unsqueeze(0).unsqueeze(2) == team.unsqueeze(0).unsqueeze(1)).to(dtype)  # (1,N,N)
    return same


# NOTE: schema length must stay consistent with features list below.
RICH_EDGE_SCHEMA_DIM = 30


def build_rich_edge_attr(
        x_t: torch.Tensor,  # (B,N,F)
        xy: torch.Tensor,  # (B,N,2)
        A: torch.Tensor,  # (B,N,N)
        prev_x_t: Optional[torch.Tensor] = None,  # (B,N,F)
) -> torch.Tensor:
    """
    Fixed-schema rich edge features (B,N,N,E).
    E = 30 (RICH_EDGE_SCHEMA_DIM).
    """
    with _autocast_disabled():
        B, N, Fnode = x_t.shape
        x32 = _nan_to_num_(x_t.float())
        xy32 = _nan_to_num_(xy.float())
        A32 = _nan_to_num_(A.float()).clamp_min(1e-8)

        # geometric
        diff = xy32.unsqueeze(2) - xy32.unsqueeze(1)  # (B,N,N,2) = i - j
        dx = diff[..., 0]
        dy = diff[..., 1]
        dist = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-12))
        inv = 1.0 / dist.clamp_min(1e-6)
        sin = dy * inv
        cos = dx * inv

        logA = torch.log(A32)  # <= 0 typically
        same_team = _make_same_team_mask(N, device=A32.device, dtype=A32.dtype).expand(B, -1, -1)

        def get_node(idx: Optional[int]) -> torch.Tensor:
            if idx is None or idx < 0 or idx >= Fnode:
                return torch.zeros((B, N), device=x32.device, dtype=x32.dtype)
            return x32[..., idx]

        hp = get_node(HP_IDX).clamp(0.0, 1.0)
        lvl = get_node(LVL_IDX)
        gold = get_node(GOLD_IDX)
        cc = get_node(CC_IDX)
        vis = get_node(VISION_IDX)
        flash = get_node(FLASH_IDX).clamp(0.0, 1.0)
        ult = get_node(ULT_IDX)

        # recent damage approx: delta of cumulative damage-to-champions
        if prev_x_t is not None and DMG_TOC_IDX is not None and 0 <= DMG_TOC_IDX < Fnode:
            prev32 = _nan_to_num_(prev_x_t.float())
            dmg_now = x32[..., DMG_TOC_IDX]
            dmg_prev = prev32[..., DMG_TOC_IDX]
            recent_dmg = F.relu(dmg_now - dmg_prev)  # (B,N)
        else:
            recent_dmg = torch.zeros((B, N), device=x32.device, dtype=x32.dtype)

        def pair(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ai = a.unsqueeze(2).expand(B, N, N)  # i
            aj = a.unsqueeze(1).expand(B, N, N)  # j
            d = ai - aj
            return ai, aj, d

        hp_i, hp_j, dhp = pair(hp)
        lvl_i, lvl_j, dlvl = pair(lvl)
        gold_i, gold_j, dgold = pair(gold)
        cc_i, cc_j, dcc = pair(cc)
        vis_i, vis_j, dvis = pair(vis)
        fl_i, fl_j, _ = pair(flash)
        ul_i, ul_j, _ = pair(ult)
        rd_i, rd_j, _ = pair(recent_dmg)

        flash_and = (fl_i * fl_j)  # both ready (soft)

        feats = [
            dx, dy, dist, sin, cos,
            A32, logA, same_team,
            hp_i, hp_j, dhp,
            lvl_i, lvl_j, dlvl,
            gold_i, gold_j, dgold,
            cc_i, cc_j, dcc,
            vis_i, vis_j, dvis,
            fl_i, fl_j, flash_and,
            ul_i, ul_j,
            rd_i, rd_j,
        ]
        e = torch.stack(feats, dim=-1)  # (B,N,N,E)
        e = _nan_to_num_(e)

        if e.shape[-1] != RICH_EDGE_SCHEMA_DIM:
            # do not silently mismatch
            raise RuntimeError(f"Edge schema dim mismatch: got {e.shape[-1]} != {RICH_EDGE_SCHEMA_DIM}")

    return e.to(dtype=x_t.dtype)


# =========================================================
# 1) Edge-aware Dynamic STGNN (ST-MPNN)
# =========================================================
class EdgeAwareMPNNLayer(nn.Module):
    """
    Edge-aware message passing:
      m_ij = phi([h_i, h_j, e_ij])
      w_ij = A_ij * sigmoid(g(e_ij))
      agg_i = sum_j w_ij * m_ij / deg_i
      h_i' = LN(ReLU(W_self h_i + agg_i)) with residual
    Returns optional edge weights for interpretability.
    """

    def __init__(self, d_in: int, d_out: int, edge_dim: int, dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        hidden = int(getattr(cfg, "MPNN_HIDDEN", max(128, d_out)))
        self.edge_dim = int(edge_dim)
        self.phi = nn.Sequential(
            nn.Linear(d_in * 2 + self.edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )
        gate_h = int(getattr(cfg, "EDGE_GATE_H", 64))
        self.gate = nn.Sequential(
            nn.Linear(self.edge_dim, gate_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_h, 1),
        )

        self.self = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None

    def forward(self, h: torch.Tensor, A: torch.Tensor, e: torch.Tensor, return_aux: bool = False):
        B, N, D = h.shape
        with _autocast_disabled():
            h32 = _nan_to_num_(h.float())
            A32 = _nan_to_num_(A.float()).clamp_min(0.0)
            e32 = _nan_to_num_(e.float())

            hi = h32.unsqueeze(2).expand(B, N, N, D)
            hj = h32.unsqueeze(1).expand(B, N, N, D)
            inp = torch.cat([hi, hj, e32], dim=-1)  # (B,N,N,2D+E)
            m = self.phi(inp)  # (B,N,N,d_out)
            m = _nan_to_num_(m)

            gate = torch.sigmoid(self.gate(e32))  # (B,N,N,1)
            gate = _nan_to_num_(gate)

            w = A32.unsqueeze(-1) * gate  # (B,N,N,1)
            m = m * w

            deg_eps = float(getattr(cfg, "MPNN_DEG_EPS", 1e-6))
            deg = A32.sum(dim=-1).clamp_min(deg_eps)  # (B,N)
            agg = m.sum(dim=2) / deg.unsqueeze(-1)  # (B,N,d_out)
            agg = _nan_to_num_(agg)

            out = F.relu(self.self(h32) + agg)
            out = _nan_to_num_(out)

        out = self.drop(out.to(dtype=h.dtype))
        if out.shape[-1] == h.shape[-1]:
            out = out + h
        if self.norm is not None:
            out = self.norm(out)

        if return_aux:
            return out, {"edge_gate": gate.detach(), "edge_weight": (A32.unsqueeze(-1) * gate).detach()}
        return out


class EdgeAwareDynamicSTGNN(nn.Module):
    """
    node_seq (B,L,N,F) -> per-t MPNN on dynamic graph (A_t from xy) with rich edge_attr
    -> team pooling per t -> temporal encoder -> head
    """

    def __init__(self, f_node: int, d_seq: int, use_lgbm_logit: bool = True):
        super().__init__()
        self.use_lgbm_logit = bool(use_lgbm_logit)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.2))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        # Ã¢Å“â€¦ categorical-safe adapter
        self.adapter = NodeFeatureAdapter(f_node=f_node, node_idx=NODE_IDX)
        f_in = int(self.adapter.out_dim) if getattr(self.adapter, "out_dim", f_node) else int(f_node)

        self.node_proj = nn.Linear(f_in, gnn_dim)

        self.edge_dim = int(getattr(cfg, "EDGE_RICH_DIM", RICH_EDGE_SCHEMA_DIM))
        if self.edge_dim != RICH_EDGE_SCHEMA_DIM:
            # enforce schema correctness (do not silently mismatch)
            raise ValueError(f"EDGE_RICH_DIM must be {RICH_EDGE_SCHEMA_DIM} for current schema, got {self.edge_dim}")

        n_layers = int(getattr(cfg, "MPNN_LAYERS", 2))
        layers: List[nn.Module] = []
        din = gnn_dim
        for _ in range(max(1, n_layers)):
            layers.append(EdgeAwareMPNNLayer(d_in=din, d_out=gnn_dim, edge_dim=self.edge_dim,
                                             dropout=gnn_drop, use_norm=gnn_norm))
            din = gnn_dim
        self.mpnn_layers = nn.ModuleList(layers)

        drop = float(getattr(cfg, "DROPOUT", 0.2))
        temp_kind = str(getattr(cfg, "EDGE_STGNN_TEMPORAL", "bigru")).lower()  # bigru | tcn | transformer
        in_dim = 3 * gnn_dim

        if temp_kind in ("bigru", "gru"):
            rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
            rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
            self.temporal = RNNEncoder("gru", d_in=in_dim, d_h=rnn_hidden, n_layers=rnn_layers,
                                       bidirectional=True, dropout=drop)
            temp_out = self.temporal.out_dim
        elif temp_kind == "tcn":
            ch = int(getattr(cfg, "TCN_CHANNELS", int(getattr(cfg, "RNN_HIDDEN", 128))))
            levels = int(getattr(cfg, "TCN_LEVELS", 3))
            kernel = int(getattr(cfg, "TCN_KERNEL", 3))
            tdrop = float(getattr(cfg, "TCN_DROPOUT", drop))
            self.temporal = TCNTemporalEncoder(d_in=in_dim, channels=ch, n_levels=levels, kernel=kernel, dropout=tdrop)
            temp_out = self.temporal.out_dim
        elif temp_kind == "transformer":
            self.temporal = TransformerTemporalEncoder(
                d_in=in_dim,
                d_model=int(getattr(cfg, "TRANS_D_MODEL", 256)),
                nhead=int(getattr(cfg, "TRANS_NHEAD", 4)),
                n_layers=int(getattr(cfg, "TRANS_LAYERS", 2)),
                dropout=float(getattr(cfg, "TRANS_DROPOUT", 0.2)),
                max_len=int(getattr(cfg, "TRANS_MAX_LEN", 512)),
            )
            temp_out = self.temporal.out_dim
        else:
            raise ValueError(f"Unknown EDGE_STGNN_TEMPORAL={temp_kind}")

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", 128))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        head_in = temp_out + (1 if self.use_lgbm_logit else 0)
        self.head = MLP(head_in, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("EdgeAwareDynamicSTGNN requires node_seq (B,L,N,F)")
        B, L, N, Fnode = node_seq.shape

        pooled_list = []
        aux_edges = [] if return_aux else None

        prev_x_t = None
        for t in range(L):
            x_t = node_seq[:, t, :, :]  # (B,N,F)
            xy = x_t[:, :, [X_IDX, Y_IDX]]  # (B,N,2)

            alive = None
            if bool(getattr(cfg, "USE_ALIVE_MASK", True)) and (ALIVE_IDX is not None) and (ALIVE_IDX < x_t.shape[-1]):
                alive = (x_t[:, :, ALIVE_IDX] > 0.5).to(dtype=x_t.dtype)
                alive = _safe_alive(alive)

            x_eff = x_t * alive.unsqueeze(-1) if alive is not None else x_t

            A = build_adjacency_from_xy(
                xy,
                soft=bool(getattr(cfg, "ADJ_SOFT", True)),
                sigma=float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125)),
                team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
                add_self_loops=True,
                alive=alive,
                clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
            )

            e = build_rich_edge_attr(x_eff, xy, A, prev_x_t=prev_x_t)  # (B,N,N,30)

            # Ã¢Å“â€¦ categorical-safe projection
            x_proj_in = self.adapter(x_eff)
            h = F.relu(self.node_proj(x_proj_in))

            last_layer_aux = None
            for layer in self.mpnn_layers:
                if return_aux:
                    h, last_layer_aux = layer(h, A, e, return_aux=True)
                else:
                    h = layer(h, A, e, return_aux=False)

            pooled = pool_team_repr(h, alive=alive)  # (B,3D)
            pooled_list.append(pooled)

            if return_aux and last_layer_aux is not None:
                aux_edges.append(last_layer_aux)

            prev_x_t = x_eff.detach()

        seq = torch.stack(pooled_list, dim=1)  # (B,L,3D)
        temp_feat = self.temporal(seq)

        if self.use_lgbm_logit and (batch.get("lgbm_logit", None) is not None):
            llog = batch["lgbm_logit"]
            if llog.ndim == 1:
                llog = llog.unsqueeze(-1)
            feat = torch.cat([temp_feat, llog], dim=-1)
        else:
            feat = temp_feat

        logit = self.head(feat)

        if return_aux:
            return logit, {
                "pooled_seq": seq.detach(),
                "temporal_feat": temp_feat.detach(),
                "edge_aux_last_each_t": aux_edges,
            }
        return logit


# =========================================================
# 2) STGCN (TemporalConv <-> GraphConv(A_t))
# =========================================================
class CausalTemporalConv(nn.Module):
    """Conv2d over time dimension with causal left padding. x: (B,C,N,L)"""

    def __init__(self, c_in: int, c_out: int, k_t: int = 3, dropout: float = 0.1):
        super().__init__()
        self.k_t = int(k_t)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, self.k_t), padding=(0, 0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.k_t - 1
        if pad > 0:
            x = F.pad(x, (pad, 0, 0, 0))
        y = self.conv(x)
        y = F.relu(y)
        y = self.drop(y)
        return y


class DynamicGraphConv(nn.Module):
    """GraphConv per time step: y_t = W (A_norm_t X_t). Input x: (B,C,N,L)"""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(c_out) if use_norm else None

    def forward(self, x: torch.Tensor, A_seq: torch.Tensor) -> torch.Tensor:
        B, C, N, L = x.shape
        out_list = []
        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())
            A32 = _nan_to_num_(A_seq.float())

            for t in range(L):
                Xt = x32[:, :, :, t].transpose(1, 2)  # (B,N,C)
                At = normalize_adj_gcn(A32[:, t, :, :])  # (B,N,N)
                Mt = torch.bmm(At, Xt)  # (B,N,C)
                Yt = self.lin(Mt)  # (B,N,Cout)
                out_list.append(Yt)

            Y = torch.stack(out_list, dim=-1)  # (B,N,Cout,L)

        Y = Y.to(dtype=x.dtype).transpose(1, 2)  # (B,Cout,N,L)
        Y = self.drop(Y)
        if self.norm is not None:
            Yn = Y.permute(0, 2, 3, 1).contiguous()  # (B,N,L,C)
            Yn = self.norm(Yn)
            Y = Yn.permute(0, 3, 1, 2).contiguous()
        return Y


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_t: int = 3, dropout: float = 0.1):
        super().__init__()
        self.t1 = CausalTemporalConv(c_in, c_out, k_t=k_t, dropout=dropout)
        self.g = DynamicGraphConv(c_out, c_out, dropout=dropout, use_norm=True)
        self.t2 = CausalTemporalConv(c_out, c_out, k_t=k_t, dropout=dropout)
        self.res = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_seq: torch.Tensor) -> torch.Tensor:
        h = self.t1(x)
        h = self.g(h, A_seq)
        h = self.t2(h)
        r = self.res(x)
        y = F.relu(h + r)
        y = self.drop(y)
        return y


class STGCNModel(nn.Module):
    """
    node_seq -> build A_t from xy each t -> STGCN blocks -> take last time -> team pooling -> head
    """

    def __init__(self, f_node: int, d_seq: int, use_lgbm_logit: bool = True):
        super().__init__()
        self.use_lgbm_logit = bool(use_lgbm_logit)

        drop = float(getattr(cfg, "DROPOUT", 0.2))
        c0 = int(getattr(cfg, "STGCN_C0", 64))
        c1 = int(getattr(cfg, "STGCN_C1", 64))
        blocks = int(getattr(cfg, "STGCN_BLOCKS", 2))
        k_t = int(getattr(cfg, "STGCN_KT", 3))

        # Ã¢Å“â€¦ categorical-safe adapter
        self.adapter = NodeFeatureAdapter(f_node=f_node, node_idx=NODE_IDX)
        f_in = int(self.adapter.out_dim) if getattr(self.adapter, "out_dim", f_node) else int(f_node)

        self.in_proj = nn.Linear(f_in, c0)

        layers: List[nn.Module] = []
        cin = c0
        for i in range(max(1, blocks)):
            cout = c1 if i > 0 else c0
            layers.append(STGCNBlock(cin, cout, k_t=k_t, dropout=drop))
            cin = cout
        self.blocks = nn.ModuleList(layers)

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", 128))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        head_in = 3 * cin + (1 if self.use_lgbm_logit else 0)
        self.head = MLP(head_in, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("STGCNModel requires node_seq (B,L,N,F)")
        B, L, N, Fnode = node_seq.shape

        A_list = []
        alive_seq = [] if return_aux else None

        for t in range(L):
            x_t = node_seq[:, t, :, :]
            xy = x_t[:, :, [X_IDX, Y_IDX]]

            alive = None
            if bool(getattr(cfg, "USE_ALIVE_MASK", True)) and (ALIVE_IDX is not None) and (ALIVE_IDX < x_t.shape[-1]):
                alive = (x_t[:, :, ALIVE_IDX] > 0.5).to(dtype=x_t.dtype)
                alive = _safe_alive(alive)

            if alive_seq is not None:
                alive_seq.append(alive.detach() if alive is not None else None)

            A_t = build_adjacency_from_xy(
                xy,
                soft=bool(getattr(cfg, "ADJ_SOFT", True)),
                sigma=float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125)),
                team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
                add_self_loops=True,
                alive=alive,
                clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
            )
            A_list.append(A_t)

        A_seq = torch.stack(A_list, dim=1)  # (B,L,N,N)

        # Ã¢Å“â€¦ categorical-safe projection
        x_in = self.adapter(node_seq)  # (B,L,N,F')
        x = F.relu(self.in_proj(x_in))  # (B,L,N,C)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,C,N,L)

        for blk in self.blocks:
            x = blk(x, A_seq)

        x_last = x[:, :, :, -1].transpose(1, 2)  # (B,N,C)

        alive_last = None
        if bool(getattr(cfg, "USE_ALIVE_MASK", True)) and (ALIVE_IDX is not None):
            a = node_seq[:, -1, :, ALIVE_IDX] if (ALIVE_IDX < Fnode) else None
            if a is not None:
                alive_last = _safe_alive((a > 0.5).to(dtype=x_last.dtype))

        pooled = pool_team_repr(x_last, alive=alive_last)  # (B,3C)

        if self.use_lgbm_logit and (batch.get("lgbm_logit", None) is not None):
            llog = batch["lgbm_logit"]
            if llog.ndim == 1:
                llog = llog.unsqueeze(-1)
            feat = torch.cat([pooled, llog], dim=-1)
        else:
            feat = pooled

        logit = self.head(feat)

        if return_aux:
            return logit, {"A_seq": A_seq.detach(), "alive_seq": alive_seq, "x_last": x_last.detach()}
        return logit


# =========================================================
# 3) Multi-scale Temporal + Dynamic Graph
# =========================================================
class SimpleGraphEmbed(nn.Module):
    """Per timestep graph embedding: project nodes -> one-step message via A -> pool_team_repr."""

    def __init__(self, f_node: int, gnn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Ã¢Å“â€¦ categorical-safe adapter
        self.adapter = NodeFeatureAdapter(f_node=f_node, node_idx=NODE_IDX)
        f_in = int(self.adapter.out_dim) if getattr(self.adapter, "out_dim", f_node) else int(f_node)

        self.proj = nn.Linear(f_in, gnn_dim)
        self.msg = nn.Linear(gnn_dim, gnn_dim, bias=False)
        self.self = nn.Linear(gnn_dim, gnn_dim, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(gnn_dim)

    def forward(self, x_t: torch.Tensor, A_t: torch.Tensor, alive: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_in = self.adapter(x_t)  # (B,N,F')
        h0 = F.relu(self.proj(x_in))  # (B,N,D)
        with _autocast_disabled():
            A_norm = normalize_adj_gcn(A_t)
            m = torch.bmm(A_norm, _nan_to_num_(h0.float()))
            h = self.msg(m) + self.self(_nan_to_num_(h0.float()))
            h = F.gelu(h)
        h = self.drop(h.to(dtype=x_t.dtype))
        h = self.norm(h)
        return pool_team_repr(h, alive=alive)  # (B,3D)


class MultiScaleDynGraph(nn.Module):
    """
    node_seq -> per-t graph pooled vector g_t -> multi-stride temporal encoders -> fusion -> head
    """

    def __init__(self, f_node: int, d_seq: int, use_lgbm_logit: bool = True):
        super().__init__()
        self.use_lgbm_logit = bool(use_lgbm_logit)

        drop = float(getattr(cfg, "DROPOUT", 0.2))
        gnn_dim = int(getattr(cfg, "MS_GNN_DIM", int(getattr(cfg, "GNN_DIM", 64))))
        self.graph_embed = SimpleGraphEmbed(f_node, gnn_dim=gnn_dim, dropout=drop)

        self.strides = list(getattr(cfg, "MS_STRIDES", [1, 2, 5]))
        base_dim = 3 * gnn_dim

        kind = str(getattr(cfg, "MS_TEMPORAL", "tcn")).lower()  # tcn | transformer | bigru
        self.encoders = nn.ModuleList()
        out_dims: List[int] = []

        for _ in self.strides:
            if kind == "tcn":
                ch = int(getattr(cfg, "MS_TCN_CHANNELS", 128))
                levels = int(getattr(cfg, "MS_TCN_LEVELS", 3))
                kernel = int(getattr(cfg, "MS_TCN_KERNEL", 3))
                tdrop = float(getattr(cfg, "MS_TCN_DROPOUT", drop))
                enc = TCNTemporalEncoder(d_in=base_dim, channels=ch, n_levels=levels, kernel=kernel, dropout=tdrop)
                out_dims.append(enc.out_dim)
                self.encoders.append(enc)
            elif kind == "transformer":
                enc = TransformerTemporalEncoder(
                    d_in=base_dim,
                    d_model=int(getattr(cfg, "MS_TRANS_D_MODEL", 256)),
                    nhead=int(getattr(cfg, "MS_TRANS_NHEAD", 4)),
                    n_layers=int(getattr(cfg, "MS_TRANS_LAYERS", 2)),
                    dropout=float(getattr(cfg, "MS_TRANS_DROPOUT", 0.2)),
                    max_len=int(getattr(cfg, "TRANS_MAX_LEN", 512)),
                )
                out_dims.append(enc.out_dim)
                self.encoders.append(enc)
            elif kind == "bigru":
                h = int(getattr(cfg, "MS_RNN_HIDDEN", 128))
                layers = int(getattr(cfg, "MS_RNN_LAYERS", 1))
                enc = RNNEncoder("gru", d_in=base_dim, d_h=h, n_layers=layers, bidirectional=True, dropout=drop)
                out_dims.append(enc.out_dim)
                self.encoders.append(enc)
            else:
                raise ValueError(f"Unknown MS_TEMPORAL={kind}")

        fuse_in = sum(out_dims) + (1 if self.use_lgbm_logit else 0)
        fuse_h = int(getattr(cfg, "MS_FUSE_H", 128))
        self.fuse = MLP(fuse_in, fuse_h, fuse_h, dropout=drop, layers=2)

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", 128))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(fuse_h, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("MultiScaleDynGraph requires node_seq")
        B, L, N, Fnode = node_seq.shape

        g_list = []
        for t in range(L):
            x_t = node_seq[:, t, :, :]
            xy = x_t[:, :, [X_IDX, Y_IDX]]

            alive = None
            if bool(getattr(cfg, "USE_ALIVE_MASK", True)) and (ALIVE_IDX is not None) and (ALIVE_IDX < x_t.shape[-1]):
                alive = (x_t[:, :, ALIVE_IDX] > 0.5).to(dtype=x_t.dtype)
                alive = _safe_alive(alive)

            A_t = build_adjacency_from_xy(
                xy,
                soft=bool(getattr(cfg, "ADJ_SOFT", True)),
                sigma=float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125)),
                team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
                add_self_loops=True,
                alive=alive,
                clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
            )
            g_t = self.graph_embed(x_t, A_t, alive=alive)  # (B,3D)
            g_list.append(g_t)

        g = torch.stack(g_list, dim=1)  # (B,L,3D)

        feats = []
        for stride, enc in zip(self.strides, self.encoders):
            if stride <= 1:
                gs = g
            else:
                gs = g[:, ::stride, :]
                if gs.shape[1] < 2:
                    gs = g[:, max(0, L - 2):, :]
            feats.append(enc(gs))

        fused = torch.cat(feats, dim=-1)

        if self.use_lgbm_logit and (batch.get("lgbm_logit", None) is not None):
            llog = batch["lgbm_logit"]
            if llog.ndim == 1:
                llog = llog.unsqueeze(-1)
            fused = torch.cat([fused, llog], dim=-1)

        z = self.fuse(fused)
        logit = self.head(z)

        if return_aux:
            return logit, {"g_seq": g.detach(), "feats": [f.detach() for f in feats], "strides": list(self.strides)}
        return logit


# =========================================================
# Trainer (kept for compatibility with your experiment.py)
# =========================================================
def train_deep_model(
        model_name: str,
        feature_set: str,
        variant_tag: str,
        tr_refs: List[Any],
        va_refs: List[Any],
        te_refs: List[Any],
        seed: int,
        device: torch.device,
        out_dir: Path,
        log_fp: Path,
        lgbm_logit_map: Optional[Dict[str, float]] = None,
        return_pred_maps: bool = True,
        prebuilt_datasets: Optional[Tuple[Any, Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Updated trainer compatible with current experiment.py
    """
    # Suppress repeated "triton not found" warnings from torch internals
    warnings.filterwarnings(
        "ignore", message=".*triton.*", category=UserWarning,
    )

    ensure_dir(out_dir)
    set_seed(int(seed))
    try:
        from train.models import reset_model_singletons
        reset_model_singletons()
    except Exception as e:
        write_log(f"[DEEP][WARN] reset_model_singletons failed (ignored): {e}", log_fp)

    # -----------------------------------------------------
    # Speed knobs (safe defaults)
    # -----------------------------------------------------
    if device.type == "cuda":
        # TF32 speeds up matmul/conv on Ampere+ with minimal accuracy impact.
        if bool(getattr(cfg, "TF32", True)):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        if bool(getattr(cfg, "CUDNN_BENCHMARK", True)):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        # Improve matmul perf where supported (PyTorch 2)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if getattr(cfg, "REQUIRE_CUDA", False) and (device.type != "cuda"):
        write_log("[DEEP] REQUIRE_CUDA=True but cuda not available -> skip", log_fp)
        rep = {"ok": False, "reason": "cuda_not_available", "model_name": model_name, "variant": variant_tag}
        save_json(out_dir / "report.json", rep)
        return rep

    # -----------------------------------------------------
    # Dataset caching policy
    # -----------------------------------------------------
    cache_train = bool(getattr(cfg, "CACHE_TRAIN_SAMPLES_IN_RAM", getattr(cfg, "CACHE_IN_RAM", False)))
    cache_eval = bool(getattr(cfg, "CACHE_EVAL_SAMPLES_IN_RAM", True))

    if prebuilt_datasets is not None:
        ds_tr, ds_va, ds_te = prebuilt_datasets
        write_log(
            f"[DEEP] Using prebuilt shared datasets: "
            f"train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}",
            log_fp,
        )
    else:
        # [P0-6] Val/Test subsampling — reduces I/O from ~10h to ~3h.
        # AUC SE at n=80K: ~0.0035 (vs 0.0017 at full size), acceptable.
        val_max_n = int(getattr(cfg, "VAL_MAX_N", 0))
        test_max_n = int(getattr(cfg, "TEST_MAX_N", 0))
        _va_refs = list(va_refs)
        _te_refs = list(te_refs)
        if val_max_n > 0 and len(_va_refs) > val_max_n:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(_va_refs), val_max_n, replace=False)
            _va_refs = [_va_refs[i] for i in sorted(idx)]
            write_log(
                f"[DEEP-SUBSAMPLE] val {len(va_refs)} -> {len(_va_refs)} "
                f"(max_n={val_max_n}, seed={seed})",
                log_fp,
            )
        if test_max_n > 0 and len(_te_refs) > test_max_n:
            rng = np.random.RandomState(seed + 1000)
            idx = rng.choice(len(_te_refs), test_max_n, replace=False)
            _te_refs = [_te_refs[i] for i in sorted(idx)]
            write_log(
                f"[DEEP-SUBSAMPLE] test {len(te_refs)} -> {len(_te_refs)} "
                f"(max_n={test_max_n}, seed={seed})",
                log_fp,
            )

        ds_tr = InMemoryFightDataset(
            tr_refs,
            feature_set=feature_set,
            model_name=model_name,
            lgbm_logit_map=lgbm_logit_map,
            cache_in_ram=cache_train,
            force_emit_logits=bool(getattr(cfg, "FORCE_EMIT_LOGITS", False)),
            split_label="train",
        )
        ds_va = InMemoryFightDataset(
            _va_refs,
            feature_set=feature_set,
            model_name=model_name,
            lgbm_logit_map=lgbm_logit_map,
            cache_in_ram=cache_eval,
            force_emit_logits=bool(getattr(cfg, "FORCE_EMIT_LOGITS", False)),
            split_label="val",
        )
        ds_te = InMemoryFightDataset(
            _te_refs,
            feature_set=feature_set,
            model_name=model_name,
            lgbm_logit_map=lgbm_logit_map,
            cache_in_ram=cache_eval,
            force_emit_logits=bool(getattr(cfg, "FORCE_EMIT_LOGITS", False)),
            split_label="test",
        )

    if len(ds_tr) < int(getattr(cfg, "MIN_TRAIN_SAMPLES", 200)):
        write_log(f"[DEEP] Not enough train samples: N={len(ds_tr)}", log_fp)
        rep = {"ok": False, "reason": "not_enough_train", "N": len(ds_tr)}
        save_json(out_dir / "report.json", rep)
        return rep

    bs = int(getattr(cfg, "BATCH_SIZE", 256))

    # Worker/prefetch policy
    num_workers_tr = int(getattr(cfg, "NUM_WORKERS", 0))
    num_workers_eval = int(getattr(cfg, "EVAL_NUM_WORKERS", num_workers_tr))
    pin_memory = bool(getattr(cfg, "PIN_MEMORY", True))
    prefetch_factor = int(getattr(cfg, "PREFETCH_FACTOR", 2))

    # If the dataset is already in RAM, extra workers usually slow things down (and may duplicate memory).
    if cache_train:
        num_workers_tr = 0
    if cache_eval:
        num_workers_eval = 0

    persistent_workers_tr = bool(getattr(cfg, "PERSISTENT_WORKERS", True)) and (num_workers_tr > 0)
    persistent_workers_eval = bool(getattr(cfg, "PERSISTENT_WORKERS", True)) and (num_workers_eval > 0)

    dl_tr_kwargs = dict(
        batch_size=bs,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=num_workers_tr,
        persistent_workers=persistent_workers_tr,
    )
    if num_workers_tr > 0:
        dl_tr_kwargs["prefetch_factor"] = prefetch_factor

    dl_ev_kwargs = dict(
        batch_size=bs,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=num_workers_eval,
        persistent_workers=persistent_workers_eval,
    )
    if num_workers_eval > 0:
        dl_ev_kwargs["prefetch_factor"] = prefetch_factor

    ld_tr = DataLoader(ds_tr, shuffle=True, **dl_tr_kwargs)
    ld_va = DataLoader(ds_va, shuffle=False, **dl_ev_kwargs)
    ld_te = DataLoader(ds_te, shuffle=False, **dl_ev_kwargs)
    ld_tr_eval = DataLoader(ds_tr, shuffle=False, **dl_ev_kwargs)

    b0 = next(iter(ld_tr), None)
    if b0 is None:
        write_log("[DEEP] first batch is None -> skip", log_fp)
        rep = {"ok": False, "reason": "first_batch_none"}
        save_json(out_dir / "report.json", rep)
        return rep

    write_log(f"[DEEP] first_batch_keys={sorted(list(b0.keys()))}", log_fp)

    f_node = int(b0["node_seq"].shape[-1]) if "node_seq" in b0 else -1
    _seq_priority = getattr(cfg, "TEMPORAL_SEQ_PRIORITY",
                            ("macro_seq", "x_seq", "extra_seq"))
    d_seq = -1
    seq_key_used = "none"
    for _sk in _seq_priority:
        if _sk in b0:
            d_seq = int(b0[_sk].shape[-1])
            seq_key_used = _sk
            break

    has_lgbm_logit = ("lgbm_logit" in b0) or ("lgbm_logit_seq" in b0)
    write_log(f"[DEEP] seq_key_used={seq_key_used} has_lgbm_logit_in_batch={has_lgbm_logit}", log_fp)

    model = build_model(
        model_name=model_name,
        f_node=f_node,
        d_seq=d_seq,
        use_lgbm_logit=bool(lgbm_logit_map),
    ).to(device)

    use_multi_task = bool(getattr(cfg, "USE_MULTI_TASK", False))
    aux_head: Optional[nn.Module] = None
    has_aux_targets = bool(
        ("y_gold_diff" in b0) or ("y_kill_diff" in b0) or ("y_obj_diff" in b0)
    )
    if use_multi_task and has_aux_targets:
        aux_head = _LogitAuxHead(hidden=int(getattr(cfg, "MTL_HEAD_HIDDEN", 16))).to(device)
        write_log("[DEEP] Multi-task enabled: logit-aux head (gold/kill/obj)", log_fp)
    elif use_multi_task and (not has_aux_targets):
        write_log("[DEEP][WARN] USE_MULTI_TASK=True but auxiliary targets are missing in batch.", log_fp)
        use_multi_task = False

    if bool(getattr(cfg, "TORCH_COMPILE", False)):
        try:
            if hasattr(torch, "compile"):
                compile_kwargs: Dict[str, Any] = {}
                compile_mode = str(getattr(cfg, "TORCH_COMPILE_MODE", "default") or "default").strip()
                if compile_mode:
                    compile_kwargs["mode"] = compile_mode
                if bool(getattr(cfg, "TORCH_COMPILE_DYNAMIC", False)):
                    compile_kwargs["dynamic"] = True
                # --- Triton availability check ---
                # The default "inductor" backend requires Triton for GPU
                # kernels.  Triton is not available on Windows and some
                # other environments.  Detect this *before* compilation so
                # we can fall back to the "eager" backend instead of
                # crashing at the first forward pass.
                _triton_ok = True
                if device.type == "cuda":
                    try:
                        import triton  # noqa: F401
                    except (ImportError, ModuleNotFoundError):
                        _triton_ok = False
                if not _triton_ok:
                    compile_kwargs["backend"] = "eager"
                    write_log(
                        "[DEEP][WARN] triton not found — falling back to "
                        "torch.compile(backend='eager'). Install triton for "
                        "full inductor performance.",
                        log_fp,
                    )
                model = torch.compile(model, **compile_kwargs)  # type: ignore
                if aux_head is not None and hasattr(torch, "compile"):
                    try:
                        aux_head = torch.compile(aux_head, **compile_kwargs)  # type: ignore
                    except Exception:
                        pass
                write_log(
                    f"[DEEP] torch.compile enabled mode={compile_kwargs.get('mode','default')} "
                    f"backend={compile_kwargs.get('backend','inductor')} "
                    f"dynamic={bool(compile_kwargs.get('dynamic', False))}",
                    log_fp,
                )
        except Exception as e:
            write_log(f"[DEEP] torch.compile failed (ignored): {e}", log_fp)

    use_amp = bool(getattr(cfg, "AMP", False)) and (device.type == "cuda")
    amp_dtype = _resolve_amp_dtype() if use_amp else torch.float32
    use_grad_scaler = bool(use_amp and (amp_dtype == torch.float16))
    # [P4-COMPAT] torch.cuda.amp.GradScaler deprecated in PyTorch 2.x
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    # Materialize lazy global modules before optimizer creation.
    if bool(getattr(cfg, "USE_ROLE_AWARE_ADJ", False)):
        try:
            model.eval()
            with torch.no_grad():
                _ = model(_to_device(b0, device))
        except Exception as e:
            write_log(f"[DEEP][WARN] role-aware warmup failed: {e}", log_fp)
        finally:
            model.train()

    lr = float(getattr(cfg, "LR", 1e-3))
    wd = float(getattr(cfg, "WEIGHT_DECAY", 1e-5))
    params = list(model.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())

    # T5: Role-Aware Adjacency — global module의 파라미터도 optimizer에 등록
    if bool(getattr(cfg, "USE_ROLE_AWARE_ADJ", False)):
        try:
            from train.models import _role_adj_module
            if _role_adj_module is not None:
                params += list(_role_adj_module.parameters())
                write_log(f"[DEEP] Added RoleAwareAdjacency R(5×5) to optimizer", log_fp)
            else:
                write_log("[DEEP][WARN] USE_ROLE_AWARE_ADJ=True but role module not materialized.", log_fp)
        except Exception as e:
            write_log(f"[DEEP][WARN] Failed to attach role-aware adjacency params: {e}", log_fp)

    # optimizer does not need duplicated parameter references
    uniq_params = []
    seen_param_ids = set()
    for p in params:
        pid = id(p)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        uniq_params.append(p)

    opt_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": wd}
    fused_requested = bool(getattr(cfg, "USE_FUSED_ADAMW", True))
    fused_used = False
    if (device.type == "cuda") and fused_requested:
        opt_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(uniq_params, **opt_kwargs)
        fused_used = bool(opt_kwargs.get("fused", False))
    except TypeError:
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(uniq_params, **opt_kwargs)
    clip_params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    write_log(
        f"[DEEP] optimizer=AdamW fused={fused_used} requested={fused_requested}",
        log_fp,
    )

    # ------------------------------------------------------------------
    # [FIX P0-2] Learning Rate Scheduler: Cosine Annealing with Linear Warm-up
    #
    # η(t) = η_max · (t / T_warm)                                 for t ≤ T_warm
    # η(t) = η_min + (η_max - η_min)/2 · (1 + cos(π·(t-T_warm)/(T_max-T_warm)))
    #                                                              for t > T_warm
    #
    # where T_warm = ceil(0.1 × T_max)
    # ------------------------------------------------------------------
    max_epochs = int(getattr(cfg, "EPOCHS", 10))
    # [P1-7] Warmup epochs from config (was hardcoded as ceil(0.1 * EPOCHS))
    warmup_epochs = int(getattr(cfg, "WARMUP_EPOCHS", max(1, int(math.ceil(0.1 * max_epochs)))))
    scheduler = None
    try:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_sched = LinearLR(
            opt,
            start_factor=1.0 / max(warmup_epochs, 1),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            opt,
            T_max=max(max_epochs - warmup_epochs, 1),
            eta_min=lr * 0.01,  # η_min = 1% of η_max
        )
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        write_log(
            f"[DEEP] LR scheduler: LinearWarmup({warmup_epochs}ep) + CosineAnnealing({max_epochs - warmup_epochs}ep)",
            log_fp,
        )
    except Exception as e:
        write_log(f"[DEEP] LR scheduler setup failed (using constant LR): {e}", log_fp)
        scheduler = None

    # ─── 기존 코드 ───
    # pos_w = float(getattr(cfg, "POS_WEIGHT", 1.0))
    # crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    # ─── 변경 후 ───
    label_eps = float(getattr(cfg, "LABEL_SMOOTHING", 0.0))

    if getattr(cfg, "USE_FOCAL_LOSS", False):
        from core.improvements import FocalLoss
        crit = FocalLoss(
            gamma=float(getattr(cfg, "FOCAL_GAMMA", 2.0)),
            alpha=float(getattr(cfg, "FOCAL_ALPHA", 0.25)),
        )
        write_log(f"[DEEP] Using FocalLoss(γ={cfg.FOCAL_GAMMA})", log_fp)
    else:
        pos_w = float(getattr(cfg, "POS_WEIGHT", 1.0))
        crit = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w], device=device)
        )

    patience = int(getattr(cfg, "PATIENCE", 3))
    # [P4-NAME] Fixed: was cfg.CLIP_NORM, but config.py defines GRAD_CLIP_NORM
    clip_norm = float(getattr(cfg, "GRAD_CLIP_NORM", 5.0))
    threshold = float(getattr(cfg, "CLS_THRESHOLD", 0.5))

    best_auc = -1.0
    best_state = None
    best_aux_state = None
    best_epoch = 0
    bad = 0
    train_start_time = time.time()

    write_log(
        f"[DEEP] Train {model_name} (fs={feature_set}, variant={variant_tag}) "
        f"Ntr={len(ds_tr)} Nva={len(ds_va)} AMP={use_amp} amp_dtype={str(amp_dtype).replace('torch.', '')} "
        f"grad_scaler={use_grad_scaler} device={device.type}",
        log_fp,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        t0 = time.time()
        loss_sum = 0.0
        loss_fight_sum = 0.0
        loss_gold_sum = 0.0
        loss_kill_sum = 0.0
        loss_obj_sum = 0.0
        grad_norm_sum = 0.0  # [P1-6] Track gradient norm for diagnostics
        n_step = 0

        for batch in ld_tr:
            if batch is None:
                continue
            batch = _to_device(batch, device)
            y = batch.get("y", None)
            if y is None:
                continue

            opt.zero_grad(set_to_none=True)

            with _autocast_ctx(use_amp, device):
                logit = _extract_main_logit(model(batch))
                y_f = y.float().view_as(logit)
                if label_eps > 0:
                    y_f = y_f * (1.0 - label_eps) + label_eps * 0.5  # T7
                loss_fight = crit(logit, y_f)
                loss = loss_fight

                loss_gold = torch.tensor(0.0, device=logit.device)
                loss_kill = torch.tensor(0.0, device=logit.device)
                loss_obj = torch.tensor(0.0, device=logit.device)
                if aux_head is not None:
                    aux_pred = aux_head(logit)
                    if "y_gold_diff" in batch:
                        tgt_gold = batch["y_gold_diff"].float().view_as(aux_pred["gold_pred"])
                        loss_gold = F.mse_loss(aux_pred["gold_pred"], tgt_gold)
                    if "y_kill_diff" in batch:
                        tgt_kill = batch["y_kill_diff"].float().view_as(aux_pred["kill_pred"])
                        loss_kill = F.mse_loss(aux_pred["kill_pred"], tgt_kill)
                    if "y_obj_diff" in batch:
                        tgt_obj = batch["y_obj_diff"].float().view_as(aux_pred["obj_pred"])
                        loss_obj = F.mse_loss(aux_pred["obj_pred"], tgt_obj)

                    lam_gold = float(getattr(cfg, "MTL_LAMBDA_GOLD", 0.1))
                    lam_kill = float(getattr(cfg, "MTL_LAMBDA_KILL", 0.05))
                    lam_obj = float(getattr(cfg, "MTL_LAMBDA_OBJ", 0.05))
                    loss = loss + lam_gold * loss_gold + lam_kill * loss_kill + lam_obj * loss_obj

            # [P1-6] Gradient computation + norm logging
            grad_norm_val = 0.0
            if use_grad_scaler:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                if clip_norm > 0:
                    grad_norm_val = float(torch.nn.utils.clip_grad_norm_(clip_params, clip_norm).item())
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                loss.backward()
                if clip_norm > 0:
                    grad_norm_val = float(torch.nn.utils.clip_grad_norm_(clip_params, clip_norm).item())
                opt.step()

            loss_sum += float(loss.detach().cpu().item())
            loss_fight_sum += float(loss_fight.detach().cpu().item())
            loss_gold_sum += float(loss_gold.detach().cpu().item())
            loss_kill_sum += float(loss_kill.detach().cpu().item())
            loss_obj_sum += float(loss_obj.detach().cpu().item())
            grad_norm_sum += grad_norm_val
            n_step += 1

            log_every = int(getattr(cfg, "LOG_EVERY", 0))
            if log_every > 0 and (n_step % log_every == 0):
                write_log(
                    f"[DEEP] epoch={epoch} step={n_step} loss={loss_sum / max(1, n_step):.4f} "
                    f"grad_norm={grad_norm_sum / max(1, n_step):.4f}",
                    log_fp,
                )

        met_va = _eval_loop(model, ld_va, device, use_amp=use_amp, threshold=threshold)
        auc_va = float(met_va.get("auc", -1.0))

        # [FIX P0-2] Step the LR scheduler
        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        avg_grad_norm = grad_norm_sum / max(1, n_step)
        write_log(
            f"[DEEP] epoch={epoch} loss={loss_sum / max(1, n_step):.4f} "
            f"val_auc={auc_va:.4f} lr={current_lr:.2e} "
            f"grad_norm={avg_grad_norm:.4f} time={time.time() - t0:.1f}s",
            log_fp,
        )
        if aux_head is not None:
            write_log(
                f"[DEEP][MTL] epoch={epoch} fight={loss_fight_sum / max(1, n_step):.4f} "
                f"gold={loss_gold_sum / max(1, n_step):.4f} "
                f"kill={loss_kill_sum / max(1, n_step):.4f} "
                f"obj={loss_obj_sum / max(1, n_step):.4f}",
                log_fp,
            )

        if auc_va > best_auc:
            best_auc = auc_va
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if aux_head is not None:
                best_aux_state = {k: v.detach().cpu().clone() for k, v in aux_head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                write_log(f"[DEEP] Early stop: best_val_auc={best_auc:.4f}", log_fp)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if aux_head is not None and best_aux_state is not None:
        aux_head.load_state_dict(best_aux_state)

    met_tr = _eval_loop(model, ld_tr_eval, device, use_amp=use_amp, threshold=threshold)
    met_va = _eval_loop(model, ld_va, device, use_amp=use_amp, threshold=threshold)
    met_te = _eval_loop(model, ld_te, device, use_amp=use_amp, threshold=threshold)

    write_log(
        f"[DEEP] DONE {model_name}({variant_tag}) | "
        f"tr_auc={met_tr.get('auc'):.4f} va_auc={met_va.get('auc'):.4f} te_auc={met_te.get('auc'):.4f}",
        log_fp,
    )

    ckpt = out_dir / "checkpoint.pt"
    try:
        aux_state = None
        if aux_head is not None:
            aux_state = aux_head.state_dict()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "aux_head_state_dict": aux_state,
                "model_name": model_name,
                "feature_set": feature_set,
                "variant": variant_tag,
                "seed": seed,
            },
            ckpt,
        )
    except Exception as e:
        write_log(f"[DEEP] checkpoint save failed: {e}", log_fp)
        ckpt = None

    pred_maps: Dict[str, Dict[str, float]] = {}
    label_maps: Dict[str, Dict[str, int]] = {}
    if return_pred_maps:
        try:
            # Collect both logit and label maps for paired statistical tests
            # logit_map: {ref_key → f_θ(x)} ∈ ℝ
            # label_map: {ref_key → y}       ∈ {0, 1}
            pred_tr, lbl_tr = _predict_logit_and_label_maps_for_loader(
                model, ld_tr_eval, device, use_amp=use_amp
            )
            pred_va, lbl_va = _predict_logit_and_label_maps_for_loader(
                model, ld_va, device, use_amp=use_amp
            )
            pred_te, lbl_te = _predict_logit_and_label_maps_for_loader(
                model, ld_te, device, use_amp=use_amp
            )
            pred_maps = {"train": pred_tr, "val": pred_va, "test": pred_te}
            label_maps = {"train": lbl_tr, "val": lbl_va, "test": lbl_te}
            write_log(
                f"[PRED] in-memory maps sizes: tr={len(pred_tr)} va={len(pred_va)} te={len(pred_te)}",
                log_fp,
            )
        except Exception as e:
            write_log(f"[PRED] predict_logit_and_label_map failed: {e}", log_fp)
            # Fallback: try logit-only maps (backward compat)
            try:
                pred_tr = _predict_logit_map_for_refs(model, ld_tr_eval, tr_refs, device, use_amp=use_amp)
                pred_va = _predict_logit_map_for_refs(model, ld_va, va_refs, device, use_amp=use_amp)
                pred_te = _predict_logit_map_for_refs(model, ld_te, te_refs, device, use_amp=use_amp)
                pred_maps = {"train": pred_tr, "val": pred_va, "test": pred_te}
                write_log(f"[PRED] Fallback logit-only maps OK", log_fp)
            except Exception as e2:
                write_log(f"[PRED] Fallback also failed: {e2}", log_fp)
                pred_maps = {}

    train_time_sec = time.time() - train_start_time

    rep: Dict[str, Any] = {
        "ok": True,
        "model_name": model_name,
        "feature_set": feature_set,
        "variant": variant_tag or "default",
        "seed": int(seed),
        "metrics": {"train": met_tr, "val": met_va, "test": met_te},
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "train_time_sec": float(train_time_sec),
        "total_epochs": int(epoch),
        "checkpoint": str(ckpt) if ckpt else None,
        "out_dir": str(out_dir),
        "seq_key_used": str(seq_key_used),
        "has_baseline_logit_in_batch": bool(has_lgbm_logit),
        "multi_task_enabled": bool(aux_head is not None),
        "pred_logit_maps": {k: {"size": len(v)} for k, v in pred_maps.items()} if pred_maps else None,
        "label_maps_available": bool(label_maps),
    }

    save_json(out_dir / "report.json", rep)

    if pred_maps:
        rep["_pred_maps_in_memory"] = pred_maps
    if label_maps:
        rep["_label_maps_in_memory"] = label_maps

    return rep


# =========================================================
# Factory hint / dims helper
# =========================================================
def infer_dims_from_batch(batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    f_node = -1
    d_seq = -1
    if batch.get("node_seq", None) is not None:
        f_node = int(batch["node_seq"].shape[-1])
    if batch.get("macro_seq", None) is not None:
        d_seq = int(batch["macro_seq"].shape[-1])
    elif batch.get("extra_seq", None) is not None:
        d_seq = int(batch["extra_seq"].shape[-1])
    elif batch.get("x_seq", None) is not None:
        d_seq = int(batch["x_seq"].shape[-1])
    return f_node, d_seq
