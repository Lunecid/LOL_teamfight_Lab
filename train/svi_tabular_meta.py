"""2024–25 tabular DL meta models for SVI (352-input contract).

EmbedMLP: per-feature numerical embeddings + MLP (Gorishniy et al. style).
TabM: parameter-efficient multi-head MLP ensemble (Gorishniy 2025 sketch).

Pure modules — training loops live in rr20260919_svi_reselection.py.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class EmbedMLP(nn.Module):
    """Each scalar feature -> d_embed, concat, then MLP tower -> logit."""

    def __init__(self, n_in: int, d_embed: int = 8, width: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_in = n_in
        self.d_embed = d_embed
        self.embed = nn.Linear(1, d_embed, bias=False)
        # shared embedding weight across features (parameter efficient)
        hid = n_in * d_embed
        self.tower = nn.Sequential(
            nn.Linear(hid, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in) -> (B, n_in, 1) -> (B, n_in, d) -> (B, n_in*d)
        e = self.embed(x.unsqueeze(-1))
        return self.tower(e.reshape(x.shape[0], -1))


class TabM(nn.Module):
    """Lightweight TabM-style ensemble: k parallel MLP heads sharing a stem."""

    def __init__(self, n_in: int, k: int = 16, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.stem = nn.Sequential(
            nn.Linear(n_in, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))
             for _ in range(k)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        logits = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=-1)  # (B, k)
        return logits.mean(dim=-1, keepdim=True)


def build_meta(family: str, n_in: int, **kwargs) -> nn.Module:
    family = family.lower()
    if family in ("embedmlp", "embed_mlp"):
        return EmbedMLP(n_in, d_embed=int(kwargs.get("d_embed", 8)),
                        width=int(kwargs.get("width", 256)),
                        dropout=float(kwargs.get("dropout", 0.1)))
    if family == "tabm":
        return TabM(n_in, k=int(kwargs.get("k", 16)),
                    width=int(kwargs.get("width", 128)),
                    dropout=float(kwargs.get("dropout", 0.1)))
    raise ValueError(f"unknown meta family {family!r}")


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# Declared HP grids (protocol §3)
EMBEDMLP_GRID: Tuple[dict, ...] = tuple(
    {"d_embed": e, "width": w, "dropout": d}
    for e in (4, 8)
    for w in (128, 256)
    for d in (0.1, 0.3)
)[:6]

TABM_GRID: Tuple[dict, ...] = tuple(
    {"k": k, "width": w, "dropout": 0.1}
    for k in (8, 16, 32)
    for w in (128, 256)
)[:6]

# Reviewer deep families — grids confirmed against run_deep_tabular_baselines / ToG revision
FT_TRANSFORMER_GRID = (
    {"d_token": 64, "depth": 2},
    {"d_token": 128, "depth": 2},
    {"d_token": 192, "depth": 3},
    {"d_token": 128, "depth": 3},
)
TABNET_GRID = (
    {"n_d": 16, "n_steps": 3},
    {"n_d": 32, "n_steps": 3},
    {"n_d": 32, "n_steps": 5},
    {"n_d": 64, "n_steps": 5},
)
SAINT_GRID = (
    {"d_token": 32, "match_disjoint": True},
    {"d_token": 64, "match_disjoint": True},
    {"d_token": 128, "match_disjoint": True},
)
