from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from core.common_torch import autocast_disabled as _autocast_disabled_ctx
from core.common_torch import nan_to_num as _nan_to_num_
from core.common_torch import resolve_node_idx

NODE_IDX: Dict[str, int] = resolve_node_idx()


@contextmanager
def _autocast_disabled():
    with _autocast_disabled_ctx():
        yield


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    return [str(x)]


def _default_node_cat_names() -> List[str]:
    return [
        "champion_id",
        "champion_name_id",
        "summoner_spell_1_id",
        "summoner_spell_2_id",
        "primary_style_id",
        "sub_style_id",
        "primary_rune_1",
        "primary_rune_2",
        "primary_rune_3",
        "primary_rune_4",
        "sub_rune_1",
        "sub_rune_2",
        "stat_perk_offense",
        "stat_perk_flex",
        "stat_perk_defense",
    ]


class NodeFeatureAdapter(nn.Module):
    def __init__(self, f_in: int, d_out: int):
        super().__init__()
        self.f_in = int(f_in)
        self.d_out = int(d_out)

        self.use_cat = bool(getattr(cfg, "NODE_USE_CAT_EMB", True))
        self.emb_dim = int(getattr(cfg, "NODE_CAT_EMB_DIM", 8))
        self.drop = nn.Dropout(float(getattr(cfg, "NODE_CAT_EMB_DROPOUT", getattr(cfg, "DROPOUT", 0.1))))

        self.champ_vocab = int(getattr(cfg, "CHAMPION_VOCAB", 2048))
        self.champ_name_vocab = int(getattr(cfg, "CHAMPION_NAME_VOCAB", 4096))
        self.rune_vocab = int(getattr(cfg, "RUNE_VOCAB", 10000))
        self.rune_style_vocab = int(getattr(cfg, "RUNE_STYLE_VOCAB", 256))
        self.stat_vocab = int(getattr(cfg, "STAT_PERK_VOCAB", 10000))
        self.spell_vocab = int(getattr(cfg, "SUMMONER_SPELL_VOCAB", 512))

        cat_names = []
        cat_names += _as_list(getattr(cfg, "NODE_CATEGORICAL_FEATURE_NAMES", None))
        if not cat_names:
            cat_names = _default_node_cat_names()

        self.cat_specs: List[Tuple[int, str]] = []
        for name in cat_names:
            if name in NODE_IDX:
                self.cat_specs.append((int(NODE_IDX[name]), name))

        cat_idx_set = {i for i, _ in self.cat_specs}
        self.num_idx: List[int] = [i for i in range(self.f_in) if i not in cat_idx_set]

        self.emb_champ = nn.Embedding(self.champ_vocab, self.emb_dim, padding_idx=0)
        self.emb_champ_name = nn.Embedding(self.champ_name_vocab, self.emb_dim, padding_idx=0)
        self.emb_rune = nn.Embedding(self.rune_vocab, self.emb_dim, padding_idx=0)
        self.emb_rune_style = nn.Embedding(self.rune_style_vocab, self.emb_dim, padding_idx=0)
        self.emb_stat = nn.Embedding(self.stat_vocab, self.emb_dim, padding_idx=0)
        self.emb_spell = nn.Embedding(self.spell_vocab, self.emb_dim, padding_idx=0)

        self._cat_table: List[Tuple[int, nn.Embedding, int]] = []
        for idx, name in self.cat_specs:
            if name == "champion_id":
                self._cat_table.append((idx, self.emb_champ, self.champ_vocab))
            elif name == "champion_name_id":
                self._cat_table.append((idx, self.emb_champ_name, self.champ_name_vocab))
            elif name.startswith("summoner_spell_"):
                self._cat_table.append((idx, self.emb_spell, self.spell_vocab))
            elif name.endswith("_style_id"):
                self._cat_table.append((idx, self.emb_rune_style, self.rune_style_vocab))
            elif name.startswith("stat_perk_"):
                self._cat_table.append((idx, self.emb_stat, self.stat_vocab))
            else:
                self._cat_table.append((idx, self.emb_rune, self.rune_vocab))

        in_dim = len(self.num_idx) + self.emb_dim * len(self._cat_table)
        if (not self.use_cat) or (len(self._cat_table) == 0):
            self.proj = nn.Linear(self.f_in, self.d_out)
        else:
            self.proj = nn.Linear(in_dim, self.d_out)

        self.norm = nn.LayerNorm(self.d_out) if bool(getattr(cfg, "NODE_CAT_EMB_NORM", True)) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.use_cat) or (len(self._cat_table) == 0):
            h = self.proj(_nan_to_num_(x.float()))
            h = F.relu(h)
            h = self.drop(h)
            if self.norm is not None:
                h = self.norm(h)
            return h.to(dtype=x.dtype)

        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())

            num = x32[..., self.num_idx] if self.num_idx else None

            embs = []
            for idx, emb, vocab in self._cat_table:
                ids = x32[..., idx].long()
                ids = ids.clamp(min=0, max=max(0, vocab - 1))
                embs.append(emb(ids))

            cat = torch.cat(embs, dim=-1) if embs else None

            if num is None:
                z = cat
            elif cat is None:
                z = num
            else:
                z = torch.cat([num, cat], dim=-1)

            h = self.proj(z)
            h = F.relu(h)
            h = _nan_to_num_(h)

        h = self.drop(h.to(dtype=x.dtype))
        if self.norm is not None:
            h = self.norm(h)
        return h
