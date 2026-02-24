# models_fixed.py
# Updated to support new config node features (champion_id + runes) safely via embeddings,
# while keeping full backward compatibility with purely numeric node features.
#
# (M-3) GATv2: hard-mask A==0 edges (no residual attention), keep self-loop for stability.
# (M-4) MPNN edge attr: transform A -> log1p(A)/log1p(A_max) to match dx/dy/dist scale.

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any, Sequence
from core.improvements import TemporalAttentionPooling
import torch
import torch.nn as nn
import torch.nn.functional as F

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# [P4-DEDUP] Unified imports via common_torch â€” single source of truth.
#
# Previously this module duplicated ~100 lines of:
#   - _autocast_disabled()  (identical to deep.py L240)
#   - _nan_to_num_()        (identical to deep.py L249)
#   - _resolve_node_idx()   (identical to deep.py L279)
#   - NODE_IDX / X_IDX / Y_IDX / ALIVE_IDX resolution
#   - 50+ lines of nested try/except for config imports
#
# All now delegated to common_torch.py.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from core.config import cfg  # type: ignore

from core.common_torch import (
    autocast_disabled as _autocast_disabled_ctx,
    nan_to_num as _nan_to_num_,
    resolve_node_idx,
    pick_xy_indices,
    pick_alive_index,
)

# models.py L1533 — 수정

# ── Module-level singleton: RoleAwareAdjacency ──────────────
# [P1-LOGIC-4 FIX] Single declaration point. Previously declared
# twice (line 44 and 59), with _get_role_adj_module() sandwiched
# between them, causing the second declaration to silently reset
# any value set by intermediate code.
_role_adj_module: Optional[nn.Module] = None


def _get_role_adj_module(device: torch.device) -> nn.Module:
    """Singleton factory for RoleAwareAdjacency (shared across all models).

    [P1-LOGIC-4 FIX] This is now the ONLY creation path.
    _build_adj() delegates here instead of duplicating init logic.
    """
    global _role_adj_module
    if _role_adj_module is None:
        from core.improvements import RoleAwareAdjacency
        init_val = float(getattr(cfg, "ROLE_ADJ_INIT", 0.0))
        _role_adj_module = RoleAwareAdjacency(n_roles=5, init_value=init_val)
    # device migration (idempotent if already on target)
    if next(_role_adj_module.parameters()).device != device:
        _role_adj_module = _role_adj_module.to(device)
    return _role_adj_module


def reset_model_singletons() -> None:
    """모든 module-level singleton을 초기화하여 실험 간 파라미터 누출을 방지.

    수학적 배경
    -----------
    Ablation study에서 Treatment A → Treatment B 순서로 실행할 때,
    singleton 모듈 θ_s의 파라미터가 Training A에서 학습된 값 θ_s^(A)로
    남아 있으면:

        θ_B^(0) = θ_s^(A)  ≠  θ_s^(init)

    이는 Treatment B의 성능이 A의 사전학습 효과를 포함하게 되어,
    Δ_B = AUC_B - AUC_baseline 이 과대평가됩니다.

    이 함수는 각 실험 run 직전에 호출하여:

        θ_s ← None  (lazy re-init 강제)

    을 보장합니다.

    Currently managed singletons
    ----------------------------
    - _role_adj_module : RoleAwareAdjacency(5×5) — Treatment T5
    """
    global _role_adj_module
    _role_adj_module = None
    # 향후 추가되는 singleton도 여기서 초기화
    # e.g., global _game_phase_encoder; _game_phase_encoder = None


def _build_adj(xy: torch.Tensor, alive: Optional[torch.Tensor], multiscale: bool = False) -> torch.Tensor:

    if multiscale:
        A = build_multiscale_adjacency_from_xy(
            xy,
            team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
            add_self_loops=True,
            alive=alive,
            clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
        )
    else:  # ← 기존 코드 그대로
        A = build_adjacency_from_xy(
            xy,
            soft=bool(getattr(cfg, "ADJ_SOFT", True)),
            sigma=float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125)),
            team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
            add_self_loops=True,
            alive=alive,
            clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
        )

    # [P1-LOGIC-4 FIX] Delegate to _get_role_adj_module() — single init path.
    # Previously duplicated inline creation logic, which could diverge from
    # the factory function and bypass reset_model_singletons().
    if bool(getattr(cfg, "USE_ROLE_AWARE_ADJ", False)):
        role_mod = _get_role_adj_module(A.device)
        A = role_mod(A)

    return A


# Backward-compatible alias: existing code calls `with _autocast_disabled():`
@contextmanager
def _autocast_disabled():
    """Thin wrapper â€” delegates to common_torch.autocast_disabled()."""
    with _autocast_disabled_ctx():
        yield


# =========================================================
# Utilities (models-specific, NOT duplicated elsewhere)
# =========================================================
def pick_first(d: dict, *keys):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """x: (..., N, D) / mask: (..., N) with 1 for keep."""
    m = mask.float()
    denom = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=dim) / denom


# =========================================================
# NODE_IDX â€” single source of truth via common_torch
# =========================================================
NODE_IDX: Dict[str, int] = resolve_node_idx()
X_IDX, Y_IDX = pick_xy_indices(NODE_IDX)
ALIVE_IDX = pick_alive_index(NODE_IDX)


def pick_temporal_seq(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, str]:
    """
    Must match main.py's seq_key priority:
      macro_seq -> extra_seq -> x_seq
    """
    for k in ("macro_seq", "extra_seq", "x_seq"):
        x = batch.get(k, None)
        if x is not None:
            return x, k
    raise KeyError("Temporal model requires one of: macro_seq / extra_seq / x_seq")


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    raise ValueError("Cannot infer batch size: batch has no tensor values.")


def _safe_alive(alive: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if alive is None:
        return None
    with torch.no_grad():
        s = float(alive.sum().item())
    if s <= 0.0:
        return None
    return alive


# =========================================================
# NEW: NodeFeatureAdapter (categorical-safe)
# =========================================================
def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    return [str(x)]


def _default_node_cat_names() -> List[str]:
    # Static metadata IDs: champion/runes/styles/spells/stat-perks.
    return [
        "champion_id",
        "champion_name_id",
        "summoner_spell_1_id", "summoner_spell_2_id",
        "primary_style_id", "sub_style_id",
        "primary_rune_1", "primary_rune_2", "primary_rune_3", "primary_rune_4",
        "sub_rune_1", "sub_rune_2",
        "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
    ]


class NodeFeatureAdapter(nn.Module):
    """
    Converts raw node features (possibly containing categorical IDs) into a dense float vector.

    - Numeric features: passed through as-is (float)
    - Categorical features: embedded (champion_id, runes, stat perks) then concatenated
    - Final: linear projection -> d_out

    Backward compatible:
    - If categorical indices are missing, falls back to numeric-only projection.
    - If cfg.NODE_USE_CAT_EMB is False, returns raw features (float) for legacy behavior.
    """

    def __init__(self, f_in: int, d_out: int):
        super().__init__()
        self.f_in = int(f_in)
        self.d_out = int(d_out)

        self.use_cat = bool(getattr(cfg, "NODE_USE_CAT_EMB", True))
        self.emb_dim = int(getattr(cfg, "NODE_CAT_EMB_DIM", 8))
        self.drop = nn.Dropout(float(getattr(cfg, "NODE_CAT_EMB_DROPOUT", getattr(cfg, "DROPOUT", 0.1))))

        # vocab sizes (safe defaults for LoL IDs)
        self.champ_vocab = int(getattr(cfg, "CHAMPION_VOCAB", 2048))
        self.champ_name_vocab = int(getattr(cfg, "CHAMPION_NAME_VOCAB", 4096))
        self.rune_vocab = int(getattr(cfg, "RUNE_VOCAB", 10000))
        self.rune_style_vocab = int(getattr(cfg, "RUNE_STYLE_VOCAB", 256))
        self.stat_vocab = int(getattr(cfg, "STAT_PERK_VOCAB", 10000))
        self.spell_vocab = int(getattr(cfg, "SUMMONER_SPELL_VOCAB", 512))

        # Which feature names to treat as categorical?
        cat_names = []
        cat_names += _as_list(getattr(cfg, "NODE_CATEGORICAL_FEATURE_NAMES", None))
        if not cat_names:
            cat_names = _default_node_cat_names()

        self.cat_specs: List[Tuple[int, str]] = []
        for name in cat_names:
            if name in NODE_IDX:
                self.cat_specs.append((int(NODE_IDX[name]), name))

        # numeric indices = everything else
        cat_idx_set = {i for i, _ in self.cat_specs}
        self.num_idx: List[int] = [i for i in range(self.f_in) if i not in cat_idx_set]

        # Embeddings (shared tables)
        self.emb_champ = nn.Embedding(self.champ_vocab, self.emb_dim, padding_idx=0)
        self.emb_champ_name = nn.Embedding(self.champ_name_vocab, self.emb_dim, padding_idx=0)
        self.emb_rune = nn.Embedding(self.rune_vocab, self.emb_dim, padding_idx=0)
        self.emb_rune_style = nn.Embedding(self.rune_style_vocab, self.emb_dim, padding_idx=0)
        self.emb_stat = nn.Embedding(self.stat_vocab, self.emb_dim, padding_idx=0)
        self.emb_spell = nn.Embedding(self.spell_vocab, self.emb_dim, padding_idx=0)

        # per-feature embedding routing
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
                # runes
                self._cat_table.append((idx, self.emb_rune, self.rune_vocab))

        in_dim = len(self.num_idx) + self.emb_dim * len(self._cat_table)
        if (not self.use_cat) or (len(self._cat_table) == 0):
            # legacy: pure numeric projection
            self.proj = nn.Linear(self.f_in, self.d_out)
        else:
            self.proj = nn.Linear(in_dim, self.d_out)

        self.norm = nn.LayerNorm(self.d_out) if bool(getattr(cfg, "NODE_CAT_EMB_NORM", True)) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., F_in) float tensor from dataloader
        return: (..., d_out) same dtype as x
        """
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
                # handle missing (-1) and out-of-range gracefully
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


# =========================================================
# Adjacency builders (distance-based, soft/hard) for N=10
# =========================================================
def build_adjacency_from_xy(
    xy: torch.Tensor,  # (B, N, 2)
    soft: bool = True,
    sigma: float = 0.125,
    team_edge_weight: float = 1.0,
    add_self_loops: bool = True,
    alive: Optional[torch.Tensor] = None,  # (B,N) 0/1
    clamp_min: float = 1e-4,
) -> torch.Tensor:
    """
    (M-3 stability) We ALWAYS add self-loops (A_ii=1) to avoid rows with all-zero adjacency.
    Dead nodes are already zeroed in features and excluded in pooling, so self-loop on dead
    does not leak information to alive nodes (it only stabilizes softmax/norm).
    """
    B, N, _ = xy.shape

    with _autocast_disabled():
        xy_f = _nan_to_num_(xy.float())
        diff = xy_f.unsqueeze(2) - xy_f.unsqueeze(1)      # (B,N,N,2)
        d2 = (diff * diff).sum(dim=-1)                    # (B,N,N)

        if soft:
            # [FIX-MATH] Adaptive sigma: if cfg.ADJ_SIGMA_ADAPTIVE, use mean
            # pairwise distance per batch instead of fixed sigma.
            _adaptive = bool(getattr(cfg, 'ADJ_SIGMA_ADAPTIVE', False))
            if _adaptive:
                _ratio = float(getattr(cfg, 'ADJ_SIGMA_RATIO', 0.5))
                with torch.no_grad():
                    mean_d = d2.sqrt().mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
                    s2 = (mean_d * _ratio) ** 2
            else:
                s2 = max(1e-8, float(sigma) ** 2)
            A = torch.exp(-0.5 * d2 / s2)
        else:
            A = (d2.sqrt() <= float(sigma)).float()

        # same-team upweight
        team = torch.zeros(N, device=A.device, dtype=torch.long)
        team[5:] = 1
        same_team = (team.unsqueeze(0).unsqueeze(2) == team.unsqueeze(0).unsqueeze(1)).to(A.dtype)  # (1,N,N)
        A = A * (1.0 + (float(team_edge_weight) - 1.0) * same_team)

        # clamp only on non-zero entries (do NOT revive zeros)
        if clamp_min is not None and float(clamp_min) > 0:
            cm = float(clamp_min)
            A = torch.where(A > 0, A.clamp_min(cm), A)

        # alive mask (dead related edges -> 0)
        if alive is not None:
            if alive.ndim == 1:
                alive = alive.unsqueeze(0)
            am = _nan_to_num_(alive.float()).clamp(0.0, 1.0).to(dtype=A.dtype)   # (B,N)
            alive_mask = am.unsqueeze(2) * am.unsqueeze(1)                       # (B,N,N)
            A = A * alive_mask

        # ALWAYS keep self-loops for numerical stability
        if add_self_loops:
            eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = torch.maximum(A, eye)

        out = A.to(dtype=xy.dtype)

    return out


def _parse_float_list(x, default: List[float]) -> List[float]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        out = []
        for t in x:
            try:
                out.append(float(t))
            except Exception:
                pass
        return out if out else list(default)
    s = str(x).strip()
    if not s:
        return list(default)
    s = s.replace(";", ",")
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out if out else list(default)


def build_multiscale_adjacency_from_xy(
    xy: torch.Tensor,
    *,
    sigmas: Optional[List[float]] = None,   # normalized sigmas
    weights: Optional[List[float]] = None,
    team_edge_weight: float = 1.0,
    add_self_loops: bool = True,
    alive: Optional[torch.Tensor] = None,
    clamp_min: float = 1e-4,
) -> torch.Tensor:
    """
    A = sum_k w_k * A(xy; sigma_k), then normalized by sum(w).
    Default sigmas from cfg.MS_SIGMAS_NORM or fallback [0.08,0.125,0.18].
    """
    default_sigmas = _parse_float_list(getattr(cfg, "MS_SIGMAS_NORM", None), [0.08, 0.125, 0.18])
    sigmas = sigmas or default_sigmas

    if weights is None:
        weights = _parse_float_list(getattr(cfg, "MS_WEIGHTS", None), [1.0] * len(sigmas))
    if len(weights) != len(sigmas):
        weights = [1.0] * len(sigmas)

    A_sum = None
    wsum = 0.0
    for s, w in zip(sigmas, weights):
        w = float(w)
        if w <= 0:
            continue
        Ak = build_adjacency_from_xy(
            xy,
            soft=True,
            sigma=float(s),
            team_edge_weight=float(team_edge_weight),
            add_self_loops=bool(add_self_loops),
            alive=alive,
            clamp_min=float(clamp_min),
        )
        A_sum = Ak * w if A_sum is None else (A_sum + Ak * w)
        wsum += w

    if A_sum is None or wsum <= 0:
        return build_adjacency_from_xy(
            xy,
            soft=True,
            sigma=float(sigmas[0] if sigmas else 0.125),
            team_edge_weight=float(team_edge_weight),
            add_self_loops=bool(add_self_loops),
            alive=alive,
            clamp_min=float(clamp_min),
        )

    A = A_sum / float(wsum)
    return A


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
            B, N, _ = A32.shape
            eye = torch.eye(N, device=A32.device, dtype=A32.dtype).unsqueeze(0)
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


# =========================================================
# GNN layers
# =========================================================
class ResGCNLayer(nn.Module):
    """Residual GCN layer."""
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.msg = nn.Linear(d_in, d_out, bias=False)
        self.self_proj = nn.Linear(d_in, d_out, bias=True)  # [FIX] renamed from self.self
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None
        self.norm2 = nn.LayerNorm(d_out) if use_norm else None  # [FIX] separate instance

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
            deg = A32.sum(dim=-1).clamp_min(deg_eps)  # (B,N)
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
    """
    Multi-head attention over nodes with adjacency bias log(A).
    N is tiny (10), so full pairwise compute is OK.

    (M-3) Hard-mask A==0 (or < hard_th) edges to make attention exactly 0 there.
    """

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
        # [BUG-FIX] norm2 was referenced in forward() but never initialized.
        # Pre-LN Transformer pattern requires separate γ,β for each sub-layer:
        #   sub-layer 1 (attention): h = LN₁(x + Attn(x))
        #   sub-layer 2 (FFN):       h₂ = LN₂(h + FFN(h))
        # Sharing one LN forces a single (γ,β) to normalize two different
        # distributions, degrading gradient flow.
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

            q = self.Wq(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)  # (B,H,N,Dh)
            k = self.Wk(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)
            v = self.Wv(x32).view(B, N, self.nhead, self.dh).transpose(1, 2)

            qi = q.unsqueeze(-2)  # (B,H,N,1,Dh)
            kj = k.unsqueeze(-3)  # (B,H,1,N,Dh)
            z = qi + kj
            z = F.leaky_relu(z, negative_slope=self.leaky_alpha)

            att = self.att.view(1, self.nhead, 1, 1, self.dh)
            e = (z * att).sum(dim=-1)  # (B,H,N,N)

            # log(A) bias (finite via clamp)
            bias = torch.log(A32.clamp_min(1e-8)).unsqueeze(1)  # (B,1,N,N)
            e = e + bias

            # (M-3) hard mask: kill edges with A==0 (or below threshold)
            hard_th = float(getattr(cfg, "GAT_HARD_MASK_TH", 0.0))
            if hard_th > 0:
                m = (A32 >= hard_th).unsqueeze(1)  # (B,1,N,N)
            else:
                m = (A32 > 0).unsqueeze(1)
            e = e.masked_fill(~m, -1e9)

            alpha = torch.softmax(e, dim=-1)
            alpha = _nan_to_num_(alpha)

            out = torch.matmul(alpha, v)  # (B,H,N,Dh)
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
    """
    Edge-aware MPNN layer.

    (M-4) Edge attribute normalization:
      - dx,dy,dist are already ~O(1) (normalized coordinates)
      - A can be up to TEAM_EDGE_WEIGHT (often <=2), and can dominate MLP input variance
      => use A_feat = log1p(A) / log1p(A_max) in [0,1]
    """

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
        self.self_proj = nn.Linear(d_in, d_out)  # [FIX] renamed from self.self
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if use_norm else None

    @staticmethod
    def _edge_A_feat(A32: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized A feature in [0,1] by default.
        Config knobs:
          - EDGE_A_LOG1P (bool, default True)
          - EDGE_A_MAX (float, optional) else uses max(1.0, TEAM_EDGE_WEIGHT)
        """
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

        # keep bounded (helps stability)
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

            # (M-4) normalized adjacency feature
            a_feat = EdgeMPNNLayer._edge_A_feat(A32)

            e = torch.stack([dx, dy, dist, a_feat], dim=-1)  # (B,N,N,4)
            e = _nan_to_num_(e)
        return e

    def forward(self, x: torch.Tensor, A: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        with _autocast_disabled():
            x32 = _nan_to_num_(x.float())
            A32 = _nan_to_num_(A.float())
            if bool(getattr(cfg, "ADJ_CLAMP_NONNEG", True)):
                A32 = A32.clamp_min(0.0)

            e = self.build_edge_attr(xy, A32)  # (B,N,N,4)

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


# =========================================================
# Graph Transformer layer
# =========================================================
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

            # hard mask bias: A==0 => -1e9 (no residual mass)
            hard_th = float(getattr(cfg, "GAT_HARD_MASK_TH", 0.0))
            if hard_th > 0:
                m = (A32 >= hard_th)
            else:
                m = (A32 > 0)

            neg_inf = torch.full_like(A32, -1e9)
            bias2 = torch.where(m, torch.log(A32.clamp_min(1e-8)), neg_inf)  # (B,N,N)
            bias = bias2.unsqueeze(1)  # (B,1,N,N)

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


# =========================================================
# Graph pooling
# =========================================================
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


# =========================================================
# GraphEncoder
# =========================================================
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


# =========================================================
# Temporal encoders (RNN/Transformer/TCN)
# =========================================================
class RNNEncoder(nn.Module):
    def __init__(self, rnn_type: str, d_in: int, d_h: int, n_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.d_h = d_h

        rnn_dropout = float(dropout) if n_layers > 1 else 0.0
        if rnn_type == "gru":
            self.rnn = nn.GRU(d_in, d_h, num_layers=n_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=bidirectional)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(d_in, d_h, num_layers=n_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            raise ValueError(rnn_type)

        rnn_out_dim = d_h * (2 if bidirectional else 1)

        # ─── T3: Attention Temporal Pooling ───
        self._use_attn_pool = bool(getattr(cfg, "USE_ATTENTION_POOL", False))
        if self._use_attn_pool:
            attn_dim = int(getattr(cfg, "ATTENTION_POOL_DIM", 64))
            self.attn_pool = TemporalAttentionPooling(rnn_out_dim, attn_dim)
            # output = [last_hidden ‖ attention_context] → 차원 2배
            self.out_dim = rnn_out_dim * 2
        else:
            self.attn_pool = None
            self.out_dim = rnn_out_dim

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_in)
        h0 : (B, d_h) or None

        Returns
        -------
        (B, out_dim) — if attn_pool OFF: out_dim = d_h * num_dirs
                       if attn_pool ON:  out_dim = d_h * num_dirs * 2
        """
        h_init = None
        if h0 is not None:
            num_dirs = 2 if self.bidirectional else 1
            B = h0.shape[0]
            h_expanded = h0.unsqueeze(0).expand(
                self.n_layers * num_dirs, B, self.d_h
            ).contiguous()

            if self.rnn_type == "lstm":
                c0 = torch.zeros_like(h_expanded)
                h_init = (h_expanded, c0)
            else:
                h_init = h_expanded

        if h_init is not None:
            out, _ = self.rnn(x, h_init)   # out: (B, L, d_h * num_dirs)
        else:
            out, _ = self.rnn(x)            # out: (B, L, d_h * num_dirs)

        # ─── T3: Attention Pooling 분기 ───
        # TemporalAttentionPooling.forward() → (output, α_weights) 튜플 반환
        # concat_last=False이므로 output = c = Σ_t α_t·h_t  ∈ ℝ^{d_h·num_dirs}
        # 여기서 수동으로 [h_T ‖ c] 결합하여 최종 출력 생성
        if self._use_attn_pool and self.attn_pool is not None:
            h_last = out[:, -1, :]                              # (B, d_h * num_dirs)
            h_attn, _attn_weights = self.attn_pool(out)         # (B, d_h * num_dirs), (B, L)
            return torch.cat([h_last, h_attn], dim=-1)          # (B, d_h * num_dirs * 2)
        else:
            return out[:, -1, :]                          # (B, d_h * num_dirs)



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
    def __init__(self, d_in: int, d_model: int, nhead: int, n_layers: int, dropout: float = 0.1, max_len: int = 512):
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
    """
    Input: (B,L,D) -> Conv1d over time (causal) -> last step embedding.
    """
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
        h = self.net(x1)        # (B,C,L)
        return h[:, :, -1]


# =========================================================
# Mamba: Selective State Space Model (Gu & Dao, COLM 2024)
# Pure PyTorch — no custom CUDA kernels required.
#
# 4th temporal paradigm:
#   GRU:    h_t = σ(W·[h_{t-1}, x_t])            O(L·D²) seq
#   Trans:  softmax(QK^T/√d)V                     O(L²·D) par
#   TCN:    Σ_k w_k · x_{t-dk}                    O(L·D·K) par
#   Mamba:  h_t = Ā_t·h_{t-1} + B̄_t·x_t           O(L·D·N) par
# =========================================================

class SelectiveScan(nn.Module):
    """
    Core selective scan. Continuous SSM discretized via ZOH:
        Ā_t = exp(Δ_t · A),  B̄_t = Δ_t · B_t
        h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
        y_t = (C_t · h_t).sum(-1)
    where Δ, B, C are input-dependent (selective).
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: S4D-Real init in log-space
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections
        self.proj_dt = nn.Linear(d_model, d_model, bias=True)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)

        # dt bias: log-uniform in [dt_min, dt_max]
        dt_init = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_sp = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.proj_dt.bias.copy_(inv_sp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> y: (B, L, D)"""
        B_sz, L, D = x.shape
        N = self.d_state

        dt = F.softplus(self.proj_dt(x))     # (B,L,D)
        B_t = self.proj_B(x)                  # (B,L,N)
        C_t = self.proj_C(x)                  # (B,L,N)
        A = -torch.exp(self.A_log.float())    # (D,N)

        # Discretize
        A_bar = torch.exp(dt.unsqueeze(-1) * A)           # (B,L,D,N)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(2)       # (B,L,D,N)

        # Sequential scan (L < 100 for teamfight windows)
        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (C_t[:, t].unsqueeze(1) * h).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)   # (B, L, D)
        y = y + self.D * x           # skip connection
        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block: pre-norm → split → [Conv1D→SiLU→SSM] ⊙ [SiLU gate] → proj + residual
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.ssm = SelectiveScan(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # Causal Conv1D
        x_c = x_main.transpose(1, 2)
        x_c = self.conv1d(x_c)[:, :, :x.shape[1]]
        x_c = x_c.transpose(1, 2)
        x_c = F.silu(x_c)

        y = self.ssm(x_c)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y)) + residual


class MambaTemporalEncoder(nn.Module):
    """
    Mamba encoder: (B,L,D_in) → Linear → [MambaBlock×n] → Norm → last_step → (B, d_model)
    """

    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 3,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,L,D_in) -> feat: (B, d_model)"""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h[:, -1, :]


# =========================================================
# Debug hook mixin
# =========================================================
class DebugHookMixin:
    _last_adj: Optional[torch.Tensor] = None
    _last_node_emb: Optional[torch.Tensor] = None

    def _maybe_store_debug(self, A: Optional[torch.Tensor], node_emb: Optional[torch.Tensor]):
        if not bool(getattr(cfg, "DEBUG_GNN", False)):
            return
        if A is not None and isinstance(A, torch.Tensor):
            self._last_adj = A.detach()
        if node_emb is not None and isinstance(node_emb, torch.Tensor):
            self._last_node_emb = node_emb.detach()


# =========================================================
# Tabular-logit passthrough
# =========================================================
class TabLogitModel(nn.Module):
    def __init__(
        self,
        prefer_key: str = "lgbm_logit",
        allow_missing: bool = True,
        missing_fill: float = 0.0,
        extra_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.prefer_key = prefer_key
        self.allow_missing = bool(allow_missing)
        self.missing_fill = float(missing_fill)
        self.extra_keys = list(extra_keys) if extra_keys else []
        self.bias = nn.Parameter(torch.zeros(1))
        self._last_used_key: str = "none"

    def _candidate_keys(self) -> List[str]:
        base = [
            self.prefer_key,
            "lgbm_logit",
            "lgb_logit",
            "lightgbm_logit",
            "tab_logit",
            "tabular_logit",
            "logit",
            "logits",
            "fusion_logit",
        ]
        seen = set()
        out = []
        for k in base + self.extra_keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        strict = bool(getattr(cfg, "STRICT_TAB_LOGIT", False))

        logit = None
        used = None
        for k in self._candidate_keys():
            v = batch.get(k, None)
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            logit = v
            used = k
            break

        if logit is None:
            if strict or (not self.allow_missing):
                raise KeyError(
                    f"TabLogitModel requires one of {self._candidate_keys()} in batch "
                    f"(set cfg.STRICT_TAB_LOGIT=False to allow fallback)."
                )
            B = _infer_batch_size(batch)
            device = None
            dtype = None
            for vv in batch.values():
                if isinstance(vv, torch.Tensor):
                    device = vv.device
                    dtype = vv.dtype
                    break
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
            logit = torch.full((B, 1), self.missing_fill, device=device, dtype=dtype)
            used = "fallback_zeros"

        if logit.ndim == 1:
            logit = logit.unsqueeze(-1)

        out = logit + self.bias.view(1, 1)
        self._last_used_key = used

        if return_aux:
            return out, {"used_logit_key": used, "bias": self.bias.detach().clone()}
        return out


# =========================================================
# Model: RNN-only
# =========================================================
class RNNOnlyModel(nn.Module):
    def __init__(self, kind: str, d_in: int):
        super().__init__()
        kind = kind.lower()
        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        if kind == "ugru":
            self.enc = RNNEncoder("gru", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=False, dropout=drop)
        elif kind == "bigru":
            self.enc = RNNEncoder("gru", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind == "ulstm":
            self.enc = RNNEncoder("lstm", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=False, dropout=drop)
        elif kind == "bilstm":
            self.enc = RNNEncoder("lstm", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind == "transformer":
            self.enc = TransformerTemporalEncoder(
                d_in=d_in,
                d_model=int(getattr(cfg, "TRANS_D_MODEL", 256)),
                nhead=int(getattr(cfg, "TRANS_NHEAD", 4)),
                n_layers=int(getattr(cfg, "TRANS_LAYERS", 3)),
                dropout=float(getattr(cfg, "TRANS_DROPOUT", 0.2)),
                max_len=int(getattr(cfg, "TRANS_MAX_LEN", 512)),
            )
        elif kind == "tcn":
            tcn_channels = int(getattr(cfg, "TCN_CHANNELS", rnn_hidden))
            tcn_levels = int(getattr(cfg, "TCN_LEVELS", 3))
            tcn_kernel = int(getattr(cfg, "TCN_KERNEL", 3))
            tcn_drop = float(getattr(cfg, "TCN_DROPOUT", drop))
            self.enc = TCNTemporalEncoder(
                d_in=d_in,
                channels=tcn_channels,
                n_levels=tcn_levels,
                kernel=tcn_kernel,
                dropout=tcn_drop,
            )
        elif kind == "mamba":
            mamba_d = int(getattr(cfg, "MAMBA_D_MODEL", rnn_hidden))
            mamba_layers = int(getattr(cfg, "MAMBA_LAYERS", 3))
            mamba_state = int(getattr(cfg, "MAMBA_D_STATE", 16))
            mamba_conv = int(getattr(cfg, "MAMBA_D_CONV", 4))
            mamba_expand = int(getattr(cfg, "MAMBA_EXPAND", 2))
            self.enc = MambaTemporalEncoder(
                d_in=d_in,
                d_model=mamba_d,
                n_layers=mamba_layers,
                d_state=mamba_state,
                d_conv=mamba_conv,
                expand=mamba_expand,
                dropout=drop,
            )
        else:
            raise ValueError(kind)

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", rnn_hidden))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.enc.out_dim, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        x, key = pick_temporal_seq(batch)
        feat = self.enc(x)
        logit = self.head(feat)
        if return_aux:
            return logit, {"temporal_feat": feat, "seq_key": key}
        return logit


# =========================================================
# [REC-1] Hybrid RNN: h₀ = MLP_tab(φ_tab)
# ---------------------------------------------------------
# Mathematical formulation:
#   φ_tab = seq_to_tabular(S) ∈ ℝ^{7D}  (last, mean, std, min, max, delta, slope)
#   h₀ = tanh(MLP_tab(LayerNorm(φ_tab))) ∈ ℝ^{d_h}
#   h_t = GRU(s(t), h_{t-1}),  h₀ = MLP_tab(φ_tab)
#
# Bayesian interpretation:
#   φ_tab acts as prior information for sequence interpretation.
#   p(h_T | S, φ_tab) ∝ p(S | h₀ = f(φ_tab)) · p(φ_tab)
#
# Related work: Wide & Deep (Cheng et al., 2016), TabNet (Arik & Pfister, 2021)
# =========================================================
class HybridRNNModel(nn.Module):
    """
    Hybrid RNN with tabular-conditioned initial hidden state.

    The model computes a tabular summary φ_tab from the input sequence using
    7-way statistical aggregation (last, mean, std, min, max, delta, slope),
    then projects this into the RNN's initial hidden state h₀.

    This bridges the Information Access Gap between LGBM (which uses φ_tab)
    and deep models (which only see raw sequences), enabling feature-level
    fusion BEFORE sequence processing.
    """

    # 7-way statistical suffixes matching seq_to_tabular()
    TAB_SUFFIXES = ("_last", "_mean", "_std", "_min", "_max", "_delta", "_slope")

    def __init__(self, kind: str, d_in: int):
        super().__init__()
        kind = kind.lower()
        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))
        h0_proj_dim = int(getattr(cfg, "HYBRID_H0_PROJ_DIM", 64))
        h0_dropout = float(getattr(cfg, "HYBRID_H0_DROPOUT", 0.15))

        # --- RNN encoder (same as RNNOnlyModel) ---
        if kind in ("bigru", "hybrid_bigru"):
            self.enc = RNNEncoder("gru", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind in ("bilstm", "hybrid_bilstm"):
            self.enc = RNNEncoder("lstm", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind in ("ugru", "hybrid_ugru"):
            self.enc = RNNEncoder("gru", d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=False, dropout=drop)
        else:
            raise ValueError(f"HybridRNNModel unsupported kind: {kind}")

        # --- Tabular → h₀ projection MLP ---
        # φ_tab dimension: 7 * d_in (7-way stats of each feature)
        d_tab = len(self.TAB_SUFFIXES) * d_in
        self.tab_to_h0 = nn.Sequential(
            nn.LayerNorm(d_tab),
            nn.Linear(d_tab, h0_proj_dim),
            nn.GELU(),
            nn.Dropout(h0_dropout),
            nn.Linear(h0_proj_dim, rnn_hidden),
            nn.Tanh(),  # bounds h₀ to [-1, 1] for RNN stability
        )

        # --- Classification head ---
        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", rnn_hidden))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.enc.out_dim, head_hidden, 1, dropout=drop, layers=head_layers)

        self._d_in = d_in
        self._rnn_hidden = rnn_hidden

    @staticmethod
    def _compute_tab_features(x: torch.Tensor) -> torch.Tensor:
        """
        GPU-native implementation of seq_to_tabular() using PyTorch operations.
        Avoids NumPy conversion overhead.

        Parameters
        ----------
        x : (B, L, D) input sequence tensor

        Returns
        -------
        φ_tab : (B, 7*D) tabular statistics
            Concatenation of [last, mean, std, min, max, delta, slope]
        """
        # x: (B, L, D)
        x_last = x[:, -1, :]                          # (B, D) — last timestep
        x_mean = x.mean(dim=1)                         # (B, D) — temporal mean
        x_std = x.std(dim=1, correction=0).clamp(min=1e-8)  # (B, D) — temporal std
        x_min = x.min(dim=1).values                    # (B, D) — temporal min
        x_max = x.max(dim=1).values                    # (B, D) — temporal max
        x_delta = x[:, -1, :] - x[:, 0, :]            # (B, D) — first-to-last delta

        # Slope: linear regression coefficient β₁ = Cov(t, x) / Var(t)
        L = x.shape[1]
        t = torch.arange(L, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        t_mean = (L - 1) / 2.0
        t_var = t.var(dim=1, correction=0).clamp(min=1e-8)  # (1, 1)
        x_slope = ((t - t_mean) * (x - x_mean.unsqueeze(1))).mean(dim=1) / t_var  # (B, D)

        return torch.cat([x_last, x_mean, x_std, x_min, x_max, x_delta, x_slope], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        x, key = pick_temporal_seq(batch)  # (B, L, D)

        # Compute tabular features → initial hidden state
        phi_tab = self._compute_tab_features(x)  # (B, 7*D)
        h0 = self.tab_to_h0(phi_tab)             # (B, d_h)

        # RNN with h₀ conditioning
        feat = self.enc(x, h0=h0)  # (B, out_dim)
        logit = self.head(feat)     # (B, 1)

        if return_aux:
            return logit, {
                "temporal_feat": feat,
                "h0": h0,
                "phi_tab_dim": phi_tab.shape[-1],
                "seq_key": key,
            }
        return logit


# =========================================================
# Node-graph helpers shared by multiple models
# =========================================================
def _extract_alive_from_raw(x_raw: torch.Tensor) -> Optional[torch.Tensor]:
    alive = None
    if bool(getattr(cfg, "USE_ALIVE_MASK", True)) and (ALIVE_IDX is not None) and (ALIVE_IDX < x_raw.shape[-1]):
        alive = (x_raw[:, :, ALIVE_IDX] > 0.5).to(dtype=x_raw.dtype)
        alive = _safe_alive(alive)
    return alive


def _build_adj(xy: torch.Tensor, alive: Optional[torch.Tensor], multiscale: bool = False) -> torch.Tensor:
    if multiscale:
        return build_multiscale_adjacency_from_xy(
            xy,
            team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
            add_self_loops=True,
            alive=alive,
            clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
        )
    return build_adjacency_from_xy(
        xy,
        soft=bool(getattr(cfg, "ADJ_SOFT", True)),
        sigma=float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125)),
        team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
        add_self_loops=True,
        alive=alive,
        clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
    )


# =========================================================
# Model: GNN-only on last frame (uses node_seq)
# =========================================================
class GNNOnlyModel(nn.Module, DebugHookMixin):
    def __init__(self, gnn_kind: str, f_node: int, *, multiscale_adj: bool = False):
        super().__init__()
        gnn_kind = gnn_kind.lower()
        if gnn_kind not in ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn"):
            raise ValueError(gnn_kind)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        self.gnn_kind = gnn_kind
        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(kind=gnn_kind, d_in=gnn_dim, d_h=gnn_dim, n_layers=2, dropout=gnn_drop, use_norm=gnn_norm)

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", int(getattr(cfg, "RNN_HIDDEN", 128))))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        drop = float(getattr(cfg, "DROPOUT", 0.1))
        self.head = MLP(3 * gnn_dim, head_hidden, 1, dropout=drop, layers=head_layers)

    def encode_last(self, node_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_raw = node_seq[:, -1, :, :]          # (B,N,Fraw)
        xy = x_raw[:, :, [X_IDX, Y_IDX]]       # (B,N,2)

        alive = _extract_alive_from_raw(x_raw)

        x = self.node_adapter(x_raw)           # (B,N,gnn_dim)
        if alive is not None:
            x = x * alive.unsqueeze(-1)

        A = _build_adj(xy, alive, multiscale=self.multiscale_adj)

        if self.gnn_kind == "mpnn":
            h = self.gnn(x, A, xy=xy)
        else:
            h = self.gnn(x, A)

        pooled = pool_team_repr(h, alive=alive)
        return pooled, h, A, alive

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("GNNOnlyModel requires node_seq")

        pooled, h, A, alive = self.encode_last(node_seq)
        self._maybe_store_debug(A, h)

        logit = self.head(pooled)
        if return_aux:
            return logit, {"node_emb": h, "pooled": pooled, "adj": A, "alive": alive}
        return logit


# =========================================================
# Model: STGNN (GNN per timestep -> temporal GRU -> head)
# =========================================================
class STGNNModel(nn.Module, DebugHookMixin):
    def __init__(self, f_node: int, *, gnn_kind: str = "graphsage", multiscale_adj: bool = False):
        super().__init__()
        gnn_kind = str(gnn_kind).lower()
        if gnn_kind not in ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn"):
            raise ValueError(gnn_kind)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        self.gnn_kind = gnn_kind
        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(kind=self.gnn_kind, d_in=gnn_dim, d_h=gnn_dim, n_layers=2, dropout=gnn_drop, use_norm=gnn_norm)

        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        self.temporal = RNNEncoder(
            rnn_type="gru",
            d_in=3 * gnn_dim,
            d_h=rnn_hidden,
            n_layers=rnn_layers,
            bidirectional=True,
            dropout=drop,
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", rnn_hidden))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.temporal.out_dim, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("STGNNModel requires node_seq")

        B, L, N, Fraw = node_seq.shape
        pooled_list = []
        last_A = None
        last_h = None
        adj_list = [] if return_aux else None

        for t in range(L):
            x_raw = node_seq[:, t, :, :]
            xy = x_raw[:, :, [X_IDX, Y_IDX]]

            alive = _extract_alive_from_raw(x_raw)

            x = self.node_adapter(x_raw)
            if alive is not None:
                x = x * alive.unsqueeze(-1)

            A = _build_adj(xy, alive, multiscale=self.multiscale_adj)

            if self.gnn_kind == "mpnn":
                h = self.gnn(x, A, xy=xy)
            else:
                h = self.gnn(x, A)

            pooled_list.append(pool_team_repr(h, alive=alive))

            last_A = A
            last_h = h
            if return_aux:
                adj_list.append(A)

        self._maybe_store_debug(last_A, last_h)

        seq = torch.stack(pooled_list, dim=1)
        temp_feat = self.temporal(seq)
        logit = self.head(temp_feat)

        if return_aux:
            return logit, {"pooled_seq": seq, "temporal_feat": temp_feat, "adj_seq": adj_list}
        return logit


# =========================================================
# Model: STGCN (GNN per timestep -> pooled_seq -> TCN -> head)
# =========================================================
class STGCNModel(nn.Module, DebugHookMixin):
    """
    Practical STGCN-style:
      - per-timestep GNN -> pooled_seq (B,L,3D)
      - temporal = TCN over pooled_seq
      - head
    """

    def __init__(self, f_node: int, gnn_kind: str = "graphsage", multiscale_adj: bool = False):
        super().__init__()
        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        self.gnn_kind = str(gnn_kind).lower()
        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(kind=self.gnn_kind, d_in=gnn_dim, d_h=gnn_dim, n_layers=2, dropout=gnn_drop, use_norm=gnn_norm)

        drop = float(getattr(cfg, "DROPOUT", 0.1))
        tcn_channels = int(getattr(cfg, "STGCN_TCN_CHANNELS", getattr(cfg, "TCN_CHANNELS", 128)))
        tcn_levels = int(getattr(cfg, "STGCN_TCN_LEVELS", getattr(cfg, "TCN_LEVELS", 3)))
        tcn_kernel = int(getattr(cfg, "STGCN_TCN_KERNEL", getattr(cfg, "TCN_KERNEL", 3)))
        tcn_drop = float(getattr(cfg, "STGCN_TCN_DROPOUT", getattr(cfg, "TCN_DROPOUT", drop)))

        self.temporal = TCNTemporalEncoder(
            d_in=3 * gnn_dim,
            channels=tcn_channels,
            n_levels=tcn_levels,
            kernel=tcn_kernel,
            dropout=tcn_drop,
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", int(getattr(cfg, "RNN_HIDDEN", 128))))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.temporal.out_dim, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("STGCNModel requires node_seq")

        B, L, N, Fraw = node_seq.shape
        pooled_list = []
        last_A = None
        last_h = None

        for t in range(L):
            x_raw = node_seq[:, t, :, :]
            xy = x_raw[:, :, [X_IDX, Y_IDX]]

            alive = _extract_alive_from_raw(x_raw)

            x = self.node_adapter(x_raw)
            if alive is not None:
                x = x * alive.unsqueeze(-1)

            A = _build_adj(xy, alive, multiscale=self.multiscale_adj)

            if self.gnn_kind == "mpnn":
                h = self.gnn(x, A, xy=xy)
            else:
                h = self.gnn(x, A)

            pooled_list.append(pool_team_repr(h, alive=alive))
            last_A, last_h = A, h

        self._maybe_store_debug(last_A, last_h)

        seq = torch.stack(pooled_list, dim=1)
        feat = self.temporal(seq)
        logit = self.head(feat)

        if return_aux:
            return logit, {"pooled_seq": seq, "temporal_feat": feat}
        return logit


# =========================================================
# Model: Edge-aware STGNN (MPNN per timestep -> BiGRU -> head)
# =========================================================
class EdgeSTGNNModel(nn.Module, DebugHookMixin):
    """
    Edge-aware STGNN:
      - per-timestep MPNN (needs xy)
      - pooled_seq -> BiGRU -> head
    """

    def __init__(self, f_node: int, multiscale_adj: bool = False):
        super().__init__()
        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(kind="mpnn", d_in=gnn_dim, d_h=gnn_dim, n_layers=2, dropout=gnn_drop, use_norm=gnn_norm)

        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        self.temporal = RNNEncoder(
            rnn_type="gru",
            d_in=3 * gnn_dim,
            d_h=rnn_hidden,
            n_layers=rnn_layers,
            bidirectional=True,
            dropout=drop,
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", rnn_hidden))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.temporal.out_dim, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("EdgeSTGNNModel requires node_seq")

        B, L, N, Fraw = node_seq.shape
        pooled_list = []
        last_A = None
        last_h = None

        for t in range(L):
            x_raw = node_seq[:, t, :, :]
            xy = x_raw[:, :, [X_IDX, Y_IDX]]

            alive = _extract_alive_from_raw(x_raw)

            x = self.node_adapter(x_raw)
            if alive is not None:
                x = x * alive.unsqueeze(-1)

            A = _build_adj(xy, alive, multiscale=self.multiscale_adj)
            h = self.gnn(x, A, xy=xy)

            pooled_list.append(pool_team_repr(h, alive=alive))
            last_A, last_h = A, h

        self._maybe_store_debug(last_A, last_h)

        seq = torch.stack(pooled_list, dim=1)
        feat = self.temporal(seq)
        logit = self.head(feat)

        if return_aux:
            return logit, {"pooled_seq": seq, "temporal_feat": feat}
        return logit


# =========================================================
# Model: STMamba (GNN per timestep → Mamba temporal → head)
#
# Graph + SSM spatio-temporal variant. Replaces BiGRU (STGNN)
# and TCN (STGCN) with Mamba for temporal aggregation.
# =========================================================
class STMambaModel(nn.Module, DebugHookMixin):
    """GNN per-timestep + Mamba temporal aggregation."""

    def __init__(self, f_node: int, *, gnn_kind: str = "graphsage",
                 multiscale_adj: bool = False):
        super().__init__()
        gnn_kind = str(gnn_kind).lower()
        if gnn_kind not in ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn"):
            raise ValueError(gnn_kind)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        self.gnn_kind = gnn_kind
        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(
            kind=gnn_kind, d_in=gnn_dim, d_h=gnn_dim,
            n_layers=2, dropout=gnn_drop, use_norm=gnn_norm,
        )

        # Mamba temporal encoder
        mamba_d = int(getattr(cfg, "MAMBA_D_MODEL",
                              int(getattr(cfg, "RNN_HIDDEN", 128))))
        mamba_layers = int(getattr(cfg, "MAMBA_LAYERS", 3))
        mamba_state = int(getattr(cfg, "MAMBA_D_STATE", 16))

        self.temporal = MambaTemporalEncoder(
            d_in=3 * gnn_dim,
            d_model=mamba_d,
            n_layers=mamba_layers,
            d_state=mamba_state,
            dropout=drop,
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN",
                                   int(getattr(cfg, "RNN_HIDDEN", 128))))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(self.temporal.out_dim, head_hidden, 1,
                        dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor],
                return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("STMambaModel requires node_seq")

        B, L, N, Fraw = node_seq.shape
        pooled_list = []
        last_A, last_h = None, None

        for t in range(L):
            x_raw = node_seq[:, t, :, :]
            xy = x_raw[:, :, [X_IDX, Y_IDX]]
            alive = _extract_alive_from_raw(x_raw)

            x = self.node_adapter(x_raw)
            if alive is not None:
                x = x * alive.unsqueeze(-1)

            A = _build_adj(xy, alive, multiscale=self.multiscale_adj)

            if self.gnn_kind == "mpnn":
                h = self.gnn(x, A, xy=xy)
            else:
                h = self.gnn(x, A)

            pooled_list.append(pool_team_repr(h, alive=alive))
            last_A, last_h = A, h

        self._maybe_store_debug(last_A, last_h)

        seq = torch.stack(pooled_list, dim=1)
        temp_feat = self.temporal(seq)
        logit = self.head(temp_feat)

        if return_aux:
            return logit, {"pooled_seq": seq, "temporal_feat": temp_feat}
        return logit


# =========================================================
# Fusion: Gated (GNN(last) + RNN(macro) + optional LGBM logit)
# =========================================================
class GatedFusion(nn.Module):
    def __init__(self, d_a: int, d_b: int, d_out: int, gate_h: int = 16, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_a + d_b, gate_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_h, d_out),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([a, b], dim=-1))
        if g.shape[-1] == 1 and a.shape[-1] != 1:
            g = g.expand_as(a)
        return g * a + (1.0 - g) * b


class FusionGatedGNNBiGRU(nn.Module, DebugHookMixin):
    """
    - node_seq -> GNN(last frame) pooled -> gnn_feat
    - macro_seq/extra_seq/x_seq -> BiGRU -> rnn_feat
    - optional lgbm_logit -> appended to final head
    """

    def __init__(self, f_node: int, d_macro: int, use_lgbm_logit: bool = True):
        super().__init__()
        self.use_lgbm_logit = bool(use_lgbm_logit)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        self.gnn_branch = GNNOnlyModel("graphsage", f_node=f_node)
        self.gnn_dim = 3 * gnn_dim

        self.rnn_enc = RNNEncoder("gru", d_macro, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        self.rnn_dim = self.rnn_enc.out_dim

        fuse_dim = max(self.gnn_dim, self.rnn_dim)
        self.gnn_proj = nn.Linear(self.gnn_dim, fuse_dim) if self.gnn_dim != fuse_dim else nn.Identity()
        self.rnn_proj = nn.Linear(self.rnn_dim, fuse_dim) if self.rnn_dim != fuse_dim else nn.Identity()

        self.fuser = GatedFusion(
            fuse_dim,
            fuse_dim,
            fuse_dim,
            gate_h=int(getattr(cfg, "FUSION_GATE_H", 8)),
            dropout=drop,
        )

        head_in = fuse_dim + (1 if self.use_lgbm_logit else 0)
        fusion_mlp_h = int(getattr(cfg, "FUSION_MLP_H", 32))
        self.head = MLP(head_in, fusion_mlp_h, 1, dropout=drop, layers=2)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("Fusion requires node_seq")

        extra_seq, seq_key = pick_temporal_seq(batch)

        gnn_feat, h, A, alive = self.gnn_branch.encode_last(node_seq)
        self._maybe_store_debug(A, h)

        rnn_feat = self.rnn_enc(extra_seq)

        a = self.gnn_proj(gnn_feat)
        b = self.rnn_proj(rnn_feat)
        fused = self.fuser(a, b)

        if self.use_lgbm_logit and (batch.get("lgbm_logit", None) is not None):
            llog = batch["lgbm_logit"]
            if isinstance(llog, torch.Tensor):
                if llog.ndim == 1:
                    llog = llog.unsqueeze(-1)
            else:
                llog = torch.as_tensor(llog).unsqueeze(-1)
            fused2 = torch.cat([fused, llog.to(device=fused.device, dtype=fused.dtype)], dim=-1)
        else:
            fused2 = fused

        logit = self.head(fused2)

        if return_aux:
            return logit, {"gnn_feat": gnn_feat, "rnn_feat": rnn_feat, "fused": fused, "adj": A, "seq_key": seq_key}
        return logit


# =========================================================
# Event-token Cross-Attention ST model
# =========================================================
class EventTokenEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        type_vocab = int(getattr(cfg, "EVENT_TYPE_VOCAB", 128))
        self.type_emb = nn.Embedding(type_vocab, d_model)
        self.actor_emb = nn.Embedding(11, d_model)      # 0..10
        self.team_emb  = nn.Embedding(3, d_model)       # 0 blue, 1 red, 2 unk
        self.cont_proj = nn.Linear(5, d_model)
        self.drop = nn.Dropout(float(getattr(cfg, "DROPOUT", 0.1)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, etype, eactor, eteam, econt):
        x = (
            self.type_emb(etype.long())
            + self.actor_emb(eactor.long().clamp(0, 10))
            + self.team_emb(eteam.long().clamp(0, 2))
            + self.cont_proj(econt.float())
        )
        x = self.drop(x)
        return self.norm(x)


class EventXAttnSTModel(nn.Module, DebugHookMixin):
    """
    Trajectory(Graph) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Â Event-token Cross-Attention
      1) node_seq -> per-timestep GNN -> pooled z_t
      2) event tokens -> embedding e_k
      3) cross-attn: Q=e_k, K/V=z_t  (events attend to trajectory phases)
      4) pool over events -> fight repr
    """

    def __init__(self, f_node: int, gnn_kind: str = "mpnn", multiscale_adj: bool = False):
        super().__init__()
        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))

        self.gnn_kind = str(gnn_kind).lower()
        self.multiscale_adj = bool(multiscale_adj)

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(
            kind=self.gnn_kind,
            d_in=gnn_dim,
            d_h=gnn_dim,
            n_layers=2,
            dropout=gnn_drop,
            use_norm=gnn_norm,
        )

        d_model = int(getattr(cfg, "XATTN_D_MODEL", 128))
        nhead   = int(getattr(cfg, "XATTN_NHEAD", 4))
        drop    = float(getattr(cfg, "DROPOUT", 0.1))

        self.z_proj = nn.Linear(3 * gnn_dim, d_model)
        self.ev_enc = EventTokenEncoder(d_model=d_model)

        self.xattn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, nhead),
            dropout=drop,
            batch_first=True,
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", int(getattr(cfg, "RNN_HIDDEN", 128))))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        self.head = MLP(2 * d_model, head_hidden, 1, dropout=drop, layers=head_layers)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("EventXAttnSTModel requires node_seq")

        etype = batch.get("event_type", None)
        eactor = batch.get("event_actor", None)
        eteam = batch.get("event_team", None)
        econt = batch.get("event_cont", None)
        emask = batch.get("event_mask", None)
        if any(x is None for x in (etype, eactor, eteam, econt, emask)):
            raise KeyError("EventXAttnSTModel requires event_type/event_actor/event_team/event_cont/event_mask")

        B, L, N, Fraw = node_seq.shape
        pooled_list = []
        last_A = None
        last_h = None

        for t in range(L):
            x_raw = node_seq[:, t, :, :]
            xy  = x_raw[:, :, [X_IDX, Y_IDX]]

            alive = _extract_alive_from_raw(x_raw)

            x = self.node_adapter(x_raw)
            if alive is not None:
                x = x * alive.unsqueeze(-1)

            A = _build_adj(xy, alive, multiscale=self.multiscale_adj)

            if self.gnn_kind == "mpnn":
                h = self.gnn(x, A, xy=xy)
            else:
                h = self.gnn(x, A)

            pooled_list.append(pool_team_repr(h, alive=alive))
            last_A, last_h = A, h

        self._maybe_store_debug(last_A, last_h)

        z_seq = torch.stack(pooled_list, dim=1)
        z_seq = self.z_proj(z_seq)
        z_last = z_seq[:, -1, :]

        e_emb = self.ev_enc(etype, eactor, eteam, econt)

        with _autocast_disabled():
            q = _nan_to_num_(e_emb.float())
            k = _nan_to_num_(z_seq.float())
            v = k
            attn_out, _ = self.xattn(q, k, v, need_weights=False)
            attn_out = _nan_to_num_(attn_out)

        attn_out = attn_out.to(dtype=e_emb.dtype)

        m = emask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        cross_pool = (attn_out * m).sum(dim=1) / denom

        fused = torch.cat([z_last, cross_pool], dim=-1)
        logit = self.head(fused)

        if return_aux:
            return logit, {
                "z_seq": z_seq,
                "z_last": z_last,
                "event_emb": e_emb,
                "attn_out": attn_out,
                "cross_pool": cross_pool,
            }
        return logit


# =========================================================
# Model factory
# =========================================================
@dataclass
class ModelSpec:
    name: str
    feature_set: str = "full"


def infer_dims_from_batch(batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    f_node = -1
    d_macro = -1
    if batch.get("node_seq", None) is not None:
        f_node = int(batch["node_seq"].shape[-1])

    if batch.get("macro_seq", None) is not None:
        d_macro = int(batch["macro_seq"].shape[-1])
    elif batch.get("extra_seq", None) is not None:
        d_macro = int(batch["extra_seq"].shape[-1])
    elif batch.get("x_seq", None) is not None:
        d_macro = int(batch["x_seq"].shape[-1])

    return f_node, d_macro


# =========================================================
# [P2-STRUCT-5] Model Registry (Open-Closed Principle)
# ---------------------------------------------------------
# Previously: build_model() was a 40+ branch if-elif chain.
#   Cyclomatic complexity: CC ≥ 26 (recommended ≤ 10)
#   Adding a new model required modifying build_model().
#
# Now: Registry pattern with @register_model decorator.
#   CC(build_model) = 2 (constant, regardless of model count)
#   New models: add @register_model("name") — no core edits.
#
# Mathematical justification:
#   Let M = |models|. Old CC = M+1. New CC = 2.
#   Maintenance cost: O(M) → O(1) per addition.
# =========================================================
from typing import Callable as _Callable

MODEL_REGISTRY: Dict[str, _Callable[..., nn.Module]] = {}


def register_model(*aliases: str):
    """Decorator to register a model factory under one or more names.

    Usage:
        @register_model("rnn_bigru", "bigru")
        def _build_bigru(f_node, d_seq, use_lgbm_logit):
            return RNNOnlyModel("bigru", d_in=d_seq)
    """
    def decorator(factory_fn: _Callable[..., nn.Module]):
        for alias in aliases:
            MODEL_REGISTRY[alias.lower().strip()] = factory_fn
        return factory_fn
    return decorator


# --- Tabular passthrough ---
@register_model("lgbm", "lightgbm", "baseline_lgbm", "tab_lgbm", "tab_logit", "tab")
def _build_tab(f_node, d_seq, use_lgbm_logit):
    return TabLogitModel(prefer_key="lgbm_logit", allow_missing=True)

# --- RNN family ---
@register_model("rnn_ugru", "ugru")
def _build_ugru(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("ugru", d_in=d_seq)

@register_model("rnn_bigru", "bigru")
def _build_bigru(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("bigru", d_in=d_seq)

@register_model("rnn_ulstm", "ulstm")
def _build_ulstm(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("ulstm", d_in=d_seq)

@register_model("rnn_bilstm", "bilstm")
def _build_bilstm(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("bilstm", d_in=d_seq)

@register_model("rnn_transformer", "transformer")
def _build_transformer(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("transformer", d_in=d_seq)

@register_model("rnn_tcn", "tcn")
def _build_tcn(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("tcn", d_in=d_seq)

@register_model("rnn_mamba", "mamba")
def _build_mamba(f_node, d_seq, use_lgbm_logit):
    return RNNOnlyModel("mamba", d_in=d_seq)

# --- Hybrid h₀-conditioned RNNs ---
@register_model("hybrid_bigru", "rnn_hybrid_bigru")
def _build_hybrid_bigru(f_node, d_seq, use_lgbm_logit):
    return HybridRNNModel("bigru", d_in=d_seq)

@register_model("hybrid_bilstm", "rnn_hybrid_bilstm")
def _build_hybrid_bilstm(f_node, d_seq, use_lgbm_logit):
    return HybridRNNModel("bilstm", d_in=d_seq)

@register_model("hybrid_ugru", "rnn_hybrid_ugru")
def _build_hybrid_ugru(f_node, d_seq, use_lgbm_logit):
    return HybridRNNModel("ugru", d_in=d_seq)

# --- GNN family ---
@register_model("gnn_gcn", "gcn")
def _build_gcn(f_node, d_seq, use_lgbm_logit):
    return GNNOnlyModel("gcn", f_node=f_node)

@register_model("gnn_graphsage", "graphsage")
def _build_graphsage(f_node, d_seq, use_lgbm_logit):
    return GNNOnlyModel("graphsage", f_node=f_node)

@register_model("gnn_graphtransformer", "graphtransformer")
def _build_graphtransformer(f_node, d_seq, use_lgbm_logit):
    return GNNOnlyModel("graphtransformer", f_node=f_node)

@register_model("gnn_gatv2", "gatv2", "gat")
def _build_gatv2(f_node, d_seq, use_lgbm_logit):
    return GNNOnlyModel("gatv2", f_node=f_node)

@register_model("gnn_mpnn", "mpnn")
def _build_mpnn(f_node, d_seq, use_lgbm_logit):
    return GNNOnlyModel("mpnn", f_node=f_node)

# --- Spatio-temporal family ---
@register_model("gnn_stgnn", "stgnn")
def _build_stgnn(f_node, d_seq, use_lgbm_logit):
    return STGNNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGNN_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STGNN_MULTISCALE_ADJ", False)),
    )

@register_model("gnn_stgcn", "stgcn")
def _build_stgcn(f_node, d_seq, use_lgbm_logit):
    return STGCNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGCN_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STGCN_MULTISCALE_ADJ", False)),
    )

@register_model("edge_stgnn", "stgnn_edge", "stgnn_mpnn", "stgnn_edge_mpnn")
def _build_edge_stgnn(f_node, d_seq, use_lgbm_logit):
    return EdgeSTGNNModel(
        f_node=f_node,
        multiscale_adj=bool(getattr(cfg, "EDGE_STGNN_MULTISCALE_ADJ", False)),
    )

@register_model("ms_stgcn", "multiscale_stgcn", "ms_dyngraph")
def _build_ms_stgcn(f_node, d_seq, use_lgbm_logit):
    return STGCNModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STGCN_GNN_KIND", "graphsage")),
        multiscale_adj=True,
    )

@register_model("ms_stgnn", "multiscale_stgnn")
def _build_ms_stgnn(f_node, d_seq, use_lgbm_logit):
    return EdgeSTGNNModel(
        f_node=f_node,
        multiscale_adj=True,
    )

@register_model("stgnn_mamba", "st_mamba", "stmamba")
def _build_stmamba(f_node, d_seq, use_lgbm_logit):
    return STMambaModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "STMAMBA_GNN_KIND", "graphsage")),
        multiscale_adj=bool(getattr(cfg, "STMAMBA_MULTISCALE_ADJ", False)),
    )

@register_model("event_xattn", "xattn", "traj_event_xattn")
def _build_xattn(f_node, d_seq, use_lgbm_logit):
    return EventXAttnSTModel(
        f_node=f_node,
        gnn_kind=str(getattr(cfg, "XATTN_GNN_KIND", "mpnn")),
        multiscale_adj=bool(getattr(cfg, "XATTN_MULTISCALE_ADJ", False)),
    )

# --- Fusion ---
@register_model("fusion_gated_gnn_bigru", "lgbm_dual_gnn_bigru", "fusion")
def _build_fusion(f_node, d_seq, use_lgbm_logit):
    return FusionGatedGNNBiGRU(f_node=f_node, d_macro=d_seq, use_lgbm_logit=use_lgbm_logit)


def build_model(model_name: str, f_node: int, d_seq: int, use_lgbm_logit: bool = True) -> nn.Module:
    """Build a model by name using the MODEL_REGISTRY.

    [P2-STRUCT-5] Cyclomatic Complexity: 2 (constant).
    Previously: 26+ (linear in number of models).
    New models can be added via @register_model without modifying this function.
    """
    m = (model_name or "").lower().strip()
    if m not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Available ({len(available)}): {available}"
        )
    return MODEL_REGISTRY[m](f_node, d_seq, use_lgbm_logit)


# =========================================================
# Optional: LGBM feature-name sanitizer
# =========================================================
def sanitize_feature_name(name: str) -> str:
    bad = ["{", "}", "[", "]", ":", '"', "'", "\\", "\n", "\r", "\t"]
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def sanitize_feature_names(names: Sequence[str]) -> List[str]:
    return [sanitize_feature_name(str(n)) for n in names]
