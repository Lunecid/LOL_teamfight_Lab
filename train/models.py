# models_fixed.py
# Updated to support new config node features (champion_id + runes) safely via embeddings,
# while keeping full backward compatibility with purely numeric node features.
#
# (M-3) GATv2: hard-mask A==0 edges (no residual attention), keep self-loop for stability.
# (M-4) MPNN edge attr: transform A -> log1p(A)/log1p(A_max) to match dx/dy/dist scale.

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
# [P4-DEDUP] Unified imports via common_torch â€" single source of truth.
#
# Previously this module duplicated ~100 lines of:
#   - _autocast_disabled()  (identical to deep.py L240)
#   - _nan_to_num_()        (identical to deep.py L249)
#   - _resolve_node_idx()   (identical to deep.py L279)
#   - NODE_IDX / X_IDX / Y_IDX / ALIVE_IDX resolution
#   - 50+ lines of nested try/except for config imports
#
# All now delegated to common_torch.py.
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
from core.config import cfg, F_GLOBAL  # type: ignore

from core.common_torch import (
    autocast_disabled as _autocast_disabled_ctx,
    nan_to_num as _nan_to_num_,
    resolve_node_idx,
    pick_xy_indices,
    pick_alive_index,
)
from train.graph_encoder import GraphEncoder, pool_team_repr
from train.node_adapter import NodeFeatureAdapter
from train.temporal_encoders import (
    MambaTemporalEncoder,
    RNNEncoder,
    TCNTemporalEncoder,
    TransformerTemporalEncoder,
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
    # [P0-1] Apply ADJ_SIGMA_FACTOR to enlarge σ for 60s positional noise robustness.
    # σ_effective = σ_base × ADJ_SIGMA_FACTOR (default 1.5)
    base_sigma = float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125))
    sigma_factor = float(getattr(cfg, "ADJ_SIGMA_FACTOR", 1.0))
    effective_sigma = base_sigma * sigma_factor

    if multiscale:
        A = build_multiscale_adjacency_from_xy(
            xy,
            team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
            add_self_loops=True,
            alive=alive,
            clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
        )
    else:
        A = build_adjacency_from_xy(
            xy,
            soft=bool(getattr(cfg, "ADJ_SOFT", True)),
            sigma=effective_sigma,
            team_edge_weight=float(getattr(cfg, "TEAM_EDGE_WEIGHT", 1.0)),
            add_self_loops=True,
            alive=alive,
            clamp_min=float(getattr(cfg, "ADJ_CLAMP_MIN", 1e-4)),
        )

    # [P1-LOGIC-4 FIX] Delegate to _get_role_adj_module() — single init path.
    if bool(getattr(cfg, "USE_ROLE_AWARE_ADJ", False)):
        role_mod = _get_role_adj_module(A.device)
        A = role_mod(A)

    return A


# Backward-compatible alias: existing code calls `with _autocast_disabled():`
@contextmanager
def _autocast_disabled():
    """Thin wrapper â€" delegates to common_torch.autocast_disabled()."""
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


# =========================================================
# NODE_IDX â€" single source of truth via common_torch
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

        # [P0-3] Optional input projection for full-info x_seq access.
        # When USE_INPUT_PROJECTION=True, the model can process x_seq (~997-dim)
        # through a projection: Linear(d_in, proj_dim) → LayerNorm → ReLU
        # This gives BiGRU access to all 10 players' individual features.
        self._use_proj = bool(getattr(cfg, "USE_INPUT_PROJECTION", False))
        proj_dim = int(getattr(cfg, "INPUT_PROJ_DIM", 256))
        if self._use_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(d_in, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(drop),
            )
            enc_d_in = proj_dim
        else:
            self.input_proj = None
            enc_d_in = d_in

        if kind == "ugru":
            self.enc = RNNEncoder("gru", enc_d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=False, dropout=drop)
        elif kind == "bigru":
            self.enc = RNNEncoder("gru", enc_d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind == "ulstm":
            self.enc = RNNEncoder("lstm", enc_d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=False, dropout=drop)
        elif kind == "bilstm":
            self.enc = RNNEncoder("lstm", enc_d_in, rnn_hidden, n_layers=rnn_layers, bidirectional=True, dropout=drop)
        elif kind == "transformer":
            self.enc = TransformerTemporalEncoder(
                d_in=enc_d_in,
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
                d_in=enc_d_in,
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
                d_in=enc_d_in,
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
        # [P0-3] Apply input projection if enabled
        if self.input_proj is not None:
            x = self.input_proj(x)
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
    # [P0-1] Apply ADJ_SIGMA_FACTOR for noise-robust adjacency
    base_sigma = float(getattr(cfg, "ADJ_SIGMA_NORM", 0.125))
    sigma_factor = float(getattr(cfg, "ADJ_SIGMA_FACTOR", 1.0))
    effective_sigma = base_sigma * sigma_factor

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
        sigma=effective_sigma,
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
        self.cont_dim = max(5, int(getattr(cfg, "EVENT_CONT_DIM", 12)))
        self.type_emb = nn.Embedding(type_vocab, d_model)
        self.actor_emb = nn.Embedding(11, d_model)      # 0..10
        self.team_emb  = nn.Embedding(3, d_model)       # 0 blue, 1 red, 2 unk
        self.cont_proj = nn.Linear(self.cont_dim, d_model)
        self.drop = nn.Dropout(float(getattr(cfg, "DROPOUT", 0.1)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, etype, eactor, eteam, econt):
        econt_f = econt.float()
        if econt_f.size(-1) > self.cont_dim:
            econt_f = econt_f[..., :self.cont_dim]
        elif econt_f.size(-1) < self.cont_dim:
            pad = self.cont_dim - int(econt_f.size(-1))
            econt_f = F.pad(econt_f, (0, pad), mode="constant", value=0.0)
        x = (
            self.type_emb(etype.long())
            + self.actor_emb(eactor.long().clamp(0, 10))
            + self.team_emb(eteam.long().clamp(0, 2))
            + self.cont_proj(econt_f)
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
        self._importance_pool = bool(getattr(cfg, "XATTN_IMPORTANCE_POOL", True))
        self._importance_prior_idx = int(getattr(cfg, "EVENT_CONT_IMPORTANCE_PRIOR_IDX", 11))
        self._importance_prior_boost = float(getattr(cfg, "XATTN_PRIOR_BOOST", 1.25))
        self.imp_mlp = None
        if self._importance_pool:
            imp_hidden = int(getattr(cfg, "XATTN_IMPORTANCE_HIDDEN", 64))
            imp_hidden = max(16, imp_hidden)
            self.imp_mlp = nn.Sequential(
                nn.Linear(2 * d_model, imp_hidden),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(imp_hidden, 1),
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
        m = emask.float().clamp(0.0, 1.0)

        attn_alpha = None
        if self.imp_mlp is not None:
            imp_in = torch.cat([attn_out, e_emb], dim=-1)
            imp_logits = self.imp_mlp(imp_in).squeeze(-1)

            if econt is not None and econt.ndim == 3:
                j = int(self._importance_prior_idx)
                if 0 <= j < int(econt.shape[-1]):
                    prior = econt[..., j].float()
                    imp_logits = imp_logits + self._importance_prior_boost * prior

            neg_inf = torch.full_like(imp_logits, -1.0e4)
            imp_logits = torch.where(m > 0.0, imp_logits, neg_inf)
            attn_alpha = torch.softmax(imp_logits, dim=1)
            attn_alpha = attn_alpha * m
            attn_alpha = attn_alpha / attn_alpha.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            cross_pool = (attn_out * attn_alpha.unsqueeze(-1)).sum(dim=1)
        else:
            m_exp = m.unsqueeze(-1)
            denom = m_exp.sum(dim=1).clamp_min(1.0)
            cross_pool = (attn_out * m_exp).sum(dim=1) / denom

        fused = torch.cat([z_last, cross_pool], dim=-1)
        logit = self.head(fused)

        if return_aux:
            return logit, {
                "z_seq": z_seq,
                "z_last": z_last,
                "event_emb": e_emb,
                "attn_out": attn_out,
                "cross_pool": cross_pool,
                "event_alpha": attn_alpha,
            }
        return logit


# =========================================================
# Layered fusion: global(BiGRU) + champion-relation(GNN) + pre-fight events(attention)
# =========================================================
class LayeredFusionGNNBiGRUXAttn(nn.Module, DebugHookMixin):
    """Tri-branch fusion model aligned with domain priors.

    1) Global branch: global-feature prefix -> selectable temporal encoder
    2) Champion relation/status branch: last-frame GNN over node graph
    3) Event branch: event tokens -> self-attention -> importance pooling
    4) Late fusion with learned branch gates
    """

    def __init__(
        self,
        f_node: int,
        d_seq: int,
        use_lgbm_logit: bool = True,
        *,
        global_kind: Optional[str] = None,
        gnn_kind: Optional[str] = None,
        event_kind: Optional[str] = None,
    ):
        super().__init__()
        self.use_lgbm_logit = bool(use_lgbm_logit)

        gnn_dim = int(getattr(cfg, "GNN_DIM", 64))
        gnn_drop = float(getattr(cfg, "GNN_DROPOUT", 0.1))
        gnn_norm = bool(getattr(cfg, "GNN_NORM", True))
        rnn_hidden = int(getattr(cfg, "RNN_HIDDEN", 128))
        rnn_layers = int(getattr(cfg, "RNN_LAYERS", 1))
        drop = float(getattr(cfg, "DROPOUT", 0.1))

        gnn_kind_eff = gnn_kind if gnn_kind is not None else str(getattr(cfg, "LAYER_FUSION_GNN_KIND", "graphsage"))
        self.gnn_kind = str(gnn_kind_eff).lower().strip()
        if self.gnn_kind not in ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn"):
            self.gnn_kind = "graphsage"
        self.multiscale_adj = bool(getattr(cfg, "LAYER_FUSION_GNN_MULTISCALE_ADJ", False))

        self.node_adapter = NodeFeatureAdapter(f_in=f_node, d_out=gnn_dim)
        self.gnn = GraphEncoder(
            kind=self.gnn_kind,
            d_in=gnn_dim,
            d_h=gnn_dim,
            n_layers=2,
            dropout=gnn_drop,
            use_norm=gnn_norm,
        )
        self.gnn_out_dim = 3 * gnn_dim

        phase_dim = 3 if bool(getattr(cfg, "USE_GAME_PHASE", False)) else 0
        gdim_cfg = int(getattr(cfg, "LAYER_FUSION_GLOBAL_DIM", 0))
        self.global_in_dim = int(gdim_cfg if gdim_cfg > 0 else (F_GLOBAL + phase_dim))
        self.global_in_dim = max(1, self.global_in_dim)
        self.global_kind = str(
            global_kind if global_kind is not None else getattr(cfg, "LAYER_FUSION_GLOBAL_KIND", "bigru")
        ).lower().strip()
        if self.global_kind not in ("ugru", "bigru", "ulstm", "bilstm", "transformer", "tcn", "mamba"):
            self.global_kind = "bigru"

        if self.global_kind == "ugru":
            self.global_temporal = RNNEncoder(
                rnn_type="gru",
                d_in=self.global_in_dim,
                d_h=rnn_hidden,
                n_layers=rnn_layers,
                bidirectional=False,
                dropout=drop,
            )
        elif self.global_kind == "bigru":
            self.global_temporal = RNNEncoder(
                rnn_type="gru",
                d_in=self.global_in_dim,
                d_h=rnn_hidden,
                n_layers=rnn_layers,
                bidirectional=True,
                dropout=drop,
            )
        elif self.global_kind == "ulstm":
            self.global_temporal = RNNEncoder(
                rnn_type="lstm",
                d_in=self.global_in_dim,
                d_h=rnn_hidden,
                n_layers=rnn_layers,
                bidirectional=False,
                dropout=drop,
            )
        elif self.global_kind == "bilstm":
            self.global_temporal = RNNEncoder(
                rnn_type="lstm",
                d_in=self.global_in_dim,
                d_h=rnn_hidden,
                n_layers=rnn_layers,
                bidirectional=True,
                dropout=drop,
            )
        elif self.global_kind == "transformer":
            self.global_temporal = TransformerTemporalEncoder(
                d_in=self.global_in_dim,
                d_model=int(getattr(cfg, "TRANS_D_MODEL", rnn_hidden)),
                nhead=int(getattr(cfg, "TRANS_NHEAD", 4)),
                n_layers=int(getattr(cfg, "TRANS_LAYERS", 2)),
                dropout=drop,
                max_len=int(getattr(cfg, "MAX_SEQ_LEN", 512)),
            )
        elif self.global_kind == "tcn":
            self.global_temporal = TCNTemporalEncoder(
                d_in=self.global_in_dim,
                channels=int(getattr(cfg, "TCN_CHANNELS", max(64, rnn_hidden))),
                n_levels=int(getattr(cfg, "TCN_LEVELS", 3)),
                kernel=int(getattr(cfg, "TCN_KERNEL", 3)),
                dropout=float(getattr(cfg, "TCN_DROPOUT", drop)),
            )
        else:
            self.global_temporal = MambaTemporalEncoder(
                d_in=self.global_in_dim,
                d_model=int(getattr(cfg, "MAMBA_D_MODEL", rnn_hidden)),
                n_layers=int(getattr(cfg, "MAMBA_LAYERS", 3)),
                d_state=int(getattr(cfg, "MAMBA_D_STATE", 16)),
                dropout=drop,
            )
        self.global_out_dim = self.global_temporal.out_dim

        d_model = int(getattr(cfg, "LAYER_FUSION_EVENT_D_MODEL", int(getattr(cfg, "XATTN_D_MODEL", 128))))
        nhead = int(getattr(cfg, "LAYER_FUSION_EVENT_NHEAD", int(getattr(cfg, "XATTN_NHEAD", 4))))
        self.event_kind = str(
            event_kind if event_kind is not None else getattr(cfg, "LAYER_FUSION_EVENT_KIND", "attn")
        ).lower().strip()
        if self.event_kind not in ("attn", "xattn", "mean"):
            self.event_kind = "attn"
        self.ev_enc = EventTokenEncoder(d_model=d_model)
        self.ev_attn = None
        self.ev_importance = None
        if self.event_kind in ("attn", "xattn"):
            self.ev_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=max(1, nhead),
                dropout=drop,
                batch_first=True,
            )
            imp_hidden = int(getattr(cfg, "LAYER_FUSION_EVENT_IMP_HIDDEN", 64))
            imp_hidden = max(16, imp_hidden)
            self.ev_importance = nn.Sequential(
                nn.Linear(2 * d_model, imp_hidden),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(imp_hidden, 1),
            )
        self.ev_prior_idx = int(getattr(cfg, "EVENT_CONT_IMPORTANCE_PRIOR_IDX", 11))
        self.ev_prior_boost = float(getattr(cfg, "XATTN_PRIOR_BOOST", 1.25))
        self.event_out_dim = d_model

        fuse_dim = int(
            getattr(
                cfg,
                "LAYER_FUSION_FUSE_DIM",
                max(self.gnn_out_dim, self.global_out_dim, self.event_out_dim),
            )
        )
        self.gnn_proj = nn.Linear(self.gnn_out_dim, fuse_dim) if self.gnn_out_dim != fuse_dim else nn.Identity()
        self.global_proj = nn.Linear(self.global_out_dim, fuse_dim) if self.global_out_dim != fuse_dim else nn.Identity()
        self.event_proj = nn.Linear(self.event_out_dim, fuse_dim) if self.event_out_dim != fuse_dim else nn.Identity()

        gate_h = int(getattr(cfg, "LAYER_FUSION_GATE_H", 64))
        gate_h = max(16, gate_h)
        self.fuse_gate = nn.Sequential(
            nn.Linear(3 * fuse_dim, gate_h),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(gate_h, 3),
        )

        head_hidden = int(getattr(cfg, "HEAD_HIDDEN", int(getattr(cfg, "RNN_HIDDEN", 128))))
        head_layers = int(getattr(cfg, "HEAD_LAYERS", 2))
        head_in = int(fuse_dim + (1 if self.use_lgbm_logit else 0))
        self.head = MLP(head_in, head_hidden, 1, dropout=drop, layers=head_layers)

    def _global_prefix(self, temporal_seq: torch.Tensor) -> torch.Tensor:
        # Contract: global features are placed at sequence prefix in macro/extra sequences.
        if temporal_seq.shape[-1] >= self.global_in_dim:
            return temporal_seq[..., :self.global_in_dim]
        pad_dim = int(self.global_in_dim - int(temporal_seq.shape[-1]))
        pad = temporal_seq.new_zeros((*temporal_seq.shape[:-1], pad_dim))
        return torch.cat([temporal_seq, pad], dim=-1)

    def _encode_node_gnn(self, node_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw = node_seq[:, -1, :, :]
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

        pooled = pool_team_repr(h, alive=alive)
        return pooled, h, A

    def _encode_event_attention(
        self,
        etype: torch.Tensor,
        eactor: torch.Tensor,
        eteam: torch.Tensor,
        econt: torch.Tensor,
        emask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_emb = self.ev_enc(etype, eactor, eteam, econt)
        m = emask.float().clamp(0.0, 1.0)
        if self.event_kind == "mean":
            alpha = m / m.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            event_feat = (e_emb * alpha.unsqueeze(-1)).sum(dim=1)
            return event_feat, alpha

        if self.ev_attn is None or self.ev_importance is None:
            alpha = m / m.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            event_feat = (e_emb * alpha.unsqueeze(-1)).sum(dim=1)
            return event_feat, alpha

        key_padding = (m <= 0.0)
        with _autocast_disabled():
            q = _nan_to_num_(e_emb.float())
            k = _nan_to_num_(e_emb.float())
            v = k
            e_ctx, _ = self.ev_attn(q, k, v, key_padding_mask=key_padding, need_weights=False)
            e_ctx = _nan_to_num_(e_ctx)
        e_ctx = e_ctx.to(dtype=e_emb.dtype)

        imp_logits = self.ev_importance(torch.cat([e_ctx, e_emb], dim=-1)).squeeze(-1)
        if econt is not None and econt.ndim == 3:
            j = int(self.ev_prior_idx)
            if 0 <= j < int(econt.shape[-1]):
                prior = econt[..., j].float()
                imp_logits = imp_logits + self.ev_prior_boost * prior

        neg_inf = torch.full_like(imp_logits, -1.0e4)
        imp_logits = torch.where(m > 0.0, imp_logits, neg_inf)
        alpha = torch.softmax(imp_logits, dim=1)
        alpha = alpha * m
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        event_feat = (e_ctx * alpha.unsqueeze(-1)).sum(dim=1)
        return event_feat, alpha

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        node_seq = batch.get("node_seq", None)
        if node_seq is None:
            raise KeyError("LayeredFusionGNNBiGRUXAttn requires node_seq")

        temporal_seq, seq_key = pick_temporal_seq(batch)
        global_src = batch.get("global_seq", None)
        if global_src is None:
            global_src = temporal_seq
        global_in = self._global_prefix(global_src)

        etype = batch.get("event_type", None)
        eactor = batch.get("event_actor", None)
        eteam = batch.get("event_team", None)
        econt = batch.get("event_cont", None)
        emask = batch.get("event_mask", None)
        if any(x is None for x in (etype, eactor, eteam, econt, emask)):
            raise KeyError("LayeredFusionGNNBiGRUXAttn requires event_type/event_actor/event_team/event_cont/event_mask")

        gnn_feat, h, A = self._encode_node_gnn(node_seq)
        self._maybe_store_debug(A, h)
        global_feat = self.global_temporal(global_in)
        event_feat, event_alpha = self._encode_event_attention(etype, eactor, eteam, econt, emask)

        g = self.gnn_proj(gnn_feat)
        r = self.global_proj(global_feat)
        e = self.event_proj(event_feat)
        fusion_input = torch.cat([g, r, e], dim=-1)
        gate = torch.softmax(self.fuse_gate(fusion_input), dim=-1)
        fused = gate[:, 0:1] * g + gate[:, 1:2] * r + gate[:, 2:3] * e

        if self.use_lgbm_logit and (batch.get("lgbm_logit", None) is not None):
            llog = batch["lgbm_logit"]
            if isinstance(llog, torch.Tensor):
                if llog.ndim == 1:
                    llog = llog.unsqueeze(-1)
            else:
                llog = torch.as_tensor(llog).unsqueeze(-1)
            fused = torch.cat([fused, llog.to(device=fused.device, dtype=fused.dtype)], dim=-1)

        logit = self.head(fused)
        if return_aux:
            return logit, {
                "seq_key": seq_key,
                "global_kind": self.global_kind,
                "gnn_kind": self.gnn_kind,
                "event_kind": self.event_kind,
                "global_in": global_in,
                "gnn_feat": gnn_feat,
                "global_feat": global_feat,
                "event_feat": event_feat,
                "branch_gate": gate,
                "event_alpha": event_alpha,
                "adj": A,
            }
        return logit


# =========================================================
# Model registry / factory (extracted)
# =========================================================
from train.model_registry import (
    ModelSpec,
    build_model,
    infer_dims_from_batch,
    register_model,
    sanitize_feature_name,
    sanitize_feature_names,
)
