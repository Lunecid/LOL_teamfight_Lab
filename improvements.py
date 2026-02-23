"""
improvements.py — Domain-Knowledge-Driven Enhancements for LoL Teamfight Prediction
=====================================================================================

This module implements 7 targeted improvements grounded in both
statistical learning theory and League of Legends domain expertise.

Mathematical Foundations
------------------------
1. Focal Loss:       ℒ_FL = -α_t (1 - p_t)^γ log(p_t)
2. Game Phase:       φ(t) = [σ((14-t)/τ), σ((t-10)/τ)·σ((28-t)/τ), σ((t-22)/τ)]
3. Attention Pool:   h_pool = Σ_t α_t · h_t,  α_t ∝ exp(w^T tanh(W_a h_t))
4. Momentum:         μ_short = (1/k)Σ Δx_{t-i},  δ = μ_short - μ_long
5. Role Adjacency:   A^role_ij = A^dist_ij · R_{role(i), role(j)}
6. Multi-Task:       ℒ = ℒ_fight + λ₁ℒ_gold + λ₂ℒ_kill
7. Label Smoothing:  y_smooth = y·(1-ε) + ε/2

Author: Improvement Module for LOL Teamfight Prediction Project
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
from config import cfg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════
#
# Standard BCE treats all samples equally:
#   ℒ_BCE = -[y log(p) + (1-y) log(1-p)]
#
# Focal Loss down-weights easy examples:
#   ℒ_FL = -α_t (1 - p_t)^γ log(p_t)
#
# where p_t = σ(z) if y=1, else 1 - σ(z).
#
# For teamfight prediction:
#   γ = 2.0:  A sample predicted at p=0.9 correctly gets weight (0.1)^2 = 0.01
#             vs 1.0 for standard BCE. This forces learning on hard/close fights.
#   α = 0.25: Mild class balancing (adjustable based on label distribution)
#
# Domain justification: Teamfights with >10k gold diff are easy to predict
# and contribute noise gradients. Focal loss focuses on the informative
# margin cases where compositions and positioning matter most.
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logit inputs.

    Parameters
    ----------
    gamma : float
        Focusing parameter. γ=0 reduces to BCE. γ=2 is the standard choice.
        Higher γ = stronger down-weighting of easy examples.
    alpha : float
        Class balancing factor for positive class. α ∈ (0, 1).
        α = 0.5 means no balancing.
    label_smoothing : float
        If > 0, applies label smoothing: y_smooth = y·(1-ε) + ε/2
    reduction : str
        'mean' | 'sum' | 'none'

    Mathematical Details
    --------------------
    Let z = logit, p = σ(z).

    For numerical stability, we compute directly from logits:
      log(p_t) = -log(1 + exp(-z))     if y=1
      log(p_t) = -z - log(1 + exp(-z)) if y=0

    The focal weight (1 - p_t)^γ is computed as:
      p_t = p if y=1, else (1-p)
      (1 - p_t) = (1-p) if y=1, else p

    Combined:
      ℒ = -α_t · (1 - p_t)^γ · log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B,) or (B,1) raw logits (before sigmoid)
        targets : (B,) or (B,1) binary labels ∈ {0, 1}
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # ── Label Smoothing ──
        # y_smooth = y · (1 - ε) + ε/2
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + eps / 2.0

        # ── Numerically stable focal loss computation ──
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)

        focal_weight = (1.0 - p_t).pow(self.gamma)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════
# 2. GAME PHASE ENCODING
# ═══════════════════════════════════════════════════════════════════
#
# League of Legends has three distinct strategic phases:
#   Early (0-14min):  Laning, 1v1/2v2 skirmishes
#   Mid   (14-25min): Objective contests, rotations, small teamfights
#   Late  (25min+):   Full 5v5 teamfights, death timers are critical
#
# We encode this as a soft 3-dim vector:
#   φ_early(t) = σ((14 - t) / τ)
#   φ_mid(t)   = σ((t - 10) / τ) · σ((28 - t) / τ)
#   φ_late(t)  = σ((t - 22) / τ)
#
# The overlap zones (10-14min, 22-28min) create smooth transitions.
# τ = 3.0 gives a ~6-minute soft boundary.
# ═══════════════════════════════════════════════════════════════════

def compute_game_phase_encoding(
    time_minutes: float,
    tau: float = 3.0,
) -> np.ndarray:
    """
    Compute soft game phase encoding for a single timestep.

    Parameters
    ----------
    time_minutes : float
        Game time in minutes.
    tau : float
        Softness of phase transitions (higher = softer boundaries).

    Returns
    -------
    phase : (3,) ndarray
        [φ_early, φ_mid, φ_late] ∈ (0, 1)^3
    """
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    t = float(time_minutes)
    early = _sigmoid((14.0 - t) / tau)
    mid = _sigmoid((t - 10.0) / tau) * _sigmoid((28.0 - t) / tau)
    late = _sigmoid((t - 22.0) / tau)

    return np.array([early, mid, late], dtype=np.float32)


def compute_game_phase_seq(
    time_norm_seq: np.ndarray,
    total_game_minutes: float,
    tau: float = 3.0,
) -> np.ndarray:
    """
    Compute game phase encoding for an entire temporal sequence.

    Parameters
    ----------
    time_norm_seq : (L,) ndarray
        Normalized time values ∈ [0, 1] from the global features.
    total_game_minutes : float
        Total game duration in minutes.
    tau : float
        Phase transition softness.

    Returns
    -------
    phase_seq : (L, 3) ndarray
        Game phase encoding for each timestep.
    """
    L = len(time_norm_seq)
    out = np.zeros((L, 3), dtype=np.float32)
    for i in range(L):
        t_min = float(time_norm_seq[i]) * total_game_minutes
        out[i] = compute_game_phase_encoding(t_min, tau)
    return out


# ═══════════════════════════════════════════════════════════════════
# 3. ATTENTION-BASED TEMPORAL POOLING
# ═══════════════════════════════════════════════════════════════════
#
# Current: z = h_T  (last hidden state only)
# Problem: Critical events at t << T (e.g., early kills, gold spikes)
#          have decayed through the RNN and may be underrepresented.
#
# Proposed: Additive attention pooling (Bahdanau et al., 2015)
#
#   e_t = w^T tanh(W_a h_t + b_a)        (attention energy)
#   α_t = softmax_t(e_t)                  (attention weight)
#   h_pool = Σ_t α_t · h_t               (weighted sum)
#   z = [h_T ‖ h_pool]                    (concatenation, optional)
#
# This allows the model to attend to arbitrary timesteps.
# The learned attention weights α_t are also interpretable:
# they reveal which timesteps the model considers most important.
#
# ───────────────────────────────────────────────────────────────
# [FIX] API contract for dual usage:
#
# Usage A — models.py (simple context-only, backward compat):
#   self.attn_pool = TemporalAttentionPooling(rnn_out_dim, attn_dim)
#   h_attn = self.attn_pool(out)       → unpacks first element of tuple
#
# Usage B — EnhancedRNNEncoder (concat_last mode):
#   self.attn_pool = TemporalAttentionPooling(d_rnn_out, d_attn, concat_last=True)
#   z, weights = self.attn_pool(h_seq, mask)
#
# forward() always returns Tuple[Tensor, Tensor] = (output, α_weights)
# ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

class TemporalAttentionPooling(nn.Module):
    """
    Attention-based temporal pooling with optional concat_last mode.

    수학적 정의:
        e_t = w^T tanh(W_a · h_t + b_a)           ∈ ℝ
        α_t = softmax(e_1, ..., e_T)_t             ∈ [0, 1], Σα=1
        c   = Σ_{t=1}^{T} α_t · h_t               ∈ ℝ^{d_h}

    concat_last=True 일 때:
        output = [h_T ‖ c]                         ∈ ℝ^{2·d_h}
    concat_last=False 일 때:
        output = c                                 ∈ ℝ^{d_h}

    Parameters
    ----------
    input_dim : int
        RNN hidden state dimension (d_h or d_h * num_directions).
    attn_dim : int
        Attention bottleneck dimension for W_a projection.
    concat_last : bool
        If True, output = [h_T ‖ c] (dim doubles).
        If False, output = c only.
    """

    def __init__(
        self,
        input_dim: int,
        attn_dim: int = 64,
        concat_last: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.concat_last = concat_last

        self.W_a = nn.Linear(input_dim, attn_dim, bias=True)   # W_a h_t + b_a
        self.w = nn.Linear(attn_dim, 1, bias=False)            # w^T (·)

        # Output dimension depends on concat_last mode
        self.out_dim = input_dim * 2 if concat_last else input_dim

    def forward(
        self,
        h_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_seq : (B, L, D) — RNN 전체 시퀀스 출력
        mask  : (B, L) or None — True/1 = valid, False/0 = padding

        Returns
        -------
        output  : (B, D) or (B, 2D) — attention-weighted representation
        weights : (B, L) — attention weights α_t (for interpretability)
        """
        # e_t = w^T tanh(W_a · h_t + b_a)
        e = self.w(torch.tanh(self.W_a(h_seq)))  # (B, L, 1)
        e = e.squeeze(-1)                         # (B, L)

        # Mask out padding positions
        if mask is not None:
            e = e.masked_fill(~mask.bool(), float('-inf'))

        # α_t = softmax(e)
        alpha = torch.softmax(e, dim=1)  # (B, L)

        # c = Σ_t α_t · h_t
        context = torch.bmm(
            alpha.unsqueeze(1),  # (B, 1, L)
            h_seq                # (B, L, D)
        ).squeeze(1)             # (B, D)

        # ── Output mode ──
        if self.concat_last:
            h_last = h_seq[:, -1, :]                       # (B, D)
            output = torch.cat([h_last, context], dim=-1)   # (B, 2D)
        else:
            output = context                                # (B, D)

        return output, alpha


# ═══════════════════════════════════════════════════════════════════
# 4. GOLD/STAT MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════
#
# Current: delta = x[-1] - x[0], slope = linear regression
# Problem: No distinction between long-term trend and recent momentum.
#
# We add MACD-like features:
#   μ_short(t) = (1/k) Σ_{i=0}^{k-1} Δx_{t-i}     (k=3, ~15 seconds)
#   μ_long(t)  = (1/T) Σ_{i=1}^{T} Δx_i
#   δ_momentum = μ_short - μ_long                    (momentum divergence)
# ═══════════════════════════════════════════════════════════════════

def compute_momentum_features(
    x_seq: np.ndarray,
    short_window: int = 3,
) -> np.ndarray:
    """
    Compute momentum features from a temporal feature sequence.

    Parameters
    ----------
    x_seq : (L, D) ndarray
        Temporal feature sequence.
    short_window : int
        Number of recent steps for short-term momentum.

    Returns
    -------
    momentum : (3*D,) ndarray
        [μ_short, μ_long, δ_momentum] concatenated.

    Mathematical Details
    --------------------
    For each feature dimension d:

      Δx_t = x_t - x_{t-1}                             (first difference)

      μ_short = (1/k) Σ_{t=T-k+1}^{T} Δx_t             (recent k-step mean change)
      μ_long  = (1/(T-1)) Σ_{t=2}^{T} Δx_t              (overall trend)
      δ       = μ_short - μ_long                         (momentum divergence)

    δ > 0: Feature is accelerating (recent gains exceed average)
    δ < 0: Feature is decelerating (recent gains below average)
    δ ≈ 0: Steady trend
    """
    L, D = x_seq.shape

    if L < 2:
        return np.zeros(3 * D, dtype=np.float32)

    diffs = np.diff(x_seq, axis=0)  # (L-1, D)

    mu_long = diffs.mean(axis=0)  # (D,)

    k = min(short_window, diffs.shape[0])
    mu_short = diffs[-k:].mean(axis=0)  # (D,)

    delta_momentum = mu_short - mu_long  # (D,)

    return np.concatenate([mu_short, mu_long, delta_momentum], axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 5. ROLE-AWARE ASYMMETRIC ADJACENCY
# ═══════════════════════════════════════════════════════════════════
#
# A^role_ij = A^dist_ij · softplus(R_{role(i), role(j)})
#
# R ∈ ℝ^{5×5}: learnable, softplus ensures non-negativity.
# ═══════════════════════════════════════════════════════════════════

class RoleAwareAdjacency(nn.Module):
    """
    역할 기반 인접 행렬 가중치.

    수학적 정의:
        A'_{ij} = A^{dist}_{ij} · softplus(R_{role(i), role(j)})

    Parameters
    ----------
    n_roles : int
        역할 수 (LoL = 5: TOP, JNG, MID, BOT, SUP)
    init_value : float
        R의 초기값. 0.0이면 softplus(0)=ln(2)≈0.693
    """

    def __init__(self, n_roles: int = 5, init_value: float = 0.0):
        super().__init__()
        self.R = nn.Parameter(torch.full((n_roles, n_roles), init_value))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        A : (B, N, N) — 거리 기반 인접 행렬, N=10

        Returns
        -------
        A' : (B, N, N) — 역할 가중치 적용된 인접 행렬
        """
        N = A.shape[-1]
        role_idx = torch.arange(N, device=A.device) % self.R.shape[0]
        R_ij = self.R[role_idx.unsqueeze(1), role_idx.unsqueeze(0)]
        weight = F.softplus(R_ij).unsqueeze(0)
        return A * weight

    def get_interaction_matrix(self) -> torch.Tensor:
        """Return the softplus-transformed R matrix for interpretation."""
        return F.softplus(self.R).detach().cpu()


# ═══════════════════════════════════════════════════════════════════
# 6. MULTI-TASK AUXILIARY LOSS
# ═══════════════════════════════════════════════════════════════════

class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head: fight outcome + auxiliary regressions.

    Outputs
    -------
    Dict with keys:
      'fight_logit' : (B, 1)
      'gold_pred'   : (B, 1)
      'kill_pred'   : (B, 1)
    """

    def __init__(self, d_in: int, d_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fight_head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        self.gold_head = nn.Sequential(
            nn.Linear(d_in, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self.kill_head = nn.Sequential(
            nn.Linear(d_in, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "fight_logit": self.fight_head(z),
            "gold_pred": self.gold_head(z),
            "kill_pred": self.kill_head(z),
        }


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task training.

    ℒ_total = ℒ_fight + λ₁·ℒ_gold + λ₂·ℒ_kill

    Optionally uses uncertainty weighting (Kendall et al., 2018):
    ℒ_total = (1/2σ₁²)·ℒ₁ + (1/2σ₂²)·ℒ₂ + (1/2σ₃²)·ℒ₃ + log(σ₁σ₂σ₃)
    """

    def __init__(
        self,
        lambda_gold: float = 0.1,
        lambda_kill: float = 0.05,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.05,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.lambda_gold = lambda_gold
        self.lambda_kill = lambda_kill
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.focal_loss = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing,
        )

        if use_uncertainty_weighting:
            self.log_var_fight = nn.Parameter(torch.zeros(1))
            self.log_var_gold = nn.Parameter(torch.zeros(1))
            self.log_var_kill = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_fight = self.focal_loss(predictions["fight_logit"], targets["y"])

        loss_gold = torch.tensor(0.0, device=loss_fight.device)
        if "gold_diff" in targets and targets["gold_diff"] is not None:
            loss_gold = F.mse_loss(
                predictions["gold_pred"].view(-1), targets["gold_diff"].view(-1).float(),
            )

        loss_kill = torch.tensor(0.0, device=loss_fight.device)
        if "kill_diff" in targets and targets["kill_diff"] is not None:
            loss_kill = F.mse_loss(
                predictions["kill_pred"].view(-1), targets["kill_diff"].view(-1).float(),
            )

        if self.use_uncertainty_weighting:
            precision_fight = torch.exp(-self.log_var_fight)
            precision_gold = torch.exp(-self.log_var_gold)
            precision_kill = torch.exp(-self.log_var_kill)
            total = (
                precision_fight * loss_fight + self.log_var_fight
                + precision_gold * loss_gold + self.log_var_gold
                + precision_kill * loss_kill + self.log_var_kill
            )
        else:
            total = loss_fight + self.lambda_gold * loss_gold + self.lambda_kill * loss_kill

        loss_dict = {
            "loss_fight": float(loss_fight.item()),
            "loss_gold": float(loss_gold.item()),
            "loss_kill": float(loss_kill.item()),
            "loss_total": float(total.item()),
        }
        return total, loss_dict


# ═══════════════════════════════════════════════════════════════════
# 7. ENHANCED RNN ENCODER WITH ALL IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════════

class EnhancedRNNEncoder(nn.Module):
    """
    Enhanced RNN encoder with attention-based temporal pooling.

    Architecture:
      Input (B, T, d_in)
        → LayerNorm → RNN (BiGRU/BiLSTM)
        → [h_T ‖ AttentionPool(h_1...h_T)]
        → LayerNorm → output
    """

    def __init__(
        self,
        rnn_type: str = "gru",
        d_in: int = 128,
        d_hidden: int = 128,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        use_attention: bool = True,
        d_attn: int = 64,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.input_norm = nn.LayerNorm(d_in)

        rnn_drop = dropout if n_layers > 1 else 0.0
        RNNClass = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = RNNClass(
            d_in, d_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=rnn_drop, bidirectional=bidirectional,
        )

        d_rnn_out = d_hidden * (2 if bidirectional else 1)

        if use_attention:
            # [FIX] correct parameter names: input_dim, attn_dim, concat_last
            self.attn_pool = TemporalAttentionPooling(
                input_dim=d_rnn_out,
                attn_dim=d_attn,
                concat_last=True,
            )
            self.out_dim = d_rnn_out * 2  # [h_T ‖ h_pool]
        else:
            self.attn_pool = None
            self.out_dim = d_rnn_out

        self.output_norm = nn.LayerNorm(self.out_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_norm(x)
        h_seq, _ = self.rnn(x)  # (B, T, d_rnn_out)

        if self.use_attention and self.attn_pool is not None:
            # [FIX] forward returns (output, weights) tuple
            z, _attn_weights = self.attn_pool(h_seq, mask)
        else:
            z = h_seq[:, -1, :]

        z = self.output_norm(z)
        return z


# ═══════════════════════════════════════════════════════════════════
# ENHANCED seq_to_tabular WITH MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════

def enhanced_seq_to_tabular(
    x_seq: np.ndarray,
    short_window: int = 3,
) -> np.ndarray:
    """
    Enhanced sequence-to-tabular conversion with momentum features.

    Original: [last, mean, std, min, max, delta, slope]  → 7D per feature
    Enhanced: + [μ_short, μ_long, δ_momentum]            → 10D per feature
    """
    L, D = x_seq.shape
    t = np.arange(L, dtype=np.float32)

    if L >= 2:
        t0 = t - t.mean()
        denom = float((t0 * t0).sum() + 1e-6)
        slope = (t0[:, None] * (x_seq - x_seq.mean(axis=0))).sum(axis=0) / denom
    else:
        slope = np.zeros((D,), dtype=np.float32)

    last = x_seq[-1]
    mean = x_seq.mean(axis=0)
    std = x_seq.std(axis=0)
    mn = x_seq.min(axis=0)
    mx = x_seq.max(axis=0)
    delta = x_seq[-1] - x_seq[0]

    momentum = compute_momentum_features(x_seq, short_window)

    return np.concatenate(
        [last, mean, std, mn, mx, delta, slope, momentum], axis=0,
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def create_improved_criterion(
    device: torch.device,
    gamma: float = 2.0,
    alpha: float = 0.25,
    label_smoothing: float = 0.05,
    multi_task: bool = False,
    lambda_gold: float = 0.1,
    lambda_kill: float = 0.05,
) -> nn.Module:
    """Factory function for the improved loss criterion."""
    if multi_task:
        return MultiTaskLoss(
            lambda_gold=lambda_gold, lambda_kill=lambda_kill,
            focal_gamma=gamma, focal_alpha=alpha, label_smoothing=label_smoothing,
        ).to(device)
    else:
        return FocalLoss(
            gamma=gamma, alpha=alpha, label_smoothing=label_smoothing,
        ).to(device)


# ═══════════════════════════════════════════════════════════════════
# FEATURE NAME CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GAME_PHASE_FEATURE_NAMES: List[str] = ["phase_early", "phase_mid", "phase_late"]
MOMENTUM_SUFFIXES: List[str] = ["mu_short", "mu_long", "delta_momentum"]


def compute_momentum_stats(x_seq: np.ndarray) -> np.ndarray:
    """MACD-like momentum features for tabular representation (cfg-aware)."""
    L, D = x_seq.shape
    k = min(int(getattr(cfg, "MOMENTUM_K_SHORT", 3)), L - 1)

    if L < 2:
        return np.zeros(3 * D, dtype=np.float32)

    dx = np.diff(x_seq, axis=0)
    mu_long = dx.mean(axis=0)
    mu_short = dx[-k:].mean(axis=0) if k > 0 else mu_long
    delta_mom = mu_short - mu_long

    return np.concatenate([mu_short, mu_long, delta_mom], axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Verify improvements work correctly
# ═══════════════════════════════════════════════════════════════════

def _self_test():
    """Quick self-test for all improvement modules."""
    print("=" * 60)
    print("Running improvement module self-tests...")
    print("=" * 60)

    # 1. Focal Loss
    fl = FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.05)
    logits = torch.randn(32)
    targets = torch.randint(0, 2, (32,)).float()
    loss = fl(logits, targets)
    assert loss.shape == (), f"Focal loss shape error: {loss.shape}"
    assert loss.item() >= 0, f"Focal loss negative: {loss.item()}"
    print(f"  [✓] Focal Loss: {loss.item():.4f}")

    # 2. Game Phase Encoding
    for t in [5.0, 15.0, 30.0]:
        phase = compute_game_phase_encoding(t)
        assert phase.shape == (3,), f"Phase shape error at t={t}"
        assert np.all(phase >= 0) and np.all(phase <= 1), f"Phase out of range at t={t}"
    print(f"  [✓] Game Phase Encoding: early={compute_game_phase_encoding(5.0)}")

    # 3. Attention Pooling — concat_last=True mode (used by EnhancedRNNEncoder)
    attn = TemporalAttentionPooling(input_dim=64, attn_dim=32, concat_last=True)
    h_seq = torch.randn(4, 10, 64)
    out, weights = attn(h_seq)
    assert out.shape == (4, 128), f"Attention output shape error: {out.shape}"
    assert weights.shape == (4, 10), f"Attention weights shape error: {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    print(f"  [✓] Temporal Attention Pooling (concat): out={out.shape}, weights={weights.shape}")

    # 3b. Attention Pooling — concat_last=False mode (used by models.py RNNEncoder)
    attn_simple = TemporalAttentionPooling(input_dim=64, attn_dim=32, concat_last=False)
    out_s, w_s = attn_simple(h_seq)
    assert out_s.shape == (4, 64), f"Simple attention output shape error: {out_s.shape}"
    print(f"  [✓] Temporal Attention Pooling (simple): out={out_s.shape}")

    # 4. Momentum Features
    x_seq = np.random.randn(12, 8).astype(np.float32)
    mom = compute_momentum_features(x_seq, short_window=3)
    assert mom.shape == (24,), f"Momentum shape error: {mom.shape}"
    print(f"  [✓] Momentum Features: shape={mom.shape}")

    # 5. Role-Aware Adjacency
    ra = RoleAwareAdjacency(n_roles=5)
    A_dist = torch.rand(4, 10, 10)
    A_role = ra(A_dist)
    assert A_role.shape == (4, 10, 10), f"Role adj shape error: {A_role.shape}"
    R = ra.get_interaction_matrix()
    assert R.shape == (5, 5), f"R matrix shape error: {R.shape}"
    print(f"  [✓] Role-Aware Adjacency: R={R.shape}, A_role={A_role.shape}")

    # 6. Multi-Task
    mt_head = MultiTaskHead(d_in=128, d_hidden=64)
    z = torch.randn(4, 128)
    preds = mt_head(z)
    assert preds["fight_logit"].shape == (4, 1)
    assert preds["gold_pred"].shape == (4, 1)
    print(f"  [✓] Multi-Task Head: fight={preds['fight_logit'].shape}")

    mt_loss = MultiTaskLoss(focal_gamma=2.0, label_smoothing=0.05)
    targets_mt = {
        "y": torch.randint(0, 2, (4,)).float(),
        "gold_diff": torch.randn(4),
        "kill_diff": torch.randn(4),
    }
    total, loss_dict = mt_loss(preds, targets_mt)
    assert total.shape == ()
    print(f"  [✓] Multi-Task Loss: total={total.item():.4f}, components={loss_dict}")

    # 7. Enhanced RNN Encoder
    enc = EnhancedRNNEncoder(
        rnn_type="gru", d_in=64, d_hidden=32,
        n_layers=2, bidirectional=True, use_attention=True,
    )
    x = torch.randn(4, 10, 64)
    z = enc(x)
    assert z.shape == (4, enc.out_dim), f"Enhanced RNN shape error: {z.shape}"
    print(f"  [✓] Enhanced RNN Encoder: out_dim={enc.out_dim}, z={z.shape}")

    # 8. Enhanced seq_to_tabular
    x_seq = np.random.randn(12, 5).astype(np.float32)
    tab = enhanced_seq_to_tabular(x_seq)
    expected = 10 * 5
    assert tab.shape == (expected,), f"Enhanced tabular shape error: {tab.shape} != ({expected},)"
    print(f"  [✓] Enhanced seq_to_tabular: {tab.shape}")

    print("=" * 60)
    print("All self-tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()