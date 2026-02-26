from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from core.improvements import TemporalAttentionPooling


class RNNEncoder(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        d_in: int,
        d_h: int,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.d_h = d_h

        rnn_dropout = float(dropout) if n_layers > 1 else 0.0
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                d_in,
                d_h,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                d_in,
                d_h,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(rnn_type)

        rnn_out_dim = d_h * (2 if bidirectional else 1)

        self._use_attn_pool = bool(getattr(cfg, "USE_ATTENTION_POOL", False))
        if self._use_attn_pool:
            attn_dim = int(getattr(cfg, "ATTENTION_POOL_DIM", 64))
            self.attn_pool = TemporalAttentionPooling(rnn_out_dim, attn_dim)
            self.out_dim = rnn_out_dim * 2
        else:
            self.attn_pool = None
            self.out_dim = rnn_out_dim

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_init = None
        if h0 is not None:
            num_dirs = 2 if self.bidirectional else 1
            B = h0.shape[0]
            h_expanded = h0.unsqueeze(0).expand(self.n_layers * num_dirs, B, self.d_h).contiguous()

            if self.rnn_type == "lstm":
                c0 = torch.zeros_like(h_expanded)
                h_init = (h_expanded, c0)
            else:
                h_init = h_expanded

        if h_init is not None:
            out, _ = self.rnn(x, h_init)
        else:
            out, _ = self.rnn(x)

        if self._use_attn_pool and self.attn_pool is not None:
            h_last = out[:, -1, :]
            h_attn, _attn_weights = self.attn_pool(out)
            return torch.cat([h_last, h_attn], dim=-1)

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
    """Input: (B,L,D) -> Conv1d over time (causal) -> last step embedding."""

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
        x1 = x.transpose(1, 2)
        h = self.net(x1)
        return h[:, :, -1]


class SelectiveScan(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_model))

        self.proj_dt = nn.Linear(d_model, d_model, bias=True)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)

        dt_init = torch.exp(torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_sp = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.proj_dt.bias.copy_(inv_sp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, L, D = x.shape
        N = self.d_state

        dt = F.softplus(self.proj_dt(x))
        B_t = self.proj_B(x)
        C_t = self.proj_C(x)
        A = -torch.exp(self.A_log.float())

        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(2)

        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (C_t[:, t].unsqueeze(1) * h).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + self.D * x
        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.ssm = SelectiveScan(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        x_c = x_main.transpose(1, 2)
        x_c = self.conv1d(x_c)[:, :, : x.shape[1]]
        x_c = x_c.transpose(1, 2)
        x_c = F.silu(x_c)

        y = self.ssm(x_c)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y)) + residual


class MambaTemporalEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h[:, -1, :]
