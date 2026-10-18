"""Modern tabular deep learning on the *same* engineered features (R2).

CoG 2026 review R2 rejected the paper's "deep learning underperforms"
conclusion on two grounds: no modern tabular architecture was tried, and
the deep models never received the feature engineering LightGBM got.  This
runs all three learners on identical rows, identical splits, and identical
engineered features:

    lightgbm    the paper's tabular baseline
    mlp         same flat feature vector, deep learner
    ft_transformer  feature-tokenizing transformer (Gorishniy et al., 2021)

The tabular matrix is 1,015 base features x 7 summary statistics, stored
suffix-major.  FT-Transformer assigns one token per feature, and attention
over ~4,900 tokens is not tractable, so each *base* becomes one token whose
input is its 7 statistics.  No information is dropped -- every column
LightGBM sees reaches the transformer, just grouped by the base it
summarizes.

Split is by PATCH by default (train 15.14, validate 15.15, test 15.16): the deep models need a
validation set for early stopping, LightGBM uses exactly the same one, and holding out a patch
means no learner is tested on the patch it was fitted on.  --split match restores the old random
70/15/15 match split for comparison.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_deep_tabular_baselines.py ^
        --shards D:/LOL_Project/fusion_2615/corpus_shards ^
        --n-matches 40000 ^
        --output D:/LOL_Project/fusion_2615/features/deep_tabular_baselines.json

ToG revision (A1).  Every default reproduces the published run EXCEPT two leaks/bugs: the TabNet
sparsity sign (``--tabnet-legacy-sparsity-sign`` restores the published subtraction), and SAINT's
batching (``--saint-batches load_order`` restores it): published evaluation batches of 1,024 rows
in load order let intersample attention reach later engagements of the same match, so SAINT now
trains, pre-trains and evaluates on match-disjoint batches at its training batch size.
References followed, and the sections implemented:

    TabNet  Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021, Sec. 3:
            M[i] = sparsemax(P[i-1] * h_i(a[i-1])), P[i] = prod_j (gamma - M[j]), P[0] = 1;
            L_sparse = sum_i sum_b sum_j -M_bj[i] log(M_bj[i] + eps) / (N_steps * B), ADDED to the
            loss with coefficient lambda_sparse; feature transformer = 2 shared + 2 step-dependent
            FC-BN-GLU layers with sqrt(0.5) residual scaling; ghost BN (virtual batch B_V,
            momentum m_B) on every BN except the input one (--tabnet-virtual-batch).
            --tabnet-arch paper adds Glorot-uniform init, drops the output dropout, uses Adam with
            staircase decay and the appendix's Forest Cover Type hyper-parameters (B = 4,096).
    SAINT   Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via Row Attention
            and Contrastive Pre-Training", arXiv 2106.01342.  Sec. 3 stage equations (1)-(2),
            z + LN(f(z)) for MSA -> FF1 -> MISA -> FF2, MISA over the (1, b, n*d) reshape of
            Algorithm 1 (--saint-arch paper: App. C dropouts 0.1 / 0.8, Sec. 5 weight decay 0.01), and
            the Sec. 4 self-supervised stage: CutMix in input space (p_cutmix = 0.3), mixup in
            embedding space (alpha = 0.2), InfoNCE between projection heads g1(r), g2(r') with
            tau = 0.7, and denoising L_j(MLP_j(r'), x) with lambda_pt = 10 (--saint-pretrain-epochs).
    FT-T    Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021,
            Sec. 3.3 and Table 12 defaults (PreNorm without the first block's first norm, ReGLU FFN
            with factor 4/3, dropout 0.2/0.1/0.0, head Linear(ReLU(LayerNorm(CLS))), no weight
            decay on tokenizer/LayerNorm/biases; --ft-arch gorishniy).

Attention in both transformers runs through torch's scaled_dot_product_attention (exact attention,
memory-efficient kernel); ``attn_impl="math"`` materialises the softmax(QK^T/sqrt(d))V matrix and
is the reference the equivalence test compares against.  Every remaining difference from the
references is written to the output JSON under "deviations", every model records "n_params", and
each model's test predictions go to <output stem>.<model>.preds.npz (y, pred, groups; test-patch
row order).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 7
N_SUFFIXES = 7
LR_SCHEDULES = ("constant", "cosine_warmup", "onecycle", "exp_decay")
TORCH_MODELS = ("mlp", "tabnet", "ft_transformer", "saint")


def load_subsample(shard_dir: Path, n_matches: int | None, seed: int, y_key: str = "y",
                   drop_unlabelled: bool = False) -> dict:
    """Rows of ``n_matches`` seeded matches (None / 0: every match), in shard then row order.

    ``drop_unlabelled`` leaves out rows whose ``y_key`` is negative (draws under the drop policy)
    while filling the matrix -- the same rows, in the same order, as loading everything and
    filtering afterwards, without a second full-corpus copy; ``n_unlabelled_dropped`` counts them.
    """
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir}")
    keep_matches = None
    if n_matches:
        every = set()
        for path in paths:
            with np.load(path, allow_pickle=True) as blob:
                every.update(np.unique(blob["groups"]).tolist())
        every = sorted(every)
        rng = np.random.default_rng(seed)
        keep_matches = set(rng.choice(np.array(every), size=min(n_matches, len(every)),
                                      replace=False).tolist())
        print(f"matches available={len(every)} sampled={len(keep_matches)}")

    # Count first, then fill one preallocated float32 matrix.  Concatenating per-shard copies and
    # then calling astype peaked at ~3x the 15 GB full-corpus matrix; rows, their order and their
    # float32 values are unchanged.
    masks, n_rows, n_dropped = [], 0, 0
    for path in paths:
        with np.load(path, allow_pickle=True) as blob:
            groups = blob["groups"]
            mask = (np.isin(groups, list(keep_matches)) if keep_matches is not None
                    else np.ones(len(groups), dtype=bool))
            if drop_unlabelled:
                labelled = blob[y_key] >= 0
                n_dropped += int((mask & ~labelled).sum())
                mask &= labelled
        masks.append(mask)
        n_rows += int(mask.sum())
    if n_rows == 0:
        raise SystemExit(f"no rows selected under {shard_dir}")

    X = None
    at = 0
    y_parts, g_parts, s_parts, p_parts = [], [], [], []
    for path, mask in zip(paths, masks):
        if not mask.any():
            continue
        with np.load(path, allow_pickle=True) as blob:
            part = blob["X"][mask]
            if X is None:
                X = np.empty((n_rows, part.shape[1]), dtype=np.float32)
            X[at:at + len(part)] = part
            at += len(part)
            del part
            y_parts.append(blob[y_key][mask])
            g_parts.append(blob["groups"][mask])
            p_parts.append(blob["patch"][mask])
            s_parts.append(np.minimum(blob["cluster_blue"][mask], blob["cluster_red"][mask]))
    return {
        "X": X,
        "y": np.concatenate(y_parts).astype(np.int64),
        "groups": np.concatenate(g_parts),
        "patch": np.concatenate(p_parts),
        "min_participants": np.concatenate(s_parts),
        "n_unlabelled_dropped": n_dropped,
    }


def split_by_patch(patch: np.ndarray, train: str, val: str, test: str):
    """Hold a whole patch out: the learner never sees the patch it is scored on."""
    tr, va, te = patch == train, patch == val, patch == test
    missing = [n for n, m in ((train, tr), (val, va), (test, te)) if not m.any()]
    if missing:
        raise SystemExit(f"patches {missing} absent; present: {sorted(set(patch.tolist()))}")
    return tr, va, te


def split_by_match(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = int(0.70 * len(shuffled))
    n_val = int(0.15 * len(shuffled))
    train_m = set(shuffled[:n_train].tolist())
    val_m = set(shuffled[n_train:n_train + n_val].tolist())
    is_train = np.array([g in train_m for g in groups])
    is_val = np.array([g in val_m for g in groups])
    return is_train, is_val, ~(is_train | is_val)


class RowView:
    """Rows ``idx`` of ``X`` served batch by batch, so no split copy of the 15 GB matrix exists."""

    def __init__(self, X: np.ndarray, idx):
        self.X = X
        self.idx = np.asarray(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, key):
        return self.X[self.idx[key]]

    @property
    def shape(self) -> tuple:
        return (len(self.idx),) + tuple(self.X.shape[1:])


LGBM_PUBLISHED = {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31,
                  "subsample": 0.9, "colsample_bytree": 0.9}


def run_lightgbm(X, y, tr, va, te, *, params: dict | None = None, n_jobs: int = -1,
                 early_stopping_rounds: int = 50) -> dict:
    """``te=None`` never touches the test rows (hyper-parameter search)."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    started = time.time()
    config = {**LGBM_PUBLISHED, **(params or {})}
    model = LGBMClassifier(**config, random_state=SEED, n_jobs=n_jobs, verbose=-1)
    model.fit(
        X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
        callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X[te])[:, 1] if te is not None else None
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    trees = model.booster_.dump_model(num_iteration=best_iteration)["tree_info"]
    leaves = int(sum(t["num_leaves"] for t in trees))
    return {
        "test_auc": float(roc_auc_score(y[te], pred)) if te is not None else None,
        "val_auc": float(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1])),
        "best_iteration": best_iteration,
        "seconds": round(time.time() - started, 1),
        # a tree with L leaves stores L leaf values and (feature, threshold) at its L - 1 splits
        "n_params": int(sum(3 * t["num_leaves"] - 2 for t in trees)),
        "n_trees": len(trees),
        "n_leaves_total": leaves,
        "config": {**config, "early_stopping_rounds": early_stopping_rounds},
        "_pred": pred,
    }


def _torch_bits():
    """Imported lazily so the module loads without torch installed."""
    import torch
    from torch import nn
    return torch, nn


def _attend(q, k, v, dropout_p: float, impl: str = "sdpa"):
    """softmax(q k^T / sqrt(d_head)) v over the last two axes (Vaswani et al., 2017, Eq. 1).

    ``sdpa`` dispatches to torch's flash / memory-efficient kernel, which never forms the
    T x T matrix.  In float16 / bfloat16 those kernels need d_head % 8 == 0; otherwise (d_token 32
    with 8 heads: d_head 4) SDPA silently falls back to the math kernel and materialises the
    (B, h, T, T) weights, 1.97 GiB per layer at batch 64 and 1,017 tokens.  q, k, v are then
    zero-padded to the next multiple of 8 with the scale pinned to 1/sqrt(d_head): every dot
    product, hence every attention weight, is unchanged, and the padded output columns are
    dropped.  ``sdpa_pad`` pads in any dtype (tested against ``math``); float32 ``sdpa`` is the
    unpadded published path.  ``math`` materialises the matrix; it is the equivalence reference.
    """
    torch, nn = _torch_bits()
    if impl in ("sdpa", "sdpa_pad"):
        d_head = q.shape[-1]
        half = q.dtype in (torch.float16, torch.bfloat16)
        pad = (-d_head) % 8 if (half or impl == "sdpa_pad") else 0
        if not pad:
            return nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        q, k, v = (nn.functional.pad(t, (0, pad)) for t in (q, k, v))
        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                                         scale=d_head ** -0.5)
        return out[..., :d_head]
    if impl != "math":
        raise ValueError(f"unknown attention impl {impl!r}")
    weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1)
    if dropout_p:
        weights = nn.functional.dropout(weights, p=dropout_p)
    return weights @ v


def _run(fn, x, checkpoint: bool):
    """Apply ``fn``; under ``checkpoint`` keep only its input and recompute it in backward."""
    torch, _ = _torch_bits()
    if checkpoint and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint as ckpt
        return ckpt(fn, x, use_reentrant=False)
    return fn(x)


def build_ft_transformer(n_tokens: int, n_stats: int, d_token: int, n_layers: int,
                         n_heads: int, dropout: float, *, arch: str = "published",
                         attn_impl: str = "sdpa", grad_checkpoint: bool = False):
    torch, nn = _torch_bits()
    if arch not in ("published", "gorishniy"):
        raise ValueError(f"unknown FT-Transformer arch {arch!r}")
    gorishniy = arch == "gorishniy"
    # Gorishniy et al. (2021) Table 12 defaults; the published arch uses ``dropout`` everywhere
    attn_drop, ffn_drop, res_drop = (0.2, 0.1, 0.0) if gorishniy else (dropout, dropout, dropout)
    d_hidden = int(d_token * 4 / 3) if gorishniy else d_token * 2

    class ReGLU(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=-1)
            return torch.relu(a) * b

    class Block(nn.Module):
        """Pre-norm encoder block over scaled_dot_product_attention.

        The stock nn.TransformerEncoderLayer materializes the token x token
        attention matrix, which is 1,000+ tokens wide here and exhausts the
        GPU.  SDPA dispatches to a flash/memory-efficient kernel that never
        forms it, so memory scales with tokens rather than tokens squared.
        """

        def __init__(self, first: bool):
            super().__init__()
            # Gorishniy et al.: "remove the first normalization from the first Transformer layer"
            self.norm1 = nn.Identity() if (gorishniy and first) else nn.LayerNorm(d_token)
            self.qkv = nn.Linear(d_token, 3 * d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.norm2 = nn.LayerNorm(d_token)
            if gorishniy:
                self.ff = nn.Sequential(
                    nn.Linear(d_token, 2 * d_hidden), ReGLU(),
                    nn.Dropout(ffn_drop), nn.Linear(d_hidden, d_token),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Linear(d_token, d_hidden), nn.GELU(),
                    nn.Dropout(ffn_drop), nn.Linear(d_hidden, d_token),
                )
            self.drop = nn.Dropout(res_drop)

        def forward(self, x):
            b, t, d = x.shape
            h = self.norm1(x)
            q, k, v = self.qkv(h).reshape(b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            attended = _attend(q, k, v, attn_drop if self.training else 0.0, attn_impl)
            x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(b, t, d)))
            return x + self.drop(self.ff(self.norm2(x)))

    class FTTransformer(nn.Module):
        """Feature tokenizer + transformer (Gorishniy et al., 2021).

        One token per base feature, embedded from that base's summary
        statistics by its own linear map, plus a CLS token read by the head.
        """

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_tokens, n_stats, d_token))
            self.bias = nn.Parameter(torch.zeros(n_tokens, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            if gorishniy:
                bound = d_token ** -0.5
                for p in (self.weight, self.bias, self.cls):
                    nn.init.uniform_(p, -bound, bound)
                self.no_weight_decay = ("weight", "bias", "cls")
            else:
                nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
                nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList(Block(i == 0) for i in range(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(d_token), nn.ReLU() if gorishniy else nn.GELU(),
                nn.Linear(d_token, 1),
            )

        def forward(self, x):  # x: (B, n_tokens, n_stats)
            tokens = torch.einsum("bts,tsd->btd", x, self.weight) + self.bias
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            for block in self.blocks:
                tokens = _run(block, tokens, grad_checkpoint and self.training)
            return self.head(tokens[:, 0]).squeeze(-1)

    return FTTransformer()


def build_saint(n_tokens: int, n_stats: int, d_token: int, n_layers: int,
                n_heads: int, dropout: float, d_misa: int = 128, *, arch: str = "published",
                attn_impl: str = "sdpa", grad_checkpoint: bool = False,
                ff_dropout: float | None = None):
    """SAINT.  ``published``: the CoG run (pre-norm, pooled intersample attention, one FF per stage).

    ``paper``: Somepalli et al. (arXiv 2106.01342) Sec. 3.1, Eqs. (1)-(2), literally --
        z1 = LN(MSA(E(x))) + E(x);  z2 = LN(FF1(z1)) + z1;  z3 = LN(MISA(z2)) + z2;  r = LN(FF2(z3)) + z3
    with MISA over the (1, b, n*d) reshape (Algorithm 1), FF = Linear-GELU-Dropout-Linear of width
    4d (the released code's multiplier; the paper states no width), attention dropout ``dropout``
    and feed-forward dropout ``ff_dropout`` (App. C: 0.1 / 0.8), no residual dropout, and a
    one-hidden-layer ReLU head on [CLS].
    """
    torch, nn = _torch_bits()
    if arch not in ("published", "paper"):
        raise ValueError(f"unknown SAINT arch {arch!r}")
    paper = arch == "paper"
    ff_drop = dropout if ff_dropout is None else ff_dropout

    class SelfAttention(nn.Module):
        """Attention across a row's feature tokens (SAINT's MSA)."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_token)
            self.qkv = nn.Linear(d_token, 3 * d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            q, k, v = self.qkv(self.norm(x)).reshape(
                b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            out = _attend(q, k, v, dropout if self.training else 0.0, attn_impl)
            return x + self.drop(self.proj(out.transpose(1, 2).reshape(b, t, d)))

    class IntersampleAttention(nn.Module):
        """Attention across the rows of a batch (SAINT's MISA).

        SAINT flattens each row's tokens into one vector before attending
        across rows.  With 1,015 tokens that vector is ~32k wide and its
        projections alone would exceed the GPU, so rows are summarized by
        pooling their tokens into ``d_misa`` dimensions first; the resulting
        per-row context is broadcast back onto every token.  Documented
        deviation, forced by the feature count.
        """

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_token)
            self.summary = nn.Linear(d_token, d_misa)
            self.qkv = nn.Linear(d_misa, 3 * d_misa)
            self.proj = nn.Linear(d_misa, d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            pooled = self.summary(self.norm(x).mean(dim=1))          # (B, d_misa)
            q, k, v = self.qkv(pooled).reshape(b, 3, n_heads, d_misa // n_heads
                                               ).permute(1, 0, 2, 3)
            out = _attend(
                q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
                dropout if self.training else 0.0, attn_impl,
            )
            context = self.proj(out.transpose(1, 2).reshape(b, d_misa))  # (B, d_token)
            return x + self.drop(context).unsqueeze(1)

    def feed_forward():
        return nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_token * 2),
                             nn.GELU(), nn.Dropout(ff_drop), nn.Linear(d_token * 2, d_token))

    class PaperStage(nn.Module):
        """One stage of Somepalli et al. Eqs. (1)-(2): each sublayer f enters as z + LN(f(z)).

        MISA follows Algorithm 1: reshape (b, n, d) -> (1, b, n*d), attend across the b rows with h
        heads of dimension n*d/h, reshape back.  Dense (n*d) x (n*d) projections would be ~3.2e9
        parameters per stage at n = 1,017, d = 32, so q/k/v/out are token-shared, W = I_n (x) W_d:
        the attention pattern across rows and the dot product over each row's concatenated tokens
        (scaled by 1/sqrt(n*d/h)) are the paper's, only the projection is structured.
        """

        def __init__(self):
            super().__init__()
            self.msa_qkv = nn.Linear(d_token, 3 * d_token)
            self.msa_out = nn.Linear(d_token, d_token)
            self.misa_qkv = nn.Linear(d_token, 3 * d_token)
            self.misa_out = nn.Linear(d_token, d_token)
            self.ff1 = self._ff()
            self.ff2 = self._ff()
            self.ln = nn.ModuleList(nn.LayerNorm(d_token) for _ in range(4))

        @staticmethod
        def _ff():
            return nn.Sequential(nn.Linear(d_token, 4 * d_token), nn.GELU(), nn.Dropout(ff_drop),
                                 nn.Linear(4 * d_token, d_token))

        def msa(self, x):  # attention across one row's tokens
            b, t, d = x.shape
            q, k, v = self.msa_qkv(x).reshape(b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            out = _attend(q, k, v, dropout if self.training else 0.0, attn_impl)
            return self.msa_out(out.transpose(1, 2).reshape(b, t, d))

        def misa(self, x):  # attention across the rows of the batch
            b, t, d = x.shape
            dh = d // n_heads
            # (b, t, 3, h, dh) -> (3, h, b, t, dh) -> (3, 1, h, b, t*dh): one row = one sequence item
            qkv = self.misa_qkv(x).reshape(b, t, 3, n_heads, dh).permute(2, 3, 0, 1, 4)
            q, k, v = qkv.reshape(3, 1, n_heads, b, t * dh)
            out = _attend(q, k, v, dropout if self.training else 0.0, attn_impl)
            return self.misa_out(out.reshape(n_heads, b, t, dh).permute(1, 2, 0, 3).reshape(b, t, d))

        def forward(self, z):
            z = z + self.ln[0](self.msa(z))
            z = z + self.ln[1](self.ff1(z))
            z = z + self.ln[2](self.misa(z))
            return z + self.ln[3](self.ff2(z))

    class SAINT(nn.Module):
        """SAINT (Somepalli et al., 2021): alternating row and column attention."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_tokens, n_stats, d_token))
            self.bias = nn.Parameter(torch.zeros(n_tokens, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.normal_(self.cls, std=0.02)
            if paper:
                self.stages = nn.ModuleList(PaperStage() for _ in range(n_layers))
                # "a simple MLP with a single hidden layer with ReLU" on the [CLS] embedding
                self.head = nn.Sequential(nn.Linear(d_token, d_token), nn.ReLU(),
                                          nn.Linear(d_token, 1))
            else:
                self.self_attn = nn.ModuleList(SelfAttention() for _ in range(n_layers))
                self.inter_attn = nn.ModuleList(IntersampleAttention() for _ in range(n_layers))
                self.ff = nn.ModuleList(feed_forward() for _ in range(n_layers))
                self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1))
            self.d_token = d_token

        def embed(self, x):  # (B, n_tokens, n_stats) -> (B, 1 + n_tokens, d)
            tokens = torch.einsum("bts,tsd->btd", x, self.weight) + self.bias
            if paper:  # "a separate single fully-connected layer with a ReLU nonlinearity"
                tokens = torch.relu(tokens)
            return torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)

        def _stage(self, i, tokens):
            if paper:
                return self.stages[i](tokens)
            tokens = self.self_attn[i](tokens)
            tokens = self.inter_attn[i](tokens)
            return tokens + self.ff[i](tokens)

        def encode(self, tokens):
            for i in range(n_layers):
                tokens = _run(partial(self._stage, i), tokens, grad_checkpoint and self.training)
            return tokens

        def forward(self, x):
            return self.head(self.encode(self.embed(x))[:, 0]).squeeze(-1)

    return SAINT()


def build_saint_pretrain_heads(n_tokens: int, n_stats: int, d_token: int, stat_mask: np.ndarray,
                               proj_dim: int | None = None, denoise_hidden: int | None = None,
                               proj_hidden: int | None = None, similarity: str = "dot"):
    """Pre-training heads of Somepalli et al. (arXiv 2106.01342) Sec. 4, discarded before fine-tuning.

    g1, g2: "an MLP with one hidden layer and a ReLU".  The authors' released code normalises each
    token of r to unit length, flattens, and applies a dense MLP n*d -> 6nd/5 -> nd/2; dense on
    n*d = 32,544 that is ~1.3e9 parameters, so here the same widths are applied token-wise (d ->
    6d/5 -> d/2, shared over tokens) and flattened.  ``similarity="dot"`` leaves the projections
    as they are, so InfoNCE scores the raw dot product z_i . z'_k / tau of Eq. (5) (as the released
    code does); ``"cosine"`` L2-normalises them first.  MLP_j: one hidden-ReLU MLP per feature
    token of hidden width 5d (the released code's sep_MLP), reconstructing that token's statistics
    (``stat_mask`` marks the real ones).
    """
    if similarity not in ("dot", "cosine"):
        raise ValueError(f"unknown contrastive similarity {similarity!r}")
    torch, nn = _torch_bits()
    proj_dim = proj_dim or max(1, d_token // 2)
    proj_hidden = proj_hidden or max(1, 6 * d_token // 5)
    hidden = denoise_hidden or 5 * d_token

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_token, proj_hidden), nn.ReLU(),
                                     nn.Linear(proj_hidden, proj_dim))

        def forward(self, r):  # (B, T, d) -> (B, T * proj_dim)
            z = self.net(nn.functional.normalize(r, dim=-1)).flatten(1)
            return nn.functional.normalize(z, dim=-1) if similarity == "cosine" else z

    class DenoiseHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.empty(n_tokens, d_token, hidden))
            self.b1 = nn.Parameter(torch.zeros(n_tokens, hidden))
            self.w2 = nn.Parameter(torch.empty(n_tokens, hidden, n_stats))
            self.b2 = nn.Parameter(torch.zeros(n_tokens, n_stats))
            for w, fan_in in ((self.w1, d_token), (self.w2, hidden)):
                nn.init.uniform_(w, -fan_in ** -0.5, fan_in ** -0.5)

        def forward(self, r):  # (B, n_tokens, d) -> (B, n_tokens, n_stats)
            h = torch.relu(torch.einsum("btd,tdh->bth", r, self.w1) + self.b1)
            return torch.einsum("bth,ths->bts", h, self.w2) + self.b2

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.g1 = ProjectionHead()
            self.g2 = ProjectionHead()
            self.denoise = DenoiseHeads()
            self.similarity = similarity
            self.register_buffer("stat_mask", torch.as_tensor(stat_mask, dtype=torch.float32))

    return Heads()


def saint_pretrain_loss(model, heads, x, *, tau: float, lambda_denoise: float, p_cutmix: float,
                        mixup_alpha: float, reduction: str = "mean", generator=None):
    """Somepalli et al. (arXiv 2106.01342) Sec. 4 pre-training objective on one batch.

        x'  = x * m + x_a * (1 - m)               CutMix in input space, P(m_j = 0) = p_cutmix
        p'  = alpha E(x') + (1 - alpha) E(x'_b)   mixup in embedding space, Eq. (4)
        r   = SAINT(E(x)),  r' = SAINT(p')
        L   = -log softmax_k(z_i . z'_k / tau)[i] + lambda_pt sum_j MSE(MLP_j(r'_j), x_j)   Eq. (5)

    with z = g1(r), z' = g2(r').  Returns (loss, contrastive, denoise); both terms are averaged
    over rows.  ``reduction="mean"`` averages squared errors over columns as the authors' released
    code does (nn.MSELoss); ``"sum"`` sums them, the paper's literal sum over j.  Heads and losses
    run in float32 even under autocast.
    """
    torch, nn = _torch_bits()
    b = x.shape[0]
    # element-wise mask on the (row, token, statistic) view = one Bernoulli draw per column
    keep = (torch.rand(x.shape, device=x.device, generator=generator) >= p_cutmix).to(x.dtype)
    partner_a = torch.randperm(b, device=x.device, generator=generator)
    x_aug = x * keep + x[partner_a] * (1 - keep)
    emb_aug = model.embed(x_aug)
    partner_b = torch.randperm(b, device=x.device, generator=generator)
    p_aug = mixup_alpha * emb_aug + (1 - mixup_alpha) * emb_aug[partner_b]
    r = model.encode(model.embed(x))
    r_aug = model.encode(p_aug)
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = heads.g1(r.float()) @ heads.g2(r_aug.float()).T / tau
        contrastive = nn.functional.cross_entropy(logits, torch.arange(b, device=x.device))
        err = (heads.denoise(r_aug[:, 1:].float()) - x.float()) ** 2 * heads.stat_mask
        if reduction == "sum":
            denoise = err.sum(dim=(1, 2)).mean()
        elif reduction == "mean":
            denoise = err.sum() / (heads.stat_mask.sum() * b)
        else:
            raise ValueError(f"unknown denoise reduction {reduction!r}")
    return contrastive + lambda_denoise * denoise, contrastive.detach(), denoise.detach()


def _sparsemax(logits, dim=-1):
    """Euclidean projection onto the simplex (Martins & Astudillo, ICML 2016, Alg. 1)."""
    torch, _ = _torch_bits()
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumsum = sorted_logits.cumsum(dim) - 1
    rank = torch.arange(1, logits.shape[dim] + 1, device=logits.device,
                        dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    rank = rank.view(shape)
    support = rank * sorted_logits > cumsum
    k = support.sum(dim=dim, keepdim=True)
    tau = cumsum.gather(dim, k - 1) / k.to(logits.dtype)
    return torch.clamp(logits - tau, min=0)


def build_tabnet(d_in: int, n_steps: int, n_d: int, n_a: int, gamma: float,
                 dropout: float, *, virtual_batch_size: int = 0, bn_momentum: float = 0.02,
                 init: str = "torch"):
    """TabNet encoder of Arik & Pfister (AAAI 2021), Sec. 3 "TabNet for Tabular Learning".

    ``init="glorot"``: Glorot-uniform weights and zero biases ("Glorot uniform initialization",
    Sec. 6 Experiments); ``"torch"``: PyTorch defaults (the published run).
    """
    torch, nn = _torch_bits()
    if init not in ("torch", "glorot"):
        raise ValueError(f"unknown TabNet init {init!r}")

    sparsemax = _sparsemax

    class GhostBatchNorm1d(nn.BatchNorm1d):
        """Ghost BN (Hoffer et al., 2017) as TabNet uses it: in training, each virtual batch of
        ~``virtual_batch_size`` rows is normalised separately; evaluation uses running statistics.
        ``virtual_batch_size <= 0`` is exactly BatchNorm1d (the published run).  The batch is cut
        into ceil(B / B_V) near-equal parts, so no virtual batch has a single row."""

        def forward(self, x):
            if virtual_batch_size <= 0 or not self.training or x.shape[0] <= virtual_batch_size:
                return nn.BatchNorm1d.forward(self, x)
            chunks = torch.tensor_split(x, math.ceil(x.shape[0] / virtual_batch_size), dim=0)
            return torch.cat([nn.BatchNorm1d.forward(self, c) for c in chunks], dim=0)

    class GLUBlock(nn.Module):
        def __init__(self, d_input, d_output):
            super().__init__()
            self.fc = nn.Linear(d_input, 2 * d_output, bias=False)
            self.bn = GhostBatchNorm1d(2 * d_output, momentum=bn_momentum)

        def forward(self, x):
            return nn.functional.glu(self.bn(self.fc(x)), dim=-1)

    class FeatureTransformer(nn.Module):
        """Two shared blocks then two step-specific ones, residual-scaled."""

        def __init__(self, shared, d_input, d_output):
            super().__init__()
            self.shared = shared
            self.specific = nn.ModuleList(
                [GLUBlock(d_output, d_output), GLUBlock(d_output, d_output)]
            )
            self.scale = 0.5 ** 0.5

        def forward(self, x):
            out = self.shared[0](x)
            out = (out + self.shared[1](out)) * self.scale
            for block in self.specific:
                out = (out + block(out)) * self.scale
            return out

    class TabNet(nn.Module):
        """TabNet (Arik & Pfister, 2021): sequential sparse feature selection.

        Operates on the flat feature vector, so all 7,105 engineered columns
        enter directly -- no token budget to respect.  Each decision step picks
        a sparse feature mask with sparsemax, and the prior scale discourages
        reusing features across steps.  ``entropy`` after a forward pass is the
        paper's L_sparse, which the training loss ADDS with lambda_sparse.
        """

        def __init__(self):
            super().__init__()
            # the paper keeps plain BN on the input features ("low-variance averaging")
            self.bn = nn.BatchNorm1d(d_in, momentum=bn_momentum)
            width = n_d + n_a
            self.shared = nn.ModuleList([GLUBlock(d_in, width), GLUBlock(width, width)])
            self.initial = FeatureTransformer(self.shared, d_in, width)
            self.steps = nn.ModuleList(
                FeatureTransformer(self.shared, d_in, width) for _ in range(n_steps)
            )
            self.attention = nn.ModuleList(
                nn.Sequential(nn.Linear(n_a, d_in, bias=False),
                              GhostBatchNorm1d(d_in, momentum=bn_momentum))
                for _ in range(n_steps)
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(n_d, 1)
            self.entropy = torch.zeros(())
            if init == "glorot":
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        def forward(self, x):
            x = self.bn(x)
            prior = torch.ones_like(x)
            attended = self.initial(x)[:, n_d:]
            decision = torch.zeros(x.shape[0], n_d, device=x.device, dtype=x.dtype)
            entropy = torch.zeros((), device=x.device, dtype=x.dtype)
            for step, attn in zip(self.steps, self.attention):
                mask = sparsemax(attn(attended) * prior)
                entropy = entropy + (-mask * torch.log(mask + 1e-10)).sum(dim=1).mean()
                prior = prior * (gamma - mask)
                out = step(x * mask)
                decision = decision + nn.functional.relu(out[:, :n_d])
                attended = out[:, n_d:]
            self.entropy = entropy / max(1, n_steps)
            return self.head(self.drop(decision)).squeeze(-1)

    return TabNet()


def build_mlp(d_in: int, hidden: int, layers: int, dropout: float):
    torch, nn = _torch_bits()

    class Squeeze(nn.Module):
        def forward(self, x):
            return x.squeeze(-1)

    blocks: list = []
    prev = d_in
    for _ in range(layers):
        blocks += [nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        prev = hidden
    blocks += [nn.Linear(prev, 1), Squeeze()]
    return nn.Sequential(*blocks)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def token_layout(names: list[str]) -> tuple[int, list[str]]:
    """(n_bases, unstructured column names) of the suffix-major feature list."""
    n_bases = len(names) // N_SUFFIXES
    return n_bases, list(names[N_SUFFIXES * n_bases:])


def make_token_transform(n_bases: int, n_extras: int):
    """Flat standardised rows -> (B, n_bases + n_extras, 7) tokens, on the device.

    names are suffix-major: column j holds base (j % n_bases) under statistic
    (j // n_bases), so the token view is a pure reshape -- (rows, 7, bases) ->
    (rows, bases, 7).  Each unstructured column (v3.3: frame_age_s) becomes its
    own token whose statistic vector is [value, 0, ..., 0].  Identical values to
    the published numpy tokeniser, built per batch instead of as a second
    full-corpus copy.
    """
    structured = N_SUFFIXES * n_bases

    def transform(xb):
        tokens = xb[:, :structured].reshape(xb.shape[0], N_SUFFIXES, n_bases).transpose(1, 2)
        if n_extras:
            torch, _ = _torch_bits()
            pad = xb.new_zeros(xb.shape[0], n_extras, N_SUFFIXES)
            pad[:, :, 0] = xb[:, structured:]
            tokens = torch.cat([tokens, pad], dim=1)
        return tokens.contiguous()

    return transform


def token_stat_mask(n_bases: int, n_extras: int) -> np.ndarray:
    mask = np.ones((n_bases + n_extras, N_SUFFIXES), dtype=np.float32)
    mask[n_bases:, 1:] = 0.0
    return mask


def make_lr_scheduler(opt, schedule: str, lr: float, total_steps: int, *,
                      warmup_frac: float = 0.05, decay_rate: float = 0.95, decay_steps: int = 500):
    """Per-optimizer-step schedules.  ``constant`` is the published behaviour (no scheduler).

    cosine_warmup  linear warm-up over ``warmup_frac`` of the steps, cosine decay to 0 at the
                   last step of the epoch budget
    onecycle       torch OneCycleLR(max_lr=lr) with torch defaults (pct_start 0.3, cosine,
                   div_factor 25, final_div_factor 1e4, beta1 cycled 0.85-0.95)
    exp_decay      TabNet's staircase exponential decay: lr * decay_rate ** (step // decay_steps)
    """
    torch, _ = _torch_bits()
    if schedule == "constant":
        return None
    total_steps = max(1, int(total_steps))
    if schedule == "cosine_warmup":
        warm = max(1, int(round(warmup_frac * total_steps)))

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            progress = min(1.0, (step - warm) / max(1, total_steps - warm))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(opt, factor)
    if schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps)
    if schedule == "exp_decay":
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: decay_rate ** (step // max(1, decay_steps)))
    raise ValueError(f"unknown lr schedule {schedule!r}; known: {LR_SCHEDULES}")


def _param_groups(model, weight_decay: float):
    """All parameters decayed, unless the model names top-level parameters to exempt
    (FT-Transformer, gorishniy arch: tokenizer, LayerNorm and biases get no decay)."""
    exempt = getattr(model, "no_weight_decay", None)
    if not exempt:
        return model.parameters()
    _, nn = _torch_bits()
    norm_params = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm)
                   for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        skip = name in exempt or name.endswith("bias") or id(p) in norm_params
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _autocast(torch, device, amp: str):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp == "bf16")


def n_match_disjoint_batches(groups: np.ndarray, batch_size: int) -> int:
    counts = np.unique(np.asarray(groups), return_counts=True)[1]
    return max(math.ceil(len(groups) / batch_size), int(counts.max()) if len(counts) else 1)


def match_disjoint_batches(groups: np.ndarray, batch_size: int, rng) -> list[np.ndarray]:
    """Positions 0..n-1 cut into batches of about ``batch_size`` holding at most one row per match.

    SAINT's intersample attention mixes the rows of a batch, and neighbouring corpus rows are
    successive engagements of one match: a batch holding two rows of a match lets the earlier one
    attend to features observed after its prediction cutoff.  Matches are put in random order with
    their rows shuffled, and the k-th row of that sequence goes to batch k mod n_batches, so a
    match's rows land in distinct batches (n_batches >= its row count) and sizes differ by <= 1.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        return []
    n_batches = n_match_disjoint_batches(groups, batch_size)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    match_rank = rng.permutation(len(counts))[inverse]
    shuffled = rng.permutation(n)
    order = shuffled[np.argsort(match_rank[shuffled], kind="stable")]
    slot = np.arange(n) % n_batches
    by_batch = order[np.argsort(slot, kind="stable")]
    return np.split(by_batch, np.cumsum(np.bincount(slot, minlength=n_batches))[:-1])


def train_torch(model, X_tr, y_tr, X_va, y_va, X_te, y_te, *, epochs, batch_size,
                lr, weight_decay, patience, tokenized, label,
                sparsity_lambda: float = 0.0, legacy_sparsity_sign: bool = False,
                lr_schedule: str = "constant", warmup_frac: float = 0.05,
                lr_decay_rate: float = 0.95, lr_decay_steps: int = 500,
                amp: str = "none", input_transform=None, eval_batch_size: int = 1024,
                device=None, seed: int = SEED, groups_tr=None, groups_va=None,
                groups_te=None) -> dict:
    """Early stopping on validation AUC.  ``X_te=None`` never touches the test rows.

    ``groups_*`` (match ids of the rows) switch that split to match-disjoint batches
    (match_disjoint_batches); evaluation batches are then fixed and predictions are returned in
    row order.  None keeps the published batching: a permutation for training, row order for
    evaluation.
    """
    torch, nn = _torch_bits()
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(_param_groups(model, weight_decay), lr=lr, weight_decay=weight_decay)
    n = len(y_tr)
    steps_per_epoch = (math.ceil(n / batch_size) if groups_tr is None
                       else n_match_disjoint_batches(groups_tr, batch_size))
    scheduler = make_lr_scheduler(opt, lr_schedule, lr, epochs * steps_per_epoch,
                                  warmup_frac=warmup_frac, decay_rate=lr_decay_rate,
                                  decay_steps=lr_decay_steps)
    loss_fn = nn.BCEWithLogitsLoss()
    # BatchNorm cannot normalise a one-row batch in training; only such a batch is skipped
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    def as_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a))

    def evaluate(X_eval, y_eval, g_eval) -> tuple[float, np.ndarray]:
        model.eval()
        if g_eval is None:
            batches = [slice(i, i + eval_batch_size) for i in range(0, len(y_eval), eval_batch_size)]
        else:  # the same fixed batches at every evaluation
            batches = match_disjoint_batches(g_eval, eval_batch_size, np.random.default_rng(seed + 1))
        pred = np.empty(len(y_eval), dtype=np.float32)
        with torch.no_grad(), _autocast(torch, device, amp):
            for rows in batches:
                xb = as_tensor(X_eval[rows]).to(device, non_blocking=True)
                if input_transform is not None:
                    xb = input_transform(xb)
                pred[rows] = torch.sigmoid(model(xb).float()).cpu().numpy()
        return float(roc_auc_score(y_eval, pred)), pred

    best = {"val_auc": -1.0, "epoch": -1, "state": None}
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        if groups_tr is None:
            order = rng.permutation(n)
            batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
        else:
            batches = match_disjoint_batches(groups_tr, batch_size, rng)
        total = torch.zeros((), device=device)
        entropy_sum = torch.zeros((), device=device)
        epoch_started = time.time()
        for idx in batches:
            if has_bn and len(idx) < 2:
                continue
            xb = as_tensor(X_tr[idx]).to(device, non_blocking=True)
            yb = as_tensor(y_tr[idx].astype(np.float32)).to(device, non_blocking=True)
            if input_transform is not None:
                xb = input_transform(xb)
            opt.zero_grad(set_to_none=True)
            with _autocast(torch, device, amp):
                logits = model(xb)
            loss = loss_fn(logits.float(), yb)
            if sparsity_lambda:
                # Arik & Pfister (2021) Sec. 3: L = L_task + lambda_sparse * L_sparse, where
                # L_sparse is the mask entropy the module exposes.  The published run subtracted it.
                sign = -1.0 if legacy_sparsity_sign else 1.0
                loss = loss + sign * sparsity_lambda * getattr(model, "entropy", 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.detach() * len(idx)
            if hasattr(model, "entropy"):
                entropy_sum += model.entropy.detach().float() * len(idx)
        train_seconds = time.time() - epoch_started
        val_auc, _ = evaluate(X_va, y_va, groups_va)
        history.append({"epoch": epoch, "loss": float(total) / n, "val_auc": val_auc,
                        "lr_end": float(opt.param_groups[0]["lr"]),
                        "train_seconds": round(train_seconds, 1),
                        "seconds": round(time.time() - epoch_started, 1)})
        if hasattr(model, "entropy"):  # TabNet: mean training-batch L_sparse over the epoch
            history[-1]["mask_entropy"] = float(entropy_sum) / n
        print(f"  [{label}] epoch {epoch}/{epochs} loss={float(total) / n:.4f}"
              f" val_auc={val_auc:.4f}", flush=True)
        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            print(f"  [{label}] early stop at epoch {epoch}", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test_auc, pred = evaluate(X_te, y_te, groups_te) if X_te is not None else (None, None)
    return {
        "test_auc": test_auc,
        "val_auc": best["val_auc"],
        "best_epoch": best["epoch"],
        "seconds": round(time.time() - started, 1),
        "tokenized": tokenized,
        "n_params": count_params(model),
        "epochs_run": len(history),
        "history": history,
        **_peak_memory(torch, device),
        "_pred": pred,
    }


def _peak_memory(torch, device) -> dict:
    """torch.cuda.max_memory_allocated / _reserved since the last reset, in GiB (None on CPU)."""
    if device.type != "cuda":
        return {"peak_gpu_mem_allocated_gb": None, "peak_gpu_mem_reserved_gb": None}
    return {"peak_gpu_mem_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3),
            "peak_gpu_mem_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 3)}


def pretrain_saint(model, heads, X_rows, *, epochs: int, batch_size: int, lr: float,
                   weight_decay: float, tau: float, lambda_denoise: float, p_cutmix: float,
                   mixup_alpha: float, reduction: str = "mean", input_transform=None,
                   amp: str = "none", device=None, label: str = "saint", seed: int = SEED,
                   groups=None) -> dict:
    """Self-supervised SAINT stage (Somepalli et al., Sec. 4) on ``X_rows`` -- TRAIN rows only.

    AdamW on encoder + heads at the fine-tuning lr / weight decay, constant lr, gradient-norm
    clipping 1.0, a fixed number of epochs (the paper gives no pre-training schedule).
    ``groups`` (match ids of ``X_rows``) switches to match-disjoint batches.
    """
    torch, _ = _torch_bits()
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    heads.to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X_rows)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        heads.train()
        if groups is None:
            order = rng.permutation(n)
            batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
        else:
            batches = match_disjoint_batches(groups, batch_size, rng)
        sums = torch.zeros(3, device=device)
        seen = 0
        epoch_started = time.time()
        for idx in batches:
            if len(idx) < 2:  # InfoNCE needs a negative
                continue
            xb = torch.from_numpy(np.ascontiguousarray(X_rows[idx])).to(device, non_blocking=True)
            if input_transform is not None:
                xb = input_transform(xb)
            opt.zero_grad(set_to_none=True)
            with _autocast(torch, device, amp):
                loss, contrastive, denoise = saint_pretrain_loss(
                    model, heads, xb, tau=tau, lambda_denoise=lambda_denoise, p_cutmix=p_cutmix,
                    mixup_alpha=mixup_alpha, reduction=reduction, generator=generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sums += torch.stack([loss.detach().float(), contrastive.float(), denoise.float()]
                                ) * len(idx)
            seen += len(idx)
        means = (sums / max(1, seen)).tolist()
        history.append({"epoch": epoch, "loss": means[0], "contrastive": means[1],
                        "denoise": means[2], "seconds": round(time.time() - epoch_started, 1)})
        print(f"  [{label}:pretrain] epoch {epoch}/{epochs} loss={means[0]:.4f}"
              f" infonce={means[1]:.4f} denoise={means[2]:.4f}", flush=True)
    return {
        "epochs": epochs, "rows": int(n), "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "tau": tau, "lambda_denoise": lambda_denoise,
        "p_cutmix": p_cutmix, "mixup_alpha": mixup_alpha, "denoise_reduction": reduction,
        "contrastive_similarity": getattr(heads, "similarity", None),
        "batching": "random" if groups is None else "match_disjoint",
        "n_params_heads": count_params(heads), "history": history,
        "seconds": round(time.time() - started, 1), **_peak_memory(torch, device),
    }


# TabNet values the published run used for flags left unset
TABNET_PUBLISHED = {"n_steps": 4, "gamma": 1.3, "lr": 2e-3, "dropout": 0.1, "virtual_batch_size": 0,
                    "bn_momentum": 0.02, "weight_decay": 1e-5, "init": "torch"}
# Arik & Pfister (2021), appendix "Forest Cover Type": N_d = N_a = 64, lambda_sparse = 1e-4,
# N_steps = 5, gamma = 1.5, B = 16,384, B_V = 512, m_B = 0.7, lr 0.02 decayed 0.95 every 500
# iterations; Adam (= AdamW, weight decay 0), Glorot uniform init (Sec. 6).
TABNET_FOREST_COVER = {"n_d": 64, "n_a": 64, "n_steps": 5, "gamma": 1.5, "lambda_sparse": 1e-4,
                       "batch_size": 16384, "virtual_batch_size": 512, "bn_momentum": 0.3,
                       "lr": 0.02, "lr_schedule": "exp_decay", "lr_decay_rate": 0.95,
                       "lr_decay_steps": 500, "weight_decay": 0.0, "dropout": 0.0,
                       "init": "glorot"}
# --tabnet-arch paper: the Forest Cover Type values, except B (16,384 rows of 7,106-wide sparsemax
# masks does not fit the GPU; 4,096 is in the paper's search space and keeps 8 virtual batches)
TABNET_PAPER = {**{k: TABNET_FOREST_COVER[k] for k in TABNET_PUBLISHED}, "batch_size": 4096,
                "lr_schedule": "exp_decay"}


def published_configs(args=None) -> dict:
    """Per-model hyper-parameters of the published run, overridden by CLI flags when given.

    Flags left unset take the published value, or the reference's value under --tabnet-arch paper
    (TABNET_PAPER) / --saint-arch paper (App. C: feed-forward dropout 0.8; Sec. 5: weight decay 0.01).
    """
    a = args if args is not None else build_parser().parse_args(
        ["--shards", ".", "--output", "unused.json"])
    wd = a.weight_decay
    tab = TABNET_PAPER if a.tabnet_arch == "paper" else TABNET_PUBLISHED

    def tabnet(value, key):
        return tab[key] if value is None else value

    saint_paper = a.saint_arch == "paper"
    return {
        "lightgbm": {**LGBM_PUBLISHED, "n_jobs": a.lgbm_threads},
        "mlp": {"hidden": a.mlp_hidden, "layers": a.mlp_layers, "dropout": a.mlp_dropout,
                "lr": a.mlp_lr, "weight_decay": 1e-4 if wd is None else wd,
                "batch_size": max(a.batch_size, 256), "lr_schedule": a.lr_schedule},
        "tabnet": {"arch": a.tabnet_arch, "n_d": a.tabnet_n_d, "n_a": a.tabnet_n_a or a.tabnet_n_d,
                   "n_steps": tabnet(a.tabnet_n_steps, "n_steps"),
                   "gamma": tabnet(a.tabnet_gamma, "gamma"),
                   "dropout": tabnet(a.tabnet_dropout, "dropout"),
                   "lambda_sparse": a.tabnet_lambda_sparse,
                   "legacy_sparsity_sign": a.tabnet_legacy_sparsity_sign,
                   "virtual_batch_size": tabnet(a.tabnet_virtual_batch, "virtual_batch_size"),
                   "bn_momentum": tabnet(a.tabnet_bn_momentum, "bn_momentum"),
                   "init": tab["init"], "lr": tabnet(a.tabnet_lr, "lr"),
                   "weight_decay": tabnet(wd, "weight_decay"),
                   "batch_size": a.tabnet_batch_size or tab.get("batch_size")
                   or max(a.batch_size, 256),
                   "lr_schedule": a.tabnet_lr_schedule or tab.get("lr_schedule") or a.lr_schedule},
        "ft_transformer": {"d_token": a.d_token, "n_layers": a.n_layers, "n_heads": a.n_heads,
                           "dropout": 0.1, "arch": a.ft_arch, "lr": a.lr,
                           "weight_decay": 1e-5 if wd is None else wd,
                           "batch_size": a.batch_size, "lr_schedule": a.lr_schedule},
        "saint": {"d_token": a.d_token, "n_layers": a.n_layers, "n_heads": a.n_heads,
                  "dropout": 0.1, "d_misa": a.saint_d_misa, "arch": a.saint_arch, "lr": a.lr,
                  "ff_dropout": a.saint_ff_dropout if a.saint_ff_dropout is not None
                  else (0.8 if saint_paper else 0.1),
                  "weight_decay": wd if wd is not None else (0.01 if saint_paper else 1e-5),
                  "batch_size": a.batch_size,
                  "lr_schedule": a.lr_schedule, "pretrain_epochs": a.saint_pretrain_epochs,
                  "pretrain_rows": a.saint_pretrain_rows,
                  "pretrain_batch_size": a.saint_pretrain_batch_size or a.batch_size,
                  "tau": a.contrastive_tau, "lambda_denoise": a.lambda_denoise,
                  "p_cutmix": a.cutmix_p, "mixup_alpha": a.mixup_alpha,
                  "denoise_reduction": a.saint_denoise_reduction,
                  "contrastive_sim": a.saint_contrastive_sim, "batching": a.saint_batches},
    }


def training_options(args=None) -> dict:
    a = args if args is not None else build_parser().parse_args(
        ["--shards", ".", "--output", "unused.json"])
    return {"epochs": a.epochs, "patience": a.patience, "warmup_frac": a.warmup_frac,
            "lr_decay_rate": a.lr_decay_rate, "lr_decay_steps": a.lr_decay_steps,
            "amp": a.amp, "grad_checkpoint": a.grad_checkpoint,
            "eval_batch_size": a.eval_batch_size, "device": a.device, "seed": SEED}


def fit_model(name: str, cfg: dict, opts: dict, X: np.ndarray, y: np.ndarray,
              tr_idx, va_idx, te_idx, layout: tuple[int, int], label: str | None = None,
              groups: np.ndarray | None = None) -> dict:
    """Fit ``name`` on rows ``tr_idx``, early-stop on ``va_idx``, score ``te_idx`` if not None.

    ``layout`` is (n_bases, n_unstructured) for the tokenised models.  SAINT pre-training, when
    ``cfg["pretrain_epochs"] > 0``, sees only rows drawn from ``tr_idx``.  ``groups`` holds the
    match id of every row of ``X``; SAINT with ``cfg["batching"] == "match_disjoint"`` (default)
    requires it and then trains, pre-trains and evaluates on match-disjoint batches, evaluating
    at its training batch size.
    """
    label = label or name
    if name == "lightgbm":
        params = {k: v for k, v in cfg.items() if k != "n_jobs"}
        return run_lightgbm(X, y, tr_idx, va_idx, te_idx, params=params,
                            n_jobs=cfg.get("n_jobs", -1))
    if name not in TORCH_MODELS:
        raise ValueError(f"unknown model {name!r}")
    torch, _ = _torch_bits()
    torch.manual_seed(opts.get("seed", SEED))
    n_bases, n_extras = layout
    n_tokens = n_bases + n_extras
    transform = None
    if name == "mlp":
        model = build_mlp(X.shape[1], hidden=cfg["hidden"], layers=cfg["layers"],
                          dropout=cfg["dropout"])
    elif name == "tabnet":
        model = build_tabnet(X.shape[1], n_steps=cfg["n_steps"], n_d=cfg["n_d"], n_a=cfg["n_a"],
                             gamma=cfg["gamma"], dropout=cfg["dropout"],
                             virtual_batch_size=cfg["virtual_batch_size"],
                             bn_momentum=cfg["bn_momentum"], init=cfg.get("init", "torch"))
    elif name == "ft_transformer":
        transform = make_token_transform(n_bases, n_extras)
        model = build_ft_transformer(n_tokens, N_SUFFIXES, cfg["d_token"], cfg["n_layers"],
                                     cfg["n_heads"], dropout=cfg["dropout"], arch=cfg["arch"],
                                     grad_checkpoint=opts["grad_checkpoint"])
    else:
        transform = make_token_transform(n_bases, n_extras)
        model = build_saint(n_tokens, N_SUFFIXES, cfg["d_token"], cfg["n_layers"], cfg["n_heads"],
                            dropout=cfg["dropout"], d_misa=cfg["d_misa"], arch=cfg["arch"],
                            grad_checkpoint=opts["grad_checkpoint"],
                            ff_dropout=cfg.get("ff_dropout"))
    amp = opts["amp"] if name in ("ft_transformer", "saint") else "none"
    disjoint = name == "saint" and cfg.get("batching", "match_disjoint") == "match_disjoint"
    if disjoint and groups is None:
        raise ValueError("SAINT with match-disjoint batching needs the match id of every row "
                         "(groups=...); intersample attention would otherwise mix engagements of "
                         "one match")

    def ids(rows):
        return None if (not disjoint or rows is None) else np.asarray(groups)[rows]

    eval_batch_size = cfg["batch_size"] if disjoint else opts["eval_batch_size"]
    pretrain = None
    if name == "saint" and cfg.get("pretrain_epochs", 0) > 0:
        rows = np.asarray(tr_idx)
        cap = int(cfg.get("pretrain_rows") or 0)
        if 0 < cap < len(rows):
            rows = np.sort(np.random.default_rng(opts.get("seed", SEED)).choice(
                rows, size=cap, replace=False))
        heads = build_saint_pretrain_heads(n_tokens, N_SUFFIXES, cfg["d_token"],
                                           token_stat_mask(n_bases, n_extras),
                                           similarity=cfg.get("contrastive_sim", "dot"))
        pretrain = pretrain_saint(
            model, heads, RowView(X, rows), epochs=cfg["pretrain_epochs"],
            batch_size=cfg["pretrain_batch_size"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], tau=cfg["tau"],
            lambda_denoise=cfg["lambda_denoise"], p_cutmix=cfg["p_cutmix"],
            mixup_alpha=cfg["mixup_alpha"], reduction=cfg["denoise_reduction"],
            input_transform=transform, amp=amp, device=opts.get("device"), label=label,
            seed=opts.get("seed", SEED), groups=ids(rows))
        del heads
    entry = train_torch(
        model, RowView(X, tr_idx), y[tr_idx], RowView(X, va_idx), y[va_idx],
        None if te_idx is None else RowView(X, te_idx), None if te_idx is None else y[te_idx],
        epochs=opts["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], patience=opts["patience"],
        tokenized=transform is not None, label=label,
        sparsity_lambda=cfg.get("lambda_sparse", 0.0),
        legacy_sparsity_sign=cfg.get("legacy_sparsity_sign", False),
        lr_schedule=cfg.get("lr_schedule", "constant"), warmup_frac=opts["warmup_frac"],
        lr_decay_rate=opts["lr_decay_rate"], lr_decay_steps=opts["lr_decay_steps"],
        amp=amp, input_transform=transform, eval_batch_size=eval_batch_size,
        device=opts.get("device"), seed=opts.get("seed", SEED),
        groups_tr=ids(tr_idx), groups_va=ids(va_idx), groups_te=ids(te_idx),
    )
    if pretrain is not None:
        entry["pretrain"] = pretrain
    if name == "tabnet":
        entry["sparsity_sign"] = "-" if cfg.get("legacy_sparsity_sign") else "+"
    entry["config"] = {**cfg, "amp": amp, "grad_checkpoint": bool(
        opts["grad_checkpoint"] and name in ("ft_transformer", "saint")),
        "eval_batch_size": eval_batch_size}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return entry


def model_deviations(name: str, cfg: dict, opts: dict | None = None) -> list[str]:
    """Every remaining difference from the cited reference, written to the output JSON."""
    opts = opts or {}
    patience = opts.get("patience", 5)
    epochs = opts.get("epochs", 30)
    if name == "lightgbm":
        return [
            "sklearn LGBMClassifier with subsample=0.9 but subsample_freq=0 (LightGBM default), "
            "so row bagging is inactive; kept to reproduce the published configuration",
            "inputs standardised with train-patch statistics (monotone per column, so tree "
            "splits are unaffected up to float32 binning)",
        ]
    if name == "mlp":
        return ["no single reference architecture: Linear-LayerNorm-GELU-Dropout stack on the "
                "standardised flat feature vector, AdamW, gradient-norm clipping 1.0; the nearest "
                "reference, Gorishniy et al. (NeurIPS 2021) Sec. 3.1, uses Linear-ReLU-Dropout "
                "blocks without normalisation",
                f"early stopping on validation AUC, patience {patience}, max {epochs} epochs, lr "
                f"schedule '{cfg.get('lr_schedule', 'constant')}', batch {cfg['batch_size']}"]
    out: list[str] = []
    if name == "tabnet":
        # implemented as in the paper: sparsemax masks, prior P[i] = prod (gamma - M[j]) with
        # P[0] = 1, 2 shared + 2 step-dependent FC-BN-GLU layers with sqrt(0.5) residuals, split
        # into d[i] / a[i], sum of ReLU(d[i]) into a final FC, L_sparse added with lambda_sparse
        out += [
            "binary target trained with BCE-with-logits on one logit (paper: softmax "
            "cross-entropy over classes; equivalent parametrisation)",
            "inputs standardised with train-patch statistics (paper: numerical columns enter as "
            "is and the input BN scales them); every column is numeric, so no categorical "
            "embeddings",
            f"early stopping on validation AUC, patience {patience}, max {epochs} epochs (paper: "
            "trained to convergence)",
            "gradient-norm clipping 1.0 (not in the paper)",
            "mask entropy eps = 1e-10 (paper: 'a small number for numerical stability')",
            "no TabNet self-supervised decoder pre-training (paper Sec. 4; its supervised "
            "results do not use it)",
        ]
        if cfg.get("init", "torch") != "glorot":
            out.append("PyTorch default initialisation (paper: Glorot uniform; --tabnet-arch paper)")
        if cfg["dropout"] > 0:
            out.append(f"dropout {cfg['dropout']} on the aggregated decision before the output "
                       "layer (not in the paper)")
        vbs, batch = cfg["virtual_batch_size"], cfg["batch_size"]
        if vbs <= 0 or vbs >= batch:
            out.append("plain BatchNorm in the feature and attentive transformers (paper: ghost BN "
                       "with virtual batch B_V < B on every BN but the input one; "
                       "--tabnet-arch paper)")
        else:
            out.append(f"ghost BN: each training batch cut into ceil(B / B_V) = "
                       f"{math.ceil(batch / vbs)} near-equal virtual batches (B = {batch}, "
                       f"B_V = {vbs}); running statistics at evaluation")
        out.append(f"BN momentum {cfg['bn_momentum']} in PyTorch convention, i.e. paper "
                   f"m_B = {1 - cfg['bn_momentum']:.2f}")
        if cfg["weight_decay"]:
            out.append(f"AdamW weight decay {cfg['weight_decay']} (paper: Adam, no weight decay)")
        schedule = cfg.get("lr_schedule", "constant")
        if schedule != "exp_decay":
            out.append(f"lr schedule '{schedule}' (paper: exponential decay)")
        else:
            out.append("lr decayed as a staircase, lr * rate ** floor(step / period) (paper App.: "
                       "'decayed 0.95 every 0.5k iterations with an exponential decay'; the "
                       "staircase reading is ours)")
        forest = TABNET_FOREST_COVER
        tuned = [(k, cfg.get(k), forest[k]) for k in
                 ("n_d", "n_a", "n_steps", "gamma", "lambda_sparse", "batch_size",
                  "virtual_batch_size", "bn_momentum", "lr")]
        tuned += [(k, opts.get(k, forest[k]), forest[k]) for k in ("lr_decay_rate", "lr_decay_steps")
                  if schedule == "exp_decay"]
        differ = [f"{k} = {v} (paper {p})" for k, v, p in tuned if v != p]
        if differ:
            out.append("hyper-parameters differing from the paper's Forest Cover Type "
                       "configuration (the paper tunes them per dataset): " + "; ".join(differ))
        # the paper's appendix "Experiment hyperparameters" search space
        space = {"n_d": (8, 16, 24, 32, 64, 128), "n_a": (8, 16, 24, 32, 64, 128),
                 "n_steps": tuple(range(3, 11)), "gamma": (1.0, 1.2, 1.5, 2.0),
                 "lambda_sparse": (0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
                 "batch_size": (256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
                 "lr": (0.005, 0.01, 0.02, 0.025), "bn_momentum": (0.4, 0.3, 0.2, 0.1, 0.05, 0.02)}
        if 0 < vbs < batch:
            space["virtual_batch_size"] = (256, 512, 1024, 2048, 4096)
        outside = [f"{k} = {cfg[k]}" for k, allowed in space.items()
                   if not any(math.isclose(cfg[k], a, rel_tol=1e-9, abs_tol=1e-12) for a in allowed)]
        if schedule == "exp_decay":
            for k, allowed in (("lr_decay_rate", (0.4, 0.8, 0.9, 0.95)),
                               ("lr_decay_steps", (500, 2000, 8000, 10000, 20000))):
                if opts.get(k, TABNET_FOREST_COVER[k]) not in allowed:
                    outside.append(f"{k} = {opts.get(k)}")
        if outside:
            out.append("outside the paper's hyper-parameter search space (N_d, N_a in {8..128}, "
                       "N_steps 3-10, gamma {1.0, 1.2, 1.5, 2.0}, lambda {0, 1e-6, ..., 0.1}, B "
                       "256-32768, B_V 256-4096, lr {0.005, 0.01, 0.02, 0.025}, m_B 0.6-0.98): "
                       + "; ".join(outside))
        if cfg.get("legacy_sparsity_sign"):
            out.append("SPARSITY TERM SUBTRACTED from the loss: the published sign bug, "
                       "reproduced on request by --tabnet-legacy-sparsity-sign")
        return out
    tokens = ("one token per engineered base feature built from its 7 summary statistics by a "
              "per-token linear map (reference: one token per scalar column); each unstructured "
              "column is its own token [value, 0, ..., 0]; attention over 1,017 tokens instead of "
              "7,107 is the reason")
    budget = (f"batch {cfg['batch_size']}, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, "
              f"lr schedule '{cfg.get('lr_schedule', 'constant')}', max {epochs} epochs, "
              f"patience {patience}, gradient-norm clipping 1.0")
    if name == "ft_transformer":
        out += [tokens, "[CLS] placed first, T0 = stack[[CLS], T] as in Sec. 3.3 (the text says "
                        "'appended'; there are no positional embeddings, so position is immaterial)",
                "q, k, v projected by one fused Linear with PyTorch's default Kaiming-uniform "
                "initialisation (Table 12: Kaiming initialisation; the authors' code uses separate "
                "W_q, W_k, W_v)",
                f"d_token {cfg['d_token']}, {cfg['n_layers']} blocks, {cfg['n_heads']} heads "
                "(paper's default FT-Transformer: 192, 3, 8)"]
        if cfg["arch"] == "published":
            p = cfg["dropout"]
            out += [
                "first block keeps its pre-attention LayerNorm (paper removes it)",
                "GELU feed-forward with hidden factor 2 (paper default ReGLU, factor 4/3)",
                f"attention / FFN / residual dropout {p} / {p} / {p} (paper default 0.2/0.1/0.0)",
                "head LayerNorm-GELU-Linear (paper LayerNorm-ReLU-Linear)",
                "weight decay on every parameter (paper: none on tokenizer, LayerNorm, biases)",
                "tokenizer weight kaiming-uniform and [CLS] N(0, 0.02) initialisation",
            ]
        out.append(f"{budget} (paper default: AdamW lr 1e-4, weight decay 1e-5, patience 16)")
        return out
    # SAINT
    out.append(tokens)
    if cfg.get("batching", "match_disjoint") == "match_disjoint":
        out.append(f"batches hold at most one row per match in training, pre-training, validation "
                   f"and test, and evaluation uses the training batch size ({cfg['batch_size']}): "
                   "intersample attention mixes the rows of a batch, and two engagements of one "
                   "match would let the earlier attend to features observed after its prediction "
                   "cutoff (the paper's rows are independent, so its random batches are this)")
    else:
        out.append(f"LEAK CHANNEL (--saint-batches load_order, the published run): evaluation "
                   f"batches of {opts.get('eval_batch_size', 1024)} rows in load order let a "
                   "validation or test row attend to later engagements of its own match")
    if cfg["arch"] == "published":
        out += [
            "embedding is linear per token (paper: fully-connected layer with ReLU per "
            "continuous feature)",
            f"intersample attention pools each row's tokens into a d_misa = {cfg['d_misa']} "
            "summary, attends across rows and broadcasts the context back to every token (paper: "
            "attention across rows over the concatenated n*d token embeddings; --saint-arch paper)",
            "one feed-forward per stage, after intersample attention (paper: FF1 after MSA and "
            "FF2 after MISA)",
            "head LayerNorm-GELU-Linear on [CLS] (paper: one-hidden-layer ReLU MLP)",
            "pre-norm residual z + drop(f(LN(z))) with residual dropout (paper Eqs. (1)-(2): "
            "z + LN(f(z)); --saint-arch paper)",
            "GELU feed-forward of width 2d (paper: two FC layers with GELU, width not stated)",
        ]
    else:
        out += [
            "continuous-feature embedding: one FC + ReLU per base token over its 7 statistics "
            "(paper: one per scalar feature)",
            "intersample attention (Algorithm 1) over the concatenated n*d embeddings uses "
            "token-shared q/k/v/output projections W = I_n (x) W_d instead of dense (n*d) x (n*d) "
            "matrices (~4.2e9 parameters per stage at n = 1,017, d = 32; the authors' released "
            "code projects n*d densely to 8 heads x 64 dims, ~6.7e7 parameters per stage at "
            "d = 32 and ~2.7e8 at d = 128); dot product scaled by 1/sqrt(n*d/h)",
            "LayerNorm over each token's d dimensions in all four sublayers (Eqs. (1)-(2) do not "
            "state the normalised axis of LN(MISA(.)); the released code normalises the whole "
            "n*d vector before intersample attention)",
            "FF1 and FF2 token-wise, width 4d, GELU (paper: width not stated; released code: 4d "
            "with GEGLU, FF2 dense on the n*d vector, ~1.3e10 parameters per stage here)",
        ]
    out += [
        f"attention dropout {cfg['dropout']}, feed-forward dropout "
        f"{cfg.get('ff_dropout', cfg['dropout'])} (paper App. C for SAINT: 0.1 / 0.8)",
        f"L = {cfg['n_layers']} stages, d = {cfg['d_token']}, h = {cfg['n_heads']} (paper Table 1: "
        "SAINT L = 1, h = 8 -- App. C's L = 6 / 4 is the SAINT-s variant; App. C: d = 32, smaller "
        "for datasets with many features or memory limits)",
        f"per-head q/k/v width d/h = {cfg['d_token'] // max(1, cfg['n_heads'])} in self-attention "
        "(App. C: 16) and, in intersample attention, "
        + (f"d_misa/h = {cfg['d_misa'] // max(1, cfg['n_heads'])} on the pooled row summary"
           if cfg["arch"] == "published" else
           f"n*d/h over the {cfg['d_token'] // max(1, cfg['n_heads'])}-wide token-shared head "
           "projections")
        + " (App. C: 64)",
        f"{budget} (paper Sec. 5 'Training': AdamW betas (0.9, 0.999), lr 1e-4, weight decay 0.01, "
        "batch 256, smaller for many-column datasets, for pre-training and fine-tuning alike; no "
        "lr schedule or stopping rule stated)",
    ]
    if not cfg.get("pretrain_epochs"):
        out.append("no self-supervised pre-training (supervised only; paper pre-trains with "
                   "contrastive + denoising losses; --saint-pretrain-epochs)")
        return out
    p = cfg["p_cutmix"]
    out += [
        "projection heads g1, g2 ('an MLP with one hidden layer and a ReLU'): the released code's "
        "widths 6d/5 and d/2 applied token-wise and shared over tokens, after its per-token L2 "
        "normalisation of r (a dense MLP on the flattened n*d vector would be ~1.3e9 parameters "
        "at d = 32)",
        ("InfoNCE on the raw dot product of the flattened projections, as Eq. (5) and the "
         "released code" if cfg.get("contrastive_sim", "dot") == "dot" else
         "InfoNCE on cosine similarities (L2-normalised projections; Eq. (5) and the released "
         "code use the raw dot product)")
        + "; one-directional as in Eq. (5) (released code: symmetric, weight 0.5)",
        f"denoising heads MLP_j: hidden width 5d = {5 * cfg['d_token']} as in the released code's "
        "sep_MLP (the paper states no width), one per base token, reconstructing its 7 "
        "statistics by MSE (statistics padded onto unstructured tokens are masked; [CLS] not "
        "reconstructed)",
        f"CutMix replaces each column with probability p_cutmix = {p} by the same column of a "
        "random batch partner, as the released code does (the paper's 'm ~ Bernoulli(p_cutmix)' "
        f"with m = 1 keeping x_i would replace {1 - p:.0%} of columns)",
        f"mixup Eq. (4) literally: p' = alpha E(x') + (1 - alpha) E(x'_b), alpha = "
        f"{cfg['mixup_alpha']} (the [CLS] embedding is mixed too; it is identical across rows)",
        "both loss terms averaged over rows (paper sums over rows: a common 1/b factor)",
        (f"pre-training: AdamW, constant lr {cfg['lr']}, weight decay {cfg['weight_decay']}, "
         f"gradient-norm clipping 1.0, batch {cfg['pretrain_batch_size']}, "
         f"{cfg['pretrain_epochs']} epoch(s) on {cfg.get('pretrain_rows') or 'all'} train-patch "
         "rows, no early stopping (the paper states no pre-training schedule)"),
        "pre-training sees TRAIN-patch rows only (paper Sec. 4 'Finetuning': pre-trained 'on all "
        "unlabeled data'); validation and test features never enter it, so no transductive "
        "pre-training on the evaluation patches",
    ]
    if cfg["denoise_reduction"] == "sum":
        out.append("denoising squared errors summed over all 7,106 columns per row, the paper's "
                   "literal sum over j; with lambda_pt = 10 this outweighs InfoNCE by orders of "
                   "magnitude; both components are logged per epoch")
    else:
        out.append("denoising squared errors averaged over columns as in the released code "
                   "(nn.MSELoss); Eq. (5)'s sum over j would, at 7,106 columns and "
                   "lambda_pt = 10, outweigh InfoNCE ~10^4-fold; both components are logged per "
                   "epoch")
    return out


def cluster_bootstrap_auc(y: np.ndarray, preds: dict, groups: np.ndarray, *, n_boot: int = 1000,
                          seed: int = SEED, pairs=None) -> dict:
    """Match-clustered percentile bootstrap of AUC and of paired AUC differences.

    Each replicate draws matches with replacement; every row of a drawn match enters with the
    match's multiplicity.  All models share each draw, so paired deltas resample matches jointly.
    The replicate AUC is the weighted Mann-Whitney statistic with ties counted 1/2, i.e. exactly
    roc_auc_score on the explicitly replicated rows.
    """
    y = np.asarray(y).astype(np.float64)
    _, g_inv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = int(g_inv.max()) + 1
    names = list(preds)
    pairs = list(pairs) if pairs is not None else [
        (a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    prep = {}
    for name in names:
        values, inv = np.unique(np.asarray(preds[name], dtype=np.float64), return_inverse=True)
        prep[name] = (inv, len(values))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)[g_inv].astype(np.float64)
        wp, wn = w * y, w * (1.0 - y)
        for name in names:
            inv, n_values = prep[name]
            pos = np.bincount(inv, weights=wp, minlength=n_values)
            neg = np.bincount(inv, weights=wn, minlength=n_values)
            denom = pos.sum() * neg.sum()
            below = np.cumsum(neg) - neg
            draws[name][b] = (pos * (below + 0.5 * neg)).sum() / denom if denom else np.nan

    def ci(values):
        lo, hi = np.nanpercentile(values, [2.5, 97.5])
        return [float(lo), float(hi)]

    point = {name: float(roc_auc_score(y, preds[name])) for name in names}
    return {
        "method": "match-clustered percentile bootstrap, matches resampled jointly across models",
        "n_boot": int(n_boot), "seed": int(seed), "n_matches": n_g, "n_rows": int(len(y)),
        "auc": {name: {"point": point[name], "ci95": ci(draws[name])} for name in names},
        "paired": {f"{a}-{b}": {"delta": point[a] - point[b], "ci95": ci(draws[a] - draws[b])}
                   for a, b in pairs},
    }


def load_reference_preds(paths, y_test: np.ndarray, groups_test: np.ndarray,
                         prefix: str = "published") -> tuple[dict, list[str]]:
    """Test predictions of earlier runs, kept only when their test rows are this run's rows.

    Reads the combined ``<stem>.preds.npz`` files (y_test, groups_test, pred_<model>) and the
    per-model ``<stem>.<model>.preds.npz`` files (y, pred, groups).  A file whose labels or match
    ids differ row for row is skipped with a note, so every pairing resamples identical rows.
    Keys are ``<prefix>_<model>``; a later file repeating a model is keyed ``ref_<stem>_<model>``.
    """
    preds: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for path in paths or []:
        path = Path(path)
        if not path.exists():
            notes.append(f"{path}: absent")
            continue
        with np.load(path, allow_pickle=True) as blob:
            if "y_test" in blob.files:
                y_ref, g_ref = blob["y_test"], blob["groups_test"]
                found = {k[5:]: blob[k] for k in blob.files if k.startswith("pred_")}
            elif {"y", "pred", "groups"} <= set(blob.files):
                y_ref, g_ref = blob["y"], blob["groups"]
                model = path.name[:-len(".preds.npz")].rsplit(".", 1)[-1]
                found = {model: blob["pred"]}
            else:
                notes.append(f"{path}: no recognised prediction arrays; not paired")
                continue
            if not (len(y_ref) == len(y_test) and np.array_equal(y_ref, y_test)
                    and np.array_equal(g_ref, groups_test)):
                notes.append(f"{path}: test rows differ from this run; not paired")
                continue
            keys = []
            for model, pred in found.items():
                key = f"{prefix}_{model}"
                if key in preds:  # a second file with the same model: name it by its file
                    key = f"ref_{path.name.split('.')[0]}_{model}"
                preds[key] = np.asarray(pred)
                keys.append(key)
            notes.append(f"{path}: paired ({', '.join(sorted(found))}) as {', '.join(keys)}")
    return preds, notes


def run_metadata(argv=None) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    meta = {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "preset": os.environ.get("LOL_CFG_PRESET"),
            "lol_output_root": os.environ.get("LOL_OUTPUT_ROOT"),
            "seed": SEED, "argv": list(sys.argv if argv is None else argv),
            "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch
        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None
        return meta
    try:  # a hidden device (CUDA_VISIBLE_DEVICES="") can report available yet have no device 0
        meta["cuda_device"] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() and torch.cuda.device_count() else None)
    except (AssertionError, RuntimeError):
        meta["cuda_device"] = None
    return meta


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_model_preds(path: Path, y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> None:
    """y, pred, groups in test-patch row order (the order rows were loaded and split)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, y=y, pred=pred, groups=groups)
    os.replace(tmp, path)


def configure_runtime(torch_threads: int | None, max_gpu_mem_gb: float | None) -> None:
    if torch_threads is None and max_gpu_mem_gb is None:
        return
    torch, _ = _torch_bits()
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    if max_gpu_mem_gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_mem_gb / total))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--n-matches", type=int, default=40000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip-ft", action="store_true")
    parser.add_argument(
        "--models", default="lightgbm,mlp,ft_transformer,tabnet,saint",
        help="comma-separated subset to run; the split and preprocessing are "
             "seed-deterministic, so subsets are comparable across invocations",
    )
    parser.add_argument("--split", choices=("patch", "match"), default="patch")
    parser.add_argument("--train-patch", default="15.14")
    parser.add_argument("--val-patch", default="15.15")
    parser.add_argument("--test-patch", default="15.16")
    parser.add_argument("--y-key", default="y_market_event",
                        help="row label; -1 rows (draws under the drop policy) are removed")
    # ---- training parity (all deep models) ----
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                        help="per-step schedule for every deep model (published: constant)")
    parser.add_argument("--warmup-frac", type=float, default=0.05,
                        help="cosine_warmup: fraction of the epoch budget spent warming up")
    parser.add_argument("--lr-decay-rate", type=float, default=0.95, help="exp_decay rate")
    parser.add_argument("--lr-decay-steps", type=int, default=500, help="exp_decay period (steps)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay for every deep model (default: published "
                             "per-model values, MLP 1e-4, TabNet/FT/SAINT 1e-5)")
    # ---- capacity ----
    parser.add_argument("--amp", choices=("none", "bf16"), default="none",
                        help="bfloat16 autocast for FT-Transformer and SAINT")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="recompute FT-Transformer blocks / SAINT stages in backward")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=None,
                        help="cap this process's CUDA caching allocator")
    parser.add_argument("--lgbm-threads", type=int, default=-1)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="match-clustered bootstrap replicates for test-AUC CIs (0: off)")
    parser.add_argument("--pair-preds", nargs="*", type=Path, default=[],
                        help="earlier prediction files (<stem>.preds.npz or <stem>.<model>.preds.npz)"
                             " to pair against in the bootstrap; used only when their test rows "
                             "are this run's rows (default: none)")
    # ---- architectures ----
    parser.add_argument("--ft-arch", choices=("published", "gorishniy"), default="published")
    parser.add_argument("--saint-arch", choices=("published", "paper"), default="published")
    parser.add_argument("--saint-d-misa", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--tabnet-arch", choices=("published", "paper"), default="published",
                        help="paper: ghost BN on every BN but the input one, Glorot-uniform init, "
                             "no output dropout, Adam with staircase lr decay, and the paper's "
                             "Forest Cover Type values for every TabNet flag left unset "
                             "(except B = 4096)")
    parser.add_argument("--tabnet-n-d", type=int, default=64)
    parser.add_argument("--tabnet-n-a", type=int, default=None, help="default: n_d (paper)")
    parser.add_argument("--tabnet-n-steps", type=int, default=None, help="published 4, paper 5")
    parser.add_argument("--tabnet-gamma", type=float, default=None, help="published 1.3, paper 1.5")
    parser.add_argument("--tabnet-lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--tabnet-lr", type=float, default=None, help="published 2e-3, paper 0.02")
    parser.add_argument("--tabnet-dropout", type=float, default=None,
                        help="dropout before the output layer (not in the paper): published 0.1, "
                             "paper 0")
    parser.add_argument("--tabnet-virtual-batch", type=int, default=None,
                        help="ghost-BN virtual batch B_V (0: plain BN): published 0, paper 512")
    parser.add_argument("--tabnet-bn-momentum", type=float, default=None,
                        help="PyTorch momentum = 1 - m_B: published 0.02, paper 0.3 (m_B = 0.7)")
    parser.add_argument("--tabnet-batch-size", type=int, default=None,
                        help="published max(--batch-size, 256), paper 4096")
    parser.add_argument("--tabnet-lr-schedule", choices=LR_SCHEDULES, default=None,
                        help="published: --lr-schedule; paper: exp_decay (--lr-decay-rate 0.95 "
                             "every --lr-decay-steps 500, the paper's Forest Cover Type decay)")
    parser.add_argument("--tabnet-legacy-sparsity-sign", action="store_true",
                        help="subtract the sparsity term as the published run did (bug)")
    parser.add_argument("--saint-ff-dropout", type=float, default=None,
                        help="SAINT feed-forward dropout: published 0.1 (= attention dropout), "
                             "paper arch 0.8 (App. C)")
    # ---- SAINT self-supervised pre-training (Somepalli et al. Sec. 4) ----
    parser.add_argument("--saint-pretrain-epochs", type=int, default=0,
                        help="0 keeps the published supervised-only SAINT")
    parser.add_argument("--saint-pretrain-rows", type=int, default=0,
                        help="cap on train-patch rows used for pre-training (0: all)")
    parser.add_argument("--saint-pretrain-batch-size", type=int, default=None,
                        help="default: --batch-size")
    parser.add_argument("--contrastive-tau", type=float, default=0.7)
    parser.add_argument("--lambda-denoise", type=float, default=10.0, help="lambda_pt")
    parser.add_argument("--cutmix-p", type=float, default=0.3, help="p_cutmix")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--saint-denoise-reduction", choices=("sum", "mean"), default="mean",
                        help="mean over columns (released code) or the paper's literal sum")
    parser.add_argument("--saint-contrastive-sim", choices=("dot", "cosine"), default="dot",
                        help="InfoNCE similarity: Eq. (5)'s raw dot product (released code) or "
                             "cosine (L2-normalised projections)")
    parser.add_argument("--saint-batches", choices=("match_disjoint", "load_order"),
                        default="match_disjoint",
                        help="SAINT batches: at most one row per match in every training, "
                             "pre-training and evaluation batch, evaluated at the training batch "
                             "size (default); load_order reproduces the published run, whose "
                             "row-order evaluation batches let intersample attention reach later "
                             "engagements of the same match")
    return parser


def prepare_matrix(data: dict, tr: np.ndarray) -> None:
    """Standardise ``data["X"]`` in place with train-row statistics; constant-on-train columns
    collapse to zero.  Bit-identical to the published ``(X - mean) / std`` without the copy."""
    X = data["X"]
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std < 1e-6] = 1.0
    np.subtract(X, mean, out=X)
    np.divide(X, std, out=X)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    started = time.time()
    configure_runtime(args.torch_threads, args.max_gpu_mem_gb)

    data = load_subsample(args.shards, args.n_matches, SEED, y_key=args.y_key,
                          drop_unlabelled=True)
    if data["n_unlabelled_dropped"]:
        print(f"label {args.y_key}: dropping {data['n_unlabelled_dropped']} rows without a label"
              " (draws)")
    y, groups = data["y"], data["groups"]
    names = json.loads((args.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    n_bases, extras = token_layout(names)
    print(f"rows={len(y)} matches={len(np.unique(groups))} features={data['X'].shape[1]}"
          f" bases={n_bases} positives={y.mean():.3f}")

    if args.split == "patch":
        tr, va, te = split_by_patch(data["patch"], args.train_patch, args.val_patch, args.test_patch)
        print(f"patch split: train={args.train_patch} val={args.val_patch} test={args.test_patch}")
    else:
        tr, va, te = split_by_match(groups, SEED)
    print(f"split rows: train={tr.sum()} val={va.sum()} test={te.sum()}")

    # standardize on train only; constant-on-train columns collapse to zero
    prepare_matrix(data, tr)
    Xz = data["X"]
    tr_idx, va_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)

    configs = published_configs(args)
    opts = training_options(args)
    protocol = ["torch RNG seeded with 7 before each deep model is built (the published runs did "
                "not seed torch, so their initialisation is not reproducible bit for bit)"]
    if "tabnet" in wanted and not args.tabnet_legacy_sparsity_sign:
        protocol.append("TabNet sparsity term ADDED to the loss as in Arik & Pfister Sec. 3; the "
                        "published TabNet row subtracted it")
    if "saint" in wanted and args.saint_batches == "match_disjoint":
        protocol.append("SAINT batches are match-disjoint and evaluated at the training batch size; "
                        "the published SAINT row used row-order evaluation batches of 1,024, through "
                        "which intersample attention could reach later engagements of the same "
                        "match")
    results: dict[str, object] = {
        "n_rows": int(len(y)),
        "n_matches": int(len(np.unique(groups))),
        "n_unlabelled_dropped": int(data["n_unlabelled_dropped"]),
        "n_features": int(Xz.shape[1]),
        "n_bases": int(n_bases),
        "n_columns_seen_by_every_model": int(len(names)),
        "split": {"kind": args.split, "train_patch": args.train_patch, "val_patch": args.val_patch,
                  "test_patch": args.test_patch, "y_key": args.y_key,
                  "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                  "train_matches": int(len(np.unique(groups[tr]))),
                  "val_matches": int(len(np.unique(groups[va]))),
                  "test_matches": int(len(np.unique(groups[te])))},
        "positive_rate": float(y.mean()),
        "models": {},
        **run_metadata(argv),
        "label_key": args.y_key,
        "training_options": opts,
        "standardisation": "train rows only",
        "deviations": {"protocol": protocol},
    }
    preds: dict[str, np.ndarray] = {}
    stem = args.output.with_suffix("")

    def finish(name: str, entry: dict) -> None:
        preds[name] = entry.pop("_pred")
        results["models"][name] = entry
        results["deviations"][name] = model_deviations(name, configs[name], opts)
        write_model_preds(stem.with_name(f"{stem.name}.{name}.preds.npz"),
                          y[te], preds[name], groups[te])
        results["wall_clock_s"] = round(time.time() - started, 1)
        write_json_atomic(args.output, results)

    if "lightgbm" in wanted:
        print("[lightgbm] identical rows, split, and features")
        lgbm = fit_model("lightgbm", configs["lightgbm"], opts, Xz, y, tr, va, te,
                         (n_bases, len(extras)), groups=groups)
        finish("lightgbm", lgbm)
        print(f"  test AUC={lgbm['test_auc']:.4f} ({lgbm['seconds']}s,"
              f" {lgbm['best_iteration']} trees)")

    if "mlp" in wanted:
        print("[mlp] same flat feature vector")
        mlp = fit_model("mlp", configs["mlp"], opts, Xz, y, tr_idx, va_idx, te_idx,
                        (n_bases, len(extras)), groups=groups)
        finish("mlp", mlp)
        print(f"  test AUC={mlp['test_auc']:.4f} ({mlp['seconds']}s, {mlp['n_params']} params)")

    if "tabnet" in wanted:
        print("[tabnet] flat feature vector, sequential sparse selection")
        tabnet = fit_model("tabnet", configs["tabnet"], opts, Xz, y, tr_idx, va_idx, te_idx,
                           (n_bases, len(extras)), groups=groups)
        finish("tabnet", tabnet)
        print(f"  test AUC={tabnet['test_auc']:.4f} ({tabnet['seconds']}s,"
              f" {tabnet['n_params']} params)")

    if not args.skip_ft and ({"ft_transformer", "saint"} & wanted):
        suffixes = [names[i * n_bases].rsplit("__", 1)[1] for i in range(N_SUFFIXES)]
        print(f"[tokeniser] {n_bases} tokens x {N_SUFFIXES} statistics"
              f" (suffixes: {suffixes})" + (f" + {len(extras)} unstructured token(s): {extras}"
                                            if extras else ""))
        results["token_config"] = {
            "d_token": args.d_token, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "batch_size": args.batch_size, "lr": args.lr,
            "n_tokens": n_bases + len(extras), "ft_arch": args.ft_arch,
            "saint_arch": args.saint_arch, "lr_schedule": args.lr_schedule,
            "amp": args.amp, "grad_checkpoint": args.grad_checkpoint,
            "saint_pretrain_epochs": args.saint_pretrain_epochs,
        }
        for name in ("ft_transformer", "saint"):
            if name not in wanted:
                continue
            entry = fit_model(name, configs[name], opts, Xz, y, tr_idx, va_idx, te_idx,
                              (n_bases, len(extras)), groups=groups)
            finish(name, entry)
            print(f"  test AUC={entry['test_auc']:.4f} ({entry['seconds']}s,"
                  f" {entry['n_params']} params)")

    if args.n_boot and preds:
        ref = "lightgbm" if "lightgbm" in preds else None
        pairs = [(m, ref) for m in preds if ref and m != ref]
        boot_preds = dict(preds)
        if args.pair_preds:
            references, notes = load_reference_preds(args.pair_preds, y[te], groups[te])
            results["pairing"] = notes
            boot_preds.update(references)
            pairs += [(m, r) for m in preds for r in references]
        boot = cluster_bootstrap_auc(y[te], boot_preds, groups[te], n_boot=args.n_boot,
                                     pairs=pairs or None)
        results["test_auc_bootstrap"] = boot
        for name in preds:
            results["models"][name]["test_auc_ci95"] = boot["auc"][name]["ci95"]
    results["wall_clock_s"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, results)
    np.savez_compressed(
        args.output.with_suffix(".preds.npz"),
        y_test=y[te], groups_test=groups[te],
        min_participants_test=data["min_participants"][te],
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
