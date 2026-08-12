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

Split is by match (70/15/15): the deep models need a validation set for
early stopping, and LightGBM uses exactly the same one, so no learner sees
an advantage in data or tuning budget.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_deep_tabular_baselines.py ^
        --shards D:/LOL_Project/fusion_2615/corpus_shards ^
        --n-matches 40000 ^
        --output D:/LOL_Project/fusion_2615/features/deep_tabular_baselines.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 7
N_SUFFIXES = 7


def load_subsample(shard_dir: Path, n_matches: int | None, seed: int) -> dict:
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

    X_parts, y_parts, g_parts, s_parts = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as blob:
            groups = blob["groups"]
            mask = (np.isin(groups, list(keep_matches)) if keep_matches is not None
                    else np.ones(len(groups), dtype=bool))
            if not mask.any():
                continue
            X_parts.append(blob["X"][mask])
            y_parts.append(blob["y"][mask])
            g_parts.append(groups[mask])
            s_parts.append(np.minimum(blob["cluster_blue"][mask], blob["cluster_red"][mask]))
    X = np.concatenate(X_parts).astype(np.float32)
    del X_parts
    return {
        "X": X,
        "y": np.concatenate(y_parts).astype(np.int64),
        "groups": np.concatenate(g_parts),
        "min_participants": np.concatenate(s_parts),
    }


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


def run_lightgbm(X, y, tr, va, te) -> dict:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    started = time.time()
    model = LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED,
        n_jobs=-1, verbose=-1,
    )
    model.fit(
        X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X[te])[:, 1]
    return {
        "test_auc": float(roc_auc_score(y[te], pred)),
        "val_auc": float(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1])),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "seconds": round(time.time() - started, 1),
        "_pred": pred,
    }


def _torch_bits():
    """Imported lazily so the module loads without torch installed."""
    import torch
    from torch import nn
    return torch, nn


def build_ft_transformer(n_tokens: int, n_stats: int, d_token: int, n_layers: int,
                         n_heads: int, dropout: float):
    torch, nn = _torch_bits()

    class Block(nn.Module):
        """Pre-norm encoder block over scaled_dot_product_attention.

        The stock nn.TransformerEncoderLayer materializes the token x token
        attention matrix, which is 1,000+ tokens wide here and exhausts the
        GPU.  SDPA dispatches to a flash/memory-efficient kernel that never
        forms it, so memory scales with tokens rather than tokens squared.
        """

        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.qkv = nn.Linear(d_token, 3 * d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.norm2 = nn.LayerNorm(d_token)
            self.ff = nn.Sequential(
                nn.Linear(d_token, d_token * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_token * 2, d_token),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, d = x.shape
            h = self.norm1(x)
            q, k, v = self.qkv(h).reshape(b, t, 3, n_heads, d // n_heads).permute(2, 0, 3, 1, 4)
            attended = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout if self.training else 0.0
            )
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
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1),
            )

        def forward(self, x):  # x: (B, n_tokens, n_stats)
            tokens = torch.einsum("bts,tsd->btd", x, self.weight) + self.bias
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            return self.head(tokens[:, 0]).squeeze(-1)

    return FTTransformer()


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


def train_torch(model, X_tr, y_tr, X_va, y_va, X_te, y_te, *, epochs, batch_size,
                lr, weight_decay, patience, tokenized, label) -> dict:
    torch, nn = _torch_bits()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    started = time.time()

    def as_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a))

    def evaluate(X_eval, y_eval) -> tuple[float, np.ndarray]:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(y_eval), 1024):
                xb = as_tensor(X_eval[i:i + 1024]).to(device, non_blocking=True)
                outs.append(torch.sigmoid(model(xb)).float().cpu().numpy())
        pred = np.concatenate(outs)
        return float(roc_auc_score(y_eval, pred)), pred

    n = len(y_tr)
    best = {"val_auc": -1.0, "epoch": -1, "state": None}
    rng = np.random.default_rng(SEED)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = as_tensor(X_tr[idx]).to(device, non_blocking=True)
            yb = as_tensor(y_tr[idx].astype(np.float32)).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        val_auc, _ = evaluate(X_va, y_va)
        print(f"  [{label}] epoch {epoch}/{epochs} loss={total / n:.4f}"
              f" val_auc={val_auc:.4f}", flush=True)
        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            print(f"  [{label}] early stop at epoch {epoch}", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test_auc, pred = evaluate(X_te, y_te)
    return {
        "test_auc": test_auc,
        "val_auc": best["val_auc"],
        "best_epoch": best["epoch"],
        "seconds": round(time.time() - started, 1),
        "tokenized": tokenized,
        "_pred": pred,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args(argv)

    data = load_subsample(args.shards, args.n_matches, SEED)
    X, y, groups = data["X"], data["y"], data["groups"]
    names = json.loads((args.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    n_bases = len(names) // N_SUFFIXES
    print(f"rows={len(y)} matches={len(np.unique(groups))} features={X.shape[1]}"
          f" bases={n_bases} positives={y.mean():.3f}")

    tr, va, te = split_by_match(groups, SEED)
    print(f"split rows: train={tr.sum()} val={va.sum()} test={te.sum()}")

    # standardize on train only; constant-on-train columns collapse to zero
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std < 1e-6] = 1.0
    Xz = (X - mean) / std
    del X

    results: dict[str, object] = {
        "n_rows": int(len(y)),
        "n_matches": int(len(np.unique(groups))),
        "n_features": int(Xz.shape[1]),
        "n_bases": int(n_bases),
        "split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "positive_rate": float(y.mean()),
        "models": {},
    }
    preds: dict[str, np.ndarray] = {}

    print("[lightgbm] identical rows, split, and features")
    lgbm = run_lightgbm(Xz, y, tr, va, te)
    preds["lightgbm"] = lgbm.pop("_pred")
    results["models"]["lightgbm"] = lgbm
    print(f"  test AUC={lgbm['test_auc']:.4f} ({lgbm['seconds']}s,"
          f" {lgbm['best_iteration']} trees)")

    print("[mlp] same flat feature vector")
    mlp = train_torch(
        build_mlp(Xz.shape[1], hidden=512, layers=3, dropout=0.1),
        Xz[tr], y[tr], Xz[va], y[va], Xz[te], y[te],
        epochs=args.epochs, batch_size=max(args.batch_size, 256), lr=1e-3,
        weight_decay=1e-4, patience=args.patience, tokenized=False, label="mlp",
    )
    preds["mlp"] = mlp.pop("_pred")
    results["models"]["mlp"] = mlp
    print(f"  test AUC={mlp['test_auc']:.4f} ({mlp['seconds']}s)")

    if not args.skip_ft:
        # names are suffix-major: column j holds base (j % n_bases) under
        # statistic (j // n_bases), so the token view is a pure reshape --
        # (rows, 7, bases) -> (rows, bases, 7).  Nothing is dropped or
        # reordered; the transformer sees exactly LightGBM's columns.
        suffixes = [names[i * n_bases].rsplit("__", 1)[1] for i in range(N_SUFFIXES)]
        print(f"[ft_transformer] {n_bases} tokens x {N_SUFFIXES} statistics"
              f" (suffixes: {suffixes})")
        tokens = Xz.reshape(len(y), N_SUFFIXES, n_bases).transpose(0, 2, 1)
        ft = train_torch(
            build_ft_transformer(n_bases, N_SUFFIXES, args.d_token, args.n_layers,
                                 args.n_heads, dropout=0.1),
            tokens[tr], y[tr], tokens[va], y[va], tokens[te], y[te],
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=1e-5, patience=args.patience, tokenized=True,
            label="ft_transformer",
        )
        preds["ft_transformer"] = ft.pop("_pred")
        results["models"]["ft_transformer"] = ft
        results["ft_config"] = {
            "d_token": args.d_token, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "batch_size": args.batch_size, "lr": args.lr,
        }
        print(f"  test AUC={ft['test_auc']:.4f} ({ft['seconds']}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
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
