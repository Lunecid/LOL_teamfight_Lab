#!/usr/bin/env python3
"""V redesign wave-4: corrected inputs + Phase A/B re-comparison.

Fixes vs INPUT_IMPL_v0:
  CAT-1  typed numeric/categorical adapters (Expanded361 / Core267)
  PACK-1 left-aligned history for RNN
  CAL-1  TRAIN-inner match stop for early stopping; V_CAL = calibration only
  HIST-1 history tokens from full observation store (bucket_only=False);
         supervised targets remain bucket samples; current token = supervised X
  LGBM   subsample_freq=1 when subsample<1

Phases:
  A — current-frame Logistic / native-LGBM / embedding-MLP × {expanded, core}
  B — same LGBM hypers: current vs H3/H5; one left-aligned GRU (expanded)

Contract: docs/V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md
Writes: outputs/v_redesign_wave4_corrected_20260919/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
BANDS = ((0.0, 10.0, "t_0_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf"))
ALPHA = {b[2]: 0.25 for b in BANDS}

from v_redesign_feature_adapters import (  # noqa: E402
    FeatureSchema,
    ProfileBundle,
    mean_one_match_weights,
    match_holdout_mask,
)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


class PosSlopeSigmoid:
    def __init__(self):
        self.ok = False
        self.coef_ = 1.0
        self.intercept_ = 0.0

    def fit(self, p, y, w):
        from sklearn.linear_model import LogisticRegression

        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y, sample_weight=w)
        coef = float(lr.coef_.ravel()[0])
        if coef <= 0:
            self.ok = False
            return self
        self.ok, self.coef_, self.intercept_ = True, coef, float(lr.intercept_.ravel()[0])
        return self

    def transform(self, p):
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        return 1.0 / (1.0 + np.exp(-(self.coef_ * logit + self.intercept_)))


def metrics(y, p, g) -> Dict[str, Any]:
    w = mean_one_match_weights(g)
    br = float(np.average((p - y) ** 2, weights=w))
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")
    return dict(n=int(len(y)), n_matches=int(len(np.unique(g))), brier=br, logloss=ll, auc=auc)


def time_balanced_brier(y, p, g, tmin) -> Dict[str, Any]:
    parts, acc, missing = {}, 0.0, []
    for lo, hi, name in BANDS:
        m = (tmin >= lo) & (tmin < hi)
        if int(m.sum()) < 50:
            parts[name] = dict(skipped=True, n=int(m.sum()))
            missing.append(name)
            continue
        sc = metrics(y[m], p[m], g[m])
        parts[name] = sc
        acc += ALPHA[name] * sc["brier"]
    if missing:
        return dict(L_time=float("inf"), bands=parts, ineligible=True, missing_bands=missing)
    return dict(L_time=float(acc), bands=parts, ineligible=False)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


# ── learners ──────────────────────────────────────────────────────────────


def fit_logistic(X, y, w, C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=2000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y, logisticregression__sample_weight=w)
    return pipe


def predict_logistic(model, X):
    return model.predict_proba(X)[:, 1]


def fit_lgbm_native(X, y, w, cat_idx, X_stop, y_stop, w_stop, seed=7):
    import lightgbm as lgb
    import pandas as pd

    cols = [f"c{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    dfv = pd.DataFrame(X_stop, columns=cols)
    for j in cat_idx:
        df[cols[j]] = df[cols[j]].astype("category")
        dfv[cols[j]] = dfv[cols[j]].astype("category")
    clf = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(
        df,
        y,
        sample_weight=w,
        eval_set=[(dfv, y_stop)],
        eval_sample_weight=[w_stop],
        categorical_feature=[cols[j] for j in cat_idx],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    return clf, cols, cat_idx


def predict_lgbm_native(clf, cols, cat_idx, X):
    import pandas as pd

    df = pd.DataFrame(X, columns=cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in cat_idx:
            df[cols[j]] = df[cols[j]].astype("category")
        return clf.predict_proba(df)[:, 1]


def fit_mlp_emb(num_tr, id_tr, y, w, num_va, id_va, y_va, w_va, n_vocab, emb_dim=8, hidden=(256, 128), dropout=0.1, seed=7, max_epochs=100, batch=1024, lr=1e-3):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_num = num_tr.shape[1]
    n_slots = id_tr.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
            d = d_num + n_slots * emb_dim
            layers: List[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, num, ids):
            e = self.emb(ids).reshape(ids.size(0), -1)
            return self.mlp(torch.cat([num, e], dim=-1)).squeeze(-1)

    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    num_tr_t = torch.from_numpy(num_tr.astype(np.float32))
    id_tr_t = torch.from_numpy(id_tr.astype(np.int64))
    y_tr_t = torch.from_numpy(y.astype(np.float32))
    w_tr_t = torch.from_numpy(w.astype(np.float32))
    num_va_t = torch.from_numpy(num_va.astype(np.float32)).to(device)
    id_va_t = torch.from_numpy(id_va.astype(np.int64)).to(device)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(device)
    w_va_t = torch.from_numpy(w_va.astype(np.float32)).to(device)

    best_state, best_br, bad, patience = None, 1e9, 0, 10
    best_epoch = 0
    n = len(y)
    for ep in range(max_epochs):
        net.train()
        idx = np.arange(n)
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(idx)
        for i in range(0, n, batch):
            sl = idx[i : i + batch]
            xb = num_tr_t[sl].to(device)
            ib = id_tr_t[sl].to(device)
            yb = y_tr_t[sl].to(device)
            wb = w_tr_t[sl].to(device)
            opt.zero_grad()
            loss = (loss_fn(net(xb, ib), yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(num_va_t, id_va_t))
            br = float((((pv - y_va_t) ** 2) * w_va_t).sum() / w_va_t.sum())
        if br < best_br - 1e-5:
            best_br, bad, best_epoch = br, 0, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.cpu()
    return dict(
        state_dict=net.state_dict(),
        n_vocab=n_vocab,
        emb_dim=emb_dim,
        hidden=list(hidden),
        dropout=dropout,
        d_num=d_num,
        n_slots=n_slots,
        best_val_brier=best_br,
        best_epoch=best_epoch,
        max_epochs_ran=ep + 1,
    )


def predict_mlp_emb(pack, num, ids, batch=65536):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(pack["n_vocab"], pack["emb_dim"], padding_idx=0)
            d = pack["d_num"] + pack["n_slots"] * pack["emb_dim"]
            layers: List[nn.Module] = []
            for h in pack["hidden"]:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(pack["dropout"])]
                d = h
            layers += [nn.Linear(d, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, num, ids):
            e = self.emb(ids).reshape(ids.size(0), -1)
            return self.mlp(torch.cat([num, e], dim=-1)).squeeze(-1)

    net = Net()
    net.load_state_dict(pack["state_dict"])
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(num), batch):
            xb = torch.from_numpy(num[i : i + batch].astype(np.float32))
            ib = torch.from_numpy(ids[i : i + batch].astype(np.int64))
            outs.append(torch.sigmoid(net(xb, ib)).numpy())
    return np.concatenate(outs)


# ── history (HIST-1) ───────────────────────────────────────────────────────


def build_history_from_store(
    store_match: np.ndarray,
    store_tmin: np.ndarray,
    store_num: np.ndarray,
    query_match: np.ndarray,
    query_tmin: np.ndarray,
    query_num: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left-aligned seq [N,K,D+1] with relative time; mask; current = query_num.

    Past tokens: last K-1 store frames with same match and t < query_t.
    Current token (slot length-1): query_num at relative_time=0.
    """
    from collections import defaultdict

    d = store_num.shape[1]
    by = defaultdict(list)
    for i, m in enumerate(store_match.astype(str)):
        by[m].append(i)
    for m in by:
        ix = np.asarray(by[m], dtype=np.int64)
        by[m] = ix[np.argsort(store_tmin[ix])]

    n = len(query_match)
    seq = np.zeros((n, K, d + 1), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)
    for i, (m, tq) in enumerate(zip(query_match.astype(str), query_tmin)):
        ix = by.get(m)
        past_num = []
        past_t = []
        if ix is not None and len(ix):
            ts = store_tmin[ix]
            # strictly earlier than query
            end = int(np.searchsorted(ts, tq, side="left"))
            take = ix[max(0, end - (K - 1)) : end]
            past_num = [store_num[j] for j in take]
            past_t = [store_tmin[j] for j in take]
        # assemble left-aligned: past then current
        tokens_num = list(past_num) + [query_num[i]]
        tokens_t = list(past_t) + [tq]
        length = min(K, len(tokens_num))
        # if more than K somehow, keep last K
        tokens_num = tokens_num[-length:]
        tokens_t = tokens_t[-length:]
        for k in range(length):
            seq[i, k, :d] = tokens_num[k]
            seq[i, k, d] = (tokens_t[k] - tq)  # already minutes; relative minutes
            mask[i, k] = 1.0
    return seq, mask, seq.reshape(n, -1)  # flat without mask; caller adds mask+champ


def flat_history_lgbm(seq_flat_no_mask, mask, champs) -> Tuple[np.ndarray, List[int]]:
    """K*(d+1) + K mask + 10 champ; cat idx = last 10."""
    X = np.hstack([seq_flat_no_mask, mask.astype(np.float64), champs.astype(np.float64)])
    cat_idx = list(range(X.shape[1] - champs.shape[1], X.shape[1]))
    return X, cat_idx


def fit_gru_history(seq, mask, champ_ids, y, w, seq_va, mask_va, champ_va, y_va, w_va, n_vocab, emb_dim=8, seed=7, max_epochs=40, batch=1024):
    """Proposed GRU: past tokens → GRU; current numeric skip; champ emb; fuse."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, K, d_tok = seq.shape  # d_tok = n_num+1 (rel time)
    d_num = d_tok - 1
    n_slots = champ_ids.shape[1]

    # standardize using current-token numeric (last valid)
    cur = []
    for i in range(n):
        L = int(mask[i].sum())
        cur.append(seq[i, L - 1, :d_num])
    cur = np.asarray(cur, dtype=np.float64)
    mean, std = cur.mean(0), cur.std(0)
    std = np.where(std < 1e-8, 1.0, std)

    def norm_seq(S, M):
        out = S.copy()
        out[..., :d_num] = (out[..., :d_num] - mean) / std
        # zero padding after scale
        out[M < 0.5] = 0.0
        return out.astype(np.float32)

    Str, Sva = norm_seq(seq, mask), norm_seq(seq_va, mask_va)

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_tok, 128)
            self.gru = nn.GRU(128, 128, num_layers=1, batch_first=True)
            self.cur = nn.Linear(d_num, 128)
            self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
            self.head = nn.Sequential(nn.Linear(128 + 128 + n_slots * emb_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

        def forward(self, x, m, ids):
            # left-aligned pack
            lengths = m.sum(dim=1).clamp(min=1).long().cpu()
            z = self.proj(x)
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            h_last = h[-1]
            # current token = last valid
            idx = (lengths - 1).to(x.device)
            cur_x = x[torch.arange(x.size(0), device=x.device), idx, :d_num]
            cur_h = self.cur(cur_x)
            e = self.emb(ids).reshape(ids.size(0), -1)
            return self.head(torch.cat([h_last, cur_h, e], dim=-1)).squeeze(-1)

    net = GRUNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_state, best_br, bad = None, 1e9, 0
    for ep in range(max_epochs):
        net.train()
        idx = np.arange(n)
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(idx)
        for i in range(0, n, batch):
            sl = idx[i : i + batch]
            xb = torch.from_numpy(Str[sl]).to(device)
            mb = torch.from_numpy(mask[sl]).to(device)
            ib = torch.from_numpy(champ_ids[sl].astype(np.int64)).to(device)
            yb = torch.from_numpy(y[sl].astype(np.float32)).to(device)
            wb = torch.from_numpy(w[sl].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = (loss_fn(net(xb, mb, ib), yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(y_va), batch * 2):
                xb = torch.from_numpy(Sva[i : i + batch * 2]).to(device)
                mb = torch.from_numpy(mask_va[i : i + batch * 2]).to(device)
                ib = torch.from_numpy(champ_va[i : i + batch * 2].astype(np.int64)).to(device)
                preds.append(torch.sigmoid(net(xb, mb, ib)).cpu().numpy())
            pv = np.concatenate(preds)
            br = float(np.average((pv - y_va) ** 2, weights=w_va))
        if br < best_br - 1e-5:
            best_br, bad = br, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state:
        net.load_state_dict(best_state)
    return dict(
        state_dict={k: v.cpu() for k, v in net.state_dict().items()},
        mean=mean,
        std=std,
        n_vocab=n_vocab,
        emb_dim=emb_dim,
        d_tok=d_tok,
        d_num=d_num,
        n_slots=n_slots,
        best_val_brier=best_br,
    )


def predict_gru(pack, seq, mask, champ_ids, batch=2048):
    import torch
    import torch.nn as nn

    d_tok, d_num, n_slots = pack["d_tok"], pack["d_num"], pack["n_slots"]
    mean, std = pack["mean"], pack["std"]

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_tok, 128)
            self.gru = nn.GRU(128, 128, num_layers=1, batch_first=True)
            self.cur = nn.Linear(d_num, 128)
            self.emb = nn.Embedding(pack["n_vocab"], pack["emb_dim"], padding_idx=0)
            self.head = nn.Sequential(nn.Linear(128 + 128 + n_slots * pack["emb_dim"], 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

        def forward(self, x, m, ids):
            lengths = m.sum(dim=1).clamp(min=1).long().cpu()
            z = self.proj(x)
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            h_last = h[-1]
            idx = (lengths - 1).to(x.device)
            cur_x = x[torch.arange(x.size(0), device=x.device), idx, :d_num]
            cur_h = self.cur(cur_x)
            e = self.emb(ids).reshape(ids.size(0), -1)
            return self.head(torch.cat([h_last, cur_h, e], dim=-1)).squeeze(-1)

    S = seq.copy()
    S[..., :d_num] = (S[..., :d_num] - mean) / std
    S[mask < 0.5] = 0.0
    net = GRUNet()
    net.load_state_dict(pack["state_dict"])
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            xb = torch.from_numpy(S[i : i + batch].astype(np.float32))
            mb = torch.from_numpy(mask[i : i + batch].astype(np.float32))
            ib = torch.from_numpy(champ_ids[i : i + batch].astype(np.int64))
            outs.append(torch.sigmoid(net(xb, mb, ib)).numpy())
    return np.concatenate(outs)


# ── main ───────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "v_redesign_wave4_corrected_20260919")
    ap.add_argument("--smoke-frac", type=float, default=0.0)
    ap.add_argument("--skip-history", action="store_true")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--profiles", default="expanded,core")
    ap.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load bucket supervised splits…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    CA = D.load_v_rows(L, "MAIN", ["V_CAL"], bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TR["names"])
    schema = FeatureSchema.from_names(names)
    print("schema groups", schema.groups, "keep", len(schema.keep_names), flush=True)

    W = D.load_outcomes(
        L, "MAIN", train_roles + ["V_CAL", "V_SELECT", "TEST"], purpose="V redesign wave4 corrected inputs"
    )

    def y_of(R):
        return np.asarray([W[m][0] for m in R["match"].tolist()], dtype=np.int64)

    if args.smoke_frac and args.smoke_frac < 1.0:
        rng = np.random.default_rng(7)
        matches = np.unique(TR["match"])
        keep_m = set(rng.choice(matches, size=max(400, int(len(matches) * args.smoke_frac)), replace=False).tolist())
        m = np.isin(TR["match"], list(keep_m))
        for k in ("match", "query_ms", "X"):
            TR[k] = TR[k][m]
        print(f"smoke TRAIN n={len(TR['match'])}", flush=True)

    yTR, yCA, ySE, yTE = y_of(TR), y_of(CA), y_of(SE), y_of(TE)
    tTR = TR["query_ms"].astype(float) / 60000.0
    tCA = CA["query_ms"].astype(float) / 60000.0
    tSE = SE["query_ms"].astype(float) / 60000.0
    tTE = TE["query_ms"].astype(float) / 60000.0
    stop = match_holdout_mask(TR["match"], 0.15, seed=7)
    fit_m = ~stop
    print(f"TRAIN fit={int(fit_m.sum())} stop={int(stop.sum())} matches_stop={len(np.unique(TR['match'][stop]))}", flush=True)

    # optional full store for history (same roles)
    store = {}
    if not args.skip_history:
        print("load full observation stores (bucket_only=False) for HIST-1…", flush=True)
        for key, roles in (("TR", train_roles), ("CA", ["V_CAL"]), ("SE", ["V_SELECT"]), ("TE", ["TEST"])):
            R = D.load_v_rows(L, "MAIN", roles, bucket_only=False)
            if args.smoke_frac and key == "TR":
                m = np.isin(R["match"], list(np.unique(TR["match"])))
                for kk in ("match", "query_ms", "X"):
                    R[kk] = R[kk][m]
            store[key] = R
            print(f"  store {key} n={len(R['match'])}", flush=True)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    raw_pred: Dict[str, Dict[str, np.ndarray]] = {}
    meta: Dict[str, Any] = {}
    bundles: Dict[str, ProfileBundle] = {}

    # ── Phase A ──
    for profile in profiles:
        print(f"\n=== Phase A profile={profile} ===", flush=True)
        bun = ProfileBundle(schema, profile).fit(TR["X"][fit_m])
        bundles[profile] = bun
        joblib.dump({"bundle_dims": bun.dims(), "schema_groups": schema.groups, "profile": profile, "ohe_n": bun.ohe.n_out, "vocab_n": bun.vocab.n_vocab}, out / "models" / f"bundle_{profile}.joblib")
        print("  dims", bun.dims(), flush=True)

        # Logistic
        name = f"A_LR_{profile}"
        path = out / "models" / f"{name}.joblib"
        Xtr = bun.matrix_onehot(TR["X"][fit_m])
        wtr = mean_one_match_weights(TR["match"][fit_m])
        if path.is_file() and not args.smoke_frac:
            print(f"  load {name}", flush=True)
            pack = joblib.load(path)
            model = pack["model"]
        else:
            print(f"  fit {name} onehot_dim={Xtr.shape[1]}…", flush=True)
            t0 = time.time()
            model = fit_logistic(Xtr, yTR[fit_m], wtr, C=args.C)
            print(f"    done {time.time()-t0:.1f}s", flush=True)
            joblib.dump({"model": model, "profile": profile, "kind": "logistic", "C": args.C}, path)
        raw_pred[name] = {
            "cal": predict_logistic(model, bun.matrix_onehot(CA["X"])),
            "sel": predict_logistic(model, bun.matrix_onehot(SE["X"])),
            "test": predict_logistic(model, bun.matrix_onehot(TE["X"])),
        }
        meta[name] = dict(phase="A", learner="logistic", input="numeric+onehot", **bun.dims())

        # LightGBM native
        name = f"A_LGBM_{profile}"
        path = out / "models" / f"{name}.joblib"
        X_all, cols_lgb, cat_idx = bun.matrix_lgbm(TR["X"])
        if path.is_file() and not args.smoke_frac:
            print(f"  load {name}", flush=True)
            pack = joblib.load(path)
            clf, cols_lgb, cat_idx = pack["model"], pack["cols"], pack["cat_idx"]
        else:
            print(f"  fit {name}…", flush=True)
            t0 = time.time()
            clf, cols_lgb, cat_idx = fit_lgbm_native(
                X_all[fit_m],
                yTR[fit_m],
                mean_one_match_weights(TR["match"][fit_m]),
                cat_idx,
                X_all[stop],
                yTR[stop],
                mean_one_match_weights(TR["match"][stop]),
            )
            print(f"    done {time.time()-t0:.1f}s best_iter={getattr(clf, 'best_iteration_', None)}", flush=True)
            joblib.dump({"model": clf, "cols": cols_lgb, "cat_idx": cat_idx, "profile": profile}, path)
        Xca, _, _ = bun.matrix_lgbm(CA["X"])
        Xse, _, _ = bun.matrix_lgbm(SE["X"])
        Xte, _, _ = bun.matrix_lgbm(TE["X"])
        raw_pred[name] = {
            "cal": predict_lgbm_native(clf, cols_lgb, cat_idx, Xca),
            "sel": predict_lgbm_native(clf, cols_lgb, cat_idx, Xse),
            "test": predict_lgbm_native(clf, cols_lgb, cat_idx, Xte),
        }
        meta[name] = dict(phase="A", learner="lgbm_native", **bun.dims())

        # Embedding MLP
        name = f"A_MLP_{profile}"
        path = out / "models" / f"{name}.joblib"
        num_all = bun.standardize_numeric(bun.numeric_raw(TR["X"]))
        id_all = bun.embedding_ids(TR["X"])
        if path.is_file() and not args.smoke_frac:
            print(f"  load {name}", flush=True)
            pack = joblib.load(path)
        else:
            print(f"  fit {name}…", flush=True)
            t0 = time.time()
            pack = fit_mlp_emb(
                num_all[fit_m],
                id_all[fit_m],
                yTR[fit_m],
                mean_one_match_weights(TR["match"][fit_m]),
                num_all[stop],
                id_all[stop],
                yTR[stop],
                mean_one_match_weights(TR["match"][stop]),
                n_vocab=bun.vocab.n_vocab,
            )
            print(f"    done {time.time()-t0:.1f}s val_brier={pack['best_val_brier']:.4f}", flush=True)
            joblib.dump({k: v for k, v in pack.items()}, path)
        raw_pred[name] = {
            "cal": predict_mlp_emb(pack, bun.standardize_numeric(bun.numeric_raw(CA["X"])), bun.embedding_ids(CA["X"])),
            "sel": predict_mlp_emb(pack, bun.standardize_numeric(bun.numeric_raw(SE["X"])), bun.embedding_ids(SE["X"])),
            "test": predict_mlp_emb(pack, bun.standardize_numeric(bun.numeric_raw(TE["X"])), bun.embedding_ids(TE["X"])),
        }
        meta[name] = dict(phase="A", learner="mlp_embedding", emb_dim=8, **bun.dims())

    # ── Phase B (expanded only) ──
    if not args.skip_history and "expanded" in bundles:
        bun = bundles["expanded"]
        print("\n=== Phase B history (expanded) ===", flush=True)

        def hist_pack(split_key, R, tmin):
            S = store[split_key]
            store_num = bun.numeric_raw(S["X"]).astype(np.float32)
            store_t = S["query_ms"].astype(float) / 60000.0
            q_num = bun.numeric_raw(R["X"]).astype(np.float32)
            return build_history_from_store(
                S["match"], store_t, store_num, R["match"], tmin, q_num, K=5
            )

        # Precompute K=5; derive K=3 by truncating left pad region… easier rebuild K=3
        for K_h in (3, 5):
            name = f"B_LGBM_H{K_h}_expanded"
            path = out / "models" / f"{name}.joblib"
            print(f"  build H{K_h} stacks…", flush=True)
            t0 = time.time()

            def make_flat(split_key, R, tmin, K):
                S = store[split_key]
                store_num = bun.numeric_raw(S["X"]).astype(np.float32)
                store_t = S["query_ms"].astype(float) / 60000.0
                q_num = bun.numeric_raw(R["X"]).astype(np.float32)
                seq, mask, flat = build_history_from_store(
                    S["match"], store_t, store_num, R["match"], tmin, q_num, K=K
                )
                ch = bun.champions_raw(R["X"])
                Xh, cat_idx = flat_history_lgbm(flat, mask, ch)
                return Xh, cat_idx, seq, mask

            Xtr_h, cat_h, _, _ = make_flat("TR", TR, tTR, K_h)
            print(f"    stacks {time.time()-t0:.1f}s dim={Xtr_h.shape[1]}", flush=True)
            if path.is_file() and not args.smoke_frac:
                pack = joblib.load(path)
                clf, cols_h, cat_h = pack["model"], pack["cols"], pack["cat_idx"]
            else:
                print(f"  fit {name}…", flush=True)
                t1 = time.time()
                clf, cols_h, cat_h = fit_lgbm_native(
                    Xtr_h[fit_m],
                    yTR[fit_m],
                    mean_one_match_weights(TR["match"][fit_m]),
                    cat_h,
                    Xtr_h[stop],
                    yTR[stop],
                    mean_one_match_weights(TR["match"][stop]),
                )
                print(f"    done {time.time()-t1:.1f}s", flush=True)
                joblib.dump({"model": clf, "cols": cols_h, "cat_idx": cat_h, "K": K_h}, path)
            Xca_h, _, _, _ = make_flat("CA", CA, tCA, K_h)
            Xse_h, _, _, _ = make_flat("SE", SE, tSE, K_h)
            Xte_h, _, _, _ = make_flat("TE", TE, tTE, K_h)
            raw_pred[name] = {
                "cal": predict_lgbm_native(clf, cols_h, cat_h, Xca_h),
                "sel": predict_lgbm_native(clf, cols_h, cat_h, Xse_h),
                "test": predict_lgbm_native(clf, cols_h, cat_h, Xte_h),
            }
            meta[name] = dict(phase="B", profile="expanded", learner="lgbm_history", K=K_h, dim=int(Xtr_h.shape[1]))

        if not args.skip_gru:
            name = "B_GRU_K5_expanded"
            path = out / "models" / f"{name}.joblib"
            print("  build GRU K=5…", flush=True)
            seq_tr, mask_tr, _ = build_history_from_store(
                store["TR"]["match"],
                store["TR"]["query_ms"].astype(float) / 60000.0,
                bun.numeric_raw(store["TR"]["X"]).astype(np.float32),
                TR["match"],
                tTR,
                bun.numeric_raw(TR["X"]).astype(np.float32),
                K=5,
            )
            id_tr = bun.embedding_ids(TR["X"])
            if path.is_file() and not args.smoke_frac:
                pack = joblib.load(path)
            else:
                print(f"  fit {name}…", flush=True)
                t0 = time.time()
                pack = fit_gru_history(
                    seq_tr[fit_m],
                    mask_tr[fit_m],
                    id_tr[fit_m],
                    yTR[fit_m],
                    mean_one_match_weights(TR["match"][fit_m]),
                    seq_tr[stop],
                    mask_tr[stop],
                    id_tr[stop],
                    yTR[stop],
                    mean_one_match_weights(TR["match"][stop]),
                    n_vocab=bun.vocab.n_vocab,
                )
                print(f"    done {time.time()-t0:.1f}s val_brier={pack['best_val_brier']:.4f}", flush=True)
                joblib.dump(pack, path)

            def gru_split(sk, R, tmin):
                seq, mask, _ = build_history_from_store(
                    store[sk]["match"],
                    store[sk]["query_ms"].astype(float) / 60000.0,
                    bun.numeric_raw(store[sk]["X"]).astype(np.float32),
                    R["match"],
                    tmin,
                    bun.numeric_raw(R["X"]).astype(np.float32),
                    K=5,
                )
                return predict_gru(pack, seq, mask, bun.embedding_ids(R["X"]))

            raw_pred[name] = {
                "cal": gru_split("CA", CA, tCA),
                "sel": gru_split("SE", SE, tSE),
                "test": gru_split("TE", TE, tTE),
            }
            meta[name] = dict(phase="B", profile="expanded", learner="gru_history", K=5, best_val_brier=pack["best_val_brier"])

    # ── calibrate on V_CAL only + select ──
    print("\ncalibrate (V_CAL only) + V_SELECT…", flush=True)
    selection, test_ledger = {}, {}
    for name, preds in raw_pred.items():
        fin = np.isfinite(preds["cal"])
        sig = PosSlopeSigmoid().fit(preds["cal"][fin], yCA[fin], mean_one_match_weights(CA["match"][fin]))
        sel_c = np.where(np.isfinite(preds["sel"]), sig.transform(preds["sel"]), np.nan)
        te_c = np.where(np.isfinite(preds["test"]), sig.transform(preds["test"]), np.nan)
        fin_s = np.isfinite(sel_c)
        tb = time_balanced_brier(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s], tSE[fin_s])
        overall = metrics(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s])
        selection[name] = dict(
            L_time=tb["L_time"],
            select_overall=overall,
            select_bands=tb["bands"],
            ineligible=tb.get("ineligible", False),
            calib_ok=sig.ok,
            calib=dict(coef=sig.coef_, intercept=sig.intercept_),
            meta=meta.get(name, {}),
        )
        fin_t = np.isfinite(te_c)
        test_ledger[name] = metrics(yTE[fin_t], te_c[fin_t], TE["match"][fin_t])
        print(f"  {name}: L_time={fmt(tb['L_time'])} AUC={fmt(overall['auc'])}", flush=True)

    eligible = {k: v for k, v in selection.items() if not v.get("ineligible") and v.get("calib_ok", True)}
    winner = min(eligible, key=lambda k: eligible[k]["L_time"]) if eligible else None
    print("MECHANICAL WINNER", winner, flush=True)

    # same-learner history delta
    hist_note = {}
    if "A_LGBM_expanded" in selection and "B_LGBM_H5_expanded" in selection:
        hist_note = dict(
            A_LGBM_expanded_L=selection["A_LGBM_expanded"]["L_time"],
            B_LGBM_H5_expanded_L=selection["B_LGBM_H5_expanded"]["L_time"],
            delta_H5_minus_current=selection["B_LGBM_H5_expanded"]["L_time"] - selection["A_LGBM_expanded"]["L_time"],
        )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md wave4",
        input_impl="corrected_v1_CAT1_PACK1_CAL1_HIST1",
        schema_groups=schema.groups,
        mechanical_winner=winner,
        history_same_learner=hist_note,
        selection=selection,
        test_ledger=test_ledger,
        census=dict(train=len(yTR), cal=len(yCA), select=len(ySE), test=len(yTE), train_fit=int(fit_m.sum()), train_stop=int(stop.sum())),
    )

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [scrub(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (out / "results.json").write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# V redesign wave-4 — corrected inputs (Phase A/B)",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        f"**Input impl:** `{payload['input_impl']}`",
        f"**Mechanical winner:** `{winner}`",
        "",
        "## V_SELECT",
        "",
        "| Candidate | Phase | L_time | Brier | AUC | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, blk in sorted(selection.items(), key=lambda kv: kv[1]["L_time"]):
        m = blk.get("meta", {})
        note = f"{m.get('learner','')}/{m.get('profile','')}"
        mark = " **" if name == winner else ""
        lines.append(
            f"| {name}{mark} | {m.get('phase','')} | {fmt(blk['L_time'])} | "
            f"{fmt(blk['select_overall']['brier'])} | {fmt(blk['select_overall']['auc'])} | {note} |"
        )
    lines += ["", "## TEST", "", "| Candidate | Brier | AUC |", "|---|---:|---:|"]
    for name, blk in test_ledger.items():
        lines.append(f"| {name} | {fmt(blk['brier'])} | {fmt(blk['auc'])} |")
    if hist_note:
        lines += [
            "",
            "## Same-learner history delta (expanded LGBM)",
            "",
            f"- Current `A_LGBM_expanded` L_time={fmt(hist_note['A_LGBM_expanded_L'])}",
            f"- History-5 `B_LGBM_H5_expanded` L_time={fmt(hist_note['B_LGBM_H5_expanded_L'])}",
            f"- Δ (H5 − current)={fmt(hist_note['delta_H5_minus_current'])} (negative ⇒ history helps)",
            "",
        ]
    lines += [
        "## Reading",
        "",
        "- CAT-1 / PACK-1 / CAL-1 / HIST-1 applied; do not mix with INPUT_IMPL_v0 scores as equal evidence.",
        "- Primary question: does corrected history beat same-learner current LGBM?",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
