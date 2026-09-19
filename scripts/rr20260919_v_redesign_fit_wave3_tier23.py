#!/usr/bin/env python3
"""V redesign Tier 2–3: history stacks + sequence models (same 210k cohort).

Tier 2: H3 / H5 flattened last-K StateV2 frames -> LightGBM (mask early pads).
Tier 3: BiGRU, BiLSTM, Transformer, TCN on (K=5) sequences; optional Mamba if importable.

Contract: docs/V2_CANDIDATE_MATRIX_20260919.md
Writes: outputs/v_redesign_wave3_tier23_20260919/{results.json,REPORT.md,models/}
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
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
BANDS = ((0.0, 10.0, "t_0_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf"))
ALPHA = {b[2]: 0.25 for b in BANDS}
WAVE2 = REPO / "outputs" / "v_redesign_wave2_20260919"


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


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def metrics(y, p, g) -> Dict[str, Any]:
    w = match_weights(g)
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


def band_mask(tmin, lo, hi):
    return (tmin >= lo) & (tmin < hi)


def time_balanced_brier(y, p, g, tmin) -> Dict[str, Any]:
    parts, acc, missing = {}, 0.0, []
    for lo, hi, name in BANDS:
        m = band_mask(tmin, lo, hi)
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


def expanded_X(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float32, copy=False), cols, champ_ix


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


def build_history_stacks(
    match: np.ndarray, tmin: np.ndarray, X: np.ndarray, K: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return seq (N,K,D), mask (N,K) 1=real, flat (N, K*D+K) with mask bits; order preserved."""
    n, d = X.shape
    _, codes = np.unique(match.astype(str), return_inverse=True)
    order = np.lexsort((tmin, codes))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    codes_o = codes[order]
    X_o = X[order]
    seq = np.zeros((n, K, d), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and codes_o[j] == codes_o[i]:
            j += 1
        # positions i..j-1 in ordered space
        for t in range(i, j):
            start = max(i, t - K + 1)
            length = t - start + 1
            # place into right-aligned window
            seq[t, K - length : K] = X_o[start : t + 1]
            mask[t, K - length : K] = 1.0
        i = j
    # unpermute to original row order
    seq = seq[inv]
    mask = mask[inv]
    flat = np.concatenate([seq.reshape(n, K * d), mask], axis=1)
    return seq, mask, flat


def fit_lgbm_flat(Xflat, y, g, seed=7):
    import lightgbm as lgb
    w = match_weights(g)
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbose=-1,
    )
    clf.fit(Xflat, y, sample_weight=w)
    return clf


def predict_lgbm_flat(clf, Xflat):
    return clf.predict_proba(Xflat)[:, 1]


def _make_seq_model(kind: str, d_in: int, hidden: int = 128):
    import torch
    import torch.nn as nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "bigru":
                self.enc = nn.GRU(d_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
                out_d = hidden * 2
            elif kind == "bilstm":
                self.enc = nn.LSTM(d_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
                out_d = hidden * 2
            elif kind == "transformer":
                self.proj = nn.Linear(d_in, 64)
                layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                out_d = 64
            elif kind == "tcn":
                # simple causal dilated conv stack
                self.enc = nn.Sequential(
                    nn.Conv1d(d_in, 64, kernel_size=3, padding=2, dilation=1),
                    nn.GELU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=2),
                    nn.GELU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=4),
                    nn.GELU(),
                )
                out_d = 64
            else:
                raise ValueError(kind)
            self.head = nn.Sequential(nn.Linear(out_d, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, x, mask):
            # x: B,K,D  mask: B,K
            if self.kind in ("bigru", "bilstm"):
                lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                out, h = self.enc(packed)
                if self.kind == "bigru":
                    # h: 4,B,H -> cat last layer fwd/bwd
                    h_cat = torch.cat([h[-2], h[-1]], dim=-1)
                else:
                    h_n = h[0]
                    h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
                return self.head(h_cat).squeeze(-1)
            if self.kind == "transformer":
                z = self.proj(x)
                # True where pad (ignore)
                key_pad = mask < 0.5
                z = self.enc(z, src_key_padding_mask=key_pad)
                # masked mean
                m = mask.unsqueeze(-1)
                pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
                return self.head(pooled).squeeze(-1)
            # tcn: B,D,K
            z = x.transpose(1, 2)
            z = self.enc(z)
            # take last real timestep per row
            idx = (mask.sum(dim=1).long() - 1).clamp(min=0)
            z_last = z[torch.arange(z.size(0), device=z.device), :, idx]
            return self.head(z_last).squeeze(-1)

    return SeqNet()


def fit_seq(kind, seq, mask, y, g, seq_va, mask_va, y_va, g_va, seed=7, max_epochs=25, batch=1024):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, K, d = seq.shape
    # standardize using last-frame (current) features
    sc = StandardScaler()
    cur = seq[:, -1, :]
    sc.fit(cur)
    def transform_seq(S):
        flat = S.reshape(-1, d)
        flat = sc.transform(flat).astype(np.float32)
        return flat.reshape(S.shape)

    Str = transform_seq(seq)
    Sva = transform_seq(seq_va)
    wtr = match_weights(g).astype(np.float32)
    wva = match_weights(g_va).astype(np.float32)

    net = _make_seq_model(kind, d).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def run_epoch(train: bool):
        if train:
            net.train()
            idx = np.arange(n)
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
            total = 0.0
            for i in range(0, n, batch):
                sl = idx[i : i + batch]
                xb = torch.from_numpy(Str[sl]).to(device)
                mb = torch.from_numpy(mask[sl]).to(device)
                yb = torch.from_numpy(y[sl].astype(np.float32)).to(device)
                wb = torch.from_numpy(wtr[sl]).to(device)
                opt.zero_grad()
                logit = net(xb, mb)
                loss = (loss_fn(logit, yb) * wb).sum() / wb.sum().clamp_min(1e-8)
                loss.backward()
                opt.step()
                total += float(loss.item())
            return total
        net.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(y_va), batch * 2):
                xb = torch.from_numpy(Sva[i : i + batch * 2]).to(device)
                mb = torch.from_numpy(mask_va[i : i + batch * 2]).to(device)
                preds.append(torch.sigmoid(net(xb, mb)).cpu().numpy())
            pv = np.concatenate(preds)
            return float(np.average((pv - y_va) ** 2, weights=wva))

    best_state, best_br, bad, patience = None, 1e9, 0, 5
    for ep in range(max_epochs):
        run_epoch(True)
        br = run_epoch(False)
        if br < best_br - 1e-5:
            best_br, bad = br, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    # Persist state_dict only — nested SeqNet is not picklable via joblib.
    return dict(
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        scaler=sc,
        kind=kind,
        best_val_brier=best_br,
        d=d,
        K=K,
        hidden=128,
    )


def _hydrate_seq_pack(pack):
    """Attach a live nn.Module from saved state_dict (lazy, in-place)."""
    if "net" in pack and pack["net"] is not None:
        return pack
    net = _make_seq_model(pack["kind"], pack["d"], hidden=pack.get("hidden", 128))
    net.load_state_dict(pack["state_dict"])
    net.eval()
    pack["net"] = net
    return pack


def predict_seq(pack, seq, mask, batch=2048):
    import torch

    pack = _hydrate_seq_pack(pack)
    sc, net, d = pack["scaler"], pack["net"], pack["d"]
    flat = seq.reshape(-1, d)
    flat = sc.transform(flat).astype(np.float32)
    S = flat.reshape(seq.shape)
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            xb = torch.from_numpy(S[i : i + batch])
            mb = torch.from_numpy(mask[i : i + batch])
            outs.append(torch.sigmoid(net(xb, mb)).numpy())
    return np.concatenate(outs)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "v_redesign_wave3_tier23_20260919")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--skip-seq", action="store_true")
    ap.add_argument("--smoke-frac", type=float, default=0.0)
    ap.add_argument("--seq-kinds", default="bigru,bilstm,transformer,tcn")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    K = args.K

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print("load splits…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    CA = D.load_v_rows(L, "MAIN", ["V_CAL"], bucket_only=True)
    SE = D.load_v_rows(L, "MAIN", ["V_SELECT"], bucket_only=True)
    TE = D.load_v_rows(L, "MAIN", ["TEST"], bucket_only=True)
    names = list(TR["names"])
    W = D.load_outcomes(
        L, "MAIN", train_roles + ["V_CAL", "V_SELECT", "TEST"],
        purpose="V redesign Tier2-3 history/sequence",
    )

    def y_of(R):
        return np.asarray([W[m][0] for m in R["match"].tolist()], dtype=np.int64)

    if args.smoke_frac and args.smoke_frac < 1.0:
        rng = np.random.default_rng(7)
        matches = np.unique(TR["match"])
        keep_m = set(rng.choice(matches, size=max(300, int(len(matches) * args.smoke_frac)), replace=False).tolist())
        m = np.isin(TR["match"], list(keep_m))
        for k in ("match", "query_ms", "X"):
            TR[k] = TR[k][m]
        print(f"smoke TRAIN n={len(TR['match'])}", flush=True)

    yTR, yCA, ySE, yTE = y_of(TR), y_of(CA), y_of(SE), y_of(TE)
    tTR = TR["query_ms"].astype(float) / 60000.0
    tCA = CA["query_ms"].astype(float) / 60000.0
    tSE = SE["query_ms"].astype(float) / 60000.0
    tTE = TE["query_ms"].astype(float) / 60000.0
    Xtr, cols, _ = expanded_X(TR["X"], names)
    Xca, _, _ = expanded_X(CA["X"], names)
    Xse, _, _ = expanded_X(SE["X"], names)
    Xte, _, _ = expanded_X(TE["X"], names)
    print(f"feats={len(cols)} TRAIN={len(yTR)}", flush=True)

    raw_pred: Dict[str, Dict[str, np.ndarray]] = {}
    meta: Dict[str, Any] = {}

    # Reference: A0 current-frame logistic from wave2
    a0p = WAVE2 / "models" / "A0_shared_logistic.joblib"
    if a0p.is_file():
        print("score reference A0 logistic (current frame)…", flush=True)
        pack = joblib.load(a0p)
        from train.state_value_experiment import logistic  # noqa: F401 — path check
        def pred_log(Xraw):
            return pack["model"].predict_proba(Xraw[:, pack["keep"]])[:, 1]
        raw_pred["REF_A0_logistic_K1"] = {
            "cal": pred_log(CA["X"]),
            "sel": pred_log(SE["X"]),
            "test": pred_log(TE["X"]),
        }
        meta["REF_A0_logistic_K1"] = dict(precedent="Tier0 winner; no history")

    for K_h in (3, 5):
        print(f"build history K={K_h}…", flush=True)
        t0 = time.time()
        _, _, flat_tr = build_history_stacks(TR["match"], tTR, Xtr, K_h)
        _, _, flat_ca = build_history_stacks(CA["match"], tCA, Xca, K_h)
        _, _, flat_se = build_history_stacks(SE["match"], tSE, Xse, K_h)
        _, _, flat_te = build_history_stacks(TE["match"], tTE, Xte, K_h)
        print(f"  stacks in {time.time()-t0:.1f}s flat_dim={flat_tr.shape[1]}", flush=True)
        name = f"H{K_h}_lgbm"
        path = out / "models" / f"{name}.joblib"
        if path.is_file():
            print(f"load {name}…", flush=True)
            clf = joblib.load(path)["model"]
        else:
            print(f"fit {name}…", flush=True)
            t1 = time.time()
            clf = fit_lgbm_flat(flat_tr, yTR, TR["match"])
            print(f"  done {time.time()-t1:.1f}s", flush=True)
            joblib.dump({"model": clf, "K": K_h, "kind": name}, path)
        raw_pred[name] = {
            "cal": predict_lgbm_flat(clf, flat_ca),
            "sel": predict_lgbm_flat(clf, flat_se),
            "test": predict_lgbm_flat(clf, flat_te),
        }
        meta[name] = dict(precedent="Hodge history; Tier2", K=K_h)

    # Sequence models on K=args.K
    print(f"build sequence tensors K={K}…", flush=True)
    seq_tr, mask_tr, _ = build_history_stacks(TR["match"], tTR, Xtr, K)
    seq_ca, mask_ca, _ = build_history_stacks(CA["match"], tCA, Xca, K)
    seq_se, mask_se, _ = build_history_stacks(SE["match"], tSE, Xse, K)
    seq_te, mask_te, _ = build_history_stacks(TE["match"], tTE, Xte, K)

    if not args.skip_seq:
        kinds = [k.strip() for k in args.seq_kinds.split(",") if k.strip()]
        # try mamba
        if "mamba" in kinds:
            try:
                import mamba_ssm  # noqa: F401
            except Exception:
                print("skip mamba (package not available)", flush=True)
                kinds = [k for k in kinds if k != "mamba"]
        for kind in kinds:
            name = f"S_{kind}_K{K}"
            path = out / "models" / f"{name}.joblib"
            if path.is_file():
                print(f"load {name}…", flush=True)
                pack = joblib.load(path)
            else:
                print(f"fit {name}…", flush=True)
                t1 = time.time()
                pack = fit_seq(kind, seq_tr, mask_tr, yTR, TR["match"], seq_ca, mask_ca, yCA, CA["match"])
                print(f"  done {time.time()-t1:.1f}s val_brier={pack['best_val_brier']:.4f}", flush=True)
                # never persist live Module
                dump = {k: v for k, v in pack.items() if k != "net"}
                joblib.dump(dump, path)
                pack = dump
            pack = _hydrate_seq_pack(pack)
            raw_pred[name] = {
                "cal": predict_seq(pack, seq_ca, mask_ca),
                "sel": predict_seq(pack, seq_se, mask_se),
                "test": predict_seq(pack, seq_te, mask_te),
            }
            meta[name] = dict(precedent=f"CoG/ToG {kind}", K=K, best_val_brier=pack["best_val_brier"])

    print("calibrate + select…", flush=True)
    selection, calibrated_test = {}, {}
    for name, preds in raw_pred.items():
        fin_cal = np.isfinite(preds["cal"])
        if int(fin_cal.sum()) < 1000:
            selection[name] = dict(skipped=True)
            continue
        sig = PosSlopeSigmoid().fit(preds["cal"][fin_cal], yCA[fin_cal], match_weights(CA["match"][fin_cal]))
        sel_c = np.where(np.isfinite(preds["sel"]), sig.transform(preds["sel"]), np.nan)
        te_c = np.where(np.isfinite(preds["test"]), sig.transform(preds["test"]), np.nan)
        fin_s = np.isfinite(sel_c)
        tb = time_balanced_brier(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s], tSE[fin_s])
        overall = metrics(ySE[fin_s], sel_c[fin_s], SE["match"][fin_s])
        selection[name] = dict(
            L_time=tb["L_time"], select_overall=overall, select_bands=tb["bands"],
            ineligible=bool(tb.get("ineligible")), calib_ok=sig.ok,
            calib=dict(coef=sig.coef_, intercept=sig.intercept_) if sig.ok else None,
            meta=meta.get(name),
        )
        calibrated_test[name] = te_c
        print(f"  {name}: L_time={tb['L_time']:.6f} AUC={overall['auc']:.4f}", flush=True)

    eligible = {
        k: v for k, v in selection.items()
        if "L_time" in v and not k.startswith("REF_") and not v.get("ineligible")
        and np.isfinite(v.get("L_time", float("inf")))
    }
    winner = min(eligible, key=lambda k: (eligible[k]["L_time"], eligible[k]["select_overall"]["logloss"], k)) if eligible else None
    print("MECHANICAL WINNER (Tier2-3)", winner, flush=True)

    test_ledger = {}
    for name, p in calibrated_test.items():
        fin = np.isfinite(p)
        test_ledger[name] = dict(all=metrics(yTE[fin], p[fin], TE["match"][fin]), bands={})
        for lo, hi, bname in BANDS:
            m = band_mask(tTE, lo, hi) & fin
            test_ledger[name]["bands"][bname] = (
                metrics(yTE[m], p[m], TE["match"][m]) if int(m.sum()) >= 50 else dict(skipped=True, n=int(m.sum()))
            )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        protocol="docs/V2_CANDIDATE_MATRIX_20260919.md Tier 2–3",
        K_sequence=K,
        mechanical_winner_tier23=winner,
        selection=selection,
        test_ledger=test_ledger,
        census=dict(train=len(yTR), cal=len(yCA), select=len(ySE), test=len(yTE), n_features=len(cols)),
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

    (out / "results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V redesign Tier 2–3 — history + sequence",
        "",
        f"Generated: {payload['generated']}",
        f"**Epistemic:** {ROLE}",
        f"**Mechanical winner (this slate):** `{winner}`",
        "",
        "## V_SELECT",
        "",
        "| Candidate | L_time | Brier | AUC | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for name, sc in sorted(selection.items(), key=lambda kv: kv[1].get("L_time", 9e9) if "L_time" in kv[1] else 9e9):
        if "L_time" not in sc:
            continue
        o = sc["select_overall"]
        mark = " **" if name == winner else ""
        note = (sc.get("meta") or {}).get("precedent", "")
        lines.append(f"| {name}{mark} | {fmt(sc['L_time'], 6)} | {fmt(o['brier'], 6)} | {fmt(o['auc'], 4)} | {note} |")
    lines += ["", "## TEST all", "", "| Candidate | Brier | AUC |", "|---|---:|---:|"]
    for name, blk in test_ledger.items():
        a = blk["all"]
        lines.append(f"| {name} | {fmt(a['brier'], 6)} | {fmt(a['auc'], 4)} |")
    lines += [
        "",
        "## Reading",
        "",
        "- Compare to Tier 0 REF_A0_logistic_K1: does history/sequence beat current-frame logistic?",
        "- Tier 4 GNN still blocked on inputs.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
