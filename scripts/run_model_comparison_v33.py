"""Full-corpus model comparison for engagement-outcome prediction (v3.3, 532,547 engagements).

The headline AUC 0.6699 rests on ONE learner: LightGBM at n=400, lr .05, 31 leaves.  A reviewer
cannot tell from that whether 0.67 is a property of the task or of the model, and the CoG reviews
asked for deep-learning baselines.  This runs several families under IDENTICAL match-grouped folds
on the IDENTICAL merged matrix, so every difference is the learner:

  lead_only     LightGBM on gold/xp/level/time differences only - how much does the 6,164-column
                feature set buy over reading who is ahead?
  linear        L2 logistic regression (textbook; fitted by gradient descent on GPU so the 13 GB
                float32 matrix is never upcast) - how much does non-linearity buy?
  lgbm_paper    the published LightGBM configuration, refit here to verify it reproduces 0.6699
  lgbm_deep     the same learner with far more capacity - is 0.67 a capacity limit?

Every model here has a published reference: gradient-boosted trees (LightGBM) and regularised
logistic regression.  Deep tabular architectures are NOT re-implemented here - the repository
already carries paper-referenced implementations of MLP, FT-Transformer, TabNet and SAINT in
scripts/run_deep_tabular_baselines.py, written to answer CoG review R2, and those are used
instead so that no architecture in the comparison is an invention of this script.

Split is by PATCH, not by match: train on 15.14, select on 15.15, test on 15.16.  A random
match split lets a learner see the same patch it is tested on, and the thing we want to claim -
that engagement outcomes are predictable - has to survive the patch changing under it.  This is
the protocol docs/RELEASE_REPORT_V3.md already used for the CoG comparison.

Test predictions are saved for every model, so scale breakdowns and paired match-level bootstraps
between any two models can be recomputed without refitting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_scale_decomposition import merge_shards, scale_class, verify_shard_manifest  # noqa: E402

SEED = 7
CLASSES = ("pick", "skirmish", "teamfight")


# ------------------------------------------------------------------ torch models

def torch_fit_predict(kind, Xtr, ytr, Xva, yva, Xte, *, epochs, batch=4096, lr=1e-3, wd=1e-2,
                      hidden=512, layers=3, dropout=0.2, seed=SEED, log=print):
    """Standardise on the training fold, fit, return validation probabilities."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    d = Xtr.shape[1]

    if kind != "linear":
        raise ValueError(
            f"{kind!r} is not defined here on purpose: deep tabular architectures must come from "
            "scripts/run_deep_tabular_baselines.py, which implements them against their papers")
    model = nn.Linear(d, 1)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    tr_idx = np.arange(len(ytr))          # early stopping uses the held-out validation PATCH
    best, best_state, patience = -1.0, None, 0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * (len(tr_idx) // batch + 1)))
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(order), batch):
            idx = np.sort(tr_idx[order[s:s + batch]])
            xb = torch.from_numpy((Xtr[idx] - mu) / sd).to(dev, non_blocking=True)
            yb = torch.from_numpy(ytr[idx].astype(np.float32)).to(dev).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
        auc = roc_auc_score(yva, _infer(model, Xva, np.arange(len(Xva)), mu, sd, dev, batch))
        if auc > best + 1e-5:
            best, best_state, patience = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        log(f"      epoch {ep+1}/{epochs} es-auc {auc:.4f}{' *' if patience == 0 else ''}")
        if patience >= 3:
            break
    if best_state:
        model.load_state_dict(best_state)
    return _infer(model, Xte, np.arange(len(Xte)), mu, sd, dev, batch), float(best), None


def _infer(model, X, idx, mu, sd, dev, batch):
    import torch
    model.eval()
    out = np.empty(len(idx), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sel = np.sort(idx[s:s + batch])
            xb = torch.from_numpy((X[sel] - mu) / sd).to(dev)
            out[s:s + len(sel)] = torch.sigmoid(model(xb)).squeeze(1).float().cpu().numpy()
    return out


# ------------------------------------------------------------------ runner

def run_model(name, X, y, tr, va, te, *, cols=None, log=print):
    """Fit on the train patch, select on the validation patch, predict the test patch."""
    started = time.time()
    sel = (lambda m: X[m][:, cols]) if cols is not None else (lambda m: X[m])
    Xtr, Xva, Xte = sel(tr), sel(va), sel(te)
    if name.startswith("lgbm") or name == "lead_only":
        from lightgbm import early_stopping, log_evaluation
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
                      colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
        if name == "lgbm_deep":
            params.update(n_estimators=3000, learning_rate=0.02, num_leaves=127,
                          min_child_samples=100, reg_lambda=1.0)
        m = LGBMClassifier(**params)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], eval_metric="auc",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        best = getattr(m, "best_iteration_", None)
        pred = m.predict_proba(Xte)[:, 1]
        val_auc = float(roc_auc_score(y[va], m.predict_proba(Xva)[:, 1]))
    else:
        pred, val_auc, best = torch_fit_predict(name, Xtr, y[tr], Xva, y[va], Xte, log=log,
                                                **({"epochs": 25, "lr": 3e-3, "wd": 1e-3}
                                                   if name == "linear" else {"epochs": 30}))
    log(f"    val auc {val_auc:.4f} best_iter {best} ({time.time()-started:.0f}s)")
    return pred, val_auc, best, round(time.time() - started, 1)


def summarise(y, pred, classes, present):
    out = {"overall_auc": float(roc_auc_score(y, pred))}
    for name in CLASSES:
        m = classes == name
        if m.sum() >= 50:
            out.setdefault("by_participation_scale", {})[name] = {
                "n": int(m.sum()), "auc": float(roc_auc_score(y[m], pred[m]))}
    for name in CLASSES:
        m = present == name
        if m.sum() >= 50:
            out.setdefault("by_presence_scale", {})[name] = {
                "n": int(m.sum()), "auc": float(roc_auc_score(y[m], pred[m]))}
    return out


def paired(y, a, b, groups, n_boot=300, seed=SEED):
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(len(uniq), size=len(uniq))])
        if len(set(y[idx].tolist())) < 2:
            continue
        draws.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    a_ = np.asarray(draws)
    return {"mean": float(a_.mean()), "lo": float(np.percentile(a_, 2.5)),
            "hi": float(np.percentile(a_, 97.5)), "n_boot": int(len(a_))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, default=Path("D:/LOL_Project/fusion_2615/corpus_shards_v33"))
    ap.add_argument("--matrix", type=Path, default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33_matrix.npy"))
    ap.add_argument("--out", type=Path, default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33.json"))
    ap.add_argument("--preds", type=Path, default=Path("D:/LOL_Project/fusion_2615/features/model_comparison_v33.preds.npz"))
    ap.add_argument("--y-key", default="y_market_event")
    ap.add_argument("--models", default="lead_only,linear,lgbm_paper,lgbm_deep")
    ap.add_argument("--train-patch", default="15.14")
    ap.add_argument("--val-patch", default="15.15")
    ap.add_argument("--test-patch", default="15.16")
    ap.add_argument("--ft-top-k", type=int, default=256)
    a = ap.parse_args()
    log = lambda *s: print(*s, flush=True)

    verify_shard_manifest(a.shards)
    data = merge_shards(a.shards, a.matrix, y_key=a.y_key)
    valid = data["y"] >= 0
    X, y, groups = data["X"], data["y"], data["groups"]
    if not valid.all():
        idx = np.flatnonzero(valid)
        log(f"label {a.y_key}: dropping {int((~valid).sum())} unlabelled rows (draws)")
        X, y, groups = X[idx], y[idx], groups[idx]
        for k in ("cluster_blue", "cluster_red", "present_blue", "present_red", "patch"):
            data[k] = data[k][idx]
    y = y.astype(np.int8)
    classes = scale_class(data["cluster_blue"], data["cluster_red"], teamfight_min=4)
    present = scale_class(data["present_blue"], data["present_red"], teamfight_min=4)
    names = data["feature_names"] or [f"f{i}" for i in range(X.shape[1])]
    log(f"matrix {X.shape} rows={len(y)} matches={len(set(groups.tolist()))} positive={y.mean():.4f}")

    patch = data["patch"]
    tr = patch == a.train_patch
    va = patch == a.val_patch
    te = patch == a.test_patch
    if not (tr.any() and va.any() and te.any()):
        raise SystemExit(f"empty patch split; patches present: {sorted(set(patch.tolist()))}")
    log(f"patch split  train {a.train_patch} n={tr.sum():,}  val {a.val_patch} n={va.sum():,}  "
        f"test {a.test_patch} n={te.sum():,}")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        log(f"  {nm:5s} matches={len(set(groups[m].tolist())):,} positive={y[m].mean():.4f}")
    lead_cols = [i for i, n in enumerate(names)
                 if n in ("goldDiff__last", "xpDiff__last", "avgLevelDiff__last", "killDiff_cum__last",
                          "csDiff_total__last", "aliveDiff__last", "time_norm__last", "frame_age_s")]
    log(f"lead-only columns: {[names[i] for i in lead_cols]}")

    results, preds = {}, {}
    out_path = a.out
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8")).get("models", {})
    if a.preds.exists():
        with np.load(a.preds, allow_pickle=False) as z:
            preds = {k: z[k] for k in z.files}

    for name in a.models.split(","):
        name = name.strip()
        if not name:
            continue
        log(f"\n=== {name} ===")
        cols = None
        if name == "lead_only":
            cols = lead_cols
        elif name == "ft_transformer":  # pragma: no cover - kept for the gain-ranking helper only
            gain_path = a.out.with_name("model_comparison_v33_gain.json")
            if gain_path.exists():
                top = json.loads(gain_path.read_text(encoding="utf-8"))["top_columns"][:a.ft_top_k]
            else:
                m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=SEED,
                                   n_jobs=-1, verbose=-1).fit(X[tr], y[tr])
                order = np.argsort(m.booster_.feature_importance("gain"))[::-1]
                top = [int(i) for i in order[:max(a.ft_top_k, 512)]]
                gain_path.write_text(json.dumps({"top_columns": top,
                                                 "top_names": [names[i] for i in top[:64]]}, indent=2), encoding="utf-8")
            cols = top[:a.ft_top_k]
            log(f"  top-{len(cols)} features by gain, e.g. {[names[i] for i in cols[:6]]}")
        pred, val_auc, best_iter, secs = run_model(name, X, y, tr, va, te, cols=cols, log=log)
        preds[name] = pred
        results[name] = {**summarise(y[te], pred, classes[te], present[te]),
                         "val_auc": val_auc, "best_iteration": best_iter, "seconds": secs,
                         "n_features": int(len(cols) if cols is not None else X.shape[1])}
        log(f"  {name}: TEST({a.test_patch}) AUC {results[name]['overall_auc']:.4f} "
            f"| val({a.val_patch}) {val_auc:.4f} ({secs:.0f}s)")
        np.savez_compressed(a.preds, y=y[te], groups=groups[te], classes=classes[te].astype(str),
                            present=present[te].astype(str), **preds)
        payload = {"y_key": a.y_key, "n": int(len(y)), "n_matches": int(len(set(groups.tolist()))),
                   "n_features": int(X.shape[1]), "seed": SEED,
                   "split": {"kind": "patch holdout", "train": a.train_patch, "val": a.val_patch,
                             "test": a.test_patch,
                             "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
                             "why": "a random match split lets a learner see the patch it is tested on"},
                   "corpus_manifest": json.loads((a.shards / "manifest.json").read_text(encoding="utf-8")),
                   "models": results}
        ref = "lgbm_paper" if "lgbm_paper" in preds else None
        if ref:
            payload["vs_lgbm_paper"] = {k: paired(y[te], preds[ref], v, groups[te])
                                        for k, v in preds.items() if k != ref}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("\n" + json.dumps({k: {"auc": round(v["overall_auc"], 4), "s": v["seconds"]}
                           for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
