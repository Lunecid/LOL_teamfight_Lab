#!/usr/bin/env python3
"""Fit 5-fold OOF shared_lgbm adapters on TRAIN (same hyperparams as freeze).

Uses fc20260915_common.train_fold hash. Each fold model is fit on TRAIN matches
with fold != k; scores held-out TRAIN engagements for OOF labels.

Calibration: reuse freeze PosSlopeSigmoid (single g) so sign(Δ) stays monotone
across OOF/final (contract §7).

Writes:
  outputs/v_redesign_20260919/models/oof_fold{0..4}.joblib
  outputs/v_redesign_20260919/labels/MAIN_TRAIN_labels.npz
  outputs/v_redesign_20260919/oof_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROLE = "EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE"
HS = (60, 90, 120)
N_FOLDS = 5


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(data_root / "worktrees" / "engagement-state-value"))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def match_weights(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g).astype(str)
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / cnt[inv]).astype(np.float64)


def expanded_X(X, names):
    keep = [i for i, n in enumerate(names) if n != "snapshot_age_s"]
    cols = [names[i] for i in keep]
    champ_ix = [j for j, c in enumerate(cols) if c.endswith("_champion_id")]
    return X[:, keep].astype(np.float64, copy=False), cols, champ_ix


def fit_lgbm(X, y, g, champ_ix, cols, seed=7):
    import lightgbm as lgb
    import pandas as pd
    w = match_weights(g)
    df = pd.DataFrame(X, columns=list(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in champ_ix:
            df[cols[j]] = df[cols[j]].astype("category")
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(df, y, sample_weight=w)
    return clf


def predict_lgbm(clf, X, cols, champ_ix):
    import pandas as pd
    df = pd.DataFrame(X, columns=list(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in champ_ix:
            df[cols[j]] = df[cols[j]].astype("category")
        return clf.predict_proba(df)[:, 1]


class PosSlopeSigmoid:
    def __init__(self, coef: float, intercept: float, ok: bool = True):
        self.coef_, self.intercept_, self.ok = coef, intercept, ok

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        z = self.coef_ * logit + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score(X, names, clf, cols, champ_ix, calib) -> np.ndarray:
    Xe, c2, _ = expanded_X(X, names)
    assert c2 == cols
    return calib.transform(predict_lgbm(clf, Xe, cols, champ_ix))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--v-dir", type=Path, default=REPO / "outputs" / "v_redesign_20260919")
    ap.add_argument("--skip-fit", action="store_true", help="reuse existing oof_fold*.joblib")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args(argv)
    data_root = args.data_root or _data_root()
    _setup(data_root)
    v_dir = args.v_dir
    model_dir = v_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = v_dir / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    import fc20260915_common as C
    import fc20260915_data as D

    freeze = json.loads((v_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    calib = PosSlopeSigmoid(
        float(freeze["calibration"]["coef"]),
        float(freeze["calibration"]["intercept"]),
        bool(freeze["calibration"]["ok"]),
    )

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(N_FOLDS)]
    print("load TRAIN bucket rows (fold0..4)…", flush=True)
    TR = D.load_v_rows(L, "MAIN", train_roles, bucket_only=True)
    names = list(TR["names"])
    W = D.load_outcomes(
        L, "MAIN", train_roles,
        purpose="V redesign OOF adapters (exploratory; prior TEST exposure elsewhere)",
    )
    X, cols, champ_ix = expanded_X(TR["X"], names)
    g = TR["match"].astype(str)
    y = np.asarray([W[m][0] for m in g.tolist()], dtype=np.int8)
    # Prefer corpus fold from sub_role when present; else hash fold
    if "sub_role" in TR and len(TR["sub_role"]) == len(g):
        sr = TR["sub_role"].astype(str)
        folds = np.array([int(s.replace("fold", "")) if s.startswith("fold") else C.train_fold(m)
                          for s, m in zip(sr.tolist(), g.tolist())], dtype=np.int8)
    else:
        folds = np.array([C.train_fold(m) for m in g.tolist()], dtype=np.int8)
    print(f"  rows={len(y)} matches={len(np.unique(g))} fold counts={np.bincount(folds, minlength=5).tolist()}", flush=True)

    oof_paths = {}
    oof_sha = {}
    if not args.skip_fit:
        for k in range(N_FOLDS):
            t0 = time.time()
            m = folds != k
            print(f"fit OOF fold {k} on {int(m.sum())} rows…", flush=True)
            clf = fit_lgbm(X[m], y[m], g[m], champ_ix, cols, seed=7 + k)
            path = model_dir / f"oof_fold{k}.joblib"
            joblib.dump({"model": clf, "cols": cols, "champ_ix": champ_ix, "fold": k, "kind": "shared_lgbm_oof"}, path)
            try:
                oof_paths[str(k)] = str(path.relative_to(REPO))
            except ValueError:
                oof_paths[str(k)] = str(path)
            oof_sha[str(k)] = file_sha256(path)
            print(f"  wrote {path.name} in {time.time()-t0:.1f}s sha={oof_sha[str(k)][:12]}", flush=True)
    else:
        for k in range(N_FOLDS):
            path = model_dir / f"oof_fold{k}.joblib"
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            oof_paths[str(k)] = str(path)
            oof_sha[str(k)] = file_sha256(path)

    adapters = {k: joblib.load(model_dir / f"oof_fold{k}.joblib") for k in range(N_FOLDS)}

    if not args.skip_labels:
        print("load TRAIN engagements (fold0..4)…", flush=True)
        E = D.load_engagements(L, "MAIN", train_roles, states=True, counts=True)
        n = len(E["match"])
        em = E["match"].astype(str)
        if "sub_role" in E:
            sr = E["sub_role"].astype(str)
            e_fold = np.array([int(s.replace("fold", "")) if s.startswith("fold") else C.train_fold(m)
                               for s, m in zip(sr.tolist(), em.tolist())], dtype=np.int8)
        else:
            e_fold = np.array([C.train_fold(m) for m in em.tolist()], dtype=np.int8)
        p_pre = np.full(n, np.nan)
        p_post = {h: np.full(n, np.nan) for h in HS}
        adapter_id = np.empty(n, dtype="U16")
        adapter_sha = np.empty(n, dtype="U64")
        for k in range(N_FOLDS):
            rows = e_fold == k
            adapter_id[rows] = f"oof_{k}"
            adapter_sha[rows] = oof_sha[str(k)]
            clf = adapters[k]["model"]
            pre = rows & (E["pre_ok"] == 1)
            if pre.any():
                p_pre[pre] = score(E["X_pre"][pre], names, clf, cols, champ_ix, calib)
            for h in HS:
                v = rows & (E[f"valid_h{h}"] == 1)
                if v.any():
                    p_post[h][v] = score(E[f"X_post_h{h}"][v], names, clf, cols, champ_ix, calib)
            print(f"  scored fold {k}: n={int(rows.sum())}", flush=True)

        arrays: Dict[str, Any] = dict(
            match=E["match"],
            sub_role=E["sub_role"],
            adapter_id=adapter_id,
            adapter_sha256=adapter_sha,
            p_pre=p_pre,
            model_version=np.asarray("v_redesign_shared_lgbm_oof_20260919"),
            state_version=np.asarray(C.STATE_VERSION),
            v_lineage=np.asarray("shared_lgbm_oof"),
        )
        for key in D.E_SCALAR + ("pre_reason",):
            arrays[key] = E[key]
        checks = {}
        for h in HS:
            v = E[f"valid_h{h}"] == 1
            delta = np.where(v, p_post[h] - p_pre, np.nan)
            Y = np.where(v, (delta > 0).astype(np.int8), -1).astype(np.int8)
            checks[f"h{h}"] = dict(
                valid_rows=int(v.sum()),
                positive=int(np.sum(Y[v] == 1)) if v.any() else 0,
                mean_abs_delta=float(np.mean(np.abs(delta[v]))) if v.any() else None,
            )
            arrays[f"endpoint_h{h}"] = E[f"endpoint_h{h}"]
            arrays[f"valid_h{h}"] = E[f"valid_h{h}"]
            arrays[f"p_post_h{h}"] = p_post[h]
            arrays[f"delta_h{h}"] = delta
            arrays[f"Y_h{h}"] = Y
            arrays[f"after_counts_h{h}"] = E[f"after_h{h}"]
        arrays["during_counts"] = E["during"]
        arrays["count_keys"] = np.asarray(E["count_keys"])
        out = lab_dir / "MAIN_TRAIN_labels.npz"
        np.savez_compressed(out, **arrays)
        train_sha = file_sha256(out)
        print(f"wrote {out} n={n} h90_pos={checks['h90']['positive']}", flush=True)
    else:
        train_sha = None
        checks = {}

    man = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        n_folds=N_FOLDS,
        fold_rule="fc20260915_common.train_fold",
        calib="freeze PosSlopeSigmoid (shared g)",
        oof_paths=oof_paths,
        oof_sha256=oof_sha,
        train_labels_sha256=train_sha,
        checks=checks,
    )
    (v_dir / "oof_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    # patch freeze deferred
    freeze["deferred"]["train_oof_adapters"] = "DONE — see oof_manifest.json + labels/MAIN_TRAIN_labels.npz"
    freeze["oof_manifest"] = "outputs/v_redesign_20260919/oof_manifest.json"
    (v_dir / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print("wrote oof_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
