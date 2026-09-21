#!/usr/bin/env python3
"""E1 — LGBM info-group arms M-F0..M-F3 per cohort.

Contract: docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md §4
Task: .ai/tasks/T018.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

GROUPS = REPO / "docs/supplementary_e1_20260921/feature_groups.json"
OUT_ROOT = REPO / "outputs/supplementary_e1_infogroups_20260921"

CANDIDATES = [
    dict(num_leaves=15, min_child_samples=100),
    dict(num_leaves=15, min_child_samples=300),
    dict(num_leaves=31, min_child_samples=100),
    dict(num_leaves=31, min_child_samples=300),
]


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


def mean_one_match_weights(g: np.ndarray) -> np.ndarray:
    w = match_weights(g)
    return w * (len(w) / np.sum(w))


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def brier(y, p, g) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = match_weights(g)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def logloss(y, p, g) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    w = match_weights(g)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(w * ll) / np.sum(w))


def load_role_pack(lab_path: Path, D, L, roles, data_root: Path):
    """Reuse rr12 join semantics via primary_fit helper if available."""
    from rr20260920_q_newv_primary_fit import load_role_pack as _load

    return _load(lab_path, D, L, roles, data_root)


def matrix_for_arm(
    X: np.ndarray, cols: List[str], p_pre: np.ndarray, col_names: List[str]
) -> np.ndarray:
    ix = [cols.index(n) for n in col_names]
    return np.column_stack([X[:, ix], p_pre.reshape(-1, 1)]).astype(np.float64)


def fit_select_arm(
    arm: str,
    col_names: List[str],
    packs: Dict[str, Dict[str, Any]],
    cols: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    import lightgbm as lgb
    import joblib

    TR, CA, SE, TE = packs["TR"], packs["CA"], packs["SE"], packs["TE"]
    Xtr = matrix_for_arm(TR["X"], cols, TR["p_pre"], col_names)
    Xca = matrix_for_arm(CA["X"], cols, CA["p_pre"], col_names)
    Xse = matrix_for_arm(SE["X"], cols, SE["p_pre"], col_names)
    Xte = matrix_for_arm(TE["X"], cols, TE["p_pre"], col_names)
    w_tr = mean_one_match_weights(TR["g"])
    w_ca = match_weights(CA["g"])  # eval metric weighting on CAL

    results = []
    best = None
    for i, hp in enumerate(CANDIDATES):
        cand_id = f"nl{hp['num_leaves']}_mcs{hp['min_child_samples']}"
        print(f"  {arm} candidate {cand_id}…", flush=True)
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=hp["num_leaves"],
            min_child_samples=hp["min_child_samples"],
            reg_lambda=5.0,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=7,
            n_jobs=4,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                Xtr,
                TR["y"],
                sample_weight=w_tr,
                eval_set=[(Xca, CA["y"])],
                eval_sample_weight=[w_ca],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        p_se = model.predict_proba(Xse)[:, 1]
        p_te = model.predict_proba(Xte)[:, 1]
        row = dict(
            cand_id=cand_id,
            hp=hp,
            best_iteration=int(getattr(model, "best_iteration_", model.n_estimators)),
            select_brier=brier(SE["y"], p_se, SE["g"]),
            select_logloss=logloss(SE["y"], p_se, SE["g"]),
            test_brier=brier(TE["y"], p_te, TE["g"]),
            test_logloss=logloss(TE["y"], p_te, TE["g"]),
            n_features=int(Xtr.shape[1]),
        )
        results.append(row)
        path = out_dir / f"{arm}_{cand_id}.joblib"
        joblib.dump(
            dict(kind="lgbm_infogroup", arm=arm, model=model, columns=col_names + ["p_pre"], hp=hp),
            path,
        )
        row["model_path"] = str(path.relative_to(REPO)).replace("\\", "/")
        if best is None or (
            row["select_brier"] < best["select_brier"] - 1e-15
            or (
                abs(row["select_brier"] - best["select_brier"]) <= 1e-15
                and (
                    row["select_logloss"] < best["select_logloss"] - 1e-15
                    or (
                        abs(row["select_logloss"] - best["select_logloss"]) <= 1e-15
                        and cand_id < best["cand_id"]
                    )
                )
            )
        ):
            best = row
            best_model = model
            best_p_te = p_te

    assert best is not None
    # save selected predictions on TEST
    np.savez_compressed(
        out_dir / f"{arm}_TEST_pred.npz",
        match=TE["g"].astype(str),
        y=TE["y"].astype(np.int8),
        p=best_p_te.astype(np.float64),
        B40=TE.get("B40", np.zeros(len(TE["y"]), dtype=np.int8)),
    )
    return dict(arm=arm, selected=best, candidates=results, n_features=Xtr.shape[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["T", "S"], required=True)
    args = ap.parse_args(argv)

    if not GROUPS.is_file():
        raise SystemExit(f"missing {GROUPS}; run e1_build_feature_groups.py first")
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))

    lab_root = (
        REPO / "outputs/q_newv_fit85_20260920/labels"
        if args.cohort == "T"
        else REPO / "outputs/q_newv_fit85_20260920_S/labels"
    )
    out_dir = OUT_ROOT / args.cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import fc20260915_data as D

    L = D.Layout(False)
    train_roles = [f"fold{k}" for k in range(C.N_FOLDS)]
    print(f"E1 load packs cohort={args.cohort}…", flush=True)
    TR = load_role_pack(lab_root / "TRAIN_oof_h90.npz", D, L, train_roles, data_root)
    CA = load_role_pack(lab_root / "Q_CAL_h90.npz", D, L, ["Q_CAL"], data_root)
    SE = load_role_pack(lab_root / "Q_SELECT_h90.npz", D, L, ["Q_SELECT"], data_root)
    TE = load_role_pack(lab_root / "TEST_h90.npz", D, L, ["TEST"], data_root)
    cols = TR["cols"]
    packs = dict(TR=TR, CA=CA, SE=SE, TE=TE)

    # attach B40 if present in label file
    for name, path in (
        ("TR", lab_root / "TRAIN_oof_h90.npz"),
        ("CA", lab_root / "Q_CAL_h90.npz"),
        ("SE", lab_root / "Q_SELECT_h90.npz"),
        ("TE", lab_root / "TEST_h90.npz"),
    ):
        z = np.load(path, allow_pickle=False)
        if "B40" in z.files:
            # join order already aligned inside load_role_pack — B40 on label rows
            # load_role_pack returns filtered rows; re-read via pack length only if keys match
            pass

    summary = dict(
        cohort=args.cohort,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        contract="docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#4",
        arms={},
    )
    for arm, spec in groups["arms"].items():
        print(f"=== {arm} dim={spec['expected_dim']} ===", flush=True)
        summary["arms"][arm] = fit_select_arm(arm, spec["columns_from_X"], packs, cols, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
