from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.fight_types import FightRef, ref_key
from data.labels import aligned_xy_from_maps, get_label_map, get_label_map_from_dataset
from core.utils import confusion_from_probs, metrics_from_probs, pretty_cm, save_json, write_log
from train.fusion_calibration import calibrate_logits_by_patch, compute_ece, find_optimal_temperature
from train.fusion_helpers import (
    StackingResult,
    _eval_single_base_auc,
    _fit_logreg,
    _logit_from_prob,
    _predict_proba,
    _safe_tag,
    _sanitize_meta_X,
    split_logit_map_by_refs,
)


# ---------------------------------------------------------------------
# Stacking: simple (fit on VAL, evaluate on TEST)
# ---------------------------------------------------------------------
def stack_simple(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    feature_set: str,
    base_names: List[str],
    base_maps: List[Dict[str, float]],
    out_dir: Path,
    log_fp: Optional[Path] = None,
    seed: int = 0,
    meta_method: str = "logreg",
    fit_on: str = "train",
    y_tr_map: Optional[Dict[str, int]] = None,
    y_va_map: Optional[Dict[str, int]] = None,
    y_te_map: Optional[Dict[str, int]] = None,
) -> StackingResult:
    """Fit meta learner on one split, evaluate on TRAIN/VAL/TEST.

    - `fit_on="train"`: fit on TRAIN, evaluate on VAL/TEST (selection-safe).
    - `fit_on="val"`: fit on VAL (legacy mode; can inflate VAL).
    - We report TRAIN metrics by *applying* the fitted meta model to TRAIN.
      This resolves the previous issue where greedy tried to read train AUC but
      stack_simple did not compute it.

    Notes
    -----
    - Base models are assumed already trained; base_maps are per-ref logits.
    - aligned_xy_from_maps decides which refs are usable (intersection of y_map and base_maps).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fp is not None:
            write_log(msg, log_fp)

    if y_tr_map is None:
        y_tr_map = get_label_map(tr_refs, feature_set, log_fp=log_fp, log_every=50000)
    if y_va_map is None:
        y_va_map = get_label_map(va_refs, feature_set, log_fp=log_fp, log_every=20000)
    if y_te_map is None:
        y_te_map = get_label_map(te_refs, feature_set, log_fp=log_fp, log_every=20000)

    Xtr, ytr, _ = aligned_xy_from_maps(tr_refs, y_tr_map, base_maps)
    Xva, yva, keys_va = aligned_xy_from_maps(va_refs, y_va_map, base_maps)
    Xte, yte, keys_te = aligned_xy_from_maps(te_refs, y_te_map, base_maps)

    # Diagnostics BEFORE sanitizing
    for split_name, X in (("TRAIN", Xtr), ("VAL", Xva), ("TEST", Xte)):
        if X is None or not isinstance(X, np.ndarray):
            continue
        if np.isnan(X).any() or np.isinf(X).any():
            nan_cols = np.isnan(X).sum(axis=0)
            inf_cols = np.isinf(X).sum(axis=0)
            _log(
                f"[FUSION] {split_name} meta X has NaN/Inf. "
                f"nan_total={int(np.isnan(X).sum())}, inf_total={int(np.isinf(X).sum())}, "
                f"nan_cols={nan_cols.tolist()}, inf_cols={inf_cols.tolist()}"
            )

    # Sanitize
    Xtr = _sanitize_meta_X(Xtr, clip=20.0)
    Xva = _sanitize_meta_X(Xva, clip=20.0)
    Xte = _sanitize_meta_X(Xte, clip=20.0)

    fit_on_norm = str(fit_on or "val").strip().lower()
    if fit_on_norm not in ("val", "train"):
        _log(f"[FUSION] invalid fit_on={fit_on!r} (expected 'val' or 'train')")
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    Xfit, yfit = (Xva, yva) if fit_on_norm == "val" else (Xtr, ytr)
    if Xfit.shape[0] < 50 or Xva.shape[0] < 50 or Xte.shape[0] < 50:
        _log(
            f"[FUSION] too few samples "
            f"(fit={Xfit.shape[0]}@{fit_on_norm}, val={Xva.shape[0]}, test={Xte.shape[0]})"
        )
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    try:
        if meta_method != "logreg":
            raise ValueError(f"unsupported meta_method={meta_method} (only logreg supported here)")
        clf = _fit_logreg(Xfit, yfit, seed=seed)
    except Exception as e:
        _log(f"[FUSION] meta fit failed: {e}")
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    # Predict on all splits (if available)
    p_tr = _predict_proba(clf, Xtr) if Xtr.size else np.array([])
    p_va = _predict_proba(clf, Xva) if Xva.size else np.array([])
    p_te = _predict_proba(clf, Xte) if Xte.size else np.array([])

    met_tr = metrics_from_probs(ytr, p_tr, threshold=0.5) if p_tr.size else {}
    met_va = metrics_from_probs(yva, p_va, threshold=0.5) if p_va.size else {}
    met_te = metrics_from_probs(yte, p_te, threshold=0.5) if p_te.size else {}

    if p_tr.size:
        _log(f"[FUSION] TRAIN: auc={met_tr.get('auc'):.4f} {pretty_cm(confusion_from_probs(ytr, p_tr, 0.5))}")
    if p_va.size:
        _log(f"[FUSION] VAL  : auc={met_va.get('auc'):.4f} {pretty_cm(confusion_from_probs(yva, p_va, 0.5))}")
    if p_te.size:
        _log(f"[FUSION] TEST : auc={met_te.get('auc'):.4f} {pretty_cm(confusion_from_probs(yte, p_te, 0.5))}")

    # store test logits for downstream dumping
    z_te = _logit_from_prob(p_te) if p_te.size else np.array([])
    pred_map_te = {k: float(z) for k, z in zip(keys_te, z_te.tolist())} if z_te.size else {}

    # Bootstrap CI for AUC (val & test)
    bootstrap_ci_results: Dict[str, Any] = {}
    try:
        from app.experiment_stats import bootstrap_auc_ci as _boot_auc_ci

        if p_va.size and len(yva) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yva, p_va, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["val"] = {
                "auc": float(_auc), "ci_low": float(_ci_lo), "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo), "n_samples": len(yva),
                "method": "bootstrap_percentile", "n_bootstrap": 2000, "alpha": 0.05,
            }
            _log(f"[FUSION][BOOTSTRAP] val AUC={_auc:.4f} 95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]")
        if p_te.size and len(yte) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yte, p_te, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["test"] = {
                "auc": float(_auc), "ci_low": float(_ci_lo), "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo), "n_samples": len(yte),
                "method": "bootstrap_percentile", "n_bootstrap": 2000, "alpha": 0.05,
            }
            _log(f"[FUSION][BOOTSTRAP] test AUC={_auc:.4f} 95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]")
    except Exception as e:
        _log(f"[FUSION][BOOTSTRAP] CI computation failed (ignored): {e}")

    rep = {
        "ok": True,
        "mode": "simple",
        "meta_method": meta_method,
        "fit_on": fit_on_norm,
        "base_names": base_names,
        "metrics": {"train": met_tr, "val": met_va, "test": met_te},
        "n": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
        "bootstrap_ci": bootstrap_ci_results if bootstrap_ci_results else None,
    }
    save_json(out_dir / "report.json", rep)

    return StackingResult(
        ok=True,
        meta_method=meta_method,
        base_names=base_names,
        metrics=rep["metrics"],
        out_dir=str(out_dir),
        pred_logit_map=pred_map_te,
    )


#Ã«â€¹Â¨Ã¬ÂÂ¼ Ã¬â€žÂ±Ã«Å Â¥ Ã­â€”Â¬Ã­ÂÂ¼
def prune_correlated_columns(
    X: np.ndarray,
    names: List[str],
    thresh: float = 0.9,
    seed: int = 0,
) -> Tuple[List[int], List[str]]:
    """Prune highly correlated candidate columns.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_cols). Each column is a candidate base logit vector (aligned on the same keys).
    names : List[str]
        Candidate names (length n_cols).
    thresh : float
        Absolute correlation threshold above which we treat a pair as redundant.
    seed : int
        Random seed to break ties / define a deterministic traversal order.

    Returns
    -------
    keep_idx : List[int]
        Indices to keep (sorted in original order).
    dropped_names : List[str]
        Names that were dropped.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] <= 1 or X.shape[0] < 2:
        return list(range(int(X.shape[1]) if X.ndim == 2 else len(names))), []

    # sanitize to avoid nan correlations from inf/nan
    Xs = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # correlation across columns
    corr = np.corrcoef(Xs, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    p = int(corr.shape[0])
    order = np.arange(p)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(order)

    dropped = set()
    keep = []

    for j in order.tolist():
        if j in dropped:
            continue
        keep.append(int(j))
        # drop others highly correlated with j
        for k in order.tolist():
            if k == j or k in dropped:
                continue
            if abs(float(corr[j, k])) >= float(thresh):
                dropped.add(int(k))

    keep_sorted = sorted(keep)
    dropped_names = [str(names[i]) for i in range(len(names)) if i not in keep_sorted]
    return keep_sorted, dropped_names


# ---------------------------------------------------------------------
# Stacking: OOF meta (train OOF diagnostics + fit on train+val, eval test)
# ---------------------------------------------------------------------
def stack_oof_meta(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    feature_set: str,
    base_names: List[str],
    base_maps_tr: List[Dict[str, float]],
    base_maps_va: List[Dict[str, float]],
    base_maps_te: List[Dict[str, float]],
    folds: List[Tuple[List[FightRef], List[FightRef]]],
    out_dir: Path,
    log_fp: Path,
    seed: int,
    meta_method: str = "logreg",
    y_tr_map: Optional[Dict[str, int]] = None,
    y_va_map: Optional[Dict[str, int]] = None,
    y_te_map: Optional[Dict[str, int]] = None,
) -> StackingResult:
    """OOF-style stacking where only the meta learner is trained in folds.

    Requires base predictions for train refs (so deep models must provide train maps).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    if y_tr_map is None:
        y_tr_map = get_label_map(tr_refs, feature_set, log_fp=log_fp, log_every=50000)
    if y_va_map is None:
        y_va_map = get_label_map(va_refs, feature_set, log_fp=log_fp, log_every=20000)
    if y_te_map is None:
        y_te_map = get_label_map(te_refs, feature_set, log_fp=log_fp, log_every=20000)

    # OOF predictions on train for diagnostics
    oof_prob: Dict[str, float] = {}

    for fi, (tr_fold, ho_fold) in enumerate(folds):
        Xtr, ytr, _ = aligned_xy_from_maps(tr_fold, y_tr_map, base_maps_tr)
        Xho, yho, keys_ho = aligned_xy_from_maps(ho_fold, y_tr_map, base_maps_tr)
        if Xtr.shape[0] < 50 or Xho.shape[0] < 10:
            continue
        try:
            if meta_method != "logreg":
                raise ValueError(f"unsupported meta_method={meta_method} (only logreg supported here)")
            clf = _fit_logreg(_sanitize_meta_X(Xtr), ytr, seed=seed + fi)
        except Exception:
            continue
        p_ho = _predict_proba(clf, _sanitize_meta_X(Xho))
        for k, p in zip(keys_ho, p_ho.tolist()):
            oof_prob[k] = float(p)

    # compute OOF metric
    keys = [ref_key(r) for r in tr_refs if ref_key(r) in y_tr_map and ref_key(r) in oof_prob]
    if keys:
        y_oof = np.asarray([y_tr_map[k] for k in keys], dtype=np.int64)
        p_oof = np.asarray([oof_prob[k] for k in keys], dtype=np.float64)
        met_oof = metrics_from_probs(y_oof, p_oof, threshold=0.5)
        write_log(f"[FUSION-OOF] train OOF auc={met_oof.get('auc'):.4f} n={len(keys)}", log_fp)
    else:
        met_oof = {}

    # Final meta evaluation policy:
    #   - VAL metric: fit on TRAIN only (unbiased wrt VAL)
    #   - TEST metric: fit on TRAIN+VAL (deployment-like)
    Xtr_full, ytr_full, _ = aligned_xy_from_maps(tr_refs, y_tr_map, base_maps_tr)
    Xva, yva, _ = aligned_xy_from_maps(va_refs, y_va_map, base_maps_va)
    Xte, yte, keys_te = aligned_xy_from_maps(te_refs, y_te_map, base_maps_te)

    if Xtr_full.shape[0] < 50:
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    Xtr_full = _sanitize_meta_X(Xtr_full)
    Xva = _sanitize_meta_X(Xva) if Xva.size else Xva
    Xte = _sanitize_meta_X(Xte) if Xte.size else Xte

    # (A) VAL-safe model (train-only)
    try:
        if meta_method != "logreg":
            raise ValueError(f"unsupported meta_method={meta_method} (only logreg supported here)")
        clf_val = _fit_logreg(Xtr_full, ytr_full, seed=seed)
    except Exception as e:
        write_log(f"[FUSION-OOF] val-model fit failed: {e}", log_fp)
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    p_va = _predict_proba(clf_val, Xva) if Xva.size else np.array([])

    # (B) final model for TEST (train+val)
    Xfit_te = np.concatenate([Xtr_full, Xva], axis=0) if Xva.size else Xtr_full
    yfit_te = np.concatenate([ytr_full, yva], axis=0) if Xva.size else ytr_full
    try:
        if meta_method != "logreg":
            raise ValueError(f"unsupported meta_method={meta_method} (only logreg supported here)")
        clf_te = _fit_logreg(Xfit_te, yfit_te, seed=seed)
    except Exception as e:
        write_log(f"[FUSION-OOF] test-model fit failed: {e}", log_fp)
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    p_te = _predict_proba(clf_te, Xte) if Xte.size else np.array([])

    met_va = metrics_from_probs(yva, p_va, threshold=0.5) if Xva.size else {}
    met_te = metrics_from_probs(yte, p_te, threshold=0.5) if Xte.size else {}

    if Xva.size:
        write_log(f"[FUSION-OOF] VAL : auc={met_va.get('auc'):.4f} {pretty_cm(confusion_from_probs(yva, p_va, 0.5))}", log_fp)
    if Xte.size:
        write_log(f"[FUSION-OOF] TEST: auc={met_te.get('auc'):.4f} {pretty_cm(confusion_from_probs(yte, p_te, 0.5))}", log_fp)

    z_te = _logit_from_prob(p_te) if Xte.size else np.array([])
    pred_map_te = {k: float(z) for k, z in zip(keys_te, z_te.tolist())}

    rep = {
        "ok": True,
        "mode": "oof_meta",
        "meta_method": meta_method,
        "fit_policy": {"val": "train_only", "test": "train_plus_val"},
        "base_names": base_names,
        "metrics": {"train_oof": met_oof, "val": met_va, "test": met_te},
        "n": {"train_oof": len(keys), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
    }
    save_json(out_dir / "report.json", rep)

    return StackingResult(ok=True, meta_method=meta_method, base_names=base_names, metrics=rep["metrics"], out_dir=str(out_dir), pred_logit_map=pred_map_te)


#greedy AUC helper
def stack_greedy_forward(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    feature_set: str,
    cand_names: List[str],
    cand_maps: List[Dict[str, float]],
    out_dir: Path,
    log_fp: Optional[Path] = None,
    seed: int = 0,
    meta_method: str = "logreg",
    max_k: int = 4,  # Ã¬ÂµÅ“Ã«Å’â‚¬Ã«Â¡Å“ Ã«Âªâ€¡ ÃªÂ°Å“ Ã«ÂªÂ¨Ã«ÂÂ¸ÃªÂ¹Å’Ã¬Â§â‚¬ Ã­â€¢Â©Ã¬Â¹Â Ã¬Â§â‚¬
    anchor_name: Optional[str] = None,  # NoneÃ¬ÂÂ´Ã«Â©Â´ "TRAIN Ã«â€¹Â¨Ã¬ÂÂ¼Ã«ÂªÂ¨Ã«ÂÂ¸ Ã¬ÂµÅ“ÃªÂ³Â " Ã¬Å¾ÂÃ«Ââ„¢ Ã¬â€žÂ Ã­Æ’Â
    stop_when_no_improve: bool = True,
    min_improve: float = 1e-6,
    y_tr_map: Optional[Dict[str, int]] = None,
    y_va_map: Optional[Dict[str, int]] = None,
    y_te_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Greedy forward selection stacking (VAL-based).

    Procedure
    ---------
    0) (Optional) prune highly correlated base columns using VAL keys
    1) pick anchor by best single-model VAL AUC (or use anchor_name)
    2) add one model at a time that maximizes VAL AUC of the stacked meta model
       (meta is fit on TRAIN via stack_simple(fit_on="train"), evaluated on VAL)
    3) stop if no improvement (optional)

    Notes
    -----
    - Selection metric is VAL AUC, with meta fitted on TRAIN for each candidate
      combo to avoid fit/eval-on-the-same-split bias during search.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    assert len(cand_names) == len(cand_maps), "cand_names and cand_maps length mismatch"
    n0 = len(cand_names)
    if n0 == 0:
        return {"ok": False, "reason": "no_candidates"}

    def _log(msg: str) -> None:
        if log_fp is not None:
            write_log(msg, log_fp)

    # ------------------------------------------------------------
    # (A) Correlation pruning on VAL keys (val-centered, deterministic)
    # ------------------------------------------------------------
    try:
        keys_va = [ref_key(r) for r in va_refs]
        # dedupe while preserving order
        keys_va = list(dict.fromkeys(keys_va))
    except Exception:
        keys_va = []

    if keys_va and len(cand_maps) > 1:
        X_cand = np.stack([np.asarray([m.get(k, 0.0) for k in keys_va], dtype=np.float32) for m in cand_maps], axis=1)
        keep_idx, dropped = prune_correlated_columns(X_cand, cand_names, thresh=0.9, seed=int(seed))

        cand_names = [cand_names[i] for i in keep_idx]
        cand_maps = [cand_maps[i] for i in keep_idx]
        _log(f"[GREEDY] Pruned {len(dropped)} correlated bases: {dropped}")

        # Optional: correlation matrix (truncate for readability)
        if X_cand.shape[1] > 1:
            corr = np.corrcoef(np.nan_to_num(X_cand, nan=0.0, posinf=0.0, neginf=0.0).T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            p = int(corr.shape[0])
            show = min(p, 12)
            _log(f"[GREEDY] Candidate corr_matrix shape={corr.shape} (show {show}x{show})")
            _log(f"[GREEDY] corr_matrix[:{show}, :{show}] =\n{corr[:show, :show]}")
    else:
        _log("[GREEDY] skip correlation pruning (insufficient val keys or only 1 candidate)")

    # refresh after pruning
    n = len(cand_names)
    if n == 0:
        return {"ok": False, "reason": "all_pruned"}

    # ------------------------------------------------------------
    # (B) Precompute labels once (reuse pre-computed maps if passed)
    # ------------------------------------------------------------
    if y_va_map is None:
        y_va_map = get_label_map(va_refs, feature_set, log_fp=log_fp, log_every=20000)
    if y_te_map is None:
        y_te_map = get_label_map(te_refs, feature_set, log_fp=log_fp, log_every=20000)
    if y_tr_map is None:
        y_tr_map = get_label_map(tr_refs, feature_set, log_fp=log_fp, log_every=20000)

    _log(f"[GREEDY] Dataset sizes: tr={len(tr_refs)}, va={len(va_refs)}, te={len(te_refs)}")
    if len(va_refs) < 50:
        _log("[GREEDY][WARN] Small val set: may cause instability")

    # ------------------------------------------------------------
    # (C) Single-model scores (raw base AUC)
    # ------------------------------------------------------------
    singles: Dict[str, Any] = {}
    for nm, mp in zip(cand_names, cand_maps):
        met_tr = _eval_single_base_auc(tr_refs, y_tr_map, mp)
        met_va = _eval_single_base_auc(va_refs, y_va_map, mp)
        met_te = _eval_single_base_auc(te_refs, y_te_map, mp)
        singles[str(nm)] = {"train": met_tr, "val": met_va, "test": met_te}

    # ------------------------------------------------------------
    # (D) Choose anchor by VAL AUC  [FIX P0-1]
    # ------------------------------------------------------------
    if anchor_name is not None and anchor_name in cand_names:
        anchor_idx = cand_names.index(anchor_name)
    else:
        best_idx = 0
        best_auc = -1.0
        for i, nm in enumerate(cand_names):
            auc = singles.get(str(nm), {}).get("val", {}).get("auc", float("nan"))
            try:
                auc = float(auc)
            except Exception:
                auc = float("nan")
            auc = auc if np.isfinite(auc) else -1.0
            if auc > best_auc:
                best_auc = auc
                best_idx = i
        anchor_idx = best_idx

    anchor = cand_names[anchor_idx]
    _log(f"[GREEDY] anchor={anchor} (picked by best single VAL AUC)")

    selected: List[int] = [anchor_idx]
    remaining: List[int] = [i for i in range(n) if i != anchor_idx]

    steps: List[Dict[str, Any]] = []
    best_overall = {
        "train_auc": -1.0,
        "val_auc": -1.0,
        "test_auc": -1.0,
        "base_names": None,
        "out_dir": None,
    }
    best_pred_map: Optional[Dict[str, float]] = None

    def _run_combo(sel_idx: List[int], tag_prefix: str, local_seed: int):
        names = [cand_names[i] for i in sel_idx]
        maps = [cand_maps[i] for i in sel_idx]
        combo_tag = _safe_tag(tag_prefix + "__" + "__".join([str(x) for x in names]))
        sub_dir = out_dir / "greedy" / combo_tag

        rep = stack_simple(
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            feature_set=feature_set,
            base_names=names,
            base_maps=maps,
            out_dir=sub_dir,
            log_fp=log_fp,
            seed=local_seed,
            meta_method=meta_method,
            fit_on="train",
            y_tr_map=y_tr_map,
            y_va_map=y_va_map,
            y_te_map=y_te_map,
        )

        train_auc = float(rep.metrics.get("train", {}).get("auc", float("nan"))) if rep.ok else float("nan")
        val_auc = float(rep.metrics.get("val", {}).get("auc", float("nan"))) if rep.ok else float("nan")
        te_auc = float(rep.metrics.get("test", {}).get("auc", float("nan"))) if rep.ok else float("nan")
        return rep, combo_tag, sub_dir, train_auc, val_auc, te_auc

    # step 1: anchor alone
    rep1, tag1, dir1, tr1, v1, t1 = _run_combo(selected, "k1", int(seed))
    steps.append({
        "k": 1,
        "tag": tag1,
        "base_names": [cand_names[i] for i in selected],
        "train_auc": tr1,
        "val_auc": v1,
        "test_auc": t1,
        "ok": rep1.ok,
        "out_dir": str(dir1),
    })

    # [FIX P0-1] Use VAL AUC for selection criterion instead of TRAIN AUC
    if rep1.ok and np.isfinite(v1) and v1 > best_overall["val_auc"]:
        best_overall = {
            "train_auc": tr1,
            "val_auc": v1,
            "test_auc": t1,
            "base_names": [cand_names[i] for i in selected],
            "out_dir": str(dir1),
        }
        best_pred_map = rep1.pred_logit_map

    cur_best_val = v1 if np.isfinite(v1) else -1.0

    # steps 2..max_k
    for k in range(2, int(max_k) + 1):
        if not remaining:
            break

        best_add = None  # (val_auc, train_auc, te_auc, idx, rep, tag, dir)
        for idx in list(remaining):
            trial_sel = selected + [idx]
            rep, tag, d, tr_auc, v, t = _run_combo(trial_sel, f"k{k}", int(seed) + 1000 * k + idx)

            _log(
                f"[GREEDY] try k={k} add={cand_names[idx]} "
                f"train_auc={tr_auc:.4f} val_auc={v:.4f} test_auc={t:.4f} ok={rep.ok}"
            )

            if (not rep.ok) or (not np.isfinite(v)):
                continue

            # [FIX P0-1] Selection by VAL AUC
            if (best_add is None) or (v > best_add[0]) or (
                v == best_add[0] and np.isfinite(t) and t > best_add[2]
            ):
                best_add = (v, tr_auc, t, idx, rep, tag, d)

        if best_add is None:
            _log(f"[GREEDY] stop: no valid addition at k={k}")
            break

        best_v, best_tr, best_t, best_idx_add, best_rep, best_tag, best_dir = best_add

        # [FIX P0-1] Stop if no improvement (VAL-based)
        if stop_when_no_improve and (best_v <= cur_best_val + float(min_improve)):
            _log(f"[GREEDY] stop: no improvement (cur_val={cur_best_val:.6f} best_next_val={best_v:.6f})")
            break

        # accept
        selected.append(int(best_idx_add))
        remaining.remove(int(best_idx_add))
        cur_best_val = float(best_v)

        steps.append({
            "k": k,
            "tag": best_tag,
            "base_names": [cand_names[i] for i in selected],
            "train_auc": float(best_tr),
            "val_auc": float(best_v),
            "test_auc": float(best_t),
            "ok": bool(best_rep.ok),
            "out_dir": str(best_dir),
        })

        # [FIX P0-1] Best overall by VAL AUC
        if np.isfinite(best_v) and best_v > best_overall["val_auc"]:
            best_overall = {
                "train_auc": float(best_tr),
                "val_auc": float(best_v),
                "test_auc": float(best_t),
                "base_names": [cand_names[i] for i in selected],
                "out_dir": str(best_dir),
            }
            best_pred_map = best_rep.pred_logit_map

        _log(
            f"[GREEDY] ACCEPT k={k} add={cand_names[best_idx_add]} -> "
            f"train_auc={best_tr:.4f} val_auc={best_v:.4f} test_auc={best_t:.4f}"
        )

    summary_disk = {
        "ok": True,
        "mode": "greedy_forward",
        "candidate_names": cand_names,
        "single_scores": singles,
        "anchor": str(anchor),
        "max_k": int(max_k),
        "steps": steps,
        "best": best_overall,
    }
    save_json(out_dir / "greedy_summary.json", summary_disk)

    # return best_pred_map in-memory only (avoid huge json)
    summary_disk["best_pred_logit_map"] = best_pred_map
    return summary_disk



def refit_meta_trainval_predict_test(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    feature_set: str,
    base_names: List[str],
    base_maps: List[Dict[str, float]],
    out_dir: Path,
    log_fp: Path,
    seed: int,
    meta_method: str = "logreg",
    y_tr_map: Optional[Dict[str, int]] = None,
    y_va_map: Optional[Dict[str, int]] = None,
    y_te_map: Optional[Dict[str, int]] = None,
) -> StackingResult:
    """
    FINAL RE-FIT:
      - Fit meta learner on (TRAIN + VAL)
      - Predict TEST
      - Save report + return test pred_logit_map

    NOTE:
      This requires base_maps to contain predictions for tr_refs/va_refs/te_refs keys.
      If train coverage is too small, it falls back to VAL-only fit (stack_simple).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels (compute once; reuse pre-computed maps if passed)
    if y_tr_map is None:
        y_tr_map = get_label_map(tr_refs, feature_set, log_fp=log_fp, log_every=50000)
    if y_va_map is None:
        y_va_map = get_label_map(va_refs, feature_set, log_fp=log_fp, log_every=20000)
    if y_te_map is None:
        y_te_map = get_label_map(te_refs, feature_set, log_fp=log_fp, log_every=20000)

    # aligned meta features
    Xtr, ytr, _ = aligned_xy_from_maps(tr_refs, y_tr_map, base_maps)
    Xva, yva, _ = aligned_xy_from_maps(va_refs, y_va_map, base_maps)
    Xte, yte, keys_te = aligned_xy_from_maps(te_refs, y_te_map, base_maps)

    # if train coverage is too small, fallback
    if Xtr.shape[0] < 50:
        write_log(
            f"[REFIT] train coverage too small (Xtr={Xtr.shape[0]}). Fallback to VAL-only fit.",
            log_fp,
        )
        return stack_simple(
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            feature_set=feature_set,
            base_names=base_names,
            base_maps=base_maps,
            out_dir=out_dir / "fallback_val_only",
            log_fp=log_fp,
            seed=seed,
            meta_method=meta_method,
            fit_on="val",
            y_tr_map=y_tr_map,
            y_va_map=y_va_map,
            y_te_map=y_te_map,
        )

    # sanitize
    Xtr = _sanitize_meta_X(Xtr, clip=20.0)
    Xva = _sanitize_meta_X(Xva, clip=20.0) if Xva.size else Xva
    Xte = _sanitize_meta_X(Xte, clip=20.0) if Xte.size else Xte

    Xfit = np.concatenate([Xtr, Xva], axis=0) if Xva.size else Xtr
    yfit = np.concatenate([ytr, yva], axis=0) if Xva.size else ytr

    try:
        if meta_method != "logreg":
            raise ValueError(f"unsupported meta_method={meta_method} (only logreg supported here)")
        clf = _fit_logreg(Xfit, yfit, seed=seed)
    except Exception as e:
        write_log(f"[REFIT] meta fit failed: {e}", log_fp)
        return StackingResult(ok=False, meta_method=meta_method, base_names=base_names, metrics={}, out_dir=str(out_dir))

    # predict (val optional; test required)
    p_va = _predict_proba(clf, Xva) if Xva.size else np.array([])
    p_te = _predict_proba(clf, Xte) if Xte.size else np.array([])

    met_va = metrics_from_probs(yva, p_va, threshold=0.5) if Xva.size else {}
    met_te = metrics_from_probs(yte, p_te, threshold=0.5) if Xte.size else {}

    if Xva.size:
        write_log(f"[REFIT] VAL : auc={met_va.get('auc'):.4f} {pretty_cm(confusion_from_probs(yva, p_va, 0.5))}", log_fp)
    if Xte.size:
        write_log(f"[REFIT] TEST: auc={met_te.get('auc'):.4f} {pretty_cm(confusion_from_probs(yte, p_te, 0.5))}", log_fp)

    # store test logits
    z_te = _logit_from_prob(p_te) if Xte.size else np.array([])
    pred_map_te = {k: float(z) for k, z in zip(keys_te, z_te.tolist())}

    # Bootstrap CI for AUC (val & test)
    bootstrap_ci_results: Dict[str, Any] = {}
    try:
        from app.experiment_stats import bootstrap_auc_ci as _boot_auc_ci

        if p_va.size and len(yva) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yva, p_va, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["val"] = {
                "auc": float(_auc), "ci_low": float(_ci_lo), "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo), "n_samples": len(yva),
                "method": "bootstrap_percentile", "n_bootstrap": 2000, "alpha": 0.05,
            }
            write_log(f"[REFIT][BOOTSTRAP] val AUC={_auc:.4f} 95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]", log_fp)
        if p_te.size and len(yte) >= 20:
            _auc, _ci_lo, _ci_hi = _boot_auc_ci(yte, p_te, n_bootstrap=2000, alpha=0.05, seed=seed)
            bootstrap_ci_results["test"] = {
                "auc": float(_auc), "ci_low": float(_ci_lo), "ci_high": float(_ci_hi),
                "ci_width": float(_ci_hi - _ci_lo), "n_samples": len(yte),
                "method": "bootstrap_percentile", "n_bootstrap": 2000, "alpha": 0.05,
            }
            write_log(f"[REFIT][BOOTSTRAP] test AUC={_auc:.4f} 95% CI=[{_ci_lo:.4f}, {_ci_hi:.4f}]", log_fp)
    except Exception as e:
        write_log(f"[REFIT][BOOTSTRAP] CI computation failed (ignored): {e}", log_fp)

    rep = {
        "ok": True,
        "mode": "refit_trainval",
        "meta_method": meta_method,
        "base_names": base_names,
        "metrics": {"val": met_va, "test": met_te},
        "n": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
        "bootstrap_ci": bootstrap_ci_results if bootstrap_ci_results else None,
    }
    save_json(out_dir / "report.json", rep)

    return StackingResult(
        ok=True,
        meta_method=meta_method,
        base_names=base_names,
        metrics=rep["metrics"],
        out_dir=str(out_dir),
        pred_logit_map=pred_map_te,
    )

# ---------------------------------------------------------------------
# NEW: Factorial stacking (run many combinations)
# ---------------------------------------------------------------------
def stack_factorial(
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    feature_set: str,
    cand_names: List[str],
    cand_maps: List[Dict[str, float]],
    out_dir: Path,
    log_fp: Path,
    seed: int,
    meta_method: str = "logreg",
    min_k: int = 2,
    max_k: int = 3,
    anchor_name: Optional[str] = "lgbm",
    anchor_must_include: bool = True,
    max_combos: int = 300,
    k: Optional[int] = None,   # Ã¢Å“â€¦ Ã«Â Ë†ÃªÂ±Â°Ã¬â€¹Å“ Ã­ËœÂ¸Ã­â„¢Ëœ Ã¬Â¶â€ÃªÂ°â‚¬
    y_tr_map: Optional[Dict[str, int]] = None,
    y_va_map: Optional[Dict[str, int]] = None,
    y_te_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:

    """
    Factorial stacking over combinations of candidate base models.

    Parameters
    ----------
    cand_names / cand_maps:
        Candidate model names and their logit maps. Must be aligned in order.
        Each map is {ref_key: logit}.
    min_k / max_k:
        Minimum/maximum number of base models to combine.
        Default keeps runtime reasonable (2~3-way combos).
    anchor_name / anchor_must_include:
        If anchor_name exists among candidates and anchor_must_include=True,
        only combos that include the anchor are evaluated. (Typical: always include baseline 'lgbm'.)
    max_combos:
        Hard cap to avoid explosion.

    Returns
    -------
    summary dict saved to out_dir/factorial_summary.json.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Ã¢Å“â€¦ legacy compat: allow k -> (min_k=max_k=k)
    if k is not None:
        try:
            kk = int(k)
            if kk > 0:
                # if user explicitly passed min_k/max_k too, prefer min_k/max_k (no override)
                if (min_k is None) or (max_k is None):
                    min_k = kk
                    max_k = kk
                else:
                    # Ã¬ÂÂ¼Ã«Â°ËœÃ¬Â ÂÃ¬Å“Â¼Ã«Â¡Å“Ã«Å â€ kÃªÂ°â‚¬ Ã«â€œÂ¤Ã¬â€“Â´Ã¬ËœÂ¤Ã«Â©Â´ min/maxÃ«Â¥Â¼ kÃ«Â¡Å“ Ã«Â§Å¾Ã¬Â¶â€Ã«Å â€ ÃªÂ²Å’ Ã¬Å¾ÂÃ¬â€”Â°Ã¬Å Â¤Ã«Å¸Â½Ã«â€¹Â¤.
                    # ÃªÂ·Â¸Ã«Å¸Â°Ã«ÂÂ° Ã¬ÂÂ´Ã«Â¯Â¸ min_k/max_kÃ«Â¥Â¼ Ã«â€žÂ£Ã¬â€”Ë†Ã¬Å“Â¼Ã«Â©Â´ ÃªÂ·Â¸ ÃªÂ°â€™Ã¬Ââ€ž Ã¬Å¡Â°Ã¬â€žÂ Ã¬â€¹Å“.
                    pass
                write_log(f"[FACTORIAL] legacy arg k={kk} received (min_k={min_k}, max_k={max_k})", log_fp)
        except Exception:
            pass


    assert len(cand_names) == len(cand_maps), "cand_names and cand_maps length mismatch"
    n = len(cand_names)
    if n < min_k:
        write_log(f"[FACTORIAL] skipped: n_candidates={n} < min_k={min_k}", log_fp)
        return {"ok": False, "reason": "too_few_candidates", "n_candidates": n}

    # determine anchor index if requested
    anchor_idx: Optional[int] = None
    if anchor_name is not None:
        for i, nm in enumerate(cand_names):
            if str(nm) == str(anchor_name):
                anchor_idx = i
                break

    # enumerate combos
    combos: List[Tuple[int, ...]] = []
    # [BUG-1 FIX] Renamed loop var from `k` to `combo_size` to avoid
    # shadowing the function parameter `k: Optional[int]`.
    for combo_size in range(int(min_k), int(min(max_k, n)) + 1):
        for comb in combinations(range(n), combo_size):
            if anchor_must_include and anchor_idx is not None and anchor_idx not in comb:
                continue
            combos.append(comb)
            if len(combos) >= int(max_combos):
                break
        if len(combos) >= int(max_combos):
            break

    if not combos:
        write_log("[FACTORIAL] no combos to run (check anchor/min_k/max_k)", log_fp)
        return {"ok": False, "reason": "no_combos"}

    write_log(
        f"[FACTORIAL] running combos={len(combos)} (n_candidates={n}, min_k={min_k}, max_k={max_k}, "
        f"anchor={anchor_name}, anchor_must_include={anchor_must_include})",
        log_fp,
    )

    results: Dict[str, Any] = {}
    best = {"tag": None, "val_auc": -1.0, "test_auc": -1.0}

    for ci, comb in enumerate(combos):
        names = [cand_names[i] for i in comb]
        maps = [cand_maps[i] for i in comb]

        combo_tag = _safe_tag("__".join([str(x) for x in names]))
        sub_dir = out_dir / "factorial" / combo_tag

        rep = stack_simple(
            tr_refs=tr_refs,
            va_refs=va_refs,
            te_refs=te_refs,
            feature_set=feature_set,
            base_names=names,
            base_maps=maps,
            out_dir=sub_dir,
            log_fp=log_fp,
            seed=seed + ci,
            meta_method=meta_method,
            fit_on="train",
            y_tr_map=y_tr_map,
            y_va_map=y_va_map,
            y_te_map=y_te_map,
        )

        val_auc = float(rep.metrics.get("val", {}).get("auc", float("nan"))) if rep.ok else float("nan")
        te_auc = float(rep.metrics.get("test", {}).get("auc", float("nan"))) if rep.ok else float("nan")

        results[combo_tag] = {
            "ok": rep.ok,
            "base_names": names,
            "out_dir": rep.out_dir,
            "val_auc": val_auc,
            "test_auc": te_auc,
        }

        if rep.ok and np.isfinite(val_auc) and val_auc > best["val_auc"]:
            best = {"tag": combo_tag, "val_auc": val_auc, "test_auc": te_auc}

        write_log(
            f"[FACTORIAL] ({ci+1}/{len(combos)}) {combo_tag} ok={rep.ok} val_auc={val_auc:.4f} test_auc={te_auc:.4f}",
            log_fp,
        )

    summary = {
        "ok": True,
        "n_candidates": n,
        "candidate_names": cand_names,
        "min_k": int(min_k),
        "max_k": int(max_k),
        "anchor_name": anchor_name,
        "anchor_must_include": bool(anchor_must_include),
        "n_combos": len(combos),
        "best": best,
        "results": results,
    }
    save_json(out_dir / "factorial_summary.json", summary)

    if best["tag"] is not None:
        write_log(f"[FACTORIAL] BEST tag={best['tag']} val_auc={best['val_auc']:.4f} test_auc={best['test_auc']:.4f}", log_fp)

    return summary


# Backward-compat alias (typo users sometimes call)
def stack_factorical(*args, **kwargs):
    return stack_factorial(*args, **kwargs)

