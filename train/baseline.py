from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import cfg
from core.fight_types import FightRef, ref_key
from gameplay.pipeline import build_ms_sequence
from data.cache_io import load_match_cache
from gameplay.features import (
    build_sequence_features,
    get_extra_feature_names,
    get_xseq_feature_names,
    prune_correlated_columns,
    seq_to_tabular,
)
from core.utils import (
    confusion_from_probs,
    metrics_from_probs,
    pretty_cm,
    sanitize_feature_names,
    save_json,
    write_log,
)
from core.common import logit
from data.indexing import count_patches_from_refs, log_patch_block
from data.file_io import dump_predictions_csv, ensure_dir, save_kv_csv, save_text_lines

logger = logging.getLogger(__name__)


# =========================================================
# [REC-4a] Recency Weighting for Patch Covariate Shift
# ---------------------------------------------------------
# Mathematical formulation:
#   w_i = exp((p_i - p_min) / τ)
#
# where:
#   p_i   = integer-encoded patch number (e.g. 15.14 → 1514)
#   p_min = min(p_i) across all training samples
#   τ     = temperature parameter controlling recency strength
#
# Properties:
#   τ → ∞  : uniform weights (no recency preference)
#   τ = 2.0: moderate recency (1.0, 1.65, 2.72, 4.48 for 4 patches)
#   τ = 1.0: aggressive recency (1.0, 2.72, 7.39, 20.1)
#
# Rationale: Train AUC 0.896 → Val AUC 0.751 gap indicates
#   P_{patch_{15.14-15.15}}(x, y) ≠ P_{patch_{15.16}}(x, y)
# Recency weighting mitigates this covariate shift by emphasizing
# samples closer to the validation/deployment distribution.
# =========================================================
def _patch_to_numeric(patch_str: str) -> int:
    """Convert patch string like '15.14' to integer 1514 for ordering."""
    try:
        parts = str(patch_str).split(".")
        if len(parts) >= 2:
            major = int(parts[0])
            minor = int(parts[1])
            return major * 100 + minor
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def compute_recency_weights(
    refs: List[FightRef],
    tau: float = 2.0,
    log_fp: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute exponential recency weights for training samples.

    Parameters
    ----------
    refs : list of FightRef
        Training references with patch information.
    tau : float
        Temperature parameter (higher = more uniform).
    log_fp : optional Path
        Log file for diagnostics.

    Returns
    -------
    weights : (N,) array of sample weights
    """
    # Extract patch numbers from refs
    patch_nums = np.array([
        _patch_to_numeric(getattr(r, "patch", getattr(r, "patch_str", "0")))
        for r in refs
    ], dtype=np.float64)

    # Handle case where all patches are the same
    p_min = patch_nums.min()
    p_range = patch_nums.max() - p_min

    if p_range < 1e-8 or tau <= 0:
        if log_fp:
            write_log(f"[RECENCY] uniform weights (p_range={p_range:.1f}, τ={tau})", log_fp)
        return np.ones(len(refs), dtype=np.float64)

    # w_i = exp((p_i - p_min) / τ)
    weights = np.exp((patch_nums - p_min) / tau)

    # Normalize to mean=1 for stable gradient magnitudes
    weights = weights / weights.mean()

    if log_fp:
        unique_patches = sorted(set(patch_nums))
        weight_by_patch = {
            int(p): float(np.exp((p - p_min) / tau))
            for p in unique_patches
        }
        write_log(
            f"[RECENCY] τ={tau}, p_range=[{int(p_min)}..{int(p_min + p_range)}], "
            f"raw_weights_by_patch={weight_by_patch}",
            log_fp,
        )

    return weights

# Optional deps
try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except Exception:
    pd = None
    HAS_PANDAS = False

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

# NOTE: SHAP triggers matplotlib/font cache building and slows down even `--help`.
# Import it lazily only when it's actually requested.
shap = None
HAS_SHAP = False


@dataclass
class TabularPlan:
    seq_key: str
    base_names: List[str]
    feat_names: List[str]


def _as_frame(X: np.ndarray, feat_names: List[str]):
    if HAS_PANDAS and pd is not None:
        return pd.DataFrame(X, columns=list(feat_names))
    return X


def _tabular_feature_names_from_base(base_names: List[str]) -> List[str]:
    # [P4-STATS] Use centralized TABULAR_SUFFIXES (was hardcoded, Issue #5)
    from core.feature_contract import tabular_feature_names
    feat_names = list(tabular_feature_names(base_names))
    return sanitize_feature_names(feat_names)


def _choose_tab_seq_key_and_names(feature_set: str, feats: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Decide which sequence to flatten based on feature_set."""
    fs = str(feature_set or "x").lower()
    if "extra" in fs and isinstance(feats.get("extra_seq", None), np.ndarray):
        return "extra_seq", list(get_extra_feature_names(feature_set))
    if isinstance(feats.get("macro_seq", None), np.ndarray):
        return "macro_seq", list(get_extra_feature_names(feature_set))
    if isinstance(feats.get("x_seq", None), np.ndarray):
        return "x_seq", list(get_xseq_feature_names(feature_set))
    return None, []


def infer_tabular_plan(
    refs: List[FightRef],
    feature_set: str,
    log_fp: Optional[Path] = None,
    max_scan: int = 5000,
) -> Optional[TabularPlan]:
    scanned = 0
    for r in refs:
        scanned += 1
        if scanned > max_scan:
            break

        pack = load_match_cache(r.match_id)
        if not pack:
            continue
        # [P0-1 FIX] t_start is a required positional arg (before *).
        # When engage_ts is provided, the engage_ts path is taken and
        # t_start is unused — pass sentinel -1.
        raw = build_ms_sequence(
            pack,
            pack["meta"]["team_map"],
            -1,
            engage_ts=_ref_engage_ts(r),
            label_end_ts=_ref_label_end_ts(r),
        )
        if not raw:
            continue

        feats = build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None), feature_set)
        seq_key, base_names = _choose_tab_seq_key_and_names(feature_set, feats)
        if seq_key is None:
            continue
        feat_names = _tabular_feature_names_from_base(base_names)

        if log_fp:
            write_log(f"[TAB PLAN] decided seq_key={seq_key} base_dim={len(base_names)} tab_dim={len(feat_names)}", log_fp)
        return TabularPlan(seq_key=seq_key, base_names=base_names, feat_names=feat_names)

    if log_fp:
        write_log("[TAB PLAN] failed to infer (no usable sample).", log_fp)
    return None

def _ref_engage_ts(r):
    ts = getattr(r, "t_start_ts", -1)
    try:
        ts = int(ts)
    except Exception:
        ts = -1
    return ts if ts >= 0 else None


def _ref_label_end_ts(r):
    ts = getattr(r, "label_end_ts", -1)
    try:
        ts = int(ts)
    except Exception:
        ts = -1
    return ts if ts >= 0 else None

def build_tabular_Xy(
    refs: List[FightRef],
    feature_set: str,
    max_samples: Optional[int] = None,
    log_fp: Optional[Path] = None,
    plan: Optional[TabularPlan] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[FightRef]]:
    used: List[FightRef] = []
    Xs: List[np.ndarray] = []
    ys: List[int] = []

    seq_key: Optional[str] = plan.seq_key if plan else None
    feat_names: List[str] = list(plan.feat_names) if plan else []

    t0 = time.time()
    for i, r in enumerate(refs):
        if max_samples and len(used) >= int(max_samples):
            break

        pack = load_match_cache(r.match_id)
        if not pack:
            continue

        # [P0-1 FIX] t_start is required positional arg; sentinel -1 when engage_ts is provided.
        raw = build_ms_sequence(
            pack,
            pack["meta"]["team_map"],
            -1,
            engage_ts=_ref_engage_ts(r),
            label_end_ts=_ref_label_end_ts(r),
        )
        if not raw:
            continue

        feats = build_sequence_features(raw, pack["meta"]["team_map"], pack["meta"].get("role_slots", None), feature_set)

        if seq_key is None:
            seq_key, base_names = _choose_tab_seq_key_and_names(feature_set, feats)
            if seq_key is None:
                continue
            feat_names = _tabular_feature_names_from_base(base_names)

        seq = feats.get(seq_key, None)
        if seq is None:
            continue

        x_tab = seq_to_tabular(np.asarray(seq, dtype=np.float32))

        if bool(getattr(cfg, "USE_MOMENTUM_FEATURES", False)):
            mom = compute_momentum_stats(np.asarray(seq, dtype=np.float32))
            x_tab = np.concatenate([x_tab, mom], axis=0)
        if x_tab.ndim != 1:
            x_tab = x_tab.reshape(-1)

        Xs.append(x_tab)
        ys.append(int(feats["y"]))
        used.append(r)

        if log_fp and (i + 1) % 2000 == 0:
            write_log(f"[TAB] built={len(used)}/{i+1} key={seq_key}", log_fp)

    if not Xs:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64), [], []
    D = Xs[0].shape[0]
    X = np.empty((len(Xs), D), dtype=np.float32)
    for i, xi in enumerate(Xs):
        X[i] = xi
    del Xs
    y = np.asarray(ys, dtype=np.int64)

    if log_fp:
        write_log(f"[TAB DONE] key={seq_key} N={len(used)} D={X.shape[1]} time={time.time()-t0:.1f}s", log_fp)

    if len(feat_names) != X.shape[1]:
        if log_fp:
            write_log(f"[TAB WARN] feat_name_mismatch names={len(feat_names)} vs D={X.shape[1]} -> fallback f0..", log_fp)
        feat_names = [f"f{i}" for i in range(X.shape[1])]

    return X, y, feat_names, used


def corr_prune_tabular(
    Xtr: np.ndarray,
    names: List[str],
    seed: int,
    threshold: float,
    max_rows: int = 50000,
) -> Tuple[np.ndarray, List[str]]:
    keep_idx, dropped = prune_correlated_columns(Xtr, names, threshold=threshold, max_rows=max_rows, seed=seed)
    return np.asarray(keep_idx, dtype=int), list(dropped)


def _lgbm_evaluate_splits(
    clf,
    Xtr_in, ytr, Xva_in, yva, Xte_in, yte,
    Xva_size: bool, Xte_size: bool,
    log_fp: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """Predict and compute metrics for all splits."""
    p_tr = clf.predict_proba(Xtr_in)[:, 1]
    p_va = clf.predict_proba(Xva_in)[:, 1] if Xva_size else np.array([])
    p_te = clf.predict_proba(Xte_in)[:, 1] if Xte_size else np.array([])

    thr = float(cfg.CLS_THRESHOLD)
    met_tr = metrics_from_probs(ytr, p_tr, threshold=thr)
    met_va = metrics_from_probs(yva, p_va, threshold=thr) if Xva_size else {}
    met_te = metrics_from_probs(yte, p_te, threshold=thr) if Xte_size else {}

    write_log(f"[LGBM] Train: auc={met_tr.get('auc'):.4f} {pretty_cm(confusion_from_probs(ytr, p_tr, thr))}", log_fp)
    if Xva_size:
        write_log(f"[LGBM] Val  : auc={met_va.get('auc'):.4f} {pretty_cm(confusion_from_probs(yva, p_va, thr))}", log_fp)
    if Xte_size:
        write_log(f"[LGBM] Test : auc={met_te.get('auc'):.4f} {pretty_cm(confusion_from_probs(yte, p_te, thr))}", log_fp)

    return p_tr, p_va, p_te, met_tr, met_va, met_te


def _lgbm_feature_analysis(
    clf, feat_names: List[str],
    Xva_in, Xva_size: bool,
    log_fp: Path,
) -> Tuple[List, Optional[List]]:
    """Extract feature importance and optional SHAP values."""
    try:
        imp = clf.booster_.feature_importance(importance_type="gain")
        fi = sorted(zip(feat_names, imp.tolist()), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.debug("Feature importance extraction failed: %s", e)
        fi = []

    shap_summary = None
    if bool(getattr(cfg, "LGB_SHAP", False)) and Xva_size:
        try:
            import shap  # type: ignore
            expl = shap.TreeExplainer(clf.booster_)
            n = min(2000, Xva_in.shape[0] if hasattr(Xva_in, "shape") else len(Xva_in))
            Xs = Xva_in.iloc[:n] if HAS_PANDAS and hasattr(Xva_in, "iloc") else Xva_in[:n]
            sv = expl.shap_values(Xs)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            mean_abs = np.mean(np.abs(sv), axis=0)
            shap_summary = sorted(zip(feat_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            write_log(f"[LGBM] SHAP computed on n={n}", log_fp)
        except Exception as e:
            write_log(f"[LGBM] SHAP skipped/failed: {e}", log_fp)

    return fi, shap_summary


def _lgbm_build_logit_map(
    splits: List[Tuple[List[FightRef], np.ndarray]],
) -> Dict[str, float]:
    """Build ref_key -> logit map from (refs, probs) pairs."""
    logit_map: Dict[str, float] = {}
    for refs_used, probs in splits:
        if probs.size == 0:
            continue
        for r, p in zip(refs_used, probs):
            logit_map[ref_key(r)] = logit(float(p))
    return logit_map


def _lgbm_save_reports(
    out_dir: Path, clf, seed: int,
    feat_names: List[str], dropped: List[str],
    fi: List, shap_summary: Optional[List],
    met_tr: Dict, met_va: Dict, met_te: Dict,
    tr_used_pc: Dict, va_used_pc: Dict, te_used_pc: Dict,
    tr_used: List, va_used: List, te_used: List,
    tab_plan: TabularPlan, logit_map: Dict,
) -> Optional[Path]:
    """Save model file, feature reports, and compact JSON report."""
    model_path = out_dir / f"lgbm_baseline_seed{seed}.txt"
    try:
        clf.booster_.save_model(str(model_path))
    except Exception as e:
        logger.warning("Failed to save LightGBM model: %s", e)
        model_path = None

    kept = list(feat_names)
    dropped_corr = list(dropped)
    save_text_lines(out_dir / "features_used.txt", kept)
    save_text_lines(out_dir / "features_dropped.txt", dropped_corr)
    if fi:
        save_kv_csv(out_dir / "feature_importance_gain.csv", fi[:1000], k="feature", v="gain")
    if shap_summary:
        save_kv_csv(out_dir / "shap_mean_abs.csv", shap_summary[:1000], k="feature", v="mean_abs_shap")

    compact_report = {
        "ok": True,
        "model_path": str(model_path) if model_path else None,
        "metrics": {"train": met_tr, "val": met_va, "test": met_te},
        "feature_importance_gain_top": fi[:200],
        "shap_mean_abs_top": shap_summary[:200] if shap_summary else None,
        "n_features_used": len(kept),
        "n_features_dropped": len(dropped_corr),
        "corr_dropped": dropped_corr[:2000],
        "kept_features": kept[:2000],
        "patch_counts_used": {"train": tr_used_pc, "val": va_used_pc, "test": te_used_pc},
        "n_used": {"train": len(tr_used), "val": len(va_used), "test": len(te_used)},
        "tab_plan": {"seq_key": tab_plan.seq_key, "base_dim": len(tab_plan.base_names)},
        "logit_map_size": int(len(logit_map)),
        "note": "logit_map is kept in-memory for deep models; not written to disk by default.",
    }
    save_json(out_dir / "report.json", compact_report)
    return model_path


def run_lgbm_baseline(
    feature_set: str,
    tr_refs: List[FightRef],
    va_refs: List[FightRef],
    te_refs: List[FightRef],
    seed: int,
    log_fp: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Train + eval a LightGBM tabular baseline, returning a logit_map keyed by ref_key."""

    out: Dict[str, Any] = {"ok": False, "model_path": None, "logit_map": {}}
    ensure_dir(out_dir)

    if not HAS_LGB:
        write_log("[LGBM] lightgbm not installed -> skip baseline", log_fp)
        return out

    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        write_log("[LGBM] failed to infer tabular plan -> skip baseline", log_fp)
        return out

    # --- Build tabular features ---
    write_log(f"[LGBM] Building tabular features (feature_set={feature_set}, seq_key={tab_plan.seq_key}) ...", log_fp)

    Xtr, ytr, feat_names, tr_used = build_tabular_Xy(tr_refs, feature_set, log_fp=log_fp, plan=tab_plan)
    Xva, yva, _, va_used = build_tabular_Xy(va_refs, feature_set, log_fp=log_fp, plan=tab_plan)
    Xte, yte, _, te_used = build_tabular_Xy(te_refs, feature_set, log_fp=log_fp, plan=tab_plan)

    tr_used_pc = count_patches_from_refs(tr_used)
    va_used_pc = count_patches_from_refs(va_used)
    te_used_pc = count_patches_from_refs(te_used)
    log_patch_block("LGBM used(train)", tr_used_pc, log_fp)
    if Xva.size:
        log_patch_block("LGBM used(val)", va_used_pc, log_fp)
    if Xte.size:
        log_patch_block("LGBM used(test)", te_used_pc, log_fp)

    if len(tr_used) < 200:
        write_log(f"[LGBM] Not enough samples for training: N={len(tr_used)}", log_fp)
        out["patch_counts_used"] = {"train": tr_used_pc, "val": va_used_pc, "test": te_used_pc}
        return out

    # --- Correlation pruning ---
    dropped: List[str] = []
    if bool(getattr(cfg, "DROP_CORR_FEATURES", False)) and Xtr.shape[1] > 1:
        keep_idx, dropped = corr_prune_tabular(
            Xtr, feat_names, seed=seed,
            threshold=float(getattr(cfg, "CORR_THRESHOLD", 0.98)),
        )
        Xtr = Xtr[:, keep_idx]
        if Xva.size:
            Xva = Xva[:, keep_idx]
        if Xte.size:
            Xte = Xte[:, keep_idx]
        feat_names = sanitize_feature_names([feat_names[i] for i in keep_idx])
        write_log(f"[LGBM] corr-prune kept={len(keep_idx)} dropped={len(dropped)}", log_fp)

    # --- Train ---
    try:
        import core.config as _cfg_mod
        params = dict(getattr(_cfg_mod, "BASELINE_LGB_PARAMS", {}))
    except Exception as e:
        logger.warning("Failed to load BASELINE_LGB_PARAMS from config: %s", e)
        params = {}
    params["random_state"] = int(seed)

    clf = lgb.LGBMClassifier(**params)
    write_log("[LGBM] Training ...", log_fp)

    sample_weight = None
    if bool(getattr(cfg, "RECENCY_WEIGHT_ENABLED", False)):
        tau = float(getattr(cfg, "RECENCY_WEIGHT_TAU", 2.0))
        sample_weight = compute_recency_weights(tr_used, tau=tau, log_fp=log_fp)
        write_log(f"[LGBM] Recency weighting enabled (τ={tau}, n={len(sample_weight)})", log_fp)

    Xtr_in = _as_frame(Xtr, feat_names)
    Xva_in = _as_frame(Xva, feat_names) if Xva.size else Xva
    Xte_in = _as_frame(Xte, feat_names) if Xte.size else Xte

    try:
        if Xva.size:
            clf.fit(
                Xtr_in, ytr, sample_weight=sample_weight,
                eval_set=[(Xva_in, yva)], eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            )
        else:
            clf.fit(Xtr_in, ytr, sample_weight=sample_weight)
    except Exception as e:
        write_log(f"[LGBM] fit failed: {e}", log_fp)
        out["patch_counts_used"] = {"train": tr_used_pc, "val": va_used_pc, "test": te_used_pc}
        return out

    # --- Evaluate ---
    try:
        p_tr, p_va, p_te, met_tr, met_va, met_te = _lgbm_evaluate_splits(
            clf, Xtr_in, ytr, Xva_in, yva, Xte_in, yte,
            bool(Xva.size), bool(Xte.size), log_fp,
        )
    except Exception as e:
        write_log(f"[LGBM] predict failed: {e}", log_fp)
        out["patch_counts_used"] = {"train": tr_used_pc, "val": va_used_pc, "test": te_used_pc}
        return out

    # --- Prediction dumps ---
    try:
        dump_predictions_csv(out_dir / "pred_train.csv", tr_used, ytr.tolist(), p_tr.tolist(), split="train")
        if Xva.size:
            dump_predictions_csv(out_dir / "pred_val.csv", va_used, yva.tolist(), p_va.tolist(), split="val")
        if Xte.size:
            dump_predictions_csv(out_dir / "pred_test.csv", te_used, yte.tolist(), p_te.tolist(), split="test")
    except Exception as e:
        write_log(f"[LGBM] dump_predictions_csv failed: {e}", log_fp)

    # --- Feature analysis ---
    fi, shap_summary = _lgbm_feature_analysis(clf, feat_names, Xva_in, bool(Xva.size), log_fp)

    # --- Build logit map ---
    logit_map = _lgbm_build_logit_map([
        (tr_used, p_tr),
        (va_used, p_va),
        (te_used, p_te),
    ])

    # --- Save reports ---
    model_path = _lgbm_save_reports(
        out_dir, clf, seed, feat_names, dropped, fi, shap_summary,
        met_tr, met_va, met_te,
        tr_used_pc, va_used_pc, te_used_pc,
        tr_used, va_used, te_used,
        tab_plan, logit_map,
    )

    out.update({
        "ok": True,
        "model_path": str(model_path) if model_path else None,
        "metrics": {"train": met_tr, "val": met_va, "test": met_te},
        "feature_importance_gain": fi[:200],
        "shap_mean_abs": shap_summary[:200] if shap_summary else None,
        "corr_dropped": dropped[:500],
        "logit_map": logit_map,
        "kept_features": list(feat_names),
        "patch_counts_used": {"train": tr_used_pc, "val": va_used_pc, "test": te_used_pc},
        "n_used": {"train": len(tr_used), "val": len(va_used), "test": len(te_used)},
        "tab_plan": {"seq_key": tab_plan.seq_key, "base_dim": len(tab_plan.base_names)},
        "out_dir": str(out_dir),
    })
    return out


def densify_logit_map(refs: List[FightRef], logit_map: Dict[str, float], default_logit: float = 0.0) -> Dict[str, float]:
    out = dict(logit_map or {})
    for r in refs:
        k = ref_key(r)
        if k not in out:
            out[k] = float(default_logit)
    return out
