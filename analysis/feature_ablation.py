"""feature_ablation.py — Feature Ablation & Validation Analysis

Four analyses for LightGBM baseline review:

  1. Single-feature ablation: remove bJNG_cs_cooldownReduction__max and
     retrain to measure performance delta.
  2. Static-attribute temporal aggregation validation: verify that
     rune IDs, champion IDs, etc. produce near-zero std/delta/slope
     and flag any anomalous high-importance static features.
  3. SHAP-based parsimonious model: retrain with top-k features to
     show performance vs. feature count trade-off.
  4. Logit pipeline integrity check: verify LightGBM logit_map
     correctly feeds downstream fusion models.

Usage:
    python -m analysis.feature_ablation --out_dir outputs/ablation [--max_samples N]

Requires a completed LightGBM baseline run with cached data.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Constants: static features that should NOT vary across timesteps
# =====================================================================
STATIC_FEATURE_PREFIXES: Tuple[str, ...] = (
    "champion_id",
    "champion_name_id",
    "primary_style_id",
    "sub_style_id",
    "primary_rune_1", "primary_rune_2", "primary_rune_3", "primary_rune_4",
    "sub_rune_1", "sub_rune_2",
    "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
    "summoner_spell_1_id", "summoner_spell_2_id",
    "blue_ban_", "red_ban_",
)

# Temporal suffixes that should be ~0 for truly static features
TEMPORAL_NOISE_SUFFIXES: Tuple[str, ...] = ("__std", "__delta", "__slope")
# Suffixes that should equal the constant value
TEMPORAL_CONST_SUFFIXES: Tuple[str, ...] = ("__last", "__mean", "__min", "__max")


def _is_static_feature(name: str) -> bool:
    """Check if a tabular feature derives from a static attribute."""
    for pfx in STATIC_FEATURE_PREFIXES:
        # Feature names have format: {slot}_{base_name}__{suffix}
        # e.g., bJNG_primary_rune_3__delta
        # We need to check if base_name starts with any static prefix
        parts = name.split("__")
        if len(parts) < 2:
            continue
        base = parts[0]  # e.g., "bJNG_primary_rune_3"
        # Strip the slot prefix (bTOP_, bJNG_, bMID_, bBOT_, bSUP_, rTOP_, ...)
        for slot in ("bTOP_", "bJNG_", "bMID_", "bBOT_", "bSUP_",
                      "rTOP_", "rJNG_", "rMID_", "rBOT_", "rSUP_"):
            if base.startswith(slot):
                attr = base[len(slot):]
                if attr.startswith(pfx) or attr == pfx.rstrip("_"):
                    return True
    return False


def _get_temporal_suffix(name: str) -> Optional[str]:
    """Extract the temporal aggregation suffix (__last, __mean, etc.)."""
    for sfx in ("__last", "__mean", "__std", "__min", "__max", "__delta", "__slope"):
        if name.endswith(sfx):
            return sfx
    return None


# =====================================================================
# 1. Single-Feature Ablation
# =====================================================================
@dataclass
class AblationResult:
    """Result of removing a single feature and retraining."""
    feature_removed: str
    baseline_auc_train: float
    baseline_auc_val: float
    baseline_auc_test: float
    ablated_auc_train: float
    ablated_auc_val: float
    ablated_auc_test: float
    delta_auc_val: float  # ablated - baseline (negative = feature was helpful)
    delta_auc_test: float
    n_features_baseline: int
    n_features_ablated: int


def run_single_feature_ablation(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
    target_feature: str = "bJNG_cs_cooldownReduction__max",
) -> Optional[AblationResult]:
    """Remove `target_feature` and retrain LightGBM, comparing AUC delta.

    Steps:
      1. Train full baseline model (all features).
      2. Drop `target_feature` column from X matrices.
      3. Retrain with identical hyperparameters.
      4. Report AUC delta on val/test.
    """
    from core.config import cfg
    from core.utils import metrics_from_probs, sanitize_feature_names, write_log
    from data.file_io import ensure_dir, save_kv_csv
    from core.utils import save_json
    from train.baseline import (
        build_tabular_Xy, corr_prune_tabular, infer_tabular_plan,
        compute_recency_weights, _as_frame,
    )

    try:
        import lightgbm as lgb
    except ImportError:
        write_log("[ABLATION] lightgbm not installed -> skip", log_fp)
        return None

    ensure_dir(out_dir)

    # --- Infer tabular plan ---
    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        write_log("[ABLATION] failed to infer tabular plan", log_fp)
        return None

    # --- Build tabular data ---
    write_log(f"[ABLATION] Building tabular features ...", log_fp)
    Xtr, ytr, feat_names, tr_used = build_tabular_Xy(
        tr_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xva, yva, _, va_used = build_tabular_Xy(
        va_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xte, yte, _, te_used = build_tabular_Xy(
        te_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )

    if len(tr_used) < 200:
        write_log(f"[ABLATION] Not enough samples: {len(tr_used)}", log_fp)
        return None

    # --- Correlation pruning (same as baseline) ---
    dropped: List[str] = []
    if bool(getattr(cfg, "DROP_CORR_FEATURES", False)) and Xtr.shape[1] > 1:
        keep_idx, dropped = corr_prune_tabular(
            Xtr, feat_names, seed=seed,
            threshold=float(getattr(cfg, "CORR_THRESHOLD", 0.98)),
        )
        Xtr = Xtr[:, keep_idx]
        Xva = Xva[:, keep_idx] if Xva.size else Xva
        Xte = Xte[:, keep_idx] if Xte.size else Xte
        feat_names = sanitize_feature_names([feat_names[i] for i in keep_idx])

    write_log(
        f"[ABLATION] After corr-prune: {len(feat_names)} features, "
        f"target='{target_feature}' present={target_feature in feat_names}",
        log_fp,
    )

    # --- LGB params ---
    try:
        import core.config as _cfg_mod
        params = dict(getattr(_cfg_mod, "BASELINE_LGB_PARAMS", {}))
    except Exception:
        params = {}
    params["random_state"] = int(seed)

    # --- Recency weights ---
    sample_weight = None
    if bool(getattr(cfg, "RECENCY_WEIGHT_ENABLED", False)):
        tau = float(getattr(cfg, "RECENCY_WEIGHT_TAU", 2.0))
        sample_weight = compute_recency_weights(tr_used, tau=tau, log_fp=log_fp)

    # ─── A. Train FULL baseline ───
    write_log("[ABLATION] Training FULL baseline ...", log_fp)
    clf_full = lgb.LGBMClassifier(**params)
    Xtr_df = _as_frame(Xtr, feat_names)
    Xva_df = _as_frame(Xva, feat_names) if Xva.size else Xva
    Xte_df = _as_frame(Xte, feat_names) if Xte.size else Xte

    if Xva.size:
        clf_full.fit(
            Xtr_df, ytr, sample_weight=sample_weight,
            eval_set=[(Xva_df, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        clf_full.fit(Xtr_df, ytr, sample_weight=sample_weight)

    p_tr_full = clf_full.predict_proba(Xtr_df)[:, 1]
    p_va_full = clf_full.predict_proba(Xva_df)[:, 1] if Xva.size else np.array([])
    p_te_full = clf_full.predict_proba(Xte_df)[:, 1] if Xte.size else np.array([])

    met_tr_full = metrics_from_probs(ytr, p_tr_full, threshold=0.5)
    met_va_full = metrics_from_probs(yva, p_va_full, threshold=0.5) if Xva.size else {}
    met_te_full = metrics_from_probs(yte, p_te_full, threshold=0.5) if Xte.size else {}

    write_log(
        f"[ABLATION] FULL  -> train_auc={met_tr_full.get('auc', 0):.4f} "
        f"val_auc={met_va_full.get('auc', 0):.4f} "
        f"test_auc={met_te_full.get('auc', 0):.4f}",
        log_fp,
    )

    # ─── B. Train ABLATED (drop target feature) ───
    if target_feature not in feat_names:
        write_log(
            f"[ABLATION] WARNING: target feature '{target_feature}' not found "
            f"in feature list (may have been corr-pruned). Skipping ablation.",
            log_fp,
        )
        return None

    drop_idx = feat_names.index(target_feature)
    keep_mask = np.ones(len(feat_names), dtype=bool)
    keep_mask[drop_idx] = False
    abl_names = [feat_names[i] for i in range(len(feat_names)) if keep_mask[i]]

    Xtr_abl = Xtr[:, keep_mask]
    Xva_abl = Xva[:, keep_mask] if Xva.size else Xva
    Xte_abl = Xte[:, keep_mask] if Xte.size else Xte

    write_log(f"[ABLATION] Training ABLATED (removed '{target_feature}') ...", log_fp)
    clf_abl = lgb.LGBMClassifier(**params)
    Xtr_abl_df = _as_frame(Xtr_abl, abl_names)
    Xva_abl_df = _as_frame(Xva_abl, abl_names) if Xva.size else Xva_abl
    Xte_abl_df = _as_frame(Xte_abl, abl_names) if Xte.size else Xte_abl

    if Xva.size:
        clf_abl.fit(
            Xtr_abl_df, ytr, sample_weight=sample_weight,
            eval_set=[(Xva_abl_df, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        clf_abl.fit(Xtr_abl_df, ytr, sample_weight=sample_weight)

    p_tr_abl = clf_abl.predict_proba(Xtr_abl_df)[:, 1]
    p_va_abl = clf_abl.predict_proba(Xva_abl_df)[:, 1] if Xva.size else np.array([])
    p_te_abl = clf_abl.predict_proba(Xte_abl_df)[:, 1] if Xte.size else np.array([])

    met_tr_abl = metrics_from_probs(ytr, p_tr_abl, threshold=0.5)
    met_va_abl = metrics_from_probs(yva, p_va_abl, threshold=0.5) if Xva.size else {}
    met_te_abl = metrics_from_probs(yte, p_te_abl, threshold=0.5) if Xte.size else {}

    write_log(
        f"[ABLATION] ABLATED -> train_auc={met_tr_abl.get('auc', 0):.4f} "
        f"val_auc={met_va_abl.get('auc', 0):.4f} "
        f"test_auc={met_te_abl.get('auc', 0):.4f}",
        log_fp,
    )

    delta_val = met_va_abl.get("auc", 0) - met_va_full.get("auc", 0)
    delta_test = met_te_abl.get("auc", 0) - met_te_full.get("auc", 0)

    write_log(
        f"[ABLATION] DELTA  -> val_auc={delta_val:+.4f} test_auc={delta_test:+.4f} "
        f"({'feature helpful' if delta_val < -0.001 else 'feature dispensable' if abs(delta_val) < 0.001 else 'feature harmful'})",
        log_fp,
    )

    result = AblationResult(
        feature_removed=target_feature,
        baseline_auc_train=met_tr_full.get("auc", 0),
        baseline_auc_val=met_va_full.get("auc", 0),
        baseline_auc_test=met_te_full.get("auc", 0),
        ablated_auc_train=met_tr_abl.get("auc", 0),
        ablated_auc_val=met_va_abl.get("auc", 0),
        ablated_auc_test=met_te_abl.get("auc", 0),
        delta_auc_val=delta_val,
        delta_auc_test=delta_test,
        n_features_baseline=len(feat_names),
        n_features_ablated=len(abl_names),
    )

    # Save report
    save_json(out_dir / "single_feature_ablation.json", {
        "target_feature": target_feature,
        "baseline": {"train_auc": result.baseline_auc_train,
                     "val_auc": result.baseline_auc_val,
                     "test_auc": result.baseline_auc_test},
        "ablated": {"train_auc": result.ablated_auc_train,
                    "val_auc": result.ablated_auc_val,
                    "test_auc": result.ablated_auc_test},
        "delta": {"val_auc": delta_val, "test_auc": delta_test},
        "n_features": {"baseline": result.n_features_baseline,
                       "ablated": result.n_features_ablated},
        "interpretation": (
            "feature helpful (AUC drops when removed)"
            if delta_val < -0.001
            else "feature dispensable (no meaningful AUC change)"
            if abs(delta_val) < 0.001
            else "feature harmful (AUC improves when removed)"
        ),
    })

    return result


# =====================================================================
# 2. Static Attribute Temporal Aggregation Validation
# =====================================================================
@dataclass
class StaticFeatureAudit:
    """Audit results for static feature temporal aggregation."""
    total_static_features: int
    anomalous_features: List[Dict[str, Any]]
    clean_features: List[str]
    verdict: str  # "clean" | "anomalous" | "severe"


def validate_static_temporal_aggregation(
    feature_set: str,
    tr_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
    max_scan: int = 5000,
    anomaly_threshold: float = 1e-6,
) -> StaticFeatureAudit:
    """Validate that static attributes have near-zero temporal variation.

    For features derived from static attributes (champion_id, rune IDs, etc.),
    temporal aggregation suffixes should satisfy:
      - __std   ~= 0  (no variation across timesteps)
      - __delta ~= 0  (no change from first to last)
      - __slope ~= 0  (no linear trend)
      - __last == __mean == __min == __max  (constant value)

    If these conditions are violated, it indicates either:
      (a) data artifacts (champion swaps, remakes), or
      (b) inappropriate temporal aggregation on static attributes.

    Parameters
    ----------
    anomaly_threshold : float
        Values below this are considered zero (numerical noise).
    """
    from core.utils import write_log, save_json
    from data.file_io import ensure_dir
    from data.cache_io import load_match_cache
    from gameplay.pipeline import build_ms_sequence
    from gameplay.features import build_sequence_features, seq_to_tabular
    from train.baseline import (
        _choose_tab_seq_key_and_names, _ref_engage_ts, _ref_label_end_ts, _ref_first_kill_ts, _ref_last_kill_ts,
        _tabular_feature_names_from_base, infer_tabular_plan,
    )

    ensure_dir(out_dir)

    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        write_log("[STATIC-AUDIT] failed to infer tabular plan", log_fp)
        return StaticFeatureAudit(0, [], [], "error")

    # Identify static feature columns in the tabular feature names
    feat_names = tab_plan.feat_names
    static_col_info: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(feat_names):
        if _is_static_feature(name):
            suffix = _get_temporal_suffix(name)
            if suffix is not None:
                static_col_info[name] = {"index": i, "suffix": suffix, "values": []}

    if not static_col_info:
        write_log("[STATIC-AUDIT] no static features found in tabular plan", log_fp)
        return StaticFeatureAudit(0, [], [], "clean")

    write_log(
        f"[STATIC-AUDIT] found {len(static_col_info)} static feature columns to audit",
        log_fp,
    )

    # Scan samples and collect actual values for static feature columns
    scanned = 0
    for r in tr_refs:
        if scanned >= max_scan:
            break

        pack = load_match_cache(r.match_id)
        if not pack:
            continue

        raw = build_ms_sequence(
            pack,
            pack["meta"]["team_map"],
            -1,
            engage_ts=_ref_engage_ts(r),
            label_end_ts=_ref_label_end_ts(r),
            first_kill_ts=_ref_first_kill_ts(r),
            last_kill_ts=_ref_last_kill_ts(r),
        )
        if not raw:
            continue

        feats = build_sequence_features(
            raw, pack["meta"]["team_map"],
            pack["meta"].get("role_slots", None),
            feature_set,
        )

        seq_key, _ = _choose_tab_seq_key_and_names(feature_set, feats)
        if seq_key is None:
            continue
        seq = feats.get(seq_key, None)
        if seq is None:
            continue

        x_tab = seq_to_tabular(np.asarray(seq, dtype=np.float32))
        if x_tab.ndim != 1:
            x_tab = x_tab.reshape(-1)

        if x_tab.shape[0] != len(feat_names):
            continue

        for name, info in static_col_info.items():
            idx = info["index"]
            if idx < x_tab.shape[0]:
                info["values"].append(float(x_tab[idx]))

        scanned += 1

    write_log(f"[STATIC-AUDIT] scanned {scanned} samples", log_fp)

    # Analyze collected values
    anomalous = []
    clean = []

    for name, info in static_col_info.items():
        vals = np.array(info["values"], dtype=np.float64)
        if vals.size == 0:
            continue

        suffix = info["suffix"]
        should_be_zero = suffix in TEMPORAL_NOISE_SUFFIXES

        mean_abs = float(np.mean(np.abs(vals)))
        std_vals = float(np.std(vals))
        nonzero_frac = float(np.mean(np.abs(vals) > anomaly_threshold))

        if should_be_zero:
            is_anomalous = nonzero_frac > 0.01  # >1% non-zero is suspicious
            if is_anomalous:
                anomalous.append({
                    "feature": name,
                    "suffix": suffix,
                    "mean_abs": mean_abs,
                    "std": std_vals,
                    "nonzero_fraction": nonzero_frac,
                    "n_samples": int(vals.size),
                    "issue": (
                        f"Static attribute with {suffix} should be ~0, "
                        f"but {nonzero_frac*100:.1f}% of samples are non-zero "
                        f"(mean|v|={mean_abs:.6f}). "
                        f"Possible causes: champion swap, remake, or data artifact."
                    ),
                })
            else:
                clean.append(name)
        else:
            clean.append(name)

    # Determine verdict
    if len(anomalous) == 0:
        verdict = "clean"
    elif len(anomalous) <= 5:
        verdict = "anomalous"
    else:
        verdict = "severe"

    write_log(
        f"[STATIC-AUDIT] VERDICT={verdict}: "
        f"{len(anomalous)} anomalous / {len(static_col_info)} total static features",
        log_fp,
    )

    for a in anomalous[:20]:
        write_log(f"[STATIC-AUDIT]   {a['feature']}: {a['issue']}", log_fp)

    result = StaticFeatureAudit(
        total_static_features=len(static_col_info),
        anomalous_features=anomalous,
        clean_features=clean,
        verdict=verdict,
    )

    save_json(out_dir / "static_feature_audit.json", {
        "total_static_features": result.total_static_features,
        "anomalous_count": len(anomalous),
        "clean_count": len(clean),
        "verdict": verdict,
        "anomalous_features": anomalous[:50],
        "recommendation": (
            "All static features have expected zero temporal variation."
            if verdict == "clean"
            else "Some static features show non-zero temporal variation. "
                 "Investigate data preprocessing for champion swaps or remakes. "
                 "Consider excluding __std, __delta, __slope suffixes for "
                 "static attributes to prevent noise fitting."
        ),
    })

    return result


# =====================================================================
# 3. SHAP-Based Parsimonious Model
# =====================================================================
@dataclass
class ParsimoniousResult:
    """Result of top-k feature selection and retraining."""
    k: int
    selected_features: List[str]
    auc_train: float
    auc_val: float
    auc_test: float


def run_shap_parsimonious_models(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
    top_k_list: Tuple[int, ...] = (25, 50, 100, 200, 500, 1000),
    shap_n_samples: int = 2000,
) -> List[ParsimoniousResult]:
    """Train parsimonious models with SHAP-selected top-k features.

    Procedure:
      1. Train full LightGBM model.
      2. Compute SHAP values on validation set (subsample).
      3. Rank features by mean |SHAP|.
      4. For each k in top_k_list, retrain with only top-k features.
      5. Report AUC for each k.

    This demonstrates the performance-complexity trade-off for reviewers.
    """
    from core.config import cfg
    from core.utils import metrics_from_probs, sanitize_feature_names, write_log, save_json
    from data.file_io import ensure_dir, save_kv_csv
    from train.baseline import (
        build_tabular_Xy, corr_prune_tabular, infer_tabular_plan,
        compute_recency_weights, _as_frame,
    )

    try:
        import lightgbm as lgb
        import shap
    except ImportError as e:
        write_log(f"[PARSIMONIOUS] Missing dependency: {e}", log_fp)
        return []

    ensure_dir(out_dir)

    # --- Build data ---
    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        write_log("[PARSIMONIOUS] failed to infer tabular plan", log_fp)
        return []

    Xtr, ytr, feat_names, tr_used = build_tabular_Xy(
        tr_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xva, yva, _, va_used = build_tabular_Xy(
        va_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xte, yte, _, te_used = build_tabular_Xy(
        te_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )

    if len(tr_used) < 200:
        write_log(f"[PARSIMONIOUS] Not enough samples: {len(tr_used)}", log_fp)
        return []

    # Correlation pruning
    if bool(getattr(cfg, "DROP_CORR_FEATURES", False)) and Xtr.shape[1] > 1:
        keep_idx, _ = corr_prune_tabular(
            Xtr, feat_names, seed=seed,
            threshold=float(getattr(cfg, "CORR_THRESHOLD", 0.98)),
        )
        Xtr = Xtr[:, keep_idx]
        Xva = Xva[:, keep_idx] if Xva.size else Xva
        Xte = Xte[:, keep_idx] if Xte.size else Xte
        feat_names = sanitize_feature_names([feat_names[i] for i in keep_idx])

    # --- LGB params ---
    try:
        import core.config as _cfg_mod
        params = dict(getattr(_cfg_mod, "BASELINE_LGB_PARAMS", {}))
    except Exception:
        params = {}
    params["random_state"] = int(seed)

    sample_weight = None
    if bool(getattr(cfg, "RECENCY_WEIGHT_ENABLED", False)):
        tau = float(getattr(cfg, "RECENCY_WEIGHT_TAU", 2.0))
        sample_weight = compute_recency_weights(tr_used, tau=tau, log_fp=log_fp)

    # --- Step 1: Train full model ---
    write_log("[PARSIMONIOUS] Training full model for SHAP ranking ...", log_fp)
    clf_full = lgb.LGBMClassifier(**params)
    Xtr_df = _as_frame(Xtr, feat_names)
    Xva_df = _as_frame(Xva, feat_names) if Xva.size else Xva
    Xte_df = _as_frame(Xte, feat_names) if Xte.size else Xte

    if Xva.size:
        clf_full.fit(
            Xtr_df, ytr, sample_weight=sample_weight,
            eval_set=[(Xva_df, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        clf_full.fit(Xtr_df, ytr, sample_weight=sample_weight)

    # Full model metrics
    p_va_full = clf_full.predict_proba(Xva_df)[:, 1] if Xva.size else np.array([])
    p_te_full = clf_full.predict_proba(Xte_df)[:, 1] if Xte.size else np.array([])
    met_va_full = metrics_from_probs(yva, p_va_full, threshold=0.5) if Xva.size else {}
    met_te_full = metrics_from_probs(yte, p_te_full, threshold=0.5) if Xte.size else {}

    # --- Step 2: Compute SHAP values ---
    write_log(f"[PARSIMONIOUS] Computing SHAP values (n={shap_n_samples}) ...", log_fp)
    try:
        explainer = shap.TreeExplainer(clf_full.booster_)
        n_shap = min(shap_n_samples, Xva.shape[0])
        Xs_shap = Xva_df.iloc[:n_shap] if hasattr(Xva_df, "iloc") else Xva[:n_shap]
        sv = explainer.shap_values(Xs_shap)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]  # class 1 SHAP values
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
    except Exception as e:
        write_log(f"[PARSIMONIOUS] SHAP computation failed: {e}", log_fp)
        # Fallback to gain-based importance
        write_log("[PARSIMONIOUS] Falling back to gain-based importance", log_fp)
        try:
            imp = clf_full.booster_.feature_importance(importance_type="gain")
            mean_abs_shap = imp.astype(np.float64)
        except Exception as e2:
            write_log(f"[PARSIMONIOUS] Gain importance also failed: {e2}", log_fp)
            return []

    # Rank features by SHAP importance
    ranking = sorted(
        zip(feat_names, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    # Save full SHAP ranking
    save_kv_csv(
        out_dir / "shap_feature_ranking.csv",
        ranking[:2000],
        k="feature",
        v="mean_abs_shap",
    )
    write_log(f"[PARSIMONIOUS] Top-10 SHAP features:", log_fp)
    for name, val in ranking[:10]:
        is_static = _is_static_feature(name)
        write_log(f"  {name}: {val:.6f}{' [STATIC]' if is_static else ''}", log_fp)

    # --- Step 3: Retrain with top-k features ---
    results: List[ParsimoniousResult] = []

    # Add full model as reference
    results.append(ParsimoniousResult(
        k=len(feat_names),
        selected_features=feat_names,
        auc_train=float(metrics_from_probs(ytr, clf_full.predict_proba(Xtr_df)[:, 1], threshold=0.5).get("auc", 0)),
        auc_val=met_va_full.get("auc", 0),
        auc_test=met_te_full.get("auc", 0),
    ))

    for k in sorted(top_k_list):
        if k >= len(feat_names):
            continue

        top_k_names = [name for name, _ in ranking[:k]]
        top_k_indices = [feat_names.index(n) for n in top_k_names if n in feat_names]
        if len(top_k_indices) < max(10, k // 2):
            continue

        Xtr_k = Xtr[:, top_k_indices]
        Xva_k = Xva[:, top_k_indices] if Xva.size else Xva
        Xte_k = Xte[:, top_k_indices] if Xte.size else Xte
        names_k = [feat_names[i] for i in top_k_indices]

        write_log(f"[PARSIMONIOUS] Training top-{k} model ...", log_fp)
        clf_k = lgb.LGBMClassifier(**params)
        Xtr_k_df = _as_frame(Xtr_k, names_k)
        Xva_k_df = _as_frame(Xva_k, names_k) if Xva.size else Xva_k
        Xte_k_df = _as_frame(Xte_k, names_k) if Xte.size else Xte_k

        try:
            if Xva.size:
                clf_k.fit(
                    Xtr_k_df, ytr, sample_weight=sample_weight,
                    eval_set=[(Xva_k_df, yva)], eval_metric="auc",
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
                )
            else:
                clf_k.fit(Xtr_k_df, ytr, sample_weight=sample_weight)
        except Exception as e:
            write_log(f"[PARSIMONIOUS] top-{k} training failed: {e}", log_fp)
            continue

        p_tr_k = clf_k.predict_proba(Xtr_k_df)[:, 1]
        p_va_k = clf_k.predict_proba(Xva_k_df)[:, 1] if Xva.size else np.array([])
        p_te_k = clf_k.predict_proba(Xte_k_df)[:, 1] if Xte.size else np.array([])

        met_tr_k = metrics_from_probs(ytr, p_tr_k, threshold=0.5)
        met_va_k = metrics_from_probs(yva, p_va_k, threshold=0.5) if Xva.size else {}
        met_te_k = metrics_from_probs(yte, p_te_k, threshold=0.5) if Xte.size else {}

        r = ParsimoniousResult(
            k=k,
            selected_features=names_k,
            auc_train=met_tr_k.get("auc", 0),
            auc_val=met_va_k.get("auc", 0),
            auc_test=met_te_k.get("auc", 0),
        )
        results.append(r)

        delta_val = met_va_k.get("auc", 0) - met_va_full.get("auc", 0)
        write_log(
            f"[PARSIMONIOUS] top-{k}: train_auc={r.auc_train:.4f} "
            f"val_auc={r.auc_val:.4f} (delta={delta_val:+.4f}) "
            f"test_auc={r.auc_test:.4f}",
            log_fp,
        )

    # Save summary
    summary = {
        "full_model": {
            "n_features": len(feat_names),
            "val_auc": met_va_full.get("auc", 0),
            "test_auc": met_te_full.get("auc", 0),
        },
        "parsimonious_models": [
            {
                "k": r.k,
                "auc_train": r.auc_train,
                "auc_val": r.auc_val,
                "auc_test": r.auc_test,
                "val_auc_retention": (
                    r.auc_val / max(1e-8, met_va_full.get("auc", 1))
                ),
            }
            for r in results
        ],
        "shap_top_20": ranking[:20],
        "static_features_in_top_20": [
            (name, val) for name, val in ranking[:20] if _is_static_feature(name)
        ],
    }
    save_json(out_dir / "parsimonious_summary.json", summary)

    return results


# =====================================================================
# 4. Logit Pipeline Integrity Check
# =====================================================================
@dataclass
class LogitPipelineCheck:
    """Logit pipeline integrity verification result."""
    logit_map_size: int
    n_train_refs: int
    n_val_refs: int
    n_test_refs: int
    train_coverage: float  # fraction of train refs covered
    val_coverage: float
    test_coverage: float
    logit_range: Tuple[float, float]
    logit_mean: float
    logit_std: float
    has_nan: bool
    has_inf: bool
    verdict: str  # "ok" | "warning" | "error"
    issues: List[str]


def verify_logit_pipeline(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
) -> LogitPipelineCheck:
    """Verify LightGBM logit outputs are correctly generated and suitable
    for downstream fusion models.

    Checks:
      1. Coverage: all refs have logit entries.
      2. Range: logits are finite and bounded (no overflow).
      3. Distribution: logits are roughly centered (no extreme bias).
      4. Downstream compatibility: logit values can be consumed by
         fusion meta-learner (logreg stacking).
    """
    from core.config import cfg
    from core.common import logit
    from core.fight_types import ref_key
    from core.utils import write_log, save_json
    from data.file_io import ensure_dir
    from train.baseline import run_lgbm_baseline

    ensure_dir(out_dir)
    issues: List[str] = []

    # Run baseline to get logit_map
    write_log("[LOGIT-CHECK] Running LightGBM baseline for logit generation ...", log_fp)
    result = run_lgbm_baseline(
        feature_set=feature_set,
        tr_refs=tr_refs,
        va_refs=va_refs,
        te_refs=te_refs,
        seed=seed,
        log_fp=log_fp,
        out_dir=out_dir / "lgbm_for_logit_check",
    )

    if not result.get("ok", False):
        write_log("[LOGIT-CHECK] LightGBM baseline failed -> cannot verify logits", log_fp)
        return LogitPipelineCheck(
            logit_map_size=0,
            n_train_refs=len(tr_refs), n_val_refs=len(va_refs), n_test_refs=len(te_refs),
            train_coverage=0, val_coverage=0, test_coverage=0,
            logit_range=(0, 0), logit_mean=0, logit_std=0,
            has_nan=False, has_inf=False,
            verdict="error", issues=["LightGBM baseline training failed"],
        )

    logit_map = result.get("logit_map", {})
    lm_size = len(logit_map)

    # --- Check 1: Coverage ---
    tr_keys = {ref_key(r) for r in tr_refs}
    va_keys = {ref_key(r) for r in va_refs}
    te_keys = {ref_key(r) for r in te_refs}

    tr_covered = sum(1 for k in tr_keys if k in logit_map)
    va_covered = sum(1 for k in va_keys if k in logit_map)
    te_covered = sum(1 for k in te_keys if k in logit_map)

    tr_cov = tr_covered / max(1, len(tr_keys))
    va_cov = va_covered / max(1, len(va_keys))
    te_cov = te_covered / max(1, len(te_keys))

    if tr_cov < 0.5:
        issues.append(f"Low train coverage: {tr_cov:.1%}")
    if va_cov < 0.8:
        issues.append(f"Low val coverage: {va_cov:.1%}")
    if te_cov < 0.8:
        issues.append(f"Low test coverage: {te_cov:.1%}")

    write_log(
        f"[LOGIT-CHECK] Coverage: train={tr_cov:.1%} ({tr_covered}/{len(tr_keys)}) "
        f"val={va_cov:.1%} ({va_covered}/{len(va_keys)}) "
        f"test={te_cov:.1%} ({te_covered}/{len(te_keys)})",
        log_fp,
    )

    # --- Check 2: Value range ---
    logit_vals = np.array(list(logit_map.values()), dtype=np.float64)

    has_nan = bool(np.isnan(logit_vals).any())
    has_inf = bool(np.isinf(logit_vals).any())

    if has_nan:
        issues.append(f"NaN values in logit_map ({np.isnan(logit_vals).sum()} entries)")
    if has_inf:
        issues.append(f"Inf values in logit_map ({np.isinf(logit_vals).sum()} entries)")

    finite_vals = logit_vals[np.isfinite(logit_vals)]
    if finite_vals.size > 0:
        logit_min = float(finite_vals.min())
        logit_max = float(finite_vals.max())
        logit_mean = float(finite_vals.mean())
        logit_std = float(finite_vals.std())
    else:
        logit_min = logit_max = logit_mean = logit_std = 0.0

    # Check for extreme values that could cause numerical issues
    if finite_vals.size > 0 and (logit_max > 15 or logit_min < -15):
        issues.append(
            f"Extreme logit values detected: [{logit_min:.2f}, {logit_max:.2f}]. "
            f"May cause sigmoid overflow in fusion."
        )

    write_log(
        f"[LOGIT-CHECK] Logit stats: n={lm_size} "
        f"range=[{logit_min:.3f}, {logit_max:.3f}] "
        f"mean={logit_mean:.3f} std={logit_std:.3f}",
        log_fp,
    )

    # --- Check 3: Downstream compatibility ---
    # Verify logit -> prob -> logit roundtrip
    if finite_vals.size > 0:
        sample_logits = finite_vals[:1000]
        probs = 1.0 / (1.0 + np.exp(-sample_logits))
        roundtrip_logits = np.array([logit(float(p)) for p in probs])
        roundtrip_error = np.abs(sample_logits - roundtrip_logits).max()
        if roundtrip_error > 0.01:
            issues.append(f"Logit roundtrip error too large: {roundtrip_error:.4f}")
        write_log(f"[LOGIT-CHECK] Roundtrip error: {roundtrip_error:.6f}", log_fp)

    # --- Check 4: Verify densify_logit_map works ---
    from train.baseline import densify_logit_map
    all_refs = list(tr_refs) + list(va_refs) + list(te_refs)
    dense_map = densify_logit_map(all_refs, logit_map, default_logit=0.0)
    all_keys = {ref_key(r) for r in all_refs}
    dense_coverage = sum(1 for k in all_keys if k in dense_map) / max(1, len(all_keys))
    if dense_coverage < 1.0:
        issues.append(f"densify_logit_map incomplete: {dense_coverage:.1%}")
    write_log(f"[LOGIT-CHECK] densify coverage: {dense_coverage:.1%}", log_fp)

    # --- Verdict ---
    if has_nan or has_inf or len(issues) > 3:
        verdict = "error"
    elif issues:
        verdict = "warning"
    else:
        verdict = "ok"

    write_log(f"[LOGIT-CHECK] VERDICT={verdict} issues={len(issues)}", log_fp)

    check = LogitPipelineCheck(
        logit_map_size=lm_size,
        n_train_refs=len(tr_refs),
        n_val_refs=len(va_refs),
        n_test_refs=len(te_refs),
        train_coverage=tr_cov,
        val_coverage=va_cov,
        test_coverage=te_cov,
        logit_range=(logit_min, logit_max),
        logit_mean=logit_mean,
        logit_std=logit_std,
        has_nan=has_nan,
        has_inf=has_inf,
        verdict=verdict,
        issues=issues,
    )

    save_json(out_dir / "logit_pipeline_check.json", {
        "logit_map_size": lm_size,
        "coverage": {"train": tr_cov, "val": va_cov, "test": te_cov},
        "logit_stats": {
            "range": [logit_min, logit_max],
            "mean": logit_mean,
            "std": logit_std,
        },
        "has_nan": has_nan,
        "has_inf": has_inf,
        "verdict": verdict,
        "issues": issues,
        "n_used": result.get("n_used", {}),
    })

    return check


# =====================================================================
# 5. Static-Feature-Excluded Ablation
# =====================================================================
def run_static_feature_exclusion_ablation(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Remove ALL temporal-noise suffixes (__std, __delta, __slope)
    of static features and retrain, measuring impact on AUC.

    This is the key experiment for reviewer concern:
    "Are rune-derived temporal features capturing signal or noise?"
    """
    from core.config import cfg
    from core.utils import metrics_from_probs, sanitize_feature_names, write_log, save_json
    from data.file_io import ensure_dir
    from train.baseline import (
        build_tabular_Xy, corr_prune_tabular, infer_tabular_plan,
        compute_recency_weights, _as_frame,
    )

    try:
        import lightgbm as lgb
    except ImportError:
        write_log("[STATIC-ABL] lightgbm not installed", log_fp)
        return None

    ensure_dir(out_dir)

    tab_plan = infer_tabular_plan(tr_refs, feature_set, log_fp=log_fp)
    if tab_plan is None:
        return None

    Xtr, ytr, feat_names, tr_used = build_tabular_Xy(
        tr_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xva, yva, _, va_used = build_tabular_Xy(
        va_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )
    Xte, yte, _, te_used = build_tabular_Xy(
        te_refs, feature_set, log_fp=log_fp, plan=tab_plan,
    )

    if len(tr_used) < 200:
        return None

    # Correlation pruning
    if bool(getattr(cfg, "DROP_CORR_FEATURES", False)) and Xtr.shape[1] > 1:
        keep_idx, _ = corr_prune_tabular(
            Xtr, feat_names, seed=seed,
            threshold=float(getattr(cfg, "CORR_THRESHOLD", 0.98)),
        )
        Xtr = Xtr[:, keep_idx]
        Xva = Xva[:, keep_idx] if Xva.size else Xva
        Xte = Xte[:, keep_idx] if Xte.size else Xte
        feat_names = sanitize_feature_names([feat_names[i] for i in keep_idx])

    # Identify static features with temporal-noise suffixes to drop
    drop_indices = []
    drop_names = []
    for i, name in enumerate(feat_names):
        if _is_static_feature(name):
            suffix = _get_temporal_suffix(name)
            if suffix in TEMPORAL_NOISE_SUFFIXES:
                drop_indices.append(i)
                drop_names.append(name)

    write_log(
        f"[STATIC-ABL] Identified {len(drop_indices)} static temporal-noise features to remove: "
        f"{drop_names[:20]}{'...' if len(drop_names) > 20 else ''}",
        log_fp,
    )

    if not drop_indices:
        write_log("[STATIC-ABL] No static temporal-noise features found", log_fp)
        return None

    # LGB params
    try:
        import core.config as _cfg_mod
        params = dict(getattr(_cfg_mod, "BASELINE_LGB_PARAMS", {}))
    except Exception:
        params = {}
    params["random_state"] = int(seed)

    sample_weight = None
    if bool(getattr(cfg, "RECENCY_WEIGHT_ENABLED", False)):
        tau = float(getattr(cfg, "RECENCY_WEIGHT_TAU", 2.0))
        sample_weight = compute_recency_weights(tr_used, tau=tau, log_fp=log_fp)

    # --- Train FULL model ---
    clf_full = lgb.LGBMClassifier(**params)
    Xtr_df = _as_frame(Xtr, feat_names)
    Xva_df = _as_frame(Xva, feat_names) if Xva.size else Xva
    Xte_df = _as_frame(Xte, feat_names) if Xte.size else Xte

    if Xva.size:
        clf_full.fit(
            Xtr_df, ytr, sample_weight=sample_weight,
            eval_set=[(Xva_df, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        clf_full.fit(Xtr_df, ytr, sample_weight=sample_weight)

    met_full = {
        "train": metrics_from_probs(ytr, clf_full.predict_proba(Xtr_df)[:, 1], threshold=0.5),
        "val": metrics_from_probs(yva, clf_full.predict_proba(Xva_df)[:, 1], threshold=0.5) if Xva.size else {},
        "test": metrics_from_probs(yte, clf_full.predict_proba(Xte_df)[:, 1], threshold=0.5) if Xte.size else {},
    }

    # --- Train CLEANED model (without static temporal-noise features) ---
    keep_mask = np.ones(len(feat_names), dtype=bool)
    for idx in drop_indices:
        keep_mask[idx] = False
    clean_names = [feat_names[i] for i in range(len(feat_names)) if keep_mask[i]]

    Xtr_clean = Xtr[:, keep_mask]
    Xva_clean = Xva[:, keep_mask] if Xva.size else Xva
    Xte_clean = Xte[:, keep_mask] if Xte.size else Xte

    clf_clean = lgb.LGBMClassifier(**params)
    Xtr_c_df = _as_frame(Xtr_clean, clean_names)
    Xva_c_df = _as_frame(Xva_clean, clean_names) if Xva.size else Xva_clean
    Xte_c_df = _as_frame(Xte_clean, clean_names) if Xte.size else Xte_clean

    if Xva.size:
        clf_clean.fit(
            Xtr_c_df, ytr, sample_weight=sample_weight,
            eval_set=[(Xva_c_df, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        clf_clean.fit(Xtr_c_df, ytr, sample_weight=sample_weight)

    met_clean = {
        "train": metrics_from_probs(ytr, clf_clean.predict_proba(Xtr_c_df)[:, 1], threshold=0.5),
        "val": metrics_from_probs(yva, clf_clean.predict_proba(Xva_c_df)[:, 1], threshold=0.5) if Xva.size else {},
        "test": metrics_from_probs(yte, clf_clean.predict_proba(Xte_c_df)[:, 1], threshold=0.5) if Xte.size else {},
    }

    delta_val = met_clean["val"].get("auc", 0) - met_full["val"].get("auc", 0)
    delta_test = met_clean["test"].get("auc", 0) - met_full["test"].get("auc", 0)

    write_log(
        f"[STATIC-ABL] FULL  -> val_auc={met_full['val'].get('auc', 0):.4f} "
        f"test_auc={met_full['test'].get('auc', 0):.4f}",
        log_fp,
    )
    write_log(
        f"[STATIC-ABL] CLEAN -> val_auc={met_clean['val'].get('auc', 0):.4f} "
        f"test_auc={met_clean['test'].get('auc', 0):.4f}",
        log_fp,
    )
    write_log(
        f"[STATIC-ABL] DELTA -> val={delta_val:+.4f} test={delta_test:+.4f}",
        log_fp,
    )

    result = {
        "n_dropped": len(drop_indices),
        "dropped_features": drop_names,
        "n_features_full": len(feat_names),
        "n_features_clean": len(clean_names),
        "full_model": {
            "train_auc": met_full["train"].get("auc", 0),
            "val_auc": met_full["val"].get("auc", 0),
            "test_auc": met_full["test"].get("auc", 0),
        },
        "clean_model": {
            "train_auc": met_clean["train"].get("auc", 0),
            "val_auc": met_clean["val"].get("auc", 0),
            "test_auc": met_clean["test"].get("auc", 0),
        },
        "delta": {"val_auc": delta_val, "test_auc": delta_test},
        "interpretation": (
            "Removing static temporal-noise features IMPROVES generalization "
            "(noise fitting confirmed). Recommend excluding __std/__delta/__slope "
            "for static attributes."
            if delta_val > 0.001
            else "Removing static temporal-noise features has NEGLIGIBLE effect. "
                 "These features are likely ignored by the model (near-zero importance)."
            if abs(delta_val) < 0.001
            else "Removing static temporal-noise features HURTS performance. "
                 "These may capture legitimate variation (champion swaps, data artifacts)."
        ),
    }

    save_json(out_dir / "static_feature_exclusion.json", result)
    return result


# =====================================================================
# Orchestrator: Run All Analyses
# =====================================================================
def run_all_analyses(
    feature_set: str,
    tr_refs: list,
    va_refs: list,
    te_refs: list,
    seed: int,
    log_fp: Path,
    out_dir: Path,
    target_feature: str = "bJNG_cs_cooldownReduction__max",
    top_k_list: Tuple[int, ...] = (25, 50, 100, 200, 500, 1000),
) -> Dict[str, Any]:
    """Run all four ablation analyses and produce a unified report."""
    from core.utils import write_log, save_json
    from data.file_io import ensure_dir

    ensure_dir(out_dir)

    write_log("=" * 70, log_fp)
    write_log("[ABLATION SUITE] Starting comprehensive feature ablation analysis", log_fp)
    write_log("=" * 70, log_fp)

    results: Dict[str, Any] = {}

    # --- Analysis 1: Single-feature ablation ---
    write_log("\n[ABLATION SUITE] === Analysis 1: Single-Feature Ablation ===", log_fp)
    t0 = time.time()
    try:
        abl_result = run_single_feature_ablation(
            feature_set, tr_refs, va_refs, te_refs, seed, log_fp,
            out_dir / "01_single_feature_ablation",
            target_feature=target_feature,
        )
        results["single_feature_ablation"] = {
            "ok": abl_result is not None,
            "target": target_feature,
            "delta_val_auc": abl_result.delta_auc_val if abl_result else None,
            "delta_test_auc": abl_result.delta_auc_test if abl_result else None,
            "time_sec": time.time() - t0,
        }
    except Exception as e:
        write_log(f"[ABLATION SUITE] Analysis 1 failed: {e}", log_fp)
        results["single_feature_ablation"] = {"ok": False, "error": str(e)}

    # --- Analysis 2: Static temporal validation ---
    write_log("\n[ABLATION SUITE] === Analysis 2: Static Temporal Validation ===", log_fp)
    t0 = time.time()
    try:
        audit = validate_static_temporal_aggregation(
            feature_set, tr_refs, seed, log_fp,
            out_dir / "02_static_temporal_validation",
        )
        results["static_temporal_validation"] = {
            "ok": True,
            "verdict": audit.verdict,
            "anomalous_count": len(audit.anomalous_features),
            "time_sec": time.time() - t0,
        }
    except Exception as e:
        write_log(f"[ABLATION SUITE] Analysis 2 failed: {e}", log_fp)
        results["static_temporal_validation"] = {"ok": False, "error": str(e)}

    # --- Analysis 2b: Static feature exclusion ablation ---
    write_log("\n[ABLATION SUITE] === Analysis 2b: Static Feature Exclusion ===", log_fp)
    t0 = time.time()
    try:
        static_abl = run_static_feature_exclusion_ablation(
            feature_set, tr_refs, va_refs, te_refs, seed, log_fp,
            out_dir / "02b_static_feature_exclusion",
        )
        results["static_feature_exclusion"] = {
            "ok": static_abl is not None,
            "delta_val_auc": static_abl.get("delta", {}).get("val_auc") if static_abl else None,
            "n_dropped": static_abl.get("n_dropped") if static_abl else 0,
            "time_sec": time.time() - t0,
        }
    except Exception as e:
        write_log(f"[ABLATION SUITE] Analysis 2b failed: {e}", log_fp)
        results["static_feature_exclusion"] = {"ok": False, "error": str(e)}

    # --- Analysis 3: SHAP-based parsimonious model ---
    write_log("\n[ABLATION SUITE] === Analysis 3: SHAP Parsimonious Model ===", log_fp)
    t0 = time.time()
    try:
        pars_results = run_shap_parsimonious_models(
            feature_set, tr_refs, va_refs, te_refs, seed, log_fp,
            out_dir / "03_parsimonious_model",
            top_k_list=top_k_list,
        )
        results["parsimonious_model"] = {
            "ok": len(pars_results) > 0,
            "n_models": len(pars_results),
            "models": [
                {"k": r.k, "val_auc": r.auc_val, "test_auc": r.auc_test}
                for r in pars_results
            ],
            "time_sec": time.time() - t0,
        }
    except Exception as e:
        write_log(f"[ABLATION SUITE] Analysis 3 failed: {e}", log_fp)
        results["parsimonious_model"] = {"ok": False, "error": str(e)}

    # --- Analysis 4: Logit pipeline verification ---
    write_log("\n[ABLATION SUITE] === Analysis 4: Logit Pipeline Check ===", log_fp)
    t0 = time.time()
    try:
        logit_check = verify_logit_pipeline(
            feature_set, tr_refs, va_refs, te_refs, seed, log_fp,
            out_dir / "04_logit_pipeline",
        )
        results["logit_pipeline"] = {
            "ok": logit_check.verdict != "error",
            "verdict": logit_check.verdict,
            "logit_map_size": logit_check.logit_map_size,
            "coverage": {
                "train": logit_check.train_coverage,
                "val": logit_check.val_coverage,
                "test": logit_check.test_coverage,
            },
            "issues": logit_check.issues,
            "time_sec": time.time() - t0,
        }
    except Exception as e:
        write_log(f"[ABLATION SUITE] Analysis 4 failed: {e}", log_fp)
        results["logit_pipeline"] = {"ok": False, "error": str(e)}

    # --- Unified Report ---
    write_log("\n" + "=" * 70, log_fp)
    write_log("[ABLATION SUITE] === UNIFIED REPORT ===", log_fp)
    write_log("=" * 70, log_fp)

    for name, info in results.items():
        status = "PASS" if info.get("ok", False) else "FAIL"
        write_log(f"  [{status}] {name}: {info}", log_fp)

    save_json(out_dir / "ablation_suite_report.json", results)
    write_log(f"\n[ABLATION SUITE] Reports saved to {out_dir}", log_fp)

    return results


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Feature Ablation Analysis Suite")
    parser.add_argument("--out_dir", type=str, default="outputs/ablation",
                        help="Output directory for analysis results")
    parser.add_argument("--feature_set", type=str, default="full",
                        help="Feature set to use (default: full)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--target_feature", type=str,
                        default="bJNG_cs_cooldownReduction__max",
                        help="Feature to ablate in single-feature analysis")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (for fast testing)")
    parser.add_argument("--top_k", type=str, default="25,50,100,200,500,1000",
                        help="Comma-separated list of top-k values for parsimonious models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = out_dir / "ablation.log"

    top_k_list = tuple(int(x) for x in args.top_k.split(","))

    # Load refs from experiment runner (assumes data is available)
    from core.utils import write_log
    write_log("[ABLATION CLI] Loading experiment references ...", log_fp)

    try:
        from experiment_runner import load_experiment_refs
        tr_refs, va_refs, te_refs = load_experiment_refs(
            feature_set=args.feature_set,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    except ImportError:
        write_log(
            "[ABLATION CLI] experiment_runner.load_experiment_refs not available. "
            "Please run from the project root with data configured.",
            log_fp,
        )
        return

    run_all_analyses(
        feature_set=args.feature_set,
        tr_refs=tr_refs,
        va_refs=va_refs,
        te_refs=te_refs,
        seed=args.seed,
        log_fp=log_fp,
        out_dir=out_dir,
        target_feature=args.target_feature,
        top_k_list=top_k_list,
    )


if __name__ == "__main__":
    main()
