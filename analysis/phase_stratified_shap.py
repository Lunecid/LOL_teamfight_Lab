"""Phase-Stratified SHAP Analysis.

Runs SHAP on existing LightGBM model separately for Early/Mid/Late game phases,
producing a Feature Importance Transition Table.

Usage:
    python -m analysis.phase_stratified_shap \
        --run_dir outputs/runs_.../run_..._seed=42 \
        --output_dir outputs/phase_shap
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


def load_lgbm_model(run_dir: Path):
    """Load a saved LightGBM model from a run directory."""
    import lightgbm as lgb

    candidates = (
        list(run_dir.rglob("lgbm_baseline_seed*.txt"))
        + list(run_dir.rglob("lgbm*.txt"))
        + list(run_dir.rglob("lgbm*.bin"))
    )
    if not candidates:
        raise FileNotFoundError(f"No LightGBM model found in {run_dir}")
    model_path = candidates[0]
    logger.info("[LOAD] LightGBM model: %s", model_path)
    return lgb.Booster(model_file=str(model_path))


def classify_phase(engage_ts_ms: int, game_start_ms: int = 0) -> str:
    """Classify engage_ts into game phase."""
    minutes = (engage_ts_ms - game_start_ms) / 60000.0
    if minutes <= 14:
        return "early"
    elif minutes <= 25:
        return "mid"
    else:
        return "late"


def run_phase_stratified_shap(
    model,
    X_test: np.ndarray,
    feat_names: List[str],
    fight_minutes: np.ndarray,
    n_samples_per_phase: int = 2000,
    seed: int = 42,
    output_dir: Path = Path("outputs/phase_shap"),
) -> Dict[str, Any]:
    """Run per-phase SHAP and produce a Transition Table.

    Parameters
    ----------
    model : LightGBM Booster or Classifier
    X_test : (N, D) feature matrix
    feat_names : list of D feature names
    fight_minutes : (N,) game minutes at engage time
    n_samples_per_phase : max samples per phase for SHAP
    seed : random seed
    output_dir : where to save results

    Returns
    -------
    dict with phase_results and transition_table
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_masks = {
        "early": fight_minutes <= 14,
        "mid": (fight_minutes > 14) & (fight_minutes <= 25),
        "late": fight_minutes > 25,
    }

    explainer = shap.TreeExplainer(model)
    phase_results: Dict[str, Any] = {}

    for phase_name, mask in phase_masks.items():
        X_phase = X_test[mask]
        n = min(n_samples_per_phase, len(X_phase))
        if n < 50:
            logger.warning("Phase %s: only %d samples, skipping", phase_name, len(X_phase))
            continue

        idx = np.random.RandomState(seed).choice(len(X_phase), n, replace=False)
        X_sample = X_phase[idx]

        logger.info("[SHAP] %s: %d samples...", phase_name, n)
        sv = explainer.shap_values(X_sample)

        # Handle binary classification (list of 2 arrays)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        top_features = [(feat_names[i], float(mean_abs[i])) for i in top_idx]

        phase_results[phase_name] = {
            "n_samples": int(n),
            "mean_abs_shap_full": mean_abs.tolist(),
            "top_10": top_features,
        }

        np.save(output_dir / f"shap_values_{phase_name}.npy", sv)
        logger.info("[SHAP] %s top-3: %s", phase_name, top_features[:3])

    # Build Transition Table
    all_top: set = set()
    for pr in phase_results.values():
        for fn, _ in pr["top_10"]:
            all_top.add(fn)

    transition: List[Dict[str, Any]] = []
    for feat in sorted(all_top):
        if feat not in feat_names:
            continue
        fidx = feat_names.index(feat)
        row: Dict[str, Any] = {"feature": feat}
        for pn in ["early", "mid", "late"]:
            if pn not in phase_results:
                row[f"{pn}_shap"] = None
                row[f"{pn}_rank"] = None
                continue
            mas = np.array(phase_results[pn]["mean_abs_shap_full"])
            val = mas[fidx]
            rank = int((mas > val).sum()) + 1
            row[f"{pn}_shap"] = round(float(val), 4)
            row[f"{pn}_rank"] = rank
        transition.append(row)

    # Save results (exclude raw arrays for JSON readability)
    result = {
        "phase_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "mean_abs_shap_full"}
            for k, v in phase_results.items()
        },
        "transition_table": transition,
    }
    with open(output_dir / "phase_shap_results.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("[DONE] Results saved to %s", output_dir)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Phase-Stratified SHAP Analysis")
    ap.add_argument("--run_dir", type=str, required=True, help="Run root directory (contains meta/ and models/)")
    ap.add_argument("--output_dir", type=str, default="outputs/phase_shap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--feature_set", type=str, default="full")
    args = ap.parse_args()

    import csv
    from core.fight_types import FightRef
    from train.baseline import build_tabular_Xy, infer_tabular_plan

    run_dir = Path(args.run_dir)

    # 1) Load test FightRefs from CSV
    refs_csv = run_dir / "meta" / "fight_refs_test.csv"
    if not refs_csv.exists():
        # fallback: search recursively
        candidates = list(run_dir.rglob("fight_refs_test.csv"))
        if not candidates:
            raise FileNotFoundError(f"fight_refs_test.csv not found under {run_dir}")
        refs_csv = candidates[0]

    te_refs = []
    with open(refs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            te_refs.append(FightRef(
                match_id=str(row["match_id"]),
                patch=str(row["patch"]),
                t_start=int(row["t_start"]),
                t_start_ts=int(row.get("t_start_ts", -1)),
                label_end_ts=int(row.get("label_end_ts", -1)),
                first_kill_ts=int(row.get("first_kill_ts", -1)),
                last_kill_ts=int(row.get("last_kill_ts", -1)),
            ))
    logger.info("[LOAD] %d test refs from %s", len(te_refs), refs_csv)

    # 2) Rebuild X_test via the same tabular pipeline as baseline
    plan = infer_tabular_plan(te_refs, args.feature_set)
    if plan is None:
        raise RuntimeError("Failed to infer tabular plan from test refs")

    X_test, y_test, feat_names, used_refs = build_tabular_Xy(
        te_refs, args.feature_set, plan=plan,
    )
    logger.info("[DATA] X_test shape=%s, feat_names=%d", X_test.shape, len(feat_names))

    # 3) Compute fight_minutes from t_start_ts
    fight_minutes = np.array([
        r.t_start_ts / 60000.0 if r.t_start_ts >= 0 else r.t_start * 1.0
        for r in used_refs
    ], dtype=np.float64)
    logger.info("[DATA] fight_minutes: min=%.1f, max=%.1f, mean=%.1f",
                fight_minutes.min(), fight_minutes.max(), fight_minutes.mean())

    # 4) Load LightGBM model
    model = load_lgbm_model(run_dir)

    # 5) Run phase-stratified SHAP
    result = run_phase_stratified_shap(
        model=model,
        X_test=X_test,
        feat_names=feat_names,
        fight_minutes=fight_minutes,
        n_samples_per_phase=args.n_samples,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    # Print summary
    for phase, pr in result.get("phase_results", {}).items():
        print(f"\n=== {phase.upper()} (n={pr['n_samples']}) ===")
        for fname, val in pr["top_10"]:
            print(f"  {fname}: {val:.4f}")
