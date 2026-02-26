from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.experiment_io import load_results_json as _load_results
from app.experiment_types import ExperimentResult, TREATMENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoL Teamfight Prediction — Systematic Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase descriptions:
  1  Baseline reproduction (5 seeds)
  2  Single-factor ablation (each treatment independently)
  3  Interaction analysis (pairwise + cumulative)
  4  Hyperparameter sensitivity analysis
  5  Final test set evaluation (ONE-TIME)

Examples:
  python experiment_runner.py --phase 1
  python experiment_runner.py --phase 2 --treatment 1
  python experiment_runner.py --phase 2 --treatment all
  python experiment_runner.py --phase 3 --top-k 3
  python experiment_runner.py --phase 4
  python experiment_runner.py --phase 5
        """,
    )

    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Experiment phase to execute")
    parser.add_argument(
        "--treatment",
        type=str,
        default="all",
        help=(
            "Treatment selection for Phase 2. "
            "'all', comma-separated ids (e.g., '1,3,5'), "
            "or study groups: interpolation_study | detector_validity | teamfight_interp_study."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K treatments for Phase 3 interaction analysis")
    parser.add_argument("--output-dir", type=str, default="./ablation_results", help="Directory to save experiment results")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment plan without executing")

    parser.add_argument("--feature-set", type=str, default="full", dest="feature_set")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-mode", type=str, default="patch_holdout", dest="split_mode")

    parser.add_argument(
        "--speed",
        action="store_true",
        default=True,
        help="Apply GPU speed overlay (RAM caching, batch scaling, torch.compile). Default: True",
    )
    parser.add_argument("--no-speed", action="store_false", dest="speed", help="Disable GPU speed overlay")
    parser.add_argument("--vram", type=float, default=24.0, help="GPU VRAM in GB for auto batch sizing (default: 24.0)")
    parser.add_argument(
        "--speed-profile",
        "--speed_profile",
        "--speed-mode",
        "--speed_mode",
        dest="speed_profile",
        type=str,
        default="auto",
        choices=["none", "auto", "rtx50", "rtx5080", "aggressive"],
        help="Runtime speed profile to combine with overlay (default: auto)",
    )
    return parser


def determine_best_treatments(
    output_dir: Path,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    """Select top treatments from Phase 2 result file."""
    phase2_data = _load_results(output_dir / "phase2_single_factor.json")
    if not phase2_data:
        print("[WARN] Phase 2 results not found, selecting all treatments")
        return list(TREATMENTS.keys()), {}

    best_ids: List[int] = []
    best_hps: Dict[int, Dict[str, Any]] = {}
    for tid_str, results_list in phase2_data.items():
        tid = int(tid_str)
        if not isinstance(results_list, list):
            continue
        val_aucs = [r.get("val_auc", -1) for r in results_list if isinstance(r, dict)]
        val_aucs = [a for a in val_aucs if a > 0]
        if val_aucs:
            best_ids.append(tid)
            if results_list and isinstance(results_list[0], dict):
                best_hps[tid] = results_list[0].get("hp_config", {})

    if not best_ids:
        print("[WARN] No treatment with positive val_auc in Phase 2. Falling back to all treatments.")
        return list(TREATMENTS.keys()), {}
    if len(best_ids) > 3:
        best_ids = best_ids[:3]
    return best_ids, best_hps


def deserialize_results(data: Any) -> List[ExperimentResult]:
    if not isinstance(data, list):
        return []

    results: List[ExperimentResult] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            result = ExperimentResult(
                treatment_id=int(item.get("treatment_id", -1)),
                treatment_name=str(item.get("treatment_name", "")),
                seed=int(item.get("seed", 0)),
                hp_config=item.get("hp_config", {}),
                train_auc=float(item.get("train_auc", -1.0)),
                val_auc=float(item.get("val_auc", -1.0)),
                test_auc=float(item.get("test_auc", -1.0)),
                train_f1=float(item.get("train_f1", -1.0)),
                val_f1=float(item.get("val_f1", -1.0)),
                test_f1=float(item.get("test_f1", -1.0)),
                val_brier=float(item.get("val_brier", -1.0)),
                val_ece=float(item.get("val_ece", -1.0)),
                best_epoch=int(item.get("best_epoch", -1)),
                train_time_sec=float(item.get("train_time_sec", -1.0)),
                n_train=int(item.get("n_train", -1)),
                n_val=int(item.get("n_val", -1)),
                n_test=int(item.get("n_test", -1)),
                n_fights_all=int(item.get("n_fights_all", -1)),
            )
            results.append(result)
        except (KeyError, ValueError, TypeError):
            continue
    return results


def deserialize_phase2_results(data: Any) -> Dict[int, List[ExperimentResult]]:
    if not isinstance(data, dict):
        return {}

    result_dict: Dict[int, List[ExperimentResult]] = {}
    for k, v in data.items():
        try:
            tid = int(k)
        except ValueError:
            continue
        if isinstance(v, list):
            result_dict[tid] = deserialize_results(v)
    return result_dict
