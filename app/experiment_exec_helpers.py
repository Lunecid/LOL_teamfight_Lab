from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.experiment_types import ExperimentResult


def build_experiment_args(
    feature_set: str = "full",
    seed: int = 7,
    split_mode: str = "patch_holdout",
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> argparse.Namespace:
    """Build a Namespace compatible with runner.py parser."""
    from runner import build_argparser

    parser = build_argparser()
    args = parser.parse_args([])

    args.feature_set = feature_set
    args.seed = seed
    args.split_mode = split_mode

    if extra_overrides:
        for k, v in extra_overrides.items():
            setattr(args, k, v)

    return args


def find_unsupported_overlay_flags(overlay: Dict[str, Any]) -> List[str]:
    unsupported: List[str] = []
    return unsupported


def pick_run_dir(
    run_root: Path,
    seed: int,
    before_run_names: Optional[set[str]] = None,
    started_at: Optional[float] = None,
) -> Optional[Path]:
    try:
        run_dirs = [d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    except Exception:
        return None

    if not run_dirs:
        return None

    def _mtime(p: Path) -> float:
        try:
            return float(p.stat().st_mtime)
        except Exception:
            return -1.0

    seed_token = f"__seed={int(seed)}"

    def _pick(cands: List[Path]) -> Optional[Path]:
        if not cands:
            return None
        cands = sorted(cands, key=_mtime, reverse=True)
        by_seed = [d for d in cands if seed_token in d.name]
        return by_seed[0] if by_seed else cands[0]

    if before_run_names:
        created = [d for d in run_dirs if d.name not in before_run_names]
        picked = _pick(created)
        if picked is not None:
            return picked

    if started_at is not None:
        recent = [d for d in run_dirs if _mtime(d) >= float(started_at) - 1.0]
        picked = _pick(recent)
        if picked is not None:
            return picked

    return _pick(run_dirs)


def parse_latest_run_result(
    experiment_tag: str,
    seed: int,
    hp_config: Dict[str, Any],
    run_dir_hint: Optional[Path] = None,
    preferred_models: Optional[List[str]] = None,
) -> ExperimentResult:
    """Parse metrics from the latest run output."""
    from core.config import RUN_DIR

    result = ExperimentResult(
        treatment_id=-1,
        treatment_name=experiment_tag,
        seed=seed,
        hp_config=hp_config,
    )

    try:
        latest_run = run_dir_hint
        if latest_run is None:
            latest_run = pick_run_dir(run_root=RUN_DIR, seed=seed, before_run_names=None, started_at=None)
        if latest_run is None:
            print(f"    [WARN] No run directories found in {RUN_DIR}")
            return result

        reports_path = latest_run / "deep_reports.json"
        if not reports_path.exists() and (latest_run / "models").exists():
            reports_path = latest_run / "models" / "deep_reports.json"

        if not reports_path.exists():
            for p in latest_run.rglob("deep_reports.json"):
                reports_path = p
                break

        if reports_path.exists():
            with open(reports_path, "r") as f:
                deep_reports = json.load(f)

            preferred = {
                str(m).strip()
                for m in (preferred_models or [])
                if str(m).strip() and str(m).strip().lower() != "lgbm"
            }
            best_report = None
            best_score = (-1, -1.0)

            for model_key, report in deep_reports.items():
                if not isinstance(report, dict) or not report.get("ok", False):
                    continue

                metrics = report.get("metrics", {})
                va_m = metrics.get("val", {})
                try:
                    va_auc = float(va_m.get("auc", float("nan")))
                except Exception:
                    va_auc = float("nan")
                va_auc_score = va_auc if np.isfinite(va_auc) else -1.0

                base_model = str(model_key).split("::", 1)[0]
                priority = 1 if (not preferred or base_model in preferred) else 0
                score = (priority, va_auc_score)
                if score > best_score:
                    best_score = score
                    best_report = report

            if isinstance(best_report, dict):
                metrics = best_report.get("metrics", {})
                tr_m = metrics.get("train", {})
                va_m = metrics.get("val", {})
                te_m = metrics.get("test", {})

                result.train_auc = float(tr_m.get("auc", -1.0))
                result.val_auc = float(va_m.get("auc", -1.0))
                result.test_auc = float(te_m.get("auc", -1.0))

                result.train_f1 = float(tr_m.get("f1", -1.0))
                result.val_f1 = float(va_m.get("f1", -1.0))
                result.test_f1 = float(te_m.get("f1", -1.0))

                result.train_ap = float(tr_m.get("ap", -1.0))
                result.val_ap = float(va_m.get("ap", -1.0))
                result.test_ap = float(te_m.get("ap", -1.0))

                result.best_epoch = int(best_report.get("best_epoch", -1))

        if result.val_auc < 0:
            csv_path = latest_run / "ablation_summary.csv"
            if csv_path.exists():
                import csv

                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    best_row = None
                    best_va = -1.0
                    for row in reader:
                        try:
                            va = float(row.get("va_auc", -1.0) or -1.0)
                        except Exception:
                            va = -1.0
                        if va > best_va:
                            best_va = va
                            best_row = row
                    if best_row is not None:
                        result.val_auc = float(best_row.get("va_auc", -1.0) or -1.0)
                        result.test_auc = float(best_row.get("te_auc", -1.0) or -1.0)
                        result.train_auc = float(best_row.get("tr_auc", -1.0) or -1.0)

        # Parse split/fight-count diagnostics when available.
        try:
            split_candidates = [
                latest_run / "meta" / "split.json",
                latest_run / "split.json",
            ]
            split_path = next((p for p in split_candidates if p.exists()), None)
            if split_path is not None:
                with open(split_path, "r", encoding="utf-8") as f:
                    split_obj = json.load(f)
                if isinstance(split_obj, dict):
                    result.n_train = int(split_obj.get("n_train", result.n_train))
                    result.n_val = int(split_obj.get("n_val", result.n_val))
                    result.n_test = int(split_obj.get("n_test", result.n_test))
        except Exception:
            pass

        try:
            index_candidates = [
                latest_run / "meta" / "fight_index.json",
                latest_run / "fight_index.json",
            ]
            index_path = next((p for p in index_candidates if p.exists()), None)
            if index_path is not None:
                with open(index_path, "r", encoding="utf-8") as f:
                    idx_obj = json.load(f)
                if isinstance(idx_obj, dict):
                    result.n_fights_all = int(idx_obj.get("n_fights", result.n_fights_all))
        except Exception:
            pass

    except Exception as e:
        print(f"    [WARN] Result parsing failed: {e}")

    return result
