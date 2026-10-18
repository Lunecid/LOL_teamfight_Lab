#!/usr/bin/env python3
"""train_model_subprocess.py — 프로세스 격리 실험 실행기

각 실험을 별도 프로세스로 실행하여 메모리 누수를 원천 차단한다.

수학적 보장:
    ∀ i ≠ j: AddressSpace(M_i) ∩ AddressSpace(M_j) = ∅

프로세스 종료 시 OS가 해당 프로세스의 모든 메모리를 100% 회수하므로,
이전 모델의 텐서/캐시가 다음 모델에 영향을 줄 수 없다.

Usage (called by experiment_runner.py or runner.py with --isolate):
    python train_model_subprocess.py \\
        --mode experiment \\
        --feature_set full \\
        --seed 7 \\
        --split_mode patch_holdout \\
        --experiment_tag Baseline \\
        --treatment_overlay '{}' \\
        --result_file /tmp/result_xyz.json

    python train_model_subprocess.py \\
        --mode pipeline \\
        --feature_set full \\
        --seed 7 \\
        --models rnn_bigru,gnn_graphsage \\
        --runner_args '["--speed_profile", "rtx5080"]' \\
        --result_file /tmp/result_xyz.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _run_experiment_mode(args: argparse.Namespace) -> dict:
    """Run a single ablation experiment (wraps run_single_experiment)."""
    from app.experiment_runtime import run_single_experiment

    overlay = json.loads(args.treatment_overlay) if args.treatment_overlay else {}

    result = run_single_experiment(
        treatment_overlay=overlay,
        seed=args.seed,
        feature_set=args.feature_set,
        split_mode=args.split_mode,
        experiment_tag=args.experiment_tag,
    )

    # Serialize ExperimentResult to dict
    from dataclasses import asdict
    return asdict(result)


def _run_pipeline_mode(args: argparse.Namespace) -> dict:
    """Run the full pipeline (wraps runner.main)."""
    from runner import main as runner_main

    runner_argv = json.loads(args.runner_args) if args.runner_args else []
    runner_argv = [str(x) for x in runner_argv]

    runner_main(argv=runner_argv)
    return {"ok": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="Subprocess-isolated model training")
    ap.add_argument("--mode", choices=["experiment", "pipeline"], required=True)
    ap.add_argument("--feature_set", type=str, default="full")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split_mode", type=str, default="patch_holdout")
    ap.add_argument("--experiment_tag", type=str, default="")
    ap.add_argument("--treatment_overlay", type=str, default="{}")
    ap.add_argument("--runner_args", type=str, default="[]")
    ap.add_argument("--result_file", type=str, required=True,
                     help="Path to write JSON result")

    args = ap.parse_args()
    result_path = Path(args.result_file)

    try:
        if args.mode == "experiment":
            result = _run_experiment_mode(args)
        else:
            result = _run_pipeline_mode(args)

        result["_subprocess_ok"] = True
        result_path.write_text(json.dumps(result, default=str), encoding="utf-8")
        sys.exit(0)

    except Exception as e:
        error_result = {
            "_subprocess_ok": False,
            "_error": str(e),
            "_traceback": traceback.format_exc(),
        }
        result_path.write_text(json.dumps(error_result, default=str), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
