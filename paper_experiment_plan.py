#!/usr/bin/env python3
"""
paper_experiment_plan.py — 논문 제출 마감 역산 최적화 실험 설계 스크립트

CoG 2026 논문 제출 (3/17) 마감에 맞춘 실험 전략.
모델 1개 학습 ≈ 12시간 제약 하에서 최소한의 실험으로
논문에 필요한 전체 산출물을 생성하는 자동화 스크립트.

Usage:
    # 전체 계획 확인 (dry-run)
    python paper_experiment_plan.py --plan

    # Round 1: 1-seed 모델 스크리닝 (6개 모델, ~3일)
    python paper_experiment_plan.py --round 1

    # Round 2: 핵심 모델 5-seed 통계 검증 (~6일)
    python paper_experiment_plan.py --round 2

    # Round 3: Treatment ablation (~4일)
    python paper_experiment_plan.py --round 3

    # Round 4: 산출물 생성 (테이블, 그래프)
    python paper_experiment_plan.py --round 4

    # 전체 자동 실행 (Round 1 → 4 순차)
    python paper_experiment_plan.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# 0. Constants
# ──────────────────────────────────────────────────────────────

DEADLINE = datetime(2026, 3, 17)
HOURS_PER_MODEL = 12

SEEDS: Tuple[int, ...] = (7, 42, 123, 256, 512)
SCREENING_SEED: int = 7

# 논문 Table 7.1에 들어갈 핵심 모델 (카테고리별 대표)
SCREENING_MODELS: List[str] = [
    "lgbm",                 # Tabular baseline
    "rnn_bigru",            # Sequential (RNN)
    "rnn_transformer",      # Sequential (Transformer)
    "gnn_graphsage",        # Graph (GNN)
    "event_xattn",          # Event cross-attention
    # Layered Fusion (제안 모델 — 핵심 기여)
    "layered_fusion@global=bigru+gnn=graphsage+event=xattn+logit=1",
]

# 논문에 반드시 포함할 모델 (5-seed 통계 검증 대상)
# Round 1 결과로 조정 가능하나 기본값은 아래 4개
CORE_MODELS: List[str] = [
    "rnn_bigru",
    "gnn_graphsage",
    "event_xattn",
    "layered_fusion@global=bigru+gnn=graphsage+event=xattn+logit=1",
]

# Treatment ablation 우선순위 (논문 차별성 기여도 순)
PRIORITY_TREATMENTS: List[int] = [1, 2, 5, 7]  # Focal, Phase, Role-Adj, Label-Smooth
FULL_TREATMENTS: List[int] = [1, 2, 3, 4, 5, 6, 7]

# 공통 속도 최적화 플래그
SPEED_FLAGS: List[str] = [
    "--speed_profile", "rtx5080",
    "--amp",
    "--tf32",
    "--cache_match_packs_in_ram",
    "--cache_train_in_ram",
    "--cache_eval_in_ram",
    "--share_datasets",              # 데이터셋 1회 빌드 후 모델 간 공유
]

OUTPUT_DIR = Path("outputs/paper_experiments")


# ──────────────────────────────────────────────────────────────
# 1. Plan & Budget
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentRound:
    name: str
    description: str
    runs: int
    hours: float
    days: float
    deliverables: List[str]


def compute_plan() -> List[ExperimentRound]:
    """역산 기반 실험 계획 생성."""
    rounds = [
        ExperimentRound(
            name="Round 1: Model Screening (1-seed)",
            description=(
                f"{len(SCREENING_MODELS)}개 대표 모델을 seed={SCREENING_SEED}로 빠르게 학습.\n"
                "    모델별 Val/Test AUC 순위 파악 → Round 2 대상 선별."
            ),
            runs=len(SCREENING_MODELS),
            hours=len(SCREENING_MODELS) * HOURS_PER_MODEL,
            days=len(SCREENING_MODELS) * HOURS_PER_MODEL / 24,
            deliverables=["모델 성능 순위표", "탈락 모델 결정"],
        ),
        ExperimentRound(
            name="Round 2: Statistical Validation (5-seed)",
            description=(
                f"핵심 {len(CORE_MODELS)}개 모델 + lgbm baseline을 5-seed로 학습.\n"
                "    통계적 유의성 확보 (95% CI, p-value)."
            ),
            runs=(len(CORE_MODELS) + 1) * len(SEEDS),  # +1 for lgbm
            hours=(len(CORE_MODELS) + 1) * len(SEEDS) * HOURS_PER_MODEL,
            days=(len(CORE_MODELS) + 1) * len(SEEDS) * HOURS_PER_MODEL / 24,
            deliverables=[
                "Table 7.1: 모델별 Val/Test AUC ± std",
                "ROC curve 비교",
                "Box plot",
            ],
        ),
        ExperimentRound(
            name="Round 3: Treatment Ablation",
            description=(
                f"우선순위 Treatment {len(PRIORITY_TREATMENTS)}개를 ablation.\n"
                "    1-seed HP search → 상위 3개 3-seed 검증."
            ),
            runs=len(PRIORITY_TREATMENTS) + 3 * 3,  # HP search + 3-seed top3
            hours=(len(PRIORITY_TREATMENTS) + 9) * HOURS_PER_MODEL,
            days=(len(PRIORITY_TREATMENTS) + 9) * HOURS_PER_MODEL / 24,
            deliverables=[
                "Table 7.2: Treatment ablation (Δ AUC, 95% CI)",
                "Forest plot",
            ],
        ),
        ExperimentRound(
            name="Round 4: Paper Artifacts",
            description="결과 취합, 테이블/그래프 자동 생성, 논문 작성.",
            runs=0,
            hours=0,
            days=5,
            deliverables=[
                "LaTeX tables (자동 생성)",
                "Forest plot (matplotlib)",
                "Reliability diagram",
                "최종 논문",
            ],
        ),
    ]
    return rounds


def print_plan():
    """실험 계획 출력."""
    now = datetime.now()
    remaining_days = (DEADLINE - now).days
    remaining_hours = remaining_days * 24

    print("=" * 72)
    print("  CoG 2026 논문 실험 설계 — 최적화 계획")
    print("=" * 72)
    print(f"  오늘:       {now.strftime('%Y-%m-%d')}")
    print(f"  마감일:     {DEADLINE.strftime('%Y-%m-%d')}")
    print(f"  잔여일:     {remaining_days}일 ({remaining_hours}h)")
    print(f"  모델 1개:   ~{HOURS_PER_MODEL}시간")
    print(f"  최대 학습:  ~{remaining_hours // HOURS_PER_MODEL}회 (직렬 기준)")
    print("=" * 72)

    rounds = compute_plan()
    total_runs = 0
    total_hours = 0
    cumul_day = 0

    for i, r in enumerate(rounds, 1):
        total_runs += r.runs
        total_hours += r.hours
        start_day = cumul_day
        cumul_day += r.days

        start_date = now + timedelta(days=start_day)
        end_date = now + timedelta(days=cumul_day)

        print(f"\n{'─' * 72}")
        print(f"  {r.name}")
        print(f"{'─' * 72}")
        print(f"  {r.description}")
        print(f"  학습 횟수:  {r.runs}회")
        print(f"  소요 시간:  ~{r.hours:.0f}h ({r.days:.1f}일)")
        print(f"  예상 기간:  {start_date.strftime('%m/%d')} → {end_date.strftime('%m/%d')}")
        print(f"  산출물:")
        for d in r.deliverables:
            print(f"    • {d}")

    print(f"\n{'=' * 72}")
    print(f"  총 학습 횟수: {total_runs}회")
    print(f"  총 소요 시간: ~{total_hours:.0f}h ({total_hours/24:.1f}일) [직렬 기준]")

    if total_hours / 24 > remaining_days:
        deficit = total_hours / 24 - remaining_days
        print(f"\n  ⚠ 직렬 실행 시 {deficit:.1f}일 초과!")
        print(f"    → GPU 병렬화 또는 Round 3 축소 필요")
        print(f"    → '--fast' 옵션으로 max_matches 제한 권장")
    else:
        margin = remaining_days - total_hours / 24
        print(f"\n  ✓ 여유: {margin:.1f}일")

    print("=" * 72)

    # 모델 목록 출력
    print("\n[스크리닝 대상 모델]")
    for i, m in enumerate(SCREENING_MODELS, 1):
        category = _categorize_model(m)
        print(f"  {i}. {m:<60s}  [{category}]")

    print(f"\n[5-seed 검증 대상 모델]")
    for i, m in enumerate(CORE_MODELS, 1):
        print(f"  {i}. {m}")

    print(f"\n[Treatment Ablation 대상]")
    from app.experiment_types import TREATMENTS
    for tid in PRIORITY_TREATMENTS:
        t = TREATMENTS[tid]
        print(f"  T{tid}: {t.name} ({t.short_name})")


def _categorize_model(name: str) -> str:
    n = name.lower()
    if "lgbm" in n or "tab" in n:
        return "Tabular Baseline"
    if "fusion" in n or "layered" in n:
        return "Fusion (제안 모델)"
    if "xattn" in n or "event" in n:
        return "Event Attention"
    if "gnn" in n or "graph" in n or "gcn" in n or "gat" in n:
        return "Graph Neural Network"
    if "transformer" in n:
        return "Transformer"
    if "rnn" in n or "gru" in n or "lstm" in n:
        return "Sequential (RNN)"
    if "mamba" in n:
        return "State Space Model"
    if "tcn" in n:
        return "Temporal CNN"
    return "Other"


# ──────────────────────────────────────────────────────────────
# 2. Round Executors
# ──────────────────────────────────────────────────────────────

def _run_command(cmd: List[str], dry_run: bool = False) -> int:
    """Run a command, printing it first."""
    cmd_str = " ".join(cmd)
    print(f"\n  [CMD] {cmd_str}")
    if dry_run:
        print("  [DRY-RUN] Skipped.")
        return 0
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode


def run_round1(args: argparse.Namespace) -> None:
    """Round 1: 모델 스크리닝 (1-seed).

    6개 대표 모델을 seed=7로 빠르게 학습하여 성능 순위 파악.
    """
    print("\n" + "=" * 72)
    print("  ROUND 1: Model Screening (1-seed)")
    print("=" * 72)

    models_str = ",".join(SCREENING_MODELS)

    cmd = [
        sys.executable, "runner.py",
        "--mode", "all",
        "--seed", str(SCREENING_SEED),
        "--feature_set", "full",
        "--models", models_str,
        "--ablation_mode", "as_is",          # 공정 비교: lgbm logit은 fusion만 수신
        "--split_mode", "patch_holdout",
        "--no_factorial_fusion",
        "--stacking_mode", "simple",
        "--oof_skip_deep",
        *SPEED_FLAGS,
    ]

    if getattr(args, "max_matches", 0) > 0:
        cmd.extend(["--max_matches", str(args.max_matches)])

    _run_command(cmd, dry_run=args.dry_run)


def run_round2(args: argparse.Namespace) -> None:
    """Round 2: 핵심 모델 5-seed 통계 검증.

    lgbm baseline + 핵심 4개 모델을 5개 seed로 반복 학습.
    """
    print("\n" + "=" * 72)
    print("  ROUND 2: Statistical Validation (5-seed)")
    print("=" * 72)

    models_str = ",".join(CORE_MODELS)

    for seed in SEEDS:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed} / {SEEDS}")
        print(f"{'─' * 60}")

        cmd = [
            sys.executable, "runner.py",
            "--mode", "all",
            "--seed", str(seed),
            "--feature_set", "full",
            "--models", models_str,
            "--ablation_mode", "as_is",          # 공정 비교: lgbm logit은 fusion만 수신
            "--split_mode", "patch_holdout",
            "--no_factorial_fusion",
            "--stacking_mode", "simple",
            "--oof_skip_deep",
            *SPEED_FLAGS,
        ]

        if getattr(args, "max_matches", 0) > 0:
            cmd.extend(["--max_matches", str(args.max_matches)])

        rc = _run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"  [WARN] Seed {seed} failed with exit code {rc}")


def run_round3(args: argparse.Namespace) -> None:
    """Round 3: Treatment Ablation.

    Phase 1 (baseline 5-seed) → Phase 2 (treatment ablation).
    """
    print("\n" + "=" * 72)
    print("  ROUND 3: Treatment Ablation")
    print("=" * 72)

    output_dir = str(OUTPUT_DIR / "ablation")

    # Phase 1: Baseline reproduction
    print("\n[Phase 1] Baseline reproduction (5 seeds)")
    cmd_p1 = [
        sys.executable, "experiment_runner.py",
        "--phase", "1",
        "--output_dir", output_dir,
        "--feature_set", "full",
    ]
    if getattr(args, "speed", False) or True:
        cmd_p1.extend(["--speed", "--vram", "16"])
    if args.dry_run:
        cmd_p1.append("--dry_run")

    _run_command(cmd_p1, dry_run=False)  # dry_run handled internally

    # Phase 2: Priority treatments
    treatments_str = ",".join(str(t) for t in PRIORITY_TREATMENTS)
    print(f"\n[Phase 2] Treatment ablation (treatments={treatments_str})")
    cmd_p2 = [
        sys.executable, "experiment_runner.py",
        "--phase", "2",
        "--treatment", treatments_str,
        "--output_dir", output_dir,
        "--feature_set", "full",
    ]
    if getattr(args, "speed", False) or True:
        cmd_p2.extend(["--speed", "--vram", "16"])
    if args.dry_run:
        cmd_p2.append("--dry_run")

    _run_command(cmd_p2, dry_run=False)


def run_round4(args: argparse.Namespace) -> None:
    """Round 4: 논문 산출물 생성.

    기존 실험 결과를 취합하여 LaTeX 테이블, 그래프를 자동 생성.
    """
    print("\n" + "=" * 72)
    print("  ROUND 4: Paper Artifact Generation")
    print("=" * 72)

    artifacts_dir = OUTPUT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n  산출물 생성 경로:", artifacts_dir)
    print("\n  다음 함수들을 호출하여 산출물을 생성하세요:")
    print()
    print("  from app.analysis_reporting import generate_latex_table, generate_full_report")
    print("  from app.analysis_plotting import (")
    print("      generate_forest_plot_matplotlib,")
    print("      generate_reliability_diagram,")
    print("      generate_interaction_heatmap,")
    print("      generate_cumulative_curve,")
    print("  )")
    print()
    print("  필요한 입력:")
    print("    - Phase 1 결과: outputs/paper_experiments/ablation/phase1_baseline.json")
    print("    - Phase 2 결과: outputs/paper_experiments/ablation/phase2_single_factor.json")
    print("    - 각 모델 결과: outputs/runs_*/ablation_summary.csv")
    print()
    print("  [논문 Table 채우기 가이드]")
    print("  ─────────────────────────")
    print("  Table 7.1: ablation_summary.csv에서 모델별 va_auc, te_auc 취합")
    print("  Table 7.2: phase2_single_factor.json에서 Δ AUC, CI 추출")
    print("  Table 7.3: Fusion 축 조합 결과 (Round 2 layered_fusion 결과)")


# ──────────────────────────────────────────────────────────────
# 3. Model Registry Loader
# ──────────────────────────────────────────────────────────────

def load_trainable_models() -> Dict[str, Dict[str, Any]]:
    """현재 코드에서 등록된 전체 학습 가능 모델 목록을 로드.

    Returns:
        Dict[alias -> {category, class_name, aliases}]
    """
    from train.model_registry import MODEL_REGISTRY

    # Reverse map: factory function → list of aliases
    factory_to_aliases: Dict[int, List[str]] = {}
    factory_to_fn: Dict[int, Any] = {}
    for alias, fn in MODEL_REGISTRY.items():
        fid = id(fn)
        if fid not in factory_to_aliases:
            factory_to_aliases[fid] = []
            factory_to_fn[fid] = fn
        factory_to_aliases[fid].append(alias)

    # Build structured output
    models: Dict[str, Dict[str, Any]] = {}
    seen_fids: set = set()

    for fid, aliases in sorted(factory_to_aliases.items(), key=lambda x: x[1][0]):
        if fid in seen_fids:
            continue
        seen_fids.add(fid)

        primary = aliases[0]
        category = _categorize_model(primary)
        fn = factory_to_fn[fid]
        class_name = getattr(fn, "__name__", str(fn))

        models[primary] = {
            "category": category,
            "factory": class_name,
            "aliases": sorted(aliases),
            "n_aliases": len(aliases),
        }

    return models


def print_trainable_models() -> None:
    """전체 학습 가능 모델 목록 출력."""
    models = load_trainable_models()

    print("=" * 72)
    print("  학습 가능한 전체 모델 목록")
    print("=" * 72)

    # Group by category
    by_cat: Dict[str, List[Tuple[str, Dict]]] = {}
    for name, info in models.items():
        cat = info["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append((name, info))

    total = 0
    for cat in sorted(by_cat.keys()):
        print(f"\n  [{cat}]")
        for name, info in by_cat[cat]:
            aliases_str = ", ".join(info["aliases"])
            print(f"    {name:<35s}  aliases: {aliases_str}")
            total += 1

    print(f"\n  총 {total}개 모델 ({sum(m['n_aliases'] for m in models.values())}개 alias)")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CoG 2026 논문 실험 최적화 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--plan", action="store_true",
        help="실험 계획만 출력 (실행하지 않음)",
    )
    ap.add_argument(
        "--round", type=int, choices=[1, 2, 3, 4],
        help="실행할 Round 번호",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Round 1→4 전체 순차 실행",
    )
    ap.add_argument(
        "--list-models", action="store_true",
        help="학습 가능한 전체 모델 목록 출력",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="실제 학습 없이 커맨드만 출력",
    )
    ap.add_argument(
        "--max_matches", type=int, default=0,
        help="데이터 제한 (빠른 테스트용, 0=전체)",
    )
    ap.add_argument(
        "--speed", action="store_true", default=True,
        help="속도 최적화 활성화 (기본: True)",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_models:
        print_trainable_models()
        return

    if args.plan:
        print_plan()
        return

    if args.round == 1:
        run_round1(args)
    elif args.round == 2:
        run_round2(args)
    elif args.round == 3:
        run_round3(args)
    elif args.round == 4:
        run_round4(args)
    elif args.all:
        for r in [1, 2, 3, 4]:
            args.round = r
            print(f"\n{'#' * 72}")
            print(f"  AUTO-RUN: Round {r}")
            print(f"{'#' * 72}")
            if r == 1:
                run_round1(args)
            elif r == 2:
                run_round2(args)
            elif r == 3:
                run_round3(args)
            elif r == 4:
                run_round4(args)
    else:
        ap.print_help()
        print("\n  Tip: '--plan'으로 전체 계획을 확인하세요.")


if __name__ == "__main__":
    main()
