#!/usr/bin/env python3
"""run_paper_full.py — 논문 실험 통합 실행 스크립트

한 번 실행으로 3-seed × 2-phase (as_is + baseline_plus) 전체 실험을 수행한다.
데이터 로딩을 최소화하기 위해:
  - match cache는 첫 실행에서 빌드되고 디스크에 캐싱됨
  - fight index도 첫 실행에서 빌드 후 캐싱됨
  - 두 번째 phase부터는 캐시 히트로 즉시 시작

실행 순서 (per seed):
  Phase A (as_is): deep-only models (LightGBM logit 없이 독립 학습)
    → bigru, transformer, stgnn, graphsage, event_xattn, layered_fusion(logit=0)

  Phase B (baseline_plus): LightGBM + logit-fusion
    → lgbm, layered_fusion(logit=1)

Usage:
    python run_paper_full.py
    python run_paper_full.py --seeds 42           # 단일 seed 테스트
    python run_paper_full.py --dry_run             # 커맨드만 출력
    python run_paper_full.py --skip_phase_a        # Phase B만 실행
    python run_paper_full.py --skip_phase_b        # Phase A만 실행
    python run_paper_full.py --max_matches 1000    # 소규모 테스트
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SEEDS = [7, 42, 123]

# Phase A: deep-only models (ablation_mode = as_is)
PHASE_A_MODELS = (
    "bigru,"
    "transformer,"
    "stgnn,"
    "gnn_graphsage,"
    "event_xattn,"
    "layered_fusion@global=bigru+gnn=graphsage+event=xattn+logit=0"
)

# Phase B: tabular + logit fusion (ablation_mode = baseline_plus)
PHASE_B_MODELS = (
    "lgbm,"
    "mlp,"
    "layered_fusion@global=bigru+gnn=graphsage+event=xattn+logit=1"
)

# Shared flags (hardware / data)
SHARED_FLAGS = [
    "--mode", "train",
    "--no_factorial_fusion",
    "--max_matches", "210000",
    "--split_mode", "patch_holdout",
    "--train_patches", "15.14",
    "--val_patches", "15.15",
    "--test_patches", "15.16",
    "--share_datasets",
    "--speed_profile", "rtx5080",
    "--amp",
    "--tf32",
    "--cache_match_packs_in_ram",
    "--batch_size", "64",
]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def build_command(
    seed: int,
    models: str,
    ablation_mode: str,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """Build runner.py command."""
    cmd = [
        sys.executable, "runner.py",
        *SHARED_FLAGS,
        "--seed", str(seed),
        "--models", models,
        "--ablation_mode", ablation_mode,
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def run_command(cmd: List[str], dry_run: bool = False, label: str = "") -> int:
    """Execute command with logging."""
    cmd_str = " ".join(cmd)
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    print(f"  [CMD] {cmd_str}\n")

    if dry_run:
        print("  [DRY-RUN] Skipped.\n")
        return 0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"\n  [{status}] {label} -> {elapsed / 60:.1f} min\n")
    return result.returncode


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paper experiment: 3-seed x 2-phase unified runner",
    )
    ap.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in SEEDS),
        help=f"Comma-separated seeds (default: {','.join(str(s) for s in SEEDS)})",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    ap.add_argument("--skip_phase_a", action="store_true", help="Skip Phase A (as_is deep models)")
    ap.add_argument("--skip_phase_b", action="store_true", help="Skip Phase B (baseline_plus)")
    ap.add_argument(
        "--max_matches", type=int, default=210000,
        help="Override max_matches (default: 210000)",
    )
    ap.add_argument("--extra_flags", type=str, default="", help="Additional flags passed to runner.py")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    extra = args.extra_flags.split() if args.extra_flags.strip() else []

    # Override max_matches in SHARED_FLAGS
    if args.max_matches != 210000:
        idx = SHARED_FLAGS.index("--max_matches")
        SHARED_FLAGS[idx + 1] = str(args.max_matches)

    total_runs = 0
    total_ok = 0
    total_fail = 0
    t_start = time.time()

    print(f"\n{'=' * 72}")
    print(f"  PAPER EXPERIMENT: Full Pipeline")
    print(f"  Seeds: {seeds}")
    print(f"  Phase A (as_is):        {'SKIP' if args.skip_phase_a else 'RUN'}")
    print(f"  Phase B (baseline_plus): {'SKIP' if args.skip_phase_b else 'RUN'}")
    print(f"  Max matches: {args.max_matches}")
    print(f"{'=' * 72}\n")

    for si, seed in enumerate(seeds):
        print(f"\n{'=' * 72}")
        print(f"  SEED {seed} ({si + 1}/{len(seeds)})")
        print(f"{'=' * 72}")

        # ─── Phase A: Deep-only models (as_is) ───────────────
        if not args.skip_phase_a:
            cmd = build_command(
                seed=seed,
                models=PHASE_A_MODELS,
                ablation_mode="as_is",
                extra_flags=extra,
            )
            rc = run_command(
                cmd,
                dry_run=args.dry_run,
                label=f"Seed {seed} - Phase A: Deep-only (as_is)",
            )
            total_runs += 1
            if rc == 0:
                total_ok += 1
            else:
                total_fail += 1

        # ─── Phase B: LightGBM + logit fusion (baseline_plus) ─
        if not args.skip_phase_b:
            cmd = build_command(
                seed=seed,
                models=PHASE_B_MODELS,
                ablation_mode="baseline_plus",
                extra_flags=extra,
            )
            rc = run_command(
                cmd,
                dry_run=args.dry_run,
                label=f"Seed {seed} - Phase B: LightGBM + Logit Fusion (baseline_plus)",
            )
            total_runs += 1
            if rc == 0:
                total_ok += 1
            else:
                total_fail += 1

    # ─── Summary ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Total runs:  {total_runs}")
    print(f"  Succeeded:   {total_ok}")
    print(f"  Failed:      {total_fail}")
    print(f"  Total time:  {elapsed / 60:.1f} min ({elapsed / 3600:.2f} hours)")
    print(f"{'=' * 72}\n")

    if total_fail > 0:
        print(f"  [WARN] {total_fail} run(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
