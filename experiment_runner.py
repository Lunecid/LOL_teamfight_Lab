"""
experiment_runner.py — Systematic Ablation Study Runner

이 스크립트는 7가지 개선안의 체계적 실험을 자동화합니다.
Phase 1~5의 전체 실험 프로토콜을 단일 진입점에서 관리합니다.

Usage:
    python experiment_runner.py --phase 1                    # Baseline reproduction
    python experiment_runner.py --phase 2 --treatment 1      # Single-factor: Focal Loss
    python experiment_runner.py --phase 2 --treatment all     # All single-factor
    python experiment_runner.py --phase 3                    # Interaction analysis
    python experiment_runner.py --phase 4                    # Sensitivity analysis
    python experiment_runner.py --phase 5                    # Final test evaluation

수학적 배경:
    각 Treatment T_i에 대해:
    Δ_i = AUC(Baseline + T_i) - AUC(Baseline)

    통계 검정:
    H_0: Δ_i ≤ 0  vs  H_1: Δ_i > 0
    검정: DeLong's test (AUC), McNemar's test (classification)
    보정: Holm-Bonferroni (m=7 multiple comparisons)
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.experiment_io import load_results_json as _load_results
from app.experiment_io import save_results_json as _save_results
from app.experiment_runner_io import (
    build_parser,
    deserialize_phase2_results,
    deserialize_results,
    determine_best_treatments,
)
from app.experiment_stats import (
    bootstrap_ci,
    holm_bonferroni,
    safe_mean as _safe_mean,
    safe_std as _safe_std,
)
from app.experiment_runtime import run_single_experiment
from app.experiment_types import AblationSummary, ExperimentResult, SEEDS, TREATMENTS, TREATMENT_GROUPS, Treatment

# ──────────────────────────────────────────────────────────────
# 5. Phase Executors
# ──────────────────────────────────────────────────────────────

def run_phase1_baseline(args: argparse.Namespace) -> List[ExperimentResult]:
    """Phase 1: Baseline Reproduction.

    5개 seed로 기존 시스템을 재현하여 정확한 기준선 확립.

    수학적 목적:
        μ_baseline = (1/S) Σ_{s=1}^{S} AUC^{(s)}
        σ_baseline = sqrt((1/(S-1)) Σ (AUC^{(s)} - μ)²)
    """
    print("=" * 70)
    print("PHASE 1: Baseline Reproduction (5 seeds)")
    print("=" * 70)

    results = []

    for seed in SEEDS:
        print(f"\n--- Baseline | seed={seed} ---")

        if args.dry_run:
            result = ExperimentResult(
                treatment_id=0, treatment_name="Baseline",
                seed=seed, hp_config={},
            )
        else:
            result = run_single_experiment(
                treatment_overlay={},  # empty = pure baseline
                seed=seed,
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag="Baseline",
            )
            result.treatment_id = 0
            result.treatment_name = "Baseline"

        results.append(result)

    _print_phase1_summary(results)
    return results


def run_phase2_single_factor(
        args: argparse.Namespace,
        treatment_ids: List[int],
        baseline_results: List[ExperimentResult],
) -> Dict[int, List[ExperimentResult]]:
    """Phase 2: Single-Factor Ablation.

    각 Treatment T_i를 독립적으로 적용하여 개별 효과 측정.

    실험 매트릭스:
        E_single = {(Baseline + T_i) | i ∈ treatment_ids}
        총 |treatment_ids| × 5 runs
    """
    print("=" * 70)
    print(f"PHASE 2: Single-Factor Ablation (treatments={treatment_ids})")
    print("=" * 70)

    all_results: Dict[int, List[ExperimentResult]] = {}

    for tid in treatment_ids:
        treatment = TREATMENTS[tid]
        print("\n" + ("-" * 50))
        print(f"Treatment T_{tid}: {treatment.name}")
        print(f"Description:\n{treatment.description}")
        print("-" * 50)

        # Step 1: HP Grid Search
        best_hp = _hp_grid_search(args, treatment)

        # Step 2: 5-seed evaluation with best HP
        results_for_treatment = []

        # Merge base overlay + best HP
        merged_overlay = {**treatment.config_overlay, **best_hp}

        for seed in SEEDS:
            print(f"  seed={seed}, hp={best_hp}")

            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=tid, treatment_name=treatment.name,
                    seed=seed, hp_config=best_hp,
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=merged_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"T{tid}_{treatment.short_name}",
                )
                result.treatment_id = tid
                result.treatment_name = treatment.name
                result.hp_config = best_hp

            results_for_treatment.append(result)

        # Detector/sample diagnostics snapshot
        valid_counts = [r for r in results_for_treatment if r.n_train >= 0 and r.n_val >= 0 and r.n_test >= 0]
        if valid_counts:
            tr_mean = float(np.mean([r.n_train for r in valid_counts]))
            va_mean = float(np.mean([r.n_val for r in valid_counts]))
            te_mean = float(np.mean([r.n_test for r in valid_counts]))
            print(f"  [DIAG] mean split sizes train/val/test = {tr_mean:.1f}/{va_mean:.1f}/{te_mean:.1f}")
        valid_fights = [r.n_fights_all for r in results_for_treatment if r.n_fights_all >= 0]
        if valid_fights:
            print(f"  [DIAG] mean detected fights (all refs) = {float(np.mean(valid_fights)):.1f}")

        all_results[tid] = results_for_treatment

    # Compute summaries and statistical tests
    summaries = _compute_ablation_summaries(all_results, baseline_results)
    _print_phase2_summary(summaries)

    return all_results


def _hp_grid_search(args: argparse.Namespace, treatment: Treatment) -> Dict[str, Any]:
    """하이퍼파라미터 그리드 서치.

    전략: Val AUC 기준 1-seed (seed=7) 빠른 탐색 후 best 선택.

    수학적 정의:
        θ* = argmax_{θ ∈ Θ} AUC_val(Baseline + T_i(θ); seed=7)
    """
    if not treatment.hp_grid:
        return {}

    print(f"  [HP Search] Grid: {treatment.hp_grid}")

    keys = list(treatment.hp_grid.keys())
    values = list(treatment.hp_grid.values())

    best_val_auc = -1.0
    best_config = {}

    for combo in itertools.product(*values):
        hp_config = dict(zip(keys, combo))
        print(f"    Trying: {hp_config} ...", end=" ")

        if args.dry_run:
            val_auc = -1.0
        else:
            # Merge treatment overlay + this HP combo
            merged_overlay = {**treatment.config_overlay, **hp_config}
            result = run_single_experiment(
                treatment_overlay=merged_overlay,
                seed=SEEDS[0],  # single seed for speed
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag=f"HP_{treatment.short_name}",
            )
            val_auc = result.val_auc

        print(f"val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = hp_config

    print(f"  [HP Search] Best: {best_config} (val_auc={best_val_auc:.4f})")
    return best_config


def _resolve_treatment_selection(raw: str) -> List[int]:
    token = str(raw or "").strip().lower()
    if token == "all":
        return sorted(TREATMENTS.keys())

    if token in TREATMENT_GROUPS:
        ids = [int(tid) for tid in TREATMENT_GROUPS[token] if int(tid) in TREATMENTS]
        if not ids:
            raise ValueError(f"Treatment group '{raw}' is empty after filtering.")
        return sorted(set(ids))

    ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    missing = [tid for tid in ids if tid not in TREATMENTS]
    if missing:
        raise ValueError(f"Unknown treatment id(s): {missing}. Available: {sorted(TREATMENTS.keys())}")
    return ids


def run_phase3_interaction(
        args: argparse.Namespace,
        phase2_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
        top_k: int = 3,
) -> Dict[str, List[ExperimentResult]]:
    """Phase 3: Interaction Analysis.

    3.1 Pairwise Interaction:
        Interaction_{i,j} = Δ_{i+j} - (Δ_i + Δ_j)

    3.2 Cumulative Addition (Forward Selection):
        AUC_k = AUC(Baseline + T_{rank1} + ... + T_{rank_k})
        MC_k = AUC_k - AUC_{k-1}  (marginal contribution)
    """
    print("=" * 70)
    print("PHASE 3: Interaction Analysis")
    print("=" * 70)

    # Rank treatments by mean Δ_val_auc
    treatment_deltas = {}
    baseline_pos = [float(r.val_auc) for r in baseline_results if r.val_auc > 0]
    baseline_all = [float(r.val_auc) for r in baseline_results]
    baseline_mean = _safe_mean(baseline_pos, default=_safe_mean(baseline_all, default=-1.0))
    if not baseline_pos:
        print(
            f"[WARN] No positive baseline val_auc found. "
            f"Fallback baseline_mean={baseline_mean:.4f} for interaction math."
        )

    for tid, results in phase2_results.items():
        val_aucs = [r.val_auc for r in results if r.val_auc > 0]
        if val_aucs:
            delta_mean = np.mean(val_aucs) - baseline_mean
            treatment_deltas[tid] = delta_mean

    ranked = sorted(treatment_deltas.items(), key=lambda x: -x[1])
    top_ids = [tid for tid, _ in ranked[:top_k]]

    print(f"\nTop-{top_k} Treatments: "
          f"{[(tid, TREATMENTS[tid].name, f'Δ={d:.4f}') for tid, d in ranked[:top_k]]}")

    all_results: Dict[str, List[ExperimentResult]] = {}

    # 3.1 Pairwise Combinations
    pairs = list(itertools.combinations(top_ids, 2))
    print(f"\n--- 3.1 Pairwise Interaction ({len(pairs)} pairs) ---")

    for i, j in pairs:
        pair_key = f"pair_{i}_{j}"
        print(f"\n  Pair: T_{i}({TREATMENTS[i].short_name}) + T_{j}({TREATMENTS[j].short_name})")

        pair_results = []
        combined_overlay = {**TREATMENTS[i].config_overlay, **TREATMENTS[j].config_overlay}

        for seed in SEEDS:
            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=-1, treatment_name=f"T{i}+T{j}",
                    seed=seed, hp_config={},
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=combined_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"T{i}+T{j}",
                )
                result.treatment_id = -1
                result.treatment_name = f"T{i}+T{j}"

            pair_results.append(result)

        all_results[pair_key] = pair_results

        # Compute interaction
        delta_i = treatment_deltas.get(i, 0.0)
        delta_j = treatment_deltas.get(j, 0.0)
        pair_mean = _safe_mean([float(r.val_auc) for r in pair_results if r.val_auc > 0], default=baseline_mean)
        delta_ij = pair_mean - baseline_mean
        interaction = delta_ij - (delta_i + delta_j)

        label = "SYNERGY" if interaction > 0.001 else ("REDUNDANCY" if interaction < -0.001 else "INDEPENDENT")
        print(f"    Δ_i={delta_i:.4f}, Δ_j={delta_j:.4f}, Δ_{{i+j}}={delta_ij:.4f}")
        print(f"    Interaction = {interaction:+.4f} ({label})")

    # 3.2 Cumulative Addition (Forward Selection)
    print(f"\n--- 3.2 Cumulative Addition (Forward Selection) ---")

    cumulative_ids = []
    prev_auc = baseline_mean

    for step, (tid, delta) in enumerate(ranked):
        cumulative_ids.append(tid)
        combo_key = f"cumul_step{step + 1}"

        # Merge all treatment overlays accumulated so far
        cumul_overlay = {}
        for cid in cumulative_ids:
            cumul_overlay.update(TREATMENTS[cid].config_overlay)

        combo_results = []
        for seed in SEEDS:
            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=-1, treatment_name=f"Cumul_step{step + 1}",
                    seed=seed, hp_config={},
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=cumul_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"Cumul_step{step + 1}",
                )
                result.treatment_id = -1
                result.treatment_name = f"Cumul_step{step + 1}"

            combo_results.append(result)

        all_results[combo_key] = combo_results

        current_auc = _safe_mean([float(r.val_auc) for r in combo_results if r.val_auc > 0], default=prev_auc)
        mc = current_auc - prev_auc

        print(f"  Step {step + 1}: +T_{tid}({TREATMENTS[tid].short_name}) → "
              f"AUC={current_auc:.4f}, MC={mc:+.4f}")
        prev_auc = current_auc

    return all_results


def run_phase4_sensitivity(
        args: argparse.Namespace,
        phase2_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
) -> Dict[str, List[ExperimentResult]]:
    """Phase 4: Hyperparameter Sensitivity Analysis.

    Phase 2에서 significant한 각 Treatment에 대해
    HP grid의 전체 surface를 5-seed로 스캔.

    수학적 목적:
        ∀θ_j ∈ Grid(T_i):
            AUC(θ_j) = (1/S) Σ_{s} AUC(Baseline + T_i(θ_j); s)

        Sensitivity = std(AUC(θ)) / mean(AUC(θ))  (coefficient of variation)

    낮은 sensitivity → 해당 HP에 robust → 실무 적용 용이
    높은 sensitivity → fine-tuning 필요 → 논문에서 주의 명시
    """
    print("=" * 70)
    print("PHASE 4: Hyperparameter Sensitivity Analysis")
    print("=" * 70)

    all_results: Dict[str, List[ExperimentResult]] = {}

    for tid, treatment in TREATMENTS.items():
        if not treatment.hp_grid:
            print(f"\n  T_{tid} ({treatment.name}): No HP grid → skip")
            continue

        print("\n" + ("-" * 50))
        print(f"Treatment T_{tid}: {treatment.name}")
        print(f"HP Grid: {treatment.hp_grid}")
        print("-" * 50)

        keys = list(treatment.hp_grid.keys())
        values = list(treatment.hp_grid.values())

        hp_aucs: Dict[str, List[float]] = {}

        for combo in itertools.product(*values):
            hp_config = dict(zip(keys, combo))
            hp_key = str(hp_config)
            combo_tag = f"T{tid}_sens_{hp_config}"

            merged_overlay = {**treatment.config_overlay, **hp_config}
            combo_results = []

            for seed in SEEDS:
                if args.dry_run:
                    result = ExperimentResult(
                        treatment_id=tid, treatment_name=treatment.name,
                        seed=seed, hp_config=hp_config,
                    )
                else:
                    result = run_single_experiment(
                        treatment_overlay=merged_overlay,
                        seed=seed,
                        feature_set=args.feature_set,
                        split_mode=args.split_mode,
                        experiment_tag=combo_tag,
                    )
                    result.treatment_id = tid
                    result.treatment_name = treatment.name
                    result.hp_config = hp_config

                combo_results.append(result)

            all_results[combo_tag] = combo_results

            val_aucs = [r.val_auc for r in combo_results if r.val_auc > 0]
            if val_aucs:
                hp_aucs[hp_key] = val_aucs
                print(f"  {hp_config}: AUC = {_safe_mean(val_aucs):.4f} ± {_safe_std(val_aucs):.4f}")

        # Sensitivity summary
        if hp_aucs:
            all_means = [np.mean(v) for v in hp_aucs.values()]
            cv = np.std(all_means) / max(np.mean(all_means), 1e-10)
            print(f"\n  Sensitivity (CV of AUC means): {cv:.4f}")
            if cv < 0.01:
                print(f"  → Robust: HP 선택이 결과에 미미한 영향")
            elif cv < 0.03:
                print(f"  → Moderate: 적정 범위 내 안정적")
            else:
                print(f"  → Sensitive: fine-tuning 필요")

    return all_results


def run_phase5_final_test(
        args: argparse.Namespace,
        best_treatment_ids: List[int],
        best_hp_configs: Dict[int, Dict[str, Any]],
) -> List[ExperimentResult]:
    """Phase 5: Final Test Set Evaluation.

    ⚠️ Test set은 이 Phase에서 단 한 번만 평가.

    Comprehensive metrics:
        - Discrimination: AUC, AP, F1
        - Calibration: Brier, ECE
        - Robustness: across seeds
    """
    print("=" * 70)
    print("PHASE 5: FINAL TEST SET EVALUATION (ONE-TIME)")
    print(f"Active treatments: {[TREATMENTS[t].name for t in best_treatment_ids]}")
    print("=" * 70)
    print("\n⚠️  WARNING: Test set is evaluated ONCE. No further tuning allowed.\n")

    # Merge all selected treatments into a single overlay
    final_overlay: Dict[str, Any] = {}
    for tid in best_treatment_ids:
        final_overlay.update(TREATMENTS[tid].config_overlay)
        final_overlay.update(best_hp_configs.get(tid, {}))

    results = []

    for seed in SEEDS:
        print(f"  Final model | seed={seed}")

        if args.dry_run:
            result = ExperimentResult(
                treatment_id=-1, treatment_name="Final_Model",
                seed=seed,
                hp_config={t: best_hp_configs.get(t, {}) for t in best_treatment_ids},
            )
        else:
            result = run_single_experiment(
                treatment_overlay=final_overlay,
                seed=seed,
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag="Final_Model",
            )
            result.treatment_id = -1
            result.treatment_name = "Final_Model"

        results.append(result)

    _print_phase5_summary(results)
    return results


# ──────────────────────────────────────────────────────────────
# 5b. Phase 6: Feature Ablation Analysis
# ──────────────────────────────────────────────────────────────

def run_phase6_feature_ablation(
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Phase 6: Feature Ablation Analysis.

    Four analyses for LightGBM baseline review:
      1. Single-feature ablation (bJNG_cs_cooldownReduction__max)
      2. Static-attribute temporal aggregation validation
      3. SHAP-based parsimonious model (top-k feature selection)
      4. Logit pipeline integrity check
    """
    from app.experiment import run as _run_experiment
    from app.experiment_runtime import reset_config_to_baseline
    from core.config import cfg

    print("=" * 70)
    print("PHASE 6: Feature Ablation Analysis")
    print("=" * 70)

    reset_config_to_baseline(cfg)

    # Build refs via the same pipeline as Phase 1
    seed = args.seed
    feature_set = args.feature_set

    phase6_dir = output_dir / "phase6_feature_ablation"
    phase6_dir.mkdir(parents=True, exist_ok=True)
    log_fp = phase6_dir / "phase6.log"

    from core.utils import write_log
    write_log("[PHASE 6] Starting feature ablation analysis", log_fp)

    if args.dry_run:
        print("  [DRY-RUN] Would run feature ablation analysis")
        print("  Analyses: single_feature_ablation, static_temporal_validation,")
        print("            parsimonious_model, logit_pipeline_check")
        return

    # Build data references (same as baseline)
    try:
        from app.experiment import _build_refs_for_phase6
        tr_refs, va_refs, te_refs = _build_refs_for_phase6(
            feature_set=feature_set,
            split_mode=args.split_mode,
            seed=seed,
            log_fp=log_fp,
        )
    except (ImportError, AttributeError):
        # Fallback: build refs directly
        write_log("[PHASE 6] Falling back to direct ref building", log_fp)
        try:
            from data.index_split import build_split_refs
            from data.cache_io import load_dataset_index
            idx = load_dataset_index()
            if not idx:
                print("[PHASE 6 ERROR] No dataset index. Run cache builder first.")
                return
            tr_refs, va_refs, te_refs = build_split_refs(
                idx, mode=args.split_mode, seed=seed,
            )
        except Exception as e:
            print(f"[PHASE 6 ERROR] Cannot build refs: {e}")
            write_log(f"[PHASE 6] Cannot build refs: {e}", log_fp)
            return

    write_log(
        f"[PHASE 6] Refs: train={len(tr_refs)} val={len(va_refs)} test={len(te_refs)}",
        log_fp,
    )

    from analysis.feature_ablation import run_all_analyses
    results = run_all_analyses(
        feature_set=feature_set,
        tr_refs=tr_refs,
        va_refs=va_refs,
        te_refs=te_refs,
        seed=seed,
        log_fp=log_fp,
        out_dir=phase6_dir,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 6 SUMMARY: Feature Ablation Analysis")
    print("=" * 70)

    for name, info in results.items():
        status = "PASS" if info.get("ok", False) else "FAIL"
        print(f"  [{status}] {name}")
        if "delta_val_auc" in info and info["delta_val_auc"] is not None:
            print(f"         delta_val_auc = {info['delta_val_auc']:+.4f}")
        if "verdict" in info:
            print(f"         verdict = {info['verdict']}")
        if "n_models" in info:
            print(f"         n_models = {info['n_models']}")
        if "issues" in info:
            for issue in info["issues"][:3]:
                print(f"         issue: {issue}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 6. Summary / Reporting Utilities
# ──────────────────────────────────────────────────────────────

def _compute_ablation_summaries(
        treatment_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
) -> List[AblationSummary]:
    """Phase 2 결과를 Treatment별로 요약.

    각 Treatment에 대해:
    1. 5-seed mean ± std
    2. Baseline 대비 Δ 및 95% CI
    3. Cohen's d effect size
    """
    baseline_val_aucs = np.array([r.val_auc for r in baseline_results])
    summaries = []

    for tid, results in treatment_results.items():
        treatment = TREATMENTS[tid]
        val_aucs = np.array([r.val_auc for r in results])
        test_aucs = np.array([r.test_auc for r in results])

        # Paired differences
        n_pairs = min(len(val_aucs), len(baseline_val_aucs))
        deltas = val_aucs[:n_pairs] - baseline_val_aucs[:n_pairs]
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas, ddof=1)) if n_pairs > 1 else 0.0

        # 95% CI (t-distribution)
        ci_low, ci_high = delta_mean, delta_mean
        if n_pairs > 1 and delta_std > 0:
            try:
                from scipy.stats import t as t_dist
                t_crit = t_dist.ppf(0.975, df=n_pairs - 1)
                se = delta_std / math.sqrt(n_pairs)
                ci_low = delta_mean - t_crit * se
                ci_high = delta_mean + t_crit * se
            except ImportError:
                # fallback: z=1.96
                se = delta_std / math.sqrt(n_pairs)
                ci_low = delta_mean - 1.96 * se
                ci_high = delta_mean + 1.96 * se

        # Cohen's d = Δ / pooled_std
        var_t = np.var(val_aucs, ddof=1) if len(val_aucs) > 1 else 0
        var_b = np.var(baseline_val_aucs, ddof=1) if len(baseline_val_aucs) > 1 else 0
        pooled_std = math.sqrt((var_t + var_b) / 2)
        cohens_d = delta_mean / max(pooled_std, 1e-10)

        summary = AblationSummary(
            treatment_id=tid,
            treatment_name=treatment.name,
            val_auc_mean=float(np.mean(val_aucs)),
            val_auc_std=float(np.std(val_aucs, ddof=1)) if len(val_aucs) > 1 else 0.0,
            val_auc_seeds=val_aucs.tolist(),
            test_auc_mean=float(np.mean(test_aucs)),
            test_auc_std=float(np.std(test_aucs, ddof=1)) if len(test_aucs) > 1 else 0.0,
            delta_val_auc_mean=delta_mean,
            delta_val_auc_std=delta_std,
            delta_val_auc_ci_low=ci_low,
            delta_val_auc_ci_high=ci_high,
            cohens_d=cohens_d,
        )

        summaries.append(summary)

    # Holm-Bonferroni correction
    p_values = [s.delong_p_value for s in summaries]
    significant = holm_bonferroni(p_values, alpha=0.05)
    for s, sig in zip(summaries, significant):
        s.significant_after_correction = sig

    return summaries


def _print_phase1_summary(results: List[ExperimentResult]) -> None:
    """Phase 1 결과 요약 출력."""
    val_aucs = [r.val_auc for r in results if r.val_auc > 0]
    test_aucs = [r.test_auc for r in results if r.test_auc > 0]

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY: Baseline Reproduction")
    print("=" * 60)

    if val_aucs:
        print(f"  Val  AUC: {_safe_mean(val_aucs):.4f} ± {_safe_std(val_aucs):.4f}")
        print(f"  Test AUC: {_safe_mean(test_aucs):.4f} ± {_safe_std(test_aucs):.4f}")
        print(f"  Seeds: {[f'{a:.4f}' for a in val_aucs]}")
        split_known = [r for r in results if r.n_train >= 0 and r.n_val >= 0 and r.n_test >= 0]
        if split_known:
            tr_mean = float(np.mean([r.n_train for r in split_known]))
            va_mean = float(np.mean([r.n_val for r in split_known]))
            te_mean = float(np.mean([r.n_test for r in split_known]))
            print(f"  Split n: train/val/test = {tr_mean:.1f}/{va_mean:.1f}/{te_mean:.1f}")
        fights_known = [r.n_fights_all for r in results if r.n_fights_all >= 0]
        if fights_known:
            print(f"  Detected fights(all): {float(np.mean(fights_known)):.1f}")
    else:
        print("  [No results yet — execute pipeline to populate]")

    print("=" * 60)


def _print_phase2_summary(summaries: List[AblationSummary]) -> None:
    """Phase 2 결과 요약 (Forest Plot 스타일 텍스트 테이블)."""
    print("\n" + "=" * 90)
    print("PHASE 2 SUMMARY: Single-Factor Ablation")
    print("=" * 90)
    print(f"{'Treatment':<30} {'Val AUC (μ±σ)':<18} {'Δ_val':<10} {'95% CI':<20} {'p-val':<8} {'Sig?':<5}")
    print("-" * 90)

    sorted_summaries = sorted(summaries, key=lambda s: -s.delta_val_auc_mean)

    for s in sorted_summaries:
        sig_marker = "***" if s.significant_after_correction else "ns"
        print(
            f"  T{s.treatment_id} {s.treatment_name:<25} "
            f"{s.val_auc_mean:.4f}±{s.val_auc_std:.4f}  "
            f"{s.delta_val_auc_mean:+.4f}   "
            f"[{s.delta_val_auc_ci_low:+.4f}, {s.delta_val_auc_ci_high:+.4f}]  "
            f"{s.delong_p_value:.4f}  "
            f"{sig_marker}"
        )

    print("-" * 90)
    print(f"  Correction: Holm-Bonferroni (m={len(summaries)}, α=0.05)")
    print(f"  Significant treatments: {sum(1 for s in summaries if s.significant_after_correction)}/{len(summaries)}")
    print("=" * 90)


def _print_phase5_summary(results: List[ExperimentResult]) -> None:
    """Phase 5 최종 결과 요약."""
    print("\n" + "=" * 70)
    print("PHASE 5 SUMMARY: FINAL TEST SET EVALUATION")
    print("=" * 70)

    val_aucs = [r.val_auc for r in results if r.val_auc > 0]
    test_aucs = [r.test_auc for r in results if r.test_auc > 0]
    val_briers = [r.val_brier for r in results if r.val_brier >= 0]
    val_eces = [r.val_ece for r in results if r.val_ece >= 0]

    metrics = {
        "Val AUC": val_aucs,
        "Test AUC": test_aucs,
        "Val Brier": val_briers,
        "Val ECE": val_eces,
    }

    for name, values in metrics.items():
        if values:
            print(f"  {name:<15}: {_safe_mean(values):.4f} ± {_safe_std(values):.4f}")

    if val_aucs and test_aucs:
        gen_gap = np.mean(val_aucs) - np.mean(test_aucs)
        print(f"\n  Generalization Gap (Val - Test): {gen_gap:+.4f}")
        if abs(gen_gap) < 0.01:
            print("  → ✓ Good generalization (gap < 0.01)")
        elif abs(gen_gap) < 0.02:
            print("  → ⚠ Moderate gap (0.01 < gap < 0.02)")
        else:
            print("  → ✗ Large gap (gap > 0.02) — potential val overfitting")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 7. Main Entry Point
# ──────────────────────────────────────────────────────────────

def run_phase_cli(args: argparse.Namespace) -> None:
    """Execute the ablation workflow from an already-parsed namespace."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"LoL Teamfight Prediction — Ablation Study")
    print(f"Phase: {args.phase}")
    print(f"Output: {output_dir}")
    print(f"Seeds: {SEEDS}")
    print(f"Dry-run: {args.dry_run}")

    # [SPEED] Propagate speed settings via environment for run_single_experiment
    if args.speed:
        os.environ["LOL_SPEED_OVERLAY"] = "1"
        os.environ["LOL_VRAM_GB"] = str(args.vram)
        os.environ["LOL_SPEED_PROFILE"] = str(args.speed_profile)
        print(f"Speed overlay: ENABLED (VRAM={args.vram:.1f} GB, profile={args.speed_profile})")
    else:
        os.environ["LOL_SPEED_OVERLAY"] = "0"
        os.environ["LOL_SPEED_PROFILE"] = "none"
        print(f"Speed overlay: DISABLED")

    print(f"{'=' * 70}\n")

    if args.phase == 1:
        results = run_phase1_baseline(args)
        _save_results(output_dir / "phase1_baseline.json", results)

    elif args.phase == 2:
        # Parse treatment IDs
        treatment_ids = _resolve_treatment_selection(args.treatment)

        # Load baseline results (must exist from Phase 1)
        baseline_path = output_dir / "phase1_baseline.json"
        baseline_data = _load_results(baseline_path) if baseline_path.exists() else None
        baseline_results = deserialize_results(baseline_data) if baseline_data else []

        if not baseline_results:
            print("[WARN] No baseline results found. Run Phase 1 first.")
            print("       Proceeding with placeholder baseline...")
            baseline_results = [ExperimentResult(0, "Baseline", s, {}) for s in SEEDS]

        results = run_phase2_single_factor(args, treatment_ids, baseline_results)
        _save_results(output_dir / "phase2_single_factor.json", results)

    elif args.phase == 3:
        # Load Phase 1 & 2 results
        baseline_data = _load_results(output_dir / "phase1_baseline.json")
        phase2_data = _load_results(output_dir / "phase2_single_factor.json")

        if not baseline_data or not phase2_data:
            print("[ERROR] Phase 1 and Phase 2 results required. Run them first.")
            return

        baseline_results = deserialize_results(baseline_data)
        phase2_results = deserialize_phase2_results(phase2_data)

        results = run_phase3_interaction(args, phase2_results, baseline_results, top_k=args.top_k)
        _save_results(output_dir / "phase3_interaction.json", results)

    elif args.phase == 4:
        # Load Phase 1 & 2 results
        baseline_data = _load_results(output_dir / "phase1_baseline.json")
        phase2_data = _load_results(output_dir / "phase2_single_factor.json")

        if not baseline_data or not phase2_data:
            print("[ERROR] Phase 1 and Phase 2 results required. Run them first.")
            return

        baseline_results = deserialize_results(baseline_data)
        phase2_results = deserialize_phase2_results(phase2_data)

        results = run_phase4_sensitivity(args, phase2_results, baseline_results)
        _save_results(output_dir / "phase4_sensitivity.json", results)

    elif args.phase == 5:
        print("[Phase 5] Final test evaluation — determining best treatments...")

        best_ids, best_hps = determine_best_treatments(output_dir)
        print(f"  Selected treatments: {best_ids}")
        print(f"  HP configs: {best_hps}")

        results = run_phase5_final_test(args, best_ids, best_hps)
        _save_results(output_dir / "phase5_final_test.json", results)

    elif args.phase == 6:
        print("[Phase 6] Feature ablation analysis (SHAP, static-attr validation, parsimonious model, logit pipeline)")
        run_phase6_feature_ablation(args, output_dir)

    print("\n[DONE] Experiment phase completed.")


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_phase_cli(args)


if __name__ == "__main__":
    main()
