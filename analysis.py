"""
analysis.py — Ablation Study 결과 분석 및 시각화

실험 결과를 통합하여:
1. Forest Plot (효과 크기 + 95% CI)
2. ROC Curve 비교
3. Reliability Diagram (Calibration)
4. Interaction Heatmap
5. Cumulative Addition Curve
6. Attention Weight Visualization
7. Role-Adjacency Matrix Heatmap

수학적 배경:
    - DeLong's Test: AUC 차이의 유의성 검정
    - McNemar's Test: 분류 결정의 pairwise 비교
    - Holm-Bonferroni: FWER 다중 비교 보정
    - Cohen's d: 표준화된 효과 크기
    - ECE: Expected Calibration Error

Usage:
    python analysis.py --results-dir ./ablation_results --output-dir ./figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# 1. Forest Plot (Effect Size Visualization)
# ──────────────────────────────────────────────────────────────

def create_forest_plot_data(
        summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Forest Plot 데이터 생성.

    Forest Plot은 메타분석에서 개별 연구의 효과 크기와
    신뢰구간을 시각적으로 비교하는 표준 도구.

    각 Treatment에 대해:
        ●─────────●─────────●
        CI_low    Δ_mean    CI_high

    수학적 정의:
        Δ_i = (1/S) Σ_{s=1}^S [AUC_improved^{(s)} - AUC_baseline^{(s)}]
        SE_i = sqrt((1/S(S-1)) Σ (Δ^{(s)} - Δ̄)²)
        CI = Δ̄ ± t_{0.025, S-1} · SE
    """
    plot_data = {
        "treatments": [],
        "deltas": [],
        "ci_lows": [],
        "ci_highs": [],
        "p_values": [],
        "significant": [],
    }

    for s in sorted(summaries, key=lambda x: -x.get("delta_val_auc_mean", 0)):
        plot_data["treatments"].append(f"T{s['treatment_id']}: {s['treatment_name']}")
        plot_data["deltas"].append(s.get("delta_val_auc_mean", 0))
        plot_data["ci_lows"].append(s.get("delta_val_auc_ci_low", 0))
        plot_data["ci_highs"].append(s.get("delta_val_auc_ci_high", 0))
        plot_data["p_values"].append(s.get("delong_p_value", 1.0))
        plot_data["significant"].append(s.get("significant_after_correction", False))

    return plot_data


def generate_forest_plot_matplotlib(
        plot_data: Dict[str, Any],
        output_path: Path,
        title: str = "Single-Factor Ablation: Effect on Val AUC",
) -> None:
    """Matplotlib를 이용한 Forest Plot 생성.

    시각적 규칙:
        - 유의한 결과: ● (채워진 원, 파란색)
        - 비유의한 결과: ○ (빈 원, 회색)
        - 빨간 점선: Δ = 0 (no effect)
        - 초록 영역: positive effect zone
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available, skipping forest plot")
        return

    n = len(plot_data["treatments"])
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.8)))

    y_positions = list(range(n))

    for i in range(n):
        color = "#2196F3" if plot_data["significant"][i] else "#9E9E9E"
        marker = "o" if plot_data["significant"][i] else "o"
        fillstyle = "full" if plot_data["significant"][i] else "none"

        # CI line
        ax.plot(
            [plot_data["ci_lows"][i], plot_data["ci_highs"][i]],
            [y_positions[i], y_positions[i]],
            color=color, linewidth=2, zorder=2,
        )

        # Point estimate
        ax.plot(
            plot_data["deltas"][i], y_positions[i],
            marker=marker, color=color, markersize=10,
            fillstyle=fillstyle, markeredgewidth=2, zorder=3,
        )

        # P-value annotation
        p = plot_data["p_values"][i]
        sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.annotate(
            f"  Δ={plot_data['deltas'][i]:+.4f} ({sig_text})",
            xy=(plot_data["ci_highs"][i], y_positions[i]),
            fontsize=9, va="center",
        )

    # Reference line at 0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7, zorder=1)

    # Positive zone shading
    xlim = ax.get_xlim()
    ax.axvspan(0, xlim[1], alpha=0.05, color="green", zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["treatments"])
    ax.set_xlabel("Δ Val AUC (vs Baseline)")
    ax.set_title(title)
    ax.invert_yaxis()

    # Legend
    sig_patch = mpatches.Patch(color="#2196F3", label="Significant (p<0.05, Holm-Bonferroni)")
    ns_patch = mpatches.Patch(color="#9E9E9E", label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Forest plot: {output_path}")


# ──────────────────────────────────────────────────────────────
# 2. Interaction Heatmap
# ──────────────────────────────────────────────────────────────

def generate_interaction_heatmap(
        interaction_matrix: np.ndarray,
        treatment_names: List[str],
        output_path: Path,
) -> None:
    """Pairwise Interaction Heatmap.

    수학적 정의:
        Interaction_{i,j} = Δ_{i+j} - (Δ_i + Δ_j)

    색상 규칙:
        빨강: Synergy (Interaction > 0)
        파랑: Redundancy (Interaction < 0)
        흰색: Independent (Interaction ≈ 0)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available")
        return

    n = len(treatment_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Symmetric colormap centered at 0
    vmax = max(abs(interaction_matrix.min()), abs(interaction_matrix.max()))
    vmax = max(vmax, 0.001)  # prevent zero range

    im = ax.imshow(interaction_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotations
    for i in range(n):
        for j in range(n):
            val = interaction_matrix[i, j]
            if i != j and not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(treatment_names, rotation=45, ha="right")
    ax.set_yticklabels(treatment_names)
    ax.set_title("Pairwise Interaction: Synergy (+) vs Redundancy (-)")

    plt.colorbar(im, ax=ax, label="Interaction Effect (ΔAUC)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Interaction heatmap: {output_path}")


# ──────────────────────────────────────────────────────────────
# 3. Cumulative Addition Curve
# ──────────────────────────────────────────────────────────────

def generate_cumulative_curve(
        steps: List[str],
        auc_means: List[float],
        auc_stds: List[float],
        output_path: Path,
) -> None:
    """Cumulative Addition Curve.

    Forward selection 결과를 시각화:

    AUC ▲
        │         ●─────── (T5: Role Adj)
        │       ●
        │     ●
        │   ●
        │ ●
        │●  Baseline
        └─────────────────────► Steps

    Marginal Contribution:
        MC_k = AUC_k - AUC_{k-1}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cumulative AUC
    x = range(len(steps))
    ax1.errorbar(x, auc_means, yerr=auc_stds, marker="o", capsize=4,
                 color="#2196F3", linewidth=2, markersize=8)
    ax1.fill_between(x,
                     [m - s for m, s in zip(auc_means, auc_stds)],
                     [m + s for m, s in zip(auc_means, auc_stds)],
                     alpha=0.15, color="#2196F3")

    # Baseline reference line
    ax1.axhline(y=auc_means[0], color="red", linestyle="--", alpha=0.5, label="Baseline")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(steps, rotation=45, ha="right")
    ax1.set_ylabel("Val AUC")
    ax1.set_title("Cumulative Addition of Improvements")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Marginal Contribution
    if len(auc_means) > 1:
        mc = [auc_means[i] - auc_means[i - 1] for i in range(1, len(auc_means))]
        colors = ["#4CAF50" if m > 0 else "#F44336" for m in mc]

        ax2.bar(range(len(mc)), mc, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xticks(range(len(mc)))
        ax2.set_xticklabels(steps[1:], rotation=45, ha="right")
        ax2.set_ylabel("Marginal Contribution (ΔAUC)")
        ax2.set_title("Marginal Contribution per Step")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Cumulative curve: {output_path}")


# ──────────────────────────────────────────────────────────────
# 4. Reliability Diagram (Calibration)
# ──────────────────────────────────────────────────────────────

def generate_reliability_diagram(
        y_true: np.ndarray,
        y_prob_baseline: np.ndarray,
        y_prob_improved: np.ndarray,
        output_path: Path,
        n_bins: int = 15,
) -> None:
    """Reliability Diagram for Calibration Assessment.

    수학적 배경:
        완벽한 calibration: P(Y=1 | p̂ = p) = p  ∀p ∈ [0,1]

        Reliability diagram은 이 조건을 시각적으로 검증:
        - x축: 예측 확률 (confidence)
        - y축: 실제 양성 비율 (accuracy)
        - 대각선: 완벽한 calibration

    ECE = Σ_{m=1}^M (|B_m|/N) · |acc(B_m) - conf(B_m)|
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _calibration_curve(y_true, y_prob, n_bins):
        """Compute calibration curve data."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            if i == 0:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            else:
                mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            n = mask.sum()
            if n > 0:
                bin_centers.append(y_prob[mask].mean())
                bin_accs.append(y_true[mask].mean())
                bin_counts.append(n)

        return np.array(bin_centers), np.array(bin_accs), np.array(bin_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Reliability diagram
    centers_b, accs_b, counts_b = _calibration_curve(y_true, y_prob_baseline, n_bins)
    centers_i, accs_i, counts_i = _calibration_curve(y_true, y_prob_improved, n_bins)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.plot(centers_b, accs_b, "o-", color="#F44336", markersize=6, label="Baseline")
    ax1.plot(centers_i, accs_i, "s-", color="#2196F3", markersize=6, label="Improved")

    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Right: Histogram of predictions
    ax2.hist(y_prob_baseline, bins=50, alpha=0.5, color="#F44336", label="Baseline", density=True)
    ax2.hist(y_prob_improved, bins=50, alpha=0.5, color="#2196F3", label="Improved", density=True)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Predictions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Reliability diagram: {output_path}")


# ──────────────────────────────────────────────────────────────
# 5. Role-Adjacency Matrix Visualization
# ──────────────────────────────────────────────────────────────

def generate_role_adjacency_heatmap(
        R_matrix: np.ndarray,
        output_path: Path,
) -> None:
    """학습된 Role-Adjacency 행렬 R의 시각화.

    R ∈ R^{5×5}, roles = [TOP, JNG, MID, BOT, SUP]

    softplus(R_{ij})가 실제 adjacency 가중치 승수:
        A'_{ij} = A^{dist}_{ij} × softplus(R_{r(i), r(j)})

    기대 패턴:
        R_{BOT,SUP} >> 평균  (봇 듀오 시너지)
        R_{JNG,MID} > 평균   (중정글 연동)
        R_{TOP,BOT} < 평균   (맵 반대편)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    roles = ["TOP", "JNG", "MID", "BOT", "SUP"]

    # softplus를 적용하여 실제 승수 표시
    R_softplus = np.log(1 + np.exp(R_matrix))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Raw R values
    im1 = ax1.imshow(R_matrix, cmap="RdBu_r", aspect="auto")
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, f"{R_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(roles)
    ax1.set_yticklabels(roles)
    ax1.set_title("Learned R (raw)")
    plt.colorbar(im1, ax=ax1)

    # Right: Softplus(R) = actual multiplier
    im2 = ax2.imshow(R_softplus, cmap="YlOrRd", aspect="auto")
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f"{R_softplus[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(roles)
    ax2.set_yticklabels(roles)
    ax2.set_title("softplus(R) = Adjacency Multiplier")
    plt.colorbar(im2, ax=ax2)

    plt.suptitle("Role-Aware Adjacency: Learned Interaction Strength", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Role adjacency heatmap: {output_path}")


# ──────────────────────────────────────────────────────────────
# 6. Comprehensive Result Table (LaTeX-ready)
# ──────────────────────────────────────────────────────────────

def generate_latex_table(
        summaries: List[Dict[str, Any]],
        baseline_summary: Dict[str, Any],
) -> str:
    r"""논문용 LaTeX 테이블 생성.

    Table 형식:
        \begin{table}[t]
        \caption{Single-factor ablation results.}
        \begin{tabular}{lccccl}
        Treatment & Val AUC & ΔAUC & 95\% CI & p-value & Sig. \\
        ...
        \end{tabular}
        \end{table}
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Single-factor ablation results on validation set (Patch 15.16). "
        r"Each treatment is applied independently to the baseline. "
        r"Results averaged over 5 random seeds with 95\% CIs from paired t-test. "
        r"Significance after Holm-Bonferroni correction ($m=7$, $\alpha=0.05$).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}lccccl@{}}",
        r"\toprule",
        r"Treatment & Val AUC ($\mu \pm \sigma$) & $\Delta$AUC & 95\% CI & $p$-value & Sig. \\",
        r"\midrule",
    ]

    # Baseline row
    b = baseline_summary
    lines.append(
        f"Baseline & ${b.get('val_auc_mean', 0):.4f} \\pm {b.get('val_auc_std', 0):.4f}$ "
        f"& --- & --- & --- & --- \\\\"
    )
    lines.append(r"\midrule")

    # Treatment rows (sorted by effect size)
    sorted_s = sorted(summaries, key=lambda x: -x.get("delta_val_auc_mean", 0))

    for s in sorted_s:
        sig = "\\checkmark" if s.get("significant_after_correction", False) else "---"
        p = s.get("delong_p_value", 1.0)
        p_str = f"<0.001" if p < 0.001 else f"{p:.3f}"

        lines.append(
            f"$\\mathcal{{T}}_{{{s['treatment_id']}}}$ {s['treatment_name']} "
            f"& ${s.get('val_auc_mean', 0):.4f} \\pm {s.get('val_auc_std', 0):.4f}$ "
            f"& ${s.get('delta_val_auc_mean', 0):+.4f}$ "
            f"& $[{s.get('delta_val_auc_ci_low', 0):+.4f}, {s.get('delta_val_auc_ci_high', 0):+.4f}]$ "
            f"& {p_str} "
            f"& {sig} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 7. Phase-Stratified Analysis
# ──────────────────────────────────────────────────────────────

def phase_stratified_auc(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fight_minutes: np.ndarray,
) -> Dict[str, float]:
    """게임 국면별 AUC 분해.

    정의:
        Early:  t ∈ [0, 14) minutes
        Mid:    t ∈ [14, 28) minutes
        Late:   t ∈ [28, ∞) minutes

    각 국면에서 독립적으로 AUC 계산:
        AUC_phase = AUC({(y_i, p_i) : t_i ∈ phase})
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    phases = {
        "early": (0, 14),
        "mid": (14, 28),
        "late": (28, float('inf')),
    }

    results = {}
    for phase_name, (t_low, t_high) in phases.items():
        mask = (fight_minutes >= t_low) & (fight_minutes < t_high)
        n_phase = mask.sum()

        if n_phase < 30:  # minimum samples for reliable AUC
            results[f"AUC_{phase_name}"] = float('nan')
            results[f"n_{phase_name}"] = int(n_phase)
            continue

        y_phase = y_true[mask]
        p_phase = y_prob[mask]

        # Check class balance
        if len(np.unique(y_phase)) < 2:
            results[f"AUC_{phase_name}"] = float('nan')
        else:
            results[f"AUC_{phase_name}"] = float(roc_auc_score(y_phase, p_phase))
        results[f"n_{phase_name}"] = int(n_phase)

    return results


def difficulty_stratified_auc(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        gold_diff: np.ndarray,
) -> Dict[str, float]:
    """교전 난이도별 AUC 분해.

    정의:
        Close:     |gold_diff| < 2000
        Moderate:  2000 ≤ |gold_diff| < 5000
        Stomp:     |gold_diff| ≥ 5000

    도메인 근거:
        - Close fights: 조합/포지셔닝이 결정적 (모델의 진짜 가치)
        - Stomp fights: 골드 차이로 대부분 예측 가능 (trivial)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    abs_gold = np.abs(gold_diff)

    strata = {
        "close": abs_gold < 2000,
        "moderate": (abs_gold >= 2000) & (abs_gold < 5000),
        "stomp": abs_gold >= 5000,
    }

    results = {}
    for name, mask in strata.items():
        n = mask.sum()
        if n < 30 or len(np.unique(y_true[mask])) < 2:
            results[f"AUC_{name}"] = float('nan')
        else:
            results[f"AUC_{name}"] = float(roc_auc_score(y_true[mask], y_prob[mask]))
        results[f"n_{name}"] = int(n)

    return results


# ──────────────────────────────────────────────────────────────
# 8. Summary Report Generator
# ──────────────────────────────────────────────────────────────

def generate_full_report(results_dir: Path, output_dir: Path) -> str:
    """전체 실험 결과 요약 보고서 생성 (데이터 구동형).

    실험 결과 JSON 파일을 읽어 실제 수치로 보고서를 채움.

    수학적 보고 항목
    ----------------
    1. Baseline: μ_base ± σ_base (S seeds)
    2. Single-factor: Δ_i = μ_i - μ_base, p-value (DeLong), Holm-corrected
    3. Interaction: Interaction_{i,j} = Δ_{i+j} - (Δ_i + Δ_j)
    4. Final: AUC_test, AP, F1, ECE, Brier

    Expected files in results_dir:
        phase1_baseline.json     — List[ExperimentResult]
        phase2_single_factor.json — {treatment_name: {summary: ..., results: [...]}}
        phase3_interaction.json  — {pairwise: [...], cumulative: [...]}
        phase5_final.json        — {metrics: {...}, treatment_combo: [...]}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Helper: safe JSON load ───
    def _load_json(name: str) -> Optional[Any]:
        fp = results_dir / name
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [WARN] Failed to load {name}: {e}")
        return None

    # ─── Load all phase results ───
    phase1 = _load_json("phase1_baseline.json")
    phase2 = _load_json("phase2_single_factor.json")
    phase3 = _load_json("phase3_interaction.json")
    phase4 = _load_json("phase4_sensitivity.json")
    phase5 = _load_json("phase5_final.json")

    lines: List[str] = []
    L = lines.append

    L("# Ablation Study Report: LoL Teamfight Prediction Improvements")
    L(f"")
    L(f"**Generated**: auto from experiment results")
    L(f"**Results directory**: `{results_dir}`")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 1. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════
    L("## 1. Executive Summary")
    L("")

    if phase1 and phase5:
        # Extract baseline mean AUC
        base_aucs = []
        for r in phase1:
            va = r.get("val_auc", r.get("metrics", {}).get("val", {}).get("auc", -1))
            if va > 0:
                base_aucs.append(va)

        base_mean = float(np.mean(base_aucs)) if base_aucs else -1
        base_std = float(np.std(base_aucs, ddof=1)) if len(base_aucs) > 1 else 0

        # Extract final test metrics
        final_met = phase5.get("metrics", {})
        final_test = final_met.get("test", {})
        final_val = final_met.get("val", {})

        L("| Metric | Baseline (μ ± σ) | Final Model | Δ |")
        L("|--------|-------------------|-------------|---|")

        if base_mean > 0:
            L(f"| Val AUC | {base_mean:.4f} ± {base_std:.4f} | "
              f"{final_val.get('auc', 'N/A'):.4f} | "
              f"{final_val.get('auc', base_mean) - base_mean:+.4f} |"
              if isinstance(final_val.get("auc"), (int, float))
              else f"| Val AUC | {base_mean:.4f} ± {base_std:.4f} | N/A | N/A |")

        test_auc = final_test.get("auc", "N/A")
        test_f1 = final_test.get("f1", "N/A")
        test_ap = final_test.get("ap", "N/A")

        if isinstance(test_auc, (int, float)):
            L(f"| Test AUC | — | {test_auc:.4f} | — |")
        if isinstance(test_f1, (int, float)):
            L(f"| Test F1 | — | {test_f1:.4f} | — |")
        if isinstance(test_ap, (int, float)):
            L(f"| Test AP | — | {test_ap:.4f} | — |")

        ece = final_test.get("ece", final_met.get("ece"))
        if isinstance(ece, (int, float)):
            L(f"| ECE | — | {ece:.4f} | — |")

        brier = final_test.get("brier", final_met.get("brier"))
        if isinstance(brier, (int, float)):
            L(f"| Brier | — | {brier:.4f} | — |")
    else:
        L("*결과 파일 미발견 — 실험 완료 후 자동 생성됩니다.*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 2. EXPERIMENTAL SETUP
    # ═══════════════════════════════════════════════════════════
    L("## 2. Experimental Setup")
    L("")
    L("- **Data**: Korean LoL matches, Patches 15.14–15.17")
    L("- **Split**: Temporal forward validation (Train: 15.14–15.15, Val: 15.16, Test: 15.17)")
    L("- **Seeds**: 5 random seeds (7, 42, 123, 256, 512)")
    L("- **Statistical Tests**: DeLong (AUC), McNemar (classification), Holm–Bonferroni correction")
    L("- **Significance level**: α = 0.05 (two-sided)")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 3. BASELINE REPRODUCTION (Phase 1)
    # ═══════════════════════════════════════════════════════════
    L("## 3. Baseline Reproduction (Phase 1)")
    L("")
    if phase1 and len(phase1) > 0:
        L(f"**{len(phase1)} seeds** evaluated:")
        L("")
        L("| Seed | Val AUC | Test AUC | Best Epoch | Train Time (s) |")
        L("|------|---------|----------|------------|----------------|")
        for r in phase1:
            seed = r.get("seed", "?")
            va = r.get("val_auc", r.get("metrics", {}).get("val", {}).get("auc", -1))
            te = r.get("test_auc", r.get("metrics", {}).get("test", {}).get("auc", -1))
            ep = r.get("best_epoch", "?")
            tt = r.get("train_time_sec", "?")
            va_s = f"{va:.4f}" if isinstance(va, (int, float)) and va > 0 else "N/A"
            te_s = f"{te:.4f}" if isinstance(te, (int, float)) and te > 0 else "N/A"
            tt_s = f"{tt:.1f}" if isinstance(tt, (int, float)) else "N/A"
            L(f"| {seed} | {va_s} | {te_s} | {ep} | {tt_s} |")

        # Compute summary stats
        va_vals = [r.get("val_auc", r.get("metrics", {}).get("val", {}).get("auc", -1))
                   for r in phase1]
        va_vals = [v for v in va_vals if isinstance(v, (int, float)) and v > 0]
        if va_vals:
            L(f"")
            L(f"**Summary**: μ = {np.mean(va_vals):.4f}, σ = {np.std(va_vals, ddof=1):.4f}, "
              f"range = [{min(va_vals):.4f}, {max(va_vals):.4f}]")
    else:
        L("*Phase 1 결과 미발견*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 4. SINGLE-FACTOR ABLATION (Phase 2)
    # ═══════════════════════════════════════════════════════════
    L("## 4. Single-Factor Ablation Results (Phase 2)")
    L("")

    summaries_for_forest = []

    if phase2 and isinstance(phase2, dict):
        L("| Treatment | Δ Val AUC (mean) | 95% CI | DeLong p | Holm Sig | Best HP |")
        L("|-----------|------------------|--------|---------|----------|---------|")

        for t_name, t_data in sorted(phase2.items()):
            summary = t_data if isinstance(t_data, dict) else {}
            # Try both flat and nested summary formats
            s = summary.get("summary", summary)

            delta = s.get("delta_val_auc_mean", s.get("delta_mean", 0))
            ci_lo = s.get("delta_val_auc_ci_low", s.get("ci_low", 0))
            ci_hi = s.get("delta_val_auc_ci_high", s.get("ci_high", 0))
            p_val = s.get("delong_p_value", s.get("p_value", 1.0))
            sig = s.get("significant_after_correction", s.get("significant", False))
            best_hp = s.get("best_hp_config", s.get("best_hp", "default"))

            sig_mark = "✓" if sig else "✗"
            hp_str = str(best_hp) if best_hp else "default"
            if len(hp_str) > 30:
                hp_str = hp_str[:27] + "..."

            L(f"| {t_name} | {delta:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | "
              f"{p_val:.4f} | {sig_mark} | {hp_str} |")

            summaries_for_forest.append({
                "treatment_id": s.get("treatment_id", 0),
                "treatment_name": t_name,
                "delta_val_auc_mean": delta,
                "delta_val_auc_ci_low": ci_lo,
                "delta_val_auc_ci_high": ci_hi,
                "delong_p_value": p_val,
                "significant_after_correction": sig,
            })

        # Generate forest plot if possible
        if summaries_for_forest:
            try:
                plot_data = create_forest_plot_data(summaries_for_forest)
                forest_path = output_dir / "forest_plot.png"
                generate_forest_plot_matplotlib(plot_data, forest_path)
                L("")
                L(f"![Forest Plot]({forest_path.name})")
            except Exception as e:
                L(f"")
                L(f"*Forest plot 생성 실패: {e}*")
    else:
        L("*Phase 2 결과 미발견*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 5. INTERACTION ANALYSIS (Phase 3)
    # ═══════════════════════════════════════════════════════════
    L("## 5. Interaction Analysis (Phase 3)")
    L("")

    if phase3 and isinstance(phase3, dict):
        # ── Pairwise Interactions ──
        pairwise = phase3.get("pairwise", [])
        if pairwise:
            L("### 5.1 Pairwise Interactions")
            L("")
            L("Interaction_{i,j} = Δ_{i+j} - (Δ_i + Δ_j)")
            L("")
            L("| Pair (i, j) | Δ_i | Δ_j | Δ_{i+j} | Interaction | Type |")
            L("|-------------|-----|-----|---------|-------------|------|")

            for pw in pairwise:
                t_i = pw.get("treatment_i", "?")
                t_j = pw.get("treatment_j", "?")
                d_i = pw.get("delta_i", 0)
                d_j = pw.get("delta_j", 0)
                d_ij = pw.get("delta_combined", 0)
                inter = pw.get("interaction", d_ij - d_i - d_j)
                itype = "Synergy" if inter > 0.001 else ("Redundancy" if inter < -0.001 else "Independent")
                L(f"| ({t_i}, {t_j}) | {d_i:+.4f} | {d_j:+.4f} | {d_ij:+.4f} | "
                  f"{inter:+.4f} | {itype} |")

            # Generate interaction heatmap
            try:
                names = sorted(set(
                    [p.get("treatment_i", "") for p in pairwise] +
                    [p.get("treatment_j", "") for p in pairwise]
                ))
                if len(names) > 1:
                    n = len(names)
                    mat = np.full((n, n), np.nan)
                    name_idx = {nm: i for i, nm in enumerate(names)}
                    for pw in pairwise:
                        i_idx = name_idx.get(pw.get("treatment_i", ""), -1)
                        j_idx = name_idx.get(pw.get("treatment_j", ""), -1)
                        if i_idx >= 0 and j_idx >= 0:
                            val = pw.get("interaction", 0)
                            mat[i_idx, j_idx] = val
                            mat[j_idx, i_idx] = val
                    heatmap_path = output_dir / "interaction_heatmap.png"
                    generate_interaction_heatmap(mat, names, heatmap_path)
                    L("")
                    L(f"![Interaction Heatmap]({heatmap_path.name})")
            except Exception as e:
                L(f"")
                L(f"*Interaction heatmap 생성 실패: {e}*")
            L("")

        # ── Cumulative Forward Selection ──
        cumulative = phase3.get("cumulative", [])
        if cumulative:
            L("### 5.2 Cumulative Forward Selection")
            L("")
            L("| Step | Added Treatment | AUC (mean ± std) | Marginal Δ |")
            L("|------|----------------|-------------------|------------|")

            step_names = []
            step_means = []
            step_stds = []

            for i, step in enumerate(cumulative):
                name = step.get("treatment_added", step.get("name", f"Step {i}"))
                auc_m = step.get("auc_mean", step.get("val_auc_mean", 0))
                auc_s = step.get("auc_std", step.get("val_auc_std", 0))
                marginal = step.get("marginal_delta", 0)

                step_names.append(name)
                step_means.append(auc_m)
                step_stds.append(auc_s)

                L(f"| {i + 1} | {name} | {auc_m:.4f} ± {auc_s:.4f} | {marginal:+.4f} |")

            # Generate cumulative curve
            if len(step_names) > 1:
                try:
                    curve_path = output_dir / "cumulative_curve.png"
                    generate_cumulative_curve(step_names, step_means, step_stds, curve_path)
                    L("")
                    L(f"![Cumulative Curve]({curve_path.name})")
                except Exception as e:
                    L(f"*Cumulative curve 생성 실패: {e}*")
    else:
        L("*Phase 3 결과 미발견*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 6. HP SENSITIVITY (Phase 4)
    # ═══════════════════════════════════════════════════════════
    L("## 6. Hyperparameter Sensitivity (Phase 4)")
    L("")

    if phase4 and isinstance(phase4, dict):
        L("| Treatment | HP | CV(AUC) | Robustness |")
        L("|-----------|----|---------:|------------|")

        for t_name, t_data in sorted(phase4.items()):
            cv = t_data.get("cv", t_data.get("sensitivity_cv", 0))
            robust = "Robust" if cv < 0.01 else ("Moderate" if cv < 0.03 else "Sensitive")
            hp_desc = t_data.get("hp_varied", "—")
            L(f"| {t_name} | {hp_desc} | {cv:.4f} | {robust} |")
    else:
        L("*Phase 4 결과 미발견 또는 미실행*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 7. FINAL MODEL (Phase 5)
    # ═══════════════════════════════════════════════════════════
    L("## 7. Final Model Performance (Phase 5)")
    L("")

    if phase5 and isinstance(phase5, dict):
        L("**⚠️ 이 결과는 단 1회 평가입니다 (one-shot test evaluation).**")
        L("")

        combo = phase5.get("treatment_combo", phase5.get("treatments_used", []))
        if combo:
            L(f"**선택된 Treatment 조합**: {', '.join(str(c) for c in combo)}")
            L("")

        met = phase5.get("metrics", {})

        for split_name in ["val", "test"]:
            split_met = met.get(split_name, {})
            if split_met:
                L(f"### {split_name.capitalize()} Set")
                L("")
                L("| Metric | Value |")
                L("|--------|------:|")
                for mk in ["auc", "ap", "f1", "precision", "recall", "acc"]:
                    mv = split_met.get(mk)
                    if isinstance(mv, (int, float)) and mv >= 0:
                        L(f"| {mk.upper()} | {mv:.4f} |")

                # ECE / Brier (if present)
                for mk in ["ece", "brier"]:
                    mv = split_met.get(mk)
                    if isinstance(mv, (int, float)):
                        L(f"| {mk.upper()} | {mv:.4f} |")
                L("")

        # Reliability diagram if ECE data exists
        ece_data = phase5.get("calibration_data")
        if ece_data:
            try:
                rel_path = output_dir / "reliability_diagram.png"
                generate_reliability_diagram(
                    bin_confidences=ece_data.get("bin_confidences", []),
                    bin_accuracies=ece_data.get("bin_accuracies", []),
                    bin_counts=ece_data.get("bin_counts", []),
                    output_path=rel_path,
                )
                L(f"![Reliability Diagram]({rel_path.name})")
            except Exception as e:
                L(f"*Reliability diagram 생성 실패: {e}*")
    else:
        L("*Phase 5 결과 미발견*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 8. QUALITATIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════
    L("## 8. Qualitative Analysis")
    L("")
    L("### 8.1 Learned Role-Adjacency Matrix R ∈ ℝ^{5×5}")
    L("")
    L("*Role-Aware Adjacency (T5)가 포함된 경우, 학습된 R 행렬을 분석.*")
    L("*checkpoint.pt에서 RoleAwareAdjacency.R_raw를 추출하여 시각화.*")

    # Try to generate role adjacency heatmap from phase5 data
    role_data = None
    if phase5:
        role_data = phase5.get("role_adjacency_matrix")
    if role_data is not None:
        try:
            role_mat = np.array(role_data)
            role_path = output_dir / "role_adjacency.png"
            role_names = ["TOP", "JGL", "MID", "ADC", "SUP"]
            generate_role_adjacency_heatmap(role_mat, role_names, role_path)
            L(f"")
            L(f"![Role Adjacency]({role_path.name})")
        except Exception as e:
            L(f"*Role adjacency heatmap 생성 실패: {e}*")
    L("")

    L("### 8.2 Temporal Attention Patterns")
    L("")
    L("*Attention Temporal Pooling (T3)가 포함된 경우, α_t 분포를 분석.*")
    L("*교전 시작 시점 대비 α_t의 peak position으로 '모델이 어느 시점에 집중하는가' 파악.*")
    L("")

    # ═══════════════════════════════════════════════════════════
    # § 9. KEY FINDINGS
    # ═══════════════════════════════════════════════════════════
    L("## 9. Key Findings & Recommendations")
    L("")

    # Auto-generate findings from phase 2 significant results
    if phase2 and isinstance(phase2, dict):
        sig_treatments = []
        nonsig_treatments = []
        for t_name, t_data in phase2.items():
            s = t_data.get("summary", t_data) if isinstance(t_data, dict) else {}
            if s.get("significant_after_correction", s.get("significant", False)):
                delta = s.get("delta_val_auc_mean", s.get("delta_mean", 0))
                sig_treatments.append((t_name, delta))
            else:
                nonsig_treatments.append(t_name)

        if sig_treatments:
            sig_treatments.sort(key=lambda x: -abs(x[1]))
            L("**통계적으로 유의한 개선**:")
            L("")
            for t_name, delta in sig_treatments:
                direction = "향상" if delta > 0 else "하락"
                L(f"- **{t_name}**: Δ = {delta:+.4f} ({direction}, Holm-Bonferroni corrected p < 0.05)")
            L("")

        if nonsig_treatments:
            L(f"**유의하지 않은 Treatment**: {', '.join(nonsig_treatments)}")
            L("")

    L("---")
    L("*이 보고서는 experiment_runner.py의 실험 결과로부터 자동 생성되었습니다.*")

    # ─── Save report ───
    report_text = "\n".join(lines)

    report_path = output_dir / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save as LaTeX table if phase2 data exists
    if summaries_for_forest:
        try:
            latex_path = output_dir / "table2_ablation.tex"
            generate_latex_table(summaries_for_forest, latex_path)
        except Exception:
            pass

    print(f"  [SAVED] Report: {report_path}")
    return report_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation study analysis and visualization")
    parser.add_argument("--results-dir", type=str, default="./ablation_results")
    parser.add_argument("--output-dir", type=str, default="./figures")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    generate_full_report(results_dir, output_dir)