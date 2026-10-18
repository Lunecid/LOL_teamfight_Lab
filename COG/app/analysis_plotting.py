from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def create_forest_plot_data(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build plotting arrays for forest-plot visualization."""
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
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping forest plot")
        return

    n = len(plot_data["treatments"])
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.8)))
    y_positions = list(range(n))

    for i in range(n):
        color = "#2196F3" if plot_data["significant"][i] else "#9E9E9E"
        fillstyle = "full" if plot_data["significant"][i] else "none"

        ax.plot(
            [plot_data["ci_lows"][i], plot_data["ci_highs"][i]],
            [y_positions[i], y_positions[i]],
            color=color,
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            plot_data["deltas"][i],
            y_positions[i],
            marker="o",
            color=color,
            markersize=10,
            fillstyle=fillstyle,
            markeredgewidth=2,
            zorder=3,
        )

        p = plot_data["p_values"][i]
        sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.annotate(
            f"  Δ={plot_data['deltas'][i]:+.4f} ({sig_text})",
            xy=(plot_data["ci_highs"][i], y_positions[i]),
            fontsize=9,
            va="center",
        )

    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    xlim = ax.get_xlim()
    ax.axvspan(0, xlim[1], alpha=0.05, color="green", zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["treatments"])
    ax.set_xlabel("Δ Val AUC (vs Baseline)")
    ax.set_title(title)
    ax.invert_yaxis()

    sig_patch = mpatches.Patch(color="#2196F3", label="Significant (p<0.05, Holm-Bonferroni)")
    ns_patch = mpatches.Patch(color="#9E9E9E", label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Forest plot: {output_path}")


def generate_interaction_heatmap(
    interaction_matrix: np.ndarray,
    treatment_names: List[str],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available")
        return

    n = len(treatment_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(interaction_matrix.min()), abs(interaction_matrix.max()))
    vmax = max(vmax, 0.001)

    im = ax.imshow(interaction_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

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


def generate_cumulative_curve(
    steps: List[str],
    auc_means: List[float],
    auc_stds: List[float],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(steps))
    ax1.errorbar(
        x,
        auc_means,
        yerr=auc_stds,
        marker="o",
        capsize=4,
        color="#2196F3",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        x,
        [m - s for m, s in zip(auc_means, auc_stds)],
        [m + s for m, s in zip(auc_means, auc_stds)],
        alpha=0.15,
        color="#2196F3",
    )
    ax1.axhline(y=auc_means[0], color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(steps, rotation=45, ha="right")
    ax1.set_ylabel("Val AUC")
    ax1.set_title("Cumulative Addition of Improvements")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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


def _calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n = int(mask.sum())
        if n > 0:
            bin_centers.append(float(y_prob[mask].mean()))
            bin_accs.append(float(y_true[mask].mean()))
            bin_counts.append(n)

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_counts)


def generate_reliability_diagram(
    y_true: Optional[np.ndarray] = None,
    y_prob_baseline: Optional[np.ndarray] = None,
    y_prob_improved: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    n_bins: int = 15,
    *,
    bin_confidences: Optional[Sequence[float]] = None,
    bin_accuracies: Optional[Sequence[float]] = None,
    bin_counts: Optional[Sequence[int]] = None,
) -> None:
    """Render reliability diagram from raw probs or pre-binned calibration data."""
    if output_path is None:
        raise ValueError("output_path is required")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    has_binned = bin_confidences is not None and bin_accuracies is not None
    if has_binned:
        centers = np.asarray(list(bin_confidences), dtype=float)
        accs = np.asarray(list(bin_accuracies), dtype=float)
        counts = np.asarray(list(bin_counts) if bin_counts is not None else np.ones_like(centers), dtype=float)

        if centers.size and accs.size:
            ax1.plot(centers, accs, "o-", color="#2196F3", markersize=6, label="Model")
        ax2.bar(np.arange(len(counts)), counts, color="#2196F3", alpha=0.7)
        ax2.set_xlabel("Calibration Bin")
        ax2.set_ylabel("Count")
        ax2.set_title("Samples per Bin")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        if y_true is None or y_prob_baseline is None or y_prob_improved is None:
            raise ValueError(
                "Either provide (y_true, y_prob_baseline, y_prob_improved) or "
                "(bin_confidences, bin_accuracies)."
            )

        centers_b, accs_b, _ = _calibration_curve(y_true, y_prob_baseline, n_bins)
        centers_i, accs_i, _ = _calibration_curve(y_true, y_prob_improved, n_bins)

        ax1.plot(centers_b, accs_b, "o-", color="#F44336", markersize=6, label="Baseline")
        ax1.plot(centers_i, accs_i, "s-", color="#2196F3", markersize=6, label="Improved")

        ax2.hist(y_prob_baseline, bins=50, alpha=0.5, color="#F44336", label="Baseline", density=True)
        ax2.hist(y_prob_improved, bins=50, alpha=0.5, color="#2196F3", label="Improved", density=True)
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of Predictions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Reliability diagram: {output_path}")


def generate_role_adjacency_heatmap(
    R_matrix: np.ndarray,
    output_path: Path,
    roles: Optional[List[str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    roles = roles or ["TOP", "JNG", "MID", "BOT", "SUP"]
    if R_matrix.shape[0] != R_matrix.shape[1]:
        raise ValueError("R_matrix must be square")

    n = R_matrix.shape[0]
    if len(roles) != n:
        roles = [f"R{i}" for i in range(n)]

    R_softplus = np.log1p(np.exp(R_matrix))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(R_matrix, cmap="RdBu_r", aspect="auto")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{R_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(roles)
    ax1.set_yticklabels(roles)
    ax1.set_title("Learned R (raw)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(R_softplus, cmap="YlOrRd", aspect="auto")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{R_softplus[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(roles)
    ax2.set_yticklabels(roles)
    ax2.set_title("softplus(R) = Adjacency Multiplier")
    plt.colorbar(im2, ax=ax2)

    plt.suptitle("Role-Aware Adjacency: Learned Interaction Strength", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Role adjacency heatmap: {output_path}")

