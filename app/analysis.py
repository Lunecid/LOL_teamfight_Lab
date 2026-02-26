"""Ablation study analysis helpers and report entrypoint."""

from __future__ import annotations

from pathlib import Path

from app.analysis_metrics import difficulty_stratified_auc, phase_stratified_auc
from app.analysis_plotting import (
    create_forest_plot_data,
    generate_cumulative_curve,
    generate_forest_plot_matplotlib,
    generate_interaction_heatmap,
    generate_reliability_diagram,
    generate_role_adjacency_heatmap,
)
from app.analysis_reporting import generate_full_report, generate_latex_table

__all__ = [
    "create_forest_plot_data",
    "generate_forest_plot_matplotlib",
    "generate_interaction_heatmap",
    "generate_cumulative_curve",
    "generate_reliability_diagram",
    "generate_role_adjacency_heatmap",
    "generate_latex_table",
    "phase_stratified_auc",
    "difficulty_stratified_auc",
    "generate_full_report",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation study analysis and visualization")
    parser.add_argument("--results-dir", type=str, default="./ablation_results")
    parser.add_argument("--output-dir", type=str, default="./figures")
    args = parser.parse_args()

    generate_full_report(Path(args.results_dir), Path(args.output_dir))
