from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.analysis_plotting import (
    create_forest_plot_data,
    generate_cumulative_curve,
    generate_forest_plot_matplotlib,
    generate_interaction_heatmap,
    generate_reliability_diagram,
    generate_role_adjacency_heatmap,
)


def generate_latex_table(
    summaries: List[Dict[str, Any]],
    baseline_summary: Dict[str, Any],
) -> str:
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

    b = baseline_summary
    lines.append(
        f"Baseline & ${b.get('val_auc_mean', 0):.4f} \\pm {b.get('val_auc_std', 0):.4f}$ "
        f"& --- & --- & --- & --- \\\\"
    )
    lines.append(r"\midrule")

    sorted_s = sorted(summaries, key=lambda x: -x.get("delta_val_auc_mean", 0))
    for s in sorted_s:
        sig = "\\checkmark" if s.get("significant_after_correction", False) else "---"
        p = s.get("delong_p_value", 1.0)
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"$\\mathcal{{T}}_{{{s['treatment_id']}}}$ {s['treatment_name']} "
            f"& ${s.get('val_auc_mean', 0):.4f} \\pm {s.get('val_auc_std', 0):.4f}$ "
            f"& ${s.get('delta_val_auc_mean', 0):+.4f}$ "
            f"& $[{s.get('delta_val_auc_ci_low', 0):+.4f}, {s.get('delta_val_auc_ci_high', 0):+.4f}]$ "
            f"& {p_str} "
            f"& {sig} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _extract_baseline_summary(phase1: Any) -> Dict[str, float]:
    if not isinstance(phase1, list):
        return {"val_auc_mean": 0.0, "val_auc_std": 0.0}
    base_aucs: List[float] = []
    for r in phase1:
        if not isinstance(r, dict):
            continue
        va = r.get("val_auc", r.get("metrics", {}).get("val", {}).get("auc", -1))
        if isinstance(va, (int, float)) and va > 0:
            base_aucs.append(float(va))
    if not base_aucs:
        return {"val_auc_mean": 0.0, "val_auc_std": 0.0}
    return {
        "val_auc_mean": float(np.mean(base_aucs)),
        "val_auc_std": float(np.std(base_aucs, ddof=1)) if len(base_aucs) > 1 else 0.0,
    }


def generate_full_report(results_dir: Path, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(name: str) -> Optional[Any]:
        fp = results_dir / name
        if not fp.exists():
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load {name}: {e}")
            return None

    phase1 = _load_json("phase1_baseline.json")
    phase2 = _load_json("phase2_single_factor.json")
    phase3 = _load_json("phase3_interaction.json")
    phase4 = _load_json("phase4_sensitivity.json")
    phase5 = _load_json("phase5_final.json")

    baseline_summary = _extract_baseline_summary(phase1)

    lines: List[str] = []
    L = lines.append
    L("# Ablation Study Report: LoL Teamfight Prediction Improvements")
    L("")
    L("**Generated**: auto from experiment results")
    L(f"**Results directory**: `{results_dir}`")
    L("")

    L("## 1. Executive Summary")
    L("")
    if phase1 and phase5:
        final_met = phase5.get("metrics", {})
        final_test = final_met.get("test", {})
        final_val = final_met.get("val", {})
        L("| Metric | Baseline (μ ± σ) | Final Model | Δ |")
        L("|--------|-------------------|-------------|---|")

        base_mean = baseline_summary.get("val_auc_mean", 0.0)
        base_std = baseline_summary.get("val_auc_std", 0.0)
        if base_mean > 0:
            if isinstance(final_val.get("auc"), (int, float)):
                val_auc = float(final_val["auc"])
                L(f"| Val AUC | {base_mean:.4f} ± {base_std:.4f} | {val_auc:.4f} | {val_auc - base_mean:+.4f} |")
            else:
                L(f"| Val AUC | {base_mean:.4f} ± {base_std:.4f} | N/A | N/A |")

        for key, label in (("auc", "Test AUC"), ("f1", "Test F1"), ("ap", "Test AP")):
            val = final_test.get(key, "N/A")
            if isinstance(val, (int, float)):
                L(f"| {label} | — | {float(val):.4f} | — |")
        for key, label in (("ece", "ECE"), ("brier", "Brier")):
            val = final_test.get(key, final_met.get(key))
            if isinstance(val, (int, float)):
                L(f"| {label} | — | {float(val):.4f} | — |")
    else:
        L("*결과 파일 미발견 — 실험 완료 후 자동 생성됩니다.*")
    L("")

    L("## 2. Experimental Setup")
    L("")
    L("- **Data**: Korean LoL matches, Patches 15.14–15.17")
    L("- **Split**: Temporal forward validation (Train: 15.14–15.15, Val: 15.16, Test: 15.17)")
    L("- **Seeds**: 5 random seeds (7, 42, 123, 256, 512)")
    L("- **Statistical Tests**: DeLong (AUC), McNemar (classification), Holm–Bonferroni correction")
    L("- **Significance level**: α = 0.05 (two-sided)")
    L("")

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
        L("")
        L(
            f"**Summary**: μ = {baseline_summary['val_auc_mean']:.4f}, "
            f"σ = {baseline_summary['val_auc_std']:.4f}"
        )
    else:
        L("*Phase 1 결과 미발견*")
    L("")

    L("## 4. Single-Factor Ablation Results (Phase 2)")
    L("")
    summaries_for_forest: List[Dict[str, Any]] = []
    if phase2 and isinstance(phase2, dict):
        L("| Treatment | Δ Val AUC (mean) | 95% CI | DeLong p | Holm Sig | Best HP |")
        L("|-----------|------------------|--------|---------|----------|---------|")

        for t_name, t_data in sorted(phase2.items()):
            summary = t_data if isinstance(t_data, dict) else {}
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

            L(
                f"| {t_name} | {delta:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | "
                f"{p_val:.4f} | {sig_mark} | {hp_str} |"
            )
            summaries_for_forest.append(
                {
                    "treatment_id": s.get("treatment_id", 0),
                    "treatment_name": t_name,
                    "val_auc_mean": s.get("val_auc_mean", 0),
                    "val_auc_std": s.get("val_auc_std", 0),
                    "delta_val_auc_mean": delta,
                    "delta_val_auc_ci_low": ci_lo,
                    "delta_val_auc_ci_high": ci_hi,
                    "delong_p_value": p_val,
                    "significant_after_correction": sig,
                }
            )

        if summaries_for_forest:
            try:
                plot_data = create_forest_plot_data(summaries_for_forest)
                forest_path = output_dir / "forest_plot.png"
                generate_forest_plot_matplotlib(plot_data, forest_path)
                L("")
                L(f"![Forest Plot]({forest_path.name})")
            except Exception as e:
                L("")
                L(f"*Forest plot 생성 실패: {e}*")
    else:
        L("*Phase 2 결과 미발견*")
    L("")

    L("## 5. Interaction Analysis (Phase 3)")
    L("")
    if phase3 and isinstance(phase3, dict):
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
                L(
                    f"| ({t_i}, {t_j}) | {d_i:+.4f} | {d_j:+.4f} | {d_ij:+.4f} | "
                    f"{inter:+.4f} | {itype} |"
                )

            try:
                names = sorted(
                    set([p.get("treatment_i", "") for p in pairwise] + [p.get("treatment_j", "") for p in pairwise])
                )
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
                L("")
                L(f"*Interaction heatmap 생성 실패: {e}*")
            L("")

        cumulative = phase3.get("cumulative", [])
        if cumulative:
            L("### 5.2 Cumulative Forward Selection")
            L("")
            L("| Step | Added Treatment | AUC (mean ± std) | Marginal Δ |")
            L("|------|----------------|-------------------|------------|")

            step_names: List[str] = []
            step_means: List[float] = []
            step_stds: List[float] = []
            for i, step in enumerate(cumulative):
                name = step.get("treatment_added", step.get("name", f"Step {i}"))
                auc_m = step.get("auc_mean", step.get("val_auc_mean", 0))
                auc_s = step.get("auc_std", step.get("val_auc_std", 0))
                marginal = step.get("marginal_delta", 0)
                step_names.append(name)
                step_means.append(auc_m)
                step_stds.append(auc_s)
                L(f"| {i + 1} | {name} | {auc_m:.4f} ± {auc_s:.4f} | {marginal:+.4f} |")

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
                for mk in ["ece", "brier"]:
                    mv = split_met.get(mk)
                    if isinstance(mv, (int, float)):
                        L(f"| {mk.upper()} | {mv:.4f} |")
                L("")

        ece_data = phase5.get("calibration_data")
        if isinstance(ece_data, dict):
            try:
                rel_path = output_dir / "reliability_diagram.png"
                generate_reliability_diagram(
                    output_path=rel_path,
                    bin_confidences=ece_data.get("bin_confidences", []),
                    bin_accuracies=ece_data.get("bin_accuracies", []),
                    bin_counts=ece_data.get("bin_counts", []),
                )
                L(f"![Reliability Diagram]({rel_path.name})")
            except Exception as e:
                L(f"*Reliability diagram 생성 실패: {e}*")
    else:
        L("*Phase 5 결과 미발견*")
    L("")

    L("## 8. Qualitative Analysis")
    L("")
    L("### 8.1 Learned Role-Adjacency Matrix R ∈ ℝ^{5×5}")
    L("")
    L("*Role-Aware Adjacency (T5)가 포함된 경우, 학습된 R 행렬을 분석.*")
    L("*checkpoint.pt에서 RoleAwareAdjacency.R_raw를 추출하여 시각화.*")

    role_data = phase5.get("role_adjacency_matrix") if isinstance(phase5, dict) else None
    if role_data is not None:
        try:
            role_mat = np.array(role_data)
            role_path = output_dir / "role_adjacency.png"
            role_names = ["TOP", "JGL", "MID", "ADC", "SUP"]
            generate_role_adjacency_heatmap(role_mat, role_path, roles=role_names)
            L("")
            L(f"![Role Adjacency]({role_path.name})")
        except Exception as e:
            L(f"*Role adjacency heatmap 생성 실패: {e}*")
    L("")

    L("### 8.2 Temporal Attention Patterns")
    L("")
    L("*Attention Temporal Pooling (T3)가 포함된 경우, α_t 분포를 분석.*")
    L("*교전 시작 시점 대비 α_t의 peak position으로 '모델이 어느 시점에 집중하는가' 파악.*")
    L("")

    L("## 9. Key Findings & Recommendations")
    L("")
    if phase2 and isinstance(phase2, dict):
        sig_treatments: List[tuple[str, float]] = []
        nonsig_treatments: List[str] = []
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

    report_text = "\n".join(lines)
    report_path = output_dir / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if summaries_for_forest:
        try:
            latex_path = output_dir / "table2_ablation.tex"
            latex_text = generate_latex_table(summaries_for_forest, baseline_summary)
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex_text)
            print(f"  [SAVED] Table: {latex_path}")
        except Exception as e:
            print(f"  [WARN] Failed to save LaTeX table: {e}")

    print(f"  [SAVED] Report: {report_path}")
    return report_text

