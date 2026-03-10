"""LLM-Augmented Strategic Interpretation.

Sends Phase-stratified SHAP results to Claude for strategic analysis
of feature importance transitions across game phases.

Usage:
    python -m analysis.llm_strategic_interpretation \
        --shap_results outputs/phase_shap/phase_shap_results.json \
        --output_dir outputs/phase_shap
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


def build_prompt(phase_shap: Dict[str, Any], phase_aucs: Dict[str, float]) -> str:
    """Convert Phase-SHAP results into an LLM prompt."""
    table = phase_shap.get("transition_table", [])

    tbl = "Feature | Early Rank | Early |SHAP| | Mid Rank | Mid |SHAP| | Late Rank | Late |SHAP|\n"
    tbl += "---|---|---|---|---|---|---\n"
    for r in table:
        tbl += (
            f"{r['feature']} | "
            f"{r.get('early_rank', 'N/A')} | {r.get('early_shap', 'N/A')} | "
            f"{r.get('mid_rank', 'N/A')} | {r.get('mid_shap', 'N/A')} | "
            f"{r.get('late_rank', 'N/A')} | {r.get('late_shap', 'N/A')}\n"
        )

    return f"""You are a League of Legends esports analyst with deep knowledge of Korean Master+ tier meta-strategy (patches 15.14-15.16). You are reviewing a gradient-boosted tree model (LightGBM, AUC 0.675) that predicts teamfight outcomes from pre-fight game state.

SHAP attribution was computed separately for three game phases on ~100K held-out engagements.

## Phase Definitions
- Early: <=14 min (laning, Voidgrubs/Horde)
- Mid: 15-25 min (1-2 item spikes, dragon stacking)
- Late: >=26 min (3+ items, Baron, team composition scaling)

## Phase AUC
- Early: {phase_aucs.get('early', 'N/A')}
- Mid: {phase_aucs.get('mid', 'N/A')}
- Late: {phase_aucs.get('late', 'N/A')}

## Feature Importance Transition Table
{tbl}

## Feature Naming Convention
- Prefix: Blue/Red, role (TOP/JNG/MID/BOT/SUP)
- Suffix: temporal statistic (last, mean, std, max, delta, slope)
- CDR=cooldown reduction, XP=experience, g/s=gold per second
- "obj prox"=proximity to objective, "(n.)"=normalized

Provide strategic analysis:

### 1. Phase Transition Narrative
Why do top features shift across phases? Connect to competitive LoL strategy.

### 2. Domain Validation
Do these importances align with how Korean Master+ coaches evaluate teamfight readiness?

### 3. Predictability Gradient
Why is late-game AUC ~0.19 higher than early? Which feature transitions explain this?

### 4. Coaching Implications
3 specific actionable recommendations for Korean Master+ ranked play.

### 5. Model Limitations by Phase
What strategic factors might the model miss at each phase, given 60-second snapshot resolution?

Be concrete - reference specific champions, items, and objectives."""


def run_interpretation(
    shap_path: Path,
    phase_aucs: Dict[str, float],
    output_dir: Path,
    api_key: Optional[str] = None,
) -> str:
    """Run LLM interpretation of phase-stratified SHAP results.

    Parameters
    ----------
    shap_path : path to phase_shap_results.json
    phase_aucs : dict with 'early', 'mid', 'late' AUC values
    output_dir : where to save interpretation
    api_key : Anthropic API key (reads ANTHROPIC_API_KEY env var if None)

    Returns
    -------
    interpretation text
    """
    import anthropic

    with open(shap_path) as f:
        shap_data = json.load(f)

    prompt = build_prompt(shap_data, phase_aucs)

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_interpretation.md").write_text(text, encoding="utf-8")
    (output_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")
    logger.info("[LLM] Saved to %s", output_dir / "llm_interpretation.md")
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="LLM Strategic Interpretation of Phase-SHAP")
    ap.add_argument("--shap_results", type=str, required=True,
                     help="Path to phase_shap_results.json")
    ap.add_argument("--output_dir", type=str, default="outputs/phase_shap")
    ap.add_argument("--early_auc", type=float, default=0.617)
    ap.add_argument("--mid_auc", type=float, default=0.713)
    ap.add_argument("--late_auc", type=float, default=0.807)
    args = ap.parse_args()

    run_interpretation(
        Path(args.shap_results),
        {"early": args.early_auc, "mid": args.mid_auc, "late": args.late_auc},
        Path(args.output_dir),
    )
