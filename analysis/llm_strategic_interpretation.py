"""LLM-Augmented Strategic Interpretation.

Sends Phase-stratified SHAP results to Claude for strategic analysis
of feature importance transitions across game phases.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_interpretation_prompt(
    phase_shap_results: Dict[str, Any],
    phase_aucs: Dict[str, float],
) -> str:
    """Convert Phase-SHAP results into an LLM prompt."""

    transition_table = phase_shap_results.get("transition_table", [])

    table_str = "Feature | Early Rank | Early |SHAP| | Mid Rank | Mid |SHAP| | Late Rank | Late |SHAP|\n"
    table_str += "---|---|---|---|---|---|---\n"
    for row in transition_table:
        table_str += (
            f"{row['feature']} | "
            f"{row.get('early_rank', 'N/A')} | {row.get('early_shap', 'N/A')} | "
            f"{row.get('mid_rank', 'N/A')} | {row.get('mid_shap', 'N/A')} | "
            f"{row.get('late_rank', 'N/A')} | {row.get('late_shap', 'N/A')}\n"
        )

    prompt = f"""You are a League of Legends esports analyst with deep knowledge of Korean Master+ tier meta-strategy (patches 15.14-15.16). You are reviewing the output of a gradient-boosted tree model (LightGBM, AUC 0.675) that predicts teamfight outcomes from pre-fight game state.

The model's feature attribution (SHAP) was computed separately for three game phases on held-out test data (~100K engagements from Korean Master+ ranked matches).

## Phase Definitions
- Early: <=14 minutes (laning phase, first objectives like Voidgrubs/Horde)
- Mid: 15-25 minutes (1-2 item power spikes, dragon stacking, Rift Herald)
- Late: >=26 minutes (3+ items, Baron contests, team composition scaling)

## Phase-Specific Model Performance
- Early AUC: {phase_aucs.get('early', 'N/A')}
- Mid AUC: {phase_aucs.get('mid', 'N/A')}
- Late AUC: {phase_aucs.get('late', 'N/A')}

## Feature Importance Transition Table
{table_str}

## Feature Naming Convention
- Prefix: Blue/Red team, role (TOP/JNG/MID/BOT/SUP)
- Suffix: temporal statistic (last=most recent, mean=average, std=variability, max=peak, delta=change, slope=trend)
- CDR = cooldown reduction, XP = experience, g/s = gold per second
- "obj prox" = proximity to objective (Horde=Voidgrubs, Dragon, Baron)
- "(n.)" = normalized value

## Analysis Request

Provide a strategic analysis in the following structure:

### 1. Phase Transition Narrative
Why do the top features shift across phases? Connect each major shift to known competitive LoL strategy. Be specific about champion classes, item breakpoints, and objective timers.

### 2. Domain Validation
Do these learned importances align with how Korean Master+ coaches evaluate teamfight readiness at each phase? Where do they agree or diverge from conventional wisdom?

### 3. Predictability Gradient
The model finds late-game fights much easier to predict than early-game fights (AUC difference ~0.19). Which feature transitions explain this? What information is available late but not early?

### 4. Coaching Implications
Based on this analysis, what 3 specific actionable recommendations would you give to a coaching staff preparing for Korean Master+ ranked play?

### 5. Model Limitations by Phase
What strategic factors at each phase might the model be missing, given it only sees 60-second state snapshots and millisecond event counts (no replay video, no ability-level cooldowns, no mechanical execution data)?

Be concrete. Reference specific champions, items, and objectives where relevant."""

    return prompt


def run_llm_interpretation(
    phase_shap_path: Path,
    phase_aucs: Dict[str, float],
    output_dir: Path,
    api_key: Optional[str] = None,
) -> str:
    """Run LLM interpretation of phase-stratified SHAP results.

    Parameters
    ----------
    phase_shap_path : path to phase_shap_results.json
    phase_aucs : dict with 'early', 'mid', 'late' AUC values
    output_dir : where to save interpretation
    api_key : Anthropic API key (reads ANTHROPIC_API_KEY env var if None)

    Returns
    -------
    interpretation text
    """
    import anthropic

    with open(phase_shap_path) as f:
        phase_shap_results = json.load(f)

    prompt = build_interpretation_prompt(phase_shap_results, phase_aucs)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    interpretation = response.content[0].text

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "llm_interpretation.md", "w") as f:
        f.write(interpretation)
    with open(output_dir / "llm_prompt.txt", "w") as f:
        f.write(prompt)

    logger.info("[LLM] Interpretation saved to %s", output_dir / "llm_interpretation.md")
    return interpretation
