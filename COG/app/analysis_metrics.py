from __future__ import annotations

from typing import Dict

import numpy as np


def phase_stratified_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fight_minutes: np.ndarray,
) -> Dict[str, float]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    phases = {
        "early": (0, 14),
        "mid": (14, 28),
        "late": (28, float("inf")),
    }

    results: Dict[str, float] = {}
    for phase_name, (t_low, t_high) in phases.items():
        mask = (fight_minutes >= t_low) & (fight_minutes < t_high)
        n_phase = int(mask.sum())

        if n_phase < 30:
            results[f"AUC_{phase_name}"] = float("nan")
            results[f"n_{phase_name}"] = n_phase
            continue

        y_phase = y_true[mask]
        p_phase = y_prob[mask]

        if len(np.unique(y_phase)) < 2:
            results[f"AUC_{phase_name}"] = float("nan")
        else:
            results[f"AUC_{phase_name}"] = float(roc_auc_score(y_phase, p_phase))
        results[f"n_{phase_name}"] = n_phase

    return results


def difficulty_stratified_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    gold_diff: np.ndarray,
) -> Dict[str, float]:
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

    results: Dict[str, float] = {}
    for name, mask in strata.items():
        n = int(mask.sum())
        if n < 30 or len(np.unique(y_true[mask])) < 2:
            results[f"AUC_{name}"] = float("nan")
        else:
            results[f"AUC_{name}"] = float(roc_auc_score(y_true[mask], y_prob[mask]))
        results[f"n_{name}"] = n

    return results

