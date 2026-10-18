from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def save_results_json(path: Path, results: Any) -> None:
    """Save experiment results as JSON with dataclass/numpy sanitization."""
    try:
        if isinstance(results, list):
            data = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results]
        elif isinstance(results, dict):
            data = {}
            for k, v in results.items():
                k_str = str(k)
                if isinstance(v, list):
                    data[k_str] = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in v]
                else:
                    data[k_str] = v
        else:
            data = results

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if k not in ("pred_logits_val", "pred_logits_test")}
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        data = _clean(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] {path}")
    except Exception as e:
        print(f"  [WARN] Failed to save results: {e}")


def load_results_json(path: Path) -> Any:
    """Load JSON results file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
