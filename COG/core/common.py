"""common.py — Shared utility functions.

Changes from original:
  [FIX-IMPORT] Added __all__ to prevent namespace pollution via wildcard imports.
  [FIX-IMPORT] Added commonly-needed re-exports (dataclass, Dict, etc.) so that
               downstream modules that DO still use `from .common import *` get a
               controlled set of symbols instead of everything in this module's
               global scope.
  [FIX-MATH]   sigmoid_np: added numerical stability clamp to prevent overflow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------
# Controlled re-exports  (downstream `from .common import *` only
# gets the names listed in __all__)
# ------------------------------------------------------------------
__all__ = [
    # stdlib re-exports
    "math", "dataclass", "field",
    # typing re-exports
    "Any", "Dict", "List", "Optional", "Sequence", "Tuple",
    # numpy
    "np",
    # utility functions
    "parse_csv_nums", "parse_csv_str",
    "percentile_safe", "sigmoid_np", "logit",
    "safe_float", "log1p_norm",
]


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_csv_nums(s: str, cast=float) -> List:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(cast(p))
        except (ValueError, TypeError):
            continue
    return out


def parse_csv_str(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]


def percentile_safe(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    try:
        return float(np.percentile(x, q))
    except (IndexError, ValueError) as e:
        return float(np.sort(x)[int(np.clip(round((q / 100.0) * (len(x) - 1)), 0, len(x) - 1))])


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clips input to [-500, 500] to avoid overflow."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -500.0, 500.0)  # [FIX-MATH] prevent exp overflow
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def safe_float(v, default: float = 0.0) -> float:
    """Robust float conversion with fallback.

    Previously lived in an external ``utils`` module; moved here so that
    feature extraction does not depend on an unversioned helper.
    """
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def log1p_norm(x: float, denom: float = 1.0) -> float:
    """log(1+x) / log(1+denom) — compresses large positive values into ~[0,1].

    Previously lived in an external ``utils`` module.

    Mathematical note
    -----------------
    .. math::
        \\hat{x} = \\frac{\\ln(1 + |x|)}{\\ln(1 + D)}

    where *D* is the denominator representing the expected maximum.
    The sign of *x* is preserved so negative values map to (-1, 0).
    """
    denom = max(abs(denom), 1e-8)
    sign = 1.0 if x >= 0 else -1.0
    return sign * math.log1p(abs(x)) / math.log1p(denom)
