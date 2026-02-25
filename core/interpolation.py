"""interpolation.py — Pluggable alpha-curve functions for temporal interpolation.

Given a normalised time fraction  t in [0, 1]  between two anchor frames,
each curve returns an  alpha in [0, 1]  that controls the blend:

    result = (1 - alpha) * value_A + alpha * value_B

Supported curves
----------------
linear       straight-line lerp
cosine       smooth ease-in/out  (good for idle->move->arrive)
exponential  1 - e^(-k*t), accelerates toward the target position
cubic        cubic Hermite ease-in-out with zero endpoint tangents
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np


# ── curve functions ──────────────────────────────────────────────
def alpha_linear(t: float, **_kw: object) -> float:
    """Straight-line interpolation: alpha = t."""
    return t


def alpha_cosine(t: float, **_kw: object) -> float:
    """Cosine ease-in-out: smooth acceleration/deceleration.

    alpha = (1 - cos(pi * t)) / 2
    """
    return (1.0 - math.cos(math.pi * t)) * 0.5


def alpha_exponential(t: float, *, k: float = 3.0, **_kw: object) -> float:
    """Exponential decay toward target: alpha = 1 - e^(-k*t).

    Higher k  -> faster convergence.
    k=3  key points:
      t=0.0 -> alpha=0.00   (at frame time: use frame position)
      t=0.3 -> alpha=0.59   (30% through: already 59% toward target)
      t=0.5 -> alpha=0.78   (halfway: 78% toward target)
      t=1.0 -> alpha=0.95   (at target time: essentially at target)
    """
    return 1.0 - math.exp(-k * t)


def alpha_cubic(t: float, **_kw: object) -> float:
    """Cubic Hermite ease-in-out with zero endpoint tangents.

    alpha = 3t^2 - 2t^3   (smoothstep)

    Provides a smooth S-curve between the two anchor points without
    requiring additional control points.
    """
    return t * t * (3.0 - 2.0 * t)


# ── registry ─────────────────────────────────────────────────────
ALPHA_CURVES: Dict[str, Callable[..., float]] = {
    "linear": alpha_linear,
    "cosine": alpha_cosine,
    "exponential": alpha_exponential,
    "exp": alpha_exponential,
    "cubic": alpha_cubic,
    "smoothstep": alpha_cubic,
}


def remap_alpha(t: float, curve: str = "linear", **kw: object) -> float:
    """Map a raw linear fraction *t* through the named curve.

    Parameters
    ----------
    t : float
        Raw linear fraction in [0, 1].
    curve : str
        Name of the curve (key in ``ALPHA_CURVES``).  Falls back to linear
        for unrecognised names.
    **kw
        Extra keyword arguments forwarded to the curve function
        (e.g. ``k`` for exponential).

    Returns
    -------
    float
        Remapped alpha in [0, 1].
    """
    fn = ALPHA_CURVES.get(curve.lower().strip(), alpha_linear)
    return float(np.clip(fn(t, **kw), 0.0, 1.0))
