"""common_torch.py — Shared PyTorch utility functions.

[P4-DEDUP] Extracted from deep.py / models.py to eliminate code duplication.
Both modules previously defined identical copies of:
  - _autocast_disabled()
  - _nan_to_num_()
  - _resolve_node_idx()
  - X_IDX / Y_IDX / ALIVE_IDX resolution

This module is the **single source of truth** for these utilities.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =========================================================
# AMP safety helpers
# =========================================================
@contextmanager
def autocast_disabled():
    """Force fp32 compute under AMP for stability in adjacency / message ops.

    Mathematical rationale
    ----------------------
    Graph adjacency operations involve:
        A_{ij} = exp(-d_{ij}^2 / 2σ^2)
    In fp16, small σ causes d^2/2σ^2 to exceed fp16 max (~65504),
    producing Inf → NaN propagation.  Forcing fp32 eliminates this.

    [P4-DEDUP] Previously duplicated in deep.py (L240) and models.py (L61).
    [P4-COMPAT] Uses torch.autocast (new API) instead of deprecated
                torch.cuda.amp.autocast (PyTorch ≥ 2.0).
    """
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    except Exception:
        yield


def nan_to_num(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with zeros.  Safe for gradient flow.

    [P4-DEDUP] Previously duplicated as ``_nan_to_num_`` in deep.py (L249)
    and models.py (L73).
    """
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================
# NODE_IDX resolution (single source of truth)
# =========================================================
def _safe_import_node_idx() -> Tuple[Optional[Dict[str, int]], Optional[List[str]]]:
    """Import NODE_IDX and NODE_FEATURE_NAMES from config, safely.

    Returns (node_idx_dict_or_None, node_feature_names_list_or_None).

    [P4-DEDUP] Previously each module (deep.py, models.py, fights.py)
    had its own 20+ line try/except chain doing the same thing.
    [P2-STRUCT-1] Now returns direct reference from config (no dict() copy).
    """
    node_idx: Optional[Dict[str, int]] = None
    node_names: Optional[List[str]] = None

    try:
        from core.config import NODE_IDX as _idx  # type: ignore
        if _idx is not None:
            node_idx = _idx  # [P2-STRUCT-1] direct ref, NOT dict()
    except ImportError:
        pass

    try:
        from core.config import NODE_FEATURE_NAMES as _names  # type: ignore
        if _names is not None:
            node_names = list(_names)
    except ImportError:
        pass

    return node_idx, node_names


def resolve_node_idx(
    override_idx: Optional[Dict[str, int]] = None,
    override_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Build NODE_IDX mapping with clear priority chain.

    Priority:
      1. override_idx (explicit caller argument)
      2. config.NODE_IDX (via feature contract)
      3. Build from override_names
      4. Build from config.NODE_FEATURE_NAMES
      5. Minimal fallback {"x_norm": 0, "y_norm": 1}

    [P4-DEDUP] Previously duplicated as ``_resolve_node_idx`` in
    deep.py (L279) and models.py (L98).
    """
    if override_idx is not None:
        return dict(override_idx)  # caller-provided: defensive copy is appropriate

    cfg_idx, cfg_names = _safe_import_node_idx()

    if cfg_idx is not None:
        return cfg_idx  # [P2-STRUCT-1] direct ref from FEATURE_CONTRACT (SSoT)

    names = override_names or cfg_names
    if names:
        return {str(n): i for i, n in enumerate(names)}

    logger.warning(
        "[common_torch] NODE_IDX resolution fell through to minimal fallback. "
        "Ensure config.py is importable."
    )
    return {"x_norm": 0, "y_norm": 1}


# =========================================================
# Index lookup helpers
# =========================================================
def idx_required(idx_map: Dict[str, int], key: str) -> int:
    """Require a key in NODE_IDX; raise clear error if missing."""
    if key not in idx_map:
        raise KeyError(
            f"[CONFIG MISMATCH] NODE_IDX has no key='{key}'. "
            f"Available keys ({len(idx_map)}): {sorted(idx_map.keys())[:10]}..."
        )
    return int(idx_map[key])


def idx_optional(idx_map: Dict[str, int], key: str) -> Optional[int]:
    """Optional key lookup — returns None if missing."""
    return int(idx_map[key]) if key in idx_map else None


def idx_first(idx_map: Dict[str, int], keys: List[str]) -> Optional[int]:
    """Return index of first matching key, or None."""
    for k in keys:
        if k in idx_map:
            return int(idx_map[k])
    return None


def pick_xy_indices(idx_map: Dict[str, int]) -> Tuple[int, int]:
    """Resolve X/Y coordinate indices with multi-name fallback."""
    x_cands = ["x_norm", "pos_x_norm", "x", "pos_x"]
    y_cands = ["y_norm", "pos_y_norm", "y", "pos_y"]

    x_i = idx_first(idx_map, x_cands)
    y_i = idx_first(idx_map, y_cands)

    if x_i is None:
        x_i = idx_map.get("x_norm", 0)
    if y_i is None:
        y_i = idx_map.get("y_norm", 1)

    return int(x_i), int(y_i)


def pick_alive_index(idx_map: Dict[str, int]) -> Optional[int]:
    """Resolve alive flag index."""
    return idx_first(idx_map, ["alive", "is_alive", "alive_flag"])