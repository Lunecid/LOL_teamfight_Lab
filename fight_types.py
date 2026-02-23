"""types.py — Core data types for the teamfight pipeline.

Changes from original:
  [FIX-IMPORT] Replaced ``from .common import *`` with explicit imports.
  [FIX-IMPORT] Added missing ``dataclass`` import (previously relied on
               transitive wildcard, which silently broke when common.py
               didn't export it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FightRef:
    """Reference to a single detected teamfight instance.

    Attributes
    ----------
    match_id : str
        Riot match identifier (e.g., ``"KR_7123456789"``).
    patch : str
        Game patch string for stratified splitting.
    t_start : int
        Minute index into ``cache["minute_ts"]`` (legacy, kept for compat).
    t_start_ts : int
        Engage timestamp in milliseconds (primary anchor; ``-1`` = unset).
    """
    match_id: str
    patch: str
    t_start: int              # minute index (legacy)
    t_start_ts: int = -1      # engage timestamp in ms (primary anchor)

    def __post_init__(self) -> None:
        # t_start_ts가 설정되지 않았으면 -1 유지 (legacy 모드)
        if self.t_start_ts < 0 and self.t_start >= 0:
            pass  # 호환성: 기존 코드에서 t_start만 쓸 때는 -1로 둠


@dataclass
class PruneSpec:
    """Column-pruning specification for tabular features."""
    x_keep: Optional[np.ndarray] = None
    extra_keep: Optional[np.ndarray] = None


def ref_key(r: FightRef) -> str:
    """Unique string key for a FightRef — used as dict/set key throughout.

    [FIX] Prioritises ``t_start_ts`` (ms-level precision) over the legacy
    minute-level ``t_start`` to avoid collisions when multiple fights
    occur within the same game-minute.
    """
    if r.t_start_ts >= 0:
        return f"{r.match_id}|t_start_ts={int(r.t_start_ts)}"
    return f"{r.match_id}|t_start={int(r.t_start)}"
