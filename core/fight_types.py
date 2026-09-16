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
    label_end_ts : int
        Label window end timestamp in milliseconds (exclusive).
        ``-1`` means "use default horizon".
    """
    match_id: str
    patch: str
    t_start: int              # minute index (legacy)
    t_start_ts: int = -1      # engage timestamp in ms (primary anchor)
    label_end_ts: int = -1    # label window end ts in ms (exclusive)
    first_kill_ts: int = -1   # cluster first kill timestamp in ms
    last_kill_ts: int = -1    # cluster last kill timestamp in ms
    det_cluster_blue: int = -1  # observed event participants, blue (-1 unknown)
    det_cluster_red: int = -1   # observed event participants, red (-1 unknown)
    det_present_blue: int = -1  # alive champions near the anchor at the cutoff
    det_present_red: int = -1   # same, red (-1 unknown)
    anchor_x: float = -1.0      # fight centre (first kill position), game units; -1 unknown
    anchor_y: float = -1.0

    @staticmethod
    def _scale_class(blue: int, red: int) -> Optional[str]:
        if blue < 0 or red < 0:
            return None
        smaller = min(blue, red)
        if smaller >= 3:
            return "teamfight"
        if smaller >= 2:
            return "skirmish"
        return "pick"

    @property
    def fight_scale(self) -> Optional[str]:
        """Scale from *observed participation* (post-hoc; known only after the fight).

        Same rule as ``classify_fight_scale``: a player counts once they appear
        in the cluster's kill/assist/death records or trigger a positioned
        event inside the fight window.
        """
        return self._scale_class(self.det_cluster_blue, self.det_cluster_red)

    @property
    def presence_scale(self) -> Optional[str]:
        """Scale from *presence at the prediction cutoff* (pre-fight, operational).

        Counts alive champions per team within the validity radius of the
        engagement anchor at the cutoff, so it is available to a predictor.
        Detection requires at least two per team, so this never yields
        ``"pick"`` -- a pick is an engagement where bodies were present but
        only one side's player traded.
        """
        return self._scale_class(self.det_present_blue, self.det_present_red)

    @property
    def commitment_gap(self) -> Optional[int]:
        """Champions that were present at the cutoff but never participated."""
        if self.det_present_blue < 0 or self.det_cluster_blue < 0:
            return None
        present = self.det_present_blue + self.det_present_red
        engaged = self.det_cluster_blue + self.det_cluster_red
        return present - engaged

    def __post_init__(self) -> None:
        # t_start_ts가 설정되지 않았으면 -1 유지 (legacy 모드)
        if self.t_start_ts < 0 and self.t_start >= 0:
            pass  # 호환성: 기존 코드에서 t_start만 쓸 때는 -1로 둠
        if self.t_start_ts >= 0 and self.label_end_ts >= 0 and self.label_end_ts <= self.t_start_ts:
            self.label_end_ts = -1


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
