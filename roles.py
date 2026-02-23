"""roles.py — Role normalisation and slot assignment.

Changes from original:
  [FIX-IMPORT] Replaced ``from .common import *`` with explicit imports.
  [FIX-IMPORT] ``ROLE_ORDER`` is now imported explicitly from ``.config``
               instead of relying on a transitive wildcard chain.
"""
from __future__ import annotations

from typing import Dict, List

from config import ROLE_ORDER


def _norm_role(x: str) -> str:
    """Normalise free-form role string to canonical Riot API names."""
    x = (x or "").upper().strip()
    if x in ("MID", "MIDDLE"):
        return "MIDDLE"
    if x in ("BOT", "BOTTOM"):
        return "BOTTOM"
    if x in ("SUP", "SUPPORT", "UTILITY"):
        return "UTILITY"
    if x in ("JNG", "JG", "JUNGLE"):
        return "JUNGLE"
    if x in ("TOP",):
        return "TOP"
    return x


def get_role_slots_from_detail(detail: dict) -> Dict[int, int]:
    """Map participantId → canonical slot (0..9).

    Slots are ordered:
      [bTOP, bJNG, bMID, bBOT, bSUP, rTOP, rJNG, rMID, rBOT, rSUP]
    """
    if not isinstance(detail, dict):
        return {pid: pid - 1 for pid in range(1, 11)}

    pid2team: Dict[int, int] = {}
    pid2role: Dict[int, str] = {}
    try:
        parts = detail.get("info", {}).get("participants", None)
        if not isinstance(parts, list) or len(parts) < 10:
            raise ValueError("bad participants")
        for p in parts:
            if not isinstance(p, dict):
                continue
            pid = int(p.get("participantId", 0) or 0)
            if pid <= 0:
                continue
            pid2team[pid] = int(p.get("teamId", 0) or 0)
            pid2role[pid] = _norm_role(
                p.get("teamPosition") or p.get("individualPosition") or ""
            )
    except (ValueError, TypeError, KeyError):
        return {pid: pid - 1 for pid in range(1, 11)}

    pid2slot: Dict[int, int] = {}
    used_slots: set = set()

    def _assign(tid: int, offset: int) -> None:
        for idx, role in enumerate(ROLE_ORDER):
            cands = sorted(
                [p for p, t in pid2team.items() if t == tid and pid2role.get(p) == role]
            )
            if cands and (offset + idx) not in used_slots and cands[0] not in pid2slot:
                pid2slot[cands[0]] = offset + idx
                used_slots.add(offset + idx)

        for slot in [offset + i for i in range(5) if (offset + i) not in used_slots]:
            cand = next(
                (p for p in sorted([p for p, t in pid2team.items() if t == tid]) if p not in pid2slot),
                None,
            )
            if cand:
                pid2slot[cand] = slot
                used_slots.add(slot)

    _assign(100, 0)
    _assign(200, 5)

    for p in range(1, 11):
        if p not in pid2slot:
            pid2slot[p] = p - 1
    return {int(k): int(v) for k, v in pid2slot.items()}
