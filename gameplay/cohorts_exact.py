"""Engagement cohorts from kill credit only (v4-exact stage 1, task B1).

Participation is the union of kill credit over the engagement's kills: killer, victim and
assisting participants (ids 1..10; killer 0 = minion/turret/monster is ignored), split by team.
This is the participant rule of fight_clustering.finalize_kill_cluster (fight_clustering.py:10-42)
without the interaction actors of the v3.3 detector.  No position, frame or grid data is read.

Cohorts (plan section 2, decision 4), with n_blue / n_red the credited counts per team:
  'T'    both teams >= 4
  'S'    both teams 2-3
  'ASYM' one team 2-3, the other >= 4
  'P'    min(n_blue, n_red) < 2
T5 flag: both teams 5 (a full 5 v 5 by kill credit).

Cohorts are realised after the engagement (they use the victims and assisters of its kills);
they define the sample, not a pre-engagement feature.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, Mapping, Set, Tuple

import numpy as np

COHORTS = ("T", "S", "ASYM", "P")


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def team_sets(tm: Mapping[int, int]) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """(blue pids, red pids) with the same rule as fights.validate_team_mapping (pid = index + 1)."""
    tm = {int(k): int(v) for k, v in (tm or {}).items()}
    if not tm:
        tm = {i: 100 if i <= 5 else 200 for i in range(1, 11)}
    tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)], dtype=np.int32)
    b = np.where(tids == 100)[0]
    r = np.where(tids == 200)[0]
    if len(b) == 0:
        unique = np.unique(tids)
        if len(unique) >= 2:
            b = np.where(tids == unique[0])[0]
            r = np.where(tids == unique[1])[0]
        else:
            b = np.arange(0, 5)
            r = np.arange(5, 10)
    return frozenset(int(x) + 1 for x in b.tolist()), frozenset(int(x) + 1 for x in r.tolist())


def kill_credit(kill: Mapping) -> Set[int]:
    """Credited participants (1..10) of one kill.  Accepts the fights._extract_kill_events format
    (killer_id, victim_id, assisting_ids) or a raw Match-V5 CHAMPION_KILL event
    (killerId, victimId, assistingParticipantIds)."""
    if "killer_id" in kill or "victim_id" in kill or "assisting_ids" in kill:
        ids = [kill.get("killer_id", 0), kill.get("victim_id", 0)] + list(kill.get("assisting_ids") or [])
    else:
        ids = [kill.get("killerId", 0), kill.get("victimId", 0)] + list(kill.get("assistingParticipantIds") or [])
    out = set()
    for v in ids:
        p = _safe_int(v)
        if 1 <= p <= 10:
            out.add(p)
    return out


def credited_participants(kills: Iterable[Mapping]) -> Set[int]:
    """Union of kill credit (killer / victim / assisters) over the kills."""
    out: Set[int] = set()
    for k in kills:
        out |= kill_credit(k)
    return out


def credited_sets(kills: Iterable[Mapping], tm: Mapping[int, int]) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """(blue credited pids, red credited pids): the credit union split by team."""
    blue, red = team_sets(tm)
    u = credited_participants(kills)
    return frozenset(u & blue), frozenset(u & red)


def cohort_label(n_blue: int, n_red: int) -> str:
    """'T' / 'S' / 'ASYM' / 'P' from the per-team credited counts."""
    lo, hi = min(int(n_blue), int(n_red)), max(int(n_blue), int(n_red))
    if lo < 2:
        return "P"
    if lo >= 4:
        return "T"
    if hi <= 3:
        return "S"
    return "ASYM"


def cohort_from_counts(n_blue: int, n_red: int) -> Dict[str, Any]:
    n_blue, n_red = int(n_blue), int(n_red)
    return {"n_blue": n_blue, "n_red": n_red, "n_min": min(n_blue, n_red), "n_max": max(n_blue, n_red),
            "cohort": cohort_label(n_blue, n_red), "t5": int(n_blue == 5 and n_red == 5)}


def cohort_record(kills: Iterable[Mapping], tm: Mapping[int, int]) -> Dict[str, Any]:
    """Cohort of an engagement from its kills: n_blue, n_red, n_min, n_max, cohort, t5, blue, red."""
    blue, red = credited_sets(kills, tm)
    rec = cohort_from_counts(len(blue), len(red))
    rec["blue"] = sorted(blue)
    rec["red"] = sorted(red)
    return rec
