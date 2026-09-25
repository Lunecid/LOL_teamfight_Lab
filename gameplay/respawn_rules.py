# Verbatim copy of outputs/diag_survival_dbscan_20260925/scripts/respawn_rules.py
# source sha256: ba34b408fb2b3af6133252bec97f39911537c01d4d1d3691c729c956cf7157f0
# Do not edit below this header; tests/test_exact_survival.py checks the body hash.
"""Patch-specific Summoner's Rift death (respawn) timer, from the reconciled research spec
(../docs/research_respawn_dbscan_20260925.json, key timer_final; 2026-09-25).

Evidence:
* Riot patch notes: 14.16 set levels 1-9 to 10/10/12/12/14/16/20/25/28 s. No Summoner's Rift
  death-timer change in 25.14-25.16 or 26.13-26.15 (full-text search of each note).
* Game data (CommunityDragon CLASSIC mTimeDeadPerLevel, 13.19-16.16): levels 10-18 = 32.5..52.5 s.
* Time factor: continuous in game time at death, 0 until 15:00, then +0.425 % / +0.30 % / +1.45 %
  per 30 s after 15 / 30 / 45 min, capped at 50 %. It matched totalTimeSpentDead for 99.5-99.8 %
  of players in 15.18-15.22 and 16.13-16.15. The 30-second step form matched 75-88 %.
* Levels 19-20 (26.1+ top-lane role quest) use the level-18 value; this is empirical (99.4 % of 352 players).
Match-V5 gameVersion 15.N = public patch 25.N; 16.N = 26.N.
"""
from __future__ import annotations

from typing import Tuple

SPEC_FILLED = True
BRW_14_16_PLUS = (10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5)


def public_patch(match_v5_patch: str) -> Tuple[int, int]:
    major, minor = (int(x) for x in str(match_v5_patch).split(".")[:2])
    return (major + 10 if major in (15, 16) else major), minor


def time_factor(game_time_s: float) -> float:
    m = float(game_time_s) / 60.0
    if m <= 15.0:
        x = 0.0
    elif m <= 30.0:
        x = 0.0085 * (m - 15.0)
    elif m <= 45.0:
        x = 0.1275 + 0.0060 * (m - 30.0)
    else:
        x = 0.2175 + 0.0290 * (m - 45.0)
    return min(x, 0.50)


def death_timer_s(match_v5_patch: str, level: int, game_time_s: float) -> float:
    p = public_patch(match_v5_patch)
    if p < (14, 16) or p > (26, 16):
        raise KeyError(f"no verified respawn rule for public patch {p[0]}.{p[1]}")
    lv = min(max(int(level), 1), 18)
    return float(BRW_14_16_PLUS[lv - 1]) * (1.0 + time_factor(game_time_s))
