"""Team gold paid per event, recovered from the corpus.

Match-V5 records the gold a kill pays (``bounty`` + ``shutdownBounty``) but writes 0 for
structures and monsters, and the wiki's per-patch history is incomplete.  The frames,
however, carry each team's total gold every 60 s, so the price the game paid for each
event type can be recovered by regression: for every team and frame interval

    dGold_team = a + sum_e price_e * n_e + c_lane * dLaneCS + c_jg * dJgCS + eps

with the event counts of that team inside the interval as regressors.  The coefficient
of an event is the average total gold the team received for it (local + global, killer
+ assisters), which is exactly the currency the ``market_event`` label needs.  Kill
bounties are entered as their own regressor (coefficient ~1 if the frames agree with the
events) and the assist share as a count of assists.

Intervals before 2:00 (no passive income yet) are dropped.  Standard errors come from
the OLS residual variance; with ~10^6 intervals they are a few gold.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from core.contract import NODE_IDX

TOWER_TIERS = ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET")
MONSTERS = ("DRAGON", "ELDER_DRAGON", "BARON_NASHOR", "RIFTHERALD", "HORDE", "ATAKHAN")
REGRESSORS: List[str] = (
    ["kill_bounty", "kills", "assists", "plates"]
    + [f"tower_{t.lower()}" for t in TOWER_TIERS]
    + ["inhibitor", "first_tower"]
    + [m.lower() for m in MONSTERS]
    + ["ward_kills", "d_lane_cs", "d_jg_cs"]
)
LANE_CS_DENOM = 400.0
JG_CS_DENOM = 250.0


def _team_of_event(e: dict, tm: Dict[int, int]) -> int:
    et = str(e.get("type", "")).upper()
    if et in ("CHAMPION_KILL", "CHAMPION_SPECIAL_KILL"):
        return int(tm.get(int(e.get("killerId", 0) or 0), 0) or 0)
    if et == "ELITE_MONSTER_KILL":
        return int(e.get("killerTeamId", 0) or 0)
    if et in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
        victim = int(e.get("teamId", 0) or 0)
        return 200 if victim == 100 else (100 if victim == 200 else 0)
    if et == "WARD_KILL":
        return int(tm.get(int(e.get("killerId", 0) or 0), 0) or 0)
    return 0


def match_rows(pack: dict, min_start_ms: int = 120_000):
    """Rows (one per team per frame interval): y = team gold change, X = regressor vector."""
    ts = np.asarray(pack["minute_ts"], dtype=np.int64)
    gold = np.asarray(pack["gold_team_minute"], dtype=np.float64)   # (T, 2) blue, red
    nm = pack["node_minute"]
    tm = {int(k): int(v) for k, v in (pack["meta"].get("team_map") or {}).items()}
    tids = np.array([tm.get(i, 100 if i <= 5 else 200) for i in range(1, 11)])
    b_idx, r_idx = np.where(tids == 100)[0], np.where(tids == 200)[0]
    li, ji = NODE_IDX.get("laneCS_norm"), NODE_IDX.get("jgCS_norm")
    lane = np.stack([nm[:, b_idx, li].sum(axis=1), nm[:, r_idx, li].sum(axis=1)], axis=1) * LANE_CS_DENOM
    jg = np.stack([nm[:, b_idx, ji].sum(axis=1), nm[:, r_idx, ji].sum(axis=1)], axis=1) * JG_CS_DENOM
    T = len(ts)
    if T < 3:
        return None, None
    counts = np.zeros((T - 1, 2, len(REGRESSORS)), dtype=np.float64)
    col = {n: i for i, n in enumerate(REGRESSORS)}
    first_tower_seen = False
    for e in sorted(pack.get("events") or [], key=lambda e: int(e.get("timestamp", 0) or 0)):
        t = int(e.get("timestamp", 0) or 0)
        i = int(np.searchsorted(ts, t, side="right") - 1)
        if i < 0 or i >= T - 1:
            continue
        team = _team_of_event(e, tm)
        if team not in (100, 200):
            continue
        k = 0 if team == 100 else 1
        et = str(e.get("type", "")).upper()
        if et == "CHAMPION_KILL":
            counts[i, k, col["kills"]] += 1
            counts[i, k, col["kill_bounty"]] += float(e.get("bounty", 0) or 0) + float(e.get("shutdownBounty", 0) or 0)
            a = e.get("assistingParticipantIds") or []
            counts[i, k, col["assists"]] += len(a) if isinstance(a, list) else 0
        elif et == "TURRET_PLATE_DESTROYED":
            counts[i, k, col["plates"]] += 1
        elif et == "BUILDING_KILL":
            if str(e.get("buildingType", "")).upper() == "INHIBITOR_BUILDING":
                counts[i, k, col["inhibitor"]] += 1
            else:
                tier = str(e.get("towerType", "")).upper()
                name = f"tower_{tier.lower()}"
                if name in col:
                    counts[i, k, col[name]] += 1
                if not first_tower_seen:
                    counts[i, k, col["first_tower"]] += 1
                    first_tower_seen = True
        elif et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            st = str(e.get("monsterSubType", "")).upper()
            name = "elder_dragon" if (mt == "DRAGON" and st == "ELDER_DRAGON") else mt.lower()
            if name in col:
                counts[i, k, col[name]] += 1
        elif et == "WARD_KILL":
            counts[i, k, col["ward_kills"]] += 1
    y = (gold[1:] - gold[:-1])                     # (T-1, 2)
    counts[:, :, col["d_lane_cs"]] = lane[1:] - lane[:-1]
    counts[:, :, col["d_jg_cs"]] = jg[1:] - jg[:-1]
    keep = ts[:-1] >= min_start_ms
    X = counts[keep].reshape(-1, len(REGRESSORS))
    Y = y[keep].reshape(-1)
    ok = np.isfinite(Y) & np.isfinite(X).all(axis=1)
    return X[ok], Y[ok]


def fit_prices(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with intercept; returns coefficients, standard errors and fit diagnostics."""
    n, p = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(1, n - A.shape[1])
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    out = {"n_intervals": int(n), "r2": r2, "intercept": float(beta[0]), "intercept_se": float(se[0]),
           "coef": {name: float(beta[i + 1]) for i, name in enumerate(REGRESSORS)},
           "se": {name: float(se[i + 1]) for i, name in enumerate(REGRESSORS)},
           "count": {name: float(X[:, i].sum()) for i, name in enumerate(REGRESSORS)}}
    return out


def price_table(fit: dict, min_count: float = 200.0) -> Dict[str, float]:
    """Rounded team-gold prices for the label.

    * ``kills``/``assists``: gold per kill / per assister beyond the event's own bounty.
    * monsters: the frames count an epic monster as one jungle CS, so the CS regressor
      absorbs one CS worth of gold; add it back and clamp at zero.
    * ``first_tower`` is not a rule we could confirm and its coefficient is small and
      negative; it is left out (towers carry their average price).
    """
    table = {}
    jg = float(fit["coef"].get("d_jg_cs", 0.0))
    for name in REGRESSORS:
        if name in ("kill_bounty", "d_lane_cs", "d_jg_cs", "first_tower"):
            continue
        if fit["count"].get(name, 0.0) < min_count:
            continue
        v = float(fit["coef"][name])
        if name in [m.lower() for m in MONSTERS]:
            v = max(0.0, v + jg)
        table[name] = float(round(v / 5.0) * 5.0)
    return table
