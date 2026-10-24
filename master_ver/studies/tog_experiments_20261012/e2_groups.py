"""T2.1 E2 group at the prediction time and its scale stratum (PRESPEC_E2_KO.md section 3, PLAN_E1_E5_KO.md Task 3).

At the prediction-time frame F(tau) the ten players' coordinates form a graph: two players of any team are linked
when their Euclidean distance is at most ``radius``.  A connected component holding players of both teams is a mixed
group.  The chosen group is the mixed group with the largest (size, count of the smaller team, -smallest participant
index); its per-team counts n_small <= n_large give the stratum.  The first-kill position is not used.

A player's position is unknown when a coordinate is not finite or x or y is 0 (the rule of
reports/q_production_20261005/headcount_tau.py lines 233-235).  Unknown players are left out of the graph and counted.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def pre_tau_group(xy: np.ndarray, team: np.ndarray, radius: float = 2000.0) -> tuple[int, int, int]:
    """(n_small, n_large, n_unknown) of the chosen mixed group at one frame, or (0, 0, n_unknown) when there is none.

    ``xy`` is the (10, 2) frame in bank participant_order 1-10 and ``team`` the (10,) team ids (two values, such as
    100 and 200).  The participant index of the tie-break is the position in these arrays.  ``radius`` must be
    finite and positive (a NaN radius would silently make every frame separated).
    """

    xy = np.asarray(xy, dtype=np.float64)
    team = np.asarray(team)
    teams = np.unique(team)
    if xy.shape != (10, 2) or team.shape != (10,) or teams.size != 2:
        raise ValueError(f"expected xy (10, 2) and team (10,) with two team ids, got {xy.shape}, {team.shape}, "
                         f"{teams.tolist()}")
    radius = float(radius)
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius must be finite and > 0, got {radius}")
    known = np.isfinite(xy).all(axis=1) & (xy[:, 0] != 0.0) & (xy[:, 1] != 0.0)
    index = np.flatnonzero(known)                       # participant positions of the players in the graph
    dx = xy[index, 0][:, None] - xy[index, 0][None, :]
    dy = xy[index, 1][:, None] - xy[index, 1][None, :]
    linked = dx * dx + dy * dy <= float(radius) * float(radius)
    n_components, label = connected_components(csr_matrix(linked), directed=False)
    mixed = []
    for component in range(n_components):
        members = index[label == component]
        n_first = int((team[members] == teams[0]).sum())
        n_second = members.size - n_first
        if n_first and n_second:
            n_small, n_large = sorted((n_first, n_second))
            mixed.append(((members.size, n_small, -int(members.min())), n_small, n_large))
    n_unknown = int((~known).sum())
    if not mixed:
        return 0, 0, n_unknown
    _, n_small, n_large = max(mixed)
    return n_small, n_large, n_unknown


def stratum(n_small: int, n_large: int) -> str:
    """Scale stratum of a chosen group's per-team counts n_small <= n_large; (0, 0) means no mixed group."""

    if not (n_small == n_large == 0 or 1 <= n_small <= n_large):
        raise ValueError(f"not the per-team counts of a mixed group, nor (0, 0): ({n_small}, {n_large})")
    if n_large == 0:
        return "separated"
    if n_small <= 1:
        return "pick"
    if n_large <= 3:
        return "small"           # both sides 2-3
    if n_small <= 3:
        return "asymmetric"      # one side 2-3, the other 4 or more
    return "teamfight"           # both sides 4 or more
