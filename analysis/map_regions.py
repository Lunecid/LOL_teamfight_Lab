"""Summoner's Rift regions for kill positions (Match-V5 coordinates).

The map is roughly 14,800 x 14,800 units, blue base bottom-left, red base
top-right.  Lanes are modelled as polylines through the outer/inner tower
positions observed in Match-V5 BUILDING_KILL events (patch 15.x):

* top:  up the left edge (x ~ 1,000) then across the top edge (y ~ 13,900)
* bot:  across the bottom edge (y ~ 1,000) then up the right edge (x ~ 13,900)
* mid:  the diagonal x ~ y

The river is the anti-diagonal x + y ~ 14,800 between the two pits.  Region
labels are a coarse, patch-stable partition meant for stratifying kill-pair
statistics, not a navmesh.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

MAP_SIZE = 14_800.0
DRAGON_PIT = np.array([9_850.0, 4_400.0])
BARON_PIT = np.array([5_000.0, 10_400.0])
BLUE_NEXUS = np.array([1_600.0, 1_600.0])
RED_NEXUS = np.array([13_200.0, 13_200.0])

# lane polylines (x, y)
LANE_TOP = np.array([[1_000.0, 3_200.0], [1_000.0, 13_800.0], [11_600.0, 13_800.0]])
LANE_BOT = np.array([[3_200.0, 1_000.0], [13_800.0, 1_000.0], [13_800.0, 11_600.0]])
LANE_MID = np.array([[3_000.0, 3_000.0], [11_800.0, 11_800.0]])

LANE_HALF_WIDTH = 900.0    # distance from lane centreline that still counts as "in lane"
PIT_RADIUS = 1_400.0
BASE_RADIUS = 3_000.0
RIVER_HALF_WIDTH = 800.0

REGIONS = ("base", "top", "mid", "bot", "dragon_pit", "baron_pit", "river", "jungle_top", "jungle_bot")


def _seg_dist_and_tangent(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from points p (n,2) to segment ab, and the unit tangent of ab."""
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((p - a) @ ab) / max(L2, 1e-9), 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(p - proj, axis=1)
    tangent = ab / np.sqrt(max(L2, 1e-9))
    return d, np.broadcast_to(tangent, p.shape)


def polyline_dist_and_tangent(p: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    best_d = np.full(p.shape[0], np.inf)
    best_t = np.zeros_like(p)
    for a, b in zip(poly[:-1], poly[1:]):
        d, t = _seg_dist_and_tangent(p, a, b)
        m = d < best_d
        best_d[m] = d[m]
        best_t[m] = t[m]
    return best_d, best_t


def classify_points(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Region label per point and, for lane points, the local lane tangent.

    Returns (labels: array of str, tangent: (n,2) unit vectors; zeros outside
    lanes).  Priority: base > pits > lane > river > jungle.
    """
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = p.shape[0]
    labels = np.full(n, "jungle_bot", dtype=object)
    tangent = np.zeros((n, 2))

    d_top, t_top = polyline_dist_and_tangent(p, LANE_TOP)
    d_mid, t_mid = polyline_dist_and_tangent(p, LANE_MID)
    d_bot, t_bot = polyline_dist_and_tangent(p, LANE_BOT)
    lane_d = np.stack([d_top, d_mid, d_bot], axis=1)
    lane_i = np.argmin(lane_d, axis=1)
    lane_min = lane_d[np.arange(n), lane_i]
    lane_names = np.array(["top", "mid", "bot"], dtype=object)
    lane_t = np.stack([t_top, t_mid, t_bot], axis=1)[np.arange(n), lane_i]

    in_lane = lane_min <= LANE_HALF_WIDTH
    labels[in_lane] = lane_names[lane_i[in_lane]]
    tangent[in_lane] = lane_t[in_lane]

    river = np.abs(p[:, 0] + p[:, 1] - MAP_SIZE) <= RIVER_HALF_WIDTH
    labels[river & ~in_lane] = "river"

    above = p[:, 1] > p[:, 0]
    jungle = ~in_lane & ~river
    labels[jungle & above] = "jungle_top"
    labels[jungle & ~above] = "jungle_bot"

    d_drag = np.linalg.norm(p - DRAGON_PIT, axis=1)
    d_baron = np.linalg.norm(p - BARON_PIT, axis=1)
    labels[d_drag <= PIT_RADIUS] = "dragon_pit"
    labels[d_baron <= PIT_RADIUS] = "baron_pit"
    tangent[(d_drag <= PIT_RADIUS) | (d_baron <= PIT_RADIUS)] = 0.0

    base = (np.linalg.norm(p - BLUE_NEXUS, axis=1) <= BASE_RADIUS) | (np.linalg.norm(p - RED_NEXUS, axis=1) <= BASE_RADIUS)
    labels[base] = "base"
    tangent[base] = 0.0
    return labels.astype(str), tangent


def along_across(disp: np.ndarray, tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose displacement vectors into |along-lane| and |across-lane| components."""
    along = np.abs(np.einsum("ij,ij->i", disp, tangent))
    total2 = np.einsum("ij,ij->i", disp, disp)
    across = np.sqrt(np.maximum(total2 - along ** 2, 0.0))
    return along, across


def region_summary(labels: np.ndarray) -> Dict[str, int]:
    u, c = np.unique(labels, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}
