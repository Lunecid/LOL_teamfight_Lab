"""Zone / tower / objective features must come from absolute positions, not the centroid-relative node XY."""
import numpy as np

from core.config import cfg
from core.contract import NODE_IDX
from gameplay.anchors import causal_anchors
from gameplay.feature_spatial import SPATIAL_FEATURE_NAMES, compute_spatial_seq_from_node


def _node_seq(L=2):
    node = np.zeros((L, 10, len(NODE_IDX)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    return node


def _xy_abs(x, y, L=2):
    div = float(getattr(cfg, "COORD_NORM_DIV", 16000.0))
    xy = np.zeros((L, 10, 2), dtype=np.float32)
    xy[:, :, 0] = x / div
    xy[:, :, 1] = y / div
    return xy


def test_zone_and_objective_follow_absolute_position():
    sample = {"anchors": causal_anchors({"events": []}, 0), "anchor_is_norm": False}
    idx = {n: i for i, n in enumerate(SPATIAL_FEATURE_NAMES)}
    at_dragon = compute_spatial_seq_from_node(_node_seq(), sample, xy_abs=_xy_abs(9837, 4401))
    at_top = compute_spatial_seq_from_node(_node_seq(), sample, xy_abs=_xy_abs(1500, 12000))
    assert at_dragon[0, idx["near_obj_dragon"]] == 1.0 and at_top[0, idx["near_obj_dragon"]] == 0.0
    assert at_top[0, idx["zone_top_lane"]] == 1.0 and at_dragon[0, idx["zone_top_lane"]] == 0.0
    assert at_dragon[0, idx["dist_tower_nearest_norm"]] != at_top[0, idx["dist_tower_nearest_norm"]]


def test_relative_xy_alone_is_origin_bound():
    sample = {"anchors": causal_anchors({"events": []}, 0), "anchor_is_norm": False}
    idx = {n: i for i, n in enumerate(SPATIAL_FEATURE_NAMES)}
    a = compute_spatial_seq_from_node(_node_seq(), sample)   # relative XY all zero -> origin
    assert a[0, idx["zone_mid_lane"]] == 1.0
