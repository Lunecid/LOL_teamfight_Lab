from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import BARON_PIT_XY, DRAGON_PIT_XY, MAP_MAX, NODE_IDX, cfg


def _find_node_idx(cands: List[str], default: Optional[int] = None) -> Optional[int]:
    for c in cands:
        if c in NODE_IDX:
            return int(NODE_IDX[c])
    return default


SPATIAL_FEATURE_NAMES: List[str] = [
    "pos_fight_x_norm",
    "pos_fight_y_norm",
    "pos_blue_x_norm",
    "pos_blue_y_norm",
    "pos_red_x_norm",
    "pos_red_y_norm",
    "dist_team_sep_norm",
    "standoff_pairs_frac",
    "mean_min_enemy_dist_norm",
    "d_standoff_pairs_frac",
    "d_mean_min_enemy_dist_norm",
    "dist_obj_nearest_norm",
    "near_obj_dragon",
    "near_obj_baron",
    "near_obj_herald",
    "near_obj_atakhan",
    "near_obj_horde",
    "dist_tower_nearest_norm",
    "in_tower_range",
    "near_tower_radius",
    "zone_top_lane",
    "zone_mid_lane",
    "zone_bot_lane",
    "zone_river",
    "zone_jungle",
]
F_SPATIAL: int = len(SPATIAL_FEATURE_NAMES)


def get_spatial_feature_names() -> List[str]:
    return list(SPATIAL_FEATURE_NAMES)


_DEFAULT_OBJ_ANCHORS_MAP = {
    "DRAGON": np.array([[float(DRAGON_PIT_XY[0]), float(DRAGON_PIT_XY[1])]], dtype=np.float32),
    "BARON": np.array([[float(BARON_PIT_XY[0]), float(BARON_PIT_XY[1])]], dtype=np.float32),
    "RIFTHERALD": np.array([[float(BARON_PIT_XY[0]), float(BARON_PIT_XY[1])]], dtype=np.float32),
    "ATAKHAN": np.zeros((0, 2), dtype=np.float32),
    "HORDE": np.zeros((0, 2), dtype=np.float32),
}
_DEFAULT_TOWER_ANCHORS_MAP = {
    "TOWER_T100": np.zeros((0, 2), dtype=np.float32),
    "TOWER_T200": np.zeros((0, 2), dtype=np.float32),
}


def _coord_div() -> float:
    return float(getattr(cfg, "COORD_NORM_DIV", MAP_MAX))


def _to_norm_xy(xy_map: np.ndarray) -> np.ndarray:
    if xy_map.size == 0:
        return xy_map.astype(np.float32)
    div = _coord_div()
    return (xy_map.astype(np.float32) / div).astype(np.float32)


def _as_xy_array(v: Any) -> np.ndarray:
    if v is None:
        return np.zeros((0, 2), np.float32)
    if isinstance(v, (float, int, np.floating, np.integer)):
        return np.zeros((0, 2), np.float32)
    try:
        arr = np.asarray(v, dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), np.float32)

    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return arr.reshape(1, 2).astype(np.float32)
        return np.zeros((0, 2), np.float32)
    if arr.ndim >= 2:
        return arr.reshape(-1, 2).astype(np.float32)
    return np.zeros((0, 2), np.float32)


def _get_anchors_norm_from_sample(sample: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    anchors = sample.get("anchors", None)
    anchor_is_norm = bool(sample.get("anchor_is_norm", getattr(cfg, "ANCHOR_IS_NORM", False)))

    obj_map = {k: v.copy() for k, v in _DEFAULT_OBJ_ANCHORS_MAP.items()}
    tower_map = {k: v.copy() for k, v in _DEFAULT_TOWER_ANCHORS_MAP.items()}

    if isinstance(anchors, dict):
        obj = anchors.get("obj", None)
        tower = anchors.get("tower", None)
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj_map[str(k).upper()] = _as_xy_array(v)
        if isinstance(tower, dict):
            for k, v in tower.items():
                tower_map[str(k).upper()] = _as_xy_array(v)

    if anchor_is_norm:
        obj_norm = {k: v.astype(np.float32) for k, v in obj_map.items()}
        tower_norm = {k: v.astype(np.float32) for k, v in tower_map.items()}
    else:
        obj_norm = {k: _to_norm_xy(v) for k, v in obj_map.items()}
        tower_norm = {k: _to_norm_xy(v) for k, v in tower_map.items()}
    return {"obj": obj_norm, "tower": tower_norm}


def _pairwise_min_dist(xy: np.ndarray, anchors: np.ndarray) -> float:
    if anchors.size == 0:
        return 1.0
    xy = np.asarray(xy, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if xy.ndim == 1:
        d = anchors - xy[None, :]
        return float(np.sqrt((d * d).sum(axis=1)).min())
    d = xy[:, None, :] - anchors[None, :, :]
    return float(np.sqrt((d * d).sum(axis=2)).min())


def _zone_onehot_from_xy_map(x_map: float, y_map: float) -> Tuple[int, int, int, int, int]:
    x = float(np.clip(x_map, 0.0, MAP_MAX))
    y = float(np.clip(y_map, 0.0, MAP_MAX))
    u = x - y
    au = abs(u)
    in_center = (3500.0 <= x <= 12500.0) and (3500.0 <= y <= 12500.0)

    if in_center and au <= 2000.0:
        core = (4800.0 <= x <= 11200.0) and (4800.0 <= y <= 11200.0)
        if core:
            return (0, 0, 0, 1, 0)

    if au <= 1100.0:
        return (0, 1, 0, 0, 0)

    top_edge = (y >= MAP_MAX * 0.70) or (x <= MAP_MAX * 0.30)
    bot_edge = (y <= MAP_MAX * 0.30) or (x >= MAP_MAX * 0.70)
    if u < 0 and top_edge:
        return (1, 0, 0, 0, 0)
    if u > 0 and bot_edge:
        return (0, 0, 1, 0, 0)
    return (0, 0, 0, 0, 1)


def _compute_standoff_signals(
    blue_xy: np.ndarray,
    red_xy: np.ndarray,
    blue_alive: np.ndarray,
    red_alive: np.ndarray,
    standoff_r_norm: float,
) -> Tuple[float, float]:
    bmask = blue_alive > 0.5
    rmask = red_alive > 0.5
    if bmask.sum() == 0 or rmask.sum() == 0:
        return 0.0, 1.0

    bxy = blue_xy[bmask]
    rxy = red_xy[rmask]
    d = bxy[:, None, :] - rxy[None, :, :]
    dist = np.sqrt((d * d).sum(axis=2))

    pairs = float(dist.size)
    close = float((dist <= standoff_r_norm).sum())
    standoff_pairs_frac = close / max(1.0, pairs)

    b_near = dist.min(axis=1)
    r_near = dist.min(axis=0)
    mean_min_enemy = float(np.concatenate([b_near, r_near], axis=0).mean())
    return float(standoff_pairs_frac), float(mean_min_enemy)


def _compute_spatial_batch(
    node_role_seq: np.ndarray,
    xi: int,
    yi: int,
    ai: Optional[int],
    standoff_r_norm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L = node_role_seq.shape[0]
    xy = node_role_seq[:, :, [xi, yi]].astype(np.float32)
    alive = node_role_seq[:, :, ai].astype(np.float32) if ai is not None else np.ones((L, 10), np.float32)

    w = alive[:, :, None]
    wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    fight_xy = (xy * w).sum(axis=1) / wsum.squeeze(-1)

    blue_xy = xy[:, :5]
    red_xy = xy[:, 5:]
    blue_w = w[:, :5]
    red_w = w[:, 5:]
    blue_sum = np.maximum(blue_w.sum(axis=1, keepdims=True), 1e-6)
    red_sum = np.maximum(red_w.sum(axis=1, keepdims=True), 1e-6)
    blue_c = (blue_xy * blue_w).sum(axis=1) / blue_sum.squeeze(-1)
    red_c = (red_xy * red_w).sum(axis=1) / red_sum.squeeze(-1)
    team_sep = np.sqrt(((blue_c - red_c) ** 2).sum(axis=1))

    d = blue_xy[:, :, None, :] - red_xy[:, None, :, :]
    dist = np.sqrt((d * d).sum(axis=-1))
    b_alive = alive[:, :5] > 0.5
    r_alive = alive[:, 5:] > 0.5
    pair_mask = b_alive[:, :, None] & r_alive[:, None, :]

    close = ((dist <= standoff_r_norm) & pair_mask).sum(axis=(1, 2)).astype(np.float32)
    total_pairs = pair_mask.sum(axis=(1, 2)).astype(np.float32)
    pairs_frac = np.where(total_pairs > 0, close / total_pairs, 0.0)

    big_val = np.float32(1e6)
    dist_masked = np.where(pair_mask, dist, big_val)
    b_near = np.where(r_alive[:, None, :].any(axis=2), dist_masked.min(axis=2), big_val)
    r_near = np.where(b_alive[:, :, None].any(axis=1), dist_masked.min(axis=1), big_val)
    all_near = np.concatenate([b_near, r_near], axis=1)
    valid_near = all_near < big_val
    mean_min_enemy = np.where(
        valid_near.any(axis=1),
        np.where(valid_near, all_near, 0.0).sum(axis=1) / np.maximum(valid_near.sum(axis=1), 1),
        1.0,
    )
    return fight_xy, blue_c, red_c, team_sep, pairs_frac, mean_min_enemy


def compute_spatial_seq_from_node(node_role_seq: np.ndarray, sample: Dict[str, Any]) -> np.ndarray:
    anchors = _get_anchors_norm_from_sample(sample)
    obj = anchors["obj"]
    tower = anchors["tower"]
    div = _coord_div()

    obj_r_norm = float(getattr(cfg, "OBJ_NEAR_RADIUS", 1400.0)) / div
    tower_range_norm = float(getattr(cfg, "TOWER_RANGE", 775.0)) / div
    tower_near_norm = float(getattr(cfg, "TOWER_NEAR_RADIUS", 900.0)) / div
    standoff_r_norm = float(getattr(cfg, "STANDOFF_RADIUS", 1800.0)) / div

    L = int(node_role_seq.shape[0])
    out = np.zeros((L, F_SPATIAL), dtype=np.float32)

    xi = _find_node_idx(["x_norm", "pos_x_norm", "pos_x"], default=0)
    yi = _find_node_idx(["y_norm", "pos_y_norm", "pos_y"], default=1)
    ai = _find_node_idx(["alive", "is_alive", "alive_flag"], default=None)

    prev_pairs_frac = 0.0
    prev_mean_min_enemy = 1.0
    for t in range(L):
        step = node_role_seq[t].astype(np.float32)
        xy = step[:, [xi, yi]].astype(np.float32)
        alive = step[:, ai].astype(np.float32) if ai is not None else np.ones((10,), np.float32)

        w = alive[:, None]
        wsum = float(w.sum()) if float(w.sum()) > 1e-6 else 1.0
        fight_xy = (xy * w).sum(axis=0) / wsum

        blue_xy = xy[:5]
        red_xy = xy[5:]
        blue_alive = alive[:5]
        red_alive = alive[5:]
        blue_w = w[:5]
        red_w = w[5:]
        blue_sum = float(blue_w.sum()) if float(blue_w.sum()) > 1e-6 else 1.0
        red_sum = float(red_w.sum()) if float(red_w.sum()) > 1e-6 else 1.0
        blue_c = (blue_xy * blue_w).sum(axis=0) / blue_sum
        red_c = (red_xy * red_w).sum(axis=0) / red_sum
        team_sep = float(np.sqrt(((blue_c - red_c) ** 2).sum()))

        pairs_frac, mean_min_enemy = _compute_standoff_signals(
            blue_xy, red_xy, blue_alive, red_alive, standoff_r_norm
        )
        d_pairs_frac = float(pairs_frac - prev_pairs_frac) if t > 0 else 0.0
        d_mean_min_enemy = float(mean_min_enemy - prev_mean_min_enemy) if t > 0 else 0.0
        prev_pairs_frac = float(pairs_frac)
        prev_mean_min_enemy = float(mean_min_enemy)

        d_dragon = _pairwise_min_dist(fight_xy, obj.get("DRAGON", np.zeros((0, 2), np.float32)))
        d_baron = _pairwise_min_dist(fight_xy, obj.get("BARON", np.zeros((0, 2), np.float32)))
        d_herald = _pairwise_min_dist(fight_xy, obj.get("RIFTHERALD", np.zeros((0, 2), np.float32)))
        d_atakhan = _pairwise_min_dist(fight_xy, obj.get("ATAKHAN", np.zeros((0, 2), np.float32)))
        d_horde = _pairwise_min_dist(fight_xy, obj.get("HORDE", np.zeros((0, 2), np.float32)))
        d_obj = min(d_dragon, d_baron, d_herald, d_atakhan, d_horde)

        near_dragon = 1.0 if d_dragon <= obj_r_norm else 0.0
        near_baron = 1.0 if d_baron <= obj_r_norm else 0.0
        near_herald = 1.0 if d_herald <= obj_r_norm else 0.0
        near_atakhan = 1.0 if d_atakhan <= obj_r_norm else 0.0
        near_horde = 1.0 if d_horde <= obj_r_norm else 0.0

        t100 = tower.get("TOWER_T100", np.zeros((0, 2), np.float32))
        t200 = tower.get("TOWER_T200", np.zeros((0, 2), np.float32))
        d_t100 = _pairwise_min_dist(fight_xy, t100)
        d_t200 = _pairwise_min_dist(fight_xy, t200)
        d_tower = min(d_t100, d_t200)

        in_tower_range = 1.0 if d_tower <= tower_range_norm else 0.0
        near_tower_radius = 1.0 if d_tower <= tower_near_norm else 0.0

        fx_map = float(fight_xy[0] * div)
        fy_map = float(fight_xy[1] * div)
        z_top, z_mid, z_bot, z_riv, z_jng = _zone_onehot_from_xy_map(fx_map, fy_map)

        row = np.asarray(
            [
                fight_xy[0],
                fight_xy[1],
                blue_c[0],
                blue_c[1],
                red_c[0],
                red_c[1],
                team_sep,
                float(pairs_frac),
                float(mean_min_enemy),
                float(d_pairs_frac),
                float(d_mean_min_enemy),
                d_obj,
                near_dragon,
                near_baron,
                near_herald,
                near_atakhan,
                near_horde,
                d_tower,
                in_tower_range,
                near_tower_radius,
                float(z_top),
                float(z_mid),
                float(z_bot),
                float(z_riv),
                float(z_jng),
            ],
            dtype=np.float32,
        )
        out[t, :] = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return out

