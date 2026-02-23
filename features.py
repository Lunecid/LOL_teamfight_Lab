# features.py (patched to match config feature contract)
from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from improvements import compute_momentum_stats, compute_game_phase_seq, GAME_PHASE_FEATURE_NAMES
# [P4-SHADOW FIX] NODE_IDX removed from this import — was immediately
# shadowed at module level by FEATURE_CONTRACT.node_idx (Issue #1).
# [P2-STRUCT-1] Now import NODE_IDX directly from config (SSoT).
from config import (
    cfg,
    CHAMPION_STATS_KEYS, DAMAGE_STATS_KEYS,
    NODE_FEATURE_NAMES, SLOT_NAMES,
    GLOBAL_FEATURE_NAMES, EVENT_FEATURE_NAMES,
    ITEM_HASH_NAMES, FEATURE_CONTRACT, F_NODE,
    MAP_MAX, CS_DENOM, DS_DENOM,
    DRAGON_PIT_XY, BARON_PIT_XY,
    NODE_IDX, EVENT_IDX, GLOBAL_IDX,  # [P2-STRUCT-1] SSoT: single import path
)

from common import safe_float, log1p_norm


# ---------------------------------------------------------------------
# Optional import: deterministic denoms for snapshot normalization
# ---------------------------------------------------------------------
try:
    from config import NODE_BASE_DENOM as _NODE_BASE_DENOM  # type: ignore
    NODE_BASE_DENOM: Dict[str, float] = dict(_NODE_BASE_DENOM)
except Exception:
    # Fallback if config doesn't export it (keeps backward compatibility)
    NODE_BASE_DENOM = dict(
        level=18.0,
        xp=20000.0,
        curGold=4000.0,
        totalGold=25000.0,
        gps=30.0,
        laneCS=400.0,
        jgCS=250.0,
        ccTime=600.0,
    )


# ─────────────────────────────────────────────────────────────
# [P2-STRUCT-1] NODE_IDX and EVENT_IDX are now imported directly
# from config.py (single source of truth).
#
# Previously this module created independent dict() copies:
#   NODE_IDX = dict(FEATURE_CONTRACT.node_idx)
#   EVENT_IDX = dict(FEATURE_CONTRACT.event_idx)
# This violated SSoT and risked silent drift.
# ─────────────────────────────────────────────────────────────


def _find_node_idx(cands: List[str], default: Optional[int] = None) -> Optional[int]:
    for c in cands:
        if c in NODE_IDX:
            return int(NODE_IDX[c])
    return default


# ---------------------------------------------------------------------
# Spatial extra features (per time-step)
# ---------------------------------------------------------------------
SPATIAL_FEATURE_NAMES: List[str] = [
    "pos_fight_x_norm", "pos_fight_y_norm",
    "pos_blue_x_norm", "pos_blue_y_norm",
    "pos_red_x_norm",  "pos_red_y_norm",
    "dist_team_sep_norm",

    "standoff_pairs_frac",
    "mean_min_enemy_dist_norm",
    "d_standoff_pairs_frac",
    "d_mean_min_enemy_dist_norm",

    "dist_obj_nearest_norm",
    "near_obj_dragon", "near_obj_baron", "near_obj_herald", "near_obj_atakhan", "near_obj_horde",

    "dist_tower_nearest_norm",
    "in_tower_range",
    "near_tower_radius",

    "zone_top_lane", "zone_mid_lane", "zone_bot_lane", "zone_river", "zone_jungle",
]
F_SPATIAL: int = len(SPATIAL_FEATURE_NAMES)


def get_spatial_feature_names() -> List[str]:
    return list(SPATIAL_FEATURE_NAMES)


# ---------------------------------------------------------------------
# Default (fallback) anchors in map units
# ---------------------------------------------------------------------
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
            return (0, 0, 0, 1, 0)  # river

    if au <= 1100.0:
        return (0, 1, 0, 0, 0)  # mid

    top_edge = (y >= MAP_MAX * 0.70) or (x <= MAP_MAX * 0.30)
    bot_edge = (y <= MAP_MAX * 0.30) or (x >= MAP_MAX * 0.70)

    if u < 0 and top_edge:
        return (1, 0, 0, 0, 0)  # top
    if u > 0 and bot_edge:
        return (0, 0, 1, 0, 0)  # bot

    return (0, 0, 0, 0, 1)  # jungle


def _compute_standoff_signals(
    blue_xy: np.ndarray,
    red_xy: np.ndarray,
    blue_alive: np.ndarray,
    red_alive: np.ndarray,
    standoff_r_norm: float,
) -> Tuple[float, float]:
    bmask = (blue_alive > 0.5)
    rmask = (red_alive > 0.5)
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



def _compute_spatial_batch(node_role_seq: np.ndarray, xi: int, yi: int, ai: Optional[int],
                           standoff_r_norm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray]:
    """[FIX-PERF] Vectorised spatial computations â€” O(L) numpy instead of O(LÃ—NÂ²) Python loop.

    Returns (fight_xy, blue_c, red_c, team_sep, pairs_frac_arr, mean_min_enemy_arr)
    each of shape (L, ...).
    """
    L = node_role_seq.shape[0]
    xy = node_role_seq[:, :, [xi, yi]].astype(np.float32)  # (L,10,2)
    alive = node_role_seq[:, :, ai].astype(np.float32) if ai is not None else np.ones((L, 10), np.float32)

    w = alive[:, :, None]  # (L,10,1)
    wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-6)  # (L,1,1)
    fight_xy = (xy * w).sum(axis=1) / wsum.squeeze(-1)  # (L,2)

    blue_xy = xy[:, :5]  # (L,5,2)
    red_xy = xy[:, 5:]
    blue_w = w[:, :5]
    red_w = w[:, 5:]
    blue_sum = np.maximum(blue_w.sum(axis=1, keepdims=True), 1e-6)
    red_sum = np.maximum(red_w.sum(axis=1, keepdims=True), 1e-6)
    blue_c = (blue_xy * blue_w).sum(axis=1) / blue_sum.squeeze(-1)  # (L,2)
    red_c = (red_xy * red_w).sum(axis=1) / red_sum.squeeze(-1)

    team_sep = np.sqrt(((blue_c - red_c) ** 2).sum(axis=1))  # (L,)

    # Pairwise distances: (L, 5, 5)
    d = blue_xy[:, :, None, :] - red_xy[:, None, :, :]  # (L,5,5,2)
    dist = np.sqrt((d * d).sum(axis=-1))  # (L,5,5)

    b_alive = (alive[:, :5] > 0.5)  # (L,5) bool
    r_alive = (alive[:, 5:] > 0.5)
    pair_mask = b_alive[:, :, None] & r_alive[:, None, :]  # (L,5,5)

    # Standoff pairs fraction
    close = ((dist <= standoff_r_norm) & pair_mask).sum(axis=(1, 2)).astype(np.float32)
    total_pairs = pair_mask.sum(axis=(1, 2)).astype(np.float32)
    pairs_frac = np.where(total_pairs > 0, close / total_pairs, 0.0)

    # Mean min enemy distance
    big_val = np.float32(1e6)
    dist_masked = np.where(pair_mask, dist, big_val)
    b_near = np.where(r_alive[:, None, :].any(axis=2), dist_masked.min(axis=2), big_val)
    r_near = np.where(b_alive[:, :, None].any(axis=1), dist_masked.min(axis=1), big_val)
    all_near = np.concatenate([b_near, r_near], axis=1)  # (L,10)
    valid_near = (all_near < big_val)
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

        row = np.asarray([
            fight_xy[0], fight_xy[1],
            blue_c[0], blue_c[1],
            red_c[0],  red_c[1],
            team_sep,

            float(pairs_frac),
            float(mean_min_enemy),
            float(d_pairs_frac),
            float(d_mean_min_enemy),

            d_obj,
            near_dragon, near_baron, near_herald, near_atakhan, near_horde,

            d_tower,
            in_tower_range,
            near_tower_radius,

            float(z_top), float(z_mid), float(z_bot), float(z_riv), float(z_jng),
        ], dtype=np.float32)

        out[t, :] = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return out


# ---------------------------------------------------------------------
# Snapshot extraction (keys aligned with config)
# ---------------------------------------------------------------------
def extract_participant_snapshot(pf: dict) -> Dict[str, float]:
    if not isinstance(pf, dict):
        pf = {}

    pos = pf.get("position") or {}
    if not isinstance(pos, dict):
        pos = {}

    csobj = pf.get("championStats") or {}
    if not isinstance(csobj, dict):
        csobj = {}

    dsobj = pf.get("damageStats") or {}
    if not isinstance(dsobj, dict):
        dsobj = {}

    # Ã¢Å“â€¦ internal names aligned to config (curGold/gps/laneCS/jgCS/ccTime)
    out: Dict[str, float] = {
        "pos_x": safe_float(pos.get("x", 0.0)),
        "pos_y": safe_float(pos.get("y", 0.0)),
        "level": safe_float(pf.get("level", 0.0)),
        "xp": safe_float(pf.get("xp", 0.0)),

        "curGold": safe_float(pf.get("currentGold", pf.get("curGold", 0.0))),
        "totalGold": safe_float(pf.get("totalGold", 0.0)),
        "gps": safe_float(pf.get("goldPerSecond", pf.get("gps", 0.0))),

        "laneCS": safe_float(pf.get("minionsKilled", pf.get("laneCS", 0.0))),
        "jgCS": safe_float(pf.get("jungleMinionsKilled", pf.get("jgCS", 0.0))),
        "ccTime": safe_float(pf.get("timeEnemySpentControlled", pf.get("ccTime", 0.0))),

        "hp_pct": float(np.clip(
            safe_float(csobj.get("health", 0.0)) / max(1.0, safe_float(csobj.get("healthMax", 1.0))),
            0.0, 1.0
        )),
        "mp_pct": float(np.clip(
            safe_float(csobj.get("power", 0.0)) / max(1.0, safe_float(csobj.get("powerMax", 1.0))),
            0.0, 1.0
        )),
    }

    for k in CHAMPION_STATS_KEYS:
        out[f"cs_{k}"] = safe_float(csobj.get(k, 0.0))
    for k in DAMAGE_STATS_KEYS:
        out[f"ds_{k}"] = safe_float(dsobj.get(k, 0.0))

    return out


def snapshot_to_node_features(snap: Dict[str, float], alive: float) -> np.ndarray:
    """snapshot 딕셔너리를 노드 피처 벡터로 변환.

    .. deprecated::
        [P2-1] 이 함수는 cs_/ds_ 정규화에 log1p_norm을 사용하며,
        실제 실행 경로인 pipeline.py의 linear clip 방식과 불일치합니다.

        정규화 방식 비교:
            pipeline.py:  v' = clip(v / denom, -10, 10)    ← 실사용 경로 (표준)
            features.py:  v' = ln(1+|v|) / ln(1+d) · sign(v)  ← 본 함수 (비표준)

        수학적 차이:
            선형 정규화는 분포 형태를 보존하고 극값만 절단.
            log1p 정규화는 큰 값을 로그 스케일로 압축 → 꼬리 분포에서 상이한 입력 생성.

        이 함수를 사용하면 pipeline.py로 생성된 데이터와 분포가 불일치하여
        모델 성능이 저하될 수 있습니다. pipeline.py의 정규화 로직을 사용하세요.

    Parameters
    ----------
    snap : Dict[str, float]
        선수 스냅샷 딕셔너리
    alive : float
        생존 여부 (0.0 or 1.0)

    Returns
    -------
    np.ndarray
        shape=(F_NODE,) 노드 피처 벡터
    """
    import warnings
    warnings.warn(
        "[P2-1] snapshot_to_node_features() uses log1p_norm for cs_/ds_ stats, "
        "which is inconsistent with pipeline.py's linear clip normalization. "
        "Use pipeline.py's normalization path for production. "
        "This function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )

    div = _coord_div()

    # ---- coordinate (never fitted-scaled) ----
    x_norm = safe_float(snap.get("pos_x", snap.get("x", 0.0))) / div
    y_norm = safe_float(snap.get("pos_y", snap.get("y", 0.0))) / div

    # ---- denoms (config-aligned) ----
    d_xp = float(NODE_BASE_DENOM.get("xp", 20000.0))
    d_cur = float(NODE_BASE_DENOM.get("curGold", 4000.0))
    d_tot = float(NODE_BASE_DENOM.get("totalGold", 25000.0))
    d_gps = float(NODE_BASE_DENOM.get("gps", 30.0))
    d_lane = float(NODE_BASE_DENOM.get("laneCS", 400.0))
    d_jg = float(NODE_BASE_DENOM.get("jgCS", 250.0))
    d_cc = float(NODE_BASE_DENOM.get("ccTime", 600.0))

    # Ã¢Å“â€¦ base keys exactly matching config's NODE_SNAPSHOT_FEATURE_NAMES
    base: Dict[str, float] = {
        "x_norm": x_norm,
        "y_norm": y_norm,

        "level_norm": safe_float(snap.get("level", 0.0)) / 18.0,
        "xp_norm": log1p_norm(safe_float(snap.get("xp", 0.0)), denom=d_xp),

        "curGold_norm": log1p_norm(safe_float(snap.get("curGold", snap.get("currentGold", 0.0))), denom=d_cur),
        "totalGold_norm": log1p_norm(safe_float(snap.get("totalGold", 0.0)), denom=d_tot),
        "gps_norm": safe_float(snap.get("gps", snap.get("goldPerSecond", 0.0))) / max(1e-6, d_gps),

        "laneCS_norm": log1p_norm(safe_float(snap.get("laneCS", snap.get("minionsKilled", 0.0))), denom=d_lane),
        "jgCS_norm": log1p_norm(safe_float(snap.get("jgCS", snap.get("jungleMinionsKilled", 0.0))), denom=d_jg),
        "ccTime_norm": log1p_norm(safe_float(snap.get("ccTime", snap.get("timeEnemySpentControlled", 0.0))), denom=d_cc),

        "hp_pct": float(snap.get("hp_pct", 0.0)),
        "mp_pct": float(snap.get("mp_pct", 0.0)),
        "alive": float(alive),
    }

    # ---- status passthrough (if snap already contains them, keep them) ----
    # e.g., has_baron, baron_remain_norm, soul_infernal, flash_ready, flash_remain_norm, vision_* ...
    for k, v in snap.items():
        if k in NODE_IDX and k not in base:
            base[k] = safe_float(v)

    # ---- stat keys (config: cs_* / ds_*) ----
    for k in CHAMPION_STATS_KEYS:
        denom = float(CS_DENOM.get(k, 1000.0))
        v = safe_float(snap.get(f"cs_{k}", 0.0))
        vv = float(np.clip(v, 0.0, 1.0)) if abs(denom - 1.0) < 1e-9 else log1p_norm(v, denom)
        base[f"cs_{k}"] = vv

    for k in DAMAGE_STATS_KEYS:
        v = safe_float(snap.get(f"ds_{k}", 0.0))
        vv = log1p_norm(v, float(DS_DENOM.get(k, 200000.0)))
        base[f"ds_{k}"] = vv

    # ---- fill vector by contract order ----
    out = np.zeros((len(NODE_FEATURE_NAMES),), dtype=np.float32)
    for i, name in enumerate(NODE_FEATURE_NAMES):
        if name in base:
            out[i] = float(base[name])
            continue

        # backward-compat aliases (if some older NODE_FEATURE_NAMES exist)
        # (We keep this to avoid silent breakage when switching branches.)
        if name == "currentGold_norm" and "curGold_norm" in base:
            out[i] = float(base["curGold_norm"]); continue
        if name == "goldPerSecond_norm" and "gps_norm" in base:
            out[i] = float(base["gps_norm"]); continue
        if name == "timeEnemySpentControlled_norm" and "ccTime_norm" in base:
            out[i] = float(base["ccTime_norm"]); continue

        # generic: if someone used raw-name without _norm in contract
        if name.endswith("_norm"):
            raw = name[:-5]
            if raw in base:
                out[i] = float(base[raw])
                continue

    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


# ---------------------------------------------------------------------
# Event minimal selection (used for node_personal)
# ---------------------------------------------------------------------
NODE_PERSONAL_EVENT_KEEP: List[str] = [
    "kills_t100", "kills_t200",
    "bounty_t100", "bounty_t200",
    "shutdown_kill_t100", "shutdown_kill_t200",
    "killstreak_t100", "killstreak_t200",
    "multikill_t100", "multikill_t200",
    "ace_t100", "ace_t200",
    "dragon_t100", "dragon_t200",
    "baron_t100", "baron_t200",
    "obj_bounty_t100", "obj_bounty_t200",
    "ward_placed_t100", "ward_placed_t200",
    "ward_kill_t100", "ward_kill_t200",
    "control_ward_placed_t100", "control_ward_placed_t200",
    "control_ward_kill_t100", "control_ward_kill_t200",
    "item_pur_t100", "item_pur_t200",
]


def minimal_event_seq(ev_seq: np.ndarray) -> np.ndarray:
    idx = [EVENT_IDX[k] for k in NODE_PERSONAL_EVENT_KEEP if k in EVENT_IDX]
    if len(idx) == 0:
        return np.zeros((ev_seq.shape[0], 0), dtype=np.float32)
    return ev_seq[:, idx].astype(np.float32)


# ---------------------------------------------------------------------
# Role reorder (robust, no "sum==0" heuristic)
# ---------------------------------------------------------------------
def reorder_node_seq_by_role(node_seq: np.ndarray, role_slots: Dict[int, int]) -> np.ndarray:
    node_seq = node_seq.astype(np.float32)
    if not role_slots:
        return node_seq

    out = np.zeros_like(node_seq, dtype=np.float32)
    filled = np.zeros((10,), dtype=bool)

    for pid, slot in role_slots.items():
        if isinstance(pid, (int, np.integer)) and isinstance(slot, (int, np.integer)):
            pid = int(pid); slot = int(slot)
            if 1 <= pid <= 10 and 0 <= slot < 10:
                out[:, slot, :] = node_seq[:, pid - 1, :]
                filled[slot] = True

    used_pid = {int(pid) for pid in role_slots.keys() if isinstance(pid, (int, np.integer))}
    remaining_pids = [pid for pid in range(1, 11) if pid not in used_pid]

    rp = 0
    for slot in range(10):
        if not filled[slot] and rp < len(remaining_pids):
            pid = remaining_pids[rp]
            out[:, slot, :] = node_seq[:, pid - 1, :]
            filled[slot] = True
            rp += 1

    return out


def reorder_ids_by_role(ids_10: Any, role_slots: Dict[int, int], fill_value: int = 0) -> np.ndarray:
    try:
        arr = np.asarray(ids_10)
    except Exception:
        return np.full((10,), fill_value, dtype=np.int64)

    if arr.size < 10:
        tmp = np.full((10,), fill_value, dtype=np.int64)
        tmp[:arr.size] = arr.astype(np.int64, copy=False)
        arr = tmp
    arr = arr.reshape(10).astype(np.int64, copy=False)

    if not role_slots:
        return arr

    out = np.full((10,), fill_value, dtype=np.int64)
    filled = np.zeros((10,), dtype=bool)

    for pid, slot in role_slots.items():
        if isinstance(pid, (int, np.integer)) and isinstance(slot, (int, np.integer)):
            pid = int(pid); slot = int(slot)
            if 1 <= pid <= 10 and 0 <= slot < 10:
                out[slot] = arr[pid - 1]
                filled[slot] = True

    used_pid = {int(pid) for pid in role_slots.keys() if isinstance(pid, (int, np.integer))}
    remaining = [pid for pid in range(1, 11) if pid not in used_pid]

    rp = 0
    for slot in range(10):
        if not filled[slot] and rp < len(remaining):
            pid = remaining[rp]
            out[slot] = arr[pid - 1]
            filled[slot] = True
            rp += 1

    return out


# ---------------------------------------------------------------------
# Sequence -> Tabular summary (tri_modal)
# ---------------------------------------------------------------------
def seq_to_tabular(x_seq: np.ndarray) -> np.ndarray:
    L, D = x_seq.shape
    t = np.arange(L, dtype=np.float32)

    if L >= 2:
        t0 = t - t.mean()
        denom = float((t0 * t0).sum() + 1e-6)
        slope = (t0[:, None] * (x_seq - x_seq.mean(axis=0))).sum(axis=0) / denom
    else:
        slope = np.zeros((D,), dtype=np.float32)

    last = x_seq[-1]
    mean = x_seq.mean(axis=0)
    std = x_seq.std(axis=0)
    mn = x_seq.min(axis=0)
    mx = x_seq.max(axis=0)
    delta = x_seq[-1] - x_seq[0]

    return np.concatenate([last, mean, std, mn, mx, delta, slope], axis=0).astype(np.float32)


# ---------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------
def _macro_base_names() -> List[str]:
    items = list(ITEM_HASH_NAMES) if bool(getattr(cfg, "USE_ITEMS", True)) else []
    return _global_base_names() + list(EVENT_FEATURE_NAMES) + items


def _global_base_names() -> List[str]:
    out = list(GLOBAL_FEATURE_NAMES)
    if bool(getattr(cfg, "USE_GAME_PHASE", False)):
        out += list(GAME_PHASE_FEATURE_NAMES)
    return out


def get_xseq_feature_names(feature_set: str) -> List[str]:
    spatial = get_spatial_feature_names()

    if feature_set == "global_only":
        return _global_base_names() + spatial

    if feature_set == "global_events":
        return _macro_base_names() + spatial

    node_names = [f"{slot}_{f}" for slot in SLOT_NAMES for f in NODE_FEATURE_NAMES]

    if feature_set == "node_personal":
        return node_names + list(NODE_PERSONAL_EVENT_KEEP) + spatial

    if feature_set == "full":
        return node_names + _macro_base_names() + spatial

    if feature_set == "tri_modal":
        raise ValueError("tri_modal has no x_seq; use get_tabular_feature_names_tri_modal().")

    raise ValueError(feature_set)


def get_extra_feature_names(feature_set: str) -> List[str]:
    spatial = get_spatial_feature_names()

    if feature_set == "global_only":
        return _global_base_names() + spatial

    if feature_set in ("global_events", "full", "tri_modal"):
        return _macro_base_names() + spatial

    if feature_set == "node_personal":
        return list(NODE_PERSONAL_EVENT_KEEP) + spatial

    raise ValueError(feature_set)


def get_gnn_extra_names(feature_set: str) -> List[str]:
    return get_extra_feature_names(feature_set)


def get_tabular_feature_names_tri_modal() -> List[str]:
    # [P4-STATS] Use centralized TABULAR_SUFFIXES (was hardcoded, Issue #5)
    from feature_contract import tabular_feature_names
    base = _macro_base_names() + get_spatial_feature_names()
    return list(tabular_feature_names(base))


# ---------------------------------------------------------------------
# Correlation pruning helper
# ---------------------------------------------------------------------
def prune_correlated_columns(
    X: np.ndarray,
    names: List[str],
    threshold: float,
    max_rows: int,
    seed: int,
    method: str = "hierarchical",
) -> Tuple[np.ndarray, List[str]]:
    """Remove highly correlated features.

    [FIX-MATH] Original greedy algorithm was ORDER-DEPENDENT:
      If corr(A,B)=0.99, corr(B,C)=0.99, corr(A,C)=0.5, different
      column orderings yield different surviving feature sets.

    New ``method="hierarchical"`` uses agglomerative clustering on
    the correlation-distance matrix, cutting at ``1 - threshold``.
    One representative per cluster is chosen (highest variance).

    Parameters
    ----------
    method : str
        ``"hierarchical"`` (default, order-invariant) or
        ``"greedy"`` (original, kept for backward compat).
    """
    if X.shape[1] <= 1:
        return np.arange(X.shape[1]), []

    Xs = X
    if X.shape[0] > max_rows:
        rs = np.random.RandomState(seed)
        Xs = X[rs.choice(X.shape[0], size=max_rows, replace=False)].astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(Xs, rowvar=False), 0.0)

    if method == "hierarchical":
        # --- Order-invariant hierarchical clustering ---
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist = 1.0 - np.abs(C)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 2.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

        # Pick highest-variance feature per cluster
        keep = np.zeros(X.shape[1], dtype=bool)
        variances = Xs.var(axis=0)
        for cl in np.unique(labels):
            members = np.where(labels == cl)[0]
            best = members[np.argmax(variances[members])]
            keep[best] = True
    else:
        # --- Original greedy (order-dependent, kept for compat) ---
        keep = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if keep[i]:
                jrel = np.where(np.abs(C[i, i + 1:]) >= threshold)[0]
                keep[jrel + (i + 1)] = False

    keep_idx = np.where(keep)[0]
    dropped = [names[j] for j in range(X.shape[1]) if not keep[j]]
    return keep_idx, dropped


# ---------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------
def build_sequence_features(
    sample: Dict[str, Any],
    team_map: Dict[int, int],
    role_slots: Dict[int, int],
    feature_set: str
) -> Dict[str, Any]:
    node_seq = sample["node_seq"]
    glob_seq = sample["glob_seq"]
    ev_seq   = sample["ev_seq"]
    item_seq = sample["item_seq"]

    node_role = reorder_node_seq_by_role(node_seq, role_slots)  # (L,10,F_NODE)
    node_flat = node_role.reshape(node_role.shape[0], -1).astype(np.float32)

    if bool(getattr(cfg, "USE_ITEMS", True)):
        macro_base = np.concatenate([glob_seq, ev_seq, item_seq], axis=1).astype(np.float32)
    else:
        macro_base = np.concatenate([glob_seq, ev_seq], axis=1).astype(np.float32)
    global_base = glob_seq.astype(np.float32)

    if bool(getattr(cfg, "USE_GAME_PHASE", False)):
        t_idx = GLOBAL_IDX.get("time_norm", None)
        if t_idx is not None and int(t_idx) < glob_seq.shape[1]:
            time_norm_seq = np.clip(glob_seq[:, int(t_idx)], 0.0, 1.0).astype(np.float32)
        else:
            time_norm_seq = np.linspace(0.0, 1.0, glob_seq.shape[0], dtype=np.float32)
        game_duration_min = float(sample.get("game_duration_min", 35.0))
        game_duration_min = max(1.0, game_duration_min)
        phase_seq = compute_game_phase_seq(
            time_norm_seq=time_norm_seq,
            total_game_minutes=game_duration_min,
            tau=float(getattr(cfg, "GAME_PHASE_TAU", 3.0)),
        ).astype(np.float32)
        macro_base = np.concatenate([macro_base, phase_seq], axis=1).astype(np.float32)
        global_base = np.concatenate([global_base, phase_seq], axis=1).astype(np.float32)

    spatial_seq = compute_spatial_seq_from_node(node_role, sample)  # (L, F_SPATIAL)

    macro_seq = np.concatenate([macro_base, spatial_seq], axis=1).astype(np.float32)
    global_plus_spatial = np.concatenate([global_base, spatial_seq], axis=1).astype(np.float32)

    y = int(sample["y"])

    def _attach_aux_targets(out: Dict[str, Any]) -> Dict[str, Any]:
        out["y_kill_diff"] = float(sample.get("y_kill_diff", 0.0))
        out["y_gold_diff"] = float(sample.get("y_gold_diff", 0.0))
        out["y_obj_diff"] = float(sample.get("y_obj_diff", 0.0))
        return out

    if feature_set == "global_only":
        return _attach_aux_targets(dict(
            x_seq=global_plus_spatial,
            extra_seq=global_plus_spatial,
            node_seq=node_role.astype(np.float32),
            y=y,
        ))

    if feature_set == "global_events":
        return _attach_aux_targets(dict(
            x_seq=macro_seq,
            extra_seq=macro_seq,
            node_seq=node_role.astype(np.float32),
            y=y,
        ))

    if feature_set == "node_personal":
        ev_min = minimal_event_seq(ev_seq)
        ev_min_plus = np.concatenate([ev_min, spatial_seq], axis=1).astype(np.float32)
        x_seq = np.concatenate([node_flat, ev_min_plus], axis=1).astype(np.float32)
        return _attach_aux_targets(dict(
            x_seq=x_seq,
            extra_seq=ev_min_plus,
            node_seq=node_role.astype(np.float32),
            y=y,
        ))

    if feature_set == "full":
        x_seq = np.concatenate([node_flat, macro_seq], axis=1).astype(np.float32)
        return _attach_aux_targets(dict(
            x_seq=x_seq,
            extra_seq=macro_seq,
            node_seq=node_role.astype(np.float32),
            y=y,
        ))

    if feature_set == "tri_modal":
        tab_x = seq_to_tabular(macro_seq)

        if bool(getattr(cfg, "USE_MOMENTUM_FEATURES", False)):
            mom = compute_momentum_stats(macro_seq)
            tab_x = np.concatenate([tab_x, mom], axis=0)
        return _attach_aux_targets(dict(
            node_seq=node_role.astype(np.float32),
            macro_seq=macro_seq.astype(np.float32),
            tab_x=tab_x.astype(np.float32),
            extra_seq=macro_seq.astype(np.float32),
            y=y,
        ))

    raise ValueError(f"Unknown feature_set={feature_set}")
