from __future__ import annotations

from core.config import cfg, F_GLOBAL, MAP_MAX, GLOBAL_IDX, NODE_IDX
from core.common import Any, Dict, Optional, Tuple, np
from core.contract import _BOOL_NODE_COLS
from core.interpolation import remap_alpha


def _interp_xy_guarded(
    cache: Dict[str, Any],
    i: int,
    j: int,
    alpha: float,
    *,
    coord_div: float,
) -> Optional[np.ndarray]:
    xy = cache.get("xy_raw_minute", None)
    if not isinstance(xy, np.ndarray) or xy.ndim != 3 or xy.shape[2] != 2:
        return None

    xi = xy[i].astype(np.float32)
    xj = xy[j].astype(np.float32)

    th = float(getattr(cfg, "XY_DISCONT_DIST_RAW", 7000.0))
    th2 = th * th
    guard_mode = str(getattr(cfg, "XY_GUARD_MODE", "midstep")).lower()
    use_alive = bool(getattr(cfg, "XY_DISCONT_USE_ALIVE", True))

    curve = str(getattr(cfg, "INTERP_XY_CURVE", "linear")).lower().strip()
    exp_k = float(getattr(cfg, "INTERP_EXP_K", 3.0))
    curved_alpha = remap_alpha(alpha, curve=curve, k=exp_k)

    alive_idx = NODE_IDX.get("alive", None)
    alive_i = None
    alive_j = None
    if use_alive and alive_idx is not None:
        alive_i = cache["node_minute"][i][:, alive_idx].astype(np.float32)
        alive_j = cache["node_minute"][j][:, alive_idx].astype(np.float32)

    out = np.zeros((10, 2), np.float32)
    for p in range(10):
        dx = float(xj[p, 0] - xi[p, 0])
        dy = float(xj[p, 1] - xi[p, 1])
        dist2 = dx * dx + dy * dy

        disc = dist2 > th2
        if use_alive and alive_i is not None and alive_j is not None:
            if (alive_i[p] > 0.5) != (alive_j[p] > 0.5):
                disc = True

        if disc:
            if guard_mode == "hold":
                xr, yr = xi[p, 0], xi[p, 1]
            else:
                xr, yr = (xi[p, 0], xi[p, 1]) if curved_alpha < 0.5 else (xj[p, 0], xj[p, 1])
        else:
            xr = (1.0 - curved_alpha) * xi[p, 0] + curved_alpha * xj[p, 0]
            yr = (1.0 - curved_alpha) * xi[p, 1] + curved_alpha * xj[p, 1]

        out[p, 0] = float(np.clip(xr / max(1e-6, coord_div), 0.0, 2.0))
        out[p, 1] = float(np.clip(yr / max(1e-6, coord_div), 0.0, 2.0))

    return out


def interpolate_node_global(cache: Dict[str, Any], q_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    ts = cache["minute_ts"]
    nm = cache["node_minute"]
    gm = cache["global_minute"]

    coord_div = float(getattr(cfg, "COORD_NORM_DIV", MAP_MAX))

    if len(ts) == 1:
        node = nm[0].astype(np.float32)
        glob = gm[0].astype(np.float32)
    else:
        idx = int(np.searchsorted(ts, q_ms) - 1)
        i = max(0, idx)
        j = min(len(ts) - 1, idx + 1)

        if ts[j] == ts[i]:
            alpha = 0.0
        else:
            alpha = float(q_ms - ts[i]) / float(ts[j] - ts[i])
            alpha = float(np.clip(alpha, 0.0, 1.0))

        scalars_method = str(getattr(cfg, "INTERP_SCALARS_METHOD", "ffill")).lower()
        if scalars_method in ("ffill", "hold", "step", "zero_order"):
            node = nm[i].copy()
            glob = gm[i].copy()
        elif scalars_method in ("bfill",):
            node = nm[j].copy()
            glob = gm[j].copy()
        else:
            exp_k = float(getattr(cfg, "INTERP_EXP_K", 3.0))
            s_alpha = remap_alpha(alpha, curve=scalars_method, k=exp_k)
            node = ((1.0 - s_alpha) * nm[i] + s_alpha * nm[j]).astype(np.float32)
            glob = ((1.0 - s_alpha) * gm[i] + s_alpha * gm[j]).astype(np.float32)

        if bool(getattr(cfg, "INTERP_XY", True)):
            xy_norm = _interp_xy_guarded(cache, i, j, alpha, coord_div=coord_div)
            if xy_norm is not None:
                xj_idx = NODE_IDX.get("x_norm", None)
                yj_idx = NODE_IDX.get("y_norm", None)
                if xj_idx is not None:
                    node[:, xj_idx] = xy_norm[:, 0]
                if yj_idx is not None:
                    node[:, yj_idx] = xy_norm[:, 1]

    # ── [P0-1] Dual-path XY handling ──────────────────────────
    # Path 1 (node_seq for GNN/spatial): preserve XY, optionally
    #   convert to centroid-relative coordinates.
    # Path 2 (extra_seq for BiGRU): XY removal handled downstream
    #   in features.py via ZERO_XY_IN_EXTRA_SEQ.
    xj_idx = NODE_IDX.get("x_norm", None)
    yj_idx = NODE_IDX.get("y_norm", None)

    if bool(getattr(cfg, "ZERO_XY_NODE_FEATURES", False)):
        # Legacy behavior: zero out all XY in node_seq
        if xj_idx is not None:
            node[:, xj_idx] = 0.0
        if yj_idx is not None:
            node[:, yj_idx] = 0.0
    elif bool(getattr(cfg, "USE_RELATIVE_XY", True)):
        # [P0-1 NEW] Convert absolute XY → centroid-relative XY
        # pos_i^{rel} = pos_i - centroid(all_10_players)
        # This removes map-position bias while preserving inter-player
        # spatial structure (team separation, formation shape).
        if xj_idx is not None and yj_idx is not None:
            xy = node[:, [xj_idx, yj_idx]]  # (10, 2)
            alive_idx_rel = NODE_IDX.get("alive", None)
            if alive_idx_rel is not None:
                alive_w = node[:, alive_idx_rel].reshape(-1, 1)  # (10, 1)
                alive_sum = max(float(alive_w.sum()), 1e-6)
                centroid = (xy * alive_w).sum(axis=0) / alive_sum  # (2,)
            else:
                centroid = xy.mean(axis=0)  # (2,)
            node[:, xj_idx] -= centroid[0]
            node[:, yj_idx] -= centroid[1]

    # Ensure time_norm is set properly when XY is zeroed
    if bool(getattr(cfg, "ZERO_XY_NODE_FEATURES", False)):
        tj = GLOBAL_IDX.get("time_norm", None)
        if tj is not None:
            t0 = float(ts[0])
            t1 = float(ts[-1])
            if t1 > t0:
                glob[tj] = float(np.clip((float(q_ms) - t0) / (t1 - t0), 0.0, 1.0))

    alive_idx = NODE_IDX.get("alive", None)
    if alive_idx is not None:
        node[:, alive_idx] = np.clip(node[:, alive_idx], 0.0, 1.0)

    if _BOOL_NODE_COLS:
        node[:, _BOOL_NODE_COLS] = np.clip(node[:, _BOOL_NODE_COLS], 0.0, 1.0)

    for k in ("baron_remain_norm", "elder_remain_norm"):
        j = NODE_IDX.get(k, None)
        if j is not None:
            node[:, j] = np.clip(node[:, j], 0.0, 1.0)

    return node.astype(np.float32), glob.astype(np.float32)


def _prev_snapshot_idx(ts: np.ndarray, ref_ms: int, *, strict_before: bool = True) -> int:
    if ts is None or len(ts) == 0:
        return -1
    side = "left" if strict_before else "right"
    idx = int(np.searchsorted(ts, int(ref_ms), side=side) - 1)
    if idx < 0:
        return -1
    return int(min(idx, len(ts) - 1))


def global_from_prev_snapshot(cache: Dict[str, Any], ref_ms: int, *, strict_before: bool = True) -> Tuple[np.ndarray, int]:
    ts = cache["minute_ts"]
    gm = cache["global_minute"]
    if ts is None or len(ts) == 0 or gm is None or len(gm) == 0:
        return np.zeros((F_GLOBAL,), dtype=np.float32), -1
    idx = _prev_snapshot_idx(ts, int(ref_ms), strict_before=bool(strict_before))
    if idx < 0:
        return np.zeros((F_GLOBAL,), dtype=np.float32), -1
    return gm[int(idx)].astype(np.float32, copy=True), int(ts[int(idx)])

