from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gameplay.fight_postmerge import safe_int


def finalize_kill_cluster(kills: List[dict]) -> dict:
    participants: set = set()
    for k in kills:
        kid = safe_int(k.get("killer_id", 0))
        vid = safe_int(k.get("victim_id", 0))
        if 1 <= kid <= 10:
            participants.add(kid)
        if 1 <= vid <= 10:
            participants.add(vid)
        for aid in k.get("assisting_ids", []) or []:
            a = safe_int(aid)
            if 1 <= a <= 10:
                participants.add(a)

    first_pos = kills[0].get("position")
    if first_pos is not None:
        cx, cy = float(first_pos[0]), float(first_pos[1])
    else:
        positions = [k["position"] for k in kills if k.get("position")]
        if positions:
            cx = float(np.mean([p[0] for p in positions]))
            cy = float(np.mean([p[1] for p in positions]))
        else:
            cx, cy = 0.0, 0.0

    return {
        "kills": kills,
        "first_kill_ts": int(kills[0]["timestamp"]),
        "last_kill_ts": int(kills[-1]["timestamp"]),
        "fight_center": (cx, cy),
        "participants": participants,
        "n_kills": len(kills),
    }


def cluster_kills_temporal(
    kill_events: List[dict],
    gap_ms: int,
) -> List[dict]:
    if not kill_events:
        return []

    sorted_kills = sorted(kill_events, key=lambda k: int(k["timestamp"]))
    clusters: List[dict] = []
    current_kills: List[dict] = [sorted_kills[0]]

    for i in range(1, len(sorted_kills)):
        gap = int(sorted_kills[i]["timestamp"]) - int(sorted_kills[i - 1]["timestamp"])
        if gap <= gap_ms:
            current_kills.append(sorted_kills[i])
        else:
            clusters.append(finalize_kill_cluster(current_kills))
            current_kills = [sorted_kills[i]]

    if current_kills:
        clusters.append(finalize_kill_cluster(current_kills))

    return clusters


def split_kill_cluster_spatial(cluster: dict, max_diameter: float) -> List[dict]:
    if not isinstance(cluster, dict):
        return []

    kills = list(cluster.get("kills", []) or [])
    if len(kills) <= 1 or float(max_diameter) <= 0.0:
        return [cluster] if kills else []

    with_pos: List[Tuple[int, dict, float, float]] = []
    without_pos: List[Tuple[int, dict]] = []
    for i, k in enumerate(kills):
        pos = k.get("position", None) if isinstance(k, dict) else None
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            try:
                with_pos.append((i, k, float(pos[0]), float(pos[1])))
            except Exception:
                without_pos.append((i, k))
        else:
            without_pos.append((i, k))

    if len(with_pos) <= 1:
        return [finalize_kill_cluster(kills)]

    r = float(max_diameter)
    r2 = r * r
    n = len(with_pos)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        _, _, xi, yi = with_pos[i]
        for j in range(i + 1, n):
            _, _, xj, yj = with_pos[j]
            dx = xi - xj
            dy = yi - yj
            if (dx * dx + dy * dy) <= r2:
                adj[i].append(j)
                adj[j].append(i)

    comp_nodes: List[List[int]] = []
    seen = [False] * n
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nx in adj[v]:
                if not seen[nx]:
                    seen[nx] = True
                    stack.append(nx)
        comp_nodes.append(comp)

    if len(comp_nodes) <= 1 and not without_pos:
        return [finalize_kill_cluster(kills)]

    comp_payload: List[List[Tuple[int, dict]]] = []
    comp_first_ts: List[int] = []
    for comp in comp_nodes:
        pairs = [(with_pos[idx][0], with_pos[idx][1]) for idx in comp]
        pairs.sort(key=lambda t: int(t[1].get("timestamp", 0)))
        comp_payload.append(pairs)
        comp_first_ts.append(int(pairs[0][1].get("timestamp", 0)) if pairs else 10**18)

    for orig_idx, k in without_pos:
        ts = int(k.get("timestamp", 0)) if isinstance(k, dict) else 0
        best_ci = 0
        best_dt = 10**18
        for ci, cts in enumerate(comp_first_ts):
            dt = abs(int(ts) - int(cts))
            if dt < best_dt:
                best_dt = dt
                best_ci = ci
        comp_payload[best_ci].append((orig_idx, k))

    out: List[dict] = []
    for pairs in comp_payload:
        if not pairs:
            continue
        pairs.sort(key=lambda t: int(t[1].get("timestamp", 0)))
        out.append(finalize_kill_cluster([k for _, k in pairs]))

    out.sort(key=lambda c: int(c.get("first_kill_ts", 0)))
    return out


def build_5s_position_grid(
    xy_minute: np.ndarray,
    minute_ts: np.ndarray,
    kill_events: List[dict],
    tm: Dict[int, int],
    *,
    cfg_obj: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(len(minute_ts))
    if T < 2:
        return minute_ts.copy(), xy_minute[:, :, :2].astype(np.float32, copy=True)

    step_ms = int(getattr(cfg_obj, "TF2_GRID_STEP_MS", 5000)) if cfg_obj is not None else 5000
    if step_ms <= 0:
        step_ms = 5000
    use_frame_interp = bool(getattr(cfg_obj, "TF2_USE_FRAME_INTERP", True)) if cfg_obj is not None else True
    use_kill_traj_interp = bool(getattr(cfg_obj, "TF2_USE_KILL_TRAJECTORY_INTERP", True)) if cfg_obj is not None else True

    t_start = int(minute_ts[0])
    t_end = int(minute_ts[-1])
    dense_ts = np.arange(t_start, t_end + step_ms, step_ms, dtype=np.int64)
    Td = int(len(dense_ts))

    left = np.searchsorted(minute_ts, dense_ts, side="right") - 1
    left = np.clip(left, 0, T - 2)
    right = left + 1

    tL = minute_ts[left].astype(np.float64)
    tR = minute_ts[right].astype(np.float64)
    denom = np.maximum(tR - tL, 1.0)
    alpha_raw = ((dense_ts.astype(np.float64) - tL) / denom).astype(np.float32)
    alpha_raw = np.clip(alpha_raw, 0.0, 1.0)
    alpha = alpha_raw.reshape(Td, 1, 1)

    xy_only = xy_minute[:, :, :2].astype(np.float32)
    xyL = xy_only[left]
    xyR = xy_only[right]
    if use_frame_interp:
        xy_dense = ((1.0 - alpha) * xyL + alpha * xyR).astype(np.float32)
    else:
        xy_dense = xyL.astype(np.float32, copy=True)

    if (not kill_events) or (not use_kill_traj_interp):
        return dense_ts, xy_dense

    for kill in kill_events:
        kill_ts_val = int(kill["timestamp"])
        kill_pos = kill.get("position")
        if kill_pos is None:
            continue
        kill_x, kill_y = float(kill_pos[0]), float(kill_pos[1])

        participants = set()
        kid = safe_int(kill.get("killer_id", 0))
        vid = safe_int(kill.get("victim_id", 0))
        if 1 <= kid <= 10:
            participants.add(kid - 1)
        if 1 <= vid <= 10:
            participants.add(vid - 1)
        for aid in kill.get("assisting_ids", []) or []:
            a = safe_int(aid)
            if 1 <= a <= 10:
                participants.add(a - 1)

        if not participants:
            continue

        m_idx = int(np.searchsorted(minute_ts, kill_ts_val, side="right")) - 1
        m_idx = max(0, min(m_idx, T - 1))
        prior_frame_ts = int(minute_ts[m_idx])

        d_start = int(np.searchsorted(dense_ts, prior_frame_ts, side="left"))
        d_end = int(np.searchsorted(dense_ts, kill_ts_val, side="right"))
        d_start = max(0, d_start)
        d_end = min(Td, d_end)

        if d_end <= d_start:
            continue

        interval_ts = dense_ts[d_start:d_end].astype(np.float64)
        interval_dur = max(1.0, float(kill_ts_val - prior_frame_ts))
        a_vec = np.clip((interval_ts - float(prior_frame_ts)) / interval_dur, 0.0, 1.0).astype(np.float32)

        for p_idx in participants:
            if p_idx < 0 or p_idx >= 10:
                continue
            start_x = float(xy_only[m_idx, p_idx, 0])
            start_y = float(xy_only[m_idx, p_idx, 1])
            xy_dense[d_start:d_end, p_idx, 0] = start_x + a_vec * (kill_x - start_x)
            xy_dense[d_start:d_end, p_idx, 1] = start_y + a_vec * (kill_y - start_y)

    return dense_ts, xy_dense
