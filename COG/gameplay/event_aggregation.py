from __future__ import annotations

from typing import Any, Dict, Tuple

from core.config import EVENT_IDX, F_EVENT, ITEM_HASH_DIM
from core.common import np, safe_float
from data.events_index import _events_in_window


def aggregate_events(events_or_pack: Any, tm: Dict[int, int], s_ms: int, e_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast aggregation with event indexing:
      - If events_or_pack is a cache pack(dict) with events_ts, use binary search slicing.
      - Else fallback to scanning a list (backward compatible).
    """
    ev = np.zeros(F_EVENT, np.float32)
    h = np.zeros(ITEM_HASH_DIM, np.float32)

    if isinstance(events_or_pack, dict):
        evs = _events_in_window(events_or_pack, s_ms, e_ms)
    else:
        events = events_or_pack if isinstance(events_or_pack, list) else []
        evs = []
        for e in events:
            if not isinstance(e, dict):
                continue
            try:
                t = int(e.get("timestamp", -1) or -1)
            except Exception:
                continue
            if s_ms <= t < e_ms:
                evs.append(e)

    for e in evs:
        if not isinstance(e, dict):
            continue

        et = str(e.get("type", "")).upper()

        if et == "CHAMPION_KILL":
            tid = tm.get(int(e.get("killerId", 0) or 0), 0)
            if tid in (100, 200):
                k = f"kills_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                b = f"bounty_t{tid}"
                shutdown = safe_float(e.get("shutdownBounty", 0.0))
                if b in EVENT_IDX:
                    ev[EVENT_IDX[b]] += safe_float(e.get("bounty", 0.0)) + shutdown

                if shutdown > 0.0:
                    ks = f"shutdown_kill_t{tid}"
                    if ks in EVENT_IDX:
                        ev[EVENT_IDX[ks]] += 1.0

                streak = max(0.0, safe_float(e.get("killStreakLength", 0.0)))
                if streak > 0.0:
                    kk = f"killstreak_t{tid}"
                    if kk in EVENT_IDX:
                        ev[EVENT_IDX[kk]] += streak

        elif et == "ELITE_MONSTER_KILL":
            mt = str(e.get("monsterType", "")).upper()
            tid = int(e.get("killerTeamId", 0) or 0)
            tag = {
                "DRAGON": "dragon",
                "BARON_NASHOR": "baron",
                "RIFTHERALD": "herald",
                "ATAKHAN": "atakhan",
                "HORDE": "horde",
            }.get(mt)
            if tag and tid in (100, 200):
                k = f"{tag}_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                kb = f"obj_bounty_t{tid}"
                if kb in EVENT_IDX:
                    ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "BUILDING_KILL":
            bt = str(e.get("buildingType", "")).upper()
            victim_team = int(e.get("teamId", 0) or 0)
            if victim_team not in (100, 200):
                continue
            tid = 200 if victim_team == 100 else 100
            if "TOWER" in bt:
                k = f"tower_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
            elif "INHIBITOR" in bt:
                k = f"inhib_t{tid}"
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
            kb = f"obj_bounty_t{tid}"
            if kb in EVENT_IDX:
                ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "TURRET_PLATE_DESTROYED":
            victim_team = int(e.get("teamId", 0) or 0)
            if victim_team not in (100, 200):
                continue
            tid = 200 if victim_team == 100 else 100
            k = f"plate_t{tid}"
            if k in EVENT_IDX:
                ev[EVENT_IDX[k]] += 1.0
            kb = f"obj_bounty_t{tid}"
            if kb in EVENT_IDX:
                ev[EVENT_IDX[kb]] += max(0.0, safe_float(e.get("bounty", 0.0)))

        elif et == "CHAMPION_SPECIAL_KILL":
            tid = tm.get(int(e.get("killerId", 0) or 0), 0)
            if tid in (100, 200):
                kt = str(e.get("killType", "")).upper()
                if "MULTI" in kt:
                    km = f"multikill_t{tid}"
                    if km in EVENT_IDX:
                        mk = max(1.0, safe_float(e.get("multiKillLength", 1.0)))
                        ev[EVENT_IDX[km]] += mk
                if "ACE" in kt:
                    ka = f"ace_t{tid}"
                    if ka in EVENT_IDX:
                        ev[EVENT_IDX[ka]] += 1.0

        elif et in ("WARD_PLACED", "WARD_KILL"):
            pid = int(e.get("creatorId", 0) or e.get("killerId", 0) or 0)
            tid = tm.get(pid, 0)
            if tid in (100, 200):
                k = ("ward_placed_t" if et == "WARD_PLACED" else "ward_kill_t") + str(tid)
                if k in EVENT_IDX:
                    ev[EVENT_IDX[k]] += 1.0
                wt = str(e.get("wardType", "")).upper()
                is_control = "CONTROL" in wt
                if is_control:
                    kc = ("control_ward_placed_t" if et == "WARD_PLACED" else "control_ward_kill_t") + str(tid)
                    if kc in EVENT_IDX:
                        ev[EVENT_IDX[kc]] += 1.0

        elif et in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"):
            pid = int(e.get("participantId", 0) or 0)
            tid = tm.get(pid, 0)
            if tid in (100, 200):
                item_id = int(e.get("itemId", 0) or 0)
                if item_id > 0:
                    h[int(item_id * 2654435761) % ITEM_HASH_DIM] += 1.0
                tag = {
                    "ITEM_PURCHASED": "pur",
                    "ITEM_SOLD": "sold",
                    "ITEM_DESTROYED": "sold",
                    "ITEM_UNDO": "undo",
                }.get(et)
                if tag:
                    k = f"item_{tag}_t{tid}"
                    if k in EVENT_IDX:
                        ev[EVENT_IDX[k]] += 1.0

    return ev, h


def compute_per_player_items_all_bins(
    events_or_pack: Any,
    tm: Dict[int, int],
    start_ms: int,
    end_ms: int,
    bin_ms: int,
    L: int,
    item_hash_dim: int = 16,
) -> np.ndarray:
    """Compute per-player cumulative item hash for all bins in one event scan.

    Accumulates ITEM_PURCHASED/SOLD/DESTROYED/UNDO events from game start (0ms)
    through each bin end, producing a per-player hash vector.

    Parameters
    ----------
    events_or_pack : dict (cache pack) or list (raw events)
    tm : participantId -> teamId mapping
    start_ms : observation window start
    end_ms : observation window end
    bin_ms : bin size in ms
    L : number of bins
    item_hash_dim : per-player hash dimension

    Returns
    -------
    np.ndarray : shape (L, 10, item_hash_dim)
    """
    h = np.zeros((10, item_hash_dim), dtype=np.float32)
    result = np.zeros((L, 10, item_hash_dim), dtype=np.float32)

    # Collect item events from game start to observation window end
    if isinstance(events_or_pack, dict):
        all_evs = _events_in_window(events_or_pack, 0, end_ms)
    else:
        all_evs = [
            e for e in (events_or_pack or [])
            if isinstance(e, dict) and 0 <= int(e.get("timestamp", -1) or -1) < end_ms
        ]

    # Filter to item events and sort by timestamp
    item_events = []
    for e in all_evs:
        et = str(e.get("type", "")).upper()
        if et in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"):
            ts = int(e.get("timestamp", -1) or -1)
            if ts >= 0:
                item_events.append((ts, e))
    item_events.sort(key=lambda x: x[0])

    # Bin end timestamps
    bin_ends = [start_ms + (i + 1) * bin_ms for i in range(L)]

    ev_idx = 0
    for bin_i in range(L):
        b_end = bin_ends[bin_i]
        while ev_idx < len(item_events) and item_events[ev_idx][0] < b_end:
            ts, e = item_events[ev_idx]
            pid = int(e.get("participantId", 0) or 0)
            if 1 <= pid <= 10:
                item_id = int(e.get("itemId", 0) or 0)
                if item_id > 0:
                    et = str(e.get("type", "")).upper()
                    hash_idx = int(item_id * 2654435761) % item_hash_dim
                    player_idx = pid - 1
                    if et == "ITEM_PURCHASED":
                        h[player_idx, hash_idx] += 1.0
                    elif et in ("ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"):
                        h[player_idx, hash_idx] = max(0.0, h[player_idx, hash_idx] - 1.0)
            ev_idx += 1
        result[bin_i] = h.copy()

    return result
