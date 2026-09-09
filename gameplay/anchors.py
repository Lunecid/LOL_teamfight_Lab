"""Causal map anchors for the spatial features.

The cache's ``meta["anchors"]`` lists the positions of every tower that lost plates or fell
and every objective that was killed *anywhere in the match*, so "distance to the nearest
tower" read from it knows which towers fall after the cutoff.  This module rebuilds the
anchors from the static map (``config/game_rules/map_anchors.json``, medians of event
positions over 2,000 matches) and the events at or before the cutoff only: a tower is an
anchor while it is still standing at the cutoff, and objective anchors are the fixed pits.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAP_ANCHORS_PATH = PROJECT_ROOT / "config" / "game_rules" / "map_anchors.json"
OBJ_KEYS = {"DRAGON": "DRAGON", "BARON_NASHOR": "BARON", "RIFTHERALD": "RIFTHERALD", "ATAKHAN": "ATAKHAN", "HORDE": "HORDE"}


@lru_cache(maxsize=1)
def load_static_anchors(path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path) if path else MAP_ANCHORS_PATH
    data = json.load(open(p, encoding="utf-8"))
    towers = [t for t in data["towers"] if str(t.get("building", "")).upper() == "TOWER_BUILDING"]
    objectives = {OBJ_KEYS.get(k, k): [[float(s["x"]), float(s["y"])] for s in v] for k, v in data["objectives"].items()}
    return {"towers": towers, "objectives": objectives}


def _tower_key(team: int, lane: str, tower: str) -> tuple:
    return (int(team), str(lane).upper(), str(tower).upper())


def standing_towers(events: List[dict], cutoff_ms: int, static: Optional[Dict[str, Any]] = None) -> Dict[str, List[List[float]]]:
    """Towers still standing at ``cutoff_ms`` (inclusive), per owning team, in game units."""
    static = static or load_static_anchors()
    destroyed: Dict[tuple, int] = {}
    for e in events or []:
        if str(e.get("type", "")).upper() != "BUILDING_KILL":
            continue
        if int(e.get("timestamp", 0) or 0) > int(cutoff_ms):
            continue
        if str(e.get("buildingType", "")).upper() != "TOWER_BUILDING":
            continue
        key = _tower_key(int(e.get("teamId", 0) or 0), str(e.get("laneType", "")), str(e.get("towerType", "")))
        pos = e.get("position")
        # nexus turrets share a key; remember the destroyed one's position to drop the right slot
        if key[2] == "NEXUS_TURRET" and isinstance(pos, dict):
            destroyed[key + (round(float(pos.get("x", 0)) / 500.0),)] = 1
        else:
            destroyed[key] = 1
    out: Dict[str, List[List[float]]] = {"TOWER_T100": [], "TOWER_T200": []}
    for t in static["towers"]:
        key = _tower_key(t["team"], t["lane"], t["tower"])
        if key[2] == "NEXUS_TURRET":
            gone = (key + (round(float(t["x"]) / 500.0),)) in destroyed
        else:
            gone = key in destroyed
        if not gone:
            out[f"TOWER_T{int(t['team'])}"].append([float(t["x"]), float(t["y"])])
    return out


def causal_anchors(cache: Dict[str, Any], cutoff_ms: int) -> Dict[str, Any]:
    """Anchors usable at the cutoff: fixed objective pits and towers standing at the cutoff."""
    static = load_static_anchors()
    return {"obj": {k: [list(p) for p in v] for k, v in static["objectives"].items()},
            "tower": standing_towers(cache.get("events") or [], int(cutoff_ms), static)}
