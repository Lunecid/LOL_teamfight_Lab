"""Named configuration presets.

A preset is the complete set of definition-level constants for one corpus version.  Select one
with ``LOL_CFG_PRESET=<name>`` (applied before ``LOL_CFG_OVERRIDES``) or call
``apply_preset(cfg, name)``.  The CFG defaults are the CoG 2026 values, so the tagged paper
state reproduces without any environment; every later corpus names its preset explicitly and
records it in the build manifest.
"""
from __future__ import annotations

from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    # CoG 2026 submission (tag v1.0-cog2026): hand-set detector constants, Eq.3 label,
    # legacy feature path (match-length time_norm, match-wide anchors).
    "cog2026": {
        "TF2_KILL_CLUSTER_GAP_MS": 18000, "CLUSTER_MAX_DIAMETER": 4000.0,
        "TF2_VALIDITY_RADIUS": 1800.0, "TF2_ENGAGE_PRE_KILL_MS": 10000, "FIGHT_HORIZON_SEC": 30,
        "LABEL_TYPE": "attention_value_win", "LABEL_EVENT_ATTRIBUTION": "window",
        "TIME_NORM_ABSOLUTE": False, "ANCHORS_CAUSAL": False, "TAB_FRAME_AGE_FEATURE": False,
    },
    # ToG extension, corpus v3.3 (docs/DEFINITION_EVIDENCE.md sections 17, 21, 22): data-derived
    # G / D, rule-anchored R / B, priced and engagement-attributed market label, clean feature path.
    "v3.3": {
        "TF2_KILL_CLUSTER_GAP_MS": 13700, "CLUSTER_MAX_DIAMETER": 4264.0,
        "TF2_VALIDITY_RADIUS": 1600.0, "TF2_ENGAGE_PRE_KILL_MS": 15000, "FIGHT_HORIZON_SEC": 35,
        "LABEL_TYPE": "market_event", "LABEL_EVENT_ATTRIBUTION": "engagement", "LABEL_GOLD_DEADZONE": 300.0,
        "TIME_NORM_ABSOLUTE": True, "TIME_NORM_DENOM_MIN": 45.0, "ANCHORS_CAUSAL": True, "TAB_FRAME_AGE_FEATURE": True,
    },
}


def apply_preset(c, name: str) -> Dict[str, Any]:
    """Set every field of the named preset on ``c``; returns the values applied."""
    key = str(name or "").strip()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; known: {sorted(PRESETS)}")
    values = PRESETS[key]
    for k, v in values.items():
        if not hasattr(c, k):
            raise KeyError(f"preset {key!r} sets unknown CFG field {k!r}")
        setattr(c, k, v)
    return dict(values)
