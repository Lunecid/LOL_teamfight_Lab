"""Presets set a complete named definition; the env hook applies it before overrides."""
import json
import os
import subprocess
import sys
from pathlib import Path

from core.config import CFG
from core.presets import PRESETS, apply_preset

REPO = Path(__file__).resolve().parents[1]


def test_v33_preset_values():
    c = CFG()
    vals = apply_preset(c, "v3.3")
    assert c.TF2_KILL_CLUSTER_GAP_MS == 13700 and c.CLUSTER_MAX_DIAMETER == 4264.0
    assert c.TF2_VALIDITY_RADIUS == 1600.0 and c.TF2_ENGAGE_PRE_KILL_MS == 15000
    assert c.LABEL_TYPE == "market_event" and c.LABEL_EVENT_ATTRIBUTION == "engagement"
    assert set(vals) == set(PRESETS["v3.3"])


def test_defaults_are_cog2026():
    c = CFG()
    for k, v in PRESETS["cog2026"].items():
        if k in ("TIME_NORM_ABSOLUTE", "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE", "LABEL_EVENT_ATTRIBUTION"):
            continue  # feature-path fixes are on by default; the cog2026 preset switches them off explicitly
        assert getattr(c, k) == v, k


def test_env_preset_then_overrides():
    env = dict(os.environ)
    env["LOL_CFG_PRESET"] = "v3.3"
    env["LOL_CFG_OVERRIDES"] = json.dumps({"FIGHT_HORIZON_SEC": 40})
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    r = subprocess.run([sys.executable, "-c", "from core.config import cfg; print(cfg.TF2_KILL_CLUSTER_GAP_MS, cfg.FIGHT_HORIZON_SEC)"],
                       cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.split()[:2] == ["13700", "40"]
