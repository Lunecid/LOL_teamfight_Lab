"""LOL_CFG_OVERRIDES must reach a fresh interpreter (and therefore multiprocessing workers)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNIPPET = "from core.config import cfg; print(cfg.TF2_VALIDITY_RADIUS, cfg.TF2_ENGAGE_PRE_KILL_MS, cfg.RUN_DIRNAME)"


def _run(overrides):
    env = dict(os.environ)
    env["LOL_CFG_OVERRIDES"] = json.dumps(overrides)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return subprocess.run([sys.executable, "-c", SNIPPET], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=300)


def test_overrides_apply_with_type_coercion():
    r = _run({"TF2_VALIDITY_RADIUS": 1600, "TF2_ENGAGE_PRE_KILL_MS": 15000.0, "RUN_DIRNAME": "runs_x/y"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[:3] == ["1600.0", "15000", "runs_x/y"]


def test_unknown_field_fails_loudly():
    r = _run({"TF2_NO_SUCH_FIELD": 1})
    assert r.returncode != 0
    assert "TF2_NO_SUCH_FIELD" in r.stderr


def test_fight_index_signature_includes_presence_gate():
    from data.index_split import _FIGHT_INDEX_CACHE_CFG_KEYS
    assert "TF2_VALIDITY_RADIUS" in _FIGHT_INDEX_CACHE_CFG_KEYS
    assert "TF2_ENGAGE_PRE_KILL_MS" in _FIGHT_INDEX_CACHE_CFG_KEYS
