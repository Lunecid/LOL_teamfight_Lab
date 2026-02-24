from __future__ import annotations

from core.common import Any, Dict, Optional
from core.fight_types import ref_key

# [FIX-IMPORT] cfg must be imported explicitly from config
from core.config import cfg

def _normalize_logit_maps(
    lgbm_logit_map: Optional[Dict[str, float]] = None,
    logit_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Normalize different input styles into:
      {"lgbm_logit": {ref_key: float}, ...}

    Rules:
      - explicit logit_maps has priority for matching keys
      - lgbm_logit_map is backward-compatible addition
    """
    out: Dict[str, Dict[str, float]] = {}

    if isinstance(logit_maps, dict):
        for k, v in logit_maps.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, dict):
                continue
            vv: Dict[str, float] = {}
            for rk, val in v.items():
                try:
                    vv[str(rk)] = float(val)
                except Exception:
                    continue
            out[k] = vv

    if lgbm_logit_map is not None and "lgbm_logit" not in out:
        vv = {}
        if isinstance(lgbm_logit_map, dict):
            for rk, val in lgbm_logit_map.items():
                try:
                    vv[str(rk)] = float(val)
                except Exception:
                    continue
        out["lgbm_logit"] = vv

    return out

def _cfg_wants_logits(model_name: str = "") -> bool:
    """
    Decide if current run/model should have *_logit keys in batch.
    Uses cfg.* hints + model_name.
    """
    toks = [str(model_name or "").lower()]
    for attr in ("DEEP_MODEL", "MODEL", "MODEL_NAME", "ARCH", "NET"):
        if hasattr(cfg, attr):
            toks.append(str(getattr(cfg, attr) or "").lower())
    s = " ".join(toks)
    return any(k in s for k in ("tablogit", "tab_logit", "logit"))
