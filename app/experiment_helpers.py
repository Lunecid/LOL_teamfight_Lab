from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import cfg
from core.utils import write_log

# Memory-safe subsampling cap for tabular baseline.
LGBM_MAX_TRAIN: int = 100_000


def subsample_refs_for_lgbm(
    refs: List[Any],
    max_n: int,
    seed: int,
    log_fp=None,
) -> List[Any]:
    if len(refs) <= max_n:
        return refs

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(refs), max_n, replace=False)
    idx.sort()
    sampled = [refs[i] for i in idx]

    if log_fp:
        write_log(
            f"[LGBM-SUBSAMPLE] {len(refs):,} -> {len(sampled):,} "
            f"(max_n={max_n:,}, seed={seed})",
            log_fp,
        )
    return sampled


def subsample_refs_for_deep(
    refs: List[Any],
    max_n: int,
    seed: int,
    log_fp=None,
) -> List[Any]:
    if len(refs) <= max_n:
        return refs

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(refs), max_n, replace=False)
    idx.sort()
    sampled = [refs[i] for i in idx]

    if log_fp:
        write_log(
            f"[DEEP-SUBSAMPLE] {len(refs):,} -> {len(sampled):,} "
            f"(max_n={max_n:,}, seed={seed})",
            log_fp,
        )
    return sampled


def resolve_alias(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    aliases = getattr(cfg, "MODEL_ALIASES", {}) or {}
    key = n.lower()
    mapped = aliases.get(key, n)
    return str(mapped)


def infer_rnn_gnn_models(model_list: List[str]) -> Tuple[str, str]:
    rnn = ""
    gnn = ""

    for m in model_list:
        ml = (m or "").lower()

        if not rnn and any(
            k in ml
            for k in (
                "rnn",
                "gru",
                "lstm",
                "bigr",
                "bilstm",
                "transformer",
                "tcn",
                "temporal",
                "multiscale",
                "ms_",
                "stride",
                "dilated",
            )
        ):
            rnn = m

        if not gnn and any(
            k in ml
            for k in (
                "gnn",
                "gcn",
                "sage",
                "stgnn",
                "graphtransformer",
                "gat",
                "gatv2",
                "mpnn",
                "stgcn",
                "st-gcn",
                "dyn",
                "dynamic",
                "edge",
                "edgeaware",
            )
        ):
            gnn = m

        if rnn and gnn:
            break

    return rnn, gnn


def infer_feature_set_for_model(model_name: str, default_fs: str) -> str:
    return default_fs


def layered_logit_override(model_name: str) -> Optional[bool]:
    m = (model_name or "").strip().lower()
    if not m.startswith(("layered_fusion", "fusion_layered")):
        return None
    if "@" not in m:
        return None

    try:
        tail = m.split("@", 1)[1]
    except Exception:
        return None
    if not tail:
        return None

    bool_map = {
        "1": True,
        "true": True,
        "on": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "off": False,
        "no": False,
        "n": False,
    }

    parts = [p.strip() for p in tail.replace(",", "+").split("+") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            continue
        key = (k or "").strip().lower()
        val = (v or "").strip().lower()
        if key not in ("logit", "lgbm", "use_logit", "use_lgbm_logit"):
            continue
        if val.startswith("logit_"):
            val = val[len("logit_") :]
        if val.startswith("lgbm_"):
            val = val[len("lgbm_") :]
        return bool_map.get(val, None)
    return None


def needs_lgbm_logit(model_name: str) -> bool:
    m = (model_name or "").lower()
    if m.startswith(("layered_fusion", "fusion_layered")):
        ovr = layered_logit_override(m)
        return bool(ovr) if ovr is not None else False
    if m.startswith(("fusion_", "lgbm_dual_")):
        return True
    if "tablogit" in m:
        return True
    if m.startswith(("logit_",)) or m.endswith(("_logit",)):
        return True
    return False


def best_variant_for(model_name: str, deep_reports: Dict[str, Any]) -> Optional[str]:
    best = None
    best_auc = -1.0
    for k, rep in deep_reports.items():
        if not k.startswith(model_name + "::"):
            continue
        try:
            auc = float(rep.get("metrics", {}).get("val", {}).get("auc", -1.0))
        except Exception:
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best = k.split("::", 1)[1]
    return best


def normalize_split_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if not m or m == "auto":
        m = str(getattr(cfg, "SPLIT_MODE", "multi_patch")).strip().lower()
    if m in ("match_id", "match", "group", "group_match"):
        return "group_match"
    if m in ("random", "rand"):
        return "random"
    if m in ("patch_forward", "forward_patch", "patch_time"):
        return "patch_forward"
    if m in ("patch_holdout", "holdout_patch"):
        return "patch_holdout"
    if m in ("multi_patch", "stratified"):
        return "multi_patch"
    return "multi_patch"
