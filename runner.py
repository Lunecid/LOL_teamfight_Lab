from __future__ import annotations

import argparse
from typing import List

from config import cfg
from experiment import run
from utils import set_seed


def _parse_model_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = x.replace(";", ",")
        return [t.strip() for t in x.split(",") if t.strip()]
    if isinstance(x, (list, tuple)):
        out = []
        for t in x:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                out.append(s)
        return out
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        return []


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default=str(cfg.MODE), help="all | build_cache | index | train | report")
    ap.add_argument(
        "--feature_set",
        type=str,
        default="full",
        help="global_only | global_events | node_personal | full | tri_modal",
    )
    ap.add_argument("--max_matches", type=int, default=int(cfg.MAX_MATCHES) if cfg.MAX_MATCHES else 0)
    ap.add_argument("--seed", type=int, default=int(cfg.SEEDS[0]) if cfg.SEEDS else 7)
    ap.add_argument("--models", type=str, default="", help="comma-separated model list override")
    ap.add_argument("--filter_loadable", action="store_true", help="filter splits by build_ms_sequence success")

    # Ablation control
    ap.add_argument(
        "--ablation_mode",
        type=str,
        default=str(getattr(cfg, "ABLATION_MODE", "baseline_plus")),
        choices=["baseline_plus", "both", "as_is"],
    )
    ap.add_argument("--require_lgbm", action="store_true", help="stop if baseline logits are required but baseline not available")

    # Fusion control
    ap.add_argument("--no_factorial_fusion", action="store_true")
    ap.add_argument("--rnn_model", type=str, default="")
    ap.add_argument("--gnn_model", type=str, default="")

    # Stacking mode
    ap.add_argument(
        "--stacking_mode",
        type=str,
        default=str(getattr(cfg, "STACKING_MODE", "simple")),
        choices=["simple", "oof"],
    )
    ap.add_argument("--oof_folds", type=int, default=int(getattr(cfg, "OOF_FOLDS", 5)))
    ap.add_argument("--oof_inner_val_ratio", type=float, default=float(getattr(cfg, "OOF_INNER_VAL_RATIO", 0.10)))
    ap.add_argument("--oof_skip_deep", action="store_true", help="OOF: only baseline OOF; deep OOF skipped (filled zeros)")
    ap.add_argument("--oof_meta", type=str, default=str(getattr(cfg, "OOF_META", "logreg")), choices=["logreg", "lgbm"])
    ap.add_argument("--oof_max_folds", type=int, default=int(getattr(cfg, "OOF_MAX_FOLDS", 0)))

    # Patch holdout split
    ap.add_argument(
        "--split_mode",
        type=str,
        default=str(getattr(cfg, "SPLIT_MODE", "auto")),
        choices=["auto", "match_id", "patch_holdout"],
    )
    ap.add_argument("--train_patches", type=str, default=str(getattr(cfg, "TRAIN_PATCHES", "")))
    ap.add_argument("--test_patches", type=str, default=str(getattr(cfg, "TEST_PATCHES", "")))
    ap.add_argument("--val_patches", type=str, default=str(getattr(cfg, "VAL_PATCHES", "")))
    ap.add_argument("--val_ratio", type=float, default=float(getattr(cfg, "VAL_RATIO", 0.15)))

    # R sweep
    ap.add_argument("--r_core_sweep", type=str, default=str(getattr(cfg, "R_CORE_SWEEP", "")), help="e.g. 1400,1800,2200,2600")
    ap.add_argument("--r_core", type=float, default=float(getattr(cfg, "STANDOFF_RADIUS", 1800.0)))

    # speed flags
    ap.add_argument("--amp", action="store_true", help="enable AMP mixed precision")
    ap.add_argument("--torch_compile", action="store_true", help="enable torch.compile (PyTorch 2)")
    ap.add_argument("--num_workers", type=int, default=int(getattr(cfg, "NUM_WORKERS", 0)))
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--eval_num_workers", type=int, default=int(getattr(cfg, "EVAL_NUM_WORKERS", 0)))
    ap.add_argument("--prefetch_factor", type=int, default=int(getattr(cfg, "PREFETCH_FACTOR", 2)))
    ap.add_argument("--tf32", action="store_true", help="enable TF32 on supported NVIDIA GPUs")
    ap.add_argument("--no_tf32", action="store_true", help="disable TF32")
    ap.add_argument("--cache_match_packs_in_ram", action="store_true", help="enable RAM LRU for match packs")
    ap.add_argument("--cache_train_in_ram", action="store_true", help="preload TRAIN samples in RAM")
    ap.add_argument("--cache_eval_in_ram", action="store_true", help="preload VAL/TEST samples in RAM")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    # cfg updates (kept compatible with the old entrypoint)
    cfg.MODE = args.mode
    cfg.MAX_MATCHES = int(args.max_matches) if int(args.max_matches) > 0 else None  # [FIX P1-2] 0 → None

    seed = int(args.seed)
    set_seed(seed)

    cfg.AMP = bool(args.amp or getattr(cfg, "AMP", False))
    cfg.TORCH_COMPILE = bool(args.torch_compile or getattr(cfg, "TORCH_COMPILE", False))
    cfg.NUM_WORKERS = int(args.num_workers)
    if args.pin_memory:
        cfg.PIN_MEMORY = True
    if args.persistent_workers:
        cfg.PERSISTENT_WORKERS = True

    cfg.EVAL_NUM_WORKERS = int(getattr(args, 'eval_num_workers', getattr(cfg, 'EVAL_NUM_WORKERS', 0)))
    cfg.PREFETCH_FACTOR = int(getattr(args, 'prefetch_factor', getattr(cfg, 'PREFETCH_FACTOR', 2)))

    if bool(getattr(args, 'tf32', False)):
        cfg.TF32 = True
    if bool(getattr(args, 'no_tf32', False)):
        cfg.TF32 = False

    if bool(getattr(args, 'cache_match_packs_in_ram', False)):
        cfg.CACHE_MATCH_PACKS_IN_RAM = True
    if bool(getattr(args, 'cache_train_in_ram', False)):
        cfg.CACHE_TRAIN_SAMPLES_IN_RAM = True
    if bool(getattr(args, 'cache_eval_in_ram', False)):
        cfg.CACHE_EVAL_SAMPLES_IN_RAM = True

    # resolve model list
    model_list = _parse_model_list(getattr(cfg, "MODEL_LIST", []))
    if str(args.models).strip():
        model_list = _parse_model_list(args.models)

    # attach resolved list for experiment
    args.model_list = model_list

    run(args)


if __name__ == "__main__":
    main()