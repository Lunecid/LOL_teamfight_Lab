from __future__ import annotations

import argparse
from typing import List

from core.config import cfg
from app.experiment import run
from train.speed import apply_speed_profile
from core.utils import set_seed


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


def _normalize_split_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if not m or m == "auto":
        return "auto"
    if m in ("match_id", "match", "group", "group_match"):
        return "group_match"
    if m in ("multi_patch", "multi", "stratified"):
        return "multi_patch"
    if m in ("random", "rand"):
        return "random"
    if m in ("patch_forward", "forward_patch", "patch_time"):
        return "patch_forward"
    if m == "patch_holdout":
        return "patch_holdout"
    return m


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
    ap.add_argument(
        "--prediction_gap_ms",
        type=int,
        default=int(getattr(cfg, "PREDICTION_GAP_MS", 0)),
        help="observation ends this many ms before engage; label horizon still starts at engage",
    )
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
    ap.add_argument("--oof_meta", type=str, default="logreg", choices=["logreg"])
    ap.add_argument("--oof_max_folds", type=int, default=int(getattr(cfg, "OOF_MAX_FOLDS", 0)))
    ap.add_argument(
        "--allow_split_leakage",
        action="store_true",
        help="allow split overlap by match_id (not recommended; default is fail-fast)",
    )

    # Patch holdout split
    ap.add_argument(
        "--split_mode",
        type=str,
        default=str(getattr(cfg, "SPLIT_MODE", "auto")),
        choices=[
            "auto",
            "match_id",
            "group_match",
            "multi_patch",
            "random",
            "patch_forward",
            "patch_holdout",
        ],
    )
    ap.add_argument("--train_patches", type=str, default=str(getattr(cfg, "TRAIN_PATCHES", "")))
    ap.add_argument("--test_patches", type=str, default=str(getattr(cfg, "TEST_PATCHES", "")))
    ap.add_argument("--val_patches", type=str, default=str(getattr(cfg, "VAL_PATCHES", "")))
    ap.add_argument("--val_ratio", type=float, default=float(getattr(cfg, "VAL_RATIO", 0.15)))

    # R sweep
    ap.add_argument("--r_core_sweep", type=str, default=str(getattr(cfg, "R_CORE_SWEEP", "")), help="e.g. 1400,1800,2200,2600")
    ap.add_argument("--r_core", type=float, default=float(getattr(cfg, "STANDOFF_RADIUS", 1800.0)))

    # speed flags
    ap.add_argument(
        "--speed_profile",
        type=str,
        default="none",
        choices=["none", "auto", "rtx50", "rtx5080", "aggressive"],
        help="hardware speed preset",
    )
    ap.add_argument("--amp", action="store_true", help="enable AMP mixed precision")
    ap.add_argument("--amp_dtype", type=str, default=None, choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--torch_compile", action="store_true", help="enable torch.compile (PyTorch 2)")
    ap.add_argument("--compile_mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--compile_dynamic", action="store_true", help="enable dynamic shapes for torch.compile")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--eval_num_workers", type=int, default=None)
    ap.add_argument("--prefetch_factor", type=int, default=None)
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
    args.split_mode = _normalize_split_mode(getattr(args, "split_mode", "auto"))
    if args.split_mode != "auto":
        cfg.SPLIT_MODE = args.split_mode

    seed = int(args.seed)
    set_seed(seed)
    cfg.PREDICTION_GAP_MS = max(0, int(getattr(args, "prediction_gap_ms", 0)))

    profile = str(getattr(args, "speed_profile", "none"))
    if profile and profile.lower() not in ("none", "off"):
        cfg.SPEED_PROFILE = profile
        apply_speed_profile(cfg, profile=profile)

    cfg.AMP = bool(args.amp or getattr(cfg, "AMP", False))
    if args.amp_dtype is not None:
        cfg.AMP_DTYPE = str(args.amp_dtype).strip()
    cfg.TORCH_COMPILE = bool(args.torch_compile or getattr(cfg, "TORCH_COMPILE", False))
    if args.compile_mode is not None:
        cfg.TORCH_COMPILE_MODE = str(args.compile_mode).strip()
    if bool(getattr(args, "compile_dynamic", False)):
        cfg.TORCH_COMPILE_DYNAMIC = True

    if args.num_workers is not None:
        cfg.NUM_WORKERS = int(args.num_workers)
    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)
    if args.pin_memory:
        cfg.PIN_MEMORY = True
    if args.persistent_workers:
        cfg.PERSISTENT_WORKERS = True

    if args.eval_num_workers is not None:
        cfg.EVAL_NUM_WORKERS = int(args.eval_num_workers)
    if args.prefetch_factor is not None:
        cfg.PREFETCH_FACTOR = int(args.prefetch_factor)

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
