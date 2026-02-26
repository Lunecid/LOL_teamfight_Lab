from __future__ import annotations

import argparse
import itertools
import sys
from typing import Dict, List, Optional, Tuple

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


_FUSION_GLOBAL_ORDER: Tuple[str, ...] = ("ugru", "bigru", "ulstm", "bilstm", "transformer", "tcn", "mamba")
_FUSION_GNN_ORDER: Tuple[str, ...] = ("gcn", "graphsage", "graphtransformer", "gatv2", "mpnn")
_FUSION_EVENT_ORDER: Tuple[str, ...] = ("attn", "xattn", "mean")
_FUSION_LOGIT_ORDER: Tuple[str, ...] = ("1", "0")

_FUSION_GLOBAL_ALIASES: Dict[str, str] = {
    "gru": "ugru",
    "ugru": "ugru",
    "bigru": "bigru",
    "lstm": "ulstm",
    "ulstm": "ulstm",
    "bilstm": "bilstm",
    "transformer": "transformer",
    "tcn": "tcn",
    "mamba": "mamba",
}
_FUSION_GNN_ALIASES: Dict[str, str] = {
    "gcn": "gcn",
    "graphsage": "graphsage",
    "sage": "graphsage",
    "gnnsage": "graphsage",
    "graphtransformer": "graphtransformer",
    "gat": "gatv2",
    "gatv2": "gatv2",
    "mpnn": "mpnn",
}
_FUSION_EVENT_ALIASES: Dict[str, str] = {
    "attn": "attn",
    "xattn": "xattn",
    "event_xattn": "xattn",
    "mean": "mean",
    "avg": "mean",
    "pool": "mean",
}
_FUSION_LOGIT_ALIASES: Dict[str, str] = {
    "1": "1",
    "true": "1",
    "on": "1",
    "yes": "1",
    "y": "1",
    "with": "1",
    "use": "1",
    "0": "0",
    "false": "0",
    "off": "0",
    "no": "0",
    "n": "0",
    "without": "0",
    "drop": "0",
}

_PAPER_CORE4_MODELS = ",".join(
    [
        "rnn_bigru",
        "gnn_graphsage",
        "rnn_transformer",
        "layered_fusion@global=bigru+gnn=graphsage+event=attn+logit=1",
    ]
)

_PAPER_CORE4_OPTIMAL_MODELS = ",".join(
    [
        "rnn_bigru",
        "gnn_graphsage",
        "event_xattn",
        "layered_fusion@global=bigru+gnn=graphsage+event=xattn+logit=1",
    ]
)


def _stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _axis_key_from_token(tok: str) -> Optional[Tuple[str, str]]:
    t = (tok or "").strip().lower()
    if not t:
        return None

    if "=" in t:
        k, v = t.split("=", 1)
        return k.strip(), v.strip()
    if ":" in t:
        k, v = t.split(":", 1)
        return k.strip(), v.strip()

    if t in ("rnn_all", "global_all"):
        return "rnn", "all"
    if t in ("gnn_all", "graph_all"):
        return "gnn", "all"
    if t in ("event_all",):
        return "event", "all"
    if t in ("logit_all", "lgbm_all"):
        return "logit", "all"

    if t.startswith(("rnn_", "global_")):
        return "rnn", t
    if t.startswith("gnn_"):
        return "gnn", t
    if t.startswith(("event_", "attn_")):
        return "event", t
    if t.startswith(("logit_", "lgbm_")):
        return "logit", t

    if t in _FUSION_GLOBAL_ALIASES:
        return "rnn", t
    if t in _FUSION_GNN_ALIASES:
        return "gnn", t
    if t in _FUSION_EVENT_ALIASES:
        return "event", t
    if t in _FUSION_LOGIT_ALIASES:
        return "logit", t

    return None


def _normalize_fusion_axis_key(k: str) -> Optional[str]:
    key = (k or "").strip().lower()
    if key in ("rnn", "global", "g"):
        return "rnn"
    if key in ("gnn", "graph", "n"):
        return "gnn"
    if key in ("event", "attn", "e"):
        return "event"
    if key in ("logit", "lgbm", "use_logit", "use_lgbm_logit", "l"):
        return "logit"
    return None


def _expand_fusion_axis_values(axis: str, raw_val: str) -> List[str]:
    v = (raw_val or "").strip().lower()
    if axis == "rnn":
        if v in ("all", "rnn_all", "global_all"):
            return list(_FUSION_GLOBAL_ORDER)
        if v.startswith("global_"):
            v = v[len("global_") :]
        if v.startswith("rnn_"):
            v = v[len("rnn_") :]
        x = _FUSION_GLOBAL_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion rnn/global token: {raw_val!r}")

    if axis == "gnn":
        if v in ("all", "gnn_all", "graph_all"):
            return list(_FUSION_GNN_ORDER)
        if v.startswith("gnn_"):
            v = v[len("gnn_") :]
        x = _FUSION_GNN_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion gnn token: {raw_val!r}")

    if axis == "event":
        if v in ("all", "event_all"):
            return list(_FUSION_EVENT_ORDER)
        if v.startswith("event_"):
            x = _FUSION_EVENT_ALIASES.get(v)
            if x is not None:
                return [x]
            v = v[len("event_") :]
        if v.startswith("attn_"):
            v = v[len("attn_") :]
        x = _FUSION_EVENT_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion event token: {raw_val!r}")

    if axis == "logit":
        if v in ("all", "logit_all", "lgbm_all"):
            return list(_FUSION_LOGIT_ORDER)
        if v.startswith("logit_"):
            v = v[len("logit_") :]
        elif v.startswith("lgbm_"):
            v = v[len("lgbm_") :]
        x = _FUSION_LOGIT_ALIASES.get(v)
        if x is not None:
            return [x]
        raise ValueError(f"Unsupported fusion logit token: {raw_val!r}")

    raise ValueError(f"Unsupported fusion axis: {axis!r}")


def _expand_single_fusion_expr(expr: str) -> List[str]:
    parts = [p.strip() for p in str(expr or "").split("+") if p.strip()]
    if not parts:
        return []

    axis_vals: Dict[str, List[str]] = {"rnn": [], "gnn": [], "event": [], "logit": []}

    for p in parts:
        kv = _axis_key_from_token(p)
        if kv is None:
            raise ValueError(f"Could not parse fusion token: {p!r}")
        raw_k, raw_v = kv
        axis = _normalize_fusion_axis_key(raw_k)
        if axis is None:
            raise ValueError(f"Unsupported fusion axis key: {raw_k!r}")
        vals = _expand_fusion_axis_values(axis, raw_v)
        for v in vals:
            if v not in axis_vals[axis]:
                axis_vals[axis].append(v)

    rnn_vals = axis_vals["rnn"] or [None]
    gnn_vals = axis_vals["gnn"] or [None]
    event_vals = axis_vals["event"] or [None]
    logit_vals = axis_vals["logit"] or [None]

    out: List[str] = []
    for rnn_v, gnn_v, event_v, logit_v in itertools.product(rnn_vals, gnn_vals, event_vals, logit_vals):
        spec_parts: List[str] = []
        if rnn_v is not None:
            spec_parts.append(f"global=rnn_{rnn_v}")
        if gnn_v is not None:
            spec_parts.append(f"gnn=gnn_{gnn_v}")
        if event_v is not None:
            event_token = "event_xattn" if event_v == "xattn" else event_v
            spec_parts.append(f"event={event_token}")
        if logit_v is not None:
            spec_parts.append(f"logit={logit_v}")

        if spec_parts:
            out.append("layered_fusion@" + "+".join(spec_parts))
        else:
            out.append("layered_fusion")

    return _stable_unique(out)


def _expand_fusion_exprs(raw: str) -> List[str]:
    exprs = _parse_model_list(raw)
    out: List[str] = []
    for expr in exprs:
        out.extend(_expand_single_fusion_expr(expr))
    return _stable_unique(out)


def _apply_paper_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "paper_preset", "none") or "none").strip().lower()
    if preset in ("", "none", "off", "false", "0"):
        return

    paper_seed = int(getattr(args, "paper_seed", 7))
    paper_max_matches = int(getattr(args, "paper_max_matches", 0))

    # Core objective:
    #   1) RNN, GNN, Attention, 3-way fusion
    #   2) single-seed fast path for paper iteration
    args.seed = int(paper_seed)
    args.feature_set = "full"
    if preset in ("core4_optimal", "core4_optimal_fast"):
        args.models = _PAPER_CORE4_OPTIMAL_MODELS
    else:
        args.models = _PAPER_CORE4_MODELS
    args.ablation_mode = "baseline_plus"
    args.require_lgbm = True
    args.no_factorial_fusion = True
    args.stacking_mode = "simple"
    args.oof_skip_deep = True

    # Hardware-aware speed defaults for RTX 50xx class.
    speed_raw = str(getattr(args, "speed_profile", "none") or "none").strip().lower()
    if speed_raw in ("none", "off", ""):
        args.speed_profile = "rtx5080"
    args.amp = True
    args.tf32 = True
    args.no_tf32 = False
    args.cache_match_packs_in_ram = True
    args.cache_eval_in_ram = True

    if preset in ("core4_1seed_fast", "core4_optimal_fast"):
        # Fast triage mode: cap matches unless user explicitly sets one.
        if paper_max_matches > 0:
            args.max_matches = int(paper_max_matches)
        elif int(getattr(args, "max_matches", 0)) <= 0:
            args.max_matches = 600
    elif paper_max_matches > 0:
        args.max_matches = int(paper_max_matches)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified experiment runner (default pipeline mode).",
        epilog=(
            "Ablation runner is also available in this same entrypoint.\n"
            "Use one of:\n"
            "  python runner.py ablation --phase 1\n"
            "  python runner.py --phase 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    ap.add_argument(
        "--paper_preset",
        type=str,
        default="none",
        choices=["none", "core4_1seed", "core4_1seed_fast", "core4_optimal", "core4_optimal_fast"],
        help=(
            "paper presets: "
            "core4_1seed uses transformer attention; "
            "core4_optimal uses event_xattn + layered_fusion(xattn) for stronger attention modeling. "
            "*_fast additionally caps max_matches for quick triage."
        ),
    )
    ap.add_argument("--paper_seed", type=int, default=7, help="seed used by --paper_preset")
    ap.add_argument(
        "--paper_max_matches",
        type=int,
        default=0,
        help="optional max_matches override for --paper_preset (0 keeps preset/default behavior)",
    )

    # Ablation control
    ap.add_argument(
        "--ablation_mode",
        type=str,
        default=str(getattr(cfg, "ABLATION_MODE", "baseline_plus")),
        choices=["baseline_plus", "both", "as_is"],
    )
    ap.add_argument("--require_lgbm", action="store_true", help="stop if baseline logits are required but baseline not available")

    # Fusion control
    ap.add_argument(
        "--fusion",
        type=str,
        default="",
        help=(
            "layered-fusion sweep expression(s). "
            "Example: rnn_ugru+gnn_all, rnn_all+gnn_gcn+event_all+logit_all"
        ),
    )
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


def _looks_like_ablation_argv(argv: List[str]) -> bool:
    if not argv:
        return False
    if argv[0].lower() in ("ablation", "abl"):
        return True
    return any(tok == "--phase" or tok.startswith("--phase=") for tok in argv)


def _run_ablation_from_argv(argv: List[str]) -> None:
    from app.experiment_runner_io import build_parser as _build_ablation_parser
    from experiment_runner import run_phase_cli as _run_phase_cli

    ab_parser = _build_ablation_parser()
    ab_args = ab_parser.parse_args(argv)
    _run_phase_cli(ab_args)


def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    if raw_argv and raw_argv[0].lower() in ("experiment", "exp", "run"):
        raw_argv = raw_argv[1:]

    if raw_argv and raw_argv[0].lower() in ("ablation", "abl"):
        _run_ablation_from_argv(raw_argv[1:])
        return

    if _looks_like_ablation_argv(raw_argv):
        _run_ablation_from_argv(raw_argv)
        return

    ap = build_argparser()
    args = ap.parse_args(raw_argv)
    _apply_paper_preset(args)

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
    models_overridden = bool(str(args.models).strip())
    if models_overridden:
        model_list = _parse_model_list(args.models)

    fusion_raw = str(getattr(args, "fusion", "")).strip()
    if fusion_raw:
        try:
            fusion_models = _expand_fusion_exprs(fusion_raw)
        except ValueError as e:
            ap.error(str(e))
            return
        if fusion_models:
            if models_overridden:
                model_list.extend(fusion_models)
            else:
                # If --fusion is provided without --models, run the expanded fusion grid only.
                model_list = list(fusion_models)

    model_list = _stable_unique(model_list)

    # attach resolved list for experiment
    args.model_list = model_list

    run(args)


if __name__ == "__main__":
    main()
