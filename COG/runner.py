from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from core.config import cfg
from app.experiment import run
from train.speed import apply_speed_profile
from core.utils import set_seed
from app.fusion_spec import (
    _parse_model_list,
    _FUSION_GLOBAL_ORDER,
    _FUSION_GNN_ORDER,
    _FUSION_EVENT_ORDER,
    _FUSION_LOGIT_ORDER,
    _FUSION_GLOBAL_ALIASES,
    _FUSION_GNN_ALIASES,
    _FUSION_EVENT_ALIASES,
    _FUSION_LOGIT_ALIASES,
    _stable_unique,
    _axis_key_from_token,
    _normalize_fusion_axis_key,
    _expand_fusion_axis_values,
    _expand_single_fusion_expr,
    _expand_fusion_exprs,
)


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
    if m in ("patch_holdout", "holdout_patch"):
        return "patch_holdout"
    return m


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
            "holdout_patch",
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

    # Dataset sharing — build datasets once, reuse across models
    ap.add_argument(
        "--share_datasets",
        action="store_true",
        help=(
            "Pre-build datasets once and share across all deep models. "
            "Eliminates repeated build_ms_sequence calls (~5h savings). "
            "Requires --cache_train_in_ram and --cache_eval_in_ram."
        ),
    )

    # Memory isolation
    ap.add_argument(
        "--isolate",
        action="store_true",
        default=False,
        help=(
            "각 모델을 별도 subprocess로 실행하여 메모리 격리 보장. "
            "프로세스 종료 시 OS가 메모리를 100%% 회수하므로 메모리 누수 원천 차단. "
            "PyCharm 등 IDE에서 메모리 크래시가 발생할 때 사용. "
            "예: python runner.py --models rnn_bigru,gnn_graphsage --isolate"
        ),
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "동시에 실행할 모델 수 (--isolate 필요). "
            "--parallel 2 → deep 모델 2개를 동시에 훈련. "
            "GPU 사용 시 VRAM이 충분한지 확인 필요. "
            "CPU 훈련(LGBM 등)이나 멀티 GPU 환경에 적합."
        ),
    )
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

    # --share_datasets implies RAM caching for both train and eval
    if bool(getattr(args, 'share_datasets', False)):
        cfg.CACHE_TRAIN_SAMPLES_IN_RAM = True
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

    # --isolate: 각 모델을 별도 subprocess로 실행
    parallel = max(1, int(getattr(args, "parallel", 1)))
    if getattr(args, "isolate", False) and len(model_list) > 1:
        _run_isolated_per_model(raw_argv, model_list, parallel=parallel)
    elif parallel > 1 and len(model_list) > 1:
        # --parallel만 지정해도 --isolate 암시
        _run_isolated_per_model(raw_argv, model_list, parallel=parallel)
    else:
        run(args)


def _strip_isolate_args(raw_argv: List[str]) -> List[str]:
    """raw_argv에서 --isolate, --parallel, --models 플래그를 제거한다."""
    base_argv: List[str] = []
    skip_next = False
    for tok in raw_argv:
        if skip_next:
            skip_next = False
            continue
        if tok == "--isolate":
            continue
        if tok in ("--models", "--parallel"):
            skip_next = True
            continue
        if tok.startswith("--models=") or tok.startswith("--parallel="):
            continue
        base_argv.append(tok)
    return base_argv


def _run_isolated_per_model(
    raw_argv: List[str],
    model_list: List[str],
    parallel: int = 1,
) -> None:
    """각 모델을 별도 파이썬 서브프로세스로 실행하여 모델별 파이썬 인터프리터와 전역 실행 상태를 분리한다.

    주의: 서브프로세스 분리는 파이썬 프로세스 간 네임스페이스 및 메모리 상태를 격리할 뿐이며,
    OS 또는 하드웨어 수준의 물리적 RAM/GPU 메모리 격리나 자원 사용량 상한(cgroup/VRAM 제한)을 보장하지 않는다.

    parallel=1: 순차 실행 (기존 동작)
    parallel≥2: 모든 모델(LGBM 포함)을 parallel 개씩 배치 단위로 동시 실행

    각 서브프로세스가 독립적으로 experiment.run()을 호출하므로
    LGBM logit이 필요한 모델은 자체적으로 baseline을 학습한다.
    따라서 LGBM을 별도 Phase로 분리할 필요 없이 동일한 병렬 풀에서 실행 가능하다.

    실행 및 실패 처리:
        각 서브프로세스의 종료 코드(returncode)를 수집하여 SUCCESS 또는 FAILED (exit=코드) 상태를 기록하며,
        모든 배치 완료 후 요약(Summary)을 출력하고 실패한 모델 목록을 보고한 뒤 SystemExit(1)로 실패를 전달한다. 성공 시에는 정상 반환한다.
    파일·데이터베이스·외부 서비스와 물리 RAM/GPU는 공유할 수 있으므로 부작용의 완전 격리를 뜻하지 않는다."""
    base_argv = _strip_isolate_args(raw_argv)

    total = len(model_list)
    mode_str = f"parallel={parallel}" if parallel > 1 else "sequential"
    print(f"[ISOLATE] {total} model(s), mode={mode_str}")
    print(f"[ISOLATE] models: {model_list}")

    def _make_cmd(model_name: str) -> List[str]:
        return [sys.executable, "-u", sys.argv[0]] + base_argv + ["--models", model_name]

    results: Dict[str, str] = {}

    if parallel <= 1:
        # 순차 실행
        for i, m in enumerate(model_list):
            print(f"\n{'='*60}")
            print(f"[ISOLATE] [{i+1}/{total}] {m}")
            print(f"{'='*60}")
            proc = subprocess.run(_make_cmd(m))
            results[m] = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    else:
        # 병렬 실행: parallel 개씩 batch
        for batch_start in range(0, total, parallel):
            batch = model_list[batch_start:batch_start + parallel]
            batch_idx = batch_start // parallel + 1
            n_batches = (total + parallel - 1) // parallel

            print(f"\n{'='*60}")
            print(f"[PARALLEL] Batch {batch_idx}/{n_batches}: {batch}")
            print(f"{'='*60}")

            # 동시 시작
            procs: List[Tuple[str, subprocess.Popen]] = []
            for m in batch:
                print(f"  [START] {m}")
                p = subprocess.Popen(
                    _make_cmd(m),
                    stdout=None,  # 부모의 stdout 공유 (실시간 출력)
                    stderr=None,
                )
                procs.append((m, p))

            # 전부 완료 대기
            for m, p in procs:
                p.wait()
                status = "SUCCESS" if p.returncode == 0 else f"FAILED (exit={p.returncode})"
                results[m] = status
                print(f"  [DONE] {m}: {status}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"[ISOLATE] Summary:")
    for m in model_list:
        print(f"  {m}: {results.get(m, 'UNKNOWN')}")
    failed = [m for m, s in results.items() if not s.startswith("SUCCESS")]
    if failed:
        print(f"[ISOLATE] {len(failed)} model(s) FAILED: {failed}")
        raise SystemExit(1)
    else:
        print(f"[ISOLATE] All {total} model(s) completed successfully.")


if __name__ == "__main__":
    main()
