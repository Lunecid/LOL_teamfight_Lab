from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch

# [P2-STRUCT-2] Unified write_log from utils.py (single source of truth).
# Previously: local stub using logging.info() — different behaviour from
# utils.write_log() which uses print(). Now all modules share the same
# observable behaviour for write_log.
from core.utils import write_log


def _cuda_device_info() -> Optional[Dict[str, object]]:
    if not torch.cuda.is_available():
        return None
    try:
        dev = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(dev)
        return {
            "name": str(getattr(props, "name", "")),
            "vram_gb": float(getattr(props, "total_memory", 0)) / (1024.0 ** 3),
        }
    except Exception:
        return None


def apply_speed_profile(cfg, profile: str = "none", log_fp: Optional[Path] = None) -> str:
    """Apply hardware-oriented runtime speed profile.

    Profiles:
      - none/off: do nothing
      - auto:     RTX 50-series -> rtx50, otherwise aggressive
      - rtx5080 / rtx50 / aggressive: high-throughput defaults
    """
    p = str(profile or "none").strip().lower()
    if p in ("", "none", "off"):
        return "none"

    info = _cuda_device_info()
    if p == "auto":
        if info is None:
            return "none"
        name = str(info.get("name", ""))
        p = "rtx50" if re.search(r"rtx\s*50\d\d", name, flags=re.IGNORECASE) else "aggressive"

    if p not in ("rtx5080", "rtx50", "aggressive"):
        return "none"

    # Precision / compiler
    setattr(cfg, "AMP", True)
    setattr(cfg, "AMP_DTYPE", "bfloat16" if p in ("rtx5080", "rtx50") else "auto")
    setattr(cfg, "TF32", True)
    setattr(cfg, "CUDNN_BENCHMARK", True)
    setattr(cfg, "TORCH_COMPILE", True)
    setattr(cfg, "TORCH_COMPILE_MODE", "max-autotune")
    setattr(cfg, "TORCH_COMPILE_DYNAMIC", True)

    # IO / dataloader
    setattr(cfg, "CACHE_MATCH_PACKS_IN_RAM", True)
    setattr(cfg, "CACHE_TRAIN_SAMPLES_IN_RAM", True)
    setattr(cfg, "CACHE_EVAL_SAMPLES_IN_RAM", True)
    setattr(cfg, "PIN_MEMORY", True)
    setattr(cfg, "PERSISTENT_WORKERS", True)

    cpu_n = int(os.cpu_count() or 8)
    num_workers = int(max(4, min(16, cpu_n // 2)))
    setattr(cfg, "NUM_WORKERS", num_workers)
    setattr(cfg, "EVAL_NUM_WORKERS", max(2, num_workers // 2))
    setattr(cfg, "PREFETCH_FACTOR", 4)

    # Conservative batch-size bump by available VRAM.
    if info is not None:
        vram = float(info.get("vram_gb", 0.0))
        if vram >= 24:
            target_bs = 512
        elif vram >= 16:
            target_bs = 256
        elif vram >= 12:
            target_bs = 192
        else:
            target_bs = 128
        try:
            curr_bs = int(getattr(cfg, "BATCH_SIZE", 64) or 64)
        except Exception:
            curr_bs = 64
        if target_bs > curr_bs:
            setattr(cfg, "BATCH_SIZE", target_bs)

    if log_fp:
        write_log(
            f"[TORCH] speed_profile={p} amp_dtype={getattr(cfg,'AMP_DTYPE','auto')} "
            f"compile={getattr(cfg,'TORCH_COMPILE',False)} mode={getattr(cfg,'TORCH_COMPILE_MODE','default')} "
            f"batch={getattr(cfg,'BATCH_SIZE', '?')} workers={getattr(cfg,'NUM_WORKERS','?')} "
            f"cache_train={getattr(cfg,'CACHE_TRAIN_SAMPLES_IN_RAM',False)}",
            log_fp,
        )
    return p


def setup_torch_speed(cfg, log_fp: Optional[Path] = None) -> None:
    """Apply optional speed knobs (TF32, cudnn benchmark, matmul precision)."""

    # TF32 (Ampere+)
    tf32 = bool(getattr(cfg, "TF32", True))
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    except Exception:
        pass

    # cudnn benchmark
    try:
        torch.backends.cudnn.benchmark = bool(getattr(cfg, "CUDNN_BENCHMARK", True))
    except Exception:
        pass

    # matmul precision (PyTorch 2)
    try:
        prec = str(getattr(cfg, "MATMUL_PRECISION", "high"))
        torch.set_float32_matmul_precision(prec)
    except Exception:
        pass

    if log_fp:
        write_log(
            f"[TORCH] TF32={tf32} cudnn.benchmark={getattr(torch.backends.cudnn,'benchmark',None)} matmul_precision={getattr(cfg,'MATMUL_PRECISION','?')}",
            log_fp,
        )
