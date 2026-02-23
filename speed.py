from __future__ import annotations

from config import cfg
from pathlib import Path
from typing import Optional

import torch

# [P2-STRUCT-2] Unified write_log from utils.py (single source of truth).
# Previously: local stub using logging.info() — different behaviour from
# utils.write_log() which uses print(). Now all modules share the same
# observable behaviour for write_log.
from utils import write_log



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