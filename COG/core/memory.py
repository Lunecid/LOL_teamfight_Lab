"""core/memory.py — 모델 간 메모리 해제 유틸리티

모델 M_i 훈련 완료 후 다음 모델 M_{i+1} 시작 전:
    RAM(t) = D_shared + θ_{M_i} + B_{M_i}
    → clear_memory() →
    RAM(t+) = D_shared + ε

이렇게 해야 M_{i+1}이 안전하게 RAM을 할당할 수 있다.
프로세스 내 GC + CUDA cache 해제를 명시적으로 수행한다.
"""
from __future__ import annotations

import gc
import sys
from typing import Any, Dict, Optional


def clear_memory(log_fp: Optional[Any] = None) -> Dict[str, Any]:
    """모델 전환 시 호출하여 Python 객체 + GPU 메모리를 해제한다.

    Returns:
        dict with memory stats before/after clearing.
    """
    stats: Dict[str, Any] = {}

    # 1) Python GC: 순환 참조 포함 전부 수거
    gc.collect()
    gc.collect()  # 2회: weak-ref + C extension cycle 잔존 해소

    # 2) GPU 메모리 해제
    try:
        import torch
        if torch.cuda.is_available():
            stats["cuda_before_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            stats["cuda_after_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass

    # 3) 한 번 더 GC (torch.cuda.empty_cache 이후 해제된 Python 래퍼 수거)
    gc.collect()

    if log_fp is not None:
        try:
            from core.utils import write_log
            cuda_msg = ""
            if "cuda_before_mb" in stats:
                cuda_msg = (
                    f" CUDA: {stats['cuda_before_mb']:.0f}MB → {stats['cuda_after_mb']:.0f}MB"
                )
            write_log(f"[MEMORY] clear_memory() done.{cuda_msg}", log_fp)
        except Exception:
            pass

    return stats


def clear_ram_cache() -> None:
    """data/ram_cache.py의 LRU match-pack 캐시를 비운다."""
    try:
        from data.ram_cache import _ram_clear
        _ram_clear()
    except ImportError:
        pass


def clear_all(log_fp: Optional[Any] = None) -> Dict[str, Any]:
    """RAM 캐시 + Python GC + GPU 메모리를 모두 해제한다.

    모델 전환 시 최대 메모리 회수를 위해 사용.
    """
    clear_ram_cache()
    return clear_memory(log_fp=log_fp)
