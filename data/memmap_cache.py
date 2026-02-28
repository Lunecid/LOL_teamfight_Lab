"""data/memmap_cache.py — Memory-mapped 데이터셋 캐시

numpy.memmap으로 전처리된 데이터셋을 디스크에 저장/로드한다.
OS 페이지 캐시 덕분에 자주 접근하는 페이지만 물리 RAM에 상주하고,
모델 간 같은 memmap 파일을 열어도 물리 메모리는 한 번만 사용된다.

사용 시나리오:
    1) 첫 번째 실행: 전처리 데이터를 .npy로 저장 (save_dataset_memmap)
    2) 이후 실행: mmap_mode='r'로 로드 (load_dataset_memmap)

메모리 효율:
    실제로 접근하는 페이지만 물리 RAM에 상주
    → 60만+ 레코드 × 150 features 데이터도 안전하게 처리 가능

Usage:
    from data.memmap_cache import save_dataset_memmap, load_dataset_memmap

    # Save (one-time preprocessing)
    save_dataset_memmap(cache_dir, X_train, y_train, "train")

    # Load (memmap — minimal RAM)
    X_train, y_train = load_dataset_memmap(cache_dir, "train")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _memmap_dir(cache_dir: Path, tag: str) -> Path:
    """memmap 파일이 저장될 디렉토리 경로를 반환한다."""
    d = cache_dir / "memmap" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_dataset_memmap(
    cache_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    tag: str = "train",
) -> Path:
    """전처리된 데이터셋을 .npy 파일로 저장한다.

    Args:
        cache_dir: 캐시 루트 디렉토리
        X: features array (n_samples, n_features) or (n_samples, T, n_features)
        y: labels array (n_samples,) or (n_samples, 1)
        tag: "train", "val", "test"

    Returns:
        저장된 디렉토리 경로
    """
    d = _memmap_dir(cache_dir, tag)
    np.save(str(d / "X.npy"), X)
    np.save(str(d / "y.npy"), y)

    # 메타데이터 저장
    meta = {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "n_samples": int(X.shape[0]),
    }
    import json
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    return d


def load_dataset_memmap(
    cache_dir: Path,
    tag: str = "train",
    mmap_mode: str = "r",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Memory-mapped로 데이터셋을 로드한다.

    OS 페이지 캐시를 활용하므로:
    - 읽기 속도: 순수 디스크 I/O보다 훨씬 빠름
    - 메모리 효율: 실제로 접근하는 페이지만 물리 RAM에 상주
    - 모델 간 공유: 여러 모델이 같은 memmap 파일을 열어도 물리 메모리는 한 번만 사용

    Args:
        cache_dir: 캐시 루트 디렉토리
        tag: "train", "val", "test"
        mmap_mode: numpy mmap_mode ('r', 'r+', 'c')

    Returns:
        (X, y) tuple, or None if cache doesn't exist
    """
    d = _memmap_dir(cache_dir, tag)
    x_path = d / "X.npy"
    y_path = d / "y.npy"

    if not x_path.exists() or not y_path.exists():
        return None

    X = np.load(str(x_path), mmap_mode=mmap_mode)
    y = np.load(str(y_path), mmap_mode=mmap_mode)
    return X, y


def memmap_exists(cache_dir: Path, tag: str = "train") -> bool:
    """memmap 캐시가 존재하는지 확인한다."""
    d = _memmap_dir(cache_dir, tag)
    return (d / "X.npy").exists() and (d / "y.npy").exists()
