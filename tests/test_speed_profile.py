"""Tests for train/speed.py profile resolution and tuning knobs."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.speed import apply_speed_profile


def _base_cfg():
    return SimpleNamespace(
        AMP=False,
        AMP_DTYPE="auto",
        TF32=False,
        CUDNN_BENCHMARK=False,
        TORCH_COMPILE=False,
        TORCH_COMPILE_MODE="default",
        TORCH_COMPILE_DYNAMIC=False,
        CACHE_MATCH_PACKS_IN_RAM=False,
        CACHE_TRAIN_SAMPLES_IN_RAM=False,
        CACHE_EVAL_SAMPLES_IN_RAM=False,
        PIN_MEMORY=False,
        PERSISTENT_WORKERS=False,
        NUM_WORKERS=0,
        EVAL_NUM_WORKERS=0,
        PREFETCH_FACTOR=2,
        BATCH_SIZE=64,
        USE_FUSED_ADAMW=False,
    )


def test_auto_profile_resolves_rtx5080(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(
        "train.speed._cuda_device_info",
        lambda: {"name": "NVIDIA GeForce RTX 5080", "vram_gb": 16.0},
    )

    applied = apply_speed_profile(cfg, profile="auto")

    assert applied == "rtx5080"
    assert cfg.AMP is True
    assert cfg.AMP_DTYPE == "bfloat16"
    assert cfg.TORCH_COMPILE is True
    assert cfg.TORCH_COMPILE_DYNAMIC is False
    assert cfg.USE_FUSED_ADAMW is True
    assert cfg.BATCH_SIZE >= 384


def test_rtx50_profile_scales_batch_by_vram(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(
        "train.speed._cuda_device_info",
        lambda: {"name": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0},
    )

    applied = apply_speed_profile(cfg, profile="rtx50")

    assert applied == "rtx50"
    assert cfg.BATCH_SIZE >= 640
    assert cfg.TORCH_COMPILE_MODE == "max-autotune"
    assert cfg.TORCH_COMPILE_DYNAMIC is False


def test_aggressive_profile_without_cuda_info(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr("train.speed._cuda_device_info", lambda: None)

    applied = apply_speed_profile(cfg, profile="aggressive")

    assert applied == "aggressive"
    assert cfg.TORCH_COMPILE is True
    assert cfg.TORCH_COMPILE_DYNAMIC is True
    assert cfg.BATCH_SIZE == 64
