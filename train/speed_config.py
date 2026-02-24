"""
speed_config.py — Phase 1 GPU Maximum Utilization Config
=========================================================

사용법:
    실험 실행 전에 import하여 cfg에 오버레이 적용:

    from core.config import cfg
    from train.speed_config import apply_speed_overlay
    apply_speed_overlay(cfg)  # 기본: 단일 GPU, 충분한 RAM

또는 experiment_runner에서 직접:
    python experiment_runner.py --phase 1
    → run_single_experiment() 내 reset_config_to_baseline() 직후에 호출

수학적 근거:
    학습 시간 T_total ≈ N_models × N_seeds × N_epochs × (T_io + T_compute + T_eval)

    최적화 전략:
    ① T_io → 0       : RAM 캐싱으로 I/O 제거
    ② T_compute ↓     : 배치 증가 → GPU 병렬성 극대화
    ③ T_eval ↓        : eval도 RAM 캐싱 + 큰 배치
    ④ N_epochs ↓      : 수렴 감지 + early stop 강화
    ⑤ kernel fusion   : torch.compile로 Python 오버헤드 제거
"""

from __future__ import annotations

import math
import logging
from typing import Any

_logger = logging.getLogger(__name__)


def apply_speed_overlay(cfg_obj: Any, vram_gb: float = 24.0) -> None:
    """GPU 최대 활용을 위한 config overlay.

    Parameters
    ----------
    cfg_obj : CFG
        config.cfg 싱글톤 인스턴스
    vram_gb : float
        GPU VRAM (GB). 배치 사이즈 자동 조정에 사용.
        - 8 GB  (RTX 3060/4060)  → BATCH_SIZE=128
        - 16 GB (RTX 4080/A4000) → BATCH_SIZE=256
        - 24 GB (RTX 3090/4090)  → BATCH_SIZE=512
        - 40+GB (A100)           → BATCH_SIZE=1024

    수학적 근거:
        GPU utilization ∝ batch_size / SM_capacity
        AMP(FP16) 메모리 사용량:
            M ≈ B × (d_model × L_seq × 10) × 2 bytes
            B=512, d=128, L=12: ~15 MB (충분한 여유)
    """

    _logger.info(f"[SPEED] Applying speed overlay (VRAM={vram_gb:.1f} GB)")

    # ═══════════════════════════════════════════════════════════
    # ① I/O 제거: RAM 캐싱 (가장 큰 효과, 예상 2–5× speedup)
    # ═══════════════════════════════════════════════════════════
    #
    # Before: 매 epoch마다 disk → JSON parse → build_ms_sequence → tensor
    # After:  첫 epoch에서 preload, 이후 zero-copy __getitem__
    #
    # 메모리 사용량 추정:
    #   200K samples × ~8KB/sample ≈ 1.6 GB RAM (충분히 감당 가능)
    setattr(cfg_obj, "CACHE_TRAIN_SAMPLES_IN_RAM", True)
    setattr(cfg_obj, "CACHE_EVAL_SAMPLES_IN_RAM", True)
    setattr(cfg_obj, "CACHE_MATCH_PACKS_IN_RAM", True)

    # RAM 캐싱 시 workers=0이 자동 적용됨 (deep.py L1436-1439)
    # → fork overhead 제거, 메모리 중복 방지

    _logger.info("[SPEED] ① RAM caching: TRAIN=True, EVAL=True, MATCH_PACKS=True")

    # ═══════════════════════════════════════════════════════════
    # ② GPU 병렬성: 배치 사이즈 극대화
    # ═══════════════════════════════════════════════════════════
    #
    # GPU utilization ∝ batch_size / SM_capacity
    # 큰 배치 → fewer gradient steps → LR 스케일링 필요
    #   LR_new = LR_base × sqrt(B_new / B_base)  (linear scaling rule)
    if vram_gb >= 40:
        bs = 1024
    elif vram_gb >= 24:
        bs = 512
    elif vram_gb >= 16:
        bs = 256
    else:
        bs = 128

    setattr(cfg_obj, "BATCH_SIZE", bs)

    # LR 스케일링: sqrt scaling (Hoffer et al., 2017)
    # η(B) = η₀ · √(B / B₀)
    base_bs = 64
    base_lr = 5e-4
    lr_scaled = base_lr * math.sqrt(bs / base_bs)
    setattr(cfg_obj, "LR", lr_scaled)

    _logger.info(f"[SPEED] ② Batch={bs}, LR={lr_scaled:.6f} (sqrt-scaled from base_bs={base_bs})")

    # ═══════════════════════════════════════════════════════════
    # ③ Kernel Fusion: torch.compile (PyTorch 2.x)
    # ═══════════════════════════════════════════════════════════
    #
    # 첫 forward에서 ~30초 컴파일 비용, 이후 20-30% throughput 향상
    # GNN의 동적 adjacency 연산에서 특히 효과적
    setattr(cfg_obj, "TORCH_COMPILE", True)

    _logger.info("[SPEED] ③ torch.compile=True (kernel fusion enabled)")

    # ═══════════════════════════════════════════════════════════
    # ④ Mixed Precision + Hardware 최적화 (확인용, 이미 True)
    # ═══════════════════════════════════════════════════════════
    setattr(cfg_obj, "AMP", True)  # FP16 자동 혼합 정밀도
    setattr(cfg_obj, "TF32", True)  # Ampere+ TF32 matmul
    setattr(cfg_obj, "CUDNN_BENCHMARK", True)  # cuDNN autotuner

    _logger.info("[SPEED] ④ AMP=True, TF32=True, CUDNN_BENCHMARK=True")

    # ═══════════════════════════════════════════════════════════
    # ⑤ Early Stopping 강화
    # ═══════════════════════════════════════════════════════════
    #
    # 큰 배치 → 빠른 수렴, patience를 줄여 불필요한 epoch 제거
    setattr(cfg_obj, "PATIENCE", 3)
    setattr(cfg_obj, "EPOCHS", 15)

    _logger.info("[SPEED] ⑤ PATIENCE=3, EPOCHS=15")

    # ═══════════════════════════════════════════════════════════
    # ⑥ DataLoader 최적화 (non-cached fallback용)
    # ═══════════════════════════════════════════════════════════
    # RAM 캐싱이 꺼질 경우를 대비한 fallback 설정
    setattr(cfg_obj, "NUM_WORKERS", 8)
    setattr(cfg_obj, "EVAL_NUM_WORKERS", 4)
    setattr(cfg_obj, "PREFETCH_FACTOR", 4)
    setattr(cfg_obj, "PIN_MEMORY", True)
    setattr(cfg_obj, "PERSISTENT_WORKERS", True)

    _logger.info("[SPEED] ⑥ DataLoader: workers=8, eval_workers=4, prefetch=4")

    # ═══════════════════════════════════════════════════════════
    # ⑦ GNN FP32 강제 해제 (선택적, 정밀도 trade-off)
    # ═══════════════════════════════════════════════════════════
    # GNN_FORCE_FP32=True는 adjacency 연산을 FP32로 강제
    # AMP 하에서 이를 끄면 ~15% 추가 속도, 단 수치 안정성 주의
    # setattr(cfg_obj, "GNN_FORCE_FP32", False)  # ← 주석 해제 시 공격적 최적화


def apply_phase1_minimal_overlay(cfg_obj: Any, vram_gb: float = 24.0) -> None:
    """Phase 1 Baseline만을 위한 최소 모델 세트 + 속도 최적화.

    Phase 1의 목적은 μ_baseline ± σ_baseline 확립이므로,
    전체 17개 모델이 아닌 대표 모델만으로도 충분할 수 있음.

    단, 논문에 전체 모델 비교가 필요하면 이 함수 대신
    apply_speed_overlay()만 사용할 것.
    """
    apply_speed_overlay(cfg_obj, vram_gb=vram_gb)

    # Phase 1에서 전체 모델 실행이 필수인지 여부에 따라
    # 아래 주석을 해제하여 모델 수를 줄일 수 있음
    #
    # 대표 모델 세트 (각 패러다임 1개씩):
    # setattr(cfg_obj, "RNN_MODELS", ("rnn_bigru", "rnn_transformer"))
    # setattr(cfg_obj, "GNN_MODELS", ("gnn_graphsage", "gnn_gatv2"))
    # → 1 + 2 + 2 + 1 = 6 models × 5 seeds = 30 runs (vs 85)

# ═══════════════════════════════════════════════════════════════
# 예상 속도 개선 분석
# ═══════════════════════════════════════════════════════════════
#
# 기본 설정 (BATCH=64, no RAM cache, no compile):
#   1 model × 1 seed × 15 epochs ≈ 10-15분 (추정)
#   17 models × 5 seeds = 85 runs ≈ 14-21시간
#
# 최적화 후 (BATCH=512, RAM cache, compile):
#   ① RAM 캐싱: T_io ≈ 0  →  ×2.0-3.0 speedup
#   ② 배치 ×8:  fewer steps → ×1.5-2.0 speedup
#   ③ compile:  kernel fusion → ×1.2-1.3 speedup
#   ④ 종합:     ×3.5-5.0 speedup
#
#   85 runs ≈ 3-5시간 (vs 14-21시간)
#
# 모델 축소 병행 시 (6 models × 5 seeds = 30 runs):
#   30 runs ≈ 1-2시간