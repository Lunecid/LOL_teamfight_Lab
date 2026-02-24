"""
experiment_runner.py — Systematic Ablation Study Runner

이 스크립트는 7가지 개선안의 체계적 실험을 자동화합니다.
Phase 1~5의 전체 실험 프로토콜을 단일 진입점에서 관리합니다.

Usage:
    python experiment_runner.py --phase 1                    # Baseline reproduction
    python experiment_runner.py --phase 2 --treatment 1      # Single-factor: Focal Loss
    python experiment_runner.py --phase 2 --treatment all     # All single-factor
    python experiment_runner.py --phase 3                    # Interaction analysis
    python experiment_runner.py --phase 4                    # Sensitivity analysis
    python experiment_runner.py --phase 5                    # Final test evaluation

수학적 배경:
    각 Treatment T_i에 대해:
    Δ_i = AUC(Baseline + T_i) - AUC(Baseline)

    통계 검정:
    H_0: Δ_i ≤ 0  vs  H_1: Δ_i > 0
    검정: DeLong's test (AUC), McNemar's test (classification)
    보정: Holm-Bonferroni (m=7 multiple comparisons)
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# [SPEED] GPU utilization overlay
from train.speed_config import apply_speed_overlay


# ──────────────────────────────────────────────────────────────
# 1. Treatment Definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class Treatment:
    """개선안 정의.

    각 Treatment는 config.py에 대한 overlay (설정 덮어쓰기)로 표현.
    이를 통해 기존 파이프라인을 수정하지 않고 실험 조건을 제어.
    """
    id: int
    name: str
    short_name: str
    config_overlay: Dict[str, Any]
    hp_grid: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""


# 7가지 개선안 정의
TREATMENTS: Dict[int, Treatment] = {
    1: Treatment(
        id=1,
        name="Focal Loss",
        short_name="focal",
        config_overlay={
            "USE_FOCAL_LOSS": True,
            "FOCAL_GAMMA": 2.0,
            "FOCAL_ALPHA": 0.25,
        },
        hp_grid={
            "FOCAL_GAMMA": [1.0, 2.0, 3.0],
        },
        description=(
            "L_FL(p_t) = -α_t (1 - p_t)^γ log(p_t)\n"
            "Hard example mining: 쉬운 샘플(gold_diff >> 0)의 기여를 줄이고\n"
            "어려운 샘플(close fights)에 집중"
        ),
    ),
    2: Treatment(
        id=2,
        name="Game Phase Encoding",
        short_name="phase",
        config_overlay={
            "USE_GAME_PHASE": True,
            "GAME_PHASE_TAU": 3.0,
        },
        hp_grid={
            "GAME_PHASE_TAU": [2.0, 3.0, 4.0],
        },
        description=(
            "φ(t) = [σ((14-t)/τ), σ((t-10)/τ)·σ((28-t)/τ), σ((t-22)/τ)]\n"
            "LoL의 3단계 게임 국면(초반/중반/후반)을 연속 인코딩"
        ),
    ),
    3: Treatment(
        id=3,
        name="Attention Temporal Pooling",
        short_name="attn_pool",
        config_overlay={
            "USE_ATTENTION_POOL": True,
            "ATTENTION_POOL_DIM": 64,
        },
        hp_grid={},
        description=(
            "α_t = softmax(w^T tanh(W_a h_t))\n"
            "c = Σ_t α_t h_t\n"
            "output = [h_T || c]\n"
            "마지막 hidden state만 사용하는 대신 전체 시퀀스 가중 합산"
        ),
    ),
    4: Treatment(
        id=4,
        name="Gold/Stat Momentum Features",
        short_name="momentum",
        config_overlay={
            "USE_MOMENTUM_FEATURES": True,
            "MOMENTUM_K_SHORT": 3,
        },
        hp_grid={
            "MOMENTUM_K_SHORT": [3, 5],
        },
        description=(
            "μ_short = (1/k) Σ Δx_{T-i}  (최근 k 스텝)\n"
            "μ_long  = (1/T) Σ Δx_t       (전체 평균)\n"
            "δ_mom   = μ_short - μ_long    (MACD-like divergence)\n"
            "단기 모멘텀 vs 장기 추세 괴리 포착"
        ),
    ),
    5: Treatment(
        id=5,
        name="Role-Aware Adjacency",
        short_name="role_adj",
        config_overlay={
            "USE_ROLE_AWARE_ADJ": True,
            "ROLE_ADJ_INIT": 0.0,
        },
        hp_grid={},
        description=(
            "A'_ij = A^dist_ij · softplus(R_{role(i), role(j)})\n"
            "R ∈ R^{5×5}: 학습 가능한 역할 상호작용 행렬\n"
            "봇 듀오, 중정글 연동 등 도메인 구조 반영"
        ),
    ),
    6: Treatment(
        id=6,
        name="Multi-Task Auxiliary Loss",
        short_name="mtl",
        config_overlay={
            "USE_MULTI_TASK": True,
            "MTL_LAMBDA_GOLD": 0.1,
            "MTL_LAMBDA_KILL": 0.05,
            "MTL_LAMBDA_OBJ": 0.05,
        },
        hp_grid={
            "MTL_LAMBDA_GOLD": [0.05, 0.1, 0.2],
        },
        description=(
            "L = L_fight + λ_g·||ĝ - g*||² + λ_k·||k̂ - k*||²\n"
            "보조 회귀 task가 implicit regularization 역할\n"
            "(Ruder 2017: multi-task → tighter generalization bounds)"
        ),
    ),
    7: Treatment(
        id=7,
        name="Label Smoothing",
        short_name="label_smooth",
        config_overlay={
            "LABEL_SMOOTHING": 0.05,
        },
        hp_grid={
            "LABEL_SMOOTHING": [0.03, 0.05, 0.10],
        },
        description=(
            "y_smooth = y·(1-ε) + ε/2\n"
            "ε=0.05: y=1→0.975, y=0→0.025\n"
            "KL regularization toward uniform; calibration 개선"
        ),
    ),
}

SEEDS: Tuple[int, ...] = (7, 42, 123, 256, 512)


# ──────────────────────────────────────────────────────────────
# 2. Statistical Testing Utilities
# ──────────────────────────────────────────────────────────────

def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """DeLong's test for comparing two AUC values.

    수학적 배경:
        Z = (AUC_A - AUC_B) / sqrt(Var(AUC_A) + Var(AUC_B) - 2·Cov(AUC_A, AUC_B))

    DeLong et al. (1988) "Comparing the areas under two or more correlated
    receiver operating characteristic curves: a nonparametric approach"

    Returns:
        p-value (two-sided)
    """
    try:
        from scipy import stats as sp_stats
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[WARN] scipy/sklearn not available for DeLong's test")
        return float('nan')

    n = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    # Structural components for variance estimation
    def compute_structural(pred, pos_idx, neg_idx):
        """Compute placement values for DeLong variance."""
        pred_pos = pred[pos_idx]
        pred_neg = pred[neg_idx]

        v10 = np.zeros(len(pos_idx))
        for j, pp in enumerate(pred_pos):
            v10[j] = np.mean((pred_neg < pp).astype(float) + 0.5 * (pred_neg == pp).astype(float))

        v01 = np.zeros(len(neg_idx))
        for i, pn in enumerate(pred_neg):
            v01[i] = np.mean((pred_pos > pn).astype(float) + 0.5 * (pred_pos == pn).astype(float))

        return v10, v01

    v10_a, v01_a = compute_structural(pred_a, pos_idx, neg_idx)
    v10_b, v01_b = compute_structural(pred_b, pos_idx, neg_idx)

    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)

    # Covariance matrix of (AUC_A, AUC_B)
    s10 = np.cov(np.stack([v10_a, v10_b]))[0, 1] if n_pos > 1 else 0
    s01 = np.cov(np.stack([v01_a, v01_b]))[0, 1] if n_neg > 1 else 0

    var_a = np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg if (n_pos > 1 and n_neg > 1) else 1e-10
    var_b = np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg if (n_pos > 1 and n_neg > 1) else 1e-10
    cov_ab = s10 / n_pos + s01 / n_neg

    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        return 1.0 if abs(auc_a - auc_b) < 1e-10 else 0.0

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))  # two-sided

    return float(p_value)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                 threshold: float = 0.5) -> float:
    """McNemar's test for paired classification comparison.

    수학적 배경:
        2×2 contingency table:
                       Model B correct    Model B wrong
        Model A correct     n_{11}          n_{10}  (= c)
        Model A wrong       n_{01}  (= b)  n_{00}

        χ² = (|b - c| - 1)² / (b + c)    (continuity correction)
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float('nan')

    y_a = (pred_a >= threshold).astype(int)
    y_b = (pred_b >= threshold).astype(int)

    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)

    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))

    if b + c == 0:
        return 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)

    return float(p_value)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni method for multiple testing correction.

    수학적 배경:
        Sorted p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
        Adjusted threshold: α_(k) = α / (m - k + 1)
        Reject H_0^(k) if p_(k) ≤ α_(k)

    Properties:
        - Controls Family-Wise Error Rate (FWER) ≤ α
        - Uniformly more powerful than Bonferroni
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [False] * m

    for rank, (original_idx, p_val) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p_val <= adjusted_alpha:
            results[original_idx] = True
        else:
            break

    return results


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    ECE = Σ_{m=1}^{M} (|B_m|/N) · |acc(B_m) - conf(B_m)|
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)

    return float(ece)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score: BS = (1/N) Σ (p_i - y_i)²"""
    return float(np.mean((y_prob - y_true) ** 2))


# ──────────────────────────────────────────────────────────────
# 3. Experiment Result Container
# ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """단일 실험 run의 결과.

    각 run은 (treatment_id, seed, hp_config) 3-tuple로 식별.
    """
    treatment_id: int
    treatment_name: str
    seed: int
    hp_config: Dict[str, Any]

    # Core metrics (train/val/test)
    train_auc: float = -1.0
    val_auc: float = -1.0
    test_auc: float = -1.0

    train_ap: float = -1.0
    val_ap: float = -1.0
    test_ap: float = -1.0

    train_f1: float = -1.0
    val_f1: float = -1.0
    test_f1: float = -1.0

    # Calibration metrics
    val_brier: float = -1.0
    val_ece: float = -1.0
    test_brier: float = -1.0
    test_ece: float = -1.0

    # Training metadata
    best_epoch: int = -1
    train_time_sec: float = -1.0

    # Prediction maps for statistical testing
    pred_logits_val: Optional[Dict[str, float]] = None
    pred_logits_test: Optional[Dict[str, float]] = None


@dataclass
class AblationSummary:
    """단일 Treatment의 5-seed 종합 결과."""
    treatment_id: int
    treatment_name: str

    # Mean ± Std over seeds
    val_auc_mean: float = 0.0
    val_auc_std: float = 0.0
    val_auc_seeds: List[float] = field(default_factory=list)

    test_auc_mean: float = 0.0
    test_auc_std: float = 0.0

    # Effect size relative to baseline
    delta_val_auc_mean: float = 0.0
    delta_val_auc_std: float = 0.0
    delta_val_auc_ci_low: float = 0.0
    delta_val_auc_ci_high: float = 0.0

    # Statistical tests
    delong_p_value: float = 1.0
    mcnemar_p_value: float = 1.0
    significant_after_correction: bool = False

    # Cohen's d effect size
    cohens_d: float = 0.0


# ──────────────────────────────────────────────────────────────
# 4. Config Overlay Mechanism
# ──────────────────────────────────────────────────────────────

def apply_config_overlay(cfg_obj: Any, overlay: Dict[str, Any]) -> None:
    """Config 객체에 Treatment-specific 설정을 덮어씌움."""
    for key, value in overlay.items():
        try:
            setattr(cfg_obj, key, value)
        except AttributeError:
            print(f"[WARN] Config attribute '{key}' does not exist, adding dynamically")
            setattr(cfg_obj, key, value)


def reset_config_to_baseline(cfg_obj: Any) -> None:
    """모든 Treatment flag를 기본값(off)으로 리셋.

    Phase 2 실험 간 교차 오염을 방지하기 위해 필수.
    """
    baseline_flags = {
        "USE_FOCAL_LOSS": False,
        "FOCAL_GAMMA": 2.0,
        "FOCAL_ALPHA": 0.25,
        "USE_GAME_PHASE": False,
        "GAME_PHASE_TAU": 3.0,
        "USE_ATTENTION_POOL": False,
        "ATTENTION_POOL_DIM": 64,
        "USE_MOMENTUM_FEATURES": False,
        "MOMENTUM_K_SHORT": 3,
        "USE_ROLE_AWARE_ADJ": False,
        "ROLE_ADJ_INIT": 0.0,
        "USE_MULTI_TASK": False,
        "MTL_LAMBDA_GOLD": 0.1,
        "MTL_LAMBDA_KILL": 0.05,
        "MTL_LAMBDA_OBJ": 0.05,
        "LABEL_SMOOTHING": 0.0,
    }
    apply_config_overlay(cfg_obj, baseline_flags)


# ──────────────────────────────────────────────────────────────
# 4b. Pipeline Bridge — 실제 실험 실행 인터페이스
# ──────────────────────────────────────────────────────────────

def _build_experiment_args(
    feature_set: str = "full",
    seed: int = 7,
    split_mode: str = "patch_holdout",
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> argparse.Namespace:
    """runner.py의 argparser와 호환되는 Namespace 생성.

    experiment.run(args)에 전달할 수 있는 형태로 변환.
    """
    from runner import build_argparser
    parser = build_argparser()
    args = parser.parse_args([])  # 기본값으로 초기화

    # 핵심 실험 파라미터 설정
    args.feature_set = feature_set
    args.seed = seed
    args.split_mode = split_mode

    # 추가 오버라이드 적용
    if extra_overrides:
        for k, v in extra_overrides.items():
            setattr(args, k, v)

    return args


def _find_unsupported_overlay_flags(overlay: Dict[str, Any]) -> List[str]:
    unsupported: List[str] = []
    return unsupported


def run_single_experiment(
    treatment_overlay: Dict[str, Any],
    seed: int,
    feature_set: str = "full",
    split_mode: str = "patch_holdout",
    experiment_tag: str = "",
) -> ExperimentResult:
    """단일 실험을 실행하고 ExperimentResult를 반환.

    이 함수는 experiment_runner와 기존 pipeline 사이의 브릿지 역할.

    수학적 흐름:
        1. Config reset → baseline state
        2. Treatment overlay 적용: cfg ← cfg ⊕ treatment_overlay
        3. experiment.run(args) 실행 → deep model training + evaluation
        4. metrics 파싱 → ExperimentResult

    Parameters
    ----------
    treatment_overlay : dict
        Treatment.config_overlay + hp_config merged
    seed : int
        Random seed for reproducibility
    feature_set : str
        Feature set identifier
    split_mode : str
        Data split strategy
    experiment_tag : str
        Human-readable experiment identifier

    Returns
    -------
    ExperimentResult with parsed metrics
    """
    from core.config import cfg, RUN_DIR
    from app.experiment import run as run_experiment
    from core.utils import set_seed

    print(f"\n    [EXEC] {experiment_tag} | seed={seed} | overlay={treatment_overlay}")
    t0 = time.time()

    # Step 1: Reset to clean baseline
    reset_config_to_baseline(cfg)

    # Step 1a: Apply GPU speed overlay (RAM cache, batch scaling, compile)
    # T_total = M × S × E × (T_io + T_compute + T_eval)
    # Speed overlay targets each term: T_io→0, T_compute↓, E↓
    _speed_env = str(os.environ.get("LOL_SPEED_OVERLAY", "1")).strip().lower()
    _use_speed = _speed_env not in ("0", "false", "off", "no")
    _vram_gb = float(os.environ.get("LOL_VRAM_GB", "24.0"))
    _speed_profile_raw = str(os.environ.get("LOL_SPEED_PROFILE", "auto" if _use_speed else "none")).strip().lower()
    _speed_profile = "none" if _speed_profile_raw in ("", "off") else _speed_profile_raw
    if _use_speed:
        apply_speed_overlay(cfg, vram_gb=_vram_gb)
        cfg.SPEED_PROFILE = _speed_profile
        try:
            from train.speed import apply_speed_profile as _apply_runtime_speed_profile
            _applied = "none" if _speed_profile == "none" else _apply_runtime_speed_profile(cfg, profile=_speed_profile)
        except Exception:
            _applied = "none"
        print(
            "    [SPEED] enabled "
            f"(vram={_vram_gb:.1f}GB, profile_req={_speed_profile}, profile_applied={_applied}, "
            f"batch={getattr(cfg, 'BATCH_SIZE', '?')}, "
            f"amp={getattr(cfg, 'AMP', False)}, "
            f"compile={getattr(cfg, 'TORCH_COMPILE', False)}, "
            f"cache_train={getattr(cfg, 'CACHE_TRAIN_SAMPLES_IN_RAM', False)})"
        )
        if _speed_profile != "none" and _applied == "none":
            print("    [SPEED] runtime profile fallback: overlay-only (likely CUDA unavailable for auto profile)")
    else:
        cfg.SPEED_PROFILE = "none"
        print("    [SPEED] disabled")

    # Step 1b: Reset module-level singletons to prevent parameter leakage
    # θ_singleton ← None → forces lazy re-initialization with fresh parameters
    try:
        from train.models import reset_model_singletons
        reset_model_singletons()
    except ImportError:
        pass  # models.py가 아직 reset 함수를 갖지 않는 환경 (backward compat)

    # Step 2: Apply treatment overlay
    apply_config_overlay(cfg, treatment_overlay)

    unsupported = _find_unsupported_overlay_flags(treatment_overlay)
    if unsupported:
        msg = " | ".join(unsupported)
        print(f"    [ERROR] Unsupported overlay: {msg}")
        return ExperimentResult(
            treatment_id=-1,
            treatment_name=experiment_tag,
            seed=seed,
            hp_config=treatment_overlay,
            train_time_sec=time.time() - t0,
        )

    # Step 3: Set seed
    set_seed(seed)

    # Step 4: Build args and run
    args = _build_experiment_args(
        feature_set=feature_set,
        seed=seed,
        split_mode=split_mode,
    )

    # Attach model list from config
    model_list = list(getattr(cfg, "MODEL_LIST", []))
    args.model_list = model_list

    run_dirs_before: set[str] = set()
    try:
        run_dirs_before = {
            d.name
            for d in RUN_DIR.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        }
    except Exception:
        run_dirs_before = set()

    try:
        run_experiment(args)
    except Exception as e:
        print(f"    [ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return ExperimentResult(
            treatment_id=-1,
            treatment_name=experiment_tag,
            seed=seed,
            hp_config=treatment_overlay,
            train_time_sec=time.time() - t0,
        )

    # Step 5: Parse results from the run directory produced by this execution
    run_dir_hint = _pick_run_dir(
        run_root=RUN_DIR,
        seed=seed,
        before_run_names=run_dirs_before,
        started_at=t0,
    )
    result = _parse_latest_run_result(
        experiment_tag=experiment_tag,
        seed=seed,
        hp_config=treatment_overlay,
        run_dir_hint=run_dir_hint,
        preferred_models=model_list,
    )
    result.train_time_sec = time.time() - t0

    print(f"    [DONE] val_auc={result.val_auc:.4f} test_auc={result.test_auc:.4f} "
          f"time={result.train_time_sec:.1f}s")

    return result


def _pick_run_dir(
    run_root: Path,
    seed: int,
    before_run_names: Optional[set[str]] = None,
    started_at: Optional[float] = None,
) -> Optional[Path]:
    try:
        run_dirs = [d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    except Exception:
        return None

    if not run_dirs:
        return None

    def _mtime(p: Path) -> float:
        try:
            return float(p.stat().st_mtime)
        except Exception:
            return -1.0

    seed_token = f"__seed={int(seed)}"

    def _pick(cands: List[Path]) -> Optional[Path]:
        if not cands:
            return None
        cands = sorted(cands, key=_mtime, reverse=True)
        by_seed = [d for d in cands if seed_token in d.name]
        return by_seed[0] if by_seed else cands[0]

    if before_run_names:
        created = [d for d in run_dirs if d.name not in before_run_names]
        picked = _pick(created)
        if picked is not None:
            return picked

    if started_at is not None:
        recent = [d for d in run_dirs if _mtime(d) >= float(started_at) - 1.0]
        picked = _pick(recent)
        if picked is not None:
            return picked

    return _pick(run_dirs)


def _parse_latest_run_result(
    experiment_tag: str,
    seed: int,
    hp_config: Dict[str, Any],
    run_dir_hint: Optional[Path] = None,
    preferred_models: Optional[List[str]] = None,
) -> ExperimentResult:
    """가장 최근 run 디렉토리에서 결과를 파싱.

    experiment.run()은 RUN_DIR/<run_tag>/ 아래에 결과를 저장.
    deep_reports.json 또는 ablation_summary.csv에서 metrics를 추출.
    """
    from core.config import RUN_DIR

    result = ExperimentResult(
        treatment_id=-1,
        treatment_name=experiment_tag,
        seed=seed,
        hp_config=hp_config,
    )

    try:
        latest_run = run_dir_hint
        if latest_run is None:
            latest_run = _pick_run_dir(run_root=RUN_DIR, seed=seed, before_run_names=None, started_at=None)
        if latest_run is None:
            print(f"    [WARN] No run directories found in {RUN_DIR}")
            return result

        # deep_reports.json 에서 metrics 파싱 (run root 우선)
        reports_path = latest_run / "deep_reports.json"
        if not reports_path.exists() and (latest_run / "models").exists():
            reports_path = latest_run / "models" / "deep_reports.json"

        # 직접 reports 검색
        if not reports_path.exists():
            for p in latest_run.rglob("deep_reports.json"):
                reports_path = p
                break

        if reports_path.exists():
            with open(reports_path, "r") as f:
                deep_reports = json.load(f)

            preferred = {
                str(m).strip()
                for m in (preferred_models or [])
                if str(m).strip() and str(m).strip().lower() != "lgbm"
            }
            best_report = None
            best_score = (-1, -1.0)

            for model_key, report in deep_reports.items():
                if not isinstance(report, dict) or not report.get("ok", False):
                    continue

                metrics = report.get("metrics", {})
                va_m = metrics.get("val", {})
                try:
                    va_auc = float(va_m.get("auc", float("nan")))
                except Exception:
                    va_auc = float("nan")
                va_auc_score = va_auc if np.isfinite(va_auc) else -1.0

                base_model = str(model_key).split("::", 1)[0]
                priority = 1 if (not preferred or base_model in preferred) else 0
                score = (priority, va_auc_score)
                if score > best_score:
                    best_score = score
                    best_report = report

            if isinstance(best_report, dict):
                metrics = best_report.get("metrics", {})
                tr_m = metrics.get("train", {})
                va_m = metrics.get("val", {})
                te_m = metrics.get("test", {})

                result.train_auc = float(tr_m.get("auc", -1.0))
                result.val_auc = float(va_m.get("auc", -1.0))
                result.test_auc = float(te_m.get("auc", -1.0))

                result.train_f1 = float(tr_m.get("f1", -1.0))
                result.val_f1 = float(va_m.get("f1", -1.0))
                result.test_f1 = float(te_m.get("f1", -1.0))

                result.train_ap = float(tr_m.get("ap", -1.0))
                result.val_ap = float(va_m.get("ap", -1.0))
                result.test_ap = float(te_m.get("ap", -1.0))

                result.best_epoch = int(best_report.get("best_epoch", -1))

        # ablation_summary.csv fallback
        if result.val_auc < 0:
            csv_path = latest_run / "ablation_summary.csv"
            if csv_path.exists():
                import csv
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    best_row = None
                    best_va = -1.0
                    for row in reader:
                        try:
                            va = float(row.get("va_auc", -1.0) or -1.0)
                        except Exception:
                            va = -1.0
                        if va > best_va:
                            best_va = va
                            best_row = row
                    if best_row is not None:
                        result.val_auc = float(best_row.get("va_auc", -1.0) or -1.0)
                        result.test_auc = float(best_row.get("te_auc", -1.0) or -1.0)
                        result.train_auc = float(best_row.get("tr_auc", -1.0) or -1.0)

    except Exception as e:
        print(f"    [WARN] Result parsing failed: {e}")

    return result


# ──────────────────────────────────────────────────────────────
# 5. Phase Executors
# ──────────────────────────────────────────────────────────────

def run_phase1_baseline(args: argparse.Namespace) -> List[ExperimentResult]:
    """Phase 1: Baseline Reproduction.

    5개 seed로 기존 시스템을 재현하여 정확한 기준선 확립.

    수학적 목적:
        μ_baseline = (1/S) Σ_{s=1}^{S} AUC^{(s)}
        σ_baseline = sqrt((1/(S-1)) Σ (AUC^{(s)} - μ)²)
    """
    print("=" * 70)
    print("PHASE 1: Baseline Reproduction (5 seeds)")
    print("=" * 70)

    results = []

    for seed in SEEDS:
        print(f"\n--- Baseline | seed={seed} ---")

        if args.dry_run:
            result = ExperimentResult(
                treatment_id=0, treatment_name="Baseline",
                seed=seed, hp_config={},
            )
        else:
            result = run_single_experiment(
                treatment_overlay={},  # empty = pure baseline
                seed=seed,
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag="Baseline",
            )
            result.treatment_id = 0
            result.treatment_name = "Baseline"

        results.append(result)

    _print_phase1_summary(results)
    return results


def run_phase2_single_factor(
        args: argparse.Namespace,
        treatment_ids: List[int],
        baseline_results: List[ExperimentResult],
) -> Dict[int, List[ExperimentResult]]:
    """Phase 2: Single-Factor Ablation.

    각 Treatment T_i를 독립적으로 적용하여 개별 효과 측정.

    실험 매트릭스:
        E_single = {(Baseline + T_i) | i ∈ treatment_ids}
        총 |treatment_ids| × 5 runs
    """
    print("=" * 70)
    print(f"PHASE 2: Single-Factor Ablation (treatments={treatment_ids})")
    print("=" * 70)

    all_results: Dict[int, List[ExperimentResult]] = {}

    for tid in treatment_ids:
        treatment = TREATMENTS[tid]
        print(f"\n{'─' * 50}")
        print(f"Treatment T_{tid}: {treatment.name}")
        print(f"Description:\n{treatment.description}")
        print(f"{'─' * 50}")

        # Step 1: HP Grid Search
        best_hp = _hp_grid_search(args, treatment)

        # Step 2: 5-seed evaluation with best HP
        results_for_treatment = []

        # Merge base overlay + best HP
        merged_overlay = {**treatment.config_overlay, **best_hp}

        for seed in SEEDS:
            print(f"  seed={seed}, hp={best_hp}")

            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=tid, treatment_name=treatment.name,
                    seed=seed, hp_config=best_hp,
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=merged_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"T{tid}_{treatment.short_name}",
                )
                result.treatment_id = tid
                result.treatment_name = treatment.name
                result.hp_config = best_hp

            results_for_treatment.append(result)

        all_results[tid] = results_for_treatment

    # Compute summaries and statistical tests
    summaries = _compute_ablation_summaries(all_results, baseline_results)
    _print_phase2_summary(summaries)

    return all_results


def _hp_grid_search(args: argparse.Namespace, treatment: Treatment) -> Dict[str, Any]:
    """하이퍼파라미터 그리드 서치.

    전략: Val AUC 기준 1-seed (seed=7) 빠른 탐색 후 best 선택.

    수학적 정의:
        θ* = argmax_{θ ∈ Θ} AUC_val(Baseline + T_i(θ); seed=7)
    """
    if not treatment.hp_grid:
        return {}

    print(f"  [HP Search] Grid: {treatment.hp_grid}")

    keys = list(treatment.hp_grid.keys())
    values = list(treatment.hp_grid.values())

    best_val_auc = -1.0
    best_config = {}

    for combo in itertools.product(*values):
        hp_config = dict(zip(keys, combo))
        print(f"    Trying: {hp_config} ...", end=" ")

        if args.dry_run:
            val_auc = -1.0
        else:
            # Merge treatment overlay + this HP combo
            merged_overlay = {**treatment.config_overlay, **hp_config}
            result = run_single_experiment(
                treatment_overlay=merged_overlay,
                seed=SEEDS[0],  # single seed for speed
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag=f"HP_{treatment.short_name}",
            )
            val_auc = result.val_auc

        print(f"val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = hp_config

    print(f"  [HP Search] Best: {best_config} (val_auc={best_val_auc:.4f})")
    return best_config


def run_phase3_interaction(
        args: argparse.Namespace,
        phase2_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
        top_k: int = 3,
) -> Dict[str, List[ExperimentResult]]:
    """Phase 3: Interaction Analysis.

    3.1 Pairwise Interaction:
        Interaction_{i,j} = Δ_{i+j} - (Δ_i + Δ_j)

    3.2 Cumulative Addition (Forward Selection):
        AUC_k = AUC(Baseline + T_{rank1} + ... + T_{rank_k})
        MC_k = AUC_k - AUC_{k-1}  (marginal contribution)
    """
    print("=" * 70)
    print("PHASE 3: Interaction Analysis")
    print("=" * 70)

    # Rank treatments by mean Δ_val_auc
    treatment_deltas = {}
    baseline_mean = np.mean([r.val_auc for r in baseline_results if r.val_auc > 0])

    for tid, results in phase2_results.items():
        val_aucs = [r.val_auc for r in results if r.val_auc > 0]
        if val_aucs:
            delta_mean = np.mean(val_aucs) - baseline_mean
            treatment_deltas[tid] = delta_mean

    ranked = sorted(treatment_deltas.items(), key=lambda x: -x[1])
    top_ids = [tid for tid, _ in ranked[:top_k]]

    print(f"\nTop-{top_k} Treatments: "
          f"{[(tid, TREATMENTS[tid].name, f'Δ={d:.4f}') for tid, d in ranked[:top_k]]}")

    all_results: Dict[str, List[ExperimentResult]] = {}

    # 3.1 Pairwise Combinations
    pairs = list(itertools.combinations(top_ids, 2))
    print(f"\n--- 3.1 Pairwise Interaction ({len(pairs)} pairs) ---")

    for i, j in pairs:
        pair_key = f"pair_{i}_{j}"
        print(f"\n  Pair: T_{i}({TREATMENTS[i].short_name}) + T_{j}({TREATMENTS[j].short_name})")

        pair_results = []
        combined_overlay = {**TREATMENTS[i].config_overlay, **TREATMENTS[j].config_overlay}

        for seed in SEEDS:
            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=-1, treatment_name=f"T{i}+T{j}",
                    seed=seed, hp_config={},
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=combined_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"T{i}+T{j}",
                )
                result.treatment_id = -1
                result.treatment_name = f"T{i}+T{j}"

            pair_results.append(result)

        all_results[pair_key] = pair_results

        # Compute interaction
        delta_i = treatment_deltas.get(i, 0.0)
        delta_j = treatment_deltas.get(j, 0.0)
        pair_mean = np.mean([r.val_auc for r in pair_results if r.val_auc > 0]) if any(
            r.val_auc > 0 for r in pair_results) else baseline_mean
        delta_ij = pair_mean - baseline_mean
        interaction = delta_ij - (delta_i + delta_j)

        label = "SYNERGY" if interaction > 0.001 else ("REDUNDANCY" if interaction < -0.001 else "INDEPENDENT")
        print(f"    Δ_i={delta_i:.4f}, Δ_j={delta_j:.4f}, Δ_{{i+j}}={delta_ij:.4f}")
        print(f"    Interaction = {interaction:+.4f} ({label})")

    # 3.2 Cumulative Addition (Forward Selection)
    print(f"\n--- 3.2 Cumulative Addition (Forward Selection) ---")

    cumulative_ids = []
    prev_auc = baseline_mean

    for step, (tid, delta) in enumerate(ranked):
        cumulative_ids.append(tid)
        combo_key = f"cumul_step{step + 1}"

        # Merge all treatment overlays accumulated so far
        cumul_overlay = {}
        for cid in cumulative_ids:
            cumul_overlay.update(TREATMENTS[cid].config_overlay)

        combo_results = []
        for seed in SEEDS:
            if args.dry_run:
                result = ExperimentResult(
                    treatment_id=-1, treatment_name=f"Cumul_step{step + 1}",
                    seed=seed, hp_config={},
                )
            else:
                result = run_single_experiment(
                    treatment_overlay=cumul_overlay,
                    seed=seed,
                    feature_set=args.feature_set,
                    split_mode=args.split_mode,
                    experiment_tag=f"Cumul_step{step + 1}",
                )
                result.treatment_id = -1
                result.treatment_name = f"Cumul_step{step + 1}"

            combo_results.append(result)

        all_results[combo_key] = combo_results

        current_auc = np.mean([r.val_auc for r in combo_results if r.val_auc > 0]) if any(
            r.val_auc > 0 for r in combo_results) else prev_auc
        mc = current_auc - prev_auc

        print(f"  Step {step + 1}: +T_{tid}({TREATMENTS[tid].short_name}) → "
              f"AUC={current_auc:.4f}, MC={mc:+.4f}")
        prev_auc = current_auc

    return all_results


def run_phase4_sensitivity(
        args: argparse.Namespace,
        phase2_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
) -> Dict[str, List[ExperimentResult]]:
    """Phase 4: Hyperparameter Sensitivity Analysis.

    Phase 2에서 significant한 각 Treatment에 대해
    HP grid의 전체 surface를 5-seed로 스캔.

    수학적 목적:
        ∀θ_j ∈ Grid(T_i):
            AUC(θ_j) = (1/S) Σ_{s} AUC(Baseline + T_i(θ_j); s)

        Sensitivity = std(AUC(θ)) / mean(AUC(θ))  (coefficient of variation)

    낮은 sensitivity → 해당 HP에 robust → 실무 적용 용이
    높은 sensitivity → fine-tuning 필요 → 논문에서 주의 명시
    """
    print("=" * 70)
    print("PHASE 4: Hyperparameter Sensitivity Analysis")
    print("=" * 70)

    all_results: Dict[str, List[ExperimentResult]] = {}

    for tid, treatment in TREATMENTS.items():
        if not treatment.hp_grid:
            print(f"\n  T_{tid} ({treatment.name}): No HP grid → skip")
            continue

        print(f"\n{'─' * 50}")
        print(f"Treatment T_{tid}: {treatment.name}")
        print(f"HP Grid: {treatment.hp_grid}")
        print(f"{'─' * 50}")

        keys = list(treatment.hp_grid.keys())
        values = list(treatment.hp_grid.values())

        hp_aucs: Dict[str, List[float]] = {}

        for combo in itertools.product(*values):
            hp_config = dict(zip(keys, combo))
            hp_key = str(hp_config)
            combo_tag = f"T{tid}_sens_{hp_config}"

            merged_overlay = {**treatment.config_overlay, **hp_config}
            combo_results = []

            for seed in SEEDS:
                if args.dry_run:
                    result = ExperimentResult(
                        treatment_id=tid, treatment_name=treatment.name,
                        seed=seed, hp_config=hp_config,
                    )
                else:
                    result = run_single_experiment(
                        treatment_overlay=merged_overlay,
                        seed=seed,
                        feature_set=args.feature_set,
                        split_mode=args.split_mode,
                        experiment_tag=combo_tag,
                    )
                    result.treatment_id = tid
                    result.treatment_name = treatment.name
                    result.hp_config = hp_config

                combo_results.append(result)

            all_results[combo_tag] = combo_results

            val_aucs = [r.val_auc for r in combo_results if r.val_auc > 0]
            if val_aucs:
                hp_aucs[hp_key] = val_aucs
                print(f"  {hp_config}: AUC = {np.mean(val_aucs):.4f} ± {np.std(val_aucs, ddof=1):.4f}")

        # Sensitivity summary
        if hp_aucs:
            all_means = [np.mean(v) for v in hp_aucs.values()]
            cv = np.std(all_means) / max(np.mean(all_means), 1e-10)
            print(f"\n  Sensitivity (CV of AUC means): {cv:.4f}")
            if cv < 0.01:
                print(f"  → Robust: HP 선택이 결과에 미미한 영향")
            elif cv < 0.03:
                print(f"  → Moderate: 적정 범위 내 안정적")
            else:
                print(f"  → Sensitive: fine-tuning 필요")

    return all_results


def run_phase5_final_test(
        args: argparse.Namespace,
        best_treatment_ids: List[int],
        best_hp_configs: Dict[int, Dict[str, Any]],
) -> List[ExperimentResult]:
    """Phase 5: Final Test Set Evaluation.

    ⚠️ Test set은 이 Phase에서 단 한 번만 평가.

    Comprehensive metrics:
        - Discrimination: AUC, AP, F1
        - Calibration: Brier, ECE
        - Robustness: across seeds
    """
    print("=" * 70)
    print("PHASE 5: FINAL TEST SET EVALUATION (ONE-TIME)")
    print(f"Active treatments: {[TREATMENTS[t].name for t in best_treatment_ids]}")
    print("=" * 70)
    print("\n⚠️  WARNING: Test set is evaluated ONCE. No further tuning allowed.\n")

    # Merge all selected treatments into a single overlay
    final_overlay: Dict[str, Any] = {}
    for tid in best_treatment_ids:
        final_overlay.update(TREATMENTS[tid].config_overlay)
        final_overlay.update(best_hp_configs.get(tid, {}))

    results = []

    for seed in SEEDS:
        print(f"  Final model | seed={seed}")

        if args.dry_run:
            result = ExperimentResult(
                treatment_id=-1, treatment_name="Final_Model",
                seed=seed,
                hp_config={t: best_hp_configs.get(t, {}) for t in best_treatment_ids},
            )
        else:
            result = run_single_experiment(
                treatment_overlay=final_overlay,
                seed=seed,
                feature_set=args.feature_set,
                split_mode=args.split_mode,
                experiment_tag="Final_Model",
            )
            result.treatment_id = -1
            result.treatment_name = "Final_Model"

        results.append(result)

    _print_phase5_summary(results)
    return results


# ──────────────────────────────────────────────────────────────
# 6. Summary / Reporting Utilities
# ──────────────────────────────────────────────────────────────

def _compute_ablation_summaries(
        treatment_results: Dict[int, List[ExperimentResult]],
        baseline_results: List[ExperimentResult],
) -> List[AblationSummary]:
    """Phase 2 결과를 Treatment별로 요약.

    각 Treatment에 대해:
    1. 5-seed mean ± std
    2. Baseline 대비 Δ 및 95% CI
    3. Cohen's d effect size
    """
    baseline_val_aucs = np.array([r.val_auc for r in baseline_results])
    summaries = []

    for tid, results in treatment_results.items():
        treatment = TREATMENTS[tid]
        val_aucs = np.array([r.val_auc for r in results])
        test_aucs = np.array([r.test_auc for r in results])

        # Paired differences
        n_pairs = min(len(val_aucs), len(baseline_val_aucs))
        deltas = val_aucs[:n_pairs] - baseline_val_aucs[:n_pairs]
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas, ddof=1)) if n_pairs > 1 else 0.0

        # 95% CI (t-distribution)
        ci_low, ci_high = delta_mean, delta_mean
        if n_pairs > 1 and delta_std > 0:
            try:
                from scipy.stats import t as t_dist
                t_crit = t_dist.ppf(0.975, df=n_pairs - 1)
                se = delta_std / math.sqrt(n_pairs)
                ci_low = delta_mean - t_crit * se
                ci_high = delta_mean + t_crit * se
            except ImportError:
                # fallback: z=1.96
                se = delta_std / math.sqrt(n_pairs)
                ci_low = delta_mean - 1.96 * se
                ci_high = delta_mean + 1.96 * se

        # Cohen's d = Δ / pooled_std
        var_t = np.var(val_aucs, ddof=1) if len(val_aucs) > 1 else 0
        var_b = np.var(baseline_val_aucs, ddof=1) if len(baseline_val_aucs) > 1 else 0
        pooled_std = math.sqrt((var_t + var_b) / 2)
        cohens_d = delta_mean / max(pooled_std, 1e-10)

        summary = AblationSummary(
            treatment_id=tid,
            treatment_name=treatment.name,
            val_auc_mean=float(np.mean(val_aucs)),
            val_auc_std=float(np.std(val_aucs, ddof=1)) if len(val_aucs) > 1 else 0.0,
            val_auc_seeds=val_aucs.tolist(),
            test_auc_mean=float(np.mean(test_aucs)),
            test_auc_std=float(np.std(test_aucs, ddof=1)) if len(test_aucs) > 1 else 0.0,
            delta_val_auc_mean=delta_mean,
            delta_val_auc_std=delta_std,
            delta_val_auc_ci_low=ci_low,
            delta_val_auc_ci_high=ci_high,
            cohens_d=cohens_d,
        )

        summaries.append(summary)

    # Holm-Bonferroni correction
    p_values = [s.delong_p_value for s in summaries]
    significant = holm_bonferroni(p_values, alpha=0.05)
    for s, sig in zip(summaries, significant):
        s.significant_after_correction = sig

    return summaries


def _print_phase1_summary(results: List[ExperimentResult]) -> None:
    """Phase 1 결과 요약 출력."""
    val_aucs = [r.val_auc for r in results if r.val_auc > 0]
    test_aucs = [r.test_auc for r in results if r.test_auc > 0]

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY: Baseline Reproduction")
    print("=" * 60)

    if val_aucs:
        print(f"  Val  AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs, ddof=1):.4f}")
        print(f"  Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs, ddof=1):.4f}")
        print(f"  Seeds: {[f'{a:.4f}' for a in val_aucs]}")
    else:
        print("  [No results yet — execute pipeline to populate]")

    print("=" * 60)


def _print_phase2_summary(summaries: List[AblationSummary]) -> None:
    """Phase 2 결과 요약 (Forest Plot 스타일 텍스트 테이블)."""
    print("\n" + "=" * 90)
    print("PHASE 2 SUMMARY: Single-Factor Ablation")
    print("=" * 90)
    print(f"{'Treatment':<30} {'Val AUC (μ±σ)':<18} {'Δ_val':<10} {'95% CI':<20} {'p-val':<8} {'Sig?':<5}")
    print("-" * 90)

    sorted_summaries = sorted(summaries, key=lambda s: -s.delta_val_auc_mean)

    for s in sorted_summaries:
        sig_marker = "***" if s.significant_after_correction else "ns"
        print(
            f"  T{s.treatment_id} {s.treatment_name:<25} "
            f"{s.val_auc_mean:.4f}±{s.val_auc_std:.4f}  "
            f"{s.delta_val_auc_mean:+.4f}   "
            f"[{s.delta_val_auc_ci_low:+.4f}, {s.delta_val_auc_ci_high:+.4f}]  "
            f"{s.delong_p_value:.4f}  "
            f"{sig_marker}"
        )

    print("-" * 90)
    print(f"  Correction: Holm-Bonferroni (m={len(summaries)}, α=0.05)")
    print(f"  Significant treatments: {sum(1 for s in summaries if s.significant_after_correction)}/{len(summaries)}")
    print("=" * 90)


def _print_phase5_summary(results: List[ExperimentResult]) -> None:
    """Phase 5 최종 결과 요약."""
    print("\n" + "=" * 70)
    print("PHASE 5 SUMMARY: FINAL TEST SET EVALUATION")
    print("=" * 70)

    val_aucs = [r.val_auc for r in results if r.val_auc > 0]
    test_aucs = [r.test_auc for r in results if r.test_auc > 0]
    val_briers = [r.val_brier for r in results if r.val_brier >= 0]
    val_eces = [r.val_ece for r in results if r.val_ece >= 0]

    metrics = {
        "Val AUC": val_aucs,
        "Test AUC": test_aucs,
        "Val Brier": val_briers,
        "Val ECE": val_eces,
    }

    for name, values in metrics.items():
        if values:
            print(f"  {name:<15}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")

    if val_aucs and test_aucs:
        gen_gap = np.mean(val_aucs) - np.mean(test_aucs)
        print(f"\n  Generalization Gap (Val - Test): {gen_gap:+.4f}")
        if abs(gen_gap) < 0.01:
            print("  → ✓ Good generalization (gap < 0.01)")
        elif abs(gen_gap) < 0.02:
            print("  → ⚠ Moderate gap (0.01 < gap < 0.02)")
        else:
            print("  → ✗ Large gap (gap > 0.02) — potential val overfitting")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 7. Main Entry Point
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoL Teamfight Prediction — Systematic Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase descriptions:
  1  Baseline reproduction (5 seeds)
  2  Single-factor ablation (each treatment independently)
  3  Interaction analysis (pairwise + cumulative)
  4  Hyperparameter sensitivity analysis
  5  Final test set evaluation (ONE-TIME)

Examples:
  python experiment_runner.py --phase 1
  python experiment_runner.py --phase 2 --treatment 1
  python experiment_runner.py --phase 2 --treatment all
  python experiment_runner.py --phase 3 --top-k 3
  python experiment_runner.py --phase 4
  python experiment_runner.py --phase 5
        """
    )

    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Experiment phase to execute")
    parser.add_argument("--treatment", type=str, default="all",
                        help="Treatment ID(s) for Phase 2. 'all' or comma-separated (e.g., '1,3,5')")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K treatments for Phase 3 interaction analysis")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                        help="Directory to save experiment results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without executing")

    # Pass-through args for the underlying experiment pipeline
    parser.add_argument("--feature-set", type=str, default="full", dest="feature_set")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-mode", type=str, default="patch_holdout", dest="split_mode")

    # [SPEED] GPU utilization overlay
    parser.add_argument("--speed", action="store_true", default=True,
                        help="Apply GPU speed overlay (RAM caching, batch scaling, torch.compile). Default: True")
    parser.add_argument("--no-speed", action="store_false", dest="speed",
                        help="Disable GPU speed overlay")
    parser.add_argument("--vram", type=float, default=24.0,
                        help="GPU VRAM in GB for auto batch sizing (default: 24.0)")
    parser.add_argument(
        "--speed-profile", "--speed_profile", "--speed-mode", "--speed_mode",
        dest="speed_profile",
        type=str,
        default="auto",
        choices=["none", "auto", "rtx50", "rtx5080", "aggressive"],
        help="Runtime speed profile to combine with overlay (default: auto)",
    )

    return parser


def _determine_best_treatments(
    output_dir: Path,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    """Phase 2/3 결과로부터 최적 Treatment 조합 결정.

    Selection 기준:
        1. Significant after Holm-Bonferroni correction
        2. Positive delta (Δ > 0)
        3. Phase 3에서 pairwise redundancy 없음

    Returns
    -------
    best_ids : List[int]
        선택된 treatment IDs
    best_hps : Dict[int, Dict[str, Any]]
        각 treatment의 최적 HP
    """
    # Phase 2 결과 로드
    phase2_data = _load_results(output_dir / "phase2_single_factor.json")
    if not phase2_data:
        print("[WARN] Phase 2 results not found, selecting all treatments")
        return list(TREATMENTS.keys()), {}

    # 각 treatment의 mean val_auc 계산 및 양의 delta 필터링
    best_ids = []
    best_hps: Dict[int, Dict[str, Any]] = {}

    for tid_str, results_list in phase2_data.items():
        tid = int(tid_str)
        if not isinstance(results_list, list):
            continue
        val_aucs = [r.get("val_auc", -1) for r in results_list if isinstance(r, dict)]
        val_aucs = [a for a in val_aucs if a > 0]

        if val_aucs:
            best_ids.append(tid)
            # HP는 첫 결과에서 추출
            if results_list and isinstance(results_list[0], dict):
                best_hps[tid] = results_list[0].get("hp_config", {})

    # Delta 기준 상위 3개로 제한 (conservative)
    if len(best_ids) > 3:
        best_ids = best_ids[:3]

    return best_ids, best_hps


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"LoL Teamfight Prediction — Ablation Study")
    print(f"Phase: {args.phase}")
    print(f"Output: {output_dir}")
    print(f"Seeds: {SEEDS}")
    print(f"Dry-run: {args.dry_run}")

    # [SPEED] Propagate speed settings via environment for run_single_experiment
    if args.speed:
        os.environ["LOL_SPEED_OVERLAY"] = "1"
        os.environ["LOL_VRAM_GB"] = str(args.vram)
        os.environ["LOL_SPEED_PROFILE"] = str(args.speed_profile)
        print(f"Speed overlay: ENABLED (VRAM={args.vram:.1f} GB, profile={args.speed_profile})")
    else:
        os.environ["LOL_SPEED_OVERLAY"] = "0"
        os.environ["LOL_SPEED_PROFILE"] = "none"
        print(f"Speed overlay: DISABLED")

    print(f"{'=' * 70}\n")

    if args.phase == 1:
        results = run_phase1_baseline(args)
        _save_results(output_dir / "phase1_baseline.json", results)

    elif args.phase == 2:
        # Parse treatment IDs
        if args.treatment.lower() == "all":
            treatment_ids = list(TREATMENTS.keys())
        else:
            treatment_ids = [int(x.strip()) for x in args.treatment.split(",")]

        # Load baseline results (must exist from Phase 1)
        baseline_path = output_dir / "phase1_baseline.json"
        baseline_data = _load_results(baseline_path) if baseline_path.exists() else None
        baseline_results = _deserialize_results(baseline_data) if baseline_data else []

        if not baseline_results:
            print("[WARN] No baseline results found. Run Phase 1 first.")
            print("       Proceeding with placeholder baseline...")
            baseline_results = [ExperimentResult(0, "Baseline", s, {}) for s in SEEDS]

        results = run_phase2_single_factor(args, treatment_ids, baseline_results)
        _save_results(output_dir / "phase2_single_factor.json", results)

    elif args.phase == 3:
        # Load Phase 1 & 2 results
        baseline_data = _load_results(output_dir / "phase1_baseline.json")
        phase2_data = _load_results(output_dir / "phase2_single_factor.json")

        if not baseline_data or not phase2_data:
            print("[ERROR] Phase 1 and Phase 2 results required. Run them first.")
            return

        baseline_results = _deserialize_results(baseline_data)
        phase2_results = _deserialize_phase2_results(phase2_data)

        results = run_phase3_interaction(args, phase2_results, baseline_results, top_k=args.top_k)
        _save_results(output_dir / "phase3_interaction.json", results)

    elif args.phase == 4:
        # Load Phase 1 & 2 results
        baseline_data = _load_results(output_dir / "phase1_baseline.json")
        phase2_data = _load_results(output_dir / "phase2_single_factor.json")

        if not baseline_data or not phase2_data:
            print("[ERROR] Phase 1 and Phase 2 results required. Run them first.")
            return

        baseline_results = _deserialize_results(baseline_data)
        phase2_results = _deserialize_phase2_results(phase2_data)

        results = run_phase4_sensitivity(args, phase2_results, baseline_results)
        _save_results(output_dir / "phase4_sensitivity.json", results)

    elif args.phase == 5:
        print("[Phase 5] Final test evaluation — determining best treatments...")

        best_ids, best_hps = _determine_best_treatments(output_dir)
        print(f"  Selected treatments: {best_ids}")
        print(f"  HP configs: {best_hps}")

        results = run_phase5_final_test(args, best_ids, best_hps)
        _save_results(output_dir / "phase5_final_test.json", results)

    print("\n[DONE] Experiment phase completed.")


# ──────────────────────────────────────────────────────────────
# 8. Serialization / Deserialization Utilities
# ──────────────────────────────────────────────────────────────

def _save_results(path: Path, results: Any) -> None:
    """결과를 JSON으로 저장."""
    try:
        if isinstance(results, list):
            data = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results]
        elif isinstance(results, dict):
            data = {}
            for k, v in results.items():
                k_str = str(k)
                if isinstance(v, list):
                    data[k_str] = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in v]
                else:
                    data[k_str] = v
        else:
            data = results

        # Remove non-serializable fields
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()
                        if k not in ("pred_logits_val", "pred_logits_test")}
            elif isinstance(obj, list):
                return [_clean(x) for x in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        data = _clean(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] {path}")
    except Exception as e:
        print(f"  [WARN] Failed to save results: {e}")


def _load_results(path: Path) -> Any:
    """JSON에서 결과 로드."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _deserialize_results(data: Any) -> List[ExperimentResult]:
    """JSON 데이터를 ExperimentResult 리스트로 역직렬화."""
    if not isinstance(data, list):
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            result = ExperimentResult(
                treatment_id=int(item.get("treatment_id", -1)),
                treatment_name=str(item.get("treatment_name", "")),
                seed=int(item.get("seed", 0)),
                hp_config=item.get("hp_config", {}),
                train_auc=float(item.get("train_auc", -1.0)),
                val_auc=float(item.get("val_auc", -1.0)),
                test_auc=float(item.get("test_auc", -1.0)),
                train_f1=float(item.get("train_f1", -1.0)),
                val_f1=float(item.get("val_f1", -1.0)),
                test_f1=float(item.get("test_f1", -1.0)),
                val_brier=float(item.get("val_brier", -1.0)),
                val_ece=float(item.get("val_ece", -1.0)),
                best_epoch=int(item.get("best_epoch", -1)),
                train_time_sec=float(item.get("train_time_sec", -1.0)),
            )
            results.append(result)
        except (KeyError, ValueError, TypeError):
            continue

    return results


def _deserialize_phase2_results(data: Any) -> Dict[int, List[ExperimentResult]]:
    """Phase 2 JSON을 Dict[int, List[ExperimentResult]]로 역직렬화."""
    if not isinstance(data, dict):
        return {}

    result_dict: Dict[int, List[ExperimentResult]] = {}
    for k, v in data.items():
        try:
            tid = int(k)
        except ValueError:
            continue
        if isinstance(v, list):
            result_dict[tid] = _deserialize_results(v)

    return result_dict


if __name__ == "__main__":
    main()
