from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Treatment:
    """Experiment treatment definition."""

    id: int
    name: str
    short_name: str
    config_overlay: Dict[str, Any]
    hp_grid: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""


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
    8: Treatment(
        id=8,
        name="Interpolation-Free 60s Baseline",
        short_name="no_interp_60s",
        config_overlay={
            # Model-side sequence interpolation off + 60s single snapshot
            "INTERP_XY": False,
            "INTERP_SCALARS_METHOD": "ffill",
            "BIN_MS": 60000,
            # Detector-side dense XY interpolation off
            "DETECT_STEP_MS": 60000,
            "TF2_GRID_STEP_MS": 60000,
            "TF2_USE_FRAME_INTERP": False,
            "TF2_USE_KILL_TRAJECTORY_INTERP": False,
        },
        hp_grid={},
        description=(
            "Control baseline without interpolation.\n"
            "- Detection: no 5s interpolation between 60s frames, no kill-trajectory interpolation\n"
            "- Model input: 60s snapshot only (BIN_MS=60000)\n"
            "Used to test whether interpolation creates meaningful gains."
        ),
    ),
    9: Treatment(
        id=9,
        name="Detector Kill-Trajectory Interpolation Off",
        short_name="det_no_killtraj_interp",
        config_overlay={
            "DETECT_STEP_MS": 5000,
            "TF2_GRID_STEP_MS": 5000,
            "TF2_USE_FRAME_INTERP": True,
            "TF2_USE_KILL_TRAJECTORY_INTERP": False,
        },
        hp_grid={},
        description=(
            "Ablation for teamfight detector validity.\n"
            "Keeps 5s dense grid interpolation but disables kill-event trajectory override,\n"
            "testing whether the kill-anchored interpolation algorithm contributes."
        ),
    ),
    10: Treatment(
        id=10,
        name="Detector 60s Grid Only",
        short_name="det_60s_grid_only",
        config_overlay={
            "DETECT_STEP_MS": 60000,
            "TF2_GRID_STEP_MS": 60000,
            "TF2_USE_FRAME_INTERP": False,
            "TF2_USE_KILL_TRAJECTORY_INTERP": False,
        },
        hp_grid={},
        description=(
            "Ablation for detector-level temporal resolution.\n"
            "Fight detection uses only raw 60s position frames (no dense interpolation)."
        ),
    ),
}


TREATMENT_GROUPS: Dict[str, Tuple[int, ...]] = {
    # User-facing study shortcuts
    "interpolation_study": (8,),
    "detector_validity": (9, 10),
    "teamfight_interp_study": (8, 9, 10),
    # Aliases
    "interp": (8,),
    "detector": (9, 10),
}


SEEDS: Tuple[int, ...] = (7, 42, 123)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    treatment_id: int
    treatment_name: str
    seed: int
    hp_config: Dict[str, Any]

    train_auc: float = -1.0
    val_auc: float = -1.0
    test_auc: float = -1.0

    train_ap: float = -1.0
    val_ap: float = -1.0
    test_ap: float = -1.0

    train_f1: float = -1.0
    val_f1: float = -1.0
    test_f1: float = -1.0

    val_brier: float = -1.0
    val_ece: float = -1.0
    test_brier: float = -1.0
    test_ece: float = -1.0

    best_epoch: int = -1
    train_time_sec: float = -1.0

    # Data diagnostics (fight/sample counts)
    n_train: int = -1
    n_val: int = -1
    n_test: int = -1
    n_fights_all: int = -1

    pred_logits_val: Optional[Dict[str, float]] = None
    pred_logits_test: Optional[Dict[str, float]] = None


@dataclass
class AblationSummary:
    """Aggregated multi-seed summary for a single treatment."""

    treatment_id: int
    treatment_name: str

    val_auc_mean: float = 0.0
    val_auc_std: float = 0.0
    val_auc_seeds: List[float] = field(default_factory=list)

    test_auc_mean: float = 0.0
    test_auc_std: float = 0.0

    delta_val_auc_mean: float = 0.0
    delta_val_auc_std: float = 0.0
    delta_val_auc_ci_low: float = 0.0
    delta_val_auc_ci_high: float = 0.0

    delong_p_value: float = 1.0
    mcnemar_p_value: float = 1.0
    significant_after_correction: bool = False

    cohens_d: float = 0.0
