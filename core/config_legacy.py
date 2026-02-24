from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


# -------------------------------------------------------------------
# Feature keys (timeline participantFrames)
# -------------------------------------------------------------------
CHAMPION_STATS_KEYS: List[str] = [
    "abilityHaste", "abilityPower", "armor", "armorPen", "armorPenPercent",
    "attackDamage", "attackSpeed", "bonusArmorPenPercent", "bonusMagicPenPercent",
    "ccReduction", "cooldownReduction", "health", "healthMax", "healthRegen",
    "lifesteal", "magicPen", "magicPenPercent", "magicResist", "movementSpeed",
    "omnivamp", "physicalVamp", "power", "powerMax", "powerRegen", "spellVamp",
]

DAMAGE_STATS_KEYS: List[str] = [
    "physicalDamageDone", "magicDamageDone", "trueDamageDone", "totalDamageDone",
    "physicalDamageDoneToChampions", "magicDamageDoneToChampions",
    "trueDamageDoneToChampions", "totalDamageDoneToChampions",
    "physicalDamageTaken", "magicDamageTaken", "trueDamageTaken", "totalDamageTaken",
]

ROLE_ORDER: List[str] = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SLOT_NAMES: List[str] = ["bTOP", "bJNG", "bMID", "bBOT", "bSUP", "rTOP", "rJNG", "rMID", "rBOT", "rSUP"]

# Summoner's Rift coordinate scale (RAW map units)
MAP_MAX: float = 16000.0  # deterministic coord normalization divisor (x/MAP_MAX, y/MAP_MAX)


# -------------------------------------------------------------------
# (NEW) Status feature design (Feature Contract)
# -------------------------------------------------------------------

DRAGON_SOUL_TYPES: List[str] = ["infernal", "ocean", "mountain", "cloud", "hextech", "chemtech"]

BUFF_DUR_SEC: Dict[str, int] = {
    "baron": 180,
    "elder": 150,
    "red": 120,
    "blue": 120,
}
FLASH_CD_SEC: int = 300  # base
VISION_RADIUS: float = 1200.0
VISION_RECENT_SEC: int = 90
VISION_CNT_DENOM: float = 10.0


# -------------------------------------------------------------------
# Node features (timeline participantFrames + derived status)
# -------------------------------------------------------------------
NODE_SNAPSHOT_FEATURE_NAMES: List[str] = [
    "x_norm", "y_norm",
    "level_norm", "xp_norm",
    "curGold_norm", "totalGold_norm", "gps_norm",
    "laneCS_norm", "jgCS_norm",
    "ccTime_norm",
    "hp_pct", "mp_pct",
    "alive",
]

NODE_STATUS_FEATURE_NAMES: List[str] = [
    "has_baron", "has_elder", "has_red", "has_blue",
    "baron_remain_norm", "elder_remain_norm", "red_remain_norm", "blue_remain_norm",
    *[f"soul_{t}" for t in DRAGON_SOUL_TYPES],
    "ult_level_norm",
    "flash_ready", "flash_remain_norm",
    "vision_ally_ward_cnt_norm",
    "vision_ward_kill_recent_norm",
    "vision_nearby_score_norm",
]

NODE_FEATURE_NAMES: List[str] = (
    NODE_SNAPSHOT_FEATURE_NAMES
    + NODE_STATUS_FEATURE_NAMES
    + [f"cs_{k}" for k in CHAMPION_STATS_KEYS]
    + [f"ds_{k}" for k in DAMAGE_STATS_KEYS]
)
F_NODE: int = len(NODE_FEATURE_NAMES)


# -------------------------------------------------------------------
# Event / Global features (unchanged)
# -------------------------------------------------------------------
EVENT_FEATURE_NAMES: List[str] = [
    "kills_t100", "kills_t200", "bounty_t100", "bounty_t200",
    "dragon_t100", "dragon_t200", "baron_t100", "baron_t200",
    "herald_t100", "herald_t200", "atakhan_t100", "atakhan_t200",
    "horde_t100", "horde_t200", "tower_t100", "tower_t200",
    "inhib_t100", "inhib_t200", "plate_t100", "plate_t200",
    "ward_placed_t100", "ward_placed_t200", "ward_kill_t100", "ward_kill_t200",
    "item_pur_t100", "item_pur_t200", "item_sold_t100", "item_sold_t200",
    "item_undo_t100", "item_undo_t200",
]
F_EVENT: int = len(EVENT_FEATURE_NAMES)

GLOBAL_FEATURE_NAMES: List[str] = [
    "time_norm",
    "goldDiff", "xpDiff", "avgLevelDiff",
    "csDiff_total", "csJgDiff",
    "aliveDiff",
    "killDiff_cum", "towerDiff_cum", "inhibDiff_cum",
    "dragonDiff_cum", "baronDiff_cum", "heraldDiff_cum",
    "atakhanDiff_cum", "plateDiff_cum", "hordeDiff_cum",
]
F_GLOBAL: int = len(GLOBAL_FEATURE_NAMES)


# -------------------------------------------------------------------
# Deterministic normalization denoms (NOT fitted scaler)
# -------------------------------------------------------------------
CS_DENOM: Dict[str, float] = {
    "abilityHaste": 200, "abilityPower": 1200, "armor": 400, "armorPen": 80,
    "armorPenPercent": 1.0, "attackDamage": 500, "attackSpeed": 3.0,
    "bonusArmorPenPercent": 1.0, "bonusMagicPenPercent": 1.0, "ccReduction": 1.0,
    "cooldownReduction": 1.0, "health": 6000, "healthMax": 6000, "healthRegen": 100,
    "lifesteal": 1.0, "magicPen": 80, "magicPenPercent": 1.0, "magicResist": 350,
    "movementSpeed": 800, "omnivamp": 1.0, "physicalVamp": 1.0, "power": 3000,
    "powerMax": 3000, "powerRegen": 100, "spellVamp": 1.0,
}
DS_DENOM: Dict[str, float] = {
    "physicalDamageDone": 300000, "magicDamageDone": 300000, "trueDamageDone": 100000, "totalDamageDone": 400000,
    "physicalDamageDoneToChampions": 100000, "magicDamageDoneToChampions": 100000,
    "trueDamageDoneToChampions": 50000, "totalDamageDoneToChampions": 150000,
    "physicalDamageTaken": 150000, "magicDamageTaken": 150000, "trueDamageTaken": 60000, "totalDamageTaken": 200000,
}

NODE_BASE_DENOM: Dict[str, float] = {
    "level": 18.0,
    "xp": 20000.0,
    "curGold": 4000.0,
    "totalGold": 25000.0,
    "gps": 30.0,
    "laneCS": 400.0,
    "jgCS": 250.0,
    "ccTime": 600.0,
    "vision_cnt": VISION_CNT_DENOM,
}


# -------------------------------------------------------------------
# Objective scoring (label shaping; configurable)
# -------------------------------------------------------------------
OBJ_SCORE: Dict[str, float] = {
    "DRAGON": 1.0, "BARON": 1.5, "RIFTHERALD": 0.8, "ATAKHAN": 1.3,
    "TOWER": 0.7, "INHIBITOR": 1.2, "PLATE": 0.3, "KILL": 0.25, "HORDE": 0.5,
}

DRAGON_PIT_XY: Tuple[float, float] = (9850.0, 4400.0)
BARON_PIT_XY: Tuple[float, float] = (5000.0, 10400.0)
TURRET_RANGE: float = 775.0


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
@dataclass
class CFG:
    # =========================================================
    # 0) Feature / Cache versioning (IMPORTANT)
    # =========================================================
    FEATURE_VERSION: str = "featV3_status_contract1"

    # =========================================================
    # 1) Data Paths
    # =========================================================
    DETAIL_DIR: Path = Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\detail")
    TIMELINE_DIR: Path = Path(r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\timeline")

    # =========================================================
    # 2) Output Paths
    # =========================================================
    OUTPUT_ROOT: Path = Path(r"D:\LOL_Project")
    CACHE_DIRNAME: str = "match_cache_fresh_v3_engage_status5"
    RUN_DIRNAME: str = "runs_teamfight_fresh_v3_engage_status9"
    META_DIRNAME: str = "meta"
    DATASET_DIRNAME: str = "dataset_teamfight_fresh_v3_engage_status4"

    # =========================================================
    # 3) Experiment / Split
    # =========================================================
    MODE: str = "all"  # all | build_cache | index | train | report | ablation
    MAX_MATCHES: Optional[int] = None

    PATCH_LEVEL: str = "major_minor"
    PATCH_ALLOWLIST: Optional[Tuple[str, ...]] = None

    SPLIT_MODE: str = "multi_patch"
    VAL_FRAC: float = 0.10
    TEST_FRAC: float = 0.10
    SEEDS: Tuple[int, ...] = (7,)
    SPLIT_GROUP_BY_MATCH_ID: bool = True

    MAX_FIGHTS_PER_MATCH: Optional[int] = 4
    FIGHT_SUBSAMPLE_STRATEGY: str = "uniform"
    FIGHT_SUBSAMPLE_SEED_OFFSET: int = 0

    # =========================================================
    # 4) Cache builder / logging
    # =========================================================
    MAX_T: Optional[int] = None
    CACHE_LOG_EVERY: int = 5000
    CACHE_VALIDATE_EXISTING: bool = False
    CACHE_REBUILD_CORRUPT: bool = True
    CACHE_LOG_ERRORS: bool = False

    # =========================================================
    # 5) Timeline resolution & interpolation
    # =========================================================
    FRAME_MS: int = 60000
    BIN_MS: int = 10000
    INTERP_METHOD: str = "linear"
    INTERP_XY: bool = True
    INTERP_SCALARS: bool = True

    # =========================================================
    # 6) Coordinate handling (NO fitted scaler on x,y)
    # =========================================================
    COORD_NORM_DIV: float = MAP_MAX

    SCALER_TYPE: Optional[str] = "standard"
    SCALER_EXCLUDE_PREFIXES: Tuple[str, ...] = (
        "x_", "y_", "pos_", "dist_", "angle_",
        "has_", "soul_", "flash_", "ult_",
    )

    # =========================================================
    # 6.1) Status feature switches & hyperparams
    # =========================================================
    USE_STATUS_FEATURES: bool = True
    USE_BUFF_STATUS: bool = True
    USE_DRAGON_SOUL_STATUS: bool = True
    USE_ULT_LEVEL: bool = True
    USE_FLASH_READY: bool = True
    USE_LOCAL_VISION: bool = True

    BUFF_DUR_SEC: Dict[str, int] = field(default_factory=lambda: dict(BUFF_DUR_SEC))
    FLASH_CD_SEC: int = FLASH_CD_SEC
    VISION_RADIUS: float = VISION_RADIUS
    VISION_RECENT_SEC: int = VISION_RECENT_SEC
    VISION_CNT_DENOM: float = VISION_CNT_DENOM

    # =========================================================
    # 7) Fight detection
    # =========================================================
    FIGHT_DETECT_ALGO: str = "engage_v2"
    FIGHT_DETECTOR: str = "engage_v2"

    # --- Anti multi-lane false positives ---
    REQUIRE_ENGAGED_PER_TEAM = 2  # 각 팀에서 "적이 R 안"인 인원 최소 3명
    REQUIRE_LCC_TOTAL = 4  # 최대 연결요소 총 인원 최소 6명
    REQUIRE_LCC_PER_TEAM = 2  # 최대 연결요소 내 팀별 최소 3명
    STANDOFF_MIN_PAIRS = 3
    CLUSTER_MAX_DIAMETER = 4000.0  # (옵션) LCC 플레이어들 최대 지름 제한 (좌표가 raw일 때)

    FIGHT_CONTEXT_MIN: int = 1
    FIGHT_HORIZON_SEC: int = 60
    FIGHT_HORIZON_MIN: int = 1

    START_OFFSET_MIN: int = 1
    FIGHT_MIN_GAP_MIN: int = 1
    DETECT_STEP_MS: int = 10000

    STANDOFF_RADIUS: float = 1800.0
    REQUIRE_ALIVE_PER_TEAM: int = 2

    ENGAGE_MIN_DIST_DROP: float = 250.0
    ENGAGE_MIN_PAIR_GAIN: int = 2
    VERIFY_KILL_IN_HORIZON: bool = True

    PROX_DIST_NORM: float = 1800.0 / MAP_MAX
    PROX_MIN_PAIRS: int = 8
    STANDOFF_NO_KILL_PREV_MIN: bool = True
    REQUIRE_SIGNAL_IN_HORIZON: bool = True

    USE_KILL_ANCHOR: bool = False
    KILL_ANCHOR_PRE_SEC: int = 15
    KILL_ANCHOR_COOLDOWN_SEC: int = 30
    DUMP_FIGHTS: bool = True
    DUMP_FIGHTS_DIRNAME: str = "fight_dumps"
    DUMP_FIGHTS_MAX_MATCHES: int = 5000
    # DUMP_FIGHTS_MATCH_ALLOWLIST = ["KR_....", "KR_...."]  # 특정 매치만 보고 싶으면

    # =========================================================
    # 8) Spatial anchors + fight type tagging
    # =========================================================
    BUILD_MAP_ANCHORS: bool = True
    MAP_ANCHOR_MIN_SAMPLES: int = 10
    MAP_ANCHOR_METHOD: str = "mean"
    PREFER_EVENT_POS: bool = True

    OBJ_NEAR_RADIUS: float = 1400.0
    TOWER_RANGE: float = TURRET_RANGE
    TOWER_NEAR_RADIUS: float = 900.0
    LANE_JUNGLE_MODE: str = "heuristic_v1"

    # =========================================================
    # 9) Labels (outcome shaping)
    # =========================================================
    W_KILL: float = 1.0
    W_GOLD: float = 0.5
    W_OBJ: float = 0.25
    GOLD_NORM: float = 1000.0
    LABEL_GOLD_METHOD: str = "linear"


    # graph ===================================================
    INTERACT_SIGMA_NORM = 0.12
    INTERACT_ALPHA_CLOSING = 4.0
    INTERACT_TOPK_ENEMY = 3
    INTERACT_TOPK_ALLY = 1
    INTERACT_EDGE_DIM = 64
    INTERACT_LAYERS = 2
    INTERACT_POOL_TOPK = 8

    # =========================================================
    # 10) Items (optional)
    # =========================================================
    USE_ITEMS: bool = True
    ITEM_HASH_DIM: int = 32

    # =========================================================
    # 11) Models / Ablation (PAPER-STYLE)
    # =========================================================
    # --- Canonical model IDs (main.py에서 이 ID로 분기) ---
    BASELINE_MODELS: Tuple[str, ...] = ("lgbm",)

    # user naming: ngru/nlstm/bigru/bilstm/transformer
    RNN_MODELS: Tuple[str, ...] = (
        "rnn_ugru",         # == ngru
        "rnn_ulstm",        # == nlstm
        "rnn_bigru",
        "rnn_bilstm",
        "rnn_transformer",
    )

    # user naming: gnn/gcn/stgnn/gnnsage/gnn-transformer
    # NOTE: "gnn"(generic)은 구현체가 없다면 gcn에 alias로 붙임
    GNN_MODELS: Tuple[str, ...] = (
        "gnn_gcn",              # generic gnn/gcn baseline
        "gnn_graphsage",
        "gnn_stgnn",
        "gnn_graphtransformer", # gnn-transformer
    )

    # (선택) 모델 이름 alias: 사용자가 말한 토큰을 내부 canonical id로 맵핑
    MODEL_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        # baselines
        "lightgbm": "lgbm",
        "lgb": "lgbm",
        "lgbm": "lgbm",

        # rnn family
        "ngru": "rnn_ugru",
        "ugru": "rnn_ugru",
        "nlstm": "rnn_ulstm",
        "ulstm": "rnn_ulstm",
        "bigru": "rnn_bigru",
        "bilstm": "rnn_bilstm",
        "transformer": "rnn_transformer",

        # gnn family
        "gnn": "gnn_gcn",  # generic -> gcn로 매핑
        "gcn": "gnn_gcn",
        "stgnn": "gnn_stgnn",
        "gnnsage": "gnn_graphsage",
        "graphsage": "gnn_graphsage",
        "gnn-transformer": "gnn_graphtransformer",
        "graphtransformer": "gnn_graphtransformer",
    })

    # --- Paper-style ablation stages (순서) ---
    # baseline
    # deep_only
    # baseline_plus_rnn
    # baseline_plus_gnn
    # rnn_plus_gnn
    # fusion_best (baseline+rnn+gnn 중 최고 조합)
    ABLATION_PLAN: Tuple[str, ...] = (
        "baseline",
        "deep_only",
        "baseline_plus_rnn",
        "baseline_plus_gnn",
        "rnn_plus_gnn",
        "fusion_best",
    )

    # best selection policy
    ABLATION_SELECT_SPLIT: str = "val"     # val | test
    ABLATION_SELECT_METRIC: str = "auc"    # auc | ap | acc | f1 ...
    FUSION_SELECT_STRATEGY: str = "grid_best"
    # grid_best: baseline×rnn×gnn 모든 조합 중 best
    # family_best: baseline best + rnn best + gnn best 하나로 fusion

    # (derived at runtime; keep for backward compatibility)
    ABLATION_GROUPS: Dict[str, Tuple[str, ...]] = field(default_factory=dict, init=False)
    MODEL_LIST: Tuple[str, ...] = field(default_factory=tuple)  # main.py의 기존 루프 호환

    # =========================================================
    # 12) Training (deep)
    # =========================================================
    REQUIRE_CUDA: bool = True
    BATCH_SIZE: int = 64
    LR: float = 2e-4
    WEIGHT_DECAY: float = 2e-4
    EPOCHS: int = 15
    PATIENCE: int = 4
    LOG_EVERY: int = 100

    GRAD_CLIP_NORM: float = 5.0

    DROPOUT: float = 0.35
    RNN_HIDDEN: int = 64
    RNN_LAYERS: int = 2

    HEAD_HIDDEN: int = 128
    HEAD_LAYERS: int = 2

    GNN_DIM: int = 64
    GNN_DROPOUT: float = 0.35
    GNN_NORM: bool = True

    ADJ_SOFT: bool = True
    ADJ_SIGMA_NORM: float = 0.125
    TEAM_EDGE_WEIGHT: float = 1.0

    TRANS_D_MODEL: int = 256
    TRANS_NHEAD: int = 4
    TRANS_LAYERS: int = 3
    TRANS_DROPOUT: float = 0.2
    TRANS_FF_MULT: int = 4
    TRANS_MAX_LEN: int = 512

    FUSION_GATE_H: int = 8
    FUSION_MLP_H: int = 32

    # =========================================================
    # 13) Reporting
    # =========================================================
    CLS_THRESHOLD: float = 0.5
    EARLY_STOP_METRIC: str = "auc"
    PREC_AT_K: Tuple[int, ...] = (50, 100, 200, 500)
    PREC_AT_FRAC: Tuple[float, ...] = (0.01, 0.05, 0.10)

    LGB_PERM_IMPORTANCE: bool = False
    LGB_SHAP: bool = True
    DEEP_PERM_IMPORTANCE: bool = False

    # =========================================================
    # 14) Multicollinearity / redundant feature removal (tabular)
    # =========================================================
    DROP_CORR_FEATURES: bool = True
    CORR_THRESHOLD: float = 0.98
    DROP_VIF_FEATURES: bool = False
    VIF_THRESHOLD: float = 12.0

    # =========================================================
    # 15) Debug / Diagnostics
    # =========================================================
    DEBUG_GNN: bool = False

    # =========================================================
    # 16) Baseline Params (LGBM / XGB)
    # =========================================================
    BASELINE_LGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_data_in_leaf=200,
        min_gain_to_split=0.0,

        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,

        reg_alpha=1.0,
        reg_lambda=5.0,

        max_bin=255,
        n_jobs=-1,
    ))

    # xgboost는 main에서 XGBClassifier로 사용할 예정 (아직 미구현이면 main에서 skip 가능)
    BASELINE_XGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=50,
        gamma=0.0,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
    ))

    # -------------------------
    # derived config build
    # -------------------------
    def __post_init__(self):
        baseline = tuple(self.BASELINE_MODELS)
        rnn = tuple(self.RNN_MODELS)
        gnn = tuple(self.GNN_MODELS)

        self.ABLATION_GROUPS = {
            "baseline": baseline,
            "deep_only": rnn + gnn,
            "baseline_plus_rnn": baseline + rnn,
            "baseline_plus_gnn": baseline + gnn,
            "rnn_plus_gnn": rnn + gnn,
            # fusion_best는 main에서 grid_best/family_best 정책으로 조합 생성
            "fusion_best": ("fusion_auto_best",),
        }

        # main.py의 기존 루프 호환용: “전체 모델 후보 집합”
        self.MODEL_LIST = baseline + rnn + gnn + ("fusion_auto_best",)


# -------------------------------------------------------------------
# Instantiate + dirs
# -------------------------------------------------------------------
cfg = CFG()

cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = cfg.OUTPUT_ROOT / "cache" / cfg.CACHE_DIRNAME
RUN_DIR = cfg.OUTPUT_ROOT / cfg.RUN_DIRNAME
META_DIR = cfg.OUTPUT_ROOT / cfg.META_DIRNAME
DATASET_DIR = cfg.OUTPUT_ROOT / "dataset" / cfg.DATASET_DIRNAME

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Export baseline params as module-level dicts (main.py expects these)
# -------------------------------------------------------------------
BASELINE_LGB_PARAMS: Dict[str, Any] = dict(cfg.BASELINE_LGB_PARAMS)

# ---- contracts (single source of truth) ----
from feature_contract import build_feature_contract
from time_contract import TimeContract

FEATURE_CONTRACT = build_feature_contract(
    node_names=NODE_FEATURE_NAMES,
    event_names=EVENT_FEATURE_NAMES,
    global_names=GLOBAL_FEATURE_NAMES,
)

# Core convention: FightRef.t_start is minute-index into cache['minute_ts']
TIME_CONTRACT = TimeContract(frame_ms=int(getattr(cfg, 'FRAME_MS', 60000)))

# ---- indices (kept for backward compatibility) ----
NODE_IDX: Dict[str, int] = dict(FEATURE_CONTRACT.node_idx)
EVENT_IDX: Dict[str, int] = dict(FEATURE_CONTRACT.event_idx)
GLOBAL_IDX: Dict[str, int] = dict(FEATURE_CONTRACT.global_idx)

# ---- dims (kept for backward compatibility) ----
F_NODE: int = int(FEATURE_CONTRACT.f_node)
F_EVENT: int = int(FEATURE_CONTRACT.f_event)
F_GLOBAL: int = int(FEATURE_CONTRACT.f_global)

# ---- item hash ----
ITEM_HASH_DIM: int = int(cfg.ITEM_HASH_DIM)
ITEM_HASH_NAMES: List[str] = [f"itemhash{i}" for i in range(ITEM_HASH_DIM)]
