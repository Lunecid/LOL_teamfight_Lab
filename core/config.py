"""config.py — Central configuration (single source of truth).

# ═══════════════════════════════════════════════════════════════
# [P2-STRUCT-3] Import DAG Layer: 1 (depends on Layer 0 only)
#
#   Layer 0: feature_contract.py   (no project imports)
#   Layer 1: config.py             ← THIS MODULE
#   Layer 2: features.py           (imports Layer 0, 1)
#   Layer 3: contract.py           (imports Layer 0, 1, 2)
#
# RULE: This module may ONLY import from feature_contract.py.
# ═══════════════════════════════════════════════════════════════

Changes from original:
  [P0-RULE]    VERIFY_KILL_IN_HORIZON default → True; USE_KILL_ANCHOR → False.
  [P0-SEED]    Paper protocol uses 3 seeds (7, 42, 123).
  [P0-CAP]     DROPOUT 0.35→0.20, RNN_HIDDEN 64→128, GNN_DIM 64→96,
               GNN_DROPOUT 0.35→0.25, TCN_DROPOUT 0.35→0.20 (underfitting fix).
  [P1-PATH]    Hardcoded Windows paths → environment-variable fallbacks.
  [P1-LABEL]   Added LABEL_TIE_STRATEGY to handle score ties (was biased to 0).
  [P1-LR]      LR 2e-4→5e-4, EPOCHS 10→15, PATIENCE 4→3 (convergence).
  [P1-DEEP]    Added DEEP_MAX_TRAIN cap for deep model train subsampling.
  [P1-SCALER]  SCALER_EXCLUDE_PREFIXES now includes "cs_", "ds_" to prevent
               double-normalisation of already-normalised champion/damage stats.
  [P1-NORM]    Added USE_CUMULATIVE_DELTA flag + TIME_NORM_CUMULATIVE for
               delta/rate features on cumulative statistics.
  [P2-CFG]     Removed ``from dataclasses import field`` / ``from typing …``
               inside class body (shadowed module-level imports, caused linter
               warnings).
  [P2-DUP]     Removed duplicate "Deterministic normalization" comment block.
  [P2-SIGMA]   Added ADJ_SIGMA_ADAPTIVE flag for game-phase aware σ.
  [REC-1]      Hybrid h₀ conditioning: HYBRID_H0_ENABLED, HYBRID_H0_PROJ_DIM,
               HYBRID_H0_DROPOUT + hybrid model entries in RNN_MODELS/ALIASES.
  [REC-4a]     Recency weighting: RECENCY_WEIGHT_ENABLED, RECENCY_WEIGHT_TAU.
  [REC-4b]     Temperature scaling: TEMP_SCALING_ENABLED.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    # core/config.py -> project root
    return Path(__file__).resolve().parents[1]


def _path_from_env_or_default(env_key: str, windows_default: str, posix_rel_default: str) -> Path:
    """Return env override when set; otherwise platform-aware default path.

    Why this exists:
      - Previous defaults used hardcoded Windows paths on all platforms.
      - On macOS/Linux those strings become literal relative directory names
        (for example, `D:\\LOL_Project`) inside the workspace.
    """
    raw = str(os.environ.get(env_key, "")).strip()
    if raw:
        return Path(raw)
    if os.name == "nt":
        return Path(windows_default)
    return (_repo_root() / posix_rel_default).resolve()


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

# Timeline championStats in some patches/regions are emitted in percent-like 0~100 scale
# for these keys (instead of canonical 0~1 ratio). We auto-correct by /100 when |v|>2.
CHAMPION_STATS_DIV100_KEYS: Tuple[str, ...] = (
    "attackSpeed",
    "armorPenPercent", "bonusArmorPenPercent", "bonusMagicPenPercent", "magicPenPercent",
    "ccReduction", "cooldownReduction",
    "lifesteal", "omnivamp", "physicalVamp", "spellVamp",
)

DAMAGE_STATS_KEYS: List[str] = [
    "physicalDamageDone", "magicDamageDone", "trueDamageDone", "totalDamageDone",
    "physicalDamageDoneToChampions", "magicDamageDoneToChampions",
    "trueDamageDoneToChampions", "totalDamageDoneToChampions",
    "physicalDamageTaken", "magicDamageTaken", "trueDamageTaken", "totalDamageTaken",
]

ROLE_ORDER: List[str] = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SLOT_NAMES: List[str] = [
    "bTOP", "bJNG", "bMID", "bBOT", "bSUP",
    "rTOP", "rJNG", "rMID", "rBOT", "rSUP",
]

# Summoner's Rift coordinate scale (RAW map units)
MAP_MAX: float = 16000.0


# -------------------------------------------------------------------
# Status feature design
# -------------------------------------------------------------------
DRAGON_SOUL_TYPES: List[str] = [
    "infernal", "ocean", "mountain", "cloud", "hextech", "chemtech",
]

BUFF_DUR_SEC: Dict[str, int] = {
    "baron": 180, "elder": 150, "red": 120, "blue": 120,
}
FLASH_CD_SEC: int = 300
VISION_RADIUS: float = 1200.0
VISION_RECENT_SEC: int = 90
VISION_CNT_DENOM: float = 10.0


# -------------------------------------------------------------------
# Rune Feature Keys
# -------------------------------------------------------------------
RUNE_FEATURE_NAMES: List[str] = [
    "primary_style_id", "sub_style_id",
    "primary_rune_1", "primary_rune_2", "primary_rune_3", "primary_rune_4",
    "sub_rune_1", "sub_rune_2",
    "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
]


# -------------------------------------------------------------------
# Node features
# -------------------------------------------------------------------
NODE_SNAPSHOT_FEATURE_NAMES: List[str] = [
    "champion_id",
    "champion_name_id",
    "summoner_spell_1_id", "summoner_spell_2_id",
    "x_norm", "y_norm",
    "level_norm", "xp_norm",
    "curGold_norm", "totalGold_norm", "gps_norm",
    "laneCS_norm", "jgCS_norm",
    "ccTime_norm",
    "hp_pct", "mp_pct",
    "alive",
]

NODE_STATUS_FEATURE_NAMES: List[str] = [
    "has_baron", "has_elder",
    "baron_remain_norm", "elder_remain_norm",
    *[f"soul_{t}" for t in DRAGON_SOUL_TYPES],
    "ult_level_norm",
]

NODE_FEATURE_NAMES: List[str] = (
    NODE_SNAPSHOT_FEATURE_NAMES
    + NODE_STATUS_FEATURE_NAMES
    + RUNE_FEATURE_NAMES
    + [f"cs_{k}" for k in CHAMPION_STATS_KEYS]
    + [f"ds_{k}" for k in DAMAGE_STATS_KEYS]
)
F_NODE: int = len(NODE_FEATURE_NAMES)


# -------------------------------------------------------------------
# Event / Global features
# -------------------------------------------------------------------
EVENT_FEATURE_NAMES: List[str] = [
    "kills_t100", "kills_t200", "bounty_t100", "bounty_t200",
    "shutdown_kill_t100", "shutdown_kill_t200",
    "killstreak_t100", "killstreak_t200",
    "multikill_t100", "multikill_t200",
    "ace_t100", "ace_t200",
    "dragon_t100", "dragon_t200", "baron_t100", "baron_t200",
    "herald_t100", "herald_t200", "atakhan_t100", "atakhan_t200",
    "horde_t100", "horde_t200", "tower_t100", "tower_t200",
    "inhib_t100", "inhib_t200", "plate_t100", "plate_t200",
    "obj_bounty_t100", "obj_bounty_t200",
    "ward_placed_t100", "ward_placed_t200", "ward_kill_t100", "ward_kill_t200",
    "control_ward_placed_t100", "control_ward_placed_t200",
    "control_ward_kill_t100", "control_ward_kill_t200",
    "item_pur_t100", "item_pur_t200", "item_sold_t100", "item_sold_t200",
    "item_undo_t100", "item_undo_t200",
]
F_EVENT: int = len(EVENT_FEATURE_NAMES)

BAN_FEATURE_NAMES: List[str] = (
    [f"blue_ban_{i}" for i in range(5)]
    + [f"red_ban_{i}" for i in range(5)]
)

GLOBAL_FEATURE_NAMES: List[str] = [
    "time_norm",
    *BAN_FEATURE_NAMES,
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
    "abilityHaste": 200, "abilityPower": 1200, "armor": 500, "armorPen": 80,
    "armorPenPercent": 1.0, "attackDamage": 600, "attackSpeed": 3.0,
    "bonusArmorPenPercent": 1.0, "bonusMagicPenPercent": 1.0, "ccReduction": 1.0,
    "cooldownReduction": 1.0, "health": 8500, "healthMax": 8500, "healthRegen": 100,
    "lifesteal": 1.0, "magicPen": 80, "magicPenPercent": 1.0, "magicResist": 400,
    "movementSpeed": 800, "omnivamp": 1.0, "physicalVamp": 1.0, "power": 3000,
    "powerMax": 3000, "powerRegen": 100, "spellVamp": 1.0,
}

DS_DENOM: Dict[str, float] = {
    "physicalDamageDone": 300000, "magicDamageDone": 300000,
    "trueDamageDone": 100000, "totalDamageDone": 400000,
    "physicalDamageDoneToChampions": 100000, "magicDamageDoneToChampions": 100000,
    "trueDamageDoneToChampions": 50000, "totalDamageDoneToChampions": 150000,
    "physicalDamageTaken": 150000, "magicDamageTaken": 150000,
    "trueDamageTaken": 60000, "totalDamageTaken": 200000,
}

NODE_BASE_DENOM: Dict[str, float] = {
    "level": 18.0,
    "xp": 28000.0,
    "curGold": 6000.0,
    "totalGold": 35000.0,
    "gps": 30.0,
    "laneCS": 400.0,
    "jgCS": 250.0,
    "ccTime": 600.0,
    "vision_cnt": VISION_CNT_DENOM,
}


# -------------------------------------------------------------------
# Objective scoring
# -------------------------------------------------------------------
OBJ_SCORE: Dict[str, float] = {
    "DRAGON": 1.0, "BARON": 1.5, "RIFTHERALD": 0.8, "ATAKHAN": 1.3,
    "TOWER": 0.7, "INHIBITOR": 1.2, "PLATE": 0.3, "KILL": 0.25, "HORDE": 0.5,
}

DRAGON_PIT_XY: Tuple[float, float] = (9850.0, 4400.0)
BARON_PIT_XY: Tuple[float, float] = (5000.0, 10400.0)
TURRET_RANGE: float = 775.0


# ===================================================================
#  CFG  — Central configuration dataclass
# ===================================================================
@dataclass
class CFG:
    # =========================================================
    # 0) Feature / Cache versioning
    # =========================================================
    FEATURE_VERSION: str = "featV7_schema_pruned_status_runes_bans_spells_styles_bin5s"

    # =========================================================
    # 1) Data Paths  [FIX-PATH] env var override 가능, 기본값은 원본 경로
    # =========================================================
    # NOTE: The hardcoded Windows paths below (DETAIL_DIR / TIMELINE_DIR, and
    # OUTPUT_ROOT in section 2) are the original author's LOCAL defaults. They are
    # intentionally kept as-is. For portability, set the LOL_DETAIL_DIR /
    # LOL_TIMELINE_DIR / LOL_OUTPUT_ROOT environment variables, which override
    # these defaults (see _path_from_env_or_default).
    DETAIL_DIR: Path = field(default_factory=lambda: _path_from_env_or_default(
        "LOL_DETAIL_DIR",
        r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\detail",
        "data/raw/matches/kr/detail",
    ))
    TIMELINE_DIR: Path = field(default_factory=lambda: _path_from_env_or_default(
        "LOL_TIMELINE_DIR",
        r"C:\Users\todtj\PycharmProjects\Lol_project\data\raw\matches\kr\timeline",
        "data/raw/matches/kr/timeline",
    ))

    # =========================================================
    # 2) Output Paths  [FIX-PATH]
    # =========================================================
    OUTPUT_ROOT: Path = field(default_factory=lambda: _path_from_env_or_default(
        "LOL_OUTPUT_ROOT",
        r"D:\LOL_Project",
        "outputs",
    ))
    CACHE_DIRNAME: str = "match_cache_fresh_v3_engage_status13"
    RUN_DIRNAME: str = "runs_teamfight_fresh_v3_engage_status13"
    META_DIRNAME: str = "meta"
    DATASET_DIRNAME: str = "dataset_teamfight_fresh_v3_engage_status4"

    # =========================================================
    # 3) Experiment / Split
    # =========================================================
    MODE: str = "all"
    MAX_MATCHES: Optional[int] = None

    PATCH_LEVEL: str = "major_minor"
    PATCH_ALLOWLIST: Optional[Tuple[str, ...]] = None

    SPLIT_MODE: str = "multi_patch"
    VAL_FRAC: float = 0.20
    TEST_FRAC: float = 0.10

    # Paper protocol seeds for three-seed mean/std.
    SEEDS: Tuple[int, ...] = (7, 42, 123)

    SPLIT_GROUP_BY_MATCH_ID: bool = True

    # [GUARD] Fail-fast when an explicitly requested patch split (train/val/test)
    # yields zero refs (almost always a missing-patch cache). True to bypass.
    ALLOW_EMPTY_SPLITS: bool = False

    # None => keep all detected fights per match (no per-match cap).
    MAX_FIGHTS_PER_MATCH: Optional[int] = None
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
    BIN_MS: int = 5000
    # Fight detector dense XY interpolation method in fights.py:
    # "zoh" keeps step-hold samples (no linear path assumption) while
    # still enabling DETECT_STEP_MS dense scanning.
    INTERP_METHOD: str = "zoh"
    # XY interpolation curve used between two anchor points.
    # Supported: "linear", "cosine", "exponential", "cubic"
    #   linear      – straight-line lerp (fast, simple)
    #   cosine      – smooth ease-in/out via cos curve
    #   exponential – 1-e^(-k*t), accelerates toward the target position
    #   cubic       – cubic Hermite (ease-in-out with zero endpoint tangents)
    # The discontinuity guard wraps whichever curve is selected.
    INTERP_XY_METHOD: str = "linear_guard_midstep"
    INTERP_XY_CURVE: str = "exponential"
    # [SPEC-FIX] "XY만 보간, 나머지 피처는 보간 금지" — node/global use
    # strict-before 60s snapshot (piecewise-constant / step-hold).
    INTERP_SCALARS_METHOD: str = "ffill"

    # Exponential decay rate for INTERP_XY_CURVE="exponential".
    # Higher k → faster convergence toward the target position.
    # k=3: at t=0.5, alpha≈0.78; at t=1.0, alpha≈0.95
    INTERP_EXP_K: float = 3.0

    XY_DISCONT_DIST_RAW: float = 7000.0
    XY_DISCONT_USE_ALIVE: bool = True
    XY_GUARD_MODE: str = "hold"

    INTERP_XY: bool = True
    INTERP_SCALARS: bool = True

    # =========================================================
    # 6) Coordinate handling
    # =========================================================
    COORD_NORM_DIV: float = MAP_MAX

    SCALER_TYPE: Optional[str] = "standard"
    # [P1-SCALER] Added "cs_", "ds_" to avoid double-normalisation of
    # already log1p-normalised champion/damage stats.
    # [P1-3 FIX] Added categorical ID prefixes to prevent StandardScaler
    # from corrupting integer indices used by embedding lookups.
    #
    #   수학적 근거:
    #     Emb(idx) requires idx ∈ Z (정수 인덱스).
    #     StandardScaler 적용 시: idx' = (idx - μ) / σ ∈ R
    #     → Emb(idx') = Emb(floor(idx'))이 되어 의미 없는 임베딩 반환
    #     또는 IndexError 발생 (idx' < 0 또는 > vocab_size)
    #
    #   해결: 범주형 ID 피처를 스케일링 대상에서 제외
    #     h_i = [MLP(x_cont); Σ_k Emb_k(id_k)]
    #     x_cont: 스케일링 대상 (연속형)
    #     id_k:   스케일링 비대상 (범주형 정수)
    SCALER_EXCLUDE_PREFIXES: Tuple[str, ...] = (
        # 좌표/방향 (이미 coord_norm_div로 정규화)
        "x_", "y_", "pos_", "dist_", "angle_",
        # 불리언/상태 (0/1 or 이미 정규화)
        "has_", "soul_", "flash_", "ult_",
        # 챔피언/데미지 스탯 (이미 denom/clip으로 정규화)
        "cs_", "ds_",
        # [P1-3 NEW] 범주형 정수 ID (임베딩 룩업용 — 절대 스케일링 금지)
        "champion_id",
        "champion_name_id",
        "primary_rune_", "sub_rune_",
        "primary_style_id", "sub_style_id",
        "stat_perk_",
        "blue_ban_", "red_ban_",
        "summoner_spell_",
    )

    # =========================================================
    # 6.1) Status feature switches
    # =========================================================
    USE_STATUS_FEATURES: bool = True
    USE_BUFF_STATUS: bool = True
    USE_DRAGON_SOUL_STATUS: bool = True
    USE_ULT_LEVEL: bool = True
    USE_FLASH_READY: bool = True
    USE_LOCAL_VISION: bool = True

    CHAMPION_NAME_VOCAB: int = 4096
    SUMMONER_SPELL_VOCAB: int = 512
    RUNE_STYLE_VOCAB: int = 256

    BUFF_DUR_SEC: Dict[str, int] = field(default_factory=lambda: dict(BUFF_DUR_SEC))
    FLASH_CD_SEC: int = FLASH_CD_SEC
    VISION_RADIUS: float = VISION_RADIUS
    VISION_RECENT_SEC: int = VISION_RECENT_SEC
    VISION_CNT_DENOM: float = VISION_CNT_DENOM

    # =========================================================
    # 6.2) Cumulative stats time normalisation  [P1-NORM]
    # =========================================================
    # If True, damage stats are converted to per-minute rates:
    #   d̂ᵢ(t) = dᵢ(t) / (t_min + ε)
    # Additionally, Δ-features are appended:
    #   Δdᵢ(t) = dᵢ(t) − dᵢ(t−1)
    USE_CUMULATIVE_DELTA: bool = True
    TIME_NORM_CUMULATIVE: bool = True

    # =========================================================
    # 7) Fight detection
    # =========================================================
    # Only supported detector: "teamfight_v2" (kill-cluster-based).
    FIGHT_DETECT_ALGO: str = "teamfight_v2"
    FIGHT_DETECTOR: str = "teamfight_v2"

    REQUIRE_ENGAGED_PER_TEAM: int = 2
    REQUIRE_LCC_TOTAL: int = 4
    REQUIRE_LCC_PER_TEAM: int = 2
    STANDOFF_MIN_PAIRS: int = 3
    CLUSTER_MAX_DIAMETER: float = 4000.0

    FIGHT_CONTEXT_SEC: int = 30
    FIGHT_CONTEXT_MIN: int = 1
    FIGHT_HORIZON_SEC: int = 30
    FIGHT_HORIZON_MIN: int = 1
    # Predict earlier than engage by this gap:
    # observation window ends at (engage_ts - prediction_gap_ms),
    # while label window starts at engage_ts and ends at
    #   - horizon_end_ts (continuous merged fight), if provided
    #   - otherwise engage_ts + horizon.
    PREDICTION_GAP_MS: int = 0
    MAX_MERGED_FIGHT_DURATION_MS = 60000

    START_OFFSET_MIN: int = 2
    FIGHT_MIN_GAP_MIN: int = 0
    # 0 => no additional start-gap filtering; clustering/overlap rules decide.
    FIGHT_MIN_GAP_MS: int = 0
    DETECT_STEP_MS: int = 10000

    # ─── teamfight_v2 parameters ────────────────────────────
    # Kill clustering: max temporal gap between consecutive kills
    # in the same fight cluster (15–20s recommended).
    TF2_KILL_CLUSTER_GAP_MS: int = 18000
    # Fight start = first_kill_ts - this offset (engage time).
    TF2_ENGAGE_PRE_KILL_MS: int = 10000
    # Small radius: at engage time, require >=2 per team within
    # this radius of fight center (first kill XY) for teamfight validity.
    TF2_VALIDITY_RADIUS: float = 1800.0
    # Large radius: events within this radius of fight center are
    # counted as fight interactions during the fight time window.
    TF2_INTERACTION_RADIUS: float = 3000.0
    # Post-fight outcome window (ms after last kill in cluster).
    TF2_POST_FIGHT_WINDOW_MS: int = 30000
    # Optional tail buffer after last kill (ms).
    TF2_TAIL_BUFFER_MS: int = 0
    # Minimum champions per team within validity radius.
    TF2_MIN_PER_TEAM: int = 2
    # Dense XY grid step used by teamfight_v2 detector.
    # Default 5s preserves current behavior.
    TF2_GRID_STEP_MS: int = 5000
    # Build dense grid with linear interpolation between 60s frames.
    TF2_USE_FRAME_INTERP: bool = True
    # Override actor trajectories from prior 60s frame toward kill position.
    # This is the key interpolation algorithm under ablation.
    TF2_USE_KILL_TRAJECTORY_INTERP: bool = True

    # ── [P0-1 FIX] Dual-path XY coordinate handling ──────────
    # Original: ZERO_XY=True zeroed XY everywhere, destroying
    # GNN adjacency (→ uniform) and 25 spatial features (→ constant).
    #
    # New design:
    #   Path 1: GNN adjacency + spatial features use raw (relative) XY
    #   Path 2: extra_seq (BiGRU input) removes only direct XY coords
    #
    # ZERO_XY_NODE_FEATURES: False → preserve XY in node_seq for GNN/spatial
    # ZERO_XY_IN_EXTRA_SEQ: True → remove raw XY from extra_seq (BiGRU)
    # USE_RELATIVE_XY: True → centroid-relative coordinates (map bias removal)
    # ADJ_SIGMA_FACTOR: 1.5 → enlarged σ absorbs 60s positional noise
    ZERO_XY_NODE_FEATURES: bool = False
    ZERO_XY_IN_EXTRA_SEQ: bool = True
    USE_RELATIVE_XY: bool = True
    ADJ_SIGMA_FACTOR: float = 1.5

    CONTINUOUS_FIGHT_MERGE: bool = True
    CONTINUOUS_FIGHT_MAX_GAP_MS: int = 15000
    CONTINUOUS_FIGHT_MERGE_RADIUS: float = 2000.0
    # If True, ward signal is valid only when ward actor(s) are inside fight radius.
    REQUIRE_WARD_ACTOR_IN_FIGHT_RADIUS: bool = True
    WARD_ACTOR_RADIUS: float = 1800.0

    STANDOFF_RADIUS: float = 1800.0
    REQUIRE_ALIVE_PER_TEAM: int = 2

    ENGAGE_MIN_DIST_DROP: float = 250.0
    ENGAGE_MIN_PAIR_GAIN: int = 2

    # Fight validity rule:
    #   a detected engage candidate is accepted only when
    #   at least one kill exists in [engage_ts, engage_ts + horizon).
    VERIFY_KILL_IN_HORIZON: bool = True
    # Additional combat-signal validation in horizon:
    #   damage proxy uses normalized Δ totalDamageDoneToChampions (team-sum),
    #   spell proxy counts SUMMONER_SPELL_USED/CAST events.
    # Rule options:
    #   - kill_only
    #   - signal_only
    #   - kill_or_signal
    #   - kill_and_signal
    FIGHT_VALIDATION_RULE: str = "kill_or_signal"
    MIN_DAMAGE_NORM_IN_HORIZON: float = 0.02
    MIN_SUMMONER_SPELLS_IN_HORIZON: int = 1

    PROX_DIST_NORM: float = 1800.0 / MAP_MAX
    PROX_MIN_PAIRS: int = 8
    STANDOFF_NO_KILL_PREV_MIN: bool = True
    # ═══════════════════════════════════════════════════════════
    # [P3-PAPER-1] SELECTION BIAS FIX
    # ─────────────────────────────────────────────────────────
    # Original: True  →  only keeps samples where at least one
    # signal event exists in the horizon window [t₀, t₀+Δ].
    # This creates Berkson's paradox:
    #   P(Y=1 | X, ∃signal ∈ horizon) ≠ P(Y=1 | X)
    # "Quiet" windows (no events) are systematically excluded,
    # causing model to over-rely on event density.
    # Fix: Default False.  Enable only for ablation study.
    # ═══════════════════════════════════════════════════════════
    REQUIRE_SIGNAL_IN_HORIZON: bool = False

    # Start-point must be engage-detected (not kill-anchored).
    USE_KILL_ANCHOR: bool = False
    KILL_ANCHOR_PRE_SEC: int = 15
    KILL_ANCHOR_COOLDOWN_SEC: int = 30

    USE_BACKTRACK: bool = True
    BACKTRACK_MAX_MS: int = 15000
    BACKTRACK_MIN_MS: int = 5000
    BACKTRACK_MIN_PAIRS: int = 3

    DUMP_FIGHTS: bool = True
    DUMP_FIGHTS_DIRNAME: str = "fight_dumps"
    DUMP_FIGHTS_MAX_MATCHES: int = 5000
    # Reuse build_fight_index() results across reruns with identical detector config.
    FIGHT_INDEX_CACHE_ENABLED: bool = True
    FIGHT_INDEX_CACHE_DIRNAME: str = "fight_index_cache"
    # build_fight_index parallelism (CPU multiprocessing).
    # 0 => auto (min(cpu_count, FIGHT_INDEX_MAX_AUTO_WORKERS)).
    FIGHT_INDEX_NUM_WORKERS: int = 0
    FIGHT_INDEX_MAX_AUTO_WORKERS: int = 8
    FIGHT_INDEX_MP_CHUNK_SIZE: int = 8

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
    LABEL_TYPE: str = "attention_value_win"
    LABEL_W_KILL: float = 1.0
    LABEL_W_ALIVE: float = 0.3

    # [P1-LABEL] Tie handling strategy.
    #   "exclude" — drop ties from training (recommended).
    #   "blue"    — ties → blue win (original behaviour, biased).
    #   "red"     — ties → red win.
    #   "random"  — ties → seeded deterministic coin flip per label window.
    LABEL_TIE_STRATEGY: str = "random"
    LABEL_TIE_SEED: int = 7

    # weighted label
    W_KILL: float = 1.0
    W_GOLD: float = 0.5
    W_OBJ: float = 0.25
    GOLD_NORM: float = 500.0
    LABEL_GOLD_METHOD: str = "linear"
    # rule-based attention-like label:
    # alpha_e = softmax(beta * prior_e), score = sum(alpha_e * sign_e * value_e)
    LABEL_ATTN_BETA: float = 2.0
    LABEL_ATTN_W_KILL: float = 1.0
    LABEL_ATTN_W_SHUTDOWN: float = 1.6
    LABEL_ATTN_W_STREAK: float = 0.35
    LABEL_ATTN_W_ASSIST: float = 0.20
    LABEL_ATTN_W_BOUNTY: float = 0.30
    LABEL_ATTN_W_OBJECTIVE: float = 1.10
    LABEL_ATTN_W_LANE: float = 0.25

    # graph (interaction)
    INTERACT_SIGMA_NORM: float = 0.12
    INTERACT_ALPHA_CLOSING: float = 4.0
    INTERACT_TOPK_ENEMY: int = 3
    INTERACT_TOPK_ALLY: int = 1
    INTERACT_EDGE_DIM: int = 64
    INTERACT_LAYERS: int = 2
    INTERACT_POOL_TOPK: int = 8

    USE_EVENT_TOKENS: bool = True
    MAX_EVENT_TOKENS: int = 64
    EVENT_TYPE_VOCAB: int = 128
    # event_cont layout (default 12 dims):
    # [t_rel, dt_end, x_norm, y_norm, val_log,
    #  is_shutdown, shutdown_norm, streak_norm, assist_norm,
    #  objective_tier, lane_priority, importance_prior]
    EVENT_CONT_DIM: int = 12
    EVENT_CONT_IMPORTANCE_PRIOR_IDX: int = 11

    XATTN_D_MODEL: int = 128
    XATTN_NHEAD: int = 4
    XATTN_GNN_KIND: str = "mpnn"
    XATTN_MULTISCALE_ADJ: bool = False
    XATTN_IMPORTANCE_POOL: bool = True
    XATTN_IMPORTANCE_HIDDEN: int = 64
    XATTN_PRIOR_BOOST: float = 1.25
    # Layered fusion (global-BiGRU + GNN + event-attention)
    LAYER_FUSION_GNN_KIND: str = "graphsage"
    LAYER_FUSION_GNN_MULTISCALE_ADJ: bool = False
    LAYER_FUSION_GLOBAL_KIND: str = "bigru"
    LAYER_FUSION_EVENT_KIND: str = "attn"
    # 0 => auto (F_GLOBAL + phase dims)
    LAYER_FUSION_GLOBAL_DIM: int = 0
    LAYER_FUSION_EVENT_D_MODEL: int = 128
    LAYER_FUSION_EVENT_NHEAD: int = 4
    LAYER_FUSION_EVENT_IMP_HIDDEN: int = 64
    LAYER_FUSION_FUSE_DIM: int = 192
    LAYER_FUSION_GATE_H: int = 64

    # Mamba hyperparameters
    MAMBA_D_MODEL: int = 128
    MAMBA_LAYERS: int = 3
    MAMBA_D_STATE: int = 16
    MAMBA_D_CONV: int = 4
    MAMBA_EXPAND: int = 2
    STMAMBA_GNN_KIND: str = "graphsage"
    STMAMBA_MULTISCALE_ADJ: bool = False
    # =========================================================
    # 10) Items (optional)
    # =========================================================
    USE_ITEMS: bool = True
    ITEM_HASH_DIM: int = 32

    # Per-player item hash (node-level item features for GNN)
    USE_PER_PLAYER_ITEMS: bool = True
    ITEM_HASH_DIM_PER_PLAYER: int = 16
    ITEM_PLAYER_PROJ_HIDDEN: int = 32

    # =========================================================
    # 11) Models / Ablation
    # =========================================================
    # [FIX] Removed `from dataclasses import field` and
    #       `from typing import Dict, Tuple` that were inside the
    #       class body — they shadowed module-level imports.
    BASELINE_MODELS: Tuple[str, ...] = ("lgbm",)

    RNN_MODELS: Tuple[str, ...] = (
        "rnn_bigru", "rnn_transformer", "rnn_tcn", "rnn_mamba",
        # [REC-1] Hybrid h₀-conditioned RNN variants
        "hybrid_bigru", "hybrid_bilstm",
    )
    GNN_MODELS: Tuple[str, ...] = (
        "gnn_graphsage", "gnn_stgnn", "gnn_gatv2",
        "stgnn_edge_mpnn", "stgcn", "stgnn_mamba",
        "ms_dyngraph", "event_xattn",
        "fusion_gated_gnn_bigru", "fusion_layered_gnn_bigru_xattn",
    )

    MODEL_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "lightgbm": "lgbm", "light_gbm": "lgbm", "lgb": "lgbm", "lgbm": "lgbm",
        "rnn_ugru": "rnn_ugru", "rnn_ulstm": "rnn_ulstm",
        "rnn_bigru": "rnn_bigru", "rnn_bilstm": "rnn_bilstm",
        "rnn_transformer": "rnn_transformer", "rnn_tcn": "rnn_tcn",
        "gru": "rnn_ugru", "ngru": "rnn_ugru", "ugru": "rnn_ugru",
        "lstm": "rnn_ulstm", "nlstm": "rnn_ulstm", "ulstm": "rnn_ulstm",
        "bigru": "rnn_bigru", "bi_gru": "rnn_bigru", "bi-gru": "rnn_bigru",
        "bilstm": "rnn_bilstm", "bi_lstm": "rnn_bilstm", "bi-lstm": "rnn_bilstm",
        "transformer": "rnn_transformer", "rnn-transformer": "rnn_transformer",
        "tcn": "rnn_tcn", "rnn-tcn": "rnn_tcn",
        "gnn_gcn": "gnn_gcn", "gnn_graphsage": "gnn_graphsage",
        "gnn_stgnn": "gnn_stgnn", "gnn_graphtransformer": "gnn_graphtransformer",
        "gnn_gatv2": "gnn_gatv2", "gnn_mpnn": "gnn_mpnn",
        "gnn": "gnn_gcn", "gcn": "gnn_gcn",
        "gnnsage": "gnn_graphsage", "gnn_sage": "gnn_graphsage",
        "graphsage": "gnn_graphsage", "sage": "gnn_graphsage",
        "stgnn": "gnn_stgnn", "st-gnn": "gnn_stgnn", "st_gnn": "gnn_stgnn",
        "gnn-transformer": "gnn_graphtransformer",
        "gnn_transformer": "gnn_graphtransformer",
        "graphtransformer": "gnn_graphtransformer",
        "graph-transformer": "gnn_graphtransformer",
        "gat": "gnn_gatv2", "gatv2": "gnn_gatv2",
        "mpnn": "gnn_mpnn",
        "edge_stgnn": "stgnn_edge_mpnn", "stgnn_edge": "stgnn_edge_mpnn",
        "stgnn_edge_mpnn": "stgnn_edge_mpnn",
        "st-gcn": "stgcn", "st_gcn": "stgcn", "stgcn": "stgcn",
        "multiscale": "ms_dyngraph", "multi_scale": "ms_dyngraph",
        "ms_dyngraph": "ms_dyngraph",
        "event_xattn": "event_xattn", "eventxattn": "event_xattn",
        "fusion_layered_gnn_bigru_xattn": "fusion_layered_gnn_bigru_xattn",
        "layered_fusion": "fusion_layered_gnn_bigru_xattn",
        "fusion_layered": "fusion_layered_gnn_bigru_xattn",
        "xattn": "event_xattn","mamba": "rnn_mamba","stmamba": "stgnn_mamba","st_mamba": "stgnn_mamba",
        # [REC-1] Hybrid h₀-conditioned model aliases
        "hybrid_bigru": "hybrid_bigru", "rnn_hybrid_bigru": "hybrid_bigru",
        "hybrid_bilstm": "hybrid_bilstm", "rnn_hybrid_bilstm": "hybrid_bilstm",
        "hybrid_ugru": "hybrid_ugru", "rnn_hybrid_ugru": "hybrid_ugru",
    })

    # =========================================================
    # 11.1) Ablation plan  [P2-ABLATION]
    # =========================================================
    ABLATION_PLAN: Tuple[str, ...] = (
        "baseline",
        "deep_only",
        "baseline_plus_rnn",
        "baseline_plus_gnn",
        "rnn_plus_gnn",
        "fusion_best",
        # [NEW] Feature-group ablation
        "ablate_no_spatial",
        "ablate_no_events",
        "ablate_no_status",
    )

    ABLATION_SELECT_SPLIT: str = "val"
    ABLATION_SELECT_METRIC: str = "auc"
    FUSION_SELECT_STRATEGY: str = "grid_best"

    ABLATION_GROUPS: Dict[str, Tuple[str, ...]] = field(default_factory=dict, init=False)
    MODEL_LIST: Tuple[str, ...] = field(default_factory=tuple)

    # =========================================================
    # 12) Training (deep)
    # =========================================================
    REQUIRE_CUDA: bool = True
    BATCH_SIZE: int = 64
    LR: float = 5e-4               # [FIX P1-3] 2e-4 → 5e-4 (용량 증대에 맞춘 상향)
    WEIGHT_DECAY: float = 2e-4
    EPOCHS: int = 15               # [FIX P0-3] 10 → 15 (수렴 여유 확보)
    PATIENCE: int = 3               # [FIX P1-4] 4 → 3 (불필요한 대기 제거)
    LOG_EVERY: int = 100
    GRAD_CLIP_NORM: float = 5.0
    DEEP_MAX_TRAIN: int = 100_000   # [FIX P1-1] deep model train 서브샘플링 한도
    LGBM_MAX_TRAIN: int = 100_000   # [FE-CONST] LGBM train subsampling (was 150K)
    # [P0-6] Val/Test subsampling inside deep.py (0 = disabled).
    # Superseded by GLOBAL_SUBSAMPLE_PER_SPLIT which caps all splits uniformly.
    VAL_MAX_N: int = 0
    TEST_MAX_N: int = 0
    # Global per-split subsample applied right after train/val/test split.
    # 0 = disabled (no global cap).  Affects ALL downstream consumers.
    GLOBAL_SUBSAMPLE_PER_SPLIT: int = 100_000
    # [P1-7] Warmup epochs — explicit config (was hardcoded as ceil(0.1 * EPOCHS))
    WARMUP_EPOCHS: int = 1

    # ---- Performance / DataLoader ----
    # Mixed precision (AMP) + TF32 can significantly speed up training on NVIDIA GPUs.
    AMP: bool = True
    # AMP dtype policy: auto | bfloat16 | float16
    AMP_DTYPE: str = "auto"
    TF32: bool = True
    CUDNN_BENCHMARK: bool = True
    TORCH_COMPILE: bool = False
    TORCH_COMPILE_MODE: str = "default"      # default | reduce-overhead | max-autotune
    TORCH_COMPILE_DYNAMIC: bool = False
    SPEED_PROFILE: str = "none"              # none | auto | rtx50 | rtx5080 | aggressive

    # DataLoader
    NUM_WORKERS: int = 4
    EVAL_NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 2

    # Dataset caching
    # - CACHE_MATCH_PACKS_IN_RAM: caches raw match packs (load_match_cache) via RAM LRU.
    # - CACHE_*_SAMPLES_IN_RAM: preloads samples (after build_ms_sequence) into RAM.
    CACHE_MATCH_PACKS_IN_RAM: bool = True
    CACHE_TRAIN_SAMPLES_IN_RAM: bool = False
    CACHE_EVAL_SAMPLES_IN_RAM: bool = False

    DROPOUT: float = 0.20           # [FIX P0-3] 0.35 → 0.20 (실효 용량 +23%)
    RNN_HIDDEN: int = 128           # [FIX P0-3] 64 → 128 (BiGRU params ×2.8)
    RNN_LAYERS: int = 2

    HEAD_HIDDEN: int = 128
    HEAD_LAYERS: int = 2

    GNN_DIM: int = 96               # [FIX P0-3] 64 → 96 (보수적 GNN 확대)
    GNN_DROPOUT: float = 0.25       # [FIX P0-3] 0.35 → 0.25 (GNN 정규화 완화)
    GNN_NORM: bool = True

    USE_ALIVE_MASK: bool = True
    GNN_FORCE_FP32: bool = True

    ADJ_SOFT: bool = True
    ADJ_SIGMA_NORM: float = 0.125
    TEAM_EDGE_WEIGHT: float = 1.0
    ADJ_CLAMP_MIN: float = 1e-4

    # [P2-SIGMA] Adaptive σ — if True, σ(t) = mean_pairwise_dist(t) * ADJ_SIGMA_RATIO
    # [IMPROVE] Enable adaptive sigma by default — sigma adapts to actual
    # mean pairwise distance per timestep, improving skirmish vs 5v5 accuracy.
    ADJ_SIGMA_ADAPTIVE: bool = True
    ADJ_SIGMA_RATIO: float = 0.5

    ADJ_CLAMP_NONNEG: bool = True
    ADJ_SYMMETRIZE: bool = False
    ADJ_NORM_ENSURE_SELFLOOP: bool = False
    ADJ_NORM_EPS: float = 1e-6
    ADJ_NORM_OUT_FP32: bool = True

    SAGE_DEG_EPS: float = 1e-6

    # Transformer temporal (L=6 stability: reduced from 256/3/4/0.2)
    TRANS_D_MODEL: int = 64
    TRANS_NHEAD: int = 4
    TRANS_LAYERS: int = 2
    TRANS_DROPOUT: float = 0.1
    TRANS_FF_MULT: int = 2
    TRANS_MAX_LEN: int = 512
    TRANS_LR: float = 1e-4
    TRANS_WARMUP_EPOCHS: int = 3
    TRANS_WARMUP_START_FACTOR: float = 0.1

    # TCN temporal
    TCN_CHANNELS: int = 64
    TCN_LEVELS: int = 3
    TCN_KERNEL: int = 3
    TCN_DROPOUT: float = 0.20       # [FIX P0-3] 0.35 → 0.20 (TCN 정규화 완화)

    # =========================================================
    # 12.1) [REC-4a] Recency Weighting for Patch Drift
    # ---------------------------------------------------------
    # w_i = exp((p_i - p_min) / τ)
    # τ → ∞ : uniform weights (no recency)
    # τ = 2.0 : moderate recency (patch 15.14→1.0, 15.17→4.48)
    # τ = 1.0 : aggressive recency
    # =========================================================
    RECENCY_WEIGHT_ENABLED: bool = True
    RECENCY_WEIGHT_TAU: float = 2.0

    # =========================================================
    # 12.2) [REC-1] Hybrid h₀ Conditioning
    # ---------------------------------------------------------
    # h₀ = MLP_tab(φ_tab) where φ_tab = seq_to_tabular(S)
    # Projects tabular summary into GRU/LSTM initial hidden state
    # =========================================================
    HYBRID_H0_ENABLED: bool = True
    HYBRID_H0_PROJ_DIM: int = 64
    HYBRID_H0_DROPOUT: float = 0.15

    # =========================================================
    # 12.3) [REC-4b] Temperature Scaling (post-hoc calibration)
    # ---------------------------------------------------------
    # P_calibrated = σ(z / T_p*)
    # T_p* = argmin_T Σ [-y_i log σ(z_i/T) - (1-y_i) log(1-σ(z_i/T))]
    # =========================================================
    TEMP_SCALING_ENABLED: bool = False

    # GATv2
    GAT_HEADS: int = 4
    GAT_LEAKY_ALPHA: float = 0.2
    GAT_HARD_MASK_TH: float = 0.0

    # MPNN
    MPNN_EDGE_DIM: int = 4
    MPNN_HIDDEN: int = 128
    MPNN_DEG_EPS: float = 1e-6

    # Fusion
    FUSION_GATE_H: int = 8
    FUSION_MLP_H: int = 32

    # Tab logit passthrough
    STRICT_TAB_LOGIT: bool = False

    # =========================================================
    # 13) Reporting
    # =========================================================
    CLS_THRESHOLD: float = 0.5
    EARLY_STOP_METRIC: str = "auc"
    PREC_AT_K: Tuple[int, ...] = (50, 100, 200, 500)
    PREC_AT_FRAC: Tuple[float, ...] = (0.01, 0.05, 0.10)
    ENABLE_MINUTEWISE_REPORT: bool = True
    ENABLE_SITUATION_REPORT: bool = True
    MINUTE_REPORT_MAX_MINUTE: int = 60
    SITUATION_CLOSE_GOLD_TH: float = 2000.0
    SITUATION_STOMP_GOLD_TH: float = 5000.0

    LGB_PERM_IMPORTANCE: bool = False
    LGB_SHAP: bool = True
    DEEP_PERM_IMPORTANCE: bool = False

    # =========================================================
    # 14) Multicollinearity / redundant feature removal
    # =========================================================
    # [FE-CONST] Constant/quasi-constant/within-fight-constant temporal
    # aggregation pruning.  Removes redundant __mean/__std/__min/__max/
    # __delta/__slope for features that are constant within a fight.
    # Only __last is retained for these features.
    DROP_CONSTANT_FEATURES: bool = True
    DROP_QUASI_CONSTANT_FEATURES: bool = True
    # Dragon soul (sparse binary): constant within fight, informative across fights.
    DROP_WITHIN_FIGHT_CONSTANT_FEATURES: bool = True

    DROP_CORR_FEATURES: bool = True
    CORR_THRESHOLD: float = 0.98
    # [P2] Added hierarchical-clustering alternative
    CORR_METHOD: str = "hierarchical"  # "greedy" (original) | "hierarchical"
    DROP_VIF_FEATURES: bool = False
    VIF_THRESHOLD: float = 12.0

    # =========================================================
    # 15) Debug / Diagnostics
    # =========================================================
    DEBUG_GNN: bool = False
    DUMP_FIGHTS_PRINT_SUMMARY: bool = False
    DIAG_MAX_FIGHT_RESULTS: int = 50
    # Large but bounded list for validated fight outputs in diagnostics.
    DIAG_MAX_VALIDATED_FIGHT_RESULTS: int = 10000

    # =========================================================
    # 16) Baseline Params (LGBM / XGB)
    # =========================================================
    BASELINE_LGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=5000, learning_rate=0.03, max_depth=6, num_leaves=31,
        min_data_in_leaf=200, min_gain_to_split=0.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, max_bin=255, n_jobs=-1,
    ))

    BASELINE_XGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=3000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=50,
        gamma=0.0, tree_method="hist", n_jobs=-1, eval_metric="logloss",
    ))

    # =========================================================
    # 17) RAM caching (legacy)
    # CACHE_IN_RAM is kept for backward compatibility; prefer CACHE_MATCH_PACKS_IN_RAM / CACHE_*_SAMPLES_IN_RAM.
    # =========================================================
    CACHE_IN_RAM: bool = False
    CACHE_RAM_MAX_MATCHES: int = 256

    # -------------------------
    # derived config build
    # -------------------------
    def __post_init__(self) -> None:
        baseline = tuple(self.BASELINE_MODELS)
        rnn = tuple(self.RNN_MODELS)
        gnn = tuple(self.GNN_MODELS)

        self.ABLATION_GROUPS = {
            "baseline": baseline,
            "deep_only": rnn + gnn,
            "baseline_plus_rnn": baseline + rnn,
            "baseline_plus_gnn": baseline + gnn,
            "rnn_plus_gnn": rnn + gnn,
            "fusion_best": ("fusion_auto_best",),
            # [NEW] Feature-group ablation groups
            "ablate_no_spatial": baseline + rnn,
            "ablate_no_events": baseline + rnn,
            "ablate_no_status": baseline + rnn,
        }

        self.MODEL_LIST = baseline + rnn + gnn + ("fusion_auto_best",)

    # config.py — CFG 클래스 내부 끝부분에 추가

    # === Ablation Treatment Flags ===
    USE_FOCAL_LOSS: bool = False
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.25

    USE_GAME_PHASE: bool = False
    GAME_PHASE_TAU: float = 3.0

    USE_ATTENTION_POOL: bool = False
    ATTENTION_POOL_DIM: int = 64

    USE_MOMENTUM_FEATURES: bool = False
    MOMENTUM_K_SHORT: int = 3

    USE_ROLE_AWARE_ADJ: bool = False
    ROLE_ADJ_INIT: float = 0.0

    USE_MULTI_TASK: bool = False
    MTL_LAMBDA_GOLD: float = 0.1
    MTL_LAMBDA_KILL: float = 0.05
    MTL_LAMBDA_OBJ: float = 0.05

    LABEL_SMOOTHING: float = 0.0

    # ── Temporal sequence selection priority ──────────────────
    # Controls which sequence key pick_temporal_seq() prefers.
    # Default: x_seq before extra_seq so BiGRU sees full raw features.
    TEMPORAL_SEQ_PRIORITY: Tuple[str, ...] = ("x_seq", "extra_seq")

    # ── [P0-3] Input Projection for RNN full-info access ──────
    # When True, RNNOnlyModel can use x_seq (~997-dim) via a projection
    # layer: Linear(997, INPUT_PROJ_DIM) → LayerNorm → ReLU → BiGRU
    # This gives BiGRU access to all 10 players' individual features.
    USE_INPUT_PROJECTION: bool = True
    INPUT_PROJ_DIM: int = 256

    # ── [P2-2] Categorical Embedding Specs for NodeFeatureAdapter ──
    # Maps categorical node features to learned embeddings instead of
    # feeding raw integer IDs into Linear layers.
    NODE_CAT_SPECS: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "champion_id": {"num_embeddings": 250, "emb_dim": 16},
        "champion_name_id": {"num_embeddings": 4096, "emb_dim": 16},
        "primary_style_id": {"num_embeddings": 256, "emb_dim": 8},
        "sub_style_id": {"num_embeddings": 256, "emb_dim": 8},
        "summoner_spell_1_id": {"num_embeddings": 512, "emb_dim": 4},
        "summoner_spell_2_id": {"num_embeddings": 512, "emb_dim": 4},
    })
# -------------------------------------------------------------------
# Singleton instance + directory creation
# -------------------------------------------------------------------
cfg = CFG()

# Directory creation is best-effort: on a machine where OUTPUT_ROOT lives on a
# missing/read-only drive, mkdir() would raise at import time and make the whole
# package unimportable. Wrap each mkdir in try/except so importing config NEVER
# raises; the dirs are still created whenever the filesystem allows it.
try:
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
CACHE_DIR = cfg.OUTPUT_ROOT / "cache" / cfg.CACHE_DIRNAME
RUN_DIR = cfg.OUTPUT_ROOT / cfg.RUN_DIRNAME
META_DIR = cfg.OUTPUT_ROOT / cfg.META_DIRNAME
DATASET_DIR = cfg.OUTPUT_ROOT / "dataset" / cfg.DATASET_DIRNAME

for d in (CACHE_DIR, RUN_DIR, META_DIR, DATASET_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# -------------------------------------------------------------------
# Export baseline params as module-level dicts
# -------------------------------------------------------------------
BASELINE_LGB_PARAMS: Dict[str, Any] = dict(cfg.BASELINE_LGB_PARAMS)

# -------------------------------------------------------------------
# Contracts (single source of truth)
# -------------------------------------------------------------------
from core.feature_contract import build_feature_contract

FEATURE_CONTRACT = build_feature_contract(
    node_names=NODE_FEATURE_NAMES,
    event_names=EVENT_FEATURE_NAMES,
    global_names=GLOBAL_FEATURE_NAMES,
)

# -------------------------------------------------------------------
# [P2-STRUCT-1] SSoT: Direct references to FEATURE_CONTRACT indices.
#
# Previously: NODE_IDX = dict(FEATURE_CONTRACT.node_idx)  ← independent copy
# Problem:    3 modules created independent dict() copies → silent drift
#             if any copy was mutated at runtime.
# Fix:        Reference FEATURE_CONTRACT.node_idx directly.
#             FEATURE_CONTRACT is frozen dataclass → immutable by contract.
#             All modules import NODE_IDX from config (single import path).
# -------------------------------------------------------------------
NODE_IDX: Dict[str, int] = FEATURE_CONTRACT.node_idx       # direct ref, NOT dict()
EVENT_IDX: Dict[str, int] = FEATURE_CONTRACT.event_idx     # direct ref, NOT dict()
GLOBAL_IDX: Dict[str, int] = FEATURE_CONTRACT.global_idx   # direct ref, NOT dict()

# Dims (backward compat)
F_NODE = int(FEATURE_CONTRACT.f_node)
F_EVENT = int(FEATURE_CONTRACT.f_event)
F_GLOBAL = int(FEATURE_CONTRACT.f_global)

# Item hash
ITEM_HASH_DIM: int = int(cfg.ITEM_HASH_DIM)
ITEM_HASH_NAMES: List[str] = [f"itemhash{i}" for i in range(ITEM_HASH_DIM)]
