"""Paths and frozen constants for objective_timing_20261002."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent

ANCHOR_RUN = ROOT / "outputs" / "anchor_frame_audit_20261002" / "run_20261002T164300Z_full"
ANCHOR_MANIFEST = ANCHOR_RUN / "RUN_MANIFEST.json"
EXPECTED_ANCHOR_MANIFEST_SHA256 = "a20aa856213ba759a8574a391eea3d080d5301cbe708f7b3af80c7cac6cca765"
COMPONENTS_NPZ = ANCHOR_RUN / "components.npz"
TERMINALS_JSON = ANCHOR_RUN / "terminals.json"

ST_OUT = ROOT / "outputs" / "anchor10_boundary_study_20261001" / "st_dbscan_20261001T131304Z"
ST_SOURCE = ST_OUT / "source"
CACHE_DIR = Path(r"D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")

AUDIT_STUDY = ROOT / "studies" / "anchor_frame_audit_20261002"

EXPECTED_TRAIN_MATCHES = 74_673
EXPECTED_TRAIN_KILLS = 3_752_984
EXPECTED_MIN1_COMPONENTS = 2_299_223
EXPECTED_ZERO_KILL = 571
EXPECTED_FULL_WINDOW = 378_703  # inclusive delta in [10000, 20000]
EXPECTED_FULL_ISOL = 271_767
EXPECTED_KILLS_SHA256 = "ad9c6377c404c04a0743f85f36894fb3c835a2ec216617ee3d248b0ce096d7a6"
# Frozen baseline file identities (also in objective_timing_review_20261002/baseline_identity.json)
EXPECTED_COMPONENTS_SHA256 = "5bec73337e8d3fc153f9c21a220576cc5bc92b44ee489b2cea05f5674c38254e"
EXPECTED_TERMINALS_SHA256 = "9c5dde7fe5e624524cb0c5a68f57e2484a98f487bba57dadef1835eefb46d620"
EXPECTED_MATCH_IDS_SHA256 = "96cd7979537bda8abad74dfdc42163e3fc00fd426dad842794933caa282da7ff"
EXPECTED_MATCH_META_SHA256 = "734c3efd319b7c1bdc56c3b0893d10fc9ab9a365661da59f15025d621ff4d5c4"

CPU_BUDGET = 4
MAX_WORKER_PROCESSES = 3
BATCH_SIZE_MATCHES = 256
SEED = 20261002

SENTINEL_MISSING = -1
SENTINEL_INF = -2

# Query status codes
ST_OBSERVED = 0
ST_NO_ACQ = 1
ST_UNKNOWN_OBS = 2
ST_NO_INTERVAL = 3

# Acquisition evidence codes (event-level)
ACQ_CONFIRMED = 0
ACQ_NEUTRAL = 1
ACQ_UNKNOWN = 2
ACQ_CONTRADICTORY = 3

# Interval kinds
INTERVAL_UNCAPED = "uncapped_to_game_end"
INTERVAL_PRE_NEXT = "pre_next_fight"

# Known elemental dragon subtypes (raw monsterSubType strings).
KNOWN_ELEMENTAL_DRAGON_SUBTYPES = frozenset(
    {
        "FIRE_DRAGON",
        "EARTH_DRAGON",
        "WATER_DRAGON",
        "AIR_DRAGON",
        "HEXTECH_DRAGON",
        "CHEMTECH_DRAGON",
    }
)

# Fixed reporting categories (individual types may be zero/absent).
CATEGORY_ORDER = (
    "any_objective",
    "any_epic_monster",
    "any_building",
    "any_nonterminal_building",
    "dragon_elemental",
    "dragon_elder",
    "dragon_unknown_subtype",
    "baron",
    "herald",
    "horde",
    "atakhan",
    "monster_other",
    "tower",
    "inhibitor",
    "building_other",
    "building_terminal",
)

CATEGORY_TO_CODE = {name: i for i, name in enumerate(CATEGORY_ORDER)}
CODE_TO_CATEGORY = {i: name for name, i in CATEGORY_TO_CODE.items()}

# Delay coverage thresholds (seconds)
DELAY_THRESHOLDS_S = (30, 60, 90, 120, 180)

# Diagnostic cohorts
COHORT_ALL = "all_components"
COHORT_FULL_WINDOW = "delta_10000_20000_inclusive"
COHORT_FULL_ISOL = "delta_10000_20000_global_isolation"
COHORT_10_15 = "delta_10000_15000_inclusive"
COHORT_STRICT = "historical_strict_eligible"

TEAM_UNKNOWN = 0
TEAM_BLUE = 100
TEAM_RED = 200
TEAM_NEUTRAL = 300


def new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "objective_timing_20261002" / f"run_{stamp}"


def report_dir() -> Path:
    return ROOT / "reports" / "feature_control_study_20261001" / "objective_timing_20261002"
