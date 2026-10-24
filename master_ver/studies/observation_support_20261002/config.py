"""Paths and frozen constants for observation_support_20261002."""
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
EXPECTED_COMPONENTS_SHA256 = "5bec73337e8d3fc153f9c21a220576cc5bc92b44ee489b2cea05f5674c38254e"
EXPECTED_TERMINALS_SHA256 = "9c5dde7fe5e624524cb0c5a68f57e2484a98f487bba57dadef1835eefb46d620"

ST_OUT = ROOT / "outputs" / "anchor10_boundary_study_20261001" / "st_dbscan_20261001T131304Z"
ST_SOURCE = ST_OUT / "source"
ST_BATCHES = ST_OUT / "run" / "batches"
ST_AGG = ST_OUT / "aggregate"
CACHE_DIR = Path(r"D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")

RAW_SOURCE_ROOTS = (
    Path(r"C:/Users/todtj/PycharmProjects/Lol_project/data/raw/matches/kr"),
    Path(r"D:/LOL_Project/fusion_2615/raw/kr"),
)

CHAMPION_FULL = (
    ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw" / "championFull_15.14.1.json"
)
EXPECTED_CHAMPION_FULL_SHA256 = "c039fb19565328c8660a7921a1fe3bcba724bc70e465354aa9413fdf8bdd5a8b"
EXPECTED_CHAMPION_FULL_BYTES = 1_532_196

PROTOCOL = ROOT / "reports" / "observation_support_20261002" / "PROTOCOL.md"
EXPECTED_PROTOCOL_SHA256 = "3d513d17e22518c95a3a114e4c9a5a8b88f12d6ac50ce58ea1a3b1e37fe61de0"
RADIUS_EVIDENCE = ROOT / "reports" / "observation_support_20261002" / "RADIUS_SOURCE_EVIDENCE.json"

ADOPTED_CFG = "t14000_s2000_m1"
ANCHOR_OFFSET_MS = 10_000
DIAGNOSTIC_RADII = (1800.0, 2000.0)
HORIZONS_S = (60, 90, 120)

EXPECTED_TRAIN_MATCHES = 74_673
EXPECTED_TRAIN_KILLS = 3_752_984
EXPECTED_MIN1_COMPONENTS = 2_299_223
EXPECTED_ZERO_KILL = 571
EXPECTED_INCLUSIVE_10_20 = 378_703
EXPECTED_STRICT_ELIGIBLE = 378_674
EXPECTED_EXACT_10000 = 29
EXPECTED_KILLS_SHA256 = "ad9c6377c404c04a0743f85f36894fb3c835a2ec216617ee3d248b0ce096d7a6"
EXPECTED_MATCH_IDS_SHA256 = "96cd7979537bda8abad74dfdc42163e3fc00fd426dad842794933caa282da7ff"
EXPECTED_MATCH_META_SHA256 = "734c3efd319b7c1bdc56c3b0893d10fc9ab9a365661da59f15025d621ff4d5c4"
EXPECTED_LEDGER_SHA256 = "2ad2d92c81c6ea2f36fed972321ac1ab5bb9c4d17a9716fc02e91e3567023825"
EXPECTED_RANGES_SHA256 = "930000be1285457359d268b2d3ccb3f6dbf1093e27a20b186ed1f5b06c35ac57"

# Verified against worktrees/.../core.config.NODE_FEATURE_NAMES (76 cols).
NODE_FEATURE_NAMES_REF = (
    "champion_id",
    "champion_name_id",
    "summoner_spell_1_id",
    "summoner_spell_2_id",
    "x_norm",
    "y_norm",
    "level_norm",
    "xp_norm",
    "curGold_norm",
    "totalGold_norm",
    "gps_norm",
    "laneCS_norm",
    "jgCS_norm",
    "ccTime_norm",
    "hp_pct",
    "mp_pct",
    "alive",
)
IDX_LEVEL_NORM = 6
IDX_XP_NORM = 7
IDX_HP_PCT = 14
IDX_ALIVE = 16

CPU_BUDGET = 4
MAX_WORKER_PROCESSES = 3
BATCH_SIZE_MATCHES = 128
SEED = 20261002

SENTINEL_MISSING = -1
SENTINEL_INF = -2

# Cache observation codes (not raw presence).
POS_KNOWN_NONZERO = 0
POS_AMBIGUOUS_ZERO = 1
POS_UNKNOWN = 2
POS_NONFINITE = 3

HP_CACHED_POSITIVE = 0
HP_AMBIGUOUS_ZERO = 1
HP_UNKNOWN = 2
HP_NONFINITE = 3
HP_NEGATIVE = 4

# Raw tri-state codes (only when original participantFrames verified).
RAW_HP_POSITIVE = 0
RAW_HP_OBSERVED_ZERO = 1
RAW_HP_UNKNOWN = 2
RAW_POS_KNOWN = 0
RAW_POS_UNKNOWN = 1

TEAM_BLUE = 100
TEAM_RED = 200


def new_run_dir(tag: str = "") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"run_{stamp}{('_' + tag) if tag else ''}"
    return ROOT / "outputs" / "observation_support_20261002" / name


def report_dir() -> Path:
    return ROOT / "reports" / "observation_support_20261002"
