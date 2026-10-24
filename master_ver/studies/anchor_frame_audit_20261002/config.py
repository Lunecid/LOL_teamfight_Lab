"""Paths and frozen diagnostic constants for anchor_frame_audit_20261002."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
ST_OUT = ROOT / "outputs" / "anchor10_boundary_study_20261001" / "st_dbscan_20261001T131304Z"
ST_SOURCE = ST_OUT / "source"
ST_BATCHES = ST_OUT / "run" / "batches"
ST_AGG = ST_OUT / "aggregate"
CACHE_DIR = Path(r"D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
DR_RULES = ROOT / "scripts" / "engagement_labels_v3_rules.py"
WT = ROOT / "worktrees" / "engagement-state-value"
EV4 = WT / "scripts" / "exact_v4"
PROTOCOL = ROOT / "reports" / "feature_control_study_20261001" / "FULL_STUDY_PROTOCOL.md"
PT_FLEX_BUNDLE = (
    ROOT / "outputs" / "reest_exact_v4_20260925" / "stage2" / "fit_q" / "frozen" / "T" / "pt_flex" / "bundle.json"
)
Q_BUNDLE = ROOT / "outputs" / "reest_exact_v4_20260925" / "stage2" / "fit_q" / "frozen" / "T" / "q" / "bundle.json"
OOF_V = ROOT / "outputs" / "reest_exact_v4_20260925" / "stage2" / "fit_v2"  # probed at runtime

ADOPTED_CFG = "t14000_s2000_m1"
EPS_T_MS = 14_000
EPS_S = 2000.0
MIN_SAMPLES = 1
ANCHOR_OFFSET_MS = 10_000
NEIGHBOR_RADIUS = 2000.0  # diagnostic distance; neighbor radius, not cluster cap
FOREIGN_NEAR_RADIUS = 2000.0

EXPECTED_TRAIN_MATCHES = 74_673
EXPECTED_TRAIN_KILLS = 3_752_984
EXPECTED_MIN1_COMPONENTS = 2_299_223
EXPECTED_ZERO_KILL = 571
EXPECTED_KILLS_SHA256 = "ad9c6377c404c04a0743f85f36894fb3c835a2ec216617ee3d248b0ce096d7a6"
EXPECTED_SPAN_MAX = 61_588
EXPECTED_DIAMETER_MAX = 7466.228097238926

# Inclusive lag diagnosis: δ = f - F_inclusive; exact when δ == 10000
INCLUSIVE_WIDTHS_MS = tuple(range(0, 10_001, 1_000))  # w=0..10000
UPPER_GAP_MS = (25_000, 30_000)  # distinguish 'frame age 20s at 10s anchor'
# Disjoint δ bands [lo, hi): endpoints explicit
DISJOINT_DELTA_BANDS = (
    (10_000, 10_001),  # exact
    (10_001, 11_000),
    (11_000, 12_000),
    (12_000, 13_000),
    (13_000, 14_000),
    (14_000, 15_000),
    (15_000, 16_000),
    (16_000, 17_000),
    (17_000, 18_000),
    (18_000, 19_000),
    (19_000, 20_000),
    (20_000, 20_001),  # inclusive 20s boundary point
    (20_001, 25_000),
    (25_000, 30_000),
    (30_000, 10**12),
)

HORIZONS_S = (60, 90, 120)
CPU_BUDGET = 4
MAX_WORKER_PROCESSES = 3
BATCH_SIZE_MATCHES = 512
SEED = 20261002

SENTINEL_MISSING = -1
SENTINEL_INF = -2


def new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "anchor_frame_audit_20261002" / f"run_{stamp}"


def report_dir() -> Path:
    return ROOT / "reports" / "feature_control_study_20261001" / "anchor_frame_audit_20261002"
