"""Paths and fixed policy for the cache source audit."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
CACHE_DIR = Path(r"D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13")
INDEX_PATH = CACHE_DIR.parent / "match_cache_fresh_v3_engage_status13_patch_index.json"
OUTPUT_ROOT = ROOT / "outputs" / "cache_source_audit_20261002"
POST_STATE_CONTRACT_PATH = ROOT / "reports" / "post_state_label_contract_20261002" / "CONTRACT.json"
FIXED_COHORT_MANIFEST_PATH = ROOT / "outputs" / "full_corpus_preflight_20260915" / "main_matches.csv"
AUDIT_SCOPE_PATH = ROOT / "reports" / "cache_source_audit_20261002" / "AUDIT_SCOPE.json"

PATCH_ROLES = {"15.14": "TRAIN", "15.15": "VALIDATION", "15.16": "TEST"}
EXPECTED_PATCH_COUNTS = {"15.14": 74_673, "15.15": 74_748, "15.16": 60_579}
EXPECTED_INDEX_COUNT = sum(EXPECTED_PATCH_COUNTS.values())
EXPECTED_INDEX_SHA256 = "41db6c8993e85ebe65dfb61f082906cfd8d5cc8229ba66c75bff3fba6e8c7017"
EXPECTED_COHORT_MANIFEST_SHA256 = "34d95c39ff836a89a350867c11faa6606d168084c87953317be25e41ec38fd6b"
DATADRAGON_PATHS = {
    "15.14": ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw" / "championFull_15.14.1.json",
    "15.15": ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw" / "championFull_15.15.1.json",
    "15.16": ROOT / "outputs" / "reest_exact_v4_20260925" / "datadragon_raw" / "championFull_15.16.1.json",
}
DATADRAGON_EXPECTED_SHA256 = {
    "15.14": "c039fb19565328c8660a7921a1fe3bcba724bc70e465354aa9413fdf8bdd5a8b",
    "15.15": "43cb5c11be017fa41203c4f41ccfe5044f779f6549f1f0f97f859e5fe2cdec5a",
    "15.16": "5c51a5ce1c390a31509ecbd0148b103fbe091c9fe9649c86e8e59b1b9b2e3b93",
}

EXPECTED_NPZ_KEYS = (
    "minute_ts",
    "node_minute",
    "global_minute",
    "gold_team_minute",
    "xy_raw_minute",
)
EXPECTED_NPZ_DTYPES = {
    "minute_ts": "int64",
    "node_minute": "float32",
    "global_minute": "float32",
    "gold_team_minute": "float32",
    "xy_raw_minute": "float32",
}
EXPECTED_CHANNEL_WIDTHS = {
    "minute_ts": 1,
    "node_minute": 76,
    "global_minute": 26,
    "gold_team_minute": 2,
    "xy_raw_minute": 2,
}
MAX_WORKERS = 3
BATCH_SIZE = 1_000


def new_run_dir() -> Path:
    """Return a fresh output path without creating or replacing anything."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return OUTPUT_ROOT / f"run_{stamp}"
