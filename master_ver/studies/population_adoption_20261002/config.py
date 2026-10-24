"""Paths and immutable policy constants for population_adoption_20261002."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent

CONTRACT_PATH = ROOT / "reports" / "population_adoption_20261002" / "CONTRACT.json"
DESIGN_PATH = ROOT / "reports" / "population_adoption_20261002" / "ADOPTED_DESIGN_KO.md"
DIAGNOSTIC_ACCEPTANCE_PATH = ROOT / "reports" / "observation_support_20261002" / "FULL_DIAGNOSTIC_ACCEPTANCE.json"
POST_STATE_CONTRACT_PATH = ROOT / "reports" / "post_state_label_contract_20261002" / "CONTRACT.json"
RADIUS_SOURCE_EVIDENCE_PATH = ROOT / "reports" / "observation_support_20261002" / "RADIUS_SOURCE_EVIDENCE.json"
INPUT_POLICY_PATH = ROOT / "reports" / "input_population_specification_20261002" / "INPUT_POLICY.json"
POPULATION_POLICY_PATH = ROOT / "reports" / "input_population_specification_20261002" / "POPULATION_POLICY.json"
DEFAULT_SOURCE_RUN = ROOT / "outputs" / "observation_support_20261002" / "run_20261002T000935Z_full_direct_full_r3"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "population_adoption_20261002"
SOURCE_LEDGER_DIR = ROOT / "outputs" / "anchor10_boundary_study_20261001" / "st_dbscan_20261001T131304Z" / "source"
SOURCE_BATCH_DIR = ROOT / "outputs" / "anchor10_boundary_study_20261001" / "st_dbscan_20261001T131304Z" / "run" / "batches"

PRIMARY_RADIUS = 1800
SENSITIVITY_RADII = (2000,)
ANCHOR_OFFSET_MS = 10_000
FRAME_LAG_MIN_MS = 10_000
FRAME_LAG_MAX_MS = 20_000
PRIMARY_PRE_MIN_MS = 120_000
MAIN_HORIZON_S = 90
SENSITIVITY_HORIZONS_S = (60, 120)

EXPECTED_MATCHES = 74_673
EXPECTED_COMPONENTS = 2_299_223
EXPECTED_KILLS = 3_752_984
EXPECTED_KILLS_SHA256 = "ad9c6377c404c04a0743f85f36894fb3c835a2ec216617ee3d248b0ce096d7a6"
EXPECTED_LABEL_SOURCE_MANIFEST_SHA256 = "39bfec2357debae5105d20f569c625673698693141680affbda676b79b334733"
SOURCE_LEDGER_SHA256 = "2ad2d92c81c6ea2f36fed972321ac1ab5bb9c4d17a9716fc02e91e3567023825"
SOURCE_RANGES_SHA256 = "930000be1285457359d268b2d3ccb3f6dbf1093e27a20b186ed1f5b06c35ac57"
SOURCE_MATCH_META_SHA256 = "734c3efd319b7c1bdc56c3b0893d10fc9ab9a365661da59f15025d621ff4d5c4"
EXPECTED_CANDIDATE_COMPONENTS = 378_703
EXPECTED_ISOLATED_CANDIDATES = 271_767

REQUIRED_RESULT_FIELDS = (
    "contract_version",
    "contract_sha256",
    "source_manifest_hashes",
    "canonical_match_key",
    "component_key",
    "firstkill_ms",
    "lastkill_ms",
    "pre_ms",
    "cohort_time_role",
    "firstkill_credit_blue",
    "firstkill_credit_red",
    "opposition_evidence",
    "critical_source_status",
    "timing_status",
    "proxy_P0_lower_upper_r1800",
    "proxy_P0_lower_upper_r2000",
    "starting_scale_group",
    "actual_participation_verified",
    "eligibility_status",
    "all_reasons",
    "final_population_ready",
)

SOURCE_ARTIFACTS = (
    "RUN_MANIFEST.json",
    "components.npz",
    "evidence.npz",
    "reference_fields.npz",
)
