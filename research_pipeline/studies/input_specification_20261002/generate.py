"""Generate a fresh declared-source inventory and timing-rule package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema_extract import (
    DEFAULT_API_INVENTORY,
    DEFAULT_LABEL_CONTRACT,
    DEFAULT_SOURCE_ROOT,
    build_inventory,
    write_field_csv,
    write_support_csv,
)
from .timing import PENDING_DECISIONS, rule_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_SOURCE_ROOT.parents[1] / "reports" / "input_population_specification_20261002" / f"run_{stamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {path.name: _sha256(path) for path in sorted(paths, key=lambda item: item.name)}


def generate(output_dir: Path) -> Path:
    """Write a new run directory; refuse any existing directory."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    inventory = build_inventory(
        source_root=DEFAULT_SOURCE_ROOT,
        api_inventory_path=DEFAULT_API_INVENTORY,
        label_contract_path=DEFAULT_LABEL_CONTRACT,
    )
    contract = json.loads(DEFAULT_LABEL_CONTRACT.read_text(encoding="utf-8"))
    rules = {
        "schema_version": "eligibility_exclusion_table_v1_20261002",
        "contract_version": contract.get("contract_version"),
        "contract_sha256": _sha256(DEFAULT_LABEL_CONTRACT),
        "main_horizon_s": 90,
        "sensitivity_horizons_s": [60, 120],
        "timing_api": {
            "module": "studies.input_specification_20261002.timing",
            "input_type": "TimingInputs",
            "function": "evaluate_timing(inputs: TimingInputs) -> dict[str, Any]",
            "other_component_anchors": "sequence of (component_first_kill_ms, component_anchor_ms) pairs or mappings with those keys",
            "required_proofs": [
                "frame_source_verified is exactly True",
                "component_coverage_verified is exactly True",
                "kill_timeline_verified is exactly True",
            ],
            "main_gate": "timing_gate_passed uses h90; h60/h120 are sensitivity statuses",
        },
        "timing_gate_passed_is_distinct_from_final_population_ready": True,
        "rule_ids": rule_table(),
        "pending_decisions": [{"id": decision, "status": "pending"} for decision in PENDING_DECISIONS],
        "pending_policy": {
            "timing_gate_passed": "may be true when all required h90 timing rules pass",
            "final_population_ready": "must remain false while any pending decision remains",
            "unknown_to_exclude_mapping": False,
        },
        "exclusion_reasons": [
            "actual_frame_source_unverified",
            "timestamp_integrity_failure",
            "pre_negative_or_invalid_kill_order",
            "no_preceding_actual_frame",
            "first_kill_frame_lag_outside_inclusive_10000_20000",
            "complete_coverage_proof_missing_or_inconsistent",
            "GAME_END_missing_nonunique_or_not_strictly_after_last_kill",
            "foreign_kill_or_other_component_anchor_in_closed_[pre,L]",
            "next_component_anchor_le_L",
            "post_endpoint_unsupported_without_clamp",
        ],
        "non_filter_pending": list(PENDING_DECISIONS),
    }
    _write_json(output_dir / "FIELD_SPECIFICATION.json", inventory)
    write_field_csv(inventory, output_dir / "FIELD_SPECIFICATION.csv")
    write_support_csv(inventory, output_dir / "STATIC_METADATA_EVENT_PAYLOADS.csv")
    _write_json(output_dir / "ELIGIBILITY_EXCLUSION_TABLE.json", rules)
    readme = f"""# Input population specification run

This run contains declared-source decomposition only. It does not scan the
full corpus, claim representative field presence, choose final V/q inputs, or
produce a population. Every field row keeps raw presence and actual support
unverified, and no zero-valued missingness mask is fabricated.

Counts: node={inventory['declared_counts']['node']}, global={inventory['declared_counts']['global']}, event-vector={inventory['declared_counts']['event']}, static-metadata={inventory['declared_counts']['static_meta_definitions']}, event-payload={inventory['declared_counts']['event_payload_definitions']}.

The pure timing interface is `studies/input_specification_20261002.timing.TimingInputs`
and `evaluate_timing`. It requires source-verified actual frame timestamps and
independent complete component/kill timeline proof. The main timing gate is
h90; h60 and h120 remain sensitivity statuses. Invalid endpoints are null and
retain sorted truncation reasons.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifact_paths = [
        output_dir / name
        for name in (
            "FIELD_SPECIFICATION.json",
            "FIELD_SPECIFICATION.csv",
            "STATIC_METADATA_EVENT_PAYLOADS.csv",
            "ELIGIBILITY_EXCLUSION_TABLE.json",
            "README.md",
        )
    ]
    implementation_paths = [
        Path(__file__),
        Path(__file__).with_name("schema_extract.py"),
        Path(__file__).with_name("timing.py"),
    ]
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "files": [path.name for path in artifact_paths] + ["RUN_MANIFEST.json", "COMPLETED.json"],
        "artifact_sha256": _hash_files(artifact_paths),
        "implementation_sha256": {path.name: _sha256(path) for path in implementation_paths},
        "source_hashes": inventory["source_hashes"],
        "counts": inventory["declared_counts"],
        "scope": inventory["scope"],
        "training_ready": False,
        "full_corpus_scanned": False,
        "representative_observed_evidence": "unverified",
    }
    _write_json(output_dir / "RUN_MANIFEST.json", manifest)
    # This marker is written last. A completed directory is never overwritten.
    _write_json(
        output_dir / "COMPLETED.json",
        {
            "status": "completed",
            "manifest": "RUN_MANIFEST.json",
            "manifest_sha256": _sha256(output_dir / "RUN_MANIFEST.json"),
        },
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="new run directory; defaults to a UTC-named report run")
    args = parser.parse_args()
    output = generate(args.output or _default_output())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
