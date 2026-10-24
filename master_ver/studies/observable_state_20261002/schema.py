"""Ordered observable fields, separated from query and cohort audit metadata."""
from __future__ import annotations

import json
from functools import lru_cache

from studies.input_specification_20261002.schema_extract import build_inventory, parse_config_literals
from .source import REPORT, ROOT

STATE_VERSION = "observable_state_v1_20261002"
POLICY_PATH = ROOT / "reports" / "input_population_specification_20261002" / "INPUT_POLICY.json"
STATIC_NAMES = (
    "champion_id", "summoner_spell_1_id", "summoner_spell_2_id",
    "primary_style_id", "sub_style_id", "primary_rune_1", "primary_rune_2",
    "primary_rune_3", "primary_rune_4", "sub_rune_1", "sub_rune_2",
    "stat_perk_offense", "stat_perk_flex", "stat_perk_defense",
)


@lru_cache(maxsize=1)
def frame_spec() -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    declared, _ = parse_config_literals()
    original = declared["NODE_FEATURE_NAMES"]
    names, indices, groups = [], [], []
    for group in ("E", "R", "D"):
        for name in policy["node_primary_frame_channels"][group]:
            names.append(name)
            indices.append(original.index(name))
            groups.append(group)
    if len(names) != 46 or len(set(indices)) != 46:
        raise ValueError("adopted frame policy is not the expected 46 unique channels")
    return tuple(names), tuple(indices), tuple(groups)


def make_field_manifest(history_fields: list[dict]) -> dict:
    """Bind executable history fields to the existing declared source inventory."""
    names, indices, groups = frame_spec()
    inventory = build_inventory()
    source_fields = {row["actual_name"]: row for row in inventory["fields"] if row["storage_block"] == "node"}
    fields = []
    for index, (name, source_index, group) in enumerate(zip(names, indices, groups)):
        original = source_fields[name]
        canonical = "last_frame_power_fraction" if name == "mp_pct" else name
        fields.append({"id": f"{group}.{canonical}", "block": "node_frame", "index": index,
                       "source_index": source_index, "group": group,
                       "source_path": f"npz.node_minute[F,participantId-1,{source_index}]",
                       "source_timestamp_rule": "latest supported frame F<=query",
                       "parent_fields": [f"cache.node.{name}"], "unit": original["numeric_unit"],
                       "dtype": "float32", "transform": "retain cached value; " + original["transform"],
                       "raw_sources": original["raw_sources"], "source_pointers": original["source_pointers"],
                       "missingness_or_presence_unknown": "raw presence unknown; zero retained",
                       "reconstruction_policy": "held observation; no interpolation or inverse clipping"})
    for index, name in enumerate(STATIC_NAMES):
        source = "champion_by_pid" if name == "champion_id" else (
            "summoner_spells_by_pid" if name.startswith("summoner_") else "runes_by_pid")
        fields.append({"id": f"C.{name}", "block": "static_ids", "index": index, "group": "C",
                       "source_path": f"meta.static_meta.{source}[participantId]",
                       "source_timestamp_rule": "fixed start selection only", "parent_fields": [source],
                       "unit": "recorded categorical ID", "dtype": "int64",
                       "transform": "identity; no name hash or ordinal normalization",
                       "missingness_or_presence_unknown": "raw presence unknown; canonical ID validation recorded separately",
                       "reconstruction_policy": "preserve cached selection; omit final effects and roles"})
    for axis in ("x", "y"):
        fields.append({"id": f"S.{axis}", "block": "xy", "index": 0 if axis == "x" else 1,
                       "group": "S", "source_path": "npz.xy_raw_minute[F,participantId-1]",
                       "source_timestamp_rule": "latest supported frame F<=query; q consumes pre only",
                       "parent_fields": ["cache.xy_raw_minute"], "unit": "map coordinate units",
                       "dtype": "float32", "transform": "identity",
                       "missingness_or_presence_unknown": "raw presence unknown; zero retained",
                       "reconstruction_policy": "no interpolation or future first-kill centering"})
    fields.extend(history_fields)
    ids = [row["id"] for row in fields]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate executable field IDs")
    required = {"id", "group", "source_path", "source_timestamp_rule", "parent_fields", "unit",
                "dtype", "transform", "missingness_or_presence_unknown", "reconstruction_policy"}
    if any(not required <= set(row) for row in fields):
        raise ValueError("executable field manifest lacks required provenance")
    return {"state_version": STATE_VERSION, "source_mode": "cache_only", "ordered_fields": fields,
            "frame_width": 46, "static_width": 14, "participant_order": list(range(1, 11)),
            "raw_presence_verified": False, "inferred_zero_masks": False,
            "main_V_groups": ["B", "E", "C", "R", "D"],
            "main_q_common_groups": ["B", "E", "C", "R"],
            "PT_direct_inputs": ["p_pre", "pre_ms/60000"],
            "audit_metadata_is_model_input": False,
            "fit_scope_transforms": "future model cards: vocabulary, scaling, imputation and spatial sigma",
            "training_ready": False, "new_models_fitted": False, "new_SVI_labels_produced": False}
