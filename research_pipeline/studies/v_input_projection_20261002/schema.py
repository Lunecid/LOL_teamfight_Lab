"""Executable schema for the lossless primary V-input projection.

The schema is deliberately small and ordered.  It names the blocks consumed by
the P1 writer without importing a model, fitting a vocabulary, or exposing
audit/target fields as model inputs.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from studies.observable_state_20261002.kernel import (
    EVENT_PAYLOAD_FIELDS,
    EVENT_STRING_ENUMS,
    EVENT_STRING_FIELDS,
    EVENT_TYPE_NAMES,
    ITEM_TYPES,
    OBJECTIVE_TYPES,
    STATIC_NAMES,
)
from studies.observable_state_20261002.schema import frame_spec


VERSION = "v_primary_raw_projection_v2_20261002"
PARTICIPANTS = 10
FRAME_WIDTH = 46
STATIC_WIDTH = 14
BANS_SHAPE = (2, 5)
OBJECTIVE_TEAM_AXIS = ("total", "blue", "red")
OBJECTIVE_TOTAL_ONLY = True
FULL_INVENTORY_COMPLETE = "unknown"

# Public writer-facing specs use stable dtype strings and trailing shapes.  A
# row's query dimension is prepended by project_match/the P1 writer.
QUERY_BLOCK_SPECS = OrderedDict(
    (
        ("node_frame", ("<f4", (10, 46))),
        ("elapsed_minutes", ("<f8", ())),
        ("frame_age_seconds", ("<f8", ())),
        ("team_growth", ("<f8", (2, 6))),
        ("team_growth_delta", ("<f8", (6,))),
        ("cached_gold", ("<f4", (2,))),
        ("team_kills", ("<i4", (2,))),
        ("player_kda", ("<i4", (10, 3))),
        # B team counts are audit-only beyond the verified total.  The model
        # block therefore has the contract's category-total axis only.
        ("objective_counts", ("<i4", (9,))),
        ("objective_age_seconds", ("<f8", (9,))),
    )
)

STATIC_BLOCK_SPECS = OrderedDict(
    (
        ("static_ids", ("<i8", (10, 14))),
        ("bans", ("<i8", (2, 5))),
        ("team_ids", ("<i8", (10,))),
    )
)

REFERENCE_SPECS = OrderedDict(
    (
        ("query_ms", ("<i8", ())),
        ("query_frame_index", ("<i4", ())),
        ("query_event_limit", ("<i4", ())),
        ("query_item_limit", ("<i4", ())),
        ("query_item_ledger_known", ("?", ())),
    )
)

# These are the stored packed fields needed by the single-owner source reader.
# ``offset_<field>`` is added by the shard reader; alternatives such as the
# unpacked event matrices are accepted by project_match for small fixtures but
# are not requested from production archives.
REQUIRED_MATCH_FIELDS = (
    "minute_ts",
    "node_frame",
    "gold_team",
    "team_ids",
    "static_ids",
    "bans",
    "event_timestamp",
    "event_type_code",
    "event_category_code",
    "event_field_width",
    "event_field_row",
    "event_field_col",
    "event_field_nonzero",
    "event_field_valid_mask",
    "event_assist_counts",
    "event_assist_values",
    "event_assist_validity",
    "event_assist_presence",
    "event_string_width",
    "event_string_row",
    "event_string_col",
    "event_string_nonzero",
    "event_string_valid_mask",
    "event_string_unknown_rows",
    "event_string_unknown_cols",
    "event_string_unknown_offsets",
    "event_string_unknown_values",
    "event_raw_rows",
    "event_raw_cols",
    "event_raw_offsets",
    "event_raw_values",
    "item_event_indices",
    "query_ms",
    "query_frame_index",
    "query_event_limit",
    "query_item_limit",
    "query_team_kills",
    "query_kda",
    "query_objective_counts",
    "query_objective_last_ms",
    "query_item_ledger_known",
)

# Raw fields retained in semantic history.  Numeric fields are intentionally
# named with the source's spelling so presence/validity and zero values remain
# lossless.  The excluded numeric fields are never emitted even when present
# in the packed bank.
HISTORY_NUMERIC_FIELDS = (
    "participantId",
    "itemId",
    "beforeId",
    "afterId",
    "killerId",
    "victimId",
    "killerTeamId",
    "teamId",
    "level",
    "skillSlot",
)
HISTORY_STRING_FIELDS = tuple(EVENT_STRING_FIELDS)
HISTORY_ASSIST_FIELD = "assistingParticipantIds"
HISTORY_ALLOWED_EVENT_TYPES = (
    "CHAMPION_KILL",
    "ELITE_MONSTER_KILL",
    "BUILDING_KILL",
    "TURRET_PLATE_DESTROYED",
    "DRAGON_SOUL_GIVEN",
    "ITEM_PURCHASED",
    "ITEM_SOLD",
    "ITEM_DESTROYED",
    "ITEM_UNDO",
    "LEVEL_UP",
    "SKILL_LEVEL_UP",
)
HISTORY_PARTITIONS = OrderedDict(
    (
        ("B.team_kills", ("CHAMPION_KILL",)),
        (
            "B.objective_structure_history",
            ("ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "DRAGON_SOUL_GIVEN"),
        ),
        ("C.item_transaction_history", ITEM_TYPES),
        ("E.confirmed_level", ("LEVEL_UP",)),
        ("R.confirmed_skill_history", ("SKILL_LEVEL_UP",)),
        ("D.player_kills_deaths_assists", ("CHAMPION_KILL",)),
    )
)


def _block_manifest(specs: OrderedDict[str, tuple[str, tuple[int, ...]]], *, query: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, (dtype, tail)) in enumerate(specs.items()):
        rows.append(
            {
                "name": name,
                "index": index,
                "dtype": dtype,
                "shape": (["Q"] + list(tail)) if query else list(tail),
                "query_leading": bool(query),
            }
        )
    return rows


def field_manifest() -> dict[str, Any]:
    """Return the ordered JSON-serializable P1 field manifest."""

    frame_names, source_indices, groups = frame_spec()
    frame_fields = [
        {
            "name": name,
            "index": index,
            "source_index": int(source_indices[index]),
            "group": group,
            "dtype": "float32",
            "source_path": f"node_frame[F,participantId-1,{int(source_indices[index])}]",
            "timestamp_rule": "latest supported frame timestamp <= query_ms",
            "policy": "retain cached value exactly; zero is retained",
        }
        for index, (name, group) in enumerate(zip(frame_names, groups))
    ]
    history_fields = []
    for name, event_types in HISTORY_PARTITIONS.items():
        history_fields.append(
            {
                "name": name,
                "event_types": list(event_types),
                "cutoff": "event timestamp <= query_ms, inclusive",
                "fields": list(HISTORY_NUMERIC_FIELDS) + list(HISTORY_STRING_FIELDS) + [HISTORY_ASSIST_FIELD],
                "source_identity": "excluded from model history; source order is retained",
            }
        )
    model_input_fields = []
    for block in QUERY_BLOCK_SPECS:
        group = "B" if block in {"elapsed_minutes", "frame_age_seconds", "team_growth", "team_growth_delta", "cached_gold", "team_kills", "objective_counts", "objective_age_seconds"} else "E/R/D"
        model_input_fields.append({"id": f"{group}.{block}", "block": block, "dtype": QUERY_BLOCK_SPECS[block][0], "shape": ["Q", *QUERY_BLOCK_SPECS[block][1]]})
    model_input_fields.extend(
        {"id": f"C.{block}", "block": block, "dtype": STATIC_BLOCK_SPECS[block][0], "shape": list(STATIC_BLOCK_SPECS[block][1])}
        for block in STATIC_BLOCK_SPECS
    )
    return {
        "version": VERSION,
        "scope": "allowlisted lossless B/E/C/R/D raw projection; no fitting or learned transform",
        "participant_order": list(range(1, PARTICIPANTS + 1)),
        "frame_width": FRAME_WIDTH,
        "static_width": STATIC_WIDTH,
        "node_frame_fields": frame_fields,
        "query_blocks": _block_manifest(QUERY_BLOCK_SPECS, query=True),
        "static_blocks": _block_manifest(STATIC_BLOCK_SPECS, query=False),
        "reference_blocks": _block_manifest(REFERENCE_SPECS, query=True),
        "model_input_fields": model_input_fields,
        "ordered_fields": model_input_fields,
        "objective": {
            "categories": list(OBJECTIVE_TYPES),
            "count_team_axis_source": list(OBJECTIVE_TEAM_AXIS),
            "model_count_axis": "total only",
            "counts_shape": [9],
            "age": "seconds since last category timestamp, independent of team",
            "owner_inference": False,
        },
        "team_growth": {
            "axis": ["blue", "red"],
            "fields": ["level_mean", "xp_sum", "curGold_sum", "totalGold_sum", "laneCS_sum", "jgCS_sum"],
            "team_mapping": "actual team_ids per participant",
            "nonfinite": "any contributing value makes that aggregate unknown (NaN)",
        },
        "history": {
            "partitions": history_fields,
            "event_types": list(HISTORY_ALLOWED_EVENT_TYPES),
            "string_enums": {name: list(values) for name, values in EVENT_STRING_ENUMS.items()},
            "numeric_source_order": list(EVENT_PAYLOAD_FIELDS),
            "same_timestamp": "all records at timestamp <= query_ms are included in stable source order",
            "missing_vs_invalid": "field_presence and field_validity are separate; explicit zero is present and valid",
        },
        "items": {
            "item_reference_required": True,
            "accepted_event_types": list(ITEM_TYPES),
            "full_inventory_complete": FULL_INVENTORY_COMPLETE,
            "whole_inventory_model_claim": False,
            "ledger": "bind accepted observed-item v3 query references separately",
        },
        "excluded_model_paths": [
            "xy", "event positions", "source identity", "source event index", "GAME_END", "winningTeam",
            "goldGain", "bounty", "shutdownBounty", "killStreakLength", "audit", "end_ms", "future suffix",
            "inferred owner/acquirer", "alive", "cooldown", "arbitrary outer fields",
        ],
        "required_match_fields": list(REQUIRED_MATCH_FIELDS),
        "fitted_vocabularies": False,
        "fitted_numeric_preprocessing": False,
        "training_ready": False,
    }


__all__ = [
    "BANS_SHAPE",
    "EVENT_STRING_FIELDS",
    "FULL_INVENTORY_COMPLETE",
    "FRAME_WIDTH",
    "HISTORY_ALLOWED_EVENT_TYPES",
    "HISTORY_ASSIST_FIELD",
    "HISTORY_NUMERIC_FIELDS",
    "HISTORY_PARTITIONS",
    "HISTORY_STRING_FIELDS",
    "ITEM_TYPES",
    "OBJECTIVE_TEAM_AXIS",
    "OBJECTIVE_TOTAL_ONLY",
    "PARTICIPANTS",
    "QUERY_BLOCK_SPECS",
    "REFERENCE_SPECS",
    "REQUIRED_MATCH_FIELDS",
    "STATIC_BLOCK_SPECS",
    "STATIC_NAMES",
    "STATIC_WIDTH",
    "VERSION",
    "field_manifest",
]
