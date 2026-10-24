"""Read-only, side-effect-free extraction of the declared cache schema.

The production feature modules are intentionally not imported here.  The
configuration lists are evaluated from a small, allow-listed AST evaluator and
the official DTO snapshot is read as JSON.  This keeps the inventory useful
when production imports would require optional ML/runtime dependencies or
would create output directories.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "worktrees" / "engagement-state-value"
DEFAULT_CONFIG = DEFAULT_SOURCE_ROOT / "core" / "config.py"
DEFAULT_PIPELINE_CACHE = DEFAULT_SOURCE_ROOT / "gameplay" / "pipeline_cache.py"
DEFAULT_CACHE_IO = DEFAULT_SOURCE_ROOT / "data" / "cache_io.py"
DEFAULT_EVENT_AGGREGATION = DEFAULT_SOURCE_ROOT / "gameplay" / "event_aggregation.py"
DEFAULT_FEATURE_CONTRACT = DEFAULT_SOURCE_ROOT / "core" / "feature_contract.py"
DEFAULT_API_INVENTORY = (
    PROJECT_ROOT
    / "reports"
    / "feature_control_study_20261001"
    / "api_snapshot"
    / "official_dto_inventory.json"
)
DEFAULT_LABEL_CONTRACT = (
    PROJECT_ROOT / "reports" / "post_state_label_contract_20261002" / "CONTRACT.json"
)


class SchemaExtractionError(ValueError):
    """Raised when a declared source cannot be read safely."""


class _LiteralEvaluator:
    """Allow-list evaluator for the simple literals used in config.py."""

    def __init__(self, bindings: Mapping[str, Any]):
        self.bindings = bindings

    def eval(self, node: ast.AST, local: Mapping[str, Any] | None = None) -> Any:
        scope = dict(self.bindings)
        if local:
            scope.update(local)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in scope:
                raise SchemaExtractionError(f"unknown literal name: {node.id}")
            return scope[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out: list[Any] = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    out.extend(self.eval(elt.value, scope))
                else:
                    out.append(self.eval(elt, scope))
            container = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)]
            return container(out)
        if isinstance(node, ast.Dict):
            return {
                self.eval(k, scope): self.eval(v, scope)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.eval(node.left, scope) + self.eval(node.right, scope)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.eval(node.operand, scope)
            return -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.ListComp):
            if len(node.generators) != 1:
                raise SchemaExtractionError("only one list-comprehension generator is allowed")
            gen = node.generators[0]
            if gen.ifs or not isinstance(gen.target, ast.Name):
                raise SchemaExtractionError("unsupported list-comprehension filter")
            values = self.eval(gen.iter, scope)
            return [
                self.eval(node.elt, {**scope, gen.target.id: value})
                for value in values
            ]
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    parts.append(str(self.eval(value.value, scope)))
                else:
                    raise SchemaExtractionError("unsupported f-string node")
            return "".join(parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "len" and len(node.args) == 1:
                return len(self.eval(node.args[0], scope))
            if node.func.id == "range" and 1 <= len(node.args) <= 3:
                values = [self.eval(arg, scope) for arg in node.args]
                return range(*values)
        raise SchemaExtractionError(f"unsupported literal AST node: {type(node).__name__}")


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def parse_config_literals(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, int]]:
    """Return selected top-level config literals and their declaration lines."""
    text = config_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(config_path))
    bindings: dict[str, Any] = {}
    lines: dict[str, int] = {}
    evaluator = _LiteralEvaluator(bindings)
    for statement in tree.body:
        name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = _target_name(statement.targets[0])
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign):
            name = _target_name(statement.target)
            value_node = statement.value
        if not name or value_node is None:
            continue
        try:
            value = evaluator.eval(value_node)
        except SchemaExtractionError:
            continue
        bindings[name] = value
        lines[name] = int(statement.lineno)
    required = {
        "NODE_FEATURE_NAMES",
        "EVENT_FEATURE_NAMES",
        "GLOBAL_FEATURE_NAMES",
        "NODE_SNAPSHOT_FEATURE_NAMES",
        "NODE_STATUS_FEATURE_NAMES",
        "RUNE_FEATURE_NAMES",
        "CHAMPION_STATS_KEYS",
        "DAMAGE_STATS_KEYS",
    }
    missing = sorted(required - bindings.keys())
    if missing:
        raise SchemaExtractionError(f"config declarations not extracted: {missing}")
    return bindings, lines


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _pointer(path: Path, needle: str, occurrence: int = 1) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen = 0
    for line_no, line in enumerate(lines, 1):
        if needle in line:
            seen += 1
            if seen == occurrence:
                return f"{_relative(path)}:{line_no}"
    return f"{_relative(path)}:?"


def _inventory_dtos(api_path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    payload = json.loads(api_path.read_text(encoding="utf-8"))
    dtos = payload.get("dtos")
    if not isinstance(dtos, dict):
        raise SchemaExtractionError("official DTO inventory has no dtos object")
    return dtos, payload


def _dto_field(
    dtos: Mapping[str, Sequence[Mapping[str, str]]], dto: str, field: str
) -> dict[str, str] | None:
    for entry in dtos.get(dto, ()):
        if str(entry.get("name")) == field:
            return {"dto": dto, "field": field, "type": str(entry.get("type", ""))}
    return None


def _api_ref(
    dtos: Mapping[str, Sequence[Mapping[str, str]]], dto: str, field: str, api_path: Path
) -> dict[str, Any]:
    found = _dto_field(dtos, dto, field)
    result: dict[str, Any] = {
        "dto": dto,
        "field": field,
        "pointer": f"{_relative(api_path)}#/dtos/{dto}/{field}",
        "official_declared": found is not None,
    }
    if found is not None:
        result["declared_type"] = found["type"]
    return result


def _evidence(raw: bool = True) -> dict[str, Any]:
    return {
        "raw_presence_unknown": bool(raw),
        "representative_observed_evidence": "unverified",
        "full_observed_evidence": "unverified",
        "per_field_actual_support": "unverified",
        "zero_as_missing_mask": "not_created",
    }


def _raw(
    path: str,
    *,
    api_refs: Sequence[dict[str, Any]] = (),
    source_kind: str = "timeline_payload",
) -> dict[str, Any]:
    return {"path": path, "source_kind": source_kind, "api_refs": list(api_refs)}


def _source_pointer_map(
    *,
    config_path: Path,
    pipeline_path: Path,
    cache_io_path: Path,
    event_path: Path,
) -> dict[str, str]:
    return {
        "node_storage": _pointer(pipeline_path, "node_minute = np.zeros"),
        "node_assignment": _pointer(pipeline_path, "node_minute[t, pid - 1] = vec"),
        "global_storage": _pointer(pipeline_path, "global_minute = np.zeros"),
        "global_assignment": _pointer(pipeline_path, "global_minute[t] = build_global_minute_vector"),
        "events_storage": _pointer(pipeline_path, "all_events.append(e)"),
        "events_save": _pointer(cache_io_path, 'json.dump(cache["events"]'),
        "event_aggregation": _pointer(event_path, "def aggregate_events"),
        "cache_load": _pointer(cache_io_path, 'node = arr["node_minute"]'),
        "feature_contract": _pointer(config_path, "FEATURE_CONTRACT = build_feature_contract"),
    }


def _node_raw_and_transform(
    name: str,
    *,
    dtos: Mapping[str, Sequence[Mapping[str, str]]],
    api_path: Path,
    pipeline_path: Path,
    cache_io_path: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    participant = lambda field: _api_ref(dtos, "ParticipantFrameDto", field, api_path)
    detail_participant = lambda field: _api_ref(dtos, "ParticipantDto", field, api_path)
    champion = lambda field: _api_ref(dtos, "ChampionStatsDto", field, api_path)
    damage = lambda field: _api_ref(dtos, "DamageStatsDto", field, api_path)
    common = lambda field: _api_ref(dtos, "EventsTimeLineDto", field, api_path)

    if name == "champion_id":
        return [_raw("detail.info.participants[].championId", api_refs=[detail_participant("championId")], source_kind="static_metadata")], "detail.info.participants[].championId", "cache_io static extraction; invalid/nonpositive values fall back to hashed champion name"
    if name == "champion_name_id":
        return [_raw("detail.info.participants[].championName", api_refs=[detail_participant("championName")], source_kind="static_metadata")], "detail.info.participants[].championName", "lowercase string -> blake2b digest -> bounded vocabulary ID; collision possible"
    if name in {"summoner_spell_1_id", "summoner_spell_2_id"}:
        field = "summoner1Id" if name.endswith("1_id") else "summoner2Id"
        return [_raw(f"detail.info.participants[].{field}", api_refs=[detail_participant(field)], source_kind="static_metadata")], f"detail.info.participants[].{field}", "safe integer conversion; invalid/missing becomes 0 in static_meta"
    if name == "x_norm":
        return [_raw("timeline.info.frames[].participantFrames[].position.x", api_refs=[_api_ref(dtos, "PositionDto", "x", api_path)])], "timeline.info.frames[].participantFrames[].position.x", "divide by COORD_NORM_DIV (MAP_MAX=16000), clip to [0,2]"
    if name == "y_norm":
        return [_raw("timeline.info.frames[].participantFrames[].position.y", api_refs=[_api_ref(dtos, "PositionDto", "y", api_path)])], "timeline.info.frames[].participantFrames[].position.y", "divide by COORD_NORM_DIV (MAP_MAX=16000), clip to [0,2]"
    timeline_fields = {
        "level_norm": ("level", "divide by 18, clip to [0,2]"),
        "xp_norm": ("xp", "divide by 20000, clip to [0,5]"),
        "curGold_norm": ("currentGold", "divide by 4000, clip to [0,5]"),
        "totalGold_norm": ("totalGold", "divide by 25000, clip to [0,5]"),
        "laneCS_norm": ("minionsKilled", "divide by 400, clip to [0,5]"),
        "jgCS_norm": ("jungleMinionsKilled", "divide by 250, clip to [0,5]"),
        "ccTime_norm": ("timeEnemySpentControlled", "divide by 600, clip to [0,5]"),
    }
    if name in timeline_fields:
        field, transform = timeline_fields[name]
        return [_raw(f"timeline.info.frames[].participantFrames[].{field}", api_refs=[participant(field)])], f"timeline.info.frames[].participantFrames[].{field}", transform
    if name == "gps_norm":
        return [_raw("timeline.info.frames[].participantFrames[].totalGold", api_refs=[participant("totalGold")])], "timeline.info.frames[].participantFrames[].totalGold + frame timestamp", "inner clip totalGold / max(1, timestamp_ms/1000) to [0,200], then divide by 30 and outer clip to [0,5]; not Riot goldPerSecond; uses elapsed frame timestamp"
    if name == "hp_pct":
        return [_raw("timeline.info.frames[].participantFrames[].championStats.health", api_refs=[champion("health")]), _raw("timeline.info.frames[].participantFrames[].championStats.healthMax", api_refs=[champion("healthMax")])], "timeline.info.frames[].participantFrames[].championStats.{health,healthMax}", "health / healthMax, clip to [0,1]; healthMax<=0 -> 0"
    if name == "mp_pct":
        return [_raw("timeline.info.frames[].participantFrames[].championStats.power", api_refs=[champion("power")]), _raw("timeline.info.frames[].participantFrames[].championStats.powerMax", api_refs=[champion("powerMax")])], "timeline.info.frames[].participantFrames[].championStats.{power,powerMax}", "power / powerMax, clip to [0,1]; powerMax<=0 -> 0"
    if name == "alive":
        return [_raw("timeline.info.frames[].participantFrames[].championStats.health", api_refs=[champion("health")])], "timeline.info.frames[].participantFrames[].championStats.health", "derived indicator health > 0; missing/invalid values are converted by production safe_float"
    if name in {"has_baron", "has_elder", "baron_remain_norm", "elder_remain_norm"}:
        event_name = "BARON_NASHOR" if "baron" in name else "DRAGON/ELDER"
        return [_raw("timeline.info.frames[].events[]", api_refs=[common("timestamp"), common("type")], source_kind="event_payload")], f"event type={event_name}; timestamp; killerTeamId", "reconstructed timer state from prior events; duration configured as Baron 180s / Elder 150s; evaluate at frame timestamp"
    if name.startswith("soul_"):
        return [_raw("timeline.info.frames[].events[]", api_refs=[common("timestamp"), common("type")], source_kind="event_payload")], "event type=ELITE_MONSTER_KILL/DRAGON_SOUL_GIVEN; monsterSubType/dragonSoul/soulType", "reconstructed team dragon count/soul state from prior events"
    if name == "ult_level_norm":
        return [_raw("timeline.info.frames[].events[]", api_refs=[common("timestamp"), common("type")], source_kind="event_payload")], "event type=SKILL_LEVEL_UP; participantId; skillSlot", "count prior skillSlot=4 events, cap at 3, divide by 3"
    rune_map = {
        "primary_style_id": "styles[0].style",
        "sub_style_id": "styles[1].style",
        "primary_rune_1": "styles[0].selections[0].perk",
        "primary_rune_2": "styles[0].selections[1].perk",
        "primary_rune_3": "styles[0].selections[2].perk",
        "primary_rune_4": "styles[0].selections[3].perk",
        "sub_rune_1": "styles[1].selections[0].perk",
        "sub_rune_2": "styles[1].selections[1].perk",
        "stat_perk_offense": "statPerks.offense",
        "stat_perk_flex": "statPerks.flex",
        "stat_perk_defense": "statPerks.defense",
    }
    if name in rune_map:
        refs = [_api_ref(dtos, "ParticipantDto", "perks", api_path)]
        refs += [_api_ref(dtos, "PerksDto", "styles", api_path), _api_ref(dtos, "PerksDto", "statPerks", api_path)]
        return [_raw(f"detail.info.participants[].perks.{rune_map[name]}", api_refs=refs, source_kind="static_metadata")], f"detail.info.participants[].perks.{rune_map[name]}", "safe integer extraction; missing/invalid selections become 0"
    if name.startswith("cs_"):
        raw_name = name[3:]
        refs = [_api_ref(dtos, "ChampionStatsDto", raw_name, api_path)]
        div100 = raw_name in {"attackSpeed", "armorPenPercent", "bonusArmorPenPercent", "bonusMagicPenPercent", "ccReduction", "cooldownReduction", "lifesteal", "omnivamp", "physicalVamp", "spellVamp", "magicPenPercent"}
        scale = "if absolute raw value > 2, divide by 100; " if div100 else ""
        return [_raw(f"timeline.info.frames[].participantFrames[].championStats.{raw_name}", api_refs=refs)], f"timeline.info.frames[].participantFrames[].championStats.{raw_name}", f"{scale}divide by CS_DENOM[{raw_name}], clip to [-10,10]"
    if name.startswith("ds_"):
        raw_name = name[3:]
        return [_raw(f"timeline.info.frames[].participantFrames[].damageStats.{raw_name}", api_refs=[_api_ref(dtos, "DamageStatsDto", raw_name, api_path)])], f"timeline.info.frames[].participantFrames[].damageStats.{raw_name}", f"divide by DS_DENOM[{raw_name}], clip to [-10,10]"
    raise SchemaExtractionError(f"no node mapping for {name}")


_EVENT_RAW_FIELDS: dict[str, tuple[str, list[str], str]] = {
    "kills": ("CHAMPION_KILL", ["killerId", "timestamp"], "count event; team from team_map[killerId]"),
    "bounty": ("CHAMPION_KILL", ["killerId", "bounty", "shutdownBounty", "timestamp"], "sum bounty + shutdownBounty by killer team"),
    "shutdown_kill": ("CHAMPION_KILL", ["killerId", "shutdownBounty", "timestamp"], "count when shutdownBounty > 0"),
    "killstreak": ("CHAMPION_KILL", ["killerId", "killStreakLength", "timestamp"], "sum nonnegative killStreakLength by killer team"),
    "multikill": ("CHAMPION_SPECIAL_KILL", ["killerId", "killType", "multiKillLength", "timestamp"], "sum multiKillLength for killType containing MULTI"),
    "ace": ("CHAMPION_SPECIAL_KILL", ["killerId", "killType", "timestamp"], "count killType containing ACE"),
    "dragon": ("ELITE_MONSTER_KILL", ["killerTeamId", "monsterType", "bounty", "timestamp"], "count DRAGON by killer team and sum objective bounty"),
    "baron": ("ELITE_MONSTER_KILL", ["killerTeamId", "monsterType", "bounty", "timestamp"], "count BARON_NASHOR by killer team and sum objective bounty"),
    "herald": ("ELITE_MONSTER_KILL", ["killerTeamId", "monsterType", "bounty", "timestamp"], "count RIFTHERALD by killer team and sum objective bounty"),
    "atakhan": ("ELITE_MONSTER_KILL", ["killerTeamId", "monsterType", "bounty", "timestamp"], "count ATAKHAN by killer team and sum objective bounty"),
    "horde": ("ELITE_MONSTER_KILL", ["killerTeamId", "monsterType", "bounty", "timestamp"], "count HORDE by killer team and sum objective bounty"),
    "tower": ("BUILDING_KILL", ["teamId", "buildingType", "bounty", "timestamp"], "count TOWER building by opposing victim team and sum objective bounty"),
    "inhib": ("BUILDING_KILL", ["teamId", "buildingType", "bounty", "timestamp"], "count INHIBITOR building by opposing victim team and sum objective bounty"),
    "plate": ("TURRET_PLATE_DESTROYED", ["teamId", "bounty", "timestamp"], "count plate by opposing victim team and sum objective bounty"),
    "obj_bounty": ("ELITE_MONSTER_KILL/BUILDING_KILL/TURRET_PLATE_DESTROYED", ["bounty", "timestamp"], "sum nonnegative objective bounty by derived team"),
    "ward_placed": ("WARD_PLACED", ["creatorId", "wardType", "timestamp"], "count by creator team"),
    "ward_kill": ("WARD_KILL", ["killerId", "wardType", "timestamp"], "count by killer team"),
    "control_ward_placed": ("WARD_PLACED", ["creatorId", "wardType", "timestamp"], "subset count when wardType contains CONTROL"),
    "control_ward_kill": ("WARD_KILL", ["killerId", "wardType", "timestamp"], "subset count when wardType contains CONTROL"),
    "item_pur": ("ITEM_PURCHASED", ["participantId", "itemId", "timestamp"], "count item event by participant team"),
    "item_sold": ("ITEM_SOLD/ITEM_DESTROYED", ["participantId", "itemId", "timestamp"], "count item event by participant team"),
    "item_undo": ("ITEM_UNDO", ["participantId", "itemId", "timestamp"], "count item event by participant team"),
}


def _event_ref(field: str, dtos: Mapping[str, Sequence[Mapping[str, str]]], api_path: Path) -> dict[str, Any]:
    dto_field = field if field in {"timestamp", "realTimestamp", "type"} else None
    if dto_field:
        return _api_ref(dtos, "EventsTimeLineDto", dto_field, api_path)
    return {
        "field": field,
        "pointer": f"{_relative(api_path)}#/event-variant-fields/{field}",
        "official_declared": False,
        "reason": "not present in official EventsTimeLineDto snapshot",
    }


def _event_source_pointer(field: str, event_path: Path, pipeline_path: Path) -> str:
    if field in {"timestamp", "type", "realTimestamp"}:
        return _pointer(pipeline_path, '"timestamp" in e and "type" in e')
    source_root = event_path.parent.parent
    candidates = [
        event_path,
        pipeline_path,
        source_root / "gameplay" / "cohorts_exact.py",
        source_root / "gameplay" / "fights.py",
        source_root / "gameplay" / "evidence_state.py",
        source_root / "gameplay" / "event_tokens.py",
        source_root / "gameplay" / "item_state.py",
        source_root / "gameplay" / "objective_timers.py",
    ]
    needle = f'e.get("{field}"'
    for candidate in candidates:
        if candidate.exists():
            pointer = _pointer(candidate, needle)
            if not pointer.endswith(":?"):
                return pointer
    return _pointer(event_path, needle)


def _static_meta_rows(
    *, dtos: Mapping[str, Sequence[Mapping[str, str]]], api_path: Path, cache_io_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("champion_by_pid", "detail.info.participants[].championId", "ParticipantDto", "championId", "safe int; fallback to champion_name_id when <=0"),
        ("champion_name_by_pid", "detail.info.participants[].championName", "ParticipantDto", "championName", "string retained; companion champion_name_id is hashed"),
        ("summoner_spells_by_pid.summoner_spell_1_id", "detail.info.participants[].summoner1Id", "ParticipantDto", "summoner1Id", "safe int; invalid -> 0"),
        ("summoner_spells_by_pid.summoner_spell_2_id", "detail.info.participants[].summoner2Id", "ParticipantDto", "summoner2Id", "safe int; invalid -> 0"),
        ("runes_by_pid.primary_style_id", "detail.info.participants[].perks.styles[0].style", "PerkStyleDto", "style", "safe int; missing -> 0"),
        ("runes_by_pid.sub_style_id", "detail.info.participants[].perks.styles[1].style", "PerkStyleDto", "style", "safe int; missing -> 0"),
        ("runes_by_pid.primary_rune_1..4", "detail.info.participants[].perks.styles[0].selections[0..3].perk", "PerkStyleSelectionDto", "perk", "safe int per selection; missing -> 0"),
        ("runes_by_pid.sub_rune_1..2", "detail.info.participants[].perks.styles[1].selections[0..1].perk", "PerkStyleSelectionDto", "perk", "safe int per selection; missing -> 0"),
        ("runes_by_pid.stat_perk_offense", "detail.info.participants[].perks.statPerks.offense", "PerkStatsDto", "offense", "safe int; missing -> 0"),
        ("runes_by_pid.stat_perk_flex", "detail.info.participants[].perks.statPerks.flex", "PerkStatsDto", "flex", "safe int; missing -> 0"),
        ("runes_by_pid.stat_perk_defense", "detail.info.participants[].perks.statPerks.defense", "PerkStatsDto", "defense", "safe int; missing -> 0"),
        ("bans.blue[0..4]/red[0..4]", "detail.info.teams[].bans[].championId", "BanDto", "championId", "first five per team; zero padding"),
    ]
    for storage_path, raw_path, dto, field, transform in specs:
        rows.append({
            "storage_path": f"meta.static_meta.{storage_path}",
            "raw_path": raw_path,
            "source_pointer": _pointer(cache_io_path, "def _extract_static_meta_from_detail"),
            "api": _api_ref(dtos, dto, field, api_path),
            "cache_declared": True,
            "cache_storage": "meta.json -> static_meta",
            "transform": transform,
            "evidence": _evidence(True),
        })
    return rows


def _event_payload_rows(
    *,
    dtos: Mapping[str, Sequence[Mapping[str, str]]],
    api_path: Path,
    event_path: Path,
    pipeline_path: Path,
) -> list[dict[str, Any]]:
    event_types = {
        "common": ["timestamp", "realTimestamp", "type"],
        "CHAMPION_KILL": ["killerId", "victimId", "assistingParticipantIds", "bounty", "shutdownBounty", "killStreakLength", "position"],
        "CHAMPION_SPECIAL_KILL": ["killerId", "killType", "multiKillLength"],
        "ELITE_MONSTER_KILL": ["killerTeamId", "killerId", "monsterType", "monsterSubType", "bounty", "position"],
        "BUILDING_KILL": ["teamId", "buildingType", "bounty", "laneType", "towerType", "position"],
        "TURRET_PLATE_DESTROYED": ["teamId", "bounty", "laneType"],
        "WARD_PLACED": ["creatorId", "wardType", "position"],
        "WARD_KILL": ["killerId", "wardType", "position"],
        "ITEM_PURCHASED/ITEM_SOLD/ITEM_DESTROYED/ITEM_UNDO": ["participantId", "itemId", "beforeId", "afterId"],
        "SKILL_LEVEL_UP": ["participantId", "skillSlot"],
        "DRAGON_SOUL_GIVEN": ["teamId", "dragonSoul", "soulType", "name"],
        "LEVEL_UP": ["participantId"],
    }
    rows: list[dict[str, Any]] = []
    for event_type, fields in event_types.items():
        for field in fields:
            official = field in {"timestamp", "realTimestamp", "type"}
            rows.append({
                "event_type": event_type,
                "payload_field": field,
                "source_pointer": _event_source_pointer(field, event_path, pipeline_path),
                "api": _event_ref(field, dtos, api_path),
                "official_api_declared": official,
                "cache_declared": True,
                "cache_storage": "events.json -> events[] dict",
                "usage_status": "catalogued_from_local_parser_or_guard_reference; corpus presence not asserted",
                "evidence": _evidence(True),
            })
    return rows


def build_inventory(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    config_path: Path | None = None,
    api_inventory_path: Path = DEFAULT_API_INVENTORY,
    label_contract_path: Path = DEFAULT_LABEL_CONTRACT,
) -> dict[str, Any]:
    """Build a declared-only inventory from the pinned local source files."""
    config_path = config_path or source_root / "core" / "config.py"
    pipeline_path = source_root / "gameplay" / "pipeline_cache.py"
    cache_io_path = source_root / "data" / "cache_io.py"
    event_path = source_root / "gameplay" / "event_aggregation.py"
    feature_contract_path = source_root / "core" / "feature_contract.py"

    bindings, declaration_lines = parse_config_literals(config_path)
    dtos, api_payload = _inventory_dtos(api_inventory_path)
    pointers = _source_pointer_map(
        config_path=config_path,
        pipeline_path=pipeline_path,
        cache_io_path=cache_io_path,
        event_path=event_path,
    )
    declaration_ref = lambda name: f"{_relative(config_path)}:{declaration_lines[name]}"

    group_names = [
        ("node_snapshot", "NODE_SNAPSHOT_FEATURE_NAMES"),
        ("node_status", "NODE_STATUS_FEATURE_NAMES"),
        ("node_rune", "RUNE_FEATURE_NAMES"),
        ("node_champion_stats", "CHAMPION_STATS_KEYS"),
        ("node_damage_stats", "DAMAGE_STATS_KEYS"),
        ("event_vector", "EVENT_FEATURE_NAMES"),
        ("global_vector", "GLOBAL_FEATURE_NAMES"),
    ]
    group_for_name: dict[str, str] = {}
    group_declarations: dict[str, str] = {}
    for group, decl in group_names:
        group_declarations[group] = declaration_ref(decl)
        for name in bindings[decl]:
            if group == "node_champion_stats":
                name = f"cs_{name}"
            elif group == "node_damage_stats":
                name = f"ds_{name}"
            group_for_name[str(name)] = group

    fields: list[dict[str, Any]] = []
    for block in ("node", "event", "global"):
        names = bindings[
            {"node": "NODE_FEATURE_NAMES", "event": "EVENT_FEATURE_NAMES", "global": "GLOBAL_FEATURE_NAMES"}[block]
        ]
        for index, actual_name in enumerate(names):
            actual_name = str(actual_name)
            group = group_for_name[actual_name]
            row: dict[str, Any] = {
                "field_id": f"{block}[{index}]",
                "storage_block": block,
                "index": index,
                "actual_name": actual_name,
                "declared_group": group,
                "declaration": {
                    "constant": {"node": "NODE", "event": "EVENT", "global": "GLOBAL"}[block] + "_FEATURE_NAMES",
                    "pointer": group_declarations[group],
                },
                "dtype": "float32",
                "cache_declared": block in {"node", "global"},
                "cache_storage": {
                    "node": "cache/<match_id>.npz::node_minute[t, participant_index, index]",
                    "event": "derived event vector; aggregate_events output is not written by cache_io",
                    "global": "cache/<match_id>.npz::global_minute[t, index]",
                }[block],
                "source_pointers": [pointers["feature_contract"]],
                "evidence": _evidence(True),
                "api_declared": False,
                "raw_parent_api_declared": False,
                "api_evidence": [],
                "raw_sources": [],
                "transform": "",
                "numeric_unit": "cache-space float32; exact source unit depends on raw path",
                "query_safety": "unverified",
                "provenance": "declared feature order only; no final V/q selection implied",
            }
            if block == "node":
                raw_sources, raw_path, transform = _node_raw_and_transform(
                    actual_name,
                    dtos=dtos,
                    api_path=api_inventory_path,
                    pipeline_path=pipeline_path,
                    cache_io_path=cache_io_path,
                )
                row["raw_sources"] = raw_sources
                row["transform"] = transform
                row["numeric_unit"] = (
                    "unitless normalized feature"
                    if actual_name.startswith(("cs_", "ds_"))
                    or actual_name.endswith(("_norm", "_pct", "_remain_norm"))
                    or actual_name == "alive"
                    else "categorical ID encoded as float32"
                )
                row["query_safety"] = "event/frame chronology source-specific; no observed support claim"
                row["source_pointers"] += [pointers["node_storage"], pointers["node_assignment"]]
            elif block == "event":
                base = actual_name.rsplit("_t", 1)[0]
                spec = _EVENT_RAW_FIELDS.get(base)
                if spec is None:
                    raise SchemaExtractionError(f"no event mapping for {actual_name}")
                event_type, event_fields, transform = spec
                refs = [_event_ref(field, dtos, api_inventory_path) for field in event_fields]
                row["raw_sources"] = [_raw(f"events[{event_type}]", api_refs=refs, source_kind="event_payload")]
                row["api_evidence"] = refs
                row["transform"] = transform
                row["numeric_unit"] = "event count or source bounty units"
                row["query_safety"] = "window cutoff delegated to aggregate_events; source timing evidence unverified"
                row["source_pointers"] += [pointers["events_storage"], pointers["event_aggregation"]]
            else:
                row["source_pointers"] += [pointers["global_storage"], pointers["global_assignment"]]
                global_map = {
                    "time_norm": ("frame index t + full T", "t / max(1,T-1); future-dependent on full match frame count; query-safe status unverified"),
                    **{f"blue_ban_{i}": ("meta.static_meta.bans.blue", "copied from static metadata; missing slots are zero-padded") for i in range(5)},
                    **{f"red_ban_{i}": ("meta.static_meta.bans.red", "copied from static metadata; missing slots are zero-padded") for i in range(5)},
                    "goldDiff": ("node_minute totalGold_norm + team_map", "team sum Blue - Red"),
                    "xpDiff": ("node_minute xp_norm + team_map", "team sum Blue - Red"),
                    "avgLevelDiff": ("node_minute level_norm + team_map", "team mean Blue - Red"),
                    "csDiff_total": ("node_minute laneCS_norm/jgCS_norm + team_map", "lane CS diff + jungle CS diff"),
                    "csJgDiff": ("node_minute jgCS_norm + team_map", "jungle CS diff"),
                    "aliveDiff": ("node_minute alive + team_map", "team sum Blue - Red"),
                    "killDiff_cum": ("history.kill from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "towerDiff_cum": ("history.tower from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "inhibDiff_cum": ("history.inhib from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "dragonDiff_cum": ("history.dragon from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "baronDiff_cum": ("history.baron from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "heraldDiff_cum": ("history.herald from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "atakhanDiff_cum": ("history.atakhan from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "plateDiff_cum": ("history.plate from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                    "hordeDiff_cum": ("history.horde from frame events", "cumulative signed team counter; frame event cutoff is unverified"),
                }
                source, transform = global_map[actual_name]
                row["raw_sources"] = [_raw(source, source_kind="derived_cache")]
                row["transform"] = transform
                row["numeric_unit"] = "unitless normalized difference or categorical ban ID"
                row["query_safety"] = (
                    "future_dependent_full_match_length" if actual_name == "time_norm"
                    else "frame-event cutoff unverified in legacy cache builder"
                    if actual_name.endswith("_cum") else "derived from query frame/static metadata"
                )
                row["api_declared"] = False
            fields.append(row)

    for row in fields:
        row["api_evidence"] = row.get("api_evidence") or [
            ref
            for raw in row["raw_sources"]
            for ref in raw.get("api_refs", [])
        ]
        row["raw_parent_api_declared"] = any(
            bool(ref.get("official_declared")) for ref in row["api_evidence"]
        )

    source_paths = [
        config_path,
        pipeline_path,
        cache_io_path,
        event_path,
        source_root / "gameplay" / "event_tokens.py",
        source_root / "gameplay" / "evidence_state.py",
        source_root / "gameplay" / "fights.py",
        source_root / "gameplay" / "cohorts_exact.py",
        source_root / "gameplay" / "item_state.py",
        source_root / "gameplay" / "objective_timers.py",
        feature_contract_path,
        api_inventory_path,
        label_contract_path,
    ]
    source_hashes = {_relative(path): _sha256(path) for path in source_paths if path.exists()}
    return {
        "schema_version": "input_field_specification_v1_20261002",
        "scope": "declared source decomposition only; no final V/q input selection",
        "source_mode": "cache_only",
        "raw_field_presence": "unknown",
        "zero_is_not_missingness_proof": True,
        "generated_from": {
            "config": _relative(config_path),
            "pipeline_cache": _relative(pipeline_path),
            "cache_io": _relative(cache_io_path),
            "event_aggregation": _relative(event_path),
            "official_dto_inventory": _relative(api_inventory_path),
            "adopted_label_contract": _relative(label_contract_path),
        },
        "source_hashes": source_hashes,
        "normalizers": {
            "scalar_node_denominators": {
                "level_norm": 18.0,
                "xp_norm": 20000.0,
                "curGold_norm": 4000.0,
                "totalGold_norm": 25000.0,
                "gps_norm": 30.0,
                "laneCS_norm": 400.0,
                "jgCS_norm": 250.0,
                "ccTime_norm": 600.0,
            },
            "scalar_node_clips": {
                "x_norm": [0.0, 2.0],
                "y_norm": [0.0, 2.0],
                "level_norm": [0.0, 2.0],
                "xp_norm": [0.0, 5.0],
                "curGold_norm": [0.0, 5.0],
                "totalGold_norm": [0.0, 5.0],
                "gps_norm": [0.0, 5.0],
                "laneCS_norm": [0.0, 5.0],
                "jgCS_norm": [0.0, 5.0],
                "ccTime_norm": [0.0, 5.0],
                "hp_pct": [0.0, 1.0],
                "mp_pct": [0.0, 1.0],
            },
            "champion_stats_div100_keys": list(bindings.get("CHAMPION_STATS_DIV100_KEYS", ())),
            "CS_DENOM": dict(bindings.get("CS_DENOM", {})),
            "DS_DENOM": dict(bindings.get("DS_DENOM", {})),
            "source_pointers": {
                "scalar_assignments": ";".join(
                    _pointer(pipeline_path, needle)
                    for needle in (
                        "DEN_LEVEL = 18.0", "DEN_XP = 20000.0", "DEN_CUR_G = 4000.0",
                        "DEN_TOT_G = 25000.0", "DEN_GPS = 30.0", "DEN_LANECS = 400.0",
                        "DEN_JGCS = 250.0", "DEN_CCT = 600.0",
                    )
                ),
                "champion_stats_normalization": _pointer(pipeline_path, "_normalize_cs_raw"),
                "champion_stats_clip": _pointer(pipeline_path, "np.clip(val, -10.0, 10.0)"),
                "damage_stats_clip": _pointer(pipeline_path, "np.clip(val, -10.0, 10.0)"),
            },
            "unit_policy": "raw DTO units are retained as declared names; normalized cache values are unitless unless a source unit is independently established",
        },
        "declared_counts": {
            "node": len(bindings["NODE_FEATURE_NAMES"]),
            "event": len(bindings["EVENT_FEATURE_NAMES"]),
            "global": len(bindings["GLOBAL_FEATURE_NAMES"]),
            "static_meta_definitions": len(_static_meta_rows(dtos=dtos, api_path=api_inventory_path, cache_io_path=cache_io_path)),
            "event_payload_definitions": len(_event_payload_rows(dtos=dtos, api_path=api_inventory_path, event_path=event_path, pipeline_path=pipeline_path)),
        },
        "cache_arrays": [
            {"name": "minute_ts", "storage": "cache/<match_id>.npz::minute_ts", "dtype": "int64", "shape": "(T,)", "timestamp_provenance": "source frame timestamp required; pipeline falls back to t*FRAME_MS when timestamp is missing/falsey; caller must provide source verification", "source_pointer": _pointer(pipeline_path, "t_ms = int(fr.get(\"timestamp\"")},
            {"name": "node_minute", "storage": "cache/<match_id>.npz::node_minute", "dtype": "float32", "shape": f"(T,10,{len(bindings['NODE_FEATURE_NAMES'])})", "field_order": "NODE_FEATURE_NAMES"},
            {"name": "global_minute", "storage": "cache/<match_id>.npz::global_minute", "dtype": "float32", "shape": f"(T,{len(bindings['GLOBAL_FEATURE_NAMES'])})", "field_order": "GLOBAL_FEATURE_NAMES"},
            {"name": "gold_team_minute", "storage": "cache/<match_id>.npz::gold_team_minute", "dtype": "float32", "shape": "(T,2)", "field_order": ["blue", "red"]},
            {"name": "xy_raw_minute", "storage": "cache/<match_id>.npz::xy_raw_minute", "dtype": "float32", "shape": "(T,10,2)", "field_order": ["x", "y"]},
            {"name": "events", "storage": "cache/<match_id>.events.json::[]", "dtype": "event dict", "shape": "variable", "field_order": "event payload definitions"},
            {"name": "meta", "storage": "cache/<match_id>.meta.json", "dtype": "JSON object", "shape": "object", "field_order": "static metadata + provenance"},
        ],
        "fields": fields,
        "static_metadata": _static_meta_rows(dtos=dtos, api_path=api_inventory_path, cache_io_path=cache_io_path),
        "event_payloads": _event_payload_rows(dtos=dtos, api_path=api_inventory_path, event_path=event_path, pipeline_path=pipeline_path),
        "evidence_policy": {
            "representative_pack": "not read by this declared-only extraction",
            "full_corpus": "not scanned",
            "per_field_actual_support": "unverified for every row",
            "raw_presence_unknown": True,
            "inferred_zero_masks": False,
        },
    }


def write_field_csv(inventory: Mapping[str, Any], path: Path) -> None:
    rows = inventory["fields"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field_id", "storage_block", "index", "actual_name", "declared_group",
                "cache_declared", "cache_storage", "api_declared", "transform",
                "raw_parent_api_declared",
                "raw_presence_unknown", "representative_observed_evidence",
                "full_observed_evidence", "per_field_actual_support", "zero_as_missing_mask",
                "source_pointers", "raw_paths", "api_pointers",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "field_id": row["field_id"],
                "storage_block": row["storage_block"],
                "index": row["index"],
                "actual_name": row["actual_name"],
                "declared_group": row["declared_group"],
                "cache_declared": row["cache_declared"],
                "cache_storage": row["cache_storage"],
                "api_declared": row["api_declared"],
                "raw_parent_api_declared": row["raw_parent_api_declared"],
                "transform": row["transform"],
                **row["evidence"],
                "source_pointers": ";".join(row["source_pointers"]),
                "raw_paths": ";".join(raw["path"] for raw in row["raw_sources"]),
                "api_pointers": ";".join(ref["pointer"] for ref in row["api_evidence"]),
            })


def write_support_csv(inventory: Mapping[str, Any], path: Path) -> None:
    rows = inventory["static_metadata"] + inventory["event_payloads"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "definition_kind", "storage_path", "event_type", "payload_field", "raw_path",
                "source_pointer", "official_api_declared", "cache_declared", "cache_storage",
                "raw_presence_unknown", "representative_observed_evidence", "full_observed_evidence",
                "per_field_actual_support", "zero_as_missing_mask", "transform", "api_pointer",
            ],
        )
        writer.writeheader()
        for row in inventory["static_metadata"]:
            writer.writerow({
                "definition_kind": "static_metadata",
                "storage_path": row["storage_path"],
                "event_type": "",
                "payload_field": "",
                "raw_path": row["raw_path"],
                "source_pointer": row["source_pointer"],
                "official_api_declared": row["api"]["official_declared"],
                "cache_declared": row["cache_declared"],
                "cache_storage": row["cache_storage"],
                **row["evidence"],
                "transform": row["transform"],
                "api_pointer": row["api"]["pointer"],
            })
        for row in inventory["event_payloads"]:
            writer.writerow({
                "definition_kind": "event_payload",
                "storage_path": "",
                "event_type": row["event_type"],
                "payload_field": row["payload_field"],
                "raw_path": "events[]." + row["payload_field"],
                "source_pointer": row["source_pointer"],
                "official_api_declared": row["official_api_declared"],
                "cache_declared": row["cache_declared"],
                "cache_storage": row["cache_storage"],
                **row["evidence"],
                "transform": "payload definition used by current parser/guard",
                "api_pointer": row["api"]["pointer"],
            })
