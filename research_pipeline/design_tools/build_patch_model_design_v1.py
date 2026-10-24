import os
"""Generate and validate design schemas; no model or match-data execution."""
from pathlib import Path
import hashlib
import importlib.util
import json

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ["LOL_RESEARCH_ROOT"])


def obj(properties, required=None):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties) if required is None else required}


def arr(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


def enum(values):
    return {"type": "string", "enum": values}


def ref(name):
    return {"$ref": "#/$defs/" + name}


def main():
    text = {"type": "string", "minLength": 1}
    nullable_text = {"type": ["string", "null"]}
    identifier = {"type": "string", "pattern": "^[A-Za-z0-9_.:-]+$"}
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    scalar = {"type": ["number", "boolean", "string", "null"]}
    value = {"anyOf": [scalar, arr({"type": ["number", "boolean", "string", "null"]})]}
    source_ref = obj({"source_id": identifier, "locator": text})
    status = enum(["VERIFIED", "UNVERIFIED", "UNKNOWN", "CONFLICT", "STRUCTURAL_NA"])
    fact = obj({"key": identifier, "value": value, "unit": text,
                "axis": enum(["scalar", "rank", "level", "threshold", "category_set"]),
                "status": status, "evidence": arr(ref("source_ref")), "reason": nullable_text})
    fact["allOf"] = [
        {"if": {"properties": {"status": {"const": "VERIFIED"}}},
         "then": {"properties": {"evidence": {"minItems": 1}, "value": {"not": {"type": "null"}}}}},
        {"if": {"properties": {"status": {"enum": ["UNKNOWN", "CONFLICT", "STRUCTURAL_NA"]}}},
         "then": {"properties": {"value": {"type": "null"}, "reason": text}}},
    ]
    skill = obj({"skill_key": identifier, "slot": enum(["PASSIVE", "Q", "W", "E", "R", "OTHER"]),
                 "form_key": nullable_text, "description_refs": arr(ref("source_ref")),
                 "numerical_facts": arr(ref("fact")), "mechanic_facts": arr(ref("fact")),
                 "unresolved_tokens": arr(text)})
    champion = obj({"champion_id": {"type": "integer", "minimum": 1}, "canonical_name": text,
                    "availability": ref("fact"), "base_facts": arr(ref("fact")),
                    "growth_facts": arr(ref("fact")), "skills": arr(ref("skill")),
                    "mechanic_facts": arr(ref("fact"))})
    entry = obj({"entity_id": identifier, "canonical_name": text,
                 "availability": ref("fact"), "facts": arr(ref("fact")),
                 "description_refs": arr(ref("source_ref"))})
    entity = obj({"entity_id": identifier,
                  "entity_type": enum(["STRUCTURE", "OBJECTIVE_SITE", "OBJECTIVE", "TEAM_ASSET"]),
                  "availability": ref("fact"), "facts": arr(ref("fact")),
                  "rule_ids": arr(identifier), "related_entity_ids": arr(identifier)})
    guard = obj({"state_key": identifier,
                 "operator": enum(["eq", "neq", "lt", "lte", "gt", "gte", "in", "supported"]),
                 "value": value})
    operations = ["spawn_after", "expire_after", "remove_on_event", "grant_to_eligible",
                  "count_threshold", "availability_by_patch", "update_counter", "schedule_respawn"]
    rule = obj({"rule_id": identifier, "trigger_type": text, "guards": arr(ref("guard")),
                "operation": enum(operations), "parameters": arr(ref("fact")),
                "state_dependencies": arr(identifier), "outputs": arr(identifier, 1),
                "status": status, "evidence": arr(ref("source_ref")),
                "exception_policy": enum(["PROPAGATE_UNKNOWN", "EXPLICIT_RULE_ONLY"]),
                "reason": nullable_text})
    source = obj({"source_id": identifier,
                  "source_class": enum(["RIOT_DDRAGON", "RIOT_PATCH_NOTE", "RIOT_OTHER", "COMMUNITYDRAGON", "LOCAL_VERIFIED"]),
                  "uri": text, "build": nullable_text, "sha256": sha,
                  "retrieved_at": text, "published_at": nullable_text})
    manifest = obj({"bundle_id": identifier, "match_patch": text, "public_patch": text,
                    "game_mode": {"const": "SUMMONERS_RIFT"}, "region_scope": arr(text, 1),
                    "ddragon_build": nullable_text, "cdragon_build": nullable_text,
                    "knowledge_cutoff": nullable_text,
                    "knowledge_time_policy": enum(["STRICT_AS_OF", "RETROSPECTIVE_VERSION_MATCHED"]),
                    "engine_interface_version": text, "ontology_version": text,
                    "geometry_version": nullable_text,
                    "compiler": obj({"version": text, "parser_version": text,
                                     "llm_model": nullable_text, "prompt_hash": {"anyOf": [sha, {"type": "null"}]}}),
                    "package_status": enum(["CANDIDATE", "VALIDATED", "PARTIALLY_SUPPORTED"])})
    semantic = obj({"artifact_id": identifier, "encoder_model": text, "encoder_revision": text,
                    "tokenizer_revision": text, "dimension": {"type": "integer", "minimum": 1},
                    "source_ids": arr(identifier, 1), "content_hash": sha,
                    "vectors_sha256": sha, "adaptation_scope": text})
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
              "title": "PatchKnowledgeBundle v1 proposal",
              "description": "Shape validation only. Cross-reference, provenance, unit, rule semantics and eligibility checks are mandatory separately.",
              **obj({"schema_version": {"const": "patch-knowledge-v1"}, "manifest": ref("manifest"),
                     "sources": arr(ref("source")),
                     "ontology": obj({"fact_keys": arr(identifier), "state_keys": arr(identifier),
                                      "relationship_types": arr(identifier), "supported_operations": arr(enum(operations))}),
                     "world": arr(ref("world_entity")), "champions": arr(ref("champion")),
                     "items": arr(ref("knowledge_entry")), "runes": arr(ref("knowledge_entry")),
                     "summoner_spells": arr(ref("knowledge_entry")), "rules": arr(ref("rule")),
                     "semantic_artifacts": arr(ref("semantic"))}),
              "$defs": {"manifest": manifest, "source": source, "source_ref": source_ref,
                        "fact": fact, "skill": skill, "champion": champion,
                        "knowledge_entry": entry, "world_entity": entity, "guard": guard,
                        "rule": rule, "semantic": semantic}}
    (OUT / "PATCH_KNOWLEDGE_SCHEMA_V1.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    contract = {
        "version": "patch-conditioned-hierarchy-v1-proposal",
        "status": "design_not_implemented_or_trained",
        "state_engine_contract": "RIFT_STATE_ENGINE_V1.json",
        "knowledge_schema": "PATCH_KNOWLEDGE_SCHEMA_V1.json",
        "baseline_features": "BASIC_FEATURE_CONTRACT_V1.json",
        "heads": ["player", "team_matchup", "world"],
        "all_heads_reach_final_fusion": True,
        "recommended_candidate": "player_temporal_then_team_set_cross_relations_then_world_set_then_joint_readout",
        "alternatives": ["same_information_flat", "parallel_readouts", "heterogeneous_graph"],
        "conditioning": ["concatenation_baseline", "FiLM_secondary"],
        "knowledge_routes": {"champion": "player_and_relations", "world_rules": "engine_and_world_context", "item_spell_rune": "player_when_state_supported"},
        "primary_task": "existing_observed_fight_gain_label_unchanged",
        "auxiliary_tasks": "optional_separate_ablations_not_required_for_named_heads",
        "cohort": {"anchor": "first_kill_ms - 10000", "frame_age_ms_inclusive": [0, 10000], "cutoff": "inclusive", "history_frames": [1, 3]},
        "patch_swap_scope": {"parameters_and_supported_entities": "bundle_update", "new_transition_operator": "engine_schema_revision", "prediction_accuracy": "empirical_not_guaranteed"},
        "llm": {"role": "offline_source_grounded_candidate_extraction", "per_fight_calls": False, "unverified_numeric_completion": False, "arbitrary_code_execution": False},
        "main_split": {"train": "15.14", "validation": "15.15", "test": "15.16"},
        "cross_patch_learning_limit": "single training patch cannot identify responses to patch-constant rule variation",
        "multi_patch_training": "separate_future_chronological_protocol_not_silent_reuse_of_current_test",
        "provenance_tuple": ["model_version", "engine_version", "bundle_hash", "feature_contract"],
        "raw_match_access_this_task": False,
    }
    (OUT / "PATCH_CONDITIONED_MODEL_V1.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = {"schema_parse": True, "contract_parse": True}
    references = []

    def walk(x):
        if isinstance(x, dict):
            if "$ref" in x:
                references.append(x["$ref"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(schema)
    checks["local_schema_refs"] = all(r.split("/")[-1] in schema["$defs"] for r in references)
    checks["local_contract_refs"] = all((OUT / contract[k]).exists() for k in ["state_engine_contract", "knowledge_schema", "baseline_features"])
    if importlib.util.find_spec("jsonschema"):
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
        checks["draft202012_meta_schema"] = True
    else:
        checks["draft202012_meta_schema"] = "NOT_RUN_jsonschema_unavailable"
    # One style pass on the complete draft, before saving the final document.
    helpers = {"__name__": "litreview_design_helpers"}
    skill = Path(os.environ["LOL_LITERATURE_KERNEL"])
    exec(compile(skill.read_text(encoding="utf-8"), str(skill), "exec"), helpers)
    draft = (OUT / "patch_knowledge_sources/design_draft.md").read_text(encoding="utf-8")
    lint = helpers["style_pass"](draft)
    (OUT / "PATCH_CONDITIONED_MODEL_V1_KO.md").write_text(draft, encoding="utf-8")
    local_sources = [ROOT / "worktrees/engagement-state-value/config/game_rules/datadragon" / name for name in
                     ["index.json", "champions_15.14.json", "champions_15.15.json", "champions_15.16.json"]]
    report = {"scope": "schema and design checks; not complete patch bundle or predictor validation",
              "checks": checks, "style_pass": lint,
              "local_source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in local_sources},
              "retrieval_limits": "Individual public champion JSON downloads failed DNS; documentation and local reduced files were inspected",
              "new_training": False, "engine_implemented": False, "complete_knowledge_bundle_built": False,
              "artifact_hashes": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in
                                  ["PATCH_KNOWLEDGE_SCHEMA_V1.json", "PATCH_CONDITIONED_MODEL_V1.json", "PATCH_CONDITIONED_MODEL_V1_KO.md"]}}
    (OUT / "PATCH_MODEL_DESIGN_CHECK_V1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "style_pass": lint}, ensure_ascii=False))


if __name__ == "__main__":
    main()
