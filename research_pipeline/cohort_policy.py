"""Load a local cohort split policy without publishing cohort-specific values."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_policy() -> dict[str, Any]:
    """Read the operator-owned policy named by LOL_COHORT_POLICY_PATH."""
    source = os.environ.get("LOL_COHORT_POLICY_PATH")
    if not source:
        raise RuntimeError("Set LOL_COHORT_POLICY_PATH to a local split-policy JSON file")
    policy = json.loads(Path(source).read_text(encoding="utf-8"))
    required = {"salt", "region_prefix", "bucket_hex_chars", "bucket_modulus", "patch_rules"}
    if not isinstance(policy, dict) or not required <= set(policy):
        raise ValueError("cohort policy is missing required fields")
    if not isinstance(policy["salt"], str) or not policy["salt"]:
        raise ValueError("cohort policy salt must be a non-empty string")
    if not isinstance(policy["patch_rules"], dict) or not policy["patch_rules"]:
        raise ValueError("cohort policy must define patch_rules")
    width, modulus = policy["bucket_hex_chars"], policy["bucket_modulus"]
    if not isinstance(width, int) or not 1 <= width <= 64 or not isinstance(modulus, int) or modulus < 1:
        raise ValueError("invalid cohort hash bucket parameters")
    for patch, rules in policy["patch_rules"].items():
        if not isinstance(patch, str) or not isinstance(rules, list) or not rules:
            raise ValueError("each patch must map to a non-empty rule list")
        for rule in rules:
            if not isinstance(rule, dict) or not {"min_bucket", "max_bucket", "split_role", "macro_role"} <= set(rule):
                raise ValueError("each split rule needs bucket bounds and role names")
            if not (0 <= rule["min_bucket"] <= rule["max_bucket"] < modulus):
                raise ValueError("split rule bucket bounds are outside the configured modulus")
    return policy


def canonical_match_key(match_id: str, region_prefix: str) -> str:
    value = str(match_id).strip().upper()
    return value if "_" in value else f"{region_prefix}_{value}"


def bucket_for(match_id: str, policy: dict[str, Any]) -> int:
    key = canonical_match_key(match_id, policy["region_prefix"])
    digest = hashlib.sha256((policy["salt"] + key).encode("utf-8")).hexdigest()
    return int(digest[: policy["bucket_hex_chars"]], 16) % policy["bucket_modulus"]


def role_for(patch: str, bucket: int, policy: dict[str, Any]) -> tuple[str, str]:
    for rule in policy["patch_rules"].get(str(patch), []):
        if rule["min_bucket"] <= bucket <= rule["max_bucket"]:
            return str(rule["split_role"]), str(rule["macro_role"])
    return "OTHER", "OTHER"


def split_roles(policy: dict[str, Any]) -> set[str]:
    return {str(rule["split_role"]) for rules in policy["patch_rules"].values() for rule in rules}


def roles_for_macro(policy: dict[str, Any], macro_roles: set[str]) -> set[str]:
    return {
        str(rule["split_role"])
        for rules in policy["patch_rules"].values()
        for rule in rules
        if str(rule["macro_role"]) in macro_roles
    }
