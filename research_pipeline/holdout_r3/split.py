"""Runtime split helpers backed by an operator-owned local cohort policy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_policy import bucket_for, load_policy, role_for, split_roles, roles_for_macro


def policy():
    return load_policy()


def bucket(match_id: str) -> int:
    return bucket_for(match_id, policy())


def roles(patch: str, value: int) -> tuple[str, str]:
    return role_for(patch, value, policy())


def role_names() -> set[str]:
    return split_roles(policy())


def fit_role_names() -> set[str]:
    return roles_for_macro(policy(), {"FIT"})


def heldout_role_names() -> set[str]:
    config = policy()
    return {
        str(rule["split_role"])
        for rules in config["patch_rules"].values()
        for rule in rules
        if str(rule["macro_role"]) not in {"FIT", "UNUSED", "OTHER"}
    }


def patches() -> tuple[str, ...]:
    return tuple(policy()["patch_rules"])


def roles_by_macro() -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for rules in policy()["patch_rules"].values():
        for rule in rules:
            grouped.setdefault(str(rule["macro_role"]), set()).add(str(rule["split_role"]))
    return {key: sorted(values) for key, values in grouped.items()}
