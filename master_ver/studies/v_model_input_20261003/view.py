"""Explicit focal-view and participant permutations of unfitted V inputs."""
from __future__ import annotations

import copy
import hashlib
from numbers import Integral
from typing import Mapping

import numpy as np

from studies.v_input_projection_20261002.schema import QUERY_BLOCK_SPECS, STATIC_BLOCK_SPECS

VERSION = "v_focal_view_20261003_v1"
SWAP_SALT = "full_study_20261001_v_swap_v2:"
TEAM_BLOCKS = ("team_growth", "cached_gold", "team_kills", "bans")
NODE_BLOCKS = ("node_frame", "player_kda", "static_ids", "team_ids")
ITEM_KEYS = frozenset(("item_ids", "counts", "offsets", "recorded_ledger_supported",
                       "reason_mask", "unassigned_observed", "whole_inventory_verified"))
FEATURE_KEYS = frozenset((*QUERY_BLOCK_SPECS, *STATIC_BLOCK_SPECS, "items"))


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return copy.deepcopy(value)


def replacement_swap(canonical_match_key: str, query_ms: int, seed: int) -> bool:
    """The key controls augmentation only and is never returned as a feature."""
    if not isinstance(canonical_match_key, str) or not canonical_match_key:
        raise ValueError("canonical match key must be a nonempty string")
    for value in (query_ms, seed):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("query_ms and seed must be nonnegative integers")
    raw = f"{SWAP_SALT}{canonical_match_key}|{int(query_ms)}|{int(seed)}".encode("utf-8")
    return bool(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2)


def _validate_history(history: Mapping, *, items: bool = False) -> None:
    expected = {"node", "unassigned"} if items else {"version", "node", "team", "global"}
    if set(history) != expected or len(history["node"]) != 10:
        raise ValueError("wrong typed history fields or participant count")
    if not items and len(history["team"]) != 2:
        raise ValueError("history must have exactly two observed-team lanes")


def _validate_features(features: Mapping) -> None:
    if set(features) != FEATURE_KEYS:
        raise ValueError("features must contain exactly the P1 numeric/static blocks and current items")
    for name, (_, shape) in (*QUERY_BLOCK_SPECS.items(), *STATIC_BLOCK_SPECS.items()):
        if np.shape(features[name]) != shape:
            raise ValueError(f"wrong shape for {name}")
    teams = np.asarray(features["team_ids"])
    if not np.all(np.isin(teams, (100, 200))) or np.sum(teams == 100) != 5:
        raise ValueError("team routing requires five players on each recorded team")
    items = features["items"]
    if set(items) != ITEM_KEYS or items["whole_inventory_verified"] is not False:
        raise ValueError("current items must retain exactly the accepted observed-ledger fields")
    ids, counts, offsets = (np.asarray(items[key]) for key in ("item_ids", "counts", "offsets"))
    if ids.ndim != 1 or counts.shape != ids.shape or offsets.shape != (11,):
        raise ValueError("invalid current-item ragged shapes")
    if not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid current-item offsets")
    for key in ("recorded_ledger_supported", "reason_mask"):
        if np.shape(items[key]) != (10,):
            raise ValueError(f"current items {key} must have ten participant entries")


def make_view(features: Mapping, nonitem_history: Mapping, *, item_history: Mapping | None = None) -> dict:
    """Build an isolated raw view; targets/audit data are not accepted here.

    Optional item contexts are passed through unchanged from the accepted v3
    reducer. Their absence explicitly means this is not a complete model input.
    """
    _validate_features(features)
    _validate_history(nonitem_history)
    if item_history is not None:
        _validate_history(item_history, items=True)
    return {"version": VERSION, "features": _plain(features),
            "nonitem_history": _plain(nonitem_history),
            "item_history": _plain(item_history), "focal_is_blue": True}


def _validate_view(view: Mapping) -> None:
    if set(view) != {"version", "features", "nonitem_history", "item_history", "focal_is_blue"} or view["version"] != VERSION:
        raise ValueError("wrong raw view schema")
    if not isinstance(view["focal_is_blue"], bool):
        raise ValueError("focal_is_blue must be boolean")
    _validate_features(view["features"])
    _validate_history(view["nonitem_history"])
    if view["item_history"] is not None:
        _validate_history(view["item_history"], items=True)


def swap_view(view: Mapping) -> dict:
    """Swap focal/opponent interpretation; participant ownership stays attached."""
    _validate_view(view)
    result = copy.deepcopy(dict(view))
    features = result["features"]
    for name in TEAM_BLOCKS:
        features[name] = np.asarray(features[name])[[1, 0]].copy()
    features["team_growth_delta"] = -np.asarray(features["team_growth_delta"])
    features["team_ids"] = 300 - np.asarray(features["team_ids"])
    result["nonitem_history"]["team"] = list(reversed(result["nonitem_history"]["team"]))
    result["focal_is_blue"] = not result["focal_is_blue"]
    return result


def swap_target(target: bool | int) -> int:
    if not isinstance(target, (bool, Integral)) or int(target) not in (0, 1):
        raise ValueError("target must be binary")
    return 1 - int(target)


def permute_players(view: Mapping, order) -> dict:
    """Reorder all participant-attached fields jointly, retaining team routing."""
    _validate_view(view)
    permutation = np.asarray(order)
    if permutation.shape != (10,) or not np.issubdtype(permutation.dtype, np.integer) or not np.array_equal(np.sort(permutation), np.arange(10)):
        raise ValueError("order must be a permutation of indices 0..9")
    result = copy.deepcopy(dict(view))
    features = result["features"]
    for name in NODE_BLOCKS:
        features[name] = np.asarray(features[name])[permutation].copy()
    source = view["features"]["items"]
    target = features["items"]
    offsets = np.asarray(source["offsets"])
    slices = [slice(int(offsets[index]), int(offsets[index + 1])) for index in permutation]
    for name in ("item_ids", "counts"):
        target[name] = np.concatenate([np.asarray(source[name])[part] for part in slices])
    lengths = np.diff(offsets)[permutation]
    target["offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(offsets.dtype)
    for name in ("recorded_ledger_supported", "reason_mask"):
        target[name] = np.asarray(source[name])[permutation].copy()
    result["nonitem_history"]["node"] = [copy.deepcopy(view["nonitem_history"]["node"][index]) for index in permutation]
    if result["item_history"] is not None:
        result["item_history"]["node"] = [copy.deepcopy(view["item_history"]["node"][index]) for index in permutation]
    return result
