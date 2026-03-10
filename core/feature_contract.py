"""feature_contract.py

# ═══════════════════════════════════════════════════════════════
# [P2-STRUCT-3] Import DAG Layer: 0 (ZERO external dependencies)
#
# Import hierarchy (acyclic by design):
#   Layer 0: feature_contract.py   ← THIS MODULE (no project imports)
#   Layer 1: config.py             (imports Layer 0 only)
#   Layer 2: features.py           (imports Layer 0, 1)
#   Layer 3: contract.py           (imports Layer 0, 1, 2)
#
# RULE: This module MUST NOT import from config, features, or contract.
# Violation creates a circular dependency → ImportError.
# ═══════════════════════════════════════════════════════════════

Single source of truth for **feature names / indices / dimensions**.

The project previously rebuilt NODE_IDX/EVENT_IDX/GLOBAL_IDX across modules.
That makes it easy to get silent drift (order, missing keys) and later shape bugs.

This module defines :class:`FeatureContract` plus helpers to build and validate it.
The contract is intentionally small and framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple


# ─────────────────────────────────────────────────────────────
# [P4-STATS] Tabular aggregation suffixes — single source of truth.
#
# Previously hardcoded independently in:
#   - baseline.py  _tabular_feature_names_from_base()  line 70
#   - features.py  get_tabular_feature_names_tri_modal() line 706
#   - features.py  seq_to_tabular()  (implicit via concat order)
#
# The order MUST match seq_to_tabular()'s np.concatenate order:
#   [last, mean, std, min, max, delta, slope]
# ─────────────────────────────────────────────────────────────
TABULAR_SUFFIXES: Tuple[str, ...] = ("last", "mean", "std", "min", "max", "delta", "slope")


def tabular_feature_names(base_names: Sequence[str]) -> Tuple[str, ...]:
    """Generate tabular feature names from base × suffixes.

    This is the canonical way to produce tabular column names.
    All modules MUST use this instead of local suffix lists.
    """
    return tuple(f"{n}__{s}" for n in base_names for s in TABULAR_SUFFIXES)


def _as_tuple_str(xs: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(x) for x in xs)


def _validate_unique(names: Sequence[str], what: str) -> None:
    seen = set()
    dups = []
    for n in names:
        if n in seen:
            dups.append(n)
        seen.add(n)
    if dups:
        raise ValueError(f"{what} has duplicate feature names (first few): {dups[:10]}")


@dataclass(frozen=True)
class FeatureContract:
    """Immutable feature contract.

    Attributes
    ----------
    node_names, event_names, global_names:
        Ordered feature names.
    node_idx, event_idx, global_idx:
        Name -> column index maps.
    f_node, f_event, f_global:
        Feature dimensions.
    """

    node_names: Tuple[str, ...]
    event_names: Tuple[str, ...]
    global_names: Tuple[str, ...]

    node_idx: Dict[str, int] = field(init=False)
    event_idx: Dict[str, int] = field(init=False)
    global_idx: Dict[str, int] = field(init=False)

    f_node: int = field(init=False)
    f_event: int = field(init=False)
    f_global: int = field(init=False)

    def __post_init__(self) -> None:
        _validate_unique(self.node_names, "node_names")
        _validate_unique(self.event_names, "event_names")
        _validate_unique(self.global_names, "global_names")

        object.__setattr__(self, "node_idx", {n: i for i, n in enumerate(self.node_names)})
        object.__setattr__(self, "event_idx", {n: i for i, n in enumerate(self.event_names)})
        object.__setattr__(self, "global_idx", {n: i for i, n in enumerate(self.global_names)})

        object.__setattr__(self, "f_node", len(self.node_names))
        object.__setattr__(self, "f_event", len(self.event_names))
        object.__setattr__(self, "f_global", len(self.global_names))

    def require_keys(self, where: str, keys: Iterable[str]) -> None:
        """Raise a clear error if required keys are missing."""
        missing = [k for k in keys if k not in self.node_idx and k not in self.event_idx and k not in self.global_idx]
        if missing:
            raise KeyError(f"{where}: missing keys in feature contract: {missing}")


def build_feature_contract(
    *,
    node_names: Sequence[str],
    event_names: Sequence[str],
    global_names: Sequence[str],
) -> FeatureContract:
    """Build the immutable feature contract from name lists."""
    return FeatureContract(
        node_names=_as_tuple_str(node_names),
        event_names=_as_tuple_str(event_names),
        global_names=_as_tuple_str(global_names),
    )


# ─────────────────────────────────────────────────────────────
# [ABLATION] Static feature identification for temporal
# aggregation validation.
#
# Static attributes (champion_id, rune IDs, summoner spells, bans)
# do not change across timesteps within a single fight sequence.
# Temporal aggregation suffixes (__std, __delta, __slope) applied
# to these features should yield ~0 values. Non-zero values
# indicate data artifacts (champion swap, remake) or noise fitting.
# ─────────────────────────────────────────────────────────────
STATIC_NODE_FEATURE_PREFIXES: Tuple[str, ...] = (
    "champion_id",
    "champion_name_id",
    "summoner_spell_1_id",
    "summoner_spell_2_id",
    "primary_style_id",
    "sub_style_id",
    "primary_rune_1",
    "primary_rune_2",
    "primary_rune_3",
    "primary_rune_4",
    "sub_rune_1",
    "sub_rune_2",
    "stat_perk_offense",
    "stat_perk_flex",
    "stat_perk_defense",
)

STATIC_GLOBAL_FEATURE_PREFIXES: Tuple[str, ...] = (
    "blue_ban_",
    "red_ban_",
)

# Legacy: only the 3 suffixes that produce ~0 values (noise).
TEMPORAL_NOISE_SUFFIXES: Tuple[str, ...] = ("__std", "__delta", "__slope")

# ─────────────────────────────────────────────────────────────
# [FE-CONST] Comprehensive constant/quasi-constant feature
# classification for redundant aggregation removal.
#
# For strictly constant features (fixed at champion select):
#   last  = the constant value                     → KEEP
#   mean  = same as last (identical, redundant)     → DROP
#   std   = 0 (FP noise only)                      → DROP
#   min   = same as last (identical, redundant)     → DROP
#   max   = same as last (identical, redundant)     → DROP
#   delta = 0 (FP noise only)                      → DROP
#   slope = 0 (FP noise only)                      → DROP
#
# For quasi-constant features (extremely unlikely to change
# within the short teamfight observation window):
#   Same reasoning applies — temporal aggregation is meaningless.
#
# Impact:
#   Strictly constant removal: ~16.2% feature reduction
#   + Quasi-constant removal:  ~26.2% total feature reduction
#
# SHAP analysis showed features like bTOP_champion_id__delta (rank 7),
# bJNG_sub_rune_1__std (rank 19), bSUP_stat_perk_offense__slope (rank 18)
# with high importance despite theoretically being 0 — a strong signal
# of overfitting to floating-point noise.
# ─────────────────────────────────────────────────────────────

# All non-last suffixes: these are redundant for any constant feature.
REDUNDANT_SUFFIXES_FOR_CONSTANT: Tuple[str, ...] = (
    "__mean", "__std", "__min", "__max", "__delta", "__slope",
)

# ─── Category B: Quasi-constant features ───────────────────
# Theoretically variable but practically constant within the
# short teamfight observation window (seconds to tens of seconds).

# Node-level quasi-constants (per-player, appear with slot prefix):
# NOTE: Dragon soul was moved to Category C (within-fight constant).
# itemhash: per-player item hash — item purchase requires recall, impossible mid-fight
QUASI_CONSTANT_NODE_FEATURE_PREFIXES: Tuple[str, ...] = (
    "itemhash",
)

# ─── Category C: Within-fight constant (sparse binary signals) ──
# Features that are constant within any single fight observation
# window but carry meaningful cross-fight signal.
#
# Dragon soul is a *sparse binary* — 0 in ~70% of games, 1 in ~30%.
# Once acquired, it never changes mid-fight (like `alive` being mostly
# 1 but critically 0 at death).  The binary state IS the signal.
#
# Non-__last aggregations are mathematically trivial:
#   mean = last (identical for constant), std = 0, delta = 0, slope = 0
# Only __last preserves the actual acquisition signal → KEEP.
WITHIN_FIGHT_CONSTANT_NODE_FEATURE_PREFIXES: Tuple[str, ...] = (
    "soul_infernal",
    "soul_ocean",
    "soul_mountain",
    "soul_cloud",
    "soul_hextech",
    "soul_chemtech",
)

# Non-slotted quasi-constants (global/spatial/item features):
# - itemhash moved to node-level (QUASI_CONSTANT_NODE_FEATURE_PREFIXES)
# - zone_*: fight anchor zone is fixed for the engagement
# - pos_fight_*: fight centroid is anchored to the engagement location
QUASI_CONSTANT_EXTRA_FEATURE_PREFIXES: Tuple[str, ...] = (
    "zone_top_lane",
    "zone_mid_lane",
    "zone_bot_lane",
    "zone_river",
    "zone_jungle",
    "pos_fight_x_norm",
    "pos_fight_y_norm",
)

_SLOT_PREFIXES: Tuple[str, ...] = (
    "bTOP_", "bJNG_", "bMID_", "bBOT_", "bSUP_",
    "rTOP_", "rJNG_", "rMID_", "rBOT_", "rSUP_",
)


def _extract_base_and_suffix(tabular_name: str) -> Tuple[str, str]:
    """Split 'bJNG_primary_rune_3__delta' into ('bJNG_primary_rune_3', '__delta').

    Returns ('', '') if the name has no __ separator.
    """
    idx = tabular_name.rfind("__")
    if idx < 0:
        return "", ""
    return tabular_name[:idx], tabular_name[idx:]


def _strip_slot_prefix(base: str) -> Tuple[str, str]:
    """Strip slot prefix from base name.

    Returns (slot_prefix, attribute_name). If no slot prefix, returns ('', base).
    """
    for slot in _SLOT_PREFIXES:
        if base.startswith(slot):
            return slot, base[len(slot):]
    return "", base


def _is_strictly_constant_base(base: str) -> bool:
    """Check if a base feature name (before __ suffix) refers to a
    strictly constant attribute."""
    _slot, attr = _strip_slot_prefix(base)

    # Slotted node features (per-player)
    if _slot:
        for pfx in STATIC_NODE_FEATURE_PREFIXES:
            if attr == pfx or attr.startswith(pfx):
                return True
        return False

    # Non-slotted global features (bans)
    for pfx in STATIC_GLOBAL_FEATURE_PREFIXES:
        if base.startswith(pfx) or base == pfx.rstrip("_"):
            return True

    return False


def _is_quasi_constant_base(base: str) -> bool:
    """Check if a base feature name refers to a quasi-constant attribute."""
    _slot, attr = _strip_slot_prefix(base)

    # Slotted node features
    if _slot:
        for pfx in QUASI_CONSTANT_NODE_FEATURE_PREFIXES:
            if attr == pfx or attr.startswith(pfx):
                return True
        return False

    # Non-slotted features (items, zones, fight position)
    for pfx in QUASI_CONSTANT_EXTRA_FEATURE_PREFIXES:
        if base == pfx or base.startswith(pfx):
            return True

    return False


def _is_within_fight_constant_base(base: str) -> bool:
    """Check if a base feature name refers to a within-fight constant (sparse binary)."""
    _slot, attr = _strip_slot_prefix(base)

    # Slotted node features (dragon souls)
    if _slot:
        for pfx in WITHIN_FIGHT_CONSTANT_NODE_FEATURE_PREFIXES:
            if attr == pfx or attr.startswith(pfx):
                return True

    return False


def classify_feature_constancy(tabular_name: str) -> str:
    """Classify a tabular feature by its temporal constancy.

    Parameters
    ----------
    tabular_name : str
        Full tabular name like "bJNG_primary_rune_3__delta".

    Returns
    -------
    str
        One of:
        - ``"strictly_constant"`` — fixed at champion select, never changes
        - ``"quasi_constant"`` — extremely unlikely to change mid-fight
        - ``"within_fight_constant"`` — sparse binary, constant within fight window
        - ``"time_varying"`` — genuinely varies across timesteps
    """
    base, suffix = _extract_base_and_suffix(tabular_name)
    if not base:
        return "time_varying"

    if _is_strictly_constant_base(base):
        return "strictly_constant"
    if _is_within_fight_constant_base(base):
        return "within_fight_constant"
    if _is_quasi_constant_base(base):
        return "quasi_constant"
    return "time_varying"


def is_constant_redundant(tabular_name: str) -> bool:
    """Return True if a tabular feature is a redundant aggregation of a
    strictly constant attribute.

    For constant features, only ``__last`` carries useful information.
    All other suffixes (mean, std, min, max, delta, slope) are either
    identical to last (redundant) or ~0 (noise from FP arithmetic).
    """
    base, suffix = _extract_base_and_suffix(tabular_name)
    if not base or suffix not in REDUNDANT_SUFFIXES_FOR_CONSTANT:
        return False
    return _is_strictly_constant_base(base)


def is_quasi_constant_redundant(tabular_name: str) -> bool:
    """Return True if a tabular feature is a redundant aggregation of a
    quasi-constant attribute.

    Quasi-constant features (item hash, fight zone/position)
    are extremely unlikely to change within the teamfight observation
    window. Temporal aggregation adds noise, not signal.
    """
    base, suffix = _extract_base_and_suffix(tabular_name)
    if not base or suffix not in REDUNDANT_SUFFIXES_FOR_CONSTANT:
        return False
    return _is_quasi_constant_base(base)


def is_within_fight_constant_redundant(tabular_name: str) -> bool:
    """Return True if a tabular feature is a redundant aggregation of a
    within-fight constant (sparse binary) attribute.

    Dragon soul features are sparse binary signals (0 in ~70%, 1 in ~30%).
    Within a single fight, the value never changes, so non-__last
    aggregations are mathematically trivial (std=0, delta=0, etc.).
    Only __last carries the meaningful acquisition signal.
    """
    base, suffix = _extract_base_and_suffix(tabular_name)
    if not base or suffix not in REDUNDANT_SUFFIXES_FOR_CONSTANT:
        return False
    return _is_within_fight_constant_base(base)


def is_static_temporal_noise(tabular_name: str) -> bool:
    """Return True if a tabular feature is a temporal-noise aggregation
    of a static attribute (should be ~0 and contributes only noise).

    .. note::
        This is the legacy function that only checks std/delta/slope
        suffixes for strictly constant features. For comprehensive
        filtering (including mean/min/max redundancy and quasi-constant
        features), use :func:`is_constant_redundant` and
        :func:`is_quasi_constant_redundant` instead, or call
        :func:`filter_constant_and_quasi_constant`.

    Parameters
    ----------
    tabular_name : str
        A tabular feature name like "bJNG_primary_rune_3__delta".

    Returns
    -------
    bool
        True if this is a static attribute with a noise suffix.
    """
    base, suffix = _extract_base_and_suffix(tabular_name)
    if not base or suffix not in TEMPORAL_NOISE_SUFFIXES:
        return False
    return _is_strictly_constant_base(base)


def filter_static_temporal_noise(
    feature_names: Sequence[str],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Identify static temporal-noise features in a feature name list.

    .. note::
        Legacy filter (std/delta/slope only). For comprehensive filtering,
        use :func:`filter_constant_and_quasi_constant`.

    Returns
    -------
    keep_indices : tuple of int
        Indices of features to keep (non-noise).
    dropped_names : tuple of str
        Names of dropped noise features.
    """
    keep = []
    dropped = []
    for i, name in enumerate(feature_names):
        if is_static_temporal_noise(name):
            dropped.append(name)
        else:
            keep.append(i)
    return tuple(keep), tuple(dropped)


def filter_constant_and_quasi_constant(
    feature_names: Sequence[str],
    *,
    drop_strictly_constant: bool = True,
    drop_quasi_constant: bool = True,
    drop_within_fight_constant: bool = True,
) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Comprehensive filter removing redundant aggregations of constant,
    quasi-constant, and within-fight-constant features.

    For all categories, only ``__last`` is retained. All other temporal
    aggregation suffixes (mean, std, min, max, delta, slope) are dropped.

    Parameters
    ----------
    feature_names : sequence of str
        Full tabular feature names.
    drop_strictly_constant : bool
        Drop redundant aggregations of strictly constant features
        (champion_id, runes, summoner spells, stat perks, bans).
    drop_quasi_constant : bool
        Drop redundant aggregations of quasi-constant features
        (item hash, fight zone/position).
    drop_within_fight_constant : bool
        Drop redundant aggregations of within-fight constant features
        (dragon soul — sparse binary signals constant within fight window).

    Returns
    -------
    keep_indices : tuple of int
        Indices of features to keep.
    dropped_constant : tuple of str
        Names of dropped strictly-constant redundant features.
    dropped_quasi : tuple of str
        Names of dropped quasi-constant and within-fight-constant
        redundant features (combined for backward compatibility).
    """
    keep = []
    dropped_const = []
    dropped_quasi = []

    for i, name in enumerate(feature_names):
        if drop_strictly_constant and is_constant_redundant(name):
            dropped_const.append(name)
        elif drop_quasi_constant and is_quasi_constant_redundant(name):
            dropped_quasi.append(name)
        elif drop_within_fight_constant and is_within_fight_constant_redundant(name):
            dropped_quasi.append(name)
        else:
            keep.append(i)

    return tuple(keep), tuple(dropped_const), tuple(dropped_quasi)