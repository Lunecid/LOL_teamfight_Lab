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