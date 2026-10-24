"""Fail-closed boundary for future V-model training.

This module is intentionally separate from the accepted P2b adapters.  It
does not fit preprocessing or a model; it only proves that a future caller
has supplied an explicit, hash-bound full-corpus acceptance and a strictly
indexed common batch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from studies.v_model_fit_input_20261003.common import (
    BAG_LAYOUT,
    BAG_NAMESPACES,
    CATEGORY_LAYOUT,
    CATEGORY_NAMESPACES,
)
from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch, build_model, family_cards
from studies.v_model_fit_input_20261003.preprocess import FitScopeEvidence, VERSION as FIT_EVIDENCE_VERSION
from studies.v_preparation_20261002.partition import assigned_role, context_membership, train_fold


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FIXED_SEEDS = (20261001, 20261002, 20261003)
ALL_MEMBERSHIP = frozenset(("fit", "stop", "refit", "dev", "cal", "target"))
STRUCTURED_FAMILIES = frozenset(("shared_slot_mlp", "team_set_mlp"))
STRUCTURED_DIMENSION_KEYS = frozenset((
    "node_numeric_dim", "team_numeric_dim", "global_numeric_dim",
    "vocab_sizes",
))
CANONICAL_VOCAB_NAMESPACES = frozenset((*CATEGORY_NAMESPACES.values(), *BAG_NAMESPACES.values()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_evidence_path(raw: str, acceptance: Path) -> Path | None:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    roots = (acceptance.parent, acceptance.parent.parent, acceptance.parent.parent.parent, Path.cwd())
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None


def _hash_evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("hash_evidence", "evidence", "hashes", "artifact_hashes"):
        value = record.get(name)
        if isinstance(value, Mapping) and value:
            return value
    raise ValueError("training acceptance needs nonempty hash evidence")


def require_training_ready(acceptance_path: str | Path) -> dict[str, Any]:
    """Load only a current, full, non-smoke acceptance record."""

    path = Path(acceptance_path)
    if not path.is_file():
        raise ValueError(f"training acceptance does not exist: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training acceptance is not readable JSON") from exc
    if not isinstance(record, Mapping):
        raise ValueError("training acceptance must be a JSON object")
    required = (
        "P2b_complete", "training_ready", "full_corpus_preprocessors_fitted",
        "full_corpus_model_consumption_verified", "full_batch_resource_verified",
        "scientific_runner_verified",
    )
    missing = [name for name in required if record.get(name) is not True]
    if missing:
        raise ValueError(f"training acceptance is not ready: {', '.join(missing)}")
    if record.get("smoke") is True or record.get("accepted_as_smoke") is True or str(record.get("status", "")).lower() == "smoke":
        raise ValueError("smoke acceptance cannot authorize training")
    evidence = _hash_evidence(record)
    for raw_name, expected in evidence.items():
        if not isinstance(raw_name, str) or not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            raise ValueError("hash evidence must map paths to lowercase SHA256 digests")
        if "smoke" in Path(raw_name).name.lower():
            raise ValueError("smoke artifact cannot authorize training")
        target = _resolve_evidence_path(raw_name, path)
        if target is None or _sha256(target) != expected:
            raise ValueError(f"stale or missing hash evidence: {raw_name}")
    return dict(record)


def _strict_indices(values: Sequence[Any], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer indices")
    result = tuple(values)
    if not result or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    result = tuple(int(value) for value in result)
    if any(value < 0 for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonnegative integers")
    return result


def _strict_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _evidence(value: FitScopeEvidence | Mapping[str, Any]) -> FitScopeEvidence:
    if isinstance(value, FitScopeEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("evidence must be FitScopeEvidence or a mapping")
    raw_membership = value.get("membership")
    if not isinstance(raw_membership, Mapping):
        raise ValueError("evidence membership must be a mapping")
    rows = _strict_indices(value.get("row_indices", ()), "evidence.row_indices")
    keys = tuple(value.get("game_keys", ()))
    roles = tuple(value.get("roles", ()))
    membership = []
    for name, flags in raw_membership.items():
        if not isinstance(name, str) or not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
            raise ValueError("evidence membership flags are malformed")
        flags = tuple(flags)
        if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
            raise ValueError("evidence membership flags must be booleans")
        membership.append((name, flags))
    version = value.get("version")
    if version != FIT_EVIDENCE_VERSION:
        raise ValueError("unsupported fit evidence version")
    return FitScopeEvidence(
        context=value.get("context"), scope=value.get("scope"), row_indices=rows,
        game_keys=keys, roles=roles, membership=tuple(sorted(membership)),
        excluded_game_keys=tuple(value.get("excluded_game_keys", ())),
        source_hash=value.get("source_hash"), version=version,
    )


def _expected_source(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in result):
        raise ValueError(f"{name} must contain canonical SHA256 game keys")
    return result


def validate_scope_evidence(
    evidence: FitScopeEvidence | Mapping[str, Any],
    expected_game_keys: Sequence[str] | None = None,
    expected_roles: Sequence[str] | None = None,
    *,
    expected_indices: Sequence[int] | None = None,
    expected_source_hash: str | None = None,
) -> FitScopeEvidence:
    """Bind fit evidence to trusted selected rows and recomputed partitions."""

    if expected_indices is None or expected_game_keys is None or expected_roles is None or expected_source_hash is None:
        raise ValueError("trusted expected indices, game keys, roles, and source hash are required")
    checked = _evidence(evidence)
    indices = _strict_indices(expected_indices, "expected_indices")
    keys = _expected_source(expected_game_keys, "expected_game_keys")
    roles = tuple(expected_roles)
    if any(not isinstance(role, str) or role not in {"TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST"} for role in roles):
        raise ValueError("expected_roles must use the canonical partition roles")
    if not len(indices) == len(keys) == len(roles) or checked.row_indices != indices or checked.game_keys != keys or checked.roles != roles:
        raise ValueError("fit evidence does not exactly match trusted selected rows")
    expected_hash = _strict_hash(expected_source_hash, "expected_source_hash")
    if checked.source_hash != expected_hash:
        raise ValueError("fit evidence source hash differs from trusted source")
    if _strict_hash(checked.source_hash, "evidence.source_hash") != expected_hash:
        raise ValueError("fit evidence source hash is missing or malformed")
    if {name for name, _ in checked.membership} != ALL_MEMBERSHIP:
        raise ValueError("all six protocol membership flags are required")
    flags_by_name = dict(checked.membership)
    if any(len(flags) != len(indices) or any(not isinstance(flag, (bool, np.bool_)) for flag in flags) for flags in flags_by_name.values()):
        raise ValueError("membership flags must align with selected rows")
    context = checked.context
    if not isinstance(context, str) or context not in ("final", *(f"oof{i}" for i in range(5))):
        raise ValueError("unsupported fit evidence context")
    fold = None if context == "final" else int(context[-1])
    for index, (key, role) in enumerate(zip(keys, roles)):
        patch = "15.14" if role == "TRAIN" else "15.16" if role == "TEST" else "15.15"
        if assigned_role(patch, key) != role:
            raise ValueError("source role differs from canonical partition")
        expected_flags = context_membership(patch, key, fold)
        if any(bool(flags_by_name[name][index]) != expected_flags[name] for name in ALL_MEMBERSHIP):
            raise ValueError("source membership differs from canonical whole-game partition")
    if context == "final":
        if checked.excluded_game_keys:
            raise ValueError("final context cannot declare OOF exclusions")
    else:
        if not checked.excluded_game_keys or len(set(checked.excluded_game_keys)) != len(checked.excluded_game_keys):
            raise ValueError("OOF context requires explicit own-fold exclusions")
        if any(not isinstance(key, str) or SHA256_RE.fullmatch(key) is None or
               assigned_role("15.14", key) != "TRAIN" or train_fold(key) != fold or key in keys
               for key in checked.excluded_game_keys):
            raise ValueError("OOF excluded game keys do not match recomputed own fold")
    if checked.scope == "FIT" and any(flags_by_name["stop"]):
        raise ValueError("FIT preprocessing cannot consume STOP rows")
    if checked.scope == "REFIT_TRAIN" and not all(flags_by_name["refit"]):
        raise ValueError("REFIT_TRAIN must contain only available TRAIN rows")
    return checked


def select_locked_card(family: str, card_id: str) -> dict[str, Any]:
    if not isinstance(family, str) or not isinstance(card_id, str):
        raise TypeError("family and card_id must be strings")
    key = family.lower()
    if key not in {"logistic", "lightgbm", "plain_mlp", *STRUCTURED_FAMILIES}:
        raise ValueError(f"unknown V model family: {family!r}")
    for card in family_cards(key):
        if card["id"] == card_id:
            return copy.deepcopy(card)
    raise ValueError(f"unknown canonical {key} card: {card_id!r}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_locked_model(
    family: str,
    card_id: str,
    *,
    acceptance_path: str | Path,
    seed: int,
    dimensions: Mapping[str, Any] | int | None = None,
    cpu_threads: int = 1,
) -> Any:
    """Construct an unfit canonical model only after the future gate passes."""

    require_training_ready(acceptance_path)
    card = select_locked_card(family, card_id)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) not in FIXED_SEEDS:
        raise ValueError(f"seed must be one of {FIXED_SEEDS}")
    threads = _positive_int(cpu_threads, "cpu_threads")
    if threads > 3:
        raise ValueError("cpu_threads must be in 1..3")
    key = family.lower()
    kwargs: dict[str, Any] = {}
    if key in STRUCTURED_FAMILIES:
        if not isinstance(dimensions, Mapping):
            raise TypeError("structured models require a dimensions mapping")
        unknown = set(dimensions) - STRUCTURED_DIMENSION_KEYS
        if unknown:
            raise ValueError(f"unlocked structured model options: {sorted(unknown)}")
        if "vocab_sizes" not in dimensions or not isinstance(dimensions["vocab_sizes"], Mapping):
            raise ValueError("structured models require canonical vocab_sizes")
        if set(dimensions["vocab_sizes"]) != CANONICAL_VOCAB_NAMESPACES:
            raise ValueError("structured vocab namespaces must match the common root contract")
        kwargs.update(
            node_numeric_dim=102,
            team_numeric_dim=8,
            global_numeric_dim=76,
            vocab_sizes=dict(dimensions["vocab_sizes"]),
            category_namespaces=dict(CATEGORY_NAMESPACES),
            bag_namespaces=dict(BAG_NAMESPACES),
            category_layout=dict(CATEGORY_LAYOUT),
            bag_layout=dict(BAG_LAYOUT),
            include_side_flag=False,
        )
        for name, expected in (("node_numeric_dim", 102), ("team_numeric_dim", 8), ("global_numeric_dim", 76)):
            if name in dimensions and _positive_int(dimensions[name], name) != expected:
                raise ValueError(f"{name} must use the canonical common width {expected}")
        for namespace, spec in kwargs["vocab_sizes"].items():
            if isinstance(spec, (bool, np.bool_)) or not isinstance(spec, (int, np.integer)) or int(spec) < 2:
                raise ValueError(f"structured vocab size for {namespace!r} must be a plain integer >= 2")
    elif key == "plain_mlp":
        if dimensions is not None and (isinstance(dimensions, Mapping) or isinstance(dimensions, (bool, np.bool_)) or not isinstance(dimensions, (int, np.integer))):
            raise TypeError("plain_mlp dimensions must be one integer input width")
        if dimensions is not None:
            kwargs["input_dim"] = _positive_int(dimensions, "input_dim")
    elif dimensions is not None:
        raise ValueError("table models do not accept dimensions")
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(int(seed))
    if key == "logistic":
        kwargs["random_state"] = int(seed)
    elif key == "lightgbm":
        kwargs.update(random_state=int(seed), num_threads=threads)
    return build_model(key, card=card, **kwargs)


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} is not an array") from exc
    return result


def _numeric_array(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.dtype.kind not in "iuf" or result.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite numeric array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric array")
    return result


def _integer_array(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.dtype.kind not in "iu" or result.dtype.kind == "b":
        raise ValueError(f"{name} must use an integer dtype before model casting")
    return result


def validate_raw_item_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError("raw item IDs must be positive integers")
    return int(value)


def _vocab_size(value: Any) -> int:
    if isinstance(value, Mapping):
        if "num_embeddings" in value or "vocab_size" in value:
            return _positive_int(value.get("num_embeddings", value.get("vocab_size")), "vocab size")
        if "size" in value:
            return _positive_int(value["size"], "vocab size")
    if hasattr(value, "num_embeddings"):
        return _positive_int(value.num_embeddings, "vocab size")
    if hasattr(value, "categories"):
        return len(value.categories) + 2
    if hasattr(value, "size"):
        return _positive_int(value.size, "vocab size") + 2
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return _positive_int(value, "vocab size")
    raise TypeError("vocab_sizes must contain integer sizes or vocabulary objects")


def _counts(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.dtype.kind not in "iuf" or result.dtype.kind == "b" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite numeric counts")
    if np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain integral nonnegative counts")
    return result


def _validate_bag(bag: Any, *, batch: int, slots: int, namespace: str, vocab_sizes: Mapping[str, Any], name: str) -> None:
    if not isinstance(bag, BagInput):
        raise TypeError(f"{name} must be the root BagInput object")
    ids = _integer_array(bag.indices, f"{name}.indices")
    counts = _counts(bag.counts, f"{name}.counts")
    if ids.ndim != 1 or counts.shape != ids.shape:
        raise ValueError(f"{name} indices/counts must be aligned flat arrays")
    if namespace not in vocab_sizes:
        raise ValueError(f"missing vocabulary namespace {namespace!r}")
    size = _vocab_size(vocab_sizes[namespace])
    if np.any(ids < 0) or np.any(ids >= size):
        raise ValueError(f"{name}.indices exceed namespace bounds")
    offsets = _integer_array(bag.offsets, f"{name}.offsets")
    if offsets.shape != (batch, slots + 1) or int(offsets[0, 0]) != 0:
        raise ValueError(f"{name}.offsets must be canonical [batch,slots+1] coverage")
    if np.any(offsets < 0) or np.any(np.diff(offsets, axis=1) < 0) or (batch > 1 and np.any(offsets[1:, 0] != offsets[:-1, -1])) or int(offsets[-1, -1]) != len(ids):
        raise ValueError(f"{name}.offsets must cover flattened bags contiguously")
    observed = _counts(bag.observed_count, f"{name}.observed_count")
    if observed.shape != (batch, slots):
        raise ValueError(f"{name}.observed_count must match bag slots")
    totals = np.zeros((batch, slots), dtype=np.float64)
    for row in range(batch):
        for slot in range(slots):
            start, stop = int(offsets[row, slot]), int(offsets[row, slot + 1])
            local_ids, local_counts = ids[start:stop], counts[start:stop]
            if np.any((local_ids == 0) & (local_counts > 0)):
                raise ValueError(f"{name} cannot assign a nonzero count to PAD ID 0")
            if np.any((local_ids != 0) & (local_counts <= 0)):
                raise ValueError(f"{name} has a nonpadding ID with nonpositive count")
            totals[row, slot] = float(local_counts.sum())
    if not np.allclose(observed, totals, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name}.observed_count disagrees with bag counts")


def validate_indexed_batch(
    batch: StructuredBatch,
    vocab_sizes: Mapping[str, Any],
    *,
    raw_item_ids: Sequence[Any] | None = None,
) -> StructuredBatch:
    """Validate the exact common ``StructuredBatch`` index contract."""

    if not isinstance(batch, StructuredBatch) or not isinstance(vocab_sizes, Mapping):
        raise TypeError("validate_indexed_batch requires StructuredBatch and vocab_sizes mapping")
    if batch.category_namespaces != CATEGORY_NAMESPACES or batch.bag_namespaces != BAG_NAMESPACES:
        raise ValueError("batch namespace declarations must match the common root contract")
    node = _numeric_array(batch.node_numeric, "node_numeric")
    team = _numeric_array(batch.team_numeric, "team_numeric")
    global_values = _numeric_array(batch.global_numeric, "global_numeric")
    if node.ndim != 3 or tuple(node.shape[1:]) != (10, 102) or team.ndim != 3 or tuple(team.shape[1:]) != (2, 8) or global_values.ndim != 2 or global_values.shape[1] != 76:
        raise ValueError("numeric fields do not match the common StructuredBatch shapes")
    size = int(node.shape[0])
    if size <= 0 or size > 1024:
        raise ValueError("indexed batch must contain 1..1024 rows")
    if team.shape[0] != size or global_values.shape[0] != size:
        raise ValueError("numeric fields disagree on batch size")
    if not np.all(np.isin(global_values[:, -1], (0, 1))):
        raise ValueError("global side flag must be 0 or 1")
    teams = _integer_array(batch.player_team_ids, "player_team_ids")
    if teams.shape != (size, 10) or not np.all(np.isin(teams, (100, 200))) or not np.all(np.sum(teams == 100, axis=1) == 5):
        raise ValueError("actual team routing requires integer 100/200 IDs and five players per team")
    support = _array(batch.current_item_supported, "current_item_supported")
    if support.dtype.kind != "b" or support.shape != (size, 10):
        raise ValueError("current_item_supported must be a boolean [batch,10] array")
    for lane, groups in CATEGORY_LAYOUT.items():
        actual = getattr(batch, f"{lane}_categories")
        if set(actual) != set(groups):
            raise ValueError(f"{lane} category keys must exactly match the common root contract")
        slots = {"node": 10, "team": 2, "global": 1}[lane]
        for group in groups:
            values = _integer_array(actual[group], f"{lane}_categories[{group!r}]")
            if values.shape != (size, slots):
                raise ValueError(f"{lane} category group {group!r} has wrong shape")
            namespace = CATEGORY_NAMESPACES[group]
            if namespace not in vocab_sizes or np.any(values < 0) or np.any(values >= _vocab_size(vocab_sizes[namespace])):
                raise ValueError(f"{lane} category group {group!r} exceeds namespace bounds")
    for lane, groups in BAG_LAYOUT.items():
        actual = getattr(batch, f"{lane}_bags")
        if set(actual) != set(groups):
            raise ValueError(f"{lane} bag keys must exactly match the common root contract")
        slots = {"node": 10, "team": 2, "global": 1}[lane]
        for group in groups:
            _validate_bag(actual[group], batch=size, slots=slots, namespace=BAG_NAMESPACES[group], vocab_sizes=vocab_sizes, name=f"{lane}_bags[{group!r}]")
    current = batch.node_bags["current_item"]
    totals = _counts(current.observed_count, "node_bags[current_item].observed_count")
    if np.any(totals[~support] != 0):
        raise ValueError("unsupported current-item ledgers must have zero counts")
    if raw_item_ids is not None:
        for value in raw_item_ids:
            validate_raw_item_id(value)
    return batch


__all__ = [
    "FIXED_SEEDS", "build_locked_model", "require_training_ready", "select_locked_card",
    "validate_indexed_batch", "validate_raw_item_id", "validate_scope_evidence",
]
