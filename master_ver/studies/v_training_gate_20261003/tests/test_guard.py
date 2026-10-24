from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from studies.v_model_fit_input_20261003.common import BAG_LAYOUT, BAG_NAMESPACES, CATEGORY_LAYOUT, CATEGORY_NAMESPACES
from studies.v_model_fit_input_20261003.models import BagInput, StructuredBatch
from studies.v_model_fit_input_20261003.preprocess import FitScopeEvidence
from studies.v_preparation_20261002.partition import context_membership
from studies.v_training_gate_20261003.guard import (
    ALL_MEMBERSHIP,
    FIXED_SEEDS,
    STRUCTURED_DIMENSION_KEYS,
    build_locked_model,
    require_training_ready,
    select_locked_card,
    validate_indexed_batch,
    validate_raw_item_id,
    validate_scope_evidence,
)


ROOT = Path(__file__).resolve().parents[3]
CURRENT_ACCEPTANCE = ROOT / "reports" / "v_model_fit_input_20261003" / "ACCEPTANCE.json"


def _scope(key: str, *, context: str = "final", scope: str = "FIT", excluded=()):
    patch = "15.14"
    fold = None if context == "final" else int(context[-1])
    flags = context_membership(patch, key, fold)
    return FitScopeEvidence(
        context=context, scope=scope, row_indices=(17,), game_keys=(key,), roles=("TRAIN",),
        membership=tuple(sorted((name, (flags[name],)) for name in ALL_MEMBERSHIP)),
        excluded_game_keys=tuple(excluded), source_hash="c" * 64,
    )


def _vocab_sizes():
    return {name: 8 for name in {**CATEGORY_NAMESPACES, **BAG_NAMESPACES}.values()}


def _empty_bag(slots: int) -> BagInput:
    return BagInput(
        indices=np.empty(0, dtype=np.int64), counts=np.empty(0, dtype=np.float32),
        offsets=np.zeros((1, slots + 1), dtype=np.int64), observed_count=np.zeros((1, slots), dtype=np.float32),
    )


def _batch() -> StructuredBatch:
    node_bags = {name: _empty_bag(10) for name in BAG_LAYOUT["node"]}
    current = BagInput(
        indices=np.asarray([2], dtype=np.int64), counts=np.asarray([1.0], dtype=np.float32),
        offsets=np.asarray([[0, 1] + [1] * 9], dtype=np.int64),
        observed_count=np.asarray([[1.0] + [0.0] * 9], dtype=np.float32),
    )
    node_bags["current_item"] = current
    return StructuredBatch(
        node_numeric=np.zeros((1, 10, 102), dtype=np.float32),
        team_numeric=np.zeros((1, 2, 8), dtype=np.float32),
        global_numeric=np.concatenate((np.zeros((1, 75), dtype=np.float32), np.ones((1, 1), dtype=np.float32)), axis=1),
        player_team_ids=np.asarray([[100] * 5 + [200] * 5], dtype=np.int64),
        current_item_supported=np.ones((1, 10), dtype=bool),
        node_categories={name: np.zeros((1, 10), dtype=np.int64) for name in CATEGORY_LAYOUT["node"]},
        team_categories={name: np.zeros((1, 2), dtype=np.int64) for name in CATEGORY_LAYOUT["team"]},
        global_categories={}, node_bags=node_bags,
        team_bags={name: _empty_bag(2) for name in BAG_LAYOUT["team"]},
        global_bags={name: _empty_bag(1) for name in BAG_LAYOUT["global"]},
        category_namespaces=dict(CATEGORY_NAMESPACES), bag_namespaces=dict(BAG_NAMESPACES),
        table_csr=np.zeros((1, 1), dtype=np.float32),
    )


def test_current_acceptance_is_rejected_before_model_or_training():
    with pytest.raises(ValueError, match="P2b_complete"):
        require_training_ready(CURRENT_ACCEPTANCE)
    with pytest.raises(ValueError):
        build_locked_model("logistic", "lr_C0.001", acceptance_path=CURRENT_ACCEPTANCE, seed=FIXED_SEEDS[0])


def test_future_readiness_requires_current_hashes_and_no_smoke(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("bound", encoding="utf-8")
    acceptance = tmp_path / "acceptance.json"
    payload = {name: True for name in (
        "P2b_complete", "training_ready", "full_corpus_preprocessors_fitted",
        "full_corpus_model_consumption_verified", "full_batch_resource_verified", "scientific_runner_verified",
    )}
    payload["evidence"] = {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()}
    acceptance.write_text(json.dumps(payload), encoding="utf-8")
    assert require_training_ready(acceptance)["training_ready"] is True
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        require_training_ready(acceptance)
    payload["evidence"] = {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()}
    payload["smoke"] = True
    acceptance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke"):
        require_training_ready(acceptance)


def test_scope_requires_trusted_exact_indices_keys_roles_and_hash():
    evidence = _scope("a" * 64)
    assert validate_scope_evidence(
        evidence, ["a" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="c" * 64
    ) is evidence
    with pytest.raises(ValueError, match="exactly match"):
        validate_scope_evidence(evidence, ["b" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="c" * 64)
    with pytest.raises(ValueError, match="exactly match"):
        validate_scope_evidence(evidence, ["a" * 64], ["TRAIN"], expected_indices=[18], expected_source_hash="c" * 64)
    with pytest.raises(ValueError, match="source hash"):
        validate_scope_evidence(evidence, ["a" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="d" * 64)


def test_scope_recomputes_own_oof_exclusion_and_stop_rules():
    oof = _scope("a" * 64, context="oof0", excluded=("f" * 64,))
    assert validate_scope_evidence(
        oof, ["a" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="c" * 64
    ) is oof
    with pytest.raises(ValueError, match="own fold"):
        validate_scope_evidence(
            replace(oof, excluded_game_keys=("b" * 64,)), ["a" * 64], ["TRAIN"],
            expected_indices=[17], expected_source_hash="c" * 64,
        )
    stop = _scope("5" * 64, scope="REFIT_TRAIN")
    assert validate_scope_evidence(
        stop, ["5" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="c" * 64
    ) is stop
    object.__setattr__(stop, "scope", "FIT")
    with pytest.raises(ValueError, match="STOP"):
        validate_scope_evidence(
            stop, ["5" * 64], ["TRAIN"], expected_indices=[17], expected_source_hash="c" * 64
        )


def test_select_locked_card_rejects_arbitrary_card_mapping_and_uses_exact_id():
    card = select_locked_card("plain_mlp", "plain_128_64_wd0.001")
    assert card["hidden"] == (128, 64)
    with pytest.raises(TypeError):
        select_locked_card("plain_mlp", {"id": "plain_128_64_wd0.001"})
    with pytest.raises(ValueError):
        select_locked_card("plain_mlp", "PLAIN_128_64_WD0.001")


def test_indexed_batch_validates_common_root_contract():
    batch = _batch()
    assert validate_indexed_batch(batch, _vocab_sizes()) is batch
    with pytest.raises(ValueError, match="integer dtype"):
        validate_indexed_batch(replace(batch, node_categories={**batch.node_categories, "champion_id": np.zeros((1, 10), dtype=np.float32)}), _vocab_sizes())
    bad_bag = replace(batch.node_bags["current_item"], indices=np.asarray([2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="integer dtype"):
        validate_indexed_batch(replace(batch, node_bags={**batch.node_bags, "current_item": bad_bag}), _vocab_sizes())
    bad_offsets = replace(batch.node_bags["current_item"], offsets=batch.node_bags["current_item"].offsets.astype(np.float32))
    with pytest.raises(ValueError, match="integer dtype"):
        validate_indexed_batch(replace(batch, node_bags={**batch.node_bags, "current_item": bad_offsets}), _vocab_sizes())


def test_indexed_batch_rejects_fractional_counts_total_mismatch_team_cast_and_missing_keys():
    batch = _batch()
    bag = replace(batch.node_bags["current_item"], counts=np.asarray([0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="integral"):
        validate_indexed_batch(replace(batch, node_bags={**batch.node_bags, "current_item": bag}), _vocab_sizes())
    bag = replace(batch.node_bags["current_item"], observed_count=np.asarray([[2.0] + [0.0] * 9], dtype=np.float32))
    with pytest.raises(ValueError, match="disagrees"):
        validate_indexed_batch(replace(batch, node_bags={**batch.node_bags, "current_item": bag}), _vocab_sizes())
    with pytest.raises(ValueError, match="integer dtype"):
        validate_indexed_batch(replace(batch, player_team_ids=batch.player_team_ids.astype(np.float32)), _vocab_sizes())
    missing = dict(batch.node_categories)
    missing.pop("champion_id")
    with pytest.raises(ValueError, match="exactly match"):
        validate_indexed_batch(replace(batch, node_categories=missing), _vocab_sizes())


def test_indexed_batch_rejects_pad_counts_empty_rows_and_nonfinite_numeric_values():
    batch = _batch()
    pad = replace(batch.node_bags["current_item"], indices=np.asarray([0], dtype=np.int64))
    with pytest.raises(ValueError, match="PAD"):
        validate_indexed_batch(replace(batch, node_bags={**batch.node_bags, "current_item": pad}), _vocab_sizes())
    empty = replace(batch, node_numeric=np.zeros((0, 10, 102), dtype=np.float32),
                    team_numeric=np.zeros((0, 2, 8), dtype=np.float32),
                    global_numeric=np.zeros((0, 76), dtype=np.float32),
                    player_team_ids=np.empty((0, 10), dtype=np.int64),
                    current_item_supported=np.empty((0, 10), dtype=bool))
    with pytest.raises(ValueError, match="1..1024"):
        validate_indexed_batch(empty, _vocab_sizes())
    nonfinite = replace(batch, global_numeric=np.concatenate((np.zeros((1, 75), dtype=np.float32), np.asarray([[np.nan]], dtype=np.float32)), axis=1))
    with pytest.raises(ValueError, match="finite"):
        validate_indexed_batch(nonfinite, _vocab_sizes())


def test_fixed_seeds_and_structured_settings_match_execution_card():
    cards = json.loads((ROOT / "reports" / "feature_control_study_20261001" / "EXECUTION_CARDS.json").read_text(encoding="utf-8"))
    assert tuple(cards["seeds"]) == FIXED_SEEDS == (20261001, 20261002, 20261003)
    assert "embedding_dim" not in STRUCTURED_DIMENSION_KEYS
    assert "current_item_group" not in STRUCTURED_DIMENSION_KEYS


def test_raw_item_validator_is_strict_before_any_cast():
    assert validate_raw_item_id(np.int64(1001)) == 1001
    for value in (True, 1.0, 0, -1):
        with pytest.raises(ValueError):
            validate_raw_item_id(value)
    with pytest.raises(ValueError):
        validate_indexed_batch(_batch(), _vocab_sizes(), raw_item_ids=[1.0])
