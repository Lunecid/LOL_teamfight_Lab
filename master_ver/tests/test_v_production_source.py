from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from studies.v_model_fit_input_20261003.common import fit_common, semantic_row, table_layout
from studies.v_model_fit_input_20261003.tests.test_common import fixture, scope
from studies.v_production_20261004.cache import CacheWriter, InputCache, transform_batch
from studies.v_production_20261004.source import CacheSource


def _metadata(start: int, roles: list[bytes], memberships: list[int], labels: list[int], swapped: list[bool]) -> dict[str, np.ndarray]:
    count = len(roles)
    return {
        "nominal_index": np.arange(start, start + count, dtype=np.int64),
        "match_index": np.arange(start, start + count, dtype=np.int32),
        "query_index": np.arange(start, start + count, dtype=np.int64),
        "query_ms": np.arange(1000 + start, 1000 + start + count, dtype=np.int64),
        "train_ordinal": np.arange(start, start + count, dtype=np.int64),
        "game_key": np.asarray([f"{start + index + 1:064x}".encode("ascii") for index in range(count)], dtype="S64"),
        "role": np.asarray(roles, dtype="S8"),
        "patch": np.asarray([b"15.14"] * count, dtype="S5"),
        "swapped": np.asarray(swapped, dtype=bool),
        "source_w": np.asarray(labels, dtype=bool),
        # These are already replacement-swapped labels, as in the production
        # cache; the source must not flip them a second time.
        "label_w": np.asarray(labels, dtype=bool),
        "winning_team": np.where(np.asarray(labels, dtype=bool), 100, 200).astype(np.int16),
        "context_membership": np.asarray(memberships, dtype=np.uint8),
    }


def _cache(tmp_path: Path) -> tuple[Path, object]:
    rows = [semantic_row(fixture()) for _ in range(6)]
    bundle = fit_common(rows, [0], **scope([f"{index + 1:064x}" for index in range(6)]))
    metadata_a = _metadata(0, [b"TRAIN", b"TRAIN", b"V_DEV"], [5, 7, 8], [0, 0, 1], [False, True, False])
    metadata_b = _metadata(3, [b"TRAIN", b"V_CAL", b"TRAIN"], [6, 16, 5], [1, 0, 1], [False, False, False])
    transformed_a = transform_batch(rows[:3], metadata_a, bundle)
    transformed_b = transform_batch(rows[3:], metadata_b, bundle)
    _, width = table_layout(bundle)
    root = tmp_path / "input_cache"
    with CacheWriter(root, total_rows=6, table_width=width, plan={"context": "final", "scope": "FIT", "rows": 6}) as writer:
        writer.append(transformed_a)
        writer.append(transformed_b)
        writer.finalize()
    return root, bundle


def test_source_filters_membership_rebatches_and_preserves_ragged_offsets(tmp_path):
    root, _bundle = _cache(tmp_path)
    with InputCache(root) as cache:
        source = CacheSource(cache, family="shared_slot_mlp", dimensions={"vocab_sizes": {"champion_id": 4}}, expected_train_rows=5)
        assert source.row_counts == {"fit": 3, "stop": 2, "refit": 4, "dev": 1, "cal": 1, "gate": 4}
        batches = list(source.iter_batches("gate", epoch=0, batch_size=3))
        assert [len(batch["target"]) for batch in batches] == [3, 1]
        assert all(batch["metadata"]["target_is_supervised"] for batch in batches)
        assert batches[0]["metadata"]["roles"] == ("TRAIN", "TRAIN", "TRAIN")
        bag = batches[0]["inputs"].node_bags["current_item"]
        assert bag.offsets.shape == (3, 11)
        assert int(bag.offsets[-1, -1]) == len(bag.indices)
        assert batches[0]["inputs"].player_team_ids.shape[0] == 3
        assert batches[0]["inputs"].table_csr.shape[0] == 3
        source.close()


def test_source_keeps_supervised_swapped_label_and_decodes_metadata(tmp_path):
    root, _bundle = _cache(tmp_path)
    with InputCache(root) as cache:
        source = CacheSource(cache, family="logistic")
        item = next(source.iter_batches("fit", batch_size=1024))
        assert item["target"].tolist() == [0.0, 0.0, 1.0]
        assert item["metadata"]["roles"] == ("TRAIN", "TRAIN", "TRAIN")
        assert all(isinstance(value, str) for value in item["metadata"]["game_keys"])
        assert item["metadata"]["target_is_supervised"] is True
        assert item["metadata"]["game_keys"][1].endswith("2")
        assert set(item["inputs"]) == {"table_csr"}
        source.close()


def test_source_rejects_active_excluded_oof_game(tmp_path):
    root, _bundle = _cache(tmp_path)
    active_train_key = f"{2:064x}"
    with InputCache(root) as cache:
        with pytest.raises(ValueError, match="excluded OOF game"):
            CacheSource(cache, family="logistic", excluded_game_keys=[active_train_key])


def test_source_allows_excluded_game_with_only_the_oof_target_bit(tmp_path):
    root, _bundle = _cache(tmp_path)
    membership = np.load(root / "metadata_context_membership.npy", mmap_mode="r+")
    membership[1] = 32  # partition.context_membership "target": held-out fold, scored later
    membership.flush()
    del membership
    excluded_key = f"{2:064x}"
    with InputCache(root, verify=False) as cache:
        source = CacheSource(cache, family="logistic", excluded_game_keys=[excluded_key])
        served = [key for item in source.iter_batches("refit", batch_size=1024) for key in item["metadata"]["game_keys"]]
        assert excluded_key not in served
        source.close()


def test_source_family_shapes_and_dimensions(tmp_path):
    root, _bundle = _cache(tmp_path)
    with InputCache(root) as cache:
        plain = CacheSource(cache, family="plain_mlp", dimensions={"vocab_sizes": {"champion_id": 4}})
        plain_item = next(plain.iter_batches("cal", batch_size=1024))
        assert plain_item["inputs"].dense_features.shape == (1, plain.table_width)
        assert plain.dimensions["vocab_sizes"] == {"champion_id": 4}
        structured = CacheSource(cache, family="team_set_mlp")
        structured_item = next(structured.iter_batches("dev", batch_size=1024))
        assert structured_item["inputs"].table_csr.shape[0] == 1
        reload_item = next(structured.iter_batches("reload", batch_size=1024))
        assert reload_item["metadata"]["roles"] == ("V_DEV",)
        plain.close()
        structured.close()


def test_source_rejects_membership_role_mismatch(tmp_path):
    root, _bundle = _cache(tmp_path)
    role = np.load(root / "metadata_role.npy", mmap_mode="r+")
    role[0] = b"V_DEV"
    role.flush()
    with InputCache(root) as cache:
        with pytest.raises(ValueError, match="membership bit"):
            CacheSource(cache)


def test_source_allows_independent_fit_refit_table_widths_and_reload_alias(tmp_path):
    fit_root, _bundle = _cache(tmp_path)
    refit_root = tmp_path / "refit_cache"
    with InputCache(fit_root) as fit:
        with CacheWriter(refit_root, total_rows=fit.rows, table_width=fit.table_width + 1, plan={"scope": "REFIT_TRAIN", "rows": fit.rows}) as writer:
            for index in range(len(fit._batches)):
                batch = fit.batch(index)
                widened = replace(
                    batch.structured,
                    table_csr=sparse.hstack(
                        [batch.structured.table_csr, sparse.csr_matrix((batch.rows, 1), dtype=np.float32)],
                        format="csr",
                    ),
                )
                writer.append(replace(batch, structured=widened))
            writer.finalize()
    with InputCache(fit_root) as fit, InputCache(refit_root) as refit:
        source = CacheSource(fit_cache=fit, refit_cache=refit, family="team_set_mlp", dimensions=refit.table_width)
        assert source.dimensions["refit_table_width"] == refit.table_width
        dev = next(source.iter_batches("reload", batch_size=1024))
        assert dev["inputs"].table_csr.shape[1] == refit.table_width
        assert dev["metadata"]["roles"] == ("V_DEV",)
        source.close()
