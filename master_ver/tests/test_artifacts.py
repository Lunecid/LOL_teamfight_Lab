from __future__ import annotations

import json

import numpy as np
import pytest

from studies.v_full_preprocess_20261003.artifacts import fit_bundle, load_bundle, save_bundle
from studies.v_full_preprocess_20261003.artifacts import _full_channel_statistics
from studies.v_model_fit_input_20261003.common import fit_common, namespace_rows
from studies.v_model_fit_input_20261003.preprocess import FitScopeEvidence
from studies.v_model_fit_input_20261003.tests.test_common import fixture, scope, semantic_row
from studies.v_stream_preprocess_20261003.numeric import _channel_statistics


def _rows_and_scope():
    rows = [semantic_row(fixture()), semantic_row(fixture())]
    # Exercise the accepted all-missing channel policy independently of the
    # other 74 global channels.
    for row in rows:
        row["numeric"]["global"][0] = np.nan
        # ``-1`` is a source sentinel.  The canonical tokenizer ignores it;
        # valid raw integer categories 0 and 1 remain in the vocabulary.
        row["categories"]["team"]["ban0"][0] = -1
    return rows, scope(["a" * 64, "b" * 64])


def _full_arrays(rows):
    return {
        lane: np.stack([row["numeric"][lane] for row in rows]).astype(np.float64)
        for lane in ("node", "team", "global")
    }


def _global_evidence(scope_data, *, rows=(10, 12), context=None, excluded=()):
    return FitScopeEvidence(
        context=context or scope_data.get("context", "final"),
        scope=scope_data.get("scope", "REFIT_TRAIN"),
        row_indices=tuple(rows),
        game_keys=tuple(scope_data["game_keys"]),
        roles=tuple(scope_data["roles"]),
        membership=tuple(
            (name, tuple(bool(value) for value in values))
            for name, values in sorted(scope_data["membership"].items())
        ),
        excluded_game_keys=tuple(excluded),
        source_hash=scope_data["source_hash"],
    )


def _fit_full(rows, scope_data):
    expected = fit_common(rows, [0, 1], **scope_data)
    categories = {
        namespace: {value for bag in values for value in bag}
        for namespace, values in namespace_rows(rows).items()
    }
    fitted = fit_bundle(_full_arrays(rows), [0, 1], _global_evidence(scope_data), categories)
    return expected, fitted


def test_full_fit_matches_common_and_shares_one_evidence_object():
    rows, scope_data = _rows_and_scope()
    expected, fitted = _fit_full(rows, scope_data)

    for lane in ("global", "node", "team"):
        actual = fitted.numeric_map[lane]
        reference = expected.numeric_map[lane]
        np.testing.assert_array_equal(actual.median, reference.median)
        np.testing.assert_array_equal(actual.mean, reference.mean)
        np.testing.assert_array_equal(actual.scale, reference.scale)
        np.testing.assert_array_equal(actual.usable, reference.usable)
        np.testing.assert_array_equal(actual.fit_count, reference.fit_count)
        assert actual.evidence is fitted.evidence
    assert not bool(fitted.numeric_map["global"].usable[0])

    champion = fitted.categorical_map["static:champion_id"]
    assert -1 not in champion.categories
    assert champion.mapping[0] == 2
    assert champion.mapping[1] == 3
    np.testing.assert_array_equal(champion.transform(np.asarray([-1, 0, 1])), [1, 2, 3])
    assert champion.categories == expected.categorical_map["static:champion_id"].categories


def test_channel_reduction_matches_canonical_nonfast_axis_order():
    rng = np.random.default_rng(20261003)
    matrix = rng.normal(size=(128, 102)).astype(np.float64)
    matrix += np.arange(matrix.size, dtype=np.float64).reshape(matrix.shape) * 1e-9
    values = matrix[:, 12].copy()

    expected_mean = float(np.mean(matrix, axis=0, dtype=np.float64)[12])
    expected_scale = float(np.std(matrix, axis=0, dtype=np.float64)[12])
    actual = _full_channel_statistics(values.copy())
    legacy = _channel_statistics(values.copy())

    assert actual[1] == expected_mean
    assert actual[2] == expected_scale
    assert (legacy[1], legacy[2]) != (expected_mean, expected_scale)


def test_bundle_reload_is_compact_and_rejects_corruption_or_existing_destination(tmp_path):
    rows, scope_data = _rows_and_scope()
    _expected, fitted = _fit_full(rows, scope_data)
    directory = tmp_path / "bundle"
    save_bundle(directory, fitted)
    assert sorted(path.name for path in directory.iterdir()) == [
        "MANIFEST.json",
        "evidence.json",
        "numeric.json",
        "vocab.json",
    ]
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"].keys() == {"evidence.json", "numeric.json", "vocab.json"}
    assert b'"evidence"' not in (directory / "numeric.json").read_bytes()
    assert b'"evidence"' not in (directory / "vocab.json").read_bytes()

    restored = load_bundle(directory)
    assert restored.fingerprint == fitted.fingerprint
    assert restored.evidence is restored.numeric_map["node"].evidence
    assert restored.evidence is restored.categorical_map["item"].evidence
    assert not restored.numeric_map["global"].median.flags.writeable
    with pytest.raises(FileExistsError):
        save_bundle(directory, fitted)

    numeric_path = directory / "numeric.json"
    numeric_path.write_text(numeric_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(directory)


def test_fit_rejects_oof_without_whole_game_exclusion():
    rows = [semantic_row(fixture())]
    evidence = FitScopeEvidence(
        context="oof0",
        scope="FIT",
        row_indices=(999,),
        game_keys=("a" * 64,),
        roles=("TRAIN",),
        membership=(("fit", (True,)),),
        source_hash="c" * 64,
    )
    categories = {namespace: set() for namespace in namespace_rows(rows)}
    with pytest.raises(ValueError, match="excluded_game_keys"):
        fit_bundle(_full_arrays(rows), [0], evidence, categories)
