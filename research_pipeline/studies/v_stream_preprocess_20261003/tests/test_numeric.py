from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from studies.v_model_fit_input_20261003.preprocess import FitScopeEvidence, NumericFieldSpec, NumericPreprocessor
from studies.v_preparation_20261002.partition import context_membership
from studies.v_training_gate_20261003.guard import ALL_MEMBERSHIP
from studies.v_stream_preprocess_20261003.numeric import fit_numeric_stream


def _evidence(n_rows: int) -> FitScopeEvidence:
    key = "a" * 64
    flags = context_membership("15.14", key, None)
    rows = tuple(range(n_rows))
    return FitScopeEvidence(
        context="final",
        scope="FIT",
        row_indices=rows,
        game_keys=(key,) * n_rows,
        roles=("TRAIN",) * n_rows,
        membership=tuple(sorted((name, (bool(flags[name]),) * n_rows) for name in ALL_MEMBERSHIP)),
        source_hash="c" * 64,
    )


def _eager(values: np.ndarray) -> tuple[float, float, float, bool, int]:
    values = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return 0.0, 0.0, 1.0, False, 0
    median = float(np.median(values[finite]))
    values[~finite] = median
    mean = float(np.mean(values, dtype=np.float64))
    scale = float(np.std(values, dtype=np.float64))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    return median, mean, scale, True, count


def _fit(reader, *, n_rows: int, count: int, spec: NumericFieldSpec, channel_count: int, **kwargs):
    evidence = _evidence(n_rows)
    kwargs.setdefault("max_scratch_bytes", count * 32)
    return fit_numeric_stream(
        reader,
        count=count,
        channel_count=channel_count,
        spec=spec,
        evidence=evidence,
        expected_indices=list(range(n_rows)),
        expected_game_keys=["a" * 64] * n_rows,
        expected_roles=["TRAIN"] * n_rows,
        expected_source_hash="c" * 64,
        **kwargs,
    )


def test_stream_fit_matches_independent_eager_player_computation_and_reload():
    values = np.full((3, 10, 2), np.nan, dtype=np.float64)
    values[:, :, 0] = np.arange(30, dtype=np.float64).reshape(3, 10)
    values[0, 0, 0] = np.nan
    values[1, 1, 0] = np.inf
    rows = [0, 2]
    selected = values[rows]
    reader = lambda channel: selected[:, :, channel].reshape(-1)
    fitted = _fit(
        reader,
        n_rows=len(rows),
        count=len(rows) * 10,
        spec=NumericFieldSpec("node", axis="player", channels=("health", "unobserved")),
        channel_count=2,
        max_scratch_bytes=len(rows) * 10 * 32,
    )

    expected = [_eager(selected[:, :, channel].reshape(-1)) for channel in range(2)]
    np.testing.assert_array_equal(fitted.median, [item[0] for item in expected])
    np.testing.assert_array_equal(fitted.mean, [item[1] for item in expected])
    np.testing.assert_array_equal(fitted.scale, [item[2] for item in expected])
    np.testing.assert_array_equal(fitted.usable, [item[3] for item in expected])
    np.testing.assert_array_equal(fitted.fit_count, [item[4] for item in expected])
    assert fitted.spec.normalized_axis == "player"

    restored = NumericPreprocessor.from_dict(fitted.to_dict())
    assert restored.fingerprint == fitted.fingerprint
    np.testing.assert_array_equal(restored.median, fitted.median)


def test_stream_fit_accepts_chunk_iterable_and_matches_team_pooling():
    values = np.asarray(
        [
            [[1.0], [3.0]],
            [[5.0], [7.0]],
            [[999.0], [999.0]],
        ],
        dtype=np.float64,
    )
    selected = values[:2]

    def reader(channel):
        pooled = selected[:, :, channel].reshape(-1)
        return (chunk for chunk in np.array_split(pooled, 3))

    fitted = _fit(
        reader,
        n_rows=2,
        count=4,
        spec={"name": "team", "axis": "paired_team", "channels": ("gold",)},
        channel_count=1,
    )
    expected = _eager(selected[:, :, 0].reshape(-1))
    np.testing.assert_array_equal(fitted.median, [expected[0]])
    np.testing.assert_array_equal(fitted.mean, [expected[1]])
    np.testing.assert_array_equal(fitted.scale, [expected[2]])
    assert fitted.spec.normalized_axis == "team"


def test_stream_fit_uses_one_for_zero_scale_but_zero_policy_for_all_missing():
    selected = np.asarray([[5.0, np.nan], [5.0, np.nan]], dtype=np.float64)
    fitted = _fit(
        lambda channel: selected[:, channel],
        n_rows=2,
        count=2,
        spec=NumericFieldSpec("global", axis="row", channels=("constant", "missing")),
        channel_count=2,
    )
    np.testing.assert_array_equal(fitted.median, [5.0, 0.0])
    np.testing.assert_array_equal(fitted.mean, [5.0, 0.0])
    np.testing.assert_array_equal(fitted.scale, [1.0, 1.0])
    assert fitted.usable.tolist() == [True, False]
    assert fitted.fit_count.tolist() == [2, 0]


def test_stream_fit_rejects_untrusted_scope_before_reading():
    evidence = _evidence(2)
    calls: list[int] = []
    bad_evidence = replace(evidence, source_hash="d" * 64)
    with pytest.raises(ValueError, match="source hash"):
        fit_numeric_stream(
            lambda channel: calls.append(channel) or np.zeros(2),
            count=2,
            channel_count=1,
            spec=NumericFieldSpec("x"),
            evidence=bad_evidence,
            expected_indices=[0, 1],
            expected_game_keys=["a" * 64] * 2,
            expected_roles=["TRAIN"] * 2,
            expected_source_hash="c" * 64,
            max_scratch_bytes=64,
        )
    assert calls == []


def test_stream_fit_rejects_dimension_and_scratch_budget_violations():
    spec = NumericFieldSpec("x", axis="row", channels=("x",))
    with pytest.raises(ValueError, match="one-dimensional"):
        _fit(
            lambda channel: np.zeros((2, 1)),
            n_rows=2,
            count=2,
            spec=spec,
            channel_count=1,
        )
    with pytest.raises(ValueError, match="max_scratch_bytes"):
        _fit(
            lambda channel: np.zeros(2),
            n_rows=2,
            count=2,
            spec=spec,
            channel_count=1,
            max_scratch_bytes=15,
        )
    with pytest.raises(ValueError, match="does not match"):
        _fit(
            lambda channel: np.zeros(3),
            n_rows=2,
            count=3,
            spec=spec,
            channel_count=1,
        )


def test_stream_fit_rejects_infinity_but_accepts_nan_as_missing():
    with pytest.raises(ValueError, match="infinity"):
        _fit(
            lambda channel: np.asarray([1.0, np.inf]),
            n_rows=2,
            count=2,
            spec=NumericFieldSpec("x", channels=("x",)),
            channel_count=1,
        )
