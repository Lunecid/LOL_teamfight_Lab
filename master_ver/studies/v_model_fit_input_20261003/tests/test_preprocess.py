from __future__ import annotations

import numpy as np
import pytest

from studies.v_model_fit_input_20261003.preprocess import (
    CategoricalFieldSpec,
    CategoricalVocabulary,
    NumericFieldSpec,
    NumericPreprocessor,
    fit_numeric,
    fit_ragged_vocabulary,
    fit_vocabulary,
    make_fit_evidence,
)


def _membership(n: int, *, stop: list[bool] | None = None, target: list[bool] | None = None):
    return {
        "fit": [True] * n,
        "refit": [True] * n,
        "stop": [False] * n if stop is None else stop,
        "target": [False] * n if target is None else target,
    }


def test_numeric_pools_player_channels_and_keeps_all_missing_zero():
    values = np.zeros((3, 10, 2), dtype=np.float64)
    values[0, :, 0] = np.arange(1, 11)
    values[1, :, 0] = np.arange(11, 21)
    values[1, 0, 0] = np.nan
    values[2, :, 0] = 1000.0
    values[:2, :, 1] = np.nan
    fitted = fit_numeric(
        values,
        fit_indices=[0, 1],
        spec=NumericFieldSpec("node", axis="player", channels=("health", "unobserved")),
        context="oof2",
        scope="FIT",
        game_keys=["g0", "g1", "g2"],
        roles=["TRAIN", "TRAIN", "TRAIN"],
        membership=_membership(3),
    )
    assert fitted.evidence.consumed_indices == (0, 1)
    assert fitted.evidence.consumed_indices_sha256
    np.testing.assert_allclose(fitted.median, [10.0, 0.0])
    np.testing.assert_allclose(fitted.mean, [10.45, 0.0])
    np.testing.assert_allclose(fitted.scale[1], 1.0)
    assert fitted.usable.tolist() == [True, False]
    transformed = fitted.transform(values)
    assert transformed.shape == values.shape
    assert np.all(transformed[:, :, 1] == 0.0)
    assert np.isfinite(transformed).all()


def test_numeric_team_channels_share_one_pooled_statistic():
    values = np.asarray([[[0.0], [10.0]], [[20.0], [30.0]], [[999.0], [999.0]]])
    fitted = fit_numeric(
        values,
        fit_indices=[0, 1],
        spec={"name": "team", "axis": "paired_team", "channels": ("gold",)},
        context="final",
        scope="REFIT_TRAIN",
        game_keys=["g0", "g1", "g2"],
        roles=["TRAIN"] * 3,
        membership=_membership(3),
    )
    np.testing.assert_allclose(fitted.median, [15.0])
    np.testing.assert_allclose(fitted.mean, [15.0])
    assert fitted.spec.normalized_axis == "team"


def test_vocabulary_distinguishes_raw_zero_and_one_from_pad_and_unk():
    values = np.asarray([[0], [1], [7], [99]])
    vocab = fit_vocabulary(
        values,
        fit_indices=[0, 1, 2],
        spec=CategoricalFieldSpec("champion", axis="row"),
        context="final",
        scope="FIT",
        game_keys=["g0", "g1", "g2", "g3"],
        roles=["TRAIN"] * 4,
        membership=_membership(4),
    )
    assert vocab.mapping == {0: 2, 1: 3, 7: 4}
    assert vocab.transform(np.asarray([[0], [1], [7], [99]])).reshape(-1).tolist() == [2, 3, 4, 1]
    assert vocab.transform(np.asarray([[0], [1]]), pad_mask=np.asarray([[True], [False]])).reshape(-1).tolist() == [0, 3]


def test_ragged_vocabulary_only_consumes_selected_rows_and_reloads(tmp_path):
    rows = (("a", 0, 0), ("future", 99), ("b", 1))
    vocab = fit_ragged_vocabulary(
        rows,
        fit_indices=[0, 2],
        name="nonitem",
        context="oof0",
        scope="FIT",
        game_keys=["g0", "g1", "g2"],
        roles=["TRAIN"] * 3,
        membership=_membership(3),
    )
    assert vocab.categories == (0, 1, "a", "b")
    transformed = vocab.transform_ragged(rows)
    assert transformed[0].tolist() == [4, 2, 2]
    assert transformed[1].tolist() == [1, 1]
    assert transformed[2].tolist() == [5, 3]
    path = tmp_path / "nonitem.json"
    vocab.save(path)
    restored = CategoricalVocabulary.load(path)
    assert restored.fingerprint == vocab.fingerprint
    assert restored.evidence.consumed_indices == (0, 2)


def test_scope_guards_reject_forbidden_rows_and_allow_refit_stop():
    with pytest.raises(ValueError, match="stop"):
        make_fit_evidence(
            [0],
            n_rows=1,
            context="final",
            scope="FIT",
            game_keys=["g0"],
            roles=["TRAIN"],
            membership=_membership(1, stop=[True]),
        )
    evidence = make_fit_evidence(
        [0],
        n_rows=1,
        context="final",
        scope="REFIT_TRAIN",
        game_keys=["g0"],
        roles=["TRAIN"],
        membership=_membership(1, stop=[True]),
    )
    assert evidence.scope == "REFIT_TRAIN"
    with pytest.raises(ValueError, match="excluded game keys"):
        make_fit_evidence(
            [0],
            n_rows=1,
            context="oof1",
            scope="FIT",
            game_keys=["held-out"],
            roles=["TRAIN"],
            membership=_membership(1),
            excluded_game_keys=["held-out"],
        )


def test_numeric_save_reload_preserves_hash_and_indices(tmp_path):
    fitted = fit_numeric(
        np.asarray([[1.0], [np.nan], [4.0]]),
        fit_indices=[2, 0],
        name="elapsed",
        context="oof4",
        scope="REFIT_TRAIN",
        game_keys=["g0", "g1", "g2"],
        roles=["TRAIN"] * 3,
        membership=_membership(3),
    )
    path = tmp_path / "numeric.json"
    fitted.save(path)
    restored = NumericPreprocessor.load(path)
    assert restored.fingerprint == fitted.fingerprint
    assert restored.consumed_indices == (2, 0)
    np.testing.assert_allclose(restored.transform([[2.0]]), fitted.transform([[2.0]]))
