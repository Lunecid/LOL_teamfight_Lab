"""q row table: q roles only, valid labels only, whole-match STOP, match weights, frozen Q_DEV keys, label lock."""
import hashlib
import json

import numpy as np
import pytest

from studies.label_production_20261005 import assemble
from studies.q_production_20261005 import rows as qrows


def _key(i: int) -> bytes:
    return hashlib.sha256(f"match-{i}".encode()).hexdigest().encode("ascii")


def _table():
    # match: 0,0 TRAIN | 1 TRAIN | 2 V_DEV (not a q row) | 3,3,3 Q_DEV (one invalid) | 4 Q_CAL | 5 TEST
    match = np.asarray([0, 0, 1, 2, 3, 3, 3, 4, 5])
    role = np.asarray([0, 0, 0, 1, 3, 3, 3, 4, 5], dtype=np.int8)
    n = len(match)
    valid = np.ones(n, dtype=bool)
    valid[3] = False      # V_DEV game
    valid[6] = False      # missing prediction
    invalid = np.zeros(n, dtype=np.int32)
    invalid[3] = assemble.INVALID["v_role_game_not_a_q_row"]
    invalid[6] = assemble.INVALID["missing_prediction"]
    svi = np.asarray([0.1, -0.2, 0.0, np.nan, 0.3, -0.1, np.nan, 0.05, -0.05])
    return {
        "canonical_match_key": np.asarray([_key(m) for m in match], dtype="S64"),
        "component_key": np.asarray([str(i).encode() for i in range(n)], dtype="S64"),
        "match_index": match.astype(np.int32), "role": role, "fold": np.asarray([0, 0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int8),
        "valid": valid, "invalid_bits": invalid,
        "direction_positive": np.where(valid, (np.nan_to_num(svi) > 0).astype(np.int8), -1).astype(np.int8),
        "exact_zero": np.where(valid, (svi == 0).astype(np.int8), -1).astype(np.int8),
        "svi_delta": svi, "p_pre": np.where(valid, 0.5, np.nan), "query_pre": np.arange(n, dtype=np.int64) * 10,
        "pre_ms": np.full(n, 600_000, dtype=np.int64), "F_pre_ms": np.full(n, 595_000, dtype=np.int64),
        "truncation_bits": np.zeros(n, dtype=np.uint8), "near_zero_bits": np.zeros(n, dtype=np.uint8),
    }


def test_rows_keep_valid_q_roles_and_weight_matches_equally():
    rows, counts = qrows.build_rows(_table())
    assert rows["role"].tolist() == [0, 0, 0, 3, 3, 4, 5]
    assert counts["excluded_invalid_q_rows"] == 1 and counts["excluded_reasons"]["missing_prediction"] == 1
    for match in np.unique(rows["match_index"]):
        assert np.isclose(rows["weight_raw"][rows["match_index"] == match].sum(), 1.0)
    assert rows["y"].tolist() == [1, 0, 0, 1, 0, 1, 0]
    assert np.allclose(rows["frame_age_s"], 5.0) and np.allclose(rows["elapsed_min"], 10.0)


def test_q_stop_is_whole_match_and_uses_the_q_salt():
    rows, counts = qrows.build_rows(_table())
    for key, stop in zip(rows["canonical_match_key"], rows["q_stop"]):
        if stop:
            assert int.from_bytes(hashlib.sha256((qrows.Q_STOP_SALT + key.decode()).encode()).digest()[:8], "big") % 10 == 0
    non_train = rows["role"] != 0
    assert not rows["q_stop"][non_train].any()
    assert counts["TRAIN"]["stop_rows"] + counts["TRAIN"]["fit_rows"] == counts["TRAIN"]["rows"]


def test_normalized_training_weights_have_mean_one():
    match = np.asarray([0, 0, 0, 1, 2, 2])
    w = qrows.normalized_weights(match)
    assert np.isclose(w.mean(), 1.0)
    assert np.isclose(w[match == 0].sum(), w[match == 1].sum())


def test_q_dev_rows_must_lie_inside_the_frozen_keys():
    table = _table()
    keys = np.char.add(np.char.add(table["canonical_match_key"], b":"), table["component_key"])
    frozen = keys[table["role"] == 3]
    _, counts = qrows.build_rows(table, frozen)
    assert counts["q_dev_frozen_keys"] == {"frozen": 3, "q_rows": 2, "q_rows_outside_frozen": 0, "frozen_not_q_rows": 1}
    with pytest.raises(ValueError, match="outside the keys frozen"):
        qrows.build_rows(table, frozen[:1])


def test_test_labels_are_sealed_out_of_the_row_table():
    rows, _ = qrows.build_rows(_table())
    public, sealed = qrows.seal_test(rows)
    test = public["role"] == 5
    assert (public["y"][test] == -1).all() and np.isnan(public["label_svi_delta"][test]).all()
    assert sealed["y"].tolist() == rows["y"][test].tolist() and sealed["engagement_key"].tolist() == rows["engagement_key"][test].tolist()
    assert (public["y"][~test] == rows["y"][~test]).all()


def test_lock_must_exist_and_bind_the_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        qrows.verify_lock(tmp_path, 90)
    for name in ("LABELS_h90.npz", "ENGAGEMENTS.npz", "Q_DEV_H90_KEYS.npy"):
        (tmp_path / name).write_bytes(b"x")
    files = {name: qrows.sha256_file(tmp_path / name) for name in ("LABELS_h90.npz", "ENGAGEMENTS.npz", "Q_DEV_H90_KEYS.npy")}
    (tmp_path / "LABEL_LOCK.json").write_text(json.dumps({"q_may_start": True, "files": files}), encoding="utf-8")
    assert qrows.verify_lock(tmp_path, 90)["q_may_start"]
    (tmp_path / "LABELS_h90.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from the label lock"):
        qrows.verify_lock(tmp_path, 90)


def test_evaluation_columns_come_from_aligned_engagements():
    table = _table()
    n = len(table["role"])
    eng = {"canonical_match_key": table["canonical_match_key"], "component_key": table["component_key"],
           "n_members": np.arange(1, n + 1, dtype=np.int64), "firstkill_credit_blue": np.ones(n, dtype=np.int64),
           "firstkill_credit_red": np.full(n, 3, dtype=np.int64)}
    rows, _ = qrows.build_rows(table, engagements=eng)
    kept = np.flatnonzero(np.isin(table["role"], (0, 3, 4, 5)) & table["valid"])
    assert np.array_equal(rows["eval_n_kills"], eng["n_members"][kept]) and set(rows["eval_firstkill_credit_red"]) == {3}
    shuffled = dict(eng, component_key=eng["component_key"][::-1].copy())
    with pytest.raises(ValueError, match="does not align"):
        qrows.build_rows(table, engagements=shuffled)
