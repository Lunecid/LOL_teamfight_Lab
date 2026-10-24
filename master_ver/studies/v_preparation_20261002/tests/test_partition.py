import hashlib

import numpy as np
import pytest

from studies.v_preparation_20261002 import partition as P


KEY = hashlib.sha256(b"fixture-match").hexdigest()


def test_role_integer_boundaries_use_five_h_without_float_rounding():
    unit = 1 << 64
    for h, expected in (
        (0, "V_DEV"),
        (unit // 5, "V_DEV"),
        (unit // 5 + 1, "V_CAL"),
        (2 * unit // 5, "V_CAL"),
        (2 * unit // 5 + 1, "Q_DEV"),
        (4 * unit // 5, "Q_DEV"),
        (4 * unit // 5 + 1, "Q_CAL"),
    ):
        assert P._role_from_u64(h) == expected


def test_partition_helpers_reject_noncanonical_keys_and_bad_excluded_folds():
    with pytest.raises(ValueError):
        P.train_fold("A" * 64)
    with pytest.raises(ValueError):
        P.train_fold("0" * 63)
    with pytest.raises(ValueError):
        P.context_membership("15.14", KEY, excluded_fold=5)
    with pytest.raises(ValueError):
        P.context_membership("15.14", KEY, excluded_fold=-1)


def test_every_train_role_is_excluded_from_its_own_oof_fold():
    fold = P.train_fold(KEY)
    membership = P.context_membership("15.14", KEY, excluded_fold=fold)
    assert membership == {"fit": False, "stop": False, "refit": False,
                          "dev": False, "cal": False, "target": True}
    assert P.context_membership("15.14", KEY)["refit"] is True


def test_q_and_test_roles_never_enter_v_fit_dev_or_cal():
    for patch in ("15.16",):
        membership = P.context_membership(patch, KEY)
        assert all(not membership[name] for name in ("fit", "stop", "refit", "dev", "cal"))
    for h in ((1 << 64) // 2, 4 * (1 << 64) // 5 + 1):
        original = P._hash_u64
        try:
            P._hash_u64 = lambda salt, key, h=h: h if salt == P.ROLE_SALT else original(salt, key)
            role = P.assigned_role("15.15", KEY)
            if role.startswith("Q_"):
                membership = P.context_membership("15.15", KEY)
                assert all(not membership[name] for name in ("fit", "stop", "refit", "dev", "cal"))
        finally:
            P._hash_u64 = original


def _bank(*, unsupported: bool = True, duplicate_ref: bool = False, wrong_ref: bool = False):
    minute_ts = np.array([0, 120_000, 240_000, 360_000], dtype=np.int64)
    query_ms = np.array([150_000, 300_000], dtype=np.int64)
    event_timestamp = np.array([140_000, 300_000], dtype=np.int64)
    nominal = np.array([
        [0, 120_000, 239_999, 150_000, 150_000, 1, 0],
        [1, 240_000, 359_999, 300_000, 300_000, 2, 0 if duplicate_ref else 1],
    ], dtype=np.int32)
    if unsupported:
        nominal = np.vstack((nominal, [2, 360_000, 419_999, 400_000, -1, -1, -1])).astype(np.int32)
    if wrong_ref:
        nominal[0, 6] = 1
    return {
        "v_nominal": nominal,
        "minute_ts": minute_ts,
        "query_ms": query_ms,
        "query_frame_index": np.array([1, 2], dtype=np.int32),
        "query_event_limit": np.array([1, 2], dtype=np.int32),
        "event_timestamp": event_timestamp,
        "end_ms": np.int64(420_000),
    }


def test_nominal_reference_rows_preserve_unsupported_rows_and_event_limits():
    result = P.nominal_reference_rows(_bank())
    assert result["bucket_id"].tolist() == [0, 1, 2]
    assert result["supported"].tolist() == [True, True, False]
    assert result["actual_ms"].tolist() == [150_000, 300_000, -1]
    assert result["frame_index"].tolist() == [1, 2, -1]
    assert result["query_ref"].tolist() == [0, 1, -1]
    assert result["event_limit"].tolist() == [1, 2, -1]


@pytest.mark.parametrize("kwargs", [{"duplicate_ref": True}, {"wrong_ref": True}])
def test_nominal_reference_rows_reject_duplicate_or_wrong_refs(kwargs):
    with pytest.raises(ValueError):
        P.nominal_reference_rows(_bank(**kwargs))
