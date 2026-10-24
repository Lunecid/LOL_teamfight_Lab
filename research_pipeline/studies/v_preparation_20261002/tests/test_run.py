import hashlib

import numpy as np
import pytest

from studies.v_preparation_20261002 import partition as P
from studies.v_preparation_20261002 import run as R


def _sampler_fixture():
    original_id = "local-original-id"
    end_ms = 360_001
    minute_ts = np.array([0, 240_000], dtype=np.int64)
    rows = []
    supported = []
    for bucket, lo in enumerate(range(120_000, end_ms, 120_000)):
        hi = min(lo + 119_999, end_ms - 1)
        value = int.from_bytes(hashlib.sha256(f"{original_id}:{bucket}".encode()).digest()[:8], "big")
        nominal = lo + value % (hi - lo + 1)
        is_supported = nominal <= int(minute_ts[-1])
        supported.append(is_supported)
        rows.append((bucket, lo, hi, nominal, nominal if is_supported else -1,
                     int(np.searchsorted(minute_ts, nominal, side="right") - 1) if is_supported else -1,
                     sum(supported[:-1]) if is_supported else -1))
    return (
        {"match_id": original_id, "canonical_match_key": "f" * 64},
        {"end_ms": np.int64(end_ms), "minute_ts": minute_ts,
         "v_nominal": np.asarray(rows, dtype=np.int32)},
        {"supported": np.asarray(supported, dtype=bool)},
    )


def test_verify_nominal_sampler_uses_original_id_and_retains_all_buckets_with_reason_masks():
    entry, bank, refs = _sampler_fixture()

    masks = R.verify_nominal_sampler(entry, bank, refs)

    assert bank["v_nominal"][:, 0].tolist() == [0, 1, 2]
    assert bank["v_nominal"][:, 4:6].tolist() == [[158_612, 0], [-1, -1], [-1, -1]]
    assert masks.tolist() == [0, 1 << R.SUPPORT_REASONS.index("query_after_last_supported_frame"),
                              1 << R.SUPPORT_REASONS.index("query_after_last_supported_frame")]


def test_verify_nominal_sampler_rejects_canonical_key_substitution():
    entry, bank, refs = _sampler_fixture()
    entry["match_id"] = entry["canonical_match_key"]

    with pytest.raises(ValueError, match="original ID or complete bucket coverage"):
        R.verify_nominal_sampler(entry, bank, refs)


def test_verify_nominal_sampler_rejects_missing_bucket():
    entry, bank, refs = _sampler_fixture()
    bank["v_nominal"] = bank["v_nominal"][:-1]

    with pytest.raises(ValueError, match="original ID or complete bucket coverage"):
        R.verify_nominal_sampler(entry, bank, refs)


def _saved_fixture(tmp_path):
    output = tmp_path / "saved"
    output.mkdir()
    keys = ("a" * 64, "b" * 64)
    entries = {
        key: {"patch": "15.14", "role": "TRAIN", "original_index": index,
              "audited_numbers": {"game_end_winning_team": 100}}
        for index, key in enumerate(keys)
    }

    matches = np.zeros(len(keys), dtype=R.MATCH_DTYPE)
    for index, key in enumerate(keys):
        matches[index] = (key.encode("ascii"), index, index + 10, index, 0, 0,
                          P.train_fold(key), P.inner_stop(key), True, index * 2, 2)

    rows = np.zeros(4, dtype=R.ROW_DTYPE)
    for index, match in enumerate(matches):
        start = index * 2
        rows[start:start + 2]["match_index"] = index
        rows[start:start + 2]["shard"] = match["shard"]
        rows[start:start + 2]["local_match"] = match["local_match"]
        rows[start:start + 2]["nominal_index"] = [0, 1]
        rows[start:start + 2]["bucket_id"] = [0, 1]
        rows[start:start + 2]["bucket_lo_ms"] = [120_000, 240_000]
        rows[start:start + 2]["bucket_hi_ms"] = [239_999, 359_999]
        rows[start:start + 2]["nominal_ms"] = [158_612, 278_303]
        rows[start]["actual_ms"] = rows[start]["nominal_ms"]
        rows[start]["query_ref"] = 0
        rows[start]["frame_index"] = 1
        rows[start]["event_limit"] = 1
        rows[start]["supported"] = True
        rows[start]["support_reason_mask"] = 0
        rows[start + 1]["actual_ms"] = -1
        rows[start + 1]["query_ref"] = -1
        rows[start + 1]["frame_index"] = -1
        rows[start + 1]["event_limit"] = -1
        rows[start + 1]["supported"] = False
        rows[start + 1]["support_reason_mask"] = 1 << R.SUPPORT_REASONS.index("query_after_last_supported_frame")

    contexts = np.zeros((6, len(keys)), dtype=np.uint8)
    for index, key in enumerate(keys):
        for context in range(6):
            flags = P.context_membership("15.14", key,
                                         excluded_fold=None if context == 0 else context - 1)
            contexts[context, index] = sum(1 << bit for bit, scope in enumerate(R.SCOPES) if flags[scope])

    np.save(output / "MATCH_TABLE.npy", matches, allow_pickle=False)
    np.save(output / "V_ROWS.npy", rows, allow_pickle=False)
    np.save(output / "CONTEXT_MEMBERSHIP.npy", contexts, allow_pickle=False)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    return output, entries, rows, digest


def _verify_fixture(output, entries, rows, digest):
    return R.verify_saved(output, entries, len(rows), expected_row_digest=digest)


def test_verify_saved_accepts_match_rows_context_and_reloaded_row_digest(tmp_path):
    output, entries, rows, digest = _saved_fixture(tmp_path)

    report = _verify_fixture(output, entries, rows, digest)

    assert report["complete_saved_match_and_nominal_coverage"] is True
    assert report["whole_match_context_membership"] is True
    assert report["saved_reference_bytes_match_validated_source"] is True
    assert report["validated_reference_row_sha256"] == digest


def test_verify_saved_rejects_corrupted_oof_membership(tmp_path):
    output, entries, rows, digest = _saved_fixture(tmp_path)
    contexts = np.load(output / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False)
    contexts[1, 0] ^= 1
    np.save(output / "CONTEXT_MEMBERSHIP.npy", contexts, allow_pickle=False)

    with pytest.raises(ValueError, match="OOF dependency membership"):
        _verify_fixture(output, entries, rows, digest)


def test_verify_saved_rejects_reassigned_nominal_rows(tmp_path):
    output, entries, rows, digest = _saved_fixture(tmp_path)
    saved = np.load(output / "V_ROWS.npy", allow_pickle=False)
    saved[0]["match_index"], saved[2]["match_index"] = 1, 0
    np.save(output / "V_ROWS.npy", saved, allow_pickle=False)

    with pytest.raises(ValueError, match="nominal source reference identity"):
        _verify_fixture(output, entries, rows, digest)


def test_verify_saved_rejects_changed_nominal_source_reference(tmp_path):
    output, entries, rows, digest = _saved_fixture(tmp_path)
    saved = np.load(output / "V_ROWS.npy", allow_pickle=False)
    saved[0]["shard"] += 1
    np.save(output / "V_ROWS.npy", saved, allow_pickle=False)

    with pytest.raises(ValueError, match="nominal source reference identity"):
        _verify_fixture(output, entries, rows, digest)


@pytest.mark.parametrize("mutation", ("event_limit", "nominal_pair", "bucket"))
def test_verify_saved_rejects_positive_nominal_reference_mutations(tmp_path, mutation):
    output, entries, rows, digest = _saved_fixture(tmp_path)
    saved = np.load(output / "V_ROWS.npy", allow_pickle=False)
    if mutation == "event_limit":
        saved[0]["event_limit"] += 1
    elif mutation == "nominal_pair":
        saved[0]["nominal_ms"] += 1
        saved[0]["actual_ms"] += 1
    else:
        saved[0]["bucket_id"] += 1
    np.save(output / "V_ROWS.npy", saved, allow_pickle=False)

    with pytest.raises(ValueError, match="reloaded nominal rows differ"):
        _verify_fixture(output, entries, rows, digest)
