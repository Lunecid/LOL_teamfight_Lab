import hashlib
from pathlib import Path

import numpy as np
import pytest

from studies.observable_state_20261002.sampling import query_support, sample_match_queries
from studies.observable_state_20261002.source import EXCLUDED_KEY, read_cached_triplet


def test_nominal_sampler_and_short_final_bucket():
    rows = sample_match_queries("fixture", 360001, np.arange(0, 360002, 60000, dtype=np.int64))
    assert len(rows) == 3
    assert rows[-1]["bucket_lo_ms"] == rows[-1]["bucket_hi_ms"] == 360000
    assert rows[-1]["nominal_query_ms"] == 360000
    assert rows[-1]["frame_index"] == 6
    for k, row in enumerate(rows):
        lo, hi = 120000 * (k + 1), min(120000 * (k + 2) - 1, 360000)
        number = int(hashlib.sha256(("fixture:" + str(k)).encode()).hexdigest()[:16], 16)
        assert row["nominal_query_ms"] == lo + number % (hi - lo + 1)


def test_unsupported_nominal_query_is_not_redrawn_or_clamped():
    full = sample_match_queries("fixture", 360001, np.arange(0, 360002, 60000, dtype=np.int64))
    limited = sample_match_queries("fixture", 360001, np.array([0, 60000, 120000], dtype=np.int64))
    assert [r["nominal_query_ms"] for r in full] == [r["nominal_query_ms"] for r in limited]
    assert all(r["actual_query_ms"] is None for r in limited)
    assert all("query_after_last_supported_frame" in r["query_support_reasons"] for r in limited)


def test_inclusive_frame_query_and_duplicates():
    assert query_support(np.array([0, 60000, 120000]), 120000) == (2, [])
    assert query_support(np.array([0, 60000, 120000]), 119999) == (1, [])
    assert query_support(np.array([0, 60000, 60000]), 60000)[0] is None


def test_excluded_match_is_rejected_before_any_file_access():
    with pytest.raises(ValueError, match="user-excluded"):
        read_cached_triplet({"canonical_match_key": EXCLUDED_KEY}, Path("unused_source"))


def test_early_match_produces_no_new_bucket():
    assert sample_match_queries("fixture", 120000, np.array([0, 120000])) == []
