"""Focused checks of the E2v nominal-V rules (query list transcription, padding, batch shapes, gates); no data run, no label read."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.external_validation_20261009 import e2v_nominal as N  # noqa: E402


def test_pad_batch_repeats_the_last_row():
    rows, n_real = N.pad_batch(["a", "b", "c"], 5)
    assert rows == ["a", "b", "c", "c", "c"] and n_real == 3
    rows, n_real = N.pad_batch(list(range(5)), 5)
    assert rows == list(range(5)) and n_real == 5


def test_batch_size_at_follows_parts_of_32_batches():
    n = 758150                                   # TEST nominal list of v_test_tau (24 parts, last part 4,486 rows)
    assert N.batch_size_at(0, n) == 1024
    assert N.batch_size_at(32767, n) == 1024
    assert N.batch_size_at(757759, n) == 1024
    assert N.batch_size_at(757760, n) == 390
    assert N.batch_size_at(n - 1, n) == 390
    assert N.batch_size_at(6440, 6441) == 6441 - 6144


def test_nominal_rows_transcribes_v_test_tau(tmp_path):
    row_dtype = [("match_index", "<i8"), ("query_ref", "<i8"), ("supported", "?")]
    rows = np.array([(0, 2, True), (0, 0, True), (1, 1, False), (1, 0, True), (2, 0, True)], dtype=row_dtype)
    matches = np.array([(6,), (6,), (5,)], dtype=[("role", "<i1")])
    mq = np.array([(10,), (20,), (30,)], dtype=[("first_query", "<i8")])
    for name, value in (("V_ROWS", rows), ("MATCH_TABLE", matches), ("MATCH_QUERIES", mq)):
        np.save(tmp_path / f"{name}.npy", value)
    out = N.nominal_rows(tmp_path, 6)
    assert out["query_index"].tolist() == [10, 12, 20]
    assert out["match_index"].tolist() == [0, 0, 1]
    assert out["v_row"].tolist() == [1, 0, 3]
    assert out["n_v_rows"] == 5 and out["n_supported"] == 4 and out["n_role_rows"] == 4


def test_workers_are_capped_at_four():
    assert N.workers_allowed(16) <= 4 and N.workers_allowed(1) == 1


def test_stage_times_do_not_touch_the_locked_log():
    from studies.external_validation_20261009 import e2bc_common as B

    assert N.STAGE_TIMES != B.STAGE_TIMES and N.STAGE_TIMES.name == "E2v_NOMINAL_STAGE_TIMES.jsonl"
    assert N.WINNER_FIELD not in N.IDENTITY_FIELDS
