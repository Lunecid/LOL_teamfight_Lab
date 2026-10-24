"""Focused checks of the E1 sampling rule helpers and the parity comparators (no raw data, no old repo needed)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studies.external_validation_20261009 as E  # noqa: E402


def test_hash_formula_golden():
    # golden digest computed independently with hashlib on b"external_20261009:KR_7907057494"
    assert E.match_hash("KR_7907057494") == "f95d8ffcb612402e25d274f33f3cc73e7f954fa813beaf6e9ea0fe8f3a4de408"


def test_select_smallest_orders_by_hash_not_id():
    # h(KR_2) = 5504ca21... < h(KR_1) = 77ce26ac...
    assert E.select_smallest(["KR_1", "KR_2"], 1) == ["KR_2"]
    ids = [f"KR_{i}" for i in range(500)]
    chosen = E.select_smallest(ids, 37)
    assert len(chosen) == 37 == len(set(chosen))
    worst = max(E.match_hash(m) for m in chosen)
    assert all(E.match_hash(m) > worst for m in set(ids) - set(chosen))
    assert E.select_smallest(ids[::-1], 37) == chosen          # input order is irrelevant
    assert E.select_smallest(ids + ids, 37) == chosen          # duplicates collapse
    assert E.select_smallest(ids[:10], 37) == E.select_smallest(ids[:10], 10)


def test_major_minor():
    assert E.major_minor("15.18.710.2811") == "15.18"
    assert E.major_minor("16.13.791.5903") == "16.13"
    assert E.major_minor("") == "" and E.major_minor(None) == "" and E.major_minor("15") == ""


def test_json_bytes_one_record_per_line_round_trip():
    obj = dict(a=1, b=dict(c=[1, 2]), matches=[dict(match_id="KR_1", x=1.5), dict(match_id="KR_2", x=None)])
    raw = E.json_bytes(obj, list_key="matches")
    assert json.loads(raw) == obj
    lines = raw.decode("utf-8").splitlines()
    assert sum(1 for ln in lines if '"match_id"' in ln) == 2
    assert json.loads(E.json_bytes(dict(matches=[]), list_key="matches")) == dict(matches=[])
    assert json.loads(E.json_bytes(dict(matches=[dict(k=1)]), list_key="matches")) == dict(matches=[dict(k=1)])


def test_parity_comparators(tmp_path):
    from studies.external_validation_20261009 import parity_compare as P

    a = dict(minute_ts=np.arange(3, dtype=np.int64), node_minute=np.ones((3, 10, 4), np.float32))
    pa, pb, pc, pd = (tmp_path / f"{k}.npz" for k in "abcd")
    np.savez_compressed(pa, **a)
    np.savez_compressed(pb, **a)
    res = P.compare_npz(pa, pb)
    assert res["equal"] and res["names_equal"]
    b = dict(a, node_minute=a["node_minute"].copy())
    b["node_minute"][1, 2, 3] = 1.25
    np.savez_compressed(pc, **b)
    res = P.compare_npz(pa, pc)
    assert not res["equal"] and res["arrays"]["node_minute"]["max_abs_diff"] == 0.25
    assert res["arrays"]["minute_ts"]["equal"]
    np.savez_compressed(pd, minute_ts=a["minute_ts"].astype(np.int32), node_minute=a["node_minute"])
    res = P.compare_npz(pa, pd)
    assert not res["equal"] and not res["arrays"]["minute_ts"]["dtype_equal"]

    m1 = dict(match_id="KR_1", static_meta=dict(bans=dict(blue=[1, 2], red=[3])), patch="16.13")
    m2 = json.loads(json.dumps(m1))
    m2["static_meta"]["bans"]["red"] = [4]
    m2["extra"] = 1
    diffs = P.json_diff_paths(m1, m2)
    assert "static_meta.bans.red[0]" in diffs and "extra" in diffs and len(diffs) == 2
    assert P.json_diff_paths(m1, json.loads(json.dumps(m1))) == []
