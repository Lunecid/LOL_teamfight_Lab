"""Control comparison: key join, per-pair differences, per-match means with equal match weight, exclusions."""
import json

import numpy as np

from studies.label_production_20261005 import assemble, controls_score


def _write(tmp_path):
    labels = tmp_path / "labels"
    ends = labels / "CONTROL_END_ROWS_h90_1"
    ends.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 12
    match_keys = np.asarray([f"{m:064d}".encode() for m in np.arange(n) // 3], dtype="S64")
    comp = np.asarray([f"c{i}".encode() for i in range(n)], dtype="S8")
    start_q = np.asarray([10, 10, -1, 20, 21, -1, -1, -1, 30, 31, 32, 33], dtype=np.int64)
    end_index = np.full(n, -1, dtype=np.int64)
    end_index[start_q >= 0] = np.arange((start_q >= 0).sum())
    assemble.write_npz(labels / "CONTROLS_MATCHING_h90.npz", {"canonical_match_key": match_keys, "component_key": comp, "control_start_query": start_q,
                                                               "control_start_ms": np.where(start_q >= 0, 1000, -1), "control_end_ms": np.where(start_q >= 0, 2000, -1)})
    (labels / "CONTROLS_MATCHING_h90.json").write_text(json.dumps({"unmatched_reasons": {"no_candidate_after_rules": 4}}), encoding="utf-8")
    assemble.write_npz(ends / "END_TABLE.npz", {"engagement_end_index": end_index})
    starts = np.unique(start_q[start_q >= 0])
    start_v = rng.uniform(0.2, 0.8, len(starts))
    end_v = rng.uniform(0.2, 0.8, int((start_q >= 0).sum()))
    outside = np.zeros(len(end_v), dtype=bool)
    outside[-1] = True
    assemble.write_npz(labels / "CONTROL_V_h90.npz", {"start_query": starts, "start_v": start_v, "end_index": np.arange(len(end_v)), "end_v": end_v,
                                                       "end_outside_training_frame_age_support": outside,
                                                       "end_after_last_stored_query": np.zeros(len(end_v), dtype=bool)})
    (labels / "CONTROL_V_h90.json").write_text(json.dumps({"inputs": {"end_rows": str(ends / "CONTROL_END_ROWS_h90.npz")}}), encoding="utf-8")
    delta = rng.normal(0, 0.1, n)
    valid = np.ones(n, dtype=bool)
    valid[4] = False                                        # a control whose label is invalid is not a pair
    perm = rng.permutation(n)                               # label rows in another order: the join is by key
    assemble.write_npz(labels / "LABELS_h90.npz", {"canonical_match_key": match_keys[perm], "component_key": comp[perm], "svi_delta": delta[perm],
                                                   "valid": valid[perm], "match_index": (np.arange(n) // 3)[perm], "pre_ms": np.full(n, 20 * 60000)[perm]})
    return labels, start_q, starts, start_v, end_index, end_v, delta, valid


def test_compare_pairs_and_match_weighting(tmp_path):
    labels, start_q, starts, start_v, end_index, end_v, delta, valid = _write(tmp_path)
    report = controls_score.compare(labels)
    pair = (start_q >= 0) & valid
    w = np.full(len(start_q), np.nan)
    w[start_q >= 0] = end_v[end_index[start_q >= 0]] - start_v[np.searchsorted(starts, start_q[start_q >= 0])]
    d = delta - w
    match = np.arange(len(start_q)) // 3
    per_match = [d[pair & (match == m)].mean() for m in np.unique(match[pair])]
    assert report["matched"] == int(pair.sum()) == 7 and report["control_but_invalid_label"] == 1 and report["unmatched"] == 4
    assert np.isclose(report["all_matched"]["d_signed"]["mean"], np.mean(per_match))
    assert report["all_matched"]["matches"] == 4                                    # pairs per match: 2, 1 (one invalid label), 1, 3
    assert report["excluding_ends_outside_support"]["rows"] == 6
    table = dict(np.load(labels / "CONTROLS_h90.npz"))
    assert np.array_equal(table["matched"], pair) and np.allclose(table["w_control"][pair], w[pair])
