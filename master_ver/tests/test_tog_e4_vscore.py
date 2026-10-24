"""Task 10 (T4.1): E4 LightGBM V scoring (studies/tog_experiments_20261012/e4_vscore.py), synthetic data only.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (sections 2, 3, 6, 7).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from studies.tog_experiments_20261012 import e4_vscore as vs


def _cand(card, loss, *, family="lightgbm", status="completed", seed=20261001):
    return {"family": family, "card_id": card, "status": status, "seed": seed, "dev_metrics": {"raw_logloss": loss}}


# ----------------------------------------------------------------------------- measurer choice (V search rule)
def test_select_measurer_takes_lowest_v_dev_log_loss_within_lightgbm():
    candidates = [_cand("lgbm_l31_m200", 0.49), _cand("lgbm_l63_m500", 0.48), _cand("v_set_h64", 0.47, family="team_set_mlp"),
                  _cand("lgbm_l63_m200", 0.40, status="failed"), _cand("lgbm_l31_m500", float("nan"))]
    assert vs.select_measurer(candidates)["card_id"] == "lgbm_l63_m500"


def test_select_measurer_breaks_ties_by_card_order_and_refuses_empty():
    tied = [_cand("lgbm_l63_m500", 0.48), _cand("lgbm_l31_m500", 0.48)]
    assert vs.select_measurer(tied)["card_id"] == "lgbm_l31_m500"        # models.family_cards order: l31_m200, l31_m500, ...
    with pytest.raises(ValueError):
        vs.select_measurer([_cand("v_set_h64", 0.47, family="team_set_mlp")])


def _consumed(tag, rows):
    return {"rows": rows, "keys_sha256": f"k{tag}", "game_keys_sha256": f"g{tag}", "batches": 1, "key_count": rows,
            "roles": ["TRAIN"], "target_sum": 1.0}


def _measurer_fixture():
    fit, stop, refit, dev = _consumed("f", 8), _consumed("s", 2), _consumed("r", 10), _consumed("d", 5)
    lgbm = {**_cand("lgbm_l63_m500", 0.4873894817692967), "context": "final",
            "checkpoint": "D:\\x\\final\\candidates\\lightgbm\\lgbm_l63_m500\\model_refit.pkl",
            "fit_metrics": {"consumed": fit}, "stop_metrics": {"consumed": stop}, "refit_metrics": {"consumed": refit},
            "dev_metrics": {"raw_logloss": 0.4873894817692967, "consumed": dev}}
    main_v = {**_cand("v_set_h128", 0.47, family="team_set_mlp"),
              "fit_metrics": {"epochs": [{"fit_consumed": fit}, {"fit_consumed": fit}]}, "stop_metrics": {"consumed": stop},
              "refit_metrics": {"epochs": [{"consumed": refit}]}, "dev_metrics": {"raw_logloss": 0.47, "consumed": dev}}
    selection = {"candidates": [_cand("lgbm_l63_m200", 0.4876), lgbm, main_v], "selected": main_v}
    excluded = {"context": "final", "excluded_game_keys": ["b", "a"]}
    excluded_sha = hashlib.sha256(json.dumps(sorted(excluded["excluded_game_keys"])).encode("utf-8")).hexdigest()
    scored = {"identity": {"context": "final", "family": "lightgbm", "card_id": "lgbm_l63_m500", "excluded_game_keys_sha256": excluded_sha,
                           "expected_rows": [8, 2, 10, 5]},
              "files": {"model_refit.pkl": "ab" * 32}}
    plan = {"context": "final", "counts": {"fit": 8, "stop": 2, "refit": 10, "dev": 5, "cal": 4, "gate": 10}}
    return selection, dict(lgbm), scored, plan, excluded


def test_measurer_problems_empty_when_selection_result_and_binding_agree():
    selection, result, scored, plan, excluded = _measurer_fixture()
    record, problems = vs.measurer_problems("final", selection, result, scored, plan, excluded,
                                            prespec={"final": ("lgbm_l63_m500", 0.487389)})
    assert problems == []
    assert record["card_id"] == "lgbm_l63_m500" and record["binding"]["dev"] is True


def test_measurer_problems_flag_prespec_card_and_row_binding():
    selection, result, scored, plan, excluded = _measurer_fixture()
    _, problems = vs.measurer_problems("final", selection, result, scored, plan, excluded,
                                       prespec={"final": ("lgbm_l63_m200", 0.487389)})
    assert any("PRESPEC" in p for p in problems)
    result["stop_metrics"] = {"consumed": _consumed("other", 2)}
    _, problems = vs.measurer_problems("final", selection, result, scored, plan, excluded,
                                       prespec={"final": ("lgbm_l63_m500", 0.487389)})
    assert any("stop" in p for p in problems)
    scored["identity"]["excluded_game_keys_sha256"] = "0" * 64
    _, problems = vs.measurer_problems("final", selection, result, scored, plan, excluded,
                                       prespec={"final": ("lgbm_l63_m500", 0.487389)})
    assert any("excluded" in p for p in problems)


# ----------------------------------------------------------------------------- scoring context per row
def test_row_contexts_final_for_q_roles_and_fold_oof_for_train():
    role = np.array([0, 0, 3, 4, 5, 1, 2, 0], dtype=np.int8)
    fold = np.array([0, 4, 0, 1, 2, 0, 0, 2], dtype=np.int8)
    assert vs.row_contexts(role, fold).tolist() == [1, 5, 0, 0, 0, -1, -1, 3]
    with pytest.raises(ValueError):
        vs.row_contexts(np.array([0], dtype=np.int8), np.array([-1], dtype=np.int8))


def test_context_query_list_is_the_sorted_union_of_both_ends():
    codes = np.array([0, 0, 1, 0], dtype=np.int8)
    pre, post = np.array([10, 5, 7, 3]), np.array([12, 6, 9, 4])
    assert vs.context_query_list(codes, 0, pre, post).tolist() == [3, 4, 5, 6, 10, 12]
    assert vs.context_query_list(codes, 1, pre, post).tolist() == [7, 9]
    assert vs.context_query_list(codes, 2, pre, post).tolist() == []
    with pytest.raises(ValueError):
        vs.context_query_list(codes, 0, np.array([10, -1, 7, 3]), post)


# ----------------------------------------------------------------------------- OOF exclusion (assemble.py:109-114)
def test_oof_leaks_flags_seen_games_and_train_games_without_target_bit():
    membership = np.zeros((6, 5), dtype=np.uint8)
    membership[0, 1] = 8            # final saw match 1 in V_DEV
    membership[2, 2] = 32           # oof1: match 2 is its held-out target (clean)
    membership[2, 3] = 0            # oof1: match 3 lacks the target bit
    membership[2, 4] = 32 | 1       # oof1: match 4 is a target but also a fit game
    codes = np.array([0, 0, 2, 2, 2], dtype=np.int8)
    role = np.array([3, 5, 0, 0, 0], dtype=np.int8)
    match_index = np.arange(5)
    assert vs.oof_leaks(codes, role, match_index, membership).tolist() == [False, True, False, True, True]
    with pytest.raises(ValueError):
        vs.oof_leaks(np.array([-1], dtype=np.int8), np.array([1], dtype=np.int8), np.array([0]), membership)


# ----------------------------------------------------------------------------- q rows = Q_ROWS_h90 rows, aligned
def _eng():
    return {"canonical_match_key": np.array([b"m0", b"m0", b"m1", b"m2", b"m3"]), "component_key": np.array([b"c0", b"c1", b"c0", b"c0", b"c0"]),
            "match_index": np.array([0, 0, 1, 2, 3], dtype=np.int32), "role": np.array([0, 1, 3, 5, 4], dtype=np.int8),
            "fold": np.array([2, 0, 0, 0, 0], dtype=np.int8), "query_pre": np.array([1, 3, 5, 7, 9]),
            "query_h90": np.array([2, 4, 6, 8, 10]), "valid_h90": np.array([True, True, True, False, True])}


def test_align_q_rows_returns_the_engagement_rows_of_the_row_table():
    eng = _eng()
    keep = np.array([True, False, True, False, True])
    q_rows = {name: eng[name][keep] for name in ("canonical_match_key", "component_key", "match_index", "role", "fold", "query_pre")}
    q_rows["match_index"] = q_rows["match_index"].astype(np.int64)
    assert vs.align_q_rows(eng, q_rows).tolist() == keep.tolist()
    q_rows["canonical_match_key"] = q_rows["canonical_match_key"][::-1].copy()      # m0, m1, m3 -> m3, m1, m0
    with pytest.raises(ValueError):
        vs.align_q_rows(eng, q_rows)


# ----------------------------------------------------------------------------- V_DEV rows and the frozen ledger
def _cache_metadata(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    membership = rng.choice(np.array([1, 4, 8, 16], dtype=np.uint8), size=n)
    role = np.where(membership == 8, b"V_DEV", np.where(membership == 16, b"V_CAL", b"TRAIN")).astype("S8")
    return {"context_membership": membership, "role": role, "nominal_index": np.arange(n, dtype=np.int64) * 3,
            "query_index": np.arange(n, dtype=np.int64) * 2 + 1, "label_w": rng.random(n) < 0.5,
            "game_key": np.array([f"{i // 7:064x}".encode() for i in range(n)], dtype="S64"), "swapped": np.zeros(n, dtype=bool)}


def test_dev_rows_and_consumed_record_match_the_training_ledger_definition():
    meta = _cache_metadata()
    rows = vs.dev_rows(meta)
    dev = meta["context_membership"] == 8
    assert np.array_equal(rows["query_index"], meta["query_index"][dev]) and rows["label"].dtype == np.int8
    consumed = vs.dev_consumed(rows)
    keys = [str(int(v)) for v in meta["nominal_index"][dev]]
    games = [bytes(v).decode() for v in meta["game_key"][dev]]
    assert consumed["rows"] == consumed["key_count"] == int(dev.sum())
    assert consumed["batches"] == -(-int(dev.sum()) // 1024)
    assert consumed["target_sum"] == float(meta["label_w"][dev].sum()) and consumed["roles"] == ["V_DEV"]
    assert consumed["keys_sha256"] == hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    assert consumed["game_keys_sha256"] == hashlib.sha256(json.dumps(games, separators=(",", ":")).encode()).hexdigest()


def test_dev_rows_refuse_swapped_or_unsorted_rows():
    meta = _cache_metadata()
    meta["swapped"] = meta["swapped"].copy()
    meta["swapped"][np.flatnonzero(meta["context_membership"] == 8)[0]] = True
    with pytest.raises(ValueError):
        vs.dev_rows(meta)
    meta = _cache_metadata()
    meta["query_index"] = meta["query_index"][::-1].copy()
    with pytest.raises(ValueError):
        vs.dev_rows(meta)


def test_reproduces_compares_at_the_fourth_decimal():
    assert vs.reproduces(0.4873894817692967, 0.4873894817692967)
    assert vs.reproduces(0.48738, 0.48741)            # both 0.4874
    assert not vs.reproduces(0.48734, 0.48736)        # 0.4873 vs 0.4874
    assert not vs.reproduces(float("nan"), 0.4874)


def test_vdev_record_passes_only_when_loss_and_consumed_rows_agree():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.95, 300)
    label = (rng.random(300) < raw).astype(np.int8)
    loss = float(np.mean(-(label * np.log(raw) + (1 - label) * np.log1p(-raw))))
    consumed = {"rows": 300, "keys_sha256": "k", "game_keys_sha256": "g", "batches": 1, "key_count": 300, "roles": ["V_DEV"],
                "target_sum": float(label.sum())}
    result = {"dev_metrics": {"raw_logloss": loss, "consumed": dict(consumed)}}
    good = vs.vdev_record("final", "lgbm_l63_m500", raw, label, consumed, result, round(loss, 6))
    assert good["passed"] and good["abs_diff"] < 1e-12
    shifted = {"dev_metrics": {"raw_logloss": loss + 1e-3, "consumed": dict(consumed)}}
    assert not vs.vdev_record("final", "lgbm_l63_m500", raw, label, consumed, shifted, round(loss, 6))["passed"]
    other = {"dev_metrics": {"raw_logloss": loss, "consumed": {**consumed, "keys_sha256": "x"}}}
    assert not vs.vdev_record("final", "lgbm_l63_m500", raw, label, consumed, other, round(loss, 6))["passed"]


# ----------------------------------------------------------------------------- parts: written once, chunk-level restart
def test_part_status_and_merge_restart_from_complete_parts(tmp_path: Path):
    identity = {"kind": "q", "context": "oof1", "card_id": "lgbm_l63_m500", "model_sha256": "m" * 64}
    index = np.arange(10, dtype=np.int64) * 3
    ranges = vs.part_ranges(len(index), 4)
    assert ranges == [(0, 4), (4, 8), (8, 10)]
    assert vs.part_status(tmp_path, 0, index[0:4], identity) == "missing"
    vs.write_part(tmp_path, 0, index[0:4], np.full(4, 0.25), identity)
    vs.write_part(tmp_path, 1, index[4:8], np.full(4, 0.5), identity)
    assert vs.part_status(tmp_path, 0, index[0:4], identity) == "done"
    with pytest.raises(ValueError):                                            # incomplete coverage
        vs.merge_parts(tmp_path, ranges, index, identity)
    vs.part_path(tmp_path, 2).write_bytes(b"partial")                          # npz without its receipt: redo
    assert vs.part_status(tmp_path, 2, index[8:10], identity) == "incomplete"
    vs.write_part(tmp_path, 2, index[8:10], np.full(2, 0.75), identity)
    merged = vs.merge_parts(tmp_path, ranges, index, identity)
    assert merged.dtype == np.float64 and merged.tolist() == [0.25] * 4 + [0.5] * 4 + [0.75] * 2
    with pytest.raises(ValueError):                                            # another model's part is never reused
        vs.part_status(tmp_path, 1, index[4:8], {**identity, "model_sha256": "n" * 64})
    with pytest.raises(FileExistsError):                                       # a finished part is written once
        vs.write_part(tmp_path, 1, index[4:8], np.full(4, 0.5), identity)


def test_merged_files_are_written_once_or_verified_on_restart(tmp_path: Path):
    arrays = {"query_index": np.arange(5, dtype=np.int64), "raw_p": np.linspace(0.1, 0.9, 5)}
    sha = vs._write_npz_or_verify(tmp_path / "PRED_x.npz", arrays)
    assert vs._write_npz_or_verify(tmp_path / "PRED_x.npz", {k: v.copy() for k, v in arrays.items()}) == sha
    with pytest.raises(ValueError):
        vs._write_npz_or_verify(tmp_path / "PRED_x.npz", {**arrays, "raw_p": arrays["raw_p"][::-1].copy()})
    assert vs._write_array_once(tmp_path / "Q.npy", arrays["query_index"]) == vs._write_array_once(tmp_path / "Q.npy", arrays["query_index"])
    with pytest.raises(ValueError):
        vs._write_array_once(tmp_path / "Q.npy", arrays["query_index"] + 1)


def test_part_status_refuses_a_changed_part(tmp_path: Path):
    identity = {"kind": "vdev", "context": "final", "card_id": "lgbm_l63_m500", "model_sha256": "m" * 64}
    index = np.arange(4, dtype=np.int64)
    vs.write_part(tmp_path, 0, index, np.full(4, 0.25), identity)
    vs.part_path(tmp_path, 0).write_bytes(vs.part_path(tmp_path, 0).read_bytes() + b"\0")
    with pytest.raises(ValueError):
        vs.part_status(tmp_path, 0, index, identity)


# ----------------------------------------------------------------------------- score_context: canonical batches
class _Surface:
    def __init__(self, scramble=False):
        self.batches, self.scramble = [], scramble

    def iter_query_rows(self, batch):
        self.batches.append(len(batch))
        order = list(batch)[::-1] if self.scramble else list(batch)
        for q in order:
            yield int(q), {"q": int(q)}, {}

    def close(self):
        pass


class _Estimator:
    def predict_proba(self, matrix, num_threads=None):
        p = 1.0 / (1.0 + np.exp(-np.asarray(matrix, dtype=np.float64)[:, 0] / 1000.0))
        return np.column_stack([1.0 - p, p])


class _Model:
    family = "lightgbm"
    estimator = _Estimator()

    def predict_proba(self, inputs):
        return self.estimator.predict_proba(inputs["table_csr"])[:, 1]


def _transform(rows, _bundle):
    return SimpleNamespace(table_csr=np.array([[row["q"]] for row in rows], dtype=np.float32))


def test_score_context_scores_canonical_1024_row_batches_in_order():
    measurer = vs.Measurer(context="oof2", card_id="lgbm_l63_m500", model=_Model(), checkpoint=Path("m.pkl"), model_sha256="m" * 64)
    surface = _Surface()
    index = np.arange(2500, dtype=np.int64) * 2
    raw = vs.score_context("oof2", index, measurer=measurer, inputs=(surface, None), transform=_transform)
    assert surface.batches == [1024, 1024, 452]
    assert raw.dtype == np.float64 and raw.shape == (2500,)
    assert np.allclose(raw, 1.0 / (1.0 + np.exp(-index / 1000.0)))
    assert measurer.thread_check is True
    with pytest.raises(ValueError):
        vs.score_context("oof2", index[:10], measurer=measurer, inputs=(_Surface(scramble=True), None), transform=_transform)
    with pytest.raises(ValueError):
        vs.score_context("final", index[:10], measurer=measurer, inputs=(_Surface(), None), transform=_transform)
