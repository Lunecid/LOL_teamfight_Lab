"""Regression tests for q-pipeline provenance / reproducibility fixes.

Covers the review's section-6 items:
  6.1 OOF evaluator hash is computed on the *final* file (after n_eng_labeled).
  6.2 cohort S fails fast without --reuse-evaluators (shared T evaluator dir).
  6.3 external data root is resolvable via CLI value / env var (no silent
      dependence on a private ~/Documents working tree).
  6.4 label<->engagement join fails on missing / duplicate keys instead of
      silently dropping rows, and records exclusion reasons separately.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts import rr20260920_q_train_oof_mlp_folds as oof
from scripts import rr20260920_review_response_rr12 as rr12


# ---------------------------------------------------------------- 6.3 data root
def test_resolve_data_root_cli_value_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("LOL_TEAMFIGHT_DATA_ROOT", str(tmp_path / "from_env"))
    got = oof.resolve_data_root(str(tmp_path / "from_cli"))
    assert got == (tmp_path / "from_cli")


def test_resolve_data_root_env_var_used_when_no_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("LOL_TEAMFIGHT_DATA_ROOT", str(tmp_path / "from_env"))
    assert oof.resolve_data_root(None) == (tmp_path / "from_env")
    # rr12 shares the same contract
    assert rr12.resolve_data_root(None) == (tmp_path / "from_env")


def test_resolve_data_root_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("LOL_TEAMFIGHT_DATA_ROOT", raising=False)
    # falls back to the legacy local default (path object, not None)
    assert oof.resolve_data_root(None).name == "LOL_Teamfight"


# ------------------------------------------------------------ 6.2 cohort guard
def test_validate_cohort_reuse_s_without_reuse_fails():
    with pytest.raises(SystemExit):
        oof.validate_cohort_reuse("S", reuse_evaluators=False)


def test_validate_cohort_reuse_s_with_reuse_ok():
    assert oof.validate_cohort_reuse("S", reuse_evaluators=True) is None


def test_validate_cohort_reuse_t_ok_either_way():
    assert oof.validate_cohort_reuse("T", reuse_evaluators=False) is None
    assert oof.validate_cohort_reuse("T", reuse_evaluators=True) is None


def test_oof_write_allowed_only_for_t_fit():
    assert oof.assert_oof_evaluator_write_allowed("T", reuse_evaluators=False) is None
    with pytest.raises(SystemExit):
        oof.assert_oof_evaluator_write_allowed("T", reuse_evaluators=True)
    with pytest.raises(SystemExit):
        oof.assert_oof_evaluator_write_allowed("S", reuse_evaluators=False)
    with pytest.raises(SystemExit):
        oof.assert_oof_evaluator_write_allowed("S", reuse_evaluators=True)


# ------------------------------------------------------ 6.1 finalize + hash
def test_finalize_fold_evaluator_hash_matches_final_file(tmp_path):
    import joblib

    fold_path = tmp_path / "V_oof_fold0_mlp_expanded.joblib"
    # mimic the initial dump written before scoring (n_eng_labeled placeholder)
    joblib.dump({"kind": "mlp_embedding_oof", "fold": 0, "n_eng_labeled": -1}, fold_path)
    hash_before = hashlib.sha256(fold_path.read_bytes()).hexdigest().upper()

    recorded = oof.finalize_fold_evaluator(fold_path, 12345)

    # n_eng_labeled was actually updated on disk
    obj = joblib.load(fold_path)
    assert obj["n_eng_labeled"] == 12345
    # the returned hash matches the FINAL bytes on disk (the whole point of 6.1)
    actual = hashlib.sha256(fold_path.read_bytes()).hexdigest().upper()
    assert recorded == actual
    # and it differs from the pre-finalize hash (file really changed)
    assert recorded != hash_before


# ------------------------------------------------------------- 6.4 join guard
def test_join_happy_path_preserves_label_order():
    lab_match = np.array(["A", "B", "C"])
    lab_s = np.array([10, 20, 30])
    eng_match = np.array(["C", "A", "B"])  # different order
    eng_s = np.array([30, 10, 20])
    idx_l, idx_e, report = rr12.join_label_engagement_indices(lab_match, lab_s, eng_match, eng_s)
    assert list(idx_l) == [0, 1, 2]
    # engagement indices point back to the matching (match, s)
    assert list(eng_match[idx_e]) == ["A", "B", "C"]
    assert report["n_missing"] == 0
    assert report["n_joined"] == 3


def test_join_missing_key_raises_by_default():
    lab_match = np.array(["A", "Z"])  # Z absent from engagements
    lab_s = np.array([1, 2])
    eng_match = np.array(["A"])
    eng_s = np.array([1])
    with pytest.raises(SystemExit):
        rr12.join_label_engagement_indices(lab_match, lab_s, eng_match, eng_s)


def test_join_missing_key_allowed_records_exclusion():
    lab_match = np.array(["A", "Z"])
    lab_s = np.array([1, 2])
    eng_match = np.array(["A"])
    eng_s = np.array([1])
    idx_l, idx_e, report = rr12.join_label_engagement_indices(
        lab_match, lab_s, eng_match, eng_s, allow_missing=True
    )
    assert list(idx_l) == [0]
    assert report["n_missing"] == 1
    assert report["missing_examples"] == [["Z", 2]]


def test_join_duplicate_engagement_keys_always_raises():
    lab_match = np.array(["A"])
    lab_s = np.array([1])
    eng_match = np.array(["A", "A"])  # duplicate (A, 1)
    eng_s = np.array([1, 1])
    with pytest.raises(SystemExit):
        rr12.join_label_engagement_indices(
            lab_match, lab_s, eng_match, eng_s, allow_missing=True
        )
