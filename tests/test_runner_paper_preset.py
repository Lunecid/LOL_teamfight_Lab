"""Tests for runner.py paper preset parser behavior."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runner import _apply_paper_preset, build_argparser


def test_paper_preset_core4_1seed_applies_expected_models_and_seed():
    ap = build_argparser()
    args = ap.parse_args(["--paper_preset", "core4_1seed", "--seed", "42"])
    _apply_paper_preset(args)

    assert int(args.seed) == 7
    assert str(args.models) != ""
    assert "rnn_bigru" in str(args.models)
    assert "gnn_graphsage" in str(args.models)
    assert "rnn_transformer" in str(args.models)
    assert "layered_fusion@" in str(args.models)
    assert bool(args.no_factorial_fusion) is True
    assert str(args.speed_profile).lower() in ("rtx5080", "rtx50", "aggressive", "auto")


def test_paper_preset_fast_sets_default_max_matches_when_unset():
    ap = build_argparser()
    args = ap.parse_args(["--paper_preset", "core4_1seed_fast"])
    _apply_paper_preset(args)

    assert int(args.max_matches) == 600


def test_paper_preset_fast_respects_explicit_paper_max_matches():
    ap = build_argparser()
    args = ap.parse_args(
        ["--paper_preset", "core4_1seed_fast", "--paper_max_matches", "321", "--max_matches", "9999"]
    )
    _apply_paper_preset(args)

    assert int(args.max_matches) == 321


def test_paper_preset_optimal_uses_event_xattn_stack():
    ap = build_argparser()
    args = ap.parse_args(["--paper_preset", "core4_optimal"])
    _apply_paper_preset(args)

    models = str(args.models)
    assert "rnn_bigru" in models
    assert "gnn_graphsage" in models
    assert "event_xattn" in models
    assert "layered_fusion@" in models
    assert "event=xattn" in models
    assert "rnn_transformer" not in models


def test_split_mode_alias_holdout_patch_is_accepted():
    ap = build_argparser()
    args = ap.parse_args(["--split_mode", "holdout_patch"])
    assert str(args.split_mode) == "holdout_patch"
