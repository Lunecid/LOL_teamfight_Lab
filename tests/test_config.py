"""Tests for core/config.py -- CFG singleton and key attributes."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.config import cfg, CFG


class TestCfgImportable:
    """Test that cfg is importable and is a CFG instance."""

    def test_cfg_is_cfg_instance(self):
        assert isinstance(cfg, CFG)

    def test_cfg_is_not_none(self):
        assert cfg is not None


class TestKeyAttributes:
    """Test that key configuration attributes exist and have expected types."""

    def test_dropout_exists(self):
        assert hasattr(cfg, "DROPOUT")
        assert isinstance(cfg.DROPOUT, float)

    def test_rnn_hidden_exists(self):
        assert hasattr(cfg, "RNN_HIDDEN")
        assert isinstance(cfg.RNN_HIDDEN, int)

    def test_lr_exists(self):
        assert hasattr(cfg, "LR")
        assert isinstance(cfg.LR, float)
        assert cfg.LR > 0

    def test_seeds_exists(self):
        assert hasattr(cfg, "SEEDS")
        assert isinstance(cfg.SEEDS, tuple)
        assert len(cfg.SEEDS) >= 1

    def test_seeds_has_five(self):
        """SEEDS expanded to 5 seeds for statistical significance."""
        assert len(cfg.SEEDS) == 5

    def test_epochs_exists(self):
        assert hasattr(cfg, "EPOCHS")
        assert isinstance(cfg.EPOCHS, int)
        assert cfg.EPOCHS > 0

    def test_patience_exists(self):
        assert hasattr(cfg, "PATIENCE")
        assert isinstance(cfg.PATIENCE, int)
        assert cfg.PATIENCE > 0

    def test_batch_size_exists(self):
        assert hasattr(cfg, "BATCH_SIZE")
        assert isinstance(cfg.BATCH_SIZE, int)
        assert cfg.BATCH_SIZE > 0

    def test_split_mode_exists(self):
        assert hasattr(cfg, "SPLIT_MODE")
        assert isinstance(cfg.SPLIT_MODE, str)

    def test_gnn_dim_exists(self):
        assert hasattr(cfg, "GNN_DIM")
        assert isinstance(cfg.GNN_DIM, int)

    def test_gnn_dropout_exists(self):
        assert hasattr(cfg, "GNN_DROPOUT")
        assert isinstance(cfg.GNN_DROPOUT, float)

    def test_model_list_populated(self):
        assert hasattr(cfg, "MODEL_LIST")
        assert isinstance(cfg.MODEL_LIST, tuple)
        assert len(cfg.MODEL_LIST) > 0

    def test_ablation_groups_populated(self):
        assert hasattr(cfg, "ABLATION_GROUPS")
        assert isinstance(cfg.ABLATION_GROUPS, dict)
        assert len(cfg.ABLATION_GROUPS) > 0


class TestAdaptiveSigma:
    """Test the ADJ_SIGMA_ADAPTIVE newly changed default."""

    def test_adj_sigma_adaptive_is_true(self):
        assert hasattr(cfg, "ADJ_SIGMA_ADAPTIVE")
        assert cfg.ADJ_SIGMA_ADAPTIVE is True

    def test_adj_sigma_ratio_exists(self):
        assert hasattr(cfg, "ADJ_SIGMA_RATIO")
        assert isinstance(cfg.ADJ_SIGMA_RATIO, float)
        assert cfg.ADJ_SIGMA_RATIO > 0


class TestNewFeatureFlags:
    """Test newly added feature flags exist."""

    def test_recency_weight_enabled(self):
        assert hasattr(cfg, "RECENCY_WEIGHT_ENABLED")
        assert isinstance(cfg.RECENCY_WEIGHT_ENABLED, bool)

    def test_hybrid_h0_enabled(self):
        assert hasattr(cfg, "HYBRID_H0_ENABLED")
        assert isinstance(cfg.HYBRID_H0_ENABLED, bool)

    def test_temp_scaling_enabled(self):
        assert hasattr(cfg, "TEMP_SCALING_ENABLED")
        assert isinstance(cfg.TEMP_SCALING_ENABLED, bool)
