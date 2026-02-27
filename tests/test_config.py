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


class TestP01DualPathXY:
    """[P0-1] Dual-path XY coordinate handling config."""

    def test_zero_xy_node_features_false(self):
        assert cfg.ZERO_XY_NODE_FEATURES is False

    def test_zero_xy_in_extra_seq_true(self):
        assert hasattr(cfg, "ZERO_XY_IN_EXTRA_SEQ")
        assert cfg.ZERO_XY_IN_EXTRA_SEQ is True

    def test_use_relative_xy_true(self):
        assert hasattr(cfg, "USE_RELATIVE_XY")
        assert cfg.USE_RELATIVE_XY is True

    def test_adj_sigma_factor(self):
        assert hasattr(cfg, "ADJ_SIGMA_FACTOR")
        assert isinstance(cfg.ADJ_SIGMA_FACTOR, float)
        assert cfg.ADJ_SIGMA_FACTOR >= 1.0


class TestP06ValTestSubsampling:
    """[P0-6] Val/Test subsampling config."""

    def test_val_max_n_exists(self):
        assert hasattr(cfg, "VAL_MAX_N")
        assert isinstance(cfg.VAL_MAX_N, int)
        assert cfg.VAL_MAX_N > 0

    def test_test_max_n_exists(self):
        assert hasattr(cfg, "TEST_MAX_N")
        assert isinstance(cfg.TEST_MAX_N, int)
        assert cfg.TEST_MAX_N > 0


class TestP03InputProjection:
    """[P0-3] Input projection config."""

    def test_use_input_projection_exists(self):
        assert hasattr(cfg, "USE_INPUT_PROJECTION")
        assert isinstance(cfg.USE_INPUT_PROJECTION, bool)

    def test_input_proj_dim_exists(self):
        assert hasattr(cfg, "INPUT_PROJ_DIM")
        assert isinstance(cfg.INPUT_PROJ_DIM, int)
        assert cfg.INPUT_PROJ_DIM > 0


class TestP22NodeCatSpecs:
    """[P2-2] Categorical embedding specs config."""

    def test_node_cat_specs_exists(self):
        assert hasattr(cfg, "NODE_CAT_SPECS")
        assert isinstance(cfg.NODE_CAT_SPECS, dict)

    def test_node_cat_specs_has_champion_id(self):
        assert "champion_id" in cfg.NODE_CAT_SPECS
        spec = cfg.NODE_CAT_SPECS["champion_id"]
        assert "num_embeddings" in spec
        assert "emb_dim" in spec
        assert spec["num_embeddings"] > 0
        assert spec["emb_dim"] > 0


class TestP17WarmupEpochs:
    """[P1-7] Warmup epochs config."""

    def test_warmup_epochs_exists(self):
        assert hasattr(cfg, "WARMUP_EPOCHS")
        assert isinstance(cfg.WARMUP_EPOCHS, int)
        assert cfg.WARMUP_EPOCHS >= 1
