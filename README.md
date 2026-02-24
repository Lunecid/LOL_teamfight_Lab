# LOL Teamfight Lab

A machine learning pipeline for predicting **League of Legends teamfight outcomes** using match timeline data from the Riot API. The project combines tabular ML (LightGBM), deep sequential models (RNN, Transformer, TCN, Mamba), and graph neural networks (GNN, GAT, ST-GNN) with ensemble fusion to analyze spatial-temporal patterns in competitive gameplay.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Models](#models)
- [Testing](#testing)
- [Key Features](#key-features)

---

## Overview

LOL Teamfight Lab processes League of Legends match detail and timeline JSON data to:

1. **Detect teamfights** from event bursts (kills, spells, objectives) and player position clustering
2. **Extract temporal features** including player positions, stats, item builds, cooldown states, and event sequences
3. **Train multiple model architectures** to predict which team wins each fight
4. **Ensemble predictions** through stacking and meta-learning for final output

The pipeline supports systematic ablation studies, multi-seed bootstrapping for confidence intervals, and patch-aware data splitting to handle covariate shift across game patches.

---

## Architecture

```
Match JSONs (detail + timeline)
        |
        v
  data.cache_io.prebuild_cache()
        |
        v
  Preprocessed Match Cache (NPZ + JSON meta)
        |
        v
  gameplay.fights.detect_fights()
        |
        v
  core.fight_types.FightRef (fight identifiers with timestamps)
        |
        v
  data.index_split.build_fight_index() -> split_refs()
        |
        v
  Train / Val / Test splits (match-grouped, patch-stratified)
        |
        v
  data.dataset.InMemoryFightDataset
        |--- gameplay.pipeline.build_ms_sequence()       [temporal features]
        |--- gameplay.features.build_sequence_features() [normalization]
        |--- collate_batch()                             [graph pooling]
        |
        v
  Model Training
        |--- train.baseline: LightGBM (tabular)
        |--- train.deep: RNN / Transformer / TCN / Mamba (sequential)
        |--- train.deep: GNN / GAT / MPNN / ST-GNN (graph)
        |
        v
  Ensemble Fusion (train.fusion)
        |--- Simple stacking
        |--- Out-of-fold stacking
        |--- Factorial stacking + meta-learner
        |
        v
  Reports (AUC, AP, minutewise, situation-aware metrics)
```

---

## Project Structure

```
LOL_teamfight_Lab/
|
|-- pyproject.toml                  # Build & package configuration
|-- requirements.txt                # Pinned dependency list
|-- __init__.py                     # Package metadata (v0.1.0)
|-- main.py                         # Wrapper calling runner.main()
|-- runner.py                       # Argument parser & main entry point
|-- experiment_runner.py            # Systematic ablation study runner
|
|-- core/                           # Configuration, contracts & utilities
|   |-- config.py                   # Central CFG dataclass (single source of truth)
|   |-- config_legacy.py            # Deprecated config (reference only)
|   |-- contract.py                 # Contract validation imports
|   |-- feature_contract.py         # Feature contract definitions
|   |-- time_contract.py            # Time/index contract for legacy compat
|   |-- fight_types.py              # FightRef dataclass
|   |-- roles.py                    # Lane/role assignment
|   |-- diagnostics.py              # Fight detection diagnostics
|   |-- improvements.py             # Domain knowledge enhancements
|   |-- timeutils.py                # Time-based calculations
|   |-- common.py                   # Shared math utilities
|   |-- common_torch.py             # PyTorch-specific utilities
|   |-- utils.py                    # General utilities (metrics, seeding)
|
|-- data/                           # Data loading, caching & splitting
|   |-- cache_io.py                 # Match cache I/O (NPZ + JSON)
|   |-- ram_cache.py                # In-RAM LRU cache for loaded matches
|   |-- file_io.py                  # File utilities (CSV, JSON writes)
|   |-- dataset.py                  # InMemoryFightDataset (PyTorch)
|   |-- index_split.py              # Data splitting strategies
|   |-- indexing.py                 # Match indexing & leakage checks
|   |-- events_index.py             # Event timestamp indexing
|   |-- labels.py                   # Ground-truth label generation
|   |-- logits.py                   # Model prediction management
|
|-- gameplay/                       # Fight detection & feature engineering
|   |-- fights.py                   # Fight detection engine
|   |-- pipeline.py                 # Core temporal feature building
|   |-- features.py                 # Feature builders & normalizers
|
|-- train/                          # Model definitions & training
|   |-- models.py                   # Model factory & architecture definitions
|   |-- deep.py                     # Deep learning training harness
|   |-- baseline.py                 # LightGBM tabular baseline
|   |-- fusion.py                   # Model stacking & fusion
|   |-- speed_config.py             # Speed/performance profiles
|   |-- speed.py                    # Performance profiling
|
|-- app/                            # Orchestration & analysis
|   |-- experiment.py               # Main training loop orchestrator
|   |-- analysis.py                 # Fight analysis utilities
|
|-- tests/                          # Unit tests
|   |-- test_common.py              # Tests for math utilities
|   |-- test_config.py              # Tests for CFG dataclass
|   |-- test_experiment_runner.py   # Tests for ablation runner
|   |-- test_fight_types.py         # Tests for FightRef
|   |-- test_index_split.py         # Tests for data splitting
|   |-- test_utils.py               # Tests for general utilities
```

---

## Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch (CUDA, AMP mixed precision, torch.compile) |
| **Gradient Boosting** | LightGBM |
| **Numerical** | NumPy |
| **ML Utilities** | scikit-learn |
| **Hardware** | CUDA GPU support, TF32 on Ampere GPUs, bf16/fp16 mixed precision |

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- League of Legends match data in JSON format (detail + timeline files)

### Install Dependencies

```bash
# Editable install (recommended for development)
pip install -e ".[all]"

# Or use requirements.txt
pip install -r requirements.txt

# PyTorch with CUDA support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configure Data Paths

Set environment variables or edit `core/config.py`:

```bash
export LOL_DETAIL_DIR="/path/to/match/details"
export LOL_TIMELINE_DIR="/path/to/match/timelines"
export LOL_OUTPUT_ROOT="/path/for/outputs"
```

---

## Usage

### Full Pipeline

```bash
# Run everything: cache -> train -> report
python main.py --mode all --seed 42
```

### Individual Steps

```bash
# Cache building only
python main.py --mode build_cache

# Training only (assumes cache exists)
python main.py --mode train --models rnn_bigru,gnn_graphsage

# Specific models
python main.py --models lgbm,rnn_transformer,fusion_auto_best --seed 7
```

### Ablation Studies

```bash
# Phase 1: Baseline reproduction
python experiment_runner.py --phase 1

# Phase 2: Single-factor treatments
python experiment_runner.py --phase 2 --treatment all
```

### Quick Test Run

```bash
python main.py --mode all --max_matches 10 --seed 7
```

---

## Configuration

All hyperparameters live in `core/config.py` as a single `CFG` dataclass:

| Section | What It Controls |
|---------|-----------------|
| **Data paths** | Detail/timeline/output directories |
| **Cache version** | Feature version for cache invalidation |
| **Fight detection** | Clustering algorithm, engagement validation, merge thresholds |
| **Features** | Node/event/global feature names, normalization |
| **Model list** | Active models, aliases, ablation groups |
| **Training** | Batch size, learning rate, epochs, patience, dropout, hidden dims |
| **Data splitting** | Val/test fractions, seeds for bootstrap CI |

---

## Models

### Baseline
- **LightGBM** -- Gradient boosting on tabular features (champion stats, objective control, gold differentials)

### Deep Sequential
- **BiGRU** -- Bidirectional GRU capturing temporal fight dynamics
- **Transformer** -- Self-attention over temporal sequences
- **TCN** -- Temporal convolutional network
- **Mamba** -- State-space model for long sequences

### Graph Neural Networks
- **GraphSage** -- Neighborhood aggregation on player interaction graphs
- **GAT** -- Graph attention network with multi-head attention
- **MPNN** -- Message passing neural network
- **ST-GNN** -- Spatial-temporal GNN combining position and time

### Ensemble
- **Simple stacking** -- Weighted average of model logits
- **Out-of-fold stacking** -- K-fold meta-learner training
- **Factorial stacking** -- All model subset combinations with meta-learner selection

---

## Testing

The project includes a unit test suite under `tests/` using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run a specific test file
pytest tests/test_utils.py
```

Test coverage includes:
- **core utilities** -- safe numeric parsing, sigmoid/logit, sign-preserving log
- **CFG dataclass** -- default values, required fields, model list population
- **FightRef** -- construction, key stability/uniqueness, label boundary validation
- **data splitting** -- match-grouped splits, patch stratification, leakage checks
- **experiment runner** -- bootstrap CI calculation, seed determinism
- **metrics** -- AUC, AP, Brier score, recall computation

---

## Key Features

- **Event-driven fight detection** -- Triggers from event bursts rather than position clustering alone; merges continuous fights within 30s / 2000 units
- **Prediction gap** -- Configurable millisecond offset to predict before engagement starts
- **Focal loss** -- Hard-negative mining for imbalanced fight outcomes
- **Game phase awareness** -- Early/mid/late game phase embeddings
- **Role-aware adjacency** -- `A_role(i,j) = A_dist(i,j) * R[role(i), role(j)]` for GNN edges
- **Multi-task learning** -- Joint prediction of fight outcome, gold swing, and kill count
- **Recency weighting** -- `w_i = exp((patch_i - patch_min) / tau)` to handle patch covariate shift
- **Temperature scaling** -- Post-hoc calibration of prediction confidence
- **Hybrid h0 conditioning** -- Projects tabular features into RNN initial hidden state
- **Mixed precision training** -- AMP with bf16/fp16 and optional torch.compile kernel fusion
- **Match-grouped splitting** -- Prevents data leakage across train/val/test sets
- **Multi-seed bootstrap** -- 5-seed runs for confidence interval reporting

---

## Code Statistics

| Metric | Value |
|--------|-------|
| Python files | 49 |
| Lines of code | ~24,400 |
| Test cases | 100+ |
| Model architectures | 15+ variants |
| Configurable hyperparameters | 150+ |
| Feature dimensions | 100+ across node/event/global |

---

## License

This project is for research and educational purposes.
